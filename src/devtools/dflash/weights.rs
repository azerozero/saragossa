//! Lecture à la demande : seuls le header et un bloc de lignes sont temporaires.

use super::{expect_shape, DFlashQuant};
use crate::{
    safetensor::bytes_to_dense_f32, AffineQuantizedTensor, ForwardRuntime, InferError, Linear,
    Result, Tensor,
};
use safetensors::{tensor::TensorInfo, SafeTensors};
use std::{
    collections::HashMap,
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

pub(super) struct DraftWeights<'a> {
    file: File,
    path: PathBuf,
    data_start: usize,
    tensors: HashMap<String, TensorInfo>,
    runtime: ForwardRuntime<'a>,
}

impl<'a> DraftWeights<'a> {
    pub(super) fn open(path: &Path, runtime: ForwardRuntime<'a>) -> Result<Self> {
        let io = |source| InferError::Io {
            path: path.to_path_buf(),
            source,
        };
        let mut file = File::open(path).map_err(io)?;
        let len = usize::try_from(file.metadata().map_err(io)?.len())
            .map_err(|_| InferError::Shape("fichier DFlash trop grand".to_string()))?;
        let mut len_bytes = [0; 8];
        file.read_exact(&mut len_bytes).map_err(io)?;
        let data_start = usize::try_from(u64::from_le_bytes(len_bytes))
            .ok()
            .and_then(|header| header.checked_add(8))
            .filter(|&start| start <= len)
            .ok_or_else(|| InferError::Config("header DFlash invalide".to_string()))?;
        // Comme LazySidecar MTP : SafeTensors exige la taille du fichier pour
        // valider les offsets. Les pages zéro des poids ne sont jamais touchées.
        let mut bytes = vec![0_u8; len];
        bytes[..8].copy_from_slice(&len_bytes);
        file.read_exact(&mut bytes[8..data_start]).map_err(io)?;
        let (_, metadata) =
            SafeTensors::read_metadata(&bytes).map_err(|source| InferError::Safetensors {
                path: path.to_path_buf(),
                source,
            })?;
        let tensors = metadata
            .tensors()
            .into_iter()
            .map(|(key, info)| (key, info.clone()))
            .collect();
        Ok(Self {
            file,
            path: path.to_path_buf(),
            data_start,
            tensors,
            runtime,
        })
    }

    fn info(&self, key: &str) -> Result<&TensorInfo> {
        self.tensors
            .get(key)
            .ok_or_else(|| InferError::MissingWeight(format!("dflash {key}")))
    }

    fn read(&self, offset: usize, len: usize) -> Result<Vec<u8>> {
        let io = |source| InferError::Io {
            path: self.path.clone(),
            source,
        };
        let mut bytes = vec![0; len];
        let mut file = &self.file;
        file.seek(SeekFrom::Start(offset as u64)).map_err(io)?;
        file.read_exact(&mut bytes).map_err(io)?;
        Ok(bytes)
    }

    pub(super) fn tensor(&self, key: &str) -> Result<Tensor> {
        let info = self.info(key)?;
        let (start, end) = info.data_offsets;
        let bytes = self.read(self.data_start + start, end - start)?;
        Tensor::from_vec(
            info.shape.clone(),
            bytes_to_dense_f32(&bytes, info.dtype, key)?,
        )
    }

    pub(super) fn codebook(&self, key: &str) -> Result<super::DFlashCodebook> {
        self.codebook_with_storage(
            key,
            crate::runtime_flags::env_flag("RETI_RUST_SINGLE_COPY_WEIGHTS", true),
        )
    }

    fn codebook_with_storage(&self, key: &str, compact: bool) -> Result<super::DFlashCodebook> {
        if !compact {
            return self.tensor(key).map(Into::into);
        }
        let info = self.info(key)?;
        let [rows, cols] = info.shape.as_slice() else {
            return Err(InferError::Dimension(format!(
                "codebook DFlash {key} attendu rang 2"
            )));
        };
        let (start, end) = info.data_offsets;
        let mut values = Vec::with_capacity(rows * cols);
        // Blocs alignés sur tous les dtypes denses : aucun codebook f32 complet
        // n'est matérialisé si les valeurs sont exactement représentables.
        for offset in (start..end).step_by(16 * 1024) {
            let bytes = self.read(self.data_start + offset, (end - offset).min(16 * 1024))?;
            for value in bytes_to_dense_f32(&bytes, info.dtype, key)? {
                if value.to_bits() & 0xffff != 0 {
                    // Même un bit perdu (y compris NaN et zéro signé) interdit
                    // la conversion : les scores du sélecteur doivent rester exacts.
                    drop(values);
                    return self.tensor(key).map(Into::into);
                }
                values.push((value.to_bits() >> 16) as u16);
            }
        }
        Ok(super::DFlashCodebook::bf16(info.shape.clone(), values))
    }

    pub(super) fn linear(
        &self,
        prefix: &str,
        rows: usize,
        cols: usize,
        quant: Option<DFlashQuant>,
    ) -> Result<Linear> {
        let key = format!("{prefix}.weight");
        let info = self.info(&key)?;
        expect_shape(&info.shape, &[rows, cols], &key)?;
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if let (Some(quant), Some(metal)) = (quant, self.runtime.metal_executor()) {
            if crate::runtime_flags::single_copy_weights_enabled() {
                let (packed_cols, groups) =
                    crate::quantization::affine_layout(rows, cols, quant.group_size, quant.bits)?;
                let (packed, scales, biases) = metal.weight_buffers_affine_with(
                    rows * packed_cols,
                    rows * groups,
                    |packed, scales, biases| {
                        self.quantize_into(&key, quant, packed, scales, biases)
                    },
                )?;
                return Linear::new_quantized(
                    AffineQuantizedTensor::new_metal_shared_bf16(
                        &[rows, packed_cols],
                        packed,
                        0,
                        rows * packed_cols,
                        &[rows, groups],
                        scales,
                        0,
                        rows * groups,
                        &[rows, groups],
                        biases,
                        0,
                        rows * groups,
                        quant.group_size,
                        quant.bits,
                    )?,
                    None,
                );
            }
        }
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        let _ = self.runtime;
        let tensor = self.tensor(&key)?;
        match quant {
            Some(DFlashQuant { group_size, bits }) => Linear::new_quantized(
                AffineQuantizedTensor::quantize(&tensor, group_size, bits)?,
                None,
            ),
            None => Linear::new(tensor, None),
        }
    }

    /// Même cœur affine que le chargeur CPU, par blocs de lignes : les codes
    /// sont calculés AVANT l'arrondi bf16, comme dans l'ancien upload Metal.
    #[cfg(any(test, all(target_os = "macos", feature = "metal")))]
    fn quantize_into(
        &self,
        key: &str,
        quant: DFlashQuant,
        packed: &mut [u32],
        scales: &mut [u16],
        biases: &mut [u16],
    ) -> Result<()> {
        let info = self.info(key)?;
        let [rows, cols] = info.shape.as_slice() else {
            return Err(InferError::Dimension(format!(
                "DFlash {key} attendu rang 2"
            )));
        };
        let (packed_cols, groups) =
            crate::quantization::affine_layout(*rows, *cols, quant.group_size, quant.bits)?;
        if *rows == 0
            || *cols == 0
            || packed.len() != rows * packed_cols
            || scales.len() != rows * groups
            || biases.len() != scales.len()
        {
            return Err(InferError::Dimension(format!(
                "DFlash {key} buffers affines invalides"
            )));
        }
        let row_bytes = cols * (info.dtype.bitsize() / 8);
        const BLOCK_ROWS: usize = 16;
        for row in (0..*rows).step_by(BLOCK_ROWS) {
            let count = BLOCK_ROWS.min(rows - row);
            let bytes = self.read(
                self.data_start + info.data_offsets.0 + row * row_bytes,
                count * row_bytes,
            )?;
            let data = bytes_to_dense_f32(&bytes, info.dtype, key)?;
            let mut scales_f32 = vec![0.0; count * groups];
            let mut biases_f32 = vec![0.0; count * groups];
            crate::quantization::quantize_affine_into(
                &data,
                *cols,
                quant.group_size,
                quant.bits,
                &mut packed[row * packed_cols..(row + count) * packed_cols],
                &mut scales_f32,
                &mut biases_f32,
            )?;
            for (out, value) in scales[row * groups..(row + count) * groups]
                .iter_mut()
                .zip(scales_f32)
            {
                *out = crate::quantization::bf16_bits(value);
            }
            for (out, value) in biases[row * groups..(row + count) * groups]
                .iter_mut()
                .zip(biases_f32)
            {
                *out = crate::quantization::bf16_bits(value);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::{serialize, tensor::TensorView, Dtype};

    const ROWS: usize = 33;
    const COLS: usize = 128;

    fn fixture(dtype: Dtype) -> tempfile::NamedTempFile {
        // Trois blocs dont un incomplet, deux groupes par ligne, groupes constants
        // et valeurs négatives : détecte les erreurs d'offset et d'arrondi des codes.
        let bytes: Vec<u8> = (0..ROWS * COLS)
            .flat_map(|i| {
                let value = if i / 64 % 3 == 0 {
                    -0.25
                } else {
                    ((i * 73 % 257) as f32 - 128.0) / 64.0
                };
                match dtype {
                    Dtype::F32 => value.to_le_bytes().to_vec(),
                    Dtype::BF16 => crate::quantization::bf16_bits(value).to_le_bytes().to_vec(),
                    Dtype::F16 => [0x0000_u16, 0x8000, 0x3c00, 0xbc00, 0x0400, 0x3555][i % 6]
                        .to_le_bytes()
                        .to_vec(),
                    _ => unreachable!("invariant: dtype de fixture"),
                }
            })
            .collect();
        let view =
            TensorView::new(dtype, vec![ROWS, COLS], &bytes).expect("invariant: fixture valide");
        // Un autre tenseur force un offset non nul pour la projection.
        let prefix =
            TensorView::new(Dtype::F32, vec![1], &[0; 4]).expect("invariant: préfixe valide");
        let serialized = serialize([("a", prefix), ("fc.weight", view)], None)
            .expect("invariant: sérialisation");
        let mut file = tempfile::NamedTempFile::new().expect("invariant: fichier temporaire");
        std::io::Write::write_all(&mut file, &serialized).expect("invariant: écriture fixture");
        file
    }

    fn legacy(path: &Path, bits: usize) -> Result<AffineQuantizedTensor> {
        // Oracle de l'ancien chargeur : lecture complète puis conversion f32.
        let tensor = crate::safetensor::load_float_tensor(path, "fc.weight")?;
        AffineQuantizedTensor::quantize(&tensor, 64, bits)
    }

    #[test]
    fn streamed_affine_bytes_match_legacy_loader() -> Result<()> {
        for dtype in [Dtype::F32, Dtype::BF16, Dtype::F16] {
            let file = fixture(dtype);
            let source = DraftWeights::open(file.path(), ForwardRuntime::cpu())?;
            for bits in [4, 8] {
                let reference = legacy(file.path(), bits)?;
                let mut packed = vec![u32::MAX; reference.packed_len()];
                let mut scales = vec![0; reference.scales_len()];
                let mut biases = vec![0; reference.biases_len()];
                source.quantize_into(
                    "fc.weight",
                    DFlashQuant {
                        group_size: 64,
                        bits,
                    },
                    &mut packed,
                    &mut scales,
                    &mut biases,
                )?;
                assert_eq!(
                    bytemuck::cast_slice::<_, u8>(&packed),
                    bytemuck::cast_slice::<_, u8>(reference.packed_data())
                );
                let bf16 = |data: &[f32]| {
                    data.iter()
                        .copied()
                        .map(crate::quantization::bf16_bits)
                        .collect::<Vec<_>>()
                };
                assert_eq!(scales, bf16(&reference.scales_f32()));
                assert_eq!(biases, bf16(&reference.biases_f32()));
                let cpu = source.linear(
                    "fc",
                    ROWS,
                    COLS,
                    Some(DFlashQuant {
                        group_size: 64,
                        bits,
                    }),
                )?;
                let crate::LinearWeight::AffineQuantized(cpu) = cpu.weight() else {
                    unreachable!("invariant: affine")
                };
                assert_eq!(cpu.packed_data(), reference.packed_data());
                assert_eq!(cpu.scales_f32(), reference.scales_f32());
                assert_eq!(cpu.biases_f32(), reference.biases_f32());
            }
        }
        Ok(())
    }

    #[test]
    fn codebook_bytes_are_lossless_or_remain_f32() -> Result<()> {
        for (dtype, inexact) in [
            (Dtype::BF16, false),
            (Dtype::F32, false),
            (Dtype::F32, true),
        ] {
            // Tous les motifs bf16, y compris ±0, subnormaux, infinis et NaN.
            let bytes: Vec<u8> = (0..=u16::MAX)
                .flat_map(|bits| {
                    if dtype == Dtype::BF16 {
                        bits.to_le_bytes().to_vec()
                    } else {
                        let low = u32::from(inexact && bits == u16::MAX);
                        ((u32::from(bits) << 16) | low).to_le_bytes().to_vec()
                    }
                })
                .collect();
            let view =
                TensorView::new(dtype, vec![256, 256], &bytes).expect("invariant: codebook valide");
            let serialized = serialize([("book", view)], None).expect("invariant: sérialisation");
            let mut file = tempfile::NamedTempFile::new().expect("invariant: fichier temporaire");
            std::io::Write::write_all(&mut file, &serialized).expect("invariant: écriture fixture");
            let source = DraftWeights::open(file.path(), ForwardRuntime::cpu())?;
            let legacy = crate::safetensor::load_float_tensor(file.path(), "book")?;
            for compact in [false, true] {
                let book = source.codebook_with_storage("book", compact)?;
                assert_eq!(book.is_bf16(), compact && !inexact);
                assert_eq!(book.shape(), legacy.shape());
                for row in 0..256 {
                    assert_eq!(
                        bytemuck::cast_slice::<_, u8>(book.row_slice(row)?.as_ref()),
                        bytemuck::cast_slice::<_, u8>(legacy.row_slice(row)?)
                    );
                }
                assert!(book.row_slice(256).is_err());
            }
        }
        Ok(())
    }

    #[test]
    fn compact_codebooks_keep_selector_scores_and_drafts() -> Result<()> {
        let file = fixture(Dtype::BF16);
        let source = DraftWeights::open(file.path(), ForwardRuntime::cpu())?;
        let selector = |compact| -> Result<super::super::DFlashSelector> {
            Ok(super::super::DFlashSelector {
                hidden_projection: Linear::new(Tensor::zeros(vec![COLS, 1])?, None)?,
                predecessor_codebook: source.codebook_with_storage("fc.weight", compact)?,
                successor_codebook: source.codebook_with_storage("fc.weight", compact)?,
                rank: COLS,
                top_k: 3,
            })
        };
        let (old, new) = (selector(false)?, selector(true)?);
        let projected: Vec<_> = (0..COLS * 2).map(|i| i as f32 / 257.0 - 0.5).collect();
        let candidates = vec![vec![(0, 0.5), (16, -0.25), (32, 1.0)]; 2];
        let bytes = |scores: Vec<Vec<f32>>| {
            scores
                .into_iter()
                .flatten()
                .map(f32::to_bits)
                .collect::<Vec<_>>()
        };
        assert_eq!(
            bytes(old.transition_scores(&projected, &candidates, 2)?),
            bytes(new.transition_scores(&projected, &candidates, 2)?)
        );
        assert_eq!(
            old.select_projected(&projected, &candidates, 2)?,
            new.select_projected(&projected, &candidates, 2)?
        );
        assert_eq!(
            bytes(vec![old.edge_scores_from(
                &projected[..COLS],
                16,
                &candidates[0]
            )?]),
            bytes(vec![new.edge_scores_from(
                &projected[..COLS],
                16,
                &candidates[0]
            )?])
        );
        Ok(())
    }

    #[test]
    fn lazy_loader_rejects_invalid_file_shape_and_group() -> Result<()> {
        let file = fixture(Dtype::F32);
        let source = DraftWeights::open(file.path(), ForwardRuntime::cpu())?;
        assert!(source.tensor("missing").is_err());
        assert!(source.linear("fc", ROWS + 1, COLS, None).is_err());
        assert!(source
            .linear(
                "fc",
                ROWS,
                COLS,
                Some(DFlashQuant {
                    group_size: 63,
                    bits: 4
                })
            )
            .is_err());
        file.as_file()
            .set_len(12)
            .expect("invariant: troncature fixture");
        assert!(DraftWeights::open(file.path(), ForwardRuntime::cpu()).is_err());
        assert!(source.tensor("fc.weight").is_err());
        Ok(())
    }

    #[test]
    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[ignore = "requiert un device Metal ; poids synthétiques uniquement"]
    fn streamed_metal_weights_match_legacy_upload() -> Result<()> {
        let metal = crate::MetalExecutor::new()?;
        let file = fixture(Dtype::BF16);
        let source = DraftWeights::open(file.path(), ForwardRuntime::metal(&metal))?;
        let quant = Some(DFlashQuant {
            group_size: 64,
            bits: 4,
        });
        let loaded = source.linear("fc", ROWS, COLS, quant)?;
        let crate::LinearWeight::AffineQuantized(weight) = loaded.weight() else {
            unreachable!("invariant: affine")
        };
        let reference = legacy(file.path(), 4)?;
        assert_eq!(weight.packed_data(), reference.packed_data());
        assert_eq!(
            weight.packed_metal_view().is_some(),
            crate::runtime_flags::single_copy_weights_enabled()
        );
        if crate::runtime_flags::single_copy_weights_enabled() {
            let rounded = |data: &[f32]| {
                data.iter()
                    .map(|v| f32::from_bits(u32::from(crate::quantization::bf16_bits(*v)) << 16))
                    .collect::<Vec<_>>()
            };
            assert_eq!(*weight.scales_f32(), rounded(&reference.scales_f32()));
            assert_eq!(*weight.biases_f32(), rounded(&reference.biases_f32()));
            let clone = weight.clone();
            assert_eq!(
                weight.packed_metal_view().map(|(b, o)| (b.contents(), o)),
                clone.packed_metal_view().map(|(b, o)| (b.contents(), o))
            );
        }
        let input = Tensor::from_vec(
            vec![1, COLS],
            (0..COLS).map(|i| i as f32 / 128.0 - 0.5).collect(),
        )?;
        let old = metal.matmul_rhs_t_affine(&input, &reference)?;
        let new = loaded.forward_with_runtime(&input, ForwardRuntime::metal(&metal))?;
        assert_eq!(
            old.data().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            new.data().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        Ok(())
    }
}
