//! Représentation et multiplication des poids quantifiés affine.

use crate::{InferError, Result, Tensor};
use rayon::prelude::*;

const PARALLEL_QUANT_MATMUL_OUTPUT_THRESHOLD: usize = 1024;
const PARALLEL_QUANT_MATMUL_INNER_THRESHOLD: usize = 128;

/// Poids affine packé en `u32`, conservé compact en mémoire.
#[derive(Clone, Debug, PartialEq)]
pub struct AffineQuantizedTensor {
    shape: Vec<usize>,
    packed_shape: Vec<usize>,
    packed: Vec<u32>,
    scales: Tensor,
    biases: Tensor,
    group_size: usize,
    bits: usize,
}

impl AffineQuantizedTensor {
    /// Construit un poids affine compact.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si les formes ne correspondent pas au packing affine.
    pub fn new(
        packed_shape: &[usize],
        packed: Vec<u32>,
        scales: Tensor,
        biases: Tensor,
        group_size: usize,
        bits: usize,
    ) -> Result<Self> {
        let params = affine_params(
            packed_shape,
            packed.len(),
            &scales,
            &biases,
            group_size,
            bits,
        )?;
        Ok(Self {
            shape: vec![params.rows, params.cols],
            packed_shape: packed_shape.to_vec(),
            packed,
            scales,
            biases,
            group_size,
            bits,
        })
    }

    /// Renvoie la forme dense logique `[rows, cols]`.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) fn packed_shape(&self) -> &[usize] {
        &self.packed_shape
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) fn packed_data(&self) -> &[u32] {
        &self.packed
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) fn scales(&self) -> &Tensor {
        &self.scales
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) fn biases(&self) -> &Tensor {
        &self.biases
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) fn group_size(&self) -> usize {
        self.group_size
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) fn bits(&self) -> usize {
        self.bits
    }

    /// Multiplie `input` par la transposée du poids logique dense.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si les dimensions de `input` sont incompatibles.
    pub fn matmul_rhs_t(&self, input: &Tensor) -> Result<Tensor> {
        let (batch, in_dim) = input.as_matrix()?;
        let [out_dim, weight_in_dim] = self.shape.as_slice() else {
            return Err(InferError::Dimension(format!(
                "poids quantifié attendu rang 2, reçu {:?}",
                self.shape
            )));
        };
        if in_dim != *weight_in_dim {
            return Err(InferError::Dimension(format!(
                "matmul quantifié x=[{batch},{in_dim}] rhs_t_source=[{out_dim},{weight_in_dim}]"
            )));
        }

        let mut out = vec![0.0_f32; batch * out_dim];
        if should_parallelize_quant_matmul(out.len(), in_dim) {
            out.par_iter_mut().enumerate().for_each(|(idx, value)| {
                let b = idx / out_dim;
                let row = idx % out_dim;
                let input_row = &input.data()[b * in_dim..(b + 1) * in_dim];
                *value = self.dot_row(input_row, row);
            });
        } else {
            for b in 0..batch {
                for row in 0..*out_dim {
                    let input_row = &input.data()[b * in_dim..(b + 1) * in_dim];
                    out[b * out_dim + row] = self.dot_row(input_row, row);
                }
            }
        }
        Tensor::from_vec(vec![batch, *out_dim], out)
    }

    /// Déquantifie le poids compact en tenseur dense.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si la représentation compacte est incohérente.
    pub fn dequantize(&self) -> Result<Tensor> {
        let [rows, cols] = self.shape.as_slice() else {
            return Err(InferError::Dimension(format!(
                "poids quantifié attendu rang 2, reçu {:?}",
                self.shape
            )));
        };
        let mut out = vec![0.0_f32; rows * cols];
        for row in 0..*rows {
            for col in 0..*cols {
                out[row * cols + col] = self.value(row, col);
            }
        }
        Tensor::from_vec(self.shape.clone(), out)
    }

    /// Déquantifie une seule ligne logique.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si `row` est hors bornes.
    pub fn row(&self, row: usize) -> Result<Vec<f32>> {
        let [rows, cols] = self.shape.as_slice() else {
            return Err(InferError::Dimension(format!(
                "poids quantifié attendu rang 2, reçu {:?}",
                self.shape
            )));
        };
        if row >= *rows {
            return Err(InferError::Dimension(format!(
                "row quantifiée {row} hors bornes pour {rows} lignes"
            )));
        }
        let mut out = Vec::with_capacity(*cols);
        for col in 0..*cols {
            out.push(self.value(row, col));
        }
        Ok(out)
    }

    fn value(&self, row: usize, col: usize) -> f32 {
        let quantized = self.bitpacked_value(row, col) as f32;
        let groups = self.shape[1] / self.group_size;
        let group = col / self.group_size;
        let affine_index = row * groups + group;
        quantized * self.scales.data()[affine_index] + self.biases.data()[affine_index]
    }

    fn dot_row(&self, input_row: &[f32], row: usize) -> f32 {
        match self.bits {
            4 if self.group_size % 8 == 0 => return self.dot_row_u4(input_row, row),
            8 if self.group_size % 4 == 0 => return self.dot_row_u8(input_row, row),
            _ => {}
        }
        let groups = self.shape[1] / self.group_size;
        let mut acc = 0.0_f32;

        for group in 0..groups {
            let affine_index = row * groups + group;
            let scale = self.scales.data()[affine_index];
            let bias = self.biases.data()[affine_index];
            for col in group * self.group_size..(group + 1) * self.group_size {
                let quantized = self.bitpacked_value(row, col) as f32;
                acc += input_row[col] * (quantized * scale + bias);
            }
        }
        acc
    }

    fn bitpacked_value(&self, row: usize, col: usize) -> u32 {
        let packed_cols = self.packed_shape[1];
        let bit_offset = col * self.bits;
        let word_col = bit_offset / 32;
        let shift = bit_offset % 32;
        let row_start = row * packed_cols;
        let mask = (1_u32 << self.bits) - 1;
        let low = self.packed[row_start + word_col] >> shift;
        if shift + self.bits <= 32 {
            return low & mask;
        }
        let high_bits = shift + self.bits - 32;
        let high_mask = (1_u32 << high_bits) - 1;
        let high = self
            .packed
            .get(row_start + word_col + 1)
            .copied()
            .unwrap_or(0)
            & high_mask;
        (low | (high << (32 - shift))) & mask
    }

    fn dot_row_u4(&self, input_row: &[f32], row: usize) -> f32 {
        let packed_cols = self.packed_shape[1];
        let groups = self.shape[1] / self.group_size;
        let words_per_group = self.group_size / 8;
        let mut acc = 0.0_f32;
        for group in 0..groups {
            let affine_index = row * groups + group;
            let scale = self.scales.data()[affine_index];
            let bias = self.biases.data()[affine_index];
            let first_word = group * words_per_group;
            for word_offset in 0..words_per_group {
                let word_col = first_word + word_offset;
                let packed = self.packed[row * packed_cols + word_col];
                let base = word_col * 8;
                acc += input_row[base] * (((packed & 0x0f) as f32) * scale + bias);
                acc += input_row[base + 1] * ((((packed >> 4) & 0x0f) as f32) * scale + bias);
                acc += input_row[base + 2] * ((((packed >> 8) & 0x0f) as f32) * scale + bias);
                acc += input_row[base + 3] * ((((packed >> 12) & 0x0f) as f32) * scale + bias);
                acc += input_row[base + 4] * ((((packed >> 16) & 0x0f) as f32) * scale + bias);
                acc += input_row[base + 5] * ((((packed >> 20) & 0x0f) as f32) * scale + bias);
                acc += input_row[base + 6] * ((((packed >> 24) & 0x0f) as f32) * scale + bias);
                acc += input_row[base + 7] * ((((packed >> 28) & 0x0f) as f32) * scale + bias);
            }
        }
        acc
    }

    fn dot_row_u8(&self, input_row: &[f32], row: usize) -> f32 {
        let packed_cols = self.packed_shape[1];
        let groups = self.shape[1] / self.group_size;
        let words_per_group = self.group_size / 4;
        let mut acc = 0.0_f32;
        for group in 0..groups {
            let affine_index = row * groups + group;
            let scale = self.scales.data()[affine_index];
            let bias = self.biases.data()[affine_index];
            let first_word = group * words_per_group;
            for word_offset in 0..words_per_group {
                let word_col = first_word + word_offset;
                let packed = self.packed[row * packed_cols + word_col];
                let base = word_col * 4;
                acc += input_row[base] * (((packed & 0xff) as f32) * scale + bias);
                acc += input_row[base + 1] * ((((packed >> 8) & 0xff) as f32) * scale + bias);
                acc += input_row[base + 2] * ((((packed >> 16) & 0xff) as f32) * scale + bias);
                acc += input_row[base + 3] * ((((packed >> 24) & 0xff) as f32) * scale + bias);
            }
        }
        acc
    }
}

struct AffineParams {
    rows: usize,
    cols: usize,
}

/// Déquantifie un poids affine MLX packé en `u32`.
///
/// # Errors
///
/// Renvoie une erreur si les formes ne correspondent pas au packing affine.
pub fn dequantize_affine_u32(
    packed_shape: &[usize],
    packed: &[u32],
    scales: &Tensor,
    biases: &Tensor,
    group_size: usize,
    bits: usize,
) -> Result<Tensor> {
    AffineQuantizedTensor::new(
        packed_shape,
        packed.to_vec(),
        scales.clone(),
        biases.clone(),
        group_size,
        bits,
    )?
    .dequantize()
}

fn affine_params(
    packed_shape: &[usize],
    packed_len: usize,
    scales: &Tensor,
    biases: &Tensor,
    group_size: usize,
    bits: usize,
) -> Result<AffineParams> {
    let [rows, packed_cols] = packed_shape else {
        return Err(InferError::Dimension(format!(
            "poids quantifié attendu rang 2, reçu {packed_shape:?}"
        )));
    };
    if *rows == 0 || *packed_cols == 0 || packed_len != rows * packed_cols {
        return Err(InferError::Shape(format!(
            "poids quantifié shape={packed_shape:?}, éléments={}",
            packed_len
        )));
    }
    if group_size == 0 || bits == 0 || bits > 16 {
        return Err(InferError::Config(format!(
            "quantification affine invalide: group_size={group_size}, bits={bits}"
        )));
    }
    let cols_times_bits = packed_cols
        .checked_mul(32)
        .ok_or_else(|| InferError::Shape("poids quantifié trop large".to_string()))?;
    if cols_times_bits % bits != 0 {
        return Err(InferError::Shape(format!(
            "packed_cols={packed_cols} incompatible avec bits={bits}"
        )));
    }
    let cols = cols_times_bits / bits;
    if cols % group_size != 0 {
        return Err(InferError::Shape(format!(
            "cols={cols} non divisible par group_size={group_size}"
        )));
    }
    let groups = cols / group_size;
    if scales.shape() != [*rows, groups] || biases.shape() != [*rows, groups] {
        return Err(InferError::Dimension(format!(
            "scales/biases attendus [{rows},{groups}], reçu scales={:?}, biases={:?}",
            scales.shape(),
            biases.shape()
        )));
    }

    Ok(AffineParams { rows: *rows, cols })
}

fn should_parallelize_quant_matmul(outputs: usize, inner: usize) -> bool {
    outputs >= PARALLEL_QUANT_MATMUL_OUTPUT_THRESHOLD
        && inner >= PARALLEL_QUANT_MATMUL_INNER_THRESHOLD
}

pub(crate) fn bytes_to_u32(bytes: &[u8], name: &str) -> Result<Vec<u32>> {
    let chunks = bytes.chunks_exact(4);
    if !chunks.remainder().is_empty() {
        return Err(InferError::Shape(format!(
            "tensor {name} U32 avec {} octets non multiple de 4",
            bytes.len()
        )));
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in chunks {
        let arr = <[u8; 4]>::try_from(chunk)
            .map_err(|_| InferError::Shape(format!("chunk U32 invalide pour {name}")))?;
        out.push(u32::from_le_bytes(arr));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn dequantizes_affine_u8_packed_rows() {
        let packed = [
            pack_lanes(&[255, 0, 0, 0], 8),
            pack_lanes(&[0, 255, 0, 0], 8),
        ];
        let scales =
            Tensor::from_vec(vec![2, 2], vec![1.0 / 255.0; 4]).expect("invariant: scales valides");
        let biases = Tensor::from_vec(vec![2, 2], vec![0.0; 4]).expect("invariant: biases valides");

        let dense = dequantize_affine_u32(&[2, 1], &packed, &scales, &biases, 2, 8)
            .expect("invariant: déquantification affine valide");

        assert_close(dense.data(), &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn dequantizes_affine_u4_packed_row() {
        let packed = [pack_lanes(&[15, 0, 7, 8, 1, 2, 3, 4], 4)];
        let scales =
            Tensor::from_vec(vec![1, 2], vec![1.0 / 15.0, 2.0]).expect("invariant: scales valides");
        let biases =
            Tensor::from_vec(vec![1, 2], vec![0.0, -1.0]).expect("invariant: biases valides");

        let dense = dequantize_affine_u32(&[1, 1], &packed, &scales, &biases, 4, 4)
            .expect("invariant: déquantification affine valide");

        assert_close(
            dense.data(),
            &[1.0, 0.0, 7.0 / 15.0, 8.0 / 15.0, 1.0, 3.0, 5.0, 7.0],
        );
    }

    #[test]
    fn dequantizes_affine_u6_bitstream_row() {
        let values = [1, 2, 3, 4, 5, 6, 7, 63, 8, 9, 10, 11, 12, 13, 14, 15];
        let packed = pack_bitstream(&values, 6);
        let scales =
            Tensor::from_vec(vec![1, 2], vec![1.0, 1.0]).expect("invariant: scales valides");
        let biases =
            Tensor::from_vec(vec![1, 2], vec![0.0, 0.0]).expect("invariant: biases valides");

        let dense = dequantize_affine_u32(&[1, packed.len()], &packed, &scales, &biases, 8, 6)
            .expect("invariant: déquantification affine 6-bit valide");

        assert_close(
            &dense.data()[..values.len()],
            &values.iter().map(|value| *value as f32).collect::<Vec<_>>(),
        );
    }

    #[test]
    fn rejects_mismatched_scale_shape() {
        let scales = Tensor::from_vec(vec![1, 1], vec![1.0]).expect("invariant: scales valides");
        let biases = Tensor::from_vec(vec![1, 1], vec![0.0]).expect("invariant: biases valides");

        let err = dequantize_affine_u32(&[2, 1], &[0, 0], &scales, &biases, 2, 8)
            .expect_err("invariant: forme scales rejetée");

        assert!(matches!(err, InferError::Dimension(_)));
    }

    #[test]
    fn compact_affine_matmul_matches_dense_dequantization() {
        let packed = vec![
            pack_lanes(&[255, 0, 0, 0], 8),
            pack_lanes(&[0, 255, 0, 0], 8),
        ];
        let scales =
            Tensor::from_vec(vec![2, 2], vec![1.0 / 255.0; 4]).expect("invariant: scales valides");
        let biases = Tensor::from_vec(vec![2, 2], vec![0.0; 4]).expect("invariant: biases valides");
        let compact = AffineQuantizedTensor::new(&[2, 1], packed, scales, biases, 2, 8)
            .expect("invariant: poids quantifié compact valide");
        let input =
            Tensor::from_vec(vec![1, 4], vec![2.0, 3.0, 5.0, 7.0]).expect("invariant: input");

        let dense = compact
            .dequantize()
            .expect("invariant: déquantification valide");
        let dense_out = input
            .matmul_rhs_t(&dense)
            .expect("invariant: matmul dense valide");
        let compact_out = compact
            .matmul_rhs_t(&input)
            .expect("invariant: matmul compact valide");

        assert_close(compact_out.data(), dense_out.data());
    }

    proptest! {
        #[test]
        fn affine_u4_row_unpack_matches_packed_lanes(
            rows in 1_usize..4,
            packed_cols in 1_usize..4,
            lanes in proptest::collection::vec(0_u32..16, 8..96),
            scales_raw in proptest::collection::vec(-2.0_f32..2.0, 1..12),
            biases_raw in proptest::collection::vec(-1.0_f32..1.0, 1..12),
        ) {
            let group_size = 8;
            let bits = 4;
            let cols = packed_cols * 8;
            let groups = cols / group_size;
            let lane_count = rows * packed_cols * 8;
            let affine_count = rows * groups;

            let packed = (0..rows * packed_cols)
                .map(|word| {
                    let mut values = [0_u32; 8];
                    for lane in 0..8 {
                        values[lane] = lanes[(word * 8 + lane) % lanes.len()];
                    }
                    pack_lanes(&values, bits)
                })
                .collect::<Vec<_>>();
            let scales = (0..affine_count)
                .map(|idx| scales_raw[idx % scales_raw.len()])
                .collect::<Vec<_>>();
            let biases = (0..affine_count)
                .map(|idx| biases_raw[idx % biases_raw.len()])
                .collect::<Vec<_>>();
            let scales = Tensor::from_vec(vec![rows, groups], scales)
                .expect("invariant: scales générées valides");
            let biases = Tensor::from_vec(vec![rows, groups], biases)
                .expect("invariant: biases générées valides");
            let compact = AffineQuantizedTensor::new(
                &[rows, packed_cols],
                packed,
                scales,
                biases,
                group_size,
                bits,
            )
            .expect("invariant: poids quantifié généré valide");

            for row in 0..rows {
                let unpacked = compact.row(row).expect("invariant: ligne dans les bornes");
                let expected = (0..cols)
                    .map(|col| {
                        let word = row * packed_cols + col / 8;
                        let lane = col % 8;
                        let quantized = lanes[(word * 8 + lane) % lanes.len()] as f32;
                        let affine = row * groups + col / group_size;
                        quantized * compact.scales.data()[affine] + compact.biases.data()[affine]
                    })
                    .collect::<Vec<_>>();
                prop_assert_eq!(unpacked, expected);
            }

            prop_assert_eq!(lane_count, rows * cols);
        }
    }

    fn pack_lanes(values: &[u32], bits: usize) -> u32 {
        values
            .iter()
            .enumerate()
            .fold(0_u32, |word, (idx, value)| word | (value << (idx * bits)))
    }

    fn pack_bitstream(values: &[u32], bits: usize) -> Vec<u32> {
        let total_bits = values.len() * bits;
        let mut out = vec![0_u32; total_bits.div_ceil(32)];
        for (idx, value) in values.iter().copied().enumerate() {
            let bit_offset = idx * bits;
            let word = bit_offset / 32;
            let shift = bit_offset % 32;
            out[word] |= value << shift;
            if shift + bits > 32 {
                out[word + 1] |= value >> (32 - shift);
            }
        }
        out
    }

    fn assert_close(left: &[f32], right: &[f32]) {
        assert_eq!(left.len(), right.len());
        for (idx, (a, b)) in left.iter().zip(right.iter()).enumerate() {
            assert!((a - b).abs() <= 1.0e-6, "index={idx} left={a} right={b}");
        }
    }
}
