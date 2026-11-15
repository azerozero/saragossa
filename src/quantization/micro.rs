//! Formats à micro-échelles MLX : codes E2M1/E4M3 et échelles sur un octet.

use crate::{InferError, Result, Tensor};
use rayon::prelude::*;

/// Format des poids et de leurs micro-échelles, sans biais affine.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MicroMode {
    /// E2M1, échelle E8M0 par groupe de 32.
    Mxfp4,
    /// E2M1, échelle E4M3 par groupe de 16.
    Nvfp4,
    /// E4M3, échelle E8M0 par groupe de 32.
    Mxfp8,
}

impl MicroMode {
    /// Reconnaît un mode MLX ; `affine` reste sur le chemin historique.
    ///
    /// # Errors
    /// Renvoie une erreur typée pour tout mode inconnu.
    pub fn parse(mode: &str) -> Result<Option<Self>> {
        match mode {
            "affine" => Ok(None),
            "mxfp4" => Ok(Some(Self::Mxfp4)),
            "nvfp4" => Ok(Some(Self::Nvfp4)),
            "mxfp8" => Ok(Some(Self::Mxfp8)),
            _ => Err(InferError::UnsupportedQuantization(mode.to_owned())),
        }
    }

    /// Nombre de poids partageant une échelle.
    #[must_use]
    pub const fn group_size(self) -> usize {
        match self {
            Self::Nvfp4 => 16,
            _ => 32,
        }
    }

    /// Nombre de bits par code de poids.
    #[must_use]
    pub const fn bits(self) -> usize {
        match self {
            Self::Mxfp8 => 8,
            _ => 4,
        }
    }

    /// Nom sérialisé par MLX.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Mxfp4 => "mxfp4",
            Self::Nvfp4 => "nvfp4",
            Self::Mxfp8 => "mxfp8",
        }
    }

    fn scale(self, code: u8) -> f32 {
        if self == Self::Nvfp4 {
            decode_e4m3(code)
        } else {
            decode_e8m0(code)
        }
    }
}

/// Décode E2M1, y compris le zéro signé (MLX `fp4.h`).
#[must_use]
pub fn decode_e2m1(code: u8) -> f32 {
    let magnitude = [0.0_f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0][usize::from(code & 7)];
    if code & 8 != 0 {
        -magnitude
    } else {
        magnitude
    }
}

/// Décode le E4M3 de MLX `fp8.h`, où 0x7f représente 480.
#[must_use]
pub fn decode_e4m3(code: u8) -> f32 {
    let exponent = (code >> 3) & 15;
    let mantissa = code & 7;
    let magnitude = if exponent == 0 {
        f32::from(mantissa) / 512.0
    } else {
        (1.0 + f32::from(mantissa) / 8.0) * f32::from_bits(u32::from(exponent + 120) << 23)
    };
    if code & 128 != 0 {
        -magnitude
    } else {
        magnitude
    }
}

/// Décode E8M0 selon MLX : 0 = 2^-127, 255 = +inf.
#[must_use]
pub fn decode_e8m0(code: u8) -> f32 {
    f32::from_bits(if code == 0 {
        0x0040_0000
    } else {
        u32::from(code) << 23
    })
}

/// Matrice MLX compacte `[N,K]`, codes U32 little-endian et échelles U8.
#[derive(Clone, Debug)]
pub struct MicroQuantizedTensor {
    #[cfg_attr(not(all(target_os = "macos", feature = "metal")), allow(dead_code))]
    weight_id: u64,
    shape: [usize; 2],
    storage: MicroStorage,
    mode: MicroMode,
}

/// Codes et échelles possédés, ou adossés aux MTLBuffer partagés que lisent
/// les kernels : la mémoire unifiée évite alors toute seconde copie.
#[derive(Clone, Debug)]
enum MicroStorage {
    Owned {
        packed: Vec<u32>,
        scales: Vec<u8>,
    },
    #[cfg(all(target_os = "macos", feature = "metal"))]
    MetalShared {
        packed: metal::Buffer,
        scales: metal::Buffer,
    },
}

/// Nombre de mots U32 et d'échelles d'une matrice micro valide.
fn micro_lengths(shape: [usize; 2], mode: MicroMode) -> Result<(usize, usize)> {
    let [rows, cols] = shape;
    let count = rows
        .checked_mul(cols)
        .ok_or_else(|| InferError::Shape("poids micro trop grand".into()))?;
    if rows == 0 || cols == 0 || cols % mode.group_size() != 0 {
        return Err(InferError::Shape(format!(
            "poids {} de forme {shape:?} invalide",
            mode.name()
        )));
    }
    Ok((count / (32 / mode.bits()), count / mode.group_size()))
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[expect(
    unsafe_code,
    reason = "vue CPU en lecture seule des échelles d'un MTLBuffer de poids partagé"
)]
fn metal_shared_u8_slice(buffer: &metal::Buffer, len: usize) -> &[u8] {
    // SAFETY: `new_metal_shared` vérifie `len <= buffer.length()` et un pointeur
    // non nul. StorageModeShared rend `contents` visible côté hôte ; `buffer`
    // est retenu par le stockage pendant la vue, et les MTLBuffer de poids ne
    // sont jamais écrits après chargement.
    unsafe { std::slice::from_raw_parts(buffer.contents().cast::<u8>(), len) }
}

impl PartialEq for MicroQuantizedTensor {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape
            && self.mode == other.mode
            && self.packed() == other.packed()
            && self.scales() == other.scales()
    }
}

impl MicroQuantizedTensor {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) fn weight_id(&self) -> u64 {
        self.weight_id
    }

    /// Valide les formes compactes avant tout accès aux codes.
    ///
    /// # Errors
    /// Refuse les dimensions nulles, les débordements et les payloads incohérents.
    pub fn new(
        shape: [usize; 2],
        packed: Vec<u32>,
        scales: Vec<u8>,
        mode: MicroMode,
    ) -> Result<Self> {
        let (words, groups) = micro_lengths(shape, mode)?;
        if packed.len() != words || scales.len() != groups {
            return Err(InferError::Shape(format!(
                "poids {} {shape:?}: {} mots, {} échelles",
                mode.name(),
                packed.len(),
                scales.len()
            )));
        }
        Ok(Self {
            weight_id: super::next_weight_id()?,
            shape,
            storage: MicroStorage::Owned { packed, scales },
            mode,
        })
    }

    /// Adosse le poids aux MTLBuffer partagés lus par Metal, sans copie CPU.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) fn new_metal_shared(
        shape: [usize; 2],
        packed: metal::Buffer,
        scales: metal::Buffer,
        mode: MicroMode,
    ) -> Result<Self> {
        let (words, groups) = micro_lengths(shape, mode)?;
        let fits = |buffer: &metal::Buffer, bytes: usize| {
            usize::try_from(buffer.length()).is_ok_and(|len| len >= bytes)
                && !buffer.contents().is_null()
        };
        let packed_bytes = words
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| InferError::Shape("poids micro trop grand".into()))?;
        if !fits(&packed, packed_bytes) || !fits(&scales, groups) {
            return Err(InferError::Shape(format!(
                "buffers {} trop courts pour {shape:?}",
                mode.name()
            )));
        }
        Ok(Self {
            weight_id: super::next_weight_id()?,
            shape,
            storage: MicroStorage::MetalShared { packed, scales },
            mode,
        })
    }

    /// Déplace codes et échelles dans des MTLBuffer partagés, puis libère
    /// les vecteurs CPU : la vue CPU et les kernels lisent la même mémoire.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) fn move_to_metal_shared(&mut self, metal: &crate::MetalExecutor) -> Result<()> {
        if let MicroStorage::Owned { packed, scales } = &self.storage {
            let packed = metal.weight_buffer_from_u32(packed)?;
            let scales = metal.weight_buffer_from_u8(scales)?;
            self.storage = MicroStorage::MetalShared { packed, scales };
        }
        Ok(())
    }

    /// Buffers partagés `(codes, échelles)` quand le poids y est adossé.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) fn metal_views(&self) -> Option<(&metal::Buffer, &metal::Buffer)> {
        match &self.storage {
            MicroStorage::Owned { .. } => None,
            MicroStorage::MetalShared { packed, scales } => Some((packed, scales)),
        }
    }

    /// Forme logique dense.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    /// Mode MLX.
    #[must_use]
    pub fn mode(&self) -> MicroMode {
        self.mode
    }
    /// Codes empaquetés, colonnes de poids faible en premier.
    #[must_use]
    pub fn packed(&self) -> &[u32] {
        match &self.storage {
            MicroStorage::Owned { packed, .. } => packed,
            #[cfg(all(target_os = "macos", feature = "metal"))]
            MicroStorage::MetalShared { packed, .. } => {
                let words = self.shape[0] * self.shape[1] / (32 / self.mode.bits());
                super::metal_shared_u32_slice(packed, 0, words)
            }
        }
    }
    /// Octets d'échelle sans conversion ni biais.
    #[must_use]
    pub fn scales(&self) -> &[u8] {
        match &self.storage {
            MicroStorage::Owned { scales, .. } => scales,
            #[cfg(all(target_os = "macos", feature = "metal"))]
            MicroStorage::MetalShared { scales, .. } => metal_shared_u8_slice(
                scales,
                self.shape[0] * self.shape[1] / self.mode.group_size(),
            ),
        }
    }

    fn value(&self, index: usize) -> f32 {
        let bits = self.mode.bits();
        let lanes = 32 / bits;
        let code =
            ((self.packed()[index / lanes] >> ((index % lanes) * bits)) & ((1 << bits) - 1)) as u8;
        let value = if bits == 4 {
            decode_e2m1(code)
        } else {
            decode_e4m3(code)
        };
        value
            * self
                .mode
                .scale(self.scales()[index / self.mode.group_size()])
    }

    /// Déquantifie une ligne sans matérialiser la matrice entière.
    ///
    /// # Errors
    /// Refuse une ligne hors matrice.
    pub fn row(&self, row: usize) -> Result<Vec<f32>> {
        if row >= self.shape[0] {
            return Err(InferError::Dimension(format!(
                "ligne micro {row} hors matrice"
            )));
        }
        Ok((row * self.shape[1]..(row + 1) * self.shape[1])
            .map(|i| self.value(i))
            .collect())
    }

    /// Oracle f32 de déquantification.
    ///
    /// # Errors
    /// Propage une erreur de construction du tenseur.
    pub fn dequantize(&self) -> Result<Tensor> {
        Tensor::from_vec(
            self.shape.to_vec(),
            (0..self.shape[0] * self.shape[1])
                .map(|i| self.value(i))
                .collect(),
        )
    }

    /// Produit CPU de référence avec accumulation f32 séquentielle par ligne.
    ///
    /// # Errors
    /// Refuse des dimensions incompatibles ou une taille de sortie débordante.
    pub fn matmul_rhs_t(&self, input: &Tensor) -> Result<Tensor> {
        let (batch, cols) = input.as_matrix()?;
        let rows = self.shape[0];
        if cols != self.shape[1] {
            return Err(InferError::Dimension(
                "matmul micro: dimensions incompatibles".into(),
            ));
        }
        let len = batch
            .checked_mul(rows)
            .ok_or_else(|| InferError::Shape("sortie micro trop grande".into()))?;
        let mut output = vec![0.0; len];
        output.par_iter_mut().enumerate().for_each(|(i, value)| {
            let x = &input.data()[(i / rows) * cols..(i / rows + 1) * cols];
            *value = x
                .iter()
                .enumerate()
                .map(|(k, x)| x * self.value((i % rows) * cols + k))
                .sum();
        });
        Tensor::from_vec(vec![batch, rows], output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manual_decode_tables() {
        let values = [0.0_f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        for (i, value) in values.into_iter().enumerate() {
            assert_eq!(decode_e2m1(i as u8).to_bits(), value.to_bits());
            assert_eq!(decode_e2m1(i as u8 + 8).to_bits(), (-value).to_bits());
        }
        for (code, value) in [
            (0, 0.0_f32),
            (1, 1.0 / 512.0),
            (7, 7.0 / 512.0),
            (8, 1.0 / 64.0),
            (56, 1.0),
            (126, 448.0),
            (127, 480.0),
        ] {
            assert_eq!(decode_e4m3(code).to_bits(), value.to_bits());
            assert_eq!(decode_e4m3(code | 128).to_bits(), (-value).to_bits());
        }
        assert_eq!(decode_e8m0(0).to_bits(), 0x0040_0000);
        assert_eq!(decode_e8m0(1).to_bits(), 0x0080_0000);
        assert_eq!(decode_e8m0(127), 1.0);
        assert_eq!(decode_e8m0(128), 2.0);
        assert_eq!(decode_e8m0(254).to_bits(), 0x7f00_0000);
        assert_eq!(decode_e8m0(255), f32::INFINITY);
    }

    #[test]
    fn manual_packing_groups_and_rows() -> Result<()> {
        for mode in [MicroMode::Mxfp4, MicroMode::Nvfp4] {
            let k = mode.group_size() * 2;
            let scales = if mode == MicroMode::Nvfp4 {
                vec![56, 64, 48, 0]
            } else {
                vec![127, 128, 126, 127]
            };
            let w = MicroQuantizedTensor::new([2, k], vec![0xfedc_ba98; 2 * k / 8], scales, mode)?;
            assert_eq!(w.row(0)?[1], -0.5);
            assert_eq!(w.row(0)?[mode.group_size() + 7], -12.0);
            assert_eq!(w.row(1)?[7], -3.0);
            assert!(w.row(2).is_err());
        }
        let w = MicroQuantizedTensor::new(
            [1, 64],
            vec![0xff7e_b838; 16],
            vec![127, 128],
            MicroMode::Mxfp8,
        )?;
        assert_eq!(&w.row(0)?[..4], &[1.0, -1.0, 448.0, -480.0]);
        assert_eq!(&w.row(0)?[32..36], &[2.0, -2.0, 896.0, -960.0]);
        let tiny =
            MicroQuantizedTensor::new([1, 32], vec![0x7654_3210; 4], vec![0], MicroMode::Mxfp4)?;
        assert_eq!(
            tiny.row(0)?[..8]
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            [0, 0x20_0000, 0x40_0000, 0x60_0000, 0x80_0000, 0xc0_0000, 0x100_0000, 0x140_0000]
        );
        let tiny =
            MicroQuantizedTensor::new([1, 32], vec![0x0101_0101; 8], vec![0], MicroMode::Mxfp8)?;
        assert_eq!(tiny.row(0)?[0].to_bits(), 0x2000); // 2^-9 × 2^-127 = 2^-136.
        Ok(())
    }

    #[test]
    fn reject_invalid_layouts_and_modes() {
        assert!(matches!(
            MicroMode::parse("int4"),
            Err(InferError::UnsupportedQuantization(_))
        ));
        for shape in [[0, 32], [1, 31], [usize::MAX, 32], [1, 32]] {
            assert!(MicroQuantizedTensor::new(shape, vec![], vec![], MicroMode::Mxfp4).is_err());
        }
    }
}
