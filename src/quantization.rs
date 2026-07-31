//! Représentation et multiplication des poids quantifiés affine.

use crate::{InferError, Result, Tensor};
use rayon::prelude::*;
use std::borrow::Cow;
use std::sync::atomic::{AtomicU64, Ordering};

const PARALLEL_QUANT_MATMUL_OUTPUT_THRESHOLD: usize = 1024;
const PARALLEL_QUANT_MATMUL_INNER_THRESHOLD: usize = 128;
static NEXT_WEIGHT_ID: AtomicU64 = AtomicU64::new(1);

/// Stockage des codes quantifiés, possédé ou adossé à la mémoire unifiée.
#[derive(Clone, Debug)]
enum PackedStorage {
    Owned(Vec<u32>),
    #[cfg(all(target_os = "macos", feature = "metal"))]
    MetalShared {
        buffer: metal::Buffer,
        offset_u32: usize,
        len_u32: usize,
    },
}

impl PackedStorage {
    fn len(&self) -> usize {
        match self {
            Self::Owned(data) => data.len(),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::MetalShared { len_u32, .. } => *len_u32,
        }
    }

    fn data(&self) -> &[u32] {
        match self {
            Self::Owned(data) => data,
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::MetalShared {
                buffer,
                offset_u32,
                len_u32,
            } => metal_shared_u32_slice(buffer, *offset_u32, *len_u32),
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn metal_view(&self) -> Option<(&metal::Buffer, usize)> {
        match self {
            Self::Owned(_) => None,
            Self::MetalShared {
                buffer, offset_u32, ..
            } => Some((
                buffer,
                offset_u32.saturating_mul(std::mem::size_of::<u32>()),
            )),
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn from_metal_shared(buffer: metal::Buffer, offset_u32: usize, len_u32: usize) -> Result<Self> {
        let end_u32 = offset_u32
            .checked_add(len_u32)
            .ok_or_else(|| InferError::Shape("vue packed Metal trop large".to_string()))?;
        let available_u32 = usize::try_from(buffer.length())
            .map_err(|_| InferError::Shape("buffer packed Metal trop grand".to_string()))?
            / std::mem::size_of::<u32>();
        if len_u32 == 0 || end_u32 > available_u32 {
            return Err(InferError::Shape(format!(
                "vue packed Metal [{offset_u32}..{end_u32}] hors buffer de {available_u32} u32"
            )));
        }
        if buffer.contents().is_null() {
            return Err(InferError::Metal(
                "MTLBuffer de poids StorageModeShared sans pointeur CPU".to_string(),
            ));
        }
        Ok(Self::MetalShared {
            buffer,
            offset_u32,
            len_u32,
        })
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[expect(
    unsafe_code,
    reason = "vue CPU en lecture seule d'un MTLBuffer de poids partagé"
)]
fn metal_shared_u32_slice(buffer: &metal::Buffer, offset_u32: usize, len_u32: usize) -> &[u32] {
    let base = buffer.contents().cast::<u32>();
    // SAFETY: le constructeur vérifie `offset_u32 + len_u32 <= buffer.length() / 4`.
    // StorageModeShared rend `contents` visible et aligné côté hôte ; `buffer` est
    // retenu par le stockage pendant toute la durée de la vue. Les MTLBuffer de
    // poids sont immuables après chargement et ne sont jamais écrits par le GPU.
    unsafe { std::slice::from_raw_parts(base.add(offset_u32), len_u32) }
}

/// Stockage des paramètres affines, f32 CPU ou bf16 en mémoire unifiée.
#[derive(Clone, Debug)]
enum AffineStorage {
    F32(Tensor),
    #[cfg(all(target_os = "macos", feature = "metal"))]
    MetalSharedBf16 {
        buffer: metal::Buffer,
        offset_bf16: usize,
        len_bf16: usize,
        shape: Vec<usize>,
    },
}

impl AffineStorage {
    fn from_f32(tensor: Tensor) -> Self {
        Self::F32(tensor)
    }

    fn shape(&self) -> &[usize] {
        match self {
            Self::F32(tensor) => tensor.shape(),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::MetalSharedBf16 { shape, .. } => shape,
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::F32(tensor) => tensor.len(),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::MetalSharedBf16 { len_bf16, .. } => *len_bf16,
        }
    }

    fn value(&self, index: usize) -> f32 {
        match self {
            Self::F32(tensor) => tensor.data()[index],
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::MetalSharedBf16 {
                buffer,
                offset_bf16,
                len_bf16,
                ..
            } => bf16_to_f32(metal_shared_u16_slice(buffer, *offset_bf16, *len_bf16)[index]),
        }
    }

    fn f32_data(&self) -> Cow<'_, [f32]> {
        match self {
            Self::F32(tensor) => Cow::Borrowed(tensor.data()),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::MetalSharedBf16 {
                buffer,
                offset_bf16,
                len_bf16,
                ..
            } => Cow::Owned(
                metal_shared_u16_slice(buffer, *offset_bf16, *len_bf16)
                    .iter()
                    .copied()
                    .map(bf16_to_f32)
                    .collect(),
            ),
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn metal_view(&self) -> Option<(&metal::Buffer, usize)> {
        match self {
            Self::F32(_) => None,
            Self::MetalSharedBf16 {
                buffer,
                offset_bf16,
                ..
            } => Some((
                buffer,
                offset_bf16.saturating_mul(std::mem::size_of::<u16>()),
            )),
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn from_metal_shared_bf16(
        shape: &[usize],
        buffer: metal::Buffer,
        offset_bf16: usize,
        len_bf16: usize,
    ) -> Result<Self> {
        let expected = checked_element_count(shape, "paramètres affines bf16")?;
        if len_bf16 != expected {
            return Err(InferError::Shape(format!(
                "paramètres affines bf16 shape={shape:?}, éléments={len_bf16}"
            )));
        }
        let end_bf16 = offset_bf16
            .checked_add(len_bf16)
            .ok_or_else(|| InferError::Shape("vue affine Metal trop large".to_string()))?;
        let available_bf16 = usize::try_from(buffer.length())
            .map_err(|_| InferError::Shape("buffer affine Metal trop grand".to_string()))?
            / std::mem::size_of::<u16>();
        if len_bf16 == 0 || end_bf16 > available_bf16 {
            return Err(InferError::Shape(format!(
                "vue affine Metal [{offset_bf16}..{end_bf16}] hors buffer de {available_bf16} bf16"
            )));
        }
        if buffer.contents().is_null() {
            return Err(InferError::Metal(
                "MTLBuffer affine StorageModeShared sans pointeur CPU".to_string(),
            ));
        }
        Ok(Self::MetalSharedBf16 {
            buffer,
            offset_bf16,
            len_bf16,
            shape: shape.to_vec(),
        })
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn move_to_metal_shared(&mut self, metal: &crate::MetalExecutor) -> Result<()> {
        let Self::F32(tensor) = self else {
            return Ok(());
        };
        let shape = tensor.shape().to_vec();
        let len = tensor.len();
        let buffer = metal.weight_buffer_from_f32_as_bf16(tensor.data())?;
        *self = Self::from_metal_shared_bf16(&shape, buffer, 0, len)?;
        Ok(())
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn set_metal_view(&mut self, buffer: metal::Buffer, offset_bf16: usize) -> Result<()> {
        let shape = self.shape().to_vec();
        let len = checked_element_count(&shape, "vue affine Metal")?;
        *self = Self::from_metal_shared_bf16(&shape, buffer, offset_bf16, len)?;
        Ok(())
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn release_cpu_data(&mut self) -> usize {
        match self {
            Self::F32(tensor) => tensor
                .release_data()
                .saturating_mul(std::mem::size_of::<f32>()),
            Self::MetalSharedBf16 { .. } => 0,
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn cpu_data_available(&self) -> bool {
        matches!(
            checked_element_count(self.shape(), "paramètres affines"),
            Ok(expected) if self.len() == expected
        )
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn values_are_bf16_exact(&self) -> bool {
        match self {
            Self::F32(tensor) => tensor
                .data()
                .iter()
                .all(|value| bf16_round(*value) == *value),
            Self::MetalSharedBf16 { .. } => true,
        }
    }
}

impl PartialEq for AffineStorage {
    fn eq(&self, other: &Self) -> bool {
        self.shape() == other.shape() && self.f32_data() == other.f32_data()
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[expect(
    unsafe_code,
    reason = "vue CPU en lecture seule d'un MTLBuffer bf16 partagé"
)]
fn metal_shared_u16_slice(buffer: &metal::Buffer, offset_u16: usize, len_u16: usize) -> &[u16] {
    let base = buffer.contents().cast::<u16>();
    // SAFETY: le constructeur vérifie `offset_u16 + len_u16 <= buffer.length() / 2`.
    // StorageModeShared rend `contents` visible et aligné côté hôte ; `buffer` est
    // retenu par le stockage pendant toute la durée de la vue. Les paramètres de
    // poids sont immuables après chargement et ne sont jamais écrits par le GPU.
    unsafe { std::slice::from_raw_parts(base.add(offset_u16), len_u16) }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn bf16_to_f32(value: u16) -> f32 {
    f32::from_bits(u32::from(value) << 16)
}

/// Poids affine packé en `u32`, conservé compact en mémoire.
#[derive(Clone, Debug)]
pub struct AffineQuantizedTensor {
    weight_id: u64,
    shape: Vec<usize>,
    packed_shape: Vec<usize>,
    packed: PackedStorage,
    scales: AffineStorage,
    biases: AffineStorage,
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
            scales.shape(),
            biases.shape(),
            group_size,
            bits,
        )?;
        Ok(Self {
            weight_id: next_weight_id()?,
            shape: vec![params.rows, params.cols],
            packed_shape: packed_shape.to_vec(),
            packed: PackedStorage::Owned(packed),
            scales: AffineStorage::from_f32(scales),
            biases: AffineStorage::from_f32(biases),
            group_size,
            bits,
        })
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[expect(
        clippy::too_many_arguments,
        reason = "constructeur interne: vue Metal et paramètres affine restent explicites"
    )]
    pub(crate) fn new_metal_shared(
        packed_shape: &[usize],
        buffer: metal::Buffer,
        offset_u32: usize,
        len_u32: usize,
        scales: Tensor,
        biases: Tensor,
        group_size: usize,
        bits: usize,
    ) -> Result<Self> {
        let params = affine_params(
            packed_shape,
            len_u32,
            scales.shape(),
            biases.shape(),
            group_size,
            bits,
        )?;
        Ok(Self {
            weight_id: next_weight_id()?,
            shape: vec![params.rows, params.cols],
            packed_shape: packed_shape.to_vec(),
            packed: PackedStorage::from_metal_shared(buffer, offset_u32, len_u32)?,
            scales: AffineStorage::from_f32(scales),
            biases: AffineStorage::from_f32(biases),
            group_size,
            bits,
        })
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[expect(
        clippy::too_many_arguments,
        reason = "constructeur interne: trois vues Metal et paramètres affine restent explicites"
    )]
    pub(crate) fn new_metal_shared_bf16(
        packed_shape: &[usize],
        packed_buffer: metal::Buffer,
        packed_offset_u32: usize,
        packed_len_u32: usize,
        scales_shape: &[usize],
        scales_buffer: metal::Buffer,
        scales_offset_bf16: usize,
        scales_len_bf16: usize,
        biases_shape: &[usize],
        biases_buffer: metal::Buffer,
        biases_offset_bf16: usize,
        biases_len_bf16: usize,
        group_size: usize,
        bits: usize,
    ) -> Result<Self> {
        let params = affine_params(
            packed_shape,
            packed_len_u32,
            scales_shape,
            biases_shape,
            group_size,
            bits,
        )?;
        Ok(Self {
            weight_id: next_weight_id()?,
            shape: vec![params.rows, params.cols],
            packed_shape: packed_shape.to_vec(),
            packed: PackedStorage::from_metal_shared(
                packed_buffer,
                packed_offset_u32,
                packed_len_u32,
            )?,
            scales: AffineStorage::from_metal_shared_bf16(
                scales_shape,
                scales_buffer,
                scales_offset_bf16,
                scales_len_bf16,
            )?,
            biases: AffineStorage::from_metal_shared_bf16(
                biases_shape,
                biases_buffer,
                biases_offset_bf16,
                biases_len_bf16,
            )?,
            group_size,
            bits,
        })
    }

    /// Renvoie l'identité stable du poids pour les caches Metal.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) fn weight_id(&self) -> u64 {
        self.weight_id
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
        self.packed.data()
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) fn packed_metal_view(&self) -> Option<(&metal::Buffer, usize)> {
        self.packed.metal_view()
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) fn move_packed_to_metal_shared(
        &mut self,
        metal: &crate::MetalExecutor,
    ) -> Result<()> {
        if self.packed.metal_view().is_none() {
            let buffer = metal.weight_buffer_from_u32(self.packed.data())?;
            self.packed = PackedStorage::from_metal_shared(buffer, 0, self.packed.len())?;
        }
        self.scales.move_to_metal_shared(metal)?;
        self.biases.move_to_metal_shared(metal)?;
        Ok(())
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) fn set_packed_metal_view(
        &mut self,
        buffer: metal::Buffer,
        offset_u32: usize,
    ) -> Result<()> {
        self.packed = PackedStorage::from_metal_shared(
            buffer,
            offset_u32,
            self.packed_shape.iter().product(),
        )?;
        Ok(())
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) fn scales_metal_view(&self) -> Option<(&metal::Buffer, usize)> {
        self.scales.metal_view()
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) fn biases_metal_view(&self) -> Option<(&metal::Buffer, usize)> {
        self.biases.metal_view()
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) fn set_affine_metal_views(
        &mut self,
        scales_buffer: metal::Buffer,
        scales_offset_bf16: usize,
        biases_buffer: metal::Buffer,
        biases_offset_bf16: usize,
    ) -> Result<()> {
        self.scales
            .set_metal_view(scales_buffer, scales_offset_bf16)?;
        self.biases
            .set_metal_view(biases_buffer, biases_offset_bf16)?;
        Ok(())
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) fn scales_shape(&self) -> &[usize] {
        self.scales.shape()
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) fn biases_shape(&self) -> &[usize] {
        self.biases.shape()
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) fn scales_f32(&self) -> Cow<'_, [f32]> {
        self.scales.f32_data()
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) fn biases_f32(&self) -> Cow<'_, [f32]> {
        self.biases.f32_data()
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) fn group_size(&self) -> usize {
        self.group_size
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) fn bits(&self) -> usize {
        self.bits
    }

    /// Libère les payloads CPU après leur copie dans les buffers Metal résidents.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) fn release_cpu_data(&mut self) -> usize {
        // MetalShared vit dans le buffer unifié : rien à libérer côté CPU.
        // Le chemin historique Owned conserve la sémantique de drop antérieure.
        let packed = if matches!(self.packed, PackedStorage::MetalShared { .. }) {
            0
        } else {
            match std::mem::replace(&mut self.packed, PackedStorage::Owned(Vec::new())) {
                PackedStorage::Owned(data) => data.len().saturating_mul(std::mem::size_of::<u32>()),
                PackedStorage::MetalShared { .. } => 0,
            }
        };
        let scales = self.scales.release_cpu_data();
        let biases = self.biases.release_cpu_data();
        packed.saturating_add(scales).saturating_add(biases)
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) fn cpu_data_available(&self) -> bool {
        self.packed.len() == self.packed_len()
            && self.scales.cpu_data_available()
            && self.biases.cpu_data_available()
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) fn metal_embedding_byte_exact(&self) -> bool {
        self.bits > 0
            && 32 % self.bits == 0
            && self.scales.values_are_bf16_exact()
            && self.biases.values_are_bf16_exact()
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) fn packed_len(&self) -> usize {
        self.packed_shape.iter().product()
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) fn scales_len(&self) -> usize {
        self.scales.shape().iter().product()
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) fn biases_len(&self) -> usize {
        self.biases.shape().iter().product()
    }

    /// Multiplie `input` par la transposée du poids logique dense.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si les dimensions de `input` sont incompatibles.
    pub fn matmul_rhs_t(&self, input: &Tensor) -> Result<Tensor> {
        self.ensure_cpu_data()?;
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
        self.ensure_cpu_data()?;
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
        self.ensure_cpu_data()?;
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

    fn ensure_cpu_data(&self) -> Result<()> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if !self.cpu_data_available() {
            return Err(InferError::Config(format!(
                "payload CPU du poids quantifié {} déjà libéré",
                self.weight_id
            )));
        }
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        if self.packed.len() != self.packed_shape.iter().product::<usize>()
            || self.scales.len() != self.scales.shape().iter().product::<usize>()
            || self.biases.len() != self.biases.shape().iter().product::<usize>()
        {
            return Err(InferError::Config(
                "payload CPU du poids quantifié incomplet".to_string(),
            ));
        }
        Ok(())
    }

    fn value(&self, row: usize, col: usize) -> f32 {
        let quantized = self.bitpacked_value(row, col) as f32;
        let groups = self.shape[1] / self.group_size;
        let group = col / self.group_size;
        let affine_index = row * groups + group;
        quantized * self.scales.value(affine_index) + self.biases.value(affine_index)
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
            let scale = self.scales.value(affine_index);
            let bias = self.biases.value(affine_index);
            for (col, &input) in input_row
                .iter()
                .enumerate()
                .take((group + 1) * self.group_size)
                .skip(group * self.group_size)
            {
                let quantized = self.bitpacked_value(row, col) as f32;
                acc += input * (quantized * scale + bias);
            }
        }
        acc
    }

    fn bitpacked_value(&self, row: usize, col: usize) -> u32 {
        let packed = self.packed.data();
        let packed_cols = self.packed_shape[1];
        let bit_offset = col * self.bits;
        let word_col = bit_offset / 32;
        let shift = bit_offset % 32;
        let row_start = row * packed_cols;
        let mask = (1_u32 << self.bits) - 1;
        let low = packed[row_start + word_col] >> shift;
        if shift + self.bits <= 32 {
            return low & mask;
        }
        let high_bits = shift + self.bits - 32;
        let high_mask = (1_u32 << high_bits) - 1;
        let high = packed.get(row_start + word_col + 1).copied().unwrap_or(0) & high_mask;
        (low | (high << (32 - shift))) & mask
    }

    fn dot_row_u4(&self, input_row: &[f32], row: usize) -> f32 {
        let packed_data = self.packed.data();
        let packed_cols = self.packed_shape[1];
        let groups = self.shape[1] / self.group_size;
        let words_per_group = self.group_size / 8;
        let mut acc = 0.0_f32;
        for group in 0..groups {
            let affine_index = row * groups + group;
            let scale = self.scales.value(affine_index);
            let bias = self.biases.value(affine_index);
            let first_word = group * words_per_group;
            for word_offset in 0..words_per_group {
                let word_col = first_word + word_offset;
                let packed = packed_data[row * packed_cols + word_col];
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
        let packed_data = self.packed.data();
        let packed_cols = self.packed_shape[1];
        let groups = self.shape[1] / self.group_size;
        let words_per_group = self.group_size / 4;
        let mut acc = 0.0_f32;
        for group in 0..groups {
            let affine_index = row * groups + group;
            let scale = self.scales.value(affine_index);
            let bias = self.biases.value(affine_index);
            let first_word = group * words_per_group;
            for word_offset in 0..words_per_group {
                let word_col = first_word + word_offset;
                let packed = packed_data[row * packed_cols + word_col];
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

impl PartialEq for AffineQuantizedTensor {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape
            && self.packed_shape == other.packed_shape
            && self.packed.data() == other.packed.data()
            && self.scales == other.scales
            && self.biases == other.biases
            && self.group_size == other.group_size
            && self.bits == other.bits
    }
}

fn next_weight_id() -> Result<u64> {
    NEXT_WEIGHT_ID
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
        .map_err(|_| InferError::Config("espace des identités de poids épuisé".to_string()))
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn bf16_round(value: f32) -> f32 {
    let bits = value.to_bits();
    let rounding = 0x7fff_u32 + ((bits >> 16) & 1);
    f32::from_bits(bits.wrapping_add(rounding) & 0xffff_0000)
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
    scales_shape: &[usize],
    biases_shape: &[usize],
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
    if scales_shape != [*rows, groups] || biases_shape != [*rows, groups] {
        return Err(InferError::Dimension(format!(
            "scales/biases attendus [{rows},{groups}], reçu scales={:?}, biases={:?}",
            scales_shape, biases_shape
        )));
    }

    Ok(AffineParams { rows: *rows, cols })
}

fn checked_element_count(shape: &[usize], label: &str) -> Result<usize> {
    shape.iter().try_fold(1_usize, |acc, dim| {
        acc.checked_mul(*dim)
            .ok_or_else(|| InferError::Shape(format!("shape trop grande pour {label}")))
    })
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

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn stable_id_survives_clone_and_cpu_payload_release() {
        let scales = Tensor::from_vec(vec![1, 1], vec![0.5]).expect("invariant: scales valides");
        let biases = Tensor::from_vec(vec![1, 1], vec![0.0]).expect("invariant: biases valides");
        let mut weight =
            AffineQuantizedTensor::new(&[1, 1], vec![0x7654_3210], scales, biases, 8, 4)
                .expect("invariant: poids affine valide");
        let clone = weight.clone();

        assert_eq!(weight.weight_id(), clone.weight_id());
        assert_eq!(weight.release_cpu_data(), 12);
        assert_eq!(weight.packed_len(), 1);
        assert_eq!(weight.scales_len(), 1);
        assert_eq!(weight.biases_len(), 1);
        assert!(weight.row(0).is_err());
    }

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
    fn dequantizes_mlx_affine_u2_byte_exact_row() {
        // MLX écrit seize codes u2 LSB-first dans un u32 :
        // [0,1,2,3] répété quatre fois devient 0xe4e4_e4e4.
        let packed = [0xe4e4_e4e4; 4];
        let scales =
            Tensor::from_vec(vec![1, 1], vec![0.5]).expect("invariant: scale u2 MLX valide");
        let biases =
            Tensor::from_vec(vec![1, 1], vec![-1.0]).expect("invariant: bias u2 MLX valide");

        let dense = dequantize_affine_u32(&[1, 4], &packed, &scales, &biases, 64, 2)
            .expect("invariant: déquantification affine u2 MLX valide");
        let expected = (0..64)
            .map(|index| ((index % 4) as f32) * 0.5 - 1.0)
            .collect::<Vec<_>>();

        assert_eq!(dense.data(), expected);
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
    fn dequantizes_mlx_affine_u3_byte_exact_row() {
        // MLX écrit huit codes u3 LSB-first dans trois octets consécutifs :
        // [0,1,2,3,4,5,6,7] devient [0x88, 0xc6, 0xfa].
        let packed = [
            0x88fa_c688,
            0xc688_fac6,
            0xfac6_88fa,
            0x88fa_c688,
            0xc688_fac6,
            0xfac6_88fa,
        ];
        let scales =
            Tensor::from_vec(vec![1, 1], vec![0.5]).expect("invariant: scale u3 MLX valide");
        let biases =
            Tensor::from_vec(vec![1, 1], vec![-1.0]).expect("invariant: bias u3 MLX valide");

        let dense = dequantize_affine_u32(&[1, 6], &packed, &scales, &biases, 64, 3)
            .expect("invariant: déquantification affine u3 MLX valide");
        let expected = (0..64)
            .map(|index| ((index % 8) as f32) * 0.5 - 1.0)
            .collect::<Vec<_>>();

        assert_eq!(dense.data(), expected);
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
                        quantized * compact.scales.value(affine) + compact.biases.value(affine)
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
