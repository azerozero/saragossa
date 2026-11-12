//! Poids f32 partagés conservant les kernels et réductions du stockage dense.

use crate::{InferError, Result, Tensor};

/// Matrice f32 immuable adossée à un buffer Metal partagé.
#[derive(Clone, Debug)]
pub struct DenseSharedTensor {
    shape: [usize; 2],
    buffer: metal::Buffer,
    len: usize,
}

impl PartialEq for DenseSharedTensor {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.data() == other.data()
    }
}

impl DenseSharedTensor {
    pub(crate) fn new(shape: &[usize], buffer: metal::Buffer) -> Result<Self> {
        let &[rows, cols] = shape else {
            return Err(InferError::Shape(format!(
                "matrice f32 partagée invalide: {shape:?}"
            )));
        };
        let len = rows
            .checked_mul(cols)
            .filter(|&n| n > 0)
            .ok_or_else(|| InferError::Shape("matrice f32 partagée vide ou trop grande".into()))?;
        let bytes = len
            .checked_mul(size_of::<f32>())
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or_else(|| InferError::Shape("vue f32 partagée trop grande".into()))?;
        if Some(bytes) != usize::try_from(buffer.length()).ok()
            || buffer.storage_mode() != metal::MTLStorageMode::Shared
            || buffer.contents().is_null()
            || buffer.contents().addr() % align_of::<f32>() != 0
        {
            return Err(InferError::Shape("buffer f32 partagé incompatible".into()));
        }
        Ok(Self {
            shape: [rows, cols],
            buffer,
            len,
        })
    }

    pub(crate) fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub(crate) fn buffer(&self) -> &metal::Buffer {
        &self.buffer
    }

    #[expect(unsafe_code, reason = "vue immuable du stockage partagé possédé")]
    pub(crate) fn data(&self) -> &[f32] {
        // SAFETY: new valide taille, alignement et stockage partagé. Le buffer
        // initialisé reste vivant et immuable pendant toute la durée de la vue.
        unsafe { std::slice::from_raw_parts(self.buffer.contents().cast(), self.len) }
    }

    #[cfg(test)]
    pub(crate) fn to_tensor(&self) -> Result<Tensor> {
        Tensor::from_vec(self.shape.to_vec(), self.data().to_vec())
    }

    pub(super) fn forward_metal(
        &self,
        input: &Tensor,
        metal: &crate::MetalExecutor,
        tiled: bool,
    ) -> Result<Tensor> {
        metal.matmul_rhs_t_shared_f32(input, self, tiled)
    }
}
