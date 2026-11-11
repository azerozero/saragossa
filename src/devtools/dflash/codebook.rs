//! Codebooks du sélecteur : stockage compact, arithmétique f32 inchangée.

use crate::{InferError, Result, Tensor};
use std::borrow::Cow;

#[derive(Debug, Clone)]
pub struct DFlashCodebook {
    shape: Vec<usize>,
    values: CodebookValues,
}

#[derive(Debug, Clone)]
enum CodebookValues {
    F32(Tensor),
    Bf16(Vec<u16>),
}

impl From<Tensor> for DFlashCodebook {
    fn from(tensor: Tensor) -> Self {
        Self {
            shape: tensor.shape().to_vec(),
            values: CodebookValues::F32(tensor),
        }
    }
}

impl DFlashCodebook {
    pub(super) fn bf16(shape: Vec<usize>, values: Vec<u16>) -> Self {
        Self {
            shape,
            values: CodebookValues::Bf16(values),
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn is_bf16(&self) -> bool {
        matches!(self.values, CodebookValues::Bf16(_))
    }

    /// Restitue uniquement la ligne consultée, sans arrondi supplémentaire.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le codebook n'est pas une matrice ou si le token
    /// sort du vocabulaire.
    pub fn row_slice(&self, token: usize) -> Result<Cow<'_, [f32]>> {
        match &self.values {
            CodebookValues::F32(tensor) => tensor.row_slice(token).map(Cow::Borrowed),
            CodebookValues::Bf16(values) => {
                let [rows, cols] = self.shape.as_slice() else {
                    return Err(InferError::Dimension(
                        "codebook DFlash attendu rang 2".to_string(),
                    ));
                };
                if token >= *rows {
                    return Err(InferError::Dimension(format!(
                        "codebook DFlash token {token} hors vocabulaire {rows}"
                    )));
                }
                Ok(Cow::Owned(
                    values[token * cols..(token + 1) * cols]
                        .iter()
                        .map(|&bits| f32::from_bits(u32::from(bits) << 16))
                        .collect(),
                ))
            }
        }
    }
}
