//! Couches d'embedding et extraction de lignes de tokens.

use crate::{AffineQuantizedTensor, InferError, Result, Tensor};

#[derive(Clone, Debug, PartialEq)]
/// Décrit le stockage des poids d'embedding.
pub enum EmbeddingWeight {
    /// Stocke une table dense f32.
    Dense(Tensor),
    /// Stocke une table affine quantifiée.
    AffineQuantized(AffineQuantizedTensor),
}

impl EmbeddingWeight {
    #[must_use]
    /// Renvoie la forme de la table d'embedding.
    pub fn shape(&self) -> &[usize] {
        match self {
            Self::Dense(tensor) => tensor.shape(),
            Self::AffineQuantized(tensor) => tensor.shape(),
        }
    }
}

/// Extrait les lignes d'embedding d'une table dense.
///
/// # Errors
///
/// Renvoie une erreur si la table ou un token est invalide.
pub fn embed_tokens(table: &Tensor, token_ids: &[usize]) -> Result<Tensor> {
    let (vocab, dim) = table.as_matrix()?;
    if token_ids.is_empty() {
        return Err(InferError::Dimension("séquence de tokens vide".to_string()));
    }
    let mut out = Vec::with_capacity(token_ids.len() * dim);
    for token_id in token_ids {
        if *token_id >= vocab {
            return Err(InferError::Dimension(format!(
                "token id {token_id} hors vocab {vocab}"
            )));
        }
        out.extend_from_slice(table.row_slice(*token_id)?);
    }
    Tensor::from_vec(vec![token_ids.len(), dim], out)
}

/// Extrait les lignes d'embedding depuis un stockage quelconque.
///
/// # Errors
///
/// Renvoie une erreur si la table ou un token est invalide.
pub fn embed_weight_tokens(table: &EmbeddingWeight, token_ids: &[usize]) -> Result<Tensor> {
    match table {
        EmbeddingWeight::Dense(table) => embed_tokens(table, token_ids),
        EmbeddingWeight::AffineQuantized(table) => embed_quantized_tokens(table, token_ids),
    }
}

fn embed_quantized_tokens(table: &AffineQuantizedTensor, token_ids: &[usize]) -> Result<Tensor> {
    let [vocab, dim] = table.shape() else {
        return Err(InferError::Dimension(format!(
            "embedding quantifié attendu rang 2, reçu {:?}",
            table.shape()
        )));
    };
    if token_ids.is_empty() {
        return Err(InferError::Dimension("séquence de tokens vide".to_string()));
    }
    let mut out = Vec::with_capacity(token_ids.len() * dim);
    for token_id in token_ids {
        if *token_id >= *vocab {
            return Err(InferError::Dimension(format!(
                "token id {token_id} hors vocab {vocab}"
            )));
        }
        out.extend(table.row(*token_id)?);
    }
    Tensor::from_vec(vec![token_ids.len(), *dim], out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gathers_embedding_rows() {
        let table = Tensor::from_vec(vec![3, 2], vec![1.0, 0.0, 0.0, 1.0, 2.0, 2.0])
            .expect("invariant: table valide");
        let out = embed_tokens(&table, &[2, 0]).expect("invariant: ids valides");
        assert_eq!(out.shape(), &[2, 2]);
        assert_eq!(out.data(), &[2.0, 2.0, 1.0, 0.0]);
    }

    #[test]
    fn gathers_quantized_embedding_rows() {
        let packed = vec![0x0000_00ff, 0x0000_ff00, 0xffff_ffff];
        let scales =
            Tensor::from_vec(vec![3, 1], vec![1.0 / 255.0; 3]).expect("invariant: scales valides");
        let biases = Tensor::from_vec(vec![3, 1], vec![0.0; 3]).expect("invariant: biases valides");
        let table = EmbeddingWeight::AffineQuantized(
            AffineQuantizedTensor::new(&[3, 1], packed, scales, biases, 4, 8)
                .expect("invariant: embedding compact valide"),
        );

        let out = embed_weight_tokens(&table, &[2, 0]).expect("invariant: ids valides");

        assert_eq!(out.shape(), &[2, 4]);
        assert_eq!(out.data(), &[1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]);
    }
}
