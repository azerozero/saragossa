//! Normalisations RMS utilisées par les couches du décodeur.

use crate::{InferError, Result, Tensor};

/// Applique une normalisation RMS ligne par ligne.
///
/// # Errors
///
/// Renvoie une erreur si les formes d'entrée ou de poids sont incompatibles.
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let (rows, dim) = x.as_matrix()?;
    let weight_data = match weight.shape() {
        [n] if *n == dim => weight.data(),
        [1, n] if *n == dim => weight.data(),
        _ => {
            return Err(InferError::Dimension(format!(
                "RMSNorm weight attendu [{dim}] ou [1,{dim}], reçu {:?}",
                weight.shape()
            )))
        }
    };
    let mut out = Vec::with_capacity(x.len());
    for row in 0..rows {
        let xs = x.row_slice(row)?;
        let mean_square = xs.iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let inv_rms = 1.0 / (mean_square + eps).sqrt();
        for col in 0..dim {
            out.push(xs[col] * inv_rms * weight_data[col]);
        }
    }
    Tensor::from_vec(vec![rows, dim], out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_matches_manual_value() {
        let x = Tensor::from_vec(vec![1, 2], vec![3.0, 4.0]).expect("invariant: x valide");
        let weight = Tensor::from_vec(vec![2], vec![1.0, 2.0]).expect("invariant: w valide");
        let out = rms_norm(&x, &weight, 0.0).expect("invariant: norm valide");
        let scale = 1.0 / ((25.0_f32 / 2.0).sqrt());
        assert!((out.data()[0] - 3.0 * scale).abs() < 1.0e-6);
        assert!((out.data()[1] - 4.0 * scale * 2.0).abs() < 1.0e-6);
    }
}
