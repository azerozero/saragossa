//! Normalisations RMS utilisées par les couches du décodeur.

use crate::{InferError, Result, Tensor};
use rayon::prelude::*;

/// Applique une LayerNorm standard ligne par ligne.
///
/// Contrairement a RMSNorm, Whisper soustrait la moyenne avant de normaliser par
/// la variance. `weight` et `bias` sont diffusés sur la dernière dimension.
///
/// # Errors
///
/// Renvoie une erreur si les formes d'entrée, de poids ou de biais sont
/// incompatibles.
pub fn layer_norm(x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f32) -> Result<Tensor> {
    let (rows, dim) = x.as_matrix()?;
    let weight_data = norm_vector(weight, dim, "LayerNorm weight")?;
    let bias_data = norm_vector(bias, dim, "LayerNorm bias")?;
    // Parallélisé par ligne (chaque ligne est indépendante, même ordre de calcul
    // intra-ligne ⇒ byte-identique). L'encodeur Whisper normalise [1500,1280] ×64.
    let input = x.data();
    let mut out = vec![0.0_f32; rows * dim];
    out.par_chunks_mut(dim)
        .enumerate()
        .for_each(|(row, out_row)| {
            let xs = &input[row * dim..(row + 1) * dim];
            let mean = xs.iter().sum::<f32>() / dim as f32;
            let variance = xs
                .iter()
                .map(|value| {
                    let centered = value - mean;
                    centered * centered
                })
                .sum::<f32>()
                / dim as f32;
            let inv_std = 1.0 / (variance + eps).sqrt();
            for (col, slot) in out_row.iter_mut().enumerate() {
                *slot = (xs[col] - mean) * inv_std * weight_data[col] + bias_data[col];
            }
        });
    Tensor::from_vec(vec![rows, dim], out)
}

/// Applique une normalisation RMS ligne par ligne.
///
/// # Errors
///
/// Renvoie une erreur si les formes d'entrée ou de poids sont incompatibles.
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let (rows, dim) = x.as_matrix()?;
    let weight_data = norm_vector(weight, dim, "RMSNorm weight")?;
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

fn norm_vector<'a>(tensor: &'a Tensor, dim: usize, label: &str) -> Result<&'a [f32]> {
    match tensor.shape() {
        [n] if *n == dim => Ok(tensor.data()),
        [1, n] if *n == dim => Ok(tensor.data()),
        _ => Err(InferError::Dimension(format!(
            "{label} attendu [{dim}] ou [1,{dim}], reçu {:?}",
            tensor.shape()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_norm_matches_manual_values() {
        let x =
            Tensor::from_vec(vec![2, 2], vec![1.0, 3.0, 2.0, 6.0]).expect("invariant: x valide");
        let weight = Tensor::from_vec(vec![2], vec![1.0, 2.0]).expect("invariant: w valide");
        let bias = Tensor::from_vec(vec![2], vec![0.5, -0.5]).expect("invariant: b valide");
        let out = layer_norm(&x, &weight, &bias, 0.0).expect("invariant: norm valide");
        assert_eq!(out.shape(), &[2, 2]);
        assert!((out.data()[0] - -0.5).abs() < 1.0e-6);
        assert!((out.data()[1] - 1.5).abs() < 1.0e-6);
        assert!((out.data()[2] - -0.5).abs() < 1.0e-6);
        assert!((out.data()[3] - 1.5).abs() < 1.0e-6);
    }

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
