//! Fréquences RoPE communes aux références CPU et aux kernels Metal.

/// Sélection HF entrelacée [11, 11, 10] : la dernière paire (31) est H.
/// L'entrelacement porte sur les axes, la rotation reste rotate-half.
pub(crate) fn mrope_axis(pair: usize) -> usize {
    match pair % 3 {
        1 if pair < 33 => 1,
        2 if pair < 30 => 2,
        _ => 0,
    }
}

pub(crate) fn apply_mrope(
    input: &crate::Tensor,
    heads: usize,
    head_dim: usize,
    positions: &[[usize; 3]],
) -> crate::Result<crate::Tensor> {
    let (rows, width) = input.as_matrix()?;
    if head_dim != 256 || heads.checked_mul(head_dim) != Some(width) || rows != positions.len() {
        return Err(crate::InferError::Dimension(
            "M-RoPE: géométrie Qwen incompatible".into(),
        ));
    }
    let frequencies = inv_freq(64, 10_000_000.0);
    let mut output = input.data().to_vec();
    for (row, position) in positions.iter().enumerate() {
        for head in 0..heads {
            let base = row * width + head * head_dim;
            for (pair, frequency) in frequencies.iter().enumerate() {
                let (sin, cos) = (position[mrope_axis(pair)] as f32 * frequency).sin_cos();
                let a = input.data()[base + pair];
                let b = input.data()[base + pair + 32];
                output[base + pair] = a * cos - b * sin;
                output[base + pair + 32] = a * sin + b * cos;
            }
        }
    }
    crate::Tensor::from_vec(input.shape().to_vec(), output)
}

/// Calcule en f32 `1 / θ^(2p/frequency_dim)`, comme transformers.
///
/// Le dénominateur peut dépasser la largeur rotée (Gemma global) : les
/// appelants ne consomment alors que le début du vecteur. Une dimension nulle
/// produit un vecteur vide pour les couches NoPE.
pub(crate) fn inv_freq(frequency_dim: usize, theta: f32) -> Vec<f32> {
    (0..frequency_dim / 2)
        .map(|pair| 1.0 / theta.powf((2 * pair) as f32 / frequency_dim as f32))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::inv_freq;

    #[test]
    fn frequencies_cover_full_partial_and_no_rotation() {
        for fd in [0, 64, 128, 256] {
            for theta in [10_000.0_f32, 1_000_000.0] {
                let frequencies = inv_freq(fd, theta);
                assert_eq!(frequencies.len(), fd / 2);
                assert!(frequencies.iter().all(|f| f.is_finite() && *f > 0.0));
                assert!(frequencies.windows(2).all(|pair| pair[0] > pair[1]));
            }
        }
        assert_eq!(inv_freq(4, 10_000.0), [1.0, 0.01]);
        assert_eq!(inv_freq(8, 10_000.0), [1.0, 0.1, 0.01, 0.001]);
        assert_ne!(inv_freq(128, 10_000.0)[1], inv_freq(256, 10_000.0)[1]);
    }
}

/// Oracle de test indépendant de la table de production, d'après le contrat
/// MLX ProportionalRoPE : rotation sur `rotation_dim`, fréquences infinies
/// (angle nul) après les `rotated_dims / 2` premières paires. Pour Qwen,
/// `rotation_dim = rotated_dims` et la queue est concaténée inchangée.
#[cfg(test)]
pub(crate) fn reference_rotate_half(
    input: &[f32],
    rotated_dims: usize,
    rotation_dim: usize,
    theta: f32,
    position: usize,
) -> Vec<f32> {
    let mut out = input.to_vec();
    for i in 0..rotated_dims / 2 {
        let frequency = 1.0 / theta.powf((2 * i) as f32 / rotation_dim as f32);
        let angle = position as f32 * frequency;
        let (sin, cos) = angle.sin_cos();
        let j = i + rotation_dim / 2;
        out[i] = input[i] * cos - input[j] * sin;
        out[j] = input[i] * sin + input[j] * cos;
    }
    out
}

#[cfg(test)]
#[test]
fn proportional_oracle_matches_mlx_basis_and_inactive_coordinates() {
    let mut basis = vec![0.0; 512];
    basis[0] = 1.0;
    let out = reference_rotate_half(&basis, 128, 512, 1_000_000.0, 1);
    assert_eq!(out[0], 1.0_f32.cos());
    assert_eq!(out[256], 1.0_f32.sin());
    assert!(out
        .iter()
        .enumerate()
        .all(|(i, &x)| i == 0 || i == 256 || x == 0.0));
    let input: Vec<_> = (0..512).map(|i| i as f32 / 32.0 - 8.0).collect();
    let out = reference_rotate_half(&input, 128, 512, 1_000_000.0, 1);
    assert_eq!(&out[64..256], &input[64..256]);
    assert_eq!(&out[320..512], &input[320..512]);
    // Calcul f64 direct : vérifie aussi le dénominateur, loin de la paire 0.
    for i in [1, 31, 63] {
        let angle = 10_f64.powf(6.0 * (-2.0 * i as f64 / 512.0));
        let expected = f64::from(input[i]) * angle.cos() - f64::from(input[i + 256]) * angle.sin();
        assert!((f64::from(out[i]) - expected).abs() < 2.0e-6);
    }
}

#[cfg(test)]
#[test]
fn mrope_interleaves_axes_and_leaves_unrotated_tail_intact() -> crate::Result<()> {
    let input = crate::Tensor::from_vec(
        vec![1, 256],
        (0..256).map(|i| (i as f32 + 1.0) / 256.0).collect(),
    )?;
    let output = apply_mrope(&input, 1, 256, &[[3, 5, 7]])?;
    assert_eq!(&output.data()[64..], &input.data()[64..]);
    for (pair, position) in [
        (0, 3.0),
        (1, 5.0),
        (2, 7.0),
        (28, 5.0),
        (29, 7.0),
        (30, 3.0),
        (31, 5.0),
    ] {
        let angle = position * 10_000_000_f64.powf(-2.0 * pair as f64 / 64.0);
        let expected = f64::from(input.data()[pair]) * angle.cos()
            - f64::from(input.data()[pair + 32]) * angle.sin();
        assert!((f64::from(output.data()[pair]) - expected).abs() < 2e-7);
    }
    Ok(())
}
