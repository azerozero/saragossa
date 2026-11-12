//! Fréquences RoPE communes aux références CPU et aux kernels Metal.

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
