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
