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

/// Paramètres du `rope_scaling` de type `llama3` (Llama 3.1, 3.2, 3.3).
///
/// Ils ne changent que la table des fréquences : les positions restent brutes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Llama3RopeScaling {
    /// Facteur d'étirement des basses fréquences.
    pub factor: f32,
    /// Facteur fixant la longueur d'onde basse (`contexte / low`).
    pub low_freq_factor: f32,
    /// Facteur fixant la longueur d'onde haute (`contexte / high`).
    pub high_freq_factor: f32,
    /// Contexte d'entraînement d'origine.
    pub original_max_position_embeddings: f32,
}

impl Llama3RopeScaling {
    /// Clé exacte des caches de fréquences et de rotations.
    pub(crate) fn key(self) -> [u32; 4] {
        [
            self.factor.to_bits(),
            self.low_freq_factor.to_bits(),
            self.high_freq_factor.to_bits(),
            self.original_max_position_embeddings.to_bits(),
        ]
    }
}

/// Table des fréquences d'une couche, avec le scaling `llama3` éventuel.
///
/// Sans scaling, renvoie exactement [`inv_freq`]. Avec scaling, suit en f32
/// l'ordre des opérations de `_compute_llama3_parameters` de transformers :
/// les longueurs d'onde courtes restent, les longues sont divisées par
/// `factor`, l'entre-deux est interpolé.
pub(crate) fn inv_freq_scaled(
    frequency_dim: usize,
    theta: f32,
    scaling: Option<Llama3RopeScaling>,
) -> Vec<f32> {
    let Some(scaling) = scaling else {
        return inv_freq(frequency_dim, theta);
    };
    // NOTE: `powf` f32 de la libm s'écarte de 1 à 2 ulp du `pow` de torch
    // (paires 3 et 16 en d64) ; la puissance f64 arrondie en f32 retrouve ses
    // bits. La table par défaut garde `powf` pour l'identité des autres modèles.
    let base = (0..frequency_dim / 2).map(|pair| {
        let exponent = (2 * pair) as f32 / frequency_dim as f32;
        1.0 / f64::from(theta).powf(f64::from(exponent)) as f32
    });
    let old_context = scaling.original_max_position_embeddings;
    let low_freq_wavelen = old_context / scaling.low_freq_factor;
    let high_freq_wavelen = old_context / scaling.high_freq_factor;
    // NOTE: transformers convertit le scalaire Python 2π en f32 avant la division.
    let two_pi = (2.0 * std::f64::consts::PI) as f32;
    base.map(|frequency| {
        let wavelen = two_pi / frequency;
        let llama = if wavelen > low_freq_wavelen {
            frequency / scaling.factor
        } else {
            frequency
        };
        if wavelen < high_freq_wavelen || wavelen > low_freq_wavelen {
            return llama;
        }
        let smooth = (old_context / wavelen - scaling.low_freq_factor)
            / (scaling.high_freq_factor - scaling.low_freq_factor);
        (1.0 - smooth) * llama / scaling.factor + smooth * llama
    })
    .collect()
}

#[cfg(test)]
mod tests {
    use super::{inv_freq, inv_freq_scaled, Llama3RopeScaling};

    fn llama3(factor: f32) -> Llama3RopeScaling {
        Llama3RopeScaling {
            factor,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_max_position_embeddings: 8192.0,
        }
    }

    /// Bits f32 de `ROPE_INIT_FUNCTIONS["llama3"]` de transformers 5.22.2
    /// (torch 2.15.0, CPU), θ = 500 000 : Llama 3.2 1B (d64, ×32) puis
    /// Llama 3.2 3B / 3.1 8B (d128, ×8). Script : tools/goldens/rope_llama3.py.
    const TRANSFORMERS_D64_X32: [u32; 32] = [
        0x3f800000, 0x3f29e1c6, 0x3ee177bc, 0x3e959ee3, 0x3e4693b0, 0x3e03c6a0, 0x3daee4ad,
        0x3d681e67, 0x3d1a08c8, 0x3ccc6f49, 0x3c87a9c3, 0x3c340d6d, 0x3beef74f, 0x3b9e9402,
        0x3b527720, 0x3aa9279b, 0x39e13620, 0x38cb98f7, 0x37a3418d, 0x3758ac81, 0x370fc8f8,
        0x36bed4f4, 0x367d45c3, 0x3628126b, 0x35df10c4, 0x359406cb, 0x35447610, 0x35025f34,
        0x34ad07a7, 0x3465a54d, 0x341864a7, 0x33ca41b0,
    ];
    const TRANSFORMERS_D128_X8: [u32; 64] = [
        0x3f800000, 0x3f508ac1, 0x3f29e1c6, 0x3f0a6384, 0x3ee177bc, 0x3eb7ab7d, 0x3e959ee3,
        0x3e73c461, 0x3e4693b0, 0x3e21c3a0, 0x3e03c6a0, 0x3dd6b19c, 0x3daee4ad, 0x3d8e7898,
        0x3d681e67, 0x3d3d1684, 0x3d1a08c8, 0x3cfaf53f, 0x3ccc6f49, 0x3ca68939, 0x3c87a9c3,
        0x3c5d06ec, 0x3c340d6d, 0x3c12ac7f, 0x3beef74f, 0x3bc2aa76, 0x3b9e9402, 0x3b812e35,
        0x3b527720, 0x3b0dfd06, 0x3ab3d11d, 0x3a60979e, 0x3a0995d2, 0x39a3f108, 0x393b2dd2,
        0x38c86886, 0x38a3418d, 0x3884fdbf, 0x3858ac81, 0x38308199, 0x380fc8f8, 0x37ea426f,
        0x37bed4f4, 0x379b7475, 0x377d45c3, 0x374e51f5, 0x3728126b, 0x3708ea0f, 0x36df10c4,
        0x36b5b687, 0x369406cb, 0x36712b80, 0x36447610, 0x36200a69, 0x36025f34, 0x35d46808,
        0x35ad07a7, 0x358cf400, 0x3565a54d, 0x353b12c7, 0x351864a7, 0x34f848c2, 0x34ca41b0,
        0x34a4c2ff,
    ];

    #[test]
    fn llama3_table_matches_transformers_bits() {
        for (dim, factor, expected) in [
            (64, 32.0, &TRANSFORMERS_D64_X32[..]),
            (128, 8.0, &TRANSFORMERS_D128_X8[..]),
        ] {
            let table = inv_freq_scaled(dim, 500_000.0, Some(llama3(factor)));
            let bits: Vec<u32> = table.iter().map(|f| f.to_bits()).collect();
            assert_eq!(bits, expected, "d{dim} x{factor}");
        }
    }

    #[test]
    fn unscaled_table_is_the_default_table() {
        for dim in [0, 64, 128, 256] {
            assert_eq!(inv_freq_scaled(dim, 1.0e6, None), inv_freq(dim, 1.0e6));
        }
    }

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
