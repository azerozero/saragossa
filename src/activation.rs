//! Fonctions d'activation élémentaires pour les tenseurs CPU.

use crate::Tensor;

/// Applique GELU élément par élément.
///
/// Whisper/HF utilisent la variante exacte `0.5*x*(1+erf(x/sqrt(2)))`.
/// L'approximation de `erf` ci-dessous évite une dépendance native et reste
/// suffisamment précise pour les oracles CPU du port.
pub fn gelu(x: &Tensor) -> Tensor {
    x.par_map(gelu_scalar)
}

pub fn silu(x: &Tensor) -> Tensor {
    x.map(|v| v / (1.0 + (-v).exp()))
}

fn gelu_scalar(x: f32) -> f32 {
    0.5 * x * (1.0 + erf_approx(x * std::f32::consts::FRAC_1_SQRT_2))
}

fn erf_approx(x: f32) -> f32 {
    // Abramowitz-Stegun 7.1.26, erreur max ~1.5e-7 en f64; ici bornée par f32.
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_4 * t - 1.453_152_1) * t) + 1.421_413_8) * t - 0.284_496_72) * t
            + 0.254_829_6)
            * t
            * (-x * x).exp();
    sign * y
}

/// Constante `√(2/π)` de l'approximation tanh du GeLU.
const GELU_TANH_COEFF: f32 = 0.797_884_6;

/// Applique le GeLU approché par tanh (`gelu_pytorch_tanh`, activation Gemma).
#[must_use]
pub fn gelu_tanh(x: &Tensor) -> Tensor {
    x.map(|v| {
        let inner = GELU_TANH_COEFF * (v + 0.044_715 * v * v * v);
        0.5 * v * (1.0 + inner.tanh())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gelu_matches_known_values() {
        let x =
            Tensor::from_vec(vec![1, 3], vec![-1.0, 0.0, 1.0]).expect("invariant: tensor valide");
        let out = gelu(&x);
        assert!((out.data()[0] - -0.158_655_26).abs() < 1.0e-6);
        assert_eq!(out.data()[1], 0.0);
        assert!((out.data()[2] - 0.841_344_7).abs() < 1.0e-6);
    }

    #[test]
    fn silu_matches_known_values() {
        let x =
            Tensor::from_vec(vec![1, 3], vec![-1.0, 0.0, 1.0]).expect("invariant: tensor valide");
        let out = silu(&x);
        assert!((out.data()[0] - -0.268_941_43).abs() < 1.0e-6);
        assert_eq!(out.data()[1], 0.0);
        assert!((out.data()[2] - 0.731_058_6).abs() < 1.0e-6);
    }

    #[test]
    fn gelu_tanh_matches_pytorch_reference() {
        // Références : torch.nn.functional.gelu(x, approximate="tanh").
        let x = Tensor::from_vec(vec![1, 5], vec![-3.0, -1.0, 0.0, 1.0, 3.0])
            .expect("invariant: tensor valide");
        let out = gelu_tanh(&x);
        let expected = [-0.003_637_392, -0.158_808, 0.0, 0.841_192, 2.996_362_6];
        for (value, expected) in out.data().iter().zip(expected) {
            assert!(
                (value - expected).abs() < 1.0e-6,
                "gelu_tanh: {value} attendu {expected}"
            );
        }
    }
}
