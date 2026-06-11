//! Fonctions d'activation élémentaires pour les tenseurs CPU.

use crate::Tensor;

pub fn silu(x: &Tensor) -> Tensor {
    x.map(|v| v / (1.0 + (-v).exp()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silu_matches_known_values() {
        let x =
            Tensor::from_vec(vec![1, 3], vec![-1.0, 0.0, 1.0]).expect("invariant: tensor valide");
        let out = silu(&x);
        assert!((out.data()[0] - -0.268_941_43).abs() < 1.0e-6);
        assert_eq!(out.data()[1], 0.0);
        assert!((out.data()[2] - 0.731_058_6).abs() < 1.0e-6);
    }
}
