//! Couches linéaires denses ou quantifiées et exécution runtime.

use crate::{AffineQuantizedTensor, ForwardRuntime, InferError, Result, Tensor};

#[derive(Clone, Debug)]
/// Représente une couche linéaire avec biais optionnel.
pub struct Linear {
    weight: LinearWeight,
    bias: Option<Tensor>,
}

#[derive(Clone, Debug, PartialEq)]
/// Décrit le stockage des poids d'une couche linéaire.
pub enum LinearWeight {
    /// Stocke les poids denses en f32.
    Dense(Tensor),
    /// Stocke les poids affine quantifiés.
    AffineQuantized(AffineQuantizedTensor),
}

impl LinearWeight {
    pub(crate) fn shape(&self) -> &[usize] {
        match self {
            Self::Dense(weight) => weight.shape(),
            Self::AffineQuantized(weight) => weight.shape(),
        }
    }

    fn as_matrix(&self) -> Result<(usize, usize)> {
        match self.shape() {
            [rows, cols] => Ok((*rows, *cols)),
            shape => Err(InferError::Dimension(format!(
                "poids Linear attendu rang 2, reçu {shape:?}"
            ))),
        }
    }

    fn matmul_rhs_t_with_runtime(
        &self,
        input: &Tensor,
        runtime: ForwardRuntime<'_>,
    ) -> Result<Tensor> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if let Some(metal) = runtime.metal_executor() {
            return match self {
                Self::Dense(weight) => metal.matmul_rhs_t_dense(input, weight),
                Self::AffineQuantized(weight) => metal.matmul_rhs_t_affine(input, weight),
            };
        }
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        let _ = runtime;
        match self {
            Self::Dense(weight) => input.matmul_rhs_t(weight),
            Self::AffineQuantized(weight) => weight.matmul_rhs_t(input),
        }
    }
}

impl Linear {
    /// Crée une couche linéaire dense.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le poids ou le biais est invalide.
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Result<Self> {
        Self::from_weight(LinearWeight::Dense(weight), bias)
    }

    /// Crée une couche linéaire quantifiée.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le poids ou le biais est invalide.
    pub fn new_quantized(weight: AffineQuantizedTensor, bias: Option<Tensor>) -> Result<Self> {
        Self::from_weight(LinearWeight::AffineQuantized(weight), bias)
    }

    /// Crée une couche depuis un stockage de poids explicite.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si les dimensions sont invalides.
    pub fn from_weight(weight: LinearWeight, bias: Option<Tensor>) -> Result<Self> {
        let (_, in_dim) = weight.as_matrix()?;
        if let Some(bias) = &bias {
            match bias.shape() {
                [out_dim] if *out_dim == weight.shape()[0] => {}
                [1, out_dim] if *out_dim == weight.shape()[0] => {}
                _ => {
                    return Err(InferError::Dimension(format!(
                        "bias Linear incompatible: weight={:?}, bias={:?}",
                        weight.shape(),
                        bias.shape()
                    )))
                }
            }
        }
        if in_dim == 0 {
            return Err(InferError::Dimension(
                "Linear avec dimension entrée nulle".to_string(),
            ));
        }
        Ok(Self { weight, bias })
    }

    /// Exécute la couche avec le runtime CPU.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si les dimensions d'entrée sont incompatibles.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_with_runtime(input, ForwardRuntime::cpu())
    }

    /// Exécute la couche avec le runtime demandé.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si les dimensions ou le runtime échouent.
    pub fn forward_with_runtime(
        &self,
        input: &Tensor,
        runtime: ForwardRuntime<'_>,
    ) -> Result<Tensor> {
        let mut out = self.weight.matmul_rhs_t_with_runtime(input, runtime)?;
        if let Some(bias) = &self.bias {
            out = out.add_row_bias(bias)?;
        }
        Ok(out)
    }

    #[must_use]
    /// Renvoie le stockage des poids.
    pub fn weight(&self) -> &LinearWeight {
        &self.weight
    }

    #[must_use]
    #[cfg_attr(
        not(all(target_os = "macos", feature = "metal")),
        expect(
            dead_code,
            reason = "utilisé par les chemins Metal, absent du build CPU pur"
        )
    )]
    pub(crate) fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn applies_dense_layer_with_bias() {
        let weight = Tensor::from_vec(vec![2, 3], vec![1.0, 0.0, 1.0, -1.0, 2.0, 0.5])
            .expect("invariant: poids valide");
        let bias = Tensor::from_vec(vec![2], vec![0.5, -0.5]).expect("invariant: bias valide");
        let layer = Linear::new(weight, Some(bias)).expect("invariant: layer valide");
        let input =
            Tensor::from_vec(vec![1, 3], vec![2.0, 3.0, 4.0]).expect("invariant: entrée valide");
        let out = layer.forward(&input).expect("invariant: forward valide");
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(out.data(), &[6.5, 5.5]);
    }

    #[test]
    fn applies_compact_affine_layer() {
        let packed = vec![0x0000_00ff, 0x0000_ff00];
        let scales =
            Tensor::from_vec(vec![2, 2], vec![1.0 / 255.0; 4]).expect("invariant: scales valides");
        let biases = Tensor::from_vec(vec![2, 2], vec![0.0; 4]).expect("invariant: biases valides");
        let weight = AffineQuantizedTensor::new(&[2, 1], packed, scales, biases, 2, 8)
            .expect("invariant: poids compact valide");
        let layer = Linear::new_quantized(weight, None).expect("invariant: layer valide");
        let input =
            Tensor::from_vec(vec![1, 4], vec![2.0, 3.0, 5.0, 7.0]).expect("invariant: entrée");

        let out = layer.forward(&input).expect("invariant: forward valide");

        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(out.data(), &[2.0, 3.0]);
    }
}
