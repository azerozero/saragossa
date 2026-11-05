//! Gated residual (« hyper-connections ») de `qwen4_exp`.
//!
//! Avec `n = RMSNorm_groupée(H)` (groupes de `hidden`, gain `1 + w`) :
//! `a = σ(W_up · SiLU(W_down·n / hc))`, `x = mean_flux(a ⊙ n)` ;
//! injection `2σ(W_inject·n / hc)` ajoutée au résiduel **non normalisé** :
//! `H′_s = H_s + inj_s · F(x)`. Le mixer final applique le même mélange sans
//! injection (transformers `Qwen4ExpTextGatedResidual`, l. 994-1030).

use super::math::{linear, rms_norm_1p, sigmoid, silu};
use crate::{Result, Tensor};

/// Poids d'un gated residual (mixer final : `inject = None`).
pub(crate) struct GatedResidual {
    pub(crate) norm: Vec<f32>,
    pub(crate) down: Tensor,
    pub(crate) up: Tensor,
    pub(crate) inject: Option<Tensor>,
}

/// Entrée de branche `x` `[T, hidden]` et poids d'injection `[T, hc]`.
pub(crate) struct Mixed {
    pub(crate) input: Tensor,
    pub(crate) inject: Option<Tensor>,
}

impl GatedResidual {
    /// Mélange les `hc` flux de `h` (`[T, hc·hidden]`).
    pub(crate) fn mix(&self, h: &Tensor, hc: usize, hidden: usize, eps: f32) -> Result<Mixed> {
        let (tokens, width) = h.as_matrix()?;
        let normed = Tensor::from_vec(
            vec![tokens, width],
            rms_norm_1p(h.data(), width, hidden, &self.norm, eps),
        )?;
        let low = linear(&normed, &self.down)?.map(|v| silu(v / hc as f32));
        let weights = linear(&low, &self.up)?.map(sigmoid);
        let mut input = vec![0.0_f32; tokens * hidden];
        for t in 0..tokens {
            let n = &normed.data()[t * width..(t + 1) * width];
            let a = &weights.data()[t * width..(t + 1) * width];
            let out = &mut input[t * hidden..(t + 1) * hidden];
            for s in 0..hc {
                for d in 0..hidden {
                    out[d] += a[s * hidden + d] * n[s * hidden + d];
                }
            }
            out.iter_mut().for_each(|v| *v /= hc as f32);
        }
        let inject = match &self.inject {
            Some(w) => Some(linear(&normed, w)?.map(|v| 2.0 * sigmoid(v / hc as f32))),
            None => None,
        };
        Ok(Mixed {
            input: Tensor::from_vec(vec![tokens, hidden], input)?,
            inject,
        })
    }
}

/// `H′_s = H_s + inj_s · F(x)` pour chaque flux `s`.
pub(crate) fn combine(h: &Tensor, branch: &Tensor, inject: &Tensor, hc: usize) -> Result<Tensor> {
    let (tokens, width) = h.as_matrix()?;
    let hidden = width / hc;
    let mut out = h.data().to_vec();
    for t in 0..tokens {
        let f = &branch.data()[t * hidden..(t + 1) * hidden];
        for s in 0..hc {
            let w = inject.data()[t * hc + s];
            let row = &mut out[t * width + s * hidden..t * width + (s + 1) * hidden];
            for (v, x) in row.iter_mut().zip(f) {
                *v += w * x;
            }
        }
    }
    Tensor::from_vec(vec![tokens, width], out)
}
