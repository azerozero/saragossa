//! Couche PLE (embedding n-gram) de `qwen4_exp`, référence sans cache.
//!
//! Transcription de `Qwen4ExpTextPLELayer` (modeling l. 1174-1247), pour chaque
//! flux `s` : `s = ⟨norm_key(K), norm_query(H)⟩/√hidden`,
//! `g = σ(sign(s)·√max(|s|, 1e-6))`, `u = g·V`,
//! `PLE = u + SiLU(conv(norm_conv(u)))` ; conv depthwise sans biais, noyau
//! `k`, dilation `ngram_size`, historique nul de `(k−1)·dilation` tokens.

use super::math::{linear, rms_norm_1p, sigmoid, silu};
use crate::{Result, Tensor};

/// Poids PLE (projections déquantifiées, normes et conv copiées).
pub(crate) struct PleWeights {
    pub(crate) key_proj: Tensor,
    pub(crate) value_proj: Tensor,
    pub(crate) norm_key: Vec<f32>,
    pub(crate) norm_query: Vec<f32>,
    pub(crate) norm_conv: Vec<f32>,
    /// `[hc·hidden, 1, k]` aplati.
    pub(crate) conv: Vec<f32>,
    pub(crate) kernel: usize,
    pub(crate) dilation: usize,
}

/// Contribution PLE `[T, hc·hidden]`, à ajouter aux flux avant l'hyper-connection
/// d'attention. `embeddings` est `[T, ple_dim]` (lignes n-gram concaténées).
pub(crate) fn ple(
    h: &Tensor,
    embeddings: &Tensor,
    w: &PleWeights,
    hc: usize,
    hidden: usize,
    eps: f32,
) -> Result<Tensor> {
    let (tokens, width) = h.as_matrix()?;
    let key = linear(embeddings, &w.key_proj)?;
    let key = rms_norm_1p(key.data(), width, hidden, &w.norm_key, eps);
    let value = linear(embeddings, &w.value_proj)?;
    let query = rms_norm_1p(h.data(), width, hidden, &w.norm_query, eps);
    let mut gated = vec![0.0_f32; tokens * width];
    let root = (hidden as f32).sqrt();
    for t in 0..tokens {
        for s in 0..hc {
            let range = t * width + s * hidden..t * width + (s + 1) * hidden;
            let dot: f32 = key[range.clone()]
                .iter()
                .zip(&query[range.clone()])
                .map(|(a, b)| a * b)
                .sum::<f32>()
                / root;
            let g = sigmoid(dot.abs().max(1.0e-6).sqrt() * dot.signum_or_zero());
            let v = &value.data()[t * hidden..(t + 1) * hidden];
            for (out, x) in gated[range].iter_mut().zip(v) {
                *out = g * x;
            }
        }
    }
    let normed = rms_norm_1p(&gated, width, hidden, &w.norm_conv, eps);
    let mut out = gated;
    let span = (w.kernel - 1) * w.dilation;
    for t in 0..tokens {
        for c in 0..width {
            let taps = &w.conv[c * w.kernel..(c + 1) * w.kernel];
            let mut acc = 0.0_f32;
            for (j, tap) in taps.iter().enumerate() {
                // y[t] = Σ_j w_j · x[t − span + j·dilation] (historique nul).
                let source = t + j * w.dilation;
                if source >= span {
                    acc += tap * normed[(source - span) * width + c];
                }
            }
            out[t * width + c] += silu(acc);
        }
    }
    Tensor::from_vec(vec![tokens, width], out)
}

trait SignumOrZero {
    fn signum_or_zero(self) -> f32;
}

impl SignumOrZero for f32 {
    /// `torch.sign` : 0 pour 0 (f32::signum renvoie ±1).
    fn signum_or_zero(self) -> f32 {
        if self > 0.0 {
            1.0
        } else if self < 0.0 {
            -1.0
        } else {
            0.0
        }
    }
}
