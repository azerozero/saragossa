//! Opérations f32 de la référence CPU `qwen4_exp` (transcriptions transformers 5.20.2).

use rayon::prelude::*;

use crate::{Result, Tensor};

/// `x · Wᵀ` pour `x` `[T, in]` et `W` `[out, in]`.
pub(crate) fn linear(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    x.matmul_rhs_t(weight)
}

/// `σ(x) = 1/(1+e⁻ˣ)`.
pub(crate) fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

/// `SiLU(x) = x·σ(x)`.
pub(crate) fn silu(value: f32) -> f32 {
    value * sigmoid(value)
}

/// `Qwen4ExpTextRMSNorm` : normalise chaque groupe de `group` valeurs (toute la
/// ligne si `group` vaut sa largeur) puis multiplie par `1 + w` (gains centrés
/// sur 0).
pub(crate) fn rms_norm_1p(
    x: &[f32],
    width: usize,
    group: usize,
    weight: &[f32],
    eps: f32,
) -> Vec<f32> {
    let mut out = x.to_vec();
    out.par_chunks_mut(width).for_each(|row| {
        for chunk in row.chunks_exact_mut(group) {
            let mean = chunk.iter().map(|v| v * v).sum::<f32>() / group as f32;
            let inv = 1.0 / (mean + eps).sqrt();
            chunk.iter_mut().for_each(|v| *v *= inv);
        }
        for (v, w) in row.iter_mut().zip(weight) {
            *v *= 1.0 + w;
        }
    });
    out
}

/// Fréquences inverses de la RoPE partielle : `θ^(−2i/rotary_dim)`, `i < rotary_dim/2`.
///
/// Calculées en f32 comme `1.0 / (base ** (arange(0, dim, 2) / dim))` amont.
pub(crate) fn inv_freq(rotary_dim: usize, theta: f64) -> Vec<f32> {
    let base = theta as f32;
    (0..rotary_dim / 2)
        .map(|i| 1.0 / base.powf((2 * i) as f32 / rotary_dim as f32))
        .collect()
}

/// RoPE partielle `rotate_half` sur les `2·freqs.len()` premières dimensions de
/// chaque tête de `x` (`[heads, head_dim]` aplati), à la position `position`.
///
/// En texte seul les trois axes mRoPE portent la même position : la
/// recomposition entrelacée `[11,11,10]` est alors l'identité.
pub(crate) fn rope_in_place(head: &mut [f32], freqs: &[f32], position: usize) {
    let half = freqs.len();
    for (i, &freq) in freqs.iter().enumerate() {
        // Même calcul que transformers : `pos·inv_freq` en f32, puis cos/sin.
        let angle = position as f32 * freq;
        let (sin, cos) = angle.sin_cos();
        let (a, b) = (head[i], head[i + half]);
        head[i] = a * cos - b * sin;
        head[i + half] = b * cos + a * sin;
    }
}

/// Erreur relative `‖a − b‖ / max(‖b‖, 1e-30)` et écart absolu maximal.
pub fn relative_error(a: &[f32], b: &[f32]) -> (f64, f64) {
    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    let mut max = 0.0_f64;
    for (&x, &y) in a.iter().zip(b) {
        let d = f64::from(x) - f64::from(y);
        num += d * d;
        den += f64::from(y) * f64::from(y);
        max = max.max(d.abs());
    }
    (num.sqrt() / den.sqrt().max(1e-30), max)
}
