//! Attention partagée par Whisper et la diarisation, avec oracle CPU.

use crate::{ForwardRuntime, InferError, Result, Tensor};

pub(crate) fn multi_head_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    heads: usize,
    causal: bool,
    runtime: ForwardRuntime<'_>,
) -> Result<Tensor> {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    if let Some(metal) = runtime.metal_executor() {
        if !causal && q.shape() == k.shape() && k.shape() == v.shape() {
            return metal.noncausal_attention_prefill(q, k, v, heads, heads);
        }
    }
    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    let _ = runtime;
    let (q_seq, dim) = q.as_matrix()?;
    let (kv_seq, k_dim) = k.as_matrix()?;
    if v.shape() != k.shape() || k_dim != dim {
        return Err(InferError::Dimension(format!(
            "attention q={:?}, k={:?}, v={:?}",
            q.shape(),
            k.shape(),
            v.shape()
        )));
    }
    if heads == 0 || dim % heads != 0 {
        return Err(InferError::Dimension(format!(
            "attention dim={dim} heads={heads}"
        )));
    }
    let head_dim = dim / heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0.0_f32; q_seq * dim];
    let mut scores = vec![0.0_f32; kv_seq];
    for head in 0..heads {
        let head_base = head * head_dim;
        for row_q in 0..q_seq {
            let mut max_score = f32::NEG_INFINITY;
            for (row_k, score_value) in scores.iter_mut().enumerate().take(kv_seq) {
                if causal && row_k > row_q {
                    *score_value = f32::NEG_INFINITY;
                    continue;
                }
                let mut dot = 0.0_f32;
                for col in 0..head_dim {
                    dot += q.data()[row_q * dim + head_base + col]
                        * k.data()[row_k * dim + head_base + col];
                }
                let score = dot * scale;
                *score_value = score;
                max_score = max_score.max(score);
            }
            let mut denom = 0.0_f32;
            for score in scores.iter_mut().take(kv_seq) {
                if *score == f32::NEG_INFINITY {
                    continue;
                }
                *score = (*score - max_score).exp();
                denom += *score;
            }
            for col in 0..head_dim {
                let mut acc = 0.0_f32;
                for (row_v, prob) in scores.iter().take(kv_seq).enumerate() {
                    if *prob == f32::NEG_INFINITY {
                        continue;
                    }
                    acc += (*prob / denom) * v.data()[row_v * dim + head_base + col];
                }
                out[row_q * dim + head_base + col] = acc;
            }
        }
    }
    Tensor::from_vec(vec![q_seq, dim], out)
}
