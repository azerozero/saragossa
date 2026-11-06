//! Attention QSA résidente de Flash-Next, régime dense (V ≤ 2051).
//!
//! Jusqu'à `budget + compress − 1 = 2051` tokens visibles, l'indexeur retient
//! tous les blocs complets plus la queue : la sélection QSA est l'attention
//! causale dense. Au-delà, la session refuse le decode (spike QSA séparé).
//!
//! Primitives réutilisées sans l'orchestrateur Qwen : pas d'`input_norm`
//! (l'entrée sort déjà de l'hyper-connection), normes q/k chargées avec leur
//! gain `1 + w` appliqué une seule fois, `q_proj` entrelacé (query | gate) par
//! tête, RoPE partielle, KV f32, gate `σ` avant `o_proj`.

use metal::{Buffer, BufferRef, ComputeCommandEncoderRef};

use crate::decode_resident::{DecodeResidentState, FullAttentionMetalState};
use crate::metal_backend::{post_dispatch_barrier, MetalExecutor, MetalLinearWeightBuffers};
use crate::{InferError, Result};

/// Visibilité maximale du régime dense : `budget + compress − 1`.
pub(crate) const DENSE_MAX_VISIBLE: usize = 2051;

/// Poids GPU d'une couche QSA (indexeur hors régime dense).
pub(crate) struct QsaLayerWeights {
    /// `[2·heads·head_dim, hidden]`, query et gate entrelacés par tête.
    pub(crate) q_proj: MetalLinearWeightBuffers,
    pub(crate) k_proj: MetalLinearWeightBuffers,
    pub(crate) v_proj: MetalLinearWeightBuffers,
    pub(crate) o_proj: MetalLinearWeightBuffers,
    /// Gains `1 + w` de `q_norm` / `k_norm` (`[head_dim]`).
    pub(crate) q_norm: Buffer,
    pub(crate) k_norm: Buffer,
}

/// Dimensions de l'attention.
#[derive(Clone, Copy, Debug)]
pub(crate) struct QsaDecodeDims {
    pub(crate) hidden: usize,
    pub(crate) heads: usize,
    pub(crate) kv_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) rotary_dim: usize,
    pub(crate) theta: f32,
    pub(crate) eps: f32,
}

impl QsaDecodeDims {
    fn q_dim(self) -> usize {
        self.heads * self.head_dim
    }
}

/// Intermédiaires de l'attention.
pub(crate) struct QsaScratch {
    pub(crate) q_gate: Buffer,
    pub(crate) q_raw: Buffer,
    pub(crate) gate: Buffer,
    pub(crate) q_roped: Buffer,
    pub(crate) k_raw: Buffer,
    pub(crate) k_roped: Buffer,
    pub(crate) v: Buffer,
    /// `[heads, capacité]`.
    pub(crate) scores: Buffer,
    pub(crate) ctx: Buffer,
    pub(crate) gated: Buffer,
}

/// Encode l'attention du token à `position` : `x [hidden]` → `out [hidden]`,
/// en ajoutant son K/V au cache.
#[expect(
    clippy::too_many_arguments,
    reason = "encodage Metal : arène, exécuteur, cache, poids, scratch et buffers explicites"
)]
pub(crate) fn encode_qsa_dense(
    arena: &DecodeResidentState,
    executor: &MetalExecutor,
    encoder: &ComputeCommandEncoderRef,
    kv: &mut FullAttentionMetalState,
    weights: &QsaLayerWeights,
    scratch: &QsaScratch,
    dims: QsaDecodeDims,
    position: usize,
    x: &BufferRef,
    out: &BufferRef,
) -> Result<()> {
    if position >= DENSE_MAX_VISIBLE {
        return Err(InferError::Config(format!(
            "Flash-Next résident : position {position}, sélection QSA creuse au-delà de \
             {DENSE_MAX_VISIBLE} tokens visibles (non prise en charge)"
        )));
    }
    for (weight, target) in [
        (&weights.q_proj, &scratch.q_gate),
        (&weights.k_proj, &scratch.k_raw),
        (&weights.v_proj, &scratch.v),
    ] {
        executor.encode_matmul_weight_buffers(encoder, x, 1, dims.hidden, weight, target, true)?;
    }
    post_dispatch_barrier(encoder);
    arena.encode_split_q_gate(
        encoder,
        &scratch.q_gate,
        &scratch.q_raw,
        &scratch.gate,
        dims.heads,
        dims.head_dim,
    )?;
    for (input, norm, target, heads) in [
        (
            &scratch.q_raw,
            &weights.q_norm,
            &scratch.q_roped,
            dims.heads,
        ),
        (
            &scratch.k_raw,
            &weights.k_norm,
            &scratch.k_roped,
            dims.kv_heads,
        ),
    ] {
        arena.encode_rms_norm_rope_decode(
            encoder,
            input,
            norm,
            target,
            heads,
            dims.head_dim,
            dims.rotary_dim,
            dims.rotary_dim,
            position,
            dims.eps,
            dims.theta,
        )?;
    }
    kv.encode_append_kv(encoder, &scratch.k_roped, &scratch.v)?;
    kv.encode_attention_decode_windowed(
        encoder,
        &scratch.q_roped,
        &scratch.scores,
        &scratch.ctx,
        0,
    )?;
    arena.encode_attn_gate(
        encoder,
        &scratch.ctx,
        &scratch.gate,
        &scratch.gated,
        dims.q_dim(),
    )?;
    executor.encode_matmul_weight_buffers(
        encoder,
        &scratch.gated,
        1,
        dims.q_dim(),
        &weights.o_proj,
        out,
        true,
    )?;
    post_dispatch_barrier(encoder);
    Ok(())
}
