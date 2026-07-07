use super::super::*;

#[cfg(all(target_os = "macos", feature = "metal"))]
pub(super) const RESIDENT_PIPELINE_WINDOW: usize = 4;

#[cfg(all(target_os = "macos", feature = "metal"))]
pub(super) enum ResidentDecodeInput<'a> {
    CpuToken(usize),
    GpuIndex(&'a metal::BufferRef),
}

#[cfg(all(target_os = "macos", feature = "metal"))]
pub(super) struct ResidentDecodeInflight {
    pub(super) command_buffer: metal::CommandBuffer,
    pub(super) index: metal::Buffer,
    pub(super) logits_readback: Option<ResidentLogitsReadback>,
    pub(super) _owned: Vec<metal::Buffer>,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
pub(super) struct ResidentLogitsReadback {
    pub(super) buffer: metal::Buffer,
    pub(super) len: usize,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
pub(super) fn resident_capture_slot(layer_ids: &[usize], layer_index: usize) -> Option<usize> {
    layer_ids
        .iter()
        .position(|candidate| *candidate == layer_index)
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[derive(Clone, Copy)]
pub(in crate::decoder) struct ResidentSampleSpec {
    pub(in crate::decoder) temperature: f32,
    pub(in crate::decoder) top_p: f32,
    pub(in crate::decoder) top_k: usize,
    pub(in crate::decoder) rng_state: u64,
    pub(super) mode: ResidentSampleMode,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[derive(Clone, Copy)]
pub(super) enum ResidentSampleMode {
    OnDevice,
    Readback,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
pub(in crate::decoder) struct ResidentPipelineOutput {
    pub(in crate::decoder) tokens: Vec<usize>,
    pub(in crate::decoder) decode: Duration,
    pub(in crate::decoder) decode_tokens: usize,
}

/// Sortie d'un pas résident piloté par embedding : soit le `final_state` post-norm
/// relu (pas de tête), soit l'argmax greedy on-device d'une tête fournie (TTS cp).
#[cfg(all(target_os = "macos", feature = "metal"))]
pub(in crate::decoder) enum ResidentEmbeddingOut {
    State(Tensor),
    Token(usize),
}

#[cfg(all(target_os = "macos", feature = "metal"))]
pub(super) fn resident_full_layer_unsupported_reason(layer: &DecoderLayer) -> Option<&'static str> {
    if layer.pre_feedforward_norm.is_some() {
        return Some("pre_feedforward_norm présent");
    }
    let Some(mlp) = layer.mlp.as_ref() else {
        return Some("MLP absent");
    };
    if layer.post_attention_norm.is_none() {
        return Some("post_attention_norm absent");
    }
    match mlp {
        FeedForward::Moe(mlp) => {
            if mlp.shared_metal_parts().is_none() && mlp.metal_parts().is_none() {
                return Some("MoE non encodable Metal");
            }
        }
        FeedForward::Dense(mlp) => {
            let (gate_proj, up_proj, down_proj) = mlp.projections();
            if gate_proj.bias().is_some() || up_proj.bias().is_some() || down_proj.bias().is_some()
            {
                return Some("MLP dense biaisé");
            }
        }
    }
    match &layer.attention {
        AttentionBlock::Full(attention) => {
            let Some(v_proj) = attention.v_proj.as_ref() else {
                return Some("full-attn v_proj absent");
            };
            if attention.q_proj.bias().is_some()
                || attention.k_proj.bias().is_some()
                || v_proj.bias().is_some()
                || attention.o_proj.bias().is_some()
            {
                return Some("full-attn projection biaisée");
            }
            if attention.q_norm.is_none() {
                return Some("full-attn q_norm absent");
            }
            if attention.k_norm.is_none() {
                return Some("full-attn k_norm absent");
            }
        }
        AttentionBlock::Linear(_) => {}
    }
    None
}
