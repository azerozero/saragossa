//! Kernels Metal pilotés en Rust pour le backend GPU expérimental.

use crate::{
    AffineQuantizedTensor, EmbeddingWeight, GatedMlp, InferError, Linear, LinearWeight, Result,
    Tensor,
};
use metal::foreign_types::ForeignTypeRef;
use metal::objc::runtime::{sel_registerName, Object, Sel};
#[cfg(test)]
use metal::CompileOptions;
use metal::{
    Buffer, BufferRef, CommandQueue, ComputeCommandEncoderRef, ComputePipelineState, Device,
    MTLCommandBufferStatus, MTLResourceOptions, MTLSize, NSUInteger,
};
use std::cell::Cell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Mutex, OnceLock};

const MATMUL_KERNELS: &str = include_str!("../kernels.metal");
#[cfg(all(target_os = "macos", feature = "metal"))]
const STEEL_ATTENTION_KERNELS: &str =
    include_str!(concat!(env!("OUT_DIR"), "/steel_attention.metal"));
#[cfg(all(target_os = "macos", feature = "metal"))]
const STEEL_ATTN_F32_BQ32_BK32_BD64: &str =
    "steel_attention_float32_bq32_bk32_bd64_wm4_wn1_maskfloat32";
const STEEL_CAUSAL_ATTN_D256_F32_BQ32_BK64_BD64X4: &str =
    "steel_attention_float32_bq32_bk64_bd64x4_wm4_wn1_causal_d256";
const MAX_MOE_TOP_K: usize = 16;
pub(crate) const MAX_SAMPLER_TOP_K: usize = 32;
const FAST_QMV_GROUP_SIZE: usize = 64;
const QMM_NA_GS128_GROUP_SIZE: usize = 128;
const FAST_QMV_BITS: usize = 4;
const FAST_QMV_U6_BITS: usize = 6;

mod attention;
mod attention_checks;
mod buffers;
mod core;
mod kernel_timing;
mod lightbatch_duo;
mod linear_attention;
mod linear_attention_encode;
mod lm_head;
mod matmul;
mod matmul_encode;
mod moe;
mod moe_encode;
mod moe_na;
mod moe_shared;
mod na_gemm;
mod pipelines;
mod prefill;
#[cfg(test)]
mod tests;
mod whisper_encode;

#[cfg(any(test, feature = "devtools"))]
pub(crate) use self::core::read_u16_buffer;
pub(crate) use self::core::write_f32_buffer;
use self::core::*;
pub(crate) use self::core::{
    commit_and_wait, commit_nonblocking, install_dispatch_barrier_scope, install_scratch_namespace,
    post_dispatch_barrier, profile_dispatch, profile_dispatch_shape, read_f32_buffer,
    read_u32_buffer, resident_concurrent_enabled, trace_dispatch_path, wait_for_completion,
    EncoderEndGuard,
};
#[doc(hidden)]
pub use self::core::{
    decode_profile_dispatch_shapes_snapshot, decode_profile_dispatch_sites_snapshot,
    decode_profile_snapshot, dump_commit_components, DispatchProfileShape, DispatchProfileSite,
};
pub(crate) use self::lightbatch_duo::{
    begin_expert_indices_collection, take_expert_indices_collection, DuoSampleParams,
};
pub(crate) use self::matmul::{whisper_bf16_gemm_enabled, whisper_decode_bf16_qmv_enabled};
use self::na_gemm::NA_GEMM_SRC;
#[cfg(test)]
use self::pipelines::KernelSources;
pub(crate) use self::whisper_encode::{
    WhisperConvWeights, WhisperDecodeKv, WhisperDecodeLayer, WhisperResidentDecoder,
    WhisperResidentEncoder, WhisperResidentLayer, WhisperResidentNorm, WhisperResidentProj,
};

fn ensure_valid_top_k(top_k: usize, expert_count: usize) -> Result<()> {
    if top_k == 0 || top_k > MAX_MOE_TOP_K || top_k > expert_count {
        return Err(InferError::Config(format!(
            "MoE Metal top_k={top_k} invalide pour {expert_count} experts (max={MAX_MOE_TOP_K})"
        )));
    }
    Ok(())
}

/// Exécute les premiers kernels Metal du backend Rust expérimental.
#[derive(Debug)]
pub struct MetalExecutor {
    device: Device,
    queue: CommandQueue,
    dense_matmul_rhs_t_f32: ComputePipelineState,
    dense_qmv_rhs_bf16_f32: ComputePipelineState,
    dense_qmv_fast_f32: ComputePipelineState,
    dense_gemm_rhs_t_f32: ComputePipelineState,
    affine_matmul_rhs_t_u32_f32: ComputePipelineState,
    affine_qmv_fast_u4_gs64_f32: ComputePipelineState,
    affine_qmv_fast_aligned_u4_gs64_f32: ComputePipelineState,
    affine_qmv_fast_u6_gs64_f32: ComputePipelineState,
    affine_qmv_fast_aligned_u6_gs64_f32: ComputePipelineState,
    affine_qmm2_fast_aligned_u4_gs64_f32: ComputePipelineState,
    affine_qmm2_fast_aligned_u8_gs64_f32: ComputePipelineState,
    affine_qmm2_fast_aligned_u8_gs128_f32: ComputePipelineState,
    affine_qmv_fast_aligned_u8_gs64_f32: ComputePipelineState,
    affine_qmv_fast_aligned_u8_gs128_f32: ComputePipelineState,
    affine_qmv_fast_aligned_u8_gs64_dot4_f32: ComputePipelineState,
    affine_qmv_fast_aligned_u8_gs128_dot4_f32: ComputePipelineState,
    affine_qmv_fast_aligned_u8_gs64_tg128_f32: ComputePipelineState,
    affine_qmv_fast_aligned_u8_gs128_tg128_f32: ComputePipelineState,
    affine_qmv_fast_aligned_u8_gs64_tg256_f32: ComputePipelineState,
    affine_qmv_fast_aligned_u8_gs128_tg256_f32: ComputePipelineState,
    affine_qmv_plus_one_fast_aligned_u8_gs64_f32: ComputePipelineState,
    affine_qmv_one_fast_u8_gs64_f32: ComputePipelineState,
    affine_qkv_split_qmv_fast_u4_gs64_f32: ComputePipelineState,
    affine_qmv_rms_fast_u4_gs64_f32: ComputePipelineState,
    affine_qmv_rms_fast_u8_gs64_f32: ComputePipelineState,
    affine_qmv_rms_fast_u8_gs128_f32: ComputePipelineState,
    affine_qkv_split_rms_qmv_fast_u4_gs64_f32: ComputePipelineState,
    affine_qkv_split_rms_qmv_fast_u8_gs64_f32: ComputePipelineState,
    affine_qmv_gated_input_fast_u4_gs64_f32: ComputePipelineState,
    affine_qmv_gated_input_fast_u8_gs64_f32: ComputePipelineState,
    embed_gather_dense_from_u32_f32: ComputePipelineState,
    embed_gather_affine_from_u32_f32: ComputePipelineState,
    swiglu_f32: ComputePipelineState,
    split_q_gate_rows_f32: ComputePipelineState,
    attn_gate_rows_f32: ComputePipelineState,
    accumulate_scaled_f32: ComputePipelineState,
    add_scaled_f32: ComputePipelineState,
    linear_attn_conv_silu_f32: ComputePipelineState,
    linear_attn_conv_silu_k4_f32: ComputePipelineState,
    linear_attn_norm_gates_f32: ComputePipelineState,
    linear_attn_norm_gates_dk128_f32: ComputePipelineState,
    linear_attn_norm_gates_inv_dk128_f32: ComputePipelineState,
    linear_attn_conv_norm_gates_k4_dk128_f32: ComputePipelineState,
    linear_attn_conv_norm_gates_k4_dk128_batch_f32: ComputePipelineState,
    linear_attn_conv_state_finalize_f32: ComputePipelineState,
    linear_attn_gated_delta_f32: ComputePipelineState,
    linear_attn_gated_delta_dk128_tg4_f32: ComputePipelineState,
    linear_attn_gated_delta_seq_dk128_tg4_f32: ComputePipelineState,
    linear_attn_gated_delta_seq_dk128_bf16_tg4_f32: ComputePipelineState,
    linear_attn_gated_delta_dk128_bf16_tg4_f32: ComputePipelineState,
    linear_attn_gated_delta_inv_dk128_tg4_f32: ComputePipelineState,
    linear_attn_rms_gate_f32: ComputePipelineState,
    linear_attn_rms_gate_dv128_f32: ComputePipelineState,
    linear_attn_rms_gate_batch_dv128_f32: ComputePipelineState,
    affine_gather_matmul_rhs_t_u32_f32: ComputePipelineState,
    affine_gather_qmv_fast_u4_gs64_f32: ComputePipelineState,
    affine_gather_qmv_fast_u8_gs64_f32: ComputePipelineState,
    affine_gather_qmv_fast_u8_gs128_f32: ComputePipelineState,
    affine_gather_qmv_fast_u8_gs64_tg128_f32: ComputePipelineState,
    affine_gather_qmv_fast_u8_gs128_tg128_f32: ComputePipelineState,
    affine_gather_qmv_fast_u8_gs64_tg256_f32: ComputePipelineState,
    affine_gather_qmv_fast_u8_gs128_tg256_f32: ComputePipelineState,
    affine_gather_qmv_tail_u4_gs64_f32: ComputePipelineState,
    affine_gather_gate_up_swiglu_fast_u4_gs64_f32: ComputePipelineState,
    affine_gather_gate_up_swiglu_fast_u8_gs64_f32: ComputePipelineState,
    affine_gather_gate_up_swiglu_fast_u8_gs128_f32: ComputePipelineState,
    affine_gate_up_swiglu_fast_u4_gs64_f32: ComputePipelineState,
    affine_gate_up_swiglu_fast_u8_gs64_f32: ComputePipelineState,
    affine_gate_up_swiglu_gate_fast_u8_gs64_f32: ComputePipelineState,
    affine_gate_up_swiglu_gate_fast_u8_gs128_f32: ComputePipelineState,
    affine_gate_up_swiglu_fast_u8_gs128_f32: ComputePipelineState,
    affine_argmax_qmv_fast_u4_gs64_f32: ComputePipelineState,
    weighted_sum_topk_f32: ComputePipelineState,
    weighted_sum_grouped_topk_f32: ComputePipelineState,
    weighted_sum_add_grouped_topk_f32: ComputePipelineState,
    weighted_sum_add_topk_f32: ComputePipelineState,
    weighted_sum_add_shared_topk_f32: ComputePipelineState,
    affine_gather_down_weighted_shared_fast_u8_gs64_f32: ComputePipelineState,
    add_sigmoid_scaled_f32: ComputePipelineState,
    add_sigmoid_scaled_rows_f32: ComputePipelineState,
    copy_f32: ComputePipelineState,
    copy_u16: ComputePipelineState,
    rms_norm_rows_f32: ComputePipelineState,
    rms_norm_simd_rows_f32: ComputePipelineState,
    add_rms_norm_rows_f32: ComputePipelineState,
    layer_norm_rows_f32: ComputePipelineState,
    add_layer_norm_rows_f32: ComputePipelineState,
    gelu_f32: ComputePipelineState,
    layer_norm_rows_f32_bf16out: ComputePipelineState,
    add_layer_norm_rows_f32_bf16out: ComputePipelineState,
    gelu_f32_bf16out: ComputePipelineState,
    add_row_bias_f32: ComputePipelineState,
    whisper_attn_decode_f32: ComputePipelineState,
    whisper_attn_decode_vec64_f32: ComputePipelineState,
    im2col_f32: ComputePipelineState,
    rms_norm_rope_heads_f32: ComputePipelineState,
    causal_attention_prefill_f32: ComputePipelineState,
    causal_attention_prefill_mid_f32: ComputePipelineState,
    causal_attention_prefill_long_f32: ComputePipelineState,
    causal_attention_prefill_batch_long_d128_f32: ComputePipelineState,
    causal_attention_prefill_batch_long_d256_f32: ComputePipelineState,
    causal_attention_prefill_batch_gqa8x4_d256_f32: ComputePipelineState,
    causal_attention_prefill_steel_d256_f32: Option<ComputePipelineState>,
    noncausal_attention_prefill_f32: ComputePipelineState,
    steel_attention_f32_bq32_bk32_bd64: Option<ComputePipelineState>,
    add_rms_norm_row_f32: ComputePipelineState,
    topk_softmax_f32: ComputePipelineState,
    topk_softmax_serial_f32: ComputePipelineState,
    topk8_softmax_256_f32: ComputePipelineState,
    topk_softmax_rows_f32: ComputePipelineState,
    sample_gumbel_blocks_f32: ComputePipelineState,
    sample_topk_blocks_f32: ComputePipelineState,
    sample_topk_finalize_f32: ComputePipelineState,
    argmax_blocks_f32: ComputePipelineState,
    argmax_finalize_f32: ComputePipelineState,
    talker_greedy_argmax_f32: ComputePipelineState,
    f32_to_bf16: ComputePipelineState,
    dequant_u8_to_bf16_t_gs64: ComputePipelineState,
    /// GEMM Neural Accelerators (`matmul2d`, bf16-in/f32-out) ; `None` si la
    /// MetalPerformancePrimitives n'est pas dispo (macOS < 26).
    na_gemm_bf16: Option<ComputePipelineState>,
    /// Variante encodeur Whisper plus proche du tiling MLX Steel NAX (BN=128).
    na_gemm_bf16_bn128: Option<ComputePipelineState>,
    pub(super) chunk_delta_seq_layout: Option<ComputePipelineState>,
    pub(super) chunk_delta_seq_layout_tc: Option<ComputePipelineState>,
    na_gemm_coop_qb: Option<ComputePipelineState>,
    na_gemm_coop_qb_gs128: Option<ComputePipelineState>,
    na_gemm_coop_qb_tiled: Option<ComputePipelineState>,
    na_gemm_coop_qb_tiled_gs128: Option<ComputePipelineState>,
    na_gemm_coop_qb_tiled_u4: Option<ComputePipelineState>,
    na_gemm_coop_qb_grouped: Option<ComputePipelineState>,
    na_gemm_coop_qb_grouped_gather: Option<ComputePipelineState>,
    na_gemm_coop_qb_grouped_gate_up_swiglu: Option<ComputePipelineState>,
    na_gemm_coop_qb_grouped_gate_up_swiglu_u4: Option<ComputePipelineState>,
    na_gemm_coop_qb_grouped_scatter: Option<ComputePipelineState>,
    na_gemm_coop_qb_grouped_scatter_u4: Option<ComputePipelineState>,
    moe_coop_gather_padded: Option<ComputePipelineState>,
    moe_coop_scatter_padded: Option<ComputePipelineState>,
    moe_g_fill_u32: Option<ComputePipelineState>,
    moe_g_histogram: Option<ComputePipelineState>,
    moe_g_offsets: Option<ComputePipelineState>,
    moe_g_perm: Option<ComputePipelineState>,
    weight_buffers: Mutex<HashMap<MetalBufferKey, Buffer>>,
    /// Cache des poids transposés bf16 (rhs^T) pour le GEMM NA, par ptr source.
    bf16_rhs_t_cache: Mutex<HashMap<usize, Buffer>>,
    scratch_buffers: Mutex<HashMap<ScratchBufferKey, Buffer>>,
    moe_stacks: Mutex<HashMap<usize, StackedMoeBuffers>>,
}

#[derive(Clone, Debug)]
pub(crate) struct StackedMoeBuffers {
    gate: StackedAffineBuffers,
    up: StackedAffineBuffers,
    down: StackedAffineBuffers,
}

#[derive(Clone, Debug)]
pub(crate) struct StackedAffineBuffers {
    packed: Buffer,
    scales: Buffer,
    biases: Buffer,
    experts: usize,
    out_dim: usize,
    in_dim: usize,
    packed_cols: usize,
    group_size: usize,
    bits: usize,
    groups: usize,
}

#[derive(Clone, Debug)]
pub(crate) enum MetalLinearWeightBuffers {
    Dense {
        rhs: Buffer,
        rhs_bf16: Option<Buffer>,
        out_dim: usize,
        in_dim: usize,
    },
    AffineQuantized {
        packed: Buffer,
        scales: Buffer,
        biases: Buffer,
        out_dim: usize,
        in_dim: usize,
        packed_cols: usize,
        group_size: usize,
        bits: usize,
        groups: usize,
    },
}

#[derive(Clone, Debug)]
pub(crate) enum MetalEmbeddingWeightBuffers {
    Dense {
        table: Buffer,
        vocab: usize,
        dim: usize,
    },
    AffineQuantized {
        packed: Buffer,
        scales: Buffer,
        biases: Buffer,
        vocab: usize,
        dim: usize,
        packed_cols: usize,
        group_size: usize,
        bits: usize,
        groups: usize,
    },
}

#[derive(Clone, Debug)]
pub(crate) struct MetalLinearAttnResidentWeights {
    pub in_proj: MetalLinearWeightBuffers,
    pub out_proj: MetalLinearWeightBuffers,
    pub conv_weight: Buffer,
    pub a_log: Buffer,
    pub dt_bias: Buffer,
    pub norm_weight: Buffer,
}

#[derive(Clone, Debug)]
pub(crate) enum MetalLinearAttnResidentPairWeights {
    Concat(MetalLinearWeightBuffers),
    Split {
        first: MetalLinearWeightBuffers,
        second: MetalLinearWeightBuffers,
    },
}

#[derive(Clone, Debug)]
pub(crate) struct MetalLinearAttnResidentDenseWeights {
    pub full: Option<MetalLinearAttnResidentWeights>,
    pub qkv_z: MetalLinearAttnResidentPairWeights,
    pub beta_gate: MetalLinearAttnResidentPairWeights,
    pub z_beta_gate: Option<MetalLinearWeightBuffers>,
    pub out_proj: MetalLinearWeightBuffers,
    pub conv_weight: Buffer,
    pub a_log: Buffer,
    pub dt_bias: Buffer,
    pub norm_weight: Buffer,
}

#[cfg(any(test, feature = "devtools"))]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct LinearAttentionStateDiff {
    pub conv_max_abs: f32,
    pub conv_mean_abs: f32,
    pub ssm_max_abs: f32,
    pub ssm_mean_abs: f32,
}

#[derive(Clone, Debug)]
pub(crate) struct MetalMoeSharedWeights {
    router: MetalLinearWeightBuffers,
    stacked: StackedMoeBuffers,
    shared_gate: MetalLinearWeightBuffers,
    shared_gate_proj: MetalLinearWeightBuffers,
    shared_up_proj: MetalLinearWeightBuffers,
    shared_down_proj: MetalLinearWeightBuffers,
}

impl MetalMoeSharedWeights {
    /// Renvoie vrai si les experts routés peuvent passer par le chemin coop.
    ///
    /// Prérequis du chemin coop (`encode_moe_shared_rows_coop`) dont les kernels
    /// groupés (`gemm_nax_coop_qb_grouped_*`) erreuraient sinon. Le routage doit
    /// retomber sur le chemin générique quand c'est faux. Le shared passe déjà par
    /// le matmul générique (toute quantification), donc seuls les experts comptent.
    pub(crate) fn coop_compatible(&self) -> bool {
        let bits = self.stacked.gate.bits;
        let ok = |b: &StackedAffineBuffers| b.bits == bits && b.group_size == 64;
        let quant_ok = bits == 8
            || (bits == FAST_QMV_BITS && moe_coop_u4_enabled() && moe_coop_fused_swiglu_enabled());
        ok(&self.stacked.gate) && ok(&self.stacked.up) && ok(&self.stacked.down) && quant_ok
    }
}

#[derive(Clone, Debug)]
pub(crate) struct MetalMoeRoutedWeights {
    router: MetalLinearWeightBuffers,
    stacked: StackedMoeBuffers,
}

impl MetalMoeRoutedWeights {
    /// Renvoie vrai si le tail routed-only peut utiliser les kernels coop groupés.
    ///
    /// Teste la compat coop des seuls experts empilés, sans routeur résolu : le
    /// routeur ne conditionne pas le chemin coop, donc cette décision se prend
    /// avant de résoudre ses buffers pour que le chemin OFF (défaut) ne paie
    /// aucune résolution inutile.
    pub(crate) fn stacked_coop_compatible(stacked: &StackedMoeBuffers) -> bool {
        let bits = stacked.gate.bits;
        let ok = |b: &StackedAffineBuffers| {
            b.bits == bits && b.group_size == 64 && b.in_dim % b.group_size == 0
        };
        let dims_ok = stacked.gate.experts == stacked.up.experts
            && stacked.gate.experts == stacked.down.experts
            && stacked.gate.out_dim == stacked.up.out_dim
            && stacked.gate.in_dim == stacked.up.in_dim
            && stacked.down.in_dim == stacked.gate.out_dim;
        let quant_ok = bits == 8
            || (bits == FAST_QMV_BITS && moe_coop_u4_enabled() && moe_coop_fused_swiglu_enabled());
        ok(&stacked.gate) && ok(&stacked.up) && ok(&stacked.down) && dims_ok && quant_ok
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PrefillAttentionSpec {
    pub seq: usize,
    pub hidden_dim: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rope_dims: usize,
    pub rope_theta: f32,
    pub eps: f32,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct LinearAttentionStepSpec {
    pub num_key_heads: usize,
    pub num_value_heads: usize,
    pub key_head_dim: usize,
    pub value_head_dim: usize,
    pub conv_kernel_dim: usize,
    pub rms_eps: f32,
}

/// Poids d'un pas linear-attn résident (références ; partagés per-op ↔ 1c).
#[derive(Clone, Copy, Debug)]
pub(crate) struct LinearAttnResidentWeights<'a> {
    pub in_proj_qkv: &'a Linear,
    pub in_proj_z: &'a Linear,
    pub in_proj_b: &'a Linear,
    pub in_proj_a: &'a Linear,
    pub out_proj: &'a Linear,
    pub conv_weight: &'a Tensor,
    pub a_log: &'a [f32],
    pub dt_bias: &'a [f32],
    pub norm_weight: &'a [f32],
}

/// Dimensions pré-calculées d'un pas linear-attn résident.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LinearAttnResidentDims {
    pub in_dim: usize,
    pub conv_dim: usize,
    pub value_dim: usize,
    pub key_dim: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct LinearAttentionMetalState {
    conv: Buffer,
    ssm: Buffer,
    conv_len: usize,
    ssm_len: usize,
    conv_dim: usize,
    conv_kernel_dim: usize,
    num_value_heads: usize,
    value_head_dim: usize,
    key_head_dim: usize,
    ssm_bf16: bool,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum PrefillAttentionLayer<'a> {
    Full {
        q_proj: &'a Linear,
        k_proj: &'a Linear,
        v_proj: &'a Linear,
        o_proj: &'a Linear,
        q_norm: &'a Tensor,
        k_norm: &'a Tensor,
        gated: bool,
    },
    Linear {
        weights: LinearAttnResidentWeights<'a>,
        spec: LinearAttentionStepSpec,
        dims: LinearAttnResidentDims,
    },
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum PrefillMoeTail<'a> {
    Dense {
        gate_proj: &'a Linear,
        up_proj: &'a Linear,
        down_proj: &'a Linear,
    },
    Routed {
        router: &'a Linear,
        experts: &'a [GatedMlp],
        top_k: usize,
    },
    Shared {
        router: &'a Linear,
        experts: &'a [GatedMlp],
        top_k: usize,
        shared_expert: &'a GatedMlp,
        shared_gate: &'a Linear,
    },
}

impl PrefillMoeTail<'_> {
    pub(crate) fn top_k(self) -> usize {
        match self {
            Self::Dense { .. } => 1,
            Self::Routed { top_k, .. } | Self::Shared { top_k, .. } => top_k,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PrefillMoeLayer<'a> {
    pub input_norm: &'a Tensor,
    pub attention: PrefillAttentionLayer<'a>,
    pub post_norm: &'a Tensor,
    pub tail: PrefillMoeTail<'a>,
}

#[derive(Debug)]
pub(crate) enum PrefillResidentLayerCache {
    Full { key: Tensor, value: Tensor },
    Linear { state: LinearAttentionMetalState },
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct MetalBufferKey {
    ptr: usize,
    len: usize,
    element: MetalBufferElement,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct ScratchBufferKey {
    label: &'static str,
    len: usize,
    element: MetalBufferElement,
    // Slot de flux (light-batch) : deux flux concurrents qui passent par les
    // mêmes labels reçoivent des buffers DISJOINTS au lieu de s'aliaser.
    namespace: u64,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum MetalBufferElement {
    F32,
    U32,
    Bf16,
}
