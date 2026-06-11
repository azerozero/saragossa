//! Kernels Metal pilotés en Rust pour le backend GPU expérimental.

use crate::{
    AffineQuantizedTensor, EmbeddingWeight, GatedMlp, InferError, Linear, LinearWeight, Result,
    Tensor,
};
use metal::foreign_types::ForeignTypeRef;
use metal::objc::runtime::{sel_registerName, Object, Sel};
use metal::{
    Buffer, BufferRef, CommandQueue, CompileOptions, ComputeCommandEncoderRef,
    ComputePipelineState, Device, MTLCommandBufferStatus, MTLResourceOptions, MTLSize, NSUInteger,
};
use std::cell::Cell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Mutex, OnceLock};

const MATMUL_KERNELS: &str = include_str!("../kernels.metal");
const MAX_MOE_TOP_K: usize = 16;
pub(crate) const MAX_SAMPLER_TOP_K: usize = 32;
const FAST_QMV_GROUP_SIZE: usize = 64;
const FAST_QMV_BITS: usize = 4;

mod attention;
mod attention_checks;
mod buffers;
mod core;
mod linear_attention;
mod linear_attention_encode;
mod lm_head;
mod matmul;
mod matmul_encode;
mod moe;
mod moe_encode;
mod moe_shared;
mod prefill;
#[cfg(test)]
mod tests;

use self::core::*;
pub(crate) use self::core::{
    commit_and_wait, commit_nonblocking, decode_profile_snapshot, install_dispatch_barrier_scope,
    post_dispatch_barrier, profile_dispatch, read_f32_buffer, read_u32_buffer,
    resident_concurrent_enabled, wait_for_completion, EncoderEndGuard,
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
    affine_matmul_rhs_t_u32_f32: ComputePipelineState,
    affine_qmv_fast_u4_gs64_f32: ComputePipelineState,
    affine_qmv_fast_aligned_u4_gs64_f32: ComputePipelineState,
    #[allow(
        dead_code,
        reason = "consommé par le verify spéculatif MTP à l'ÉTAPE 3"
    )]
    affine_qmm2_fast_aligned_u4_gs64_f32: ComputePipelineState,
    affine_qmv_fast_aligned_u8_gs64_f32: ComputePipelineState,
    affine_qkv_split_qmv_fast_u4_gs64_f32: ComputePipelineState,
    affine_qmv_rms_fast_u4_gs64_f32: ComputePipelineState,
    affine_qkv_split_rms_qmv_fast_u4_gs64_f32: ComputePipelineState,
    affine_qmv_gated_input_fast_u4_gs64_f32: ComputePipelineState,
    embed_gather_dense_from_u32_f32: ComputePipelineState,
    embed_gather_affine_from_u32_f32: ComputePipelineState,
    swiglu_f32: ComputePipelineState,
    accumulate_scaled_f32: ComputePipelineState,
    linear_attn_conv_silu_f32: ComputePipelineState,
    linear_attn_norm_gates_f32: ComputePipelineState,
    linear_attn_gated_delta_f32: ComputePipelineState,
    linear_attn_rms_gate_f32: ComputePipelineState,
    affine_gather_matmul_rhs_t_u32_f32: ComputePipelineState,
    affine_gather_qmv_fast_u4_gs64_f32: ComputePipelineState,
    affine_gather_qmv_tail_u4_gs64_f32: ComputePipelineState,
    affine_gather_gate_up_swiglu_fast_u4_gs64_f32: ComputePipelineState,
    affine_gate_up_swiglu_fast_u4_gs64_f32: ComputePipelineState,
    affine_argmax_qmv_fast_u4_gs64_f32: ComputePipelineState,
    weighted_sum_topk_f32: ComputePipelineState,
    weighted_sum_grouped_topk_f32: ComputePipelineState,
    weighted_sum_add_grouped_topk_f32: ComputePipelineState,
    weighted_sum_add_topk_f32: ComputePipelineState,
    add_sigmoid_scaled_f32: ComputePipelineState,
    copy_f32: ComputePipelineState,
    rms_norm_rows_f32: ComputePipelineState,
    add_rms_norm_rows_f32: ComputePipelineState,
    rms_norm_rope_heads_f32: ComputePipelineState,
    causal_attention_prefill_f32: ComputePipelineState,
    add_rms_norm_row_f32: ComputePipelineState,
    topk_softmax_f32: ComputePipelineState,
    topk_softmax_serial_f32: ComputePipelineState,
    topk_softmax_rows_f32: ComputePipelineState,
    sample_topk_blocks_f32: ComputePipelineState,
    sample_topk_finalize_f32: ComputePipelineState,
    argmax_blocks_f32: ComputePipelineState,
    argmax_finalize_f32: ComputePipelineState,
    weight_buffers: Mutex<HashMap<MetalBufferKey, Buffer>>,
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
    pub out_proj: MetalLinearWeightBuffers,
    pub conv_weight: Buffer,
    pub a_log: Buffer,
    pub dt_bias: Buffer,
    pub norm_weight: Buffer,
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

#[derive(Clone, Debug)]
pub(crate) struct MetalMoeRoutedWeights {
    router: MetalLinearWeightBuffers,
    stacked: StackedMoeBuffers,
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
#[derive(Clone, Copy)]
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
#[derive(Clone, Copy)]
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
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PrefillMoeLayer<'a> {
    pub input_norm: &'a Tensor,
    pub q_proj: &'a Linear,
    pub k_proj: &'a Linear,
    pub v_proj: &'a Linear,
    pub o_proj: &'a Linear,
    pub q_norm: &'a Tensor,
    pub k_norm: &'a Tensor,
    pub post_norm: &'a Tensor,
    pub router: &'a Linear,
    pub experts: &'a [GatedMlp],
    pub top_k: usize,
}

#[derive(Clone, Copy, Debug)]
enum MoeProjection {
    Gate,
    Up,
    Down,
}

impl MoeProjection {
    fn affine_weight(self, expert: &GatedMlp) -> Result<&AffineQuantizedTensor> {
        let (gate, up, down) = expert.projections();
        let linear = match self {
            Self::Gate => gate,
            Self::Up => up,
            Self::Down => down,
        };
        ensure_biasless(linear, self.name())?;
        match linear.weight() {
            LinearWeight::AffineQuantized(weight) => Ok(weight),
            LinearWeight::Dense(_) => Err(InferError::Config(format!(
                "expert MoE {} non affine quantifié",
                self.name()
            ))),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Gate => "gate",
            Self::Up => "up",
            Self::Down => "down",
        }
    }

    fn packed_label(self) -> &'static str {
        match self {
            Self::Gate => "moe_gate_packed",
            Self::Up => "moe_up_packed",
            Self::Down => "moe_down_packed",
        }
    }

    fn scales_label(self) -> &'static str {
        match self {
            Self::Gate => "moe_gate_scales",
            Self::Up => "moe_up_scales",
            Self::Down => "moe_down_scales",
        }
    }

    fn biases_label(self) -> &'static str {
        match self {
            Self::Gate => "moe_gate_biases",
            Self::Up => "moe_up_biases",
            Self::Down => "moe_down_biases",
        }
    }
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
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum MetalBufferElement {
    F32,
    U32,
    Bf16,
}

fn pipeline(library: &metal::Library, device: &Device, name: &str) -> Result<ComputePipelineState> {
    let function = library
        .get_function(name, None)
        .map_err(|message| InferError::Metal(format!("kernel {name}: {message}")))?;
    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|message| InferError::Metal(format!("pipeline {name}: {message}")))
}

impl MetalExecutor {
    /// Crée le device Metal et compile les kernels embarqués.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si Metal est indisponible ou si la compilation échoue.
    pub fn new() -> Result<Self> {
        let device = Device::system_default()
            .ok_or_else(|| InferError::Metal("aucun device Metal disponible".to_string()))?;
        let queue = device.new_command_queue();
        let compile_options = CompileOptions::new();
        compile_options.set_fast_math_enabled(true);
        let library = device
            .new_library_with_source(MATMUL_KERNELS, &compile_options)
            .map_err(|message| InferError::Metal(format!("compile kernels: {message}")))?;
        let dense_matmul_rhs_t_f32 = pipeline(&library, &device, "dense_matmul_rhs_t_f32")?;
        let affine_matmul_rhs_t_u32_f32 =
            pipeline(&library, &device, "affine_matmul_rhs_t_u32_f32")?;
        let affine_qmv_fast_u4_gs64_f32 =
            pipeline(&library, &device, "affine_qmv_fast_u4_gs64_f32")?;
        let affine_qmv_fast_aligned_u4_gs64_f32 =
            pipeline(&library, &device, "affine_qmv_fast_aligned_u4_gs64_f32")?;
        let affine_qmm2_fast_aligned_u4_gs64_f32 =
            pipeline(&library, &device, "affine_qmm2_fast_aligned_u4_gs64_f32")?;
        let affine_qmv_fast_aligned_u8_gs64_f32 =
            pipeline(&library, &device, "affine_qmv_fast_aligned_u8_gs64_f32")?;
        let affine_qkv_split_qmv_fast_u4_gs64_f32 =
            pipeline(&library, &device, "affine_qkv_split_qmv_fast_u4_gs64_f32")?;
        let affine_qmv_rms_fast_u4_gs64_f32 =
            pipeline(&library, &device, "affine_qmv_rms_fast_u4_gs64_f32")?;
        let affine_qkv_split_rms_qmv_fast_u4_gs64_f32 = pipeline(
            &library,
            &device,
            "affine_qkv_split_rms_qmv_fast_u4_gs64_f32",
        )?;
        let affine_qmv_gated_input_fast_u4_gs64_f32 =
            pipeline(&library, &device, "affine_qmv_gated_input_fast_u4_gs64_f32")?;
        let embed_gather_dense_from_u32_f32 =
            pipeline(&library, &device, "embed_gather_dense_from_u32_f32")?;
        let embed_gather_affine_from_u32_f32 =
            pipeline(&library, &device, "embed_gather_affine_from_u32_f32")?;
        let swiglu_f32 = pipeline(&library, &device, "swiglu_f32")?;
        let accumulate_scaled_f32 = pipeline(&library, &device, "accumulate_scaled_f32")?;
        let linear_attn_conv_silu_f32 = pipeline(&library, &device, "linear_attn_conv_silu_f32")?;
        let linear_attn_norm_gates_f32 = pipeline(&library, &device, "linear_attn_norm_gates_f32")?;
        let linear_attn_gated_delta_f32 =
            pipeline(&library, &device, "linear_attn_gated_delta_f32")?;
        let linear_attn_rms_gate_f32 = pipeline(&library, &device, "linear_attn_rms_gate_f32")?;
        let affine_gather_matmul_rhs_t_u32_f32 =
            pipeline(&library, &device, "affine_gather_matmul_rhs_t_u32_f32")?;
        let affine_gather_qmv_fast_u4_gs64_f32 =
            pipeline(&library, &device, "affine_gather_qmv_fast_u4_gs64_f32")?;
        let affine_gather_qmv_tail_u4_gs64_f32 =
            pipeline(&library, &device, "affine_gather_qmv_tail_u4_gs64_f32")?;
        let affine_gather_gate_up_swiglu_fast_u4_gs64_f32 = pipeline(
            &library,
            &device,
            "affine_gather_gate_up_swiglu_fast_u4_gs64_f32",
        )?;
        let affine_gate_up_swiglu_fast_u4_gs64_f32 =
            pipeline(&library, &device, "affine_gate_up_swiglu_fast_u4_gs64_f32")?;
        let affine_argmax_qmv_fast_u4_gs64_f32 =
            pipeline(&library, &device, "affine_argmax_qmv_fast_u4_gs64_f32")?;
        let weighted_sum_topk_f32 = pipeline(&library, &device, "weighted_sum_topk_f32")?;
        let weighted_sum_grouped_topk_f32 =
            pipeline(&library, &device, "weighted_sum_grouped_topk_f32")?;
        let weighted_sum_add_grouped_topk_f32 =
            pipeline(&library, &device, "weighted_sum_add_grouped_topk_f32")?;
        let weighted_sum_add_topk_f32 = pipeline(&library, &device, "weighted_sum_add_topk_f32")?;
        let add_sigmoid_scaled_f32 = pipeline(&library, &device, "add_sigmoid_scaled_f32")?;
        let copy_f32 = pipeline(&library, &device, "copy_f32")?;
        let rms_norm_rows_f32 = pipeline(&library, &device, "rms_norm_rows_f32")?;
        let add_rms_norm_rows_f32 = pipeline(&library, &device, "add_rms_norm_rows_f32")?;
        let rms_norm_rope_heads_f32 = pipeline(&library, &device, "rms_norm_rope_heads_f32")?;
        let causal_attention_prefill_f32 =
            pipeline(&library, &device, "causal_attention_prefill_f32")?;
        let add_rms_norm_row_f32 = pipeline(&library, &device, "add_rms_norm_row_f32")?;
        let topk_softmax_f32 = pipeline(&library, &device, "topk_softmax_f32")?;
        let topk_softmax_serial_f32 = pipeline(&library, &device, "topk_softmax_serial_f32")?;
        let topk_softmax_rows_f32 = pipeline(&library, &device, "topk_softmax_rows_f32")?;
        let sample_topk_blocks_f32 = pipeline(&library, &device, "sample_topk_blocks_f32")?;
        let sample_topk_finalize_f32 = pipeline(&library, &device, "sample_topk_finalize_f32")?;
        let argmax_blocks_f32 = pipeline(&library, &device, "argmax_blocks_f32")?;
        let argmax_finalize_f32 = pipeline(&library, &device, "argmax_finalize_f32")?;

        Ok(Self {
            device,
            queue,
            dense_matmul_rhs_t_f32,
            affine_matmul_rhs_t_u32_f32,
            affine_qmv_fast_u4_gs64_f32,
            affine_qmv_fast_aligned_u4_gs64_f32,
            affine_qmm2_fast_aligned_u4_gs64_f32,
            affine_qmv_fast_aligned_u8_gs64_f32,
            affine_qkv_split_qmv_fast_u4_gs64_f32,
            affine_qmv_rms_fast_u4_gs64_f32,
            affine_qkv_split_rms_qmv_fast_u4_gs64_f32,
            affine_qmv_gated_input_fast_u4_gs64_f32,
            embed_gather_dense_from_u32_f32,
            embed_gather_affine_from_u32_f32,
            swiglu_f32,
            accumulate_scaled_f32,
            linear_attn_conv_silu_f32,
            linear_attn_norm_gates_f32,
            linear_attn_gated_delta_f32,
            linear_attn_rms_gate_f32,
            affine_gather_matmul_rhs_t_u32_f32,
            affine_gather_qmv_fast_u4_gs64_f32,
            affine_gather_qmv_tail_u4_gs64_f32,
            affine_gather_gate_up_swiglu_fast_u4_gs64_f32,
            affine_gate_up_swiglu_fast_u4_gs64_f32,
            affine_argmax_qmv_fast_u4_gs64_f32,
            weighted_sum_topk_f32,
            weighted_sum_grouped_topk_f32,
            weighted_sum_add_grouped_topk_f32,
            weighted_sum_add_topk_f32,
            add_sigmoid_scaled_f32,
            copy_f32,
            rms_norm_rows_f32,
            add_rms_norm_rows_f32,
            rms_norm_rope_heads_f32,
            causal_attention_prefill_f32,
            add_rms_norm_row_f32,
            topk_softmax_f32,
            topk_softmax_serial_f32,
            topk_softmax_rows_f32,
            sample_topk_blocks_f32,
            sample_topk_finalize_f32,
            argmax_blocks_f32,
            argmax_finalize_f32,
            weight_buffers: Mutex::new(HashMap::new()),
            scratch_buffers: Mutex::new(HashMap::new()),
            moe_stacks: Mutex::new(HashMap::new()),
        })
    }
}
