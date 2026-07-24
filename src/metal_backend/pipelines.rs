//! Construction des pipelines Metal.

use super::*;
use metal::{
    CompileOptions, ComputePipelineState, Device, FunctionConstantValues, MTLDataType, NSUInteger,
};
use std::borrow::Cow;
use std::ffi::c_void;
use std::path::{Path, PathBuf};

fn pipeline(library: &metal::Library, device: &Device, name: &str) -> Result<ComputePipelineState> {
    let function = library
        .get_function(name, None)
        .map_err(|message| InferError::Metal(format!("kernel {name}: {message}")))?;
    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|message| InferError::Metal(format!("pipeline {name}: {message}")))
}

#[derive(Debug)]
pub(super) struct KernelSources {
    pub(super) matmul: Cow<'static, str>,
    pub(super) na_gemm: Cow<'static, str>,
    pub(super) steel_attention: Cow<'static, str>,
}

impl KernelSources {
    fn load() -> Result<Self> {
        Self::from_runtime_path(std::env::var_os("RETI_RUST_KERNELS_PATH").map(PathBuf::from))
    }

    pub(super) fn from_runtime_path(path: Option<PathBuf>) -> Result<Self> {
        let Some(root) = path else {
            return Ok(Self {
                matmul: Cow::Borrowed(MATMUL_KERNELS),
                na_gemm: Cow::Borrowed(NA_GEMM_SRC),
                steel_attention: Cow::Borrowed(STEEL_ATTENTION_KERNELS),
            });
        };
        Ok(Self {
            matmul: Cow::Owned(read_runtime_kernel(&root, &["kernels.metal"])?),
            na_gemm: Cow::Owned(read_runtime_kernel(&root, &["na_gemm", "na_gemm.metal"])?),
            steel_attention: Cow::Owned(read_runtime_kernel(
                &root,
                &["steel_attention", "steel_attention.metal"],
            )?),
        })
    }
}

fn read_runtime_kernel(root: &Path, names: &[&str]) -> Result<String> {
    for name in names {
        let path = root.join(name);
        if path.is_file() {
            let bytes = std::fs::read(&path).map_err(|source| InferError::Io {
                path: path.clone(),
                source,
            })?;
            eprintln!(
                "source runtime : {} (md5 {:x})",
                path.display(),
                md5::compute(&bytes)
            );
            return String::from_utf8(bytes).map_err(|source| {
                InferError::Config(format!(
                    "source runtime {} non UTF-8: {source}",
                    path.display()
                ))
            });
        }
    }
    Err(InferError::Config(format!(
        "RETI_RUST_KERNELS_PATH={} ne contient aucun de: {}",
        root.display(),
        names.join(", ")
    )))
}

/// Compile le GEMM NA dans une bibliothèque SÉPARÉE (MPP `#include`). `None` si la
/// MetalPerformancePrimitives est absente (macOS < 26) → fallback f32.
fn compile_na_gemm(device: &Device, source: &str) -> Option<ComputePipelineState> {
    let options = CompileOptions::new();
    options.set_fast_math_enabled(true);
    let library = device.new_library_with_source(source, &options).ok()?;
    let function = library.get_function("gemm_nax", None).ok()?;
    device
        .new_compute_pipeline_state_with_function(&function)
        .ok()
}

fn compile_na_gemm_named(
    device: &Device,
    source: &str,
    name: &str,
) -> Option<ComputePipelineState> {
    let options = CompileOptions::new();
    options.set_fast_math_enabled(true);
    let library = device.new_library_with_source(source, &options).ok()?;
    let function = library.get_function(name, None).ok()?;
    device
        .new_compute_pipeline_state_with_function(&function)
        .ok()
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn compile_steel_attention(device: &Device, source: &str) -> Option<ComputePipelineState> {
    if matches!(
        std::env::var("RETI_STT_STEEL_ATTN").as_deref(),
        Ok("0" | "false" | "off" | "no")
    ) {
        return None;
    }
    let options = CompileOptions::new();
    options.set_fast_math_enabled(true);
    let library = device.new_library_with_source(source, &options).ok()?;
    let constants = FunctionConstantValues::new();
    set_metal_bool_constant(&constants, 200, false);
    set_metal_bool_constant(&constants, 201, false);
    set_metal_bool_constant(&constants, 300, false);
    set_metal_bool_constant(&constants, 301, false);
    set_metal_bool_constant(&constants, 302, false);
    let function = library
        .get_function(STEEL_ATTN_F32_BQ32_BK32_BD64, Some(constants))
        .ok()?;
    device
        .new_compute_pipeline_state_with_function(&function)
        .ok()
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn compile_steel_causal_d256_attention(
    device: &Device,
    source: &str,
) -> Option<ComputePipelineState> {
    let options = CompileOptions::new();
    options.set_fast_math_enabled(true);
    let library = match device.new_library_with_source(source, &options) {
        Ok(library) => library,
        Err(message) => {
            trace_steel_causal_d256_compile_error("compile", &message);
            return None;
        }
    };
    let constants = FunctionConstantValues::new();
    set_metal_bool_constant(&constants, 200, false);
    set_metal_bool_constant(&constants, 201, false);
    set_metal_bool_constant(&constants, 300, false);
    set_metal_bool_constant(&constants, 301, true);
    set_metal_bool_constant(&constants, 302, false);
    let function =
        match library.get_function(STEEL_CAUSAL_ATTN_D256_F32_BQ32_BK64_BD64X4, Some(constants)) {
            Ok(function) => function,
            Err(message) => {
                trace_steel_causal_d256_compile_error("function", &message);
                return None;
            }
        };
    match device.new_compute_pipeline_state_with_function(&function) {
        Ok(pipeline) => Some(pipeline),
        Err(message) => {
            trace_steel_causal_d256_compile_error("pipeline", &message);
            None
        }
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn trace_steel_causal_d256_compile_error(stage: &str, message: &str) {
    if crate::runtime_flags::trace_prefill_enabled() {
        eprintln!("steel causal d256 {stage}: {message}");
    }
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
fn compile_steel_attention(_device: &Device, _source: &str) -> Option<ComputePipelineState> {
    None
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
fn compile_steel_causal_d256_attention(
    _device: &Device,
    _source: &str,
) -> Option<ComputePipelineState> {
    None
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn set_metal_bool_constant(
    constants: &metal::FunctionConstantValuesRef,
    index: NSUInteger,
    value: bool,
) {
    let raw = u8::from(value);
    constants.set_constant_value_at_index(
        (&raw as *const u8).cast::<c_void>(),
        MTLDataType::Bool,
        index,
    );
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
        let sources = KernelSources::load()?;
        let compile_options = CompileOptions::new();
        compile_options.set_fast_math_enabled(true);
        let library = device
            .new_library_with_source(sources.matmul.as_ref(), &compile_options)
            .map_err(|message| InferError::Metal(format!("compile kernels: {message}")))?;
        let dense_matmul_rhs_t_f32 = pipeline(&library, &device, "dense_matmul_rhs_t_f32")?;
        let dense_qmv_rhs_bf16_f32 = pipeline(&library, &device, "dense_qmv_rhs_bf16_f32")?;
        let dense_qmv_fast_f32 = pipeline(&library, &device, "dense_qmv_fast_f32")?;
        let dense_gemm_rhs_t_f32 = pipeline(&library, &device, "dense_gemm_rhs_t_f32")?;
        let affine_matmul_rhs_t_u32_f32 =
            pipeline(&library, &device, "affine_matmul_rhs_t_u32_f32")?;
        let affine_qmv_fast_u4_gs64_f32 =
            pipeline(&library, &device, "affine_qmv_fast_u4_gs64_f32")?;
        let affine_qmv_fast_aligned_u4_gs64_f32 =
            pipeline(&library, &device, "affine_qmv_fast_aligned_u4_gs64_f32")?;
        let affine_qmv_fast_u4_gs64_align64_f32 =
            pipeline(&library, &device, "affine_qmv_fast_u4_gs64_align64_f32")?;
        let affine_qmv_fast_u6_gs64_f32 =
            pipeline(&library, &device, "affine_qmv_fast_u6_gs64_f32")?;
        let affine_qmv_fast_aligned_u6_gs64_f32 =
            pipeline(&library, &device, "affine_qmv_fast_aligned_u6_gs64_f32")?;
        let affine_qmm2_fast_aligned_u4_gs64_f32 =
            pipeline(&library, &device, "affine_qmm2_fast_aligned_u4_gs64_f32")?;
        let affine_qmm2_fast_aligned_u8_gs64_f32 =
            pipeline(&library, &device, "affine_qmm2_fast_aligned_u8_gs64_f32")?;
        let affine_qmm2_fast_aligned_u8_gs128_f32 =
            pipeline(&library, &device, "affine_qmm2_fast_aligned_u8_gs128_f32")?;
        let affine_qmv_fast_aligned_u8_gs64_f32 =
            pipeline(&library, &device, "affine_qmv_fast_aligned_u8_gs64_f32")?;
        let affine_qmv_fast_u8_gs64_align64_f32 =
            pipeline(&library, &device, "affine_qmv_fast_u8_gs64_align64_f32")?;
        let affine_qmv_fast_aligned_u8_gs128_f32 =
            pipeline(&library, &device, "affine_qmv_fast_aligned_u8_gs128_f32")?;
        let affine_qmv_fast_aligned_u8_gs64_dot4_f32 = pipeline(
            &library,
            &device,
            "affine_qmv_fast_aligned_u8_gs64_dot4_f32",
        )?;
        let affine_qmv_fast_aligned_u8_gs128_dot4_f32 = pipeline(
            &library,
            &device,
            "affine_qmv_fast_aligned_u8_gs128_dot4_f32",
        )?;
        let affine_qmv_fast_aligned_u8_gs64_tg128_f32 = pipeline(
            &library,
            &device,
            "affine_qmv_fast_aligned_u8_gs64_tg128_f32",
        )?;
        let affine_qmv_fast_aligned_u8_gs128_tg128_f32 = pipeline(
            &library,
            &device,
            "affine_qmv_fast_aligned_u8_gs128_tg128_f32",
        )?;
        let affine_qmv_fast_aligned_u8_gs64_tg256_f32 = pipeline(
            &library,
            &device,
            "affine_qmv_fast_aligned_u8_gs64_tg256_f32",
        )?;
        let affine_qmv_fast_aligned_u8_gs128_tg256_f32 = pipeline(
            &library,
            &device,
            "affine_qmv_fast_aligned_u8_gs128_tg256_f32",
        )?;
        let affine_qmv_plus_one_fast_aligned_u8_gs64_f32 = pipeline(
            &library,
            &device,
            "affine_qmv_plus_one_fast_aligned_u8_gs64_f32",
        )?;
        let affine_qmv_one_fast_u8_gs64_f32 =
            pipeline(&library, &device, "affine_qmv_one_fast_u8_gs64_f32")?;
        let affine_qkv_split_qmv_fast_u4_gs64_f32 =
            pipeline(&library, &device, "affine_qkv_split_qmv_fast_u4_gs64_f32")?;
        let affine_qmv_rms_fast_u4_gs64_f32 =
            pipeline(&library, &device, "affine_qmv_rms_fast_u4_gs64_f32")?;
        let affine_qmv_rms_fast_u8_gs64_f32 =
            pipeline(&library, &device, "affine_qmv_rms_fast_u8_gs64_f32")?;
        let affine_qmv_rms_fast_u8_gs128_f32 =
            pipeline(&library, &device, "affine_qmv_rms_fast_u8_gs128_f32")?;
        let affine_qkv_split_rms_qmv_fast_u4_gs64_f32 = pipeline(
            &library,
            &device,
            "affine_qkv_split_rms_qmv_fast_u4_gs64_f32",
        )?;
        let affine_qkv_split_rms_qmv_fast_u8_gs64_f32 = pipeline(
            &library,
            &device,
            "affine_qkv_split_rms_qmv_fast_u8_gs64_f32",
        )?;
        let affine_qmv_gated_input_fast_u4_gs64_f32 =
            pipeline(&library, &device, "affine_qmv_gated_input_fast_u4_gs64_f32")?;
        let affine_qmv_gated_input_fast_u8_gs64_f32 =
            pipeline(&library, &device, "affine_qmv_gated_input_fast_u8_gs64_f32")?;
        let embed_gather_dense_from_u32_f32 =
            pipeline(&library, &device, "embed_gather_dense_from_u32_f32")?;
        let embed_gather_affine_from_u32_f32 =
            pipeline(&library, &device, "embed_gather_affine_from_u32_f32")?;
        let swiglu_f32 = pipeline(&library, &device, "swiglu_f32")?;
        let geglu_tanh_f32 = pipeline(&library, &device, "geglu_tanh_f32")?;
        let split_q_gate_rows_f32 = pipeline(&library, &device, "split_q_gate_rows_f32")?;
        let attn_gate_rows_f32 = pipeline(&library, &device, "attn_gate_rows_f32")?;
        let accumulate_scaled_f32 = pipeline(&library, &device, "accumulate_scaled_f32")?;
        let add_scaled_f32 = pipeline(&library, &device, "add_scaled_f32")?;
        let linear_attn_conv_silu_f32 = pipeline(&library, &device, "linear_attn_conv_silu_f32")?;
        let linear_attn_conv_silu_k4_f32 =
            pipeline(&library, &device, "linear_attn_conv_silu_k4_f32")?;
        let linear_attn_norm_gates_f32 = pipeline(&library, &device, "linear_attn_norm_gates_f32")?;
        let linear_attn_norm_gates_dk128_f32 =
            pipeline(&library, &device, "linear_attn_norm_gates_dk128_f32")?;
        let linear_attn_norm_gates_inv_dk128_f32 =
            pipeline(&library, &device, "linear_attn_norm_gates_inv_dk128_f32")?;
        let linear_attn_conv_norm_gates_k4_dk128_f32 = pipeline(
            &library,
            &device,
            "linear_attn_conv_norm_gates_k4_dk128_f32",
        )?;
        let linear_attn_conv_norm_gates_k4_dk128_batch_f32 = pipeline(
            &library,
            &device,
            "linear_attn_conv_norm_gates_k4_dk128_batch_f32",
        )?;
        let linear_attn_conv_state_finalize_f32 =
            pipeline(&library, &device, "linear_attn_conv_state_finalize_f32")?;
        let linear_attn_gated_delta_f32 =
            pipeline(&library, &device, "linear_attn_gated_delta_f32")?;
        let linear_attn_gated_delta_dk128_tg4_f32 =
            pipeline(&library, &device, "linear_attn_gated_delta_dk128_tg4_f32")?;
        let linear_attn_gated_delta_seq_dk128_tg4_f32 = pipeline(
            &library,
            &device,
            "linear_attn_gated_delta_seq_dk128_tg4_f32",
        )?;
        let linear_attn_gated_delta_seq_dk128_bf16_tg4_f32 = pipeline(
            &library,
            &device,
            "linear_attn_gated_delta_seq_dk128_bf16_tg4_f32",
        )?;
        let linear_attn_gated_delta_dk128_bf16_tg4_f32 = pipeline(
            &library,
            &device,
            "linear_attn_gated_delta_dk128_bf16_tg4_f32",
        )?;
        let linear_attn_gated_delta_inv_dk128_tg4_f32 = pipeline(
            &library,
            &device,
            "linear_attn_gated_delta_inv_dk128_tg4_f32",
        )?;
        let linear_attn_rms_gate_f32 = pipeline(&library, &device, "linear_attn_rms_gate_f32")?;
        let linear_attn_rms_gate_dv128_f32 =
            pipeline(&library, &device, "linear_attn_rms_gate_dv128_f32")?;
        let linear_attn_rms_gate_batch_dv128_f32 =
            pipeline(&library, &device, "linear_attn_rms_gate_batch_dv128_f32")?;
        let affine_gather_matmul_rhs_t_u32_f32 =
            pipeline(&library, &device, "affine_gather_matmul_rhs_t_u32_f32")?;
        let affine_gather_qmv_fast_u4_gs64_f32 =
            pipeline(&library, &device, "affine_gather_qmv_fast_u4_gs64_f32")?;
        let affine_gather_qmv_fast_u8_gs64_f32 =
            pipeline(&library, &device, "affine_gather_qmv_fast_u8_gs64_f32")?;
        let affine_gather_qmv_fast_u8_gs128_f32 =
            pipeline(&library, &device, "affine_gather_qmv_fast_u8_gs128_f32")?;
        let affine_gather_qmv_fast_u8_gs64_tg128_f32 = pipeline(
            &library,
            &device,
            "affine_gather_qmv_fast_u8_gs64_tg128_f32",
        )?;
        let affine_gather_qmv_fast_u8_gs128_tg128_f32 = pipeline(
            &library,
            &device,
            "affine_gather_qmv_fast_u8_gs128_tg128_f32",
        )?;
        let affine_gather_qmv_fast_u8_gs64_tg256_f32 = pipeline(
            &library,
            &device,
            "affine_gather_qmv_fast_u8_gs64_tg256_f32",
        )?;
        let affine_gather_qmv_fast_u8_gs128_tg256_f32 = pipeline(
            &library,
            &device,
            "affine_gather_qmv_fast_u8_gs128_tg256_f32",
        )?;
        let affine_gather_qmv_tail_u4_gs64_f32 =
            pipeline(&library, &device, "affine_gather_qmv_tail_u4_gs64_f32")?;
        let affine_gather_gate_up_swiglu_fast_u4_gs64_f32 = pipeline(
            &library,
            &device,
            "affine_gather_gate_up_swiglu_fast_u4_gs64_f32",
        )?;
        let affine_gather_gate_up_swiglu_fast_u8_gs64_f32 = pipeline(
            &library,
            &device,
            "affine_gather_gate_up_swiglu_fast_u8_gs64_f32",
        )?;
        let affine_gather_gate_up_swiglu_fast_u8_gs128_f32 = pipeline(
            &library,
            &device,
            "affine_gather_gate_up_swiglu_fast_u8_gs128_f32",
        )?;
        let affine_gate_up_swiglu_fast_u4_gs64_f32 =
            pipeline(&library, &device, "affine_gate_up_swiglu_fast_u4_gs64_f32")?;
        let affine_gate_up_swiglu_fast_u8_gs64_f32 =
            pipeline(&library, &device, "affine_gate_up_swiglu_fast_u8_gs64_f32")?;
        let affine_gate_up_swiglu_gate_fast_u8_gs64_f32 = pipeline(
            &library,
            &device,
            "affine_gate_up_swiglu_gate_fast_u8_gs64_f32",
        )?;
        let affine_gate_up_swiglu_gate_fast_u8_gs128_f32 = pipeline(
            &library,
            &device,
            "affine_gate_up_swiglu_gate_fast_u8_gs128_f32",
        )?;
        let affine_gate_up_swiglu_fast_u8_gs128_f32 =
            pipeline(&library, &device, "affine_gate_up_swiglu_fast_u8_gs128_f32")?;
        let affine_argmax_qmv_fast_u4_gs64_f32 =
            pipeline(&library, &device, "affine_argmax_qmv_fast_u4_gs64_f32")?;
        let weighted_sum_topk_f32 = pipeline(&library, &device, "weighted_sum_topk_f32")?;
        let scale_topk_scores_f32 = pipeline(&library, &device, "scale_topk_scores_f32")?;
        let weighted_sum_grouped_topk_f32 =
            pipeline(&library, &device, "weighted_sum_grouped_topk_f32")?;
        let weighted_sum_add_grouped_topk_f32 =
            pipeline(&library, &device, "weighted_sum_add_grouped_topk_f32")?;
        let weighted_sum_add_topk_f32 = pipeline(&library, &device, "weighted_sum_add_topk_f32")?;
        let weighted_sum_add_shared_topk_f32 =
            pipeline(&library, &device, "weighted_sum_add_shared_topk_f32")?;
        let affine_gather_down_weighted_shared_fast_u8_gs64_f32 = pipeline(
            &library,
            &device,
            "affine_gather_down_weighted_shared_fast_u8_gs64_f32",
        )?;
        let add_sigmoid_scaled_f32 = pipeline(&library, &device, "add_sigmoid_scaled_f32")?;
        let add_sigmoid_scaled_rows_f32 =
            pipeline(&library, &device, "add_sigmoid_scaled_rows_f32")?;
        let copy_f32 = pipeline(&library, &device, "copy_f32")?;
        let copy_u16 = pipeline(&library, &device, "copy_u16")?;
        let rms_norm_rows_f32 = pipeline(&library, &device, "rms_norm_rows_f32")?;
        let rms_norm_simd_rows_f32 = pipeline(&library, &device, "rms_norm_simd_rows_f32")?;
        let add_rms_norm_rows_f32 = pipeline(&library, &device, "add_rms_norm_rows_f32")?;
        let layer_norm_rows_f32 = pipeline(&library, &device, "layer_norm_rows_f32")?;
        let add_layer_norm_rows_f32 = pipeline(&library, &device, "add_layer_norm_rows_f32")?;
        let gelu_f32 = pipeline(&library, &device, "gelu_f32")?;
        let layer_norm_rows_f32_bf16out =
            pipeline(&library, &device, "layer_norm_rows_f32_bf16out")?;
        let add_layer_norm_rows_f32_bf16out =
            pipeline(&library, &device, "add_layer_norm_rows_f32_bf16out")?;
        let gelu_f32_bf16out = pipeline(&library, &device, "gelu_f32_bf16out")?;
        let add_row_bias_f32 = pipeline(&library, &device, "add_row_bias_f32")?;
        let whisper_attn_decode_f32 = pipeline(&library, &device, "whisper_attn_decode_f32")?;
        let whisper_attn_decode_vec64_f32 =
            pipeline(&library, &device, "whisper_attn_decode_vec64_f32")?;
        let im2col_f32 = pipeline(&library, &device, "im2col_f32")?;
        let rms_norm_rope_heads_f32 = pipeline(&library, &device, "rms_norm_rope_heads_f32")?;
        let rms_norm_heads_no_scale_f32 =
            pipeline(&library, &device, "rms_norm_heads_no_scale_f32")?;
        let causal_attention_prefill_f32 =
            pipeline(&library, &device, "causal_attention_prefill_f32")?;
        let causal_attention_prefill_mid_f32 =
            pipeline(&library, &device, "causal_attention_prefill_mid_f32")?;
        let causal_attention_prefill_long_f32 =
            pipeline(&library, &device, "causal_attention_prefill_long_f32")?;
        let windowed_attention_prefill_f32 =
            pipeline(&library, &device, "windowed_attention_prefill_f32")?;
        let windowed_attention_prefill_mid_f32 =
            pipeline(&library, &device, "windowed_attention_prefill_mid_f32")?;
        let windowed_attention_prefill_long_f32 =
            pipeline(&library, &device, "windowed_attention_prefill_long_f32")?;
        let causal_attention_prefill_batch_long_d128_f32 = pipeline(
            &library,
            &device,
            "causal_attention_prefill_batch_long_d128_f32",
        )?;
        let causal_attention_prefill_batch_long_d256_f32 = pipeline(
            &library,
            &device,
            "causal_attention_prefill_batch_long_d256_f32",
        )?;
        let causal_attention_prefill_batch_gqa8x4_d256_f32 = pipeline(
            &library,
            &device,
            "causal_attention_prefill_batch_gqa8x4_d256_f32",
        )?;
        let causal_attention_prefill_steel_d256_f32 =
            compile_steel_causal_d256_attention(&device, sources.steel_attention.as_ref());
        let noncausal_attention_prefill_f32 =
            pipeline(&library, &device, "noncausal_attention_prefill_f32")?;
        let steel_attention_f32_bq32_bk32_bd64 =
            compile_steel_attention(&device, sources.steel_attention.as_ref());
        let add_rms_norm_row_f32 = pipeline(&library, &device, "add_rms_norm_row_f32")?;
        let topk_softmax_f32 = pipeline(&library, &device, "topk_softmax_f32")?;
        let topk_softmax_serial_f32 = pipeline(&library, &device, "topk_softmax_serial_f32")?;
        let topk8_softmax_256_f32 = pipeline(&library, &device, "topk8_softmax_256_f32")?;
        let topk_softmax_rows_f32 = pipeline(&library, &device, "topk_softmax_rows_f32")?;
        let sample_gumbel_blocks_f32 = pipeline(&library, &device, "sample_gumbel_blocks_f32")?;
        let sample_topk_blocks_f32 = pipeline(&library, &device, "sample_topk_blocks_f32")?;
        let sample_topk_finalize_f32 = pipeline(&library, &device, "sample_topk_finalize_f32")?;
        let argmax_blocks_f32 = pipeline(&library, &device, "argmax_blocks_f32")?;
        let argmax_finalize_f32 = pipeline(&library, &device, "argmax_finalize_f32")?;
        let talker_greedy_argmax_f32 = pipeline(&library, &device, "talker_greedy_argmax_f32")?;
        let f32_to_bf16 = pipeline(&library, &device, "f32_to_bf16")?;
        let dequant_u8_to_bf16_t_gs64 = pipeline(&library, &device, "dequant_u8_to_bf16_t_gs64")?;
        let na_source = sources.na_gemm.as_ref();
        let compile_na = |name| compile_na_gemm_named(&device, na_source, name);
        let na_gemm_bf16 = compile_na_gemm(&device, na_source);
        let na_gemm_bf16_bn128 = compile_na("gemm_nax_bf16_bn128");
        let chunk_delta_seq_layout = compile_na("chunk_delta_seq_layout");
        let chunk_delta_seq_layout_tc = compile_na("chunk_delta_seq_layout_tc");
        let na_gemm_coop_qb = compile_na("gemm_nax_coop_qb");
        let na_gemm_coop_qb_gs128 = compile_na("gemm_nax_coop_qb_gs128");
        let na_gemm_coop_qb_tiled = compile_na("gemm_nax_coop_qb_tiled");
        let na_gemm_coop_qb_tiled_gs128 = compile_na("gemm_nax_coop_qb_tiled_gs128");
        let na_gemm_coop_qb_tiled_u4 = compile_na("gemm_nax_coop_qb_tiled_u4");
        let na_gemm_coop_qb_tiled_u4_align64 = compile_na("gemm_nax_coop_qb_tiled_u4_align64");
        let na_gemm_coop_qb_grouped = compile_na("gemm_nax_coop_qb_grouped");
        let na_gemm_coop_qb_grouped_gather = compile_na("gemm_nax_coop_qb_grouped_gather");
        let na_gemm_coop_qb_grouped_gate_up_swiglu =
            compile_na("gemm_nax_coop_qb_grouped_gate_up_swiglu");
        let na_gemm_coop_qb_grouped_gate_up_swiglu_u4 =
            compile_na("gemm_nax_coop_qb_grouped_gate_up_swiglu_u4");
        let na_gemm_coop_qb_grouped_scatter = compile_na("gemm_nax_coop_qb_grouped_scatter");
        let na_gemm_coop_qb_grouped_scatter_u4 = compile_na("gemm_nax_coop_qb_grouped_scatter_u4");
        let moe_coop_gather_padded = compile_na("moe_coop_gather_padded");
        let moe_coop_scatter_padded = compile_na("moe_coop_scatter_padded");
        let moe_g_fill_u32 = compile_na("moe_g_fill_u32");
        let moe_g_histogram = compile_na("moe_g_histogram");
        let moe_g_offsets = compile_na("moe_g_offsets");
        let moe_g_perm = compile_na("moe_g_perm");

        Ok(Self {
            device,
            queue,
            dense_matmul_rhs_t_f32,
            dense_qmv_rhs_bf16_f32,
            dense_qmv_fast_f32,
            dense_gemm_rhs_t_f32,
            affine_matmul_rhs_t_u32_f32,
            affine_qmv_fast_u4_gs64_f32,
            affine_qmv_fast_aligned_u4_gs64_f32,
            affine_qmv_fast_u4_gs64_align64_f32,
            affine_qmv_fast_u6_gs64_f32,
            affine_qmv_fast_aligned_u6_gs64_f32,
            affine_qmm2_fast_aligned_u4_gs64_f32,
            affine_qmm2_fast_aligned_u8_gs64_f32,
            affine_qmm2_fast_aligned_u8_gs128_f32,
            affine_qmv_fast_aligned_u8_gs64_f32,
            affine_qmv_fast_u8_gs64_align64_f32,
            affine_qmv_fast_aligned_u8_gs128_f32,
            affine_qmv_fast_aligned_u8_gs64_dot4_f32,
            affine_qmv_fast_aligned_u8_gs128_dot4_f32,
            affine_qmv_fast_aligned_u8_gs64_tg128_f32,
            affine_qmv_fast_aligned_u8_gs128_tg128_f32,
            affine_qmv_fast_aligned_u8_gs64_tg256_f32,
            affine_qmv_fast_aligned_u8_gs128_tg256_f32,
            affine_qmv_plus_one_fast_aligned_u8_gs64_f32,
            affine_qmv_one_fast_u8_gs64_f32,
            affine_qkv_split_qmv_fast_u4_gs64_f32,
            affine_qmv_rms_fast_u4_gs64_f32,
            affine_qmv_rms_fast_u8_gs64_f32,
            affine_qmv_rms_fast_u8_gs128_f32,
            affine_qkv_split_rms_qmv_fast_u4_gs64_f32,
            affine_qkv_split_rms_qmv_fast_u8_gs64_f32,
            affine_qmv_gated_input_fast_u4_gs64_f32,
            affine_qmv_gated_input_fast_u8_gs64_f32,
            embed_gather_dense_from_u32_f32,
            embed_gather_affine_from_u32_f32,
            swiglu_f32,
            geglu_tanh_f32,
            split_q_gate_rows_f32,
            attn_gate_rows_f32,
            accumulate_scaled_f32,
            add_scaled_f32,
            linear_attn_conv_silu_f32,
            linear_attn_conv_silu_k4_f32,
            linear_attn_norm_gates_f32,
            linear_attn_norm_gates_dk128_f32,
            linear_attn_norm_gates_inv_dk128_f32,
            linear_attn_conv_norm_gates_k4_dk128_f32,
            linear_attn_conv_norm_gates_k4_dk128_batch_f32,
            linear_attn_conv_state_finalize_f32,
            linear_attn_gated_delta_f32,
            linear_attn_gated_delta_dk128_tg4_f32,
            linear_attn_gated_delta_seq_dk128_tg4_f32,
            linear_attn_gated_delta_seq_dk128_bf16_tg4_f32,
            linear_attn_gated_delta_dk128_bf16_tg4_f32,
            linear_attn_gated_delta_inv_dk128_tg4_f32,
            linear_attn_rms_gate_f32,
            linear_attn_rms_gate_dv128_f32,
            linear_attn_rms_gate_batch_dv128_f32,
            affine_gather_matmul_rhs_t_u32_f32,
            affine_gather_qmv_fast_u4_gs64_f32,
            affine_gather_qmv_fast_u8_gs64_f32,
            affine_gather_qmv_fast_u8_gs128_f32,
            affine_gather_qmv_fast_u8_gs64_tg128_f32,
            affine_gather_qmv_fast_u8_gs128_tg128_f32,
            affine_gather_qmv_fast_u8_gs64_tg256_f32,
            affine_gather_qmv_fast_u8_gs128_tg256_f32,
            affine_gather_qmv_tail_u4_gs64_f32,
            affine_gather_gate_up_swiglu_fast_u4_gs64_f32,
            affine_gather_gate_up_swiglu_fast_u8_gs64_f32,
            affine_gather_gate_up_swiglu_fast_u8_gs128_f32,
            affine_gate_up_swiglu_fast_u4_gs64_f32,
            affine_gate_up_swiglu_fast_u8_gs64_f32,
            affine_gate_up_swiglu_gate_fast_u8_gs64_f32,
            affine_gate_up_swiglu_gate_fast_u8_gs128_f32,
            affine_gate_up_swiglu_fast_u8_gs128_f32,
            affine_argmax_qmv_fast_u4_gs64_f32,
            weighted_sum_topk_f32,
            scale_topk_scores_f32,
            weighted_sum_grouped_topk_f32,
            weighted_sum_add_grouped_topk_f32,
            weighted_sum_add_topk_f32,
            weighted_sum_add_shared_topk_f32,
            affine_gather_down_weighted_shared_fast_u8_gs64_f32,
            add_sigmoid_scaled_f32,
            add_sigmoid_scaled_rows_f32,
            copy_f32,
            copy_u16,
            rms_norm_rows_f32,
            rms_norm_simd_rows_f32,
            add_rms_norm_rows_f32,
            layer_norm_rows_f32,
            add_layer_norm_rows_f32,
            gelu_f32,
            layer_norm_rows_f32_bf16out,
            add_layer_norm_rows_f32_bf16out,
            gelu_f32_bf16out,
            add_row_bias_f32,
            whisper_attn_decode_f32,
            whisper_attn_decode_vec64_f32,
            im2col_f32,
            rms_norm_rope_heads_f32,
            rms_norm_heads_no_scale_f32,
            causal_attention_prefill_f32,
            causal_attention_prefill_mid_f32,
            causal_attention_prefill_long_f32,
            windowed_attention_prefill_f32,
            windowed_attention_prefill_mid_f32,
            windowed_attention_prefill_long_f32,
            causal_attention_prefill_batch_long_d128_f32,
            causal_attention_prefill_batch_long_d256_f32,
            causal_attention_prefill_batch_gqa8x4_d256_f32,
            causal_attention_prefill_steel_d256_f32,
            noncausal_attention_prefill_f32,
            steel_attention_f32_bq32_bk32_bd64,
            add_rms_norm_row_f32,
            topk_softmax_f32,
            topk_softmax_serial_f32,
            topk8_softmax_256_f32,
            topk_softmax_rows_f32,
            sample_gumbel_blocks_f32,
            sample_topk_blocks_f32,
            sample_topk_finalize_f32,
            argmax_blocks_f32,
            argmax_finalize_f32,
            talker_greedy_argmax_f32,
            f32_to_bf16,
            dequant_u8_to_bf16_t_gs64,
            na_gemm_bf16,
            na_gemm_bf16_bn128,
            chunk_delta_seq_layout,
            chunk_delta_seq_layout_tc,
            na_gemm_coop_qb,
            na_gemm_coop_qb_gs128,
            na_gemm_coop_qb_tiled,
            na_gemm_coop_qb_tiled_gs128,
            na_gemm_coop_qb_tiled_u4,
            na_gemm_coop_qb_tiled_u4_align64,
            na_gemm_coop_qb_grouped,
            na_gemm_coop_qb_grouped_gather,
            na_gemm_coop_qb_grouped_gate_up_swiglu,
            na_gemm_coop_qb_grouped_gate_up_swiglu_u4,
            na_gemm_coop_qb_grouped_scatter,
            na_gemm_coop_qb_grouped_scatter_u4,
            moe_coop_gather_padded,
            moe_coop_scatter_padded,
            moe_g_fill_u32,
            moe_g_histogram,
            moe_g_offsets,
            moe_g_perm,
            weight_buffers: Mutex::new(HashMap::new()),
            bf16_rhs_t_cache: Mutex::new(HashMap::new()),
            scratch_buffers: Mutex::new(HashMap::new()),
            moe_stacks: Mutex::new(HashMap::new()),
            concat_buffers: Mutex::new(HashMap::new()),
        })
    }
}
