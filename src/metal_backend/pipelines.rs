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
    pub(super) steel_attention_nax: Cow<'static, str>,
}

impl KernelSources {
    fn load() -> Result<Self> {
        Self::from_runtime_path(
            crate::runtime_flags::env_var_os("SARAGOSSA_RUST_KERNELS_PATH").map(PathBuf::from),
        )
    }

    pub(super) fn from_runtime_path(path: Option<PathBuf>) -> Result<Self> {
        let Some(root) = path else {
            return Ok(Self {
                matmul: Cow::Borrowed(MATMUL_KERNELS),
                na_gemm: Cow::Borrowed(NA_GEMM_SRC),
                steel_attention: Cow::Borrowed(STEEL_ATTENTION_KERNELS),
                steel_attention_nax: Cow::Borrowed(STEEL_ATTENTION_NAX_KERNELS),
            });
        };
        Ok(Self {
            matmul: Cow::Owned(read_runtime_kernel(&root, &["kernels.metal"])?),
            na_gemm: Cow::Owned(read_runtime_kernel(&root, &["na_gemm", "na_gemm.metal"])?),
            steel_attention: Cow::Owned(read_runtime_kernel(
                &root,
                &["steel_attention", "steel_attention.metal"],
            )?),
            // Optionnelle : un répertoire préparé avant le NAX reste valide.
            steel_attention_nax: match read_runtime_kernel(
                &root,
                &["steel_attention_nax", "steel_attention_nax.metal"],
            ) {
                Ok(source) => Cow::Owned(source),
                Err(_) => Cow::Borrowed(STEEL_ATTENTION_NAX_KERNELS),
            },
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
        "SARAGOSSA_RUST_KERNELS_PATH={} ne contient aucun de: {}",
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
        crate::runtime_flags::env_var("SARAGOSSA_STT_STEEL_ATTN").as_deref(),
        Some("0" | "false" | "off" | "no")
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

/// Pipelines Steel causaux tirés d'une seule compilation de la bibliothèque :
/// le d256 du prefill Qwen et le d128 avec sinks du prefill flint (K/V f32,
/// puis K/V bf16). `None` pour un pipeline qui ne compile pas (trace sous
/// `SARAGOSSA_RUST_TRACE_PREFILL`).
#[cfg(all(target_os = "macos", feature = "metal"))]
fn compile_steel_causal_attentions(device: &Device, source: &str) -> SteelCausalAttentions {
    let options = CompileOptions::new();
    options.set_fast_math_enabled(true);
    let library = match device.new_library_with_source(source, &options) {
        Ok(library) => library,
        Err(message) => {
            trace_steel_compile_error("steel causal", "compile", &message);
            return (None, None, None);
        }
    };
    let d256 = steel_causal_function(
        device,
        &library,
        STEEL_CAUSAL_ATTN_D256_F32_BQ32_BK64_BD64X4,
        false,
    );
    let sink_d128 = steel_causal_function(device, &library, STEEL_SINK_ATTN_D128_F32, true);
    let sink_d128_kv_bf16 =
        steel_causal_function(device, &library, STEEL_SINK_ATTN_D128_KV_BF16, true);
    (d256, sink_d128, sink_d128_kv_bf16)
}

/// d256 Qwen, d128 avec sinks sur K/V f32, puis sur K/V bf16.
type SteelCausalAttentions = (
    Option<ComputePipelineState>,
    Option<ComputePipelineState>,
    Option<ComputePipelineState>,
);

#[cfg(all(target_os = "macos", feature = "metal"))]
fn steel_causal_function(
    device: &Device,
    library: &metal::Library,
    name: &str,
    sinks: bool,
) -> Option<ComputePipelineState> {
    let constants = FunctionConstantValues::new();
    set_metal_bool_constant(&constants, 200, false);
    set_metal_bool_constant(&constants, 201, false);
    set_metal_bool_constant(&constants, 300, false);
    set_metal_bool_constant(&constants, 301, true);
    set_metal_bool_constant(&constants, 302, sinks);
    let function = match library.get_function(name, Some(constants)) {
        Ok(function) => function,
        Err(message) => {
            trace_steel_compile_error(name, "function", &message);
            return None;
        }
    };
    match device.new_compute_pipeline_state_with_function(&function) {
        Ok(pipeline) => Some(pipeline),
        Err(message) => {
            trace_steel_compile_error(name, "pipeline", &message);
            None
        }
    }
}

/// Compile l'attention causale NAX d256 (bibliothèque séparée, MPP). `None`
/// si MetalPerformancePrimitives manque (macOS < 26) ou sans accélérateur.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub(super) fn compile_steel_causal_nax_d256_attention(
    device: &Device,
    source: &str,
    shape: attention::NaxCausalAttnShape,
) -> Option<attention::NaxCausalAttention> {
    compile_nax_causal_attention(device, source, shape, false, false)
}

/// Compile l'attention causale NAX d128 avec sinks f32 : couches pleines
/// flint, ou glissantes si `window` (constante 305, longueur en `buffer(9)`).
/// Le pipeline sans fenêtre élimine ce code à la compilation : il reste celui
/// d'avant la fenêtre. `None` dans les mêmes cas que le d256.
///
/// NOTE: `compile_nax_function` recompile toute la source NAX : avec le d256
/// actif, la source est compilée deux fois au démarrage. Coût accepté plutôt
/// qu'une `Library` partagée : `load_ms` de flint-int4 inchangé avec et sans
/// ce pipeline (4,5 s, le cache du compilateur Metal absorbe la seconde).
#[cfg(all(target_os = "macos", feature = "metal"))]
pub(super) fn compile_nax_sink_d128_attention(
    device: &Device,
    source: &str,
    shape: attention::NaxCausalAttnShape,
    window: bool,
) -> Option<attention::NaxCausalAttention> {
    compile_nax_causal_attention(device, source, shape, true, window)
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn compile_nax_causal_attention(
    device: &Device,
    source: &str,
    shape: attention::NaxCausalAttnShape,
    sinks: bool,
    window: bool,
) -> Option<attention::NaxCausalAttention> {
    let pipeline = compile_nax_function(
        device,
        source,
        shape.host_name,
        &[
            (200, false),
            (201, false),
            (300, false),
            (301, true),
            (302, sinks),
            (303, false),
            (304, shape.head_major),
            (305, window),
        ],
    )?;
    Some(attention::NaxCausalAttention { pipeline, shape })
}

/// Compile le gated delta NAX figé, sans cache d'état (constante 200).
pub(super) fn compile_gated_delta_nax(
    device: &Device,
    source: &str,
) -> Option<ComputePipelineState> {
    compile_nax_function(device, source, GATED_DELTA_NAX_HK16_HV48, &[(200, false)])
}

fn compile_nax_function(
    device: &Device,
    source: &str,
    name: &str,
    bool_constants: &[(u64, bool)],
) -> Option<ComputePipelineState> {
    let options = CompileOptions::new();
    options.set_fast_math_enabled(true);
    let library = match device.new_library_with_source(source, &options) {
        Ok(library) => library,
        Err(message) => {
            trace_steel_compile_error(name, "nax compile", &message);
            return None;
        }
    };
    let constants = FunctionConstantValues::new();
    for &(index, value) in bool_constants {
        set_metal_bool_constant(&constants, index, value);
    }
    let function = match library.get_function(name, Some(constants)) {
        Ok(function) => function,
        Err(message) => {
            trace_steel_compile_error(name, "nax function", &message);
            return None;
        }
    };
    match device.new_compute_pipeline_state_with_function(&function) {
        Ok(pipeline) => Some(pipeline),
        Err(message) => {
            trace_steel_compile_error(name, "nax pipeline", &message);
            None
        }
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn trace_steel_compile_error(kernel: &str, stage: &str, message: &str) {
    if crate::runtime_flags::trace_prefill_enabled() {
        eprintln!("{kernel} {stage}: {message}");
    }
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
fn compile_steel_attention(_device: &Device, _source: &str) -> Option<ComputePipelineState> {
    None
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
fn compile_steel_causal_attentions(_device: &Device, _source: &str) -> SteelCausalAttentions {
    (None, None, None)
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
        let affine_qmv_fast_aligned_u2_gs64_f32 =
            pipeline(&library, &device, "affine_qmv_fast_aligned_u2_gs64_f32")?;
        let affine_qmv_fast_aligned_u4_gs64_f32 =
            pipeline(&library, &device, "affine_qmv_fast_aligned_u4_gs64_f32")?;
        let affine_qmv_fast_aligned_u4_gs64_residual_f32 = pipeline(
            &library,
            &device,
            "affine_qmv_fast_aligned_u4_gs64_residual_f32",
        )?;
        let affine_qmv_fast_aligned_u3_gs64_f32 =
            pipeline(&library, &device, "affine_qmv_fast_aligned_u3_gs64_f32")?;
        let affine_qmv_fast_u4_gs64_align64_f32 =
            pipeline(&library, &device, "affine_qmv_fast_u4_gs64_align64_f32")?;
        let affine_qmv_fast_u6_gs64_f32 =
            pipeline(&library, &device, "affine_qmv_fast_u6_gs64_f32")?;
        let affine_qmv_fast_aligned_u6_gs64_f32 =
            pipeline(&library, &device, "affine_qmv_fast_aligned_u6_gs64_f32")?;
        let affine_qmm2_fast_aligned_u4_gs64_f32 =
            pipeline(&library, &device, "affine_qmm2_fast_aligned_u4_gs64_f32")?;
        let affine_qmm8_fast_aligned_u4_gs64_f32 =
            pipeline(&library, &device, "affine_qmm8_fast_aligned_u4_gs64_f32")?;
        let affine_qmm2_fast_aligned_u3_gs64_f32 =
            pipeline(&library, &device, "affine_qmm2_fast_aligned_u3_gs64_f32")?;
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
        let affine_qkv_split_qmv_fast_u3_gs64_f32 =
            pipeline(&library, &device, "affine_qkv_split_qmv_fast_u3_gs64_f32")?;
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
        let affine_qkv_split_rms_qmv_fast_u3_gs64_f32 = pipeline(
            &library,
            &device,
            "affine_qkv_split_rms_qmv_fast_u3_gs64_f32",
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
        let linear_attn_conv_state_shift_f32 =
            pipeline(&library, &device, "linear_attn_conv_state_shift_f32")?;
        let linear_attn_gated_delta_f32 =
            pipeline(&library, &device, "linear_attn_gated_delta_f32")?;
        let linear_attn_gated_delta_dk128_tg4_f32 =
            pipeline(&library, &device, "linear_attn_gated_delta_dk128_tg4_f32")?;
        let linear_attn_gated_delta_seq_dk128_tg4_f32 = pipeline(
            &library,
            &device,
            "linear_attn_gated_delta_seq_dk128_tg4_f32",
        )?;
        let linear_attn_gated_delta_seq_capture_dk128_tg4_f32 = pipeline(
            &library,
            &device,
            "linear_attn_gated_delta_seq_capture_dk128_tg4_f32",
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
        let affine_gather_qmv_fast_u3_gs64_f32 =
            pipeline(&library, &device, "affine_gather_qmv_fast_u3_gs64_f32")?;
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
        let affine_gather_qmv_tail_u3_gs64_f32 =
            pipeline(&library, &device, "affine_gather_qmv_tail_u3_gs64_f32")?;
        let affine_gather_qmv_tail_u4_gs64_f32 =
            pipeline(&library, &device, "affine_gather_qmv_tail_u4_gs64_f32")?;
        let affine_gather_gate_up_swiglu_fast_u3_gs64_f32 = pipeline(
            &library,
            &device,
            "affine_gather_gate_up_swiglu_fast_u3_gs64_f32",
        )?;
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
        let copy_cols_f32 = pipeline(&library, &device, "copy_cols_f32")?;
        let copy_u16 = pipeline(&library, &device, "copy_u16")?;
        let rms_norm_rows_f32 = pipeline(&library, &device, "rms_norm_rows_f32")?;
        let rms_norm_simd_rows_f32 = pipeline(&library, &device, "rms_norm_simd_rows_f32")?;
        let rms_norm_simd_u3_rows_f32 = pipeline(&library, &device, "rms_norm_simd_u3_rows_f32")?;
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
        let attention_prefill_sink_f32 = pipeline(&library, &device, "attention_prefill_sink_f32")?;
        let attention_prefill_sink_bf16 =
            pipeline(&library, &device, "attention_prefill_sink_bf16")?;
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
        let (
            causal_attention_prefill_steel_d256_f32,
            attention_prefill_steel_sink_d128_f32,
            attention_prefill_steel_sink_d128_kv_bf16,
        ) = compile_steel_causal_attentions(&device, sources.steel_attention.as_ref());
        let causal_attention_prefill_steel_nax_d256 =
            if crate::runtime_flags::prefill_attn_nax_enabled() {
                let pipeline = compile_steel_causal_nax_d256_attention(
                    &device,
                    sources.steel_attention_nax.as_ref(),
                    attention::NaxCausalAttnShape::from_flags(),
                );
                if pipeline.is_none() {
                    eprintln!("attention prefill NAX indisponible : repli Steel f32");
                }
                pipeline
            } else {
                None
            };
        let attention_prefill_nax_sink_d128 =
            if crate::runtime_flags::prefill_attn_nax_sinks_enabled() {
                let pipeline = compile_nax_sink_d128_attention(
                    &device,
                    sources.steel_attention_nax.as_ref(),
                    attention::NAX_SINK_ATTN_D128,
                    false,
                );
                if pipeline.is_none() {
                    eprintln!("attention prefill NAX avec sinks indisponible : repli Steel d128");
                }
                pipeline
            } else {
                None
            };
        let attention_prefill_nax_sink_d128_window =
            if crate::runtime_flags::prefill_attn_nax_window_enabled() {
                let pipeline = compile_nax_sink_d128_attention(
                    &device,
                    sources.steel_attention_nax.as_ref(),
                    attention::NAX_SINK_ATTN_D128,
                    true,
                );
                if pipeline.is_none() {
                    eprintln!("attention prefill NAX fenêtrée indisponible : repli naïf bf16");
                }
                pipeline
            } else {
                None
            };
        let noncausal_attention_prefill_f32 =
            pipeline(&library, &device, "noncausal_attention_prefill_f32")?;
        let steel_attention_f32_bq32_bk32_bd64 =
            compile_steel_attention(&device, sources.steel_attention.as_ref());
        let add_rms_norm_row_f32 = pipeline(&library, &device, "add_rms_norm_row_f32")?;
        let topk_softmax_f32 = pipeline(&library, &device, "topk_softmax_f32")?;
        let topk_sigmoid_f32 = pipeline(&library, &device, "topk_sigmoid_f32")?;
        let topk_sigmoid_rows_f32 = pipeline(&library, &device, "topk_sigmoid_rows_f32")?;
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
        let gated_delta_nax_hk16_hv48 = if crate::runtime_flags::prefill_linear_nax_enabled() {
            let pipeline = compile_gated_delta_nax(&device, GATED_DELTA_NAX_KERNELS);
            if pipeline.is_none() {
                eprintln!("gated delta NAX indisponible : repli chunk-delta");
            }
            pipeline
        } else {
            None
        };
        let na_gemm_coop_qb = compile_na("gemm_nax_coop_qb");
        let na_gemm_coop_qb_gs128 = compile_na("gemm_nax_coop_qb_gs128");
        let na_gemm_coop_qb_tiled = compile_na("gemm_nax_coop_qb_tiled");
        let na_gemm_coop_qb_tiled_gs128 = compile_na("gemm_nax_coop_qb_tiled_gs128");
        let na_gemm_coop_qb_tiled_u4 = compile_na("gemm_nax_coop_qb_tiled_u4");
        let na_gemm_coop_qb_tiled_u4_x3 = compile_na("gemm_nax_coop_qb_tiled_u4_x3");
        let na_gemm_coop_qb_tiled_u4_align64 = compile_na("gemm_nax_coop_qb_tiled_u4_align64");
        let na_gemm_coop_qb_tiled_u4_gate_up_swiglu =
            compile_na("gemm_nax_coop_qb_tiled_u4_gate_up_swiglu");
        let na_gemm_skinny_u4 = compile_na("gemm_nax_skinny_u4");
        let na_gemm_skinny_u4_x3 = compile_na("gemm_nax_skinny_u4_x3");
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
            affine_qmv_fast_aligned_u2_gs64_f32,
            affine_qmv_fast_aligned_u4_gs64_f32,
            affine_qmv_fast_aligned_u4_gs64_residual_f32,
            affine_qmv_fast_aligned_u3_gs64_f32,
            affine_qmv_fast_u4_gs64_align64_f32,
            affine_qmv_fast_u6_gs64_f32,
            affine_qmv_fast_aligned_u6_gs64_f32,
            affine_qmm2_fast_aligned_u4_gs64_f32,
            affine_qmm8_fast_aligned_u4_gs64_f32,
            affine_qmm2_fast_aligned_u3_gs64_f32,
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
            affine_qkv_split_qmv_fast_u3_gs64_f32,
            affine_qmv_rms_fast_u4_gs64_f32,
            affine_qmv_rms_fast_u8_gs64_f32,
            affine_qmv_rms_fast_u8_gs128_f32,
            affine_qkv_split_rms_qmv_fast_u4_gs64_f32,
            affine_qkv_split_rms_qmv_fast_u3_gs64_f32,
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
            linear_attn_conv_state_shift_f32,
            linear_attn_gated_delta_f32,
            linear_attn_gated_delta_dk128_tg4_f32,
            linear_attn_gated_delta_seq_dk128_tg4_f32,
            linear_attn_gated_delta_seq_capture_dk128_tg4_f32,
            linear_attn_gated_delta_seq_dk128_bf16_tg4_f32,
            linear_attn_gated_delta_dk128_bf16_tg4_f32,
            linear_attn_gated_delta_inv_dk128_tg4_f32,
            linear_attn_rms_gate_f32,
            linear_attn_rms_gate_dv128_f32,
            linear_attn_rms_gate_batch_dv128_f32,
            affine_gather_matmul_rhs_t_u32_f32,
            affine_gather_qmv_fast_u3_gs64_f32,
            affine_gather_qmv_fast_u4_gs64_f32,
            affine_gather_qmv_fast_u8_gs64_f32,
            affine_gather_qmv_fast_u8_gs128_f32,
            affine_gather_qmv_fast_u8_gs64_tg128_f32,
            affine_gather_qmv_fast_u8_gs128_tg128_f32,
            affine_gather_qmv_fast_u8_gs64_tg256_f32,
            affine_gather_qmv_fast_u8_gs128_tg256_f32,
            affine_gather_qmv_tail_u3_gs64_f32,
            affine_gather_qmv_tail_u4_gs64_f32,
            affine_gather_gate_up_swiglu_fast_u3_gs64_f32,
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
            copy_cols_f32,
            copy_u16,
            rms_norm_rows_f32,
            rms_norm_simd_rows_f32,
            rms_norm_simd_u3_rows_f32,
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
            attention_prefill_sink_f32,
            attention_prefill_sink_bf16,
            causal_attention_prefill_batch_long_d128_f32,
            causal_attention_prefill_batch_long_d256_f32,
            causal_attention_prefill_batch_gqa8x4_d256_f32,
            causal_attention_prefill_steel_d256_f32,
            attention_prefill_steel_sink_d128_f32,
            attention_prefill_steel_sink_d128_kv_bf16,
            prefill_kv_bf16_store_sinks: crate::runtime_flags::flint_prefill_kv_bf16_store_enabled(
            ),
            prefill_sink_steel: crate::runtime_flags::flint_prefill_steel_enabled(),
            qmm_na_u4_min_dim: QMM_NA_U4_DEFAULT_MIN_DIM,
            prefill_window_kv: crate::runtime_flags::flint_window_kv_enabled(),
            causal_attention_prefill_steel_nax_d256,
            attention_prefill_nax_sink_d128,
            attention_prefill_nax_sink_d128_window,
            prefill_kv_bf16_mirror: crate::runtime_flags::prefill_kv_bf16_mirror_enabled(),
            prefill_kv_bf16_store_min_rows: (crate::runtime_flags::prefill_kv_bf16_store_enabled()
                && crate::decode_resident::kv_bf16_dense_by_default())
            .then(crate::runtime_flags::kv_bf16_greedy_min_tokens),
            noncausal_attention_prefill_f32,
            steel_attention_f32_bq32_bk32_bd64,
            add_rms_norm_row_f32,
            topk_softmax_f32,
            topk_sigmoid_f32,
            topk_sigmoid_rows_f32,
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
            gated_delta_nax_hk16_hv48,
            na_gemm_coop_qb,
            na_gemm_coop_qb_gs128,
            na_gemm_coop_qb_tiled,
            na_gemm_coop_qb_tiled_gs128,
            na_gemm_coop_qb_tiled_u4,
            na_gemm_coop_qb_tiled_u4_x3,
            na_gemm_skinny_u4,
            na_gemm_skinny_u4_x3,
            na_gemm_coop_qb_tiled_u4_align64,
            na_gemm_coop_qb_tiled_u4_gate_up_swiglu,
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
            scratch_buffers: Mutex::new(ScratchBufferCache::default()),
            moe_stacks: Mutex::new(HashMap::new()),
            concat_buffers: Mutex::new(HashMap::new()),
            preserve_weight_caches: std::sync::atomic::AtomicBool::new(false),
            #[cfg(test)]
            scratch_releases: std::sync::atomic::AtomicUsize::new(0),
        })
    }
}
