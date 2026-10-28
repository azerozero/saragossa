//! Chemin full-attn « en lignes » fusionné (vérification MTP, position hôte).
//!
//! Pour `rows` positions contiguës, le RoPE (Q puis K), l'append KV (K puis V)
//! et l'attention tiennent chacun en UN dispatch au lieu d'un par ligne. Le
//! kernel 2-passes en lignes lit chaque bloc KV une fois pour `gqa × rows`
//! simdgroups (le L1 sert les relectures), là où la boucle par ligne relisait
//! tout le KV par ligne. Les kernels reprennent l'arithmétique des kernels
//! mono-ligne (même ordre d'accumulation) : résultats bit-identiques, prouvés
//! par `full_attn_rows_fused_matches_row_loop` (tests).

use super::attention::FullAttentionMetalState;
use super::gpu_position::to_u32;
use super::kernels::compile_kernel;
use super::utils::{
    byte_offset, flash_sdpa_enabled, sdpa_2pass_blocks, sdpa_2pass_enabled, sdpa_2pass_min_len,
};
use super::*;

/// Source compilée deux fois : `KV_T` → `float`/`bfloat`, `SUFFIX` → `f32`/`bf16`.
/// Le kernel RoPE en lignes (f32 seul) n'est pris que dans la variante f32.
const ROWS_KERNELS_TEMPLATE: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// Flash single-pass en lignes : threadgroup (q_head, ligne), len = dims.w + ligne.
kernel void attention_decode_flash_rows_SUFFIX(
    device const float* q [[buffer(0)]],
    device const KV_T* keys [[buffer(1)]],
    device const KV_T* values [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint4& dims [[buffer(4)]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint2 tg [[threadgroup_position_in_grid]]
) {
    constexpr uint BN = 32u;
    constexpr uint BD = 32u;
    constexpr uint MAX_PER_THREAD = 8u;

    const uint q_heads = dims.x;
    const uint kv_heads = dims.y;
    const uint head_dim = dims.z;
    const uint q_head = tg.x;
    const uint row = tg.y;
    const uint len = dims.w + row;
    if (q_head >= q_heads || len == 0u || head_dim == 0u) {
        return;
    }

    const uint elems_per_thread = (head_dim + BD - 1u) / BD;
    if (elems_per_thread > MAX_PER_THREAD) {
        return;
    }

    const uint group = q_heads / kv_heads;
    const uint kv_head = q_head / group;
    const uint kv_dim = kv_heads * head_dim;
    const uint q_start = row * q_heads * head_dim + q_head * head_dim;
    const uint kv_head_start = kv_head * head_dim;
    const float scale = rsqrt(float(head_dim));

    thread float q_lane[MAX_PER_THREAD];
    thread float o_lane[MAX_PER_THREAD];
    for (uint j = 0u; j < MAX_PER_THREAD; ++j) {
        const uint c = simd_lid * elems_per_thread + j;
        q_lane[j] = (j < elems_per_thread && c < head_dim)
            ? (q[q_start + c] * scale)
            : 0.0f;
        o_lane[j] = 0.0f;
    }

    float max_score = -INFINITY;
    float sum_exp_score = 0.0f;
    for (uint r = simd_gid; r < len; r += BN) {
        const uint k_start = r * kv_dim + kv_head_start;
        float score = 0.0f;
        for (uint j = 0u; j < elems_per_thread; ++j) {
            const uint c = simd_lid * elems_per_thread + j;
            if (c < head_dim) {
                score += q_lane[j] * float(keys[k_start + c]);
            }
        }
        score = simd_sum(score);

        const float new_max = max(max_score, score);
        const float factor = fast::exp(max_score - new_max);
        const float exp_score = fast::exp(score - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;

        const uint v_start = r * kv_dim + kv_head_start;
        for (uint j = 0u; j < elems_per_thread; ++j) {
            const uint c = simd_lid * elems_per_thread + j;
            if (c < head_dim) {
                o_lane[j] = o_lane[j] * factor + exp_score * float(values[v_start + c]);
            }
        }
    }

    threadgroup float max_scores[BN];
    threadgroup float sum_exp_scores[BN];
    threadgroup float outputs[BN * BD];
    if (simd_lid == 0u) {
        max_scores[simd_gid] = max_score;
        sum_exp_scores[simd_gid] = sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float partial_max = max_scores[simd_lid];
    const float global_max = simd_max(partial_max);
    const float partial_factor = fast::exp(partial_max - global_max);
    const float global_sum = simd_sum(sum_exp_scores[simd_lid] * partial_factor);

    for (uint j = 0u; j < elems_per_thread; ++j) {
        outputs[simd_lid * BD + simd_gid] = o_lane[j];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float reduced = simd_sum(outputs[simd_gid * BD + simd_lid] * partial_factor);
        reduced = (global_sum == 0.0f) ? reduced : (reduced / global_sum);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_lid == 0u) {
            const uint c = simd_gid * elems_per_thread + j;
            if (c < head_dim) {
                out[q_start + c] = reduced;
            }
        }
    }
}

// Append de `p.z` cellules : `out[i] = in[row * p.y + col]`, `cols = p.x`.
// `out` est lié à l'offset de la première ligne KV cible (lignes contiguës).
kernel void copy_rows_at_SUFFIX(
    device const float* input [[buffer(0)]],
    device KV_T* output [[buffer(1)]],
    constant uint3& p [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= p.z) { return; }
    const uint row = i / p.x;
    const uint col = i - row * p.x;
    output[i] = KV_T(input[row * p.y + col]);
}

// rms_norm par tête + RoPE de `rows` lignes : threadgroup (tête, ligne), position
// `dims.w + ligne`, strides d'entrée/sortie en éléments (`strides.x`, `strides.y`).
kernel void rms_norm_rope_heads_decode_rows_f32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint4& dims [[buffer(3)]],
    constant float2& params [[buffer(4)]],
    constant uint& frequency_dim [[buffer(5)]],
    constant uint2& strides [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 tg [[threadgroup_position_in_grid]]
) {
    const uint heads = dims.x;
    const uint head_dim = dims.y;
    const uint rope_dims = dims.z;
    const uint head = tg.x;
    const uint row = tg.y;
    const uint position = dims.w + row;
    const float eps = params.x;
    const float base_theta = params.y;
    if (head >= heads) { return; }
    input += row * strides.x;
    out += row * strides.y;

    threadgroup float partial[256];
    const uint start = head * head_dim;
    float sumsq = 0.0f;
    for (uint col = tid; col < head_dim; col += 256u) {
        const float value = input[start + col];
        sumsq += value * value;
    }
    partial[tid] = sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) { partial[tid] += partial[tid + stride]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv_rms = rsqrt((partial[0] / float(head_dim)) + eps);
    const uint pairs = rope_dims / 2u;
    for (uint col = tid; col < head_dim; col += 256u) {
        float value = input[start + col] * inv_rms * weight[col];
        if (col < rope_dims) {
            const uint pair = (col < pairs) ? col : (col - pairs);
            const float exponent = float(2u * pair) / float(frequency_dim);
            const float angle = float(position) / pow(base_theta, exponent);
            const float c = cos(angle);
            const float s = sin(angle);
            const float first = input[start + pair] * inv_rms * weight[pair];
            const float second = input[start + pair + pairs] * inv_rms * weight[pair + pairs];
            value = (col < pairs) ? (first * c - second * s) : (first * s + second * c);
        }
        out[start + col] = value;
    }
}
"#;

/// Source compilée par `head_dim` (`DHEAD` → `128`/`256`) et dtype KV : 2-passes en
/// lignes et flash en lignes spécialisé (boucles déroulées sur `HEAD_DIM / 32`
/// éléments par lane, comme `attention_decode_flash_d256_*`).
const ROWS_HEAD_KERNELS_TEMPLATE: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

kernel void attention_decode_flash_rows_dDHEAD_SUFFIX(
    device const float* q [[buffer(0)]],
    device const KV_T* keys [[buffer(1)]],
    device const KV_T* values [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint4& dims [[buffer(4)]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint2 tg [[threadgroup_position_in_grid]]
) {
    constexpr uint BN = 32u;
    constexpr uint BD = 32u;
    constexpr uint HEAD_DIM = DHEADu;
    constexpr uint ELEMS_PER_THREAD = HEAD_DIM / BD;

    const uint q_heads = dims.x;
    const uint kv_heads = dims.y;
    const uint q_head = tg.x;
    const uint row = tg.y;
    const uint len = dims.w + row;
    if (q_head >= q_heads || len == 0u) {
        return;
    }

    const uint group = q_heads / kv_heads;
    const uint kv_head = q_head / group;
    const uint kv_dim = kv_heads * HEAD_DIM;
    const uint q_start = row * q_heads * HEAD_DIM + q_head * HEAD_DIM;
    const uint kv_head_start = kv_head * HEAD_DIM;
    const float scale = rsqrt(float(HEAD_DIM));

    thread float q_lane[ELEMS_PER_THREAD];
    thread float o_lane[ELEMS_PER_THREAD];
    for (uint j = 0u; j < ELEMS_PER_THREAD; ++j) {
        const uint c = simd_lid * ELEMS_PER_THREAD + j;
        q_lane[j] = q[q_start + c] * scale;
        o_lane[j] = 0.0f;
    }

    float max_score = -INFINITY;
    float sum_exp_score = 0.0f;
    for (uint r = simd_gid; r < len; r += BN) {
        const uint k_start = r * kv_dim + kv_head_start;
        float score = 0.0f;
        for (uint j = 0u; j < ELEMS_PER_THREAD; ++j) {
            const uint c = simd_lid * ELEMS_PER_THREAD + j;
            score += q_lane[j] * float(keys[k_start + c]);
        }
        score = simd_sum(score);

        const float new_max = max(max_score, score);
        const float factor = fast::exp(max_score - new_max);
        const float exp_score = fast::exp(score - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;

        const uint v_start = r * kv_dim + kv_head_start;
        for (uint j = 0u; j < ELEMS_PER_THREAD; ++j) {
            const uint c = simd_lid * ELEMS_PER_THREAD + j;
            o_lane[j] = o_lane[j] * factor + exp_score * float(values[v_start + c]);
        }
    }

    threadgroup float max_scores[BN];
    threadgroup float sum_exp_scores[BN];
    threadgroup float outputs[BN * BD];
    if (simd_lid == 0u) {
        max_scores[simd_gid] = max_score;
        sum_exp_scores[simd_gid] = sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float partial_max = max_scores[simd_lid];
    const float global_max = simd_max(partial_max);
    const float partial_factor = fast::exp(partial_max - global_max);
    const float global_sum = simd_sum(sum_exp_scores[simd_lid] * partial_factor);

    for (uint j = 0u; j < ELEMS_PER_THREAD; ++j) {
        outputs[simd_lid * BD + simd_gid] = o_lane[j];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float reduced = simd_sum(outputs[simd_gid * BD + simd_lid] * partial_factor);
        reduced = (global_sum == 0.0f) ? reduced : (reduced / global_sum);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_lid == 0u) {
            out[q_start + simd_gid * ELEMS_PER_THREAD + j] = reduced;
        }
    }
}

// 2-passes split-K en lignes, passe 1 : threadgroup (32, gqa, lignes) par
// (kv_head, bloc) ; la ligne `row` attend sur `dims.z + row` positions. Les
// partiels sont indexés par `slot = row * q_heads + q_head`, lisibles tels
// quels par la passe 2 mono-ligne dispatchée sur `q_heads × lignes`.
kernel void attention_decode_2pass_1_rows_dDHEAD_SUFFIX(
    device const float* q [[buffer(0)]],
    device const KV_T* keys [[buffer(1)]],
    device const KV_T* values [[buffer(2)]],
    device float* partials [[buffer(3)]],
    device float* sums [[buffer(4)]],
    device float* maxs [[buffer(5)]],
    constant uint4& dims [[buffer(6)]],
    uint3 tidtg [[thread_position_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
    constexpr uint BD = 32u;
    constexpr uint D = DHEADu;
    constexpr uint QK = D / BD;
    const uint q_heads = dims.x;
    const uint kv_heads = dims.y;
    const uint blocks = dims.w;
    const uint gqa = q_heads / kv_heads;
    const uint kv_head = tid.x;
    const uint block_idx = tid.y;
    const uint q_in_group = tidtg.y;
    const uint row = tidtg.z;
    const uint len = dims.z + row;
    const uint q_head = kv_head * gqa + q_in_group;
    if (q_head >= q_heads || block_idx >= blocks) { return; }
    const uint slot = row * q_heads + q_head;
    const uint kv_dim = kv_heads * D;
    const float scale = rsqrt(float(D));

    float ql[QK];
    {
        const device float* q_ = q + slot * D + simd_lid * QK;
        for (uint j = 0u; j < QK; ++j) { ql[j] = scale * q_[j]; }
    }
    const device KV_T* k_ = keys + block_idx * kv_dim + kv_head * D + simd_lid * QK;
    const device KV_T* v_ = values + block_idx * kv_dim + kv_head * D + simd_lid * QK;

    float o[QK];
    for (uint j = 0u; j < QK; ++j) { o[j] = 0.0f; }
    float max_score = -INFINITY;
    float sum_exp = 0.0f;

    for (uint i = block_idx; i < len; i += blocks) {
        float score = 0.0f;
        for (uint j = 0u; j < QK; ++j) { score += ql[j] * float(k_[j]); }
        score = simd_sum(score);
        const float new_max = max(max_score, score);
        const float factor = fast::exp(max_score - new_max);
        const float exp_score = fast::exp(score - new_max);
        max_score = new_max;
        sum_exp = sum_exp * factor + exp_score;
        for (uint j = 0u; j < QK; ++j) { o[j] = o[j] * factor + exp_score * float(v_[j]); }
        k_ += blocks * kv_dim;
        v_ += blocks * kv_dim;
    }

    if (simd_lid == 0u) {
        sums[slot * blocks + block_idx] = sum_exp;
        maxs[slot * blocks + block_idx] = max_score;
    }
    device float* out_ = partials + (slot * blocks + block_idx) * D + simd_lid * QK;
    for (uint j = 0u; j < QK; ++j) { out_[j] = o[j]; }
}

"#;

/// Pipelines du chemin en lignes, compilés au premier usage (hors MTP : aucun coût).
#[derive(Debug)]
pub(super) struct RowsKernels {
    /// Flash en lignes générique (`head_dim ≤ 256`, multiple de 32), par dtype KV.
    flash_rows: ComputePipelineState,
    flash_rows_bf16: ComputePipelineState,
    /// Flash en lignes spécialisé d256, par dtype KV.
    flash_rows_d256: ComputePipelineState,
    flash_rows_d256_bf16: ComputePipelineState,
    /// 2-passes en lignes (passe 1), par `head_dim` puis dtype KV.
    two_pass_rows_d128: ComputePipelineState,
    two_pass_rows_d128_bf16: ComputePipelineState,
    two_pass_rows_d256: ComputePipelineState,
    two_pass_rows_d256_bf16: ComputePipelineState,
    rope_rows: ComputePipelineState,
    copy_rows: ComputePipelineState,
    copy_rows_bf16: ComputePipelineState,
}

fn rows_source(kv_type: &str, suffix: &str) -> String {
    ROWS_KERNELS_TEMPLATE
        .replace("KV_T", kv_type)
        .replace("SUFFIX", suffix)
}

fn rows_head_source(kv_type: &str, suffix: &str, head_dim: usize) -> String {
    ROWS_HEAD_KERNELS_TEMPLATE
        .replace("KV_T", kv_type)
        .replace("SUFFIX", suffix)
        .replace("DHEAD", &head_dim.to_string())
}

impl DecodeResidentState {
    pub(super) fn rows_kernels(&self) -> Result<&RowsKernels> {
        if let Some(kernels) = self.rows_kernels.get() {
            return Ok(kernels);
        }
        let f32_source = rows_source("float", "f32");
        let bf16_source = rows_source("bfloat", "bf16");
        let compile = |source: &str, name: &str| compile_kernel(&self.device, source, name);
        let kernels = RowsKernels {
            flash_rows: compile(&f32_source, "attention_decode_flash_rows_f32")?,
            flash_rows_bf16: compile(&bf16_source, "attention_decode_flash_rows_bf16")?,
            flash_rows_d256: compile(
                &rows_head_source("float", "f32", 256),
                "attention_decode_flash_rows_d256_f32",
            )?,
            flash_rows_d256_bf16: compile(
                &rows_head_source("bfloat", "bf16", 256),
                "attention_decode_flash_rows_d256_bf16",
            )?,
            two_pass_rows_d128: compile(
                &rows_head_source("float", "f32", 128),
                "attention_decode_2pass_1_rows_d128_f32",
            )?,
            two_pass_rows_d128_bf16: compile(
                &rows_head_source("bfloat", "bf16", 128),
                "attention_decode_2pass_1_rows_d128_bf16",
            )?,
            two_pass_rows_d256: compile(
                &rows_head_source("float", "f32", 256),
                "attention_decode_2pass_1_rows_d256_f32",
            )?,
            two_pass_rows_d256_bf16: compile(
                &rows_head_source("bfloat", "bf16", 256),
                "attention_decode_2pass_1_rows_d256_bf16",
            )?,
            rope_rows: compile(&f32_source, "rms_norm_rope_heads_decode_rows_f32")?,
            copy_rows: compile(&f32_source, "copy_rows_at_f32")?,
            copy_rows_bf16: compile(&bf16_source, "copy_rows_at_bf16")?,
        };
        Ok(self.rows_kernels.get_or_init(|| kernels))
    }

    /// rms_norm par tête + RoPE de `rows` lignes en un dispatch : la ligne `row`
    /// lit `input[input_offset + row·input_stride ..]`, tourne à `position + row`
    /// et écrit `output[row·output_stride ..]` (strides en éléments f32).
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si une dimension déborde `u32` ou si les kernels ne compilent pas.
    #[expect(
        clippy::too_many_arguments,
        reason = "mirror des paramètres rms_norm+RoPE (dims + position + theta) et strides"
    )]
    pub(super) fn encode_rms_norm_rope_decode_rows(
        &self,
        encoder: &ComputeCommandEncoderRef,
        kernels: &RowsKernels,
        input: &BufferRef,
        input_offset: u64,
        input_stride: usize,
        weight: &BufferRef,
        output: &BufferRef,
        output_stride: usize,
        num_heads: usize,
        head_dim: usize,
        rope_dims: usize,
        rope_frequency_dim: usize,
        position: usize,
        rows: usize,
        eps: f32,
        base_theta: f32,
    ) -> Result<()> {
        let dims = [
            to_u32(num_heads, "rope rows num_heads")?,
            to_u32(head_dim, "rope rows head_dim")?,
            to_u32(rope_dims, "rope rows rope_dims")?,
            to_u32(position, "rope rows position")?,
        ];
        let params: [f32; 2] = [eps, base_theta];
        let frequency_dim = to_u32(rope_frequency_dim, "rope rows frequency_dim")?;
        let strides = [
            to_u32(input_stride, "rope rows input_stride")?,
            to_u32(output_stride, "rope rows output_stride")?,
        ];
        encoder.set_compute_pipeline_state(&kernels.rope_rows);
        encoder.set_buffer(0, Some(input), input_offset);
        encoder.set_buffer(1, Some(weight), 0);
        encoder.set_buffer(2, Some(output), 0);
        set_bytes(encoder, 3, &dims);
        set_bytes(encoder, 4, &params);
        set_bytes(encoder, 5, &[frequency_dim]);
        set_bytes(encoder, 6, &strides);
        crate::metal_backend::profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(u64::from(dims[0]), to_u64(rows, "rope rows")?, 1),
            MTLSize::new(256, 1, 1),
        );
        crate::metal_backend::post_dispatch_barrier(encoder);
        Ok(())
    }
}

impl FullAttentionMetalState {
    /// Le chemin en lignes couvre les KV denses sous flash SDPA, `head_dim ≤ 256`
    /// multiple de 32 (le 2-passes exige en plus `head_dim` 128/256 et une GQA
    /// réelle, sinon le flash en lignes prend le relais quelle que soit la
    /// longueur). Un KV quantifié (d128/d256 par construction) passe toujours
    /// par le 2-passes en lignes, comme son chemin mono-ligne.
    pub(super) fn rows_fused_supported(&self) -> bool {
        flash_sdpa_enabled() && self.head_dim() <= 256 && self.head_dim() % 32 == 0
    }

    /// Append des `rows` K puis V (lignes contiguës à partir de `len`) en deux
    /// dispatches ; `k_buf`/`v_buf` sont lus à `offset + row·stride` (éléments).
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si la capacité est dépassée ou si une dimension déborde.
    #[expect(
        clippy::too_many_arguments,
        reason = "encodage Metal: buffers K/V, offsets et strides sources, lignes"
    )]
    pub(super) fn encode_append_kv_rows(
        &mut self,
        encoder: &ComputeCommandEncoderRef,
        kernels: &RowsKernels,
        k_buf: &BufferRef,
        k_offset: u64,
        k_stride: usize,
        v_buf: &BufferRef,
        v_offset: u64,
        v_stride: usize,
        rows: usize,
    ) -> Result<()> {
        let kv_dim = self.kv_dim();
        // Capacité vérifiée AVANT tout encodage : rien n'est lié ni dispatché hors
        // du KV (le `advance_len` final ne fait que consommer les lignes).
        let fits = self
            .len()
            .checked_add(rows)
            .is_some_and(|next| next <= self.capacity());
        if rows == 0 || !fits {
            return Err(InferError::Dimension(format!(
                "append KV rows: {} + {rows} lignes hors capacité {}",
                self.len(),
                self.capacity()
            )));
        }
        if let Some(quant) = self.quant() {
            for (tensor, payload, input, offset, stride) in [
                (&quant.keys, self.keys(), k_buf, k_offset, k_stride),
                (&quant.values, self.values(), v_buf, v_offset, v_stride),
            ] {
                tensor.encode_rows(
                    encoder,
                    payload,
                    input,
                    offset,
                    stride,
                    self.len(),
                    rows,
                    kv_dim,
                )?;
            }
            return self.advance_len(rows);
        }
        let cells = rows
            .checked_mul(kv_dim)
            .ok_or_else(|| InferError::Dimension("append KV rows déborde".to_string()))?;
        let element = self.keys().element();
        let pipeline = match element {
            GpuElement::F32 => &kernels.copy_rows,
            GpuElement::Bf16 => &kernels.copy_rows_bf16,
            GpuElement::U32 | GpuElement::U8 => {
                return Err(InferError::Metal(format!(
                    "append KV rows vers un buffer {element:?} invalide"
                )));
            }
        };
        let base = self
            .len()
            .checked_mul(kv_dim)
            .ok_or_else(|| InferError::Dimension("append KV rows offset déborde".to_string()))?;
        let output_offset = byte_offset(base, element, "append KV rows offset")?;
        let n = to_u32(cells, "append KV rows cells")?;
        let cols = to_u32(kv_dim, "append KV rows kv_dim")?;
        for (input, offset, stride, output) in [
            (k_buf, k_offset, k_stride, self.keys().buffer()),
            (v_buf, v_offset, v_stride, self.values().buffer()),
        ] {
            let p = [cols, to_u32(stride, "append KV rows stride")?, n];
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(input), offset);
            encoder.set_buffer(1, Some(output), output_offset);
            set_bytes(encoder, 2, &p);
            let width = pipeline.thread_execution_width().max(1);
            crate::metal_backend::profile_dispatch();
            encoder.dispatch_threads(MTLSize::new(u64::from(n), 1, 1), MTLSize::new(width, 1, 1));
            crate::metal_backend::post_dispatch_barrier(encoder);
        }
        self.advance_len(rows)
    }

    /// Attention des `rows` dernières lignes appendues : la ligne `row` (requête
    /// `q_rows[row·q_dim ..]`) attend sur `len - rows + row + 1` positions et écrit
    /// `out_rows[row·q_dim ..]`. 2-passes en lignes dès `len ≥ seuil` (chunks de
    /// lignes bornés par le threadgroup), flash en lignes sinon.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si `rows > len`, si le KV n'est pas f32/bf16 ou si une
    /// dimension déborde.
    pub(super) fn encode_attention_decode_rows(
        &self,
        encoder: &ComputeCommandEncoderRef,
        kernels: &RowsKernels,
        q_rows: &BufferRef,
        out_rows: &BufferRef,
        rows: usize,
    ) -> Result<()> {
        self.encode_attention_decode_rows_chunked(
            encoder,
            kernels,
            q_rows,
            out_rows,
            rows,
            rows_group_override(),
        )
    }

    /// Cœur de [`Self::encode_attention_decode_rows`] ; `max_rows_per_group`
    /// borne les lignes par threadgroup du 2-passes (tests : chunks forcés).
    pub(super) fn encode_attention_decode_rows_chunked(
        &self,
        encoder: &ComputeCommandEncoderRef,
        kernels: &RowsKernels,
        q_rows: &BufferRef,
        out_rows: &BufferRef,
        rows: usize,
        max_rows_per_group: Option<usize>,
    ) -> Result<()> {
        let len = self.len();
        if rows == 0 || rows > len {
            return Err(InferError::Dimension(format!(
                "attention rows: rows={rows} hors [1, len={len}]"
            )));
        }
        let (q_heads, kv_heads, head_dim) = (self.q_heads(), self.kv_heads(), self.head_dim());
        let gqa = q_heads / kv_heads.max(1);
        let len0 = len - rows + 1;
        let element = self.keys().element();
        let q_dim = q_heads * head_dim;
        // Même choix flash/2-passes que `encode_attention_decode_dims`, évalué sur la
        // longueur de la DERNIÈRE ligne du chunk (toujours 2-passes si quantifié).
        let quant = self.quant();
        let two_pass_for = |length: usize| {
            quant.is_some()
                || sdpa_2pass_enabled()
                    && flash_sdpa_enabled()
                    && matches!(head_dim, 128 | 256)
                    && gqa > 1
                    && q_heads % kv_heads == 0
                    && length >= sdpa_2pass_min_len()
        };
        // `None` pour un KV quantifié : `two_pass_for` exclut alors le flash.
        let flash = match (element, head_dim == 256) {
            (GpuElement::F32, true) => Some(&kernels.flash_rows_d256),
            (GpuElement::F32, false) => Some(&kernels.flash_rows),
            (GpuElement::Bf16, true) => Some(&kernels.flash_rows_d256_bf16),
            (GpuElement::Bf16, false) => Some(&kernels.flash_rows_bf16),
            (GpuElement::U8, _) if quant.is_some() => None,
            (GpuElement::U32 | GpuElement::U8, _) => {
                return Err(InferError::Metal(format!(
                    "attention rows sur KV {element:?} invalide"
                )));
            }
        };
        let pass1 = match (quant, element, head_dim) {
            (Some(quant), _, _) => &quant.attention,
            (None, GpuElement::F32, 128) => &kernels.two_pass_rows_d128,
            (None, GpuElement::F32, _) => &kernels.two_pass_rows_d256,
            (None, GpuElement::Bf16, 128) => &kernels.two_pass_rows_d128_bf16,
            (None, GpuElement::Bf16, _) => &kernels.two_pass_rows_d256_bf16,
            (None, GpuElement::U32 | GpuElement::U8, _) => {
                return Err(InferError::Metal(format!(
                    "attention rows 2pass sur KV {element:?} invalide"
                )));
            }
        };
        let pass2 = if head_dim == 128 {
            self.attention_2pass_2_d128()
        } else {
            self.attention_2pass_2()
        };
        // Frontière de partition entre la première et la dernière ligne (seuil
        // flash/2-passes ou nombre de blocs split-K) : chunks d'une ligne, chacun
        // avec le choix et les blocs de SA longueur, pour rester bit-identique à la
        // boucle par ligne.
        let boundary = two_pass_for(len0) != two_pass_for(len)
            || (two_pass_for(len) && sdpa_2pass_blocks(len0) != sdpa_2pass_blocks(len));
        let threads_per_row = 32 * gqa;
        let group_limit = usize::try_from(pass1.max_total_threads_per_threadgroup())
            .unwrap_or(threads_per_row)
            / threads_per_row;
        let rows_per_group = if boundary {
            1
        } else {
            max_rows_per_group
                .unwrap_or(usize::MAX)
                .min(group_limit)
                .clamp(1, rows)
        };
        let kv_heads_u32 = to_u32(kv_heads, "attention rows kv_heads")?;
        let mut start = 0;
        while start < rows {
            let chunk = rows_per_group.min(rows - start);
            let chunk_len0 = len0 + start;
            let row_offset = byte_offset(start * q_dim, GpuElement::F32, "attention rows offset")?;
            if !two_pass_for(chunk_len0 + chunk - 1) {
                let flash = flash.ok_or_else(|| {
                    InferError::Metal("attention rows: flash sur KV quantifié".to_string())
                })?;
                let dims = [
                    to_u32(q_heads, "attention rows q_heads")?,
                    kv_heads_u32,
                    to_u32(head_dim, "attention rows head_dim")?,
                    to_u32(chunk_len0, "attention rows len")?,
                ];
                encoder.set_compute_pipeline_state(flash);
                encoder.set_buffer(0, Some(q_rows), row_offset);
                encoder.set_buffer(1, Some(self.keys().buffer()), 0);
                encoder.set_buffer(2, Some(self.values().buffer()), 0);
                encoder.set_buffer(3, Some(out_rows), row_offset);
                set_bytes(encoder, 4, &dims);
                crate::metal_backend::profile_dispatch();
                encoder.dispatch_thread_groups(
                    MTLSize::new(u64::from(dims[0]), to_u64(chunk, "attention rows")?, 1),
                    MTLSize::new(1024, 1, 1),
                );
                crate::metal_backend::post_dispatch_barrier(encoder);
                start += chunk;
                continue;
            }
            let blocks = sdpa_2pass_blocks(chunk_len0 + chunk - 1);
            let blocks_u32 = to_u32(blocks, "attention rows blocks")?;
            let reduce_cells = q_heads
                .checked_mul(chunk)
                .and_then(|slots| slots.checked_mul(blocks))
                .ok_or_else(|| {
                    InferError::Dimension("attention rows reduce déborde".to_string())
                })?;
            let partial_cells = reduce_cells.checked_mul(head_dim).ok_or_else(|| {
                InferError::Dimension("attention rows partials débordent".to_string())
            })?;
            let partials = self.scratch().lease(partial_cells, GpuElement::F32)?;
            let sums = self.scratch().lease(reduce_cells, GpuElement::F32)?;
            let maxs = self.scratch().lease(reduce_cells, GpuElement::F32)?;
            let dims = [
                to_u32(q_heads, "attention rows q_heads")?,
                kv_heads_u32,
                to_u32(chunk_len0, "attention rows len")?,
                blocks_u32,
            ];
            encoder.set_compute_pipeline_state(pass1);
            encoder.set_buffer(0, Some(q_rows), row_offset);
            encoder.set_buffer(1, Some(self.keys().buffer()), 0);
            encoder.set_buffer(2, Some(self.values().buffer()), 0);
            encoder.set_buffer(3, Some(partials.tensor().buffer()), 0);
            encoder.set_buffer(4, Some(sums.tensor().buffer()), 0);
            encoder.set_buffer(5, Some(maxs.tensor().buffer()), 0);
            set_bytes(encoder, 6, &dims);
            if let Some(quant) = quant {
                encoder.set_buffer(7, Some(quant.keys.scales().buffer()), 0);
                encoder.set_buffer(8, Some(quant.values.scales().buffer()), 0);
            }
            crate::metal_backend::profile_dispatch();
            encoder.dispatch_thread_groups(
                MTLSize::new(u64::from(kv_heads_u32), u64::from(blocks_u32), 1),
                MTLSize::new(
                    32,
                    to_u64(gqa, "attention rows gqa")?,
                    to_u64(chunk, "attention rows chunk")?,
                ),
            );
            crate::metal_backend::post_dispatch_barrier(encoder);

            encoder.set_compute_pipeline_state(pass2);
            encoder.set_buffer(0, Some(partials.tensor().buffer()), 0);
            encoder.set_buffer(1, Some(sums.tensor().buffer()), 0);
            encoder.set_buffer(2, Some(maxs.tensor().buffer()), 0);
            encoder.set_buffer(3, Some(out_rows), row_offset);
            set_bytes(encoder, 4, &[blocks_u32]);
            crate::metal_backend::profile_dispatch();
            encoder.dispatch_thread_groups(
                MTLSize::new(to_u64(q_heads * chunk, "attention rows slots")?, 1, 1),
                MTLSize::new(32, 32, 1),
            );
            crate::metal_backend::post_dispatch_barrier(encoder);
            start += chunk;
        }
        Ok(())
    }
}

/// Diagnostic : `SARAGOSSA_RUST_FULL_ROWS_GROUP=n` borne les lignes par threadgroup
/// du 2-passes en lignes (1 = un dispatch par ligne, comme la boucle historique).
fn rows_group_override() -> Option<usize> {
    static VALUE: OnceLock<Option<usize>> = OnceLock::new();
    *VALUE.get_or_init(|| {
        crate::runtime_flags::env_var("SARAGOSSA_RUST_FULL_ROWS_GROUP")
            .and_then(|value| value.parse().ok())
            .filter(|rows| *rows > 0)
    })
}

fn to_u64(value: usize, label: &str) -> Result<u64> {
    u64::try_from(value).map_err(|_| InferError::Dimension(format!("{label} hors u64")))
}

fn set_bytes<T: Copy>(encoder: &ComputeCommandEncoderRef, index: u64, values: &[T]) {
    encoder.set_bytes(
        index,
        std::mem::size_of_val(values) as u64,
        values.as_ptr().cast::<c_void>(),
    );
}
