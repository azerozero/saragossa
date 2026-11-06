//! Kernels Metal propres à Flash-Next (`qwen4_exp`) : hyper-connections et PLE.
//!
//! Chaque kernel transcrit l'ordre des opérations f32 de la référence CPU
//! (`src/qwen4_exp/residual.rs`, `src/qwen4_exp/ple.rs`) pour que l'écart reste au niveau de
//! l'accumulation.

use metal::{ComputePipelineState, Device};

use super::super::kernels::compile_kernels;
use crate::Result;

const SOURCE: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// Norme RMS par groupe de `group` valeurs, gain `1 + w` (w de la largeur
// totale) : un threadgroup de 1024 par groupe, réduction simd.
kernel void flash_next_rms_norm_1p_groups_f32(
    device const float* x [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& group [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint g [[threadgroup_position_in_grid]]
) {
    threadgroup float partial[32];
    const uint base = g * group;
    float ss = 0.0f;
    for (uint col = tid; col < group; col += 1024u) {
        const float v = x[base + col];
        ss += v * v;
    }
    ss = simd_sum(ss);
    if (lane == 0u) {
        partial[simd] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total = simd_sum(lane < 32u ? partial[lane] : 0.0f);
    const float inv = 1.0f / sqrt(total / float(group) + eps);
    for (uint col = tid; col < group; col += 1024u) {
        const uint i = base + col;
        out[i] = (x[i] * inv) * (1.0f + w[i]);
    }
}

// Injection, sommes partielles : `partial[s, c] = Σ_{d ∈ tranche c} W[s, d]·n[d]`,
// une tranche contiguë d'environ `width / chunks` valeurs par threadgroup (256).
kernel void flash_next_hc_inject_partial_f32(
    device const float* normed [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device float* partial [[buffer(2)]],
    constant uint2& dims [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint2 tg [[threadgroup_position_in_grid]]
) {
    threadgroup float shared[8];
    const uint width = dims.x;
    const uint chunks = dims.y;
    // Tranches [c·width/chunks, (c+1)·width/chunks) : couvrent toute la ligne.
    const uint start = tg.x * width / chunks;
    const uint end = (tg.x + 1u) * width / chunks;
    const uint row = tg.y * width;
    float acc = 0.0f;
    for (uint d = start + tid; d < end; d += 256u) {
        acc += w[row + d] * normed[d];
    }
    acc = simd_sum(acc);
    if (lane == 0u) {
        shared[simd] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float total = 0.0f;
        for (uint k = 0u; k < 8u; ++k) {
            total += shared[k];
        }
        partial[tg.y * chunks + tg.x] = total;
    }
}

// `low[i] = SiLU((Σ_c partial[i, c]) / hc)` : réduit les `chunks` sommes
// partielles du Down (1 tranche : `partial` peut être `low`, en place).
kernel void flash_next_hc_silu_scale_f32(
    device const float* partial [[buffer(0)]],
    device float* low [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant float& hc [[buffer(3)]],
    constant uint& chunks [[buffer(4)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= n) {
        return;
    }
    float sum = 0.0f;
    for (uint c = 0u; c < chunks; ++c) {
        sum += partial[i * chunks + c];
    }
    const float v = sum / hc;
    low[i] = v * (1.0f / (1.0f + exp(-v)));
}

// qmv u8 gs64 en split-K (Down des hyper-connections, peu de lignes, entrée
// longue) : un simdgroup par (ligne, tranche de mots), 8 lignes par
// threadgroup ; `partial[row, c]`. Déquantification de
// `affine_qmv_fast_aligned_u8_gs64_f32` : `float(q)·scale + bias`.
// dims = (out_dim, in_dim, packed_cols, groups), `chunks` divise `packed_cols`.
kernel void flash_next_qmv_u8_gs64_splitk_f32(
    device const float* lhs [[buffer(0)]],
    device const uint* packed [[buffer(1)]],
    device const bfloat* scales [[buffer(2)]],
    device const bfloat* biases [[buffer(3)]],
    device float* partial [[buffer(4)]],
    constant uint4& dims [[buffer(5)]],
    constant uint& chunks [[buffer(6)]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint2 tile [[threadgroup_position_in_grid]]
) {
    const uint out_dim = dims.x;
    const uint packed_cols = dims.z;
    const uint groups = dims.w;
    const uint row = tile.y * 8u + simd_gid;
    if (row >= out_dim) {
        return;
    }
    const uint words = packed_cols / chunks;
    const uint first = tile.x * words;
    float acc = 0.0f;
    for (uint w = first + lane; w < first + words; w += 32u) {
        const uint word = packed[row * packed_cols + w];
        const uint col = w * 4u;
        const uint affine_index = row * groups + min(col / 64u, groups - 1u);
        const float scale = scales[affine_index];
        const float bias = biases[affine_index];
        for (uint k = 0u; k < 4u; ++k) {
            const float q = float((word >> (8u * k)) & 0xffu);
            acc += lhs[col + k] * ((q * scale) + bias);
        }
    }
    acc = simd_sum(acc);
    if (lane == 0u) {
        partial[row * chunks + tile.x] = acc;
    }
}

// Entrée de branche `x[d] = (Σ_s σ(up[s,d])·n[s,d]) / hc`.
kernel void flash_next_hc_mix_f32(
    device const float* normed [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* x [[buffer(2)]],
    constant uint2& dims [[buffer(3)]],
    uint d [[thread_position_in_grid]]
) {
    const uint hidden = dims.x;
    const uint hc = dims.y;
    if (d >= hidden) {
        return;
    }
    float acc = 0.0f;
    for (uint s = 0u; s < hc; ++s) {
        const uint i = s * hidden + d;
        acc += (1.0f / (1.0f + exp(-up[i]))) * normed[i];
    }
    x[d] = acc / float(hc);
}

// `out[s,d] = h[s,d] + 2σ(inj[s]/hc)·f[d]`, `inj[s]` réduit ici des
// `chunks` sommes partielles de l'injection. dims = (hidden, hc, chunks, 0).
kernel void flash_next_hc_combine_f32(
    device const float* h [[buffer(0)]],
    device const float* partial [[buffer(1)]],
    device const float* f [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint4& dims [[buffer(4)]],
    uint i [[thread_position_in_grid]]
) {
    const uint hidden = dims.x;
    const uint hc = dims.y;
    const uint chunks = dims.z;
    if (i >= hidden * hc) {
        return;
    }
    const uint s = i / hidden;
    float inj = 0.0f;
    for (uint c = 0u; c < chunks; ++c) {
        inj += partial[s * chunks + c];
    }
    const float w = 2.0f * (1.0f / (1.0f + exp(-(inj / float(hc)))));
    out[i] = h[i] + w * f[i % hidden];
}

// PLE : `emb[i] = lut[octet[i]]`, avec `lut` = E4M3 × échelle per-tensor
// (les NaN sont refusés côté CPU à la lecture).
kernel void flash_next_ple_gather_f32(
    device const uchar* bytes [[buffer(0)]],
    device const float* lut [[buffer(1)]],
    device float* emb [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < n) {
        emb[i] = lut[bytes[i]];
    }
}

// PLE : par flux `s`, `d = ⟨k̂_s, q̂_s⟩/√hidden`,
// `g = σ(sign(d)·√max(|d|, 1e-6))`, `gated_s = g·value`.
kernel void flash_next_ple_gate_f32(
    device const float* key [[buffer(0)]],
    device const float* query [[buffer(1)]],
    device const float* value [[buffer(2)]],
    device float* gated [[buffer(3)]],
    constant uint& hidden [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint s [[threadgroup_position_in_grid]]
) {
    threadgroup float partial[256];
    const uint base = s * hidden;
    float acc = 0.0f;
    for (uint d = tid; d < hidden; d += 256u) {
        acc += key[base + d] * query[base + d];
    }
    partial[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float dot = partial[0] / sqrt(float(hidden));
    const float sign = dot > 0.0f ? 1.0f : (dot < 0.0f ? -1.0f : 0.0f);
    const float g = 1.0f / (1.0f + exp(-(sqrt(max(fabs(dot), 1.0e-6f)) * sign)));
    for (uint d = tid; d < hidden; d += 256u) {
        gated[base + d] = g * value[d];
    }
}

// PLE : conv depthwise dilatée sur l'historique de `normed`, puis
// `out = h + (gated + SiLU(acc))`. `ring` garde les `span = (k−1)·dilation`
// dernières colonnes ; le slot du pas courant est relu (t − span) avant
// d'être réécrit. dims = (largeur, noyau, dilation, pas).
kernel void flash_next_ple_conv_add_f32(
    device const float* h [[buffer(0)]],
    device const float* gated [[buffer(1)]],
    device const float* normed [[buffer(2)]],
    device float* ring [[buffer(3)]],
    device const float* conv [[buffer(4)]],
    device float* out [[buffer(5)]],
    constant uint4& dims [[buffer(6)]],
    uint c [[thread_position_in_grid]]
) {
    const uint width = dims.x;
    const uint k = dims.y;
    const uint dilation = dims.z;
    const uint step = dims.w;
    if (c >= width) {
        return;
    }
    const uint span = (k - 1u) * dilation;
    float acc = 0.0f;
    for (uint j = 0u; j < k; ++j) {
        const uint back = span - j * dilation;
        if (back == 0u) {
            acc += conv[c * k + j] * normed[c];
        } else if (step >= back) {
            acc += conv[c * k + j] * ring[((step - back) % span) * width + c];
        }
    }
    const float ple = gated[c] + acc * (1.0f / (1.0f + exp(-acc)));
    out[c] = h[c] + ple;
    if (span > 0u) {
        ring[(step % span) * width + c] = normed[c];
    }
}

// QSA creuse — pooling d'un bloc de l'indexeur : moyenne des 4 clés brutes
// (`raw[k, d]`), `k_layernorm` à gain `1 + w`, RoPE (`freqs`, 2·|freqs|
// premières dimensions, `rotate_half`) à la position `start` du bloc ; écrit
// `pooled[block, :]`. Un threadgroup de `d` threads (d = 128).
kernel void flash_next_index_pool_f32(
    device const float* raw [[buffer(0)]],
    device const float* norm [[buffer(1)]],
    device const float* freqs [[buffer(2)]],
    device float* pooled [[buffer(3)]],
    constant uint4& dims [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]
) {
    const uint d = dims.x;
    const uint compress = dims.y;
    const uint block = dims.z;
    const uint half_rot = dims.w;
    threadgroup float values[256];
    threadgroup float partial[8];
    float v = 0.0f;
    for (uint t = 0u; t < compress; ++t) {
        v += raw[t * d + tid];
    }
    v = v / float(compress);
    const float ss = simd_sum(v * v);
    if (lane == 0u) {
        partial[simd] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint k = 0u; k < d / 32u; ++k) {
        total += partial[k];
    }
    const float inv = 1.0f / sqrt(total / float(d) + eps);
    values[tid] = (v * inv) * norm[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float out = values[tid];
    const float position = float(block * compress);
    if (tid < half_rot) {
        const float angle = position * freqs[tid];
        out = values[tid] * cos(angle) - values[tid + half_rot] * sin(angle);
    } else if (tid < 2u * half_rot) {
        const float angle = position * freqs[tid - half_rot];
        out = values[tid] * cos(angle) + values[tid - half_rot] * sin(angle);
    }
    pooled[block * d + tid] = out;
}

// QSA creuse — scores des blocs complets : `s_b = Σ_h ReLU(q_h·k̄_b)/√d + 0`
// (zéros signés normalisés). Un simdgroup par bloc, 8 blocs par threadgroup.
// dims = (têtes, d, blocs complets, 0).
kernel void flash_next_index_scores_f32(
    device const float* q [[buffer(0)]],
    device const float* pooled [[buffer(1)]],
    device float* scores [[buffer(2)]],
    constant uint4& dims [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint tg [[threadgroup_position_in_grid]]
) {
    const uint heads = dims.x;
    const uint d = dims.y;
    const uint blocks = dims.z;
    const uint b = tg * 8u + simd;
    if (b >= blocks) {
        return;
    }
    float s = 0.0f;
    for (uint h = 0u; h < heads; ++h) {
        float acc = 0.0f;
        for (uint i = lane; i < d; i += 32u) {
            acc += q[h * d + i] * pooled[b * d + i];
        }
        s += max(simd_sum(acc), 0.0f);
    }
    if (lane == 0u) {
        scores[b] = s / sqrt(float(d)) + 0.0f;
    }
}

// QSA creuse — top-`keep` blocs par radix exact sur les bits f32 (scores ≥ 0,
// ordre des bits = ordre des valeurs) : 4 passes de 8 bits du poids fort au
// poids faible donnent la valeur seuil T et le nombre `r` d'ex æquo à T à
// retenir ; la compaction en ordre d'indice garde les scores > T et les `r`
// premiers égaux à T (plus petits indices, comme `select_for_query`), triés.
// Un threadgroup de 1024 threads.
kernel void flash_next_index_topk_f32(
    device const float* scores [[buffer(0)]],
    device uint* selected [[buffer(1)]],
    constant uint2& dims [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]
) {
    const uint count = dims.x;
    const uint keep = dims.y;
    threadgroup atomic_uint hist[256];
    threadgroup uint shared_state[2];
    threadgroup uint scan_a[32];
    threadgroup uint scan_b[32];
    uint prefix = 0u;
    uint prefix_mask = 0u;
    uint remaining = keep;
    for (int shift = 24; shift >= 0; shift -= 8) {
        if (tid < 256u) {
            atomic_store_explicit(&hist[tid], 0u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < count; i += 1024u) {
            const uint bits = as_type<uint>(scores[i] + 0.0f);
            if ((bits & prefix_mask) == prefix) {
                atomic_fetch_add_explicit(&hist[(bits >> uint(shift)) & 255u], 1u, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            uint above = 0u;
            for (int bin = 255; bin >= 0; --bin) {
                const uint c = atomic_load_explicit(&hist[bin], memory_order_relaxed);
                if (above + c >= remaining) {
                    shared_state[0] = prefix | (uint(bin) << uint(shift));
                    shared_state[1] = remaining - above;
                    break;
                }
                above += c;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        prefix = shared_state[0];
        remaining = shared_state[1];
        prefix_mask |= 255u << uint(shift);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint threshold = prefix;
    uint out_base = 0u;
    uint eq_taken = 0u;
    for (uint start = 0u; start < count; start += 1024u) {
        const uint i = start + tid;
        const uint bits = i < count ? as_type<uint>(scores[i] + 0.0f) : 0u;
        const uint gt = (i < count && bits > threshold) ? 1u : 0u;
        const uint eq = (i < count && bits == threshold) ? 1u : 0u;
        // Rang des égaux dans le morceau (préfixe exclusif simd puis threadgroup).
        const uint eq_simd = simd_prefix_exclusive_sum(eq);
        const uint eq_total_simd = simd_sum(eq);
        if (lane == 0u) {
            scan_a[simd] = eq_total_simd;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint eq_before = 0u;
        uint eq_chunk = 0u;
        for (uint k = 0u; k < 32u; ++k) {
            eq_before += k < simd ? scan_a[k] : 0u;
            eq_chunk += scan_a[k];
        }
        const uint take_eq = (eq == 1u && eq_taken + eq_before + eq_simd < remaining) ? 1u : 0u;
        const uint sel = gt | take_eq;
        const uint sel_simd = simd_prefix_exclusive_sum(sel);
        const uint sel_total_simd = simd_sum(sel);
        if (lane == 0u) {
            scan_b[simd] = sel_total_simd;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint sel_before = 0u;
        uint sel_chunk = 0u;
        for (uint k = 0u; k < 32u; ++k) {
            sel_before += k < simd ? scan_b[k] : 0u;
            sel_chunk += scan_b[k];
        }
        if (sel == 1u) {
            selected[out_base + sel_before + sel_simd] = i;
        }
        out_base += sel_chunk;
        eq_taken = min(remaining, eq_taken + eq_chunk);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// QSA creuse — copie des lignes K/V retenues (blocs × compress, puis la queue
// `[compress·complets, visible)`) dans un KV compact, dans l'ordre du CPU.
// dims = (kv_dim, keep, complets, visible) ; `compress` en buffer 6.
kernel void flash_next_gather_kv_f32(
    device const float* keys [[buffer(0)]],
    device const float* values [[buffer(1)]],
    device const uint* selected [[buffer(2)]],
    device float* gathered_keys [[buffer(3)]],
    device float* gathered_values [[buffer(4)]],
    constant uint4& dims [[buffer(5)]],
    constant uint& compress [[buffer(6)]],
    uint2 g [[thread_position_in_grid]]
) {
    const uint kv_dim = dims.x;
    const uint keep = dims.y;
    const uint complete = dims.z;
    const uint visible = dims.w;
    const uint picked = keep * compress;
    const uint count = picked + (visible - complete * compress);
    const uint row = g.y;
    const uint col = g.x;
    if (row >= count || col >= kv_dim) {
        return;
    }
    const uint token = row < picked
        ? selected[row / compress] * compress + row % compress
        : complete * compress + (row - picked);
    gathered_keys[row * kv_dim + col] = keys[token * kv_dim + col];
    gathered_values[row * kv_dim + col] = values[token * kv_dim + col];
}
"#;

/// Kernels des oracles seulement.
#[cfg(test)]
const TEST_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// `dst[offset + i] = src[i]`, bit à bit (indices d'experts).
kernel void flash_next_copy_u32(
    device const uint* src [[buffer(0)]],
    device uint* dst [[buffer(1)]],
    constant uint2& dims [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < dims.x) {
        dst[dims.y + i] = src[i];
    }
}
"#;

/// Pipelines des kernels Flash-Next.
pub(crate) struct FlashNextKernels {
    pub(super) rms_norm_1p_groups: ComputePipelineState,
    pub(super) hc_silu_scale: ComputePipelineState,
    pub(super) hc_mix: ComputePipelineState,
    pub(super) hc_combine: ComputePipelineState,
    pub(super) hc_inject_partial: ComputePipelineState,
    pub(super) qmv_u8_splitk: ComputePipelineState,
    pub(super) index_pool: ComputePipelineState,
    pub(super) index_scores: ComputePipelineState,
    pub(super) index_topk: ComputePipelineState,
    pub(super) gather_kv: ComputePipelineState,
    pub(super) ple_gather: ComputePipelineState,
    pub(super) ple_gate: ComputePipelineState,
    pub(super) ple_conv_add: ComputePipelineState,
    /// Oracles : copie d'indices u32 (routage GPU par couche).
    #[cfg(test)]
    pub(super) copy_u32: ComputePipelineState,
}

impl FlashNextKernels {
    /// Compile les kernels (sans fast-math : `exp` et divisions suivent l'oracle).
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si la compilation Metal échoue.
    pub(crate) fn new(device: &Device) -> Result<Self> {
        let mut pipelines = compile_kernels(
            device,
            SOURCE,
            &[
                "flash_next_rms_norm_1p_groups_f32",
                "flash_next_hc_silu_scale_f32",
                "flash_next_hc_mix_f32",
                "flash_next_hc_combine_f32",
                "flash_next_hc_inject_partial_f32",
                "flash_next_qmv_u8_gs64_splitk_f32",
                "flash_next_index_pool_f32",
                "flash_next_index_scores_f32",
                "flash_next_index_topk_f32",
                "flash_next_gather_kv_f32",
                "flash_next_ple_gather_f32",
                "flash_next_ple_gate_f32",
                "flash_next_ple_conv_add_f32",
            ],
            false,
        )?
        .into_iter();
        let mut next = || {
            pipelines
                .next()
                .ok_or_else(|| crate::InferError::Metal("pipeline Flash-Next absent".to_string()))
        };
        let rms_norm_1p_groups = next()?;
        // La norme groupée suppose 1024 threads (32 simdgroups, `partial[32]`).
        if rms_norm_1p_groups.max_total_threads_per_threadgroup() < 1024 {
            return Err(crate::InferError::Metal(format!(
                "norme groupée Flash-Next : {} threads par threadgroup, 1024 requis",
                rms_norm_1p_groups.max_total_threads_per_threadgroup()
            )));
        }
        let kernels = Self {
            rms_norm_1p_groups,
            hc_silu_scale: next()?,
            hc_mix: next()?,
            hc_combine: next()?,
            hc_inject_partial: next()?,
            qmv_u8_splitk: next()?,
            index_pool: next()?,
            index_scores: next()?,
            index_topk: next()?,
            gather_kv: next()?,
            ple_gather: next()?,
            ple_gate: next()?,
            ple_conv_add: next()?,
            #[cfg(test)]
            copy_u32: compile_kernels(device, TEST_SOURCE, &["flash_next_copy_u32"], false)?
                .pop()
                .ok_or_else(|| crate::InferError::Metal("pipeline copy_u32 absent".to_string()))?,
        };
        // Le top-k de l'indexeur suppose 1024 threads (32 simdgroups, pas de 1024).
        if kernels.index_topk.max_total_threads_per_threadgroup() < 1024 {
            return Err(crate::InferError::Metal(format!(
                "top-k de l'indexeur Flash-Next : {} threads par threadgroup, 1024 requis",
                kernels.index_topk.max_total_threads_per_threadgroup()
            )));
        }
        Ok(kernels)
    }
}
