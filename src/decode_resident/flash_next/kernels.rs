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
// totale) : un threadgroup de 256 par groupe.
kernel void flash_next_rms_norm_1p_groups_f32(
    device const float* x [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& group [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint g [[threadgroup_position_in_grid]]
) {
    threadgroup float partial[256];
    const uint base = g * group;
    float ss = 0.0f;
    for (uint col = tid; col < group; col += 256u) {
        const float v = x[base + col];
        ss += v * v;
    }
    partial[tid] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv = 1.0f / sqrt(partial[0] / float(group) + eps);
    for (uint col = tid; col < group; col += 256u) {
        const uint i = base + col;
        out[i] = (x[i] * inv) * (1.0f + w[i]);
    }
}

// `low = SiLU(low / hc)` en place (`hc` puissance de deux : division exacte).
kernel void flash_next_hc_silu_scale_f32(
    device float* low [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    constant float& hc [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= n) {
        return;
    }
    const float v = low[i] / hc;
    low[i] = v * (1.0f / (1.0f + exp(-v)));
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

// `out[s,d] = h[s,d] + 2σ(inj[s]/hc)·f[d]`.
kernel void flash_next_hc_combine_f32(
    device const float* h [[buffer(0)]],
    device const float* inject [[buffer(1)]],
    device const float* f [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint2& dims [[buffer(4)]],
    uint i [[thread_position_in_grid]]
) {
    const uint hidden = dims.x;
    const uint hc = dims.y;
    if (i >= hidden * hc) {
        return;
    }
    const uint s = i / hidden;
    const float w = 2.0f * (1.0f / (1.0f + exp(-(inject[s] / float(hc)))));
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
"#;

/// Pipelines des kernels Flash-Next.
pub(crate) struct FlashNextKernels {
    pub(super) rms_norm_1p_groups: ComputePipelineState,
    pub(super) hc_silu_scale: ComputePipelineState,
    pub(super) hc_mix: ComputePipelineState,
    pub(super) hc_combine: ComputePipelineState,
    pub(super) ple_gather: ComputePipelineState,
    pub(super) ple_gate: ComputePipelineState,
    pub(super) ple_conv_add: ComputePipelineState,
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
        Ok(Self {
            rms_norm_1p_groups: next()?,
            hc_silu_scale: next()?,
            hc_mix: next()?,
            hc_combine: next()?,
            ple_gather: next()?,
            ple_gate: next()?,
            ple_conv_add: next()?,
        })
    }
}
