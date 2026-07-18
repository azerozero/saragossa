//! Sources Metal du decode résident et compilation des pipelines.

use super::*;

/// `scores` est un buffer **device** (loué au pool) → `len` non borné.
pub(super) const ATTENTION_DECODE_KERNEL: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

kernel void attention_decode_naive_f32(
    device const float* q [[buffer(0)]],
    device const float* keys [[buffer(1)]],
    device const float* values [[buffer(2)]],
    device float* scores [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant uint4& dims [[buffer(5)]],
    constant uint& window_start [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint q_head [[threadgroup_position_in_grid]]
) {
    const uint q_heads = dims.x;
    const uint kv_heads = dims.y;
    const uint head_dim = dims.z;
    const uint len = dims.w;
    if (q_head >= q_heads || len == 0u) {
        return;
    }

    const uint group = q_heads / kv_heads;
    const uint kv_head = q_head / group;
    const uint kv_dim = kv_heads * head_dim;
    const uint q_start = q_head * head_dim;
    const uint kv_head_start = kv_head * head_dim;
    const uint score_base = q_head * len;
    const float scale = rsqrt(float(head_dim));

    threadgroup float partial[256];

    // Phase 1 : scores[r] = dot(q[head], keys[r, kv_head]) * scale
    for (uint r = window_start; r < len; ++r) {
        const uint k_start = r * kv_dim + kv_head_start;
        float dot = 0.0f;
        for (uint c = tid; c < head_dim; c += 256u) {
            dot += q[q_start + c] * keys[k_start + c];
        }
        partial[tid] = dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 128u; stride > 0u; stride >>= 1u) {
            if (tid < stride) {
                partial[tid] += partial[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0u) {
            scores[score_base + r] = partial[0] * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Phase 2 : softmax causal (thread 0, sériel sur len — acceptable en 1b)
    if (tid == 0u) {
        float max_score = scores[score_base + window_start];
        for (uint r = window_start + 1u; r < len; ++r) {
            max_score = max(max_score, scores[score_base + r]);
        }
        float sum = 0.0f;
        for (uint r = window_start; r < len; ++r) {
            const float value = exp(scores[score_base + r] - max_score);
            scores[score_base + r] = value;
            sum += value;
        }
        const float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
        for (uint r = window_start; r < len; ++r) {
            scores[score_base + r] *= inv_sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3 : out[head, c] = Σ_r softmax[r] * values[r, kv_head, c]
    for (uint c = tid; c < head_dim; c += 256u) {
        float acc = 0.0f;
        for (uint r = window_start; r < len; ++r) {
            acc += scores[score_base + r] * values[r * kv_dim + kv_head_start + c];
        }
        out[q_start + c] = acc;
    }
}

kernel void attention_decode_naive_bf16(
    device const float* q [[buffer(0)]],
    device const bfloat* keys [[buffer(1)]],
    device const bfloat* values [[buffer(2)]],
    device float* scores [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant uint4& dims [[buffer(5)]],
    constant uint& window_start [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint q_head [[threadgroup_position_in_grid]]
) {
    const uint q_heads = dims.x;
    const uint kv_heads = dims.y;
    const uint head_dim = dims.z;
    const uint len = dims.w;
    if (q_head >= q_heads || len == 0u) {
        return;
    }

    const uint group = q_heads / kv_heads;
    const uint kv_head = q_head / group;
    const uint kv_dim = kv_heads * head_dim;
    const uint q_start = q_head * head_dim;
    const uint kv_head_start = kv_head * head_dim;
    const uint score_base = q_head * len;
    const float scale = rsqrt(float(head_dim));

    threadgroup float partial[256];

    for (uint r = window_start; r < len; ++r) {
        const uint k_start = r * kv_dim + kv_head_start;
        float dot = 0.0f;
        for (uint c = tid; c < head_dim; c += 256u) {
            dot += q[q_start + c] * float(keys[k_start + c]);
        }
        partial[tid] = dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 128u; stride > 0u; stride >>= 1u) {
            if (tid < stride) {
                partial[tid] += partial[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0u) {
            scores[score_base + r] = partial[0] * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        float max_score = scores[score_base + window_start];
        for (uint r = window_start + 1u; r < len; ++r) {
            max_score = max(max_score, scores[score_base + r]);
        }
        float sum = 0.0f;
        for (uint r = window_start; r < len; ++r) {
            const float value = exp(scores[score_base + r] - max_score);
            scores[score_base + r] = value;
            sum += value;
        }
        const float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
        for (uint r = window_start; r < len; ++r) {
            scores[score_base + r] *= inv_sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint c = tid; c < head_dim; c += 256u) {
        float acc = 0.0f;
        for (uint r = window_start; r < len; ++r) {
            acc += scores[score_base + r] * float(values[r * kv_dim + kv_head_start + c]);
        }
        out[q_start + c] = acc;
    }
}

kernel void attention_decode_flash_f32(
    device const float* q [[buffer(0)]],
    device const float* keys [[buffer(1)]],
    device const float* values [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint4& dims [[buffer(4)]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint q_head [[threadgroup_position_in_grid]]
) {
    constexpr uint BN = 32u;
    constexpr uint BD = 32u;
    constexpr uint MAX_PER_THREAD = 8u;

    const uint q_heads = dims.x;
    const uint kv_heads = dims.y;
    const uint head_dim = dims.z;
    const uint len = dims.w;
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
    const uint q_start = q_head * head_dim;
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
                score += q_lane[j] * keys[k_start + c];
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
                o_lane[j] = o_lane[j] * factor + exp_score * values[v_start + c];
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

kernel void attention_decode_flash_bf16(
    device const float* q [[buffer(0)]],
    device const bfloat* keys [[buffer(1)]],
    device const bfloat* values [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint4& dims [[buffer(4)]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint q_head [[threadgroup_position_in_grid]]
) {
    constexpr uint BN = 32u;
    constexpr uint BD = 32u;
    constexpr uint MAX_PER_THREAD = 8u;

    const uint q_heads = dims.x;
    const uint kv_heads = dims.y;
    const uint head_dim = dims.z;
    const uint len = dims.w;
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
    const uint q_start = q_head * head_dim;
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

kernel void attention_decode_flash_d256_f32(
    device const float* q [[buffer(0)]],
    device const float* keys [[buffer(1)]],
    device const float* values [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint4& dims [[buffer(4)]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint q_head [[threadgroup_position_in_grid]]
) {
    constexpr uint BN = 32u;
    constexpr uint BD = 32u;
    constexpr uint HEAD_DIM = 256u;
    constexpr uint ELEMS_PER_THREAD = HEAD_DIM / BD;

    const uint q_heads = dims.x;
    const uint kv_heads = dims.y;
    const uint len = dims.w;
    if (q_head >= q_heads || len == 0u) {
        return;
    }

    const uint group = q_heads / kv_heads;
    const uint kv_head = q_head / group;
    const uint kv_dim = kv_heads * HEAD_DIM;
    const uint q_start = q_head * HEAD_DIM;
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
            score += q_lane[j] * keys[k_start + c];
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
            o_lane[j] = o_lane[j] * factor + exp_score * values[v_start + c];
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

kernel void attention_decode_flash_d256_bf16(
    device const float* q [[buffer(0)]],
    device const bfloat* keys [[buffer(1)]],
    device const bfloat* values [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint4& dims [[buffer(4)]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint q_head [[threadgroup_position_in_grid]]
) {
    constexpr uint BN = 32u;
    constexpr uint BD = 32u;
    constexpr uint HEAD_DIM = 256u;
    constexpr uint ELEMS_PER_THREAD = HEAD_DIM / BD;

    const uint q_heads = dims.x;
    const uint kv_heads = dims.y;
    const uint len = dims.w;
    if (q_head >= q_heads || len == 0u) {
        return;
    }

    const uint group = q_heads / kv_heads;
    const uint kv_head = q_head / group;
    const uint kv_dim = kv_heads * HEAD_DIM;
    const uint q_start = q_head * HEAD_DIM;
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

// SDPA decode 2-passes split-K (head_dim 128/256), aligné sur `sdpa_vector_2pass`
// de mlx : passe 1 = un threadgroup par (kv_head, bloc) avec gqa_factor
// simdgroups (un q_head chacun) → les lectures KV redondantes du groupe GQA
// touchent le L1 du threadgroup (PAS la DRAM), et la longueur est découpée en
// `blocks` (parallélisme + tuiles L1). Passe 2 = réduction online-softmax des
// `blocks` partiels par q_head. Gain visé : lecture KV unique (≈ ×8 moins de
// trafic DRAM qu'en single-pass où chaque q_head est un threadgroup distinct).
kernel void attention_decode_2pass_1_d128_f32(
    device const float* q [[buffer(0)]],
    device const float* keys [[buffer(1)]],
    device const float* values [[buffer(2)]],
    device float* partials [[buffer(3)]],
    device float* sums [[buffer(4)]],
    device float* maxs [[buffer(5)]],
    constant uint4& dims [[buffer(6)]],
    uint3 tidtg [[thread_position_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
    constexpr uint BD = 32u;
    constexpr uint D = 128u;
    constexpr uint QK = D / BD; // 4 éléments/lane
    const uint q_heads = dims.x;
    const uint kv_heads = dims.y;
    const uint len = dims.z;
    const uint blocks = dims.w;
    const uint gqa = q_heads / kv_heads;
    const uint kv_head = tid.x;
    const uint block_idx = tid.y;
    const uint q_in_group = tidtg.y; // simdgroup = un q_head du groupe GQA
    const uint q_head = kv_head * gqa + q_in_group;
    if (q_head >= q_heads || block_idx >= blocks) { return; }
    const uint kv_dim = kv_heads * D;
    const float scale = rsqrt(float(D));

    float ql[QK];
    {
        const device float* q_ = q + q_head * D + simd_lid * QK;
        for (uint j = 0u; j < QK; ++j) { ql[j] = scale * q_[j]; }
    }
    const device float* k_ = keys + block_idx * kv_dim + kv_head * D + simd_lid * QK;
    const device float* v_ = values + block_idx * kv_dim + kv_head * D + simd_lid * QK;

    float o[QK];
    for (uint j = 0u; j < QK; ++j) { o[j] = 0.0f; }
    float max_score = -INFINITY;
    float sum_exp = 0.0f;

    for (uint i = block_idx; i < len; i += blocks) {
        float score = 0.0f;
        for (uint j = 0u; j < QK; ++j) { score += ql[j] * k_[j]; }
        score = simd_sum(score);
        const float new_max = max(max_score, score);
        const float factor = fast::exp(max_score - new_max);
        const float exp_score = fast::exp(score - new_max);
        max_score = new_max;
        sum_exp = sum_exp * factor + exp_score;
        for (uint j = 0u; j < QK; ++j) { o[j] = o[j] * factor + exp_score * v_[j]; }
        k_ += blocks * kv_dim;
        v_ += blocks * kv_dim;
    }

    if (simd_lid == 0u) {
        sums[q_head * blocks + block_idx] = sum_exp;
        maxs[q_head * blocks + block_idx] = max_score;
    }
    device float* out_ = partials + (q_head * blocks + block_idx) * D + simd_lid * QK;
    for (uint j = 0u; j < QK; ++j) { out_[j] = o[j]; }
}

kernel void attention_decode_2pass_1_d128_bf16(
    device const float* q [[buffer(0)]],
    device const bfloat* keys [[buffer(1)]],
    device const bfloat* values [[buffer(2)]],
    device float* partials [[buffer(3)]],
    device float* sums [[buffer(4)]],
    device float* maxs [[buffer(5)]],
    constant uint4& dims [[buffer(6)]],
    uint3 tidtg [[thread_position_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
    constexpr uint BD = 32u;
    constexpr uint D = 128u;
    constexpr uint QK = D / BD; // 4 éléments/lane
    const uint q_heads = dims.x;
    const uint kv_heads = dims.y;
    const uint len = dims.z;
    const uint blocks = dims.w;
    const uint gqa = q_heads / kv_heads;
    const uint kv_head = tid.x;
    const uint block_idx = tid.y;
    const uint q_in_group = tidtg.y; // simdgroup = un q_head du groupe GQA
    const uint q_head = kv_head * gqa + q_in_group;
    if (q_head >= q_heads || block_idx >= blocks) { return; }
    const uint kv_dim = kv_heads * D;
    const float scale = rsqrt(float(D));

    float ql[QK];
    {
        const device float* q_ = q + q_head * D + simd_lid * QK;
        for (uint j = 0u; j < QK; ++j) { ql[j] = scale * q_[j]; }
    }
    const device bfloat* k_ = keys + block_idx * kv_dim + kv_head * D + simd_lid * QK;
    const device bfloat* v_ = values + block_idx * kv_dim + kv_head * D + simd_lid * QK;

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
        sums[q_head * blocks + block_idx] = sum_exp;
        maxs[q_head * blocks + block_idx] = max_score;
    }
    device float* out_ = partials + (q_head * blocks + block_idx) * D + simd_lid * QK;
    for (uint j = 0u; j < QK; ++j) { out_[j] = o[j]; }
}

kernel void attention_decode_2pass_1_d256_f32(
    device const float* q [[buffer(0)]],
    device const float* keys [[buffer(1)]],
    device const float* values [[buffer(2)]],
    device float* partials [[buffer(3)]],
    device float* sums [[buffer(4)]],
    device float* maxs [[buffer(5)]],
    constant uint4& dims [[buffer(6)]],
    uint3 tidtg [[thread_position_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
    constexpr uint BD = 32u;
    constexpr uint D = 256u;
    constexpr uint QK = D / BD; // 8 éléments/lane
    const uint q_heads = dims.x;
    const uint kv_heads = dims.y;
    const uint len = dims.z;
    const uint blocks = dims.w;
    const uint gqa = q_heads / kv_heads;
    const uint kv_head = tid.x;
    const uint block_idx = tid.y;
    const uint q_in_group = tidtg.y; // simdgroup = un q_head du groupe GQA
    const uint q_head = kv_head * gqa + q_in_group;
    if (q_head >= q_heads || block_idx >= blocks) { return; }
    const uint kv_dim = kv_heads * D;
    const float scale = rsqrt(float(D));

    float ql[QK];
    {
        const device float* q_ = q + q_head * D + simd_lid * QK;
        for (uint j = 0u; j < QK; ++j) { ql[j] = scale * q_[j]; }
    }
    const device float* k_ = keys + block_idx * kv_dim + kv_head * D + simd_lid * QK;
    const device float* v_ = values + block_idx * kv_dim + kv_head * D + simd_lid * QK;

    float o[QK];
    for (uint j = 0u; j < QK; ++j) { o[j] = 0.0f; }
    float max_score = -INFINITY;
    float sum_exp = 0.0f;

    for (uint i = block_idx; i < len; i += blocks) {
        float score = 0.0f;
        for (uint j = 0u; j < QK; ++j) { score += ql[j] * k_[j]; }
        score = simd_sum(score);
        const float new_max = max(max_score, score);
        const float factor = fast::exp(max_score - new_max);
        const float exp_score = fast::exp(score - new_max);
        max_score = new_max;
        sum_exp = sum_exp * factor + exp_score;
        for (uint j = 0u; j < QK; ++j) { o[j] = o[j] * factor + exp_score * v_[j]; }
        k_ += blocks * kv_dim;
        v_ += blocks * kv_dim;
    }

    if (simd_lid == 0u) {
        sums[q_head * blocks + block_idx] = sum_exp;
        maxs[q_head * blocks + block_idx] = max_score;
    }
    device float* out_ = partials + (q_head * blocks + block_idx) * D + simd_lid * QK;
    for (uint j = 0u; j < QK; ++j) { out_[j] = o[j]; }
}

kernel void attention_decode_2pass_1_d256_bf16(
    device const float* q [[buffer(0)]],
    device const bfloat* keys [[buffer(1)]],
    device const bfloat* values [[buffer(2)]],
    device float* partials [[buffer(3)]],
    device float* sums [[buffer(4)]],
    device float* maxs [[buffer(5)]],
    constant uint4& dims [[buffer(6)]],
    uint3 tidtg [[thread_position_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
    constexpr uint BD = 32u;
    constexpr uint D = 256u;
    constexpr uint QK = D / BD; // 8 éléments/lane
    const uint q_heads = dims.x;
    const uint kv_heads = dims.y;
    const uint len = dims.z;
    const uint blocks = dims.w;
    const uint gqa = q_heads / kv_heads;
    const uint kv_head = tid.x;
    const uint block_idx = tid.y;
    const uint q_in_group = tidtg.y; // simdgroup = un q_head du groupe GQA
    const uint q_head = kv_head * gqa + q_in_group;
    if (q_head >= q_heads || block_idx >= blocks) { return; }
    const uint kv_dim = kv_heads * D;
    const float scale = rsqrt(float(D));

    float ql[QK];
    {
        const device float* q_ = q + q_head * D + simd_lid * QK;
        for (uint j = 0u; j < QK; ++j) { ql[j] = scale * q_[j]; }
    }
    const device bfloat* k_ = keys + block_idx * kv_dim + kv_head * D + simd_lid * QK;
    const device bfloat* v_ = values + block_idx * kv_dim + kv_head * D + simd_lid * QK;

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
        sums[q_head * blocks + block_idx] = sum_exp;
        maxs[q_head * blocks + block_idx] = max_score;
    }
    device float* out_ = partials + (q_head * blocks + block_idx) * D + simd_lid * QK;
    for (uint j = 0u; j < QK; ++j) { out_[j] = o[j]; }
}

kernel void attention_decode_2pass_2_d256_f32(
    device const float* partials [[buffer(0)]],
    device const float* sums [[buffer(1)]],
    device const float* maxs [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& blocks [[buffer(4)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
    constexpr uint BN = 32u;
    constexpr uint BD = 32u;
    constexpr uint D = 256u;
    constexpr uint EPT = D / BD; // 8
    const uint head_idx = tid.x;

    const device float* part = partials + head_idx * blocks * D + simd_gid * D + simd_lid * EPT;
    const device float* sm = sums + head_idx * blocks;
    const device float* mx = maxs + head_idx * blocks;
    device float* out_ = out + head_idx * D + simd_gid * EPT;

    float o[EPT];
    for (uint i = 0u; i < EPT; ++i) { o[i] = 0.0f; }
    threadgroup float outputs[BN * BD];

    float max_score = -INFINITY;
    for (uint b = 0u; b < blocks / BN; ++b) {
        max_score = max(max_score, mx[simd_lid + BN * b]);
    }
    max_score = simd_max(max_score);

    float sum_exp = 0.0f;
    for (uint b = 0u; b < blocks / BN; ++b) {
        const float factor = fast::exp(mx[simd_lid + BN * b] - max_score);
        sum_exp += factor * sm[simd_lid + BN * b];
    }
    sum_exp = simd_sum(sum_exp);

    const device float* mx_b = mx;
    const device float* part_b = part;
    for (uint b = 0u; b < blocks / BN; ++b) {
        const float factor = fast::exp(mx_b[simd_gid] - max_score);
        for (uint i = 0u; i < EPT; ++i) { o[i] += factor * part_b[i]; }
        mx_b += BN;
        part_b += BN * D;
    }

    for (uint i = 0u; i < EPT; ++i) {
        outputs[simd_lid * BD + simd_gid] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float reduced = simd_sum(outputs[simd_gid * BD + simd_lid]);
        reduced = (sum_exp == 0.0f) ? reduced : (reduced / sum_exp);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_lid == 0u) {
            out_[i] = reduced;
        }
    }
}

kernel void attention_decode_2pass_2_d128_f32(
    device const float* partials [[buffer(0)]],
    device const float* sums [[buffer(1)]],
    device const float* maxs [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& blocks [[buffer(4)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
    constexpr uint BN = 32u;
    constexpr uint BD = 32u;
    constexpr uint D = 128u;
    constexpr uint EPT = D / BD; // 4
    const uint head_idx = tid.x;

    const device float* part = partials + head_idx * blocks * D + simd_gid * D + simd_lid * EPT;
    const device float* sm = sums + head_idx * blocks;
    const device float* mx = maxs + head_idx * blocks;
    device float* out_ = out + head_idx * D + simd_gid * EPT;

    float o[EPT];
    for (uint i = 0u; i < EPT; ++i) { o[i] = 0.0f; }
    threadgroup float outputs[BN * BD];

    float max_score = -INFINITY;
    for (uint b = 0u; b < blocks / BN; ++b) {
        max_score = max(max_score, mx[simd_lid + BN * b]);
    }
    max_score = simd_max(max_score);

    float sum_exp = 0.0f;
    for (uint b = 0u; b < blocks / BN; ++b) {
        const float factor = fast::exp(mx[simd_lid + BN * b] - max_score);
        sum_exp += factor * sm[simd_lid + BN * b];
    }
    sum_exp = simd_sum(sum_exp);

    const device float* mx_b = mx;
    const device float* part_b = part;
    for (uint b = 0u; b < blocks / BN; ++b) {
        const float factor = fast::exp(mx_b[simd_gid] - max_score);
        for (uint i = 0u; i < EPT; ++i) { o[i] += factor * part_b[i]; }
        mx_b += BN;
        part_b += BN * D;
    }

    for (uint i = 0u; i < EPT; ++i) {
        outputs[simd_lid * BD + simd_gid] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float reduced = simd_sum(outputs[simd_gid * BD + simd_lid]);
        reduced = (sum_exp == 0.0f) ? reduced : (reduced / sum_exp);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_lid == 0u) {
            out_[i] = reduced;
        }
    }
}
"#;

/// Kernels du gate de sortie full-attn (modèle `attn_output_gate=true`).
///
/// `split_q_gate_f32` désinterleave la projection q_proj `[2*q_dim]` (layout par
/// tête `[q_head | gate_head]`, `start = head·2·head_dim`) en `q [q_dim]` et
/// `gate [q_dim]` RÉSIDENTS sur GPU — reproduit `split_attention_gate`
/// (decoder.rs) sans readback CPU. `attn_gate_f32` applique `out = ctx · σ(gate)`
/// (le gate de sortie, après l'attention, hors readback).
pub(super) const GATE_KERNELS: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void split_q_gate_f32(
    device const float* proj [[buffer(0)]],
    device float* q [[buffer(1)]],
    device float* gate [[buffer(2)]],
    constant uint2& dims [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint num_heads = dims.x;
    const uint head_dim = dims.y;
    const uint q_dim = num_heads * head_dim;
    if (gid >= q_dim) { return; }
    const uint head = gid / head_dim;
    const uint col = gid % head_dim;
    const uint base = head * 2u * head_dim;
    q[gid] = proj[base + col];
    gate[gid] = proj[base + head_dim + col];
}

kernel void attn_gate_f32(
    device const float* ctx [[buffer(0)]],
    device const float* gate [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) { return; }
    const float g = 1.0f / (1.0f + exp(-gate[gid]));
    out[gid] = ctx[gid] * g;
}
"#;

/// Kernels du chemin full-attn résident decode : RoPE+norm à une POSITION
/// (single-query) et copie à un OFFSET (append KV device-side).
///
/// `rms_norm_rope_heads_decode_f32` : rms_norm par tête + RoPE à la position
/// `dims.w` (PAS l'index de ligne comme le kernel de prefill, qui roterait à 0).
/// Reproduit `rms_norm_rope_heads_at` (decoder.rs) pour le token courant.
/// `copy_at_f32` : `out[i] = in[i]` ; en liant `out` à un offset, écrit la ligne
/// KV[position] device-side (append, hazard read-after-write prouvé en R3).
/// `copy_at_f32_to_bf16` applique le même append vers un KV compact bf16.
pub(super) const ROPE_DECODE_COPY_KERNELS: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void rms_norm_rope_heads_decode_f32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint4& dims [[buffer(3)]],
    constant float2& params [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint head [[threadgroup_position_in_grid]]
) {
    const uint heads = dims.x;
    const uint head_dim = dims.y;
    const uint rope_dims = dims.z;
    const uint position = dims.w;
    const float eps = params.x;
    const float base_theta = params.y;
    if (head >= heads) { return; }

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
            // Rotate-half : la paire (pair, pair+pairs) tourne a la frequence
            // d'exposant 2*pair/rope_dims (miroir CPU rms_norm_rope_heads_at).
            const uint pair = (col < pairs) ? col : (col - pairs);
            const float exponent = float(2u * pair) / float(rope_dims);
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

kernel void copy_at_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= n) { return; }
    output[i] = input[i];
}

kernel void copy_at_f32_to_bf16(
    device const float* input [[buffer(0)]],
    device bfloat* output [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= n) { return; }
    output[i] = bfloat(input[i]);
}
"#;

/// Compile un kernel Metal embarqué et renvoie son pipeline de calcul.
///
/// # Errors
///
/// Renvoie une erreur si la source ne compile pas, si la fonction est absente,
/// ou si le pipeline est invalide.
pub(super) fn compile_kernel(
    device: &Device,
    source: &str,
    name: &str,
) -> Result<ComputePipelineState> {
    let options = CompileOptions::new();
    options.set_fast_math_enabled(true);
    let library = device
        .new_library_with_source(source, &options)
        .map_err(|error| InferError::Metal(format!("compilation kernel {name}: {error}")))?;
    let function = library
        .get_function(name, None)
        .map_err(|error| InferError::Metal(format!("fonction kernel {name}: {error}")))?;
    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|error| InferError::Metal(format!("pipeline kernel {name}: {error}")))
}
