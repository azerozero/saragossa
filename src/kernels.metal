
#include <metal_stdlib>
using namespace metal;

kernel void dense_matmul_rhs_t_f32(
    device const float* lhs [[buffer(0)]],
    device const float* rhs [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint3& dims [[buffer(3)]],
    uint lane [[thread_index_in_threadgroup]],
    uint2 tile [[threadgroup_position_in_grid]]
) {
    const uint batch = dims.x;
    const uint out_dim = dims.y;
    const uint in_dim = dims.z;

    const uint o = tile.x;
    const uint b = tile.y;
    if (b >= batch || o >= out_dim) {
        return;
    }

    float acc = 0.0f;
    for (uint k = lane; k < in_dim; k += 32) {
        acc += lhs[(b * in_dim) + k] * rhs[(o * in_dim) + k];
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        out[(b * out_dim) + o] = acc;
    }
}

kernel void affine_matmul_rhs_t_u32_f32(
    device const float* lhs [[buffer(0)]],
    device const uint* packed [[buffer(1)]],
    device const bfloat* scales [[buffer(2)]],
    device const bfloat* biases [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant uint4& dims [[buffer(5)]],
    constant uint4& quant [[buffer(6)]],
    uint lane [[thread_index_in_threadgroup]],
    uint2 tile [[threadgroup_position_in_grid]]
) {
    const uint batch = dims.x;
    const uint out_dim = dims.y;
    const uint in_dim = dims.z;
    const uint packed_cols = dims.w;
    const uint group_size = quant.x;
    const uint bits = quant.y;
    const uint groups = quant.z;

    const uint row = tile.x;
    const uint b = tile.y;
    if (b >= batch || row >= out_dim) {
        return;
    }

    const uint values_per_word = 32 / bits;
    const uint mask = (1u << bits) - 1u;
    float acc = 0.0f;

    for (uint word_col = lane; word_col < packed_cols; word_col += 32) {
        const uint col_base = word_col * values_per_word;
        const uint word = packed[(row * packed_cols) + word_col];
        if ((group_size % values_per_word) == 0u) {
            const uint group = min(col_base / group_size, groups - 1u);
            const uint affine_index = (row * groups) + group;
            const float scale = scales[affine_index];
            const float bias = biases[affine_index];
            for (uint word_lane = 0u; word_lane < values_per_word; ++word_lane) {
                const uint col = col_base + word_lane;
                if (col < in_dim) {
                    const uint q = (word >> (word_lane * bits)) & mask;
                    acc += lhs[(b * in_dim) + col] * ((float(q) * scale) + bias);
                }
            }
        } else {
            for (uint word_lane = 0u; word_lane < values_per_word; ++word_lane) {
                const uint col = col_base + word_lane;
                if (col < in_dim) {
                    const uint group = min(col / group_size, groups - 1u);
                    const uint affine_index = (row * groups) + group;
                    const float scale = scales[affine_index];
                    const float bias = biases[affine_index];
                    const uint q = (word >> (word_lane * bits)) & mask;
                    acc += lhs[(b * in_dim) + col] * ((float(q) * scale) + bias);
                }
            }
        }
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        out[(b * out_dim) + row] = acc;
    }
}

kernel void embed_gather_dense_from_u32_f32(
    device const float* table [[buffer(0)]],
    device const uint* token_index [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint2& dims [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    const uint vocab = dims.x;
    const uint dim = dims.y;
    const uint token = token_index[0];
    if (token >= vocab || tid >= dim) {
        return;
    }
    out[tid] = table[token * dim + tid];
}

kernel void embed_gather_affine_from_u32_f32(
    device const uint* packed [[buffer(0)]],
    device const bfloat* scales [[buffer(1)]],
    device const bfloat* biases [[buffer(2)]],
    device const uint* token_index [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant uint4& dims [[buffer(5)]],
    constant uint4& quant [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    const uint vocab = dims.x;
    const uint dim = dims.y;
    const uint packed_cols = dims.z;
    const uint groups = dims.w;
    const uint group_size = quant.x;
    const uint bits = quant.y;
    const uint token = token_index[0];
    if (token >= vocab || tid >= dim || bits == 0u) {
        return;
    }
    const uint values_per_word = 32u / bits;
    const uint mask = (1u << bits) - 1u;
    const uint word_col = tid / values_per_word;
    const uint lane = tid % values_per_word;
    const uint word = packed[token * packed_cols + word_col];
    const uint q = (word >> (lane * bits)) & mask;
    const uint group = min(tid / group_size, groups - 1u);
    const uint affine_index = token * groups + group;
    out[tid] = float(q) * scales[affine_index] + biases[affine_index];
}

kernel void affine_qmv_fast_u4_gs64_f32(
    device const float* lhs [[buffer(0)]],
    device const uint* packed [[buffer(1)]],
    device const bfloat* scales [[buffer(2)]],
    device const bfloat* biases [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant uint4& dims [[buffer(5)]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint2 tile [[threadgroup_position_in_grid]]
) {
    const uint out_dim = dims.x;
    const uint in_dim = dims.y;
    const uint packed_cols = dims.z;
    const uint groups = dims.w;
    const uint results_per_simdgroup = 4u;
    const uint simdgroups = 2u;
    const uint values_per_thread = 16u;
    const uint block_size = values_per_thread * 32u;
    const uint row_base = tile.y * (simdgroups * results_per_simdgroup) +
        simd_gid * results_per_simdgroup;
    if (row_base >= out_dim) {
        return;
    }

    const uint row_bytes = packed_cols * 4u;
    const uint scale_step_per_thread = 4u;
    const device uchar* ws = ((const device uchar*)packed) +
        row_base * row_bytes + simd_lid * 8u;
    const device bfloat* scale_base = scales + row_base * groups + simd_lid / scale_step_per_thread;
    const device bfloat* bias_base = biases + row_base * groups + simd_lid / scale_step_per_thread;
    const device float* x = lhs + tile.x * in_dim + simd_lid * values_per_thread;

    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint k = 0u; k < in_dim; k += block_size) {
        float xt[16];
        float sum = 0.0f;
        for (uint i = 0u; i < values_per_thread; i += 4u) {
            const float x0 = x[i];
            const float x1 = x[i + 1u];
            const float x2 = x[i + 2u];
            const float x3 = x[i + 3u];
            sum += x0 + x1 + x2 + x3;
            xt[i] = x0;
            xt[i + 1u] = x1 / 16.0f;
            xt[i + 2u] = x2 / 256.0f;
            xt[i + 3u] = x3 / 4096.0f;
        }

        for (uint row = 0u; row < results_per_simdgroup; ++row) {
            if (row_base + row < out_dim) {
                const device ushort* w16 = (const device ushort*)(ws + row * row_bytes);
                const float scale = scale_base[row * groups];
                const float bias = bias_base[row * groups];
                float accum = 0.0f;
                for (uint i = 0u; i < 4u; ++i) {
                    const ushort word = w16[i];
                    accum += xt[4u * i] * float(word & 0x000fu);
                    accum += xt[4u * i + 1u] * float(word & 0x00f0u);
                    accum += xt[4u * i + 2u] * float(word & 0x0f00u);
                    accum += xt[4u * i + 3u] * float(word & 0xf000u);
                }
                result[row] += scale * accum + sum * bias;
            }
        }

        ws += 256u;
        scale_base += 8u;
        bias_base += 8u;
        x += block_size;
    }

    for (uint row = 0u; row < results_per_simdgroup; ++row) {
        const float reduced = simd_sum(result[row]);
        if (simd_lid == 0u && row_base + row < out_dim) {
            out[tile.x * out_dim + row_base + row] = reduced;
        }
    }
}

kernel void affine_qmv_fast_aligned_u4_gs64_f32(
    device const float* lhs [[buffer(0)]],
    device const uint* packed [[buffer(1)]],
    device const bfloat* scales [[buffer(2)]],
    device const bfloat* biases [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant uint4& dims [[buffer(5)]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint2 tile [[threadgroup_position_in_grid]]
) {
    const uint out_dim = dims.x;
    const uint in_dim = dims.y;
    const uint packed_cols = dims.z;
    const uint groups = dims.w;
    const uint results_per_simdgroup = 4u;
    const uint simdgroups = 2u;
    const uint values_per_thread = 16u;
    const uint block_size = values_per_thread * 32u;
    const uint row_base = tile.y * (simdgroups * results_per_simdgroup) +
        simd_gid * results_per_simdgroup;

    const uint row_bytes = packed_cols * 4u;
    const uint scale_step_per_thread = 4u;
    const device uchar* ws = ((const device uchar*)packed) +
        row_base * row_bytes + simd_lid * 8u;
    const device bfloat* scale_base = scales + row_base * groups + simd_lid / scale_step_per_thread;
    const device bfloat* bias_base = biases + row_base * groups + simd_lid / scale_step_per_thread;
    const device float* x = lhs + tile.x * in_dim + simd_lid * values_per_thread;

    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint k = 0u; k < in_dim; k += block_size) {
        float xt[16];
        float sum = 0.0f;
        for (uint i = 0u; i < values_per_thread; i += 4u) {
            const float x0 = x[i];
            const float x1 = x[i + 1u];
            const float x2 = x[i + 2u];
            const float x3 = x[i + 3u];
            sum += x0 + x1 + x2 + x3;
            xt[i] = x0;
            xt[i + 1u] = x1 / 16.0f;
            xt[i + 2u] = x2 / 256.0f;
            xt[i + 3u] = x3 / 4096.0f;
        }

        for (uint row = 0u; row < results_per_simdgroup; ++row) {
            const device ushort* w16 = (const device ushort*)(ws + row * row_bytes);
            const float scale = scale_base[row * groups];
            const float bias = bias_base[row * groups];
            float accum = 0.0f;
            for (uint i = 0u; i < 4u; ++i) {
                const ushort word = w16[i];
                accum += xt[4u * i] * float(word & 0x000fu);
                accum += xt[4u * i + 1u] * float(word & 0x00f0u);
                accum += xt[4u * i + 2u] * float(word & 0x0f00u);
                accum += xt[4u * i + 3u] * float(word & 0xf000u);
            }
            result[row] += scale * accum + sum * bias;
        }

        ws += 256u;
        scale_base += 8u;
        bias_base += 8u;
        x += block_size;
    }

    for (uint row = 0u; row < results_per_simdgroup; ++row) {
        const float reduced = simd_sum(result[row]);
        if (simd_lid == 0u) {
            out[tile.x * out_dim + row_base + row] = reduced;
        }
    }
}

kernel void affine_qmv_fast_aligned_u8_gs64_f32(
    device const float* lhs [[buffer(0)]],
    device const uint* packed [[buffer(1)]],
    device const bfloat* scales [[buffer(2)]],
    device const bfloat* biases [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant uint4& dims [[buffer(5)]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint2 tile [[threadgroup_position_in_grid]]
) {
    const uint out_dim = dims.x;
    const uint in_dim = dims.y;
    const uint packed_cols = dims.z;
    const uint groups = dims.w;
    const uint results_per_simdgroup = 4u;
    const uint simdgroups = 2u;
    const uint values_per_word = 4u;
    const uint words_per_thread = 4u;
    const uint values_per_thread = values_per_word * words_per_thread;
    const uint values_per_block = values_per_thread * 32u;
    const uint words_per_block = values_per_block / values_per_word;
    const uint row_base = tile.y * (simdgroups * results_per_simdgroup) +
        simd_gid * results_per_simdgroup;

    const device uint* ws = packed + row_base * packed_cols + simd_lid;
    const device bfloat* scale_base = scales + row_base * groups;
    const device bfloat* bias_base = biases + row_base * groups;
    const device float* x = lhs + tile.x * in_dim + simd_lid * values_per_word;

    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint k = 0u; k < in_dim; k += values_per_block) {
        float xt[16];
        for (uint word = 0u; word < words_per_thread; ++word) {
            const uint base = word * values_per_word;
            const uint x_offset = word * 32u * values_per_word;
            xt[base] = x[x_offset];
            xt[base + 1u] = x[x_offset + 1u];
            xt[base + 2u] = x[x_offset + 2u];
            xt[base + 3u] = x[x_offset + 3u];
        }

        for (uint row = 0u; row < results_per_simdgroup; ++row) {
            const device uint* row_words = ws + row * packed_cols;
            float accum = 0.0f;
            for (uint word = 0u; word < words_per_thread; ++word) {
                const uint packed_word = row_words[word * 32u];
                const uint group = (simd_lid + word * 32u) / 16u;
                const float scale = scale_base[row * groups + group];
                const float bias = bias_base[row * groups + group];
                const uint base = word * values_per_word;
                accum += xt[base] * ((float(packed_word & 0x000000ffu) * scale) + bias);
                accum += xt[base + 1u] * ((float((packed_word >> 8u) & 0x000000ffu) * scale) + bias);
                accum += xt[base + 2u] * ((float((packed_word >> 16u) & 0x000000ffu) * scale) + bias);
                accum += xt[base + 3u] * ((float((packed_word >> 24u) & 0x000000ffu) * scale) + bias);
            }
            result[row] += accum;
        }

        ws += words_per_block;
        scale_base += values_per_block / 64u;
        bias_base += values_per_block / 64u;
        x += values_per_block;
    }

    for (uint row = 0u; row < results_per_simdgroup; ++row) {
        const float reduced = simd_sum(result[row]);
        if (simd_lid == 0u) {
            out[tile.x * out_dim + row_base + row] = reduced;
        }
    }
}

// qmm petit-M (M=2) : variante de `affine_qmv_fast_aligned` traitant DEUX lignes
// d'activation à la fois, le poids étant lu UNE SEULE FOIS par mot et réutilisé
// pour les 2 lignes → V_batch ≈ 1.1·D (cf. ÉTAPE 1). Sortie bit-identique à 2
// appels de l'aligned. Pour le verify spéculatif MTP (depth-1, M=K+1=2).
// lhs = [2, in_dim] ; out = [2, out_dim] ; dims = [out_dim, in_dim, packed_cols, groups].
kernel void affine_qmm2_fast_aligned_u4_gs64_f32(
    device const float* lhs [[buffer(0)]],
    device const uint* packed [[buffer(1)]],
    device const bfloat* scales [[buffer(2)]],
    device const bfloat* biases [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant uint4& dims [[buffer(5)]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint2 tile [[threadgroup_position_in_grid]]
) {
    const uint out_dim = dims.x;
    const uint in_dim = dims.y;
    const uint packed_cols = dims.z;
    const uint groups = dims.w;
    const uint results_per_simdgroup = 4u;
    const uint simdgroups = 2u;
    const uint values_per_thread = 16u;
    const uint block_size = values_per_thread * 32u;
    const uint row_base = tile.y * (simdgroups * results_per_simdgroup) +
        simd_gid * results_per_simdgroup;

    const uint row_bytes = packed_cols * 4u;
    const uint scale_step_per_thread = 4u;
    const device uchar* ws = ((const device uchar*)packed) +
        row_base * row_bytes + simd_lid * 8u;
    const device bfloat* scale_base = scales + row_base * groups + simd_lid / scale_step_per_thread;
    const device bfloat* bias_base = biases + row_base * groups + simd_lid / scale_step_per_thread;
    const device float* x0 = lhs + simd_lid * values_per_thread;
    const device float* x1 = lhs + in_dim + simd_lid * values_per_thread;

    float r0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float r1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint k = 0u; k < in_dim; k += block_size) {
        float xt0[16];
        float xt1[16];
        float s0 = 0.0f;
        float s1 = 0.0f;
        for (uint i = 0u; i < values_per_thread; i += 4u) {
            const float a0 = x0[i]; const float a1 = x0[i + 1u];
            const float a2 = x0[i + 2u]; const float a3 = x0[i + 3u];
            s0 += a0 + a1 + a2 + a3;
            xt0[i] = a0; xt0[i + 1u] = a1 / 16.0f; xt0[i + 2u] = a2 / 256.0f; xt0[i + 3u] = a3 / 4096.0f;
            const float c0 = x1[i]; const float c1 = x1[i + 1u];
            const float c2 = x1[i + 2u]; const float c3 = x1[i + 3u];
            s1 += c0 + c1 + c2 + c3;
            xt1[i] = c0; xt1[i + 1u] = c1 / 16.0f; xt1[i + 2u] = c2 / 256.0f; xt1[i + 3u] = c3 / 4096.0f;
        }
        for (uint row = 0u; row < results_per_simdgroup; ++row) {
            const device ushort* w16 = (const device ushort*)(ws + row * row_bytes);
            const float scale = scale_base[row * groups];
            const float bias = bias_base[row * groups];
            float ac0 = 0.0f;
            float ac1 = 0.0f;
            for (uint i = 0u; i < 4u; ++i) {
                const ushort word = w16[i];
                ac0 += xt0[4u * i] * float(word & 0x000fu);
                ac0 += xt0[4u * i + 1u] * float(word & 0x00f0u);
                ac0 += xt0[4u * i + 2u] * float(word & 0x0f00u);
                ac0 += xt0[4u * i + 3u] * float(word & 0xf000u);
                ac1 += xt1[4u * i] * float(word & 0x000fu);
                ac1 += xt1[4u * i + 1u] * float(word & 0x00f0u);
                ac1 += xt1[4u * i + 2u] * float(word & 0x0f00u);
                ac1 += xt1[4u * i + 3u] * float(word & 0xf000u);
            }
            r0[row] += scale * ac0 + s0 * bias;
            r1[row] += scale * ac1 + s1 * bias;
        }
        ws += 256u;
        scale_base += 8u;
        bias_base += 8u;
        x0 += block_size;
        x1 += block_size;
    }

    for (uint row = 0u; row < results_per_simdgroup; ++row) {
        const float v0 = simd_sum(r0[row]);
        const float v1 = simd_sum(r1[row]);
        if (simd_lid == 0u) {
            out[row_base + row] = v0;
            out[out_dim + row_base + row] = v1;
        }
    }
}

kernel void swiglu_f32(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& len [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) {
        return;
    }
    const float g = gate[gid];
    out[gid] = (g / (1.0f + exp(-g))) * up[gid];
}

kernel void accumulate_scaled_f32(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant float& scale [[buffer(2)]],
    constant uint& len [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) {
        return;
    }
    dst[gid] += src[gid] * scale;
}

kernel void linear_attn_conv_silu_f32(
    device const float* qkv [[buffer(0)]],
    device const float* conv_weight [[buffer(1)]],
    device float* conv_state [[buffer(2)]],
    device float* conv_out [[buffer(3)]],
    constant uint2& dims [[buffer(4)]],
    uint channel [[thread_position_in_grid]]
) {
    const uint conv_dim = dims.x;
    const uint kernel_width = dims.y;
    if (channel >= conv_dim || kernel_width == 0u) {
        return;
    }

    const uint keep = kernel_width - 1u;
    float acc = 0.0f;
    for (uint k = 0u; k < kernel_width; ++k) {
        const float input = (k < keep) ? conv_state[k * conv_dim + channel] : qkv[channel];
        acc += input * conv_weight[channel * kernel_width + k];
    }
    conv_out[channel] = acc / (1.0f + exp(-acc));

    if (keep > 0u) {
        for (uint row = 0u; row + 1u < keep; ++row) {
            conv_state[row * conv_dim + channel] = conv_state[(row + 1u) * conv_dim + channel];
        }
        conv_state[(keep - 1u) * conv_dim + channel] = qkv[channel];
    }
}

kernel void linear_attn_norm_gates_f32(
    device const float* conv_out [[buffer(0)]],
    device const float* beta_input [[buffer(1)]],
    device const float* gate_input [[buffer(2)]],
    device const float* a_log [[buffer(3)]],
    device const float* dt_bias [[buffer(4)]],
    device float* q_norm [[buffer(5)]],
    device float* k_norm [[buffer(6)]],
    device float* beta [[buffer(7)]],
    device float* decay [[buffer(8)]],
    constant uint4& dims [[buffer(9)]],
    constant float2& scales [[buffer(10)]],
    uint lane [[thread_index_in_threadgroup]],
    uint item [[threadgroup_position_in_grid]]
) {
    const uint key_heads = dims.x;
    const uint value_heads = dims.y;
    const uint key_head_dim = dims.z;
    const uint key_dim = key_heads * key_head_dim;

    if (item < key_heads) {
        const uint base = item * key_head_dim;
        float q_ss = 0.0f;
        float k_ss = 0.0f;
        for (uint col = lane; col < key_head_dim; col += 32u) {
            const float q = conv_out[base + col];
            const float k = conv_out[key_dim + base + col];
            q_ss += q * q;
            k_ss += k * k;
        }
        const float q_mean = simd_sum(q_ss) / float(key_head_dim);
        const float k_mean = simd_sum(k_ss) / float(key_head_dim);
        const float q_inv = rsqrt(q_mean + 1.0e-6f) * scales.x;
        const float k_inv = rsqrt(k_mean + 1.0e-6f) * scales.y;
        for (uint col = lane; col < key_head_dim; col += 32u) {
            q_norm[base + col] = conv_out[base + col] * q_inv;
            k_norm[base + col] = conv_out[key_dim + base + col] * k_inv;
        }
    }

    if (item < value_heads && lane == 0u) {
        const float b = beta_input[item];
        beta[item] = 1.0f / (1.0f + exp(-b));
        const float dt_arg = gate_input[item] + dt_bias[item];
        const float dt = (dt_arg > 20.0f) ? dt_arg : log(1.0f + exp(dt_arg));
        decay[item] = exp(-exp(a_log[item]) * dt);
    }
}

kernel void linear_attn_gated_delta_f32(
    device const float* conv_out [[buffer(0)]],
    device const float* q_norm [[buffer(1)]],
    device const float* k_norm [[buffer(2)]],
    device const float* beta [[buffer(3)]],
    device const float* decay [[buffer(4)]],
    device float* ssm_state [[buffer(5)]],
    device float* y [[buffer(6)]],
    constant uint4& dims [[buffer(7)]],
    uint lane [[thread_index_in_threadgroup]],
    uint2 tile [[threadgroup_position_in_grid]]
) {
    const uint value_heads = dims.x;
    const uint value_head_dim = dims.y;
    const uint key_head_dim = dims.z;
    const uint repeat = dims.w;
    const uint value_head = tile.y;
    const uint value_col = tile.x;
    if (value_head >= value_heads || value_col >= value_head_dim || repeat == 0u) {
        return;
    }

    const uint key_heads = value_heads / repeat;
    const uint key_dim = key_heads * key_head_dim;
    const uint key_head = value_head / repeat;
    const uint key_base = key_head * key_head_dim;
    const uint value_index = value_head * value_head_dim + value_col;
    const uint state_base = value_index * key_head_dim;
    const float d = decay[value_head];

    float kv_part = 0.0f;
    for (uint col = lane; col < key_head_dim; col += 32u) {
        const uint state_index = state_base + col;
        const float decayed = ssm_state[state_index] * d;
        ssm_state[state_index] = decayed;
        kv_part += decayed * k_norm[key_base + col];
    }
    const float kv_mem = simd_sum(kv_part);
    const float v = conv_out[key_dim * 2u + value_index];
    const float delta = (v - kv_mem) * beta[value_head];

    float y_part = 0.0f;
    for (uint col = lane; col < key_head_dim; col += 32u) {
        const uint state_index = state_base + col;
        const float updated = ssm_state[state_index] + delta * k_norm[key_base + col];
        ssm_state[state_index] = updated;
        y_part += updated * q_norm[key_base + col];
    }
    const float out = simd_sum(y_part);
    if (lane == 0u) {
        y[value_index] = out;
    }
}

kernel void linear_attn_rms_gate_f32(
    device const float* y [[buffer(0)]],
    device const float* z [[buffer(1)]],
    device const float* norm_weight [[buffer(2)]],
    device float* gated [[buffer(3)]],
    constant uint2& dims [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint lane [[thread_index_in_threadgroup]],
    uint value_head [[threadgroup_position_in_grid]]
) {
    const uint value_heads = dims.x;
    const uint value_head_dim = dims.y;
    if (value_head >= value_heads) {
        return;
    }

    const uint base = value_head * value_head_dim;
    float ss = 0.0f;
    for (uint col = lane; col < value_head_dim; col += 32u) {
        const float v = y[base + col];
        ss += v * v;
    }
    const float mean = simd_sum(ss) / float(value_head_dim);
    const float inv = rsqrt(mean + eps);
    for (uint col = lane; col < value_head_dim; col += 32u) {
        const uint idx = base + col;
        const float zg = z[idx];
        gated[idx] = y[idx] * inv * norm_weight[col] * (zg / (1.0f + exp(-zg)));
    }
}

kernel void affine_gather_matmul_rhs_t_u32_f32(
    device const float* lhs [[buffer(0)]],
    device const uint* packed [[buffer(1)]],
    device const bfloat* scales [[buffer(2)]],
    device const bfloat* biases [[buffer(3)]],
    device const uint* expert_indices [[buffer(4)]],
    device float* out [[buffer(5)]],
    constant uint4& dims [[buffer(6)]],
    constant uint4& quant [[buffer(7)]],
    uint lane [[thread_index_in_threadgroup]],
    uint2 tile [[threadgroup_position_in_grid]]
) {
    const uint topk = dims.x;
    const uint out_dim = dims.y;
    const uint in_dim = dims.z;
    const uint packed_cols = dims.w;
    const uint group_size = quant.x;
    const uint bits = quant.y;
    const uint groups = quant.z;
    const uint lhs_rows = quant.w;

    const uint row = tile.x;
    const uint slot = tile.y;
    if (slot >= topk || row >= out_dim) {
        return;
    }

    const uint expert = expert_indices[slot];
    uint lhs_row = 0u;
    if (lhs_rows == 1u) {
        lhs_row = 0u;
    } else if (lhs_rows == topk) {
        lhs_row = slot;
    } else {
        const uint slots_per_row = (topk > lhs_rows) ? (topk / lhs_rows) : 1u;
        lhs_row = min(slot / slots_per_row, lhs_rows - 1u);
    }
    const uint values_per_word = 32 / bits;
    const uint mask = (1u << bits) - 1u;
    float acc = 0.0f;

    for (uint word_col = lane; word_col < packed_cols; word_col += 32) {
        const uint col_base = word_col * values_per_word;
        const uint packed_index = ((expert * out_dim + row) * packed_cols) + word_col;
        const uint word = packed[packed_index];
        if ((group_size % values_per_word) == 0u) {
            const uint group = min(col_base / group_size, groups - 1u);
            const uint affine_index = ((expert * out_dim + row) * groups) + group;
            const float scale = scales[affine_index];
            const float bias = biases[affine_index];
            for (uint word_lane = 0u; word_lane < values_per_word; ++word_lane) {
                const uint col = col_base + word_lane;
                if (col < in_dim) {
                    const uint q = (word >> (word_lane * bits)) & mask;
                    acc += lhs[(lhs_row * in_dim) + col] * ((float(q) * scale) + bias);
                }
            }
        } else {
            for (uint word_lane = 0u; word_lane < values_per_word; ++word_lane) {
                const uint col = col_base + word_lane;
                if (col < in_dim) {
                    const uint group = min(col / group_size, groups - 1u);
                    const uint affine_index = ((expert * out_dim + row) * groups) + group;
                    const float scale = scales[affine_index];
                    const float bias = biases[affine_index];
                    const uint q = (word >> (word_lane * bits)) & mask;
                    acc += lhs[(lhs_row * in_dim) + col] * ((float(q) * scale) + bias);
                }
            }
        }
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        out[(slot * out_dim) + row] = acc;
    }
}

kernel void affine_gather_qmv_fast_u4_gs64_f32(
    device const float* lhs [[buffer(0)]],
    device const uint* packed [[buffer(1)]],
    device const bfloat* scales [[buffer(2)]],
    device const bfloat* biases [[buffer(3)]],
    device const uint* expert_indices [[buffer(4)]],
    device float* out [[buffer(5)]],
    constant uint4& dims [[buffer(6)]],
    constant uint4& quant [[buffer(7)]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint2 tile [[threadgroup_position_in_grid]]
) {
    const uint topk = dims.x;
    const uint out_dim = dims.y;
    const uint in_dim = dims.z;
    const uint packed_cols = dims.w;
    const uint groups = quant.z;
    const uint lhs_rows = quant.w;
    const uint slot = tile.x;
    const uint row_base = tile.y * 8u + simd_gid * 4u;
    if (slot >= topk || row_base >= out_dim) {
        return;
    }

    const uint expert = expert_indices[slot];
    uint lhs_row = 0u;
    if (lhs_rows == 1u) {
        lhs_row = 0u;
    } else if (lhs_rows == topk) {
        lhs_row = slot;
    } else {
        const uint slots_per_row = (topk > lhs_rows) ? (topk / lhs_rows) : 1u;
        lhs_row = min(slot / slots_per_row, lhs_rows - 1u);
    }
    const uint row_bytes = packed_cols * 4u;
    const uint scale_step_per_thread = 4u;
    const device uchar* ws = ((const device uchar*)packed) +
        ((expert * out_dim + row_base) * row_bytes) + simd_lid * 8u;
    const device bfloat* scale_base = scales +
        ((expert * out_dim + row_base) * groups) + simd_lid / scale_step_per_thread;
    const device bfloat* bias_base = biases +
        ((expert * out_dim + row_base) * groups) + simd_lid / scale_step_per_thread;
    const device float* x = lhs + lhs_row * in_dim + simd_lid * 16u;

    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint k = 0u; k < in_dim; k += 512u) {
        float xt[16];
        float sum = 0.0f;
        for (uint i = 0u; i < 16u; i += 4u) {
            const float x0 = x[i];
            const float x1 = x[i + 1u];
            const float x2 = x[i + 2u];
            const float x3 = x[i + 3u];
            sum += x0 + x1 + x2 + x3;
            xt[i] = x0;
            xt[i + 1u] = x1 / 16.0f;
            xt[i + 2u] = x2 / 256.0f;
            xt[i + 3u] = x3 / 4096.0f;
        }

        for (uint row = 0u; row < 4u; ++row) {
            if (row_base + row < out_dim) {
                const device ushort* w16 = (const device ushort*)(ws + row * row_bytes);
                const float scale = scale_base[row * groups];
                const float bias = bias_base[row * groups];
                float accum = 0.0f;
                for (uint i = 0u; i < 4u; ++i) {
                    const uint word = uint(w16[i]);
                    accum += xt[4u * i] * float(word & 0x000fu);
                    accum += xt[4u * i + 1u] * float(word & 0x00f0u);
                    accum += xt[4u * i + 2u] * float(word & 0x0f00u);
                    accum += xt[4u * i + 3u] * float(word & 0xf000u);
                }
                result[row] += scale * accum + sum * bias;
            }
        }
        ws += 256u;
        scale_base += 8u;
        bias_base += 8u;
        x += 512u;
    }

    for (uint row = 0u; row < 4u; ++row) {
        const float reduced = simd_sum(result[row]);
        if (simd_lid == 0u && row_base + row < out_dim) {
            out[(slot * out_dim) + row_base + row] = reduced;
        }
    }
}

kernel void affine_gather_qmv_tail_u4_gs64_f32(
    device const float* lhs [[buffer(0)]],
    device const uint* packed [[buffer(1)]],
    device const bfloat* scales [[buffer(2)]],
    device const bfloat* biases [[buffer(3)]],
    device const uint* expert_indices [[buffer(4)]],
    device float* out [[buffer(5)]],
    constant uint4& dims [[buffer(6)]],
    constant uint4& quant [[buffer(7)]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint2 tile [[threadgroup_position_in_grid]]
) {
    const uint topk = dims.x;
    const uint out_dim = dims.y;
    const uint in_dim = dims.z;
    const uint packed_cols = dims.w;
    const uint groups = quant.z;
    const uint lhs_rows = quant.w;
    const uint slot = tile.x;
    const uint row_base = tile.y * 8u + simd_gid * 4u;
    if (slot >= topk || row_base >= out_dim) {
        return;
    }

    const uint expert = expert_indices[slot];
    uint lhs_row = 0u;
    if (lhs_rows == 1u) {
        lhs_row = 0u;
    } else if (lhs_rows == topk) {
        lhs_row = slot;
    } else {
        const uint slots_per_row = (topk > lhs_rows) ? (topk / lhs_rows) : 1u;
        lhs_row = min(slot / slots_per_row, lhs_rows - 1u);
    }
    const uint row_bytes = packed_cols * 4u;
    const uint scale_step_per_thread = 4u;
    const device uchar* ws = ((const device uchar*)packed) +
        ((expert * out_dim + row_base) * row_bytes) + simd_lid * 8u;
    const device bfloat* scale_base = scales +
        ((expert * out_dim + row_base) * groups) + simd_lid / scale_step_per_thread;
    const device bfloat* bias_base = biases +
        ((expert * out_dim + row_base) * groups) + simd_lid / scale_step_per_thread;
    const device float* x = lhs + lhs_row * in_dim + simd_lid * 16u;

    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint k = 0u; k < in_dim; k += 512u) {
        const bool lane_active = k + simd_lid * 16u < in_dim;
        float xt[16];
        float sum = 0.0f;
        if (lane_active) {
            for (uint i = 0u; i < 16u; i += 4u) {
                const float x0 = x[i];
                const float x1 = x[i + 1u];
                const float x2 = x[i + 2u];
                const float x3 = x[i + 3u];
                sum += x0 + x1 + x2 + x3;
                xt[i] = x0;
                xt[i + 1u] = x1 / 16.0f;
                xt[i + 2u] = x2 / 256.0f;
                xt[i + 3u] = x3 / 4096.0f;
            }
        }

        for (uint row = 0u; row < 4u; ++row) {
            if (lane_active && row_base + row < out_dim) {
                const device ushort* w16 = (const device ushort*)(ws + row * row_bytes);
                const float scale = scale_base[row * groups];
                const float bias = bias_base[row * groups];
                float accum = 0.0f;
                for (uint i = 0u; i < 4u; ++i) {
                    const uint word = uint(w16[i]);
                    accum += xt[4u * i] * float(word & 0x000fu);
                    accum += xt[4u * i + 1u] * float(word & 0x00f0u);
                    accum += xt[4u * i + 2u] * float(word & 0x0f00u);
                    accum += xt[4u * i + 3u] * float(word & 0xf000u);
                }
                result[row] += scale * accum + sum * bias;
            }
        }
        ws += 256u;
        scale_base += 8u;
        bias_base += 8u;
        x += 512u;
    }

    for (uint row = 0u; row < 4u; ++row) {
        const float reduced = simd_sum(result[row]);
        if (simd_lid == 0u && row_base + row < out_dim) {
            out[(slot * out_dim) + row_base + row] = reduced;
        }
    }
}

kernel void affine_gather_gate_up_swiglu_fast_u4_gs64_f32(
    device const float* lhs [[buffer(0)]],
    device const uint* gate_packed [[buffer(1)]],
    device const bfloat* gate_scales [[buffer(2)]],
    device const bfloat* gate_biases [[buffer(3)]],
    device const uint* up_packed [[buffer(4)]],
    device const bfloat* up_scales [[buffer(5)]],
    device const bfloat* up_biases [[buffer(6)]],
    device const uint* expert_indices [[buffer(7)]],
    device float* out [[buffer(8)]],
    constant uint4& dims [[buffer(9)]],
    constant uint4& quant [[buffer(10)]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint2 tile [[threadgroup_position_in_grid]]
) {
    const uint topk = dims.x;
    const uint out_dim = dims.y;
    const uint in_dim = dims.z;
    const uint packed_cols = dims.w;
    const uint groups = quant.z;
    const uint lhs_rows = quant.w;
    const uint slot = tile.x;
    const uint row_base = tile.y * 8u + simd_gid * 4u;
    if (slot >= topk || row_base >= out_dim) {
        return;
    }

    const uint expert = expert_indices[slot];
    uint lhs_row = 0u;
    if (lhs_rows == 1u) {
        lhs_row = 0u;
    } else if (lhs_rows == topk) {
        lhs_row = slot;
    } else {
        const uint slots_per_row = (topk > lhs_rows) ? (topk / lhs_rows) : 1u;
        lhs_row = min(slot / slots_per_row, lhs_rows - 1u);
    }
    const uint row_bytes = packed_cols * 4u;
    const uint scale_step_per_thread = 4u;
    const device uchar* gate_ws = ((const device uchar*)gate_packed) +
        ((expert * out_dim + row_base) * row_bytes) + simd_lid * 8u;
    const device uchar* up_ws = ((const device uchar*)up_packed) +
        ((expert * out_dim + row_base) * row_bytes) + simd_lid * 8u;
    const device bfloat* gate_scale_base = gate_scales +
        ((expert * out_dim + row_base) * groups) + simd_lid / scale_step_per_thread;
    const device bfloat* gate_bias_base = gate_biases +
        ((expert * out_dim + row_base) * groups) + simd_lid / scale_step_per_thread;
    const device bfloat* up_scale_base = up_scales +
        ((expert * out_dim + row_base) * groups) + simd_lid / scale_step_per_thread;
    const device bfloat* up_bias_base = up_biases +
        ((expert * out_dim + row_base) * groups) + simd_lid / scale_step_per_thread;
    const device float* x = lhs + lhs_row * in_dim + simd_lid * 16u;

    float gate_result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float up_result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint k = 0u; k < in_dim; k += 512u) {
        const bool lane_active = k + simd_lid * 16u < in_dim;
        float xt[16];
        float sum = 0.0f;
        if (lane_active) {
            for (uint i = 0u; i < 16u; i += 4u) {
                const float x0 = x[i];
                const float x1 = x[i + 1u];
                const float x2 = x[i + 2u];
                const float x3 = x[i + 3u];
                sum += x0 + x1 + x2 + x3;
                xt[i] = x0;
                xt[i + 1u] = x1 / 16.0f;
                xt[i + 2u] = x2 / 256.0f;
                xt[i + 3u] = x3 / 4096.0f;
            }
        }

        for (uint row = 0u; row < 4u; ++row) {
            if (lane_active && row_base + row < out_dim) {
                const device ushort* gate_w16 = (const device ushort*)(gate_ws + row * row_bytes);
                const device ushort* up_w16 = (const device ushort*)(up_ws + row * row_bytes);
                const float gate_scale = gate_scale_base[row * groups];
                const float gate_bias = gate_bias_base[row * groups];
                const float up_scale = up_scale_base[row * groups];
                const float up_bias = up_bias_base[row * groups];
                float gate_accum = 0.0f;
                float up_accum = 0.0f;
                for (uint i = 0u; i < 4u; ++i) {
                    const uint gate_word = uint(gate_w16[i]);
                    const uint up_word = uint(up_w16[i]);
                    gate_accum += xt[4u * i] * float(gate_word & 0x000fu);
                    gate_accum += xt[4u * i + 1u] * float(gate_word & 0x00f0u);
                    gate_accum += xt[4u * i + 2u] * float(gate_word & 0x0f00u);
                    gate_accum += xt[4u * i + 3u] * float(gate_word & 0xf000u);
                    up_accum += xt[4u * i] * float(up_word & 0x000fu);
                    up_accum += xt[4u * i + 1u] * float(up_word & 0x00f0u);
                    up_accum += xt[4u * i + 2u] * float(up_word & 0x0f00u);
                    up_accum += xt[4u * i + 3u] * float(up_word & 0xf000u);
                }
                gate_result[row] += gate_scale * gate_accum + sum * gate_bias;
                up_result[row] += up_scale * up_accum + sum * up_bias;
            }
        }
        gate_ws += 256u;
        up_ws += 256u;
        gate_scale_base += 8u;
        gate_bias_base += 8u;
        up_scale_base += 8u;
        up_bias_base += 8u;
        x += 512u;
    }

    for (uint row = 0u; row < 4u; ++row) {
        const float gate = simd_sum(gate_result[row]);
        const float up = simd_sum(up_result[row]);
        if (simd_lid == 0u && row_base + row < out_dim) {
            out[(slot * out_dim) + row_base + row] = (gate / (1.0f + exp(-gate))) * up;
        }
    }
}

// Variante NON-gather, batch=1 du fusé gate+up+swiglu : fond les deux QMV
// (gate_proj, up_proj) du SHARED-expert et le swiglu en UN dispatch. Calque exact
// de `affine_qmv_fast_u4_gs64_f32` (single-row, 2 simdgroups × 4 lignes) mais avec
// deux poids 4-bit gs64 et la fusion swiglu en sortie. Exige in_dim % 512 == 0.
kernel void affine_gate_up_swiglu_fast_u4_gs64_f32(
    device const float* lhs [[buffer(0)]],
    device const uint* gate_packed [[buffer(1)]],
    device const bfloat* gate_scales [[buffer(2)]],
    device const bfloat* gate_biases [[buffer(3)]],
    device const uint* up_packed [[buffer(4)]],
    device const bfloat* up_scales [[buffer(5)]],
    device const bfloat* up_biases [[buffer(6)]],
    device float* out [[buffer(7)]],
    constant uint4& dims [[buffer(8)]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint2 tile [[threadgroup_position_in_grid]]
) {
    const uint out_dim = dims.x;
    const uint in_dim = dims.y;
    const uint packed_cols = dims.z;
    const uint groups = dims.w;
    const uint results_per_simdgroup = 4u;
    const uint simdgroups = 2u;
    const uint values_per_thread = 16u;
    const uint block_size = values_per_thread * 32u;
    const uint row_base = tile.y * (simdgroups * results_per_simdgroup) +
        simd_gid * results_per_simdgroup;
    if (row_base >= out_dim) {
        return;
    }

    const uint row_bytes = packed_cols * 4u;
    const uint scale_step_per_thread = 4u;
    const device uchar* gate_ws = ((const device uchar*)gate_packed) +
        row_base * row_bytes + simd_lid * 8u;
    const device uchar* up_ws = ((const device uchar*)up_packed) +
        row_base * row_bytes + simd_lid * 8u;
    const device bfloat* gate_scale_base =
        gate_scales + row_base * groups + simd_lid / scale_step_per_thread;
    const device bfloat* gate_bias_base =
        gate_biases + row_base * groups + simd_lid / scale_step_per_thread;
    const device bfloat* up_scale_base =
        up_scales + row_base * groups + simd_lid / scale_step_per_thread;
    const device bfloat* up_bias_base =
        up_biases + row_base * groups + simd_lid / scale_step_per_thread;
    const device float* x = lhs + tile.x * in_dim + simd_lid * values_per_thread;

    float gate_result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float up_result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint k = 0u; k < in_dim; k += block_size) {
        float xt[16];
        float sum = 0.0f;
        for (uint i = 0u; i < values_per_thread; i += 4u) {
            const float x0 = x[i];
            const float x1 = x[i + 1u];
            const float x2 = x[i + 2u];
            const float x3 = x[i + 3u];
            sum += x0 + x1 + x2 + x3;
            xt[i] = x0;
            xt[i + 1u] = x1 / 16.0f;
            xt[i + 2u] = x2 / 256.0f;
            xt[i + 3u] = x3 / 4096.0f;
        }

        for (uint row = 0u; row < results_per_simdgroup; ++row) {
            if (row_base + row < out_dim) {
                const device ushort* gate_w16 = (const device ushort*)(gate_ws + row * row_bytes);
                const device ushort* up_w16 = (const device ushort*)(up_ws + row * row_bytes);
                const float gate_scale = gate_scale_base[row * groups];
                const float gate_bias = gate_bias_base[row * groups];
                const float up_scale = up_scale_base[row * groups];
                const float up_bias = up_bias_base[row * groups];
                float gate_accum = 0.0f;
                float up_accum = 0.0f;
                for (uint i = 0u; i < 4u; ++i) {
                    const ushort gate_word = gate_w16[i];
                    const ushort up_word = up_w16[i];
                    gate_accum += xt[4u * i] * float(gate_word & 0x000fu);
                    gate_accum += xt[4u * i + 1u] * float(gate_word & 0x00f0u);
                    gate_accum += xt[4u * i + 2u] * float(gate_word & 0x0f00u);
                    gate_accum += xt[4u * i + 3u] * float(gate_word & 0xf000u);
                    up_accum += xt[4u * i] * float(up_word & 0x000fu);
                    up_accum += xt[4u * i + 1u] * float(up_word & 0x00f0u);
                    up_accum += xt[4u * i + 2u] * float(up_word & 0x0f00u);
                    up_accum += xt[4u * i + 3u] * float(up_word & 0xf000u);
                }
                gate_result[row] += gate_scale * gate_accum + sum * gate_bias;
                up_result[row] += up_scale * up_accum + sum * up_bias;
            }
        }

        gate_ws += 256u;
        up_ws += 256u;
        gate_scale_base += 8u;
        gate_bias_base += 8u;
        up_scale_base += 8u;
        up_bias_base += 8u;
        x += block_size;
    }

    for (uint row = 0u; row < results_per_simdgroup; ++row) {
        const float gate = simd_sum(gate_result[row]);
        const float up = simd_sum(up_result[row]);
        if (simd_lid == 0u && row_base + row < out_dim) {
            out[tile.x * out_dim + row_base + row] = (gate / (1.0f + exp(-gate))) * up;
        }
    }
}

kernel void affine_qkv_split_qmv_fast_u4_gs64_f32(
    device const float* lhs [[buffer(0)]],
    device const uint* packed [[buffer(1)]],
    device const bfloat* scales [[buffer(2)]],
    device const bfloat* biases [[buffer(3)]],
    device float* out [[buffer(4)]],
    device float* q_out [[buffer(5)]],
    device float* gate_out [[buffer(6)]],
    constant uint4& dims [[buffer(7)]],
    constant uint2& q_dims [[buffer(8)]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint2 tile [[threadgroup_position_in_grid]]
) {
    const uint out_dim = dims.x;
    const uint in_dim = dims.y;
    const uint packed_cols = dims.z;
    const uint groups = dims.w;
    const uint q_heads = q_dims.x;
    const uint head_dim = q_dims.y;
    const uint q_gate_dim = q_heads * head_dim * 2u;
    const uint results_per_simdgroup = 4u;
    const uint simdgroups = 2u;
    const uint values_per_thread = 16u;
    const uint block_size = values_per_thread * 32u;
    const uint row_base = tile.y * (simdgroups * results_per_simdgroup) +
        simd_gid * results_per_simdgroup;
    if (row_base >= out_dim) {
        return;
    }

    const uint row_bytes = packed_cols * 4u;
    const uint scale_step_per_thread = 4u;
    const device uchar* ws = ((const device uchar*)packed) +
        row_base * row_bytes + simd_lid * 8u;
    const device bfloat* scale_base = scales + row_base * groups + simd_lid / scale_step_per_thread;
    const device bfloat* bias_base = biases + row_base * groups + simd_lid / scale_step_per_thread;
    const device float* x = lhs + tile.x * in_dim + simd_lid * values_per_thread;

    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint k = 0u; k < in_dim; k += block_size) {
        float xt[16];
        float sum = 0.0f;
        for (uint i = 0u; i < values_per_thread; i += 4u) {
            const float x0 = x[i];
            const float x1 = x[i + 1u];
            const float x2 = x[i + 2u];
            const float x3 = x[i + 3u];
            sum += x0 + x1 + x2 + x3;
            xt[i] = x0;
            xt[i + 1u] = x1 / 16.0f;
            xt[i + 2u] = x2 / 256.0f;
            xt[i + 3u] = x3 / 4096.0f;
        }

        for (uint row = 0u; row < results_per_simdgroup; ++row) {
            if (row_base + row < out_dim) {
                const device ushort* w16 = (const device ushort*)(ws + row * row_bytes);
                const float scale = scale_base[row * groups];
                const float bias = bias_base[row * groups];
                float accum = 0.0f;
                for (uint i = 0u; i < 4u; ++i) {
                    const ushort word = w16[i];
                    accum += xt[4u * i] * float(word & 0x000fu);
                    accum += xt[4u * i + 1u] * float(word & 0x00f0u);
                    accum += xt[4u * i + 2u] * float(word & 0x0f00u);
                    accum += xt[4u * i + 3u] * float(word & 0xf000u);
                }
                result[row] += scale * accum + sum * bias;
            }
        }
        ws += block_size / 2u;
        scale_base += 32u / scale_step_per_thread;
        bias_base += 32u / scale_step_per_thread;
        x += block_size;
    }

    for (uint row = 0u; row < results_per_simdgroup; ++row) {
        const uint out_row = row_base + row;
        const float reduced = simd_sum(result[row]);
        if (simd_lid == 0u && out_row < out_dim) {
            if (out_row < q_gate_dim) {
                const uint pair = out_row / (2u * head_dim);
                const uint col = out_row - pair * 2u * head_dim;
                if (col < head_dim) {
                    q_out[pair * head_dim + col] = reduced;
                } else {
                    gate_out[pair * head_dim + (col - head_dim)] = reduced;
                }
            } else {
                out[tile.x * out_dim + out_row] = reduced;
            }
        }
    }
}

kernel void affine_qmv_rms_fast_u4_gs64_f32(
    device const float* lhs [[buffer(0)]],
    device const float* rms_weight [[buffer(1)]],
    device const uint* packed [[buffer(2)]],
    device const bfloat* scales [[buffer(3)]],
    device const bfloat* biases [[buffer(4)]],
    device float* out [[buffer(5)]],
    constant uint4& dims [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint2 tile [[threadgroup_position_in_grid]]
) {
    const uint out_dim = dims.x;
    const uint in_dim = dims.y;
    const uint packed_cols = dims.z;
    const uint groups = dims.w;
    const uint results_per_simdgroup = 4u;
    const uint simdgroups = 2u;
    const uint values_per_thread = 16u;
    const uint block_size = values_per_thread * 32u;
    const uint row_base = tile.y * (simdgroups * results_per_simdgroup) +
        simd_gid * results_per_simdgroup;
    if (row_base >= out_dim) {
        return;
    }

    threadgroup float inv_rms_shared;
    float sumsq = 0.0f;
    if (simd_gid == 0u) {
        const device float* norm_x = lhs + tile.x * in_dim + simd_lid * values_per_thread;
        for (uint k = 0u; k < in_dim; k += block_size) {
            for (uint i = 0u; i < values_per_thread; ++i) {
                const float value = norm_x[i];
                sumsq += value * value;
            }
            norm_x += block_size;
        }
        const float reduced_sumsq = simd_sum(sumsq);
        if (simd_lid == 0u) {
            inv_rms_shared = rsqrt((reduced_sumsq / float(in_dim)) + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_rms = inv_rms_shared;

    const uint row_bytes = packed_cols * 4u;
    const uint scale_step_per_thread = 4u;
    const device uchar* ws = ((const device uchar*)packed) +
        row_base * row_bytes + simd_lid * 8u;
    const device bfloat* scale_base = scales + row_base * groups + simd_lid / scale_step_per_thread;
    const device bfloat* bias_base = biases + row_base * groups + simd_lid / scale_step_per_thread;
    const device float* x = lhs + tile.x * in_dim + simd_lid * values_per_thread;
    const device float* gamma = rms_weight + simd_lid * values_per_thread;

    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint k = 0u; k < in_dim; k += block_size) {
        float xt[16];
        float sum = 0.0f;
        for (uint i = 0u; i < values_per_thread; i += 4u) {
            const float x0 = x[i] * inv_rms * gamma[i];
            const float x1 = x[i + 1u] * inv_rms * gamma[i + 1u];
            const float x2 = x[i + 2u] * inv_rms * gamma[i + 2u];
            const float x3 = x[i + 3u] * inv_rms * gamma[i + 3u];
            sum += x0 + x1 + x2 + x3;
            xt[i] = x0;
            xt[i + 1u] = x1 / 16.0f;
            xt[i + 2u] = x2 / 256.0f;
            xt[i + 3u] = x3 / 4096.0f;
        }

        for (uint row = 0u; row < results_per_simdgroup; ++row) {
            if (row_base + row < out_dim) {
                const device ushort* w16 = (const device ushort*)(ws + row * row_bytes);
                const float scale = scale_base[row * groups];
                const float bias = bias_base[row * groups];
                float accum = 0.0f;
                for (uint i = 0u; i < 4u; ++i) {
                    const ushort word = w16[i];
                    accum += xt[4u * i] * float(word & 0x000fu);
                    accum += xt[4u * i + 1u] * float(word & 0x00f0u);
                    accum += xt[4u * i + 2u] * float(word & 0x0f00u);
                    accum += xt[4u * i + 3u] * float(word & 0xf000u);
                }
                result[row] += scale * accum + sum * bias;
            }
        }
        ws += block_size / 2u;
        scale_base += 32u / scale_step_per_thread;
        bias_base += 32u / scale_step_per_thread;
        x += block_size;
        gamma += block_size;
    }

    for (uint row = 0u; row < results_per_simdgroup; ++row) {
        const float reduced = simd_sum(result[row]);
        if (simd_lid == 0u && row_base + row < out_dim) {
            out[tile.x * out_dim + row_base + row] = reduced;
        }
    }
}

kernel void affine_qkv_split_rms_qmv_fast_u4_gs64_f32(
    device const float* lhs [[buffer(0)]],
    device const float* rms_weight [[buffer(1)]],
    device const uint* packed [[buffer(2)]],
    device const bfloat* scales [[buffer(3)]],
    device const bfloat* biases [[buffer(4)]],
    device float* out [[buffer(5)]],
    device float* q_out [[buffer(6)]],
    device float* gate_out [[buffer(7)]],
    constant uint4& dims [[buffer(8)]],
    constant uint2& q_dims [[buffer(9)]],
    constant float& eps [[buffer(10)]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint2 tile [[threadgroup_position_in_grid]]
) {
    const uint out_dim = dims.x;
    const uint in_dim = dims.y;
    const uint packed_cols = dims.z;
    const uint groups = dims.w;
    const uint q_heads = q_dims.x;
    const uint head_dim = q_dims.y;
    const uint q_gate_dim = q_heads * head_dim * 2u;
    const uint results_per_simdgroup = 4u;
    const uint simdgroups = 2u;
    const uint values_per_thread = 16u;
    const uint block_size = values_per_thread * 32u;
    const uint row_base = tile.y * (simdgroups * results_per_simdgroup) +
        simd_gid * results_per_simdgroup;
    if (row_base >= out_dim) {
        return;
    }

    threadgroup float inv_rms_shared;
    float sumsq = 0.0f;
    if (simd_gid == 0u) {
        const device float* norm_x = lhs + tile.x * in_dim + simd_lid * values_per_thread;
        for (uint k = 0u; k < in_dim; k += block_size) {
            for (uint i = 0u; i < values_per_thread; ++i) {
                const float value = norm_x[i];
                sumsq += value * value;
            }
            norm_x += block_size;
        }
        const float reduced_sumsq = simd_sum(sumsq);
        if (simd_lid == 0u) {
            inv_rms_shared = rsqrt((reduced_sumsq / float(in_dim)) + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_rms = inv_rms_shared;

    const uint row_bytes = packed_cols * 4u;
    const uint scale_step_per_thread = 4u;
    const device uchar* ws = ((const device uchar*)packed) +
        row_base * row_bytes + simd_lid * 8u;
    const device bfloat* scale_base = scales + row_base * groups + simd_lid / scale_step_per_thread;
    const device bfloat* bias_base = biases + row_base * groups + simd_lid / scale_step_per_thread;
    const device float* x = lhs + tile.x * in_dim + simd_lid * values_per_thread;
    const device float* gamma = rms_weight + simd_lid * values_per_thread;

    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint k = 0u; k < in_dim; k += block_size) {
        float xt[16];
        float sum = 0.0f;
        for (uint i = 0u; i < values_per_thread; i += 4u) {
            const float x0 = x[i] * inv_rms * gamma[i];
            const float x1 = x[i + 1u] * inv_rms * gamma[i + 1u];
            const float x2 = x[i + 2u] * inv_rms * gamma[i + 2u];
            const float x3 = x[i + 3u] * inv_rms * gamma[i + 3u];
            sum += x0 + x1 + x2 + x3;
            xt[i] = x0;
            xt[i + 1u] = x1 / 16.0f;
            xt[i + 2u] = x2 / 256.0f;
            xt[i + 3u] = x3 / 4096.0f;
        }

        for (uint row = 0u; row < results_per_simdgroup; ++row) {
            if (row_base + row < out_dim) {
                const device ushort* w16 = (const device ushort*)(ws + row * row_bytes);
                const float scale = scale_base[row * groups];
                const float bias = bias_base[row * groups];
                float accum = 0.0f;
                for (uint i = 0u; i < 4u; ++i) {
                    const ushort word = w16[i];
                    accum += xt[4u * i] * float(word & 0x000fu);
                    accum += xt[4u * i + 1u] * float(word & 0x00f0u);
                    accum += xt[4u * i + 2u] * float(word & 0x0f00u);
                    accum += xt[4u * i + 3u] * float(word & 0xf000u);
                }
                result[row] += scale * accum + sum * bias;
            }
        }
        ws += block_size / 2u;
        scale_base += 32u / scale_step_per_thread;
        bias_base += 32u / scale_step_per_thread;
        x += block_size;
        gamma += block_size;
    }

    for (uint row = 0u; row < results_per_simdgroup; ++row) {
        const uint out_row = row_base + row;
        const float reduced = simd_sum(result[row]);
        if (simd_lid == 0u && out_row < out_dim) {
            if (out_row < q_gate_dim) {
                const uint pair = out_row / (2u * head_dim);
                const uint col = out_row - pair * 2u * head_dim;
                if (col < head_dim) {
                    q_out[pair * head_dim + col] = reduced;
                } else {
                    gate_out[pair * head_dim + (col - head_dim)] = reduced;
                }
            } else {
                out[tile.x * out_dim + out_row] = reduced;
            }
        }
    }
}

kernel void affine_qmv_gated_input_fast_u4_gs64_f32(
    device const float* lhs [[buffer(0)]],
    device const float* gate [[buffer(1)]],
    device const uint* packed [[buffer(2)]],
    device const bfloat* scales [[buffer(3)]],
    device const bfloat* biases [[buffer(4)]],
    device float* out [[buffer(5)]],
    constant uint4& dims [[buffer(6)]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint2 tile [[threadgroup_position_in_grid]]
) {
    const uint out_dim = dims.x;
    const uint in_dim = dims.y;
    const uint packed_cols = dims.z;
    const uint groups = dims.w;
    const uint results_per_simdgroup = 4u;
    const uint simdgroups = 2u;
    const uint values_per_thread = 16u;
    const uint block_size = values_per_thread * 32u;
    const uint row_base = tile.y * (simdgroups * results_per_simdgroup) +
        simd_gid * results_per_simdgroup;
    if (row_base >= out_dim) {
        return;
    }

    const uint row_bytes = packed_cols * 4u;
    const uint scale_step_per_thread = 4u;
    const device uchar* ws = ((const device uchar*)packed) +
        row_base * row_bytes + simd_lid * 8u;
    const device bfloat* scale_base = scales + row_base * groups + simd_lid / scale_step_per_thread;
    const device bfloat* bias_base = biases + row_base * groups + simd_lid / scale_step_per_thread;
    const device float* x = lhs + tile.x * in_dim + simd_lid * values_per_thread;
    const device float* g = gate + tile.x * in_dim + simd_lid * values_per_thread;

    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint k = 0u; k < in_dim; k += block_size) {
        float xt[16];
        float sum = 0.0f;
        for (uint i = 0u; i < values_per_thread; i += 4u) {
            const float x0 = x[i] * (1.0f / (1.0f + exp(-g[i])));
            const float x1 = x[i + 1u] * (1.0f / (1.0f + exp(-g[i + 1u])));
            const float x2 = x[i + 2u] * (1.0f / (1.0f + exp(-g[i + 2u])));
            const float x3 = x[i + 3u] * (1.0f / (1.0f + exp(-g[i + 3u])));
            sum += x0 + x1 + x2 + x3;
            xt[i] = x0;
            xt[i + 1u] = x1 / 16.0f;
            xt[i + 2u] = x2 / 256.0f;
            xt[i + 3u] = x3 / 4096.0f;
        }

        for (uint row = 0u; row < results_per_simdgroup; ++row) {
            if (row_base + row < out_dim) {
                const device ushort* w16 = (const device ushort*)(ws + row * row_bytes);
                const float scale = scale_base[row * groups];
                const float bias = bias_base[row * groups];
                float accum = 0.0f;
                for (uint i = 0u; i < 4u; ++i) {
                    const ushort word = w16[i];
                    accum += xt[4u * i] * float(word & 0x000fu);
                    accum += xt[4u * i + 1u] * float(word & 0x00f0u);
                    accum += xt[4u * i + 2u] * float(word & 0x0f00u);
                    accum += xt[4u * i + 3u] * float(word & 0xf000u);
                }
                result[row] += scale * accum + sum * bias;
            }
        }
        ws += block_size / 2u;
        scale_base += 32u / scale_step_per_thread;
        bias_base += 32u / scale_step_per_thread;
        x += block_size;
        g += block_size;
    }

    for (uint row = 0u; row < results_per_simdgroup; ++row) {
        const float reduced = simd_sum(result[row]);
        if (simd_lid == 0u && row_base + row < out_dim) {
            out[tile.x * out_dim + row_base + row] = reduced;
        }
    }
}

kernel void affine_argmax_qmv_fast_u4_gs64_f32(
    device const float* lhs [[buffer(0)]],
    device const uint* packed [[buffer(1)]],
    device const bfloat* scales [[buffer(2)]],
    device const bfloat* biases [[buffer(3)]],
    device float* partial_values [[buffer(4)]],
    device uint* partial_indices [[buffer(5)]],
    constant uint4& dims [[buffer(6)]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint2 tile [[threadgroup_position_in_grid]]
) {
    const uint out_dim = dims.x;
    const uint in_dim = dims.y;
    const uint packed_cols = dims.z;
    const uint groups = dims.w;
    const uint row_base = tile.y * 8u + simd_gid * 4u;
    if (row_base >= out_dim) {
        return;
    }

    const uint row_bytes = packed_cols * 4u;
    const uint scale_step_per_thread = 4u;
    const device uchar* ws = ((const device uchar*)packed) + row_base * row_bytes + simd_lid * 8u;
    const device bfloat* scale_base = scales + row_base * groups + simd_lid / scale_step_per_thread;
    const device bfloat* bias_base = biases + row_base * groups + simd_lid / scale_step_per_thread;
    const device float* x = lhs + tile.x * in_dim + simd_lid * 16u;

    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint k = 0u; k < in_dim; k += 512u) {
        float xt[16];
        float sum = 0.0f;
        for (uint i = 0u; i < 16u; i += 4u) {
            const float x0 = x[i];
            const float x1 = x[i + 1u];
            const float x2 = x[i + 2u];
            const float x3 = x[i + 3u];
            sum += x0 + x1 + x2 + x3;
            xt[i] = x0;
            xt[i + 1u] = x1 / 16.0f;
            xt[i + 2u] = x2 / 256.0f;
            xt[i + 3u] = x3 / 4096.0f;
        }

        for (uint row = 0u; row < 4u; ++row) {
            if (row_base + row < out_dim) {
                const device ushort* w16 = (const device ushort*)(ws + row * row_bytes);
                const float scale = scale_base[row * groups];
                const float bias = bias_base[row * groups];
                float accum = 0.0f;
                for (uint i = 0u; i < 4u; ++i) {
                    const uint word = uint(w16[i]);
                    accum += xt[4u * i] * float(word & 0x000fu);
                    accum += xt[4u * i + 1u] * float(word & 0x00f0u);
                    accum += xt[4u * i + 2u] * float(word & 0x0f00u);
                    accum += xt[4u * i + 3u] * float(word & 0xf000u);
                }
                result[row] += scale * accum + sum * bias;
            }
        }

        ws += 256u;
        scale_base += 8u;
        bias_base += 8u;
        x += 512u;
    }

    threadgroup float group_values[8];
    threadgroup uint group_indices[8];
    for (uint row = 0u; row < 4u; ++row) {
        const uint index = simd_gid * 4u + row;
        const uint out_row = row_base + row;
        const float reduced = simd_sum(result[row]);
        if (simd_lid == 0u) {
            group_values[index] = (out_row < out_dim) ? reduced : -3.402823466e38f;
            group_indices[index] = out_row;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_gid == 0u && simd_lid == 0u) {
        float best_value = group_values[0];
        uint best_index = group_indices[0];
        for (uint idx = 1u; idx < 8u; ++idx) {
            const float value = group_values[idx];
            const uint index = group_indices[idx];
            if (value > best_value || (value == best_value && index < best_index)) {
                best_value = value;
                best_index = index;
            }
        }
        partial_values[tile.y] = best_value;
        partial_indices[tile.y] = best_index;
    }
}

kernel void weighted_sum_topk_f32(
    device const float* src [[buffer(0)]],
    device const float* scores [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint2& dims [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint topk = dims.x;
    const uint out_dim = dims.y;
    if (gid >= out_dim) {
        return;
    }
    float acc = 0.0f;
    for (uint slot = 0; slot < topk; ++slot) {
        acc += src[(slot * out_dim) + gid] * scores[slot];
    }
    out[gid] = acc;
}

kernel void weighted_sum_grouped_topk_f32(
    device const float* src [[buffer(0)]],
    device const float* scores [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint4& dims [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint rows = dims.x;
    const uint topk_per_row = dims.y;
    const uint out_dim = dims.z;
    const uint total = rows * out_dim;
    if (gid >= total) {
        return;
    }
    const uint row = gid / out_dim;
    const uint col = gid - row * out_dim;
    const uint slot_base = row * topk_per_row;
    float acc = 0.0f;
    for (uint slot = 0; slot < topk_per_row; ++slot) {
        const uint source_slot = slot_base + slot;
        acc += src[(source_slot * out_dim) + col] * scores[source_slot];
    }
    out[gid] = acc;
}

kernel void weighted_sum_add_grouped_topk_f32(
    device const float* src [[buffer(0)]],
    device const float* scores [[buffer(1)]],
    device const float* residual [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint4& dims [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint rows = dims.x;
    const uint topk_per_row = dims.y;
    const uint out_dim = dims.z;
    const uint total = rows * out_dim;
    if (gid >= total) {
        return;
    }
    const uint row = gid / out_dim;
    const uint col = gid - row * out_dim;
    const uint slot_base = row * topk_per_row;
    float acc = residual[gid];
    for (uint slot = 0; slot < topk_per_row; ++slot) {
        const uint source_slot = slot_base + slot;
        acc += src[(source_slot * out_dim) + col] * scores[source_slot];
    }
    out[gid] = acc;
}

kernel void weighted_sum_add_topk_f32(
    device const float* src [[buffer(0)]],
    device const float* scores [[buffer(1)]],
    device const float* residual [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint2& dims [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint topk = dims.x;
    const uint out_dim = dims.y;
    if (gid >= out_dim) {
        return;
    }
    float acc = residual[gid];
    for (uint slot = 0; slot < topk; ++slot) {
        acc += src[(slot * out_dim) + gid] * scores[slot];
    }
    out[gid] = acc;
}

kernel void add_sigmoid_scaled_f32(
    device const float* src [[buffer(0)]],
    device const float* gate [[buffer(1)]],
    device float* dst [[buffer(2)]],
    constant uint& len [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) {
        return;
    }
    const float scale = 1.0f / (1.0f + exp(-gate[0]));
    dst[gid] += src[gid] * scale;
}

kernel void copy_f32(
    device const float* src [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& len [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) {
        return;
    }
    out[gid] = src[gid];
}

kernel void rms_norm_rows_f32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& dim [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]
) {
    threadgroup float partial[256];
    float sumsq = 0.0f;
    const uint offset = row * dim;
    for (uint col = tid; col < dim; col += 256u) {
        const float value = input[offset + col];
        sumsq += value * value;
    }
    partial[tid] = sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv_rms = rsqrt((partial[0] / float(dim)) + eps);
    for (uint col = tid; col < dim; col += 256u) {
        out[offset + col] = input[offset + col] * inv_rms * weight[col];
    }
}

kernel void add_rms_norm_rows_f32(
    device const float* left [[buffer(0)]],
    device const float* right [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device float* summed [[buffer(3)]],
    device float* normed [[buffer(4)]],
    constant uint& dim [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]
) {
    threadgroup float partial[256];
    float sumsq = 0.0f;
    const uint offset = row * dim;
    for (uint col = tid; col < dim; col += 256u) {
        const float value = left[offset + col] + right[offset + col];
        summed[offset + col] = value;
        sumsq += value * value;
    }
    partial[tid] = sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv_rms = rsqrt((partial[0] / float(dim)) + eps);
    for (uint col = tid; col < dim; col += 256u) {
        normed[offset + col] = summed[offset + col] * inv_rms * weight[col];
    }
}

kernel void rms_norm_rope_heads_f32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint4& dims [[buffer(3)]],
    constant float2& params [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 tile [[threadgroup_position_in_grid]]
) {
    const uint seq = dims.x;
    const uint heads = dims.y;
    const uint head_dim = dims.z;
    const uint rope_dims = dims.w;
    const float eps = params.x;
    const float base_theta = params.y;
    const uint head = tile.x;
    const uint pos = tile.y;
    if (head >= heads || pos >= seq) {
        return;
    }

    threadgroup float partial[256];
    const uint dim = heads * head_dim;
    const uint start = pos * dim + head * head_dim;
    float sumsq = 0.0f;
    for (uint col = tid; col < head_dim; col += 256u) {
        const float value = input[start + col];
        sumsq += value * value;
    }
    partial[tid] = sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv_rms = rsqrt((partial[0] / float(head_dim)) + eps);
    for (uint col = tid; col < head_dim; col += 256u) {
        float value = input[start + col] * inv_rms * weight[col];
        if (col < rope_dims) {
            const uint pair = col / 2u;
            const float exponent = float(2u * pair) / float(rope_dims);
            const float angle = float(pos) / pow(base_theta, exponent);
            const float c = cos(angle);
            const float s = sin(angle);
            const uint even_col = pair * 2u;
            const uint odd_col = even_col + 1u;
            const float even = input[start + even_col] * inv_rms * weight[even_col];
            const float odd = input[start + odd_col] * inv_rms * weight[odd_col];
            value = (col == even_col) ? (even * c - odd * s) : (even * s + odd * c);
        }
        out[start + col] = value;
    }
}

kernel void causal_attention_prefill_f32(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint4& dims [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 tile [[threadgroup_position_in_grid]]
) {
    const uint seq = dims.x;
    const uint q_heads = dims.y;
    const uint kv_heads = dims.z;
    const uint head_dim = dims.w;
    const uint q_head = tile.x;
    const uint pos = tile.y;
    if (q_head >= q_heads || pos >= seq || seq > 256u) {
        return;
    }

    threadgroup float partial[256];
    threadgroup float scores[256];
    const uint kv_group = q_heads / kv_heads;
    const uint kv_head = q_head / kv_group;
    const uint q_dim = q_heads * head_dim;
    const uint kv_dim = kv_heads * head_dim;
    const uint q_start = pos * q_dim + q_head * head_dim;
    const uint kv_head_start = kv_head * head_dim;
    const float scale = rsqrt(float(head_dim));

    for (uint row = 0u; row <= pos; ++row) {
        float dot = 0.0f;
        const uint k_start = row * kv_dim + kv_head_start;
        for (uint col = tid; col < head_dim; col += 256u) {
            dot += q[q_start + col] * k[k_start + col];
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
            scores[row] = partial[0] * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        float max_score = scores[0];
        for (uint row = 1u; row <= pos; ++row) {
            max_score = max(max_score, scores[row]);
        }
        float sum = 0.0f;
        for (uint row = 0u; row <= pos; ++row) {
            const float value = exp(scores[row] - max_score);
            scores[row] = value;
            sum += value;
        }
        const float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
        for (uint row = 0u; row <= pos; ++row) {
            scores[row] *= inv_sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint out_start = pos * q_dim + q_head * head_dim;
    for (uint col = tid; col < head_dim; col += 256u) {
        float acc = 0.0f;
        for (uint row = 0u; row <= pos; ++row) {
            const uint v_start = row * kv_dim + kv_head_start;
            acc += scores[row] * v[v_start + col];
        }
        out[out_start + col] = acc;
    }
}

kernel void rms_norm_row_f32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& dim [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float partial[256];
    float sum = 0.0f;
    for (uint col = tid; col < dim; col += 256u) {
        const float value = input[col];
        sum += value * value;
    }
    partial[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv_rms = rsqrt((partial[0] / float(dim)) + eps);
    for (uint col = tid; col < dim; col += 256u) {
        out[col] = input[col] * inv_rms * weight[col];
    }
}

kernel void add_rms_norm_row_f32(
    device const float* left [[buffer(0)]],
    device const float* right [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device float* summed [[buffer(3)]],
    device float* normed [[buffer(4)]],
    constant uint& dim [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float partial[256];
    float sumsq = 0.0f;
    for (uint col = tid; col < dim; col += 256u) {
        const float value = left[col] + right[col];
        summed[col] = value;
        sumsq += value * value;
    }
    partial[tid] = sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv_rms = rsqrt((partial[0] / float(dim)) + eps);
    for (uint col = tid; col < dim; col += 256u) {
        normed[col] = summed[col] * inv_rms * weight[col];
    }
}

static inline bool topk_better(float value, uint index, float best_value, uint best_index) {
    return (value > best_value) || ((value == best_value) && (index < best_index));
}

kernel void topk_softmax_serial_f32(
    device const float* logits [[buffer(0)]],
    device uint* indices [[buffer(1)]],
    device float* scores [[buffer(2)]],
    constant uint2& dims [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint count = dims.x;
    const uint topk = dims.y;
    if (gid != 0u) {
        return;
    }

    float best_values[16];
    uint best_indices[16];
    for (uint slot = 0u; slot < topk; ++slot) {
        best_values[slot] = -3.402823466e38f;
        best_indices[slot] = 0u;
    }

    for (uint idx = 0u; idx < count; ++idx) {
        const float value = logits[idx];
        for (uint slot = 0u; slot < topk; ++slot) {
            if (value > best_values[slot]) {
                for (uint shift = topk - 1u; shift > slot; --shift) {
                    best_values[shift] = best_values[shift - 1u];
                    best_indices[shift] = best_indices[shift - 1u];
                }
                best_values[slot] = value;
                best_indices[slot] = idx;
                break;
            }
        }
    }

    const float max_value = best_values[0];
    float denom = 0.0f;
    for (uint slot = 0u; slot < topk; ++slot) {
        denom += exp(best_values[slot] - max_value);
    }
    denom = max(denom, 1.0e-20f);
    for (uint slot = 0u; slot < topk; ++slot) {
        indices[slot] = best_indices[slot];
        scores[slot] = exp(best_values[slot] - max_value) / denom;
    }
}

kernel void topk_softmax_f32(
    device const float* logits [[buffer(0)]],
    device uint* indices [[buffer(1)]],
    device float* scores [[buffer(2)]],
    constant uint2& dims [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]]
) {
    const uint count = dims.x;
    const uint topk = dims.y;
    const uint width = 32u;
    const uint max_topk = 16u;
    threadgroup float partial_values[32u * 16u];
    threadgroup uint partial_indices[32u * 16u];

    float best_values[16];
    uint best_indices[16];
    for (uint slot = 0u; slot < max_topk; ++slot) {
        best_values[slot] = -3.402823466e38f;
        best_indices[slot] = 0xffffffffu;
    }

    for (uint idx = tid; idx < count; idx += width) {
        const float value = logits[idx];
        for (uint slot = 0u; slot < topk; ++slot) {
            if (topk_better(value, idx, best_values[slot], best_indices[slot])) {
                for (uint shift = topk - 1u; shift > slot; --shift) {
                    best_values[shift] = best_values[shift - 1u];
                    best_indices[shift] = best_indices[shift - 1u];
                }
                best_values[slot] = value;
                best_indices[slot] = idx;
                break;
            }
        }
    }

    const uint local_base = tid * max_topk;
    for (uint slot = 0u; slot < max_topk; ++slot) {
        partial_values[local_base + slot] = best_values[slot];
        partial_indices[local_base + slot] = best_indices[slot];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = width >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            const uint left_base = tid * max_topk;
            const uint right_base = (tid + stride) * max_topk;
            float merged_values[16];
            uint merged_indices[16];
            for (uint slot = 0u; slot < max_topk; ++slot) {
                merged_values[slot] = partial_values[left_base + slot];
                merged_indices[slot] = partial_indices[left_base + slot];
            }
            for (uint src = 0u; src < topk; ++src) {
                const float value = partial_values[right_base + src];
                const uint index = partial_indices[right_base + src];
                for (uint slot = 0u; slot < topk; ++slot) {
                    if (topk_better(value, index, merged_values[slot], merged_indices[slot])) {
                        for (uint shift = topk - 1u; shift > slot; --shift) {
                            merged_values[shift] = merged_values[shift - 1u];
                            merged_indices[shift] = merged_indices[shift - 1u];
                        }
                        merged_values[slot] = value;
                        merged_indices[slot] = index;
                        break;
                    }
                }
            }
            for (uint slot = 0u; slot < topk; ++slot) {
                partial_values[left_base + slot] = merged_values[slot];
                partial_indices[left_base + slot] = merged_indices[slot];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        const float max_value = partial_values[0];
        float denom = 0.0f;
        for (uint slot = 0u; slot < topk; ++slot) {
            denom += exp(partial_values[slot] - max_value);
        }
        denom = max(denom, 1.0e-20f);
        for (uint slot = 0u; slot < topk; ++slot) {
            indices[slot] = partial_indices[slot];
            scores[slot] = exp(partial_values[slot] - max_value) / denom;
        }
    }
}

kernel void topk_softmax_rows_f32(
    device const float* logits [[buffer(0)]],
    device uint* indices [[buffer(1)]],
    device float* scores [[buffer(2)]],
    constant uint3& dims [[buffer(3)]],
    uint row [[thread_position_in_grid]]
) {
    const uint rows = dims.x;
    const uint count = dims.y;
    const uint topk = dims.z;
    if (row >= rows) {
        return;
    }
    const uint row_logits = row * count;
    const uint row_topk = row * topk;

    float best_values[16];
    uint best_indices[16];
    for (uint slot = 0u; slot < topk; ++slot) {
        best_values[slot] = -3.402823466e38f;
        best_indices[slot] = 0u;
    }

    for (uint idx = 0u; idx < count; ++idx) {
        const float value = logits[row_logits + idx];
        for (uint slot = 0u; slot < topk; ++slot) {
            if (value > best_values[slot]) {
                for (uint shift = topk - 1u; shift > slot; --shift) {
                    best_values[shift] = best_values[shift - 1u];
                    best_indices[shift] = best_indices[shift - 1u];
                }
                best_values[slot] = value;
                best_indices[slot] = idx;
                break;
            }
        }
    }

    const float max_value = best_values[0];
    float denom = 0.0f;
    for (uint slot = 0u; slot < topk; ++slot) {
        denom += exp(best_values[slot] - max_value);
    }
    denom = max(denom, 1.0e-20f);
    for (uint slot = 0u; slot < topk; ++slot) {
        indices[row_topk + slot] = best_indices[slot];
        scores[row_topk + slot] = exp(best_values[slot] - max_value) / denom;
    }
}

static inline float splitmix_unit_f32(ulong state) {
    state += 0x9E3779B97F4A7C15ul;
    ulong z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ul;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBul;
    z = z ^ (z >> 31);
    const uint mantissa = uint(z >> 40) & 0x00ffffffu;
    return float(mantissa) / 16777216.0f;
}

kernel void sample_topk_blocks_f32(
    device const float* logits [[buffer(0)]],
    device float* partial_values [[buffer(1)]],
    device uint* partial_indices [[buffer(2)]],
    constant uint2& dims [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]
) {
    const uint count = dims.x;
    const uint topk = dims.y;
    threadgroup float values[256];
    threadgroup uint indices[256];

    const uint idx = group * 256u + tid;
    values[tid] = (idx < count) ? logits[idx] : -3.402823466e38f;
    indices[tid] = idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        const uint out_base = group * topk;
        for (uint slot = 0u; slot < topk; ++slot) {
            float best_value = -3.402823466e38f;
            uint best_index = 0xffffffffu;
            uint best_lane = 0u;
            for (uint lane = 0u; lane < 256u; ++lane) {
                const float value = values[lane];
                const uint index = indices[lane];
                if (topk_better(value, index, best_value, best_index)) {
                    best_value = value;
                    best_index = index;
                    best_lane = lane;
                }
            }
            partial_values[out_base + slot] = best_value;
            partial_indices[out_base + slot] = best_index;
            values[best_lane] = -3.402823466e38f;
        }
    }
}

kernel void sample_topk_finalize_f32(
    device const float* partial_values [[buffer(0)]],
    device const uint* partial_indices [[buffer(1)]],
    device uint* out_index [[buffer(2)]],
    constant uint2& dims [[buffer(3)]],
    constant float2& params [[buffer(4)]],
    constant ulong& rng_state [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]]
) {
    const uint count = dims.x;
    const uint topk = dims.y;
    const float temperature = max(params.x, 0.0001f);
    const float top_p = params.y;
    const uint width = 32u;
    const uint max_topk = 32u;
    threadgroup float merged_partial_values[32u * 32u];
    threadgroup uint merged_partial_indices[32u * 32u];

    float best_values[32];
    uint best_indices[32];
    for (uint slot = 0u; slot < max_topk; ++slot) {
        best_values[slot] = -3.402823466e38f;
        best_indices[slot] = 0xffffffffu;
    }

    for (uint idx = tid; idx < count; idx += width) {
        const float value = partial_values[idx];
        const uint index = partial_indices[idx];
        for (uint slot = 0u; slot < topk; ++slot) {
            if (topk_better(value, index, best_values[slot], best_indices[slot])) {
                for (uint shift = topk - 1u; shift > slot; --shift) {
                    best_values[shift] = best_values[shift - 1u];
                    best_indices[shift] = best_indices[shift - 1u];
                }
                best_values[slot] = value;
                best_indices[slot] = index;
                break;
            }
        }
    }

    const uint local_base = tid * max_topk;
    for (uint slot = 0u; slot < max_topk; ++slot) {
        merged_partial_values[local_base + slot] = best_values[slot];
        merged_partial_indices[local_base + slot] = best_indices[slot];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = width >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            const uint left_base = tid * max_topk;
            const uint right_base = (tid + stride) * max_topk;
            float merged_values[32];
            uint merged_indices[32];
            for (uint slot = 0u; slot < max_topk; ++slot) {
                merged_values[slot] = merged_partial_values[left_base + slot];
                merged_indices[slot] = merged_partial_indices[left_base + slot];
            }
            for (uint src = 0u; src < topk; ++src) {
                const float value = merged_partial_values[right_base + src];
                const uint index = merged_partial_indices[right_base + src];
                for (uint slot = 0u; slot < topk; ++slot) {
                    if (topk_better(value, index, merged_values[slot], merged_indices[slot])) {
                        for (uint shift = topk - 1u; shift > slot; --shift) {
                            merged_values[shift] = merged_values[shift - 1u];
                            merged_indices[shift] = merged_indices[shift - 1u];
                        }
                        merged_values[slot] = value;
                        merged_indices[slot] = index;
                        break;
                    }
                }
            }
            for (uint slot = 0u; slot < topk; ++slot) {
                merged_partial_values[left_base + slot] = merged_values[slot];
                merged_partial_indices[left_base + slot] = merged_indices[slot];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        const float max_value = merged_partial_values[0];
        float probs[32];
        float denom = 0.0f;
        for (uint slot = 0u; slot < topk; ++slot) {
            const float value = exp((merged_partial_values[slot] - max_value) / temperature);
            probs[slot] = value;
            denom += value;
        }
        denom = max(denom, 1.0e-20f);

        float kept_sum = 0.0f;
        float cumulative = 0.0f;
        uint kept = 0u;
        for (uint slot = 0u; slot < topk; ++slot) {
            kept = slot + 1u;
            kept_sum += probs[slot];
            cumulative += probs[slot] / denom;
            if (slot > 0u && top_p < 1.0f && cumulative >= top_p) {
                break;
            }
        }

        const float roll = splitmix_unit_f32(rng_state) * max(kept_sum, 1.0e-20f);
        float acc = 0.0f;
        uint chosen = merged_partial_indices[0];
        for (uint slot = 0u; slot < kept; ++slot) {
            acc += probs[slot];
            if (roll <= acc) {
                chosen = merged_partial_indices[slot];
                break;
            }
        }
        out_index[0] = chosen;
    }
}

kernel void sample_topk_topp_f32(
    device const float* logits [[buffer(0)]],
    device uint* out_index [[buffer(1)]],
    constant uint2& dims [[buffer(2)]],
    constant float2& params [[buffer(3)]],
    constant ulong& rng_state [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]]
) {
    const uint count = dims.x;
    const uint topk = dims.y;
    const float temperature = max(params.x, 0.0001f);
    const float top_p = params.y;
    const uint width = 32u;
    const uint max_topk = 32u;
    threadgroup float partial_values[32u * 32u];
    threadgroup uint partial_indices[32u * 32u];

    float best_values[32];
    uint best_indices[32];
    for (uint slot = 0u; slot < max_topk; ++slot) {
        best_values[slot] = -3.402823466e38f;
        best_indices[slot] = 0xffffffffu;
    }

    for (uint idx = tid; idx < count; idx += width) {
        const float value = logits[idx];
        for (uint slot = 0u; slot < topk; ++slot) {
            if (topk_better(value, idx, best_values[slot], best_indices[slot])) {
                for (uint shift = topk - 1u; shift > slot; --shift) {
                    best_values[shift] = best_values[shift - 1u];
                    best_indices[shift] = best_indices[shift - 1u];
                }
                best_values[slot] = value;
                best_indices[slot] = idx;
                break;
            }
        }
    }

    const uint local_base = tid * max_topk;
    for (uint slot = 0u; slot < max_topk; ++slot) {
        partial_values[local_base + slot] = best_values[slot];
        partial_indices[local_base + slot] = best_indices[slot];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = width >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            const uint left_base = tid * max_topk;
            const uint right_base = (tid + stride) * max_topk;
            float merged_values[32];
            uint merged_indices[32];
            for (uint slot = 0u; slot < max_topk; ++slot) {
                merged_values[slot] = partial_values[left_base + slot];
                merged_indices[slot] = partial_indices[left_base + slot];
            }
            for (uint src = 0u; src < topk; ++src) {
                const float value = partial_values[right_base + src];
                const uint index = partial_indices[right_base + src];
                for (uint slot = 0u; slot < topk; ++slot) {
                    if (topk_better(value, index, merged_values[slot], merged_indices[slot])) {
                        for (uint shift = topk - 1u; shift > slot; --shift) {
                            merged_values[shift] = merged_values[shift - 1u];
                            merged_indices[shift] = merged_indices[shift - 1u];
                        }
                        merged_values[slot] = value;
                        merged_indices[slot] = index;
                        break;
                    }
                }
            }
            for (uint slot = 0u; slot < topk; ++slot) {
                partial_values[left_base + slot] = merged_values[slot];
                partial_indices[left_base + slot] = merged_indices[slot];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        const float max_value = partial_values[0];
        float probs[32];
        float denom = 0.0f;
        for (uint slot = 0u; slot < topk; ++slot) {
            const float value = exp((partial_values[slot] - max_value) / temperature);
            probs[slot] = value;
            denom += value;
        }
        denom = max(denom, 1.0e-20f);

        float kept_sum = 0.0f;
        float cumulative = 0.0f;
        uint kept = 0u;
        for (uint slot = 0u; slot < topk; ++slot) {
            kept = slot + 1u;
            kept_sum += probs[slot];
            cumulative += probs[slot] / denom;
            if (slot > 0u && top_p < 1.0f && cumulative >= top_p) {
                break;
            }
        }

        const float roll = splitmix_unit_f32(rng_state) * max(kept_sum, 1.0e-20f);
        float acc = 0.0f;
        uint chosen = partial_indices[0];
        for (uint slot = 0u; slot < kept; ++slot) {
            acc += probs[slot];
            if (roll <= acc) {
                chosen = partial_indices[slot];
                break;
            }
        }
        out_index[0] = chosen;
    }
}

kernel void argmax_blocks_f32(
    device const float* logits [[buffer(0)]],
    device float* partial_values [[buffer(1)]],
    device uint* partial_indices [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]
) {
    threadgroup float values[256];
    threadgroup uint indices[256];

    const uint idx = group * 256u + tid;
    float value = -3.402823466e38f;
    uint best_idx = 0u;
    if (idx < count) {
        value = logits[idx];
        best_idx = idx;
    }
    values[tid] = value;
    indices[tid] = best_idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            const float other_value = values[tid + stride];
            const uint other_index = indices[tid + stride];
            if (other_value > values[tid] ||
                (other_value == values[tid] && other_index < indices[tid])) {
                values[tid] = other_value;
                indices[tid] = other_index;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        partial_values[group] = values[0];
        partial_indices[group] = indices[0];
    }
}

kernel void argmax_finalize_f32(
    device const float* partial_values [[buffer(0)]],
    device const uint* partial_indices [[buffer(1)]],
    device uint* out_index [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float values[256];
    threadgroup uint indices[256];

    float best_value = -3.402823466e38f;
    uint best_index = 0u;
    for (uint idx = tid; idx < count; idx += 256u) {
        const float value = partial_values[idx];
        const uint index = partial_indices[idx];
        if (value > best_value || (value == best_value && index < best_index)) {
            best_value = value;
            best_index = index;
        }
    }
    values[tid] = best_value;
    indices[tid] = best_index;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            const float other_value = values[tid + stride];
            const uint other_index = indices[tid + stride];
            if (other_value > values[tid] ||
                (other_value == values[tid] && other_index < indices[tid])) {
                values[tid] = other_value;
                indices[tid] = other_index;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        out_index[0] = indices[0];
    }
}
