
#include <metal_stdlib>
using namespace metal;

static inline float dot4_u8_affine(uint packed_word, float4 x, float scale, float bias) {
    const float4 lanes = float4(
        float(packed_word & 0x000000ffu),
        float((packed_word >> 8u) & 0x000000ffu),
        float((packed_word >> 16u) & 0x000000ffu),
        float((packed_word >> 24u) & 0x000000ffu)
    );
    return dot(x, lanes * scale + float4(bias));
}

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

kernel void dense_qmv_rhs_bf16_f32(
    device const float* lhs [[buffer(0)]],
    device const bfloat* rhs [[buffer(1)]],
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
    for (uint k = lane; k < in_dim; k += 32u) {
        acc += lhs[(b * in_dim) + k] * float(rhs[(o * in_dim) + k]);
    }
    acc = simd_sum(acc);
    if (lane == 0u) {
        out[(b * out_dim) + o] = acc;
    }
}

// QMV dense f32 batch>=1, out_dim multiple de 8, in_dim multiple de 512.
// Reutilise chaque bloc d'activation pour 8 lignes de poids (2 simdgroups x 4
// lignes), utile pour les routeurs MoE denses [256, 2048]. L'ordre
// d'accumulation par ligne/lane reste celui du kernel dense scalaire:
// lane, lane+32, lane+64, ...
kernel void dense_qmv_fast_f32(
    device const float* lhs [[buffer(0)]],
    device const float* rhs [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint3& dims [[buffer(3)]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint2 tile [[threadgroup_position_in_grid]]
) {
    const uint batch = dims.x;
    const uint out_dim = dims.y;
    const uint in_dim = dims.z;
    const uint b = tile.x;
    const uint row_base = tile.y * 8u + simd_gid * 4u;
    if (b >= batch || row_base >= out_dim) {
        return;
    }

    const uint values_per_thread = 16u;
    const uint values_per_block = values_per_thread * 32u;
    const device float* x = lhs + b * in_dim + simd_lid;
    const device float* row0 = rhs + row_base * in_dim + simd_lid;

    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint k = 0u; k < in_dim; k += values_per_block) {
        float xt[16];
        for (uint i = 0u; i < values_per_thread; ++i) {
            xt[i] = x[k + i * 32u];
        }
        for (uint row = 0u; row < 4u; ++row) {
            if (row_base + row < out_dim) {
                const device float* w = row0 + row * in_dim;
                float accum = 0.0f;
                for (uint i = 0u; i < values_per_thread; ++i) {
                    accum += xt[i] * w[k + i * 32u];
                }
                result[row] += accum;
            }
        }
    }

    for (uint row = 0u; row < 4u; ++row) {
        const float reduced = simd_sum(result[row]);
        if (simd_lid == 0u && row_base + row < out_dim) {
            out[b * out_dim + row_base + row] = reduced;
        }
    }
}

// GEMM dense tuilé `out[batch,out_dim] = lhs[batch,in_dim] · rhs[out_dim,in_dim]^T`.
// Le kernel qmv (`dense_matmul_rhs_t_f32`) re-lit la ligne de poids pour CHAQUE
// ligne de batch → memory-bound pour batch≫1 (encodeur Whisper : ~7,5 To de
// trafic). Ici une tuile threadgroup 64×64 est calculée par 16×16=256 threads,
// chaque thread tenant un micro-bloc 4×4 en registres : lhs/rhs chargés en
// mémoire threadgroup par bandes de TK=16 et réutilisés 64× (registres) → trafic
// global ÷~64 et réutilisation registre. Accumulation in_dim en ordre croissant
// (t→kk), comme le `dot` CPU séquentiel.
kernel void dense_gemm_rhs_t_f32(
    device const float* lhs [[buffer(0)]],
    device const float* rhs [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint3& dims [[buffer(3)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[threadgroup_position_in_grid]]
) {
    const uint batch = dims.x;
    const uint out_dim = dims.y;
    const uint in_dim = dims.z;
    constexpr uint TM = 64u; // lignes batch par tuile
    constexpr uint TN = 64u; // colonnes out par tuile
    constexpr uint TK = 16u; // profondeur in_dim par bande
    constexpr uint TT = 4u;  // micro-bloc 4×4 par thread

    threadgroup float lhs_tile[TM][TK];
    threadgroup float rhs_tile[TN][TK];

    const uint row0 = gid.y * TM;
    const uint col0 = gid.x * TN;
    const uint tindex = tid.y * 16u + tid.x;
    const uint tiles = (in_dim + TK - 1u) / TK;

    float acc[TT][TT];
    for (uint i = 0u; i < TT; ++i) {
        for (uint j = 0u; j < TT; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    for (uint t = 0u; t < tiles; ++t) {
        const uint k0 = t * TK;
        for (uint idx = tindex; idx < TM * TK; idx += 256u) {
            const uint r = idx / TK;
            const uint c = idx - r * TK;
            const uint gr = row0 + r;
            const uint gc = k0 + c;
            lhs_tile[r][c] = (gr < batch && gc < in_dim) ? lhs[gr * in_dim + gc] : 0.0f;
        }
        for (uint idx = tindex; idx < TN * TK; idx += 256u) {
            const uint r = idx / TK;
            const uint c = idx - r * TK;
            const uint gr = col0 + r;
            const uint gc = k0 + c;
            rhs_tile[r][c] = (gr < out_dim && gc < in_dim) ? rhs[gr * in_dim + gc] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0u; kk < TK; ++kk) {
            float a[TT];
            float b[TT];
            for (uint i = 0u; i < TT; ++i) {
                a[i] = lhs_tile[tid.y * TT + i][kk];
            }
            for (uint j = 0u; j < TT; ++j) {
                b[j] = rhs_tile[tid.x * TT + j][kk];
            }
            for (uint i = 0u; i < TT; ++i) {
                for (uint j = 0u; j < TT; ++j) {
                    acc[i][j] += a[i] * b[j];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = 0u; i < TT; ++i) {
        const uint gr = row0 + tid.y * TT + i;
        if (gr >= batch) {
            continue;
        }
        for (uint j = 0u; j < TT; ++j) {
            const uint gc = col0 + tid.x * TT + j;
            if (gc < out_dim) {
                out[gr * out_dim + gc] = acc[i][j];
            }
        }
    }
}

// Conversion f32 → bf16 (élément par élément), pour alimenter le GEMM NA en bf16.
kernel void f32_to_bf16(
    device const float* in [[buffer(0)]],
    device bfloat* out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) {
        return;
    }
    out[gid] = bfloat(in[gid]);
}

// Dé-quantifie un poids 8-bit gs64 `W[out_dim, in_dim]` (packed u8 + scales/biases
// bf16) en bf16 DENSE TRANSPOSÉ `wt[in_dim, out_dim]` (= `W^T`, layout `[K,N]`
// row-major attendu par `encode_na_gemm`). Un thread par élément. Pour le GEMM NA
// bf16 du prefill : `w = float(octet)*scale + bias` (même dé-quant que le qmv),
// arrondi bf16 (RTNE matériel) au stockage.
kernel void dequant_u8_to_bf16_t_gs64(
    device const uint* packed [[buffer(0)]],
    device const bfloat* scales [[buffer(1)]],
    device const bfloat* biases [[buffer(2)]],
    device bfloat* wt [[buffer(3)]],
    constant uint4& dims [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint out_dim = dims.x; // N
    const uint in_dim = dims.y;  // K
    const uint packed_cols = dims.z;
    const uint total = out_dim * in_dim;
    if (gid >= total) {
        return;
    }
    const uint n = gid / in_dim;       // ligne out
    const uint k = gid - n * in_dim;   // position in_dim
    const uint groups = in_dim / 64u;
    const uint word = packed[n * packed_cols + (k >> 2u)];
    const uint shift = (k & 3u) * 8u;
    const uint q = (word >> shift) & 0x000000ffu;
    const uint group = k / 64u;
    const float scale = scales[n * groups + group];
    const float bias = biases[n * groups + group];
    wt[k * out_dim + n] = bfloat((float(q) * scale) + bias);
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

    const uint mask = (1u << bits) - 1u;
    float acc = 0.0f;

    for (uint col = lane; col < in_dim; col += 32) {
        const uint bit_offset = col * bits;
        const uint word_col = bit_offset / 32u;
        const uint shift = bit_offset - (word_col * 32u);
        const uint row_base = row * packed_cols;
        uint q = packed[row_base + word_col] >> shift;
        if ((shift + bits) > 32u && (word_col + 1u) < packed_cols) {
            q |= packed[row_base + word_col + 1u] << (32u - shift);
        }
        q &= mask;
        const uint group = min(col / group_size, groups - 1u);
        const uint affine_index = (row * groups) + group;
        const float scale = scales[affine_index];
        const float bias = biases[affine_index];
        acc += lhs[(b * in_dim) + col] * ((float(q) * scale) + bias);
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
    constant float& embedding_scale [[buffer(4)]],
    constant uint& recast_bf16 [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    const uint vocab = dims.x;
    const uint dim = dims.y;
    const uint token = token_index[0];
    if (token >= vocab || tid >= dim) {
        return;
    }
    const float value = table[token * dim + tid] * embedding_scale;
    out[tid] = recast_bf16 != 0u ? float(bfloat(value)) : value;
}

kernel void embed_gather_affine_from_u32_f32(
    device const uint* packed [[buffer(0)]],
    device const bfloat* scales [[buffer(1)]],
    device const bfloat* biases [[buffer(2)]],
    device const uint* token_index [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant uint4& dims [[buffer(5)]],
    constant uint4& quant [[buffer(6)]],
    constant float& embedding_scale [[buffer(7)]],
    constant uint& recast_bf16 [[buffer(8)]],
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
    const float value =
        (float(q) * scales[affine_index] + biases[affine_index]) * embedding_scale;
    out[tid] = recast_bf16 != 0u ? float(bfloat(value)) : value;
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

kernel void affine_qmv_fast_u4_gs64_align64_f32(
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
    const uint values_per_block = values_per_thread * 32u;
    if (simd_gid >= simdgroups) {
        return;
    }
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
    const uint full_blocks = in_dim / values_per_block;
    for (uint block = 0u; block < full_blocks; ++block) {
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
            if (row_base + row >= out_dim) {
                continue;
            }
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
        x += values_per_block;
    }

    const uint tail_values = in_dim - full_blocks * values_per_block;
    const uint tail_offset = simd_lid * values_per_thread;
    if (tail_values > 0u && tail_offset < tail_values) {
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
            if (row_base + row >= out_dim) {
                continue;
            }
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

    for (uint row = 0u; row < results_per_simdgroup; ++row) {
        const float reduced = simd_sum(result[row]);
        if (simd_lid == 0u && row_base + row < out_dim) {
            out[tile.x * out_dim + row_base + row] = reduced;
        }
    }
}

static inline uint unpack_u6_affine(device const uint* packed, uint packed_cols, uint row_base, uint col) {
    const uint bit_offset = col * 6u;
    const uint word_col = bit_offset / 32u;
    const uint shift = bit_offset - (word_col * 32u);
    uint q = packed[row_base + word_col] >> shift;
    if ((shift + 6u) > 32u && (word_col + 1u) < packed_cols) {
        q |= packed[row_base + word_col + 1u] << (32u - shift);
    }
    return q & 0x3fu;
}

kernel void affine_qmv_fast_u6_gs64_f32(
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
    const uint row = tile.y * 2u + simd_gid;
    const uint b = tile.x;
    if (row >= out_dim) {
        return;
    }

    const uint row_base = row * packed_cols;
    float acc = 0.0f;
    for (uint col = simd_lid; col < in_dim; col += 32u) {
        const uint q = unpack_u6_affine(packed, packed_cols, row_base, col);
        const uint group = min(col / 64u, groups - 1u);
        const uint affine_index = row * groups + group;
        const float scale = scales[affine_index];
        const float bias = biases[affine_index];
        acc += lhs[b * in_dim + col] * ((float(q) * scale) + bias);
    }
    acc = simd_sum(acc);
    if (simd_lid == 0u) {
        out[b * out_dim + row] = acc;
    }
}

kernel void affine_qmv_fast_aligned_u6_gs64_f32(
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
    const uint row = tile.y * 2u + simd_gid;
    const uint b = tile.x;

    const uint row_base = row * packed_cols;
    float acc = 0.0f;
    for (uint col = simd_lid; col < in_dim; col += 32u) {
        const uint q = unpack_u6_affine(packed, packed_cols, row_base, col);
        const uint group = min(col / 64u, groups - 1u);
        const uint affine_index = row * groups + group;
        const float scale = scales[affine_index];
        const float bias = biases[affine_index];
        acc += lhs[b * in_dim + col] * ((float(q) * scale) + bias);
    }
    acc = simd_sum(acc);
    if (simd_lid == 0u) {
        out[b * out_dim + row] = acc;
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
    if (simd_gid >= simdgroups) {
        return;
    }
    const uint row_base = tile.y * (simdgroups * results_per_simdgroup) +
        simd_gid * results_per_simdgroup;
    if (row_base >= out_dim) {
        return;
    }

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
            if (row_base + row >= out_dim) {
                continue;
            }
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

kernel void affine_qmv_fast_u8_gs64_align64_f32(
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
    if (simd_gid >= simdgroups) {
        return;
    }
    const uint row_base = tile.y * (simdgroups * results_per_simdgroup) +
        simd_gid * results_per_simdgroup;
    if (row_base >= out_dim) {
        return;
    }

    const device uint* ws = packed + row_base * packed_cols + simd_lid;
    const device bfloat* scale_base = scales + row_base * groups;
    const device bfloat* bias_base = biases + row_base * groups;
    const device float* x = lhs + tile.x * in_dim + simd_lid * values_per_word;

    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const uint full_blocks = in_dim / values_per_block;
    for (uint block = 0u; block < full_blocks; ++block) {
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
            if (row_base + row >= out_dim) {
                continue;
            }
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

    const uint tail_values = in_dim - full_blocks * values_per_block;
    if (tail_values > 0u) {
        for (uint row = 0u; row < results_per_simdgroup; ++row) {
            if (row_base + row >= out_dim) {
                continue;
            }
            const device uint* row_words = ws + row * packed_cols;
            float accum = 0.0f;
            for (uint word = 0u; word < words_per_thread; ++word) {
                const uint value_offset = (word * 32u + simd_lid) * values_per_word;
                if (value_offset >= tail_values) {
                    continue;
                }
                const uint packed_word = row_words[word * 32u];
                const uint group = value_offset / 64u;
                const float scale = scale_base[row * groups + group];
                const float bias = bias_base[row * groups + group];
                const uint x_offset = word * 32u * values_per_word;
                accum += x[x_offset] * ((float(packed_word & 0x000000ffu) * scale) + bias);
                accum += x[x_offset + 1u] * ((float((packed_word >> 8u) & 0x000000ffu) * scale) + bias);
                accum += x[x_offset + 2u] * ((float((packed_word >> 16u) & 0x000000ffu) * scale) + bias);
                accum += x[x_offset + 3u] * ((float((packed_word >> 24u) & 0x000000ffu) * scale) + bias);
            }
            result[row] += accum;
        }
    }

    for (uint row = 0u; row < results_per_simdgroup; ++row) {
        const float reduced = simd_sum(result[row]);
        if (simd_lid == 0u && row_base + row < out_dim) {
            out[tile.x * out_dim + row_base + row] = reduced;
        }
    }
}

kernel void affine_qmv_plus_one_fast_aligned_u8_gs64_f32(
    device const float* lhs [[buffer(0)]],
    device const uint* packed [[buffer(1)]],
    device const bfloat* scales [[buffer(2)]],
    device const bfloat* biases [[buffer(3)]],
    device const uint* extra_packed [[buffer(4)]],
    device const bfloat* extra_scales [[buffer(5)]],
    device const bfloat* extra_biases [[buffer(6)]],
    device float* out [[buffer(7)]],
    device float* extra_out [[buffer(8)]],
    constant uint4& dims [[buffer(9)]],
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
    const uint out_groups = (out_dim + 7u) / 8u;

    if (tile.y == out_groups) {
        if (simd_gid != 0u) {
            return;
        }
        const device uint* ws = extra_packed + simd_lid;
        const device bfloat* scale_base = extra_scales;
        const device bfloat* bias_base = extra_biases;
        const device float* x = lhs + tile.x * in_dim + simd_lid * values_per_word;

        float result = 0.0f;
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

            float accum = 0.0f;
            for (uint word = 0u; word < words_per_thread; ++word) {
                const uint packed_word = ws[word * 32u];
                const uint group = (simd_lid + word * 32u) / 16u;
                const float scale = scale_base[group];
                const float bias = bias_base[group];
                const uint base = word * values_per_word;
                    accum += xt[base] * ((float(packed_word & 0x000000ffu) * scale) + bias);
                    accum += xt[base + 1u] * ((float((packed_word >> 8u) & 0x000000ffu) * scale) + bias);
                    accum += xt[base + 2u] * ((float((packed_word >> 16u) & 0x000000ffu) * scale) + bias);
                    accum += xt[base + 3u] * ((float((packed_word >> 24u) & 0x000000ffu) * scale) + bias);
            }
            result += accum;

            ws += words_per_block;
            scale_base += values_per_block / 64u;
            bias_base += values_per_block / 64u;
            x += values_per_block;
        }

        const float reduced = simd_sum(result);
        if (simd_lid == 0u) {
            extra_out[tile.x] = reduced;
        }
        return;
    }
    if (tile.y > out_groups) {
        return;
    }

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
            const uint out_row = row_base + row;
            if (out_row < out_dim) {
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
        }

        ws += words_per_block;
        scale_base += values_per_block / 64u;
        bias_base += values_per_block / 64u;
        x += values_per_block;
    }

    for (uint row = 0u; row < results_per_simdgroup; ++row) {
        const uint out_row = row_base + row;
        const float reduced = simd_sum(result[row]);
        if (simd_lid == 0u && out_row < out_dim) {
            out[tile.x * out_dim + out_row] = reduced;
        }
    }
}

// Cas oQ Qwen3.6/MoE : shared_expert_gate = une seule sortie u8 gs64
// (1x2048). Le qmv u8 rapide historique exige out_dim % 8 == 0, donc ce poids
// retombe sinon sur le kernel generique. Celui-ci garde exactement le meme
// dequant u8 gs64 que `affine_qmv_fast_aligned_u8_gs64_f32`, mais avec un seul
// simdgroup et une seule reduction scalaire.
kernel void affine_qmv_one_fast_u8_gs64_f32(
    device const float* lhs [[buffer(0)]],
    device const uint* packed [[buffer(1)]],
    device const bfloat* scales [[buffer(2)]],
    device const bfloat* biases [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant uint4& dims [[buffer(5)]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint2 tile [[threadgroup_position_in_grid]]
) {
    const uint in_dim = dims.y;
    const uint packed_cols = dims.z;
    const uint groups = dims.w;
    const uint values_per_word = 4u;
    const uint words_per_thread = 4u;
    const uint values_per_thread = values_per_word * words_per_thread;
    const uint values_per_block = values_per_thread * 32u;
    const uint words_per_block = values_per_block / values_per_word;

    const device uint* ws = packed + simd_lid;
    const device bfloat* scale_base = scales;
    const device bfloat* bias_base = biases;
    const device float* x = lhs + tile.x * in_dim + simd_lid * values_per_word;

    float result = 0.0f;
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

        float accum = 0.0f;
        for (uint word = 0u; word < words_per_thread; ++word) {
            const uint packed_word = ws[word * 32u];
            const uint group = (simd_lid + word * 32u) / 16u;
            const float scale = scale_base[group];
            const float bias = bias_base[group];
            const uint base = word * values_per_word;
                    accum += xt[base] * ((float(packed_word & 0x000000ffu) * scale) + bias);
                    accum += xt[base + 1u] * ((float((packed_word >> 8u) & 0x000000ffu) * scale) + bias);
                    accum += xt[base + 2u] * ((float((packed_word >> 16u) & 0x000000ffu) * scale) + bias);
                    accum += xt[base + 3u] * ((float((packed_word >> 24u) & 0x000000ffu) * scale) + bias);
        }
        result += accum;

        ws += words_per_block;
        scale_base += values_per_block / 64u;
        bias_base += values_per_block / 64u;
        x += values_per_block;
    }

    const float reduced = simd_sum(result);
    if (simd_lid == 0u) {
        out[tile.x] = reduced;
    }
}

// qmm petit-M (M=2) 8-bit : variante de `affine_qmv_fast_aligned_u8` traitant
// DEUX lignes d'activation, le mot de poids étant lu UNE fois et réutilisé pour
// les 2 lignes (accumulateurs r0/r1 disjoints, même ordre d'accumulation par
// ligne que le qmv u8 → sortie bit-identique à 2 appels). Pour le duo
// light-batch sur les modèles DWQ (attention/LA/lm_head en 8-bit gs64).
// lhs = [2, in_dim] ; out = [2, out_dim] ; dims = [out_dim, in_dim, packed_cols, groups].
kernel void affine_qmm2_fast_aligned_u8_gs64_f32(
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
    if (simd_gid >= simdgroups) {
        return;
    }
    const uint row_base = tile.y * (simdgroups * results_per_simdgroup) +
        simd_gid * results_per_simdgroup;
    if (row_base >= out_dim) {
        return;
    }

    const device uint* ws = packed + row_base * packed_cols + simd_lid;
    const device bfloat* scale_base = scales + row_base * groups;
    const device bfloat* bias_base = biases + row_base * groups;
    const device float* x0 = lhs + simd_lid * values_per_word;
    const device float* x1 = lhs + in_dim + simd_lid * values_per_word;

    float r0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float r1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint k = 0u; k < in_dim; k += values_per_block) {
        float xt0[16];
        float xt1[16];
        for (uint word = 0u; word < words_per_thread; ++word) {
            const uint base = word * values_per_word;
            const uint x_offset = word * 32u * values_per_word;
            xt0[base] = x0[x_offset];
            xt0[base + 1u] = x0[x_offset + 1u];
            xt0[base + 2u] = x0[x_offset + 2u];
            xt0[base + 3u] = x0[x_offset + 3u];
            xt1[base] = x1[x_offset];
            xt1[base + 1u] = x1[x_offset + 1u];
            xt1[base + 2u] = x1[x_offset + 2u];
            xt1[base + 3u] = x1[x_offset + 3u];
        }

        for (uint row = 0u; row < results_per_simdgroup; ++row) {
            const device uint* row_words = ws + row * packed_cols;
            float ac0 = 0.0f;
            float ac1 = 0.0f;
            for (uint word = 0u; word < words_per_thread; ++word) {
                const uint packed_word = row_words[word * 32u];
                const uint group = (simd_lid + word * 32u) / 16u;
                const float scale = scale_base[row * groups + group];
                const float bias = bias_base[row * groups + group];
                const uint base = word * values_per_word;
                const float w0 = (float(packed_word & 0x000000ffu) * scale) + bias;
                const float w1 = (float((packed_word >> 8u) & 0x000000ffu) * scale) + bias;
                const float w2 = (float((packed_word >> 16u) & 0x000000ffu) * scale) + bias;
                const float w3 = (float((packed_word >> 24u) & 0x000000ffu) * scale) + bias;
                ac0 += xt0[base] * w0;
                ac0 += xt0[base + 1u] * w1;
                ac0 += xt0[base + 2u] * w2;
                ac0 += xt0[base + 3u] * w3;
                ac1 += xt1[base] * w0;
                ac1 += xt1[base + 1u] * w1;
                ac1 += xt1[base + 2u] * w2;
                ac1 += xt1[base + 3u] * w3;
            }
            r0[row] += ac0;
            r1[row] += ac1;
        }

        ws += words_per_block;
        scale_base += values_per_block / 64u;
        bias_base += values_per_block / 64u;
        x0 += values_per_block;
        x1 += values_per_block;
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

// Variantes oQ8 gs128 : meme packing u8 que gs64, mais une scale/bias couvre
// 128 valeurs au lieu de 64. Le groupe intra-bloc devient position/128, et le
// curseur scales/biases avance de 512/128 groupes par bloc.
kernel void affine_qmv_fast_aligned_u8_gs128_f32(
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
            if (row_base + row >= out_dim) {
                continue;
            }
            const device uint* row_words = ws + row * packed_cols;
            float accum = 0.0f;
            for (uint word = 0u; word < words_per_thread; ++word) {
                const uint packed_word = row_words[word * 32u];
                const uint group = (simd_lid + word * 32u) / 32u;
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
        scale_base += values_per_block / 128u;
        bias_base += values_per_block / 128u;
        x += values_per_block;
    }

    for (uint row = 0u; row < results_per_simdgroup; ++row) {
        const float reduced = simd_sum(result[row]);
        if (simd_lid == 0u) {
            out[tile.x * out_dim + row_base + row] = reduced;
        }
    }
}

kernel void affine_qmv_fast_aligned_u8_gs64_dot4_f32(
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
    if (row_base >= out_dim) {
        return;
    }

    const device uint* ws = packed + row_base * packed_cols + simd_lid;
    const device bfloat* scale_base = scales + row_base * groups;
    const device bfloat* bias_base = biases + row_base * groups;
    const device float* x = lhs + tile.x * in_dim + simd_lid * values_per_word;

    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint k = 0u; k < in_dim; k += values_per_block) {
        float4 xt[4];
        for (uint word = 0u; word < words_per_thread; ++word) {
            const uint x_offset = word * 32u * values_per_word;
            xt[word] = float4(x[x_offset], x[x_offset + 1u], x[x_offset + 2u], x[x_offset + 3u]);
        }

        for (uint row = 0u; row < results_per_simdgroup; ++row) {
            if (row_base + row >= out_dim) {
                continue;
            }
            const device uint* row_words = ws + row * packed_cols;
            float accum = 0.0f;
            for (uint word = 0u; word < words_per_thread; ++word) {
                const uint packed_word = row_words[word * 32u];
                const uint group = (simd_lid + word * 32u) / 16u;
                const float scale = scale_base[row * groups + group];
                const float bias = bias_base[row * groups + group];
                accum += dot4_u8_affine(packed_word, xt[word], scale, bias);
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
        if (simd_lid == 0u && row_base + row < out_dim) {
            out[tile.x * out_dim + row_base + row] = reduced;
        }
    }
}

kernel void affine_qmv_fast_aligned_u8_gs128_dot4_f32(
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
    if (row_base >= out_dim) {
        return;
    }

    const device uint* ws = packed + row_base * packed_cols + simd_lid;
    const device bfloat* scale_base = scales + row_base * groups;
    const device bfloat* bias_base = biases + row_base * groups;
    const device float* x = lhs + tile.x * in_dim + simd_lid * values_per_word;

    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint k = 0u; k < in_dim; k += values_per_block) {
        float4 xt[4];
        for (uint word = 0u; word < words_per_thread; ++word) {
            const uint x_offset = word * 32u * values_per_word;
            xt[word] = float4(x[x_offset], x[x_offset + 1u], x[x_offset + 2u], x[x_offset + 3u]);
        }

        for (uint row = 0u; row < results_per_simdgroup; ++row) {
            if (row_base + row >= out_dim) {
                continue;
            }
            const device uint* row_words = ws + row * packed_cols;
            float accum = 0.0f;
            for (uint word = 0u; word < words_per_thread; ++word) {
                const uint packed_word = row_words[word * 32u];
                const uint group = (simd_lid + word * 32u) / 32u;
                const float scale = scale_base[row * groups + group];
                const float bias = bias_base[row * groups + group];
                accum += dot4_u8_affine(packed_word, xt[word], scale, bias);
            }
            result[row] += accum;
        }

        ws += words_per_block;
        scale_base += values_per_block / 128u;
        bias_base += values_per_block / 128u;
        x += values_per_block;
    }

    for (uint row = 0u; row < results_per_simdgroup; ++row) {
        const float reduced = simd_sum(result[row]);
        if (simd_lid == 0u && row_base + row < out_dim) {
            out[tile.x * out_dim + row_base + row] = reduced;
        }
    }
}

kernel void affine_qmv_fast_aligned_u8_gs64_tg128_f32(
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
    const uint simdgroups = 4u;
    const uint values_per_word = 4u;
    const uint words_per_thread = 4u;
    const uint values_per_thread = values_per_word * words_per_thread;
    const uint values_per_block = values_per_thread * 32u;
    const uint words_per_block = values_per_block / values_per_word;
    if (simd_gid >= simdgroups) {
        return;
    }
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
        if (simd_lid == 0u && row_base + row < out_dim) {
            out[tile.x * out_dim + row_base + row] = reduced;
        }
    }
}

kernel void affine_qmv_fast_aligned_u8_gs128_tg128_f32(
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
    const uint simdgroups = 4u;
    const uint values_per_word = 4u;
    const uint words_per_thread = 4u;
    const uint values_per_thread = values_per_word * words_per_thread;
    const uint values_per_block = values_per_thread * 32u;
    const uint words_per_block = values_per_block / values_per_word;
    if (simd_gid >= simdgroups) {
        return;
    }
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
                const uint group = (simd_lid + word * 32u) / 32u;
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
        scale_base += values_per_block / 128u;
        bias_base += values_per_block / 128u;
        x += values_per_block;
    }

    for (uint row = 0u; row < results_per_simdgroup; ++row) {
        const float reduced = simd_sum(result[row]);
        if (simd_lid == 0u && row_base + row < out_dim) {
            out[tile.x * out_dim + row_base + row] = reduced;
        }
    }
}

kernel void affine_qmv_fast_aligned_u8_gs64_tg256_f32(
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
    const uint simdgroups = 8u;
    const uint values_per_word = 4u;
    const uint words_per_thread = 4u;
    const uint values_per_thread = values_per_word * words_per_thread;
    const uint values_per_block = values_per_thread * 32u;
    const uint words_per_block = values_per_block / values_per_word;
    if (simd_gid >= simdgroups) {
        return;
    }
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
        if (simd_lid == 0u && row_base + row < out_dim) {
            out[tile.x * out_dim + row_base + row] = reduced;
        }
    }
}

kernel void affine_qmv_fast_aligned_u8_gs128_tg256_f32(
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
    const uint simdgroups = 8u;
    const uint values_per_word = 4u;
    const uint words_per_thread = 4u;
    const uint values_per_thread = values_per_word * words_per_thread;
    const uint values_per_block = values_per_thread * 32u;
    const uint words_per_block = values_per_block / values_per_word;
    if (simd_gid >= simdgroups) {
        return;
    }
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
                const uint group = (simd_lid + word * 32u) / 32u;
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
        scale_base += values_per_block / 128u;
        bias_base += values_per_block / 128u;
        x += values_per_block;
    }

    for (uint row = 0u; row < results_per_simdgroup; ++row) {
        const float reduced = simd_sum(result[row]);
        if (simd_lid == 0u && row_base + row < out_dim) {
            out[tile.x * out_dim + row_base + row] = reduced;
        }
    }
}

kernel void affine_qmm2_fast_aligned_u8_gs128_f32(
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
    const device float* x0 = lhs + simd_lid * values_per_word;
    const device float* x1 = lhs + in_dim + simd_lid * values_per_word;

    float r0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float r1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint k = 0u; k < in_dim; k += values_per_block) {
        float xt0[16];
        float xt1[16];
        for (uint word = 0u; word < words_per_thread; ++word) {
            const uint base = word * values_per_word;
            const uint x_offset = word * 32u * values_per_word;
            xt0[base] = x0[x_offset];
            xt0[base + 1u] = x0[x_offset + 1u];
            xt0[base + 2u] = x0[x_offset + 2u];
            xt0[base + 3u] = x0[x_offset + 3u];
            xt1[base] = x1[x_offset];
            xt1[base + 1u] = x1[x_offset + 1u];
            xt1[base + 2u] = x1[x_offset + 2u];
            xt1[base + 3u] = x1[x_offset + 3u];
        }

        for (uint row = 0u; row < results_per_simdgroup; ++row) {
            const device uint* row_words = ws + row * packed_cols;
            float ac0 = 0.0f;
            float ac1 = 0.0f;
            for (uint word = 0u; word < words_per_thread; ++word) {
                const uint packed_word = row_words[word * 32u];
                const uint group = (simd_lid + word * 32u) / 32u;
                const float scale = scale_base[row * groups + group];
                const float bias = bias_base[row * groups + group];
                const uint base = word * values_per_word;
                const float w0 = (float(packed_word & 0x000000ffu) * scale) + bias;
                const float w1 = (float((packed_word >> 8u) & 0x000000ffu) * scale) + bias;
                const float w2 = (float((packed_word >> 16u) & 0x000000ffu) * scale) + bias;
                const float w3 = (float((packed_word >> 24u) & 0x000000ffu) * scale) + bias;
                ac0 += xt0[base] * w0;
                ac0 += xt0[base + 1u] * w1;
                ac0 += xt0[base + 2u] * w2;
                ac0 += xt0[base + 3u] * w3;
                ac1 += xt1[base] * w0;
                ac1 += xt1[base + 1u] * w1;
                ac1 += xt1[base + 2u] * w2;
                ac1 += xt1[base + 3u] * w3;
            }
            r0[row] += ac0;
            r1[row] += ac1;
        }

        ws += words_per_block;
        scale_base += values_per_block / 128u;
        bias_base += values_per_block / 128u;
        x0 += values_per_block;
        x1 += values_per_block;
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

kernel void geglu_tanh_f32(
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
    const float inner = 0.7978846f * (g + 0.044715f * g * g * g);
    // NOTE: le tanh fast-math passe par exp(2x) et déborde en NaN dès inner≈44
    // (les grosses activations Gemma poussent inner≈4690). On sature l'argument :
    // tanh(±20)=±1 exactement en f32, comme la référence CPU f32::tanh saturante.
    out[gid] = (0.5f * g * (1.0f + tanh(clamp(inner, -20.0f, 20.0f)))) * up[gid];
}

// Désinterleave la projection q_proj batchée `[seq, 2*q_dim]` (layout par tête
// `[q_head | gate_head]`, `start = head*2*head_dim`) en `q [seq, q_dim]` et
// `gate [seq, q_dim]`. Version BATCHÉE (seq rows) de `split_q_gate_f32` (decode).
kernel void split_q_gate_rows_f32(
    device const float* proj [[buffer(0)]],
    device float* q [[buffer(1)]],
    device float* gate [[buffer(2)]],
    constant uint4& dims [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint seq = dims.x;
    const uint num_heads = dims.y;
    const uint head_dim = dims.z;
    const uint q_dim = num_heads * head_dim;
    const uint total = seq * q_dim;
    if (gid >= total) { return; }
    const uint row = gid / q_dim;
    const uint j = gid % q_dim;
    const uint head = j / head_dim;
    const uint col = j % head_dim;
    const uint row_stride = dims.w == 0u ? 2u * q_dim : dims.w;
    const uint base = row * row_stride + head * 2u * head_dim;
    q[gid] = proj[base + col];
    gate[gid] = proj[base + head_dim + col];
}

// Gate de sortie batché : `out[i] = ctx[i] * sigmoid(gate[i])` sur `seq*q_dim`.
kernel void attn_gate_rows_f32(
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

kernel void add_scaled_f32(
    device const float* left [[buffer(0)]],
    device const float* right [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant float& scale [[buffer(3)]],
    constant uint& len [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) {
        return;
    }
    out[gid] = left[gid] + right[gid] * scale;
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

kernel void linear_attn_conv_silu_k4_f32(
    device const float* qkv [[buffer(0)]],
    device const float* conv_weight [[buffer(1)]],
    device float* conv_state [[buffer(2)]],
    device float* conv_out [[buffer(3)]],
    constant uint2& dims [[buffer(4)]],
    uint channel [[thread_position_in_grid]]
) {
    const uint conv_dim = dims.x;
    const uint kernel_width = dims.y;
    if (channel >= conv_dim || kernel_width != 4u) {
        return;
    }

    const uint idx0 = channel;
    const uint idx1 = conv_dim + channel;
    const uint idx2 = (2u * conv_dim) + channel;
    const uint w = channel * 4u;
    const float x = qkv[channel];
    float acc = 0.0f;
    acc += conv_state[idx0] * conv_weight[w];
    acc += conv_state[idx1] * conv_weight[w + 1u];
    acc += conv_state[idx2] * conv_weight[w + 2u];
    acc += x * conv_weight[w + 3u];
    conv_out[channel] = acc / (1.0f + exp(-acc));

    conv_state[idx0] = conv_state[idx1];
    conv_state[idx1] = conv_state[idx2];
    conv_state[idx2] = x;
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

kernel void linear_attn_norm_gates_dk128_f32(
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
    if (key_head_dim != 128u) {
        return;
    }
    const uint key_dim = key_heads * 128u;

    if (item < key_heads) {
        const uint base = item * 128u;
        const uint idx0 = base + lane;
        const uint idx1 = idx0 + 32u;
        const uint idx2 = idx0 + 64u;
        const uint idx3 = idx0 + 96u;
        const uint kidx0 = key_dim + idx0;
        const uint kidx1 = key_dim + idx1;
        const uint kidx2 = key_dim + idx2;
        const uint kidx3 = key_dim + idx3;
        const float q0 = conv_out[idx0];
        const float q1 = conv_out[idx1];
        const float q2 = conv_out[idx2];
        const float q3 = conv_out[idx3];
        const float k0 = conv_out[kidx0];
        const float k1 = conv_out[kidx1];
        const float k2 = conv_out[kidx2];
        const float k3 = conv_out[kidx3];
        float q_ss = 0.0f;
        float k_ss = 0.0f;
        q_ss += q0 * q0;
        k_ss += k0 * k0;
        q_ss += q1 * q1;
        k_ss += k1 * k1;
        q_ss += q2 * q2;
        k_ss += k2 * k2;
        q_ss += q3 * q3;
        k_ss += k3 * k3;
        const float q_mean = simd_sum(q_ss) / float(key_head_dim);
        const float k_mean = simd_sum(k_ss) / float(key_head_dim);
        const float q_inv = rsqrt(q_mean + 1.0e-6f) * scales.x;
        const float k_inv = rsqrt(k_mean + 1.0e-6f) * scales.y;
        q_norm[idx0] = q0 * q_inv;
        q_norm[idx1] = q1 * q_inv;
        q_norm[idx2] = q2 * q_inv;
        q_norm[idx3] = q3 * q_inv;
        k_norm[idx0] = k0 * k_inv;
        k_norm[idx1] = k1 * k_inv;
        k_norm[idx2] = k2 * k_inv;
        k_norm[idx3] = k3 * k_inv;
    }

    if (item < value_heads && lane == 0u) {
        const float b = beta_input[item];
        beta[item] = 1.0f / (1.0f + exp(-b));
        const float dt_arg = gate_input[item] + dt_bias[item];
        const float dt = (dt_arg > 20.0f) ? dt_arg : log(1.0f + exp(dt_arg));
        decay[item] = exp(-exp(a_log[item]) * dt);
    }
}

kernel void linear_attn_norm_gates_inv_dk128_f32(
    device const float* conv_out [[buffer(0)]],
    device const float* beta_input [[buffer(1)]],
    device const float* gate_input [[buffer(2)]],
    device const float* a_log [[buffer(3)]],
    device const float* dt_bias [[buffer(4)]],
    device float* q_inv [[buffer(5)]],
    device float* k_inv [[buffer(6)]],
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
    if (key_head_dim != 128u) {
        return;
    }
    const uint key_dim = key_heads * 128u;

    if (item < key_heads) {
        const uint base = item * 128u;
        const uint idx0 = base + lane;
        const uint idx1 = idx0 + 32u;
        const uint idx2 = idx0 + 64u;
        const uint idx3 = idx0 + 96u;
        const uint kidx0 = key_dim + idx0;
        const uint kidx1 = key_dim + idx1;
        const uint kidx2 = key_dim + idx2;
        const uint kidx3 = key_dim + idx3;
        const float q0 = conv_out[idx0];
        const float q1 = conv_out[idx1];
        const float q2 = conv_out[idx2];
        const float q3 = conv_out[idx3];
        const float k0 = conv_out[kidx0];
        const float k1 = conv_out[kidx1];
        const float k2 = conv_out[kidx2];
        const float k3 = conv_out[kidx3];
        float q_ss = 0.0f;
        float k_ss = 0.0f;
        q_ss += q0 * q0;
        k_ss += k0 * k0;
        q_ss += q1 * q1;
        k_ss += k1 * k1;
        q_ss += q2 * q2;
        k_ss += k2 * k2;
        q_ss += q3 * q3;
        k_ss += k3 * k3;
        const float q_mean = simd_sum(q_ss) / float(key_head_dim);
        const float k_mean = simd_sum(k_ss) / float(key_head_dim);
        if (lane == 0u) {
            q_inv[item] = rsqrt(q_mean + 1.0e-6f) * scales.x;
            k_inv[item] = rsqrt(k_mean + 1.0e-6f) * scales.y;
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

static inline float linear_attn_conv_k4_channel(
    device const float* qkv,
    device const float* conv_weight,
    device float* conv_state,
    uint conv_dim,
    uint channel
) {
    const uint w = channel * 4u;
    const float x = qkv[channel];
    float acc = 0.0f;
    acc += conv_state[channel] * conv_weight[w];
    acc += conv_state[conv_dim + channel] * conv_weight[w + 1u];
    acc += conv_state[(2u * conv_dim) + channel] * conv_weight[w + 2u];
    acc += x * conv_weight[w + 3u];
    conv_state[channel] = conv_state[conv_dim + channel];
    conv_state[conv_dim + channel] = conv_state[(2u * conv_dim) + channel];
    conv_state[(2u * conv_dim) + channel] = x;
    return acc / (1.0f + exp(-acc));
}

kernel void linear_attn_conv_norm_gates_k4_dk128_f32(
    device const float* qkv [[buffer(0)]],
    device const float* beta_input [[buffer(1)]],
    device const float* gate_input [[buffer(2)]],
    device const float* conv_weight [[buffer(3)]],
    device float* conv_state [[buffer(4)]],
    device const float* a_log [[buffer(5)]],
    device const float* dt_bias [[buffer(6)]],
    device float* conv_out [[buffer(7)]],
    device float* q_norm [[buffer(8)]],
    device float* k_norm [[buffer(9)]],
    device float* beta [[buffer(10)]],
    device float* decay [[buffer(11)]],
    constant uint4& dims [[buffer(12)]],
    constant float2& scales [[buffer(13)]],
    uint lane [[thread_index_in_threadgroup]],
    uint item [[threadgroup_position_in_grid]]
) {
    const uint key_heads = dims.x;
    const uint value_heads = dims.y;
    const uint key_head_dim = dims.z;
    const uint value_head_dim = dims.w;
    if (key_head_dim != 128u || value_head_dim != 128u) {
        return;
    }

    const uint key_dim = key_heads * 128u;
    const uint conv_dim = (2u * key_dim) + (value_heads * 128u);

    if (item < key_heads) {
        const uint base = item * 128u;
        const uint idx0 = base + lane;
        const uint idx1 = idx0 + 32u;
        const uint idx2 = idx0 + 64u;
        const uint idx3 = idx0 + 96u;
        const uint kidx0 = key_dim + idx0;
        const uint kidx1 = key_dim + idx1;
        const uint kidx2 = key_dim + idx2;
        const uint kidx3 = key_dim + idx3;

        const float q0 = linear_attn_conv_k4_channel(qkv, conv_weight, conv_state, conv_dim, idx0);
        const float q1 = linear_attn_conv_k4_channel(qkv, conv_weight, conv_state, conv_dim, idx1);
        const float q2 = linear_attn_conv_k4_channel(qkv, conv_weight, conv_state, conv_dim, idx2);
        const float q3 = linear_attn_conv_k4_channel(qkv, conv_weight, conv_state, conv_dim, idx3);
        const float k0 = linear_attn_conv_k4_channel(qkv, conv_weight, conv_state, conv_dim, kidx0);
        const float k1 = linear_attn_conv_k4_channel(qkv, conv_weight, conv_state, conv_dim, kidx1);
        const float k2 = linear_attn_conv_k4_channel(qkv, conv_weight, conv_state, conv_dim, kidx2);
        const float k3 = linear_attn_conv_k4_channel(qkv, conv_weight, conv_state, conv_dim, kidx3);
        float q_ss = 0.0f;
        float k_ss = 0.0f;
        q_ss += q0 * q0;
        k_ss += k0 * k0;
        q_ss += q1 * q1;
        k_ss += k1 * k1;
        q_ss += q2 * q2;
        k_ss += k2 * k2;
        q_ss += q3 * q3;
        k_ss += k3 * k3;
        const float q_mean = simd_sum(q_ss) / float(key_head_dim);
        const float k_mean = simd_sum(k_ss) / float(key_head_dim);
        const float q_inv = rsqrt(q_mean + 1.0e-6f) * scales.x;
        const float k_inv = rsqrt(k_mean + 1.0e-6f) * scales.y;
        q_norm[idx0] = q0 * q_inv;
        q_norm[idx1] = q1 * q_inv;
        q_norm[idx2] = q2 * q_inv;
        q_norm[idx3] = q3 * q_inv;
        k_norm[idx0] = k0 * k_inv;
        k_norm[idx1] = k1 * k_inv;
        k_norm[idx2] = k2 * k_inv;
        k_norm[idx3] = k3 * k_inv;
    }

    if (item < value_heads) {
        const uint vbase = (2u * key_dim) + (item * 128u);
        const uint idx0 = vbase + lane;
        const uint idx1 = idx0 + 32u;
        const uint idx2 = idx0 + 64u;
        const uint idx3 = idx0 + 96u;

        conv_out[idx0] = linear_attn_conv_k4_channel(qkv, conv_weight, conv_state, conv_dim, idx0);
        conv_out[idx1] = linear_attn_conv_k4_channel(qkv, conv_weight, conv_state, conv_dim, idx1);
        conv_out[idx2] = linear_attn_conv_k4_channel(qkv, conv_weight, conv_state, conv_dim, idx2);
        conv_out[idx3] = linear_attn_conv_k4_channel(qkv, conv_weight, conv_state, conv_dim, idx3);

        if (lane == 0u) {
            const float b = beta_input[item];
            beta[item] = 1.0f / (1.0f + exp(-b));
            const float dt_arg = gate_input[item] + dt_bias[item];
            const float dt = (dt_arg > 20.0f) ? dt_arg : log(1.0f + exp(dt_arg));
            decay[item] = exp(-exp(a_log[item]) * dt);
        }
    }
}

// Brick #8 prefill : conv causale 4-tap BATCHÉE — token t lit sa fenêtre [t-3..t]
// depuis la séquence qkv (ou conv_state initial pour t<3), sans update séquentiel de
// conv_state. Sortie byte-identique au per-token. (L'update de conv_state aux 3
// derniers tokens est fait par linear_attn_conv_state_finalize_f32.)
static inline float linear_attn_conv_k4_batch_channel(
    device const float* qkv,
    device const float* conv_weight,
    device const float* conv_state0,
    uint conv_dim,
    uint channel,
    uint token,
    uint batch
) {
    const uint w = channel * 4u;
    float acc = 0.0f;
    for (int k = 0; k < 4; k++) {
        int p = int(token) - 3 + k;
        float x;
        if (p >= 0) {
            x = qkv[uint(p) * conv_dim + channel];
        } else {
            x = conv_state0[uint(p + 3) * conv_dim + channel];
        }
        acc += x * conv_weight[w + uint(k)];
    }
    return acc / (1.0f + exp(-acc));
}

kernel void linear_attn_conv_norm_gates_k4_dk128_batch_f32(
    device const float* qkv [[buffer(0)]],
    device const float* beta_input [[buffer(1)]],
    device const float* gate_input [[buffer(2)]],
    device const float* conv_weight [[buffer(3)]],
    device const float* conv_state0 [[buffer(4)]],
    device const float* a_log [[buffer(5)]],
    device const float* dt_bias [[buffer(6)]],
    device float* conv_out [[buffer(7)]],
    device float* q_norm [[buffer(8)]],
    device float* k_norm [[buffer(9)]],
    device float* beta [[buffer(10)]],
    device float* decay [[buffer(11)]],
    constant uint4& dims [[buffer(12)]],
    constant float2& scales [[buffer(13)]],
    constant uint& batch [[buffer(14)]],
    uint lane [[thread_index_in_threadgroup]],
    uint2 tg [[threadgroup_position_in_grid]]
) {
    const uint key_heads = dims.x;
    const uint value_heads = dims.y;
    const uint key_head_dim = dims.z;
    const uint value_head_dim = dims.w;
    if (key_head_dim != 128u || value_head_dim != 128u) {
        return;
    }
    const uint item = tg.x;
    const uint token = tg.y;
    if (token >= batch) {
        return;
    }
    const uint key_dim = key_heads * 128u;
    const uint conv_dim = (2u * key_dim) + (value_heads * 128u);
    const uint qrow = token * key_dim;
    const uint crow = token * conv_dim;
    const uint grow = token * value_heads;

    if (item < key_heads) {
        const uint base = item * 128u;
        const uint l0 = base + lane, l1 = l0 + 32u, l2 = l0 + 64u, l3 = l0 + 96u;
        const uint k0c = key_dim + l0, k1c = key_dim + l1, k2c = key_dim + l2, k3c = key_dim + l3;
        const float q0 = linear_attn_conv_k4_batch_channel(qkv, conv_weight, conv_state0, conv_dim, l0, token, batch);
        const float q1 = linear_attn_conv_k4_batch_channel(qkv, conv_weight, conv_state0, conv_dim, l1, token, batch);
        const float q2 = linear_attn_conv_k4_batch_channel(qkv, conv_weight, conv_state0, conv_dim, l2, token, batch);
        const float q3 = linear_attn_conv_k4_batch_channel(qkv, conv_weight, conv_state0, conv_dim, l3, token, batch);
        const float kk0 = linear_attn_conv_k4_batch_channel(qkv, conv_weight, conv_state0, conv_dim, k0c, token, batch);
        const float kk1 = linear_attn_conv_k4_batch_channel(qkv, conv_weight, conv_state0, conv_dim, k1c, token, batch);
        const float kk2 = linear_attn_conv_k4_batch_channel(qkv, conv_weight, conv_state0, conv_dim, k2c, token, batch);
        const float kk3 = linear_attn_conv_k4_batch_channel(qkv, conv_weight, conv_state0, conv_dim, k3c, token, batch);
        float q_ss = q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3;
        float k_ss = kk0 * kk0 + kk1 * kk1 + kk2 * kk2 + kk3 * kk3;
        const float q_inv = rsqrt(simd_sum(q_ss) / float(key_head_dim) + 1.0e-6f) * scales.x;
        const float k_inv = rsqrt(simd_sum(k_ss) / float(key_head_dim) + 1.0e-6f) * scales.y;
        q_norm[qrow + l0] = q0 * q_inv;
        q_norm[qrow + l1] = q1 * q_inv;
        q_norm[qrow + l2] = q2 * q_inv;
        q_norm[qrow + l3] = q3 * q_inv;
        k_norm[qrow + l0] = kk0 * k_inv;
        k_norm[qrow + l1] = kk1 * k_inv;
        k_norm[qrow + l2] = kk2 * k_inv;
        k_norm[qrow + l3] = kk3 * k_inv;
    }

    if (item < value_heads) {
        const uint vbase = (2u * key_dim) + (item * 128u);
        const uint l0 = vbase + lane, l1 = l0 + 32u, l2 = l0 + 64u, l3 = l0 + 96u;
        conv_out[crow + l0] = linear_attn_conv_k4_batch_channel(qkv, conv_weight, conv_state0, conv_dim, l0, token, batch);
        conv_out[crow + l1] = linear_attn_conv_k4_batch_channel(qkv, conv_weight, conv_state0, conv_dim, l1, token, batch);
        conv_out[crow + l2] = linear_attn_conv_k4_batch_channel(qkv, conv_weight, conv_state0, conv_dim, l2, token, batch);
        conv_out[crow + l3] = linear_attn_conv_k4_batch_channel(qkv, conv_weight, conv_state0, conv_dim, l3, token, batch);
        if (lane == 0u) {
            const float b = beta_input[grow + item];
            beta[grow + item] = 1.0f / (1.0f + exp(-b));
            const float dt_arg = gate_input[grow + item] + dt_bias[item];
            const float dt = (dt_arg > 20.0f) ? dt_arg : log(1.0f + exp(dt_arg));
            decay[grow + item] = exp(-exp(a_log[item]) * dt);
        }
    }
}

// Met conv_state aux 3 derniers tokens de la séquence (état pour le décode suivant).
// 1 thread/canal. Suppose batch >= 3 (prefill). conv_state[k*cd+c] = qkv[(batch-3+k)*cd+c].
kernel void linear_attn_conv_state_finalize_f32(
    device const float* qkv [[buffer(0)]],
    device float* conv_state [[buffer(1)]],
    constant uint2& dims [[buffer(2)]], // (conv_dim, batch)
    uint gid [[thread_position_in_grid]]
) {
    const uint conv_dim = dims.x;
    const uint batch = dims.y;
    if (gid >= conv_dim || batch < 3u) {
        return;
    }
    conv_state[gid] = qkv[(batch - 3u) * conv_dim + gid];
    conv_state[conv_dim + gid] = qkv[(batch - 2u) * conv_dim + gid];
    conv_state[(2u * conv_dim) + gid] = qkv[(batch - 1u) * conv_dim + gid];
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

kernel void linear_attn_gated_delta_dk128_tg4_f32(
    device const float* conv_out [[buffer(0)]],
    device const float* q_norm [[buffer(1)]],
    device const float* k_norm [[buffer(2)]],
    device const float* beta [[buffer(3)]],
    device const float* decay [[buffer(4)]],
    device float* ssm_state [[buffer(5)]],
    device float* y [[buffer(6)]],
    constant uint4& dims [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]
) {
    const uint value_heads = dims.x;
    const uint value_head_dim = dims.y;
    const uint key_head_dim = dims.z;
    const uint repeat = dims.w;
    const uint lane = tid.x;
    const uint value_col = gid.y;
    const uint value_head = gid.z;
    if (value_head >= value_heads || value_col >= value_head_dim || repeat == 0u || lane >= 32u || key_head_dim != 128u) {
        return;
    }

    const uint key_heads = value_heads / repeat;
    const uint key_dim = key_heads * key_head_dim;
    const uint key_head = value_head / repeat;
    const uint key_base = key_head * key_head_dim;
    const uint value_index = value_head * value_head_dim + value_col;
    const uint state_base = value_index * key_head_dim;
    const float d = decay[value_head];

    const uint idx0 = state_base + lane;
    const uint idx1 = idx0 + 32u;
    const uint idx2 = idx0 + 64u;
    const uint idx3 = idx0 + 96u;
    const uint key0 = key_base + lane;
    const uint key1 = key0 + 32u;
    const uint key2 = key0 + 64u;
    const uint key3 = key0 + 96u;

    float state0 = ssm_state[idx0] * d;
    float state1 = ssm_state[idx1] * d;
    float state2 = ssm_state[idx2] * d;
    float state3 = ssm_state[idx3] * d;

    float kv_part = 0.0f;
    kv_part += state0 * k_norm[key0];
    kv_part += state1 * k_norm[key1];
    kv_part += state2 * k_norm[key2];
    kv_part += state3 * k_norm[key3];
    const float kv_mem = simd_sum(kv_part);
    const float v = conv_out[key_dim * 2u + value_index];
    const float delta = (v - kv_mem) * beta[value_head];

    state0 += delta * k_norm[key0];
    state1 += delta * k_norm[key1];
    state2 += delta * k_norm[key2];
    state3 += delta * k_norm[key3];

    float y_part = 0.0f;
    y_part += state0 * q_norm[key0];
    y_part += state1 * q_norm[key1];
    y_part += state2 * q_norm[key2];
    y_part += state3 * q_norm[key3];
    const float out = simd_sum(y_part);
    if (lane == 0u) {
        y[value_index] = out;
    }

    ssm_state[idx0] = state0;
    ssm_state[idx1] = state1;
    ssm_state[idx2] = state2;
    ssm_state[idx3] = state3;
}

kernel void linear_attn_gated_delta_seq_dk128_tg4_f32(
    device const float* conv_out [[buffer(0)]],
    device const float* q_norm [[buffer(1)]],
    device const float* k_norm [[buffer(2)]],
    device const float* beta [[buffer(3)]],
    device const float* decay [[buffer(4)]],
    device float* ssm_state [[buffer(5)]],
    device float* y [[buffer(6)]],
    constant uint4& dims [[buffer(7)]],
    constant uint& steps [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]
) {
    const uint value_heads = dims.x;
    const uint value_head_dim = dims.y;
    const uint key_head_dim = dims.z;
    const uint repeat = dims.w;
    const uint lane = tid.x;
    const uint value_col = gid.y;
    const uint value_head = gid.z;
    if (value_head >= value_heads || value_col >= value_head_dim || repeat == 0u || lane >= 32u || key_head_dim != 128u) {
        return;
    }

    const uint key_heads = value_heads / repeat;
    const uint key_dim = key_heads * 128u;
    const uint value_dim = value_heads * value_head_dim;
    const uint conv_dim = (2u * key_dim) + value_dim;
    const uint key_head = value_head / repeat;
    const uint key_base = key_head * 128u;
    const uint value_index = value_head * value_head_dim + value_col;
    const uint state_base = value_index * 128u;

    const uint idx0 = state_base + lane;
    const uint idx1 = idx0 + 32u;
    const uint idx2 = idx0 + 64u;
    const uint idx3 = idx0 + 96u;
    const uint key0 = key_base + lane;
    const uint key1 = key0 + 32u;
    const uint key2 = key0 + 64u;
    const uint key3 = key0 + 96u;

    float state0 = ssm_state[idx0];
    float state1 = ssm_state[idx1];
    float state2 = ssm_state[idx2];
    float state3 = ssm_state[idx3];

    for (uint t = 0u; t < steps; ++t) {
        const uint key_offset = t * key_dim;
        const uint value_offset = t * value_dim;
        const uint conv_offset = t * conv_dim;
        const uint gate_offset = t * value_heads;
        const float d = decay[gate_offset + value_head];
        const float beta_v = beta[gate_offset + value_head];

        state0 *= d;
        state1 *= d;
        state2 *= d;
        state3 *= d;

        const float k0 = k_norm[key_offset + key0];
        const float k1 = k_norm[key_offset + key1];
        const float k2 = k_norm[key_offset + key2];
        const float k3 = k_norm[key_offset + key3];
        float kv_part = 0.0f;
        kv_part += state0 * k0;
        kv_part += state1 * k1;
        kv_part += state2 * k2;
        kv_part += state3 * k3;
        const float kv_mem = simd_sum(kv_part);
        const float v = conv_out[conv_offset + (2u * key_dim) + value_index];
        const float delta = (v - kv_mem) * beta_v;

        state0 += delta * k0;
        state1 += delta * k1;
        state2 += delta * k2;
        state3 += delta * k3;

        const float q0 = q_norm[key_offset + key0];
        const float q1 = q_norm[key_offset + key1];
        const float q2 = q_norm[key_offset + key2];
        const float q3 = q_norm[key_offset + key3];
        float y_part = 0.0f;
        y_part += state0 * q0;
        y_part += state1 * q1;
        y_part += state2 * q2;
        y_part += state3 * q3;
        const float out = simd_sum(y_part);
        if (lane == 0u) {
            y[value_offset + value_index] = out;
        }
    }

    ssm_state[idx0] = state0;
    ssm_state[idx1] = state1;
    ssm_state[idx2] = state2;
    ssm_state[idx3] = state3;
}

kernel void linear_attn_gated_delta_seq_dk128_bf16_tg4_f32(
    device const float* conv_out [[buffer(0)]],
    device const float* q_norm [[buffer(1)]],
    device const float* k_norm [[buffer(2)]],
    device const float* beta [[buffer(3)]],
    device const float* decay [[buffer(4)]],
    device bfloat* ssm_state [[buffer(5)]],
    device float* y [[buffer(6)]],
    constant uint4& dims [[buffer(7)]],
    constant uint& steps [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]
) {
    const uint value_heads = dims.x;
    const uint value_head_dim = dims.y;
    const uint key_head_dim = dims.z;
    const uint repeat = dims.w;
    const uint lane = tid.x;
    const uint value_col = gid.y;
    const uint value_head = gid.z;
    if (value_head >= value_heads || value_col >= value_head_dim || repeat == 0u || lane >= 32u || key_head_dim != 128u) {
        return;
    }

    const uint key_heads = value_heads / repeat;
    const uint key_dim = key_heads * 128u;
    const uint value_dim = value_heads * value_head_dim;
    const uint conv_dim = (2u * key_dim) + value_dim;
    const uint key_head = value_head / repeat;
    const uint key_base = key_head * 128u;
    const uint value_index = value_head * value_head_dim + value_col;
    const uint state_base = value_index * 128u;

    const uint idx0 = state_base + lane;
    const uint idx1 = idx0 + 32u;
    const uint idx2 = idx0 + 64u;
    const uint idx3 = idx0 + 96u;
    const uint key0 = key_base + lane;
    const uint key1 = key0 + 32u;
    const uint key2 = key0 + 64u;
    const uint key3 = key0 + 96u;

    float state0 = float(ssm_state[idx0]);
    float state1 = float(ssm_state[idx1]);
    float state2 = float(ssm_state[idx2]);
    float state3 = float(ssm_state[idx3]);

    for (uint t = 0u; t < steps; ++t) {
        const uint key_offset = t * key_dim;
        const uint value_offset = t * value_dim;
        const uint conv_offset = t * conv_dim;
        const uint gate_offset = t * value_heads;
        const float d = decay[gate_offset + value_head];
        const float beta_v = beta[gate_offset + value_head];

        state0 *= d;
        state1 *= d;
        state2 *= d;
        state3 *= d;

        const float k0 = k_norm[key_offset + key0];
        const float k1 = k_norm[key_offset + key1];
        const float k2 = k_norm[key_offset + key2];
        const float k3 = k_norm[key_offset + key3];
        float kv_part = 0.0f;
        kv_part += state0 * k0;
        kv_part += state1 * k1;
        kv_part += state2 * k2;
        kv_part += state3 * k3;
        const float kv_mem = simd_sum(kv_part);
        const float v = conv_out[conv_offset + (2u * key_dim) + value_index];
        const float delta = (v - kv_mem) * beta_v;

        state0 += delta * k0;
        state1 += delta * k1;
        state2 += delta * k2;
        state3 += delta * k3;

        const float q0 = q_norm[key_offset + key0];
        const float q1 = q_norm[key_offset + key1];
        const float q2 = q_norm[key_offset + key2];
        const float q3 = q_norm[key_offset + key3];
        float y_part = 0.0f;
        y_part += state0 * q0;
        y_part += state1 * q1;
        y_part += state2 * q2;
        y_part += state3 * q3;
        const float out = simd_sum(y_part);
        if (lane == 0u) {
            y[value_offset + value_index] = out;
        }
    }

    ssm_state[idx0] = bfloat(state0);
    ssm_state[idx1] = bfloat(state1);
    ssm_state[idx2] = bfloat(state2);
    ssm_state[idx3] = bfloat(state3);
}

kernel void linear_attn_gated_delta_dk128_bf16_tg4_f32(
    device const float* conv_out [[buffer(0)]],
    device const float* q_norm [[buffer(1)]],
    device const float* k_norm [[buffer(2)]],
    device const float* beta [[buffer(3)]],
    device const float* decay [[buffer(4)]],
    device bfloat* ssm_state [[buffer(5)]],
    device float* y [[buffer(6)]],
    constant uint4& dims [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]
) {
    const uint value_heads = dims.x;
    const uint value_head_dim = dims.y;
    const uint key_head_dim = dims.z;
    const uint repeat = dims.w;
    const uint lane = tid.x;
    const uint value_col = gid.y;
    const uint value_head = gid.z;
    if (value_head >= value_heads || value_col >= value_head_dim || repeat == 0u || lane >= 32u || key_head_dim != 128u) {
        return;
    }

    const uint key_heads = value_heads / repeat;
    const uint key_dim = key_heads * key_head_dim;
    const uint key_head = value_head / repeat;
    const uint key_base = key_head * key_head_dim;
    const uint value_index = value_head * value_head_dim + value_col;
    const uint state_base = value_index * key_head_dim;
    const float d = decay[value_head];

    const uint idx0 = state_base + lane;
    const uint idx1 = idx0 + 32u;
    const uint idx2 = idx0 + 64u;
    const uint idx3 = idx0 + 96u;
    const uint key0 = key_base + lane;
    const uint key1 = key0 + 32u;
    const uint key2 = key0 + 64u;
    const uint key3 = key0 + 96u;

    float state0 = float(ssm_state[idx0]) * d;
    float state1 = float(ssm_state[idx1]) * d;
    float state2 = float(ssm_state[idx2]) * d;
    float state3 = float(ssm_state[idx3]) * d;

    float kv_part = 0.0f;
    kv_part += state0 * k_norm[key0];
    kv_part += state1 * k_norm[key1];
    kv_part += state2 * k_norm[key2];
    kv_part += state3 * k_norm[key3];
    const float kv_mem = simd_sum(kv_part);
    const float v = conv_out[key_dim * 2u + value_index];
    const float delta = (v - kv_mem) * beta[value_head];

    state0 += delta * k_norm[key0];
    state1 += delta * k_norm[key1];
    state2 += delta * k_norm[key2];
    state3 += delta * k_norm[key3];

    float y_part = 0.0f;
    y_part += state0 * q_norm[key0];
    y_part += state1 * q_norm[key1];
    y_part += state2 * q_norm[key2];
    y_part += state3 * q_norm[key3];
    const float out = simd_sum(y_part);
    if (lane == 0u) {
        y[value_index] = out;
    }

    ssm_state[idx0] = bfloat(state0);
    ssm_state[idx1] = bfloat(state1);
    ssm_state[idx2] = bfloat(state2);
    ssm_state[idx3] = bfloat(state3);
}

kernel void linear_attn_gated_delta_inv_dk128_tg4_f32(
    device const float* conv_out [[buffer(0)]],
    device const float* q_inv [[buffer(1)]],
    device const float* k_inv [[buffer(2)]],
    device const float* beta [[buffer(3)]],
    device const float* decay [[buffer(4)]],
    device float* ssm_state [[buffer(5)]],
    device float* y [[buffer(6)]],
    constant uint4& dims [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]
) {
    const uint value_heads = dims.x;
    const uint value_head_dim = dims.y;
    const uint key_head_dim = dims.z;
    const uint repeat = dims.w;
    const uint lane = tid.x;
    const uint value_col = gid.y;
    const uint value_head = gid.z;
    if (value_head >= value_heads || value_col >= value_head_dim || repeat == 0u || lane >= 32u || key_head_dim != 128u) {
        return;
    }

    const uint key_heads = value_heads / repeat;
    const uint key_dim = key_heads * 128u;
    const uint key_head = value_head / repeat;
    const uint key_base = key_head * 128u;
    const uint value_index = value_head * value_head_dim + value_col;
    const uint state_base = value_index * 128u;
    const float d = decay[value_head];
    const float q_scale = q_inv[key_head];
    const float k_scale = k_inv[key_head];

    const uint idx0 = state_base + lane;
    const uint idx1 = idx0 + 32u;
    const uint idx2 = idx0 + 64u;
    const uint idx3 = idx0 + 96u;
    const uint key0 = key_base + lane;
    const uint key1 = key0 + 32u;
    const uint key2 = key0 + 64u;
    const uint key3 = key0 + 96u;

    const float k0 = conv_out[key_dim + key0] * k_scale;
    const float k1 = conv_out[key_dim + key1] * k_scale;
    const float k2 = conv_out[key_dim + key2] * k_scale;
    const float k3 = conv_out[key_dim + key3] * k_scale;

    float state0 = ssm_state[idx0] * d;
    float state1 = ssm_state[idx1] * d;
    float state2 = ssm_state[idx2] * d;
    float state3 = ssm_state[idx3] * d;

    float kv_part = 0.0f;
    kv_part += state0 * k0;
    kv_part += state1 * k1;
    kv_part += state2 * k2;
    kv_part += state3 * k3;
    const float kv_mem = simd_sum(kv_part);
    const float v = conv_out[key_dim * 2u + value_index];
    const float delta = (v - kv_mem) * beta[value_head];

    state0 += delta * k0;
    state1 += delta * k1;
    state2 += delta * k2;
    state3 += delta * k3;

    const float q0 = conv_out[key0] * q_scale;
    const float q1 = conv_out[key1] * q_scale;
    const float q2 = conv_out[key2] * q_scale;
    const float q3 = conv_out[key3] * q_scale;

    float y_part = 0.0f;
    y_part += state0 * q0;
    y_part += state1 * q1;
    y_part += state2 * q2;
    y_part += state3 * q3;
    const float out = simd_sum(y_part);
    if (lane == 0u) {
        y[value_index] = out;
    }

    ssm_state[idx0] = state0;
    ssm_state[idx1] = state1;
    ssm_state[idx2] = state2;
    ssm_state[idx3] = state3;
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

kernel void linear_attn_rms_gate_dv128_f32(
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
    if (value_head >= value_heads || value_head_dim != 128u) {
        return;
    }

    const uint base = value_head * 128u;
    const uint idx0 = base + lane;
    const uint idx1 = idx0 + 32u;
    const uint idx2 = idx0 + 64u;
    const uint idx3 = idx0 + 96u;
    const float y0 = y[idx0];
    const float y1 = y[idx1];
    const float y2 = y[idx2];
    const float y3 = y[idx3];
    float ss = 0.0f;
    ss += y0 * y0;
    ss += y1 * y1;
    ss += y2 * y2;
    ss += y3 * y3;
    const float mean = simd_sum(ss) / float(value_head_dim);
    const float inv = rsqrt(mean + eps);

    const float z0 = z[idx0];
    const float z1 = z[idx1];
    const float z2 = z[idx2];
    const float z3 = z[idx3];
    gated[idx0] = y0 * inv * norm_weight[lane] * (z0 / (1.0f + exp(-z0)));
    gated[idx1] = y1 * inv * norm_weight[lane + 32u] * (z1 / (1.0f + exp(-z1)));
    gated[idx2] = y2 * inv * norm_weight[lane + 64u] * (z2 / (1.0f + exp(-z2)));
    gated[idx3] = y3 * inv * norm_weight[lane + 96u] * (z3 / (1.0f + exp(-z3)));
}

// Brick #8 prefill : variante BATCHÉE (grid.y = token) du rms-gate dv128. Élimine la
// boucle per-token du prefill linéaire (1 dispatch au lieu de `batch`). Élément-par-
// token indépendant → sortie byte-identique au per-token. value_head_dim=128 requis.
kernel void linear_attn_rms_gate_batch_dv128_f32(
    device const float* y [[buffer(0)]],
    device const float* z [[buffer(1)]],
    device const float* norm_weight [[buffer(2)]],
    device float* gated [[buffer(3)]],
    constant uint3& dims [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint lane [[thread_index_in_threadgroup]],
    uint2 tg [[threadgroup_position_in_grid]]
) {
    const uint value_heads = dims.x;
    const uint value_head_dim = dims.y;
    const uint batch = dims.z;
    const uint value_head = tg.x;
    const uint token = tg.y;
    if (value_head >= value_heads || token >= batch || value_head_dim != 128u) {
        return;
    }
    const uint base = token * value_heads * 128u + value_head * 128u;
    const uint idx0 = base + lane;
    const uint idx1 = idx0 + 32u;
    const uint idx2 = idx0 + 64u;
    const uint idx3 = idx0 + 96u;
    const float y0 = y[idx0];
    const float y1 = y[idx1];
    const float y2 = y[idx2];
    const float y3 = y[idx3];
    float ss = 0.0f;
    ss += y0 * y0;
    ss += y1 * y1;
    ss += y2 * y2;
    ss += y3 * y3;
    const float mean = simd_sum(ss) / float(value_head_dim);
    const float inv = rsqrt(mean + eps);
    const float z0 = z[idx0];
    const float z1 = z[idx1];
    const float z2 = z[idx2];
    const float z3 = z[idx3];
    gated[idx0] = y0 * inv * norm_weight[lane] * (z0 / (1.0f + exp(-z0)));
    gated[idx1] = y1 * inv * norm_weight[lane + 32u] * (z1 / (1.0f + exp(-z1)));
    gated[idx2] = y2 * inv * norm_weight[lane + 64u] * (z2 / (1.0f + exp(-z2)));
    gated[idx3] = y3 * inv * norm_weight[lane + 96u] * (z3 / (1.0f + exp(-z3)));
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
    const uint results_per_simdgroup = 4u;
    const uint row_base = tile.y * 8u + simd_gid * results_per_simdgroup;
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

kernel void affine_gather_qmv_fast_u8_gs64_f32(
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
    const uint results_per_simdgroup = 4u;
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

    const uint values_per_word = 4u;
    const uint words_per_thread = 4u;
    const uint values_per_thread = values_per_word * words_per_thread;
    const uint values_per_block = values_per_thread * 32u;
    const uint words_per_block = values_per_block / values_per_word;
    const device uint* ws = packed + ((expert * out_dim + row_base) * packed_cols) + simd_lid;
    const device bfloat* scale_base = scales + ((expert * out_dim + row_base) * groups);
    const device bfloat* bias_base = biases + ((expert * out_dim + row_base) * groups);
    const device float* x = lhs + lhs_row * in_dim + simd_lid * values_per_word;

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
            if (row_base + row < out_dim) {
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
        }

        ws += words_per_block;
        scale_base += values_per_block / 64u;
        bias_base += values_per_block / 64u;
        x += values_per_block;
    }

    for (uint row = 0u; row < results_per_simdgroup; ++row) {
        const float reduced = simd_sum(result[row]);
        if (simd_lid == 0u && row_base + row < out_dim) {
            out[(slot * out_dim) + row_base + row] = reduced;
        }
    }
}

kernel void affine_gather_qmv_fast_u8_gs128_f32(
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
    const uint results_per_simdgroup = 4u;
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

    const uint values_per_word = 4u;
    const uint words_per_thread = 4u;
    const uint values_per_thread = values_per_word * words_per_thread;
    const uint values_per_block = values_per_thread * 32u;
    const uint words_per_block = values_per_block / values_per_word;
    const device uint* ws = packed + ((expert * out_dim + row_base) * packed_cols) + simd_lid;
    const device bfloat* scale_base = scales + ((expert * out_dim + row_base) * groups);
    const device bfloat* bias_base = biases + ((expert * out_dim + row_base) * groups);
    const device float* x = lhs + lhs_row * in_dim + simd_lid * values_per_word;

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
            if (row_base + row < out_dim) {
                const device uint* row_words = ws + row * packed_cols;
                float accum = 0.0f;
                for (uint word = 0u; word < words_per_thread; ++word) {
                    const uint packed_word = row_words[word * 32u];
                    const uint group = (simd_lid + word * 32u) / 32u;
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
        }

        ws += words_per_block;
        scale_base += values_per_block / 128u;
        bias_base += values_per_block / 128u;
        x += values_per_block;
    }

    for (uint row = 0u; row < results_per_simdgroup; ++row) {
        const float reduced = simd_sum(result[row]);
        if (simd_lid == 0u && row_base + row < out_dim) {
            out[(slot * out_dim) + row_base + row] = reduced;
        }
    }
}

kernel void affine_gather_qmv_fast_u8_gs64_tg128_f32(
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
    const uint results_per_simdgroup = 4u;
    const uint row_base = tile.y * 16u + simd_gid * 4u;
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

    const uint values_per_word = 4u;
    const uint words_per_thread = 4u;
    const uint values_per_thread = values_per_word * words_per_thread;
    const uint values_per_block = values_per_thread * 32u;
    const uint words_per_block = values_per_block / values_per_word;
    const device uint* ws = packed + ((expert * out_dim + row_base) * packed_cols) + simd_lid;
    const device bfloat* scale_base = scales + ((expert * out_dim + row_base) * groups);
    const device bfloat* bias_base = biases + ((expert * out_dim + row_base) * groups);
    const device float* x = lhs + lhs_row * in_dim + simd_lid * values_per_word;

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
            if (row_base + row < out_dim) {
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
        }

        ws += words_per_block;
        scale_base += values_per_block / 64u;
        bias_base += values_per_block / 64u;
        x += values_per_block;
    }

    for (uint row = 0u; row < results_per_simdgroup; ++row) {
        const float reduced = simd_sum(result[row]);
        if (simd_lid == 0u && row_base + row < out_dim) {
            out[(slot * out_dim) + row_base + row] = reduced;
        }
    }
}

kernel void affine_gather_qmv_fast_u8_gs128_tg128_f32(
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
    const uint results_per_simdgroup = 4u;
    const uint row_base = tile.y * 16u + simd_gid * 4u;
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

    const uint values_per_word = 4u;
    const uint words_per_thread = 4u;
    const uint values_per_thread = values_per_word * words_per_thread;
    const uint values_per_block = values_per_thread * 32u;
    const uint words_per_block = values_per_block / values_per_word;
    const device uint* ws = packed + ((expert * out_dim + row_base) * packed_cols) + simd_lid;
    const device bfloat* scale_base = scales + ((expert * out_dim + row_base) * groups);
    const device bfloat* bias_base = biases + ((expert * out_dim + row_base) * groups);
    const device float* x = lhs + lhs_row * in_dim + simd_lid * values_per_word;

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
            if (row_base + row < out_dim) {
                const device uint* row_words = ws + row * packed_cols;
                float accum = 0.0f;
                for (uint word = 0u; word < words_per_thread; ++word) {
                    const uint packed_word = row_words[word * 32u];
                    const uint group = (simd_lid + word * 32u) / 32u;
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
        }

        ws += words_per_block;
        scale_base += values_per_block / 128u;
        bias_base += values_per_block / 128u;
        x += values_per_block;
    }

    for (uint row = 0u; row < results_per_simdgroup; ++row) {
        const float reduced = simd_sum(result[row]);
        if (simd_lid == 0u && row_base + row < out_dim) {
            out[(slot * out_dim) + row_base + row] = reduced;
        }
    }
}

kernel void affine_gather_qmv_fast_u8_gs64_tg256_f32(
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
    const uint results_per_simdgroup = 4u;
    const uint row_base = tile.y * 32u + simd_gid * 4u;
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

    const uint values_per_word = 4u;
    const uint words_per_thread = 4u;
    const uint values_per_thread = values_per_word * words_per_thread;
    const uint values_per_block = values_per_thread * 32u;
    const uint words_per_block = values_per_block / values_per_word;
    const device uint* ws = packed + ((expert * out_dim + row_base) * packed_cols) + simd_lid;
    const device bfloat* scale_base = scales + ((expert * out_dim + row_base) * groups);
    const device bfloat* bias_base = biases + ((expert * out_dim + row_base) * groups);
    const device float* x = lhs + lhs_row * in_dim + simd_lid * values_per_word;

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
            if (row_base + row < out_dim) {
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
        }

        ws += words_per_block;
        scale_base += values_per_block / 64u;
        bias_base += values_per_block / 64u;
        x += values_per_block;
    }

    for (uint row = 0u; row < results_per_simdgroup; ++row) {
        const float reduced = simd_sum(result[row]);
        if (simd_lid == 0u && row_base + row < out_dim) {
            out[(slot * out_dim) + row_base + row] = reduced;
        }
    }
}

kernel void affine_gather_qmv_fast_u8_gs128_tg256_f32(
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
    const uint results_per_simdgroup = 4u;
    const uint row_base = tile.y * 32u + simd_gid * 4u;
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

    const uint values_per_word = 4u;
    const uint words_per_thread = 4u;
    const uint values_per_thread = values_per_word * words_per_thread;
    const uint values_per_block = values_per_thread * 32u;
    const uint words_per_block = values_per_block / values_per_word;
    const device uint* ws = packed + ((expert * out_dim + row_base) * packed_cols) + simd_lid;
    const device bfloat* scale_base = scales + ((expert * out_dim + row_base) * groups);
    const device bfloat* bias_base = biases + ((expert * out_dim + row_base) * groups);
    const device float* x = lhs + lhs_row * in_dim + simd_lid * values_per_word;

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
            if (row_base + row < out_dim) {
                const device uint* row_words = ws + row * packed_cols;
                float accum = 0.0f;
                for (uint word = 0u; word < words_per_thread; ++word) {
                    const uint packed_word = row_words[word * 32u];
                    const uint group = (simd_lid + word * 32u) / 32u;
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
        }

        ws += words_per_block;
        scale_base += values_per_block / 128u;
        bias_base += values_per_block / 128u;
        x += values_per_block;
    }

    for (uint row = 0u; row < results_per_simdgroup; ++row) {
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
    const uint results_per_simdgroup = 4u;
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

    for (uint row = 0u; row < results_per_simdgroup; ++row) {
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
    constant uint& activation [[buffer(11)]],
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
    const uint results_per_simdgroup = 4u;
    const uint row_base = tile.y * 8u + simd_gid * results_per_simdgroup;
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
            float activated;
            if (activation == 1u) {
                const float inner = 0.7978846f * (gate + 0.044715f * gate * gate * gate);
                activated = 0.5f * gate *
                    (1.0f + tanh(clamp(inner, -20.0f, 20.0f)));
            } else {
                activated = gate / (1.0f + exp(-gate));
            }
            out[(slot * out_dim) + row_base + row] = activated * up;
        }
    }
}

kernel void affine_gather_gate_up_swiglu_fast_u8_gs64_f32(
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
    const uint results_per_simdgroup = 4u;
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

    const uint values_per_word = 4u;
    const uint words_per_thread = 4u;
    const uint values_per_thread = values_per_word * words_per_thread;
    const uint values_per_block = values_per_thread * 32u;
    const uint words_per_block = values_per_block / values_per_word;
    const device uint* gate_ws =
        gate_packed + ((expert * out_dim + row_base) * packed_cols) + simd_lid;
    const device uint* up_ws =
        up_packed + ((expert * out_dim + row_base) * packed_cols) + simd_lid;
    const device bfloat* gate_scale_base =
        gate_scales + ((expert * out_dim + row_base) * groups);
    const device bfloat* gate_bias_base =
        gate_biases + ((expert * out_dim + row_base) * groups);
    const device bfloat* up_scale_base =
        up_scales + ((expert * out_dim + row_base) * groups);
    const device bfloat* up_bias_base =
        up_biases + ((expert * out_dim + row_base) * groups);
    const device float* x = lhs + lhs_row * in_dim + simd_lid * values_per_word;

    float gate_result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float up_result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
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
            if (row_base + row < out_dim) {
                const device uint* gate_row_words = gate_ws + row * packed_cols;
                const device uint* up_row_words = up_ws + row * packed_cols;
                float gate_accum = 0.0f;
                float up_accum = 0.0f;
                for (uint word = 0u; word < words_per_thread; ++word) {
                    const uint gate_word = gate_row_words[word * 32u];
                    const uint up_word = up_row_words[word * 32u];
                    const uint group = (simd_lid + word * 32u) / 16u;
                    const float gate_scale = gate_scale_base[row * groups + group];
                    const float gate_bias = gate_bias_base[row * groups + group];
                    const float up_scale = up_scale_base[row * groups + group];
                    const float up_bias = up_bias_base[row * groups + group];
                    const uint base = word * values_per_word;
                    gate_accum += xt[base] * ((float(gate_word & 0x000000ffu) * gate_scale) + gate_bias);
                    gate_accum += xt[base + 1u] * ((float((gate_word >> 8u) & 0x000000ffu) * gate_scale) + gate_bias);
                    gate_accum += xt[base + 2u] * ((float((gate_word >> 16u) & 0x000000ffu) * gate_scale) + gate_bias);
                    gate_accum += xt[base + 3u] * ((float((gate_word >> 24u) & 0x000000ffu) * gate_scale) + gate_bias);
                    up_accum += xt[base] * ((float(up_word & 0x000000ffu) * up_scale) + up_bias);
                    up_accum += xt[base + 1u] * ((float((up_word >> 8u) & 0x000000ffu) * up_scale) + up_bias);
                    up_accum += xt[base + 2u] * ((float((up_word >> 16u) & 0x000000ffu) * up_scale) + up_bias);
                    up_accum += xt[base + 3u] * ((float((up_word >> 24u) & 0x000000ffu) * up_scale) + up_bias);
                }
                gate_result[row] += gate_accum;
                up_result[row] += up_accum;
            }
        }

        gate_ws += words_per_block;
        up_ws += words_per_block;
        gate_scale_base += values_per_block / 64u;
        gate_bias_base += values_per_block / 64u;
        up_scale_base += values_per_block / 64u;
        up_bias_base += values_per_block / 64u;
        x += values_per_block;
    }

    for (uint row = 0u; row < results_per_simdgroup; ++row) {
        const float gate = simd_sum(gate_result[row]);
        const float up = simd_sum(up_result[row]);
        if (simd_lid == 0u && row_base + row < out_dim) {
            out[(slot * out_dim) + row_base + row] = (gate / (1.0f + exp(-gate))) * up;
        }
    }
}

kernel void affine_gather_gate_up_swiglu_fast_u8_gs128_f32(
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

    const uint results_per_simdgroup = 4u;
    const uint values_per_word = 4u;
    const uint words_per_thread = 4u;
    const uint values_per_thread = values_per_word * words_per_thread;
    const uint values_per_block = values_per_thread * 32u;
    const uint words_per_block = values_per_block / values_per_word;
    const device uint* gate_ws =
        gate_packed + ((expert * out_dim + row_base) * packed_cols) + simd_lid;
    const device uint* up_ws =
        up_packed + ((expert * out_dim + row_base) * packed_cols) + simd_lid;
    const device bfloat* gate_scale_base =
        gate_scales + ((expert * out_dim + row_base) * groups);
    const device bfloat* gate_bias_base =
        gate_biases + ((expert * out_dim + row_base) * groups);
    const device bfloat* up_scale_base =
        up_scales + ((expert * out_dim + row_base) * groups);
    const device bfloat* up_bias_base =
        up_biases + ((expert * out_dim + row_base) * groups);
    const device float* x = lhs + lhs_row * in_dim + simd_lid * values_per_word;

    float gate_result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float up_result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
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
            if (row_base + row < out_dim) {
                const device uint* gate_row_words = gate_ws + row * packed_cols;
                const device uint* up_row_words = up_ws + row * packed_cols;
                float gate_accum = 0.0f;
                float up_accum = 0.0f;
                for (uint word = 0u; word < words_per_thread; ++word) {
                    const uint gate_word = gate_row_words[word * 32u];
                    const uint up_word = up_row_words[word * 32u];
                    const uint group = (simd_lid + word * 32u) / 32u;
                    const float gate_scale = gate_scale_base[row * groups + group];
                    const float gate_bias = gate_bias_base[row * groups + group];
                    const float up_scale = up_scale_base[row * groups + group];
                    const float up_bias = up_bias_base[row * groups + group];
                    const uint base = word * values_per_word;
                    gate_accum += xt[base] * ((float(gate_word & 0x000000ffu) * gate_scale) + gate_bias);
                    gate_accum += xt[base + 1u] * ((float((gate_word >> 8u) & 0x000000ffu) * gate_scale) + gate_bias);
                    gate_accum += xt[base + 2u] * ((float((gate_word >> 16u) & 0x000000ffu) * gate_scale) + gate_bias);
                    gate_accum += xt[base + 3u] * ((float((gate_word >> 24u) & 0x000000ffu) * gate_scale) + gate_bias);
                    up_accum += xt[base] * ((float(up_word & 0x000000ffu) * up_scale) + up_bias);
                    up_accum += xt[base + 1u] * ((float((up_word >> 8u) & 0x000000ffu) * up_scale) + up_bias);
                    up_accum += xt[base + 2u] * ((float((up_word >> 16u) & 0x000000ffu) * up_scale) + up_bias);
                    up_accum += xt[base + 3u] * ((float((up_word >> 24u) & 0x000000ffu) * up_scale) + up_bias);
                }
                gate_result[row] += gate_accum;
                up_result[row] += up_accum;
            }
        }

        gate_ws += words_per_block;
        up_ws += words_per_block;
        gate_scale_base += values_per_block / 128u;
        gate_bias_base += values_per_block / 128u;
        up_scale_base += values_per_block / 128u;
        up_bias_base += values_per_block / 128u;
        x += values_per_block;
    }

    for (uint row = 0u; row < results_per_simdgroup; ++row) {
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

kernel void affine_gate_up_swiglu_fast_u8_gs64_f32(
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
    const uint values_per_word = 4u;
    const uint words_per_thread = 4u;
    const uint values_per_thread = values_per_word * words_per_thread;
    const uint values_per_block = values_per_thread * 32u;
    const uint words_per_block = values_per_block / values_per_word;
    const uint row_base = tile.y * (simdgroups * results_per_simdgroup) +
        simd_gid * results_per_simdgroup;

    const device uint* gate_ws = gate_packed + row_base * packed_cols + simd_lid;
    const device uint* up_ws = up_packed + row_base * packed_cols + simd_lid;
    const device bfloat* gate_scale_base = gate_scales + row_base * groups;
    const device bfloat* gate_bias_base = gate_biases + row_base * groups;
    const device bfloat* up_scale_base = up_scales + row_base * groups;
    const device bfloat* up_bias_base = up_biases + row_base * groups;
    const device float* x = lhs + tile.x * in_dim + simd_lid * values_per_word;

    float gate_result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float up_result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
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
            const device uint* gate_row_words = gate_ws + row * packed_cols;
            const device uint* up_row_words = up_ws + row * packed_cols;
            float gate_accum = 0.0f;
            float up_accum = 0.0f;
            for (uint word = 0u; word < words_per_thread; ++word) {
                const uint gate_word = gate_row_words[word * 32u];
                const uint up_word = up_row_words[word * 32u];
                const uint group = (simd_lid + word * 32u) / 16u;
                const float gate_scale = gate_scale_base[row * groups + group];
                const float gate_bias = gate_bias_base[row * groups + group];
                const float up_scale = up_scale_base[row * groups + group];
                const float up_bias = up_bias_base[row * groups + group];
                const uint base = word * values_per_word;
                gate_accum += xt[base] * ((float(gate_word & 0x000000ffu) * gate_scale) + gate_bias);
                gate_accum += xt[base + 1u] * ((float((gate_word >> 8u) & 0x000000ffu) * gate_scale) + gate_bias);
                gate_accum += xt[base + 2u] * ((float((gate_word >> 16u) & 0x000000ffu) * gate_scale) + gate_bias);
                gate_accum += xt[base + 3u] * ((float((gate_word >> 24u) & 0x000000ffu) * gate_scale) + gate_bias);
                up_accum += xt[base] * ((float(up_word & 0x000000ffu) * up_scale) + up_bias);
                up_accum += xt[base + 1u] * ((float((up_word >> 8u) & 0x000000ffu) * up_scale) + up_bias);
                up_accum += xt[base + 2u] * ((float((up_word >> 16u) & 0x000000ffu) * up_scale) + up_bias);
                up_accum += xt[base + 3u] * ((float((up_word >> 24u) & 0x000000ffu) * up_scale) + up_bias);
            }
            gate_result[row] += gate_accum;
            up_result[row] += up_accum;
        }

        gate_ws += words_per_block;
        up_ws += words_per_block;
        gate_scale_base += values_per_block / 64u;
        gate_bias_base += values_per_block / 64u;
        up_scale_base += values_per_block / 64u;
        up_bias_base += values_per_block / 64u;
        x += values_per_block;
    }

    for (uint row = 0u; row < results_per_simdgroup; ++row) {
        const float gate = simd_sum(gate_result[row]);
        const float up = simd_sum(up_result[row]);
        if (simd_lid == 0u) {
            out[tile.x * out_dim + row_base + row] = (gate / (1.0f + exp(-gate))) * up;
        }
    }
}

kernel void affine_gate_up_swiglu_gate_fast_u8_gs64_f32(
    device const float* lhs [[buffer(0)]],
    device const uint* gate_packed [[buffer(1)]],
    device const bfloat* gate_scales [[buffer(2)]],
    device const bfloat* gate_biases [[buffer(3)]],
    device const uint* up_packed [[buffer(4)]],
    device const bfloat* up_scales [[buffer(5)]],
    device const bfloat* up_biases [[buffer(6)]],
    device const uint* shared_gate_packed [[buffer(7)]],
    device const bfloat* shared_gate_scales [[buffer(8)]],
    device const bfloat* shared_gate_biases [[buffer(9)]],
    device float* out [[buffer(10)]],
    device float* shared_gate_out [[buffer(11)]],
    constant uint4& dims [[buffer(12)]],
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
    const uint out_groups = (out_dim + 7u) / 8u;

    if (tile.y == out_groups) {
        if (simd_gid != 0u) {
            return;
        }
        const device uint* ws = shared_gate_packed + simd_lid;
        const device bfloat* scale_base = shared_gate_scales;
        const device bfloat* bias_base = shared_gate_biases;
        const device float* x = lhs + simd_lid * values_per_word;

        float result = 0.0f;
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

            float accum = 0.0f;
            for (uint word = 0u; word < words_per_thread; ++word) {
                const uint packed_word = ws[word * 32u];
                const uint group = (simd_lid + word * 32u) / 16u;
                const float scale = scale_base[group];
                const float bias = bias_base[group];
                const uint base = word * values_per_word;
                    accum += xt[base] * ((float(packed_word & 0x000000ffu) * scale) + bias);
                    accum += xt[base + 1u] * ((float((packed_word >> 8u) & 0x000000ffu) * scale) + bias);
                    accum += xt[base + 2u] * ((float((packed_word >> 16u) & 0x000000ffu) * scale) + bias);
                    accum += xt[base + 3u] * ((float((packed_word >> 24u) & 0x000000ffu) * scale) + bias);
            }
            result += accum;

            ws += words_per_block;
            scale_base += values_per_block / 64u;
            bias_base += values_per_block / 64u;
            x += values_per_block;
        }

        const float reduced = simd_sum(result);
        if (simd_lid == 0u) {
            shared_gate_out[0] = reduced;
        }
        return;
    }
    if (tile.y > out_groups) {
        return;
    }

    const uint row_base = tile.y * (simdgroups * results_per_simdgroup) +
        simd_gid * results_per_simdgroup;

    const device uint* gate_ws = gate_packed + row_base * packed_cols + simd_lid;
    const device uint* up_ws = up_packed + row_base * packed_cols + simd_lid;
    const device bfloat* gate_scale_base = gate_scales + row_base * groups;
    const device bfloat* gate_bias_base = gate_biases + row_base * groups;
    const device bfloat* up_scale_base = up_scales + row_base * groups;
    const device bfloat* up_bias_base = up_biases + row_base * groups;
    const device float* x = lhs + simd_lid * values_per_word;

    float gate_result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float up_result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
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
            const uint out_row = row_base + row;
            if (out_row < out_dim) {
                const device uint* gate_row_words = gate_ws + row * packed_cols;
                const device uint* up_row_words = up_ws + row * packed_cols;
                float gate_accum = 0.0f;
                float up_accum = 0.0f;
                for (uint word = 0u; word < words_per_thread; ++word) {
                    const uint gate_word = gate_row_words[word * 32u];
                    const uint up_word = up_row_words[word * 32u];
                    const uint group = (simd_lid + word * 32u) / 16u;
                    const float gate_scale = gate_scale_base[row * groups + group];
                    const float gate_bias = gate_bias_base[row * groups + group];
                    const float up_scale = up_scale_base[row * groups + group];
                    const float up_bias = up_bias_base[row * groups + group];
                    const uint base = word * values_per_word;
                    gate_accum += xt[base] * ((float(gate_word & 0x000000ffu) * gate_scale) + gate_bias);
                    gate_accum += xt[base + 1u] * ((float((gate_word >> 8u) & 0x000000ffu) * gate_scale) + gate_bias);
                    gate_accum += xt[base + 2u] * ((float((gate_word >> 16u) & 0x000000ffu) * gate_scale) + gate_bias);
                    gate_accum += xt[base + 3u] * ((float((gate_word >> 24u) & 0x000000ffu) * gate_scale) + gate_bias);
                    up_accum += xt[base] * ((float(up_word & 0x000000ffu) * up_scale) + up_bias);
                    up_accum += xt[base + 1u] * ((float((up_word >> 8u) & 0x000000ffu) * up_scale) + up_bias);
                    up_accum += xt[base + 2u] * ((float((up_word >> 16u) & 0x000000ffu) * up_scale) + up_bias);
                    up_accum += xt[base + 3u] * ((float((up_word >> 24u) & 0x000000ffu) * up_scale) + up_bias);
                }
                gate_result[row] += gate_accum;
                up_result[row] += up_accum;
            }
        }

        gate_ws += words_per_block;
        up_ws += words_per_block;
        gate_scale_base += values_per_block / 64u;
        gate_bias_base += values_per_block / 64u;
        up_scale_base += values_per_block / 64u;
        up_bias_base += values_per_block / 64u;
        x += values_per_block;
    }

    for (uint row = 0u; row < results_per_simdgroup; ++row) {
        const uint out_row = row_base + row;
        const float gate = simd_sum(gate_result[row]);
        const float up = simd_sum(up_result[row]);
        if (simd_lid == 0u && out_row < out_dim) {
            out[out_row] = (gate / (1.0f + exp(-gate))) * up;
        }
    }
}

kernel void affine_gate_up_swiglu_gate_fast_u8_gs128_f32(
    device const float* lhs [[buffer(0)]],
    device const uint* gate_packed [[buffer(1)]],
    device const bfloat* gate_scales [[buffer(2)]],
    device const bfloat* gate_biases [[buffer(3)]],
    device const uint* up_packed [[buffer(4)]],
    device const bfloat* up_scales [[buffer(5)]],
    device const bfloat* up_biases [[buffer(6)]],
    device const uint* shared_gate_packed [[buffer(7)]],
    device const bfloat* shared_gate_scales [[buffer(8)]],
    device const bfloat* shared_gate_biases [[buffer(9)]],
    device float* out [[buffer(10)]],
    device float* shared_gate_out [[buffer(11)]],
    constant uint4& dims [[buffer(12)]],
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
    const uint out_groups = (out_dim + 7u) / 8u;

    if (tile.y == out_groups) {
        if (simd_gid != 0u) {
            return;
        }
        const device uint* ws = shared_gate_packed + simd_lid;
        const device bfloat* scale_base = shared_gate_scales;
        const device bfloat* bias_base = shared_gate_biases;
        const device float* x = lhs + simd_lid * values_per_word;

        float result = 0.0f;
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

            float accum = 0.0f;
            for (uint word = 0u; word < words_per_thread; ++word) {
                const uint packed_word = ws[word * 32u];
                const uint group = (simd_lid + word * 32u) / 32u;
                const float scale = scale_base[group];
                const float bias = bias_base[group];
                const uint base = word * values_per_word;
                    accum += xt[base] * ((float(packed_word & 0x000000ffu) * scale) + bias);
                    accum += xt[base + 1u] * ((float((packed_word >> 8u) & 0x000000ffu) * scale) + bias);
                    accum += xt[base + 2u] * ((float((packed_word >> 16u) & 0x000000ffu) * scale) + bias);
                    accum += xt[base + 3u] * ((float((packed_word >> 24u) & 0x000000ffu) * scale) + bias);
            }
            result += accum;

            ws += words_per_block;
            scale_base += values_per_block / 128u;
            bias_base += values_per_block / 128u;
            x += values_per_block;
        }

        const float reduced = simd_sum(result);
        if (simd_lid == 0u) {
            shared_gate_out[0] = reduced;
        }
        return;
    }
    if (tile.y > out_groups || simd_gid >= simdgroups) {
        return;
    }

    const uint row_base = tile.y * (simdgroups * results_per_simdgroup) +
        simd_gid * results_per_simdgroup;

    const device uint* gate_ws = gate_packed + row_base * packed_cols + simd_lid;
    const device uint* up_ws = up_packed + row_base * packed_cols + simd_lid;
    const device bfloat* gate_scale_base = gate_scales + row_base * groups;
    const device bfloat* gate_bias_base = gate_biases + row_base * groups;
    const device bfloat* up_scale_base = up_scales + row_base * groups;
    const device bfloat* up_bias_base = up_biases + row_base * groups;
    const device float* x = lhs + simd_lid * values_per_word;

    float gate_result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float up_result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
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
            const uint out_row = row_base + row;
            if (out_row < out_dim) {
                const device uint* gate_row_words = gate_ws + row * packed_cols;
                const device uint* up_row_words = up_ws + row * packed_cols;
                float gate_accum = 0.0f;
                float up_accum = 0.0f;
                for (uint word = 0u; word < words_per_thread; ++word) {
                    const uint gate_word = gate_row_words[word * 32u];
                    const uint up_word = up_row_words[word * 32u];
                    const uint group = (simd_lid + word * 32u) / 32u;
                    const float gate_scale = gate_scale_base[row * groups + group];
                    const float gate_bias = gate_bias_base[row * groups + group];
                    const float up_scale = up_scale_base[row * groups + group];
                    const float up_bias = up_bias_base[row * groups + group];
                    const uint base = word * values_per_word;
                    gate_accum += xt[base] * ((float(gate_word & 0x000000ffu) * gate_scale) + gate_bias);
                    gate_accum += xt[base + 1u] * ((float((gate_word >> 8u) & 0x000000ffu) * gate_scale) + gate_bias);
                    gate_accum += xt[base + 2u] * ((float((gate_word >> 16u) & 0x000000ffu) * gate_scale) + gate_bias);
                    gate_accum += xt[base + 3u] * ((float((gate_word >> 24u) & 0x000000ffu) * gate_scale) + gate_bias);
                    up_accum += xt[base] * ((float(up_word & 0x000000ffu) * up_scale) + up_bias);
                    up_accum += xt[base + 1u] * ((float((up_word >> 8u) & 0x000000ffu) * up_scale) + up_bias);
                    up_accum += xt[base + 2u] * ((float((up_word >> 16u) & 0x000000ffu) * up_scale) + up_bias);
                    up_accum += xt[base + 3u] * ((float((up_word >> 24u) & 0x000000ffu) * up_scale) + up_bias);
                }
                gate_result[row] += gate_accum;
                up_result[row] += up_accum;
            }
        }

        gate_ws += words_per_block;
        up_ws += words_per_block;
        gate_scale_base += values_per_block / 128u;
        gate_bias_base += values_per_block / 128u;
        up_scale_base += values_per_block / 128u;
        up_bias_base += values_per_block / 128u;
        x += values_per_block;
    }

    for (uint row = 0u; row < results_per_simdgroup; ++row) {
        const uint out_row = row_base + row;
        const float gate = simd_sum(gate_result[row]);
        const float up = simd_sum(up_result[row]);
        if (simd_lid == 0u && out_row < out_dim) {
            out[out_row] = (gate / (1.0f + exp(-gate))) * up;
        }
    }
}

kernel void affine_gate_up_swiglu_fast_u8_gs128_f32(
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
    const uint values_per_word = 4u;
    const uint words_per_thread = 4u;
    const uint values_per_thread = values_per_word * words_per_thread;
    const uint values_per_block = values_per_thread * 32u;
    const uint words_per_block = values_per_block / values_per_word;
    const uint row_base = tile.y * (simdgroups * results_per_simdgroup) +
        simd_gid * results_per_simdgroup;

    const device uint* gate_ws = gate_packed + row_base * packed_cols + simd_lid;
    const device uint* up_ws = up_packed + row_base * packed_cols + simd_lid;
    const device bfloat* gate_scale_base = gate_scales + row_base * groups;
    const device bfloat* gate_bias_base = gate_biases + row_base * groups;
    const device bfloat* up_scale_base = up_scales + row_base * groups;
    const device bfloat* up_bias_base = up_biases + row_base * groups;
    const device float* x = lhs + tile.x * in_dim + simd_lid * values_per_word;

    float gate_result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float up_result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
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
            const device uint* gate_row_words = gate_ws + row * packed_cols;
            const device uint* up_row_words = up_ws + row * packed_cols;
            float gate_accum = 0.0f;
            float up_accum = 0.0f;
            for (uint word = 0u; word < words_per_thread; ++word) {
                const uint gate_word = gate_row_words[word * 32u];
                const uint up_word = up_row_words[word * 32u];
                const uint group = (simd_lid + word * 32u) / 32u;
                const float gate_scale = gate_scale_base[row * groups + group];
                const float gate_bias = gate_bias_base[row * groups + group];
                const float up_scale = up_scale_base[row * groups + group];
                const float up_bias = up_bias_base[row * groups + group];
                const uint base = word * values_per_word;
                gate_accum += xt[base] * ((float(gate_word & 0x000000ffu) * gate_scale) + gate_bias);
                gate_accum += xt[base + 1u] * ((float((gate_word >> 8u) & 0x000000ffu) * gate_scale) + gate_bias);
                gate_accum += xt[base + 2u] * ((float((gate_word >> 16u) & 0x000000ffu) * gate_scale) + gate_bias);
                gate_accum += xt[base + 3u] * ((float((gate_word >> 24u) & 0x000000ffu) * gate_scale) + gate_bias);
                up_accum += xt[base] * ((float(up_word & 0x000000ffu) * up_scale) + up_bias);
                up_accum += xt[base + 1u] * ((float((up_word >> 8u) & 0x000000ffu) * up_scale) + up_bias);
                up_accum += xt[base + 2u] * ((float((up_word >> 16u) & 0x000000ffu) * up_scale) + up_bias);
                up_accum += xt[base + 3u] * ((float((up_word >> 24u) & 0x000000ffu) * up_scale) + up_bias);
            }
            gate_result[row] += gate_accum;
            up_result[row] += up_accum;
        }

        gate_ws += words_per_block;
        up_ws += words_per_block;
        gate_scale_base += values_per_block / 128u;
        gate_bias_base += values_per_block / 128u;
        up_scale_base += values_per_block / 128u;
        up_bias_base += values_per_block / 128u;
        x += values_per_block;
    }

    for (uint row = 0u; row < results_per_simdgroup; ++row) {
        const float gate = simd_sum(gate_result[row]);
        const float up = simd_sum(up_result[row]);
        if (simd_lid == 0u) {
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

kernel void affine_qmv_rms_fast_u8_gs64_f32(
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
    const uint values_per_word = 4u;
    const uint words_per_thread = 4u;
    const uint values_per_thread = values_per_word * words_per_thread;
    const uint values_per_block = values_per_thread * 32u;
    const uint words_per_block = values_per_block / values_per_word;
    const uint row_base = tile.y * (simdgroups * results_per_simdgroup) +
        simd_gid * results_per_simdgroup;

    threadgroup float inv_rms_shared;
    float sumsq = 0.0f;
    if (simd_gid == 0u) {
        const device float* norm_x = lhs + tile.x * in_dim + simd_lid * values_per_thread;
        for (uint k = 0u; k < in_dim; k += values_per_block) {
            for (uint i = 0u; i < values_per_thread; ++i) {
                const float value = norm_x[i];
                sumsq += value * value;
            }
            norm_x += values_per_block;
        }
        const float reduced_sumsq = simd_sum(sumsq);
        if (simd_lid == 0u) {
            inv_rms_shared = rsqrt((reduced_sumsq / float(in_dim)) + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_rms = inv_rms_shared;

    const device uint* ws = packed + row_base * packed_cols + simd_lid;
    const device bfloat* scale_base = scales + row_base * groups;
    const device bfloat* bias_base = biases + row_base * groups;
    const device float* x = lhs + tile.x * in_dim + simd_lid * values_per_word;
    const device float* gamma = rms_weight + simd_lid * values_per_word;

    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint k = 0u; k < in_dim; k += values_per_block) {
        float xt[16];
        for (uint word = 0u; word < words_per_thread; ++word) {
            const uint base = word * values_per_word;
            const uint x_offset = word * 32u * values_per_word;
            xt[base] = x[x_offset] * inv_rms * gamma[x_offset];
            xt[base + 1u] = x[x_offset + 1u] * inv_rms * gamma[x_offset + 1u];
            xt[base + 2u] = x[x_offset + 2u] * inv_rms * gamma[x_offset + 2u];
            xt[base + 3u] = x[x_offset + 3u] * inv_rms * gamma[x_offset + 3u];
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
        gamma += values_per_block;
    }

    for (uint row = 0u; row < results_per_simdgroup; ++row) {
        const float reduced = simd_sum(result[row]);
        if (simd_lid == 0u) {
            out[tile.x * out_dim + row_base + row] = reduced;
        }
    }
}

kernel void affine_qmv_rms_fast_u8_gs128_f32(
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
    const uint values_per_word = 4u;
    const uint words_per_thread = 4u;
    const uint values_per_thread = values_per_word * words_per_thread;
    const uint values_per_block = values_per_thread * 32u;
    const uint words_per_block = values_per_block / values_per_word;
    const uint row_base = tile.y * (simdgroups * results_per_simdgroup) +
        simd_gid * results_per_simdgroup;

    threadgroup float inv_rms_shared;
    float sumsq = 0.0f;
    if (simd_gid == 0u) {
        const device float* norm_x = lhs + tile.x * in_dim + simd_lid * values_per_thread;
        for (uint k = 0u; k < in_dim; k += values_per_block) {
            for (uint i = 0u; i < values_per_thread; ++i) {
                const float value = norm_x[i];
                sumsq += value * value;
            }
            norm_x += values_per_block;
        }
        const float reduced_sumsq = simd_sum(sumsq);
        if (simd_lid == 0u) {
            inv_rms_shared = rsqrt((reduced_sumsq / float(in_dim)) + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_rms = inv_rms_shared;

    const device uint* ws = packed + row_base * packed_cols + simd_lid;
    const device bfloat* scale_base = scales + row_base * groups;
    const device bfloat* bias_base = biases + row_base * groups;
    const device float* x = lhs + tile.x * in_dim + simd_lid * values_per_word;
    const device float* gamma = rms_weight + simd_lid * values_per_word;

    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint k = 0u; k < in_dim; k += values_per_block) {
        float xt[16];
        for (uint word = 0u; word < words_per_thread; ++word) {
            const uint base = word * values_per_word;
            const uint x_offset = word * 32u * values_per_word;
            xt[base] = x[x_offset] * inv_rms * gamma[x_offset];
            xt[base + 1u] = x[x_offset + 1u] * inv_rms * gamma[x_offset + 1u];
            xt[base + 2u] = x[x_offset + 2u] * inv_rms * gamma[x_offset + 2u];
            xt[base + 3u] = x[x_offset + 3u] * inv_rms * gamma[x_offset + 3u];
        }

        for (uint row = 0u; row < results_per_simdgroup; ++row) {
            const device uint* row_words = ws + row * packed_cols;
            float accum = 0.0f;
            for (uint word = 0u; word < words_per_thread; ++word) {
                const uint packed_word = row_words[word * 32u];
                const uint group = (simd_lid + word * 32u) / 32u;
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
        scale_base += values_per_block / 128u;
        bias_base += values_per_block / 128u;
        x += values_per_block;
        gamma += values_per_block;
    }

    for (uint row = 0u; row < results_per_simdgroup; ++row) {
        const float reduced = simd_sum(result[row]);
        if (simd_lid == 0u) {
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

kernel void affine_qkv_split_rms_qmv_fast_u8_gs64_f32(
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
    const uint values_per_word = 4u;
    const uint words_per_thread = 4u;
    const uint values_per_thread = values_per_word * words_per_thread;
    const uint values_per_block = values_per_thread * 32u;
    const uint words_per_block = values_per_block / values_per_word;
    const uint row_base = tile.y * (simdgroups * results_per_simdgroup) +
        simd_gid * results_per_simdgroup;

    threadgroup float inv_rms_shared;
    float sumsq = 0.0f;
    if (simd_gid == 0u) {
        const device float* norm_x = lhs + tile.x * in_dim + simd_lid * values_per_thread;
        for (uint k = 0u; k < in_dim; k += values_per_block) {
            for (uint i = 0u; i < values_per_thread; ++i) {
                const float value = norm_x[i];
                sumsq += value * value;
            }
            norm_x += values_per_block;
        }
        const float reduced_sumsq = simd_sum(sumsq);
        if (simd_lid == 0u) {
            inv_rms_shared = rsqrt((reduced_sumsq / float(in_dim)) + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_rms = inv_rms_shared;

    const device uint* ws = packed + row_base * packed_cols + simd_lid;
    const device bfloat* scale_base = scales + row_base * groups;
    const device bfloat* bias_base = biases + row_base * groups;
    const device float* x = lhs + tile.x * in_dim + simd_lid * values_per_word;
    const device float* gamma = rms_weight + simd_lid * values_per_word;

    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint k = 0u; k < in_dim; k += values_per_block) {
        float xt[16];
        for (uint word = 0u; word < words_per_thread; ++word) {
            const uint base = word * values_per_word;
            const uint x_offset = word * 32u * values_per_word;
            xt[base] = x[x_offset] * inv_rms * gamma[x_offset];
            xt[base + 1u] = x[x_offset + 1u] * inv_rms * gamma[x_offset + 1u];
            xt[base + 2u] = x[x_offset + 2u] * inv_rms * gamma[x_offset + 2u];
            xt[base + 3u] = x[x_offset + 3u] * inv_rms * gamma[x_offset + 3u];
        }

        for (uint row = 0u; row < results_per_simdgroup; ++row) {
            const uint out_row = row_base + row;
            if (out_row < out_dim) {
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
        }

        ws += words_per_block;
        scale_base += values_per_block / 64u;
        bias_base += values_per_block / 64u;
        x += values_per_block;
        gamma += values_per_block;
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

kernel void affine_qmv_gated_input_fast_u8_gs64_f32(
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
    const device float* g = gate + tile.x * in_dim + simd_lid * values_per_word;

    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint k = 0u; k < in_dim; k += values_per_block) {
        float xt[16];
        for (uint word = 0u; word < words_per_thread; ++word) {
            const uint base = word * values_per_word;
            const uint x_offset = word * 32u * values_per_word;
            xt[base] = x[x_offset] * (1.0f / (1.0f + exp(-g[x_offset])));
            xt[base + 1u] = x[x_offset + 1u] * (1.0f / (1.0f + exp(-g[x_offset + 1u])));
            xt[base + 2u] = x[x_offset + 2u] * (1.0f / (1.0f + exp(-g[x_offset + 2u])));
            xt[base + 3u] = x[x_offset + 3u] * (1.0f / (1.0f + exp(-g[x_offset + 3u])));
        }

        for (uint row = 0u; row < results_per_simdgroup; ++row) {
            const uint out_row = row_base + row;
            if (out_row < out_dim) {
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
        }
        ws += words_per_block;
        scale_base += values_per_block / 64u;
        bias_base += values_per_block / 64u;
        x += values_per_block;
        g += values_per_block;
    }

    for (uint row = 0u; row < results_per_simdgroup; ++row) {
        const uint out_row = row_base + row;
        const float reduced = simd_sum(result[row]);
        if (simd_lid == 0u && out_row < out_dim) {
            out[tile.x * out_dim + out_row] = reduced;
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

kernel void scale_topk_scores_f32(
    device const uint* indices [[buffer(0)]],
    device float* scores [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < count) {
        scores[gid] *= scales[indices[gid]];
    }
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

kernel void weighted_sum_add_shared_topk_f32(
    device const float* src [[buffer(0)]],
    device const float* scores [[buffer(1)]],
    device const float* residual [[buffer(2)]],
    device const float* shared [[buffer(3)]],
    device const float* shared_gate [[buffer(4)]],
    device float* out [[buffer(5)]],
    constant uint2& dims [[buffer(6)]],
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
    const float shared_scale = 1.0f / (1.0f + fast::exp(-shared_gate[0]));
    out[gid] = acc + shared[gid] * shared_scale;
}

kernel void affine_gather_down_weighted_shared_fast_u8_gs64_f32(
    device const float* lhs [[buffer(0)]],
    device const uint* packed [[buffer(1)]],
    device const bfloat* scales [[buffer(2)]],
    device const bfloat* biases [[buffer(3)]],
    device const uint* expert_indices [[buffer(4)]],
    device const float* scores [[buffer(5)]],
    device const float* residual [[buffer(6)]],
    device const float* shared [[buffer(7)]],
    device const float* shared_gate [[buffer(8)]],
    device float* out [[buffer(9)]],
    constant uint4& dims [[buffer(10)]],
    constant uint& groups [[buffer(11)]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint tile [[threadgroup_position_in_grid]]
) {
    const uint topk = dims.x;
    const uint out_dim = dims.y;
    const uint in_dim = dims.z;
    const uint packed_cols = dims.w;
    const uint results_per_simdgroup = 4u;
    const uint row_base = tile * 8u + simd_gid * results_per_simdgroup;
    if (row_base >= out_dim) {
        return;
    }

    float acc[4];
    for (uint row = 0u; row < results_per_simdgroup; ++row) {
        const uint out_row = row_base + row;
        acc[row] = (out_row < out_dim) ? residual[out_row] : 0.0f;
    }

    const uint values_per_word = 4u;
    const uint words_per_thread = 4u;
    const uint values_per_thread = values_per_word * words_per_thread;
    const uint values_per_block = values_per_thread * 32u;
    const uint words_per_block = values_per_block / values_per_word;

    for (uint slot = 0u; slot < topk; ++slot) {
        const uint expert = expert_indices[slot];
        const device uint* ws =
            packed + ((expert * out_dim + row_base) * packed_cols) + simd_lid;
        const device bfloat* scale_base = scales + ((expert * out_dim + row_base) * groups);
        const device bfloat* bias_base = biases + ((expert * out_dim + row_base) * groups);
        const device float* x = lhs + slot * in_dim + simd_lid * values_per_word;

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
                if (row_base + row < out_dim) {
                    const device uint* row_words = ws + row * packed_cols;
                    float partial = 0.0f;
                    for (uint word = 0u; word < words_per_thread; ++word) {
                        const uint packed_word = row_words[word * 32u];
                        const uint group = (simd_lid + word * 32u) / 16u;
                        const float scale = scale_base[row * groups + group];
                        const float bias = bias_base[row * groups + group];
                        const uint base = word * values_per_word;
                        partial +=
                            xt[base] * ((float(packed_word & 0x000000ffu) * scale) + bias);
                        partial += xt[base + 1u] *
                            ((float((packed_word >> 8u) & 0x000000ffu) * scale) + bias);
                        partial += xt[base + 2u] *
                            ((float((packed_word >> 16u) & 0x000000ffu) * scale) + bias);
                        partial += xt[base + 3u] *
                            ((float((packed_word >> 24u) & 0x000000ffu) * scale) + bias);
                    }
                    result[row] += partial;
                }
            }

            ws += words_per_block;
            scale_base += values_per_block / 64u;
            bias_base += values_per_block / 64u;
            x += values_per_block;
        }

        const float score = scores[slot];
        for (uint row = 0u; row < results_per_simdgroup; ++row) {
            const float reduced = simd_sum(result[row]);
            if (simd_lid == 0u && row_base + row < out_dim) {
                acc[row] += reduced * score;
            }
        }
    }

    const float shared_scale = 1.0f / (1.0f + fast::exp(-shared_gate[0]));
    for (uint row = 0u; row < results_per_simdgroup; ++row) {
        const uint out_row = row_base + row;
        if (simd_lid == 0u && out_row < out_dim) {
            out[out_row] = acc[row] + shared[out_row] * shared_scale;
        }
    }
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
    const float scale = 1.0f / (1.0f + fast::exp(-gate[0]));
    dst[gid] += src[gid] * scale;
}

kernel void add_sigmoid_scaled_rows_f32(
    device const float* src [[buffer(0)]],
    device const float* gate [[buffer(1)]],
    device float* dst [[buffer(2)]],
    constant uint2& dims [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint rows = dims.x;
    const uint row_dim = dims.y;
    const uint len = rows * row_dim;
    if (gid >= len || row_dim == 0u) {
        return;
    }
    const uint row = gid / row_dim;
    const float scale = 1.0f / (1.0f + fast::exp(-gate[row]));
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

kernel void copy_u16(
    device const ushort* src [[buffer(0)]],
    device ushort* out [[buffer(1)]],
    constant uint& len [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) {
        return;
    }
    out[gid] = src[gid];
}

// rms_norm multi-rows reproduisant BIT-À-BIT le prologue rms des kernels
// fusionnés (`affine_qmv_rms_fast`, `affine_qkv_split_rms_qmv_fast`) : même
// partition (1 simdgroup, 16 valeurs consécutives par thread, blocs de 512),
// même ordre d'accumulation séquentiel par thread, même `simd_sum`, même
// expression de normalisation `x * inv_rms * gamma`. Permet de dé-fusionner
// rms+qmv en rms_simd → qmm2 sans changer un bit (chemin duo light-batch).
// Préconditions : `dim % 512 == 0` (mêmes gates que les kernels fusionnés),
// threadgroup de 32 threads, 1 threadgroup par row.
kernel void rms_norm_simd_rows_f32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& dim [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint row [[threadgroup_position_in_grid]]
) {
    const uint values_per_thread = 16u;
    const uint block_size = values_per_thread * 32u;
    const uint offset = row * dim;

    const device float* norm_x = input + offset + simd_lid * values_per_thread;
    float sumsq = 0.0f;
    for (uint k = 0u; k < dim; k += block_size) {
        for (uint i = 0u; i < values_per_thread; ++i) {
            const float value = norm_x[i];
            sumsq += value * value;
        }
        norm_x += block_size;
    }
    const float reduced_sumsq = simd_sum(sumsq);
    const float inv_rms = rsqrt((reduced_sumsq / float(dim)) + eps);

    const device float* x = input + offset + simd_lid * values_per_thread;
    const device float* gamma = weight + simd_lid * values_per_thread;
    device float* o = out + offset + simd_lid * values_per_thread;
    for (uint k = 0u; k < dim; k += block_size) {
        for (uint i = 0u; i < values_per_thread; ++i) {
            o[i] = x[i] * inv_rms * gamma[i];
        }
        x += block_size;
        gamma += block_size;
        o += block_size;
    }
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

// LayerNorm Whisper (moyenne + variance + weight + biais), une ligne par
// threadgroup. Reproduit `norm.rs::layer_norm` (deux réductions : moyenne puis
// variance ; `1/sqrt(var+eps)`, PAS rsqrt, pour coller à la CPU). Le seul écart
// possible vs CPU = l'ordre de réduction (arbre vs séquentiel) ≈ 1e-7.
kernel void layer_norm_rows_f32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& dim [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]
) {
    threadgroup float partial[256];
    const uint offset = row * dim;

    float sum = 0.0f;
    for (uint col = tid; col < dim; col += 256u) {
        sum += input[offset + col];
    }
    partial[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float mean = partial[0] / float(dim);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sumsq = 0.0f;
    for (uint col = tid; col < dim; col += 256u) {
        const float d = input[offset + col] - mean;
        sumsq += d * d;
    }
    partial[tid] = sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv_std = 1.0f / sqrt(partial[0] / float(dim) + eps);

    for (uint col = tid; col < dim; col += 256u) {
        out[offset + col] = (input[offset + col] - mean) * inv_std * weight[col] + bias[col];
    }
}

kernel void layer_norm_rows_f32_bf16out(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* out [[buffer(3)]],
    device bfloat* out_bf16 [[buffer(4)]],
    constant uint& dim [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]
) {
    threadgroup float partial[256];
    const uint offset = row * dim;

    float sum = 0.0f;
    for (uint col = tid; col < dim; col += 256u) {
        sum += input[offset + col];
    }
    partial[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float mean = partial[0] / float(dim);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sumsq = 0.0f;
    for (uint col = tid; col < dim; col += 256u) {
        const float d = input[offset + col] - mean;
        sumsq += d * d;
    }
    partial[tid] = sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv_std = 1.0f / sqrt(partial[0] / float(dim) + eps);

    for (uint col = tid; col < dim; col += 256u) {
        const uint index = offset + col;
        const float value = (input[index] - mean) * inv_std * weight[col] + bias[col];
        out[index] = value;
        out_bf16[index] = bfloat(value);
    }
}

// Fusion résiduel + LayerNorm : `summed = left + right` (nouveau h) puis
// `normed = LayerNorm(summed)`. Évite un aller-retour dans le chemin résident.
kernel void add_layer_norm_rows_f32(
    device const float* left [[buffer(0)]],
    device const float* right [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device const float* bias [[buffer(3)]],
    device float* summed [[buffer(4)]],
    device float* normed [[buffer(5)]],
    constant uint& dim [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    uint tid [[thread_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]
) {
    threadgroup float partial[256];
    const uint offset = row * dim;

    float sum = 0.0f;
    for (uint col = tid; col < dim; col += 256u) {
        const float value = left[offset + col] + right[offset + col];
        summed[offset + col] = value;
        sum += value;
    }
    partial[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float mean = partial[0] / float(dim);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sumsq = 0.0f;
    for (uint col = tid; col < dim; col += 256u) {
        const float d = summed[offset + col] - mean;
        sumsq += d * d;
    }
    partial[tid] = sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv_std = 1.0f / sqrt(partial[0] / float(dim) + eps);

    for (uint col = tid; col < dim; col += 256u) {
        normed[offset + col] = (summed[offset + col] - mean) * inv_std * weight[col] + bias[col];
    }
}

kernel void add_layer_norm_rows_f32_bf16out(
    device const float* left [[buffer(0)]],
    device const float* right [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device const float* bias [[buffer(3)]],
    device float* summed [[buffer(4)]],
    device float* normed [[buffer(5)]],
    device bfloat* normed_bf16 [[buffer(6)]],
    constant uint& dim [[buffer(7)]],
    constant float& eps [[buffer(8)]],
    uint tid [[thread_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]
) {
    threadgroup float partial[256];
    const uint offset = row * dim;

    float sum = 0.0f;
    for (uint col = tid; col < dim; col += 256u) {
        const float value = left[offset + col] + right[offset + col];
        summed[offset + col] = value;
        sum += value;
    }
    partial[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float mean = partial[0] / float(dim);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sumsq = 0.0f;
    for (uint col = tid; col < dim; col += 256u) {
        const float d = summed[offset + col] - mean;
        sumsq += d * d;
    }
    partial[tid] = sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv_std = 1.0f / sqrt(partial[0] / float(dim) + eps);

    for (uint col = tid; col < dim; col += 256u) {
        const uint index = offset + col;
        const float value = (summed[index] - mean) * inv_std * weight[col] + bias[col];
        normed[index] = value;
        normed_bf16[index] = bfloat(value);
    }
}

// GELU exact Whisper/HF : `0.5·x·(1+erf(x/√2))`, erf approximé par le MÊME
// polynôme Abramowitz-Stegun 7.1.26 que `activation.rs::gelu_scalar` (élément par
// élément ⇒ pas de réduction ; seuls `exp`/`sqrt` Metal vs Rust peuvent différer
// au dernier ULP).
kernel void gelu_f32(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& len [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) {
        return;
    }
    const float x = input[gid];
    const float a = x * 0.7071067811865476f;
    const float s = a < 0.0f ? -1.0f : 1.0f;
    const float ax = fabs(a);
    const float t = 1.0f / (1.0f + 0.3275911f * ax);
    const float poly =
        ((((1.0614054f * t - 1.4531521f) * t + 1.4214138f) * t - 0.28449672f) * t + 0.2548296f) * t;
    const float erf = s * (1.0f - poly * exp(-ax * ax));
    out[gid] = 0.5f * x * (1.0f + erf);
}

kernel void gelu_f32_bf16out(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    device bfloat* out_bf16 [[buffer(2)]],
    constant uint& len [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) {
        return;
    }
    const float x = input[gid];
    const float a = x * 0.7071067811865476f;
    const float s = a < 0.0f ? -1.0f : 1.0f;
    const float ax = fabs(a);
    const float t = 1.0f / (1.0f + 0.3275911f * ax);
    const float poly =
        ((((1.0614054f * t - 1.4531521f) * t + 1.4214138f) * t - 0.28449672f) * t + 0.2548296f) * t;
    const float erf = s * (1.0f - poly * exp(-ax * ax));
    const float value = 0.5f * x * (1.0f + erf);
    out[gid] = value;
    out_bf16[gid] = bfloat(value);
}

// Ajoute le biais ligne à ligne (épilogue des projections Whisper), in-place.
kernel void add_row_bias_f32(
    device float* data [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    constant uint2& dims [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total = dims.x * dims.y;
    if (gid >= total) {
        return;
    }
    data[gid] += bias[gid % dims.y];
}

// im2col conv1d Whisper : déplie `input [frames, in_ch]` (frames-major) en
// `output [out_frames, in_ch·kernel]` (colonne = in_ch·kernel + k, alignée sur le
// layout poids `[out, in_ch, k]`), padding zéro. Le GEMM tuilé qui suit calcule
// alors le conv1d exactement. `t_in = t_out·stride + k − padding`.
kernel void im2col_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint4& dims [[buffer(2)]],
    constant uint2& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint out_frames = dims.x;
    const uint in_ch = dims.y;
    const uint kernel_size = dims.z;
    const uint frames = dims.w;
    const uint stride = params.x;
    const uint padding = params.y;
    const uint cols = in_ch * kernel_size;
    if (gid >= out_frames * cols) {
        return;
    }
    const uint t_out = gid / cols;
    const uint col = gid - t_out * cols;
    const uint ich = col / kernel_size;
    const uint k = col - ich * kernel_size;
    const int t_in = int(t_out * stride + k) - int(padding);
    float value = 0.0f;
    if (t_in >= 0 && t_in < int(frames)) {
        value = input[uint(t_in) * in_ch + ich];
    }
    output[gid] = value;
}

// Attention single-query Whisper (decode) : la requête `q [dim]` attend tout le
// KV `[len, dim]` (self caché OU cross statique), une tête par threadgroup.
// Reproduit `whisper/decoder.rs::single_query_attention` au plus près : dot
// par-clé séquentiel (= CPU), max EXACT (ordre-indépendant), contexte sommé
// par-clé séquentiel avec `/denom` par terme (= CPU). Seuls le `denom` (réduction
// arbre) et `exp` (ULP Metal) diffèrent ≈ 1e-7.
kernel void whisper_attn_decode_f32(
    device const float* q [[buffer(0)]],
    device const float* keys [[buffer(1)]],
    device const float* values [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint4& dims [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint head [[threadgroup_position_in_grid]]
) {
    const uint heads = dims.x;
    const uint head_dim = dims.y;
    const uint len = dims.z;
    if (head >= heads || len == 0u || len > 2048u) {
        return;
    }
    const uint dim = heads * head_dim;
    const uint head_base = head * head_dim;
    const float scale = 1.0f / sqrt(float(head_dim));

    threadgroup float sc[2048];
    threadgroup float partial[256];

    for (uint r = tid; r < len; r += 256u) {
        const uint k_base = r * dim + head_base;
        float dot = 0.0f;
        for (uint c = 0u; c < head_dim; ++c) {
            dot += q[head_base + c] * keys[k_base + c];
        }
        sc[r] = dot * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_max = -INFINITY;
    for (uint r = tid; r < len; r += 256u) {
        local_max = max(local_max, sc[r]);
    }
    partial[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial[tid] = max(partial[tid], partial[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float max_score = partial[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_sum = 0.0f;
    for (uint r = tid; r < len; r += 256u) {
        const float e = exp(sc[r] - max_score);
        sc[r] = e;
        local_sum += e;
    }
    partial[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float denom = partial[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint c = tid; c < head_dim; c += 256u) {
        float acc = 0.0f;
        for (uint r = 0u; r < len; ++r) {
            acc += (sc[r] / denom) * values[r * dim + head_base + c];
        }
        out[head_base + c] = acc;
    }
}

// Attention single-query Whisper, variante vector/online softmax pour le cross
// decode large-v3-turbo (`head_dim=64`). Adaptée de l'algorithme MLX
// `sdpa_vector` (MIT) au layout reti `[len, heads*64]`.
kernel void whisper_attn_decode_vec64_f32(
    device const float* q [[buffer(0)]],
    device const float* keys [[buffer(1)]],
    device const float* values [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint4& dims [[buffer(4)]],
    uint head [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
    constexpr uint BN = 32u;
    constexpr uint D = 64u;
    constexpr uint EPT = 2u;

    const uint heads = dims.x;
    const uint len = dims.z;
    if (head >= heads || len == 0u || len > 2048u) {
        return;
    }
    const uint dim = heads * D;
    const uint head_base = head * D;
    const uint lane_col = simd_lid * EPT;
    const float scale = 0.125f; // 1/sqrt(64)

    threadgroup float outputs[BN * BN];
    threadgroup float max_scores[BN];
    threadgroup float sum_exp_scores[BN];

    const float q0 = q[head_base + lane_col] * scale;
    const float q1 = q[head_base + lane_col + 1u] * scale;
    float o0 = 0.0f;
    float o1 = 0.0f;
    float max_score = -INFINITY;
    float sum_exp_score = 0.0f;

    for (uint row = simd_gid; row < len; row += BN) {
        const uint kv_base = row * dim + head_base + lane_col;
        float score = q0 * keys[kv_base] + q1 * keys[kv_base + 1u];
        score = simd_sum(score);

        const float new_max = max(max_score, score);
        const float old_factor = fast::exp(max_score - new_max);
        const float exp_score = fast::exp(score - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * old_factor + exp_score;
        o0 = o0 * old_factor + exp_score * values[kv_base];
        o1 = o1 * old_factor + exp_score * values[kv_base + 1u];
    }

    if (simd_lid == 0u) {
        max_scores[simd_gid] = max_score;
        sum_exp_scores[simd_gid] = sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float source_max = max_scores[simd_lid];
    const float final_max = simd_max(source_max);
    const float source_factor = fast::exp(source_max - final_max);
    const float denom = simd_sum(sum_exp_scores[simd_lid] * source_factor);

    outputs[simd_lid * BN + simd_gid] = o0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float acc0 = simd_sum(outputs[simd_gid * BN + simd_lid] * source_factor);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    outputs[simd_lid * BN + simd_gid] = o1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float acc1 = simd_sum(outputs[simd_gid * BN + simd_lid] * source_factor);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_lid == 0u) {
        const uint out_base = head_base + simd_gid * EPT;
        const float inv = denom == 0.0f ? 0.0f : 1.0f / denom;
        out[out_base] = acc0 * inv;
        out[out_base + 1u] = acc1 * inv;
    }
}

kernel void rms_norm_rope_heads_f32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint4& dims [[buffer(3)]],
    constant float4& params [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 tile [[threadgroup_position_in_grid]]
) {
    const uint seq = dims.x;
    const uint heads = dims.y;
    const uint head_dim = dims.z;
    const uint rope_dims = dims.w;
    const float eps = params.x;
    const float base_theta = params.y;
    const float frequency_dim = params.z;
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
    const uint pairs = rope_dims / 2u;
    for (uint col = tid; col < head_dim; col += 256u) {
        float value = input[start + col] * inv_rms * weight[col];
        if (col < rope_dims) {
            // Rotate-half : la paire (pair, pair+pairs) tourne a la frequence
            // d'exposant 2*pair/frequency_dim (miroir CPU rms_norm_rope_heads_at).
            const uint pair = (col < pairs) ? col : (col - pairs);
            const float exponent = float(2u * pair) / frequency_dim;
            const float angle = float(pos) / pow(base_theta, exponent);
            const float c = cos(angle);
            const float s = sin(angle);
            const float first = input[start + pair] * inv_rms * weight[pair];
            const float second = input[start + pair + pairs] * inv_rms * weight[pair + pairs];
            value = (col < pairs) ? (first * c - second * s) : (first * s + second * c);
        }
        out[start + col] = value;
    }
}

kernel void rms_norm_heads_no_scale_f32(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint3& dims [[buffer(2)]],
    constant float& eps [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 tile [[threadgroup_position_in_grid]]
) {
    const uint seq = dims.x;
    const uint heads = dims.y;
    const uint head_dim = dims.z;
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
        out[start + col] = input[start + col] * inv_rms;
    }
}

kernel void causal_attention_prefill_f32(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint4& dims [[buffer(4)]],
    constant float2& scale_params [[buffer(5)]],
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
    const float scale = scale_params.y > 0.0f ? scale_params.x : rsqrt(float(head_dim));

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

// Variante « moyenne » du prefill causal : même arithmétique que le kernel court
// mais scores[2048] en mémoire threadgroup. Elle couvre les prompts usuels sans
// les trois passes de recalcul du kernel long, tout en gardant l'ordre exact :
// dot réduit en arbre, max/somme séquentiels 0..=pos, normalisation avant V.
kernel void causal_attention_prefill_mid_f32(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint4& dims [[buffer(4)]],
    constant float2& scale_params [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 tile [[threadgroup_position_in_grid]]
) {
    const uint seq = dims.x;
    const uint q_heads = dims.y;
    const uint kv_heads = dims.z;
    const uint head_dim = dims.w;
    const uint q_head = tile.x;
    const uint pos = tile.y;
    if (q_head >= q_heads || pos >= seq || seq > 2048u) {
        return;
    }

    threadgroup float partial[256];
    threadgroup float scores[2048];
    const uint kv_group = q_heads / kv_heads;
    const uint kv_head = q_head / kv_group;
    const uint q_dim = q_heads * head_dim;
    const uint kv_dim = kv_heads * head_dim;
    const uint q_start = pos * q_dim + q_head * head_dim;
    const uint kv_head_start = kv_head * head_dim;
    const float scale = scale_params.y > 0.0f ? scale_params.x : rsqrt(float(head_dim));

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

// Variante « longue » du prefill causal : lève le plafond seq > 256 de
// `causal_attention_prefill_f32` (qui sortait SANS écrire, laissant l'attention à
// zéro/déchet au-delà de 256 positions) tout en restant BYTE-IDENTIQUE. Le score
// de chaque ligne est RECALCULÉ à chaque passe (max, somme, sortie) au lieu d'être
// stocké dans un `scores[256]` : arithmétique inchangée (dot réduit en arbre,
// max/somme séquentiels dans l'ordre 0..=pos, normalisation avant le produit V),
// mais mémoire threadgroup en O(head_dim) → seq non borné. Exige head_dim <= 256
// (vrai pour tous les modèles résidents : head_dim = 128).
kernel void causal_attention_prefill_long_f32(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint4& dims [[buffer(4)]],
    constant float2& scale_params [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 tile [[threadgroup_position_in_grid]]
) {
    const uint seq = dims.x;
    const uint q_heads = dims.y;
    const uint kv_heads = dims.z;
    const uint head_dim = dims.w;
    const uint q_head = tile.x;
    const uint pos = tile.y;
    if (q_head >= q_heads || pos >= seq || head_dim > 256u) {
        return;
    }

    threadgroup float partial[256];
    const uint kv_group = q_heads / kv_heads;
    const uint kv_head = q_head / kv_group;
    const uint q_dim = q_heads * head_dim;
    const uint kv_dim = kv_heads * head_dim;
    const uint q_start = pos * q_dim + q_head * head_dim;
    const uint kv_head_start = kv_head * head_dim;
    const float scale = scale_params.y > 0.0f ? scale_params.x : rsqrt(float(head_dim));

    // Passe 1 : maximum des scores (ordre 0..=pos), score recalculé en arbre.
    float max_score = -INFINITY;
    for (uint row = 0u; row <= pos; ++row) {
        const uint k_start = row * kv_dim + kv_head_start;
        float dot = 0.0f;
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
        max_score = max(max_score, partial[0] * scale);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Passe 2 : somme des exponentielles (même ordre), score recalculé.
    float sum = 0.0f;
    for (uint row = 0u; row <= pos; ++row) {
        const uint k_start = row * kv_dim + kv_head_start;
        float dot = 0.0f;
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
        sum += exp(partial[0] * scale - max_score);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;

    // Passe 3 : sortie = Σ (exp(score-max) * inv_sum) * v, une colonne par thread.
    const uint out_start = pos * q_dim + q_head * head_dim;
    const uint my_col = tid;
    float acc = 0.0f;
    for (uint row = 0u; row <= pos; ++row) {
        const uint k_start = row * kv_dim + kv_head_start;
        float dot = 0.0f;
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
        const float weight = exp(partial[0] * scale - max_score) * inv_sum;
        if (my_col < head_dim) {
            const uint v_start = row * kv_dim + kv_head_start;
            acc += weight * v[v_start + my_col];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (my_col < head_dim) {
        out[out_start + my_col] = acc;
    }
}

// Variantes sliding-window du prefill causal. Elles restent séparées des
// kernels historiques afin que les chemins Qwen et Gemma globaux conservent
// strictement leur code et leur ordre arithmétique.
kernel void windowed_attention_prefill_f32(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint* dims [[buffer(4)]],
    constant float2& scale_params [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 tile [[threadgroup_position_in_grid]]
) {
    const uint seq = dims[0];
    const uint q_heads = dims[1];
    const uint kv_heads = dims[2];
    const uint head_dim = dims[3];
    const uint window = dims[4];
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
    const uint row_start = (pos + 1u > window) ? (pos + 1u - window) : 0u;
    const float scale = scale_params.y > 0.0f ? scale_params.x : rsqrt(float(head_dim));

    for (uint row = row_start; row <= pos; ++row) {
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
        float max_score = scores[row_start];
        for (uint row = row_start + 1u; row <= pos; ++row) {
            max_score = max(max_score, scores[row]);
        }
        float sum = 0.0f;
        for (uint row = row_start; row <= pos; ++row) {
            const float value = exp(scores[row] - max_score);
            scores[row] = value;
            sum += value;
        }
        const float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
        for (uint row = row_start; row <= pos; ++row) {
            scores[row] *= inv_sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint out_start = pos * q_dim + q_head * head_dim;
    for (uint col = tid; col < head_dim; col += 256u) {
        float acc = 0.0f;
        for (uint row = row_start; row <= pos; ++row) {
            const uint v_start = row * kv_dim + kv_head_start;
            acc += scores[row] * v[v_start + col];
        }
        out[out_start + col] = acc;
    }
}

kernel void windowed_attention_prefill_mid_f32(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint* dims [[buffer(4)]],
    constant float2& scale_params [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 tile [[threadgroup_position_in_grid]]
) {
    const uint seq = dims[0];
    const uint q_heads = dims[1];
    const uint kv_heads = dims[2];
    const uint head_dim = dims[3];
    const uint window = dims[4];
    const uint q_head = tile.x;
    const uint pos = tile.y;
    if (q_head >= q_heads || pos >= seq || seq > 2048u) {
        return;
    }

    threadgroup float partial[256];
    threadgroup float scores[2048];
    const uint kv_group = q_heads / kv_heads;
    const uint kv_head = q_head / kv_group;
    const uint q_dim = q_heads * head_dim;
    const uint kv_dim = kv_heads * head_dim;
    const uint q_start = pos * q_dim + q_head * head_dim;
    const uint kv_head_start = kv_head * head_dim;
    const uint row_start = (pos + 1u > window) ? (pos + 1u - window) : 0u;
    const float scale = scale_params.y > 0.0f ? scale_params.x : rsqrt(float(head_dim));

    for (uint row = row_start; row <= pos; ++row) {
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
        float max_score = scores[row_start];
        for (uint row = row_start + 1u; row <= pos; ++row) {
            max_score = max(max_score, scores[row]);
        }
        float sum = 0.0f;
        for (uint row = row_start; row <= pos; ++row) {
            const float value = exp(scores[row] - max_score);
            scores[row] = value;
            sum += value;
        }
        const float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
        for (uint row = row_start; row <= pos; ++row) {
            scores[row] *= inv_sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint out_start = pos * q_dim + q_head * head_dim;
    for (uint col = tid; col < head_dim; col += 256u) {
        float acc = 0.0f;
        for (uint row = row_start; row <= pos; ++row) {
            const uint v_start = row * kv_dim + kv_head_start;
            acc += scores[row] * v[v_start + col];
        }
        out[out_start + col] = acc;
    }
}

kernel void windowed_attention_prefill_long_f32(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint* dims [[buffer(4)]],
    constant float2& scale_params [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 tile [[threadgroup_position_in_grid]]
) {
    const uint seq = dims[0];
    const uint q_heads = dims[1];
    const uint kv_heads = dims[2];
    const uint head_dim = dims[3];
    const uint window = dims[4];
    const uint q_head = tile.x;
    const uint pos = tile.y;
    if (q_head >= q_heads || pos >= seq || head_dim > 256u) {
        return;
    }

    threadgroup float partial[256];
    const uint kv_group = q_heads / kv_heads;
    const uint kv_head = q_head / kv_group;
    const uint q_dim = q_heads * head_dim;
    const uint kv_dim = kv_heads * head_dim;
    const uint q_start = pos * q_dim + q_head * head_dim;
    const uint kv_head_start = kv_head * head_dim;
    const uint row_start = (pos + 1u > window) ? (pos + 1u - window) : 0u;
    const float scale = scale_params.y > 0.0f ? scale_params.x : rsqrt(float(head_dim));

    float max_score = -INFINITY;
    for (uint row = row_start; row <= pos; ++row) {
        const uint k_start = row * kv_dim + kv_head_start;
        float dot = 0.0f;
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
        max_score = max(max_score, partial[0] * scale);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float sum = 0.0f;
    for (uint row = row_start; row <= pos; ++row) {
        const uint k_start = row * kv_dim + kv_head_start;
        float dot = 0.0f;
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
        sum += exp(partial[0] * scale - max_score);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;

    const uint out_start = pos * q_dim + q_head * head_dim;
    const uint my_col = tid;
    float acc = 0.0f;
    for (uint row = row_start; row <= pos; ++row) {
        const uint k_start = row * kv_dim + kv_head_start;
        float dot = 0.0f;
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
        const float weight = exp(partial[0] * scale - max_score) * inv_sum;
        if (my_col < head_dim) {
            const uint v_start = row * kv_dim + kv_head_start;
            acc += weight * v[v_start + my_col];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (my_col < head_dim) {
        out[out_start + my_col] = acc;
    }
}

// D-PLONG-B : variante longue opt-in pour les modèles 27B/30B. Une simdgroup
// calcule une requête causale complète (position, q_head) avec online-softmax :
// Q/K/V ont déjà reçu RMSNorm+RoPE côté encodeur Rust. L'ordre de réduction
// change par rapport au fallback long byte-identique, donc le dispatch reste
// strictement gaté par RETI_RUST_PREFILL_ATTN_BATCH_LONG.
kernel void causal_attention_prefill_batch_long_d128_f32(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint4& dims [[buffer(4)]],
    constant float2& scale_params [[buffer(5)]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint2 tile [[threadgroup_position_in_grid]]
) {
    constexpr uint BD = 32u;
    constexpr uint D = 128u;
    constexpr uint QK = D / BD;
    const uint seq = dims.x;
    const uint q_heads = dims.y;
    const uint kv_heads = dims.z;
    const uint head_dim = dims.w;
    const uint q_head = tile.x;
    const uint pos = tile.y;
    if (head_dim != D || q_heads == 0u || kv_heads == 0u || (q_heads % kv_heads) != 0u ||
        q_head >= q_heads || pos >= seq) {
        return;
    }

    const uint kv_group = q_heads / kv_heads;
    const uint kv_head = q_head / kv_group;
    const uint q_dim = q_heads * D;
    const uint kv_dim = kv_heads * D;
    const uint lane_base = simd_lid * QK;
    const float scale = scale_params.y > 0.0f ? scale_params.x : rsqrt(float(D));

    float ql[QK];
    const device float* q_ptr = q + pos * q_dim + q_head * D + lane_base;
    for (uint j = 0u; j < QK; ++j) {
        ql[j] = q_ptr[j] * scale;
    }

    float o[QK];
    for (uint j = 0u; j < QK; ++j) {
        o[j] = 0.0f;
    }
    float max_score = -INFINITY;
    float sum_exp = 0.0f;

    for (uint row = 0u; row <= pos; ++row) {
        const device float* k_ptr = k + row * kv_dim + kv_head * D + lane_base;
        const device float* v_ptr = v + row * kv_dim + kv_head * D + lane_base;
        float score = 0.0f;
        for (uint j = 0u; j < QK; ++j) {
            score += ql[j] * k_ptr[j];
        }
        score = simd_sum(score);
        const float new_max = max(max_score, score);
        const float factor = fast::exp(max_score - new_max);
        const float exp_score = fast::exp(score - new_max);
        max_score = new_max;
        sum_exp = sum_exp * factor + exp_score;
        for (uint j = 0u; j < QK; ++j) {
            o[j] = o[j] * factor + exp_score * v_ptr[j];
        }
    }

    const float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    device float* out_ptr = out + pos * q_dim + q_head * D + lane_base;
    for (uint j = 0u; j < QK; ++j) {
        out_ptr[j] = o[j] * inv_sum;
    }
}

kernel void causal_attention_prefill_batch_long_d256_f32(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint4& dims [[buffer(4)]],
    constant float2& scale_params [[buffer(5)]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint2 tile [[threadgroup_position_in_grid]]
) {
    constexpr uint BD = 32u;
    constexpr uint D = 256u;
    constexpr uint QK = D / BD;
    const uint seq = dims.x;
    const uint q_heads = dims.y;
    const uint kv_heads = dims.z;
    const uint head_dim = dims.w;
    const uint q_head = tile.x;
    const uint pos = tile.y;
    if (head_dim != D || q_heads == 0u || kv_heads == 0u || (q_heads % kv_heads) != 0u ||
        q_head >= q_heads || pos >= seq) {
        return;
    }

    const uint kv_group = q_heads / kv_heads;
    const uint kv_head = q_head / kv_group;
    const uint q_dim = q_heads * D;
    const uint kv_dim = kv_heads * D;
    const uint lane_base = simd_lid * QK;
    const float scale = scale_params.y > 0.0f ? scale_params.x : rsqrt(float(D));

    float ql[QK];
    const device float* q_ptr = q + pos * q_dim + q_head * D + lane_base;
    for (uint j = 0u; j < QK; ++j) {
        ql[j] = q_ptr[j] * scale;
    }

    float o[QK];
    for (uint j = 0u; j < QK; ++j) {
        o[j] = 0.0f;
    }
    float max_score = -INFINITY;
    float sum_exp = 0.0f;

    for (uint row = 0u; row <= pos; ++row) {
        const device float* k_ptr = k + row * kv_dim + kv_head * D + lane_base;
        const device float* v_ptr = v + row * kv_dim + kv_head * D + lane_base;
        float score = 0.0f;
        for (uint j = 0u; j < QK; ++j) {
            score += ql[j] * k_ptr[j];
        }
        score = simd_sum(score);
        const float new_max = max(max_score, score);
        const float factor = fast::exp(max_score - new_max);
        const float exp_score = fast::exp(score - new_max);
        max_score = new_max;
        sum_exp = sum_exp * factor + exp_score;
        for (uint j = 0u; j < QK; ++j) {
            o[j] = o[j] * factor + exp_score * v_ptr[j];
        }
    }

    const float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    device float* out_ptr = out + pos * q_dim + q_head * D + lane_base;
    for (uint j = 0u; j < QK; ++j) {
        out_ptr[j] = o[j] * inv_sum;
    }
}

// Variante 35B d256 GQA8x4 : meme ordre online-softmax par requete que
// le kernel d256 precedent, mais un threadgroup calcule 4 positions
// consecutives. Chaque ligne K/V est chargee une fois pour 8 tetes Q et
// 4 positions Q.
kernel void causal_attention_prefill_batch_gqa8x4_d256_f32(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint4& dims [[buffer(4)]],
    constant float2& scale_params [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint2 tile [[threadgroup_position_in_grid]]
) {
    constexpr uint D = 256u;
    constexpr uint QK = 8u;
    constexpr uint GQA = 8u;
    constexpr uint BQ = 4u;
    const uint seq = dims.x;
    const uint q_heads = dims.y;
    const uint kv_heads = dims.z;
    const uint head_dim = dims.w;
    const uint kv_head = tile.x;
    const uint base_pos = tile.y * BQ;
    if (head_dim != D || kv_heads == 0u || q_heads != kv_heads * GQA ||
        kv_head >= kv_heads || base_pos >= seq || simd_gid >= GQA) {
        return;
    }

    threadgroup float k_tile[D];
    threadgroup float v_tile[D];
    const uint q_head = kv_head * GQA + simd_gid;
    const uint q_dim = q_heads * D;
    const uint kv_dim = kv_heads * D;
    const uint lane_base = simd_lid * QK;
    const float scale = scale_params.y > 0.0f ? scale_params.x : rsqrt(float(D));

    float q0[QK], q1[QK], q2[QK], q3[QK];
    float o0[QK], o1[QK], o2[QK], o3[QK];
    for (uint j = 0u; j < QK; ++j) {
        q0[j] = q[(base_pos + 0u) * q_dim + q_head * D + lane_base + j] * scale;
        q1[j] = (base_pos + 1u < seq)
            ? q[(base_pos + 1u) * q_dim + q_head * D + lane_base + j] * scale
            : 0.0f;
        q2[j] = (base_pos + 2u < seq)
            ? q[(base_pos + 2u) * q_dim + q_head * D + lane_base + j] * scale
            : 0.0f;
        q3[j] = (base_pos + 3u < seq)
            ? q[(base_pos + 3u) * q_dim + q_head * D + lane_base + j] * scale
            : 0.0f;
        o0[j] = 0.0f;
        o1[j] = 0.0f;
        o2[j] = 0.0f;
        o3[j] = 0.0f;
    }
    float max0 = -INFINITY, max1 = -INFINITY, max2 = -INFINITY, max3 = -INFINITY;
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    const uint last_pos = min(seq - 1u, base_pos + BQ - 1u);

    for (uint row = 0u; row <= last_pos; ++row) {
        const uint kv_base = row * kv_dim + kv_head * D;
        if (tid < D) {
            k_tile[tid] = k[kv_base + tid];
            v_tile[tid] = v[kv_base + tid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (row <= base_pos + 0u) {
            float score = 0.0f;
            for (uint j = 0u; j < QK; ++j) score += q0[j] * k_tile[lane_base + j];
            score = simd_sum(score);
            const float new_max = max(max0, score);
            const float factor = fast::exp(max0 - new_max);
            const float exp_score = fast::exp(score - new_max);
            max0 = new_max;
            sum0 = sum0 * factor + exp_score;
            for (uint j = 0u; j < QK; ++j) o0[j] = o0[j] * factor + exp_score * v_tile[lane_base + j];
        }
        if (base_pos + 1u < seq && row <= base_pos + 1u) {
            float score = 0.0f;
            for (uint j = 0u; j < QK; ++j) score += q1[j] * k_tile[lane_base + j];
            score = simd_sum(score);
            const float new_max = max(max1, score);
            const float factor = fast::exp(max1 - new_max);
            const float exp_score = fast::exp(score - new_max);
            max1 = new_max;
            sum1 = sum1 * factor + exp_score;
            for (uint j = 0u; j < QK; ++j) o1[j] = o1[j] * factor + exp_score * v_tile[lane_base + j];
        }
        if (base_pos + 2u < seq && row <= base_pos + 2u) {
            float score = 0.0f;
            for (uint j = 0u; j < QK; ++j) score += q2[j] * k_tile[lane_base + j];
            score = simd_sum(score);
            const float new_max = max(max2, score);
            const float factor = fast::exp(max2 - new_max);
            const float exp_score = fast::exp(score - new_max);
            max2 = new_max;
            sum2 = sum2 * factor + exp_score;
            for (uint j = 0u; j < QK; ++j) o2[j] = o2[j] * factor + exp_score * v_tile[lane_base + j];
        }
        if (base_pos + 3u < seq && row <= base_pos + 3u) {
            float score = 0.0f;
            for (uint j = 0u; j < QK; ++j) score += q3[j] * k_tile[lane_base + j];
            score = simd_sum(score);
            const float new_max = max(max3, score);
            const float factor = fast::exp(max3 - new_max);
            const float exp_score = fast::exp(score - new_max);
            max3 = new_max;
            sum3 = sum3 * factor + exp_score;
            for (uint j = 0u; j < QK; ++j) o3[j] = o3[j] * factor + exp_score * v_tile[lane_base + j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv0 = (sum0 > 0.0f) ? (1.0f / sum0) : 0.0f;
    device float* out0 = out + (base_pos + 0u) * q_dim + q_head * D + lane_base;
    for (uint j = 0u; j < QK; ++j) out0[j] = o0[j] * inv0;
    if (base_pos + 1u < seq) {
        const float inv1 = (sum1 > 0.0f) ? (1.0f / sum1) : 0.0f;
        device float* out1 = out + (base_pos + 1u) * q_dim + q_head * D + lane_base;
        for (uint j = 0u; j < QK; ++j) out1[j] = o1[j] * inv1;
    }
    if (base_pos + 2u < seq) {
        const float inv2 = (sum2 > 0.0f) ? (1.0f / sum2) : 0.0f;
        device float* out2 = out + (base_pos + 2u) * q_dim + q_head * D + lane_base;
        for (uint j = 0u; j < QK; ++j) out2[j] = o2[j] * inv2;
    }
    if (base_pos + 3u < seq) {
        const float inv3 = (sum3 > 0.0f) ? (1.0f / sum3) : 0.0f;
        device float* out3 = out + (base_pos + 3u) * q_dim + q_head * D + lane_base;
        for (uint j = 0u; j < QK; ++j) out3[j] = o3[j] * inv3;
    }
}

kernel void noncausal_attention_prefill_f32(
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
    if (q_head >= q_heads || pos >= seq || seq > 2048u || head_dim > 256u) {
        return;
    }

    // Score row-parallèle : chaque thread traite un sous-ensemble de lignes KV et
    // calcule le dot COMPLET (head_dim) sans barrière. L'ancien kernel réduisait
    // chaque score sur 256 threads (8 barrières) × seq lignes → ~12 k barrières
    // par threadgroup (30 k threadgroups × 32 couches). Max/sum du softmax = UNE
    // réduction parallèle chacune (et non une boucle sérielle de seq).
    threadgroup float partial[256];
    threadgroup float scores[2048];
    threadgroup float q_tile[256];
    const uint kv_group = q_heads / kv_heads;
    const uint kv_head = q_head / kv_group;
    const uint q_dim = q_heads * head_dim;
    const uint kv_dim = kv_heads * head_dim;
    const uint q_start = pos * q_dim + q_head * head_dim;
    const uint kv_head_start = kv_head * head_dim;
    const float scale = rsqrt(float(head_dim));

    for (uint col = tid; col < head_dim; col += 256u) {
        q_tile[col] = q[q_start + col];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint row = tid; row < seq; row += 256u) {
        const uint k_start = row * kv_dim + kv_head_start;
        float dot = 0.0f;
        for (uint col = 0u; col < head_dim; ++col) {
            dot += q_tile[col] * k[k_start + col];
        }
        scores[row] = dot * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_max = -INFINITY;
    for (uint row = tid; row < seq; row += 256u) {
        local_max = max(local_max, scores[row]);
    }
    partial[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial[tid] = max(partial[tid], partial[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float max_score = partial[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_sum = 0.0f;
    for (uint row = tid; row < seq; row += 256u) {
        const float value = exp(scores[row] - max_score);
        scores[row] = value;
        local_sum += value;
    }
    partial[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float sum = partial[0];
    const float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint out_start = pos * q_dim + q_head * head_dim;
    for (uint col = tid; col < head_dim; col += 256u) {
        float acc = 0.0f;
        for (uint row = 0u; row < seq; ++row) {
            const uint v_start = row * kv_dim + kv_head_start;
            acc += scores[row] * v[v_start + col];
        }
        out[out_start + col] = acc * inv_sum;
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

kernel void topk8_softmax_256_f32(
    device const float* logits [[buffer(0)]],
    device uint* indices [[buffer(1)]],
    device float* scores [[buffer(2)]],
    constant uint2& dims [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]]
) {
    const uint count = dims.x;
    const uint topk = dims.y;
    if (count != 256u || topk != 8u) {
        return;
    }
    threadgroup float partial_values[32u * 8u];
    threadgroup uint partial_indices[32u * 8u];

    float best_values[8];
    uint best_indices[8];
    for (uint slot = 0u; slot < 8u; ++slot) {
        best_values[slot] = -3.402823466e38f;
        best_indices[slot] = 0xffffffffu;
    }

    for (uint idx = tid; idx < 256u; idx += 32u) {
        const float value = logits[idx];
        for (uint slot = 0u; slot < 8u; ++slot) {
            if (topk_better(value, idx, best_values[slot], best_indices[slot])) {
                for (uint shift = 7u; shift > slot; --shift) {
                    best_values[shift] = best_values[shift - 1u];
                    best_indices[shift] = best_indices[shift - 1u];
                }
                best_values[slot] = value;
                best_indices[slot] = idx;
                break;
            }
        }
    }

    const uint local_base = tid * 8u;
    for (uint slot = 0u; slot < 8u; ++slot) {
        partial_values[local_base + slot] = best_values[slot];
        partial_indices[local_base + slot] = best_indices[slot];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 16u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            const uint left_base = tid * 8u;
            const uint right_base = (tid + stride) * 8u;
            float merged_values[8];
            uint merged_indices[8];
            for (uint slot = 0u; slot < 8u; ++slot) {
                merged_values[slot] = partial_values[left_base + slot];
                merged_indices[slot] = partial_indices[left_base + slot];
            }
            for (uint src = 0u; src < 8u; ++src) {
                const float value = partial_values[right_base + src];
                const uint index = partial_indices[right_base + src];
                for (uint slot = 0u; slot < 8u; ++slot) {
                    if (topk_better(value, index, merged_values[slot], merged_indices[slot])) {
                        for (uint shift = 7u; shift > slot; --shift) {
                            merged_values[shift] = merged_values[shift - 1u];
                            merged_indices[shift] = merged_indices[shift - 1u];
                        }
                        merged_values[slot] = value;
                        merged_indices[slot] = index;
                        break;
                    }
                }
            }
            for (uint slot = 0u; slot < 8u; ++slot) {
                partial_values[left_base + slot] = merged_values[slot];
                partial_indices[left_base + slot] = merged_indices[slot];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        const float max_value = partial_values[0];
        float denom = 0.0f;
        for (uint slot = 0u; slot < 8u; ++slot) {
            denom += exp(partial_values[slot] - max_value);
        }
        denom = max(denom, 1.0e-20f);
        for (uint slot = 0u; slot < 8u; ++slot) {
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

static inline float splitmix_gumbel_f32(ulong rng_state, uint index) {
    const ulong mixed = rng_state ^ (ulong(index) * 0xD1B54A32D192ED03ul);
    float u = splitmix_unit_f32(mixed);
    u = fmin(fmax(u, 5.960464477539063e-8f), 0.9999999403953552f);
    return -log(-log(u));
}

kernel void sample_gumbel_blocks_f32(
    device const float* logits [[buffer(0)]],
    device float* partial_values [[buffer(1)]],
    device uint* partial_indices [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    constant float& temperature_in [[buffer(4)]],
    constant ulong& rng_state [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]
) {
    threadgroup float values[256];
    threadgroup uint indices[256];

    const uint idx = group * 256u + tid;
    const float temperature = max(temperature_in, 0.0001f);
    float value = -3.402823466e38f;
    uint best_idx = 0u;
    if (idx < count) {
        value = logits[idx] / temperature + splitmix_gumbel_f32(rng_state, idx);
        best_idx = idx;
    }
    values[tid] = value;
    indices[tid] = best_idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            const float other_value = values[tid + stride];
            const uint other_index = indices[tid + stride];
            if (topk_better(other_value, other_index, values[tid], indices[tid])) {
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
    threadgroup float best_values[256];
    threadgroup uint best_indices[256];

    const uint idx = group * 256u + tid;
    values[tid] = (idx < count) ? logits[idx] : -3.402823466e38f;
    indices[tid] = idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint out_base = group * topk;
    for (uint slot = 0u; slot < topk; ++slot) {
        best_values[tid] = values[tid];
        best_indices[tid] = indices[tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = 128u; stride > 0u; stride >>= 1u) {
            if (tid < stride) {
                const float other_value = best_values[tid + stride];
                const uint other_index = best_indices[tid + stride];
                if (topk_better(other_value, other_index, best_values[tid], best_indices[tid])) {
                    best_values[tid] = other_value;
                    best_indices[tid] = other_index;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        const uint selected = best_indices[0];
        if (tid == 0u) {
            partial_values[out_base + slot] = best_values[0];
            partial_indices[out_base + slot] = selected;
        }
        if (indices[tid] == selected) {
            values[tid] = -3.402823466e38f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
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
    const uint width = 128u;
    const uint max_topk = 32u;
    threadgroup float merged_partial_values[128u * 32u];
    threadgroup uint merged_partial_indices[128u * 32u];

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

// Argmax greedy du talker TTS (cb0). Réplique EXACTEMENT `greedy_talker_token`
// (CPU) : (1) supprime la plage [suppress_start, count) SAUF `eos` ; (2) quantifie
// les valeurs finies par floor(x*4)/4 (granularité tie-break MLX, `mlx_greedy_logit`)
// et laisse les non-finies telles quelles ; (3) argmax tie-break index le plus bas.
// Un seul threadgroup (vocab talker = 3072), grid-stride sur 256 threads. `floor`
// et le scale par 4/0.25 sont exacts en IEEE754 → bit-identique au CPU.
kernel void talker_greedy_argmax_f32(
    device const float* logits [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    constant uint& suppress_start [[buffer(2)]],
    constant uint& eos [[buffer(3)]],
    device uint* out_index [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float values[256];
    threadgroup uint indices[256];

    float best_value = -3.402823466e38f;
    uint best_index = 0u;
    for (uint idx = tid; idx < count; idx += 256u) {
        if (idx >= suppress_start && idx != eos) {
            continue;
        }
        float value = logits[idx];
        if (isfinite(value)) {
            value = floor(value * 4.0f) * 0.25f;
        }
        if (value > best_value || (value == best_value && idx < best_index)) {
            best_value = value;
            best_index = idx;
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
