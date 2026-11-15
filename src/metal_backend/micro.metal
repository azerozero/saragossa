// Codes MLX : E2M1, E4M3 (0x7f = 480), échelles E8M0 ou E4M3.
inline float micro_e2m1(uint q) {
    float magnitude = float(as_type<half>(ushort((q & 7u) << 9))) * 16384.0f;
    return (q & 8u) ? -magnitude : magnitude;
}
inline float micro_e4m3(uint q) {
    float v = float(as_type<half>(ushort((q & 127u) << 7))) * 256.0f;
    return (q & 128u) ? -v : v;
}
template<uint G> inline float micro_scale(uchar q) {
    if (G == 16) return micro_e4m3(q);
    return as_type<float>(q == 0 ? 0x00400000u : uint(q) << 23);
}
template<uint B> inline float micro_code(uint q) {
    if (B == 4) return micro_e2m1(q);
    return micro_e4m3(q);
}
// Metal flush les sous-normaux f32 pendant une multiplication. Les échelles
// extrêmes E8M0 se reconstruisent donc par exposant entier, sans arithmétique
// flottante. Les codes E2M1/E4M3 ont au plus quatre bits significatifs : même
// le décalage maximal (10 bits) reste exact dans le significand f32.
template<uint B, uint G> inline float micro_scaled(uint q, uchar scale) {
    float value = micro_code<B>(q);
    if (G == 16 || (scale >= 10 && scale <= 246)) {
        return value * micro_scale<G>(scale);
    }
    uint bits = as_type<uint>(value), magnitude = bits & 0x7fffffffu;
    uint sign = bits & 0x80000000u;
    if (scale == 255) return value * INFINITY;
    if (magnitude == 0) return value;
    int exponent = int(magnitude >> 23) + int(scale) - 127;
    if (exponent >= 255) return as_type<float>(sign | 0x7f800000u);
    uint fraction = magnitude & 0x007fffffu;
    if (exponent > 0) return as_type<float>(sign | (uint(exponent) << 23) | fraction);
    return as_type<float>(sign | ((fraction | 0x00800000u) >> uint(1 - exponent)));
}
template<uint B, uint G> inline float micro_value(device const uint* w, device const uchar* scales, size_t i) {
    constexpr uint P = 32 / B;
    uint q = (w[i / P] >> ((i % P) * B)) & ((1u << B) - 1u);
    return micro_scaled<B, G>(q, scales[i / G]);
}

// Le store NAX partagé ne masque que M : avec N non multiple de 64, les
// colonnes du dernier bloc déborderaient sur la ligne suivante.
inline void micro_store_c(const thread metal::vec<float, 8>& src, device float* dst,
                          uint m_base, uint n_base, uint N, uint M) {
    short2 sc = NaxFrag::get_coord();
    for (short i = 0; i < 2; i++) {
        uint row = m_base + uint(sc.y) + uint(i) * 8u;
        if (row >= M) continue;
        for (short j = 0; j < 4; j++) {
            uint col = n_base + uint(sc.x) + uint(j);
            if (col < N) dst[size_t(row) * N + col] = src[i * 4 + j];
        }
    }
}

// Une ligne par simdgroup, un mot U32 par lane et par pas. L'échelle est
// décodée une fois par mot : dans la plage normale, `code * échelle` est le
// même produit f32 exact que `micro_scaled`, et l'ordre des additions ne
// change pas. Les échelles extrêmes gardent le chemin entier de `micro_scaled`.
template<uint B, uint G>
inline void micro_qmv(device const float* x, device const uint* w, device const uchar* scales,
                      device float* y, constant uint3& dims, uint2 tg, uint sg, uint lane) {
    constexpr uint P = 32 / B;
    uint row = tg.x * 4 + sg, K = dims.z;
    if (row >= dims.y) return;
    device const uint* wr = w + size_t(row) * (K / P);
    device const uchar* sr = scales + size_t(row) * (K / G);
    device const float* xr = x + size_t(tg.y) * K;
    float acc = 0;
    for (uint k = lane * P; k < K; k += 32 * P) {
        uint word = wr[k / P];
        uchar scale = sr[k / G];
        device const float4* xv = (device const float4*)(xr + k);
        if (G == 16 || (scale >= 10 && scale <= 246)) {
            float s = micro_scale<G>(scale);
            for (uint v = 0; v < P / 4; ++v) {
                float4 xs = xv[v];
                for (uint j = 0; j < 4; ++j) {
                    uint q = (word >> ((v * 4 + j) * B)) & ((1u << B) - 1u);
                    acc += xs[j] * (micro_code<B>(q) * s);
                }
            }
        } else {
            for (uint j = 0; j < P; ++j) {
                float value = micro_scaled<B, G>((word >> (j * B)) & ((1u << B) - 1u), scale);
                acc += xr[k + j] * value;
            }
        }
    }
    acc = simd_sum(acc);
    if (lane == 0) y[size_t(tg.y) * dims.y + row] = acc;
}

// Chaque thread charge une demi-ligne (32 poids) et écrit quatre vecteurs
// bf16 alignés. K est multiple de 64 ; le masque N traite les dernières lignes.
template<uint B, uint G>
inline void micro_load_b(threadgroup bfloat* tile, device const uint* w,
                         device const uchar* scales, uint n0, uint k0,
                         uint N, uint K, uint lid) {
    uint row = n0 + lid / 2u;
    uint col = k0 + (lid & 1u) * 32u;
    threadgroup metal::vec<bfloat, 8>* dst =
        (threadgroup metal::vec<bfloat, 8>*)(tile + (lid / 2u) * 72u + (lid & 1u) * 32u);
    if (row >= N) {
        for (uint block = 0; block < 4; ++block) dst[block] = metal::vec<bfloat, 8>(bfloat(0));
        return;
    }
    constexpr uint P = 32 / B;
    size_t offset = size_t(row) * K + col;
    uchar s0 = scales[offset / G];
    uchar s1 = G == 16 ? scales[offset / G + 1] : s0;
    for (uint block = 0; block < 4; ++block) {
        uint word0 = w[offset / P + block * (8 / P)];
        uint word1 = B == 8 ? w[offset / P + block * 2 + 1] : word0;
        uchar scale = block < 2 ? s0 : s1;
        metal::vec<bfloat, 8> out;
        for (uint j = 0; j < 8; ++j) {
            uint word = j < P ? word0 : word1;
            out[j] = bfloat(micro_scaled<B, G>((word >> ((j % P) * B)) & ((1u << B) - 1u), scale));
        }
        dst[block] = out;
    }
}

kernel void micro_qmv_mxfp4(device const float* x [[buffer(0)]], device const uint* w [[buffer(1)]], device const uchar* scales [[buffer(2)]], device float* y [[buffer(4)]], constant uint3& dims [[buffer(5)]], uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    micro_qmv<4, 32>(x, w, scales, y, dims, tg, sg, lane);
}
kernel void micro_embed_mxfp4(device const uint* w [[buffer(0)]], device const uchar* scales [[buffer(1)]], device const uint* ids [[buffer(2)]], device float* y [[buffer(3)]], constant uint2& dims [[buffer(4)]], constant float& factor [[buffer(5)]], constant uint& recast [[buffer(6)]], uint2 i [[thread_position_in_grid]]) {
    uint row = ids[i.y];
    float value = row < dims.y ? micro_value<4, 32>(w, scales, size_t(row) * dims.x + i.x) : 0.0f;
    if (factor != 1.0f) value *= factor;
    y[size_t(i.y) * dims.x + i.x] = recast ? float(bfloat(value)) : value;
}
kernel void micro_dequant_mxfp4(device const uint* w [[buffer(0)]], device const uchar* scales [[buffer(1)]], device float* y [[buffer(2)]], constant uint& count [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i < count) y[i] = micro_value<4, 32>(w, scales, i);
}
kernel void micro_gemm_mxfp4(device const bfloat* A [[buffer(0)]],
                                      device const uint* Bp   [[buffer(1)]],
                                      device const uchar* Bs [[buffer(2)]],
                                      device float* C         [[buffer(4)]],
                                      constant uint3& mnk     [[buffer(5)]],
                                      uint2 tgid [[threadgroup_position_in_grid]],
                                      uint lid [[thread_index_in_threadgroup]],
                                      uint sgid [[simdgroup_index_in_threadgroup]]) {
  uint M = mnk.x, N = mnk.y, K = mnk.z;
  constexpr uint BN = 64, BK = 64, BKP = BK + 8, WM = 2, WN = 2;
  threadgroup metal::vec<bfloat,8> Ws8[BN * BKP / 8];
  threadgroup bfloat* Ws = (threadgroup bfloat*)Ws8;
  uint sg_m = sgid / WN;
  uint sg_n = sgid - sg_m * WN;
  uint m0 = tgid.x * 64u + sg_m * 32u;
  uint n0 = tgid.y * BN;
  uint ns = sg_n * 32u;
  metal::vec<float,8> c0a = float(0), c0b = float(0), c1a = float(0), c1b = float(0);
  for (uint k = 0; k < K; k += BK) {
    micro_load_b<4, 32>(Ws, Bp, Bs, n0, k, N, K, lid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kk = 0; kk < BK; kk += 16u) {
      metal::vec<bfloat,8> af0, af1, b0, b1;
      if (m0 + 32u <= M) {
        NaxFrag::load_a(af0, A + m0 * K + k + kk, int(K));
        NaxFrag::load_a(af1, A + (m0 + 16u) * K + k + kk, int(K));
      } else {
        NaxFrag::load_a_masked(af0, A, m0, k + kk, K, M);
        NaxFrag::load_a_masked(af1, A, m0 + 16u, k + kk, K, M);
      }
      NaxFrag::load_a_tg(b0, Ws + ns * BKP + kk, int(BKP));
      NaxFrag::load_a_tg(b1, Ws + (ns + 16u) * BKP + kk, int(BKP));
      NaxFrag::mma(c0a, c0b, af0, b0, b1);
      NaxFrag::mma(c1a, c1b, af1, b0, b1);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  micro_store_c(c0a, C, m0, n0 + ns, N, M);
  micro_store_c(c0b, C, m0, n0 + ns + 16u, N, M);
  micro_store_c(c1a, C, m0 + 16u, n0 + ns, N, M);
  micro_store_c(c1b, C, m0 + 16u, n0 + ns + 16u, N, M);
}

kernel void micro_qmv_nvfp4(device const float* x [[buffer(0)]], device const uint* w [[buffer(1)]], device const uchar* scales [[buffer(2)]], device float* y [[buffer(4)]], constant uint3& dims [[buffer(5)]], uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    micro_qmv<4, 16>(x, w, scales, y, dims, tg, sg, lane);
}
kernel void micro_embed_nvfp4(device const uint* w [[buffer(0)]], device const uchar* scales [[buffer(1)]], device const uint* ids [[buffer(2)]], device float* y [[buffer(3)]], constant uint2& dims [[buffer(4)]], constant float& factor [[buffer(5)]], constant uint& recast [[buffer(6)]], uint2 i [[thread_position_in_grid]]) {
    uint row = ids[i.y];
    float value = row < dims.y ? micro_value<4, 16>(w, scales, size_t(row) * dims.x + i.x) : 0.0f;
    if (factor != 1.0f) value *= factor;
    y[size_t(i.y) * dims.x + i.x] = recast ? float(bfloat(value)) : value;
}
kernel void micro_dequant_nvfp4(device const uint* w [[buffer(0)]], device const uchar* scales [[buffer(1)]], device float* y [[buffer(2)]], constant uint& count [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i < count) y[i] = micro_value<4, 16>(w, scales, i);
}
kernel void micro_gemm_nvfp4(device const bfloat* A [[buffer(0)]],
                                      device const uint* Bp   [[buffer(1)]],
                                      device const uchar* Bs [[buffer(2)]],
                                      device float* C         [[buffer(4)]],
                                      constant uint3& mnk     [[buffer(5)]],
                                      uint2 tgid [[threadgroup_position_in_grid]],
                                      uint lid [[thread_index_in_threadgroup]],
                                      uint sgid [[simdgroup_index_in_threadgroup]]) {
  uint M = mnk.x, N = mnk.y, K = mnk.z;
  constexpr uint BN = 64, BK = 64, BKP = BK + 8, WM = 2, WN = 2;
  threadgroup metal::vec<bfloat,8> Ws8[BN * BKP / 8];
  threadgroup bfloat* Ws = (threadgroup bfloat*)Ws8;
  uint sg_m = sgid / WN;
  uint sg_n = sgid - sg_m * WN;
  uint m0 = tgid.x * 64u + sg_m * 32u;
  uint n0 = tgid.y * BN;
  uint ns = sg_n * 32u;
  metal::vec<float,8> c0a = float(0), c0b = float(0), c1a = float(0), c1b = float(0);
  for (uint k = 0; k < K; k += BK) {
    micro_load_b<4, 16>(Ws, Bp, Bs, n0, k, N, K, lid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kk = 0; kk < BK; kk += 16u) {
      metal::vec<bfloat,8> af0, af1, b0, b1;
      if (m0 + 32u <= M) {
        NaxFrag::load_a(af0, A + m0 * K + k + kk, int(K));
        NaxFrag::load_a(af1, A + (m0 + 16u) * K + k + kk, int(K));
      } else {
        NaxFrag::load_a_masked(af0, A, m0, k + kk, K, M);
        NaxFrag::load_a_masked(af1, A, m0 + 16u, k + kk, K, M);
      }
      NaxFrag::load_a_tg(b0, Ws + ns * BKP + kk, int(BKP));
      NaxFrag::load_a_tg(b1, Ws + (ns + 16u) * BKP + kk, int(BKP));
      NaxFrag::mma(c0a, c0b, af0, b0, b1);
      NaxFrag::mma(c1a, c1b, af1, b0, b1);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  micro_store_c(c0a, C, m0, n0 + ns, N, M);
  micro_store_c(c0b, C, m0, n0 + ns + 16u, N, M);
  micro_store_c(c1a, C, m0 + 16u, n0 + ns, N, M);
  micro_store_c(c1b, C, m0 + 16u, n0 + ns + 16u, N, M);
}

kernel void micro_qmv_mxfp8(device const float* x [[buffer(0)]], device const uint* w [[buffer(1)]], device const uchar* scales [[buffer(2)]], device float* y [[buffer(4)]], constant uint3& dims [[buffer(5)]], uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    micro_qmv<8, 32>(x, w, scales, y, dims, tg, sg, lane);
}
kernel void micro_embed_mxfp8(device const uint* w [[buffer(0)]], device const uchar* scales [[buffer(1)]], device const uint* ids [[buffer(2)]], device float* y [[buffer(3)]], constant uint2& dims [[buffer(4)]], constant float& factor [[buffer(5)]], constant uint& recast [[buffer(6)]], uint2 i [[thread_position_in_grid]]) {
    uint row = ids[i.y];
    float value = row < dims.y ? micro_value<8, 32>(w, scales, size_t(row) * dims.x + i.x) : 0.0f;
    if (factor != 1.0f) value *= factor;
    y[size_t(i.y) * dims.x + i.x] = recast ? float(bfloat(value)) : value;
}
kernel void micro_dequant_mxfp8(device const uint* w [[buffer(0)]], device const uchar* scales [[buffer(1)]], device float* y [[buffer(2)]], constant uint& count [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i < count) y[i] = micro_value<8, 32>(w, scales, i);
}
kernel void micro_gemm_mxfp8(device const bfloat* A [[buffer(0)]],
                                      device const uint* Bp   [[buffer(1)]],
                                      device const uchar* Bs [[buffer(2)]],
                                      device float* C         [[buffer(4)]],
                                      constant uint3& mnk     [[buffer(5)]],
                                      uint2 tgid [[threadgroup_position_in_grid]],
                                      uint lid [[thread_index_in_threadgroup]],
                                      uint sgid [[simdgroup_index_in_threadgroup]]) {
  uint M = mnk.x, N = mnk.y, K = mnk.z;
  constexpr uint BN = 64, BK = 64, BKP = BK + 8, WM = 2, WN = 2;
  threadgroup metal::vec<bfloat,8> Ws8[BN * BKP / 8];
  threadgroup bfloat* Ws = (threadgroup bfloat*)Ws8;
  uint sg_m = sgid / WN;
  uint sg_n = sgid - sg_m * WN;
  uint m0 = tgid.x * 64u + sg_m * 32u;
  uint n0 = tgid.y * BN;
  uint ns = sg_n * 32u;
  metal::vec<float,8> c0a = float(0), c0b = float(0), c1a = float(0), c1b = float(0);
  for (uint k = 0; k < K; k += BK) {
    micro_load_b<8, 32>(Ws, Bp, Bs, n0, k, N, K, lid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kk = 0; kk < BK; kk += 16u) {
      metal::vec<bfloat,8> af0, af1, b0, b1;
      if (m0 + 32u <= M) {
        NaxFrag::load_a(af0, A + m0 * K + k + kk, int(K));
        NaxFrag::load_a(af1, A + (m0 + 16u) * K + k + kk, int(K));
      } else {
        NaxFrag::load_a_masked(af0, A, m0, k + kk, K, M);
        NaxFrag::load_a_masked(af1, A, m0 + 16u, k + kk, K, M);
      }
      NaxFrag::load_a_tg(b0, Ws + ns * BKP + kk, int(BKP));
      NaxFrag::load_a_tg(b1, Ws + (ns + 16u) * BKP + kk, int(BKP));
      NaxFrag::mma(c0a, c0b, af0, b0, b1);
      NaxFrag::mma(c1a, c1b, af1, b0, b1);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  micro_store_c(c0a, C, m0, n0 + ns, N, M);
  micro_store_c(c0b, C, m0, n0 + ns + 16u, N, M);
  micro_store_c(c1a, C, m0 + 16u, n0 + ns, N, M);
  micro_store_c(c1b, C, m0 + 16u, n0 + ns + 16u, N, M);
}
