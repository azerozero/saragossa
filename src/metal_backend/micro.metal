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

// R lignes par simdgroup, un mot U32 par lane et par pas : le vecteur x est
// lu une fois pour les R lignes (R = 1 retrouve l'ancien noyau). L'échelle est
// décodée une fois par mot : dans la plage normale, `code * échelle` est le
// même produit f32 exact que `micro_scaled`, et l'ordre des additions de chaque
// ligne ne dépend pas de R. Les échelles extrêmes gardent le chemin entier.
template<uint B, uint G, uint R>
inline void micro_qmv(device const float* x, device const uint* w, device const uchar* scales,
                      device float* y, constant uint3& dims, uint2 tg, uint sg, uint lane) {
    constexpr uint P = 32 / B;
    uint row0 = (tg.x * 4 + sg) * R, K = dims.z;
    if (row0 >= dims.y) return;
    uint rows = min(R, dims.y - row0);
    device const float* xr = x + size_t(tg.y) * K;
    float acc[R];
    for (uint r = 0; r < R; ++r) acc[r] = 0;
    for (uint k = lane * P; k < K; k += 32 * P) {
        device const float4* xv = (device const float4*)(xr + k);
        float4 xs[P / 4];
        for (uint v = 0; v < P / 4; ++v) xs[v] = xv[v];
        for (uint r = 0; r < R; ++r) {
            if (r >= rows) break;
            size_t row = row0 + r;
            uint word = w[row * (K / P) + k / P];
            uchar scale = scales[row * (K / G) + k / G];
            if (G == 16 || (scale >= 10 && scale <= 246)) {
                float s = micro_scale<G>(scale);
                for (uint v = 0; v < P / 4; ++v) {
                    for (uint j = 0; j < 4; ++j) {
                        uint q = (word >> ((v * 4 + j) * B)) & ((1u << B) - 1u);
                        acc[r] += xs[v][j] * (micro_code<B>(q) * s);
                    }
                }
            } else {
                for (uint j = 0; j < P; ++j) {
                    float value = micro_scaled<B, G>((word >> (j * B)) & ((1u << B) - 1u), scale);
                    acc[r] += xs[j / 4][j % 4] * value;
                }
            }
        }
    }
    for (uint r = 0; r < R; ++r) {
        float total = simd_sum(acc[r]);
        if (lane == 0 && r < rows) y[size_t(tg.y) * dims.y + row0 + r] = total;
    }
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
        // Même produit f32 que `micro_scaled` dans la plage normale, avec
        // l'échelle décodée une fois par bloc de huit poids.
        if (G == 16 || (scale >= 10 && scale <= 246)) {
            float s = micro_scale<G>(scale);
            for (uint j = 0; j < 8; ++j) {
                uint word = j < P ? word0 : word1;
                out[j] = bfloat(micro_code<B>((word >> ((j % P) * B)) & ((1u << B) - 1u)) * s);
            }
        } else {
            for (uint j = 0; j < 8; ++j) {
                uint word = j < P ? word0 : word1;
                out[j] = bfloat(micro_scaled<B, G>((word >> ((j % P) * B)) & ((1u << B) - 1u), scale));
            }
        }
        dst[block] = out;
    }
}

kernel void micro_qmv_mxfp4(device const float* x [[buffer(0)]], device const uint* w [[buffer(1)]], device const uchar* scales [[buffer(2)]], device float* y [[buffer(4)]], constant uint3& dims [[buffer(5)]], uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    micro_qmv<4, 32, 1>(x, w, scales, y, dims, tg, sg, lane);
}
kernel void micro_qmv4_mxfp4(device const float* x [[buffer(0)]], device const uint* w [[buffer(1)]], device const uchar* scales [[buffer(2)]], device float* y [[buffer(4)]], constant uint3& dims [[buffer(5)]], uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    micro_qmv<4, 32, 4>(x, w, scales, y, dims, tg, sg, lane);
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
    NAX_KK_UNROLL
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
    micro_qmv<4, 16, 1>(x, w, scales, y, dims, tg, sg, lane);
}
kernel void micro_qmv4_nvfp4(device const float* x [[buffer(0)]], device const uint* w [[buffer(1)]], device const uchar* scales [[buffer(2)]], device float* y [[buffer(4)]], constant uint3& dims [[buffer(5)]], uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    micro_qmv<4, 16, 4>(x, w, scales, y, dims, tg, sg, lane);
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
    NAX_KK_UNROLL
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
    micro_qmv<8, 32, 1>(x, w, scales, y, dims, tg, sg, lane);
}
kernel void micro_qmv4_mxfp8(device const float* x [[buffer(0)]], device const uint* w [[buffer(1)]], device const uchar* scales [[buffer(2)]], device float* y [[buffer(4)]], constant uint3& dims [[buffer(5)]], uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    micro_qmv<8, 32, 4>(x, w, scales, y, dims, tg, sg, lane);
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
    NAX_KK_UNROLL
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

// MLP fusionné gate+up+SwiGLU à micro-échelles : mêmes tuiles et même ordre
// d'accumulation que `micro_gemm_*`, deux tuiles B par pas de K, sortie H bf16
// par le store SwiGLU du u4 fusionné. Ce kernel est tiré d'une compilation en
// fast-math (`micro_pipeline`), comme `swiglu_f32` : H reste égal au bit aux
// passes séparées gate, up, swiglu, conversion bf16.
#define MICRO_GATE_UP_SWIGLU(NAME, B, G)                                                     \
kernel void micro_gemm_gate_up_swiglu_##NAME(device const bfloat* A [[buffer(0)]],          \
    device const uint* Gp [[buffer(1)]], device const uchar* Gs [[buffer(2)]],              \
    device const uint* Up [[buffer(3)]], device const uchar* Us [[buffer(4)]],              \
    device bfloat* H [[buffer(5)]], constant uint3& mnk [[buffer(6)]],                      \
    uint2 tgid [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]], \
    uint sgid [[simdgroup_index_in_threadgroup]]) {                                         \
  uint M = mnk.x, N = mnk.y, K = mnk.z;                                                     \
  constexpr uint BN = 64, BK = 64, BKP = BK + 8, WN = 2;                                    \
  threadgroup metal::vec<bfloat,8> Wg8[BN * BKP / 8];                                       \
  threadgroup metal::vec<bfloat,8> Wu8[BN * BKP / 8];                                       \
  threadgroup bfloat* Wg = (threadgroup bfloat*)Wg8;                                        \
  threadgroup bfloat* Wu = (threadgroup bfloat*)Wu8;                                        \
  uint sg_m = sgid / WN, sg_n = sgid - sg_m * WN;                                           \
  uint m0 = tgid.x * 64u + sg_m * 32u, n0 = tgid.y * BN, ns = sg_n * 32u;                   \
  metal::vec<float,8> g0a = float(0), g0b = float(0), g1a = float(0), g1b = float(0);       \
  metal::vec<float,8> u0a = float(0), u0b = float(0), u1a = float(0), u1b = float(0);       \
  for (uint k = 0; k < K; k += BK) {                                                        \
    micro_load_b<B, G>(Wg, Gp, Gs, n0, k, N, K, lid);                                       \
    micro_load_b<B, G>(Wu, Up, Us, n0, k, N, K, lid);                                       \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                        \
    for (uint kk = 0; kk < BK; kk += 16u) {                                                 \
      metal::vec<bfloat,8> af0, af1, b0, b1;                                                \
      if (m0 + 32u <= M) {                                                                  \
        NaxFrag::load_a(af0, A + m0 * K + k + kk, int(K));                                  \
        NaxFrag::load_a(af1, A + (m0 + 16u) * K + k + kk, int(K));                          \
      } else {                                                                              \
        NaxFrag::load_a_masked(af0, A, m0, k + kk, K, M);                                   \
        NaxFrag::load_a_masked(af1, A, m0 + 16u, k + kk, K, M);                             \
      }                                                                                     \
      NaxFrag::load_a_tg(b0, Wg + ns * BKP + kk, int(BKP));                                 \
      NaxFrag::load_a_tg(b1, Wg + (ns + 16u) * BKP + kk, int(BKP));                         \
      NaxFrag::mma(g0a, g0b, af0, b0, b1);                                                  \
      NaxFrag::mma(g1a, g1b, af1, b0, b1);                                                  \
      NaxFrag::load_a_tg(b0, Wu + ns * BKP + kk, int(BKP));                                 \
      NaxFrag::load_a_tg(b1, Wu + (ns + 16u) * BKP + kk, int(BKP));                         \
      NaxFrag::mma(u0a, u0b, af0, b0, b1);                                                  \
      NaxFrag::mma(u1a, u1b, af1, b0, b1);                                                  \
    }                                                                                       \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                        \
  }                                                                                         \
  NaxFrag::store_swiglu_bf16_masked(g0a, u0a, H, m0, n0 + ns, N, M);                                  \
  NaxFrag::store_swiglu_bf16_masked(g0b, u0b, H, m0, n0 + ns + 16u, N, M);                            \
  NaxFrag::store_swiglu_bf16_masked(g1a, u1a, H, m0 + 16u, n0 + ns, N, M);                            \
  NaxFrag::store_swiglu_bf16_masked(g1b, u1b, H, m0 + 16u, n0 + ns + 16u, N, M);                      \
}
MICRO_GATE_UP_SWIGLU(mxfp4, 4, 32)
MICRO_GATE_UP_SWIGLU(nvfp4, 4, 16)
MICRO_GATE_UP_SWIGLU(mxfp8, 8, 32)
