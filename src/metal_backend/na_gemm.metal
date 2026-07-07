
#include <metal_stdlib>
#include <metal_tensor>
#include <metal_simdgroup_matrix>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp;
kernel void gemm_nax(device const bfloat* Ap [[buffer(0)]],
                     device const bfloat* Bp [[buffer(1)]],
                     device float* Cp         [[buffer(2)]],
                     constant uint3& mnk       [[buffer(3)]],
                     uint2 tgid [[threadgroup_position_in_grid]]) {
  uint M = mnk.x, N = mnk.y, K = mnk.z;
  auto A = tensor<device bfloat, dextents<int32_t,2>, tensor_inline>((device bfloat*)Ap, dextents<int32_t,2>(int32_t(K), int32_t(M)));
  auto B = tensor<device bfloat, dextents<int32_t,2>, tensor_inline>((device bfloat*)Bp, dextents<int32_t,2>(int32_t(N), int32_t(K)));
  auto C = tensor<device float,  dextents<int32_t,2>, tensor_inline>(Cp, dextents<int32_t,2>(int32_t(N), int32_t(M)));
  constexpr auto desc = tensor_ops::matmul2d_descriptor(
      64, 32, static_cast<int>(dynamic_extent), false, false, false);
  tensor_ops::matmul2d<desc, execution_simdgroups<4>> op;
  uint m0 = tgid.x * 64, n0 = tgid.y * 32;
  auto mA = A.slice(0, m0);
  auto mB = B.slice(n0, 0);
  auto mC = C.slice(n0, m0);
  op.run(mA, mB, mC);
}
kernel void gemm_nax_bf16_bn128(device const bfloat* Ap [[buffer(0)]],
                                device const bfloat* Bp [[buffer(1)]],
                                device float* Cp         [[buffer(2)]],
                                constant uint3& mnk       [[buffer(3)]],
                                uint2 tgid [[threadgroup_position_in_grid]]) {
  uint M = mnk.x, N = mnk.y, K = mnk.z;
  auto A = tensor<device bfloat, dextents<int32_t,2>, tensor_inline>((device bfloat*)Ap, dextents<int32_t,2>(int32_t(K), int32_t(M)));
  auto B = tensor<device bfloat, dextents<int32_t,2>, tensor_inline>((device bfloat*)Bp, dextents<int32_t,2>(int32_t(N), int32_t(K)));
  auto C = tensor<device float,  dextents<int32_t,2>, tensor_inline>(Cp, dextents<int32_t,2>(int32_t(N), int32_t(M)));
  constexpr auto desc = tensor_ops::matmul2d_descriptor(
      64, 128, static_cast<int>(dynamic_extent), false, false, false);
  tensor_ops::matmul2d<desc, execution_simdgroups<8>> op;
  uint m0 = tgid.x * 64, n0 = tgid.y * 128;
  auto mA = A.slice(0, m0);
  auto mB = B.slice(n0, 0);
  auto mC = C.slice(n0, m0);
  op.run(mA, mB, mC);
}
// ===== Port qmm_t_nax brique 1 : primitive cooperative_tensor à fragments registres.
// Transcription de BaseNAXFrag (mlx steel/gemm/nax.h:27) : fragment 16x16, 8 élts/thread
// (2 lignes x 4 cols, lignes espacées de 8), mapping lane->coord par bit-twiddling, MMA
// 16x32x16 via mpp cooperative_tensor en registres. Probe = GEMM C[M,N]=A[M,K]·B[N,K]^T.
struct NaxFrag {
  static short2 get_coord() {
    ushort lane = __metal_get_thread_index_in_simdgroup(ushort());
    short qid = lane >> 2;
    short fm = (qid & 4) | ((lane >> 1) & 3);
    short fn = ((qid & 2) | (lane & 1)) * 4;
    return short2(fn, fm);
  }
  static void load_a(thread metal::vec<bfloat,8>& dst, const device bfloat* src, int str_x) {
    short2 sc = get_coord();
    src += sc.y * str_x + sc.x;
    for (short i = 0; i < 2; i++)
      for (short j = 0; j < 4; j++)
        dst[i * 4 + j] = src[(i * 8) * str_x + j];
  }
  static void load_a_masked(thread metal::vec<bfloat,8>& dst, const device bfloat* src,
                            uint m_base, uint k_base, uint K, uint M) {
    short2 sc = get_coord();
    for (short i = 0; i < 2; i++) {
      uint row = m_base + uint(sc.y) + uint(i) * 8u;
      for (short j = 0; j < 4; j++) {
        uint col = k_base + uint(sc.x) + uint(j);
        dst[i * 4 + j] = (row < M) ? src[row * K + col] : bfloat(0.0f);
      }
    }
  }
  static void store_c(const thread metal::vec<float,8>& src, device float* dst, int str_x) {
    short2 sc = get_coord();
    dst += sc.y * str_x + sc.x;
    for (short i = 0; i < 2; i++)
      for (short j = 0; j < 4; j++)
        dst[(i * 8) * str_x + j] = src[i * 4 + j];
  }
  static void store_c_masked(const thread metal::vec<float,8>& src, device float* dst,
                             uint m_base, uint n_base, uint N, uint M) {
    short2 sc = get_coord();
    for (short i = 0; i < 2; i++) {
      uint row = m_base + uint(sc.y) + uint(i) * 8u;
      if (row < M) {
        device float* row_dst = dst + row * N + n_base + uint(sc.x);
        for (short j = 0; j < 4; j++) { row_dst[j] = src[i * 4 + j]; }
      }
    }
  }
  static void load_a_tg(thread metal::vec<bfloat,8>& dst, const threadgroup bfloat* src, int str_x) {
    short2 sc = get_coord();
    src += sc.y * str_x + sc.x;
    for (short i = 0; i < 2; i++)
      for (short j = 0; j < 4; j++)
        dst[i * 4 + j] = src[(i * 8) * str_x + j];
  }
  // Charge un fragment 16×16 de poids B[N,K] quantifié u8 gs64 en le dé-quantifiant
  // (packed u32 4-par-mot, scales/biases bf16 par groupe de 64). n_base=ligne N, k_base=col K.
  static void load_b_quant(thread metal::vec<bfloat,8>& dst,
                           const device uint* packed, const device bfloat* scales,
                           const device bfloat* biases, uint n_base, uint k_base, uint K) {
    short2 sc = get_coord();
    uint groups = K / 64u;
    uint packed_cols = K / 4u;
    for (short i = 0; i < 2; i++) {
      uint nn = n_base + uint(sc.y) + uint(i) * 8u;
      for (short j = 0; j < 4; j++) {
        uint kk = k_base + uint(sc.x) + uint(j);
        uint word = packed[nn * packed_cols + (kk >> 2)];
        uint q = (word >> ((kk & 3u) * 8u)) & 0xffu;
        uint g = kk / 64u;
        float s = float(scales[nn * groups + g]);
        float b = float(biases[nn * groups + g]);
        dst[i * 4 + j] = bfloat(float(q) * s + b);
      }
    }
  }
  // Variante u4 gs64 : 8 poids par mot u32.
  static void load_b_quant_u4(thread metal::vec<bfloat,8>& dst,
                              const device uint* packed, const device bfloat* scales,
                              const device bfloat* biases, uint n_base, uint k_base, uint K) {
    short2 sc = get_coord();
    uint groups = K / 64u;
    uint packed_cols = K / 8u;
    for (short i = 0; i < 2; i++) {
      uint nn = n_base + uint(sc.y) + uint(i) * 8u;
      for (short j = 0; j < 4; j++) {
        uint kk = k_base + uint(sc.x) + uint(j);
        uint word = packed[nn * packed_cols + (kk >> 3)];
        uint q = (word >> ((kk & 7u) * 4u)) & 0x0fu;
        uint g = kk / 64u;
        float s = float(scales[nn * groups + g]);
        float b = float(biases[nn * groups + g]);
        dst[i * 4 + j] = bfloat(float(q) * s + b);
      }
    }
  }
  // Variante dense gs128 : même layout packed, mais un scale/bias couvre 128 colonnes K.
  static void load_b_quant_gs128(thread metal::vec<bfloat,8>& dst,
                                 const device uint* packed, const device bfloat* scales,
                                 const device bfloat* biases, uint n_base, uint k_base, uint K) {
    short2 sc = get_coord();
    uint groups = K / 128u;
    uint packed_cols = K / 4u;
    for (short i = 0; i < 2; i++) {
      uint nn = n_base + uint(sc.y) + uint(i) * 8u;
      for (short j = 0; j < 4; j++) {
        uint kk = k_base + uint(sc.x) + uint(j);
        uint word = packed[nn * packed_cols + (kk >> 2)];
        uint q = (word >> ((kk & 3u) * 8u)) & 0xffu;
        uint g = kk / 128u;
        float s = float(scales[nn * groups + g]);
        float b = float(biases[nn * groups + g]);
        dst[i * 4 + j] = bfloat(float(q) * s + b);
      }
    }
  }
  // Charge un fragment A 16×16 en GATHERANT depuis input[token] (token=perm[row]/top_k),
  // f32→bf16. Padding (perm sentinel) → 0. Fusionne le gather DANS le GEMM (zéro buffer).
  static void load_a_gathered(thread metal::vec<bfloat,8>& dst, const device float* input,
                              const device uint* perm, uint m0, uint k_base, uint K, uint top_k) {
    short2 sc = get_coord();
    for (short i = 0; i < 2; i++) {
      uint row = m0 + uint(sc.y) + uint(i) * 8u;
      uint slot = perm[row];
      if (slot == 0xFFFFFFFFu) {
        for (short j = 0; j < 4; j++) { dst[i * 4 + j] = bfloat(0.0f); }
      } else {
        const device float* src = input + (slot / top_k) * K + k_base + uint(sc.x);
        for (short j = 0; j < 4; j++) { dst[i * 4 + j] = bfloat(src[j]); }
      }
    }
  }
  // Écrit le fragment C 16×32 en SCATTERANT vers C[slot] (slot=perm[row]). Padding ignoré.
  static void store_c_scattered(const thread metal::vec<float,8>& src, device float* C,
                                const device uint* perm, uint m0, uint n0, uint N) {
    short2 sc = get_coord();
    for (short i = 0; i < 2; i++) {
      uint row = m0 + uint(sc.y) + uint(i) * 8u;
      uint slot = perm[row];
      if (slot != 0xFFFFFFFFu) {
        device float* dst = C + slot * N + n0 + uint(sc.x);
        for (short j = 0; j < 4; j++) { dst[j] = src[i * 4 + j]; }
      }
    }
  }
  static void store_swiglu_bf16(const thread metal::vec<float,8>& gate,
                                const thread metal::vec<float,8>& up,
                                device bfloat* dst, int str_x) {
    short2 sc = get_coord();
    dst += sc.y * str_x + sc.x;
    for (short i = 0; i < 2; i++) {
      for (short j = 0; j < 4; j++) {
        short idx = i * 4 + j;
        float g = gate[idx];
        dst[(i * 8) * str_x + j] = bfloat((g / (1.0f + exp(-g))) * up[idx]);
      }
    }
  }
  static void mma(thread metal::vec<float,8>& Cn0, thread metal::vec<float,8>& Cn1,
                  const thread metal::vec<bfloat,8>& A,
                  const thread metal::vec<bfloat,8>& Bn0, const thread metal::vec<bfloat,8>& Bn1) {
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 16, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ct_a = op.get_left_input_cooperative_tensor<bfloat, bfloat, float>();
    auto ct_b = op.get_right_input_cooperative_tensor<bfloat, bfloat, float>();
    auto ct_c = op.get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), float>();
    for (short i = 0; i < 8; i++) ct_a[i] = A[i];
    for (short i = 0; i < 8; i++) { ct_b[i] = Bn0[i]; ct_b[8 + i] = Bn1[i]; }
    for (short i = 0; i < 8; i++) { ct_c[i] = Cn0[i]; ct_c[8 + i] = Cn1[i]; }
    op.run(ct_a, ct_b, ct_c);
    for (short i = 0; i < 8; i++) { Cn0[i] = ct_c[i]; Cn1[i] = ct_c[8 + i]; }
  }
};
// Port qmm_t_nax brique 3 : GEMM naïf (le plus rapide mesuré) + dé-quant B EN kernel.
// A bf16 [M,K], B poids u8 gs64 (packed/scales/biases) [N,K], C f32 [M,N]. C=A·deq(B)^T.
// Zéro matérialisation bf16 (le poids reste packed, dé-quant par fragment). Grid (M/16,N/32).
kernel void gemm_nax_coop_qb(device const bfloat* A [[buffer(0)]],
                             device const uint* Bp   [[buffer(1)]],
                             device const bfloat* Bs [[buffer(2)]],
                             device const bfloat* Bb [[buffer(3)]],
                             device float* C         [[buffer(4)]],
                             constant uint3& mnk     [[buffer(5)]],
                             uint2 tgid [[threadgroup_position_in_grid]]) {
  uint N = mnk.y, K = mnk.z;
  uint m0 = tgid.x * 16u, n0 = tgid.y * 32u;
  metal::vec<float,8> Cn0 = float(0), Cn1 = float(0);
  for (uint k = 0; k < K; k += 16u) {
    metal::vec<bfloat,8> Af, Bn0, Bn1;
    NaxFrag::load_a(Af, A + m0 * K + k, int(K));
    NaxFrag::load_b_quant(Bn0, Bp, Bs, Bb, n0, k, K);
    NaxFrag::load_b_quant(Bn1, Bp, Bs, Bb, n0 + 16u, k, K);
    NaxFrag::mma(Cn0, Cn1, Af, Bn0, Bn1);
  }
  NaxFrag::store_c(Cn0, C + m0 * N + n0, int(N));
  NaxFrag::store_c(Cn1, C + m0 * N + n0 + 16u, int(N));
}
// Variante dense gs128. M et N doivent être alignés 16×32 côté gate Rust.
kernel void gemm_nax_coop_qb_gs128(device const bfloat* A [[buffer(0)]],
                                   device const uint* Bp   [[buffer(1)]],
                                   device const bfloat* Bs [[buffer(2)]],
                                   device const bfloat* Bb [[buffer(3)]],
                                   device float* C         [[buffer(4)]],
                                   constant uint3& mnk     [[buffer(5)]],
                                   uint2 tgid [[threadgroup_position_in_grid]]) {
  uint M = mnk.x, N = mnk.y, K = mnk.z;
  uint m0 = tgid.x * 16u, n0 = tgid.y * 32u;
  metal::vec<float,8> Cn0 = float(0), Cn1 = float(0);
  for (uint k = 0; k < K; k += 16u) {
    metal::vec<bfloat,8> Af, Bn0, Bn1;
    NaxFrag::load_a_masked(Af, A, m0, k, K, M);
    NaxFrag::load_b_quant_gs128(Bn0, Bp, Bs, Bb, n0, k, K);
    NaxFrag::load_b_quant_gs128(Bn1, Bp, Bs, Bb, n0 + 16u, k, K);
    NaxFrag::mma(Cn0, Cn1, Af, Bn0, Bn1);
  }
  NaxFrag::store_c_masked(Cn0, C, m0, n0, N, M);
  NaxFrag::store_c_masked(Cn1, C, m0, n0 + 16u, N, M);
}
static void load_b_tile_quant_gs64(threadgroup bfloat* Ws,
                                   const device uint* Bp,
                                   const device bfloat* Bs,
                                   const device bfloat* Bb,
                                   uint n0,
                                   uint k0,
                                   uint N,
                                   uint K,
                                   uint lid,
                                   uint threads) {
  constexpr uint BN = 64, BK = 64;
  uint packed_cols = K / 4u;
  uint groups = K / 64u;
  for (uint idx = lid; idx < BN * BK; idx += threads) {
    uint nl = idx / BK, kl = idx - nl * BK;
    uint nn = n0 + nl, kk = k0 + kl;
    bfloat w = bfloat(0.0f);
    if (nn < N && kk < K) {
      uint word = Bp[nn * packed_cols + (kk >> 2u)];
      uint q = (word >> ((kk & 3u) * 8u)) & 0xffu;
      uint g = kk / 64u;
      float s = float(Bs[nn * groups + g]);
      float b = float(Bb[nn * groups + g]);
      w = bfloat(float(q) * s + b);
    }
    Ws[idx] = w;
  }
}
static void load_b_tile_quant_u4_gs64(threadgroup bfloat* Ws,
                                      const device uint* Bp,
                                      const device bfloat* Bs,
                                      const device bfloat* Bb,
                                      uint n0,
                                      uint k0,
                                      uint N,
                                      uint K,
                                      uint lid,
                                      uint threads) {
  constexpr uint BN = 64, BK = 64;
  uint packed_cols = K / 8u;
  uint groups = K / 64u;
  for (uint idx = lid; idx < BN * BK; idx += threads) {
    uint nl = idx / BK, kl = idx - nl * BK;
    uint nn = n0 + nl, kk = k0 + kl;
    bfloat w = bfloat(0.0f);
    if (nn < N && kk < K) {
      uint word = Bp[nn * packed_cols + (kk >> 3u)];
      uint q = (word >> ((kk & 7u) * 4u)) & 0x0fu;
      uint g = kk / 64u;
      float s = float(Bs[nn * groups + g]);
      float b = float(Bb[nn * groups + g]);
      w = bfloat(float(q) * s + b);
    }
    Ws[idx] = w;
  }
}
static void load_b_tile_quant_gs128(threadgroup bfloat* Ws,
                                    const device uint* Bp,
                                    const device bfloat* Bs,
                                    const device bfloat* Bb,
                                    uint n0,
                                    uint k0,
                                    uint N,
                                    uint K,
                                    uint lid,
                                    uint threads) {
  constexpr uint BN = 64, BK = 64;
  uint packed_cols = K / 4u;
  uint groups = K / 128u;
  for (uint idx = lid; idx < BN * BK; idx += threads) {
    uint nl = idx / BK, kl = idx - nl * BK;
    uint nn = n0 + nl, kk = k0 + kl;
    bfloat w = bfloat(0.0f);
    if (nn < N && kk < K) {
      uint word = Bp[nn * packed_cols + (kk >> 2u)];
      uint q = (word >> ((kk & 3u) * 8u)) & 0xffu;
      uint g = kk / 128u;
      float s = float(Bs[nn * groups + g]);
      float b = float(Bb[nn * groups + g]);
      w = bfloat(float(q) * s + b);
    }
    Ws[idx] = w;
  }
}
// GEMM NA quantifié tuilé : BM=BN=BK=64, WM=WN=2. Chaque threadgroup dé-quantifie
// une tuile B u8→bf16 en mémoire threadgroup, puis 4 simdgroups calculent chacun
// 32x32 de C avec deux fragments M. Zéro matérialisation du poids bf16 global.
kernel void gemm_nax_coop_qb_tiled(device const bfloat* A [[buffer(0)]],
                                   device const uint* Bp   [[buffer(1)]],
                                   device const bfloat* Bs [[buffer(2)]],
                                   device const bfloat* Bb [[buffer(3)]],
                                   device float* C         [[buffer(4)]],
                                   constant uint3& mnk     [[buffer(5)]],
                                   uint2 tgid [[threadgroup_position_in_grid]],
                                   uint lid [[thread_index_in_threadgroup]],
                                   uint sgid [[simdgroup_index_in_threadgroup]]) {
  uint M = mnk.x, N = mnk.y, K = mnk.z;
  constexpr uint BN = 64, BK = 64, WM = 2, WN = 2;
  threadgroup bfloat Ws[BN * BK];
  uint sg_m = sgid / WN;
  uint sg_n = sgid - sg_m * WN;
  uint m0 = tgid.x * 64u + sg_m * 32u;
  uint n0 = tgid.y * BN;
  uint ns = sg_n * 32u;
  metal::vec<float,8> c0a = float(0), c0b = float(0), c1a = float(0), c1b = float(0);
  for (uint k = 0; k < K; k += BK) {
    load_b_tile_quant_gs64(Ws, Bp, Bs, Bb, n0, k, N, K, lid, WM * WN * 32u);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kk = 0; kk < BK; kk += 16u) {
      metal::vec<bfloat,8> af0, af1, b0, b1;
      NaxFrag::load_a_masked(af0, A, m0, k + kk, K, M);
      NaxFrag::load_a_masked(af1, A, m0 + 16u, k + kk, K, M);
      NaxFrag::load_a_tg(b0, Ws + ns * BK + kk, int(BK));
      NaxFrag::load_a_tg(b1, Ws + (ns + 16u) * BK + kk, int(BK));
      NaxFrag::mma(c0a, c0b, af0, b0, b1);
      NaxFrag::mma(c1a, c1b, af1, b0, b1);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  NaxFrag::store_c_masked(c0a, C, m0, n0 + ns, N, M);
  NaxFrag::store_c_masked(c0b, C, m0, n0 + ns + 16u, N, M);
  NaxFrag::store_c_masked(c1a, C, m0 + 16u, n0 + ns, N, M);
  NaxFrag::store_c_masked(c1b, C, m0 + 16u, n0 + ns + 16u, N, M);
}
kernel void gemm_nax_coop_qb_tiled_u4(device const bfloat* A [[buffer(0)]],
                                      device const uint* Bp   [[buffer(1)]],
                                      device const bfloat* Bs [[buffer(2)]],
                                      device const bfloat* Bb [[buffer(3)]],
                                      device float* C         [[buffer(4)]],
                                      constant uint3& mnk     [[buffer(5)]],
                                      uint2 tgid [[threadgroup_position_in_grid]],
                                      uint lid [[thread_index_in_threadgroup]],
                                      uint sgid [[simdgroup_index_in_threadgroup]]) {
  uint M = mnk.x, N = mnk.y, K = mnk.z;
  constexpr uint BN = 64, BK = 64, WM = 2, WN = 2;
  threadgroup bfloat Ws[BN * BK];
  uint sg_m = sgid / WN;
  uint sg_n = sgid - sg_m * WN;
  uint m0 = tgid.x * 64u + sg_m * 32u;
  uint n0 = tgid.y * BN;
  uint ns = sg_n * 32u;
  metal::vec<float,8> c0a = float(0), c0b = float(0), c1a = float(0), c1b = float(0);
  for (uint k = 0; k < K; k += BK) {
    load_b_tile_quant_u4_gs64(Ws, Bp, Bs, Bb, n0, k, N, K, lid, WM * WN * 32u);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kk = 0; kk < BK; kk += 16u) {
      metal::vec<bfloat,8> af0, af1, b0, b1;
      NaxFrag::load_a_masked(af0, A, m0, k + kk, K, M);
      NaxFrag::load_a_masked(af1, A, m0 + 16u, k + kk, K, M);
      NaxFrag::load_a_tg(b0, Ws + ns * BK + kk, int(BK));
      NaxFrag::load_a_tg(b1, Ws + (ns + 16u) * BK + kk, int(BK));
      NaxFrag::mma(c0a, c0b, af0, b0, b1);
      NaxFrag::mma(c1a, c1b, af1, b0, b1);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  NaxFrag::store_c_masked(c0a, C, m0, n0 + ns, N, M);
  NaxFrag::store_c_masked(c0b, C, m0, n0 + ns + 16u, N, M);
  NaxFrag::store_c_masked(c1a, C, m0 + 16u, n0 + ns, N, M);
  NaxFrag::store_c_masked(c1b, C, m0 + 16u, n0 + ns + 16u, N, M);
}
kernel void gemm_nax_coop_qb_tiled_gs128(device const bfloat* A [[buffer(0)]],
                                         device const uint* Bp   [[buffer(1)]],
                                         device const bfloat* Bs [[buffer(2)]],
                                         device const bfloat* Bb [[buffer(3)]],
                                         device float* C         [[buffer(4)]],
                                         constant uint3& mnk     [[buffer(5)]],
                                         uint2 tgid [[threadgroup_position_in_grid]],
                                         uint lid [[thread_index_in_threadgroup]],
                                         uint sgid [[simdgroup_index_in_threadgroup]]) {
  uint M = mnk.x, N = mnk.y, K = mnk.z;
  constexpr uint BN = 64, BK = 64, WM = 2, WN = 2;
  threadgroup bfloat Ws[BN * BK];
  uint sg_m = sgid / WN;
  uint sg_n = sgid - sg_m * WN;
  uint m0 = tgid.x * 64u + sg_m * 32u;
  uint n0 = tgid.y * BN;
  uint ns = sg_n * 32u;
  metal::vec<float,8> c0a = float(0), c0b = float(0), c1a = float(0), c1b = float(0);
  for (uint k = 0; k < K; k += BK) {
    load_b_tile_quant_gs128(Ws, Bp, Bs, Bb, n0, k, N, K, lid, WM * WN * 32u);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kk = 0; kk < BK; kk += 16u) {
      metal::vec<bfloat,8> af0, af1, b0, b1;
      NaxFrag::load_a_masked(af0, A, m0, k + kk, K, M);
      NaxFrag::load_a_masked(af1, A, m0 + 16u, k + kk, K, M);
      NaxFrag::load_a_tg(b0, Ws + ns * BK + kk, int(BK));
      NaxFrag::load_a_tg(b1, Ws + (ns + 16u) * BK + kk, int(BK));
      NaxFrag::mma(c0a, c0b, af0, b0, b1);
      NaxFrag::mma(c1a, c1b, af1, b0, b1);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  NaxFrag::store_c_masked(c0a, C, m0, n0 + ns, N, M);
  NaxFrag::store_c_masked(c0b, C, m0, n0 + ns + 16u, N, M);
  NaxFrag::store_c_masked(c1a, C, m0 + 16u, n0 + ns, N, M);
  NaxFrag::store_c_masked(c1b, C, m0 + 16u, n0 + ns + 16u, N, M);
}
// Port qmm_t_nax brique 7 : GEMM groupé + GATHER FUSÉ (gate/up). Lit A directement
// depuis input[token] f32 via perm (zéro buffer a_padded). mnkt=(max_m,N,K,top_k).
kernel void gemm_nax_coop_qb_grouped_gather(device const float* input [[buffer(0)]],
                                            device const uint* Bp    [[buffer(1)]],
                                            device const bfloat* Bs  [[buffer(2)]],
                                            device const bfloat* Bb  [[buffer(3)]],
                                            device float* C          [[buffer(4)]],
                                            device const uint* tile_expert [[buffer(5)]],
                                            device const uint* perm  [[buffer(6)]],
                                            constant uint4& mnkt     [[buffer(7)]],
                                            uint2 tgid [[threadgroup_position_in_grid]]) {
  uint N = mnkt.y, K = mnkt.z, top_k = mnkt.w;
  uint e = tile_expert[tgid.x];
  if (e == 0xFFFFFFFFu) { return; }
  uint m0 = tgid.x * 16u, n0 = tgid.y * 32u;
  uint pe = e * N * (K / 4u);
  uint ge = e * N * (K / 64u);
  metal::vec<float,8> Cn0 = float(0), Cn1 = float(0);
  for (uint k = 0; k < K; k += 16u) {
    metal::vec<bfloat,8> Af, Bn0, Bn1;
    NaxFrag::load_a_gathered(Af, input, perm, m0, k, K, top_k);
    NaxFrag::load_b_quant(Bn0, Bp + pe, Bs + ge, Bb + ge, n0, k, K);
    NaxFrag::load_b_quant(Bn1, Bp + pe, Bs + ge, Bb + ge, n0 + 16u, k, K);
    NaxFrag::mma(Cn0, Cn1, Af, Bn0, Bn1);
  }
  NaxFrag::store_c(Cn0, C + m0 * N + n0, int(N));
  NaxFrag::store_c(Cn1, C + m0 * N + n0 + 16u, int(N));
}
// F2 : GEMM gate + GEMM up dans le même dispatch, puis SwiGLU f32 -> bf16
// directement dans l'épilogue. mnkt=(max_m,N,K,top_k).
kernel void gemm_nax_coop_qb_grouped_gate_up_swiglu(
                                            device const float* input [[buffer(0)]],
                                            device const uint* Gp    [[buffer(1)]],
                                            device const bfloat* Gs  [[buffer(2)]],
                                            device const bfloat* Gb  [[buffer(3)]],
                                            device const uint* Up    [[buffer(4)]],
                                            device const bfloat* Us  [[buffer(5)]],
                                            device const bfloat* Ub  [[buffer(6)]],
                                            device bfloat* H         [[buffer(7)]],
                                            device const uint* tile_expert [[buffer(8)]],
                                            device const uint* perm  [[buffer(9)]],
                                            constant uint4& mnkt     [[buffer(10)]],
                                            uint2 tgid [[threadgroup_position_in_grid]]) {
  uint N = mnkt.y, K = mnkt.z, top_k = mnkt.w;
  uint e = tile_expert[tgid.x];
  if (e == 0xFFFFFFFFu) { return; }
  uint m0 = tgid.x * 16u, n0 = tgid.y * 32u;
  uint pe = e * N * (K / 4u);
  uint ge = e * N * (K / 64u);
  metal::vec<float,8> Gn0 = float(0), Gn1 = float(0);
  metal::vec<float,8> Un0 = float(0), Un1 = float(0);
  for (uint k = 0; k < K; k += 16u) {
    metal::vec<bfloat,8> Af, Bn0, Bn1;
    NaxFrag::load_a_gathered(Af, input, perm, m0, k, K, top_k);
    NaxFrag::load_b_quant(Bn0, Gp + pe, Gs + ge, Gb + ge, n0, k, K);
    NaxFrag::load_b_quant(Bn1, Gp + pe, Gs + ge, Gb + ge, n0 + 16u, k, K);
    NaxFrag::mma(Gn0, Gn1, Af, Bn0, Bn1);
    NaxFrag::load_b_quant(Bn0, Up + pe, Us + ge, Ub + ge, n0, k, K);
    NaxFrag::load_b_quant(Bn1, Up + pe, Us + ge, Ub + ge, n0 + 16u, k, K);
    NaxFrag::mma(Un0, Un1, Af, Bn0, Bn1);
  }
  NaxFrag::store_swiglu_bf16(Gn0, Un0, H + m0 * N + n0, int(N));
  NaxFrag::store_swiglu_bf16(Gn1, Un1, H + m0 * N + n0 + 16u, int(N));
}
kernel void gemm_nax_coop_qb_grouped_gate_up_swiglu_u4(
                                            device const float* input [[buffer(0)]],
                                            device const uint* Gp    [[buffer(1)]],
                                            device const bfloat* Gs  [[buffer(2)]],
                                            device const bfloat* Gb  [[buffer(3)]],
                                            device const uint* Up    [[buffer(4)]],
                                            device const bfloat* Us  [[buffer(5)]],
                                            device const bfloat* Ub  [[buffer(6)]],
                                            device bfloat* H         [[buffer(7)]],
                                            device const uint* tile_expert [[buffer(8)]],
                                            device const uint* perm  [[buffer(9)]],
                                            constant uint4& mnkt     [[buffer(10)]],
                                            uint2 tgid [[threadgroup_position_in_grid]]) {
  uint N = mnkt.y, K = mnkt.z, top_k = mnkt.w;
  uint e = tile_expert[tgid.x];
  if (e == 0xFFFFFFFFu) { return; }
  uint m0 = tgid.x * 16u, n0 = tgid.y * 32u;
  uint pe = e * N * (K / 8u);
  uint ge = e * N * (K / 64u);
  metal::vec<float,8> Gn0 = float(0), Gn1 = float(0);
  metal::vec<float,8> Un0 = float(0), Un1 = float(0);
  for (uint k = 0; k < K; k += 16u) {
    metal::vec<bfloat,8> Af, Bn0, Bn1;
    NaxFrag::load_a_gathered(Af, input, perm, m0, k, K, top_k);
    NaxFrag::load_b_quant_u4(Bn0, Gp + pe, Gs + ge, Gb + ge, n0, k, K);
    NaxFrag::load_b_quant_u4(Bn1, Gp + pe, Gs + ge, Gb + ge, n0 + 16u, k, K);
    NaxFrag::mma(Gn0, Gn1, Af, Bn0, Bn1);
    NaxFrag::load_b_quant_u4(Bn0, Up + pe, Us + ge, Ub + ge, n0, k, K);
    NaxFrag::load_b_quant_u4(Bn1, Up + pe, Us + ge, Ub + ge, n0 + 16u, k, K);
    NaxFrag::mma(Un0, Un1, Af, Bn0, Bn1);
  }
  NaxFrag::store_swiglu_bf16(Gn0, Un0, H + m0 * N + n0, int(N));
  NaxFrag::store_swiglu_bf16(Gn1, Un1, H + m0 * N + n0 + 16u, int(N));
}
// Port qmm_t_nax brique 7 : GEMM groupé + SCATTER FUSÉ (down). A=hidden2 padé contigu,
// écrit C[slot] via perm (zéro buffer down_res, zéro passe scatter). C=scratch.down par slot.
kernel void gemm_nax_coop_qb_grouped_scatter(device const bfloat* A   [[buffer(0)]],
                                             device const uint* Bp    [[buffer(1)]],
                                             device const bfloat* Bs  [[buffer(2)]],
                                             device const bfloat* Bb  [[buffer(3)]],
                                             device float* C          [[buffer(4)]],
                                             device const uint* tile_expert [[buffer(5)]],
                                             device const uint* perm  [[buffer(6)]],
                                             constant uint3& mnk      [[buffer(7)]],
                                             uint2 tgid [[threadgroup_position_in_grid]]) {
  uint N = mnk.y, K = mnk.z;
  uint e = tile_expert[tgid.x];
  if (e == 0xFFFFFFFFu) { return; }
  uint m0 = tgid.x * 16u, n0 = tgid.y * 32u;
  uint pe = e * N * (K / 4u);
  uint ge = e * N * (K / 64u);
  metal::vec<float,8> Cn0 = float(0), Cn1 = float(0);
  for (uint k = 0; k < K; k += 16u) {
    metal::vec<bfloat,8> Af, Bn0, Bn1;
    NaxFrag::load_a(Af, A + m0 * K + k, int(K));
    NaxFrag::load_b_quant(Bn0, Bp + pe, Bs + ge, Bb + ge, n0, k, K);
    NaxFrag::load_b_quant(Bn1, Bp + pe, Bs + ge, Bb + ge, n0 + 16u, k, K);
    NaxFrag::mma(Cn0, Cn1, Af, Bn0, Bn1);
  }
  NaxFrag::store_c_scattered(Cn0, C, perm, m0, n0, N);
  NaxFrag::store_c_scattered(Cn1, C, perm, m0, n0 + 16u, N);
}
kernel void gemm_nax_coop_qb_grouped_scatter_u4(device const bfloat* A   [[buffer(0)]],
                                             device const uint* Bp    [[buffer(1)]],
                                             device const bfloat* Bs  [[buffer(2)]],
                                             device const bfloat* Bb  [[buffer(3)]],
                                             device float* C          [[buffer(4)]],
                                             device const uint* tile_expert [[buffer(5)]],
                                             device const uint* perm  [[buffer(6)]],
                                             constant uint3& mnk      [[buffer(7)]],
                                             uint2 tgid [[threadgroup_position_in_grid]]) {
  uint N = mnk.y, K = mnk.z;
  uint e = tile_expert[tgid.x];
  if (e == 0xFFFFFFFFu) { return; }
  uint m0 = tgid.x * 16u, n0 = tgid.y * 32u;
  uint pe = e * N * (K / 8u);
  uint ge = e * N * (K / 64u);
  metal::vec<float,8> Cn0 = float(0), Cn1 = float(0);
  for (uint k = 0; k < K; k += 16u) {
    metal::vec<bfloat,8> Af, Bn0, Bn1;
    NaxFrag::load_a(Af, A + m0 * K + k, int(K));
    NaxFrag::load_b_quant_u4(Bn0, Bp + pe, Bs + ge, Bb + ge, n0, k, K);
    NaxFrag::load_b_quant_u4(Bn1, Bp + pe, Bs + ge, Bb + ge, n0 + 16u, k, K);
    NaxFrag::mma(Cn0, Cn1, Af, Bn0, Bn1);
  }
  NaxFrag::store_c_scattered(Cn0, C, perm, m0, n0, N);
  NaxFrag::store_c_scattered(Cn1, C, perm, m0, n0 + 16u, N);
}
// Port qmm_t_nax brique 6 : GROUPING SUR GPU (zéro readback). histogram atomique →
// scan offsets 16-alignés + remplit tile_expert (sentinel 0xFFFFFFFF) → build perm
// atomique. Tout en 1 command buffer, comme adjust_matrix_offsets de MLX.
kernel void moe_g_fill_u32(device uint* buf [[buffer(0)]],
                           constant uint2& nv [[buffer(1)]], // (n, value)
                           uint gid [[thread_position_in_grid]]) {
  if (gid >= nv.x) { return; }
  buf[gid] = nv.y;
}
kernel void moe_g_histogram(device const uint* indices [[buffer(0)]],
                            device atomic_uint* counts [[buffer(1)]],
                            constant uint& total [[buffer(2)]],
                            uint gid [[thread_position_in_grid]]) {
  if (gid >= total) { return; }
  atomic_fetch_add_explicit(&counts[indices[gid]], 1u, memory_order_relaxed);
}
// 1 threadgroup, 1 thread : scan exclusif des counts padés à 16 → padded_offset +
// remplit tile_expert + cursor=padded_offset. experts petit (256).
kernel void moe_g_offsets(device const uint* counts [[buffer(0)]],
                          device uint* padded_offset [[buffer(1)]],
                          device uint* tile_expert [[buffer(2)]],
                          device atomic_uint* cursor [[buffer(3)]],
                          constant uint& experts [[buffer(4)]],
                          uint gid [[thread_position_in_grid]]) {
  if (gid != 0) { return; }
  uint moff = 0u, toff = 0u;
  for (uint e = 0; e < experts; e++) {
    padded_offset[e] = moff;
    atomic_store_explicit(&cursor[e], moff, memory_order_relaxed);
    uint tiles = (counts[e] + 15u) / 16u;
    for (uint t = 0; t < tiles; t++) { tile_expert[toff + t] = e; }
    moff += tiles * 16u;
    toff += tiles;
  }
}
kernel void moe_g_perm(device const uint* indices [[buffer(0)]],
                       device atomic_uint* cursor [[buffer(1)]],
                       device uint* perm [[buffer(2)]],
                       constant uint& total [[buffer(3)]],
                       uint gid [[thread_position_in_grid]]) {
  if (gid >= total) { return; }
  uint pos = atomic_fetch_add_explicit(&cursor[indices[gid]], 1u, memory_order_relaxed);
  perm[pos] = gid;
}
// Port qmm_t_nax brique 5 : gather/scatter padés (orchestration MoE). perm[p]=slot
// (token*top_k+k) valide, 0xFFFFFFFF = padding. Chaque expert padé à 16 lignes.
kernel void moe_coop_gather_padded(device const float* input [[buffer(0)]],
                                   device const uint* perm   [[buffer(1)]],
                                   device bfloat* out        [[buffer(2)]],
                                   constant uint4& dims      [[buffer(3)]],
                                   uint gid [[thread_position_in_grid]]) {
  uint mpad = dims.x, hidden = dims.y, top_k = dims.z;
  uint total = mpad * hidden;
  if (gid >= total) { return; }
  uint p = gid / hidden, h = gid - p * hidden;
  uint slot = perm[p];
  out[gid] = (slot == 0xFFFFFFFFu) ? bfloat(0.0f) : bfloat(input[(slot / top_k) * hidden + h]);
}
kernel void moe_coop_scatter_padded(device const float* result [[buffer(0)]],
                                    device const uint* perm    [[buffer(1)]],
                                    device float* scratch      [[buffer(2)]],
                                    constant uint4& dims       [[buffer(3)]],
                                    uint gid [[thread_position_in_grid]]) {
  uint mpad = dims.x, out_dim = dims.y;
  uint total = mpad * out_dim;
  if (gid >= total) { return; }
  uint p = gid / out_dim, o = gid - p * out_dim;
  uint slot = perm[p];
  if (slot == 0xFFFFFFFFu) { return; }
  scratch[slot * out_dim + o] = result[gid];
}
// Port qmm_t_nax brique 4 : GEMM quantifié GROUPÉ (MoE). UN dispatch tous experts.
// A bf16 [M_padded,K] = activations triées+paddées par expert (16-aligné). Bp/Bs/Bb
// = poids empilés [experts,N,K]. tile_expert[tuile M] → expert (offsets packed/scales/
// biases). C=A·deq(B_e)^T. Grid (M_padded/16, N/32). C'est la technique mlx gather_qmm.
kernel void gemm_nax_coop_qb_grouped(device const bfloat* A [[buffer(0)]],
                                     device const uint* Bp   [[buffer(1)]],
                                     device const bfloat* Bs [[buffer(2)]],
                                     device const bfloat* Bb [[buffer(3)]],
                                     device float* C         [[buffer(4)]],
                                     device const uint* tile_expert [[buffer(5)]],
                                     constant uint3& mnk     [[buffer(6)]],
                                     uint2 tgid [[threadgroup_position_in_grid]]) {
  uint N = mnk.y, K = mnk.z;
  uint e = tile_expert[tgid.x];
  if (e == 0xFFFFFFFFu) { return; } // tuile padding inutilisée (grouping GPU)
  uint m0 = tgid.x * 16u, n0 = tgid.y * 32u;
  uint pe = e * N * (K / 4u);
  uint ge = e * N * (K / 64u);
  metal::vec<float,8> Cn0 = float(0), Cn1 = float(0);
  for (uint k = 0; k < K; k += 16u) {
    metal::vec<bfloat,8> Af, Bn0, Bn1;
    NaxFrag::load_a(Af, A + m0 * K + k, int(K));
    NaxFrag::load_b_quant(Bn0, Bp + pe, Bs + ge, Bb + ge, n0, k, K);
    NaxFrag::load_b_quant(Bn1, Bp + pe, Bs + ge, Bb + ge, n0 + 16u, k, K);
    NaxFrag::mma(Cn0, Cn1, Af, Bn0, Bn1);
  }
  NaxFrag::store_c(Cn0, C + m0 * N + n0, int(N));
  NaxFrag::store_c(Cn1, C + m0 * N + n0 + 16u, int(N));
}
// Port chunked-DeltaNet brique 4 : version LAYOUT RÉEL. 1 threadgroup/value_head, boucle
// chunks. Lit les vrais buffers : k/q_norm [T, key_dim] par key_head (GQA repeat), v depuis
// conv_out interleavé [T, conv_dim] à 2*key_dim+value_base, beta/decay [T, value_heads],
// ssm_state [value_dim, 128] par tête. dims=(value_heads, value_head_dim, repeat, steps).
// value_head_dim=128 requis (buffers threadgroup). Remplace linear_attn_gated_delta_seq.
kernel void chunk_delta_seq_layout(device const float* conv_out [[buffer(0)]],
                                   device const float* q_norm [[buffer(1)]],
                                   device const float* k_norm [[buffer(2)]],
                                   device const float* beta [[buffer(3)]],
                                   device const float* decay [[buffer(4)]],
                                   device float* ssm_state [[buffer(5)]],
                                   device float* y [[buffer(6)]],
                                   constant uint4& dims [[buffer(7)]], // (vheads, vhd, repeat, steps)
                                   uint gid [[threadgroup_position_in_grid]],
                                   uint tid [[thread_index_in_threadgroup]],
                                   uint nthreads [[threads_per_threadgroup]]) {
  const uint vheads = dims.x, vhd = dims.y, repeat = dims.z, steps = dims.w;
  const uint dk = 128u, CM = 16u;
  const uint vh = gid;
  if (vh >= vheads || repeat == 0u) { return; }
  const uint key_heads = vheads / repeat;
  const uint key_dim = key_heads * dk;
  const uint value_dim = vheads * vhd;
  const uint conv_dim = 2u * key_dim + value_dim;
  const uint key_base = (vh / repeat) * dk;
  const uint value_base = vh * vhd;
  threadgroup float gamma[16];
  threadgroup float A[16 * 16];
  threadgroup float P[16 * 16];
  threadgroup float u[16 * 128];
  threadgroup float delta[16 * 128];
  threadgroup float qs[16 * 128];
  const uint c = tid;
  for (uint c0 = 0; c0 < steps; c0 += CM) {
    const uint nc = min(CM, steps - c0);
    if (tid == 0) {
      float g = 1.0f;
      for (uint i = 0; i < nc; i++) { g *= decay[(c0 + i) * vheads + vh]; gamma[i] = g; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (c < vhd) {
      const uint srow = (value_base + c) * dk;
      for (uint i = 0; i < nc; i++) {
        float ks = 0.0f, qsv = 0.0f;
        const uint kq = (c0 + i) * key_dim + key_base;
        for (uint k = 0; k < dk; k++) {
          float s = ssm_state[srow + k];
          ks += s * k_norm[kq + k];
          qsv += s * q_norm[kq + k];
        }
        float vv = conv_out[(c0 + i) * conv_dim + 2u * key_dim + value_base + c];
        u[i * vhd + c] = beta[(c0 + i) * vheads + vh] * (vv - gamma[i] * ks);
        qs[i * vhd + c] = qsv;
      }
    }
    for (uint e = tid; e < nc * nc; e += nthreads) {
      uint i = e / nc, j = e % nc;
      float kk = 0.0f, qk = 0.0f;
      const uint ki = (c0 + i) * key_dim + key_base;
      const uint kj = (c0 + j) * key_dim + key_base;
      for (uint k = 0; k < dk; k++) {
        kk += k_norm[ki + k] * k_norm[kj + k];
        qk += q_norm[ki + k] * k_norm[kj + k];
      }
      float r = gamma[i] / gamma[j];
      A[i * CM + j] = (j < i) ? (beta[(c0 + i) * vheads + vh] * r * kk) : 0.0f;
      P[i * CM + j] = (j <= i) ? (r * qk) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (c < vhd) {
      for (uint i = 0; i < nc; i++) {
        float acc = u[i * vhd + c];
        for (uint j = 0; j < i; j++) { acc -= A[i * CM + j] * delta[j * vhd + c]; }
        delta[i * vhd + c] = acc;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (c < vhd) {
      for (uint i = 0; i < nc; i++) {
        float yv = gamma[i] * qs[i * vhd + c];
        for (uint j = 0; j <= i; j++) { yv += P[i * CM + j] * delta[j * vhd + c]; }
        y[(c0 + i) * value_dim + value_base + c] = yv;
      }
      const uint srow = (value_base + c) * dk;
      const float glast = gamma[nc - 1];
      for (uint k = 0; k < dk; k++) {
        float s = glast * ssm_state[srow + k];
        for (uint j = 0; j < nc; j++) {
          s += (glast / gamma[j]) * delta[j * vhd + c] * k_norm[(c0 + j) * key_dim + key_base + k];
        }
        ssm_state[srow + k] = s;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}
// Brique 6c : chunk_delta_seq_layout en TENSOR-CORES (KS/QS + state-update sur
// simdgroup_matrix, avec les offsets
// multi-tête/GQA). 128 threads = 4 simdgroups. value_head_dim=128 requis.
kernel void chunk_delta_seq_layout_tc(device const float* conv_out [[buffer(0)]],
                                      device const float* q_norm [[buffer(1)]],
                                      device const float* k_norm [[buffer(2)]],
                                      device const float* beta [[buffer(3)]],
                                      device const float* decay [[buffer(4)]],
                                      device float* ssm_state [[buffer(5)]],
                                      device float* y [[buffer(6)]],
                                      constant uint4& dims [[buffer(7)]], // (vheads,vhd,repeat,steps)
                                      uint gid [[threadgroup_position_in_grid]],
                                      uint tid [[thread_index_in_threadgroup]],
                                      uint nthreads [[threads_per_threadgroup]]) {
  const uint vheads = dims.x, vhd = dims.y, repeat = dims.z, steps = dims.w;
  const uint dk = 128u, CM = 16u;
  const uint vh = gid;
  if (vh >= vheads || repeat == 0u) { return; }
  const uint key_heads = vheads / repeat;
  const uint key_dim = key_heads * dk;
  const uint value_dim = vheads * vhd;
  const uint conv_dim = 2u * key_dim + value_dim;
  const uint key_base = (vh / repeat) * dk;
  const uint value_base = vh * vhd;
  threadgroup float gamma[16];
  threadgroup float A[16 * 16];
  threadgroup float P[16 * 16];
  threadgroup float u[16 * 128];
  threadgroup float delta[16 * 128];
  threadgroup float qs[16 * 128];
  const uint c = tid;
  const uint sgid = tid / 32u;
  for (uint c0 = 0; c0 < steps; c0 += CM) {
    const uint nc = min(CM, steps - c0);
    if (tid == 0) {
      float g = 1.0f;
      for (uint i = 0; i < nc; i++) { g *= decay[(c0 + i) * vheads + vh]; gamma[i] = g; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint tt = sgid; tt < (16u / 8u) * (vhd / 8u); tt += 4u) {
      uint i0 = (tt / (vhd / 8u)) * 8u, c0t = (tt % (vhd / 8u)) * 8u;
      metal::simdgroup_float8x8 acck = metal::make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
      metal::simdgroup_float8x8 accq = metal::make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
      for (uint kt = 0; kt < dk / 8u; kt++) {
        uint k0 = kt * 8u;
        metal::simdgroup_float8x8 ak, aq, bs;
        metal::simdgroup_load(ak, k_norm + (c0 + i0) * key_dim + key_base + k0, key_dim);
        metal::simdgroup_load(aq, q_norm + (c0 + i0) * key_dim + key_base + k0, key_dim);
        metal::simdgroup_load(bs, ssm_state + (value_base + c0t) * dk + k0, dk, ulong2(0, 0), true);
        metal::simdgroup_multiply_accumulate(acck, ak, bs, acck);
        metal::simdgroup_multiply_accumulate(accq, aq, bs, accq);
      }
      metal::simdgroup_store(acck, u + i0 * vhd + c0t, vhd);
      metal::simdgroup_store(accq, qs + i0 * vhd + c0t, vhd);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (c < vhd) {
      for (uint i = 0; i < nc; i++) {
        float vv = conv_out[(c0 + i) * conv_dim + 2u * key_dim + value_base + c];
        u[i * vhd + c] = beta[(c0 + i) * vheads + vh] * (vv - gamma[i] * u[i * vhd + c]);
      }
    }
    for (uint e = tid; e < nc * nc; e += nthreads) {
      uint i = e / nc, j = e % nc;
      float kk = 0.0f, qk = 0.0f;
      const uint ki = (c0 + i) * key_dim + key_base;
      const uint kj = (c0 + j) * key_dim + key_base;
      for (uint k = 0; k < dk; k++) {
        kk += k_norm[ki + k] * k_norm[kj + k];
        qk += q_norm[ki + k] * k_norm[kj + k];
      }
      float r = gamma[i] / gamma[j];
      A[i * CM + j] = (j < i) ? (beta[(c0 + i) * vheads + vh] * r * kk) : 0.0f;
      P[i * CM + j] = (j <= i) ? (r * qk) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (c < vhd) {
      for (uint i = 0; i < nc; i++) {
        float acc = u[i * vhd + c];
        for (uint j = 0; j < i; j++) { acc -= A[i * CM + j] * delta[j * vhd + c]; }
        delta[i * vhd + c] = acc;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float glast = gamma[nc - 1];
    if (c < vhd) {
      for (uint i = 0; i < nc; i++) {
        float yv = gamma[i] * qs[i * vhd + c];
        for (uint j = 0; j <= i; j++) { yv += P[i * CM + j] * delta[j * vhd + c]; }
        y[(c0 + i) * value_dim + value_base + c] = yv;
      }
      for (uint j = 0; j < CM; j++) {
        delta[j * vhd + c] = (j < nc) ? (glast / gamma[j]) * delta[j * vhd + c] : 0.0f;
      }
      const uint srow = (value_base + c) * dk;
      for (uint k = 0; k < dk; k++) { ssm_state[srow + k] *= glast; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t = sgid; t < (vhd / 8u) * (dk / 8u); t += 4u) {
      uint c0t = (t / (dk / 8u)) * 8u, k0t = (t % (dk / 8u)) * 8u;
      metal::simdgroup_float8x8 acc;
      metal::simdgroup_load(acc, ssm_state + (value_base + c0t) * dk + k0t, dk);
      for (uint jt = 0; jt < CM / 8u; jt++) {
        metal::simdgroup_float8x8 a, b;
        metal::simdgroup_load(a, delta + (jt * 8u) * vhd + c0t, vhd, ulong2(0, 0), true);
        metal::simdgroup_load(b, k_norm + (c0 + jt * 8u) * key_dim + key_base + k0t, key_dim);
        metal::simdgroup_multiply_accumulate(acc, a, b, acc);
      }
      metal::simdgroup_store(acc, ssm_state + (value_base + c0t) * dk + k0t, dk);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}
