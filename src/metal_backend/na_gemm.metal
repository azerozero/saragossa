
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
  static void load_a_masked_mk(thread metal::vec<bfloat,8>& dst, const device bfloat* src,
                               uint m_base, uint k_base, uint K, uint M) {
    short2 sc = get_coord();
    for (short i = 0; i < 2; i++) {
      uint row = m_base + uint(sc.y) + uint(i) * 8u;
      for (short j = 0; j < 4; j++) {
        uint col = k_base + uint(sc.x) + uint(j);
        dst[i * 4 + j] = (row < M && col < K)
            ? src[row * K + col]
            : bfloat(0.0f);
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
  // SwiGLU f32 sur les accumulateurs gate/up puis sortie bf16, lignes >= M ignorées.
  // Même expression que `swiglu_f32` puis `f32_to_bf16` (arrondi bfloat()) :
  // la fusion dense gate+up+SwiGLU reste bit-identique aux trois passes séparées.
  static void store_swiglu_bf16_masked(const thread metal::vec<float,8>& gate,
                                       const thread metal::vec<float,8>& up,
                                       device bfloat* dst, uint m_base, uint n_base, uint N, uint M) {
    short2 sc = get_coord();
    for (short i = 0; i < 2; i++) {
      uint row = m_base + uint(sc.y) + uint(i) * 8u;
      if (row < M) {
        device bfloat* row_dst = dst + row * N + n_base + uint(sc.x);
        for (short j = 0; j < 4; j++) {
          float g = gate[i * 4 + j];
          row_dst[j] = bfloat((g / (1.0f + exp(-g))) * up[i * 4 + j]);
        }
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
  // Fragment A 16×16 depuis un lhs f32 (vérification spéculative), lignes ≥ M à zéro.
  // SPLIT : hi = bf16(x), lo = bf16(x - hi) pour un produit « bf16×3 » proche du f32.
  template <bool SPLIT>
  static void load_a_f32_masked(thread metal::vec<bfloat,8>& hi, thread metal::vec<bfloat,8>& lo,
                                const device float* src, uint m_base, uint k_base, uint K, uint M) {
    short2 sc = get_coord();
    for (short i = 0; i < 2; i++) {
      uint row = m_base + uint(sc.y) + uint(i) * 8u;
      const device float* p = src + row * K + k_base + uint(sc.x);
      for (short j = 0; j < 4; j++) {
        float x = (row < M) ? p[j] : 0.0f;
        bfloat h = bfloat(x);
        hi[i * 4 + j] = h;
        if (SPLIT) { lo[i * 4 + j] = bfloat(x - float(h)); }
      }
    }
  }
  // Fragment B 16×16 u4 gs64 : un mot u32 (8 poids) et un scale/bias par ligne.
  template <bool SPLIT>
  static void load_b_u4_split(thread metal::vec<bfloat,8>& hi, thread metal::vec<bfloat,8>& lo,
                              const device uint* packed, const device bfloat* scales,
                              const device bfloat* biases, uint n_base, uint k_base, uint K) {
    short2 sc = get_coord();
    uint groups = K / 64u;
    uint packed_cols = K / 8u;
    uint kk = k_base + uint(sc.x);
    uint shift = (kk & 7u) * 4u;
    for (short i = 0; i < 2; i++) {
      uint nn = n_base + uint(sc.y) + uint(i) * 8u;
      uint word = packed[nn * packed_cols + (kk >> 3)] >> shift;
      float s = float(scales[nn * groups + (kk >> 6)]);
      float b = float(biases[nn * groups + (kk >> 6)]);
      for (short j = 0; j < 4; j++) {
        float w = float((word >> (4u * uint(j))) & 0x0fu) * s + b;
        bfloat h = bfloat(w);
        hi[i * 4 + j] = h;
        if (SPLIT) { lo[i * 4 + j] = bfloat(w - float(h)); }
      }
    }
  }
  static void store_c_tg(const thread metal::vec<float,8>& src, threadgroup float* dst, int str_x) {
    short2 sc = get_coord();
    dst += sc.y * str_x + sc.x;
    for (short i = 0; i < 2; i++)
      for (short j = 0; j < 4; j++)
        dst[(i * 8) * str_x + j] = src[i * 4 + j];
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
// Variante vectorisée du chargeur u4 gs64 pour BK = 64 = group_size : chaque thread
// (128 par threadgroup) dé-quantifie une demi-ligne de la tuile (32 poids = 4 mots
// u32) avec UN scale et UN bias lus une seule fois, contre un mot + scale + bias
// relus par poids dans la version scalaire (16 fois moins de lectures device).
// Ws est paddé à BKP = BK + 8 colonnes pour étaler les fragments sur les banques.
// Précondition : K % 64 == 0 et k0 + 64 <= K (pas de masque sur K).
// SPLIT : Wl reçoit en plus lo = bf16(w - hi), le poids déquantifié f32 étant
// w = q·s + b ; hi + lo porte ~16 bits de mantisse au lieu de 8.
template <bool SPLIT>
static void load_b_tile_quant_u4_gs64_split(threadgroup bfloat* Ws,
                                            threadgroup bfloat* Wl,
                                            const device uint* Bp,
                                          const device bfloat* Bs,
                                          const device bfloat* Bb,
                                          uint n0,
                                          uint k0,
                                          uint N,
                                          uint K,
                                          uint lid) {
  constexpr uint BKP = 64u + 8u;
  uint nl = lid >> 1u;
  uint kl0 = (lid & 1u) * 32u;
  uint nn = n0 + nl;
  // Ws est déclaré en vec<bfloat,8> par les appelants : les stores de 8 poids
  // sont alignés sur 16 octets (stride de ligne 144 octets, offsets multiples de 16).
  threadgroup metal::vec<bfloat,8>* dst =
      (threadgroup metal::vec<bfloat,8>*)(Ws + nl * BKP + kl0);
  threadgroup metal::vec<bfloat,8>* dst_lo =
      (threadgroup metal::vec<bfloat,8>*)(Wl + nl * BKP + kl0);
  if (nn >= N) {
    for (uint w = 0; w < 4u; w++) {
      dst[w] = metal::vec<bfloat,8>(bfloat(0.0f));
      if (SPLIT) { dst_lo[w] = metal::vec<bfloat,8>(bfloat(0.0f)); }
    }
    return;
  }
  uint packed_cols = K / 8u;
  uint groups = K / 64u;
  uint g = k0 / 64u;
  float s = float(Bs[nn * groups + g]);
  float b = float(Bb[nn * groups + g]);
  const device metal::vec<uint,4>* src =
      (const device metal::vec<uint,4>*)(Bp + nn * packed_cols + ((k0 + kl0) >> 3u));
  metal::vec<uint,4> words = *src;
  for (uint w = 0; w < 4u; w++) {
    uint word = words[w];
    metal::vec<bfloat,8> out, out_lo;
    for (uint j = 0; j < 8u; j++) {
      float v = float((word >> (4u * j)) & 0x0fu) * s + b;
      out[j] = bfloat(v);
      if (SPLIT) { out_lo[j] = bfloat(v - float(out[j])); }
    }
    dst[w] = out;
    if (SPLIT) { dst_lo[w] = out_lo; }
  }
}
static void load_b_tile_quant_u4_gs64_vec(threadgroup bfloat* Ws,
                                          const device uint* Bp,
                                          const device bfloat* Bs,
                                          const device bfloat* Bb,
                                          uint n0,
                                          uint k0,
                                          uint N,
                                          uint K,
                                          uint lid) {
  load_b_tile_quant_u4_gs64_split<false>(Ws, Ws, Bp, Bs, Bb, n0, k0, N, K, lid);
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
    load_b_tile_quant_u4_gs64_vec(Ws, Bp, Bs, Bb, n0, k, N, K, lid);
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
  NaxFrag::store_c_masked(c0a, C, m0, n0 + ns, N, M);
  NaxFrag::store_c_masked(c0b, C, m0, n0 + ns + 16u, N, M);
  NaxFrag::store_c_masked(c1a, C, m0 + 16u, n0 + ns, N, M);
  NaxFrag::store_c_masked(c1b, C, m0 + 16u, n0 + ns + 16u, N, M);
}
// Variante « bf16×3 » de `gemm_nax_coop_qb_tiled_u4` pour les petites
// projections de flint : A est lu en f32 et scindé en hi + lo, la tuile B
// déquantifiée aussi, et chaque pas fait hi·hi + hi·lo + lo·hi (lo·lo, d'ordre
// 2⁻¹⁶, est abandonné). Les deux arrondis bf16 du GEMM tuilé (activations et
// poids déquantifiés, unité 2⁻⁸ chacun) faisaient diverger le greedy de flint ;
// ici l'erreur relative descend vers 2⁻¹⁶ pour 3× les MMA. Deux tuiles B
// (hi, lo) : 18 Ko de threadgroup, ce qui exclut une variante gate+up fusionnée
// (quatre tuiles, au-delà des 32 Ko).
kernel void gemm_nax_coop_qb_tiled_u4_x3(device const float* A [[buffer(0)]],
                                         device const uint* Bp   [[buffer(1)]],
                                         device const bfloat* Bs [[buffer(2)]],
                                         device const bfloat* Bb [[buffer(3)]],
                                         device float* C         [[buffer(4)]],
                                         constant uint3& mnk     [[buffer(5)]],
                                         uint2 tgid [[threadgroup_position_in_grid]],
                                         uint lid [[thread_index_in_threadgroup]],
                                         uint sgid [[simdgroup_index_in_threadgroup]]) {
  uint M = mnk.x, N = mnk.y, K = mnk.z;
  constexpr uint BN = 64, BK = 64, BKP = BK + 8, WN = 2;
  threadgroup metal::vec<bfloat,8> Wh8[BN * BKP / 8];
  threadgroup metal::vec<bfloat,8> Wl8[BN * BKP / 8];
  threadgroup bfloat* Wh = (threadgroup bfloat*)Wh8;
  threadgroup bfloat* Wl = (threadgroup bfloat*)Wl8;
  uint sg_m = sgid / WN;
  uint sg_n = sgid - sg_m * WN;
  uint m0 = tgid.x * 64u + sg_m * 32u;
  uint n0 = tgid.y * BN;
  uint ns = sg_n * 32u;
  metal::vec<float,8> c0a = float(0), c0b = float(0), c1a = float(0), c1b = float(0);
  for (uint k = 0; k < K; k += BK) {
    load_b_tile_quant_u4_gs64_split<true>(Wh, Wl, Bp, Bs, Bb, n0, k, N, K, lid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kk = 0; kk < BK; kk += 16u) {
      metal::vec<bfloat,8> a0h, a0l, a1h, a1l, b0h, b1h, b0l, b1l;
      NaxFrag::load_a_f32_masked<true>(a0h, a0l, A, m0, k + kk, K, M);
      NaxFrag::load_a_f32_masked<true>(a1h, a1l, A, m0 + 16u, k + kk, K, M);
      NaxFrag::load_a_tg(b0h, Wh + ns * BKP + kk, int(BKP));
      NaxFrag::load_a_tg(b1h, Wh + (ns + 16u) * BKP + kk, int(BKP));
      NaxFrag::load_a_tg(b0l, Wl + ns * BKP + kk, int(BKP));
      NaxFrag::load_a_tg(b1l, Wl + (ns + 16u) * BKP + kk, int(BKP));
      NaxFrag::mma(c0a, c0b, a0h, b0h, b1h);
      NaxFrag::mma(c0a, c0b, a0h, b0l, b1l);
      NaxFrag::mma(c0a, c0b, a0l, b0h, b1h);
      NaxFrag::mma(c1a, c1b, a1h, b0h, b1h);
      NaxFrag::mma(c1a, c1b, a1h, b0l, b1l);
      NaxFrag::mma(c1a, c1b, a1l, b0h, b1h);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  NaxFrag::store_c_masked(c0a, C, m0, n0 + ns, N, M);
  NaxFrag::store_c_masked(c0b, C, m0, n0 + ns + 16u, N, M);
  NaxFrag::store_c_masked(c1a, C, m0 + 16u, n0 + ns, N, M);
  NaxFrag::store_c_masked(c1b, C, m0 + 16u, n0 + ns + 16u, N, M);
}
// MLP dense fusionné gate+up+SwiGLU (prefill) : mêmes tuiles 64×64×64 et
// fragments que `gemm_nax_coop_qb_tiled_u4`, deux tuiles B (gate, up) par pas
// de K, SwiGLU en f32 sur les accumulateurs et sortie H bf16 [M, N] prête pour
// le down_proj. Évite deux sorties f32 [M, N], la passe swiglu et la conversion
// bf16 de H ; bit-identique aux passes séparées (oracle dans les tests).
kernel void gemm_nax_coop_qb_tiled_u4_gate_up_swiglu(
    device const bfloat* A [[buffer(0)]],
    device const uint* Gp   [[buffer(1)]],
    device const bfloat* Gs [[buffer(2)]],
    device const bfloat* Gb [[buffer(3)]],
    device const uint* Up   [[buffer(4)]],
    device const bfloat* Us [[buffer(5)]],
    device const bfloat* Ub [[buffer(6)]],
    device bfloat* H        [[buffer(7)]],
    constant uint3& mnk     [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]]) {
  uint M = mnk.x, N = mnk.y, K = mnk.z;
  constexpr uint BN = 64, BK = 64, BKP = BK + 8, WN = 2;
  threadgroup metal::vec<bfloat,8> Wg8[BN * BKP / 8];
  threadgroup metal::vec<bfloat,8> Wu8[BN * BKP / 8];
  threadgroup bfloat* Wg = (threadgroup bfloat*)Wg8;
  threadgroup bfloat* Wu = (threadgroup bfloat*)Wu8;
  uint sg_m = sgid / WN;
  uint sg_n = sgid - sg_m * WN;
  uint m0 = tgid.x * 64u + sg_m * 32u;
  uint n0 = tgid.y * BN;
  uint ns = sg_n * 32u;
  metal::vec<float,8> g0a = float(0), g0b = float(0), g1a = float(0), g1b = float(0);
  metal::vec<float,8> u0a = float(0), u0b = float(0), u1a = float(0), u1b = float(0);
  for (uint k = 0; k < K; k += BK) {
    load_b_tile_quant_u4_gs64_vec(Wg, Gp, Gs, Gb, n0, k, N, K, lid);
    load_b_tile_quant_u4_gs64_vec(Wu, Up, Us, Ub, n0, k, N, K, lid);
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
      NaxFrag::load_a_tg(b0, Wg + ns * BKP + kk, int(BKP));
      NaxFrag::load_a_tg(b1, Wg + (ns + 16u) * BKP + kk, int(BKP));
      NaxFrag::mma(g0a, g0b, af0, b0, b1);
      NaxFrag::mma(g1a, g1b, af1, b0, b1);
      NaxFrag::load_a_tg(b0, Wu + ns * BKP + kk, int(BKP));
      NaxFrag::load_a_tg(b1, Wu + (ns + 16u) * BKP + kk, int(BKP));
      NaxFrag::mma(u0a, u0b, af0, b0, b1);
      NaxFrag::mma(u1a, u1b, af1, b0, b1);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  NaxFrag::store_swiglu_bf16_masked(g0a, u0a, H, m0, n0 + ns, N, M);
  NaxFrag::store_swiglu_bf16_masked(g0b, u0b, H, m0, n0 + ns + 16u, N, M);
  NaxFrag::store_swiglu_bf16_masked(g1a, u1a, H, m0 + 16u, n0 + ns, N, M);
  NaxFrag::store_swiglu_bf16_masked(g1b, u1b, H, m0 + 16u, n0 + ns + 16u, N, M);
}
// Variante gs64 pour K aligné sur 64 mais non sur 512. La réduction garde des
// fenêtres de 512 pour préserver le chemin nominal ; la dernière fenêtre est
// bornée et les chargeurs injectent zéro pour toute lane au-delà de K.
kernel void gemm_nax_coop_qb_tiled_u4_align64(
    device const bfloat* A [[buffer(0)]],
    device const uint* Bp   [[buffer(1)]],
    device const bfloat* Bs [[buffer(2)]],
    device const bfloat* Bb [[buffer(3)]],
    device float* C         [[buffer(4)]],
    constant uint3& mnk     [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]]) {
  uint M = mnk.x, N = mnk.y, K = mnk.z;
  constexpr uint BN = 64, BK = 64, BKP = BK + 8, REDUCTION_BLOCK = 512, WM = 2, WN = 2;
  threadgroup metal::vec<bfloat,8> Ws8[BN * BKP / 8];
  threadgroup bfloat* Ws = (threadgroup bfloat*)Ws8;
  uint sg_m = sgid / WN;
  uint sg_n = sgid - sg_m * WN;
  uint m0 = tgid.x * 64u + sg_m * 32u;
  uint n0 = tgid.y * BN;
  uint ns = sg_n * 32u;
  metal::vec<float,8> c0a = float(0), c0b = float(0), c1a = float(0), c1b = float(0);
  for (uint kb = 0; kb < K; kb += REDUCTION_BLOCK) {
    uint kend = min(kb + REDUCTION_BLOCK, K);
    for (uint k = kb; k < kend; k += BK) {
      load_b_tile_quant_u4_gs64_vec(Ws, Bp, Bs, Bb, n0, k, N, K, lid);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint kk = 0; kk < BK; kk += 16u) {
        metal::vec<bfloat,8> af0, af1, b0, b1;
        NaxFrag::load_a_masked_mk(af0, A, m0, k + kk, K, M);
        NaxFrag::load_a_masked_mk(af1, A, m0 + 16u, k + kk, K, M);
        NaxFrag::load_a_tg(b0, Ws + ns * BKP + kk, int(BKP));
        NaxFrag::load_a_tg(b1, Ws + (ns + 16u) * BKP + kk, int(BKP));
        NaxFrag::mma(c0a, c0b, af0, b0, b1);
        NaxFrag::mma(c1a, c1b, af1, b0, b1);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  NaxFrag::store_c_masked(c0a, C, m0, n0 + ns, N, M);
  NaxFrag::store_c_masked(c0b, C, m0, n0 + ns + 16u, N, M);
  NaxFrag::store_c_masked(c1a, C, m0 + 16u, n0 + ns, N, M);
  NaxFrag::store_c_masked(c1b, C, m0 + 16u, n0 + ns + 16u, N, M);
}
// GEMM « skinny » u4 gs64 pour la vérification spéculative (M ≤ 16 par tuile) :
// C[M,N] = A[M,K]·deq(B[N,K])ᵀ avec A f32 lu directement. Grille (N/32, ceil(M/16)),
// threadgroup = 4 simdgroups qui se partagent K (split-K, K % 512 == 0), chacun
// accumulant un fragment 16×32 ; réduction en mémoire threadgroup, écriture des
// seules lignes < M. X3 : A et B scindés en bf16 hi + lo et trois MMA
// (hi·hi + hi·lo + lo·hi), précision proche du f32 pour le tri greedy.
template <bool X3>
static inline void gemm_nax_skinny_u4_body(device const float* A, device const uint* Bp,
                                           device const bfloat* Bs, device const bfloat* Bb,
                                           device float* C, uint M, uint N, uint K,
                                           uint2 tgid, uint sgid, uint lid,
                                           threadgroup float* red) {
  constexpr uint SG = 4u;
  uint m0 = tgid.y * 16u, n0 = tgid.x * 32u;
  uint span = K / SG;
  uint kb = sgid * span, ke = kb + span;
  metal::vec<float,8> Cn0 = float(0), Cn1 = float(0);
  for (uint k = kb; k < ke; k += 16u) {
    metal::vec<bfloat,8> Ah, Al, Bh0, Bl0, Bh1, Bl1;
    NaxFrag::load_a_f32_masked<X3>(Ah, Al, A, m0, k, K, M);
    NaxFrag::load_b_u4_split<X3>(Bh0, Bl0, Bp, Bs, Bb, n0, k, K);
    NaxFrag::load_b_u4_split<X3>(Bh1, Bl1, Bp, Bs, Bb, n0 + 16u, k, K);
    NaxFrag::mma(Cn0, Cn1, Ah, Bh0, Bh1);
    if (X3) {
      NaxFrag::mma(Cn0, Cn1, Ah, Bl0, Bl1);
      NaxFrag::mma(Cn0, Cn1, Al, Bh0, Bh1);
    }
  }
  threadgroup float* mine = red + sgid * 512u;
  NaxFrag::store_c_tg(Cn0, mine, 32);
  NaxFrag::store_c_tg(Cn1, mine + 16u, 32);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint tid = sgid * 32u + lid;
  for (uint idx = tid; idx < 512u; idx += SG * 32u) {
    uint m = idx >> 5u, nn = idx & 31u;
    if (m0 + m < M) {
      C[(m0 + m) * N + n0 + nn] = red[idx] + red[512u + idx] + red[1024u + idx] + red[1536u + idx];
    }
  }
}
kernel void gemm_nax_skinny_u4(device const float* A [[buffer(0)]],
                               device const uint* Bp   [[buffer(1)]],
                               device const bfloat* Bs [[buffer(2)]],
                               device const bfloat* Bb [[buffer(3)]],
                               device float* C         [[buffer(4)]],
                               constant uint4& dims    [[buffer(5)]],
                               constant uint& batch    [[buffer(6)]],
                               uint2 tgid [[threadgroup_position_in_grid]],
                               uint sgid [[simdgroup_index_in_threadgroup]],
                               uint lid [[thread_index_in_simdgroup]]) {
  threadgroup float red[4u * 512u];
  gemm_nax_skinny_u4_body<false>(A, Bp, Bs, Bb, C, batch, dims.x, dims.y, tgid, sgid, lid, red);
}
kernel void gemm_nax_skinny_u4_x3(device const float* A [[buffer(0)]],
                                  device const uint* Bp   [[buffer(1)]],
                                  device const bfloat* Bs [[buffer(2)]],
                                  device const bfloat* Bb [[buffer(3)]],
                                  device float* C         [[buffer(4)]],
                                  constant uint4& dims    [[buffer(5)]],
                                  constant uint& batch    [[buffer(6)]],
                                  uint2 tgid [[threadgroup_position_in_grid]],
                                  uint sgid [[simdgroup_index_in_threadgroup]],
                                  uint lid [[thread_index_in_simdgroup]]) {
  threadgroup float red[4u * 512u];
  gemm_nax_skinny_u4_body<true>(A, Bp, Bs, Bb, C, batch, dims.x, dims.y, tgid, sgid, lid, red);
}
// Variante « ×2 » du skinny u4 gs64 : l'entier q ∈ [0, 15] est exact en bf16, donc
// le MMA porte sur q brut et seule l'activation est scindée (hi + lo, deux MMA au
// lieu de trois, sans déquantification ni scission de B). L'échelle et le biais
// changent tous les 64 de K : chaque groupe accumule P = Σ A·q dans ses propres
// fragments, puis l'épilogue f32 ajoute s·P + b·Σ_g A au résultat. Σ_g A (somme
// f32 de l'activation sur le groupe) se réduit par shuffles : les lanes d'une même
// ligne de fragment ne diffèrent que par les bits 0 et 3 (cf. `get_coord`).
// Les produits A_hi·q et A_lo·q sont exacts en f32 ; l'erreur vient de A_hi + A_lo
// (2⁻¹⁶ relatif) et des sommes f32, contre 2⁻¹⁶ sur A **et** sur B pour le ×3.
// Split-K : span = K/4 multiple de 128 (K % 512 == 0), aucun groupe n'est coupé.
static inline void gemm_nax_skinny_u4_x2_body(device const float* A, device const uint* Bp,
                                              device const bfloat* Bs, device const bfloat* Bb,
                                              device float* C, uint M, uint N, uint K,
                                              uint2 tgid, uint sgid, uint lid,
                                              threadgroup float* red) {
  constexpr uint SG = 4u;
  uint m0 = tgid.y * 16u, n0 = tgid.x * 32u;
  uint span = K / SG;
  uint kb = sgid * span, ke = kb + span;
  uint groups = K / 64u;
  uint packed_cols = K / 8u;
  short2 sc = NaxFrag::get_coord();
  // Lignes A de ce thread (fragment A et fragment C partagent le mapping lane→ligne).
  uint row0 = m0 + uint(sc.y), row1 = row0 + 8u;
  // Colonnes C de ce thread : n0 + sc.x + j (Cn0) et + 16 (Cn1).
  uint nc = n0 + uint(sc.x);
  // Lignes B (poids) chargées par ce thread : n + sc.y + i·8, fragments 0 et 1.
  uint nb = n0 + uint(sc.y);
  metal::vec<float,8> Cn0 = float(0), Cn1 = float(0);
  for (uint g0 = kb; g0 < ke; g0 += 64u) {
    metal::vec<float,8> P0 = float(0), P1 = float(0);
    float sa0 = 0.0f, sa1 = 0.0f;
    for (uint k = g0; k < g0 + 64u; k += 16u) {
      metal::vec<bfloat,8> Ah, Al, Q0, Q1;
      uint kk = k + uint(sc.x);
      for (short i = 0; i < 2; i++) {
        uint row = i == 0 ? row0 : row1;
        const device float* pa = A + row * K + kk;
        float rs = 0.0f;
        for (short j = 0; j < 4; j++) {
          float x = (row < M) ? pa[j] : 0.0f;
          bfloat h = bfloat(x);
          Ah[i * 4 + j] = h;
          Al[i * 4 + j] = bfloat(x - float(h));
          rs += x;
        }
        if (i == 0) { sa0 += rs; } else { sa1 += rs; }
      }
      uint shift = (kk & 7u) * 4u;
      for (short i = 0; i < 2; i++) {
        uint w0 = Bp[(nb + uint(i) * 8u) * packed_cols + (kk >> 3)] >> shift;
        uint w1 = Bp[(nb + 16u + uint(i) * 8u) * packed_cols + (kk >> 3)] >> shift;
        for (short j = 0; j < 4; j++) {
          Q0[i * 4 + j] = bfloat(float((w0 >> (4u * uint(j))) & 0x0fu));
          Q1[i * 4 + j] = bfloat(float((w1 >> (4u * uint(j))) & 0x0fu));
        }
      }
      NaxFrag::mma(P0, P1, Ah, Q0, Q1);
      NaxFrag::mma(P0, P1, Al, Q0, Q1);
    }
    sa0 += simd_shuffle_xor(sa0, ushort(1));
    sa0 += simd_shuffle_xor(sa0, ushort(8));
    sa1 += simd_shuffle_xor(sa1, ushort(1));
    sa1 += simd_shuffle_xor(sa1, ushort(8));
    uint g = g0 >> 6u;
    for (short j = 0; j < 4; j++) {
      uint i0 = (nc + uint(j)) * groups + g;
      uint i1 = (nc + 16u + uint(j)) * groups + g;
      float s0 = float(Bs[i0]), b0 = float(Bb[i0]);
      float s1 = float(Bs[i1]), b1 = float(Bb[i1]);
      Cn0[j] += fma(s0, P0[j], b0 * sa0);
      Cn0[4 + j] += fma(s0, P0[4 + j], b0 * sa1);
      Cn1[j] += fma(s1, P1[j], b1 * sa0);
      Cn1[4 + j] += fma(s1, P1[4 + j], b1 * sa1);
    }
  }
  threadgroup float* mine = red + sgid * 512u;
  NaxFrag::store_c_tg(Cn0, mine, 32);
  NaxFrag::store_c_tg(Cn1, mine + 16u, 32);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint tid = sgid * 32u + lid;
  for (uint idx = tid; idx < 512u; idx += SG * 32u) {
    uint m = idx >> 5u, nn = idx & 31u;
    if (m0 + m < M) {
      C[(m0 + m) * N + n0 + nn] = red[idx] + red[512u + idx] + red[1024u + idx] + red[1536u + idx];
    }
  }
}
kernel void gemm_nax_skinny_u4_x2(device const float* A [[buffer(0)]],
                                  device const uint* Bp   [[buffer(1)]],
                                  device const bfloat* Bs [[buffer(2)]],
                                  device const bfloat* Bb [[buffer(3)]],
                                  device float* C         [[buffer(4)]],
                                  constant uint4& dims    [[buffer(5)]],
                                  constant uint& batch    [[buffer(6)]],
                                  uint2 tgid [[threadgroup_position_in_grid]],
                                  uint sgid [[simdgroup_index_in_threadgroup]],
                                  uint lid [[thread_index_in_simdgroup]]) {
  threadgroup float red[4u * 512u];
  gemm_nax_skinny_u4_x2_body(A, Bp, Bs, Bb, C, batch, dims.x, dims.y, tgid, sgid, lid, red);
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
// MoE groupé u4 TUILÉ (flint) : les kernels ci-dessus font un simdgroup par
// tuile 16×32 et re-déquantifient le poids de l'expert toutes les 16 lignes,
// relisant aussi A pour chaque tuile N : borné mémoire/L2. Ici une tuile de BM
// lignes (un seul expert, `moe_g_offsets` padde chaque expert à BM) × 64
// colonnes par threadgroup de 4 simdgroups (2×2) : le poids est déquantifié
// une fois par BM lignes en mémoire threadgroup (chargeur u4 vectorisé), A une
// fois par tuile. Même arithmétique que les kernels 16×32 (bfloat(x),
// bfloat(q·s+b), MMA 16×32×16 dans le même ordre de K) : sorties identiques.
// Préconditions côté Rust : N % 64 == 0, K % 64 == 0.
constexpr constant uint MOE_TILED_BN = 64u, MOE_TILED_BK = 64u, MOE_TILED_BKP = 72u;
// gate+up fusionnés + SwiGLU. A est gathéré (perm/top_k) f32→bf16 en mémoire
// threadgroup : BM×72 bf16 + deux tuiles poids 64×72 (22,5 Ko pour BM = 32).
template <uint BM>
static void moe_grouped_gate_up_swiglu_u4_tiled(
    device const float* input, device const uint* Gp, device const bfloat* Gs,
    device const bfloat* Gb, device const uint* Up, device const bfloat* Us,
    device const bfloat* Ub, device bfloat* H, device const uint* tile_expert,
    device const uint* perm, uint4 mnkt, uint2 tgid, uint lid, uint sgid,
    threadgroup bfloat* As, threadgroup bfloat* Wg, threadgroup bfloat* Wu) {
  constexpr uint BN = MOE_TILED_BN, BK = MOE_TILED_BK, BKP = MOE_TILED_BKP;
  constexpr uint FM = BM / 32u;          // fragments M par simdgroup
  constexpr uint TPR = 128u / BM;        // threads par ligne de A
  constexpr uint CPT = BK / TPR;         // colonnes A par thread (16 ou 32)
  uint N = mnkt.y, K = mnkt.z, top_k = mnkt.w;
  uint e = tile_expert[tgid.x];
  // Tuile sentinelle : tout le threadgroup sort avant la moindre barrière.
  if (e == 0xFFFFFFFFu) { return; }
  uint m0 = tgid.x * BM, n0 = tgid.y * BN;
  uint pe = e * N * (K / 8u);
  uint ge = e * N * (K / 64u);
  uint sg_m = sgid >> 1u, sg_n = sgid & 1u;
  uint ms = sg_m * 16u * FM, ns = sg_n * 32u;
  // Ligne de A chargée par ce thread : perm lu une fois, padding → zéros.
  uint arow = lid / TPR, acol = (lid % TPR) * CPT;
  uint slot = perm[m0 + arow];
  bool avalid = slot != 0xFFFFFFFFu;
  const device float* asrc = input + (avalid ? (slot / top_k) * K + acol : 0u);
  threadgroup metal::vec<bfloat,8>* adst =
      (threadgroup metal::vec<bfloat,8>*)(As + arow * BKP + acol);
  metal::vec<float,8> g[FM][2], u[FM][2];
  for (uint f = 0; f < FM; f++) {
    g[f][0] = float(0); g[f][1] = float(0); u[f][0] = float(0); u[f][1] = float(0);
  }
  for (uint k = 0; k < K; k += BK) {
    for (uint v = 0; v < CPT / 8u; v++) {
      metal::vec<bfloat,8> out = bfloat(0.0f);
      if (avalid) {
        const device float4* s4 = (const device float4*)(asrc + k + v * 8u);
        float4 lo = s4[0], hi = s4[1];
        out = metal::vec<bfloat,8>(bfloat(lo.x), bfloat(lo.y), bfloat(lo.z), bfloat(lo.w),
                                   bfloat(hi.x), bfloat(hi.y), bfloat(hi.z), bfloat(hi.w));
      }
      adst[v] = out;
    }
    load_b_tile_quant_u4_gs64_vec(Wg, Gp + pe, Gs + ge, Gb + ge, n0, k, N, K, lid);
    load_b_tile_quant_u4_gs64_vec(Wu, Up + pe, Us + ge, Ub + ge, n0, k, N, K, lid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kk = 0; kk < BK; kk += 16u) {
      metal::vec<bfloat,8> af[FM], b0, b1;
      for (uint f = 0; f < FM; f++) {
        NaxFrag::load_a_tg(af[f], As + (ms + 16u * f) * BKP + kk, int(BKP));
      }
      NaxFrag::load_a_tg(b0, Wg + ns * BKP + kk, int(BKP));
      NaxFrag::load_a_tg(b1, Wg + (ns + 16u) * BKP + kk, int(BKP));
      for (uint f = 0; f < FM; f++) { NaxFrag::mma(g[f][0], g[f][1], af[f], b0, b1); }
      NaxFrag::load_a_tg(b0, Wu + ns * BKP + kk, int(BKP));
      NaxFrag::load_a_tg(b1, Wu + (ns + 16u) * BKP + kk, int(BKP));
      for (uint f = 0; f < FM; f++) { NaxFrag::mma(u[f][0], u[f][1], af[f], b0, b1); }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint f = 0; f < FM; f++) {
    device bfloat* dst = H + (m0 + ms + 16u * f) * N + n0 + ns;
    NaxFrag::store_swiglu_bf16(g[f][0], u[f][0], dst, int(N));
    NaxFrag::store_swiglu_bf16(g[f][1], u[f][1], dst + 16u, int(N));
  }
}
// down + scatter par slot : A = hidden2 bf16 contigu [max_m, K], lu en
// fragments directement (pas de gather), une tuile poids 64×72 partagée.
template <uint BM>
static void moe_grouped_scatter_u4_tiled(
    device const bfloat* A, device const uint* Bp, device const bfloat* Bs,
    device const bfloat* Bb, device float* C, device const uint* tile_expert,
    device const uint* perm, uint3 mnk, uint2 tgid, uint lid, uint sgid,
    threadgroup bfloat* Ws) {
  constexpr uint BN = MOE_TILED_BN, BK = MOE_TILED_BK, BKP = MOE_TILED_BKP;
  constexpr uint FM = BM / 32u;
  uint N = mnk.y, K = mnk.z;
  uint e = tile_expert[tgid.x];
  if (e == 0xFFFFFFFFu) { return; }
  uint m0 = tgid.x * BM, n0 = tgid.y * BN;
  uint pe = e * N * (K / 8u);
  uint ge = e * N * (K / 64u);
  uint sg_m = sgid >> 1u, sg_n = sgid & 1u;
  uint ms = m0 + sg_m * 16u * FM, ns = sg_n * 32u;
  metal::vec<float,8> c[FM][2];
  for (uint f = 0; f < FM; f++) { c[f][0] = float(0); c[f][1] = float(0); }
  for (uint k = 0; k < K; k += BK) {
    load_b_tile_quant_u4_gs64_vec(Ws, Bp + pe, Bs + ge, Bb + ge, n0, k, N, K, lid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kk = 0; kk < BK; kk += 16u) {
      metal::vec<bfloat,8> b0, b1;
      NaxFrag::load_a_tg(b0, Ws + ns * BKP + kk, int(BKP));
      NaxFrag::load_a_tg(b1, Ws + (ns + 16u) * BKP + kk, int(BKP));
      for (uint f = 0; f < FM; f++) {
        metal::vec<bfloat,8> af;
        NaxFrag::load_a(af, A + (ms + 16u * f) * K + k + kk, int(K));
        NaxFrag::mma(c[f][0], c[f][1], af, b0, b1);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint f = 0; f < FM; f++) {
    NaxFrag::store_c_scattered(c[f][0], C, perm, ms + 16u * f, n0 + ns, N);
    NaxFrag::store_c_scattered(c[f][1], C, perm, ms + 16u * f, n0 + ns + 16u, N);
  }
}
#define MOE_TILED_KERNELS(BM)                                                              \
kernel void gemm_nax_coop_qb_grouped_gate_up_swiglu_u4_bm##BM(                             \
    device const float* input [[buffer(0)]], device const uint* Gp [[buffer(1)]],          \
    device const bfloat* Gs [[buffer(2)]], device const bfloat* Gb [[buffer(3)]],          \
    device const uint* Up [[buffer(4)]], device const bfloat* Us [[buffer(5)]],            \
    device const bfloat* Ub [[buffer(6)]], device bfloat* H [[buffer(7)]],                 \
    device const uint* tile_expert [[buffer(8)]], device const uint* perm [[buffer(9)]],   \
    constant uint4& mnkt [[buffer(10)]], uint2 tgid [[threadgroup_position_in_grid]],      \
    uint lid [[thread_index_in_threadgroup]], uint sgid [[simdgroup_index_in_threadgroup]]) { \
  threadgroup metal::vec<bfloat,8> As8[BM * MOE_TILED_BKP / 8];                            \
  threadgroup metal::vec<bfloat,8> Wg8[MOE_TILED_BN * MOE_TILED_BKP / 8];                  \
  threadgroup metal::vec<bfloat,8> Wu8[MOE_TILED_BN * MOE_TILED_BKP / 8];                  \
  moe_grouped_gate_up_swiglu_u4_tiled<BM>(input, Gp, Gs, Gb, Up, Us, Ub, H, tile_expert,   \
      perm, mnkt, tgid, lid, sgid, (threadgroup bfloat*)As8, (threadgroup bfloat*)Wg8,     \
      (threadgroup bfloat*)Wu8);                                                           \
}                                                                                          \
kernel void gemm_nax_coop_qb_grouped_scatter_u4_bm##BM(                                    \
    device const bfloat* A [[buffer(0)]], device const uint* Bp [[buffer(1)]],             \
    device const bfloat* Bs [[buffer(2)]], device const bfloat* Bb [[buffer(3)]],          \
    device float* C [[buffer(4)]], device const uint* tile_expert [[buffer(5)]],           \
    device const uint* perm [[buffer(6)]], constant uint3& mnk [[buffer(7)]],              \
    uint2 tgid [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]], \
    uint sgid [[simdgroup_index_in_threadgroup]]) {                                        \
  threadgroup metal::vec<bfloat,8> Ws8[MOE_TILED_BN * MOE_TILED_BKP / 8];                  \
  moe_grouped_scatter_u4_tiled<BM>(A, Bp, Bs, Bb, C, tile_expert, perm, mnk, tgid, lid,    \
      sgid, (threadgroup bfloat*)Ws8);                                                     \
}
// NOTE: BM = 64 mesuré 3,4× plus lent que BM = 32 sur flint-int4 (MoE routé à
// 32k 4,5 s en 16×32, 3,2 s en BM 32, 15 s en BM 64) : seul BM = 32 est instancié.
MOE_TILED_KERNELS(32)
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
// 1 threadgroup, 1 thread : scan exclusif des counts padés à `tile_m` lignes
// (16 pour les kernels 16×32, BM pour les tuilés : une tuile ne traverse
// jamais deux experts) → padded_offset + remplit tile_expert + cursor=
// padded_offset. experts petit (256).
kernel void moe_g_offsets(device const uint* counts [[buffer(0)]],
                          device uint* padded_offset [[buffer(1)]],
                          device uint* tile_expert [[buffer(2)]],
                          device atomic_uint* cursor [[buffer(3)]],
                          constant uint2& experts_tile [[buffer(4)]], // (experts, tile_m)
                          uint gid [[thread_position_in_grid]]) {
  if (gid != 0) { return; }
  uint experts = experts_tile.x, tile_m = experts_tile.y;
  uint moff = 0u, toff = 0u;
  for (uint e = 0; e < experts; e++) {
    padded_offset[e] = moff;
    atomic_store_explicit(&cursor[e], moff, memory_order_relaxed);
    uint tiles = (counts[e] + tile_m - 1u) / tile_m;
    for (uint t = 0; t < tiles; t++) { tile_expert[toff + t] = e; }
    moff += tiles * tile_m;
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
