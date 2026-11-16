//! Tirage top-k/top-p sur GPU des codebooks du code predictor Qwen3-TTS.
//!
//! Un threadgroup reproduit `token_distribution_top_k_top_p` puis
//! `sample_from_token_distribution` (`crate::sampling`) sur un vocabulaire d'au
//! plus [`TTS_SAMPLER_MAX_VOCAB`] logits : softmax avec température sur tout le
//! vocabulaire, tri complet par (probabilité décroissante, indice croissant),
//! troncature top-k, noyau top-p, renormalisation puis tirage cumulatif.
//! L'uniforme est `splitmix_unit_f32` sur l'état SplitMix64 du tirage, le
//! même que `DeterministicSampler::next_unit_f32`.
//!
//! Seuls diffèrent les arrondis : le maximum et la somme de la softmax sont
//! réduits en arbre au lieu d'un cumul séquentiel, et `exp` vient de Metal.
//! Les tirages peuvent donc basculer sur une quasi-égalité ; l'oracle
//! `fused_codebooks_match_cpu_oracle` mesure l'écart sur les probabilités et
//! les tirages. Le kernel est compilé sans fast-math.
//!
//! La pénalité de répétition n'existe pas ici : la référence ne l'applique
//! qu'à cb0 (historique talker), jamais aux codebooks du code predictor.

use super::kernels::compile_kernels;
use super::*;

/// Vocabulaire maximal trié dans un threadgroup (2 éléments par thread).
pub(crate) const TTS_SAMPLER_MAX_VOCAB: usize = 2048;
/// Couples (probabilité, indice) écrits par tirage dans la trace d'oracle.
pub(crate) const TTS_SAMPLER_TRACE_K: usize = 64;
const THREADS: usize = TTS_SAMPLER_MAX_VOCAB / 2;

const TTS_SAMPLER_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint SORT_N = 2048u;
constant uint THREADS = 1024u;
constant uint TRACE_K = 64u;
constant uint PARTIAL_K = 64u;

static inline float tts_splitmix_unit_f32(ulong state) {
    state += 0x9E3779B97F4A7C15ul;
    ulong z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ul;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBul;
    z = z ^ (z >> 31);
    const uint mantissa = uint(z >> 40) & 0x00ffffffu;
    return float(mantissa) / 16777216.0f;
}

// Ordre `by_rank` du CPU : probabilité décroissante, puis indice croissant.
static inline bool tts_ranks_before(float pa, uint ia, float pb, uint ib) {
    return pa > pb || (pa == pb && ia < ib);
}

// dims = (count, top_k, slot, drapeaux : bit 0 trace, bit 1 sélection
// partielle), params = (temperature, top_p).
// `rng_state` est l'état AVANT le tirage (DeterministicSampler::state()).
kernel void tts_sample_codebook_f32(
    device const float* logits [[buffer(0)]],
    device uint* codes [[buffer(1)]],
    device float* trace [[buffer(2)]],
    constant uint4& dims [[buffer(3)]],
    constant float2& params [[buffer(4)]],
    constant ulong& rng_state [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float probs[SORT_N];
    threadgroup uint ids[SORT_N];
    threadgroup float reduce[THREADS];

    const uint count = dims.x;
    const float temperature = max(params.x, 0.0001f);
    const float top_p = params.y;

    float local_max = -INFINITY;
    for (uint i = tid; i < count; i += THREADS) {
        local_max = fmax(local_max, logits[i]);
    }
    reduce[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = THREADS / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            reduce[tid] = fmax(reduce[tid], reduce[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float max_value = reduce[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_sum = 0.0f;
    for (uint i = tid; i < SORT_N; i += THREADS) {
        float value = -1.0f;
        if (i < count) {
            value = precise::exp((logits[i] - max_value) / temperature);
            local_sum += value;
        }
        probs[i] = value;
        ids[i] = i;
    }
    reduce[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = THREADS / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            reduce[tid] += reduce[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float total = reduce[0];
    for (uint i = tid; i < count; i += THREADS) {
        probs[i] = probs[i] / total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if ((dims.w & 2u) != 0u && dims.y <= PARTIAL_K) {
        // Sélection partielle exacte des PARTIAL_K premiers rangs : chaque
        // simdgroup trie son bloc de 64, puis cinq niveaux fusionnent deux blocs
        // en gardant les 64 meilleurs. L'ordre (probabilité, indice) est strict :
        // ces rangs sont ceux du tri complet, au bit près.
        const uint lane = tid % 32u;
        const uint simd = tid / 32u;
        for (uint k = 2u; k <= PARTIAL_K; k <<= 1u) {
            for (uint j = k >> 1u; j > 0u; j >>= 1u) {
                const uint a = (tid / j) * 2u * j + (tid % j);
                const uint b = a + j;
                const float pa = probs[a];
                const float pb = probs[b];
                const uint ia = ids[a];
                const uint ib = ids[b];
                const bool descending = k == PARTIAL_K || (a & k) == 0u;
                if (descending != tts_ranks_before(pa, ia, pb, ib)) {
                    probs[a] = pb;
                    probs[b] = pa;
                    ids[a] = ib;
                    ids[b] = ia;
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        for (uint span = 1u; span < SORT_N / PARTIAL_K; span <<= 1u) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const uint block = simd * 2u * span;
            if (block < SORT_N / PARTIAL_K) {
                const uint base_a = block * PARTIAL_K;
                const uint base_b = (block + span) * PARTIAL_K;
                // Bloc A décroissant contre B renversé : le meilleur de chaque
                // couple forme une suite bitonique des 64 premiers de A ∪ B.
                for (uint r = lane; r < PARTIAL_K; r += 32u) {
                    const float pa = probs[base_a + r];
                    const uint ia = ids[base_a + r];
                    const float pb = probs[base_b + PARTIAL_K - 1u - r];
                    const uint ib = ids[base_b + PARTIAL_K - 1u - r];
                    if (!tts_ranks_before(pa, ia, pb, ib)) {
                        probs[base_a + r] = pb;
                        ids[base_a + r] = ib;
                    }
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
                for (uint j = PARTIAL_K >> 1u; j > 0u; j >>= 1u) {
                    const uint a = base_a + (lane / j) * 2u * j + (lane % j);
                    const uint b = a + j;
                    const float pa = probs[a];
                    const float pb = probs[b];
                    const uint ia = ids[a];
                    const uint ib = ids[b];
                    if (!tts_ranks_before(pa, ia, pb, ib)) {
                        probs[a] = pb;
                        probs[b] = pa;
                        ids[a] = ib;
                        ids[b] = ia;
                    }
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    } else {
    // Tri bitonique complet, décroissant par rang ; le bourrage (-1) finit en queue.
    for (uint k = 2u; k <= SORT_N; k <<= 1u) {
        for (uint j = k >> 1u; j > 0u; j >>= 1u) {
            const uint a = (tid / j) * 2u * j + (tid % j);
            const uint b = a + j;
            const float pa = probs[a];
            const float pb = probs[b];
            const uint ia = ids[a];
            const uint ib = ids[b];
            const bool a_first = tts_ranks_before(pa, ia, pb, ib);
            if (((a & k) == 0u) != a_first) {
                probs[a] = pb;
                probs[b] = pa;
                ids[a] = ib;
                ids[b] = ia;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    }

    if (tid != 0u) {
        return;
    }
    const uint top_k = min(dims.y, count);
    float top_sum = 0.0f;
    for (uint r = 0u; r < top_k; ++r) {
        top_sum += probs[r];
    }
    uint kept = 1u;
    if (top_sum > FLT_EPSILON) {
        float cumulative = 0.0f;
        for (uint r = 0u; r < top_k; ++r) {
            cumulative += probs[r] / top_sum;
            kept = r + 1u;
            if (r == 0u || cumulative < top_p) {
                continue;
            }
            break;
        }
    }
    float kept_sum = 0.0f;
    for (uint r = 0u; r < kept; ++r) {
        kept_sum += probs[r];
    }
    // Repli CPU sur une masse dégénérée : argmax, ici le premier rang.
    const bool degenerate = top_sum <= FLT_EPSILON || kept_sum <= FLT_EPSILON;
    if (degenerate) {
        kept = 1u;
    }
    float normalized_sum = 0.0f;
    for (uint r = 0u; r < kept; ++r) {
        normalized_sum += degenerate ? 1.0f : probs[r] / kept_sum;
    }
    const float roll = tts_splitmix_unit_f32(rng_state) * normalized_sum;
    float acc = 0.0f;
    uint chosen = ids[kept - 1u];
    for (uint r = 0u; r < kept; ++r) {
        const float prob = degenerate ? 1.0f : probs[r] / kept_sum;
        if (prob <= 0.0f) {
            continue;
        }
        acc += prob;
        if (roll <= acc) {
            chosen = ids[r];
            break;
        }
    }
    const uint slot = dims.z;
    codes[slot] = chosen;
    if ((dims.w & 1u) != 0u) {
        device float* row = trace + slot * 2u * TRACE_K;
        for (uint r = 0u; r < TRACE_K; ++r) {
            const bool live = r < kept;
            row[r] = live ? (degenerate ? 1.0f : probs[r] / kept_sum) : 0.0f;
            row[TRACE_K + r] = as_type<float>(live ? ids[r] : 0xffffffffu);
        }
    }
}
"#;

/// Paramètres d'un tirage de codebook ; `top_k` dans `1..count`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TtsCodebookSample {
    pub(crate) temperature: f32,
    pub(crate) top_k: usize,
    pub(crate) top_p: f32,
}

pub(crate) struct TtsSamplerKernel {
    pipeline: ComputePipelineState,
}

impl std::fmt::Debug for TtsSamplerKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TtsSamplerKernel").finish_non_exhaustive()
    }
}

impl DecodeResidentState {
    fn tts_sampler_kernel(&self) -> Result<&TtsSamplerKernel> {
        if let Some(kernel) = self.tts_sampler_kernel.get() {
            return Ok(kernel);
        }
        let pipeline = compile_kernels(
            &self.device,
            TTS_SAMPLER_KERNEL,
            &["tts_sample_codebook_f32"],
            false,
        )?
        .pop()
        .ok_or_else(|| InferError::Metal("pipeline tts_sample_codebook_f32 absent".into()))?;
        if (pipeline.max_total_threads_per_threadgroup() as usize) < THREADS {
            return Err(InferError::Metal(format!(
                "tirage TTS: {THREADS} threads par threadgroup requis"
            )));
        }
        Ok(self
            .tts_sampler_kernel
            .get_or_init(|| TtsSamplerKernel { pipeline }))
    }

    /// Encode le tirage d'un codebook : lit `count` logits, écrit l'indice tiré
    /// dans `codes[slot]` et, si `trace`, la distribution retenue dans
    /// `trace_buffer[slot]` (64 probabilités puis 64 indices en bits f32).
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si `count` dépasse [`TTS_SAMPLER_MAX_VOCAB`], si
    /// `top_k` sort de `1..count` ou si le kernel ne compile pas.
    #[expect(
        clippy::too_many_arguments,
        reason = "buffers d'entrée, de sortie et de trace, slot et paramètres de tirage"
    )]
    pub(crate) fn encode_tts_codebook_sample(
        &self,
        encoder: &ComputeCommandEncoderRef,
        logits: &BufferRef,
        count: usize,
        codes: &BufferRef,
        slot: usize,
        trace: Option<&BufferRef>,
        sample: TtsCodebookSample,
        rng_state: u64,
    ) -> Result<()> {
        if count == 0 || count > TTS_SAMPLER_MAX_VOCAB || sample.top_k == 0 || sample.top_k >= count
        {
            return Err(InferError::Config(format!(
                "tirage TTS GPU: vocab={count}, top_k={} hors domaine",
                sample.top_k
            )));
        }
        let to_u32 = |value: usize, label: &str| {
            u32::try_from(value).map_err(|_| InferError::Dimension(format!("{label} hors u32")))
        };
        let dims = [
            to_u32(count, "tirage TTS vocab")?,
            to_u32(sample.top_k, "tirage TTS top_k")?,
            to_u32(slot, "tirage TTS slot")?,
            u32::from(trace.is_some())
                | (u32::from(crate::runtime_flags::tts_sampler_partial_topk_enabled()) << 1),
        ];
        let params = [sample.temperature, sample.top_p];
        let kernel = self.tts_sampler_kernel()?;
        encoder.set_compute_pipeline_state(&kernel.pipeline);
        encoder.set_buffer(0, Some(logits), 0);
        encoder.set_buffer(1, Some(codes), 0);
        // Sans trace, le kernel n'écrit jamais l'argument 2 ; il reste lié.
        encoder.set_buffer(2, Some(trace.unwrap_or(codes)), 0);
        set_bytes(encoder, 3, &dims);
        set_bytes(encoder, 4, &params);
        set_bytes(encoder, 5, &[rng_state]);
        crate::metal_backend::profile_dispatch();
        encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(THREADS as u64, 1, 1));
        crate::metal_backend::post_dispatch_barrier(encoder);
        Ok(())
    }
}

fn set_bytes<T: Copy>(encoder: &ComputeCommandEncoderRef, index: u64, values: &[T]) {
    encoder.set_bytes(
        index,
        std::mem::size_of_val(values) as u64,
        values.as_ptr().cast::<c_void>(),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sélection partielle contre tri complet : mêmes codes et mêmes traces au
    /// bit près, logits aléatoires avec ex aequo et vocabulaires partiels.
    #[test]
    fn partial_topk_matches_full_sort() -> Result<()> {
        let Some(state) = super::super::tests::try_state()? else {
            return Ok(());
        };
        let mut seed = 0x2545_F491_4F6C_DD1D_u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for case in 0..48_usize {
            let count = [2048, 2048, 1931, 777][case % 4];
            let top_k = [50, 64, 1, 17][case / 4 % 4];
            // Valeurs entières répétées : nombreux ex aequo exacts.
            let logits = (0..count)
                .map(|_| {
                    let raw = next();
                    if case % 3 == 0 {
                        (raw % 7) as f32
                    } else {
                        (raw % 20_000) as f32 / 1000.0 - 10.0
                    }
                })
                .collect::<Vec<_>>();
            let sample = TtsCodebookSample {
                temperature: [0.9, 1.0, 0.3][case % 3],
                top_k,
                top_p: [1.0, 0.8, 0.95][case / 3 % 3],
            };
            let rng = next();
            let run = |partial: bool| -> Result<(Vec<u32>, Vec<u32>)> {
                crate::runtime_flags::with_tts_sampler_partial_topk(partial, || {
                    let input = super::super::tests::lease_with(&state, &logits)?;
                    let codes = state.scratch().lease(1, GpuElement::U32)?;
                    let trace = state
                        .scratch()
                        .lease(2 * TTS_SAMPLER_TRACE_K, GpuElement::F32)?;
                    let command_buffer = state.queue().new_command_buffer();
                    let encoder = command_buffer.new_compute_command_encoder();
                    state.encode_tts_codebook_sample(
                        encoder,
                        input.tensor().buffer(),
                        count,
                        codes.tensor().buffer(),
                        0,
                        Some(trace.tensor().buffer()),
                        sample,
                        rng,
                    )?;
                    encoder.end_encoding();
                    crate::metal_backend::commit_and_wait(command_buffer)?;
                    let codes = crate::metal_backend::read_u32_buffer(codes.tensor().buffer(), 1)?;
                    let trace = crate::metal_backend::read_f32_buffer(
                        trace.tensor().buffer(),
                        2 * TTS_SAMPLER_TRACE_K,
                    )?;
                    Ok((codes, trace.iter().map(|value| value.to_bits()).collect()))
                })
            };
            assert_eq!(run(true)?, run(false)?, "cas {case}");
        }
        Ok(())
    }
    /// Mesure GPU du tirage : 150 tirages (10 trames de 15 têtes) dans un
    /// command buffer, top-64 partiel contre tri complet.
    /// `cargo test --release --all-features tts_sampler_bench -- --ignored --nocapture`
    #[test]
    #[ignore = "bench GPU, à lancer sous verrou"]
    fn tts_sampler_bench() -> Result<()> {
        let Some(state) = super::super::tests::try_state()? else {
            return Ok(());
        };
        let logits = (0..2048_u32)
            .map(|i| ((i.wrapping_mul(2_654_435_761) >> 8) % 20_000) as f32 / 1000.0 - 10.0)
            .collect::<Vec<_>>();
        let sample = TtsCodebookSample {
            temperature: 0.9,
            top_k: 50,
            top_p: 1.0,
        };
        let input = super::super::tests::lease_with(&state, &logits)?;
        let codes = state.scratch().lease(15, GpuElement::U32)?;
        for partial in [true, false, true, false] {
            let mut best = f64::MAX;
            for _ in 0..20 {
                let elapsed = crate::runtime_flags::with_tts_sampler_partial_topk(partial, || {
                    let command_buffer = state.queue().new_command_buffer();
                    let encoder = command_buffer.new_compute_command_encoder();
                    for draw in 0..150_usize {
                        state.encode_tts_codebook_sample(
                            encoder,
                            input.tensor().buffer(),
                            2048,
                            codes.tensor().buffer(),
                            draw % 15,
                            None,
                            sample,
                            draw as u64,
                        )?;
                    }
                    encoder.end_encoding();
                    let started = std::time::Instant::now();
                    crate::metal_backend::commit_and_wait(command_buffer)?;
                    Ok::<_, InferError>(started.elapsed().as_secs_f64())
                })?;
                best = best.min(elapsed);
            }
            println!(
                "tts_sampler partial={partial} us_par_tirage={:.2} us_par_trame={:.1}",
                best * 1e6 / 150.0,
                best * 1e6 / 10.0
            );
        }
        Ok(())
    }
}
