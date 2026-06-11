//! Génération, préfill et tête MTP du décodeur causal.

use super::attention_ops::full_attention_context_cached;
#[cfg(all(target_os = "macos", feature = "metal"))]
use super::attention_ops::AttentionLayout;
use super::mtp::{concat_row_pair, push_generated};
use super::*;

struct DraftVerifyResult {
    consumed: Vec<usize>,
    rejected: Option<usize>,
    stopped: bool,
}

struct DraftVerify<'a> {
    cache: &'a mut CausalDecoderCache,
    generated: &'a mut Vec<usize>,
    context: &'a mut Vec<usize>,
    proposals: Vec<usize>,
    primary: usize,
    max_new_tokens: usize,
    options: &'a GenerationOptions,
    sampler: &'a mut DeterministicSampler,
    stats: &'a mut SpeculativeStats,
}

struct MtpVerifyResult {
    accepted_all: bool,
    verify_token: usize,
}

struct MtpVerify<'a> {
    cache: &'a mut CausalDecoderCache,
    final_state: &'a mut Tensor,
    pending: &'a mut usize,
    generated: &'a mut Vec<usize>,
    drafts: Vec<usize>,
    primary: usize,
    max_new_tokens: usize,
    options: &'a GenerationOptions,
    sampler: &'a mut DeterministicSampler,
    stats: &'a mut SpeculativeStats,
}

struct SpeculativeCacheSnapshot {
    cpu: CausalDecoderCache,
    #[cfg(all(target_os = "macos", feature = "metal"))]
    linear_metal: Vec<Option<crate::metal_backend::LinearAttentionMetalState>>,
    #[cfg(all(target_os = "macos", feature = "metal"))]
    full_lens: Vec<Option<usize>>,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
struct DecodeProfiler {
    snapshot: Option<(u64, u64, u64, u64)>,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl DecodeProfiler {
    fn start() -> Self {
        // Profil decode (RETI_RUST_DECODE_PROFILE) : borne le cumul Metal autour
        // de la boucle de decode steady-state pour un split encode/wait/read par
        // token (cf. /tmp/rust_infer_plan.md, phase 1a).
        Self {
            snapshot: decode_profile_enabled().then(crate::metal_backend::decode_profile_snapshot),
        }
    }

    fn report_decode_loop(&self, decode: Duration, decode_tokens: usize) {
        let Some((cb0, wait0, read0, dispatch0)) = self.snapshot else {
            return;
        };
        let (cb1, wait1, read1, dispatch1) = crate::metal_backend::decode_profile_snapshot();
        let n = decode_tokens.max(1) as f64;
        let total_us = decode.as_micros() as f64 / n;
        let wait_us = wait1.saturating_sub(wait0) as f64 / 1000.0 / n;
        let read_us = read1.saturating_sub(read0) as f64 / 1000.0 / n;
        let encode_us = (total_us - wait_us - read_us).max(0.0);
        let cb = cb1.saturating_sub(cb0) as f64 / n;
        let dispatches = dispatch1.saturating_sub(dispatch0) as f64 / n;
        eprintln!(
            "decode profile total_us={total_us:.0} encode_us={encode_us:.0} wait_us={wait_us:.0} read_us={read_us:.0} cmd_buffers/tok={cb:.1} dispatches/tok={dispatches:.1}",
        );
    }

    fn report_gpu_sections(decoder: &CausalDecoder, cache: &CausalDecoderCache) {
        // Classement per-section GPU (tranche 3, `RETI_RUST_GPU_COUNTERS`).
        let Some(arena) = cache.resident.as_ref() else {
            return;
        };
        if let Some(report) = arena.state.gpu_timer().and_then(GpuSectionTimer::report) {
            eprintln!("{report}");
            if let Some(micro) = decoder.profile_moe_and_overhead(arena.state.queue()) {
                eprintln!("{micro}");
            }
        }
    }

    fn report_topk_microbench(decoder: &CausalDecoder) {
        if !topk_bench_enabled() {
            return;
        }
        let Some(metal) = decoder.forward_runtime().metal_executor() else {
            return;
        };
        match metal.profile_topk_softmax_kernel(256, 8, 4096) {
            Ok(report) => eprintln!("{report}"),
            Err(error) => eprintln!("topk microbench erreur: {error}"),
        }
    }
}

impl CausalDecoder {
    pub(super) fn next_final_state_cached(
        &self,
        cache: &mut CausalDecoderCache,
        token_id: usize,
    ) -> Result<Tensor> {
        if cache.layers.len() != self.layers.len() {
            return Err(InferError::Dimension(format!(
                "cache couches={} incompatible avec décodeur couches={}",
                cache.layers.len(),
                self.layers.len()
            )));
        }
        let position = cache.position;
        let runtime = self.forward_runtime();
        let mut hidden = embed_weight_tokens(&self.embed_tokens, &[token_id])?;
        for (layer_index, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward_cached(
                &self.config,
                &hidden,
                &mut cache.layers[layer_index],
                position,
                runtime,
            )?;
        }
        cache.position += 1;
        rms_norm(&hidden, &self.final_norm, self.config.rms_eps)
    }

    fn next_final_states_batched(
        &self,
        cache: &mut CausalDecoderCache,
        token_ids: &[usize],
    ) -> Result<Tensor> {
        if token_ids.is_empty() {
            return Err(InferError::Dimension("verify MTP batch vide".to_string()));
        }
        if cache.layers.len() != self.layers.len() {
            return Err(InferError::Dimension(format!(
                "cache couches={} incompatible avec décodeur couches={}",
                cache.layers.len(),
                self.layers.len()
            )));
        }
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if let Some(output) =
            self.next_final_states_resident_verify(cache, token_ids, false, false)?
        {
            return Ok(output.states);
        }
        let position_offset = cache.position;
        let runtime = self.forward_runtime();
        let mut hidden = embed_weight_tokens(&self.embed_tokens, token_ids)?;
        for (layer_index, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward_prefill(
                &self.config,
                &hidden,
                &mut cache.layers[layer_index],
                position_offset,
                runtime,
            )?;
        }
        cache.position += token_ids.len();
        rms_norm(&hidden, &self.final_norm, self.config.rms_eps)
    }

    fn next_final_states_batched_with_tokens(
        &self,
        cache: &mut CausalDecoderCache,
        token_ids: &[usize],
        capture_linear: bool,
    ) -> Result<(Tensor, Option<Vec<usize>>, Option<ResidentVerifyCaptures>)> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if let Some(output) =
            self.next_final_states_resident_verify(cache, token_ids, true, capture_linear)?
        {
            return Ok((output.states, output.tokens, output.captures));
        }
        self.next_final_states_batched(cache, token_ids)
            .map(|states| (states, None, None))
    }

    pub(super) fn logits_from_final_state(&self, final_state: &Tensor) -> Result<Tensor> {
        let logits = self
            .lm_head
            .forward_with_runtime(final_state, self.forward_runtime())?;
        Tensor::row(logits.last_row()?.to_vec())
    }

    pub(crate) fn sample_token_from_state(
        &self,
        final_state: &Tensor,
        options: &GenerationOptions,
        sampler: &mut DeterministicSampler,
    ) -> Result<usize> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if options.temperature <= f32::EPSILON && gpu_argmax_enabled() {
            if let Some(metal) = self.forward_runtime().metal_executor() {
                if self.lm_head.bias().is_none() {
                    return metal.argmax_linear_biasless(final_state, &self.lm_head);
                }
            }
        }
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if options.temperature > f32::EPSILON
            && options.top_k > 0
            && options.top_k <= crate::metal_backend::MAX_SAMPLER_TOP_K
            && gpu_sampler_enabled()
        {
            if let Some(metal) = self.forward_runtime().metal_executor() {
                if self.lm_head.bias().is_none() {
                    let state = sampler.state();
                    let token = metal.sample_linear_biasless_topk_topp(
                        final_state,
                        &self.lm_head,
                        options.temperature,
                        options.top_p,
                        options.top_k,
                        state,
                    )?;
                    sampler.advance();
                    return Ok(token);
                }
            }
        }
        let logits = self.logits_from_final_state(final_state)?;
        sample_token_top_k_top_p(
            logits.as_row()?,
            options.temperature,
            options.top_p,
            options.top_k,
            sampler,
        )
    }

    /// Génère des tokens greedy avec les options par défaut.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le prompt ou le forward échoue.
    pub fn generate_greedy(&self, prompt: &[usize], max_new_tokens: usize) -> Result<Vec<usize>> {
        self.generate_greedy_with_options(prompt, max_new_tokens, &GenerationOptions::default())
    }

    /// Génère des tokens greedy avec options explicites.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le prompt, les options ou le forward échouent.
    pub fn generate_greedy_with_options(
        &self,
        prompt: &[usize],
        max_new_tokens: usize,
        options: &GenerationOptions,
    ) -> Result<Vec<usize>> {
        Ok(self
            .generate_greedy_timed_with_options(prompt, max_new_tokens, options)?
            .tokens)
    }

    /// Génère en greedy avec cache K/V incrémental.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le prompt, les options ou le forward échouent.
    pub fn generate_greedy_cached_with_options(
        &self,
        prompt: &[usize],
        max_new_tokens: usize,
        options: &GenerationOptions,
    ) -> Result<Vec<usize>> {
        Ok(self
            .generate_greedy_timed_with_options(prompt, max_new_tokens, options)?
            .tokens)
    }

    /// Génère en greedy avec cache K/V incrémental et timings prefill/decode.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le prompt, les options ou le forward échouent.
    pub fn generate_greedy_timed_with_options(
        &self,
        prompt: &[usize],
        max_new_tokens: usize,
        options: &GenerationOptions,
    ) -> Result<GenerationOutput> {
        if prompt.is_empty() {
            return Err(InferError::Dimension("prompt token vide".to_string()));
        }
        if max_new_tokens == 0 {
            return Ok(GenerationOutput {
                tokens: Vec::new(),
                timings: GenerationTimings::default(),
            });
        }
        let prefill_started = Instant::now();
        let (mut cache, mut final_state) = self.prefill_cache_state(prompt)?;
        // Decode résident COMPLET (1c) si le flag est ON ET le modèle est supporté
        // (validation en amont, tout-ou-rien). Sinon, decode résident full-attn (1b)
        // si son flag est ON : alloue/seed le KV GPU des couches full-attn.
        // Decode résident COMPLET (1c) : greedy uniquement (argmax on-device) et
        // modèle supporté. L'arène doit être prête tout-ou-rien : si un état par
        // couche manque, on retombe sur le per-op (et le résident 1b si activé).
        #[cfg(all(target_os = "macos", feature = "metal"))]
        let resident_sampling = options.temperature > f32::EPSILON
            && options.top_k > 0
            && options.top_k <= crate::metal_backend::MAX_SAMPLER_TOP_K
            && gpu_sampler_enabled();
        #[cfg(all(target_os = "macos", feature = "metal"))]
        let mut use_resident_full = decode_resident_full_enabled()
            && (options.temperature <= f32::EPSILON || resident_sampling)
            && self.supports_resident_full_decode();
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if use_resident_full {
            use_resident_full = self.setup_resident_full_decode(&mut cache, max_new_tokens)?;
        }
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if decode_resident_enabled() && !use_resident_full {
            self.setup_resident_decode(&mut cache, max_new_tokens)?;
        }
        let prefill = prefill_started.elapsed();
        let mut generated = Vec::with_capacity(max_new_tokens);
        let mut sampler = DeterministicSampler::new(options.seed);
        let mut decode = Duration::ZERO;
        let mut decode_tokens = 0_usize;
        let mut token = self.sample_token_from_state(&final_state, options, &mut sampler)?;
        #[cfg(all(target_os = "macos", feature = "metal"))]
        let decode_profiler = DecodeProfiler::start();
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if use_resident_full && decode_pipeline_enabled() {
            if let Some(output) = self.decode_tokens_resident_pipelined(
                &mut cache,
                token,
                max_new_tokens,
                options,
                &mut sampler,
            )? {
                decode += output.decode;
                decode_tokens += output.decode_tokens;
                generated = output.tokens;
                decode_profiler.report_decode_loop(decode, decode_tokens);
                DecodeProfiler::report_gpu_sections(self, &cache);
                DecodeProfiler::report_topk_microbench(self);
                return Ok(GenerationOutput {
                    tokens: generated,
                    timings: GenerationTimings {
                        prefill,
                        decode,
                        decode_tokens,
                    },
                });
            }
        }
        for step in 0..max_new_tokens {
            if options.stop_token_ids.contains(&token) {
                break;
            }
            generated.push(token);
            if step + 1 < max_new_tokens {
                let decode_started = Instant::now();
                #[cfg(all(target_os = "macos", feature = "metal"))]
                if use_resident_full {
                    // 1c : forward complet + argmax/sampler en UN command buffer.
                    token = if options.temperature > f32::EPSILON {
                        self.decode_token_resident_sampled(
                            &mut cache,
                            token,
                            options,
                            &mut sampler,
                        )?
                    } else {
                        self.decode_token_resident(&mut cache, token)?
                    };
                    decode += decode_started.elapsed();
                    decode_tokens += 1;
                    continue;
                }
                final_state = self.next_decode_state(&mut cache, token)?;
                token = self.sample_token_from_state(&final_state, options, &mut sampler)?;
                decode += decode_started.elapsed();
                decode_tokens += 1;
            }
        }
        #[cfg(all(target_os = "macos", feature = "metal"))]
        decode_profiler.report_decode_loop(decode, decode_tokens);
        #[cfg(all(target_os = "macos", feature = "metal"))]
        DecodeProfiler::report_gpu_sections(self, &cache);
        #[cfg(all(target_os = "macos", feature = "metal"))]
        DecodeProfiler::report_topk_microbench(self);
        Ok(GenerationOutput {
            tokens: generated,
            timings: GenerationTimings {
                prefill,
                decode,
                decode_tokens,
            },
        })
    }

    /// Génère en greedy via une boucle spéculative propose-and-verify, avec un
    /// draft injecté. Phase A uniquement : verify séquentiel correct, pas de
    /// batch=N et pas de vraie tête MTP câblée.
    ///
    /// Le provider reçoit le contexte déjà émis (`prompt + tokens générés`) et
    /// renvoie au plus `max_draft_tokens` propositions. En greedy, un draft n'est
    /// émis que s'il égale l'argmax trunk; sinon on restaure le snapshot et on
    /// rejoue le préfixe accepté + la correction trunk.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le prompt est vide, si `temperature > 0`, si le
    /// rollback linear-attn Metal résident n'est pas sûr, ou si un forward de
    /// vérification échoue.
    pub fn generate_greedy_speculative_with_draft<F>(
        &self,
        prompt: &[usize],
        max_new_tokens: usize,
        options: &GenerationOptions,
        max_draft_tokens: usize,
        mut draft: F,
    ) -> Result<SpeculativeOutput>
    where
        F: FnMut(&[usize], usize) -> Vec<usize>,
    {
        if prompt.is_empty() {
            return Err(InferError::Dimension("prompt token vide".to_string()));
        }
        if options.temperature > f32::EPSILON {
            return Err(InferError::Config(
                "decode spéculatif Phase A supporte uniquement greedy temperature=0".to_string(),
            ));
        }
        self.ensure_speculative_cache_snapshot_supported()?;
        if max_new_tokens == 0 {
            return Ok(SpeculativeOutput {
                tokens: Vec::new(),
                stats: SpeculativeStats::default(),
            });
        }

        let (mut cache, final_state) = self.prefill_cache_state(prompt)?;
        let mut sampler = DeterministicSampler::new(options.seed);
        let mut token = self.sample_token_from_state(&final_state, options, &mut sampler)?;
        let mut generated = Vec::with_capacity(max_new_tokens);
        let mut context = prompt.to_vec();
        let mut stats = SpeculativeStats::default();

        while generated.len() < max_new_tokens {
            if options.stop_token_ids.contains(&token) {
                break;
            }
            let primary = token;
            push_generated(&mut generated, &mut context, primary);
            if generated.len() >= max_new_tokens || options.stop_token_ids.contains(&primary) {
                break;
            }

            let proposals = draft(&context, max_draft_tokens)
                .into_iter()
                .take(max_draft_tokens)
                .collect::<Vec<_>>();
            if proposals.is_empty() {
                token = self.advance_greedy_token(&mut cache, primary, options, &mut sampler)?;
                continue;
            }
            stats.proposed += proposals.len();

            let snapshot = self.snapshot_speculative_cache(&cache)?;
            let base_generated_len = generated.len();
            let base_context_len = context.len();
            let verified = self.verify_draft_proposals(DraftVerify {
                cache: &mut cache,
                generated: &mut generated,
                context: &mut context,
                proposals,
                primary,
                max_new_tokens,
                options,
                sampler: &mut sampler,
                stats: &mut stats,
            })?;
            if verified.stopped {
                return Ok(SpeculativeOutput {
                    tokens: generated,
                    stats,
                });
            }
            let consumed = verified.consumed;

            if let Some(correction) = verified.rejected {
                self.restore_speculative_cache(&mut cache, snapshot)?;
                generated.truncate(base_generated_len);
                context.truncate(base_context_len);
                stats.rollbacks += 1;
                let mut replay = vec![primary];
                for accepted in consumed.into_iter().skip(1) {
                    if generated.len() >= max_new_tokens {
                        break;
                    }
                    push_generated(&mut generated, &mut context, accepted);
                    replay.push(accepted);
                }
                if generated.len() >= max_new_tokens || options.stop_token_ids.contains(&correction)
                {
                    break;
                }
                push_generated(&mut generated, &mut context, correction);
                replay.push(correction);
                token =
                    self.replay_prefix_for_next_token(&mut cache, &replay, options, &mut sampler)?;
            } else {
                let replay_last = *consumed
                    .last()
                    .expect("invariant: consumed contient au moins primary");
                if generated.len() >= max_new_tokens
                    || options.stop_token_ids.contains(&replay_last)
                {
                    break;
                }
                token =
                    self.advance_greedy_token(&mut cache, replay_last, options, &mut sampler)?;
            }
        }

        Ok(SpeculativeOutput {
            tokens: generated,
            stats,
        })
    }

    fn snapshot_speculative_cache(
        &self,
        cache: &CausalDecoderCache,
    ) -> Result<SpeculativeCacheSnapshot> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(metal) = self.forward_runtime().metal_executor() else {
                return Ok(SpeculativeCacheSnapshot {
                    cpu: cache.clone(),
                    linear_metal: Vec::new(),
                    full_lens: Vec::new(),
                });
            };
            let linear_states = cache
                .layers
                .iter()
                .map(|layer| layer.linear.metal_state())
                .collect::<Vec<_>>();
            let linear_metal = metal.snapshot_linear_attn_states(&linear_states)?;
            let mut full_lens = Vec::with_capacity(cache.layers.len());
            for layer in &cache.layers {
                full_lens.push(layer.full.as_ref().map(FullAttentionMetalState::len));
            }
            Ok(SpeculativeCacheSnapshot {
                cpu: cache.clone(),
                linear_metal,
                full_lens,
            })
        }
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = self;
            Ok(SpeculativeCacheSnapshot { cpu: cache.clone() })
        }
    }

    fn restore_speculative_cache(
        &self,
        cache: &mut CausalDecoderCache,
        snapshot: SpeculativeCacheSnapshot,
    ) -> Result<()> {
        if cache.layers.len() != snapshot.cpu.layers.len() {
            return Err(InferError::Dimension(format!(
                "restore cache spéculatif: couches={} snapshot={}",
                cache.layers.len(),
                snapshot.cpu.layers.len()
            )));
        }
        cache.position = snapshot.cpu.position;
        for (layer, snap_layer) in cache.layers.iter_mut().zip(snapshot.cpu.layers.iter()) {
            layer.keys = snap_layer.keys.clone();
            layer.values = snap_layer.values.clone();
            layer.kv_dim = snap_layer.kv_dim;
            layer.linear.restore_cpu_state_from(&snap_layer.linear);
        }
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(metal) = self.forward_runtime().metal_executor() else {
                return Ok(());
            };
            let linear_metal = snapshot.linear_metal;
            let full_lens = snapshot.full_lens;
            if linear_metal.len() != cache.layers.len() || full_lens.len() != cache.layers.len() {
                return Err(InferError::Dimension(
                    "snapshot résident incomplet".to_string(),
                ));
            }
            let can_restore_batched =
                cache
                    .layers
                    .iter()
                    .zip(linear_metal.iter())
                    .all(|(layer, snapshot)| {
                        matches!(
                            (layer.linear.metal_state(), snapshot.as_ref()),
                            (Some(_), Some(_)) | (None, None)
                        )
                    });
            if can_restore_batched {
                let pairs = cache
                    .layers
                    .iter()
                    .zip(linear_metal.iter())
                    .filter_map(|(layer, snapshot)| {
                        Some((layer.linear.metal_state()?, snapshot.as_ref()?))
                    })
                    .collect::<Vec<_>>();
                metal.restore_linear_attn_states(&pairs)?;
            } else {
                for (layer, linear_snapshot) in
                    cache.layers.iter_mut().zip(linear_metal.into_iter())
                {
                    layer
                        .linear
                        .restore_metal_state_snapshot(metal, linear_snapshot)?;
                }
            }
            for (layer, full_len) in cache.layers.iter_mut().zip(full_lens.into_iter()) {
                if let (Some(full), Some(len)) = (layer.full.as_mut(), full_len) {
                    full.truncate(len)?;
                }
            }
        }
        Ok(())
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn restore_mtp_resident_progress(
        &self,
        cache: &mut CausalDecoderCache,
        captures: &ResidentVerifyCaptures,
        accepted_len: usize,
    ) -> Result<()> {
        if accepted_len == 0 {
            return Err(InferError::Dimension(
                "rollback MTP résident sans position acceptée".to_string(),
            ));
        }
        let row = accepted_len - 1;
        cache.position = captures
            .base_position
            .checked_add(accepted_len)
            .ok_or_else(|| InferError::Dimension("position rollback MTP déborde".to_string()))?;
        for layer in &mut cache.layers {
            if let Some(full) = layer.full.as_mut() {
                full.truncate(cache.position)?;
            }
        }
        let Some(metal) = self.forward_runtime().metal_executor() else {
            return Ok(());
        };
        let pairs = cache
            .layers
            .iter()
            .zip(captures.linear.iter())
            .filter_map(|(layer, captures)| {
                let current = layer.linear.metal_state()?;
                let capture = captures.as_ref()?.get(row)?;
                Some((current, capture))
            })
            .collect::<Vec<_>>();
        metal.restore_linear_attn_states(&pairs)
    }

    fn verify_draft_proposals(&self, verify: DraftVerify<'_>) -> Result<DraftVerifyResult> {
        let DraftVerify {
            cache,
            generated,
            context,
            proposals,
            primary,
            max_new_tokens,
            options,
            sampler,
            stats,
        } = verify;
        let mut consumed = vec![primary];
        let mut rejected = None;
        let mut stopped = false;

        for proposal in proposals {
            if generated.len() >= max_new_tokens {
                break;
            }
            let verify_input = *consumed
                .last()
                .expect("invariant: consumed contient au moins primary");
            let target_state = self.next_decode_state(cache, verify_input)?;
            stats.verifications += 1;
            let target = self.sample_token_from_state(&target_state, options, sampler)?;
            if options.stop_token_ids.contains(&target) {
                stopped = true;
                break;
            }
            if proposal == target {
                stats.accepted += 1;
                push_generated(generated, context, proposal);
                consumed.push(proposal);
                if options.stop_token_ids.contains(&proposal) {
                    break;
                }
            } else {
                stats.rejected += 1;
                rejected = Some(target);
                break;
            }
        }

        Ok(DraftVerifyResult {
            consumed,
            rejected,
            stopped,
        })
    }

    /// Mesure un decode MTP greedy avec verify trunk séquentiel.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si la tête MTP est absente, si le mode n'est pas greedy,
    /// ou si un forward échoue.
    pub fn generate_greedy_mtp_sequential_with_options(
        &self,
        prompt: &[usize],
        max_new_tokens: usize,
        options: &GenerationOptions,
        max_draft_tokens: usize,
    ) -> Result<SpeculativeOutput> {
        if self.mtp.is_none() {
            return Err(InferError::Config(
                "decode MTP demandé sans sidecar MTP chargé".to_string(),
            ));
        }
        if options.temperature > f32::EPSILON {
            return Err(InferError::Config(
                "mesure MTP B.5 supporte uniquement greedy temperature=0".to_string(),
            ));
        }
        if max_draft_tokens == 0 {
            return Err(InferError::Config(
                "mesure MTP B.5 sans draft token".to_string(),
            ));
        }
        self.ensure_speculative_cache_snapshot_supported()?;
        if prompt.is_empty() {
            return Err(InferError::Dimension("prompt token vide".to_string()));
        }
        if max_new_tokens == 0 {
            return Ok(SpeculativeOutput {
                tokens: Vec::new(),
                stats: SpeculativeStats::default(),
            });
        }

        let (mut cache, mut final_state) = self.prefill_cache_state(prompt)?;
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let mut use_resident_full = false;
            if decode_resident_full_enabled()
                && options.temperature <= f32::EPSILON
                && self.supports_resident_full_decode()
            {
                use_resident_full = self.setup_resident_full_decode(&mut cache, max_new_tokens)?;
            }
            if decode_resident_enabled() && !use_resident_full {
                self.setup_resident_decode(&mut cache, max_new_tokens)?;
            }
        }
        let mut sampler = DeterministicSampler::new(options.seed);
        let mut pending = self.sample_token_from_state(&final_state, options, &mut sampler)?;
        let mut generated = Vec::with_capacity(max_new_tokens);
        let mut stats = SpeculativeStats::default();
        stats.proposed_by_position.resize(max_draft_tokens, 0);
        stats.accepted_by_position.resize(max_draft_tokens, 0);
        #[cfg(all(target_os = "macos", feature = "metal"))]
        let decode_profiler = DecodeProfiler::start();
        let decode_started = Instant::now();

        while generated.len() < max_new_tokens {
            let primary = pending;
            if options.stop_token_ids.contains(&primary) {
                break;
            }
            generated.push(primary);
            if generated.len() >= max_new_tokens {
                break;
            }

            let mut mtp_cache = LayerKvCache::default();
            let mut draft_hidden = final_state.clone();
            let mut draft_token = primary;
            let mut drafts = Vec::with_capacity(max_draft_tokens);
            #[cfg(all(target_os = "macos", feature = "metal"))]
            let mtp_resident = self.start_mtp_resident_draft(&mut cache, &final_state)?;
            #[cfg(not(all(target_os = "macos", feature = "metal")))]
            let mtp_resident = false;
            #[cfg(all(target_os = "macos", feature = "metal"))]
            if mtp_resident {
                if let Some(resident_drafts) =
                    self.next_mtp_drafts_resident(&mut cache, draft_token, max_draft_tokens)?
                {
                    drafts = resident_drafts;
                    draft_token = drafts.last().copied().unwrap_or(draft_token);
                }
            }
            if drafts.is_empty() {
                for position in 0..max_draft_tokens {
                    let draft = if mtp_resident {
                        #[cfg(all(target_os = "macos", feature = "metal"))]
                        {
                            self.next_mtp_draft_resident(&mut cache, draft_token, position)?
                        }
                        #[cfg(not(all(target_os = "macos", feature = "metal")))]
                        unreachable!("invariant: mtp_resident=false hors Metal")
                    } else {
                        let (draft, next_hidden) = self.mtp_forward_greedy_with_hidden(
                            &draft_hidden,
                            draft_token,
                            position,
                            &mut mtp_cache,
                            options,
                            &mut sampler,
                        )?;
                        draft_hidden = next_hidden;
                        draft
                    };
                    drafts.push(draft);
                    draft_token = draft;
                }
            }

            let verified = self.verify_mtp_drafts(MtpVerify {
                cache: &mut cache,
                final_state: &mut final_state,
                pending: &mut pending,
                generated: &mut generated,
                drafts,
                primary,
                max_new_tokens,
                options,
                sampler: &mut sampler,
                stats: &mut stats,
            })?;

            if verified.accepted_all {
                final_state = self.next_decode_state(&mut cache, verified.verify_token)?;
                pending = self.sample_token_from_state(&final_state, options, &mut sampler)?;
                stats.verifications += 1;
            }
        }

        #[cfg(all(target_os = "macos", feature = "metal"))]
        decode_profiler.report_decode_loop(decode_started.elapsed(), generated.len());
        Ok(SpeculativeOutput {
            tokens: generated,
            stats,
        })
    }

    /// Mesure un decode MTP greedy avec verify trunk batché.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si la tête MTP est absente, si le mode n'est pas greedy,
    /// ou si un forward échoue.
    pub fn generate_greedy_mtp_batched_with_options(
        &self,
        prompt: &[usize],
        max_new_tokens: usize,
        options: &GenerationOptions,
        max_draft_tokens: usize,
    ) -> Result<SpeculativeOutput> {
        if self.mtp.is_none() {
            return Err(InferError::Config(
                "decode MTP demandé sans sidecar MTP chargé".to_string(),
            ));
        }
        if options.temperature > f32::EPSILON {
            return Err(InferError::Config(
                "mesure MTP B.5 supporte uniquement greedy temperature=0".to_string(),
            ));
        }
        if max_draft_tokens == 0 {
            return Err(InferError::Config(
                "mesure MTP B.5 sans draft token".to_string(),
            ));
        }
        self.ensure_speculative_cache_snapshot_supported()?;
        if prompt.is_empty() {
            return Err(InferError::Dimension("prompt token vide".to_string()));
        }
        if max_new_tokens == 0 {
            return Ok(SpeculativeOutput {
                tokens: Vec::new(),
                stats: SpeculativeStats::default(),
            });
        }

        let (mut cache, mut final_state) = self.prefill_cache_state(prompt)?;
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let mut use_resident_full = false;
            if decode_resident_full_enabled()
                && options.temperature <= f32::EPSILON
                && self.supports_resident_full_decode()
            {
                use_resident_full = self.setup_resident_full_decode(&mut cache, max_new_tokens)?;
            }
            if decode_resident_enabled() && !use_resident_full {
                self.setup_resident_decode(&mut cache, max_new_tokens)?;
            }
        }
        let mut sampler = DeterministicSampler::new(options.seed);
        let mut pending = self.sample_token_from_state(&final_state, options, &mut sampler)?;
        let mut generated = Vec::with_capacity(max_new_tokens);
        let mut stats = SpeculativeStats::default();
        stats.proposed_by_position.resize(max_draft_tokens, 0);
        stats.accepted_by_position.resize(max_draft_tokens, 0);
        #[cfg(all(target_os = "macos", feature = "metal"))]
        let decode_profiler = DecodeProfiler::start();
        let decode_started = Instant::now();

        while generated.len() < max_new_tokens {
            let primary = pending;
            if options.stop_token_ids.contains(&primary) {
                break;
            }
            generated.push(primary);
            if generated.len() >= max_new_tokens {
                break;
            }

            let mut mtp_cache = LayerKvCache::default();
            let mut draft_hidden = final_state.clone();
            let mut draft_token = primary;
            let mut drafts = Vec::with_capacity(max_draft_tokens);
            #[cfg(all(target_os = "macos", feature = "metal"))]
            let mtp_resident = self.start_mtp_resident_draft(&mut cache, &final_state)?;
            #[cfg(not(all(target_os = "macos", feature = "metal")))]
            let mtp_resident = false;
            #[cfg(all(target_os = "macos", feature = "metal"))]
            if mtp_resident {
                if let Some(resident_drafts) =
                    self.next_mtp_drafts_resident(&mut cache, draft_token, max_draft_tokens)?
                {
                    drafts = resident_drafts;
                    draft_token = drafts.last().copied().unwrap_or(draft_token);
                }
            }
            if drafts.is_empty() {
                for position in 0..max_draft_tokens {
                    let draft = if mtp_resident {
                        #[cfg(all(target_os = "macos", feature = "metal"))]
                        {
                            self.next_mtp_draft_resident(&mut cache, draft_token, position)?
                        }
                        #[cfg(not(all(target_os = "macos", feature = "metal")))]
                        unreachable!("invariant: mtp_resident=false hors Metal")
                    } else {
                        let (draft, next_hidden) = self.mtp_forward_greedy_with_hidden(
                            &draft_hidden,
                            draft_token,
                            position,
                            &mut mtp_cache,
                            options,
                            &mut sampler,
                        )?;
                        draft_hidden = next_hidden;
                        draft
                    };
                    drafts.push(draft);
                    draft_token = draft;
                }
            }
            let verified = self.verify_mtp_drafts_batched(MtpVerify {
                cache: &mut cache,
                final_state: &mut final_state,
                pending: &mut pending,
                generated: &mut generated,
                drafts,
                primary,
                max_new_tokens,
                options,
                sampler: &mut sampler,
                stats: &mut stats,
            })?;

            if verified.accepted_all {
                final_state = self.next_decode_state(&mut cache, verified.verify_token)?;
                pending = self.sample_token_from_state(&final_state, options, &mut sampler)?;
                stats.verifications += 1;
            }
        }

        #[cfg(all(target_os = "macos", feature = "metal"))]
        decode_profiler.report_decode_loop(decode_started.elapsed(), generated.len());
        Ok(SpeculativeOutput {
            tokens: generated,
            stats,
        })
    }

    fn verify_mtp_drafts(&self, verify: MtpVerify<'_>) -> Result<MtpVerifyResult> {
        let MtpVerify {
            cache,
            final_state,
            pending,
            generated,
            drafts,
            primary,
            max_new_tokens,
            options,
            sampler,
            stats,
        } = verify;
        let mut verify_token = primary;
        let mut accepted_all = true;

        for (position, draft) in drafts.into_iter().enumerate() {
            let target_state = self.next_decode_state(cache, verify_token)?;
            let target = self.sample_token_from_state(&target_state, options, sampler)?;
            stats.verifications += 1;
            stats.proposed += 1;
            stats.proposed_by_position[position] += 1;
            if draft == target {
                stats.accepted += 1;
                stats.accepted_by_position[position] += 1;
                generated.push(draft);
                *final_state = target_state;
                verify_token = draft;
                if generated.len() >= max_new_tokens || options.stop_token_ids.contains(&draft) {
                    accepted_all = false;
                    break;
                }
            } else {
                stats.rejected += 1;
                *pending = target;
                *final_state = target_state;
                accepted_all = false;
                break;
            }
        }

        Ok(MtpVerifyResult {
            accepted_all,
            verify_token,
        })
    }

    fn verify_mtp_drafts_batched(&self, verify: MtpVerify<'_>) -> Result<MtpVerifyResult> {
        let MtpVerify {
            cache,
            final_state,
            pending,
            generated,
            drafts,
            primary,
            max_new_tokens,
            options,
            sampler,
            stats,
        } = verify;
        if drafts.is_empty() {
            return Ok(MtpVerifyResult {
                accepted_all: true,
                verify_token: primary,
            });
        }

        #[cfg(all(target_os = "macos", feature = "metal"))]
        let capture_rollback = options.temperature <= f32::EPSILON
            && gpu_argmax_enabled()
            && cache.resident.is_some()
            && self.supports_resident_full_decode();
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        let capture_rollback = false;
        let mut snapshot = if capture_rollback {
            None
        } else {
            Some(self.snapshot_speculative_cache(cache)?)
        };
        let mut verify_tokens = Vec::with_capacity(drafts.len());
        verify_tokens.push(primary);
        verify_tokens.extend(drafts.iter().take(drafts.len().saturating_sub(1)).copied());
        let (target_states, target_tokens, resident_captures) =
            if options.temperature <= f32::EPSILON && gpu_argmax_enabled() {
                self.next_final_states_batched_with_tokens(cache, &verify_tokens, capture_rollback)?
            } else {
                (
                    self.next_final_states_batched(cache, &verify_tokens)?,
                    None,
                    None,
                )
            };
        let target_tokens = if target_tokens.is_some() {
            target_tokens
        } else if options.temperature <= f32::EPSILON && gpu_argmax_enabled() {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            {
                if let (Some(metal), Some(arena)) = (
                    self.forward_runtime().metal_executor(),
                    cache.resident.as_ref(),
                ) {
                    metal.argmax_linear_biasless_rows_buffers(&target_states, &arena.lm_head)?
                } else {
                    None
                }
            }
            #[cfg(not(all(target_os = "macos", feature = "metal")))]
            {
                None
            }
        } else {
            None
        };
        let mut verify_token = primary;
        let mut accepted_all = true;

        for (position, draft) in drafts.into_iter().enumerate() {
            let target_state = Tensor::row(target_states.row_slice(position)?.to_vec())?;
            let target = match target_tokens
                .as_ref()
                .and_then(|tokens| tokens.get(position))
            {
                Some(token) => *token,
                None => self.sample_token_from_state(&target_state, options, sampler)?,
            };
            stats.verifications += 1;
            stats.proposed += 1;
            stats.proposed_by_position[position] += 1;
            if draft == target {
                stats.accepted += 1;
                stats.accepted_by_position[position] += 1;
                generated.push(draft);
                *final_state = target_state;
                verify_token = draft;
                if generated.len() >= max_new_tokens || options.stop_token_ids.contains(&draft) {
                    accepted_all = false;
                    if position + 1 < verify_tokens.len() {
                        let replay_len = position + 1;
                        #[cfg(all(target_os = "macos", feature = "metal"))]
                        if let Some(captures) = resident_captures.as_ref() {
                            self.restore_mtp_resident_progress(cache, captures, replay_len)?;
                        } else {
                            let snapshot = snapshot.take().ok_or_else(|| {
                                InferError::Config("snapshot MTP fallback absent".to_string())
                            })?;
                            self.restore_speculative_cache(cache, snapshot)?;
                            for token in verify_tokens.iter().take(replay_len) {
                                *final_state = self.next_decode_state(cache, *token)?;
                            }
                        }
                        #[cfg(not(all(target_os = "macos", feature = "metal")))]
                        {
                            let snapshot = snapshot.take().ok_or_else(|| {
                                InferError::Config("snapshot MTP fallback absent".to_string())
                            })?;
                            self.restore_speculative_cache(cache, snapshot)?;
                            for token in verify_tokens.iter().take(replay_len) {
                                *final_state = self.next_decode_state(cache, *token)?;
                            }
                        }
                    }
                    break;
                }
            } else {
                stats.rejected += 1;
                *pending = target;
                *final_state = target_state;
                accepted_all = false;
                if position + 1 < verify_tokens.len() {
                    let replay_len = position + 1;
                    #[cfg(all(target_os = "macos", feature = "metal"))]
                    if let Some(captures) = resident_captures.as_ref() {
                        self.restore_mtp_resident_progress(cache, captures, replay_len)?;
                    } else {
                        let snapshot = snapshot.take().ok_or_else(|| {
                            InferError::Config("snapshot MTP fallback absent".to_string())
                        })?;
                        self.restore_speculative_cache(cache, snapshot)?;
                        for token in verify_tokens.iter().take(replay_len) {
                            *final_state = self.next_decode_state(cache, *token)?;
                        }
                    }
                    #[cfg(not(all(target_os = "macos", feature = "metal")))]
                    {
                        let snapshot = snapshot.take().ok_or_else(|| {
                            InferError::Config("snapshot MTP fallback absent".to_string())
                        })?;
                        self.restore_speculative_cache(cache, snapshot)?;
                        for token in verify_tokens.iter().take(replay_len) {
                            *final_state = self.next_decode_state(cache, *token)?;
                        }
                    }
                }
                break;
            }
        }

        Ok(MtpVerifyResult {
            accepted_all,
            verify_token,
        })
    }

    fn mtp_forward_greedy_with_hidden(
        &self,
        hidden: &Tensor,
        next_id: usize,
        position: usize,
        mtp_cache: &mut LayerKvCache,
        options: &GenerationOptions,
        sampler: &mut DeterministicSampler,
    ) -> Result<(usize, Tensor)> {
        let post = self.mtp_forward_post_hidden(hidden, next_id, position, mtp_cache)?;
        let token = self.sample_token_from_state(&post, options, sampler)?;
        Ok((token, post))
    }

    fn mtp_forward_post_hidden(
        &self,
        hidden: &Tensor,
        next_id: usize,
        position: usize,
        mtp_cache: &mut LayerKvCache,
    ) -> Result<Tensor> {
        let head = self
            .mtp
            .as_ref()
            .ok_or_else(|| InferError::Config("forward MTP sans tête chargée".to_string()))?;
        let embedding = embed_weight_tokens(&self.embed_tokens, &[next_id])?;
        let embedding = rms_norm(&embedding, &head.pre_fc_norm_embedding, self.config.rms_eps)?;
        let hidden = rms_norm(hidden, &head.pre_fc_norm_hidden, self.config.rms_eps)?;
        let x = concat_row_pair(&embedding, &hidden)?;
        let x = head.fc.forward_with_runtime(&x, self.forward_runtime())?;

        let residual = x.clone();
        let normed = rms_norm(&x, &head.layer.input_norm, self.config.rms_eps)?;
        let context = full_attention_context_cached(
            &self.config,
            &normed,
            mtp_cache,
            position,
            &head.layer.attention,
            self.forward_runtime(),
        )?;
        let attn = head
            .layer
            .attention
            .o_proj
            .forward_with_runtime(&context, self.forward_runtime())?;
        let h = residual.add(&attn)?;

        let residual = h.clone();
        let mlp_input = rms_norm(&h, &head.layer.post_attention_norm, self.config.rms_eps)?;
        let mlp = head
            .layer
            .mlp
            .forward_with_runtime(&mlp_input, self.forward_runtime())?;
        let h = residual.add(&mlp)?;
        rms_norm(&h, &head.norm, self.config.rms_eps)
    }

    fn ensure_speculative_cache_snapshot_supported(&self) -> Result<()> {
        let _ = self;
        Ok(())
    }

    fn advance_greedy_token(
        &self,
        cache: &mut CausalDecoderCache,
        token_id: usize,
        options: &GenerationOptions,
        sampler: &mut DeterministicSampler,
    ) -> Result<usize> {
        let state = self.next_decode_state(cache, token_id)?;
        self.sample_token_from_state(&state, options, sampler)
    }

    fn replay_prefix_for_next_token(
        &self,
        cache: &mut CausalDecoderCache,
        tokens: &[usize],
        options: &GenerationOptions,
        sampler: &mut DeterministicSampler,
    ) -> Result<usize> {
        let mut state = None;
        for token in tokens {
            state = Some(self.next_decode_state(cache, *token)?);
        }
        let state = state
            .ok_or_else(|| InferError::Dimension("replay spéculatif sans token".to_string()))?;
        self.sample_token_from_state(&state, options, sampler)
    }

    /// Pré-remplit le cache K/V et renvoie les logits du dernier token du prompt.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le prompt est vide ou si un passage forward échoue.
    pub fn prefill_cache(&self, prompt: &[usize]) -> Result<(CausalDecoderCache, Tensor)> {
        let (cache, final_state) = self.prefill_cache_state(prompt)?;
        let logits = self.logits_from_final_state(&final_state)?;
        Ok((cache, logits))
    }

    fn prefill_cache_state(&self, prompt: &[usize]) -> Result<(CausalDecoderCache, Tensor)> {
        if prompt.is_empty() {
            return Err(InferError::Dimension("prompt token vide".to_string()));
        }
        if prefix_cache_enabled() {
            if let Some(hit) = self.prefix_cache_get(prompt)? {
                return Ok(hit);
            }
        }
        let state = if self.can_prefill_batched() && prefill_batched_enabled() {
            self.prefill_cache_state_batched(prompt)
        } else {
            self.prefill_cache_state_tokenwise(prompt)
        }?;
        if prefix_cache_enabled() {
            self.prefix_cache_put(prompt, &state)?;
        }
        Ok(state)
    }

    fn prefix_cache_get(&self, prompt: &[usize]) -> Result<Option<(CausalDecoderCache, Tensor)>> {
        let mut cache = self
            .prefix_cache
            .lock()
            .map_err(|_| InferError::Config("cache préfixe empoisonné".to_string()))?;
        let Some(index) = cache
            .entries
            .iter()
            .position(|entry| entry.tokens == prompt)
        else {
            return Ok(None);
        };
        let entry = cache.entries.remove(index);
        let output = (entry.cache.clone(), entry.final_state.clone());
        cache.entries.insert(0, entry);
        Ok(Some(output))
    }

    fn prefix_cache_put(
        &self,
        prompt: &[usize],
        state: &(CausalDecoderCache, Tensor),
    ) -> Result<()> {
        let capacity = prefix_cache_capacity();
        if capacity == 0 {
            return Ok(());
        }
        let mut cache = self
            .prefix_cache
            .lock()
            .map_err(|_| InferError::Config("cache préfixe empoisonné".to_string()))?;
        if let Some(index) = cache
            .entries
            .iter()
            .position(|entry| entry.tokens == prompt)
        {
            cache.entries.remove(index);
        }
        cache.entries.insert(
            0,
            PrefixCacheEntry {
                tokens: prompt.to_vec(),
                cache: state.0.clone(),
                final_state: state.1.clone(),
            },
        );
        cache.entries.truncate(capacity);
        Ok(())
    }

    pub(crate) fn prefill_cache_state_tokenwise(
        &self,
        prompt: &[usize],
    ) -> Result<(CausalDecoderCache, Tensor)> {
        let mut cache = self.empty_cache();
        let mut last_state = None;
        for token_id in prompt {
            last_state = Some(self.next_final_state_cached(&mut cache, *token_id)?);
        }
        let state =
            last_state.ok_or_else(|| InferError::Dimension("prompt token vide".to_string()))?;
        Ok((cache, state))
    }

    fn prefill_cache_state_batched(
        &self,
        prompt: &[usize],
    ) -> Result<(CausalDecoderCache, Tensor)> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if let Some(state) = self.prefill_cache_state_metal_resident(prompt)? {
            return Ok(state);
        }
        let runtime = self.forward_runtime();
        let mut cache = self.empty_cache();
        let mut hidden = embed_weight_tokens(&self.embed_tokens, prompt)?;
        for (layer_index, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward_prefill(
                &self.config,
                &hidden,
                &mut cache.layers[layer_index],
                0,
                runtime,
            )?;
        }
        cache.position = prompt.len();
        let final_hidden = Tensor::row(hidden.last_row()?.to_vec())?;
        let final_state = rms_norm(&final_hidden, &self.final_norm, self.config.rms_eps)?;
        Ok((cache, final_state))
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn prefill_cache_state_metal_resident(
        &self,
        prompt: &[usize],
    ) -> Result<Option<(CausalDecoderCache, Tensor)>> {
        if !prefill_resident_enabled() || self.config.attn_output_gate {
            return Ok(None);
        }
        let Some(metal) = self.forward_runtime().metal_executor() else {
            return Ok(None);
        };
        let Some(rope_theta) = self.config.rope_theta else {
            return Ok(None);
        };
        let Some(head_dim) = self.config.head_dim else {
            return Ok(None);
        };
        let hidden = embed_weight_tokens(&self.embed_tokens, prompt)?;
        let (seq, hidden_dim) = hidden.as_matrix()?;
        let spec = crate::metal_backend::PrefillAttentionSpec {
            seq,
            hidden_dim,
            q_heads: self.config.num_attention_heads,
            kv_heads: self.config.num_key_value_heads,
            head_dim,
            rope_dims: self.config.rope_dims.unwrap_or(head_dim),
            rope_theta,
            eps: self.config.rms_eps,
        };
        let mut layers = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let Some(prefill_layer) = layer.prefill_moe_layer() else {
                return Ok(None);
            };
            layers.push(prefill_layer);
        }
        match metal.qwen_moe_prefill_resident(&hidden, &layers, spec) {
            Ok((final_hidden, kv)) => {
                if kv.len() != self.layers.len() {
                    return Err(InferError::Dimension(format!(
                        "prefill résident kv couches={} attendu={}",
                        kv.len(),
                        self.layers.len()
                    )));
                }
                let layout = AttentionLayout {
                    num_attention_heads: self.config.num_attention_heads,
                    num_key_value_heads: self.config.num_key_value_heads,
                    head_dim,
                    rope_dims: self.config.rope_dims.unwrap_or(head_dim),
                };
                let mut cache = self.empty_cache();
                for (layer_cache, (key, value)) in cache.layers.iter_mut().zip(kv.iter()) {
                    layer_cache.append_batch(key, value, &layout)?;
                }
                cache.position = prompt.len();
                let final_hidden = Tensor::row(final_hidden.last_row()?.to_vec())?;
                let final_state = rms_norm(&final_hidden, &self.final_norm, self.config.rms_eps)?;
                Ok(Some((cache, final_state)))
            }
            Err(error) => {
                if trace_prefill_enabled() {
                    eprintln!("prefill résident fallback: {error}");
                }
                Ok(None)
            }
        }
    }

    fn can_prefill_batched(&self) -> bool {
        self.layers
            .iter()
            .all(DecoderLayer::supports_batched_prefill)
    }

    #[cfg(test)]
    pub(crate) fn generate_greedy_full_with_options(
        &self,
        prompt: &[usize],
        max_new_tokens: usize,
        options: &GenerationOptions,
    ) -> Result<Vec<usize>> {
        if prompt.is_empty() {
            return Err(InferError::Dimension("prompt token vide".to_string()));
        }
        let mut tokens = prompt.to_vec();
        let mut generated = Vec::with_capacity(max_new_tokens);
        let mut sampler = DeterministicSampler::new(options.seed);
        for _ in 0..max_new_tokens {
            let logits = self.next_logits(&tokens)?;
            let token = sample_token_top_k_top_p(
                logits.as_row()?,
                options.temperature,
                options.top_p,
                options.top_k,
                &mut sampler,
            )?;
            if options.stop_token_ids.contains(&token) {
                break;
            }
            tokens.push(token);
            generated.push(token);
        }
        Ok(generated)
    }
}
