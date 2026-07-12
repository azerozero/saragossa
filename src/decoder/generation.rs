//! Génération, préfill et tête MTP du décodeur causal.

use crate::sampling::{
    lossless_speculative_sample, sample_from_token_distribution, token_distribution_top_k_top_p,
    TokenDistribution,
};
#[cfg(feature = "devtools")]
use crate::DFlashDraft;

use super::attention_ops::full_attention_context_cached;
#[cfg(all(target_os = "macos", feature = "metal"))]
use super::attention_ops::AttentionLayout;
use super::mtp::concat_row_pair;
use super::*;

#[cfg(all(target_os = "macos", feature = "metal"))]
#[derive(Clone, Copy, Debug)]
struct MtpProfilePoint {
    started: Instant,
    command_buffers: u64,
    wait_ns: u64,
    read_ns: u64,
    dispatches: u64,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[derive(Clone, Copy, Debug, Default)]
struct MtpProfileBucket {
    calls: u64,
    command_buffers: u64,
    wait_ns: u64,
    read_ns: u64,
    dispatches: u64,
    wall_ns: u128,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[derive(Clone, Debug, Default)]
struct MtpStepProfile {
    fused_one: MtpProfileBucket,
    fused_two: MtpProfileBucket,
    draft_setup: MtpProfileBucket,
    draft: MtpProfileBucket,
    verify: MtpProfileBucket,
    history: MtpProfileBucket,
    bonus_next: MtpProfileBucket,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl MtpStepProfile {
    fn report(&self, generated: usize, verifications: usize) {
        let denom = generated.max(1) as f64;
        eprintln!(
            "mtp_step_profile generated={generated} verifications={verifications} \
             note=per_generated_token"
        );
        self.report_bucket("fused_one", self.fused_one, denom);
        self.report_bucket("fused_two", self.fused_two, denom);
        self.report_bucket("draft_setup", self.draft_setup, denom);
        self.report_bucket("draft", self.draft, denom);
        self.report_bucket("verify", self.verify, denom);
        self.report_bucket("history", self.history, denom);
        self.report_bucket("bonus_next", self.bonus_next, denom);
    }

    fn report_bucket(&self, label: &str, bucket: MtpProfileBucket, denom: f64) {
        if bucket.calls == 0 {
            return;
        }
        eprintln!(
            "mtp_step_profile label={label} calls={} wall_ms={:.3} wait_ms={:.3} \
             read_ms={:.3} cmd_buffers={} dispatches={} cb_per_tok={:.3} \
             dispatch_per_tok={:.3}",
            bucket.calls,
            bucket.wall_ns as f64 / 1_000_000.0,
            bucket.wait_ns as f64 / 1_000_000.0,
            bucket.read_ns as f64 / 1_000_000.0,
            bucket.command_buffers,
            bucket.dispatches,
            bucket.command_buffers as f64 / denom,
            bucket.dispatches as f64 / denom,
        );
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn mtp_step_profile_enabled() -> bool {
    #[cfg(feature = "devtools")]
    {
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED
            .get_or_init(|| crate::decoder::flags::env_flag("RETI_RUST_MTP_STEP_PROFILE", false))
    }
    #[cfg(not(feature = "devtools"))]
    {
        false
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn mtp_profile_start(enabled: bool) -> Option<MtpProfilePoint> {
    enabled.then(|| {
        let (command_buffers, wait_ns, read_ns, dispatches) =
            crate::metal_backend::decode_profile_snapshot();
        MtpProfilePoint {
            started: Instant::now(),
            command_buffers,
            wait_ns,
            read_ns,
            dispatches,
        }
    })
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn mtp_profile_record(bucket: &mut MtpProfileBucket, start: Option<MtpProfilePoint>) {
    let Some(start) = start else {
        return;
    };
    let (command_buffers, wait_ns, read_ns, dispatches) =
        crate::metal_backend::decode_profile_snapshot();
    bucket.calls += 1;
    bucket.command_buffers += command_buffers.saturating_sub(start.command_buffers);
    bucket.wait_ns += wait_ns.saturating_sub(start.wait_ns);
    bucket.read_ns += read_ns.saturating_sub(start.read_ns);
    bucket.dispatches += dispatches.saturating_sub(start.dispatches);
    bucket.wall_ns += start.started.elapsed().as_nanos();
}

struct MtpVerifyResult {
    accepted_all: bool,
    verify_token: usize,
    accepted_history: Vec<(Tensor, usize)>,
}

struct MtpVerify<'a> {
    cache: &'a mut CausalDecoderCache,
    final_state: &'a mut Tensor,
    pending: &'a mut usize,
    generated: &'a mut Vec<usize>,
    drafts: Vec<usize>,
    draft_distributions: Vec<Option<TokenDistribution>>,
    primary: usize,
    max_new_tokens: usize,
    options: &'a GenerationOptions,
    sampler: &'a mut DeterministicSampler,
    stats: &'a mut SpeculativeStats,
}

#[cfg(feature = "devtools")]
struct DFlashVerify<'a> {
    draft: &'a DFlashDraft,
    cache: &'a mut CausalDecoderCache,
    projected_context: &'a mut Tensor,
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

#[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
struct ResidentLinearXrayProbe {
    layer_index: usize,
    normed_max_abs: f32,
    normed_mean_abs: f32,
    attn_max_abs: f32,
    attn_mean_abs: f32,
    attn_cpu_normed_max_abs: f32,
    attn_cpu_normed_mean_abs: f32,
    init_state_conv_max_abs: f32,
    init_state_conv_mean_abs: f32,
    init_state_ssm_max_abs: f32,
    init_state_ssm_mean_abs: f32,
    state_conv_max_abs: f32,
    state_conv_mean_abs: f32,
    state_ssm_max_abs: f32,
    state_ssm_mean_abs: f32,
    state_cpu_normed_conv_max_abs: f32,
    state_cpu_normed_conv_mean_abs: f32,
    state_cpu_normed_ssm_max_abs: f32,
    state_cpu_normed_ssm_mean_abs: f32,
}

pub(super) fn longest_prefix_entry_index(
    entries: &[PrefixCacheEntry],
    prompt: &[usize],
) -> Option<usize> {
    let mut best = None;
    let mut best_len = 0usize;
    for (index, entry) in entries.iter().enumerate() {
        let len = entry.tokens.len();
        if len > best_len && len <= prompt.len() && prompt.starts_with(&entry.tokens) {
            best = Some(index);
            best_len = len;
        }
    }
    best
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
    site_snapshot:
        Option<std::collections::HashMap<crate::metal_backend::DispatchProfileSite, u64>>,
    shape_snapshot:
        Option<std::collections::HashMap<crate::metal_backend::DispatchProfileShape, u64>>,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl DecodeProfiler {
    fn start() -> Self {
        // Profil decode (RETI_RUST_DECODE_PROFILE) : borne le cumul Metal autour
        // de la boucle de decode steady-state pour un split encode/wait/read par
        // token (phase 1a du decode résident).
        Self {
            snapshot: decode_profile_enabled().then(crate::metal_backend::decode_profile_snapshot),
            site_snapshot: decode_profile_enabled()
                .then(crate::metal_backend::decode_profile_dispatch_sites_snapshot),
            shape_snapshot: decode_profile_enabled()
                .then(crate::metal_backend::decode_profile_dispatch_shapes_snapshot),
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
        if let Some(site0) = &self.site_snapshot {
            let site1 = crate::metal_backend::decode_profile_dispatch_sites_snapshot();
            let mut deltas = site1
                .into_iter()
                .filter_map(|(site, count1)| {
                    let count0 = site0.get(&site).copied().unwrap_or(0);
                    let delta = count1.saturating_sub(count0);
                    (delta > 0).then_some((site, delta))
                })
                .collect::<Vec<_>>();
            deltas.sort_by(|(_, left), (_, right)| right.cmp(left));
            for (site, count) in deltas.into_iter().take(16) {
                let per_token = count as f64 / n;
                eprintln!(
                    "decode dispatch_site per_tok={per_token:.1} count={count} at {}:{}:{}",
                    compact_source_path(site.file),
                    site.line,
                    site.column
                );
            }
        }
        if let Some(shape0) = &self.shape_snapshot {
            let shape1 = crate::metal_backend::decode_profile_dispatch_shapes_snapshot();
            let mut deltas = shape1
                .into_iter()
                .filter_map(|(shape, count1)| {
                    let count0 = shape0.get(&shape).copied().unwrap_or(0);
                    let delta = count1.saturating_sub(count0);
                    (delta > 0).then_some((shape, delta))
                })
                .collect::<Vec<_>>();
            deltas.sort_by(|(_, left), (_, right)| right.cmp(left));
            for (shape, count) in deltas.into_iter().take(20) {
                let per_token = count as f64 / n;
                eprintln!(
                    "decode dispatch_shape per_tok={per_token:.1} count={count} kind={} batch={} lhs_rows={} topk={} in_dim={} out_dim={} group_size={} bits={}",
                    shape.kind,
                    shape.batch,
                    shape.lhs_rows,
                    shape.topk,
                    shape.in_dim,
                    shape.out_dim,
                    shape.group_size,
                    shape.bits
                );
            }
        }
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

#[cfg(all(target_os = "macos", feature = "metal"))]
fn compact_source_path(path: &'static str) -> &'static str {
    path.strip_prefix(concat!(env!("CARGO_MANIFEST_DIR"), "/"))
        .unwrap_or(path)
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
        let mut hidden = self.embed_scaled(&[token_id])?;
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
            self.next_final_states_resident_verify(cache, token_ids, false, false, None)?
        {
            return Ok(output.states);
        }
        let position_offset = cache.position;
        let runtime = self.forward_runtime();
        let mut hidden = self.embed_scaled(token_ids)?;
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
            self.next_final_states_resident_verify(cache, token_ids, true, capture_linear, None)?
        {
            return Ok((output.states, output.tokens, output.captures));
        }
        self.next_final_states_batched(cache, token_ids)
            .map(|states| (states, None, None))
    }

    pub fn logits_from_final_state(&self, final_state: &Tensor) -> Result<Tensor> {
        self.logits_from_linear_state(final_state, &self.lm_head)
    }

    fn logits_from_mtp_draft_state(&self, final_state: &Tensor) -> Result<Tensor> {
        self.logits_from_linear_state(final_state, self.mtp_draft_lm_head())
    }

    fn logits_from_linear_state(&self, final_state: &Tensor, head: &Linear) -> Result<Tensor> {
        let logits = head.forward_with_runtime(final_state, self.forward_runtime())?;
        self.finalize_logits(&logits)
    }

    pub(super) fn finalize_logits(&self, logits: &Tensor) -> Result<Tensor> {
        let mut row = logits.last_row()?.to_vec();
        if let Some(softcap) = self.config.final_logit_softcapping {
            if softcap > 0.0 {
                for value in &mut row {
                    *value = (*value / softcap).tanh() * softcap;
                }
            }
        }
        Tensor::row(row)
    }

    /// Argmax greedy talker (cb0) sur GPU : `codec_head·final_state` puis argmax
    /// quantifié + suppression de la plage `[suppress_start, vocab)` sauf `eos`,
    /// directement on-device (1 `u32` relu, pas de readback full-vocab ni d'argmax
    /// CPU). Byte-identique à `logits_from_final_state` + `greedy_talker_token`.
    ///
    /// Renvoie `Ok(None)` si l'executor Metal est absent (l'appelant retombe sur le
    /// chemin CPU).
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si l'argmax Metal échoue.
    pub fn talker_greedy_token(
        &self,
        final_state: &Tensor,
        suppress_start: usize,
        eos: usize,
    ) -> Result<Option<usize>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if let Some(metal) = self.forward_runtime().metal_executor() {
            return metal
                .talker_greedy_token_biasless(final_state, &self.lm_head, suppress_start, eos)
                .map(Some);
        }
        let _ = (final_state, suppress_start, eos);
        Ok(None)
    }

    pub(crate) fn sample_token_from_state(
        &self,
        final_state: &Tensor,
        options: &GenerationOptions,
        sampler: &mut DeterministicSampler,
    ) -> Result<usize> {
        self.sample_token_from_linear_state(final_state, &self.lm_head, options, sampler)
    }

    fn sample_mtp_token_from_state(
        &self,
        final_state: &Tensor,
        options: &GenerationOptions,
        sampler: &mut DeterministicSampler,
    ) -> Result<usize> {
        self.sample_token_from_linear_state(final_state, self.mtp_draft_lm_head(), options, sampler)
    }

    fn sample_token_from_linear_state(
        &self,
        final_state: &Tensor,
        head: &Linear,
        options: &GenerationOptions,
        sampler: &mut DeterministicSampler,
    ) -> Result<usize> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if options.token_constraint.is_none()
            && options.temperature <= f32::EPSILON
            && gpu_argmax_enabled()
        {
            if let Some(metal) = self.forward_runtime().metal_executor() {
                if head.bias().is_none() {
                    return metal.argmax_linear_biasless(final_state, head);
                }
            }
        }
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if options.token_constraint.is_none()
            && options.temperature > f32::EPSILON
            && options.top_k == 0
            && options.top_p >= 1.0
            && gpu_sampler_enabled()
        {
            if let Some(metal) = self.forward_runtime().metal_executor() {
                if head.bias().is_none() {
                    let state = sampler.state();
                    let token = metal.sample_linear_biasless_gumbel(
                        final_state,
                        head,
                        options.temperature,
                        state,
                    )?;
                    sampler.advance();
                    return Ok(token);
                }
            }
        }
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if options.token_constraint.is_none()
            && options.temperature > f32::EPSILON
            && options.top_k > 0
            && options.top_k <= crate::metal_backend::MAX_SAMPLER_TOP_K
            && gpu_sampler_enabled()
        {
            if let Some(metal) = self.forward_runtime().metal_executor() {
                if head.bias().is_none() {
                    let state = sampler.state();
                    let token = metal.sample_linear_biasless_topk_topp(
                        final_state,
                        head,
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
        let logits = self.logits_from_linear_state(final_state, head)?;
        if let Some(constraint) = &options.token_constraint {
            let mut masked = logits.as_row()?.to_vec();
            constraint.mask_logits(&mut masked)?;
            let token = sample_token_top_k_top_p(
                &masked,
                options.temperature,
                options.top_p,
                options.top_k,
                sampler,
            )?;
            constraint.accept_token(token)?;
            return Ok(token);
        }
        sample_token_top_k_top_p(
            logits.as_row()?,
            options.temperature,
            options.top_p,
            options.top_k,
            sampler,
        )
    }

    fn token_distribution_from_state(
        &self,
        final_state: &Tensor,
        options: &GenerationOptions,
    ) -> Result<TokenDistribution> {
        let logits = self.logits_from_final_state(final_state)?;
        token_distribution_top_k_top_p(
            logits.as_row()?,
            options.temperature,
            options.top_p,
            options.top_k,
        )
    }

    fn token_distribution_from_mtp_state(
        &self,
        final_state: &Tensor,
        options: &GenerationOptions,
    ) -> Result<TokenDistribution> {
        let logits = self.logits_from_mtp_draft_state(final_state)?;
        token_distribution_top_k_top_p(
            logits.as_row()?,
            options.temperature,
            options.top_p,
            options.top_k,
        )
    }

    fn sample_mtp_token_and_distribution_from_state(
        &self,
        final_state: &Tensor,
        options: &GenerationOptions,
        sampler: &mut DeterministicSampler,
    ) -> Result<(usize, TokenDistribution)> {
        let distribution = self.token_distribution_from_mtp_state(final_state, options)?;
        let token = sample_from_token_distribution(&distribution, sampler)?;
        Ok((token, distribution))
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
        let (cache, final_state) = self.prefill_cache_state(prompt)?;
        // Decode résident COMPLET (1c) si le flag est ON ET le modèle est supporté
        // (validation en amont, tout-ou-rien). Sinon, decode résident full-attn (1b)
        // si son flag est ON : alloue/seed le KV GPU des couches full-attn.
        let prefill = prefill_started.elapsed();
        self.generate_greedy_timed_from_prompt_state_with_options(
            CausalDecoderPromptState::new(cache, final_state),
            prefill,
            max_new_tokens,
            options,
        )
    }

    /// Génère depuis un état de prompt pré-rempli.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si l'état, les options ou le forward échouent.
    pub fn generate_greedy_timed_from_prompt_state_with_options(
        &self,
        state: CausalDecoderPromptState,
        prefill: Duration,
        max_new_tokens: usize,
        options: &GenerationOptions,
    ) -> Result<GenerationOutput> {
        self.generate_greedy_timed_from_prompt_state_inner(
            state,
            prefill,
            max_new_tokens,
            options,
            true,
            |_| true,
        )
    }

    /// Génère depuis un état pré-rempli en appelant `on_token` par token.
    ///
    /// Arrête la boucle si `on_token` renvoie `false`.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si l'état, les options ou le forward échouent.
    pub fn generate_greedy_timed_from_prompt_state_with_options_and_callback(
        &self,
        state: CausalDecoderPromptState,
        prefill: Duration,
        max_new_tokens: usize,
        options: &GenerationOptions,
        on_token: impl FnMut(usize) -> bool,
    ) -> Result<GenerationOutput> {
        self.generate_greedy_timed_from_prompt_state_inner(
            state,
            prefill,
            max_new_tokens,
            options,
            false,
            on_token,
        )
    }

    fn generate_greedy_timed_from_prompt_state_inner(
        &self,
        state: CausalDecoderPromptState,
        prefill: Duration,
        max_new_tokens: usize,
        options: &GenerationOptions,
        allow_pipelined: bool,
        on_token: impl FnMut(usize) -> bool,
    ) -> Result<GenerationOutput> {
        if max_new_tokens == 0 {
            return Ok(GenerationOutput {
                tokens: Vec::new(),
                timings: GenerationTimings {
                    prefill,
                    decode: Duration::ZERO,
                    decode_tokens: 0,
                },
            });
        }
        if state.position() == 0 {
            return Err(InferError::Dimension("état de prompt vide".to_string()));
        }
        let (cache, final_state) = state.into_parts();
        self.generate_greedy_timed_from_prefilled_state(
            cache,
            final_state,
            prefill,
            max_new_tokens,
            options,
            allow_pipelined,
            on_token,
        )
    }

    fn generate_greedy_timed_from_prefilled_state(
        &self,
        mut cache: CausalDecoderCache,
        mut final_state: Tensor,
        prefill: Duration,
        max_new_tokens: usize,
        options: &GenerationOptions,
        allow_pipelined: bool,
        mut on_token: impl FnMut(usize) -> bool,
    ) -> Result<GenerationOutput> {
        if max_new_tokens == 0 {
            return Ok(GenerationOutput {
                tokens: Vec::new(),
                timings: GenerationTimings {
                    prefill,
                    decode: Duration::ZERO,
                    decode_tokens: 0,
                },
            });
        }
        // Decode résident COMPLET (1c) : greedy uniquement (argmax on-device) et
        // modèle supporté. L'arène doit être prête tout-ou-rien : si un état par
        // couche manque, on retombe sur le per-op (et le résident 1b si activé).
        let guided_sampling = options.token_constraint.is_some();
        #[cfg(all(target_os = "macos", feature = "metal"))]
        let resident_sampling = super::resident::resident_sampling_supported(options);
        #[cfg(all(target_os = "macos", feature = "metal"))]
        let mut use_resident_full = decode_resident_full_enabled()
            && !guided_sampling
            && (options.temperature <= f32::EPSILON || resident_sampling)
            && self.supports_resident_full_decode();
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if use_resident_full {
            use_resident_full = self.setup_resident_full_decode(
                &mut cache,
                max_new_tokens,
                options.temperature > f32::EPSILON,
            )?;
        }
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if decode_resident_enabled() && !use_resident_full && !guided_sampling {
            self.setup_resident_decode(
                &mut cache,
                max_new_tokens,
                options.temperature > f32::EPSILON,
            )?;
        }
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        let _ = guided_sampling;
        let mut generated = Vec::with_capacity(max_new_tokens);
        let mut sampler = DeterministicSampler::new(options.seed);
        let mut decode = Duration::ZERO;
        let mut decode_tokens = 0_usize;
        let mut token = self.sample_token_from_state(&final_state, options, &mut sampler)?;
        #[cfg(all(target_os = "macos", feature = "metal"))]
        let decode_profiler = DecodeProfiler::start();
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if allow_pipelined
            && use_resident_full
            && decode_pipeline_enabled()
            && decode_min_interval().is_none()
            && options.stop_sequences.is_empty()
            && oracle_dump_logits_topk().is_none()
            && (options.temperature <= f32::EPSILON
                || super::resident::resident_sampling_on_device(options))
        {
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
        let throttle_interval = decode_min_interval();
        let mut throttle_last: Option<Instant> = None;
        for step in 0..max_new_tokens {
            if options.stop_token_ids.contains(&token) {
                break;
            }
            generated.push(token);
            if !on_token(token) {
                break;
            }
            if generated_matches_stop_sequence(&generated, &options.stop_sequences) {
                break;
            }
            if let Some(interval) = throttle_interval {
                if let Some(last) = throttle_last {
                    let elapsed = last.elapsed();
                    if elapsed < interval {
                        std::thread::sleep(interval - elapsed);
                    }
                }
                throttle_last = Some(Instant::now());
            }
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

    fn restore_mtp_snapshot_replay(
        &self,
        cache: &mut CausalDecoderCache,
        snapshot: &mut Option<SpeculativeCacheSnapshot>,
        verify_tokens: &[usize],
        replay_len: usize,
        final_state: &mut Tensor,
    ) -> Result<()> {
        let snapshot = snapshot
            .take()
            .ok_or_else(|| InferError::Config("snapshot MTP fallback absent".to_string()))?;
        self.restore_speculative_cache(cache, snapshot)?;
        for token in verify_tokens.iter().take(replay_len) {
            *final_state = self.next_decode_state(cache, *token)?;
        }
        Ok(())
    }

    /// Mesure un decode MTP greedy avec verify trunk batché.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si la tête MTP est absente ou si un forward échoue.
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
                loop_duration: Duration::default(),
            });
        }

        let use_committed_mtp_history = mtp_history_policy() == MtpHistoryPolicy::Committed;
        let (mut cache, mut final_state, mut mtp_history_cache) = if use_committed_mtp_history {
            let (cache, final_state, mtp_history) =
                self.prefill_cache_state_with_mtp_history(prompt)?;
            (cache, final_state, Some(mtp_history))
        } else {
            let (cache, final_state) = self.prefill_cache_state(prompt)?;
            (cache, final_state, None)
        };
        let mut resident_mtp_history_len = mtp_history_cache.as_ref().map_or(0, LayerKvCache::len);
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let mut use_resident_full = false;
            if decode_resident_full_enabled()
                && options.temperature <= f32::EPSILON
                && self.supports_resident_full_decode()
            {
                use_resident_full = self.setup_resident_full_decode(
                    &mut cache,
                    max_new_tokens,
                    options.temperature > f32::EPSILON,
                )?;
            }
            if decode_resident_enabled() && !use_resident_full {
                self.setup_resident_decode(
                    &mut cache,
                    max_new_tokens,
                    options.temperature > f32::EPSILON,
                )?;
            }
            if use_resident_full {
                if let Some(mtp_history) = mtp_history_cache.as_ref() {
                    self.seed_mtp_resident_history(&mut cache, mtp_history)?;
                }
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
        #[cfg(all(target_os = "macos", feature = "metal"))]
        let profile_mtp_steps = mtp_step_profile_enabled();
        #[cfg(all(target_os = "macos", feature = "metal"))]
        let mut mtp_step_profile = MtpStepProfile::default();
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

            let cycle_mtp_offset = if mtp_history_cache.is_some() {
                resident_mtp_history_len
            } else {
                0
            };
            #[cfg(all(target_os = "macos", feature = "metal"))]
            if options.temperature <= f32::EPSILON
                && use_committed_mtp_history
                && max_draft_tokens == 1
                && mtp_history_cache.is_some()
            {
                let profile_start = mtp_profile_start(profile_mtp_steps);
                if let Some((draft, target, target_state, bonus)) = self
                    .next_mtp_spec_one_resident(
                        &mut cache,
                        &final_state,
                        primary,
                        cycle_mtp_offset,
                        &options.stop_token_ids,
                    )?
                {
                    mtp_profile_record(&mut mtp_step_profile.fused_one, profile_start);
                    stats.verifications += 1;
                    stats.proposed += 1;
                    stats.proposed_by_position[0] += 1;
                    // En mode fresh, le chemin résident ignore l'historique MTP
                    // committed (self-only, prouvé inerte) : on ne le fait pas
                    // croître ni ne le tronque (évite un truncate au-delà de la
                    // longueur réelle, qui exposerait des lignes KV périmées).
                    let fresh = mtp_fresh_cache_enabled();
                    let committed_len = cycle_mtp_offset.checked_add(1).ok_or_else(|| {
                        InferError::Dimension("taille historique MTP déborde".to_string())
                    })?;
                    if options.stop_token_ids.contains(&target) {
                        if draft == target {
                            stats.accepted += 1;
                            stats.accepted_by_position[0] += 1;
                        } else {
                            stats.rejected += 1;
                        }
                        pending = target;
                        final_state = target_state;
                        if !fresh {
                            self.truncate_mtp_resident_history(&mut cache, committed_len)?;
                            resident_mtp_history_len = committed_len;
                        }
                        continue;
                    }
                    if draft == target {
                        stats.accepted += 1;
                        stats.accepted_by_position[0] += 1;
                        generated.push(draft);
                        if !fresh {
                            resident_mtp_history_len =
                                committed_len.checked_add(1).ok_or_else(|| {
                                    InferError::Dimension(
                                        "taille historique MTP déborde".to_string(),
                                    )
                                })?;
                        }
                        if let Some((bonus_token, bonus_state)) = bonus {
                            final_state = bonus_state;
                            if generated.len() < max_new_tokens {
                                pending = bonus_token;
                                stats.verifications += 1;
                            }
                        } else {
                            final_state = target_state;
                            if generated.len() < max_new_tokens {
                                final_state = self.next_decode_state(&mut cache, draft)?;
                                pending = self.sample_token_from_state(
                                    &final_state,
                                    options,
                                    &mut sampler,
                                )?;
                                stats.verifications += 1;
                            }
                        }
                    } else {
                        stats.rejected += 1;
                        pending = target;
                        final_state = target_state;
                        if !fresh {
                            self.truncate_mtp_resident_history(&mut cache, committed_len)?;
                            resident_mtp_history_len = committed_len;
                        }
                    }
                    continue;
                } else {
                    mtp_profile_record(&mut mtp_step_profile.fused_one, profile_start);
                }
            }
            #[cfg(all(target_os = "macos", feature = "metal"))]
            if options.temperature <= f32::EPSILON
                && use_committed_mtp_history
                && max_draft_tokens == 2
                && mtp_history_cache.is_some()
                && max_new_tokens.saturating_sub(generated.len()) >= 2
            {
                let profile_start = mtp_profile_start(profile_mtp_steps);
                if let Some(output) = self.next_mtp_spec_two_resident(
                    &mut cache,
                    &final_state,
                    primary,
                    cycle_mtp_offset,
                    &options.stop_token_ids,
                )? {
                    mtp_profile_record(&mut mtp_step_profile.fused_two, profile_start);
                    for position in 0..output.checked {
                        stats.verifications += 1;
                        stats.proposed += 1;
                        stats.proposed_by_position[position] += 1;
                        if output.accepted_for_stats[position] {
                            stats.accepted += 1;
                            stats.accepted_by_position[position] += 1;
                        } else {
                            stats.rejected += 1;
                            break;
                        }
                        if options.stop_token_ids.contains(&output.targets[position]) {
                            break;
                        }
                    }
                    for token in output.drafts.iter().take(output.accepted_generated) {
                        generated.push(*token);
                    }
                    if output.bonus_verified && generated.len() < max_new_tokens {
                        stats.verifications += 1;
                    }
                    final_state = output.final_state;
                    pending = output.pending;
                    resident_mtp_history_len = cycle_mtp_offset
                        .checked_add(output.committed_rows)
                        .ok_or_else(|| {
                            InferError::Dimension("taille historique MTP déborde".to_string())
                        })?;
                    continue;
                } else {
                    mtp_profile_record(&mut mtp_step_profile.fused_two, profile_start);
                }
            }
            let mut local_mtp_cache = LayerKvCache::default();
            let mut draft_hidden = final_state.clone();
            let mut draft_token = primary;
            let mut drafts = Vec::with_capacity(max_draft_tokens);
            let mut draft_distributions = Vec::with_capacity(max_draft_tokens);
            #[cfg(all(target_os = "macos", feature = "metal"))]
            let mtp_resident = if options.temperature <= f32::EPSILON {
                let profile_start = mtp_profile_start(profile_mtp_steps);
                let resident = self.start_mtp_resident_draft(
                    &mut cache,
                    &final_state,
                    mtp_history_cache.as_ref().map(|_| resident_mtp_history_len),
                )?;
                mtp_profile_record(&mut mtp_step_profile.draft_setup, profile_start);
                resident
            } else {
                false
            };
            #[cfg(not(all(target_os = "macos", feature = "metal")))]
            let mtp_resident = false;
            #[cfg(all(target_os = "macos", feature = "metal"))]
            if mtp_resident {
                let profile_start = mtp_profile_start(profile_mtp_steps);
                if let Some(resident_drafts) = self.next_mtp_drafts_resident(
                    &mut cache,
                    draft_token,
                    max_draft_tokens,
                    cycle_mtp_offset,
                )? {
                    drafts = resident_drafts;
                    draft_token = drafts.last().copied().unwrap_or(draft_token);
                }
                mtp_profile_record(&mut mtp_step_profile.draft, profile_start);
            }
            if drafts.is_empty() {
                for position in 0..max_draft_tokens {
                    let absolute_position = cycle_mtp_offset
                        .checked_add(position)
                        .ok_or_else(|| InferError::Dimension("position MTP déborde".to_string()))?;
                    let draft = if mtp_resident {
                        #[cfg(all(target_os = "macos", feature = "metal"))]
                        {
                            self.next_mtp_draft_resident(
                                &mut cache,
                                draft_token,
                                absolute_position,
                            )?
                        }
                        #[cfg(not(all(target_os = "macos", feature = "metal")))]
                        unreachable!("invariant: mtp_resident=false hors Metal")
                    } else {
                        let mtp_cache = if let Some(history_cache) = mtp_history_cache.as_mut() {
                            history_cache
                        } else {
                            &mut local_mtp_cache
                        };
                        if options.temperature > f32::EPSILON {
                            let next_hidden = self.mtp_forward_post_hidden(
                                &draft_hidden,
                                draft_token,
                                absolute_position,
                                mtp_cache,
                            )?;
                            let (draft, distribution) = self
                                .sample_mtp_token_and_distribution_from_state(
                                    &next_hidden,
                                    options,
                                    &mut sampler,
                                )?;
                            draft_distributions.push(Some(distribution));
                            draft_hidden = next_hidden;
                            draft
                        } else {
                            let (draft, next_hidden) = self.mtp_forward_greedy_with_hidden(
                                &draft_hidden,
                                draft_token,
                                absolute_position,
                                mtp_cache,
                                options,
                                &mut sampler,
                            )?;
                            draft_hidden = next_hidden;
                            draft
                        }
                    };
                    drafts.push(draft);
                    draft_token = draft;
                }
            } else {
                draft_distributions.resize(drafts.len(), None);
            }
            #[cfg(all(target_os = "macos", feature = "metal"))]
            let profile_start = mtp_profile_start(profile_mtp_steps);
            let verified = self.verify_mtp_drafts_batched(MtpVerify {
                cache: &mut cache,
                final_state: &mut final_state,
                pending: &mut pending,
                generated: &mut generated,
                drafts,
                draft_distributions,
                primary,
                max_new_tokens,
                options,
                sampler: &mut sampler,
                stats: &mut stats,
            })?;
            #[cfg(all(target_os = "macos", feature = "metal"))]
            mtp_profile_record(&mut mtp_step_profile.verify, profile_start);
            let MtpVerifyResult {
                accepted_all,
                verify_token,
                accepted_history,
            } = verified;

            if let Some(history_cache) = mtp_history_cache.as_mut() {
                #[cfg(all(target_os = "macos", feature = "metal"))]
                let profile_start = mtp_profile_start(profile_mtp_steps);
                let committed_len = cycle_mtp_offset.checked_add(1).ok_or_else(|| {
                    InferError::Dimension("taille historique MTP déborde".to_string())
                })?;
                if mtp_resident {
                    #[cfg(all(target_os = "macos", feature = "metal"))]
                    {
                        self.truncate_mtp_resident_history(&mut cache, committed_len)?;
                        self.append_mtp_resident_history_steps(
                            &mut cache,
                            &accepted_history,
                            committed_len,
                        )?;
                        resident_mtp_history_len = committed_len
                            .checked_add(accepted_history.len())
                            .ok_or_else(|| {
                                InferError::Dimension("taille historique MTP déborde".to_string())
                            })?;
                    }
                    #[cfg(not(all(target_os = "macos", feature = "metal")))]
                    {
                        let _ = history_cache;
                    }
                } else {
                    history_cache.truncate(committed_len)?;
                    for (history_state, history_token) in &accepted_history {
                        let position = history_cache.len();
                        let _ = self.mtp_forward_post_hidden(
                            history_state,
                            *history_token,
                            position,
                            history_cache,
                        )?;
                    }
                    resident_mtp_history_len = history_cache.len();
                }
                #[cfg(all(target_os = "macos", feature = "metal"))]
                mtp_profile_record(&mut mtp_step_profile.history, profile_start);
            }

            if accepted_all {
                #[cfg(all(target_os = "macos", feature = "metal"))]
                let profile_start = mtp_profile_start(profile_mtp_steps);
                final_state = self.next_decode_state(&mut cache, verify_token)?;
                pending = self.sample_token_from_state(&final_state, options, &mut sampler)?;
                stats.verifications += 1;
                #[cfg(all(target_os = "macos", feature = "metal"))]
                mtp_profile_record(&mut mtp_step_profile.bonus_next, profile_start);
            }
        }

        let loop_duration = decode_started.elapsed();
        #[cfg(all(target_os = "macos", feature = "metal"))]
        decode_profiler.report_decode_loop(loop_duration, generated.len());
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if profile_mtp_steps {
            mtp_step_profile.report(generated.len(), stats.verifications);
        }
        Ok(SpeculativeOutput {
            tokens: generated,
            stats,
            loop_duration,
        })
    }

    /// Génère en greedy avec un draft DFlash et une vérification trunk batchée.
    ///
    /// Ce chemin est l'oracle fonctionnel: les captures de hidden target et le
    /// mini-décodeur draft restent génériques. Les kernels DFlash Metal pourront
    /// remplacer la proposition sans toucher à l'acceptance/rollback.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si DFlash est incompatible, si le mode n'est pas greedy,
    /// ou si une capture/projection/vérification échoue.
    #[cfg(feature = "devtools")]
    pub fn generate_greedy_dflash_batched_with_options(
        &self,
        prompt: &[usize],
        max_new_tokens: usize,
        options: &GenerationOptions,
        draft: &DFlashDraft,
        max_draft_tokens: usize,
    ) -> Result<SpeculativeOutput> {
        if options.temperature > f32::EPSILON {
            return Err(InferError::Config(
                "decode DFlash supporte uniquement greedy temperature=0".to_string(),
            ));
        }
        if max_draft_tokens == 0 {
            return Err(InferError::Config(
                "decode DFlash sans draft token".to_string(),
            ));
        }
        if prompt.is_empty() {
            return Err(InferError::Dimension("prompt token vide".to_string()));
        }
        self.validate_dflash_draft(draft)?;
        self.ensure_speculative_cache_snapshot_supported()?;
        if max_new_tokens == 0 {
            return Ok(SpeculativeOutput {
                tokens: Vec::new(),
                stats: SpeculativeStats::default(),
                loop_duration: Duration::default(),
            });
        }

        let draft_limit = max_draft_tokens.min(draft.info.block_size);
        let (mut cache, mut final_state, target_hidden) =
            self.prefill_cache_state_with_layer_capture(prompt, &draft.info.target_layer_ids)?;
        // Réserve l'avance spéculative : un cycle propose jusqu'à `draft_limit` tokens au-delà du
        // dernier token retenu avant de tronquer les rejets ; sans cette marge le KV résident
        // déborde en fin de génération (capacité = prompt + max_new_tokens exactement).
        self.setup_resident_decode_from_prefill(&mut cache, max_new_tokens + draft_limit + 1)?;
        let mut projected_context =
            draft.project_target_hidden(&target_hidden, self.forward_runtime())?;
        let mut sampler = DeterministicSampler::new(options.seed);
        let mut pending = self.sample_token_from_state(&final_state, options, &mut sampler)?;
        let mut generated = Vec::with_capacity(max_new_tokens);
        let mut stats = SpeculativeStats::default();
        stats.proposed_by_position.resize(draft_limit, 0);
        stats.accepted_by_position.resize(draft_limit, 0);
        let loop_started = Instant::now();

        while generated.len() < max_new_tokens {
            let primary = pending;
            if options.stop_token_ids.contains(&primary) {
                break;
            }
            generated.push(primary);
            if generated.len() >= max_new_tokens || options.stop_token_ids.contains(&primary) {
                break;
            }

            let drafts = self.dflash_draft_greedy_tokens(
                draft,
                &projected_context,
                primary,
                draft_limit,
                options,
                &mut sampler,
            )?;
            let verified = self.verify_dflash_drafts_batched(DFlashVerify {
                draft,
                cache: &mut cache,
                projected_context: &mut projected_context,
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

            if generated.len() >= max_new_tokens
                || generated
                    .last()
                    .is_some_and(|token| options.stop_token_ids.contains(token))
            {
                break;
            }
            if verified.accepted_all {
                let (state, target_hidden) = self.next_final_state_with_layer_capture(
                    &mut cache,
                    verified.verify_token,
                    &draft.info.target_layer_ids,
                )?;
                final_state = state;
                pending = self.sample_token_from_state(&final_state, options, &mut sampler)?;
                self.append_dflash_projected_context(
                    draft,
                    &mut projected_context,
                    &target_hidden,
                )?;
                stats.verifications += 1;
            }
        }

        Ok(SpeculativeOutput {
            tokens: generated,
            stats,
            loop_duration: loop_started.elapsed(),
        })
    }

    /// Génère une référence AR via le même chemin target-capture que DFlash.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si DFlash est incompatible, si le prompt est vide ou si
    /// un forward avec capture échoue.
    #[cfg(feature = "devtools")]
    pub fn generate_greedy_dflash_reference_with_options(
        &self,
        prompt: &[usize],
        max_new_tokens: usize,
        options: &GenerationOptions,
        draft: &DFlashDraft,
    ) -> Result<Vec<usize>> {
        if options.temperature > f32::EPSILON {
            return Err(InferError::Config(
                "référence DFlash supporte uniquement greedy temperature=0".to_string(),
            ));
        }
        if prompt.is_empty() {
            return Err(InferError::Dimension("prompt token vide".to_string()));
        }
        self.validate_dflash_draft(draft)?;
        if max_new_tokens == 0 {
            return Ok(Vec::new());
        }

        let (mut cache, final_state, _) =
            self.prefill_cache_state_with_layer_capture(prompt, &draft.info.target_layer_ids)?;
        self.setup_resident_decode_from_prefill(&mut cache, max_new_tokens)?;
        let mut sampler = DeterministicSampler::new(options.seed);
        let mut token = self.sample_token_from_state(&final_state, options, &mut sampler)?;
        let mut generated = Vec::with_capacity(max_new_tokens);
        for step in 0..max_new_tokens {
            if options.stop_token_ids.contains(&token) {
                break;
            }
            generated.push(token);
            if generated_matches_stop_sequence(&generated, &options.stop_sequences) {
                break;
            }
            if step + 1 < max_new_tokens {
                let (state, _) = self.next_final_state_with_layer_capture(
                    &mut cache,
                    token,
                    &draft.info.target_layer_ids,
                )?;
                token = self.sample_token_from_state(&state, options, &mut sampler)?;
            }
        }
        Ok(generated)
    }

    /// Compare un pas per-op et un pas résident full avec captures par couche.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le mode n'est pas greedy, si le résident full n'est
    /// pas disponible ou si une capture de couche échoue.
    #[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
    pub fn resident_linear_xray(
        &self,
        prompt: &[usize],
        options: &GenerationOptions,
    ) -> Result<ResidentLinearXrayReport> {
        if options.temperature > f32::EPSILON {
            return Err(InferError::Config(
                "xray résident-linear supporte uniquement greedy temperature=0".to_string(),
            ));
        }
        if prompt.is_empty() {
            return Err(InferError::Dimension("prompt token vide".to_string()));
        }
        let layer_ids = (0..self.layers.len()).collect::<Vec<_>>();
        validate_capture_layer_ids(&layer_ids, self.layers.len())?;
        let (base_cache, final_state) = self.prefill_cache_state_uncached(prompt)?;
        let mut sampler = DeterministicSampler::new(options.seed);
        let input_token = self.sample_token_from_state(&final_state, options, &mut sampler)?;

        #[cfg(all(target_os = "macos", feature = "metal"))]
        let reference_metal = self.snapshot_prefix_cache_linear_metal(&base_cache)?;
        #[cfg(all(target_os = "macos", feature = "metal"))]
        let resident_metal = self.snapshot_prefix_cache_linear_metal(&base_cache)?;
        #[cfg(all(target_os = "macos", feature = "metal"))]
        let probe_metal = self.snapshot_prefix_cache_linear_metal(&base_cache)?;

        let probe =
            self.resident_linear_first_layer_probe(&base_cache, probe_metal, input_token)?;

        let mut reference_cache = base_cache.clone();
        #[cfg(all(target_os = "macos", feature = "metal"))]
        self.restore_prefix_cache_linear_metal(&mut reference_cache, reference_metal)?;
        let (reference_states, reference_hidden) = self
            .next_final_states_batched_with_layer_capture(
                &mut reference_cache,
                &[input_token],
                &layer_ids,
            )?;

        let mut resident_cache = base_cache.clone();
        #[cfg(all(target_os = "macos", feature = "metal"))]
        self.restore_prefix_cache_linear_metal(&mut resident_cache, resident_metal)?;
        if !self.setup_resident_full_decode(&mut resident_cache, 2, false)? {
            return Err(InferError::Config(
                "xray résident-linear: setup résident full indisponible".to_string(),
            ));
        }
        let resident_output = self
            .next_final_states_resident_verify(
                &mut resident_cache,
                &[input_token],
                false,
                false,
                Some(&layer_ids),
            )?
            .ok_or_else(|| {
                InferError::Config("xray résident-linear: verify résident absent".to_string())
            })?;
        let resident_hidden = resident_output.target_hidden.ok_or_else(|| {
            InferError::Config("xray résident-linear: capture résident absente".to_string())
        })?;

        let reference_state = Tensor::row(reference_states.row_slice(0)?.to_vec())?;
        let resident_state = Tensor::row(resident_output.states.row_slice(0)?.to_vec())?;
        let reference_token =
            self.sample_token_from_state(&reference_state, options, &mut sampler)?;
        let resident_token =
            self.sample_token_from_state(&resident_state, options, &mut sampler)?;
        let (final_max_abs, final_mean_abs) =
            diff_stats_same_len(reference_state.data(), resident_state.data())?;

        let hidden = self.final_norm.len();
        let mut layer_diffs = Vec::with_capacity(self.layers.len());
        for layer_index in 0..self.layers.len() {
            let start = layer_index.checked_mul(hidden).ok_or_else(|| {
                InferError::Dimension("xray résident-linear offset déborde".to_string())
            })?;
            let end = start.checked_add(hidden).ok_or_else(|| {
                InferError::Dimension("xray résident-linear fin déborde".to_string())
            })?;
            let reference_slice = reference_hidden.data().get(start..end).ok_or_else(|| {
                InferError::Dimension(format!(
                    "xray référence couche {layer_index} hors capture {:?}",
                    reference_hidden.shape()
                ))
            })?;
            let resident_slice = resident_hidden.data().get(start..end).ok_or_else(|| {
                InferError::Dimension(format!(
                    "xray résident couche {layer_index} hors capture {:?}",
                    resident_hidden.shape()
                ))
            })?;
            let (max_abs, mean_abs) = diff_stats_same_len(reference_slice, resident_slice)?;
            let attention_kind = if self.config.is_full_attention_layer(layer_index) {
                "full"
            } else {
                "linear"
            };
            layer_diffs.push(ResidentLinearXrayLayerDiff {
                layer_index,
                attention_kind: attention_kind.to_string(),
                max_abs,
                mean_abs,
            });
        }

        Ok(ResidentLinearXrayReport {
            input_token,
            reference_token,
            resident_token,
            final_max_abs,
            final_mean_abs,
            probe_layer_index: probe.as_ref().map(|probe| probe.layer_index),
            probe_normed_max_abs: probe.as_ref().map(|probe| probe.normed_max_abs),
            probe_normed_mean_abs: probe.as_ref().map(|probe| probe.normed_mean_abs),
            probe_attn_max_abs: probe.as_ref().map(|probe| probe.attn_max_abs),
            probe_attn_mean_abs: probe.as_ref().map(|probe| probe.attn_mean_abs),
            probe_attn_cpu_normed_max_abs: probe
                .as_ref()
                .map(|probe| probe.attn_cpu_normed_max_abs),
            probe_attn_cpu_normed_mean_abs: probe
                .as_ref()
                .map(|probe| probe.attn_cpu_normed_mean_abs),
            probe_init_state_conv_max_abs: probe
                .as_ref()
                .map(|probe| probe.init_state_conv_max_abs),
            probe_init_state_conv_mean_abs: probe
                .as_ref()
                .map(|probe| probe.init_state_conv_mean_abs),
            probe_init_state_ssm_max_abs: probe.as_ref().map(|probe| probe.init_state_ssm_max_abs),
            probe_init_state_ssm_mean_abs: probe
                .as_ref()
                .map(|probe| probe.init_state_ssm_mean_abs),
            probe_state_conv_max_abs: probe.as_ref().map(|probe| probe.state_conv_max_abs),
            probe_state_conv_mean_abs: probe.as_ref().map(|probe| probe.state_conv_mean_abs),
            probe_state_ssm_max_abs: probe.as_ref().map(|probe| probe.state_ssm_max_abs),
            probe_state_ssm_mean_abs: probe.as_ref().map(|probe| probe.state_ssm_mean_abs),
            probe_state_cpu_normed_conv_max_abs: probe
                .as_ref()
                .map(|probe| probe.state_cpu_normed_conv_max_abs),
            probe_state_cpu_normed_conv_mean_abs: probe
                .as_ref()
                .map(|probe| probe.state_cpu_normed_conv_mean_abs),
            probe_state_cpu_normed_ssm_max_abs: probe
                .as_ref()
                .map(|probe| probe.state_cpu_normed_ssm_max_abs),
            probe_state_cpu_normed_ssm_mean_abs: probe
                .as_ref()
                .map(|probe| probe.state_cpu_normed_ssm_mean_abs),
            layer_diffs,
        })
    }

    #[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
    fn resident_linear_first_layer_probe(
        &self,
        base_cache: &CausalDecoderCache,
        linear_metal: Vec<Option<crate::metal_backend::LinearAttentionMetalState>>,
        input_token: usize,
    ) -> Result<Option<ResidentLinearXrayProbe>> {
        let Some(layer_index) =
            (0..self.layers.len()).find(|index| !self.config.is_full_attention_layer(*index))
        else {
            return Ok(None);
        };
        if layer_index != 0 {
            return Ok(None);
        }
        let layer = &self.layers[layer_index];
        let AttentionBlock::Linear(attention) = &layer.attention else {
            return Ok(None);
        };
        let Some(metal) = self.forward_runtime().metal_executor() else {
            return Ok(None);
        };

        let input = self.embed_scaled(&[input_token])?;
        let reference_normed = rms_norm(&input, &layer.input_norm, self.config.rms_eps)?;
        let reference_metal = self.copy_prefix_cache_linear_metal(&linear_metal)?;

        let mut reference_cache = base_cache.clone();
        self.restore_prefix_cache_linear_metal(&mut reference_cache, reference_metal)?;
        let reference_state_before = {
            let state = reference_cache.layers[layer_index]
                .linear
                .metal_state()
                .ok_or_else(|| {
                    InferError::Metal("xray linear: état référence absent avant step".to_string())
                })?;
            metal.snapshot_linear_attn_state(state)?
        };
        let reference_attn = attention.forward_cached_with_runtime(
            self.config.linear_attention_config()?,
            &reference_normed,
            &mut reference_cache.layers[layer_index].linear,
            self.forward_runtime(),
        )?;
        let reference_state_after = {
            let state = reference_cache.layers[layer_index]
                .linear
                .metal_state()
                .ok_or_else(|| {
                    InferError::Metal("xray linear: état référence absent après step".to_string())
                })?;
            metal.snapshot_linear_attn_state(state)?
        };

        let mut probe_cache = base_cache.clone();
        self.restore_prefix_cache_linear_metal(&mut probe_cache, linear_metal)?;
        if !self.setup_resident_full_decode(&mut probe_cache, 2, false)? {
            return Ok(None);
        }

        let la_config = self.config.linear_attention_config()?;
        let la_spec = crate::metal_backend::LinearAttentionStepSpec {
            num_key_heads: la_config.num_key_heads,
            num_value_heads: la_config.num_value_heads,
            key_head_dim: la_config.key_head_dim,
            value_head_dim: la_config.value_head_dim,
            conv_kernel_dim: la_config.conv_kernel_dim,
            rms_eps: la_config.rms_eps,
        };
        let key_dim = la_config.key_dim()?;
        let value_dim = la_config.value_dim()?;
        let conv_dim = key_dim
            .checked_mul(2)
            .and_then(|twice| twice.checked_add(value_dim))
            .ok_or_else(|| InferError::Shape("xray linear conv_dim déborde".to_string()))?;
        let res_dims = crate::metal_backend::LinearAttnResidentDims {
            in_dim: self.final_norm.len(),
            conv_dim,
            value_dim,
            key_dim,
        };
        let state = probe_cache.layers[layer_index]
            .linear
            .metal_state()
            .ok_or_else(|| {
                InferError::Metal("xray linear: état Metal linear-attn absent".to_string())
            })?
            .clone();
        let gpu_normed_state = metal.snapshot_linear_attn_state(&state)?;
        let cpu_normed_state = metal.snapshot_linear_attn_state(&state)?;
        let init_state_diff =
            metal.diff_linear_attn_states(&reference_state_before, &gpu_normed_state)?;

        let CausalDecoderCache { resident, .. } = &mut probe_cache;
        let arena = resident
            .as_mut()
            .ok_or_else(|| InferError::Metal("xray linear: arène résidente absente".to_string()))?;
        let hidden = self.final_norm.len();
        arena.state.upload(&arena.hidden_a, input.as_row()?)?;
        let input_norm =
            metal.cached_buffer_from_f32(layer.input_norm.data(), "xray_linear_input_norm")?;
        let normed = arena
            .state
            .scratch()
            .lease(hidden, crate::decode_resident::GpuElement::F32)?;
        let attn_out = arena
            .state
            .scratch()
            .lease(hidden, crate::decode_resident::GpuElement::F32)?;
        let attn_cpu_normed_out = arena
            .state
            .scratch()
            .lease(hidden, crate::decode_resident::GpuElement::F32)?;

        let command_buffer = arena.state.queue().new_command_buffer();
        let _barrier_guard = crate::metal_backend::resident_concurrent_enabled()
            .then(crate::metal_backend::install_dispatch_barrier_scope);
        let _namespace_guard =
            crate::metal_backend::install_scratch_namespace(arena.state.scratch_namespace());
        let encoder = super::new_resident_compute_encoder(command_buffer);
        metal.encode_rms_norm_rows(
            encoder,
            arena.hidden_a.buffer(),
            &input_norm,
            normed.tensor().buffer(),
            1,
            hidden,
            self.config.rms_eps,
        )?;
        encoder.end_encoding();
        crate::metal_backend::commit_and_wait(command_buffer)?;
        let resident_normed =
            crate::metal_backend::read_f32_buffer(normed.tensor().buffer(), hidden)?;

        let command_buffer = arena.state.queue().new_command_buffer();
        let _barrier_guard = crate::metal_backend::resident_concurrent_enabled()
            .then(crate::metal_backend::install_dispatch_barrier_scope);
        let _namespace_guard =
            crate::metal_backend::install_scratch_namespace(arena.state.scratch_namespace());
        let encoder = super::new_resident_compute_encoder(command_buffer);
        let mut owned = Vec::new();
        metal.encode_linear_attn_resident(
            encoder,
            &mut owned,
            normed.tensor().buffer(),
            attn_out.tensor().buffer(),
            attention.resident_weights(),
            &gpu_normed_state,
            la_spec,
            res_dims,
        )?;
        encoder.end_encoding();
        crate::metal_backend::commit_and_wait(command_buffer)?;
        let resident_attn =
            crate::metal_backend::read_f32_buffer(attn_out.tensor().buffer(), hidden)?;
        let resident_state_after = metal.snapshot_linear_attn_state(&gpu_normed_state)?;

        arena
            .state
            .upload(normed.tensor(), reference_normed.as_row()?)?;
        let command_buffer = arena.state.queue().new_command_buffer();
        let _barrier_guard = crate::metal_backend::resident_concurrent_enabled()
            .then(crate::metal_backend::install_dispatch_barrier_scope);
        let _namespace_guard =
            crate::metal_backend::install_scratch_namespace(arena.state.scratch_namespace());
        let encoder = super::new_resident_compute_encoder(command_buffer);
        let mut owned = Vec::new();
        metal.encode_linear_attn_resident(
            encoder,
            &mut owned,
            normed.tensor().buffer(),
            attn_cpu_normed_out.tensor().buffer(),
            attention.resident_weights(),
            &cpu_normed_state,
            la_spec,
            res_dims,
        )?;
        encoder.end_encoding();
        crate::metal_backend::commit_and_wait(command_buffer)?;
        let resident_attn_cpu_normed =
            crate::metal_backend::read_f32_buffer(attn_cpu_normed_out.tensor().buffer(), hidden)?;
        let resident_cpu_normed_state_after =
            metal.snapshot_linear_attn_state(&cpu_normed_state)?;

        let (normed_max_abs, normed_mean_abs) =
            diff_stats_same_len(reference_normed.data(), &resident_normed)?;
        let (attn_max_abs, attn_mean_abs) =
            diff_stats_same_len(reference_attn.data(), &resident_attn)?;
        let (attn_cpu_normed_max_abs, attn_cpu_normed_mean_abs) =
            diff_stats_same_len(reference_attn.data(), &resident_attn_cpu_normed)?;
        let state_diff =
            metal.diff_linear_attn_states(&reference_state_after, &resident_state_after)?;
        let state_cpu_normed_diff = metal
            .diff_linear_attn_states(&reference_state_after, &resident_cpu_normed_state_after)?;
        Ok(Some(ResidentLinearXrayProbe {
            layer_index,
            normed_max_abs,
            normed_mean_abs,
            attn_max_abs,
            attn_mean_abs,
            attn_cpu_normed_max_abs,
            attn_cpu_normed_mean_abs,
            init_state_conv_max_abs: init_state_diff.conv_max_abs,
            init_state_conv_mean_abs: init_state_diff.conv_mean_abs,
            init_state_ssm_max_abs: init_state_diff.ssm_max_abs,
            init_state_ssm_mean_abs: init_state_diff.ssm_mean_abs,
            state_conv_max_abs: state_diff.conv_max_abs,
            state_conv_mean_abs: state_diff.conv_mean_abs,
            state_ssm_max_abs: state_diff.ssm_max_abs,
            state_ssm_mean_abs: state_diff.ssm_mean_abs,
            state_cpu_normed_conv_max_abs: state_cpu_normed_diff.conv_max_abs,
            state_cpu_normed_conv_mean_abs: state_cpu_normed_diff.conv_mean_abs,
            state_cpu_normed_ssm_max_abs: state_cpu_normed_diff.ssm_max_abs,
            state_cpu_normed_ssm_mean_abs: state_cpu_normed_diff.ssm_mean_abs,
        }))
    }

    fn verify_mtp_drafts_batched(&self, verify: MtpVerify<'_>) -> Result<MtpVerifyResult> {
        let MtpVerify {
            cache,
            final_state,
            pending,
            generated,
            drafts,
            draft_distributions,
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
                accepted_history: Vec::new(),
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
        let mut accepted_history = Vec::new();

        for (position, draft) in drafts.into_iter().enumerate() {
            let target_state = Tensor::row(target_states.row_slice(position)?.to_vec())?;
            let (target, accepted_now) = if options.temperature > f32::EPSILON {
                let target_distribution =
                    self.token_distribution_from_state(&target_state, options)?;
                let draft_distribution = draft_distributions
                    .get(position)
                    .and_then(Option::as_ref)
                    .ok_or_else(|| {
                        InferError::Config(
                            "sampling MTP temp>0 sans distribution draft".to_string(),
                        )
                    })?;
                let sample = lossless_speculative_sample(
                    &target_distribution,
                    draft_distribution,
                    draft,
                    sampler,
                )?;
                (sample.token, sample.accepted)
            } else {
                let target = match target_tokens
                    .as_ref()
                    .and_then(|tokens| tokens.get(position))
                {
                    Some(token) => *token,
                    None => self.sample_token_from_state(&target_state, options, sampler)?,
                };
                (target, draft == target)
            };
            self.trace_mtp_verify_decision(position, draft, target, accepted_now, generated.len())?;
            stats.verifications += 1;
            stats.proposed += 1;
            stats.proposed_by_position[position] += 1;
            if options.stop_token_ids.contains(&target) {
                if accepted_now {
                    stats.accepted += 1;
                    stats.accepted_by_position[position] += 1;
                } else {
                    stats.rejected += 1;
                }
                *pending = target;
                *final_state = target_state;
                accepted_all = false;
                if position + 1 < verify_tokens.len() {
                    let replay_len = position + 1;
                    #[cfg(all(target_os = "macos", feature = "metal"))]
                    if let Some(captures) = resident_captures.as_ref() {
                        self.restore_mtp_resident_progress(cache, captures, replay_len)?;
                    } else {
                        self.restore_mtp_snapshot_replay(
                            cache,
                            &mut snapshot,
                            &verify_tokens,
                            replay_len,
                            final_state,
                        )?;
                    }
                    #[cfg(not(all(target_os = "macos", feature = "metal")))]
                    {
                        self.restore_mtp_snapshot_replay(
                            cache,
                            &mut snapshot,
                            &verify_tokens,
                            replay_len,
                            final_state,
                        )?;
                    }
                }
                break;
            }
            if accepted_now {
                stats.accepted += 1;
                stats.accepted_by_position[position] += 1;
                accepted_history.push((target_state.clone(), draft));
                generated.push(draft);
                *final_state = target_state;
                verify_token = draft;
                if generated.len() >= max_new_tokens {
                    accepted_all = false;
                    if position + 1 < verify_tokens.len() {
                        let replay_len = position + 1;
                        #[cfg(all(target_os = "macos", feature = "metal"))]
                        if let Some(captures) = resident_captures.as_ref() {
                            self.restore_mtp_resident_progress(cache, captures, replay_len)?;
                        } else {
                            self.restore_mtp_snapshot_replay(
                                cache,
                                &mut snapshot,
                                &verify_tokens,
                                replay_len,
                                final_state,
                            )?;
                        }
                        #[cfg(not(all(target_os = "macos", feature = "metal")))]
                        {
                            self.restore_mtp_snapshot_replay(
                                cache,
                                &mut snapshot,
                                &verify_tokens,
                                replay_len,
                                final_state,
                            )?;
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
                        self.restore_mtp_snapshot_replay(
                            cache,
                            &mut snapshot,
                            &verify_tokens,
                            replay_len,
                            final_state,
                        )?;
                    }
                    #[cfg(not(all(target_os = "macos", feature = "metal")))]
                    {
                        self.restore_mtp_snapshot_replay(
                            cache,
                            &mut snapshot,
                            &verify_tokens,
                            replay_len,
                            final_state,
                        )?;
                    }
                }
                break;
            }
        }

        Ok(MtpVerifyResult {
            accepted_all,
            verify_token,
            accepted_history,
        })
    }

    #[cfg(feature = "devtools")]
    fn verify_dflash_drafts_batched(&self, verify: DFlashVerify<'_>) -> Result<MtpVerifyResult> {
        let DFlashVerify {
            draft,
            cache,
            projected_context,
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
                accepted_history: Vec::new(),
            });
        }

        let snapshot = self.snapshot_speculative_cache(cache)?;
        let mut verify_tokens = Vec::with_capacity(drafts.len());
        verify_tokens.push(primary);
        verify_tokens.extend(drafts.iter().take(drafts.len().saturating_sub(1)).copied());
        let (target_states, target_hidden) = self.next_final_states_batched_with_layer_capture(
            cache,
            &verify_tokens,
            &draft.info.target_layer_ids,
        )?;
        let mut accepted_all = true;
        let mut verify_token = primary;
        let mut committed_len = 0usize;

        for (position, draft_token) in drafts.into_iter().enumerate() {
            let target_state = Tensor::row(target_states.row_slice(position)?.to_vec())?;
            let target = self.sample_token_from_state(&target_state, options, sampler)?;
            stats.verifications += 1;
            stats.proposed += 1;
            if let Some(slot) = stats.proposed_by_position.get_mut(position) {
                *slot += 1;
            }
            committed_len = position + 1;
            if draft_token == target {
                stats.accepted += 1;
                if let Some(slot) = stats.accepted_by_position.get_mut(position) {
                    *slot += 1;
                }
                generated.push(draft_token);
                *final_state = target_state;
                verify_token = draft_token;
                if generated.len() >= max_new_tokens
                    || options.stop_token_ids.contains(&draft_token)
                {
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

        if !accepted_all && committed_len < verify_tokens.len() {
            self.restore_speculative_cache(cache, snapshot)?;
            let replay_tokens = &verify_tokens[..committed_len];
            let (replay_states, replay_hidden) = self
                .next_final_states_batched_with_layer_capture(
                    cache,
                    replay_tokens,
                    &draft.info.target_layer_ids,
                )?;
            *final_state = Tensor::row(
                replay_states
                    .row_slice(committed_len.saturating_sub(1))?
                    .to_vec(),
            )?;
            self.append_dflash_projected_context(draft, projected_context, &replay_hidden)?;
        } else {
            let committed_hidden = take_tensor_rows(&target_hidden, committed_len)?;
            self.append_dflash_projected_context(draft, projected_context, &committed_hidden)?;
        }

        Ok(MtpVerifyResult {
            accepted_all,
            verify_token,
            accepted_history: Vec::new(),
        })
    }

    #[cfg(feature = "devtools")]
    fn dflash_draft_greedy_tokens(
        &self,
        draft: &DFlashDraft,
        projected_context: &Tensor,
        primary: usize,
        max_draft_tokens: usize,
        options: &GenerationOptions,
        sampler: &mut DeterministicSampler,
    ) -> Result<Vec<usize>> {
        let block_len = max_draft_tokens
            .checked_add(1)
            .ok_or_else(|| InferError::Shape("DFlash block_len déborde".to_string()))?;
        let mut noise_tokens = Vec::with_capacity(block_len);
        noise_tokens.push(primary);
        noise_tokens.resize(block_len, draft.info.mask_token_id);
        let noise_embedding = self.embed_scaled(&noise_tokens)?;
        let draft_states = draft.forward_projected_context(
            &noise_embedding,
            projected_context,
            self.forward_runtime(),
        )?;
        let (rows, hidden) = draft_states.as_matrix()?;
        if rows != block_len || hidden != self.final_norm.len() {
            return Err(InferError::Dimension(format!(
                "DFlash states {:?}, attendu [{block_len},{}]",
                draft_states.shape(),
                self.final_norm.len()
            )));
        }
        let mut tokens = Vec::with_capacity(max_draft_tokens);
        // DFlash-MLX lit `draft_hidden[:, 1:, :]`: le premier rang porte le
        // `staged_first`, les propositions commencent sur les rangs masque.
        for row in 1..rows {
            let state = Tensor::row(draft_states.row_slice(row)?.to_vec())?;
            tokens.push(self.sample_token_from_state(&state, options, sampler)?);
        }
        Ok(tokens)
    }

    #[cfg(feature = "devtools")]
    fn append_dflash_projected_context(
        &self,
        draft: &DFlashDraft,
        projected_context: &mut Tensor,
        target_hidden: &Tensor,
    ) -> Result<()> {
        let projected = draft.project_target_hidden(target_hidden, self.forward_runtime())?;
        *projected_context = append_tensor_rows(projected_context, &projected)?;
        Ok(())
    }

    #[cfg(feature = "devtools")]
    fn validate_dflash_draft(&self, draft: &DFlashDraft) -> Result<()> {
        let hidden = self.final_norm.len();
        if draft.info.hidden_size != hidden {
            return Err(InferError::Dimension(format!(
                "DFlash hidden={} incompatible avec trunk hidden={hidden}",
                draft.info.hidden_size
            )));
        }
        validate_capture_layer_ids(&draft.info.target_layer_ids, self.layers.len())
    }

    #[cfg(feature = "devtools")]
    fn prefill_cache_state_with_layer_capture(
        &self,
        prompt: &[usize],
        layer_ids: &[usize],
    ) -> Result<(CausalDecoderCache, Tensor, Tensor)> {
        if prompt.is_empty() {
            return Err(InferError::Dimension("prompt token vide".to_string()));
        }
        validate_capture_layer_ids(layer_ids, self.layers.len())?;
        let runtime = self.forward_runtime();
        let mut cache = self.empty_cache();
        let mut hidden = self.embed_scaled(prompt)?;
        let (seq, hidden_dim) = hidden.as_matrix()?;
        let mut captures = vec![None; layer_ids.len()];
        for (layer_index, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward_prefill(
                &self.config,
                &hidden,
                &mut cache.layers[layer_index],
                0,
                runtime,
            )?;
            capture_layer_output(layer_ids, layer_index, &hidden, &mut captures);
        }
        cache.position = prompt.len();
        let final_hidden = Tensor::row(hidden.last_row()?.to_vec())?;
        let final_state = rms_norm(&final_hidden, &self.final_norm, self.config.rms_eps)?;
        let target_hidden = concat_layer_captures(captures, seq, hidden_dim)?;
        Ok((cache, final_state, target_hidden))
    }

    #[cfg(feature = "devtools")]
    fn next_final_state_with_layer_capture(
        &self,
        cache: &mut CausalDecoderCache,
        token_id: usize,
        layer_ids: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        let (states, target_hidden) =
            self.next_final_states_batched_with_layer_capture(cache, &[token_id], layer_ids)?;
        Ok((Tensor::row(states.row_slice(0)?.to_vec())?, target_hidden))
    }

    #[cfg(feature = "devtools")]
    fn next_final_states_batched_with_layer_capture(
        &self,
        cache: &mut CausalDecoderCache,
        token_ids: &[usize],
        layer_ids: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        if token_ids.is_empty() {
            return Err(InferError::Dimension(
                "capture hidden batch vide".to_string(),
            ));
        }
        validate_capture_layer_ids(layer_ids, self.layers.len())?;
        if cache.layers.len() != self.layers.len() {
            return Err(InferError::Dimension(format!(
                "cache couches={} incompatible avec décodeur couches={}",
                cache.layers.len(),
                self.layers.len()
            )));
        }
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if let Some(output) =
            self.next_final_states_resident_verify(cache, token_ids, false, false, Some(layer_ids))?
        {
            if let Some(target_hidden) = output.target_hidden {
                return Ok((output.states, target_hidden));
            }
        }
        let position_offset = cache.position;
        let runtime = self.forward_runtime();
        let mut hidden = self.embed_scaled(token_ids)?;
        let (seq, hidden_dim) = hidden.as_matrix()?;
        let mut captures = vec![None; layer_ids.len()];
        for (layer_index, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward_prefill(
                &self.config,
                &hidden,
                &mut cache.layers[layer_index],
                position_offset,
                runtime,
            )?;
            capture_layer_output(layer_ids, layer_index, &hidden, &mut captures);
        }
        cache.position += token_ids.len();
        let final_states = rms_norm(&hidden, &self.final_norm, self.config.rms_eps)?;
        let target_hidden = concat_layer_captures(captures, seq, hidden_dim)?;
        Ok((final_states, target_hidden))
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
        self.trace_mtp_boundary(hidden, &post, next_id, position)?;
        let token = self.sample_mtp_token_from_state(&post, options, sampler)?;
        Ok((token, post))
    }

    #[cfg(feature = "devtools")]
    fn trace_mtp_boundary(
        &self,
        input_hidden: &Tensor,
        draft_hidden: &Tensor,
        input_token: usize,
        position: usize,
    ) -> Result<()> {
        let Some(path) = std::env::var_os("RETI_RUST_MTP_TRACE") else {
            return Ok(());
        };
        let limit = std::env::var("RETI_RUST_MTP_TRACE_LIMIT")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(4);
        if position >= limit {
            return Ok(());
        }
        let input_row = input_hidden.as_row()?;
        let draft_row = draft_hidden.as_row()?;
        let target_logits = self.logits_from_linear_state(draft_hidden, &self.lm_head)?;
        let draft_logits = self.logits_from_mtp_draft_state(draft_hidden)?;
        let target_top = top_logits(target_logits.as_row()?, 8);
        let draft_top = top_logits(draft_logits.as_row()?, 8);
        let line = format!(
            "{{\"position\":{position},\"input_token\":{input_token},\
             \"has_draft_lm_head\":{},\"input_hidden_l2\":{},\
             \"draft_hidden_l2\":{},\"input_hidden_first8\":{},\
             \"draft_hidden_first8\":{},\"target_lm_head_top8\":{},\
             \"draft_lm_head_top8\":{}}}\n",
            self.mtp_draft_lm_head.is_some(),
            l2_norm(input_row),
            l2_norm(draft_row),
            json_f32_prefix(input_row, 8),
            json_f32_prefix(draft_row, 8),
            json_top_logits(&target_top),
            json_top_logits(&draft_top)
        );
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(std::path::PathBuf::from(path))
            .map_err(|source| InferError::Config(format!("ouverture trace MTP: {source}")))?;
        file.write_all(line.as_bytes())
            .map_err(|source| InferError::Config(format!("écriture trace MTP: {source}")))
    }

    #[cfg(not(feature = "devtools"))]
    fn trace_mtp_boundary(
        &self,
        _input_hidden: &Tensor,
        _draft_hidden: &Tensor,
        _input_token: usize,
        _position: usize,
    ) -> Result<()> {
        Ok(())
    }

    #[cfg(feature = "devtools")]
    fn trace_mtp_verify_decision(
        &self,
        position: usize,
        draft: usize,
        target: usize,
        accepted: bool,
        generated_len_before: usize,
    ) -> Result<()> {
        let Some(path) = std::env::var_os("RETI_RUST_MTP_VERIFY_TRACE") else {
            return Ok(());
        };
        let line = format!(
            "{{\"position\":{position},\"draft\":{draft},\"target\":{target},\
             \"accepted\":{accepted},\"generated_len_before\":{generated_len_before}}}\n"
        );
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(std::path::PathBuf::from(path))
            .map_err(|source| {
                InferError::Config(format!("ouverture trace verify MTP: {source}"))
            })?;
        file.write_all(line.as_bytes())
            .map_err(|source| InferError::Config(format!("écriture trace verify MTP: {source}")))
    }

    #[cfg(not(feature = "devtools"))]
    fn trace_mtp_verify_decision(
        &self,
        _position: usize,
        _draft: usize,
        _target: usize,
        _accepted: bool,
        _generated_len_before: usize,
    ) -> Result<()> {
        Ok(())
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
        let embedding = self.embed_scaled(&[next_id])?;
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

    /// Préremplit le cache sans consulter ni alimenter le prefix cache.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le prompt est vide ou si un passage forward échoue.
    pub fn prefill_cache_uncached(&self, prompt: &[usize]) -> Result<(CausalDecoderCache, Tensor)> {
        let (cache, final_state) = self.prefill_cache_state_uncached(prompt)?;
        let logits = self.logits_from_final_state(&final_state)?;
        Ok((cache, logits))
    }

    /// Pré-remplit un prompt sans consulter le cache interne du décodeur.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le prompt est vide ou si le forward échoue.
    pub fn prefill_prompt_state_uncached(
        &self,
        prompt: &[usize],
    ) -> Result<CausalDecoderPromptState> {
        let (cache, final_state) = self.prefill_cache_state_uncached(prompt)?;
        Ok(CausalDecoderPromptState::new(cache, final_state))
    }

    /// Prolonge un état de prompt avec un suffixe tokenisé.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si l'état ou le forward du suffixe échoue.
    pub fn extend_prompt_state(
        &self,
        state: &mut CausalDecoderPromptState,
        suffix: &[usize],
    ) -> Result<()> {
        if suffix.is_empty() {
            return Ok(());
        }
        let final_states = self.next_final_states_batched(&mut state.cache, suffix)?;
        state.final_state = Tensor::row(final_states.last_row()?.to_vec())?;
        Ok(())
    }

    /// Capture les états Metal attachés à un état de prompt.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si la copie GPU du snapshot échoue.
    pub fn snapshot_prompt_state_metal(
        &self,
        state: &CausalDecoderPromptState,
    ) -> Result<CausalDecoderPromptMetalSnapshot> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            Ok(CausalDecoderPromptMetalSnapshot {
                linear: self.snapshot_prefix_cache_linear_metal(&state.cache)?,
            })
        }
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (self, state);
            Ok(CausalDecoderPromptMetalSnapshot::default())
        }
    }

    /// Copie un snapshot Metal pour le restaurer sans aliaser l'entrée cache.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si la copie GPU du snapshot échoue.
    pub fn copy_prompt_state_metal_snapshot(
        &self,
        snapshot: &CausalDecoderPromptMetalSnapshot,
    ) -> Result<CausalDecoderPromptMetalSnapshot> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            Ok(CausalDecoderPromptMetalSnapshot {
                linear: self.copy_prefix_cache_linear_metal(&snapshot.linear)?,
            })
        }
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (self, snapshot);
            Ok(CausalDecoderPromptMetalSnapshot::default())
        }
    }

    /// Restaure les états Metal dans un état de prompt cloné.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le nombre de couches diverge ou si le blit échoue.
    pub fn restore_prompt_state_metal(
        &self,
        state: &mut CausalDecoderPromptState,
        snapshot: CausalDecoderPromptMetalSnapshot,
    ) -> Result<()> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            self.restore_prefix_cache_linear_metal(&mut state.cache, snapshot.linear)
        }
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (self, state, snapshot);
            Ok(())
        }
    }

    pub(super) fn prefill_cache_state(
        &self,
        prompt: &[usize],
    ) -> Result<(CausalDecoderCache, Tensor)> {
        if prompt.is_empty() {
            return Err(InferError::Dimension("prompt token vide".to_string()));
        }
        if prefix_cache_enabled() {
            if let Some(hit) = self.prefix_cache_get(prompt)? {
                return Ok(hit);
            }
        }
        let state = self.prefill_cache_state_uncached(prompt)?;
        if prefix_cache_enabled() {
            self.prefix_cache_put(prompt, &state)?;
        }
        Ok(state)
    }

    fn prefill_cache_state_uncached(
        &self,
        prompt: &[usize],
    ) -> Result<(CausalDecoderCache, Tensor)> {
        if self.can_prefill_batched() && prefill_batched_enabled() {
            self.prefill_cache_state_batched(prompt)
        } else {
            self.prefill_cache_state_tokenwise(prompt)
        }
    }

    fn prefill_cache_state_with_mtp_history(
        &self,
        prompt: &[usize],
    ) -> Result<(CausalDecoderCache, Tensor, LayerKvCache)> {
        if prompt.is_empty() {
            return Err(InferError::Dimension("prompt token vide".to_string()));
        }
        let runtime = self.forward_runtime();
        let mut cache = self.empty_cache();
        let mut hidden = self.embed_scaled(prompt)?;
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
        let final_states = rms_norm(&hidden, &self.final_norm, self.config.rms_eps)?;
        let final_state = Tensor::row(final_states.last_row()?.to_vec())?;
        let mut mtp_history = LayerKvCache::default();
        for index in 0..prompt.len().saturating_sub(1) {
            let state = Tensor::row(final_states.row_slice(index)?.to_vec())?;
            let token_id = prompt[index + 1];
            let position = mtp_history.len();
            let _ = self.mtp_forward_post_hidden(&state, token_id, position, &mut mtp_history)?;
        }
        Ok((cache, final_state, mtp_history))
    }

    fn prefix_cache_get(&self, prompt: &[usize]) -> Result<Option<(CausalDecoderCache, Tensor)>> {
        let mut cache = self
            .prefix_cache
            .lock()
            .map_err(|_| InferError::Config("cache préfixe empoisonné".to_string()))?;
        let Some(index) = longest_prefix_entry_index(&cache.entries, prompt) else {
            return Ok(None);
        };
        let entry = cache.entries.remove(index);
        let cached_len = entry.tokens.len();
        let mut output = (entry.cache.clone(), entry.final_state.clone());
        #[cfg(all(target_os = "macos", feature = "metal"))]
        let linear_metal_entry = entry.linear_metal.clone();
        cache.entries.insert(0, entry);
        drop(cache);

        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let linear_metal = self.copy_prefix_cache_linear_metal(&linear_metal_entry)?;
            self.restore_prefix_cache_linear_metal(&mut output.0, linear_metal)?;
        }

        if cached_len < prompt.len() {
            self.extend_prefix_cache_suffix(&mut output, &prompt[cached_len..])?;
            self.prefix_cache_put(prompt, &output)?;
        }
        Ok(Some(output))
    }

    fn extend_prefix_cache_suffix(
        &self,
        state: &mut (CausalDecoderCache, Tensor),
        suffix: &[usize],
    ) -> Result<()> {
        if suffix.is_empty() {
            return Ok(());
        }
        let final_states = self.next_final_states_batched(&mut state.0, suffix)?;
        state.1 = Tensor::row(final_states.last_row()?.to_vec())?;
        Ok(())
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn snapshot_prefix_cache_linear_metal(
        &self,
        cache: &CausalDecoderCache,
    ) -> Result<Vec<Option<crate::metal_backend::LinearAttentionMetalState>>> {
        let states = cache
            .layers
            .iter()
            .map(|layer| layer.linear.metal_state())
            .collect::<Vec<_>>();
        if states.iter().all(Option::is_none) {
            return Ok(Vec::new());
        }
        let Some(metal) = self.forward_runtime().metal_executor() else {
            return Err(InferError::Config(
                "prefix-cache Metal sans executor Metal".to_string(),
            ));
        };
        metal.snapshot_linear_attn_states(&states)
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn copy_prefix_cache_linear_metal(
        &self,
        states: &[Option<crate::metal_backend::LinearAttentionMetalState>],
    ) -> Result<Vec<Option<crate::metal_backend::LinearAttentionMetalState>>> {
        if states.is_empty() {
            return Ok(Vec::new());
        }
        let Some(metal) = self.forward_runtime().metal_executor() else {
            return Err(InferError::Config(
                "prefix-cache Metal sans executor Metal".to_string(),
            ));
        };
        let state_refs = states.iter().map(Option::as_ref).collect::<Vec<_>>();
        metal.snapshot_linear_attn_states(&state_refs)
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn restore_prefix_cache_linear_metal(
        &self,
        cache: &mut CausalDecoderCache,
        snapshots: Vec<Option<crate::metal_backend::LinearAttentionMetalState>>,
    ) -> Result<()> {
        if snapshots.is_empty() {
            return Ok(());
        }
        if snapshots.len() != cache.layers.len() {
            return Err(InferError::Dimension(format!(
                "prefix-cache Metal: snapshots={} couches={}",
                snapshots.len(),
                cache.layers.len()
            )));
        }
        let Some(metal) = self.forward_runtime().metal_executor() else {
            return Err(InferError::Config(
                "prefix-cache Metal sans executor Metal".to_string(),
            ));
        };
        for (layer, snapshot) in cache.layers.iter_mut().zip(snapshots.into_iter()) {
            layer.linear.restore_metal_state_snapshot(metal, snapshot)?;
        }
        Ok(())
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
        #[cfg(all(target_os = "macos", feature = "metal"))]
        let linear_metal = self.snapshot_prefix_cache_linear_metal(&state.0)?;
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
                #[cfg(all(target_os = "macos", feature = "metal"))]
                linear_metal,
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
        let chunk_size = prefill_chunk_size();
        if chunk_size > 0 && chunk_size < prompt.len() {
            return self.prefill_cache_state_batched_chunked(prompt, chunk_size);
        }
        let runtime = self.forward_runtime();
        let mut cache = self.empty_cache();
        let mut hidden = self.embed_scaled(prompt)?;
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

    fn prefill_cache_state_batched_chunked(
        &self,
        prompt: &[usize],
        chunk_size: usize,
    ) -> Result<(CausalDecoderCache, Tensor)> {
        if chunk_size == 0 {
            return Err(InferError::Dimension(
                "taille de chunk prefill nulle".to_string(),
            ));
        }
        let runtime = self.forward_runtime();
        let mut cache = self.empty_cache();
        let mut last_hidden = None;
        for chunk in prompt.chunks(chunk_size) {
            let position_offset = cache.position;
            let mut hidden = self.embed_scaled(chunk)?;
            for (layer_index, layer) in self.layers.iter().enumerate() {
                hidden = layer.forward_prefill(
                    &self.config,
                    &hidden,
                    &mut cache.layers[layer_index],
                    position_offset,
                    runtime,
                )?;
            }
            cache.position += chunk.len();
            last_hidden = Some(hidden);
        }
        let hidden =
            last_hidden.ok_or_else(|| InferError::Dimension("prompt token vide".to_string()))?;
        let final_hidden = Tensor::row(hidden.last_row()?.to_vec())?;
        let final_state = rms_norm(&final_hidden, &self.final_norm, self.config.rms_eps)?;
        Ok((cache, final_state))
    }

    #[cfg(test)]
    pub(crate) fn prefill_cache_state_batched_for_test(
        &self,
        prompt: &[usize],
    ) -> Result<(CausalDecoderCache, Tensor)> {
        self.prefill_cache_state_batched(prompt)
    }

    #[cfg(test)]
    pub(crate) fn prefill_cache_state_batched_chunked_for_test(
        &self,
        prompt: &[usize],
        chunk_size: usize,
    ) -> Result<(CausalDecoderCache, Tensor)> {
        self.prefill_cache_state_batched_chunked(prompt, chunk_size)
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn prefill_cache_state_metal_resident(
        &self,
        prompt: &[usize],
    ) -> Result<Option<(CausalDecoderCache, Tensor)>> {
        if !prefill_resident_enabled() {
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
        // Le kernel prefill (rms_norm+RoPE fusionné) n'implémente que le
        // rotate-half : tout autre appariement retombe sur le chemin CPU.
        if self.config.rope_style != RopeStyle::Halves {
            return Ok(None);
        }
        // Le prefill résident applique une base RoPE unique, des positions brutes
        // et aucun masque fenêtré : exclu dès qu'une couche surcharge sa base, son
        // échelle de positions ou sa fenêtre (Gemma 3).
        let has_layer_overrides = self.layers.iter().any(|layer| match &layer.attention {
            AttentionBlock::Full(attention) => {
                attention.rope_theta.is_some()
                    || attention.rope_position_scale.is_some()
                    || attention.sliding_window.is_some()
            }
            AttentionBlock::Linear(_) => false,
        });
        if has_layer_overrides {
            return Ok(None);
        }
        let hidden = self.embed_scaled(prompt)?;
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
            let Some(prefill_layer) = layer.prefill_moe_layer(&self.config) else {
                return Ok(None);
            };
            layers.push(prefill_layer);
        }
        match metal.qwen_moe_prefill_resident(&hidden, &layers, spec) {
            Ok((final_hidden, layer_states)) => {
                if layer_states.len() != self.layers.len() {
                    return Err(InferError::Dimension(format!(
                        "prefill résident états couches={} attendu={}",
                        layer_states.len(),
                        self.layers.len()
                    )));
                }
                let layout = AttentionLayout {
                    num_attention_heads: self.config.num_attention_heads,
                    num_key_value_heads: self.config.num_key_value_heads,
                    head_dim,
                    rope_dims: self.config.rope_dims.unwrap_or(head_dim),
                    attn_scalar: self.config.query_pre_attn_scalar.unwrap_or(head_dim as f32),
                    sliding_window: None,
                };
                let mut cache = self.empty_cache();
                for (layer_cache, layer_state) in
                    cache.layers.iter_mut().zip(layer_states.into_iter())
                {
                    match layer_state {
                        crate::metal_backend::PrefillResidentLayerCache::Full { key, value } => {
                            layer_cache.append_batch(&key, &value, &layout)?;
                        }
                        crate::metal_backend::PrefillResidentLayerCache::Linear { state } => {
                            layer_cache
                                .linear
                                .restore_metal_state_snapshot(metal, Some(state))?;
                        }
                    }
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
        let linear_enabled = prefill_linear_batched_enabled();
        self.layers
            .iter()
            .all(|layer| layer.supports_batched_prefill(linear_enabled))
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
            if generated_matches_stop_sequence(&generated, &options.stop_sequences) {
                break;
            }
        }
        Ok(generated)
    }
}

fn generated_matches_stop_sequence(generated: &[usize], stop_sequences: &[Vec<usize>]) -> bool {
    stop_sequences
        .iter()
        .any(|sequence| !sequence.is_empty() && generated.ends_with(sequence))
}

#[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
fn diff_stats_same_len(left: &[f32], right: &[f32]) -> Result<(f32, f32)> {
    if left.len() != right.len() {
        return Err(InferError::Dimension(format!(
            "diff len gauche={} droite={}",
            left.len(),
            right.len()
        )));
    }
    if left.is_empty() {
        return Ok((0.0, 0.0));
    }
    let mut max_abs = 0.0_f32;
    let mut sum_abs = 0.0_f32;
    for (a, b) in left.iter().zip(right) {
        let diff = (a - b).abs();
        max_abs = max_abs.max(diff);
        sum_abs += diff;
    }
    Ok((max_abs, sum_abs / left.len() as f32))
}

#[cfg(feature = "devtools")]
fn validate_capture_layer_ids(layer_ids: &[usize], layer_count: usize) -> Result<()> {
    if layer_ids.is_empty() {
        return Err(InferError::Config(
            "capture hidden sans couche cible".to_string(),
        ));
    }
    for (offset, layer_id) in layer_ids.iter().copied().enumerate() {
        if layer_id >= layer_count {
            return Err(InferError::Dimension(format!(
                "capture hidden couche {layer_id} hors décodeur layers={layer_count}"
            )));
        }
        if layer_ids[..offset].contains(&layer_id) {
            return Err(InferError::Config(format!(
                "capture hidden couche dupliquée: {layer_id}"
            )));
        }
    }
    Ok(())
}

#[cfg(feature = "devtools")]
fn capture_layer_output(
    layer_ids: &[usize],
    layer_index: usize,
    hidden: &Tensor,
    captures: &mut [Option<Tensor>],
) {
    for (slot, target) in layer_ids.iter().copied().enumerate() {
        if target == layer_index {
            captures[slot] = Some(hidden.clone());
        }
    }
}

#[cfg(feature = "devtools")]
fn l2_norm(values: &[f32]) -> f32 {
    values.iter().map(|value| value * value).sum::<f32>().sqrt()
}

#[cfg(feature = "devtools")]
fn json_f32_prefix(values: &[f32], limit: usize) -> String {
    let items = values
        .iter()
        .take(limit)
        .map(|value| format!("{value:.8}"))
        .collect::<Vec<_>>();
    format!("[{}]", items.join(","))
}

#[cfg(feature = "devtools")]
fn top_logits(values: &[f32], limit: usize) -> Vec<(usize, f32)> {
    let mut top = Vec::with_capacity(limit.min(values.len()));
    for (index, value) in values.iter().copied().enumerate() {
        if top.len() < limit {
            top.push((index, value));
            sort_top_logits(&mut top);
        } else if let Some((_, last_value)) = top.last().copied() {
            if value > last_value {
                let last = top.len() - 1;
                top[last] = (index, value);
                sort_top_logits(&mut top);
            }
        }
    }
    top
}

#[cfg(feature = "devtools")]
fn sort_top_logits(values: &mut [(usize, f32)]) {
    values.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.0.cmp(&right.0))
    });
}

#[cfg(feature = "devtools")]
fn json_top_logits(values: &[(usize, f32)]) -> String {
    let items = values
        .iter()
        .map(|(index, value)| format!("{{\"token\":{index},\"logit\":{value:.8}}}"))
        .collect::<Vec<_>>();
    format!("[{}]", items.join(","))
}

#[cfg(feature = "devtools")]
fn concat_layer_captures(
    captures: Vec<Option<Tensor>>,
    expected_rows: usize,
    hidden_dim: usize,
) -> Result<Tensor> {
    let capture_count = captures.len();
    let output_cols = capture_count
        .checked_mul(hidden_dim)
        .ok_or_else(|| InferError::Shape("capture hidden colonnes débordent".to_string()))?;
    let mut out = vec![0.0_f32; expected_rows * output_cols];
    for (capture_index, capture) in captures.into_iter().enumerate() {
        let capture = capture.ok_or_else(|| {
            InferError::Config(format!("capture hidden manquante index={capture_index}"))
        })?;
        let (rows, cols) = capture.as_matrix()?;
        if rows != expected_rows || cols != hidden_dim {
            return Err(InferError::Dimension(format!(
                "capture hidden {:?}, attendu [{expected_rows},{hidden_dim}]",
                capture.shape()
            )));
        }
        for row in 0..expected_rows {
            let src_base = row * hidden_dim;
            let dst_base = row * output_cols + capture_index * hidden_dim;
            out[dst_base..dst_base + hidden_dim]
                .copy_from_slice(&capture.data()[src_base..src_base + hidden_dim]);
        }
    }
    Tensor::from_vec(vec![expected_rows, output_cols], out)
}

#[cfg(feature = "devtools")]
fn take_tensor_rows(tensor: &Tensor, rows: usize) -> Result<Tensor> {
    let (total_rows, cols) = tensor.as_matrix()?;
    if rows == 0 || rows > total_rows {
        return Err(InferError::Dimension(format!(
            "take rows={rows} invalide pour {:?}",
            tensor.shape()
        )));
    }
    let len = rows
        .checked_mul(cols)
        .ok_or_else(|| InferError::Shape("take rows déborde".to_string()))?;
    Tensor::from_vec(vec![rows, cols], tensor.data()[..len].to_vec())
}

#[cfg(feature = "devtools")]
fn append_tensor_rows(left: &Tensor, right: &Tensor) -> Result<Tensor> {
    let (_, left_cols) = left.as_matrix()?;
    let (_, right_cols) = right.as_matrix()?;
    if left_cols != right_cols {
        return Err(InferError::Dimension(format!(
            "append rows colonnes incompatibles left={:?} right={:?}",
            left.shape(),
            right.shape()
        )));
    }
    let rows = left.shape()[0]
        .checked_add(right.shape()[0])
        .ok_or_else(|| InferError::Shape("append rows déborde".to_string()))?;
    let mut data = Vec::with_capacity(left.data().len() + right.data().len());
    data.extend_from_slice(left.data());
    data.extend_from_slice(right.data());
    Tensor::from_vec(vec![rows, left_cols], data)
}
