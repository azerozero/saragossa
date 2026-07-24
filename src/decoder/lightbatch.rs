//! Génération light-batch : M flux LLM concurrents d'UNE conversation (E2.1).
//!
//! Baseline **time-slicing** : les flux sont décodés en alternance token-à-token
//! avec les kernels mono-flux existants — chaque flux possède son propre
//! [`CausalDecoderCache`] (KV par couche + état conv/ssm + arène résidente) et
//! son propre [`DeterministicSampler`]. Aucun état n'est partagé entre flux en
//! dehors des poids (lecture seule) → chaque flux est **byte-identique** au même
//! flux décodé seul (mêmes kernels, même état, même ordre par flux).
//!
//! Le slot de flux namespace le scratch label-keyed de l'exécuteur partagé
//! (E2.0) : deux arènes résidentes encodées sur le même thread ne s'aliasent
//! jamais, même quand leurs command buffers seront un jour in-flight ensemble.

use super::*;

/// Accumulateur de la disjonction d'experts à M=2
/// (`RETI_RUST_LIGHTBATCH_EXPERT_STATS=1`) : pour chaque (couche MoE, pas duo),
/// n(2) = nombre d'experts DISTINCTS dans l'union des top-k des 2 flux
/// (top_k ≤ n(2) ≤ 2·top_k). C'est LA donnée du plafond du gain MoE batché :
/// trafic routé dédupliqué idéal = n(2)/(2·top_k) du trafic par-flux actuel.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[derive(Default)]
struct ExpertStats {
    steps: u64,
    samples: u64,
    sum_distinct: u64,
    top_k: usize,
    /// `histogram[n]` = occurrences de n experts distincts (0 ≤ n ≤ 2·top_k).
    histogram: Vec<u64>,
    /// `(somme, compte)` par index de couche MoE (ordre d'encodage du pas duo).
    per_layer: Vec<(u64, u64)>,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn expert_stats() -> &'static Mutex<ExpertStats> {
    static STATS: OnceLock<Mutex<ExpertStats>> = OnceLock::new();
    STATS.get_or_init(|| Mutex::new(ExpertStats::default()))
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn reset_expert_stats() {
    if let Ok(mut stats) = expert_stats().lock() {
        *stats = ExpertStats::default();
    }
}

/// Enregistre les paires d'indices d'un pas duo (une entrée par couche MoE).
/// Diagnostic pur : toute erreur de lecture est ignorée (jamais d'impact sur la
/// génération).
#[cfg(all(target_os = "macos", feature = "metal"))]
pub(super) fn record_expert_stats(pairs: &[([metal::Buffer; 2], usize)]) {
    let Ok(mut stats) = expert_stats().lock() else {
        return;
    };
    stats.steps += 1;
    for (layer, ([first, second], top_k)) in pairs.iter().enumerate() {
        let (Ok(indices_a), Ok(indices_b)) = (
            crate::metal_backend::read_u32_buffer(first, *top_k),
            crate::metal_backend::read_u32_buffer(second, *top_k),
        ) else {
            continue;
        };
        let mut distinct: Vec<u32> = indices_a;
        for index in indices_b {
            if !distinct.contains(&index) {
                distinct.push(index);
            }
        }
        let n = distinct.len();
        stats.samples += 1;
        stats.sum_distinct += n as u64;
        stats.top_k = stats.top_k.max(*top_k);
        if stats.histogram.len() < 2 * *top_k + 1 {
            stats.histogram.resize(2 * *top_k + 1, 0);
        }
        if let Some(bucket) = stats.histogram.get_mut(n) {
            *bucket += 1;
        }
        if stats.per_layer.len() <= layer {
            stats.per_layer.resize(layer + 1, (0, 0));
        }
        stats.per_layer[layer].0 += n as u64;
        stats.per_layer[layer].1 += 1;
    }
}

/// Imprime la synthèse de la disjonction d'experts du run (puis remet à zéro).
#[cfg(all(target_os = "macos", feature = "metal"))]
fn report_expert_stats() {
    let Ok(stats) = expert_stats().lock() else {
        return;
    };
    if stats.samples == 0 {
        eprintln!(
            "lightbatch expert_stats: aucun échantillon (pas duo MoE non exercé — \
             vérifier RETI_RUST_LIGHTBATCH_MOE2 et le mode duo)"
        );
        return;
    }
    let mean = stats.sum_distinct as f64 / stats.samples as f64;
    let top_k = stats.top_k as f64;
    eprintln!(
        "lightbatch expert_stats: steps={} couches_moe={} top_k={} n2_moyen={:.2} \
         (distincts/couche, bornes [{}..{}]) overlap_moyen={:.2} \
         trafic_route_deduplique_ideal={:.3}x (vs 2x par-flux actuel)",
        stats.steps,
        stats.per_layer.len(),
        stats.top_k,
        mean,
        stats.top_k,
        2 * stats.top_k,
        2.0 * top_k - mean,
        mean / (2.0 * top_k),
    );
    let histogram = stats
        .histogram
        .iter()
        .enumerate()
        .skip(stats.top_k)
        .map(|(n, count)| format!("{n}:{count}"))
        .collect::<Vec<_>>()
        .join(" ");
    eprintln!("lightbatch expert_stats histogramme n2: {histogram}");
    let per_layer = stats
        .per_layer
        .iter()
        .enumerate()
        .map(|(layer, (sum, count))| {
            let layer_mean = if *count > 0 {
                *sum as f64 / *count as f64
            } else {
                0.0
            };
            format!("L{layer}={layer_mean:.1}")
        })
        .collect::<Vec<_>>()
        .join(" ");
    eprintln!("lightbatch expert_stats par couche: {per_layer}");
}

/// Progression d'UN flux du light-batch (bookkeeping du time-slicing).
///
/// Reproduit EXACTEMENT la sémantique de la boucle solo
/// (`generate_greedy_timed_with_options`) : push du token courant si non-stop,
/// puis decode du suivant tant que `step + 1 < max_new_tokens`.
struct LightBatchStream {
    /// Index du flux (0 = flux principal, convention E2.5).
    index: usize,
    /// Budget de génération PROPRE au flux (les flux d'un batch peuvent
    /// demander des longueurs différentes).
    max_new: usize,
    cache: CausalDecoderCache,
    sampler: DeterministicSampler,
    /// Prochain token d'entrée (déjà échantillonné, pas encore poussé).
    token: usize,
    generated: Vec<usize>,
    step: usize,
    done: bool,
    use_resident_full: bool,
    prefill: Duration,
    decode: Duration,
    decode_tokens: usize,
}

impl CausalDecoder {
    /// Génère M flux en alternance token-à-token (light-batch, baseline E2.1).
    ///
    /// Chaque prompt est pré-rempli séquentiellement puis les flux vivants sont
    /// décodés un token chacun par tour (time-slicing). Le decode passe par le
    /// chemin résident complet (1c) quand le modèle le supporte (slot scratch
    /// par flux), sinon par le per-op — comme la boucle solo. Les sorties sont
    /// rendues dans l'ordre des prompts, avec timings par flux.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si `prompts` est vide, si un prompt est vide ou si un
    /// forward échoue.
    pub fn generate_greedy_lightbatch_with_options(
        &self,
        prompts: &[Vec<usize>],
        max_new_tokens: usize,
        options: &GenerationOptions,
    ) -> Result<Vec<GenerationOutput>> {
        if max_new_tokens == 0 {
            return Ok(prompts
                .iter()
                .map(|_| GenerationOutput {
                    tokens: Vec::new(),
                    timings: GenerationTimings::default(),
                })
                .collect());
        }
        let maxes = vec![max_new_tokens; prompts.len()];
        self.generate_greedy_lightbatch_streaming_with_options(prompts, &maxes, options, |_, _| {})
    }

    /// Variante streaming du light-batch (E2.5) : `on_token(flux, token)` est
    /// appelé pour CHAQUE token émis, dans l'ordre d'émission de chaque flux —
    /// le flux 0 (principal, prioritaire) peut être consommé incrémentalement
    /// pendant que le flux de fond avance. `max_new_tokens` est PAR FLUX
    /// (chaque flux suit la sémantique solo avec SON budget). La priorité du
    /// principal se règle via `RETI_RUST_LIGHTBATCH_BG_STRIDE` (flux de fond
    /// décodé 1 pas sur N, défaut 1).
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si `prompts` est vide, si les longueurs `prompts` /
    /// `max_new_tokens` divergent, si un budget est nul, ou si un forward
    /// échoue.
    pub fn generate_greedy_lightbatch_streaming_with_options(
        &self,
        prompts: &[Vec<usize>],
        max_new_tokens: &[usize],
        options: &GenerationOptions,
        on_token: impl FnMut(usize, usize),
    ) -> Result<Vec<GenerationOutput>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        let stride = lightbatch_background_stride();
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        let stride = 1;
        self.lightbatch_generate(prompts, max_new_tokens, options, stride, on_token)
    }

    fn lightbatch_generate(
        &self,
        prompts: &[Vec<usize>],
        max_new_tokens: &[usize],
        options: &GenerationOptions,
        stride: u64,
        mut on_token: impl FnMut(usize, usize),
    ) -> Result<Vec<GenerationOutput>> {
        if prompts.is_empty() {
            return Err(InferError::Dimension("light-batch sans prompt".to_string()));
        }
        if prompts.len() != max_new_tokens.len() {
            return Err(InferError::Dimension(format!(
                "light-batch: {} prompts pour {} budgets",
                prompts.len(),
                max_new_tokens.len()
            )));
        }
        if max_new_tokens.contains(&0) {
            return Err(InferError::Dimension(
                "light-batch: budget de génération nul".to_string(),
            ));
        }
        let mut streams = Vec::with_capacity(prompts.len());
        for (slot, (prompt, max_new)) in prompts.iter().zip(max_new_tokens).enumerate() {
            streams.push(self.prefill_lightbatch_stream(prompt, *max_new, options, slot)?);
        }
        // Pas duo qmm2 (E2.2) : M=2 et modèle supporté → 2 flux dans UN command
        // buffer, projections denses batchées ; greedy (argmax) ou sampled
        // (E2.4, sampler GPU avec rng PAR FLUX — `use_resident_full` garantit
        // déjà greedy OU resident_sampling). Sinon time-slicing E2.1.
        #[cfg(all(target_os = "macos", feature = "metal"))]
        let duo_ok = streams.len() == 2
            && lightbatch_qmm2_enabled()
            && (options.temperature <= f32::EPSILON
                || super::resident::resident_sampling_on_device(options))
            && streams.iter().all(|stream| {
                stream.use_resident_full && self.supports_resident_duo(&stream.cache)
            });
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if trace_lightbatch_enabled() {
            eprintln!(
                "lightbatch: mode={} (flux={})",
                if duo_ok { "duo-qmm2" } else { "time-slicing" },
                streams.len()
            );
        }
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if lightbatch_expert_stats_enabled() {
            reset_expert_stats();
        }
        let mut round: u64 = 0;
        loop {
            let mut advanced = false;
            round = round.wrapping_add(1);
            // Priorité du principal (E2.5) : les flux de fond (index > 0) ne
            // décodent qu'un tour sur `stride` — sauf si le principal est fini.
            let background_turn = stride <= 1 || round % stride == 0 || streams[0].done;
            #[cfg(all(target_os = "macos", feature = "metal"))]
            if duo_ok {
                if !background_turn {
                    let will_decode =
                        self.resolve_lightbatch_push(&mut streams[0], options, &mut on_token);
                    if will_decode {
                        self.decode_lightbatch_step(&mut streams[0], options)?;
                    }
                    continue;
                }
                let will_decode = [
                    self.resolve_lightbatch_push(&mut streams[0], options, &mut on_token),
                    self.resolve_lightbatch_push(&mut streams[1], options, &mut on_token),
                ];
                if will_decode == [true, true] {
                    let decode_started = Instant::now();
                    let (left, right) = streams.split_at_mut(1);
                    let first = &mut left[0];
                    let second = &mut right[0];
                    // Sampled (E2.4) : un spec par flux depuis SON sampler,
                    // avancé d'un cran par token comme le solo.
                    let samples = if options.temperature > f32::EPSILON {
                        let specs = [
                            super::resident::resident_sample_spec(options, &first.sampler)?,
                            super::resident::resident_sample_spec(options, &second.sampler)?,
                        ];
                        first.sampler.advance();
                        second.sampler.advance();
                        Some(specs)
                    } else {
                        None
                    };
                    let tokens = self.decode_tokens_resident_duo(
                        &mut first.cache,
                        &mut second.cache,
                        [first.token, second.token],
                        samples,
                    )?;
                    let elapsed = decode_started.elapsed();
                    for (stream, token) in [first, second].into_iter().zip(tokens) {
                        stream.token = token;
                        stream.decode += elapsed;
                        stream.decode_tokens += 1;
                        stream.step += 1;
                    }
                    continue;
                }
                // Queue de batch : au plus un flux décode encore → pas solo.
                for (stream, decode) in streams.iter_mut().zip(will_decode) {
                    if decode {
                        advanced |= self.decode_lightbatch_step(stream, options)?;
                    }
                }
                if !advanced && streams.iter().all(|stream| stream.done) {
                    break;
                }
                continue;
            }
            for stream in &mut streams {
                if stream.index > 0 && !background_turn {
                    continue;
                }
                advanced |= self.advance_lightbatch_stream(stream, options, &mut on_token)?;
            }
            if !advanced && streams.iter().all(|stream| stream.done) {
                break;
            }
        }
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if lightbatch_expert_stats_enabled() {
            report_expert_stats();
        }
        Ok(streams
            .into_iter()
            .map(|stream| GenerationOutput {
                tokens: stream.generated,
                timings: GenerationTimings {
                    prefill: stream.prefill,
                    decode: stream.decode,
                    decode_tokens: stream.decode_tokens,
                },
            })
            .collect())
    }

    /// Pré-remplit UN flux et prépare son arène résidente (slot scratch dédié).
    fn prefill_lightbatch_stream(
        &self,
        prompt: &[usize],
        max_new_tokens: usize,
        options: &GenerationOptions,
        slot: usize,
    ) -> Result<LightBatchStream> {
        let prefill_started = Instant::now();
        let (mut cache, final_state) = self.prefill_cache_state(prompt)?;
        // Mêmes prédicats que la boucle solo : résident complet (1c) si supporté
        // (greedy ou sampler GPU), sinon résident 1b puis per-op.
        #[cfg(all(target_os = "macos", feature = "metal"))]
        let use_resident_full = {
            let resident_sampling = super::resident::resident_sampling_supported(options);
            let mut use_resident_full = decode_resident_full_enabled()
                && (options.temperature <= f32::EPSILON || resident_sampling)
                && self.supports_resident_full_decode();
            if use_resident_full {
                use_resident_full = self.setup_resident_full_decode_with_slot(
                    &mut cache,
                    max_new_tokens,
                    0,
                    u64::try_from(slot).unwrap_or(u64::MAX),
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
            use_resident_full
        };
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        let use_resident_full = {
            let _ = slot;
            false
        };
        let mut sampler = DeterministicSampler::new(options.seed);
        let token = self.sample_token_from_state(&final_state, options, &mut sampler)?;
        Ok(LightBatchStream {
            index: slot,
            max_new: max_new_tokens,
            cache,
            sampler,
            token,
            generated: Vec::with_capacity(max_new_tokens),
            step: 0,
            done: false,
            use_resident_full,
            prefill: prefill_started.elapsed(),
            decode: Duration::ZERO,
            decode_tokens: 0,
        })
    }

    /// Avance UN flux d'un pas (push du token courant + decode du suivant).
    ///
    /// Renvoie `true` si un decode a été encodé (le flux reste vivant).
    fn advance_lightbatch_stream(
        &self,
        stream: &mut LightBatchStream,
        options: &GenerationOptions,
        on_token: &mut impl FnMut(usize, usize),
    ) -> Result<bool> {
        if !self.resolve_lightbatch_push(stream, options, on_token) {
            return Ok(false);
        }
        self.decode_lightbatch_step(stream, options)
    }

    /// Résout le bookkeeping du tour pour UN flux (stop-check + push +
    /// émission streaming), sans décoder. Renvoie `true` si le flux doit
    /// décoder un token ce tour (sémantique strictement identique à la boucle
    /// solo, avec le budget PROPRE du flux).
    fn resolve_lightbatch_push(
        &self,
        stream: &mut LightBatchStream,
        options: &GenerationOptions,
        on_token: &mut impl FnMut(usize, usize),
    ) -> bool {
        if stream.done {
            return false;
        }
        if options.stop_token_ids.contains(&stream.token) {
            stream.done = true;
            return false;
        }
        stream.generated.push(stream.token);
        on_token(stream.index, stream.token);
        if stream.step + 1 >= stream.max_new {
            stream.done = true;
            return false;
        }
        true
    }

    /// Décode le prochain token d'UN flux (chemin solo) et met à jour ses
    /// compteurs. Renvoie `true` (le flux reste vivant).
    fn decode_lightbatch_step(
        &self,
        stream: &mut LightBatchStream,
        options: &GenerationOptions,
    ) -> Result<bool> {
        let decode_started = Instant::now();
        stream.token = self.decode_lightbatch_token(stream, options)?;
        stream.decode += decode_started.elapsed();
        stream.decode_tokens += 1;
        stream.step += 1;
        Ok(true)
    }

    /// Décode UN token du flux par le chemin solo correspondant (résident 1c ou
    /// per-op), sur l'état du flux uniquement.
    fn decode_lightbatch_token(
        &self,
        stream: &mut LightBatchStream,
        options: &GenerationOptions,
    ) -> Result<usize> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if stream.use_resident_full {
            return if options.temperature > f32::EPSILON {
                self.decode_token_resident_sampled(
                    &mut stream.cache,
                    stream.token,
                    options,
                    &mut stream.sampler,
                )
            } else {
                self.decode_token_resident(&mut stream.cache, stream.token)
            };
        }
        let state = self.next_decode_state(&mut stream.cache, stream.token)?;
        self.sample_token_from_state(&state, options, &mut stream.sampler)
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::gqa_weights;
    use super::*;

    fn tiny_model() -> CausalDecoder {
        let config = CausalDecoderConfig {
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: Some(2),
            rope_theta: Some(10_000.0),
            ..CausalDecoderConfig::default()
        };
        CausalDecoder::from_tensors(gqa_weights(), config).expect("invariant: modèle GQA valide")
    }

    #[test]
    fn lightbatch_rejects_empty_prompt_list() {
        let model = tiny_model();
        let err = model
            .generate_greedy_lightbatch_with_options(&[], 8, &GenerationOptions::default())
            .expect_err("invariant: light-batch vide rejeté");
        assert!(matches!(err, InferError::Dimension(_)));
    }

    #[test]
    fn lightbatch_zero_budget_returns_empty_streams() {
        let model = tiny_model();
        let outputs = model
            .generate_greedy_lightbatch_with_options(
                &[vec![0, 1], vec![1, 0]],
                0,
                &GenerationOptions::default(),
            )
            .expect("invariant: budget nul valide");
        assert_eq!(outputs.len(), 2);
        assert!(outputs.iter().all(|output| output.tokens.is_empty()));
    }

    /// Oracle byte-identité (bookkeeping) : chaque flux du batch == le même flux
    /// décodé SEUL par l'API prod, en greedy T=0 (mêmes prompts, même seed).
    #[test]
    fn lightbatch_streams_match_solo_greedy() {
        let model = tiny_model();
        let options = GenerationOptions::default();
        let max_new = 24;
        let prompt_a = vec![0_usize, 1];
        let prompt_b = vec![1_usize, 0, 1];

        let solo_a = model
            .generate_greedy_cached_with_options(&prompt_a, max_new, &options)
            .expect("invariant: solo A valide");
        let solo_b = model
            .generate_greedy_cached_with_options(&prompt_b, max_new, &options)
            .expect("invariant: solo B valide");
        let batch = model
            .generate_greedy_lightbatch_with_options(&[prompt_a, prompt_b], max_new, &options)
            .expect("invariant: light-batch valide");

        assert_eq!(batch[0].tokens, solo_a, "flux A divergent du solo");
        assert_eq!(batch[1].tokens, solo_b, "flux B divergent du solo");
    }

    /// Le stop-token interrompt chaque flux indépendamment, comme en solo.
    #[test]
    fn lightbatch_streams_stop_independently() {
        let model = tiny_model();
        let max_new = 24;
        let prompt_a = vec![0_usize, 1];
        let prompt_b = vec![1_usize, 0, 1];
        let baseline = model
            .generate_greedy_lightbatch_with_options(
                &[prompt_a.clone(), prompt_b.clone()],
                max_new,
                &GenerationOptions::default(),
            )
            .expect("invariant: baseline valide");
        // Stoppe sur le 3e token du flux A : A doit être tronqué AVANT ce token,
        // B doit suivre la même règle sur sa propre séquence.
        let stop = baseline[0].tokens[2];
        let options = GenerationOptions {
            stop_token_ids: vec![stop],
            ..GenerationOptions::default()
        };
        let solo_a = model
            .generate_greedy_cached_with_options(&prompt_a, max_new, &options)
            .expect("invariant: solo A stop valide");
        let solo_b = model
            .generate_greedy_cached_with_options(&prompt_b, max_new, &options)
            .expect("invariant: solo B stop valide");
        let stopped = model
            .generate_greedy_lightbatch_with_options(&[prompt_a, prompt_b], max_new, &options)
            .expect("invariant: light-batch stop valide");
        assert_eq!(stopped[0].tokens, solo_a);
        assert_eq!(stopped[1].tokens, solo_b);
    }

    /// Streaming (E2.5) : le callback reçoit exactement les tokens émis, par
    /// flux et dans l'ordre, avec le budget PROPRE de chaque flux.
    #[test]
    fn lightbatch_streaming_callback_matches_outputs_with_per_stream_budgets() {
        let model = tiny_model();
        let options = GenerationOptions::default();
        let prompts = [vec![0_usize, 1], vec![1_usize, 0, 1]];
        let maxes = [3_usize, 5];
        let mut streamed: [Vec<usize>; 2] = [Vec::new(), Vec::new()];
        let outputs = model
            .generate_greedy_lightbatch_streaming_with_options(
                &prompts,
                &maxes,
                &options,
                |stream, token| streamed[stream].push(token),
            )
            .expect("invariant: light-batch streaming valide");

        assert_eq!(outputs[0].tokens, streamed[0], "flux 0 streaming ≠ sortie");
        assert_eq!(outputs[1].tokens, streamed[1], "flux 1 streaming ≠ sortie");
        assert!(outputs[0].tokens.len() <= 3, "budget flux 0 dépassé");
        assert!(outputs[1].tokens.len() <= 5, "budget flux 1 dépassé");
        // Chaque flux == le solo avec SON budget.
        let solo_a = model
            .generate_greedy_cached_with_options(&prompts[0], 3, &options)
            .expect("invariant: solo A valide");
        let solo_b = model
            .generate_greedy_cached_with_options(&prompts[1], 5, &options)
            .expect("invariant: solo B valide");
        assert_eq!(outputs[0].tokens, solo_a);
        assert_eq!(outputs[1].tokens, solo_b);
    }

    /// Priorité du principal (E2.5) : le stride du flux de fond ne change PAS
    /// les séquences par flux (seulement QUAND les pas du fond s'exécutent).
    #[test]
    fn lightbatch_background_stride_preserves_per_stream_outputs() {
        let model = tiny_model();
        let options = GenerationOptions::default();
        let prompts = [vec![0_usize, 1], vec![1_usize, 0, 1]];
        let maxes = [16_usize, 16];
        let baseline = model
            .lightbatch_generate(&prompts, &maxes, &options, 1, |_, _| {})
            .expect("invariant: stride 1 valide");
        for stride in [2_u64, 3] {
            let strided = model
                .lightbatch_generate(&prompts, &maxes, &options, stride, |_, _| {})
                .expect("invariant: stride > 1 valide");
            assert_eq!(strided[0].tokens, baseline[0].tokens, "stride={stride}");
            assert_eq!(strided[1].tokens, baseline[1].tokens, "stride={stride}");
        }
    }

    /// Les budgets invalides sont rejetés (longueurs divergentes, budget nul).
    #[test]
    fn lightbatch_streaming_rejects_invalid_budgets() {
        let model = tiny_model();
        let options = GenerationOptions::default();
        let prompts = [vec![0_usize, 1], vec![1_usize]];
        let err = model
            .generate_greedy_lightbatch_streaming_with_options(&prompts, &[4], &options, |_, _| {})
            .expect_err("invariant: longueurs divergentes rejetées");
        assert!(matches!(err, InferError::Dimension(_)));
        let err = model
            .generate_greedy_lightbatch_streaming_with_options(
                &prompts,
                &[4, 0],
                &options,
                |_, _| {},
            )
            .expect_err("invariant: budget nul rejeté");
        assert!(matches!(err, InferError::Dimension(_)));
    }
}
