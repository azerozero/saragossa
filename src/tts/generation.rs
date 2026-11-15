use super::model::{GeneratedCodes, TtsStopReason};
use super::profile::{FrameProfile, Phase};
use super::{
    greedy_talker_token, greedy_token, repeat_frame_stop_tripped, sample_talker_token,
    tts_repeat_frame_stop, TtsModel, TtsSampleParams, CLONE_GENERATION_HARD_CAP,
};
use crate::sampling::DeterministicSampler;
use crate::{CausalDecoderCache, InferError, Result, Tensor};
use std::time::Instant;

/// Poids de la trame fusionnée du code predictor (jamais construits hors Metal).
#[cfg(all(target_os = "macos", feature = "metal"))]
type CpFusedWeights = crate::decoder::TtsCodebookWeights;
#[cfg(not(all(target_os = "macos", feature = "metal")))]
type CpFusedWeights = std::convert::Infallible;

impl TtsModel {
    #[cfg(test)]
    pub(super) fn generate_codes_greedy(
        &self,
        text: &str,
        max_frames: usize,
    ) -> Result<Vec<Vec<i32>>> {
        self.generate_codes_greedy_trace(
            text,
            max_frames,
            super::tts_generation_trace_enabled(),
            &mut |_| Ok(()),
        )
        .map(|output| output.codes)
    }

    pub(super) fn generate_codes_clone_sampled_trace(
        &self,
        text: &str,
        requested_max_frames: usize,
        effective_max_frames: usize,
        params: TtsSampleParams,
        trace: bool,
        on_frame: &mut dyn FnMut(&[Vec<i32>]) -> Result<()>,
    ) -> Result<GeneratedCodes> {
        let cfg = &self.assets.model_config.talker_config;
        let mode = self.clone_ctx.as_ref().map_or_else(
            || match self.assets.model_kind() {
                super::TtsModelKind::CustomVoice => "customvoice-sampled",
                _ => "voicedesign-sampled",
            },
            |ctx| ctx.mode.label(),
        );
        if trace {
            eprintln!(
                "qwen3-tts rust diag: start mode={mode} requested_max_frames={requested_max_frames} effective_max_frames={effective_max_frames} hard_cap={CLONE_GENERATION_HARD_CAP} temp={} top_k={} top_p={} rep_penalty={} seed={}",
                params.temperature,
                params.top_k,
                params.top_p,
                params.repetition_penalty,
                params.seed,
            );
        }

        let started = Instant::now();
        let prepared = self.prepare_inputs(text)?;
        if trace {
            let ref_frames = self.clone_ctx.as_ref().map_or(0, |ctx| ctx.ref_codes.len());
            eprintln!(
                "qwen3-tts rust diag: prepared input_rows={} trailing_rows={} ref_frames={} elapsed_ms={}",
                prepared.input.shape()[0],
                prepared.trailing.shape()[0],
                ref_frames,
                started.elapsed().as_millis()
            );
        }
        let n_groups = usize::try_from(cfg.num_code_groups)
            .map_err(|_| InferError::Config("num_code_groups TTS négatif".to_string()))?;
        if n_groups == 0 {
            return Err(InferError::Config("num_code_groups TTS nul".to_string()));
        }
        let eos = cfg.codec_eos_token_id;
        let vocab = cfg.vocab_size;
        let suppress_start = vocab.checked_sub(1024).ok_or_else(|| {
            InferError::Config(format!("vocab TTS trop petit pour suppression: {vocab}"))
        })?;
        let suppress = (suppress_start..vocab)
            .filter(|token| *token != eos)
            .collect::<Vec<_>>();

        let hidden = self.hidden_dim()?;
        if trace {
            eprintln!(
                "qwen3-tts rust diag: prefill start hidden={hidden} groups={n_groups} eos={eos} vocab={vocab} suppress_start={suppress_start}"
            );
        }
        let prefill_started = Instant::now();
        let (mut talker_cache, mut final_state) = self.prefill_talker(&prepared.input)?;
        if trace {
            eprintln!(
                "qwen3-tts rust diag: prefill done elapsed_ms={}",
                prefill_started.elapsed().as_millis()
            );
        }
        let talker_resident = self
            .talker
            .setup_resident_decode_from_prefill(&mut talker_cache, effective_max_frames)?;
        if trace {
            eprintln!("qwen3-tts rust diag: talker resident={talker_resident}");
        }
        let mut cp_resident_cache = if n_groups > 1 {
            self.code_predictor
                .new_resident_decode_cache(n_groups + 1)?
        } else {
            None
        };
        if trace {
            eprintln!(
                "qwen3-tts rust diag: cp resident={}",
                cp_resident_cache.is_some()
            );
        }

        let subtalker = self
            .generation_config
            .as_ref()
            .map(|config| config.subtalker_params(params.seed));
        let cp_fused = match (&cp_resident_cache, &subtalker) {
            (Some(_), Some(sub)) => self.cp_fused_weights(sub)?,
            _ => None,
        };
        if trace {
            eprintln!("qwen3-tts rust diag: cp fused={}", cp_fused.is_some());
        }

        let mut stop_reason = TtsStopReason::Cap;
        let mut generated = Vec::with_capacity(effective_max_frames);
        let mut trailing_idx = 0_usize;
        let repeat_stop = tts_repeat_frame_stop();
        let mut repeat_run = 1_usize;
        let mut cb0_history = Vec::with_capacity(effective_max_frames);
        let mut sampler = DeterministicSampler::new(params.seed);

        let mut profile = FrameProfile::new();
        let mut logits = {
            let _qmm_scope = crate::runtime_flags::install_tts_qmm_na_dequant_scope(
                crate::runtime_flags::tts_qmm_na_dequant_enabled(),
            );
            self.talker.logits_from_final_state(&final_state)?
        };
        for step in 0..effective_max_frames {
            let qmm_scope = crate::runtime_flags::install_tts_qmm_na_dequant_scope(
                crate::runtime_flags::tts_qmm_na_dequant_enabled(),
            );
            let tok0 = profile.time(Phase::Cb0Sample, || {
                sample_talker_token(
                    logits.as_row()?,
                    &params,
                    &suppress,
                    &cb0_history,
                    Some(eos),
                    &mut sampler,
                )
            })?;
            if trace {
                eprintln!("qwen3-tts rust diag: frame={step} cb0={tok0}");
            }
            if tok0 == eos {
                stop_reason = TtsStopReason::Eos;
                if trace {
                    eprintln!(
                        "qwen3-tts rust diag: eos frame={step} generated_frames={}",
                        generated.len()
                    );
                }
                break;
            }

            let codes_frame = self.predict_codebooks(
                tok0,
                &final_state,
                n_groups,
                cp_resident_cache.as_mut(),
                cp_fused.as_ref(),
                subtalker.as_ref().map(|p| (p, &mut sampler)),
                &mut profile,
            )?;

            let next_input = profile.time(Phase::InputEmbed, || {
                self.next_talker_input(&prepared, &mut trailing_idx, &codes_frame)
            })?;
            (final_state, logits) = profile.time(Phase::Talker, || {
                if talker_resident {
                    self.talker
                        .decode_state_logits_resident_from_embedding(
                            &mut talker_cache,
                            &next_input,
                        )?
                        .ok_or_else(|| {
                            InferError::Metal(
                                "decode résident talker indisponible en cours de frame".to_string(),
                            )
                        })
                } else {
                    let state = self
                        .talker
                        .next_state_from_embedding(&mut talker_cache, &next_input)?;
                    let logits = self.talker.logits_from_final_state(&state)?;
                    Ok((state, logits))
                }
            })?;

            let repeated_frame = generated.last().is_some_and(|prev| prev == &codes_frame);
            repeat_run = if repeated_frame {
                repeat_run.saturating_add(1)
            } else {
                1
            };
            cb0_history.push(tok0);
            generated.push(codes_frame);
            // Le callback streaming peut appeler un LLM sur le même thread.
            drop(qmm_scope);
            profile.time(Phase::Codec, || on_frame(&generated))?;
            if trace {
                let last = generated
                    .last()
                    .ok_or_else(|| InferError::Dimension("frame TTS manquante".to_string()))?;
                eprintln!(
                    "qwen3-tts rust diag: frame={step} done generated_frames={} trailing_idx={} codes={last:?}",
                    generated.len(),
                    trailing_idx
                );
            }
            if repeat_frame_stop_tripped(repeat_run, repeat_stop) {
                stop_reason = TtsStopReason::Repeat;
                eprintln!(
                    "qwen3-tts rust diag: stop repeat_frame run={repeat_run} threshold={repeat_stop} generated_frames={}",
                    generated.len()
                );
                break;
            }
        }
        if trace {
            eprintln!(
                "qwen3-tts rust diag: stop generated_frames={} reached_cap={} reason={} requested_max_frames={} effective_max_frames={effective_max_frames}",
                generated.len(),
                stop_reason == TtsStopReason::Cap,
                stop_reason.as_str(),
                requested_max_frames
            );
        }
        profile.report(mode, generated.len());
        Ok(GeneratedCodes {
            codes: generated,
            stop_reason,
        })
    }

    /// Poids de la trame fusionnée si le kill-switch et les paramètres le
    /// permettent : température positive, `1 <= top_k < vocab`, `top_p` dans
    /// `(0, 1]`. Sinon `None` et le chemin pas à pas (tirage CPU) reste actif.
    fn cp_fused_weights(&self, sub: &TtsSampleParams) -> Result<Option<CpFusedWeights>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let supported = crate::runtime_flags::tts_cp_fused_enabled()
                && self.code_predictor_embeddings_bf16_exact
                && sub.temperature.is_finite()
                && sub.temperature > f32::EPSILON
                && sub.top_k >= 1
                && sub.top_p.is_finite()
                && sub.top_p > 0.0
                && sub.top_p <= 1.0;
            if !supported {
                return Ok(None);
            }
            let Some(weights) = self.code_predictor.resolve_tts_codebook_weights(
                &self.code_predictor_projection,
                &self.code_predictor_heads,
                &self.code_predictor_embeddings,
            )?
            else {
                return Ok(None);
            };
            Ok((sub.top_k < weights.vocab()).then_some(weights))
        }
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = sub;
            Ok(None)
        }
    }

    /// Entrée talker de la trame suivante : ligne de texte restante (ou pad)
    /// plus la somme des embeddings codec des 16 codes, dans l'ordre historique.
    pub(super) fn next_talker_input(
        &self,
        prepared: &super::PreparedVoiceDesign,
        trailing_idx: &mut usize,
        codes_frame: &[i32],
    ) -> Result<Tensor> {
        let text_embed = if *trailing_idx < prepared.trailing.shape()[0] {
            let row = Tensor::row(prepared.trailing.row_slice(*trailing_idx)?.to_vec())?;
            *trailing_idx += 1;
            row
        } else {
            prepared.tts_pad.clone()
        };
        let (tok0, rest) = codes_frame
            .split_first()
            .ok_or_else(|| InferError::Dimension("trame TTS vide".to_string()))?;
        let mut codec_embed = self.codec_embed(&[*tok0])?;
        for (codebook, code) in rest.iter().copied().enumerate() {
            codec_embed = codec_embed.add(&self.code_predictor_embed(codebook, &[code])?)?;
        }
        text_embed.add(&codec_embed)
    }

    /// Sélectionne cb0 (token talker) depuis `final_state` : argmax greedy GPU
    /// (quantifié + suppression de la plage `[suppress_start, vocab)` sauf `eos`)
    /// si Metal est dispo, sinon fallback CPU `logits_from_final_state` +
    /// `greedy_talker_token`. Byte-identique entre les deux chemins.
    fn talker_cb0(
        &self,
        final_state: &Tensor,
        suppress_start: i32,
        eos: i32,
        suppress: &[i32],
    ) -> Result<i32> {
        let suppress_start_usize = usize::try_from(suppress_start)
            .map_err(|_| InferError::Config("suppress_start TTS négatif".to_string()))?;
        let eos_usize =
            usize::try_from(eos).map_err(|_| InferError::Config("eos TTS négatif".to_string()))?;
        match self
            .talker
            .talker_greedy_token(final_state, suppress_start_usize, eos_usize)?
        {
            Some(token) => i32::try_from(token)
                .map_err(|_| InferError::Config(format!("cb0 talker hors i32: {token}"))),
            None => {
                let logits = self.talker.logits_from_final_state(final_state)?;
                greedy_talker_token(logits.as_row()?, suppress)
            }
        }
    }

    /// Prédit les codes d'un frame : `tok0` (talker) suivi des `n_groups-1`
    /// codebooks du code_predictor, en decode KV-caché.
    ///
    /// Si `cp_resident` est `Some`, chaque pas du code_predictor passe par le
    /// decode GPU résident (1 pas = 1 command buffer) ; sinon le decode reste
    /// per-op. Les deux chemins sont algorithmiquement identiques (préfixe
    /// `[final_state, e0]` puis un pas par codebook). En échantillonné, avec
    /// `cp_fused`, les 16 pas tiennent dans un seul command buffer.
    #[expect(
        clippy::too_many_arguments,
        reason = "état talker, caches résident et fusionné, tirage et profil de trame"
    )]
    fn predict_codebooks(
        &self,
        tok0: i32,
        final_state: &Tensor,
        n_groups: usize,
        cp_resident: Option<&mut CausalDecoderCache>,
        cp_fused: Option<&CpFusedWeights>,
        mut sampling: Option<(&TtsSampleParams, &mut DeterministicSampler)>,
        profile: &mut FrameProfile,
    ) -> Result<Vec<i32>> {
        let _qmm_scope = crate::runtime_flags::install_tts_qmm_na_dequant_scope(
            crate::runtime_flags::tts_qmm_na_dequant_enabled(),
        );
        let mut codes_frame = vec![tok0];
        if n_groups <= 1 {
            return Ok(codes_frame);
        }
        let e0 = self.codec_embed(&[tok0])?;
        match cp_resident {
            Some(cp_cache) => {
                #[cfg(all(target_os = "macos", feature = "metal"))]
                if let (Some(weights), Some((params, sampler))) = (cp_fused, sampling.as_mut()) {
                    let sample = crate::decode_resident::TtsCodebookSample {
                        temperature: params.temperature,
                        top_k: params.top_k,
                        top_p: params.top_p,
                    };
                    let (codes, _) = profile.time(Phase::CpSteps, || {
                        self.code_predictor
                            .decode_tts_codebooks_resident(
                                cp_cache,
                                weights,
                                final_state.as_row()?,
                                e0.as_row()?,
                                sample,
                                sampler.state(),
                                false,
                            )?
                            .ok_or_else(|| InferError::Metal("cp fusionné: arène absente".into()))
                    })?;
                    for code in codes {
                        // Un tirage GPU par codebook : le flux CPU avance d'autant.
                        sampler.advance();
                        codes_frame.push(i32::try_from(code).map_err(|_| {
                            InferError::Config(format!("code cp hors i32: {code}"))
                        })?);
                    }
                    return Ok(codes_frame);
                }
                #[cfg(not(all(target_os = "macos", feature = "metal")))]
                let _ = cp_fused;
                // position 0 : final_state (seed KV, pas de code). Puis un pas par
                // codebook avec la tête dans le command buffer du pas. En greedy,
                // on relit un u32 ; en sampling, uniquement les logits. Aucun état
                // cp relu pour projeter la tête sur CPU. KV remis à zéro par frame.
                self.code_predictor.reset_resident_decode_cache(cp_cache)?;
                let fs_proj = profile.time(Phase::CpProjection, || {
                    self.code_predictor_projection.forward(final_state)
                })?;
                profile.time(Phase::CpSteps, || {
                    self.code_predictor
                        .decode_step_resident_from_embedding(cp_cache, &fs_proj)?
                        .ok_or_else(|| {
                            InferError::Metal("cp résident: pas final_state absent".to_string())
                        })
                })?;
                let mut next_input = profile.time(Phase::CpProjection, || {
                    self.code_predictor_projection.forward(&e0)
                })?;
                for code_idx in 0..(n_groups - 1) {
                    let head = self.code_predictor_heads.get(code_idx).ok_or_else(|| {
                        InferError::MissingWeight(format!("code_predictor.lm_head.{code_idx}"))
                    })?;
                    let code = if let Some((params, sampler)) = sampling.as_mut() {
                        let logits = profile.time(Phase::CpSteps, || {
                            self.code_predictor
                                .decode_logits_resident_from_embedding_head(
                                    cp_cache,
                                    &next_input,
                                    head,
                                )?
                                .ok_or_else(|| {
                                    InferError::Metal("cp résident: logits absents".into())
                                })
                        })?;
                        profile.time(Phase::CpSample, || {
                            sample_talker_token(logits.as_row()?, params, &[], &[], None, sampler)
                        })?
                    } else {
                        let token = profile.time(Phase::CpSteps, || {
                            self.code_predictor
                                .decode_token_resident_from_embedding_head(
                                    cp_cache,
                                    &next_input,
                                    head,
                                )?
                                .ok_or_else(|| {
                                    InferError::Metal("cp résident: token absent".into())
                                })
                        })?;
                        i32::try_from(token)
                            .map_err(|_| InferError::Config(format!("code cp hors i32: {token}")))?
                    };
                    codes_frame.push(code);
                    if code_idx + 1 < n_groups - 1 {
                        let embed = self.code_predictor_embed(code_idx, &[code])?;
                        next_input = profile.time(Phase::CpProjection, || {
                            self.code_predictor_projection.forward(&embed)
                        })?;
                    }
                }
            }
            None => {
                let hidden = self.hidden_dim()?;
                let mut prefix_rows = Vec::with_capacity(2 * hidden);
                prefix_rows.extend_from_slice(final_state.as_row()?);
                prefix_rows.extend_from_slice(e0.as_row()?);
                let prefix = Tensor::from_vec(vec![2, hidden], prefix_rows)?;
                let cp_prefix = self.code_predictor_projection.forward(&prefix)?;
                let (mut cp_cache, mut cp_state) = self
                    .code_predictor
                    .prefill_cache_from_embeddings(&cp_prefix)?;
                for code_idx in 0..(n_groups - 1) {
                    let head = self.code_predictor_heads.get(code_idx).ok_or_else(|| {
                        InferError::MissingWeight(format!("code_predictor.lm_head.{code_idx}"))
                    })?;
                    let logits = head.forward(&cp_state)?;
                    let code = if let Some((params, sampler)) = sampling.as_mut() {
                        sample_talker_token(logits.as_row()?, params, &[], &[], None, sampler)?
                    } else {
                        greedy_token(logits.as_row()?, &[])?
                    };
                    codes_frame.push(code);
                    if code_idx + 1 < n_groups - 1 {
                        let embed = self.code_predictor_embed(code_idx, &[code])?;
                        let cp_in = self.code_predictor_projection.forward(&embed)?;
                        cp_state = self
                            .code_predictor
                            .next_state_from_embedding(&mut cp_cache, &cp_in)?;
                    }
                }
            }
        }
        Ok(codes_frame)
    }

    pub(super) fn generate_codes_greedy_trace(
        &self,
        text: &str,
        max_frames: usize,
        trace: bool,
        on_frame: &mut dyn FnMut(&[Vec<i32>]) -> Result<()>,
    ) -> Result<GeneratedCodes> {
        let cfg = &self.assets.model_config.talker_config;
        let mode = self
            .clone_ctx
            .as_ref()
            .map_or("voicedesign", |ctx| ctx.mode.label());
        let effective_max_frames = self.greedy_frame_cap(max_frames);
        if trace {
            eprintln!(
                "qwen3-tts rust diag: start mode={mode} requested_max_frames={max_frames} effective_max_frames={effective_max_frames} hard_cap={CLONE_GENERATION_HARD_CAP}"
            );
        }

        let started = Instant::now();
        let prepared = self.prepare_inputs(text)?;
        if trace {
            let ref_frames = self.clone_ctx.as_ref().map_or(0, |ctx| ctx.ref_codes.len());
            eprintln!(
                "qwen3-tts rust diag: prepared input_rows={} trailing_rows={} ref_frames={} elapsed_ms={}",
                prepared.input.shape()[0],
                prepared.trailing.shape()[0],
                ref_frames,
                started.elapsed().as_millis()
            );
        }
        let n_groups = usize::try_from(cfg.num_code_groups)
            .map_err(|_| InferError::Config("num_code_groups TTS négatif".to_string()))?;
        if n_groups == 0 {
            return Err(InferError::Config("num_code_groups TTS nul".to_string()));
        }
        let eos = cfg.codec_eos_token_id;
        let vocab = cfg.vocab_size;
        let suppress_start = vocab.checked_sub(1024).ok_or_else(|| {
            InferError::Config(format!("vocab TTS trop petit pour suppression: {vocab}"))
        })?;
        let suppress = (suppress_start..vocab)
            .filter(|token| *token != eos)
            .collect::<Vec<_>>();

        let hidden = self.hidden_dim()?;
        if trace {
            eprintln!(
                "qwen3-tts rust diag: prefill start hidden={hidden} groups={n_groups} eos={eos} vocab={vocab} suppress_start={suppress_start}"
            );
        }
        let prefill_started = Instant::now();
        let (mut talker_cache, mut final_state) = self.prefill_talker(&prepared.input)?;
        if trace {
            eprintln!(
                "qwen3-tts rust diag: prefill done elapsed_ms={}",
                prefill_started.elapsed().as_millis()
            );
        }
        // Decode résident GPU du talker : un pas = un command buffer (vs ~28 couches
        // per-op). Seed le KV GPU depuis le cache du prefill ; si indisponible, on
        // reste sur le per-op (`next_state_from_embedding`). Tout-ou-rien.
        let talker_resident = self
            .talker
            .setup_resident_decode_from_prefill(&mut talker_cache, effective_max_frames)?;
        if trace {
            eprintln!("qwen3-tts rust diag: talker resident={talker_resident}");
        }
        // Code predictor : arène résidente allouée une fois, remise à zéro par
        // frame (séquence courte de num_code_groups+1 positions).
        let mut cp_resident_cache = if n_groups > 1 {
            self.code_predictor
                .new_resident_decode_cache(n_groups + 1)?
        } else {
            None
        };
        if trace {
            eprintln!(
                "qwen3-tts rust diag: cp resident={}",
                cp_resident_cache.is_some()
            );
        }
        let mut stop_reason = TtsStopReason::Cap;
        let mut generated = Vec::with_capacity(effective_max_frames);
        let mut trailing_idx = 0_usize;
        let repeat_stop = tts_repeat_frame_stop();
        let mut repeat_run = 1_usize;
        let mut profile = FrameProfile::new();

        for step in 0..effective_max_frames {
            let qmm_scope = crate::runtime_flags::install_tts_qmm_na_dequant_scope(
                crate::runtime_flags::tts_qmm_na_dequant_enabled(),
            );
            // cb0 talker : argmax greedy GPU (quantifié + suppression) on-device,
            // sinon fallback CPU. Réplique greedy_talker_token byte-identique.
            let tok0 = profile.time(Phase::Cb0Sample, || {
                self.talker_cb0(&final_state, suppress_start, eos, &suppress)
            })?;
            if trace {
                eprintln!("qwen3-tts rust diag: frame={step} cb0={tok0}");
            }
            if tok0 == eos {
                stop_reason = TtsStopReason::Eos;
                if trace {
                    eprintln!(
                        "qwen3-tts rust diag: eos frame={step} generated_frames={}",
                        generated.len()
                    );
                }
                break;
            }

            let codes_frame = self.predict_codebooks(
                tok0,
                &final_state,
                n_groups,
                cp_resident_cache.as_mut(),
                None,
                None,
                &mut profile,
            )?;

            let next_input = profile.time(Phase::InputEmbed, || {
                self.next_talker_input(&prepared, &mut trailing_idx, &codes_frame)
            })?;
            final_state = profile.time(Phase::Talker, || {
                if talker_resident {
                    self.talker
                        .decode_step_resident_from_embedding(&mut talker_cache, &next_input)?
                        .ok_or_else(|| {
                            InferError::Metal(
                                "decode résident talker indisponible en cours de frame".to_string(),
                            )
                        })
                } else {
                    self.talker
                        .next_state_from_embedding(&mut talker_cache, &next_input)
                }
            })?;
            let repeated_frame = generated.last().is_some_and(|prev| prev == &codes_frame);
            repeat_run = if repeated_frame {
                repeat_run.saturating_add(1)
            } else {
                1
            };
            generated.push(codes_frame);
            // Hook streaming : permet au décodage codec incrémental d'émettre
            // l'audio du préfixe au fil de la génération (byte-identique par
            // causalité). No-op pour le chemin batch.
            drop(qmm_scope);
            profile.time(Phase::Codec, || on_frame(&generated))?;
            if trace {
                let last = generated
                    .last()
                    .ok_or_else(|| InferError::Dimension("frame TTS manquante".to_string()))?;
                eprintln!(
                    "qwen3-tts rust diag: frame={step} done generated_frames={} trailing_idx={} codes={last:?}",
                    generated.len(),
                    trailing_idx
                );
            }
            if repeat_frame_stop_tripped(repeat_run, repeat_stop) {
                stop_reason = TtsStopReason::Repeat;
                eprintln!(
                    "qwen3-tts rust diag: stop repeat_frame run={repeat_run} threshold={repeat_stop} generated_frames={}",
                    generated.len()
                );
                break;
            }
        }
        if trace {
            eprintln!(
                "qwen3-tts rust diag: stop generated_frames={} reached_cap={} reason={} requested_max_frames={} effective_max_frames={effective_max_frames}",
                generated.len(),
                stop_reason == TtsStopReason::Cap,
                stop_reason.as_str(),
                max_frames
            );
        }
        profile.report(mode, generated.len());
        Ok(GeneratedCodes {
            codes: generated,
            stop_reason,
        })
    }

    #[cfg(test)]
    pub(super) fn probe_greedy_logits(
        &self,
        text: &str,
        max_frames: usize,
        target_frame: usize,
        target_group: usize,
    ) -> Result<(Vec<Vec<i32>>, Vec<f32>)> {
        let _qmm_scope = crate::runtime_flags::install_tts_qmm_na_dequant_scope(
            crate::runtime_flags::tts_qmm_na_dequant_enabled(),
        );
        let cfg = &self.assets.model_config.talker_config;
        let prepared = self.prepare_inputs(text)?;
        let n_groups = usize::try_from(cfg.num_code_groups)
            .map_err(|_| InferError::Config("num_code_groups TTS négatif".to_string()))?;
        if target_group >= n_groups {
            return Err(InferError::Config(format!(
                "target_group TTS hors bornes: {target_group}"
            )));
        }
        let eos = cfg.codec_eos_token_id;
        let vocab = cfg.vocab_size;
        let suppress_start = vocab.checked_sub(1024).ok_or_else(|| {
            InferError::Config(format!("vocab TTS trop petit pour suppression: {vocab}"))
        })?;
        let suppress = (suppress_start..vocab)
            .filter(|token| *token != eos)
            .collect::<Vec<_>>();
        let hidden = self.hidden_dim()?;
        let (mut talker_cache, mut final_state) =
            self.talker.prefill_cache_from_embeddings(&prepared.input)?;
        let mut logits = self.talker.logits_from_final_state(&final_state)?;
        let mut generated = Vec::with_capacity(max_frames);
        let mut trailing_idx = 0_usize;

        for frame_idx in 0..max_frames {
            if frame_idx == target_frame && target_group == 0 {
                return Ok((generated, logits.as_row()?.to_vec()));
            }
            let tok0 = greedy_talker_token(logits.as_row()?, &suppress)?;
            if tok0 == eos {
                break;
            }

            let mut codes_frame = vec![tok0];
            let e0 = self.codec_embed(&[tok0])?;
            let mut cp_rows = Vec::with_capacity((n_groups + 1) * hidden);
            cp_rows.extend_from_slice(final_state.as_row()?);
            cp_rows.extend_from_slice(e0.as_row()?);

            for code_idx in 0..(n_groups - 1) {
                if code_idx > 0 {
                    let prev = codes_frame[code_idx];
                    let embed = self.code_predictor_embed(code_idx - 1, &[prev])?;
                    cp_rows.extend_from_slice(embed.as_row()?);
                }
                let cp_input =
                    Tensor::from_vec(vec![cp_rows.len() / hidden, hidden], cp_rows.clone())?;
                let cp_input = self.code_predictor_projection.forward(&cp_input)?;
                let (_cp_cache, cp_state) = self
                    .code_predictor
                    .prefill_cache_from_embeddings(&cp_input)?;
                let head = self.code_predictor_heads.get(code_idx).ok_or_else(|| {
                    InferError::MissingWeight(format!("code_predictor.lm_head.{code_idx}"))
                })?;
                let code_logits = head.forward(&cp_state)?;
                let group = code_idx + 1;
                if frame_idx == target_frame && group == target_group {
                    return Ok((generated, code_logits.as_row()?.to_vec()));
                }
                codes_frame.push(greedy_token(code_logits.as_row()?, &[])?);
            }

            let text_embed = if trailing_idx < prepared.trailing.shape()[0] {
                let row = Tensor::row(prepared.trailing.row_slice(trailing_idx)?.to_vec())?;
                trailing_idx += 1;
                row
            } else {
                prepared.tts_pad.clone()
            };
            let mut codec_embed = self.codec_embed(&[tok0])?;
            for (codebook, code) in codes_frame.iter().skip(1).copied().enumerate() {
                codec_embed = codec_embed.add(&self.code_predictor_embed(codebook, &[code])?)?;
            }
            let next_input = text_embed.add(&codec_embed)?;
            final_state = self
                .talker
                .next_state_from_embedding(&mut talker_cache, &next_input)?;
            logits = self.talker.logits_from_final_state(&final_state)?;
            generated.push(codes_frame);
        }
        Err(InferError::Dimension(format!(
            "point probe TTS non atteint: frame={target_frame}, group={target_group}"
        )))
    }
}
