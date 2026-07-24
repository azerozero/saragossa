use super::{
    clone_effective_frame_cap, clone_sample_seed, greedy_talker_token, greedy_token,
    repeat_frame_stop_tripped, sample_talker_token, tts_generation_trace_enabled,
    tts_repeat_frame_stop, TtsModel, TtsSampleParams, CLONE_GENERATION_HARD_CAP,
    CLONE_SAMPLE_REPETITION_PENALTY, CLONE_SAMPLE_TEMPERATURE, CLONE_SAMPLE_TOP_K,
    CLONE_SAMPLE_TOP_P,
};
use crate::sampling::DeterministicSampler;
use crate::{CausalDecoderCache, InferError, Result, Tensor};
use std::time::Instant;

impl TtsModel {
    pub(super) fn generate_codes_greedy(
        &self,
        text: &str,
        max_frames: usize,
    ) -> Result<Vec<Vec<i32>>> {
        self.generate_codes_greedy_trace(
            text,
            max_frames,
            tts_generation_trace_enabled(),
            &mut |_| Ok(()),
        )
    }

    pub(super) fn generate_codes_clone_sampled(
        &self,
        text: &str,
        max_frames: usize,
    ) -> Result<Vec<Vec<i32>>> {
        let target_tokens = self.encode_ids(text)?.len();
        let effective_max_frames = clone_effective_frame_cap(max_frames, target_tokens);
        let params = TtsSampleParams {
            temperature: CLONE_SAMPLE_TEMPERATURE,
            top_k: CLONE_SAMPLE_TOP_K,
            top_p: CLONE_SAMPLE_TOP_P,
            repetition_penalty: CLONE_SAMPLE_REPETITION_PENALTY,
            seed: clone_sample_seed(),
        };
        self.generate_codes_clone_sampled_trace(
            text,
            max_frames,
            effective_max_frames,
            params,
            tts_generation_trace_enabled(),
            &mut |_| Ok(()),
        )
    }

    pub(super) fn generate_codes_clone_sampled_trace(
        &self,
        text: &str,
        requested_max_frames: usize,
        effective_max_frames: usize,
        params: TtsSampleParams,
        trace: bool,
        on_frame: &mut dyn FnMut(&[Vec<i32>]) -> Result<()>,
    ) -> Result<Vec<Vec<i32>>> {
        let cfg = &self.assets.model_config.talker_config;
        let mode = self
            .clone_ctx
            .as_ref()
            .map_or("clone-sampled", |ctx| ctx.mode.label());
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
        let (mut talker_cache, mut final_state) =
            self.talker.prefill_cache_from_embeddings(&prepared.input)?;
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

        let mut generated = Vec::with_capacity(effective_max_frames);
        let mut trailing_idx = 0_usize;
        let repeat_stop = tts_repeat_frame_stop();
        let mut repeat_run = 1_usize;
        let mut cb0_history = Vec::with_capacity(effective_max_frames);
        let mut sampler = DeterministicSampler::new(params.seed);

        for step in 0..effective_max_frames {
            let logits = self.talker.logits_from_final_state(&final_state)?;
            let tok0 = sample_talker_token(
                logits.as_row()?,
                &params,
                &suppress,
                &cb0_history,
                Some(eos),
                &mut sampler,
            )?;
            if trace {
                eprintln!("qwen3-tts rust diag: frame={step} cb0={tok0}");
            }
            if tok0 == eos {
                if trace {
                    eprintln!(
                        "qwen3-tts rust diag: eos frame={step} generated_frames={}",
                        generated.len()
                    );
                }
                break;
            }

            let codes_frame =
                self.predict_codebooks(tok0, &final_state, n_groups, cp_resident_cache.as_mut())?;

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
            final_state = if talker_resident {
                self.talker
                    .decode_step_resident_from_embedding(&mut talker_cache, &next_input)?
                    .ok_or_else(|| {
                        InferError::Metal(
                            "decode résident talker indisponible en cours de frame".to_string(),
                        )
                    })?
            } else {
                self.talker
                    .next_state_from_embedding(&mut talker_cache, &next_input)?
            };

            let repeated_frame = generated.last().is_some_and(|prev| prev == &codes_frame);
            repeat_run = if repeated_frame {
                repeat_run.saturating_add(1)
            } else {
                1
            };
            cb0_history.push(tok0);
            generated.push(codes_frame);
            on_frame(&generated)?;
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
                if trace {
                    eprintln!(
                        "qwen3-tts rust diag: stop repeat_frame run={repeat_run} threshold={repeat_stop} generated_frames={}",
                        generated.len()
                    );
                }
                break;
            }
        }
        if trace {
            eprintln!(
                "qwen3-tts rust diag: stop generated_frames={} reached_cap={} requested_max_frames={} effective_max_frames={effective_max_frames}",
                generated.len(),
                generated.len() == effective_max_frames,
                requested_max_frames
            );
        }
        Ok(generated)
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
    /// `[final_state, e0]` puis un pas par codebook).
    fn predict_codebooks(
        &self,
        tok0: i32,
        final_state: &Tensor,
        n_groups: usize,
        cp_resident: Option<&mut CausalDecoderCache>,
    ) -> Result<Vec<i32>> {
        let mut codes_frame = vec![tok0];
        if n_groups <= 1 {
            return Ok(codes_frame);
        }
        let e0 = self.codec_embed(&[tok0])?;
        match cp_resident {
            Some(cp_cache) => {
                // position 0 : final_state (seed KV, pas de code). Puis un pas par
                // codebook avec la tête FUSIONNÉE on-device (matmul tête + argmax
                // greedy dans le command buffer du pas → readback d'1 u32, plus de
                // cp_state relu ni de matmul de tête CPU). KV remis à zéro par frame.
                self.code_predictor.reset_resident_decode_cache(cp_cache)?;
                let fs_proj = self.code_predictor_projection.forward(final_state)?;
                self.code_predictor
                    .decode_step_resident_from_embedding(cp_cache, &fs_proj)?
                    .ok_or_else(|| {
                        InferError::Metal("cp résident: pas final_state absent".to_string())
                    })?;
                let mut next_input = self.code_predictor_projection.forward(&e0)?;
                for code_idx in 0..(n_groups - 1) {
                    let head = self.code_predictor_heads.get(code_idx).ok_or_else(|| {
                        InferError::MissingWeight(format!("code_predictor.lm_head.{code_idx}"))
                    })?;
                    let token = self
                        .code_predictor
                        .decode_token_resident_from_embedding_head(cp_cache, &next_input, head)?
                        .ok_or_else(|| {
                            InferError::Metal("cp résident: token absent".to_string())
                        })?;
                    let code = i32::try_from(token)
                        .map_err(|_| InferError::Config(format!("code cp hors i32: {token}")))?;
                    codes_frame.push(code);
                    if code_idx + 1 < n_groups - 1 {
                        let embed = self.code_predictor_embed(code_idx, &[code])?;
                        next_input = self.code_predictor_projection.forward(&embed)?;
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
                    let code = greedy_token(head.forward(&cp_state)?.as_row()?, &[])?;
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
    ) -> Result<Vec<Vec<i32>>> {
        let cfg = &self.assets.model_config.talker_config;
        let mode = self
            .clone_ctx
            .as_ref()
            .map_or("voicedesign", |ctx| ctx.mode.label());
        let effective_max_frames = if self.clone_ctx.is_some() {
            max_frames.min(CLONE_GENERATION_HARD_CAP)
        } else {
            max_frames
        };
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
        let (mut talker_cache, mut final_state) =
            self.talker.prefill_cache_from_embeddings(&prepared.input)?;
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
        let mut generated = Vec::with_capacity(effective_max_frames);
        let mut trailing_idx = 0_usize;
        let repeat_stop = tts_repeat_frame_stop();
        let mut repeat_run = 1_usize;

        for step in 0..effective_max_frames {
            // cb0 talker : argmax greedy GPU (quantifié + suppression) on-device,
            // sinon fallback CPU. Réplique greedy_talker_token byte-identique.
            let tok0 = self.talker_cb0(&final_state, suppress_start, eos, &suppress)?;
            if trace {
                eprintln!("qwen3-tts rust diag: frame={step} cb0={tok0}");
            }
            if tok0 == eos {
                if trace {
                    eprintln!(
                        "qwen3-tts rust diag: eos frame={step} generated_frames={}",
                        generated.len()
                    );
                }
                break;
            }

            let codes_frame =
                self.predict_codebooks(tok0, &final_state, n_groups, cp_resident_cache.as_mut())?;

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
            final_state = if talker_resident {
                self.talker
                    .decode_step_resident_from_embedding(&mut talker_cache, &next_input)?
                    .ok_or_else(|| {
                        InferError::Metal(
                            "decode résident talker indisponible en cours de frame".to_string(),
                        )
                    })?
            } else {
                self.talker
                    .next_state_from_embedding(&mut talker_cache, &next_input)?
            };
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
            on_frame(&generated)?;
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
                if trace {
                    eprintln!(
                        "qwen3-tts rust diag: stop repeat_frame run={repeat_run} threshold={repeat_stop} generated_frames={}",
                        generated.len()
                    );
                }
                break;
            }
        }
        if trace {
            eprintln!(
                "qwen3-tts rust diag: stop generated_frames={} reached_cap={} requested_max_frames={} effective_max_frames={effective_max_frames}",
                generated.len(),
                generated.len() == effective_max_frames,
                max_frames
            );
        }
        Ok(generated)
    }

    #[cfg(test)]
    pub(super) fn probe_greedy_logits(
        &self,
        text: &str,
        max_frames: usize,
        target_frame: usize,
        target_group: usize,
    ) -> Result<(Vec<Vec<i32>>, Vec<f32>)> {
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
