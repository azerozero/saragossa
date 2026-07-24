use super::{
    clone_effective_frame_cap, clone_sample_seed, tts_generation_trace_enabled,
    tts_internal_profile_enabled, tts_stream_first_lot, usize_from_i32, TtsForwardOutput, TtsModel,
    TtsSampleParams, TtsStreamDecodeState, TtsSynthesisOutput, CLONE_SAMPLE_REPETITION_PENALTY,
    CLONE_SAMPLE_TEMPERATURE, CLONE_SAMPLE_TOP_K, CLONE_SAMPLE_TOP_P,
};
use crate::{InferError, Result};
use std::time::{Duration, Instant};

impl TtsModel {
    /// Exécute le forward talker autoregressif sur le préfixe VoiceDesign.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si la tokenisation, la préparation des embeddings ou le
    /// forward échoue.
    pub fn forward_voicedesign_prefix(&self, text: &str) -> Result<TtsForwardOutput> {
        let prepared = self.prepare_voicedesign_inputs(text)?;
        let (cache, final_state) = self.talker.prefill_cache_from_embeddings(&prepared.input)?;
        let logits = self.talker.logits_from_final_state(&final_state)?;
        Ok(TtsForwardOutput {
            cache,
            logits,
            final_state,
        })
    }

    /// Synthétise un texte en PCM f32 mono avec décodage greedy déterministe.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si la génération des codes ou le codec échoue.
    /// Renvoie la fréquence d'échantillonnage du codec (Hz) — pour le playback
    /// streaming où le taux est requis avant la fin de la synthèse.
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.codec.sample_rate()
    }

    pub fn synthesize_greedy(&self, text: &str, max_frames: usize) -> Result<TtsSynthesisOutput> {
        let codes = self.generate_codes_greedy(text, max_frames)?;
        let samples = self.decode_codes_for_mode(&codes)?;
        Ok(TtsSynthesisOutput {
            codes,
            samples,
            sample_rate: self.codec.sample_rate(),
        })
    }

    /// Synthétise avec la politique de décodage adaptée au mode.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si la génération des codes ou le codec échoue.
    pub fn synthesize_default(&self, text: &str, max_frames: usize) -> Result<TtsSynthesisOutput> {
        let codes = if self.clone_ctx.is_some() {
            self.generate_codes_clone_sampled(text, max_frames)?
        } else {
            self.generate_codes_greedy(text, max_frames)?
        };
        let samples = self.decode_codes_for_mode(&codes)?;
        Ok(TtsSynthesisOutput {
            codes,
            samples,
            sample_rate: self.codec.sample_rate(),
        })
    }

    /// Synthétise en streaming avec la politique de décodage adaptée au mode.
    ///
    /// VoiceDesign conserve le chemin greedy historique. Le clone Base/ICL passe
    /// par le sampling + cap adaptatif utilisés par [`Self::synthesize_default`],
    /// ce qui évite de réintroduire le chemin greedy clone qui boucle.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si la génération, le codec, ou `on_audio` échoue.
    pub fn synthesize_default_streaming<F>(
        &self,
        text: &str,
        max_frames: usize,
        on_audio: F,
    ) -> Result<TtsSynthesisOutput>
    where
        F: FnMut(&[f32]) -> Result<()>,
    {
        if self.clone_ctx.is_some() {
            let target_tokens = self.encode_ids(text)?.len();
            let effective_max_frames = clone_effective_frame_cap(max_frames, target_tokens);
            let params = TtsSampleParams {
                temperature: CLONE_SAMPLE_TEMPERATURE,
                top_k: CLONE_SAMPLE_TOP_K,
                top_p: CLONE_SAMPLE_TOP_P,
                repetition_penalty: CLONE_SAMPLE_REPETITION_PENALTY,
                seed: clone_sample_seed(),
            };
            let trace = tts_generation_trace_enabled();
            return self.synthesize_streaming_from_frames(
                |on_frame| {
                    self.generate_codes_clone_sampled_trace(
                        text,
                        max_frames,
                        effective_max_frames,
                        params,
                        trace,
                        on_frame,
                    )
                },
                on_audio,
            );
        }
        self.synthesize_greedy_streaming(text, max_frames, on_audio)
    }

    /// Synthétise en STREAMING : `on_audio(&[f32])` est appelé avec chaque nouveau
    /// segment PCM dès qu'un lot de frames est généré puis décodé, pour un premier
    /// son immédiat (TTFA basse). La concaténation des segments est **byte-identique**
    /// à [`Self::synthesize_greedy`] : le codec étant entièrement causal, décoder le
    /// préfixe [0..N] puis émettre le delta reproduit exactement l'audio batch.
    ///
    /// Stratégie « préfixe croissant, émission du delta » : on re-décode le préfixe
    /// courant par lots à taille croissante (1er lot petit → TTFA basse ; ×2 ensuite
    /// → coût total O(N) borné, le codec GPU étant rapide). Le `TtsSynthesisOutput`
    /// renvoyé agrège tout l'audio émis (== batch).
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si la génération, le codec, ou `on_audio` échoue.
    pub fn synthesize_greedy_streaming<F>(
        &self,
        text: &str,
        max_frames: usize,
        on_audio: F,
    ) -> Result<TtsSynthesisOutput>
    where
        F: FnMut(&[f32]) -> Result<()>,
    {
        self.synthesize_streaming_from_frames(
            |on_frame| {
                self.generate_codes_greedy_trace(
                    text,
                    max_frames,
                    tts_generation_trace_enabled(),
                    on_frame,
                )
            },
            on_audio,
        )
    }

    fn synthesize_streaming_from_frames<F, G>(
        &self,
        generate_codes: G,
        mut on_audio: F,
    ) -> Result<TtsSynthesisOutput>
    where
        F: FnMut(&[f32]) -> Result<()>,
        G: FnOnce(&mut dyn FnMut(&[Vec<i32>]) -> Result<()>) -> Result<Vec<Vec<i32>>>,
    {
        let first_lot = tts_stream_first_lot();
        let mut all_samples: Vec<f32> = Vec::new();
        let mut threshold = first_lot;
        let mut decode_state = self.new_stream_decode_state();
        let profile = tts_internal_profile_enabled();
        let profile_started = Instant::now();
        let mut codec_calls = 0_usize;
        let mut codec_total = Duration::ZERO;
        let mut first_codec = None;
        let mut first_emit = None;
        let mut first_emit_frames = 0_usize;
        let mut on_frame = |generated: &[Vec<i32>]| -> Result<()> {
            if generated.len() < threshold {
                return Ok(());
            }
            let codec_started = Instant::now();
            let pcm = self.decode_codes_for_mode_streaming(&mut decode_state, generated)?;
            let codec_elapsed = codec_started.elapsed();
            if profile {
                codec_calls += 1;
                codec_total += codec_elapsed;
                first_codec.get_or_insert(codec_elapsed);
                if !pcm.is_empty() && first_emit.is_none() {
                    first_emit = Some(profile_started.elapsed());
                    first_emit_frames = generated.len();
                }
            }
            if !pcm.is_empty() {
                on_audio(&pcm)?;
                all_samples.extend_from_slice(&pcm);
            }
            // Lots à taille croissante (×2) : O(N) total, peu de dispatches.
            threshold = threshold.saturating_mul(2).max(generated.len() + 1);
            Ok(())
        };
        let codes = generate_codes(&mut on_frame)?;
        // Flush final : décode le suffixe non couvert par le dernier lot.
        let final_flush_started = Instant::now();
        let pcm = self.decode_codes_for_mode_streaming(&mut decode_state, &codes)?;
        let final_flush = final_flush_started.elapsed();
        if profile {
            codec_calls += 1;
            codec_total += final_flush;
            first_codec.get_or_insert(final_flush);
            if !pcm.is_empty() && first_emit.is_none() {
                first_emit = Some(profile_started.elapsed());
                first_emit_frames = codes.len();
            }
        }
        if !pcm.is_empty() {
            on_audio(&pcm)?;
            all_samples.extend_from_slice(&pcm);
        }
        if profile {
            eprintln!(
                "perf tts.internal.stream_codec first_lot={} frames={} samples={} codec_calls={} codec_ms={:.3} first_codec_ms={:.3} first_emit_ms={:.3} first_emit_frames={} final_flush_ms={:.3} total_ms={:.3}",
                first_lot,
                codes.len(),
                all_samples.len(),
                codec_calls,
                codec_total.as_secs_f64() * 1e3,
                first_codec.unwrap_or_default().as_secs_f64() * 1e3,
                first_emit.unwrap_or_default().as_secs_f64() * 1e3,
                first_emit_frames,
                final_flush.as_secs_f64() * 1e3,
                profile_started.elapsed().as_secs_f64() * 1e3,
            );
        }
        Ok(TtsSynthesisOutput {
            codes,
            samples: all_samples,
            sample_rate: self.codec.sample_rate(),
        })
    }

    pub(super) fn decode_codes_for_mode(&self, codes: &[Vec<i32>]) -> Result<Vec<f32>> {
        let Some(ctx) = self.clone_ctx.as_ref() else {
            return self.codec.decode_codes(codes);
        };
        if codes.is_empty() {
            return Ok(Vec::new());
        }
        if ctx.mode.is_xvec_only() {
            return self.codec.decode_codes(codes);
        }
        let mut prefixed = ctx.ref_codes.clone();
        prefixed.extend_from_slice(codes);
        let mut samples = self.codec.decode_codes(&prefixed)?;
        let skip = ctx
            .ref_codes
            .len()
            .checked_mul(usize_from_i32(
                self.assets.codec_config.decode_upsample_rate,
                "decode_upsample_rate",
            )?)
            .ok_or_else(|| InferError::Shape("trim clone TTS trop grand".to_string()))?;
        if skip >= samples.len() {
            return Ok(Vec::new());
        }
        samples.drain(0..skip);
        Ok(samples)
    }

    fn new_stream_decode_state(&self) -> TtsStreamDecodeState {
        match self.clone_ctx.as_ref() {
            Some(ctx) if !ctx.mode.is_xvec_only() => {
                TtsStreamDecodeState::FullPrefixDelta { emitted: 0 }
            }
            _ => TtsStreamDecodeState::Incremental(self.codec.new_stream_state()),
        }
    }

    fn decode_codes_for_mode_streaming(
        &self,
        state: &mut TtsStreamDecodeState,
        codes: &[Vec<i32>],
    ) -> Result<Vec<f32>> {
        match state {
            TtsStreamDecodeState::Incremental(codec_state) => {
                self.codec.decode_codes_streaming(codec_state, codes)
            }
            TtsStreamDecodeState::FullPrefixDelta { emitted } => {
                let pcm = self.decode_codes_for_mode(codes)?;
                if pcm.len() < *emitted {
                    return Err(InferError::Dimension(format!(
                        "streaming TTS PCM régressif: {} < {}",
                        pcm.len(),
                        *emitted
                    )));
                }
                let delta = pcm[*emitted..].to_vec();
                *emitted = pcm.len();
                Ok(delta)
            }
        }
    }
}
