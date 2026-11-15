use super::model::{GeneratedCodes, TtsStopReason};
use super::{
    clone_effective_frame_cap, clone_sample_seed, tts_generation_trace_enabled,
    tts_internal_profile_enabled, tts_stream_first_lot, usize_from_i32, TtsForwardOutput, TtsModel,
    TtsSampleParams, TtsStreamDecodeState, TtsSynthesisOutput, CLONE_GENERATION_HARD_CAP,
    CLONE_SAMPLE_REPETITION_PENALTY, CLONE_SAMPLE_TEMPERATURE, CLONE_SAMPLE_TOP_K,
    CLONE_SAMPLE_TOP_P,
};
use crate::{InferError, Result};
use std::time::{Duration, Instant};

impl TtsModel {
    /// Graine par requête ; `None` rétablit la graine déterministe 0 hors clonage.
    pub fn set_seed(&mut self, seed: Option<u64>) {
        self.seed = seed;
    }

    fn default_sample_params(&self) -> Option<TtsSampleParams> {
        if self.clone_ctx.is_some() {
            return Some(TtsSampleParams {
                temperature: CLONE_SAMPLE_TEMPERATURE,
                top_k: CLONE_SAMPLE_TOP_K,
                top_p: CLONE_SAMPLE_TOP_P,
                repetition_penalty: CLONE_SAMPLE_REPETITION_PENALTY,
                seed: self.seed.unwrap_or_else(clone_sample_seed),
            });
        }
        self.generation_config
            .as_ref()
            .map(|config| config.talker_params(self.seed.unwrap_or(0)))
    }

    fn default_frame_cap(&self, text: &str, max_frames: usize) -> Result<usize> {
        if self.clone_ctx.is_some() {
            Ok(clone_effective_frame_cap(
                max_frames,
                self.encode_ids(text)?.len(),
            ))
        } else {
            Ok(self
                .generation_config
                .as_ref()
                .map_or(max_frames, |config| max_frames.min(config.max_new_tokens)))
        }
    }

    /// Exécute le forward talker autoregressif sur le préfixe VoiceDesign.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si la tokenisation, la préparation des embeddings ou le
    /// forward échoue.
    pub fn forward_voicedesign_prefix(&self, text: &str) -> Result<TtsForwardOutput> {
        let _qmm_scope = crate::runtime_flags::install_tts_qmm_na_dequant_scope(
            crate::runtime_flags::tts_qmm_na_dequant_enabled(),
        );
        let prepared = self.prepare_voicedesign_inputs(text)?;
        let (cache, final_state) = self.prefill_talker(&prepared.input)?;
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
        let generated = self.generate_codes_greedy_trace(
            text,
            max_frames,
            tts_generation_trace_enabled(),
            &mut |_| Ok(()),
        )?;
        let codes = generated.codes;
        let samples = self.decode_codes_for_mode(&codes)?;
        Ok(TtsSynthesisOutput {
            frame_limit_reached: generated.stop_reason == TtsStopReason::Cap,
            stop_reason: generated.stop_reason,
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
        if crate::runtime_flags::tts_greedy_enabled() {
            return self.synthesize_greedy(text, max_frames);
        }
        let effective_max_frames = self.default_frame_cap(text, max_frames)?;
        let generated = if let Some(params) = self.default_sample_params() {
            self.generate_codes_clone_sampled_trace(
                text,
                max_frames,
                effective_max_frames,
                params,
                tts_generation_trace_enabled(),
                &mut |_| Ok(()),
            )?
        } else {
            self.generate_codes_greedy_trace(
                text,
                effective_max_frames,
                tts_generation_trace_enabled(),
                &mut |_| Ok(()),
            )?
        };
        let codes = generated.codes;
        let samples = self.decode_codes_for_mode(&codes)?;
        Ok(TtsSynthesisOutput {
            frame_limit_reached: generated.stop_reason == TtsStopReason::Cap,
            stop_reason: generated.stop_reason,
            codes,
            samples,
            sample_rate: self.codec.sample_rate(),
        })
    }

    /// Synthétise en streaming avec la politique de décodage adaptée au mode.
    ///
    /// Utilise les mêmes paramètres et la même graine que [`Self::synthesize_default`].
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
        if crate::runtime_flags::tts_greedy_enabled() {
            return self.synthesize_greedy_streaming(text, max_frames, on_audio);
        }
        let effective_max_frames = self.default_frame_cap(text, max_frames)?;
        if let Some(params) = self.default_sample_params() {
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
        self.synthesize_greedy_streaming(text, effective_max_frames, on_audio)
    }

    /// Synthétise en STREAMING : `on_audio(&[f32])` est appelé avec chaque nouveau
    /// segment PCM dès qu'un lot de frames est généré puis décodé, pour un premier
    /// son avant la fin de la synthèse. Le `TtsSynthesisOutput` agrège les segments.
    /// La parité avec [`Self::synthesize_greedy`] reste soumise aux tests du codec.
    ///
    /// Lots fixes si le codec conserve ses contextes causaux ; seuils doublés
    /// pour le repli CPU et le clone ICL, qui redécodent tout le préfixe.
    /// `SARAGOSSA_TTS_STREAM_LOT` règle la taille du lot (défaut : 4 trames).
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

    pub(super) fn greedy_frame_cap(&self, max_frames: usize) -> usize {
        if self.clone_ctx.is_some() {
            max_frames.min(CLONE_GENERATION_HARD_CAP)
        } else {
            max_frames
        }
    }

    fn synthesize_streaming_from_frames<F, G>(
        &self,
        generate_codes: G,
        on_audio: F,
    ) -> Result<TtsSynthesisOutput>
    where
        F: FnMut(&[f32]) -> Result<()>,
        G: FnOnce(&mut dyn FnMut(&[Vec<i32>]) -> Result<()>) -> Result<GeneratedCodes>,
    {
        let mut decode_state = self.new_stream_decode_state();
        let fixed_lots = uses_fixed_lots(&decode_state);
        let decode =
            |codes: &[Vec<i32>]| self.decode_codes_for_mode_streaming(&mut decode_state, codes);
        if crate::runtime_flags::tts_codec_overlap_enabled() {
            return synthesize_streaming_overlapped(
                generate_codes,
                on_audio,
                decode,
                tts_stream_first_lot(),
                fixed_lots,
                self.codec.sample_rate(),
            );
        }
        synthesize_streaming_from_frames(
            generate_codes,
            on_audio,
            decode,
            tts_stream_first_lot(),
            fixed_lots,
            self.codec.sample_rate(),
        )
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

fn uses_fixed_lots(state: &TtsStreamDecodeState) -> bool {
    matches!(state, TtsStreamDecodeState::Incremental(codec) if codec.is_incremental())
}

// L'injection du codec et des trames vérifie le découpage et l'arrêt du callback
// sans charger de modèle ; la production utilise exactement cette boucle.
fn synthesize_streaming_from_frames<F, G, D>(
    generate_codes: G,
    mut on_audio: F,
    mut decode: D,
    first_lot: usize,
    fixed_lots: bool,
    sample_rate: u32,
) -> Result<TtsSynthesisOutput>
where
    F: FnMut(&[f32]) -> Result<()>,
    G: FnOnce(&mut dyn FnMut(&[Vec<i32>]) -> Result<()>) -> Result<GeneratedCodes>,
    D: FnMut(&[Vec<i32>]) -> Result<Vec<f32>>,
{
    let mut all_samples: Vec<f32> = Vec::new();
    let mut threshold = first_lot;
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
        let pcm = decode(generated)?;
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
        threshold = next_lot_threshold(generated.len(), threshold, first_lot, fixed_lots);
        Ok(())
    };
    let generated = generate_codes(&mut on_frame)?;
    let codes = generated.codes;
    // Flush final : décode le suffixe non couvert par le dernier lot.
    let final_flush_started = Instant::now();
    let pcm = decode(&codes)?;
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
        frame_limit_reached: generated.stop_reason == TtsStopReason::Cap,
        stop_reason: generated.stop_reason,
        codes,
        samples: all_samples,
        sample_rate,
    })
}

/// Seuil du lot suivant, partagé par les variantes synchrone et recouverte.
fn next_lot_threshold(generated: usize, threshold: usize, first_lot: usize, fixed: bool) -> usize {
    // Seul le codec à contextes causaux évite de redécoder le passé.
    if fixed {
        generated.saturating_add(first_lot)
    } else {
        threshold.saturating_mul(2).max(generated.saturating_add(1))
    }
}

/// Variante de [`synthesize_streaming_from_frames`] où le codec tourne sur un
/// thread dédié : la trame suivante du talker se calcule pendant le décodage
/// du lot courant (double tampon). Les lots arrivés pendant un décodage sont
/// fusionnés en un seul appel ; le codec étant causal, le PCM ne dépend pas du
/// découpage. `on_audio` reste appelé sur le thread appelant, dans l'ordre,
/// au fil des trames puis à la fin.
fn synthesize_streaming_overlapped<F, G, D>(
    generate_codes: G,
    mut on_audio: F,
    decode: D,
    first_lot: usize,
    fixed_lots: bool,
    sample_rate: u32,
) -> Result<TtsSynthesisOutput>
where
    F: FnMut(&[f32]) -> Result<()>,
    G: FnOnce(&mut dyn FnMut(&[Vec<i32>]) -> Result<()>) -> Result<GeneratedCodes>,
    D: FnMut(&[Vec<i32>]) -> Result<Vec<f32>> + Send,
{
    use std::sync::mpsc;
    let profile = tts_internal_profile_enabled();
    let profile_started = Instant::now();
    std::thread::scope(|scope| {
        let (job_tx, job_rx) = mpsc::channel::<Vec<Vec<i32>>>();
        let (pcm_tx, pcm_rx) = mpsc::channel::<Result<Vec<f32>>>();
        let worker = scope.spawn(move || codec_worker(job_rx, pcm_tx, decode));
        let mut all_samples: Vec<f32> = Vec::new();
        let mut first_emit = None;
        let mut emit = |pcm: Result<Vec<f32>>, all: &mut Vec<f32>| -> Result<()> {
            let pcm = pcm?;
            if !pcm.is_empty() {
                first_emit.get_or_insert_with(|| profile_started.elapsed());
                on_audio(&pcm)?;
                all.extend_from_slice(&pcm);
            }
            Ok(())
        };
        // Le worker s'arrête sur une erreur codec : la remonter plutôt que l'arrêt.
        let worker_gone = || match pcm_rx.try_recv() {
            Ok(Err(error)) => error,
            _ => InferError::Metal("thread codec TTS arrêté".to_string()),
        };
        let outcome = (|| {
            let job_tx = job_tx;
            let mut sent = 0_usize;
            let mut threshold = first_lot;
            let generated = generate_codes(&mut |generated: &[Vec<i32>]| -> Result<()> {
                while let Ok(pcm) = pcm_rx.try_recv() {
                    emit(pcm, &mut all_samples)?;
                }
                if generated.len() < threshold {
                    return Ok(());
                }
                job_tx
                    .send(generated[sent..].to_vec())
                    .map_err(|_| worker_gone())?;
                sent = generated.len();
                threshold = next_lot_threshold(generated.len(), threshold, first_lot, fixed_lots);
                Ok(())
            })?;
            // Flush final : le suffixe non couvert par le dernier lot.
            if generated.codes.len() > sent {
                job_tx
                    .send(generated.codes[sent..].to_vec())
                    .map_err(|_| worker_gone())?;
            }
            drop(job_tx);
            for pcm in pcm_rx.iter() {
                emit(pcm, &mut all_samples)?;
            }
            Ok(generated)
        })();
        // Canal de travaux fermé dans tous les cas : le thread codec se termine.
        let stats = worker
            .join()
            .map_err(|_| InferError::Metal("thread codec TTS en panique".to_string()))?;
        let generated = outcome?;
        if profile {
            eprintln!(
                "perf tts.internal.stream_codec overlap=1 first_lot={} frames={} samples={} codec_calls={} codec_ms={:.3} first_codec_ms={:.3} first_emit_ms={:.3} total_ms={:.3}",
                first_lot,
                generated.codes.len(),
                all_samples.len(),
                stats.calls,
                stats.total.as_secs_f64() * 1e3,
                stats.first.unwrap_or_default().as_secs_f64() * 1e3,
                first_emit.unwrap_or_default().as_secs_f64() * 1e3,
                profile_started.elapsed().as_secs_f64() * 1e3,
            );
        }
        Ok(TtsSynthesisOutput {
            frame_limit_reached: generated.stop_reason == TtsStopReason::Cap,
            stop_reason: generated.stop_reason,
            codes: generated.codes,
            samples: all_samples,
            sample_rate,
        })
    })
}

#[derive(Default)]
struct CodecWorkerStats {
    calls: usize,
    total: Duration,
    first: Option<Duration>,
}

/// Décode les lots reçus dans l'ordre ; s'arrête au premier échec ou quand
/// l'un des canaux est fermé.
fn codec_worker<D>(
    jobs: std::sync::mpsc::Receiver<Vec<Vec<i32>>>,
    results: std::sync::mpsc::Sender<Result<Vec<f32>>>,
    mut decode: D,
) -> CodecWorkerStats
where
    D: FnMut(&[Vec<i32>]) -> Result<Vec<f32>>,
{
    let mut stats = CodecWorkerStats::default();
    let mut prefix = Vec::new();
    while let Ok(job) = jobs.recv() {
        prefix.extend(job);
        while let Ok(more) = jobs.try_recv() {
            prefix.extend(more);
        }
        let started = Instant::now();
        let pcm = decode(&prefix);
        let elapsed = started.elapsed();
        stats.calls += 1;
        stats.total += elapsed;
        stats.first.get_or_insert(elapsed);
        let failed = pcm.is_err();
        if results.send(pcm).is_err() || failed {
            break;
        }
    }
    stats
}

#[cfg(test)]
mod tests;
