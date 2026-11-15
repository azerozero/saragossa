//! Oracle de la trame fusionnée du code predictor (`SARAGOSSA_TTS_CP_FUSED`).
//!
//! Seuils fixés avant la première mesure, jamais élargis :
//! - logits fusionnés contre pas à pas (projection CPU, mêmes codes imposés) :
//!   `max_abs <= 1e-3`, la projection passant du CPU au GPU en amont de 5 couches ;
//! - probabilités GPU contre `token_distribution_top_k_top_p` sur les mêmes
//!   logits, comparées en vecteurs denses (hors support = 0) : `max_abs <= 1e-6`,
//!   même ordre hors ex aequo. Avec `top_p = 1`, le CPU coupe le noyau dès que
//!   le cumul arrondi atteint 1 ; la queue restante, sous l'ulp de 1, peut donc
//!   différer d'un côté à l'autre : elle est comptée et reste sous ce seuil ;
//! - tirages : le tirage CPU sur les logits GPU, au même état RNG, rend le même
//!   code ; un écart n'est admis que si l'uniforme tombe à moins de 1e-5 d'une
//!   frontière cumulée CPU (quasi-égalité), et il est compté.

use super::*;
use crate::decode_resident::TtsCodebookSample;
use crate::sampling::token_distribution_top_k_top_p;

const LOGITS_MAX_ABS: f32 = 1e-3;
const PROBS_MAX_ABS: f32 = 1e-6;
const NEAR_TIE: f32 = 1e-5;

#[test]
#[ignore = "inférence: snapshots TTS locaux et verrou GPU requis"]
fn fused_codebooks_match_cpu_oracle() -> Result<()> {
    for (env, repo) in [
        (
            "SARAGOSSA_QWEN3_TTS_CUSTOMVOICE_DIR",
            "models--mlx-community--Qwen3-TTS-12Hz-1.7B-CustomVoice-6bit",
        ),
        (
            "SARAGOSSA_QWEN3_TTS_VOICEDESIGN_DIR",
            "models--mlx-community--Qwen3-TTS-12Hz-1.7B-VoiceDesign-6bit",
        ),
    ] {
        let path = local_tts_snapshot(env, repo)
            .ok_or_else(|| InferError::Config(format!("snapshot requis: {repo}")))?;
        let model = TtsModel::load_local(path)?;
        let config = model
            .generation_config
            .as_ref()
            .expect("invariant: snapshot avec sampling");
        let weights = model
            .code_predictor
            .resolve_tts_codebook_weights(
                &model.code_predictor_projection,
                &model.code_predictor_heads,
                &model.code_predictor_embeddings,
            )?
            .expect("invariant: trame fusionnée disponible");
        let prepared = model.prepare_inputs("Bonjour, comment allez-vous ?")?;
        let cfg = &model.assets.model_config.talker_config;
        let eos = cfg.codec_eos_token_id;
        let suppress = ((cfg.vocab_size - 1024)..cfg.vocab_size)
            .filter(|id| *id != eos)
            .collect::<Vec<_>>();
        let (mut max_logits, mut max_probs) = (0.0_f32, 0.0_f32);
        let (mut draws, mut near_ties, mut support_diffs) = (0_usize, 0_usize, 0_usize);
        for seed in [0, 17, 42] {
            let params = config.talker_params(seed);
            let sub = config.subtalker_params(seed);
            assert!(
                sub.top_k <= crate::decode_resident::TTS_SAMPLER_TRACE_K,
                "top_k {} dépasse la trace GPU de {} rangs",
                sub.top_k,
                crate::decode_resident::TTS_SAMPLER_TRACE_K
            );
            let sample = TtsCodebookSample {
                temperature: sub.temperature,
                top_k: sub.top_k,
                top_p: sub.top_p,
            };
            let mut rng = DeterministicSampler::new(seed);
            let (mut talker, mut state) = model.prefill_talker(&prepared.input)?;
            assert!(model
                .talker
                .setup_resident_decode_from_prefill(&mut talker, 16)?);
            let mut fused_cp = model
                .code_predictor
                .new_resident_decode_cache(17)?
                .expect("invariant: code predictor résident");
            let mut step_cp = model
                .code_predictor
                .new_resident_decode_cache(17)?
                .expect("invariant: code predictor résident");
            let mut logits = model.talker.logits_from_final_state(&state)?;
            let mut history = Vec::new();
            let mut trailing = 0;
            for frame in 0..16 {
                let tok = sample_talker_token(
                    logits.data(),
                    &params,
                    &suppress,
                    &history,
                    Some(eos),
                    &mut rng,
                )?;
                if tok == eos {
                    break;
                }
                let e0 = model.codec_embed(&[tok])?;
                let (codes, trace) = model
                    .code_predictor
                    .decode_tts_codebooks_resident(
                        &mut fused_cp,
                        &weights,
                        state.data(),
                        e0.data(),
                        sample,
                        rng.state(),
                        true,
                    )?
                    .expect("invariant: arène résidente");
                let trace = trace.expect("invariant: trace demandée");
                let vocab = weights.vocab();

                // Pas à pas de référence, codes fusionnés imposés.
                model
                    .code_predictor
                    .reset_resident_decode_cache(&mut step_cp)?;
                let seed_row = model.code_predictor_projection.forward(&state)?;
                model
                    .code_predictor
                    .decode_step_resident_from_embedding(&mut step_cp, &seed_row)?
                    .expect("invariant: amorce résidente");
                let mut input = model.code_predictor_projection.forward(&e0)?;
                let mut frame_codes = vec![tok];
                for (head_index, head) in model.code_predictor_heads.iter().enumerate() {
                    let reference = model
                        .code_predictor
                        .decode_logits_resident_from_embedding_head(&mut step_cp, &input, head)?
                        .expect("invariant: logits résidents");
                    let fused = &trace.logits[head_index * vocab..(head_index + 1) * vocab];
                    for (&a, &b) in reference.data().iter().zip(fused) {
                        assert!(a.is_finite() && b.is_finite());
                        max_logits = max_logits.max((a - b).abs());
                    }

                    let cpu = token_distribution_top_k_top_p(
                        fused,
                        sample.temperature,
                        sample.top_p,
                        sample.top_k,
                    )?;
                    let gpu = &trace.distributions[head_index];
                    // Comparaison dense : un token hors d'un support compte pour 0.
                    let mut dense = vec![(0.0_f32, 0.0_f32); vocab];
                    for &(token, prob) in &cpu {
                        dense[token].0 = prob;
                    }
                    for &(token, prob) in gpu {
                        dense[token].1 = prob;
                    }
                    for &(cp, gp) in &dense {
                        max_probs = max_probs.max((cp - gp).abs());
                    }
                    if cpu.len() != gpu.len() {
                        support_diffs += 1;
                    }
                    for (rank, (&(ct, cp), &(gt, gp))) in cpu.iter().zip(gpu).enumerate() {
                        assert!(
                            ct == gt || (cp - gp).abs() <= PROBS_MAX_ABS,
                            "ordre {repo} seed={seed} frame={frame} rang={rank}"
                        );
                    }

                    let mut probe = rng.clone();
                    let cpu_code = crate::sampling::sample_from_token_distribution(&cpu, &mut rng)?;
                    let gpu_code = codes[head_index] as usize;
                    draws += 1;
                    if cpu_code != gpu_code {
                        let total = cpu.iter().map(|(_, p)| *p).sum::<f32>();
                        let roll = probe.next_unit_f32() * total;
                        let mut acc = 0.0_f32;
                        let gap = cpu
                            .iter()
                            .map(|(_, p)| {
                                acc += p;
                                (roll - acc).abs()
                            })
                            .fold(f32::INFINITY, f32::min);
                        assert!(
                            gap < NEAR_TIE,
                            "tirage {repo} seed={seed} frame={frame} tête={head_index}: CPU {cpu_code} GPU {gpu_code}, écart {gap:e}"
                        );
                        near_ties += 1;
                    }
                    let code = i32::try_from(gpu_code).expect("invariant: code < vocab");
                    frame_codes.push(code);
                    input = model
                        .code_predictor_projection
                        .forward(&model.code_predictor_embed(head_index, &[code])?)?;
                }
                let next = model.next_talker_input(&prepared, &mut trailing, &frame_codes)?;
                (state, logits) = model
                    .talker
                    .decode_state_logits_resident_from_embedding(&mut talker, &next)?
                    .expect("invariant: talker résident");
                history.push(tok);
            }
        }
        eprintln!(
            "oracle fusionné {repo}: logits max_abs={max_logits:e}, probabilités max_abs={max_probs:e}, tirages={draws}, quasi-égalités={near_ties}, supports différents={support_diffs}"
        );
        assert!(
            max_logits <= LOGITS_MAX_ABS,
            "logits fusionnés: {max_logits:e}"
        );
        assert!(max_probs <= PROBS_MAX_ABS, "probabilités: {max_probs:e}");
    }
    Ok(())
}

/// Couches TTS fusionnées (`SARAGOSSA_TTS_CP_FUSED_KERNELS`) contre le chemin
/// général : mêmes logits au bit et mêmes codes sur 3 graines × 16 trames.
#[test]
#[ignore = "inférence: snapshots TTS locaux et verrou GPU requis"]
fn tts_cp_fused_kernels_match_general_layers() -> Result<()> {
    for (env, repo) in [
        (
            "SARAGOSSA_QWEN3_TTS_CUSTOMVOICE_DIR",
            "models--mlx-community--Qwen3-TTS-12Hz-1.7B-CustomVoice-6bit",
        ),
        (
            "SARAGOSSA_QWEN3_TTS_VOICEDESIGN_DIR",
            "models--mlx-community--Qwen3-TTS-12Hz-1.7B-VoiceDesign-6bit",
        ),
    ] {
        let path = local_tts_snapshot(env, repo)
            .ok_or_else(|| InferError::Config(format!("snapshot requis: {repo}")))?;
        let model = TtsModel::load_local(path)?;
        let config = model
            .generation_config
            .as_ref()
            .expect("invariant: snapshot avec sampling");
        let resolve = || {
            model
                .code_predictor
                .resolve_tts_codebook_weights(
                    &model.code_predictor_projection,
                    &model.code_predictor_heads,
                    &model.code_predictor_embeddings,
                )
                .map(|weights| weights.expect("invariant: trame fusionnée disponible"))
        };
        let fused_weights = resolve()?;
        let mut general_weights = resolve()?;
        assert!(
            general_weights.take_fused_kernels(),
            "couches fusionnées inactives sur {repo}"
        );
        let prepared = model.prepare_inputs("Bonjour, comment allez-vous ?")?;
        let mut frames = 0_usize;
        for seed in [0, 17, 42] {
            let sub = config.subtalker_params(seed);
            let sample = TtsCodebookSample {
                temperature: sub.temperature,
                top_k: sub.top_k,
                top_p: sub.top_p,
            };
            let mut rng = DeterministicSampler::new(seed);
            let (mut talker, mut state) = model.prefill_talker(&prepared.input)?;
            assert!(model
                .talker
                .setup_resident_decode_from_prefill(&mut talker, 16)?);
            let mut caches = [(); 2].map(|()| {
                model
                    .code_predictor
                    .new_resident_decode_cache(17)
                    .map(|cache| cache.expect("invariant: code predictor résident"))
            });
            let [Ok(fused_cp), Ok(general_cp)] = &mut caches else {
                return Err(InferError::Config("cache code predictor".into()));
            };
            let mut trailing = 0;
            for frame in 0..16 {
                let tok = (seed as i32 + frame) % 2048;
                let e0 = model.codec_embed(&[tok])?;
                let run = |cache: &mut _, weights| {
                    model
                        .code_predictor
                        .decode_tts_codebooks_resident(
                            cache,
                            weights,
                            state.data(),
                            e0.data(),
                            sample,
                            rng.state(),
                            true,
                        )
                        .map(|out| out.expect("invariant: arène résidente"))
                };
                let (codes, fused_trace) = run(fused_cp, &fused_weights)?;
                let (general_codes, general_trace) = run(general_cp, &general_weights)?;
                let (fused_trace, general_trace) = (
                    fused_trace.expect("invariant: trace"),
                    general_trace.expect("invariant: trace"),
                );
                assert_eq!(codes, general_codes, "{repo} seed={seed} frame={frame}");
                assert!(
                    fused_trace
                        .logits
                        .iter()
                        .zip(&general_trace.logits)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "logits {repo} seed={seed} frame={frame}"
                );
                for _ in 0..codes.len() {
                    rng.advance();
                }
                let mut frame_codes = vec![tok];
                frame_codes.extend(codes.iter().map(|&code| code as i32));
                let next = model.next_talker_input(&prepared, &mut trailing, &frame_codes)?;
                (state, _) = model
                    .talker
                    .decode_state_logits_resident_from_embedding(&mut talker, &next)?
                    .expect("invariant: talker résident");
                frames += 1;
            }
        }
        eprintln!("couches TTS fusionnées {repo}: {frames} trames identiques au bit");
    }
    Ok(())
}
