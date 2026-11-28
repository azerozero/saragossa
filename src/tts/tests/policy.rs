use super::*;

#[test]
#[ignore = "inférence: snapshot CustomVoice local et verrou GPU requis"]
fn customvoice_request_options_reset_and_change_generated_codes() -> Result<()> {
    let path = local_tts_snapshot(
        "SARAGOSSA_QWEN3_TTS_CUSTOMVOICE_DIR",
        "models--mlx-community--Qwen3-TTS-12Hz-1.7B-CustomVoice-6bit",
    )
    .ok_or_else(|| InferError::Config("snapshot CustomVoice local requis".into()))?;
    let mut model = TtsModel::load_local(path)?;
    assert!(model.codec.gpu_active(), "le test exige le codec Metal");
    let text = "Bonjour, comment allez-vous ?";
    crate::runtime_flags::with_tts_greedy_for_test(false, || -> Result<()> {
        let default = model.synthesize_default(text, 16)?;
        model.set_voice(Some("vivian"))?;
        let vivian = model.synthesize_default(text, 16)?;
        assert_ne!(
            default.codes, vivian.codes,
            "le locuteur doit conditionner les codes"
        );
        model.set_voice(None)?;
        model.set_seed(Some(17));
        let seeded = model.synthesize_default(text, 16)?;
        assert_ne!(
            default.codes, seeded.codes,
            "la graine doit piloter le sampling"
        );
        model.set_seed(None);
        assert!(model.set_voice(Some("inconnue")).is_err());
        let reset = model.synthesize_default_streaming(text, 16, |_| Ok(()))?;
        assert_eq!(
            default.codes, reset.codes,
            "voix et graine par défaut rétablies"
        );
        assert_eq!(default.stop_reason, reset.stop_reason);
        Ok(())
    })
}

/// Oracle du changement de tête uniquement : mêmes couches Metal et entrées,
/// ancienne projection CPU contre projection fusionnée. Seuil fixé à 1e-4
/// avant mesure, sans tolérance sur les identifiants tirés à graine égale.
#[test]
#[cfg(gpu_metal)]
#[ignore = "inférence: snapshots TTS locaux et verrou GPU requis"]
fn sampled_resident_heads_match_cpu_oracle() -> Result<()> {
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
        let prepared = model.prepare_inputs("Bonjour, comment allez-vous ?")?;
        let cfg = &model.assets.model_config.talker_config;
        let suppress = ((cfg.vocab_size - 1024)..cfg.vocab_size)
            .filter(|id| *id != cfg.codec_eos_token_id)
            .collect::<Vec<_>>();
        let mut max_cp = 0.0_f32;
        let mut max_talker = 0.0_f32;
        for seed in [0, 17, 42] {
            let params = config.talker_params(seed);
            let sub = config.subtalker_params(seed);
            let mut cpu_rng = DeterministicSampler::new(seed);
            let mut gpu_rng = DeterministicSampler::new(seed);
            let (mut old_cache, mut state) = model.prefill_talker(&prepared.input)?;
            let (mut new_cache, _) = model.prefill_talker(&prepared.input)?;
            assert!(model
                .talker
                .setup_resident_decode_from_prefill(&mut old_cache, 16)?);
            assert!(model
                .talker
                .setup_resident_decode_from_prefill(&mut new_cache, 16)?);
            let mut old_cp = model
                .code_predictor
                .new_resident_decode_cache(17)?
                .expect("invariant: code predictor résident");
            let mut new_cp = model
                .code_predictor
                .new_resident_decode_cache(17)?
                .expect("invariant: code predictor résident");
            let mut logits = model.talker.logits_from_final_state(&state)?;
            let mut history = Vec::new();
            for frame in 0..16 {
                let old_logits = model.talker.logits_from_final_state(&state)?;
                max_talker = max_talker.max(head_logits_close(old_logits.data(), logits.data()));
                let tok = sample_talker_token(
                    old_logits.data(),
                    &params,
                    &suppress,
                    &history,
                    Some(cfg.codec_eos_token_id),
                    &mut cpu_rng,
                )?;
                assert_eq!(
                    tok,
                    sample_talker_token(
                        logits.data(),
                        &params,
                        &suppress,
                        &history,
                        Some(cfg.codec_eos_token_id),
                        &mut gpu_rng
                    )?,
                    "talker seed={seed} frame={frame}"
                );
                if tok == cfg.codec_eos_token_id {
                    break;
                }
                model
                    .code_predictor
                    .reset_resident_decode_cache(&mut old_cp)?;
                model
                    .code_predictor
                    .reset_resident_decode_cache(&mut new_cp)?;
                let projected = model.code_predictor_projection.forward(&state)?;
                assert!(model
                    .code_predictor
                    .decode_step_resident_from_embedding(&mut old_cp, &projected)?
                    .is_some());
                assert!(model
                    .code_predictor
                    .decode_step_resident_from_embedding(&mut new_cp, &projected)?
                    .is_some());
                let mut codec = model.codec_embed(&[tok])?;
                let mut input = model.code_predictor_projection.forward(&codec)?;
                for (index, head) in model.code_predictor_heads.iter().enumerate() {
                    let old_state = model
                        .code_predictor
                        .decode_step_resident_from_embedding(&mut old_cp, &input)?
                        .expect("invariant: état résident");
                    let cpu = head.forward(&old_state)?;
                    let gpu = model
                        .code_predictor
                        .decode_logits_resident_from_embedding_head(&mut new_cp, &input, head)?
                        .expect("invariant: logits résidents");
                    max_cp = max_cp.max(head_logits_close(cpu.data(), gpu.data()));
                    let code = sample_talker_token(cpu.data(), &sub, &[], &[], None, &mut cpu_rng)?;
                    assert_eq!(
                        code,
                        sample_talker_token(gpu.data(), &sub, &[], &[], None, &mut gpu_rng)?,
                        "cp seed={seed} frame={frame} tête={index}"
                    );
                    let embed = model.code_predictor_embed(index, &[code])?;
                    codec = codec.add(&embed)?;
                    input = model.code_predictor_projection.forward(&embed)?;
                }
                let text = if frame < prepared.trailing.shape()[0] {
                    crate::Tensor::row(prepared.trailing.row_slice(frame)?.to_vec())?
                } else {
                    prepared.tts_pad.clone()
                };
                let input = text.add(&codec)?;
                state = model
                    .talker
                    .decode_step_resident_from_embedding(&mut old_cache, &input)?
                    .expect("invariant: état talker résident");
                let (new_state, new_logits) = model
                    .talker
                    .decode_state_logits_resident_from_embedding(&mut new_cache, &input)?
                    .expect("invariant: état/logits talker résidents");
                assert_eq!(
                    state.data(),
                    new_state.data(),
                    "la fusion ne modifie pas les couches"
                );
                logits = new_logits;
                history.push(tok);
            }
        }
        eprintln!("oracle {repo}: max_abs cp={max_cp:e}, talker={max_talker:e}; tokens identiques pour graines 0/17/42");
    }
    Ok(())
}

#[cfg(gpu_metal)]
fn head_logits_close(cpu: &[f32], gpu: &[f32]) -> f32 {
    assert_eq!(cpu.len(), gpu.len());
    let mut max_abs = 0.0_f32;
    for (&a, &b) in cpu.iter().zip(gpu) {
        assert!(a.is_finite() && b.is_finite());
        max_abs = max_abs.max((a - b).abs());
    }
    assert!(
        max_abs <= 1e-4,
        "logits CPU/Metal: max_abs={max_abs:e} > 1e-4"
    );
    max_abs
}
