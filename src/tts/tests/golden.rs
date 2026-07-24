use super::*;

#[test]
#[ignore = "live: charge les payloads Qwen3-TTS VoiceDesign et exécute le talker"]
fn live_loads_voicedesign_payloads_and_forwards_talker() -> Result<()> {
    let Some(model_dir) = local_tts_snapshot(
        "RETI_QWEN3_TTS_VOICEDESIGN_DIR",
        "models--mlx-community--Qwen3-TTS-12Hz-1.7B-VoiceDesign-6bit",
    ) else {
        eprintln!("skip: snapshot Qwen3-TTS VoiceDesign absent du cache HF");
        return Ok(());
    };
    let model = TtsModel::load_local(model_dir)?;
    let out = model.forward_voicedesign_prefix("Bonjour.")?;
    let summary = model.payload_summary();
    eprintln!(
        "tts payloads: talker_tensors={} codec_tensors={} codec_payload_bytes={} codec_payload_bytes_read={} codec_payload_checksum={:#x} logits_shape={:?}",
        summary.talker_tensor_count,
        summary.codec_tensor_count,
        summary.codec_payload_bytes,
        summary.codec_payload_bytes_read,
        summary.codec_payload_checksum,
        out.logits.shape()
    );
    assert!(summary.talker_tensor_count > 0);
    assert!(summary.codec_tensor_count > 0);
    assert!(summary.codec_payload_bytes > 0);
    assert_eq!(
        summary.codec_payload_bytes_read,
        summary.codec_payload_bytes
    );
    // Le snapshot est un poids réel (HF, live-only) : son checksum FNV-1a
    // 64 n'est pas un vecteur connu à figer ici. L'égalité exacte sur
    // vecteur de référence vit dans les tests isolés
    // `fnv1a64_update_matches_official_reference_vectors` et
    // `payload_summary_checksum_hashes_raw_tensor_bytes_fnv1a64`
    // (payload synthétique déterministe) ; ce test-ci reste un smoke
    // check de non-nullité sur données live.
    assert_ne!(summary.codec_payload_checksum, 0);
    assert_eq!(
        out.logits.shape(),
        &[
            1,
            model.assets.model_config.talker_config.vocab_size as usize
        ]
    );
    Ok(())
}

/// Talker VoiceDesign metal-rs ≡ golden mlx-rs figé (sans mlx-rs). Même critère :
/// argmax identique + `max_abs<=0.20`.
#[test]
#[ignore = "golden: charge Qwen3-TTS VoiceDesign (cache HF) pour le talker metal-rs"]
fn golden_voicedesign_talker_logits_matches_fixture() -> Result<()> {
    const TEXT: &str = "Bonjour.";
    const MAX_ABS_TOLERANCE: f32 = 0.20;

    let Some(model_dir) = local_tts_snapshot(
        "RETI_QWEN3_TTS_VOICEDESIGN_DIR",
        "models--mlx-community--Qwen3-TTS-12Hz-1.7B-VoiceDesign-6bit",
    ) else {
        eprintln!("skip: snapshot Qwen3-TTS VoiceDesign absent du cache HF");
        return Ok(());
    };
    let rust = TtsModel::load_local(model_dir)?;
    let rust_logits = rust
        .forward_voicedesign_prefix(TEXT)?
        .logits
        .as_row()?
        .to_vec();

    let (_, golden) = crate::golden::read_f32("voicedesign_talker_logits")?;
    if rust_logits.len() != golden.len() {
        return Err(InferError::Dimension(format!(
            "logits TTS len rust={} golden={}",
            rust_logits.len(),
            golden.len()
        )));
    }
    let max_abs = max_abs_same_len(&rust_logits, &golden)?;
    assert_eq!(argmax_index(&rust_logits)?, argmax_index(&golden)?);
    assert!(
        max_abs <= MAX_ABS_TOLERANCE,
        "drift talker TTS max_abs={max_abs} > {MAX_ABS_TOLERANCE}"
    );
    Ok(())
}

/// Pipeline TTS VoiceDesign metal-rs ≡ golden mlx-rs figé (sans mlx-rs). Mêmes
/// invariants : rate 24 kHz, longueurs égales, codes frame 0 + 6 premiers groupes
/// frame 1 identiques, `max_abs<=0.5`, `rms<=0.1`.
#[test]
#[ignore = "golden: charge Qwen3-TTS VoiceDesign (cache HF) pour la synthèse metal-rs"]
fn golden_voicedesign_e2e_audio_matches_fixture() -> Result<()> {
    const TEXT: &str = "Bonjour, test réel.";
    const MAX_FRAMES: usize = 2;
    const PROBE_GROUP: usize = 6;
    const MAX_ABS_TOLERANCE: f32 = 0.50;
    const RMS_TOLERANCE: f32 = 0.10;

    let Some(model_dir) = local_tts_snapshot(
        "RETI_QWEN3_TTS_VOICEDESIGN_DIR",
        "models--mlx-community--Qwen3-TTS-12Hz-1.7B-VoiceDesign-6bit",
    ) else {
        eprintln!("skip: snapshot Qwen3-TTS VoiceDesign absent du cache HF");
        return Ok(());
    };
    let rust = TtsModel::load_local(model_dir)?;
    let rust_audio = rust.synthesize_greedy(TEXT, MAX_FRAMES)?;

    let (_, golden_samples) = crate::golden::read_f32("voicedesign_e2e_samples")?;
    let (codes_shape, codes_flat) = crate::golden::read_i32("voicedesign_e2e_codes")?;
    let frames = codes_shape[0];
    let groups = codes_shape[1];
    let golden_codes: Vec<Vec<i32>> = (0..frames)
        .map(|frame| codes_flat[frame * groups..(frame + 1) * groups].to_vec())
        .collect();

    let common = rust_audio.samples.len().min(golden_samples.len());
    if common == 0 {
        return Err(InferError::Dimension("audio TTS e2e vide".to_string()));
    }
    let max_abs = rust_audio
        .samples
        .iter()
        .zip(golden_samples.iter())
        .take(common)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f32, f32::max);
    let rms = (rust_audio
        .samples
        .iter()
        .zip(golden_samples.iter())
        .take(common)
        .map(|(left, right)| {
            let diff = left - right;
            diff * diff
        })
        .sum::<f32>()
        / common as f32)
        .sqrt();

    assert_eq!(rust_audio.sample_rate, 24_000);
    assert_eq!(rust_audio.samples.len(), golden_samples.len());
    assert_eq!(rust_audio.codes.first(), golden_codes.first());
    assert_eq!(
        &rust_audio.codes[1][..PROBE_GROUP],
        &golden_codes[1][..PROBE_GROUP]
    );
    assert!(
        max_abs <= MAX_ABS_TOLERANCE,
        "drift audio TTS max_abs={max_abs} > {MAX_ABS_TOLERANCE}"
    );
    assert!(
        rms <= RMS_TOLERANCE,
        "drift audio TTS rms={rms} > {RMS_TOLERANCE}"
    );
    Ok(())
}

/// Inputs ICL clone metal-rs ≡ golden mlx-rs figé (sans mlx-rs). Mêmes critères :
/// ids identiques + `max_abs<=0.18` sur input/trailing/tts_pad.
#[test]
#[ignore = "golden: charge Qwen3-TTS Base clone (cache HF) pour prepare_icl metal-rs"]
fn golden_clone_icl_inputs_matches_fixture() -> Result<()> {
    const TEXT: &str = "Bonjour, ceci est un test.";
    const MAX_ABS_TOLERANCE: f32 = 0.18;

    let Some(model_dir) = local_tts_snapshot(
        "RETI_QWEN3_TTS_BASE_DIR",
        "models--mlx-community--Qwen3-TTS-12Hz-1.7B-Base-6bit",
    ) else {
        eprintln!("skip: snapshot Qwen3-TTS Base absent du cache HF");
        return Ok(());
    };
    let (wav_bytes, ref_text) = clone_reference_assets()?;
    let rust = TtsModel::load_clone_local(&model_dir, &wav_bytes, &ref_text)?;
    let prepared = rust.prepare_inputs(TEXT)?;
    let rust_ids = clone_icl_rust_ids(&rust, TEXT)?;

    let (_, golden_ids) = crate::golden::read_i32("clone_icl_ids")?;
    let (_, golden_input) = crate::golden::read_f32("clone_icl_input")?;
    let (_, golden_trailing) = crate::golden::read_f32("clone_icl_trailing")?;
    let (_, golden_tts_pad) = crate::golden::read_f32("clone_icl_tts_pad")?;

    assert_eq!(rust_ids, golden_ids);
    let input_max_abs = max_abs_same_len(prepared.input.data(), &golden_input)?;
    let trailing_max_abs = max_abs_same_len(prepared.trailing.data(), &golden_trailing)?;
    let tts_pad_max_abs = max_abs_same_len(prepared.tts_pad.data(), &golden_tts_pad)?;
    assert!(
        input_max_abs <= MAX_ABS_TOLERANCE,
        "drift input ICL max_abs={input_max_abs} > {MAX_ABS_TOLERANCE}"
    );
    assert!(
        trailing_max_abs <= MAX_ABS_TOLERANCE,
        "drift trailing ICL max_abs={trailing_max_abs} > {MAX_ABS_TOLERANCE}"
    );
    assert!(
        tts_pad_max_abs <= MAX_ABS_TOLERANCE,
        "drift tts_pad ICL max_abs={tts_pad_max_abs} > {MAX_ABS_TOLERANCE}"
    );
    Ok(())
}

/// Premier logit clone metal-rs ≡ golden mlx-rs figé (sans mlx-rs). Mêmes critères :
/// cb0 identique + `max_abs<=0.56`.
#[test]
#[ignore = "golden: charge Qwen3-TTS Base clone (cache HF) pour le 1er cb0 metal-rs"]
fn golden_clone_first_cb0_matches_fixture() -> Result<()> {
    const TEXT: &str = "Bonjour, ceci est un test.";
    const MAX_ABS_TOLERANCE: f32 = 0.56;

    let Some(model_dir) = local_tts_snapshot(
        "RETI_QWEN3_TTS_BASE_DIR",
        "models--mlx-community--Qwen3-TTS-12Hz-1.7B-Base-6bit",
    ) else {
        eprintln!("skip: snapshot Qwen3-TTS Base absent du cache HF");
        return Ok(());
    };
    let (wav_bytes, ref_text) = clone_reference_assets()?;
    let rust = TtsModel::load_clone_local(&model_dir, &wav_bytes, &ref_text)?;
    let (_prefix, rust_logits) = rust.probe_greedy_logits(TEXT, 1, 0, 0)?;
    let cfg = &rust.assets.model_config.talker_config;
    let suppress_start = cfg.vocab_size.checked_sub(1024).ok_or_else(|| {
        InferError::Config(format!(
            "vocab TTS trop petit pour suppression: {}",
            cfg.vocab_size
        ))
    })?;
    let suppress = (suppress_start..cfg.vocab_size)
        .filter(|token| *token != cfg.codec_eos_token_id)
        .collect::<Vec<_>>();
    let rust_cb0 = greedy_talker_token(&rust_logits, &suppress)?;

    let (_, golden_logits) = crate::golden::read_f32("clone_first_cb0_logits")?;
    let (_, golden_token) = crate::golden::read_i32("clone_first_cb0_token")?;
    let max_abs = max_abs_same_len(&rust_logits, &golden_logits)?;
    assert_eq!(rust_cb0, golden_token[0]);
    assert!(
        max_abs <= MAX_ABS_TOLERANCE,
        "drift first cb0 logits max_abs={max_abs} > {MAX_ABS_TOLERANCE}"
    );
    Ok(())
}

/// Pipeline clone e2e metal-rs ≡ golden mlx-rs figé (sans mlx-rs). Mêmes tolérances :
/// `max_abs<=0.95`, `rms<=0.25`.
#[test]
#[ignore = "golden: charge Qwen3-TTS Base clone (cache HF) pour la synthèse metal-rs"]
fn golden_clone_e2e_audio_matches_fixture() -> Result<()> {
    const TEXT: &str = "Bonjour, ceci est un test.";
    const MAX_FRAMES: usize = 2;
    const MAX_ABS_TOLERANCE: f32 = 0.95;
    const RMS_TOLERANCE: f32 = 0.25;

    let Some(model_dir) = local_tts_snapshot(
        "RETI_QWEN3_TTS_BASE_DIR",
        "models--mlx-community--Qwen3-TTS-12Hz-1.7B-Base-6bit",
    ) else {
        eprintln!("skip: snapshot Qwen3-TTS Base absent du cache HF");
        return Ok(());
    };
    let (wav_bytes, ref_text) = clone_reference_assets()?;
    let rust = TtsModel::load_clone_local(&model_dir, &wav_bytes, &ref_text)?;
    let rust_audio = rust.synthesize_greedy(TEXT, MAX_FRAMES)?;

    let (_, golden_samples) = crate::golden::read_f32("clone_e2e_samples")?;
    let common = rust_audio.samples.len().min(golden_samples.len());
    if common == 0 {
        return Err(InferError::Dimension("audio clone e2e vide".to_string()));
    }
    let max_abs = rust_audio
        .samples
        .iter()
        .zip(golden_samples.iter())
        .take(common)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f32, f32::max);
    let rms = (rust_audio
        .samples
        .iter()
        .zip(golden_samples.iter())
        .take(common)
        .map(|(left, right)| {
            let diff = left - right;
            diff * diff
        })
        .sum::<f32>()
        / common as f32)
        .sqrt();
    assert!(
        max_abs <= MAX_ABS_TOLERANCE,
        "drift clone max_abs={max_abs} > {MAX_ABS_TOLERANCE}"
    );
    assert!(
        rms <= RMS_TOLERANCE,
        "drift clone rms={rms} > {RMS_TOLERANCE}"
    );
    Ok(())
}

#[test]
#[ignore = "live: charge le snapshot Qwen3-TTS Base pour diagnostiquer le cap de frames clone"]
fn live_clone_generation_diagnoses_frame_cap() -> Result<()> {
    const TEXT: &str = "Bonjour, ceci est un test.";
    const MAX_FRAMES: usize = 2;

    let Some(model_dir) = local_tts_snapshot(
        "RETI_QWEN3_TTS_BASE_DIR",
        "models--mlx-community--Qwen3-TTS-12Hz-1.7B-Base-6bit",
    ) else {
        eprintln!("skip: snapshot Qwen3-TTS Base absent du cache HF");
        return Ok(());
    };
    let wav = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../voices/reti-fr.wav");
    let txt = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../voices/reti-fr.txt");
    let wav_bytes = std::fs::read(&wav).map_err(|source| InferError::Io {
        path: wav.clone(),
        source,
    })?;
    let ref_text = std::fs::read_to_string(&txt).map_err(|source| InferError::Io {
        path: txt.clone(),
        source,
    })?;
    let rust = TtsModel::load_clone_local(&model_dir, &wav_bytes, &ref_text)?;
    let prepared = rust.prepare_inputs(TEXT)?;
    let ref_frames = rust.clone_ctx.as_ref().map_or(0, |ctx| ctx.ref_codes.len());
    let codes = rust.generate_codes_greedy_trace(TEXT, MAX_FRAMES, true, &mut |_| Ok(()))?;
    let reached_eos = codes.len() < MAX_FRAMES;
    let metrics = format!(
        "text={TEXT:?}\nrequested_max_frames={MAX_FRAMES}\nhard_cap={CLONE_GENERATION_HARD_CAP}\neffective_max_frames={}\ninput_rows={}\ntrailing_rows={}\nref_frames={ref_frames}\ngenerated_frames={}\nreached_eos={reached_eos}\nfirst_frame={:?}\n",
        MAX_FRAMES.min(CLONE_GENERATION_HARD_CAP),
        prepared.input.shape()[0],
        prepared.trailing.shape()[0],
        codes.len(),
        codes.first()
    );
    eprint!("qwen3-tts clone generation diagnostic:\n{metrics}");
    std::fs::write("/tmp/qwen3_tts_clone_generation_diag.txt", &metrics).map_err(|source| {
        InferError::Io {
            path: PathBuf::from("/tmp/qwen3_tts_clone_generation_diag.txt"),
            source,
        }
    })?;
    assert!(
        codes.len() <= MAX_FRAMES.min(CLONE_GENERATION_HARD_CAP),
        "le clone a dépassé le cap de frames"
    );
    Ok(())
}

#[test]
#[ignore = "live: charge le contrat header-only d'un snapshot Qwen3-TTS Base"]
fn live_loads_base_snapshot_contract() -> Result<()> {
    let Some(model_dir) = local_tts_snapshot(
        "RETI_QWEN3_TTS_BASE_DIR",
        "models--mlx-community--Qwen3-TTS-12Hz-1.7B-Base-6bit",
    ) else {
        eprintln!("skip: snapshot Qwen3-TTS Base absent du cache HF");
        return Ok(());
    };
    let assets = TtsAssets::load_local(model_dir)?;
    assert_eq!(assets.model_kind(), TtsModelKind::Base);
    assert!(assets.clone_capable());
    assert!(assets.talker_catalog.has_speaker_encoder_weights);
    assert!(assets.codec_catalog.has_encoder_weights);
    Ok(())
}
