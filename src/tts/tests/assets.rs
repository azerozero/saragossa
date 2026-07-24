use super::*;

#[test]
fn parses_voicedesign_config_defaults() -> Result<()> {
    let cfg: TtsModelConfig = serde_json::from_str(&model_config_json("voice_design", false))
        .map_err(|source| InferError::Json {
            path: PathBuf::from("inline"),
            source,
        })?;
    assert_eq!(cfg.model_kind(), TtsModelKind::VoiceDesign);
    assert_eq!(cfg.talker_config.hidden_act, "silu");
    assert_eq!(
        cfg.talker_config.codec_language_id.get("french").copied(),
        Some(42)
    );
    assert!(cfg.speaker_encoder_config.is_none());
    Ok(())
}

#[test]
fn catalog_detects_clone_capable_weights() {
    let talker = TtsTalkerCatalog::from_keys(vec![
        "speaker_encoder.blocks.0.weight".to_string(),
        "talker.model.text_embedding.weight".to_string(),
    ]);
    let codec = TtsCodecCatalog::from_keys(vec![
        "decoder.model.layers.0.weight".to_string(),
        "encoder.model.layers.0.weight".to_string(),
        "rvq.layers.0._codebook.cluster_usage".to_string(),
        "rvq.layers.0._codebook.embedding_sum".to_string(),
    ]);
    assert!(talker.has_talker_weights);
    assert!(talker.has_speaker_encoder_weights);
    assert!(codec.has_decoder_weights);
    assert!(codec.has_encoder_weights);
    assert!(codec.has_codebook_stats);
}

#[test]
fn loads_local_tts_assets_without_loading_payloads() -> Result<()> {
    let tmp = tempfile::tempdir().map_err(|source| InferError::Io {
        path: PathBuf::from("tempdir"),
        source,
    })?;
    let root = tmp.path();
    let speech = root.join("speech_tokenizer");
    std::fs::create_dir_all(&speech).map_err(|source| InferError::Io {
        path: speech.clone(),
        source,
    })?;
    write(root.join("config.json"), &model_config_json("base", true))?;
    write(root.join("vocab.json"), "{}")?;
    write(root.join("merges.txt"), "#version: 0.2\n")?;
    write(speech.join("config.json"), &codec_config_json(true))?;
    write_safetensors(
        &root.join("model.safetensors"),
        &[
            "speaker_encoder.blocks.0.weight",
            "talker.model.text_embedding.weight",
        ],
    )?;
    write_safetensors(
        &speech.join("model.safetensors"),
        &[
            "decoder.model.layers.0.weight",
            "encoder.model.layers.0.weight",
            "rvq.layers.0._codebook.cluster_usage",
            "rvq.layers.0._codebook.embedding_sum",
        ],
    )?;

    let assets = TtsAssets::load_local(root)?;
    assert_eq!(assets.model_kind(), TtsModelKind::Base);
    assert!(assets.clone_capable());
    assert_eq!(assets.talker_catalog.tensor_count, 2);
    assert_eq!(assets.codec_catalog.tensor_count, 4);
    Ok(())
}

#[test]
fn rejects_missing_talker_weights() -> Result<()> {
    let tmp = tempfile::tempdir().map_err(|source| InferError::Io {
        path: PathBuf::from("tempdir"),
        source,
    })?;
    let root = tmp.path();
    let speech = root.join("speech_tokenizer");
    std::fs::create_dir_all(&speech).map_err(|source| InferError::Io {
        path: speech.clone(),
        source,
    })?;
    write(
        root.join("config.json"),
        &model_config_json("voice_design", false),
    )?;
    write(root.join("vocab.json"), "{}")?;
    write(root.join("merges.txt"), "#version: 0.2\n")?;
    write(speech.join("config.json"), &codec_config_json(false))?;
    write_safetensors(&root.join("model.safetensors"), &["not_talker.weight"])?;
    write_safetensors(
        &speech.join("model.safetensors"),
        &["decoder.model.layers.0.weight"],
    )?;

    let err = TtsAssets::load_local(root).expect_err("invariant: poids talker absents");
    assert!(err.to_string().contains("talker.* weights"));
    Ok(())
}

#[test]
#[ignore = "live: charge le contrat header-only d'un snapshot Qwen3-TTS VoiceDesign"]
fn live_loads_voicedesign_snapshot_contract() -> Result<()> {
    let Some(model_dir) = local_tts_snapshot(
        "RETI_QWEN3_TTS_VOICEDESIGN_DIR",
        "models--mlx-community--Qwen3-TTS-12Hz-1.7B-VoiceDesign-6bit",
    ) else {
        eprintln!("skip: snapshot Qwen3-TTS VoiceDesign absent du cache HF");
        return Ok(());
    };
    let assets = TtsAssets::load_local(model_dir)?;
    assert_eq!(assets.model_kind(), TtsModelKind::VoiceDesign);
    assert!(!assets.clone_capable());
    assert!(assets.talker_catalog.tensor_count > 0);
    assert!(assets.codec_catalog.tensor_count > 0);
    Ok(())
}
