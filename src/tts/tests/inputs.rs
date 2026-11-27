use super::*;

#[test]
#[ignore = "inférence: snapshots TTS locaux et verrou GPU requis"]
fn default_instruction_only_conditions_voicedesign() -> Result<()> {
    for (env, repo, has_default) in [
        (
            "SARAGOSSA_QWEN3_TTS_CUSTOMVOICE_DIR",
            "models--mlx-community--Qwen3-TTS-12Hz-1.7B-CustomVoice-6bit",
            false,
        ),
        (
            "SARAGOSSA_QWEN3_TTS_VOICEDESIGN_DIR",
            "models--mlx-community--Qwen3-TTS-12Hz-1.7B-VoiceDesign-6bit",
            true,
        ),
    ] {
        let path = local_tts_snapshot(env, repo)
            .ok_or_else(|| InferError::Config(format!("snapshot requis: {repo}")))?;
        let mut model = TtsModel::load_local(path)?;
        assert_eq!(!model.voice_design_instruction.is_empty(), has_default);
        let text = "Bonjour, comment allez-vous ?";
        let default = model.prepare_inputs(text)?;
        model = model.with_voice_design_instruction("");
        let bare = model.prepare_inputs(text)?;
        model = model.with_voice_design_instruction(DEFAULT_INSTRUCT);
        let instructed = model.prepare_inputs(text)?;
        let ids = model.encode_ids(&format!("<|im_start|>user\n{DEFAULT_INSTRUCT}<|im_end|>\n"))?;
        let prefix = model.text_embed(&ids)?;
        assert_eq!(
            instructed.input.shape()[0],
            bare.input.shape()[0] + ids.len()
        );
        assert_eq!(&instructed.input.data()[..prefix.len()], prefix.data());
        assert_eq!(&instructed.input.data()[prefix.len()..], bare.input.data());
        assert_eq!(instructed.trailing.data(), bare.trailing.data());
        assert_eq!(instructed.tts_pad.data(), bare.tts_pad.data());
        let expected = if has_default { &instructed } else { &bare };
        assert_eq!(default.input.data(), expected.input.data());
        if !has_default {
            model = model.with_voice_design_instruction("");
            model.set_voice(Some("ryan"))?;
            let male = model.prepare_inputs(text)?;
            assert_eq!(male.input.shape(), bare.input.shape());
            assert_ne!(male.input.data(), bare.input.data());
        }
    }
    Ok(())
}

/// Portée de requête sur un modèle partagé (`serve --engines N`) : mêmes
/// entrées que `set_voice`, restaurée au retour, propre à chaque thread et au
/// seul modèle visé.
#[test]
#[ignore = "inférence: snapshots TTS locaux et verrou GPU requis"]
fn request_voice_matches_set_voice_per_thread() -> Result<()> {
    let path = local_tts_snapshot(
        "SARAGOSSA_QWEN3_TTS_CUSTOMVOICE_DIR",
        "models--mlx-community--Qwen3-TTS-12Hz-1.7B-CustomVoice-6bit",
    )
    .ok_or_else(|| InferError::Config("snapshot CustomVoice requis".into()))?;
    let mut model = TtsModel::load_local(&path)?;
    let other = TtsModel::load_local(&path)?;
    let text = "Bonjour, comment allez-vous ?";
    let serena = model.prepare_inputs(text)?.input.data().to_vec();
    model.set_voice(Some("ryan"))?;
    let ryan = model.prepare_inputs(text)?.input.data().to_vec();
    model.set_voice(None)?;
    assert_ne!(serena, ryan);
    let (model, serena, ryan) = (&model, &serena, &ryan);
    // Un autre modèle appelé pendant la portée garde sa voix.
    let nested = model.with_request_voice(Some("ryan"), Some(7), || {
        other.prepare_inputs(text).map(|p| p.input.data().to_vec())
    })??;
    assert_eq!(&nested, serena);
    std::thread::scope(|scope| -> Result<()> {
        let handles: Vec<_> = [(Some("ryan"), ryan), (None, serena)]
            .into_iter()
            .map(|(voice, expected)| {
                scope.spawn(move || -> Result<()> {
                    for _ in 0..3 {
                        model.with_request_voice(voice, Some(7), || -> Result<()> {
                            assert_eq!(model.prepare_inputs(text)?.input.data(), &expected[..]);
                            assert_eq!(model.effective_seed(), Some(7));
                            Ok(())
                        })??;
                    }
                    assert_eq!(model.prepare_inputs(text)?.input.data(), &serena[..]);
                    assert_eq!(model.effective_seed(), None);
                    Ok(())
                })
            })
            .collect();
        for handle in handles {
            handle
                .join()
                .map_err(|_| InferError::Config("thread voix en panique".into()))??;
        }
        Ok(())
    })?;
    assert!(model
        .with_request_voice(Some("inconnue"), None, || ())
        .is_err());
    assert_eq!(model.prepare_inputs(text)?.input.data(), &serena[..]);
    Ok(())
}
