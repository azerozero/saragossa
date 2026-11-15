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
