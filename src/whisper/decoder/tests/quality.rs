//! Recettes locales : le PCM long est construit depuis la voix embarquée.
use super::*;

fn voice_16k() -> Vec<f32> {
    let bytes = include_bytes!("../../../../assets/voices/reti-fr.wav");
    let mut wav =
        hound::WavReader::new(std::io::Cursor::new(bytes)).expect("invariant: WAV embarqué valide");
    assert_eq!(wav.spec().channels, 1);
    assert_eq!(wav.spec().bits_per_sample, 16);
    let rate = wav.spec().sample_rate;
    let pcm: Vec<f32> = wav
        .samples::<i16>()
        .map(|s| f32::from(s.expect("invariant: PCM embarqué valide")) / 32768.0)
        .collect();
    crate::tts_clone::resample(&pcm, rate, 16_000)
        .expect("invariant: fréquence WAV embarquée valide")
}

fn long_voice() -> Vec<f32> {
    let voice = voice_16k();
    assert!(voice.len() < 28 * 16_000);
    // Deux fenêtres identiques de 30 s ; chaque pause laisse la coupe à 30 s.
    let mut block = vec![0.0; 16_000];
    block.extend_from_slice(&voice);
    block.resize(30 * 16_000, 0.0);
    block.repeat(2)
}

#[test]
fn long_fixture_covers_every_sample() {
    let pcm = long_voice();
    let ranges = super::super::super::segments::audio_segments(&pcm);
    assert_eq!(ranges, vec![0..480_000, 480_000..960_000]);
    assert_eq!(ranges.iter().map(|r| r.len()).sum::<usize>(), pcm.len());
    for range in ranges {
        assert!(pcm[range].iter().any(|x| x.abs() > 0.01));
    }
}

#[test]
#[ignore = "quality: Whisper local et Metal requis, aucun téléchargement"]
fn quality_whisper_long_audio_keeps_every_chunk() -> Result<()> {
    let path = local_whisper_tiny_dir()
        .ok_or_else(|| InferError::Config("snapshot Whisper local requis".into()))?;
    let model = WhisperModel::from_model_dir(path)?;
    let runtime = live_runtime();
    let pcm = long_voice();
    let (one, language) = model.transcribe(&pcm[..480_000], "fr", runtime)?;
    assert_eq!(language, "fr");
    assert!(
        !one.trim().is_empty(),
        "le témoin parlé doit être transcrit"
    );
    let (all, language) = model.transcribe(&pcm, "fr", runtime)?;
    assert_eq!(language, "fr");
    // Oracle indépendant du découpage : deux occurrences, dans le même ordre.
    assert_eq!(all, format!("{one} {one}"));
    Ok(())
}

#[test]
#[ignore = "quality: Whisper local et Metal requis, aucun téléchargement"]
fn quality_whisper_five_seconds_silence_is_empty() -> Result<()> {
    let path = local_whisper_tiny_dir()
        .ok_or_else(|| InferError::Config("snapshot Whisper local requis".into()))?;
    let model = WhisperModel::from_model_dir(path)?;
    let (text, language) = model.transcribe(&vec![0.0; 5 * 16_000], "fr", live_runtime())?;
    assert_eq!(language, "fr");
    assert!(
        text.trim().is_empty(),
        "hallucination sur silence: {text:?}"
    );
    Ok(())
}
