use super::*;
use std::path::PathBuf;

#[test]
fn reflect_pad_time_mirrors_without_edges() {
    let x = Ncl::new(1, 4, vec![1.0, 2.0, 3.0, 4.0]).expect("invariant: NCL valide");
    let out = x.reflect_pad_time(1);
    assert_eq!(out.data, vec![2.0, 1.0, 2.0, 3.0, 4.0, 3.0]);
}

/// x-vector ECAPA metal-rs ≡ golden mlx-rs figé (sans mlx-rs ; charge Qwen3-TTS Base
/// pour l'encodeur speaker metal-rs). Mêmes tolérances que `live_speaker_*`.
#[test]
#[ignore = "golden: charge Qwen3-TTS Base (cache HF) pour l'encodeur speaker metal-rs"]
fn golden_speaker_xvector_matches_fixture() -> Result<()> {
    const MAX_ABS_TOLERANCE: f32 = 0.000_05;
    const RMS_TOLERANCE: f32 = 0.000_01;

    let Some(model_dir) = local_tts_snapshot(
        "RETI_QWEN3_TTS_BASE_DIR",
        "models--mlx-community--Qwen3-TTS-12Hz-1.7B-Base-6bit",
    ) else {
        eprintln!("skip: snapshot Qwen3-TTS Base absent du cache HF");
        return Ok(());
    };
    let assets = crate::tts::TtsAssets::load_local(&model_dir)?;
    let speaker = TtsSpeakerEncoder::load(&model_dir, &assets.model_config)?;

    let wav = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../voices/reti-fr.wav");
    let bytes = std::fs::read(&wav).map_err(|source| InferError::Io {
        path: wav.clone(),
        source,
    })?;
    let pcm = crate::tts_clone::load_wav_24k(&bytes)?;
    let mel = crate::tts_clone::log_mel_24k(&pcm)?;
    let rust = speaker.embed_mel(&mel)?;

    let (_, golden) = crate::golden::read_f32("speaker_xvector")?;
    if rust.shape() != [1, golden.len()] {
        return Err(InferError::Dimension(format!(
            "x-vector shape rust={:?} golden_len={}",
            rust.shape(),
            golden.len()
        )));
    }
    let max_abs = rust
        .data()
        .iter()
        .zip(golden.iter())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f32, f32::max);
    let rms = (rust
        .data()
        .iter()
        .zip(golden.iter())
        .map(|(left, right)| {
            let diff = left - right;
            diff * diff
        })
        .sum::<f32>()
        / golden.len() as f32)
        .sqrt();
    assert!(
        max_abs <= MAX_ABS_TOLERANCE,
        "drift x-vector max_abs={max_abs} > {MAX_ABS_TOLERANCE}"
    );
    assert!(
        rms <= RMS_TOLERANCE,
        "drift x-vector rms={rms} > {RMS_TOLERANCE}"
    );
    Ok(())
}

fn local_tts_snapshot(env_var: &str, cache_name: &str) -> Option<PathBuf> {
    if let Ok(dir) = std::env::var(env_var) {
        let path = PathBuf::from(dir);
        if path.is_dir() {
            return Some(path);
        }
    }
    let snapshot = crate::hf_resolve::hf_cache_dir_from_env().and_then(|hub| {
        let snapshots = hub.join(cache_name).join("snapshots");
        let mut entries = std::fs::read_dir(snapshots)
            .ok()?
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| path.is_dir())
            .collect::<Vec<_>>();
        entries.sort();
        entries.pop()
    });
    crate::test_support::require_real_model(snapshot, "snapshot Qwen3-TTS Base")
}
