//! Bench ADDITIF du décodeur Whisper STT (chemin `src/stt_rust.rs`, metal-rs).
//!
//! Charge le modèle + l'exécuteur Metal UNE fois, jette un warm-up, puis mesure
//! la **médiane** de N transcriptions sur un WAV. `rtf = compute / durée_audio`.
//! Aucune surface de prod touchée. Mesure GPU idle uniquement.
//!
//! Usage :
//!   RETI_WHISPER_DIR=<dir> cargo run -p saragossa --release \
//!     --features metal --example bench_stt -- [wav] [runs]

use std::path::PathBuf;
use std::time::Instant;

use saragossa::{ForwardRuntime, WhisperModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let wav = args.next().map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../voices/reti-fr.wav"),
        PathBuf::from,
    );
    let runs: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(5);

    let model_dir = whisper_dir().ok_or("définir RETI_WHISPER_DIR vers un dossier Whisper HF")?;
    let lang = std::env::var("RETI_WHISPER_LANG").unwrap_or_else(|_| "fr".to_string());

    let mut samples = load_wav_16k_mono(&wav)?;
    // RETI_REPEAT=N concatène l'audio N fois (toujours tronqué à 30 s par
    // l'encodeur) → montre comment la rtf baisse quand la durée audio amortit le
    // coût FIXE de l'encodeur 30 s.
    if let Ok(repeat) = std::env::var("RETI_REPEAT") {
        if let Ok(n) = repeat.parse::<usize>() {
            if n > 1 {
                samples = samples
                    .iter()
                    .copied()
                    .cycle()
                    .take(samples.len() * n)
                    .collect();
            }
        }
    }
    let secs = samples.len() as f32 / 16_000.0;
    eprintln!(
        "modèle={} | wav={} ({:.2}s, {} échantillons) | lang={lang} | runs={runs}",
        model_dir.display(),
        wav.display(),
        secs,
        samples.len()
    );

    let t_load = Instant::now();
    let model = WhisperModel::from_model_dir(&model_dir)?;
    eprintln!("chargé en {} ms", t_load.elapsed().as_millis());

    #[cfg(feature = "metal")]
    let metal = saragossa::MetalExecutor::new()?;
    #[cfg(feature = "metal")]
    let runtime = ForwardRuntime::metal(&metal);
    #[cfg(not(feature = "metal"))]
    let runtime = ForwardRuntime::cpu();

    // Warm-up jeté (compile pipelines, chauffe caches).
    let (warm_text, _) = model.transcribe(&samples, &lang, runtime)?;
    eprintln!("warm-up: {:?}", warm_text);

    let mut durations = Vec::with_capacity(runs);
    for run in 0..runs {
        let t0 = Instant::now();
        let (text, _) = model.transcribe(&samples, &lang, runtime)?;
        let dt = t0.elapsed().as_secs_f32();
        durations.push(dt);
        eprintln!(
            "run {run}: {:.0} ms | rtf={:.3} | {:?}",
            dt * 1000.0,
            dt / secs.max(0.001),
            text
        );
    }
    durations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = durations[durations.len() / 2];
    eprintln!(
        "\n==> médiane: {:.0} ms | rtf={:.3} (cible < 0.30)",
        median * 1000.0,
        median / secs.max(0.001)
    );
    Ok(())
}

fn whisper_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("RETI_WHISPER_DIR") {
        let dir = PathBuf::from(dir);
        if dir.join("config.json").is_file() {
            return Some(dir);
        }
    }
    let home = std::env::var("HOME").ok()?;
    let base = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--openai--whisper-large-v3-turbo/snapshots");
    std::fs::read_dir(base)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .find(|p| p.join("config.json").is_file())
}

fn load_wav_16k_mono(path: &PathBuf) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let raw: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().filter_map(Result::ok).collect(),
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .filter_map(Result::ok)
                .map(|s| s as f32 / max)
                .collect()
        }
    };
    // Downmix vers mono si nécessaire.
    let mono: Vec<f32> = if spec.channels > 1 {
        let ch = spec.channels as usize;
        raw.chunks(ch)
            .map(|c| c.iter().sum::<f32>() / ch as f32)
            .collect()
    } else {
        raw
    };
    if spec.sample_rate == 16_000 {
        Ok(mono)
    } else {
        // Rééchantillonnage linéaire simple vers 16 kHz (bench only).
        let ratio = 16_000.0 / spec.sample_rate as f32;
        let out_len = (mono.len() as f32 * ratio) as usize;
        Ok((0..out_len)
            .map(|i| {
                let src = i as f32 / ratio;
                let idx = src as usize;
                let frac = src - idx as f32;
                let a = mono.get(idx).copied().unwrap_or(0.0);
                let b = mono.get(idx + 1).copied().unwrap_or(a);
                a + (b - a) * frac
            })
            .collect())
    }
}
