#![cfg(feature = "stt")]
//! STT en flux sur poids réels : identité batch/flux et mesures sur FLEURS.
//!
//! Ignorés par défaut. `SARAGOSSA_WHISPER_DIR` désigne un snapshot Whisper HF
//! local et `SARAGOSSA_FLEURS_DIR` le jeu `fleurs-wer-v1` (`manifest.json`).
//!
//! ```text
//! SARAGOSSA_WHISPER_DIR=… SARAGOSSA_FLEURS_DIR=… cargo test --release \
//!   --test stt_streaming -- --ignored --nocapture --test-threads=1
//! ```
#![cfg(all(target_os = "macos", feature = "metal"))]

use std::path::PathBuf;
use std::time::Instant;

use saragossa::{ForwardRuntime, MetalExecutor, WhisperModel, WhisperStream};

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

const SAMPLE_RATE: f64 = 16_000.0;

struct Clip {
    id: u64,
    lang: &'static str,
    samples: Vec<f32>,
    reference: String,
}

fn env_dir(name: &str) -> TestResult<PathBuf> {
    std::env::var_os(name)
        .map(PathBuf::from)
        .ok_or_else(|| format!("{name} requis").into())
}

fn load_clips() -> TestResult<Vec<Clip>> {
    let root = env_dir("SARAGOSSA_FLEURS_DIR")?;
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
    let clips = manifest["clips"].as_array().ok_or("manifest sans clips")?;
    clips
        .iter()
        .map(|clip| {
            let file = clip["file"].as_str().ok_or("clip sans fichier")?;
            let samples = saragossa::wire_audio::wav_to_mono_16k(&std::fs::read(root.join(file))?)
                .map_err(|e| e.0)?;
            Ok(Clip {
                id: clip["id"].as_u64().unwrap_or_default(),
                lang: if clip["lang"].as_str() == Some("fr_fr") {
                    "fr"
                } else {
                    "en"
                },
                samples,
                reference: clip["normalized"].as_str().unwrap_or_default().to_string(),
            })
        })
        .collect()
}

fn load_model() -> TestResult<(WhisperModel, MetalExecutor)> {
    let model = WhisperModel::from_model_dir(env_dir("SARAGOSSA_WHISPER_DIR")?)?;
    Ok((model, MetalExecutor::new()?))
}

/// Minuscules, apostrophes conservées, ponctuation remplacée par des espaces.
fn normalize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .replace('’', "'")
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '\'' {
                c
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .map(str::to_owned)
        .collect()
}

fn word_errors(reference: &[String], hypothesis: &[String]) -> usize {
    let mut row: Vec<usize> = (0..=hypothesis.len()).collect();
    for (i, r) in reference.iter().enumerate() {
        let mut diagonal = row[0];
        row[0] = i + 1;
        for (j, h) in hypothesis.iter().enumerate() {
            let substitution = diagonal + usize::from(r != h);
            diagonal = row[j + 1];
            row[j + 1] = substitution.min(row[j] + 1).min(diagonal + 1);
        }
    }
    row[hypothesis.len()]
}

/// Client temps réel simulé : à chaque envoi, tout l'audio arrivé depuis le
/// précédent part en un morceau, au plus tôt toutes les `chunk_s` secondes et
/// jamais avant la fin du traitement précédent ; le dernier part dès la fin de
/// l'audio. Horloge virtuelle = durée
/// audio ; seules les durées de traitement sont mesurées.
struct StreamRun {
    text: String,
    last_hypothesis: String,
    last_committed: String,
    final_latency: f64,
    partial_delays: Vec<f64>,
}

fn run_stream(
    model: &WhisperModel,
    runtime: ForwardRuntime<'_>,
    samples: &[f32],
    lang: &str,
    chunk_s: f64,
) -> TestResult<StreamRun> {
    let duration = samples.len() as f64 / SAMPLE_RATE;
    let mut transcribe = |s: &[f32]| model.transcribe(s, lang, runtime).map(|(text, _)| text);
    let mut stream = WhisperStream::new();
    let (mut sent, mut clock) = (0, chunk_s);
    let mut run = StreamRun {
        text: String::new(),
        last_hypothesis: String::new(),
        last_committed: String::new(),
        final_latency: 0.0,
        partial_delays: Vec::new(),
    };
    loop {
        let available = ((clock * SAMPLE_RATE) as usize).min(samples.len());
        let chunk = &samples[sent..available];
        sent = available;
        let started = Instant::now();
        if available == samples.len() {
            run.text = stream.finish(chunk, &mut transcribe)?;
            run.final_latency = clock + started.elapsed().as_secs_f64() - duration;
            return Ok(run);
        }
        let update = stream.push(chunk, &mut transcribe)?;
        let done = clock + started.elapsed().as_secs_f64();
        run.partial_delays
            .push(done - available as f64 / SAMPLE_RATE);
        run.last_hypothesis = format!("{} {}", update.committed, update.partial);
        run.last_committed = update.committed;
        // La fin de parole (VAD) déclenche l'envoi final sans attendre le pas suivant.
        clock = done.max((clock + chunk_s).min(duration));
    }
}

#[test]
#[ignore = "poids Whisper et FLEURS locaux requis"]
fn fleurs_stream_final_and_deltas_match_batch() -> TestResult {
    let (model, metal) = load_model()?;
    let runtime = ForwardRuntime::metal(&metal);
    for clip in load_clips()? {
        let (batch, _) = model.transcribe(&clip.samples, clip.lang, runtime)?;
        let mut deltas = String::new();
        let (streamed, _) =
            model.transcribe_with_deltas(&clip.samples, clip.lang, runtime, &mut |delta| {
                deltas.push_str(delta);
                Ok(())
            })?;
        assert_eq!(streamed, batch, "clip {}", clip.id);
        assert_eq!(deltas, batch, "deltas clip {}", clip.id);
        let run = run_stream(&model, runtime, &clip.samples, clip.lang, 0.5)?;
        assert_eq!(run.text, batch, "flux d'entrée clip {}", clip.id);
        let verbose = model.transcribe_verbose(&clip.samples, clip.lang, runtime)?;
        assert!(!verbose.segments.is_empty(), "segments clip {}", clip.id);
        assert!(verbose
            .segments
            .windows(2)
            .all(|pair| pair[0].end <= pair[1].start + 1e-3));
        assert!(verbose
            .segments
            .iter()
            .all(|s| s.start <= s.end && s.end <= verbose.duration + 1e-3));
    }
    Ok(())
}

#[test]
#[ignore = "poids Whisper et FLEURS locaux requis"]
fn fleurs_long_stream_matches_batch_across_windows() -> TestResult {
    let (model, metal) = load_model()?;
    let runtime = ForwardRuntime::metal(&metal);
    let clips = load_clips()?;
    for lang in ["en", "fr"] {
        let long: Vec<f32> = clips
            .iter()
            .filter(|clip| clip.lang == lang)
            .take(8)
            .flat_map(|clip| clip.samples.iter().copied())
            .collect();
        let (batch, _) = model.transcribe(&long, lang, runtime)?;
        let run = run_stream(&model, runtime, &long, lang, 1.0)?;
        assert_eq!(run.text, batch, "long {lang}");
    }
    Ok(())
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len().max(1) as f64
}

fn percentile(values: &mut [f64], q: f64) -> f64 {
    values.sort_by(f64::total_cmp);
    values
        .get(((values.len() as f64 - 1.0) * q).round() as usize)
        .copied()
        .unwrap_or(0.0)
}

#[test]
#[ignore = "mesure: poids Whisper et FLEURS locaux requis"]
fn fleurs_stream_latency_and_wer_report() -> TestResult {
    let (model, metal) = load_model()?;
    let runtime = ForwardRuntime::metal(&metal);
    let clips = load_clips()?;
    // Chauffe : pipelines Metal et poids résidents.
    model.transcribe(&clips[0].samples, clips[0].lang, runtime)?;
    let chunk_s: f64 = std::env::var("STT_STREAM_CHUNK_S")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.5);
    let (mut batch_latency, mut final_latency, mut partial) = (vec![], vec![], vec![]);
    let (mut words, mut batch_err, mut stream_err, mut hyp_err) = (0, 0, 0, 0);
    let (mut committed_words, mut revised_words, mut identical) = (0, 0, 0);
    for clip in &clips {
        let started = Instant::now();
        let (batch, _) = model.transcribe(&clip.samples, clip.lang, runtime)?;
        batch_latency.push(started.elapsed().as_secs_f64());
        let run = run_stream(&model, runtime, &clip.samples, clip.lang, chunk_s)?;
        final_latency.push(run.final_latency);
        partial.extend(run.partial_delays.iter().copied());
        identical += usize::from(run.text == batch);
        let reference = normalize(&clip.reference);
        words += reference.len();
        batch_err += word_errors(&reference, &normalize(&batch));
        stream_err += word_errors(&reference, &normalize(&run.text));
        hyp_err += word_errors(&reference, &normalize(&run.last_hypothesis));
        let committed = normalize(&run.last_committed);
        let final_words = normalize(&run.text);
        committed_words += committed.len();
        revised_words += committed
            .iter()
            .zip(final_words.iter().chain(std::iter::repeat(&String::new())))
            .skip_while(|(a, b)| a == b)
            .count();
    }
    // Audio long : 8 clips concaténés par langue, fenêtres de 30 s fermées en flux.
    let mut long_rows = Vec::new();
    for lang in ["en", "fr"] {
        let long: Vec<f32> = clips
            .iter()
            .filter(|clip| clip.lang == lang)
            .take(8)
            .flat_map(|clip| clip.samples.iter().copied())
            .collect();
        let started = Instant::now();
        let (batch, _) = model.transcribe(&long, lang, runtime)?;
        let batch_s = started.elapsed().as_secs_f64();
        let run = run_stream(&model, runtime, &long, lang, chunk_s)?;
        long_rows.push(format!(
            "long {lang} {:.1} s : batch {batch_s:.3} s, flux final {:.3} s, identique {}",
            long.len() as f64 / SAMPLE_RATE,
            run.final_latency,
            run.text == batch
        ));
    }
    let pct = |errors: usize| 100.0 * errors as f64 / words.max(1) as f64;
    println!("chunk {chunk_s} s, {} clips, {words} mots", clips.len());
    println!(
        "latence finale batch: moy {:.3} s p50 {:.3} p95 {:.3}",
        mean(&batch_latency),
        percentile(&mut batch_latency.clone(), 0.5),
        percentile(&mut batch_latency, 0.95)
    );
    println!(
        "latence finale flux : moy {:.3} s p50 {:.3} p95 {:.3}",
        mean(&final_latency),
        percentile(&mut final_latency.clone(), 0.5),
        percentile(&mut final_latency, 0.95)
    );
    println!(
        "délai partiel       : moy {:.3} s p50 {:.3} p95 {:.3} ({} partiels)",
        mean(&partial),
        percentile(&mut partial.clone(), 0.5),
        percentile(&mut partial.clone(), 0.95),
        partial.len()
    );
    println!(
        "WER batch {:.2} %, flux final {:.2} %, dernière hypothèse partielle {:.2} %",
        pct(batch_err),
        pct(stream_err),
        pct(hyp_err)
    );
    println!(
        "textes identiques {identical}/{} ; mots stabilisés révisés {revised_words}/{committed_words}",
        clips.len()
    );
    for row in long_rows {
        println!("{row}");
    }
    Ok(())
}
