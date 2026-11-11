use super::*;
use crate::golden::external::*;
use serde::Deserialize;
use sha2::{Digest, Sha256};

#[derive(Deserialize)]
struct Fixture {
    language: String,
    prefix: Vec<u32>,
    tokens: Vec<u32>,
    text: String,
    segments: Vec<Segment>,
    timestamp_tokens: Vec<u32>,
}

#[derive(Deserialize)]
struct Segment {
    start: f64,
    end: f64,
    tokens: Vec<u32>,
    text: String,
}

fn check_audio(stem: &str) -> TestResult {
    let (header, _) = read(stem)?;
    let audio = std::fs::read(crate::golden::dir().join("whisper_samples16k.bin"))?;
    assert_eq!(
        header.source["audio"]["sha256"],
        format!("{:x}", Sha256::digest(audio))
    );
    Ok(())
}

#[test]
fn external_whisper_fixtures_parse() -> TestResult {
    for name in ["whisper_tiny", "whisper_small"] {
        let stem = format!("whisper/{name}");
        let fixture: Fixture = json(&stem)?;
        assert_eq!(fixture.language, "fr");
        assert_eq!(fixture.prefix.len(), 4);
        assert!(!fixture.tokens.is_empty() && !fixture.text.is_empty());
        assert!(!fixture.segments.is_empty() && !fixture.timestamp_tokens.is_empty());
        let mut previous_end = 0.0;
        for segment in fixture.segments {
            assert!(segment.start.is_finite() && segment.end.is_finite());
            assert!(segment.start >= previous_end - 0.02 && segment.end >= segment.start);
            assert!(segment.end <= 30.0);
            for timestamp in [segment.start, segment.end] {
                assert!((timestamp / 0.02 - (timestamp / 0.02).round()).abs() < 0.001);
            }
            assert!(!segment.tokens.is_empty() && !segment.text.is_empty());
            previous_end = segment.end;
        }
        check_audio(&stem)?;
    }
    Ok(())
}

fn compare(name: &str, env: &str) -> TestResult {
    let Some(path) = local(env) else {
        return Ok(());
    };
    let stem = format!("whisper/{name}");
    check_model(&stem, &path)?;
    check_audio(&stem)?;
    let fixture: Fixture = json(&stem)?;
    let (_, samples) = crate::golden::read_f32("whisper_samples16k")?;
    let model = WhisperModel::from_model_dir(path)?;
    let runtime = ForwardRuntime::cpu();
    let features = model.encoder.encode_samples(&samples, runtime)?;
    let mut prefix = vec![model.sot_token];
    prefix.push(
        model
            .language_token(&fixture.language)
            .ok_or("langue absente")?,
    );
    prefix.extend([model.transcribe_token, model.no_timestamps_token]);
    assert_eq!(prefix, fixture.prefix);
    let tokens =
        model
            .decoder
            .generate_greedy(&features, &prefix, 128, model.eot_token, runtime)?;
    let text = model.tokenizer.decode(&tokens, true)?.trim().to_string();
    eprintln!(
        "{name}: Rust={text:?} tokens={tokens:?}; référence={:?} tokens={:?}",
        fixture.text, fixture.tokens
    );
    eprintln!("{name}: horodatages non comparables: transcribe utilise notimestamps et ne retourne pas de segments temporels (tolérance cible 0.02s)");
    assert!(
        text == fixture.text && tokens == fixture.tokens,
        "{name}: texte ou tokens divergents (voir diagnostic)"
    );
    Ok(())
}

#[test]
#[ignore = "CPU réel: SARAGOSSA_WHISPER_TINY_DIR"]
fn external_whisper_tiny_matches_transformers() -> TestResult {
    compare("whisper_tiny", "SARAGOSSA_WHISPER_TINY_DIR")
}

#[test]
#[ignore = "CPU réel: SARAGOSSA_WHISPER_SMALL_DIR"]
fn external_whisper_small_matches_transformers() -> TestResult {
    compare("whisper_small", "SARAGOSSA_WHISPER_SMALL_DIR")
}
