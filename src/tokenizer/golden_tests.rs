//! Local tokenizer parity only: no weights, network or model construction.

use super::RustTokenizer;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::fmt::Debug;
use std::path::{Path, PathBuf};

pub(crate) type TestResult = Result<(), Box<dyn std::error::Error + Send + Sync>>;
const FAMILIES: &[&str] = &["qwen3", "gemma3", "gemma4", "whisper", "qwen3-tts"];

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Header {
    tokenizer_sha256: String,
    source: String,
    tokenizers_version: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Golden {
    text: String,
    ids: Vec<u32>,
    ids_special: Vec<u32>,
    decoded: String,
}

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/tokenizer")
        .join(format!("{name}.jsonl"))
}

fn read_goldens(family: &str) -> (Header, Vec<Golden>) {
    let data = std::fs::read_to_string(fixture(family)).expect("invariant: golden lisible");
    let mut lines = data.lines();
    let header: Header = serde_json::from_str(lines.next().expect("invariant: en-tête présent"))
        .expect("invariant: en-tête valide");
    assert_eq!(header.tokenizers_version, "0.22.2");
    assert_eq!(header.tokenizer_sha256.len(), 64);
    assert!(header
        .tokenizer_sha256
        .bytes()
        .all(|c| c.is_ascii_hexdigit()));
    assert!(!header.source.is_empty());
    let rows = lines
        .map(|line| serde_json::from_str(line).expect("invariant: golden valide"))
        .collect();
    (header, rows)
}

fn local_path(family: &str, source: &str) -> Option<PathBuf> {
    let variable = format!(
        "SARAGOSSA_TOKENIZER_{}",
        family.replace('-', "_").to_uppercase()
    );
    if let Some(path) = std::env::var_os(&variable) {
        // An explicit override must reach file/SHA validation, even if missing.
        return Some(PathBuf::from(path));
    }
    let hub = crate::hf_resolve::hf_cache_dir_from_env()?;
    let dir = if let Some(name) = source.strip_prefix("local:") {
        // Qwen's local export has no pinned public repository/revision.
        let mut snapshots = Vec::new();
        for repo in std::fs::read_dir(hub).ok()?.flatten() {
            if repo
                .file_name()
                .to_string_lossy()
                .ends_with(&format!("--{name}"))
            {
                if let Ok(entries) = std::fs::read_dir(repo.path().join("snapshots")) {
                    snapshots.extend(entries.flatten().map(|entry| entry.path()));
                }
            }
        }
        snapshots.sort();
        snapshots
            .into_iter()
            .find(|dir| dir.join("tokenizer.json").is_file())?
    } else {
        let (repo, revision) = source.split_once('@')?;
        hub.join(format!(
            "models--{}/snapshots/{revision}",
            repo.replace('/', "--")
        ))
    };
    if family == "qwen3-tts" {
        // A partially present fixture must fail instead of silently skipping.
        (dir.join("vocab.json").exists() || dir.join("merges.txt").exists()).then_some(dir)
    } else {
        let file = dir.join("tokenizer.json");
        file.is_file().then_some(file)
    }
}

pub(crate) fn check_family(
    family: &str,
    check: impl FnOnce(&Path, &[Golden]) -> TestResult,
) -> TestResult {
    let (header, rows) = read_goldens(family);
    let Some(path) = local_path(family, &header.source) else {
        eprintln!("SKIP tokenizer {family}: fichiers locaux absents");
        return Ok(());
    };
    let files = if family == "qwen3-tts" {
        assert!(
            !path.join("tokenizer.json").exists(),
            "le golden TTS cible le fallback vocab/merges"
        );
        vec![path.join("vocab.json"), path.join("merges.txt")]
    } else {
        vec![path.clone()]
    };
    let mut hash = Sha256::new();
    for file in files {
        hash.update(std::fs::read(file)?);
    }
    assert_eq!(
        format!("{:x}", hash.finalize()),
        header.tokenizer_sha256,
        "{family}: tokenizer différent de la référence ({})",
        header.source
    );
    check(&path, &rows)?;
    eprintln!("PASS tokenizer {family}: {} textes comparés", rows.len());
    Ok(())
}

fn compare<T: PartialEq + Debug>(text: &str, field: &str, expected: &[T], actual: &[T]) -> bool {
    if expected == actual {
        return true;
    }
    let first = expected
        .iter()
        .zip(actual)
        .position(|(a, b)| a != b)
        .unwrap_or(expected.len().min(actual.len()));
    eprintln!("texte={text:?}\n{field}: premier index divergent={first}\nattendu={expected:?}\nobtenu={actual:?}");
    false
}

pub(crate) fn compare_rows(
    rows: &[Golden],
    encode: impl Fn(&str, bool) -> Result<Vec<u32>, Box<dyn std::error::Error + Send + Sync>>,
    decode: impl Fn(&[u32]) -> Result<String, Box<dyn std::error::Error + Send + Sync>>,
) -> TestResult {
    let mut equal = true;
    for row in rows {
        let ids = encode(&row.text, false)?;
        equal &= compare(&row.text, "ids", &row.ids, &ids);
        equal &= compare(
            &row.text,
            "ids_special",
            &row.ids_special,
            &encode(&row.text, true)?,
        );
        let decoded = decode(&ids)?;
        if decoded != row.decoded {
            eprintln!(
                "texte={:?}\ndecoded attendu={:?}\nobtenu={decoded:?}",
                row.text, row.decoded
            );
            equal &= compare(
                &row.text,
                "decoded (caractères)",
                &row.decoded.chars().collect::<Vec<_>>(),
                &decoded.chars().collect::<Vec<_>>(),
            );
        }
    }
    assert!(
        equal,
        "écarts tokenizer Rust/HF ci-dessus; ne pas réécrire le golden"
    );
    Ok(())
}

pub(crate) fn check_rust(family: &str) -> TestResult {
    check_family(family, |path, rows| {
        let tokenizer = RustTokenizer::from_file(path)?;
        compare_rows(
            rows,
            |text, special| {
                Ok(if special {
                    tokenizer.encode_with_special_tokens(text)?
                } else {
                    tokenizer.encode(text)?
                })
            },
            |ids| Ok(tokenizer.decode(ids, true)?),
        )
    })
}

#[test]
fn qwen3() -> TestResult {
    check_rust("qwen3")
}
#[test]
fn gemma3() -> TestResult {
    check_rust("gemma3")
}
#[test]
fn gemma4() -> TestResult {
    check_rust("gemma4")
}

#[test]
fn explicit_missing_override_fails_but_automatic_absence_skips() -> TestResult {
    let temp = tempfile::tempdir()?;
    let missing = temp.path().join("missing");
    let executable = std::env::current_exe()?;
    for (family, test) in [
        ("QWEN3", "tokenizer::golden_tests::qwen3"),
        ("GEMMA3", "tokenizer::golden_tests::gemma3"),
        ("GEMMA4", "tokenizer::golden_tests::gemma4"),
        ("WHISPER", "whisper::decoder::golden_tests::whisper"),
        ("QWEN3_TTS", "tts::tokenizer::golden_tests::qwen3_tts"),
    ] {
        let variable = format!("SARAGOSSA_TOKENIZER_{family}");
        // Child processes isolate overrides from concurrent tests.
        let mut command = std::process::Command::new(&executable);
        command
            .args([test, "--exact", "--nocapture"])
            .env("HF_HUB_CACHE", &missing)
            .env_remove(&variable);
        let skipped = command.output()?;
        assert!(skipped.status.success(), "{test}: {skipped:?}");
        assert!(String::from_utf8_lossy(&skipped.stderr).contains("SKIP tokenizer"));

        let failed = command.env(&variable, &missing).output()?;
        assert!(!failed.status.success(), "{variable}: {failed:?}");
        assert!(String::from_utf8_lossy(&failed.stdout).contains("1 failed"));
        assert!(!String::from_utf8_lossy(&failed.stderr).contains("SKIP tokenizer"));
    }
    Ok(())
}

#[test]
fn fixtures_match_corpus_and_rust_chat_templates() {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Text {
        text: String,
    }
    let corpus: Vec<String> = std::fs::read_to_string(fixture("corpus"))
        .expect("invariant: corpus lisible")
        .lines()
        .map(|line| {
            serde_json::from_str::<Text>(line)
                .expect("invariant: corpus valide")
                .text
        })
        .collect();
    assert!(!corpus.is_empty());
    for family in FAMILIES {
        let (_, rows) = read_goldens(family);
        assert_eq!(
            rows.iter().map(|row| &row.text).collect::<Vec<_>>(),
            corpus.iter().collect::<Vec<_>>(),
            "{family}"
        );
    }
    use crate::chat_template::{
        render_gemma4_chat, render_gemma_chat, render_qwen_chatml, ChatTemplateMessage,
    };
    let messages = [
        ChatTemplateMessage::new("system", "Réponds en français."),
        ChatTemplateMessage::new("user", "  Bonjour 👩‍💻  "),
        ChatTemplateMessage::new("assistant", "Salut !\n"),
        ChatTemplateMessage::new("user", "Écris café en Python.\n"),
    ];
    let rendered = [
        render_qwen_chatml(&messages, true, false),
        render_gemma_chat(&messages, true),
        render_gemma4_chat(&messages, true, false),
    ];
    assert!(corpus.len() >= rendered.len());
    assert_eq!(&corpus[corpus.len() - rendered.len()..], &rendered);
}
