//! Lecture stricte des références externes, partagée avec les tests de serve.
use serde::de::DeserializeOwned;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

pub type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

#[derive(Deserialize)]
pub struct Header {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub source: serde_json::Value,
    pub payload_sha256: String,
}

pub fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/goldens")
}

pub fn read(name: &str) -> TestResult<(Header, Vec<u8>)> {
    let stem = root().join(name);
    let header: Header = serde_json::from_slice(&std::fs::read(stem.with_extension("json"))?)?;
    let bytes = std::fs::read(stem.with_extension("bin"))?;
    let width = match header.dtype.as_str() {
        "f32" => 4,
        "utf8" => 1,
        other => return Err(format!("dtype externe inconnu: {other}").into()),
    };
    let count = header
        .shape
        .iter()
        .try_fold(width, |n: usize, d| n.checked_mul(*d));
    assert_eq!(count, Some(bytes.len()), "{name}: taille/shape");
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        header.payload_sha256,
        "{name}: SHA256"
    );
    assert!(header.source["model"].is_string());
    assert!(header.source["path"]
        .as_str()
        .is_some_and(|p| p.starts_with("${")));
    // Le snapshot TTS utilise vocab/merges et tokenizer_config, sans tokenizer.json.
    let tokenizer = if header.source["sha256"]["tokenizer.json"].is_string() {
        "tokenizer.json"
    } else {
        "tokenizer_config.json"
    };
    for file in ["config.json", tokenizer] {
        assert_eq!(
            header.source["sha256"][file].as_str().map(str::len),
            Some(64)
        );
    }
    assert!(!header.source["versions"]
        .as_object()
        .ok_or("versions absentes")?
        .is_empty());
    Ok((header, bytes))
}

pub fn json<T: DeserializeOwned>(name: &str) -> TestResult<T> {
    let (header, bytes) = read(name)?;
    assert_eq!(header.dtype, "utf8");
    Ok(serde_json::from_slice(&bytes)?)
}

pub fn check_model(name: &str, path: &Path) -> TestResult {
    let (header, _) = read(name)?;
    for (file, expected) in header.source["sha256"].as_object().ok_or("SHA absents")? {
        let digest = format!("{:x}", Sha256::digest(std::fs::read(path.join(file))?));
        assert_eq!(
            Some(digest.as_str()),
            expected.as_str(),
            "{name}/{file}: provenance différente"
        );
    }
    Ok(())
}

/// Sans assets locaux, saute la comparaison sauf en mode réel obligatoire.
pub fn local(env: &str) -> TestResult<Option<PathBuf>> {
    local_path(
        env,
        std::env::var_os(env),
        std::env::var("SARAGOSSA_REQUIRE_REAL_MODELS").as_deref() == Ok("1"),
    )
}

fn local_path(
    env: &str,
    value: Option<std::ffi::OsString>,
    required: bool,
) -> TestResult<Option<PathBuf>> {
    let path = value
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .filter(|path| path.is_dir());
    if path.is_none() {
        let reason =
            format!("{env}: répertoire local absent, vide ou introuvable (aucun téléchargement)");
        if required {
            return Err(format!("SARAGOSSA_REQUIRE_REAL_MODELS=1: {reason}").into());
        }
        eprintln!("skip: {reason}");
    }
    Ok(path)
}

#[test]
fn local_assets_skip_or_fail_explicitly() -> TestResult {
    let directory = tempfile::tempdir()?;
    let missing = directory.path().join("absent").into_os_string();
    let file = directory.path().join("file");
    std::fs::write(&file, [])?;
    for value in [
        None,
        Some("".into()),
        Some(missing),
        Some(file.into_os_string()),
    ] {
        assert!(local_path("TEST_MODEL", value.clone(), false)?.is_none());
        let error = local_path("TEST_MODEL", value, true).expect_err("assets absents");
        assert!(error
            .to_string()
            .contains("SARAGOSSA_REQUIRE_REAL_MODELS=1"));
    }
    for required in [false, true] {
        assert_eq!(
            local_path(
                "TEST_MODEL",
                Some(directory.path().as_os_str().to_owned()),
                required
            )?,
            Some(directory.path().to_path_buf())
        );
    }
    Ok(())
}

/// Contrôle les identifiants exacts, y compris la disparition d'un écart.
pub fn assert_known_differences(actual: &[String], expected: &[&str]) {
    let mut actual = actual.to_vec();
    actual.sort();
    let mut expected = expected.to_vec();
    expected.sort();
    assert_eq!(
        actual, expected,
        "écarts modifiés : examiner puis mettre à jour le suivi, jamais les références"
    );
}

pub const ROLE_ALTERNATION_ERROR: &str =
    "TemplateError: Conversation roles must alternate user/assistant/user/assistant/...";

#[derive(Deserialize)]
pub struct ChatFixture {
    pub model: String,
    pub cases: Vec<ChatCase>,
    pub unavailable: Option<String>,
}

#[derive(Deserialize)]
pub struct ChatCase {
    pub name: String,
    pub messages: Box<serde_json::value::RawValue>,
    pub tools: Box<serde_json::value::RawValue>,
    pub add_generation_prompt: bool,
    pub enable_thinking: Option<bool>,
    pub rendered: Option<String>,
    pub token_ids: Option<Vec<u32>>,
    pub error: Option<String>,
}

pub const CHAT_MODELS: [(&str, &str); 4] = [
    ("qwen38", "SARAGOSSA_QWEN38_DIR"),
    ("qwen36_moe", "SARAGOSSA_QWEN36_MOE_DIR"),
    ("gemma3", "SARAGOSSA_GEMMA3_DIR"),
    ("gemma4", "SARAGOSSA_GEMMA4_DIR"),
];

pub fn difference(label: &str, actual: &str, expected: &str) -> Option<String> {
    if actual == expected {
        return None;
    }
    let offset = actual
        .bytes()
        .zip(expected.bytes())
        .position(|(a, b)| a != b)
        .unwrap_or(actual.len().min(expected.len()));
    Some(format!(
        "{label}: premier octet divergent={offset}, Rust={actual:?}, référence={expected:?}"
    ))
}
