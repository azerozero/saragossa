//! Encodage et décodage des tokens via tokenizers.

use crate::{InferError, Result};
use std::io::Write;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

mod incremental;
pub use incremental::ByteLevelDecodeStream;

#[cfg(test)]
pub(crate) mod golden_tests;

#[derive(Clone)]
pub struct RustTokenizer {
    path: PathBuf,
    inner: Tokenizer,
    qwen_profile: Option<crate::chat_template::QwenProfile>,
}

impl std::fmt::Debug for RustTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RustTokenizer")
            .field("path", &self.path)
            .field("vocab_size", &self.vocab_size())
            .finish()
    }
}

impl RustTokenizer {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let inner = Tokenizer::from_file(path).map_err(|e| InferError::Tokenizer {
            path: path.to_path_buf(),
            message: e.to_string(),
        })?;
        let template = std::fs::read_to_string(path.with_file_name("chat_template.jinja"))
            .ok()
            .filter(|text| !text.trim().is_empty())
            .or_else(|| {
                let text =
                    std::fs::read_to_string(path.with_file_name("tokenizer_config.json")).ok()?;
                let config: serde_json::Value = serde_json::from_str(&text).ok()?;
                config["chat_template"].as_str().map(str::to_owned)
            });
        if let Some(message) = foreign_regex_warning(path, &inner) {
            warn_once(path, &message);
        }
        Ok(Self {
            path: path.to_path_buf(),
            qwen_profile: template
                .as_deref()
                .and_then(crate::chat_template::QwenProfile::from_template),
            inner,
        })
    }

    /// Tokenizer depuis un `tokenizer.json` déjà lu et ajusté, sans profil de
    /// template ; `path` ne sert qu'aux messages d'erreur.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le JSON n'est pas un tokenizer valide.
    pub fn from_json(path: impl AsRef<Path>, json: &str) -> Result<Self> {
        let path = path.as_ref();
        let inner: Tokenizer =
            json.parse()
                .map_err(|e: tokenizers::Error| InferError::Tokenizer {
                    path: path.to_path_buf(),
                    message: e.to_string(),
                })?;
        Ok(Self {
            path: path.to_path_buf(),
            inner,
            qwen_profile: None,
        })
    }

    /// Profil de rendu associé au template local, lu une seule fois au chargement.
    ///
    /// # Errors
    ///
    /// Refuse un profil absent, illisible ou non reconnu, sans repli Qwen 3.6.
    pub fn qwen_profile(&self) -> Result<crate::chat_template::QwenProfile> {
        self.qwen_profile.ok_or_else(|| InferError::Tokenizer {
            path: self.path.clone(),
            message: "profil Qwen inconnu : fournir chat_template.jinja ou tokenizer_config.json avec un chat_template reconnu".into(),
        })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.encode_inner(text, false)
    }

    /// Encode un texte en activant les tokens spéciaux du tokenizer.
    ///
    /// Pour la complétion brute des modèles type Llama qui exigent leur BOS
    /// (`<|begin_of_text|>`) via le post-processor. Le chemin templaté (ChatML
    /// Qwen) garde [`Self::encode`] : ses tokens spéciaux sont déjà littéraux.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le tokenizer rejette le texte.
    pub fn encode_with_special_tokens(&self, text: &str) -> Result<Vec<u32>> {
        self.encode_inner(text, true)
    }

    fn encode_inner(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let encoding =
            self.inner
                .encode(text, add_special_tokens)
                .map_err(|e| InferError::Tokenizer {
                    path: self.path.clone(),
                    message: e.to_string(),
                })?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.inner
            .decode(ids, skip_special_tokens)
            .map_err(|e| InferError::Tokenizer {
                path: self.path.clone(),
                message: e.to_string(),
            })
    }

    /// Ouvre un flux uniquement pour le décodeur ByteLevel pur, sans cleanup.
    /// Les autres décodeurs doivent conserver le décodage complet.
    pub fn byte_level_decode_stream(
        &self,
        skip_special_tokens: bool,
    ) -> Option<ByteLevelDecodeStream<'_>> {
        incremental::new(self, skip_special_tokens)
    }

    /// Décode les bytes visibles d'un token isolé.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le tokenizer rejette le token.
    pub fn decode_token_bytes(&self, id: u32) -> Result<Vec<u8>> {
        if let Some(byte) = self
            .inner
            .id_to_token(id)
            .as_deref()
            .and_then(byte_fallback_token)
        {
            return Ok(vec![byte]);
        }
        self.decode(&[id], true).map(String::into_bytes)
    }

    /// Raw HF surfaces preserve split UTF-8 bytes and SentencePiece spaces.
    pub(crate) fn schema_token_bytes(&self) -> Result<Vec<Vec<u8>>> {
        let serialized = self
            .inner
            .to_string(false)
            .map_err(|error| InferError::Config(format!("json_schema tokenizer: {error}")))?;
        let value: serde_json::Value = serde_json::from_str(&serialized)
            .map_err(|error| InferError::Config(format!("json_schema tokenizer: {error}")))?;
        // WordLevel is used by the small, single-token integration fixtures.
        if value["model"]["type"].as_str() == Some("WordLevel") && value["decoder"].is_null() {
            return (0..self.vocab_size())
                .map(|id| {
                    let id = u32::try_from(id)
                        .map_err(|_| InferError::Config("token id hors u32".into()))?;
                    self.decode_token_bytes(id)
                })
                .collect();
        }
        llguidance::token_bytes_from_tokenizer_json(&value)
            .map_err(|error| InferError::Config(format!("json_schema tokenizer: {error}")))
    }

    #[must_use]
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }
}

/// Regex du pré-tokenizer de `tokenizer.json` des Qwen3.5, 3.6 et 3.8.
pub(crate) const QWEN35_REGEX: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
/// `model_type` des familles livrées avec [`QWEN35_REGEX`].
const QWEN35_MODEL_TYPES: &[&str] = &[
    "qwen3_5",
    "qwen3_5_text",
    "qwen3_5_moe",
    "qwen3_5_moe_text",
    "qwen4_exp",
    "qwen4_exp_text",
];

/// Avertissement si un Qwen3.5/3.6/3.8 porte une autre regex, typiquement
/// celle de `Qwen2Tokenizer` réécrite par un `save_pretrained` de
/// transformers 5.x : les marques combinantes (dévanagari, arabe vocalisé,
/// thaï) sont alors découpées autrement qu'à l'entraînement. Les modèles de
/// décision qui veulent la regex Qwen2 passent par
/// [`RustTokenizer::from_json`] et ne sont pas concernés.
fn foreign_regex_warning(path: &Path, inner: &Tokenizer) -> Option<String> {
    let config = std::fs::read_to_string(path.with_file_name("config.json")).ok()?;
    let config: serde_json::Value = serde_json::from_str(&config).ok()?;
    if !QWEN35_MODEL_TYPES.contains(&config["model_type"].as_str()?) {
        return None;
    }
    let pre = serde_json::to_value(inner.get_pre_tokenizer()).ok()?;
    let found = first_regex(&pre);
    (found != Some(QWEN35_REGEX)).then(|| {
        format!(
            "avertissement : {} porte la regex de pré-tokenizer {}, pas celle de Qwen3.5 \
             attendue pour ce modèle : {QWEN35_REGEX} ; recopier le tokenizer.json d'origine",
            path.display(),
            found.unwrap_or("(aucune)")
        )
    })
}

fn first_regex(value: &serde_json::Value) -> Option<&str> {
    match value {
        serde_json::Value::Object(map) => map
            .get("pattern")
            .and_then(|p| p["Regex"].as_str())
            .or_else(|| map.values().find_map(first_regex)),
        serde_json::Value::Array(items) => items.iter().find_map(first_regex),
        _ => None,
    }
}

fn warn_once(path: &Path, message: &str) {
    warn_once_to(path, message, &mut std::io::stderr());
}

/// Écrit `message` dans `out` à la première alerte pour ce fichier ; renvoie
/// `true` si l'alerte est émise (même si l'écriture échoue).
fn warn_once_to(path: &Path, message: &str, out: &mut impl Write) -> bool {
    static WARNED: std::sync::Mutex<Vec<PathBuf>> = std::sync::Mutex::new(Vec::new());
    let mut warned = WARNED
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    // Un même fichier atteint par deux chemins (lien, `..`) n'alerte qu'une fois.
    let path = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    if warned.contains(&path) {
        return false;
    }
    warned.push(path);
    // NOTE: Un stderr fermé ou plein ne doit pas faire échouer le chargement.
    let _ = writeln!(out, "{message}");
    true
}

fn byte_fallback_token(token: &str) -> Option<u8> {
    let hex = token.strip_prefix("<0x")?.strip_suffix('>')?;
    if hex.len() != 2 {
        return None;
    }
    u8::from_str_radix(hex, 16).ok()
}

#[cfg(test)]
pub(crate) fn save_test_tokenizer(path: &Path) {
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;

    let vocab_path = path.with_file_name("test-vocab.json");
    std::fs::write(&vocab_path, r#"{"<unk>":0,"bonjour":1,"reti":2}"#)
        .expect("invariant: écriture vocab tokenizer");
    let vocab_path = vocab_path
        .to_str()
        .expect("invariant: chemin vocab UTF-8")
        .to_string();
    let model = WordLevel::from_file(&vocab_path, "<unk>".to_string())
        .expect("invariant: modèle WordLevel valide");
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(Whitespace));
    tokenizer
        .save(path, true)
        .expect("invariant: sauvegarde tokenizer");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen_profile_requires_readable_recognized_metadata() {
        use crate::chat_template::QwenProfile;
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("tokenizer.json");
        save_test_tokenizer(&path);
        let load_profile = || {
            RustTokenizer::from_file(&path)
                .expect("tokenizer")
                .qwen_profile()
        };
        assert!(load_profile()
            .expect_err("missing profile")
            .to_string()
            .contains("profil Qwen inconnu"));
        let template = tmp.path().join("chat_template.jinja");
        std::fs::create_dir(&template).expect("unreadable template file");
        assert!(load_profile().is_err());
        std::fs::remove_dir(&template).expect("remove directory");
        let config = tmp.path().join("tokenizer_config.json");
        for invalid in [
            "{",
            "{}",
            r#"{"chat_template":"unrecognized"}"#,
            r#"{"description":"Reasoning effort is set to"}"#,
        ] {
            std::fs::write(&config, invalid).expect("write config");
            assert!(load_profile().is_err(), "{invalid}");
        }
        std::fs::write(
            &config,
            r#"{"chat_template":"<|im_start|> Reasoning effort is set to"}"#,
        )
        .expect("write config");
        assert_eq!(load_profile().expect("Qwen38"), QwenProfile::Qwen38);
        std::fs::write(&template, "<|im_start|> enable_thinking last_query_index")
            .expect("write template");
        assert_eq!(load_profile().expect("Qwen36"), QwenProfile::Qwen36);
        std::fs::write(&template, "unknown").expect("write template");
        assert!(load_profile().is_err());
    }

    #[test]
    fn loads_local_tokenizer_and_roundtrips_ids() {
        let tmp = tempfile::tempdir().expect("invariant: tempdir");
        let path = tmp.path().join("tokenizer.json");
        save_test_tokenizer(&path);
        let tokenizer = RustTokenizer::from_file(&path).expect("invariant: tokenizer chargeable");
        let ids = tokenizer
            .encode("bonjour reti")
            .expect("invariant: encode valide");
        assert_eq!(ids, vec![1, 2]);
        assert_eq!(
            tokenizer
                .decode(&ids, false)
                .expect("invariant: decode valide"),
            "bonjour reti"
        );
        assert_eq!(tokenizer.token_to_id("bonjour"), Some(1));
    }

    #[test]
    fn warns_on_foreign_qwen35_regex_only() {
        use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
        use tokenizers::SplitDelimiterBehavior;
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("tokenizer.json");
        save_test_tokenizer(&path);
        let mut inner = Tokenizer::from_file(&path).expect("tokenizer");
        let warning = |inner: &Tokenizer, model_type: &str| {
            std::fs::write(
                tmp.path().join("config.json"),
                format!(r#"{{"model_type":"{model_type}"}}"#),
            )
            .expect("config");
            foreign_regex_warning(&path, inner)
        };
        // Sans regex (Whitespace) ou avec celle de Qwen2 : avertissement.
        assert!(warning(&inner, "qwen3_5").is_some());
        let qwen2 = crate::decision::letters::QWEN2_REGEX;
        for (regex, warns) in [(qwen2, true), (QWEN35_REGEX, false)] {
            let split = Split::new(
                SplitPattern::Regex(regex.into()),
                SplitDelimiterBehavior::Isolated,
                false,
            )
            .expect("split");
            inner.with_pre_tokenizer(Some(split));
            let message = warning(&inner, "qwen3_5_moe");
            assert_eq!(message.is_some(), warns, "{regex}");
            if let Some(message) = message {
                assert!(message.contains("tokenizer.json") && message.contains(QWEN35_REGEX));
            }
            // Autre famille : jamais d'avertissement.
            assert!(warning(&inner, "qwen3").is_none());
        }
        std::fs::remove_file(tmp.path().join("config.json")).expect("remove");
        assert!(foreign_regex_warning(&path, &inner).is_none());
    }

    #[test]
    fn regex_warning_emitted_once_and_never_fails_loading() {
        use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
        use tokenizers::SplitDelimiterBehavior;
        struct Broken;
        impl Write for Broken {
            fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
                Err(std::io::Error::other("stderr fermé"))
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Err(std::io::Error::other("stderr fermé"))
            }
        }
        let tmp = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir(tmp.path().join("sub")).expect("sub");
        let path = tmp.path().join("tokenizer.json");
        // Émission unique, y compris par un autre chemin vers le même fichier.
        std::fs::write(&path, "").expect("fichier");
        let mut out = Vec::new();
        assert!(warn_once_to(&path, "alerte", &mut out));
        let alias = tmp.path().join("sub/../tokenizer.json");
        assert!(!warn_once_to(&alias, "alerte", &mut out));
        assert_eq!(out, b"alerte\n");
        // Écrivain défaillant : ni panique ni erreur.
        let other = tmp.path().join("sub/tokenizer.json");
        assert!(warn_once_to(&other, "alerte", &mut Broken));
        // Regex Qwen2 sur un qwen3_5 : alerte, mais le chargement aboutit.
        save_test_tokenizer(&path);
        let mut inner = Tokenizer::from_file(&path).expect("tokenizer");
        let split = Split::new(
            SplitPattern::Regex(crate::decision::letters::QWEN2_REGEX.into()),
            SplitDelimiterBehavior::Isolated,
            false,
        )
        .expect("split");
        inner.with_pre_tokenizer(Some(split));
        inner.save(&path, false).expect("save");
        std::fs::write(
            tmp.path().join("config.json"),
            r#"{"model_type":"qwen3_5"}"#,
        )
        .expect("config");
        let tokenizer = RustTokenizer::from_file(&path).expect("chargement poursuivi");
        assert!(foreign_regex_warning(&path, &inner).is_some());
        assert_eq!(tokenizer.token_to_id("reti"), Some(2));
    }

    #[test]
    fn parses_byte_fallback_token_surface() {
        assert_eq!(byte_fallback_token("<0xC3>"), Some(0xC3));
        assert_eq!(byte_fallback_token("<0x0a>"), Some(0x0A));
        assert_eq!(byte_fallback_token("<0xZZ>"), None);
        assert_eq!(byte_fallback_token("bonjour"), None);
    }
}
