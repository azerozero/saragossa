//! Encodage et décodage des tokens via tokenizers.

use crate::{InferError, Result};
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

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
        Ok(Self {
            path: path.to_path_buf(),
            qwen_profile: template
                .as_deref()
                .and_then(crate::chat_template::QwenProfile::from_template),
            inner,
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

    #[must_use]
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }
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
    fn parses_byte_fallback_token_surface() {
        assert_eq!(byte_fallback_token("<0xC3>"), Some(0xC3));
        assert_eq!(byte_fallback_token("<0x0a>"), Some(0x0A));
        assert_eq!(byte_fallback_token("<0xZZ>"), None);
        assert_eq!(byte_fallback_token("bonjour"), None);
    }
}
