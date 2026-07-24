use crate::catalog::read_safetensors_keys;
use crate::Result;
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TtsTalkerCatalog {
    keys: Vec<String>,
    pub tensor_count: usize,
    pub has_talker_weights: bool,
    pub has_speaker_encoder_weights: bool,
}

impl TtsTalkerCatalog {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let keys = read_safetensors_keys(path)?;
        Ok(Self::from_keys(keys))
    }

    #[must_use]
    pub fn from_keys(mut keys: Vec<String>) -> Self {
        keys.sort();
        let has_talker_weights = keys.iter().any(|key| key.starts_with("talker."));
        let has_speaker_encoder_weights =
            keys.iter().any(|key| key.starts_with("speaker_encoder."));
        Self {
            tensor_count: keys.len(),
            keys,
            has_talker_weights,
            has_speaker_encoder_weights,
        }
    }

    #[must_use]
    pub fn keys(&self) -> &[String] {
        &self.keys
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TtsCodecCatalog {
    keys: Vec<String>,
    pub tensor_count: usize,
    pub has_decoder_weights: bool,
    pub has_encoder_weights: bool,
    pub has_codebook_stats: bool,
}

impl TtsCodecCatalog {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let keys = read_safetensors_keys(path)?;
        Ok(Self::from_keys(keys))
    }

    #[must_use]
    pub fn from_keys(mut keys: Vec<String>) -> Self {
        keys.sort();
        let has_decoder_weights = keys.iter().any(|key| is_decoder_weight_key(key));
        let has_encoder_weights = keys.iter().any(|key| is_encoder_weight_key(key));
        let has_codebook_stats = keys.iter().any(|key| {
            key.ends_with("._codebook.cluster_usage")
                || key.ends_with("._codebook.embedding_sum")
                || key.ends_with(".codebook.cluster_usage")
                || key.ends_with(".codebook.embed_sum")
        });
        Self {
            tensor_count: keys.len(),
            keys,
            has_decoder_weights,
            has_encoder_weights,
            has_codebook_stats,
        }
    }

    #[must_use]
    pub fn keys(&self) -> &[String] {
        &self.keys
    }
}

fn is_decoder_weight_key(key: &str) -> bool {
    key.starts_with("decoder.")
        || key.starts_with("model.decoder.")
        || key.contains(".decode.")
        || key.contains("semantic_model")
}

fn is_encoder_weight_key(key: &str) -> bool {
    key.starts_with("encoder.")
        || key.starts_with("model.encoder.")
        || key.contains(".encode.")
        || key.contains("seanet_encoder")
}
