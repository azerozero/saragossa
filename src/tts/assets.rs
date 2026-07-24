use super::{
    read_json, require_file, TtsCodecCatalog, TtsCodecConfig, TtsModelConfig, TtsModelKind,
    TtsTalkerCatalog,
};
use crate::{InferError, Result};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct TtsAssets {
    /// Répertoire racine du snapshot Qwen3-TTS local.
    pub model_dir: PathBuf,
    /// Config racine du talker Qwen3-TTS.
    pub model_config: TtsModelConfig,
    /// Config du speech tokenizer.
    pub codec_config: TtsCodecConfig,
    /// Vocabulaire BPE Qwen3-TTS (`vocab.json`).
    pub vocab_path: PathBuf,
    /// Merges BPE Qwen3-TTS (`merges.txt`).
    pub merges_path: PathBuf,
    /// Poids racine (`model.safetensors`) : talker + éventuellement speaker encoder.
    pub talker_weights: PathBuf,
    /// Poids du speech tokenizer (`speech_tokenizer/model.safetensors`).
    pub codec_weights: PathBuf,
    /// Catalogue des clés du safetensors racine.
    pub talker_catalog: TtsTalkerCatalog,
    /// Catalogue des clés du speech tokenizer.
    pub codec_catalog: TtsCodecCatalog,
}

impl TtsAssets {
    /// Charge le contrat d'un snapshot Qwen3-TTS local sans charger les poids.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si un artefact obligatoire manque, si une config JSON
    /// est invalide, ou si les safetensors ne ressemblent pas à un checkpoint TTS.
    pub fn load_local(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        if !model_dir.is_dir() {
            return Err(InferError::MissingArtifact {
                path: model_dir.to_path_buf(),
                what: "tts model_dir",
            });
        }

        let config_path = model_dir.join("config.json");
        let vocab_path = model_dir.join("vocab.json");
        let merges_path = model_dir.join("merges.txt");
        let talker_weights = model_dir.join("model.safetensors");
        let speech_dir = model_dir.join("speech_tokenizer");
        let codec_config_path = speech_dir.join("config.json");
        let codec_weights = speech_dir.join("model.safetensors");

        require_file(&config_path, "config.json")?;
        require_file(&vocab_path, "vocab.json")?;
        require_file(&merges_path, "merges.txt")?;
        require_file(&talker_weights, "model.safetensors")?;
        require_file(&codec_config_path, "speech_tokenizer/config.json")?;
        require_file(&codec_weights, "speech_tokenizer/model.safetensors")?;

        let model_config = read_json(&config_path)?;
        let codec_config = read_json(&codec_config_path)?;
        let talker_catalog = TtsTalkerCatalog::from_path(&talker_weights)?;
        let codec_catalog = TtsCodecCatalog::from_path(&codec_weights)?;

        if !talker_catalog.has_talker_weights {
            return Err(InferError::MissingArtifact {
                path: talker_weights,
                what: "talker.* weights",
            });
        }
        if !codec_catalog.has_decoder_weights {
            return Err(InferError::MissingArtifact {
                path: codec_weights,
                what: "speech_tokenizer decoder weights",
            });
        }

        Ok(Self {
            model_dir: model_dir.to_path_buf(),
            model_config,
            codec_config,
            vocab_path,
            merges_path,
            talker_weights,
            codec_weights,
            talker_catalog,
            codec_catalog,
        })
    }

    #[must_use]
    pub fn model_kind(&self) -> TtsModelKind {
        self.model_config.model_kind()
    }

    #[must_use]
    pub fn clone_capable(&self) -> bool {
        self.model_config.speaker_encoder_config.is_some()
            && self.codec_config.encoder_config.is_some()
            && self.talker_catalog.has_speaker_encoder_weights
            && self.codec_catalog.has_encoder_weights
    }
}
