use crate::{InferError, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TtsModelKind {
    VoiceDesign,
    Base,
    CustomVoice,
    Other(String),
}

#[derive(Debug, Clone, Deserialize)]
pub struct TtsQuantConfig {
    pub group_size: i32,
    pub bits: i32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TtsCodePredictorConfig {
    pub head_dim: i32,
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub num_hidden_layers: i32,
    pub num_code_groups: i32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub vocab_size: i32,
    #[serde(default = "default_code_predictor_max_pos")]
    pub max_position_embeddings: i32,
    #[serde(default)]
    pub attention_bias: bool,
}

fn default_code_predictor_max_pos() -> i32 {
    65_536
}

#[derive(Debug, Clone, Deserialize)]
pub struct TtsRopeScaling {
    #[serde(default)]
    pub interleaved: bool,
    pub mrope_section: Vec<i32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TtsTalkerConfig {
    pub head_dim: i32,
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub num_hidden_layers: i32,
    pub num_code_groups: i32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub vocab_size: i32,
    pub text_vocab_size: i32,
    pub text_hidden_size: i32,
    #[serde(default = "default_talker_max_pos")]
    pub max_position_embeddings: i32,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    pub rope_scaling: TtsRopeScaling,
    pub code_predictor_config: TtsCodePredictorConfig,
    pub codec_bos_id: i32,
    pub codec_eos_token_id: i32,
    pub codec_pad_id: i32,
    pub codec_think_id: i32,
    pub codec_nothink_id: i32,
    pub codec_think_bos_id: i32,
    pub codec_think_eos_id: i32,
    #[serde(default)]
    pub codec_language_id: HashMap<String, i32>,
    #[serde(default)]
    pub spk_id: HashMap<String, i32>,
}

fn default_talker_max_pos() -> i32 {
    32_768
}

fn default_hidden_act() -> String {
    "silu".to_string()
}

#[derive(Debug, Clone, Deserialize)]
pub struct TtsSpeakerEncoderConfig {
    pub enc_dim: i32,
    #[serde(default = "default_speaker_sr")]
    pub sample_rate: i32,
}

fn default_speaker_sr() -> i32 {
    24_000
}

#[derive(Debug, Clone, Deserialize)]
pub struct TtsModelConfig {
    pub talker_config: TtsTalkerConfig,
    pub quantization: TtsQuantConfig,
    pub tts_bos_token_id: i32,
    pub tts_eos_token_id: i32,
    pub tts_pad_token_id: i32,
    #[serde(default = "default_tts_model_type")]
    pub tts_model_type: String,
    #[serde(default)]
    pub speaker_encoder_config: Option<TtsSpeakerEncoderConfig>,
}

impl TtsModelConfig {
    #[must_use]
    pub fn model_kind(&self) -> TtsModelKind {
        match self.tts_model_type.as_str() {
            "voice_design" => TtsModelKind::VoiceDesign,
            "base" => TtsModelKind::Base,
            "custom_voice" => TtsModelKind::CustomVoice,
            other => TtsModelKind::Other(other.to_string()),
        }
    }
}

impl TtsModelConfig {
    /// CustomVoice choisit Serena en l'absence de voix explicite.
    /// VoiceDesign et Base n'utilisent pas de locuteur nommé.
    pub fn speaker_id(&self, voice: Option<&str>) -> Result<Option<i32>> {
        if self.model_kind() != TtsModelKind::CustomVoice {
            return Ok(None);
        }
        let voice = voice.unwrap_or("serena").to_lowercase();
        self.talker_config
            .spk_id
            .get(&voice)
            .copied()
            .filter(|id| *id >= 0 && *id < self.talker_config.vocab_size)
            .map(Some)
            .ok_or_else(|| InferError::Config(format!("voix CustomVoice inconnue: {voice}")))
    }
}

impl TtsTalkerConfig {
    pub(super) fn codec_prompt_ids(&self, speaker: Option<i32>) -> Vec<i32> {
        let mut ids = match self.codec_language_id.get("french") {
            Some(&language) => vec![
                self.codec_think_id,
                self.codec_think_bos_id,
                language,
                self.codec_think_eos_id,
            ],
            None => vec![
                self.codec_nothink_id,
                self.codec_think_bos_id,
                self.codec_think_eos_id,
            ],
        };
        ids.extend(speaker);
        ids.extend([self.codec_pad_id, self.codec_bos_id]);
        ids
    }
}

fn default_tts_model_type() -> String {
    "voice_design".to_string()
}

#[derive(Debug, Clone, Deserialize)]
pub struct TtsDecoderConfig {
    pub latent_dim: i32,
    pub codebook_dim: i32,
    pub codebook_size: i32,
    pub decoder_dim: i32,
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub head_dim: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub num_hidden_layers: i32,
    pub num_quantizers: i32,
    pub num_semantic_quantizers: i32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub layer_scale_initial_scale: f32,
    #[serde(default = "default_codec_max_pos")]
    pub max_position_embeddings: i32,
    #[serde(default)]
    pub attention_bias: bool,
    pub upsample_rates: Vec<i32>,
    pub upsampling_ratios: Vec<i32>,
}

fn default_codec_max_pos() -> i32 {
    8_000
}

#[derive(Debug, Clone, Deserialize)]
pub struct TtsEncoderConfig {
    pub audio_channels: i32,
    pub codebook_dim: i32,
    pub codebook_size: i32,
    pub compress: i32,
    pub dilation_growth_rate: i32,
    pub head_dim: i32,
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub kernel_size: i32,
    pub last_kernel_size: i32,
    pub layer_scale_initial_scale: f32,
    pub num_attention_heads: i32,
    pub num_filters: i32,
    pub num_hidden_layers: i32,
    pub num_key_value_heads: i32,
    pub num_quantizers: i32,
    pub num_residual_layers: i32,
    pub num_semantic_quantizers: i32,
    pub residual_kernel_size: i32,
    pub rope_theta: f32,
    pub sampling_rate: i32,
    pub sliding_window: i32,
    #[serde(rename = "_frame_rate")]
    pub frame_rate: f32,
    pub upsampling_ratios: Vec<i32>,
    #[serde(default = "default_true")]
    pub use_causal_conv: bool,
    #[serde(default)]
    pub use_conv_shortcut: bool,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize)]
pub struct TtsCodecConfig {
    pub decoder_config: TtsDecoderConfig,
    pub decode_upsample_rate: i32,
    pub output_sample_rate: i32,
    pub encoder_valid_num_quantizers: i32,
    #[serde(default)]
    pub encoder_config: Option<TtsEncoderConfig>,
}

pub(super) fn require_file(path: &Path, what: &'static str) -> Result<()> {
    if path.is_file() {
        Ok(())
    } else {
        Err(InferError::MissingArtifact {
            path: path.to_path_buf(),
            what,
        })
    }
}

pub(super) fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let file = std::fs::File::open(path).map_err(|source| InferError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::from_reader(file).map_err(|source| InferError::Json {
        path: path.to_path_buf(),
        source,
    })
}
