//! Modèle Qwen3-TTS pour le port Rust pur : synthèse audio de bout en bout.
//!
//! Ce module porte le modèle TTS complet : configs typées, chemins tokenizer
//! BPE, payloads talker/codec et synthèse PCM (`synthesize_greedy` et variantes
//! streaming), le tout vérifié octet-près contre l'oracle mlx-rs.

mod assets;
mod catalog;
mod config;
mod constants;
mod generation;
mod helpers;
mod inputs;
mod load;
mod model;
mod payload;
mod sampling;
mod synth;
#[cfg(test)]
mod tests;
mod tokenizer;
mod weights;

pub use assets::TtsAssets;
pub use catalog::{TtsCodecCatalog, TtsTalkerCatalog};
pub use config::{
    TtsCodePredictorConfig, TtsCodecConfig, TtsDecoderConfig, TtsEncoderConfig, TtsModelConfig,
    TtsModelKind, TtsQuantConfig, TtsRopeScaling, TtsSpeakerEncoderConfig, TtsTalkerConfig,
};
pub use constants::DEFAULT_INSTRUCT;
pub use model::{TtsForwardOutput, TtsModel, TtsPayloadSummary, TtsSynthesisOutput};
pub(crate) use payload::SafetensorPayload;

use config::{read_json, require_file};
use constants::{
    CLONE_DEFAULT_FRAMES_PER_TOKEN, CLONE_DEFAULT_MIN_FRAMES, CLONE_GENERATION_HARD_CAP,
    CLONE_SAMPLE_REPETITION_PENALTY, CLONE_SAMPLE_TEMPERATURE, CLONE_SAMPLE_TOP_K,
    CLONE_SAMPLE_TOP_P, DEFAULT_REPEAT_FRAME_STOP, QWEN3_TTS_SPECIAL_TOKENS,
};
#[cfg(test)]
use helpers::tts_repeat_frame_stop_from_env;
use helpers::{
    add_into, clone_effective_frame_cap, clone_sample_seed, gather_rows_i32, push_rows,
    repeat_frame_stop_tripped, tts_generation_trace_enabled, tts_internal_profile_enabled,
    tts_repeat_frame_stop, tts_stream_first_lot, usize_from_i32,
};
use model::{
    PreparedVoiceDesign, TtsCloneContext, TtsCloneMode, TtsSampleParams, TtsStreamDecodeState,
};
use sampling::{greedy_talker_token, greedy_token, sample_talker_token};
use tokenizer::load_qwen3_tts_tokenizer;
use weights::{copy_dense, insert_linear, read_linear_layer};
