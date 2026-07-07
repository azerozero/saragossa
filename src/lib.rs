//! Moteur d'inférence Rust pur de reti (kernels metal-rs) : LLM, STT Whisper,
//! TTS Qwen3 — backend prod `--backends rust-metal`, repli CPU `--backends rust`.

#![deny(unsafe_code)]

pub mod activation;
pub mod assets;
pub mod catalog;
pub mod chat_template;
pub mod config;
#[cfg(all(target_os = "macos", feature = "metal"))]
pub(crate) mod decode_resident;
pub mod decoder;
#[cfg(feature = "devtools")]
pub mod devtools;
pub mod embedding;
pub mod error;
#[cfg(test)]
mod golden;
pub mod linear;
pub mod linear_attention;
#[cfg(all(target_os = "macos", feature = "metal"))]
pub mod metal_backend;
pub mod mlp;
pub mod norm;
pub mod quantization;
pub mod qwen_loader;
pub mod runtime;
pub mod runtime_flags;
pub mod runtime_preset;
pub mod safetensor;
pub mod sampling;
pub mod tensor;
pub mod tokenizer;
pub mod tts;
pub mod tts_clone;
pub(crate) mod tts_codec;
#[cfg(all(target_os = "macos", feature = "metal"))]
pub(crate) mod tts_codec_gpu;
pub mod tts_mimi;
pub mod tts_speaker;
pub mod whisper;

pub use activation::{gelu, gelu_tanh, silu};
pub use assets::{ModelAssets, MtpWeightsInfo};
pub use catalog::WeightCatalog;
pub use chat_template::{
    qwen_assistant_history_content, render_gemma4_chat, render_gemma_chat, render_qwen_chatml,
    ChatTemplateMessage, GEMMA4_END_OF_TURN, GEMMA4_START_OF_TURN, GEMMA_END_OF_TURN,
    GEMMA_START_OF_TURN, QWEN_EMPTY_THINK_BLOCK, QWEN_IM_END, QWEN_IM_START,
};
pub use config::{ModelConfig, QuantConfig, RawModelConfig, RopeScalingConfig};
pub use decoder::{
    force_resident_full_linear_decode, CausalDecoder, CausalDecoderCache, CausalDecoderConfig,
    GenerationOptions, RopeStyle, SpeculativeOutput, SpeculativeStats,
};
#[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
pub use decoder::{ResidentLinearXrayLayerDiff, ResidentLinearXrayReport};
#[cfg(feature = "devtools")]
pub use devtools::{
    load_dflash_draft_for_target, load_dflash_draft_weights_for_target, DFlashAttentionWeights,
    DFlashDraft, DFlashDraftInfo, DFlashDraftLayer,
};
pub use embedding::{embed_tokens, embed_weight_tokens, EmbeddingWeight};
pub use error::{InferError, Result};
pub use linear::{Linear, LinearWeight};
#[cfg(all(target_os = "macos", feature = "metal"))]
pub use metal_backend::MetalExecutor;
pub use mlp::{Activation, FeedForward, GatedMlp, MoeMlp};
pub use norm::{layer_norm, rms_norm};
pub use quantization::{dequantize_affine_u32, AffineQuantizedTensor};
pub use qwen_loader::{
    load_causal_decoder, load_causal_decoder_from_shards, load_qwen_causal_decoder,
    load_qwen_causal_decoder_from_shards, verify_decoder_contract,
    verify_decoder_contract_from_shards, verify_qwen_decoder_contract,
    verify_qwen_decoder_contract_from_shards, DecoderContract, QwenDecoderContract,
};
pub use runtime::ForwardRuntime;
pub use runtime_preset::{
    apply_runtime_preset_for_model_dir, runtime_preset_for_model_dir, RuntimePreset,
};
pub use safetensor::{load_f32_tensor, load_f32_tensors, load_float_tensor, load_float_tensors};
pub use sampling::{argmax, sample_token, sample_token_top_k_top_p, softmax, DeterministicSampler};
pub use tensor::Tensor;
pub use tokenizer::RustTokenizer;
pub use tts::{
    TtsAssets, TtsCodecCatalog, TtsForwardOutput, TtsModel, TtsModelConfig, TtsModelKind,
    TtsPayloadSummary, TtsSynthesisOutput, TtsTalkerCatalog,
};
pub use whisper::{WhisperConfig, WhisperDecoder, WhisperEncoder, WhisperModel};
