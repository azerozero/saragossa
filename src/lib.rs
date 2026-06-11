//! Noyau d'inférence Rust pur pour le backend expérimental `--backends rust`.

#![deny(unsafe_code)]

pub mod activation;
pub mod assets;
pub mod catalog;
pub mod chat_template;
pub mod config;
#[cfg(all(target_os = "macos", feature = "metal"))]
pub(crate) mod decode_resident;
pub mod decoder;
pub mod embedding;
pub mod error;
pub mod linear;
pub mod linear_attention;
#[cfg(all(target_os = "macos", feature = "metal"))]
pub mod metal_backend;
pub mod mlp;
pub mod norm;
pub mod quantization;
pub mod qwen_loader;
pub mod rope;
pub mod runtime;
pub mod safetensor;
pub mod sampling;
pub mod tensor;
pub mod tokenizer;

pub use activation::silu;
pub use assets::{ModelAssets, MtpWeightsInfo};
pub use catalog::WeightCatalog;
pub use chat_template::{
    qwen_assistant_history_content, render_qwen_chatml, ChatTemplateMessage,
    QWEN_EMPTY_THINK_BLOCK, QWEN_IM_END, QWEN_IM_START,
};
pub use config::{ModelConfig, QuantConfig, RawModelConfig};
pub use decoder::{
    CausalDecoder, CausalDecoderCache, CausalDecoderConfig, GenerationOptions, SpeculativeOutput,
    SpeculativeStats,
};
pub use embedding::{embed_tokens, embed_weight_tokens, EmbeddingWeight};
pub use error::{InferError, Result};
pub use linear::{Linear, LinearWeight};
#[cfg(all(target_os = "macos", feature = "metal"))]
pub use metal_backend::MetalExecutor;
pub use mlp::{FeedForward, GatedMlp, MoeMlp};
pub use norm::rms_norm;
pub use quantization::{dequantize_affine_u32, AffineQuantizedTensor};
pub use qwen_loader::{
    load_qwen_causal_decoder, load_qwen_causal_decoder_from_shards, verify_qwen_decoder_contract,
    verify_qwen_decoder_contract_from_shards, QwenDecoderContract,
};
pub use rope::apply_rope;
pub use runtime::ForwardRuntime;
pub use safetensor::{load_f32_tensor, load_f32_tensors, load_float_tensor, load_float_tensors};
pub use sampling::{argmax, sample_token, sample_token_top_k_top_p, softmax, DeterministicSampler};
pub use tensor::Tensor;
pub use tokenizer::RustTokenizer;
