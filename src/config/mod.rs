//! Chargement et normalisation des configurations de modèles Qwen.

mod accessors;
mod gemma_defaults;
mod resolve;
mod types;

pub use types::{ModelConfig, QuantConfig, RawModelConfig, RopeScalingConfig};

#[cfg(test)]
use resolve::bf16_round;

#[cfg(test)]
mod tests;
