//! Référence CPU f32 de la famille `qwen4_exp` (Qwen3.8-Flash-Next).
//!
//! Transcription des modules transformers 5.20.2 (`modeling_qwen4_exp.py`,
//! attention `sdpa`), sans cache : passage unique sur toute la séquence. Le
//! decode incrémental, le prefill découpé et les chemins Metal viendront avec
//! le chemin résident ; cette référence sert d'oracle.

mod attention;
mod forward;
pub(crate) mod math;
mod moe;
pub(crate) mod ngram;
mod ple;
pub(crate) mod residual;

#[cfg(test)]
mod tests;

pub use forward::{LayerOutput, QsaTrace, Qwen4ExpReference};
