//! Decode résident de Flash-Next (`qwen4_exp`), un token par command buffer.
//!
//! Réutilise les primitives Metal existantes (qmv, MoE, DeltaNet, attention)
//! sans l'orchestrateur des couches Qwen : Flash-Next n'a pas d'`input_norm`
//! ni de `post_attention_norm`, mais des hyper-connections à 4 flux.

#![allow(
    dead_code,
    unused_imports,
    reason = "briques du decode Flash-Next, câblées couche par couche (étape 4)"
)]

mod attention;
mod hc;
mod kernels;
mod ple;

#[cfg(test)]
mod tests;

pub(crate) use attention::{
    encode_qsa_dense, QsaDecodeDims, QsaLayerWeights, QsaScratch, DENSE_MAX_VISIBLE,
};
pub(crate) use hc::{encode_hc_combine, encode_hc_mix, HcDims, HcScratch, HcWeights};
pub(crate) use kernels::FlashNextKernels;
pub(crate) use ple::{encode_ple, PleScratch, PleState, PleWeights};

use metal::{ComputeCommandEncoderRef, MTLSize};

/// Passe une petite valeur constante (`set_bytes`) au kernel.
fn set_value<T>(encoder: &ComputeCommandEncoderRef, index: u64, value: &T) {
    encoder.set_bytes(
        index,
        std::mem::size_of::<T>() as u64,
        (value as *const T).cast(),
    );
}

/// Dispatch 1D d'un thread par élément, threadgroups de 256.
fn dispatch_1d(encoder: &ComputeCommandEncoderRef, n: usize) {
    encoder.dispatch_thread_groups(
        MTLSize::new(n.div_ceil(256) as u64, 1, 1),
        MTLSize::new(256, 1, 1),
    );
}
