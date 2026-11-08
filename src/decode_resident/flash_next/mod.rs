//! Decode résident de Flash-Next (`qwen4_exp`), un token par command buffer.
//!
//! Réutilise les primitives Metal existantes (qmv, MoE, DeltaNet, attention)
//! sans l'orchestrateur des couches Qwen : Flash-Next n'a pas d'`input_norm`
//! ni de `post_attention_norm`, mais des hyper-connections à 4 flux.

// NOTE: le parent `decode_resident` tolère le code mort ; ce module le
// signale de nouveau, et seules les entrées sans appelant de production
// portent un `allow` ciblé.
#![warn(dead_code)]

mod attention;
mod encoders;
mod hc;
mod kernels;
mod ple;
#[cfg(feature = "devtools")]
mod profile;
mod session;
mod speculative;
mod weights;

#[cfg(test)]
mod tests;

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
    crate::metal_backend::profile_dispatch();
    encoder.dispatch_thread_groups(
        MTLSize::new(n.div_ceil(256) as u64, 1, 1),
        MTLSize::new(256, 1, 1),
    );
}

/// Dispatch de `groups` threadgroups de `width` threads (normes, réductions par flux).
fn dispatch_groups(encoder: &ComputeCommandEncoderRef, groups: usize, width: u64) {
    crate::metal_backend::profile_dispatch();
    encoder.dispatch_thread_groups(MTLSize::new(groups as u64, 1, 1), MTLSize::new(width, 1, 1));
}
