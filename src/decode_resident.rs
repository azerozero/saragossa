//! État GPU résident pour le decode « 1 token = 1 command buffer ».
//!
//! Fondation de la tranche 1b/1c du decode résident. Le decode actuel fait un
//! `commit`+`wait`+readback par opération (~80-120 syncs/token,
//! `wait_us` ≈ 91 % du temps/token mesuré) : le CPU et le GPU ne se recouvrent
//! jamais. Le chemin résident vise un seul command buffer par token, les
//! tenseurs intermédiaires restant **GPU-résidents**, avec une unique lecture
//! CPU à la fin (l'id du token).
//!
//! Ce module isole la **discipline d'ownership/liveness des buffers Metal** hors
//! du module `metal_backend` et la verrouille par des tests différentiels.
//! Il répond à trois réserves de la revue Codex :
//!  - **(B) aliasing scratch.** `MetalExecutor::scratch_buffer` renvoie le MÊME
//!    `MTLBuffer` par label (`metal_backend`) → aliasing silencieux si deux
//!    tenseurs vivants partagent un label. Ici le scratch est **à bail**
//!    ([`ScratchLease`]) : deux bails vivants ne partagent JAMAIS de buffer ;
//!    seuls des bails à liveness disjointe réutilisent. La poignée scratch est
//!    `&GpuTensor` empruntée au bail → le borrow-checker interdit d'encoder une
//!    op après la libération du scratch (pas de clobber d'un buffer rendu au
//!    pool alors que le GPU le référence encore).
//!  - **(C) concurrence.** Un [`DecodeResidentState`] = UN decode ; jamais
//!    partagé entre deux decodes simultanés. La sérialisation des decodes est
//!    garantie côté reti (un seul decode LLM actif à la fois) ; ici l'état est
//!    volontairement conçu pour vivre comme local de la session de génération.
//!  - **(D) état résident clonable sans aliasing.** Pas d'impl `Clone` : comme
//!    `LinearAttentionCache` (qui met son état Metal à `None` au clone), un état
//!    résident GPU est lié à une session et ne doit jamais être aliasé par copie.
//!
//! Réserve **(E)** : ce module EST le jalon 1a.5 (briques minimales + test de
//! liveness sur des couches synthétiques) à dérisquer avant 1b.

// NOTE: fondation des tranches 1b/1c. Les briques publiques ne sont pour l'instant
// exercées que par les tests ; la boucle decode résidente (1b) les câblera. Le
// `dead_code` sera retiré à ce moment-là.
#![allow(
    dead_code,
    reason = "fondation du decode résident (1b/1c) — câblée à la phase suivante"
)]

use crate::metal_backend::{
    commit_and_wait, read_f32_buffer, LinearAttentionMetalState, LinearAttentionStepSpec,
    LinearAttnResidentDims, MetalExecutor, MetalLinearAttnResidentDenseWeights,
    MetalLinearWeightBuffers, MetalMoeRoutedWeights, MetalMoeSharedWeights,
};
use crate::{InferError, Result};
use metal::{
    Buffer, BufferRef, CommandQueue, CompileOptions, ComputeCommandEncoderRef,
    ComputePipelineState, Device, MTLResourceOptions, MTLSize,
};
use std::cell::RefCell;
use std::ffi::c_void;
use std::sync::{Arc, Mutex, OnceLock};

mod arena;
mod attention;
mod duo;
mod kernels;
mod layers;
mod types;
mod utils;

#[cfg(test)]
mod tests;

pub(crate) use self::arena::{GpuElement, GpuTensor, ScratchLease, ScratchPool};
pub(crate) use self::attention::FullAttentionMetalState;
pub(crate) use self::types::{
    FullAttnDenseLayerWeights, FullAttnLayerDims, FullAttnLayerWeights, FullAttnRoutedLayerWeights,
    GpuSectionTimer, LinearAttnDenseLayerWeights, LinearAttnLayerWeights,
};

/// État résident d'UN decode (réserves C/D).
///
/// Détient le `Device`, les buffers **persistants** (futurs KV-cache full-attn,
/// conv/ssm linear-attn, ping-pong des résiduels — vivants jusqu'au drop) et le
/// [`ScratchPool`] des intermédiaires transitoires. Volontairement **non
/// clonable** : un état résident GPU est lié à une session.
#[derive(Debug)]
pub(crate) struct DecodeResidentState {
    device: Device,
    queue: CommandQueue,
    options: MTLResourceOptions,
    persistent: Vec<Buffer>,
    scratch: ScratchPool,
    attention_decode_naive: ComputePipelineState,
    attention_decode_flash: ComputePipelineState,
    attention_decode_flash_d256: ComputePipelineState,
    attention_decode_2pass_1: ComputePipelineState,
    attention_decode_2pass_1_d128: ComputePipelineState,
    attention_decode_naive_bf16: ComputePipelineState,
    attention_decode_flash_bf16: ComputePipelineState,
    attention_decode_flash_d256_bf16: ComputePipelineState,
    attention_decode_2pass_1_bf16: ComputePipelineState,
    attention_decode_2pass_1_d128_bf16: ComputePipelineState,
    attention_decode_2pass_2: ComputePipelineState,
    attention_decode_2pass_2_d128: ComputePipelineState,
    split_q_gate_kernel: ComputePipelineState,
    attn_gate_kernel: ComputePipelineState,
    rope_decode_kernel: ComputePipelineState,
    copy_at_kernel: ComputePipelineState,
    copy_at_f32_to_bf16_kernel: ComputePipelineState,
    /// Instrumentation per-section (tranche 3), active si `RETI_RUST_GPU_COUNTERS`.
    timer: Option<GpuSectionTimer>,
    /// Slot de flux (light-batch) : namespace du scratch label-keyed de
    /// l'exécuteur partagé. `0` = mono-flux historique.
    scratch_namespace: u64,
}
