//! Hyper-connections résidentes de Flash-Next (gated residual), un token.
//!
//! `n = RMSNorm_groupée(H)·(1+w)`, `a = σ(W_up·SiLU(W_down·n/hc))`,
//! `x = mean_s(a ⊙ n)`, injection `2σ(W_inject·n/hc)` sur le résiduel non
//! normalisé : `H′_s = H_s + inj_s·F(x)`. Le mixer final n'a pas d'injection.

use metal::{Buffer, BufferRef, ComputeCommandEncoderRef, MTLSize};

use super::kernels::FlashNextKernels;
use super::ple::encode_rms_norm_1p;
use super::{dispatch_1d, set_value};
use crate::metal_backend::{post_dispatch_barrier, MetalExecutor, MetalLinearWeightBuffers};
use crate::Result;

/// Poids GPU d'une hyper-connection.
pub(crate) struct HcWeights {
    /// Gains `w` de la norme groupée, largeur `hc·hidden` (f32).
    pub(crate) norm: Buffer,
    pub(crate) down: MetalLinearWeightBuffers,
    pub(crate) up: MetalLinearWeightBuffers,
    /// `[hc, hc·hidden]` en f32 ; `None` pour le mixer final.
    pub(crate) inject: Option<Buffer>,
}

/// Intermédiaires réutilisés d'une couche à l'autre (encodage séquentiel).
pub(crate) struct HcScratch {
    pub(crate) normed: Buffer,
    pub(crate) low: Buffer,
    pub(crate) up: Buffer,
    /// Sommes partielles de l'injection `[hc, INJECT_CHUNKS]`.
    pub(crate) inject: Buffer,
}

/// Tranches de la réduction de l'injection (16 × 640 à largeur 10240).
pub(crate) const INJECT_CHUNKS: usize = 16;

/// Dimensions d'une hyper-connection.
#[derive(Clone, Copy, Debug)]
pub(crate) struct HcDims {
    pub(crate) hidden: usize,
    pub(crate) hc: usize,
    pub(crate) lowrank: usize,
    pub(crate) eps: f32,
}

impl HcDims {
    pub(crate) fn width(self) -> usize {
        self.hidden * self.hc
    }
}

/// Mélange les flux `h` (`[hc·hidden]`) vers l'entrée de branche `x`
/// (`[hidden]`) ; écrit l'injection brute `W_inject·n` dans `scratch.inject`.
#[expect(
    clippy::too_many_arguments,
    reason = "encodage Metal : pipelines, poids, scratch et buffers explicites"
)]
pub(crate) fn encode_hc_mix(
    kernels: &FlashNextKernels,
    executor: &MetalExecutor,
    encoder: &ComputeCommandEncoderRef,
    weights: &HcWeights,
    scratch: &HcScratch,
    dims: HcDims,
    h: &BufferRef,
    x: &BufferRef,
) -> Result<()> {
    let width = dims.width();
    encode_rms_norm_1p(kernels, encoder, h, &weights.norm, &scratch.normed, dims);
    post_dispatch_barrier(encoder);

    executor.encode_matmul_weight_buffers(
        encoder,
        &scratch.normed,
        1,
        width,
        &weights.down,
        &scratch.low,
        true,
    )?;
    post_dispatch_barrier(encoder);
    if let Some(inject) = &weights.inject {
        // Injection : 4 lignes trop courtes pour un GEMV ; tranches en parallèle,
        // sommes partielles réduites dans `encode_hc_combine`.
        encoder.set_compute_pipeline_state(&kernels.hc_inject_partial);
        encoder.set_buffer(0, Some(&scratch.normed), 0);
        encoder.set_buffer(1, Some(inject), 0);
        encoder.set_buffer(2, Some(&scratch.inject), 0);
        set_value(encoder, 3, &[width as u32, INJECT_CHUNKS as u32]);
        crate::metal_backend::profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(INJECT_CHUNKS as u64, dims.hc as u64, 1),
            MTLSize::new(256, 1, 1),
        );
    }
    encoder.set_compute_pipeline_state(&kernels.hc_silu_scale);
    encoder.set_buffer(0, Some(&scratch.low), 0);
    set_value(encoder, 1, &(dims.lowrank as u32));
    set_value(encoder, 2, &(dims.hc as f32));
    dispatch_1d(encoder, dims.lowrank);
    post_dispatch_barrier(encoder);

    executor.encode_matmul_weight_buffers(
        encoder,
        &scratch.low,
        1,
        dims.lowrank,
        &weights.up,
        &scratch.up,
        true,
    )?;
    post_dispatch_barrier(encoder);

    encoder.set_compute_pipeline_state(&kernels.hc_mix);
    encoder.set_buffer(0, Some(&scratch.normed), 0);
    encoder.set_buffer(1, Some(&scratch.up), 0);
    encoder.set_buffer(2, Some(x), 0);
    set_value(encoder, 3, &[dims.hidden as u32, dims.hc as u32]);
    dispatch_1d(encoder, dims.hidden);
    post_dispatch_barrier(encoder);
    Ok(())
}

/// `out = h + 2σ(inject/hc)·f` sur les `hc` flux ; `inject` est le produit
/// brut laissé par [`encode_hc_mix`] dans `scratch.inject`.
pub(crate) fn encode_hc_combine(
    kernels: &FlashNextKernels,
    encoder: &ComputeCommandEncoderRef,
    scratch: &HcScratch,
    dims: HcDims,
    h: &BufferRef,
    f: &BufferRef,
    out: &BufferRef,
) {
    encoder.set_compute_pipeline_state(&kernels.hc_combine);
    encoder.set_buffer(0, Some(h), 0);
    encoder.set_buffer(1, Some(&scratch.inject), 0);
    encoder.set_buffer(2, Some(f), 0);
    encoder.set_buffer(3, Some(out), 0);
    set_value(
        encoder,
        4,
        &[dims.hidden as u32, dims.hc as u32, INJECT_CHUNKS as u32],
    );
    dispatch_1d(encoder, dims.width());
    post_dispatch_barrier(encoder);
}
