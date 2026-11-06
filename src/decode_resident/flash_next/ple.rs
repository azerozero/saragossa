//! PLE résident de Flash-Next (embedding n-gram de la couche d'indice 1).
//!
//! Protocole de synchronisation : le CPU hache le token, lit les lignes E4M3
//! de la table (lectures parallèles) dans un buffer partagé, **avant le commit**
//! du command buffer du token ; le GPU les décode par table de 256 valeurs
//! (E4M3 × échelle). Le buffer d'octets n'est réécrit qu'au token suivant,
//! après l'attente du command buffer précédent : le decode attend chaque token.
//!
//! État résident : contexte n-gram (2 tokens précédents, CPU) et anneau des
//! `span = (k−1)·dilation` dernières colonnes `normed` (GPU), indexé par le pas.

use metal::{Buffer, BufferRef, ComputeCommandEncoderRef};

use super::hc::HcDims;
use super::kernels::FlashNextKernels;
use super::{dispatch_1d, dispatch_groups, set_value};
use crate::metal_backend::{
    post_dispatch_barrier, write_u8_buffer, MetalExecutor, MetalLinearWeightBuffers,
};
use crate::qwen4_exp::ngram::NgramContext;
use crate::{InferError, Result};

/// Poids GPU du PLE.
pub(crate) struct PleWeights {
    /// `[hc·hidden, ple_dim]`.
    pub(crate) key_proj: MetalLinearWeightBuffers,
    /// `[hidden, ple_dim]`.
    pub(crate) value_proj: MetalLinearWeightBuffers,
    pub(crate) norm_key: Buffer,
    pub(crate) norm_query: Buffer,
    pub(crate) norm_conv: Buffer,
    /// `[hc·hidden, 1, kernel]` aplati.
    pub(crate) conv: Buffer,
    /// E4M3 × échelle per-tensor, 256 valeurs.
    pub(crate) lut: Buffer,
    pub(crate) kernel: usize,
    pub(crate) dilation: usize,
    pub(crate) ple_dim: usize,
}

/// Intermédiaires du PLE (largeur `hc·hidden` sauf mention).
pub(crate) struct PleScratch {
    /// `[ple_dim]`.
    pub(crate) emb: Buffer,
    pub(crate) key: Buffer,
    pub(crate) key_normed: Buffer,
    /// `[hidden]`.
    pub(crate) value: Buffer,
    pub(crate) query: Buffer,
    pub(crate) gated: Buffer,
    pub(crate) normed: Buffer,
}

/// État PLE d'une séquence.
pub(crate) struct PleState {
    /// Octets E4M3 des lignes du token courant (`[ple_dim]`, partagé CPU/GPU).
    pub(crate) bytes: Buffer,
    /// Anneau `[span, hc·hidden]` des colonnes `normed` passées.
    pub(crate) ring: Buffer,
    pub(crate) context: NgramContext,
    /// Position du token courant dans la séquence.
    pub(crate) step: usize,
}

impl PleState {
    /// Remet la séquence à zéro : contexte EOS et historique de conv vide (le
    /// kernel ignore les colonnes antérieures au pas 0).
    pub(crate) fn reset(&mut self, ngram_size: usize, eos: u64) {
        self.context = NgramContext::new(ngram_size, eos);
        self.step = 0;
    }

    /// Copie les octets lus pour le token courant dans le buffer partagé.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si la longueur ne correspond pas au buffer.
    pub(crate) fn stage_bytes(&mut self, rows: &[u8]) -> Result<()> {
        if rows.len() as u64 != self.bytes.length() {
            return Err(InferError::Dimension(format!(
                "PLE : {} octets lus, {} attendus",
                rows.len(),
                self.bytes.length()
            )));
        }
        // NOTE: le GPU ne lit plus `bytes` : le command buffer précédent est
        // terminé (voir l'en-tête).
        write_u8_buffer(&self.bytes, rows)?;
        Ok(())
    }
}

/// Ajoute la contribution PLE du token courant : `out = h + PLE(h, lignes)`.
/// Les octets du token doivent avoir été déposés par [`PleState::stage_bytes`].
#[expect(
    clippy::too_many_arguments,
    reason = "encodage Metal : pipelines, poids, état, scratch et buffers explicites"
)]
pub(crate) fn encode_ple(
    kernels: &FlashNextKernels,
    executor: &MetalExecutor,
    encoder: &ComputeCommandEncoderRef,
    weights: &PleWeights,
    state: &mut PleState,
    scratch: &PleScratch,
    dims: HcDims,
    h: &BufferRef,
    out: &BufferRef,
) -> Result<()> {
    let width = dims.width();
    encoder.set_compute_pipeline_state(&kernels.ple_gather);
    encoder.set_buffer(0, Some(&state.bytes), 0);
    encoder.set_buffer(1, Some(&weights.lut), 0);
    encoder.set_buffer(2, Some(&scratch.emb), 0);
    set_value(encoder, 3, &(weights.ple_dim as u32));
    dispatch_1d(encoder, weights.ple_dim);
    post_dispatch_barrier(encoder);

    for (weight, target) in [
        (&weights.key_proj, &scratch.key),
        (&weights.value_proj, &scratch.value),
    ] {
        executor.encode_matmul_weight_buffers(
            encoder,
            &scratch.emb,
            1,
            weights.ple_dim,
            weight,
            target,
            true,
        )?;
    }
    post_dispatch_barrier(encoder);
    for (input, norm, target) in [
        (&*scratch.key, &weights.norm_key, &scratch.key_normed),
        (h, &weights.norm_query, &scratch.query),
    ] {
        encode_rms_norm_1p(kernels, encoder, input, norm, target, dims);
    }
    post_dispatch_barrier(encoder);

    encoder.set_compute_pipeline_state(&kernels.ple_gate);
    encoder.set_buffer(0, Some(&scratch.key_normed), 0);
    encoder.set_buffer(1, Some(&scratch.query), 0);
    encoder.set_buffer(2, Some(&scratch.value), 0);
    encoder.set_buffer(3, Some(&scratch.gated), 0);
    set_value(encoder, 4, &(dims.hidden as u32));
    dispatch_groups(encoder, dims.hc);
    post_dispatch_barrier(encoder);

    encode_rms_norm_1p(
        kernels,
        encoder,
        &scratch.gated,
        &weights.norm_conv,
        &scratch.normed,
        dims,
    );
    post_dispatch_barrier(encoder);

    let step = u32::try_from(state.step)
        .map_err(|_| InferError::Dimension(format!("PLE : pas {} hors u32", state.step)))?;
    encoder.set_compute_pipeline_state(&kernels.ple_conv_add);
    encoder.set_buffer(0, Some(h), 0);
    encoder.set_buffer(1, Some(&scratch.gated), 0);
    encoder.set_buffer(2, Some(&scratch.normed), 0);
    encoder.set_buffer(3, Some(&state.ring), 0);
    encoder.set_buffer(4, Some(&weights.conv), 0);
    encoder.set_buffer(5, Some(out), 0);
    set_value(
        encoder,
        6,
        &[
            width as u32,
            weights.kernel as u32,
            weights.dilation as u32,
            step,
        ],
    );
    dispatch_1d(encoder, width);
    post_dispatch_barrier(encoder);
    state.step += 1;
    Ok(())
}

/// Norme RMS groupée par `hidden`, gain `1 + w` (largeur `hc·hidden`).
pub(crate) fn encode_rms_norm_1p(
    kernels: &FlashNextKernels,
    encoder: &ComputeCommandEncoderRef,
    input: &BufferRef,
    weight: &BufferRef,
    out: &BufferRef,
    dims: HcDims,
) {
    encoder.set_compute_pipeline_state(&kernels.rms_norm_1p_groups);
    encoder.set_buffer(0, Some(input), 0);
    encoder.set_buffer(1, Some(weight), 0);
    encoder.set_buffer(2, Some(out), 0);
    set_value(encoder, 3, &(dims.hidden as u32));
    set_value(encoder, 4, &dims.eps);
    dispatch_groups(encoder, dims.hc);
}
