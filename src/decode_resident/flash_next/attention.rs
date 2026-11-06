//! Attention QSA résidente de Flash-Next.
//!
//! Jusqu'à `budget + compress − 1 = 2051` tokens visibles, l'indexeur retient
//! tous les blocs complets plus la queue : la sélection est l'attention causale
//! dense sur le KV. Au-delà (régime creux), l'indexeur score les blocs complets,
//! retient les `budget / compress` meilleurs (top-512, radix exact) et les lignes
//! K/V retenues sont copiées dans un KV compact (≤ 2051 lignes, 8 Mo) sur lequel
//! tourne l'attention existante : en decode, une seule requête, la copie est
//! bornée (la matérialisation par requête n'explose qu'au prefill).
//!
//! Le cache de clés poolées de l'indexeur est alimenté dès le régime dense : un
//! bloc est poolé au token qui le complète, avant les scores de ce token.
//!
//! Primitives réutilisées sans l'orchestrateur Qwen : pas d'`input_norm`
//! (l'entrée sort déjà de l'hyper-connection), normes q/k chargées avec leur
//! gain `1 + w` appliqué une seule fois, `q_proj` entrelacé (query | gate) par
//! tête, RoPE partielle, KV f32, gate `σ` avant `o_proj`.

use metal::{Buffer, BufferRef, ComputeCommandEncoderRef};

use metal::MTLSize;

use super::kernels::FlashNextKernels;
use super::set_value;
use crate::decode_resident::{DecodeResidentState, FullAttentionMetalState};
use crate::metal_backend::{post_dispatch_barrier, MetalExecutor, MetalLinearWeightBuffers};
use crate::{InferError, Result};

/// Visibilité maximale du régime dense : `budget + compress − 1` (2051 pour
/// Flash-Next). Au-delà, l'indexeur écarte des blocs.
pub(crate) fn dense_max_visible(budget: usize, compress: usize) -> usize {
    budget + compress.saturating_sub(1)
}

/// Poids GPU d'une couche QSA (indexeur hors régime dense).
pub(crate) struct QsaLayerWeights {
    /// `[2·heads·head_dim, hidden]`, query et gate entrelacés par tête.
    pub(crate) q_proj: MetalLinearWeightBuffers,
    pub(crate) k_proj: MetalLinearWeightBuffers,
    pub(crate) v_proj: MetalLinearWeightBuffers,
    pub(crate) o_proj: MetalLinearWeightBuffers,
    /// Gains `1 + w` de `q_norm` / `k_norm` (`[head_dim]`).
    pub(crate) q_norm: Buffer,
    pub(crate) k_norm: Buffer,
    /// Indexeur (absent des oracles du seul régime dense).
    pub(crate) indexer: Option<IndexerWeights>,
}

/// Poids GPU de l'indexeur d'une couche QSA.
pub(crate) struct IndexerWeights {
    /// `[(têtes + 1)·d_idx, hidden]` : requêtes puis clé brute.
    pub(crate) qk_proj: MetalLinearWeightBuffers,
    /// Gains `1 + w` de `q_layernorm` / `k_layernorm` (`[d_idx]`).
    pub(crate) q_norm: Buffer,
    pub(crate) k_norm: Buffer,
    /// Fréquences inverses de la RoPE partielle (`[rotary_dim / 2]`), calculées
    /// comme la référence CPU.
    pub(crate) freqs: Buffer,
}

/// Dimensions de l'indexeur.
#[derive(Clone, Copy, Debug)]
pub(crate) struct IndexerDims {
    pub(crate) heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) compress: usize,
    /// Blocs retenus en régime creux (`budget / compress` = 512).
    pub(crate) keep: usize,
}

/// État de l'indexeur d'une couche : clés poolées des blocs complets et clés
/// brutes du bloc en cours (ligne `position mod compress`).
pub(crate) struct IndexerState {
    pub(crate) pooled: Buffer,
    pub(crate) raw: Buffer,
}

/// Intermédiaires de l'indexeur, partagés entre couches.
pub(crate) struct IndexerScratch {
    pub(crate) qk: Buffer,
    pub(crate) q: Buffer,
    pub(crate) scores: Buffer,
    /// `[keep]` u32, blocs retenus triés.
    pub(crate) selected: Buffer,
    /// Oracles : buffer de sélection imposée, gardé vivant jusqu'au pas suivant.
    #[cfg(test)]
    pub(crate) forced_keepalive: std::cell::RefCell<Option<Buffer>>,
}

/// Régime creux d'une couche : indexeur et KV compact du gather.
pub(crate) struct QsaSparse<'a> {
    pub(crate) kernels: &'a FlashNextKernels,
    pub(crate) weights: &'a IndexerWeights,
    pub(crate) state: &'a IndexerState,
    pub(crate) scratch: &'a IndexerScratch,
    pub(crate) gathered: &'a mut FullAttentionMetalState,
    pub(crate) dims: IndexerDims,
    /// Oracles : blocs imposés à la place du top-k (quasi-égalités CPU).
    #[cfg(test)]
    pub(crate) forced: Option<Vec<u32>>,
}

/// Dimensions de l'attention.
#[derive(Clone, Copy, Debug)]
pub(crate) struct QsaDecodeDims {
    pub(crate) hidden: usize,
    pub(crate) heads: usize,
    pub(crate) kv_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) rotary_dim: usize,
    pub(crate) theta: f32,
    pub(crate) eps: f32,
    /// Visibilité maximale du régime dense ([`dense_max_visible`]).
    pub(crate) dense_max: usize,
}

impl QsaDecodeDims {
    fn q_dim(self) -> usize {
        self.heads * self.head_dim
    }
}

/// Intermédiaires de l'attention.
pub(crate) struct QsaScratch {
    pub(crate) q_gate: Buffer,
    pub(crate) q_raw: Buffer,
    pub(crate) gate: Buffer,
    pub(crate) q_roped: Buffer,
    pub(crate) k_raw: Buffer,
    pub(crate) k_roped: Buffer,
    pub(crate) v: Buffer,
    /// `[heads, capacité]`.
    pub(crate) scores: Buffer,
    pub(crate) ctx: Buffer,
    pub(crate) gated: Buffer,
}

/// Encode l'attention du token à `position` : `x [hidden]` → `out [hidden]`,
/// en ajoutant son K/V au cache. Sans `sparse`, seul le régime dense est admis.
#[expect(
    clippy::too_many_arguments,
    reason = "encodage Metal : arène, exécuteur, cache, poids, scratch et buffers explicites"
)]
pub(crate) fn encode_qsa(
    arena: &DecodeResidentState,
    executor: &MetalExecutor,
    encoder: &ComputeCommandEncoderRef,
    kv: &mut FullAttentionMetalState,
    weights: &QsaLayerWeights,
    scratch: &QsaScratch,
    dims: QsaDecodeDims,
    sparse: Option<QsaSparse<'_>>,
    position: usize,
    x: &BufferRef,
    out: &BufferRef,
) -> Result<()> {
    if sparse.is_none() && position >= dims.dense_max {
        return Err(InferError::Config(format!(
            "Flash-Next résident : position {position}, sélection QSA creuse au-delà de \
             {} tokens visibles sans indexeur",
            dims.dense_max
        )));
    }
    for (weight, target) in [
        (&weights.q_proj, &scratch.q_gate),
        (&weights.k_proj, &scratch.k_raw),
        (&weights.v_proj, &scratch.v),
    ] {
        executor.encode_matmul_weight_buffers(encoder, x, 1, dims.hidden, weight, target, true)?;
    }
    if let Some(sparse) = &sparse {
        executor.encode_matmul_weight_buffers(
            encoder,
            x,
            1,
            dims.hidden,
            &sparse.weights.qk_proj,
            &sparse.scratch.qk,
            true,
        )?;
    }
    post_dispatch_barrier(encoder);
    arena.encode_split_q_gate(
        encoder,
        &scratch.q_gate,
        &scratch.q_raw,
        &scratch.gate,
        dims.heads,
        dims.head_dim,
    )?;
    for (input, norm, target, heads) in [
        (
            &scratch.q_raw,
            &weights.q_norm,
            &scratch.q_roped,
            dims.heads,
        ),
        (
            &scratch.k_raw,
            &weights.k_norm,
            &scratch.k_roped,
            dims.kv_heads,
        ),
    ] {
        arena.encode_rms_norm_rope_decode(
            encoder,
            input,
            norm,
            target,
            heads,
            dims.head_dim,
            dims.rotary_dim,
            dims.rotary_dim,
            position,
            dims.eps,
            dims.theta,
        )?;
    }
    kv.encode_append_kv(encoder, &scratch.k_roped, &scratch.v)?;
    let attended = match sparse {
        Some(sparse) => encode_indexer(arena, executor, encoder, kv, sparse, dims, position)?,
        None => None,
    };
    match attended {
        Some(gathered) => gathered.encode_attention_decode_windowed(
            encoder,
            &scratch.q_roped,
            &scratch.scores,
            &scratch.ctx,
            0,
        )?,
        None => kv.encode_attention_decode_windowed(
            encoder,
            &scratch.q_roped,
            &scratch.scores,
            &scratch.ctx,
            0,
        )?,
    }
    arena.encode_attn_gate(
        encoder,
        &scratch.ctx,
        &scratch.gate,
        &scratch.gated,
        dims.q_dim(),
    )?;
    executor.encode_matmul_weight_buffers(
        encoder,
        &scratch.gated,
        1,
        dims.q_dim(),
        &weights.o_proj,
        out,
        true,
    )?;
    post_dispatch_barrier(encoder);
    Ok(())
}

/// Indexeur du token à `position` : range sa clé brute, poole le bloc qu'il
/// complète, puis, en régime creux, score les blocs complets, retient les
/// `keep` meilleurs et copie les lignes K/V retenues dans le KV compact, qu'il
/// renvoie. `None` en régime dense (attention sur tout le KV).
fn encode_indexer<'a>(
    arena: &DecodeResidentState,
    executor: &MetalExecutor,
    encoder: &ComputeCommandEncoderRef,
    kv: &FullAttentionMetalState,
    sparse: QsaSparse<'a>,
    dims: QsaDecodeDims,
    position: usize,
) -> Result<Option<&'a mut FullAttentionMetalState>> {
    let QsaSparse {
        kernels,
        weights,
        state,
        scratch,
        gathered,
        dims: idx,
        #[cfg(test)]
        forced,
    } = sparse;
    let d = idx.head_dim;
    let f32_size = std::mem::size_of::<f32>() as u64;
    // Clé brute → ligne `position mod compress` du bloc en cours.
    executor.encode_copy_with_offsets(
        encoder,
        &scratch.qk,
        (idx.heads * d) as u64 * f32_size,
        &state.raw,
        ((position % idx.compress) * d) as u64 * f32_size,
        d,
    )?;
    post_dispatch_barrier(encoder);
    if position % idx.compress == idx.compress - 1 {
        let block = position / idx.compress;
        encoder.set_compute_pipeline_state(&kernels.index_pool);
        encoder.set_buffer(0, Some(&state.raw), 0);
        encoder.set_buffer(1, Some(&weights.k_norm), 0);
        encoder.set_buffer(2, Some(&weights.freqs), 0);
        encoder.set_buffer(3, Some(&state.pooled), 0);
        set_value(
            encoder,
            4,
            &[
                d as u32,
                idx.compress as u32,
                block as u32,
                (dims.rotary_dim / 2) as u32,
            ],
        );
        set_value(encoder, 5, &dims.eps);
        crate::metal_backend::profile_dispatch();
        encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(d as u64, 1, 1));
        post_dispatch_barrier(encoder);
    }
    let visible = position + 1;
    let complete = visible / idx.compress;
    if complete <= idx.keep {
        return Ok(None);
    }
    // Requêtes de l'indexeur : `q_layernorm` puis RoPE à la position du token.
    arena.encode_rms_norm_rope_decode(
        encoder,
        &scratch.qk,
        &weights.q_norm,
        &scratch.q,
        idx.heads,
        d,
        dims.rotary_dim,
        dims.rotary_dim,
        position,
        dims.eps,
        dims.theta,
    )?;
    encoder.set_compute_pipeline_state(&kernels.index_scores);
    encoder.set_buffer(0, Some(&scratch.q), 0);
    encoder.set_buffer(1, Some(&state.pooled), 0);
    encoder.set_buffer(2, Some(&scratch.scores), 0);
    set_value(
        encoder,
        3,
        &[idx.heads as u32, d as u32, complete as u32, 0],
    );
    crate::metal_backend::profile_dispatch();
    encoder.dispatch_thread_groups(
        MTLSize::new(complete.div_ceil(8) as u64, 1, 1),
        MTLSize::new(256, 1, 1),
    );
    post_dispatch_barrier(encoder);
    #[cfg(test)]
    let imposed = match forced {
        Some(blocks) => {
            if blocks.len() != idx.keep || blocks.iter().any(|&b| b as usize >= complete) {
                return Err(InferError::Dimension(format!(
                    "sélection QSA imposée invalide ({} blocs sur {complete})",
                    blocks.len()
                )));
            }
            let forced_buffer = executor.device().new_buffer_with_data(
                blocks.as_ptr().cast(),
                (blocks.len() * std::mem::size_of::<u32>()) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            encoder.set_compute_pipeline_state(&kernels.copy_u32);
            encoder.set_buffer(0, Some(&forced_buffer), 0);
            encoder.set_buffer(1, Some(&scratch.selected), 0);
            set_value(encoder, 2, &[idx.keep as u32, 0]);
            super::dispatch_1d(encoder, idx.keep);
            post_dispatch_barrier(encoder);
            // Le buffer doit survivre au command buffer : on le retient jusqu'au
            // prochain pas (le pas attend sa fin).
            *scratch.forced_keepalive.borrow_mut() = Some(forced_buffer);
            true
        }
        None => false,
    };
    #[cfg(not(test))]
    let imposed = false;
    if !imposed {
        encoder.set_compute_pipeline_state(&kernels.index_topk);
        encoder.set_buffer(0, Some(&scratch.scores), 0);
        encoder.set_buffer(1, Some(&scratch.selected), 0);
        set_value(encoder, 2, &[complete as u32, idx.keep as u32]);
        crate::metal_backend::profile_dispatch();
        encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(1024, 1, 1));
        post_dispatch_barrier(encoder);
    }
    let count = idx.keep * idx.compress + (visible - complete * idx.compress);
    let kv_dim = dims.kv_heads * dims.head_dim;
    let (keys, values) = kv.dense_f32_buffers()?;
    let (gathered_keys, gathered_values) = gathered.dense_f32_buffers()?;
    encoder.set_compute_pipeline_state(&kernels.gather_kv);
    encoder.set_buffer(0, Some(keys), 0);
    encoder.set_buffer(1, Some(values), 0);
    encoder.set_buffer(2, Some(&scratch.selected), 0);
    encoder.set_buffer(3, Some(gathered_keys), 0);
    encoder.set_buffer(4, Some(gathered_values), 0);
    set_value(
        encoder,
        5,
        &[
            kv_dim as u32,
            idx.keep as u32,
            complete as u32,
            visible as u32,
        ],
    );
    set_value(encoder, 6, &(idx.compress as u32));
    crate::metal_backend::profile_dispatch();
    encoder.dispatch_threads(
        MTLSize::new(kv_dim as u64, count as u64, 1),
        MTLSize::new(256, 1, 1),
    );
    post_dispatch_barrier(encoder);
    gathered.set_gathered_len(count)?;
    Ok(Some(gathered))
}
