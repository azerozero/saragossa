//! Drafter DFlash 2 résident : proposition du bloc spéculatif sur GPU.
//!
//! Le mini-décodeur (5 couches sur le 27B) tourne sur `block` lignes (ancre +
//! masques) dans l'encoder partagé : projections par les matmuls du tronc
//! (quantifiées ou denses), convolution causale dynamique en un kernel, attention
//! en lignes sur un KV de contexte par couche (K/V projetés des hidden states
//! cibles, positions absolues), top-k des logits sur GPU. Le parcours des
//! codebooks du sélecteur reste sur CPU (`DFlashSelector::select_projected`).
//!
//! Sémantique identique à `DFlashDraft::forward_projected_context` (oracle CPU,
//! test `dflash_resident_block_matches_cpu`).

use super::attention::FullAttentionMetalState;
use super::kernels::compile_kernel;
use super::rows::RowsKernels;
use super::utils::byte_offset_f32;
use super::*;
use crate::devtools::{DFlashConv, DFlashDraft};

const DFLASH_KERNELS: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Convolution causale groupée à noyau dynamique (DFlash 2), zéro-remplie au début
// du bloc : out[t,c] = Σ_off (base[which,off,c] + dyn[t,which,off,g(c)]) · x[t-off,c].
kernel void dflash_conv_rows_f32(
    device const float* input [[buffer(0)]],
    device const float* base_kernel [[buffer(1)]],
    device const float* dynamic [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint4& dims [[buffer(4)]],
    constant uint2& params [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint rows = dims.x;
    const uint width = dims.y;
    const uint ksize = dims.z;
    const uint groups = dims.w;
    const uint c = gid.x;
    const uint t = gid.y;
    if (c >= width || t >= rows) { return; }
    const uint group_size = params.x;
    const uint which = params.y;
    const uint per_token = 2u * ksize * groups;
    const uint g = c / group_size;
    const uint taps = min(ksize, t + 1u);
    float acc = 0.0f;
    for (uint off = 0u; off < taps; ++off) {
        const float base = base_kernel[(which * ksize + off) * width + c];
        const float dyn = dynamic[t * per_token + (which * ksize + off) * groups + g];
        acc += (base + dyn) * input[(t - off) * width + c];
    }
    out[t * width + c] = acc;
}

// Top-k par ligne : k tours d'argmax (plus petit index à égalité) ; la valeur
// retenue est neutralisée dans `logits` (scratch consommé). Ligne = tg + dims.z.
kernel void dflash_topk_rows_f32(
    device float* logits [[buffer(0)]],
    device uint* out_index [[buffer(1)]],
    device float* out_value [[buffer(2)]],
    constant uint4& dims [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg [[threadgroup_position_in_grid]]
) {
    const uint cols = dims.x;
    const uint k = dims.y;
    const uint row = tg + dims.z;
    device float* line = logits + row * cols;
    threadgroup float best_val[1024];
    threadgroup uint best_idx[1024];
    for (uint round = 0u; round < k; ++round) {
        float v = -INFINITY;
        uint idx = 0xffffffffu;
        for (uint col = tid; col < cols; col += 1024u) {
            const float x = line[col];
            if (x > v || (x == v && col < idx)) { v = x; idx = col; }
        }
        best_val[tid] = v;
        best_idx[tid] = idx;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 512u; stride > 0u; stride >>= 1u) {
            if (tid < stride) {
                const float ov = best_val[tid + stride];
                const uint oi = best_idx[tid + stride];
                if (ov > best_val[tid] || (ov == best_val[tid] && oi < best_idx[tid])) {
                    best_val[tid] = ov;
                    best_idx[tid] = oi;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0u) {
            out_index[tg * k + round] = best_idx[0];
            out_value[tg * k + round] = best_val[0];
            if (best_idx[0] != 0xffffffffu) { line[best_idx[0]] = -INFINITY; }
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }
}
"#;

const TOPK_THREADS: usize = 1024;
/// Lignes par matmul lors de l'append de contexte (prefill du drafter).
const CONTEXT_CHUNK_ROWS: usize = 64;

struct DFlashResidentConv {
    base_kernel: Buffer,
    kernel_projection: MetalLinearWeightBuffers,
    kernel_size: usize,
    group_size: usize,
    /// `2 · kernel_size · groups` : colonnes du noyau dynamique par token.
    dynamic_cols: usize,
}

struct DFlashResidentLayer {
    input_norm: Buffer,
    post_norm: Buffer,
    q_norm: Buffer,
    k_norm: Buffer,
    q_proj: MetalLinearWeightBuffers,
    k_proj: MetalLinearWeightBuffers,
    v_proj: MetalLinearWeightBuffers,
    o_proj: MetalLinearWeightBuffers,
    gate_proj: MetalLinearWeightBuffers,
    up_proj: MetalLinearWeightBuffers,
    down_proj: MetalLinearWeightBuffers,
    attention_conv: Option<DFlashResidentConv>,
    mlp_conv: Option<DFlashResidentConv>,
    sliding_window: Option<usize>,
    kv: FullAttentionMetalState,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct DFlashResidentDims {
    pub hidden: usize,
    pub inter: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// `target_layers · hidden` : colonnes des hidden states cibles concaténés.
    pub context_cols: usize,
    pub eps: f32,
    pub theta: f32,
}

/// Drafter DFlash résident : poids Metal, KV de contexte par couche et kernels.
pub(crate) struct DFlashResident {
    dims: DFlashResidentDims,
    hidden_norm: Buffer,
    fc: MetalLinearWeightBuffers,
    layers: Vec<DFlashResidentLayer>,
    norm: Buffer,
    selector_projection: Option<MetalLinearWeightBuffers>,
    conv_kernel: ComputePipelineState,
    topk_kernel: ComputePipelineState,
    context_len: usize,
}

impl DFlashResident {
    /// Résout les poids du draft en buffers Metal et alloue un KV de contexte de
    /// `capacity` positions par couche (contexte + bloc transitoire).
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si une forme est incompatible avec les kernels en
    /// lignes (`head_dim` multiple de 32, ≤ 256), si les kernels ne compilent pas
    /// ou si une allocation échoue.
    pub(crate) fn new(
        state: &DecodeResidentState,
        executor: &MetalExecutor,
        draft: &DFlashDraft,
        capacity: usize,
    ) -> Result<Self> {
        let info = &draft.info;
        if info.head_dim == 0 || info.head_dim % 32 != 0 || info.head_dim > 256 {
            return Err(InferError::Config(format!(
                "drafter résident: head_dim={} hors des kernels en lignes (multiple de 32, ≤ 256)",
                info.head_dim
            )));
        }
        if info.conv_kernel_size > 0 && info.hidden_size % info.conv_group_size != 0 {
            return Err(InferError::Config(format!(
                "drafter résident: conv_group_size={} ne divise pas hidden={}",
                info.conv_group_size, info.hidden_size
            )));
        }
        let context_cols = info
            .target_layer_ids
            .len()
            .checked_mul(info.hidden_size)
            .ok_or_else(|| InferError::Shape("drafter résident: contexte déborde".to_string()))?;
        let dims = DFlashResidentDims {
            hidden: info.hidden_size,
            inter: info.intermediate_size,
            q_heads: info.num_attention_heads,
            kv_heads: info.num_key_value_heads,
            head_dim: info.head_dim,
            context_cols,
            eps: info.rms_norm_eps,
            theta: info.rope_theta,
        };
        let conv_kernel = compile_kernel(state.device(), DFLASH_KERNELS, "dflash_conv_rows_f32")?;
        let topk_kernel = compile_kernel(state.device(), DFLASH_KERNELS, "dflash_topk_rows_f32")?;
        if (topk_kernel.max_total_threads_per_threadgroup() as usize) < TOPK_THREADS {
            return Err(InferError::Metal(format!(
                "drafter résident: top-k exige {TOPK_THREADS} threads par threadgroup"
            )));
        }
        let resolve_conv = |conv: &DFlashConv| -> Result<DFlashResidentConv> {
            let groups = info.hidden_size / conv.group_size;
            Ok(DFlashResidentConv {
                base_kernel: executor
                    .cached_buffer_from_f32(conv.base_kernel.data(), "dflash_conv_base")?,
                kernel_projection: executor.resolve_linear_weight_buffers(
                    conv.kernel_projection.weight(),
                    "dflash_conv_projection",
                )?,
                kernel_size: conv.kernel_size,
                group_size: conv.group_size,
                dynamic_cols: 2 * conv.kernel_size * groups,
            })
        };
        let mut layers = Vec::with_capacity(draft.layers.len());
        for (layer_idx, layer) in draft.layers.iter().enumerate() {
            let (gate, up, down) = layer.mlp.projections();
            let sliding_window = info
                .layer_types
                .get(layer_idx)
                .is_some_and(|kind| kind == "sliding_attention")
                .then_some(info.sliding_window)
                .flatten();
            layers.push(DFlashResidentLayer {
                input_norm: executor
                    .cached_buffer_from_f32(layer.input_norm.data(), "dflash_input_norm")?,
                post_norm: executor
                    .cached_buffer_from_f32(layer.post_attention_norm.data(), "dflash_post_norm")?,
                q_norm: executor
                    .cached_buffer_from_f32(layer.attention.q_norm.data(), "dflash_q_norm")?,
                k_norm: executor
                    .cached_buffer_from_f32(layer.attention.k_norm.data(), "dflash_k_norm")?,
                q_proj: executor
                    .resolve_linear_weight_buffers(layer.attention.q_proj.weight(), "dflash_q")?,
                k_proj: executor
                    .resolve_linear_weight_buffers(layer.attention.k_proj.weight(), "dflash_k")?,
                v_proj: executor
                    .resolve_linear_weight_buffers(layer.attention.v_proj.weight(), "dflash_v")?,
                o_proj: executor
                    .resolve_linear_weight_buffers(layer.attention.o_proj.weight(), "dflash_o")?,
                gate_proj: executor.resolve_linear_weight_buffers(gate.weight(), "dflash_gate")?,
                up_proj: executor.resolve_linear_weight_buffers(up.weight(), "dflash_up")?,
                down_proj: executor.resolve_linear_weight_buffers(down.weight(), "dflash_down")?,
                attention_conv: layer
                    .attention_conv
                    .as_ref()
                    .map(resolve_conv)
                    .transpose()?,
                mlp_conv: layer.mlp_conv.as_ref().map(resolve_conv).transpose()?,
                sliding_window,
                kv: state.full_attention_dense(
                    capacity,
                    dims.q_heads,
                    dims.kv_heads,
                    dims.head_dim,
                    false,
                )?,
            });
        }
        Ok(Self {
            dims,
            hidden_norm: executor
                .cached_buffer_from_f32(draft.hidden_norm.data(), "dflash_hidden_norm")?,
            fc: executor.resolve_linear_weight_buffers(draft.fc.weight(), "dflash_fc")?,
            layers,
            norm: executor.cached_buffer_from_f32(draft.norm.data(), "dflash_norm")?,
            selector_projection: draft
                .selector
                .as_ref()
                .map(|selector| {
                    executor.resolve_linear_weight_buffers(
                        selector.hidden_projection.weight(),
                        "dflash_selector_projection",
                    )
                })
                .transpose()?,
            conv_kernel,
            topk_kernel,
            context_len: 0,
        })
    }

    pub(crate) fn dims(&self) -> DFlashResidentDims {
        self.dims
    }

    /// Positions de contexte déjà projetées dans le KV de chaque couche.
    pub(crate) fn context_len(&self) -> usize {
        self.context_len
    }

    /// Déclare `rows` positions de contexte jamais projetées (début du prompt hors
    /// fenêtre) : le KV avance sans écriture, les positions restent absolues.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si une couche n'est pas fenêtrée (elle lirait ces
    /// positions) ou si le KV déborde.
    pub(crate) fn skip_context(&mut self, rows: usize) -> Result<()> {
        if rows == 0 {
            return Ok(());
        }
        if self
            .layers
            .iter()
            .any(|layer| layer.sliding_window.is_none())
        {
            return Err(InferError::Config(
                "drafter résident: skip_context exige des couches fenêtrées".to_string(),
            ));
        }
        let next = self.context_len + rows;
        for layer in &mut self.layers {
            layer.kv.truncate(next)?;
        }
        self.context_len = next;
        Ok(())
    }

    /// Ramène le contexte à `len` positions (rejet d'un bloc déjà projeté).
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si `len` dépasse le contexte courant.
    pub(crate) fn truncate_context(&mut self, len: usize) -> Result<()> {
        if len > self.context_len {
            return Err(InferError::Dimension(format!(
                "drafter résident: truncate {len} > contexte {}",
                self.context_len
            )));
        }
        for layer in &mut self.layers {
            layer.kv.truncate(len)?;
        }
        self.context_len = len;
        Ok(())
    }

    /// Projette `rows` hidden states cibles `[rows, context_cols]` (f32, contigus)
    /// en contexte compact (fc + norm) et appose K/V dans le KV de chaque couche
    /// aux positions `context_len..`.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le contexte déborde la capacité ou si un encodage échoue.
    pub(crate) fn encode_context_append(
        &mut self,
        state: &DecodeResidentState,
        executor: &MetalExecutor,
        encoder: &ComputeCommandEncoderRef,
        owned: &mut Vec<Buffer>,
        target_hidden: &BufferRef,
        rows: usize,
    ) -> Result<()> {
        if rows == 0 {
            return Ok(());
        }
        let dims = self.dims;
        let kv_dim = dims.kv_heads * dims.head_dim;
        let kernels = state.rows_kernels()?;
        let mut start = 0usize;
        while start < rows {
            let chunk = (rows - start).min(CONTEXT_CHUNK_ROWS);
            let projected = state
                .scratch()
                .lease(chunk * dims.hidden, GpuElement::F32)?;
            let context = state
                .scratch()
                .lease(chunk * dims.hidden, GpuElement::F32)?;
            let input_offset = byte_offset_f32(start * dims.context_cols, "drafter contexte")?;
            let input = if input_offset == 0 {
                None
            } else {
                // Les matmuls lisent à l'offset 0 : copie du chunk dans un bail.
                let chunk_in = state
                    .scratch()
                    .lease(chunk * dims.context_cols, GpuElement::F32)?;
                executor.encode_copy_with_offsets(
                    encoder,
                    target_hidden,
                    input_offset,
                    chunk_in.tensor().buffer(),
                    0,
                    chunk * dims.context_cols,
                )?;
                Some(chunk_in)
            };
            let lhs = input
                .as_ref()
                .map_or(target_hidden, |lease| lease.tensor().buffer());
            expect_out_dim(
                executor.encode_matmul_weight_buffers(
                    encoder,
                    lhs,
                    chunk,
                    dims.context_cols,
                    &self.fc,
                    projected.tensor().buffer(),
                    false,
                )?,
                dims.hidden,
                "fc",
            )?;
            executor.encode_rms_norm_rows(
                encoder,
                projected.tensor().buffer(),
                &self.hidden_norm,
                context.tensor().buffer(),
                chunk,
                dims.hidden,
                dims.eps,
            )?;
            let k_raw = state.scratch().lease(chunk * kv_dim, GpuElement::F32)?;
            let k_roped = state.scratch().lease(chunk * kv_dim, GpuElement::F32)?;
            let v_raw = state.scratch().lease(chunk * kv_dim, GpuElement::F32)?;
            let position = self.context_len;
            for layer in &mut self.layers {
                expect_out_dim(
                    executor.encode_matmul_weight_buffers(
                        encoder,
                        context.tensor().buffer(),
                        chunk,
                        dims.hidden,
                        &layer.k_proj,
                        k_raw.tensor().buffer(),
                        false,
                    )?,
                    kv_dim,
                    "k_proj",
                )?;
                expect_out_dim(
                    executor.encode_matmul_weight_buffers(
                        encoder,
                        context.tensor().buffer(),
                        chunk,
                        dims.hidden,
                        &layer.v_proj,
                        v_raw.tensor().buffer(),
                        false,
                    )?,
                    kv_dim,
                    "v_proj",
                )?;
                state.encode_rms_norm_rope_decode_rows(
                    encoder,
                    kernels,
                    k_raw.tensor().buffer(),
                    0,
                    kv_dim,
                    &layer.k_norm,
                    k_roped.tensor().buffer(),
                    kv_dim,
                    dims.kv_heads,
                    dims.head_dim,
                    dims.head_dim,
                    dims.head_dim,
                    position,
                    chunk,
                    dims.eps,
                    dims.theta,
                )?;
                layer.kv.encode_append_kv_rows(
                    encoder,
                    kernels,
                    k_roped.tensor().buffer(),
                    0,
                    kv_dim,
                    v_raw.tensor().buffer(),
                    0,
                    kv_dim,
                    chunk,
                )?;
            }
            self.context_len += chunk;
            owned.push(projected.tensor().buffer().clone());
            owned.push(context.tensor().buffer().clone());
            owned.push(k_raw.tensor().buffer().clone());
            owned.push(k_roped.tensor().buffer().clone());
            owned.push(v_raw.tensor().buffer().clone());
            if let Some(lease) = input {
                owned.push(lease.tensor().buffer().clone());
            }
            start += chunk;
        }
        Ok(())
    }

    /// Mini-décodeur sur `block` lignes : `noise` `[block, hidden]` (embeddings
    /// ancre + masques) → `out_states` `[block, hidden]` (sortie normée, prête
    /// pour la tête). Le KV de contexte est inchangé en sortie.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si `block` est nul, si le bloc déborde la capacité du
    /// KV ou si un encodage échoue.
    #[expect(
        clippy::too_many_arguments,
        reason = "bloc du mini-décodeur : état, exécuteur, encoder, entrée et sortie"
    )]
    pub(crate) fn encode_block(
        &mut self,
        state: &DecodeResidentState,
        executor: &MetalExecutor,
        encoder: &ComputeCommandEncoderRef,
        owned: &mut Vec<Buffer>,
        noise: &BufferRef,
        block: usize,
        out_states: &BufferRef,
    ) -> Result<()> {
        if block == 0 {
            return Err(InferError::Dimension(
                "drafter résident: bloc vide".to_string(),
            ));
        }
        let dims = self.dims;
        let elements = block * dims.hidden;
        let ping = state.scratch().lease(elements, GpuElement::F32)?;
        let pong = state.scratch().lease(elements, GpuElement::F32)?;
        executor.encode_copy_with_offsets(
            encoder,
            noise,
            0,
            ping.tensor().buffer(),
            0,
            elements,
        )?;
        let mut current = ping.tensor().buffer();
        let mut other = pong.tensor().buffer();
        let context_len = self.context_len;
        let kernels = state.rows_kernels()?;
        for layer in &mut self.layers {
            encode_block_layer(
                layer,
                &self.conv_kernel,
                dims,
                context_len,
                state,
                executor,
                encoder,
                owned,
                kernels,
                current,
                other,
                block,
            )?;
            std::mem::swap(&mut current, &mut other);
        }
        executor.encode_rms_norm_rows(
            encoder,
            current,
            &self.norm,
            out_states,
            block,
            dims.hidden,
            dims.eps,
        )?;
        owned.push(ping.tensor().buffer().clone());
        owned.push(pong.tensor().buffer().clone());
        Ok(())
    }

    /// Tête sur les lignes `1..block` des états du bloc : logits par `lm_head`,
    /// top-`k` par ligne (`out_index`/`out_value` : `[(block − 1) · k]`), et
    /// projection du sélecteur `[block, rank]` dans `out_projected` si le draft en a
    /// un (la ligne 0, l'ancre, est calculée puis ignorée par l'appelant).
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si `block < 2`, si `k` est nul ou si un encodage échoue.
    #[expect(
        clippy::too_many_arguments,
        reason = "tête du bloc : états, poids de la tête, sorties top-k et projection"
    )]
    pub(crate) fn encode_block_candidates(
        &self,
        state: &DecodeResidentState,
        executor: &MetalExecutor,
        encoder: &ComputeCommandEncoderRef,
        owned: &mut Vec<Buffer>,
        states: &BufferRef,
        block: usize,
        lm_head: &MetalLinearWeightBuffers,
        top_k: usize,
        out_index: &BufferRef,
        out_value: &BufferRef,
        out_projected: Option<&BufferRef>,
    ) -> Result<()> {
        if block < 2 || top_k == 0 {
            return Err(InferError::Dimension(format!(
                "drafter résident: candidats sur block={block}, top_k={top_k}"
            )));
        }
        let dims = self.dims;
        let vocab = executor.linear_weight_out_dim(lm_head);
        if top_k > vocab {
            return Err(InferError::Dimension(format!(
                "drafter résident: top_k={top_k} > vocabulaire {vocab}"
            )));
        }
        let logits = state.scratch().lease(block * vocab, GpuElement::F32)?;
        expect_out_dim(
            executor.encode_matmul_weight_buffers(
                encoder,
                states,
                block,
                dims.hidden,
                lm_head,
                logits.tensor().buffer(),
                false,
            )?,
            vocab,
            "lm_head",
        )?;
        encoder.set_compute_pipeline_state(&self.topk_kernel);
        encoder.set_buffer(0, Some(logits.tensor().buffer()), 0);
        encoder.set_buffer(1, Some(out_index), 0);
        encoder.set_buffer(2, Some(out_value), 0);
        set_bytes(
            encoder,
            3,
            &[to_u32(vocab, "topk vocab")?, to_u32(top_k, "topk k")?, 1, 0],
        );
        encoder.dispatch_thread_groups(
            MTLSize::new((block - 1) as u64, 1, 1),
            MTLSize::new(TOPK_THREADS as u64, 1, 1),
        );
        crate::metal_backend::post_dispatch_barrier(encoder);
        if let (Some(projection), Some(out)) = (self.selector_projection.as_ref(), out_projected) {
            let rank = executor.linear_weight_out_dim(projection);
            expect_out_dim(
                executor.encode_matmul_weight_buffers(
                    encoder,
                    states,
                    block,
                    dims.hidden,
                    projection,
                    out,
                    false,
                )?,
                rank,
                "selector_projection",
            )?;
        }
        owned.push(logits.tensor().buffer().clone());
        Ok(())
    }
}

/// Une couche du mini-décodeur sur `block` lignes : `layer_in` → `layer_out`.
#[expect(
    clippy::too_many_arguments,
    reason = "data flow d'une couche : état, exécuteur, kernels, ping-pong"
)]
fn encode_block_layer(
    layer: &mut DFlashResidentLayer,
    conv_kernel: &ComputePipelineState,
    dims: DFlashResidentDims,
    context_len: usize,
    state: &DecodeResidentState,
    executor: &MetalExecutor,
    encoder: &ComputeCommandEncoderRef,
    owned: &mut Vec<Buffer>,
    kernels: &RowsKernels,
    layer_in: &BufferRef,
    layer_out: &BufferRef,
    block: usize,
) -> Result<()> {
    let hidden = dims.hidden;
    let q_dim = dims.q_heads * dims.head_dim;
    let kv_dim = dims.kv_heads * dims.head_dim;
    let elements = block * hidden;
    let normed = state.scratch().lease(elements, GpuElement::F32)?;
    executor.encode_rms_norm_rows(
        encoder,
        layer_in,
        &layer.input_norm,
        normed.tensor().buffer(),
        block,
        hidden,
        dims.eps,
    )?;
    let (attn_in, attn_dynamic) = encode_conv_prepare(
        layer.attention_conv.as_ref(),
        conv_kernel,
        state,
        executor,
        encoder,
        normed,
        block,
        hidden,
    )?;
    let q_raw = state.scratch().lease(block * q_dim, GpuElement::F32)?;
    let k_raw = state.scratch().lease(block * kv_dim, GpuElement::F32)?;
    let v_raw = state.scratch().lease(block * kv_dim, GpuElement::F32)?;
    for (weight, out, out_dim, label) in [
        (&layer.q_proj, &q_raw, q_dim, "q_proj"),
        (&layer.k_proj, &k_raw, kv_dim, "k_proj"),
        (&layer.v_proj, &v_raw, kv_dim, "v_proj"),
    ] {
        expect_out_dim(
            executor.encode_matmul_weight_buffers(
                encoder,
                attn_in.tensor().buffer(),
                block,
                hidden,
                weight,
                out.tensor().buffer(),
                false,
            )?,
            out_dim,
            label,
        )?;
    }
    let ctx = state.scratch().lease(block * q_dim, GpuElement::F32)?;
    encode_block_attention(
        layer,
        dims,
        context_len,
        state,
        executor,
        encoder,
        kernels,
        q_raw.tensor().buffer(),
        k_raw.tensor().buffer(),
        v_raw.tensor().buffer(),
        ctx.tensor().buffer(),
        block,
    )?;
    let o_out = state.scratch().lease(elements, GpuElement::F32)?;
    expect_out_dim(
        executor.encode_matmul_weight_buffers(
            encoder,
            ctx.tensor().buffer(),
            block,
            q_dim,
            &layer.o_proj,
            o_out.tensor().buffer(),
            false,
        )?,
        hidden,
        "o_proj",
    )?;
    let attn_out = encode_conv_finish(
        layer.attention_conv.as_ref(),
        conv_kernel,
        state,
        encoder,
        o_out,
        attn_dynamic.as_ref(),
        block,
        hidden,
    )?;
    let attention_state = state.scratch().lease(elements, GpuElement::F32)?;
    executor.encode_add_scaled(
        encoder,
        owned,
        layer_in,
        attn_out.tensor().buffer(),
        attention_state.tensor().buffer(),
        1.0,
        elements,
    )?;
    let mlp_normed = state.scratch().lease(elements, GpuElement::F32)?;
    executor.encode_rms_norm_rows(
        encoder,
        attention_state.tensor().buffer(),
        &layer.post_norm,
        mlp_normed.tensor().buffer(),
        block,
        hidden,
        dims.eps,
    )?;
    let (mlp_in, mlp_dynamic) = encode_conv_prepare(
        layer.mlp_conv.as_ref(),
        conv_kernel,
        state,
        executor,
        encoder,
        mlp_normed,
        block,
        hidden,
    )?;
    let inter_elements = block * dims.inter;
    let gate = state.scratch().lease(inter_elements, GpuElement::F32)?;
    let up = state.scratch().lease(inter_elements, GpuElement::F32)?;
    let act = state.scratch().lease(inter_elements, GpuElement::F32)?;
    for (weight, out, label) in [
        (&layer.gate_proj, &gate, "gate_proj"),
        (&layer.up_proj, &up, "up_proj"),
    ] {
        expect_out_dim(
            executor.encode_matmul_weight_buffers(
                encoder,
                mlp_in.tensor().buffer(),
                block,
                hidden,
                weight,
                out.tensor().buffer(),
                false,
            )?,
            dims.inter,
            label,
        )?;
    }
    executor.encode_swiglu(
        encoder,
        owned,
        gate.tensor().buffer(),
        up.tensor().buffer(),
        act.tensor().buffer(),
        inter_elements,
    )?;
    let down = state.scratch().lease(elements, GpuElement::F32)?;
    expect_out_dim(
        executor.encode_matmul_weight_buffers(
            encoder,
            act.tensor().buffer(),
            block,
            dims.inter,
            &layer.down_proj,
            down.tensor().buffer(),
            false,
        )?,
        hidden,
        "down_proj",
    )?;
    let mlp_out = encode_conv_finish(
        layer.mlp_conv.as_ref(),
        conv_kernel,
        state,
        encoder,
        down,
        mlp_dynamic.as_ref(),
        block,
        hidden,
    )?;
    executor.encode_add_scaled(
        encoder,
        owned,
        attention_state.tensor().buffer(),
        mlp_out.tensor().buffer(),
        layer_out,
        1.0,
        elements,
    )?;
    // Les bails vivent jusqu'au wait : le pool possède les buffers (R1).
    for lease in [
        attn_in,
        q_raw,
        k_raw,
        v_raw,
        ctx,
        attn_out,
        attention_state,
        mlp_in,
        gate,
        up,
        act,
        mlp_out,
    ] {
        owned.push(lease.tensor().buffer().clone());
    }
    if let Some(lease) = attn_dynamic {
        owned.push(lease.tensor().buffer().clone());
    }
    if let Some(lease) = mlp_dynamic {
        owned.push(lease.tensor().buffer().clone());
    }
    Ok(())
}

/// Attention du bloc : RoPE Q/K aux positions `context_len + ligne`, K/V du bloc
/// apposés transitoirement après le contexte, attention causale en lignes (la
/// fenêtre glissante bornant chaque ligne à ses `window` dernières positions),
/// puis KV ramené à `context_len`. Repli ligne par ligne sur le kernel naïf
/// fenêtré hors chemin en lignes ou sous `SARAGOSSA_RUST_DFLASH_WINDOWED_ROWS=0`.
#[expect(
    clippy::too_many_arguments,
    reason = "Q/K/V bruts, contexte de sortie et dimensions d'une couche"
)]
fn encode_block_attention(
    layer: &mut DFlashResidentLayer,
    dims: DFlashResidentDims,
    context_len: usize,
    state: &DecodeResidentState,
    executor: &MetalExecutor,
    encoder: &ComputeCommandEncoderRef,
    kernels: &RowsKernels,
    q_raw: &BufferRef,
    k_raw: &BufferRef,
    v_raw: &BufferRef,
    ctx: &BufferRef,
    block: usize,
) -> Result<()> {
    let q_dim = dims.q_heads * dims.head_dim;
    let kv_dim = dims.kv_heads * dims.head_dim;
    let q_roped = state.scratch().lease(block * q_dim, GpuElement::F32)?;
    let k_roped = state.scratch().lease(block * kv_dim, GpuElement::F32)?;
    for (input, stride, weight, out, heads) in [
        (q_raw, q_dim, &layer.q_norm, &q_roped, dims.q_heads),
        (k_raw, kv_dim, &layer.k_norm, &k_roped, dims.kv_heads),
    ] {
        state.encode_rms_norm_rope_decode_rows(
            encoder,
            kernels,
            input,
            0,
            stride,
            weight,
            out.tensor().buffer(),
            stride,
            heads,
            dims.head_dim,
            dims.head_dim,
            dims.head_dim,
            context_len,
            block,
            dims.eps,
            dims.theta,
        )?;
    }
    let window = layer.sliding_window;
    let windowed = window.is_some_and(|window| context_len + block > window);
    if (!windowed || crate::runtime_flags::dflash_windowed_rows())
        && layer.kv.rows_fused_supported()
    {
        layer.kv.encode_append_kv_rows(
            encoder,
            kernels,
            k_roped.tensor().buffer(),
            0,
            kv_dim,
            v_raw,
            0,
            kv_dim,
            block,
        )?;
        layer.kv.encode_attention_decode_rows(
            encoder,
            kernels,
            q_roped.tensor().buffer(),
            ctx,
            block,
            window.filter(|_| windowed),
        )?;
    } else {
        // Ligne par ligne : chaque ligne n'attend que le contexte et les lignes ≤ elle.
        let q_row = state.scratch().lease(q_dim, GpuElement::F32)?;
        let ctx_row = state.scratch().lease(q_dim, GpuElement::F32)?;
        let scores = state
            .scratch()
            .lease(dims.q_heads * layer.kv.capacity(), GpuElement::F32)?;
        for row in 0..block {
            let q_offset = byte_offset_f32(row * q_dim, "drafter q ligne")?;
            let kv_offset = byte_offset_f32(row * kv_dim, "drafter kv ligne")?;
            layer.kv.encode_append_kv_rows(
                encoder,
                kernels,
                k_roped.tensor().buffer(),
                kv_offset,
                kv_dim,
                v_raw,
                kv_offset,
                kv_dim,
                1,
            )?;
            executor.encode_copy_with_offsets(
                encoder,
                q_roped.tensor().buffer(),
                q_offset,
                q_row.tensor().buffer(),
                0,
                q_dim,
            )?;
            let window_start =
                window.map_or(0, |window| (context_len + row + 1).saturating_sub(window));
            layer.kv.encode_attention_decode_windowed(
                encoder,
                q_row.tensor().buffer(),
                scores.tensor().buffer(),
                ctx_row.tensor().buffer(),
                window_start,
            )?;
            executor.encode_copy_with_offsets(
                encoder,
                ctx_row.tensor().buffer(),
                0,
                ctx,
                q_offset,
                q_dim,
            )?;
        }
    }
    layer.kv.truncate(context_len)
}

/// `prepare` de la convolution : noyau dynamique (projection) puis entrée convolée
/// (noyau 0). Sans convolution, renvoie l'entrée telle quelle.
#[expect(
    clippy::too_many_arguments,
    reason = "convolution d'une sous-couche : état, exécuteur, entrée et dimensions"
)]
fn encode_conv_prepare(
    conv: Option<&DFlashResidentConv>,
    conv_kernel: &ComputePipelineState,
    state: &DecodeResidentState,
    executor: &MetalExecutor,
    encoder: &ComputeCommandEncoderRef,
    input: ScratchLease,
    rows: usize,
    width: usize,
) -> Result<(ScratchLease, Option<ScratchLease>)> {
    let Some(conv) = conv else {
        return Ok((input, None));
    };
    let dynamic = state
        .scratch()
        .lease(rows * conv.dynamic_cols, GpuElement::F32)?;
    expect_out_dim(
        executor.encode_matmul_weight_buffers(
            encoder,
            input.tensor().buffer(),
            rows,
            width,
            &conv.kernel_projection,
            dynamic.tensor().buffer(),
            false,
        )?,
        conv.dynamic_cols,
        "kernel_projection",
    )?;
    let output = state.scratch().lease(rows * width, GpuElement::F32)?;
    encode_conv(
        conv,
        conv_kernel,
        encoder,
        input.tensor().buffer(),
        dynamic.tensor().buffer(),
        output.tensor().buffer(),
        rows,
        width,
        0,
    )?;
    Ok((output, Some(dynamic)))
}

/// `finish` de la convolution : sortie de la sous-couche convolée (noyau 1) avec le
/// noyau dynamique de `prepare`. Sans convolution, renvoie l'entrée telle quelle.
#[expect(
    clippy::too_many_arguments,
    reason = "convolution d'une sous-couche : état, entrée, noyau dynamique et dimensions"
)]
fn encode_conv_finish(
    conv: Option<&DFlashResidentConv>,
    conv_kernel: &ComputePipelineState,
    state: &DecodeResidentState,
    encoder: &ComputeCommandEncoderRef,
    input: ScratchLease,
    dynamic: Option<&ScratchLease>,
    rows: usize,
    width: usize,
) -> Result<ScratchLease> {
    let (Some(conv), Some(dynamic)) = (conv, dynamic) else {
        return Ok(input);
    };
    let output = state.scratch().lease(rows * width, GpuElement::F32)?;
    encode_conv(
        conv,
        conv_kernel,
        encoder,
        input.tensor().buffer(),
        dynamic.tensor().buffer(),
        output.tensor().buffer(),
        rows,
        width,
        1,
    )?;
    Ok(output)
}

#[expect(
    clippy::too_many_arguments,
    reason = "kernel de convolution : trois buffers, dimensions et noyau choisi"
)]
fn encode_conv(
    conv: &DFlashResidentConv,
    conv_kernel: &ComputePipelineState,
    encoder: &ComputeCommandEncoderRef,
    input: &BufferRef,
    dynamic: &BufferRef,
    output: &BufferRef,
    rows: usize,
    width: usize,
    which: u32,
) -> Result<()> {
    let groups = width / conv.group_size;
    encoder.set_compute_pipeline_state(conv_kernel);
    encoder.set_buffer(0, Some(input), 0);
    encoder.set_buffer(1, Some(&conv.base_kernel), 0);
    encoder.set_buffer(2, Some(dynamic), 0);
    encoder.set_buffer(3, Some(output), 0);
    set_bytes(
        encoder,
        4,
        &[
            to_u32(rows, "conv rows")?,
            to_u32(width, "conv width")?,
            to_u32(conv.kernel_size, "conv kernel")?,
            to_u32(groups, "conv groups")?,
        ],
    );
    set_bytes(
        encoder,
        5,
        &[to_u32(conv.group_size, "conv group_size")?, which],
    );
    let threads = (conv_kernel.max_total_threads_per_threadgroup() as usize)
        .min(width)
        .max(1);
    encoder.dispatch_threads(
        MTLSize::new(width as u64, rows as u64, 1),
        MTLSize::new(threads as u64, 1, 1),
    );
    // Sortie consommée par le matmul suivant : barrière en mode concurrent.
    crate::metal_backend::post_dispatch_barrier(encoder);
    Ok(())
}

fn expect_out_dim(actual: usize, expected: usize, label: &str) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(InferError::Dimension(format!(
            "drafter résident: {label} sort {actual}, attendu {expected}"
        )))
    }
}

fn to_u32(value: usize, label: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| InferError::Dimension(format!("{label} hors u32")))
}

fn set_bytes<T: Copy>(encoder: &ComputeCommandEncoderRef, index: u64, values: &[T]) {
    encoder.set_bytes(
        index,
        std::mem::size_of_val(values) as u64,
        values.as_ptr().cast::<c_void>(),
    );
}
