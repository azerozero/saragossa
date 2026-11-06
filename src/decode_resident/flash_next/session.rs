//! Session de decode résident Flash-Next : un token = un command buffer.
//!
//! Par token : embedding (CPU, une ligne u8 déquantifiée, recopiée sur les
//! `hc` flux), hachage n-gram et lecture parallèle des lignes PLE avant le
//! commit, puis sur GPU les 48 couches (hc d'attention, DeltaNet ou QSA dense,
//! hc du MLP, MoE), le mixer final et le lm_head. Le decode attend chaque
//! token : les buffers partagés (embedding, octets PLE) ne sont réécrits
//! qu'après la fin du command buffer précédent.

use metal::{Buffer, CommandQueue, MTLResourceOptions};

use super::attention::{encode_qsa_dense, QsaDecodeDims, QsaScratch, DENSE_MAX_VISIBLE};
use super::hc::{encode_hc_combine, encode_hc_mix, HcDims, HcScratch};
use super::kernels::FlashNextKernels;
use super::ple::{encode_ple, PleScratch, PleState};
use super::weights::{FlashNextWeights, MixerWeights};
use crate::decode_resident::{DecodeResidentState, FullAttentionMetalState};
use crate::metal_backend::{
    commit_and_wait, read_f32_buffer, read_u32_buffer, write_f32_buffer, LinearAttentionMetalState,
    LinearAttentionStepSpec, LinearAttnResidentDims, MetalExecutor,
};
use crate::qwen4_exp::ngram::NgramContext;
use crate::qwen_loader::qwen4_exp::Qwen4ExpCheckpoint;
use crate::{InferError, MemoryGuard, Result};

/// État récurrent ou cache KV d'une couche.
enum LayerState {
    Linear(LinearAttentionMetalState),
    Full(Box<FullAttentionMetalState>),
}

/// Sorties d'un pas.
pub(crate) struct StepOutput {
    pub(crate) argmax: u32,
    /// Logits `[vocab]` si demandés.
    pub(crate) logits: Option<Vec<f32>>,
    /// Résiduel élargi `[hc·hidden]` après chaque couche, si demandé.
    pub(crate) layers: Option<Vec<Vec<f32>>>,
    /// Préparation CPU avant le commit (embedding, hachage, lectures PLE).
    pub(crate) host: std::time::Duration,
}

/// Decode résident Flash-Next d'une séquence (≤ 2051 tokens).
pub(crate) struct FlashNextSession {
    ckpt: Qwen4ExpCheckpoint,
    executor: MetalExecutor,
    arena: DecodeResidentState,
    kernels: FlashNextKernels,
    queue: CommandQueue,
    weights: FlashNextWeights,
    states: Vec<LayerState>,
    ple_state: PleState,
    h: [Buffer; 2],
    x: Buffer,
    branch: Buffer,
    logits: Buffer,
    index: Buffer,
    capture: Buffer,
    hc_scratch: HcScratch,
    ple_scratch: PleScratch,
    qsa_scratch: QsaScratch,
    hc_dims: HcDims,
    qsa_dims: QsaDecodeDims,
    linear_spec: LinearAttentionStepSpec,
    linear_dims: LinearAttnResidentDims,
    position: usize,
    /// Oracles : experts imposés au prochain pas, par couche (consommés).
    #[cfg(test)]
    pub(crate) forced_routing: Vec<Option<Vec<u32>>>,
}

fn zeros(executor: &MetalExecutor, len: usize) -> Buffer {
    executor.device().new_buffer(
        (len.max(1) * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

impl FlashNextSession {
    /// Ouvre le checkpoint, charge le tronc sur GPU et alloue les états pour
    /// `capacity` tokens (au plus 2051, régime QSA dense).
    ///
    /// # Errors
    ///
    /// Renvoie une erreur de lecture, de chargement, de garde mémoire, ou si
    /// `capacity` dépasse le régime dense.
    pub(crate) fn open(
        ckpt: Qwen4ExpCheckpoint,
        capacity: usize,
        guard: Option<&MemoryGuard>,
    ) -> Result<Self> {
        if capacity == 0 || capacity > DENSE_MAX_VISIBLE {
            return Err(InferError::Config(format!(
                "Flash-Next résident : capacité {capacity} hors ]0, {DENSE_MAX_VISIBLE}]"
            )));
        }
        let executor = MetalExecutor::new()?;
        let arena = DecodeResidentState::new(executor.device().clone())?;
        let kernels = FlashNextKernels::new(executor.device())?;
        let weights = FlashNextWeights::load(&executor, &ckpt, guard)?;
        let c = &ckpt.config;
        let hc_dims = HcDims {
            hidden: c.hidden,
            hc: c.hc_count,
            lowrank: c.hc_lowrank,
            eps: c.rms_eps,
        };
        let qsa_dims = QsaDecodeDims {
            hidden: c.hidden,
            heads: c.heads,
            kv_heads: c.kv_heads,
            head_dim: c.head_dim,
            rotary_dim: c.rotary_dim,
            theta: c.rope_theta as f32,
            eps: c.rms_eps,
        };
        let linear_spec = LinearAttentionStepSpec {
            num_key_heads: c.linear_key_heads,
            num_value_heads: c.linear_value_heads,
            key_head_dim: c.linear_key_dim,
            value_head_dim: c.linear_value_dim,
            conv_kernel_dim: c.conv_kernel,
            rms_eps: c.rms_eps,
            l2_qk_sigmoid_gate: true,
        };
        let key_dim = c.linear_key_heads * c.linear_key_dim;
        let value_dim = c.linear_value_heads * c.linear_value_dim;
        let linear_dims = LinearAttnResidentDims {
            in_dim: c.hidden,
            conv_dim: 2 * key_dim + value_dim,
            value_dim,
            key_dim,
        };
        let width = hc_dims.width();
        let (q_dim, kv_dim) = (c.heads * c.head_dim, c.kv_heads * c.head_dim);
        let mut session = Self {
            arena,
            kernels,
            queue: executor.device().new_command_queue(),
            weights,
            states: Vec::with_capacity(c.layer_kinds.len()),
            ple_state: PleState {
                bytes: executor
                    .device()
                    .new_buffer(c.ple_dim as u64, MTLResourceOptions::StorageModeShared),
                ring: zeros(&executor, (c.ple_conv_kernel - 1) * c.ngram_size * width),
                context: NgramContext::new(c.ngram_size, c.eos),
                step: 0,
            },
            h: [zeros(&executor, width), zeros(&executor, width)],
            x: zeros(&executor, c.hidden),
            branch: zeros(&executor, c.hidden),
            logits: zeros(&executor, c.vocab),
            index: executor
                .device()
                .new_buffer(4, MTLResourceOptions::StorageModeShared),
            capture: zeros(&executor, c.layer_kinds.len() * width),
            hc_scratch: HcScratch {
                normed: zeros(&executor, width),
                low: zeros(&executor, c.hc_lowrank),
                up: zeros(&executor, width),
                inject: zeros(&executor, c.hc_count),
            },
            ple_scratch: PleScratch {
                emb: zeros(&executor, c.ple_dim),
                key: zeros(&executor, width),
                key_normed: zeros(&executor, width),
                value: zeros(&executor, c.hidden),
                query: zeros(&executor, width),
                gated: zeros(&executor, width),
                normed: zeros(&executor, width),
            },
            qsa_scratch: QsaScratch {
                q_gate: zeros(&executor, 2 * q_dim),
                q_raw: zeros(&executor, q_dim),
                gate: zeros(&executor, q_dim),
                q_roped: zeros(&executor, q_dim),
                k_raw: zeros(&executor, kv_dim),
                k_roped: zeros(&executor, kv_dim),
                v: zeros(&executor, kv_dim),
                scores: zeros(&executor, c.heads * capacity),
                ctx: zeros(&executor, q_dim),
                gated: zeros(&executor, q_dim),
            },
            hc_dims,
            qsa_dims,
            linear_spec,
            linear_dims,
            position: 0,
            #[cfg(test)]
            forced_routing: Vec::new(),
            executor,
            ckpt,
        };
        session.states = session.fresh_states(capacity)?;
        Ok(session)
    }

    fn fresh_states(&self, capacity: usize) -> Result<Vec<LayerState>> {
        let c = &self.ckpt.config;
        let spec = self.linear_spec;
        let dims = self.linear_dims;
        let conv_len = (spec.conv_kernel_dim - 1) * dims.conv_dim;
        let ssm_len = spec.num_value_heads * spec.value_head_dim * spec.key_head_dim;
        c.layer_kinds
            .iter()
            .map(|kind| match kind {
                crate::qwen_loader::qwen4_exp::Qwen4ExpLayerKind::Linear => {
                    let mut state = None;
                    self.executor.ensure_linear_attention_metal_state(
                        &mut state,
                        &vec![0.0; conv_len],
                        &vec![0.0; ssm_len],
                        dims.conv_dim,
                        conv_len,
                        ssm_len,
                        spec,
                    )?;
                    state
                        .map(LayerState::Linear)
                        .ok_or_else(|| InferError::Metal("état DeltaNet non alloué".to_string()))
                }
                crate::qwen_loader::qwen4_exp::Qwen4ExpLayerKind::Full => {
                    Ok(LayerState::Full(Box::new(
                        self.arena.full_attention_dense(
                            capacity, c.heads, c.kv_heads, c.head_dim, false,
                        )?,
                    )))
                }
            })
            .collect()
    }

    /// Position du prochain token.
    pub(crate) fn position(&self) -> usize {
        self.position
    }

    /// Décode `token` à la position courante. `logits` et `layers` demandent la
    /// relecture des logits et des résiduels par couche (oracles).
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si la capacité est atteinte, sur lecture n-gram ou
    /// sur encodage et exécution Metal.
    pub(crate) fn step(&mut self, token: usize, logits: bool, layers: bool) -> Result<StepOutput> {
        let c = &self.ckpt.config;
        if self.position >= DENSE_MAX_VISIBLE {
            return Err(InferError::Config(format!(
                "Flash-Next résident : {DENSE_MAX_VISIBLE} tokens atteints (régime dense)"
            )));
        }
        let host_started = std::time::Instant::now();
        // Embedding : ligne u8 déquantifiée, répétée sur les `hc` flux.
        let row = self.ckpt.affine_rows(
            &Qwen4ExpCheckpoint::trunk("embed_tokens"),
            None,
            token..token + 1,
        )?;
        let mut streams = Vec::with_capacity(self.hc_dims.width());
        for _ in 0..c.hc_count {
            streams.extend_from_slice(row.data());
        }
        write_f32_buffer(&self.h[0], &streams)?;
        // PLE : hachage et lectures n-gram terminés avant le commit.
        let p = &self.ckpt.ngram_params;
        let ids = self.ple_state.context.step(
            token as u64,
            c.heads_per_ngram,
            &p.multipliers,
            &p.vocab_sizes,
            &p.offsets,
        );
        let mut bytes = vec![0_u8; c.ple_dim];
        self.ckpt.ngram_rows_raw(&ids, &mut bytes)?;
        self.ple_state.stage_bytes(&bytes)?;
        let host = host_started.elapsed();

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let mut owned = Vec::new();
        let width = self.hc_dims.width();
        let mut cur = 0_usize;
        for (index, (layer, state)) in self
            .weights
            .layers
            .iter()
            .zip(self.states.iter_mut())
            .enumerate()
        {
            if index == c.ple_layer {
                encode_ple(
                    &self.kernels,
                    &self.executor,
                    encoder,
                    &self.weights.ple,
                    &mut self.ple_state,
                    &self.ple_scratch,
                    self.hc_dims,
                    &self.h[cur],
                    &self.h[1 - cur],
                )?;
                cur = 1 - cur;
            }
            encode_hc_mix(
                &self.kernels,
                &self.executor,
                encoder,
                &layer.attn_hc,
                &self.hc_scratch,
                self.hc_dims,
                &self.h[cur],
                &self.x,
            )?;
            match (&layer.mixer, state) {
                (MixerWeights::Linear(w), LayerState::Linear(state)) => {
                    self.executor.encode_linear_attn_resident_dense_buffers(
                        encoder,
                        &self.x,
                        None,
                        &self.branch,
                        w,
                        state,
                        self.linear_spec,
                        self.linear_dims,
                    )?;
                }
                (MixerWeights::Full(w), LayerState::Full(kv)) => {
                    encode_qsa_dense(
                        &self.arena,
                        &self.executor,
                        encoder,
                        kv,
                        w,
                        &self.qsa_scratch,
                        self.qsa_dims,
                        self.position,
                        &self.x,
                        &self.branch,
                    )?;
                }
                _ => {
                    return Err(InferError::Config(format!(
                        "couche {index} : poids et état de types différents"
                    )))
                }
            }
            crate::metal_backend::post_dispatch_barrier(encoder);
            encode_hc_combine(
                &self.kernels,
                encoder,
                &self.hc_scratch,
                self.hc_dims,
                &self.h[cur],
                &self.branch,
                &self.h[1 - cur],
            );
            cur = 1 - cur;
            encode_hc_mix(
                &self.kernels,
                &self.executor,
                encoder,
                &layer.mlp_hc,
                &self.hc_scratch,
                self.hc_dims,
                &self.h[cur],
                &self.x,
            )?;
            #[cfg(test)]
            if let Some(experts) = self.forced_routing.get_mut(index).and_then(Option::take) {
                crate::metal_backend::force_next_topk(experts);
            }
            self.executor.encode_moe_shared_buffers(
                encoder,
                &mut owned,
                &self.x,
                None,
                &self.branch,
                c.hidden,
                &layer.moe,
                c.top_k,
            )?;
            #[cfg(test)]
            if crate::metal_backend::forced_topk_pending() {
                return Err(InferError::Metal(format!(
                    "couche {index} : routage imposé non consommé par le MoE"
                )));
            }
            crate::metal_backend::post_dispatch_barrier(encoder);
            encode_hc_combine(
                &self.kernels,
                encoder,
                &self.hc_scratch,
                self.hc_dims,
                &self.h[cur],
                &self.branch,
                &self.h[1 - cur],
            );
            cur = 1 - cur;
            if layers {
                self.executor.encode_copy_with_offsets(
                    encoder,
                    &self.h[cur],
                    0,
                    &self.capture,
                    (index * width * std::mem::size_of::<f32>()) as u64,
                    width,
                )?;
            }
        }
        encode_hc_mix(
            &self.kernels,
            &self.executor,
            encoder,
            &self.weights.mixer,
            &self.hc_scratch,
            self.hc_dims,
            &self.h[cur],
            &self.x,
        )?;
        if logits {
            self.executor.encode_matmul_weight_buffers(
                encoder,
                &self.x,
                1,
                c.hidden,
                &self.weights.lm_head,
                &self.logits,
                true,
            )?;
        }
        self.executor.encode_lm_head_argmax_buffers(
            encoder,
            &mut owned,
            &self.x,
            &self.weights.lm_head,
            &self.index,
            c.hidden,
        )?;
        encoder.end_encoding();
        commit_and_wait(command_buffer)?;
        self.position += 1;
        let argmax = read_u32_buffer(&self.index, 1)?
            .first()
            .copied()
            .ok_or_else(|| InferError::Metal("argmax résident absent".to_string()))?;
        Ok(StepOutput {
            argmax,
            logits: logits
                .then(|| read_f32_buffer(&self.logits, c.vocab))
                .transpose()?,
            layers: layers
                .then(|| {
                    read_f32_buffer(&self.capture, c.layer_kinds.len() * width)
                        .map(|all| all.chunks_exact(width).map(<[f32]>::to_vec).collect())
                })
                .transpose()?,
            host,
        })
    }
}
