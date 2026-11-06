//! Session de decode résident Flash-Next : un token = un command buffer.
//!
//! Par token : embedding (CPU, une ligne u8 déquantifiée, recopiée sur les
//! `hc` flux), hachage n-gram et lecture parallèle des lignes PLE avant le
//! commit, puis sur GPU les 48 couches (hc d'attention, DeltaNet ou QSA dense,
//! hc du MLP, MoE), le mixer final et le lm_head. Le decode attend chaque
//! token : les buffers partagés (embedding, octets PLE) ne sont réécrits
//! qu'après la fin du command buffer précédent.

use metal::{Buffer, CommandQueue, MTLResourceOptions};

use super::attention::{dense_max_visible, encode_qsa_dense, QsaDecodeDims, QsaScratch};
use super::encoders::{Brick, Encoders};
use super::hc::{encode_hc_combine, encode_hc_mix, HcDims, HcScratch};
use super::kernels::FlashNextKernels;
use super::ple::{encode_ple, PleScratch, PleState};
use super::weights::{FlashNextWeights, LoadStage, MixerWeights};
use crate::decode_resident::{DecodeResidentState, FullAttentionMetalState};
use crate::metal_backend::{
    commit_and_wait, read_f32_buffer, read_u32_buffer, write_f32_buffer, LinearAttentionMetalState,
    LinearAttentionStepSpec, LinearAttnResidentDims, MetalExecutor,
};
use crate::qwen4_exp::ngram::NgramContext;
use crate::qwen_loader::qwen4_exp::{Qwen4ExpCheckpoint, Qwen4ExpConfig};
use crate::{InferError, MemoryGuard, Result};

/// État récurrent ou cache KV d'une couche.
enum LayerState {
    Linear(LinearAttentionMetalState),
    Full(Box<FullAttentionMetalState>),
}

/// Sorties d'un pas.
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "pas encore d'entrée de production (serve, étape 9) : exercé par les oracles"
    )
)]
pub(crate) struct StepOutput {
    pub(crate) argmax: u32,
    /// Logits `[vocab]` si demandés.
    pub(crate) logits: Option<Vec<f32>>,
    /// Résiduel élargi `[hc·hidden]` après chaque couche, si demandé.
    pub(crate) layers: Option<Vec<Vec<f32>>>,
    /// Experts routés par le GPU `[couche][slot]`, si les résiduels sont demandés
    /// (oracles seulement).
    #[cfg(test)]
    pub(crate) routing: Option<Vec<Vec<u32>>>,
    /// Préparation CPU avant le commit (embedding, hachage, lectures PLE).
    pub(crate) host: std::time::Duration,
}

/// Decode résident Flash-Next d'une séquence (≤ 2051 tokens).
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "pas encore d'entrée de production (serve, étape 9) : exercé par les oracles"
    )
)]
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
    capacity: usize,
    /// États DeltaNet initiaux (nuls), pour [`Self::reset`].
    initial_linear: Vec<Option<LinearAttentionMetalState>>,
    /// Profil GPU par brique (`SARAGOSSA_FLASH_NEXT_PROFILE=1`).
    #[cfg(feature = "devtools")]
    profiler: Option<super::profile::StepProfiler>,
    /// Vrai après une erreur d'exécution GPU : états incertains, pas refusés.
    poisoned: bool,
    /// Oracles : experts routés par couche du dernier pas `[couches × top_k]`.
    #[cfg(test)]
    routing_capture: Buffer,
    /// Oracles : prochaine lecture n-gram en échec.
    #[cfg(test)]
    pub(crate) fail_next_ngram_read: bool,
    /// Oracles : échec de `reset` injecté après la restauration de la couche donnée.
    #[cfg(test)]
    pub(crate) fail_reset_after_layer: Option<usize>,
    /// Oracles : échec d'encodage injecté après la couche donnée.
    #[cfg(test)]
    pub(crate) fail_encode_after_layer: Option<usize>,
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

/// Octets des états et intermédiaires d'une session de `capacity` tokens :
/// - KV f32 des couches QSA ;
/// - états conv/SSM du DeltaNet, **deux fois** (états vivants et instantanés
///   initiaux du `reset`) ;
/// - anneau et octets PLE, buffers de la session (résiduels, logits, capture,
///   scratch hc/PLE/QSA) ;
/// - scratch interne de l'exécuteur, alloué au premier pas (DeltaNet, MoE,
///   argmax), majoré.
pub(crate) fn session_state_bytes(c: &Qwen4ExpConfig, capacity: usize) -> u64 {
    let f32_bytes = std::mem::size_of::<f32>();
    let width = c.hc_width();
    let full = c
        .layer_kinds
        .iter()
        .filter(|k| **k == crate::qwen_loader::qwen4_exp::Qwen4ExpLayerKind::Full)
        .count();
    let linear = c.layer_kinds.len() - full;
    let kv = full * capacity * 2 * c.kv_heads * c.head_dim;
    let conv_dim =
        2 * c.linear_key_heads * c.linear_key_dim + c.linear_value_heads * c.linear_value_dim;
    let value_dim = c.linear_value_heads * c.linear_value_dim;
    let deltanet = 2
        * linear
        * ((c.conv_kernel - 1) * conv_dim
            + c.linear_value_heads * c.linear_value_dim * c.linear_key_dim);
    let ple = (c.ple_conv_kernel - 1) * c.ngram_size * width + c.ple_dim;
    let q_dim = c.heads * c.head_dim;
    let kv_dim = c.kv_heads * c.head_dim;
    let session = 2 * width // résiduels ping-pong
        + 2 * c.hidden // entrée de branche et sortie de mélangeur
        + c.vocab + 1 // logits et argmax
        + c.layer_kinds.len() * width // capture des résiduels
        + 2 * width + c.hc_lowrank + c.hc_count * super::hc::INJECT_CHUNKS // scratch hc
        + c.ple_dim + 5 * width + c.hidden // scratch PLE
        + 6 * q_dim + 3 * kv_dim + c.heads * capacity; // scratch QSA
    let executor = 4 * conv_dim + 6 * value_dim + 4 * c.linear_value_heads // DeltaNet
        + 2 * c.experts + 2 * c.top_k * (3 * c.moe_inter + c.hidden) // MoE routé
        + 3 * c.shared_inter + 2 * c.hidden // expert partagé
        + c.vocab; // argmax du lm_head
    ((kv + deltanet + ple + session + executor) * f32_bytes) as u64
}

/// Refuse, avant toute allocation, les variantes que le chemin résident ne
/// porte pas, et renvoie la visibilité maximale du régime dense.
///
/// # Errors
///
/// Renvoie [`InferError::Config`] pour une variante non prise en charge.
pub(crate) fn validate_resident_config(c: &Qwen4ExpConfig) -> Result<usize> {
    let refuse = |what: String| -> Result<usize> {
        Err(InferError::Config(format!(
            "Flash-Next résident : {what} non pris en charge"
        )))
    };
    if !c.norm_topk_prob {
        return refuse("norm_topk_prob = false".to_string());
    }
    if !c.sigmoid_output_gate {
        return refuse("gate DeltaNet SiLU".to_string());
    }
    if c.top_k == 0 || c.top_k > 16 || c.top_k > c.experts {
        return refuse(format!("top-k {} sur {} experts", c.top_k, c.experts));
    }
    if c.linear_key_dim != 128 || c.linear_value_dim != 128 || c.conv_kernel != 4 {
        return refuse(format!(
            "DeltaNet d_k {} / d_v {} / noyau {}",
            c.linear_key_dim, c.linear_value_dim, c.conv_kernel
        ));
    }
    if c.linear_key_heads == 0 || c.linear_value_heads % c.linear_key_heads != 0 {
        return refuse(format!(
            "DeltaNet {} têtes V pour {} têtes K",
            c.linear_value_heads, c.linear_key_heads
        ));
    }
    if c.kv_heads == 0 || c.heads % c.kv_heads != 0 || c.rotary_dim > c.head_dim {
        return refuse(format!(
            "attention {} têtes / {} KV, RoPE {} sur {}",
            c.heads, c.kv_heads, c.rotary_dim, c.head_dim
        ));
    }
    if c.indexer_compress == 0 || c.indexer_budget % c.indexer_compress != 0 {
        return refuse(format!(
            "budget QSA {} par blocs de {}",
            c.indexer_budget, c.indexer_compress
        ));
    }
    if c.ple_conv_kernel == 0 || c.ple_layer >= c.layer_kinds.len() {
        return refuse(format!(
            "PLE couche {} noyau {}",
            c.ple_layer, c.ple_conv_kernel
        ));
    }
    Ok(dense_max_visible(c.indexer_budget, c.indexer_compress))
}

#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "pas encore d'entrée de production (serve, étape 9) : exercé par les oracles"
    )
)]
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
        let dense_max = validate_resident_config(&ckpt.config)?;
        if capacity == 0 || capacity > dense_max {
            return Err(InferError::Config(format!(
                "Flash-Next résident : capacité {capacity} hors ]0, {dense_max}]"
            )));
        }
        let executor = MetalExecutor::new()?;
        let arena = DecodeResidentState::new(executor.device().clone())?;
        let kernels = FlashNextKernels::new(executor.device())?;
        let weights = FlashNextWeights::load(&executor, &ckpt, guard)?;
        let c = &ckpt.config;
        LoadStage {
            resident: session_state_bytes(c, capacity),
            transient: 0,
        }
        .check(guard)?;
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
            dense_max,
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
                inject: zeros(&executor, c.hc_count * super::hc::INJECT_CHUNKS),
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
            capacity,
            initial_linear: Vec::new(),
            #[cfg(feature = "devtools")]
            profiler: if super::profile::profile_requested() {
                Some(super::profile::StepProfiler::new(executor.device())?)
            } else {
                None
            },
            poisoned: false,
            #[cfg(test)]
            routing_capture: executor.device().new_buffer(
                (c.layer_kinds.len() * c.top_k * std::mem::size_of::<u32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            #[cfg(test)]
            fail_next_ngram_read: false,
            #[cfg(test)]
            fail_reset_after_layer: None,
            #[cfg(test)]
            fail_encode_after_layer: None,
            #[cfg(test)]
            forced_routing: Vec::new(),
            executor,
            ckpt,
        };
        session.states = session.fresh_states(capacity)?;
        session.initial_linear = session
            .states
            .iter()
            .map(|state| match state {
                LayerState::Linear(state) => {
                    session.executor.snapshot_linear_attn_state(state).map(Some)
                }
                LayerState::Full(_) => Ok(None),
            })
            .collect::<Result<_>>()?;
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
                    // KV épinglé en f32 : ni `SARAGOSSA_RUST_KV_BF16` ni le KV
                    // quantifié ne s'appliquent (l'oracle N2 compare à 1e-5).
                    let kv = self.arena.full_attention_with_format(
                        capacity,
                        c.heads,
                        c.kv_heads,
                        c.head_dim,
                        crate::decode_resident::KvFormat::F32,
                        None,
                    )?;
                    Ok(LayerState::Full(Box::new(kv)))
                }
            })
            .collect()
    }

    /// Octets alloués par la session hors poids (oracles) : buffers propres,
    /// états DeltaNet et instantanés, KV f32.
    #[cfg(test)]
    pub(crate) fn owned_state_bytes(&self) -> u64 {
        let c = &self.ckpt.config;
        let own = [
            &self.h[0],
            &self.h[1],
            &self.x,
            &self.branch,
            &self.logits,
            &self.index,
            &self.capture,
            &self.hc_scratch.normed,
            &self.hc_scratch.low,
            &self.hc_scratch.up,
            &self.hc_scratch.inject,
            &self.ple_scratch.emb,
            &self.ple_scratch.key,
            &self.ple_scratch.key_normed,
            &self.ple_scratch.value,
            &self.ple_scratch.query,
            &self.ple_scratch.gated,
            &self.ple_scratch.normed,
            &self.qsa_scratch.q_gate,
            &self.qsa_scratch.q_raw,
            &self.qsa_scratch.gate,
            &self.qsa_scratch.q_roped,
            &self.qsa_scratch.k_raw,
            &self.qsa_scratch.k_roped,
            &self.qsa_scratch.v,
            &self.qsa_scratch.scores,
            &self.qsa_scratch.ctx,
            &self.qsa_scratch.gated,
            &self.ple_state.bytes,
            &self.ple_state.ring,
        ]
        .iter()
        .map(|b| b.length())
        .sum::<u64>();
        let states: u64 = self
            .states
            .iter()
            .map(|state| match state {
                LayerState::Linear(state) => MetalExecutor::linear_attn_state_bytes(state),
                LayerState::Full(kv) => {
                    (kv.capacity() * 2 * c.kv_heads * c.head_dim * std::mem::size_of::<f32>())
                        as u64
                }
            })
            .sum();
        let snapshots: u64 = self
            .initial_linear
            .iter()
            .flatten()
            .map(MetalExecutor::linear_attn_state_bytes)
            .sum();
        own + states + snapshots
    }

    /// Octets alloués sur le device Metal de la session (oracles).
    #[cfg(test)]
    pub(crate) fn device_allocated_bytes(&self) -> u64 {
        self.executor.device().current_allocated_size()
    }

    /// Profil GPU accumulé, s'il est actif.
    #[cfg(feature = "devtools")]
    pub(crate) fn profiler_mut(&mut self) -> Option<&mut super::profile::StepProfiler> {
        self.profiler.as_mut()
    }

    /// Position du prochain token (oracles).
    #[cfg(test)]
    pub(crate) fn position(&self) -> usize {
        self.position
    }

    /// Remet la session au début d'une séquence : états DeltaNet initiaux, KV
    /// vide, PLE remis à zéro. La session est invalidée pendant la
    /// restauration et ne redevient utilisable qu'après un succès complet, ce
    /// qui lève aussi l'invalidation d'une erreur GPU antérieure.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur de copie Metal ou de troncature du KV.
    pub(crate) fn reset(&mut self) -> Result<()> {
        // Invalide pendant la restauration : un échec partiel laisse des états
        // mêlés, que seul un `reset` complet réhabilite.
        self.poisoned = true;
        for (index, (state, initial)) in
            self.states.iter_mut().zip(&self.initial_linear).enumerate()
        {
            #[cfg(not(test))]
            let _ = index;
            #[cfg(test)]
            if self
                .fail_reset_after_layer
                .take_if(|after| *after < index)
                .is_some()
            {
                return Err(InferError::Metal(format!(
                    "reset en échec à la couche {index} (injecté)"
                )));
            }
            match (state, initial) {
                (LayerState::Linear(state), Some(initial)) => {
                    self.executor.restore_linear_attn_state(state, initial)?;
                }
                (LayerState::Full(kv), None) => kv.truncate(0)?,
                _ => {
                    return Err(InferError::Config(
                        "reset Flash-Next : états et instantanés désalignés".to_string(),
                    ))
                }
            }
        }
        let c = &self.ckpt.config;
        self.ple_state.reset(c.ngram_size, c.eos);
        self.position = 0;
        self.poisoned = false;
        Ok(())
    }

    /// Décode `token` à la position courante. `logits` et `layers` demandent la
    /// relecture des logits et des résiduels par couche (oracles).
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si la capacité est atteinte, sur lecture n-gram ou
    /// sur encodage et exécution Metal.
    pub(crate) fn step(&mut self, token: usize, logits: bool, layers: bool) -> Result<StepOutput> {
        if self.poisoned {
            return Err(InferError::Metal(
                "Flash-Next résident : session invalidée (erreur GPU ou reset incomplet)"
                    .to_string(),
            ));
        }
        let c = &self.ckpt.config;
        if self.position >= self.capacity {
            return Err(InferError::Config(format!(
                "Flash-Next résident : capacité de {} tokens atteinte",
                self.capacity
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
        // PLE : hachage sur un clone du contexte (engagé seulement si le pas
        // aboutit) et lectures n-gram terminées avant le commit.
        let p = &self.ckpt.ngram_params;
        let mut context = self.ple_state.context.clone();
        let ids = context.step(
            token as u64,
            c.heads_per_ngram,
            &p.multipliers,
            &p.vocab_sizes,
            &p.offsets,
        );
        let mut bytes = vec![0_u8; c.ple_dim];
        #[cfg(test)]
        if std::mem::take(&mut self.fail_next_ngram_read) {
            return Err(InferError::Config(
                "lecture n-gram en échec (injectée)".to_string(),
            ));
        }
        self.ckpt.ngram_rows_raw(&ids, &mut bytes)?;
        self.ple_state.stage_bytes(&bytes)?;
        let host = host_started.elapsed();

        // Métadonnées avancées pendant l'encodage, restaurées si celui-ci échoue :
        // rien n'est soumis, le GPU n'a donc rien modifié.
        let kv_lens: Vec<Option<usize>> = self
            .states
            .iter()
            .map(|state| match state {
                LayerState::Full(kv) => Some(kv.len()),
                LayerState::Linear(_) => None,
            })
            .collect();
        let ple_step = self.ple_state.step;
        let queue = self.queue.clone();
        let command_buffer = queue.new_command_buffer();
        #[cfg(feature = "devtools")]
        let mut profiler = self.profiler.take();
        #[cfg(feature = "devtools")]
        let mut encoders = match profiler.as_mut() {
            Some(profiler) => {
                profiler.begin_step();
                Encoders::profiled(command_buffer, profiler)
            }
            None => Encoders::single(command_buffer),
        };
        #[cfg(not(feature = "devtools"))]
        let mut encoders = Encoders::single(command_buffer);
        let encoded = self.encode_token(&mut encoders, logits, layers);
        if let Err(error) = encoded {
            drop(encoders);
            #[cfg(feature = "devtools")]
            {
                if let Some(profiler) = profiler.as_mut() {
                    profiler.abort_step();
                }
                self.profiler = profiler;
            }
            self.ple_state.step = ple_step;
            for (state, len) in self.states.iter_mut().zip(kv_lens) {
                if let (LayerState::Full(kv), Some(len)) = (state, len) {
                    if kv.truncate(len).is_err() {
                        self.poisoned = true;
                    }
                }
            }
            return Err(error);
        }
        encoders.end();
        #[cfg(feature = "devtools")]
        if let Some(profiler) = profiler.as_ref() {
            profiler.encode_resolve(command_buffer);
        }
        if let Err(error) = commit_and_wait(command_buffer) {
            // Exécution GPU partielle possible : états récurrents et KV incertains.
            self.poisoned = true;
            return Err(error);
        }
        #[cfg(feature = "devtools")]
        {
            if let Some(profiler) = profiler.as_mut() {
                profiler.end_step()?;
            }
            self.profiler = profiler;
        }
        self.ple_state.context = context;
        self.position += 1;
        let width = self.hc_dims.width();
        let (vocab, layer_count) = (self.ckpt.config.vocab, self.ckpt.config.layer_kinds.len());
        let argmax = read_u32_buffer(&self.index, 1)?
            .first()
            .copied()
            .ok_or_else(|| InferError::Metal("argmax résident absent".to_string()))?;
        Ok(StepOutput {
            argmax,
            logits: logits
                .then(|| read_f32_buffer(&self.logits, vocab))
                .transpose()?,
            layers: layers
                .then(|| {
                    read_f32_buffer(&self.capture, layer_count * width)
                        .map(|all| all.chunks_exact(width).map(<[f32]>::to_vec).collect())
                })
                .transpose()?,
            #[cfg(test)]
            routing: layers
                .then(|| {
                    read_u32_buffer(&self.routing_capture, layer_count * self.ckpt.config.top_k)
                        .map(|all| {
                            all.chunks_exact(self.ckpt.config.top_k)
                                .map(<[u32]>::to_vec)
                                .collect()
                        })
                })
                .transpose()?,
            host,
        })
    }

    /// Encode les couches, le mixer final et le lm_head du token courant.
    fn encode_token(
        &mut self,
        encoders: &mut Encoders<'_>,
        logits: bool,
        layers: bool,
    ) -> Result<()> {
        let c = &self.ckpt.config;
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
                let encoder = encoders.at(Brick::Ple)?;
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
            let encoder = encoders.at(Brick::Hc)?;
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
                    let encoder = encoders.at(Brick::DeltaNet)?;
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
                    let encoder = encoders.at(Brick::Qsa)?;
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
            let encoder = encoders.at(Brick::Hc)?;
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
            let encoder = encoders.at(Brick::Moe)?;
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
            #[cfg(test)]
            if layers {
                let indices = crate::metal_backend::take_last_topk_indices().ok_or_else(|| {
                    InferError::Metal(format!("couche {index} : routage GPU non capturé"))
                })?;
                encoder.set_compute_pipeline_state(&self.kernels.copy_u32);
                encoder.set_buffer(0, Some(&indices), 0);
                encoder.set_buffer(1, Some(&self.routing_capture), 0);
                super::set_value(encoder, 2, &[c.top_k as u32, (index * c.top_k) as u32]);
                super::dispatch_1d(encoder, c.top_k);
                crate::metal_backend::post_dispatch_barrier(encoder);
            }
            let encoder = encoders.at(Brick::Hc)?;
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
            #[cfg(test)]
            if self
                .fail_encode_after_layer
                .take_if(|after| *after == index)
                .is_some()
            {
                return Err(InferError::Metal(format!(
                    "encodage en échec après la couche {index} (injecté)"
                )));
            }
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
        let encoder = encoders.at(Brick::Mixer)?;
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
        let encoder = encoders.at(Brick::LmHead)?;
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
        Ok(())
    }
}
