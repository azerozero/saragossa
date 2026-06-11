//! Décodeur causal, cache KV et orchestration CPU/GPU du modèle.

#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::decode_resident::{
    DecodeResidentState, FullAttentionMetalState, FullAttnDenseLayerWeights, FullAttnLayerDims,
    FullAttnLayerWeights, FullAttnRoutedLayerWeights, GpuElement, GpuSectionTimer, GpuTensor,
    LinearAttnDenseLayerWeights, LinearAttnLayerWeights, ScratchLease,
};
use crate::linear_attention::{
    LinearAttention, LinearAttentionCache, LinearAttentionConfig, LinearAttentionWeights,
};
#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::metal_backend::{
    LinearAttentionMetalState, LinearAttentionStepSpec, LinearAttnResidentDims,
    MetalEmbeddingWeightBuffers, MetalLinearAttnResidentDenseWeights,
    MetalLinearAttnResidentWeights, MetalLinearWeightBuffers, MetalMoeRoutedWeights,
    MetalMoeSharedWeights,
};
use crate::{
    embed_weight_tokens, load_f32_tensors, rms_norm, sample_token_top_k_top_p, softmax,
    DeterministicSampler, EmbeddingWeight, FeedForward, ForwardRuntime, GatedMlp, InferError,
    Linear, LinearWeight, ModelConfig, Result, Tensor,
};
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

mod attention;
mod attention_cache;
mod attention_ops;
pub(crate) mod flags;
mod generation;
mod loading;
mod mtp;
mod resident;

use self::flags::*;
pub(in crate::decoder) use self::loading::*;

#[cfg(test)]
mod tests;

#[derive(Clone, Debug)]
/// Décrit les dimensions et options d'un décodeur causal.
pub struct CausalDecoderConfig {
    /// Définit l'epsilon des normalisations RMS.
    pub rms_eps: f32,
    /// Définit la base RoPE optionnelle.
    pub rope_theta: Option<f32>,
    /// Définit le nombre de couches du décodeur.
    pub num_hidden_layers: usize,
    /// Définit le nombre de têtes d'attention requête.
    pub num_attention_heads: usize,
    /// Définit le nombre de têtes clé/valeur.
    pub num_key_value_heads: usize,
    /// Surcharge la dimension de tête.
    pub head_dim: Option<usize>,
    /// Surcharge la dimension RoPE.
    pub rope_dims: Option<usize>,
    /// Active la porte de sortie attention.
    pub attn_output_gate: bool,
    /// Définit l'intervalle des couches full-attention.
    pub full_attention_interval: Option<usize>,
    /// Définit le nombre de têtes valeur linéaires.
    pub linear_num_value_heads: Option<usize>,
    /// Définit le nombre de têtes clé linéaires.
    pub linear_num_key_heads: Option<usize>,
    /// Définit la dimension des têtes clé linéaires.
    pub linear_key_head_dim: Option<usize>,
    /// Définit la dimension des têtes valeur linéaires.
    pub linear_value_head_dim: Option<usize>,
    /// Définit la largeur de convolution linéaire.
    pub linear_conv_kernel_dim: Option<usize>,
    /// Définit le nombre d'experts MoE.
    pub num_experts: Option<usize>,
    /// Définit le nombre d'experts sélectionnés par token.
    pub num_experts_per_tok: usize,
    /// Définit la dimension intermédiaire des experts.
    pub moe_intermediate_size: usize,
    /// Définit la dimension intermédiaire de l'expert partagé.
    pub shared_expert_intermediate_size: usize,
}

#[derive(Clone, Debug)]
/// Paramètre une génération autoregressive.
pub struct GenerationOptions {
    /// Liste les tokens qui arrêtent la génération.
    pub stop_token_ids: Vec<usize>,
    /// Définit la température de sampling.
    pub temperature: f32,
    /// Définit le seuil nucleus sampling.
    pub top_p: f32,
    /// Définit le top-k sampling (`0` = désactivé).
    pub top_k: usize,
    /// Définit la graine du sampler déterministe.
    pub seed: u64,
}

impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            stop_token_ids: Vec::new(),
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            seed: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
/// Mesure les durées de préfill et decode.
pub struct GenerationTimings {
    /// Mesure le temps passé en préfill.
    pub prefill: Duration,
    /// Mesure le temps passé en decode.
    pub decode: Duration,
    /// Compte les tokens produits en decode.
    pub decode_tokens: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// Contient les tokens générés et leurs timings.
pub struct GenerationOutput {
    /// Liste les tokens générés.
    pub tokens: Vec<usize>,
    /// Mesure les timings de génération.
    pub timings: GenerationTimings,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn new_resident_compute_encoder(
    command_buffer: &metal::CommandBufferRef,
) -> &metal::ComputeCommandEncoderRef {
    if crate::metal_backend::resident_concurrent_enabled() {
        command_buffer
            .compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Concurrent)
    } else {
        command_buffer.new_compute_command_encoder()
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
/// Compte les décisions d'une génération spéculative.
pub struct SpeculativeStats {
    /// Nombre de tokens proposés par le draft.
    pub proposed: usize,
    /// Nombre de tokens draft acceptés par l'argmax trunk.
    pub accepted: usize,
    /// Nombre de propositions rejetées.
    pub rejected: usize,
    /// Nombre de forwards trunk utilisés pour vérifier les propositions.
    pub verifications: usize,
    /// Nombre de restaurations du cache après rejet.
    pub rollbacks: usize,
    /// Nombre de propositions vérifiées par position draft.
    pub proposed_by_position: Vec<usize>,
    /// Nombre de propositions acceptées par position draft.
    pub accepted_by_position: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// Contient la sortie et les statistiques spéculatives.
pub struct SpeculativeOutput {
    /// Tokens générés par la boucle spéculative.
    pub tokens: Vec<usize>,
    /// Compteurs de vérification et d'acceptance.
    pub stats: SpeculativeStats,
}

#[derive(Clone, Debug)]
struct MtpHead {
    pre_fc_norm_embedding: Tensor,
    pre_fc_norm_hidden: Tensor,
    fc: Linear,
    layer: MtpLayer,
    norm: Tensor,
}

#[derive(Clone, Debug)]
struct MtpLayer {
    input_norm: Tensor,
    attention: FullAttention,
    post_attention_norm: Tensor,
    // FeedForward (Dense GatedMlp OU MoE) : le 27B-OptiQ a une tête MTP à MLP
    // DENSE quantifié 4-bit ; les variantes A3B ont une MLP MoE.
    mlp: FeedForward,
}

/// Cache K/V d'un décodeur causal.
#[derive(Debug)]
pub struct CausalDecoderCache {
    layers: Vec<LayerKvCache>,
    position: usize,
    // Arène du decode résident COMPLET (1c) : ping-pong hidden + buffer u32 du
    // token + pool/queue/pipelines. Liée à la session → jamais clonée (cf. infra).
    #[cfg(all(target_os = "macos", feature = "metal"))]
    resident: Option<ResidentArena>,
}

// Clone MANUEL (comme LayerKvCache / LinearAttentionCache) : l'arène résidente
// détient des buffers GPU liés à la session ; le prefix-cache CLONE le cache, donc
// l'arène est remise à `None` au clone (sinon double-ownership des MTLBuffer).
impl Clone for CausalDecoderCache {
    fn clone(&self) -> Self {
        Self {
            layers: self.layers.clone(),
            position: self.position,
            #[cfg(all(target_os = "macos", feature = "metal"))]
            resident: None,
        }
    }
}

/// Arène persistante du decode résident complet (1c), stockée dans le cache pour
/// vivre sur toute la session de génération : l'état `DecodeResidentState` (pool
/// scratch à bail, queue, pipelines gate/RoPE), les deux buffers hidden ping-pong
/// et le buffer `u32` recevant l'argmax du token. Non clonable (réserve Codex D).
#[cfg(all(target_os = "macos", feature = "metal"))]
#[derive(Debug)]
struct ResidentArena {
    state: DecodeResidentState,
    hidden_a: GpuTensor,
    hidden_b: GpuTensor,
    index: GpuTensor,
    index_ring: Vec<GpuTensor>,
    layers: Vec<ResidentLayerBuffers>,
    dense_tail_score: metal::Buffer,
    embed_tokens: MetalEmbeddingWeightBuffers,
    final_norm: metal::Buffer,
    lm_head: MetalLinearWeightBuffers,
    mtp: Option<ResidentMtpArena>,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[derive(Debug)]
struct ResidentMtpArena {
    pre_fc_norm_embedding: metal::Buffer,
    pre_fc_norm_hidden: metal::Buffer,
    fc: MetalLinearWeightBuffers,
    layer: ResidentFullDenseBuffers,
    norm: metal::Buffer,
    kv: FullAttentionMetalState,
    hidden_a: GpuTensor,
    hidden_b: GpuTensor,
    current_is_a: bool,
    index: GpuTensor,
    draft_indices: GpuTensor,
    embedding: GpuTensor,
    concat: GpuTensor,
    fc_out: GpuTensor,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[derive(Debug)]
struct ResidentVerifyCaptures {
    base_position: usize,
    linear: Vec<Option<Vec<LinearAttentionMetalState>>>,
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
#[derive(Debug)]
struct ResidentVerifyCaptures;

#[cfg(all(target_os = "macos", feature = "metal"))]
#[derive(Debug)]
struct ResidentVerifyOutput {
    states: Tensor,
    tokens: Option<Vec<usize>>,
    captures: Option<ResidentVerifyCaptures>,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[derive(Debug)]
enum ResidentLayerBuffers {
    FullMoe(ResidentFullMoeBuffers),
    FullRouted(ResidentFullRoutedBuffers),
    FullDense(ResidentFullDenseBuffers),
    LinearMoe(ResidentLinearMoeBuffers),
    LinearDense(ResidentLinearDenseBuffers),
    Other,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[derive(Debug)]
struct ResidentFullMoeBuffers {
    input_norm: metal::Buffer,
    qkv_proj: MetalLinearWeightBuffers,
    o_proj: MetalLinearWeightBuffers,
    q_norm: metal::Buffer,
    k_norm: metal::Buffer,
    post_norm: metal::Buffer,
    moe: MetalMoeSharedWeights,
    top_k: usize,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[derive(Debug)]
struct ResidentFullRoutedBuffers {
    input_norm: metal::Buffer,
    qkv_proj: MetalLinearWeightBuffers,
    o_proj: MetalLinearWeightBuffers,
    q_norm: metal::Buffer,
    k_norm: metal::Buffer,
    post_norm: metal::Buffer,
    moe: MetalMoeRoutedWeights,
    top_k: usize,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[derive(Debug)]
struct ResidentLinearMoeBuffers {
    input_norm: metal::Buffer,
    linear: MetalLinearAttnResidentWeights,
    post_norm: metal::Buffer,
    moe: MetalMoeSharedWeights,
    top_k: usize,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[derive(Debug)]
struct ResidentFullDenseBuffers {
    input_norm: metal::Buffer,
    qkv_proj: Option<MetalLinearWeightBuffers>,
    q_proj: MetalLinearWeightBuffers,
    k_proj: MetalLinearWeightBuffers,
    v_proj: MetalLinearWeightBuffers,
    o_proj: MetalLinearWeightBuffers,
    q_norm: metal::Buffer,
    k_norm: metal::Buffer,
    post_norm: metal::Buffer,
    gate_proj: MetalLinearWeightBuffers,
    up_proj: MetalLinearWeightBuffers,
    down_proj: MetalLinearWeightBuffers,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[derive(Debug)]
struct ResidentLinearDenseBuffers {
    input_norm: metal::Buffer,
    linear: MetalLinearAttnResidentDenseWeights,
    post_norm: metal::Buffer,
    gate_proj: MetalLinearWeightBuffers,
    up_proj: MetalLinearWeightBuffers,
    down_proj: MetalLinearWeightBuffers,
}

#[derive(Debug, Default)]
struct LayerKvCache {
    keys: Vec<f32>,
    values: Vec<f32>,
    kv_dim: Option<usize>,
    linear: LinearAttentionCache,
    // KV full-attn résident GPU (decode résident, flag RETI_RUST_DECODE_RESIDENT).
    #[cfg(all(target_os = "macos", feature = "metal"))]
    full: Option<FullAttentionMetalState>,
}

// Clone MANUEL (réserve Codex R2) : l'état Metal résident (`full`) est lié à une
// session → remis à `None` au clone, comme `LinearAttentionCache::clone` met son
// `metal` à `None`. Sinon le prefix-cache (qui CLONE le cache, decoder.rs:565/594)
// dupliquerait la poignée du MTLBuffer → double-ownership / aliasing GPU.
impl Clone for LayerKvCache {
    fn clone(&self) -> Self {
        Self {
            keys: self.keys.clone(),
            values: self.values.clone(),
            kv_dim: self.kv_dim,
            linear: self.linear.clone(),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            full: None,
        }
    }
}

impl CausalDecoderCache {
    /// Renvoie la position absolue du prochain token.
    #[must_use]
    pub fn position(&self) -> usize {
        self.position
    }

    /// Renvoie le nombre de caches de couche.
    #[must_use]
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Renvoie `true` si aucun token n'a été pré-rempli.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.position == 0
    }
}

impl Default for CausalDecoderConfig {
    fn default() -> Self {
        Self {
            rms_eps: 1.0e-6,
            rope_theta: Some(10_000.0),
            num_hidden_layers: 1,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: None,
            rope_dims: None,
            attn_output_gate: false,
            full_attention_interval: None,
            linear_num_value_heads: None,
            linear_num_key_heads: None,
            linear_key_head_dim: None,
            linear_value_head_dim: None,
            linear_conv_kernel_dim: None,
            num_experts: None,
            num_experts_per_tok: 1,
            moe_intermediate_size: 0,
            shared_expert_intermediate_size: 0,
        }
    }
}

impl From<&ModelConfig> for CausalDecoderConfig {
    fn from(config: &ModelConfig) -> Self {
        Self {
            rms_eps: config.rms_norm_eps,
            rope_theta: Some(config.rope_theta),
            num_hidden_layers: config.num_hidden_layers,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            head_dim: Some(config.head_dim()),
            rope_dims: Some(config.rope_dims()),
            attn_output_gate: config.attn_output_gate.unwrap_or(false),
            full_attention_interval: config.full_attention_interval,
            linear_num_value_heads: config.linear_num_value_heads,
            linear_num_key_heads: config.linear_num_key_heads,
            linear_key_head_dim: config.linear_key_head_dim,
            linear_value_head_dim: config.linear_value_head_dim,
            linear_conv_kernel_dim: config.linear_conv_kernel_dim,
            num_experts: config.num_experts,
            num_experts_per_tok: config.num_experts_per_tok.unwrap_or(1),
            moe_intermediate_size: config.moe_intermediate_size.unwrap_or(0),
            shared_expert_intermediate_size: config.shared_expert_intermediate_size.unwrap_or(0),
        }
    }
}

impl CausalDecoderConfig {
    fn is_full_attention_layer(&self, layer_index: usize) -> bool {
        match self.full_attention_interval {
            Some(interval) if interval > 0 => (layer_index + 1) % interval == 0,
            _ => true,
        }
    }

    fn linear_attention_config(&self) -> Result<LinearAttentionConfig> {
        Ok(LinearAttentionConfig {
            num_key_heads: self
                .linear_num_key_heads
                .ok_or_else(|| InferError::Config("linear_num_key_heads manquant".to_string()))?,
            num_value_heads: self
                .linear_num_value_heads
                .ok_or_else(|| InferError::Config("linear_num_value_heads manquant".to_string()))?,
            key_head_dim: self
                .linear_key_head_dim
                .ok_or_else(|| InferError::Config("linear_key_head_dim manquant".to_string()))?,
            value_head_dim: self
                .linear_value_head_dim
                .ok_or_else(|| InferError::Config("linear_value_head_dim manquant".to_string()))?,
            conv_kernel_dim: self
                .linear_conv_kernel_dim
                .ok_or_else(|| InferError::Config("linear_conv_kernel_dim manquant".to_string()))?,
            rms_eps: self.rms_eps,
        })
    }
}

#[derive(Clone, Debug)]
/// Exécute un modèle causal Qwen en CPU ou Metal.
pub struct CausalDecoder {
    config: CausalDecoderConfig,
    embed_tokens: EmbeddingWeight,
    layers: Vec<DecoderLayer>,
    final_norm: Tensor,
    lm_head: Linear,
    mtp: Option<MtpHead>,
    prefix_cache: Arc<Mutex<PrefixCache>>,
    #[cfg(all(target_os = "macos", feature = "metal"))]
    runtime: DecoderRuntime,
}

#[derive(Clone, Debug, Default)]
struct PrefixCache {
    entries: Vec<PrefixCacheEntry>,
}

#[derive(Clone, Debug)]
struct PrefixCacheEntry {
    tokens: Vec<usize>,
    cache: CausalDecoderCache,
    final_state: Tensor,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[derive(Clone, Debug, Default)]
struct DecoderRuntime {
    metal: Option<Arc<crate::MetalExecutor>>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum DecoderTensor {
    Dense(Tensor),
    LinearWeight(LinearWeight),
    ExpertLinearWeights {
        shape: Vec<usize>,
        weights: Vec<LinearWeight>,
    },
}

impl DecoderTensor {
    pub(crate) fn shape(&self) -> &[usize] {
        match self {
            Self::Dense(tensor) => tensor.shape(),
            Self::LinearWeight(weight) => weight.shape(),
            Self::ExpertLinearWeights { shape, .. } => shape,
        }
    }
}

impl From<Tensor> for DecoderTensor {
    fn from(value: Tensor) -> Self {
        Self::Dense(value)
    }
}

#[derive(Clone, Debug)]
struct DecoderLayer {
    input_norm: Tensor,
    attention: AttentionBlock,
    post_attention_norm: Option<Tensor>,
    mlp: Option<FeedForward>,
}

#[derive(Clone, Debug)]
enum AttentionBlock {
    Full(Box<FullAttention>),
    Linear(Box<LinearAttention>),
}

#[derive(Clone, Debug)]
struct FullAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: Option<Tensor>,
    k_norm: Option<Tensor>,
}

impl CausalDecoder {
    /// Charge un décodeur depuis un fichier safetensors.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le chargement ou la construction échoue.
    pub fn from_safetensors(path: impl AsRef<Path>, config: CausalDecoderConfig) -> Result<Self> {
        Self::from_tensors(load_f32_tensors(path)?, config)
    }

    /// Construit un décodeur depuis des tenseurs denses.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si des poids requis manquent ou sont invalides.
    pub fn from_tensors(
        tensors: HashMap<String, Tensor>,
        config: CausalDecoderConfig,
    ) -> Result<Self> {
        let tensors = tensors
            .into_iter()
            .map(|(name, tensor)| (name, DecoderTensor::Dense(tensor)))
            .collect();
        Self::from_decoder_tensors(tensors, config)
    }

    pub(crate) fn from_decoder_tensors(
        mut tensors: HashMap<String, DecoderTensor>,
        config: CausalDecoderConfig,
    ) -> Result<Self> {
        let embed_tokens = take_embedding_weight(&mut tensors, "embed_tokens.weight")?;
        if config.num_hidden_layers == 0 {
            return Err(InferError::Config(
                "décodeur sans couche cachée".to_string(),
            ));
        }
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_index in 0..config.num_hidden_layers {
            layers.push(DecoderLayer::from_tensors(
                &mut tensors,
                layer_index,
                &config,
            )?);
        }
        let final_norm = take_dense(&mut tensors, "norm.weight")?;
        let lm_head = linear_from(&mut tensors, "lm_head")?;
        Ok(Self {
            config,
            embed_tokens,
            layers,
            final_norm,
            lm_head,
            mtp: None,
            prefix_cache: Arc::new(Mutex::new(PrefixCache::default())),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            runtime: DecoderRuntime::default(),
        })
    }

    /// Calcule les logits du prochain token sans cache.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le forward échoue.
    pub fn next_logits(&self, token_ids: &[usize]) -> Result<Tensor> {
        let runtime = self.forward_runtime();
        let mut hidden = embed_weight_tokens(&self.embed_tokens, token_ids)?;
        for layer in &self.layers {
            hidden = layer.forward(&self.config, &hidden, runtime)?;
        }
        let final_state = rms_norm(&hidden, &self.final_norm, self.config.rms_eps)?;
        let logits = self.lm_head.forward_with_runtime(&final_state, runtime)?;
        Tensor::row(logits.last_row()?.to_vec())
    }

    /// Active le runtime Metal pour les kernels de couche disponibles.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si Metal est indisponible ou si un kernel ne compile pas.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn with_metal_runtime(mut self) -> Result<Self> {
        self.runtime.metal = Some(Arc::new(crate::MetalExecutor::new()?));
        Ok(self)
    }

    /// Active le runtime Metal avec un executor déjà initialisé.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[must_use]
    pub fn with_metal_executor(mut self, executor: crate::MetalExecutor) -> Self {
        self.runtime.metal = Some(Arc::new(executor));
        self
    }

    /// Charge une tête MTP depuis un sidecar safetensors.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le sidecar est incomplet ou incompatible.
    pub fn with_mtp_sidecar(mut self, path: impl AsRef<Path>) -> Result<Self> {
        self.mtp = Some(MtpHead::from_sidecar(path, &self.config)?);
        Ok(self)
    }

    fn forward_runtime(&self) -> ForwardRuntime<'_> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if let Some(metal) = self.runtime.metal.as_deref() {
            return ForwardRuntime::metal(metal);
        }
        ForwardRuntime::cpu()
    }

    /// Crée un cache K/V vide pour ce décodeur.
    #[must_use]
    pub fn empty_cache(&self) -> CausalDecoderCache {
        CausalDecoderCache {
            layers: vec![LayerKvCache::default(); self.layers.len()],
            position: 0,
            #[cfg(all(target_os = "macos", feature = "metal"))]
            resident: None,
        }
    }

    /// Avance le décodeur d'un token en alimentant le cache K/V.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le forward avec cache échoue.
    pub fn next_logits_cached(
        &self,
        cache: &mut CausalDecoderCache,
        token_id: usize,
    ) -> Result<Tensor> {
        let final_state = self.next_final_state_cached(cache, token_id)?;
        self.logits_from_final_state(&final_state)
    }
}
