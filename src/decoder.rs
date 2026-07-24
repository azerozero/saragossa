//! Décodeur causal, cache KV et orchestration CPU/GPU du modèle.

#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::decode_resident::{
    install_pipeline_scratch_slot, DecodeResidentState, FullAttentionMetalState,
    FullAttnDenseLayerWeights, FullAttnLayerDims, FullAttnLayerWeights, FullAttnRoutedLayerWeights,
    GemmaParallelMoeTailWeights, GpuElement, GpuSectionTimer, GpuTensor,
    LinearAttnDenseLayerWeights, LinearAttnLayerWeights, ScratchLease,
};
use crate::linear_attention::{
    LinearAttention, LinearAttentionCache, LinearAttentionConfig, LinearAttentionWeights,
};
#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::metal_backend::{
    LinearAttentionMetalState, LinearAttentionStepSpec, LinearAttnResidentDims,
    MetalEmbeddingWeightBuffers, MetalLinearAttnResidentDenseWeights, MetalLinearWeightBuffers,
    MetalMoeRoutedWeights, MetalMoeSharedWeights,
};
use crate::runtime_flags::qwen_embed_bf16_enabled;
use crate::{
    embed_weight_tokens, load_f32_tensors, process_purge_registry, rms_norm,
    sample_token_top_k_top_p, softmax, DeterministicSampler, EmbeddingWeight, FeedForward,
    ForwardRuntime, GatedMlp, InferError, Linear, LinearWeight, MemoryGuard, ModelConfig,
    PurgeOutcome, Result, Tensor,
};
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

mod attention;
mod attention_cache;
mod attention_ops;
#[cfg(test)]
pub(crate) use attention_ops::{
    gemma_global_attention_prefill_oracle, GemmaGlobalAttentionOracleSpec,
};
mod duo;
pub(crate) mod flags;
mod generation;
mod lightbatch;
mod loading;
mod mtp;
mod mtp_adaptive;
mod resident;

use self::flags::*;
pub(in crate::decoder) use self::loading::*;

/// Force le decode résident complet sur les couches linéaires du processus.
pub fn force_resident_full_linear_decode() {
    flags::force_resident_full_linear_decode();
}

#[cfg(all(test, target_os = "macos", feature = "metal", feature = "devtools"))]
mod qwen35_oracles;
#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
/// Sélectionne l'appariement des dimensions tournées par le RoPE.
///
/// NOTE: La convention du moteur est `Halves` (« rotate-half », le
/// `traditional=false` de mlx) : c'est celle des checkpoints Qwen, Llama,
/// Mistral et Gemma — le moteur appliquait historiquement `Interleaved`,
/// divergeant de mlx_lm sur tous les modèles. `Interleaved` est conservé
/// comme référence de test et levier de diagnostic ; les kernels Metal
/// résidents n'implémentent QUE `Halves` (gates en amont).
pub enum RopeStyle {
    /// Tourne les moitiés `(i, i+d/2)` (« rotate-half », convention moteur).
    #[default]
    Halves,
    /// Tourne les paires adjacentes `(2i, 2i+1)` (legacy, tests/diagnostic).
    Interleaved,
}

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
    /// Surcharge les têtes K/V des couches globales Gemma 4.
    pub num_global_key_value_heads: Option<usize>,
    /// Surcharge la dimension de tête.
    pub head_dim: Option<usize>,
    /// Surcharge la dimension de tête des couches globales Gemma 4.
    pub global_head_dim: Option<usize>,
    /// Surcharge la dimension RoPE.
    pub rope_dims: Option<usize>,
    /// Surcharge la dimension RoPE des couches globales Gemma 4.
    pub rope_full_dims: Option<usize>,
    /// Surcharge la dimension RoPE des couches locales Gemma 4.
    pub rope_sliding_dims: Option<usize>,
    /// Active la porte de sortie attention.
    pub attn_output_gate: bool,
    /// Liste le type d'attention de chaque couche Gemma 4.
    pub layer_types: Vec<String>,
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
    /// Met à l'échelle les embeddings d'entrée (`√hidden` Gemma ; `None` ailleurs).
    pub embed_scale: Option<f32>,
    /// Définit la base RoPE des couches locales Gemma 3 (sliding-window).
    pub rope_local_base_freq: Option<f32>,
    /// Définit la base RoPE des couches globales Gemma 4.
    pub rope_full_base_freq: Option<f32>,
    /// Met à l'échelle les positions RoPE (`1/factor` du rope_scaling linear).
    pub rope_position_scale: Option<f32>,
    /// Définit la taille de fenêtre des couches locales Gemma 3.
    pub sliding_window: Option<usize>,
    /// Définit la période de la couche globale Gemma 3.
    pub sliding_window_pattern: Option<usize>,
    /// Indique si les couches globales Gemma 4 partagent K et V.
    pub attention_k_eq_v: bool,
    /// Active la RMSNorm sans poids sur V (Gemma 4).
    pub attention_value_norm: bool,
    /// Active le bloc MoE parallèle Gemma 4.
    pub parallel_moe: bool,
    /// Déclare le softcapping final des logits Gemma.
    pub final_logit_softcapping: Option<f32>,
    /// Surcharge le facteur d'échelle des scores d'attention (Gemma).
    pub query_pre_attn_scalar: Option<f32>,
    /// Sélectionne l'activation MLP (GeGLU pour Gemma, SwiGLU sinon).
    pub activation: crate::Activation,
    /// Sélectionne l'appariement RoPE (rotate-half pour Gemma).
    pub rope_style: RopeStyle,
    /// Indique si le modèle est réellement un Gemma 4. Les types de couches
    /// `full_attention`/`sliding_attention` sont partagés avec d'autres
    /// architectures hybrides (Qwen3.5/3.6 `qwen3_5_moe` les emploie aussi) :
    /// sans ce drapeau, les helpers `is_gemma4_*_layer` se déclenchaient à tort
    /// hors Gemma et baked des `rope_dims = head_dim` (RoPE pleine) au lieu du
    /// RoPE partiel correct (`partial_rotary_factor`).
    pub is_gemma4: bool,
    /// Indique que le décodeur appartient à la famille Qwen.
    pub is_qwen: bool,
}

#[derive(Clone)]
/// Paramètre une génération autoregressive.
pub struct GenerationOptions {
    /// Liste les tokens qui arrêtent la génération.
    pub stop_token_ids: Vec<usize>,
    /// Liste les séquences de tokens qui arrêtent la génération après émission.
    pub stop_sequences: Vec<Vec<usize>>,
    /// Définit la température de sampling.
    pub temperature: f32,
    /// Définit le seuil nucleus sampling.
    pub top_p: f32,
    /// Définit le top-k sampling (`0` = désactivé).
    pub top_k: usize,
    /// Définit la graine du sampler déterministe.
    pub seed: u64,
    /// Applique une contrainte de tokens côté CPU avant le sampling.
    pub token_constraint: Option<Arc<dyn crate::guided::TokenConstraint>>,
}

impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            stop_token_ids: Vec::new(),
            stop_sequences: Vec::new(),
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            seed: 0,
            token_constraint: None,
        }
    }
}

impl std::fmt::Debug for GenerationOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GenerationOptions")
            .field("stop_token_ids", &self.stop_token_ids)
            .field("stop_sequences", &self.stop_sequences)
            .field("temperature", &self.temperature)
            .field("top_p", &self.top_p)
            .field("top_k", &self.top_k)
            .field("seed", &self.seed)
            .field("token_constraint", &self.token_constraint.is_some())
            .finish()
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

#[derive(Clone, Debug)]
/// Capture un préfill exact à une frontière de prompt.
pub struct CausalDecoderPromptState {
    cache: CausalDecoderCache,
    final_state: Tensor,
}

impl CausalDecoderPromptState {
    /// Construit un état de prompt depuis un cache et un état final.
    #[must_use]
    pub fn new(cache: CausalDecoderCache, final_state: Tensor) -> Self {
        Self { cache, final_state }
    }

    /// Renvoie la position absolue du prochain token.
    #[must_use]
    pub fn position(&self) -> usize {
        self.cache.position()
    }

    /// Estime l'empreinte CPU du snapshot.
    #[must_use]
    pub fn estimated_cpu_bytes(&self) -> usize {
        self.cache.estimated_cpu_bytes().saturating_add(
            self.final_state
                .len()
                .saturating_mul(std::mem::size_of::<f32>()),
        )
    }

    fn into_parts(self) -> (CausalDecoderCache, Tensor) {
        (self.cache, self.final_state)
    }
}

#[derive(Clone, Debug, Default)]
/// Capture les états Metal associés à un état de prompt.
pub struct CausalDecoderPromptMetalSnapshot {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    linear: Vec<Option<crate::metal_backend::LinearAttentionMetalState>>,
}

impl CausalDecoderPromptMetalSnapshot {
    /// Estime l'empreinte GPU du snapshot.
    #[must_use]
    pub fn estimated_bytes(&self) -> usize {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            self.linear
                .iter()
                .flatten()
                .map(crate::metal_backend::LinearAttentionMetalState::estimated_bytes)
                .sum()
        }
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            0
        }
    }
}

#[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
#[derive(Clone, Debug, PartialEq)]
/// Décrit la divergence d'une couche du decode résident.
pub struct ResidentLinearXrayLayerDiff {
    /// Indice de la couche comparée.
    pub layer_index: usize,
    /// Type d'attention de la couche (`full` ou `linear`).
    pub attention_kind: String,
    /// Ecart absolu maximum entre la référence et le résident.
    pub max_abs: f32,
    /// Ecart absolu moyen entre la référence et le résident.
    pub mean_abs: f32,
}

#[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
#[derive(Clone, Debug, PartialEq)]
/// Contient le diagnostic d'un pas résident-linear.
pub struct ResidentLinearXrayReport {
    /// Token injecté dans le pas de decode comparé.
    pub input_token: usize,
    /// Token greedy produit par la référence per-op.
    pub reference_token: usize,
    /// Token greedy produit par le résident full.
    pub resident_token: usize,
    /// Ecart absolu maximum sur le final state post-norm.
    pub final_max_abs: f32,
    /// Ecart absolu moyen sur le final state post-norm.
    pub final_mean_abs: f32,
    /// Indice de la couche linear-attn sondée finement.
    pub probe_layer_index: Option<usize>,
    /// Ecart maximum après input RMSNorm de la couche sondée.
    pub probe_normed_max_abs: Option<f32>,
    /// Ecart moyen après input RMSNorm de la couche sondée.
    pub probe_normed_mean_abs: Option<f32>,
    /// Ecart maximum après linear-attn de la couche sondée.
    pub probe_attn_max_abs: Option<f32>,
    /// Ecart moyen après linear-attn de la couche sondée.
    pub probe_attn_mean_abs: Option<f32>,
    /// Ecart maximum après linear-attn avec `normed` CPU injecté.
    pub probe_attn_cpu_normed_max_abs: Option<f32>,
    /// Ecart moyen après linear-attn avec `normed` CPU injecté.
    pub probe_attn_cpu_normed_mean_abs: Option<f32>,
    /// Ecart maximum du cache conv avant le step resident.
    pub probe_init_state_conv_max_abs: Option<f32>,
    /// Ecart moyen du cache conv avant le step resident.
    pub probe_init_state_conv_mean_abs: Option<f32>,
    /// Ecart maximum du cache SSM avant le step resident.
    pub probe_init_state_ssm_max_abs: Option<f32>,
    /// Ecart moyen du cache SSM avant le step resident.
    pub probe_init_state_ssm_mean_abs: Option<f32>,
    /// Ecart maximum du cache conv après linear-attn.
    pub probe_state_conv_max_abs: Option<f32>,
    /// Ecart moyen du cache conv après linear-attn.
    pub probe_state_conv_mean_abs: Option<f32>,
    /// Ecart maximum du cache SSM après linear-attn.
    pub probe_state_ssm_max_abs: Option<f32>,
    /// Ecart moyen du cache SSM après linear-attn.
    pub probe_state_ssm_mean_abs: Option<f32>,
    /// Ecart maximum du cache conv avec `normed` CPU injecté.
    pub probe_state_cpu_normed_conv_max_abs: Option<f32>,
    /// Ecart moyen du cache conv avec `normed` CPU injecté.
    pub probe_state_cpu_normed_conv_mean_abs: Option<f32>,
    /// Ecart maximum du cache SSM avec `normed` CPU injecté.
    pub probe_state_cpu_normed_ssm_max_abs: Option<f32>,
    /// Ecart moyen du cache SSM avec `normed` CPU injecté.
    pub probe_state_cpu_normed_ssm_mean_abs: Option<f32>,
    /// Divergence mesurée après chaque couche.
    pub layer_diffs: Vec<ResidentLinearXrayLayerDiff>,
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
    /// Temps decode-only de la boucle spéculative.
    pub loop_duration: Duration,
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
    pipeline_hidden_ring: Vec<(GpuTensor, GpuTensor)>,
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
    draft_lm_head: MetalLinearWeightBuffers,
    kv: FullAttentionMetalState,
    #[cfg(feature = "devtools")]
    append_oracle_kv: FullAttentionMetalState,
    #[cfg(feature = "devtools")]
    append_oracle_len: usize,
    hidden_a: GpuTensor,
    hidden_b: GpuTensor,
    current_is_a: bool,
    index: GpuTensor,
    draft_indices: GpuTensor,
    verify_hidden_rows: GpuTensor,
    pending_append_indices: GpuTensor,
    pending_append_start: usize,
    pending_append_count: usize,
    embedding: GpuTensor,
    concat: GpuTensor,
    fc_out: GpuTensor,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[derive(Debug)]
struct ResidentVerifyCaptures {
    base_position: usize,
    linear: Vec<Option<Vec<LinearAttentionMetalState>>>,
    _linear_leases: Vec<ScratchLease>,
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
    #[cfg_attr(
        not(feature = "devtools"),
        expect(dead_code, reason = "lu par le xray résident-linear (devtools)")
    )]
    target_hidden: Option<Tensor>,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[derive(Debug)]
struct ResidentMtpSpecTwoOutput {
    drafts: [usize; 2],
    targets: [usize; 3],
    accepted_for_stats: [bool; 2],
    checked: usize,
    accepted_generated: usize,
    committed_rows: usize,
    pending: usize,
    final_state: Tensor,
    bonus_verified: bool,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[derive(Debug)]
#[allow(
    clippy::large_enum_variant,
    reason = "arène résidente allouée une fois: boxer les variantes ajouterait une allocation et une indirection au decode"
)]
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
    qkv_proj: Option<MetalLinearWeightBuffers>,
    q_proj: MetalLinearWeightBuffers,
    k_proj: MetalLinearWeightBuffers,
    v_proj: MetalLinearWeightBuffers,
    o_proj: MetalLinearWeightBuffers,
    q_norm: metal::Buffer,
    k_norm: metal::Buffer,
    post_norm: metal::Buffer,
    pre_feedforward_norm: Option<metal::Buffer>,
    post_feedforward_norm: Option<metal::Buffer>,
    layer_scalar: Option<f32>,
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
    pre_feedforward_norm: Option<metal::Buffer>,
    post_feedforward_norm: Option<metal::Buffer>,
    layer_scalar: Option<f32>,
    moe: MetalMoeRoutedWeights,
    top_k: usize,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[derive(Debug)]
struct ResidentLinearMoeBuffers {
    input_norm: metal::Buffer,
    linear: MetalLinearAttnResidentDenseWeights,
    post_norm: metal::Buffer,
    moe: MetalMoeSharedWeights,
    top_k: usize,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[derive(Debug)]
struct ResidentFullDenseBuffers {
    input_norm: metal::Buffer,
    qkv_proj: Option<MetalLinearWeightBuffers>,
    value_norm: bool,
    q_proj: MetalLinearWeightBuffers,
    k_proj: MetalLinearWeightBuffers,
    v_proj: MetalLinearWeightBuffers,
    o_proj: MetalLinearWeightBuffers,
    q_norm: metal::Buffer,
    k_norm: metal::Buffer,
    post_norm: metal::Buffer,
    pre_feedforward_norm: Option<metal::Buffer>,
    post_feedforward_norm: Option<metal::Buffer>,
    layer_scalar: Option<f32>,
    gate_proj: MetalLinearWeightBuffers,
    up_proj: MetalLinearWeightBuffers,
    down_proj: MetalLinearWeightBuffers,
    parallel_moe: Option<ResidentGemmaParallelMoeBuffers>,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[derive(Debug)]
struct ResidentGemmaParallelMoeBuffers {
    post_feedforward_norm_1: metal::Buffer,
    moe: MetalMoeRoutedWeights,
    top_k: usize,
    router_norm: Option<(metal::Buffer, f32)>,
    per_expert_scale: Option<metal::Buffer>,
    pre_feedforward_norm_2: metal::Buffer,
    post_feedforward_norm_2: metal::Buffer,
    dense_inter_dim: usize,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl ResidentFullDenseBuffers {
    fn parallel_moe_weights(&self) -> Option<GemmaParallelMoeTailWeights<'_>> {
        let parallel = self.parallel_moe.as_ref()?;
        Some(GemmaParallelMoeTailWeights {
            dense_gate_proj: &self.gate_proj,
            dense_up_proj: &self.up_proj,
            dense_down_proj: &self.down_proj,
            pre_feedforward_norm: self.pre_feedforward_norm.as_ref()?,
            post_feedforward_norm_1: &parallel.post_feedforward_norm_1,
            moe: &parallel.moe,
            top_k: parallel.top_k,
            router_norm: parallel
                .router_norm
                .as_ref()
                .map(|(weight, eps)| (weight.as_ref(), *eps)),
            per_expert_scale: parallel
                .per_expert_scale
                .as_ref()
                .map(metal::Buffer::as_ref),
            pre_feedforward_norm_2: &parallel.pre_feedforward_norm_2,
            post_feedforward_norm_2: &parallel.post_feedforward_norm_2,
            post_feedforward_norm: self.post_feedforward_norm.as_ref()?,
            layer_scalar: self.layer_scalar,
            dense_inter_dim: parallel.dense_inter_dim,
        })
    }
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

    /// Estime l'empreinte CPU du cache.
    #[must_use]
    pub fn estimated_cpu_bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| {
                layer
                    .keys
                    .len()
                    .saturating_add(layer.values.len())
                    .saturating_mul(std::mem::size_of::<f32>())
                    .saturating_add(layer.linear.estimated_cpu_bytes())
            })
            .sum()
    }

    #[cfg(test)]
    pub(crate) fn layer_kv_for_test(
        &self,
        layer_index: usize,
    ) -> Option<(&[f32], &[f32], Option<usize>)> {
        self.layers
            .get(layer_index)
            .map(|layer| (layer.keys.as_slice(), layer.values.as_slice(), layer.kv_dim))
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
            num_global_key_value_heads: None,
            head_dim: None,
            global_head_dim: None,
            rope_dims: None,
            rope_full_dims: None,
            rope_sliding_dims: None,
            attn_output_gate: false,
            layer_types: Vec::new(),
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
            embed_scale: None,
            rope_local_base_freq: None,
            rope_full_base_freq: None,
            rope_position_scale: None,
            sliding_window: None,
            sliding_window_pattern: None,
            attention_k_eq_v: false,
            attention_value_norm: false,
            parallel_moe: false,
            final_logit_softcapping: None,
            query_pre_attn_scalar: None,
            activation: crate::Activation::Silu,
            rope_style: RopeStyle::Halves,
            is_gemma4: false,
            is_qwen: false,
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
            num_global_key_value_heads: config.num_global_key_value_heads,
            head_dim: Some(config.head_dim()),
            global_head_dim: config.global_head_dim,
            rope_dims: Some(config.rope_dims()),
            rope_full_dims: config.is_gemma4().then(|| {
                (config.global_head_dim.unwrap_or_else(|| config.head_dim()) as f32
                    * config.rope_full_partial_rotary_factor.unwrap_or(1.0))
                    as usize
            }),
            rope_sliding_dims: config.is_gemma4().then(|| {
                (config.head_dim() as f32
                    * config.rope_sliding_partial_rotary_factor.unwrap_or(1.0))
                    as usize
            }),
            attn_output_gate: config.attn_output_gate.unwrap_or(false),
            layer_types: config.layer_types.clone(),
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
            embed_scale: config.embed_scale(),
            rope_local_base_freq: config.rope_local_base_freq,
            rope_full_base_freq: config.rope_full_base_freq,
            rope_position_scale: config.rope_position_scale(),
            sliding_window: config.sliding_window,
            sliding_window_pattern: config.sliding_window_pattern,
            attention_k_eq_v: config.attention_k_eq_v,
            attention_value_norm: config.is_gemma4(),
            parallel_moe: config.enable_moe_block,
            final_logit_softcapping: config.final_logit_softcapping,
            query_pre_attn_scalar: if config.is_gemma4() {
                Some(config.query_pre_attn_scalar.unwrap_or(1.0))
            } else {
                config.query_pre_attn_scalar
            },
            activation: if config.uses_gelu_tanh() {
                crate::Activation::GeluTanh
            } else {
                crate::Activation::Silu
            },
            // Rotate-half pour TOUTE la famille supportée (Qwen, Llama, Mistral,
            // Gemma) : convention d'entraînement des checkpoints, alignée mlx_lm.
            rope_style: RopeStyle::Halves,
            is_gemma4: config.is_gemma4(),
            is_qwen: config.model_type.starts_with("qwen"),
        }
    }
}

impl CausalDecoderConfig {
    fn layer_kind(&self, layer_index: usize) -> Option<&str> {
        self.layer_types.get(layer_index).map(String::as_str)
    }

    fn is_gemma4_full_layer(&self, layer_index: usize) -> bool {
        // GARDE: `full_attention` est aussi un type de couche de qwen3_5_moe
        // (Qwen3.5/3.6 hybride). Sans le drapeau `is_gemma4`, on traitait ces
        // couches comme Gemma → RoPE pleine (`rope_dims = head_dim`) au lieu du
        // RoPE partiel correct, corrompant Q/K et donc l'attention.
        self.is_gemma4 && self.layer_kind(layer_index) == Some("full_attention")
    }

    fn is_gemma4_sliding_layer(&self, layer_index: usize) -> bool {
        self.is_gemma4 && self.layer_kind(layer_index) == Some("sliding_attention")
    }

    fn is_full_attention_layer(&self, layer_index: usize) -> bool {
        match self.full_attention_interval {
            Some(interval) if interval > 0 => (layer_index + 1) % interval == 0,
            _ => true,
        }
    }

    fn is_resident_full_attention_layer(&self, layer_index: usize) -> bool {
        if self.is_gemma4 {
            self.is_gemma4_full_layer(layer_index) || self.is_gemma4_sliding_layer(layer_index)
        } else {
            self.is_full_attention_layer(layer_index)
        }
    }

    /// Indique si une couche est locale (sliding-window) dans le motif Gemma 3.
    ///
    /// Gemma 3 alterne couches locales et globales : une couche globale toutes
    /// les `sliding_window_pattern` couches (la `pattern`-ième), locales sinon.
    /// Sans motif déclaré, aucune couche n'est locale.
    fn is_local_sliding_layer(&self, layer_index: usize) -> bool {
        if !self.layer_types.is_empty() {
            return self.is_gemma4_sliding_layer(layer_index);
        }
        matches!(
            self.sliding_window_pattern,
            Some(pattern) if pattern > 0 && (layer_index + 1) % pattern != 0
        )
    }

    fn layer_head_dim(&self, layer_index: usize) -> Result<usize> {
        let head_dim = if self.is_gemma4_full_layer(layer_index) {
            self.global_head_dim.or(self.head_dim)
        } else {
            self.head_dim
        };
        head_dim.ok_or_else(|| InferError::Dimension("head_dim manquant".to_string()))
    }

    fn layer_num_key_value_heads(&self, layer_index: usize) -> usize {
        if self.is_gemma4_full_layer(layer_index) {
            self.num_global_key_value_heads
                .unwrap_or(self.num_key_value_heads)
        } else {
            self.num_key_value_heads
        }
    }

    fn layer_rope_dims(&self, layer_index: usize, head_dim: usize) -> usize {
        if self.is_gemma4_full_layer(layer_index) {
            self.rope_full_dims.unwrap_or(head_dim)
        } else if self.is_gemma4_sliding_layer(layer_index) {
            self.rope_sliding_dims.unwrap_or(head_dim)
        } else {
            self.rope_dims.unwrap_or(head_dim)
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn resident_full_attention_kv_dim(&self, layer_index: usize) -> Result<usize> {
        let head_dim = self.layer_head_dim(layer_index)?;
        let kv_heads = self.layer_num_key_value_heads(layer_index);
        kv_heads
            .checked_mul(head_dim)
            .ok_or_else(|| InferError::Dimension("kv_dim résident déborde".to_string()))
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn resident_full_attn_layer_dims(
        &self,
        layer_index: usize,
        hidden: usize,
        position: usize,
        eps: f32,
        theta: f32,
    ) -> Result<FullAttnLayerDims> {
        let head_dim = self.layer_head_dim(layer_index)?;
        Ok(FullAttnLayerDims {
            hidden,
            q_heads: self.num_attention_heads,
            kv_heads: self.layer_num_key_value_heads(layer_index),
            head_dim,
            attn_scalar: self.query_pre_attn_scalar.unwrap_or(head_dim as f32),
            rope_dims: self.layer_rope_dims(layer_index, head_dim),
            rope_frequency_dim: self.layer_rope_frequency_dim(layer_index, head_dim),
            position,
            window_start: 0,
            eps,
            theta: self.layer_rope_theta_override(layer_index).unwrap_or(theta),
            attn_output_gate: self.attn_output_gate,
        })
    }

    fn layer_rope_frequency_dim(&self, layer_index: usize, head_dim: usize) -> usize {
        if self.is_gemma4_full_layer(layer_index) {
            head_dim
        } else {
            self.layer_rope_dims(layer_index, head_dim)
        }
    }

    /// Renvoie la base RoPE locale d'une couche Gemma 3, ou `None` (base globale).
    ///
    /// Les couches locales utilisent `rope_local_base_freq`, les globales
    /// `rope_theta`. Hors Gemma 3 renvoie `None` → la base unique `rope_theta`
    /// s'applique (byte-identique).
    fn layer_rope_theta_override(&self, layer_index: usize) -> Option<f32> {
        if self.is_gemma4_full_layer(layer_index) {
            return self.rope_full_base_freq.or(self.rope_theta);
        }
        self.rope_local_base_freq
            .filter(|_| self.is_local_sliding_layer(layer_index))
    }

    /// Renvoie la fenêtre d'attention d'une couche locale Gemma 3, ou `None`.
    ///
    /// Seules les couches locales du motif `sliding_window_pattern` masquent
    /// leur attention aux `sliding_window` derniers tokens ; les couches
    /// globales (et tous les modèles non-Gemma) attendent sur tout le contexte.
    fn layer_sliding_window(&self, layer_index: usize) -> Option<usize> {
        self.sliding_window
            .filter(|window| *window > 0 && self.is_local_sliding_layer(layer_index))
    }

    /// Renvoie l'échelle des positions RoPE d'une couche, ou `None` (positions brutes).
    ///
    /// Gemma 3 ≥4B n'étire que les couches globales (linear ×8) : les couches
    /// locales du motif sliding gardent leurs positions brutes sur
    /// `rope_local_base_freq`, comme `initialize_rope` de mlx_lm. Sans motif
    /// sliding (linear historique type Llama 2), l'échelle s'applique partout.
    fn layer_rope_position_scale(&self, layer_index: usize) -> Option<f32> {
        if !self.layer_types.is_empty() {
            return None;
        }
        self.rope_position_scale
            .filter(|_| !self.is_local_sliding_layer(layer_index))
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
    mtp_draft_lm_head: Option<Linear>,
    prefix_cache: Arc<Mutex<PrefixCache>>,
    memory_guard: Option<MemoryGuard>,
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
    #[cfg(all(target_os = "macos", feature = "metal"))]
    linear_metal: Vec<Option<crate::metal_backend::LinearAttentionMetalState>>,
}

impl PrefixCacheEntry {
    fn estimated_bytes(&self) -> u64 {
        let mut bytes = self
            .tokens
            .len()
            .saturating_mul(std::mem::size_of::<usize>())
            .saturating_add(self.cache.estimated_cpu_bytes())
            .saturating_add(
                self.final_state
                    .len()
                    .saturating_mul(std::mem::size_of::<f32>()),
            );
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            bytes = bytes.saturating_add(
                self.linear_metal
                    .iter()
                    .flatten()
                    .map(crate::metal_backend::LinearAttentionMetalState::estimated_bytes)
                    .sum::<usize>(),
            );
        }
        usize_to_u64_saturating(bytes)
    }
}

impl PrefixCache {
    fn evict_lru_entry(&mut self) -> Option<u64> {
        self.entries.pop().map(|entry| entry.estimated_bytes())
    }
}

fn usize_to_u64_saturating(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

impl CausalDecoder {
    fn register_prefix_cache_purgeable(&self) {
        let cache = Arc::downgrade(&self.prefix_cache);
        let name = format!("decoder-prefix-cache:{:p}", Arc::as_ptr(&self.prefix_cache));
        if let Ok(mut registry) = process_purge_registry().lock() {
            registry.register(100, name, move |_, _| {
                let Some(cache) = cache.upgrade() else {
                    return PurgeOutcome::Empty;
                };
                let outcome = match cache.lock() {
                    Ok(mut cache) => cache
                        .evict_lru_entry()
                        .map(|bytes| PurgeOutcome::Purged { bytes })
                        .unwrap_or(PurgeOutcome::Empty),
                    Err(_) => PurgeOutcome::Empty,
                };
                outcome
            });
        }
    }
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
    parallel_moe: Option<FeedForward>,
    // NOTE: Présentes seulement sur Gemma (double norme feed-forward) ; leur
    // présence bascule le forward sur le câblage Gemma (post-attn + pre/post FFN).
    pre_feedforward_norm: Option<Tensor>,
    post_feedforward_norm: Option<Tensor>,
    pre_feedforward_norm_2: Option<Tensor>,
    post_feedforward_norm_1: Option<Tensor>,
    post_feedforward_norm_2: Option<Tensor>,
    layer_scalar: Option<Tensor>,
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
    v_proj: Option<Linear>,
    o_proj: Linear,
    q_norm: Option<Tensor>,
    k_norm: Option<Tensor>,
    num_key_value_heads: Option<usize>,
    head_dim: Option<usize>,
    rope_dims: Option<usize>,
    rope_frequency_dim: Option<usize>,
    value_norm: bool,
    // NOTE: Surcharge la base RoPE de la couche (Gemma 3 : locale vs globale) ;
    // `None` = base unique `config.rope_theta` (Qwen/Llama/Mistral, byte-identique).
    rope_theta: Option<f32>,
    // NOTE: Échelle des positions RoPE de la couche (rope_scaling linear des
    // couches globales Gemma 3 ≥4B) ; `None` = positions brutes (byte-identique).
    rope_position_scale: Option<f32>,
    // NOTE: Fenêtre d'attention des couches locales Gemma 3 ; `None` = attention
    // causale pleine (toutes les autres couches et tous les modèles non-Gemma).
    sliding_window: Option<usize>,
}

impl CausalDecoder {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn has_resident_linear_attention_layer(&self) -> bool {
        self.layers
            .iter()
            .any(|layer| matches!(&layer.attention, AttentionBlock::Linear(_)))
    }

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
            mtp_draft_lm_head: None,
            prefix_cache: Arc::new(Mutex::new(PrefixCache::default())),
            memory_guard: None,
            #[cfg(all(target_os = "macos", feature = "metal"))]
            runtime: DecoderRuntime::default(),
        })
    }

    /// Attache une garde mémoire aux allocations résidentes de ce décodeur.
    #[must_use]
    pub fn with_memory_guard(mut self, guard: MemoryGuard) -> Self {
        self.register_prefix_cache_purgeable();
        self.memory_guard = Some(guard);
        self
    }

    /// Plonge des tokens et applique l'échelle d'embedding éventuelle (`√hidden` Gemma).
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si la plongée échoue.
    pub(in crate::decoder) fn embed_scaled(&self, token_ids: &[usize]) -> Result<Tensor> {
        let hidden = embed_weight_tokens(&self.embed_tokens, token_ids)?;
        let hidden = match self.config.embed_scale {
            Some(scale) => hidden.map(|value| value * scale),
            None => hidden,
        };
        Ok(recast_qwen_embedding(
            hidden,
            self.config.is_qwen,
            qwen_embed_bf16_enabled(),
        ))
    }

    /// Calcule les logits du prochain token sans cache.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le forward échoue.
    pub fn next_logits(&self, token_ids: &[usize]) -> Result<Tensor> {
        let runtime = self.forward_runtime();
        let mut hidden = self.embed_scaled(token_ids)?;
        for layer in &self.layers {
            hidden = layer.forward(&self.config, &hidden, runtime)?;
        }
        let final_state = rms_norm(&hidden, &self.final_norm, self.config.rms_eps)?;
        let logits = self.lm_head.forward_with_runtime(&final_state, runtime)?;
        self.finalize_logits(&logits)
    }

    /// Pré-remplit le cache depuis des embeddings déjà préparés.
    ///
    /// Qwen3-TTS prépare ses entrées comme un overlay texte+codec ; elles ne
    /// proviennent donc pas directement de `embed_tokens(token_id)`.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si la forme des embeddings ou le forward échoue.
    pub fn prefill_cache_from_embeddings(
        &self,
        input_embeds: &Tensor,
    ) -> Result<(CausalDecoderCache, Tensor)> {
        let (seq, hidden_dim) = input_embeds.as_matrix()?;
        if seq == 0 {
            return Err(InferError::Dimension(
                "embeddings préfixe vides".to_string(),
            ));
        }
        let expected_hidden = self.final_norm.len();
        if hidden_dim != expected_hidden {
            return Err(InferError::Dimension(format!(
                "embeddings préfixe attendus hidden={expected_hidden}, reçu {hidden_dim}"
            )));
        }
        let runtime = self.forward_runtime();
        let mut cache = self.empty_cache();
        let mut hidden = input_embeds.clone();
        for (layer_index, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward_prefill(
                &self.config,
                &hidden,
                &mut cache.layers[layer_index],
                0,
                runtime,
            )?;
        }
        cache.position = seq;
        let final_hidden = Tensor::row(hidden.last_row()?.to_vec())?;
        let final_state = rms_norm(&final_hidden, &self.final_norm, self.config.rms_eps)?;
        Ok((cache, final_state))
    }

    /// Avance le décodeur d'une position depuis un embedding déjà préparé.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si la forme de l'embedding ou le forward échoue.
    pub fn next_state_from_embedding(
        &self,
        cache: &mut CausalDecoderCache,
        input_embed: &Tensor,
    ) -> Result<Tensor> {
        let (rows, hidden_dim) = input_embed.as_matrix()?;
        if rows != 1 {
            return Err(InferError::Dimension(format!(
                "embedding decode attendu [1, hidden], reçu {:?}",
                input_embed.shape()
            )));
        }
        let expected_hidden = self.final_norm.len();
        if hidden_dim != expected_hidden {
            return Err(InferError::Dimension(format!(
                "embedding decode attendu hidden={expected_hidden}, reçu {hidden_dim}"
            )));
        }
        if cache.layers.len() != self.layers.len() {
            return Err(InferError::Dimension(format!(
                "cache couches={} incompatible avec décodeur couches={}",
                cache.layers.len(),
                self.layers.len()
            )));
        }
        let position = cache.position;
        let runtime = self.forward_runtime();
        let mut hidden = input_embed.clone();
        for (layer_index, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward_cached(
                &self.config,
                &hidden,
                &mut cache.layers[layer_index],
                position,
                runtime,
            )?;
        }
        cache.position += 1;
        rms_norm(&hidden, &self.final_norm, self.config.rms_eps)
    }

    /// Prépare l'arène GPU résidente pour décoder depuis des embeddings (TTS).
    ///
    /// Renvoie `true` si l'arène est prête (decode résident complet applicable et
    /// KV seedé depuis le prefill) ; `false` sinon (l'appelant reste sur le
    /// per-op via [`Self::next_state_from_embedding`]). Sans la feature Metal,
    /// renvoie toujours `false`.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si la compilation des kernels, une allocation ou un seed
    /// résident échoue.
    pub fn setup_resident_decode_from_prefill(
        &self,
        cache: &mut CausalDecoderCache,
        max_steps: usize,
    ) -> Result<bool> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            if !self.supports_resident_full_decode() {
                return Ok(false);
            }
            // Chemins greedy (TTS talker argmax, spéculatif/DFlash) → KV f32 exact.
            self.setup_resident_full_decode(cache, max_steps, 0, false)
        }
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (cache, max_steps);
            Ok(false)
        }
    }

    /// Avance le décodeur d'une position depuis un embedding via le decode GPU
    /// résident (1 token = 1 command buffer) et renvoie le `final_state` post-norm.
    ///
    /// Renvoie `Ok(None)` si le résident n'est pas disponible (executor Metal ou
    /// arène absente, ou build sans Metal) → l'appelant retombe sur
    /// [`Self::next_state_from_embedding`]. Nécessite un appel préalable réussi à
    /// [`Self::setup_resident_decode_from_prefill`].
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si la forme de l'embedding est invalide ou si un encodage
    /// Metal échoue.
    pub fn decode_step_resident_from_embedding(
        &self,
        cache: &mut CausalDecoderCache,
        embedding: &Tensor,
    ) -> Result<Option<Tensor>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            match self.next_resident_embedding_step(cache, embedding, None)? {
                Some(crate::decoder::resident::ResidentEmbeddingOut::State(state)) => {
                    Ok(Some(state))
                }
                Some(crate::decoder::resident::ResidentEmbeddingOut::Token(_)) => Err(
                    InferError::Metal("pas résident embedding: token inattendu".to_string()),
                ),
                None => Ok(None),
            }
        }
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (cache, embedding);
            Ok(None)
        }
    }

    /// Avance le décodeur d'un pas depuis un embedding via le decode GPU résident
    /// et renvoie directement le token = argmax greedy on-device de `head`
    /// (lm_head/tête fournie). Matmul + argmax fusionnés dans le command buffer du
    /// pas → aucun readback du `final_state`, aucun matmul de tête CPU. Réservé aux
    /// têtes greedy SANS suppression (TTS code_predictor) ; la tête est résolue en
    /// buffers Metal en interne (cache par pointeur).
    ///
    /// Renvoie `Ok(None)` si le résident n'est pas disponible (fallback per-op).
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si la forme de l'embedding est invalide, si la tête a un
    /// biais ou si un encodage Metal échoue.
    pub fn decode_token_resident_from_embedding_head(
        &self,
        cache: &mut CausalDecoderCache,
        embedding: &Tensor,
        head: &crate::Linear,
    ) -> Result<Option<usize>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(metal) = self.forward_runtime().metal_executor() else {
                return Ok(None);
            };
            let head_buffers = metal.resolve_linear_weight_buffers(head.weight(), "tts_cp_head")?;
            match self.next_resident_embedding_step(cache, embedding, Some(&head_buffers))? {
                Some(crate::decoder::resident::ResidentEmbeddingOut::Token(token)) => {
                    Ok(Some(token))
                }
                Some(crate::decoder::resident::ResidentEmbeddingOut::State(_)) => Err(
                    InferError::Metal("pas résident tête: état inattendu".to_string()),
                ),
                None => Ok(None),
            }
        }
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (cache, embedding, head);
            Ok(None)
        }
    }

    /// Crée un cache résident « from scratch » (sans préfixe) pour décoder une
    /// courte séquence entièrement en GPU depuis des embeddings (TTS
    /// code_predictor). L'arène est allouée une fois ; la remettre à zéro entre
    /// séquences avec [`Self::reset_resident_decode_cache`].
    ///
    /// Renvoie `Ok(None)` si le decode résident n'est pas applicable (build sans
    /// Metal, ou modèle inéligible).
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si la compilation des kernels ou une allocation échoue.
    pub fn new_resident_decode_cache(
        &self,
        max_steps: usize,
    ) -> Result<Option<CausalDecoderCache>> {
        let mut cache = self.empty_cache();
        if self.setup_resident_decode_from_prefill(&mut cache, max_steps)? {
            Ok(Some(cache))
        } else {
            Ok(None)
        }
    }

    /// Remet un cache résident à zéro pour un nouveau décodage from-scratch : vide
    /// le KV résident de chaque couche full-attn (`len = 0`) et remet la position
    /// à 0. Réservé aux décodeurs full-attn (le code_predictor TTS) ; les états
    /// linear-attn ne sont pas réinitialisés ici.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si la troncature d'un état KV résident échoue.
    pub fn reset_resident_decode_cache(&self, cache: &mut CausalDecoderCache) -> Result<()> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        for layer_cache in cache.layers.iter_mut() {
            if let Some(full) = layer_cache.full.as_mut() {
                full.truncate(0)?;
            }
        }
        cache.position = 0;
        Ok(())
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

    /// Charge une projection logits dédiée aux drafts MTP.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le sidecar est incomplet ou incompatible.
    pub fn with_mtp_draft_lm_head_sidecar(mut self, path: impl AsRef<Path>) -> Result<Self> {
        self.mtp_draft_lm_head = Some(mtp::load_mtp_draft_lm_head(
            path,
            self.final_norm.data().len(),
        )?);
        Ok(self)
    }

    fn mtp_draft_lm_head(&self) -> &Linear {
        match self.mtp_draft_lm_head.as_ref() {
            Some(head) => head,
            None => &self.lm_head,
        }
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

fn bf16_round_f32(value: f32) -> f32 {
    let bits = value.to_bits();
    let rounding = 0x7fff_u32 + ((bits >> 16) & 1);
    f32::from_bits(bits.wrapping_add(rounding) & 0xffff_0000)
}

fn recast_qwen_embedding(hidden: Tensor, is_qwen: bool, enabled: bool) -> Tensor {
    if is_qwen && enabled {
        hidden.map(bf16_round_f32)
    } else {
        hidden
    }
}
