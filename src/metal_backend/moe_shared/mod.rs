//! Encodage Metal du MoE avec expert partagé (Qwen3-MoE : routed + shared).
//!
//! Chemin RÉSIDENT : tout est encodé dans l'encoder fourni par l'appelant
//! (decode résident, prefill, duo light-batch), sans commit ni readback
//! intermédiaire — les indices d'experts sélectionnés restent sur GPU.
//!
//! # Phases d'un forward (un token)
//!
//! 1. **Router** : QMV `input × router` → logits routeur (`expert_count`).
//! 2. **Gating top-k** : kernel `topk_softmax` → `indices` (u32) + `scores`
//!    (f32). Sémantique « norm_topk_prob » : softmax stable (max soustrait)
//!    restreint aux k logits max, donc les k scores somment à 1. Égalités
//!    départagées par l'indice le plus bas, même règle que le tri stable CPU
//!    de `mlp::top_k_indices` (au signe de zéro près : `total_cmp` ordonne
//!    `-0.0 < +0.0` là où le kernel les traite égaux).
//! 3. **Gather routed** : projections gate/up des k experts, lues par
//!    indirection GPU dans les poids empilés (`stacked`) via `indices`.
//! 4. **Activation expert** : `silu(gate) ⊙ up → hidden` pour Qwen, ou
//!    `gelu_tanh(gate) ⊙ up` pour Gemma. La fusion gate+up+SwiGLU reste réservée
//!    au chemin Qwen ; Gemma réutilise le gather déplié puis `geglu_tanh_f32`.
//! 5. **Down** : gather down-proj par expert → `top_k` lignes de `out_dim`.
//! 6. **Combine** : `Σ scores·down` (+ résiduel attention si `residual`)
//!    puis shared expert gaté : `out += σ(gate_scalar) · shared_down`.
//!
//! Le shared expert (MLP SwiGLU dense + gate scalaire sigmoïde) suit les mêmes
//! étapes projection/SwiGLU/down, sans routage. Des fusions opportunistes u8
//! (qmv+gate scalaire, gate+up+SwiGLU+gate, down pondéré+résiduel+shared)
//! réduisent le nombre de dispatches — le MoE decode est dispatch-bound — et
//! chacune garde un repli déplié au résultat identique.
//!
//! # Recouvrement routed‖shared (`moe_shared_route_overlap_enabled`)
//!
//! Par défaut, `encode_moe_shared_buffers` réordonne l'encodage en vagues :
//! projections routed ET shared, barrière, SwiGLU restants, barrière, down
//! routed ET shared, barrière, combine. Les deux chaînes n'ayant aucune
//! dépendance croisée avant le combine, elles peuvent se recouvrir sur GPU.
//! Dans cette fenêtre les barrières par-dispatch (`post_dispatch_barrier`)
//! sont suspendues (`suspend_dispatch_barrier_scope`) — sans effet sous
//! l'encodeur SÉRIE, qui ordonne déjà ses dispatches. Sous l'encodeur
//! CONCURRENT (`resident_concurrent_enabled`), le recouvrement corrompt la
//! sortie (charabia) malgré des scratch disjoints et des barrières de phase
//! explicites — sémantique subtile du dispatch concurrent — : les barrières
//! par-dispatch restent donc ACTIVES (`install_dispatch_barrier_scope`) dans
//! ce mode, routed/shared sérialisés mais corrects. Le chemin série prod
//! reste byte-identique avec ou sans le réordonnancement.
//!
//! # Scratch : mémoïsation et invariants d'aliasing
//!
//! Les scratch viennent de `private_*_buffer`, mémoïsés par (label, taille,
//! élément, namespace) : le même label rend le MÊME `MTLBuffer` d'un appel à
//! l'autre, donc toutes les couches d'un forward partagent un unique jeu de
//! scratch. C'est valide parce que les phases d'une couche sont ordonnées par
//! les barrières et que les couches se suivent dans le command buffer.
//! Invariants :
//!
//! - les labels routed et shared sont deux à deux DISJOINTS (test
//!   `moe_shared_route_overlap_buffers_are_disjoint`) — précondition du
//!   recouvrement routed‖shared ;
//! - `output_buffer` est écrit par le combine routed puis relu/réécrit par
//!   l'ajout shared (`add_sigmoid_scaled`) : jamais aliasé à un scratch ;
//! - le light-batch isole ses flux par namespace de scratch
//!   (`install_scratch_namespace`), sans changer les clés mono-flux.

use super::*;

mod profiling;
mod routed;
mod scratch;
mod shapes;
mod shared;

/// Regroupe les dimensions validées d'un bloc MoE shared.
pub(super) struct MoeSharedBufferShape {
    /// Nombre d'experts routés (= sorties du routeur).
    pub(super) expert_count: usize,
    /// Dimension intermédiaire des experts routés (sortie gate/up).
    pub(super) inter_dim: usize,
    /// Dimension de sortie du bloc (down-proj, = hidden du modèle).
    pub(super) out_dim: usize,
    /// Dimension intermédiaire du shared expert.
    pub(super) shared_inter_dim: usize,
}

struct MoeSharedLinearShape<'a> {
    expert_count: usize,
    inter_dim: usize,
    out_dim: usize,
    shared_inter_dim: usize,
    stacked: StackedMoeBuffers,
    shared_gate_proj: &'a Linear,
    shared_up_proj: &'a Linear,
    shared_down_proj: &'a Linear,
}

struct MoeRoutedSharedPhase<'a> {
    encoder: &'a ComputeCommandEncoderRef,
    owned_buffers: &'a mut Vec<Buffer>,
    input_buffer: &'a BufferRef,
    rows: usize,
    stacked: &'a StackedMoeBuffers,
    indices: &'a BufferRef,
    gate: &'a BufferRef,
    up: &'a BufferRef,
    hidden: &'a BufferRef,
    down: &'a BufferRef,
    slots: usize,
    inter_dim: usize,
    swiglu_label: &'static str,
}

/// Regroupe les scratch GPU d'un forward MoE shared mono-token.
///
/// Buffers mémoïsés par label (voir doc de module) ; les champs `shared_*`
/// sont disjoints des champs routed — invariant du recouvrement routed‖shared.
pub(super) struct MoeSharedScratch {
    /// Logits du routeur (`expert_count`).
    pub(super) router: Buffer,
    /// Indices u32 des k experts sélectionnés.
    pub(super) indices: Buffer,
    /// Scores softmax-top-k des k experts (somme = 1).
    pub(super) scores: Buffer,
    /// Sorties gate des k experts (`top_k × inter_dim`), chemin déplié.
    pub(super) gate: Buffer,
    /// Sorties up des k experts (`top_k × inter_dim`), chemin déplié.
    pub(super) up: Buffer,
    /// Activations SwiGLU des k experts (`top_k × inter_dim`).
    pub(super) hidden: Buffer,
    /// Sorties down par expert (`top_k × out_dim`), avant pondération.
    pub(super) down: Buffer,
    /// Logit scalaire du gate shared (sigmoïde appliquée au combine).
    pub(super) shared_gate: Buffer,
    /// Sortie gate_proj du shared expert (`shared_inter_dim`), chemin déplié.
    pub(super) shared_proj_gate: Buffer,
    /// Sortie up_proj du shared expert (`shared_inter_dim`), chemin déplié.
    pub(super) shared_up: Buffer,
    /// Activations SwiGLU du shared expert (`shared_inter_dim`).
    pub(super) shared_hidden: Buffer,
    /// Sortie down_proj du shared expert (`out_dim`).
    pub(super) shared_down: Buffer,
}

/// Regroupe les scratch GPU d'un forward MoE shared batché (`rows` tokens).
///
/// Mêmes rôles que [`MoeSharedScratch`], dimensionnés `rows × …` ; le gating
/// produit `rows × top_k` slots contigus (ligne-majeur).
pub(super) struct MoeSharedRowsScratch {
    /// Logits du routeur (`rows × expert_count`).
    pub(super) router: Buffer,
    /// Indices u32 des experts sélectionnés (`rows × top_k`).
    pub(super) indices: Buffer,
    /// Scores softmax-top-k (`rows × top_k`, somme = 1 par ligne).
    pub(super) scores: Buffer,
    /// Sorties gate routées par slot (`rows × top_k × inter_dim`).
    #[allow(
        dead_code,
        reason = "chemin coop n'utilise pas ces scratch routés par slot"
    )]
    pub(super) gate: Buffer,
    /// Sorties up routées par slot (`rows × top_k × inter_dim`).
    #[allow(
        dead_code,
        reason = "chemin coop n'utilise pas ces scratch routés par slot"
    )]
    pub(super) up: Buffer,
    /// Activations SwiGLU routées par slot (`rows × top_k × inter_dim`).
    #[allow(
        dead_code,
        reason = "chemin coop n'utilise pas ces scratch routés par slot"
    )]
    pub(super) hidden: Buffer,
    /// Sorties down par slot (`rows × top_k × out_dim`), avant pondération.
    pub(super) down: Buffer,
    /// Logits scalaires des gates shared (`rows`).
    pub(super) shared_gate: Buffer,
    /// Sorties gate_proj du shared expert (`rows × shared_inter_dim`).
    pub(super) shared_proj_gate: Buffer,
    /// Sorties up_proj du shared expert (`rows × shared_inter_dim`).
    pub(super) shared_up: Buffer,
    /// Activations SwiGLU du shared expert (`rows × shared_inter_dim`).
    pub(super) shared_hidden: Buffer,
    /// Sorties down_proj du shared expert (`rows × out_dim`).
    pub(super) shared_down: Buffer,
}

/// Regroupe les dimensions validées d'un MoE routed-only (sans shared expert).
pub(super) struct MoeRoutedBufferShape {
    pub(super) expert_count: usize,
    pub(super) inter_dim: usize,
    pub(super) out_dim: usize,
}

/// Regroupe les scratch GPU d'un forward MoE routed-only mono-token.
struct MoeRoutedScratch {
    router: Buffer,
    indices: Buffer,
    scores: Buffer,
    gate: Buffer,
    up: Buffer,
    hidden: Buffer,
    down: Buffer,
}

/// Regroupe les scratch GPU d'un forward MoE routed-only batché.
struct MoeRoutedRowsScratch {
    router: Buffer,
    indices: Buffer,
    scores: Buffer,
    gate: Buffer,
    up: Buffer,
    hidden: Buffer,
    down: Buffer,
}
