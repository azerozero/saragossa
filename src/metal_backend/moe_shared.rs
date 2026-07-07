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
//! 4. **SwiGLU** : `silu(gate) ⊙ up → hidden`. Fusion gate+up+SwiGLU en un
//!    dispatch quand les poids s'y prêtent, sinon repli déplié exact.
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

impl MetalExecutor {
    /// Alloue (ou récupère mémoïsés) les scratch du MoE routed-only.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si une taille déborde ou si l'allocation Metal échoue.
    fn allocate_moe_routed_scratch(
        &self,
        top_k: usize,
        expert_count: usize,
        inter_dim: usize,
        out_dim: usize,
    ) -> Result<MoeRoutedScratch> {
        Ok(MoeRoutedScratch {
            router: self.private_f32_buffer(expert_count, "moe_routed_router_logits")?,
            indices: self.private_u32_buffer(top_k, "moe_routed_indices")?,
            scores: self.private_f32_buffer(top_k, "moe_routed_scores")?,
            gate: self.private_f32_buffer(
                checked_len(top_k, inter_dim, "moe routed gate")?,
                "moe_routed_gate",
            )?,
            up: self.private_f32_buffer(
                checked_len(top_k, inter_dim, "moe routed up")?,
                "moe_routed_up",
            )?,
            hidden: self.private_f32_buffer(
                checked_len(top_k, inter_dim, "moe routed hidden")?,
                "moe_routed_hidden",
            )?,
            down: self.private_f32_buffer(
                checked_len(top_k, out_dim, "moe routed down")?,
                "moe_routed_down",
            )?,
        })
    }

    /// Alloue (ou récupère mémoïsés) les scratch du MoE shared mono-token.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si une taille déborde ou si l'allocation Metal échoue.
    pub(super) fn allocate_moe_shared_scratch(
        &self,
        top_k: usize,
        expert_count: usize,
        inter_dim: usize,
        out_dim: usize,
        shared_inter_dim: usize,
    ) -> Result<MoeSharedScratch> {
        Ok(MoeSharedScratch {
            router: self.private_f32_buffer(expert_count, "moe_shared_router_logits")?,
            indices: self.private_u32_buffer(top_k, "moe_shared_indices")?,
            scores: self.private_f32_buffer(top_k, "moe_shared_scores")?,
            gate: self.private_f32_buffer(
                checked_len(top_k, inter_dim, "moe shared gate")?,
                "moe_shared_gate",
            )?,
            up: self.private_f32_buffer(
                checked_len(top_k, inter_dim, "moe shared up")?,
                "moe_shared_up",
            )?,
            hidden: self.private_f32_buffer(
                checked_len(top_k, inter_dim, "moe shared hidden")?,
                "moe_shared_hidden",
            )?,
            down: self.private_f32_buffer(
                checked_len(top_k, out_dim, "moe shared down")?,
                "moe_shared_down",
            )?,
            shared_gate: self.private_f32_buffer(1, "moe_shared_gate_scalar")?,
            shared_proj_gate: self.private_f32_buffer(shared_inter_dim, "moe_shared_proj_gate")?,
            shared_up: self.private_f32_buffer(shared_inter_dim, "moe_shared_proj_up")?,
            shared_hidden: self.private_f32_buffer(shared_inter_dim, "moe_shared_proj_hidden")?,
            shared_down: self.private_f32_buffer(out_dim, "moe_shared_proj_down")?,
        })
    }

    /// Alloue (ou récupère mémoïsés) les scratch du MoE shared batché.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si une taille déborde ou si l'allocation Metal échoue.
    pub(super) fn allocate_moe_shared_rows_scratch(
        &self,
        rows: usize,
        top_k: usize,
        expert_count: usize,
        inter_dim: usize,
        out_dim: usize,
        shared_inter_dim: usize,
    ) -> Result<MoeSharedRowsScratch> {
        let total_topk = checked_len(rows, top_k, "moe shared rows topk total")?;
        Ok(MoeSharedRowsScratch {
            router: self.private_f32_buffer(
                checked_len(rows, expert_count, "moe shared rows router")?,
                "moe_shared_rows_router_logits",
            )?,
            indices: self.private_u32_buffer(total_topk, "moe_shared_rows_indices")?,
            scores: self.private_f32_buffer(total_topk, "moe_shared_rows_scores")?,
            gate: self.private_f32_buffer(
                checked_len(total_topk, inter_dim, "moe shared rows gate")?,
                "moe_shared_rows_gate",
            )?,
            up: self.private_f32_buffer(
                checked_len(total_topk, inter_dim, "moe shared rows up")?,
                "moe_shared_rows_up",
            )?,
            hidden: self.private_f32_buffer(
                checked_len(total_topk, inter_dim, "moe shared rows hidden")?,
                "moe_shared_rows_hidden",
            )?,
            down: self.private_f32_buffer(
                checked_len(total_topk, out_dim, "moe shared rows down")?,
                "moe_shared_rows_down",
            )?,
            shared_gate: self.private_f32_buffer(rows, "moe_shared_rows_gate_scalar")?,
            shared_proj_gate: self.private_f32_buffer(
                checked_len(rows, shared_inter_dim, "moe shared rows proj gate")?,
                "moe_shared_rows_proj_gate",
            )?,
            shared_up: self.private_f32_buffer(
                checked_len(rows, shared_inter_dim, "moe shared rows proj up")?,
                "moe_shared_rows_proj_up",
            )?,
            shared_hidden: self.private_f32_buffer(
                checked_len(rows, shared_inter_dim, "moe shared rows proj hidden")?,
                "moe_shared_rows_proj_hidden",
            )?,
            shared_down: self.private_f32_buffer(
                checked_len(rows, out_dim, "moe shared rows proj down")?,
                "moe_shared_rows_proj_down",
            )?,
        })
    }

    /// Valide la cohérence dimensionnelle routeur/experts d'un MoE routed-only.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si `top_k` est invalide ou si une dimension diverge.
    pub(super) fn check_moe_routed_buffer_shapes(
        &self,
        in_dim: usize,
        weights: &MetalMoeRoutedWeights,
        top_k: usize,
    ) -> Result<MoeRoutedBufferShape> {
        let expert_count = self.linear_weight_out_dim(&weights.router);
        ensure_valid_top_k(top_k, expert_count)?;
        if expert_count != weights.stacked.gate.experts {
            return Err(InferError::Dimension(format!(
                "routeur MoE routed experts={expert_count}, poids experts={}",
                weights.stacked.gate.experts
            )));
        }
        if in_dim != weights.stacked.gate.in_dim || in_dim != weights.stacked.up.in_dim {
            return Err(InferError::Dimension(format!(
                "MoE routed in_dim={in_dim}, gate_in={}, up_in={}",
                weights.stacked.gate.in_dim, weights.stacked.up.in_dim
            )));
        }
        if weights.stacked.gate.out_dim != weights.stacked.up.out_dim
            || weights.stacked.down.in_dim != weights.stacked.gate.out_dim
        {
            return Err(InferError::Dimension(format!(
                "MoE routed inter dims gate={} up={} down_in={}",
                weights.stacked.gate.out_dim,
                weights.stacked.up.out_dim,
                weights.stacked.down.in_dim
            )));
        }
        Ok(MoeRoutedBufferShape {
            expert_count,
            inter_dim: weights.stacked.gate.out_dim,
            out_dim: weights.stacked.down.out_dim,
        })
    }

    /// Valide routeur, experts empilés et shared expert ; renvoie les dimensions.
    ///
    /// Le gate shared doit sortir un scalaire (out_dim = 1) et le down du
    /// shared expert doit rejoindre `out_dim` des experts routés.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si `top_k` est invalide ou si une dimension diverge.
    pub(super) fn check_moe_shared_buffer_shapes(
        &self,
        in_dim: usize,
        weights: &MetalMoeSharedWeights,
        top_k: usize,
    ) -> Result<MoeSharedBufferShape> {
        let expert_count = self.linear_weight_out_dim(&weights.router);
        ensure_valid_top_k(top_k, expert_count)?;
        if expert_count != weights.stacked.gate.experts {
            return Err(InferError::Dimension(format!(
                "routeur MoE shared experts={expert_count}, poids experts={}",
                weights.stacked.gate.experts
            )));
        }
        if in_dim != weights.stacked.gate.in_dim || in_dim != weights.stacked.up.in_dim {
            return Err(InferError::Dimension(format!(
                "MoE shared in_dim={in_dim}, gate_in={}, up_in={}",
                weights.stacked.gate.in_dim, weights.stacked.up.in_dim
            )));
        }
        if weights.stacked.gate.out_dim != weights.stacked.up.out_dim
            || weights.stacked.down.in_dim != weights.stacked.gate.out_dim
        {
            return Err(InferError::Dimension(format!(
                "MoE shared inter dims gate={} up={} down_in={}",
                weights.stacked.gate.out_dim,
                weights.stacked.up.out_dim,
                weights.stacked.down.in_dim
            )));
        }
        let shared_gate_out_dim = self.linear_weight_out_dim(&weights.shared_gate);
        if shared_gate_out_dim != 1 {
            return Err(InferError::Dimension(format!(
                "shared gate sort {shared_gate_out_dim}, attendu 1"
            )));
        }
        let shared_inter_dim = self.linear_weight_out_dim(&weights.shared_gate_proj);
        let shared_up_dim = self.linear_weight_out_dim(&weights.shared_up_proj);
        let shared_down_dim = self.linear_weight_out_dim(&weights.shared_down_proj);
        let shared_down_in_dim = self.linear_weight_in_dim(&weights.shared_down_proj);
        if shared_inter_dim != shared_up_dim || shared_inter_dim != shared_down_in_dim {
            return Err(InferError::Dimension(format!(
                "shared expert dims gate={shared_inter_dim}, up={shared_up_dim}, down_in={shared_down_in_dim}"
            )));
        }
        if shared_down_dim != weights.stacked.down.out_dim {
            return Err(InferError::Dimension(format!(
                "shared expert out={shared_down_dim}, MoE out={}",
                weights.stacked.down.out_dim
            )));
        }
        Ok(MoeSharedBufferShape {
            expert_count,
            inter_dim: weights.stacked.gate.out_dim,
            out_dim: weights.stacked.down.out_dim,
            shared_inter_dim,
        })
    }

    /// Encode le MoE routé seul dans un encoder partagé.
    ///
    /// Reprend le préfixe routed de [`Self::encode_moe_shared_buffers`] sans
    /// shared-expert. `residual = Some(buf)` fusionne `buf + MoE` via
    /// `weighted_sum_add`, comme le tail résident shared.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si une dimension est incompatible ou si un dispatch échoue.
    #[expect(
        clippy::too_many_arguments,
        reason = "mirror des poids MoE routed résolus (routeur + experts)"
    )]
    pub(crate) fn encode_moe_routed_buffers(
        &self,
        encoder: &ComputeCommandEncoderRef,
        owned_buffers: &mut Vec<Buffer>,
        input_buffer: &BufferRef,
        residual: Option<&BufferRef>,
        output_buffer: &BufferRef,
        in_dim: usize,
        weights: &MetalMoeRoutedWeights,
        top_k: usize,
    ) -> Result<()> {
        let shape = self.check_moe_routed_buffer_shapes(in_dim, weights, top_k)?;
        let scratch = self.allocate_moe_routed_scratch(
            top_k,
            shape.expert_count,
            shape.inter_dim,
            shape.out_dim,
        )?;

        let router_out_dim = self.encode_matmul_weight_buffers(
            encoder,
            input_buffer,
            1,
            in_dim,
            &weights.router,
            &scratch.router,
            false,
        )?;
        if router_out_dim != shape.expert_count {
            return Err(InferError::Dimension(format!(
                "routeur MoE routed sort {router_out_dim}, attendu {}",
                shape.expert_count
            )));
        }
        self.encode_topk_softmax(
            encoder,
            owned_buffers,
            &scratch.router,
            &scratch.indices,
            &scratch.scores,
            shape.expert_count,
            top_k,
        )?;
        if !self.encode_gather_gate_up_swiglu(
            encoder,
            owned_buffers,
            input_buffer,
            1,
            &weights.stacked.gate,
            &weights.stacked.up,
            &scratch.indices,
            top_k,
            &scratch.hidden,
        )? {
            self.encode_gather_matmul(
                encoder,
                owned_buffers,
                input_buffer,
                1,
                &weights.stacked.gate,
                &scratch.indices,
                top_k,
                &scratch.gate,
            )?;
            self.encode_gather_matmul(
                encoder,
                owned_buffers,
                input_buffer,
                1,
                &weights.stacked.up,
                &scratch.indices,
                top_k,
                &scratch.up,
            )?;
            self.encode_swiglu(
                encoder,
                owned_buffers,
                &scratch.gate,
                &scratch.up,
                &scratch.hidden,
                checked_len(top_k, shape.inter_dim, "moe routed swiglu")?,
            )?;
        }
        self.encode_gather_matmul(
            encoder,
            owned_buffers,
            &scratch.hidden,
            top_k,
            &weights.stacked.down,
            &scratch.indices,
            top_k,
            &scratch.down,
        )?;
        match residual {
            Some(residual_buffer) => self.encode_weighted_sum_add_topk(
                encoder,
                owned_buffers,
                &scratch.down,
                &scratch.scores,
                residual_buffer,
                output_buffer,
                top_k,
                shape.out_dim,
            )?,
            None => self.encode_weighted_sum_topk(
                encoder,
                owned_buffers,
                &scratch.down,
                &scratch.scores,
                output_buffer,
                top_k,
                shape.out_dim,
            )?,
        }
        Ok(())
    }

    /// Encode le MoE routé + shared-expert dans un encoder PARTAGÉ, résultat dans
    /// `output_buffer` (RÉSIDENT, pas de commit/readback). `residual = Some(buf)`
    /// fusionne `buf + MoE` via `weighted_sum_add` (résiduel `attention_state` de
    /// l'orchestration 1c) ; `None` reproduit le per-op (`weighted_sum`).
    ///
    /// Cœur extrait de [`Self::moe_gated_router_topk_shared`] (désormais wrapper,
    /// per-op bit-identique). Réutilisé pour chaîner une couche sans commit.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si une dimension est incompatible ou si un dispatch échoue.
    #[expect(
        clippy::too_many_arguments,
        reason = "mirror des poids MoE (routeur + experts + shared-expert)"
    )]
    pub(crate) fn encode_moe_shared(
        &self,
        encoder: &ComputeCommandEncoderRef,
        owned_buffers: &mut Vec<Buffer>,
        input_buffer: &BufferRef,
        residual: Option<&BufferRef>,
        output_buffer: &BufferRef,
        in_dim: usize,
        router: &Linear,
        experts: &[GatedMlp],
        top_k: usize,
        shared_expert: &GatedMlp,
        shared_gate: &Linear,
    ) -> Result<()> {
        ensure_biasless(router, "router")?;
        ensure_biasless(shared_gate, "shared_gate")?;
        let expert_count = linear_out_dim(router.weight())?;
        ensure_valid_top_k(top_k, expert_count)?;
        if expert_count != experts.len() {
            return Err(InferError::Dimension(format!(
                "routeur MoE shared experts={expert_count}, poids experts={}",
                experts.len()
            )));
        }
        let stacked = self.stacked_moe_buffers(experts)?;
        if in_dim != stacked.gate.in_dim || in_dim != stacked.up.in_dim {
            return Err(InferError::Dimension(format!(
                "MoE shared in_dim={in_dim}, gate_in={}, up_in={}",
                stacked.gate.in_dim, stacked.up.in_dim
            )));
        }
        if stacked.gate.out_dim != stacked.up.out_dim || stacked.down.in_dim != stacked.gate.out_dim
        {
            return Err(InferError::Dimension(format!(
                "MoE shared inter dims gate={} up={} down_in={}",
                stacked.gate.out_dim, stacked.up.out_dim, stacked.down.in_dim
            )));
        }
        let (shared_gate_proj, shared_up_proj, shared_down_proj) = shared_expert.projections();
        ensure_biasless(shared_gate_proj, "shared_gate_proj")?;
        ensure_biasless(shared_up_proj, "shared_up_proj")?;
        ensure_biasless(shared_down_proj, "shared_down_proj")?;
        let shared_gate_out_dim = linear_out_dim(shared_gate.weight())?;
        if shared_gate_out_dim != 1 {
            return Err(InferError::Dimension(format!(
                "shared gate sort {shared_gate_out_dim}, attendu 1"
            )));
        }
        let shared_inter_dim = linear_out_dim(shared_gate_proj.weight())?;
        let shared_up_dim = linear_out_dim(shared_up_proj.weight())?;
        let shared_down_dim = linear_out_dim(shared_down_proj.weight())?;
        let shared_down_in_dim = linear_in_dim(shared_down_proj.weight())?;
        if shared_inter_dim != shared_up_dim || shared_inter_dim != shared_down_in_dim {
            return Err(InferError::Dimension(format!(
                "shared expert dims gate={shared_inter_dim}, up={shared_up_dim}, down_in={shared_down_in_dim}"
            )));
        }
        if shared_down_dim != stacked.down.out_dim {
            return Err(InferError::Dimension(format!(
                "shared expert out={shared_down_dim}, MoE out={}",
                stacked.down.out_dim
            )));
        }
        let inter_dim = stacked.gate.out_dim;
        let out_dim = stacked.down.out_dim;

        let scratch = self.allocate_moe_shared_scratch(
            top_k,
            expert_count,
            inter_dim,
            out_dim,
            shared_inter_dim,
        )?;

        let router_out_dim = self.encode_matmul_weight(
            encoder,
            owned_buffers,
            input_buffer,
            1,
            in_dim,
            router.weight(),
            &scratch.router,
        )?;
        if router_out_dim != expert_count {
            return Err(InferError::Dimension(format!(
                "routeur MoE shared sort {router_out_dim}, attendu {expert_count}"
            )));
        }
        self.encode_topk_softmax(
            encoder,
            owned_buffers,
            &scratch.router,
            &scratch.indices,
            &scratch.scores,
            expert_count,
            top_k,
        )?;
        if !self.encode_gather_gate_up_swiglu(
            encoder,
            owned_buffers,
            input_buffer,
            1,
            &stacked.gate,
            &stacked.up,
            &scratch.indices,
            top_k,
            &scratch.hidden,
        )? {
            self.encode_gather_matmul(
                encoder,
                owned_buffers,
                input_buffer,
                1,
                &stacked.gate,
                &scratch.indices,
                top_k,
                &scratch.gate,
            )?;
            self.encode_gather_matmul(
                encoder,
                owned_buffers,
                input_buffer,
                1,
                &stacked.up,
                &scratch.indices,
                top_k,
                &scratch.up,
            )?;
            self.encode_swiglu(
                encoder,
                owned_buffers,
                &scratch.gate,
                &scratch.up,
                &scratch.hidden,
                checked_len(top_k, inter_dim, "moe shared swiglu")?,
            )?;
        }
        self.encode_gather_matmul(
            encoder,
            owned_buffers,
            &scratch.hidden,
            top_k,
            &stacked.down,
            &scratch.indices,
            top_k,
            &scratch.down,
        )?;
        // Résiduel optionnel : Some → attention_state + MoE (fusion 1c), None → MoE seul (per-op).
        match residual {
            Some(residual_buffer) => self.encode_weighted_sum_add_topk(
                encoder,
                owned_buffers,
                &scratch.down,
                &scratch.scores,
                residual_buffer,
                output_buffer,
                top_k,
                out_dim,
            )?,
            None => self.encode_weighted_sum_topk(
                encoder,
                owned_buffers,
                &scratch.down,
                &scratch.scores,
                output_buffer,
                top_k,
                out_dim,
            )?,
        }
        let projected_gate_dim = self.encode_matmul_weight(
            encoder,
            owned_buffers,
            input_buffer,
            1,
            in_dim,
            shared_gate.weight(),
            &scratch.shared_gate,
        )?;
        if projected_gate_dim != 1 {
            return Err(InferError::Dimension(format!(
                "shared gate Metal sort {projected_gate_dim}, attendu 1"
            )));
        }
        // Shared-expert : gate_proj + up_proj + swiglu fondus en 1 dispatch (tranche
        // 3, kill-switch `RETI_RUST_FUSED_SHARED_GATE_UP=0`) — attaque le poste dispatch-bound
        // du MoE (6 micro-QMV série du shared-expert). Sinon le chemin 2 QMV + swiglu
        // (résultat identique ; le fusé est ==CPU/tolérance, cf. test colocalisé).
        let fused_shared = can_fuse_shared_gate_up_weights(shared_gate_proj, shared_up_proj)
            && self.encode_gate_up_swiglu_fast(
                encoder,
                input_buffer,
                shared_gate_proj,
                shared_up_proj,
                &scratch.shared_hidden,
                in_dim,
            )?;
        if !fused_shared {
            let projected_shared_gate_dim = self.encode_matmul_weight(
                encoder,
                owned_buffers,
                input_buffer,
                1,
                in_dim,
                shared_gate_proj.weight(),
                &scratch.shared_proj_gate,
            )?;
            let projected_shared_up_dim = self.encode_matmul_weight(
                encoder,
                owned_buffers,
                input_buffer,
                1,
                in_dim,
                shared_up_proj.weight(),
                &scratch.shared_up,
            )?;
            if projected_shared_gate_dim != shared_inter_dim
                || projected_shared_up_dim != shared_inter_dim
            {
                return Err(InferError::Dimension(format!(
                    "shared expert Metal proj gate={projected_shared_gate_dim}, up={projected_shared_up_dim}, attendu {shared_inter_dim}"
                )));
            }
            self.encode_swiglu(
                encoder,
                owned_buffers,
                &scratch.shared_proj_gate,
                &scratch.shared_up,
                &scratch.shared_hidden,
                shared_inter_dim,
            )?;
        }
        let projected_shared_down_dim = self.encode_matmul_weight(
            encoder,
            owned_buffers,
            &scratch.shared_hidden,
            1,
            shared_inter_dim,
            shared_down_proj.weight(),
            &scratch.shared_down,
        )?;
        if projected_shared_down_dim != out_dim {
            return Err(InferError::Dimension(format!(
                "shared expert Metal down sort {projected_shared_down_dim}, attendu {out_dim}"
            )));
        }
        self.encode_add_sigmoid_scaled(
            encoder,
            &scratch.shared_down,
            &scratch.shared_gate,
            output_buffer,
            out_dim,
        )?;
        Ok(())
    }

    /// Encode le MoE routed + shared complet (poids résolus) dans un encoder partagé.
    ///
    /// Chemin chaud du decode résident. Par défaut le réordonnancement
    /// routed‖shared s'applique (voir doc de module) ; sinon déroulé
    /// séquentiel routed puis shared, même résultat. `residual = Some(buf)`
    /// fusionne `buf + MoE` dans le combine (résiduel attention de
    /// l'orchestration 1c) ; `None` reproduit le per-op (`weighted_sum` puis
    /// ajout shared).
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si une dimension est incompatible ou si un dispatch échoue.
    #[expect(
        clippy::too_many_arguments,
        reason = "mirror des poids MoE résolus (routeur + experts + shared-expert)"
    )]
    pub(crate) fn encode_moe_shared_buffers(
        &self,
        encoder: &ComputeCommandEncoderRef,
        owned_buffers: &mut Vec<Buffer>,
        input_buffer: &BufferRef,
        residual: Option<&BufferRef>,
        output_buffer: &BufferRef,
        in_dim: usize,
        weights: &MetalMoeSharedWeights,
        top_k: usize,
    ) -> Result<()> {
        let shape = self.check_moe_shared_buffer_shapes(in_dim, weights, top_k)?;
        let expert_count = shape.expert_count;
        let inter_dim = shape.inter_dim;
        let out_dim = shape.out_dim;
        let shared_inter_dim = shape.shared_inter_dim;

        let scratch = self.allocate_moe_shared_scratch(
            top_k,
            expert_count,
            inter_dim,
            out_dim,
            shared_inter_dim,
        )?;

        let router_out_dim = self.encode_matmul_weight_buffers(
            encoder,
            input_buffer,
            1,
            in_dim,
            &weights.router,
            &scratch.router,
            false,
        )?;
        if router_out_dim != expert_count {
            return Err(InferError::Dimension(format!(
                "routeur MoE shared sort {router_out_dim}, attendu {expert_count}"
            )));
        }
        self.encode_topk_softmax(
            encoder,
            owned_buffers,
            &scratch.router,
            &scratch.indices,
            &scratch.scores,
            expert_count,
            top_k,
        )?;
        if moe_shared_route_overlap_enabled() {
            // L'overlap routed‖shared suspend les barrières par-dispatch pour laisser
            // les deux experts se recouvrir. C'est un no-op en SÉRIE (l'encodeur série
            // sérialise déjà), mais sous l'encodeur CONCURRENT le recouvrement corrompt
            // le MoE (sortie charabia) malgré des buffers disjoints et des barrières de
            // phase explicites — sémantique subtile du concurrent. On NE suspend donc
            // qu'en série ; sous concurrent on garde les barrières ACTIVES (routed/shared
            // sérialisés mais corrects). Le chemin série prod reste byte-identique.
            let barrier_guard = if resident_concurrent_enabled() {
                install_dispatch_barrier_scope()
            } else {
                suspend_dispatch_barrier_scope()
            };
            let routed_fused = self.encode_gather_gate_up_swiglu(
                encoder,
                owned_buffers,
                input_buffer,
                1,
                &weights.stacked.gate,
                &weights.stacked.up,
                &scratch.indices,
                top_k,
                &scratch.hidden,
            )?;
            if !routed_fused {
                self.encode_gather_matmul(
                    encoder,
                    owned_buffers,
                    input_buffer,
                    1,
                    &weights.stacked.gate,
                    &scratch.indices,
                    top_k,
                    &scratch.gate,
                )?;
                self.encode_gather_matmul(
                    encoder,
                    owned_buffers,
                    input_buffer,
                    1,
                    &weights.stacked.up,
                    &scratch.indices,
                    top_k,
                    &scratch.up,
                )?;
            }
            // Échelle de fusions shared (du plus au moins fusionné) : qmv+gate
            // scalaire, puis gate+up+SwiGLU+gate, puis gate+up+SwiGLU seul —
            // chaque étage supprime des micro-dispatches série du shared-expert ;
            // à défaut, dépliage exact en QMV séparés.
            let shared_gate_qmv_fused = fused_shared_gate_qmv_u8_enabled()
                && self.encode_qmv_plus_shared_gate_fast_buffers(
                    encoder,
                    input_buffer,
                    &weights.shared_gate_proj,
                    &weights.shared_gate,
                    &scratch.shared_proj_gate,
                    &scratch.shared_gate,
                    in_dim,
                )?;
            let fused_shared_with_gate = !shared_gate_qmv_fused
                && fused_shared_gate_scalar_u8_enabled()
                && self.encode_gate_up_swiglu_shared_gate_fast_buffers(
                    encoder,
                    input_buffer,
                    &weights.shared_gate_proj,
                    &weights.shared_up_proj,
                    &weights.shared_gate,
                    &scratch.shared_hidden,
                    &scratch.shared_gate,
                    in_dim,
                )?;
            if !shared_gate_qmv_fused && !fused_shared_with_gate {
                let projected_gate_dim = self.encode_matmul_weight_buffers(
                    encoder,
                    input_buffer,
                    1,
                    in_dim,
                    &weights.shared_gate,
                    &scratch.shared_gate,
                    false,
                )?;
                if projected_gate_dim != 1 {
                    return Err(InferError::Dimension(format!(
                        "shared gate Metal sort {projected_gate_dim}, attendu 1"
                    )));
                }
            }
            let fused_shared = !shared_gate_qmv_fused
                && (fused_shared_with_gate
                    || (can_fuse_shared_gate_up_buffers(
                        &weights.shared_gate_proj,
                        &weights.shared_up_proj,
                    ) && self.encode_gate_up_swiglu_fast_buffers(
                        encoder,
                        input_buffer,
                        &weights.shared_gate_proj,
                        &weights.shared_up_proj,
                        &scratch.shared_hidden,
                        in_dim,
                    )?));
            if !fused_shared {
                // La fusion qmv+gate a déjà écrit gate_proj : on reprend la
                // dimension validée par check_moe_shared_buffer_shapes au lieu
                // de ré-encoder la projection.
                let projected_shared_gate_dim = if shared_gate_qmv_fused {
                    shared_inter_dim
                } else {
                    self.encode_matmul_weight_buffers(
                        encoder,
                        input_buffer,
                        1,
                        in_dim,
                        &weights.shared_gate_proj,
                        &scratch.shared_proj_gate,
                        false,
                    )?
                };
                let projected_shared_up_dim = self.encode_matmul_weight_buffers(
                    encoder,
                    input_buffer,
                    1,
                    in_dim,
                    &weights.shared_up_proj,
                    &scratch.shared_up,
                    false,
                )?;
                if projected_shared_gate_dim != shared_inter_dim
                    || projected_shared_up_dim != shared_inter_dim
                {
                    return Err(InferError::Dimension(format!(
                        "shared expert Metal proj gate={projected_shared_gate_dim}, up={projected_shared_up_dim}, attendu {shared_inter_dim}"
                    )));
                }
            }
            memory_barrier_buffers(encoder);
            if !routed_fused {
                self.encode_swiglu(
                    encoder,
                    owned_buffers,
                    &scratch.gate,
                    &scratch.up,
                    &scratch.hidden,
                    checked_len(top_k, inter_dim, "moe shared swiglu")?,
                )?;
            }
            if !fused_shared {
                self.encode_swiglu(
                    encoder,
                    owned_buffers,
                    &scratch.shared_proj_gate,
                    &scratch.shared_up,
                    &scratch.shared_hidden,
                    shared_inter_dim,
                )?;
            }
            memory_barrier_buffers(encoder);
            // Tail fusé : gather-down pondéré + résiduel + shared en UN dispatch.
            // Le kernel lit shared_down et le gate scalaire, donc le down du
            // shared DOIT être encodé (et barré) avant — d'où l'inversion
            // shared-down-avant-routed-down et le early-return si le kernel accepte.
            let mut shared_down_done = false;
            if residual.is_some() && fused_moe_down_weighted_u8_enabled() {
                let projected_shared_down_dim = self.encode_matmul_weight_buffers(
                    encoder,
                    &scratch.shared_hidden,
                    1,
                    shared_inter_dim,
                    &weights.shared_down_proj,
                    &scratch.shared_down,
                    false,
                )?;
                if projected_shared_down_dim != out_dim {
                    return Err(InferError::Dimension(format!(
                        "shared expert Metal down sort {projected_shared_down_dim}, attendu {out_dim}"
                    )));
                }
                shared_down_done = true;
                memory_barrier_buffers(encoder);
                if let Some(residual_buffer) = residual {
                    if self.encode_gather_down_weighted_shared_u8_gs64(
                        encoder,
                        &scratch.hidden,
                        &weights.stacked.down,
                        &scratch.indices,
                        &scratch.scores,
                        residual_buffer,
                        &scratch.shared_down,
                        &scratch.shared_gate,
                        output_buffer,
                        top_k,
                    )? {
                        drop(barrier_guard);
                        return Ok(());
                    }
                }
            }
            self.encode_gather_matmul(
                encoder,
                owned_buffers,
                &scratch.hidden,
                top_k,
                &weights.stacked.down,
                &scratch.indices,
                top_k,
                &scratch.down,
            )?;
            if !shared_down_done {
                let projected_shared_down_dim = self.encode_matmul_weight_buffers(
                    encoder,
                    &scratch.shared_hidden,
                    1,
                    shared_inter_dim,
                    &weights.shared_down_proj,
                    &scratch.shared_down,
                    false,
                )?;
                if projected_shared_down_dim != out_dim {
                    return Err(InferError::Dimension(format!(
                        "shared expert Metal down sort {projected_shared_down_dim}, attendu {out_dim}"
                    )));
                }
            }
            memory_barrier_buffers(encoder);
            drop(barrier_guard);
            match residual {
                Some(residual_buffer) => self.encode_weighted_sum_add_shared_topk(
                    encoder,
                    &scratch.down,
                    &scratch.scores,
                    residual_buffer,
                    &scratch.shared_down,
                    &scratch.shared_gate,
                    output_buffer,
                    top_k,
                    out_dim,
                )?,
                None => {
                    self.encode_weighted_sum_topk(
                        encoder,
                        owned_buffers,
                        &scratch.down,
                        &scratch.scores,
                        output_buffer,
                        top_k,
                        out_dim,
                    )?;
                    self.encode_add_sigmoid_scaled(
                        encoder,
                        &scratch.shared_down,
                        &scratch.shared_gate,
                        output_buffer,
                        out_dim,
                    )?;
                }
            }
            return Ok(());
        }
        if !self.encode_gather_gate_up_swiglu(
            encoder,
            owned_buffers,
            input_buffer,
            1,
            &weights.stacked.gate,
            &weights.stacked.up,
            &scratch.indices,
            top_k,
            &scratch.hidden,
        )? {
            self.encode_gather_matmul(
                encoder,
                owned_buffers,
                input_buffer,
                1,
                &weights.stacked.gate,
                &scratch.indices,
                top_k,
                &scratch.gate,
            )?;
            self.encode_gather_matmul(
                encoder,
                owned_buffers,
                input_buffer,
                1,
                &weights.stacked.up,
                &scratch.indices,
                top_k,
                &scratch.up,
            )?;
            self.encode_swiglu(
                encoder,
                owned_buffers,
                &scratch.gate,
                &scratch.up,
                &scratch.hidden,
                checked_len(top_k, inter_dim, "moe shared swiglu")?,
            )?;
        }
        self.encode_gather_matmul(
            encoder,
            owned_buffers,
            &scratch.hidden,
            top_k,
            &weights.stacked.down,
            &scratch.indices,
            top_k,
            &scratch.down,
        )?;
        // Même échelle de fusions shared que le chemin recouvert (voir plus haut) :
        // qmv+gate scalaire, puis gate+up+SwiGLU+gate, puis gate+up+SwiGLU seul.
        let shared_gate_qmv_fused = fused_shared_gate_qmv_u8_enabled()
            && self.encode_qmv_plus_shared_gate_fast_buffers(
                encoder,
                input_buffer,
                &weights.shared_gate_proj,
                &weights.shared_gate,
                &scratch.shared_proj_gate,
                &scratch.shared_gate,
                in_dim,
            )?;
        let fused_shared_with_gate = !shared_gate_qmv_fused
            && fused_shared_gate_scalar_u8_enabled()
            && self.encode_gate_up_swiglu_shared_gate_fast_buffers(
                encoder,
                input_buffer,
                &weights.shared_gate_proj,
                &weights.shared_up_proj,
                &weights.shared_gate,
                &scratch.shared_hidden,
                &scratch.shared_gate,
                in_dim,
            )?;
        if !shared_gate_qmv_fused && !fused_shared_with_gate {
            let projected_gate_dim = self.encode_matmul_weight_buffers(
                encoder,
                input_buffer,
                1,
                in_dim,
                &weights.shared_gate,
                &scratch.shared_gate,
                false,
            )?;
            if projected_gate_dim != 1 {
                return Err(InferError::Dimension(format!(
                    "shared gate Metal sort {projected_gate_dim}, attendu 1"
                )));
            }
        }
        let fused_shared = !shared_gate_qmv_fused
            && (fused_shared_with_gate
                || (can_fuse_shared_gate_up_buffers(
                    &weights.shared_gate_proj,
                    &weights.shared_up_proj,
                ) && self.encode_gate_up_swiglu_fast_buffers(
                    encoder,
                    input_buffer,
                    &weights.shared_gate_proj,
                    &weights.shared_up_proj,
                    &scratch.shared_hidden,
                    in_dim,
                )?));
        if !fused_shared {
            // La fusion qmv+gate a déjà écrit gate_proj : on reprend la
            // dimension validée par check_moe_shared_buffer_shapes au lieu
            // de ré-encoder la projection.
            let projected_shared_gate_dim = if shared_gate_qmv_fused {
                shared_inter_dim
            } else {
                self.encode_matmul_weight_buffers(
                    encoder,
                    input_buffer,
                    1,
                    in_dim,
                    &weights.shared_gate_proj,
                    &scratch.shared_proj_gate,
                    false,
                )?
            };
            let projected_shared_up_dim = self.encode_matmul_weight_buffers(
                encoder,
                input_buffer,
                1,
                in_dim,
                &weights.shared_up_proj,
                &scratch.shared_up,
                false,
            )?;
            if projected_shared_gate_dim != shared_inter_dim
                || projected_shared_up_dim != shared_inter_dim
            {
                return Err(InferError::Dimension(format!(
                    "shared expert Metal proj gate={projected_shared_gate_dim}, up={projected_shared_up_dim}, attendu {shared_inter_dim}"
                )));
            }
            self.encode_swiglu(
                encoder,
                owned_buffers,
                &scratch.shared_proj_gate,
                &scratch.shared_up,
                &scratch.shared_hidden,
                shared_inter_dim,
            )?;
        }
        let projected_shared_down_dim = self.encode_matmul_weight_buffers(
            encoder,
            &scratch.shared_hidden,
            1,
            shared_inter_dim,
            &weights.shared_down_proj,
            &scratch.shared_down,
            false,
        )?;
        if projected_shared_down_dim != out_dim {
            return Err(InferError::Dimension(format!(
                "shared expert Metal down sort {projected_shared_down_dim}, attendu {out_dim}"
            )));
        }
        match residual {
            Some(residual_buffer) => self.encode_weighted_sum_add_shared_topk(
                encoder,
                &scratch.down,
                &scratch.scores,
                residual_buffer,
                &scratch.shared_down,
                &scratch.shared_gate,
                output_buffer,
                top_k,
                out_dim,
            )?,
            None => {
                self.encode_weighted_sum_topk(
                    encoder,
                    owned_buffers,
                    &scratch.down,
                    &scratch.scores,
                    output_buffer,
                    top_k,
                    out_dim,
                )?;
                self.encode_add_sigmoid_scaled(
                    encoder,
                    &scratch.shared_down,
                    &scratch.shared_gate,
                    output_buffer,
                    out_dim,
                )?;
            }
        }
        Ok(())
    }

    /// Encode le MoE shared pour `rows` tokens (prefill, light-batch).
    ///
    /// Variante batchée sans recouvrement ni fusions mono-token : gating par
    /// ligne (`topk_softmax_rows`), gathers sur `rows × top_k` slots, combine
    /// groupé par ligne puis ajout shared par ligne. `rows == 1` délègue au
    /// chemin mono-token.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si `rows == 0`, si une dimension est incompatible
    /// ou si un dispatch échoue.
    #[expect(
        clippy::too_many_arguments,
        reason = "mirror batché des poids MoE (routeur + experts + shared-expert)"
    )]
    pub(crate) fn encode_moe_shared_buffers_rows(
        &self,
        encoder: &ComputeCommandEncoderRef,
        owned_buffers: &mut Vec<Buffer>,
        input_buffer: &BufferRef,
        residual: Option<&BufferRef>,
        output_buffer: &BufferRef,
        rows: usize,
        in_dim: usize,
        weights: &MetalMoeSharedWeights,
        top_k: usize,
    ) -> Result<()> {
        if rows == 0 {
            return Err(InferError::Dimension(
                "MoE shared rows: batch vide".to_string(),
            ));
        }
        if rows == 1 {
            // Le chemin mono-token porte les fusions et le recouvrement
            // routed‖shared : y déléguer garde rows=1 identique au decode résident.
            return self.encode_moe_shared_buffers(
                encoder,
                owned_buffers,
                input_buffer,
                residual,
                output_buffer,
                in_dim,
                weights,
                top_k,
            );
        }

        let shape = self.check_moe_shared_buffer_shapes(in_dim, weights, top_k)?;
        let expert_count = shape.expert_count;
        let inter_dim = shape.inter_dim;
        let out_dim = shape.out_dim;
        let shared_inter_dim = shape.shared_inter_dim;
        trace_dispatch_path("moe_shared_rows", rows, out_dim, in_dim);
        let total_topk = checked_len(rows, top_k, "moe shared rows topk total")?;
        let scratch = self.allocate_moe_shared_rows_scratch(
            rows,
            top_k,
            expert_count,
            inter_dim,
            out_dim,
            shared_inter_dim,
        )?;

        let router_out_dim = self.encode_matmul_weight_buffers(
            encoder,
            input_buffer,
            rows,
            in_dim,
            &weights.router,
            &scratch.router,
            false,
        )?;
        if router_out_dim != expert_count {
            return Err(InferError::Dimension(format!(
                "routeur MoE shared rows sort {router_out_dim}, attendu {expert_count}"
            )));
        }
        self.encode_topk_softmax_rows(
            encoder,
            &scratch.router,
            &scratch.indices,
            &scratch.scores,
            rows,
            expert_count,
            top_k,
        )?;

        if !self.encode_gather_gate_up_swiglu(
            encoder,
            owned_buffers,
            input_buffer,
            rows,
            &weights.stacked.gate,
            &weights.stacked.up,
            &scratch.indices,
            total_topk,
            &scratch.hidden,
        )? {
            self.encode_gather_matmul(
                encoder,
                owned_buffers,
                input_buffer,
                rows,
                &weights.stacked.gate,
                &scratch.indices,
                total_topk,
                &scratch.gate,
            )?;
            self.encode_gather_matmul(
                encoder,
                owned_buffers,
                input_buffer,
                rows,
                &weights.stacked.up,
                &scratch.indices,
                total_topk,
                &scratch.up,
            )?;
            self.encode_swiglu(
                encoder,
                owned_buffers,
                &scratch.gate,
                &scratch.up,
                &scratch.hidden,
                checked_len(total_topk, inter_dim, "moe shared rows swiglu")?,
            )?;
        }
        self.encode_gather_matmul(
            encoder,
            owned_buffers,
            &scratch.hidden,
            total_topk,
            &weights.stacked.down,
            &scratch.indices,
            total_topk,
            &scratch.down,
        )?;

        let projected_gate_dim = self.encode_matmul_weight_buffers(
            encoder,
            input_buffer,
            rows,
            in_dim,
            &weights.shared_gate,
            &scratch.shared_gate,
            false,
        )?;
        if projected_gate_dim != 1 {
            return Err(InferError::Dimension(format!(
                "shared gate rows Metal sort {projected_gate_dim}, attendu 1"
            )));
        }
        let projected_shared_gate_dim = self.encode_matmul_weight_buffers(
            encoder,
            input_buffer,
            rows,
            in_dim,
            &weights.shared_gate_proj,
            &scratch.shared_proj_gate,
            false,
        )?;
        let projected_shared_up_dim = self.encode_matmul_weight_buffers(
            encoder,
            input_buffer,
            rows,
            in_dim,
            &weights.shared_up_proj,
            &scratch.shared_up,
            false,
        )?;
        if projected_shared_gate_dim != shared_inter_dim
            || projected_shared_up_dim != shared_inter_dim
        {
            return Err(InferError::Dimension(format!(
                "shared expert rows Metal proj gate={projected_shared_gate_dim}, up={projected_shared_up_dim}, attendu {shared_inter_dim}"
            )));
        }
        self.encode_swiglu(
            encoder,
            owned_buffers,
            &scratch.shared_proj_gate,
            &scratch.shared_up,
            &scratch.shared_hidden,
            checked_len(rows, shared_inter_dim, "moe shared rows shared swiglu")?,
        )?;
        let projected_shared_down_dim = self.encode_matmul_weight_buffers(
            encoder,
            &scratch.shared_hidden,
            rows,
            shared_inter_dim,
            &weights.shared_down_proj,
            &scratch.shared_down,
            false,
        )?;
        if projected_shared_down_dim != out_dim {
            return Err(InferError::Dimension(format!(
                "shared expert rows Metal down sort {projected_shared_down_dim}, attendu {out_dim}"
            )));
        }

        match residual {
            Some(residual_buffer) => self.encode_weighted_sum_add_grouped_topk(
                encoder,
                owned_buffers,
                &scratch.down,
                &scratch.scores,
                residual_buffer,
                output_buffer,
                rows,
                top_k,
                out_dim,
            )?,
            None => self.encode_weighted_sum_grouped_topk(
                encoder,
                owned_buffers,
                &scratch.down,
                &scratch.scores,
                output_buffer,
                rows,
                top_k,
                out_dim,
            )?,
        }
        self.encode_add_sigmoid_scaled_rows(
            encoder,
            &scratch.shared_down,
            &scratch.shared_gate,
            output_buffer,
            rows,
            out_dim,
        )?;
        Ok(())
    }

    /// Microbenche les segments du MoE shared (route, gate/up, down, tails).
    ///
    /// Mesure chaque segment isolé (warmup 8, 64 itérations, un command
    /// buffer par itération) sur une route figée par le premier top-k ;
    /// `overhead_ms` (coût commit+wait à vide) est soustrait pour la
    /// colonne « pur » du rapport.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si `input` n'est pas batch=1, si une dimension est
    /// incompatible ou si un encodage échoue.
    #[expect(
        clippy::too_many_arguments,
        reason = "microbench MoE: dimensions, poids et overhead restent explicites"
    )]
    pub(crate) fn profile_moe_shared_segments(
        &self,
        input: &Tensor,
        router: &Linear,
        experts: &[GatedMlp],
        top_k: usize,
        shared_expert: &GatedMlp,
        shared_gate: &Linear,
        overhead_ms: f64,
    ) -> Result<String> {
        let (batch, in_dim) = input.as_matrix()?;
        if batch != 1 {
            return Err(InferError::Dimension(format!(
                "MoE split attend batch=1, reçu {batch}"
            )));
        }
        let weights =
            self.resolve_moe_shared_weights(router, experts, shared_expert, shared_gate)?;
        let shape = self.check_moe_shared_buffer_shapes(in_dim, &weights, top_k)?;
        let scratch = self.allocate_moe_shared_scratch(
            top_k,
            shape.expert_count,
            shape.inter_dim,
            shape.out_dim,
            shape.shared_inter_dim,
        )?;
        let input_buffer = self.upload_f32_buffer(input.data(), "moe_split_input")?;
        let output_buffer = self.new_f32_buffer(shape.out_dim, "moe_split_output")?;
        let residual_zeros = vec![0.0_f32; shape.out_dim];
        let residual_buffer = self.upload_f32_buffer(&residual_zeros, "moe_split_residual")?;
        let iters = 64_u32;
        let warmup = 8_u32;

        let route_topk = profile_moe_segment(self, warmup, iters, |encoder, owned| {
            let router_out_dim = self.encode_matmul_weight_buffers(
                encoder,
                &input_buffer,
                1,
                in_dim,
                &weights.router,
                &scratch.router,
                false,
            )?;
            if router_out_dim != shape.expert_count {
                return Err(InferError::Dimension(format!(
                    "split routeur sort {router_out_dim}, attendu {}",
                    shape.expert_count
                )));
            }
            self.encode_topk_softmax(
                encoder,
                owned,
                &scratch.router,
                &scratch.indices,
                &scratch.scores,
                shape.expert_count,
                top_k,
            )
        })?;

        let routed_gate_up = profile_moe_segment(self, warmup, iters, |encoder, owned| {
            if self.encode_gather_gate_up_swiglu(
                encoder,
                owned,
                &input_buffer,
                1,
                &weights.stacked.gate,
                &weights.stacked.up,
                &scratch.indices,
                top_k,
                &scratch.hidden,
            )? {
                return Ok(());
            }
            self.encode_gather_matmul(
                encoder,
                owned,
                &input_buffer,
                1,
                &weights.stacked.gate,
                &scratch.indices,
                top_k,
                &scratch.gate,
            )?;
            self.encode_gather_matmul(
                encoder,
                owned,
                &input_buffer,
                1,
                &weights.stacked.up,
                &scratch.indices,
                top_k,
                &scratch.up,
            )?;
            self.encode_swiglu(
                encoder,
                owned,
                &scratch.gate,
                &scratch.up,
                &scratch.hidden,
                checked_len(top_k, shape.inter_dim, "split routed swiglu")?,
            )
        })?;

        let shared_gate_ms = profile_moe_segment(self, warmup, iters, |encoder, _owned| {
            let projected_gate_dim = self.encode_matmul_weight_buffers(
                encoder,
                &input_buffer,
                1,
                in_dim,
                &weights.shared_gate,
                &scratch.shared_gate,
                false,
            )?;
            if projected_gate_dim != 1 {
                return Err(InferError::Dimension(format!(
                    "split shared gate sort {projected_gate_dim}, attendu 1"
                )));
            }
            Ok(())
        })?;

        let shared_gate_up = profile_moe_segment(self, warmup, iters, |encoder, owned| {
            if can_fuse_shared_gate_up_buffers(&weights.shared_gate_proj, &weights.shared_up_proj)
                && self.encode_gate_up_swiglu_fast_buffers(
                    encoder,
                    &input_buffer,
                    &weights.shared_gate_proj,
                    &weights.shared_up_proj,
                    &scratch.shared_hidden,
                    in_dim,
                )?
            {
                return Ok(());
            }
            let projected_shared_gate_dim = self.encode_matmul_weight_buffers(
                encoder,
                &input_buffer,
                1,
                in_dim,
                &weights.shared_gate_proj,
                &scratch.shared_proj_gate,
                false,
            )?;
            let projected_shared_up_dim = self.encode_matmul_weight_buffers(
                encoder,
                &input_buffer,
                1,
                in_dim,
                &weights.shared_up_proj,
                &scratch.shared_up,
                false,
            )?;
            if projected_shared_gate_dim != shape.shared_inter_dim
                || projected_shared_up_dim != shape.shared_inter_dim
            {
                return Err(InferError::Dimension(format!(
                    "split shared expert gate={projected_shared_gate_dim}, up={projected_shared_up_dim}, attendu {}",
                    shape.shared_inter_dim
                )));
            }
            self.encode_swiglu(
                encoder,
                owned,
                &scratch.shared_proj_gate,
                &scratch.shared_up,
                &scratch.shared_hidden,
                shape.shared_inter_dim,
            )
        })?;

        let routed_down = profile_moe_segment(self, warmup, iters, |encoder, owned| {
            self.encode_gather_matmul(
                encoder,
                owned,
                &scratch.hidden,
                top_k,
                &weights.stacked.down,
                &scratch.indices,
                top_k,
                &scratch.down,
            )
        })?;

        let shared_down = profile_moe_segment(self, warmup, iters, |encoder, _owned| {
            let projected_shared_down_dim = self.encode_matmul_weight_buffers(
                encoder,
                &scratch.shared_hidden,
                1,
                shape.shared_inter_dim,
                &weights.shared_down_proj,
                &scratch.shared_down,
                false,
            )?;
            if projected_shared_down_dim != shape.out_dim {
                return Err(InferError::Dimension(format!(
                    "split shared down sort {projected_shared_down_dim}, attendu {}",
                    shape.out_dim
                )));
            }
            Ok(())
        })?;

        let tail_plain = profile_moe_segment(self, warmup, iters, |encoder, owned| {
            self.encode_weighted_sum_topk(
                encoder,
                owned,
                &scratch.down,
                &scratch.scores,
                &output_buffer,
                top_k,
                shape.out_dim,
            )?;
            self.encode_add_sigmoid_scaled(
                encoder,
                &scratch.shared_down,
                &scratch.shared_gate,
                &output_buffer,
                shape.out_dim,
            )
        })?;

        let tail_residual = profile_moe_segment(self, warmup, iters, |encoder, _owned| {
            self.encode_weighted_sum_add_shared_topk(
                encoder,
                &scratch.down,
                &scratch.scores,
                &residual_buffer,
                &scratch.shared_down,
                &scratch.shared_gate,
                &output_buffer,
                top_k,
                shape.out_dim,
            )
        })?;

        let pure = |segment_ms: f64| (segment_ms - overhead_ms).max(0.0);
        Ok(format!(
            "split MoE fixed-route ({iters} itér, ms+CB/pur): route {route_topk:.3}/{route_pure:.3}, \
             routed_gate_up {routed_gate_up:.3}/{routed_gate_up_pure:.3}, \
             shared_gate {shared_gate_ms:.3}/{shared_gate_pure:.3}, \
             shared_gate_up {shared_gate_up:.3}/{shared_gate_up_pure:.3}, \
             routed_down {routed_down:.3}/{routed_down_pure:.3}, \
             shared_down {shared_down:.3}/{shared_down_pure:.3}, \
             tail_plain {tail_plain:.3}/{tail_plain_pure:.3}, \
             tail_residual {tail_residual:.3}/{tail_residual_pure:.3}",
            route_pure = pure(route_topk),
            routed_gate_up_pure = pure(routed_gate_up),
            shared_gate_pure = pure(shared_gate_ms),
            shared_gate_up_pure = pure(shared_gate_up),
            routed_down_pure = pure(routed_down),
            shared_down_pure = pure(shared_down),
            tail_plain_pure = pure(tail_plain),
            tail_residual_pure = pure(tail_residual),
        ))
    }
}

/// Mesure la durée moyenne (ms) d'un segment encodé, commit+wait inclus.
///
/// # Errors
///
/// Renvoie une erreur si `iters == 0` ou si l'encodage/le commit échoue.
fn profile_moe_segment<F>(
    metal: &MetalExecutor,
    warmup: u32,
    iters: u32,
    mut encode: F,
) -> Result<f64>
where
    F: FnMut(&ComputeCommandEncoderRef, &mut Vec<Buffer>) -> Result<()>,
{
    if iters == 0 {
        return Err(InferError::Dimension("MoE split iters nul".to_string()));
    }
    for _ in 0..warmup {
        let command_buffer = metal.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let encoder_guard = EncoderEndGuard::new(encoder);
        let mut owned = Vec::new();
        encode(encoder, &mut owned)?;
        encoder_guard.end();
        commit_and_wait(command_buffer)?;
    }
    let started = std::time::Instant::now();
    for _ in 0..iters {
        let command_buffer = metal.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let encoder_guard = EncoderEndGuard::new(encoder);
        let mut owned = Vec::new();
        encode(encoder, &mut owned)?;
        encoder_guard.end();
        commit_and_wait(command_buffer)?;
    }
    Ok(started.elapsed().as_secs_f64() * 1000.0 / f64::from(iters))
}
