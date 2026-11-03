//! Encodage Metal du MoE routed avec expert partagé.

use super::*;

mod phases;

struct MoeSharedBufferEncodeContext<'a> {
    encoder: &'a ComputeCommandEncoderRef,
    owned_buffers: &'a mut Vec<Buffer>,
    input_buffer: &'a BufferRef,
    residual: Option<&'a BufferRef>,
    output_buffer: &'a BufferRef,
    in_dim: usize,
    weights: &'a MetalMoeSharedWeights,
    top_k: usize,
    shape: MoeSharedBufferShape,
    scratch: MoeSharedScratch,
}

impl MoeSharedBufferEncodeContext<'_> {
    fn routed_phase(&mut self) -> MoeRoutedSharedPhase<'_> {
        MoeRoutedSharedPhase {
            encoder: self.encoder,
            owned_buffers: &mut *self.owned_buffers,
            input_buffer: self.input_buffer,
            rows: 1,
            stacked: &self.weights.stacked,
            indices: &self.scratch.indices,
            gate: &self.scratch.gate,
            up: &self.scratch.up,
            hidden: &self.scratch.hidden,
            down: &self.scratch.down,
            slots: self.top_k,
            inter_dim: self.shape.inter_dim,
            swiglu_label: "moe shared swiglu",
        }
    }
}

impl MetalExecutor {
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
        let shape = self.check_moe_shared_linear_shapes(
            in_dim,
            router,
            experts,
            top_k,
            shared_expert,
            shared_gate,
        )?;

        let scratch = self.allocate_moe_shared_scratch(
            top_k,
            shape.expert_count,
            shape.inter_dim,
            shape.out_dim,
            shape.shared_inter_dim,
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
        if router_out_dim != shape.expert_count {
            return Err(InferError::Dimension(format!(
                "routeur MoE shared sort {router_out_dim}, attendu {}",
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
        let mut routed_phase = MoeRoutedSharedPhase {
            encoder,
            owned_buffers,
            input_buffer,
            rows: 1,
            stacked: &shape.stacked,
            indices: &scratch.indices,
            gate: &scratch.gate,
            up: &scratch.up,
            hidden: &scratch.hidden,
            down: &scratch.down,
            slots: top_k,
            inter_dim: shape.inter_dim,
            swiglu_label: "moe shared swiglu",
        };
        self.encode_moe_shared_routed_phase(&mut routed_phase)?;
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
        self.encode_moe_shared_linear_expert_phase(
            encoder,
            owned_buffers,
            input_buffer,
            output_buffer,
            in_dim,
            shared_gate,
            &shape,
            &scratch,
        )
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
        let scratch = self.allocate_moe_shared_scratch(
            top_k,
            shape.expert_count,
            shape.inter_dim,
            shape.out_dim,
            shape.shared_inter_dim,
        )?;
        let mut context = MoeSharedBufferEncodeContext {
            encoder,
            owned_buffers,
            input_buffer,
            residual,
            output_buffer,
            in_dim,
            weights,
            top_k,
            shape,
            scratch,
        };

        self.encode_moe_shared_route_phase(&mut context)?;
        if moe_shared_route_overlap_enabled() {
            self.encode_moe_shared_overlap_phases(&mut context)
        } else {
            self.encode_moe_shared_sequential_phases(&mut context)
        }
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

        let mut routed_phase = MoeRoutedSharedPhase {
            encoder,
            owned_buffers,
            input_buffer,
            rows,
            stacked: &weights.stacked,
            indices: &scratch.indices,
            gate: &scratch.gate,
            up: &scratch.up,
            hidden: &scratch.hidden,
            down: &scratch.down,
            slots: total_topk,
            inter_dim,
            swiglu_label: "moe shared rows swiglu",
        };
        self.encode_moe_shared_routed_phase(&mut routed_phase)?;

        self.encode_moe_shared_rows_expert_phase(
            encoder,
            owned_buffers,
            input_buffer,
            rows,
            in_dim,
            weights,
            &scratch,
            shared_inter_dim,
            out_dim,
        )?;
        self.encode_moe_shared_rows_combine_phase(
            encoder,
            owned_buffers,
            residual,
            output_buffer,
            rows,
            top_k,
            out_dim,
            &scratch,
        )
    }

    /// Queue MoE shared de la vérification résidente en lignes (`rows` petits).
    ///
    /// Reprend la structure du mono-ligne au lieu du chemin batché générique
    /// du prefill : top-k parallèle par ligne, gate+up+SwiGLU partagé fusionné
    /// (jusqu'à [`crate::runtime_flags::moe_verify_shared_fused_max_rows`]
    /// lignes) et combine résiduel + routé + partagé en un dispatch, soit
    /// 8 dispatches par couche au lieu de 11. Niveaux de
    /// [`crate::runtime_flags::moe_verify_tail_level`] : 0 délègue à
    /// [`Self::encode_moe_shared_buffers_rows`]. `rows == 1` délègue au
    /// mono-ligne.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si `rows == 0`, si une dimension est incompatible
    /// ou si un dispatch échoue.
    #[expect(
        clippy::too_many_arguments,
        reason = "mirror batché des poids MoE (routeur + experts + shared-expert)"
    )]
    pub(crate) fn encode_moe_shared_buffers_verify_rows(
        &self,
        encoder: &ComputeCommandEncoderRef,
        owned_buffers: &mut Vec<Buffer>,
        input_buffer: &BufferRef,
        residual: &BufferRef,
        output_buffer: &BufferRef,
        rows: usize,
        in_dim: usize,
        weights: &MetalMoeSharedWeights,
        top_k: usize,
    ) -> Result<()> {
        let level = crate::runtime_flags::moe_verify_tail_level();
        if rows <= 1 || level == 0 {
            return self.encode_moe_shared_buffers_rows(
                encoder,
                owned_buffers,
                input_buffer,
                Some(residual),
                output_buffer,
                rows,
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
        trace_dispatch_path("moe_shared_verify_rows", rows, out_dim, in_dim);
        let total_topk = checked_len(rows, top_k, "moe shared verify rows topk total")?;
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
                "routeur MoE shared verify rows sort {router_out_dim}, attendu {expert_count}"
            )));
        }
        self.encode_topk_softmax_verify_rows(
            encoder,
            &scratch.router,
            &scratch.indices,
            &scratch.scores,
            rows,
            expert_count,
            top_k,
        )?;
        if level == 1 {
            // Ablation : top-k seul, le reste suit le chemin batché générique.
            let mut routed_phase = MoeRoutedSharedPhase {
                encoder,
                owned_buffers,
                input_buffer,
                rows,
                stacked: &weights.stacked,
                indices: &scratch.indices,
                gate: &scratch.gate,
                up: &scratch.up,
                hidden: &scratch.hidden,
                down: &scratch.down,
                slots: total_topk,
                inter_dim,
                swiglu_label: "moe shared verify rows swiglu",
            };
            self.encode_moe_shared_routed_phase(&mut routed_phase)?;
            self.encode_moe_shared_rows_expert_phase(
                encoder,
                owned_buffers,
                input_buffer,
                rows,
                in_dim,
                weights,
                &scratch,
                shared_inter_dim,
                out_dim,
            )?;
            return self.encode_moe_shared_rows_combine_phase(
                encoder,
                owned_buffers,
                Some(residual),
                output_buffer,
                rows,
                top_k,
                out_dim,
                &scratch,
            );
        }

        // Ordre du mono : gate/up routé, gate scalaire et gate/up partagés,
        // puis down routé, down partagé, combine.
        let mut routed_phase = MoeRoutedSharedPhase {
            encoder,
            owned_buffers,
            input_buffer,
            rows,
            stacked: &weights.stacked,
            indices: &scratch.indices,
            gate: &scratch.gate,
            up: &scratch.up,
            hidden: &scratch.hidden,
            down: &scratch.down,
            slots: total_topk,
            inter_dim,
            swiglu_label: "moe shared verify rows swiglu",
        };
        let routed_fused = self.encode_moe_shared_routed_gate_up_phase(&mut routed_phase)?;
        if !routed_fused {
            self.encode_swiglu(
                encoder,
                owned_buffers,
                &scratch.gate,
                &scratch.up,
                &scratch.hidden,
                checked_len(total_topk, inter_dim, "moe shared verify rows swiglu")?,
            )?;
        }
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
                "shared gate verify rows Metal sort {projected_gate_dim}, attendu 1"
            )));
        }
        let shared_fused = rows <= crate::runtime_flags::moe_verify_shared_fused_max_rows()
            && can_fuse_shared_gate_up_buffers(&weights.shared_gate_proj, &weights.shared_up_proj)
            && self.encode_gate_up_swiglu_fast_buffers_rows(
                encoder,
                input_buffer,
                &weights.shared_gate_proj,
                &weights.shared_up_proj,
                &scratch.shared_hidden,
                rows,
                in_dim,
            )?;
        if !shared_fused {
            self.encode_moe_shared_rows_gate_up(
                encoder,
                owned_buffers,
                input_buffer,
                rows,
                in_dim,
                weights,
                &scratch,
                shared_inter_dim,
            )?;
        }
        let mut routed_phase = MoeRoutedSharedPhase {
            encoder,
            owned_buffers,
            input_buffer,
            rows,
            stacked: &weights.stacked,
            indices: &scratch.indices,
            gate: &scratch.gate,
            up: &scratch.up,
            hidden: &scratch.hidden,
            down: &scratch.down,
            slots: total_topk,
            inter_dim,
            swiglu_label: "moe shared verify rows swiglu",
        };
        self.encode_moe_shared_routed_down_phase(&mut routed_phase)?;
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
                "shared expert verify rows down sort {projected_shared_down_dim}, attendu {out_dim}"
            )));
        }
        self.encode_weighted_sum_add_grouped_topk_shared(
            encoder,
            &scratch.down,
            &scratch.scores,
            residual,
            Some((&scratch.shared_down, &scratch.shared_gate)),
            output_buffer,
            rows,
            top_k,
            out_dim,
        )
    }
}
