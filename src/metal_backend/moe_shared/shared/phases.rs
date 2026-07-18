//! Phases internes des encodeurs MoE avec expert partagé.

use super::*;

impl MetalExecutor {
    /// Encode le shared expert à partir des poids linéaires non résolus.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si une dimension diverge ou si un dispatch échoue.
    #[expect(
        clippy::too_many_arguments,
        reason = "phase shared: poids linéaires, buffers et dimensions restent explicites"
    )]
    pub(super) fn encode_moe_shared_linear_expert_phase(
        &self,
        encoder: &ComputeCommandEncoderRef,
        owned_buffers: &mut Vec<Buffer>,
        input_buffer: &BufferRef,
        output_buffer: &BufferRef,
        in_dim: usize,
        shared_gate: &Linear,
        shape: &MoeSharedLinearShape<'_>,
        scratch: &MoeSharedScratch,
    ) -> Result<()> {
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
        let fused_shared =
            can_fuse_shared_gate_up_weights(shape.shared_gate_proj, shape.shared_up_proj)
                && self.encode_gate_up_swiglu_fast(
                    encoder,
                    input_buffer,
                    shape.shared_gate_proj,
                    shape.shared_up_proj,
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
                shape.shared_gate_proj.weight(),
                &scratch.shared_proj_gate,
            )?;
            let projected_shared_up_dim = self.encode_matmul_weight(
                encoder,
                owned_buffers,
                input_buffer,
                1,
                in_dim,
                shape.shared_up_proj.weight(),
                &scratch.shared_up,
            )?;
            if projected_shared_gate_dim != shape.shared_inter_dim
                || projected_shared_up_dim != shape.shared_inter_dim
            {
                return Err(InferError::Dimension(format!(
                    "shared expert Metal proj gate={projected_shared_gate_dim}, up={projected_shared_up_dim}, attendu {}",
                    shape.shared_inter_dim
                )));
            }
            self.encode_swiglu(
                encoder,
                owned_buffers,
                &scratch.shared_proj_gate,
                &scratch.shared_up,
                &scratch.shared_hidden,
                shape.shared_inter_dim,
            )?;
        }
        let projected_shared_down_dim = self.encode_matmul_weight(
            encoder,
            owned_buffers,
            &scratch.shared_hidden,
            1,
            shape.shared_inter_dim,
            shape.shared_down_proj.weight(),
            &scratch.shared_down,
        )?;
        if projected_shared_down_dim != shape.out_dim {
            return Err(InferError::Dimension(format!(
                "shared expert Metal down sort {projected_shared_down_dim}, attendu {}",
                shape.out_dim
            )));
        }
        self.encode_add_sigmoid_scaled(
            encoder,
            &scratch.shared_down,
            &scratch.shared_gate,
            output_buffer,
            shape.out_dim,
        )?;
        Ok(())
    }

    /// Encode le routeur et le top-k du chemin shared résident.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le routeur ou le top-k échoue.
    pub(super) fn encode_moe_shared_route_phase(
        &self,
        context: &mut MoeSharedBufferEncodeContext<'_>,
    ) -> Result<()> {
        let router_out_dim = self.encode_matmul_weight_buffers(
            context.encoder,
            context.input_buffer,
            1,
            context.in_dim,
            &context.weights.router,
            &context.scratch.router,
            false,
        )?;
        if router_out_dim != context.shape.expert_count {
            return Err(InferError::Dimension(format!(
                "routeur MoE shared sort {router_out_dim}, attendu {}",
                context.shape.expert_count
            )));
        }
        self.encode_topk_softmax(
            context.encoder,
            context.owned_buffers,
            &context.scratch.router,
            &context.scratch.indices,
            &context.scratch.scores,
            context.shape.expert_count,
            context.top_k,
        )
    }

    /// Encode les phases recouvertes routed et shared.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si une phase ou une fusion échoue.
    pub(super) fn encode_moe_shared_overlap_phases(
        &self,
        context: &mut MoeSharedBufferEncodeContext<'_>,
    ) -> Result<()> {
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
        let routed_fused =
            self.encode_moe_shared_routed_gate_up_phase(&mut context.routed_phase())?;
        let shared_fused = self.encode_moe_shared_gate_up_phase(context)?;
        memory_barrier_buffers(context.encoder);
        if !routed_fused {
            self.encode_swiglu(
                context.encoder,
                context.owned_buffers,
                &context.scratch.gate,
                &context.scratch.up,
                &context.scratch.hidden,
                checked_len(context.top_k, context.shape.inter_dim, "moe shared swiglu")?,
            )?;
        }
        if !shared_fused {
            self.encode_swiglu(
                context.encoder,
                context.owned_buffers,
                &context.scratch.shared_proj_gate,
                &context.scratch.shared_up,
                &context.scratch.shared_hidden,
                context.shape.shared_inter_dim,
            )?;
        }
        memory_barrier_buffers(context.encoder);
        if self.encode_moe_shared_overlap_down_phase(context)? {
            drop(barrier_guard);
            return Ok(());
        }
        memory_barrier_buffers(context.encoder);
        drop(barrier_guard);
        self.encode_moe_shared_combine_phase(context)
    }

    /// Encode les phases séquentielles routed puis shared.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si une projection ou une activation échoue.
    pub(super) fn encode_moe_shared_sequential_phases(
        &self,
        context: &mut MoeSharedBufferEncodeContext<'_>,
    ) -> Result<()> {
        self.encode_moe_shared_routed_phase(&mut context.routed_phase())?;
        let shared_fused = self.encode_moe_shared_gate_up_phase(context)?;
        if !shared_fused {
            self.encode_swiglu(
                context.encoder,
                context.owned_buffers,
                &context.scratch.shared_proj_gate,
                &context.scratch.shared_up,
                &context.scratch.shared_hidden,
                context.shape.shared_inter_dim,
            )?;
        }
        self.encode_moe_shared_down_phase(context)?;
        self.encode_moe_shared_combine_phase(context)
    }

    /// Encode les projections gate/up du shared expert résident.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si une projection ou une fusion échoue.
    pub(super) fn encode_moe_shared_gate_up_phase(
        &self,
        context: &mut MoeSharedBufferEncodeContext<'_>,
    ) -> Result<bool> {
        // Échelle de fusions shared (du plus au moins fusionné) : qmv+gate
        // scalaire, puis gate+up+SwiGLU+gate, puis gate+up+SwiGLU seul —
        // chaque étage supprime des micro-dispatches série du shared-expert ;
        // à défaut, dépliage exact en QMV séparés.
        let shared_gate_qmv_fused = fused_shared_gate_qmv_u8_enabled()
            && self.encode_qmv_plus_shared_gate_fast_buffers(
                context.encoder,
                context.input_buffer,
                &context.weights.shared_gate_proj,
                &context.weights.shared_gate,
                &context.scratch.shared_proj_gate,
                &context.scratch.shared_gate,
                context.in_dim,
            )?;
        let fused_shared_with_gate = !shared_gate_qmv_fused
            && fused_shared_gate_scalar_u8_enabled()
            && self.encode_gate_up_swiglu_shared_gate_fast_buffers(
                context.encoder,
                context.input_buffer,
                &context.weights.shared_gate_proj,
                &context.weights.shared_up_proj,
                &context.weights.shared_gate,
                &context.scratch.shared_hidden,
                &context.scratch.shared_gate,
                context.in_dim,
            )?;
        if !shared_gate_qmv_fused && !fused_shared_with_gate {
            let projected_gate_dim = self.encode_matmul_weight_buffers(
                context.encoder,
                context.input_buffer,
                1,
                context.in_dim,
                &context.weights.shared_gate,
                &context.scratch.shared_gate,
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
                    &context.weights.shared_gate_proj,
                    &context.weights.shared_up_proj,
                ) && self.encode_gate_up_swiglu_fast_buffers(
                    context.encoder,
                    context.input_buffer,
                    &context.weights.shared_gate_proj,
                    &context.weights.shared_up_proj,
                    &context.scratch.shared_hidden,
                    context.in_dim,
                )?));
        if !fused_shared {
            // La fusion qmv+gate a déjà écrit gate_proj : on reprend la
            // dimension validée par check_moe_shared_buffer_shapes au lieu
            // de ré-encoder la projection.
            let projected_shared_gate_dim = if shared_gate_qmv_fused {
                context.shape.shared_inter_dim
            } else {
                self.encode_matmul_weight_buffers(
                    context.encoder,
                    context.input_buffer,
                    1,
                    context.in_dim,
                    &context.weights.shared_gate_proj,
                    &context.scratch.shared_proj_gate,
                    false,
                )?
            };
            let projected_shared_up_dim = self.encode_matmul_weight_buffers(
                context.encoder,
                context.input_buffer,
                1,
                context.in_dim,
                &context.weights.shared_up_proj,
                &context.scratch.shared_up,
                false,
            )?;
            if projected_shared_gate_dim != context.shape.shared_inter_dim
                || projected_shared_up_dim != context.shape.shared_inter_dim
            {
                return Err(InferError::Dimension(format!(
                    "shared expert Metal proj gate={projected_shared_gate_dim}, up={projected_shared_up_dim}, attendu {}",
                    context.shape.shared_inter_dim
                )));
            }
        }
        Ok(fused_shared)
    }

    /// Encode les projections down du chemin recouvert.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si une projection ou la fusion finale échoue.
    pub(super) fn encode_moe_shared_overlap_down_phase(
        &self,
        context: &mut MoeSharedBufferEncodeContext<'_>,
    ) -> Result<bool> {
        // Tail fusé : gather-down pondéré + résiduel + shared en UN dispatch.
        // Le kernel lit shared_down et le gate scalaire, donc le down du
        // shared DOIT être encodé (et barré) avant — d'où l'inversion
        // shared-down-avant-routed-down et le early-return si le kernel accepte.
        let mut shared_down_done = false;
        if context.residual.is_some() && fused_moe_down_weighted_u8_enabled() {
            self.encode_moe_shared_down_phase(context)?;
            shared_down_done = true;
            memory_barrier_buffers(context.encoder);
            if let Some(residual_buffer) = context.residual {
                if self.encode_gather_down_weighted_shared_u8_gs64(
                    context.encoder,
                    &context.scratch.hidden,
                    &context.weights.stacked.down,
                    &context.scratch.indices,
                    &context.scratch.scores,
                    residual_buffer,
                    &context.scratch.shared_down,
                    &context.scratch.shared_gate,
                    context.output_buffer,
                    context.top_k,
                )? {
                    return Ok(true);
                }
            }
        }
        self.encode_moe_shared_routed_down_phase(&mut context.routed_phase())?;
        if !shared_down_done {
            self.encode_moe_shared_down_phase(context)?;
        }
        Ok(false)
    }

    /// Encode la projection down du shared expert résident.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si la projection échoue ou si sa dimension diverge.
    pub(super) fn encode_moe_shared_down_phase(
        &self,
        context: &MoeSharedBufferEncodeContext<'_>,
    ) -> Result<()> {
        let projected_shared_down_dim = self.encode_matmul_weight_buffers(
            context.encoder,
            &context.scratch.shared_hidden,
            1,
            context.shape.shared_inter_dim,
            &context.weights.shared_down_proj,
            &context.scratch.shared_down,
            false,
        )?;
        if projected_shared_down_dim != context.shape.out_dim {
            return Err(InferError::Dimension(format!(
                "shared expert Metal down sort {projected_shared_down_dim}, attendu {}",
                context.shape.out_dim
            )));
        }
        Ok(())
    }

    /// Combine les experts routés, le résiduel et le shared expert.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si la réduction ou l'ajout shared échoue.
    pub(super) fn encode_moe_shared_combine_phase(
        &self,
        context: &mut MoeSharedBufferEncodeContext<'_>,
    ) -> Result<()> {
        match context.residual {
            Some(residual_buffer) => self.encode_weighted_sum_add_shared_topk(
                context.encoder,
                &context.scratch.down,
                &context.scratch.scores,
                residual_buffer,
                &context.scratch.shared_down,
                &context.scratch.shared_gate,
                context.output_buffer,
                context.top_k,
                context.shape.out_dim,
            ),
            None => {
                self.encode_weighted_sum_topk(
                    context.encoder,
                    context.owned_buffers,
                    &context.scratch.down,
                    &context.scratch.scores,
                    context.output_buffer,
                    context.top_k,
                    context.shape.out_dim,
                )?;
                self.encode_add_sigmoid_scaled(
                    context.encoder,
                    &context.scratch.shared_down,
                    &context.scratch.shared_gate,
                    context.output_buffer,
                    context.shape.out_dim,
                )
            }
        }
    }

    /// Encode le shared expert batché.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si une projection ou une activation échoue.
    #[expect(
        clippy::too_many_arguments,
        reason = "phase shared batchée: poids, buffers et dimensions restent explicites"
    )]
    pub(super) fn encode_moe_shared_rows_expert_phase(
        &self,
        encoder: &ComputeCommandEncoderRef,
        owned_buffers: &mut Vec<Buffer>,
        input_buffer: &BufferRef,
        rows: usize,
        in_dim: usize,
        weights: &MetalMoeSharedWeights,
        scratch: &MoeSharedRowsScratch,
        shared_inter_dim: usize,
        out_dim: usize,
    ) -> Result<()> {
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
        Ok(())
    }

    /// Combine les experts routés et shared du batch.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si la réduction ou l'ajout shared échoue.
    #[expect(
        clippy::too_many_arguments,
        reason = "combine batché: résiduel, scores et dimensions restent explicites"
    )]
    pub(super) fn encode_moe_shared_rows_combine_phase(
        &self,
        encoder: &ComputeCommandEncoderRef,
        owned_buffers: &mut Vec<Buffer>,
        residual: Option<&BufferRef>,
        output_buffer: &BufferRef,
        rows: usize,
        top_k: usize,
        out_dim: usize,
        scratch: &MoeSharedRowsScratch,
    ) -> Result<()> {
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
}
