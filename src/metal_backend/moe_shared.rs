//! Encodage Metal du MoE avec expert partagé.

use super::*;

struct MoeSharedBufferShape {
    expert_count: usize,
    inter_dim: usize,
    out_dim: usize,
    shared_inter_dim: usize,
}

struct MoeSharedScratch {
    router: Buffer,
    indices: Buffer,
    scores: Buffer,
    gate: Buffer,
    up: Buffer,
    hidden: Buffer,
    down: Buffer,
    shared_gate: Buffer,
    shared_proj_gate: Buffer,
    shared_up: Buffer,
    shared_hidden: Buffer,
    shared_down: Buffer,
}

struct MoeRoutedBufferShape {
    expert_count: usize,
    inter_dim: usize,
    out_dim: usize,
}

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

    fn allocate_moe_shared_scratch(
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

    fn check_moe_routed_buffer_shapes(
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

    fn check_moe_shared_buffer_shapes(
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
        let fused_shared = fused_shared_gate_up_enabled()
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
            let barrier_guard = suspend_dispatch_barrier_scope();
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
            let fused_shared = fused_shared_gate_up_enabled()
                && self.encode_gate_up_swiglu_fast_buffers(
                    encoder,
                    input_buffer,
                    &weights.shared_gate_proj,
                    &weights.shared_up_proj,
                    &scratch.shared_hidden,
                    in_dim,
                )?;
            if !fused_shared {
                let projected_shared_gate_dim = self.encode_matmul_weight_buffers(
                    encoder,
                    input_buffer,
                    1,
                    in_dim,
                    &weights.shared_gate_proj,
                    &scratch.shared_proj_gate,
                    false,
                )?;
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
            memory_barrier_buffers(encoder);
            drop(barrier_guard);
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
            self.encode_add_sigmoid_scaled(
                encoder,
                &scratch.shared_down,
                &scratch.shared_gate,
                output_buffer,
                out_dim,
            )?;
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
        let fused_shared = fused_shared_gate_up_enabled()
            && self.encode_gate_up_swiglu_fast_buffers(
                encoder,
                input_buffer,
                &weights.shared_gate_proj,
                &weights.shared_up_proj,
                &scratch.shared_hidden,
                in_dim,
            )?;
        if !fused_shared {
            let projected_shared_gate_dim = self.encode_matmul_weight_buffers(
                encoder,
                input_buffer,
                1,
                in_dim,
                &weights.shared_gate_proj,
                &scratch.shared_proj_gate,
                false,
            )?;
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
        self.encode_add_sigmoid_scaled(
            encoder,
            &scratch.shared_down,
            &scratch.shared_gate,
            output_buffer,
            out_dim,
        )?;
        Ok(())
    }
}
