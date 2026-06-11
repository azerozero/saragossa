//! Encodage Metal des couches full-attention en decode.

use super::attention_checks::TailMoeSharedShape;
use super::*;

const FULL_ATTN_PREFILL_TG_WIDTH: u64 = 256;

struct TailMoeSharedScratch {
    residual: Buffer,
    context: Buffer,
    norm: Buffer,
    o: Buffer,
    attention_state: Buffer,
    normed: Buffer,
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
    output: Buffer,
}

#[expect(
    clippy::too_many_arguments,
    reason = "wrappers d'encodage Metal: buffers, dimensions et offsets restent explicites"
)]
impl MetalExecutor {
    fn allocate_tail_moe_shared_scratch(
        &self,
        residual: &Tensor,
        context: &Tensor,
        shape: &TailMoeSharedShape<'_>,
        top_k: usize,
    ) -> Result<TailMoeSharedScratch> {
        let hidden_dim = shape.hidden_dim;
        let inter_dim = shape.inter_dim;
        let shared_inter_dim = shape.shared_inter_dim;
        Ok(TailMoeSharedScratch {
            residual: self.upload_f32_buffer(residual.data(), "tail_shared_residual")?,
            context: self.upload_f32_buffer(context.data(), "tail_shared_context")?,
            norm: self.cached_buffer_from_f32(shape.norm_weight.data(), "tail_shared_norm")?,
            o: self.private_f32_buffer(hidden_dim, "tail_shared_o")?,
            attention_state: self.private_f32_buffer(hidden_dim, "tail_shared_attention_state")?,
            normed: self.private_f32_buffer(hidden_dim, "tail_shared_normed")?,
            router: self.private_f32_buffer(shape.expert_count, "tail_shared_router_logits")?,
            indices: self.private_u32_buffer(top_k, "tail_shared_indices")?,
            scores: self.private_f32_buffer(top_k, "tail_shared_scores")?,
            gate: self.private_f32_buffer(
                checked_len(top_k, inter_dim, "tail shared gate")?,
                "tail_shared_gate",
            )?,
            up: self.private_f32_buffer(
                checked_len(top_k, inter_dim, "tail shared up")?,
                "tail_shared_up",
            )?,
            hidden: self.private_f32_buffer(
                checked_len(top_k, inter_dim, "tail shared hidden")?,
                "tail_shared_hidden",
            )?,
            down: self.private_f32_buffer(
                checked_len(top_k, hidden_dim, "tail shared down")?,
                "tail_shared_down",
            )?,
            shared_gate: self.private_f32_buffer(1, "tail_shared_gate_scalar")?,
            shared_proj_gate: self.private_f32_buffer(shared_inter_dim, "tail_shared_proj_gate")?,
            shared_up: self.private_f32_buffer(shared_inter_dim, "tail_shared_proj_up")?,
            shared_hidden: self.private_f32_buffer(shared_inter_dim, "tail_shared_proj_hidden")?,
            shared_down: self.private_f32_buffer(hidden_dim, "tail_shared_proj_down")?,
            output: self.new_f32_buffer(hidden_dim, "tail_shared_output")?,
        })
    }

    /// Fusionne la fin d'une couche Qwen MoE après le contexte d'attention.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si les projections ne sont pas biasless ou si les
    /// formes ne correspondent pas au chemin MoE empilé.
    pub(crate) fn full_attention_tail_moe(
        &self,
        residual: &Tensor,
        context: &Tensor,
        o_proj: &Linear,
        post_norm: &Tensor,
        router: &Linear,
        experts: &[GatedMlp],
        top_k: usize,
        eps: f32,
    ) -> Result<Tensor> {
        let shape = self
            .check_tail_moe_shapes(residual, context, o_proj, post_norm, router, experts, top_k)?;
        let batch = shape.batch;
        let hidden_dim = shape.hidden_dim;
        let context_dim = shape.context_dim;
        let norm_weight = shape.norm_weight;
        let expert_count = shape.expert_count;
        let inter_dim = shape.inter_dim;
        let stacked = shape.stacked;
        let residual_buffer = self.upload_f32_buffer(residual.data(), "tail_residual")?;
        let context_buffer = self.upload_f32_buffer(context.data(), "tail_context")?;
        let norm_weight_buffer = self.cached_buffer_from_f32(norm_weight.data(), "tail_norm")?;
        let o_buffer = self.private_f32_buffer(hidden_dim, "tail_o")?;
        let attention_state_buffer = self.private_f32_buffer(hidden_dim, "tail_attention_state")?;
        let normed_buffer = self.private_f32_buffer(hidden_dim, "tail_normed")?;
        let router_buffer = self.private_f32_buffer(expert_count, "tail_router_logits")?;
        let indices_buffer = self.private_u32_buffer(top_k, "tail_indices")?;
        let scores_buffer = self.private_f32_buffer(top_k, "tail_scores")?;
        let gate_buffer =
            self.private_f32_buffer(checked_len(top_k, inter_dim, "tail gate")?, "tail_gate")?;
        let up_buffer =
            self.private_f32_buffer(checked_len(top_k, inter_dim, "tail up")?, "tail_up")?;
        let hidden_buffer =
            self.private_f32_buffer(checked_len(top_k, inter_dim, "tail hidden")?, "tail_hidden")?;
        let down_buffer =
            self.private_f32_buffer(checked_len(top_k, hidden_dim, "tail down")?, "tail_down")?;
        let output_buffer = self.new_f32_buffer(hidden_dim, "tail_output")?;
        let mut owned_buffers = Vec::new();

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let encoder_guard = EncoderEndGuard::new(encoder);
        let projected_dim = self.encode_matmul_weight(
            encoder,
            &mut owned_buffers,
            &context_buffer,
            batch,
            context_dim,
            o_proj.weight(),
            &o_buffer,
        )?;
        if projected_dim != hidden_dim {
            return Err(InferError::Dimension(format!(
                "tail MoE o_proj sort {projected_dim}, attendu {hidden_dim}"
            )));
        }
        self.encode_add_rms_norm(
            encoder,
            &mut owned_buffers,
            &residual_buffer,
            &o_buffer,
            &norm_weight_buffer,
            &attention_state_buffer,
            &normed_buffer,
            hidden_dim,
            eps,
        )?;
        let router_out_dim = self.encode_matmul_weight(
            encoder,
            &mut owned_buffers,
            &normed_buffer,
            batch,
            hidden_dim,
            router.weight(),
            &router_buffer,
        )?;
        if router_out_dim != expert_count {
            return Err(InferError::Dimension(format!(
                "tail MoE routeur sort {router_out_dim}, attendu {expert_count}"
            )));
        }
        self.encode_topk_softmax(
            encoder,
            &mut owned_buffers,
            &router_buffer,
            &indices_buffer,
            &scores_buffer,
            expert_count,
            top_k,
        )?;
        if !self.encode_gather_gate_up_swiglu(
            encoder,
            &mut owned_buffers,
            &normed_buffer,
            1,
            &stacked.gate,
            &stacked.up,
            &indices_buffer,
            top_k,
            &hidden_buffer,
        )? {
            self.encode_gather_matmul(
                encoder,
                &mut owned_buffers,
                &normed_buffer,
                1,
                &stacked.gate,
                &indices_buffer,
                top_k,
                &gate_buffer,
            )?;
            self.encode_gather_matmul(
                encoder,
                &mut owned_buffers,
                &normed_buffer,
                1,
                &stacked.up,
                &indices_buffer,
                top_k,
                &up_buffer,
            )?;
            self.encode_swiglu(
                encoder,
                &mut owned_buffers,
                &gate_buffer,
                &up_buffer,
                &hidden_buffer,
                checked_len(top_k, inter_dim, "tail swiglu")?,
            )?;
        }
        self.encode_gather_matmul(
            encoder,
            &mut owned_buffers,
            &hidden_buffer,
            top_k,
            &stacked.down,
            &indices_buffer,
            top_k,
            &down_buffer,
        )?;
        self.encode_weighted_sum_add_topk(
            encoder,
            &mut owned_buffers,
            &down_buffer,
            &scores_buffer,
            &attention_state_buffer,
            &output_buffer,
            top_k,
            hidden_dim,
        )?;
        encoder_guard.end();
        commit_and_wait(command_buffer)?;

        let output = read_f32_buffer(&output_buffer, hidden_dim)?;
        Tensor::from_vec(vec![1, hidden_dim], output)
    }

    /// Fusionne la fin d'une couche Qwen MoE avec shared expert.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si les poids shared ou MoE ne correspondent pas aux
    /// formes attendues par le chemin Metal.
    pub(crate) fn full_attention_tail_moe_shared(
        &self,
        residual: &Tensor,
        context: &Tensor,
        o_proj: &Linear,
        post_norm: &Tensor,
        router: &Linear,
        experts: &[GatedMlp],
        top_k: usize,
        shared_expert: &GatedMlp,
        shared_gate: &Linear,
        eps: f32,
    ) -> Result<Tensor> {
        let shape = self.check_tail_moe_shared_shapes(
            residual,
            context,
            o_proj,
            post_norm,
            router,
            experts,
            top_k,
            shared_expert,
            shared_gate,
        )?;
        let batch = shape.batch;
        let hidden_dim = shape.hidden_dim;
        let context_dim = shape.context_dim;
        let expert_count = shape.expert_count;
        let inter_dim = shape.inter_dim;
        let shared_gate_proj = shape.shared_gate_proj;
        let shared_up_proj = shape.shared_up_proj;
        let shared_down_proj = shape.shared_down_proj;
        let shared_inter_dim = shape.shared_inter_dim;
        let scratch = self.allocate_tail_moe_shared_scratch(residual, context, &shape, top_k)?;
        let stacked = shape.stacked;
        let TailMoeSharedScratch {
            residual: residual_buffer,
            context: context_buffer,
            norm: norm_weight_buffer,
            o: o_buffer,
            attention_state: attention_state_buffer,
            normed: normed_buffer,
            router: router_buffer,
            indices: indices_buffer,
            scores: scores_buffer,
            gate: gate_buffer,
            up: up_buffer,
            hidden: hidden_buffer,
            down: down_buffer,
            shared_gate: shared_gate_buffer,
            shared_proj_gate: shared_proj_gate_buffer,
            shared_up: shared_up_buffer,
            shared_hidden: shared_hidden_buffer,
            shared_down: shared_down_buffer,
            output: output_buffer,
        } = scratch;
        let mut owned_buffers = Vec::new();

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let encoder_guard = EncoderEndGuard::new(encoder);
        let projected_dim = self.encode_matmul_weight(
            encoder,
            &mut owned_buffers,
            &context_buffer,
            batch,
            context_dim,
            o_proj.weight(),
            &o_buffer,
        )?;
        if projected_dim != hidden_dim {
            return Err(InferError::Dimension(format!(
                "tail MoE shared o_proj sort {projected_dim}, attendu {hidden_dim}"
            )));
        }
        self.encode_add_rms_norm(
            encoder,
            &mut owned_buffers,
            &residual_buffer,
            &o_buffer,
            &norm_weight_buffer,
            &attention_state_buffer,
            &normed_buffer,
            hidden_dim,
            eps,
        )?;
        let router_out_dim = self.encode_matmul_weight(
            encoder,
            &mut owned_buffers,
            &normed_buffer,
            batch,
            hidden_dim,
            router.weight(),
            &router_buffer,
        )?;
        if router_out_dim != expert_count {
            return Err(InferError::Dimension(format!(
                "tail MoE shared routeur sort {router_out_dim}, attendu {expert_count}"
            )));
        }
        self.encode_topk_softmax(
            encoder,
            &mut owned_buffers,
            &router_buffer,
            &indices_buffer,
            &scores_buffer,
            expert_count,
            top_k,
        )?;
        if !self.encode_gather_gate_up_swiglu(
            encoder,
            &mut owned_buffers,
            &normed_buffer,
            1,
            &stacked.gate,
            &stacked.up,
            &indices_buffer,
            top_k,
            &hidden_buffer,
        )? {
            self.encode_gather_matmul(
                encoder,
                &mut owned_buffers,
                &normed_buffer,
                1,
                &stacked.gate,
                &indices_buffer,
                top_k,
                &gate_buffer,
            )?;
            self.encode_gather_matmul(
                encoder,
                &mut owned_buffers,
                &normed_buffer,
                1,
                &stacked.up,
                &indices_buffer,
                top_k,
                &up_buffer,
            )?;
            self.encode_swiglu(
                encoder,
                &mut owned_buffers,
                &gate_buffer,
                &up_buffer,
                &hidden_buffer,
                checked_len(top_k, inter_dim, "tail shared swiglu")?,
            )?;
        }
        self.encode_gather_matmul(
            encoder,
            &mut owned_buffers,
            &hidden_buffer,
            top_k,
            &stacked.down,
            &indices_buffer,
            top_k,
            &down_buffer,
        )?;
        self.encode_weighted_sum_add_topk(
            encoder,
            &mut owned_buffers,
            &down_buffer,
            &scores_buffer,
            &attention_state_buffer,
            &output_buffer,
            top_k,
            hidden_dim,
        )?;
        let projected_gate_dim = self.encode_matmul_weight(
            encoder,
            &mut owned_buffers,
            &normed_buffer,
            batch,
            hidden_dim,
            shared_gate.weight(),
            &shared_gate_buffer,
        )?;
        if projected_gate_dim != 1 {
            return Err(InferError::Dimension(format!(
                "tail shared gate Metal sort {projected_gate_dim}, attendu 1"
            )));
        }
        let projected_shared_gate_dim = self.encode_matmul_weight(
            encoder,
            &mut owned_buffers,
            &normed_buffer,
            batch,
            hidden_dim,
            shared_gate_proj.weight(),
            &shared_proj_gate_buffer,
        )?;
        let projected_shared_up_dim = self.encode_matmul_weight(
            encoder,
            &mut owned_buffers,
            &normed_buffer,
            batch,
            hidden_dim,
            shared_up_proj.weight(),
            &shared_up_buffer,
        )?;
        if projected_shared_gate_dim != shared_inter_dim
            || projected_shared_up_dim != shared_inter_dim
        {
            return Err(InferError::Dimension(format!(
                "tail shared expert Metal proj gate={projected_shared_gate_dim}, up={projected_shared_up_dim}, attendu {shared_inter_dim}"
            )));
        }
        self.encode_swiglu(
            encoder,
            &mut owned_buffers,
            &shared_proj_gate_buffer,
            &shared_up_buffer,
            &shared_hidden_buffer,
            shared_inter_dim,
        )?;
        let projected_shared_down_dim = self.encode_matmul_weight(
            encoder,
            &mut owned_buffers,
            &shared_hidden_buffer,
            batch,
            shared_inter_dim,
            shared_down_proj.weight(),
            &shared_down_buffer,
        )?;
        if projected_shared_down_dim != hidden_dim {
            return Err(InferError::Dimension(format!(
                "tail shared expert Metal down sort {projected_shared_down_dim}, attendu {hidden_dim}"
            )));
        }
        self.encode_add_sigmoid_scaled(
            encoder,
            &shared_down_buffer,
            &shared_gate_buffer,
            &output_buffer,
            hidden_dim,
        )?;
        encoder_guard.end();
        commit_and_wait(command_buffer)?;

        let output = read_f32_buffer(&output_buffer, hidden_dim)?;
        Tensor::from_vec(vec![1, hidden_dim], output)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "kernel spécialisé full-attn: QKV concat + split q/gate"
    )]
    pub(crate) fn encode_full_attn_qkv_split_buffers(
        &self,
        encoder: &ComputeCommandEncoderRef,
        lhs_buffer: &BufferRef,
        in_dim: usize,
        weight: &MetalLinearWeightBuffers,
        qkv_output_buffer: &BufferRef,
        q_output_buffer: &BufferRef,
        gate_output_buffer: &BufferRef,
        q_heads: usize,
        head_dim: usize,
    ) -> Result<Option<usize>> {
        if !fused_attn_epilogue_enabled() {
            return Ok(None);
        }
        let MetalLinearWeightBuffers::AffineQuantized {
            packed,
            scales,
            biases,
            out_dim,
            in_dim: weight_in_dim,
            packed_cols,
            group_size,
            bits,
            groups,
        } = weight
        else {
            return Ok(None);
        };
        if in_dim != *weight_in_dim {
            return Err(InferError::Dimension(format!(
                "full-attn qkv split x=[1,{in_dim}] rhs=[{out_dim},{weight_in_dim}]"
            )));
        }
        if !fast_affine_qmv_enabled(*out_dim)
            || *bits != FAST_QMV_BITS
            || *group_size != FAST_QMV_GROUP_SIZE
            || in_dim % 512 != 0
        {
            return Ok(None);
        }
        let q_dim = q_heads
            .checked_mul(head_dim)
            .ok_or_else(|| InferError::Dimension("full-attn q_dim déborde".to_string()))?;
        let q_gate_dim = q_dim
            .checked_mul(2)
            .ok_or_else(|| InferError::Dimension("full-attn q_gate_dim déborde".to_string()))?;
        if *out_dim < q_gate_dim {
            return Err(InferError::Dimension(format!(
                "full-attn qkv split out_dim={out_dim}, q_gate_dim={q_gate_dim}"
            )));
        }
        let fast_dims = [
            checked_u32(*out_dim, "full qkv split out_dim")?,
            checked_u32(in_dim, "full qkv split in_dim")?,
            checked_u32(*packed_cols, "full qkv split packed_cols")?,
            checked_u32(*groups, "full qkv split groups")?,
        ];
        let q_dims = [
            checked_u32(q_heads, "full qkv split q_heads")?,
            checked_u32(head_dim, "full qkv split head_dim")?,
        ];
        encoder.set_compute_pipeline_state(&self.affine_qkv_split_qmv_fast_u4_gs64_f32);
        encoder.set_buffer(0, Some(lhs_buffer), 0);
        encoder.set_buffer(1, Some(packed), 0);
        encoder.set_buffer(2, Some(scales), 0);
        encoder.set_buffer(3, Some(biases), 0);
        encoder.set_buffer(4, Some(qkv_output_buffer), 0);
        encoder.set_buffer(5, Some(q_output_buffer), 0);
        encoder.set_buffer(6, Some(gate_output_buffer), 0);
        set_u32_bytes(encoder, 7, &fast_dims, "full_qkv_split_dims")?;
        set_u32_bytes(encoder, 8, &q_dims, "full_qkv_split_q_dims")?;
        profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(
                1,
                checked_nsuint(out_dim.div_ceil(8), "full qkv split out groups")?,
                1,
            ),
            MTLSize::new(64, 1, 1),
        );
        post_dispatch_barrier(encoder);
        Ok(Some(*out_dim))
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "kernel spécialisé full-attn: rms_norm prologue + QKV concat + split q/gate"
    )]
    pub(crate) fn encode_full_attn_qkv_split_rms_buffers(
        &self,
        encoder: &ComputeCommandEncoderRef,
        lhs_buffer: &BufferRef,
        rms_weight_buffer: &BufferRef,
        rms_eps: f32,
        in_dim: usize,
        weight: &MetalLinearWeightBuffers,
        qkv_output_buffer: &BufferRef,
        q_output_buffer: &BufferRef,
        gate_output_buffer: &BufferRef,
        q_heads: usize,
        head_dim: usize,
    ) -> Result<Option<usize>> {
        if !fused_rms_prologue_enabled() || !fused_attn_epilogue_enabled() {
            return Ok(None);
        }
        let MetalLinearWeightBuffers::AffineQuantized {
            packed,
            scales,
            biases,
            out_dim,
            in_dim: weight_in_dim,
            packed_cols,
            group_size,
            bits,
            groups,
        } = weight
        else {
            return Ok(None);
        };
        if in_dim != *weight_in_dim {
            return Err(InferError::Dimension(format!(
                "full-attn qkv split rms x=[1,{in_dim}] rhs=[{out_dim},{weight_in_dim}]"
            )));
        }
        if !fast_affine_qmv_enabled(*out_dim)
            || *bits != FAST_QMV_BITS
            || *group_size != FAST_QMV_GROUP_SIZE
            || in_dim % 512 != 0
        {
            return Ok(None);
        }
        let q_dim = q_heads
            .checked_mul(head_dim)
            .ok_or_else(|| InferError::Dimension("full-attn q_dim déborde".to_string()))?;
        let q_gate_dim = q_dim
            .checked_mul(2)
            .ok_or_else(|| InferError::Dimension("full-attn q_gate_dim déborde".to_string()))?;
        if *out_dim < q_gate_dim {
            return Err(InferError::Dimension(format!(
                "full-attn qkv split rms out_dim={out_dim}, q_gate_dim={q_gate_dim}"
            )));
        }
        let fast_dims = [
            checked_u32(*out_dim, "full qkv split rms out_dim")?,
            checked_u32(in_dim, "full qkv split rms in_dim")?,
            checked_u32(*packed_cols, "full qkv split rms packed_cols")?,
            checked_u32(*groups, "full qkv split rms groups")?,
        ];
        let q_dims = [
            checked_u32(q_heads, "full qkv split rms q_heads")?,
            checked_u32(head_dim, "full qkv split rms head_dim")?,
        ];
        encoder.set_compute_pipeline_state(&self.affine_qkv_split_rms_qmv_fast_u4_gs64_f32);
        encoder.set_buffer(0, Some(lhs_buffer), 0);
        encoder.set_buffer(1, Some(rms_weight_buffer), 0);
        encoder.set_buffer(2, Some(packed), 0);
        encoder.set_buffer(3, Some(scales), 0);
        encoder.set_buffer(4, Some(biases), 0);
        encoder.set_buffer(5, Some(qkv_output_buffer), 0);
        encoder.set_buffer(6, Some(q_output_buffer), 0);
        encoder.set_buffer(7, Some(gate_output_buffer), 0);
        set_u32_bytes(encoder, 8, &fast_dims, "full_qkv_split_rms_dims")?;
        set_u32_bytes(encoder, 9, &q_dims, "full_qkv_split_rms_q_dims")?;
        set_f32_bytes(encoder, 10, &[rms_eps], "full_qkv_split_rms_eps")?;
        profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(
                1,
                checked_nsuint(out_dim.div_ceil(8), "full qkv split rms out groups")?,
                1,
            ),
            MTLSize::new(64, 1, 1),
        );
        post_dispatch_barrier(encoder);
        Ok(Some(*out_dim))
    }

    pub(crate) fn encode_full_attn_o_proj_gated_buffers(
        &self,
        encoder: &ComputeCommandEncoderRef,
        ctx_buffer: &BufferRef,
        gate_buffer: &BufferRef,
        in_dim: usize,
        weight: &MetalLinearWeightBuffers,
        output_buffer: &BufferRef,
    ) -> Result<Option<usize>> {
        if !fused_attn_epilogue_enabled() {
            return Ok(None);
        }
        let MetalLinearWeightBuffers::AffineQuantized {
            packed,
            scales,
            biases,
            out_dim,
            in_dim: weight_in_dim,
            packed_cols,
            group_size,
            bits,
            groups,
        } = weight
        else {
            return Ok(None);
        };
        if in_dim != *weight_in_dim {
            return Err(InferError::Dimension(format!(
                "full-attn o_proj gated x=[1,{in_dim}] rhs=[{out_dim},{weight_in_dim}]"
            )));
        }
        if !fast_affine_qmv_enabled(*out_dim)
            || *bits != FAST_QMV_BITS
            || *group_size != FAST_QMV_GROUP_SIZE
            || in_dim % 512 != 0
        {
            return Ok(None);
        }
        let fast_dims = [
            checked_u32(*out_dim, "full o gated out_dim")?,
            checked_u32(in_dim, "full o gated in_dim")?,
            checked_u32(*packed_cols, "full o gated packed_cols")?,
            checked_u32(*groups, "full o gated groups")?,
        ];
        encoder.set_compute_pipeline_state(&self.affine_qmv_gated_input_fast_u4_gs64_f32);
        encoder.set_buffer(0, Some(ctx_buffer), 0);
        encoder.set_buffer(1, Some(gate_buffer), 0);
        encoder.set_buffer(2, Some(packed), 0);
        encoder.set_buffer(3, Some(scales), 0);
        encoder.set_buffer(4, Some(biases), 0);
        encoder.set_buffer(5, Some(output_buffer), 0);
        set_u32_bytes(encoder, 6, &fast_dims, "full_o_gated_dims")?;
        profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(
                1,
                checked_nsuint(out_dim.div_ceil(8), "full o gated out groups")?,
                1,
            ),
            MTLSize::new(64, 1, 1),
        );
        post_dispatch_barrier(encoder);
        Ok(Some(*out_dim))
    }

    pub(super) fn encode_rms_norm_rope_heads(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input_buffer: &BufferRef,
        weight_buffer: &BufferRef,
        output_buffer: &BufferRef,
        spec: PrefillAttentionSpec,
        heads: usize,
    ) -> Result<()> {
        let dims = [
            checked_u32(spec.seq, "rope seq")?,
            checked_u32(heads, "rope heads")?,
            checked_u32(spec.head_dim, "rope head_dim")?,
            checked_u32(spec.rope_dims, "rope dims")?,
        ];
        encoder.set_compute_pipeline_state(&self.rms_norm_rope_heads_f32);
        encoder.set_buffer(0, Some(input_buffer), 0);
        encoder.set_buffer(1, Some(weight_buffer), 0);
        encoder.set_buffer(2, Some(output_buffer), 0);
        set_u32_bytes(encoder, 3, &dims, "rms_rope_dims")?;
        set_f32_bytes(encoder, 4, &[spec.eps, spec.rope_theta], "rms_rope_params")?;
        profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(
                checked_nsuint(heads, "rope heads")?,
                checked_nsuint(spec.seq, "rope seq")?,
                1,
            ),
            MTLSize::new(FULL_ATTN_PREFILL_TG_WIDTH, 1, 1),
        );
        post_dispatch_barrier(encoder);
        Ok(())
    }

    pub(super) fn encode_causal_attention_prefill(
        &self,
        encoder: &ComputeCommandEncoderRef,
        q_buffer: &BufferRef,
        k_buffer: &BufferRef,
        v_buffer: &BufferRef,
        output_buffer: &BufferRef,
        spec: PrefillAttentionSpec,
    ) -> Result<()> {
        let dims = [
            checked_u32(spec.seq, "attention seq")?,
            checked_u32(spec.q_heads, "attention q_heads")?,
            checked_u32(spec.kv_heads, "attention kv_heads")?,
            checked_u32(spec.head_dim, "attention head_dim")?,
        ];
        encoder.set_compute_pipeline_state(&self.causal_attention_prefill_f32);
        encoder.set_buffer(0, Some(q_buffer), 0);
        encoder.set_buffer(1, Some(k_buffer), 0);
        encoder.set_buffer(2, Some(v_buffer), 0);
        encoder.set_buffer(3, Some(output_buffer), 0);
        set_u32_bytes(encoder, 4, &dims, "causal_prefill_dims")?;
        profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(
                checked_nsuint(spec.q_heads, "attention q_heads")?,
                checked_nsuint(spec.seq, "attention seq")?,
                1,
            ),
            MTLSize::new(FULL_ATTN_PREFILL_TG_WIDTH, 1, 1),
        );
        post_dispatch_barrier(encoder);
        Ok(())
    }
}
