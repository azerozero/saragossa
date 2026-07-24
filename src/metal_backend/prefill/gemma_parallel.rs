//! Encodage du tail FFN parallèle Gemma 4 pour le prefill résident.

use super::*;

impl MetalExecutor {
    /// Encode le tail parallèle dense + MoE Gemma 4 sur toutes les lignes.
    #[expect(
        clippy::too_many_arguments,
        reason = "tail Gemma parallèle batché: branches, normes et scratch restent explicites"
    )]
    pub(crate) fn encode_gemma_parallel_tail_rows(
        &self,
        encoder: &ComputeCommandEncoderRef,
        owned: &mut Vec<Buffer>,
        hidden_state: &BufferRef,
        layer_out: &BufferRef,
        rows: usize,
        hidden: usize,
        eps: f32,
        dense_gate_proj: &MetalLinearWeightBuffers,
        dense_up_proj: &MetalLinearWeightBuffers,
        dense_down_proj: &MetalLinearWeightBuffers,
        pre_feedforward_norm: &BufferRef,
        post_feedforward_norm_1: &BufferRef,
        moe: &MetalMoeRoutedWeights,
        top_k: usize,
        router_norm: Option<(&BufferRef, f32)>,
        per_expert_scale: Option<&BufferRef>,
        pre_feedforward_norm_2: &BufferRef,
        post_feedforward_norm_2: &BufferRef,
        post_feedforward_norm: &BufferRef,
        layer_scalar: Option<f32>,
        dense_inter_dim: usize,
        dense_input: &BufferRef,
        dense_gate: &BufferRef,
        dense_up: &BufferRef,
        dense_geglu: &BufferRef,
        dense_down: &BufferRef,
        dense_out: &BufferRef,
        moe_input: &BufferRef,
        moe_out: &BufferRef,
        ffn_out: &BufferRef,
        ffn_normed: &BufferRef,
    ) -> Result<()> {
        let hidden_len = checked_len(rows, hidden, "prefill Gemma parallèle hidden")?;
        self.encode_gemma_dense_branch_rows(
            encoder,
            owned,
            hidden_state,
            dense_out,
            rows,
            hidden,
            eps,
            dense_gate_proj,
            dense_up_proj,
            dense_down_proj,
            pre_feedforward_norm,
            post_feedforward_norm_1,
            dense_inter_dim,
            dense_input,
            dense_gate,
            dense_up,
            dense_geglu,
            dense_down,
        )?;
        self.encode_rms_norm_rows(
            encoder,
            hidden_state,
            pre_feedforward_norm_2,
            moe_input,
            rows,
            hidden,
            eps,
        )?;
        let router_input = if let Some((router_norm, router_eps)) = router_norm {
            self.encode_rms_norm_rows(
                encoder,
                hidden_state,
                router_norm,
                ffn_normed,
                rows,
                hidden,
                router_eps,
            )?;
            ffn_normed
        } else {
            hidden_state
        };
        self.encode_moe_routed_buffers_rows_with_router_input_and_activation(
            encoder,
            owned,
            moe_input,
            router_input,
            per_expert_scale,
            None,
            ffn_normed,
            rows,
            hidden,
            moe,
            top_k,
            crate::Activation::GeluTanh,
        )?;
        self.encode_rms_norm_rows(
            encoder,
            ffn_normed,
            post_feedforward_norm_2,
            moe_out,
            rows,
            hidden,
            eps,
        )?;
        self.encode_add_scaled(encoder, owned, dense_out, moe_out, ffn_out, 1.0, hidden_len)?;
        self.encode_rms_norm_rows(
            encoder,
            ffn_out,
            post_feedforward_norm,
            ffn_normed,
            rows,
            hidden,
            eps,
        )?;
        self.encode_add_scaled(
            encoder,
            owned,
            hidden_state,
            ffn_normed,
            layer_out,
            1.0,
            hidden_len,
        )?;
        if let Some(scale) = layer_scalar {
            self.encode_accumulate_scaled(
                encoder,
                owned,
                layer_out,
                layer_out,
                scale - 1.0,
                hidden_len,
            )?;
        }
        Ok(())
    }

    /// Exécute le tail parallèle Gemma 4 standalone dans un command buffer.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si la variante, les dimensions ou Metal sont invalides.
    #[cfg(test)]
    pub(crate) fn gemma_parallel_tail_prefill_resident(
        &self,
        hidden_state: &Tensor,
        tail: PrefillMoeTail<'_>,
        eps: f32,
    ) -> Result<Tensor> {
        let (rows, hidden) = hidden_state.as_matrix()?;
        let PrefillMoeTail::GemmaParallel {
            dense_gate_proj,
            dense_up_proj,
            dense_down_proj,
            pre_feedforward_norm,
            post_feedforward_norm_1,
            router,
            experts,
            top_k,
            router_norm,
            per_expert_scale,
            pre_feedforward_norm_2,
            post_feedforward_norm_2,
            post_feedforward_norm,
            layer_scalar,
            dense_inter_dim,
        } = tail
        else {
            return Err(InferError::Config(
                "tail standalone attendu GemmaParallel".to_string(),
            ));
        };
        let shape = self.check_prefill_gemma_parallel_tail_shape(
            hidden,
            dense_inter_dim,
            dense_gate_proj,
            dense_up_proj,
            dense_down_proj,
            pre_feedforward_norm,
            post_feedforward_norm_1,
            router,
            experts,
            top_k,
            router_norm,
            per_expert_scale,
            pre_feedforward_norm_2,
            post_feedforward_norm_2,
            post_feedforward_norm,
            layer_scalar,
        )?;
        let scratch =
            self.allocate_prefill_gemma_parallel_tail_scratch(rows, hidden, dense_inter_dim)?;
        let PrefillResidentTailShape::GemmaParallel {
            dense_gate_proj,
            dense_up_proj,
            dense_down_proj,
            pre_feedforward_norm,
            post_feedforward_norm_1,
            moe,
            router_norm,
            per_expert_scale,
            pre_feedforward_norm_2,
            post_feedforward_norm_2,
            post_feedforward_norm,
            layer_scalar,
            ..
        } = shape
        else {
            return Err(InferError::Config(
                "shape standalone GemmaParallel incohérent".to_string(),
            ));
        };
        let PrefillResidentTailScratch::GemmaParallel {
            dense_input,
            dense_gate,
            dense_up,
            dense_geglu,
            dense_down,
            dense_out,
            moe_input,
            moe_out,
            ffn_out,
            ffn_normed,
        } = scratch
        else {
            return Err(InferError::Config(
                "scratch standalone GemmaParallel incohérent".to_string(),
            ));
        };

        let hidden_len = checked_len(rows, hidden, "standalone Gemma parallèle hidden")?;
        let hidden_buffer =
            self.upload_f32_buffer(hidden_state.data(), "standalone_gemma_parallel_hidden")?;
        let output_buffer =
            self.uncached_f32_buffer(hidden_len, "standalone_gemma_parallel_output")?;
        let mut owned = Vec::new();
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let encoder_guard = EncoderEndGuard::new(encoder);
        self.encode_gemma_parallel_tail_rows(
            encoder,
            &mut owned,
            &hidden_buffer,
            &output_buffer,
            rows,
            hidden,
            eps,
            &dense_gate_proj,
            &dense_up_proj,
            &dense_down_proj,
            &pre_feedforward_norm,
            &post_feedforward_norm_1,
            &moe,
            top_k,
            router_norm
                .as_ref()
                .map(|(weight, eps)| (weight.as_ref(), *eps)),
            per_expert_scale.as_ref().map(Buffer::as_ref),
            &pre_feedforward_norm_2,
            &post_feedforward_norm_2,
            &post_feedforward_norm,
            layer_scalar,
            dense_inter_dim,
            &dense_input,
            &dense_gate,
            &dense_up,
            &dense_geglu,
            &dense_down,
            &dense_out,
            &moe_input,
            &moe_out,
            &ffn_out,
            &ffn_normed,
        )?;
        encoder_guard.end();
        commit_and_wait(command_buffer)?;
        Tensor::from_vec(
            vec![rows, hidden],
            read_f32_buffer(&output_buffer, hidden_len)?,
        )
    }
}
