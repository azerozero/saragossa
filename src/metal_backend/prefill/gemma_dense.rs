//! Encodage du tail FFN dense Gemma 4 pour le prefill résident.

use super::*;

impl MetalExecutor {
    /// Encode une branche dense Gemma 4 normalisée, sans ajout résiduel.
    #[expect(
        clippy::too_many_arguments,
        reason = "branche dense Gemma batchée: buffers, poids et dimensions restent explicites"
    )]
    pub(super) fn encode_gemma_dense_branch_rows(
        &self,
        encoder: &ComputeCommandEncoderRef,
        owned: &mut Vec<Buffer>,
        hidden_state: &BufferRef,
        branch_out: &BufferRef,
        rows: usize,
        hidden: usize,
        eps: f32,
        gate_proj: &MetalLinearWeightBuffers,
        up_proj: &MetalLinearWeightBuffers,
        down_proj: &MetalLinearWeightBuffers,
        pre_feedforward_norm: &BufferRef,
        post_feedforward_norm: &BufferRef,
        inter_dim: usize,
        ffn_input: &BufferRef,
        gate: &BufferRef,
        up: &BufferRef,
        geglu: &BufferRef,
        down: &BufferRef,
    ) -> Result<()> {
        let inter_len = checked_len(rows, inter_dim, "prefill Gemma dense inter")?;
        self.encode_rms_norm_rows(
            encoder,
            hidden_state,
            pre_feedforward_norm,
            ffn_input,
            rows,
            hidden,
            eps,
        )?;
        let gate_dim = self.encode_matmul_weight_buffers(
            encoder, ffn_input, rows, hidden, gate_proj, gate, false,
        )?;
        let up_dim = self
            .encode_matmul_weight_buffers(encoder, ffn_input, rows, hidden, up_proj, up, false)?;
        if gate_dim != inter_dim || up_dim != inter_dim {
            return Err(InferError::Dimension(format!(
                "prefill Gemma dense gate/up sortent gate={gate_dim} up={up_dim}, attendu {inter_dim}"
            )));
        }
        self.encode_geglu_tanh(encoder, owned, gate, up, geglu, inter_len)?;
        let down_dim = self.encode_matmul_weight_buffers(
            encoder, geglu, rows, inter_dim, down_proj, down, false,
        )?;
        if down_dim != hidden {
            return Err(InferError::Dimension(format!(
                "prefill Gemma dense down sort {down_dim}, attendu {hidden}"
            )));
        }
        self.encode_rms_norm_rows(
            encoder,
            down,
            post_feedforward_norm,
            branch_out,
            rows,
            hidden,
            eps,
        )
    }

    /// Encode le tail dense Gemma 4 sur toutes les lignes du prefill.
    #[expect(
        clippy::too_many_arguments,
        reason = "tail Gemma dense batché: buffers, poids et dimensions restent explicites"
    )]
    pub(super) fn encode_gemma_dense_tail_rows(
        &self,
        encoder: &ComputeCommandEncoderRef,
        owned: &mut Vec<Buffer>,
        hidden_state: &BufferRef,
        layer_out: &BufferRef,
        rows: usize,
        hidden: usize,
        eps: f32,
        gate_proj: &MetalLinearWeightBuffers,
        up_proj: &MetalLinearWeightBuffers,
        down_proj: &MetalLinearWeightBuffers,
        pre_feedforward_norm: &BufferRef,
        post_feedforward_norm: &BufferRef,
        layer_scalar: Option<f32>,
        inter_dim: usize,
        ffn_input: &BufferRef,
        gate: &BufferRef,
        up: &BufferRef,
        geglu: &BufferRef,
        down: &BufferRef,
        ffn_normed: &BufferRef,
    ) -> Result<()> {
        let hidden_len = checked_len(rows, hidden, "prefill Gemma dense hidden")?;
        self.encode_gemma_dense_branch_rows(
            encoder,
            owned,
            hidden_state,
            ffn_normed,
            rows,
            hidden,
            eps,
            gate_proj,
            up_proj,
            down_proj,
            pre_feedforward_norm,
            post_feedforward_norm,
            inter_dim,
            ffn_input,
            gate,
            up,
            geglu,
            down,
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

    /// Exécute le tail dense Gemma 4 standalone dans un command buffer.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si la variante, les dimensions ou Metal sont invalides.
    #[cfg(test)]
    pub(crate) fn gemma_dense_tail_prefill_resident(
        &self,
        hidden_state: &Tensor,
        tail: PrefillMoeTail<'_>,
        eps: f32,
    ) -> Result<Tensor> {
        let (rows, hidden) = hidden_state.as_matrix()?;
        let PrefillMoeTail::GemmaDense {
            gate_proj,
            up_proj,
            down_proj,
            pre_feedforward_norm,
            post_feedforward_norm,
            layer_scalar,
            inter_dim,
        } = tail
        else {
            return Err(InferError::Config(
                "tail standalone attendu GemmaDense".to_string(),
            ));
        };
        let shape = self.check_prefill_gemma_dense_tail_shape(
            hidden,
            inter_dim,
            gate_proj,
            up_proj,
            down_proj,
            pre_feedforward_norm,
            post_feedforward_norm,
            layer_scalar,
        )?;
        let scratch = self.allocate_prefill_gemma_dense_tail_scratch(rows, hidden, inter_dim)?;
        let PrefillResidentTailShape::GemmaDense {
            gate_proj,
            up_proj,
            down_proj,
            pre_feedforward_norm,
            post_feedforward_norm,
            layer_scalar,
            ..
        } = shape
        else {
            return Err(InferError::Config(
                "shape standalone GemmaDense incohérent".to_string(),
            ));
        };
        let PrefillResidentTailScratch::GemmaDense {
            ffn_input,
            gate,
            up,
            geglu,
            down,
            ffn_normed,
        } = scratch
        else {
            return Err(InferError::Config(
                "scratch standalone GemmaDense incohérent".to_string(),
            ));
        };
        let hidden_len = checked_len(rows, hidden, "standalone Gemma dense hidden")?;
        let hidden_buffer =
            self.upload_f32_buffer(hidden_state.data(), "standalone_gemma_dense_hidden")?;
        let output_buffer =
            self.uncached_f32_buffer(hidden_len, "standalone_gemma_dense_output")?;
        let mut owned = Vec::new();
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let encoder_guard = EncoderEndGuard::new(encoder);
        self.encode_gemma_dense_tail_rows(
            encoder,
            &mut owned,
            &hidden_buffer,
            &output_buffer,
            rows,
            hidden,
            eps,
            &gate_proj,
            &up_proj,
            &down_proj,
            &pre_feedforward_norm,
            &post_feedforward_norm,
            layer_scalar,
            inter_dim,
            &ffn_input,
            &gate,
            &up,
            &geglu,
            &down,
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
