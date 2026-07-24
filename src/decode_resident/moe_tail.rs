//! Encodage des tails MoE Gemma résidents.

use super::*;

impl DecodeResidentState {
    /// Encode le tail dense + MoE parallèle Gemma 4 depuis le résiduel post-attention.
    #[expect(
        clippy::too_many_arguments,
        reason = "tail Gemma parallèle: buffers + poids nécessaires à l'encodage"
    )]
    pub(crate) fn encode_gemma_parallel_moe_tail(
        &self,
        executor: &MetalExecutor,
        encoder: &ComputeCommandEncoderRef,
        owned: &mut Vec<Buffer>,
        hidden_state: &BufferRef,
        layer_out: &BufferRef,
        rows: usize,
        hidden: usize,
        eps: f32,
        gemma: GemmaParallelMoeTailWeights<'_>,
    ) -> Result<()> {
        if crate::runtime_flags::trace_resident_enabled() {
            static TRACED: OnceLock<()> = OnceLock::new();
            TRACED.get_or_init(|| {
                eprintln!("decode résident: tail parallèle Gemma4 actif");
            });
        }
        let hidden_len = rows.checked_mul(hidden).ok_or_else(|| {
            InferError::Dimension("decode Gemma parallèle hidden déborde".to_string())
        })?;
        let dense_inter_len = rows.checked_mul(gemma.dense_inter_dim).ok_or_else(|| {
            InferError::Dimension("decode Gemma parallèle inter déborde".to_string())
        })?;
        let dense_input = self.scratch().lease(hidden_len, GpuElement::F32)?;
        let dense_gate = self.scratch().lease(dense_inter_len, GpuElement::F32)?;
        let dense_up = self.scratch().lease(dense_inter_len, GpuElement::F32)?;
        let dense_geglu = self.scratch().lease(dense_inter_len, GpuElement::F32)?;
        let dense_down = self.scratch().lease(hidden_len, GpuElement::F32)?;
        let dense_out = self.scratch().lease(hidden_len, GpuElement::F32)?;
        let moe_input = self.scratch().lease(hidden_len, GpuElement::F32)?;
        let moe_out = self.scratch().lease(hidden_len, GpuElement::F32)?;
        let ffn_out = self.scratch().lease(hidden_len, GpuElement::F32)?;
        let ffn_normed = self.scratch().lease(hidden_len, GpuElement::F32)?;

        executor.encode_gemma_parallel_tail_rows(
            encoder,
            owned,
            hidden_state,
            layer_out,
            rows,
            hidden,
            eps,
            gemma.dense_gate_proj,
            gemma.dense_up_proj,
            gemma.dense_down_proj,
            gemma.pre_feedforward_norm,
            gemma.post_feedforward_norm_1,
            gemma.moe,
            gemma.top_k,
            gemma.router_norm,
            gemma.per_expert_scale,
            gemma.pre_feedforward_norm_2,
            gemma.post_feedforward_norm_2,
            gemma.post_feedforward_norm,
            gemma.layer_scalar,
            gemma.dense_inter_dim,
            dense_input.tensor().buffer(),
            dense_gate.tensor().buffer(),
            dense_up.tensor().buffer(),
            dense_geglu.tensor().buffer(),
            dense_down.tensor().buffer(),
            dense_out.tensor().buffer(),
            moe_input.tensor().buffer(),
            moe_out.tensor().buffer(),
            ffn_out.tensor().buffer(),
            ffn_normed.tensor().buffer(),
        )
    }

    /// Encode le tail MoE routed Gemma 4 : pré-RMSNorm FFN, experts GeGLU, post-RMSNorm.
    #[expect(
        clippy::too_many_arguments,
        reason = "tail Gemma MoE: buffers + poids nécessaires à l'encodage"
    )]
    pub(super) fn encode_gemma_moe_tail(
        &self,
        executor: &MetalExecutor,
        encoder: &ComputeCommandEncoderRef,
        owned: &mut Vec<Buffer>,
        hidden_state: &BufferRef,
        layer_out: &BufferRef,
        hidden: usize,
        eps: f32,
        gemma: GemmaMoeTailWeights<'_>,
    ) -> Result<()> {
        let ffn_input = self.scratch().lease(hidden, GpuElement::F32)?;
        let moe_out = self.scratch().lease(hidden, GpuElement::F32)?;
        let ffn_normed = self.scratch().lease(hidden, GpuElement::F32)?;

        executor.encode_rms_norm_rows(
            encoder,
            hidden_state,
            gemma.pre_feedforward_norm,
            ffn_input.tensor().buffer(),
            1,
            hidden,
            eps,
        )?;
        executor.encode_moe_routed_buffers_with_activation(
            encoder,
            owned,
            ffn_input.tensor().buffer(),
            None,
            moe_out.tensor().buffer(),
            hidden,
            gemma.moe,
            gemma.top_k,
            crate::Activation::GeluTanh,
        )?;
        executor.encode_rms_norm_rows(
            encoder,
            moe_out.tensor().buffer(),
            gemma.post_feedforward_norm,
            ffn_normed.tensor().buffer(),
            1,
            hidden,
            eps,
        )?;
        executor.encode_add_scaled(
            encoder,
            owned,
            hidden_state,
            ffn_normed.tensor().buffer(),
            layer_out,
            1.0,
            hidden,
        )?;
        if let Some(scale) = gemma.layer_scalar {
            executor.encode_accumulate_scaled(
                encoder,
                owned,
                layer_out,
                layer_out,
                scale - 1.0,
                hidden,
            )?;
        }
        Ok(())
    }
}
