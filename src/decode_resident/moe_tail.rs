//! Encodage des tails MoE Gemma résidents.

use super::*;

impl DecodeResidentState {
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
