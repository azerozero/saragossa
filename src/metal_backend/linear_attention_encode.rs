//! Kernels élémentaires Metal de linear-attention.

use super::*;

const LINEAR_ATTN_TG_WIDTH: u64 = 32;

#[expect(
    clippy::too_many_arguments,
    reason = "wrappers d'encodage Metal: buffers, dimensions et offsets restent explicites"
)]
impl MetalExecutor {
    pub(super) fn encode_linear_attn_conv(
        &self,
        encoder: &ComputeCommandEncoderRef,
        qkv_buffer: &BufferRef,
        conv_weight_buffer: &BufferRef,
        conv_state_buffer: &BufferRef,
        conv_out_buffer: &BufferRef,
        conv_dim: usize,
        kernel: usize,
    ) -> Result<()> {
        self.encode_linear_attn_conv_with_offset(
            encoder,
            qkv_buffer,
            0,
            conv_weight_buffer,
            conv_state_buffer,
            conv_out_buffer,
            conv_dim,
            kernel,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "signature d'encodage Metal: buffers + offset + dimensions"
    )]
    pub(super) fn encode_linear_attn_conv_with_offset(
        &self,
        encoder: &ComputeCommandEncoderRef,
        qkv_buffer: &BufferRef,
        qkv_offset: u64,
        conv_weight_buffer: &BufferRef,
        conv_state_buffer: &BufferRef,
        conv_out_buffer: &BufferRef,
        conv_dim: usize,
        kernel: usize,
    ) -> Result<()> {
        let dims = [
            checked_u32(conv_dim, "linear-attn conv_dim")?,
            checked_u32(kernel, "linear-attn conv kernel")?,
        ];
        encoder.set_compute_pipeline_state(&self.linear_attn_conv_silu_f32);
        encoder.set_buffer(0, Some(qkv_buffer), qkv_offset);
        encoder.set_buffer(1, Some(conv_weight_buffer), 0);
        encoder.set_buffer(2, Some(conv_state_buffer), 0);
        encoder.set_buffer(3, Some(conv_out_buffer), 0);
        set_u32_bytes(encoder, 4, &dims, "linear_attn_conv_dims")?;
        self.dispatch_1d(encoder, &self.linear_attn_conv_silu_f32, conv_dim)
    }

    pub(super) fn encode_linear_attn_norm_gates(
        &self,
        encoder: &ComputeCommandEncoderRef,
        conv_out_buffer: &BufferRef,
        beta_input_buffer: &BufferRef,
        gate_input_buffer: &BufferRef,
        a_log_buffer: &BufferRef,
        dt_bias_buffer: &BufferRef,
        q_norm_buffer: &BufferRef,
        k_norm_buffer: &BufferRef,
        beta_buffer: &BufferRef,
        decay_buffer: &BufferRef,
        spec: LinearAttentionStepSpec,
    ) -> Result<()> {
        self.encode_linear_attn_norm_gates_with_offsets(
            encoder,
            conv_out_buffer,
            beta_input_buffer,
            0,
            gate_input_buffer,
            0,
            a_log_buffer,
            dt_bias_buffer,
            q_norm_buffer,
            k_norm_buffer,
            beta_buffer,
            decay_buffer,
            spec,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "signature d'encodage Metal: buffers + offsets + dimensions"
    )]
    pub(super) fn encode_linear_attn_norm_gates_with_offsets(
        &self,
        encoder: &ComputeCommandEncoderRef,
        conv_out_buffer: &BufferRef,
        beta_input_buffer: &BufferRef,
        beta_input_offset: u64,
        gate_input_buffer: &BufferRef,
        gate_input_offset: u64,
        a_log_buffer: &BufferRef,
        dt_bias_buffer: &BufferRef,
        q_norm_buffer: &BufferRef,
        k_norm_buffer: &BufferRef,
        beta_buffer: &BufferRef,
        decay_buffer: &BufferRef,
        spec: LinearAttentionStepSpec,
    ) -> Result<()> {
        let dims = [
            checked_u32(spec.num_key_heads, "linear-attn key heads")?,
            checked_u32(spec.num_value_heads, "linear-attn value heads")?,
            checked_u32(spec.key_head_dim, "linear-attn key head dim")?,
            checked_u32(spec.value_head_dim, "linear-attn value head dim")?,
        ];
        let inv = (spec.key_head_dim as f32).powf(-0.5);
        let scales = [inv * inv, inv];
        let groups = spec.num_key_heads.max(spec.num_value_heads);
        encoder.set_compute_pipeline_state(&self.linear_attn_norm_gates_f32);
        encoder.set_buffer(0, Some(conv_out_buffer), 0);
        encoder.set_buffer(1, Some(beta_input_buffer), beta_input_offset);
        encoder.set_buffer(2, Some(gate_input_buffer), gate_input_offset);
        encoder.set_buffer(3, Some(a_log_buffer), 0);
        encoder.set_buffer(4, Some(dt_bias_buffer), 0);
        encoder.set_buffer(5, Some(q_norm_buffer), 0);
        encoder.set_buffer(6, Some(k_norm_buffer), 0);
        encoder.set_buffer(7, Some(beta_buffer), 0);
        encoder.set_buffer(8, Some(decay_buffer), 0);
        set_u32_bytes(encoder, 9, &dims, "linear_attn_norm_dims")?;
        set_f32_bytes(encoder, 10, &scales, "linear_attn_norm_scales")?;
        profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(checked_nsuint(groups, "linear-attn norm groups")?, 1, 1),
            MTLSize::new(LINEAR_ATTN_TG_WIDTH, 1, 1),
        );
        post_dispatch_barrier(encoder);
        Ok(())
    }

    pub(super) fn encode_linear_attn_gated_delta(
        &self,
        encoder: &ComputeCommandEncoderRef,
        conv_out_buffer: &BufferRef,
        q_norm_buffer: &BufferRef,
        k_norm_buffer: &BufferRef,
        beta_buffer: &BufferRef,
        decay_buffer: &BufferRef,
        ssm_state_buffer: &BufferRef,
        y_buffer: &BufferRef,
        spec: LinearAttentionStepSpec,
    ) -> Result<()> {
        let repeat = spec
            .num_value_heads
            .checked_div(spec.num_key_heads)
            .ok_or_else(|| InferError::Metal("linear-attn repeat nul".to_string()))?;
        let dims = [
            checked_u32(spec.num_value_heads, "linear-attn value heads")?,
            checked_u32(spec.value_head_dim, "linear-attn value head dim")?,
            checked_u32(spec.key_head_dim, "linear-attn key head dim")?,
            checked_u32(repeat, "linear-attn repeat")?,
        ];
        encoder.set_compute_pipeline_state(&self.linear_attn_gated_delta_f32);
        encoder.set_buffer(0, Some(conv_out_buffer), 0);
        encoder.set_buffer(1, Some(q_norm_buffer), 0);
        encoder.set_buffer(2, Some(k_norm_buffer), 0);
        encoder.set_buffer(3, Some(beta_buffer), 0);
        encoder.set_buffer(4, Some(decay_buffer), 0);
        encoder.set_buffer(5, Some(ssm_state_buffer), 0);
        encoder.set_buffer(6, Some(y_buffer), 0);
        set_u32_bytes(encoder, 7, &dims, "linear_attn_delta_dims")?;
        profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(
                checked_nsuint(spec.value_head_dim, "linear-attn value dim")?,
                checked_nsuint(spec.num_value_heads, "linear-attn value heads")?,
                1,
            ),
            MTLSize::new(LINEAR_ATTN_TG_WIDTH, 1, 1),
        );
        post_dispatch_barrier(encoder);
        Ok(())
    }

    pub(super) fn encode_linear_attn_rms_gate(
        &self,
        encoder: &ComputeCommandEncoderRef,
        y_buffer: &BufferRef,
        z_buffer: &BufferRef,
        norm_weight_buffer: &BufferRef,
        gated_buffer: &BufferRef,
        spec: LinearAttentionStepSpec,
    ) -> Result<()> {
        self.encode_linear_attn_rms_gate_with_offset(
            encoder,
            y_buffer,
            z_buffer,
            0,
            norm_weight_buffer,
            gated_buffer,
            spec,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "signature d'encodage Metal: buffers + offset + dimensions"
    )]
    pub(super) fn encode_linear_attn_rms_gate_with_offset(
        &self,
        encoder: &ComputeCommandEncoderRef,
        y_buffer: &BufferRef,
        z_buffer: &BufferRef,
        z_offset: u64,
        norm_weight_buffer: &BufferRef,
        gated_buffer: &BufferRef,
        spec: LinearAttentionStepSpec,
    ) -> Result<()> {
        let dims = [
            checked_u32(spec.num_value_heads, "linear-attn value heads")?,
            checked_u32(spec.value_head_dim, "linear-attn value head dim")?,
        ];
        encoder.set_compute_pipeline_state(&self.linear_attn_rms_gate_f32);
        encoder.set_buffer(0, Some(y_buffer), 0);
        encoder.set_buffer(1, Some(z_buffer), z_offset);
        encoder.set_buffer(2, Some(norm_weight_buffer), 0);
        encoder.set_buffer(3, Some(gated_buffer), 0);
        set_u32_bytes(encoder, 4, &dims, "linear_attn_rms_dims")?;
        set_f32_bytes(encoder, 5, &[spec.rms_eps], "linear_attn_rms_eps")?;
        profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(
                checked_nsuint(spec.num_value_heads, "linear-attn rms heads")?,
                1,
                1,
            ),
            MTLSize::new(LINEAR_ATTN_TG_WIDTH, 1, 1),
        );
        post_dispatch_barrier(encoder);
        Ok(())
    }
}
