//! Orchestration Metal des pas linear-attention.

use super::*;

type LinearAttentionStateCaptures = (
    Vec<Option<Vec<LinearAttentionMetalState>>>,
    Vec<crate::decode_resident::ScratchLease>,
);

#[expect(
    clippy::too_many_arguments,
    reason = "wrappers d'encodage Metal: buffers, dimensions et offsets restent explicites"
)]
impl MetalExecutor {
    fn uncached_linear_ssm_buffer(
        &self,
        len: usize,
        bf16: bool,
        label: &'static str,
    ) -> Result<metal::Buffer> {
        if bf16 {
            if len == 0 {
                return Err(InferError::Metal(format!("buffer {label} vide")));
            }
            return Ok(self
                .device
                .new_buffer(byte_len::<u16>(len)?, MTLResourceOptions::StorageModeShared));
        }
        self.uncached_f32_buffer(len, label)
    }

    fn buffer_from_linear_ssm_seed(
        &self,
        seed: &[f32],
        bf16: bool,
        label: &'static str,
    ) -> Result<metal::Buffer> {
        if bf16 {
            return self.buffer_from_f32_as_bf16(seed, label);
        }
        self.buffer_from_f32(seed, label)
    }

    pub(crate) fn lease_linear_attn_state_captures(
        &self,
        scratch: &crate::decode_resident::ScratchPool,
        states: &[Option<&LinearAttentionMetalState>],
        rows: usize,
    ) -> Result<LinearAttentionStateCaptures> {
        let _ = self;
        if rows == 0 {
            return Err(InferError::Dimension(
                "captures linear-attn rows vide".to_string(),
            ));
        }
        let mut leases = Vec::new();
        let mut captures = Vec::with_capacity(states.len());
        for state in states {
            let Some(state) = state else {
                captures.push(None);
                continue;
            };
            let mut rows_out = Vec::with_capacity(rows);
            for _ in 0..rows {
                let conv =
                    scratch.lease(state.conv_len, crate::decode_resident::GpuElement::F32)?;
                let ssm_element = if state.ssm_bf16 {
                    crate::decode_resident::GpuElement::Bf16
                } else {
                    crate::decode_resident::GpuElement::F32
                };
                let ssm = scratch.lease(state.ssm_len, ssm_element)?;
                rows_out.push(LinearAttentionMetalState {
                    conv: conv.tensor().buffer().clone(),
                    ssm: ssm.tensor().buffer().clone(),
                    conv_len: state.conv_len,
                    ssm_len: state.ssm_len,
                    conv_dim: state.conv_dim,
                    conv_kernel_dim: state.conv_kernel_dim,
                    num_value_heads: state.num_value_heads,
                    value_head_dim: state.value_head_dim,
                    key_head_dim: state.key_head_dim,
                    ssm_bf16: state.ssm_bf16,
                });
                leases.push(conv);
                leases.push(ssm);
            }
            captures.push(Some(rows_out));
        }
        Ok((captures, leases))
    }

    fn encode_capture_linear_attn_state(
        &self,
        encoder: &ComputeCommandEncoderRef,
        state: &LinearAttentionMetalState,
        capture: &LinearAttentionMetalState,
    ) -> Result<()> {
        self.encode_capture_linear_attn_conv_state(encoder, state, capture)?;
        if state.ssm_bf16 {
            return self.encode_copy_u16(encoder, &state.ssm, &capture.ssm, state.ssm_len);
        }
        self.encode_copy_with_offsets(encoder, &state.ssm, 0, &capture.ssm, 0, state.ssm_len)
    }

    fn encode_capture_linear_attn_conv_state(
        &self,
        encoder: &ComputeCommandEncoderRef,
        state: &LinearAttentionMetalState,
        capture: &LinearAttentionMetalState,
    ) -> Result<()> {
        if state.conv_len != capture.conv_len
            || state.ssm_len != capture.ssm_len
            || state.ssm_bf16 != capture.ssm_bf16
        {
            return Err(InferError::Shape(format!(
                "capture linear-attn: capture conv={}/ssm={}/bf16={} ≠ état conv={}/ssm={}/bf16={}",
                capture.conv_len,
                capture.ssm_len,
                capture.ssm_bf16,
                state.conv_len,
                state.ssm_len,
                state.ssm_bf16
            )));
        }
        self.encode_copy_with_offsets(encoder, &state.conv, 0, &capture.conv, 0, state.conv_len)
    }

    /// Exécute le step cached linear attention avec état SSM résident GPU.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si les dimensions divergent ou si Metal échoue.
    pub(crate) fn linear_attention_cached_step_resident(
        &self,
        input: &Tensor,
        in_proj_qkv: &Linear,
        in_proj_z: &Linear,
        in_proj_b: &Linear,
        in_proj_a: &Linear,
        out_proj: &Linear,
        conv_weight: &Tensor,
        a_log: &Tensor,
        dt_bias: &Tensor,
        norm_weight: &Tensor,
        conv_state_seed: &[f32],
        ssm_state_seed: &[f32],
        state: &mut Option<LinearAttentionMetalState>,
        spec: LinearAttentionStepSpec,
    ) -> Result<Tensor> {
        ensure_biasless(in_proj_qkv, "linear_attn.in_proj_qkv")?;
        ensure_biasless(in_proj_z, "linear_attn.in_proj_z")?;
        ensure_biasless(in_proj_b, "linear_attn.in_proj_b")?;
        ensure_biasless(in_proj_a, "linear_attn.in_proj_a")?;
        ensure_biasless(out_proj, "linear_attn.out_proj")?;
        let (batch, in_dim) = input.as_matrix()?;
        if batch != 1 {
            return Err(InferError::Dimension(format!(
                "linear-attn résident Metal attend batch=1, reçu {batch}"
            )));
        }
        if spec.num_key_heads == 0
            || spec.num_value_heads == 0
            || spec.key_head_dim == 0
            || spec.value_head_dim == 0
            || spec.conv_kernel_dim < 2
            || spec.num_value_heads % spec.num_key_heads != 0
        {
            return Err(InferError::Dimension(format!(
                "linear-attn résident Metal dims invalides: key_heads={}, value_heads={}, key_dim={}, value_dim={}, kernel={}",
                spec.num_key_heads,
                spec.num_value_heads,
                spec.key_head_dim,
                spec.value_head_dim,
                spec.conv_kernel_dim
            )));
        }
        let key_dim = checked_len(spec.num_key_heads, spec.key_head_dim, "linear-attn key_dim")?;
        let value_dim = checked_len(
            spec.num_value_heads,
            spec.value_head_dim,
            "linear-attn value_dim",
        )?;
        let conv_dim = key_dim
            .checked_mul(2)
            .and_then(|twice| twice.checked_add(value_dim))
            .ok_or_else(|| InferError::Shape("linear-attn conv_dim trop grand".to_string()))?;
        let keep = spec.conv_kernel_dim - 1;
        let conv_len = checked_len(keep, conv_dim, "linear-attn conv state")?;
        let ssm_len = checked_len(
            checked_len(
                spec.num_value_heads,
                spec.value_head_dim,
                "linear-attn state heads",
            )?,
            spec.key_head_dim,
            "linear-attn state",
        )?;
        if conv_state_seed.len() != conv_len || ssm_state_seed.len() != ssm_len {
            return Err(InferError::Dimension(format!(
                "linear-attn résident seed conv={}, ssm={}, attendu conv={conv_len}, ssm={ssm_len}",
                conv_state_seed.len(),
                ssm_state_seed.len()
            )));
        }
        expect_linear_shape(
            in_proj_qkv.weight(),
            conv_dim,
            in_dim,
            "linear_attn.in_proj_qkv",
        )?;
        expect_linear_shape(
            in_proj_z.weight(),
            value_dim,
            in_dim,
            "linear_attn.in_proj_z",
        )?;
        expect_linear_shape(
            in_proj_b.weight(),
            spec.num_value_heads,
            in_dim,
            "linear_attn.in_proj_b",
        )?;
        expect_linear_shape(
            in_proj_a.weight(),
            spec.num_value_heads,
            in_dim,
            "linear_attn.in_proj_a",
        )?;
        expect_linear_in(out_proj.weight(), value_dim, "linear_attn.out_proj")?;
        match conv_weight.shape() {
            [channels, kernel, one]
                if *channels == conv_dim && *kernel == spec.conv_kernel_dim && *one == 1 => {}
            [channels, one, kernel]
                if *channels == conv_dim && *one == 1 && *kernel == spec.conv_kernel_dim => {}
            shape => {
                return Err(InferError::Dimension(format!(
                    "linear_attn.conv1d.weight résident attendu [{conv_dim},{},1] ou [{conv_dim},1,{}], reçu {shape:?}",
                    spec.conv_kernel_dim, spec.conv_kernel_dim
                )))
            }
        }
        let a_log = dense_vector(a_log, spec.num_value_heads, "linear_attn.A_log")?;
        let dt_bias = dense_vector(dt_bias, spec.num_value_heads, "linear_attn.dt_bias")?;
        let norm_weight =
            dense_vector(norm_weight, spec.value_head_dim, "linear_attn.norm.weight")?;
        let out_dim = linear_out_dim(out_proj.weight())?;
        self.ensure_linear_attention_metal_state(
            state,
            conv_state_seed,
            ssm_state_seed,
            conv_dim,
            conv_len,
            ssm_len,
            spec,
        )?;
        let Some(state) = state.as_ref() else {
            return Err(InferError::Metal(
                "état linear-attn résident non initialisé".to_string(),
            ));
        };

        let input_buffer = self.upload_f32_buffer(input.data(), "linear_attn_input")?;
        let output_buffer = self.new_f32_buffer(out_dim, "linear_attn_output")?;
        let mut owned_buffers = Vec::new();

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let encoder_guard = EncoderEndGuard::new(encoder);
        self.encode_linear_attn_resident(
            encoder,
            &mut owned_buffers,
            &input_buffer,
            &output_buffer,
            LinearAttnResidentWeights {
                in_proj_qkv,
                in_proj_z,
                in_proj_b,
                in_proj_a,
                out_proj,
                conv_weight,
                a_log,
                dt_bias,
                norm_weight,
            },
            state,
            spec,
            LinearAttnResidentDims {
                in_dim,
                conv_dim,
                value_dim,
                key_dim,
            },
        )?;
        encoder_guard.end();
        set_commit_label("lin_step_res");
        commit_and_wait(command_buffer)?;

        let output = read_f32_buffer(&output_buffer, out_dim)?;
        Tensor::from_vec(vec![1, out_dim], output)
    }

    /// Exécute plusieurs positions linear-attn en batch : les projections et la
    /// sortie sont encodées en batch, tandis que le scan conv/SSM reste ordonné.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si les dimensions divergent ou si Metal échoue.
    #[expect(
        clippy::too_many_arguments,
        reason = "mirror du wrapper résident single-step"
    )]
    pub(crate) fn linear_attention_cached_batch_resident(
        &self,
        input: &Tensor,
        in_proj_qkv: &Linear,
        in_proj_z: &Linear,
        in_proj_b: &Linear,
        in_proj_a: &Linear,
        out_proj: &Linear,
        conv_weight: &Tensor,
        a_log: &Tensor,
        dt_bias: &Tensor,
        norm_weight: &Tensor,
        conv_state_seed: &[f32],
        ssm_state_seed: &[f32],
        state: &mut Option<LinearAttentionMetalState>,
        spec: LinearAttentionStepSpec,
    ) -> Result<Tensor> {
        ensure_biasless(in_proj_qkv, "linear_attn.in_proj_qkv")?;
        ensure_biasless(in_proj_z, "linear_attn.in_proj_z")?;
        ensure_biasless(in_proj_b, "linear_attn.in_proj_b")?;
        ensure_biasless(in_proj_a, "linear_attn.in_proj_a")?;
        ensure_biasless(out_proj, "linear_attn.out_proj")?;
        let (batch, in_dim) = input.as_matrix()?;
        if batch == 0 {
            return Err(InferError::Dimension(
                "linear-attn résident batch vide".to_string(),
            ));
        }
        if spec.num_key_heads == 0
            || spec.num_value_heads == 0
            || spec.key_head_dim == 0
            || spec.value_head_dim == 0
            || spec.conv_kernel_dim < 2
            || spec.num_value_heads % spec.num_key_heads != 0
        {
            return Err(InferError::Dimension(format!(
                "linear-attn résident Metal dims invalides: key_heads={}, value_heads={}, key_dim={}, value_dim={}, kernel={}",
                spec.num_key_heads,
                spec.num_value_heads,
                spec.key_head_dim,
                spec.value_head_dim,
                spec.conv_kernel_dim
            )));
        }
        let key_dim = checked_len(spec.num_key_heads, spec.key_head_dim, "linear-attn key_dim")?;
        let value_dim = checked_len(
            spec.num_value_heads,
            spec.value_head_dim,
            "linear-attn value_dim",
        )?;
        let conv_dim = key_dim
            .checked_mul(2)
            .and_then(|twice| twice.checked_add(value_dim))
            .ok_or_else(|| InferError::Shape("linear-attn conv_dim trop grand".to_string()))?;
        let keep = spec.conv_kernel_dim - 1;
        let conv_len = checked_len(keep, conv_dim, "linear-attn conv state")?;
        let ssm_len = checked_len(
            checked_len(
                spec.num_value_heads,
                spec.value_head_dim,
                "linear-attn state heads",
            )?,
            spec.key_head_dim,
            "linear-attn state",
        )?;
        if conv_state_seed.len() != conv_len || ssm_state_seed.len() != ssm_len {
            return Err(InferError::Dimension(format!(
                "linear-attn résident seed conv={}, ssm={}, attendu conv={conv_len}, ssm={ssm_len}",
                conv_state_seed.len(),
                ssm_state_seed.len()
            )));
        }
        expect_linear_shape(
            in_proj_qkv.weight(),
            conv_dim,
            in_dim,
            "linear_attn.in_proj_qkv",
        )?;
        expect_linear_shape(
            in_proj_z.weight(),
            value_dim,
            in_dim,
            "linear_attn.in_proj_z",
        )?;
        expect_linear_shape(
            in_proj_b.weight(),
            spec.num_value_heads,
            in_dim,
            "linear_attn.in_proj_b",
        )?;
        expect_linear_shape(
            in_proj_a.weight(),
            spec.num_value_heads,
            in_dim,
            "linear_attn.in_proj_a",
        )?;
        expect_linear_in(out_proj.weight(), value_dim, "linear_attn.out_proj")?;
        match conv_weight.shape() {
            [channels, kernel, one]
                if *channels == conv_dim && *kernel == spec.conv_kernel_dim && *one == 1 => {}
            [channels, one, kernel]
                if *channels == conv_dim && *one == 1 && *kernel == spec.conv_kernel_dim => {}
            shape => {
                return Err(InferError::Dimension(format!(
                    "linear_attn.conv1d.weight résident attendu [{conv_dim},{},1] ou [{conv_dim},1,{}], reçu {shape:?}",
                    spec.conv_kernel_dim, spec.conv_kernel_dim
                )))
            }
        }
        let a_log = dense_vector(a_log, spec.num_value_heads, "linear_attn.A_log")?;
        let dt_bias = dense_vector(dt_bias, spec.num_value_heads, "linear_attn.dt_bias")?;
        let norm_weight =
            dense_vector(norm_weight, spec.value_head_dim, "linear_attn.norm.weight")?;
        let out_dim = linear_out_dim(out_proj.weight())?;
        self.ensure_linear_attention_metal_state(
            state,
            conv_state_seed,
            ssm_state_seed,
            conv_dim,
            conv_len,
            ssm_len,
            spec,
        )?;
        let Some(state) = state.as_ref() else {
            return Err(InferError::Metal(
                "état linear-attn résident non initialisé".to_string(),
            ));
        };
        let input_buffer = self.upload_f32_buffer(input.data(), "linear_attn_batch_input")?;
        let qkv_buffer = self.private_f32_buffer(
            checked_len(batch, conv_dim, "linear-attn batch qkv")?,
            "linear_attn_batch_qkv",
        )?;
        let z_buffer = self.private_f32_buffer(
            checked_len(batch, value_dim, "linear-attn batch z")?,
            "linear_attn_batch_z",
        )?;
        let beta_input_buffer = self.private_f32_buffer(
            checked_len(batch, spec.num_value_heads, "linear-attn batch beta input")?,
            "linear_attn_batch_beta_input",
        )?;
        let gate_input_buffer = self.private_f32_buffer(
            checked_len(batch, spec.num_value_heads, "linear-attn batch gate input")?,
            "linear_attn_batch_gate_input",
        )?;
        let conv_weight_buffer =
            self.cached_buffer_from_f32(conv_weight.data(), "linear_attn_conv_weight")?;
        let a_log_buffer = self.cached_buffer_from_f32(a_log, "linear_attn_a_log")?;
        let dt_bias_buffer = self.cached_buffer_from_f32(dt_bias, "linear_attn_dt_bias")?;
        let norm_weight_buffer =
            self.cached_buffer_from_f32(norm_weight, "linear_attn_norm_weight")?;
        let gated_batch_buffer = self.private_f32_buffer(
            checked_len(batch, value_dim, "linear-attn batch gated")?,
            "linear_attn_batch_gated",
        )?;
        let output_buffer = self.new_f32_buffer(
            checked_len(batch, out_dim, "linear-attn batch output")?,
            "linear_attn_batch_output",
        )?;
        let use_seq_delta_batch = linear_delta_dk128_enabled() && spec.key_head_dim == 128;
        let mut owned_buffers = Vec::new();

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let encoder_guard = EncoderEndGuard::new(encoder);
        self.encode_matmul_weight(
            encoder,
            &mut owned_buffers,
            &input_buffer,
            batch,
            in_dim,
            in_proj_qkv.weight(),
            &qkv_buffer,
        )?;
        self.encode_matmul_weight(
            encoder,
            &mut owned_buffers,
            &input_buffer,
            batch,
            in_dim,
            in_proj_z.weight(),
            &z_buffer,
        )?;
        self.encode_matmul_weight(
            encoder,
            &mut owned_buffers,
            &input_buffer,
            batch,
            in_dim,
            in_proj_b.weight(),
            &beta_input_buffer,
        )?;
        self.encode_matmul_weight(
            encoder,
            &mut owned_buffers,
            &input_buffer,
            batch,
            in_dim,
            in_proj_a.weight(),
            &gate_input_buffer,
        )?;
        if use_seq_delta_batch {
            let conv_out_buffer = self.private_f32_buffer(
                checked_len(batch, conv_dim, "linear-attn batch conv_out")?,
                "linear_attn_batch_conv_out_seq",
            )?;
            let q_norm_buffer = self.private_f32_buffer(
                checked_len(batch, key_dim, "linear-attn batch q_norm")?,
                "linear_attn_batch_q_norm_seq",
            )?;
            let k_norm_buffer = self.private_f32_buffer(
                checked_len(batch, key_dim, "linear-attn batch k_norm")?,
                "linear_attn_batch_k_norm_seq",
            )?;
            let beta_buffer = self.private_f32_buffer(
                checked_len(batch, spec.num_value_heads, "linear-attn batch beta")?,
                "linear_attn_batch_beta_seq",
            )?;
            let decay_buffer = self.private_f32_buffer(
                checked_len(batch, spec.num_value_heads, "linear-attn batch decay")?,
                "linear_attn_batch_decay_seq",
            )?;
            let y_buffer = self.private_f32_buffer(
                checked_len(batch, value_dim, "linear-attn batch y")?,
                "linear_attn_batch_y_seq",
            )?;
            if spec.conv_kernel_dim == 4 && spec.value_head_dim == 128 && spec.key_head_dim == 128 {
                // Brick #8/#9 : conv+norm+gates BATCHÉ (1 dispatch + finalize conv_state) au
                // lieu de la boucle per-token (~16384 dispatches/couche → 2). Fait le calcul
                // fusé conv+norm+gates → correct quel que soit RETI_RUST_LINEAR_CONV_NORM_FUSED.
                self.encode_linear_attn_conv_norm_gates_k4_dk128_batch(
                    encoder,
                    &qkv_buffer,
                    &beta_input_buffer,
                    &gate_input_buffer,
                    &conv_weight_buffer,
                    &state.conv,
                    &a_log_buffer,
                    &dt_bias_buffer,
                    &conv_out_buffer,
                    &q_norm_buffer,
                    &k_norm_buffer,
                    &beta_buffer,
                    &decay_buffer,
                    batch,
                    spec,
                )?;
            } else {
                for pos in 0..batch {
                    let qkv_offset =
                        byte_offset_f32(pos * conv_dim, "linear-attn qkv batch offset")?;
                    let conv_out_offset =
                        byte_offset_f32(pos * conv_dim, "linear-attn conv_out batch offset")?;
                    let gate_offset = byte_offset_f32(
                        pos * spec.num_value_heads,
                        "linear-attn gate batch offset",
                    )?;
                    let q_norm_offset =
                        byte_offset_f32(pos * key_dim, "linear-attn q_norm batch offset")?;
                    let k_norm_offset =
                        byte_offset_f32(pos * key_dim, "linear-attn k_norm batch offset")?;
                    let beta_offset = byte_offset_f32(
                        pos * spec.num_value_heads,
                        "linear-attn beta batch offset",
                    )?;
                    let decay_offset = byte_offset_f32(
                        pos * spec.num_value_heads,
                        "linear-attn decay batch offset",
                    )?;
                    if linear_conv_norm_fused_enabled()
                        && spec.conv_kernel_dim == 4
                        && spec.value_head_dim == 128
                    {
                        self.encode_linear_attn_conv_norm_gates_k4_dk128_with_all_offsets(
                            encoder,
                            &qkv_buffer,
                            qkv_offset,
                            &beta_input_buffer,
                            gate_offset,
                            &gate_input_buffer,
                            gate_offset,
                            &conv_weight_buffer,
                            &state.conv,
                            &a_log_buffer,
                            &dt_bias_buffer,
                            &conv_out_buffer,
                            conv_out_offset,
                            &q_norm_buffer,
                            q_norm_offset,
                            &k_norm_buffer,
                            k_norm_offset,
                            &beta_buffer,
                            beta_offset,
                            &decay_buffer,
                            decay_offset,
                            spec,
                        )?;
                    } else {
                        self.encode_linear_attn_conv_with_offsets(
                            encoder,
                            &qkv_buffer,
                            qkv_offset,
                            &conv_weight_buffer,
                            &state.conv,
                            &conv_out_buffer,
                            conv_out_offset,
                            conv_dim,
                            spec.conv_kernel_dim,
                        )?;
                        memory_barrier_buffers(encoder);
                        self.encode_linear_attn_norm_gates_with_all_offsets(
                            encoder,
                            &conv_out_buffer,
                            conv_out_offset,
                            &beta_input_buffer,
                            gate_offset,
                            &gate_input_buffer,
                            gate_offset,
                            &a_log_buffer,
                            &dt_bias_buffer,
                            &q_norm_buffer,
                            q_norm_offset,
                            &k_norm_buffer,
                            k_norm_offset,
                            &beta_buffer,
                            beta_offset,
                            &decay_buffer,
                            decay_offset,
                            spec,
                        )?;
                    }
                }
            }
            if linear_chunked_enabled()
                && spec.key_head_dim == 128
                && spec.value_head_dim == 128
                && !state.ssm_bf16
                && self.chunk_delta_seq_layout.is_some()
            {
                // Port chunked-DeltaNet : forme chunkée (T/C étapes au lieu de T).
                self.encode_chunk_delta_seq_layout(
                    encoder,
                    &conv_out_buffer,
                    &q_norm_buffer,
                    &k_norm_buffer,
                    &beta_buffer,
                    &decay_buffer,
                    &state.ssm,
                    &y_buffer,
                    batch,
                    spec,
                )?;
            } else {
                self.encode_linear_attn_gated_delta_seq_dk128(
                    encoder,
                    &conv_out_buffer,
                    &q_norm_buffer,
                    &k_norm_buffer,
                    &beta_buffer,
                    &decay_buffer,
                    &state.ssm,
                    state.ssm_bf16,
                    &y_buffer,
                    batch,
                    spec,
                )?;
            }
            if spec.value_head_dim == 128 {
                // Brick #8 : rms-gate batché (1 dispatch) au lieu de la boucle per-token.
                self.encode_linear_attn_rms_gate_batch(
                    encoder,
                    &y_buffer,
                    &z_buffer,
                    &norm_weight_buffer,
                    &gated_batch_buffer,
                    batch,
                    spec,
                )?;
            } else {
                for pos in 0..batch {
                    let y_offset = byte_offset_f32(pos * value_dim, "linear-attn y batch offset")?;
                    let z_offset = byte_offset_f32(pos * value_dim, "linear-attn z batch offset")?;
                    let gated_offset =
                        byte_offset_f32(pos * value_dim, "linear-attn gated batch offset")?;
                    self.encode_linear_attn_rms_gate_with_offsets(
                        encoder,
                        &y_buffer,
                        y_offset,
                        &z_buffer,
                        z_offset,
                        &norm_weight_buffer,
                        &gated_batch_buffer,
                        gated_offset,
                        spec,
                    )?;
                }
            }
        } else {
            let conv_out_buffer =
                self.private_f32_buffer(conv_dim, "linear_attn_batch_conv_out")?;
            let q_norm_buffer = self.private_f32_buffer(key_dim, "linear_attn_batch_q_norm")?;
            let k_norm_buffer = self.private_f32_buffer(key_dim, "linear_attn_batch_k_norm")?;
            let beta_buffer =
                self.private_f32_buffer(spec.num_value_heads, "linear_attn_batch_beta")?;
            let decay_buffer =
                self.private_f32_buffer(spec.num_value_heads, "linear_attn_batch_decay")?;
            let y_buffer = self.private_f32_buffer(value_dim, "linear_attn_batch_y")?;
            let gated_row_buffer =
                self.private_f32_buffer(value_dim, "linear_attn_batch_gated_row")?;
            for pos in 0..batch {
                let qkv_offset = byte_offset_f32(pos * conv_dim, "linear-attn qkv batch offset")?;
                let z_offset = byte_offset_f32(pos * value_dim, "linear-attn z batch offset")?;
                let gate_offset =
                    byte_offset_f32(pos * spec.num_value_heads, "linear-attn gate batch offset")?;
                self.encode_linear_attn_conv_with_offset(
                    encoder,
                    &qkv_buffer,
                    qkv_offset,
                    &conv_weight_buffer,
                    &state.conv,
                    &conv_out_buffer,
                    conv_dim,
                    spec.conv_kernel_dim,
                )?;
                memory_barrier_buffers(encoder);
                self.encode_linear_attn_norm_gates_with_offsets(
                    encoder,
                    &conv_out_buffer,
                    &beta_input_buffer,
                    gate_offset,
                    &gate_input_buffer,
                    gate_offset,
                    &a_log_buffer,
                    &dt_bias_buffer,
                    &q_norm_buffer,
                    &k_norm_buffer,
                    &beta_buffer,
                    &decay_buffer,
                    spec,
                )?;
                self.encode_linear_attn_gated_delta(
                    encoder,
                    &conv_out_buffer,
                    &q_norm_buffer,
                    &k_norm_buffer,
                    &beta_buffer,
                    &decay_buffer,
                    &state.ssm,
                    state.ssm_bf16,
                    &y_buffer,
                    spec,
                )?;
                self.encode_linear_attn_rms_gate_with_offset(
                    encoder,
                    &y_buffer,
                    &z_buffer,
                    z_offset,
                    &norm_weight_buffer,
                    &gated_row_buffer,
                    spec,
                )?;
                let gated_offset =
                    byte_offset_f32(pos * value_dim, "linear-attn gated batch offset")?;
                self.encode_copy_with_offsets(
                    encoder,
                    &gated_row_buffer,
                    0,
                    &gated_batch_buffer,
                    gated_offset,
                    value_dim,
                )?;
            }
        }
        self.encode_matmul_weight(
            encoder,
            &mut owned_buffers,
            &gated_batch_buffer,
            batch,
            value_dim,
            out_proj.weight(),
            &output_buffer,
        )?;
        encoder_guard.end();
        set_commit_label("lin_batch");
        commit_and_wait(command_buffer)?;
        set_commit_label("post_linear");

        let output = read_f32_buffer(&output_buffer, batch * out_dim)?;
        Tensor::from_vec(vec![batch, out_dim], output)
    }

    /// Encode un batch linear-attn à partir de buffers déjà résidents.
    ///
    /// Contrairement à [`Self::linear_attention_cached_batch_resident`], ce
    /// helper ne fait ni upload d'entrée, ni commit, ni readback. Il conserve les
    /// kernels batchés du chemin prefill linéaire pour pouvoir s'enchaîner dans
    /// le command buffer prefill global.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si les dimensions divergent ou si Metal échoue.
    #[expect(
        clippy::too_many_arguments,
        reason = "mirror du wrapper résident batch sans upload/readback"
    )]
    pub(crate) fn encode_linear_attn_batch_resident(
        &self,
        encoder: &ComputeCommandEncoderRef,
        owned_buffers: &mut Vec<Buffer>,
        input_buffer: &BufferRef,
        output_buffer: &BufferRef,
        rows: usize,
        weights: LinearAttnResidentWeights<'_>,
        state: &LinearAttentionMetalState,
        spec: LinearAttentionStepSpec,
        dims: LinearAttnResidentDims,
    ) -> Result<()> {
        ensure_biasless(weights.in_proj_qkv, "linear_attn.in_proj_qkv")?;
        ensure_biasless(weights.in_proj_z, "linear_attn.in_proj_z")?;
        ensure_biasless(weights.in_proj_b, "linear_attn.in_proj_b")?;
        ensure_biasless(weights.in_proj_a, "linear_attn.in_proj_a")?;
        ensure_biasless(weights.out_proj, "linear_attn.out_proj")?;
        if rows == 0 {
            return Err(InferError::Dimension(
                "linear-attn résident batch vide".to_string(),
            ));
        }
        if spec.num_key_heads == 0
            || spec.num_value_heads == 0
            || spec.key_head_dim == 0
            || spec.value_head_dim == 0
            || spec.conv_kernel_dim < 2
            || spec.num_value_heads % spec.num_key_heads != 0
        {
            return Err(InferError::Dimension(format!(
                "linear-attn résident Metal dims invalides: key_heads={}, value_heads={}, key_dim={}, value_dim={}, kernel={}",
                spec.num_key_heads,
                spec.num_value_heads,
                spec.key_head_dim,
                spec.value_head_dim,
                spec.conv_kernel_dim
            )));
        }
        let LinearAttnResidentDims {
            in_dim,
            conv_dim,
            value_dim,
            key_dim,
        } = dims;
        let expected_key_dim =
            checked_len(spec.num_key_heads, spec.key_head_dim, "linear-attn key_dim")?;
        let expected_value_dim = checked_len(
            spec.num_value_heads,
            spec.value_head_dim,
            "linear-attn value_dim",
        )?;
        let expected_conv_dim = expected_key_dim
            .checked_mul(2)
            .and_then(|twice| twice.checked_add(expected_value_dim))
            .ok_or_else(|| InferError::Shape("linear-attn conv_dim trop grand".to_string()))?;
        if key_dim != expected_key_dim
            || value_dim != expected_value_dim
            || conv_dim != expected_conv_dim
        {
            return Err(InferError::Dimension(format!(
                "linear-attn résident dims incohérentes: key_dim={key_dim}/{expected_key_dim}, value_dim={value_dim}/{expected_value_dim}, conv_dim={conv_dim}/{expected_conv_dim}"
            )));
        }
        expect_linear_shape(
            weights.in_proj_qkv.weight(),
            conv_dim,
            in_dim,
            "linear_attn.in_proj_qkv",
        )?;
        expect_linear_shape(
            weights.in_proj_z.weight(),
            value_dim,
            in_dim,
            "linear_attn.in_proj_z",
        )?;
        expect_linear_shape(
            weights.in_proj_b.weight(),
            spec.num_value_heads,
            in_dim,
            "linear_attn.in_proj_b",
        )?;
        expect_linear_shape(
            weights.in_proj_a.weight(),
            spec.num_value_heads,
            in_dim,
            "linear_attn.in_proj_a",
        )?;
        expect_linear_in(weights.out_proj.weight(), value_dim, "linear_attn.out_proj")?;
        match weights.conv_weight.shape() {
            [channels, kernel, one]
                if *channels == conv_dim && *kernel == spec.conv_kernel_dim && *one == 1 => {}
            [channels, one, kernel]
                if *channels == conv_dim && *one == 1 && *kernel == spec.conv_kernel_dim => {}
            shape => {
                return Err(InferError::Dimension(format!(
                    "linear_attn.conv1d.weight résident attendu [{conv_dim},{},1] ou [{conv_dim},1,{}], reçu {shape:?}",
                    spec.conv_kernel_dim, spec.conv_kernel_dim
                )))
            }
        }
        if weights.a_log.len() != spec.num_value_heads {
            return Err(InferError::Dimension(format!(
                "linear_attn.A_log len={} attendu {}",
                weights.a_log.len(),
                spec.num_value_heads
            )));
        }
        if weights.dt_bias.len() != spec.num_value_heads {
            return Err(InferError::Dimension(format!(
                "linear_attn.dt_bias len={} attendu {}",
                weights.dt_bias.len(),
                spec.num_value_heads
            )));
        }
        if weights.norm_weight.len() != spec.value_head_dim {
            return Err(InferError::Dimension(format!(
                "linear_attn.norm.weight len={} attendu {}",
                weights.norm_weight.len(),
                spec.value_head_dim
            )));
        }
        let qkv_buffer = self.private_f32_buffer(
            checked_len(rows, conv_dim, "linear-attn batch qkv")?,
            "linear_attn_batch_resident_qkv",
        )?;
        let z_buffer = self.private_f32_buffer(
            checked_len(rows, value_dim, "linear-attn batch z")?,
            "linear_attn_batch_resident_z",
        )?;
        let beta_input_buffer = self.private_f32_buffer(
            checked_len(rows, spec.num_value_heads, "linear-attn batch beta input")?,
            "linear_attn_batch_resident_beta_input",
        )?;
        let gate_input_buffer = self.private_f32_buffer(
            checked_len(rows, spec.num_value_heads, "linear-attn batch gate input")?,
            "linear_attn_batch_resident_gate_input",
        )?;
        let conv_weight_buffer =
            self.cached_buffer_from_f32(weights.conv_weight.data(), "linear_attn_conv_weight")?;
        let a_log_buffer = self.cached_buffer_from_f32(weights.a_log, "linear_attn_a_log")?;
        let dt_bias_buffer = self.cached_buffer_from_f32(weights.dt_bias, "linear_attn_dt_bias")?;
        let norm_weight_buffer =
            self.cached_buffer_from_f32(weights.norm_weight, "linear_attn_norm_weight")?;
        let gated_batch_buffer = self.private_f32_buffer(
            checked_len(rows, value_dim, "linear-attn batch gated")?,
            "linear_attn_batch_resident_gated",
        )?;
        let use_seq_delta_batch = linear_delta_dk128_enabled() && spec.key_head_dim == 128;
        self.encode_matmul_weight(
            encoder,
            owned_buffers,
            input_buffer,
            rows,
            in_dim,
            weights.in_proj_qkv.weight(),
            &qkv_buffer,
        )?;
        self.encode_matmul_weight(
            encoder,
            owned_buffers,
            input_buffer,
            rows,
            in_dim,
            weights.in_proj_z.weight(),
            &z_buffer,
        )?;
        self.encode_matmul_weight(
            encoder,
            owned_buffers,
            input_buffer,
            rows,
            in_dim,
            weights.in_proj_b.weight(),
            &beta_input_buffer,
        )?;
        self.encode_matmul_weight(
            encoder,
            owned_buffers,
            input_buffer,
            rows,
            in_dim,
            weights.in_proj_a.weight(),
            &gate_input_buffer,
        )?;
        if use_seq_delta_batch {
            let conv_out_buffer = self.private_f32_buffer(
                checked_len(rows, conv_dim, "linear-attn batch conv_out")?,
                "linear_attn_batch_resident_conv_out_seq",
            )?;
            let q_norm_buffer = self.private_f32_buffer(
                checked_len(rows, key_dim, "linear-attn batch q_norm")?,
                "linear_attn_batch_resident_q_norm_seq",
            )?;
            let k_norm_buffer = self.private_f32_buffer(
                checked_len(rows, key_dim, "linear-attn batch k_norm")?,
                "linear_attn_batch_resident_k_norm_seq",
            )?;
            let beta_buffer = self.private_f32_buffer(
                checked_len(rows, spec.num_value_heads, "linear-attn batch beta")?,
                "linear_attn_batch_resident_beta_seq",
            )?;
            let decay_buffer = self.private_f32_buffer(
                checked_len(rows, spec.num_value_heads, "linear-attn batch decay")?,
                "linear_attn_batch_resident_decay_seq",
            )?;
            let y_buffer = self.private_f32_buffer(
                checked_len(rows, value_dim, "linear-attn batch y")?,
                "linear_attn_batch_resident_y_seq",
            )?;
            if spec.conv_kernel_dim == 4 && spec.value_head_dim == 128 && spec.key_head_dim == 128 {
                self.encode_linear_attn_conv_norm_gates_k4_dk128_batch(
                    encoder,
                    &qkv_buffer,
                    &beta_input_buffer,
                    &gate_input_buffer,
                    &conv_weight_buffer,
                    &state.conv,
                    &a_log_buffer,
                    &dt_bias_buffer,
                    &conv_out_buffer,
                    &q_norm_buffer,
                    &k_norm_buffer,
                    &beta_buffer,
                    &decay_buffer,
                    rows,
                    spec,
                )?;
            } else {
                for pos in 0..rows {
                    let qkv_offset =
                        byte_offset_f32(pos * conv_dim, "linear-attn qkv batch offset")?;
                    let conv_out_offset =
                        byte_offset_f32(pos * conv_dim, "linear-attn conv_out batch offset")?;
                    let gate_offset = byte_offset_f32(
                        pos * spec.num_value_heads,
                        "linear-attn gate batch offset",
                    )?;
                    let q_norm_offset =
                        byte_offset_f32(pos * key_dim, "linear-attn q_norm batch offset")?;
                    let k_norm_offset =
                        byte_offset_f32(pos * key_dim, "linear-attn k_norm batch offset")?;
                    let beta_offset = byte_offset_f32(
                        pos * spec.num_value_heads,
                        "linear-attn beta batch offset",
                    )?;
                    let decay_offset = byte_offset_f32(
                        pos * spec.num_value_heads,
                        "linear-attn decay batch offset",
                    )?;
                    if linear_conv_norm_fused_enabled()
                        && spec.conv_kernel_dim == 4
                        && spec.value_head_dim == 128
                    {
                        self.encode_linear_attn_conv_norm_gates_k4_dk128_with_all_offsets(
                            encoder,
                            &qkv_buffer,
                            qkv_offset,
                            &beta_input_buffer,
                            gate_offset,
                            &gate_input_buffer,
                            gate_offset,
                            &conv_weight_buffer,
                            &state.conv,
                            &a_log_buffer,
                            &dt_bias_buffer,
                            &conv_out_buffer,
                            conv_out_offset,
                            &q_norm_buffer,
                            q_norm_offset,
                            &k_norm_buffer,
                            k_norm_offset,
                            &beta_buffer,
                            beta_offset,
                            &decay_buffer,
                            decay_offset,
                            spec,
                        )?;
                    } else {
                        self.encode_linear_attn_conv_with_offsets(
                            encoder,
                            &qkv_buffer,
                            qkv_offset,
                            &conv_weight_buffer,
                            &state.conv,
                            &conv_out_buffer,
                            conv_out_offset,
                            conv_dim,
                            spec.conv_kernel_dim,
                        )?;
                        memory_barrier_buffers(encoder);
                        self.encode_linear_attn_norm_gates_with_all_offsets(
                            encoder,
                            &conv_out_buffer,
                            conv_out_offset,
                            &beta_input_buffer,
                            gate_offset,
                            &gate_input_buffer,
                            gate_offset,
                            &a_log_buffer,
                            &dt_bias_buffer,
                            &q_norm_buffer,
                            q_norm_offset,
                            &k_norm_buffer,
                            k_norm_offset,
                            &beta_buffer,
                            beta_offset,
                            &decay_buffer,
                            decay_offset,
                            spec,
                        )?;
                    }
                }
            }
            if linear_chunked_enabled()
                && spec.key_head_dim == 128
                && spec.value_head_dim == 128
                && !state.ssm_bf16
                && self.chunk_delta_seq_layout.is_some()
            {
                self.encode_chunk_delta_seq_layout(
                    encoder,
                    &conv_out_buffer,
                    &q_norm_buffer,
                    &k_norm_buffer,
                    &beta_buffer,
                    &decay_buffer,
                    &state.ssm,
                    &y_buffer,
                    rows,
                    spec,
                )?;
            } else {
                self.encode_linear_attn_gated_delta_seq_dk128(
                    encoder,
                    &conv_out_buffer,
                    &q_norm_buffer,
                    &k_norm_buffer,
                    &beta_buffer,
                    &decay_buffer,
                    &state.ssm,
                    state.ssm_bf16,
                    &y_buffer,
                    rows,
                    spec,
                )?;
            }
            if spec.value_head_dim == 128 {
                self.encode_linear_attn_rms_gate_batch(
                    encoder,
                    &y_buffer,
                    &z_buffer,
                    &norm_weight_buffer,
                    &gated_batch_buffer,
                    rows,
                    spec,
                )?;
            } else {
                for pos in 0..rows {
                    let y_offset = byte_offset_f32(pos * value_dim, "linear-attn y batch offset")?;
                    let z_offset = byte_offset_f32(pos * value_dim, "linear-attn z batch offset")?;
                    let gated_offset =
                        byte_offset_f32(pos * value_dim, "linear-attn gated batch offset")?;
                    self.encode_linear_attn_rms_gate_with_offsets(
                        encoder,
                        &y_buffer,
                        y_offset,
                        &z_buffer,
                        z_offset,
                        &norm_weight_buffer,
                        &gated_batch_buffer,
                        gated_offset,
                        spec,
                    )?;
                }
            }
        } else {
            let conv_out_buffer =
                self.private_f32_buffer(conv_dim, "linear_attn_batch_resident_conv_out")?;
            let q_norm_buffer =
                self.private_f32_buffer(key_dim, "linear_attn_batch_resident_q_norm")?;
            let k_norm_buffer =
                self.private_f32_buffer(key_dim, "linear_attn_batch_resident_k_norm")?;
            let beta_buffer =
                self.private_f32_buffer(spec.num_value_heads, "linear_attn_batch_resident_beta")?;
            let decay_buffer =
                self.private_f32_buffer(spec.num_value_heads, "linear_attn_batch_resident_decay")?;
            let y_buffer = self.private_f32_buffer(value_dim, "linear_attn_batch_resident_y")?;
            let gated_row_buffer =
                self.private_f32_buffer(value_dim, "linear_attn_batch_resident_gated_row")?;
            for pos in 0..rows {
                let qkv_offset = byte_offset_f32(pos * conv_dim, "linear-attn qkv batch offset")?;
                let z_offset = byte_offset_f32(pos * value_dim, "linear-attn z batch offset")?;
                let gate_offset =
                    byte_offset_f32(pos * spec.num_value_heads, "linear-attn gate batch offset")?;
                self.encode_linear_attn_conv_with_offset(
                    encoder,
                    &qkv_buffer,
                    qkv_offset,
                    &conv_weight_buffer,
                    &state.conv,
                    &conv_out_buffer,
                    conv_dim,
                    spec.conv_kernel_dim,
                )?;
                memory_barrier_buffers(encoder);
                self.encode_linear_attn_norm_gates_with_offsets(
                    encoder,
                    &conv_out_buffer,
                    &beta_input_buffer,
                    gate_offset,
                    &gate_input_buffer,
                    gate_offset,
                    &a_log_buffer,
                    &dt_bias_buffer,
                    &q_norm_buffer,
                    &k_norm_buffer,
                    &beta_buffer,
                    &decay_buffer,
                    spec,
                )?;
                self.encode_linear_attn_gated_delta(
                    encoder,
                    &conv_out_buffer,
                    &q_norm_buffer,
                    &k_norm_buffer,
                    &beta_buffer,
                    &decay_buffer,
                    &state.ssm,
                    state.ssm_bf16,
                    &y_buffer,
                    spec,
                )?;
                self.encode_linear_attn_rms_gate_with_offset(
                    encoder,
                    &y_buffer,
                    &z_buffer,
                    z_offset,
                    &norm_weight_buffer,
                    &gated_row_buffer,
                    spec,
                )?;
                let gated_offset =
                    byte_offset_f32(pos * value_dim, "linear-attn gated batch offset")?;
                self.encode_copy_with_offsets(
                    encoder,
                    &gated_row_buffer,
                    0,
                    &gated_batch_buffer,
                    gated_offset,
                    value_dim,
                )?;
            }
        }
        self.encode_matmul_weight(
            encoder,
            owned_buffers,
            &gated_batch_buffer,
            rows,
            value_dim,
            weights.out_proj.weight(),
            output_buffer,
        )?;
        Ok(())
    }

    /// Encode le pas linear-attn résident (9 kernels chaînés) dans un encoder
    /// PARTAGÉ, écrivant l'output dans `output_buffer` (RÉSIDENT, pas de readback).
    ///
    /// Cœur extrait de [`Self::linear_attention_cached_step_resident`] (dont la
    /// méthode publique est désormais un wrapper : upload + encoder + cet encode +
    /// commit + readback). Réutilisé par l'orchestration résidente 1c pour chaîner
    /// la couche sans commit intermédiaire. Les états `conv`/`ssm` (résidents)
    /// sont lus puis mis à jour in-place par les kernels conv/gated_delta.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si un dispatch ou une allocation de buffer échoue.
    #[expect(
        clippy::too_many_arguments,
        reason = "mirror des paramètres du pas linear-attn (poids + dims)"
    )]
    pub(crate) fn encode_linear_attn_resident(
        &self,
        encoder: &ComputeCommandEncoderRef,
        owned_buffers: &mut Vec<Buffer>,
        input_buffer: &BufferRef,
        output_buffer: &BufferRef,
        weights: LinearAttnResidentWeights<'_>,
        state: &LinearAttentionMetalState,
        spec: LinearAttentionStepSpec,
        dims: LinearAttnResidentDims,
    ) -> Result<()> {
        let LinearAttnResidentDims {
            in_dim,
            conv_dim,
            value_dim,
            key_dim,
        } = dims;
        let qkv_buffer = self.private_f32_buffer(conv_dim, "linear_attn_qkv")?;
        let z_buffer = self.private_f32_buffer(value_dim, "linear_attn_z")?;
        let beta_input_buffer =
            self.private_f32_buffer(spec.num_value_heads, "linear_attn_beta_input")?;
        let gate_input_buffer =
            self.private_f32_buffer(spec.num_value_heads, "linear_attn_gate_input")?;
        let conv_weight_buffer =
            self.cached_buffer_from_f32(weights.conv_weight.data(), "linear_attn_conv_weight")?;
        let a_log_buffer = self.cached_buffer_from_f32(weights.a_log, "linear_attn_a_log")?;
        let dt_bias_buffer = self.cached_buffer_from_f32(weights.dt_bias, "linear_attn_dt_bias")?;
        let norm_weight_buffer =
            self.cached_buffer_from_f32(weights.norm_weight, "linear_attn_norm_weight")?;
        let conv_out_buffer = self.private_f32_buffer(conv_dim, "linear_attn_conv_out")?;
        let q_norm_buffer = self.private_f32_buffer(key_dim, "linear_attn_q_norm")?;
        let k_norm_buffer = self.private_f32_buffer(key_dim, "linear_attn_k_norm")?;
        let beta_buffer = self.private_f32_buffer(spec.num_value_heads, "linear_attn_beta")?;
        let decay_buffer = self.private_f32_buffer(spec.num_value_heads, "linear_attn_decay")?;
        let y_buffer = self.private_f32_buffer(value_dim, "linear_attn_y")?;
        let gated_buffer = self.private_f32_buffer(value_dim, "linear_attn_gated")?;
        self.encode_matmul_weight(
            encoder,
            owned_buffers,
            input_buffer,
            1,
            in_dim,
            weights.in_proj_qkv.weight(),
            &qkv_buffer,
        )?;
        self.encode_matmul_weight(
            encoder,
            owned_buffers,
            input_buffer,
            1,
            in_dim,
            weights.in_proj_z.weight(),
            &z_buffer,
        )?;
        self.encode_matmul_weight(
            encoder,
            owned_buffers,
            input_buffer,
            1,
            in_dim,
            weights.in_proj_b.weight(),
            &beta_input_buffer,
        )?;
        self.encode_matmul_weight(
            encoder,
            owned_buffers,
            input_buffer,
            1,
            in_dim,
            weights.in_proj_a.weight(),
            &gate_input_buffer,
        )?;
        self.encode_linear_attn_conv(
            encoder,
            &qkv_buffer,
            &conv_weight_buffer,
            &state.conv,
            &conv_out_buffer,
            conv_dim,
            spec.conv_kernel_dim,
        )?;
        self.encode_linear_attn_norm_gates(
            encoder,
            &conv_out_buffer,
            &beta_input_buffer,
            &gate_input_buffer,
            &a_log_buffer,
            &dt_bias_buffer,
            &q_norm_buffer,
            &k_norm_buffer,
            &beta_buffer,
            &decay_buffer,
            spec,
        )?;
        self.encode_linear_attn_gated_delta(
            encoder,
            &conv_out_buffer,
            &q_norm_buffer,
            &k_norm_buffer,
            &beta_buffer,
            &decay_buffer,
            &state.ssm,
            state.ssm_bf16,
            &y_buffer,
            spec,
        )?;
        self.encode_linear_attn_rms_gate(
            encoder,
            &y_buffer,
            &z_buffer,
            &norm_weight_buffer,
            &gated_buffer,
            spec,
        )?;
        self.encode_matmul_weight(
            encoder,
            owned_buffers,
            &gated_buffer,
            1,
            value_dim,
            weights.out_proj.weight(),
            output_buffer,
        )?;
        Ok(())
    }

    /// Variante résidente du pas linear-attn utilisant des buffers de poids déjà
    /// résolus au setup de la génération.
    #[expect(
        clippy::too_many_arguments,
        reason = "mirror des paramètres du pas linear-attn (poids + dims)"
    )]
    pub(crate) fn encode_linear_attn_resident_buffers(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input_buffer: &BufferRef,
        input_norm: Option<(&BufferRef, f32)>,
        output_buffer: &BufferRef,
        weights: &MetalLinearAttnResidentWeights,
        state: &LinearAttentionMetalState,
        spec: LinearAttentionStepSpec,
        dims: LinearAttnResidentDims,
    ) -> Result<()> {
        let LinearAttnResidentDims {
            in_dim,
            conv_dim,
            value_dim,
            key_dim,
        } = dims;
        let in_proj_dim = conv_dim
            .checked_add(value_dim)
            .and_then(|value| value.checked_add(spec.num_value_heads))
            .and_then(|value| value.checked_add(spec.num_value_heads))
            .ok_or_else(|| {
                InferError::Dimension("linear-attn in_proj concat déborde".to_string())
            })?;
        let in_proj_buffer = self.private_f32_buffer(in_proj_dim, "linear_attn_in_proj")?;
        let conv_out_buffer = self.private_f32_buffer(conv_dim, "linear_attn_conv_out")?;
        let q_norm_buffer = self.private_f32_buffer(key_dim, "linear_attn_q_norm")?;
        let k_norm_buffer = self.private_f32_buffer(key_dim, "linear_attn_k_norm")?;
        let beta_buffer = self.private_f32_buffer(spec.num_value_heads, "linear_attn_beta")?;
        let decay_buffer = self.private_f32_buffer(spec.num_value_heads, "linear_attn_decay")?;
        let y_buffer = self.private_f32_buffer(value_dim, "linear_attn_y")?;
        let gated_buffer = self.private_f32_buffer(value_dim, "linear_attn_gated")?;
        let use_inv_delta = linear_inv_delta_enabled()
            && spec.key_head_dim == 128
            && spec.value_head_dim == 128
            && linear_delta_dk128_enabled();
        let in_proj_fused = match input_norm {
            Some((norm_weight, eps)) => self
                .encode_matmul_weight_buffers_rms_prologue(
                    encoder,
                    input_buffer,
                    norm_weight,
                    eps,
                    in_dim,
                    &weights.in_proj,
                    &in_proj_buffer,
                )?
                .is_some(),
            None => false,
        };
        if !in_proj_fused {
            let normed_buffer = match input_norm {
                Some((norm_weight, eps)) => {
                    let normed_buffer = self.private_f32_buffer(in_dim, "linear_attn_normed")?;
                    self.encode_rms_norm_rows(
                        encoder,
                        input_buffer,
                        norm_weight,
                        &normed_buffer,
                        1,
                        in_dim,
                        eps,
                    )?;
                    Some(normed_buffer)
                }
                None => None,
            };
            let matmul_input = normed_buffer.as_ref().map_or(input_buffer, |buffer| buffer);
            self.encode_matmul_weight_buffers(
                encoder,
                matmul_input,
                1,
                in_dim,
                &weights.in_proj,
                &in_proj_buffer,
                false,
            )?;
        }
        let z_offset = byte_offset_f32(conv_dim, "linear-attn z offset")?;
        let beta_offset = byte_offset_f32(
            conv_dim.checked_add(value_dim).ok_or_else(|| {
                InferError::Dimension("linear-attn beta offset déborde".to_string())
            })?,
            "linear-attn beta offset",
        )?;
        let gate_offset = byte_offset_f32(
            conv_dim
                .checked_add(value_dim)
                .and_then(|value| value.checked_add(spec.num_value_heads))
                .ok_or_else(|| {
                    InferError::Dimension("linear-attn gate offset déborde".to_string())
                })?,
            "linear-attn gate offset",
        )?;
        if linear_conv_norm_fused_enabled()
            && !use_inv_delta
            && spec.conv_kernel_dim == 4
            && spec.key_head_dim == 128
            && spec.value_head_dim == 128
        {
            self.encode_linear_attn_conv_norm_gates_k4_dk128_with_offsets(
                encoder,
                &in_proj_buffer,
                0,
                &in_proj_buffer,
                beta_offset,
                &in_proj_buffer,
                gate_offset,
                &weights.conv_weight,
                &state.conv,
                &weights.a_log,
                &weights.dt_bias,
                &conv_out_buffer,
                &q_norm_buffer,
                &k_norm_buffer,
                &beta_buffer,
                &decay_buffer,
                spec,
            )?;
        } else {
            self.encode_linear_attn_conv_with_offset(
                encoder,
                &in_proj_buffer,
                0,
                &weights.conv_weight,
                &state.conv,
                &conv_out_buffer,
                conv_dim,
                spec.conv_kernel_dim,
            )?;
            if use_inv_delta {
                let q_inv_buffer =
                    self.private_f32_buffer(spec.num_key_heads, "linear_attn_q_inv")?;
                let k_inv_buffer =
                    self.private_f32_buffer(spec.num_key_heads, "linear_attn_k_inv")?;
                self.encode_linear_attn_norm_gates_inv_dk128_with_offsets(
                    encoder,
                    &conv_out_buffer,
                    &in_proj_buffer,
                    beta_offset,
                    &in_proj_buffer,
                    gate_offset,
                    &weights.a_log,
                    &weights.dt_bias,
                    &q_inv_buffer,
                    &k_inv_buffer,
                    &beta_buffer,
                    &decay_buffer,
                    spec,
                )?;
                self.encode_linear_attn_gated_delta_inv_dk128(
                    encoder,
                    &conv_out_buffer,
                    &q_inv_buffer,
                    &k_inv_buffer,
                    &beta_buffer,
                    &decay_buffer,
                    &state.ssm,
                    state.ssm_bf16,
                    &y_buffer,
                    spec,
                )?;
            } else {
                self.encode_linear_attn_norm_gates_with_offsets(
                    encoder,
                    &conv_out_buffer,
                    &in_proj_buffer,
                    beta_offset,
                    &in_proj_buffer,
                    gate_offset,
                    &weights.a_log,
                    &weights.dt_bias,
                    &q_norm_buffer,
                    &k_norm_buffer,
                    &beta_buffer,
                    &decay_buffer,
                    spec,
                )?;
            }
        }
        if !use_inv_delta {
            self.encode_linear_attn_gated_delta(
                encoder,
                &conv_out_buffer,
                &q_norm_buffer,
                &k_norm_buffer,
                &beta_buffer,
                &decay_buffer,
                &state.ssm,
                state.ssm_bf16,
                &y_buffer,
                spec,
            )?;
        }
        self.encode_linear_attn_rms_gate_with_offset(
            encoder,
            &y_buffer,
            &in_proj_buffer,
            z_offset,
            &weights.norm_weight,
            &gated_buffer,
            spec,
        )?;
        self.encode_matmul_weight_buffers(
            encoder,
            &gated_buffer,
            1,
            value_dim,
            &weights.out_proj,
            output_buffer,
            false,
        )?;
        Ok(())
    }

    /// Variante dense résidente par paires de projections concaténées quand les
    /// poids OptiQ ont le même format quantifié.
    #[expect(
        clippy::too_many_arguments,
        reason = "mirror des paramètres du pas linear-attn dense (poids + dims)"
    )]
    pub(crate) fn encode_linear_attn_resident_dense_buffers(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input_buffer: &BufferRef,
        input_norm: Option<(&BufferRef, f32)>,
        output_buffer: &BufferRef,
        weights: &MetalLinearAttnResidentDenseWeights,
        state: &LinearAttentionMetalState,
        spec: LinearAttentionStepSpec,
        dims: LinearAttnResidentDims,
    ) -> Result<()> {
        if linear_full_concat_enabled() {
            if let Some(full) = weights.full.as_ref() {
                return self.encode_linear_attn_resident_buffers(
                    encoder,
                    input_buffer,
                    input_norm,
                    output_buffer,
                    full,
                    state,
                    spec,
                    dims,
                );
            }
        }
        let LinearAttnResidentDims {
            in_dim,
            conv_dim,
            value_dim,
            key_dim,
        } = dims;
        enum PairOutput {
            Concat(Buffer),
            Split { first: Buffer, second: Buffer },
        }

        let conv_out_buffer = self.private_f32_buffer(conv_dim, "linear_attn_conv_out")?;
        let q_norm_buffer = self.private_f32_buffer(key_dim, "linear_attn_q_norm")?;
        let k_norm_buffer = self.private_f32_buffer(key_dim, "linear_attn_k_norm")?;
        let beta_buffer = self.private_f32_buffer(spec.num_value_heads, "linear_attn_beta")?;
        let decay_buffer = self.private_f32_buffer(spec.num_value_heads, "linear_attn_decay")?;
        let y_buffer = self.private_f32_buffer(value_dim, "linear_attn_y")?;
        let gated_buffer = self.private_f32_buffer(value_dim, "linear_attn_gated")?;

        let qkv_z_concat_len = conv_dim
            .checked_add(value_dim)
            .ok_or_else(|| InferError::Dimension("linear-attn qkv+z déborde".to_string()))?;
        let beta_gate_concat_len = spec
            .num_value_heads
            .checked_mul(2)
            .ok_or_else(|| InferError::Dimension("linear-attn beta+gate déborde".to_string()))?;
        let z_beta_gate_concat_len = value_dim
            .checked_add(beta_gate_concat_len)
            .ok_or_else(|| InferError::Dimension("linear-attn z+beta+gate déborde".to_string()))?;

        let normed_buffer = match input_norm {
            Some((norm_weight, eps)) => {
                let normed = self.private_f32_buffer(in_dim, "linear_attn_normed")?;
                self.encode_rms_norm_rows(
                    encoder,
                    input_buffer,
                    norm_weight,
                    &normed,
                    1,
                    in_dim,
                    eps,
                )?;
                Some(normed)
            }
            None => None,
        };
        let matmul_input = normed_buffer.as_ref().map_or(input_buffer, |buffer| buffer);

        if let (
            true,
            MetalLinearAttnResidentPairWeights::Split {
                first: qkv_weight, ..
            },
            Some(z_beta_gate_weight),
        ) = (
            linear_z_beta_gate_enabled(),
            &weights.qkv_z,
            weights.z_beta_gate.as_ref(),
        ) {
            let qkv_buffer = self.private_f32_buffer(conv_dim, "linear_attn_qkv")?;
            let z_beta_gate_buffer =
                self.private_f32_buffer(z_beta_gate_concat_len, "linear_attn_z_beta_gate")?;
            let (qkv_dim, z_beta_gate_dim) = if linear_pair_barrier_coalesce_enabled() {
                let barrier_guard = suspend_dispatch_barrier_scope();
                let qkv_dim = self.encode_matmul_weight_buffers(
                    encoder,
                    matmul_input,
                    1,
                    in_dim,
                    qkv_weight,
                    &qkv_buffer,
                    false,
                )?;
                let z_beta_gate_dim = self.encode_matmul_weight_buffers(
                    encoder,
                    matmul_input,
                    1,
                    in_dim,
                    z_beta_gate_weight,
                    &z_beta_gate_buffer,
                    false,
                )?;
                drop(barrier_guard);
                memory_barrier_buffers(encoder);
                (qkv_dim, z_beta_gate_dim)
            } else {
                let qkv_dim = self.encode_matmul_weight_buffers(
                    encoder,
                    matmul_input,
                    1,
                    in_dim,
                    qkv_weight,
                    &qkv_buffer,
                    false,
                )?;
                let z_beta_gate_dim = self.encode_matmul_weight_buffers(
                    encoder,
                    matmul_input,
                    1,
                    in_dim,
                    z_beta_gate_weight,
                    &z_beta_gate_buffer,
                    false,
                )?;
                (qkv_dim, z_beta_gate_dim)
            };
            if qkv_dim != conv_dim {
                return Err(InferError::Dimension(format!(
                    "linear-attn qkv concat attendu {conv_dim}, reçu {qkv_dim}"
                )));
            }
            if z_beta_gate_dim != z_beta_gate_concat_len {
                return Err(InferError::Dimension(format!(
                    "linear-attn z+beta+gate attendu {z_beta_gate_concat_len}, reçu {z_beta_gate_dim}"
                )));
            }
            self.encode_linear_attn_conv(
                encoder,
                &qkv_buffer,
                &weights.conv_weight,
                &state.conv,
                &conv_out_buffer,
                conv_dim,
                spec.conv_kernel_dim,
            )?;
            let beta_offset = byte_offset_f32(value_dim, "linear-attn zbg beta offset")?;
            let gate_offset = byte_offset_f32(
                value_dim.checked_add(spec.num_value_heads).ok_or_else(|| {
                    InferError::Dimension("linear-attn zbg gate offset déborde".to_string())
                })?,
                "linear-attn zbg gate offset",
            )?;
            self.encode_linear_attn_norm_gates_with_offsets(
                encoder,
                &conv_out_buffer,
                &z_beta_gate_buffer,
                beta_offset,
                &z_beta_gate_buffer,
                gate_offset,
                &weights.a_log,
                &weights.dt_bias,
                &q_norm_buffer,
                &k_norm_buffer,
                &beta_buffer,
                &decay_buffer,
                spec,
            )?;
            self.encode_linear_attn_gated_delta(
                encoder,
                &conv_out_buffer,
                &q_norm_buffer,
                &k_norm_buffer,
                &beta_buffer,
                &decay_buffer,
                &state.ssm,
                state.ssm_bf16,
                &y_buffer,
                spec,
            )?;
            self.encode_linear_attn_rms_gate_with_offset(
                encoder,
                &y_buffer,
                &z_beta_gate_buffer,
                0,
                &weights.norm_weight,
                &gated_buffer,
                spec,
            )?;
            self.encode_matmul_weight_buffers(
                encoder,
                &gated_buffer,
                1,
                value_dim,
                &weights.out_proj,
                output_buffer,
                false,
            )?;
            return Ok(());
        }

        let coalesce_pair_barriers = linear_pair_barrier_coalesce_enabled();
        let barrier_guard = if coalesce_pair_barriers {
            Some(suspend_dispatch_barrier_scope())
        } else {
            None
        };

        let qkv_z = match &weights.qkv_z {
            MetalLinearAttnResidentPairWeights::Concat(weight) => {
                let buffer = self.private_f32_buffer(qkv_z_concat_len, "linear_attn_qkv_z")?;
                self.encode_matmul_weight_buffers(
                    encoder,
                    matmul_input,
                    1,
                    in_dim,
                    weight,
                    &buffer,
                    false,
                )?;
                PairOutput::Concat(buffer)
            }
            MetalLinearAttnResidentPairWeights::Split { first, second } => {
                let qkv_buffer = self.private_f32_buffer(conv_dim, "linear_attn_qkv")?;
                let z_buffer = self.private_f32_buffer(value_dim, "linear_attn_z")?;
                self.encode_matmul_weight_buffers(
                    encoder,
                    matmul_input,
                    1,
                    in_dim,
                    first,
                    &qkv_buffer,
                    false,
                )?;
                self.encode_matmul_weight_buffers(
                    encoder,
                    matmul_input,
                    1,
                    in_dim,
                    second,
                    &z_buffer,
                    false,
                )?;
                PairOutput::Split {
                    first: qkv_buffer,
                    second: z_buffer,
                }
            }
        };

        let beta_gate = match &weights.beta_gate {
            MetalLinearAttnResidentPairWeights::Concat(weight) => {
                let buffer =
                    self.private_f32_buffer(beta_gate_concat_len, "linear_attn_beta_gate_input")?;
                self.encode_matmul_weight_buffers(
                    encoder,
                    matmul_input,
                    1,
                    in_dim,
                    weight,
                    &buffer,
                    false,
                )?;
                PairOutput::Concat(buffer)
            }
            MetalLinearAttnResidentPairWeights::Split { first, second } => {
                let beta_input_buffer =
                    self.private_f32_buffer(spec.num_value_heads, "linear_attn_beta_input")?;
                let gate_input_buffer =
                    self.private_f32_buffer(spec.num_value_heads, "linear_attn_gate_input")?;
                self.encode_matmul_weight_buffers(
                    encoder,
                    matmul_input,
                    1,
                    in_dim,
                    first,
                    &beta_input_buffer,
                    false,
                )?;
                self.encode_matmul_weight_buffers(
                    encoder,
                    matmul_input,
                    1,
                    in_dim,
                    second,
                    &gate_input_buffer,
                    false,
                )?;
                PairOutput::Split {
                    first: beta_input_buffer,
                    second: gate_input_buffer,
                }
            }
        };
        drop(barrier_guard);
        if coalesce_pair_barriers {
            memory_barrier_buffers(encoder);
        }

        match &qkv_z {
            PairOutput::Concat(buffer) => self.encode_linear_attn_conv_with_offset(
                encoder,
                buffer,
                0,
                &weights.conv_weight,
                &state.conv,
                &conv_out_buffer,
                conv_dim,
                spec.conv_kernel_dim,
            )?,
            PairOutput::Split { first, .. } => self.encode_linear_attn_conv(
                encoder,
                first,
                &weights.conv_weight,
                &state.conv,
                &conv_out_buffer,
                conv_dim,
                spec.conv_kernel_dim,
            )?,
        }
        match &beta_gate {
            PairOutput::Concat(buffer) => {
                let gate_offset = byte_offset_f32(spec.num_value_heads, "linear-attn gate offset")?;
                self.encode_linear_attn_norm_gates_with_offsets(
                    encoder,
                    &conv_out_buffer,
                    buffer,
                    0,
                    buffer,
                    gate_offset,
                    &weights.a_log,
                    &weights.dt_bias,
                    &q_norm_buffer,
                    &k_norm_buffer,
                    &beta_buffer,
                    &decay_buffer,
                    spec,
                )?;
            }
            PairOutput::Split { first, second } => self.encode_linear_attn_norm_gates(
                encoder,
                &conv_out_buffer,
                first,
                second,
                &weights.a_log,
                &weights.dt_bias,
                &q_norm_buffer,
                &k_norm_buffer,
                &beta_buffer,
                &decay_buffer,
                spec,
            )?,
        }
        self.encode_linear_attn_gated_delta(
            encoder,
            &conv_out_buffer,
            &q_norm_buffer,
            &k_norm_buffer,
            &beta_buffer,
            &decay_buffer,
            &state.ssm,
            state.ssm_bf16,
            &y_buffer,
            spec,
        )?;
        match &qkv_z {
            PairOutput::Concat(buffer) => {
                let z_offset = byte_offset_f32(conv_dim, "linear-attn z offset")?;
                self.encode_linear_attn_rms_gate_with_offset(
                    encoder,
                    &y_buffer,
                    buffer,
                    z_offset,
                    &weights.norm_weight,
                    &gated_buffer,
                    spec,
                )?;
            }
            PairOutput::Split { second, .. } => self.encode_linear_attn_rms_gate(
                encoder,
                &y_buffer,
                second,
                &weights.norm_weight,
                &gated_buffer,
                spec,
            )?,
        }
        self.encode_matmul_weight_buffers(
            encoder,
            &gated_buffer,
            1,
            value_dim,
            &weights.out_proj,
            output_buffer,
            false,
        )?;
        Ok(())
    }

    fn profile_linear_attn_full_segments(
        &self,
        input: &Tensor,
        input_norm: Option<(&Tensor, f32)>,
        weights: &MetalLinearAttnResidentWeights,
        spec: LinearAttentionStepSpec,
        dims: LinearAttnResidentDims,
        overhead_ms: f64,
    ) -> Result<String> {
        let (batch, in_dim) = input.as_matrix()?;
        if batch != 1 || in_dim != dims.in_dim {
            return Err(InferError::Dimension(format!(
                "split linear-attn full attend [1,{}], reçu {:?}",
                dims.in_dim,
                input.shape()
            )));
        }
        let in_proj_dim = dims
            .conv_dim
            .checked_add(dims.value_dim)
            .and_then(|value| value.checked_add(spec.num_value_heads))
            .and_then(|value| value.checked_add(spec.num_value_heads))
            .ok_or_else(|| {
                InferError::Dimension("split linear-attn full in_proj déborde".to_string())
            })?;
        let conv_len = checked_len(
            dims.conv_dim,
            spec.conv_kernel_dim,
            "split linear-attn full conv state",
        )?;
        let ssm_head_len = checked_len(
            spec.value_head_dim,
            spec.key_head_dim,
            "split linear-attn full ssm head",
        )?;
        let ssm_len = checked_len(
            spec.num_value_heads,
            ssm_head_len,
            "split linear-attn full ssm state",
        )?;
        let conv_seed = vec![0.0_f32; conv_len];
        let ssm_seed = vec![0.0_f32; ssm_len];
        let mut state = None;
        self.ensure_linear_attention_metal_state(
            &mut state,
            &conv_seed,
            &ssm_seed,
            dims.conv_dim,
            conv_len,
            ssm_len,
            spec,
        )?;
        let state = state.as_ref().ok_or_else(|| {
            InferError::Metal("état linear-attn full split non initialisé".to_string())
        })?;

        let input_buffer = self.upload_f32_buffer(input.data(), "linear_split_full_input")?;
        let input_norm_buffer = input_norm
            .map(|(norm, eps)| {
                Ok((
                    self.cached_buffer_from_f32(norm.data(), "linear_split_full_input_norm")?,
                    eps,
                ))
            })
            .transpose()?;
        let normed_buffer = if input_norm_buffer.is_some() {
            Some(self.private_f32_buffer(dims.in_dim, "linear_split_full_normed")?)
        } else {
            None
        };
        let in_proj_buffer = self.private_f32_buffer(in_proj_dim, "linear_split_full_in_proj")?;
        let conv_out_buffer =
            self.private_f32_buffer(dims.conv_dim, "linear_split_full_conv_out")?;
        let q_norm_buffer = self.private_f32_buffer(dims.key_dim, "linear_split_full_q_norm")?;
        let k_norm_buffer = self.private_f32_buffer(dims.key_dim, "linear_split_full_k_norm")?;
        let beta_buffer =
            self.private_f32_buffer(spec.num_value_heads, "linear_split_full_beta")?;
        let decay_buffer =
            self.private_f32_buffer(spec.num_value_heads, "linear_split_full_decay")?;
        let y_buffer = self.private_f32_buffer(dims.value_dim, "linear_split_full_y")?;
        let gated_buffer = self.private_f32_buffer(dims.value_dim, "linear_split_full_gated")?;
        let output_buffer = self.private_f32_buffer(dims.in_dim, "linear_split_full_output")?;
        let z_offset = byte_offset_f32(dims.conv_dim, "split linear-attn full z offset")?;
        let beta_start = dims.conv_dim.checked_add(dims.value_dim).ok_or_else(|| {
            InferError::Dimension("split linear-attn full beta offset déborde".to_string())
        })?;
        let beta_offset = byte_offset_f32(beta_start, "split linear-attn full beta offset")?;
        let gate_start = beta_start
            .checked_add(spec.num_value_heads)
            .ok_or_else(|| {
                InferError::Dimension("split linear-attn full gate offset déborde".to_string())
            })?;
        let gate_offset = byte_offset_f32(gate_start, "split linear-attn full gate offset")?;
        let iters = 64_u32;
        let warmup = 8_u32;

        let full_in_proj_ms = profile_linear_segment(self, warmup, iters, |encoder, _owned| {
            let fused = match &input_norm_buffer {
                Some((norm_weight, eps)) => self
                    .encode_matmul_weight_buffers_rms_prologue(
                        encoder,
                        &input_buffer,
                        norm_weight,
                        *eps,
                        dims.in_dim,
                        &weights.in_proj,
                        &in_proj_buffer,
                    )?
                    .is_some(),
                None => false,
            };
            if !fused {
                let matmul_input = if let (Some((norm_weight, eps)), Some(normed)) =
                    (&input_norm_buffer, &normed_buffer)
                {
                    self.encode_rms_norm_rows(
                        encoder,
                        &input_buffer,
                        norm_weight,
                        normed,
                        1,
                        dims.in_dim,
                        *eps,
                    )?;
                    normed
                } else {
                    &input_buffer
                };
                let out = self.encode_matmul_weight_buffers(
                    encoder,
                    matmul_input,
                    1,
                    dims.in_dim,
                    &weights.in_proj,
                    &in_proj_buffer,
                    false,
                )?;
                if out != in_proj_dim {
                    return Err(InferError::Dimension(format!(
                        "split linear-attn full in_proj sort {out}, attendu {in_proj_dim}"
                    )));
                }
            }
            Ok(())
        })?;
        let conv_ms = profile_linear_segment(self, warmup, iters, |encoder, _owned| {
            self.encode_linear_attn_conv_with_offset(
                encoder,
                &in_proj_buffer,
                0,
                &weights.conv_weight,
                &state.conv,
                &conv_out_buffer,
                dims.conv_dim,
                spec.conv_kernel_dim,
            )
        })?;
        let norm_gates_ms = profile_linear_segment(self, warmup, iters, |encoder, _owned| {
            self.encode_linear_attn_norm_gates_with_offsets(
                encoder,
                &conv_out_buffer,
                &in_proj_buffer,
                beta_offset,
                &in_proj_buffer,
                gate_offset,
                &weights.a_log,
                &weights.dt_bias,
                &q_norm_buffer,
                &k_norm_buffer,
                &beta_buffer,
                &decay_buffer,
                spec,
            )
        })?;
        let delta_ms = profile_linear_segment(self, warmup, iters, |encoder, _owned| {
            self.encode_linear_attn_gated_delta(
                encoder,
                &conv_out_buffer,
                &q_norm_buffer,
                &k_norm_buffer,
                &beta_buffer,
                &decay_buffer,
                &state.ssm,
                state.ssm_bf16,
                &y_buffer,
                spec,
            )
        })?;
        let rms_gate_ms = profile_linear_segment(self, warmup, iters, |encoder, _owned| {
            self.encode_linear_attn_rms_gate_with_offset(
                encoder,
                &y_buffer,
                &in_proj_buffer,
                z_offset,
                &weights.norm_weight,
                &gated_buffer,
                spec,
            )
        })?;
        let out_proj_ms = profile_linear_segment(self, warmup, iters, |encoder, _owned| {
            let out = self.encode_matmul_weight_buffers(
                encoder,
                &gated_buffer,
                1,
                dims.value_dim,
                &weights.out_proj,
                &output_buffer,
                false,
            )?;
            if out != dims.in_dim {
                return Err(InferError::Dimension(format!(
                    "split linear-attn full out_proj sort {out}, attendu {}",
                    dims.in_dim
                )));
            }
            Ok(())
        })?;

        let pure = |segment_ms: f64| (segment_ms - overhead_ms).max(0.0);
        Ok(format!(
            "split linear-attn full ({iters} itér, ms+CB/pur): \
             full_in_proj {full_in_proj_ms:.3}/{in_pure:.3}, \
             conv {conv_ms:.3}/{conv_pure:.3}, norm_gates {norm_gates_ms:.3}/{norm_pure:.3}, \
             delta {delta_ms:.3}/{delta_pure:.3}, rms_gate {rms_gate_ms:.3}/{rms_pure:.3}, \
             out_proj {out_proj_ms:.3}/{out_pure:.3}",
            in_pure = pure(full_in_proj_ms),
            conv_pure = pure(conv_ms),
            norm_pure = pure(norm_gates_ms),
            delta_pure = pure(delta_ms),
            rms_pure = pure(rms_gate_ms),
            out_pure = pure(out_proj_ms),
        ))
    }

    /// Profil segmenté d'un pas linear-attn dense résident.
    ///
    /// Diagnostic uniquement: chaque segment est rejoué dans son propre command
    /// buffer pour classer les coûts. Les temps ne doivent pas être additionnés
    /// comme un temps de couche réel, mais ils indiquent les noyaux à fusionner ou
    /// spécialiser.
    pub(crate) fn profile_linear_attn_dense_segments(
        &self,
        input: &Tensor,
        input_norm: Option<(&Tensor, f32)>,
        weights: LinearAttnResidentWeights<'_>,
        spec: LinearAttentionStepSpec,
        dims: LinearAttnResidentDims,
        overhead_ms: f64,
    ) -> Result<String> {
        let (batch, in_dim) = input.as_matrix()?;
        if batch != 1 || in_dim != dims.in_dim {
            return Err(InferError::Dimension(format!(
                "split linear-attn attend [1,{}], reçu {:?}",
                dims.in_dim,
                input.shape()
            )));
        }
        let weights = self.resolve_linear_attn_resident_dense_weights(weights)?;
        if let Some(full) = weights.full.as_ref() {
            return self.profile_linear_attn_full_segments(
                input,
                input_norm,
                full,
                spec,
                dims,
                overhead_ms,
            );
        }
        let (qkv_weight, z_beta_gate_weight) = match (&weights.qkv_z, weights.z_beta_gate.as_ref())
        {
            (
                MetalLinearAttnResidentPairWeights::Split {
                    first: qkv_weight, ..
                },
                Some(z_beta_gate_weight),
            ) if linear_z_beta_gate_enabled() => (qkv_weight, z_beta_gate_weight),
            _ => {
                return Err(InferError::Config(
                    "split linear-attn implémenté pour qkv split + z_beta_gate".to_string(),
                ))
            }
        };

        let conv_len = checked_len(
            dims.conv_dim,
            spec.conv_kernel_dim,
            "split linear-attn conv state",
        )?;
        let ssm_head_len = checked_len(
            spec.value_head_dim,
            spec.key_head_dim,
            "split linear-attn ssm head",
        )?;
        let ssm_len = checked_len(
            spec.num_value_heads,
            ssm_head_len,
            "split linear-attn ssm state",
        )?;
        let conv_seed = vec![0.0_f32; conv_len];
        let ssm_seed = vec![0.0_f32; ssm_len];
        let mut state = None;
        self.ensure_linear_attention_metal_state(
            &mut state,
            &conv_seed,
            &ssm_seed,
            dims.conv_dim,
            conv_len,
            ssm_len,
            spec,
        )?;
        let state = state.as_ref().ok_or_else(|| {
            InferError::Metal("état linear-attn split non initialisé".to_string())
        })?;

        let input_buffer = self.upload_f32_buffer(input.data(), "linear_split_input")?;
        let input_norm_buffer = input_norm
            .map(|(norm, eps)| {
                Ok((
                    self.cached_buffer_from_f32(norm.data(), "linear_split_input_norm")?,
                    eps,
                ))
            })
            .transpose()?;
        let normed_buffer = if input_norm_buffer.is_some() {
            Some(self.private_f32_buffer(dims.in_dim, "linear_split_normed")?)
        } else {
            None
        };
        let matmul_input = match normed_buffer.as_ref() {
            Some(buffer) => buffer,
            None => &input_buffer,
        };

        let qkv_buffer = self.private_f32_buffer(dims.conv_dim, "linear_split_qkv")?;
        let beta_gate_len =
            checked_len(spec.num_value_heads, 2, "split linear-attn beta_gate len")?;
        let z_beta_gate_len = dims.value_dim.checked_add(beta_gate_len).ok_or_else(|| {
            InferError::Dimension("split linear-attn z_beta_gate déborde".to_string())
        })?;
        let z_beta_gate_buffer =
            self.private_f32_buffer(z_beta_gate_len, "linear_split_z_beta_gate")?;
        let conv_out_buffer = self.private_f32_buffer(dims.conv_dim, "linear_split_conv_out")?;
        let q_norm_buffer = self.private_f32_buffer(dims.key_dim, "linear_split_q_norm")?;
        let k_norm_buffer = self.private_f32_buffer(dims.key_dim, "linear_split_k_norm")?;
        let beta_buffer = self.private_f32_buffer(spec.num_value_heads, "linear_split_beta")?;
        let decay_buffer = self.private_f32_buffer(spec.num_value_heads, "linear_split_decay")?;
        let y_buffer = self.private_f32_buffer(dims.value_dim, "linear_split_y")?;
        let gated_buffer = self.private_f32_buffer(dims.value_dim, "linear_split_gated")?;
        let output_buffer = self.private_f32_buffer(dims.in_dim, "linear_split_output")?;
        let iters = 64_u32;
        let warmup = 8_u32;

        let input_norm_ms = if let (Some((norm_weight, eps)), Some(normed)) =
            (&input_norm_buffer, &normed_buffer)
        {
            profile_linear_segment(self, warmup, iters, |encoder, _owned| {
                self.encode_rms_norm_rows(
                    encoder,
                    &input_buffer,
                    norm_weight,
                    normed,
                    1,
                    dims.in_dim,
                    *eps,
                )
            })?
        } else {
            0.0
        };
        let qkv_ms = profile_linear_segment(self, warmup, iters, |encoder, owned| {
            let out = self.encode_matmul_weight_buffers(
                encoder,
                matmul_input,
                1,
                dims.in_dim,
                qkv_weight,
                &qkv_buffer,
                false,
            )?;
            if out != dims.conv_dim {
                return Err(InferError::Dimension(format!(
                    "split linear-attn qkv sort {out}, attendu {}",
                    dims.conv_dim
                )));
            }
            let _ = owned;
            Ok(())
        })?;
        let z_beta_gate_ms = profile_linear_segment(self, warmup, iters, |encoder, owned| {
            let out = self.encode_matmul_weight_buffers(
                encoder,
                matmul_input,
                1,
                dims.in_dim,
                z_beta_gate_weight,
                &z_beta_gate_buffer,
                false,
            )?;
            if out != z_beta_gate_len {
                return Err(InferError::Dimension(format!(
                    "split linear-attn zbg sort {out}, attendu {z_beta_gate_len}"
                )));
            }
            let _ = owned;
            Ok(())
        })?;
        let conv_ms = profile_linear_segment(self, warmup, iters, |encoder, _owned| {
            self.encode_linear_attn_conv(
                encoder,
                &qkv_buffer,
                &weights.conv_weight,
                &state.conv,
                &conv_out_buffer,
                dims.conv_dim,
                spec.conv_kernel_dim,
            )
        })?;
        let beta_offset = byte_offset_f32(dims.value_dim, "split linear-attn beta offset")?;
        let gate_start = dims
            .value_dim
            .checked_add(spec.num_value_heads)
            .ok_or_else(|| {
                InferError::Dimension("split linear-attn gate offset déborde".to_string())
            })?;
        let gate_offset = byte_offset_f32(gate_start, "split linear-attn gate offset")?;
        let norm_gates_ms = profile_linear_segment(self, warmup, iters, |encoder, _owned| {
            self.encode_linear_attn_norm_gates_with_offsets(
                encoder,
                &conv_out_buffer,
                &z_beta_gate_buffer,
                beta_offset,
                &z_beta_gate_buffer,
                gate_offset,
                &weights.a_log,
                &weights.dt_bias,
                &q_norm_buffer,
                &k_norm_buffer,
                &beta_buffer,
                &decay_buffer,
                spec,
            )
        })?;
        let delta_ms = profile_linear_segment(self, warmup, iters, |encoder, _owned| {
            self.encode_linear_attn_gated_delta(
                encoder,
                &conv_out_buffer,
                &q_norm_buffer,
                &k_norm_buffer,
                &beta_buffer,
                &decay_buffer,
                &state.ssm,
                state.ssm_bf16,
                &y_buffer,
                spec,
            )
        })?;
        let rms_gate_ms = profile_linear_segment(self, warmup, iters, |encoder, _owned| {
            self.encode_linear_attn_rms_gate_with_offset(
                encoder,
                &y_buffer,
                &z_beta_gate_buffer,
                0,
                &weights.norm_weight,
                &gated_buffer,
                spec,
            )
        })?;
        let out_proj_ms = profile_linear_segment(self, warmup, iters, |encoder, owned| {
            let out = self.encode_matmul_weight_buffers(
                encoder,
                &gated_buffer,
                1,
                dims.value_dim,
                &weights.out_proj,
                &output_buffer,
                false,
            )?;
            if out != dims.in_dim {
                return Err(InferError::Dimension(format!(
                    "split linear-attn out_proj sort {out}, attendu {}",
                    dims.in_dim
                )));
            }
            let _ = owned;
            Ok(())
        })?;

        let pure = |segment_ms: f64| (segment_ms - overhead_ms).max(0.0);
        Ok(format!(
            "split linear-attn fixed-input ({iters} itér, ms+CB/pur): \
             input_norm {input_norm_ms:.3}/{input_norm_pure:.3}, \
             qkv {qkv_ms:.3}/{qkv_pure:.3}, zbg {z_beta_gate_ms:.3}/{zbg_pure:.3}, \
             conv {conv_ms:.3}/{conv_pure:.3}, norm_gates {norm_gates_ms:.3}/{norm_pure:.3}, \
             delta {delta_ms:.3}/{delta_pure:.3}, rms_gate {rms_gate_ms:.3}/{rms_pure:.3}, \
             out_proj {out_proj_ms:.3}/{out_pure:.3}",
            input_norm_pure = pure(input_norm_ms),
            qkv_pure = pure(qkv_ms),
            zbg_pure = pure(z_beta_gate_ms),
            conv_pure = pure(conv_ms),
            norm_pure = pure(norm_gates_ms),
            delta_pure = pure(delta_ms),
            rms_pure = pure(rms_gate_ms),
            out_pure = pure(out_proj_ms),
        ))
    }

    /// Variante résidente dense sur plusieurs positions contiguës. Les matmuls
    /// lisent les poids une seule fois pour `rows`, puis le scan conv/SSM reste
    /// strictement ordonné position par position.
    #[expect(
        clippy::too_many_arguments,
        reason = "mirror batché du pas linear-attn dense résident"
    )]
    pub(crate) fn encode_linear_attn_resident_dense_buffers_rows(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input_buffer: &BufferRef,
        input_norm: Option<(&BufferRef, f32)>,
        output_buffer: &BufferRef,
        rows: usize,
        weights: &MetalLinearAttnResidentDenseWeights,
        state: &LinearAttentionMetalState,
        captures: Option<&[LinearAttentionMetalState]>,
        spec: LinearAttentionStepSpec,
        dims: LinearAttnResidentDims,
    ) -> Result<()> {
        if rows == 0 {
            return Err(InferError::Dimension(
                "linear-attn résident rows vide".to_string(),
            ));
        }
        if rows == 1 {
            self.encode_linear_attn_resident_dense_buffers(
                encoder,
                input_buffer,
                input_norm,
                output_buffer,
                weights,
                state,
                spec,
                dims,
            )?;
            if let Some(captures) = captures {
                let capture = captures.first().ok_or_else(|| {
                    InferError::Dimension("capture linear-attn row 0 absente".to_string())
                })?;
                self.encode_capture_linear_attn_state(encoder, state, capture)?;
            }
            return Ok(());
        }
        if let Some(captures) = captures {
            let capture_rows = rows.saturating_sub(1);
            if captures.len() < capture_rows {
                return Err(InferError::Dimension(format!(
                    "captures linear-attn rows={} < rows utiles={capture_rows}",
                    captures.len(),
                )));
            }
        }
        let LinearAttnResidentDims {
            in_dim,
            conv_dim,
            value_dim,
            key_dim,
        } = dims;
        enum PairOutput {
            Concat {
                buffer: Buffer,
                stride: usize,
            },
            Split {
                first: Buffer,
                first_stride: usize,
                second: Buffer,
                second_stride: usize,
            },
        }

        let normed_buffer = match input_norm {
            Some((norm_weight, eps)) => {
                let normed_len = checked_len(rows, in_dim, "linear-attn rows normed")?;
                let normed = self.private_f32_buffer(normed_len, "linear_attn_rows_normed")?;
                self.encode_rms_norm_rows(
                    encoder,
                    input_buffer,
                    norm_weight,
                    &normed,
                    rows,
                    in_dim,
                    eps,
                )?;
                Some(normed)
            }
            None => None,
        };
        let matmul_input = normed_buffer.as_ref().map_or(input_buffer, |buffer| buffer);
        let conv_out_buffer = self.private_f32_buffer(conv_dim, "linear_attn_rows_conv_out")?;
        let q_norm_buffer = self.private_f32_buffer(key_dim, "linear_attn_rows_q_norm")?;
        let k_norm_buffer = self.private_f32_buffer(key_dim, "linear_attn_rows_k_norm")?;
        let beta_buffer = self.private_f32_buffer(spec.num_value_heads, "linear_attn_rows_beta")?;
        let decay_buffer =
            self.private_f32_buffer(spec.num_value_heads, "linear_attn_rows_decay")?;
        let y_buffer = self.private_f32_buffer(value_dim, "linear_attn_rows_y")?;
        let gated_row_buffer = self.private_f32_buffer(value_dim, "linear_attn_rows_gated_row")?;
        let gated_batch_buffer = self.private_f32_buffer(
            checked_len(rows, value_dim, "linear-attn rows gated")?,
            "linear_attn_rows_gated",
        )?;

        let qkv_z_concat_len = conv_dim
            .checked_add(value_dim)
            .ok_or_else(|| InferError::Dimension("linear-attn qkv+z déborde".to_string()))?;
        let beta_gate_concat_len = spec
            .num_value_heads
            .checked_mul(2)
            .ok_or_else(|| InferError::Dimension("linear-attn beta+gate déborde".to_string()))?;

        let qkv_z = match &weights.qkv_z {
            MetalLinearAttnResidentPairWeights::Concat(weight) => {
                let buffer = self.private_f32_buffer(
                    checked_len(rows, qkv_z_concat_len, "linear-attn rows qkv+z")?,
                    "linear_attn_rows_qkv_z",
                )?;
                self.encode_matmul_weight_buffers(
                    encoder,
                    matmul_input,
                    rows,
                    in_dim,
                    weight,
                    &buffer,
                    false,
                )?;
                PairOutput::Concat {
                    buffer,
                    stride: qkv_z_concat_len,
                }
            }
            MetalLinearAttnResidentPairWeights::Split { first, second } => {
                let qkv_buffer = self.private_f32_buffer(
                    checked_len(rows, conv_dim, "linear-attn rows qkv")?,
                    "linear_attn_rows_qkv",
                )?;
                let z_buffer = self.private_f32_buffer(
                    checked_len(rows, value_dim, "linear-attn rows z")?,
                    "linear_attn_rows_z",
                )?;
                self.encode_matmul_weight_buffers(
                    encoder,
                    matmul_input,
                    rows,
                    in_dim,
                    first,
                    &qkv_buffer,
                    false,
                )?;
                self.encode_matmul_weight_buffers(
                    encoder,
                    matmul_input,
                    rows,
                    in_dim,
                    second,
                    &z_buffer,
                    false,
                )?;
                PairOutput::Split {
                    first: qkv_buffer,
                    first_stride: conv_dim,
                    second: z_buffer,
                    second_stride: value_dim,
                }
            }
        };

        let beta_gate = match &weights.beta_gate {
            MetalLinearAttnResidentPairWeights::Concat(weight) => {
                let buffer = self.private_f32_buffer(
                    checked_len(rows, beta_gate_concat_len, "linear-attn rows beta+gate")?,
                    "linear_attn_rows_beta_gate_input",
                )?;
                self.encode_matmul_weight_buffers(
                    encoder,
                    matmul_input,
                    rows,
                    in_dim,
                    weight,
                    &buffer,
                    false,
                )?;
                PairOutput::Concat {
                    buffer,
                    stride: beta_gate_concat_len,
                }
            }
            MetalLinearAttnResidentPairWeights::Split { first, second } => {
                let beta_input_buffer = self.private_f32_buffer(
                    checked_len(rows, spec.num_value_heads, "linear-attn rows beta input")?,
                    "linear_attn_rows_beta_input",
                )?;
                let gate_input_buffer = self.private_f32_buffer(
                    checked_len(rows, spec.num_value_heads, "linear-attn rows gate input")?,
                    "linear_attn_rows_gate_input",
                )?;
                self.encode_matmul_weight_buffers(
                    encoder,
                    matmul_input,
                    rows,
                    in_dim,
                    first,
                    &beta_input_buffer,
                    false,
                )?;
                self.encode_matmul_weight_buffers(
                    encoder,
                    matmul_input,
                    rows,
                    in_dim,
                    second,
                    &gate_input_buffer,
                    false,
                )?;
                PairOutput::Split {
                    first: beta_input_buffer,
                    first_stride: spec.num_value_heads,
                    second: gate_input_buffer,
                    second_stride: spec.num_value_heads,
                }
            }
        };

        for row in 0..rows {
            match &qkv_z {
                PairOutput::Concat { buffer, stride } => {
                    let qkv_offset = byte_offset_f32(row * *stride, "linear-attn rows qkv offset")?;
                    self.encode_linear_attn_conv_with_offset(
                        encoder,
                        buffer,
                        qkv_offset,
                        &weights.conv_weight,
                        &state.conv,
                        &conv_out_buffer,
                        conv_dim,
                        spec.conv_kernel_dim,
                    )?;
                }
                PairOutput::Split {
                    first,
                    first_stride,
                    ..
                } => {
                    let qkv_offset =
                        byte_offset_f32(row * *first_stride, "linear-attn rows qkv offset")?;
                    self.encode_linear_attn_conv_with_offset(
                        encoder,
                        first,
                        qkv_offset,
                        &weights.conv_weight,
                        &state.conv,
                        &conv_out_buffer,
                        conv_dim,
                        spec.conv_kernel_dim,
                    )?;
                }
            }
            memory_barrier_buffers(encoder);
            match &beta_gate {
                PairOutput::Concat { buffer, stride } => {
                    let beta_offset =
                        byte_offset_f32(row * *stride, "linear-attn rows beta offset")?;
                    let gate_offset = byte_offset_f32(
                        row * *stride + spec.num_value_heads,
                        "linear-attn rows gate offset",
                    )?;
                    self.encode_linear_attn_norm_gates_with_offsets(
                        encoder,
                        &conv_out_buffer,
                        buffer,
                        beta_offset,
                        buffer,
                        gate_offset,
                        &weights.a_log,
                        &weights.dt_bias,
                        &q_norm_buffer,
                        &k_norm_buffer,
                        &beta_buffer,
                        &decay_buffer,
                        spec,
                    )?;
                }
                PairOutput::Split {
                    first,
                    first_stride,
                    second,
                    second_stride,
                } => {
                    let beta_offset =
                        byte_offset_f32(row * *first_stride, "linear-attn rows beta offset")?;
                    let gate_offset =
                        byte_offset_f32(row * *second_stride, "linear-attn rows gate offset")?;
                    self.encode_linear_attn_norm_gates_with_offsets(
                        encoder,
                        &conv_out_buffer,
                        first,
                        beta_offset,
                        second,
                        gate_offset,
                        &weights.a_log,
                        &weights.dt_bias,
                        &q_norm_buffer,
                        &k_norm_buffer,
                        &beta_buffer,
                        &decay_buffer,
                        spec,
                    )?;
                }
            }
            self.encode_linear_attn_gated_delta(
                encoder,
                &conv_out_buffer,
                &q_norm_buffer,
                &k_norm_buffer,
                &beta_buffer,
                &decay_buffer,
                &state.ssm,
                state.ssm_bf16,
                &y_buffer,
                spec,
            )?;
            if let Some(captures) = captures.filter(|_| row + 1 < rows) {
                self.encode_capture_linear_attn_state(encoder, state, &captures[row])?;
            }
            match &qkv_z {
                PairOutput::Concat { buffer, stride } => {
                    let z_offset =
                        byte_offset_f32(row * *stride + conv_dim, "linear-attn rows z offset")?;
                    self.encode_linear_attn_rms_gate_with_offset(
                        encoder,
                        &y_buffer,
                        buffer,
                        z_offset,
                        &weights.norm_weight,
                        &gated_row_buffer,
                        spec,
                    )?;
                }
                PairOutput::Split {
                    second,
                    second_stride,
                    ..
                } => {
                    let z_offset =
                        byte_offset_f32(row * *second_stride, "linear-attn rows z offset")?;
                    self.encode_linear_attn_rms_gate_with_offset(
                        encoder,
                        &y_buffer,
                        second,
                        z_offset,
                        &weights.norm_weight,
                        &gated_row_buffer,
                        spec,
                    )?;
                }
            }
            let gated_offset = byte_offset_f32(row * value_dim, "linear-attn rows gated offset")?;
            self.encode_copy_with_offsets(
                encoder,
                &gated_row_buffer,
                0,
                &gated_batch_buffer,
                gated_offset,
                value_dim,
            )?;
        }
        self.encode_matmul_weight_buffers(
            encoder,
            &gated_batch_buffer,
            rows,
            value_dim,
            &weights.out_proj,
            output_buffer,
            false,
        )?;
        Ok(())
    }

    pub(super) fn ensure_linear_attention_metal_state(
        &self,
        state: &mut Option<LinearAttentionMetalState>,
        conv_seed: &[f32],
        ssm_seed: &[f32],
        conv_dim: usize,
        conv_len: usize,
        ssm_len: usize,
        spec: LinearAttentionStepSpec,
    ) -> Result<()> {
        let ssm_bf16 = linear_ssm_bf16_enabled()
            && linear_delta_dk128_enabled()
            && !linear_inv_delta_enabled()
            && spec.key_head_dim == 128
            && spec.value_head_dim % 4 == 0;
        let valid = state.as_ref().is_some_and(|state| {
            state.conv_len == conv_len
                && state.ssm_len == ssm_len
                && state.conv_dim == conv_dim
                && state.conv_kernel_dim == spec.conv_kernel_dim
                && state.num_value_heads == spec.num_value_heads
                && state.value_head_dim == spec.value_head_dim
                && state.key_head_dim == spec.key_head_dim
                && state.ssm_bf16 == ssm_bf16
        });
        if valid {
            return Ok(());
        }
        *state = Some(LinearAttentionMetalState {
            conv: self.buffer_from_f32(conv_seed, "linear_attn_resident_conv")?,
            ssm: self.buffer_from_linear_ssm_seed(
                ssm_seed,
                ssm_bf16,
                "linear_attn_resident_ssm",
            )?,
            conv_len,
            ssm_len,
            conv_dim,
            conv_kernel_dim: spec.conv_kernel_dim,
            num_value_heads: spec.num_value_heads,
            value_head_dim: spec.value_head_dim,
            key_head_dim: spec.key_head_dim,
            ssm_bf16,
        });
        Ok(())
    }

    /// Snapshot (copie profonde, blit) de l'état GDN/linear-attn résident.
    ///
    /// MTP : avant un verify spéculatif batché M positions, on capture l'état
    /// (`conv` + `ssm`, ≈ 3.27 Mo/couche) pour pouvoir le rejouer J pas après
    /// acceptation partielle (rollback sans re-matmul). Le blit ≈ 0.4 ms pour les
    /// 48 couches (157 Mo @ ~400 Go/s) → < 1 % d'overhead.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur d'allocation ou de soumission Metal.
    #[allow(dead_code, reason = "câblé au verify spéculatif MTP à l'ÉTAPE 3")]
    pub(crate) fn snapshot_linear_attn_state(
        &self,
        state: &LinearAttentionMetalState,
    ) -> Result<LinearAttentionMetalState> {
        let conv = self.uncached_f32_buffer(state.conv_len, "linear_attn_snapshot_conv")?;
        let ssm = self.uncached_linear_ssm_buffer(
            state.ssm_len,
            state.ssm_bf16,
            "linear_attn_snapshot_ssm",
        )?;
        let command_buffer = self.queue.new_command_buffer();
        let blit = command_buffer.new_blit_command_encoder();
        blit.copy_from_buffer(&state.conv, 0, &conv, 0, byte_len::<f32>(state.conv_len)?);
        blit.copy_from_buffer(&state.ssm, 0, &ssm, 0, linear_ssm_byte_len(state)?);
        blit.end_encoding();
        commit_and_wait(command_buffer)?;
        Ok(LinearAttentionMetalState {
            conv,
            ssm,
            conv_len: state.conv_len,
            ssm_len: state.ssm_len,
            conv_dim: state.conv_dim,
            conv_kernel_dim: state.conv_kernel_dim,
            num_value_heads: state.num_value_heads,
            value_head_dim: state.value_head_dim,
            key_head_dim: state.key_head_dim,
            ssm_bf16: state.ssm_bf16,
        })
    }

    pub(crate) fn snapshot_linear_attn_states(
        &self,
        states: &[Option<&LinearAttentionMetalState>],
    ) -> Result<Vec<Option<LinearAttentionMetalState>>> {
        let mut snapshots = Vec::with_capacity(states.len());
        let command_buffer = self.queue.new_command_buffer();
        let blit = command_buffer.new_blit_command_encoder();
        let mut copied = false;
        for state in states {
            let Some(state) = state else {
                snapshots.push(None);
                continue;
            };
            let conv = self.uncached_f32_buffer(state.conv_len, "linear_attn_snapshot_conv")?;
            let ssm = self.uncached_linear_ssm_buffer(
                state.ssm_len,
                state.ssm_bf16,
                "linear_attn_snapshot_ssm",
            )?;
            blit.copy_from_buffer(&state.conv, 0, &conv, 0, byte_len::<f32>(state.conv_len)?);
            blit.copy_from_buffer(&state.ssm, 0, &ssm, 0, linear_ssm_byte_len(state)?);
            copied = true;
            snapshots.push(Some(LinearAttentionMetalState {
                conv,
                ssm,
                conv_len: state.conv_len,
                ssm_len: state.ssm_len,
                conv_dim: state.conv_dim,
                conv_kernel_dim: state.conv_kernel_dim,
                num_value_heads: state.num_value_heads,
                value_head_dim: state.value_head_dim,
                key_head_dim: state.key_head_dim,
                ssm_bf16: state.ssm_bf16,
            }));
        }
        blit.end_encoding();
        if copied {
            commit_and_wait(command_buffer)?;
        }
        Ok(snapshots)
    }

    #[cfg(any(test, feature = "devtools"))]
    /// Compare deux états linear-attn déjà synchronisés côté GPU.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si les formes divergent ou si un readback échoue.
    pub(crate) fn diff_linear_attn_states(
        &self,
        left: &LinearAttentionMetalState,
        right: &LinearAttentionMetalState,
    ) -> Result<LinearAttentionStateDiff> {
        if left.conv_len != right.conv_len
            || left.ssm_len != right.ssm_len
            || left.ssm_bf16 != right.ssm_bf16
        {
            return Err(InferError::Shape(format!(
                "diff linear-attn: gauche conv={}/ssm={}/bf16={} != droite conv={}/ssm={}/bf16={}",
                left.conv_len,
                left.ssm_len,
                left.ssm_bf16,
                right.conv_len,
                right.ssm_len,
                right.ssm_bf16
            )));
        }
        let left_conv = read_f32_buffer(&left.conv, left.conv_len)?;
        let right_conv = read_f32_buffer(&right.conv, right.conv_len)?;
        let (conv_max_abs, conv_mean_abs) = diff_f32_slices(&left_conv, &right_conv)?;
        let (ssm_max_abs, ssm_mean_abs) = if left.ssm_bf16 {
            let left_words = read_u16_buffer(&left.ssm, left.ssm_len)?;
            let right_words = read_u16_buffer(&right.ssm, right.ssm_len)?;
            let left_ssm = left_words
                .into_iter()
                .map(|word| f32::from_bits(u32::from(word) << 16))
                .collect::<Vec<_>>();
            let right_ssm = right_words
                .into_iter()
                .map(|word| f32::from_bits(u32::from(word) << 16))
                .collect::<Vec<_>>();
            diff_f32_slices(&left_ssm, &right_ssm)?
        } else {
            let left_ssm = read_f32_buffer(&left.ssm, left.ssm_len)?;
            let right_ssm = read_f32_buffer(&right.ssm, right.ssm_len)?;
            diff_f32_slices(&left_ssm, &right_ssm)?
        };
        Ok(LinearAttentionStateDiff {
            conv_max_abs,
            conv_mean_abs,
            ssm_max_abs,
            ssm_mean_abs,
        })
    }

    /// Restaure l'état GDN/linear-attn résident depuis un snapshot (blit inverse,
    /// in-place). Utilisé sur rejet ou acceptation partielle du verify spéculatif.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si les longueurs diffèrent ou si le blit échoue.
    #[allow(dead_code, reason = "câblé au verify spéculatif MTP à l'ÉTAPE 3")]
    pub(crate) fn restore_linear_attn_state(
        &self,
        state: &LinearAttentionMetalState,
        snapshot: &LinearAttentionMetalState,
    ) -> Result<()> {
        if state.conv_len != snapshot.conv_len
            || state.ssm_len != snapshot.ssm_len
            || state.ssm_bf16 != snapshot.ssm_bf16
        {
            return Err(InferError::Shape(format!(
                "restore linear-attn: snapshot conv={}/ssm={}/bf16={} ≠ état conv={}/ssm={}/bf16={}",
                snapshot.conv_len,
                snapshot.ssm_len,
                snapshot.ssm_bf16,
                state.conv_len,
                state.ssm_len,
                state.ssm_bf16
            )));
        }
        let command_buffer = self.queue.new_command_buffer();
        let blit = command_buffer.new_blit_command_encoder();
        blit.copy_from_buffer(
            &snapshot.conv,
            0,
            &state.conv,
            0,
            byte_len::<f32>(state.conv_len)?,
        );
        blit.copy_from_buffer(&snapshot.ssm, 0, &state.ssm, 0, linear_ssm_byte_len(state)?);
        blit.end_encoding();
        commit_and_wait(command_buffer)?;
        Ok(())
    }

    pub(crate) fn restore_linear_attn_states(
        &self,
        pairs: &[(&LinearAttentionMetalState, &LinearAttentionMetalState)],
    ) -> Result<()> {
        if pairs.is_empty() {
            return Ok(());
        }
        let command_buffer = self.queue.new_command_buffer();
        let blit = command_buffer.new_blit_command_encoder();
        for (state, snapshot) in pairs {
            if state.conv_len != snapshot.conv_len
                || state.ssm_len != snapshot.ssm_len
                || state.ssm_bf16 != snapshot.ssm_bf16
            {
                return Err(InferError::Shape(format!(
                    "restore linear-attn batch: snapshot conv={}/ssm={}/bf16={} ≠ état conv={}/ssm={}/bf16={}",
                    snapshot.conv_len,
                    snapshot.ssm_len,
                    snapshot.ssm_bf16,
                    state.conv_len,
                    state.ssm_len,
                    state.ssm_bf16
                )));
            }
            blit.copy_from_buffer(
                &snapshot.conv,
                0,
                &state.conv,
                0,
                byte_len::<f32>(state.conv_len)?,
            );
            blit.copy_from_buffer(&snapshot.ssm, 0, &state.ssm, 0, linear_ssm_byte_len(state)?);
        }
        blit.end_encoding();
        commit_and_wait(command_buffer)
    }
}

#[cfg(any(test, feature = "devtools"))]
fn diff_f32_slices(left: &[f32], right: &[f32]) -> Result<(f32, f32)> {
    if left.len() != right.len() {
        return Err(InferError::Dimension(format!(
            "diff f32: gauche={} droite={}",
            left.len(),
            right.len()
        )));
    }
    let mut max_abs = 0.0_f32;
    let mut sum_abs = 0.0_f64;
    for (left, right) in left.iter().zip(right.iter()) {
        let delta = (*left - *right).abs();
        max_abs = max_abs.max(delta);
        sum_abs += f64::from(delta);
    }
    let mean_abs = if left.is_empty() {
        0.0
    } else {
        (sum_abs / left.len() as f64) as f32
    };
    Ok((max_abs, mean_abs))
}

fn linear_ssm_byte_len(state: &LinearAttentionMetalState) -> Result<u64> {
    if state.ssm_bf16 {
        return byte_len::<u16>(state.ssm_len);
    }
    byte_len::<f32>(state.ssm_len)
}

fn profile_linear_segment<F>(
    metal: &MetalExecutor,
    warmup: u32,
    iters: u32,
    mut encode: F,
) -> Result<f64>
where
    F: FnMut(&ComputeCommandEncoderRef, &mut Vec<Buffer>) -> Result<()>,
{
    if iters == 0 {
        return Err(InferError::Dimension(
            "linear-attn split iters nul".to_string(),
        ));
    }
    for _ in 0..warmup {
        let command_buffer = metal.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let encoder_guard = EncoderEndGuard::new(encoder);
        let mut owned = Vec::new();
        encode(encoder, &mut owned)?;
        encoder_guard.end();
        set_commit_label("linear_attn");
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
        set_commit_label("linear_attn");
        commit_and_wait(command_buffer)?;
    }
    Ok(started.elapsed().as_secs_f64() * 1000.0 / f64::from(iters))
}
