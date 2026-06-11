//! Orchestration Metal des pas linear-attention.

use super::*;

#[expect(
    clippy::too_many_arguments,
    reason = "wrappers d'encodage Metal: buffers, dimensions et offsets restent explicites"
)]
impl MetalExecutor {
    pub(crate) fn allocate_linear_attn_state_captures(
        &self,
        states: &[Option<&LinearAttentionMetalState>],
        rows: usize,
    ) -> Result<Vec<Option<Vec<LinearAttentionMetalState>>>> {
        if rows == 0 {
            return Err(InferError::Dimension(
                "captures linear-attn rows vide".to_string(),
            ));
        }
        states
            .iter()
            .map(|state| {
                let Some(state) = state else {
                    return Ok(None);
                };
                let captures = (0..rows)
                    .map(|_| {
                        Ok(LinearAttentionMetalState {
                            conv: self
                                .uncached_f32_buffer(state.conv_len, "linear_attn_capture_conv")?,
                            ssm: self
                                .uncached_f32_buffer(state.ssm_len, "linear_attn_capture_ssm")?,
                            conv_len: state.conv_len,
                            ssm_len: state.ssm_len,
                            conv_dim: state.conv_dim,
                            conv_kernel_dim: state.conv_kernel_dim,
                            num_value_heads: state.num_value_heads,
                            value_head_dim: state.value_head_dim,
                            key_head_dim: state.key_head_dim,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(Some(captures))
            })
            .collect()
    }

    fn encode_capture_linear_attn_state(
        &self,
        encoder: &ComputeCommandEncoderRef,
        state: &LinearAttentionMetalState,
        capture: &LinearAttentionMetalState,
    ) -> Result<()> {
        if state.conv_len != capture.conv_len || state.ssm_len != capture.ssm_len {
            return Err(InferError::Shape(format!(
                "capture linear-attn: capture conv={}/ssm={} ≠ état conv={}/ssm={}",
                capture.conv_len, capture.ssm_len, state.conv_len, state.ssm_len
            )));
        }
        self.encode_copy_with_offsets(encoder, &state.conv, 0, &capture.conv, 0, state.conv_len)?;
        self.encode_copy_with_offsets(encoder, &state.ssm, 0, &capture.ssm, 0, state.ssm_len)
    }

    /// Exécute le step cached d'une linear attention hybride sur GPU.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si les dimensions divergent ou si Metal échoue.
    pub(crate) fn linear_attention_cached_step(
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
        conv_state: &mut [f32],
        ssm_state: &mut [f32],
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
                "linear-attn Metal attend batch=1, reçu {batch}"
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
                "linear-attn Metal dims invalides: key_heads={}, value_heads={}, key_dim={}, value_dim={}, kernel={}",
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
        let state_len = checked_len(
            checked_len(
                spec.num_value_heads,
                spec.value_head_dim,
                "linear-attn state heads",
            )?,
            spec.key_head_dim,
            "linear-attn state",
        )?;
        if conv_state.len() != keep * conv_dim || ssm_state.len() != state_len {
            return Err(InferError::Dimension(format!(
                "linear-attn Metal state conv={}, ssm={}, attendu conv={}, ssm={state_len}",
                conv_state.len(),
                ssm_state.len(),
                keep * conv_dim
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
                    "linear_attn.conv1d.weight Metal attendu [{conv_dim},{},1] ou [{conv_dim},1,{}], reçu {shape:?}",
                    spec.conv_kernel_dim, spec.conv_kernel_dim
                )))
            }
        }
        let a_log = dense_vector(a_log, spec.num_value_heads, "linear_attn.A_log")?;
        let dt_bias = dense_vector(dt_bias, spec.num_value_heads, "linear_attn.dt_bias")?;
        let norm_weight =
            dense_vector(norm_weight, spec.value_head_dim, "linear_attn.norm.weight")?;
        let out_dim = linear_out_dim(out_proj.weight())?;

        let input_buffer = self.upload_f32_buffer(input.data(), "linear_attn_input")?;
        let qkv_buffer = self.private_f32_buffer(conv_dim, "linear_attn_qkv")?;
        let z_buffer = self.private_f32_buffer(value_dim, "linear_attn_z")?;
        let beta_input_buffer =
            self.private_f32_buffer(spec.num_value_heads, "linear_attn_beta_input")?;
        let gate_input_buffer =
            self.private_f32_buffer(spec.num_value_heads, "linear_attn_gate_input")?;
        let conv_weight_buffer =
            self.cached_buffer_from_f32(conv_weight.data(), "linear_attn_conv_weight")?;
        let a_log_buffer = self.cached_buffer_from_f32(a_log, "linear_attn_a_log")?;
        let dt_bias_buffer = self.cached_buffer_from_f32(dt_bias, "linear_attn_dt_bias")?;
        let norm_weight_buffer =
            self.cached_buffer_from_f32(norm_weight, "linear_attn_norm_weight")?;
        let conv_state_buffer = self.upload_f32_buffer(conv_state, "linear_attn_conv_state")?;
        let ssm_state_buffer = self.upload_f32_buffer(ssm_state, "linear_attn_ssm_state")?;
        let conv_out_buffer = self.private_f32_buffer(conv_dim, "linear_attn_conv_out")?;
        let q_norm_buffer = self.private_f32_buffer(key_dim, "linear_attn_q_norm")?;
        let k_norm_buffer = self.private_f32_buffer(key_dim, "linear_attn_k_norm")?;
        let beta_buffer = self.private_f32_buffer(spec.num_value_heads, "linear_attn_beta")?;
        let decay_buffer = self.private_f32_buffer(spec.num_value_heads, "linear_attn_decay")?;
        let y_buffer = self.private_f32_buffer(value_dim, "linear_attn_y")?;
        let gated_buffer = self.private_f32_buffer(value_dim, "linear_attn_gated")?;
        let output_buffer = self.new_f32_buffer(out_dim, "linear_attn_output")?;
        let mut owned_buffers = Vec::new();

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let encoder_guard = EncoderEndGuard::new(encoder);
        self.encode_matmul_weight(
            encoder,
            &mut owned_buffers,
            &input_buffer,
            1,
            in_dim,
            in_proj_qkv.weight(),
            &qkv_buffer,
        )?;
        self.encode_matmul_weight(
            encoder,
            &mut owned_buffers,
            &input_buffer,
            1,
            in_dim,
            in_proj_z.weight(),
            &z_buffer,
        )?;
        self.encode_matmul_weight(
            encoder,
            &mut owned_buffers,
            &input_buffer,
            1,
            in_dim,
            in_proj_b.weight(),
            &beta_input_buffer,
        )?;
        self.encode_matmul_weight(
            encoder,
            &mut owned_buffers,
            &input_buffer,
            1,
            in_dim,
            in_proj_a.weight(),
            &gate_input_buffer,
        )?;
        self.encode_linear_attn_conv(
            encoder,
            &qkv_buffer,
            &conv_weight_buffer,
            &conv_state_buffer,
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
            &ssm_state_buffer,
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
            &mut owned_buffers,
            &gated_buffer,
            1,
            value_dim,
            out_proj.weight(),
            &output_buffer,
        )?;
        encoder_guard.end();
        commit_and_wait(command_buffer)?;

        let output = read_f32_buffer(&output_buffer, out_dim)?;
        let next_conv = read_f32_buffer(&conv_state_buffer, conv_state.len())?;
        let next_ssm = read_f32_buffer(&ssm_state_buffer, ssm_state.len())?;
        conv_state.copy_from_slice(&next_conv);
        ssm_state.copy_from_slice(&next_ssm);
        Tensor::from_vec(vec![1, out_dim], output)
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
        let conv_out_buffer = self.private_f32_buffer(conv_dim, "linear_attn_batch_conv_out")?;
        let q_norm_buffer = self.private_f32_buffer(key_dim, "linear_attn_batch_q_norm")?;
        let k_norm_buffer = self.private_f32_buffer(key_dim, "linear_attn_batch_k_norm")?;
        let beta_buffer =
            self.private_f32_buffer(spec.num_value_heads, "linear_attn_batch_beta")?;
        let decay_buffer =
            self.private_f32_buffer(spec.num_value_heads, "linear_attn_batch_decay")?;
        let y_buffer = self.private_f32_buffer(value_dim, "linear_attn_batch_y")?;
        let gated_row_buffer = self.private_f32_buffer(value_dim, "linear_attn_batch_gated_row")?;
        let gated_batch_buffer = self.private_f32_buffer(
            checked_len(batch, value_dim, "linear-attn batch gated")?,
            "linear_attn_batch_gated",
        )?;
        let output_buffer = self.new_f32_buffer(
            checked_len(batch, out_dim, "linear-attn batch output")?,
            "linear_attn_batch_output",
        )?;
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
            let gated_offset = byte_offset_f32(pos * value_dim, "linear-attn gated batch offset")?;
            self.encode_copy_with_offsets(
                encoder,
                &gated_row_buffer,
                0,
                &gated_batch_buffer,
                gated_offset,
                value_dim,
            )?;
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
        commit_and_wait(command_buffer)?;

        let output = read_f32_buffer(&output_buffer, batch * out_dim)?;
        Tensor::from_vec(vec![batch, out_dim], output)
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
        self.encode_linear_attn_gated_delta(
            encoder,
            &conv_out_buffer,
            &q_norm_buffer,
            &k_norm_buffer,
            &beta_buffer,
            &decay_buffer,
            &state.ssm,
            &y_buffer,
            spec,
        )?;
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
            if captures.len() < rows {
                return Err(InferError::Dimension(format!(
                    "captures linear-attn rows={} < rows={rows}",
                    captures.len()
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
                &y_buffer,
                spec,
            )?;
            if let Some(captures) = captures {
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
        let valid = state.as_ref().is_some_and(|state| {
            state.conv_len == conv_len
                && state.ssm_len == ssm_len
                && state.conv_dim == conv_dim
                && state.conv_kernel_dim == spec.conv_kernel_dim
                && state.num_value_heads == spec.num_value_heads
                && state.value_head_dim == spec.value_head_dim
                && state.key_head_dim == spec.key_head_dim
        });
        if valid {
            return Ok(());
        }
        *state = Some(LinearAttentionMetalState {
            conv: self.buffer_from_f32(conv_seed, "linear_attn_resident_conv")?,
            ssm: self.buffer_from_f32(ssm_seed, "linear_attn_resident_ssm")?,
            conv_len,
            ssm_len,
            conv_dim,
            conv_kernel_dim: spec.conv_kernel_dim,
            num_value_heads: spec.num_value_heads,
            value_head_dim: spec.value_head_dim,
            key_head_dim: spec.key_head_dim,
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
        let ssm = self.uncached_f32_buffer(state.ssm_len, "linear_attn_snapshot_ssm")?;
        let command_buffer = self.queue.new_command_buffer();
        let blit = command_buffer.new_blit_command_encoder();
        blit.copy_from_buffer(&state.conv, 0, &conv, 0, byte_len::<f32>(state.conv_len)?);
        blit.copy_from_buffer(&state.ssm, 0, &ssm, 0, byte_len::<f32>(state.ssm_len)?);
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
            let ssm = self.uncached_f32_buffer(state.ssm_len, "linear_attn_snapshot_ssm")?;
            blit.copy_from_buffer(&state.conv, 0, &conv, 0, byte_len::<f32>(state.conv_len)?);
            blit.copy_from_buffer(&state.ssm, 0, &ssm, 0, byte_len::<f32>(state.ssm_len)?);
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
            }));
        }
        blit.end_encoding();
        if copied {
            commit_and_wait(command_buffer)?;
        }
        Ok(snapshots)
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
        if state.conv_len != snapshot.conv_len || state.ssm_len != snapshot.ssm_len {
            return Err(InferError::Shape(format!(
                "restore linear-attn: snapshot conv={}/ssm={} ≠ état conv={}/ssm={}",
                snapshot.conv_len, snapshot.ssm_len, state.conv_len, state.ssm_len
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
        blit.copy_from_buffer(
            &snapshot.ssm,
            0,
            &state.ssm,
            0,
            byte_len::<f32>(state.ssm_len)?,
        );
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
            if state.conv_len != snapshot.conv_len || state.ssm_len != snapshot.ssm_len {
                return Err(InferError::Shape(format!(
                    "restore linear-attn batch: snapshot conv={}/ssm={} ≠ état conv={}/ssm={}",
                    snapshot.conv_len, snapshot.ssm_len, state.conv_len, state.ssm_len
                )));
            }
            blit.copy_from_buffer(
                &snapshot.conv,
                0,
                &state.conv,
                0,
                byte_len::<f32>(state.conv_len)?,
            );
            blit.copy_from_buffer(
                &snapshot.ssm,
                0,
                &state.ssm,
                0,
                byte_len::<f32>(state.ssm_len)?,
            );
        }
        blit.end_encoding();
        commit_and_wait(command_buffer)
    }
}
