//! Projection lm_head et argmax Metal.

use super::*;

const LM_HEAD_TG_WIDTH: u64 = 256;

#[expect(
    clippy::too_many_arguments,
    reason = "wrappers d'encodage Metal: buffers, dimensions et offsets restent explicites"
)]
impl MetalExecutor {
    /// Échantillonne une projection linéaire biasless sans lire les logits.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si l'entrée, le top-k ou la projection sont invalides.
    pub(crate) fn sample_linear_biasless_topk_topp(
        &self,
        input: &Tensor,
        linear: &Linear,
        temperature: f32,
        top_p: f32,
        top_k: usize,
        rng_state: u64,
    ) -> Result<usize> {
        ensure_biasless(linear, "sample")?;
        let (batch, in_dim) = input.as_matrix()?;
        if batch != 1 {
            return Err(InferError::Dimension(format!(
                "sample Metal attend batch=1, reçu {batch}"
            )));
        }
        if top_k == 0 || top_k > MAX_SAMPLER_TOP_K {
            return Err(InferError::Dimension(format!(
                "sample Metal top_k={top_k} invalide (max={MAX_SAMPLER_TOP_K})"
            )));
        }
        let out_dim = linear_out_dim(linear.weight())?;
        let input_buffer = self.upload_f32_buffer(input.data(), "sample_input")?;
        let logits_buffer = self.private_f32_buffer(out_dim, "sample_logits")?;
        let index_buffer = self.new_u32_buffer(1, "sample_index")?;
        let mut owned_buffers = Vec::new();

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let encoder_guard = EncoderEndGuard::new(encoder);
        let projected_dim = self.encode_matmul_weight(
            encoder,
            &mut owned_buffers,
            &input_buffer,
            batch,
            in_dim,
            linear.weight(),
            &logits_buffer,
        )?;
        if projected_dim != out_dim {
            return Err(InferError::Dimension(format!(
                "sample Metal projection sort {projected_dim}, attendu {out_dim}"
            )));
        }
        self.encode_sample_topk_topp(
            encoder,
            &logits_buffer,
            &index_buffer,
            out_dim,
            top_k,
            temperature,
            top_p,
            rng_state,
        )?;
        encoder_guard.end();
        commit_and_wait(command_buffer)?;

        let index = read_u32_buffer(&index_buffer, 1)?
            .into_iter()
            .next()
            .ok_or_else(|| InferError::Metal("sample Metal sans index".to_string()))?;
        usize::try_from(index)
            .map_err(|_| InferError::Metal(format!("sample Metal index trop grand: {index}")))
    }

    /// Échantillonne une projection biasless par Gumbel-max exact.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si l'entrée ou la projection sont invalides.
    pub(crate) fn sample_linear_biasless_gumbel(
        &self,
        input: &Tensor,
        linear: &Linear,
        temperature: f32,
        rng_state: u64,
    ) -> Result<usize> {
        ensure_biasless(linear, "sample_gumbel")?;
        let (batch, in_dim) = input.as_matrix()?;
        if batch != 1 {
            return Err(InferError::Dimension(format!(
                "sample Gumbel Metal attend batch=1, reçu {batch}"
            )));
        }
        let out_dim = linear_out_dim(linear.weight())?;
        let input_buffer = self.upload_f32_buffer(input.data(), "sample_gumbel_input")?;
        let logits_buffer = self.private_f32_buffer(out_dim, "sample_gumbel_logits")?;
        let index_buffer = self.new_u32_buffer(1, "sample_gumbel_index")?;
        let mut owned_buffers = Vec::new();

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let encoder_guard = EncoderEndGuard::new(encoder);
        let projected_dim = self.encode_matmul_weight(
            encoder,
            &mut owned_buffers,
            &input_buffer,
            batch,
            in_dim,
            linear.weight(),
            &logits_buffer,
        )?;
        if projected_dim != out_dim {
            return Err(InferError::Dimension(format!(
                "sample Gumbel projection sort {projected_dim}, attendu {out_dim}"
            )));
        }
        self.encode_sample_gumbel(
            encoder,
            &logits_buffer,
            &index_buffer,
            out_dim,
            temperature,
            rng_state,
        )?;
        encoder_guard.end();
        commit_and_wait(command_buffer)?;

        let index = read_u32_buffer(&index_buffer, 1)?
            .into_iter()
            .next()
            .ok_or_else(|| InferError::Metal("sample Gumbel Metal sans index".to_string()))?;
        usize::try_from(index).map_err(|_| {
            InferError::Metal(format!("sample Gumbel Metal index trop grand: {index}"))
        })
    }

    /// Calcule l'argmax d'une projection linéaire biasless sans lire le vocab complet.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si l'entrée n'est pas un batch unitaire ou si la
    /// projection n'est pas compatible avec le kernel Metal.
    pub(crate) fn argmax_linear_biasless(&self, input: &Tensor, linear: &Linear) -> Result<usize> {
        ensure_biasless(linear, "argmax")?;
        let (batch, in_dim) = input.as_matrix()?;
        if batch != 1 {
            return Err(InferError::Dimension(format!(
                "argmax Metal attend batch=1, reçu {batch}"
            )));
        }
        let input_buffer = self.upload_f32_buffer(input.data(), "argmax_input")?;
        let index_buffer = self.new_u32_buffer(1, "argmax_index")?;
        let mut owned_buffers = Vec::new();

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let encoder_guard = EncoderEndGuard::new(encoder);
        self.encode_lm_head_argmax(
            encoder,
            &mut owned_buffers,
            &input_buffer,
            linear,
            &index_buffer,
            in_dim,
        )?;
        encoder_guard.end();
        commit_and_wait(command_buffer)?;

        let index = read_u32_buffer(&index_buffer, 1)?
            .into_iter()
            .next()
            .ok_or_else(|| InferError::Metal("argmax Metal sans index".to_string()))?;
        usize::try_from(index)
            .map_err(|_| InferError::Metal(format!("argmax Metal index trop grand: {index}")))
    }

    /// Argmax greedy du talker TTS (cb0) sur GPU : matmul `codec_head·input` puis
    /// `talker_greedy_argmax` (quantification `floor(x*4)/4` + suppression de la
    /// plage `[suppress_start, vocab)` sauf `eos`, tie-break index le plus bas).
    /// Renvoie l'id du token (1 `u32` relu), tuant le readback full-vocab + l'argmax
    /// CPU `greedy_talker_token`. Byte-identique au CPU (mêmes logits GPU, même
    /// quantification exacte). `linear` doit être biasless.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si `batch != 1`, si la projection déborde ou si l'encodage
    /// Metal échoue.
    pub(crate) fn talker_greedy_token_biasless(
        &self,
        input: &Tensor,
        linear: &Linear,
        suppress_start: usize,
        eos: usize,
    ) -> Result<usize> {
        ensure_biasless(linear, "talker_greedy")?;
        let (batch, in_dim) = input.as_matrix()?;
        if batch != 1 {
            return Err(InferError::Dimension(format!(
                "talker greedy attend batch=1, reçu {batch}"
            )));
        }
        let out_dim = linear_out_dim(linear.weight())?;
        let input_buffer = self.upload_f32_buffer(input.data(), "talker_argmax_input")?;
        let logits_buffer = self.private_f32_buffer(out_dim, "talker_argmax_logits")?;
        let index_buffer = self.new_u32_buffer(1, "talker_argmax_index")?;
        let mut owned_buffers = Vec::new();

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let encoder_guard = EncoderEndGuard::new(encoder);
        let projected = self.encode_matmul_weight(
            encoder,
            &mut owned_buffers,
            &input_buffer,
            batch,
            in_dim,
            linear.weight(),
            &logits_buffer,
        )?;
        if projected != out_dim {
            return Err(InferError::Dimension(format!(
                "talker greedy projection sort {projected}, attendu {out_dim}"
            )));
        }
        self.encode_talker_greedy_argmax(
            encoder,
            &logits_buffer,
            out_dim,
            suppress_start,
            eos,
            &index_buffer,
        )?;
        encoder_guard.end();
        commit_and_wait(command_buffer)?;

        let index = read_u32_buffer(&index_buffer, 1)?
            .into_iter()
            .next()
            .ok_or_else(|| InferError::Metal("talker greedy sans index".to_string()))?;
        usize::try_from(index)
            .map_err(|_| InferError::Metal(format!("talker greedy index trop grand: {index}")))
    }

    /// Encode l'argmax greedy talker (quantifié + suppression de plage) sur un
    /// buffer de logits déjà calculé, écrivant 1 `u32` dans `out_index`. Un seul
    /// threadgroup de 256, grid-stride (vocab talker = 3072).
    pub(crate) fn encode_talker_greedy_argmax(
        &self,
        encoder: &ComputeCommandEncoderRef,
        logits_buffer: &BufferRef,
        count: usize,
        suppress_start: usize,
        eos: usize,
        out_index: &BufferRef,
    ) -> Result<()> {
        let count = checked_u32(count, "talker argmax count")?;
        let suppress_start = checked_u32(suppress_start, "talker argmax suppress_start")?;
        let eos = checked_u32(eos, "talker argmax eos")?;
        encoder.set_compute_pipeline_state(&self.talker_greedy_argmax_f32);
        encoder.set_buffer(0, Some(logits_buffer), 0);
        set_u32_bytes(encoder, 1, &[count], "talker_argmax_count")?;
        set_u32_bytes(encoder, 2, &[suppress_start], "talker_argmax_suppress")?;
        set_u32_bytes(encoder, 3, &[eos], "talker_argmax_eos")?;
        encoder.set_buffer(4, Some(out_index), 0);
        profile_dispatch();
        encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(LM_HEAD_TG_WIDTH, 1, 1));
        post_dispatch_barrier(encoder);
        Ok(())
    }

    /// Encode `lm_head·input` puis l'argmax GPU dans l'encoder PARTAGÉ, écrivant
    /// l'id du token (1 `u32`) dans `index_buffer`. **Aucun readback des logits**
    /// (131K) : seul le `u32` final est lu par l'appelant. Cœur extrait d'
    /// [`Self::argmax_linear_biasless`] (désormais wrapper), réutilisé par le decode
    /// résident 1c pour clore le command buffer unique. `lm_head` doit être biasless
    /// (garanti en amont par `supports_resident_full_decode`).
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si une dimension déborde ou si l'encodage échoue.
    pub(crate) fn encode_lm_head_argmax(
        &self,
        encoder: &ComputeCommandEncoderRef,
        owned_buffers: &mut Vec<metal::Buffer>,
        input_buffer: &BufferRef,
        lm_head: &Linear,
        index_buffer: &BufferRef,
        in_dim: usize,
    ) -> Result<()> {
        let out_dim = linear_out_dim(lm_head.weight())?;
        let batch = 1;
        if let LinearWeight::AffineQuantized(weight) = lm_head.weight() {
            if can_use_fast_affine_argmax_qmv(batch, in_dim, weight) {
                let partial_count = out_dim.div_ceil(8);
                let partial_values =
                    self.private_f32_buffer(partial_count, "argmax_partial_values")?;
                let partial_indices =
                    self.private_u32_buffer(partial_count, "argmax_partial_indices")?;
                self.encode_affine_argmax_qmv_fast(
                    encoder,
                    input_buffer,
                    weight,
                    &partial_values,
                    &partial_indices,
                    batch,
                    in_dim,
                    out_dim,
                )?;
                self.encode_argmax_finalize(
                    encoder,
                    &partial_values,
                    &partial_indices,
                    index_buffer,
                    partial_count,
                )?;
                return Ok(());
            }
        }
        self.encode_argmax_projected(
            encoder,
            owned_buffers,
            input_buffer,
            lm_head.weight(),
            index_buffer,
            batch,
            in_dim,
            out_dim,
        )
    }

    pub(crate) fn encode_lm_head_argmax_buffers(
        &self,
        encoder: &ComputeCommandEncoderRef,
        owned_buffers: &mut Vec<metal::Buffer>,
        input_buffer: &BufferRef,
        lm_head: &MetalLinearWeightBuffers,
        index_buffer: &BufferRef,
        in_dim: usize,
    ) -> Result<()> {
        let out_dim = self.linear_weight_out_dim(lm_head);
        let batch = 1;
        if let MetalLinearWeightBuffers::Dense {
            rhs_bf16: Some(rhs_bf16),
            in_dim: rhs_in_dim,
            ..
        } = lm_head
        {
            if super::whisper_decode_bf16_qmv_enabled() && *rhs_in_dim == in_dim {
                let logits_buffer = self.private_f32_buffer(out_dim, "argmax_logits")?;
                let partial_count = out_dim.div_ceil(256);
                let partial_values =
                    self.private_f32_buffer(partial_count, "argmax_partial_values")?;
                let partial_indices =
                    self.private_u32_buffer(partial_count, "argmax_partial_indices")?;
                self.encode_dense_qmv_rhs_bf16(
                    encoder,
                    input_buffer,
                    rhs_bf16,
                    &logits_buffer,
                    batch,
                    out_dim,
                    in_dim,
                )?;
                self.encode_argmax(
                    encoder,
                    owned_buffers,
                    &logits_buffer,
                    &partial_values,
                    &partial_indices,
                    index_buffer,
                    out_dim,
                )?;
                return Ok(());
            }
        }
        if let MetalLinearWeightBuffers::AffineQuantized {
            group_size, bits, ..
        } = lm_head
        {
            if fast_argmax_qmv_enabled()
                && *bits == FAST_QMV_BITS
                && *group_size == FAST_QMV_GROUP_SIZE
                && in_dim % 512 == 0
                && self.linear_weight_in_dim(lm_head) == in_dim
            {
                self.encode_lm_head_argmax_fast_buffers_with_offsets(
                    encoder,
                    owned_buffers,
                    input_buffer,
                    lm_head,
                    index_buffer,
                    0,
                    0,
                    in_dim,
                    out_dim,
                )?;
                return Ok(());
            }
        }
        let logits_buffer = self.private_f32_buffer(out_dim, "argmax_logits")?;
        let partial_count = out_dim.div_ceil(256);
        let partial_values = self.private_f32_buffer(partial_count, "argmax_partial_values")?;
        let partial_indices = self.private_u32_buffer(partial_count, "argmax_partial_indices")?;
        let projected_dim = self.encode_matmul_weight_buffers(
            encoder,
            input_buffer,
            batch,
            in_dim,
            lm_head,
            &logits_buffer,
            true,
        )?;
        if projected_dim != out_dim {
            return Err(InferError::Dimension(format!(
                "argmax Metal projection sort {projected_dim}, attendu {out_dim}"
            )));
        }
        self.encode_argmax(
            encoder,
            owned_buffers,
            &logits_buffer,
            &partial_values,
            &partial_indices,
            index_buffer,
            out_dim,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "sampler lm_head: buffers, dimensions et paramètres sampling explicites"
    )]
    pub(crate) fn encode_lm_head_sample_topk_topp_buffers(
        &self,
        encoder: &ComputeCommandEncoderRef,
        _owned_buffers: &mut Vec<metal::Buffer>,
        input_buffer: &BufferRef,
        lm_head: &MetalLinearWeightBuffers,
        index_buffer: &BufferRef,
        in_dim: usize,
        temperature: f32,
        top_p: f32,
        top_k: usize,
        rng_state: u64,
    ) -> Result<()> {
        let out_dim = self.linear_weight_out_dim(lm_head);
        let logits_buffer = self.private_f32_buffer(out_dim, "sample_logits")?;
        let projected_dim = self.encode_matmul_weight_buffers(
            encoder,
            input_buffer,
            1,
            in_dim,
            lm_head,
            &logits_buffer,
            false,
        )?;
        if projected_dim != out_dim {
            return Err(InferError::Dimension(format!(
                "sample Metal projection sort {projected_dim}, attendu {out_dim}"
            )));
        }
        self.encode_sample_topk_topp(
            encoder,
            &logits_buffer,
            index_buffer,
            out_dim,
            top_k,
            temperature,
            top_p,
            rng_state,
        )
    }

    /// Encode le `lm_head` puis un sampling Gumbel-max (`top_k=0`, `top_p=1`).
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si la projection sort une dimension inattendue.
    pub(crate) fn encode_lm_head_sample_gumbel_buffers(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input_buffer: &BufferRef,
        lm_head: &MetalLinearWeightBuffers,
        index_buffer: &BufferRef,
        in_dim: usize,
        temperature: f32,
        rng_state: u64,
    ) -> Result<()> {
        let out_dim = self.linear_weight_out_dim(lm_head);
        let logits_buffer = self.private_f32_buffer(out_dim, "sample_gumbel_logits")?;
        let projected_dim = self.encode_matmul_weight_buffers(
            encoder,
            input_buffer,
            1,
            in_dim,
            lm_head,
            &logits_buffer,
            false,
        )?;
        if projected_dim != out_dim {
            return Err(InferError::Dimension(format!(
                "sample Gumbel projection sort {projected_dim}, attendu {out_dim}"
            )));
        }
        self.encode_sample_gumbel(
            encoder,
            &logits_buffer,
            index_buffer,
            out_dim,
            temperature,
            rng_state,
        )
    }

    /// Encode le `lm_head` dans un buffer partagé pour sampling CPU exact.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si la projection sort une dimension inattendue.
    pub(crate) fn encode_lm_head_logits_readback_buffers(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input_buffer: &BufferRef,
        lm_head: &MetalLinearWeightBuffers,
        in_dim: usize,
    ) -> Result<(metal::Buffer, usize)> {
        let out_dim = self.linear_weight_out_dim(lm_head);
        let logits_buffer = self.uncached_f32_buffer(out_dim, "sample_logits_readback")?;
        let projected_dim = self.encode_matmul_weight_buffers(
            encoder,
            input_buffer,
            1,
            in_dim,
            lm_head,
            &logits_buffer,
            false,
        )?;
        if projected_dim != out_dim {
            return Err(InferError::Dimension(format!(
                "sample readback projection sort {projected_dim}, attendu {out_dim}"
            )));
        }
        Ok((logits_buffer, out_dim))
    }

    pub(crate) fn encode_lm_head_argmax_buffers_with_index_offset(
        &self,
        encoder: &ComputeCommandEncoderRef,
        owned_buffers: &mut Vec<metal::Buffer>,
        input_buffer: &BufferRef,
        lm_head: &MetalLinearWeightBuffers,
        index_buffer: &BufferRef,
        input_offset: u64,
        index_offset: u64,
        in_dim: usize,
    ) -> Result<()> {
        let out_dim = self.linear_weight_out_dim(lm_head);
        if let MetalLinearWeightBuffers::AffineQuantized {
            group_size, bits, ..
        } = lm_head
        {
            if fast_argmax_qmv_enabled()
                && *bits == FAST_QMV_BITS
                && *group_size == FAST_QMV_GROUP_SIZE
                && in_dim % 512 == 0
                && self.linear_weight_in_dim(lm_head) == in_dim
            {
                self.encode_lm_head_argmax_fast_buffers_with_offsets(
                    encoder,
                    owned_buffers,
                    input_buffer,
                    lm_head,
                    index_buffer,
                    input_offset,
                    index_offset,
                    in_dim,
                    out_dim,
                )?;
                return Ok(());
            }
        }

        let logits_buffer = self.private_f32_buffer(out_dim, "argmax_logits")?;
        let partial_count = out_dim.div_ceil(256);
        let partial_values = self.private_f32_buffer(partial_count, "argmax_partial_values")?;
        let partial_indices = self.private_u32_buffer(partial_count, "argmax_partial_indices")?;
        let row_input;
        let matmul_input = if input_offset == 0 {
            input_buffer
        } else {
            row_input = self.private_f32_buffer(in_dim, "argmax_row_input")?;
            self.encode_copy_with_offsets(
                encoder,
                input_buffer,
                input_offset,
                &row_input,
                0,
                in_dim,
            )?;
            &row_input
        };
        let projected_dim = self.encode_matmul_weight_buffers(
            encoder,
            matmul_input,
            1,
            in_dim,
            lm_head,
            &logits_buffer,
            false,
        )?;
        if projected_dim != out_dim {
            return Err(InferError::Dimension(format!(
                "argmax Metal projection sort {projected_dim}, attendu {out_dim}"
            )));
        }
        self.encode_argmax_blocks(
            encoder,
            &logits_buffer,
            &partial_values,
            &partial_indices,
            out_dim,
        )?;
        self.encode_argmax_finalize_with_offset(
            encoder,
            &partial_values,
            &partial_indices,
            index_buffer,
            index_offset,
            partial_count,
        )
    }

    /// Encode `lm_head` + argmax pour deux lignes contiguës déjà résidentes.
    ///
    /// La projection passe par le matmul batché (`rows=2`), qui route vers qmm2
    /// pour les poids quantifiés éligibles. Les deux argmax restent séparés afin
    /// de conserver le tie-break existant.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si les dimensions divergent ou si Metal échoue.
    pub(crate) fn encode_lm_head_argmax_two_rows_buffers(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input_buffer: &BufferRef,
        lm_head: &MetalLinearWeightBuffers,
        index_buffer: &BufferRef,
        input_offset: u64,
        index_offset: u64,
        in_dim: usize,
    ) -> Result<()> {
        self.encode_lm_head_argmax_rows_buffers(
            encoder,
            input_buffer,
            lm_head,
            index_buffer,
            input_offset,
            index_offset,
            2,
            in_dim,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "lm_head rows: buffers + offsets + dimensions explicites"
    )]
    pub(crate) fn encode_lm_head_argmax_rows_buffers(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input_buffer: &BufferRef,
        lm_head: &MetalLinearWeightBuffers,
        index_buffer: &BufferRef,
        input_offset: u64,
        index_offset: u64,
        rows: usize,
        in_dim: usize,
    ) -> Result<()> {
        if rows == 0 {
            return Err(InferError::Dimension(
                "argmax rows exige au moins une ligne".to_string(),
            ));
        }
        let out_dim = self.linear_weight_out_dim(lm_head);
        let logits_len = rows
            .checked_mul(out_dim)
            .ok_or_else(|| InferError::Dimension("argmax rows logits déborde".to_string()))?;
        let logits_buffer = self.private_f32_buffer(logits_len, "argmax_rows_logits")?;
        let input_rows;
        let matmul_input = if input_offset == 0 {
            input_buffer
        } else {
            let input_len = rows
                .checked_mul(in_dim)
                .ok_or_else(|| InferError::Dimension("argmax rows input déborde".to_string()))?;
            input_rows = self.private_f32_buffer(input_len, "argmax_rows_input")?;
            self.encode_copy_with_offsets(
                encoder,
                input_buffer,
                input_offset,
                &input_rows,
                0,
                input_len,
            )?;
            &input_rows
        };
        let projected_dim = self.encode_matmul_weight_buffers(
            encoder,
            matmul_input,
            rows,
            in_dim,
            lm_head,
            &logits_buffer,
            true,
        )?;
        if projected_dim != out_dim {
            return Err(InferError::Dimension(format!(
                "argmax rows projection sort {projected_dim}, attendu {out_dim}"
            )));
        }
        let partial_count = out_dim.div_ceil(256);
        for row in 0..rows {
            let partial_values = self.private_f32_buffer(partial_count, "argmax_partial_values")?;
            let partial_indices =
                self.private_u32_buffer(partial_count, "argmax_partial_indices")?;
            let logits_offset = byte_offset_f32(
                row.checked_mul(out_dim).ok_or_else(|| {
                    InferError::Dimension("argmax rows logits offset déborde".to_string())
                })?,
                "argmax rows logits offset",
            )?;
            let row_index_offset = row
                .checked_mul(std::mem::size_of::<u32>())
                .and_then(|value| u64::try_from(value).ok())
                .ok_or_else(|| InferError::Metal("argmax rows index offset déborde".to_string()))?;
            let out_offset = index_offset
                .checked_add(row_index_offset)
                .ok_or_else(|| InferError::Metal("argmax rows index offset déborde".to_string()))?;
            self.encode_argmax_blocks_with_offset(
                encoder,
                &logits_buffer,
                logits_offset,
                &partial_values,
                &partial_indices,
                out_dim,
            )?;
            self.encode_argmax_finalize_with_offset(
                encoder,
                &partial_values,
                &partial_indices,
                index_buffer,
                out_offset,
                partial_count,
            )?;
        }
        Ok(())
    }

    pub(crate) fn argmax_linear_biasless_rows_buffers(
        &self,
        input: &Tensor,
        lm_head: &MetalLinearWeightBuffers,
    ) -> Result<Option<Vec<usize>>> {
        let (batch, in_dim) = input.as_matrix()?;
        if batch == 0 {
            return Ok(Some(Vec::new()));
        }
        let input_buffer = self.upload_f32_buffer(input.data(), "argmax_rows_input")?;
        let index_buffer = self.new_u32_buffer(batch, "argmax_rows_index")?;
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let encoder_guard = EncoderEndGuard::new(encoder);
        let mut owned_buffers = Vec::new();
        for row in 0..batch {
            let input_offset = row
                .checked_mul(in_dim)
                .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
                .and_then(|value| u64::try_from(value).ok())
                .ok_or_else(|| InferError::Metal("argmax rows input offset déborde".to_string()))?;
            let index_offset = row
                .checked_mul(std::mem::size_of::<u32>())
                .and_then(|value| u64::try_from(value).ok())
                .ok_or_else(|| InferError::Metal("argmax rows index offset déborde".to_string()))?;
            self.encode_lm_head_argmax_buffers_with_index_offset(
                encoder,
                &mut owned_buffers,
                &input_buffer,
                lm_head,
                &index_buffer,
                input_offset,
                index_offset,
                in_dim,
            )?;
        }
        encoder_guard.end();
        commit_and_wait(command_buffer)?;

        let raw = read_u32_buffer(&index_buffer, batch)?;
        raw.into_iter()
            .map(|index| {
                usize::try_from(index).map_err(|_| {
                    InferError::Metal(format!("argmax rows index trop grand: {index}"))
                })
            })
            .collect::<Result<Vec<_>>>()
            .map(Some)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "argmax fast avec offsets explicites pour batcher le verify MTP"
    )]
    fn encode_lm_head_argmax_fast_buffers_with_offsets(
        &self,
        encoder: &ComputeCommandEncoderRef,
        _owned_buffers: &mut Vec<metal::Buffer>,
        input_buffer: &BufferRef,
        lm_head: &MetalLinearWeightBuffers,
        index_buffer: &BufferRef,
        input_offset: u64,
        index_offset: u64,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<()> {
        let MetalLinearWeightBuffers::AffineQuantized {
            packed,
            scales,
            biases,
            packed_cols,
            group_size,
            bits,
            groups,
            ..
        } = lm_head
        else {
            return Err(InferError::Metal(
                "argmax fast offset requiert des poids affine quantifiés".to_string(),
            ));
        };
        if !fast_argmax_qmv_enabled()
            || *bits != FAST_QMV_BITS
            || *group_size != FAST_QMV_GROUP_SIZE
            || in_dim % 512 != 0
            || self.linear_weight_in_dim(lm_head) != in_dim
        {
            return Err(InferError::Metal(
                "argmax fast offset indisponible".to_string(),
            ));
        }
        let partial_count = out_dim.div_ceil(8);
        let partial_values = self.private_f32_buffer(partial_count, "argmax_partial_values")?;
        let partial_indices = self.private_u32_buffer(partial_count, "argmax_partial_indices")?;
        let dims = [
            checked_u32(out_dim, "argmax fast out_dim")?,
            checked_u32(in_dim, "argmax fast in_dim")?,
            checked_u32(*packed_cols, "argmax fast packed_cols")?,
            checked_u32(*groups, "argmax fast groups")?,
        ];
        encoder.set_compute_pipeline_state(&self.affine_argmax_qmv_fast_u4_gs64_f32);
        encoder.set_buffer(0, Some(input_buffer), input_offset);
        encoder.set_buffer(1, Some(packed), 0);
        encoder.set_buffer(2, Some(scales), 0);
        encoder.set_buffer(3, Some(biases), 0);
        encoder.set_buffer(4, Some(&partial_values), 0);
        encoder.set_buffer(5, Some(&partial_indices), 0);
        set_u32_bytes(encoder, 6, &dims, "argmax_fast_dims")?;
        profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(
                checked_nsuint(1, "argmax fast batch")?,
                checked_nsuint(out_dim.div_ceil(8), "argmax fast out groups")?,
                1,
            ),
            MTLSize::new(64, 1, 1),
        );
        post_dispatch_barrier(encoder);
        self.encode_argmax_finalize_with_offset(
            encoder,
            &partial_values,
            &partial_indices,
            index_buffer,
            index_offset,
            partial_count,
        )
    }

    pub(super) fn encode_argmax_projected(
        &self,
        encoder: &ComputeCommandEncoderRef,
        owned_buffers: &mut Vec<metal::Buffer>,
        input_buffer: &BufferRef,
        weight: &LinearWeight,
        output_index: &BufferRef,
        batch: usize,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<()> {
        let logits_buffer = self.private_f32_buffer(out_dim, "argmax_logits")?;
        let partial_count = out_dim.div_ceil(256);
        let partial_values = self.private_f32_buffer(partial_count, "argmax_partial_values")?;
        let partial_indices = self.private_u32_buffer(partial_count, "argmax_partial_indices")?;
        let projected_dim = self.encode_matmul_weight(
            encoder,
            owned_buffers,
            input_buffer,
            batch,
            in_dim,
            weight,
            &logits_buffer,
        )?;
        if projected_dim != out_dim {
            return Err(InferError::Dimension(format!(
                "argmax Metal projection sort {projected_dim}, attendu {out_dim}"
            )));
        }
        self.encode_argmax(
            encoder,
            owned_buffers,
            &logits_buffer,
            &partial_values,
            &partial_indices,
            output_index,
            out_dim,
        )
    }

    pub(super) fn encode_affine_argmax_qmv_fast(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input_buffer: &BufferRef,
        weight: &AffineQuantizedTensor,
        partial_values: &BufferRef,
        partial_indices: &BufferRef,
        batch: usize,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<()> {
        let [packed_rows, packed_cols] = weight.packed_shape() else {
            return Err(InferError::Dimension(format!(
                "packed_shape argmax attendu rang 2, reçu {:?}",
                weight.packed_shape()
            )));
        };
        if *packed_rows != out_dim {
            return Err(InferError::Dimension(format!(
                "argmax packed_rows={packed_rows}, out_dim={out_dim}"
            )));
        }
        let groups = in_dim
            .checked_div(weight.group_size())
            .ok_or_else(|| InferError::Metal("group_size argmax nul".to_string()))?;
        let packed_buffer = self.cached_buffer_from_u32(weight.packed_data(), "argmax_packed")?;
        let scales_buffer =
            self.cached_buffer_from_f32_as_bf16(weight.scales().data(), "argmax_scales")?;
        let biases_buffer =
            self.cached_buffer_from_f32_as_bf16(weight.biases().data(), "argmax_biases")?;
        let dims = [
            checked_u32(out_dim, "argmax fast out_dim")?,
            checked_u32(in_dim, "argmax fast in_dim")?,
            checked_u32(*packed_cols, "argmax fast packed_cols")?,
            checked_u32(groups, "argmax fast groups")?,
        ];
        encoder.set_compute_pipeline_state(&self.affine_argmax_qmv_fast_u4_gs64_f32);
        encoder.set_buffer(0, Some(input_buffer), 0);
        encoder.set_buffer(1, Some(&packed_buffer), 0);
        encoder.set_buffer(2, Some(&scales_buffer), 0);
        encoder.set_buffer(3, Some(&biases_buffer), 0);
        encoder.set_buffer(4, Some(partial_values), 0);
        encoder.set_buffer(5, Some(partial_indices), 0);
        set_u32_bytes(encoder, 6, &dims, "argmax_fast_dims")?;
        profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(
                checked_nsuint(batch, "argmax fast batch")?,
                checked_nsuint(out_dim.div_ceil(8), "argmax fast out groups")?,
                1,
            ),
            MTLSize::new(64, 1, 1),
        );
        post_dispatch_barrier(encoder);
        Ok(())
    }

    pub(super) fn encode_add_rms_norm(
        &self,
        encoder: &ComputeCommandEncoderRef,
        _owned_buffers: &mut Vec<metal::Buffer>,
        left_buffer: &BufferRef,
        right_buffer: &BufferRef,
        weight_buffer: &BufferRef,
        summed_buffer: &BufferRef,
        normed_buffer: &BufferRef,
        dim: usize,
        eps: f32,
    ) -> Result<()> {
        let dim = checked_u32(dim, "add rms dim")?;
        encoder.set_compute_pipeline_state(&self.add_rms_norm_row_f32);
        encoder.set_buffer(0, Some(left_buffer), 0);
        encoder.set_buffer(1, Some(right_buffer), 0);
        encoder.set_buffer(2, Some(weight_buffer), 0);
        encoder.set_buffer(3, Some(summed_buffer), 0);
        encoder.set_buffer(4, Some(normed_buffer), 0);
        set_u32_bytes(encoder, 5, &[dim], "add_rms_dim")?;
        set_f32_bytes(encoder, 6, &[eps], "add_rms_eps")?;
        profile_dispatch();
        encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(LM_HEAD_TG_WIDTH, 1, 1));
        post_dispatch_barrier(encoder);
        Ok(())
    }

    pub(super) fn encode_argmax(
        &self,
        encoder: &ComputeCommandEncoderRef,
        _owned_buffers: &mut Vec<metal::Buffer>,
        logits_buffer: &BufferRef,
        partial_values: &BufferRef,
        partial_indices: &BufferRef,
        output_index: &BufferRef,
        count: usize,
    ) -> Result<()> {
        let partial_count = count.div_ceil(256);
        self.encode_argmax_blocks(
            encoder,
            logits_buffer,
            partial_values,
            partial_indices,
            count,
        )?;
        self.encode_argmax_finalize(
            encoder,
            partial_values,
            partial_indices,
            output_index,
            partial_count,
        )
    }

    fn encode_argmax_blocks(
        &self,
        encoder: &ComputeCommandEncoderRef,
        logits_buffer: &BufferRef,
        partial_values: &BufferRef,
        partial_indices: &BufferRef,
        count: usize,
    ) -> Result<()> {
        self.encode_argmax_blocks_with_offset(
            encoder,
            logits_buffer,
            0,
            partial_values,
            partial_indices,
            count,
        )
    }

    pub(super) fn encode_argmax_blocks_with_offset(
        &self,
        encoder: &ComputeCommandEncoderRef,
        logits_buffer: &BufferRef,
        logits_offset: u64,
        partial_values: &BufferRef,
        partial_indices: &BufferRef,
        count: usize,
    ) -> Result<()> {
        let partial_count = count.div_ceil(256);
        let count = checked_u32(count, "argmax count")?;
        let partial_count = checked_u32(partial_count, "argmax partial_count")?;

        encoder.set_compute_pipeline_state(&self.argmax_blocks_f32);
        encoder.set_buffer(0, Some(logits_buffer), logits_offset);
        encoder.set_buffer(1, Some(partial_values), 0);
        encoder.set_buffer(2, Some(partial_indices), 0);
        set_u32_bytes(encoder, 3, &[count], "argmax_count")?;
        profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(NSUInteger::from(partial_count), 1, 1),
            MTLSize::new(LM_HEAD_TG_WIDTH, 1, 1),
        );
        post_dispatch_barrier(encoder);
        Ok(())
    }

    pub(super) fn encode_argmax_finalize(
        &self,
        encoder: &ComputeCommandEncoderRef,
        partial_values: &BufferRef,
        partial_indices: &BufferRef,
        output_index: &BufferRef,
        partial_count: usize,
    ) -> Result<()> {
        self.encode_argmax_finalize_with_offset(
            encoder,
            partial_values,
            partial_indices,
            output_index,
            0,
            partial_count,
        )
    }

    pub(super) fn encode_argmax_finalize_with_offset(
        &self,
        encoder: &ComputeCommandEncoderRef,
        partial_values: &BufferRef,
        partial_indices: &BufferRef,
        output_index: &BufferRef,
        output_offset: u64,
        partial_count: usize,
    ) -> Result<()> {
        let partial_count = checked_u32(partial_count, "argmax partial_count")?;
        encoder.set_compute_pipeline_state(&self.argmax_finalize_f32);
        encoder.set_buffer(0, Some(partial_values), 0);
        encoder.set_buffer(1, Some(partial_indices), 0);
        encoder.set_buffer(2, Some(output_index), output_offset);
        set_u32_bytes(encoder, 3, &[partial_count], "argmax_partial_count")?;
        profile_dispatch();
        encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(LM_HEAD_TG_WIDTH, 1, 1));
        post_dispatch_barrier(encoder);
        Ok(())
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "encodage Gumbel: buffers, offsets et paramètres sampling explicites"
    )]
    pub(super) fn encode_sample_gumbel_with_offsets(
        &self,
        encoder: &ComputeCommandEncoderRef,
        logits_buffer: &BufferRef,
        logits_offset: u64,
        output_index: &BufferRef,
        index_offset: u64,
        count: usize,
        temperature: f32,
        rng_state: u64,
    ) -> Result<()> {
        let count_u32 = checked_u32(count, "sample Gumbel count")?;
        let partial_count = count.div_ceil(256);
        let partial_values = self.private_f32_buffer(partial_count, "gumbel_partial_values")?;
        let partial_indices = self.private_u32_buffer(partial_count, "gumbel_partial_indices")?;

        encoder.set_compute_pipeline_state(&self.sample_gumbel_blocks_f32);
        encoder.set_buffer(0, Some(logits_buffer), logits_offset);
        encoder.set_buffer(1, Some(&partial_values), 0);
        encoder.set_buffer(2, Some(&partial_indices), 0);
        set_u32_bytes(encoder, 3, &[count_u32], "gumbel_count")?;
        set_f32_bytes(encoder, 4, &[temperature], "gumbel_temperature")?;
        set_u64_bytes(encoder, 5, &[rng_state], "gumbel_rng")?;
        profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(checked_nsuint(partial_count, "gumbel partial_count")?, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        post_dispatch_barrier(encoder);

        self.encode_argmax_finalize_with_offset(
            encoder,
            &partial_values,
            &partial_indices,
            output_index,
            index_offset,
            partial_count,
        )
    }

    fn encode_sample_gumbel(
        &self,
        encoder: &ComputeCommandEncoderRef,
        logits_buffer: &BufferRef,
        output_index: &BufferRef,
        count: usize,
        temperature: f32,
        rng_state: u64,
    ) -> Result<()> {
        self.encode_sample_gumbel_with_offsets(
            encoder,
            logits_buffer,
            0,
            output_index,
            0,
            count,
            temperature,
            rng_state,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "encodage sampler: paramètres sampling et buffers explicites"
    )]
    fn encode_sample_topk_topp(
        &self,
        encoder: &ComputeCommandEncoderRef,
        logits_buffer: &BufferRef,
        output_index: &BufferRef,
        count: usize,
        top_k: usize,
        temperature: f32,
        top_p: f32,
        rng_state: u64,
    ) -> Result<()> {
        self.encode_sample_topk_topp_with_offsets(
            encoder,
            logits_buffer,
            0,
            output_index,
            0,
            count,
            top_k,
            temperature,
            top_p,
            rng_state,
        )
    }

    /// Variante à offsets du sampler top-k/top-p : `logits_offset` adresse la
    /// ligne du flux dans des logits `[M, vocab]`, `index_offset` la case u32 du
    /// flux (duo light-batch, `rng_state` PAR FLUX). Mêmes kernels que le solo.
    #[expect(
        clippy::too_many_arguments,
        reason = "encodage sampler: buffers + offsets + paramètres sampling explicites"
    )]
    pub(super) fn encode_sample_topk_topp_with_offsets(
        &self,
        encoder: &ComputeCommandEncoderRef,
        logits_buffer: &BufferRef,
        logits_offset: u64,
        output_index: &BufferRef,
        index_offset: u64,
        count: usize,
        top_k: usize,
        temperature: f32,
        top_p: f32,
        rng_state: u64,
    ) -> Result<()> {
        if top_k == 0 && top_p >= 1.0 {
            return self.encode_sample_gumbel_with_offsets(
                encoder,
                logits_buffer,
                logits_offset,
                output_index,
                index_offset,
                count,
                temperature,
                rng_state,
            );
        }
        if top_k == 0 || top_k > MAX_SAMPLER_TOP_K || top_k > count {
            return Err(InferError::Dimension(format!(
                "sample top_k={top_k} invalide pour count={count}"
            )));
        }
        let dims = [
            checked_u32(count, "sample count")?,
            checked_u32(top_k, "sample top_k")?,
        ];
        let params = [temperature, top_p];
        let partial_count = count.div_ceil(256);
        let partial_len = partial_count
            .checked_mul(top_k)
            .ok_or_else(|| InferError::Metal("sample partial top-k déborde".to_string()))?;
        let partial_values = self.private_f32_buffer(partial_len, "sample_partial_values")?;
        let partial_indices = self.private_u32_buffer(partial_len, "sample_partial_indices")?;

        encoder.set_compute_pipeline_state(&self.sample_topk_blocks_f32);
        encoder.set_buffer(0, Some(logits_buffer), logits_offset);
        encoder.set_buffer(1, Some(&partial_values), 0);
        encoder.set_buffer(2, Some(&partial_indices), 0);
        set_u32_bytes(encoder, 3, &dims, "sample_block_dims")?;
        profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(checked_nsuint(partial_count, "sample partial_count")?, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        post_dispatch_barrier(encoder);

        let finalize_dims = [
            checked_u32(partial_len, "sample partial_len")?,
            checked_u32(top_k, "sample top_k")?,
        ];
        encoder.set_compute_pipeline_state(&self.sample_topk_finalize_f32);
        encoder.set_buffer(0, Some(&partial_values), 0);
        encoder.set_buffer(1, Some(&partial_indices), 0);
        encoder.set_buffer(2, Some(output_index), index_offset);
        set_u32_bytes(encoder, 3, &finalize_dims, "sample_finalize_dims")?;
        set_f32_bytes(encoder, 4, &params, "sample_params")?;
        set_u64_bytes(encoder, 5, &[rng_state], "sample_rng")?;
        profile_dispatch();
        encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(128, 1, 1));
        post_dispatch_barrier(encoder);
        Ok(())
    }
}
