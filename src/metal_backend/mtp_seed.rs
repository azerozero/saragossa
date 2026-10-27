//! Semis batché de l'historique MTP : normes, FC, projections K/V, norme et
//! RoPE de K de la tête en un command buffer par bloc, sans relecture
//! intermédiaire. Le chemin CPU de `seed_mtp_history_chunk` reste l'oracle.

use super::*;

/// Poids de la tête MTP lus par le semis.
#[derive(Clone, Copy)]
pub(crate) struct MtpSeedWeights<'a> {
    pub embed_tokens: &'a EmbeddingWeight,
    pub embed_scale: f32,
    pub embed_recast_bf16: bool,
    pub pre_fc_norm_embedding: &'a Tensor,
    pub pre_fc_norm_hidden: &'a Tensor,
    pub fc: &'a Linear,
    pub input_norm: &'a Tensor,
    pub k_proj: &'a Linear,
    pub v_proj: &'a Linear,
    pub k_norm: &'a Tensor,
}

/// États finaux d'un bloc du semis.
#[derive(Clone, Copy)]
pub(crate) enum MtpSeedStates<'a> {
    /// `[seq, hidden]` déjà normés, téléversés.
    Normed(&'a [f32]),
    /// Lignes `first_row..first_row + seq` du hidden final non normé, lues en
    /// place dans le buffer du prefill ; la norme finale passe sur le GPU au
    /// lieu d'une relecture et d'une norme CPU de tout le prompt.
    Raw {
        rows: &'a SharedF32Rows,
        first_row: usize,
        final_norm: &'a Tensor,
    },
}

impl MetalExecutor {
    /// K (normé, tourné à partir de `position_offset`) et V de la tête MTP pour
    /// `spec.seq` paires (embedding du token suivant, état final normé),
    /// chacun `[seq, kv_heads * head_dim]`.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si une projection porte un biais, si les dimensions
    /// divergent de `spec` ou si l'exécution Metal échoue.
    pub(crate) fn mtp_seed_kv_rows(
        &self,
        tokens: &[usize],
        states: MtpSeedStates<'_>,
        weights: MtpSeedWeights<'_>,
        spec: PrefillAttentionSpec,
        position_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let rows = spec.seq;
        let hidden = spec.hidden_dim;
        let rows_hidden = checked_len(rows, hidden, "semis MTP états")?;
        let states_len = match states {
            MtpSeedStates::Normed(states) => states.len(),
            MtpSeedStates::Raw {
                rows: raw,
                first_row,
                ..
            } => raw
                .len()
                .saturating_sub(checked_len(first_row, hidden, "semis MTP états")?)
                .min(rows_hidden),
        };
        if rows == 0 || tokens.len() != rows || states_len != rows_hidden {
            return Err(InferError::Dimension(format!(
                "semis MTP : {} tokens et {states_len} états pour {rows}×{hidden}",
                tokens.len(),
            )));
        }
        let concat_dim = checked_len(hidden, 2, "semis MTP concat")?;
        let kv_dim = checked_len(spec.kv_heads, spec.head_dim, "semis MTP kv_dim")?;
        // Les kernels lisent `dim` coefficients par norme et découpent K|V à
        // `kv_dim` : une forme divergente lirait hors du buffer ou mélangerait
        // K et V, là où l'oracle CPU refuse le sidecar.
        let final_norm = match states {
            MtpSeedStates::Raw { final_norm, .. } => Some(final_norm),
            MtpSeedStates::Normed(_) => None,
        };
        for (norm, dim, label) in final_norm
            .map(|norm| (norm, hidden, "final_norm"))
            .into_iter()
            .chain([
                (
                    weights.pre_fc_norm_embedding,
                    hidden,
                    "pre_fc_norm_embedding",
                ),
                (weights.pre_fc_norm_hidden, hidden, "pre_fc_norm_hidden"),
                (weights.input_norm, hidden, "input_norm"),
                (weights.k_norm, spec.head_dim, "k_norm"),
            ])
        {
            if !matches!(norm.shape(), [n] | [1, n] if *n == dim) {
                return Err(InferError::Dimension(format!(
                    "semis MTP : {label} {:?}, attendu [{dim}]",
                    norm.shape()
                )));
            }
        }
        for (linear, out_dim, in_dim, label) in [
            (weights.fc, hidden, concat_dim, "fc"),
            (weights.k_proj, kv_dim, hidden, "k_proj"),
            (weights.v_proj, kv_dim, hidden, "v_proj"),
        ] {
            if !matches!(linear.weight().shape(), [out, input] if *out == out_dim && *input == in_dim)
            {
                return Err(InferError::Dimension(format!(
                    "semis MTP : {label} {:?}, attendu [{out_dim}, {in_dim}]",
                    linear.weight().shape()
                )));
            }
            ensure_biasless(linear, "semis MTP")?;
        }
        if spec.rope_dims == 0 || spec.rope_dims > spec.head_dim || spec.rope_dims % 2 != 0 {
            return Err(InferError::Dimension(format!(
                "semis MTP : rope_dims {} pour head_dim {}",
                spec.rope_dims, spec.head_dim
            )));
        }
        let kv_len = checked_len(rows, kv_dim, "semis MTP kv")?;
        let state_input = match states {
            MtpSeedStates::Normed(states) => {
                (self.upload_f32_buffer(states, "mtp_seed_states")?, 0, None)
            }
            MtpSeedStates::Raw {
                rows: raw,
                first_row,
                final_norm,
            } => {
                let offset = checked_len(first_row, hidden, "semis MTP états")?
                    .checked_mul(std::mem::size_of::<f32>())
                    .and_then(|bytes| u64::try_from(bytes).ok())
                    .and_then(|bytes| bytes.checked_add(raw.byte_offset()))
                    .ok_or_else(|| {
                        InferError::Dimension("semis MTP : offset états déborde".to_string())
                    })?;
                let norm = self.cached_buffer_from_f32(final_norm.data(), "mtp_seed_final_norm")?;
                (raw.buffer().to_owned(), offset, Some(norm))
            }
        };
        let state_buffer = match &state_input {
            (_, _, Some(_)) => self.new_f32_buffer(rows_hidden, "mtp_seed_states_normed")?,
            (buffer, _, None) => buffer.clone(),
        };
        let concat = self.new_f32_buffer(
            checked_len(rows, concat_dim, "semis MTP concat")?,
            "mtp_seed_concat",
        )?;
        let fc_out = self.new_f32_buffer(rows_hidden, "mtp_seed_fc")?;
        let normed = self.new_f32_buffer(rows_hidden, "mtp_seed_normed")?;
        let k = self.new_f32_buffer(kv_len, "mtp_seed_k")?;
        let k_rope = self.new_f32_buffer(kv_len, "mtp_seed_k_rope")?;
        let v = self.new_f32_buffer(kv_len, "mtp_seed_v")?;
        let embedding_norm = self.cached_buffer_from_f32(
            weights.pre_fc_norm_embedding.data(),
            "mtp_seed_pre_fc_norm_embedding",
        )?;
        let hidden_norm = self.cached_buffer_from_f32(
            weights.pre_fc_norm_hidden.data(),
            "mtp_seed_pre_fc_norm_hidden",
        )?;
        let input_norm =
            self.cached_buffer_from_f32(weights.input_norm.data(), "mtp_seed_input_norm")?;
        let k_norm = self.cached_buffer_from_f32(weights.k_norm.data(), "mtp_seed_k_norm")?;

        let mut owned_buffers = Vec::new();
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let encoder_guard = EncoderEndGuard::new(encoder);
        let (embed_buffer, embed_dim) = self.encode_embed_weight_tokens(
            encoder,
            weights.embed_tokens,
            tokens,
            weights.embed_scale,
            weights.embed_recast_bf16,
        )?;
        if embed_dim != hidden {
            return Err(InferError::Dimension(format!(
                "semis MTP : embedding {embed_dim}, attendu {hidden}"
            )));
        }
        if let (raw, offset, Some(final_norm)) = &state_input {
            self.encode_rms_norm_rows_strided(
                encoder,
                raw,
                *offset,
                final_norm,
                &state_buffer,
                0,
                rows,
                hidden,
                hidden,
                spec.eps,
            )?;
        }
        // Ligne de concat = [norme(embedding) | norme(état)], comme `concat_rows`.
        for (input, weight, half) in [
            (&embed_buffer, &embedding_norm, 0),
            (&state_buffer, &hidden_norm, 1),
        ] {
            self.encode_rms_norm_rows_strided(
                encoder,
                input,
                0,
                weight,
                &concat,
                u64::try_from(half * hidden * std::mem::size_of::<f32>()).map_err(|_| {
                    InferError::Dimension("semis MTP : offset concat déborde".to_string())
                })?,
                rows,
                hidden,
                concat_dim,
                spec.eps,
            )?;
        }
        let fc_dim = self.encode_matmul_weight(
            encoder,
            &mut owned_buffers,
            &concat,
            rows,
            concat_dim,
            weights.fc.weight(),
            &fc_out,
        )?;
        if fc_dim != hidden {
            return Err(InferError::Dimension(format!(
                "semis MTP : fc sort {fc_dim}, attendu {hidden}"
            )));
        }
        self.encode_rms_norm_rows(
            encoder,
            &fc_out,
            &input_norm,
            &normed,
            rows,
            hidden,
            spec.eps,
        )?;
        // K|V concaténés : la sortie de 2·kv_dim passe le seuil du GEMM tuilé,
        // que chaque projection seule manque. Deux matmuls si les poids ne se
        // concatènent pas.
        let kv_sources = [weights.k_proj.weight(), weights.v_proj.weight()];
        let kv_concat = if self.concat_reusable(&kv_sources) {
            match self.resolve_concat_linear_weight_buffers(&kv_sources, "mtp_seed_kv_proj") {
                Ok(buffers) => Some(buffers),
                Err(InferError::Dimension(_)) => None,
                Err(error) => return Err(error),
            }
        } else {
            None
        };
        match kv_concat {
            Some(kv_weights) => {
                let kv_concat_dim = checked_len(kv_dim, 2, "semis MTP kv concat")?;
                let kv =
                    self.new_f32_buffer(checked_len(kv_len, 2, "semis MTP kv")?, "mtp_seed_kv")?;
                let out_dim = self.encode_matmul_weight_buffers(
                    encoder,
                    &normed,
                    rows,
                    hidden,
                    &kv_weights,
                    &kv,
                    false,
                )?;
                if out_dim != kv_concat_dim {
                    return Err(InferError::Dimension(format!(
                        "semis MTP : projection K|V sort {out_dim}, attendu {kv_concat_dim}"
                    )));
                }
                for (column, output) in [(0, &k), (kv_dim, &v)] {
                    self.encode_copy_cols(
                        encoder,
                        &kv,
                        u64::try_from(column * std::mem::size_of::<f32>()).map_err(|_| {
                            InferError::Dimension("semis MTP : offset K|V déborde".to_string())
                        })?,
                        output,
                        kv_dim,
                        kv_concat_dim,
                        kv_len,
                    )?;
                }
            }
            None => {
                for (linear, output) in [(weights.k_proj, &k), (weights.v_proj, &v)] {
                    let out_dim = self.encode_matmul_weight(
                        encoder,
                        &mut owned_buffers,
                        &normed,
                        rows,
                        hidden,
                        linear.weight(),
                        output,
                    )?;
                    if out_dim != kv_dim {
                        return Err(InferError::Dimension(format!(
                            "semis MTP : projection K/V sort {out_dim}, attendu {kv_dim}"
                        )));
                    }
                }
            }
        }
        if spec.value_norm {
            self.encode_rms_norm_heads_no_scale_rows(
                encoder,
                &v,
                &v,
                rows,
                spec.kv_heads,
                spec.head_dim,
                spec.eps,
            )?;
        }
        self.encode_rms_norm_rope_heads(
            encoder,
            &k,
            &k_norm,
            &k_rope,
            spec,
            spec.kv_heads,
            position_offset,
        )?;
        encoder_guard.end();
        commit_and_wait(command_buffer)?;
        Ok((
            Tensor::from_vec(vec![rows, kv_dim], read_f32_buffer(&k_rope, kv_len)?)?,
            Tensor::from_vec(vec![rows, kv_dim], read_f32_buffer(&v, kv_len)?)?,
        ))
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "copie de colonnes: buffers, offset et dimensions explicites"
    )]
    pub(super) fn encode_copy_cols(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input_buffer: &BufferRef,
        input_offset: u64,
        output_buffer: &BufferRef,
        cols: usize,
        src_stride: usize,
        len: usize,
    ) -> Result<()> {
        encoder.set_compute_pipeline_state(&self.copy_cols_f32);
        encoder.set_buffer(0, Some(input_buffer), input_offset);
        encoder.set_buffer(1, Some(output_buffer), 0);
        set_u32_bytes(encoder, 2, &[checked_u32(cols, "copy cols")?], "copy_cols")?;
        set_u32_bytes(
            encoder,
            3,
            &[checked_u32(src_stride, "copy cols stride")?],
            "copy_cols_stride",
        )?;
        set_u32_bytes(
            encoder,
            4,
            &[checked_u32(len, "copy cols len")?],
            "copy_cols_len",
        )?;
        self.dispatch_1d(encoder, &self.copy_cols_f32, len)
    }
}
