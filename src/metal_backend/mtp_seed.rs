//! Semis batché de l'historique MTP : normes, FC, projections K/V, norme et
//! RoPE de K de la tête en un command buffer par bloc, sans relecture
//! intermédiaire. Le chemin CPU de `seed_mtp_history_chunk` reste l'oracle.

use super::matmul_encode::MATMUL_ROW_TG_WIDTH;
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
        states: &[f32],
        weights: MtpSeedWeights<'_>,
        spec: PrefillAttentionSpec,
        position_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let rows = spec.seq;
        let hidden = spec.hidden_dim;
        let rows_hidden = checked_len(rows, hidden, "semis MTP états")?;
        if rows == 0 || tokens.len() != rows || states.len() != rows_hidden {
            return Err(InferError::Dimension(format!(
                "semis MTP : {} tokens et {} états pour {rows}×{hidden}",
                tokens.len(),
                states.len()
            )));
        }
        for linear in [weights.fc, weights.k_proj, weights.v_proj] {
            ensure_biasless(linear, "semis MTP")?;
        }
        let concat_dim = checked_len(hidden, 2, "semis MTP concat")?;
        let kv_dim = checked_len(spec.kv_heads, spec.head_dim, "semis MTP kv_dim")?;
        let kv_len = checked_len(rows, kv_dim, "semis MTP kv")?;
        let state_buffer = self.upload_f32_buffer(states, "mtp_seed_states")?;
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
        // Ligne de concat = [norme(embedding) | norme(état)], comme `concat_rows`.
        for (input, weight, half) in [
            (&embed_buffer, &embedding_norm, 0),
            (&state_buffer, &hidden_norm, 1),
        ] {
            self.encode_rms_norm_rows_strided(
                encoder,
                input,
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
        reason = "norme par ligne à pas de sortie: buffers, offset et dimensions explicites"
    )]
    fn encode_rms_norm_rows_strided(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input_buffer: &BufferRef,
        weight_buffer: &BufferRef,
        output_buffer: &BufferRef,
        output_offset: u64,
        rows: usize,
        dim: usize,
        out_stride: usize,
        eps: f32,
    ) -> Result<()> {
        encoder.set_compute_pipeline_state(&self.rms_norm_rows_strided_f32);
        encoder.set_buffer(0, Some(input_buffer), 0);
        encoder.set_buffer(1, Some(weight_buffer), 0);
        encoder.set_buffer(2, Some(output_buffer), output_offset);
        set_u32_bytes(
            encoder,
            3,
            &[checked_u32(dim, "rms strided dim")?],
            "rms_strided_dim",
        )?;
        set_f32_bytes(encoder, 4, &[eps], "rms_strided_eps")?;
        set_u32_bytes(
            encoder,
            5,
            &[checked_u32(out_stride, "rms strided stride")?],
            "rms_strided_stride",
        )?;
        profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(checked_nsuint(rows, "rms strided rows")?, 1, 1),
            MTLSize::new(MATMUL_ROW_TG_WIDTH, 1, 1),
        );
        post_dispatch_barrier_buffer(encoder, output_buffer);
        Ok(())
    }
}
