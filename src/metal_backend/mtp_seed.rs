//! Semis batché de l'historique MTP : normes, FC, projections K/V, norme et
//! RoPE de K de la tête, sans relecture intermédiaire. Les blocs s'encodent
//! dans le prefill ou dans leur propre command buffer sur le chemin de repli.
//! Le chemin CPU de `seed_mtp_history_chunk` reste l'oracle.

use super::buffers::embedding_indices;
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
    /// Scratch du prefill, consommé avant sa prochaine écriture GPU.
    Buffer {
        buffer: &'a BufferRef,
        final_norm: &'a Tensor,
    },
    /// Lignes `first_row..first_row + seq` du hidden final non normé, lues en
    /// place dans le buffer du prefill ; la norme finale passe sur le GPU au
    /// lieu d'une relecture et d'une norme CPU de tout le prompt.
    Raw {
        rows: &'a SharedF32Rows,
        first_row: usize,
        final_norm: &'a Tensor,
    },
}

/// Semis en blocs identiques à l'ancien chemin, indépendants du découpage du
/// prefill. Seul un bloc de hidden reste vivant entre deux morceaux.
pub(crate) struct MtpPrefillSeed<'a> {
    weights: MtpSeedWeights<'a>,
    spec: PrefillAttentionSpec,
    final_norm: &'a Tensor,
    tokens: &'a [usize],
    block_rows: usize,
    #[cfg(test)]
    pub(super) reuse_scratch_capacity: bool,
    pending: Buffer,
    filled: usize,
    seeded: usize,
    completed: bool,
    // Le setup résident vient après le prefill et peut refuser l'arène.
    // Les replis CPU et les préfixes réutilisés doivent encore lire les f32
    // d'origine : élargir du bf16 ne préserverait pas ces valeurs.
    output: [Buffer; 2],
}

impl<'a> MtpPrefillSeed<'a> {
    pub(crate) fn new(
        metal: &MetalExecutor,
        weights: MtpSeedWeights<'a>,
        spec: PrefillAttentionSpec,
        final_norm: &'a Tensor,
        tokens: &'a [usize],
        block_rows: usize,
    ) -> Result<Self> {
        if tokens.is_empty() || block_rows == 0 {
            return Err(InferError::Dimension("semis MTP vide".to_string()));
        }
        let kv_dim = checked_len(spec.kv_heads, spec.head_dim, "semis MTP kv_dim")?;
        let len = checked_len(tokens.len(), kv_dim, "semis MTP historique")?;
        Ok(Self {
            weights,
            spec,
            final_norm,
            tokens,
            block_rows,
            #[cfg(test)]
            reuse_scratch_capacity: true,
            pending: metal.private_f32_buffer(
                checked_len(block_rows, spec.hidden_dim, "semis MTP bloc")?,
                "mtp_seed_pending",
            )?,
            filled: 0,
            seeded: 0,
            completed: false,
            output: [
                metal.uncached_f32_buffer(len, "mtp_history_k")?,
                metal.uncached_f32_buffer(len, "mtp_history_v")?,
            ],
        })
    }

    pub(super) fn encode_chunk(
        &mut self,
        metal: &MetalExecutor,
        encoder: &ComputeCommandEncoderRef,
        hidden: &BufferRef,
        start: usize,
        rows: usize,
    ) -> Result<()> {
        self.completed = false;
        if start != self.seeded + self.filled {
            return Err(InferError::Dimension("semis MTP non contigu".to_string()));
        }
        let mut from = 0;
        let rows = rows.min(self.tokens.len().saturating_sub(start));
        let dim = self.spec.hidden_dim;
        let kv_dim = checked_len(self.spec.kv_heads, self.spec.head_dim, "semis MTP kv_dim")?;
        while from < rows {
            let count = (self.block_rows - self.filled).min(rows - from);
            metal.encode_copy_with_offsets(
                encoder,
                hidden,
                byte_len::<f32>(from * dim)?,
                &self.pending,
                byte_len::<f32>(self.filled * dim)?,
                count * dim,
            )?;
            from += count;
            self.filled += count;
            if self.filled == self.block_rows || self.seeded + self.filled == self.tokens.len() {
                // Le semis a son propre découpage, indépendant du prefill.
                // Reprendre les capacités du bloc plein jusque dans les
                // scratch internes des GEMM, sans changer leurs dimensions.
                #[cfg(test)]
                let capacity = if self.reuse_scratch_capacity {
                    self.block_rows
                } else {
                    self.filled
                };
                #[cfg(not(test))]
                let capacity = self.block_rows;
                let _scratch_rows = enter_scratch_row_capacity(self.filled, capacity);
                let (k, v) = metal.encode_mtp_seed_kv_rows(
                    encoder,
                    &self.tokens[self.seeded..self.seeded + self.filled],
                    MtpSeedStates::Buffer {
                        buffer: &self.pending,
                        final_norm: self.final_norm,
                    },
                    self.weights,
                    PrefillAttentionSpec {
                        seq: self.filled,
                        ..self.spec
                    },
                    self.seeded,
                )?;
                for (source, target) in [(&k, &self.output[0]), (&v, &self.output[1])] {
                    metal.encode_copy_with_offsets(
                        encoder,
                        source,
                        0,
                        target,
                        byte_len::<f32>(self.seeded * kv_dim)?,
                        self.filled * kv_dim,
                    )?;
                }
                self.seeded += self.filled;
                self.filled = 0;
            }
        }
        Ok(())
    }

    /// Le prefill publie cette barrière après l'attente du command buffer.
    pub(super) fn complete(&mut self) {
        self.completed = true;
    }

    /// Refuse de publier une vue tant que le GPU peut encore l'écrire.
    pub(crate) fn into_rows(self) -> Result<(PrefillKvRows, PrefillKvRows)> {
        if !self.completed || self.seeded != self.tokens.len() || self.filled != 0 {
            return Err(InferError::Dimension("semis MTP incomplet".to_string()));
        }
        let len = checked_len(
            self.seeded,
            self.spec.kv_heads * self.spec.head_dim,
            "semis MTP historique",
        )?;
        let [k, v] = self.output;
        // SAFETY: buffers f32 partagés propres au semis, taille validée à
        // l'allocation, GPU terminé par le prefill ; plus aucun écrivain.
        #[expect(unsafe_code, reason = "publication de l'historique MTP figé")]
        unsafe {
            Ok((
                PrefillKvRows::Shared(SharedF32Rows::new(k, 0, len)?),
                PrefillKvRows::Shared(SharedF32Rows::new(v, 0, len)?),
            ))
        }
    }
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
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let guard = EncoderEndGuard::new(encoder);
        let (k, v) =
            self.encode_mtp_seed_kv_rows(encoder, tokens, states, weights, spec, position_offset)?;
        guard.end();
        commit_and_wait(command_buffer)?;
        let kv_dim = checked_len(spec.kv_heads, spec.head_dim, "semis MTP kv_dim")?;
        let len = checked_len(spec.seq, kv_dim, "semis MTP kv")?;
        Ok((
            Tensor::from_vec(vec![spec.seq, kv_dim], read_f32_buffer(&k, len)?)?,
            Tensor::from_vec(vec![spec.seq, kv_dim], read_f32_buffer(&v, len)?)?,
        ))
    }

    fn encode_mtp_seed_kv_rows(
        &self,
        encoder: &ComputeCommandEncoderRef,
        tokens: &[usize],
        states: MtpSeedStates<'_>,
        weights: MtpSeedWeights<'_>,
        spec: PrefillAttentionSpec,
        position_offset: usize,
    ) -> Result<(Buffer, Buffer)> {
        let rows = spec.seq;
        let hidden = spec.hidden_dim;
        let rows_hidden = checked_len(rows, hidden, "semis MTP états")?;
        let states_len = match states {
            MtpSeedStates::Normed(states) => states.len(),
            MtpSeedStates::Buffer { buffer, .. } => {
                (buffer.length() as usize / std::mem::size_of::<f32>()).min(rows_hidden)
            }
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
            MtpSeedStates::Raw { final_norm, .. } | MtpSeedStates::Buffer { final_norm, .. } => {
                Some(final_norm)
            }
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
            MtpSeedStates::Buffer { buffer, final_norm } => (
                buffer.to_owned(),
                0,
                Some(self.cached_buffer_from_f32(final_norm.data(), "mtp_seed_final_norm")?),
            ),
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
        // Les indices restent propres au bloc : un upload dans le scratch
        // réécrirait les tokens des dispatchs précédents avant le commit.
        let indices = self.buffer_from_slice(
            &embedding_indices(weights.embed_tokens, tokens)?,
            "mtp_seed_indices",
        )?;
        let embed_weights = self.resolve_embedding_weight_buffers(weights.embed_tokens)?;
        let [_, embed_dim] = weights.embed_tokens.shape() else {
            return Err(InferError::Dimension(
                "semis MTP : embedding attendu rang 2".to_string(),
            ));
        };
        let embed_dim = *embed_dim;
        if embed_dim != hidden {
            return Err(InferError::Dimension(format!(
                "semis MTP : embedding {embed_dim}, attendu {hidden}"
            )));
        }
        let embed_buffer = self.new_f32_buffer(rows_hidden, "mtp_seed_embedding")?;
        self.encode_embeddings_from_indices_scaled(
            encoder,
            &embed_weights,
            &indices,
            &embed_buffer,
            embed_dim,
            rows,
            weights.embed_scale,
            weights.embed_recast_bf16,
        )?;
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
        Ok((k_rope, v))
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
