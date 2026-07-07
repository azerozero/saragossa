use super::super::*;
#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::metal_backend::MetalExecutor;

impl CausalDecoder {
    #[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
    fn reset_mtp_append_oracle(mtp: &mut ResidentMtpArena) {
        mtp.append_oracle_len = 0;
    }

    #[cfg(all(target_os = "macos", feature = "metal", not(feature = "devtools")))]
    fn reset_mtp_append_oracle(_mtp: &mut ResidentMtpArena) {}

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(in crate::decoder) fn start_mtp_resident_draft(
        &self,
        cache: &mut CausalDecoderCache,
        final_state: &Tensor,
        history_len: Option<usize>,
    ) -> Result<bool> {
        if let Some(history_len) = history_len {
            self.flush_mtp_resident_pending_append(cache, history_len)?;
        }
        let Some(arena) = cache.resident.as_mut() else {
            return Ok(false);
        };
        let Some(mtp) = arena.mtp.as_mut() else {
            return Ok(false);
        };
        let (rows, hidden) = final_state.as_matrix()?;
        if rows != 1 || hidden != self.final_norm.data().len() {
            return Err(InferError::Dimension(format!(
                "MTP résident final_state=[{rows},{hidden}], attendu [1,{}]",
                self.final_norm.data().len()
            )));
        }
        arena.state.upload(&mtp.hidden_a, final_state.as_row()?)?;
        mtp.kv.truncate(history_len.unwrap_or(0))?;
        mtp.current_is_a = true;
        Ok(true)
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(in crate::decoder) fn seed_mtp_resident_history(
        &self,
        cache: &mut CausalDecoderCache,
        history: &LayerKvCache,
    ) -> Result<()> {
        let Some(arena) = cache.resident.as_mut() else {
            return Ok(());
        };
        let Some(mtp) = arena.mtp.as_mut() else {
            return Ok(());
        };
        mtp.pending_append_count = 0;
        Self::reset_mtp_append_oracle(mtp);
        mtp.kv.seed(&history.keys, &history.values, history.len())
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(in crate::decoder) fn truncate_mtp_resident_history(
        &self,
        cache: &mut CausalDecoderCache,
        len: usize,
    ) -> Result<()> {
        let Some(arena) = cache.resident.as_mut() else {
            return Ok(());
        };
        let Some(mtp) = arena.mtp.as_mut() else {
            return Ok(());
        };
        mtp.pending_append_count = 0;
        Self::reset_mtp_append_oracle(mtp);
        mtp.kv.truncate(len)
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(in crate::decoder) fn flush_mtp_resident_pending_append(
        &self,
        cache: &mut CausalDecoderCache,
        expected_history_len: usize,
    ) -> Result<()> {
        let Some(metal) = self.forward_runtime().metal_executor() else {
            return Ok(());
        };
        let head_dim = self
            .config
            .head_dim
            .ok_or_else(|| InferError::Dimension("head_dim manquant (MTP résident)".to_string()))?;
        let theta = self
            .config
            .rope_theta
            .ok_or_else(|| InferError::Config("rope_theta manquant (MTP résident)".to_string()))?;
        let hidden = self.final_norm.data().len();
        let CausalDecoderCache { resident, .. } = cache;
        let Some(arena) = resident.as_mut() else {
            return Ok(());
        };
        let pending = arena
            .mtp
            .as_ref()
            .map(|mtp| mtp.pending_append_count)
            .unwrap_or(0);
        if pending == 0 {
            if let Some(mtp) = arena.mtp.as_mut() {
                mtp.kv.truncate(expected_history_len)?;
            }
            return Ok(());
        }
        let command_buffer = arena.state.queue().new_command_buffer();
        let _barrier_guard = crate::metal_backend::resident_concurrent_enabled()
            .then(crate::metal_backend::install_dispatch_barrier_scope);
        let _namespace_guard =
            crate::metal_backend::install_scratch_namespace(arena.state.scratch_namespace());
        let encoder = new_resident_compute_encoder(command_buffer);
        let encoder_guard = crate::metal_backend::EncoderEndGuard::new(encoder);
        let mut owned = Vec::new();
        let mut scratch = Vec::new();
        let dims = FullAttnLayerDims {
            hidden,
            q_heads: self.config.num_attention_heads,
            kv_heads: self.config.num_key_value_heads,
            head_dim,
            rope_dims: self.config.rope_dims.unwrap_or(head_dim),
            position: 0,
            eps: self.config.rms_eps,
            theta,
            attn_output_gate: self.config.attn_output_gate,
        };
        let Some(mtp) = arena.mtp.as_mut() else {
            return Ok(());
        };
        self.encode_mtp_pending_append_rows(
            metal,
            &arena.state,
            &arena.embed_tokens,
            &arena.dense_tail_score,
            mtp,
            encoder_guard.encoder(),
            &mut owned,
            dims,
            expected_history_len,
            &mut scratch,
        )?;
        encoder_guard.end();
        crate::metal_backend::commit_and_wait(command_buffer)?;
        #[cfg(feature = "devtools")]
        if crate::decoder::flags::env_flag("RETI_RUST_MTP_APPEND_KV_ORACLE", false) {
            if let Some(mtp) = arena.mtp.as_mut() {
                self.check_mtp_append_kv_oracle(mtp, expected_history_len)?;
            }
        }
        drop(scratch);
        Ok(())
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[expect(
        clippy::too_many_arguments,
        reason = "encodage interne MTP: executor, arène, encoder, dims et liveness scratch"
    )]
    fn encode_mtp_pending_append_rows(
        &self,
        metal: &MetalExecutor,
        state: &DecodeResidentState,
        embed_tokens: &MetalEmbeddingWeightBuffers,
        dense_tail_score: &metal::Buffer,
        mtp: &mut ResidentMtpArena,
        encoder: &metal::ComputeCommandEncoderRef,
        owned: &mut Vec<metal::Buffer>,
        dims: FullAttnLayerDims,
        expected_history_len: usize,
        scratch: &mut Vec<ScratchLease>,
    ) -> Result<()> {
        let pending_count = mtp.pending_append_count;
        if pending_count == 0 {
            mtp.kv.truncate(expected_history_len)?;
            return Ok(());
        }
        let pending_start = mtp.pending_append_start;
        let pending_end = pending_start
            .checked_add(pending_count)
            .ok_or_else(|| InferError::Dimension("MTP pending append déborde".to_string()))?;
        if pending_count > 2 || pending_end != expected_history_len {
            return Err(InferError::Dimension(format!(
                "MTP pending append incohérent start={pending_start} count={pending_count} attendu_end={expected_history_len}"
            )));
        }
        {
            mtp.kv.truncate(pending_start)?;
            #[cfg(feature = "devtools")]
            if crate::decoder::flags::env_flag("RETI_RUST_MTP_APPEND_KV_ORACLE", false) {
                Self::encode_copy_mtp_kv_prefix(
                    metal,
                    encoder,
                    &mtp.kv,
                    &mtp.append_oracle_kv,
                    pending_start,
                )?;
                mtp.append_oracle_kv.truncate(pending_start)?;
                Self::encode_mtp_append_rows_serial_to_kv(
                    metal,
                    state,
                    embed_tokens,
                    dense_tail_score,
                    &mtp.pre_fc_norm_embedding,
                    &mtp.pre_fc_norm_hidden,
                    &mtp.fc,
                    &mtp.layer,
                    &mtp.verify_hidden_rows,
                    &mtp.pending_append_indices,
                    &mut mtp.append_oracle_kv,
                    encoder,
                    owned,
                    dims,
                    pending_count,
                    pending_start,
                    scratch,
                )?;
                mtp.append_oracle_len = pending_end;
            }
            #[cfg(feature = "devtools")]
            if !crate::decoder::flags::env_flag("RETI_RUST_MTP_APPEND_KV_ORACLE", false) {
                mtp.append_oracle_len = 0;
            }
            Self::encode_mtp_append_rows_serial_to_kv(
                metal,
                state,
                embed_tokens,
                dense_tail_score,
                &mtp.pre_fc_norm_embedding,
                &mtp.pre_fc_norm_hidden,
                &mtp.fc,
                &mtp.layer,
                &mtp.verify_hidden_rows,
                &mtp.pending_append_indices,
                &mut mtp.kv,
                encoder,
                owned,
                dims,
                pending_count,
                pending_start,
                scratch,
            )?;
            mtp.pending_append_count = 0;
            mtp.current_is_a = true;
        }
        Ok(())
    }

    #[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
    fn encode_copy_mtp_kv_prefix(
        metal: &MetalExecutor,
        encoder: &metal::ComputeCommandEncoderRef,
        source: &FullAttentionMetalState,
        target: &FullAttentionMetalState,
        rows: usize,
    ) -> Result<()> {
        let cells = rows
            .checked_mul(source.kv_dim())
            .ok_or_else(|| InferError::Dimension("MTP oracle KV prefix déborde".to_string()))?;
        if cells == 0 {
            return Ok(());
        }
        if source.kv_dim() != target.kv_dim()
            || source.keys().element() != target.keys().element()
            || source.values().element() != target.values().element()
        {
            return Err(InferError::Dimension(
                "MTP oracle KV dtype/dim incompatible".to_string(),
            ));
        }
        match source.keys().element() {
            GpuElement::F32 => {
                metal.encode_copy_with_offsets(
                    encoder,
                    source.keys().buffer(),
                    0,
                    target.keys().buffer(),
                    0,
                    cells,
                )?;
                metal.encode_copy_with_offsets(
                    encoder,
                    source.values().buffer(),
                    0,
                    target.values().buffer(),
                    0,
                    cells,
                )?;
            }
            GpuElement::Bf16 => {
                metal.encode_copy_u16_with_offsets(
                    encoder,
                    source.keys().buffer(),
                    0,
                    target.keys().buffer(),
                    0,
                    cells,
                )?;
                metal.encode_copy_u16_with_offsets(
                    encoder,
                    source.values().buffer(),
                    0,
                    target.values().buffer(),
                    0,
                    cells,
                )?;
            }
            GpuElement::U32 => {
                return Err(InferError::Metal(
                    "MTP oracle KV ne supporte pas u32".to_string(),
                ));
            }
        }
        Ok(())
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[expect(
        clippy::too_many_arguments,
        reason = "oracle bit-identique de l'ancien append MTP série"
    )]
    fn encode_mtp_append_rows_serial_to_kv(
        metal: &MetalExecutor,
        state: &DecodeResidentState,
        embed_tokens: &MetalEmbeddingWeightBuffers,
        dense_tail_score: &metal::Buffer,
        pre_fc_norm_embedding: &metal::Buffer,
        pre_fc_norm_hidden: &metal::Buffer,
        fc: &MetalLinearWeightBuffers,
        layer: &ResidentFullDenseBuffers,
        hidden_rows: &GpuTensor,
        token_indices: &GpuTensor,
        kv: &mut FullAttentionMetalState,
        encoder: &metal::ComputeCommandEncoderRef,
        owned: &mut Vec<metal::Buffer>,
        dims: FullAttnLayerDims,
        rows: usize,
        start_position: usize,
        scratch: &mut Vec<ScratchLease>,
    ) -> Result<()> {
        let hidden = dims.hidden;
        let concat_width = hidden
            .checked_mul(2)
            .ok_or_else(|| InferError::Dimension("MTP oracle concat déborde".to_string()))?;
        for row in 0..rows {
            let row_offset = row
                .checked_mul(hidden)
                .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
                .and_then(|value| u64::try_from(value).ok())
                .ok_or_else(|| {
                    InferError::Metal("MTP oracle append row offset déborde".to_string())
                })?;
            let token_offset = row
                .checked_mul(std::mem::size_of::<u32>())
                .and_then(|value| u64::try_from(value).ok())
                .ok_or_else(|| {
                    InferError::Metal("MTP oracle append token offset déborde".to_string())
                })?;
            let embed_row = state.scratch().lease(hidden, GpuElement::F32)?;
            let hidden_row = state.scratch().lease(hidden, GpuElement::F32)?;
            let embedding_norm = state.scratch().lease(hidden, GpuElement::F32)?;
            let hidden_norm = state.scratch().lease(hidden, GpuElement::F32)?;
            let concat = state.scratch().lease(concat_width, GpuElement::F32)?;
            let fc_out = state.scratch().lease(hidden, GpuElement::F32)?;
            let layer_out = state.scratch().lease(hidden, GpuElement::F32)?;
            metal.encode_embedding_from_index_buffers_with_offset(
                encoder,
                embed_tokens,
                token_indices.buffer(),
                token_offset,
                embed_row.tensor().buffer(),
                hidden,
            )?;
            metal.encode_copy_with_offsets(
                encoder,
                hidden_rows.buffer(),
                row_offset,
                hidden_row.tensor().buffer(),
                0,
                hidden,
            )?;
            metal.encode_rms_norm_rows(
                encoder,
                embed_row.tensor().buffer(),
                pre_fc_norm_embedding,
                embedding_norm.tensor().buffer(),
                1,
                hidden,
                dims.eps,
            )?;
            metal.encode_rms_norm_rows(
                encoder,
                hidden_row.tensor().buffer(),
                pre_fc_norm_hidden,
                hidden_norm.tensor().buffer(),
                1,
                hidden,
                dims.eps,
            )?;
            metal.encode_copy_with_offsets(
                encoder,
                embedding_norm.tensor().buffer(),
                0,
                concat.tensor().buffer(),
                0,
                hidden,
            )?;
            let hidden_offset = hidden
                .checked_mul(std::mem::size_of::<f32>())
                .and_then(|value| u64::try_from(value).ok())
                .ok_or_else(|| InferError::Metal("MTP oracle concat offset déborde".to_string()))?;
            metal.encode_copy_with_offsets(
                encoder,
                hidden_norm.tensor().buffer(),
                0,
                concat.tensor().buffer(),
                hidden_offset,
                hidden,
            )?;
            let fc_dim = metal.encode_matmul_weight_buffers(
                encoder,
                concat.tensor().buffer(),
                1,
                concat_width,
                fc,
                fc_out.tensor().buffer(),
                false,
            )?;
            if fc_dim != hidden {
                return Err(InferError::Dimension(format!(
                    "MTP oracle fc sort {fc_dim}, attendu {hidden}"
                )));
            }
            let weights = FullAttnDenseLayerWeights {
                input_norm: &layer.input_norm,
                qkv_proj: layer.qkv_proj.as_ref(),
                q_proj: &layer.q_proj,
                k_proj: &layer.k_proj,
                v_proj: &layer.v_proj,
                o_proj: &layer.o_proj,
                q_norm: &layer.q_norm,
                k_norm: &layer.k_norm,
                post_norm: &layer.post_norm,
                gate_proj: &layer.gate_proj,
                up_proj: &layer.up_proj,
                down_proj: &layer.down_proj,
                tail_score: dense_tail_score,
            };
            let mut row_dims = dims;
            row_dims.position = start_position.checked_add(row).ok_or_else(|| {
                InferError::Dimension("MTP oracle append position déborde".to_string())
            })?;
            state.encode_full_attn_dense_layer(
                metal,
                encoder,
                owned,
                kv,
                weights,
                row_dims,
                fc_out.tensor().buffer(),
                layer_out.tensor().buffer(),
            )?;
            scratch.push(embed_row);
            scratch.push(hidden_row);
            scratch.push(embedding_norm);
            scratch.push(hidden_norm);
            scratch.push(concat);
            scratch.push(fc_out);
            scratch.push(layer_out);
        }
        Ok(())
    }

    #[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
    fn check_mtp_append_kv_oracle(
        &self,
        mtp: &mut ResidentMtpArena,
        expected_len: usize,
    ) -> Result<()> {
        if mtp.append_oracle_len == 0 {
            return Ok(());
        }
        if mtp.append_oracle_len != expected_len {
            return Err(InferError::Dimension(format!(
                "MTP oracle KV len={} attendu={expected_len}",
                mtp.append_oracle_len
            )));
        }
        let cells = expected_len
            .checked_mul(mtp.kv.kv_dim())
            .ok_or_else(|| InferError::Dimension("MTP oracle compare déborde".to_string()))?;
        Self::compare_mtp_kv_buffer_bits(
            "keys",
            mtp.kv.keys(),
            mtp.append_oracle_kv.keys(),
            cells,
        )?;
        Self::compare_mtp_kv_buffer_bits(
            "values",
            mtp.kv.values(),
            mtp.append_oracle_kv.values(),
            cells,
        )?;
        eprintln!(
            "mtp_append_kv_oracle bit_equal=true rows={} cells={}",
            expected_len, cells
        );
        mtp.append_oracle_len = 0;
        Ok(())
    }

    #[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
    fn compare_mtp_kv_buffer_bits(
        label: &str,
        actual: &GpuTensor,
        oracle: &GpuTensor,
        cells: usize,
    ) -> Result<()> {
        if actual.element() != oracle.element() {
            return Err(InferError::Dimension(format!(
                "MTP oracle {label} dtype {:?} != {:?}",
                actual.element(),
                oracle.element()
            )));
        }
        match actual.element() {
            GpuElement::F32 => {
                let left = crate::metal_backend::read_f32_buffer(actual.buffer(), cells)?;
                let right = crate::metal_backend::read_f32_buffer(oracle.buffer(), cells)?;
                for (index, (left, right)) in left.iter().zip(right.iter()).enumerate() {
                    if left.to_bits() != right.to_bits() {
                        return Err(InferError::Config(format!(
                            "MTP oracle KV {label} divergent index={index} actual={left:e} oracle={right:e}"
                        )));
                    }
                }
            }
            GpuElement::Bf16 => {
                let left = crate::metal_backend::read_u16_buffer(actual.buffer(), cells)?;
                let right = crate::metal_backend::read_u16_buffer(oracle.buffer(), cells)?;
                for (index, (left, right)) in left.iter().zip(right.iter()).enumerate() {
                    if left != right {
                        return Err(InferError::Config(format!(
                            "MTP oracle KV {label} divergent index={index} actual={left:#06x} oracle={right:#06x}"
                        )));
                    }
                }
            }
            GpuElement::U32 => {
                return Err(InferError::Metal("MTP oracle KV u32 inattendu".to_string()));
            }
        }
        Ok(())
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(in crate::decoder) fn append_mtp_resident_history_steps(
        &self,
        cache: &mut CausalDecoderCache,
        entries: &[(Tensor, usize)],
        start_position: usize,
    ) -> Result<()> {
        if entries.is_empty() {
            return Ok(());
        }
        let Some(metal) = self.forward_runtime().metal_executor() else {
            return Ok(());
        };
        let head_dim = self
            .config
            .head_dim
            .ok_or_else(|| InferError::Dimension("head_dim manquant (MTP résident)".to_string()))?;
        let theta = self
            .config
            .rope_theta
            .ok_or_else(|| InferError::Config("rope_theta manquant (MTP résident)".to_string()))?;
        let eps = self.config.rms_eps;
        let rope_dims = self.config.rope_dims.unwrap_or(head_dim);
        let q_heads = self.config.num_attention_heads;
        let kv_heads = self.config.num_key_value_heads;
        let hidden = self.final_norm.data().len();
        let CausalDecoderCache { resident, .. } = cache;
        let arena = resident.as_mut().ok_or_else(|| {
            InferError::Metal("arène résidente absente (MTP history résident)".to_string())
        })?;
        let mtp = arena
            .mtp
            .as_mut()
            .ok_or_else(|| InferError::Metal("arène MTP résidente absente".to_string()))?;
        if entries.len() > mtp.draft_indices.len() {
            return Err(InferError::Dimension(format!(
                "MTP history batch={} > capacité résidente {}",
                entries.len(),
                mtp.draft_indices.len()
            )));
        }
        let mut hidden_rows = Vec::with_capacity(entries.len() * hidden);
        let mut token_ids = Vec::with_capacity(entries.len());
        for (state, token) in entries {
            let (rows, cols) = state.as_matrix()?;
            if rows != 1 || cols != hidden {
                return Err(InferError::Dimension(format!(
                    "MTP history hidden={:?}, attendu [1,{hidden}]",
                    state.shape()
                )));
            }
            hidden_rows.extend_from_slice(state.as_row()?);
            token_ids.push(u32::try_from(*token).map_err(|_| {
                InferError::Dimension(format!("token MTP history hors u32: {token}"))
            })?);
        }
        arena.state.upload_u32(&mtp.draft_indices, &token_ids)?;
        let state_rows = arena
            .state
            .scratch()
            .lease(hidden_rows.len(), GpuElement::F32)?;
        arena.state.upload(state_rows.tensor(), &hidden_rows)?;

        let command_buffer = arena.state.queue().new_command_buffer();
        let _barrier_guard = crate::metal_backend::resident_concurrent_enabled()
            .then(crate::metal_backend::install_dispatch_barrier_scope);
        let _namespace_guard =
            crate::metal_backend::install_scratch_namespace(arena.state.scratch_namespace());
        let encoder = new_resident_compute_encoder(command_buffer);
        let encoder_guard = crate::metal_backend::EncoderEndGuard::new(encoder);
        let mut owned: Vec<metal::Buffer> = Vec::new();
        let mut scratch: Vec<ScratchLease> = vec![state_rows];

        for index in 0..entries.len() {
            let state_offset = index
                .checked_mul(hidden)
                .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
                .and_then(|value| u64::try_from(value).ok())
                .ok_or_else(|| InferError::Metal("MTP history state offset déborde".to_string()))?;
            let token_offset = index
                .checked_mul(std::mem::size_of::<u32>())
                .and_then(|value| u64::try_from(value).ok())
                .ok_or_else(|| InferError::Metal("MTP history token offset déborde".to_string()))?;
            let position = start_position
                .checked_add(index)
                .ok_or_else(|| InferError::Dimension("MTP history position déborde".to_string()))?;
            let embedding_norm = arena.state.scratch().lease(hidden, GpuElement::F32)?;
            let hidden_norm = arena.state.scratch().lease(hidden, GpuElement::F32)?;
            let layer_out = arena.state.scratch().lease(hidden, GpuElement::F32)?;
            metal.encode_copy_with_offsets(
                encoder_guard.encoder(),
                scratch[0].tensor().buffer(),
                state_offset,
                mtp.hidden_a.buffer(),
                0,
                hidden,
            )?;
            metal.encode_embedding_from_index_buffers_with_offset(
                encoder_guard.encoder(),
                &arena.embed_tokens,
                mtp.draft_indices.buffer(),
                token_offset,
                mtp.embedding.buffer(),
                hidden,
            )?;
            metal.encode_rms_norm_rows(
                encoder_guard.encoder(),
                mtp.embedding.buffer(),
                &mtp.pre_fc_norm_embedding,
                embedding_norm.tensor().buffer(),
                1,
                hidden,
                eps,
            )?;
            metal.encode_rms_norm_rows(
                encoder_guard.encoder(),
                mtp.hidden_a.buffer(),
                &mtp.pre_fc_norm_hidden,
                hidden_norm.tensor().buffer(),
                1,
                hidden,
                eps,
            )?;
            metal.encode_copy_with_offsets(
                encoder_guard.encoder(),
                embedding_norm.tensor().buffer(),
                0,
                mtp.concat.buffer(),
                0,
                hidden,
            )?;
            let hidden_offset = hidden
                .checked_mul(std::mem::size_of::<f32>())
                .and_then(|value| u64::try_from(value).ok())
                .ok_or_else(|| InferError::Metal("MTP concat hidden offset déborde".to_string()))?;
            metal.encode_copy_with_offsets(
                encoder_guard.encoder(),
                hidden_norm.tensor().buffer(),
                0,
                mtp.concat.buffer(),
                hidden_offset,
                hidden,
            )?;
            let fc_dim = metal.encode_matmul_weight_buffers(
                encoder_guard.encoder(),
                mtp.concat.buffer(),
                1,
                hidden.checked_mul(2).ok_or_else(|| {
                    InferError::Dimension("MTP fc input hidden déborde".to_string())
                })?,
                &mtp.fc,
                mtp.fc_out.buffer(),
                false,
            )?;
            if fc_dim != hidden {
                return Err(InferError::Dimension(format!(
                    "MTP fc sort {fc_dim}, attendu {hidden}"
                )));
            }
            let weights = FullAttnDenseLayerWeights {
                input_norm: &mtp.layer.input_norm,
                qkv_proj: mtp.layer.qkv_proj.as_ref(),
                q_proj: &mtp.layer.q_proj,
                k_proj: &mtp.layer.k_proj,
                v_proj: &mtp.layer.v_proj,
                o_proj: &mtp.layer.o_proj,
                q_norm: &mtp.layer.q_norm,
                k_norm: &mtp.layer.k_norm,
                post_norm: &mtp.layer.post_norm,
                gate_proj: &mtp.layer.gate_proj,
                up_proj: &mtp.layer.up_proj,
                down_proj: &mtp.layer.down_proj,
                tail_score: &arena.dense_tail_score,
            };
            let dims = FullAttnLayerDims {
                hidden,
                q_heads,
                kv_heads,
                head_dim,
                rope_dims,
                position,
                eps,
                theta,
                attn_output_gate: self.config.attn_output_gate,
            };
            arena.state.encode_full_attn_dense_layer(
                metal,
                encoder_guard.encoder(),
                &mut owned,
                &mut mtp.kv,
                weights,
                dims,
                mtp.fc_out.buffer(),
                layer_out.tensor().buffer(),
            )?;
            metal.encode_rms_norm_rows(
                encoder_guard.encoder(),
                layer_out.tensor().buffer(),
                &mtp.norm,
                mtp.hidden_b.buffer(),
                1,
                hidden,
                eps,
            )?;
            scratch.push(embedding_norm);
            scratch.push(hidden_norm);
            scratch.push(layer_out);
        }

        encoder_guard.end();
        crate::metal_backend::commit_and_wait(command_buffer)?;
        drop(scratch);
        mtp.current_is_a = true;
        Ok(())
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(in crate::decoder) fn next_mtp_draft_resident(
        &self,
        cache: &mut CausalDecoderCache,
        token_id: usize,
        position: usize,
    ) -> Result<usize> {
        let Some(metal) = self.forward_runtime().metal_executor() else {
            return Err(InferError::Metal(
                "executor Metal absent pour MTP résident".to_string(),
            ));
        };
        let head_dim = self
            .config
            .head_dim
            .ok_or_else(|| InferError::Dimension("head_dim manquant (MTP résident)".to_string()))?;
        let theta = self
            .config
            .rope_theta
            .ok_or_else(|| InferError::Config("rope_theta manquant (MTP résident)".to_string()))?;
        let eps = self.config.rms_eps;
        let rope_dims = self.config.rope_dims.unwrap_or(head_dim);
        let q_heads = self.config.num_attention_heads;
        let kv_heads = self.config.num_key_value_heads;
        let hidden = self.final_norm.data().len();
        let token_u32 = u32::try_from(token_id)
            .map_err(|_| InferError::Dimension(format!("token MTP hors u32: {token_id}")))?;

        let CausalDecoderCache { resident, .. } = cache;
        let arena = resident.as_mut().ok_or_else(|| {
            InferError::Metal("arène résidente absente (MTP résident)".to_string())
        })?;
        let mtp = arena
            .mtp
            .as_mut()
            .ok_or_else(|| InferError::Metal("arène MTP résidente absente".to_string()))?;

        arena.state.upload_u32(&mtp.index, &[token_u32])?;
        let (input_hidden, output_hidden) = if mtp.current_is_a {
            (mtp.hidden_a.buffer(), mtp.hidden_b.buffer())
        } else {
            (mtp.hidden_b.buffer(), mtp.hidden_a.buffer())
        };
        let embedding_norm = arena.state.scratch().lease(hidden, GpuElement::F32)?;
        let hidden_norm = arena.state.scratch().lease(hidden, GpuElement::F32)?;
        let layer_out = arena.state.scratch().lease(hidden, GpuElement::F32)?;
        let command_buffer = arena.state.queue().new_command_buffer();
        let _barrier_guard = crate::metal_backend::resident_concurrent_enabled()
            .then(crate::metal_backend::install_dispatch_barrier_scope);
        let _namespace_guard =
            crate::metal_backend::install_scratch_namespace(arena.state.scratch_namespace());
        let encoder = new_resident_compute_encoder(command_buffer);
        let mut owned: Vec<metal::Buffer> = Vec::new();

        metal.encode_embedding_from_index_buffers(
            encoder,
            &arena.embed_tokens,
            mtp.index.buffer(),
            mtp.embedding.buffer(),
            hidden,
        )?;
        metal.encode_rms_norm_rows(
            encoder,
            mtp.embedding.buffer(),
            &mtp.pre_fc_norm_embedding,
            embedding_norm.tensor().buffer(),
            1,
            hidden,
            eps,
        )?;
        metal.encode_rms_norm_rows(
            encoder,
            input_hidden,
            &mtp.pre_fc_norm_hidden,
            hidden_norm.tensor().buffer(),
            1,
            hidden,
            eps,
        )?;
        metal.encode_copy_with_offsets(
            encoder,
            embedding_norm.tensor().buffer(),
            0,
            mtp.concat.buffer(),
            0,
            hidden,
        )?;
        let hidden_offset = hidden
            .checked_mul(std::mem::size_of::<f32>())
            .and_then(|value| u64::try_from(value).ok())
            .ok_or_else(|| InferError::Metal("MTP concat hidden offset déborde".to_string()))?;
        metal.encode_copy_with_offsets(
            encoder,
            hidden_norm.tensor().buffer(),
            0,
            mtp.concat.buffer(),
            hidden_offset,
            hidden,
        )?;
        let fc_dim = metal.encode_matmul_weight_buffers(
            encoder,
            mtp.concat.buffer(),
            1,
            hidden
                .checked_mul(2)
                .ok_or_else(|| InferError::Dimension("MTP fc input hidden déborde".to_string()))?,
            &mtp.fc,
            mtp.fc_out.buffer(),
            false,
        )?;
        if fc_dim != hidden {
            return Err(InferError::Dimension(format!(
                "MTP fc sort {fc_dim}, attendu {hidden}"
            )));
        }
        let weights = FullAttnDenseLayerWeights {
            input_norm: &mtp.layer.input_norm,
            qkv_proj: mtp.layer.qkv_proj.as_ref(),
            q_proj: &mtp.layer.q_proj,
            k_proj: &mtp.layer.k_proj,
            v_proj: &mtp.layer.v_proj,
            o_proj: &mtp.layer.o_proj,
            q_norm: &mtp.layer.q_norm,
            k_norm: &mtp.layer.k_norm,
            post_norm: &mtp.layer.post_norm,
            gate_proj: &mtp.layer.gate_proj,
            up_proj: &mtp.layer.up_proj,
            down_proj: &mtp.layer.down_proj,
            tail_score: &arena.dense_tail_score,
        };
        let dims = FullAttnLayerDims {
            hidden,
            q_heads,
            kv_heads,
            head_dim,
            rope_dims,
            position,
            eps,
            theta,
            attn_output_gate: self.config.attn_output_gate,
        };
        arena.state.encode_full_attn_dense_layer(
            metal,
            encoder,
            &mut owned,
            &mut mtp.kv,
            weights,
            dims,
            mtp.fc_out.buffer(),
            layer_out.tensor().buffer(),
        )?;
        metal.encode_rms_norm_rows(
            encoder,
            layer_out.tensor().buffer(),
            &mtp.norm,
            output_hidden,
            1,
            hidden,
            eps,
        )?;
        metal.encode_lm_head_argmax_buffers(
            encoder,
            &mut owned,
            output_hidden,
            &mtp.draft_lm_head,
            mtp.index.buffer(),
            hidden,
        )?;
        encoder.end_encoding();
        crate::metal_backend::commit_and_wait(command_buffer)?;

        let raw = crate::metal_backend::read_u32_buffer(mtp.index.buffer(), 1)?
            .into_iter()
            .next()
            .ok_or_else(|| InferError::Metal("MTP résident sans index".to_string()))?;
        mtp.current_is_a = !mtp.current_is_a;
        usize::try_from(raw).map_err(|_| InferError::Metal(format!("index MTP trop grand: {raw}")))
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(in crate::decoder) fn next_mtp_drafts_resident(
        &self,
        cache: &mut CausalDecoderCache,
        first_token_id: usize,
        max_draft_tokens: usize,
        position_offset: usize,
    ) -> Result<Option<Vec<usize>>> {
        if max_draft_tokens == 0 {
            return Ok(Some(Vec::new()));
        }
        let Some(metal) = self.forward_runtime().metal_executor() else {
            return Ok(None);
        };
        let head_dim = self
            .config
            .head_dim
            .ok_or_else(|| InferError::Dimension("head_dim manquant (MTP résident)".to_string()))?;
        let theta = self
            .config
            .rope_theta
            .ok_or_else(|| InferError::Config("rope_theta manquant (MTP résident)".to_string()))?;
        let eps = self.config.rms_eps;
        let rope_dims = self.config.rope_dims.unwrap_or(head_dim);
        let q_heads = self.config.num_attention_heads;
        let kv_heads = self.config.num_key_value_heads;
        let hidden = self.final_norm.data().len();
        let token_u32 = u32::try_from(first_token_id)
            .map_err(|_| InferError::Dimension(format!("token MTP hors u32: {first_token_id}")))?;

        let CausalDecoderCache { resident, .. } = cache;
        let arena = resident.as_mut().ok_or_else(|| {
            InferError::Metal("arène résidente absente (MTP résident)".to_string())
        })?;
        let mtp = arena
            .mtp
            .as_mut()
            .ok_or_else(|| InferError::Metal("arène MTP résidente absente".to_string()))?;
        if max_draft_tokens > mtp.draft_indices.len() {
            return Err(InferError::Dimension(format!(
                "draft MTP max={max_draft_tokens} > capacité résidente {}",
                mtp.draft_indices.len()
            )));
        }

        arena.state.upload_u32(&mtp.index, &[token_u32])?;
        let command_buffer = arena.state.queue().new_command_buffer();
        let _barrier_guard = crate::metal_backend::resident_concurrent_enabled()
            .then(crate::metal_backend::install_dispatch_barrier_scope);
        let _namespace_guard =
            crate::metal_backend::install_scratch_namespace(arena.state.scratch_namespace());
        let encoder = new_resident_compute_encoder(command_buffer);
        let encoder_guard = crate::metal_backend::EncoderEndGuard::new(encoder);
        let mut owned: Vec<metal::Buffer> = Vec::new();
        let mut scratch: Vec<ScratchLease> = Vec::new();
        let mut current_is_a = mtp.current_is_a;

        for position in 0..max_draft_tokens {
            let (input_hidden, output_hidden) = if current_is_a {
                (mtp.hidden_a.buffer(), mtp.hidden_b.buffer())
            } else {
                (mtp.hidden_b.buffer(), mtp.hidden_a.buffer())
            };
            let (index_buffer, index_offset) = if position == 0 {
                (mtp.index.buffer(), 0)
            } else {
                let offset = position
                    .checked_sub(1)
                    .and_then(|value| value.checked_mul(std::mem::size_of::<u32>()))
                    .and_then(|value| u64::try_from(value).ok())
                    .ok_or_else(|| {
                        InferError::Metal("MTP draft input index offset déborde".to_string())
                    })?;
                (mtp.draft_indices.buffer(), offset)
            };
            let output_index_offset = position
                .checked_mul(std::mem::size_of::<u32>())
                .and_then(|value| u64::try_from(value).ok())
                .ok_or_else(|| {
                    InferError::Metal("MTP draft output index offset déborde".to_string())
                })?;
            let embedding_norm = arena.state.scratch().lease(hidden, GpuElement::F32)?;
            let hidden_norm = arena.state.scratch().lease(hidden, GpuElement::F32)?;
            let layer_out = arena.state.scratch().lease(hidden, GpuElement::F32)?;

            metal.encode_embedding_from_index_buffers_with_offset(
                encoder_guard.encoder(),
                &arena.embed_tokens,
                index_buffer,
                index_offset,
                mtp.embedding.buffer(),
                hidden,
            )?;
            metal.encode_rms_norm_rows(
                encoder_guard.encoder(),
                mtp.embedding.buffer(),
                &mtp.pre_fc_norm_embedding,
                embedding_norm.tensor().buffer(),
                1,
                hidden,
                eps,
            )?;
            metal.encode_rms_norm_rows(
                encoder_guard.encoder(),
                input_hidden,
                &mtp.pre_fc_norm_hidden,
                hidden_norm.tensor().buffer(),
                1,
                hidden,
                eps,
            )?;
            metal.encode_copy_with_offsets(
                encoder_guard.encoder(),
                embedding_norm.tensor().buffer(),
                0,
                mtp.concat.buffer(),
                0,
                hidden,
            )?;
            let hidden_offset = hidden
                .checked_mul(std::mem::size_of::<f32>())
                .and_then(|value| u64::try_from(value).ok())
                .ok_or_else(|| InferError::Metal("MTP concat hidden offset déborde".to_string()))?;
            metal.encode_copy_with_offsets(
                encoder_guard.encoder(),
                hidden_norm.tensor().buffer(),
                0,
                mtp.concat.buffer(),
                hidden_offset,
                hidden,
            )?;
            let fc_dim = metal.encode_matmul_weight_buffers(
                encoder_guard.encoder(),
                mtp.concat.buffer(),
                1,
                hidden.checked_mul(2).ok_or_else(|| {
                    InferError::Dimension("MTP fc input hidden déborde".to_string())
                })?,
                &mtp.fc,
                mtp.fc_out.buffer(),
                false,
            )?;
            if fc_dim != hidden {
                return Err(InferError::Dimension(format!(
                    "MTP fc sort {fc_dim}, attendu {hidden}"
                )));
            }
            let weights = FullAttnDenseLayerWeights {
                input_norm: &mtp.layer.input_norm,
                qkv_proj: mtp.layer.qkv_proj.as_ref(),
                q_proj: &mtp.layer.q_proj,
                k_proj: &mtp.layer.k_proj,
                v_proj: &mtp.layer.v_proj,
                o_proj: &mtp.layer.o_proj,
                q_norm: &mtp.layer.q_norm,
                k_norm: &mtp.layer.k_norm,
                post_norm: &mtp.layer.post_norm,
                gate_proj: &mtp.layer.gate_proj,
                up_proj: &mtp.layer.up_proj,
                down_proj: &mtp.layer.down_proj,
                tail_score: &arena.dense_tail_score,
            };
            let dims = FullAttnLayerDims {
                hidden,
                q_heads,
                kv_heads,
                head_dim,
                rope_dims,
                position: position_offset.checked_add(position).ok_or_else(|| {
                    InferError::Dimension("MTP position résidente déborde".to_string())
                })?,
                eps,
                theta,
                attn_output_gate: self.config.attn_output_gate,
            };
            arena.state.encode_full_attn_dense_layer(
                metal,
                encoder_guard.encoder(),
                &mut owned,
                &mut mtp.kv,
                weights,
                dims,
                mtp.fc_out.buffer(),
                layer_out.tensor().buffer(),
            )?;
            metal.encode_rms_norm_rows(
                encoder_guard.encoder(),
                layer_out.tensor().buffer(),
                &mtp.norm,
                output_hidden,
                1,
                hidden,
                eps,
            )?;
            metal.encode_lm_head_argmax_buffers_with_index_offset(
                encoder_guard.encoder(),
                &mut owned,
                output_hidden,
                &mtp.draft_lm_head,
                mtp.draft_indices.buffer(),
                0,
                output_index_offset,
                hidden,
            )?;
            current_is_a = !current_is_a;
            scratch.push(embedding_norm);
            scratch.push(hidden_norm);
            scratch.push(layer_out);
        }

        encoder_guard.end();
        crate::metal_backend::commit_and_wait(command_buffer)?;
        drop(scratch);
        mtp.current_is_a = current_is_a;
        let raw =
            crate::metal_backend::read_u32_buffer(mtp.draft_indices.buffer(), max_draft_tokens)?;
        raw.into_iter()
            .map(|index| {
                usize::try_from(index)
                    .map_err(|_| InferError::Metal(format!("index MTP trop grand: {index}")))
            })
            .collect::<Result<Vec<_>>>()
            .map(Some)
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(in crate::decoder) fn next_mtp_spec_one_resident(
        &self,
        cache: &mut CausalDecoderCache,
        final_state: &Tensor,
        primary: usize,
        history_len: usize,
        stop_token_ids: &[usize],
    ) -> Result<Option<(usize, usize, Tensor, Option<(usize, Tensor)>)>> {
        let Some(metal) = self.forward_runtime().metal_executor() else {
            return Ok(None);
        };
        let head_dim = self
            .config
            .head_dim
            .ok_or_else(|| InferError::Dimension("head_dim manquant (MTP résident)".to_string()))?;
        let theta = self
            .config
            .rope_theta
            .ok_or_else(|| InferError::Config("rope_theta manquant (MTP résident)".to_string()))?;
        let eps = self.config.rms_eps;
        let rope_dims = self.config.rope_dims.unwrap_or(head_dim);
        let q_heads = self.config.num_attention_heads;
        let kv_heads = self.config.num_key_value_heads;
        let hidden = self.final_norm.data().len();
        // Mode fresh (défaut) : cache MTP self-only (position 0, vidé par
        // pas), ni draft#2 ni historique committed (prouvé inerte : committed
        // ≡ cycle byte-identique). `mtp_history` = position/troncature du cache.
        let fresh = crate::decoder::flags::mtp_fresh_cache_enabled();
        let mtp_history = if fresh { 0 } else { history_len };
        let primary_u32 = u32::try_from(primary)
            .map_err(|_| InferError::Dimension(format!("token MTP hors u32: {primary}")))?;
        let linear_dims = if self
            .layers
            .iter()
            .enumerate()
            .any(|(index, _)| !self.config.is_full_attention_layer(index))
        {
            let la_config = self.config.linear_attention_config()?;
            let la_spec = LinearAttentionStepSpec {
                num_key_heads: la_config.num_key_heads,
                num_value_heads: la_config.num_value_heads,
                key_head_dim: la_config.key_head_dim,
                value_head_dim: la_config.value_head_dim,
                conv_kernel_dim: la_config.conv_kernel_dim,
                rms_eps: la_config.rms_eps,
            };
            let key_dim = la_config.key_dim()?;
            let value_dim = la_config.value_dim()?;
            let conv_dim = key_dim
                .checked_mul(2)
                .and_then(|twice| twice.checked_add(value_dim))
                .ok_or_else(|| {
                    InferError::Shape("conv_dim déborde (MTP verify résident)".to_string())
                })?;
            Some((la_spec, key_dim, value_dim, conv_dim))
        } else {
            None
        };

        let CausalDecoderCache {
            layers,
            position: cache_position,
            resident,
        } = cache;
        let Some(arena) = resident.as_mut() else {
            return Ok(None);
        };
        if arena.state.gpu_timer().is_some() {
            return Ok(None);
        }
        let trunk_batch_supported = self.layers.iter().enumerate().all(|(index, layer)| {
            match (
                self.config.is_full_attention_layer(index),
                layer.mlp.as_ref(),
            ) {
                (true, Some(FeedForward::Dense(_))) => {
                    matches!(
                        arena.layers.get(index),
                        Some(ResidentLayerBuffers::FullDense(_))
                    )
                }
                (false, Some(FeedForward::Dense(_))) => {
                    matches!(
                        arena.layers.get(index),
                        Some(ResidentLayerBuffers::LinearDense(_))
                    )
                }
                _ => false,
            }
        });
        if !trunk_batch_supported {
            return Ok(None);
        }
        let Some(mtp) = arena.mtp.as_mut() else {
            return Ok(None);
        };
        let (rows, state_hidden) = final_state.as_matrix()?;
        if rows != 1 || state_hidden != hidden {
            return Err(InferError::Dimension(format!(
                "MTP fused final_state=[{rows},{state_hidden}], attendu [1,{hidden}]"
            )));
        }

        arena.state.upload(&mtp.hidden_a, final_state.as_row()?)?;
        arena.state.upload_u32(&mtp.index, &[primary_u32])?;
        mtp.kv.truncate(mtp_history)?;
        mtp.current_is_a = true;

        let embed = self.embed_scaled(&[primary])?;
        let (_, embed_hidden) = embed.as_matrix()?;
        if embed_hidden != hidden {
            return Err(InferError::Dimension(format!(
                "embedding hidden={embed_hidden}, attendu {hidden}"
            )));
        }
        arena.state.upload(&arena.hidden_a, embed.data())?;

        let linear_captures = if linear_dims.is_some() {
            let capture_rows = 1usize;
            let linear_states = layers
                .iter()
                .map(|layer| layer.linear.metal_state())
                .collect::<Vec<_>>();
            Some(metal.lease_linear_attn_state_captures(
                arena.state.scratch(),
                &linear_states,
                capture_rows,
            )?)
        } else {
            None
        };
        let command_buffer = arena.state.queue().new_command_buffer();
        let _barrier_guard = crate::metal_backend::resident_concurrent_enabled()
            .then(crate::metal_backend::install_dispatch_barrier_scope);
        let _namespace_guard =
            crate::metal_backend::install_scratch_namespace(arena.state.scratch_namespace());
        let encoder = new_resident_compute_encoder(command_buffer);
        let encoder_guard = crate::metal_backend::EncoderEndGuard::new(encoder);
        let mut owned: Vec<metal::Buffer> = Vec::new();
        let mut scratch: Vec<ScratchLease> = Vec::new();

        {
            let mtp = arena
                .mtp
                .as_mut()
                .ok_or_else(|| InferError::Metal("arène MTP résidente absente".to_string()))?;
            let embedding_norm = arena.state.scratch().lease(hidden, GpuElement::F32)?;
            let hidden_norm = arena.state.scratch().lease(hidden, GpuElement::F32)?;
            let layer_out = arena.state.scratch().lease(hidden, GpuElement::F32)?;
            metal.encode_embedding_from_index_buffers(
                encoder_guard.encoder(),
                &arena.embed_tokens,
                mtp.index.buffer(),
                mtp.embedding.buffer(),
                hidden,
            )?;
            metal.encode_rms_norm_rows(
                encoder_guard.encoder(),
                mtp.embedding.buffer(),
                &mtp.pre_fc_norm_embedding,
                embedding_norm.tensor().buffer(),
                1,
                hidden,
                eps,
            )?;
            metal.encode_rms_norm_rows(
                encoder_guard.encoder(),
                mtp.hidden_a.buffer(),
                &mtp.pre_fc_norm_hidden,
                hidden_norm.tensor().buffer(),
                1,
                hidden,
                eps,
            )?;
            metal.encode_copy_with_offsets(
                encoder_guard.encoder(),
                embedding_norm.tensor().buffer(),
                0,
                mtp.concat.buffer(),
                0,
                hidden,
            )?;
            let hidden_offset = hidden
                .checked_mul(std::mem::size_of::<f32>())
                .and_then(|value| u64::try_from(value).ok())
                .ok_or_else(|| InferError::Metal("MTP concat hidden offset déborde".to_string()))?;
            metal.encode_copy_with_offsets(
                encoder_guard.encoder(),
                hidden_norm.tensor().buffer(),
                0,
                mtp.concat.buffer(),
                hidden_offset,
                hidden,
            )?;
            let fc_dim = metal.encode_matmul_weight_buffers(
                encoder_guard.encoder(),
                mtp.concat.buffer(),
                1,
                hidden.checked_mul(2).ok_or_else(|| {
                    InferError::Dimension("MTP fc input hidden déborde".to_string())
                })?,
                &mtp.fc,
                mtp.fc_out.buffer(),
                false,
            )?;
            if fc_dim != hidden {
                return Err(InferError::Dimension(format!(
                    "MTP fc sort {fc_dim}, attendu {hidden}"
                )));
            }
            let weights = FullAttnDenseLayerWeights {
                input_norm: &mtp.layer.input_norm,
                qkv_proj: mtp.layer.qkv_proj.as_ref(),
                q_proj: &mtp.layer.q_proj,
                k_proj: &mtp.layer.k_proj,
                v_proj: &mtp.layer.v_proj,
                o_proj: &mtp.layer.o_proj,
                q_norm: &mtp.layer.q_norm,
                k_norm: &mtp.layer.k_norm,
                post_norm: &mtp.layer.post_norm,
                gate_proj: &mtp.layer.gate_proj,
                up_proj: &mtp.layer.up_proj,
                down_proj: &mtp.layer.down_proj,
                tail_score: &arena.dense_tail_score,
            };
            let dims = FullAttnLayerDims {
                hidden,
                q_heads,
                kv_heads,
                head_dim,
                rope_dims,
                position: mtp_history,
                eps,
                theta,
                attn_output_gate: self.config.attn_output_gate,
            };
            arena.state.encode_full_attn_dense_layer(
                metal,
                encoder_guard.encoder(),
                &mut owned,
                &mut mtp.kv,
                weights,
                dims,
                mtp.fc_out.buffer(),
                layer_out.tensor().buffer(),
            )?;
            metal.encode_rms_norm_rows(
                encoder_guard.encoder(),
                layer_out.tensor().buffer(),
                &mtp.norm,
                mtp.hidden_b.buffer(),
                1,
                hidden,
                eps,
            )?;
            metal.encode_lm_head_argmax_buffers_with_index_offset(
                encoder_guard.encoder(),
                &mut owned,
                mtp.hidden_b.buffer(),
                &mtp.draft_lm_head,
                mtp.draft_indices.buffer(),
                0,
                0,
                hidden,
            )?;
            scratch.push(embedding_norm);
            scratch.push(hidden_norm);
            scratch.push(layer_out);
        }

        let verify_rows = 2usize;
        let verify_elements = verify_rows
            .checked_mul(hidden)
            .ok_or_else(|| InferError::Dimension("verify MTP rows déborde".to_string()))?;
        let verify_a = arena
            .state
            .scratch()
            .lease(verify_elements, GpuElement::F32)?;
        let verify_b = arena
            .state
            .scratch()
            .lease(verify_elements, GpuElement::F32)?;
        let draft_embed = arena.state.scratch().lease(hidden, GpuElement::F32)?;
        arena.state.upload(verify_a.tensor(), embed.data())?;
        {
            let mtp = arena
                .mtp
                .as_ref()
                .ok_or_else(|| InferError::Metal("arène MTP résidente absente".to_string()))?;
            metal.encode_embedding_from_index_buffers(
                encoder_guard.encoder(),
                &arena.embed_tokens,
                mtp.draft_indices.buffer(),
                draft_embed.tensor().buffer(),
                hidden,
            )?;
        }
        let draft_row_offset = hidden
            .checked_mul(std::mem::size_of::<f32>())
            .and_then(|value| u64::try_from(value).ok())
            .ok_or_else(|| InferError::Metal("verify draft row offset déborde".to_string()))?;
        metal.encode_copy_with_offsets(
            encoder_guard.encoder(),
            draft_embed.tensor().buffer(),
            0,
            verify_a.tensor().buffer(),
            draft_row_offset,
            hidden,
        )?;

        let mut current = verify_a.tensor().buffer();
        let mut other = verify_b.tensor().buffer();
        let target_position = *cache_position;
        for (index, layer) in self.layers.iter().enumerate() {
            let layer_cache = &mut layers[index];
            if self.config.is_full_attention_layer(index) {
                let dims = FullAttnLayerDims {
                    hidden,
                    q_heads,
                    kv_heads,
                    head_dim,
                    rope_dims,
                    position: target_position,
                    eps,
                    theta,
                    attn_output_gate: self.config.attn_output_gate,
                };
                self.encode_resident_full_dense_layer_rows(
                    metal,
                    arena,
                    layer_cache,
                    layer,
                    index,
                    encoder_guard.encoder(),
                    &mut owned,
                    dims,
                    verify_rows,
                    current,
                    other,
                )?;
            } else {
                let Some((la_spec, key_dim, value_dim, conv_dim)) = linear_dims else {
                    return Err(InferError::Config(
                        "dims linear-attn résidentes absentes".to_string(),
                    ));
                };
                let res_dims = LinearAttnResidentDims {
                    in_dim: hidden,
                    conv_dim,
                    value_dim,
                    key_dim,
                };
                let captures = linear_captures
                    .as_ref()
                    .and_then(|(captures, _)| captures.get(index))
                    .and_then(|captures| captures.as_ref())
                    .map(Vec::as_slice);
                self.encode_resident_linear_dense_layer_rows(
                    metal,
                    arena,
                    layer_cache,
                    layer,
                    index,
                    encoder_guard.encoder(),
                    &mut owned,
                    la_spec,
                    res_dims,
                    verify_rows,
                    hidden,
                    eps,
                    current,
                    other,
                    captures,
                )?;
            }
            std::mem::swap(&mut current, &mut other);
        }

        let verify_final = arena
            .state
            .scratch()
            .lease(verify_elements, GpuElement::F32)?;
        let target_final = arena.state.scratch().lease(hidden, GpuElement::F32)?;
        let bonus_final = arena.state.scratch().lease(hidden, GpuElement::F32)?;
        let target_indices = arena.state.scratch().lease(verify_rows, GpuElement::U32)?;
        metal.encode_rms_norm_rows(
            encoder_guard.encoder(),
            current,
            &arena.final_norm,
            verify_final.tensor().buffer(),
            verify_rows,
            hidden,
            eps,
        )?;
        metal.encode_copy_with_offsets(
            encoder_guard.encoder(),
            verify_final.tensor().buffer(),
            0,
            target_final.tensor().buffer(),
            0,
            hidden,
        )?;
        metal.encode_copy_with_offsets(
            encoder_guard.encoder(),
            verify_final.tensor().buffer(),
            draft_row_offset,
            bonus_final.tensor().buffer(),
            0,
            hidden,
        )?;
        metal.encode_lm_head_argmax_two_rows_buffers(
            encoder_guard.encoder(),
            verify_final.tensor().buffer(),
            &arena.lm_head,
            target_indices.tensor().buffer(),
            0,
            0,
            hidden,
        )?;

        if !fresh {
            let mtp = arena
                .mtp
                .as_mut()
                .ok_or_else(|| InferError::Metal("arène MTP résidente absente".to_string()))?;
            let embedding_norm = arena.state.scratch().lease(hidden, GpuElement::F32)?;
            let hidden_norm = arena.state.scratch().lease(hidden, GpuElement::F32)?;
            let layer_out = arena.state.scratch().lease(hidden, GpuElement::F32)?;
            metal.encode_embedding_from_index_buffers(
                encoder_guard.encoder(),
                &arena.embed_tokens,
                target_indices.tensor().buffer(),
                mtp.embedding.buffer(),
                hidden,
            )?;
            metal.encode_rms_norm_rows(
                encoder_guard.encoder(),
                mtp.embedding.buffer(),
                &mtp.pre_fc_norm_embedding,
                embedding_norm.tensor().buffer(),
                1,
                hidden,
                eps,
            )?;
            metal.encode_rms_norm_rows(
                encoder_guard.encoder(),
                target_final.tensor().buffer(),
                &mtp.pre_fc_norm_hidden,
                hidden_norm.tensor().buffer(),
                1,
                hidden,
                eps,
            )?;
            metal.encode_copy_with_offsets(
                encoder_guard.encoder(),
                embedding_norm.tensor().buffer(),
                0,
                mtp.concat.buffer(),
                0,
                hidden,
            )?;
            let hidden_offset = hidden
                .checked_mul(std::mem::size_of::<f32>())
                .and_then(|value| u64::try_from(value).ok())
                .ok_or_else(|| InferError::Metal("MTP concat hidden offset déborde".to_string()))?;
            metal.encode_copy_with_offsets(
                encoder_guard.encoder(),
                hidden_norm.tensor().buffer(),
                0,
                mtp.concat.buffer(),
                hidden_offset,
                hidden,
            )?;
            let fc_dim = metal.encode_matmul_weight_buffers(
                encoder_guard.encoder(),
                mtp.concat.buffer(),
                1,
                hidden.checked_mul(2).ok_or_else(|| {
                    InferError::Dimension("MTP fc input hidden déborde".to_string())
                })?,
                &mtp.fc,
                mtp.fc_out.buffer(),
                false,
            )?;
            if fc_dim != hidden {
                return Err(InferError::Dimension(format!(
                    "MTP fc sort {fc_dim}, attendu {hidden}"
                )));
            }
            let weights = FullAttnDenseLayerWeights {
                input_norm: &mtp.layer.input_norm,
                qkv_proj: mtp.layer.qkv_proj.as_ref(),
                q_proj: &mtp.layer.q_proj,
                k_proj: &mtp.layer.k_proj,
                v_proj: &mtp.layer.v_proj,
                o_proj: &mtp.layer.o_proj,
                q_norm: &mtp.layer.q_norm,
                k_norm: &mtp.layer.k_norm,
                post_norm: &mtp.layer.post_norm,
                gate_proj: &mtp.layer.gate_proj,
                up_proj: &mtp.layer.up_proj,
                down_proj: &mtp.layer.down_proj,
                tail_score: &arena.dense_tail_score,
            };
            let dims = FullAttnLayerDims {
                hidden,
                q_heads,
                kv_heads,
                head_dim,
                rope_dims,
                position: history_len.checked_add(1).ok_or_else(|| {
                    InferError::Dimension("MTP append position déborde".to_string())
                })?,
                eps,
                theta,
                attn_output_gate: self.config.attn_output_gate,
            };
            arena.state.encode_full_attn_dense_layer(
                metal,
                encoder_guard.encoder(),
                &mut owned,
                &mut mtp.kv,
                weights,
                dims,
                mtp.fc_out.buffer(),
                layer_out.tensor().buffer(),
            )?;
            metal.encode_rms_norm_rows(
                encoder_guard.encoder(),
                layer_out.tensor().buffer(),
                &mtp.norm,
                mtp.hidden_a.buffer(),
                1,
                hidden,
                eps,
            )?;
            scratch.push(embedding_norm);
            scratch.push(hidden_norm);
            scratch.push(layer_out);
            mtp.current_is_a = true;
        }

        scratch.push(verify_a);
        scratch.push(verify_b);
        scratch.push(draft_embed);
        scratch.push(verify_final);

        encoder_guard.end();
        crate::metal_backend::commit_and_wait(command_buffer)?;
        let draft_raw = {
            let arena = resident.as_ref().ok_or_else(|| {
                InferError::Metal("arène résidente absente après MTP fused".to_string())
            })?;
            crate::metal_backend::read_u32_buffer(
                arena
                    .mtp
                    .as_ref()
                    .ok_or_else(|| InferError::Metal("arène MTP résidente absente".to_string()))?
                    .draft_indices
                    .buffer(),
                1,
            )?
            .into_iter()
            .next()
            .ok_or_else(|| InferError::Metal("MTP fused sans draft".to_string()))?
        };
        let mut target_rows =
            crate::metal_backend::read_u32_buffer(target_indices.tensor().buffer(), verify_rows)?
                .into_iter();
        let target_raw = target_rows
            .next()
            .ok_or_else(|| InferError::Metal("MTP fused sans target".to_string()))?;
        let bonus_raw = target_rows
            .next()
            .ok_or_else(|| InferError::Metal("MTP fused sans bonus".to_string()))?;
        let state = crate::metal_backend::read_f32_buffer(target_final.tensor().buffer(), hidden)?;
        let bonus_state =
            crate::metal_backend::read_f32_buffer(bonus_final.tensor().buffer(), hidden)?;
        drop(scratch);
        drop(target_indices);
        drop(bonus_final);
        drop(target_final);
        let draft = usize::try_from(draft_raw)
            .map_err(|_| InferError::Metal(format!("draft MTP trop grand: {draft_raw}")))?;
        let target = usize::try_from(target_raw)
            .map_err(|_| InferError::Metal(format!("target MTP trop grand: {target_raw}")))?;
        let bonus = usize::try_from(bonus_raw)
            .map_err(|_| InferError::Metal(format!("bonus MTP trop grand: {bonus_raw}")))?;
        let keep_bonus = draft == target && !stop_token_ids.contains(&target);
        if keep_bonus {
            *cache_position = target_position.checked_add(verify_rows).ok_or_else(|| {
                InferError::Dimension("position verify MTP batch déborde".to_string())
            })?;
        } else {
            let committed_position = target_position.checked_add(1).ok_or_else(|| {
                InferError::Dimension("position rollback MTP batch déborde".to_string())
            })?;
            *cache_position = committed_position;
            for layer in layers.iter_mut() {
                if let Some(full) = layer.full.as_mut() {
                    full.truncate(committed_position)?;
                }
            }
            if let Some(linear_captures) = linear_captures.as_ref() {
                let pairs = layers
                    .iter()
                    .zip(linear_captures.0.iter())
                    .filter_map(|(layer, captures)| {
                        let current = layer.linear.metal_state()?;
                        let capture = captures.as_ref()?.first()?;
                        Some((current, capture))
                    })
                    .collect::<Vec<_>>();
                metal.restore_linear_attn_states(&pairs)?;
            }
            if let Some(arena) = resident.as_mut() {
                if let Some(mtp) = arena.mtp.as_mut() {
                    mtp.kv.truncate(mtp_history.checked_add(1).ok_or_else(|| {
                        InferError::Dimension("taille historique MTP déborde".to_string())
                    })?)?;
                }
            }
        }
        let bonus = if keep_bonus {
            Some((bonus, Tensor::row(bonus_state)?))
        } else {
            None
        };
        Ok(Some((draft, target, Tensor::row(state)?, bonus)))
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(in crate::decoder) fn next_mtp_spec_two_resident(
        &self,
        cache: &mut CausalDecoderCache,
        final_state: &Tensor,
        primary: usize,
        history_len: usize,
        stop_token_ids: &[usize],
    ) -> Result<Option<ResidentMtpSpecTwoOutput>> {
        let Some(metal) = self.forward_runtime().metal_executor() else {
            return Ok(None);
        };
        let head_dim = self
            .config
            .head_dim
            .ok_or_else(|| InferError::Dimension("head_dim manquant (MTP résident)".to_string()))?;
        let theta = self
            .config
            .rope_theta
            .ok_or_else(|| InferError::Config("rope_theta manquant (MTP résident)".to_string()))?;
        let eps = self.config.rms_eps;
        let rope_dims = self.config.rope_dims.unwrap_or(head_dim);
        let q_heads = self.config.num_attention_heads;
        let kv_heads = self.config.num_key_value_heads;
        let hidden = self.final_norm.data().len();
        let primary_u32 = u32::try_from(primary)
            .map_err(|_| InferError::Dimension(format!("token MTP hors u32: {primary}")))?;
        let linear_dims = if self
            .layers
            .iter()
            .enumerate()
            .any(|(index, _)| !self.config.is_full_attention_layer(index))
        {
            let la_config = self.config.linear_attention_config()?;
            let la_spec = LinearAttentionStepSpec {
                num_key_heads: la_config.num_key_heads,
                num_value_heads: la_config.num_value_heads,
                key_head_dim: la_config.key_head_dim,
                value_head_dim: la_config.value_head_dim,
                conv_kernel_dim: la_config.conv_kernel_dim,
                rms_eps: la_config.rms_eps,
            };
            let key_dim = la_config.key_dim()?;
            let value_dim = la_config.value_dim()?;
            let conv_dim = key_dim
                .checked_mul(2)
                .and_then(|twice| twice.checked_add(value_dim))
                .ok_or_else(|| {
                    InferError::Shape("conv_dim déborde (MTP verify résident)".to_string())
                })?;
            Some((la_spec, key_dim, value_dim, conv_dim))
        } else {
            None
        };

        let CausalDecoderCache {
            layers,
            position: cache_position,
            resident,
        } = cache;
        let Some(arena) = resident.as_mut() else {
            return Ok(None);
        };
        if arena.state.gpu_timer().is_some() {
            return Ok(None);
        }
        let trunk_batch_supported = self.layers.iter().enumerate().all(|(index, layer)| {
            match (
                self.config.is_full_attention_layer(index),
                layer.mlp.as_ref(),
            ) {
                (true, Some(FeedForward::Dense(_))) => {
                    matches!(
                        arena.layers.get(index),
                        Some(ResidentLayerBuffers::FullDense(_))
                    )
                }
                (false, Some(FeedForward::Dense(_))) => {
                    matches!(
                        arena.layers.get(index),
                        Some(ResidentLayerBuffers::LinearDense(_))
                    )
                }
                _ => false,
            }
        });
        if !trunk_batch_supported {
            return Ok(None);
        }
        let Some(mtp) = arena.mtp.as_mut() else {
            return Ok(None);
        };
        if mtp.draft_indices.len() < 2 {
            return Ok(None);
        }
        let (rows, state_hidden) = final_state.as_matrix()?;
        if rows != 1 || state_hidden != hidden {
            return Err(InferError::Dimension(format!(
                "MTP fused D2 final_state=[{rows},{state_hidden}], attendu [1,{hidden}]"
            )));
        }

        arena.state.upload(&mtp.hidden_a, final_state.as_row()?)?;
        arena.state.upload_u32(&mtp.index, &[primary_u32])?;
        let verify_rows = 3usize;
        let verify_elements = verify_rows
            .checked_mul(hidden)
            .ok_or_else(|| InferError::Dimension("verify MTP rows déborde".to_string()))?;
        let linear_captures = if linear_dims.is_some() {
            let capture_rows = verify_rows.saturating_sub(1);
            let linear_states = layers
                .iter()
                .map(|layer| layer.linear.metal_state())
                .collect::<Vec<_>>();
            Some(metal.lease_linear_attn_state_captures(
                arena.state.scratch(),
                &linear_states,
                capture_rows,
            )?)
        } else {
            None
        };
        let command_buffer = arena.state.queue().new_command_buffer();
        let _barrier_guard = crate::metal_backend::resident_concurrent_enabled()
            .then(crate::metal_backend::install_dispatch_barrier_scope);
        let _namespace_guard =
            crate::metal_backend::install_scratch_namespace(arena.state.scratch_namespace());
        let encoder = new_resident_compute_encoder(command_buffer);
        let encoder_guard = crate::metal_backend::EncoderEndGuard::new(encoder);
        let mut owned: Vec<metal::Buffer> = Vec::new();
        let mut scratch: Vec<ScratchLease> = Vec::new();

        let append_dims = FullAttnLayerDims {
            hidden,
            q_heads,
            kv_heads,
            head_dim,
            rope_dims,
            position: 0,
            eps,
            theta,
            attn_output_gate: self.config.attn_output_gate,
        };
        {
            let mtp = arena
                .mtp
                .as_mut()
                .ok_or_else(|| InferError::Metal("arène MTP résidente absente".to_string()))?;
            self.encode_mtp_pending_append_rows(
                metal,
                &arena.state,
                &arena.embed_tokens,
                &arena.dense_tail_score,
                mtp,
                encoder_guard.encoder(),
                &mut owned,
                append_dims,
                history_len,
                &mut scratch,
            )?;
        }

        {
            let mtp = arena
                .mtp
                .as_mut()
                .ok_or_else(|| InferError::Metal("arène MTP résidente absente".to_string()))?;
            mtp.current_is_a = true;
        }

        {
            let mtp = arena
                .mtp
                .as_mut()
                .ok_or_else(|| InferError::Metal("arène MTP résidente absente".to_string()))?;
            let mut current_is_a = true;
            for draft_pos in 0..2usize {
                let (input_hidden, output_hidden) = if current_is_a {
                    (mtp.hidden_a.buffer(), mtp.hidden_b.buffer())
                } else {
                    (mtp.hidden_b.buffer(), mtp.hidden_a.buffer())
                };
                let (index_buffer, index_offset) = if draft_pos == 0 {
                    (mtp.index.buffer(), 0)
                } else {
                    (mtp.draft_indices.buffer(), 0)
                };
                let output_index_offset = draft_pos
                    .checked_mul(std::mem::size_of::<u32>())
                    .and_then(|value| u64::try_from(value).ok())
                    .ok_or_else(|| {
                        InferError::Metal("MTP draft output index offset déborde".to_string())
                    })?;
                let embedding_norm = arena.state.scratch().lease(hidden, GpuElement::F32)?;
                let hidden_norm = arena.state.scratch().lease(hidden, GpuElement::F32)?;
                let layer_out = arena.state.scratch().lease(hidden, GpuElement::F32)?;
                metal.encode_embedding_from_index_buffers_with_offset(
                    encoder_guard.encoder(),
                    &arena.embed_tokens,
                    index_buffer,
                    index_offset,
                    mtp.embedding.buffer(),
                    hidden,
                )?;
                metal.encode_rms_norm_rows(
                    encoder_guard.encoder(),
                    mtp.embedding.buffer(),
                    &mtp.pre_fc_norm_embedding,
                    embedding_norm.tensor().buffer(),
                    1,
                    hidden,
                    eps,
                )?;
                metal.encode_rms_norm_rows(
                    encoder_guard.encoder(),
                    input_hidden,
                    &mtp.pre_fc_norm_hidden,
                    hidden_norm.tensor().buffer(),
                    1,
                    hidden,
                    eps,
                )?;
                metal.encode_copy_with_offsets(
                    encoder_guard.encoder(),
                    embedding_norm.tensor().buffer(),
                    0,
                    mtp.concat.buffer(),
                    0,
                    hidden,
                )?;
                let hidden_offset = hidden
                    .checked_mul(std::mem::size_of::<f32>())
                    .and_then(|value| u64::try_from(value).ok())
                    .ok_or_else(|| {
                        InferError::Metal("MTP concat hidden offset déborde".to_string())
                    })?;
                metal.encode_copy_with_offsets(
                    encoder_guard.encoder(),
                    hidden_norm.tensor().buffer(),
                    0,
                    mtp.concat.buffer(),
                    hidden_offset,
                    hidden,
                )?;
                let fc_dim = metal.encode_matmul_weight_buffers(
                    encoder_guard.encoder(),
                    mtp.concat.buffer(),
                    1,
                    hidden.checked_mul(2).ok_or_else(|| {
                        InferError::Dimension("MTP fc input hidden déborde".to_string())
                    })?,
                    &mtp.fc,
                    mtp.fc_out.buffer(),
                    false,
                )?;
                if fc_dim != hidden {
                    return Err(InferError::Dimension(format!(
                        "MTP fc sort {fc_dim}, attendu {hidden}"
                    )));
                }
                let weights = FullAttnDenseLayerWeights {
                    input_norm: &mtp.layer.input_norm,
                    qkv_proj: mtp.layer.qkv_proj.as_ref(),
                    q_proj: &mtp.layer.q_proj,
                    k_proj: &mtp.layer.k_proj,
                    v_proj: &mtp.layer.v_proj,
                    o_proj: &mtp.layer.o_proj,
                    q_norm: &mtp.layer.q_norm,
                    k_norm: &mtp.layer.k_norm,
                    post_norm: &mtp.layer.post_norm,
                    gate_proj: &mtp.layer.gate_proj,
                    up_proj: &mtp.layer.up_proj,
                    down_proj: &mtp.layer.down_proj,
                    tail_score: &arena.dense_tail_score,
                };
                let dims = FullAttnLayerDims {
                    hidden,
                    q_heads,
                    kv_heads,
                    head_dim,
                    rope_dims,
                    position: history_len.checked_add(draft_pos).ok_or_else(|| {
                        InferError::Dimension("MTP position résidente déborde".to_string())
                    })?,
                    eps,
                    theta,
                    attn_output_gate: self.config.attn_output_gate,
                };
                arena.state.encode_full_attn_dense_layer(
                    metal,
                    encoder_guard.encoder(),
                    &mut owned,
                    &mut mtp.kv,
                    weights,
                    dims,
                    mtp.fc_out.buffer(),
                    layer_out.tensor().buffer(),
                )?;
                metal.encode_rms_norm_rows(
                    encoder_guard.encoder(),
                    layer_out.tensor().buffer(),
                    &mtp.norm,
                    output_hidden,
                    1,
                    hidden,
                    eps,
                )?;
                metal.encode_lm_head_argmax_buffers_with_index_offset(
                    encoder_guard.encoder(),
                    &mut owned,
                    output_hidden,
                    &mtp.draft_lm_head,
                    mtp.draft_indices.buffer(),
                    0,
                    output_index_offset,
                    hidden,
                )?;
                current_is_a = !current_is_a;
                scratch.push(embedding_norm);
                scratch.push(hidden_norm);
                scratch.push(layer_out);
            }
            mtp.current_is_a = current_is_a;
        }

        let verify_a = arena
            .state
            .scratch()
            .lease(verify_elements, GpuElement::F32)?;
        let verify_b = arena
            .state
            .scratch()
            .lease(verify_elements, GpuElement::F32)?;
        let embed = self.embed_scaled(&[primary])?;
        let embed_hidden = embed.data().len();
        if embed_hidden != hidden {
            return Err(InferError::Dimension(format!(
                "embedding hidden={embed_hidden}, attendu {hidden}"
            )));
        }
        arena.state.upload(verify_a.tensor(), embed.data())?;
        for row in 1..verify_rows {
            let draft_embed = arena.state.scratch().lease(hidden, GpuElement::F32)?;
            let source_index_offset = row
                .checked_sub(1)
                .and_then(|value| value.checked_mul(std::mem::size_of::<u32>()))
                .and_then(|value| u64::try_from(value).ok())
                .ok_or_else(|| {
                    InferError::Metal("verify draft index offset déborde".to_string())
                })?;
            metal.encode_embedding_from_index_buffers_with_offset(
                encoder_guard.encoder(),
                &arena.embed_tokens,
                arena
                    .mtp
                    .as_ref()
                    .ok_or_else(|| InferError::Metal("arène MTP résidente absente".to_string()))?
                    .draft_indices
                    .buffer(),
                source_index_offset,
                draft_embed.tensor().buffer(),
                hidden,
            )?;
            let target_row_offset = row
                .checked_mul(hidden)
                .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
                .and_then(|value| u64::try_from(value).ok())
                .ok_or_else(|| InferError::Metal("verify draft row offset déborde".to_string()))?;
            metal.encode_copy_with_offsets(
                encoder_guard.encoder(),
                draft_embed.tensor().buffer(),
                0,
                verify_a.tensor().buffer(),
                target_row_offset,
                hidden,
            )?;
            scratch.push(draft_embed);
        }

        let mut current = verify_a.tensor().buffer();
        let mut other = verify_b.tensor().buffer();
        let target_position = *cache_position;
        for (index, layer) in self.layers.iter().enumerate() {
            let layer_cache = &mut layers[index];
            if self.config.is_full_attention_layer(index) {
                let dims = FullAttnLayerDims {
                    hidden,
                    q_heads,
                    kv_heads,
                    head_dim,
                    rope_dims,
                    position: target_position,
                    eps,
                    theta,
                    attn_output_gate: self.config.attn_output_gate,
                };
                self.encode_resident_full_dense_layer_rows(
                    metal,
                    arena,
                    layer_cache,
                    layer,
                    index,
                    encoder_guard.encoder(),
                    &mut owned,
                    dims,
                    verify_rows,
                    current,
                    other,
                )?;
            } else {
                let Some((la_spec, key_dim, value_dim, conv_dim)) = linear_dims else {
                    return Err(InferError::Config(
                        "dims linear-attn résidentes absentes".to_string(),
                    ));
                };
                let res_dims = LinearAttnResidentDims {
                    in_dim: hidden,
                    conv_dim,
                    value_dim,
                    key_dim,
                };
                let captures = linear_captures
                    .as_ref()
                    .and_then(|(captures, _)| captures.get(index))
                    .and_then(|captures| captures.as_ref())
                    .map(Vec::as_slice);
                self.encode_resident_linear_dense_layer_rows(
                    metal,
                    arena,
                    layer_cache,
                    layer,
                    index,
                    encoder_guard.encoder(),
                    &mut owned,
                    la_spec,
                    res_dims,
                    verify_rows,
                    hidden,
                    eps,
                    current,
                    other,
                    captures,
                )?;
            }
            std::mem::swap(&mut current, &mut other);
        }

        let verify_final = arena
            .state
            .scratch()
            .lease(verify_elements, GpuElement::F32)?;
        let target_indices = arena.state.scratch().lease(verify_rows, GpuElement::U32)?;
        metal.encode_rms_norm_rows(
            encoder_guard.encoder(),
            current,
            &arena.final_norm,
            verify_final.tensor().buffer(),
            verify_rows,
            hidden,
            eps,
        )?;
        metal.encode_lm_head_argmax_rows_buffers(
            encoder_guard.encoder(),
            verify_final.tensor().buffer(),
            &arena.lm_head,
            target_indices.tensor().buffer(),
            0,
            0,
            verify_rows,
            hidden,
        )?;

        let committed_base_len = history_len.checked_add(1).ok_or_else(|| {
            InferError::Dimension("taille historique MTP append déborde".to_string())
        })?;
        {
            let mtp = arena
                .mtp
                .as_mut()
                .ok_or_else(|| InferError::Metal("arène MTP résidente absente".to_string()))?;
            for row in 0..verify_rows {
                let row_offset = row
                    .checked_mul(hidden)
                    .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
                    .and_then(|value| u64::try_from(value).ok())
                    .ok_or_else(|| {
                        InferError::Metal("MTP verify hidden row offset déborde".to_string())
                    })?;
                metal.encode_copy_with_offsets(
                    encoder_guard.encoder(),
                    verify_final.tensor().buffer(),
                    row_offset,
                    mtp.verify_hidden_rows.buffer(),
                    row_offset,
                    hidden,
                )?;
            }
            mtp.kv.truncate(committed_base_len)?;
            mtp.current_is_a = true;
        }

        scratch.push(verify_a);
        scratch.push(verify_b);

        encoder_guard.end();
        crate::metal_backend::commit_and_wait(command_buffer)?;

        let (draft_values, target_values) = {
            let arena = resident.as_ref().ok_or_else(|| {
                InferError::Metal("arène résidente absente après MTP fused D2".to_string())
            })?;
            let mtp = arena
                .mtp
                .as_ref()
                .ok_or_else(|| InferError::Metal("arène MTP résidente absente".to_string()))?;
            (
                crate::metal_backend::read_u32_buffer(mtp.draft_indices.buffer(), 2)?,
                crate::metal_backend::read_u32_buffer(
                    target_indices.tensor().buffer(),
                    verify_rows,
                )?,
            )
        };
        let state_values =
            crate::metal_backend::read_f32_buffer(verify_final.tensor().buffer(), verify_elements)?;
        drop(scratch);
        drop(target_indices);

        #[cfg(feature = "devtools")]
        if crate::decoder::flags::env_flag("RETI_RUST_MTP_APPEND_KV_ORACLE", false) {
            if let Some(arena) = resident.as_mut() {
                if let Some(mtp) = arena.mtp.as_mut() {
                    self.check_mtp_append_kv_oracle(mtp, history_len)?;
                }
            }
        }

        let mut draft_iter = draft_values.into_iter();
        let draft0 = draft_iter
            .next()
            .ok_or_else(|| InferError::Metal("MTP fused D2 sans draft0".to_string()))?;
        let draft1 = draft_iter
            .next()
            .ok_or_else(|| InferError::Metal("MTP fused D2 sans draft1".to_string()))?;
        let mut target_iter = target_values.into_iter();
        let target0 = target_iter
            .next()
            .ok_or_else(|| InferError::Metal("MTP fused D2 sans target0".to_string()))?;
        let target1 = target_iter
            .next()
            .ok_or_else(|| InferError::Metal("MTP fused D2 sans target1".to_string()))?;
        let target2 = target_iter
            .next()
            .ok_or_else(|| InferError::Metal("MTP fused D2 sans target2".to_string()))?;
        let drafts = [
            usize::try_from(draft0)
                .map_err(|_| InferError::Metal(format!("draft MTP trop grand: {draft0}")))?,
            usize::try_from(draft1)
                .map_err(|_| InferError::Metal(format!("draft MTP trop grand: {draft1}")))?,
        ];
        let targets = [
            usize::try_from(target0)
                .map_err(|_| InferError::Metal(format!("target MTP trop grand: {target0}")))?,
            usize::try_from(target1)
                .map_err(|_| InferError::Metal(format!("target MTP trop grand: {target1}")))?,
            usize::try_from(target2)
                .map_err(|_| InferError::Metal(format!("target MTP trop grand: {target2}")))?,
        ];

        let accepted0 = drafts[0] == targets[0];
        let mut accepted_for_stats = [accepted0, false];
        let mut checked = 1usize;
        let mut accepted_generated = 0usize;
        let mut committed_rows = 1usize;
        let mut final_row = 0usize;
        let mut pending = targets[0];
        let mut bonus_verified = false;
        if accepted0 && !stop_token_ids.contains(&targets[0]) {
            accepted_generated = 1;
            committed_rows = 2;
            final_row = 1;
            pending = targets[1];
            checked = 2;
            let accepted1 = drafts[1] == targets[1];
            accepted_for_stats[1] = accepted1;
            if accepted1 && !stop_token_ids.contains(&targets[1]) {
                accepted_generated = 2;
                committed_rows = 3;
                final_row = 2;
                pending = targets[2];
                bonus_verified = true;
            }
        }

        let committed_position = target_position
            .checked_add(committed_rows)
            .ok_or_else(|| InferError::Dimension("position verify MTP D2 déborde".to_string()))?;
        *cache_position = committed_position;
        if committed_rows < verify_rows {
            for layer in layers.iter_mut() {
                if let Some(full) = layer.full.as_mut() {
                    full.truncate(committed_position)?;
                }
            }
            if let Some(linear_captures) = linear_captures.as_ref() {
                let capture_row = committed_rows.checked_sub(1).ok_or_else(|| {
                    InferError::Dimension("rollback MTP D2 sans position".to_string())
                })?;
                let pairs = layers
                    .iter()
                    .zip(linear_captures.0.iter())
                    .filter_map(|(layer, captures)| {
                        let current = layer.linear.metal_state()?;
                        let capture = captures.as_ref()?.get(capture_row)?;
                        Some((current, capture))
                    })
                    .collect::<Vec<_>>();
                metal.restore_linear_attn_states(&pairs)?;
            }
        }
        if let Some(arena) = resident.as_mut() {
            if let Some(mtp) = arena.mtp.as_mut() {
                mtp.kv.truncate(committed_base_len)?;
                let pending_append_count = committed_rows.saturating_sub(1);
                if pending_append_count > 0 {
                    let pending_tokens = [target0, target1];
                    arena.state.upload_u32(
                        &mtp.pending_append_indices,
                        &pending_tokens[..pending_append_count],
                    )?;
                    mtp.pending_append_start = committed_base_len;
                    mtp.pending_append_count = pending_append_count;
                } else {
                    mtp.pending_append_count = 0;
                }
                mtp.current_is_a = true;
            }
        }

        let final_offset = final_row
            .checked_mul(hidden)
            .ok_or_else(|| InferError::Dimension("offset final_state MTP déborde".to_string()))?;
        let final_end = final_offset
            .checked_add(hidden)
            .ok_or_else(|| InferError::Dimension("fin final_state MTP déborde".to_string()))?;
        let final_slice = state_values.get(final_offset..final_end).ok_or_else(|| {
            InferError::Dimension("final_state MTP fused D2 incomplet".to_string())
        })?;
        Ok(Some(ResidentMtpSpecTwoOutput {
            drafts,
            targets,
            accepted_for_stats,
            checked,
            accepted_generated,
            committed_rows,
            pending,
            final_state: Tensor::row(final_slice.to_vec())?,
            bonus_verified,
        }))
    }
}
