//! Prototype D1 fresh : deux cycles préencodés, aucune lecture CPU entre eux.
use super::super::*;
use crate::decode_resident::{check_gpu_position_rows, GpuPosition};
use crate::metal_backend::mtp_control::*;
use crate::metal_backend::{EncoderEndGuard, MetalExecutor};

pub(in crate::decoder) struct MtpGpuBlockOutput {
    pub tokens: Vec<usize>,
    pub pending: usize,
    pub final_state: Tensor,
    pub accepted: usize,
    pub rejected: usize,
    pub verifications: usize,
    pub eos: bool,
}

impl CausalDecoder {
    // Le chemin fresh de la tête est indépendant de la position du trunk.
    fn encode_mtp_block_head(
        &self,
        metal: &MetalExecutor,
        arena: &mut ResidentArena,
        encoder: &metal::ComputeCommandEncoderRef,
        owned: &mut Vec<metal::Buffer>,
    ) -> Result<()> {
        let hidden = self.final_norm.data().len();
        let head_dim = self
            .config
            .head_dim
            .ok_or_else(|| InferError::Config("MTP head_dim absent".into()))?;
        let theta = self
            .config
            .rope_theta
            .ok_or_else(|| InferError::Config("MTP rope_theta absent".into()))?;
        let mtp = arena
            .mtp
            .as_mut()
            .ok_or_else(|| InferError::Metal("arène MTP absente".into()))?;
        mtp.kv.truncate(0)?;
        let embedding_norm = arena.state.scratch().lease(hidden, GpuElement::F32)?;
        let hidden_norm = arena.state.scratch().lease(hidden, GpuElement::F32)?;
        let layer_out = arena.state.scratch().lease(hidden, GpuElement::F32)?;
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
            self.config.rms_eps,
        )?;
        metal.encode_rms_norm_rows(
            encoder,
            mtp.hidden_a.buffer(),
            &mtp.pre_fc_norm_hidden,
            hidden_norm.tensor().buffer(),
            1,
            hidden,
            self.config.rms_eps,
        )?;
        metal.encode_copy_with_offsets(
            encoder,
            embedding_norm.tensor().buffer(),
            0,
            mtp.concat.buffer(),
            0,
            hidden,
        )?;
        let hidden_bytes = hidden
            .checked_mul(4)
            .and_then(|n| u64::try_from(n).ok())
            .ok_or_else(|| InferError::Dimension("MTP hidden offset déborde".into()))?;
        metal.encode_copy_with_offsets(
            encoder,
            hidden_norm.tensor().buffer(),
            0,
            mtp.concat.buffer(),
            hidden_bytes,
            hidden,
        )?;
        let width = metal.encode_matmul_weight_buffers(
            encoder,
            mtp.concat.buffer(),
            1,
            hidden
                .checked_mul(2)
                .ok_or_else(|| InferError::Dimension("MTP concat déborde".into()))?,
            &mtp.fc,
            mtp.fc_out.buffer(),
            false,
        )?;
        if width != hidden {
            return Err(InferError::Dimension("MTP fc hidden incompatible".into()));
        }
        let weights = FullAttnDenseLayerWeights {
            input_norm: &mtp.layer.input_norm,
            qkv_proj: mtp.layer.qkv_proj.as_ref(),
            qkv_proj_without_gate: false,
            value_norm: mtp.layer.value_norm,
            q_proj: &mtp.layer.q_proj,
            k_proj: &mtp.layer.k_proj,
            v_proj: &mtp.layer.v_proj,
            o_proj: &mtp.layer.o_proj,
            q_norm: &mtp.layer.q_norm,
            k_norm: &mtp.layer.k_norm,
            post_norm: &mtp.layer.post_norm,
            pre_feedforward_norm: mtp.layer.pre_feedforward_norm.as_ref(),
            post_feedforward_norm: mtp.layer.post_feedforward_norm.as_ref(),
            layer_scalar: mtp.layer.layer_scalar,
            gate_proj: &mtp.layer.gate_proj,
            up_proj: &mtp.layer.up_proj,
            down_proj: &mtp.layer.down_proj,
            tail_score: &arena.dense_tail_score,
            parallel_moe: mtp.layer.parallel_moe_weights(),
        };
        let dims = FullAttnLayerDims {
            hidden,
            q_heads: self.config.num_attention_heads,
            kv_heads: self.config.num_key_value_heads,
            head_dim,
            attn_scalar: self.config.query_pre_attn_scalar.unwrap_or(head_dim as f32),
            rope_dims: self.config.rope_dims.unwrap_or(head_dim),
            rope_frequency_dim: self.config.rope_dims.unwrap_or(head_dim),
            position: 0,
            window_start: 0,
            eps: self.config.rms_eps,
            theta,
            attn_output_gate: self.config.attn_output_gate,
        };
        arena.state.encode_full_attn_dense_layer(
            metal,
            encoder,
            owned,
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
            mtp.hidden_b.buffer(),
            1,
            hidden,
            self.config.rms_eps,
        )?;
        metal.encode_lm_head_argmax_buffers_with_index_offset(
            encoder,
            owned,
            mtp.hidden_b.buffer(),
            &mtp.draft_lm_head,
            mtp.draft_indices.buffer(),
            0,
            0,
            hidden,
        )?;
        Ok(())
    }

    pub(in crate::decoder) fn next_mtp_gpu_block(
        &self,
        cache: &mut CausalDecoderCache,
        final_state: &Tensor,
        primary: usize,
        remaining: usize,
        stops: &[usize],
        kernels: &MtpD1Kernels,
    ) -> Result<Option<MtpGpuBlockOutput>> {
        if !mtp_fresh_cache_enabled()
            || remaining < 2
            || self.config.sliding_window.is_some()
            || !self.config.attn_output_gate
        {
            return Ok(None);
        }
        let Some(metal) = self.forward_runtime().metal_executor() else {
            return Ok(None);
        };
        let Some(arena) = cache.resident.as_ref() else {
            return Ok(None);
        };
        if arena.mtp.is_none() {
            return Ok(None);
        }
        let max_position = cache
            .position
            .checked_add(2)
            .ok_or_else(|| InferError::Dimension("MTP bloc position déborde".into()))?;
        // Valider TOUTES les couches avant de modifier les caches/encoder le CB.
        for (index, (layer, layer_cache)) in self.layers.iter().zip(&cache.layers).enumerate() {
            if !matches!(layer.mlp.as_ref(), Some(FeedForward::Dense(_))) {
                return Ok(None);
            }
            if self.config.is_resident_full_attention_layer(index) {
                let Some(kv) = layer_cache.full.as_ref() else {
                    return Ok(None);
                };
                if !matches!(
                    arena.layers.get(index),
                    Some(ResidentLayerBuffers::FullDense(_))
                ) || check_gpu_position_rows(
                    kv.capacity(),
                    kv.kv_dim(),
                    kv.format(),
                    0,
                    max_position,
                    2,
                )
                .is_err()
                {
                    return Ok(None);
                }
            } else if layer_cache.linear.metal_state().is_none()
                || !matches!(
                    arena.layers.get(index),
                    Some(ResidentLayerBuffers::LinearDense(_))
                )
            {
                return Ok(None);
            }
        }
        let hidden = self.final_norm.data().len();
        let (rows, width) = final_state.as_matrix()?;
        if rows != 1 || width != hidden {
            return Err(InferError::Dimension(
                "MTP bloc final_state incompatible".into(),
            ));
        }
        let elements = hidden
            .checked_mul(2)
            .ok_or_else(|| InferError::Dimension("MTP bloc hidden déborde".into()))?;
        let primary = u32::try_from(primary)
            .map_err(|_| InferError::Dimension("MTP primary hors u32".into()))?;
        let stop_ids = stops
            .iter()
            .map(|&s| {
                u32::try_from(s).map_err(|_| InferError::Dimension("MTP stop hors u32".into()))
            })
            .collect::<Result<Vec<_>>>()?;
        let init = mtp_d1_control_init(cache.position, primary, remaining, 4)?;
        let ctrl_lease = arena.state.scratch().lease(MTP_CTRL_LEN, GpuElement::U32)?;
        let output_lease = arena.state.scratch().lease(4, GpuElement::U32)?;
        let stops_lease = arena
            .state
            .scratch()
            .lease(stop_ids.len().max(1), GpuElement::U32)?;
        arena.state.upload_u32(ctrl_lease.tensor(), &init)?;
        arena.state.upload_u32(
            stops_lease.tensor(),
            if stop_ids.is_empty() { &[0] } else { &stop_ids },
        )?;
        let ctrl = ctrl_lease.tensor().buffer().clone();
        let output = output_lease.tensor().buffer().clone();
        let stop_buffer = stops_lease.tensor().buffer().clone();
        let theta = self
            .config
            .rope_theta
            .ok_or_else(|| InferError::Config("rope_theta absent".into()))?;
        let linear_dims = if self.has_resident_linear_attention_layer() {
            let cfg = self.config.linear_attention_config()?;
            let key_dim = cfg.key_dim()?;
            let value_dim = cfg.value_dim()?;
            let conv_dim = key_dim
                .checked_mul(2)
                .and_then(|n| n.checked_add(value_dim))
                .ok_or_else(|| InferError::Dimension("MTP conv_dim déborde".into()))?;
            Some((
                LinearAttentionStepSpec {
                    num_key_heads: cfg.num_key_heads,
                    num_value_heads: cfg.num_value_heads,
                    key_head_dim: cfg.key_head_dim,
                    value_head_dim: cfg.value_head_dim,
                    conv_kernel_dim: cfg.conv_kernel_dim,
                    rms_eps: cfg.rms_eps,
                },
                LinearAttnResidentDims {
                    in_dim: hidden,
                    conv_dim,
                    value_dim,
                    key_dim,
                },
            ))
        } else {
            None
        };
        let CausalDecoderCache {
            layers,
            position: cache_position,
            resident,
        } = cache;
        let arena = resident
            .as_mut()
            .ok_or_else(|| InferError::Metal("arène disparue".into()))?;
        let mtp = arena
            .mtp
            .as_ref()
            .ok_or_else(|| InferError::Metal("arène MTP disparue".into()))?;
        arena.state.upload(&mtp.hidden_a, final_state.as_row()?)?;
        arena.state.upload_u32(&mtp.index, &[primary])?;
        let states = layers
            .iter()
            .map(|layer| layer.linear.metal_state())
            .collect::<Vec<_>>();
        let pre = metal.lease_linear_attn_state_captures(arena.state.scratch(), &states, 1)?;
        let captured = metal.lease_linear_attn_state_captures(arena.state.scratch(), &states, 1)?;
        // Buffers de sortie conservés pour les deux cycles ; réutilisation GPU
        // ordonnée, sans remise au pool avant la fin du command buffer.
        let a = arena.state.scratch().lease(elements, GpuElement::F32)?;
        let b = arena.state.scratch().lease(elements, GpuElement::F32)?;
        let verify_final = arena.state.scratch().lease(elements, GpuElement::F32)?;
        let targets = arena.state.scratch().lease(2, GpuElement::U32)?;
        let queue = arena.state.queue().to_owned();
        let command_buffer = queue.new_command_buffer();
        let _barriers = crate::metal_backend::resident_concurrent_enabled()
            .then(crate::metal_backend::install_dispatch_barrier_scope);
        let _namespace =
            crate::metal_backend::install_scratch_namespace(arena.state.scratch_namespace());
        let encoder = new_resident_compute_encoder(command_buffer);
        let guard = EncoderEndGuard::new(encoder);
        let mut owned = Vec::new();
        for _ in 0..2 {
            let pairs = layers
                .iter()
                .zip(&pre.0)
                .filter_map(|(layer, captures)| {
                    Some((layer.linear.metal_state()?, captures.as_ref()?.first()?))
                })
                .collect::<Vec<_>>();
            metal.encode_capture_linear_attn_states(guard.encoder(), &pairs)?;
            self.encode_mtp_block_head(metal, arena, guard.encoder(), &mut owned)?;
            let mtp = arena
                .mtp
                .as_ref()
                .ok_or_else(|| InferError::Metal("MTP absent".into()))?;
            metal.encode_embedding_from_index_buffers_scaled(
                guard.encoder(),
                &arena.embed_tokens,
                mtp.index.buffer(),
                a.tensor().buffer(),
                hidden,
                self.config.embed_scale.unwrap_or(1.0),
                self.config.is_qwen && qwen_embed_bf16_enabled(),
            )?;
            metal.encode_embedding_from_index_buffers(
                guard.encoder(),
                &arena.embed_tokens,
                mtp.draft_indices.buffer(),
                mtp.embedding.buffer(),
                hidden,
            )?;
            let offset = u64::try_from(hidden)
                .ok()
                .and_then(|n| n.checked_mul(4))
                .ok_or_else(|| InferError::Dimension("MTP row offset déborde".into()))?;
            metal.encode_copy_with_offsets(
                guard.encoder(),
                mtp.embedding.buffer(),
                0,
                a.tensor().buffer(),
                offset,
                hidden,
            )?;
            let mut current = a.tensor().buffer();
            let mut other = b.tensor().buffer();
            for (index, layer) in self.layers.iter().enumerate() {
                let layer_cache = &mut layers[index];
                if self.config.is_resident_full_attention_layer(index) {
                    let dims = self.config.resident_windowed_full_attn_layer_dims(
                        index,
                        hidden,
                        *cache_position,
                        self.config.rms_eps,
                        theta,
                    )?;
                    self.encode_resident_full_dense_layer_rows_gpu_position(
                        metal,
                        arena,
                        layer_cache,
                        layer,
                        index,
                        guard.encoder(),
                        &mut owned,
                        dims,
                        GpuPosition {
                            ctrl: &ctrl,
                            slot: MTP_CTRL_POSITION,
                            max_position,
                        },
                        2,
                        current,
                        other,
                    )?;
                } else {
                    let (spec, dims) = linear_dims
                        .ok_or_else(|| InferError::Config("MTP linear dims absentes".into()))?;
                    let captures = captured
                        .0
                        .get(index)
                        .and_then(Option::as_ref)
                        .map(Vec::as_slice);
                    self.encode_resident_linear_dense_layer_rows(
                        metal,
                        arena,
                        layer_cache,
                        layer,
                        index,
                        guard.encoder(),
                        &mut owned,
                        spec,
                        dims,
                        2,
                        hidden,
                        self.config.rms_eps,
                        current,
                        other,
                        captures,
                    )?;
                }
                std::mem::swap(&mut current, &mut other);
            }
            metal.encode_rms_norm_rows(
                guard.encoder(),
                current,
                &arena.final_norm,
                verify_final.tensor().buffer(),
                2,
                hidden,
                self.config.rms_eps,
            )?;
            metal.encode_lm_head_argmax_two_rows_buffers(
                guard.encoder(),
                verify_final.tensor().buffer(),
                &arena.lm_head,
                targets.tensor().buffer(),
                0,
                0,
                hidden,
            )?;
            let mtp = arena
                .mtp
                .as_ref()
                .ok_or_else(|| InferError::Metal("MTP absent".into()))?;
            kernels.encode_mtp_d1_decision(
                guard.encoder(),
                &MtpD1Buffers {
                    ctrl: &ctrl,
                    draft_index: mtp.draft_indices.buffer(),
                    target_indices: targets.tensor().buffer(),
                    target_indices_offset: 0,
                    target_hidden: verify_final.tensor().buffer(),
                    target_hidden_offset: 0,
                    next_hidden: mtp.hidden_a.buffer(),
                    next_index: mtp.index.buffer(),
                    stop_ids: &stop_buffer,
                    output_tokens: &output,
                },
                MtpD1Dims {
                    hidden_dim: hidden,
                    stop_count: stop_ids.len(),
                    output_capacity: 4,
                },
            )?;
            for ((layer, capture), before) in layers.iter().zip(&captured.0).zip(&pre.0) {
                if let (Some(state), Some(row0), Some(before)) = (
                    layer.linear.metal_state(),
                    capture.as_ref().and_then(|v| v.first()),
                    before.as_ref().and_then(|v| v.first()),
                ) {
                    kernels.encode_linear_state_rollback(
                        guard.encoder(),
                        &ctrl,
                        state,
                        row0,
                        before,
                    )?;
                }
            }
        }
        guard.end();
        // Seul point de synchronisation du bloc. Toutes les lectures suivent.
        crate::metal_backend::commit_and_wait(command_buffer)?;
        let values = crate::metal_backend::read_u32_buffer(&ctrl, MTP_CTRL_LEN)?;
        let count = values[MTP_CTRL_OUT_COUNT] as usize;
        if count > remaining.min(4) {
            return Err(InferError::Metal("sortie MTP bloc hors budget".into()));
        }
        let tokens = crate::metal_backend::read_u32_buffer(&output, count)?
            .into_iter()
            .map(|v| v as usize)
            .collect();
        let mtp = arena
            .mtp
            .as_mut()
            .ok_or_else(|| InferError::Metal("MTP absent après bloc".into()))?;
        let final_state = Tensor::row(crate::metal_backend::read_f32_buffer(
            mtp.hidden_a.buffer(),
            hidden,
        )?)?;
        *cache_position = values[MTP_CTRL_POSITION] as usize;
        for layer in layers {
            if let Some(kv) = layer.full.as_mut() {
                kv.truncate(*cache_position)?;
            }
        }
        mtp.kv.truncate(0)?;
        mtp.current_is_a = true;
        mtp.pending_append_count = 0;
        Ok(Some(MtpGpuBlockOutput {
            tokens,
            pending: values[MTP_CTRL_PRIMARY] as usize,
            final_state,
            accepted: values[MTP_CTRL_ACCEPTED] as usize,
            rejected: values[MTP_CTRL_REJECTED] as usize,
            verifications: values[MTP_CTRL_VERIFICATIONS] as usize,
            eos: values[MTP_CTRL_STOPPED] == MTP_STOP_EOS,
        }))
    }
}
