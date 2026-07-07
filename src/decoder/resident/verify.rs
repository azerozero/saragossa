use super::super::*;
use super::types::*;

impl CausalDecoder {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(in crate::decoder) fn next_final_states_resident_verify(
        &self,
        cache: &mut CausalDecoderCache,
        token_ids: &[usize],
        emit_argmax: bool,
        capture_linear: bool,
        capture_layer_ids: Option<&[usize]>,
    ) -> Result<Option<ResidentVerifyOutput>> {
        if token_ids.is_empty() {
            return Ok(None);
        }
        let Some(metal) = self.forward_runtime().metal_executor() else {
            return Ok(None);
        };
        let head_dim = self.config.head_dim.ok_or_else(|| {
            InferError::Dimension("head_dim manquant (verify résident)".to_string())
        })?;
        let theta = self.config.rope_theta.ok_or_else(|| {
            InferError::Config("rope_theta manquant (verify résident)".to_string())
        })?;
        let eps = self.config.rms_eps;
        let rope_dims = self.config.rope_dims.unwrap_or(head_dim);
        let q_heads = self.config.num_attention_heads;
        let kv_heads = self.config.num_key_value_heads;
        let hidden = self.final_norm.data().len();
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
                    InferError::Shape("conv_dim déborde (verify résident)".to_string())
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

        let layer_batch_supported = token_ids.len() > 1
            && self.layers.iter().enumerate().all(|(index, layer)| {
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
                    (false, Some(FeedForward::Moe(_))) if prefill_moe_rows_enabled() => {
                        matches!(
                            arena.layers.get(index),
                            Some(ResidentLayerBuffers::LinearMoe(_))
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
        if layer_batch_supported {
            let base_position = *cache_position;
            let embed = self.embed_scaled(token_ids)?;
            let (batch, embed_hidden) = embed.as_matrix()?;
            if batch != token_ids.len() || embed_hidden != hidden {
                return Err(InferError::Dimension(format!(
                    "embedding batch={batch} hidden={embed_hidden}, attendu batch={} hidden={hidden}",
                    token_ids.len()
                )));
            }
            let batch_elements = batch
                .checked_mul(hidden)
                .ok_or_else(|| InferError::Dimension("verify batch hidden déborde".to_string()))?;
            let batch_a = arena
                .state
                .scratch()
                .lease(batch_elements, GpuElement::F32)?;
            let batch_b = arena
                .state
                .scratch()
                .lease(batch_elements, GpuElement::F32)?;
            arena.state.upload(batch_a.tensor(), embed.data())?;

            let row_in = arena.state.scratch().lease(hidden, GpuElement::F32)?;
            let row_out = arena.state.scratch().lease(hidden, GpuElement::F32)?;
            let final_states = arena
                .state
                .scratch()
                .lease(batch_elements, GpuElement::F32)?;
            let target_capture_cols = capture_layer_ids
                .map(|layer_ids| layer_ids.len().saturating_mul(hidden))
                .unwrap_or(0);
            let target_captures = if target_capture_cols > 0 {
                let capture_elements = batch.checked_mul(target_capture_cols).ok_or_else(|| {
                    InferError::Dimension("capture hidden résident déborde".to_string())
                })?;
                Some(
                    arena
                        .state
                        .scratch()
                        .lease(capture_elements, GpuElement::F32)?,
                )
            } else {
                None
            };
            let linear_captures = if capture_linear {
                let capture_rows = batch.saturating_sub(1);
                if capture_rows == 0 {
                    None
                } else {
                    let linear_states = layers
                        .iter()
                        .map(|layer| layer.linear.metal_state())
                        .collect::<Vec<_>>();
                    Some(metal.lease_linear_attn_state_captures(
                        arena.state.scratch(),
                        &linear_states,
                        capture_rows,
                    )?)
                }
            } else {
                None
            };
            let command_buffer = arena.state.queue().new_command_buffer();
            let _barrier_guard = crate::metal_backend::resident_concurrent_enabled()
                .then(crate::metal_backend::install_dispatch_barrier_scope);
            let _namespace_guard =
                crate::metal_backend::install_scratch_namespace(arena.state.scratch_namespace());
            let encoder = new_resident_compute_encoder(command_buffer);
            let mut owned: Vec<metal::Buffer> = Vec::new();
            let mut current = batch_a.tensor().buffer();
            let mut other = batch_b.tensor().buffer();

            for (index, layer) in self.layers.iter().enumerate() {
                let layer_cache = &mut layers[index];
                if self.config.is_full_attention_layer(index) {
                    for row in 0..batch {
                        let row_offset = row
                            .checked_mul(hidden)
                            .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
                            .and_then(|value| u64::try_from(value).ok())
                            .ok_or_else(|| {
                                InferError::Metal("verify row offset déborde".to_string())
                            })?;
                        metal.encode_copy_with_offsets(
                            encoder,
                            current,
                            row_offset,
                            row_in.tensor().buffer(),
                            0,
                            hidden,
                        )?;
                        let dims = FullAttnLayerDims {
                            hidden,
                            q_heads,
                            kv_heads,
                            head_dim,
                            rope_dims,
                            position: *cache_position + row,
                            eps,
                            theta,
                            attn_output_gate: self.config.attn_output_gate,
                        };
                        self.encode_resident_full_layer(
                            metal,
                            arena,
                            layer_cache,
                            layer,
                            index,
                            encoder,
                            &mut owned,
                            dims,
                            row_in.tensor().buffer(),
                            row_out.tensor().buffer(),
                        )?;
                        metal.encode_copy_with_offsets(
                            encoder,
                            row_out.tensor().buffer(),
                            0,
                            other,
                            row_offset,
                            hidden,
                        )?;
                    }
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
                        .and_then(|captures| captures.as_deref());
                    match layer.mlp.as_ref() {
                        Some(FeedForward::Moe(_)) => self.encode_resident_linear_layer_rows(
                            metal,
                            arena,
                            layer_cache,
                            layer,
                            index,
                            encoder,
                            &mut owned,
                            la_spec,
                            res_dims,
                            batch,
                            hidden,
                            eps,
                            current,
                            other,
                            captures,
                        )?,
                        Some(FeedForward::Dense(_)) => self
                            .encode_resident_linear_dense_layer_rows(
                                metal,
                                arena,
                                layer_cache,
                                layer,
                                index,
                                encoder,
                                &mut owned,
                                la_spec,
                                res_dims,
                                batch,
                                hidden,
                                eps,
                                current,
                                other,
                                captures,
                            )?,
                        _ => {
                            return Err(InferError::Config(
                                "MLP attendu (verify résident batché)".to_string(),
                            ));
                        }
                    }
                }
                if let (Some(layer_ids), Some(target_captures)) =
                    (capture_layer_ids, target_captures.as_ref())
                {
                    if let Some(slot) = resident_capture_slot(layer_ids, index) {
                        for row in 0..batch {
                            let source_offset = row
                                .checked_mul(hidden)
                                .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
                                .and_then(|value| u64::try_from(value).ok())
                                .ok_or_else(|| {
                                    InferError::Metal(
                                        "capture hidden source offset déborde".to_string(),
                                    )
                                })?;
                            let target_offset = row
                                .checked_mul(target_capture_cols)
                                .and_then(|value| value.checked_add(slot * hidden))
                                .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
                                .and_then(|value| u64::try_from(value).ok())
                                .ok_or_else(|| {
                                    InferError::Metal(
                                        "capture hidden target offset déborde".to_string(),
                                    )
                                })?;
                            metal.encode_copy_with_offsets(
                                encoder,
                                other,
                                source_offset,
                                target_captures.tensor().buffer(),
                                target_offset,
                                hidden,
                            )?;
                        }
                    }
                }
                std::mem::swap(&mut current, &mut other);
            }
            metal.encode_rms_norm_rows(
                encoder,
                current,
                &arena.final_norm,
                final_states.tensor().buffer(),
                batch,
                hidden,
                eps,
            )?;
            let final_indices = if emit_argmax {
                let indices = arena.state.scratch().lease(batch, GpuElement::U32)?;
                for row in 0..batch {
                    let input_offset = row
                        .checked_mul(hidden)
                        .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
                        .and_then(|value| u64::try_from(value).ok())
                        .ok_or_else(|| {
                            InferError::Metal("verify argmax input offset déborde".to_string())
                        })?;
                    let index_offset = row
                        .checked_mul(std::mem::size_of::<u32>())
                        .and_then(|value| u64::try_from(value).ok())
                        .ok_or_else(|| {
                            InferError::Metal("verify argmax index offset déborde".to_string())
                        })?;
                    metal.encode_lm_head_argmax_buffers_with_index_offset(
                        encoder,
                        &mut owned,
                        final_states.tensor().buffer(),
                        &arena.lm_head,
                        indices.tensor().buffer(),
                        input_offset,
                        index_offset,
                        hidden,
                    )?;
                }
                Some(indices)
            } else {
                None
            };
            encoder.end_encoding();
            crate::metal_backend::commit_and_wait(command_buffer)?;
            *cache_position += batch;
            let output = crate::metal_backend::read_f32_buffer(
                final_states.tensor().buffer(),
                batch_elements,
            )?;
            let tokens = final_indices
                .as_ref()
                .map(|indices| {
                    crate::metal_backend::read_u32_buffer(indices.tensor().buffer(), batch)?
                        .into_iter()
                        .map(|index| {
                            usize::try_from(index).map_err(|_| {
                                InferError::Metal(format!(
                                    "verify argmax index trop grand: {index}"
                                ))
                            })
                        })
                        .collect::<Result<Vec<_>>>()
                })
                .transpose()?;
            let target_hidden = target_captures
                .as_ref()
                .map(|captures| {
                    let values = crate::metal_backend::read_f32_buffer(
                        captures.tensor().buffer(),
                        batch * target_capture_cols,
                    )?;
                    Tensor::from_vec(vec![batch, target_capture_cols], values)
                })
                .transpose()?;
            return Ok(Some(ResidentVerifyOutput {
                states: Tensor::from_vec(vec![batch, hidden], output)?,
                tokens,
                captures: linear_captures.map(|linear| ResidentVerifyCaptures {
                    base_position,
                    linear: linear.0,
                    _linear_leases: linear.1,
                }),
                target_hidden,
            }));
        }

        let mut token_inputs = Vec::with_capacity(token_ids.len());
        for token_id in token_ids {
            let embed = self.embed_scaled(&[*token_id])?;
            let (_, embed_hidden) = embed.as_matrix()?;
            if embed_hidden != hidden {
                return Err(InferError::Dimension(format!(
                    "embedding hidden={embed_hidden}, attendu {hidden}"
                )));
            }
            let input = arena.state.persistent(hidden, GpuElement::F32)?;
            arena.state.upload(&input, embed.data())?;
            token_inputs.push(input);
        }

        let final_states = arena
            .state
            .scratch()
            .lease(token_ids.len() * hidden, GpuElement::F32)?;
        let target_capture_cols = capture_layer_ids
            .map(|layer_ids| layer_ids.len().saturating_mul(hidden))
            .unwrap_or(0);
        let target_captures = if target_capture_cols > 0 {
            let capture_elements = token_ids
                .len()
                .checked_mul(target_capture_cols)
                .ok_or_else(|| {
                    InferError::Dimension("capture hidden résident déborde".to_string())
                })?;
            Some(
                arena
                    .state
                    .scratch()
                    .lease(capture_elements, GpuElement::F32)?,
            )
        } else {
            None
        };
        let final_normed = arena.state.scratch().lease(hidden, GpuElement::F32)?;
        let command_buffer = arena.state.queue().new_command_buffer();
        let _barrier_guard = crate::metal_backend::resident_concurrent_enabled()
            .then(crate::metal_backend::install_dispatch_barrier_scope);
        let _namespace_guard =
            crate::metal_backend::install_scratch_namespace(arena.state.scratch_namespace());
        let encoder = new_resident_compute_encoder(command_buffer);
        let mut owned: Vec<metal::Buffer> = Vec::new();

        for (token_pos, input) in token_inputs.iter().enumerate() {
            let mut current = input.buffer();
            let mut other = arena.hidden_b.buffer();
            let position = *cache_position + token_pos;
            for (index, layer) in self.layers.iter().enumerate() {
                let layer_cache = &mut layers[index];
                if self.config.is_full_attention_layer(index) {
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
                    self.encode_resident_full_layer(
                        metal,
                        arena,
                        layer_cache,
                        layer,
                        index,
                        encoder,
                        &mut owned,
                        dims,
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
                    self.encode_resident_linear_layer(
                        metal,
                        arena,
                        layer_cache,
                        layer,
                        index,
                        encoder,
                        &mut owned,
                        la_spec,
                        res_dims,
                        hidden,
                        eps,
                        current,
                        other,
                    )?;
                }
                if let (Some(layer_ids), Some(target_captures)) =
                    (capture_layer_ids, target_captures.as_ref())
                {
                    if let Some(slot) = resident_capture_slot(layer_ids, index) {
                        let target_offset = token_pos
                            .checked_mul(target_capture_cols)
                            .and_then(|value| value.checked_add(slot * hidden))
                            .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
                            .and_then(|value| u64::try_from(value).ok())
                            .ok_or_else(|| {
                                InferError::Metal(
                                    "capture hidden target offset déborde".to_string(),
                                )
                            })?;
                        metal.encode_copy_with_offsets(
                            encoder,
                            other,
                            0,
                            target_captures.tensor().buffer(),
                            target_offset,
                            hidden,
                        )?;
                    }
                }
                std::mem::swap(&mut current, &mut other);
            }
            metal.encode_rms_norm_rows(
                encoder,
                current,
                &arena.final_norm,
                final_normed.tensor().buffer(),
                1,
                hidden,
                eps,
            )?;
            let final_offset = token_pos
                .checked_mul(hidden)
                .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
                .and_then(|value| u64::try_from(value).ok())
                .ok_or_else(|| InferError::Metal("verify final offset déborde".to_string()))?;
            metal.encode_copy_with_offsets(
                encoder,
                final_normed.tensor().buffer(),
                0,
                final_states.tensor().buffer(),
                final_offset,
                hidden,
            )?;
        }
        let final_indices = if emit_argmax {
            let indices = arena
                .state
                .scratch()
                .lease(token_ids.len(), GpuElement::U32)?;
            for row in 0..token_ids.len() {
                let input_offset = row
                    .checked_mul(hidden)
                    .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
                    .and_then(|value| u64::try_from(value).ok())
                    .ok_or_else(|| {
                        InferError::Metal("verify argmax input offset déborde".to_string())
                    })?;
                let index_offset = row
                    .checked_mul(std::mem::size_of::<u32>())
                    .and_then(|value| u64::try_from(value).ok())
                    .ok_or_else(|| {
                        InferError::Metal("verify argmax index offset déborde".to_string())
                    })?;
                metal.encode_lm_head_argmax_buffers_with_index_offset(
                    encoder,
                    &mut owned,
                    final_states.tensor().buffer(),
                    &arena.lm_head,
                    indices.tensor().buffer(),
                    input_offset,
                    index_offset,
                    hidden,
                )?;
            }
            Some(indices)
        } else {
            None
        };
        encoder.end_encoding();
        crate::metal_backend::commit_and_wait(command_buffer)?;
        *cache_position += token_ids.len();
        let output = crate::metal_backend::read_f32_buffer(
            final_states.tensor().buffer(),
            token_ids.len() * hidden,
        )?;
        let tokens = final_indices
            .as_ref()
            .map(|indices| {
                crate::metal_backend::read_u32_buffer(indices.tensor().buffer(), token_ids.len())?
                    .into_iter()
                    .map(|index| {
                        usize::try_from(index).map_err(|_| {
                            InferError::Metal(format!("verify argmax index trop grand: {index}"))
                        })
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?;
        let target_hidden = target_captures
            .as_ref()
            .map(|captures| {
                let values = crate::metal_backend::read_f32_buffer(
                    captures.tensor().buffer(),
                    token_ids.len() * target_capture_cols,
                )?;
                Tensor::from_vec(vec![token_ids.len(), target_capture_cols], values)
            })
            .transpose()?;
        Ok(Some(ResidentVerifyOutput {
            states: Tensor::from_vec(vec![token_ids.len(), hidden], output)?,
            tokens,
            captures: None,
            target_hidden,
        }))
    }
}
