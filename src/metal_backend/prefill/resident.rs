//! Orchestration du prefill résident Qwen MoE.

use super::*;

pub(super) struct ResidentPrefillEncodeContext<'a> {
    profile_sections: bool,
    section_profile: &'a mut PrefillSectionProfile,
    kernel_timing: &'a mut Option<PrefillKernelTiming>,
    command_buffer: &'a metal::CommandBufferRef,
    fallback_encoder: Option<&'a ComputeCommandEncoderRef>,
    owned_buffers: &'a mut Vec<Buffer>,
}

/// Morceau de lignes `[start, start + rows)` d'un prefill de `total` lignes.
#[derive(Clone, Copy)]
pub(super) struct ResidentPrefillChunk {
    pub(super) start: usize,
    pub(super) rows: usize,
    pub(super) total: usize,
}

fn f32_byte_offset(len: usize) -> Result<u64> {
    len.checked_mul(std::mem::size_of::<f32>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| InferError::Shape(format!("offset f32 {len} déborde")))
}

#[expect(
    clippy::too_many_arguments,
    reason = "orchestration prefill: couches, dimensions et caches restent explicites"
)]
impl MetalExecutor {
    #[cfg(test)]
    pub(crate) fn qwen_moe_prefill_resident(
        &self,
        input: &Tensor,
        layers: &[PrefillMoeLayer<'_>],
        spec: PrefillAttentionSpec,
    ) -> Result<(Tensor, Vec<PrefillResidentLayerCache>)> {
        let (output, caches, _) =
            self.qwen_moe_prefill_resident_with_captures(input, layers, spec, &[], 0, false)?;
        Ok((output, caches))
    }

    pub(crate) fn qwen_moe_prefill_resident_with_captures(
        &self,
        input: &Tensor,
        layers: &[PrefillMoeLayer<'_>],
        spec: PrefillAttentionSpec,
        capture_layer_ids: &[usize],
        chunk_tokens: usize,
        last_row_only: bool,
    ) -> Result<(Tensor, Vec<PrefillResidentLayerCache>, Option<Tensor>)> {
        self.qwen_moe_prefill_resident_from(
            input,
            layers,
            spec,
            capture_layer_ids,
            chunk_tokens,
            None,
            last_row_only,
        )
    }

    /// Indique si un prefill peut prolonger un préfixe : toutes les attentions
    /// complètes doivent passer par l'attention Steel/NAX à décalage.
    pub(crate) fn prefill_resident_extend_supported(&self, layers: &[PrefillMoeLayer<'_>]) -> bool {
        layers.iter().all(|layer| match layer.attention {
            PrefillAttentionLayer::Full { .. } => {
                self.causal_attention_prefill_chunk_supported(layer.attention_spec)
            }
            PrefillAttentionLayer::Linear { .. } => true,
        })
    }

    /// Prefill résident avec capture du flux résiduel en sortie des couches
    /// `capture_layer_ids` (toutes les positions). Les captures sont copiées sur
    /// le GPU pendant l'encodage, relues une fois à la fin et rendues en
    /// `[seq, n * hidden]`, la disposition attendue par le drafter DFlash.
    /// `None` quand aucune couche n'est demandée. `chunk_tokens` borne la
    /// taille d'un morceau (0 : un seul morceau), cf.
    /// `prefill_resident_chunk_tokens`. Avec `prefix`, `input` ne porte que les
    /// `spec.seq - prefix.rows` positions nouvelles ; sorties, captures et KV
    /// rendus ne couvrent que celles-ci, l'état linéaire du préfixe avance en
    /// place. `last_row_only` rend la seule dernière ligne `[1, hidden]` : la
    /// dernière couche, si elle est en attention complète et non capturée, ne
    /// calcule alors que son K/V sur les morceaux non finaux (ses autres sorties
    /// n'alimentent rien), et seule cette ligne est relue.
    pub(crate) fn qwen_moe_prefill_resident_from(
        &self,
        input: &Tensor,
        layers: &[PrefillMoeLayer<'_>],
        spec: PrefillAttentionSpec,
        capture_layer_ids: &[usize],
        chunk_tokens: usize,
        prefix: Option<&PrefillResidentPrefix<'_>>,
        last_row_only: bool,
    ) -> Result<(Tensor, Vec<PrefillResidentLayerCache>, Option<Tensor>)> {
        let profile_sections = prefill_profile_sections_enabled();
        if profile_sections {
            reset_prefill_f32_to_bf16_shapes();
        }
        let start = prefix.map_or(0, |prefix| prefix.rows);
        let (rows, hidden_dim) = input.as_matrix()?;
        if rows.checked_add(start) != Some(spec.seq) || hidden_dim != spec.hidden_dim {
            return Err(InferError::Dimension(format!(
                "prefill résident input=[{rows},{hidden_dim}] après {start}, spec seq={} hidden={}",
                spec.seq, spec.hidden_dim
            )));
        }
        if layers.is_empty() || rows == 0 {
            return Err(InferError::Config(
                "prefill résident sans couche ou sans token".to_string(),
            ));
        }
        if prefix.is_some_and(|prefix| prefix.layers.len() != layers.len())
            || (start > 0 && !self.prefill_resident_extend_supported(layers))
        {
            return Err(InferError::Config(
                "prefill résident prolongé hors périmètre".to_string(),
            ));
        }
        let chunk_rows = self.resident_prefill_chunk_rows(layers, rows, chunk_tokens);
        let chunk_count = rows.div_ceil(chunk_rows);
        let mut kernel_timing = if profile_sections {
            None
        } else {
            PrefillKernelTiming::try_new(&self.device, layers.len() * chunk_count)
        };
        let profile_total_started = profile_sections.then(std::time::Instant::now);
        let mut section_profile = PrefillSectionProfile::default();
        let trace = !profile_sections && trace_prefill_enabled();
        let total_started = trace.then(std::time::Instant::now);
        let hidden_len = checked_len(rows, hidden_dim, "prefill résident hidden")?;
        let chunk_hidden_capacity =
            checked_len(chunk_rows, hidden_dim, "prefill résident hidden morceau")?;
        let input_buffer = self.upload_f32_buffer(input.data(), "resident_input")?;
        let hidden_a = self.private_f32_buffer(chunk_hidden_capacity, "resident_hidden_a")?;
        let hidden_b = self.private_f32_buffer(chunk_hidden_capacity, "resident_hidden_b")?;
        let chunk_input = if chunk_count > 1 {
            Some(self.private_f32_buffer(chunk_hidden_capacity, "resident_chunk_input")?)
        } else {
            None
        };
        if let Some(bad) = capture_layer_ids.iter().find(|&&id| id >= layers.len()) {
            return Err(InferError::Config(format!(
                "prefill résident capture couche {bad} hors des {} couches",
                layers.len()
            )));
        }
        let capture_buffer = if capture_layer_ids.is_empty() {
            None
        } else {
            let len = checked_len(
                capture_layer_ids.len(),
                hidden_len,
                "prefill résident captures",
            )?;
            Some(self.uncached_f32_buffer(len, "resident_captures")?)
        };
        let final_output = if (chunk_count > 1 && !last_row_only) || private_scratch_enabled() {
            Some(self.uncached_f32_buffer(hidden_len, "resident_final_output")?)
        } else {
            None
        };
        // NOTE: seuls le KV pleine longueur et l'état linéaire survivent aux
        // morceaux ; tout le reste est un scratch dimensionné sur un morceau.
        let mut layer_cache_buffers = Vec::with_capacity(layers.len());
        for (layer_index, layer) in layers.iter().enumerate() {
            let (_, kv_dim, layer_shape) =
                self.resident_prefill_layer_shape(layer, layer_index, spec, hidden_dim)?;
            let prefix_layer = prefix.map(|prefix| &prefix.layers[layer_index]);
            let cache = match (prefix_layer, &layer_shape.attention) {
                (
                    Some(PrefillResidentPrefixLayer::Linear { state }),
                    PrefillResidentAttentionShape::Linear { .. },
                ) => PrefillResidentLayerCacheBuffer::Linear {
                    state: (*state).clone(),
                },
                (None, _)
                | (
                    Some(PrefillResidentPrefixLayer::Full { .. }),
                    PrefillResidentAttentionShape::Full { .. },
                ) => self.allocate_prefill_resident_layer_cache(&layer_shape, spec.seq, kv_dim)?,
                _ => {
                    return Err(InferError::Config(format!(
                        "préfixe résident incohérent couche {layer_index}"
                    )))
                }
            };
            if let (
                Some(PrefillResidentPrefixLayer::Full { key, value }),
                PrefillResidentLayerCacheBuffer::Full {
                    key: key_cache,
                    value: value_cache,
                    ..
                },
            ) = (prefix_layer, &cache)
            {
                let prefix_len = checked_len(start, kv_dim, "préfixe résident kv")?;
                let key_len: usize = key.iter().map(|segment| segment.len()).sum();
                let value_len: usize = value.iter().map(|segment| segment.len()).sum();
                if key_len != prefix_len || value_len != prefix_len {
                    return Err(InferError::Dimension(format!(
                        "préfixe résident KV couche {layer_index}: {key_len}/{value_len} attendu {prefix_len}"
                    )));
                }
                for (buffer, segments) in [(key_cache, key), (value_cache, value)] {
                    let mut offset = 0;
                    for segment in segments {
                        write_f32_buffer_at(buffer, offset, segment)?;
                        offset += segment.len();
                    }
                }
            }
            layer_cache_buffers.push(cache);
        }
        let mut owned_buffers = Vec::new();

        let command_buffer = self.queue.new_command_buffer();
        let use_shared_encoder = kernel_timing
            .as_ref()
            .is_none_or(PrefillKernelTiming::uses_dispatch_boundary);
        let encoder = if use_shared_encoder {
            Some(command_buffer.new_compute_command_encoder())
        } else {
            None
        };
        let mut encoder_guard = encoder.map(EncoderEndGuard::new);
        let fallback_encoder = encoder_guard.as_ref().map(EncoderEndGuard::encoder);
        if profile_sections {
            let guard = encoder_guard
                .take()
                .expect("invariant: garde encodeur initialisée");
            guard.end();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            ensure_completed(command_buffer.status())?;
        }
        let encode_started = trace.then(std::time::Instant::now);
        let mut current_buffer = input_buffer.clone();
        {
            let mut encode_context = ResidentPrefillEncodeContext {
                profile_sections,
                section_profile: &mut section_profile,
                kernel_timing: &mut kernel_timing,
                command_buffer,
                fallback_encoder,
                owned_buffers: &mut owned_buffers,
            };
            for chunk_start in (start..spec.seq).step_by(chunk_rows) {
                let chunk = ResidentPrefillChunk {
                    start: chunk_start,
                    rows: chunk_rows.min(spec.seq - chunk_start),
                    total: spec.seq,
                };
                let chunk_hidden_len =
                    checked_len(chunk.rows, hidden_dim, "prefill résident hidden morceau")?;
                let chunk_offset_bytes = f32_byte_offset(checked_len(
                    chunk.start - start,
                    hidden_dim,
                    "morceau offset",
                )?)?;
                if let Some(chunk_input) = chunk_input.as_ref() {
                    self.run_resident_prefill_section(
                        &mut encode_context,
                        "chunk_input",
                        |encoder, _owned| {
                            self.encode_copy_with_offsets(
                                encoder,
                                &input_buffer,
                                chunk_offset_bytes,
                                chunk_input,
                                0,
                                chunk_hidden_len,
                            )
                        },
                    )?;
                    current_buffer = chunk_input.clone();
                }
                for (layer_index, layer) in layers.iter().enumerate() {
                    let layer_spec = PrefillAttentionSpec {
                        seq: chunk.rows,
                        ..layer.attention_spec
                    };
                    let (q_dim, kv_dim, layer_shape) =
                        self.resident_prefill_layer_shape(layer, layer_index, spec, hidden_dim)?;
                    let kv_only = last_row_only
                        && layer_index + 1 == layers.len()
                        && chunk.start + chunk.rows < spec.seq
                        && matches!(layer.attention, PrefillAttentionLayer::Full { .. })
                        && !capture_layer_ids.contains(&layer_index);
                    // Scratch dimensionné sur `chunk_rows` : le dernier morceau, plus
                    // court, réutilise les mêmes buffers du cache de scratch.
                    let scratch = self.allocate_prefill_resident_layer_scratch(
                        &layer_shape,
                        PrefillAttentionSpec {
                            seq: chunk_rows,
                            ..layer_spec
                        },
                        hidden_dim,
                        q_dim,
                        kv_dim,
                        layer.tail.top_k(),
                    )?;
                    let tail_shape = layer_shape.tail;
                    let PrefillResidentLayerScratch {
                        input_norm: input_norm_buffer,
                        post_norm: post_norm_buffer,
                        normed: normed_buffer,
                        attention: attention_scratch,
                        attention_state: attention_state_buffer,
                        post_normed: post_normed_buffer,
                        tail: tail_scratch,
                    } = scratch;
                    let output_buffer = if layer_index % 2 == 0 {
                        hidden_a.clone()
                    } else {
                        hidden_b.clone()
                    };

                    self.encode_resident_input_norm_phase(
                        &mut encode_context,
                        &current_buffer,
                        &input_norm_buffer,
                        &normed_buffer,
                        layer_spec,
                        hidden_dim,
                    )?;
                    let attention_output_buffer = self.encode_resident_attention_phase(
                        &mut encode_context,
                        layer.attention,
                        attention_scratch,
                        &normed_buffer,
                        &layer_cache_buffers[layer_index],
                        chunk,
                        layer_spec,
                        hidden_dim,
                        q_dim,
                        layer_index,
                        kv_only,
                    )?;
                    if kv_only {
                        continue;
                    }
                    self.encode_resident_residual_phase(
                        &mut encode_context,
                        &current_buffer,
                        &attention_output_buffer,
                        &post_norm_buffer,
                        &attention_state_buffer,
                        &post_normed_buffer,
                        layer.post_norm_before_residual,
                        layer_spec,
                        hidden_dim,
                    )?;
                    self.encode_resident_moe_tail_phase(
                        &mut encode_context,
                        layer.tail,
                        tail_shape,
                        tail_scratch,
                        &post_normed_buffer,
                        &attention_state_buffer,
                        &output_buffer,
                        layer_spec,
                        hidden_dim,
                        chunk_hidden_len,
                        layer_index,
                    )?;
                    current_buffer = output_buffer;
                    if let Some(capture_buffer) = capture_buffer.as_ref() {
                        for slot in capture_layer_ids
                            .iter()
                            .enumerate()
                            .filter_map(|(slot, &id)| (id == layer_index).then_some(slot))
                        {
                            let offset = f32_byte_offset(
                                checked_len(slot, hidden_len, "prefill résident capture offset")?
                                    .checked_add((chunk.start - start) * hidden_dim)
                                    .ok_or_else(|| {
                                        InferError::Shape(
                                            "prefill résident capture offset déborde".to_string(),
                                        )
                                    })?,
                            )?;
                            self.run_resident_prefill_section(
                                &mut encode_context,
                                "capture_copy",
                                |encoder, _owned| {
                                    self.encode_copy_with_offsets(
                                        encoder,
                                        &current_buffer,
                                        0,
                                        capture_buffer,
                                        offset,
                                        chunk_hidden_len,
                                    )
                                },
                            )?;
                        }
                    }
                }
                if let Some(final_output) = final_output.as_ref() {
                    self.run_resident_prefill_section(
                        &mut encode_context,
                        "final_copy",
                        |encoder, _owned| {
                            self.encode_copy_with_offsets(
                                encoder,
                                &current_buffer,
                                0,
                                final_output,
                                chunk_offset_bytes,
                                chunk_hidden_len,
                            )
                        },
                    )?;
                }
            }
        }
        let final_output_used = final_output.is_some();
        let final_read_buffer = final_output.unwrap_or(current_buffer);
        let encode_elapsed = encode_started.map(|started| started.elapsed());
        let wait_elapsed = if profile_sections {
            None
        } else {
            if let Some(guard) = encoder_guard.take() {
                guard.end();
            }
            if let Some(timing) = kernel_timing.as_ref() {
                timing.encode_resolve(command_buffer)?;
            }
            let wait_started = trace.then(std::time::Instant::now);
            command_buffer.commit();
            command_buffer.wait_until_completed();
            ensure_completed(command_buffer.status())?;
            if let Some(timing) = kernel_timing.as_ref() {
                if let Err(error) = timing.dump_report() {
                    eprintln!("gpu_timestamps report_error={error}");
                }
            }
            wait_started.map(|started| started.elapsed())
        };

        let read_started = trace.then(std::time::Instant::now);
        let profile_read_started = profile_sections.then(std::time::Instant::now);
        let output = if last_row_only {
            // Dernière ligne du morceau final, ou de la sortie pleine longueur
            // quand le scratch privé l'impose.
            let last_row = if final_output_used {
                rows - 1
            } else {
                (rows - 1) % chunk_rows
            };
            Tensor::from_vec(
                vec![1, hidden_dim],
                read_f32_buffer_at(
                    &final_read_buffer,
                    checked_len(last_row, hidden_dim, "prefill résident dernière ligne")?,
                    hidden_dim,
                )?,
            )?
        } else {
            Tensor::from_vec(
                vec![rows, hidden_dim],
                read_f32_buffer(&final_read_buffer, hidden_len)?,
            )?
        };
        let captures = capture_buffer
            .map(|buffer| -> Result<Tensor> {
                let count = capture_layer_ids.len();
                let raw = read_f32_buffer(&buffer, count * hidden_len)?;
                // [n, seq, hidden] sur le GPU -> [seq, n * hidden] attendu par le drafter.
                let mut out = vec![0.0_f32; count * hidden_len];
                for slot in 0..count {
                    for row in 0..rows {
                        let src = slot * hidden_len + row * hidden_dim;
                        let dst = (row * count + slot) * hidden_dim;
                        out[dst..dst + hidden_dim].copy_from_slice(&raw[src..src + hidden_dim]);
                    }
                }
                Tensor::from_vec(vec![rows, count * hidden_dim], out)
            })
            .transpose()?;
        let mut layer_caches = Vec::with_capacity(layer_cache_buffers.len());
        for cache_buffer in layer_cache_buffers {
            match cache_buffer {
                PrefillResidentLayerCacheBuffer::Full { key, value, kv_dim } => {
                    // Seules les positions nouvelles repartent au CPU : le préfixe
                    // y est déjà.
                    let kv_len = checked_len(rows, kv_dim, "prefill résident readback kv")?;
                    let kv_offset = checked_len(start, kv_dim, "prefill résident readback kv")?;
                    // Prefill complet : le buffer, que plus rien n'écrit, devient
                    // le stockage CPU du K/V. Extension : il porte aussi le
                    // préfixe, seules les lignes nouvelles sont relues.
                    let (key, value) = if start == 0
                        && crate::runtime_flags::prefill_kv_shared_enabled()
                    {
                        (
                            PrefillKvRows::Shared(SharedF32Rows::new(key, 0, kv_len)?),
                            PrefillKvRows::Shared(SharedF32Rows::new(value, 0, kv_len)?),
                        )
                    } else {
                        (
                            PrefillKvRows::Owned(read_f32_buffer_at(&key, kv_offset, kv_len)?),
                            PrefillKvRows::Owned(read_f32_buffer_at(&value, kv_offset, kv_len)?),
                        )
                    };
                    layer_caches.push(PrefillResidentLayerCache::Full { key, value, kv_dim });
                }
                PrefillResidentLayerCacheBuffer::Linear { state } => {
                    layer_caches.push(PrefillResidentLayerCache::Linear { state });
                }
            }
        }
        let read_elapsed = read_started.map(|started| started.elapsed());
        if let Some(started) = profile_read_started {
            section_profile.add("readback", started.elapsed().as_micros(), 0);
        }
        if let Some(total_started) = total_started {
            eprintln!(
                "prefill_resident profile encode_us={} wait_us={} read_us={} total_us={}",
                encode_elapsed.map_or(0, |elapsed| elapsed.as_micros()),
                wait_elapsed.map_or(0, |elapsed| elapsed.as_micros()),
                read_elapsed.map_or(0, |elapsed| elapsed.as_micros()),
                total_started.elapsed().as_micros()
            );
        }
        if let Some(profile_total_started) = profile_total_started {
            let conversion_shapes = take_prefill_f32_to_bf16_shapes();
            self.profile_f32_to_bf16_conversions(&mut section_profile, &conversion_shapes)?;
            eprintln!(
                "prefill_section_run seq={} layers={} total_wall_us={}",
                spec.seq,
                layers.len(),
                profile_total_started.elapsed().as_micros()
            );
            section_profile.dump();
        }
        Ok((output, layer_caches, captures))
    }

    /// Lignes par morceau : les `rows` positions à calculer sont coupées en
    /// morceaux égaux d'au plus `target` lignes, si toutes les attentions
    /// complètes passent par le chemin Steel/NAX à décalage ; sinon un seul
    /// morceau.
    fn resident_prefill_chunk_rows(
        &self,
        layers: &[PrefillMoeLayer<'_>],
        rows: usize,
        target: usize,
    ) -> usize {
        // NOTE: les timestamps GPU plafonnent à 8192 échantillons ; un seul
        // morceau garde le profil par couche lisible.
        if target == 0
            || rows <= target
            || crate::runtime_flags::gpu_timestamps_enabled()
            || !self.prefill_resident_extend_supported(layers)
        {
            return rows;
        }
        rows.div_ceil(rows.div_ceil(target))
    }

    fn resident_prefill_layer_shape<'a>(
        &self,
        layer: &PrefillMoeLayer<'a>,
        layer_index: usize,
        spec: PrefillAttentionSpec,
        hidden_dim: usize,
    ) -> Result<(usize, usize, PrefillResidentLayerShape<'a>)> {
        let layer_spec = layer.attention_spec;
        if layer_spec.seq != spec.seq
            || layer_spec.hidden_dim != hidden_dim
            || layer_spec.q_heads != spec.q_heads
            || layer_spec.eps != spec.eps
        {
            return Err(InferError::Dimension(format!(
                "prefill résident spec incohérent couche {layer_index}: {layer_spec:?}"
            )));
        }
        let q_dim = checked_len(
            layer_spec.q_heads,
            layer_spec.head_dim,
            "prefill résident q_dim couche",
        )?;
        let kv_dim = checked_len(
            layer_spec.kv_heads,
            layer_spec.head_dim,
            "prefill résident kv_dim couche",
        )?;
        let shape = self.check_prefill_resident_layer_shapes(
            *layer,
            layer_index,
            hidden_dim,
            q_dim,
            kv_dim,
            layer_spec,
        )?;
        Ok((q_dim, kv_dim, shape))
    }

    #[inline]
    pub(super) fn run_resident_prefill_section<F>(
        &self,
        context: &mut ResidentPrefillEncodeContext<'_>,
        label: &'static str,
        encode: F,
    ) -> Result<()>
    where
        F: FnOnce(&ComputeCommandEncoderRef, &mut Vec<Buffer>) -> Result<()>,
    {
        if context.profile_sections {
            self.run_prefill_profile_section(context.section_profile, label, encode)
        } else {
            time_prefill_pass(
                context.kernel_timing.as_mut(),
                context.command_buffer,
                context.fallback_encoder,
                label,
                |encoder| encode(encoder, context.owned_buffers),
            )
        }
    }

    #[inline]
    fn encode_resident_input_norm_phase(
        &self,
        context: &mut ResidentPrefillEncodeContext<'_>,
        current_buffer: &BufferRef,
        input_norm_buffer: &BufferRef,
        normed_buffer: &BufferRef,
        spec: PrefillAttentionSpec,
        hidden_dim: usize,
    ) -> Result<()> {
        self.run_resident_prefill_section(context, "input_norm", |encoder, _owned| {
            self.encode_rms_norm_rows(
                encoder,
                current_buffer,
                input_norm_buffer,
                normed_buffer,
                spec.seq,
                hidden_dim,
                spec.eps,
            )
        })
    }

    #[inline]
    fn encode_resident_residual_phase(
        &self,
        context: &mut ResidentPrefillEncodeContext<'_>,
        current_buffer: &BufferRef,
        attention_output_buffer: &BufferRef,
        post_norm_buffer: &BufferRef,
        attention_state_buffer: &BufferRef,
        post_normed_buffer: &BufferRef,
        post_norm_before_residual: bool,
        spec: PrefillAttentionSpec,
        hidden_dim: usize,
    ) -> Result<()> {
        self.run_resident_prefill_section(context, "o_postnorm", |encoder, owned| {
            if post_norm_before_residual {
                self.encode_rms_norm_rows(
                    encoder,
                    attention_output_buffer,
                    post_norm_buffer,
                    post_normed_buffer,
                    spec.seq,
                    hidden_dim,
                    spec.eps,
                )?;
                return self.encode_add_scaled(
                    encoder,
                    owned,
                    current_buffer,
                    post_normed_buffer,
                    attention_state_buffer,
                    1.0,
                    checked_len(spec.seq, hidden_dim, "resident Gemma attention state")?,
                );
            }
            self.encode_add_rms_norm_rows(
                encoder,
                current_buffer,
                attention_output_buffer,
                post_norm_buffer,
                attention_state_buffer,
                post_normed_buffer,
                spec.seq,
                hidden_dim,
                spec.eps,
            )
        })
    }

    #[inline]
    fn encode_resident_moe_tail_phase(
        &self,
        context: &mut ResidentPrefillEncodeContext<'_>,
        tail: PrefillMoeTail<'_>,
        tail_shape: PrefillResidentTailShape,
        tail_scratch: PrefillResidentTailScratch,
        post_normed_buffer: &BufferRef,
        attention_state_buffer: &BufferRef,
        output_buffer: &BufferRef,
        spec: PrefillAttentionSpec,
        hidden_dim: usize,
        hidden_len: usize,
        layer_index: usize,
    ) -> Result<()> {
        match (tail, tail_shape, tail_scratch) {
            (
                PrefillMoeTail::Dense { .. },
                PrefillResidentTailShape::Dense {
                    gate_proj,
                    up_proj,
                    down_proj,
                    inter_dim,
                },
                PrefillResidentTailScratch::Dense { unfused, down },
            ) => self.run_resident_prefill_section(context, "tail_dense", |encoder, owned| {
                if let Some((gate, up, hidden)) = &unfused {
                    let gate_dim = self.encode_matmul_weight_buffers(
                        encoder,
                        post_normed_buffer,
                        spec.seq,
                        hidden_dim,
                        &gate_proj,
                        gate,
                        false,
                    )?;
                    let up_dim = self.encode_matmul_weight_buffers(
                        encoder,
                        post_normed_buffer,
                        spec.seq,
                        hidden_dim,
                        &up_proj,
                        up,
                        false,
                    )?;
                    if gate_dim != inter_dim || up_dim != inter_dim {
                        return Err(InferError::Dimension(format!(
                            "prefill résident dense gate/up sortent gate={gate_dim} up={up_dim}, attendu {inter_dim}"
                        )));
                    }
                    let inter_len = checked_len(spec.seq, inter_dim, "resident dense inter")?;
                    self.encode_swiglu(encoder, owned, gate, up, hidden, inter_len)?;
                    let down_dim = self.encode_matmul_weight_buffers(
                        encoder, hidden, spec.seq, inter_dim, &down_proj, &down, false,
                    )?;
                    if down_dim != hidden_dim {
                        return Err(InferError::Dimension(format!(
                            "prefill résident dense down sort {down_dim}, attendu {hidden_dim}"
                        )));
                    }
                } else {
                    self.encode_fused_dense_mlp(
                        encoder,
                        post_normed_buffer,
                        spec.seq,
                        hidden_dim,
                        &gate_proj,
                        &up_proj,
                        &down_proj,
                        inter_dim,
                        &down,
                    )?;
                }
                self.encode_copy(
                    encoder,
                    attention_state_buffer,
                    output_buffer,
                    hidden_len,
                )?;
                self.encode_accumulate_scaled(
                    encoder,
                    owned,
                    &down,
                    output_buffer,
                    1.0,
                    hidden_len,
                )
            }),
            (
                PrefillMoeTail::GemmaDense { .. },
                PrefillResidentTailShape::GemmaDense {
                    gate_proj,
                    up_proj,
                    down_proj,
                    pre_feedforward_norm,
                    post_feedforward_norm,
                    layer_scalar,
                    inter_dim,
                },
                PrefillResidentTailScratch::GemmaDense {
                    ffn_input,
                    gate,
                    up,
                    geglu,
                    down,
                    ffn_normed,
                },
            ) => self.run_resident_prefill_section(
                context,
                "tail_gemma_dense",
                |encoder, owned| {
                    self.encode_gemma_dense_tail_rows(
                        encoder,
                        owned,
                        attention_state_buffer,
                        output_buffer,
                        spec.seq,
                        hidden_dim,
                        spec.eps,
                        &gate_proj,
                        &up_proj,
                        &down_proj,
                        &pre_feedforward_norm,
                        &post_feedforward_norm,
                        layer_scalar,
                        inter_dim,
                        &ffn_input,
                        &gate,
                        &up,
                        &geglu,
                        &down,
                        &ffn_normed,
                    )
                },
            ),
            (
                PrefillMoeTail::GemmaParallel { top_k, .. },
                PrefillResidentTailShape::GemmaParallel {
                    dense_gate_proj,
                    dense_up_proj,
                    dense_down_proj,
                    pre_feedforward_norm,
                    post_feedforward_norm_1,
                    moe,
                    router_norm,
                    per_expert_scale,
                    pre_feedforward_norm_2,
                    post_feedforward_norm_2,
                    post_feedforward_norm,
                    layer_scalar,
                    dense_inter_dim,
                },
                PrefillResidentTailScratch::GemmaParallel {
                    dense_input,
                    dense_gate,
                    dense_up,
                    dense_geglu,
                    dense_down,
                    dense_out,
                    moe_input,
                    moe_out,
                    ffn_out,
                    ffn_normed,
                },
            ) => self.run_resident_prefill_section(
                context,
                "tail_gemma_parallel",
                |encoder, owned| {
                    self.encode_gemma_parallel_tail_rows(
                        encoder,
                        owned,
                        attention_state_buffer,
                        output_buffer,
                        spec.seq,
                        hidden_dim,
                        spec.eps,
                        &dense_gate_proj,
                        &dense_up_proj,
                        &dense_down_proj,
                        &pre_feedforward_norm,
                        &post_feedforward_norm_1,
                        &moe,
                        top_k,
                        router_norm
                            .as_ref()
                            .map(|(weight, eps)| (weight.as_ref(), *eps)),
                        per_expert_scale.as_ref().map(Buffer::as_ref),
                        &pre_feedforward_norm_2,
                        &post_feedforward_norm_2,
                        &post_feedforward_norm,
                        layer_scalar,
                        dense_inter_dim,
                        &dense_input,
                        &dense_gate,
                        &dense_up,
                        &dense_geglu,
                        &dense_down,
                        &dense_out,
                        &moe_input,
                        &moe_out,
                        &ffn_out,
                        &ffn_normed,
                    )
                },
            ),
            (
                PrefillMoeTail::Routed { router, top_k, .. },
                PrefillResidentTailShape::Routed {
                    expert_count,
                    stacked,
                },
                PrefillResidentTailScratch::Routed {
                    router: router_buffer,
                    indices: indices_buffer,
                    scores: scores_buffer,
                    gate: gate_buffer,
                    up: up_buffer,
                    hidden: hidden_buffer,
                    down: down_buffer,
                },
            ) => {
                let total_topk = checked_len(spec.seq, top_k, "resident topk total")?;
                let inter_dim = stacked.gate.out_dim;
                // Bascule routed-only coop gatée par un flag DÉDIÉ (défaut OFF) :
                // l'oracle greedy 30B n'est pas qualifié (cf.
                // `moe_routed_coop_prefill_enabled`). Le défaut garde le chemin
                // gather-qmv par lignes, byte-identique à la base. Le routeur n'est
                // résolu que dans la branche coop.
                if moe_routed_coop_prefill_enabled()
                    && MetalMoeRoutedWeights::stacked_coop_compatible(&stacked)
                {
                    let weights = MetalMoeRoutedWeights {
                        router: self.resolve_linear_weight_buffers(
                            router.weight(),
                            "resident_moe_router",
                        )?,
                        stacked,
                    };
                    self.run_resident_prefill_section(
                        context,
                        "moe_routed_coop",
                        |encoder, owned| {
                            self.encode_moe_routed_rows_coop(
                                encoder,
                                owned,
                                post_normed_buffer,
                                Some(attention_state_buffer),
                                output_buffer,
                                &router_buffer,
                                &indices_buffer,
                                &scores_buffer,
                                &down_buffer,
                                spec.seq,
                                hidden_dim,
                                &weights,
                                top_k,
                            )
                        },
                    )
                } else {
                    self.run_resident_prefill_section(
                        context,
                        "router_topk",
                        |encoder, owned| {
                            self.encode_matmul_weight(
                                encoder,
                                owned,
                                post_normed_buffer,
                                spec.seq,
                                hidden_dim,
                                router.weight(),
                                &router_buffer,
                            )?;
                            self.encode_topk_softmax_rows(
                                encoder,
                                &router_buffer,
                                &indices_buffer,
                                &scores_buffer,
                                spec.seq,
                                expert_count,
                                top_k,
                            )
                        },
                    )?;
                    self.run_resident_prefill_section(
                        context,
                        "moe_gate_up",
                        |encoder, owned| {
                            if !self.encode_gather_gate_up_swiglu(
                                encoder,
                                owned,
                                post_normed_buffer,
                                spec.seq,
                                &stacked.gate,
                                &stacked.up,
                                &indices_buffer,
                                total_topk,
                                &hidden_buffer,
                            )? {
                                self.encode_gather_matmul(
                                    encoder,
                                    owned,
                                    post_normed_buffer,
                                    spec.seq,
                                    &stacked.gate,
                                    &indices_buffer,
                                    total_topk,
                                    &gate_buffer,
                                )?;
                                self.encode_gather_matmul(
                                    encoder,
                                    owned,
                                    post_normed_buffer,
                                    spec.seq,
                                    &stacked.up,
                                    &indices_buffer,
                                    total_topk,
                                    &up_buffer,
                                )?;
                                self.encode_swiglu(
                                    encoder,
                                    owned,
                                    &gate_buffer,
                                    &up_buffer,
                                    &hidden_buffer,
                                    checked_len(total_topk, inter_dim, "resident swiglu")?,
                                )?;
                            }
                            Ok(())
                        },
                    )?;
                    self.run_resident_prefill_section(
                        context,
                        "moe_down",
                        |encoder, owned| {
                            self.encode_gather_matmul(
                                encoder,
                                owned,
                                &hidden_buffer,
                                total_topk,
                                &stacked.down,
                                &indices_buffer,
                                total_topk,
                                &down_buffer,
                            )
                        },
                    )?;
                    self.run_resident_prefill_section(
                        context,
                        "moe_weighted_sum",
                        |encoder, owned| {
                            self.encode_weighted_sum_add_grouped_topk(
                                encoder,
                                owned,
                                &down_buffer,
                                &scores_buffer,
                                attention_state_buffer,
                                output_buffer,
                                spec.seq,
                                top_k,
                                hidden_dim,
                            )
                        },
                    )
                }
            }
            (
                PrefillMoeTail::Shared { top_k, .. },
                PrefillResidentTailShape::Shared { weights },
                PrefillResidentTailScratch::Shared,
            ) => {
                if moe_coop_enabled() && weights.coop_compatible() {
                    self.run_resident_prefill_section(
                        context,
                        "moe_shared_coop",
                        |encoder, owned| {
                            self.encode_moe_shared_rows_coop(
                                encoder,
                                owned,
                                post_normed_buffer,
                                Some(attention_state_buffer),
                                output_buffer,
                                spec.seq,
                                hidden_dim,
                                &weights,
                                top_k,
                            )
                        },
                    )
                } else {
                    self.run_resident_prefill_section(
                        context,
                        "moe_shared_rows",
                        |encoder, owned| {
                            self.encode_moe_shared_buffers_rows(
                                encoder,
                                owned,
                                post_normed_buffer,
                                Some(attention_state_buffer),
                                output_buffer,
                                spec.seq,
                                hidden_dim,
                                &weights,
                                top_k,
                            )
                        },
                    )
                }
            }
            _ => Err(InferError::Dimension(format!(
                "prefill résident tail MoE incohérent couche {layer_index}"
            ))),
        }
    }
}
