use super::super::*;
use super::types::*;

impl CausalDecoder {
    /// Forward d'UN token de decode par le chemin résident COMPLET (1c) : embed
    /// gather (CPU, input upload) → boucle des 40 couches en **UN** command buffer →
    /// final_norm → lm_head + argmax GPU → lecture d'**1 `u32`** (l'id du prochain
    /// token). Aucun readback au milieu ; le seul aller-retour CPU restant est la
    /// dépendance auto-régressive irréductible (embed du token en entrée, argmax en
    /// sortie). Renvoie directement le token (greedy argmax on-device).
    ///
    /// Préconditions validées en amont ([`Self::supports_resident_full_decode`]) et
    /// arène prête ([`Self::setup_resident_full_decode`]). Les états par couche
    /// (KV full-attn seedé, conv/ssm linear-attn peuplé au prefill résident) sont
    /// réutilisés in-place.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si l'arène ou un état résident est absent, ou si un
    /// encodage Metal échoue.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(in crate::decoder) fn decode_token_resident(
        &self,
        cache: &mut CausalDecoderCache,
        token_id: usize,
    ) -> Result<usize> {
        let inflight = self.enqueue_decode_token_resident(
            cache,
            ResidentDecodeInput::CpuToken(token_id),
            None,
            None,
            None,
        )?;
        crate::metal_backend::wait_for_completion(&inflight.command_buffer)?;
        let raw = crate::metal_backend::read_u32_buffer(&inflight.index, 1)?
            .into_iter()
            .next()
            .ok_or_else(|| InferError::Metal("decode résident sans token".to_string()))?;
        usize::try_from(raw)
            .map_err(|_| InferError::Metal(format!("token résident hors usize: {raw}")))
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(in crate::decoder) fn decode_token_resident_sampled(
        &self,
        cache: &mut CausalDecoderCache,
        token_id: usize,
        options: &GenerationOptions,
        sampler: &mut DeterministicSampler,
    ) -> Result<usize> {
        let sample = resident_sample_spec_for_solo(options, sampler)?;
        let inflight = self.enqueue_decode_token_resident(
            cache,
            ResidentDecodeInput::CpuToken(token_id),
            None,
            None,
            Some(sample),
        )?;
        crate::metal_backend::wait_for_completion(&inflight.command_buffer)?;
        if let Some(readback) = inflight.logits_readback {
            let logits = crate::metal_backend::read_f32_buffer(&readback.buffer, readback.len)?;
            let logits = self.finalize_logits(&Tensor::from_vec(vec![1, readback.len], logits)?)?;
            return sample_token_top_k_top_p(
                logits.as_row()?,
                options.temperature,
                options.top_p,
                options.top_k,
                sampler,
            );
        }
        sampler.advance();
        let raw = crate::metal_backend::read_u32_buffer(&inflight.index, 1)?
            .into_iter()
            .next()
            .ok_or_else(|| InferError::Metal("decode résident sans token".to_string()))?;
        usize::try_from(raw)
            .map_err(|_| InferError::Metal(format!("token résident hors usize: {raw}")))
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(in crate::decoder) fn decode_tokens_resident_pipelined(
        &self,
        cache: &mut CausalDecoderCache,
        first_token: usize,
        max_new_tokens: usize,
        options: &GenerationOptions,
        sampler: &mut DeterministicSampler,
    ) -> Result<Option<ResidentPipelineOutput>> {
        let Some(arena) = cache.resident.as_ref() else {
            return Ok(None);
        };
        if arena.state.gpu_timer().is_some() || arena.index_ring.len() < RESIDENT_PIPELINE_WINDOW {
            return Ok(None);
        }

        let mut generated = Vec::with_capacity(max_new_tokens);
        if options.stop_token_ids.contains(&first_token) {
            return Ok(Some(ResidentPipelineOutput {
                tokens: generated,
                decode: Duration::ZERO,
                decode_tokens: 0,
            }));
        }
        generated.push(first_token);
        if max_new_tokens == 1 {
            return Ok(Some(ResidentPipelineOutput {
                tokens: generated,
                decode: Duration::ZERO,
                decode_tokens: 0,
            }));
        }

        let started = Instant::now();
        let mut inflight: std::collections::VecDeque<ResidentDecodeInflight> =
            std::collections::VecDeque::new();
        let mut enqueued = 0_usize;
        let decode_limit = max_new_tokens - 1;
        let mut previous_index: Option<metal::Buffer> = None;
        let mut stopped = false;
        while generated.len() < max_new_tokens && !stopped {
            while enqueued < decode_limit && inflight.len() < RESIDENT_PIPELINE_WINDOW {
                let slot = enqueued % RESIDENT_PIPELINE_WINDOW;
                let output_index = cache
                    .resident
                    .as_ref()
                    .and_then(|arena| arena.index_ring.get(enqueued % RESIDENT_PIPELINE_WINDOW))
                    .ok_or_else(|| InferError::Metal("ring argmax résident absent".to_string()))?
                    .buffer()
                    .clone();
                let input = match previous_index.as_ref() {
                    Some(index) => ResidentDecodeInput::GpuIndex(index),
                    None => ResidentDecodeInput::CpuToken(first_token),
                };
                let sample = if options.temperature > f32::EPSILON {
                    let spec = resident_sample_spec(options, sampler)?;
                    sampler.advance();
                    Some(spec)
                } else {
                    None
                };
                let pipeline_slot = self.config.parallel_moe.then_some(slot);
                let step = self.enqueue_decode_token_resident(
                    cache,
                    input,
                    Some(&output_index),
                    pipeline_slot,
                    sample,
                )?;
                previous_index = Some(output_index);
                inflight.push_back(step);
                enqueued += 1;
            }

            let Some(step) = inflight.pop_front() else {
                break;
            };
            crate::metal_backend::wait_for_completion(&step.command_buffer)?;
            let raw = crate::metal_backend::read_u32_buffer(&step.index, 1)?
                .into_iter()
                .next()
                .ok_or_else(|| InferError::Metal("decode pipeline sans token".to_string()))?;
            let token = usize::try_from(raw)
                .map_err(|_| InferError::Metal(format!("token pipeline hors usize: {raw}")))?;
            if options.stop_token_ids.contains(&token) {
                stopped = true;
            } else {
                generated.push(token);
            }
        }

        for step in inflight {
            crate::metal_backend::wait_for_completion(&step.command_buffer)?;
        }

        Ok(Some(ResidentPipelineOutput {
            tokens: generated,
            decode: started.elapsed(),
            decode_tokens: enqueued,
        }))
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn enqueue_decode_token_resident(
        &self,
        cache: &mut CausalDecoderCache,
        input: ResidentDecodeInput<'_>,
        output_index: Option<&metal::BufferRef>,
        pipeline_slot: Option<usize>,
        sample: Option<ResidentSampleSpec>,
    ) -> Result<ResidentDecodeInflight> {
        let Some(metal) = self.forward_runtime().metal_executor() else {
            return Err(InferError::Metal(
                "decode résident sans executor Metal".to_string(),
            ));
        };
        let theta = self.config.rope_theta.ok_or_else(|| {
            InferError::Config("rope_theta manquant (decode résident)".to_string())
        })?;
        let eps = self.config.rms_eps;

        let hidden = self.final_norm.data().len();
        let position = cache.position;
        let linear_dims = if self.has_resident_linear_attention_layer() {
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
                    InferError::Shape("conv_dim déborde (decode résident)".to_string())
                })?;
            Some((la_spec, key_dim, value_dim, conv_dim))
        } else {
            None
        };

        // Borrows disjoints : l'arène (`resident`) vs les états par couche (`layers`).
        let CausalDecoderCache {
            layers,
            position: cache_position,
            resident,
        } = cache;
        let arena = resident.as_mut().ok_or_else(|| {
            InferError::Metal("arène résidente absente (decode résident)".to_string())
        })?;

        // Instrumentation per-section (tranche 3, `RETI_RUST_GPU_COUNTERS`) :
        // `Some` → forward segmenté en 1 command buffer par couche + 1 pour lm_head
        // (chronométrés CPU) ; `None` → command buffer unique 1c.4, inchangé. La
        // segmentation ne change ni les kernels ni l'ordre ni l'état GPU persistant
        // (ping-pong/KV/ssm) → résultat numérique identique.
        let timer = arena.state.gpu_timer();

        let mut command_buffer = arena.state.queue().new_command_buffer();
        let _barrier_guard = crate::metal_backend::resident_concurrent_enabled()
            .then(crate::metal_backend::install_dispatch_barrier_scope);
        let scratch_namespace = match pipeline_slot {
            Some(slot) => pipeline_scratch_namespace(arena.state.scratch_namespace(), slot)?,
            None => arena.state.scratch_namespace(),
        };
        let _namespace_guard = crate::metal_backend::install_scratch_namespace(scratch_namespace);
        let _resident_scratch_guard = pipeline_slot.map(install_pipeline_scratch_slot);
        let mut encoder = new_resident_compute_encoder(command_buffer);
        let mut owned: Vec<metal::Buffer> = Vec::new();

        let (hidden_a, hidden_b) = match pipeline_slot {
            Some(slot) => {
                let (hidden_a, hidden_b) =
                    arena.pipeline_hidden_ring.get(slot).ok_or_else(|| {
                        InferError::Metal(format!("ping-pong pipeline absent pour le slot {slot}"))
                    })?;
                (hidden_a, hidden_b)
            }
            None => (&arena.hidden_a, &arena.hidden_b),
        };

        match input {
            ResidentDecodeInput::CpuToken(token_id) => {
                // Embed gather (CPU) — input upload, PAS un readback.
                let embed = self.embed_scaled(&[token_id])?;
                let (_, embed_hidden) = embed.as_matrix()?;
                if embed_hidden != hidden {
                    return Err(InferError::Dimension(format!(
                        "embedding hidden={embed_hidden}, attendu {hidden}"
                    )));
                }
                arena.state.upload(hidden_a, embed.data())?;
            }
            ResidentDecodeInput::GpuIndex(index_buffer) => {
                metal.encode_embedding_from_index_buffers_scaled(
                    encoder,
                    &arena.embed_tokens,
                    index_buffer,
                    hidden_a.buffer(),
                    hidden,
                    self.config.embed_scale.unwrap_or(1.0),
                    self.config.is_qwen && qwen_embed_bf16_enabled(),
                )?;
            }
        }

        let mut current = hidden_a;
        let mut other = hidden_b;
        for (index, layer) in self.layers.iter().enumerate() {
            let layer_cache = &mut layers[index];
            let is_full = self.config.is_resident_full_attention_layer(index);
            if is_full {
                let dims = self
                    .config
                    .resident_windowed_full_attn_layer_dims(index, hidden, position, eps, theta)?;
                self.encode_resident_full_layer(
                    metal,
                    arena,
                    layer_cache,
                    layer,
                    index,
                    encoder,
                    &mut owned,
                    dims,
                    current.buffer(),
                    other.buffer(),
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
                    current.buffer(),
                    other.buffer(),
                )?;
            }
            std::mem::swap(&mut current, &mut other);

            // Segmentation per-section : 1 command buffer par couche, chronométré
            // CPU via le wait (le GPU n'exécute qu'au commit). Repart sur un CB neuf.
            if let Some(timer) = timer {
                encoder.end_encoding();
                let started = Instant::now();
                crate::metal_backend::commit_and_wait(command_buffer)?;
                timer.record_layer(is_full, started.elapsed().as_nanos());
                command_buffer = arena.state.queue().new_command_buffer();
                encoder = new_resident_compute_encoder(command_buffer);
            }
        }

        // final_norm (scratch) → lm_head + argmax/sampler GPU → 1 u32.
        let final_normed = arena.state.scratch().lease(hidden, GpuElement::F32)?;
        metal.encode_rms_norm_rows(
            encoder,
            current.buffer(),
            &arena.final_norm,
            final_normed.tensor().buffer(),
            1,
            hidden,
            eps,
        )?;
        let output = output_index.unwrap_or_else(|| arena.index.buffer());
        let mut logits_readback = None;
        if let Some(sample) = sample {
            match sample.mode {
                ResidentSampleMode::OnDevice if sample.top_k == 0 => {
                    metal.encode_lm_head_sample_gumbel_buffers(
                        encoder,
                        final_normed.tensor().buffer(),
                        &arena.lm_head,
                        output,
                        hidden,
                        sample.temperature,
                        sample.rng_state,
                    )?;
                }
                ResidentSampleMode::OnDevice => {
                    metal.encode_lm_head_sample_topk_topp_buffers(
                        encoder,
                        &mut owned,
                        final_normed.tensor().buffer(),
                        &arena.lm_head,
                        output,
                        hidden,
                        sample.temperature,
                        sample.top_p,
                        sample.top_k,
                        sample.rng_state,
                    )?;
                }
                ResidentSampleMode::Readback => {
                    let (buffer, len) = metal.encode_lm_head_logits_readback_buffers(
                        encoder,
                        final_normed.tensor().buffer(),
                        &arena.lm_head,
                        hidden,
                    )?;
                    logits_readback = Some(ResidentLogitsReadback { buffer, len });
                }
            }
        } else {
            metal.encode_lm_head_argmax_buffers(
                encoder,
                &mut owned,
                final_normed.tensor().buffer(),
                &arena.lm_head,
                output,
                hidden,
            )?;
        }
        // Section lm_head (instrumentée) ou commit unique du token (chemin 1c.4).
        let lmhead_started = timer.map(|_| Instant::now());
        encoder.end_encoding();
        crate::metal_backend::commit_nonblocking(command_buffer);
        if let (Some(timer), Some(started)) = (timer, lmhead_started) {
            timer.record_lmhead(started.elapsed().as_nanos());
        }
        // Diagnostic C1B (hors prod, gaté `RETI_RUST_ORACLE_DUMP_LOGITS`) : relit
        // l'état post-`final_norm` du token courant et recalcule les logits pleins
        // (lm_head), pour dumper les top-k et mesurer la marge top1-top2 (near-tie
        // bf16 vs dégradation). Sans effet sur la trajectoire : le token émis reste
        // l'argmax on-device déjà encodé ; ce chemin est non-pipeliné (cf. la garde
        // du greedy loop) donc la complétion GPU forcée ici est sûre.
        if let Some(k) = crate::decoder::flags::oracle_dump_logits_topk() {
            crate::metal_backend::wait_for_completion(command_buffer)?;
            let normed =
                crate::metal_backend::read_f32_buffer(final_normed.tensor().buffer(), hidden)?;
            let state = Tensor::from_vec(vec![1, hidden], normed)?;
            let logits = self.logits_from_final_state(&state)?;
            let row = logits.as_row()?;
            let mut order: Vec<usize> = (0..row.len()).collect();
            order.sort_unstable_by(|&a, &b| {
                row[b]
                    .partial_cmp(&row[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let top = order
                .iter()
                .take(k)
                .map(|&i| format!("{i}:{:.6}", row[i]))
                .collect::<Vec<_>>()
                .join(" ");
            eprintln!("oracle_logit_topk pos={position} {top}");
        }
        drop(final_normed);
        *cache_position += 1;
        Ok(ResidentDecodeInflight {
            command_buffer: command_buffer.to_owned(),
            index: output_index
                .map(|buffer| buffer.to_owned())
                .unwrap_or_else(|| arena.index.buffer().clone()),
            logits_readback,
            _owned: owned,
        })
    }

    /// Décode UN pas en résident depuis un EMBEDDING d'entrée (TTS talker/cp) : la
    /// boucle des couches tourne en **UN** command buffer, puis selon `head` :
    /// `None` → renvoie le `final_state` **post `final_norm`** relu (talker, l'argmax
    /// reste côté hôte) ; `Some(head)` → matmul `head` + argmax greedy on-device dans
    /// le même command buffer, renvoie le token (cp, tue le readback `cp_state` + le
    /// matmul de tête CPU). Réutilise les kernels résidents et le KV seedé par
    /// [`Self::setup_resident_full_decode`].
    ///
    /// Renvoie `Ok(None)` si l'executor Metal ou l'arène résidente est absent
    /// (l'appelant retombe sur le per-op).
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si l'embedding n'a pas la forme `[1, hidden]`, si un état
    /// résident est absent ou si un encodage Metal échoue.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(in crate::decoder) fn next_resident_embedding_step(
        &self,
        cache: &mut CausalDecoderCache,
        embedding: &Tensor,
        head: Option<&crate::metal_backend::MetalLinearWeightBuffers>,
    ) -> Result<Option<ResidentEmbeddingOut>> {
        let Some(metal) = self.forward_runtime().metal_executor() else {
            return Ok(None);
        };
        let hidden = self.final_norm.data().len();
        let (rows, embed_hidden) = embedding.as_matrix()?;
        if rows != 1 || embed_hidden != hidden {
            return Err(InferError::Dimension(format!(
                "embedding decode résident attendu [1, {hidden}], reçu {:?}",
                embedding.shape()
            )));
        }
        let theta = self.config.rope_theta.ok_or_else(|| {
            InferError::Config("rope_theta manquant (decode résident embedding)".to_string())
        })?;
        let eps = self.config.rms_eps;
        let position = cache.position;
        let linear_dims = if self.has_resident_linear_attention_layer() {
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
                    InferError::Shape("conv_dim déborde (decode résident embedding)".to_string())
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

        // Embedding fourni → input upload (PAS un readback).
        arena.state.upload(&arena.hidden_a, embedding.as_row()?)?;

        let command_buffer = arena.state.queue().new_command_buffer();
        let _barrier_guard = crate::metal_backend::resident_concurrent_enabled()
            .then(crate::metal_backend::install_dispatch_barrier_scope);
        let _namespace_guard =
            crate::metal_backend::install_scratch_namespace(arena.state.scratch_namespace());
        let encoder = new_resident_compute_encoder(command_buffer);
        let mut owned: Vec<metal::Buffer> = Vec::new();

        let mut current = &arena.hidden_a;
        let mut other = &arena.hidden_b;
        for (index, layer) in self.layers.iter().enumerate() {
            let layer_cache = &mut layers[index];
            if self.config.is_resident_full_attention_layer(index) {
                let dims = self
                    .config
                    .resident_windowed_full_attn_layer_dims(index, hidden, position, eps, theta)?;
                self.encode_resident_full_layer(
                    metal,
                    arena,
                    layer_cache,
                    layer,
                    index,
                    encoder,
                    &mut owned,
                    dims,
                    current.buffer(),
                    other.buffer(),
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
                    current.buffer(),
                    other.buffer(),
                )?;
            }
            std::mem::swap(&mut current, &mut other);
        }

        let final_normed = arena.state.scratch().lease(hidden, GpuElement::F32)?;
        metal.encode_rms_norm_rows(
            encoder,
            current.buffer(),
            &arena.final_norm,
            final_normed.tensor().buffer(),
            1,
            hidden,
            eps,
        )?;
        match head {
            None => {
                encoder.end_encoding();
                crate::metal_backend::commit_and_wait(command_buffer)?;
                let state =
                    crate::metal_backend::read_f32_buffer(final_normed.tensor().buffer(), hidden)?;
                drop(final_normed);
                *cache_position += 1;
                Ok(Some(ResidentEmbeddingOut::State(Tensor::row(state)?)))
            }
            Some(head) => {
                // Tête fournie : matmul + argmax greedy on-device dans le MÊME command
                // buffer → readback d'1 u32 (pas de cp_state relu + matmul tête CPU).
                let token_index = arena.state.scratch().lease(1, GpuElement::U32)?;
                metal.encode_lm_head_argmax_buffers(
                    encoder,
                    &mut owned,
                    final_normed.tensor().buffer(),
                    head,
                    token_index.tensor().buffer(),
                    hidden,
                )?;
                encoder.end_encoding();
                crate::metal_backend::commit_and_wait(command_buffer)?;
                let raw = crate::metal_backend::read_u32_buffer(token_index.tensor().buffer(), 1)?
                    .into_iter()
                    .next()
                    .ok_or_else(|| InferError::Metal("token cp résident absent".to_string()))?;
                drop(token_index);
                drop(final_normed);
                *cache_position += 1;
                let token = usize::try_from(raw).map_err(|_| {
                    InferError::Metal(format!("token cp résident hors usize: {raw}"))
                })?;
                Ok(Some(ResidentEmbeddingOut::Token(token)))
            }
        }
    }

    /// Dispatch d'un pas de decode per-op (qui prend lui-même le chemin résident
    /// full-attn 1b si `cache.full` est présent). Le chemin résident COMPLET (1c)
    /// est dispatché séparément dans la boucle de génération (renvoie le token).
    /// Hors metal, le garde interne est faux et on retombe sur le per-op cache.
    ///
    /// # Errors
    ///
    /// Propage les erreurs du forward.
    pub(in crate::decoder) fn next_decode_state(
        &self,
        cache: &mut CausalDecoderCache,
        token_id: usize,
    ) -> Result<Tensor> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if let Some(output) =
            self.next_final_states_resident_verify(cache, &[token_id], false, false, None)?
        {
            return Tensor::row(output.states.row_slice(0)?.to_vec());
        }
        self.next_final_state_cached(cache, token_id)
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn pipeline_scratch_namespace(base: u64, slot: usize) -> Result<u64> {
    const TAG: u64 = 1_u64 << 63;
    const SLOT_BITS: u32 = 8;
    let slot = u64::try_from(slot + 1)
        .map_err(|_| InferError::Metal("slot pipeline hors u64".to_string()))?;
    if slot >= (1_u64 << SLOT_BITS) || base > ((TAG - 1) >> SLOT_BITS) {
        return Err(InferError::Metal(
            "namespace scratch pipeline hors plage".to_string(),
        ));
    }
    Ok(TAG | (base << SLOT_BITS) | slot)
}

#[cfg(all(target_os = "macos", feature = "metal"))]
pub(in crate::decoder) fn resident_sample_spec(
    options: &GenerationOptions,
    sampler: &DeterministicSampler,
) -> Result<ResidentSampleSpec> {
    if options.temperature <= f32::EPSILON {
        return Err(InferError::Config(
            "sampling résident appelé en greedy".to_string(),
        ));
    }
    if !resident_sampling_on_device(options) {
        return Err(InferError::Config(format!(
            "sampling résident top_k={} invalide (max={})",
            options.top_k,
            crate::metal_backend::MAX_SAMPLER_TOP_K
        )));
    }
    Ok(ResidentSampleSpec {
        temperature: options.temperature,
        top_p: options.top_p,
        top_k: options.top_k,
        rng_state: sampler.state(),
        mode: ResidentSampleMode::OnDevice,
    })
}

#[cfg(all(target_os = "macos", feature = "metal"))]
pub(in crate::decoder) fn resident_sampling_on_device(options: &GenerationOptions) -> bool {
    if !resident_sampling_params_valid(options) || !crate::decoder::flags::gpu_sampler_enabled() {
        return false;
    }
    (options.top_k == 0 && options.top_p >= 1.0)
        || (options.top_k > 0 && options.top_k <= crate::metal_backend::MAX_SAMPLER_TOP_K)
}

#[cfg(all(target_os = "macos", feature = "metal"))]
pub(in crate::decoder) fn resident_sampling_supported(options: &GenerationOptions) -> bool {
    resident_sampling_params_valid(options) && crate::decoder::flags::gpu_sampler_enabled()
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn resident_sampling_params_valid(options: &GenerationOptions) -> bool {
    options.temperature > f32::EPSILON
        && options.temperature.is_finite()
        && options.top_p.is_finite()
        && options.top_p > 0.0
        && options.top_p <= 1.0
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn resident_sample_spec_for_solo(
    options: &GenerationOptions,
    sampler: &DeterministicSampler,
) -> Result<ResidentSampleSpec> {
    if resident_sampling_on_device(options) {
        return resident_sample_spec(options, sampler);
    }
    if !resident_sampling_supported(options) {
        return Err(InferError::Config(
            "sampling résident paramètres invalides".to_string(),
        ));
    }
    Ok(ResidentSampleSpec {
        temperature: options.temperature,
        top_p: options.top_p,
        top_k: options.top_k,
        rng_state: sampler.state(),
        mode: ResidentSampleMode::Readback,
    })
}
