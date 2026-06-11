//! Decode résident Metal complet et préparation des arènes.

use super::*;
#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::metal_backend::MetalExecutor;

#[cfg(all(target_os = "macos", feature = "metal"))]
const RESIDENT_PIPELINE_WINDOW: usize = 4;

#[cfg(all(target_os = "macos", feature = "metal"))]
enum ResidentDecodeInput<'a> {
    CpuToken(usize),
    GpuIndex(&'a metal::BufferRef),
}

#[cfg(all(target_os = "macos", feature = "metal"))]
struct ResidentDecodeInflight {
    command_buffer: metal::CommandBuffer,
    index: metal::Buffer,
    _owned: Vec<metal::Buffer>,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[derive(Clone, Copy)]
struct ResidentSampleSpec {
    temperature: f32,
    top_p: f32,
    top_k: usize,
    rng_state: u64,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
pub(super) struct ResidentPipelineOutput {
    pub(super) tokens: Vec<usize>,
    pub(super) decode: Duration,
    pub(super) decode_tokens: usize,
}

impl CausalDecoder {
    /// Renvoie `true` si le decode résident COMPLET (1c) est applicable : un
    /// executor Metal, des dimensions GQA valides, un lm_head biasless (argmax
    /// GPU), et TOUTES les couches encodables en résident
    /// ([`DecoderLayer::supports_resident_full`]).
    ///
    /// Validation EN AMONT (réserve Codex MAJEUR 6) : le forward résident est
    /// tout-ou-rien — soit le command buffer unique est entièrement GPU, soit on
    /// retombe sur le per-op AVANT de commencer, jamais un readback CPU au milieu.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(super) fn supports_resident_full_decode(&self) -> bool {
        if self.forward_runtime().metal_executor().is_none() {
            return false;
        }
        if self.config.head_dim.is_none() {
            return false;
        }
        let q_heads = self.config.num_attention_heads;
        let kv_heads = self.config.num_key_value_heads;
        if q_heads == 0 || kv_heads == 0 || q_heads % kv_heads != 0 {
            return false;
        }
        if self.lm_head.bias().is_some() {
            return false;
        }
        // Le chemin full-attn résident fusionne TOUJOURS norm+RoPE à la position
        // (`encode_rms_norm_rope_decode`) → exiger rope_theta présent (sinon le
        // per-op ferait rms_norm sans RoPE, divergence).
        if self.config.rope_theta.is_none() {
            return false;
        }
        self.layers.iter().all(DecoderLayer::supports_resident_full)
    }

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
    pub(super) fn decode_token_resident(
        &self,
        cache: &mut CausalDecoderCache,
        token_id: usize,
    ) -> Result<usize> {
        let inflight = self.enqueue_decode_token_resident(
            cache,
            ResidentDecodeInput::CpuToken(token_id),
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
    pub(super) fn decode_token_resident_sampled(
        &self,
        cache: &mut CausalDecoderCache,
        token_id: usize,
        options: &GenerationOptions,
        sampler: &mut DeterministicSampler,
    ) -> Result<usize> {
        let sample = resident_sample_spec(options, sampler)?;
        let inflight = self.enqueue_decode_token_resident(
            cache,
            ResidentDecodeInput::CpuToken(token_id),
            None,
            Some(sample),
        )?;
        crate::metal_backend::wait_for_completion(&inflight.command_buffer)?;
        sampler.advance();
        let raw = crate::metal_backend::read_u32_buffer(&inflight.index, 1)?
            .into_iter()
            .next()
            .ok_or_else(|| InferError::Metal("decode résident sans token".to_string()))?;
        usize::try_from(raw)
            .map_err(|_| InferError::Metal(format!("token résident hors usize: {raw}")))
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(super) fn decode_tokens_resident_pipelined(
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
                let step =
                    self.enqueue_decode_token_resident(cache, input, Some(&output_index), sample)?;
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
        sample: Option<ResidentSampleSpec>,
    ) -> Result<ResidentDecodeInflight> {
        let Some(metal) = self.forward_runtime().metal_executor() else {
            return Err(InferError::Metal(
                "decode résident sans executor Metal".to_string(),
            ));
        };
        let head_dim = self.config.head_dim.ok_or_else(|| {
            InferError::Dimension("head_dim manquant (decode résident)".to_string())
        })?;
        let theta = self.config.rope_theta.ok_or_else(|| {
            InferError::Config("rope_theta manquant (decode résident)".to_string())
        })?;
        let eps = self.config.rms_eps;
        let rope_dims = self.config.rope_dims.unwrap_or(head_dim);
        let q_heads = self.config.num_attention_heads;
        let kv_heads = self.config.num_key_value_heads;

        let hidden = self.final_norm.data().len();
        let position = cache.position;
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
        let mut encoder = new_resident_compute_encoder(command_buffer);
        let mut owned: Vec<metal::Buffer> = Vec::new();

        match input {
            ResidentDecodeInput::CpuToken(token_id) => {
                // Embed gather (CPU) — input upload, PAS un readback.
                let embed = embed_weight_tokens(&self.embed_tokens, &[token_id])?;
                let (_, embed_hidden) = embed.as_matrix()?;
                if embed_hidden != hidden {
                    return Err(InferError::Dimension(format!(
                        "embedding hidden={embed_hidden}, attendu {hidden}"
                    )));
                }
                arena.state.upload(&arena.hidden_a, embed.data())?;
            }
            ResidentDecodeInput::GpuIndex(index_buffer) => {
                metal.encode_embedding_from_index_buffers(
                    encoder,
                    &arena.embed_tokens,
                    index_buffer,
                    arena.hidden_a.buffer(),
                    hidden,
                )?;
            }
        }

        let mut current = &arena.hidden_a;
        let mut other = &arena.hidden_b;
        for (index, layer) in self.layers.iter().enumerate() {
            let layer_cache = &mut layers[index];
            let is_full = self.config.is_full_attention_layer(index);
            if is_full {
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
        if let Some(sample) = sample {
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
        drop(final_normed);
        *cache_position += 1;
        Ok(ResidentDecodeInflight {
            command_buffer: command_buffer.to_owned(),
            index: output_index
                .map(|buffer| buffer.to_owned())
                .unwrap_or_else(|| arena.index.buffer().clone()),
            _owned: owned,
        })
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(super) fn next_final_states_resident_verify(
        &self,
        cache: &mut CausalDecoderCache,
        token_ids: &[usize],
        emit_argmax: bool,
        capture_linear: bool,
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
            let embed = embed_weight_tokens(&self.embed_tokens, token_ids)?;
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
            let linear_captures = if capture_linear {
                let linear_states = layers
                    .iter()
                    .map(|layer| layer.linear.metal_state())
                    .collect::<Vec<_>>();
                Some(metal.allocate_linear_attn_state_captures(&linear_states, batch)?)
            } else {
                None
            };
            let command_buffer = arena.state.queue().new_command_buffer();
            let _barrier_guard = crate::metal_backend::resident_concurrent_enabled()
                .then(crate::metal_backend::install_dispatch_barrier_scope);
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
                    self.encode_resident_linear_dense_layer_rows(
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
                        linear_captures
                            .as_ref()
                            .and_then(|captures| captures.get(index))
                            .and_then(|captures| captures.as_deref()),
                    )?;
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
            return Ok(Some(ResidentVerifyOutput {
                states: Tensor::from_vec(vec![batch, hidden], output)?,
                tokens,
                captures: linear_captures.map(|linear| ResidentVerifyCaptures {
                    base_position,
                    linear,
                }),
            }));
        }

        let mut token_inputs = Vec::with_capacity(token_ids.len());
        for token_id in token_ids {
            let embed = embed_weight_tokens(&self.embed_tokens, &[*token_id])?;
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
        let final_normed = arena.state.scratch().lease(hidden, GpuElement::F32)?;
        let command_buffer = arena.state.queue().new_command_buffer();
        let _barrier_guard = crate::metal_backend::resident_concurrent_enabled()
            .then(crate::metal_backend::install_dispatch_barrier_scope);
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
        Ok(Some(ResidentVerifyOutput {
            states: Tensor::from_vec(vec![token_ids.len(), hidden], output)?,
            tokens,
            captures: None,
        }))
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(super) fn start_mtp_resident_draft(
        &self,
        cache: &mut CausalDecoderCache,
        final_state: &Tensor,
    ) -> Result<bool> {
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
        mtp.kv.truncate(0)?;
        mtp.current_is_a = true;
        Ok(true)
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(super) fn next_mtp_draft_resident(
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
            &arena.lm_head,
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
    pub(super) fn next_mtp_drafts_resident(
        &self,
        cache: &mut CausalDecoderCache,
        first_token_id: usize,
        max_draft_tokens: usize,
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
                output_hidden,
                1,
                hidden,
                eps,
            )?;
            metal.encode_lm_head_argmax_buffers_with_index_offset(
                encoder_guard.encoder(),
                &mut owned,
                output_hidden,
                &arena.lm_head,
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
    #[expect(
        clippy::too_many_arguments,
        reason = "routage résident: état couche + arena + ping-pong + encoder"
    )]
    fn encode_resident_full_layer(
        &self,
        metal: &MetalExecutor,
        arena: &ResidentArena,
        layer_cache: &mut LayerKvCache,
        layer: &DecoderLayer,
        index: usize,
        encoder: &metal::ComputeCommandEncoderRef,
        owned: &mut Vec<metal::Buffer>,
        dims: FullAttnLayerDims,
        layer_in: &metal::BufferRef,
        layer_out: &metal::BufferRef,
    ) -> Result<()> {
        let AttentionBlock::Full(attention) = &layer.attention else {
            return Err(InferError::Config(
                "couche full-attn attendue (decode résident)".to_string(),
            ));
        };
        if attention.q_norm.is_none() || attention.k_norm.is_none() {
            return Err(InferError::Config(
                "q_norm/k_norm manquant (decode résident)".to_string(),
            ));
        }
        if layer.post_attention_norm.is_none() {
            return Err(InferError::Config(
                "post_norm manquant (decode résident)".to_string(),
            ));
        }
        let kv = layer_cache.full.as_mut().ok_or_else(|| {
            InferError::Metal("KV full-attn résident absent (decode résident)".to_string())
        })?;
        match layer.mlp.as_ref() {
            Some(FeedForward::Moe(_)) => {
                match arena.layers.get(index).ok_or_else(|| {
                    InferError::Config(format!("poids résidents couche {index} absents"))
                })? {
                    ResidentLayerBuffers::FullMoe(resolved) => {
                        let weights = FullAttnLayerWeights {
                            input_norm: &resolved.input_norm,
                            qkv_proj: &resolved.qkv_proj,
                            o_proj: &resolved.o_proj,
                            q_norm: &resolved.q_norm,
                            k_norm: &resolved.k_norm,
                            post_norm: &resolved.post_norm,
                            moe: &resolved.moe,
                            top_k: resolved.top_k,
                        };
                        arena.state.encode_full_attn_layer(
                            metal, encoder, owned, kv, weights, dims, layer_in, layer_out,
                        )
                    }
                    ResidentLayerBuffers::FullRouted(resolved) => {
                        let weights = FullAttnRoutedLayerWeights {
                            input_norm: &resolved.input_norm,
                            qkv_proj: &resolved.qkv_proj,
                            o_proj: &resolved.o_proj,
                            q_norm: &resolved.q_norm,
                            k_norm: &resolved.k_norm,
                            post_norm: &resolved.post_norm,
                            moe: &resolved.moe,
                            top_k: resolved.top_k,
                        };
                        arena.state.encode_full_attn_routed_layer(
                            metal, encoder, owned, kv, weights, dims, layer_in, layer_out,
                        )
                    }
                    _ => Err(InferError::Config(format!(
                        "poids full-attn MoE résidents absents couche {index}"
                    ))),
                }
            }
            Some(FeedForward::Dense(_)) => {
                let ResidentLayerBuffers::FullDense(resolved) =
                    arena.layers.get(index).ok_or_else(|| {
                        InferError::Config(format!("poids résidents couche {index} absents"))
                    })?
                else {
                    return Err(InferError::Config(format!(
                        "poids full-attn dense résidents absents couche {index}"
                    )));
                };
                let weights = FullAttnDenseLayerWeights {
                    input_norm: &resolved.input_norm,
                    qkv_proj: resolved.qkv_proj.as_ref(),
                    q_proj: &resolved.q_proj,
                    k_proj: &resolved.k_proj,
                    v_proj: &resolved.v_proj,
                    o_proj: &resolved.o_proj,
                    q_norm: &resolved.q_norm,
                    k_norm: &resolved.k_norm,
                    post_norm: &resolved.post_norm,
                    gate_proj: &resolved.gate_proj,
                    up_proj: &resolved.up_proj,
                    down_proj: &resolved.down_proj,
                    tail_score: &arena.dense_tail_score,
                };
                arena.state.encode_full_attn_dense_layer(
                    metal, encoder, owned, kv, weights, dims, layer_in, layer_out,
                )
            }
            None => Err(InferError::Config(
                "MLP attendu (decode résident)".to_string(),
            )),
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[expect(
        clippy::too_many_arguments,
        reason = "routage résident: état couche + arena + ping-pong + encoder"
    )]
    fn encode_resident_linear_layer(
        &self,
        metal: &MetalExecutor,
        arena: &ResidentArena,
        layer_cache: &mut LayerKvCache,
        layer: &DecoderLayer,
        index: usize,
        encoder: &metal::ComputeCommandEncoderRef,
        owned: &mut Vec<metal::Buffer>,
        la_spec: LinearAttentionStepSpec,
        res_dims: LinearAttnResidentDims,
        hidden: usize,
        eps: f32,
        layer_in: &metal::BufferRef,
        layer_out: &metal::BufferRef,
    ) -> Result<()> {
        let AttentionBlock::Linear(_) = &layer.attention else {
            return Err(InferError::Config(
                "couche linear-attn attendue (decode résident)".to_string(),
            ));
        };
        if layer.post_attention_norm.is_none() {
            return Err(InferError::Config(
                "post_norm manquant (decode résident)".to_string(),
            ));
        }
        let state = layer_cache.linear.metal_state().ok_or_else(|| {
            InferError::Metal("état linear-attn résident absent (decode résident)".to_string())
        })?;
        match layer.mlp.as_ref() {
            Some(FeedForward::Moe(_)) => {
                let ResidentLayerBuffers::LinearMoe(resolved) =
                    arena.layers.get(index).ok_or_else(|| {
                        InferError::Config(format!("poids résidents couche {index} absents"))
                    })?
                else {
                    return Err(InferError::Config(format!(
                        "poids linear-attn MoE résidents absents couche {index}"
                    )));
                };
                let weights = LinearAttnLayerWeights {
                    input_norm: &resolved.input_norm,
                    linear: &resolved.linear,
                    post_norm: &resolved.post_norm,
                    moe: &resolved.moe,
                    top_k: resolved.top_k,
                };
                arena.state.encode_linear_attn_layer(
                    metal, encoder, owned, state, weights, la_spec, res_dims, hidden, eps,
                    layer_in, layer_out,
                )
            }
            Some(FeedForward::Dense(_)) => {
                let ResidentLayerBuffers::LinearDense(resolved) =
                    arena.layers.get(index).ok_or_else(|| {
                        InferError::Config(format!("poids résidents couche {index} absents"))
                    })?
                else {
                    return Err(InferError::Config(format!(
                        "poids linear-attn dense résidents absents couche {index}"
                    )));
                };
                let weights = LinearAttnDenseLayerWeights {
                    input_norm: &resolved.input_norm,
                    linear: &resolved.linear,
                    post_norm: &resolved.post_norm,
                    gate_proj: &resolved.gate_proj,
                    up_proj: &resolved.up_proj,
                    down_proj: &resolved.down_proj,
                    tail_score: &arena.dense_tail_score,
                };
                arena.state.encode_linear_attn_dense_layer(
                    metal, encoder, owned, state, weights, la_spec, res_dims, hidden, eps,
                    layer_in, layer_out,
                )
            }
            None => Err(InferError::Config(
                "MLP attendu (decode résident)".to_string(),
            )),
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[allow(
        dead_code,
        reason = "prototype full-attn batché gardé hors hot path après mesure défavorable"
    )]
    #[expect(
        clippy::too_many_arguments,
        reason = "routage full-attn résident batché: état couche + arena + ping-pong + encoder"
    )]
    fn encode_resident_full_dense_layer_rows(
        &self,
        metal: &MetalExecutor,
        arena: &ResidentArena,
        layer_cache: &mut LayerKvCache,
        layer: &DecoderLayer,
        index: usize,
        encoder: &metal::ComputeCommandEncoderRef,
        owned: &mut Vec<metal::Buffer>,
        dims: FullAttnLayerDims,
        rows: usize,
        layer_in: &metal::BufferRef,
        layer_out: &metal::BufferRef,
    ) -> Result<()> {
        let AttentionBlock::Full(attention) = &layer.attention else {
            return Err(InferError::Config(
                "couche full-attn attendue (verify résident batché)".to_string(),
            ));
        };
        if attention.q_norm.is_none() || attention.k_norm.is_none() {
            return Err(InferError::Config(
                "q_norm/k_norm manquant (verify résident batché)".to_string(),
            ));
        }
        if !matches!(layer.mlp.as_ref(), Some(FeedForward::Dense(_))) {
            return Err(InferError::Config(
                "MLP dense attendu (verify full résident batché)".to_string(),
            ));
        }
        if layer.post_attention_norm.is_none() {
            return Err(InferError::Config(
                "post_norm manquant (verify full résident batché)".to_string(),
            ));
        }
        let kv = layer_cache.full.as_mut().ok_or_else(|| {
            InferError::Metal("KV full-attn résident absent (verify batché)".to_string())
        })?;
        let ResidentLayerBuffers::FullDense(resolved) = arena
            .layers
            .get(index)
            .ok_or_else(|| InferError::Config(format!("poids résidents couche {index} absents")))?
        else {
            return Err(InferError::Config(format!(
                "poids full-attn dense résidents absents couche {index}"
            )));
        };
        let weights = FullAttnDenseLayerWeights {
            input_norm: &resolved.input_norm,
            qkv_proj: resolved.qkv_proj.as_ref(),
            q_proj: &resolved.q_proj,
            k_proj: &resolved.k_proj,
            v_proj: &resolved.v_proj,
            o_proj: &resolved.o_proj,
            q_norm: &resolved.q_norm,
            k_norm: &resolved.k_norm,
            post_norm: &resolved.post_norm,
            gate_proj: &resolved.gate_proj,
            up_proj: &resolved.up_proj,
            down_proj: &resolved.down_proj,
            tail_score: &arena.dense_tail_score,
        };
        arena.state.encode_full_attn_dense_layer_rows(
            metal, encoder, owned, kv, weights, dims, rows, layer_in, layer_out,
        )
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[expect(
        clippy::too_many_arguments,
        reason = "routage résident batché: état couche + arena + ping-pong + encoder"
    )]
    fn encode_resident_linear_dense_layer_rows(
        &self,
        metal: &MetalExecutor,
        arena: &ResidentArena,
        layer_cache: &mut LayerKvCache,
        layer: &DecoderLayer,
        index: usize,
        encoder: &metal::ComputeCommandEncoderRef,
        owned: &mut Vec<metal::Buffer>,
        la_spec: LinearAttentionStepSpec,
        res_dims: LinearAttnResidentDims,
        rows: usize,
        hidden: usize,
        eps: f32,
        layer_in: &metal::BufferRef,
        layer_out: &metal::BufferRef,
        captures: Option<&[LinearAttentionMetalState]>,
    ) -> Result<()> {
        let AttentionBlock::Linear(_) = &layer.attention else {
            return Err(InferError::Config(
                "couche linear-attn attendue (verify résident batché)".to_string(),
            ));
        };
        if !matches!(layer.mlp.as_ref(), Some(FeedForward::Dense(_))) {
            return Err(InferError::Config(
                "MLP dense attendu (verify résident batché)".to_string(),
            ));
        }
        if layer.post_attention_norm.is_none() {
            return Err(InferError::Config(
                "post_norm manquant (verify résident batché)".to_string(),
            ));
        }
        let state = layer_cache.linear.metal_state().ok_or_else(|| {
            InferError::Metal("état linear-attn résident absent (verify batché)".to_string())
        })?;
        let ResidentLayerBuffers::LinearDense(resolved) = arena
            .layers
            .get(index)
            .ok_or_else(|| InferError::Config(format!("poids résidents couche {index} absents")))?
        else {
            return Err(InferError::Config(format!(
                "poids linear-attn dense résidents absents couche {index}"
            )));
        };
        let weights = LinearAttnDenseLayerWeights {
            input_norm: &resolved.input_norm,
            linear: &resolved.linear,
            post_norm: &resolved.post_norm,
            gate_proj: &resolved.gate_proj,
            up_proj: &resolved.up_proj,
            down_proj: &resolved.down_proj,
            tail_score: &arena.dense_tail_score,
        };
        arena.state.encode_linear_attn_dense_layer_rows(
            metal, encoder, owned, state, weights, la_spec, res_dims, rows, hidden, eps, layer_in,
            layer_out, captures,
        )
    }

    /// Microbench (tranche 3, `RETI_RUST_GPU_COUNTERS`) isolant le **MoE** et le
    /// **surcoût commit/wait par command buffer** — les deux inconnues que la
    /// segmentation per-couche ne sépare pas (chaque couche = attn + MoE + 1 CB).
    ///
    /// `overhead` = K command buffers VIDES (commit/wait seul). `moe` = K MoE
    /// résidents (couche 0) via [`MetalExecutor::moe_gated_router_topk_shared`]
    /// (1 CB chacun). `MoE pur ≈ moe − overhead` ; permet de retrancher le MoE des
    /// temps de couche pour isoler le mécanisme d'attention. Mesure pure (aucun
    /// effet sur la génération). `None` si le modèle/backend ne s'y prête pas.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(super) fn profile_moe_and_overhead(
        &self,
        queue: &metal::CommandQueueRef,
    ) -> Option<String> {
        let metal = self.forward_runtime().metal_executor()?;
        let Some(FeedForward::Moe(mlp)) = self.layers.first()?.mlp.as_ref() else {
            return None;
        };
        let (router, experts, top_k, shared_expert, shared_gate) = mlp.shared_metal_parts()?;
        // Entrées VARIÉES (embeddings de tokens distincts) → le routeur sélectionne
        // des experts différents à chaque itération → lectures de poids FROIDES
        // (réalistes) au lieu d'un seul jeu d'experts cache-résident.
        let inputs: Vec<Tensor> = (0..64_usize)
            .filter_map(|token| embed_weight_tokens(&self.embed_tokens, &[token]).ok())
            .collect();
        if inputs.is_empty() {
            return None;
        }
        let iters = 256_u32;
        let warmup = 32_u32;
        let empty_cb = || -> Result<()> {
            let command_buffer = queue.new_command_buffer();
            command_buffer.new_compute_command_encoder().end_encoding();
            crate::metal_backend::commit_and_wait(command_buffer)
        };
        for _ in 0..warmup {
            empty_cb().ok()?;
        }
        let overhead_started = Instant::now();
        for _ in 0..iters {
            empty_cb().ok()?;
        }
        let overhead_ms = overhead_started.elapsed().as_secs_f64() * 1000.0 / f64::from(iters);
        let run_moe = |step: usize| {
            metal.moe_gated_router_topk_shared(
                &inputs[step % inputs.len()],
                router,
                experts,
                top_k,
                shared_expert,
                shared_gate,
            )
        };
        for step in 0..warmup as usize {
            run_moe(step).ok()?;
        }
        let moe_started = Instant::now();
        for step in 0..iters as usize {
            run_moe(step).ok()?;
        }
        let moe_ms = moe_started.elapsed().as_secs_f64() * 1000.0 / f64::from(iters);
        let moe_pure = (moe_ms - overhead_ms).max(0.0);
        let layers = self.layers.len() as f64;
        Some(format!(
            "gpu microbench (couche 0, {iters} itér) : overhead/CB {overhead_ms:.3} ms | \
             MoE+CB {moe_ms:.3} ms | MoE pur ≈ {moe_pure:.3} ms/couche → \
             MoE total ≈ {total:.3} ms/token (×{n})",
            total = moe_pure * layers,
            n = self.layers.len(),
        ))
    }

    /// Dispatch d'un pas de decode per-op (qui prend lui-même le chemin résident
    /// full-attn 1b si `cache.full` est présent). Le chemin résident COMPLET (1c)
    /// est dispatché séparément dans la boucle de génération (renvoie le token).
    ///
    /// # Errors
    ///
    /// Propage les erreurs du forward.
    pub(super) fn next_decode_state(
        &self,
        cache: &mut CausalDecoderCache,
        token_id: usize,
    ) -> Result<Tensor> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if let Some(output) =
            self.next_final_states_resident_verify(cache, &[token_id], false, false)?
        {
            return Tensor::row(output.states.row_slice(0)?.to_vec());
        }
        self.next_final_state_cached(cache, token_id)
    }

    /// Prépare le decode full-attn résident : alloue les buffers KV GPU des
    /// couches full-attn et les seed depuis le K/V (rope'd) du prefill (CPU,
    /// `keys`/`values`). Une couche sans KV prefill cohérent reste sur le chemin
    /// CPU (`full` laissé à `None`) → l'activation est sûre et incrémentale.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si la compilation du kernel résident, une allocation ou
    /// le seed échoue.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(super) fn setup_resident_decode(
        &self,
        cache: &mut CausalDecoderCache,
        max_new_tokens: usize,
    ) -> Result<()> {
        let Some(metal) = self.forward_runtime().metal_executor() else {
            return Ok(());
        };
        let Some(head_dim) = self.config.head_dim else {
            return Ok(());
        };
        let q_heads = self.config.num_attention_heads;
        let kv_heads = self.config.num_key_value_heads;
        if q_heads == 0 || kv_heads == 0 || head_dim == 0 || q_heads % kv_heads != 0 {
            return Ok(());
        }
        let kv_dim = kv_heads * head_dim;
        let prefill_len = cache.position;
        let capacity = prefill_len
            .checked_add(max_new_tokens)
            .ok_or_else(|| InferError::Dimension("capacité KV résidente déborde".to_string()))?;
        if capacity == 0 {
            return Ok(());
        }
        let arena = DecodeResidentState::new(metal.device().clone())?;
        for (index, layer_cache) in cache.layers.iter_mut().enumerate() {
            if !self.config.is_full_attention_layer(index) {
                continue;
            }
            // Seed depuis le K/V (rope'd) du prefill ; si absent/incohérent, la
            // couche reste sur le chemin CPU (oracle), pas d'erreur.
            if layer_cache.keys.len() != prefill_len * kv_dim
                || layer_cache.values.len() != prefill_len * kv_dim
            {
                continue;
            }
            let mut full = arena.full_attention(capacity, q_heads, kv_heads, head_dim)?;
            full.seed(&layer_cache.keys, &layer_cache.values, prefill_len)?;
            layer_cache.full = Some(full);
        }
        Ok(())
    }

    /// Prépare le decode résident COMPLET (1c) : valide que TOUS les états par
    /// couche sont prêts (KV full-attn seedable depuis le prefill, état conv/ssm
    /// linear-attn peuplé par le prefill résident per-op), alloue l'arène + le
    /// ping-pong hidden + le buffer `u32` du token, puis seed les KV full-attn.
    ///
    /// Renvoie `Ok(false)` (tout-ou-rien, MAJEUR 6) si une précondition manque
    /// → l'appelant retombe sur le per-op SANS mutation laissée (validation AVANT
    /// seeding) ; `Ok(true)` si l'arène résidente est prête dans `cache.resident`.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si la compilation des kernels, une allocation ou un seed
    /// échoue.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(super) fn setup_resident_full_decode(
        &self,
        cache: &mut CausalDecoderCache,
        max_new_tokens: usize,
    ) -> Result<bool> {
        let Some(metal) = self.forward_runtime().metal_executor() else {
            return Ok(false);
        };
        let Some(head_dim) = self.config.head_dim else {
            return Ok(false);
        };
        let q_heads = self.config.num_attention_heads;
        let kv_heads = self.config.num_key_value_heads;
        if q_heads == 0 || kv_heads == 0 || head_dim == 0 || q_heads % kv_heads != 0 {
            return Ok(false);
        }
        let hidden = self.final_norm.data().len();
        if hidden == 0 {
            return Ok(false);
        }
        let kv_dim = kv_heads * head_dim;
        let prefill_len = cache.position;
        let capacity = prefill_len
            .checked_add(max_new_tokens)
            .ok_or_else(|| InferError::Dimension("capacité KV résidente déborde".to_string()))?;
        if capacity == 0 {
            return Ok(false);
        }
        // Validation AVANT toute mutation (tout-ou-rien) : KV full-attn seedable et
        // état conv/ssm linear-attn présent (peuplé par le prefill résident per-op).
        for (index, layer_cache) in cache.layers.iter().enumerate() {
            if self.config.is_full_attention_layer(index) {
                if layer_cache.keys.len() != prefill_len * kv_dim
                    || layer_cache.values.len() != prefill_len * kv_dim
                {
                    return Ok(false);
                }
            } else if layer_cache.linear.metal_state().is_none() {
                return Ok(false);
            }
        }
        let mut layer_buffers = Vec::with_capacity(self.layers.len());
        for (index, layer) in self.layers.iter().enumerate() {
            if self.config.is_full_attention_layer(index) {
                let AttentionBlock::Full(attention) = &layer.attention else {
                    return Ok(false);
                };
                let (Some(q_norm), Some(k_norm)) =
                    (attention.q_norm.as_ref(), attention.k_norm.as_ref())
                else {
                    return Ok(false);
                };
                let Some(post_norm) = layer.post_attention_norm.as_ref() else {
                    return Ok(false);
                };
                match layer.mlp.as_ref() {
                    Some(FeedForward::Moe(mlp)) => {
                        let input_norm = metal.cached_buffer_from_f32(
                            layer.input_norm.data(),
                            "resident_full_input_norm",
                        )?;
                        let qkv_proj = metal.resolve_concat_linear_weight_buffers(
                            &[
                                attention.q_proj.weight(),
                                attention.k_proj.weight(),
                                attention.v_proj.weight(),
                            ],
                            "resident_full_qkv_proj",
                        )?;
                        let o_proj = metal.resolve_linear_weight_buffers(
                            attention.o_proj.weight(),
                            "resident_full_o_proj",
                        )?;
                        let q_norm =
                            metal.cached_buffer_from_f32(q_norm.data(), "resident_full_q_norm")?;
                        let k_norm =
                            metal.cached_buffer_from_f32(k_norm.data(), "resident_full_k_norm")?;
                        let post_norm = metal
                            .cached_buffer_from_f32(post_norm.data(), "resident_full_post_norm")?;
                        if let Some((router, experts, top_k, shared_expert, shared_gate)) =
                            mlp.shared_metal_parts()
                        {
                            layer_buffers.push(ResidentLayerBuffers::FullMoe(
                                ResidentFullMoeBuffers {
                                    input_norm,
                                    qkv_proj,
                                    o_proj,
                                    q_norm,
                                    k_norm,
                                    post_norm,
                                    moe: metal.resolve_moe_shared_weights(
                                        router,
                                        experts,
                                        shared_expert,
                                        shared_gate,
                                    )?,
                                    top_k,
                                },
                            ));
                        } else if let Some((router, experts, top_k)) = mlp.metal_parts() {
                            layer_buffers.push(ResidentLayerBuffers::FullRouted(
                                ResidentFullRoutedBuffers {
                                    input_norm,
                                    qkv_proj,
                                    o_proj,
                                    q_norm,
                                    k_norm,
                                    post_norm,
                                    moe: metal.resolve_moe_routed_weights(router, experts)?,
                                    top_k,
                                },
                            ));
                        } else {
                            return Ok(false);
                        }
                    }
                    Some(FeedForward::Dense(mlp)) => {
                        let (gate_proj, up_proj, down_proj) = mlp.projections();
                        let qkv_proj = match metal.resolve_concat_linear_weight_buffers(
                            &[
                                attention.q_proj.weight(),
                                attention.k_proj.weight(),
                                attention.v_proj.weight(),
                            ],
                            "resident_dense_full_qkv_proj",
                        ) {
                            Ok(weights) => Some(weights),
                            Err(InferError::Dimension(_)) => None,
                            Err(error) => return Err(error),
                        };
                        layer_buffers.push(ResidentLayerBuffers::FullDense(
                            ResidentFullDenseBuffers {
                                input_norm: metal.cached_buffer_from_f32(
                                    layer.input_norm.data(),
                                    "resident_dense_full_input_norm",
                                )?,
                                qkv_proj,
                                q_proj: metal.resolve_linear_weight_buffers(
                                    attention.q_proj.weight(),
                                    "resident_dense_full_q_proj",
                                )?,
                                k_proj: metal.resolve_linear_weight_buffers(
                                    attention.k_proj.weight(),
                                    "resident_dense_full_k_proj",
                                )?,
                                v_proj: metal.resolve_linear_weight_buffers(
                                    attention.v_proj.weight(),
                                    "resident_dense_full_v_proj",
                                )?,
                                o_proj: metal.resolve_linear_weight_buffers(
                                    attention.o_proj.weight(),
                                    "resident_dense_full_o_proj",
                                )?,
                                q_norm: metal.cached_buffer_from_f32(
                                    q_norm.data(),
                                    "resident_dense_full_q_norm",
                                )?,
                                k_norm: metal.cached_buffer_from_f32(
                                    k_norm.data(),
                                    "resident_dense_full_k_norm",
                                )?,
                                post_norm: metal.cached_buffer_from_f32(
                                    post_norm.data(),
                                    "resident_dense_full_post_norm",
                                )?,
                                gate_proj: metal.resolve_linear_weight_buffers(
                                    gate_proj.weight(),
                                    "resident_dense_gate_proj",
                                )?,
                                up_proj: metal.resolve_linear_weight_buffers(
                                    up_proj.weight(),
                                    "resident_dense_up_proj",
                                )?,
                                down_proj: metal.resolve_linear_weight_buffers(
                                    down_proj.weight(),
                                    "resident_dense_down_proj",
                                )?,
                            },
                        ));
                    }
                    _ => layer_buffers.push(ResidentLayerBuffers::Other),
                }
            } else {
                let AttentionBlock::Linear(linear) = &layer.attention else {
                    return Ok(false);
                };
                let Some(post_norm) = layer.post_attention_norm.as_ref() else {
                    return Ok(false);
                };
                match layer.mlp.as_ref() {
                    Some(FeedForward::Moe(mlp)) => {
                        let Some((router, experts, top_k, shared_expert, shared_gate)) =
                            mlp.shared_metal_parts()
                        else {
                            return Ok(false);
                        };
                        layer_buffers.push(ResidentLayerBuffers::LinearMoe(
                            ResidentLinearMoeBuffers {
                                input_norm: metal.cached_buffer_from_f32(
                                    layer.input_norm.data(),
                                    "resident_linear_input_norm",
                                )?,
                                linear: metal.resolve_linear_attn_resident_weights(
                                    linear.resident_weights(),
                                )?,
                                post_norm: metal.cached_buffer_from_f32(
                                    post_norm.data(),
                                    "resident_linear_post_norm",
                                )?,
                                moe: metal.resolve_moe_shared_weights(
                                    router,
                                    experts,
                                    shared_expert,
                                    shared_gate,
                                )?,
                                top_k,
                            },
                        ));
                    }
                    Some(FeedForward::Dense(mlp)) => {
                        let (gate_proj, up_proj, down_proj) = mlp.projections();
                        layer_buffers.push(ResidentLayerBuffers::LinearDense(
                            ResidentLinearDenseBuffers {
                                input_norm: metal.cached_buffer_from_f32(
                                    layer.input_norm.data(),
                                    "resident_dense_linear_input_norm",
                                )?,
                                linear: metal.resolve_linear_attn_resident_dense_weights(
                                    linear.resident_weights(),
                                )?,
                                post_norm: metal.cached_buffer_from_f32(
                                    post_norm.data(),
                                    "resident_dense_linear_post_norm",
                                )?,
                                gate_proj: metal.resolve_linear_weight_buffers(
                                    gate_proj.weight(),
                                    "resident_dense_gate_proj",
                                )?,
                                up_proj: metal.resolve_linear_weight_buffers(
                                    up_proj.weight(),
                                    "resident_dense_up_proj",
                                )?,
                                down_proj: metal.resolve_linear_weight_buffers(
                                    down_proj.weight(),
                                    "resident_dense_down_proj",
                                )?,
                            },
                        ));
                    }
                    _ => layer_buffers.push(ResidentLayerBuffers::Other),
                }
            }
        }
        let mut arena = DecodeResidentState::new(metal.device().clone())?;
        for (index, layer_cache) in cache.layers.iter_mut().enumerate() {
            if !self.config.is_full_attention_layer(index) {
                continue;
            }
            let mut full = arena.full_attention(capacity, q_heads, kv_heads, head_dim)?;
            full.seed(&layer_cache.keys, &layer_cache.values, prefill_len)?;
            layer_cache.full = Some(full);
        }
        let hidden_a = arena.persistent(hidden, GpuElement::F32)?;
        let hidden_b = arena.persistent(hidden, GpuElement::F32)?;
        let index = arena.persistent(1, GpuElement::U32)?;
        let mut index_ring = Vec::with_capacity(RESIDENT_PIPELINE_WINDOW);
        for _ in 0..RESIDENT_PIPELINE_WINDOW {
            index_ring.push(arena.persistent(1, GpuElement::U32)?);
        }
        let mtp = if let Some(head) = self.mtp.as_ref() {
            let mlp = match &head.layer.mlp {
                FeedForward::Dense(mlp) => mlp,
                _ => return Ok(false),
            };
            let (gate_proj, up_proj, down_proj) = mlp.projections();
            let q_norm =
                head.layer.attention.q_norm.as_ref().ok_or_else(|| {
                    InferError::Config("q_norm MTP manquant (résident)".to_string())
                })?;
            let k_norm =
                head.layer.attention.k_norm.as_ref().ok_or_else(|| {
                    InferError::Config("k_norm MTP manquant (résident)".to_string())
                })?;
            let qkv_proj = match metal.resolve_concat_linear_weight_buffers(
                &[
                    head.layer.attention.q_proj.weight(),
                    head.layer.attention.k_proj.weight(),
                    head.layer.attention.v_proj.weight(),
                ],
                "resident_mtp_qkv_proj",
            ) {
                Ok(weights) => Some(weights),
                Err(InferError::Dimension(_)) => None,
                Err(error) => return Err(error),
            };
            let kv = arena.full_attention(capacity, q_heads, kv_heads, head_dim)?;
            let hidden_a = arena.persistent(hidden, GpuElement::F32)?;
            let hidden_b = arena.persistent(hidden, GpuElement::F32)?;
            let index = arena.persistent(1, GpuElement::U32)?;
            let draft_indices = arena.persistent(max_new_tokens.max(1), GpuElement::U32)?;
            let embedding = arena.persistent(hidden, GpuElement::F32)?;
            let concat = arena.persistent(
                hidden.checked_mul(2).ok_or_else(|| {
                    InferError::Dimension("MTP concat hidden déborde".to_string())
                })?,
                GpuElement::F32,
            )?;
            let fc_out = arena.persistent(hidden, GpuElement::F32)?;
            Some(ResidentMtpArena {
                pre_fc_norm_embedding: metal.cached_buffer_from_f32(
                    head.pre_fc_norm_embedding.data(),
                    "resident_mtp_pre_fc_norm_embedding",
                )?,
                pre_fc_norm_hidden: metal.cached_buffer_from_f32(
                    head.pre_fc_norm_hidden.data(),
                    "resident_mtp_pre_fc_norm_hidden",
                )?,
                fc: metal.resolve_linear_weight_buffers(head.fc.weight(), "resident_mtp_fc")?,
                layer: ResidentFullDenseBuffers {
                    input_norm: metal.cached_buffer_from_f32(
                        head.layer.input_norm.data(),
                        "resident_mtp_input_norm",
                    )?,
                    qkv_proj,
                    q_proj: metal.resolve_linear_weight_buffers(
                        head.layer.attention.q_proj.weight(),
                        "resident_mtp_q_proj",
                    )?,
                    k_proj: metal.resolve_linear_weight_buffers(
                        head.layer.attention.k_proj.weight(),
                        "resident_mtp_k_proj",
                    )?,
                    v_proj: metal.resolve_linear_weight_buffers(
                        head.layer.attention.v_proj.weight(),
                        "resident_mtp_v_proj",
                    )?,
                    o_proj: metal.resolve_linear_weight_buffers(
                        head.layer.attention.o_proj.weight(),
                        "resident_mtp_o_proj",
                    )?,
                    q_norm: metal.cached_buffer_from_f32(q_norm.data(), "resident_mtp_q_norm")?,
                    k_norm: metal.cached_buffer_from_f32(k_norm.data(), "resident_mtp_k_norm")?,
                    post_norm: metal.cached_buffer_from_f32(
                        head.layer.post_attention_norm.data(),
                        "resident_mtp_post_norm",
                    )?,
                    gate_proj: metal
                        .resolve_linear_weight_buffers(gate_proj.weight(), "resident_mtp_gate")?,
                    up_proj: metal
                        .resolve_linear_weight_buffers(up_proj.weight(), "resident_mtp_up")?,
                    down_proj: metal
                        .resolve_linear_weight_buffers(down_proj.weight(), "resident_mtp_down")?,
                },
                norm: metal.cached_buffer_from_f32(head.norm.data(), "resident_mtp_norm")?,
                kv,
                hidden_a,
                hidden_b,
                current_is_a: true,
                index,
                draft_indices,
                embedding,
                concat,
                fc_out,
            })
        } else {
            None
        };
        cache.resident = Some(ResidentArena {
            state: arena,
            hidden_a,
            hidden_b,
            index,
            index_ring,
            layers: layer_buffers,
            dense_tail_score: metal.cached_buffer_from_f32(&[1.0], "resident_dense_tail_score")?,
            embed_tokens: metal.resolve_embedding_weight_buffers(&self.embed_tokens)?,
            final_norm: metal
                .cached_buffer_from_f32(self.final_norm.data(), "resident_final_norm")?,
            lm_head: metal
                .resolve_linear_weight_buffers(self.lm_head.weight(), "resident_lm_head")?,
            mtp,
        });
        Ok(true)
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn resident_sample_spec(
    options: &GenerationOptions,
    sampler: &DeterministicSampler,
) -> Result<ResidentSampleSpec> {
    if options.temperature <= f32::EPSILON {
        return Err(InferError::Config(
            "sampling résident appelé en greedy".to_string(),
        ));
    }
    if options.top_k == 0 || options.top_k > crate::metal_backend::MAX_SAMPLER_TOP_K {
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
    })
}
