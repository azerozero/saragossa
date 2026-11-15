//! Trame échantillonnée du code predictor Qwen3-TTS dans un command buffer.
//!
//! Le chemin pas à pas fait 16 allers-retours CPU↔GPU par trame : projection
//! `small_to_mtp` sur CPU, un command buffer par pas, relecture des logits et
//! tirage CPU. Ici les 16 pas (amorce `final_state`, puis cb0 et 14 codebooks)
//! s'enchaînent dans un seul command buffer : projection et biais, 5 couches,
//! norme finale, tête, tirage top-k/top-p GPU et gather de l'embedding du code
//! tiré, qui devient l'entrée du pas suivant. Le CPU relit 15 `u32` par trame.
//!
//! Arithmétique : couches et têtes sont les mêmes kernels que le chemin pas à
//! pas ; la projection passe du CPU au GPU (arrondi d'ordre de réduction) et
//! le tirage suit `tts_sampler`. Le glouton n'emprunte jamais ce chemin.

use super::super::*;
use crate::decode_resident::{TtsCodebookSample, TTS_SAMPLER_MAX_VOCAB, TTS_SAMPLER_TRACE_K};

/// Poids GPU d'une trame fusionnée, résolus une fois par génération.
pub(crate) struct TtsCodebookWeights {
    projection: MetalLinearWeightBuffers,
    projection_bias: Option<metal::Buffer>,
    heads: Vec<MetalLinearWeightBuffers>,
    embeddings: Vec<MetalEmbeddingWeightBuffers>,
    talker_hidden: usize,
    vocab: usize,
}

/// Distributions et logits relus pour l'oracle des tests.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "relu par l'oracle fused_codebooks_match_cpu_oracle"
    )
)]
pub(crate) struct TtsCodebookTrace {
    /// Logits de chaque tête, `[têtes, vocab]`.
    pub(crate) logits: Vec<f32>,
    /// Distribution retenue par tirage : (token, probabilité normalisée), par rang.
    pub(crate) distributions: Vec<Vec<(usize, f32)>>,
}

impl TtsCodebookWeights {
    /// Taille du vocabulaire de chaque tête.
    pub(crate) fn vocab(&self) -> usize {
        self.vocab
    }
}

impl CausalDecoder {
    /// Résout les poids du code predictor pour [`Self::decode_tts_codebooks_resident`].
    ///
    /// Renvoie `Ok(None)` sans executor Metal, sans arène résidente possible, si
    /// une couche n'est pas full-attn résidente ou si le vocabulaire dépasse le
    /// tri du kernel de tirage : l'appelant garde alors le chemin pas à pas.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si une forme est incohérente ou si un upload échoue.
    pub(crate) fn resolve_tts_codebook_weights(
        &self,
        projection: &Linear,
        heads: &[Linear],
        embeddings: &[Tensor],
    ) -> Result<Option<TtsCodebookWeights>> {
        let Some(metal) = self.forward_runtime().metal_executor() else {
            return Ok(None);
        };
        if !self.supports_resident_full_decode()
            || self.plan.layers.iter().any(|layer| !layer.resident_full)
            || heads.is_empty()
            || embeddings.len() + 1 < heads.len()
        {
            return Ok(None);
        }
        let projection_buffers =
            metal.resolve_linear_weight_buffers(projection.weight(), "tts_cp_projection")?;
        let hidden = self.final_norm.len();
        if metal.linear_weight_out_dim(&projection_buffers) != hidden {
            return Err(InferError::Dimension(
                "projection code predictor: sortie ≠ hidden".into(),
            ));
        }
        let [_, talker_hidden] = projection.weight().shape() else {
            return Err(InferError::Dimension(
                "projection code predictor rang ≠ 2".into(),
            ));
        };
        let projection_bias = projection
            .bias()
            .map(|bias| metal.cached_buffer_from_f32(bias.data(), "tts_cp_projection_bias"))
            .transpose()?;
        let heads = heads
            .iter()
            .map(|head| {
                if head.bias().is_some() {
                    return Err(InferError::Config(
                        "tête code predictor avec biais non supportée".into(),
                    ));
                }
                metal.resolve_linear_weight_buffers(head.weight(), "tts_cp_head")
            })
            .collect::<Result<Vec<_>>>()?;
        let vocab = metal.linear_weight_out_dim(&heads[0]);
        if vocab > TTS_SAMPLER_MAX_VOCAB
            || heads
                .iter()
                .any(|head| metal.linear_weight_out_dim(head) != vocab)
        {
            return Ok(None);
        }
        let embeddings = embeddings
            .iter()
            .take(heads.len() - 1)
            .map(|table| {
                let (rows, dim) = table.as_matrix()?;
                if rows != vocab || dim != *talker_hidden {
                    return Err(InferError::Dimension(format!(
                        "embedding code predictor [{rows}, {dim}], attendu [{vocab}, {talker_hidden}]"
                    )));
                }
                // Les tables viennent de poids bf16 : le stockage bf16 est exact.
                Ok(MetalEmbeddingWeightBuffers::DenseBf16 {
                    table: metal.cached_buffer_from_f32_as_bf16(table.data(), "tts_cp_embedding")?,
                    vocab: rows,
                    dim,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Some(TtsCodebookWeights {
            projection: projection_buffers,
            projection_bias,
            heads,
            embeddings,
            talker_hidden: *talker_hidden,
            vocab,
        }))
    }

    /// Décode les codebooks d'une trame en un command buffer et renvoie les
    /// `heads.len()` codes tirés. `final_state` et `first_embedding` sont des
    /// lignes talker (non projetées). Le tirage `i` utilise l'état RNG
    /// `rng_state_for_draw(rng_state, i)` : l'appelant avance ensuite son
    /// sampler d'autant de tirages. Le cache est remis à zéro au début.
    ///
    /// Renvoie `Ok(None)` si l'arène résidente est absente.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si une forme est invalide ou si un encodage échoue.
    #[expect(
        clippy::too_many_arguments,
        reason = "cache, poids, deux lignes d'entrée, paramètres de tirage et trace"
    )]
    pub(crate) fn decode_tts_codebooks_resident(
        &self,
        cache: &mut CausalDecoderCache,
        weights: &TtsCodebookWeights,
        final_state: &[f32],
        first_embedding: &[f32],
        sample: TtsCodebookSample,
        rng_state: u64,
        trace: bool,
    ) -> Result<Option<(Vec<u32>, Option<TtsCodebookTrace>)>> {
        let Some(metal) = self.forward_runtime().metal_executor() else {
            return Ok(None);
        };
        if final_state.len() != weights.talker_hidden
            || first_embedding.len() != weights.talker_hidden
        {
            return Err(InferError::Dimension(format!(
                "entrées code predictor attendues de {} éléments",
                weights.talker_hidden
            )));
        }
        let heads = weights.heads.len();
        let vocab = weights.vocab;
        let hidden = self.final_norm.len();
        let theta = self.config.rope_theta.ok_or_else(|| {
            InferError::Config("rope_theta manquant (trame TTS résidente)".to_string())
        })?;
        let eps = self.config.rms_eps;
        self.reset_resident_decode_cache(cache)?;
        let CausalDecoderCache {
            layers,
            position: cache_position,
            resident,
        } = cache;
        let Some(arena) = resident.as_mut() else {
            return Ok(None);
        };

        let scratch = arena.state.scratch();
        let seed = scratch.lease(weights.talker_hidden, GpuElement::F32)?;
        let first = scratch.lease(weights.talker_hidden, GpuElement::F32)?;
        let embed = scratch.lease(weights.talker_hidden, GpuElement::F32)?;
        let codes = scratch.lease(heads, GpuElement::U32)?;
        let final_normed = scratch.lease(hidden, GpuElement::F32)?;
        let logits = (0..heads)
            .map(|_| scratch.lease(vocab, GpuElement::F32))
            .collect::<Result<Vec<_>>>()?;
        let trace_lease = trace
            .then(|| scratch.lease(heads * 2 * TTS_SAMPLER_TRACE_K, GpuElement::F32))
            .transpose()?;
        arena.state.upload(seed.tensor(), final_state)?;
        arena.state.upload(first.tensor(), first_embedding)?;

        let command_buffer = arena.state.queue().new_command_buffer();
        let _barrier_guard = crate::metal_backend::resident_concurrent_enabled()
            .then(crate::metal_backend::install_dispatch_barrier_scope);
        let _namespace_guard =
            crate::metal_backend::install_scratch_namespace(arena.state.scratch_namespace());
        let encoder = new_resident_compute_encoder(command_buffer);
        let mut owned: Vec<metal::Buffer> = Vec::new();

        for step in 0..=heads {
            let input = match step {
                0 => seed.tensor().buffer(),
                1 => first.tensor().buffer(),
                _ => embed.tensor().buffer(),
            };
            let projected = metal.encode_matmul_weight_buffers(
                encoder,
                input,
                1,
                weights.talker_hidden,
                &weights.projection,
                arena.hidden_a.buffer(),
                false,
            )?;
            if projected != hidden {
                return Err(InferError::Dimension(format!(
                    "projection code predictor sort {projected}, attendu {hidden}"
                )));
            }
            if let Some(bias) = &weights.projection_bias {
                metal.encode_add_row_bias(encoder, arena.hidden_a.buffer(), 0, bias, 1, hidden)?;
            }
            let position = *cache_position;
            let mut current = &arena.hidden_a;
            let mut other = &arena.hidden_b;
            for (index, layer) in self.layers.iter().enumerate() {
                let dims = self.plan.layers[index]
                    .resident
                    .resident_windowed_full_attn_layer_dims(hidden, position, eps, theta)?;
                self.encode_resident_full_layer(
                    metal,
                    arena,
                    &mut layers[index],
                    layer,
                    index,
                    encoder,
                    &mut owned,
                    dims,
                    current.buffer(),
                    other.buffer(),
                )?;
                std::mem::swap(&mut current, &mut other);
            }
            *cache_position += 1;
            if step == 0 {
                // L'amorce ne produit qu'un KV : ni norme, ni tête.
                continue;
            }
            let head = step - 1;
            metal.encode_rms_norm_rows(
                encoder,
                current.buffer(),
                &arena.final_norm,
                final_normed.tensor().buffer(),
                1,
                hidden,
                eps,
            )?;
            metal.encode_lm_head_logits_readback_buffers(
                encoder,
                final_normed.tensor().buffer(),
                &weights.heads[head],
                hidden,
                logits[head].tensor().buffer(),
            )?;
            arena.state.encode_tts_codebook_sample(
                encoder,
                logits[head].tensor().buffer(),
                vocab,
                codes.tensor().buffer(),
                head,
                trace_lease.as_ref().map(|lease| &**lease.tensor().buffer()),
                sample,
                crate::sampling::rng_state_for_draw(rng_state, head),
            )?;
            if head + 1 < heads {
                metal.encode_embedding_from_index_buffers_with_offset(
                    encoder,
                    &weights.embeddings[head],
                    codes.tensor().buffer(),
                    (head * std::mem::size_of::<u32>()) as u64,
                    embed.tensor().buffer(),
                    weights.talker_hidden,
                )?;
            }
        }
        encoder.end_encoding();
        crate::metal_backend::commit_and_wait(command_buffer)?;
        drop(owned);
        let codes = crate::metal_backend::read_u32_buffer(codes.tensor().buffer(), heads)?;
        let trace = match trace_lease {
            None => None,
            Some(lease) => {
                let mut all_logits = Vec::with_capacity(heads * vocab);
                for lease in &logits {
                    all_logits.extend(crate::metal_backend::read_f32_buffer(
                        lease.tensor().buffer(),
                        vocab,
                    )?);
                }
                let raw = crate::metal_backend::read_f32_buffer(
                    lease.tensor().buffer(),
                    heads * 2 * TTS_SAMPLER_TRACE_K,
                )?;
                let distributions = raw
                    .chunks_exact(2 * TTS_SAMPLER_TRACE_K)
                    .map(|row| {
                        let (probs, ids) = row.split_at(TTS_SAMPLER_TRACE_K);
                        probs
                            .iter()
                            .zip(ids)
                            .map(|(prob, id)| (id.to_bits(), *prob))
                            .take_while(|(id, _)| *id != u32::MAX)
                            .map(|(id, prob)| (id as usize, prob))
                            .collect()
                    })
                    .collect();
                Some(TtsCodebookTrace {
                    logits: all_logits,
                    distributions,
                })
            }
        };
        Ok(Some((codes, trace)))
    }
}
