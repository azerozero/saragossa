//! Oracle interne image → LLM, sans ouverture des transports média.

use super::*;
use crate::vision::prompt::{insert_embeddings, SuppliedImage};

#[cfg(test)]
pub(super) mod tests;

impl CausalDecoderCache {
    /// Position RoPE du prochain token. L'adresse KV reste `position()`.
    pub(crate) fn current_pos(&self) -> Result<usize> {
        let tokens = i64::try_from(self.position)
            .map_err(|_| InferError::Dimension("compteur KV hors i64".into()))?;
        tokens
            .checked_add(self.rope_delta)
            .and_then(|p| usize::try_from(p).ok())
            .ok_or_else(|| InferError::Dimension("tokens + rope_delta hors limites".into()))
    }
}

impl CausalDecoder {
    #[cfg(all(test, target_os = "macos", feature = "metal"))]
    pub(crate) fn vision_forced_logits(
        &self,
        cache: &mut CausalDecoderCache,
        tokens: &[usize],
        resident: bool,
    ) -> Result<Vec<Vec<f32>>> {
        if resident && !self.setup_resident_full_decode(cache, tokens.len(), 0, false)? {
            return Err(InferError::Config(
                "oracle vision sans decode résident".into(),
            ));
        }
        tokens
            .iter()
            .map(|&token| {
                if resident {
                    self.resident_reference_step(cache, token)
                } else {
                    self.next_logits_cached(cache, token)
                        .map(|t| t.data().to_vec())
                }
            })
            .collect()
    }

    /// Entrée réservée aux oracles : embeddings validés, aucune lecture média.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "API interne de qualification vision, transport encore fermé"
        )
    )]
    pub(crate) fn prefill_supplied_vision(
        &self,
        tokens: &[usize],
        image_pad: usize,
        images: &[SuppliedImage<'_>],
        chunk_tokens: usize,
        resident: bool,
    ) -> Result<(CausalDecoderCache, Tensor)> {
        let (cache, state) =
            self.prefill_supplied_vision_state(tokens, image_pad, images, chunk_tokens, resident)?;
        Ok((cache, self.logits_from_final_state(&state)?))
    }

    pub(super) fn prefill_supplied_vision_state(
        &self,
        tokens: &[usize],
        image_pad: usize,
        images: &[SuppliedImage<'_>],
        chunk_tokens: usize,
        resident: bool,
    ) -> Result<(CausalDecoderCache, Tensor)> {
        if chunk_tokens == 0
            || self.config.rope_theta != Some(10_000_000.0)
            || self.config.rope_style != RopeStyle::Halves
            || self.config.head_dim != Some(256)
            || self.config.rope_dims != Some(64)
            || self.config.rope_position_scale.unwrap_or(1.0) != 1.0
            || self.config.sliding_window.is_some()
            || self.config.rope_full_base_freq.is_some()
            || !self.config.no_rope_layers.is_empty()
        {
            return Err(InferError::Config(
                "prefill vision hors contrat Qwen".into(),
            ));
        }
        let (input, positions) =
            insert_embeddings(tokens, image_pad, &self.embed_scaled(tokens)?, images)?;
        let delta = positions.delta_at(tokens.len())?;
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if resident {
            let (mut cache, hidden, _) = self
                .prefill_resident_cache_hidden_captures_and_mtp(
                    tokens,
                    &[],
                    usize::MAX,
                    true,
                    chunk_tokens,
                    None,
                    Some((&input, &positions.rows)),
                )?
                .ok_or_else(|| InferError::Config("prefill vision résident indisponible".into()))?;
            cache.rope_delta = delta;
            cache.vision_positions = Some(positions);
            let hidden = hidden.into_tensor(self.final_norm.len())?;
            let final_state = rms_norm(&hidden, &self.final_norm, self.config.rms_eps)?;
            return Ok((cache, final_state));
        }
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        if resident {
            return Err(InferError::Config("Metal indisponible".into()));
        }
        let mut cache = self.empty_cache();
        let mut final_hidden = None;
        let width = self.final_norm.len();
        for start in (0..tokens.len()).step_by(chunk_tokens) {
            let end = tokens.len().min(start.saturating_add(chunk_tokens));
            let mut hidden = Tensor::from_vec(
                vec![end - start, width],
                input.data()[start * width..end * width].to_vec(),
            )?;
            let runtime = self
                .forward_runtime()
                .with_rope_positions(&positions.rows[start..end]);
            for (layer, kv) in self.layers.iter().zip(&mut cache.layers) {
                hidden = layer.forward_prefill(&self.config, &hidden, kv, start, runtime)?;
            }
            cache.position = end;
            final_hidden = Some(Tensor::row(hidden.row_slice(end - start - 1)?.to_vec())?);
        }
        let hidden =
            final_hidden.ok_or_else(|| InferError::Dimension("prompt vision vide".into()))?;
        cache.rope_delta = delta;
        cache.vision_positions = Some(positions);
        let final_state = rms_norm(&hidden, &self.final_norm, self.config.rms_eps)?;
        Ok((cache, final_state))
    }
}
