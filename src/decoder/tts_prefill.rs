//! Extension VoiceDesign conservant l'arithmétique du prefill per-op.

use super::attention_ops::{causal_attention_suffix, full_attention_prepare_rows, sigmoid_scalar};
use super::*;

impl CausalDecoderCache {
    /// Un préfixe mémorisé ne grandit plus : sa copie ne doit pas immobiliser
    /// la réserve d'ajout de chaque couche. Clone écarte les états résidents.
    pub(crate) fn compact_tts_snapshot(&self) -> Self {
        let mut snapshot = self.clone();
        for layer in &mut snapshot.layers {
            layer.keys = layer.keys.compact_snapshot();
            layer.values = layer.values.compact_snapshot();
        }
        snapshot
    }
}

impl CausalDecoder {
    pub(crate) fn extend_tts_prefill(
        &self,
        cache: &mut CausalDecoderCache,
        input: &Tensor,
    ) -> Result<Tensor> {
        if input.as_matrix()? != (1, self.final_norm.len())
            || cache.layers.len() != self.layers.len()
            || cache.position == 0
        {
            return Err(InferError::Dimension(
                "extension prefill TTS incompatible".into(),
            ));
        }
        let position = cache.position;
        let next_position = position
            .checked_add(1)
            .ok_or_else(|| InferError::Dimension("position prefill TTS hors limites".into()))?;
        let runtime = self.forward_runtime();
        let mut hidden = input.clone();
        for (layer, kv) in self.layers.iter().zip(&mut cache.layers) {
            // VoiceDesign ne contient ni récurrence ni MoE ; refuser une autre
            // architecture plutôt que réutiliser un état incomplet.
            let AttentionBlock::Full(attention) = &layer.attention else {
                return Err(InferError::Config(
                    "prefill TTS sans attention pleine".into(),
                ));
            };
            if matches!(layer.mlp, Some(FeedForward::Moe(_)))
                || kv.len() != position
                || kv.base_rows() != 0
            {
                return Err(InferError::Config("cache prefill TTS incompatible".into()));
            }
            let normed = rms_norm(&hidden, &layer.input_norm, self.config.rms_eps)?;
            let (q, gate, k, v, layout) = full_attention_prepare_rows(
                &self.config,
                &normed,
                position,
                attention,
                runtime,
                true,
            )?;
            kv.append_batch(&k, &v, &layout)?;
            let dim = layout.num_key_value_heads * layout.head_dim;
            let keys = Tensor::from_vec(vec![next_position, dim], kv.keys.to_vec())?;
            let values = Tensor::from_vec(vec![next_position, dim], kv.values.to_vec())?;
            let mut context = causal_attention_suffix(
                &q,
                &keys,
                &values,
                &layout,
                attention.sinks.as_ref().map(Tensor::data),
                position,
            )?;
            if let Some(gate) = gate {
                context = context.mul_elementwise(&gate.map(sigmoid_scalar))?;
            }
            let output = attention.o_proj.forward_with_runtime(&context, runtime)?;
            hidden = layer.combine_attention_and_mlp(&self.config, &hidden, &output, runtime)?;
        }
        cache.position = next_position;
        rms_norm(&hidden, &self.final_norm, self.config.rms_eps)
    }
}
