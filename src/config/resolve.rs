use super::gemma_defaults::{apply_gemma3_defaults, apply_gemma4_defaults};
use super::types::{ModelConfig, RawModelConfig};
use crate::{InferError, Result};

/// Arrondit un f32 au bf16 le plus proche (round-to-nearest-even), renvoyé en f32.
pub(super) fn bf16_round(value: f32) -> f32 {
    let bits = value.to_bits();
    let rounding = 0x7fff + ((bits >> 16) & 1);
    f32::from_bits(bits.wrapping_add(rounding) & 0xffff_0000)
}

impl RawModelConfig {
    /// Résout la configuration brute en configuration normalisée.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si la désérialisation ou la validation échoue.
    pub fn resolve(self) -> Result<ModelConfig> {
        let top_model_type = self.model_type.clone();
        let quant = self.quantization.or(self.quantization_config);
        let rest = self.rest;
        let top_eos = rest.get("eos_token_id").cloned();
        let top_vocab_size = rest.get("vocab_size").cloned();
        let mut base = self.text_config.unwrap_or(rest);

        if let serde_json::Value::Object(ref mut map) = base {
            if !top_model_type.is_empty() {
                map.insert(
                    "model_type".to_string(),
                    serde_json::Value::String(top_model_type),
                );
            }
            if let Some(rope) = map.get("rope_parameters").cloned() {
                if let Some(theta) = rope.get("rope_theta").cloned() {
                    map.entry("rope_theta".to_string()).or_insert(theta);
                }
                if let Some(factor) = rope.get("partial_rotary_factor").cloned() {
                    map.entry("partial_rotary_factor".to_string())
                        .or_insert(factor);
                }
            }
            if let Some(eos) = top_eos {
                map.entry("eos_token_id".to_string()).or_insert(eos);
            }
            apply_gemma3_defaults(map, top_vocab_size.clone());
            apply_gemma4_defaults(map, top_vocab_size);
            map.remove("quantization");
            map.remove("quantization_config");
        }

        let mut cfg: ModelConfig =
            serde_json::from_value(base).map_err(|e| InferError::Config(e.to_string()))?;
        cfg.quantization = quant;
        if cfg.num_experts_per_tok.is_none() {
            cfg.num_experts_per_tok = cfg.top_k_experts;
        }
        if cfg.shared_expert_intermediate_size == Some(0) {
            cfg.shared_expert_intermediate_size = None;
        }
        validate_config(&cfg)?;
        Ok(cfg)
    }
}

fn validate_config(cfg: &ModelConfig) -> Result<()> {
    if cfg.hidden_size == 0
        || cfg.num_hidden_layers == 0
        || cfg.num_attention_heads == 0
        || cfg.num_key_value_heads == 0
        || cfg.vocab_size == 0
    {
        return Err(InferError::Config(format!(
            "dimensions nulles: hidden={}, layers={}, heads={}, kv_heads={}, vocab={}",
            cfg.hidden_size,
            cfg.num_hidden_layers,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.vocab_size
        )));
    }
    if cfg.hidden_size % cfg.num_attention_heads != 0 && cfg.head_dim.is_none() {
        return Err(InferError::Config(format!(
            "hidden_size {} non divisible par num_attention_heads {} sans head_dim explicite",
            cfg.hidden_size, cfg.num_attention_heads
        )));
    }
    if cfg.num_attention_heads % cfg.num_key_value_heads != 0 {
        return Err(InferError::Config(format!(
            "num_attention_heads {} non multiple de num_key_value_heads {}",
            cfg.num_attention_heads, cfg.num_key_value_heads
        )));
    }
    // NOTE: Pas de contrainte `hidden_size % head_dim` quand head_dim est
    // explicite : q_dim (= heads·head_dim) peut différer de hidden_size
    // (Gemma 3 1B : 4×256 = 1024 vs hidden 1152) ; les formes réelles des
    // projections sont validées par le contrat du chargeur.
    if let Some(0) = cfg.head_dim {
        return Err(InferError::Config("head_dim explicite nul".to_string()));
    }
    Ok(())
}
