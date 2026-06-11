//! Chargement et normalisation des configurations de modèles Qwen.

use crate::{InferError, Result};
use serde::{Deserialize, Deserializer};
use std::collections::HashMap;
use std::path::Path;

#[derive(Clone, Debug, Deserialize, PartialEq)]
/// Décrit la configuration normalisée d'un modèle Qwen.
pub struct ModelConfig {
    /// Indique le type de modèle déclaré.
    #[serde(default)]
    pub model_type: String,
    /// Définit la dimension cachée principale.
    pub hidden_size: usize,
    /// Définit le nombre de couches du décodeur.
    pub num_hidden_layers: usize,
    /// Définit le nombre de têtes d'attention requête.
    pub num_attention_heads: usize,
    /// Définit le nombre de têtes clé/valeur.
    pub num_key_value_heads: usize,
    /// Surcharge la dimension de tête calculée.
    #[serde(default)]
    pub head_dim: Option<usize>,
    /// Définit la dimension intermédiaire MLP dense.
    #[serde(default)]
    pub intermediate_size: usize,
    /// Définit l'epsilon des normalisations RMS.
    pub rms_norm_eps: f32,
    /// Définit la base angulaire RoPE.
    pub rope_theta: f32,
    /// Définit la taille du vocabulaire.
    pub vocab_size: usize,
    /// Liste les tokens de fin de séquence.
    #[serde(
        default,
        rename = "eos_token_id",
        deserialize_with = "deserialize_token_ids"
    )]
    pub eos_token_ids: Vec<usize>,
    /// Indique si lm_head partage les embeddings.
    #[serde(default)]
    pub tie_word_embeddings: bool,
    /// Décrit la quantification déclarée.
    #[serde(default)]
    pub quantization: Option<QuantConfig>,
    /// Définit l'intervalle des couches full-attention.
    #[serde(default)]
    pub full_attention_interval: Option<usize>,
    /// Active la porte de sortie attention.
    #[serde(default)]
    pub attn_output_gate: Option<bool>,
    /// Définit la fraction RoPE appliquée.
    #[serde(default)]
    pub partial_rotary_factor: Option<f32>,
    /// Définit le nombre de têtes valeur linéaires.
    #[serde(default)]
    pub linear_num_value_heads: Option<usize>,
    /// Définit le nombre de têtes clé linéaires.
    #[serde(default)]
    pub linear_num_key_heads: Option<usize>,
    /// Définit la dimension des têtes clé linéaires.
    #[serde(default)]
    pub linear_key_head_dim: Option<usize>,
    /// Définit la dimension des têtes valeur linéaires.
    #[serde(default)]
    pub linear_value_head_dim: Option<usize>,
    /// Définit la largeur de convolution linéaire.
    #[serde(default)]
    pub linear_conv_kernel_dim: Option<usize>,
    /// Définit le nombre d'experts MoE.
    #[serde(default)]
    pub num_experts: Option<usize>,
    /// Définit le nombre d'experts sélectionnés par token.
    #[serde(default)]
    pub num_experts_per_tok: Option<usize>,
    /// Définit la dimension intermédiaire des experts MoE.
    #[serde(default)]
    pub moe_intermediate_size: Option<usize>,
    /// Définit la dimension intermédiaire de l'expert partagé.
    #[serde(default)]
    pub shared_expert_intermediate_size: Option<usize>,
    /// Définit le nombre de couches MTP sidecar.
    #[serde(default)]
    pub mtp_num_hidden_layers: Option<usize>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
/// Décrit les paramètres de quantification d'un modèle.
pub struct QuantConfig {
    /// Définit la taille de groupe de quantification.
    #[serde(default)]
    pub group_size: Option<usize>,
    /// Définit le nombre de bits par poids.
    #[serde(default)]
    pub bits: Option<usize>,
    /// Indique la méthode de quantification.
    #[serde(default)]
    pub quant_method: Option<String>,
    /// Indique le format de quantification.
    #[serde(default)]
    pub fmt: Option<String>,
    /// Conserve les champs inconnus du fichier source.
    #[serde(default, flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
/// Représente la configuration brute lue depuis JSON.
pub struct RawModelConfig {
    #[serde(default)]
    model_type: String,
    #[serde(default)]
    text_config: Option<serde_json::Value>,
    #[serde(default)]
    quantization: Option<QuantConfig>,
    #[serde(default)]
    quantization_config: Option<QuantConfig>,
    #[serde(flatten)]
    rest: serde_json::Value,
}

impl ModelConfig {
    /// Charge et normalise une configuration depuis un fichier JSON.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si la lecture, le JSON ou la validation échoue.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = std::fs::File::open(path).map_err(|source| InferError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let raw: RawModelConfig =
            serde_json::from_reader(file).map_err(|source| InferError::Json {
                path: path.to_path_buf(),
                source,
            })?;
        raw.resolve()
    }

    #[must_use]
    /// Renvoie la dimension d'une tête d'attention.
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or_else(|| self.hidden_size / self.num_attention_heads)
    }

    #[must_use]
    /// Renvoie le nombre de dimensions RoPE.
    pub fn rope_dims(&self) -> usize {
        match self.partial_rotary_factor {
            Some(factor) => (self.head_dim() as f32 * factor) as usize,
            None => self.head_dim(),
        }
    }

    #[must_use]
    /// Renvoie la dimension intermédiaire MLP effective.
    pub fn mlp_intermediate_size(&self) -> usize {
        if self.intermediate_size > 0 {
            self.intermediate_size
        } else {
            self.moe_intermediate_size.unwrap_or(0)
        }
    }

    #[must_use]
    /// Indique si le modèle alterne attention linéaire et full-attention.
    pub fn is_hybrid(&self) -> bool {
        self.full_attention_interval.is_some()
    }

    #[must_use]
    /// Indique si une couche utilise la full-attention.
    pub fn is_full_attention_layer(&self, layer_index: usize) -> bool {
        match self.full_attention_interval {
            Some(interval) if interval > 0 => (layer_index + 1) % interval == 0,
            _ => true,
        }
    }

    #[must_use]
    /// Indique si la configuration décrit un MoE.
    pub fn is_moe(&self) -> bool {
        self.model_type.contains("moe") || self.num_experts.unwrap_or(0) > 0
    }
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
            map.remove("quantization");
            map.remove("quantization_config");
        }

        let mut cfg: ModelConfig =
            serde_json::from_value(base).map_err(|e| InferError::Config(e.to_string()))?;
        cfg.quantization = quant;
        validate_config(&cfg)?;
        Ok(cfg)
    }
}

fn deserialize_token_ids<'de, D>(deserializer: D) -> std::result::Result<Vec<usize>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Option::<serde_json::Value>::deserialize(deserializer)?;
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    match value {
        serde_json::Value::Null => Ok(Vec::new()),
        serde_json::Value::Number(n) => n
            .as_u64()
            .and_then(|id| usize::try_from(id).ok())
            .map(|id| vec![id])
            .ok_or_else(|| serde::de::Error::custom("eos_token_id invalide")),
        serde_json::Value::Array(items) => items
            .into_iter()
            .map(|item| match item {
                serde_json::Value::Number(n) => n
                    .as_u64()
                    .and_then(|id| usize::try_from(id).ok())
                    .ok_or_else(|| serde::de::Error::custom("eos_token_id invalide")),
                _ => Err(serde::de::Error::custom("eos_token_id non numérique")),
            })
            .collect(),
        _ => Err(serde::de::Error::custom(
            "eos_token_id doit être nombre ou liste",
        )),
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
    if let Some(head_dim) = cfg.head_dim {
        if cfg.hidden_size % head_dim != 0 {
            return Err(InferError::Config(format!(
                "hidden_size {} non divisible par head_dim explicite {}",
                cfg.hidden_size, head_dim
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn resolves_flat_qwen_config() {
        let raw = r#"{
            "model_type":"qwen3",
            "hidden_size":1024,
            "num_hidden_layers":28,
            "num_attention_heads":16,
            "num_key_value_heads":8,
            "head_dim":128,
            "intermediate_size":3072,
            "rms_norm_eps":0.000001,
            "rope_theta":1000000.0,
            "vocab_size":151936
        }"#;
        let cfg: RawModelConfig = serde_json::from_str(raw).expect("invariant: JSON valide");
        let cfg = cfg.resolve().expect("invariant: config valide");
        assert_eq!(cfg.model_type, "qwen3");
        assert_eq!(cfg.head_dim(), 128);
        assert_eq!(cfg.rope_dims(), 128);
        assert!(cfg.eos_token_ids.is_empty());
    }

    proptest! {
        #[test]
        fn parses_single_eos_token_id(id in 0_usize..1_000_000) {
            let raw = format!(
                r#"{{
                    "model_type":"qwen3",
                    "hidden_size":1024,
                    "num_hidden_layers":2,
                    "num_attention_heads":16,
                    "num_key_value_heads":8,
                    "head_dim":64,
                    "intermediate_size":3072,
                    "rms_norm_eps":0.000001,
                    "rope_theta":1000000.0,
                    "vocab_size":151936,
                    "eos_token_id":{id}
                }}"#
            );
            let cfg: RawModelConfig = serde_json::from_str(&raw)
                .expect("invariant: JSON généré valide");
            let cfg = cfg.resolve().expect("invariant: config générée valide");
            prop_assert_eq!(cfg.eos_token_ids, vec![id]);
        }

        #[test]
        fn parses_eos_token_id_arrays(ids in proptest::collection::vec(0_usize..1_000_000, 0..8)) {
            let ids_json = serde_json::to_string(&ids).expect("invariant: ids sérialisables");
            let raw = format!(
                r#"{{
                    "model_type":"qwen3",
                    "hidden_size":1024,
                    "num_hidden_layers":2,
                    "num_attention_heads":16,
                    "num_key_value_heads":8,
                    "head_dim":64,
                    "intermediate_size":3072,
                    "rms_norm_eps":0.000001,
                    "rope_theta":1000000.0,
                    "vocab_size":151936,
                    "eos_token_id":{ids_json}
                }}"#
            );
            let cfg: RawModelConfig = serde_json::from_str(&raw)
                .expect("invariant: JSON généré valide");
            let cfg = cfg.resolve().expect("invariant: config générée valide");
            prop_assert_eq!(cfg.eos_token_ids, ids);
        }
    }

    #[test]
    fn resolves_nested_text_config_and_rope_parameters() {
        let raw = r#"{
            "model_type":"qwen3_5_moe",
            "quantization_config":{
                "group_size":64,
                "bits":4,
                "quant_method":"mx",
                "model.layers.0.self_attn.q_proj":{"group_size":64,"bits":8}
            },
            "text_config":{
                "model_type":"qwen3_5_moe_text",
                "hidden_size":2048,
                "num_hidden_layers":40,
                "num_attention_heads":16,
                "num_key_value_heads":4,
                "attn_output_gate":true,
                "head_dim":128,
                "intermediate_size":6144,
                "rms_norm_eps":0.000001,
                "vocab_size":152064,
                "eos_token_id":248044,
                "rope_parameters":{
                    "rope_theta":1000000.0,
                    "partial_rotary_factor":0.25
                },
                "full_attention_interval":4,
                "linear_conv_kernel_dim":4,
                "linear_key_head_dim":128,
                "linear_num_key_heads":16,
                "linear_num_value_heads":32,
                "linear_value_head_dim":128
            }
        }"#;
        let cfg: RawModelConfig = serde_json::from_str(raw).expect("invariant: JSON valide");
        let cfg = cfg.resolve().expect("invariant: config valide");
        assert_eq!(cfg.model_type, "qwen3_5_moe");
        assert_eq!(cfg.eos_token_ids, vec![248044]);
        assert_eq!(cfg.rope_theta, 1_000_000.0);
        assert_eq!(cfg.attn_output_gate, Some(true));
        assert_eq!(cfg.rope_dims(), 32);
        assert!(cfg.is_hybrid());
        assert!(!cfg.is_full_attention_layer(0));
        assert!(cfg.is_full_attention_layer(3));
        assert_eq!(cfg.linear_num_key_heads, Some(16));
        assert_eq!(cfg.linear_num_value_heads, Some(32));
        assert_eq!(
            cfg.quantization
                .as_ref()
                .and_then(|q| q.group_size)
                .expect("invariant: quantization présente"),
            64
        );
        assert_eq!(
            cfg.quantization
                .as_ref()
                .and_then(|q| q.extra.get("model.layers.0.self_attn.q_proj"))
                .and_then(|v| v.get("bits"))
                .and_then(serde_json::Value::as_u64),
            Some(8)
        );
    }

    #[test]
    fn resolves_moe_config_without_dense_intermediate_size() {
        let raw = r#"{
            "model_type":"qwen3_5_moe",
            "text_config":{
                "hidden_size":2048,
                "num_hidden_layers":40,
                "num_attention_heads":16,
                "num_key_value_heads":4,
                "head_dim":128,
                "moe_intermediate_size":512,
                "rms_norm_eps":0.000001,
                "vocab_size":248320,
                "rope_parameters":{"rope_theta":10000000.0}
            }
        }"#;
        let cfg: RawModelConfig = serde_json::from_str(raw).expect("invariant: JSON valide");
        let cfg = cfg.resolve().expect("invariant: config MoE valide");
        assert_eq!(cfg.intermediate_size, 0);
        assert_eq!(cfg.mlp_intermediate_size(), 512);
    }

    #[test]
    fn resolves_top_level_eos_list_into_nested_text_config() {
        let raw = r#"{
            "model_type":"qwen3_5_moe",
            "eos_token_id":[248044,248045],
            "text_config":{
                "hidden_size":2048,
                "num_hidden_layers":40,
                "num_attention_heads":16,
                "num_key_value_heads":4,
                "head_dim":128,
                "moe_intermediate_size":512,
                "rms_norm_eps":0.000001,
                "vocab_size":248320,
                "rope_theta":10000000.0
            }
        }"#;
        let cfg: RawModelConfig = serde_json::from_str(raw).expect("invariant: JSON valide");
        let cfg = cfg.resolve().expect("invariant: config valide");
        assert_eq!(cfg.eos_token_ids, vec![248044, 248045]);
    }
}
