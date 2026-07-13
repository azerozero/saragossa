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
    /// Surcharge les têtes K/V des couches globales Gemma 4.
    #[serde(default)]
    pub num_global_key_value_heads: Option<usize>,
    /// Surcharge la dimension de tête calculée.
    #[serde(default)]
    pub head_dim: Option<usize>,
    /// Surcharge la dimension de tête des couches globales Gemma 4.
    #[serde(default)]
    pub global_head_dim: Option<usize>,
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
    /// Définit le nombre d'experts sélectionnés par token (nom Gemma 4).
    #[serde(default)]
    pub top_k_experts: Option<usize>,
    /// Définit la dimension intermédiaire des experts MoE.
    #[serde(default)]
    pub moe_intermediate_size: Option<usize>,
    /// Définit la dimension intermédiaire de l'expert partagé.
    #[serde(default)]
    pub shared_expert_intermediate_size: Option<usize>,
    /// Définit le nombre de couches MTP sidecar.
    #[serde(default)]
    pub mtp_num_hidden_layers: Option<usize>,
    /// Indique l'activation MLP déclarée (`gelu_pytorch_tanh` pour Gemma).
    ///
    /// Champ distinct de `hidden_act` (pas un alias serde) : les configs Gemma 2
    /// sérialisent LES DEUX clés, et un alias provoquerait `duplicate field`.
    #[serde(default)]
    pub hidden_activation: Option<String>,
    /// Indique l'activation MLP déclarée sous la clé historique `hidden_act`.
    #[serde(default)]
    pub hidden_act: Option<String>,
    /// Définit la base RoPE des couches locales Gemma 3 (sliding-window).
    #[serde(default)]
    pub rope_local_base_freq: Option<f32>,
    /// Définit la base RoPE des couches globales Gemma 4.
    #[serde(default)]
    pub rope_full_base_freq: Option<f32>,
    /// Définit la fraction RoPE des couches globales Gemma 4.
    #[serde(default)]
    pub rope_full_partial_rotary_factor: Option<f32>,
    /// Définit la fraction RoPE des couches locales Gemma 4.
    #[serde(default)]
    pub rope_sliding_partial_rotary_factor: Option<f32>,
    /// Définit la taille de fenêtre des couches sliding-window (Gemma 3).
    #[serde(default)]
    pub sliding_window: Option<usize>,
    /// Définit la période de la couche globale Gemma 3 (1 globale / N locales).
    #[serde(default)]
    pub sliding_window_pattern: Option<usize>,
    /// Liste le type d'attention de chaque couche Gemma 4.
    #[serde(default)]
    pub layer_types: Vec<String>,
    /// Indique si les couches globales Gemma 4 partagent K et V.
    #[serde(default)]
    pub attention_k_eq_v: bool,
    /// Active le bloc MoE parallèle Gemma 4.
    #[serde(default)]
    pub enable_moe_block: bool,
    /// Surcharge le facteur d'échelle des scores d'attention Gemma.
    #[serde(default)]
    pub query_pre_attn_scalar: Option<f32>,
    /// Déclare le softcapping des scores d'attention (Gemma 2, non supporté).
    #[serde(default)]
    pub attn_logit_softcapping: Option<f32>,
    /// Déclare le softcapping des logits finaux (Gemma 2, non supporté).
    #[serde(default)]
    pub final_logit_softcapping: Option<f32>,
    /// Décrit le scaling RoPE déclaré (`null`/absent → aucun scaling).
    #[serde(default)]
    pub rope_scaling: Option<RopeScalingConfig>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
/// Décrit la section `rope_scaling` d'une configuration HF.
pub struct RopeScalingConfig {
    /// Indique le type de scaling sous la clé moderne `rope_type`.
    #[serde(default)]
    pub rope_type: Option<String>,
    /// Indique le type de scaling sous la clé historique `type`.
    ///
    /// Champ distinct de `rope_type` (pas un alias serde) : des configs HF en
    /// migration sérialisent LES DEUX clés, et un alias provoquerait
    /// `duplicate field`.
    #[serde(default, rename = "type")]
    pub legacy_type: Option<String>,
    /// Définit le facteur d'étirement des positions.
    #[serde(default)]
    pub factor: Option<f32>,
}

impl RopeScalingConfig {
    /// Renvoie le type de scaling effectif.
    ///
    /// Même résolution que `initialize_rope` de mlx_lm : la clé historique
    /// `type` prime sur `rope_type`, défaut `default` (aucun scaling).
    #[must_use]
    pub fn scaling_type(&self) -> &str {
        self.legacy_type
            .as_deref()
            .or(self.rope_type.as_deref())
            .unwrap_or("default")
    }
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

    #[must_use]
    /// Indique si le modèle suit une architecture Gemma.
    pub fn is_gemma(&self) -> bool {
        self.model_type.starts_with("gemma")
    }

    #[must_use]
    /// Indique si la configuration suit l'architecture Gemma 4 textuelle.
    pub fn is_gemma4(&self) -> bool {
        matches!(
            self.model_type.as_str(),
            "gemma4" | "gemma4_text" | "gemma4_unified" | "gemma4_unified_text"
        )
    }

    #[must_use]
    /// Indique si la couche déclarée est une couche Gemma 4 globale.
    pub fn is_gemma4_full_layer(&self, layer_index: usize) -> bool {
        self.layer_types
            .get(layer_index)
            .is_some_and(|kind| kind == "full_attention")
    }

    #[must_use]
    /// Indique si la couche déclarée est une couche Gemma 4 locale.
    pub fn is_gemma4_sliding_layer(&self, layer_index: usize) -> bool {
        self.layer_types
            .get(layer_index)
            .is_some_and(|kind| kind == "sliding_attention")
    }

    #[must_use]
    /// Renvoie la dimension de tête effective de la couche.
    pub fn layer_head_dim(&self, layer_index: usize) -> usize {
        if self.is_gemma4_full_layer(layer_index) {
            self.global_head_dim.unwrap_or_else(|| self.head_dim())
        } else {
            self.head_dim()
        }
    }

    #[must_use]
    /// Renvoie le nombre de têtes K/V effectif de la couche.
    pub fn layer_num_key_value_heads(&self, layer_index: usize) -> usize {
        if self.is_gemma4_full_layer(layer_index) {
            self.num_global_key_value_heads
                .unwrap_or(self.num_key_value_heads)
        } else {
            self.num_key_value_heads
        }
    }

    #[must_use]
    /// Renvoie les dimensions RoPE tournées de la couche.
    pub fn layer_rope_dims(&self, layer_index: usize) -> usize {
        let head_dim = self.layer_head_dim(layer_index);
        let factor = if self.is_gemma4_full_layer(layer_index) {
            self.rope_full_partial_rotary_factor
                .or(self.partial_rotary_factor)
        } else if self.is_gemma4_sliding_layer(layer_index) {
            self.rope_sliding_partial_rotary_factor
                .or(self.partial_rotary_factor)
        } else {
            self.partial_rotary_factor
        };
        factor.map_or(head_dim, |value| (head_dim as f32 * value) as usize)
    }

    #[must_use]
    /// Indique si l'activation MLP est un GeLU tanh-approché (Gemma).
    pub fn uses_gelu_tanh(&self) -> bool {
        self.hidden_activation
            .as_deref()
            .or(self.hidden_act.as_deref())
            .is_some_and(|act| act.contains("gelu"))
    }

    #[must_use]
    /// Renvoie l'échelle multiplicative des positions RoPE (`1/factor`), ou `None`.
    ///
    /// Seul le type `linear` étire les positions (`scale = 1/factor`, comme le
    /// `initialize_rope` de mlx_lm — Gemma 3 ≥4B : linear ×8 des couches
    /// globales). `default`, `null`, absent ou facteur invalide → `None`
    /// (positions brutes, byte-identique) ; les autres types (yarn, llama3…)
    /// renvoient aussi `None`, leur refus éventuel est porté par le chargeur.
    pub fn rope_position_scale(&self) -> Option<f32> {
        let scaling = self.rope_scaling.as_ref()?;
        if scaling.scaling_type() != "linear" {
            return None;
        }
        scaling
            .factor
            .filter(|factor| factor.is_finite() && *factor > 0.0)
            .map(|factor| 1.0 / factor)
    }

    #[must_use]
    /// Renvoie le facteur d'échelle des embeddings d'entrée (`√hidden` pour Gemma).
    ///
    /// L'échelle est arrondie en bf16 comme dans les implémentations de référence
    /// (HF et mlx_lm castent `hidden_size**0.5` dans le dtype bf16 du modèle) :
    /// c'est la constante vue à l'entraînement, pas le `√hidden` f32 exact.
    pub fn embed_scale(&self) -> Option<f32> {
        self.is_gemma()
            .then(|| bf16_round((self.hidden_size as f32).sqrt()))
    }
}

/// Arrondit un f32 au bf16 le plus proche (round-to-nearest-even), renvoyé en f32.
fn bf16_round(value: f32) -> f32 {
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

/// Complète une config Gemma 3 avec les défauts de mlx_lm.
///
/// Les configs mlx-community des Gemma 3 multimodaux (4B/12B/27B) sont
/// minimales : leur `text_config` n'énumère que les clés hors-défaut
/// (hidden_size, intermediate_size, num_hidden_layers, rope_scaling,
/// sliding_window). mlx_lm reconstruit le reste à deux niveaux — le wrapper
/// `gemma3.py` (vocab 262208 écrasant celui du text_config, 8 têtes Q / 4 KV)
/// puis `gemma3_text.ModelArgs` (head_dim 256, rope_theta 1e6, base locale
/// 1e4, query_pre_attn_scalar 256, motif sliding 6, GeLU tanh câblé en dur).
/// Sans ces défauts, le parsing échoue (champs requis absents) ou le forward
/// diverge silencieusement (base locale, motif sliding, activation).
fn apply_gemma3_defaults(
    map: &mut serde_json::Map<String, serde_json::Value>,
    top_vocab_size: Option<serde_json::Value>,
) {
    let model_type = map.get("model_type").and_then(|v| v.as_str()).unwrap_or("");
    let multimodal = model_type == "gemma3";
    if !multimodal && model_type != "gemma3_text" {
        return;
    }
    if multimodal {
        // gemma3.py __post_init__ : le vocab du wrapper (top-level, défaut
        // 262208) ÉCRASE celui du text_config ; têtes Q/KV par défaut.
        let vocab = top_vocab_size.unwrap_or_else(|| serde_json::json!(262_208));
        map.insert("vocab_size".to_string(), vocab);
        map.entry("num_attention_heads".to_string())
            .or_insert(serde_json::json!(8));
        map.entry("num_key_value_heads".to_string())
            .or_insert(serde_json::json!(4));
    }
    // gemma3_text.ModelArgs : défauts du tronc texte (valeurs du 1B) ;
    // l'activation est `gelu_approx` câblée en dur dans le MLP de mlx_lm.
    for (key, value) in [
        ("hidden_size", serde_json::json!(1152)),
        ("num_hidden_layers", serde_json::json!(26)),
        ("intermediate_size", serde_json::json!(6912)),
        ("num_attention_heads", serde_json::json!(4)),
        ("num_key_value_heads", serde_json::json!(1)),
        ("head_dim", serde_json::json!(256)),
        ("rms_norm_eps", serde_json::json!(1.0e-6)),
        ("vocab_size", serde_json::json!(262_144)),
        ("rope_theta", serde_json::json!(1_000_000.0)),
        ("rope_local_base_freq", serde_json::json!(10_000.0)),
        ("query_pre_attn_scalar", serde_json::json!(256)),
        ("sliding_window", serde_json::json!(512)),
        ("sliding_window_pattern", serde_json::json!(6)),
        ("hidden_activation", serde_json::json!("gelu_pytorch_tanh")),
    ] {
        map.entry(key.to_string()).or_insert(value);
    }
}

/// Complète une config Gemma 4 avec les défauts et alias de mlx_lm/HF.
///
/// Gemma 4 sépare les paramètres RoPE par type de couche et nomme le top-k MoE
/// `top_k_experts`. Le décodeur Saragossa garde une config normalisée plate :
/// cette fonction conserve les champs d'origine tout en ajoutant les alias
/// consommés par le loader et le forward.
fn apply_gemma4_defaults(
    map: &mut serde_json::Map<String, serde_json::Value>,
    top_vocab_size: Option<serde_json::Value>,
) {
    let model_type = map.get("model_type").and_then(|v| v.as_str()).unwrap_or("");
    if !matches!(
        model_type,
        "gemma4" | "gemma4_text" | "gemma4_unified" | "gemma4_unified_text"
    ) {
        return;
    }

    let vocab = top_vocab_size.unwrap_or_else(|| serde_json::json!(262_144));
    map.entry("vocab_size".to_string()).or_insert(vocab);
    for (key, value) in [
        ("hidden_size", serde_json::json!(1536)),
        ("num_hidden_layers", serde_json::json!(35)),
        ("intermediate_size", serde_json::json!(6144)),
        ("num_attention_heads", serde_json::json!(8)),
        ("num_key_value_heads", serde_json::json!(1)),
        ("head_dim", serde_json::json!(256)),
        ("global_head_dim", serde_json::json!(512)),
        ("rms_norm_eps", serde_json::json!(1.0e-6)),
        ("rope_theta", serde_json::json!(1_000_000.0)),
        ("rope_local_base_freq", serde_json::json!(10_000.0)),
        ("rope_full_partial_rotary_factor", serde_json::json!(0.25)),
        ("rope_sliding_partial_rotary_factor", serde_json::json!(1.0)),
        ("sliding_window", serde_json::json!(512)),
        ("sliding_window_pattern", serde_json::json!(5)),
        ("hidden_activation", serde_json::json!("gelu_pytorch_tanh")),
        ("final_logit_softcapping", serde_json::json!(30.0)),
        ("tie_word_embeddings", serde_json::json!(true)),
    ] {
        map.entry(key.to_string()).or_insert(value);
    }

    if let Some(top_k) = map.get("top_k_experts").cloned() {
        map.entry("num_experts_per_tok".to_string())
            .or_insert(top_k);
    }

    let Some(rope_parameters) = map
        .get("rope_parameters")
        .and_then(|value| value.as_object())
        .cloned()
    else {
        return;
    };
    if let Some(full) = rope_parameters
        .get("full_attention")
        .and_then(|value| value.as_object())
    {
        if let Some(theta) = full.get("rope_theta").cloned() {
            map.insert("rope_theta".to_string(), theta.clone());
            map.entry("rope_full_base_freq".to_string())
                .or_insert(theta);
        }
        if let Some(factor) = full.get("partial_rotary_factor").cloned() {
            map.entry("rope_full_partial_rotary_factor".to_string())
                .or_insert(factor);
        }
    }
    if let Some(sliding) = rope_parameters
        .get("sliding_attention")
        .and_then(|value| value.as_object())
    {
        if let Some(theta) = sliding.get("rope_theta").cloned() {
            map.entry("rope_local_base_freq".to_string())
                .or_insert(theta);
        }
        if let Some(factor) = sliding.get("partial_rotary_factor").cloned() {
            map.entry("rope_sliding_partial_rotary_factor".to_string())
                .or_insert(factor);
        }
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
    // NOTE: Pas de contrainte `hidden_size % head_dim` quand head_dim est
    // explicite : q_dim (= heads·head_dim) peut différer de hidden_size
    // (Gemma 3 1B : 4×256 = 1024 vs hidden 1152) ; les formes réelles des
    // projections sont validées par le contrat du chargeur.
    if let Some(0) = cfg.head_dim {
        return Err(InferError::Config("head_dim explicite nul".to_string()));
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

    #[test]
    fn resolves_zero_shared_expert_as_absent() {
        let raw = r#"{
            "model_type":"qwen3_moe",
            "hidden_size":2048,
            "num_hidden_layers":48,
            "num_attention_heads":32,
            "num_key_value_heads":4,
            "head_dim":128,
            "intermediate_size":5472,
            "moe_intermediate_size":768,
            "num_experts":128,
            "num_experts_per_tok":8,
            "shared_expert_intermediate_size":0,
            "rms_norm_eps":0.000001,
            "rope_theta":10000000.0,
            "vocab_size":151936
        }"#;
        let cfg: RawModelConfig = serde_json::from_str(raw).expect("invariant: JSON valide");
        let cfg = cfg.resolve().expect("invariant: config MoE valide");
        assert_eq!(cfg.shared_expert_intermediate_size, None);
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
    fn resolves_gemma3_text_config_with_local_layers() {
        // Reflet du config.json mlx-community/gemma-3-1b-it-4bit.
        let raw = r#"{
            "model_type":"gemma3_text",
            "attn_logit_softcapping":null,
            "final_logit_softcapping":null,
            "head_dim":256,
            "hidden_activation":"gelu_pytorch_tanh",
            "hidden_size":1152,
            "intermediate_size":6912,
            "num_attention_heads":4,
            "num_hidden_layers":26,
            "num_key_value_heads":1,
            "query_pre_attn_scalar":256,
            "rms_norm_eps":1e-06,
            "rope_local_base_freq":10000,
            "rope_scaling":null,
            "rope_theta":1000000,
            "sliding_window":512,
            "sliding_window_pattern":6,
            "eos_token_id":1,
            "vocab_size":262144
        }"#;
        let cfg: RawModelConfig = serde_json::from_str(raw).expect("invariant: JSON valide");
        let cfg = cfg.resolve().expect("invariant: config Gemma 3 valide");
        assert!(cfg.is_gemma());
        assert!(cfg.uses_gelu_tanh());
        // √1152 = 33.941… arrondi bf16 → 34.0 (constante vue à l'entraînement).
        assert_eq!(cfg.embed_scale(), Some(34.0));
        assert_eq!(cfg.rope_local_base_freq, Some(10_000.0));
        assert_eq!(cfg.sliding_window, Some(512));
        assert_eq!(cfg.sliding_window_pattern, Some(6));
        assert_eq!(cfg.query_pre_attn_scalar, Some(256.0));
        assert_eq!(cfg.attn_logit_softcapping, None);
        assert_eq!(cfg.final_logit_softcapping, None);
        // `rope_scaling: null` (1B) → aucune échelle de positions.
        assert_eq!(cfg.rope_scaling, None);
        assert_eq!(cfg.rope_position_scale(), None);
        assert_eq!(cfg.eos_token_ids, vec![1]);
    }

    #[test]
    fn resolves_minimal_gemma3_multimodal_config_with_mlx_defaults() {
        // Reflet exact du config.json mlx-community/gemma-3-4b-it-4bit : le
        // text_config n'énumère que les clés hors-défaut, le reste vient des
        // défauts mlx_lm (wrapper gemma3.py + gemma3_text.ModelArgs).
        let raw = r#"{
            "architectures":["Gemma3ForConditionalGeneration"],
            "boi_token_index":255999,
            "eoi_token_index":256000,
            "eos_token_id":[1,106],
            "image_token_index":262144,
            "model_type":"gemma3",
            "quantization":{"group_size":64,"bits":4},
            "text_config":{
                "hidden_size":2560,
                "intermediate_size":10240,
                "model_type":"gemma3_text",
                "num_hidden_layers":34,
                "rope_scaling":{"factor":8.0,"rope_type":"linear"},
                "sliding_window":1024
            },
            "vision_config":{"model_type":"siglip_vision_model","skip_vision":true}
        }"#;
        let cfg: RawModelConfig = serde_json::from_str(raw).expect("invariant: JSON valide");
        let cfg = cfg.resolve().expect("invariant: config 4B valide");
        assert_eq!(cfg.model_type, "gemma3");
        assert!(cfg.is_gemma());
        assert_eq!(cfg.hidden_size, 2560);
        assert_eq!(cfg.num_hidden_layers, 34);
        // Défauts du wrapper gemma3.py : vocab 262208, 8 têtes Q, 4 KV.
        assert_eq!(cfg.vocab_size, 262_208);
        assert_eq!(cfg.num_attention_heads, 8);
        assert_eq!(cfg.num_key_value_heads, 4);
        // Défauts gemma3_text.ModelArgs : head_dim, bases RoPE, motif sliding.
        assert_eq!(cfg.head_dim(), 256);
        assert_eq!(cfg.rms_norm_eps, 1.0e-6);
        assert_eq!(cfg.rope_theta, 1_000_000.0);
        assert_eq!(cfg.rope_local_base_freq, Some(10_000.0));
        assert_eq!(cfg.query_pre_attn_scalar, Some(256.0));
        assert_eq!(cfg.sliding_window, Some(1024));
        assert_eq!(cfg.sliding_window_pattern, Some(6));
        // GeLU tanh : câblé en dur par gemma3_text, aucune clé déclarée.
        assert!(cfg.uses_gelu_tanh());
        assert_eq!(cfg.rope_position_scale(), Some(0.125));
        assert_eq!(cfg.eos_token_ids, vec![1, 106]);
        assert_eq!(cfg.quantization.as_ref().and_then(|q| q.bits), Some(4));
    }

    #[test]
    fn resolves_gemma4_moe_text_config_with_layer_specific_defaults() {
        // Reflet compact du config.json mlx-community/gemma-4-26b-a4b-it-4bit.
        let raw = r#"{
            "architectures":["Gemma4ForConditionalGeneration"],
            "model_type":"gemma4",
            "text_config":{
                "model_type":"gemma4_text",
                "hidden_size":2816,
                "num_hidden_layers":30,
                "num_attention_heads":16,
                "num_key_value_heads":8,
                "num_global_key_value_heads":2,
                "head_dim":256,
                "global_head_dim":512,
                "intermediate_size":2112,
                "num_experts":128,
                "top_k_experts":8,
                "moe_intermediate_size":704,
                "rms_norm_eps":1e-06,
                "final_logit_softcapping":30.0,
                "attention_k_eq_v":true,
                "enable_moe_block":true,
                "layer_types":[
                    "sliding_attention","sliding_attention","sliding_attention",
                    "sliding_attention","sliding_attention","full_attention"
                ],
                "rope_parameters":{
                    "full_attention":{
                        "partial_rotary_factor":0.25,
                        "rope_theta":1000000.0,
                        "rope_type":"proportional"
                    },
                    "sliding_attention":{
                        "rope_theta":10000.0,
                        "rope_type":"default"
                    }
                }
            }
        }"#;
        let cfg: RawModelConfig = serde_json::from_str(raw).expect("invariant: JSON valide");
        let cfg = cfg.resolve().expect("invariant: config Gemma 4 valide");

        assert!(cfg.is_gemma());
        assert!(cfg.is_gemma4());
        assert!(cfg.uses_gelu_tanh());
        assert_eq!(cfg.vocab_size, 262_144);
        assert_eq!(cfg.num_experts_per_tok, Some(8));
        assert_eq!(cfg.top_k_experts, Some(8));
        assert_eq!(cfg.final_logit_softcapping, Some(30.0));
        assert!(cfg.attention_k_eq_v);
        assert!(cfg.enable_moe_block);
        assert!(cfg.tie_word_embeddings);
        assert_eq!(cfg.rope_theta, 1_000_000.0);
        assert_eq!(cfg.rope_local_base_freq, Some(10_000.0));
        assert_eq!(cfg.rope_full_base_freq, Some(1_000_000.0));
        assert_eq!(cfg.rope_full_partial_rotary_factor, Some(0.25));
        assert_eq!(cfg.rope_sliding_partial_rotary_factor, Some(1.0));
        assert!(cfg.is_gemma4_sliding_layer(0));
        assert!(cfg.is_gemma4_full_layer(5));
        assert_eq!(cfg.layer_head_dim(0), 256);
        assert_eq!(cfg.layer_head_dim(5), 512);
        assert_eq!(cfg.layer_num_key_value_heads(0), 8);
        assert_eq!(cfg.layer_num_key_value_heads(5), 2);
        assert_eq!(cfg.layer_rope_dims(0), 256);
        assert_eq!(cfg.layer_rope_dims(5), 128);
    }

    #[test]
    fn gemma3_wrapper_vocab_overrides_text_config_value() {
        // gemma3.py __post_init__ : le vocab du wrapper (top-level, défaut
        // 262208) écrase TOUJOURS celui du text_config.
        let raw = r#"{
            "model_type":"gemma3",
            "vocab_size":262145,
            "text_config":{
                "hidden_size":2560,
                "num_hidden_layers":34,
                "vocab_size":7
            }
        }"#;
        let cfg: RawModelConfig = serde_json::from_str(raw).expect("invariant: JSON valide");
        let cfg = cfg.resolve().expect("invariant: config valide");
        assert_eq!(cfg.vocab_size, 262_145);
    }

    #[test]
    fn gemma3_defaults_do_not_leak_outside_gemma3() {
        // Une config Qwen incomplète reste rejetée : les défauts ne
        // s'appliquent qu'aux model_type gemma3/gemma3_text.
        let raw = r#"{"model_type":"qwen3","hidden_size":1024,"num_hidden_layers":2}"#;
        let cfg: RawModelConfig = serde_json::from_str(raw).expect("invariant: JSON valide");
        assert!(cfg.resolve().is_err());

        // gemma2 (flat, complet) ne reçoit pas non plus les défauts Gemma 3.
        let raw = r#"{
            "model_type":"gemma2",
            "head_dim":256,
            "hidden_act":"gelu_pytorch_tanh",
            "hidden_size":2304,
            "intermediate_size":9216,
            "num_attention_heads":8,
            "num_hidden_layers":26,
            "num_key_value_heads":4,
            "rms_norm_eps":1e-06,
            "rope_theta":10000.0,
            "sliding_window":4096,
            "vocab_size":256000
        }"#;
        let cfg: RawModelConfig = serde_json::from_str(raw).expect("invariant: JSON valide");
        let cfg = cfg.resolve().expect("invariant: config Gemma 2 valide");
        assert_eq!(cfg.rope_local_base_freq, None);
        assert_eq!(cfg.sliding_window_pattern, None);
    }

    #[test]
    fn rope_position_scale_follows_linear_rope_scaling() {
        // Reflet du text_config mlx-community/gemma-3-4b-it-4bit (couches
        // globales : linear ×8 → positions multipliées par 1/8).
        let raw = r#"{
            "model_type":"gemma3_text",
            "head_dim":256,
            "hidden_size":2560,
            "intermediate_size":10240,
            "num_attention_heads":8,
            "num_hidden_layers":34,
            "num_key_value_heads":4,
            "rms_norm_eps":1e-06,
            "rope_local_base_freq":10000,
            "rope_scaling":{"factor":8.0,"rope_type":"linear"},
            "rope_theta":1000000,
            "sliding_window":1024,
            "sliding_window_pattern":6,
            "vocab_size":262208
        }"#;
        let cfg: RawModelConfig = serde_json::from_str(raw).expect("invariant: JSON valide");
        let cfg = cfg.resolve().expect("invariant: config Gemma 3 4B valide");
        assert_eq!(cfg.rope_position_scale(), Some(0.125));
    }

    #[test]
    fn rope_position_scale_ignores_non_linear_types() {
        // Llama 3.2 déclare un type `llama3` (non implémenté) : aucune échelle,
        // statu quo du chargeur générique (scaling ignoré hors Gemma).
        let scaling: RopeScalingConfig = serde_json::from_str(
            r#"{"factor":32.0,"high_freq_factor":4.0,"low_freq_factor":1.0,
                "original_max_position_embeddings":8192,"rope_type":"llama3"}"#,
        )
        .expect("invariant: rope_scaling llama3 parsable");
        assert_eq!(scaling.scaling_type(), "llama3");

        let mut cfg = reference_qwen_config();
        cfg.rope_scaling = Some(scaling);
        assert_eq!(cfg.rope_position_scale(), None);
    }

    #[test]
    fn rope_scaling_legacy_type_key_takes_precedence() {
        // Configs HF en migration : `type` (historique) prime sur `rope_type`,
        // comme `initialize_rope` de mlx_lm.
        let scaling: RopeScalingConfig =
            serde_json::from_str(r#"{"type":"linear","rope_type":"default","factor":4.0}"#)
                .expect("invariant: rope_scaling double clé parsable");
        assert_eq!(scaling.scaling_type(), "linear");

        let mut cfg = reference_qwen_config();
        cfg.rope_scaling = Some(scaling);
        assert_eq!(cfg.rope_position_scale(), Some(0.25));
    }

    #[test]
    fn rope_position_scale_rejects_invalid_factors() {
        let mut cfg = reference_qwen_config();
        for factor in [None, Some(0.0), Some(-8.0), Some(f32::NAN)] {
            cfg.rope_scaling = Some(RopeScalingConfig {
                rope_type: Some("linear".to_string()),
                legacy_type: None,
                factor,
            });
            assert_eq!(cfg.rope_position_scale(), None, "factor={factor:?}");
        }
    }

    fn reference_qwen_config() -> ModelConfig {
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
        cfg.resolve().expect("invariant: config Qwen valide")
    }

    #[test]
    fn parses_gemma2_duplicate_activation_keys() {
        // Les configs Gemma 2 sérialisent `hidden_act` ET `hidden_activation` :
        // deux champs distincts (un alias serde casserait en `duplicate field`).
        let raw = r#"{
            "model_type":"gemma2",
            "attn_logit_softcapping":50.0,
            "final_logit_softcapping":30.0,
            "head_dim":256,
            "hidden_act":"gelu_pytorch_tanh",
            "hidden_activation":"gelu_pytorch_tanh",
            "hidden_size":2304,
            "intermediate_size":9216,
            "num_attention_heads":8,
            "num_hidden_layers":26,
            "num_key_value_heads":4,
            "query_pre_attn_scalar":256,
            "rms_norm_eps":1e-06,
            "rope_theta":10000.0,
            "sliding_window":4096,
            "vocab_size":256000
        }"#;
        let cfg: RawModelConfig = serde_json::from_str(raw).expect("invariant: JSON valide");
        let cfg = cfg.resolve().expect("invariant: config Gemma 2 parsable");
        assert!(cfg.is_gemma());
        assert!(cfg.uses_gelu_tanh());
        assert_eq!(cfg.attn_logit_softcapping, Some(50.0));
        assert_eq!(cfg.final_logit_softcapping, Some(30.0));
    }

    #[test]
    fn uses_gelu_tanh_falls_back_to_hidden_act_key() {
        let raw = r#"{
            "model_type":"gemma",
            "hidden_act":"gelu_pytorch_tanh",
            "hidden_size":2048,
            "num_hidden_layers":18,
            "num_attention_heads":8,
            "num_key_value_heads":1,
            "head_dim":256,
            "intermediate_size":16384,
            "rms_norm_eps":0.000001,
            "rope_theta":10000.0,
            "vocab_size":256000
        }"#;
        let cfg: RawModelConfig = serde_json::from_str(raw).expect("invariant: JSON valide");
        let cfg = cfg.resolve().expect("invariant: config valide");
        assert!(cfg.uses_gelu_tanh());
    }

    #[test]
    fn embed_scale_is_none_outside_gemma() {
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
            "vocab_size":151936,
            "hidden_act":"silu"
        }"#;
        let cfg: RawModelConfig = serde_json::from_str(raw).expect("invariant: JSON valide");
        let cfg = cfg.resolve().expect("invariant: config valide");
        assert_eq!(cfg.embed_scale(), None);
        assert!(!cfg.uses_gelu_tanh());
    }

    #[test]
    fn bf16_round_matches_reference_values() {
        assert_eq!(bf16_round(1.0), 1.0);
        assert_eq!(bf16_round(33.941_125), 34.0);
        assert_eq!(bf16_round(50.596_443), 50.5);
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
