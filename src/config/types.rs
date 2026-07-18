use serde::{Deserialize, Deserializer};
use std::collections::HashMap;

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
    pub(super) model_type: String,
    #[serde(default)]
    pub(super) text_config: Option<serde_json::Value>,
    #[serde(default)]
    pub(super) quantization: Option<QuantConfig>,
    #[serde(default)]
    pub(super) quantization_config: Option<QuantConfig>,
    #[serde(flatten)]
    pub(super) rest: serde_json::Value,
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
