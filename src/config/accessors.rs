use super::resolve::bf16_round;
use super::types::{ModelConfig, RawModelConfig, RopeScalingConfig};
use crate::{InferError, Result};
use std::path::Path;

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
