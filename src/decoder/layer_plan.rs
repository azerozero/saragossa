//! Décisions par couche figées au chargement, sans valider les poids.
//!
//! Les dimensions absentes restent optionnelles : l'assemblage garde son
//! inférence depuis Q et les chemins résidents produisent leurs erreurs au
//! même endroit qu'avant. Les classifications assemblée/résidente sont distinctes.

use super::*;
use std::fmt;
use std::ops::Deref;

#[derive(Clone)]
pub(crate) struct ModelPlan {
    pub(crate) layers: Vec<LayerPlan>,
    pub(crate) fingerprint: String,
    pub(crate) attention_sinks: bool,
}

// Le fingerprint ne participe pas à son propre Debug : le hash est celui
// du plan complet affiché, sans dépendance circulaire ni état provisoire.
impl fmt::Debug for ModelPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ModelPlan")
            .field("layers", &self.layers)
            .field("attention_sinks", &self.attention_sinks)
            .finish()
    }
}

impl ModelPlan {
    pub(crate) fn build(config: &CausalDecoderConfig) -> Self {
        let mut plan = Self {
            layers: (0..config.num_hidden_layers)
                .map(|index| LayerPlan::build(config, index))
                .collect(),
            fingerprint: String::new(),
            attention_sinks: config.attention_sinks,
        };
        plan.fingerprint = format!("{:x}", md5::compute(format!("{plan:?}")));
        plan
    }
}

#[derive(Clone, Debug)]
pub(crate) struct LayerPlan {
    pub(crate) index: usize,
    pub(crate) mixer: MixerPlan,
    pub(crate) resident_full: bool,
    pub(crate) dense_mlp_expected: Option<bool>,
    // Le résident utilise les dimensions configurées même si l'intervalle
    // choisit Linear à l'assemblage (configuration Gemma 4 contradictoire).
    pub(crate) resident: FullPlan,
}

#[derive(Clone, Debug)]
pub(crate) enum MixerPlan {
    Full(FullPlan),
    Linear,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct FullPlan {
    pub(crate) head_dim: Option<usize>,
    pub(crate) kv_heads: usize,
    pub(crate) kv_dim: Option<usize>,
    pub(crate) rope_dims: Option<usize>,
    // None conserve le repli sur le head_dim inféré lors de l'assemblage.
    pub(crate) rope_frequency_dim: Option<usize>,
    pub(crate) rope_theta_override: Option<f32>,
    pub(crate) rope_position_scale: Option<f32>,
    pub(crate) sliding_window: Option<usize>,
    pub(crate) no_rope: bool,
    pub(crate) v_shared_with_k: bool,
    pub(crate) gemma_full: bool,
    pub(crate) gemma_sliding: bool,
    pub(crate) local_sliding: bool,
    pub(crate) q_heads: usize,
    pub(crate) query_pre_attn_scalar: Option<f32>,
    pub(crate) attn_output_gate: bool,
}

impl LayerPlan {
    pub(crate) fn build(config: &CausalDecoderConfig, index: usize) -> Self {
        let resolver = LayerResolver(config);
        let head_dim = resolver.layer_head_dim(index).ok();
        let kv_heads = resolver.layer_num_key_value_heads(index);
        let gemma_full = resolver.is_gemma4_full_layer(index);
        let gemma_sliding = resolver.is_gemma4_sliding_layer(index);
        // Résout uniquement le choix d'option ; unwrap_or(head_dim) reste
        // différé jusqu'à ce que la dimension (configurée ou inférée) existe.
        let rope_dims = if gemma_full {
            config.rope_full_dims
        } else if gemma_sliding {
            config.rope_sliding_dims
        } else {
            config.rope_dims
        };
        let full = FullPlan {
            head_dim,
            kv_heads,
            kv_dim: head_dim.and_then(|dim| kv_heads.checked_mul(dim)),
            rope_dims,
            rope_frequency_dim: if gemma_full { None } else { rope_dims },
            rope_theta_override: resolver.layer_rope_theta_override(index),
            rope_position_scale: resolver.layer_rope_position_scale(index),
            sliding_window: resolver.layer_sliding_window(index),
            no_rope: resolver.is_no_rope_layer(index),
            v_shared_with_k: config.attention_k_eq_v && gemma_full,
            gemma_full,
            gemma_sliding,
            local_sliding: resolver.is_local_sliding_layer(index),
            q_heads: config.num_attention_heads,
            query_pre_attn_scalar: config.query_pre_attn_scalar,
            attn_output_gate: config.attn_output_gate,
        };
        Self {
            index,
            mixer: if resolver.is_full_attention_layer(index) {
                MixerPlan::Full(full)
            } else {
                MixerPlan::Linear
            },
            resident_full: resolver.is_resident_full_attention_layer(index),
            dense_mlp_expected: (!config.mlp_only_layers.is_empty())
                .then(|| config.mlp_only_layers.contains(&index)),
            resident: full,
        }
    }

    pub(crate) fn is_full(&self) -> bool {
        matches!(self.mixer, MixerPlan::Full(_))
    }
}

impl FullPlan {
    pub(crate) fn head_dim(&self) -> Result<usize> {
        self.head_dim
            .ok_or_else(|| InferError::Dimension("head_dim manquant".to_string()))
    }

    pub(crate) fn rope_dims(&self, head_dim: usize) -> usize {
        self.rope_dims.unwrap_or(head_dim)
    }

    pub(crate) fn rope_frequency_dim(&self, head_dim: usize) -> usize {
        self.rope_frequency_dim.unwrap_or(head_dim)
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) fn kv_dim(&self) -> Result<usize> {
        self.head_dim()?;
        self.kv_dim
            .ok_or_else(|| InferError::Dimension("kv_dim résident déborde".to_string()))
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) fn resident_full_attn_layer_dims(
        &self,
        hidden: usize,
        position: usize,
        eps: f32,
        theta: f32,
    ) -> Result<FullAttnLayerDims> {
        let head_dim = self.head_dim()?;
        Ok(FullAttnLayerDims {
            hidden,
            q_heads: self.q_heads,
            kv_heads: self.kv_heads,
            head_dim,
            attn_scalar: self.query_pre_attn_scalar.unwrap_or(head_dim as f32),
            // NoPE garde la QK-RMSNorm du kernel fusionné et saute la rotation.
            rope_dims: if self.no_rope {
                0
            } else {
                self.rope_dims(head_dim)
            },
            rope_frequency_dim: self.rope_frequency_dim(head_dim),
            position,
            window_start: 0,
            eps,
            theta: self.rope_theta_override.unwrap_or(theta),
            attn_output_gate: self.attn_output_gate,
        })
    }
}

// Résolveur transitoire au chargement : les corps historiques restent ici
// pour conserver exactement leurs gardes, priorités et replis.
struct LayerResolver<'a>(&'a CausalDecoderConfig);
impl Deref for LayerResolver<'_> {
    type Target = CausalDecoderConfig;
    fn deref(&self) -> &Self::Target {
        self.0
    }
}
impl LayerResolver<'_> {
    fn layer_kind(&self, layer_index: usize) -> Option<&str> {
        self.layer_types.get(layer_index).map(String::as_str)
    }

    fn is_gemma4_full_layer(&self, layer_index: usize) -> bool {
        // GARDE: `full_attention` est aussi un type de couche de qwen3_5_moe
        // (Qwen3.5/3.6 hybride). Sans le drapeau `is_gemma4`, on traitait ces
        // couches comme Gemma → RoPE pleine (`rope_dims = head_dim`) au lieu du
        // RoPE partiel correct, corrompant Q/K et donc l'attention.
        self.is_gemma4 && self.layer_kind(layer_index) == Some("full_attention")
    }

    fn is_gemma4_sliding_layer(&self, layer_index: usize) -> bool {
        self.is_gemma4 && self.layer_kind(layer_index) == Some("sliding_attention")
    }

    fn is_full_attention_layer(&self, layer_index: usize) -> bool {
        match self.full_attention_interval {
            Some(interval) if interval > 0 => (layer_index + 1) % interval == 0,
            _ => true,
        }
    }

    fn is_resident_full_attention_layer(&self, layer_index: usize) -> bool {
        if self.is_gemma4 {
            self.is_gemma4_full_layer(layer_index) || self.is_gemma4_sliding_layer(layer_index)
        } else {
            self.is_full_attention_layer(layer_index)
        }
    }

    /// Indique si une couche est locale (sliding-window) dans le motif Gemma 3.
    ///
    /// Gemma 3 alterne couches locales et globales : une couche globale toutes
    /// les `sliding_window_pattern` couches (la `pattern`-ième), locales sinon.
    /// Sans motif déclaré, aucune couche n'est locale.
    fn is_local_sliding_layer(&self, layer_index: usize) -> bool {
        if !self.layer_types.is_empty() {
            // NOTE: Hors Gemma 2/3, l'étiquette suffit (flint) ; Gemma 4 passe
            // par la même étiquette. Seul Gemma 3 garde son motif historique.
            return (self.is_gemma4 || self.sliding_from_layer_types)
                && self.layer_kind(layer_index) == Some("sliding_attention");
        }
        matches!(
            self.sliding_window_pattern,
            Some(pattern) if pattern > 0 && (layer_index + 1) % pattern != 0
        )
    }

    fn layer_head_dim(&self, layer_index: usize) -> Result<usize> {
        let head_dim = if self.is_gemma4_full_layer(layer_index) {
            self.global_head_dim.or(self.head_dim)
        } else {
            self.head_dim
        };
        head_dim.ok_or_else(|| InferError::Dimension("head_dim manquant".to_string()))
    }

    fn layer_num_key_value_heads(&self, layer_index: usize) -> usize {
        if self.is_gemma4_full_layer(layer_index) {
            self.num_global_key_value_heads
                .unwrap_or(self.num_key_value_heads)
        } else {
            self.num_key_value_heads
        }
    }

    fn layer_rope_dims(&self, layer_index: usize, head_dim: usize) -> usize {
        if self.is_gemma4_full_layer(layer_index) {
            self.rope_full_dims.unwrap_or(head_dim)
        } else if self.is_gemma4_sliding_layer(layer_index) {
            self.rope_sliding_dims.unwrap_or(head_dim)
        } else {
            self.rope_dims.unwrap_or(head_dim)
        }
    }

    /// Indique si la couche saute le RoPE (NoPE) en gardant la QK-RMSNorm.
    fn is_no_rope_layer(&self, layer_index: usize) -> bool {
        self.no_rope_layers.contains(&layer_index)
    }

    fn layer_rope_frequency_dim(&self, layer_index: usize, head_dim: usize) -> usize {
        if self.is_gemma4_full_layer(layer_index) {
            head_dim
        } else {
            self.layer_rope_dims(layer_index, head_dim)
        }
    }

    /// Renvoie la base RoPE locale d'une couche Gemma 3, ou `None` (base globale).
    ///
    /// Les couches locales utilisent `rope_local_base_freq`, les globales
    /// `rope_theta`. Hors Gemma 3 renvoie `None` → la base unique `rope_theta`
    /// s'applique (byte-identique).
    fn layer_rope_theta_override(&self, layer_index: usize) -> Option<f32> {
        if self.is_gemma4_full_layer(layer_index) {
            return self.rope_full_base_freq.or(self.rope_theta);
        }
        self.rope_local_base_freq
            .filter(|_| self.is_local_sliding_layer(layer_index))
    }

    /// Renvoie la fenêtre d'attention d'une couche locale Gemma 3, ou `None`.
    ///
    /// Seules les couches locales du motif `sliding_window_pattern` masquent
    /// leur attention aux `sliding_window` derniers tokens ; les couches
    /// globales (et tous les modèles non-Gemma) attendent sur tout le contexte.
    fn layer_sliding_window(&self, layer_index: usize) -> Option<usize> {
        self.sliding_window
            .filter(|window| *window > 0 && self.is_local_sliding_layer(layer_index))
    }

    /// Renvoie l'échelle des positions RoPE d'une couche, ou `None` (positions brutes).
    ///
    /// Gemma 3 ≥4B n'étire que les couches globales (linear ×8) : les couches
    /// locales du motif sliding gardent leurs positions brutes sur
    /// `rope_local_base_freq`, comme `initialize_rope` de mlx_lm. Sans motif
    /// sliding (linear historique type Llama 2), l'échelle s'applique partout.
    fn layer_rope_position_scale(&self, layer_index: usize) -> Option<f32> {
        if !self.layer_types.is_empty() {
            return None;
        }
        self.rope_position_scale
            .filter(|_| !self.is_local_sliding_layer(layer_index))
    }
}
