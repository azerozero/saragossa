//! Sous-ensemble du `config.json` HF (`BertModel`) dont le forward a besoin.

use serde::Deserialize;

/// Config BERT minimale (le reste du `config.json` est ignoré). Les défauts
/// couvrent `multilingual-e5-small` si une clé venait à manquer (eps, pad).
#[derive(Debug, Clone, Deserialize)]
pub(super) struct BertConfig {
    /// Dimension cachée (= dimension de sortie de l'embedding). E5-small = 384.
    pub hidden_size: i32,
    /// Nombre de couches transformer. E5-small = 12.
    pub num_hidden_layers: i32,
    /// Nombre de têtes d'attention. E5-small = 12.
    pub num_attention_heads: i32,
    /// Epsilon des LayerNorm. BERT/E5 = 1e-12.
    #[serde(default = "default_eps")]
    pub layer_norm_eps: f32,
    /// Id du token de padding (E5 = 0). Sert au repli sur une entrée vide.
    #[serde(default)]
    pub pad_token_id: u32,
}

fn default_eps() -> f32 {
    1e-12
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_e5_small_config_subset() {
        // Extrait réel du config.json e5-small : les clés inconnues sont ignorées,
        // les clés absentes (eps fourni ici) prennent leur défaut documenté.
        let cfg: BertConfig = serde_json::from_str(
            r#"{
                "architectures": ["BertModel"],
                "hidden_size": 384,
                "num_hidden_layers": 12,
                "num_attention_heads": 12,
                "layer_norm_eps": 1e-12,
                "pad_token_id": 0,
                "vocab_size": 250037
            }"#,
        )
        .expect("sous-ensemble valide");
        assert_eq!(cfg.hidden_size, 384);
        assert_eq!(cfg.num_hidden_layers, 12);
        assert_eq!(cfg.num_attention_heads, 12);
        assert_eq!(cfg.pad_token_id, 0);
        assert!(cfg.layer_norm_eps > 0.0);
    }

    #[test]
    fn missing_optional_keys_take_defaults() {
        let cfg: BertConfig = serde_json::from_str(
            r#"{"hidden_size": 8, "num_hidden_layers": 1, "num_attention_heads": 2}"#,
        )
        .expect("clés optionnelles absentes");
        assert_eq!(cfg.layer_norm_eps, 1e-12);
        assert_eq!(cfg.pad_token_id, 0);
    }
}
