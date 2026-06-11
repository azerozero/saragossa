//! Catalogue des poids connus et de leurs clés safetensors.

use crate::{InferError, Result};
use std::collections::BTreeSet;
use std::io::Read;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightCatalog {
    keys: Vec<String>,
    weight_prefix: String,
    lm_head_prefix: String,
}

impl WeightCatalog {
    /// Construit le catalogue depuis les shards safetensors.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si un shard est illisible ou invalide.
    pub fn from_shards(shards: &[PathBuf]) -> Result<Self> {
        let mut keys = BTreeSet::new();
        for shard in shards {
            for key in read_safetensors_keys(shard)? {
                keys.insert(key);
            }
        }
        let keys = keys.into_iter().collect::<Vec<_>>();
        let (weight_prefix, lm_head_prefix) = detect_weight_prefix(&keys);
        Ok(Self {
            keys,
            weight_prefix,
            lm_head_prefix,
        })
    }

    #[must_use]
    pub fn keys(&self) -> &[String] {
        &self.keys
    }

    /// Renvoie le préfixe des poids du modèle.
    #[must_use]
    pub fn weight_prefix(&self) -> &str {
        &self.weight_prefix
    }

    /// Renvoie le préfixe de la tête LM.
    #[must_use]
    pub fn lm_head_prefix(&self) -> &str {
        &self.lm_head_prefix
    }

    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.keys.len()
    }

    #[must_use]
    pub fn contains(&self, key: &str) -> bool {
        self.keys
            .binary_search_by(|probe| probe.as_str().cmp(key))
            .is_ok()
    }
}

/// Lit les clés d'un fichier safetensors sans charger les poids.
///
/// # Errors
///
/// Renvoie une erreur si le fichier ou son header JSON est invalide.
pub fn read_safetensors_keys(path: impl AsRef<Path>) -> Result<Vec<String>> {
    let path = path.as_ref();
    let mut file = std::fs::File::open(path).map_err(|source| InferError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut len_bytes = [0_u8; 8];
    file.read_exact(&mut len_bytes)
        .map_err(|source| InferError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    let header_len = u64::from_le_bytes(len_bytes);
    if header_len == 0 || header_len > 128 * 1024 * 1024 {
        return Err(InferError::SafetensorsHeader {
            path: path.to_path_buf(),
            message: format!("taille header invalide: {header_len}"),
        });
    }
    let header_len = usize::try_from(header_len).map_err(|_| InferError::SafetensorsHeader {
        path: path.to_path_buf(),
        message: format!("taille header non représentable: {header_len}"),
    })?;
    let mut header = vec![0_u8; header_len];
    file.read_exact(&mut header)
        .map_err(|source| InferError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    let json: serde_json::Value =
        serde_json::from_slice(&header).map_err(|source| InferError::Json {
            path: path.to_path_buf(),
            source,
        })?;
    let object = json
        .as_object()
        .ok_or_else(|| InferError::SafetensorsHeader {
            path: path.to_path_buf(),
            message: "header JSON non objet".to_string(),
        })?;
    let mut keys = object
        .keys()
        .filter(|key| key.as_str() != "__metadata__")
        .cloned()
        .collect::<Vec<_>>();
    keys.sort();
    Ok(keys)
}

fn detect_weight_prefix(keys: &[String]) -> (String, String) {
    let has = |key: &str| {
        keys.binary_search_by(|probe| probe.as_str().cmp(key))
            .is_ok()
    };
    if has("language_model.model.embed_tokens.weight")
        || has("language_model.model.layers.0.input_layernorm.weight")
    {
        (
            "language_model.model.".to_string(),
            "language_model.lm_head.".to_string(),
        )
    } else if has("model.language_model.embed_tokens.weight")
        || has("model.language_model.layers.0.input_layernorm.weight")
    {
        ("model.language_model.".to_string(), "lm_head.".to_string())
    } else if has("model.embed_tokens.weight") || has("model.layers.0.input_layernorm.weight") {
        ("model.".to_string(), "lm_head.".to_string())
    } else {
        (String::new(), String::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::{serialize, Dtype, View};
    use std::borrow::Cow;

    #[test]
    fn reads_header_keys_without_tensor_load() {
        let tmp = tempfile::NamedTempFile::new().expect("invariant: fichier temporaire");
        write_safetensors(
            tmp.path(),
            &[
                "model.embed_tokens.weight",
                "model.layers.0.input_layernorm.weight",
                "lm_head.weight",
            ],
        );

        let keys = read_safetensors_keys(tmp.path()).expect("invariant: header lisible");
        assert_eq!(keys.len(), 3);
        assert!(keys.contains(&"model.embed_tokens.weight".to_string()));
    }

    #[test]
    fn detects_qwen_weight_prefix() {
        let tmp = tempfile::NamedTempFile::new().expect("invariant: fichier temporaire");
        write_safetensors(
            tmp.path(),
            &[
                "language_model.model.embed_tokens.weight",
                "language_model.model.layers.0.input_layernorm.weight",
                "language_model.lm_head.weight",
            ],
        );
        let catalog = WeightCatalog::from_shards(&[tmp.path().to_path_buf()])
            .expect("invariant: catalog chargeable");
        assert_eq!(catalog.tensor_count(), 3);
        assert_eq!(catalog.weight_prefix(), "language_model.model.");
        assert_eq!(catalog.lm_head_prefix(), "language_model.lm_head.");
    }

    fn write_safetensors(path: &Path, names: &[&str]) {
        let tensors = names
            .iter()
            .map(|name| {
                (
                    *name,
                    F32View {
                        shape: vec![1],
                        data: 1.0_f32.to_le_bytes().to_vec(),
                    },
                )
            })
            .collect::<Vec<_>>();
        let buffer = serialize(tensors, None).expect("invariant: safetensors sérialisable");
        std::fs::write(path, buffer).expect("invariant: écriture safetensors");
    }

    struct F32View {
        shape: Vec<usize>,
        data: Vec<u8>,
    }

    impl View for F32View {
        fn dtype(&self) -> Dtype {
            Dtype::F32
        }

        fn shape(&self) -> &[usize] {
            &self.shape
        }

        fn data(&self) -> Cow<'_, [u8]> {
            Cow::Borrowed(&self.data)
        }

        fn data_len(&self) -> usize {
            self.data.len()
        }
    }
}
