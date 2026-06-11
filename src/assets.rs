//! Résolution des chemins de modèles, ressources et artefacts locaux.

use crate::{
    catalog::read_safetensors_keys, InferError, ModelConfig, Result, RustTokenizer, WeightCatalog,
};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct ModelAssets {
    /// Répertoire racine du modèle local.
    pub model_dir: PathBuf,
    /// Configuration Qwen normalisée.
    pub config: ModelConfig,
    /// Tokenizer Rust associé au modèle.
    pub tokenizer: RustTokenizer,
    /// Shards safetensors du trunk, hors sidecars MTP.
    pub shards: Vec<PathBuf>,
    /// Catalogue des poids du trunk.
    pub catalog: WeightCatalog,
    /// Informations de présence du sidecar MTP.
    pub mtp: MtpWeightsInfo,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MtpWeightsInfo {
    /// Chemin du sidecar MTP détecté.
    pub path: Option<PathBuf>,
    /// Nombre de tenseurs présents dans le sidecar.
    pub tensor_count: usize,
    /// Indique si `mtp.fc.weight` existe après normalisation des clés.
    pub has_fc_weight: bool,
    /// Indique si au moins un tenseur `mtp.*` existe.
    pub has_mtp_tensors: bool,
}

impl MtpWeightsInfo {
    #[must_use]
    pub fn is_available(&self) -> bool {
        self.has_fc_weight && self.has_mtp_tensors
    }
}

impl ModelAssets {
    pub fn load_local(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        if !model_dir.is_dir() {
            return Err(InferError::MissingArtifact {
                path: model_dir.to_path_buf(),
                what: "model_dir",
            });
        }

        let config_path = model_dir.join("config.json");
        require_file(&config_path, "config.json")?;
        let tokenizer_path = model_dir.join("tokenizer.json");
        require_file(&tokenizer_path, "tokenizer.json")?;

        let raw_config_json = read_json_value(&config_path)?;
        let config = ModelConfig::from_file(&config_path)?;
        let tokenizer = RustTokenizer::from_file(&tokenizer_path)?;
        let shards = list_safetensor_shards(model_dir, &raw_config_json)?;
        if shards.is_empty() {
            return Err(InferError::MissingArtifact {
                path: model_dir.to_path_buf(),
                what: "*.safetensors",
            });
        }
        let catalog = WeightCatalog::from_shards(&shards)?;
        let mtp = detect_mtp_weights(model_dir, &raw_config_json)?;

        Ok(Self {
            model_dir: model_dir.to_path_buf(),
            config,
            tokenizer,
            shards,
            catalog,
            mtp,
        })
    }

    pub fn encode_prompt(&self, prompt: &str) -> Result<Vec<u32>> {
        self.tokenizer.encode(prompt)
    }

    pub fn decode_tokens(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.tokenizer.decode(ids, skip_special_tokens)
    }

    #[must_use]
    pub fn stop_token_ids(&self) -> Vec<usize> {
        let mut ids = self.config.eos_token_ids.clone();
        for token in ["<|endoftext|>", "<|im_end|>", "<|endofprompt|>"] {
            if let Some(id) = self
                .tokenizer
                .token_to_id(token)
                .and_then(|id| usize::try_from(id).ok())
            {
                push_unique_id(&mut ids, id);
            }
        }
        ids
    }
}

pub fn list_safetensor_shards(
    model_dir: impl AsRef<Path>,
    raw_config_json: &serde_json::Value,
) -> Result<Vec<PathBuf>> {
    let model_dir = model_dir.as_ref();
    let mtp_sidecars = mtp_sidecar_candidates(model_dir, raw_config_json);
    let mut files = Vec::new();
    for entry in std::fs::read_dir(model_dir).map_err(|source| InferError::Io {
        path: model_dir.to_path_buf(),
        source,
    })? {
        let path = entry
            .map_err(|source| InferError::Io {
                path: model_dir.to_path_buf(),
                source,
            })?
            .path();
        if path.extension().and_then(|e| e.to_str()) == Some("safetensors")
            && !mtp_sidecars.iter().any(|sidecar| sidecar == &path)
        {
            files.push(path);
        }
    }
    files.sort();
    Ok(files)
}

fn require_file(path: &Path, what: &'static str) -> Result<()> {
    if path.is_file() {
        Ok(())
    } else {
        Err(InferError::MissingArtifact {
            path: path.to_path_buf(),
            what,
        })
    }
}

fn read_json_value(path: &Path) -> Result<serde_json::Value> {
    let file = std::fs::File::open(path).map_err(|source| InferError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::from_reader(file).map_err(|source| InferError::Json {
        path: path.to_path_buf(),
        source,
    })
}

fn mtp_sidecar_candidates(model_dir: &Path, config: &serde_json::Value) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Some(rel) = config_mtp_file(config) {
        push_unique(&mut paths, model_dir.join(rel));
    }
    for rel in [
        "mtp.safetensors",
        "mtp/weights.safetensors",
        "model-mtp.safetensors",
        "model_extra_tensors.safetensors",
    ] {
        push_unique(&mut paths, model_dir.join(rel));
    }
    paths
}

fn detect_mtp_weights(model_dir: &Path, config: &serde_json::Value) -> Result<MtpWeightsInfo> {
    for path in mtp_sidecar_candidates(model_dir, config) {
        if !path.is_file() {
            continue;
        }
        let keys = read_safetensors_keys(&path)?;
        let normalized = keys
            .iter()
            .map(|key| normalize_mtp_sidecar_key(key))
            .collect::<Vec<_>>();
        let has_fc_weight = normalized.iter().any(|key| key == "mtp.fc.weight");
        let has_mtp_tensors = normalized.iter().any(|key| key.starts_with("mtp."));
        return Ok(MtpWeightsInfo {
            path: Some(path),
            tensor_count: keys.len(),
            has_fc_weight,
            has_mtp_tensors,
        });
    }
    Ok(MtpWeightsInfo::default())
}

fn normalize_mtp_sidecar_key(key: &str) -> String {
    if let Some(rest) = key.strip_prefix("language_model.mtp.") {
        format!("mtp.{rest}")
    } else if key.starts_with("mtp.") {
        key.to_string()
    } else if is_bare_mtp_sidecar_key(key) {
        format!("mtp.{key}")
    } else {
        key.to_string()
    }
}

fn is_bare_mtp_sidecar_key(key: &str) -> bool {
    key == "fc.weight"
        || key == "fc.scales"
        || key == "fc.biases"
        || key.starts_with("layers.")
        || key.starts_with("pre_fc_norm_")
        || key.starts_with("norm.")
}

fn config_mtp_file(config: &serde_json::Value) -> Option<&str> {
    let direct = config
        .get("mlx_lm_extra_tensors")
        .and_then(|v| v.get("mtp_file"))
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty());
    direct.or_else(|| {
        config
            .get("text_config")
            .and_then(|v| v.get("mlx_lm_extra_tensors"))
            .and_then(|v| v.get("mtp_file"))
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
    })
}

fn push_unique(paths: &mut Vec<PathBuf>, path: PathBuf) {
    if !paths.iter().any(|p| p == &path) {
        paths.push(path);
    }
}

fn push_unique_id(ids: &mut Vec<usize>, id: usize) {
    if !ids.contains(&id) {
        ids.push(id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::save_test_tokenizer;
    use safetensors::{serialize, Dtype, View};
    use std::borrow::Cow;

    #[test]
    fn loads_local_assets_and_excludes_mtp_sidecars() {
        let tmp = tempfile::tempdir().expect("invariant: tempdir");
        write_config(
            tmp.path(),
            r#"{
                "model_type":"qwen3",
                "hidden_size":4,
                "num_hidden_layers":1,
                "num_attention_heads":2,
                "num_key_value_heads":2,
                "intermediate_size":8,
                "rms_norm_eps":0.000001,
                "rope_theta":10000.0,
                "vocab_size":3,
                "eos_token_id":2,
                "mlx_lm_extra_tensors":{"mtp_file":"mtp/weights.safetensors"}
            }"#,
        );
        save_test_tokenizer(&tmp.path().join("tokenizer.json"));
        write_safetensors(&tmp.path().join("model.safetensors"), "dummy.weight");
        std::fs::create_dir_all(tmp.path().join("mtp")).expect("invariant: mkdir mtp");
        write_safetensors(&tmp.path().join("mtp/weights.safetensors"), "fc.weight");

        let assets = ModelAssets::load_local(tmp.path()).expect("invariant: assets chargeables");
        assert_eq!(assets.config.hidden_size, 4);
        assert_eq!(assets.shards.len(), 1);
        assert_eq!(assets.catalog.tensor_count(), 1);
        assert!(assets.mtp.is_available());
        assert_eq!(assets.mtp.tensor_count, 1);
        assert_eq!(
            assets.encode_prompt("bonjour reti").expect("encode"),
            vec![1, 2]
        );
        assert_eq!(assets.stop_token_ids(), vec![2]);
    }

    #[test]
    fn mtp_sidecar_without_fc_weight_is_not_available() {
        let tmp = tempfile::tempdir().expect("invariant: tempdir");
        write_config(
            tmp.path(),
            r#"{
                "model_type":"qwen3",
                "hidden_size":4,
                "num_hidden_layers":1,
                "num_attention_heads":2,
                "num_key_value_heads":2,
                "intermediate_size":8,
                "rms_norm_eps":0.000001,
                "rope_theta":10000.0,
                "vocab_size":3,
                "mtp_num_hidden_layers":1
            }"#,
        );
        save_test_tokenizer(&tmp.path().join("tokenizer.json"));
        write_safetensors(&tmp.path().join("model.safetensors"), "dummy.weight");
        write_safetensors(
            &tmp.path().join("mtp.safetensors"),
            "layers.0.input_layernorm.weight",
        );

        let assets = ModelAssets::load_local(tmp.path()).expect("invariant: assets chargeables");

        assert!(!assets.mtp.is_available());
        assert!(assets.mtp.has_mtp_tensors);
        assert!(!assets.mtp.has_fc_weight);
    }

    #[test]
    fn missing_safetensors_is_explicit_error() {
        let tmp = tempfile::tempdir().expect("invariant: tempdir");
        write_config(
            tmp.path(),
            r#"{
                "model_type":"qwen3",
                "hidden_size":4,
                "num_hidden_layers":1,
                "num_attention_heads":2,
                "num_key_value_heads":2,
                "intermediate_size":8,
                "rms_norm_eps":0.000001,
                "rope_theta":10000.0,
                "vocab_size":3
            }"#,
        );
        save_test_tokenizer(&tmp.path().join("tokenizer.json"));

        let err = ModelAssets::load_local(tmp.path()).expect_err("invariant: shards requis");
        assert!(matches!(
            err,
            InferError::MissingArtifact {
                what: "*.safetensors",
                ..
            }
        ));
    }

    fn write_config(dir: &Path, json: &str) {
        std::fs::write(dir.join("config.json"), json).expect("invariant: écriture config");
    }

    fn write_safetensors(path: &Path, name: &'static str) {
        let data = [1.0_f32]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let buffer = serialize(
            [(
                name,
                F32View {
                    shape: vec![1],
                    data,
                },
            )],
            None,
        )
        .expect("invariant: safetensors sérialisable");
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
