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
    /// Tête lue dans un shard du trunk (pas de sidecar) : ses tenseurs sont
    /// déjà comptés par le catalogue.
    pub embedded: bool,
}

impl MtpWeightsInfo {
    #[must_use]
    pub fn is_available(&self) -> bool {
        self.has_fc_weight && self.has_mtp_tensors
    }
}

impl ModelAssets {
    pub fn load_local(model_dir: impl AsRef<Path>) -> Result<Self> {
        Self::load_local_with_mtp(
            model_dir.as_ref(),
            crate::runtime_flags::mtp_sidecar_override(),
        )
    }

    /// [`Self::load_local`] avec l'override MTP explicite : les tests y passent
    /// `None` pour ne pas dépendre de l'environnement des oracles.
    fn load_local_with_mtp(model_dir: &Path, sidecar_override: Option<PathBuf>) -> Result<Self> {
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
        let mtp = detect_mtp_weights(model_dir, &raw_config_json, sidecar_override, &shards)?;

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
        let ids = self.tokenizer.encode(prompt)?;
        self.reject_media_tokens(&ids)?;
        Ok(ids)
    }

    /// Encode un prompt brut en activant les tokens spéciaux (BOS) du modèle.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le tokenizer rejette le prompt.
    pub fn encode_prompt_with_special(&self, prompt: &str) -> Result<Vec<u32>> {
        let ids = self.tokenizer.encode_with_special_tokens(prompt)?;
        self.reject_media_tokens(&ids)?;
        Ok(ids)
    }

    fn reject_media_tokens(&self, ids: &[u32]) -> Result<()> {
        for marker in ["<|image_pad|>", "<|video_pad|>"] {
            if self
                .tokenizer
                .token_to_id(marker)
                .is_some_and(|id| ids.contains(&id))
            {
                crate::qwen_loader::vision::require_implementation(&self.catalog)?;
            }
        }
        Ok(())
    }

    pub fn decode_tokens(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.tokenizer.decode(ids, skip_special_tokens)
    }

    #[must_use]
    pub fn stop_token_ids(&self) -> Vec<usize> {
        let mut ids = self.config.eos_token_ids.clone();
        // Marqueurs de fin de tour par famille : Qwen (ChatML) et Gemma
        // (`<end_of_turn>`, absent des eos_token_ids du 1B). Les tokenizers
        // des autres familles ne connaissent pas ces tokens → liste inchangée.
        for token in [
            "<|endoftext|>",
            "<|im_end|>",
            "<|endofprompt|>",
            crate::chat_template::GEMMA_END_OF_TURN,
            crate::chat_template::GEMMA4_END_OF_TURN,
        ] {
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

fn detect_mtp_weights(
    model_dir: &Path,
    config: &serde_json::Value,
    sidecar_override: Option<PathBuf>,
    shards: &[PathBuf],
) -> Result<MtpWeightsInfo> {
    // Tête posée hors du répertoire du modèle (dossier en lecture seule) :
    // l'override explicite remplace la détection et ne se replie jamais
    // silencieusement sur un sidecar local.
    if let Some(path) = sidecar_override {
        require_file(&path, "sidecar MTP (SARAGOSSA_RUST_MTP_SIDECAR)")?;
        let info = mtp_weights_info(path)?;
        if !info.is_available() {
            // Sans `mtp.fc.weight` ni tenseur `mtp.*`, le MTP disparaîtrait en
            // silence derrière un flag explicite.
            return Err(InferError::MissingWeight(format!(
                "sidecar MTP explicite incomplet (mtp.fc.weight et mtp.* requis) : {}",
                info.path
                    .as_deref()
                    .map_or_else(String::new, |p| p.display().to_string())
            )));
        }
        return Ok(info);
    }
    for path in mtp_sidecar_candidates(model_dir, config) {
        if path.is_file() {
            return mtp_weights_info(path);
        }
    }
    embedded_mtp_weights(shards)
}

/// Tête MTP embarquée dans les shards (checkpoint HF bf16 sans sidecar) : le
/// shard qui porte toutes les clés `mtp.*` sert de sidecar, seuls ses tenseurs
/// MTP sont lus à leur offset. Les clés nues (`layers.*`) ne comptent pas ici,
/// elles désignent le trunk.
fn embedded_mtp_weights(shards: &[PathBuf]) -> Result<MtpWeightsInfo> {
    let mut found: Option<(PathBuf, Vec<String>)> = None;
    for shard in shards {
        let keys: Vec<String> = read_safetensors_keys(shard)?
            .iter()
            .filter(|key| key.starts_with("mtp.") || key.starts_with("language_model.mtp."))
            .map(|key| normalize_mtp_sidecar_key(key))
            .collect();
        if keys.is_empty() {
            continue;
        }
        if found.is_some() {
            // ponytail: une tête répartie sur plusieurs shards reste non prise
            // en charge (LazySidecar lit un seul fichier).
            return Ok(MtpWeightsInfo::default());
        }
        found = Some((shard.clone(), keys));
    }
    Ok(
        found.map_or_else(MtpWeightsInfo::default, |(path, keys)| MtpWeightsInfo {
            path: Some(path),
            tensor_count: keys.len(),
            has_fc_weight: keys.iter().any(|key| key == "mtp.fc.weight"),
            has_mtp_tensors: true,
            embedded: true,
        }),
    )
}

fn mtp_weights_info(path: PathBuf) -> Result<MtpWeightsInfo> {
    let keys = read_safetensors_keys(&path)?;
    let normalized = keys
        .iter()
        .map(|key| normalize_mtp_sidecar_key(key))
        .collect::<Vec<_>>();
    let has_fc_weight = normalized.iter().any(|key| key == "mtp.fc.weight");
    let has_mtp_tensors = normalized.iter().any(|key| key.starts_with("mtp."));
    Ok(MtpWeightsInfo {
        path: Some(path),
        tensor_count: keys.len(),
        has_fc_weight,
        has_mtp_tensors,
        embedded: false,
    })
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
    fn refuses_image_and_video_placeholders_in_both_encoders() -> Result<()> {
        let tmp = tempfile::tempdir().expect("invariant: tempdir");
        let path = tmp.path().join("tokenizer.json");
        save_test_tokenizer(&path);
        let mut tokenizer = tokenizers::Tokenizer::from_file(&path)
            .expect("invariant: tokenizer synthétique valide");
        tokenizer
            .add_special_tokens([
                tokenizers::AddedToken::from("<|image_pad|>", true),
                tokenizers::AddedToken::from("<|video_pad|>", true),
            ])
            .expect("invariant: marqueurs synthétiques valides");
        tokenizer
            .save(&path, false)
            .expect("invariant: tokenizer sérialisable");
        let assets = ModelAssets {
            model_dir: tmp.path().into(),
            config: crate::qwen_loader::test_fixtures::test_config(),
            tokenizer: RustTokenizer::from_file(path)?,
            catalog: WeightCatalog::from_shards(&[])?,
            shards: Vec::new(),
            mtp: Default::default(),
        };
        assert!(assets.encode_prompt("bonjour reti").is_ok());
        for marker in ["<|image_pad|>", "<|video_pad|>"] {
            assert!(assets.encode_prompt(marker).is_err());
            assert!(assets.encode_prompt_with_special(marker).is_err());
        }
        Ok(())
    }

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

        let assets = ModelAssets::load_local_with_mtp(tmp.path(), None)
            .expect("invariant: assets chargeables");
        assert_eq!(assets.config.hidden_size, 4);
        assert_eq!(assets.shards.len(), 1);
        assert_eq!(assets.catalog.tensor_count(), 1);
        assert!(assets.mtp.is_available());
        assert_eq!(assets.mtp.tensor_count, 1);
        assert_eq!(
            assets
                .encode_prompt("bonjour reti")
                .expect("invariant: encode"),
            vec![1, 2]
        );
        assert_eq!(assets.stop_token_ids(), vec![2]);
    }

    #[test]
    fn stop_token_ids_include_gemma_end_of_turn() {
        let tmp = tempfile::tempdir().expect("invariant: tempdir");
        // Config Gemma 1B-like : eos déclaré = [1] seulement, le tokenizer
        // porte <end_of_turn> → la liste des stops doit l'ajouter.
        write_config(
            tmp.path(),
            r#"{
                "model_type":"gemma3_text",
                "hidden_size":4,
                "num_hidden_layers":1,
                "num_attention_heads":2,
                "num_key_value_heads":2,
                "intermediate_size":8,
                "rms_norm_eps":0.000001,
                "rope_theta":10000.0,
                "vocab_size":4,
                "eos_token_id":1
            }"#,
        );
        save_gemma_test_tokenizer(&tmp.path().join("tokenizer.json"));
        write_safetensors(&tmp.path().join("model.safetensors"), "dummy.weight");

        let assets = ModelAssets::load_local_with_mtp(tmp.path(), None)
            .expect("invariant: assets chargeables");
        assert_eq!(assets.stop_token_ids(), vec![1, 3]);
    }

    fn save_gemma_test_tokenizer(path: &Path) {
        use tokenizers::models::wordlevel::WordLevel;
        use tokenizers::pre_tokenizers::whitespace::Whitespace;
        use tokenizers::Tokenizer;

        let vocab_path = path.with_file_name("gemma-test-vocab.json");
        std::fs::write(
            &vocab_path,
            r#"{"<unk>":0,"<eos>":1,"bonjour":2,"<end_of_turn>":3}"#,
        )
        .expect("invariant: écriture vocab tokenizer");
        let vocab_path = vocab_path
            .to_str()
            .expect("invariant: chemin vocab UTF-8")
            .to_string();
        let model = WordLevel::from_file(&vocab_path, "<unk>".to_string())
            .expect("invariant: modèle WordLevel valide");
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(Whitespace));
        tokenizer
            .save(path, true)
            .expect("invariant: sauvegarde tokenizer");
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

        let assets = ModelAssets::load_local_with_mtp(tmp.path(), None)
            .expect("invariant: assets chargeables");

        assert!(!assets.mtp.is_available());
        assert!(assets.mtp.has_mtp_tensors);
        assert!(!assets.mtp.has_fc_weight);
    }

    #[test]
    fn explicit_mtp_sidecar_replaces_detection_and_must_exist() {
        let model = tempfile::tempdir().expect("invariant: tempdir");
        let outside = tempfile::tempdir().expect("invariant: tempdir");
        write_safetensors(
            &model.path().join("mtp.safetensors"),
            "layers.0.input_layernorm.weight",
        );
        let explicit = outside.path().join("tete.safetensors");
        write_safetensors(&explicit, "mtp.fc.weight");
        let config = serde_json::json!({});

        let info = super::detect_mtp_weights(model.path(), &config, Some(explicit.clone()), &[])
            .expect("invariant: sidecar explicite lisible");
        assert_eq!(info.path.as_deref(), Some(explicit.as_path()));
        assert!(info.is_available());

        let foreign = outside.path().join("autre.safetensors");
        write_safetensors(&foreign, "dummy.weight");
        assert!(matches!(
            super::detect_mtp_weights(model.path(), &config, Some(foreign), &[]),
            Err(InferError::MissingWeight(_))
        ));

        // Tenseur mtp.* présent mais sans `fc` : la tête serait inutilisable.
        let without_fc = outside.path().join("sans-fc.safetensors");
        write_safetensors(&without_fc, "mtp.norm.weight");
        assert!(matches!(
            super::detect_mtp_weights(model.path(), &config, Some(without_fc), &[]),
            Err(InferError::MissingWeight(_))
        ));

        let missing = outside.path().join("absent.safetensors");
        assert!(matches!(
            super::detect_mtp_weights(model.path(), &config, Some(missing), &[]),
            Err(InferError::MissingArtifact { .. })
        ));
    }

    #[test]
    fn embedded_mtp_head_in_trunk_shard_is_detected_without_sidecar() {
        let tmp = tempfile::tempdir().expect("invariant: tempdir");
        let trunk = tmp.path().join("model-00001-of-00002.safetensors");
        let last = tmp.path().join("model-00002-of-00002.safetensors");
        write_safetensors(&trunk, "model.layers.0.input_layernorm.weight");
        write_safetensors_keys(
            &last,
            &["lm_head.weight", "mtp.fc.weight", "mtp.norm.weight"],
        );
        let config = serde_json::json!({});
        let shards = vec![trunk.clone(), last.clone()];

        let info = super::detect_mtp_weights(tmp.path(), &config, None, &shards)
            .expect("invariant: shards lisibles");
        assert_eq!(info.path.as_deref(), Some(last.as_path()));
        assert!(info.embedded && info.is_available());
        assert_eq!(info.tensor_count, 2);

        // Ni sidecar ni clé `mtp.*` : tête absente, l'appelant avertit.
        let info = super::detect_mtp_weights(tmp.path(), &config, None, &shards[..1])
            .expect("invariant: shard lisible");
        assert_eq!(info, MtpWeightsInfo::default());

        // Un sidecar présent garde la priorité sur la tête embarquée.
        write_safetensors(&tmp.path().join("mtp.safetensors"), "mtp.fc.weight");
        let info = super::detect_mtp_weights(tmp.path(), &config, None, &shards)
            .expect("invariant: sidecar lisible");
        assert!(!info.embedded && info.is_available());
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

        let err = ModelAssets::load_local_with_mtp(tmp.path(), None)
            .expect_err("invariant: shards requis");
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
        write_safetensors_keys(path, &[name]);
    }

    fn write_safetensors_keys(path: &Path, names: &[&'static str]) {
        let data = [1.0_f32]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let buffer = serialize(
            names.iter().map(|name| {
                (
                    *name,
                    F32View {
                        shape: vec![1],
                        data: data.clone(),
                    },
                )
            }),
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
