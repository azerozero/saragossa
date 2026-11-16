//! Contrat des poids vision Qwen. Aucun payload vision n'est chargé à ce stade.

use super::{TensorEntryRef, WeightCatalog};
use crate::{InferError, ModelConfig, Result};
use std::collections::{BTreeMap, HashMap, HashSet};

const PREFIXES: [&str; 3] = ["model.visual.", "vision_tower.", "visual."];

/// Reconnaît les trois espaces de noms observés dans les checkpoints Qwen.
#[must_use]
pub fn tensor_name(name: &str) -> Option<(&str, &str)> {
    PREFIXES
        .iter()
        .find_map(|prefix| name.strip_prefix(prefix).map(|suffix| (*prefix, suffix)))
}

/// Nombre de tenseurs vision, sans lire leurs payloads.
#[must_use]
pub fn tensor_count(catalog: &WeightCatalog) -> usize {
    catalog
        .keys()
        .iter()
        .filter(|name| tensor_name(name).is_some())
        .count()
}

/// Point de refus commun à toute demande de vision.
///
/// # Errors
/// Refuse toujours : le kill-switch ne peut pas rendre l'encodeur disponible.
pub fn require_implementation(catalog: &WeightCatalog) -> Result<()> {
    let enabled = crate::runtime_flags::vision_enabled()?;
    Err(InferError::Config(format!(
        "vision non implémentée : {} tenseurs vision reconnus ; {}",
        tensor_count(catalog),
        if enabled {
            "VISION=1 ne permet pas encore de traiter des médias"
        } else {
            "VISION=0 interdit les médias (chargement texte seul autorisé)"
        }
    )))
}

fn incompatible(message: impl Into<String>) -> InferError {
    InferError::Config(format!(
        "contrat vision Qwen incompatible : {}",
        message.into()
    ))
}

fn expected_shapes(raw: &serde_json::Value, mlx: bool) -> Result<BTreeMap<String, Vec<usize>>> {
    let dimension = |key: &str| -> Result<usize> {
        raw[key]
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| incompatible(format!("dimension absente ou invalide : {key}")))
    };
    let (depth, h, mid, out) = (
        dimension("depth")?,
        dimension("hidden_size")?,
        dimension("intermediate_size")?,
        dimension("out_hidden_size")?,
    );
    // Liste bornée des quatre profils audités : aucune taille non fiable ne
    // sert d'allocation ou de multiplicateur avant ce contrôle.
    if !matches!(
        (depth, h, mid, out),
        (27, 1152, 4304, 5120 | 2048 | 4096) | (24, 1024, 4096, 2560)
    ) {
        return Err(incompatible("profil non inventorié"));
    }
    for (key, expected) in [
        ("patch_size", 16),
        ("temporal_patch_size", 2),
        ("spatial_merge_size", 2),
        ("in_channels", 3),
        ("num_heads", 16),
        ("num_position_embeddings", 2304),
    ] {
        if dimension(key)? != expected {
            return Err(incompatible(key));
        }
    }
    if raw["hidden_act"] != "gelu_pytorch_tanh"
        || raw["deepstack_visual_indexes"]
            .as_array()
            .is_none_or(|v| !v.is_empty())
    {
        return Err(incompatible("activation ou DeepStack"));
    }
    if let Some(rope) = raw.get("rope_parameters") {
        if rope["rope_theta"].as_f64() != Some(10_000.0) || rope["rope_type"] != "axial" {
            return Err(incompatible("RoPE du ViT"));
        }
    }
    let mut shapes = BTreeMap::from([
        (
            "patch_embed.proj.weight".into(),
            if mlx {
                vec![h, 2, 16, 16, 3]
            } else {
                vec![h, 3, 2, 16, 16]
            },
        ),
        ("patch_embed.proj.bias".into(), vec![h]),
        ("pos_embed.weight".into(), vec![2304, h]),
        ("merger.norm.weight".into(), vec![h]),
        ("merger.norm.bias".into(), vec![h]),
        ("merger.linear_fc1.weight".into(), vec![4 * h, 4 * h]),
        ("merger.linear_fc1.bias".into(), vec![4 * h]),
        ("merger.linear_fc2.weight".into(), vec![out, 4 * h]),
        ("merger.linear_fc2.bias".into(), vec![out]),
    ]);
    for layer in 0..depth {
        for (stem, shape) in [
            ("norm1", vec![h]),
            ("norm2", vec![h]),
            ("attn.qkv", vec![3 * h, h]),
            ("attn.proj", vec![h, h]),
            ("mlp.linear_fc1", vec![mid, h]),
            ("mlp.linear_fc2", vec![h, mid]),
        ] {
            shapes.insert(format!("blocks.{layer}.{stem}.bias"), vec![shape[0]]);
            shapes.insert(format!("blocks.{layer}.{stem}.weight"), shape);
        }
    }
    Ok(shapes)
}

pub(super) fn validate_headers(
    config: &ModelConfig,
    entries: &HashMap<String, TensorEntryRef>,
) -> Result<()> {
    let enabled = crate::runtime_flags::vision_enabled()?;
    if let Err(error) = validate_supported_headers(config, entries) {
        if enabled {
            return Err(error);
        }
        // Les chemins dense et quantifié peuvent contrôler le même catalogue.
        // Une tour inutilisée ne doit ni bloquer le texte ni répéter le diagnostic.
        static WARNING: std::sync::Once = std::sync::Once::new();
        WARNING.call_once(|| {
            eprintln!("avertissement : VISION=0, tour vision ignorée : {error}");
        });
    }
    Ok(())
}

fn validate_supported_headers(
    config: &ModelConfig,
    entries: &HashMap<String, TensorEntryRef>,
) -> Result<()> {
    for name in entries.keys() {
        if name
            .split('.')
            .any(|part| matches!(part, "visual" | "vision_tower"))
            && tensor_name(name).is_none()
        {
            return Err(incompatible(format!("préfixe inconnu : {name}")));
        }
    }
    let mut vision = entries
        .iter()
        .filter_map(|(name, entry)| {
            tensor_name(name).map(|(prefix, suffix)| (prefix, suffix, entry))
        })
        .peekable();
    let Some(&(prefix, _, _)) = vision.peek() else {
        // Les exports texte seuls peuvent conserver vision_config sans poids.
        return Ok(());
    };
    let raw = config
        .vision_config
        .as_ref()
        .ok_or_else(|| incompatible("vision_config absente"))?;
    if raw["out_hidden_size"].as_u64() != Some(config.hidden_size as u64) {
        return Err(incompatible(
            "projection vision différente de la largeur texte",
        ));
    }
    let mut expected = expected_shapes(raw, prefix == "vision_tower.")?;
    let mut seen = HashSet::new();
    for (current_prefix, suffix, entry) in vision {
        if current_prefix != prefix || !seen.insert(suffix) {
            return Err(incompatible("préfixes mélangés ou tenseur dupliqué"));
        }
        let shape = expected
            .remove(suffix)
            .ok_or_else(|| incompatible(format!("tenseur inconnu : {suffix}")))?;
        if entry.entry.shape != shape || entry.entry.dtype != safetensors::Dtype::BF16 {
            return Err(incompatible(format!("forme/dtype de {suffix}")));
        }
        entry.entry.payload_len(suffix)?;
    }
    if !expected.is_empty() {
        return Err(incompatible(format!(
            "{} tenseurs manquants",
            expected.len()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen_loader::{test_fixtures::test_config, ShardTensorEntry};

    #[test]
    fn validates_every_inventoried_tensor_and_rejects_corruption() -> Result<()> {
        let manifest: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/fixtures/vision/manifest.json"))
                .map_err(|e| incompatible(e.to_string()))?;
        for checkpoint in manifest["checkpoints"]
            .as_array()
            .ok_or_else(|| incompatible("fixtures"))?
        {
            let mut config = test_config();
            config.vision_config = Some(checkpoint["vision_config"].clone());
            config.hidden_size = checkpoint["vision_config"]["out_hidden_size"]
                .as_u64()
                .ok_or_else(|| incompatible("projection"))?
                as usize;
            let mut entries = HashMap::new();
            for tensor in checkpoint["tensors"]
                .as_array()
                .ok_or_else(|| incompatible("tensors"))?
            {
                let name = tensor["name"]
                    .as_str()
                    .ok_or_else(|| incompatible("name"))?;
                let entry: ShardTensorEntry = serde_json::from_value(tensor.clone())
                    .map_err(|e| incompatible(e.to_string()))?;
                entries.insert(
                    name.into(),
                    TensorEntryRef {
                        shard_index: 0,
                        entry,
                    },
                );
            }
            validate_supported_headers(&config, &entries)?;
            config.hidden_size += 1;
            assert!(validate_supported_headers(&config, &entries).is_err());
            config.hidden_size -= 1;
            let key = format!(
                "{}patch_embed.proj.weight",
                checkpoint["prefix"]
                    .as_str()
                    .ok_or_else(|| incompatible("prefix"))?
            );
            let original = entries.remove(&key).ok_or_else(|| incompatible("patch"))?;
            assert!(validate_supported_headers(&config, &entries).is_err());
            let mut corrupt = original.clone();
            corrupt.entry.shape.swap(1, 4);
            entries.insert(key.clone(), corrupt);
            assert!(validate_supported_headers(&config, &entries).is_err());
            let mut wrong_dtype = original.clone();
            wrong_dtype.entry.dtype = safetensors::Dtype::F32;
            entries.insert(key.clone(), wrong_dtype);
            assert!(validate_supported_headers(&config, &entries).is_err());
            entries.insert(key.clone(), original.clone());
            entries.insert(format!("unknown.{key}"), original.clone());
            assert!(validate_supported_headers(&config, &entries).is_err());
            entries.remove(&format!("unknown.{key}"));
            entries.insert(key.clone(), original.clone());
            entries.insert(
                format!(
                    "{}unknown.weight",
                    checkpoint["prefix"]
                        .as_str()
                        .ok_or_else(|| incompatible("prefix"))?
                ),
                original,
            );
            assert!(validate_supported_headers(&config, &entries).is_err());
        }
        Ok(())
    }

    #[test]
    fn media_request_is_always_refused() -> Result<()> {
        let catalog = WeightCatalog::from_shards(&[])?;
        assert!(require_implementation(&catalog).is_err());
        assert_eq!(
            tensor_name("model.visual.pos_embed.weight"),
            Some(("model.visual.", "pos_embed.weight"))
        );
        assert_eq!(tensor_name("model.visualish.weight"), None);
        Ok(())
    }

    #[test]
    fn rejects_truncated_vision_payload_before_loading_weights(
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("truncated.safetensors");
        let header =
            br#"{"visual.pos_embed.weight":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        std::fs::write(&path, bytes)?;
        assert!(crate::qwen_loader::read_shard_header(&path).is_err());
        Ok(())
    }

    #[test]
    fn loader_honours_vision_flag_without_regressing_text(
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        const CHILD: &str = "SARAGOSSA_TEST_VISION_CHILD";
        if let Ok(mode) = std::env::var(CHILD) {
            let directory = tempfile::tempdir()?;
            let path = directory.path().join("model.safetensors");
            crate::qwen_loader::test_fixtures::write_safetensors(&path, "model.", "lm_head.", None);
            let shards = vec![path];
            let catalog = WeightCatalog::from_shards(&shards)?;
            let result = crate::qwen_loader::load_causal_decoder_from_shards(
                &test_config(),
                &shards,
                &catalog,
            );
            if mode == "0" {
                assert!(result.is_ok());
            } else {
                let error = result.err().ok_or("demande vision acceptée")?.to_string();
                assert!(error.contains(if mode == "1" {
                    "vision non implémentée"
                } else {
                    "booléen invalide"
                }));
            }
            return Ok(());
        }
        for mode in ["0", "1", "invalid"] {
            let output = std::process::Command::new(std::env::current_exe()?)
                .args(["--exact", "qwen_loader::vision::tests::loader_honours_vision_flag_without_regressing_text", "--nocapture"])
                .env(CHILD, mode).env("VISION", mode).output()?;
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        Ok(())
    }

    #[test]
    fn text_loader_ignores_unknown_and_quantized_vision(
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        const CHILD: &str = "SARAGOSSA_TEST_UNSUPPORTED_VISION_CHILD";
        if let Ok(mode) = std::env::var(CHILD) {
            use safetensors::{serialize, tensor::TensorView, Dtype};

            let directory = tempfile::tempdir()?;
            let text = directory.path().join("text.safetensors");
            crate::qwen_loader::test_fixtures::write_safetensors(&text, "model.", "lm_head.", None);
            let config = test_config();
            let shards = [text.clone()];
            let catalog = WeightCatalog::from_shards(&shards)?;
            let reference = if mode != "1" {
                Some(
                    crate::qwen_loader::load_causal_decoder_from_shards(
                        &config, &shards, &catalog,
                    )?
                    .next_logits(&[0, 1])?
                    .data()
                    .to_vec(),
                )
            } else {
                None
            };
            for quantized in [false, true] {
                let mut config = config.clone();
                config.vision_config = Some(serde_json::json!({
                    "depth": 1, "hidden_size": 2, "intermediate_size": 4,
                    "out_hidden_size": config.hidden_size,
                }));
                let vision = directory.path().join("vision.safetensors");
                let payload = [0_u8; 4];
                let mut tensors = vec![(
                    "vision_tower.blocks.0.attn.qkv.weight",
                    TensorView::new(
                        if quantized { Dtype::U32 } else { Dtype::BF16 },
                        vec![1, if quantized { 1 } else { 2 }],
                        &payload,
                    )?,
                )];
                if quantized {
                    for suffix in [
                        "vision_tower.blocks.0.attn.qkv.scales",
                        "vision_tower.blocks.0.attn.qkv.biases",
                    ] {
                        tensors.push((suffix, TensorView::new(Dtype::F32, vec![1, 1], &payload)?));
                    }
                }
                std::fs::write(&vision, serialize(tensors, None)?)?;
                let shards = [text.clone(), vision];
                let catalog = WeightCatalog::from_shards(&shards)?;
                for _ in 0..2 {
                    let result = crate::qwen_loader::load_causal_decoder_from_shards(
                        &config, &shards, &catalog,
                    );
                    if let Some(expected) = &reference {
                        assert_eq!(result?.next_logits(&[0, 1])?.data(), expected);
                    } else {
                        let error = result.err().ok_or("demande vision acceptée")?.to_string();
                        assert!(error.contains("vision non implémentée"));
                    }
                }
            }
            return Ok(());
        }
        for mode in ["default", "0", "1"] {
            let mut command = std::process::Command::new(std::env::current_exe()?);
            command
                .args([
                    "--exact",
                    "qwen_loader::vision::tests::text_loader_ignores_unknown_and_quantized_vision",
                    "--nocapture",
                ])
                .env(CHILD, mode)
                .env_remove("VISION");
            if mode != "default" {
                command.env("VISION", mode);
            }
            let output = command.output()?;
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(output.status.success(), "{stderr}");
            assert_eq!(
                stderr.matches("tour vision ignorée").count(),
                usize::from(mode != "1"),
                "{stderr}"
            );
        }
        Ok(())
    }
}
