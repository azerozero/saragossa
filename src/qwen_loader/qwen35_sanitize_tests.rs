//! Régression HF brut / MLX sanitizé, sans poids externes ni GPU.

use super::test_fixtures::{test_config, write_hybrid_safetensors};
use super::*;
use safetensors::{serialize, tensor::TensorView, SafeTensors};

const OFFSET_NORMS: &[&str] = &[
    "norm.weight",
    "layers.0.input_layernorm.weight",
    "layers.0.post_attention_layernorm.weight",
    "layers.1.input_layernorm.weight",
    "layers.1.self_attn.q_norm.weight",
    "layers.1.self_attn.k_norm.weight",
];

fn dense(shape: &[usize], values: &[f32]) -> DecoderTensor {
    DecoderTensor::Dense(
        Tensor::from_vec(shape.to_vec(), values.to_vec()).expect("invariant: tenseur synthétique"),
    )
}

fn data(tensors: &HashMap<String, DecoderTensor>, key: &str) -> Vec<f32> {
    match &tensors[key] {
        DecoderTensor::Dense(tensor) => tensor.data().to_vec(),
        _ => unreachable!("invariant: fixture dense"),
    }
}

#[test]
fn qwen35_offset_is_scoped_to_hf_layout_family_and_exact_norms() {
    for family in [
        "qwen3_5",
        "qwen3_5_text",
        "qwen3_5_moe",
        "qwen3_5_moe_text",
        "qwen3",
        "llama",
        "gemma4",
    ] {
        for hf_layout in [false, true] {
            let mut config = test_config();
            config.model_type = family.to_string();
            let conv_shape = if hf_layout { [2, 1, 2] } else { [2, 2, 1] };
            let mut tensors = HashMap::from([
                (
                    "layers.0.linear_attn.conv1d.weight".to_string(),
                    dense(&conv_shape, &[0.25, 0.5, 0.75, 1.0]),
                ),
                (
                    "layers.0.linear_attn.norm.weight".to_string(),
                    dense(&[2], &[0.5, 0.75]),
                ),
                (
                    "embed_tokens.weight".to_string(),
                    dense(&[1, 2], &[0.5, -0.25]),
                ),
                ("lm_head.weight".to_string(), dense(&[1, 2], &[0.25, 0.5])),
                (
                    "layers.0.linear_attn.A_log".to_string(),
                    dense(&[1], &[-2.0]),
                ),
                (
                    "layers.0.linear_attn.dt_bias".to_string(),
                    dense(&[1], &[0.25]),
                ),
            ]);
            for key in OFFSET_NORMS {
                tensors.insert((*key).to_string(), dense(&[2], &[0.5, -0.25]));
            }
            let before = tensors.clone();
            bake_qwen35_norm_offset(&config, &mut tensors);
            for key in tensors.keys() {
                let expected = if family.starts_with("qwen3_5")
                    && hf_layout
                    && OFFSET_NORMS.contains(&key.as_str())
                {
                    vec![1.5, 0.75]
                } else {
                    data(&before, key)
                };
                assert_eq!(
                    data(&tensors, key),
                    expected,
                    "{family}, {hf_layout}, {key}"
                );
            }
        }
    }
}

/// Réencode la fixture hybride en BF16 HF ou MLX. Les poids de norme HF
/// valent zéro, ceux de MLX un : sans sanitize, les logits HF s'annulent.
fn write_checkpoint(path: &Path, prefix: &str, head: &str, hf: bool, tied: bool) {
    let baseline = tempfile::NamedTempFile::new().expect("invariant: fichier temporaire");
    write_hybrid_safetensors(baseline.path());
    let bytes = std::fs::read(baseline.path()).expect("invariant: fixture lisible");
    let source = SafeTensors::deserialize(&bytes).expect("invariant: fixture valide");
    let mut owned = Vec::new();
    for (name, tensor) in source.tensors() {
        if tied && name == "lm_head.weight" {
            continue;
        }
        let target = name.strip_prefix("model.").unwrap_or(&name);
        let name = if target == "lm_head.weight" {
            format!("{head}weight")
        } else {
            format!("{prefix}{target}")
        };
        let mut shape = tensor.shape().to_vec();
        if hf && target.ends_with("conv1d.weight") {
            shape.swap(1, 2);
        }
        let payload = tensor
            .data()
            .chunks_exact(4)
            .flat_map(|chunk| {
                let mut value = f32::from_le_bytes(chunk.try_into().expect("invariant: f32"));
                if hf && OFFSET_NORMS.contains(&target) {
                    value -= 1.0;
                }
                ((value.to_bits() >> 16) as u16).to_le_bytes()
            })
            .collect::<Vec<_>>();
        owned.push((name, shape, payload));
    }
    // Ces sections ne doivent ni fournir les poids texte, ni déclencher
    // l'offset dans un export MLX dont le tronc est déjà sanitizé.
    for name in [
        "model.visual.norm.weight",
        "mtp.norm.weight",
        "mtp.layers.0.linear_attn.conv1d.weight",
        "model.visual.linear_attn.conv1d.weight",
    ] {
        let shape = if name.ends_with("conv1d.weight") {
            vec![2, 1, 2]
        } else {
            vec![2]
        };
        let count: usize = shape.iter().product();
        owned.push((name.to_string(), shape, vec![0; count * 2]));
    }
    let views = owned.iter().map(|(name, shape, payload)| {
        (
            name.as_str(),
            TensorView::new(Dtype::BF16, shape.clone(), payload)
                .expect("invariant: payload BF16 valide"),
        )
    });
    let output = serialize(views, None).expect("invariant: sérialisation BF16");
    std::fs::write(path, output).expect("invariant: écriture fixture");
}

#[test]
fn qwen35_hf_bf16_matches_sanitized_mlx_tensors_and_cpu_logits() -> Result<()> {
    for tied in [false, true] {
        let mut config = test_config();
        config.model_type = "qwen3_5".to_string();
        config.num_hidden_layers = 2;
        config.full_attention_interval = Some(2);
        config.attn_output_gate = Some(true);
        config.linear_num_key_heads = Some(1);
        config.linear_num_value_heads = Some(1);
        config.linear_key_head_dim = Some(2);
        config.linear_value_head_dim = Some(2);
        config.linear_conv_kernel_dim = Some(2);
        config.tie_word_embeddings = tied;
        let mut reference: Option<(HashMap<String, DecoderTensor>, Vec<f32>)> = None;
        for (prefix, head, hf) in [
            ("language_model.model.", "language_model.lm_head.", false),
            ("model.language_model.", "lm_head.", true),
            ("model.", "lm_head.", true),
            // Le préfixe seul ne prouve pas qu'un checkpoint est sanitizé.
            ("language_model.model.", "language_model.lm_head.", true),
        ] {
            let tmp = tempfile::NamedTempFile::new().expect("invariant: fichier temporaire");
            write_checkpoint(tmp.path(), prefix, head, hf, tied);
            let shards = [tmp.path().to_path_buf()];
            let catalog = WeightCatalog::from_shards(&shards)?;
            let tensors =
                load_decoder_tensors(&config, &shards, &catalog, &QwenPrefixes::detect(&catalog))?;
            assert!(!tensors
                .keys()
                .any(|key| key.contains("mtp") || key.contains("visual")));
            let decoder = load_causal_decoder_from_shards(&config, &shards, &catalog)?;
            let logits = decoder.next_logits(&[0, 1])?.data().to_vec();
            assert!(logits.iter().any(|value| value.abs() > 0.1));
            if let Some((expected, expected_logits)) = &reference {
                assert_eq!(tensors.len(), expected.len());
                for key in expected.keys() {
                    assert_eq!(data(&tensors, key), data(expected, key), "{prefix}{key}");
                }
                assert_eq!(&logits, expected_logits, "{prefix}, tied={tied}");
            } else {
                reference = Some((tensors, logits));
            }
        }
    }
    Ok(())
}
