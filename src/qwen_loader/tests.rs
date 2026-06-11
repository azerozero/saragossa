use super::test_fixtures::*;
use super::*;

use crate::{GenerationOptions, QuantConfig};
use safetensors::serialize;
use std::collections::HashMap;

#[test]
fn loads_model_prefixed_qwen_weights() {
    let tmp = tempfile::NamedTempFile::new().expect("invariant: fichier temporaire");
    write_safetensors(tmp.path(), "model.", "lm_head.", None);
    let catalog = WeightCatalog::from_shards(&[tmp.path().to_path_buf()])
        .expect("invariant: catalog chargeable");
    let model =
        load_qwen_causal_decoder_from_shards(&test_config(), &[tmp.path().to_path_buf()], &catalog)
            .expect("invariant: modèle Qwen minimal chargeable");

    let logits = model
        .next_logits(&[0])
        .expect("invariant: forward minimal valide");
    assert_eq!(logits.shape(), &[1, 3]);
}

#[test]
fn verifies_qwen_contract_from_headers() {
    let tmp = tempfile::NamedTempFile::new().expect("invariant: fichier temporaire");
    write_safetensors(tmp.path(), "model.", "lm_head.", None);
    let catalog = WeightCatalog::from_shards(&[tmp.path().to_path_buf()])
        .expect("invariant: catalog chargeable");

    let contract = verify_qwen_decoder_contract_from_shards(
        &test_config(),
        &[tmp.path().to_path_buf()],
        &catalog,
    )
    .expect("invariant: contrat Qwen minimal valide");

    assert_eq!(contract.shard_count, 1);
    assert_eq!(contract.required_specs, 8);
    assert_eq!(contract.present_specs, 8);
    assert!(contract.optional_specs > 0);
}

#[test]
fn contract_rejects_bad_header_shape() {
    let tmp = tempfile::NamedTempFile::new().expect("invariant: fichier temporaire");
    write_safetensors(
        tmp.path(),
        "model.",
        "lm_head.",
        Some((
            "model.layers.0.self_attn.q_proj.weight",
            TensorFixture::ones(vec![1, 2]),
        )),
    );
    let catalog = WeightCatalog::from_shards(&[tmp.path().to_path_buf()])
        .expect("invariant: catalog chargeable");

    let err = verify_qwen_decoder_contract_from_shards(
        &test_config(),
        &[tmp.path().to_path_buf()],
        &catalog,
    )
    .expect_err("invariant: forme q_proj invalide rejetée");

    assert!(matches!(err, InferError::Dimension(_)));
}

#[test]
fn loads_language_model_prefixed_qwen_weights() {
    let tmp = tempfile::NamedTempFile::new().expect("invariant: fichier temporaire");
    write_safetensors(
        tmp.path(),
        "language_model.model.",
        "language_model.lm_head.",
        None,
    );
    let catalog = WeightCatalog::from_shards(&[tmp.path().to_path_buf()])
        .expect("invariant: catalog chargeable");
    let model =
        load_qwen_causal_decoder_from_shards(&test_config(), &[tmp.path().to_path_buf()], &catalog)
            .expect("invariant: variante language_model chargeable");

    let generated = model
        .generate_greedy(&[0], 1)
        .expect("invariant: génération minimale valide");
    assert_eq!(generated.len(), 1);
}

#[test]
fn loads_grouped_query_attention_shapes() {
    let tmp = tempfile::NamedTempFile::new().expect("invariant: fichier temporaire");
    write_gqa_safetensors(tmp.path());
    let catalog = WeightCatalog::from_shards(&[tmp.path().to_path_buf()])
        .expect("invariant: catalog chargeable");
    let mut config = test_config();
    config.hidden_size = 4;
    config.num_attention_heads = 2;
    config.num_key_value_heads = 1;
    config.head_dim = Some(2);

    let model =
        load_qwen_causal_decoder_from_shards(&config, &[tmp.path().to_path_buf()], &catalog)
            .expect("invariant: GQA supporté quand les formes correspondent");
    let logits = model
        .next_logits(&[0, 1])
        .expect("invariant: forward GQA valide");

    assert_eq!(logits.shape(), &[1, 3]);
    assert!(logits.data().iter().all(|value| value.is_finite()));
}

#[test]
fn loads_partial_rope_when_rotary_dims_are_even() {
    let tmp = tempfile::NamedTempFile::new().expect("invariant: fichier temporaire");
    write_head4_safetensors(tmp.path());
    let catalog = WeightCatalog::from_shards(&[tmp.path().to_path_buf()])
        .expect("invariant: catalog chargeable");
    let mut config = test_config();
    config.hidden_size = 4;
    config.num_attention_heads = 1;
    config.num_key_value_heads = 1;
    config.head_dim = Some(4);
    config.partial_rotary_factor = Some(0.5);

    let model =
        load_qwen_causal_decoder_from_shards(&config, &[tmp.path().to_path_buf()], &catalog)
            .expect("invariant: RoPE partiel pair supporté");
    let logits = model
        .next_logits(&[0, 1])
        .expect("invariant: forward RoPE partiel valide");

    assert_eq!(logits.shape(), &[1, 3]);
    assert!(logits.data().iter().all(|value| value.is_finite()));
}

#[test]
fn loads_qk_norm_when_present() {
    let tmp = tempfile::NamedTempFile::new().expect("invariant: fichier temporaire");
    write_qk_norm_safetensors(tmp.path());
    let catalog = WeightCatalog::from_shards(&[tmp.path().to_path_buf()])
        .expect("invariant: catalog chargeable");

    let model =
        load_qwen_causal_decoder_from_shards(&test_config(), &[tmp.path().to_path_buf()], &catalog)
            .expect("invariant: q_norm/k_norm chargeables");
    let logits = model
        .next_logits(&[0, 1])
        .expect("invariant: forward q_norm/k_norm valide");

    assert_eq!(logits.shape(), &[1, 3]);
    assert!(logits.data().iter().all(|value| value.is_finite()));
}

#[test]
fn rejects_partial_qk_norm() {
    let tmp = tempfile::NamedTempFile::new().expect("invariant: fichier temporaire");
    let mut tensors = base_tensors("model.", "lm_head.");
    tensors.push((
        "model.layers.0.self_attn.q_norm.weight".to_string(),
        TensorFixture::ones(vec![2]),
    ));
    let buffer = serialize(tensors, None).expect("invariant: safetensors sérialisable");
    std::fs::write(tmp.path(), buffer).expect("invariant: écriture safetensors");
    let catalog = WeightCatalog::from_shards(&[tmp.path().to_path_buf()])
        .expect("invariant: catalog chargeable");

    let err =
        load_qwen_causal_decoder_from_shards(&test_config(), &[tmp.path().to_path_buf()], &catalog)
            .expect_err("invariant: q_norm/k_norm partiels rejetés");
    assert!(matches!(err, InferError::Config(_)));
}

#[test]
fn loads_dense_qwen_mlp_when_present() {
    let tmp = tempfile::NamedTempFile::new().expect("invariant: fichier temporaire");
    write_safetensors_with_mlp(tmp.path());
    let catalog = WeightCatalog::from_shards(&[tmp.path().to_path_buf()])
        .expect("invariant: catalog chargeable");

    let model =
        load_qwen_causal_decoder_from_shards(&test_config(), &[tmp.path().to_path_buf()], &catalog)
            .expect("invariant: MLP dense chargeable");
    let logits = model
        .next_logits(&[0])
        .expect("invariant: forward MLP valide");

    assert_eq!(logits.shape(), &[1, 3]);
    assert!(logits.data().iter().all(|value| value.is_finite()));
}

#[test]
fn loads_dense_qwen_layer_counts() {
    for layer_count in [1_usize, 2, 3, 4, 5, 6] {
        let tmp = tempfile::NamedTempFile::new().expect("invariant: fichier temporaire");
        write_layered_safetensors(tmp.path(), layer_count);
        let catalog = WeightCatalog::from_shards(&[tmp.path().to_path_buf()])
            .expect("invariant: catalog chargeable");
        let mut config = test_config();
        config.num_hidden_layers = layer_count;

        let model =
            load_qwen_causal_decoder_from_shards(&config, &[tmp.path().to_path_buf()], &catalog)
                .expect("invariant: modèle dense multi-couches chargeable");
        let logits = model
            .next_logits(&[0, 1])
            .expect("invariant: forward multi-couches valide");

        assert_eq!(logits.shape(), &[1, 3]);
        assert!(logits.data().iter().all(|value| value.is_finite()));
    }
}

#[test]
fn loader_cached_greedy_matches_full_sequence_for_layered_shards() {
    for layer_count in [1_usize, 2, 3, 4, 5, 6] {
        let tmp = tempfile::NamedTempFile::new().expect("invariant: fichier temporaire");
        write_layered_safetensors(tmp.path(), layer_count);
        let catalog = WeightCatalog::from_shards(&[tmp.path().to_path_buf()])
            .expect("invariant: catalog chargeable");
        let mut config = test_config();
        config.num_hidden_layers = layer_count;
        let model =
            load_qwen_causal_decoder_from_shards(&config, &[tmp.path().to_path_buf()], &catalog)
                .expect("invariant: modèle shardé multi-couches chargeable");

        let full = model
            .generate_greedy_full_with_options(&[0, 1], 4, &GenerationOptions::default())
            .expect("invariant: greedy complet shardé valide");
        let cached = model
            .generate_greedy_cached_with_options(&[0, 1], 4, &GenerationOptions::default())
            .expect("invariant: greedy cache shardé valide");

        assert_eq!(cached, full, "layer_count={layer_count}");
    }
}

#[test]
fn rejects_extra_layer_weights() {
    let tmp = tempfile::NamedTempFile::new().expect("invariant: fichier temporaire");
    write_safetensors(
        tmp.path(),
        "model.",
        "lm_head.",
        Some((
            "model.layers.1.input_layernorm.weight",
            TensorFixture::ones(vec![2]),
        )),
    );
    let catalog = WeightCatalog::from_shards(&[tmp.path().to_path_buf()])
        .expect("invariant: catalog chargeable");

    let err =
        load_qwen_causal_decoder_from_shards(&test_config(), &[tmp.path().to_path_buf()], &catalog)
            .expect_err("invariant: poids layer 1 rejeté");
    assert!(matches!(err, InferError::Config(_)));
}

#[test]
fn loads_dense_weights_even_when_quant_config_is_present() {
    let tmp = tempfile::NamedTempFile::new().expect("invariant: fichier temporaire");
    write_safetensors(tmp.path(), "model.", "lm_head.", None);
    let catalog = WeightCatalog::from_shards(&[tmp.path().to_path_buf()])
        .expect("invariant: catalog chargeable");
    let mut config = test_config();
    config.quantization = Some(QuantConfig {
        group_size: Some(64),
        bits: Some(4),
        quant_method: Some("mx".to_string()),
        fmt: None,
        extra: HashMap::new(),
    });

    let model =
        load_qwen_causal_decoder_from_shards(&config, &[tmp.path().to_path_buf()], &catalog)
            .expect("invariant: poids denses compatibles avec config quantifiée");
    let logits = model
        .next_logits(&[0])
        .expect("invariant: forward dense avec config quant valide");
    assert_eq!(logits.shape(), &[1, 3]);
}

#[test]
fn loads_affine_quantized_linear_weight_compactly() {
    let tmp = tempfile::NamedTempFile::new().expect("invariant: fichier temporaire");
    write_quantized_q_proj_safetensors(tmp.path());
    let catalog = WeightCatalog::from_shards(&[tmp.path().to_path_buf()])
        .expect("invariant: catalog chargeable");
    let mut config = test_config();
    config.hidden_size = 4;
    config.num_attention_heads = 1;
    config.num_key_value_heads = 1;
    config.head_dim = Some(4);
    config.quantization = Some(QuantConfig {
        group_size: Some(4),
        bits: Some(8),
        quant_method: Some("mx".to_string()),
        fmt: None,
        extra: HashMap::new(),
    });

    let prefixes = QwenPrefixes::detect(&catalog);
    let tensors = load_decoder_tensors(&config, &[tmp.path().to_path_buf()], &catalog, &prefixes)
        .expect("invariant: poids affine quantifié chargeable");
    assert!(matches!(
        tensors.get("layers.0.self_attn.q_proj.weight"),
        Some(DecoderTensor::LinearWeight(LinearWeight::AffineQuantized(
            _
        )))
    ));

    let model =
        load_qwen_causal_decoder_from_shards(&config, &[tmp.path().to_path_buf()], &catalog)
            .expect("invariant: poids affine quantifié compact");
    let logits = model
        .next_logits(&[0, 1])
        .expect("invariant: forward quant affine compact valide");

    assert_eq!(logits.shape(), &[1, 3]);
    assert!(logits.data().iter().all(|value| value.is_finite()));
}

#[test]
fn verifies_affine_quantized_contract_compactly() {
    let tmp = tempfile::NamedTempFile::new().expect("invariant: fichier temporaire");
    write_quantized_q_proj_safetensors(tmp.path());
    let catalog = WeightCatalog::from_shards(&[tmp.path().to_path_buf()])
        .expect("invariant: catalog chargeable");
    let mut config = test_config();
    config.hidden_size = 4;
    config.num_attention_heads = 1;
    config.num_key_value_heads = 1;
    config.head_dim = Some(4);
    config.quantization = Some(QuantConfig {
        group_size: Some(4),
        bits: Some(8),
        quant_method: Some("mx".to_string()),
        fmt: None,
        extra: HashMap::new(),
    });

    let contract =
        verify_qwen_decoder_contract_from_shards(&config, &[tmp.path().to_path_buf()], &catalog)
            .expect("invariant: contrat quantifié compact valide");

    assert!(contract.required_specs >= 8);
    assert!(contract.present_specs >= contract.required_specs);
}

#[test]
fn verifies_per_tensor_quantization_override() {
    let tmp = tempfile::NamedTempFile::new().expect("invariant: fichier temporaire");
    write_quantized_q_proj_safetensors(tmp.path());
    let catalog = WeightCatalog::from_shards(&[tmp.path().to_path_buf()])
        .expect("invariant: catalog chargeable");
    let mut config = test_config();
    config.hidden_size = 4;
    config.num_attention_heads = 1;
    config.num_key_value_heads = 1;
    config.head_dim = Some(4);
    config.quantization = Some(QuantConfig {
        group_size: Some(4),
        bits: Some(4),
        quant_method: Some("mx".to_string()),
        fmt: None,
        extra: HashMap::from([(
            "model.layers.0.self_attn.q_proj".to_string(),
            serde_json::json!({"bits": 8, "group_size": 4}),
        )]),
    });

    verify_qwen_decoder_contract_from_shards(&config, &[tmp.path().to_path_buf()], &catalog)
        .expect("invariant: override quantification par tenseur appliqué");
}

#[test]
fn loads_quantized_companions_from_other_shard() {
    let dir = tempfile::tempdir().expect("invariant: tempdir");
    let weights = dir.path().join("weights.safetensors");
    let companions = dir.path().join("companions.safetensors");
    write_quantized_q_proj_split_safetensors(&weights, &companions);

    let shards = vec![weights, companions];
    let catalog = WeightCatalog::from_shards(&shards).expect("invariant: catalog chargeable");
    let mut config = test_config();
    config.hidden_size = 4;
    config.num_attention_heads = 1;
    config.num_key_value_heads = 1;
    config.head_dim = Some(4);
    config.quantization = Some(QuantConfig {
        group_size: Some(4),
        bits: Some(8),
        quant_method: Some("mx".to_string()),
        fmt: None,
        extra: HashMap::new(),
    });

    let model = load_qwen_causal_decoder_from_shards(&config, &shards, &catalog)
        .expect("invariant: companions quantifiés cross-shard chargeables");
    let logits = model
        .next_logits(&[0, 1])
        .expect("invariant: forward cross-shard valide");
    assert_eq!(logits.shape(), &[1, 3]);
}

#[test]
fn loads_affine_quantized_embedding_compactly() {
    let tmp = tempfile::NamedTempFile::new().expect("invariant: fichier temporaire");
    write_quantized_embedding_safetensors(tmp.path());
    let catalog = WeightCatalog::from_shards(&[tmp.path().to_path_buf()])
        .expect("invariant: catalog chargeable");
    let mut config = test_config();
    config.hidden_size = 4;
    config.num_attention_heads = 1;
    config.num_key_value_heads = 1;
    config.head_dim = Some(4);
    config.quantization = Some(QuantConfig {
        group_size: Some(4),
        bits: Some(8),
        quant_method: Some("mx".to_string()),
        fmt: None,
        extra: HashMap::new(),
    });

    let prefixes = QwenPrefixes::detect(&catalog);
    let tensors = load_decoder_tensors(&config, &[tmp.path().to_path_buf()], &catalog, &prefixes)
        .expect("invariant: embedding quantifié chargeable");
    assert!(matches!(
        tensors.get("embed_tokens.weight"),
        Some(DecoderTensor::LinearWeight(LinearWeight::AffineQuantized(
            _
        )))
    ));

    let model =
        load_qwen_causal_decoder_from_shards(&config, &[tmp.path().to_path_buf()], &catalog)
            .expect("invariant: embedding quantifié compact");
    let logits = model
        .next_logits(&[0, 1])
        .expect("invariant: forward embedding compact valide");

    assert_eq!(logits.shape(), &[1, 3]);
    assert!(logits.data().iter().all(|value| value.is_finite()));
}

#[test]
fn accepts_fp8_quantized_config_for_dense_weights() {
    let tmp = tempfile::NamedTempFile::new().expect("invariant: fichier temporaire");
    write_safetensors(tmp.path(), "model.", "lm_head.", None);
    let catalog = WeightCatalog::from_shards(&[tmp.path().to_path_buf()])
        .expect("invariant: catalog chargeable");
    let mut config = test_config();
    config.quantization = Some(QuantConfig {
        group_size: None,
        bits: None,
        quant_method: Some("fp8".to_string()),
        fmt: Some("e4m3".to_string()),
        extra: HashMap::new(),
    });

    let model =
        load_qwen_causal_decoder_from_shards(&config, &[tmp.path().to_path_buf()], &catalog)
            .expect("invariant: config FP8 acceptée");
    let logits = model
        .next_logits(&[0])
        .expect("invariant: forward config FP8 valide");
    assert_eq!(logits.shape(), &[1, 3]);
}

#[test]
fn loads_fp8_weight_with_scalar_scale_inv() {
    let tmp = tempfile::NamedTempFile::new().expect("invariant: fichier temporaire");
    write_fp8_q_proj_safetensors(tmp.path());
    let catalog = WeightCatalog::from_shards(&[tmp.path().to_path_buf()])
        .expect("invariant: catalog chargeable");
    let mut config = test_config();
    config.quantization = Some(QuantConfig {
        group_size: None,
        bits: None,
        quant_method: Some("fp8".to_string()),
        fmt: Some("e4m3".to_string()),
        extra: HashMap::new(),
    });

    let model =
        load_qwen_causal_decoder_from_shards(&config, &[tmp.path().to_path_buf()], &catalog)
            .expect("invariant: poids FP8 chargeable");
    let logits = model
        .next_logits(&[0, 1])
        .expect("invariant: forward FP8 valide");

    assert_eq!(logits.shape(), &[1, 3]);
    assert!(logits.data().iter().all(|value| value.is_finite()));
}

#[test]
fn applies_fp8_block_scale_inv() {
    let tensor = Tensor::from_vec(
        vec![3, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    )
    .expect("invariant: tensor test valide");
    let scaled = apply_fp8_scales(
        tensor,
        &[10.0, 20.0, 30.0, 40.0],
        &[2, 2],
        "test.weight_scale_inv",
        2,
    )
    .expect("invariant: scales par blocs valides");

    assert_eq!(
        scaled.data(),
        &[10.0, 20.0, 60.0, 40.0, 50.0, 120.0, 210.0, 240.0, 360.0]
    );
}

#[test]
fn loads_dense_moe_qwen_layer() {
    let tmp = tempfile::NamedTempFile::new().expect("invariant: fichier temporaire");
    write_moe_safetensors(tmp.path());
    let catalog = WeightCatalog::from_shards(&[tmp.path().to_path_buf()])
        .expect("invariant: catalog chargeable");
    let mut config = test_config();
    config.model_type = "qwen3_5_moe".to_string();
    config.num_experts = Some(2);
    config.num_experts_per_tok = Some(1);
    config.moe_intermediate_size = Some(2);
    config.shared_expert_intermediate_size = Some(2);

    let model =
        load_qwen_causal_decoder_from_shards(&config, &[tmp.path().to_path_buf()], &catalog)
            .expect("invariant: MoE dense chargeable");
    let logits = model
        .next_logits(&[0, 1])
        .expect("invariant: forward MoE dense valide");

    assert_eq!(logits.shape(), &[1, 3]);
    assert!(logits.data().iter().all(|value| value.is_finite()));
}

#[test]
fn loads_moe_qwen_without_shared_expert() {
    let tmp = tempfile::NamedTempFile::new().expect("invariant: fichier temporaire");
    write_moe_safetensors_without_shared(tmp.path());
    let catalog = WeightCatalog::from_shards(&[tmp.path().to_path_buf()])
        .expect("invariant: catalog chargeable");
    let mut config = test_config();
    config.model_type = "qwen3_moe".to_string();
    config.num_experts = Some(2);
    config.num_experts_per_tok = Some(1);
    config.moe_intermediate_size = Some(2);
    config.shared_expert_intermediate_size = None;

    let model =
        load_qwen_causal_decoder_from_shards(&config, &[tmp.path().to_path_buf()], &catalog)
            .expect("invariant: MoE sans shared expert chargeable");
    let logits = model
        .next_logits(&[0, 1])
        .expect("invariant: forward MoE sans shared expert valide");

    assert_eq!(logits.shape(), &[1, 3]);
    verify_qwen_decoder_contract_from_shards(&config, &[tmp.path().to_path_buf()], &catalog)
        .expect("invariant: contrat MoE sans shared expert valide");
}

#[test]
fn loads_affine_quantized_moe_expert_weights_compactly() {
    let tmp = tempfile::NamedTempFile::new().expect("invariant: fichier temporaire");
    write_quantized_moe_safetensors(tmp.path());
    let catalog = WeightCatalog::from_shards(&[tmp.path().to_path_buf()])
        .expect("invariant: catalog chargeable");
    let mut config = test_config();
    config.model_type = "qwen3_5_moe".to_string();
    config.hidden_size = 4;
    config.num_attention_heads = 1;
    config.num_key_value_heads = 1;
    config.head_dim = Some(4);
    config.intermediate_size = 0;
    config.num_experts = Some(2);
    config.num_experts_per_tok = Some(1);
    config.moe_intermediate_size = Some(4);
    config.shared_expert_intermediate_size = Some(4);
    config.quantization = Some(QuantConfig {
        group_size: Some(4),
        bits: Some(8),
        quant_method: Some("mx".to_string()),
        fmt: None,
        extra: HashMap::new(),
    });

    let prefixes = QwenPrefixes::detect(&catalog);
    let tensors = load_decoder_tensors(&config, &[tmp.path().to_path_buf()], &catalog, &prefixes)
        .expect("invariant: poids MoE quantifiés chargeables");
    assert!(matches!(
        tensors.get("layers.0.mlp.switch_mlp.gate_proj.weight"),
        Some(DecoderTensor::ExpertLinearWeights { weights, .. })
            if matches!(weights.first(), Some(LinearWeight::AffineQuantized(_)))
    ));

    let model =
        load_qwen_causal_decoder_from_shards(&config, &[tmp.path().to_path_buf()], &catalog)
            .expect("invariant: MoE quantifié compact chargeable");
    let logits = model
        .next_logits(&[0, 1])
        .expect("invariant: forward MoE quantifié valide");

    assert_eq!(logits.shape(), &[1, 3]);
    assert!(logits.data().iter().all(|value| value.is_finite()));
}

#[test]
fn loads_hybrid_linear_attention_qwen_layers() {
    let tmp = tempfile::NamedTempFile::new().expect("invariant: fichier temporaire");
    write_hybrid_safetensors(tmp.path());
    let catalog = WeightCatalog::from_shards(&[tmp.path().to_path_buf()])
        .expect("invariant: catalog chargeable");

    let mut hybrid = test_config();
    hybrid.num_hidden_layers = 2;
    hybrid.full_attention_interval = Some(2);
    hybrid.attn_output_gate = Some(true);
    hybrid.linear_num_key_heads = Some(1);
    hybrid.linear_num_value_heads = Some(1);
    hybrid.linear_key_head_dim = Some(2);
    hybrid.linear_value_head_dim = Some(2);
    hybrid.linear_conv_kernel_dim = Some(2);
    let model =
        load_qwen_causal_decoder_from_shards(&hybrid, &[tmp.path().to_path_buf()], &catalog)
            .expect("invariant: hybride linear-attn chargeable");
    let logits = model
        .next_logits(&[0, 1])
        .expect("invariant: forward hybride valide");

    assert_eq!(logits.shape(), &[1, 3]);
    assert!(logits.data().iter().all(|value| value.is_finite()));
}

#[test]
fn loads_f16_required_weight_by_dense_dequantization() {
    let tmp = tempfile::NamedTempFile::new().expect("invariant: fichier temporaire");
    write_safetensors(
        tmp.path(),
        "model.",
        "lm_head.",
        Some((
            "model.layers.0.self_attn.q_proj.weight",
            TensorFixture::f16_zeros(vec![2, 2]),
        )),
    );
    let catalog = WeightCatalog::from_shards(&[tmp.path().to_path_buf()])
        .expect("invariant: catalog chargeable");

    let model =
        load_qwen_causal_decoder_from_shards(&test_config(), &[tmp.path().to_path_buf()], &catalog)
            .expect("invariant: F16 dense déquantifié en F32 chargeable");
    let logits = model
        .next_logits(&[0])
        .expect("invariant: forward F16 dense valide");
    assert_eq!(logits.shape(), &[1, 3]);
}

#[test]
fn rejects_non_float_required_weight() {
    let tmp = tempfile::NamedTempFile::new().expect("invariant: fichier temporaire");
    write_safetensors(
        tmp.path(),
        "model.",
        "lm_head.",
        Some((
            "model.layers.0.self_attn.q_proj.weight",
            TensorFixture::i32_zeros(vec![2, 2]),
        )),
    );
    let catalog = WeightCatalog::from_shards(&[tmp.path().to_path_buf()])
        .expect("invariant: catalog chargeable");

    let err =
        load_qwen_causal_decoder_from_shards(&test_config(), &[tmp.path().to_path_buf()], &catalog)
            .expect_err("invariant: dtype entier rejeté");
    assert!(matches!(err, InferError::UnsupportedDtype { .. }));
}
