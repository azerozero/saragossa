//! Contrat historique : aucune fixture ne charge de poids externes.
use super::*;

fn fixtures() -> Vec<(&'static str, ModelConfig)> {
    use crate::qwen_loader::{flint_tests, gemma4_test_fixtures, test_fixtures};
    let dense = test_fixtures::test_config();
    let mut moe = dense.clone();
    moe.model_type = "qwen3_5_moe".into();
    moe.num_experts = Some(2);
    moe.num_experts_per_tok = Some(1);
    moe.moe_intermediate_size = Some(2);
    moe.shared_expert_intermediate_size = Some(2);
    let mut hybrid = dense.clone();
    hybrid.model_type = "qwen3_5".into();
    hybrid.num_hidden_layers = 2;
    hybrid.full_attention_interval = Some(2);
    hybrid.attn_output_gate = Some(true);
    hybrid.linear_num_key_heads = Some(1);
    hybrid.linear_num_value_heads = Some(1);
    hybrid.linear_key_head_dim = Some(2);
    hybrid.linear_value_head_dim = Some(2);
    hybrid.linear_conv_kernel_dim = Some(2);
    hybrid.partial_rotary_factor = Some(0.5);
    hybrid.layer_types = vec!["linear_attention".into(), "full_attention".into()];
    let mut hybrid_moe = hybrid.clone();
    hybrid_moe.model_type = moe.model_type.clone();
    hybrid_moe.num_experts = moe.num_experts;
    hybrid_moe.moe_intermediate_size = moe.moe_intermediate_size;
    let mut gemma3 = test_fixtures::gemma_test_config();
    gemma3.model_type = "gemma3_text".into();
    gemma3.num_hidden_layers = 7;
    gemma3.sliding_window = Some(512);
    gemma3.sliding_window_pattern = Some(6);
    gemma3.rope_local_base_freq = Some(10_000.0);
    gemma3.rope_scaling = Some(
        serde_json::from_value(serde_json::json!({"rope_type":"linear", "factor":8.0})).unwrap(),
    );
    let mut divergent = hybrid_moe.clone();
    divergent.global_head_dim = Some(8);
    divergent.num_global_key_value_heads = Some(3);
    divergent.rope_full_partial_rotary_factor = Some(0.25);
    divergent.attention_k_eq_v = true;
    let mut partial = gemma4_test_fixtures::gemma4_unified_dense_config();
    partial.rope_full_partial_rotary_factor = None;
    partial.rope_sliding_partial_rotary_factor = None;
    partial.partial_rotary_factor = Some(0.25);
    vec![
        ("qwen_dense", dense),
        ("qwen_moe", moe),
        ("qwen_hybrid", hybrid),
        ("qwen_hybrid_moe", hybrid_moe),
        ("gemma3", gemma3),
        (
            "gemma4_dense",
            gemma4_test_fixtures::gemma4_unified_dense_config(),
        ),
        ("gemma4_moe", gemma4_test_fixtures::gemma4_moe_config()),
        ("flint", flint_tests::flint_config()),
        ("qwen_labels_divergent", divergent),
        ("gemma4_partial_options", partial),
    ]
}

#[derive(Debug, PartialEq)]
struct Decisions {
    index: usize,
    kind: Option<String>,
    gemma_full: bool,
    gemma_sliding: bool,
    full: bool,
    resident_full: bool,
    local: bool,
    head_dim: std::result::Result<usize, String>,
    kv_heads: usize,
    // Plusieurs replis figent aussi l'appel avec une dimension inférée depuis Q.
    rope_dims: [usize; 3],
    rope_frequency_dim: [usize; 3],
    theta_bits: Option<u32>,
    scale_bits: Option<u32>,
    window: Option<usize>,
    no_rope: bool,
    v_shared: bool,
    dense_mlp: Option<bool>,
    sinks: bool,
}

fn legacy(config: &CausalDecoderConfig, index: usize) -> Decisions {
    Decisions {
        index,
        kind: config.layer_kind(index).map(str::to_owned),
        gemma_full: config.is_gemma4_full_layer(index),
        gemma_sliding: config.is_gemma4_sliding_layer(index),
        full: config.is_full_attention_layer(index),
        resident_full: config.is_resident_full_attention_layer(index),
        local: config.is_local_sliding_layer(index),
        head_dim: config.layer_head_dim(index).map_err(|e| e.to_string()),
        kv_heads: config.layer_num_key_value_heads(index),
        rope_dims: [0, 2, 128].map(|h| config.layer_rope_dims(index, h)),
        rope_frequency_dim: [0, 2, 128].map(|h| config.layer_rope_frequency_dim(index, h)),
        theta_bits: config.layer_rope_theta_override(index).map(f32::to_bits),
        scale_bits: config.layer_rope_position_scale(index).map(f32::to_bits),
        window: config.layer_sliding_window(index),
        no_rope: config.is_no_rope_layer(index),
        v_shared: config.attention_k_eq_v && config.is_gemma4_full_layer(index),
        dense_mlp: (!config.mlp_only_layers.is_empty())
            .then(|| config.mlp_only_layers.contains(&index)),
        sinks: config.attention_sinks,
    }
}

#[test]
fn historical_layer_decisions() {
    let mut snapshot = String::new();
    for (family, model) in fixtures() {
        snapshot.push_str(&format!("{family}\n"));
        let config = CausalDecoderConfig::from(&model);
        // Inclut la première couche hors de la liste déclarée.
        for index in 0..=config.num_hidden_layers {
            snapshot.push_str(&format!("{:?}\n", legacy(&config, index)));
            snapshot.push_str(&format!(
                "loader {:?}\n",
                (
                    model.is_gemma4_full_layer(index),
                    model.is_gemma4_sliding_layer(index),
                    model.layer_head_dim(index),
                    model.layer_num_key_value_heads(index),
                    model.layer_rope_dims(index),
                    model.is_full_attention_layer(index),
                )
            ));
        }
    }
    for interval in [None, Some(0), Some(1), Some(3)] {
        let config = CausalDecoderConfig {
            full_attention_interval: interval,
            sliding_window_pattern: interval,
            rope_position_scale: Some(-0.0),
            no_rope_layers: vec![0, 3],
            ..CausalDecoderConfig::default()
        };
        snapshot.push_str(&format!("absent_options {interval:?}\n"));
        for index in [0, 1, 2, 3] {
            snapshot.push_str(&format!("{:?}\n", legacy(&config, index)));
        }
    }
    assert_eq!(snapshot, include_str!("layer_plan_snapshots/decisions.txt"));
}

#[test]
fn historical_attention_errors_and_inferred_dimension() {
    use super::attention_ops::full_attention_from_tensors;
    let config = CausalDecoderConfig::default();
    let mut tensors = HashMap::new();
    for name in ["q_proj", "k_proj", "v_proj", "o_proj"] {
        tensors.insert(
            format!("layers.0.self_attn.{name}.weight"),
            DecoderTensor::Dense(Tensor::from_vec(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]).unwrap()),
        );
    }
    for norm in ["q_norm", "k_norm"] {
        let mut partial = tensors.clone();
        partial.insert(
            format!("layers.0.self_attn.{norm}.weight"),
            DecoderTensor::Dense(Tensor::from_vec(vec![2], vec![1.0; 2]).unwrap()),
        );
        let error = full_attention_from_tensors(
            &config,
            &mut partial,
            "layers.0",
            &LayerPlan::build(&config, 0).resident,
        )
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            "config modèle invalide: q_norm/k_norm partiels pour layers.0.self_attn"
        );
    }
    let AttentionBlock::Full(attention) = full_attention_from_tensors(
        &config,
        &mut tensors,
        "layers.0",
        &LayerPlan::build(&config, 0).resident,
    )
    .unwrap() else {
        panic!("full attendu")
    };
    assert_eq!(attention.head_dim, Some(2));
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn resident_dims_debug(d: FullAttnLayerDims) -> String {
    format!(
        "{:?}",
        (
            d.hidden,
            d.q_heads,
            d.kv_heads,
            d.head_dim,
            d.attn_scalar.to_bits(),
            d.rope_dims,
            d.rope_frequency_dim,
            d.position,
            d.window_start,
            d.eps.to_bits(),
            d.theta.to_bits(),
            d.attn_output_gate
        )
    )
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn historical_resident_dimensions() {
    let mut configs: Vec<_> = fixtures()
        .into_iter()
        .map(|(name, c)| (name, CausalDecoderConfig::from(&c)))
        .collect();
    configs.push(("absent", CausalDecoderConfig::default()));
    configs.push((
        "overflow",
        CausalDecoderConfig {
            head_dim: Some(usize::MAX),
            num_key_value_heads: 2,
            ..CausalDecoderConfig::default()
        },
    ));
    let mut snapshot = String::new();
    for (family, config) in configs {
        snapshot.push_str(&format!("{family}\n"));
        for index in 0..=config.num_hidden_layers {
            snapshot.push_str(&format!(
                "kv {index}: {:?}\n",
                config
                    .resident_full_attention_kv_dim(index)
                    .map_err(|e| e.to_string())
            ));
            for position in [0, 1, 511, 512, 513] {
                snapshot.push_str(&format!(
                    "dims {index}/{position}: {:?}\n",
                    LayerPlan::build(&config, index)
                        .resident
                        .resident_windowed_full_attn_layer_dims(64, position, 1e-6, 123.0)
                        .map(resident_dims_debug)
                        .map_err(|e| e.to_string())
                ));
            }
        }
    }
    assert_eq!(snapshot, include_str!("layer_plan_snapshots/resident.txt"));
}

fn planned(config: &CausalDecoderConfig, index: usize) -> Decisions {
    let layer = layer_plan::LayerPlan::build(config, index);
    let full = &layer.resident;
    Decisions {
        index: layer.index,
        kind: config.layer_types.get(index).cloned(),
        gemma_full: full.gemma_full,
        gemma_sliding: full.gemma_sliding,
        full: layer.is_full(),
        resident_full: layer.resident_full,
        local: full.local_sliding,
        head_dim: full.head_dim().map_err(|e| e.to_string()),
        kv_heads: full.kv_heads,
        rope_dims: [0, 2, 128].map(|h| full.rope_dims(h)),
        rope_frequency_dim: [0, 2, 128].map(|h| full.rope_frequency_dim(h)),
        theta_bits: full.rope_theta_override.map(f32::to_bits),
        scale_bits: full.rope_position_scale.map(f32::to_bits),
        window: full.sliding_window,
        no_rope: full.no_rope,
        v_shared: full.v_shared_with_k,
        dense_mlp: layer.dense_mlp_expected,
        sinks: config.attention_sinks,
    }
}

#[test]
fn plan_matches_every_legacy_decision() {
    let mut configs: Vec<_> = fixtures()
        .into_iter()
        .map(|(_, c)| CausalDecoderConfig::from(&c))
        .collect();
    for interval in [None, Some(0), Some(1), Some(3)] {
        configs.push(CausalDecoderConfig {
            num_hidden_layers: 4,
            full_attention_interval: interval,
            sliding_window_pattern: interval,
            rope_position_scale: Some(-0.0),
            no_rope_layers: vec![0, 3],
            ..CausalDecoderConfig::default()
        });
    }
    let mut contradictory = configs[5].clone();
    contradictory.full_attention_interval = Some(2);
    configs.push(contradictory);
    configs.push(CausalDecoderConfig {
        head_dim: Some(usize::MAX),
        num_key_value_heads: 2,
        ..CausalDecoderConfig::default()
    });
    for config in configs {
        let plan = ModelPlan::build(&config);
        assert_eq!(plan.layers.len(), config.num_hidden_layers);
        assert_eq!(
            plan.fingerprint,
            format!("{:x}", md5::compute(format!("{plan:?}")))
        );
        for index in 0..=config.num_hidden_layers {
            assert_eq!(legacy(&config, index), planned(&config, index));
            #[cfg(all(target_os = "macos", feature = "metal"))]
            {
                let layer = layer_plan::LayerPlan::build(&config, index);
                let full = &layer.resident;
                assert_eq!(
                    config
                        .resident_full_attention_kv_dim(index)
                        .map_err(|e| e.to_string()),
                    full.kv_dim().map_err(|e| e.to_string())
                );
                fn dims(d: FullAttnLayerDims) -> String {
                    format!(
                        "{:?}",
                        (
                            d.hidden,
                            d.q_heads,
                            d.kv_heads,
                            d.head_dim,
                            d.attn_scalar.to_bits(),
                            d.rope_dims,
                            d.rope_frequency_dim,
                            d.position,
                            d.window_start,
                            d.eps.to_bits(),
                            d.theta.to_bits(),
                            d.attn_output_gate
                        )
                    )
                }
                assert_eq!(
                    config
                        .resident_full_attn_layer_dims(index, 64, 7, 1e-6, 123.0)
                        .map(dims)
                        .map_err(|e| e.to_string()),
                    full.resident_full_attn_layer_dims(64, 7, 1e-6, 123.0)
                        .map(dims)
                        .map_err(|e| e.to_string())
                );
            }
        }
    }
}
