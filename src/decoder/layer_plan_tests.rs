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

#[test]
fn historical_layer_decisions() {
    let mut snapshot = String::new();
    for (family, model) in fixtures() {
        snapshot.push_str(&format!("{family}\n"));
        let config = CausalDecoderConfig::from(&model);
        // Inclut la première couche hors de la liste déclarée.
        for index in 0..=config.num_hidden_layers {
            snapshot.push_str(&format!("{:?}\n", planned(&config, index)));
            let loader = layer_plan::LoaderLayerPlan::build(&model, index);
            snapshot.push_str(&format!(
                "loader {:?}\n",
                (
                    loader.gemma_full,
                    loader.gemma_sliding,
                    loader.head_dim(),
                    loader.kv_heads,
                    loader.rope_dims(),
                    loader.full,
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
            snapshot.push_str(&format!("{:?}\n", planned(&config, index)));
        }
    }
    assert_eq!(snapshot, include_str!("layer_plan_snapshots/decisions.txt"));
}

#[test]
fn historical_attention_errors_and_inferred_dimension() {
    use super::attention_ops::full_attention_from_tensors;
    let config = CausalDecoderConfig::default();
    let attention_tensors = || {
        let mut tensors = HashMap::new();
        for name in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            tensors.insert(
                format!("layers.0.self_attn.{name}.weight"),
                DecoderTensor::Dense(
                    Tensor::from_vec(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]).unwrap(),
                ),
            );
        }
        tensors
    };
    for norm in ["q_norm", "k_norm"] {
        let mut partial = attention_tensors();
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
    // Même avec une dimension configurée, l'inférence depuis Q reste évaluée
    // avant le repli : déplacer sa validation changerait l'erreur observable.
    let invalid = CausalDecoderConfig {
        num_attention_heads: 0,
        head_dim: Some(2),
        ..config.clone()
    };
    let error = full_attention_from_tensors(
        &invalid,
        &mut attention_tensors(),
        "layers.0",
        &LayerPlan::build(&invalid, 0).resident,
    )
    .unwrap_err();
    assert_eq!(
        error.to_string(),
        "dimension incompatible: num_attention_heads nul pour inférer head_dim"
    );
    let AttentionBlock::Full(attention) = full_attention_from_tensors(
        &config,
        &mut attention_tensors(),
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
                LayerPlan::build(&config, index)
                    .resident
                    .kv_dim()
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
fn plan_keeps_distinct_assembly_and_resident_classification() {
    let model = crate::qwen_loader::gemma4_test_fixtures::gemma4_unified_dense_config();
    let mut config = CausalDecoderConfig::from(&model);
    config.full_attention_interval = Some(2);
    let plan = ModelPlan::build(&config);
    assert!(!plan.layers[0].is_full());
    assert!(plan.layers[0].resident_full);
    assert_eq!(
        plan.layers[0].resident.head_dim().unwrap(),
        model.head_dim()
    );
    assert!(plan.layers[1].is_full());
    assert!(plan.layers[1].resident_full);
}

#[test]
fn model_plan_debug_snapshots() {
    for (family, model) in fixtures() {
        let config = CausalDecoderConfig::from(&model);
        let plan = ModelPlan::build(&config);
        assert_eq!(plan.layers.len(), config.num_hidden_layers);
        assert_eq!(
            plan.fingerprint,
            format!("{:x}", md5::compute(format!("{plan:?}")))
        );
        let actual = format!("{plan:#?}\n");
        let expected = match family {
            "qwen_dense" => include_str!("layer_plan_snapshots/qwen_dense.txt"),
            "qwen_moe" => include_str!("layer_plan_snapshots/qwen_moe.txt"),
            "qwen_hybrid" => include_str!("layer_plan_snapshots/qwen_hybrid.txt"),
            "qwen_hybrid_moe" => include_str!("layer_plan_snapshots/qwen_hybrid_moe.txt"),
            "gemma3" => include_str!("layer_plan_snapshots/gemma3.txt"),
            "gemma4_dense" => include_str!("layer_plan_snapshots/gemma4_dense.txt"),
            "gemma4_moe" => include_str!("layer_plan_snapshots/gemma4_moe.txt"),
            "flint" => include_str!("layer_plan_snapshots/flint.txt"),
            "qwen_labels_divergent" => {
                include_str!("layer_plan_snapshots/qwen_labels_divergent.txt")
            }
            "gemma4_partial_options" => {
                include_str!("layer_plan_snapshots/gemma4_partial_options.txt")
            }
            _ => panic!("fixture sans snapshot"),
        };
        assert_eq!(actual, expected, "{family}");
    }
}
