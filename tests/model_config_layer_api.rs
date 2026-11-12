use saragossa::{ModelConfig, RawModelConfig};

#[test]
fn public_layer_accessors_keep_loader_semantics() {
    // Ce test externe au crate protège aussi la visibilité et les signatures.
    let full: fn(&ModelConfig, usize) -> bool = ModelConfig::is_gemma4_full_layer;
    let sliding: fn(&ModelConfig, usize) -> bool = ModelConfig::is_gemma4_sliding_layer;
    let head_dim: fn(&ModelConfig, usize) -> usize = ModelConfig::layer_head_dim;
    let kv_heads: fn(&ModelConfig, usize) -> usize = ModelConfig::layer_num_key_value_heads;
    let rope_dims: fn(&ModelConfig, usize) -> usize = ModelConfig::layer_rope_dims;
    let raw: RawModelConfig = serde_json::from_str(
        r#"{
            "model_type": "qwen3",
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 32,
            "rms_norm_eps": 0.000001,
            "rope_theta": 10000.0,
            "vocab_size": 16
        }"#,
    )
    .expect("invariant: configuration synthétique valide");
    let mut model = raw.resolve().expect("invariant: configuration résolue");
    model.layer_types = vec!["sliding_attention".into(), "full_attention".into()];
    model.global_head_dim = Some(8);
    model.num_global_key_value_heads = Some(3);
    model.partial_rotary_factor = Some(0.5);
    model.rope_full_partial_rotary_factor = Some(0.25);
    model.rope_sliding_partial_rotary_factor = Some(0.75);
    model.full_attention_interval = Some(2);
    for (index, expected) in [
        (0, (false, true, 4, 2, 3)),
        (1, (true, false, 8, 3, 2)),
        (2, (false, false, 4, 2, 2)),
        (usize::MAX, (false, false, 4, 2, 2)),
    ] {
        assert_eq!(
            (
                full(&model, index),
                sliding(&model, index),
                head_dim(&model, index),
                kv_heads(&model, index),
                rope_dims(&model, index),
            ),
            expected,
        );
    }
    // La division de repli reste différée : classifier une couche ne réclame
    // aucune dimension, et une dimension globale suffit pour la couche pleine.
    model.head_dim = None;
    model.num_attention_heads = 0;
    assert!(full(&model, 1));
    assert!(sliding(&model, 0));
    assert_eq!(kv_heads(&model, 0), 2);
    assert_eq!(head_dim(&model, 1), 8);
    assert_eq!(rope_dims(&model, 1), 2);
}
