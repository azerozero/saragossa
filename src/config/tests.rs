use super::*;
use proptest::prelude::*;

#[test]
fn resolves_flat_qwen_config() {
    let raw = r#"{
            "model_type":"qwen3",
            "hidden_size":1024,
            "num_hidden_layers":28,
            "num_attention_heads":16,
            "num_key_value_heads":8,
            "head_dim":128,
            "intermediate_size":3072,
            "rms_norm_eps":0.000001,
            "rope_theta":1000000.0,
            "vocab_size":151936
        }"#;
    let cfg: RawModelConfig = serde_json::from_str(raw).expect("invariant: JSON valide");
    let cfg = cfg.resolve().expect("invariant: config valide");
    assert_eq!(cfg.model_type, "qwen3");
    assert_eq!(cfg.head_dim(), 128);
    assert_eq!(cfg.rope_dims(), 128);
    assert!(cfg.eos_token_ids.is_empty());
}

#[test]
fn resolves_zero_shared_expert_as_absent() {
    let raw = r#"{
            "model_type":"qwen3_moe",
            "hidden_size":2048,
            "num_hidden_layers":48,
            "num_attention_heads":32,
            "num_key_value_heads":4,
            "head_dim":128,
            "intermediate_size":5472,
            "moe_intermediate_size":768,
            "num_experts":128,
            "num_experts_per_tok":8,
            "shared_expert_intermediate_size":0,
            "rms_norm_eps":0.000001,
            "rope_theta":10000000.0,
            "vocab_size":151936
        }"#;
    let cfg: RawModelConfig = serde_json::from_str(raw).expect("invariant: JSON valide");
    let cfg = cfg.resolve().expect("invariant: config MoE valide");
    assert_eq!(cfg.shared_expert_intermediate_size, None);
}

proptest! {
    #[test]
    fn parses_single_eos_token_id(id in 0_usize..1_000_000) {
        let raw = format!(
            r#"{{
                    "model_type":"qwen3",
                    "hidden_size":1024,
                    "num_hidden_layers":2,
                    "num_attention_heads":16,
                    "num_key_value_heads":8,
                    "head_dim":64,
                    "intermediate_size":3072,
                    "rms_norm_eps":0.000001,
                    "rope_theta":1000000.0,
                    "vocab_size":151936,
                    "eos_token_id":{id}
                }}"#
        );
        let cfg: RawModelConfig = serde_json::from_str(&raw)
            .expect("invariant: JSON généré valide");
        let cfg = cfg.resolve().expect("invariant: config générée valide");
        prop_assert_eq!(cfg.eos_token_ids, vec![id]);
    }

    #[test]
    fn parses_eos_token_id_arrays(ids in proptest::collection::vec(0_usize..1_000_000, 0..8)) {
        let ids_json = serde_json::to_string(&ids).expect("invariant: ids sérialisables");
        let raw = format!(
            r#"{{
                    "model_type":"qwen3",
                    "hidden_size":1024,
                    "num_hidden_layers":2,
                    "num_attention_heads":16,
                    "num_key_value_heads":8,
                    "head_dim":64,
                    "intermediate_size":3072,
                    "rms_norm_eps":0.000001,
                    "rope_theta":1000000.0,
                    "vocab_size":151936,
                    "eos_token_id":{ids_json}
                }}"#
        );
        let cfg: RawModelConfig = serde_json::from_str(&raw)
            .expect("invariant: JSON généré valide");
        let cfg = cfg.resolve().expect("invariant: config générée valide");
        prop_assert_eq!(cfg.eos_token_ids, ids);
    }
}

#[test]
fn resolves_nested_text_config_and_rope_parameters() {
    let raw = r#"{
            "model_type":"qwen3_5_moe",
            "quantization_config":{
                "group_size":64,
                "bits":4,
                "quant_method":"mx",
                "model.layers.0.self_attn.q_proj":{"group_size":64,"bits":8}
            },
            "text_config":{
                "model_type":"qwen3_5_moe_text",
                "hidden_size":2048,
                "num_hidden_layers":40,
                "num_attention_heads":16,
                "num_key_value_heads":4,
                "attn_output_gate":true,
                "head_dim":128,
                "intermediate_size":6144,
                "rms_norm_eps":0.000001,
                "vocab_size":152064,
                "eos_token_id":248044,
                "rope_parameters":{
                    "rope_theta":1000000.0,
                    "partial_rotary_factor":0.25
                },
                "full_attention_interval":4,
                "linear_conv_kernel_dim":4,
                "linear_key_head_dim":128,
                "linear_num_key_heads":16,
                "linear_num_value_heads":32,
                "linear_value_head_dim":128
            }
        }"#;
    let cfg: RawModelConfig = serde_json::from_str(raw).expect("invariant: JSON valide");
    let cfg = cfg.resolve().expect("invariant: config valide");
    assert_eq!(cfg.model_type, "qwen3_5_moe");
    assert_eq!(cfg.eos_token_ids, vec![248044]);
    assert_eq!(cfg.rope_theta, 1_000_000.0);
    assert_eq!(cfg.attn_output_gate, Some(true));
    assert_eq!(cfg.rope_dims(), 32);
    assert!(cfg.is_hybrid());
    assert!(!cfg.is_full_attention_layer(0));
    assert!(cfg.is_full_attention_layer(3));
    assert_eq!(cfg.linear_num_key_heads, Some(16));
    assert_eq!(cfg.linear_num_value_heads, Some(32));
    assert_eq!(
        cfg.quantization
            .as_ref()
            .and_then(|q| q.group_size)
            .expect("invariant: quantization présente"),
        64
    );
    assert_eq!(
        cfg.quantization
            .as_ref()
            .and_then(|q| q.extra.get("model.layers.0.self_attn.q_proj"))
            .and_then(|v| v.get("bits"))
            .and_then(serde_json::Value::as_u64),
        Some(8)
    );
}

#[test]
fn resolves_moe_config_without_dense_intermediate_size() {
    let raw = r#"{
            "model_type":"qwen3_5_moe",
            "text_config":{
                "hidden_size":2048,
                "num_hidden_layers":40,
                "num_attention_heads":16,
                "num_key_value_heads":4,
                "head_dim":128,
                "moe_intermediate_size":512,
                "rms_norm_eps":0.000001,
                "vocab_size":248320,
                "rope_parameters":{"rope_theta":10000000.0}
            }
        }"#;
    let cfg: RawModelConfig = serde_json::from_str(raw).expect("invariant: JSON valide");
    let cfg = cfg.resolve().expect("invariant: config MoE valide");
    assert_eq!(cfg.intermediate_size, 0);
    assert_eq!(cfg.mlp_intermediate_size(), 512);
}

#[test]
fn resolves_gemma3_text_config_with_local_layers() {
    // Reflet du config.json mlx-community/gemma-3-1b-it-4bit.
    let raw = r#"{
            "model_type":"gemma3_text",
            "attn_logit_softcapping":null,
            "final_logit_softcapping":null,
            "head_dim":256,
            "hidden_activation":"gelu_pytorch_tanh",
            "hidden_size":1152,
            "intermediate_size":6912,
            "num_attention_heads":4,
            "num_hidden_layers":26,
            "num_key_value_heads":1,
            "query_pre_attn_scalar":256,
            "rms_norm_eps":1e-06,
            "rope_local_base_freq":10000,
            "rope_scaling":null,
            "rope_theta":1000000,
            "sliding_window":512,
            "sliding_window_pattern":6,
            "eos_token_id":1,
            "vocab_size":262144
        }"#;
    let cfg: RawModelConfig = serde_json::from_str(raw).expect("invariant: JSON valide");
    let cfg = cfg.resolve().expect("invariant: config Gemma 3 valide");
    assert!(cfg.is_gemma());
    assert!(cfg.uses_gelu_tanh());
    // √1152 = 33.941… arrondi bf16 → 34.0 (constante vue à l'entraînement).
    assert_eq!(cfg.embed_scale(), Some(34.0));
    assert_eq!(cfg.rope_local_base_freq, Some(10_000.0));
    assert_eq!(cfg.sliding_window, Some(512));
    assert_eq!(cfg.sliding_window_pattern, Some(6));
    assert_eq!(cfg.query_pre_attn_scalar, Some(256.0));
    assert_eq!(cfg.attn_logit_softcapping, None);
    assert_eq!(cfg.final_logit_softcapping, None);
    // `rope_scaling: null` (1B) → aucune échelle de positions.
    assert_eq!(cfg.rope_scaling, None);
    assert_eq!(cfg.rope_position_scale(), None);
    assert_eq!(cfg.eos_token_ids, vec![1]);
}

#[test]
fn resolves_minimal_gemma3_multimodal_config_with_mlx_defaults() {
    // Reflet exact du config.json mlx-community/gemma-3-4b-it-4bit : le
    // text_config n'énumère que les clés hors-défaut, le reste vient des
    // défauts mlx_lm (wrapper gemma3.py + gemma3_text.ModelArgs).
    let raw = r#"{
            "architectures":["Gemma3ForConditionalGeneration"],
            "boi_token_index":255999,
            "eoi_token_index":256000,
            "eos_token_id":[1,106],
            "image_token_index":262144,
            "model_type":"gemma3",
            "quantization":{"group_size":64,"bits":4},
            "text_config":{
                "hidden_size":2560,
                "intermediate_size":10240,
                "model_type":"gemma3_text",
                "num_hidden_layers":34,
                "rope_scaling":{"factor":8.0,"rope_type":"linear"},
                "sliding_window":1024
            },
            "vision_config":{"model_type":"siglip_vision_model","skip_vision":true}
        }"#;
    let cfg: RawModelConfig = serde_json::from_str(raw).expect("invariant: JSON valide");
    let cfg = cfg.resolve().expect("invariant: config 4B valide");
    assert_eq!(cfg.model_type, "gemma3");
    assert!(cfg.is_gemma());
    assert_eq!(cfg.hidden_size, 2560);
    assert_eq!(cfg.num_hidden_layers, 34);
    // Défauts du wrapper gemma3.py : vocab 262208, 8 têtes Q, 4 KV.
    assert_eq!(cfg.vocab_size, 262_208);
    assert_eq!(cfg.num_attention_heads, 8);
    assert_eq!(cfg.num_key_value_heads, 4);
    // Défauts gemma3_text.ModelArgs : head_dim, bases RoPE, motif sliding.
    assert_eq!(cfg.head_dim(), 256);
    assert_eq!(cfg.rms_norm_eps, 1.0e-6);
    assert_eq!(cfg.rope_theta, 1_000_000.0);
    assert_eq!(cfg.rope_local_base_freq, Some(10_000.0));
    assert_eq!(cfg.query_pre_attn_scalar, Some(256.0));
    assert_eq!(cfg.sliding_window, Some(1024));
    assert_eq!(cfg.sliding_window_pattern, Some(6));
    // GeLU tanh : câblé en dur par gemma3_text, aucune clé déclarée.
    assert!(cfg.uses_gelu_tanh());
    assert_eq!(cfg.rope_position_scale(), Some(0.125));
    assert_eq!(cfg.eos_token_ids, vec![1, 106]);
    assert_eq!(cfg.quantization.as_ref().and_then(|q| q.bits), Some(4));
}

#[test]
fn resolves_gemma4_moe_text_config_with_layer_specific_defaults() {
    // Reflet compact du config.json mlx-community/gemma-4-26b-a4b-it-4bit.
    let raw = r#"{
            "architectures":["Gemma4ForConditionalGeneration"],
            "model_type":"gemma4",
            "text_config":{
                "model_type":"gemma4_text",
                "hidden_size":2816,
                "num_hidden_layers":30,
                "num_attention_heads":16,
                "num_key_value_heads":8,
                "num_global_key_value_heads":2,
                "head_dim":256,
                "global_head_dim":512,
                "intermediate_size":2112,
                "num_experts":128,
                "top_k_experts":8,
                "moe_intermediate_size":704,
                "rms_norm_eps":1e-06,
                "final_logit_softcapping":30.0,
                "attention_k_eq_v":true,
                "enable_moe_block":true,
                "layer_types":[
                    "sliding_attention","sliding_attention","sliding_attention",
                    "sliding_attention","sliding_attention","full_attention"
                ],
                "rope_parameters":{
                    "full_attention":{
                        "partial_rotary_factor":0.25,
                        "rope_theta":1000000.0,
                        "rope_type":"proportional"
                    },
                    "sliding_attention":{
                        "rope_theta":10000.0,
                        "rope_type":"default"
                    }
                }
            }
        }"#;
    let cfg: RawModelConfig = serde_json::from_str(raw).expect("invariant: JSON valide");
    let cfg = cfg.resolve().expect("invariant: config Gemma 4 valide");

    assert!(cfg.is_gemma());
    assert!(cfg.is_gemma4());
    assert!(cfg.uses_gelu_tanh());
    assert_eq!(cfg.vocab_size, 262_144);
    assert_eq!(cfg.num_experts_per_tok, Some(8));
    assert_eq!(cfg.top_k_experts, Some(8));
    assert_eq!(cfg.final_logit_softcapping, Some(30.0));
    assert!(cfg.attention_k_eq_v);
    assert!(cfg.enable_moe_block);
    assert!(cfg.tie_word_embeddings);
    assert_eq!(cfg.rope_theta, 1_000_000.0);
    assert_eq!(cfg.rope_local_base_freq, Some(10_000.0));
    assert_eq!(cfg.rope_full_base_freq, Some(1_000_000.0));
    assert_eq!(cfg.rope_full_partial_rotary_factor, Some(0.25));
    assert_eq!(cfg.rope_sliding_partial_rotary_factor, Some(1.0));
    assert!(cfg.is_gemma4_sliding_layer(0));
    assert!(cfg.is_gemma4_full_layer(5));
    assert_eq!(cfg.layer_head_dim(0), 256);
    assert_eq!(cfg.layer_head_dim(5), 512);
    assert_eq!(cfg.layer_num_key_value_heads(0), 8);
    assert_eq!(cfg.layer_num_key_value_heads(5), 2);
    assert_eq!(cfg.layer_rope_dims(0), 256);
    assert_eq!(cfg.layer_rope_dims(5), 128);
}

#[test]
fn gemma3_wrapper_vocab_overrides_text_config_value() {
    // gemma3.py __post_init__ : le vocab du wrapper (top-level, défaut
    // 262208) écrase TOUJOURS celui du text_config.
    let raw = r#"{
            "model_type":"gemma3",
            "vocab_size":262145,
            "text_config":{
                "hidden_size":2560,
                "num_hidden_layers":34,
                "vocab_size":7
            }
        }"#;
    let cfg: RawModelConfig = serde_json::from_str(raw).expect("invariant: JSON valide");
    let cfg = cfg.resolve().expect("invariant: config valide");
    assert_eq!(cfg.vocab_size, 262_145);
}

#[test]
fn gemma3_defaults_do_not_leak_outside_gemma3() {
    // Une config Qwen incomplète reste rejetée : les défauts ne
    // s'appliquent qu'aux model_type gemma3/gemma3_text.
    let raw = r#"{"model_type":"qwen3","hidden_size":1024,"num_hidden_layers":2}"#;
    let cfg: RawModelConfig = serde_json::from_str(raw).expect("invariant: JSON valide");
    assert!(cfg.resolve().is_err());

    // gemma2 (flat, complet) ne reçoit pas non plus les défauts Gemma 3.
    let raw = r#"{
            "model_type":"gemma2",
            "head_dim":256,
            "hidden_act":"gelu_pytorch_tanh",
            "hidden_size":2304,
            "intermediate_size":9216,
            "num_attention_heads":8,
            "num_hidden_layers":26,
            "num_key_value_heads":4,
            "rms_norm_eps":1e-06,
            "rope_theta":10000.0,
            "sliding_window":4096,
            "vocab_size":256000
        }"#;
    let cfg: RawModelConfig = serde_json::from_str(raw).expect("invariant: JSON valide");
    let cfg = cfg.resolve().expect("invariant: config Gemma 2 valide");
    assert_eq!(cfg.rope_local_base_freq, None);
    assert_eq!(cfg.sliding_window_pattern, None);
}

#[test]
fn rope_position_scale_follows_linear_rope_scaling() {
    // Reflet du text_config mlx-community/gemma-3-4b-it-4bit (couches
    // globales : linear ×8 → positions multipliées par 1/8).
    let raw = r#"{
            "model_type":"gemma3_text",
            "head_dim":256,
            "hidden_size":2560,
            "intermediate_size":10240,
            "num_attention_heads":8,
            "num_hidden_layers":34,
            "num_key_value_heads":4,
            "rms_norm_eps":1e-06,
            "rope_local_base_freq":10000,
            "rope_scaling":{"factor":8.0,"rope_type":"linear"},
            "rope_theta":1000000,
            "sliding_window":1024,
            "sliding_window_pattern":6,
            "vocab_size":262208
        }"#;
    let cfg: RawModelConfig = serde_json::from_str(raw).expect("invariant: JSON valide");
    let cfg = cfg.resolve().expect("invariant: config Gemma 3 4B valide");
    assert_eq!(cfg.rope_position_scale(), Some(0.125));
}

#[test]
fn rope_position_scale_ignores_non_linear_types() {
    // Llama 3.2 déclare un type `llama3` (non implémenté) : aucune échelle,
    // statu quo du chargeur générique (scaling ignoré hors Gemma).
    let scaling: RopeScalingConfig = serde_json::from_str(
        r#"{"factor":32.0,"high_freq_factor":4.0,"low_freq_factor":1.0,
                "original_max_position_embeddings":8192,"rope_type":"llama3"}"#,
    )
    .expect("invariant: rope_scaling llama3 parsable");
    assert_eq!(scaling.scaling_type(), "llama3");

    let mut cfg = reference_qwen_config();
    cfg.rope_scaling = Some(scaling);
    assert_eq!(cfg.rope_position_scale(), None);
}

#[test]
fn rope_scaling_legacy_type_key_takes_precedence() {
    // Configs HF en migration : `type` (historique) prime sur `rope_type`,
    // comme `initialize_rope` de mlx_lm.
    let scaling: RopeScalingConfig =
        serde_json::from_str(r#"{"type":"linear","rope_type":"default","factor":4.0}"#)
            .expect("invariant: rope_scaling double clé parsable");
    assert_eq!(scaling.scaling_type(), "linear");

    let mut cfg = reference_qwen_config();
    cfg.rope_scaling = Some(scaling);
    assert_eq!(cfg.rope_position_scale(), Some(0.25));
}

#[test]
fn rope_position_scale_rejects_invalid_factors() {
    let mut cfg = reference_qwen_config();
    for factor in [None, Some(0.0), Some(-8.0), Some(f32::NAN)] {
        cfg.rope_scaling = Some(RopeScalingConfig {
            rope_type: Some("linear".to_string()),
            legacy_type: None,
            factor,
        });
        assert_eq!(cfg.rope_position_scale(), None, "factor={factor:?}");
    }
}

fn reference_qwen_config() -> ModelConfig {
    let raw = r#"{
            "model_type":"qwen3",
            "hidden_size":1024,
            "num_hidden_layers":28,
            "num_attention_heads":16,
            "num_key_value_heads":8,
            "head_dim":128,
            "intermediate_size":3072,
            "rms_norm_eps":0.000001,
            "rope_theta":1000000.0,
            "vocab_size":151936
        }"#;
    let cfg: RawModelConfig = serde_json::from_str(raw).expect("invariant: JSON valide");
    cfg.resolve().expect("invariant: config Qwen valide")
}

#[test]
fn parses_gemma2_duplicate_activation_keys() {
    // Les configs Gemma 2 sérialisent `hidden_act` ET `hidden_activation` :
    // deux champs distincts (un alias serde casserait en `duplicate field`).
    let raw = r#"{
            "model_type":"gemma2",
            "attn_logit_softcapping":50.0,
            "final_logit_softcapping":30.0,
            "head_dim":256,
            "hidden_act":"gelu_pytorch_tanh",
            "hidden_activation":"gelu_pytorch_tanh",
            "hidden_size":2304,
            "intermediate_size":9216,
            "num_attention_heads":8,
            "num_hidden_layers":26,
            "num_key_value_heads":4,
            "query_pre_attn_scalar":256,
            "rms_norm_eps":1e-06,
            "rope_theta":10000.0,
            "sliding_window":4096,
            "vocab_size":256000
        }"#;
    let cfg: RawModelConfig = serde_json::from_str(raw).expect("invariant: JSON valide");
    let cfg = cfg.resolve().expect("invariant: config Gemma 2 parsable");
    assert!(cfg.is_gemma());
    assert!(cfg.uses_gelu_tanh());
    assert_eq!(cfg.attn_logit_softcapping, Some(50.0));
    assert_eq!(cfg.final_logit_softcapping, Some(30.0));
}

#[test]
fn uses_gelu_tanh_falls_back_to_hidden_act_key() {
    let raw = r#"{
            "model_type":"gemma",
            "hidden_act":"gelu_pytorch_tanh",
            "hidden_size":2048,
            "num_hidden_layers":18,
            "num_attention_heads":8,
            "num_key_value_heads":1,
            "head_dim":256,
            "intermediate_size":16384,
            "rms_norm_eps":0.000001,
            "rope_theta":10000.0,
            "vocab_size":256000
        }"#;
    let cfg: RawModelConfig = serde_json::from_str(raw).expect("invariant: JSON valide");
    let cfg = cfg.resolve().expect("invariant: config valide");
    assert!(cfg.uses_gelu_tanh());
}

#[test]
fn embed_scale_is_none_outside_gemma() {
    let raw = r#"{
            "model_type":"qwen3",
            "hidden_size":1024,
            "num_hidden_layers":28,
            "num_attention_heads":16,
            "num_key_value_heads":8,
            "head_dim":128,
            "intermediate_size":3072,
            "rms_norm_eps":0.000001,
            "rope_theta":1000000.0,
            "vocab_size":151936,
            "hidden_act":"silu"
        }"#;
    let cfg: RawModelConfig = serde_json::from_str(raw).expect("invariant: JSON valide");
    let cfg = cfg.resolve().expect("invariant: config valide");
    assert_eq!(cfg.embed_scale(), None);
    assert!(!cfg.uses_gelu_tanh());
}

#[test]
fn bf16_round_matches_reference_values() {
    assert_eq!(bf16_round(1.0), 1.0);
    assert_eq!(bf16_round(33.941_125), 34.0);
    assert_eq!(bf16_round(50.596_443), 50.5);
}

#[test]
fn resolves_top_level_eos_list_into_nested_text_config() {
    let raw = r#"{
            "model_type":"qwen3_5_moe",
            "eos_token_id":[248044,248045],
            "text_config":{
                "hidden_size":2048,
                "num_hidden_layers":40,
                "num_attention_heads":16,
                "num_key_value_heads":4,
                "head_dim":128,
                "moe_intermediate_size":512,
                "rms_norm_eps":0.000001,
                "vocab_size":248320,
                "rope_theta":10000000.0
            }
        }"#;
    let cfg: RawModelConfig = serde_json::from_str(raw).expect("invariant: JSON valide");
    let cfg = cfg.resolve().expect("invariant: config valide");
    assert_eq!(cfg.eos_token_ids, vec![248044, 248045]);
}
