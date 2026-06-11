//! Fixtures safetensors synthétiques du loader Qwen.

use super::*;
use safetensors::{serialize, Dtype, View};
use std::borrow::Cow;

pub(super) fn test_config() -> ModelConfig {
    ModelConfig {
        model_type: "qwen3".to_string(),
        hidden_size: 2,
        num_hidden_layers: 1,
        num_attention_heads: 1,
        num_key_value_heads: 1,
        head_dim: Some(2),
        intermediate_size: 4,
        rms_norm_eps: 1.0e-6,
        rope_theta: 10_000.0,
        vocab_size: 3,
        eos_token_ids: Vec::new(),
        tie_word_embeddings: false,
        quantization: None,
        full_attention_interval: None,
        attn_output_gate: None,
        partial_rotary_factor: None,
        linear_num_value_heads: None,
        linear_num_key_heads: None,
        linear_key_head_dim: None,
        linear_value_head_dim: None,
        linear_conv_kernel_dim: None,
        num_experts: None,
        num_experts_per_tok: None,
        moe_intermediate_size: None,
        shared_expert_intermediate_size: None,
        mtp_num_hidden_layers: None,
    }
}

pub(super) fn write_safetensors(
    path: &Path,
    weight_prefix: &str,
    lm_head_prefix: &str,
    override_tensor: Option<(&str, TensorFixture)>,
) {
    let mut tensors = base_tensors(weight_prefix, lm_head_prefix);
    if let Some((name, fixture)) = override_tensor {
        tensors.retain(|(existing, _)| existing != name);
        tensors.push((name.to_string(), fixture));
    }
    let buffer = serialize(tensors, None).expect("invariant: safetensors sérialisable");
    std::fs::write(path, buffer).expect("invariant: écriture safetensors");
}

pub(super) fn write_safetensors_with_mlp(path: &Path) {
    let mut tensors = base_tensors("model.", "lm_head.");
    tensors.extend([
        (
            "model.layers.0.post_attention_layernorm.weight".to_string(),
            TensorFixture::ones(vec![2]),
        ),
        (
            "model.layers.0.mlp.gate_proj.weight".to_string(),
            TensorFixture::f32(vec![4, 2], vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5, -0.5, 0.5]),
        ),
        (
            "model.layers.0.mlp.up_proj.weight".to_string(),
            TensorFixture::f32(vec![4, 2], vec![1.0, 0.0, 0.0, 1.0, 0.25, 0.75, 0.75, 0.25]),
        ),
        (
            "model.layers.0.mlp.down_proj.weight".to_string(),
            TensorFixture::f32(vec![2, 4], vec![0.5, 0.0, 0.25, 0.0, 0.0, 0.5, 0.0, 0.25]),
        ),
    ]);
    let buffer = serialize(tensors, None).expect("invariant: safetensors sérialisable");
    std::fs::write(path, buffer).expect("invariant: écriture safetensors");
}

pub(super) fn write_layered_safetensors(path: &Path, layer_count: usize) {
    let mut tensors = base_tensors("model.", "lm_head.");
    for layer in 1..layer_count {
        tensors.extend(attention_layer_tensors("model.", layer, 2));
    }
    let buffer = serialize(tensors, None).expect("invariant: safetensors sérialisable");
    std::fs::write(path, buffer).expect("invariant: écriture safetensors");
}

pub(super) fn write_gqa_safetensors(path: &Path) {
    let tensors = vec![
        (
            "model.embed_tokens.weight".to_string(),
            TensorFixture::f32(
                vec![3, 4],
                vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5],
            ),
        ),
        (
            "model.layers.0.input_layernorm.weight".to_string(),
            TensorFixture::ones(vec![4]),
        ),
        (
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            TensorFixture::identity4(),
        ),
        (
            "model.layers.0.self_attn.k_proj.weight".to_string(),
            TensorFixture::f32(vec![2, 4], vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        ),
        (
            "model.layers.0.self_attn.v_proj.weight".to_string(),
            TensorFixture::f32(vec![2, 4], vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        ),
        (
            "model.layers.0.self_attn.o_proj.weight".to_string(),
            TensorFixture::identity4(),
        ),
        (
            "model.norm.weight".to_string(),
            TensorFixture::ones(vec![4]),
        ),
        (
            "lm_head.weight".to_string(),
            TensorFixture::f32(
                vec![3, 4],
                vec![1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            ),
        ),
    ];
    let buffer = serialize(tensors, None).expect("invariant: safetensors sérialisable");
    std::fs::write(path, buffer).expect("invariant: écriture safetensors");
}

pub(super) fn write_head4_safetensors(path: &Path) {
    let tensors = vec![
        (
            "model.embed_tokens.weight".to_string(),
            TensorFixture::f32(
                vec![3, 4],
                vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5],
            ),
        ),
        (
            "model.layers.0.input_layernorm.weight".to_string(),
            TensorFixture::ones(vec![4]),
        ),
        (
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            TensorFixture::identity4(),
        ),
        (
            "model.layers.0.self_attn.k_proj.weight".to_string(),
            TensorFixture::identity4(),
        ),
        (
            "model.layers.0.self_attn.v_proj.weight".to_string(),
            TensorFixture::identity4(),
        ),
        (
            "model.layers.0.self_attn.o_proj.weight".to_string(),
            TensorFixture::identity4(),
        ),
        (
            "model.norm.weight".to_string(),
            TensorFixture::ones(vec![4]),
        ),
        (
            "lm_head.weight".to_string(),
            TensorFixture::f32(
                vec![3, 4],
                vec![1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            ),
        ),
    ];
    let buffer = serialize(tensors, None).expect("invariant: safetensors sérialisable");
    std::fs::write(path, buffer).expect("invariant: écriture safetensors");
}

pub(super) fn write_qk_norm_safetensors(path: &Path) {
    let mut tensors = base_tensors("model.", "lm_head.");
    tensors.extend([
        (
            "model.layers.0.self_attn.q_norm.weight".to_string(),
            TensorFixture::f32(vec![2], vec![2.0, 0.25]),
        ),
        (
            "model.layers.0.self_attn.k_norm.weight".to_string(),
            TensorFixture::f32(vec![2], vec![0.25, 2.0]),
        ),
    ]);
    let buffer = serialize(tensors, None).expect("invariant: safetensors sérialisable");
    std::fs::write(path, buffer).expect("invariant: écriture safetensors");
}

pub(super) fn write_quantized_q_proj_safetensors(path: &Path) {
    let mut tensors = vec![
        (
            "model.embed_tokens.weight".to_string(),
            TensorFixture::f32(
                vec![3, 4],
                vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5],
            ),
        ),
        (
            "model.layers.0.input_layernorm.weight".to_string(),
            TensorFixture::ones(vec![4]),
        ),
        (
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            TensorFixture::u32(
                vec![4, 1],
                vec![
                    pack_lanes(&[255, 0, 0, 0], 8),
                    pack_lanes(&[0, 255, 0, 0], 8),
                    pack_lanes(&[0, 0, 255, 0], 8),
                    pack_lanes(&[0, 0, 0, 255], 8),
                ],
            ),
        ),
        (
            "model.layers.0.self_attn.q_proj.scales".to_string(),
            TensorFixture::f32(vec![4, 1], vec![1.0 / 255.0; 4]),
        ),
        (
            "model.layers.0.self_attn.q_proj.biases".to_string(),
            TensorFixture::f32(vec![4, 1], vec![0.0; 4]),
        ),
        (
            "model.layers.0.self_attn.k_proj.weight".to_string(),
            TensorFixture::identity4(),
        ),
        (
            "model.layers.0.self_attn.v_proj.weight".to_string(),
            TensorFixture::identity4(),
        ),
        (
            "model.layers.0.self_attn.o_proj.weight".to_string(),
            TensorFixture::identity4(),
        ),
        (
            "model.norm.weight".to_string(),
            TensorFixture::ones(vec![4]),
        ),
        (
            "lm_head.weight".to_string(),
            TensorFixture::f32(
                vec![3, 4],
                vec![1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            ),
        ),
    ];
    tensors.extend([
        (
            "model.layers.0.post_attention_layernorm.weight".to_string(),
            TensorFixture::ones(vec![4]),
        ),
        (
            "model.layers.0.mlp.gate_proj.weight".to_string(),
            TensorFixture::identity4(),
        ),
        (
            "model.layers.0.mlp.up_proj.weight".to_string(),
            TensorFixture::identity4(),
        ),
        (
            "model.layers.0.mlp.down_proj.weight".to_string(),
            TensorFixture::identity4(),
        ),
    ]);
    let buffer = serialize(tensors, None).expect("invariant: safetensors sérialisable");
    std::fs::write(path, buffer).expect("invariant: écriture safetensors");
}

pub(super) fn write_quantized_q_proj_split_safetensors(weights: &Path, companions: &Path) {
    let mut tensors = vec![
        (
            "model.embed_tokens.weight".to_string(),
            TensorFixture::f32(
                vec![3, 4],
                vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5],
            ),
        ),
        (
            "model.layers.0.input_layernorm.weight".to_string(),
            TensorFixture::ones(vec![4]),
        ),
        (
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            TensorFixture::u32(
                vec![4, 1],
                vec![
                    pack_lanes(&[255, 0, 0, 0], 8),
                    pack_lanes(&[0, 255, 0, 0], 8),
                    pack_lanes(&[0, 0, 255, 0], 8),
                    pack_lanes(&[0, 0, 0, 255], 8),
                ],
            ),
        ),
        (
            "model.layers.0.self_attn.k_proj.weight".to_string(),
            TensorFixture::identity4(),
        ),
        (
            "model.layers.0.self_attn.v_proj.weight".to_string(),
            TensorFixture::identity4(),
        ),
        (
            "model.layers.0.self_attn.o_proj.weight".to_string(),
            TensorFixture::identity4(),
        ),
        (
            "model.norm.weight".to_string(),
            TensorFixture::ones(vec![4]),
        ),
        (
            "lm_head.weight".to_string(),
            TensorFixture::f32(
                vec![3, 4],
                vec![1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            ),
        ),
    ];
    tensors.extend([
        (
            "model.layers.0.post_attention_layernorm.weight".to_string(),
            TensorFixture::ones(vec![4]),
        ),
        (
            "model.layers.0.mlp.gate_proj.weight".to_string(),
            TensorFixture::identity4(),
        ),
        (
            "model.layers.0.mlp.up_proj.weight".to_string(),
            TensorFixture::identity4(),
        ),
        (
            "model.layers.0.mlp.down_proj.weight".to_string(),
            TensorFixture::identity4(),
        ),
    ]);
    let buffer = serialize(tensors, None).expect("invariant: safetensors sérialisable");
    std::fs::write(weights, buffer).expect("invariant: écriture weights");
    let buffer = serialize(
        [
            (
                "model.layers.0.self_attn.q_proj.scales",
                TensorFixture::f32(vec![4, 1], vec![1.0 / 255.0; 4]),
            ),
            (
                "model.layers.0.self_attn.q_proj.biases",
                TensorFixture::f32(vec![4, 1], vec![0.0; 4]),
            ),
        ],
        None,
    )
    .expect("invariant: safetensors companions sérialisable");
    std::fs::write(companions, buffer).expect("invariant: écriture companions");
}

pub(super) fn write_quantized_embedding_safetensors(path: &Path) {
    let mut tensors = vec![
        (
            "model.embed_tokens.weight".to_string(),
            TensorFixture::u32(
                vec![3, 1],
                vec![
                    pack_lanes(&[255, 0, 0, 0], 8),
                    pack_lanes(&[0, 255, 0, 0], 8),
                    pack_lanes(&[128, 128, 128, 128], 8),
                ],
            ),
        ),
        (
            "model.embed_tokens.scales".to_string(),
            TensorFixture::f32(vec![3, 1], vec![1.0 / 255.0; 3]),
        ),
        (
            "model.embed_tokens.biases".to_string(),
            TensorFixture::f32(vec![3, 1], vec![0.0; 3]),
        ),
        (
            "model.layers.0.input_layernorm.weight".to_string(),
            TensorFixture::ones(vec![4]),
        ),
        (
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            TensorFixture::identity4(),
        ),
        (
            "model.layers.0.self_attn.k_proj.weight".to_string(),
            TensorFixture::identity4(),
        ),
        (
            "model.layers.0.self_attn.v_proj.weight".to_string(),
            TensorFixture::identity4(),
        ),
        (
            "model.layers.0.self_attn.o_proj.weight".to_string(),
            TensorFixture::identity4(),
        ),
        (
            "model.norm.weight".to_string(),
            TensorFixture::ones(vec![4]),
        ),
        (
            "lm_head.weight".to_string(),
            TensorFixture::f32(
                vec![3, 4],
                vec![1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            ),
        ),
    ];
    tensors.extend([
        (
            "model.layers.0.post_attention_layernorm.weight".to_string(),
            TensorFixture::ones(vec![4]),
        ),
        (
            "model.layers.0.mlp.gate_proj.weight".to_string(),
            TensorFixture::identity4(),
        ),
        (
            "model.layers.0.mlp.up_proj.weight".to_string(),
            TensorFixture::identity4(),
        ),
        (
            "model.layers.0.mlp.down_proj.weight".to_string(),
            TensorFixture::identity4(),
        ),
    ]);
    let buffer = serialize(tensors, None).expect("invariant: safetensors sérialisable");
    std::fs::write(path, buffer).expect("invariant: écriture safetensors");
}

pub(super) fn write_fp8_q_proj_safetensors(path: &Path) {
    let mut tensors = base_tensors("model.", "lm_head.");
    tensors.retain(|(name, _)| name != "model.layers.0.self_attn.q_proj.weight");
    tensors.extend([
        (
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            TensorFixture::f8_e4m3(vec![2, 2], vec![0x38, 0x00, 0x00, 0x38]),
        ),
        (
            "model.layers.0.self_attn.q_proj.weight_scale_inv".to_string(),
            TensorFixture::f32(vec![1], vec![2.0]),
        ),
    ]);
    let buffer = serialize(tensors, None).expect("invariant: safetensors sérialisable");
    std::fs::write(path, buffer).expect("invariant: écriture safetensors");
}

pub(super) fn write_moe_safetensors(path: &Path) {
    let mut tensors = base_tensors("model.", "lm_head.");
    tensors.extend([
        (
            "model.layers.0.post_attention_layernorm.weight".to_string(),
            TensorFixture::ones(vec![2]),
        ),
        (
            "model.layers.0.mlp.gate.weight".to_string(),
            TensorFixture::f32(vec![2, 2], vec![4.0, 0.0, 0.0, 4.0]),
        ),
        (
            "model.layers.0.mlp.switch_mlp.gate_proj.weight".to_string(),
            TensorFixture::f32(vec![2, 2, 2], vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.5]),
        ),
        (
            "model.layers.0.mlp.switch_mlp.up_proj.weight".to_string(),
            TensorFixture::f32(vec![2, 2, 2], vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.5]),
        ),
        (
            "model.layers.0.mlp.switch_mlp.down_proj.weight".to_string(),
            TensorFixture::f32(vec![2, 2, 2], vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.5]),
        ),
        (
            "model.layers.0.mlp.shared_expert_gate.weight".to_string(),
            TensorFixture::f32(vec![1, 2], vec![1.0, 1.0]),
        ),
        (
            "model.layers.0.mlp.shared_expert.gate_proj.weight".to_string(),
            TensorFixture::identity2(),
        ),
        (
            "model.layers.0.mlp.shared_expert.up_proj.weight".to_string(),
            TensorFixture::identity2(),
        ),
        (
            "model.layers.0.mlp.shared_expert.down_proj.weight".to_string(),
            TensorFixture::identity2(),
        ),
    ]);
    let buffer = serialize(tensors, None).expect("invariant: safetensors sérialisable");
    std::fs::write(path, buffer).expect("invariant: écriture safetensors");
}

pub(super) fn write_moe_safetensors_without_shared(path: &Path) {
    let mut tensors = base_tensors("model.", "lm_head.");
    tensors.extend([
        (
            "model.layers.0.post_attention_layernorm.weight".to_string(),
            TensorFixture::ones(vec![2]),
        ),
        (
            "model.layers.0.mlp.gate.weight".to_string(),
            TensorFixture::f32(vec![2, 2], vec![4.0, 0.0, 0.0, 4.0]),
        ),
        (
            "model.layers.0.mlp.switch_mlp.gate_proj.weight".to_string(),
            TensorFixture::f32(vec![2, 2, 2], vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.5]),
        ),
        (
            "model.layers.0.mlp.switch_mlp.up_proj.weight".to_string(),
            TensorFixture::f32(vec![2, 2, 2], vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.5]),
        ),
        (
            "model.layers.0.mlp.switch_mlp.down_proj.weight".to_string(),
            TensorFixture::f32(vec![2, 2, 2], vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.5]),
        ),
    ]);
    let buffer = serialize(tensors, None).expect("invariant: safetensors sérialisable");
    std::fs::write(path, buffer).expect("invariant: écriture safetensors");
}

pub(super) fn write_hybrid_safetensors(path: &Path) {
    let mut tensors = vec![
        (
            "model.embed_tokens.weight".to_string(),
            TensorFixture::f32(vec![3, 2], vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
        ),
        (
            "model.norm.weight".to_string(),
            TensorFixture::ones(vec![2]),
        ),
        (
            "lm_head.weight".to_string(),
            TensorFixture::f32(vec![3, 2], vec![1.0, 0.0, -1.0, 0.0, 0.0, 1.0]),
        ),
        (
            "model.layers.0.input_layernorm.weight".to_string(),
            TensorFixture::ones(vec![2]),
        ),
        (
            "model.layers.0.linear_attn.in_proj_qkv.weight".to_string(),
            TensorFixture::f32(
                vec![6, 2],
                vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 1.0],
            ),
        ),
        (
            "model.layers.0.linear_attn.in_proj_z.weight".to_string(),
            TensorFixture::identity2(),
        ),
        (
            "model.layers.0.linear_attn.in_proj_b.weight".to_string(),
            TensorFixture::f32(vec![1, 2], vec![1.0, 0.0]),
        ),
        (
            "model.layers.0.linear_attn.in_proj_a.weight".to_string(),
            TensorFixture::f32(vec![1, 2], vec![0.0, 1.0]),
        ),
        (
            "model.layers.0.linear_attn.out_proj.weight".to_string(),
            TensorFixture::identity2(),
        ),
        (
            "model.layers.0.linear_attn.conv1d.weight".to_string(),
            TensorFixture::f32(
                vec![6, 2, 1],
                vec![
                    0.25, 1.0, 0.25, 1.0, 0.25, 1.0, 0.25, 1.0, 0.25, 1.0, 0.25, 1.0,
                ],
            ),
        ),
        (
            "model.layers.0.linear_attn.A_log".to_string(),
            TensorFixture::f32(vec![1], vec![0.0]),
        ),
        (
            "model.layers.0.linear_attn.dt_bias".to_string(),
            TensorFixture::f32(vec![1], vec![0.0]),
        ),
        (
            "model.layers.0.linear_attn.norm.weight".to_string(),
            TensorFixture::ones(vec![2]),
        ),
    ];
    tensors.extend(gated_attention_layer_tensors("model.", 1));
    let buffer = serialize(tensors, None).expect("invariant: safetensors sérialisable");
    std::fs::write(path, buffer).expect("invariant: écriture safetensors");
}

pub(super) fn gated_attention_layer_tensors(
    weight_prefix: &str,
    layer: usize,
) -> Vec<(String, TensorFixture)> {
    vec![
        (
            format!("{weight_prefix}layers.{layer}.input_layernorm.weight"),
            TensorFixture::ones(vec![2]),
        ),
        (
            format!("{weight_prefix}layers.{layer}.self_attn.q_proj.weight"),
            TensorFixture::f32(vec![4, 2], vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0]),
        ),
        (
            format!("{weight_prefix}layers.{layer}.self_attn.k_proj.weight"),
            TensorFixture::identity2(),
        ),
        (
            format!("{weight_prefix}layers.{layer}.self_attn.v_proj.weight"),
            TensorFixture::identity2(),
        ),
        (
            format!("{weight_prefix}layers.{layer}.self_attn.o_proj.weight"),
            TensorFixture::identity2(),
        ),
        (
            format!("{weight_prefix}layers.{layer}.self_attn.q_norm.weight"),
            TensorFixture::ones(vec![2]),
        ),
        (
            format!("{weight_prefix}layers.{layer}.self_attn.k_norm.weight"),
            TensorFixture::ones(vec![2]),
        ),
    ]
}

pub(super) fn write_quantized_moe_safetensors(path: &Path) {
    let mut tensors = vec![
        (
            "model.embed_tokens.weight".to_string(),
            TensorFixture::f32(
                vec![3, 4],
                vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5],
            ),
        ),
        (
            "model.layers.0.input_layernorm.weight".to_string(),
            TensorFixture::ones(vec![4]),
        ),
        (
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            TensorFixture::identity4(),
        ),
        (
            "model.layers.0.self_attn.k_proj.weight".to_string(),
            TensorFixture::identity4(),
        ),
        (
            "model.layers.0.self_attn.v_proj.weight".to_string(),
            TensorFixture::identity4(),
        ),
        (
            "model.layers.0.self_attn.o_proj.weight".to_string(),
            TensorFixture::identity4(),
        ),
        (
            "model.layers.0.post_attention_layernorm.weight".to_string(),
            TensorFixture::ones(vec![4]),
        ),
        (
            "model.norm.weight".to_string(),
            TensorFixture::ones(vec![4]),
        ),
        (
            "lm_head.weight".to_string(),
            TensorFixture::f32(
                vec![3, 4],
                vec![1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            ),
        ),
    ];
    tensors.extend(quantized_linear_2d(
        "model.layers.0.mlp.gate",
        vec![2, 1],
        vec![
            pack_lanes(&[255, 0, 0, 0], 8),
            pack_lanes(&[0, 255, 0, 0], 8),
        ],
        vec![2, 1],
    ));
    for suffix in ["gate_proj", "up_proj", "down_proj"] {
        tensors.extend(quantized_expert_identity4(&format!(
            "model.layers.0.mlp.switch_mlp.{suffix}"
        )));
    }
    tensors.extend(quantized_linear_2d(
        "model.layers.0.mlp.shared_expert_gate",
        vec![1, 1],
        vec![pack_lanes(&[255, 255, 255, 255], 8)],
        vec![1, 1],
    ));
    for suffix in ["gate_proj", "up_proj", "down_proj"] {
        tensors.extend(quantized_linear_2d(
            &format!("model.layers.0.mlp.shared_expert.{suffix}"),
            vec![4, 1],
            identity4_packed(),
            vec![4, 1],
        ));
    }
    let buffer = serialize(tensors, None).expect("invariant: safetensors sérialisable");
    std::fs::write(path, buffer).expect("invariant: écriture safetensors");
}

pub(super) fn quantized_expert_identity4(prefix: &str) -> Vec<(String, TensorFixture)> {
    let mut packed = Vec::new();
    packed.extend(identity4_packed());
    packed.extend(identity4_packed());
    quantized_linear_2d(prefix, vec![2, 4, 1], packed, vec![2, 4, 1])
}

pub(super) fn quantized_linear_2d(
    prefix: &str,
    weight_shape: Vec<usize>,
    packed: Vec<u32>,
    affine_shape: Vec<usize>,
) -> Vec<(String, TensorFixture)> {
    let affine_len = affine_shape.iter().product();
    vec![
        (
            format!("{prefix}.weight"),
            TensorFixture::u32(weight_shape, packed),
        ),
        (
            format!("{prefix}.scales"),
            TensorFixture::f32(affine_shape.clone(), vec![1.0 / 255.0; affine_len]),
        ),
        (
            format!("{prefix}.biases"),
            TensorFixture::f32(affine_shape, vec![0.0; affine_len]),
        ),
    ]
}

pub(super) fn identity4_packed() -> Vec<u32> {
    vec![
        pack_lanes(&[255, 0, 0, 0], 8),
        pack_lanes(&[0, 255, 0, 0], 8),
        pack_lanes(&[0, 0, 255, 0], 8),
        pack_lanes(&[0, 0, 0, 255], 8),
    ]
}

pub(super) fn base_tensors(
    weight_prefix: &str,
    lm_head_prefix: &str,
) -> Vec<(String, TensorFixture)> {
    let mut tensors = vec![
        (
            format!("{weight_prefix}embed_tokens.weight"),
            TensorFixture::f32(vec![3, 2], vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
        ),
        (
            format!("{weight_prefix}norm.weight"),
            TensorFixture::ones(vec![2]),
        ),
        (
            format!("{lm_head_prefix}weight"),
            TensorFixture::f32(vec![3, 2], vec![1.0, 0.0, -1.0, 0.0, 0.0, 1.0]),
        ),
    ];
    tensors.extend(attention_layer_tensors(weight_prefix, 0, 2));
    tensors
}

pub(super) fn attention_layer_tensors(
    weight_prefix: &str,
    layer: usize,
    hidden: usize,
) -> Vec<(String, TensorFixture)> {
    let identity = if hidden == 2 {
        TensorFixture::identity2
    } else {
        TensorFixture::identity4
    };
    vec![
        (
            format!("{weight_prefix}layers.{layer}.input_layernorm.weight"),
            TensorFixture::ones(vec![hidden]),
        ),
        (
            format!("{weight_prefix}layers.{layer}.self_attn.q_proj.weight"),
            identity(),
        ),
        (
            format!("{weight_prefix}layers.{layer}.self_attn.k_proj.weight"),
            identity(),
        ),
        (
            format!("{weight_prefix}layers.{layer}.self_attn.v_proj.weight"),
            identity(),
        ),
        (
            format!("{weight_prefix}layers.{layer}.self_attn.o_proj.weight"),
            identity(),
        ),
    ]
}

#[derive(Debug, Clone)]
pub(super) struct TensorFixture {
    dtype: Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl TensorFixture {
    fn f32(shape: Vec<usize>, values: Vec<f32>) -> Self {
        Self {
            dtype: Dtype::F32,
            shape,
            data: values.into_iter().flat_map(f32::to_le_bytes).collect(),
        }
    }

    pub(super) fn ones(shape: Vec<usize>) -> Self {
        let len = shape.iter().product();
        Self::f32(shape, vec![1.0; len])
    }

    fn identity2() -> Self {
        Self::f32(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0])
    }

    fn identity4() -> Self {
        Self::f32(
            vec![4, 4],
            vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
        )
    }

    fn u32(shape: Vec<usize>, values: Vec<u32>) -> Self {
        Self {
            dtype: Dtype::U32,
            shape,
            data: values.into_iter().flat_map(u32::to_le_bytes).collect(),
        }
    }

    fn f8_e4m3(shape: Vec<usize>, data: Vec<u8>) -> Self {
        Self {
            dtype: Dtype::F8_E4M3,
            shape,
            data,
        }
    }

    pub(super) fn f16_zeros(shape: Vec<usize>) -> Self {
        let len = shape.iter().product::<usize>() * 2;
        Self {
            dtype: Dtype::F16,
            shape,
            data: vec![0_u8; len],
        }
    }

    pub(super) fn i32_zeros(shape: Vec<usize>) -> Self {
        let len = shape.iter().product::<usize>() * 4;
        Self {
            dtype: Dtype::I32,
            shape,
            data: vec![0_u8; len],
        }
    }
}

impl View for TensorFixture {
    fn dtype(&self) -> Dtype {
        self.dtype
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

pub(super) fn pack_lanes(values: &[u32], bits: usize) -> u32 {
    values
        .iter()
        .enumerate()
        .fold(0_u32, |word, (idx, value)| word | (value << (idx * bits)))
}
