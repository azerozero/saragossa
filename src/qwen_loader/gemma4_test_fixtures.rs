//! Fixtures Gemma 4 synthétiques du loader.

use super::*;
use safetensors::{serialize, Dtype, View};
use std::borrow::Cow;
use std::path::Path;

const PREFIX: &str = "language_model.model.";
const LM_HEAD_PREFIX: &str = "language_model.lm_head.";
const HIDDEN: usize = 4;
const INTERMEDIATE: usize = 4;
const LOCAL_HEAD_DIM: usize = 4;
const GLOBAL_HEAD_DIM: usize = 8;
const EXPERTS: usize = 2;
pub(super) const EMBED_VALUES: [f32; 12] = [
    0.11, -0.23, 0.37, 0.41, -0.17, 0.29, 0.53, -0.31, 0.47, 0.13, -0.59, 0.19,
];

pub(super) fn gemma4_moe_config() -> ModelConfig {
    let mut config = gemma4_dense_config("gemma4");
    config.enable_moe_block = true;
    config.num_experts = Some(EXPERTS);
    config.num_experts_per_tok = Some(1);
    config.top_k_experts = Some(1);
    config.moe_intermediate_size = Some(INTERMEDIATE);
    config
}

pub(super) fn gemma4_unified_dense_config() -> ModelConfig {
    gemma4_dense_config("gemma4_unified")
}

pub(super) fn write_gemma4_moe_safetensors(path: &Path) {
    write_gemma4_safetensors(path, true, false);
}

pub(super) fn write_gemma4_unified_dense_safetensors(path: &Path) {
    write_gemma4_safetensors(path, false, false);
}

pub(super) fn write_gemma4_unified_dense_explicit_lm_head(path: &Path) {
    write_gemma4_safetensors(path, false, true);
}

fn gemma4_dense_config(model_type: &str) -> ModelConfig {
    let mut config = super::test_fixtures::test_config();
    config.model_type = model_type.to_string();
    config.hidden_size = HIDDEN;
    config.num_hidden_layers = 2;
    config.num_attention_heads = 1;
    config.num_key_value_heads = 1;
    config.num_global_key_value_heads = Some(1);
    config.head_dim = Some(LOCAL_HEAD_DIM);
    config.global_head_dim = Some(GLOBAL_HEAD_DIM);
    config.intermediate_size = INTERMEDIATE;
    config.rms_norm_eps = 1.0e-6;
    config.rope_theta = 1_000_000.0;
    config.vocab_size = 3;
    config.tie_word_embeddings = true;
    config.hidden_activation = Some("gelu_pytorch_tanh".to_string());
    config.rope_local_base_freq = Some(10_000.0);
    config.rope_full_base_freq = Some(1_000_000.0);
    config.rope_full_partial_rotary_factor = Some(0.25);
    config.rope_sliding_partial_rotary_factor = Some(0.5);
    config.sliding_window = Some(2);
    config.layer_types = vec![
        "sliding_attention".to_string(),
        "full_attention".to_string(),
    ];
    config.attention_k_eq_v = true;
    config.query_pre_attn_scalar = Some(1.0);
    config.final_logit_softcapping = Some(30.0);
    config
}

fn write_gemma4_safetensors(path: &Path, moe: bool, explicit_lm_head: bool) {
    let mut tensors = vec![
        (
            format!("{PREFIX}embed_tokens.weight"),
            TensorFixture::f32(vec![3, HIDDEN], EMBED_VALUES.to_vec()),
        ),
        (
            format!("{PREFIX}norm.weight"),
            TensorFixture::ones(vec![HIDDEN]),
        ),
    ];
    if explicit_lm_head {
        tensors.push((
            format!("{LM_HEAD_PREFIX}weight"),
            TensorFixture::f32(vec![3, HIDDEN], EMBED_VALUES.to_vec()),
        ));
    }
    for layer in 0..2 {
        tensors.extend(gemma4_attention_layer(layer));
        tensors.extend(gemma4_dense_mlp_layer(layer));
        if moe {
            tensors.extend(gemma4_parallel_moe_layer(layer));
        }
    }
    let buffer = serialize(tensors, None).expect("invariant: safetensors sérialisable");
    std::fs::write(path, buffer).expect("invariant: écriture safetensors");
}

fn gemma4_attention_layer(layer: usize) -> Vec<(String, TensorFixture)> {
    let full = layer == 1;
    let head_dim = if full {
        GLOBAL_HEAD_DIM
    } else {
        LOCAL_HEAD_DIM
    };
    let mut tensors = vec![
        (
            format!("{PREFIX}layers.{layer}.input_layernorm.weight"),
            TensorFixture::ones(vec![HIDDEN]),
        ),
        (
            format!("{PREFIX}layers.{layer}.self_attn.q_proj.weight"),
            projection(head_dim, HIDDEN, 0.75),
        ),
        (
            format!("{PREFIX}layers.{layer}.self_attn.k_proj.weight"),
            projection(head_dim, HIDDEN, 0.5),
        ),
        (
            format!("{PREFIX}layers.{layer}.self_attn.o_proj.weight"),
            projection(HIDDEN, head_dim, 0.8),
        ),
        (
            format!("{PREFIX}layers.{layer}.self_attn.q_norm.weight"),
            TensorFixture::ones(vec![head_dim]),
        ),
        (
            format!("{PREFIX}layers.{layer}.self_attn.k_norm.weight"),
            TensorFixture::ones(vec![head_dim]),
        ),
    ];
    if !full {
        tensors.push((
            format!("{PREFIX}layers.{layer}.self_attn.v_proj.weight"),
            projection(head_dim, HIDDEN, 0.6),
        ));
    }
    tensors
}

fn gemma4_dense_mlp_layer(layer: usize) -> Vec<(String, TensorFixture)> {
    vec![
        (
            format!("{PREFIX}layers.{layer}.post_attention_layernorm.weight"),
            TensorFixture::ones(vec![HIDDEN]),
        ),
        (
            format!("{PREFIX}layers.{layer}.pre_feedforward_layernorm.weight"),
            TensorFixture::ones(vec![HIDDEN]),
        ),
        (
            format!("{PREFIX}layers.{layer}.post_feedforward_layernorm.weight"),
            TensorFixture::ones(vec![HIDDEN]),
        ),
        (
            format!("{PREFIX}layers.{layer}.layer_scalar"),
            TensorFixture::f32(vec![1], vec![1.0]),
        ),
        (
            format!("{PREFIX}layers.{layer}.mlp.gate_proj.weight"),
            projection(INTERMEDIATE, HIDDEN, 0.7),
        ),
        (
            format!("{PREFIX}layers.{layer}.mlp.up_proj.weight"),
            projection(INTERMEDIATE, HIDDEN, 0.9),
        ),
        (
            format!("{PREFIX}layers.{layer}.mlp.down_proj.weight"),
            projection(HIDDEN, INTERMEDIATE, 0.5),
        ),
    ]
}

fn gemma4_parallel_moe_layer(layer: usize) -> Vec<(String, TensorFixture)> {
    vec![
        (
            format!("{PREFIX}layers.{layer}.pre_feedforward_layernorm_2.weight"),
            TensorFixture::ones(vec![HIDDEN]),
        ),
        (
            format!("{PREFIX}layers.{layer}.post_feedforward_layernorm_1.weight"),
            TensorFixture::ones(vec![HIDDEN]),
        ),
        (
            format!("{PREFIX}layers.{layer}.post_feedforward_layernorm_2.weight"),
            TensorFixture::ones(vec![HIDDEN]),
        ),
        (
            format!("{PREFIX}layers.{layer}.router.proj.weight"),
            TensorFixture::f32(
                vec![EXPERTS, HIDDEN],
                vec![2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
            ),
        ),
        (
            format!("{PREFIX}layers.{layer}.router.scale"),
            TensorFixture::ones(vec![HIDDEN]),
        ),
        (
            format!("{PREFIX}layers.{layer}.router.per_expert_scale"),
            TensorFixture::ones(vec![EXPERTS]),
        ),
        (
            format!("{PREFIX}layers.{layer}.experts.switch_glu.gate_proj.weight"),
            expert_projection(EXPERTS, INTERMEDIATE, HIDDEN, 0.8),
        ),
        (
            format!("{PREFIX}layers.{layer}.experts.switch_glu.up_proj.weight"),
            expert_projection(EXPERTS, INTERMEDIATE, HIDDEN, 0.9),
        ),
        (
            format!("{PREFIX}layers.{layer}.experts.switch_glu.down_proj.weight"),
            expert_projection(EXPERTS, HIDDEN, INTERMEDIATE, 0.6),
        ),
    ]
}

fn projection(rows: usize, cols: usize, scale: f32) -> TensorFixture {
    let values = (0..rows)
        .flat_map(|row| (0..cols).map(move |col| if row % cols == col { scale } else { 0.0 }))
        .collect();
    TensorFixture::f32(vec![rows, cols], values)
}

fn expert_projection(experts: usize, rows: usize, cols: usize, scale: f32) -> TensorFixture {
    let values = (0..experts)
        .flat_map(|expert| {
            let expert_scale = scale * (expert as f32 + 1.0);
            (0..rows).flat_map(move |row| {
                (0..cols).map(
                    move |col| {
                        if row % cols == col {
                            expert_scale
                        } else {
                            0.0
                        }
                    },
                )
            })
        })
        .collect();
    TensorFixture::f32(vec![experts, rows, cols], values)
}

#[derive(Debug, Clone)]
struct TensorFixture {
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

    fn ones(shape: Vec<usize>) -> Self {
        let len = shape.iter().product();
        Self::f32(shape, vec![1.0; len])
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
