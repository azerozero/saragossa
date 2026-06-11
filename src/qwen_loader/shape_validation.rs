//! Validation des formes logiques du chargeur Qwen.

use super::*;

trait TensorShapes {
    fn shape_for(&self, name: &str) -> Option<&[usize]>;

    fn contains_tensor(&self, name: &str) -> bool {
        self.shape_for(name).is_some()
    }
}

impl TensorShapes for HashMap<String, DecoderTensor> {
    fn shape_for(&self, name: &str) -> Option<&[usize]> {
        self.get(name).map(DecoderTensor::shape)
    }
}

impl TensorShapes for HashMap<String, TensorMeta> {
    fn shape_for(&self, name: &str) -> Option<&[usize]> {
        self.get(name).map(|meta| meta.shape.as_slice())
    }
}

pub(super) fn validate_decoder_shapes(
    config: &ModelConfig,
    tensors: &HashMap<String, DecoderTensor>,
) -> Result<()> {
    validate_decoder_shape_map(config, tensors)
}

pub(super) fn validate_decoder_meta_shapes(
    config: &ModelConfig,
    tensors: &HashMap<String, TensorMeta>,
) -> Result<()> {
    validate_decoder_shape_map(config, tensors)
}

fn validate_decoder_shape_map<T: TensorShapes>(config: &ModelConfig, tensors: &T) -> Result<()> {
    let hidden = config.hidden_size;
    let vocab = config.vocab_size;
    let q_dim = config.num_attention_heads * config.head_dim();
    let kv_dim = config.num_key_value_heads * config.head_dim();
    expect_shape(tensors, "embed_tokens.weight", &[vocab, hidden])?;
    for layer in 0..config.num_hidden_layers {
        expect_shape(
            tensors,
            &layer_target(layer, "input_layernorm.weight"),
            &[hidden],
        )?;
        if config.is_full_attention_layer(layer) {
            validate_full_attention_shapes(config, tensors, layer, q_dim, kv_dim)?;
        } else {
            validate_linear_attention_shapes(config, tensors, layer)?;
        }
        validate_mlp_shapes(config, tensors, layer)?;
    }
    expect_shape(tensors, "norm.weight", &[hidden])?;
    expect_shape(tensors, "lm_head.weight", &[vocab, hidden])?;
    Ok(())
}

fn validate_full_attention_shapes(
    config: &ModelConfig,
    tensors: &impl TensorShapes,
    layer: usize,
    q_dim: usize,
    kv_dim: usize,
) -> Result<()> {
    let hidden = config.hidden_size;
    let q_proj_dim = if config.attn_output_gate.unwrap_or(false) {
        q_dim
            .checked_mul(2)
            .ok_or_else(|| InferError::Shape("q_proj gated dimension trop grande".to_string()))?
    } else {
        q_dim
    };
    expect_shape(
        tensors,
        &layer_target(layer, "self_attn.q_proj.weight"),
        &[q_proj_dim, hidden],
    )?;
    expect_shape(
        tensors,
        &layer_target(layer, "self_attn.k_proj.weight"),
        &[kv_dim, hidden],
    )?;
    expect_shape(
        tensors,
        &layer_target(layer, "self_attn.v_proj.weight"),
        &[kv_dim, hidden],
    )?;
    expect_shape(
        tensors,
        &layer_target(layer, "self_attn.o_proj.weight"),
        &[hidden, q_dim],
    )?;
    validate_qk_norm_shapes(config, tensors, layer)
}

fn validate_linear_attention_shapes(
    config: &ModelConfig,
    tensors: &impl TensorShapes,
    layer: usize,
) -> Result<()> {
    let hidden = config.hidden_size;
    let key_heads = config
        .linear_num_key_heads
        .ok_or_else(|| InferError::Config("linear_num_key_heads manquant".to_string()))?;
    let value_heads = config
        .linear_num_value_heads
        .ok_or_else(|| InferError::Config("linear_num_value_heads manquant".to_string()))?;
    let key_head_dim = config
        .linear_key_head_dim
        .ok_or_else(|| InferError::Config("linear_key_head_dim manquant".to_string()))?;
    let value_head_dim = config
        .linear_value_head_dim
        .ok_or_else(|| InferError::Config("linear_value_head_dim manquant".to_string()))?;
    let kernel = config
        .linear_conv_kernel_dim
        .ok_or_else(|| InferError::Config("linear_conv_kernel_dim manquant".to_string()))?;
    let key_dim = key_heads
        .checked_mul(key_head_dim)
        .ok_or_else(|| InferError::Shape("linear key_dim trop grand".to_string()))?;
    let value_dim = value_heads
        .checked_mul(value_head_dim)
        .ok_or_else(|| InferError::Shape("linear value_dim trop grand".to_string()))?;
    let conv_dim = key_dim
        .checked_mul(2)
        .and_then(|twice| twice.checked_add(value_dim))
        .ok_or_else(|| InferError::Shape("linear conv_dim trop grand".to_string()))?;

    expect_shape(
        tensors,
        &layer_target(layer, "linear_attn.in_proj_qkv.weight"),
        &[conv_dim, hidden],
    )?;
    expect_shape(
        tensors,
        &layer_target(layer, "linear_attn.in_proj_z.weight"),
        &[value_dim, hidden],
    )?;
    expect_shape(
        tensors,
        &layer_target(layer, "linear_attn.in_proj_b.weight"),
        &[value_heads, hidden],
    )?;
    expect_shape(
        tensors,
        &layer_target(layer, "linear_attn.in_proj_a.weight"),
        &[value_heads, hidden],
    )?;
    expect_shape(
        tensors,
        &layer_target(layer, "linear_attn.out_proj.weight"),
        &[hidden, value_dim],
    )?;
    expect_conv1d_shape(
        tensors,
        &layer_target(layer, "linear_attn.conv1d.weight"),
        conv_dim,
        kernel,
    )?;
    expect_shape(
        tensors,
        &layer_target(layer, "linear_attn.A_log"),
        &[value_heads],
    )?;
    expect_shape(
        tensors,
        &layer_target(layer, "linear_attn.dt_bias"),
        &[value_heads],
    )?;
    expect_shape(
        tensors,
        &layer_target(layer, "linear_attn.norm.weight"),
        &[value_head_dim],
    )?;
    Ok(())
}

fn validate_qk_norm_shapes(
    config: &ModelConfig,
    tensors: &impl TensorShapes,
    layer: usize,
) -> Result<()> {
    let q_norm = layer_target(layer, "self_attn.q_norm.weight");
    let k_norm = layer_target(layer, "self_attn.k_norm.weight");
    match (
        tensors.contains_tensor(&q_norm),
        tensors.contains_tensor(&k_norm),
    ) {
        (true, true) => {
            expect_shape(tensors, &q_norm, &[config.head_dim()])?;
            expect_shape(tensors, &k_norm, &[config.head_dim()])?;
            Ok(())
        }
        (false, false) => Ok(()),
        _ => Err(InferError::Config(format!(
            "q_norm/k_norm partiels pour couche {layer}"
        ))),
    }
}

fn expect_conv1d_shape(
    tensors: &impl TensorShapes,
    name: &str,
    channels: usize,
    kernel: usize,
) -> Result<()> {
    let shape = tensors
        .shape_for(name)
        .ok_or_else(|| InferError::MissingWeight(name.to_string()))?;
    match shape {
        [actual_channels, actual_kernel, one]
            if *actual_channels == channels && *actual_kernel == kernel && *one == 1 =>
        {
            Ok(())
        }
        [actual_channels, one, actual_kernel]
            if *actual_channels == channels && *one == 1 && *actual_kernel == kernel =>
        {
            Ok(())
        }
        shape => Err(InferError::Dimension(format!(
            "{name} attendu [{channels},{kernel},1] ou [{channels},1,{kernel}], reçu {shape:?}"
        ))),
    }
}

fn validate_mlp_shapes(
    config: &ModelConfig,
    tensors: &impl TensorShapes,
    layer: usize,
) -> Result<()> {
    let layer_mlp = MLP_LAYER_WEIGHTS
        .iter()
        .map(|suffix| layer_target(layer, suffix))
        .collect::<Vec<_>>();
    if !layer_mlp
        .iter()
        .filter(|target| !target.ends_with("post_attention_layernorm.weight"))
        .any(|target| tensors.contains_tensor(target))
    {
        return validate_moe_shapes(config, tensors, layer);
    }
    let hidden = config.hidden_size;
    let intermediate = config.mlp_intermediate_size();
    if intermediate == 0 {
        return Err(InferError::Config(
            "MLP présent sans intermediate_size exploitable".to_string(),
        ));
    }
    expect_shape(
        tensors,
        &layer_target(layer, "post_attention_layernorm.weight"),
        &[hidden],
    )?;
    expect_shape(
        tensors,
        &layer_target(layer, "mlp.gate_proj.weight"),
        &[intermediate, hidden],
    )?;
    expect_shape(
        tensors,
        &layer_target(layer, "mlp.up_proj.weight"),
        &[intermediate, hidden],
    )?;
    expect_shape(
        tensors,
        &layer_target(layer, "mlp.down_proj.weight"),
        &[hidden, intermediate],
    )?;
    Ok(())
}

fn validate_moe_shapes(
    config: &ModelConfig,
    tensors: &impl TensorShapes,
    layer: usize,
) -> Result<()> {
    let layer_moe = MOE_LAYER_WEIGHTS
        .iter()
        .map(|suffix| layer_target(layer, suffix))
        .collect::<Vec<_>>();
    if !layer_moe
        .iter()
        .filter(|target| !target.ends_with("post_attention_layernorm.weight"))
        .any(|target| tensors.contains_tensor(target))
    {
        return Ok(());
    }
    let hidden = config.hidden_size;
    let experts = config
        .num_experts
        .ok_or_else(|| InferError::Config("MoE présent sans num_experts".to_string()))?;
    let intermediate = config
        .moe_intermediate_size
        .ok_or_else(|| InferError::Config("MoE présent sans moe_intermediate_size".to_string()))?;
    expect_shape(
        tensors,
        &layer_target(layer, "post_attention_layernorm.weight"),
        &[hidden],
    )?;
    expect_shape(
        tensors,
        &layer_target(layer, "mlp.gate.weight"),
        &[experts, hidden],
    )?;
    expect_shape(
        tensors,
        &layer_target(layer, "mlp.switch_mlp.gate_proj.weight"),
        &[experts, intermediate, hidden],
    )?;
    expect_shape(
        tensors,
        &layer_target(layer, "mlp.switch_mlp.up_proj.weight"),
        &[experts, intermediate, hidden],
    )?;
    expect_shape(
        tensors,
        &layer_target(layer, "mlp.switch_mlp.down_proj.weight"),
        &[experts, hidden, intermediate],
    )?;
    if let Some(shared) = config.shared_expert_intermediate_size {
        expect_shape(
            tensors,
            &layer_target(layer, "mlp.shared_expert_gate.weight"),
            &[1, hidden],
        )?;
        expect_shape(
            tensors,
            &layer_target(layer, "mlp.shared_expert.gate_proj.weight"),
            &[shared, hidden],
        )?;
        expect_shape(
            tensors,
            &layer_target(layer, "mlp.shared_expert.up_proj.weight"),
            &[shared, hidden],
        )?;
        expect_shape(
            tensors,
            &layer_target(layer, "mlp.shared_expert.down_proj.weight"),
            &[hidden, shared],
        )?;
    }
    Ok(())
}

fn expect_shape(tensors: &impl TensorShapes, name: &str, expected: &[usize]) -> Result<()> {
    let shape = tensors
        .shape_for(name)
        .ok_or_else(|| InferError::MissingWeight(name.to_string()))?;
    if shape != expected {
        return Err(InferError::Dimension(format!(
            "{name} attendu {:?}, reçu {:?}",
            expected, shape
        )));
    }
    Ok(())
}
