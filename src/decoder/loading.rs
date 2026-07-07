//! Assemblage des poids du décodeur depuis les tensors chargés.

use super::*;

pub(in crate::decoder) fn linear_from(
    tensors: &mut HashMap<String, DecoderTensor>,
    prefix: &str,
) -> Result<Linear> {
    let weight = take_linear_weight(tensors, &format!("{prefix}.weight"))?;
    let bias = take_optional_dense(tensors, &format!("{prefix}.bias"))?;
    Linear::from_weight(weight, bias)
}

pub(in crate::decoder) fn optional_mlp_from(
    config: &CausalDecoderConfig,
    tensors: &mut HashMap<String, DecoderTensor>,
    layer_prefix: &str,
) -> Result<(Option<Tensor>, Option<FeedForward>)> {
    let keys = [
        format!("{layer_prefix}.mlp.gate_proj.weight"),
        format!("{layer_prefix}.mlp.up_proj.weight"),
        format!("{layer_prefix}.mlp.down_proj.weight"),
    ];
    if !keys.iter().any(|key| tensors.contains_key(key)) {
        return optional_moe_from(config, tensors, layer_prefix);
    }
    let post_norm_key = format!("{layer_prefix}.post_attention_layernorm.weight");
    let post_attention_norm = take_dense(tensors, &post_norm_key)?;
    let gate_proj = linear_from(tensors, &format!("{layer_prefix}.mlp.gate_proj"))?;
    let up_proj = linear_from(tensors, &format!("{layer_prefix}.mlp.up_proj"))?;
    let down_proj = linear_from(tensors, &format!("{layer_prefix}.mlp.down_proj"))?;
    Ok((
        Some(post_attention_norm),
        Some(FeedForward::Dense(Box::new(
            GatedMlp::new(gate_proj, up_proj, down_proj).with_activation(config.activation),
        ))),
    ))
}

pub(in crate::decoder) fn optional_moe_from(
    config: &CausalDecoderConfig,
    tensors: &mut HashMap<String, DecoderTensor>,
    layer_prefix: &str,
) -> Result<(Option<Tensor>, Option<FeedForward>)> {
    let keys = [
        format!("{layer_prefix}.mlp.gate.weight"),
        format!("{layer_prefix}.mlp.switch_mlp.gate_proj.weight"),
        format!("{layer_prefix}.mlp.switch_mlp.up_proj.weight"),
        format!("{layer_prefix}.mlp.switch_mlp.down_proj.weight"),
    ];
    if !keys.iter().any(|key| tensors.contains_key(key)) {
        return Ok((None, None));
    }

    let expert_count = config
        .num_experts
        .ok_or_else(|| InferError::Config("poids MoE présents sans num_experts".to_string()))?;
    let post_norm_key = format!("{layer_prefix}.post_attention_layernorm.weight");
    let post_attention_norm = take_dense(tensors, &post_norm_key)?;
    let router = linear_from(tensors, &format!("{layer_prefix}.mlp.gate"))?;
    let gate_weights = take_expert_linear_weights(
        tensors,
        &format!("{layer_prefix}.mlp.switch_mlp.gate_proj.weight"),
        expert_count,
    )?;
    let up_weights = take_expert_linear_weights(
        tensors,
        &format!("{layer_prefix}.mlp.switch_mlp.up_proj.weight"),
        expert_count,
    )?;
    let down_weights = take_expert_linear_weights(
        tensors,
        &format!("{layer_prefix}.mlp.switch_mlp.down_proj.weight"),
        expert_count,
    )?;
    let experts = split_experts(gate_weights, up_weights, down_weights, expert_count)?;
    let shared_expert = optional_shared_expert(tensors, layer_prefix)?;
    let shared_expert_gate = if shared_expert.is_some() {
        Some(linear_from(
            tensors,
            &format!("{layer_prefix}.mlp.shared_expert_gate"),
        )?)
    } else {
        None
    };
    let moe = crate::MoeMlp::new(
        router,
        experts,
        shared_expert,
        shared_expert_gate,
        config.num_experts_per_tok,
    )?;
    Ok((
        Some(post_attention_norm),
        Some(FeedForward::Moe(Box::new(moe))),
    ))
}

pub(in crate::decoder) fn optional_gemma4_parallel_moe_from(
    config: &CausalDecoderConfig,
    tensors: &mut HashMap<String, DecoderTensor>,
    layer_prefix: &str,
) -> Result<Option<FeedForward>> {
    let keys = [
        format!("{layer_prefix}.router.proj.weight"),
        format!("{layer_prefix}.experts.switch_glu.gate_proj.weight"),
        format!("{layer_prefix}.experts.switch_glu.up_proj.weight"),
        format!("{layer_prefix}.experts.switch_glu.down_proj.weight"),
    ];
    if !keys.iter().any(|key| tensors.contains_key(key)) {
        return Ok(None);
    }

    let expert_count = config.num_experts.ok_or_else(|| {
        InferError::Config("poids MoE Gemma4 présents sans num_experts".to_string())
    })?;
    let router = linear_from(tensors, &format!("{layer_prefix}.router.proj"))?;
    let gate_weights = take_expert_linear_weights(
        tensors,
        &format!("{layer_prefix}.experts.switch_glu.gate_proj.weight"),
        expert_count,
    )?;
    let up_weights = take_expert_linear_weights(
        tensors,
        &format!("{layer_prefix}.experts.switch_glu.up_proj.weight"),
        expert_count,
    )?;
    let down_weights = take_expert_linear_weights(
        tensors,
        &format!("{layer_prefix}.experts.switch_glu.down_proj.weight"),
        expert_count,
    )?;
    let experts = split_experts_with_activation(
        gate_weights,
        up_weights,
        down_weights,
        expert_count,
        config.activation,
    )?;
    let router_scale = take_dense(tensors, &format!("{layer_prefix}.router.scale"))?;
    let router_scale = router_scale.map(|value| value * (router_scale.len() as f32).powf(-0.5));
    let per_expert_scale = take_dense(tensors, &format!("{layer_prefix}.router.per_expert_scale"))?;
    let moe = crate::MoeMlp::new(router, experts, None, None, config.num_experts_per_tok)?
        .with_router_norm(router_scale, config.rms_eps)
        .with_per_expert_scale(per_expert_scale);
    Ok(Some(FeedForward::Moe(Box::new(moe))))
}

fn optional_shared_expert(
    tensors: &mut HashMap<String, DecoderTensor>,
    layer_prefix: &str,
) -> Result<Option<GatedMlp>> {
    let prefix = format!("{layer_prefix}.mlp.shared_expert");
    let keys = [
        format!("{prefix}.gate_proj.weight"),
        format!("{prefix}.up_proj.weight"),
        format!("{prefix}.down_proj.weight"),
    ];
    if !keys.iter().any(|key| tensors.contains_key(key)) {
        return Ok(None);
    }
    let gate_proj = linear_from(tensors, &format!("{prefix}.gate_proj"))?;
    let up_proj = linear_from(tensors, &format!("{prefix}.up_proj"))?;
    let down_proj = linear_from(tensors, &format!("{prefix}.down_proj"))?;
    Ok(Some(GatedMlp::new(gate_proj, up_proj, down_proj)))
}

pub(in crate::decoder) fn split_experts(
    gate_weights: Vec<LinearWeight>,
    up_weights: Vec<LinearWeight>,
    down_weights: Vec<LinearWeight>,
    expert_count: usize,
) -> Result<Vec<GatedMlp>> {
    split_experts_with_activation(
        gate_weights,
        up_weights,
        down_weights,
        expert_count,
        crate::Activation::Silu,
    )
}

pub(in crate::decoder) fn split_experts_with_activation(
    gate_weights: Vec<LinearWeight>,
    up_weights: Vec<LinearWeight>,
    down_weights: Vec<LinearWeight>,
    expert_count: usize,
    activation: crate::Activation,
) -> Result<Vec<GatedMlp>> {
    if gate_weights.len() != expert_count
        || up_weights.len() != expert_count
        || down_weights.len() != expert_count
    {
        return Err(InferError::Dimension(format!(
            "nombre experts MoE incohérent: gate={}, up={}, down={}, attendu={expert_count}",
            gate_weights.len(),
            up_weights.len(),
            down_weights.len()
        )));
    }

    let mut experts = Vec::with_capacity(expert_count);
    for ((gate_weight, up_weight), down_weight) in
        gate_weights.into_iter().zip(up_weights).zip(down_weights)
    {
        let (gate_rows, gate_cols) = linear_weight_shape(&gate_weight)?;
        let (up_rows, up_cols) = linear_weight_shape(&up_weight)?;
        let (down_rows, down_cols) = linear_weight_shape(&down_weight)?;
        if gate_rows != up_rows || gate_cols != up_cols || down_cols != gate_rows {
            return Err(InferError::Dimension(format!(
                "formes expert MoE incompatibles: gate=[{gate_rows},{gate_cols}], up=[{up_rows},{up_cols}], down=[{down_rows},{down_cols}]"
            )));
        }
        let gate = Linear::from_weight(gate_weight, None)?;
        let up = Linear::from_weight(up_weight, None)?;
        let down = Linear::from_weight(down_weight, None)?;
        experts.push(GatedMlp::new(gate, up, down).with_activation(activation));
    }
    Ok(experts)
}

fn linear_weight_shape(weight: &LinearWeight) -> Result<(usize, usize)> {
    match weight.shape() {
        [rows, cols] => Ok((*rows, *cols)),
        shape => Err(InferError::Dimension(format!(
            "poids expert Linear attendu rang 2, reçu {shape:?}"
        ))),
    }
}

pub(in crate::decoder) fn split_dense_expert_weights(
    tensor: Tensor,
    expert_count: usize,
) -> Result<Vec<LinearWeight>> {
    let (rows, cols) = match tensor.shape() {
        [experts, rows, cols] if *experts == expert_count => (*rows, *cols),
        shape => {
            return Err(InferError::Dimension(format!(
                "poids expert attendu [{expert_count}, rows, cols], reçu {shape:?}"
            )))
        }
    };
    let mut out = Vec::with_capacity(expert_count);
    for expert in 0..expert_count {
        out.push(LinearWeight::Dense(expert_matrix(
            &tensor, expert, rows, cols,
        )?));
    }
    Ok(out)
}

fn expert_matrix(tensor: &Tensor, expert: usize, rows: usize, cols: usize) -> Result<Tensor> {
    let stride = rows
        .checked_mul(cols)
        .ok_or_else(|| InferError::Shape("poids expert trop grand".to_string()))?;
    let start = expert
        .checked_mul(stride)
        .ok_or_else(|| InferError::Shape("index expert trop grand".to_string()))?;
    let end = start
        .checked_add(stride)
        .ok_or_else(|| InferError::Shape("slice expert trop grand".to_string()))?;
    let data = tensor.data().get(start..end).ok_or_else(|| {
        InferError::Shape(format!(
            "slice expert {expert} hors bornes pour shape {:?}",
            tensor.shape()
        ))
    })?;
    Tensor::from_vec(vec![rows, cols], data.to_vec())
}

pub(in crate::decoder) fn take_dense(
    tensors: &mut HashMap<String, DecoderTensor>,
    key: &str,
) -> Result<Tensor> {
    match tensors
        .remove(key)
        .ok_or_else(|| InferError::MissingWeight(key.to_string()))?
    {
        DecoderTensor::Dense(tensor) => Ok(tensor),
        DecoderTensor::LinearWeight(_) => Err(InferError::Config(format!(
            "poids compact inattendu pour tenseur dense {key}"
        ))),
        DecoderTensor::ExpertLinearWeights { .. } => Err(InferError::Config(format!(
            "poids experts inattendus pour tenseur dense {key}"
        ))),
    }
}

pub(in crate::decoder) fn take_embedding_weight(
    tensors: &mut HashMap<String, DecoderTensor>,
    key: &str,
) -> Result<EmbeddingWeight> {
    match tensors
        .remove(key)
        .ok_or_else(|| InferError::MissingWeight(key.to_string()))?
    {
        DecoderTensor::Dense(tensor) => Ok(EmbeddingWeight::Dense(tensor)),
        DecoderTensor::LinearWeight(LinearWeight::Dense(tensor)) => {
            Ok(EmbeddingWeight::Dense(tensor))
        }
        DecoderTensor::LinearWeight(LinearWeight::AffineQuantized(tensor)) => {
            Ok(EmbeddingWeight::AffineQuantized(tensor))
        }
        DecoderTensor::ExpertLinearWeights { .. } => Err(InferError::Config(format!(
            "poids experts inattendus pour embedding {key}"
        ))),
    }
}

pub(in crate::decoder) fn take_optional_dense(
    tensors: &mut HashMap<String, DecoderTensor>,
    key: &str,
) -> Result<Option<Tensor>> {
    match tensors.remove(key) {
        Some(DecoderTensor::Dense(tensor)) => Ok(Some(tensor)),
        Some(DecoderTensor::LinearWeight(_)) => Err(InferError::Config(format!(
            "poids compact inattendu pour tenseur dense optionnel {key}"
        ))),
        Some(DecoderTensor::ExpertLinearWeights { .. }) => Err(InferError::Config(format!(
            "poids experts inattendus pour tenseur dense optionnel {key}"
        ))),
        None => Ok(None),
    }
}

fn take_linear_weight(
    tensors: &mut HashMap<String, DecoderTensor>,
    key: &str,
) -> Result<LinearWeight> {
    match tensors
        .remove(key)
        .ok_or_else(|| InferError::MissingWeight(key.to_string()))?
    {
        DecoderTensor::Dense(tensor) => Ok(LinearWeight::Dense(tensor)),
        DecoderTensor::LinearWeight(weight) => Ok(weight),
        DecoderTensor::ExpertLinearWeights { .. } => Err(InferError::Config(format!(
            "poids experts inattendus pour Linear 2D {key}"
        ))),
    }
}

fn take_expert_linear_weights(
    tensors: &mut HashMap<String, DecoderTensor>,
    key: &str,
    expert_count: usize,
) -> Result<Vec<LinearWeight>> {
    match tensors
        .remove(key)
        .ok_or_else(|| InferError::MissingWeight(key.to_string()))?
    {
        DecoderTensor::ExpertLinearWeights { weights, .. } => {
            if weights.len() != expert_count {
                return Err(InferError::Dimension(format!(
                    "poids experts {key} count={}, attendu={expert_count}",
                    weights.len()
                )));
            }
            Ok(weights)
        }
        DecoderTensor::Dense(tensor) => split_dense_expert_weights(tensor, expert_count),
        DecoderTensor::LinearWeight(_) => Err(InferError::Config(format!(
            "poids Linear 2D inattendu pour experts MoE {key}"
        ))),
    }
}
