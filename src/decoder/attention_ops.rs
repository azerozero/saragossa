//! Primitives de calcul attention et RoPE.

use super::*;

#[derive(Clone, Copy, Debug)]
pub(super) struct AttentionLayout {
    pub(super) num_attention_heads: usize,
    pub(super) num_key_value_heads: usize,
    pub(super) head_dim: usize,
    pub(super) rope_dims: usize,
}

pub(super) fn full_attention_from_tensors(
    tensors: &mut HashMap<String, DecoderTensor>,
    prefix: &str,
) -> Result<AttentionBlock> {
    let q_proj = linear_from(tensors, &format!("{prefix}.self_attn.q_proj"))?;
    let k_proj = linear_from(tensors, &format!("{prefix}.self_attn.k_proj"))?;
    let v_proj = linear_from(tensors, &format!("{prefix}.self_attn.v_proj"))?;
    let o_proj = linear_from(tensors, &format!("{prefix}.self_attn.o_proj"))?;
    let q_norm = take_optional_dense(tensors, &format!("{prefix}.self_attn.q_norm.weight"))?;
    let k_norm = take_optional_dense(tensors, &format!("{prefix}.self_attn.k_norm.weight"))?;
    if q_norm.is_some() != k_norm.is_some() {
        return Err(InferError::Config(format!(
            "q_norm/k_norm partiels pour {prefix}.self_attn"
        )));
    }
    Ok(AttentionBlock::Full(Box::new(FullAttention {
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        q_norm,
        k_norm,
    })))
}

pub(super) fn linear_attention_from_tensors(
    tensors: &mut HashMap<String, DecoderTensor>,
    prefix: &str,
) -> Result<AttentionBlock> {
    let prefix = format!("{prefix}.linear_attn");
    Ok(AttentionBlock::Linear(Box::new(LinearAttention::new(
        LinearAttentionWeights {
            in_proj_qkv: linear_from(tensors, &format!("{prefix}.in_proj_qkv"))?,
            in_proj_z: linear_from(tensors, &format!("{prefix}.in_proj_z"))?,
            in_proj_b: linear_from(tensors, &format!("{prefix}.in_proj_b"))?,
            in_proj_a: linear_from(tensors, &format!("{prefix}.in_proj_a"))?,
            out_proj: linear_from(tensors, &format!("{prefix}.out_proj"))?,
            conv1d_weight: take_dense(tensors, &format!("{prefix}.conv1d.weight"))?,
            a_log: take_dense(tensors, &format!("{prefix}.A_log"))?,
            dt_bias: take_dense(tensors, &format!("{prefix}.dt_bias"))?,
            norm_weight: take_dense(tensors, &format!("{prefix}.norm.weight"))?,
        },
    ))))
}

pub(super) fn full_attention_forward(
    config: &CausalDecoderConfig,
    normed: &Tensor,
    attention: &FullAttention,
    runtime: ForwardRuntime<'_>,
) -> Result<Tensor> {
    let (q_projection, k_projection, v) = project_qkv(normed, attention, runtime)?;
    let (mut q, gate) = split_attention_gate(config, &q_projection)?;
    let mut k = k_projection;
    let layout = attention_layout(config, &q, &k, &v)?;
    if let (Some(theta), Some(q_norm), Some(k_norm)) =
        (config.rope_theta, &attention.q_norm, &attention.k_norm)
    {
        q = rms_norm_rope_heads_at(
            &q,
            layout.num_attention_heads,
            layout.head_dim,
            layout.rope_dims,
            q_norm,
            config.rms_eps,
            theta,
            0,
        )?;
        k = rms_norm_rope_heads_at(
            &k,
            layout.num_key_value_heads,
            layout.head_dim,
            layout.rope_dims,
            k_norm,
            config.rms_eps,
            theta,
            0,
        )?;
    } else if let (Some(q_norm), Some(k_norm)) = (&attention.q_norm, &attention.k_norm) {
        q = rms_norm_heads(
            &q,
            layout.num_attention_heads,
            layout.head_dim,
            q_norm,
            config.rms_eps,
        )?;
        k = rms_norm_heads(
            &k,
            layout.num_key_value_heads,
            layout.head_dim,
            k_norm,
            config.rms_eps,
        )?;
    }
    if let Some(theta) = config
        .rope_theta
        .filter(|_| attention.q_norm.is_none() || attention.k_norm.is_none())
    {
        q = apply_rope_heads(
            &q,
            layout.num_attention_heads,
            layout.head_dim,
            layout.rope_dims,
            theta,
        )?;
        k = apply_rope_heads(
            &k,
            layout.num_key_value_heads,
            layout.head_dim,
            layout.rope_dims,
            theta,
        )?;
    }
    let mut context = causal_attention(&q, &k, &v, &layout)?;
    if let Some(gate) = gate {
        context = context.mul_elementwise(&gate.map(sigmoid_scalar))?;
    }
    attention.o_proj.forward_with_runtime(&context, runtime)
}

pub(super) fn full_attention_context_cached(
    config: &CausalDecoderConfig,
    normed: &Tensor,
    cache: &mut LayerKvCache,
    position: usize,
    attention: &FullAttention,
    runtime: ForwardRuntime<'_>,
) -> Result<Tensor> {
    let (q_projection, k_projection, v) = project_qkv(normed, attention, runtime)?;
    let (mut q, gate) = split_attention_gate(config, &q_projection)?;
    let mut k = k_projection;
    let layout = attention_layout(config, &q, &k, &v)?;
    if let (Some(theta), Some(q_norm), Some(k_norm)) =
        (config.rope_theta, &attention.q_norm, &attention.k_norm)
    {
        q = rms_norm_rope_heads_at(
            &q,
            layout.num_attention_heads,
            layout.head_dim,
            layout.rope_dims,
            q_norm,
            config.rms_eps,
            theta,
            position,
        )?;
        k = rms_norm_rope_heads_at(
            &k,
            layout.num_key_value_heads,
            layout.head_dim,
            layout.rope_dims,
            k_norm,
            config.rms_eps,
            theta,
            position,
        )?;
    } else if let (Some(q_norm), Some(k_norm)) = (&attention.q_norm, &attention.k_norm) {
        q = rms_norm_heads(
            &q,
            layout.num_attention_heads,
            layout.head_dim,
            q_norm,
            config.rms_eps,
        )?;
        k = rms_norm_heads(
            &k,
            layout.num_key_value_heads,
            layout.head_dim,
            k_norm,
            config.rms_eps,
        )?;
    }
    if let Some(theta) = config
        .rope_theta
        .filter(|_| attention.q_norm.is_none() || attention.k_norm.is_none())
    {
        q = apply_rope_heads_at(
            &q,
            layout.num_attention_heads,
            layout.head_dim,
            layout.rope_dims,
            theta,
            position,
        )?;
        k = apply_rope_heads_at(
            &k,
            layout.num_key_value_heads,
            layout.head_dim,
            layout.rope_dims,
            theta,
            position,
        )?;
    }
    // Chemin résident GPU (flag, KV résident présent) OU chemin CPU (oracle).
    let mut context = match full_attention_resident_context(cache, &q, &k, &v)? {
        Some(context) => context,
        None => {
            cache.append(&k, &v, &layout)?;
            cached_attention_one(&q, cache, &layout)?
        }
    };
    // Gate de sortie full-attn : appliqué APRÈS le contexte (hors kernel), à
    // l'identique pour les deux chemins (réserve : le kernel produit le brut).
    if let Some(gate) = gate {
        context = context.mul_elementwise(&gate.map(sigmoid_scalar))?;
    }
    Ok(context)
}

/// Chemin résident GPU de l'attention decode full-attn (flag
/// `RETI_RUST_DECODE_RESIDENT`). Renvoie `Some(contexte brut [1, q_dim])` si la
/// couche a un KV résident (`LayerKvCache::full`), sinon `None` → chemin CPU
/// `cached_attention_one`. Append du K/V (rope'd) du token courant par écriture
/// résidente (réserve R3), puis attention single-query sur le KV résident.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub(super) fn full_attention_resident_context(
    cache: &mut LayerKvCache,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
) -> Result<Option<Tensor>> {
    let Some(full) = cache.full.as_mut() else {
        return Ok(None);
    };
    full.append_row(k.data(), v.data())?;
    let context = full.attention_decode(q.data())?;
    Ok(Some(Tensor::from_vec(vec![1, context.len()], context)?))
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
pub(super) fn full_attention_resident_context(
    _cache: &mut LayerKvCache,
    _q: &Tensor,
    _k: &Tensor,
    _v: &Tensor,
) -> Result<Option<Tensor>> {
    Ok(None)
}

pub(super) fn full_attention_context_prefill(
    config: &CausalDecoderConfig,
    normed: &Tensor,
    cache: &mut LayerKvCache,
    position_offset: usize,
    attention: &FullAttention,
    runtime: ForwardRuntime<'_>,
) -> Result<Tensor> {
    let (q_projection, k_projection, v) = project_qkv(normed, attention, runtime)?;
    let (mut q, gate) = split_attention_gate(config, &q_projection)?;
    let mut k = k_projection;
    let layout = attention_layout(config, &q, &k, &v)?;
    if let (Some(theta), Some(q_norm), Some(k_norm)) =
        (config.rope_theta, &attention.q_norm, &attention.k_norm)
    {
        q = rms_norm_rope_heads_at(
            &q,
            layout.num_attention_heads,
            layout.head_dim,
            layout.rope_dims,
            q_norm,
            config.rms_eps,
            theta,
            position_offset,
        )?;
        k = rms_norm_rope_heads_at(
            &k,
            layout.num_key_value_heads,
            layout.head_dim,
            layout.rope_dims,
            k_norm,
            config.rms_eps,
            theta,
            position_offset,
        )?;
    } else if let (Some(q_norm), Some(k_norm)) = (&attention.q_norm, &attention.k_norm) {
        q = rms_norm_heads(
            &q,
            layout.num_attention_heads,
            layout.head_dim,
            q_norm,
            config.rms_eps,
        )?;
        k = rms_norm_heads(
            &k,
            layout.num_key_value_heads,
            layout.head_dim,
            k_norm,
            config.rms_eps,
        )?;
    }
    if let Some(theta) = config
        .rope_theta
        .filter(|_| attention.q_norm.is_none() || attention.k_norm.is_none())
    {
        q = apply_rope_heads_at(
            &q,
            layout.num_attention_heads,
            layout.head_dim,
            layout.rope_dims,
            theta,
            position_offset,
        )?;
        k = apply_rope_heads_at(
            &k,
            layout.num_key_value_heads,
            layout.head_dim,
            layout.rope_dims,
            theta,
            position_offset,
        )?;
    }
    let mut context = if cache.len() == 0 {
        cache.append_batch(&k, &v, &layout)?;
        causal_attention(&q, &k, &v, &layout)?
    } else {
        cached_attention_prefill_rows(&q, &k, &v, cache, &layout)?
    };
    if let Some(gate) = gate {
        context = context.mul_elementwise(&gate.map(sigmoid_scalar))?;
    }
    Ok(context)
}

pub(super) fn project_qkv(
    normed: &Tensor,
    attention: &FullAttention,
    runtime: ForwardRuntime<'_>,
) -> Result<(Tensor, Tensor, Tensor)> {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    if let Some(metal) = runtime.metal_executor() {
        if attention.q_proj.bias().is_none()
            && attention.k_proj.bias().is_none()
            && attention.v_proj.bias().is_none()
        {
            return metal.project_three_biasless(
                normed,
                &attention.q_proj,
                &attention.k_proj,
                &attention.v_proj,
            );
        }
    }
    Ok((
        attention.q_proj.forward_with_runtime(normed, runtime)?,
        attention.k_proj.forward_with_runtime(normed, runtime)?,
        attention.v_proj.forward_with_runtime(normed, runtime)?,
    ))
}

pub(super) fn split_attention_gate(
    config: &CausalDecoderConfig,
    q_projection: &Tensor,
) -> Result<(Tensor, Option<Tensor>)> {
    if !config.attn_output_gate {
        return Ok((q_projection.clone(), None));
    }
    let (rows, cols) = q_projection.as_matrix()?;
    let q_dim = config.num_attention_heads
        * config
            .head_dim
            .ok_or_else(|| InferError::Dimension("head_dim manquant".to_string()))?;
    if cols != 2 * q_dim {
        return Err(InferError::Dimension(format!(
            "q_proj gated attendu [seq,{}], reçu {:?}",
            2 * q_dim,
            q_projection.shape()
        )));
    }
    let mut q = Vec::with_capacity(rows * q_dim);
    let mut gate = Vec::with_capacity(rows * q_dim);
    for row in 0..rows {
        let source = q_projection.row_slice(row)?;
        for head in 0..config.num_attention_heads {
            let start = head * 2 * q_dim / config.num_attention_heads;
            let head_dim = q_dim / config.num_attention_heads;
            q.extend_from_slice(&source[start..start + head_dim]);
            gate.extend_from_slice(&source[start + head_dim..start + 2 * head_dim]);
        }
    }
    Ok((
        Tensor::from_vec(vec![rows, q_dim], q)?,
        Some(Tensor::from_vec(vec![rows, q_dim], gate)?),
    ))
}

pub(super) fn sigmoid_scalar(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

pub(super) fn attention_layout(
    config: &CausalDecoderConfig,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
) -> Result<AttentionLayout> {
    let (_, q_dim) = q.as_matrix()?;
    let (_, k_dim) = k.as_matrix()?;
    let (_, v_dim) = v.as_matrix()?;
    let num_attention_heads = config.num_attention_heads;
    let num_key_value_heads = config.num_key_value_heads;
    if num_attention_heads == 0 || num_key_value_heads == 0 {
        return Err(InferError::Dimension(format!(
            "attention heads invalides: q_heads={num_attention_heads}, kv_heads={num_key_value_heads}"
        )));
    }
    if num_attention_heads % num_key_value_heads != 0 {
        return Err(InferError::Dimension(format!(
            "q_heads {num_attention_heads} non divisible par kv_heads {num_key_value_heads}"
        )));
    }
    let head_dim = match config.head_dim {
        Some(dim) if dim > 0 => dim,
        Some(_) => return Err(InferError::Dimension("head_dim explicite nul".to_string())),
        None => q_dim.checked_div(num_attention_heads).ok_or_else(|| {
            InferError::Dimension(format!(
                "q_dim {q_dim} incompatible avec q_heads {num_attention_heads}"
            ))
        })?,
    };
    if q_dim != num_attention_heads * head_dim {
        return Err(InferError::Dimension(format!(
            "q_dim {q_dim} attendu {} pour q_heads={num_attention_heads}, head_dim={head_dim}",
            num_attention_heads * head_dim
        )));
    }
    let expected_kv_dim = num_key_value_heads * head_dim;
    if k_dim != expected_kv_dim || v_dim != expected_kv_dim {
        return Err(InferError::Dimension(format!(
            "kv dims attendues {expected_kv_dim}, reçu k={k_dim}, v={v_dim}"
        )));
    }
    let rope_dims = config.rope_dims.unwrap_or(head_dim);
    if rope_dims == 0 || rope_dims > head_dim || rope_dims % 2 != 0 {
        return Err(InferError::Dimension(format!(
            "rope_dims {rope_dims} invalide pour head_dim {head_dim}"
        )));
    }
    Ok(AttentionLayout {
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        rope_dims,
    })
}

pub(super) fn apply_rope_heads(
    x: &Tensor,
    heads: usize,
    head_dim: usize,
    rope_dims: usize,
    base_theta: f32,
) -> Result<Tensor> {
    apply_rope_heads_at(x, heads, head_dim, rope_dims, base_theta, 0)
}

pub(super) fn rms_norm_heads(
    x: &Tensor,
    heads: usize,
    head_dim: usize,
    weight: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    let (seq, dim) = x.as_matrix()?;
    if heads == 0 || head_dim == 0 || dim != heads * head_dim {
        return Err(InferError::Dimension(format!(
            "RMSNorm heads invalide: x={:?}, heads={heads}, head_dim={head_dim}",
            x.shape()
        )));
    }
    let weight_data = match weight.shape() {
        [n] if *n == head_dim => weight.data(),
        [1, n] if *n == head_dim => weight.data(),
        _ => {
            return Err(InferError::Dimension(format!(
                "RMSNorm head weight attendu [{head_dim}] ou [1,{head_dim}], reçu {:?}",
                weight.shape()
            )))
        }
    };
    let mut out = x.data().to_vec();
    for pos in 0..seq {
        let row_start = pos * dim;
        for head in 0..heads {
            let head_start = row_start + head * head_dim;
            let xs = &x.data()[head_start..head_start + head_dim];
            let mean_square = xs.iter().map(|value| value * value).sum::<f32>() / head_dim as f32;
            let inv_rms = 1.0 / (mean_square + eps).sqrt();
            for col in 0..head_dim {
                out[head_start + col] = xs[col] * inv_rms * weight_data[col];
            }
        }
    }
    Tensor::from_vec(vec![seq, dim], out)
}

struct RmsHeadSpec<'a> {
    heads: usize,
    head_dim: usize,
    rope_dims: usize,
    weight: &'a Tensor,
    eps: f32,
    base_theta: f32,
    position_offset: usize,
}

#[expect(
    clippy::too_many_arguments,
    reason = "helper CPU miroir des paramètres RoPE/RMS utilisés par les kernels"
)]
pub(super) fn rms_norm_rope_heads_at(
    x: &Tensor,
    heads: usize,
    head_dim: usize,
    rope_dims: usize,
    weight: &Tensor,
    eps: f32,
    base_theta: f32,
    position_offset: usize,
) -> Result<Tensor> {
    rms_norm_rope_heads_with(
        x,
        RmsHeadSpec {
            heads,
            head_dim,
            rope_dims,
            weight,
            eps,
            base_theta,
            position_offset,
        },
    )
}

fn rms_norm_rope_heads_with(x: &Tensor, spec: RmsHeadSpec<'_>) -> Result<Tensor> {
    let (seq, dim) = x.as_matrix()?;
    if spec.heads == 0 || spec.head_dim == 0 || dim != spec.heads * spec.head_dim {
        return Err(InferError::Dimension(format!(
            "RMSNorm/RoPE heads invalide: x={:?}, heads={}, head_dim={}",
            x.shape(),
            spec.heads,
            spec.head_dim
        )));
    }
    if spec.rope_dims == 0 || spec.rope_dims > spec.head_dim || spec.rope_dims % 2 != 0 {
        return Err(InferError::Dimension(format!(
            "RMSNorm/RoPE rope_dims {} invalide pour head_dim {}",
            spec.rope_dims, spec.head_dim
        )));
    }
    let weight_data = match spec.weight.shape() {
        [n] if *n == spec.head_dim => spec.weight.data(),
        [1, n] if *n == spec.head_dim => spec.weight.data(),
        _ => {
            return Err(InferError::Dimension(format!(
                "RMSNorm/RoPE weight attendu [{}] ou [1,{}], reçu {:?}",
                spec.head_dim,
                spec.head_dim,
                spec.weight.shape()
            )))
        }
    };
    let pairs = spec.rope_dims / 2;
    let rotations = (0..seq)
        .map(|pos| rope_rotations(spec.position_offset + pos, spec.rope_dims, spec.base_theta))
        .collect::<Result<Vec<_>>>()?;
    let mut out = vec![0.0_f32; x.len()];
    for (pos, rotation) in rotations.iter().enumerate().take(seq) {
        let row_start = pos * dim;
        for head in 0..spec.heads {
            let head_start = row_start + head * spec.head_dim;
            let xs = &x.data()[head_start..head_start + spec.head_dim];
            let mean_square =
                xs.iter().map(|value| value * value).sum::<f32>() / spec.head_dim as f32;
            let inv_rms = 1.0 / (mean_square + spec.eps).sqrt();
            for pair in 0..pairs {
                let even = xs[2 * pair] * inv_rms * weight_data[2 * pair];
                let odd = xs[2 * pair + 1] * inv_rms * weight_data[2 * pair + 1];
                let (cos, sin) = rotation[pair];
                out[head_start + 2 * pair] = even * cos - odd * sin;
                out[head_start + 2 * pair + 1] = even * sin + odd * cos;
            }
            for col in spec.rope_dims..spec.head_dim {
                out[head_start + col] = xs[col] * inv_rms * weight_data[col];
            }
        }
    }
    Tensor::from_vec(vec![seq, dim], out)
}

pub(super) fn apply_rope_heads_at(
    x: &Tensor,
    heads: usize,
    head_dim: usize,
    rope_dims: usize,
    base_theta: f32,
    position_offset: usize,
) -> Result<Tensor> {
    let (seq, dim) = x.as_matrix()?;
    if dim != heads * head_dim {
        return Err(InferError::Dimension(format!(
            "RoPE heads={heads}, head_dim={head_dim}, dim reçu {dim}"
        )));
    }
    if base_theta <= 0.0 {
        return Err(InferError::Dimension(format!(
            "RoPE base_theta invalide: {base_theta}"
        )));
    }
    let pairs = rope_dims / 2;
    let rotations = (0..seq)
        .map(|pos| rope_rotations(position_offset + pos, rope_dims, base_theta))
        .collect::<Result<Vec<_>>>()?;
    let mut out = x.data().to_vec();
    for (pos, rotation) in rotations.iter().enumerate().take(seq) {
        let row_start = pos * dim;
        for head in 0..heads {
            let head_start = row_start + head * head_dim;
            for (pair, &(cos, sin)) in rotation.iter().enumerate().take(pairs) {
                let even_index = head_start + 2 * pair;
                let odd_index = even_index + 1;
                let even = x.data()[even_index];
                let odd = x.data()[odd_index];
                out[even_index] = even * cos - odd * sin;
                out[odd_index] = even * sin + odd * cos;
            }
        }
    }
    Tensor::from_vec(vec![seq, dim], out)
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct RopeRotationKey {
    position: usize,
    rope_dims: usize,
    base_theta_bits: u32,
}

type RopeRotations = Arc<Vec<(f32, f32)>>;
type RopeRotationCache = Mutex<HashMap<RopeRotationKey, RopeRotations>>;

pub(super) fn rope_rotations(
    position: usize,
    rope_dims: usize,
    base_theta: f32,
) -> Result<RopeRotations> {
    static CACHE: OnceLock<RopeRotationCache> = OnceLock::new();
    let key = RopeRotationKey {
        position,
        rope_dims,
        base_theta_bits: base_theta.to_bits(),
    };
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    {
        let rotations = cache
            .lock()
            .map_err(|_| InferError::Config("cache RoPE empoisonné".to_string()))?;
        if let Some(rotations) = rotations.get(&key) {
            return Ok(rotations.clone());
        }
    }
    let pairs = rope_dims / 2;
    let mut rotations = Vec::with_capacity(pairs);
    for pair in 0..pairs {
        let exponent = (2 * pair) as f32 / rope_dims as f32;
        let angle = position as f32 / base_theta.powf(exponent);
        rotations.push((angle.cos(), angle.sin()));
    }
    let rotations = Arc::new(rotations);
    let mut cache = cache
        .lock()
        .map_err(|_| InferError::Config("cache RoPE empoisonné".to_string()))?;
    cache.insert(key, rotations.clone());
    Ok(rotations)
}

pub(super) fn cached_attention_one(
    q: &Tensor,
    cache: &mut LayerKvCache,
    layout: &AttentionLayout,
) -> Result<Tensor> {
    let (seq, dim) = q.as_matrix()?;
    let expected_q_dim = layout.num_attention_heads * layout.head_dim;
    let expected_kv_dim = layout.num_key_value_heads * layout.head_dim;
    if seq != 1 || dim != expected_q_dim || cache.kv_dim != Some(expected_kv_dim) {
        return Err(InferError::Dimension(format!(
            "attention cache shapes q={:?}, cache_dim={:?}, attendu q=[1,{expected_q_dim}] kv_dim={expected_kv_dim}",
            q.shape(),
            cache.kv_dim
        )));
    }
    let cache_len = cache.len();
    if cache_len == 0 || cache.keys.len() != cache.values.len() {
        return Err(InferError::Dimension(format!(
            "cache KV incohérent: len={}, keys={}, values={}",
            cache_len,
            cache.keys.len(),
            cache.values.len()
        )));
    }

    let inv_scale = 1.0 / (layout.head_dim as f32).sqrt();
    let kv_group = layout.num_attention_heads / layout.num_key_value_heads;
    let q_row = q.row_slice(0)?;
    let mut out = vec![0.0_f32; expected_q_dim];
    let keys = &cache.keys;
    let values = &cache.values;
    if cache_len >= attention_parallel_threshold() {
        out.par_chunks_mut(layout.head_dim)
            .enumerate()
            .for_each_init(
                || vec![0.0_f32; cache_len],
                |scores, (q_head, out_head)| {
                    let kv_head = q_head / kv_group;
                    let q_start = q_head * layout.head_dim;
                    let k_start = kv_head * layout.head_dim;
                    let q_slice = &q_row[q_start..q_start + layout.head_dim];
                    for (row, score) in scores.iter_mut().enumerate() {
                        let key_start = row * expected_kv_dim + k_start;
                        let k_slice = &keys[key_start..key_start + layout.head_dim];
                        *score = dot_product(q_slice, k_slice) * inv_scale;
                    }
                    softmax_in_place(scores, 1.0);
                    for (row, prob) in scores.iter().copied().enumerate() {
                        let value_start = row * expected_kv_dim + k_start;
                        let v_slice = &values[value_start..value_start + layout.head_dim];
                        for col in 0..layout.head_dim {
                            out_head[col] += prob * v_slice[col];
                        }
                    }
                },
            );
        return Tensor::from_vec(vec![1, expected_q_dim], out);
    }
    let mut scores = vec![0.0_f32; cache_len];
    for q_head in 0..layout.num_attention_heads {
        let kv_head = q_head / kv_group;
        let q_start = q_head * layout.head_dim;
        let k_start = kv_head * layout.head_dim;
        let q_slice = &q_row[q_start..q_start + layout.head_dim];
        for (row, score) in scores.iter_mut().enumerate() {
            let key_start = row * expected_kv_dim + k_start;
            let k_slice = &keys[key_start..key_start + layout.head_dim];
            *score = dot_product(q_slice, k_slice) * inv_scale;
        }
        softmax_in_place(&mut scores, 1.0);
        for (row, prob) in scores.iter().copied().enumerate() {
            let value_start = row * expected_kv_dim + k_start;
            let v_slice = &values[value_start..value_start + layout.head_dim];
            for col in 0..layout.head_dim {
                out[q_start + col] += prob * v_slice[col];
            }
        }
    }
    Tensor::from_vec(vec![1, expected_q_dim], out)
}

fn cached_attention_prefill_rows(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    cache: &mut LayerKvCache,
    layout: &AttentionLayout,
) -> Result<Tensor> {
    let (seq, dim) = q.as_matrix()?;
    let (k_seq, k_dim) = k.as_matrix()?;
    let (v_seq, v_dim) = v.as_matrix()?;
    let expected_q_dim = layout.num_attention_heads * layout.head_dim;
    let expected_kv_dim = layout.num_key_value_heads * layout.head_dim;
    if seq == 0
        || k_seq != seq
        || v_seq != seq
        || dim != expected_q_dim
        || k_dim != expected_kv_dim
        || v_dim != expected_kv_dim
    {
        return Err(InferError::Dimension(format!(
            "attention batch cached shapes q={:?}, k={:?}, v={:?}, attendu q=[seq,{expected_q_dim}] kv=[seq,{expected_kv_dim}]",
            q.shape(),
            k.shape(),
            v.shape()
        )));
    }
    let mut data = Vec::with_capacity(seq * expected_q_dim);
    for pos in 0..seq {
        let q_row = Tensor::row(q.row_slice(pos)?.to_vec())?;
        let k_row = Tensor::row(k.row_slice(pos)?.to_vec())?;
        let v_row = Tensor::row(v.row_slice(pos)?.to_vec())?;
        let context = match full_attention_resident_context(cache, &q_row, &k_row, &v_row)? {
            Some(context) => context,
            None => {
                cache.append(&k_row, &v_row, layout)?;
                cached_attention_one(&q_row, cache, layout)?
            }
        };
        data.extend_from_slice(context.as_row()?);
    }
    Tensor::from_vec(vec![seq, expected_q_dim], data)
}

pub(super) fn softmax_in_place(values: &mut [f32], temperature: f32) {
    let temperature = temperature.max(0.000_1);
    let max = values
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |left, right| left.max(right));
    let mut sum = 0.0_f32;
    for value in values.iter_mut() {
        *value = ((*value - max) / temperature).exp();
        sum += *value;
    }
    if sum <= f32::EPSILON {
        let uniform = 1.0 / values.len() as f32;
        values.fill(uniform);
        return;
    }
    for value in values {
        *value /= sum;
    }
}

pub(super) fn dot_product(left: &[f32], right: &[f32]) -> f32 {
    let mut sum = 0.0_f32;
    for idx in 0..left.len() {
        sum += left[idx] * right[idx];
    }
    sum
}

pub(super) fn attention_parallel_threshold() -> usize {
    crate::decoder::flags::attention_parallel_threshold()
}

pub(super) fn causal_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    layout: &AttentionLayout,
) -> Result<Tensor> {
    let (seq, dim) = q.as_matrix()?;
    let (k_seq, k_dim) = k.as_matrix()?;
    let (v_seq, v_dim) = v.as_matrix()?;
    let expected_q_dim = layout.num_attention_heads * layout.head_dim;
    let expected_kv_dim = layout.num_key_value_heads * layout.head_dim;
    if seq != k_seq
        || seq != v_seq
        || dim != expected_q_dim
        || k_dim != expected_kv_dim
        || v_dim != expected_kv_dim
    {
        return Err(InferError::Dimension(format!(
            "attention shapes q={:?}, k={:?}, v={:?}",
            q.shape(),
            k.shape(),
            v.shape()
        )));
    }

    let scale = (layout.head_dim as f32).sqrt();
    let kv_group = layout.num_attention_heads / layout.num_key_value_heads;
    let mut out = vec![0.0_f32; seq * expected_q_dim];
    for pos in 0..seq {
        let q_row = q.row_slice(pos)?;
        for q_head in 0..layout.num_attention_heads {
            let kv_head = q_head / kv_group;
            let q_start = q_head * layout.head_dim;
            let q_slice = &q_row[q_start..q_start + layout.head_dim];
            let mut scores = Vec::with_capacity(pos + 1);
            for row in 0..=pos {
                let key = k.row_slice(row)?;
                let k_start = kv_head * layout.head_dim;
                let k_slice = &key[k_start..k_start + layout.head_dim];
                let dot = q_slice
                    .iter()
                    .zip(k_slice.iter())
                    .map(|(a, b)| a * b)
                    .sum::<f32>();
                scores.push(dot / scale);
            }
            let probs = softmax(&scores, 1.0);
            for (row, prob) in probs.iter().enumerate() {
                let value = v.row_slice(row)?;
                let v_start = kv_head * layout.head_dim;
                let v_slice = &value[v_start..v_start + layout.head_dim];
                let out_start = pos * expected_q_dim + q_start;
                for col in 0..layout.head_dim {
                    out[out_start + col] += prob * v_slice[col];
                }
            }
        }
    }
    Tensor::from_vec(vec![seq, expected_q_dim], out)
}
