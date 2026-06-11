//! Implémentation CPU/Metal de l'attention linéaire récurrente.

#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::decoder::flags::{env_flag, trace_linear_attn_enabled};
use crate::{silu, ForwardRuntime, InferError, Linear, Result, Tensor};
use rayon::prelude::*;
#[cfg(all(target_os = "macos", feature = "metal"))]
use std::sync::OnceLock;

#[derive(Clone, Copy, Debug)]
pub(crate) struct LinearAttentionConfig {
    pub num_key_heads: usize,
    pub num_value_heads: usize,
    pub key_head_dim: usize,
    pub value_head_dim: usize,
    pub conv_kernel_dim: usize,
    pub rms_eps: f32,
}

impl LinearAttentionConfig {
    pub(crate) fn key_dim(self) -> Result<usize> {
        checked_mul(self.num_key_heads, self.key_head_dim, "linear key_dim")
    }

    pub(crate) fn value_dim(self) -> Result<usize> {
        checked_mul(
            self.num_value_heads,
            self.value_head_dim,
            "linear value_dim",
        )
    }

    fn conv_dim(self) -> Result<usize> {
        let key_dim = self.key_dim()?;
        let value_dim = self.value_dim()?;
        key_dim
            .checked_mul(2)
            .and_then(|twice| twice.checked_add(value_dim))
            .ok_or_else(|| InferError::Shape("linear conv_dim trop grand".to_string()))
    }

    fn validate(self) -> Result<()> {
        if self.num_key_heads == 0
            || self.num_value_heads == 0
            || self.key_head_dim == 0
            || self.value_head_dim == 0
            || self.conv_kernel_dim == 0
        {
            return Err(InferError::Dimension(format!(
                "linear-attn dims invalides: key_heads={}, value_heads={}, key_dim={}, value_dim={}, kernel={}",
                self.num_key_heads,
                self.num_value_heads,
                self.key_head_dim,
                self.value_head_dim,
                self.conv_kernel_dim
            )));
        }
        if self.num_value_heads % self.num_key_heads != 0 {
            return Err(InferError::Dimension(format!(
                "linear-attn value_heads {} non divisible par key_heads {}",
                self.num_value_heads, self.num_key_heads
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub(crate) struct LinearAttention {
    in_proj_qkv: Linear,
    in_proj_z: Linear,
    in_proj_b: Linear,
    in_proj_a: Linear,
    out_proj: Linear,
    conv1d_weight: Tensor,
    a_log: Tensor,
    dt_bias: Tensor,
    norm_weight: Tensor,
}

#[derive(Clone, Debug)]
pub(crate) struct LinearAttentionWeights {
    pub in_proj_qkv: Linear,
    pub in_proj_z: Linear,
    pub in_proj_b: Linear,
    pub in_proj_a: Linear,
    pub out_proj: Linear,
    pub conv1d_weight: Tensor,
    pub a_log: Tensor,
    pub dt_bias: Tensor,
    pub norm_weight: Tensor,
}

#[derive(Debug, Default)]
pub(crate) struct LinearAttentionCache {
    conv: Vec<f32>,
    conv_dim: Option<usize>,
    ssm: Vec<f32>,
    ssm_shape: Option<(usize, usize, usize)>,
    #[cfg(all(target_os = "macos", feature = "metal"))]
    metal: Option<crate::metal_backend::LinearAttentionMetalState>,
}

impl Clone for LinearAttentionCache {
    fn clone(&self) -> Self {
        Self {
            conv: self.conv.clone(),
            conv_dim: self.conv_dim,
            ssm: self.ssm.clone(),
            ssm_shape: self.ssm_shape,
            #[cfg(all(target_os = "macos", feature = "metal"))]
            metal: None,
        }
    }
}

impl LinearAttention {
    pub(crate) fn new(weights: LinearAttentionWeights) -> Self {
        Self {
            in_proj_qkv: weights.in_proj_qkv,
            in_proj_z: weights.in_proj_z,
            in_proj_b: weights.in_proj_b,
            in_proj_a: weights.in_proj_a,
            out_proj: weights.out_proj,
            conv1d_weight: weights.conv1d_weight,
            a_log: weights.a_log,
            dt_bias: weights.dt_bias,
            norm_weight: weights.norm_weight,
        }
    }

    #[cfg(test)]
    pub(crate) fn forward(&self, config: LinearAttentionConfig, x: &Tensor) -> Result<Tensor> {
        self.forward_with_runtime(config, x, ForwardRuntime::cpu())
    }

    pub(crate) fn forward_with_runtime(
        &self,
        config: LinearAttentionConfig,
        x: &Tensor,
        runtime: ForwardRuntime<'_>,
    ) -> Result<Tensor> {
        let mut cache = LinearAttentionCache::default();
        self.forward_inner(config, x, Some(&mut cache), runtime)
    }

    #[cfg(test)]
    pub(crate) fn forward_cached(
        &self,
        config: LinearAttentionConfig,
        x: &Tensor,
        cache: &mut LinearAttentionCache,
    ) -> Result<Tensor> {
        let (seq, _) = x.as_matrix()?;
        if seq != 1 {
            return Err(InferError::Dimension(format!(
                "linear-attn cached attend un seul token, reçu {:?}",
                x.shape()
            )));
        }
        self.forward_inner(config, x, Some(cache), ForwardRuntime::cpu())
    }

    pub(crate) fn forward_cached_with_runtime(
        &self,
        config: LinearAttentionConfig,
        x: &Tensor,
        cache: &mut LinearAttentionCache,
        runtime: ForwardRuntime<'_>,
    ) -> Result<Tensor> {
        let (seq, _) = x.as_matrix()?;
        if seq != 1 {
            return Err(InferError::Dimension(format!(
                "linear-attn cached attend un seul token, reçu {:?}",
                x.shape()
            )));
        }
        self.forward_inner(config, x, Some(cache), runtime)
    }

    pub(crate) fn forward_cached_batch_with_runtime(
        &self,
        config: LinearAttentionConfig,
        x: &Tensor,
        cache: &mut LinearAttentionCache,
        runtime: ForwardRuntime<'_>,
    ) -> Result<Tensor> {
        let (seq, _) = x.as_matrix()?;
        if seq == 0 {
            return Err(InferError::Dimension(
                "linear-attn cached batch vide".to_string(),
            ));
        }
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if runtime.metal_executor().is_some() && seq > 1 {
            return self.forward_cached_rows_with_runtime(config, x, cache, runtime);
        }
        self.forward_inner(config, x, Some(cache), runtime)
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn forward_cached_rows_with_runtime(
        &self,
        config: LinearAttentionConfig,
        x: &Tensor,
        cache: &mut LinearAttentionCache,
        runtime: ForwardRuntime<'_>,
    ) -> Result<Tensor> {
        if let Some(metal) = runtime.metal_executor() {
            ensure_conv_cache(config, cache)?;
            ensure_ssm_cache(config, cache)?;
            match metal.linear_attention_cached_batch_resident(
                x,
                &self.in_proj_qkv,
                &self.in_proj_z,
                &self.in_proj_b,
                &self.in_proj_a,
                &self.out_proj,
                &self.conv1d_weight,
                &self.a_log,
                &self.dt_bias,
                &self.norm_weight,
                cache.conv.as_slice(),
                cache.ssm.as_slice(),
                &mut cache.metal,
                crate::metal_backend::LinearAttentionStepSpec {
                    num_key_heads: config.num_key_heads,
                    num_value_heads: config.num_value_heads,
                    key_head_dim: config.key_head_dim,
                    value_head_dim: config.value_head_dim,
                    conv_kernel_dim: config.conv_kernel_dim,
                    rms_eps: config.rms_eps,
                },
            ) {
                Ok(output) => return Ok(output),
                Err(error) => {
                    if trace_linear_attn_enabled() {
                        eprintln!("linear-attn resident batch gpu fallback: {error}");
                    }
                }
            }
        }
        let (seq, _) = x.as_matrix()?;
        let mut data = Vec::new();
        let mut out_cols = None;
        for row in 0..seq {
            let input = Tensor::row(x.row_slice(row)?.to_vec())?;
            let output = self.forward_cached_with_runtime(config, &input, cache, runtime)?;
            let (_, cols) = output.as_matrix()?;
            match out_cols {
                Some(expected) if expected != cols => {
                    return Err(InferError::Dimension(format!(
                        "linear-attn batch out_dim={cols}, attendu {expected}"
                    )))
                }
                Some(_) => {}
                None => {
                    out_cols = Some(cols);
                    data.reserve(seq * cols);
                }
            }
            data.extend_from_slice(output.as_row()?);
        }
        let cols = out_cols
            .ok_or_else(|| InferError::Dimension("linear-attn cached batch vide".to_string()))?;
        Tensor::from_vec(vec![seq, cols], data)
    }

    fn forward_inner(
        &self,
        config: LinearAttentionConfig,
        x: &Tensor,
        cache: Option<&mut LinearAttentionCache>,
        runtime: ForwardRuntime<'_>,
    ) -> Result<Tensor> {
        config.validate()?;
        validate_weights(self, config)?;
        let (seq, _) = x.as_matrix()?;
        let mut local_cache;
        let cache = match cache {
            Some(cache) => cache,
            None => {
                local_cache = LinearAttentionCache::default();
                &mut local_cache
            }
        };

        #[cfg(all(target_os = "macos", feature = "metal"))]
        if let Some(metal) = runtime.metal_executor() {
            if seq == 1 && linear_attn_resident_step_enabled() {
                let had_resident_state = cache.metal.is_some();
                match self.forward_resident_metal_step(config, x, cache, metal) {
                    Ok(output) => return Ok(output),
                    Err(error) if had_resident_state => return Err(error),
                    Err(error) => {
                        cache.metal = None;
                        if trace_linear_attn_enabled() {
                            eprintln!("linear-attn resident gpu fallback: {error}");
                        }
                    }
                }
            }
            if seq == 1 && linear_attn_fused_step_enabled() {
                match self.forward_fused_metal_step(config, x, cache, metal) {
                    Ok(output) => return Ok(output),
                    Err(error) => {
                        if trace_linear_attn_enabled() {
                            eprintln!("linear-attn fused gpu fallback: {error}");
                        }
                    }
                }
            }
        }

        let (qkv, z, beta_input, gate_input) = self.input_projections(x, runtime)?;
        let conv_out = depthwise_causal_conv(&qkv, &self.conv1d_weight, config, cache)?;
        let conv_out = silu(&conv_out);

        let key_dim = config.key_dim()?;
        let value_dim = config.value_dim()?;
        let q = slice_columns(&conv_out, 0, key_dim)?;
        let k = slice_columns(&conv_out, key_dim, key_dim)?;
        let v = slice_columns(&conv_out, key_dim * 2, value_dim)?;
        let q = rms_norm_heads_unit(&q, config.num_key_heads, config.key_head_dim, 1.0e-6)?
            .map(|value| value * key_scale(config.key_head_dim, true));
        let k = rms_norm_heads_unit(&k, config.num_key_heads, config.key_head_dim, 1.0e-6)?
            .map(|value| value * key_scale(config.key_head_dim, false));
        let beta = beta_input.map(sigmoid);
        let g = compute_decay(
            &gate_input,
            &self.a_log,
            &self.dt_bias,
            config.num_value_heads,
        )?;
        let y = gated_delta(&q, &k, &v, &g, &beta, config, cache)?;
        let y = rms_norm_heads(
            &y,
            config.num_value_heads,
            config.value_head_dim,
            &self.norm_weight,
            config.rms_eps,
        )?;
        let gated = y.mul_elementwise(&silu(&z))?;
        self.out_proj.forward_with_runtime(&gated, runtime)
    }

    fn input_projections(
        &self,
        x: &Tensor,
        runtime: ForwardRuntime<'_>,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if let Some(metal) = runtime.metal_executor() {
            match metal.project_four_biasless(
                x,
                &self.in_proj_qkv,
                &self.in_proj_z,
                &self.in_proj_b,
                &self.in_proj_a,
            ) {
                Ok(projections) => return Ok(projections),
                Err(error) => {
                    if trace_linear_attn_enabled() {
                        eprintln!("linear-attn project4 gpu fallback: {error}");
                    }
                }
            }
        }
        Ok((
            self.in_proj_qkv.forward_with_runtime(x, runtime)?,
            self.in_proj_z.forward_with_runtime(x, runtime)?,
            self.in_proj_b.forward_with_runtime(x, runtime)?,
            self.in_proj_a.forward_with_runtime(x, runtime)?,
        ))
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn forward_fused_metal_step(
        &self,
        config: LinearAttentionConfig,
        x: &Tensor,
        cache: &mut LinearAttentionCache,
        metal: &crate::MetalExecutor,
    ) -> Result<Tensor> {
        let (seq, _) = x.as_matrix()?;
        if seq != 1 {
            return Err(InferError::Dimension(format!(
                "linear-attn fused Metal attend seq=1, reçu {seq}"
            )));
        }
        ensure_conv_cache(config, cache)?;
        ensure_ssm_cache(config, cache)?;
        metal.linear_attention_cached_step(
            x,
            &self.in_proj_qkv,
            &self.in_proj_z,
            &self.in_proj_b,
            &self.in_proj_a,
            &self.out_proj,
            &self.conv1d_weight,
            &self.a_log,
            &self.dt_bias,
            &self.norm_weight,
            cache.conv.as_mut_slice(),
            cache.ssm.as_mut_slice(),
            crate::metal_backend::LinearAttentionStepSpec {
                num_key_heads: config.num_key_heads,
                num_value_heads: config.num_value_heads,
                key_head_dim: config.key_head_dim,
                value_head_dim: config.value_head_dim,
                conv_kernel_dim: config.conv_kernel_dim,
                rms_eps: config.rms_eps,
            },
        )
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn forward_resident_metal_step(
        &self,
        config: LinearAttentionConfig,
        x: &Tensor,
        cache: &mut LinearAttentionCache,
        metal: &crate::MetalExecutor,
    ) -> Result<Tensor> {
        let (seq, _) = x.as_matrix()?;
        if seq != 1 {
            return Err(InferError::Dimension(format!(
                "linear-attn résident Metal attend seq=1, reçu {seq}"
            )));
        }
        ensure_conv_cache(config, cache)?;
        ensure_ssm_cache(config, cache)?;
        metal.linear_attention_cached_step_resident(
            x,
            &self.in_proj_qkv,
            &self.in_proj_z,
            &self.in_proj_b,
            &self.in_proj_a,
            &self.out_proj,
            &self.conv1d_weight,
            &self.a_log,
            &self.dt_bias,
            &self.norm_weight,
            cache.conv.as_slice(),
            cache.ssm.as_slice(),
            &mut cache.metal,
            crate::metal_backend::LinearAttentionStepSpec {
                num_key_heads: config.num_key_heads,
                num_value_heads: config.num_value_heads,
                key_head_dim: config.key_head_dim,
                value_head_dim: config.value_head_dim,
                conv_kernel_dim: config.conv_kernel_dim,
                rms_eps: config.rms_eps,
            },
        )
    }

    /// Renvoie les poids du pas linear-attn résident (références partagées avec le
    /// chemin per-op), pour le chaînage d'UNE couche du decode résident complet (1c).
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) fn resident_weights(&self) -> crate::metal_backend::LinearAttnResidentWeights<'_> {
        crate::metal_backend::LinearAttnResidentWeights {
            in_proj_qkv: &self.in_proj_qkv,
            in_proj_z: &self.in_proj_z,
            in_proj_b: &self.in_proj_b,
            in_proj_a: &self.in_proj_a,
            out_proj: &self.out_proj,
            conv_weight: &self.conv1d_weight,
            a_log: self.a_log.data(),
            dt_bias: self.dt_bias.data(),
            norm_weight: self.norm_weight.data(),
        }
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl LinearAttentionCache {
    /// Renvoie l'état conv/ssm résident GPU s'il a été créé (par le chemin résident
    /// per-op pendant le prefill). Le decode résident complet (1c) le réutilise
    /// directement (l'état récurrent du prompt y est déjà consommé).
    pub(crate) fn metal_state(&self) -> Option<&crate::metal_backend::LinearAttentionMetalState> {
        self.metal.as_ref()
    }

    #[allow(
        dead_code,
        reason = "fallback single-layer conservé pour debug/tests Metal"
    )]
    pub(crate) fn snapshot_metal_state(
        &self,
        metal: &crate::MetalExecutor,
    ) -> Result<Option<crate::metal_backend::LinearAttentionMetalState>> {
        self.metal
            .as_ref()
            .map(|state| metal.snapshot_linear_attn_state(state))
            .transpose()
    }

    pub(crate) fn restore_metal_state_snapshot(
        &mut self,
        metal: &crate::MetalExecutor,
        snapshot: Option<crate::metal_backend::LinearAttentionMetalState>,
    ) -> Result<()> {
        match (self.metal.as_ref(), snapshot) {
            (Some(current), Some(snapshot)) => metal.restore_linear_attn_state(current, &snapshot),
            (None, Some(snapshot)) => {
                self.metal = Some(snapshot);
                Ok(())
            }
            (_, None) => {
                self.metal = None;
                Ok(())
            }
        }
    }
}

impl LinearAttentionCache {
    pub(crate) fn restore_cpu_state_from(&mut self, snapshot: &Self) {
        self.conv = snapshot.conv.clone();
        self.conv_dim = snapshot.conv_dim;
        self.ssm = snapshot.ssm.clone();
        self.ssm_shape = snapshot.ssm_shape;
    }
}

fn validate_weights(attn: &LinearAttention, config: LinearAttentionConfig) -> Result<()> {
    let conv_dim = config.conv_dim()?;
    let value_dim = config.value_dim()?;
    expect_linear_out(&attn.in_proj_qkv, conv_dim, "linear_attn.in_proj_qkv")?;
    expect_linear_out(&attn.in_proj_z, value_dim, "linear_attn.in_proj_z")?;
    expect_linear_out(
        &attn.in_proj_b,
        config.num_value_heads,
        "linear_attn.in_proj_b",
    )?;
    expect_linear_out(
        &attn.in_proj_a,
        config.num_value_heads,
        "linear_attn.in_proj_a",
    )?;
    expect_linear_out(&attn.out_proj, 0, "linear_attn.out_proj")?;
    match attn.a_log.shape() {
        [n] if *n == config.num_value_heads => {}
        shape => {
            return Err(InferError::Dimension(format!(
                "linear_attn.A_log attendu [{}], reçu {shape:?}",
                config.num_value_heads
            )))
        }
    }
    match attn.dt_bias.shape() {
        [n] if *n == config.num_value_heads => {}
        shape => {
            return Err(InferError::Dimension(format!(
                "linear_attn.dt_bias attendu [{}], reçu {shape:?}",
                config.num_value_heads
            )))
        }
    }
    match attn.norm_weight.shape() {
        [n] if *n == config.value_head_dim => {}
        [1, n] if *n == config.value_head_dim => {}
        shape => {
            return Err(InferError::Dimension(format!(
                "linear_attn.norm.weight attendu [{}], reçu {shape:?}",
                config.value_head_dim
            )))
        }
    }
    let conv_weight = conv_weight_shape(&attn.conv1d_weight, conv_dim, config.conv_kernel_dim)?;
    if conv_weight != (conv_dim, config.conv_kernel_dim) {
        return Err(InferError::Dimension(
            "linear_attn.conv1d.weight shape incohérente".to_string(),
        ));
    }
    if value_dim == 0 {
        return Err(InferError::Dimension(
            "linear-attn value_dim nul".to_string(),
        ));
    }
    Ok(())
}

fn expect_linear_out(linear: &Linear, expected: usize, name: &str) -> Result<()> {
    let shape = linear.weight().shape();
    let [out, _] = shape else {
        return Err(InferError::Dimension(format!(
            "{name}.weight attendu rang 2, reçu {shape:?}"
        )));
    };
    if expected != 0 && *out != expected {
        return Err(InferError::Dimension(format!(
            "{name}.weight sort {out}, attendu {expected}"
        )));
    }
    Ok(())
}

fn depthwise_causal_conv(
    qkv: &Tensor,
    weight: &Tensor,
    config: LinearAttentionConfig,
    cache: &mut LinearAttentionCache,
) -> Result<Tensor> {
    let (seq, conv_dim) = qkv.as_matrix()?;
    let expected_conv_dim = config.conv_dim()?;
    if conv_dim != expected_conv_dim {
        return Err(InferError::Dimension(format!(
            "linear qkv conv_dim reçu {conv_dim}, attendu {expected_conv_dim}"
        )));
    }
    conv_weight_shape(weight, conv_dim, config.conv_kernel_dim)?;
    ensure_conv_cache(config, cache)?;
    let mut history = cache.conv.clone();
    let keep = config.conv_kernel_dim - 1;
    if history.len() != keep * expected_conv_dim {
        return Err(InferError::Dimension(format!(
            "cache conv linear-attn len {}, attendu {}",
            history.len(),
            keep * expected_conv_dim
        )));
    }
    let mut conv_input = Vec::with_capacity((keep + seq) * conv_dim);
    conv_input.extend_from_slice(&history);
    conv_input.extend_from_slice(qkv.data());

    let mut out = vec![0.0_f32; seq * conv_dim];
    for pos in 0..seq {
        for channel in 0..conv_dim {
            let mut acc = 0.0_f32;
            for kernel_index in 0..config.conv_kernel_dim {
                let input_row = pos + kernel_index;
                let input = conv_input[input_row * conv_dim + channel];
                acc += input
                    * conv_weight_value(weight, channel, kernel_index, config.conv_kernel_dim)?;
            }
            out[pos * conv_dim + channel] = acc;
        }
    }
    let start = seq * conv_dim;
    let end = start + keep * conv_dim;
    history.clear();
    history.extend_from_slice(conv_input.get(start..end).ok_or_else(|| {
        InferError::Shape("nouvel état conv linear-attn hors bornes".to_string())
    })?);
    cache.conv = history;
    cache.conv_dim = Some(conv_dim);
    Tensor::from_vec(vec![seq, conv_dim], out)
}

fn ensure_conv_cache(
    config: LinearAttentionConfig,
    cache: &mut LinearAttentionCache,
) -> Result<()> {
    let conv_dim = config.conv_dim()?;
    let keep = config.conv_kernel_dim - 1;
    let state_len = keep
        .checked_mul(conv_dim)
        .ok_or_else(|| InferError::Shape("cache conv linear-attn trop grand".to_string()))?;
    match cache.conv_dim {
        Some(dim) if dim == conv_dim => {}
        Some(dim) => {
            return Err(InferError::Dimension(format!(
                "cache conv linear-attn dim {dim} incompatible avec {conv_dim}"
            )))
        }
        None => {
            cache.conv = vec![0.0; state_len];
            cache.conv_dim = Some(conv_dim);
        }
    }
    if cache.conv.len() != state_len {
        return Err(InferError::Dimension(format!(
            "cache conv linear-attn len {}, attendu {state_len}",
            cache.conv.len()
        )));
    }
    Ok(())
}

fn gated_delta(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    config: LinearAttentionConfig,
    cache: &mut LinearAttentionCache,
) -> Result<Tensor> {
    let (seq, key_dim) = q.as_matrix()?;
    let (_, k_dim) = k.as_matrix()?;
    let (_, value_dim) = v.as_matrix()?;
    if key_dim != config.key_dim()? || k_dim != key_dim || value_dim != config.value_dim()? {
        return Err(InferError::Dimension(format!(
            "gated-delta shapes q={:?}, k={:?}, v={:?}",
            q.shape(),
            k.shape(),
            v.shape()
        )));
    }
    let (g_rows, g_cols) = g.as_matrix()?;
    let (beta_rows, beta_cols) = beta.as_matrix()?;
    if g_rows != seq
        || beta_rows != seq
        || g_cols != config.num_value_heads
        || beta_cols != config.num_value_heads
    {
        return Err(InferError::Dimension(format!(
            "gated-delta gates g={:?}, beta={:?}, attendu [{seq},{}]",
            g.shape(),
            beta.shape(),
            config.num_value_heads
        )));
    }

    ensure_ssm_cache(config, cache)?;

    let repeat = config.num_value_heads / config.num_key_heads;
    let mut out = vec![0.0_f32; seq * value_dim];
    for pos in 0..seq {
        let q_row = q.row_slice(pos)?;
        let k_row = k.row_slice(pos)?;
        let v_row = v.row_slice(pos)?;
        let g_row = g.row_slice(pos)?;
        let beta_row = beta.row_slice(pos)?;
        let state_head_len = config.value_head_dim * config.key_head_dim;
        let out_row = &mut out[pos * value_dim..(pos + 1) * value_dim];
        cache
            .ssm
            .par_chunks_mut(state_head_len)
            .zip(out_row.par_chunks_mut(config.value_head_dim))
            .enumerate()
            .for_each(|(value_head, (state_head, out_head))| {
                let key_head = value_head / repeat;
                let q_head =
                    &q_row[key_head * config.key_head_dim..(key_head + 1) * config.key_head_dim];
                let k_head =
                    &k_row[key_head * config.key_head_dim..(key_head + 1) * config.key_head_dim];
                let v_head = &v_row
                    [value_head * config.value_head_dim..(value_head + 1) * config.value_head_dim];
                let decay = g_row[value_head];
                let beta_value = beta_row[value_head];

                for value_col in 0..config.value_head_dim {
                    let row_start = value_col * config.key_head_dim;
                    let state_row = &mut state_head[row_start..row_start + config.key_head_dim];
                    let mut kv_mem = 0.0_f32;
                    for (state_value, key_value) in state_row.iter_mut().zip(k_head.iter()) {
                        *state_value *= decay;
                        kv_mem += *state_value * *key_value;
                    }
                    let delta = (v_head[value_col] - kv_mem) * beta_value;
                    let mut y = 0.0_f32;
                    for ((state_value, key_value), query_value) in
                        state_row.iter_mut().zip(k_head.iter()).zip(q_head.iter())
                    {
                        *state_value += delta * *key_value;
                        y += *state_value * *query_value;
                    }
                    out_head[value_col] = y;
                }
            });
    }
    Tensor::from_vec(vec![seq, value_dim], out)
}

fn ensure_ssm_cache(config: LinearAttentionConfig, cache: &mut LinearAttentionCache) -> Result<()> {
    let state_len = config
        .num_value_heads
        .checked_mul(config.value_head_dim)
        .and_then(|len| len.checked_mul(config.key_head_dim))
        .ok_or_else(|| InferError::Shape("cache ssm linear-attn trop grand".to_string()))?;
    let expected_shape = (
        config.num_value_heads,
        config.value_head_dim,
        config.key_head_dim,
    );
    match cache.ssm_shape {
        Some(shape) if shape == expected_shape => {}
        Some(shape) => {
            return Err(InferError::Dimension(format!(
                "cache ssm linear-attn shape {shape:?} incompatible"
            )))
        }
        None => {
            cache.ssm = vec![0.0; state_len];
            cache.ssm_shape = Some(expected_shape);
        }
    }
    if cache.ssm.len() != state_len {
        return Err(InferError::Dimension(format!(
            "cache ssm linear-attn len {}, attendu {state_len}",
            cache.ssm.len()
        )));
    }
    Ok(())
}

fn rms_norm_heads_unit(x: &Tensor, heads: usize, head_dim: usize, eps: f32) -> Result<Tensor> {
    let (seq, dim) = x.as_matrix()?;
    if heads == 0 || head_dim == 0 || dim != heads * head_dim {
        return Err(InferError::Dimension(format!(
            "RMSNorm unit heads invalide: x={:?}, heads={heads}, head_dim={head_dim}",
            x.shape()
        )));
    }
    let mut out = x.data().to_vec();
    for pos in 0..seq {
        let row_start = pos * dim;
        for head in 0..heads {
            let head_start = row_start + head * head_dim;
            let xs = &x.data()[head_start..head_start + head_dim];
            let mean_square = xs.iter().map(|value| value * value).sum::<f32>() / head_dim as f32;
            let inv_rms = 1.0 / (mean_square + eps).sqrt();
            for col in 0..head_dim {
                out[head_start + col] = xs[col] * inv_rms;
            }
        }
    }
    Tensor::from_vec(vec![seq, dim], out)
}

fn rms_norm_heads(
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

fn slice_columns(x: &Tensor, start: usize, len: usize) -> Result<Tensor> {
    let (rows, cols) = x.as_matrix()?;
    let end = start
        .checked_add(len)
        .ok_or_else(|| InferError::Shape("slice colonnes trop large".to_string()))?;
    if end > cols {
        return Err(InferError::Dimension(format!(
            "slice colonnes [{start},{end}) hors bornes pour {:?}",
            x.shape()
        )));
    }
    let mut out = Vec::with_capacity(rows * len);
    for row in 0..rows {
        let row_start = row * cols;
        out.extend_from_slice(&x.data()[row_start + start..row_start + end]);
    }
    Tensor::from_vec(vec![rows, len], out)
}

fn compute_decay(
    a: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    num_value_heads: usize,
) -> Result<Tensor> {
    let (seq, cols) = a.as_matrix()?;
    if cols != num_value_heads {
        return Err(InferError::Dimension(format!(
            "linear_attn.in_proj_a sort {cols}, attendu {num_value_heads}"
        )));
    }
    let a_log_data = vector_data(a_log, num_value_heads, "A_log")?;
    let dt_bias_data = vector_data(dt_bias, num_value_heads, "dt_bias")?;
    let mut out = vec![0.0_f32; seq * num_value_heads];
    for pos in 0..seq {
        let row = a.row_slice(pos)?;
        for head in 0..num_value_heads {
            let dt = softplus(row[head] + dt_bias_data[head]);
            out[pos * num_value_heads + head] = (-(a_log_data[head].exp()) * dt).exp();
        }
    }
    Tensor::from_vec(vec![seq, num_value_heads], out)
}

fn vector_data<'a>(tensor: &'a Tensor, len: usize, name: &str) -> Result<&'a [f32]> {
    match tensor.shape() {
        [n] if *n == len => Ok(tensor.data()),
        [1, n] if *n == len => Ok(tensor.data()),
        shape => Err(InferError::Dimension(format!(
            "linear_attn.{name} attendu [{len}], reçu {shape:?}"
        ))),
    }
}

fn conv_weight_shape(weight: &Tensor, conv_dim: usize, kernel: usize) -> Result<(usize, usize)> {
    match weight.shape() {
        [channels, k, one] if *channels == conv_dim && *k == kernel && *one == 1 => {
            Ok((*channels, *k))
        }
        [channels, one, k] if *channels == conv_dim && *one == 1 && *k == kernel => {
            Ok((*channels, *k))
        }
        shape => Err(InferError::Dimension(format!(
            "conv1d linear-attn attendu [{conv_dim},{kernel},1] ou [{conv_dim},1,{kernel}], reçu {shape:?}"
        ))),
    }
}

fn conv_weight_value(
    weight: &Tensor,
    channel: usize,
    kernel_index: usize,
    kernel: usize,
) -> Result<f32> {
    match weight.shape() {
        [_, _, one] if *one == 1 => {
            let index = channel
                .checked_mul(kernel)
                .and_then(|base| base.checked_add(kernel_index))
                .ok_or_else(|| InferError::Shape("index conv1d trop grand".to_string()))?;
            Ok(weight.data()[index])
        }
        [_, one, _] if *one == 1 => {
            let index = channel
                .checked_mul(kernel)
                .and_then(|base| base.checked_add(kernel_index))
                .ok_or_else(|| InferError::Shape("index conv1d trop grand".to_string()))?;
            Ok(weight.data()[index])
        }
        shape => Err(InferError::Dimension(format!(
            "conv1d linear-attn shape invalide: {shape:?}"
        ))),
    }
}

fn key_scale(head_dim: usize, squared: bool) -> f32 {
    let inv = (head_dim as f32).powf(-0.5);
    if squared {
        inv * inv
    } else {
        inv
    }
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else {
        (1.0 + value.exp()).ln()
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn linear_attn_resident_step_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        // Défaut **ON** (phase 1a full-rust : +43 % decode mesuré, sortie greedy
        // bit-identique). Kill-switch : `RETI_RUST_LINEAR_ATTN_RESIDENT=0`
        // (ou `false`/`off`) pour revenir au chemin par-op (ex. régression /
        // diagnostic). Toute autre valeur, ou variable absente → résident.
        env_flag("RETI_RUST_LINEAR_ATTN_RESIDENT", true)
    })
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn linear_attn_fused_step_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_LINEAR_ATTN_FUSED", false))
}

fn checked_mul(left: usize, right: usize, label: &str) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| InferError::Shape(format!("{label} trop grand")))
}

#[cfg(test)]
mod tests;
