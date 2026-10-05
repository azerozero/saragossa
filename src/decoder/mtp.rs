//! Chargement et forward de la tête MTP.
//!
//! Supporte le format **OptiQ int4 + MLP dense** (27B-OptiQ : `.weight` u32 packé
//! 4-bit gs64 + `.scales`/`.biases` bf16, MLP `gate/up/down_proj`) en plus des
//! normes/`fc` bf16. Les variantes A3B (MLP MoE) ne sont pas chargées par ce
//! chemin (erreur claire) — le 27B est la cible MTP.

use super::*;
use crate::quantization::bytes_to_u32;
use crate::safetensor::bytes_to_dense_f32;
use crate::AffineQuantizedTensor;
#[cfg(all(target_os = "macos", feature = "metal"))]
use safetensors::Dtype;
use safetensors::{tensor::TensorView, SafeTensors};
use std::collections::HashMap;

pub(super) struct MtpLoadContext<'a> {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    metal: Option<&'a crate::MetalExecutor>,
    marker: std::marker::PhantomData<&'a ()>,
}

impl MtpLoadContext<'_> {
    pub(super) fn cpu() -> Self {
        Self {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            metal: None,
            marker: std::marker::PhantomData,
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(super) fn metal(executor: &crate::MetalExecutor) -> MtpLoadContext<'_> {
        MtpLoadContext {
            metal: Some(executor),
            marker: std::marker::PhantomData,
        }
    }
}

impl MtpHead {
    pub(super) fn from_sidecar(
        path: impl AsRef<Path>,
        _config: &CausalDecoderConfig,
        context: &MtpLoadContext<'_>,
    ) -> Result<Self> {
        let bytes = std::fs::read(path.as_ref())
            .map_err(|source| InferError::Config(format!("lecture sidecar MTP: {source}")))?;
        let st = SafeTensors::deserialize(&bytes)
            .map_err(|source| InferError::Config(format!("parse sidecar MTP: {source}")))?;
        // Résout les clés normalisées (sans préfixe `language_model.`) → clé réelle.
        let resolve: HashMap<String, String> = st
            .names()
            .into_iter()
            .map(|name| (normalize_mtp_sidecar_key(name), name.to_string()))
            .collect();

        if !resolve.contains_key("mtp.layers.0.mlp.gate_proj.weight") {
            return Err(InferError::Config(
                "tête MTP MoE (A3B) non supportée par ce chemin — cible = 27B dense".to_string(),
            ));
        }

        let pre_fc_norm_embedding =
            load_mtp_norm(&st, &resolve, "mtp.pre_fc_norm_embedding.weight")?;
        let pre_fc_norm_hidden = load_mtp_norm(&st, &resolve, "mtp.pre_fc_norm_hidden.weight")?;
        let fc = load_mtp_linear(&st, &resolve, "mtp.fc", context)?;
        let input_norm = load_mtp_norm(&st, &resolve, "mtp.layers.0.input_layernorm.weight")?;
        let attention = FullAttention {
            q_proj: load_mtp_linear(&st, &resolve, "mtp.layers.0.self_attn.q_proj", context)?,
            k_proj: load_mtp_linear(&st, &resolve, "mtp.layers.0.self_attn.k_proj", context)?,
            v_proj: Some(load_mtp_linear(
                &st,
                &resolve,
                "mtp.layers.0.self_attn.v_proj",
                context,
            )?),
            o_proj: load_mtp_linear(&st, &resolve, "mtp.layers.0.self_attn.o_proj", context)?,
            q_norm: Some(load_mtp_norm(
                &st,
                &resolve,
                "mtp.layers.0.self_attn.q_norm.weight",
            )?),
            k_norm: Some(load_mtp_norm(
                &st,
                &resolve,
                "mtp.layers.0.self_attn.k_norm.weight",
            )?),
            num_key_value_heads: None,
            head_dim: None,
            rope_dims: None,
            rope_frequency_dim: None,
            value_norm: false,
            // MTP = sidecar Qwen, base RoPE unique de la config, positions
            // brutes, attention pleine.
            rope_theta: None,
            rope_position_scale: None,
            sliding_window: None,
        };
        let post_attention_norm = load_mtp_norm(
            &st,
            &resolve,
            "mtp.layers.0.post_attention_layernorm.weight",
        )?;
        let mlp = FeedForward::Dense(Box::new(GatedMlp::new(
            load_mtp_linear(&st, &resolve, "mtp.layers.0.mlp.gate_proj", context)?,
            load_mtp_linear(&st, &resolve, "mtp.layers.0.mlp.up_proj", context)?,
            load_mtp_linear(&st, &resolve, "mtp.layers.0.mlp.down_proj", context)?,
        )));
        let norm = load_mtp_norm(&st, &resolve, "mtp.norm.weight")?;
        Ok(Self {
            pre_fc_norm_embedding,
            pre_fc_norm_hidden,
            fc,
            layer: MtpLayer {
                input_norm,
                attention,
                post_attention_norm,
                mlp,
            },
            norm,
        })
    }
}

pub(super) fn concat_row_pair(left: &Tensor, right: &Tensor) -> Result<Tensor> {
    let left = left.as_row()?;
    let right = right.as_row()?;
    let mut out = Vec::with_capacity(left.len() + right.len());
    out.extend_from_slice(left);
    out.extend_from_slice(right);
    Tensor::row(out)
}

fn normalize_mtp_sidecar_key(key: &str) -> String {
    if let Some(rest) = key.strip_prefix("language_model.mtp.") {
        format!("mtp.{rest}")
    } else if key.starts_with("mtp.") {
        key.to_string()
    } else if is_bare_mtp_sidecar_key(key) {
        format!("mtp.{key}")
    } else {
        key.to_string()
    }
}

fn is_bare_mtp_sidecar_key(key: &str) -> bool {
    key.starts_with("fc.")
        || key.starts_with("layers.")
        || key.starts_with("pre_fc_norm_")
        || key.starts_with("norm.")
}

/// Résout une vue tensor par clé normalisée.
fn st_view<'a>(
    st: &'a SafeTensors,
    resolve: &HashMap<String, String>,
    norm_key: &str,
) -> Result<TensorView<'a>> {
    let actual = resolve
        .get(norm_key)
        .ok_or_else(|| InferError::MissingWeight(norm_key.to_string()))?;
    st.tensor(actual)
        .map_err(|source| InferError::Config(format!("MTP tensor {norm_key}: {source}")))
}

/// Charge une norme bf16 [dim] + correction unit-offset (mean<0.5 → +1.0).
fn load_mtp_norm(st: &SafeTensors, resolve: &HashMap<String, String>, key: &str) -> Result<Tensor> {
    let view = st_view(st, resolve, key)?;
    let data = bytes_to_dense_f32(view.data(), view.dtype(), key)?;
    let tensor = Tensor::from_vec(view.shape().to_vec(), data)?;
    if tensor.rank() == 1 && !tensor.is_empty() {
        let mean = tensor.data().iter().sum::<f32>() / tensor.len() as f32;
        if mean < 0.5 {
            return Ok(tensor.map(|value| value + 1.0));
        }
    }
    Ok(tensor)
}

/// Charge un Linear MTP : quantifié 4-bit gs64 si `.scales` présent, sinon dense bf16.
fn load_mtp_linear(
    st: &SafeTensors,
    resolve: &HashMap<String, String>,
    prefix: &str,
    _context: &MtpLoadContext<'_>,
) -> Result<Linear> {
    let weight = st_view(st, resolve, &format!("{prefix}.weight"))?;
    let scales_key = format!("{prefix}.scales");
    if resolve.contains_key(&scales_key) {
        // Quantifié affine 4-bit (.weight u32 packé, .scales/.biases bf16).
        let packed_shape = weight.shape().to_vec();
        let scales_view = st_view(st, resolve, &scales_key)?;
        let biases_key = format!("{prefix}.biases");
        let biases_view = st_view(st, resolve, &biases_key)?;
        let (_, group_size) = infer_mtp_affine_layout(
            &packed_shape,
            scales_view.shape(),
            Some(4),
            None,
            &scales_key,
        )?;
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if let Some(quant) = mtp_affine_from_native_bf16(
            &weight,
            &scales_view,
            &biases_view,
            group_size,
            4,
            _context,
            prefix,
        )? {
            return Linear::from_weight(LinearWeight::AffineQuantized(quant), None);
        }
        let packed = bytes_to_u32(weight.data(), &format!("{prefix}.weight"))?;
        let scales = Tensor::from_vec(
            scales_view.shape().to_vec(),
            bytes_to_dense_f32(scales_view.data(), scales_view.dtype(), &scales_key)?,
        )?;
        let biases = Tensor::from_vec(
            biases_view.shape().to_vec(),
            bytes_to_dense_f32(biases_view.data(), biases_view.dtype(), &biases_key)?,
        )?;
        let quant =
            AffineQuantizedTensor::new(&packed_shape, packed, scales, biases, group_size, 4)?;
        Linear::from_weight(LinearWeight::AffineQuantized(quant), None)
    } else {
        // Dense bf16 (ex. mtp.fc).
        let data = bytes_to_dense_f32(weight.data(), weight.dtype(), prefix)?;
        let dense = Tensor::from_vec(weight.shape().to_vec(), data)?;
        Linear::new(dense, None)
    }
}

pub(super) fn load_mtp_draft_lm_head(
    path: impl AsRef<Path>,
    expected_in_dim: usize,
    _context: &MtpLoadContext<'_>,
) -> Result<Linear> {
    let bytes = std::fs::read(path.as_ref())
        .map_err(|source| InferError::Config(format!("lecture draft lm_head MTP: {source}")))?;
    let st = SafeTensors::deserialize(&bytes)
        .map_err(|source| InferError::Config(format!("parse draft lm_head MTP: {source}")))?;
    let resolve: HashMap<String, String> = st
        .names()
        .into_iter()
        .map(|name| (normalize_mtp_draft_lm_head_key(name), name.to_string()))
        .collect();
    let head = load_mtp_draft_linear(&st, &resolve, "lm_head", expected_in_dim, _context)?;
    let shape = head.weight().shape();
    if shape.len() != 2 || shape[1] != expected_in_dim {
        return Err(InferError::Dimension(format!(
            "draft lm_head MTP input={}, attendu {expected_in_dim}",
            shape.get(1).copied().unwrap_or(0)
        )));
    }
    Ok(head)
}

fn normalize_mtp_draft_lm_head_key(key: &str) -> String {
    for prefix in [
        "language_model._mtplx_draft_lm_head.",
        "model._mtplx_draft_lm_head.",
        "_mtplx_draft_lm_head.",
        "language_model.lm_head.",
        "model.lm_head.",
        "lm_head.",
    ] {
        if let Some(rest) = key.strip_prefix(prefix) {
            return format!("lm_head.{rest}");
        }
    }
    if matches!(key, "weight" | "scales" | "biases") {
        format!("lm_head.{key}")
    } else {
        key.to_string()
    }
}

fn load_mtp_draft_linear(
    st: &SafeTensors,
    resolve: &HashMap<String, String>,
    prefix: &str,
    expected_in_dim: usize,
    _context: &MtpLoadContext<'_>,
) -> Result<Linear> {
    let weight = st_view(st, resolve, &format!("{prefix}.weight"))?;
    let scales_key = format!("{prefix}.scales");
    if resolve.contains_key(&scales_key) {
        let packed_shape = weight.shape().to_vec();
        let scales_view = st_view(st, resolve, &scales_key)?;
        let biases_key = format!("{prefix}.biases");
        let biases_view = st_view(st, resolve, &biases_key)?;
        let (bits, group_size) = infer_mtp_affine_layout(
            &packed_shape,
            scales_view.shape(),
            None,
            Some(expected_in_dim),
            &scales_key,
        )?;
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if let Some(quant) = mtp_affine_from_native_bf16(
            &weight,
            &scales_view,
            &biases_view,
            group_size,
            bits,
            _context,
            prefix,
        )? {
            return Linear::from_weight(LinearWeight::AffineQuantized(quant), None);
        }
        let packed = bytes_to_u32(weight.data(), &format!("{prefix}.weight"))?;
        let scales = Tensor::from_vec(
            scales_view.shape().to_vec(),
            bytes_to_dense_f32(scales_view.data(), scales_view.dtype(), &scales_key)?,
        )?;
        let biases = Tensor::from_vec(
            biases_view.shape().to_vec(),
            bytes_to_dense_f32(biases_view.data(), biases_view.dtype(), &biases_key)?,
        )?;
        let quant =
            AffineQuantizedTensor::new(&packed_shape, packed, scales, biases, group_size, bits)?;
        Linear::from_weight(LinearWeight::AffineQuantized(quant), None)
    } else {
        let data = bytes_to_dense_f32(weight.data(), weight.dtype(), prefix)?;
        let dense = Tensor::from_vec(weight.shape().to_vec(), data)?;
        Linear::new(dense, None)
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn mtp_affine_from_native_bf16(
    weight: &TensorView<'_>,
    scales: &TensorView<'_>,
    biases: &TensorView<'_>,
    group_size: usize,
    bits: usize,
    context: &MtpLoadContext<'_>,
    prefix: &str,
) -> Result<Option<AffineQuantizedTensor>> {
    if !crate::runtime_flags::single_copy_weights_enabled()
        || scales.dtype() != Dtype::BF16
        || biases.dtype() != Dtype::BF16
    {
        return Ok(None);
    }
    let Some(metal) = context.metal else {
        return Ok(None);
    };
    let packed = bytes_to_u32(weight.data(), &format!("{prefix}.weight"))?;
    let packed_len = packed.len();
    let scales_len = scales.data().len() / std::mem::size_of::<u16>();
    let biases_len = biases.data().len() / std::mem::size_of::<u16>();
    Ok(Some(AffineQuantizedTensor::new_metal_shared_bf16(
        weight.shape(),
        metal.weight_buffer_from_u32(&packed)?,
        0,
        packed_len,
        scales.shape(),
        metal.weight_buffer_from_bf16_bytes(scales.data())?,
        0,
        scales_len,
        biases.shape(),
        metal.weight_buffer_from_bf16_bytes(biases.data())?,
        0,
        biases_len,
        group_size,
        bits,
    )?))
}

fn infer_mtp_affine_layout(
    packed_shape: &[usize],
    scales_shape: &[usize],
    fixed_bits: Option<usize>,
    expected_cols: Option<usize>,
    key: &str,
) -> Result<(usize, usize)> {
    let &[rows, packed_cols] = packed_shape else {
        return Err(InferError::Dimension(format!(
            "MTP {key} poids quantifié attendu rang 2, reçu {packed_shape:?}"
        )));
    };
    let &[scale_rows, groups] = scales_shape else {
        return Err(InferError::Dimension(format!(
            "MTP {key} scales attendues rang 2, reçu {scales_shape:?}"
        )));
    };
    if scale_rows != rows {
        return Err(InferError::Dimension(format!(
            "MTP {key} scales rows={scale_rows}, poids rows={rows}"
        )));
    }
    if groups == 0 {
        return Err(InferError::Shape(format!("MTP {key} scales sans groupe")));
    }
    let packed_bits = packed_cols
        .checked_mul(32)
        .ok_or_else(|| InferError::Shape(format!("MTP {key} poids trop large")))?;
    let bits = match (fixed_bits, expected_cols) {
        (Some(bits), _) => bits,
        (None, Some(cols)) if cols > 0 && packed_bits % cols == 0 => packed_bits / cols,
        (None, Some(cols)) => {
            return Err(InferError::Dimension(format!(
                "MTP {key} packed_bits={packed_bits} non divisible par input={cols}"
            )))
        }
        (None, None) => {
            return Err(InferError::Config(format!(
                "MTP {key} bits ou input attendu requis"
            )))
        }
    };
    if bits == 0 || bits > 16 {
        return Err(InferError::Config(format!(
            "MTP {key} bits inférés invalides: {bits}"
        )));
    }
    let cols = match expected_cols {
        Some(cols) => cols,
        None if packed_bits % bits == 0 => packed_bits / bits,
        None => {
            return Err(InferError::Shape(format!(
                "MTP {key} packed_cols={packed_cols} incompatible avec bits={bits}"
            )))
        }
    };
    if cols == 0 || cols % groups != 0 {
        return Err(InferError::Dimension(format!(
            "MTP {key} cols={cols} incompatibles avec groups={groups}"
        )));
    }
    Ok((bits, cols / groups))
}

#[cfg(test)]
mod tests {
    use super::{infer_mtp_affine_layout, normalize_mtp_sidecar_key};

    #[test]
    fn normalizes_bare_quantized_fc_sidecar_keys() {
        // Sidecar mlx-community/*-MTP-4bit : `fc` quantifié (weight + scales + biases).
        for key in ["fc.weight", "fc.scales", "fc.biases"] {
            assert_eq!(normalize_mtp_sidecar_key(key), format!("mtp.{key}"));
        }
        assert_eq!(normalize_mtp_sidecar_key("mtp.fc.weight"), "mtp.fc.weight");
        assert_eq!(
            normalize_mtp_sidecar_key("lm_head.weight"),
            "lm_head.weight"
        );
    }

    #[test]
    fn infer_mtp_affine_group_size_accepts_gs32_and_gs64() {
        let (_, gs32) = infer_mtp_affine_layout(&[12_288, 640], &[12_288, 160], Some(4), None, "q")
            .expect("invariant: gs32 MTPLX valide");
        let (_, gs64) = infer_mtp_affine_layout(&[12_288, 640], &[12_288, 80], Some(4), None, "q")
            .expect("invariant: gs64 OptiQ valide");
        assert_eq!(gs32, 32);
        assert_eq!(gs64, 64);
    }

    #[test]
    fn infer_mtp_affine_layout_accepts_q3_draft_head() {
        let (bits, group_size) =
            infer_mtp_affine_layout(&[152_064, 480], &[152_064, 80], None, Some(5120), "head")
                .expect("invariant: q3 gs64 draft head valide");
        assert_eq!(bits, 3);
        assert_eq!(group_size, 64);
    }
}
