//! Chargement et forward de la tête MTP.
//!
//! Supporte le format **OptiQ int4 + MLP dense** (27B-OptiQ : `.weight` u32 packé
//! 4-bit gs64 + `.scales`/`.biases` bf16, MLP `gate/up/down_proj`) en plus des
//! normes/`fc` bf16. Les variantes A3B (MLP MoE) ne sont pas chargées par ce
//! chemin (erreur claire) — le 27B est la cible MTP.

use super::*;
use crate::safetensor::bytes_to_dense_f32;
use crate::AffineQuantizedTensor;
use safetensors::{tensor::TensorView, SafeTensors};
use std::collections::HashMap;

impl MtpHead {
    pub(super) fn from_sidecar(
        path: impl AsRef<Path>,
        _config: &CausalDecoderConfig,
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
        let fc = load_mtp_linear(&st, &resolve, "mtp.fc")?;
        let input_norm = load_mtp_norm(&st, &resolve, "mtp.layers.0.input_layernorm.weight")?;
        let attention = FullAttention {
            q_proj: load_mtp_linear(&st, &resolve, "mtp.layers.0.self_attn.q_proj")?,
            k_proj: load_mtp_linear(&st, &resolve, "mtp.layers.0.self_attn.k_proj")?,
            v_proj: load_mtp_linear(&st, &resolve, "mtp.layers.0.self_attn.v_proj")?,
            o_proj: load_mtp_linear(&st, &resolve, "mtp.layers.0.self_attn.o_proj")?,
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
        };
        let post_attention_norm = load_mtp_norm(
            &st,
            &resolve,
            "mtp.layers.0.post_attention_layernorm.weight",
        )?;
        let mlp = FeedForward::Dense(Box::new(GatedMlp::new(
            load_mtp_linear(&st, &resolve, "mtp.layers.0.mlp.gate_proj")?,
            load_mtp_linear(&st, &resolve, "mtp.layers.0.mlp.up_proj")?,
            load_mtp_linear(&st, &resolve, "mtp.layers.0.mlp.down_proj")?,
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

pub(super) fn push_generated(generated: &mut Vec<usize>, context: &mut Vec<usize>, token: usize) {
    generated.push(token);
    context.push(token);
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
    key == "fc.weight"
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
) -> Result<Linear> {
    let weight = st_view(st, resolve, &format!("{prefix}.weight"))?;
    let scales_key = format!("{prefix}.scales");
    if resolve.contains_key(&scales_key) {
        // Quantifié affine 4-bit gs64 (.weight u32 packé, .scales/.biases bf16).
        let shape = weight.shape().to_vec();
        let packed: Vec<u32> = weight
            .data()
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        let scales_view = st_view(st, resolve, &scales_key)?;
        let scales = Tensor::from_vec(
            scales_view.shape().to_vec(),
            bytes_to_dense_f32(scales_view.data(), scales_view.dtype(), &scales_key)?,
        )?;
        let biases_view = st_view(st, resolve, &format!("{prefix}.biases"))?;
        let biases = Tensor::from_vec(
            biases_view.shape().to_vec(),
            bytes_to_dense_f32(
                biases_view.data(),
                biases_view.dtype(),
                &format!("{prefix}.biases"),
            )?,
        )?;
        let quant = AffineQuantizedTensor::new(&shape, packed, scales, biases, 64, 4)?;
        Linear::from_weight(LinearWeight::AffineQuantized(quant), None)
    } else {
        // Dense bf16 (ex. mtp.fc).
        let data = bytes_to_dense_f32(weight.data(), weight.dtype(), prefix)?;
        let dense = Tensor::from_vec(weight.shape().to_vec(), data)?;
        Linear::new(dense, None)
    }
}
