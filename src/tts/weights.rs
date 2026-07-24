use super::SafetensorPayload;
use crate::decoder::DecoderTensor;
use crate::{AffineQuantizedTensor, LinearWeight, Result};
use std::collections::HashMap;

pub(super) fn read_linear_layer(
    payload: &SafetensorPayload,
    source_prefix: &str,
    group_size: usize,
    bits: usize,
) -> Result<crate::Linear> {
    let weight = read_linear_weight(payload, source_prefix, group_size, bits)?;
    let bias_key = format!("{source_prefix}.bias");
    let bias = if payload.contains(&bias_key) {
        Some(payload.read_dense_tensor(&bias_key)?)
    } else {
        None
    };
    crate::Linear::from_weight(weight, bias)
}

pub(super) fn insert_linear(
    tensors: &mut HashMap<String, DecoderTensor>,
    payload: &SafetensorPayload,
    source_prefix: &str,
    target_prefix: &str,
    group_size: usize,
    bits: usize,
) -> Result<()> {
    tensors.insert(
        format!("{target_prefix}.weight"),
        DecoderTensor::LinearWeight(read_linear_weight(
            payload,
            source_prefix,
            group_size,
            bits,
        )?),
    );
    let bias_key = format!("{source_prefix}.bias");
    if payload.contains(&bias_key) {
        tensors.insert(
            format!("{target_prefix}.bias"),
            DecoderTensor::Dense(payload.read_dense_tensor(&bias_key)?),
        );
    }
    Ok(())
}

fn read_linear_weight(
    payload: &SafetensorPayload,
    source_prefix: &str,
    group_size: usize,
    bits: usize,
) -> Result<LinearWeight> {
    let weight_key = format!("{source_prefix}.weight");
    let scales_key = format!("{source_prefix}.scales");
    if payload.contains(&scales_key) {
        let packed = payload.read_u32_tensor(&weight_key)?;
        let scales = payload.read_dense_tensor(&scales_key)?;
        let biases = payload.read_dense_tensor(&format!("{source_prefix}.biases"))?;
        let packed_shape = payload.entry(&weight_key)?.shape.clone();
        return Ok(LinearWeight::AffineQuantized(AffineQuantizedTensor::new(
            &packed_shape,
            packed,
            scales,
            biases,
            group_size,
            bits,
        )?));
    }
    Ok(LinearWeight::Dense(payload.read_dense_tensor(&weight_key)?))
}

pub(super) fn copy_dense(
    tensors: &mut HashMap<String, DecoderTensor>,
    payload: &SafetensorPayload,
    source: &str,
    target: &str,
) -> Result<()> {
    tensors.insert(
        target.to_string(),
        DecoderTensor::Dense(payload.read_dense_tensor(source)?),
    );
    Ok(())
}
