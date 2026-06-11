//! Chargement des poids quantifiés et des échelles FP8 Qwen.

use super::*;

pub(super) fn quantized_contract_shape(
    config: &ModelConfig,
    spec: &TensorSpec,
    entry: &ShardTensorEntry,
    entries: &HashMap<String, TensorEntryRef>,
) -> Result<Vec<usize>> {
    let quant = config.quantization.as_ref().ok_or_else(|| {
        InferError::Config(format!(
            "poids quantifié {} sans quantization_config",
            spec.source
        ))
    })?;
    let (group_size, bits) = quant_params_for(quant, &spec.source)?;
    let scales_key = replace_weight_suffix(&spec.source, ".scales")?;
    let biases_key = replace_weight_suffix(&spec.source, ".biases")?;
    let scales = entries
        .get(&scales_key)
        .ok_or_else(|| InferError::MissingWeight(scales_key.clone()))?
        .entry
        .clone();
    let biases = entries
        .get(&biases_key)
        .ok_or_else(|| InferError::MissingWeight(biases_key.clone()))?
        .entry
        .clone();
    validate_dense_contract_dtype(&scales_key, scales.dtype)?;
    validate_dense_contract_dtype(&biases_key, biases.dtype)?;

    if is_moe_expert_weight(&spec.target) && entry.shape.len() == 3 {
        return quantized_expert_contract_shape(
            &entry.shape,
            &scales,
            &biases,
            group_size,
            bits,
            &scales_key,
            &biases_key,
        );
    }
    quantized_linear_contract_shape(
        &entry.shape,
        &scales,
        &biases,
        group_size,
        bits,
        &scales_key,
        &biases_key,
    )
}

fn quantized_linear_contract_shape(
    packed_shape: &[usize],
    scales: &ShardTensorEntry,
    biases: &ShardTensorEntry,
    group_size: usize,
    bits: usize,
    scales_key: &str,
    biases_key: &str,
) -> Result<Vec<usize>> {
    let [rows, packed_cols] = packed_shape else {
        return Err(InferError::Dimension(format!(
            "poids quantifié attendu rang 2, reçu {packed_shape:?}"
        )));
    };
    let cols = unpacked_cols(*packed_cols, group_size, bits)?;
    let groups = cols / group_size;
    expect_entry_shape(scales_key, scales, &[*rows, groups])?;
    expect_entry_shape(biases_key, biases, &[*rows, groups])?;
    Ok(vec![*rows, cols])
}

fn quantized_expert_contract_shape(
    packed_shape: &[usize],
    scales: &ShardTensorEntry,
    biases: &ShardTensorEntry,
    group_size: usize,
    bits: usize,
    scales_key: &str,
    biases_key: &str,
) -> Result<Vec<usize>> {
    let [experts, rows, packed_cols] = packed_shape else {
        return Err(InferError::Dimension(format!(
            "poids expert quantifié attendu rang 3, reçu {packed_shape:?}"
        )));
    };
    let cols = unpacked_cols(*packed_cols, group_size, bits)?;
    let groups = cols / group_size;
    expect_entry_shape(scales_key, scales, &[*experts, *rows, groups])?;
    expect_entry_shape(biases_key, biases, &[*experts, *rows, groups])?;
    Ok(vec![*experts, *rows, cols])
}

fn unpacked_cols(packed_cols: usize, group_size: usize, bits: usize) -> Result<usize> {
    let cols_times_bits = packed_cols
        .checked_mul(32)
        .ok_or_else(|| InferError::Shape("poids quantifié trop large".to_string()))?;
    if cols_times_bits % bits != 0 {
        return Err(InferError::Shape(format!(
            "packed_cols={packed_cols} incompatible avec bits={bits}"
        )));
    }
    let cols = cols_times_bits / bits;
    if cols % group_size != 0 {
        return Err(InferError::Shape(format!(
            "cols={cols} non divisible par group_size={group_size}"
        )));
    }
    Ok(cols)
}

pub(super) fn validate_dense_contract_dtype(name: &str, dtype: Dtype) -> Result<()> {
    match dtype {
        Dtype::F32 | Dtype::BF16 | Dtype::F16 | Dtype::F8_E4M3 | Dtype::F8_E5M2 => Ok(()),
        _ => Err(InferError::UnsupportedDtype {
            name: name.to_string(),
            dtype,
        }),
    }
}

pub(super) fn validate_optional_fp8_scale_inv(
    spec: &TensorSpec,
    weight: &ShardTensorEntry,
    entries: &HashMap<String, TensorEntryRef>,
) -> Result<()> {
    let scale_key = replace_weight_suffix(&spec.source, ".weight_scale_inv")?;
    let Some(entry_ref) = entries.get(&scale_key) else {
        return Ok(());
    };
    let entry = &entry_ref.entry;
    validate_dense_contract_dtype(&scale_key, entry.dtype)?;
    validate_fp8_scale_shape(&weight.shape, &entry.shape, &scale_key, FP8_SCALE_BLOCK)
}

fn validate_fp8_scale_shape(
    weight_shape: &[usize],
    scale_shape: &[usize],
    scale_key: &str,
    block: usize,
) -> Result<()> {
    if element_count(scale_shape, scale_key)? == 1 {
        return Ok(());
    }
    let [rows, cols] = weight_shape else {
        return Err(InferError::Dimension(format!(
            "scale FP8 {scale_key} matriciel pour poids non rang 2: {weight_shape:?}"
        )));
    };
    let expected = [
        div_ceil_checked(*rows, block, scale_key)?,
        div_ceil_checked(*cols, block, scale_key)?,
    ];
    if scale_shape != expected {
        return Err(InferError::Dimension(format!(
            "scale FP8 {scale_key} attendu {:?} ou scalaire, reçu {:?}",
            expected, scale_shape
        )));
    }
    Ok(())
}

fn element_count(shape: &[usize], name: &str) -> Result<usize> {
    if shape.is_empty() {
        return Ok(1);
    }
    shape.iter().try_fold(1_usize, |acc, dim| {
        acc.checked_mul(*dim)
            .ok_or_else(|| InferError::Shape(format!("shape trop grande pour {name}")))
    })
}

fn div_ceil_checked(value: usize, divisor: usize, name: &str) -> Result<usize> {
    if divisor == 0 {
        return Err(InferError::Config(format!("diviseur nul pour {name}")));
    }
    value
        .checked_add(divisor - 1)
        .map(|sum| sum / divisor)
        .ok_or_else(|| InferError::Shape(format!("ceil_div trop grand pour {name}")))
}

fn expect_entry_shape(name: &str, entry: &ShardTensorEntry, expected: &[usize]) -> Result<()> {
    if entry.shape != expected {
        return Err(InferError::Dimension(format!(
            "{name} attendu {:?}, reçu {:?}",
            expected, entry.shape
        )));
    }
    Ok(())
}

fn read_entry_bytes(shard: &ShardHeader, entry: &ShardTensorEntry) -> Result<Vec<u8>> {
    let len = entry.data_offsets[1] - entry.data_offsets[0];
    let offset = shard
        .data_start
        .checked_add(entry.data_offsets[0] as u64)
        .ok_or_else(|| InferError::Shape("offset safetensors trop grand".to_string()))?;
    let mut file = std::fs::File::open(&shard.path).map_err(|source| InferError::Io {
        path: shard.path.clone(),
        source,
    })?;
    file.seek(SeekFrom::Start(offset))
        .map_err(|source| InferError::Io {
            path: shard.path.clone(),
            source,
        })?;
    let mut bytes = vec![0_u8; len];
    file.read_exact(&mut bytes)
        .map_err(|source| InferError::Io {
            path: shard.path.clone(),
            source,
        })?;
    Ok(bytes)
}

pub(super) fn tensor_from_entry(
    config: &ModelConfig,
    spec: &TensorSpec,
    entry_ref: &TensorEntryRef,
    headers: &[ShardHeader],
    entries: &HashMap<String, TensorEntryRef>,
) -> Result<DecoderTensor> {
    let entry = &entry_ref.entry;
    let shard = headers
        .get(entry_ref.shard_index)
        .ok_or_else(|| InferError::Shape("index shard invalide".to_string()))?;
    if entry.dtype == Dtype::U32 && spec.source.ends_with(".weight") {
        return quantized_tensor_from_entry(config, spec, entry, shard, headers, entries);
    }
    let bytes = read_entry_bytes(shard, entry)?;
    let mut tensor = tensor_from_safetensor_parts(&spec.source, entry.dtype, &entry.shape, &bytes)?;
    if is_fp8_weight(entry.dtype, &spec.source) {
        tensor = apply_fp8_weight_scale_inv(spec, tensor, headers, entries)?;
    }
    Ok(DecoderTensor::Dense(tensor))
}

pub(super) fn quantized_tensor_from_entry(
    config: &ModelConfig,
    spec: &TensorSpec,
    entry: &ShardTensorEntry,
    shard: &ShardHeader,
    headers: &[ShardHeader],
    entries: &HashMap<String, TensorEntryRef>,
) -> Result<DecoderTensor> {
    let quant = config.quantization.as_ref().ok_or_else(|| {
        InferError::Config(format!(
            "poids quantifié {} sans quantization_config",
            spec.source
        ))
    })?;
    let (group_size, bits) = quant_params_for(quant, &spec.source)?;
    let scales_key = replace_weight_suffix(&spec.source, ".scales")?;
    let biases_key = replace_weight_suffix(&spec.source, ".biases")?;
    let scales = tensor_from_named_entry(headers, entries, &scales_key)?;
    let biases = tensor_from_named_entry(headers, entries, &biases_key)?;
    let bytes = read_entry_bytes(shard, entry)?;
    let packed = bytes_to_u32(&bytes, &spec.source)?;
    if is_moe_expert_weight(&spec.target) && entry.shape.len() == 3 {
        return quantized_expert_weights_from_parts(
            &entry.shape,
            packed,
            scales,
            biases,
            group_size,
            bits,
        );
    }
    let weight =
        AffineQuantizedTensor::new(&entry.shape, packed, scales, biases, group_size, bits)?;
    Ok(DecoderTensor::LinearWeight(LinearWeight::AffineQuantized(
        weight,
    )))
}

fn is_moe_expert_weight(target: &str) -> bool {
    target.contains(".mlp.switch_mlp.") && target.ends_with(".weight")
}

fn quantized_expert_weights_from_parts(
    packed_shape: &[usize],
    packed: Vec<u32>,
    scales: Tensor,
    biases: Tensor,
    group_size: usize,
    bits: usize,
) -> Result<DecoderTensor> {
    let [experts, rows, packed_cols] = packed_shape else {
        return Err(InferError::Dimension(format!(
            "poids expert quantifié attendu rang 3, reçu {packed_shape:?}"
        )));
    };
    if scales.shape().len() != 3 || biases.shape().len() != 3 {
        return Err(InferError::Dimension(format!(
            "scales/biases experts attendus rang 3, reçu scales={:?}, biases={:?}",
            scales.shape(),
            biases.shape()
        )));
    }
    let cols = packed_cols
        .checked_mul(32)
        .and_then(|value| value.checked_div(bits))
        .ok_or_else(|| InferError::Shape("poids expert quantifié trop large".to_string()))?;
    if cols % group_size != 0 {
        return Err(InferError::Shape(format!(
            "expert cols={cols} non divisible par group_size={group_size}"
        )));
    }
    let groups = cols / group_size;
    if scales.shape() != [*experts, *rows, groups] || biases.shape() != [*experts, *rows, groups] {
        return Err(InferError::Dimension(format!(
            "scales/biases experts attendus [{experts},{rows},{groups}], reçu scales={:?}, biases={:?}",
            scales.shape(),
            biases.shape()
        )));
    }
    let packed_stride = rows
        .checked_mul(*packed_cols)
        .ok_or_else(|| InferError::Shape("stride expert packed trop grand".to_string()))?;
    let affine_stride = rows
        .checked_mul(groups)
        .ok_or_else(|| InferError::Shape("stride expert affine trop grand".to_string()))?;
    let mut weights = Vec::with_capacity(*experts);
    for expert in 0..*experts {
        let packed_start = expert
            .checked_mul(packed_stride)
            .ok_or_else(|| InferError::Shape("offset expert packed trop grand".to_string()))?;
        let affine_start = expert
            .checked_mul(affine_stride)
            .ok_or_else(|| InferError::Shape("offset expert affine trop grand".to_string()))?;
        let packed_slice = packed
            .get(packed_start..packed_start + packed_stride)
            .ok_or_else(|| InferError::Shape(format!("slice packed expert {expert} invalide")))?
            .to_vec();
        let scales_slice = scales
            .data()
            .get(affine_start..affine_start + affine_stride)
            .ok_or_else(|| InferError::Shape(format!("slice scales expert {expert} invalide")))?
            .to_vec();
        let biases_slice = biases
            .data()
            .get(affine_start..affine_start + affine_stride)
            .ok_or_else(|| InferError::Shape(format!("slice biases expert {expert} invalide")))?
            .to_vec();
        let scales = Tensor::from_vec(vec![*rows, groups], scales_slice)
            .map_err(|err| InferError::Shape(format!("scales expert {expert} invalides: {err}")))?;
        let biases = Tensor::from_vec(vec![*rows, groups], biases_slice)
            .map_err(|err| InferError::Shape(format!("biases expert {expert} invalides: {err}")))?;
        let weight = AffineQuantizedTensor::new(
            &[*rows, *packed_cols],
            packed_slice,
            scales,
            biases,
            group_size,
            bits,
        )?;
        weights.push(LinearWeight::AffineQuantized(weight));
    }
    Ok(DecoderTensor::ExpertLinearWeights {
        shape: vec![*experts, *rows, cols],
        weights,
    })
}

fn tensor_from_named_entry(
    headers: &[ShardHeader],
    entries: &HashMap<String, TensorEntryRef>,
    name: &str,
) -> Result<Tensor> {
    let entry_ref = entries
        .get(name)
        .ok_or_else(|| InferError::MissingWeight(name.to_string()))?;
    let shard = headers
        .get(entry_ref.shard_index)
        .ok_or_else(|| InferError::Shape("index shard invalide".to_string()))?;
    let entry = &entry_ref.entry;
    let bytes = read_entry_bytes(shard, entry)?;
    tensor_from_safetensor_parts(name, entry.dtype, &entry.shape, &bytes)
}

pub(super) fn is_fp8_weight(dtype: Dtype, source: &str) -> bool {
    matches!(dtype, Dtype::F8_E4M3 | Dtype::F8_E5M2) && source.ends_with(".weight")
}

pub(super) fn apply_fp8_weight_scale_inv(
    spec: &TensorSpec,
    tensor: Tensor,
    headers: &[ShardHeader],
    entries: &HashMap<String, TensorEntryRef>,
) -> Result<Tensor> {
    let scale_key = replace_weight_suffix(&spec.source, ".weight_scale_inv")?;
    let Some(entry_ref) = entries.get(&scale_key) else {
        return Ok(tensor);
    };
    let shard = headers
        .get(entry_ref.shard_index)
        .ok_or_else(|| InferError::Shape("index shard invalide".to_string()))?;
    let entry = &entry_ref.entry;
    let bytes = read_entry_bytes(shard, entry)?;
    let scales = bytes_to_dense_f32(&bytes, entry.dtype, &scale_key)?;
    apply_fp8_scales(tensor, &scales, &entry.shape, &scale_key, FP8_SCALE_BLOCK)
}

pub(super) fn apply_fp8_scales(
    tensor: Tensor,
    scales: &[f32],
    scale_shape: &[usize],
    scale_key: &str,
    block: usize,
) -> Result<Tensor> {
    if scales.len() == 1 {
        let scale = scales
            .first()
            .ok_or_else(|| InferError::Shape(format!("scale FP8 {scale_key} vide")))?;
        return Ok(tensor.map(|value| value * *scale));
    }
    let (rows, cols) = tensor.as_matrix()?;
    validate_fp8_scale_shape(tensor.shape(), scale_shape, scale_key, block)?;
    let [scale_rows, scale_cols] = scale_shape else {
        return Err(InferError::Dimension(format!(
            "scale FP8 {scale_key} attendu rang 2, reçu {scale_shape:?}"
        )));
    };
    if scales.len() != scale_rows * scale_cols {
        return Err(InferError::Shape(format!(
            "scale FP8 {scale_key} shape={scale_shape:?}, éléments={}",
            scales.len()
        )));
    }
    let mut out = tensor.data().to_vec();
    for row in 0..rows {
        let scale_row = row / block;
        for col in 0..cols {
            let scale_col = col / block;
            let scale = scales[scale_row * scale_cols + scale_col];
            out[row * cols + col] *= scale;
        }
    }
    Tensor::from_vec(vec![rows, cols], out)
}

fn replace_weight_suffix(source: &str, suffix: &str) -> Result<String> {
    let base = source.strip_suffix(".weight").ok_or_else(|| {
        InferError::Config(format!("poids quantifié sans suffixe .weight: {source}"))
    })?;
    Ok(format!("{base}{suffix}"))
}
