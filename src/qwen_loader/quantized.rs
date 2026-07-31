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
    if bits == 0 || group_size == 0 {
        return Err(InferError::Shape(format!(
            "paramètres de quantification invalides : bits={bits}, group_size={group_size}"
        )));
    }
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
    context: &DecoderLoadContext<'_>,
) -> Result<DecoderTensor> {
    let entry = &entry_ref.entry;
    let shard = headers
        .get(entry_ref.shard_index)
        .ok_or_else(|| InferError::Shape("index shard invalide".to_string()))?;
    if entry.dtype == Dtype::U32 && spec.source.ends_with(".weight") {
        return quantized_tensor_from_entry(config, spec, entry, shard, headers, entries, context);
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
    context: &DecoderLoadContext<'_>,
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
    #[cfg(all(target_os = "macos", feature = "metal"))]
    if crate::runtime_flags::single_copy_weights_enabled() {
        if let Some(metal) = context.metal {
            let scales_entry = named_entry(headers, entries, &scales_key)?;
            let biases_entry = named_entry(headers, entries, &biases_key)?;
            if scales_entry.entry.dtype == Dtype::BF16 && biases_entry.entry.dtype == Dtype::BF16 {
                let scales = bf16_entry_to_metal(headers, scales_entry, &scales_key, metal)?;
                let biases = bf16_entry_to_metal(headers, biases_entry, &biases_key, metal)?;
                let bytes = read_entry_bytes(shard, entry)?;
                let packed = bytes_to_u32(&bytes, &spec.source)?;
                let packed_len = packed.len();
                let packed_buffer = metal.weight_buffer_from_u32(&packed)?;
                if is_moe_expert_weight(&spec.target) && entry.shape.len() == 3 {
                    return quantized_expert_weights_from_metal_views(
                        &entry.shape,
                        packed_buffer,
                        packed_len,
                        scales,
                        biases,
                        group_size,
                        bits,
                    );
                }
                let weight = AffineQuantizedTensor::new_metal_shared_bf16(
                    &entry.shape,
                    packed_buffer,
                    0,
                    packed_len,
                    &scales.shape,
                    scales.buffer,
                    0,
                    scales.len,
                    &biases.shape,
                    biases.buffer,
                    0,
                    biases.len,
                    group_size,
                    bits,
                )?;
                return Ok(DecoderTensor::LinearWeight(LinearWeight::AffineQuantized(
                    weight,
                )));
            }
        }
    }
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
            context,
        );
    }
    #[cfg(all(target_os = "macos", feature = "metal"))]
    if crate::runtime_flags::single_copy_weights_enabled() {
        if let Some(metal) = context.metal {
            let len_u32 = packed.len();
            let buffer = metal.weight_buffer_from_u32(&packed)?;
            let weight = AffineQuantizedTensor::new_metal_shared(
                &entry.shape,
                buffer,
                0,
                len_u32,
                scales,
                biases,
                group_size,
                bits,
            )?;
            return Ok(DecoderTensor::LinearWeight(LinearWeight::AffineQuantized(
                weight,
            )));
        }
    }
    let weight =
        AffineQuantizedTensor::new(&entry.shape, packed, scales, biases, group_size, bits)?;
    Ok(DecoderTensor::LinearWeight(LinearWeight::AffineQuantized(
        weight,
    )))
}

#[cfg(all(target_os = "macos", feature = "metal"))]
struct MetalBf16Entry {
    buffer: metal::Buffer,
    shape: Vec<usize>,
    len: usize,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn named_entry<'a>(
    headers: &[ShardHeader],
    entries: &'a HashMap<String, TensorEntryRef>,
    name: &str,
) -> Result<&'a TensorEntryRef> {
    let entry_ref = entries
        .get(name)
        .ok_or_else(|| InferError::MissingWeight(name.to_string()))?;
    if headers.get(entry_ref.shard_index).is_none() {
        return Err(InferError::Shape("index shard invalide".to_string()));
    }
    Ok(entry_ref)
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn bf16_entry_to_metal(
    headers: &[ShardHeader],
    entry_ref: &TensorEntryRef,
    name: &str,
    metal: &crate::MetalExecutor,
) -> Result<MetalBf16Entry> {
    let shard = headers
        .get(entry_ref.shard_index)
        .ok_or_else(|| InferError::Shape("index shard invalide".to_string()))?;
    let entry = &entry_ref.entry;
    if entry.dtype != Dtype::BF16 {
        return Err(InferError::UnsupportedDtype {
            name: name.to_string(),
            dtype: entry.dtype,
        });
    }
    let len = element_count(&entry.shape, name)?;
    let expected_bytes = len
        .checked_mul(std::mem::size_of::<u16>())
        .ok_or_else(|| InferError::Shape(format!("payload bf16 trop grand pour {name}")))?;
    let payload_bytes = entry.data_offsets[1]
        .checked_sub(entry.data_offsets[0])
        .ok_or_else(|| InferError::Shape(format!("offsets bf16 inversés pour {name}")))?;
    if payload_bytes != expected_bytes {
        return Err(InferError::Shape(format!(
            "tensor {name} BF16 shape={:?} attend {expected_bytes} octets, reçu {}",
            entry.shape, payload_bytes
        )));
    }
    let file_offset = shard
        .data_start
        .checked_add(entry.data_offsets[0] as u64)
        .ok_or_else(|| InferError::Shape(format!("offset bf16 trop grand pour {name}")))?;
    Ok(MetalBf16Entry {
        buffer: metal.weight_buffer_from_bf16_file(&shard.path, file_offset, expected_bytes)?,
        shape: entry.shape.clone(),
        len,
    })
}

fn is_moe_expert_weight(target: &str) -> bool {
    (target.contains(".mlp.switch_mlp.") || target.contains(".experts.switch_glu."))
        && target.ends_with(".weight")
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn quantized_expert_weights_from_metal_views(
    packed_shape: &[usize],
    packed_buffer: metal::Buffer,
    packed_len: usize,
    scales: MetalBf16Entry,
    biases: MetalBf16Entry,
    group_size: usize,
    bits: usize,
) -> Result<DecoderTensor> {
    let [experts, rows, packed_cols] = packed_shape else {
        return Err(InferError::Dimension(format!(
            "poids expert quantifié attendu rang 3, reçu {packed_shape:?}"
        )));
    };
    if group_size == 0 || bits == 0 {
        return Err(InferError::Config(format!(
            "quantification expert invalide: group_size={group_size}, bits={bits}"
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
    let expected_affine_shape = [*experts, *rows, groups];
    if scales.shape != expected_affine_shape || biases.shape != expected_affine_shape {
        return Err(InferError::Dimension(format!(
            "scales/biases experts attendus {expected_affine_shape:?}, reçu scales={:?}, biases={:?}",
            scales.shape, biases.shape
        )));
    }
    let packed_stride = rows
        .checked_mul(*packed_cols)
        .ok_or_else(|| InferError::Shape("stride expert packed trop grand".to_string()))?;
    let affine_stride = rows
        .checked_mul(groups)
        .ok_or_else(|| InferError::Shape("stride expert affine trop grand".to_string()))?;
    let expected_packed_len = experts
        .checked_mul(packed_stride)
        .ok_or_else(|| InferError::Shape("stack packed expert trop grand".to_string()))?;
    if packed_len != expected_packed_len {
        return Err(InferError::Shape(format!(
            "stack packed expert attend {expected_packed_len} u32, reçu {packed_len}"
        )));
    }
    let affine_shape = [*rows, groups];
    let mut weights = Vec::with_capacity(*experts);
    for expert in 0..*experts {
        let packed_offset = expert
            .checked_mul(packed_stride)
            .ok_or_else(|| InferError::Shape("offset expert packed trop grand".to_string()))?;
        let affine_offset = expert
            .checked_mul(affine_stride)
            .ok_or_else(|| InferError::Shape("offset expert affine trop grand".to_string()))?;
        let weight = AffineQuantizedTensor::new_metal_shared_bf16(
            &[*rows, *packed_cols],
            packed_buffer.clone(),
            packed_offset,
            packed_stride,
            &affine_shape,
            scales.buffer.clone(),
            affine_offset,
            affine_stride,
            &affine_shape,
            biases.buffer.clone(),
            affine_offset,
            affine_stride,
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

fn quantized_expert_weights_from_parts(
    packed_shape: &[usize],
    packed: Vec<u32>,
    scales: Tensor,
    biases: Tensor,
    group_size: usize,
    bits: usize,
    context: &DecoderLoadContext<'_>,
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
    if group_size == 0 {
        return Err(InferError::Shape(
            "group_size de quantification expert nul".to_string(),
        ));
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
    #[cfg(all(target_os = "macos", feature = "metal"))]
    let shared_buffer = if crate::runtime_flags::single_copy_weights_enabled() {
        context
            .metal
            .map(|metal| metal.weight_buffer_from_u32(&packed))
            .transpose()?
    } else {
        None
    };
    let mut weights = Vec::with_capacity(*experts);
    for expert in 0..*experts {
        let packed_start = expert
            .checked_mul(packed_stride)
            .ok_or_else(|| InferError::Shape("offset expert packed trop grand".to_string()))?;
        let affine_start = expert
            .checked_mul(affine_stride)
            .ok_or_else(|| InferError::Shape("offset expert affine trop grand".to_string()))?;
        let packed_range = packed_start..packed_start + packed_stride;
        let packed_slice = packed
            .get(packed_range.clone())
            .ok_or_else(|| InferError::Shape(format!("slice packed expert {expert} invalide")))?;
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
        #[cfg(all(target_os = "macos", feature = "metal"))]
        let weight = match &shared_buffer {
            Some(buffer) => AffineQuantizedTensor::new_metal_shared(
                &[*rows, *packed_cols],
                buffer.clone(),
                packed_start,
                packed_stride,
                scales,
                biases,
                group_size,
                bits,
            )?,
            None => AffineQuantizedTensor::new(
                &[*rows, *packed_cols],
                packed_slice.to_vec(),
                scales,
                biases,
                group_size,
                bits,
            )?,
        };
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        let weight = AffineQuantizedTensor::new(
            &[*rows, *packed_cols],
            packed_slice.to_vec(),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unpacked_cols_rejects_zero_bits_or_group_size() {
        // Régression : un override de quantification malformé (bits=0 ou
        // group_size=0) doit renvoyer une erreur, jamais paniquer (modulo par zéro).
        assert!(unpacked_cols(64, 32, 0).is_err());
        assert!(unpacked_cols(64, 0, 4).is_err());
        // Cas nominal : 64 packed_cols × 32 / 4 bits = 512 cols, divisible par 32.
        assert_eq!(unpacked_cols(64, 32, 4).expect("config quant valide"), 512);
    }
}
