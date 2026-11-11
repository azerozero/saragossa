use super::*;

fn load_bits(bits: &[u16], context: &DecoderLoadContext<'_>) -> DecoderTensor {
    let bytes: Vec<u8> = bits.iter().flat_map(|value| value.to_le_bytes()).collect();
    let view = safetensors::tensor::TensorView::new(Dtype::BF16, vec![2, bits.len() / 2], &bytes)
        .expect("invariant: fixture bf16 paire");
    let file = tempfile::NamedTempFile::new().expect("invariant: fichier temporaire");
    let name = "lm_head.weight";
    std::fs::write(
        file.path(),
        safetensors::serialize([(name, view)], None).expect("invariant: fixture sérialisable"),
    )
    .expect("invariant: écriture fixture");
    let headers =
        read_shard_headers(&[file.path().to_path_buf()]).expect("invariant: entête valide");
    let entries = index_shard_entries(&headers).expect("invariant: index valide");
    let spec = TensorSpec {
        source: name.to_string(),
        target: name.to_string(),
        required: true,
    };
    tensor_from_entry(
        &test_fixtures::test_config(),
        &spec,
        &entries[name],
        &headers,
        &entries,
        context,
    )
    .expect("invariant: fixture chargeable")
}

#[test]
fn dense_bf16_cpu_fallback_preserves_file_bits() {
    let bits: Vec<u16> = (0..=u16::MAX).collect();
    let DecoderTensor::Dense(tensor) = load_bits(&bits, &DecoderLoadContext::cpu()) else {
        panic!("le contexte CPU doit garder le stockage f32");
    };
    let actual: Vec<u32> = tensor.data().iter().map(|value| value.to_bits()).collect();
    let expected: Vec<u32> = bits.iter().map(|value| u32::from(*value) << 16).collect();
    assert_eq!(actual, expected);
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
#[ignore = "requiert un device Metal, RETI_RUST_SINGLE_COPY_WEIGHTS=1"]
fn dense_bf16_file_storage_has_no_f32_copy() -> Result<()> {
    let metal = crate::MetalExecutor::new()?;
    let bits: Vec<u16> = (0..=u16::MAX).collect();
    let DecoderTensor::LinearWeight(LinearWeight::DenseBf16(weight)) =
        load_bits(&bits, &DecoderLoadContext::metal(&metal))
    else {
        panic!("stockage dense bf16 partagé attendu");
    };
    assert_eq!(weight.data(), bits);
    assert_eq!(weight.buffer().length(), (bits.len() * 2) as u64);
    assert_eq!(
        weight.clone().buffer().contents(),
        weight.buffer().contents()
    );
    let resolved = metal
        .resolve_linear_weight_buffers(&LinearWeight::DenseBf16(weight.clone()), "test_bf16")?;
    let crate::metal_backend::MetalLinearWeightBuffers::DenseBf16 { rhs, .. } = resolved else {
        panic!("le dispatch résident doit conserver le stockage bf16");
    };
    assert_eq!(rhs.contents(), weight.buffer().contents());
    // La tête liée partage aussi le même payload, sans clone de matrice.
    let mut tensors = HashMap::from([(
        EMBED_TOKENS_WEIGHT.to_string(),
        DecoderTensor::LinearWeight(LinearWeight::DenseBf16(weight.clone())),
    )]);
    tie_lm_head_to_embeddings(&test_fixtures::tied_test_config(), &mut tensors);
    let DecoderTensor::LinearWeight(LinearWeight::DenseBf16(head)) = &tensors[LM_HEAD_WEIGHT]
    else {
        panic!("tête liée bf16 attendue");
    };
    assert_eq!(head.buffer().contents(), weight.buffer().contents());
    Ok(())
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
#[ignore = "requiert un device Metal, RETI_RUST_SINGLE_COPY_WEIGHTS=0"]
fn dense_bf16_kill_switch_keeps_f32() -> Result<()> {
    let metal = crate::MetalExecutor::new()?;
    let DecoderTensor::Dense(tensor) =
        load_bits(&[0x3f80, 0xc000], &DecoderLoadContext::metal(&metal))
    else {
        panic!("le kill-switch doit garder le stockage f32");
    };
    assert_eq!(tensor.data(), &[1.0, -2.0]);
    Ok(())
}
