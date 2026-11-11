//! Comparaison des chargements CPU et Metal sur des safetensors synthétiques.

use super::*;
use crate::metal_backend::weight_file_tests::buffer_bytes;
use safetensors::tensor::TensorView;

fn affine_weights(tensor: DecoderTensor) -> Vec<AffineQuantizedTensor> {
    let weights = match tensor {
        DecoderTensor::LinearWeight(weight) => vec![weight],
        DecoderTensor::ExpertLinearWeights { weights, .. } => weights,
        _ => panic!("poids affine attendu"),
    };
    weights
        .into_iter()
        .map(|weight| match weight {
            LinearWeight::AffineQuantized(weight) => weight,
            _ => panic!("poids affine attendu"),
        })
        .collect()
}

#[test]
fn direct_file_dense_and_moe_match_old_buffers() -> Result<()> {
    let metal = crate::MetalExecutor::new()?;
    let single_copy = crate::runtime_flags::single_copy_weights_enabled();
    for experts in [1, 3] {
        for dtype in [Dtype::BF16, Dtype::F32] {
            for packed_dtype in [Dtype::U32, Dtype::U8] {
                let name = if experts == 1 {
                    "lm_head.weight"
                } else {
                    "layers.0.mlp.switch_mlp.gate_proj.weight"
                };
                let base = name.strip_suffix(".weight").expect("invariant: suffixe");
                let scales_name = format!("{base}.scales");
                let biases_name = format!("{base}.biases");
                let codes: Vec<u32> = (0..experts * 16)
                    .map(|i| (i as u32).wrapping_mul(0x1357_9bdf))
                    .collect();
                let packed: Vec<u8> = codes.iter().flat_map(|v| v.to_le_bytes()).collect();
                let params = |bias: bool| -> Vec<u8> {
                    (0..experts * 2)
                        .flat_map(|i| {
                            let value = if bias {
                                -0.25 - i as f32
                            } else {
                                0.125 + i as f32
                            };
                            if dtype == Dtype::BF16 {
                                crate::quantization::bf16_bits(value).to_le_bytes().to_vec()
                            } else {
                                value.to_le_bytes().to_vec()
                            }
                        })
                        .collect()
                };
                let scales = params(false);
                let biases = params(true);
                let cols = if packed_dtype == Dtype::U32 { 8 } else { 32 };
                let shape = if experts == 1 {
                    vec![2, cols]
                } else {
                    vec![experts, 2, cols]
                };
                let affine_shape = if experts == 1 {
                    vec![2, 1]
                } else {
                    vec![experts, 2, 1]
                };
                let tensors = [
                    (name, TensorView::new(packed_dtype, shape, &packed)),
                    (
                        scales_name.as_str(),
                        TensorView::new(dtype, affine_shape.clone(), &scales),
                    ),
                    (
                        biases_name.as_str(),
                        TensorView::new(dtype, affine_shape, &biases),
                    ),
                ]
                .into_iter()
                .map(|(name, view)| (name, view.expect("invariant: fixture valide")));
                let file = tempfile::NamedTempFile::new().expect("invariant: fichier temporaire");
                std::fs::write(
                    file.path(),
                    safetensors::serialize(tensors, None).expect("invariant: sérialisation"),
                )
                .expect("invariant: fixture écrite");
                let headers = read_shard_headers(&[file.path().to_path_buf()])?;
                let entries = index_shard_entries(&headers)?;
                let spec = TensorSpec {
                    source: name.into(),
                    target: name.into(),
                    required: true,
                };
                let mut config = test_fixtures::test_config();
                config.quantization = Some(
                    serde_json::from_str(r#"{"group_size":64,"bits":4}"#)
                        .expect("invariant: configuration"),
                );
                let load = |context: &DecoderLoadContext<'_>| {
                    tensor_from_entry(&config, &spec, &entries[name], &headers, &entries, context)
                        .map(affine_weights)
                };
                let cpu = load(&DecoderLoadContext::cpu())?;
                let direct = load(&DecoderLoadContext::metal(&metal))?;
                assert_eq!(direct.len(), experts);
                for (expert, (got, old)) in direct.iter().zip(&cpu).enumerate() {
                    assert_eq!(got.packed_data(), old.packed_data());
                    assert_eq!(got.scales_f32(), old.scales_f32());
                    assert_eq!(got.biases_f32(), old.biases_f32());
                    if !single_copy {
                        assert!(got.packed_metal_view().is_none());
                        continue;
                    }
                    let (buffer, offset) =
                        got.packed_metal_view().expect("invariant: buffer direct");
                    assert_eq!(offset, expert * 16 * 4);
                    assert_eq!(
                        buffer_bytes(buffer),
                        buffer_bytes(&metal.weight_buffer_from_u32(&codes)?)
                    );
                    if dtype == Dtype::BF16 {
                        for (view, bytes) in [
                            (got.scales_metal_view(), &scales),
                            (got.biases_metal_view(), &biases),
                        ] {
                            let (buffer, offset) = view.expect("invariant: paramètres natifs");
                            assert_eq!(offset, expert * 2 * 2);
                            assert_eq!(
                                buffer_bytes(buffer),
                                buffer_bytes(&metal.weight_buffer_from_bf16_bytes(bytes)?)
                            );
                        }
                    }
                }
                // Même fichier, header incohérent : refus avant lecture de chaque payload.
                for key in [name, scales_name.as_str(), biases_name.as_str()] {
                    let mut invalid = entries.clone();
                    invalid
                        .get_mut(key)
                        .expect("invariant: clé présente")
                        .entry
                        .data_offsets[1] -= 1;
                    assert!(tensor_from_entry(
                        &config,
                        &spec,
                        &invalid[name],
                        &headers,
                        &invalid,
                        &DecoderLoadContext::metal(&metal)
                    )
                    .is_err());
                }
            }
        }
    }
    Ok(())
}
