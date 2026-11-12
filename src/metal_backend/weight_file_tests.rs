//! Oracles de copie sur de petits buffers partagés, sans calcul GPU.

use crate::Result;

#[expect(
    unsafe_code,
    reason = "lecture des buffers partagés immuables des fixtures"
)]
pub(crate) fn buffer_bytes(buffer: &metal::Buffer) -> Vec<u8> {
    assert_eq!(buffer.storage_mode(), metal::MTLStorageMode::Shared);
    // SAFETY: buffer partagé vivant, initialisé, sans commande GPU ni écrivain ;
    // la tranche u8 est alignée et bornée par la longueur allouée par Metal.
    unsafe { std::slice::from_raw_parts(buffer.contents().cast(), buffer.length() as usize) }
        .to_vec()
}

#[test]
fn direct_file_buffers_match_copies_and_reject_short_reads() -> Result<()> {
    let Some(metal) = crate::test_support::metal_executor()? else {
        return Ok(());
    };
    let codes = [0x7654_3210_u32, 0xfedc_ba98, 0, u32::MAX];
    let bytes: Vec<_> = codes.iter().flat_map(|v| v.to_le_bytes()).collect();
    let file = tempfile::NamedTempFile::new().expect("invariant: fichier temporaire");
    let mut payload = vec![0xaa; 13];
    payload.extend_from_slice(&bytes);
    std::fs::write(file.path(), payload).expect("invariant: fixture écrite");
    let direct = metal.weight_buffer_from_u32_file(file.path(), 13, bytes.len())?;
    let copied = metal.weight_buffer_from_u32(&codes)?;
    assert_eq!(buffer_bytes(&direct), buffer_bytes(&copied));
    let direct = metal.weight_buffer_from_bf16_file(file.path(), 13, bytes.len())?;
    let copied = metal.weight_buffer_from_bf16_bytes(&bytes)?;
    assert_eq!(buffer_bytes(&direct), buffer_bytes(&copied));
    assert!(metal
        .weight_buffer_from_u32_file(file.path(), 13, 3)
        .is_err());
    assert!(metal
        .weight_buffer_from_u32_file(file.path(), 13, 0)
        .is_err());
    assert!(metal
        .weight_buffer_from_bf16_file(file.path(), 13, 3)
        .is_err());
    assert!(metal
        .weight_buffer_from_u32_file(file.path(), 13, 20)
        .is_err());
    assert!(metal
        .weight_buffer_from_u32_file(file.path(), u64::MAX, 4)
        .is_err());
    for len in [8_usize * 1024 * 1024 * 1024, usize::MAX - 3] {
        assert!(matches!(
            metal.weight_buffer_from_u32_file(file.path(), 0, len),
            Err(crate::InferError::Shape(_))
        ));
        assert!(matches!(
            metal.weight_buffer_from_bf16_file(file.path(), 0, len),
            Err(crate::InferError::Shape(_))
        ));
    }
    Ok(())
}

#[test]
fn direct_file_shared_f32_preserves_dense_dispatch_bits() -> Result<()> {
    use super::{commit_and_wait, read_f32_buffer, MetalLinearWeightBuffers};
    use crate::{ForwardRuntime, Linear, LinearWeight, Tensor};
    let Some(metal) = crate::test_support::metal_executor()? else {
        return Ok(());
    };
    for (rows, cols) in [(3, 35), (8, 512), (33, 1024)] {
        let values: Vec<_> = (0..rows * cols)
            .map(|i| ((i % 23) as f32 - 11.0) / 7.0)
            .collect();
        let bytes: Vec<_> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let file = tempfile::NamedTempFile::new().expect("invariant: fichier temporaire");
        std::fs::write(file.path(), &bytes).expect("invariant: fixture écrite");
        let buffer = metal.weight_buffer_dense_f32_file(
            file.as_file(),
            0,
            bytes.len(),
            safetensors::Dtype::F32,
            "fixture",
        )?;
        assert_eq!(buffer_bytes(&buffer), bytes);
        let dense = Tensor::from_vec(vec![rows, cols], values)?;
        let shared = crate::linear::DenseSharedTensor::new(&[rows, cols], buffer)?;
        let weights = [
            LinearWeight::Dense(dense.clone()),
            LinearWeight::DenseShared(shared.clone()),
        ];
        let resolved = weights
            .iter()
            .map(|w| metal.resolve_linear_weight_buffers(w, "fixture"))
            .collect::<Result<Vec<_>>>()?;
        let MetalLinearWeightBuffers::Dense { rhs, .. } = &resolved[1] else {
            panic!("dispatch f32 attendu");
        };
        assert_eq!(rhs.contents(), shared.buffer().contents());
        let bits = |data: &[f32]| data.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        for batch in [1, 3, 17] {
            let input = Tensor::from_vec(
                vec![batch, cols],
                (0..batch * cols)
                    .map(|i| ((i % 19) as f32 - 9.0) / 11.0)
                    .collect(),
            )?;
            let linear = Linear::from_weight(weights[1].clone(), None)?;
            assert_eq!(
                bits(
                    linear
                        .forward_with_runtime(&input, ForwardRuntime::cpu())?
                        .data()
                ),
                bits(input.matmul_rhs_t(&dense)?.data())
            );
            assert_eq!(
                bits(
                    metal
                        .matmul_rhs_t_shared_f32(&input, &shared, false)?
                        .data()
                ),
                bits(metal.matmul_rhs_t_dense(&input, &dense)?.data())
            );
            if batch > 1 {
                assert_eq!(
                    bits(metal.matmul_rhs_t_shared_f32(&input, &shared, true)?.data()),
                    bits(metal.matmul_rhs_t_dense_tiled(&input, &dense)?.data())
                );
            }
            let lhs = metal.buffer_from_f32(input.data(), "fixture")?;
            let out = metal.uncached_f32_buffer(batch * rows, "fixture")?;
            for resident in [false, true] {
                let mut outputs = Vec::new();
                for index in 0..2 {
                    let cb = metal.queue.new_command_buffer();
                    let encoder = cb.new_compute_command_encoder();
                    if resident {
                        metal.encode_matmul_weight_buffers(
                            encoder,
                            &lhs,
                            batch,
                            cols,
                            &resolved[index],
                            &out,
                            false,
                        )?;
                    } else {
                        metal.encode_matmul_weight(
                            encoder,
                            &mut Vec::new(),
                            &lhs,
                            batch,
                            cols,
                            &weights[index],
                            &out,
                        )?;
                    }
                    encoder.end_encoding();
                    commit_and_wait(cb)?;
                    outputs.push(bits(&read_f32_buffer(&out, batch * rows)?));
                }
                assert_eq!(outputs[0], outputs[1]);
            }
        }
        let concat =
            metal.resolve_concat_linear_weight_buffers(&[&weights[1], &weights[0]], "fixture")?;
        let MetalLinearWeightBuffers::Dense { rhs, .. } = concat else {
            panic!("concat f32 attendue");
        };
        assert_eq!(buffer_bytes(&rhs), bytes.repeat(2));
    }
    Ok(())
}
