// Les deux poids ont les mêmes octets, jamais deux quantifications distinctes.
// Les sentinelles changent entre deux essais : une lecture du préfixe ou des
// blocs voisins qui contribue au résultat doit rompre l'égalité bit à bit.
fn offset_weight(
    metal: &MetalExecutor,
    weight: &AffineQuantizedTensor,
    offsets: [usize; 3],
    sentinel: u16,
) -> Result<AffineQuantizedTensor> {
    let data = [
        bytemuck::cast_slice::<u32, u16>(weight.packed_data()).to_vec(),
        super::buffers::f32_slice_to_bf16(weight.scales_f32().as_ref()),
        super::buffers::f32_slice_to_bf16(weight.biases_f32().as_ref()),
    ];
    let mut buffers = Vec::new();
    for (data, offset) in data.iter().zip(offsets) {
        let mut guarded = vec![sentinel; offset / 2];
        guarded.extend_from_slice(data);
        guarded.extend_from_slice(&[sentinel; 256]);
        buffers.push(metal.buffer_from_slice(&guarded, "offset_guarded_weight")?);
    }
    let mut alias = weight.clone();
    alias.set_packed_metal_view(buffers[0].clone(), offsets[0] / 4)?;
    alias.set_affine_metal_views(
        buffers[1].clone(),
        offsets[1] / 2,
        buffers[2].clone(),
        offsets[2] / 2,
    )?;
    Ok(alias)
}

struct OffsetKernelGuard;

impl Drop for OffsetKernelGuard {
    fn drop(&mut self) {
        super::matmul_select::OFFSET_TEST_KERNEL.with(|value| value.set(None));
        super::profiling::OFFSET_TEST_DISPATCHES.with(|value| *value.borrow_mut() = None);
    }
}

fn offset_dispatch(
    metal: &MetalExecutor,
    weight: &AffineQuantizedTensor,
    batch: usize,
    resident: bool,
) -> Result<(Vec<u32>, Vec<String>)> {
    let (out_dim, in_dim) = (weight.shape()[0], weight.shape()[1]);
    let lhs = metal.upload_f32_buffer(
        &(0..batch * in_dim)
            .map(|i| ((i % 29) as f32 - 14.0) * 0.03125)
            .collect::<Vec<_>>(),
        "offset_lhs",
    )?;
    let count = batch * out_dim;
    let output = metal.upload_f32_buffer(&vec![-12345.0; count + 32], "offset_output")?;
    let weight = LinearWeight::AffineQuantized(weight.clone());
    let mut owned = Vec::new();
    let cb = metal.queue.new_command_buffer();
    let encoder = cb.new_compute_command_encoder();
    let guard = EncoderEndGuard::new(encoder);
    super::profiling::OFFSET_TEST_DISPATCHES.with(|value| *value.borrow_mut() = Some(Vec::new()));
    if resident {
        let buffers = metal.resolve_linear_weight_buffers(&weight, "offset_resident")?;
        metal
            .encode_matmul_weight_buffers(encoder, &lhs, batch, in_dim, &buffers, &output, false)?;
    } else {
        metal.encode_matmul_weight(encoder, &mut owned, &lhs, batch, in_dim, &weight, &output)?;
    }
    guard.end();
    commit_and_wait(cb)?;
    let trace = super::profiling::OFFSET_TEST_DISPATCHES
        .with(|value| value.borrow_mut().take())
        .expect("invariant: capture active");
    let mut bits = super::byte_helpers::read_u32_buffer(&output, count + 32)?;
    assert!(bits[count..].iter().all(|v| *v == (-12345.0_f32).to_bits()));
    bits.truncate(count);
    assert!(bits
        .iter()
        .all(|v| f32::from_bits(*v).is_finite() && *v != (-12345.0_f32).to_bits()));
    Ok((bits, trace))
}

#[test]
fn affine_offsets_alignment_without_device() {
    use AffineMatmulKernel::*;
    for kernel in [
        Qmm2,
        QmvT,
        Skinny,
        Qmm8,
        QmmNaFusedTiledU4,
        QmmNaFusedTiledU4Align64,
        FastQmvU4,
        FastQmvU4Align64,
        FastQmvU2,
        FastQmvU3,
        FastQmvU6,
        FastQmvOneU8,
        QmmNaFusedTiledU8,
        QmmNaU8Gs128,
        QmmNaU8,
        FastQmvU8,
        FastQmvU8Align64,
    ] {
        assert_eq!(kernel.with_offsets(0, 0, 0), kernel);
        assert_eq!(kernel.with_offsets(64, 128, 192), kernel);
        assert_eq!(
            kernel.with_offsets(10240 * 2560, 10240 * 320, 10240 * 320),
            kernel
        );
        assert_eq!(kernel.with_offsets(1, 0, 0), Fallback);
        assert_eq!(kernel.with_offsets(0, 1, 0), Fallback);
        assert_eq!(kernel.with_offsets(0, 0, 1), Fallback);
        let vector = matches!(kernel, QmmNaFusedTiledU4 | QmmNaFusedTiledU4Align64);
        assert_eq!(
            kernel.with_offsets(4, 2, 6),
            if vector { Fallback } else { kernel }
        );
        assert_eq!(kernel.with_offsets(16, 2, 6), kernel);
    }
}

#[test]
#[ignore = "device Metal avec pipelines NA requis ; aucun modèle"]
fn affine_offsets_fast_kernels_bit_exact() -> Result<()> {
    use AffineMatmulKernel::*;
    let metal = MetalExecutor::new()?;
    let _reset = OffsetKernelGuard;
    // (kernel, bits, group_size, M, N, K). Tailles minimales, bords N et K
    // compris ; les deux branches bf16/x3 du tuilé u4 sont distinctes.
    for (kernel, bits, gs, m, n, k) in [
        (Qmm2, 4, 64, 2, 64, 512),
        (Qmm2, 3, 64, 2, 64, 512),
        (Qmm2, 8, 64, 2, 64, 512),
        (Qmm2, 8, 128, 2, 64, 512),
        (Qmm8, 4, 64, 7, 64, 512),
        (QmvT, 4, 64, 3, 64, 512),
        (QmvT, 4, 64, 3, 64, 8704),
        (QmvT, 4, 64, 3, 65536, 512),
        (QmvT, 4, 64, 16, 64, 512),
        (Skinny, 4, 64, 7, 64, 512),
        (QmmNaFusedTiledU4, 4, 64, 17, 2048, 2048),
        (QmmNaFusedTiledU4, 4, 64, 17, 64, 512),
        (QmmNaFusedTiledU4Align64, 4, 64, 17, 64, 576),
        (FastQmvU4, 4, 64, 1, 64, 512),
        (FastQmvU4, 4, 64, 1, 13, 512),
        (FastQmvU4Align64, 4, 64, 1, 13, 576),
        (FastQmvU2, 2, 64, 1, 8, 256),
        (FastQmvU2, 2, 64, 1, 64, 512),
        (FastQmvU3, 3, 64, 1, 64, 512),
        (FastQmvU3, 3, 64, 1, 16, 768),
        (FastQmvU6, 6, 64, 1, 13, 512),
        (FastQmvU6, 6, 64, 1, 64, 512),
        (FastQmvOneU8, 8, 64, 1, 1, 512),
        (FastQmvU8, 8, 64, 1, 64, 512),
        (FastQmvU8, 8, 128, 1, 64, 512),
        (FastQmvU8, 8, 128, 1, 24, 512),
        (FastQmvU8, 8, 64, 1, 24, 512),
        (FastQmvU8Align64, 8, 64, 1, 24, 576),
        (QmmNaFusedTiledU8, 8, 64, 17, 64, 512),
        (QmmNaFusedTiledU8, 8, 128, 17, 64, 512),
        (QmmNaU8Gs128, 8, 128, 16, 64, 512),
        (QmmNaU8, 8, 64, 16, 64, 512),
    ] {
        let weight = match bits {
            2 => test_affine_mlx_u2(n, k)?,
            3 => test_affine_mlx_u3(n, k)?,
            4 => test_affine_varied_group(n, k, gs)?,
            6 => test_affine_varied_u6(n, k)?,
            _ => test_affine_varied_u8_group(n, k, gs)?,
        };
        super::matmul_select::OFFSET_TEST_KERNEL.with(|value| value.set(Some(kernel)));
        for resident in [false, true] {
            // QmmNaU8 est un chemin owned uniquement ; le résident garde son
            // repli historique. Le tester comme rapide masquerait une régression.
            if resident && kernel == QmmNaU8 {
                continue;
            }
            let reference = offset_dispatch(&metal, &weight, m, resident)?;
            assert!(!reference.1.is_empty(), "{kernel:?}");
            assert!(
                reference
                    .1
                    .iter()
                    .all(|t| !t.starts_with("affine_matmul_rhs_t")),
                "{kernel:?}"
            );
            for sentinel in [0x7fc1, 0xdead] {
                let shifted = offset_weight(&metal, &weight, [64, 128, 192], sentinel)?;
                let views = [
                    shifted.packed_metal_view(),
                    shifted.scales_metal_view(),
                    shifted.biases_metal_view(),
                ];
                let snapshots = views
                    .iter()
                    .map(|v| {
                        let (b, _) = v.expect("invariant: vue Metal");
                        super::byte_helpers::read_u16_buffer(b, b.length() as usize / 2)
                    })
                    .collect::<Result<Vec<_>>>()?;
                assert_eq!(
                    offset_dispatch(&metal, &shifted, m, resident)?,
                    reference,
                    "{kernel:?} resident={resident} M={m} N={n} K={k}"
                );
                for (view, snapshot) in views.iter().zip(snapshots) {
                    let (b, _) = view.expect("invariant: vue Metal");
                    assert_eq!(
                        super::byte_helpers::read_u16_buffer(b, b.length() as usize / 2)?,
                        snapshot
                    );
                }
            }
            if matches!(kernel, QmmNaFusedTiledU4 | QmmNaFusedTiledU4Align64) {
                // 4 est valide pour le repli uint*, mais pas pour le uint4 NA.
                let misaligned = offset_weight(&metal, &weight, [4, 2, 6], 0x7fc1)?;
                let result = offset_dispatch(&metal, &misaligned, m, resident)?;
                assert_eq!(
                    result.1,
                    vec![format!("affine_matmul_rhs_t_u32_f32 M={m} N={n} K={k}")]
                );
                super::matmul_select::OFFSET_TEST_KERNEL.with(|value| value.set(Some(Fallback)));
                assert_eq!(result, offset_dispatch(&metal, &weight, m, resident)?);
                super::matmul_select::OFFSET_TEST_KERNEL.with(|value| value.set(Some(kernel)));
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "device Metal requis ; sélection réelle sans surcharge"]
fn affine_offsets_natural_selection_and_standalone() -> Result<()> {
    let metal = MetalExecutor::new()?;
    let _reset = OffsetKernelGuard;
    for bits in [2, 3, 4, 6, 8] {
        let weight = match bits {
            2 => test_affine_mlx_u2(64, 512)?,
            3 => test_affine_mlx_u3(64, 512)?,
            4 => test_affine_varied(64, 512)?,
            6 => test_affine_varied_u6(64, 512)?,
            _ => test_affine_varied_u8(64, 512)?,
        };
        let shifted = offset_weight(&metal, &weight, [64, 128, 192], 0x7fc1)?;
        for batch in [1, 2, 3, 8, 17] {
            for resident in [false, true] {
                let reference = offset_dispatch(&metal, &weight, batch, resident)?;
                assert!(reference
                    .1
                    .iter()
                    .all(|t| !t.starts_with("affine_matmul_rhs_t")));
                assert_eq!(
                    offset_dispatch(&metal, &shifted, batch, resident)?,
                    reference
                );
            }
            let input = Tensor::from_vec(
                vec![batch, 512],
                (0..batch * 512)
                    .map(|i| (i % 19) as f32 * 0.03125)
                    .collect(),
            )?;
            let standalone = |weight: &AffineQuantizedTensor| -> Result<(Vec<u32>, Vec<String>)> {
                super::profiling::OFFSET_TEST_DISPATCHES
                    .with(|value| *value.borrow_mut() = Some(Vec::new()));
                let output = metal.matmul_rhs_t_affine(&input, weight)?;
                let trace = super::profiling::OFFSET_TEST_DISPATCHES
                    .with(|value| value.borrow_mut().take())
                    .expect("invariant: capture active");
                assert!(!trace.is_empty());
                assert!(trace.iter().all(|t| !t.starts_with("affine_matmul_rhs_t")));
                Ok((output.data().iter().map(|v| v.to_bits()).collect(), trace))
            };
            assert_eq!(standalone(&weight)?, standalone(&shifted)?);
        }
    }
    // Forme de z suffisamment grande pour le prefill NA nominal (pas x3).
    let weight = test_affine_varied(2048, 5120)?;
    let shifted = offset_weight(&metal, &weight, [64, 128, 192], 0xdead)?;
    for resident in [false, true] {
        // Le sélecteur réserve le GEMM NA aux lots d'au moins une tuile.
        let reference = offset_dispatch(&metal, &weight, 128, resident)?;
        assert!(reference
            .1
            .iter()
            .any(|t| t.starts_with("gemm_nax_coop_qb_tiled_u4 ")));
        assert_eq!(offset_dispatch(&metal, &shifted, 128, resident)?, reference);
    }
    Ok(())
}

#[cfg(feature = "devtools")]
#[test]
#[ignore = "device Metal requis ; chemin résiduel pairwise p2"]
fn affine_offsets_residual_bit_exact() -> Result<()> {
    use crate::runtime_flags::pairwise::{Case, Guard};
    let metal = MetalExecutor::new()?;
    let _p2 = Guard::set(
        Case {
            p2: true,
            ..Case::default()
        },
        false,
    )?;
    let weight = test_affine_varied(64, 512)?;
    let lhs = metal.upload_f32_buffer(&vec![0.125; 512], "offset_residual_lhs")?;
    let residual = metal.upload_f32_buffer(&vec![1.25; 64], "offset_residual")?;
    let run = |weight: &AffineQuantizedTensor| -> Result<Vec<u32>> {
        let buffers = metal.resolve_linear_weight_buffers(
            &LinearWeight::AffineQuantized(weight.clone()),
            "offset_residual_weight",
        )?;
        let output = metal.upload_f32_buffer(&[-12345.0; 96], "offset_residual_output")?;
        let cb = metal.queue.new_command_buffer();
        let encoder = cb.new_compute_command_encoder();
        let guard = EncoderEndGuard::new(encoder);
        assert!(metal.encode_qmv_u4_residual_buffers(
            encoder, &lhs, &buffers, &residual, &output, 512, 64
        )?);
        guard.end();
        commit_and_wait(cb)?;
        let bits = super::byte_helpers::read_u32_buffer(&output, 96)?;
        assert!(bits[64..].iter().all(|v| *v == (-12345.0_f32).to_bits()));
        Ok(bits)
    };
    let reference = run(&weight)?;
    for sentinel in [0x7fc1, 0xdead] {
        let shifted = offset_weight(&metal, &weight, [64, 128, 192], sentinel)?;
        assert_eq!(run(&shifted)?, reference);
    }
    for part in 0..3 {
        let mut buffers = metal.resolve_linear_weight_buffers(
            &LinearWeight::AffineQuantized(weight.clone()),
            "offset_residual_misaligned",
        )?;
        if let MetalLinearWeightBuffers::AffineQuantized {
            packed_offset,
            scales_offset,
            biases_offset,
            ..
        } = &mut buffers
        {
            match part {
                0 => *packed_offset += 2,
                1 => *scales_offset += 1,
                _ => *biases_offset += 1,
            }
        }
        let output = metal.upload_f32_buffer(&[-12345.0; 64], "offset_residual_untouched")?;
        let cb = metal.queue.new_command_buffer();
        let encoder = cb.new_compute_command_encoder();
        let guard = EncoderEndGuard::new(encoder);
        assert!(!metal.encode_qmv_u4_residual_buffers(
            encoder, &lhs, &buffers, &residual, &output, 512, 64
        )?);
        guard.end();
        commit_and_wait(cb)?;
        assert_eq!(read_f32_buffer(&output, 64)?, vec![-12345.0; 64]);
    }
    Ok(())
}
