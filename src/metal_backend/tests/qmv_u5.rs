fn test_affine_varied_u5(out_dim: usize, in_dim: usize) -> Result<AffineQuantizedTensor> {
    let bits = 5;
    let packed_cols = in_dim * bits / 32;
    let groups = in_dim / 64;
    let mut packed = vec![0_u32; out_dim * packed_cols];
    for row in 0..out_dim {
        for col in 0..in_dim {
            let q = ((row * 11 + col * 7 + (col / 64) * 13) % 32) as u32;
            let bit_offset = col * bits;
            let word_col = bit_offset / 32;
            let shift = bit_offset - word_col * 32;
            let row_word = row * packed_cols + word_col;
            packed[row_word] |= q << shift;
            if shift + bits > 32 && word_col + 1 < packed_cols {
                packed[row_word + 1] |= q >> (32 - shift);
            }
        }
    }
    let scales = Tensor::from_vec(
        vec![out_dim, groups],
        (0..out_dim * groups)
            .map(|i| bf16_round(0.000_7 + 0.000_03 * ((i % 17) as f32)))
            .collect(),
    )?;
    let biases = Tensor::from_vec(
        vec![out_dim, groups],
        (0..out_dim * groups)
            .map(|i| bf16_round(-0.035 + 0.000_4 * ((i % 23) as f32)))
            .collect(),
    )?;
    AffineQuantizedTensor::new(&[out_dim, packed_cols], packed, scales, biases, 64, bits)
}

#[test]
fn affine_qmv_u5_eligibility_without_device() -> Result<()> {
    for (batch, in_dim, out_dim) in [(1, 64, 1), (2, 128, 2), (3, 192, 13), (1, 512, 64)] {
        let weight = test_affine_varied_u5(out_dim, in_dim)?;
        let enabled = fast_affine_qmv_enabled(out_dim);
        assert_eq!(can_use_fast_affine_qmv_u5(batch, in_dim, &weight), enabled);
        assert_eq!(
            can_use_fast_affine_qmv_u5_buffers(batch, in_dim, out_dim, 64, 5),
            enabled
        );
        assert!(!can_use_fast_affine_qmv_u5(batch, in_dim + 64, &weight));
    }
    for (batch, in_dim, group_size, bits) in [
        (0, 64, 64, 5),
        (1, 0, 64, 5),
        (1, 65, 64, 5),
        (1, 64, 32, 5),
        (1, 128, 128, 5),
        (1, 64, 64, 3),
        (1, 64, 64, 4),
        (1, 64, 64, 6),
        (1, 64, 64, 8),
    ] {
        assert!(!can_use_fast_affine_qmv_u5_buffers(
            batch, in_dim, 13, group_size, bits
        ));
    }
    Ok(())
}

#[test]
#[ignore = "device Metal requis : petit oracle sans modèle"]
fn affine_qmv_u5_gs64_dequantizes_like_cpu_bit_exact() -> Result<()> {
    // Un lancement explicite doit échouer sans device, pas réussir sans calcul.
    let executor = MetalExecutor::new()?;
    for (out_dim, in_dim) in [(1, 64), (2, 128), (13, 192)] {
        let weight = test_affine_varied_u5(out_dim, in_dim)?;
        let mut identity = vec![0.0; in_dim * in_dim];
        for col in 0..in_dim {
            identity[col * in_dim + col] = 1.0;
        }
        let input = Tensor::from_vec(vec![in_dim, in_dim], identity)?;
        let gpu = executor.matmul_rhs_t_affine(&input, &weight)?;
        let cpu = weight.dequantize()?;
        for col in 0..in_dim {
            for row in 0..out_dim {
                assert_eq!(
                    gpu.data()[col * out_dim + row].to_bits(),
                    cpu.data()[row * in_dim + col].to_bits(),
                    "déquant u5 in_dim={in_dim}, row={row}, col={col}"
                );
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "device Metal requis : petit oracle sans modèle"]
fn affine_qmv_u5_gs64_matches_cpu_generic_and_routes() -> Result<()> {
    use super::profiling::take_dispatch_shapes_for_test;

    let executor = MetalExecutor::new()?;
    for (out_dim, in_dim, batch) in [
        (1, 64, 1),
        (2, 128, 2),
        (13, 192, 3),
        (64, 512, 1),
        (65, 512, 2),
        (256, 1024, 1),
        (8, 5120, 3),
        (16, 17408, 2),
    ] {
        let weight = test_affine_varied_u5(out_dim, in_dim)?;
        assert_eq!(
            executor.select_owned_affine_matmul_kernel(batch, in_dim, &weight, false),
            AffineMatmulKernel::FastQmvU5
        );
        let lhs: Vec<f32> = (0..batch)
            .flat_map(|row| varied_row(in_dim, 53 + row))
            .collect();
        let input = Tensor::from_vec(vec![batch, in_dim], lhs.clone())?;
        let cpu = input.matmul_rhs_t(&weight.dequantize()?)?;
        let generic = generic_affine_reference(&executor, &weight, &lhs, batch, "qmv_u5_generic")?;
        let aligned = out_dim % 8 == 0 && in_dim % 512 == 0;
        let kind = if aligned {
            "affine_qmv_u5_aligned_gs64"
        } else {
            "affine_qmv_u5_tail_gs64"
        };
        let dispatch = DispatchProfileShape::matmul(kind, batch, in_dim, out_dim, 64, 5);
        take_dispatch_shapes_for_test();
        let direct = executor.matmul_rhs_t_affine(&input, &weight)?;
        assert_eq!(take_dispatch_shapes_for_test(), [dispatch]);
        // L'affine après 16/32 produits change l'association f32 comme u4.
        // Même epsilon absolu que l'oracle u4 align64 ; le tail reste bit exact.
        if aligned {
            assert_close_eps(direct.data(), &generic, 1.0e-3);
        } else {
            assert_bits_equal(direct.data(), &generic, "qmv u5/générique");
        }
        assert_all_close_scaled(direct.data(), cpu.data(), "qmv u5/CPU");

        let owned = LinearWeight::AffineQuantized(weight);
        let buffers = executor.resolve_linear_weight_buffers(&owned, "qmv_u5_resident")?;
        let lhs_buf = executor.upload_f32_buffer(&lhs, "qmv_u5_lhs")?;
        // La sentinelle révèle une sortie manquante et une écriture après la fin.
        let sentinel = -12345.0;
        let output =
            executor.upload_f32_buffer(&vec![sentinel; batch * out_dim + 32], "qmv_u5_out")?;
        for resident in [false, true] {
            write_f32_buffer(&output, &vec![sentinel; batch * out_dim + 32])?;
            take_dispatch_shapes_for_test();
            let cb = executor.queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            let mut owned_buffers = Vec::new();
            let actual_out = if resident {
                executor.encode_matmul_weight_buffers(
                    encoder, &lhs_buf, batch, in_dim, &buffers, &output, false,
                )?
            } else {
                executor.encode_matmul_weight(
                    encoder,
                    &mut owned_buffers,
                    &lhs_buf,
                    batch,
                    in_dim,
                    &owned,
                    &output,
                )?
            };
            assert_eq!(actual_out, out_dim);
            encoder.end_encoding();
            commit_and_wait(cb)?;
            assert_eq!(take_dispatch_shapes_for_test(), [dispatch]);
            let actual = read_f32_buffer(&output, batch * out_dim + 32)?;
            assert!(actual[batch * out_dim..].iter().all(|&x| x == sentinel));
            assert_bits_equal(&actual[..batch * out_dim], direct.data(), "route u5/mono");
            assert_all_close_scaled(&actual[..batch * out_dim], cpu.data(), "route u5/CPU");
        }
    }
    Ok(())
}

#[test]
#[ignore = "device Metal requis : offsets u5 uniquement, sans modèle"]
fn affine_qmv_u5_offsets_bit_exact() -> Result<()> {
    let executor = MetalExecutor::new()?;
    let _reset = OffsetKernelGuard;
    super::matmul_select::OFFSET_TEST_KERNEL
        .with(|value| value.set(Some(AffineMatmulKernel::FastQmvU5)));
    for (out_dim, in_dim, batch) in [
        (1, 64, 1),
        (2, 128, 2),
        (13, 192, 3),
        (8, 512, 3),
        (8, 1024, 3),
    ] {
        let weight = test_affine_varied_u5(out_dim, in_dim)?;
        for resident in [false, true] {
            let reference = offset_dispatch(&executor, &weight, batch, resident)?;
            assert!(!reference.1.is_empty());
            assert!(reference
                .1
                .iter()
                .all(|name| name.contains("qmv_fast") && name.contains("u5")));
            for offsets in [[4, 2, 6], [64, 128, 192]] {
                for sentinel in [0x7fc1, 0xdead] {
                    let shifted = offset_weight(&executor, &weight, offsets, sentinel)?;
                    let actual = offset_dispatch(&executor, &shifted, batch, resident)?;
                    assert_eq!(
                        actual, reference,
                        "u5 offsets={offsets:?}, resident={resident}"
                    );
                }
            }
        }
    }
    Ok(())
}
