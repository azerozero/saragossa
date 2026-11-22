#[test]
#[ignore = "device Metal NAX requis : sélection TTS u5/u6/u8, sans modèle"]
fn tts_qmm_na_dequant_scope_controls_selection() -> Result<()> {
    use crate::runtime_flags::install_tts_qmm_na_dequant_scope;

    let metal = MetalExecutor::new()?;
    assert!(metal.na_gemm_bf16.is_some());
    super::flags::with_qmm_u6_override(false, || -> Result<()> {
        // N impair exclut les GEMM fusionnés u8 ; les deux branches du
        // sélecteur owned doivent respecter la portée TTS.
        for weight in [
            test_affine_varied_u5(13, 512)?,
            test_affine_varied_u6(13, 512)?,
            test_affine_varied_u8(13, 512)?,
        ] {
            let select = || {
                metal.select_owned_affine_matmul_kernel(qmm_na_min_batch(), 512, &weight, false)
            };
            assert_eq!(select(), AffineMatmulKernel::QmmNaDequant);
            {
                let _tts = install_tts_qmm_na_dequant_scope(false);
                assert_ne!(select(), AffineMatmulKernel::QmmNaDequant);
                {
                    let _opt_in = install_tts_qmm_na_dequant_scope(true);
                    assert_eq!(select(), AffineMatmulKernel::QmmNaDequant);
                }
                assert_ne!(select(), AffineMatmulKernel::QmmNaDequant);
            }
            assert_eq!(select(), AffineMatmulKernel::QmmNaDequant);
        }
        Ok(())
    })
}

#[test]
#[ignore = "device Metal requis : déquant u3/u5/u6 bf16 transposée, sans modèle"]
fn dequant_u56_bf16_transpose_matches_cpu() -> Result<()> {
    let metal = MetalExecutor::new()?;
    for bits in [3, 5, 6] {
        for (n, k) in [(1, 64), (13, 192), (64, 512)] {
            let weight = test_affine_varied_u356(bits, n, k)?;
            let cpu = weight.dequantize()?;
            for offsets in [[0, 0, 0], [4, 2, 6], [64, 128, 192]] {
                let shifted = offset_weight(&metal, &weight, offsets, 0xdead)?;
                let (packed, po) = shifted.packed_metal_view().expect("invariant: vue packed");
                let (scales, so) = shifted.scales_metal_view().expect("invariant: vue scales");
                let (biases, bo) = shifted.biases_metal_view().expect("invariant: vue biases");
                let output =
                    metal.buffer_from_slice(&vec![0xdead_u16; n * k + 32], "dequant_out")?;
                let cb = metal.queue.new_command_buffer();
                let enc = cb.new_compute_command_encoder();
                metal.encode_dequant_qweight_to_bf16_t(
                    enc,
                    packed,
                    po as u64,
                    scales,
                    so as u64,
                    biases,
                    bo as u64,
                    &output,
                    n,
                    k,
                    k * bits / 32,
                    bits,
                )?;
                enc.end_encoding();
                commit_and_wait(cb)?;
                let actual = super::byte_helpers::read_u16_buffer(&output, n * k + 32)?;
                let expected = super::buffers::f32_slice_to_bf16(cpu.data());
                for row in 0..n {
                    for col in 0..k {
                        assert_eq!(
                            actual[col * n + row],
                            expected[row * k + col],
                            "u{bits} {n}x{k}, row={row}, col={col}, offsets={offsets:?}"
                        );
                    }
                }
                assert!(actual[n * k..].iter().all(|&v| v == 0xdead));
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "device Metal NAX requis : prefill u3/u5/u6, sans modèle"]
fn qmm_u56_prefill_matches_bf16_cpu_and_routes() -> Result<()> {
    let metal = MetalExecutor::new()?;
    assert!(
        metal.na_gemm_bf16.is_some(),
        "oracle NAX explicite : pipeline requise"
    );
    super::flags::with_qmm_u6_override(false, || -> Result<()> {
        for bits in [3, 5, 6] {
            for (m, n, k) in [(32, 13, 512), (33, 64, 1024), (524, 128, 512)] {
                let weight = test_affine_varied_u356(bits, n, k)?;
                assert_eq!(
                    metal.select_owned_affine_matmul_kernel(m, k, &weight, false),
                    AffineMatmulKernel::QmmNaDequant
                );
                // Le decode mono-ligne garde son qmv (u3 : N % 8 requis).
                let mono = match bits {
                    3 if n % 8 == 0 => Some(AffineMatmulKernel::FastQmvU3),
                    3 => None,
                    5 => Some(AffineMatmulKernel::FastQmvU5),
                    _ => Some(AffineMatmulKernel::FastQmvU6),
                };
                let selected = metal.select_owned_affine_matmul_kernel(1, k, &weight, false);
                assert_ne!(selected, AffineMatmulKernel::QmmNaDequant);
                if let Some(mono) = mono {
                    assert_eq!(selected, mono);
                }
                let strict = {
                    let _greedy = install_verify_greedy_scope(true);
                    with_verify_strict_override(true, || {
                        metal.select_owned_affine_matmul_kernel(m, k, &weight, false)
                    })
                };
                assert_ne!(strict, AffineMatmulKernel::QmmNaDequant);
                let lhs: Vec<f32> = (0..m * k)
                    .map(|i| ((i * 37 % 31) as f32 - 15.0) / 64.0)
                    .collect();
                // Contrat u8 : les deux opérandes sont arrondis en bf16 avant
                // MMA. L'oracle teste cet arrondi séparément de la réduction f32.
                let cpu_weight = weight.dequantize()?.map(bf16_round);
                let cpu = Tensor::from_vec(vec![m, k], lhs.clone())?.matmul_rhs_t(&cpu_weight)?;
                let lhs = metal.upload_f32_buffer(&lhs, "prefill_u56_lhs")?;
                let out =
                    metal.upload_f32_buffer(&vec![-12345.0; m * n + 32], "prefill_u56_out")?;
                let weight = LinearWeight::AffineQuantized(weight);
                let mut scratch = Vec::new();
                let cb = metal.queue.new_command_buffer();
                let enc = cb.new_compute_command_encoder();
                super::profiling::OFFSET_TEST_DISPATCHES
                    .with(|v| *v.borrow_mut() = Some(Vec::new()));
                metal.encode_matmul_weight(enc, &mut scratch, &lhs, m, k, &weight, &out)?;
                enc.end_encoding();
                commit_and_wait(cb)?;
                let trace = super::profiling::OFFSET_TEST_DISPATCHES
                    .with(|v| v.borrow_mut().take())
                    .expect("invariant: capture active");
                assert!(trace
                    .iter()
                    .any(|v| v.starts_with(&format!("dequant_u{bits}_to_bf16_t_gs64"))));
                assert!(trace.iter().any(|v| v.starts_with("gemm_nax")));
                assert!(!trace.iter().any(|v| v.contains("qmv")));
                let actual = read_f32_buffer(&out, m * n + 32)?;
                assert!(actual[m * n..].iter().all(|&v| v == -12345.0));
                // Epsilon de l'oracle GEMM u4 align64, aucune tolérance élargie.
                assert_close_eps(&actual[..m * n], cpu.data(), 1.0e-3);
            }
        }
        Ok(())
    })
}

#[test]
#[ignore = "device Metal NAX requis : prefill u3 tuilé, sans modèle"]
fn qmm_u3_tiled_prefill_matches_bf16_cpu_and_routes() -> Result<()> {
    let metal = MetalExecutor::new()?;
    assert!(
        metal.na_gemm_coop_qb_tiled_u3.is_some(),
        "oracle NAX explicite : pipeline requise"
    );
    // M couvre une tuile partielle (40) et plusieurs tuiles (130) ; N et K
    // atteignent le plancher 2048 du GEMM tuilé.
    for (m, n, k) in [(40, 2048, 2048), (130, 2112, 2560)] {
        let weight = test_affine_varied_u356(3, n, k)?;
        assert_eq!(
            metal.select_owned_affine_matmul_kernel(m, k, &weight, false),
            AffineMatmulKernel::QmmNaFusedTiledU3
        );
        assert_eq!(
            metal.select_resident_affine_matmul_kernel(m, k, n, 64, 3, false),
            AffineMatmulKernel::QmmNaFusedTiledU3
        );
        assert_eq!(
            metal.select_owned_affine_matmul_kernel(1, k, &weight, false),
            AffineMatmulKernel::FastQmvU3
        );
        let lhs: Vec<f32> = (0..m * k)
            .map(|i| ((i * 37 % 31) as f32 - 15.0) / 64.0)
            .collect();
        let cpu_weight = weight.dequantize()?.map(bf16_round);
        let cpu = Tensor::from_vec(vec![m, k], lhs.clone())?.matmul_rhs_t(&cpu_weight)?;
        let lhs = metal.upload_f32_buffer(&lhs, "prefill_u3_lhs")?;
        let out = metal.upload_f32_buffer(&vec![-12345.0; m * n + 32], "prefill_u3_out")?;
        let weight = LinearWeight::AffineQuantized(weight);
        let mut scratch = Vec::new();
        let cb = metal.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        super::profiling::OFFSET_TEST_DISPATCHES.with(|v| *v.borrow_mut() = Some(Vec::new()));
        metal.encode_matmul_weight(enc, &mut scratch, &lhs, m, k, &weight, &out)?;
        enc.end_encoding();
        commit_and_wait(cb)?;
        let trace = super::profiling::OFFSET_TEST_DISPATCHES
            .with(|v| v.borrow_mut().take())
            .expect("invariant: capture active");
        assert!(trace
            .iter()
            .any(|v| v.starts_with("gemm_nax_coop_qb_tiled_u3")));
        assert!(!trace.iter().any(|v| v.contains("qmv") || v.contains("dequant")));
        let actual = read_f32_buffer(&out, m * n + 32)?;
        assert!(actual[m * n..].iter().all(|&v| v == -12345.0));
        // Même epsilon que l'oracle u5/u6 ci-dessus.
        assert_close_eps(&actual[..m * n], cpu.data(), 1.0e-3);
    }
    Ok(())
}

#[test]
#[ignore = "device Metal requis : identité mono/lot stricte u5/u6"]
fn qmv_u56_strict_rows_are_bit_exact() -> Result<()> {
    let metal = MetalExecutor::new()?;
    let _greedy = install_verify_greedy_scope(true);
    with_verify_strict_override(true, || {
        super::flags::with_qmm_u6_override(true, || -> Result<()> {
            for bits in [5, 6] {
                for (n, k) in [(13, 192), (16, 512), (8, 1024)] {
                    let weight = if bits == 5 {
                        test_affine_varied_u5(n, k)?
                    } else {
                        test_affine_varied_u6(n, k)?
                    };
                    let linear = LinearWeight::AffineQuantized(weight);
                    let lhs: Vec<_> = (0..9).flat_map(|r| varied_row(k, 53 + r)).collect();
                    let (owned, resident) =
                        routed_affine_outputs(&metal, &linear, &lhs, 9, "strict_u56")?;
                    let mut mono = Vec::new();
                    for row in lhs.chunks_exact(k) {
                        mono.extend(
                            routed_affine_outputs(&metal, &linear, row, 1, "strict_u56_mono")?.0,
                        );
                    }
                    for actual in [&owned, &resident] {
                        assert_eq!(
                            actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            mono.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                        );
                    }
                }
            }
            Ok(())
        })
    })
}

#[test]
#[ignore = "device Metal requis : qmv u3/u5/u6 en lignes contre qmv mono, au bit"]
fn qmv_rows_match_mono_bitwise() -> Result<()> {
    let metal = MetalExecutor::new()?;
    // u3 : 1536 et 1792 couvrent un bloc de 512 incomplet et le demi-bloc
    // final ; u5/u6 : 1024 prend la branche à 32 codes par lane, 1536 celle à 16.
    let cases: [(usize, &[(usize, usize)]); 3] = [
        (3, &[(16, 512), (8, 1536), (24, 1792)]),
        (5, &[(16, 512), (8, 1536), (24, 1024)]),
        (6, &[(16, 512), (8, 1536), (24, 1024)]),
    ];
    for (bits, shapes) in cases {
        for &(n, k) in shapes {
            let linear = LinearWeight::AffineQuantized(test_affine_varied_u356(bits, n, k)?);
            for batch in (3..=17).chain([31]) {
                let lhs: Vec<_> = (0..batch).flat_map(|r| varied_row(k, 71 + r)).collect();
                let mut mono = Vec::new();
                for row in lhs.chunks_exact(k) {
                    mono.extend(routed_affine_outputs(&metal, &linear, row, 1, "rows_mono")?.0);
                }
                let mono: Vec<_> = mono.iter().map(|v| v.to_bits()).collect();
                // Hors strict, le skinny u3 prend la main dès qmm_skinny_min_batch().
                let _greedy = install_verify_greedy_scope(true);
                let (owned, resident) = with_verify_strict_override(true, || {
                    routed_affine_outputs(&metal, &linear, &lhs, batch, "rows")
                })?;
                for actual in [&owned, &resident] {
                    assert_eq!(
                        actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        mono,
                        "u{bits} {n}x{k} batch={batch}"
                    );
                }
            }
        }
    }
    Ok(())
}

/// Poids u3/u5/u6 gs64 au bitstream MLX, échelles et biais bf16 variés.
fn test_affine_varied_u356(bits: usize, n: usize, k: usize) -> Result<AffineQuantizedTensor> {
    match bits {
        5 => test_affine_varied_u5(n, k),
        6 => test_affine_varied_u6(n, k),
        _ => {
            let packed = test_affine_mlx_u3(n, k)?.packed_data().to_vec();
            let groups = k / 64;
            let scales = Tensor::from_vec(
                vec![n, groups],
                (0..n * groups)
                    .map(|i| bf16_round(0.003 + 0.000_2 * ((i % 13) as f32)))
                    .collect(),
            )?;
            let biases = Tensor::from_vec(
                vec![n, groups],
                (0..n * groups)
                    .map(|i| bf16_round(-0.01 + 0.000_5 * ((i % 19) as f32)))
                    .collect(),
            )?;
            AffineQuantizedTensor::new(&[n, k * 3 / 32], packed, scales, biases, 64, 3)
        }
    }
}
