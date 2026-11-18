use super::*;

#[test]
#[ignore = "requiert un device Metal ; poids synthétiques uniquement"]
fn dense_bf16_qmv_matches_f32() -> Result<()> {
    let executor = MetalExecutor::new()?;
    for (out_dim, in_dim) in [(3, 35), (8, 512)] {
        let values: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| ((i % 23) as f32 - 11.0) / 16.0)
            .collect();
        let bits: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let buffer = executor.weight_buffer_from_bf16_bytes(&bits)?;
        let weight = crate::DenseBf16Tensor::new(&[out_dim, in_dim], buffer)?;
        let dense = Tensor::from_vec(vec![out_dim, in_dim], values)?;
        let linear = LinearWeight::DenseBf16(weight.clone());
        let resolved = executor.resolve_linear_weight_buffers(&linear, "test_dense_bf16")?;
        for batch in [1, 3] {
            let input = Tensor::from_vec(
                vec![batch, in_dim],
                (0..batch * in_dim)
                    .map(|i| ((i % 19) as f32 - 9.0) / 7.0)
                    .collect(),
            )?;
            assert_eq!(weight.matmul_rhs_t(&input)?, input.matmul_rhs_t(&dense)?);
            let expected = executor.matmul_rhs_t_dense(&input, &dense)?;
            let actual = executor.matmul_rhs_t_dense_bf16(&input, &weight)?;
            let exact = |actual: &[f32]| {
                assert_eq!(
                    actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    expected
                        .data()
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>()
                );
            };
            exact(actual.data());
            let lhs = executor.buffer_from_f32(input.data(), "test_bf16_lhs")?;
            let out = executor.uncached_f32_buffer(batch * out_dim, "test_bf16_out")?;
            let cb = executor.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            executor
                .encode_matmul_weight_buffers(enc, &lhs, batch, in_dim, &resolved, &out, false)?;
            enc.end_encoding();
            commit_and_wait(cb)?;
            exact(&read_f32_buffer(&out, batch * out_dim)?);
            let cb = executor.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            executor.encode_matmul_weight(
                enc,
                &mut Vec::new(),
                &lhs,
                batch,
                in_dim,
                &linear,
                &out,
            )?;
            enc.end_encoding();
            commit_and_wait(cb)?;
            exact(&read_f32_buffer(&out, batch * out_dim)?);
        }
        assert!(weight
            .matmul_rhs_t(&Tensor::zeros(vec![1, in_dim + 1])?)
            .is_err());
    }
    Ok(())
}

#[test]
#[ignore = "requiert Metal avec GEMM NA et SARAGOSSA_RUST_DENSE_BF16_GEMM actif (défaut) ; poids synthétiques"]
fn dense_bf16_prefill_gemm_matches_f32() -> Result<()> {
    dense_bf16_prefill_matches_f32("gemm_nax_bf16_nt_tiled_x2")
}

#[test]
#[ignore = "requiert Metal avec GEMM NA ; poids synthétiques"]
fn decision_dense_bf16_gemm_rounds_activation_once() -> Result<()> {
    crate::runtime_flags::with_decision_noncausal(|| {
        dense_bf16_prefill_matches_f32("gemm_nax_bf16_nt_bn128")
    })
}

#[test]
#[ignore = "requiert Metal avec GEMM NA ; kill-switch isolé en sous-processus ; poids synthétiques"]
fn dense_bf16_prefill_kill_switch_selects_qmv(
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    const CHILD: &str = "SARAGOSSA_TEST_DENSE_BF16_PREFILL_QMV_CHILD";
    if std::env::var_os(CHILD).is_some() {
        dense_bf16_prefill_matches_f32("dense_qmv_rhs_bf16_f32")?;
        return Ok(());
    }
    // Le flag est mémorisé par OnceLock : l'isoler sans modifier l'environnement
    // des autres tests, même si le processus parent a déjà sélectionné le GEMM.
    let output = std::process::Command::new(std::env::current_exe()?)
        .args([
            "--exact",
            "metal_backend::tests::dense_bf16::dense_bf16_prefill_kill_switch_selects_qmv",
            "--ignored",
            "--nocapture",
        ])
        .env(CHILD, "1")
        .env("SARAGOSSA_RUST_DENSE_BF16_GEMM", "0")
        .output()?;
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(())
}

fn dense_bf16_prefill_matches_f32(expected_kernel: &str) -> Result<()> {
    let executor = MetalExecutor::new()?;
    // Sans pipeline NA, un qmv ne prouverait pas l'effet du kill-switch.
    assert!(
        executor.na_gemm_bf16_nt_tiled_x2.is_some(),
        "ce test exige le pipeline GEMM NA bf16"
    );
    // 70 lignes : une tuile M pleine et une tuile masquée ; deux tuiles N.
    let (batch, out_dim, in_dim) = (70, 256, 192);
    let values: Vec<f32> = (0..out_dim * in_dim)
        .map(|i| ((i % 23) as f32 - 11.0) / 16.0)
        .collect();
    let bits: Vec<u8> = values
        .iter()
        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect();
    let weight = crate::DenseBf16Tensor::new(
        &[out_dim, in_dim],
        executor.weight_buffer_from_bf16_bytes(&bits)?,
    )?;
    let resolved = executor
        .resolve_linear_weight_buffers(&LinearWeight::DenseBf16(weight), "test_bf16_gemm")?;
    // Activations non représentables en bf16 : le lo de la scission compte.
    let input = Tensor::from_vec(
        vec![batch, in_dim],
        (0..batch * in_dim)
            .map(|i| ((i % 19) as f32 - 9.0) / 7.0 + (i % 5) as f32 * 1e-3)
            .collect(),
    )?;
    let decision = expected_kernel == "gemm_nax_bf16_nt_bn128";
    // Le GEMM de décision arrondit l'activation en bf16 (au plus proche, pair).
    let reference = if decision {
        let rounded = input.data().iter().map(|&v| {
            let bits = v.to_bits();
            f32::from_bits((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff_0000)
        });
        Tensor::from_vec(vec![batch, in_dim], rounded.collect())?
    } else {
        input.clone()
    };
    let expected = reference.matmul_rhs_t(&Tensor::from_vec(vec![out_dim, in_dim], values)?)?;
    let lhs = executor.buffer_from_f32(input.data(), "test_bf16_gemm_lhs")?;
    let out = executor.uncached_f32_buffer(batch * out_dim, "test_bf16_gemm_out")?;
    let cb = executor.queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    super::profiling::OFFSET_TEST_DISPATCHES.with(|trace| {
        *trace.borrow_mut() = Some(Vec::new());
    });
    let encoded =
        executor.encode_matmul_weight_buffers(enc, &lhs, batch, in_dim, &resolved, &out, false);
    let trace = super::profiling::OFFSET_TEST_DISPATCHES
        .with(|trace| trace.borrow_mut().take())
        .expect("invariant: capture de dispatch active");
    enc.end_encoding();
    encoded?;
    commit_and_wait(cb)?;
    let mut expected_trace = Vec::new();
    if decision {
        expected_trace.push(format!("f32_to_bf16 M={} N=1 K=0", batch * in_dim));
    }
    expected_trace.push(format!(
        "{expected_kernel} M={batch} N={out_dim} K={in_dim}"
    ));
    assert_eq!(
        trace, expected_trace,
        "le kernel attendu doit être réellement dispatché"
    );
    let actual = read_f32_buffer(&out, batch * out_dim)?;
    // Tolérance f32 relative : le GEMM n'arrondit pas les activations en bf16.
    for (index, (a, e)) in actual.iter().zip(expected.data()).enumerate() {
        assert!(
            (a - e).abs() <= 1e-3 * e.abs().max(1.0),
            "{index}: {a} != {e}"
        );
    }
    Ok(())
}

#[test]
#[ignore = "requiert un device Metal ; poids synthétiques uniquement"]
fn dense_bf16_embedding_matches_f32() -> Result<()> {
    let executor = MetalExecutor::new()?;
    let values = vec![1.0, -2.0, 0.25, 4.0, -0.5, 3.0];
    let dense = Tensor::from_vec(vec![3, 2], values)?;
    let bits: Vec<u8> = dense
        .data()
        .iter()
        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect();
    let weight =
        crate::DenseBf16Tensor::new(&[3, 2], executor.weight_buffer_from_bf16_bytes(&bits)?)?;
    let table = EmbeddingWeight::DenseBf16(weight.clone());
    let expected = crate::embedding::embed_tokens(&dense, &[2, 0])?;
    assert_eq!(
        crate::embedding::embed_weight_tokens(&table, &[2, 0])?,
        expected
    );
    assert!(crate::embedding::embed_weight_tokens(&table, &[3]).is_err());
    let resolved = executor.resolve_embedding_weight_buffers(&table)?;
    let MetalEmbeddingWeightBuffers::DenseBf16 { table: buffer, .. } = &resolved else {
        panic!("embedding bf16 attendu");
    };
    assert_eq!(buffer.contents(), weight.buffer().contents());
    let indices = executor.buffer_from_u32(&[2, 0], "test_bf16_indices")?;
    let out = executor.uncached_f32_buffer(4, "test_bf16_embedding_out")?;
    for recast in [false, true] {
        let cb = executor.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        executor.encode_embeddings_from_indices_scaled(
            enc, &resolved, &indices, &out, 2, 2, 3.0, recast,
        )?;
        enc.end_encoding();
        commit_and_wait(cb)?;
        assert_eq!(
            read_f32_buffer(&out, 4)?,
            expected.data().iter().map(|v| v * 3.0).collect::<Vec<_>>()
        );
    }
    Ok(())
}

fn stage_probe_run(
    metal: &MetalExecutor,
    stage: bool,
    buffers: [&metal::Buffer; 3],
    dims: [u32; 3],
    reps: usize,
) -> Result<f64> {
    let (pipeline, rows) = if stage {
        (&metal.dense_qmv_rhs_bf16_stage_f32, 4)
    } else {
        (&metal.dense_qmv_rhs_bf16_f32, 1)
    };
    let cb = metal.queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    for (index, buffer) in buffers.iter().enumerate() {
        enc.set_buffer(index as u64, Some(buffer), 0);
    }
    set_u32_bytes(enc, 3, &dims, "stage_dims")?;
    for _ in 0..reps {
        enc.dispatch_thread_groups(
            MTLSize::new(u64::from(dims[1]).div_ceil(rows), u64::from(dims[0]), 1),
            MTLSize::new(32 * rows, 1, 1),
        );
        post_dispatch_barrier_buffer(enc, buffers[2]);
    }
    enc.end_encoding();
    let start = std::time::Instant::now();
    commit_and_wait(cb)?;
    Ok(start.elapsed().as_secs_f64() / reps as f64)
}

/// Compare le qmv bf16 à blocs de 256 au qmv par ligne, au bit près, puis
/// mesure leur bande passante si `probe` (formes du 27B BF16).
fn dense_bf16_stage_vs_row(shapes: &[(usize, usize, usize)], probe: bool) -> Result<()> {
    let metal = MetalExecutor::new()?;
    for &(n, k, batch) in shapes {
        let bits: Vec<u16> = (0..n * k)
            .map(|i| {
                let v = ((i.wrapping_mul(2_654_435_761) >> 7) % 4093) as f32 / 4093.0 - 0.5;
                (v.to_bits() >> 16) as u16
            })
            .collect();
        let rhs = metal.buffer_from_slice(&bits, "stage_rhs")?;
        drop(bits);
        // Activations non représentables en bf16, signes mêlés.
        let x: Vec<f32> = (0..batch * k)
            .map(|i| ((i * 40_503 % 997) as f32 / 997.0 - 0.5) * 1.37)
            .collect();
        let lhs = metal.upload_f32_buffer(&x, "stage_lhs")?;
        let dims = [batch as u32, n as u32, k as u32];
        let row_out = metal.uncached_f32_buffer(batch * n, "stage_row_out")?;
        let stage_out = metal.uncached_f32_buffer(batch * n, "stage_out")?;
        stage_probe_run(&metal, false, [&lhs, &rhs, &row_out], dims, 1)?;
        stage_probe_run(&metal, true, [&lhs, &rhs, &stage_out], dims, 1)?;
        let bits_of = |buffer| -> Result<Vec<u32>> {
            Ok(read_f32_buffer(buffer, batch * n)?
                .iter()
                .map(|v| v.to_bits())
                .collect())
        };
        let expected = bits_of(&row_out)?;
        assert!(expected.iter().all(|&b| b != 0), "N={n} K={k}");
        assert_eq!(bits_of(&stage_out)?, expected, "N={n} K={k} M={batch}");
        if !probe {
            continue;
        }
        let reps = if n * k > 1 << 26 { 8 } else { 32 };
        let mut samples = [Vec::new(), Vec::new()];
        for trial in 0..9 {
            for stage in [trial % 2 == 1, trial % 2 == 0] {
                let t = stage_probe_run(&metal, stage, [&lhs, &rhs, &stage_out], dims, reps)?;
                if trial >= 2 {
                    samples[usize::from(stage)].push(t);
                }
            }
        }
        for (stage, values) in samples.iter_mut().enumerate() {
            values.sort_by(f64::total_cmp);
            eprintln!(
                "N={n} K={k} stage={} median={:.3}ms {:.0}GB/s",
                stage == 1,
                values[3] * 1e3,
                (n * k * 2) as f64 / values[3] / 1e9
            );
        }
    }
    Ok(())
}

#[test]
#[ignore = "requiert un device Metal ; poids synthétiques uniquement"]
fn dense_bf16_qmv_stage_matches_row_qmv_bitwise() -> Result<()> {
    dense_bf16_stage_vs_row(
        &[(37, 512, 2), (5, 256, 1), (1024, 5120, 3), (48, 5120, 1)],
        false,
    )
}

// Temps commit/wait amortis sur plusieurs dispatches, ordre alterné : contrôle
// local, pas une distribution sur GPU au repos.
#[test]
#[ignore = "sonde bande passante qmv bf16 par ligne contre blocs de 256, GPU au repos"]
fn dense_bf16_qmv_stage_bandwidth_probe() -> Result<()> {
    dense_bf16_stage_vs_row(
        &[
            (5120, 17408, 1),
            (17408, 5120, 1),
            (10240, 5120, 1),
            (6144, 5120, 1),
            (5120, 6144, 1),
            (248_320, 5120, 1),
        ],
        true,
    )
}
