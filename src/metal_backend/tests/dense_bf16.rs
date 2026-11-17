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

/// Pipelines du GEMM bf16 par défaut et de sa variante à double tampon,
/// compilés depuis la source NA du runtime (sans dispatch).
fn bf16_gemm_pair(device: &Device) -> Result<(ComputePipelineState, ComputePipelineState)> {
    let source = format!(
        "{}{}",
        super::flags::nax_source_prelude(),
        include_str!("../na_gemm.metal")
    );
    let library = device
        .new_library_with_source(&source, &super::pipelines::compile_options(true))
        .map_err(InferError::Metal)?;
    let p = |name: &str| super::pipelines::pipeline(&library, device, name);
    Ok((
        p("gemm_nax_bf16_nt_tiled_x2")?,
        p("gemm_nax_bf16_nt_tiled_x2_db")?,
    ))
}

#[test]
#[ignore = "requiert un device Metal ; compilation seule, aucun dispatch"]
fn dense_bf16_db_compiles() -> Result<()> {
    let device =
        Device::system_default().ok_or_else(|| InferError::Metal("aucun device".into()))?;
    bf16_gemm_pair(&device).map(|_| ())
}

/// Encode un GEMM bf16 et renvoie la sortie lue sur CPU.
fn run_bf16_gemm(
    executor: &MetalExecutor,
    pso: &ComputePipelineState,
    lhs: &Buffer,
    rhs: &Buffer,
    (m, n, k): (usize, usize, usize),
) -> Result<Vec<f32>> {
    let out = executor.uncached_f32_buffer(m * n, "test_bf16_db_out")?;
    let cb = executor.queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    executor.encode_dense_bf16_gemm(enc, pso, "test", lhs, rhs, &out, m, n, k)?;
    enc.end_encoding();
    commit_and_wait(cb)?;
    read_f32_buffer(&out, m * n)
}

#[test]
#[ignore = "requiert Metal avec GEMM NA ; lancer sous verrou GPU ; poids synthétiques"]
fn dense_bf16_db_bits() -> Result<()> {
    let executor = MetalExecutor::new()?;
    let (base, db) = bf16_gemm_pair(&executor.device)?;
    // K=64 : un seul pas (pas de préchargement) ; 70 et 130 lignes : tuile M
    // masquée ; K=5120 : forme du 27B, 80 pas alternant les deux tampons.
    for (m, n, k) in [
        (70, 128, 64),
        (130, 256, 192),
        (64, 64, 1024),
        (200, 192, 5120),
    ] {
        // Valeurs pseudo-aléatoires non représentables en bf16 côté A.
        let mut state = 0x9e37_79b9_u32 ^ (m * 31 + n * 7 + k) as u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        let a: Vec<f32> = (0..m * k).map(|_| next()).collect();
        let bits: Vec<u8> = (0..n * k)
            .flat_map(|_| ((next().to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let lhs = executor.buffer_from_f32(&a, "test_bf16_db_lhs")?;
        let rhs = executor.weight_buffer_from_bf16_bytes(&bits)?;
        let expected = run_bf16_gemm(&executor, &base, &lhs, &rhs, (m, n, k))?;
        let actual = run_bf16_gemm(&executor, &db, &lhs, &rhs, (m, n, k))?;
        let first = expected
            .iter()
            .zip(&actual)
            .position(|(e, a)| e.to_bits() != a.to_bits());
        assert_eq!(
            first, None,
            "M={m} N={n} K={k} : double tampon non bit-identique"
        );
    }
    Ok(())
}

/// Sonde ABBA du GEMM bf16 sur les formes du 27B BF16 : médiane de 7 mesures
/// par bras, bras alternés A B B A dans le même processus.
#[test]
#[ignore = "sonde de performance : nécessite Metal, lancer sous verrou GPU au repos"]
fn dense_bf16_db_probe() -> Result<()> {
    let executor = MetalExecutor::new()?;
    let (base, db) = bf16_gemm_pair(&executor.device)?;
    // gate/up, down, q+gate pleine attention, in_proj qkv linéaire.
    for (n, k) in [(17408, 5120), (5120, 17408), (12288, 5120), (10240, 5120)] {
        let bits = vec![0x3e_u8; n * k * 2];
        let rhs = executor.weight_buffer_from_bf16_bytes(&bits)?;
        for m in [512, 4096] {
            let lhs = executor.buffer_from_f32(&vec![0.125; m * k], "probe_bf16_lhs")?;
            let out = executor.uncached_f32_buffer(m * n, "probe_bf16_out")?;
            let time = |pso: &ComputePipelineState| -> Result<f64> {
                let cb = executor.queue.new_command_buffer();
                let enc = cb.new_compute_command_encoder();
                for _ in 0..4 {
                    executor
                        .encode_dense_bf16_gemm(enc, pso, "probe", &lhs, &rhs, &out, m, n, k)?;
                }
                enc.end_encoding();
                let start = std::time::Instant::now();
                commit_and_wait(cb)?;
                Ok(start.elapsed().as_secs_f64() / 4.0)
            };
            let (mut ta, mut tb) = (Vec::new(), Vec::new());
            for round in 0..8 {
                // Deux tours de chauffe, puis ABBA.
                let (x, y) = if round % 2 == 0 {
                    (&base, &db)
                } else {
                    (&db, &base)
                };
                let (sx, sy) = (time(x)?, time(y)?);
                if round >= 2 {
                    let (a, b) = if round % 2 == 0 { (sx, sy) } else { (sy, sx) };
                    ta.push(a);
                    tb.push(b);
                }
            }
            ta.sort_by(f64::total_cmp);
            tb.sort_by(f64::total_cmp);
            let (a, b) = (ta[ta.len() / 2], tb[tb.len() / 2]);
            let tflops = |s: f64| 2.0 * (m * n * k) as f64 / s / 1e12;
            eprintln!(
                "BF16_DB M={m} N={n} K={k} base_ms={:.3} db_ms={:.3} gain_pct={:+.2} base_TFLOPS={:.2} db_TFLOPS={:.2}",
                a * 1e3,
                b * 1e3,
                100.0 * (a / b - 1.0),
                tflops(a),
                tflops(b)
            );
        }
    }
    Ok(())
}
