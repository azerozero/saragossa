#[test]
#[ignore = "microbench NA vs f32 sur shapes encodeur (GPU idle)"]
fn na_gemm_microbench() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let shapes = [
        ("attn 1500x1280x1280", 1500usize, 1280usize, 1280usize),
        ("fc1  1500x5120x1280", 1500, 5120, 1280),
        ("fc2  1500x1280x5120", 1500, 1280, 5120),
    ];
    for (label, m, n, k) in shapes {
        let lhs: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect();
        let rhs: Vec<f32> = (0..n * k).map(|i| ((i % 13) as f32 - 6.0) * 0.01).collect();
        // f32 : matmul tuilé ISOLÉ (dispatch seul, buffers pré-uploadés).
        let lhs_buf = executor.upload_f32_buffer(&lhs, "mb_lhs")?;
        let rhs_buf = executor.cached_buffer_from_f32(&rhs, "mb_rhs")?;
        let out_f32 = executor
            .device
            .new_buffer((m * n * 4) as u64, MTLResourceOptions::StorageModeShared);
        let time_dispatch =
            |encode: &dyn Fn(&ComputeCommandEncoderRef) -> Result<()>| -> Result<f64> {
                let mut ms = Vec::new();
                for _ in 0..6 {
                    let cb = executor.queue.new_command_buffer();
                    let enc = cb.new_compute_command_encoder();
                    let g = EncoderEndGuard::new(enc);
                    encode(enc)?;
                    g.end();
                    let t = std::time::Instant::now();
                    commit_and_wait(cb)?;
                    ms.push(t.elapsed().as_secs_f64() * 1000.0);
                }
                ms.sort_by(|a, b| a.partial_cmp(b).expect("invariant: temps fini"));
                Ok(ms[3])
            };
        let f32_med = time_dispatch(&|enc| {
            executor.encode_dense_gemm(enc, &lhs_buf, &rhs_buf, &out_f32, m, n, k)
        })?;

        // NA : matmul2d ISOLÉ (lhs+rhs^T bf16 PRÉ-construits, dispatch seul).
        let Some(pso) = executor.na_gemm_bf16.clone() else {
            eprintln!("NA indisponible");
            return Ok(());
        };
        let mut rhs_t = vec![0.0f32; k * n];
        for nn in 0..n {
            for kk in 0..k {
                rhs_t[kk * n + nn] = rhs[nn * k + kk];
            }
        }
        let a_f32 = executor.upload_f32_buffer(&lhs, "mb_a_f32")?;
        let bt_f32 = executor.upload_f32_buffer(&rhs_t, "mb_bt_f32")?;
        let a_bf16 = executor
            .device
            .new_buffer((m * k * 2) as u64, MTLResourceOptions::StorageModeShared);
        let b_bf16 = executor
            .device
            .new_buffer((k * n * 2) as u64, MTLResourceOptions::StorageModeShared);
        let out_na = executor
            .device
            .new_buffer((m * n * 4) as u64, MTLResourceOptions::StorageModeShared);
        let out_na_bn128 = executor
            .device
            .new_buffer((m * n * 4) as u64, MTLResourceOptions::StorageModeShared);
        {
            let cb = executor.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            let g = EncoderEndGuard::new(enc);
            executor.encode_f32_to_bf16(enc, &a_f32, &a_bf16, m * k)?;
            executor.encode_f32_to_bf16(enc, &bt_f32, &b_bf16, k * n)?;
            g.end();
            commit_and_wait(cb)?;
        }
        let mnk = [m as u32, n as u32, k as u32];
        let width = pso.thread_execution_width().max(1);
        let na_med = time_dispatch(&|enc| {
            enc.set_compute_pipeline_state(&pso);
            enc.set_buffer(0, Some(&a_bf16), 0);
            enc.set_buffer(1, Some(&b_bf16), 0);
            enc.set_buffer(2, Some(&out_na), 0);
            enc.set_bytes(3, 12, mnk.as_ptr().cast::<std::ffi::c_void>());
            enc.dispatch_thread_groups(
                MTLSize::new(m.div_ceil(64) as u64, n.div_ceil(32) as u64, 1),
                MTLSize::new(width * 4, 1, 1),
            );
            Ok(())
        })?;
        let na_bn128_med = if let Some(pso) = executor.na_gemm_bf16_bn128.clone() {
            let width = pso.thread_execution_width().max(1);
            Some(time_dispatch(&|enc| {
                enc.set_compute_pipeline_state(&pso);
                enc.set_buffer(0, Some(&a_bf16), 0);
                enc.set_buffer(1, Some(&b_bf16), 0);
                enc.set_buffer(2, Some(&out_na_bn128), 0);
                enc.set_bytes(3, 12, mnk.as_ptr().cast::<std::ffi::c_void>());
                enc.dispatch_thread_groups(
                    MTLSize::new(m.div_ceil(64) as u64, n.div_ceil(128) as u64, 1),
                    MTLSize::new(width * 8, 1, 1),
                );
                Ok(())
            })?)
        } else {
            None
        };
        let flops = 2.0 * m as f64 * n as f64 * k as f64;
        eprintln!(
            "{label}: f32 {:.2}ms ({:.1} TF) | NA {:.2}ms ({:.1} TF) | {:.2}x (dispatch isolé)",
            f32_med,
            flops / (f32_med / 1000.0) / 1e12,
            na_med,
            flops / (na_med / 1000.0) / 1e12,
            f32_med / na_med,
        );
        if let Some(ms) = na_bn128_med {
            eprintln!(
                "  bf16 BN128: {:.2}ms ({:.1} TF) | vs NA32 {:.2}x",
                ms,
                flops / (ms / 1000.0) / 1e12,
                na_med / ms,
            );
        }
    }
    Ok(())
}

#[test]
#[ignore = "microbench QMV decode M=1 f32 vs poids bf16"]
fn qmv_m1_bf16_microbench() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let shapes = [
        ("router 1x256x2048", 256usize, 2048usize),
        ("self/o 1x1280x1280", 1280usize, 1280usize),
        ("fc1    1x5120x1280", 5120, 1280),
        ("fc2    1x1280x5120", 1280, 5120),
        ("lmhead 1x51865x1280", 51_865, 1280),
    ];
    for (label, out_dim, in_dim) in shapes {
        let lhs: Vec<f32> = (0..in_dim)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.01)
            .collect();
        let rhs: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.01)
            .collect();
        let lhs_buf = executor.upload_f32_buffer(&lhs, "qmv_m1_lhs")?;
        let rhs_f32 = executor.cached_buffer_from_f32(&rhs, "qmv_m1_rhs_f32")?;
        let rhs_bf16 = executor.cached_buffer_from_f32_as_bf16(&rhs, "qmv_m1_rhs_bf16")?;
        let out_f32 = executor.private_f32_buffer(out_dim, "qmv_m1_out_f32")?;
        let out_bf16 = executor.private_f32_buffer(out_dim, "qmv_m1_out_bf16")?;

        let run_once = |bf16: bool| -> Result<f64> {
            let cb = executor.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            let guard = EncoderEndGuard::new(enc);
            if bf16 {
                executor.encode_dense_qmv_rhs_bf16(
                    enc, &lhs_buf, &rhs_bf16, &out_bf16, 1, out_dim, in_dim,
                )?;
            } else if can_use_dense_qmv_fast(1, in_dim, out_dim) {
                let dims = [1_u32, out_dim as u32, in_dim as u32];
                enc.set_compute_pipeline_state(&executor.dense_qmv_fast_f32);
                enc.set_buffer(0, Some(&lhs_buf), 0);
                enc.set_buffer(1, Some(&rhs_f32), 0);
                enc.set_buffer(2, Some(&out_f32), 0);
                set_u32_bytes(enc, 3, &dims, "qmv_m1_fast_dims")?;
                enc.dispatch_thread_groups(
                    MTLSize::new(1, (out_dim as u64).div_ceil(8), 1),
                    MTLSize::new(64, 1, 1),
                );
            } else {
                executor
                    .encode_dense_qmv(enc, &lhs_buf, &rhs_f32, &out_f32, 0, 1, out_dim, in_dim)?;
            }
            guard.end();
            let started = std::time::Instant::now();
            commit_and_wait(cb)?;
            Ok(started.elapsed().as_secs_f64() * 1000.0)
        };

        for _ in 0..3 {
            let _ = run_once(false)?;
            let _ = run_once(true)?;
        }
        let mut f32_ms = Vec::new();
        let mut bf16_ms = Vec::new();
        for _ in 0..10 {
            f32_ms.push(run_once(false)?);
            bf16_ms.push(run_once(true)?);
        }
        f32_ms.sort_by(|a, b| a.partial_cmp(b).expect("invariant: temps fini"));
        bf16_ms.sort_by(|a, b| a.partial_cmp(b).expect("invariant: temps fini"));
        let f32_med = f32_ms[f32_ms.len() / 2];
        let bf16_med = bf16_ms[bf16_ms.len() / 2];
        let flops = 2.0 * out_dim as f64 * in_dim as f64;
        eprintln!(
            "{label}: f32 {:.4}ms ({:.2} TF) | rhs_bf16 {:.4}ms ({:.2} TF) | {:.2}x",
            f32_med,
            flops / (f32_med / 1000.0) / 1e12,
            bf16_med,
            flops / (bf16_med / 1000.0) / 1e12,
            f32_med / bf16_med,
        );
    }
    Ok(())
}

#[test]
fn na_gemm_matches_bf16_cpu() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let (m, n, k) = (128usize, 64usize, 256usize); // tuiles 64×32 alignées
    let lhs: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32 - 8.0) * 0.03).collect();
    let rhs: Vec<f32> = (0..n * k).map(|i| ((i % 13) as f32 - 6.0) * 0.04).collect();
    let x = Tensor::from_vec(vec![m, k], lhs.clone())?;
    let w = Tensor::from_vec(vec![n, k], rhs.clone())?;
    let Some(gpu) = executor.na_gemm(&x, &w)? else {
        eprintln!("skip: NA matmul2d indisponible (macOS < 26 ?)");
        return Ok(());
    };
    // Référence bf16-input (RTNE) / accumulation f32.
    let bf = |f: f32| {
        let b = f.to_bits();
        f32::from_bits((b + (0x7fff + ((b >> 16) & 1))) & 0xffff_0000)
    };
    let mut cpu = vec![0.0f32; m * n];
    for mm in 0..m {
        for nn in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += bf(lhs[mm * k + kk]) * bf(rhs[nn * k + kk]);
            }
            cpu[mm * n + nn] = acc;
        }
    }
    assert_eq!(gpu.shape(), &[m, n]);
    assert_close_eps(gpu.data(), &cpu, 1.0e-3);
    if let Some(pso) = executor.na_gemm_bf16_bn128.as_ref() {
        let mut rhs_t = vec![0.0f32; k * n];
        for nn in 0..n {
            for kk in 0..k {
                rhs_t[kk * n + nn] = rhs[nn * k + kk];
            }
        }
        let a_f32 = executor.upload_f32_buffer(&lhs, "na128_a_f32")?;
        let bt_f32 = executor.upload_f32_buffer(&rhs_t, "na128_bt_f32")?;
        let a_bf16 = executor
            .device
            .new_buffer((m * k * 2) as u64, MTLResourceOptions::StorageModeShared);
        let b_bf16 = executor
            .device
            .new_buffer((k * n * 2) as u64, MTLResourceOptions::StorageModeShared);
        let out = executor
            .device
            .new_buffer((m * n * 4) as u64, MTLResourceOptions::StorageModeShared);
        let mnk = [m as u32, n as u32, k as u32];
        let width = pso.thread_execution_width().max(1);
        let cb = executor.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        let guard = EncoderEndGuard::new(enc);
        executor.encode_f32_to_bf16(enc, &a_f32, &a_bf16, m * k)?;
        executor.encode_f32_to_bf16(enc, &bt_f32, &b_bf16, k * n)?;
        enc.set_compute_pipeline_state(pso);
        enc.set_buffer(0, Some(&a_bf16), 0);
        enc.set_buffer(1, Some(&b_bf16), 0);
        enc.set_buffer(2, Some(&out), 0);
        enc.set_bytes(3, 12, mnk.as_ptr().cast::<std::ffi::c_void>());
        enc.dispatch_thread_groups(
            MTLSize::new(m.div_ceil(64) as u64, n.div_ceil(128) as u64, 1),
            MTLSize::new(width * 8, 1, 1),
        );
        guard.end();
        commit_and_wait(cb)?;
        assert_close_eps(&read_f32_buffer(&out, m * n)?, &cpu, 1.0e-3);
    }
    Ok(())
}

// Garde : NA_GEMM_SRC DOIT compiler — sinon TOUS les pipelines NA tombent en None
// silencieusement (et toutes les features opt-in tombent en fallback). Un seul kernel
// fautif casse toute la library ; ce test attrape ça à la frontière.
#[test]
fn na_gemm_src_compiles() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let options = CompileOptions::new();
    options.set_fast_math_enabled(true);
    // NOTE: metal_tensor (MetalPerformancePrimitives) n'existe que sur les
    // SDK récents — les runners CI plus vieux n'ont pas le header ; en prod
    // le chemin NA dégrade proprement (Option → fallback f32), donc on saute.
    let probe = "#include <metal_tensor>\nkernel void mpp_probe() {}\n";
    if executor
        .device
        .new_library_with_source(probe, &options)
        .is_err()
    {
        eprintln!("metal_tensor indisponible sur ce SDK — test NA sauté");
        return Ok(());
    }
    if let Err(e) = executor
        .device
        .new_library_with_source(NA_GEMM_SRC, &options)
    {
        panic!("NA_GEMM_SRC ne compile pas (casse tous les pipelines NA) : {e}");
    }
    Ok(())
}

#[test]
fn kernel_sources_absent_use_embedded() -> Result<()> {
    let sources = KernelSources::from_runtime_path(None)?;
    assert!(matches!(sources.matmul, std::borrow::Cow::Borrowed(_)));
    assert!(matches!(sources.na_gemm, std::borrow::Cow::Borrowed(_)));
    assert!(matches!(
        sources.steel_attention,
        std::borrow::Cow::Borrowed(_)
    ));
    Ok(())
}

#[test]
fn kernel_sources_runtime_path_reads_files() -> std::result::Result<(), Box<dyn std::error::Error>>
{
    let dir = tempfile::tempdir()?;
    std::fs::write(dir.path().join("kernels.metal"), "kernel-main")?;
    std::fs::write(dir.path().join("na_gemm"), "kernel-na")?;
    std::fs::write(dir.path().join("steel_attention"), "kernel-steel")?;

    let sources = KernelSources::from_runtime_path(Some(dir.path().to_path_buf()))?;

    assert_eq!(sources.matmul.as_ref(), "kernel-main");
    assert_eq!(sources.na_gemm.as_ref(), "kernel-na");
    assert_eq!(sources.steel_attention.as_ref(), "kernel-steel");
    Ok(())
}

// Brick #9 : débit du VRAI kernel recurrence gated_delta_seq aux params réels du 35B
// (steps=8192, 32 value-heads), pour trancher le puzzle « recurrence = oMLX mais 7s ? ».
#[ignore = "perf manuel"]
#[test]
fn gated_delta_seq_real_perf() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let spec = LinearAttentionStepSpec {
        num_key_heads: 16,
        num_value_heads: 32,
        key_head_dim: 128,
        value_head_dim: 128,
        conv_kernel_dim: 4,
        rms_eps: 1.0e-6,
    };
    let steps = 8192usize;
    let key_dim = spec.num_key_heads * spec.key_head_dim;
    let value_dim = spec.num_value_heads * spec.value_head_dim;
    let conv_dim = key_dim * 2 + value_dim;
    let state_len = spec.num_value_heads * spec.value_head_dim * spec.key_head_dim;
    let conv_out: Vec<f32> = (0..steps * conv_dim)
        .map(|i| ((i % 23) as f32 - 11.0) * 0.01)
        .collect();
    let qn: Vec<f32> = (0..steps * key_dim)
        .map(|i| ((i % 19) as f32 - 9.0) * 0.01)
        .collect();
    let kn: Vec<f32> = (0..steps * key_dim)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.01)
        .collect();
    let beta: Vec<f32> = (0..steps * spec.num_value_heads)
        .map(|i| 0.2 + (i % 8) as f32 * 0.05)
        .collect();
    let decay: Vec<f32> = (0..steps * spec.num_value_heads)
        .map(|i| 0.95 + (i % 5) as f32 * 0.008)
        .collect();
    let ssm: Vec<f32> = vec![0.0; state_len];
    let cb_buf = executor.upload_f32_buffer(&conv_out, "gds_conv")?;
    let qb = executor.upload_f32_buffer(&qn, "gds_q")?;
    let kb = executor.upload_f32_buffer(&kn, "gds_k")?;
    let bb = executor.upload_f32_buffer(&beta, "gds_beta")?;
    let db = executor.upload_f32_buffer(&decay, "gds_decay")?;
    let sb = executor.upload_f32_buffer(&ssm, "gds_ssm")?;
    let yb = executor.device.new_buffer(
        (steps * value_dim * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    {
        let cb = executor.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        let g = EncoderEndGuard::new(enc);
        executor.encode_linear_attn_gated_delta_seq_dk128(
            enc, &cb_buf, &qb, &kb, &bb, &db, &sb, false, &yb, steps, spec,
        )?;
        g.end();
        commit_and_wait(cb)?;
    }
    let iters = 6u32;
    let cb = executor.queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    let g = EncoderEndGuard::new(enc);
    for _ in 0..iters {
        executor.encode_linear_attn_gated_delta_seq_dk128(
            enc, &cb_buf, &qb, &kb, &bb, &db, &sb, false, &yb, steps, spec,
        )?;
    }
    g.end();
    let t0 = std::time::Instant::now();
    commit_and_wait(cb)?;
    let dt = t0.elapsed().as_secs_f64() / f64::from(iters);
    let line = format!(
        "[gated_delta_seq_real] steps={steps} 32heads : {:.2} ms/call (×30 couches = {:.2} s)",
        dt * 1.0e3,
        dt * 30.0
    );
    eprintln!("{line}");
    let _ = std::fs::write("/private/tmp/reti-metal-gds.txt", &line);
    Ok(())
}

// Brick #11 : coût du conv batché (reads stridés) aux params réels — suspect des 8s.
#[ignore = "perf manuel"]
#[test]
fn conv_norm_gates_batch_real_perf() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let spec = LinearAttentionStepSpec {
        num_key_heads: 16,
        num_value_heads: 32,
        key_head_dim: 128,
        value_head_dim: 128,
        conv_kernel_dim: 4,
        rms_eps: 1.0e-6,
    };
    let batch = 8192usize;
    let key_dim = spec.num_key_heads * spec.key_head_dim;
    let value_dim = spec.num_value_heads * spec.value_head_dim;
    let conv_dim = key_dim * 2 + value_dim;
    let qkv = executor.upload_f32_buffer(&vec![0.01f32; batch * conv_dim], "cb_qkv")?;
    let bin = executor.upload_f32_buffer(&vec![0.1f32; batch * spec.num_value_heads], "cb_bin")?;
    let gin = executor.upload_f32_buffer(&vec![0.1f32; batch * spec.num_value_heads], "cb_gin")?;
    let cw = executor.upload_f32_buffer(&vec![0.1f32; conv_dim * 4], "cb_cw")?;
    let cs = executor.upload_f32_buffer(&vec![0.0f32; 3 * conv_dim], "cb_cs")?;
    let al = executor.upload_f32_buffer(&vec![0.1f32; spec.num_value_heads], "cb_al")?;
    let dtb = executor.upload_f32_buffer(&vec![0.1f32; spec.num_value_heads], "cb_dtb")?;
    let co = executor.upload_f32_buffer(&vec![0.0f32; batch * conv_dim], "cb_co")?;
    let qn = executor.upload_f32_buffer(&vec![0.0f32; batch * key_dim], "cb_qn")?;
    let kn = executor.upload_f32_buffer(&vec![0.0f32; batch * key_dim], "cb_kn")?;
    let bo = executor.upload_f32_buffer(&vec![0.0f32; batch * spec.num_value_heads], "cb_bo")?;
    let de = executor.upload_f32_buffer(&vec![0.0f32; batch * spec.num_value_heads], "cb_de")?;
    let iters = 6u32;
    let cb = executor.queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    let g = EncoderEndGuard::new(enc);
    for _ in 0..iters {
        executor.encode_linear_attn_conv_norm_gates_k4_dk128_batch(
            enc, &qkv, &bin, &gin, &cw, &cs, &al, &dtb, &co, &qn, &kn, &bo, &de, batch, spec,
        )?;
    }
    g.end();
    let t0 = std::time::Instant::now();
    commit_and_wait(cb)?;
    let dt = t0.elapsed().as_secs_f64() / f64::from(iters);
    let line = format!(
        "[conv_norm_gates_batch] batch={batch} : {:.2} ms/call (×30 couches = {:.2} s)",
        dt * 1.0e3,
        dt * 30.0
    );
    eprintln!("{line}");
    let _ = std::fs::write("/private/tmp/reti-metal-cvb.txt", &line);
    Ok(())
}

// Brick #2 campagne : débit RÉEL du GEMM dense quantifié au shape in_proj du 35B
// (M=16384, K=2048, N=8192, u8 gs64), via le chemin prod encode_matmul_weight (qmv
// si RETI_RUST_QMM_NA off, NA-matmul2d si on). Décisif : TFLOP/s vs le pic ~26.
#[ignore = "perf manuel"]
#[test]
fn dense_gemm_inproj_throughput() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let (m, k, n) = (16384usize, 2048usize, 8192usize);
    let weight = crate::LinearWeight::AffineQuantized(test_affine_varied_u8(n, k)?);
    let lhs: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect();
    let lhs_buf = executor.upload_f32_buffer(&lhs, "ip_lhs")?;
    let out = executor
        .device
        .new_buffer((m * n * 4) as u64, MTLResourceOptions::StorageModeShared);
    let mut owned = Vec::new();
    {
        let cb = executor.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        let g = EncoderEndGuard::new(enc);
        executor.encode_matmul_weight(enc, &mut owned, &lhs_buf, m, k, &weight, &out)?;
        g.end();
        commit_and_wait(cb)?;
    }
    let iters = 8u32;
    let cb = executor.queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    let g = EncoderEndGuard::new(enc);
    for _ in 0..iters {
        executor.encode_matmul_weight(enc, &mut owned, &lhs_buf, m, k, &weight, &out)?;
    }
    g.end();
    let t0 = std::time::Instant::now();
    commit_and_wait(cb)?;
    let dt = t0.elapsed().as_secs_f64() / f64::from(iters);
    let tflops = 2.0 * m as f64 * k as f64 * n as f64 / dt / 1.0e12;
    let na = std::env::var("RETI_RUST_QMM_NA").is_ok();
    let line = format!(
        "[dense_inproj] qmm_na={na} M={m} K={k} N={n} : {:.2} ms/GEMM, {tflops:.1} TFLOP/s",
        dt * 1.0e3
    );
    eprintln!("{line}");
    Ok(())
}

// Port qmm_t_nax brique 1 : la primitive cooperative_tensor à fragments registres
// (NaxFrag) doit calculer C[M,N]=A[M,K]·B[N,K]^T juste — valide le mapping fragment↔thread.
