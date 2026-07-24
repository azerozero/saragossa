// Port qmm_t_nax brique 3 : GEMM naïf + dé-quant B u8 gs64 EN kernel doit calculer juste.

impl MetalExecutor {
    /// Exécute le coop routé avec des poids et activations de modèle réels.
    pub(crate) fn moe_routed_rows_coop_real_for_test(
        &self,
        input: &Tensor,
        router: &Linear,
        experts: &[GatedMlp],
        top_k: usize,
    ) -> Result<Tensor> {
        let (rows, hidden) = input.as_matrix()?;
        let stacked = self.stacked_moe_buffers(experts)?;
        if !MetalMoeRoutedWeights::stacked_coop_compatible(&stacked) {
            return Err(InferError::Config(
                "poids MoE réels incompatibles avec le coop routé".to_string(),
            ));
        }
        let expert_count = stacked.gate.experts;
        let out_dim = stacked.down.out_dim;
        let weights = MetalMoeRoutedWeights {
            router: self.resolve_linear_weight_buffers(
                router.weight(),
                "qwen35_oracle_router",
            )?,
            stacked,
        };
        let slots = rows
            .checked_mul(top_k)
            .ok_or_else(|| InferError::Dimension("oracle coop slots débordent".to_string()))?;
        let input_buffer = self.upload_f32_buffer(input.data(), "qwen35_oracle_input")?;
        let router_buffer =
            self.uncached_f32_buffer(rows * expert_count, "qwen35_oracle_router_out")?;
        let indices_buffer = self.uncached_u32_buffer(slots, "qwen35_oracle_indices")?;
        let scores_buffer = self.uncached_f32_buffer(slots, "qwen35_oracle_scores")?;
        let down_buffer =
            self.uncached_f32_buffer(slots * out_dim, "qwen35_oracle_down")?;
        let output_buffer =
            self.uncached_f32_buffer(rows * out_dim, "qwen35_oracle_output")?;

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let guard = EncoderEndGuard::new(encoder);
        let mut owned = Vec::<metal::Buffer>::new();
        self.encode_moe_routed_rows_coop(
            encoder,
            &mut owned,
            &input_buffer,
            None,
            &output_buffer,
            &router_buffer,
            &indices_buffer,
            &scores_buffer,
            &down_buffer,
            rows,
            hidden,
            &weights,
            top_k,
        )?;
        guard.end();
        commit_and_wait(command_buffer)?;
        Tensor::from_vec(
            vec![rows, out_dim],
            read_f32_buffer(&output_buffer, rows * out_dim)?,
        )
    }
}

#[test]
fn coop_qb_matches_cpu() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let Some(pso) = executor.na_gemm_coop_qb.clone() else {
        eprintln!("skip: coop qb indisponible (macOS < 26 ?)");
        return Ok(());
    };
    run_coop_qb_matches_cpu(&executor, &pso, 64)
}

// Chantier B : la variante dense gs128 doit utiliser le scale/bias par 128 colonnes K.
#[test]
fn coop_qb_gs128_matches_cpu() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let Some(pso) = executor.na_gemm_coop_qb_gs128.clone() else {
        eprintln!("skip: coop qb gs128 indisponible (macOS < 26 ?)");
        return Ok(());
    };
    run_coop_qb_matches_cpu(&executor, &pso, 128)
}

#[test]
fn coop_qb_tiled_matches_cpu() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let Some(pso) = executor.na_gemm_coop_qb_tiled.clone() else {
        eprintln!("skip: coop qb tiled indisponible (macOS < 26 ?)");
        return Ok(());
    };
    run_coop_qb_tiled_matches_cpu(&executor, &pso, 64)
}

#[test]
fn coop_qb_tiled_gs128_matches_cpu() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let Some(pso) = executor.na_gemm_coop_qb_tiled_gs128.clone() else {
        eprintln!("skip: coop qb tiled gs128 indisponible (macOS < 26 ?)");
        return Ok(());
    };
    run_coop_qb_tiled_matches_cpu(&executor, &pso, 128)
}

fn run_coop_qb_tiled_matches_cpu(
    executor: &MetalExecutor,
    pso: &ComputePipelineState,
    group_size: usize,
) -> Result<()> {
    let (m, n, k) = (65usize, 128usize, 512usize);
    let weight = test_affine_varied_u8_group(n, k, group_size)?;
    let mut lhs = Vec::with_capacity(m * k);
    for row in 0..m {
        lhs.extend_from_slice(&varied_row(k, 71 + row));
    }
    let a_f32 = executor.upload_f32_buffer(&lhs, "qb_tiled_a")?;
    let packed = executor.buffer_from_slice(weight.packed_data(), "qb_tiled_packed")?;
    let scales = executor.buffer_from_f32_as_bf16(weight.scales().data(), "qb_tiled_scales")?;
    let biases = executor.buffer_from_f32_as_bf16(weight.biases().data(), "qb_tiled_biases")?;
    let a_bf16 = executor
        .device
        .new_buffer((m * k * 2) as u64, MTLResourceOptions::StorageModeShared);
    let out = executor.uncached_f32_buffer(m * n, "qb_tiled_out")?;
    let mnk = [m as u32, n as u32, k as u32];
    let cb = executor.queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    let guard = EncoderEndGuard::new(enc);
    executor.encode_f32_to_bf16(enc, &a_f32, &a_bf16, m * k)?;
    enc.set_compute_pipeline_state(pso);
    enc.set_buffer(0, Some(&a_bf16), 0);
    enc.set_buffer(1, Some(&packed), 0);
    enc.set_buffer(2, Some(&scales), 0);
    enc.set_buffer(3, Some(&biases), 0);
    enc.set_buffer(4, Some(&out), 0);
    enc.set_bytes(5, 12, mnk.as_ptr().cast::<std::ffi::c_void>());
    let width = pso.thread_execution_width().max(1);
    enc.dispatch_thread_groups(
        MTLSize::new(m.div_ceil(64) as u64, (n / 64) as u64, 1),
        MTLSize::new(width * 4, 1, 1),
    );
    guard.end();
    commit_and_wait(cb)?;
    let gpu = read_f32_buffer(&out, m * n)?;
    let cpu = qmm_na_qb_reference(&weight, &lhs, m)?;
    assert_close_eps(&gpu, &cpu, 5.0e-2);
    Ok(())
}

fn run_coop_qb_matches_cpu(
    executor: &MetalExecutor,
    pso: &ComputePipelineState,
    group_size: usize,
) -> Result<()> {
    let m = if group_size == 128 { 17 } else { 16 };
    let (n, k) = (64usize, group_size * 2);
    let groups = k / group_size;
    let lhs: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32 - 8.0) * 0.03).collect();
    let q: Vec<u8> = (0..n * k).map(|i| ((i * 7 + 3) % 256) as u8).collect();
    let scales: Vec<f32> = (0..n * groups)
        .map(|i| 0.001 + (i % 5) as f32 * 0.0003)
        .collect();
    let biases: Vec<f32> = (0..n * groups)
        .map(|i| -0.1 + (i % 7) as f32 * 0.02)
        .collect();
    let mut packed = vec![0u32; n * (k / 4)];
    for nn in 0..n {
        for kk in 0..k {
            packed[nn * (k / 4) + kk / 4] |= u32::from(q[nn * k + kk]) << ((kk % 4) * 8);
        }
    }
    let a_f32 = executor.upload_f32_buffer(&lhs, "qb_a")?;
    let s_f32 = executor.upload_f32_buffer(&scales, "qb_s")?;
    let bi_f32 = executor.upload_f32_buffer(&biases, "qb_b")?;
    let a_bf16 = executor
        .device
        .new_buffer((m * k * 2) as u64, MTLResourceOptions::StorageModeShared);
    let s_bf16 = executor.device.new_buffer(
        (n * groups * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let bi_bf16 = executor.device.new_buffer(
        (n * groups * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let packed_buf = executor.device.new_buffer_with_data(
        packed.as_ptr().cast::<std::ffi::c_void>(),
        (packed.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let out = executor
        .device
        .new_buffer((m * n * 4) as u64, MTLResourceOptions::StorageModeShared);
    let mnk = [m as u32, n as u32, k as u32];
    let cb = executor.queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    let guard = EncoderEndGuard::new(enc);
    executor.encode_f32_to_bf16(enc, &a_f32, &a_bf16, m * k)?;
    executor.encode_f32_to_bf16(enc, &s_f32, &s_bf16, n * groups)?;
    executor.encode_f32_to_bf16(enc, &bi_f32, &bi_bf16, n * groups)?;
    enc.set_compute_pipeline_state(pso);
    enc.set_buffer(0, Some(&a_bf16), 0);
    enc.set_buffer(1, Some(&packed_buf), 0);
    enc.set_buffer(2, Some(&s_bf16), 0);
    enc.set_buffer(3, Some(&bi_bf16), 0);
    enc.set_buffer(4, Some(&out), 0);
    enc.set_bytes(5, 12, mnk.as_ptr().cast::<std::ffi::c_void>());
    enc.dispatch_thread_groups(
        MTLSize::new(m.div_ceil(16) as u64, (n / 32) as u64, 1),
        MTLSize::new(32, 1, 1),
    );
    guard.end();
    commit_and_wait(cb)?;
    let gpu = read_f32_buffer(&out, m * n)?;
    let bf = |f: f32| {
        let b = f.to_bits();
        f32::from_bits((b + (0x7fff + ((b >> 16) & 1))) & 0xffff_0000)
    };
    let mut cpu = vec![0.0f32; m * n];
    for mm in 0..m {
        for nn in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                let g = kk / group_size;
                let s = bf(scales[nn * groups + g]);
                let bi = bf(biases[nn * groups + g]);
                let deq = bf(f32::from(q[nn * k + kk]) * s + bi);
                acc += bf(lhs[mm * k + kk]) * deq;
            }
            cpu[mm * n + nn] = acc;
        }
    }
    assert_close_eps(&gpu, &cpu, 5.0e-2);
    Ok(())
}

#[ignore = "perf manuel, pas correctness"]
#[test]
fn coop_qb_perf() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let Some(pso) = executor.na_gemm_coop_qb.clone() else {
        return Ok(());
    };
    let (m, n, k) = (512usize, 512usize, 2048usize);
    let groups = k / 64;
    let lhs: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32 - 8.0) * 0.03).collect();
    let q: Vec<u8> = (0..n * k).map(|i| ((i * 7 + 3) % 256) as u8).collect();
    let scales: Vec<f32> = (0..n * groups)
        .map(|i| 0.001 + (i % 5) as f32 * 0.0003)
        .collect();
    let biases: Vec<f32> = (0..n * groups)
        .map(|i| -0.1 + (i % 7) as f32 * 0.02)
        .collect();
    let mut packed = vec![0u32; n * (k / 4)];
    for nn in 0..n {
        for kk in 0..k {
            packed[nn * (k / 4) + kk / 4] |= u32::from(q[nn * k + kk]) << ((kk % 4) * 8);
        }
    }
    let a_f32 = executor.upload_f32_buffer(&lhs, "qbp_a")?;
    let s_f32 = executor.upload_f32_buffer(&scales, "qbp_s")?;
    let bi_f32 = executor.upload_f32_buffer(&biases, "qbp_b")?;
    let a_bf16 = executor
        .device
        .new_buffer((m * k * 2) as u64, MTLResourceOptions::StorageModeShared);
    let s_bf16 = executor.device.new_buffer(
        (n * groups * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let bi_bf16 = executor.device.new_buffer(
        (n * groups * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let packed_buf = executor.device.new_buffer_with_data(
        packed.as_ptr().cast::<std::ffi::c_void>(),
        (packed.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let out = executor
        .device
        .new_buffer((m * n * 4) as u64, MTLResourceOptions::StorageModeShared);
    {
        let cb = executor.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        let guard = EncoderEndGuard::new(enc);
        executor.encode_f32_to_bf16(enc, &a_f32, &a_bf16, m * k)?;
        executor.encode_f32_to_bf16(enc, &s_f32, &s_bf16, n * groups)?;
        executor.encode_f32_to_bf16(enc, &bi_f32, &bi_bf16, n * groups)?;
        guard.end();
        commit_and_wait(cb)?;
    }
    let mnk = [m as u32, n as u32, k as u32];
    let iters = 200u32;
    let cb = executor.queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    let guard = EncoderEndGuard::new(enc);
    enc.set_compute_pipeline_state(&pso);
    enc.set_buffer(0, Some(&a_bf16), 0);
    enc.set_buffer(1, Some(&packed_buf), 0);
    enc.set_buffer(2, Some(&s_bf16), 0);
    enc.set_buffer(3, Some(&bi_bf16), 0);
    enc.set_buffer(4, Some(&out), 0);
    enc.set_bytes(5, 12, mnk.as_ptr().cast::<std::ffi::c_void>());
    for _ in 0..iters {
        enc.dispatch_thread_groups(
            MTLSize::new((m / 16) as u64, (n / 32) as u64, 1),
            MTLSize::new(32, 1, 1),
        );
    }
    guard.end();
    let t0 = std::time::Instant::now();
    commit_and_wait(cb)?;
    let dt = t0.elapsed().as_secs_f64() / f64::from(iters);
    let gflops = (2.0 * m as f64 * n as f64 * k as f64) / dt / 1.0e9;
    eprintln!(
        "[coop_qb_perf batched] {m}×{n}×{k} : {:.4} ms/GEMM, {gflops:.0} GFLOP/s (dense=13494)",
        dt * 1.0e3
    );
    Ok(())
}

// Port qmm_t_nax brique 4 : GEMM quantifié GROUPÉ MoE — chaque tuile M utilise l'expert
// désigné par tile_expert, poids empilé [experts,N,K]. Valide le gather par index/expert.
#[test]
fn coop_qb_grouped_matches_cpu() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let Some(pso) = executor.na_gemm_coop_qb_grouped.clone() else {
        eprintln!("skip: coop qb grouped indisponible (macOS < 26 ?)");
        return Ok(());
    };
    let (experts, n, k) = (2usize, 64usize, 128usize);
    let groups = k / 64;
    let m_pad = experts * 16; // 1 tuile de 16 lignes par expert
    let tile_expert: Vec<u32> = vec![0, 1]; // tuile 0 → expert 0, tuile 1 → expert 1
    let lhs: Vec<f32> = (0..m_pad * k)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.03)
        .collect();
    let q: Vec<u8> = (0..experts * n * k)
        .map(|i| ((i * 7 + 3) % 256) as u8)
        .collect();
    let scales: Vec<f32> = (0..experts * n * groups)
        .map(|i| 0.001 + (i % 5) as f32 * 0.0003)
        .collect();
    let biases: Vec<f32> = (0..experts * n * groups)
        .map(|i| -0.1 + (i % 7) as f32 * 0.02)
        .collect();
    let mut packed = vec![0u32; experts * n * (k / 4)];
    for e in 0..experts {
        for nn in 0..n {
            for kk in 0..k {
                let base = (e * n + nn) * (k / 4);
                packed[base + kk / 4] |= u32::from(q[(e * n + nn) * k + kk]) << ((kk % 4) * 8);
            }
        }
    }
    let a_f32 = executor.upload_f32_buffer(&lhs, "qbg_a")?;
    let s_f32 = executor.upload_f32_buffer(&scales, "qbg_s")?;
    let bi_f32 = executor.upload_f32_buffer(&biases, "qbg_b")?;
    let a_bf16 = executor.device.new_buffer(
        (m_pad * k * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let s_bf16 = executor.device.new_buffer(
        (experts * n * groups * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let bi_bf16 = executor.device.new_buffer(
        (experts * n * groups * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let packed_buf = executor.device.new_buffer_with_data(
        packed.as_ptr().cast::<std::ffi::c_void>(),
        (packed.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let te_buf = executor.device.new_buffer_with_data(
        tile_expert.as_ptr().cast::<std::ffi::c_void>(),
        (tile_expert.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let out = executor.device.new_buffer(
        (m_pad * n * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let mnk = [m_pad as u32, n as u32, k as u32];
    let cb = executor.queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    let guard = EncoderEndGuard::new(enc);
    executor.encode_f32_to_bf16(enc, &a_f32, &a_bf16, m_pad * k)?;
    executor.encode_f32_to_bf16(enc, &s_f32, &s_bf16, experts * n * groups)?;
    executor.encode_f32_to_bf16(enc, &bi_f32, &bi_bf16, experts * n * groups)?;
    enc.set_compute_pipeline_state(&pso);
    enc.set_buffer(0, Some(&a_bf16), 0);
    enc.set_buffer(1, Some(&packed_buf), 0);
    enc.set_buffer(2, Some(&s_bf16), 0);
    enc.set_buffer(3, Some(&bi_bf16), 0);
    enc.set_buffer(4, Some(&out), 0);
    enc.set_buffer(5, Some(&te_buf), 0);
    enc.set_bytes(6, 12, mnk.as_ptr().cast::<std::ffi::c_void>());
    enc.dispatch_thread_groups(
        MTLSize::new((m_pad / 16) as u64, (n / 32) as u64, 1),
        MTLSize::new(32, 1, 1),
    );
    guard.end();
    commit_and_wait(cb)?;
    let gpu = read_f32_buffer(&out, m_pad * n)?;
    let bf = |f: f32| {
        let b = f.to_bits();
        f32::from_bits((b + (0x7fff + ((b >> 16) & 1))) & 0xffff_0000)
    };
    let mut cpu = vec![0.0f32; m_pad * n];
    for mm in 0..m_pad {
        let e = tile_expert[mm / 16] as usize;
        for nn in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                let g = kk / 64;
                let s = bf(scales[(e * n + nn) * groups + g]);
                let bi = bf(biases[(e * n + nn) * groups + g]);
                let deq = bf(f32::from(q[(e * n + nn) * k + kk]) * s + bi);
                acc += bf(lhs[mm * k + kk]) * deq;
            }
            cpu[mm * n + nn] = acc;
        }
    }
    assert_close_eps(&gpu, &cpu, 5.0e-2);
    Ok(())
}

#[test]
fn coop_qb_grouped_u4_gate_up_swiglu_scatter_matches_cpu() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let (Some(pso_gate_up), Some(pso_scatter)) = (
        executor.na_gemm_coop_qb_grouped_gate_up_swiglu_u4.clone(),
        executor.na_gemm_coop_qb_grouped_scatter_u4.clone(),
    ) else {
        eprintln!("skip: coop qb grouped u4 indisponible (macOS < 26 ?)");
        return Ok(());
    };

    let (experts, rows, top_k, inter, hidden) = (2_usize, 16_usize, 2_usize, 64_usize, 64_usize);
    let max_m = rows * top_k;
    let groups = hidden / FAST_QMV_GROUP_SIZE;
    let tile_expert = vec![0_u32, 1_u32];
    let perm = (0..max_m as u32).collect::<Vec<_>>();
    let input = (0..rows * hidden)
        .map(|idx| ((idx % 29) as f32 - 14.0) * 0.02)
        .collect::<Vec<_>>();
    let q_gate = (0..experts * inter * hidden)
        .map(|idx| ((idx * 7 + 3) % 16) as u8)
        .collect::<Vec<_>>();
    let q_up = (0..experts * inter * hidden)
        .map(|idx| ((idx * 5 + 9) % 16) as u8)
        .collect::<Vec<_>>();
    let q_down = (0..experts * hidden * inter)
        .map(|idx| ((idx * 11 + 1) % 16) as u8)
        .collect::<Vec<_>>();
    let scales = (0..experts * inter * groups)
        .map(|idx| bf16_round(0.002 + (idx % 5) as f32 * 0.0002))
        .collect::<Vec<_>>();
    let biases = (0..experts * inter * groups)
        .map(|idx| bf16_round(-0.03 + (idx % 7) as f32 * 0.004))
        .collect::<Vec<_>>();
    let down_scales = (0..experts * hidden * groups)
        .map(|idx| bf16_round(0.0015 + (idx % 3) as f32 * 0.0003))
        .collect::<Vec<_>>();
    let down_biases = (0..experts * hidden * groups)
        .map(|idx| bf16_round(-0.02 + (idx % 5) as f32 * 0.003))
        .collect::<Vec<_>>();

    let pack_u4 = |q: &[u8], experts: usize, out_dim: usize, in_dim: usize| {
        let packed_cols = in_dim / 8;
        let mut packed = vec![0_u32; experts * out_dim * packed_cols];
        for e in 0..experts {
            for row in 0..out_dim {
                for col in 0..in_dim {
                    let word = (e * out_dim + row) * packed_cols + col / 8;
                    packed[word] |=
                        u32::from(q[(e * out_dim + row) * in_dim + col]) << ((col % 8) * 4);
                }
            }
        }
        packed
    };
    let gate_packed = pack_u4(&q_gate, experts, inter, hidden);
    let up_packed = pack_u4(&q_up, experts, inter, hidden);
    let down_packed = pack_u4(&q_down, experts, hidden, inter);

    let input_buf = executor.upload_f32_buffer(&input, "coop_u4_input")?;
    let gate_buf = executor.buffer_from_slice(&gate_packed, "coop_u4_gate")?;
    let up_buf = executor.buffer_from_slice(&up_packed, "coop_u4_up")?;
    let down_buf = executor.buffer_from_slice(&down_packed, "coop_u4_down")?;
    let scale_buf = executor.buffer_from_f32_as_bf16(&scales, "coop_u4_scales")?;
    let bias_buf = executor.buffer_from_f32_as_bf16(&biases, "coop_u4_biases")?;
    let down_scale_buf = executor.buffer_from_f32_as_bf16(&down_scales, "coop_u4_down_scales")?;
    let down_bias_buf = executor.buffer_from_f32_as_bf16(&down_biases, "coop_u4_down_biases")?;
    let tile_buf = executor.upload_u32_buffer(&tile_expert, "coop_u4_tile")?;
    let perm_buf = executor.upload_u32_buffer(&perm, "coop_u4_perm")?;
    let hidden_bf16 = executor.private_bf16_buffer(max_m * inter, "coop_u4_hidden")?;
    let out_buf = executor.uncached_f32_buffer(max_m * hidden, "coop_u4_out")?;

    let gate_dims = [max_m as u32, inter as u32, hidden as u32, top_k as u32];
    let down_dims = [max_m as u32, hidden as u32, inter as u32];
    let cb = executor.queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    let guard = EncoderEndGuard::new(enc);
    enc.set_compute_pipeline_state(&pso_gate_up);
    enc.set_buffer(0, Some(&input_buf), 0);
    enc.set_buffer(1, Some(&gate_buf), 0);
    enc.set_buffer(2, Some(&scale_buf), 0);
    enc.set_buffer(3, Some(&bias_buf), 0);
    enc.set_buffer(4, Some(&up_buf), 0);
    enc.set_buffer(5, Some(&scale_buf), 0);
    enc.set_buffer(6, Some(&bias_buf), 0);
    enc.set_buffer(7, Some(&hidden_bf16), 0);
    enc.set_buffer(8, Some(&tile_buf), 0);
    enc.set_buffer(9, Some(&perm_buf), 0);
    enc.set_bytes(10, 16, gate_dims.as_ptr().cast::<std::ffi::c_void>());
    enc.dispatch_thread_groups(
        MTLSize::new((max_m / 16) as u64, inter.div_ceil(32) as u64, 1),
        MTLSize::new(32, 1, 1),
    );
    enc.set_compute_pipeline_state(&pso_scatter);
    enc.set_buffer(0, Some(&hidden_bf16), 0);
    enc.set_buffer(1, Some(&down_buf), 0);
    enc.set_buffer(2, Some(&down_scale_buf), 0);
    enc.set_buffer(3, Some(&down_bias_buf), 0);
    enc.set_buffer(4, Some(&out_buf), 0);
    enc.set_buffer(5, Some(&tile_buf), 0);
    enc.set_buffer(6, Some(&perm_buf), 0);
    enc.set_bytes(7, 12, down_dims.as_ptr().cast::<std::ffi::c_void>());
    enc.dispatch_thread_groups(
        MTLSize::new((max_m / 16) as u64, hidden.div_ceil(32) as u64, 1),
        MTLSize::new(32, 1, 1),
    );
    guard.end();
    commit_and_wait(cb)?;
    let gpu = read_f32_buffer(&out_buf, max_m * hidden)?;

    let deq = |q: &[u8], e: usize, row: usize, col: usize, out_dim: usize, in_dim: usize| {
        let g = col / FAST_QMV_GROUP_SIZE;
        let s = scales[(e * out_dim + row) * groups + g];
        let b = biases[(e * out_dim + row) * groups + g];
        bf16_round(f32::from(q[(e * out_dim + row) * in_dim + col]) * s + b)
    };
    let deq_down = |e: usize, row: usize, col: usize| {
        let g = col / FAST_QMV_GROUP_SIZE;
        let s = down_scales[(e * hidden + row) * groups + g];
        let b = down_biases[(e * hidden + row) * groups + g];
        bf16_round(f32::from(q_down[(e * hidden + row) * inter + col]) * s + b)
    };
    let mut hidden_cpu = vec![0.0_f32; max_m * inter];
    for row in 0..max_m {
        let expert = tile_expert[row / 16] as usize;
        let token = perm[row] as usize / top_k;
        for col in 0..inter {
            let mut gate_acc = 0.0_f32;
            let mut up_acc = 0.0_f32;
            for kk in 0..hidden {
                let a = bf16_round(input[token * hidden + kk]);
                gate_acc += a * deq(&q_gate, expert, col, kk, inter, hidden);
                up_acc += a * deq(&q_up, expert, col, kk, inter, hidden);
            }
            hidden_cpu[row * inter + col] =
                bf16_round((gate_acc / (1.0 + (-gate_acc).exp())) * up_acc);
        }
    }
    let mut cpu = vec![0.0_f32; max_m * hidden];
    for row in 0..max_m {
        let expert = tile_expert[row / 16] as usize;
        let slot = perm[row] as usize;
        for out in 0..hidden {
            let mut acc = 0.0_f32;
            for kk in 0..inter {
                acc += hidden_cpu[row * inter + kk] * deq_down(expert, out, kk);
            }
            cpu[slot * hidden + out] = acc;
        }
    }
    assert_close_eps(&gpu, &cpu, 2.5e-1);
    Ok(())
}

/// D-30B — diagnostic : fused u4 gate/up/swiglu + scatter aux dims RÉELLES du
/// Qwen3-30B (inter=768, hidden=2048 → 32 groupes), vs oracle CPU f32. Le test
/// `..._matches_cpu` ne couvre que hidden=64 (1 groupe) ; celui-ci expose un
/// éventuel bug de group/stride u4 au vrai K.
#[test]
fn coop_qb_grouped_u4_gate_up_swiglu_scatter_matches_cpu_30b_scale() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let (Some(pso_gate_up), Some(pso_scatter)) = (
        executor.na_gemm_coop_qb_grouped_gate_up_swiglu_u4.clone(),
        executor.na_gemm_coop_qb_grouped_scatter_u4.clone(),
    ) else {
        eprintln!("skip: coop qb grouped u4 indisponible (macOS < 26 ?)");
        return Ok(());
    };

    let (experts, rows, top_k, inter, hidden) = (2_usize, 16_usize, 2_usize, 768_usize, 2048_usize);
    let max_m = rows * top_k;
    let groups = hidden / FAST_QMV_GROUP_SIZE;
    let tile_expert = vec![0_u32, 1_u32];
    let perm = (0..max_m as u32).collect::<Vec<_>>();
    let input = (0..rows * hidden)
        .map(|idx| ((idx % 29) as f32 - 14.0) * 0.02)
        .collect::<Vec<_>>();
    let q_gate = (0..experts * inter * hidden)
        .map(|idx| ((idx * 7 + 3) % 16) as u8)
        .collect::<Vec<_>>();
    let q_up = (0..experts * inter * hidden)
        .map(|idx| ((idx * 5 + 9) % 16) as u8)
        .collect::<Vec<_>>();
    let q_down = (0..experts * hidden * inter)
        .map(|idx| ((idx * 11 + 1) % 16) as u8)
        .collect::<Vec<_>>();
    let down_groups = inter / FAST_QMV_GROUP_SIZE;
    let scales = (0..experts * inter * groups)
        .map(|idx| bf16_round(0.002 + (idx % 5) as f32 * 0.0002))
        .collect::<Vec<_>>();
    let biases = (0..experts * inter * groups)
        .map(|idx| bf16_round(-0.03 + (idx % 7) as f32 * 0.004))
        .collect::<Vec<_>>();
    let down_scales = (0..experts * hidden * down_groups)
        .map(|idx| bf16_round(0.0015 + (idx % 3) as f32 * 0.0003))
        .collect::<Vec<_>>();
    let down_biases = (0..experts * hidden * down_groups)
        .map(|idx| bf16_round(-0.02 + (idx % 5) as f32 * 0.003))
        .collect::<Vec<_>>();

    let pack_u4 = |q: &[u8], experts: usize, out_dim: usize, in_dim: usize| {
        let packed_cols = in_dim / 8;
        let mut packed = vec![0_u32; experts * out_dim * packed_cols];
        for e in 0..experts {
            for row in 0..out_dim {
                for col in 0..in_dim {
                    let word = (e * out_dim + row) * packed_cols + col / 8;
                    packed[word] |=
                        u32::from(q[(e * out_dim + row) * in_dim + col]) << ((col % 8) * 4);
                }
            }
        }
        packed
    };
    let gate_packed = pack_u4(&q_gate, experts, inter, hidden);
    let up_packed = pack_u4(&q_up, experts, inter, hidden);
    let down_packed = pack_u4(&q_down, experts, hidden, inter);

    let input_buf = executor.upload_f32_buffer(&input, "coop_u4_30b_input")?;
    let gate_buf = executor.buffer_from_slice(&gate_packed, "coop_u4_30b_gate")?;
    let up_buf = executor.buffer_from_slice(&up_packed, "coop_u4_30b_up")?;
    let down_buf = executor.buffer_from_slice(&down_packed, "coop_u4_30b_down")?;
    let scale_buf = executor.buffer_from_f32_as_bf16(&scales, "coop_u4_30b_scales")?;
    let bias_buf = executor.buffer_from_f32_as_bf16(&biases, "coop_u4_30b_biases")?;
    let down_scale_buf = executor.buffer_from_f32_as_bf16(&down_scales, "coop_u4_30b_dscales")?;
    let down_bias_buf = executor.buffer_from_f32_as_bf16(&down_biases, "coop_u4_30b_dbiases")?;
    let tile_buf = executor.upload_u32_buffer(&tile_expert, "coop_u4_30b_tile")?;
    let perm_buf = executor.upload_u32_buffer(&perm, "coop_u4_30b_perm")?;
    let hidden_bf16 = executor.private_bf16_buffer(max_m * inter, "coop_u4_30b_hidden")?;
    let out_buf = executor.uncached_f32_buffer(max_m * hidden, "coop_u4_30b_out")?;

    let gate_dims = [max_m as u32, inter as u32, hidden as u32, top_k as u32];
    let down_dims = [max_m as u32, hidden as u32, inter as u32];
    let cb = executor.queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    let guard = EncoderEndGuard::new(enc);
    enc.set_compute_pipeline_state(&pso_gate_up);
    enc.set_buffer(0, Some(&input_buf), 0);
    enc.set_buffer(1, Some(&gate_buf), 0);
    enc.set_buffer(2, Some(&scale_buf), 0);
    enc.set_buffer(3, Some(&bias_buf), 0);
    enc.set_buffer(4, Some(&up_buf), 0);
    enc.set_buffer(5, Some(&scale_buf), 0);
    enc.set_buffer(6, Some(&bias_buf), 0);
    enc.set_buffer(7, Some(&hidden_bf16), 0);
    enc.set_buffer(8, Some(&tile_buf), 0);
    enc.set_buffer(9, Some(&perm_buf), 0);
    enc.set_bytes(10, 16, gate_dims.as_ptr().cast::<std::ffi::c_void>());
    enc.dispatch_thread_groups(
        MTLSize::new((max_m / 16) as u64, inter.div_ceil(32) as u64, 1),
        MTLSize::new(32, 1, 1),
    );
    enc.set_compute_pipeline_state(&pso_scatter);
    enc.set_buffer(0, Some(&hidden_bf16), 0);
    enc.set_buffer(1, Some(&down_buf), 0);
    enc.set_buffer(2, Some(&down_scale_buf), 0);
    enc.set_buffer(3, Some(&down_bias_buf), 0);
    enc.set_buffer(4, Some(&out_buf), 0);
    enc.set_buffer(5, Some(&tile_buf), 0);
    enc.set_buffer(6, Some(&perm_buf), 0);
    enc.set_bytes(7, 12, down_dims.as_ptr().cast::<std::ffi::c_void>());
    enc.dispatch_thread_groups(
        MTLSize::new((max_m / 16) as u64, hidden.div_ceil(32) as u64, 1),
        MTLSize::new(32, 1, 1),
    );
    guard.end();
    commit_and_wait(cb)?;
    let gpu = read_f32_buffer(&out_buf, max_m * hidden)?;

    let deq = |q: &[u8], e: usize, row: usize, col: usize, out_dim: usize, in_dim: usize| {
        let g = col / FAST_QMV_GROUP_SIZE;
        let s = scales[(e * out_dim + row) * groups + g];
        let b = biases[(e * out_dim + row) * groups + g];
        bf16_round(f32::from(q[(e * out_dim + row) * in_dim + col]) * s + b)
    };
    let deq_down = |e: usize, row: usize, col: usize| {
        let g = col / FAST_QMV_GROUP_SIZE;
        let s = down_scales[(e * hidden + row) * down_groups + g];
        let b = down_biases[(e * hidden + row) * down_groups + g];
        bf16_round(f32::from(q_down[(e * hidden + row) * inter + col]) * s + b)
    };
    let mut hidden_cpu = vec![0.0_f32; max_m * inter];
    for row in 0..max_m {
        let expert = tile_expert[row / 16] as usize;
        let token = perm[row] as usize / top_k;
        for col in 0..inter {
            let mut gate_acc = 0.0_f32;
            let mut up_acc = 0.0_f32;
            for kk in 0..hidden {
                let a = bf16_round(input[token * hidden + kk]);
                gate_acc += a * deq(&q_gate, expert, col, kk, inter, hidden);
                up_acc += a * deq(&q_up, expert, col, kk, inter, hidden);
            }
            hidden_cpu[row * inter + col] =
                bf16_round((gate_acc / (1.0 + (-gate_acc).exp())) * up_acc);
        }
    }
    let mut cpu = vec![0.0_f32; max_m * hidden];
    for row in 0..max_m {
        let expert = tile_expert[row / 16] as usize;
        let slot = perm[row] as usize;
        for out in 0..hidden {
            let mut acc = 0.0_f32;
            for kk in 0..inter {
                acc += hidden_cpu[row * inter + kk] * deq_down(expert, out, kk);
            }
            cpu[slot * hidden + out] = acc;
        }
    }
    let mut max_abs = 0.0_f32;
    let mut max_rel = 0.0_f32;
    for (g, c) in gpu.iter().zip(cpu.iter()) {
        let abs = (g - c).abs();
        max_abs = max_abs.max(abs);
        let denom = c.abs().max(1e-3);
        max_rel = max_rel.max(abs / denom);
    }
    eprintln!("coop_u4_30b_scale max_abs={max_abs:.6} max_rel={max_rel:.4}");
    // Tolérance bf16 tensor-core sur K=2048 : ~1e-2 rel attendu si correct.
    assert!(
        max_rel < 5.0e-2,
        "fused u4 grouped diverge de l'oracle f32 aux dims 30B: max_abs={max_abs} max_rel={max_rel}"
    );
    Ok(())
}

/// D-30B — chemin routed-only coop COMPLET (`encode_moe_routed_rows_coop`) avec
/// routage RÉEL (router → topk → grouping GPU → experts → weighted-sum) vs oracle
/// CPU bâti sur les indices/scores lus DEPUIS le GPU. Isole grouping + weighted-sum
/// (l'expert GEMM est déjà prouvé exact). Reproduit le bug 30B si présent.
#[test]
fn moe_routed_rows_coop_full_matches_cpu() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    if executor.na_gemm_coop_qb_grouped_gate_up_swiglu_u4.is_none() {
        eprintln!("skip: coop qb grouped u4 indisponible (macOS < 26 ?)");
        return Ok(());
    }
    // gate/up SwiGLU fusé + u4 requis par coop_compatible.
    let _fused = moe_coop_fused_swiglu_enabled();

    // Échelle 30B : 128 experts, top_k=8, ≥256 lignes → nombreux experts en
    // multi-tuiles inégales, le régime du run réel divergent. inter/hidden réduits
    // pour un oracle CPU rapide (le bug ne reproduit PAS ici : il vit dans
    // l'intégration résidente 48 couches, cf. RAPPORT_D30B).
    let (experts, rows, top_k, inter, hidden) =
        (128_usize, 256_usize, 8_usize, 128_usize, 256_usize);
    let n_check = rows;
    let total = rows * top_k;
    let groups_gu = hidden / FAST_QMV_GROUP_SIZE;
    let groups_dn = inter / FAST_QMV_GROUP_SIZE;

    let input: Vec<f32> = (0..rows * hidden)
        .map(|i| ((i % 23) as f32 - 11.0) * 0.03)
        .collect();
    // Router dense f32 [experts, hidden] : chaque token favorise des experts distincts.
    let router: Vec<f32> = (0..experts * hidden)
        .map(|i| {
            let e = i / hidden;
            let h = i % hidden;
            (((e * 13 + h * 7) % 19) as f32 - 9.0) * 0.02
        })
        .collect();

    let pack_u4 = |q: &[u8], e: usize, out_dim: usize, in_dim: usize| {
        let pc = in_dim / 8;
        let mut p = vec![0_u32; e * out_dim * pc];
        for ee in 0..e {
            for r in 0..out_dim {
                for c in 0..in_dim {
                    let w = (ee * out_dim + r) * pc + c / 8;
                    p[w] |= u32::from(q[(ee * out_dim + r) * in_dim + c]) << ((c % 8) * 4);
                }
            }
        }
        p
    };
    let q_gate: Vec<u8> = (0..experts * inter * hidden)
        .map(|i| ((i * 7 + 3) % 16) as u8)
        .collect();
    let q_up: Vec<u8> = (0..experts * inter * hidden)
        .map(|i| ((i * 5 + 9) % 16) as u8)
        .collect();
    let q_down: Vec<u8> = (0..experts * hidden * inter)
        .map(|i| ((i * 11 + 1) % 16) as u8)
        .collect();
    let s_gu: Vec<f32> = (0..experts * inter * groups_gu)
        .map(|i| bf16_round(0.002 + (i % 5) as f32 * 0.0002))
        .collect();
    let b_gu: Vec<f32> = (0..experts * inter * groups_gu)
        .map(|i| bf16_round(-0.03 + (i % 7) as f32 * 0.004))
        .collect();
    let s_dn: Vec<f32> = (0..experts * hidden * groups_dn)
        .map(|i| bf16_round(0.0015 + (i % 3) as f32 * 0.0003))
        .collect();
    let b_dn: Vec<f32> = (0..experts * hidden * groups_dn)
        .map(|i| bf16_round(-0.02 + (i % 5) as f32 * 0.003))
        .collect();

    let mk_stacked = |q: &[u8],
                      s: &[f32],
                      b: &[f32],
                      out_dim: usize,
                      in_dim: usize|
     -> Result<StackedAffineBuffers> {
        Ok(StackedAffineBuffers {
            packed: executor.buffer_from_slice(&pack_u4(q, experts, out_dim, in_dim), "st_p")?,
            scales: executor.buffer_from_f32_as_bf16(s, "st_s")?,
            biases: executor.buffer_from_f32_as_bf16(b, "st_b")?,
            experts,
            out_dim,
            in_dim,
            packed_cols: in_dim / 8,
            group_size: FAST_QMV_GROUP_SIZE,
            bits: FAST_QMV_BITS,
            groups: in_dim / FAST_QMV_GROUP_SIZE,
        })
    };
    let stacked = StackedMoeBuffers {
        gate: mk_stacked(&q_gate, &s_gu, &b_gu, inter, hidden)?,
        up: mk_stacked(&q_up, &s_gu, &b_gu, inter, hidden)?,
        down: mk_stacked(&q_down, &s_dn, &b_dn, hidden, inter)?,
    };
    let router_buf_w = executor.upload_f32_buffer(&router, "rt_router_w")?;
    let weights = MetalMoeRoutedWeights {
        router: MetalLinearWeightBuffers::Dense {
            rhs: router_buf_w,
            rhs_bf16: None,
            out_dim: experts,
            in_dim: hidden,
        },
        stacked,
    };

    let input_buf = executor.upload_f32_buffer(&input, "rt_input")?;
    let router_buf = executor.uncached_f32_buffer(rows * experts, "rt_router")?;
    let indices_buf = executor.uncached_u32_buffer(total, "rt_indices")?;
    let scores_buf = executor.uncached_f32_buffer(total, "rt_scores")?;
    let down_buf = executor.uncached_f32_buffer(total * hidden, "rt_down")?;
    let output_buf = executor.uncached_f32_buffer(rows * hidden, "rt_out")?;

    let cb = executor.queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    let guard = EncoderEndGuard::new(enc);
    let mut owned: Vec<metal::Buffer> = Vec::new();
    executor.encode_moe_routed_rows_coop(
        enc,
        &mut owned,
        &input_buf,
        None,
        &output_buf,
        &router_buf,
        &indices_buf,
        &scores_buf,
        &down_buf,
        rows,
        hidden,
        &weights,
        top_k,
    )?;
    guard.end();
    commit_and_wait(cb)?;
    let gpu = read_f32_buffer(&output_buf, rows * hidden)?;
    let indices = read_u32_buffer(&indices_buf, total)?;
    let scores = read_f32_buffer(&scores_buf, total)?;

    // Oracle CPU : experts sélectionnés par le GPU, arithmétique bf16 comme les kernels.
    let deq = |q: &[u8],
               s: &[f32],
               b: &[f32],
               e: usize,
               r: usize,
               c: usize,
               out_dim: usize,
               in_dim: usize,
               grp: usize| {
        let g = c / FAST_QMV_GROUP_SIZE;
        bf16_round(
            f32::from(q[(e * out_dim + r) * in_dim + c]) * s[(e * out_dim + r) * grp + g]
                + b[(e * out_dim + r) * grp + g],
        )
    };
    let mut cpu = vec![0.0_f32; n_check * hidden];
    for t in 0..n_check {
        for k in 0..top_k {
            let e = indices[t * top_k + k] as usize;
            let sc = scores[t * top_k + k];
            // hidden2 = silu(gate·x)·(up·x), en bf16.
            let mut h2 = vec![0.0_f32; inter];
            for (col, h2_value) in h2.iter_mut().enumerate().take(inter) {
                let mut ga = 0.0_f32;
                let mut ua = 0.0_f32;
                for kk in 0..hidden {
                    let a = bf16_round(input[t * hidden + kk]);
                    ga += a * deq(&q_gate, &s_gu, &b_gu, e, col, kk, inter, hidden, groups_gu);
                    ua += a * deq(&q_up, &s_gu, &b_gu, e, col, kk, inter, hidden, groups_gu);
                }
                *h2_value = bf16_round((ga / (1.0 + (-ga).exp())) * ua);
            }
            for o in 0..hidden {
                let mut acc = 0.0_f32;
                for (kk, &h2_value) in h2.iter().enumerate().take(inter) {
                    acc += h2_value
                        * deq(&q_down, &s_dn, &b_dn, e, o, kk, hidden, inter, groups_dn);
                }
                cpu[t * hidden + o] += sc * acc;
            }
        }
    }
    let mut max_abs = 0.0_f32;
    let mut max_rel = 0.0_f32;
    for (g, c) in gpu.iter().zip(cpu.iter()) {
        let abs = (g - c).abs();
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(abs / c.abs().max(1e-3));
    }
    eprintln!(
        "moe_routed_coop_full max_abs={max_abs:.6} max_rel={max_rel:.4} indices[..8]={:?}",
        &indices[..8.min(indices.len())]
    );
    assert!(
        max_rel < 5.0e-2,
        "routed coop complet diverge de l'oracle f32: max_abs={max_abs} max_rel={max_rel}"
    );
    Ok(())
}

/// D-COOP-2 — oracle GPU gather vs coop groupé aux dimensions 30B réelles
/// (`hidden=2048`, `inter=768`, `experts=128`, `top_k=8`). Le routeur nul force
/// les mêmes 8 experts pour chaque row : avec 24 rows, chaque expert sélectionné
/// reçoit >16 lignes, donc le chemin coop exerce plusieurs tiles et du padding
/// sentinelle. Le contrat est numérique, pas bit-identique : gather QMV par ligne
/// et coop groupé tensor-core n'ont pas le même ordre de réduction.
#[test]
fn moe_routed_rows_coop_matches_gather_30b_multitile() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    if executor.na_gemm_coop_qb_grouped_gate_up_swiglu_u4.is_none() {
        eprintln!("skip: coop qb grouped u4 indisponible (macOS < 26 ?)");
        return Ok(());
    }

    let (experts, rows, top_k, inter, hidden) =
        (128_usize, 24_usize, 8_usize, 768_usize, 2048_usize);
    let total = rows * top_k;

    let mut input = Vec::with_capacity(rows * hidden);
    for row in 0..rows {
        input.extend_from_slice(&varied_row(hidden, 200 + row));
    }
    let router = vec![0.0_f32; experts * hidden];
    let stacked = StackedMoeBuffers {
        gate: test_stacked_affine_varied_u4_group(&executor, experts, inter, hidden, 3)?,
        up: test_stacked_affine_varied_u4_group(&executor, experts, inter, hidden, 11)?,
        down: test_stacked_affine_varied_u4_group(&executor, experts, hidden, inter, 19)?,
    };
    let router_buf_w = executor.upload_f32_buffer(&router, "dcoop2_router_w")?;
    let weights = MetalMoeRoutedWeights {
        router: MetalLinearWeightBuffers::Dense {
            rhs: router_buf_w,
            rhs_bf16: None,
            out_dim: experts,
            in_dim: hidden,
        },
        stacked,
    };

    let input_buf = executor.upload_f32_buffer(&input, "dcoop2_input")?;
    let router_buf = executor.uncached_f32_buffer(rows * experts, "dcoop2_router")?;
    let indices_buf = executor.uncached_u32_buffer(total, "dcoop2_indices")?;
    let scores_buf = executor.uncached_f32_buffer(total, "dcoop2_scores")?;
    let coop_down = executor.uncached_f32_buffer(total * hidden, "dcoop2_coop_down")?;
    let coop_out = executor.uncached_f32_buffer(rows * hidden, "dcoop2_coop_out")?;

    let cb = executor.queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    let guard = EncoderEndGuard::new(enc);
    let mut owned = Vec::new();
    executor.encode_moe_routed_rows_coop(
        enc,
        &mut owned,
        &input_buf,
        None,
        &coop_out,
        &router_buf,
        &indices_buf,
        &scores_buf,
        &coop_down,
        rows,
        hidden,
        &weights,
        top_k,
    )?;
    guard.end();
    commit_and_wait(cb)?;

    let indices = read_u32_buffer(&indices_buf, total)?;
    let mut counts = vec![0_usize; experts];
    for expert in &indices {
        let idx = usize::try_from(*expert)
            .map_err(|_| InferError::Metal(format!("expert hors usize: {expert}")))?;
        counts[idx] += 1;
    }
    let selected = counts
        .iter()
        .filter(|count| **count > 0)
        .copied()
        .collect::<Vec<_>>();
    assert_eq!(
        selected.len(),
        top_k,
        "routeur nul attendu sur exactement top_k experts"
    );
    for count in selected {
        assert_eq!(
            count, rows,
            "chaque expert sélectionné reçoit une row par token"
        );
        assert!(count > 16, "le test doit forcer le régime multi-tile");
    }

    let gather_hidden = executor.private_f32_buffer(total * inter, "dcoop2_gather_hidden")?;
    let gather_down = executor.uncached_f32_buffer(total * hidden, "dcoop2_gather_down")?;
    let gather_out = executor.uncached_f32_buffer(rows * hidden, "dcoop2_gather_out")?;

    let cb = executor.queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    let guard = EncoderEndGuard::new(enc);
    let mut owned = Vec::new();
    let used_fused = executor.encode_gather_gate_up_swiglu(
        enc,
        &mut owned,
        &input_buf,
        rows,
        &weights.stacked.gate,
        &weights.stacked.up,
        &indices_buf,
        total,
        &gather_hidden,
    )?;
    assert!(
        used_fused,
        "gather gate/up u4 rapide attendu aux dimensions 30B"
    );
    executor.encode_gather_matmul(
        enc,
        &mut owned,
        &gather_hidden,
        total,
        &weights.stacked.down,
        &indices_buf,
        total,
        &gather_down,
    )?;
    executor.encode_weighted_sum_grouped_topk(
        enc,
        &mut owned,
        &gather_down,
        &scores_buf,
        &gather_out,
        rows,
        top_k,
        hidden,
    )?;
    guard.end();
    commit_and_wait(cb)?;

    let coop = read_f32_buffer(&coop_out, rows * hidden)?;
    let gather = read_f32_buffer(&gather_out, rows * hidden)?;
    let mut max_abs = 0.0_f32;
    let mut sum_abs = 0.0_f64;
    for (left, right) in coop.iter().zip(gather.iter()) {
        let abs = (left - right).abs();
        max_abs = max_abs.max(abs);
        sum_abs += f64::from(abs);
    }
    let mean_abs = sum_abs / coop.len() as f64;
    eprintln!("dcoop2 gather_vs_coop max_abs={max_abs:.6} mean_abs={mean_abs:.6}");
    assert!(
        max_abs < 1.0 && mean_abs < 5.0e-3,
        "routed coop 30B multi-tile hors classe numérique: max_abs={max_abs} mean_abs={mean_abs}"
    );
    Ok(())
}

/// Parité BIT-À-BIT u4 vs u8 des kernels MoE coop groupés, à poids logiques
/// identiques (q ∈ [0,15], scales/biases identiques). Les deux chemins dé-quantifient
/// `bfloat(q*s+b)` et alimentent la MÊME MMA tensor-core ; ils ne diffèrent que par
/// l'extraction du nibble et le stride packed (K/8 vs K/4). Le chemin u8 est le
/// référentiel byte-identique en prod (défaut ON depuis 94c5a89). Une égalité
/// bit-à-bit prouve que le port u4 (packing, ordre des nibbles, offset de groupe) est
/// correct — donc toute divergence de l'oracle greedy 4-bit vient de la précision
/// bf16/tensor-core structurelle, PAS d'un bug du kernel u4.
#[test]
fn coop_qb_grouped_u4_bit_identical_to_u8_same_weights() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let (Some(pso_gu_u4), Some(pso_sc_u4), Some(pso_gu_u8), Some(pso_sc_u8)) = (
        executor.na_gemm_coop_qb_grouped_gate_up_swiglu_u4.clone(),
        executor.na_gemm_coop_qb_grouped_scatter_u4.clone(),
        executor.na_gemm_coop_qb_grouped_gate_up_swiglu.clone(),
        executor.na_gemm_coop_qb_grouped_scatter.clone(),
    ) else {
        eprintln!("skip: kernels coop groupés indisponibles (macOS < 26 ?)");
        return Ok(());
    };

    let (experts, rows, top_k, inter, hidden) = (2_usize, 16_usize, 2_usize, 128_usize, 128_usize);
    let max_m = rows * top_k;
    let groups = hidden / FAST_QMV_GROUP_SIZE;
    let tile_expert = vec![0_u32, 1_u32];
    let perm = (0..max_m as u32).collect::<Vec<_>>();
    let input = (0..rows * hidden)
        .map(|idx| ((idx % 29) as f32 - 14.0) * 0.02)
        .collect::<Vec<_>>();
    // q ∈ [0,15] : valide pour u4 ET u8 (sous-ensemble), donc dé-quant identique.
    let q_gate = (0..experts * inter * hidden)
        .map(|idx| ((idx * 7 + 3) % 16) as u8)
        .collect::<Vec<_>>();
    let q_up = (0..experts * inter * hidden)
        .map(|idx| ((idx * 5 + 9) % 16) as u8)
        .collect::<Vec<_>>();
    let q_down = (0..experts * hidden * inter)
        .map(|idx| ((idx * 11 + 1) % 16) as u8)
        .collect::<Vec<_>>();
    let scales = (0..experts * inter * groups)
        .map(|idx| bf16_round(0.002 + (idx % 5) as f32 * 0.0002))
        .collect::<Vec<_>>();
    let biases = (0..experts * inter * groups)
        .map(|idx| bf16_round(-0.03 + (idx % 7) as f32 * 0.004))
        .collect::<Vec<_>>();
    let down_scales = (0..experts * hidden * groups)
        .map(|idx| bf16_round(0.0015 + (idx % 3) as f32 * 0.0003))
        .collect::<Vec<_>>();
    let down_biases = (0..experts * hidden * groups)
        .map(|idx| bf16_round(-0.02 + (idx % 5) as f32 * 0.003))
        .collect::<Vec<_>>();

    let pack_u4 = |q: &[u8], experts: usize, out_dim: usize, in_dim: usize| {
        let packed_cols = in_dim / 8;
        let mut packed = vec![0_u32; experts * out_dim * packed_cols];
        for e in 0..experts {
            for row in 0..out_dim {
                for col in 0..in_dim {
                    let word = (e * out_dim + row) * packed_cols + col / 8;
                    packed[word] |=
                        u32::from(q[(e * out_dim + row) * in_dim + col]) << ((col % 8) * 4);
                }
            }
        }
        packed
    };
    let pack_u8 = |q: &[u8], experts: usize, out_dim: usize, in_dim: usize| {
        let packed_cols = in_dim / 4;
        let mut packed = vec![0_u32; experts * out_dim * packed_cols];
        for e in 0..experts {
            for row in 0..out_dim {
                for col in 0..in_dim {
                    let word = (e * out_dim + row) * packed_cols + col / 4;
                    packed[word] |=
                        u32::from(q[(e * out_dim + row) * in_dim + col]) << ((col % 4) * 8);
                }
            }
        }
        packed
    };

    let input_buf = executor.upload_f32_buffer(&input, "coop_bid_input")?;
    let scale_buf = executor.buffer_from_f32_as_bf16(&scales, "coop_bid_scales")?;
    let bias_buf = executor.buffer_from_f32_as_bf16(&biases, "coop_bid_biases")?;
    let down_scale_buf = executor.buffer_from_f32_as_bf16(&down_scales, "coop_bid_dscales")?;
    let down_bias_buf = executor.buffer_from_f32_as_bf16(&down_biases, "coop_bid_dbiases")?;
    let tile_buf = executor.upload_u32_buffer(&tile_expert, "coop_bid_tile")?;
    let perm_buf = executor.upload_u32_buffer(&perm, "coop_bid_perm")?;

    let gate_dims = [max_m as u32, inter as u32, hidden as u32, top_k as u32];
    let down_dims = [max_m as u32, hidden as u32, inter as u32];

    // Un chemin (gate_up_swiglu -> scatter) paramétré par les PSOs et le packing.
    let run_chain = |gate_packed: &[u32],
                     up_packed: &[u32],
                     down_packed: &[u32],
                     pso_gu: &metal::ComputePipelineState,
                     pso_sc: &metal::ComputePipelineState,
                     tag: &'static str|
     -> Result<Vec<f32>> {
        let gate_buf = executor.buffer_from_slice(gate_packed, "coop_bid_gate")?;
        let up_buf = executor.buffer_from_slice(up_packed, "coop_bid_up")?;
        let down_buf = executor.buffer_from_slice(down_packed, "coop_bid_down")?;
        let hidden_bf16 = executor.private_bf16_buffer(max_m * inter, "coop_bid_hidden")?;
        let out_buf = executor.uncached_f32_buffer(max_m * hidden, tag)?;
        let cb = executor.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        let guard = EncoderEndGuard::new(enc);
        enc.set_compute_pipeline_state(pso_gu);
        enc.set_buffer(0, Some(&input_buf), 0);
        enc.set_buffer(1, Some(&gate_buf), 0);
        enc.set_buffer(2, Some(&scale_buf), 0);
        enc.set_buffer(3, Some(&bias_buf), 0);
        enc.set_buffer(4, Some(&up_buf), 0);
        enc.set_buffer(5, Some(&scale_buf), 0);
        enc.set_buffer(6, Some(&bias_buf), 0);
        enc.set_buffer(7, Some(&hidden_bf16), 0);
        enc.set_buffer(8, Some(&tile_buf), 0);
        enc.set_buffer(9, Some(&perm_buf), 0);
        enc.set_bytes(10, 16, gate_dims.as_ptr().cast::<std::ffi::c_void>());
        enc.dispatch_thread_groups(
            MTLSize::new((max_m / 16) as u64, inter.div_ceil(32) as u64, 1),
            MTLSize::new(32, 1, 1),
        );
        enc.set_compute_pipeline_state(pso_sc);
        enc.set_buffer(0, Some(&hidden_bf16), 0);
        enc.set_buffer(1, Some(&down_buf), 0);
        enc.set_buffer(2, Some(&down_scale_buf), 0);
        enc.set_buffer(3, Some(&down_bias_buf), 0);
        enc.set_buffer(4, Some(&out_buf), 0);
        enc.set_buffer(5, Some(&tile_buf), 0);
        enc.set_buffer(6, Some(&perm_buf), 0);
        enc.set_bytes(7, 12, down_dims.as_ptr().cast::<std::ffi::c_void>());
        enc.dispatch_thread_groups(
            MTLSize::new((max_m / 16) as u64, hidden.div_ceil(32) as u64, 1),
            MTLSize::new(32, 1, 1),
        );
        guard.end();
        commit_and_wait(cb)?;
        read_f32_buffer(&out_buf, max_m * hidden)
    };

    let out_u4 = run_chain(
        &pack_u4(&q_gate, experts, inter, hidden),
        &pack_u4(&q_up, experts, inter, hidden),
        &pack_u4(&q_down, experts, hidden, inter),
        &pso_gu_u4,
        &pso_sc_u4,
        "coop_bid_out_u4",
    )?;
    let out_u8 = run_chain(
        &pack_u8(&q_gate, experts, inter, hidden),
        &pack_u8(&q_up, experts, inter, hidden),
        &pack_u8(&q_down, experts, hidden, inter),
        &pso_gu_u8,
        &pso_sc_u8,
        "coop_bid_out_u8",
    )?;

    assert_eq!(out_u4.len(), out_u8.len());
    for (idx, (a, b)) in out_u4.iter().zip(out_u8.iter()).enumerate() {
        assert!(
            a.to_bits() == b.to_bits(),
            "divergence bit-à-bit u4/u8 index={idx} u4={a} u8={b}"
        );
    }
    Ok(())
}

/// Parité BIT-À-BIT u4 vs u8 du GEMM NA tuilé dense (`gemm_nax_coop_qb_tiled*`),
/// à poids logiques identiques (q ∈ [0,15], scales/biases identiques). Le chemin u4
/// dense est DÉFAUT ON (RETI_RUST_QMM_NA_FUSED_TILED_U4) ; ce test prouve que le
/// dé-paquetage nibble/stride u4 est correct (aucun bug), donc sa sensibilité aux
/// near-ties de l'oracle greedy est PUREMENT la précision bf16 tensor-core, la même
/// que le chemin u8 promu en prod (6156f29).
#[test]
fn coop_qb_tiled_u4_bit_identical_to_u8_same_weights() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let (Some(pso_u4), Some(pso_u8)) = (
        executor.na_gemm_coop_qb_tiled_u4.clone(),
        executor.na_gemm_coop_qb_tiled.clone(),
    ) else {
        eprintln!("skip: coop qb tiled indisponible (macOS < 26 ?)");
        return Ok(());
    };
    let (m, n, k) = (65_usize, 128_usize, 512_usize);
    let groups = k / FAST_QMV_GROUP_SIZE;
    let lhs = (0..m * k)
        .map(|idx| ((idx % 23) as f32 - 11.0) * 0.017)
        .collect::<Vec<_>>();
    // q ∈ [0,15] : valide pour u4 ET u8, donc dé-quant bf16 identique des deux côtés.
    let q = (0..n * k)
        .map(|idx| ((idx * 7 + 3) % 16) as u8)
        .collect::<Vec<_>>();
    let scales = (0..n * groups)
        .map(|idx| bf16_round(0.002 + (idx % 5) as f32 * 0.0002))
        .collect::<Vec<_>>();
    let biases = (0..n * groups)
        .map(|idx| bf16_round(-0.03 + (idx % 7) as f32 * 0.004))
        .collect::<Vec<_>>();
    let mut packed_u4 = vec![0_u32; n * (k / 8)];
    let mut packed_u8 = vec![0_u32; n * (k / 4)];
    for nn in 0..n {
        for kk in 0..k {
            packed_u4[nn * (k / 8) + kk / 8] |= u32::from(q[nn * k + kk]) << ((kk % 8) * 4);
            packed_u8[nn * (k / 4) + kk / 4] |= u32::from(q[nn * k + kk]) << ((kk % 4) * 8);
        }
    }
    let a_f32 = executor.upload_f32_buffer(&lhs, "tiled_bid_a")?;
    let scale_buf = executor.buffer_from_f32_as_bf16(&scales, "tiled_bid_s")?;
    let bias_buf = executor.buffer_from_f32_as_bf16(&biases, "tiled_bid_b")?;
    let a_bf16 = executor.private_bf16_buffer(m * k, "tiled_bid_a_bf16")?;
    let mnk = [m as u32, n as u32, k as u32];
    let run = |packed: &[u32],
               pso: &metal::ComputePipelineState,
               tag: &'static str|
     -> Result<Vec<f32>> {
        let packed_buf = executor.buffer_from_slice(packed, "tiled_bid_packed")?;
        let out = executor.uncached_f32_buffer(m * n, tag)?;
        let cb = executor.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        let guard = EncoderEndGuard::new(enc);
        executor.encode_f32_to_bf16(enc, &a_f32, &a_bf16, m * k)?;
        enc.set_compute_pipeline_state(pso);
        enc.set_buffer(0, Some(&a_bf16), 0);
        enc.set_buffer(1, Some(&packed_buf), 0);
        enc.set_buffer(2, Some(&scale_buf), 0);
        enc.set_buffer(3, Some(&bias_buf), 0);
        enc.set_buffer(4, Some(&out), 0);
        enc.set_bytes(5, 12, mnk.as_ptr().cast::<std::ffi::c_void>());
        let width = pso.thread_execution_width().max(1);
        enc.dispatch_thread_groups(
            MTLSize::new(m.div_ceil(64) as u64, (n / 64) as u64, 1),
            MTLSize::new(width * 4, 1, 1),
        );
        guard.end();
        commit_and_wait(cb)?;
        read_f32_buffer(&out, m * n)
    };
    let out_u4 = run(&packed_u4, &pso_u4, "tiled_bid_out_u4")?;
    let out_u8 = run(&packed_u8, &pso_u8, "tiled_bid_out_u8")?;
    assert_eq!(out_u4.len(), out_u8.len());
    for (idx, (a, b)) in out_u4.iter().zip(out_u8.iter()).enumerate() {
        assert!(
            a.to_bits() == b.to_bits(),
            "divergence bit-à-bit tuilé u4/u8 index={idx} u4={a} u8={b}"
        );
    }
    Ok(())
}
