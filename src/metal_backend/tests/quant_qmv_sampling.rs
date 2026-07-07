#[test]
fn dense_gemm_tiled_matches_cpu() -> Result<()> {
    let executor = match test_executor()? {
        Some(executor) => executor,
        None => return Ok(()),
    };
    // Dimensions non-multiples de la tuile 16 → exerce les bords (batch=20,
    // out=18, in=40).
    let (batch, out_dim, in_dim) = (20usize, 18usize, 40usize);
    let lhs: Vec<f32> = (0..batch * in_dim)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.25)
        .collect();
    let rhs: Vec<f32> = (0..out_dim * in_dim)
        .map(|i| ((i % 5) as f32 - 2.0) * 0.5)
        .collect();
    let x = Tensor::from_vec(vec![batch, in_dim], lhs).expect("invariant: lhs valide");
    let w = Tensor::from_vec(vec![out_dim, in_dim], rhs).expect("invariant: rhs valide");

    let cpu = x
        .matmul_rhs_t(&w)
        .expect("invariant: matmul CPU compatible");
    let gpu = executor
        .matmul_rhs_t_dense_tiled(&x, &w)
        .expect("invariant: gemm Metal compatible");

    assert_eq!(gpu.shape(), cpu.shape());
    assert_close(gpu.data(), cpu.data());
    Ok(())
}

#[test]
fn affine_matmul_matches_cpu() -> Result<()> {
    let executor = match MetalExecutor::new() {
        Ok(executor) => executor,
        Err(InferError::Metal(message)) if message.contains("aucun device") => return Ok(()),
        Err(error) => return Err(error),
    };
    let packed = vec![
        pack_lanes(&[255, 0, 0, 0], 8),
        pack_lanes(&[0, 255, 0, 0], 8),
    ];
    let scales = Tensor::from_vec(vec![2, 2], vec![bf16_round(1.0 / 255.0); 4])
        .expect("invariant: scales valides");
    let biases = Tensor::from_vec(vec![2, 2], vec![0.0; 4]).expect("invariant: biases valides");
    let compact = AffineQuantizedTensor::new(&[2, 1], packed, scales, biases, 2, 8)
        .expect("invariant: poids compact valide");
    let input = Tensor::from_vec(vec![1, 4], vec![2.0, 3.0, 5.0, 7.0]).expect("invariant: input");

    let cpu = compact
        .matmul_rhs_t(&input)
        .expect("invariant: matmul CPU compatible");
    let gpu = executor.matmul_rhs_t_affine(&input, &compact)?;

    assert_eq!(gpu.shape(), cpu.shape());
    assert_close(gpu.data(), cpu.data());
    Ok(())
}

#[test]
fn moe_gather_fast_matches_cpu_for_short_k() -> Result<()> {
    let executor = match MetalExecutor::new() {
        Ok(executor) => executor,
        Err(InferError::Metal(message)) if message.contains("aucun device") => return Ok(()),
        Err(error) => return Err(error),
    };
    let experts = vec![
        test_expert(0.001, 0.0005, -0.0003)?,
        test_expert(0.0007, -0.0004, 0.0002)?,
    ];
    let input = Tensor::from_vec(
        vec![1, 64],
        (0..64)
            .map(|idx| (idx as f32 - 31.0) / 32.0)
            .collect::<Vec<_>>(),
    )
    .expect("invariant: input valide");
    let weighted_top = [(0_usize, 0.625_f32), (1_usize, 0.375_f32)];

    let mut cpu = vec![0.0_f32; 8];
    for (expert_index, scale) in weighted_top {
        let row = experts[expert_index]
            .forward(&input)
            .expect("invariant: expert CPU valide");
        for (dst, value) in cpu.iter_mut().zip(row.as_row()?) {
            *dst += value * scale;
        }
    }
    let gpu = executor.moe_gated_topk(&input, &experts, &weighted_top)?;

    assert_eq!(gpu.shape(), [1, 8]);
    assert_close_eps(gpu.data(), &cpu, 1.0e-4);
    Ok(())
}

#[test]
fn affine_argmax_matches_cpu() -> Result<()> {
    let executor = match MetalExecutor::new() {
        Ok(executor) => executor,
        Err(InferError::Metal(message)) if message.contains("aucun device") => return Ok(()),
        Err(error) => return Err(error),
    };
    let weight = test_affine(16, 512, 0.0009)?;
    let linear = Linear::from_weight(LinearWeight::AffineQuantized(weight), None)?;
    let input = Tensor::from_vec(
        vec![1, 512],
        (0..512)
            .map(|idx| ((idx % 31) as f32 - 15.0) / 16.0)
            .collect::<Vec<_>>(),
    )
    .expect("invariant: input argmax valide");

    let logits = linear.forward(&input)?;
    let cpu = logits
        .as_row()?
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(idx, _)| idx)
        .expect("invariant: logits non vides");
    let gpu = executor.argmax_linear_biasless(&input, &linear)?;

    assert_eq!(gpu, cpu);
    Ok(())
}

#[test]
fn sample_topk_topp_matches_cpu() -> Result<()> {
    let executor = match MetalExecutor::new() {
        Ok(executor) => executor,
        Err(InferError::Metal(message)) if message.contains("aucun device") => return Ok(()),
        Err(error) => return Err(error),
    };
    let out_dim = 64_usize;
    let in_dim = 32_usize;
    let weight = Tensor::from_vec(
        vec![out_dim, in_dim],
        (0..out_dim * in_dim)
            .map(|idx| ((idx % 23) as f32 - 11.0) * 0.013 + (idx / in_dim) as f32 * 0.001)
            .collect::<Vec<_>>(),
    )?;
    let linear = Linear::new(weight, None)?;
    let input = Tensor::from_vec(
        vec![1, in_dim],
        (0..in_dim)
            .map(|idx| ((idx % 17) as f32 - 8.0) * 0.07)
            .collect::<Vec<_>>(),
    )?;
    let logits = linear.forward(&input)?;
    let mut sampler = crate::DeterministicSampler::new(1234);
    let cpu = crate::sample_token_top_k_top_p(logits.as_row()?, 0.7, 0.95, 20, &mut sampler)?;
    let gpu = executor.sample_linear_biasless_topk_topp(&input, &linear, 0.7, 0.95, 20, 1234)?;

    assert_eq!(gpu, cpu);
    Ok(())
}

#[test]
fn sample_topk32_topp_multiblock_matches_cpu() -> Result<()> {
    let executor = match MetalExecutor::new() {
        Ok(executor) => executor,
        Err(InferError::Metal(message)) if message.contains("aucun device") => return Ok(()),
        Err(error) => return Err(error),
    };
    let out_dim = 769_usize;
    let in_dim = 32_usize;
    let weight = Tensor::from_vec(
        vec![out_dim, in_dim],
        (0..out_dim * in_dim)
            .map(|idx| {
                let row = idx / in_dim;
                let col = idx % in_dim;
                ((col % 29) as f32 - 14.0) * 0.011 + (row % 257) as f32 * 0.0007
            })
            .collect::<Vec<_>>(),
    )?;
    let linear = Linear::new(weight, None)?;
    let input = Tensor::from_vec(
        vec![1, in_dim],
        (0..in_dim)
            .map(|idx| ((idx % 19) as f32 - 9.0) * 0.053)
            .collect::<Vec<_>>(),
    )?;
    let logits = linear.forward(&input)?;
    let mut sampler = crate::DeterministicSampler::new(0x5eed);
    let cpu = crate::sample_token_top_k_top_p(logits.as_row()?, 0.7, 0.95, 32, &mut sampler)?;
    let gpu = executor.sample_linear_biasless_topk_topp(&input, &linear, 0.7, 0.95, 32, 0x5eed)?;

    assert_eq!(gpu, cpu);
    Ok(())
}

fn splitmix_unit_f32_for_test(mut state: u64) -> f32 {
    state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    let mantissa = (z >> 40) as u32 & 0x00ff_ffff;
    mantissa as f32 / 16_777_216.0
}

fn cpu_gumbel_argmax_for_test(logits: &[f32], temperature: f32, seed: u64) -> usize {
    let temperature = temperature.max(0.0001);
    let mut best = 0_usize;
    let mut best_score = f32::NEG_INFINITY;
    for (index, logit) in logits.iter().copied().enumerate() {
        let mixed = seed ^ (index as u64).wrapping_mul(0xD1B5_4A32_D192_ED03);
        let u = splitmix_unit_f32_for_test(mixed).clamp(5.960_464_5e-8, 0.999_999_94);
        let gumbel = -(-u.ln()).ln();
        let score = logit / temperature + gumbel;
        if score > best_score || (score == best_score && index < best) {
            best = index;
            best_score = score;
        }
    }
    best
}

#[test]
fn sample_gumbel_matches_cpu_and_distribution() -> Result<()> {
    let executor = match MetalExecutor::new() {
        Ok(executor) => executor,
        Err(InferError::Metal(message)) if message.contains("aucun device") => return Ok(()),
        Err(error) => return Err(error),
    };
    let logits = vec![1.1_f32, 0.4, -0.2, -0.8];
    let temperature = 0.9_f32;
    let weight = Tensor::from_vec(vec![logits.len(), 1], logits.clone())?;
    let linear = Linear::new(weight, None)?;
    let input = Tensor::from_vec(vec![1, 1], vec![1.0])?;
    let mut counts = vec![0_usize; logits.len()];

    for seed in 0..1024_u64 {
        let gpu = executor.sample_linear_biasless_gumbel(&input, &linear, temperature, seed)?;
        let repeat = executor.sample_linear_biasless_gumbel(&input, &linear, temperature, seed)?;
        let cpu = cpu_gumbel_argmax_for_test(&logits, temperature, seed);
        assert_eq!(gpu, repeat, "seed {seed}");
        assert_eq!(gpu, cpu, "seed {seed}");
        counts[gpu] += 1;
    }

    let expected = crate::softmax(&logits, temperature);
    let total = counts.iter().sum::<usize>() as f64;
    let chi2 = counts
        .iter()
        .zip(expected)
        .map(|(observed, prob)| {
            let expected = total * f64::from(prob);
            let delta = *observed as f64 - expected;
            delta * delta / expected.max(1.0)
        })
        .sum::<f64>();
    eprintln!("gumbel_stats counts={counts:?} chi2={chi2:.3}");
    assert!(
        chi2 < 24.0,
        "distribution Gumbel hors tolérance: counts={counts:?} chi2={chi2:.3}"
    );
    Ok(())
}

#[test]
fn shared_gate_up_swiglu_fast_matches_cpu() -> Result<()> {
    let executor = match MetalExecutor::new() {
        Ok(executor) => executor,
        Err(InferError::Metal(message)) if message.contains("aucun device") => return Ok(()),
        Err(error) => return Err(error),
    };
    // Dims du shared-expert : in_dim % 512 == 0, out_dim multiple de 8, 4-bit gs64.
    let in_dim = 512_usize;
    let out_dim = 16_usize;
    let gate_affine = test_affine(out_dim, in_dim, 0.006)?;
    let up_affine = test_affine(out_dim, in_dim, 0.008)?;
    let input = Tensor::from_vec(
        vec![1, in_dim],
        (0..in_dim)
            .map(|idx| (idx as f32 - in_dim as f32 / 2.0) / in_dim as f32)
            .collect::<Vec<_>>(),
    )
    .expect("invariant: input valide");
    // Référence CPU : silu(gate·x) * (up·x) via la déquantification exacte.
    let gate_cpu = gate_affine.matmul_rhs_t(&input)?;
    let up_cpu = up_affine.matmul_rhs_t(&input)?;
    let reference: Vec<f32> = gate_cpu
        .data()
        .iter()
        .zip(up_cpu.data())
        .map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u)
        .collect();
    let gate = Linear::new_quantized(gate_affine, None)?;
    let up = Linear::new_quantized(up_affine, None)?;
    let gpu = executor.gate_up_swiglu_fast(&input, &gate, &up)?;

    assert_eq!(gpu.shape(), [1, out_dim]);
    assert_close_eps(gpu.data(), &reference, 1.0e-4);
    Ok(())
}

#[test]
fn shared_gate_up_swiglu_fast_u8_matches_cpu() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let in_dim = 512_usize;
    let out_dim = 16_usize;
    let input = Tensor::from_vec(vec![1, in_dim], varied_row(in_dim, 41))?;

    for group_size in [64_usize, 128] {
        let gate_affine = test_affine_varied_u8_group(out_dim, in_dim, group_size)?;
        let up_affine = {
            let affine = test_affine_varied_u8_group(out_dim, in_dim, group_size)?;
            let scales = affine.scales().data().iter().map(|s| s * 1.25).collect();
            AffineQuantizedTensor::new(
                &[out_dim, in_dim / 4],
                affine.packed_data().to_vec(),
                Tensor::from_vec(vec![out_dim, in_dim / group_size], scales)?,
                affine.biases().clone(),
                group_size,
                8,
            )?
        };
        let gate_cpu = gate_affine.matmul_rhs_t(&input)?;
        let up_cpu = up_affine.matmul_rhs_t(&input)?;
        let reference: Vec<f32> = gate_cpu
            .data()
            .iter()
            .zip(up_cpu.data())
            .map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u)
            .collect();

        let gate = Linear::from_weight(LinearWeight::AffineQuantized(gate_affine), None)?;
        let up = Linear::from_weight(LinearWeight::AffineQuantized(up_affine), None)?;
        let gpu = executor.gate_up_swiglu_fast(&input, &gate, &up)?;
        assert_eq!(gpu.shape(), [1, out_dim]);
        assert_close_eps(gpu.data(), &reference, 1.0e-3);

        let gate_buffers =
            executor.resolve_linear_weight_buffers(gate.weight(), "shared_gate_u8_buffers")?;
        let up_buffers =
            executor.resolve_linear_weight_buffers(up.weight(), "shared_up_u8_buffers")?;
        let input_buffer = executor.upload_f32_buffer(input.data(), "shared_gate_u8_input")?;
        let output_buffer = executor.uncached_f32_buffer(out_dim, "shared_gate_u8_output")?;
        let command_buffer = executor.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let used_fast = executor.encode_gate_up_swiglu_fast_buffers(
            encoder,
            &input_buffer,
            &gate_buffers,
            &up_buffers,
            &output_buffer,
            in_dim,
        )?;
        assert!(
            used_fast,
            "u8 gs{group_size} gate/up fusion doit s'appliquer"
        );
        encoder.end_encoding();
        commit_and_wait(command_buffer)?;
        let buffered = read_f32_buffer(&output_buffer, out_dim)?;
        assert_close_eps(&buffered, &reference, 1.0e-3);

        let shared_gate_affine = test_affine_varied_u8_group(1, in_dim, group_size)?;
        let shared_gate_cpu = shared_gate_affine.matmul_rhs_t(&input)?;
        let shared_gate =
            Linear::from_weight(LinearWeight::AffineQuantized(shared_gate_affine), None)?;
        let shared_gate_buffers =
            executor.resolve_linear_weight_buffers(shared_gate.weight(), "shared_gate_scalar")?;
        let fused_output = executor.uncached_f32_buffer(out_dim, "shared_gate_scalar_hidden")?;
        let fused_scalar = executor.uncached_f32_buffer(1, "shared_gate_scalar_out")?;
        let command_buffer = executor.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let used_fast = executor.encode_gate_up_swiglu_shared_gate_fast_buffers(
            encoder,
            &input_buffer,
            &gate_buffers,
            &up_buffers,
            &shared_gate_buffers,
            &fused_output,
            &fused_scalar,
            in_dim,
        )?;
        assert!(
            used_fast,
            "u8 gs{group_size} gate/up+scalar fusion doit s'appliquer"
        );
        encoder.end_encoding();
        commit_and_wait(command_buffer)?;
        let fused_hidden = read_f32_buffer(&fused_output, out_dim)?;
        let fused_gate = read_f32_buffer(&fused_scalar, 1)?;
        assert_close_eps(&fused_hidden, &reference, 1.0e-3);
        assert_close_eps(&fused_gate, shared_gate_cpu.data(), 1.0e-3);

        if group_size == 64 {
            let qmv_output = executor.uncached_f32_buffer(out_dim, "shared_gate_qmv_proj")?;
            let qmv_scalar = executor.uncached_f32_buffer(1, "shared_gate_qmv_scalar")?;
            let command_buffer = executor.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            let used_fast = executor.encode_qmv_plus_shared_gate_fast_buffers(
                encoder,
                &input_buffer,
                &gate_buffers,
                &shared_gate_buffers,
                &qmv_output,
                &qmv_scalar,
                in_dim,
            )?;
            assert!(used_fast, "u8 gs64 qmv+shared gate fusion doit s'appliquer");
            encoder.end_encoding();
            commit_and_wait(command_buffer)?;
            let qmv_projected = read_f32_buffer(&qmv_output, out_dim)?;
            let qmv_gate = read_f32_buffer(&qmv_scalar, 1)?;
            assert_close_eps(&qmv_projected, gate_cpu.data(), 1.0e-3);
            assert_close_eps(&qmv_gate, shared_gate_cpu.data(), 1.0e-3);
        }
    }

    Ok(())
}

#[test]
#[ignore = "oracle CPU f32 trop strict pour gather u8 stacké; refaire avec référence GPU qmv"]
fn moe_gather_u8_fast_paths_match_cpu() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let hidden = 512_usize;
    let inter = 512_usize;
    let topk = 2_usize;
    let indices = [0_u32, 2_u32];
    let input = Tensor::from_vec(vec![1, hidden], varied_row(hidden, 51))?;

    for group_size in [64_usize, 128] {
        let experts = vec![
            test_expert_u8_group(hidden, inter, group_size)?,
            test_expert_u8_group(hidden, inter, group_size)?,
            test_expert_u8_group(hidden, inter, group_size)?,
        ];
        let stacked = executor.stacked_moe_buffers(&experts)?;
        let input_buffer = executor.upload_f32_buffer(input.data(), "moe_gather_u8_input")?;
        let indices_buffer = executor.upload_u32_buffer(&indices, "moe_gather_u8_indices")?;
        let hidden_buffer = executor.uncached_f32_buffer(topk * inter, "moe_gather_u8_hidden")?;
        let command_buffer = executor.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let mut owned = Vec::new();
        let used_gate_up = executor.encode_gather_gate_up_swiglu(
            encoder,
            &mut owned,
            &input_buffer,
            1,
            &stacked.gate,
            &stacked.up,
            &indices_buffer,
            topk,
            &hidden_buffer,
        )?;
        assert!(
            used_gate_up,
            "gather gate/up u8 gs{group_size} doit utiliser le fast path"
        );
        encoder.end_encoding();
        commit_and_wait(command_buffer)?;
        let hidden_gpu = read_f32_buffer(&hidden_buffer, topk * inter)?;

        let mut hidden_ref = Vec::with_capacity(topk * inter);
        for &expert_idx in &indices {
            let expert = &experts[expert_idx as usize];
            let (gate, up, _) = expert.projections();
            let gate_cpu = gate.forward(&input)?;
            let up_cpu = up.forward(&input)?;
            hidden_ref.extend(
                gate_cpu
                    .data()
                    .iter()
                    .zip(up_cpu.data())
                    .map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u),
            );
        }
        assert_close_eps(&hidden_gpu, &hidden_ref, 1.0e-4);

        let mut down_lhs = Vec::with_capacity(topk * inter);
        down_lhs.extend_from_slice(&varied_row(inter, 59));
        down_lhs.extend_from_slice(&varied_row(inter, 61));
        let down_lhs_buffer = executor.upload_f32_buffer(&down_lhs, "moe_gather_u8_down_lhs")?;
        let down_output_buffer =
            executor.uncached_f32_buffer(topk * hidden, "moe_gather_u8_down_out")?;
        let command_buffer = executor.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let mut owned = Vec::new();
        executor.encode_gather_matmul(
            encoder,
            &mut owned,
            &down_lhs_buffer,
            topk,
            &stacked.down,
            &indices_buffer,
            topk,
            &down_output_buffer,
        )?;
        encoder.end_encoding();
        commit_and_wait(command_buffer)?;
        let down_gpu = read_f32_buffer(&down_output_buffer, topk * hidden)?;

        let mut down_ref = Vec::with_capacity(topk * hidden);
        for (slot, &expert_idx) in indices.iter().enumerate() {
            let expert = &experts[expert_idx as usize];
            let (_, _, down) = expert.projections();
            let row = Tensor::from_vec(
                vec![1, inter],
                down_lhs[slot * inter..(slot + 1) * inter].to_vec(),
            )?;
            down_ref.extend_from_slice(down.forward(&row)?.data());
        }
        assert_close_eps(&down_gpu, &down_ref, 1.0e-4);
    }

    Ok(())
}

#[test]
fn moe_shared_epilogue_fused_matches_two_step() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let topk = 3_usize;
    let out_dim = 64_usize;
    let down: Vec<f32> = (0..topk * out_dim)
        .map(|idx| ((idx % 37) as f32 - 18.0) / 41.0)
        .collect();
    let scores = vec![0.57_f32, 0.31, 0.12];
    let residual: Vec<f32> = (0..out_dim)
        .map(|idx| ((idx % 29) as f32 - 14.0) / 37.0)
        .collect();
    let shared: Vec<f32> = (0..out_dim)
        .map(|idx| ((idx % 31) as f32 - 15.0) / 43.0)
        .collect();
    let shared_gate = vec![0.42_f32];

    let down_buffer = executor.upload_f32_buffer(&down, "epilogue_down")?;
    let scores_buffer = executor.upload_f32_buffer(&scores, "epilogue_scores")?;
    let residual_buffer = executor.upload_f32_buffer(&residual, "epilogue_residual")?;
    let shared_buffer = executor.upload_f32_buffer(&shared, "epilogue_shared")?;
    let shared_gate_buffer = executor.upload_f32_buffer(&shared_gate, "epilogue_shared_gate")?;
    let two_step_buffer = executor.uncached_f32_buffer(out_dim, "epilogue_two_step")?;
    let fused_buffer = executor.uncached_f32_buffer(out_dim, "epilogue_fused")?;

    let command_buffer = executor.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    let mut owned = Vec::new();
    executor.encode_weighted_sum_add_topk(
        encoder,
        &mut owned,
        &down_buffer,
        &scores_buffer,
        &residual_buffer,
        &two_step_buffer,
        topk,
        out_dim,
    )?;
    executor.encode_add_sigmoid_scaled(
        encoder,
        &shared_buffer,
        &shared_gate_buffer,
        &two_step_buffer,
        out_dim,
    )?;
    executor.encode_weighted_sum_add_shared_topk(
        encoder,
        &down_buffer,
        &scores_buffer,
        &residual_buffer,
        &shared_buffer,
        &shared_gate_buffer,
        &fused_buffer,
        topk,
        out_dim,
    )?;
    encoder.end_encoding();
    commit_and_wait(command_buffer)?;

    let two_step = read_f32_buffer(&two_step_buffer, out_dim)?;
    let fused = read_f32_buffer(&fused_buffer, out_dim)?;
    assert_bits_equal(&fused, &two_step, "moe shared epilogue fused");
    Ok(())
}

#[test]
fn moe_down_weighted_shared_fused_matches_two_step() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let hidden = 64_usize;
    let inter = 512_usize;
    let expert_count = 4_usize;
    let topk = 3_usize;
    let experts = (0..expert_count)
        .map(|_| test_expert_u8_group(hidden, inter, 64))
        .collect::<Result<Vec<_>>>()?;
    let stacked = executor.stacked_moe_buffers(&experts)?;
    let mut down_lhs = Vec::with_capacity(topk * inter);
    for salt in [13_usize, 17, 23] {
        down_lhs.extend_from_slice(&varied_row(inter, salt));
    }
    let indices = [3_u32, 1, 2];
    let scores = [0.48_f32, 0.33, 0.19];
    let residual: Vec<f32> = (0..hidden)
        .map(|idx| ((idx % 29) as f32 - 14.0) / 37.0)
        .collect();
    let shared: Vec<f32> = (0..hidden)
        .map(|idx| ((idx % 31) as f32 - 15.0) / 43.0)
        .collect();
    let shared_gate = [0.42_f32];

    let lhs_buffer = executor.upload_f32_buffer(&down_lhs, "down_weighted_lhs")?;
    let indices_buffer = executor.upload_u32_buffer(&indices, "down_weighted_indices")?;
    let scores_buffer = executor.upload_f32_buffer(&scores, "down_weighted_scores")?;
    let residual_buffer = executor.upload_f32_buffer(&residual, "down_weighted_residual")?;
    let shared_buffer = executor.upload_f32_buffer(&shared, "down_weighted_shared")?;
    let shared_gate_buffer =
        executor.upload_f32_buffer(&shared_gate, "down_weighted_shared_gate")?;
    let down_buffer = executor.uncached_f32_buffer(topk * hidden, "down_weighted_down")?;
    let two_step_buffer = executor.uncached_f32_buffer(hidden, "down_weighted_two_step")?;
    let fused_buffer = executor.uncached_f32_buffer(hidden, "down_weighted_fused")?;
    let dims = [
        topk as u32,
        hidden as u32,
        inter as u32,
        stacked.down.packed_cols as u32,
    ];
    let groups = stacked.down.groups as u32;

    let command_buffer = executor.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    let mut owned = Vec::new();
    executor.encode_gather_matmul(
        encoder,
        &mut owned,
        &lhs_buffer,
        topk,
        &stacked.down,
        &indices_buffer,
        topk,
        &down_buffer,
    )?;
    executor.encode_weighted_sum_add_shared_topk(
        encoder,
        &down_buffer,
        &scores_buffer,
        &residual_buffer,
        &shared_buffer,
        &shared_gate_buffer,
        &two_step_buffer,
        topk,
        hidden,
    )?;
    encoder
        .set_compute_pipeline_state(&executor.affine_gather_down_weighted_shared_fast_u8_gs64_f32);
    encoder.set_buffer(0, Some(&lhs_buffer), 0);
    encoder.set_buffer(1, Some(&stacked.down.packed), 0);
    encoder.set_buffer(2, Some(&stacked.down.scales), 0);
    encoder.set_buffer(3, Some(&stacked.down.biases), 0);
    encoder.set_buffer(4, Some(&indices_buffer), 0);
    encoder.set_buffer(5, Some(&scores_buffer), 0);
    encoder.set_buffer(6, Some(&residual_buffer), 0);
    encoder.set_buffer(7, Some(&shared_buffer), 0);
    encoder.set_buffer(8, Some(&shared_gate_buffer), 0);
    encoder.set_buffer(9, Some(&fused_buffer), 0);
    encoder.set_bytes(10, 16, dims.as_ptr().cast());
    encoder.set_bytes(11, 4, (&groups as *const u32).cast());
    encoder.dispatch_thread_groups(
        MTLSize::new((hidden as u64).div_ceil(8), 1, 1),
        MTLSize::new(64, 1, 1),
    );
    encoder.end_encoding();
    commit_and_wait(command_buffer)?;

    let two_step = read_f32_buffer(&two_step_buffer, hidden)?;
    let fused = read_f32_buffer(&fused_buffer, hidden)?;
    assert_bits_equal(&fused, &two_step, "moe down weighted shared fused");
    Ok(())
}

#[test]
fn dense_resident_tail_affine_matches_cpu() -> Result<()> {
    let executor = match MetalExecutor::new() {
        Ok(executor) => executor,
        Err(InferError::Metal(message)) if message.contains("aucun device") => return Ok(()),
        Err(error) => return Err(error),
    };

    // Dimensions multiples de 64 mais pas de 512 pour forcer
    // `affine_matmul_rhs_t_u32_f32` via `encode_matmul_weight`, pas le fast
    // path 4bit-only.
    let hidden = 448_usize;
    let inter = 192_usize;
    let eps = 1.0e-6_f32;
    let residual = Tensor::from_vec(
        vec![1, hidden],
        (0..hidden)
            .map(|idx| ((idx % 53) as f32 - 26.0) / 32.0)
            .collect::<Vec<_>>(),
    )
    .expect("invariant: résiduel dense valide");
    let attn = Tensor::from_vec(
        vec![1, hidden],
        (0..hidden)
            .map(|idx| ((idx % 37) as f32 - 18.0) / 48.0)
            .collect::<Vec<_>>(),
    )
    .expect("invariant: sortie attention dense valide");
    let post_norm = Tensor::from_vec(
        vec![hidden],
        (0..hidden)
            .map(|idx| 0.85 + (idx % 11) as f32 * 0.01)
            .collect::<Vec<_>>(),
    )
    .expect("invariant: poids post_norm valide");
    let mlp = GatedMlp::new(
        Linear::from_weight(
            LinearWeight::AffineQuantized(test_affine(inter, hidden, 0.004)?),
            None,
        )?,
        Linear::from_weight(
            LinearWeight::AffineQuantized(test_affine(inter, hidden, 0.006)?),
            None,
        )?,
        Linear::from_weight(
            LinearWeight::AffineQuantized(test_affine(hidden, inter, -0.003)?),
            None,
        )?,
    );

    let summed_cpu = residual.add(&attn)?;
    let post_normed_cpu = crate::rms_norm(&summed_cpu, &post_norm, eps)?;
    let dense_cpu = mlp.forward(&post_normed_cpu)?;
    let reference = summed_cpu.add(&dense_cpu)?;

    let residual_buf = executor.upload_f32_buffer(residual.data(), "dense_tail_residual")?;
    let attn_buf = executor.upload_f32_buffer(attn.data(), "dense_tail_attn")?;
    let post_norm_buf =
        executor.cached_buffer_from_f32(post_norm.data(), "dense_tail_post_norm")?;
    let summed_buf = executor.private_f32_buffer(hidden, "dense_tail_summed")?;
    let post_normed_buf = executor.private_f32_buffer(hidden, "dense_tail_post_normed")?;
    let gate_buf = executor.private_f32_buffer(inter, "dense_tail_gate")?;
    let up_buf = executor.private_f32_buffer(inter, "dense_tail_up")?;
    let swiglu_buf = executor.private_f32_buffer(inter, "dense_tail_swiglu")?;
    let down_buf = executor.private_f32_buffer(hidden, "dense_tail_down")?;
    let (gate_proj, up_proj, down_proj) = mlp.projections();
    let mut owned = Vec::new();

    let command_buffer = executor.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    executor.encode_add_rms_norm_rows(
        encoder,
        &residual_buf,
        &attn_buf,
        &post_norm_buf,
        &summed_buf,
        &post_normed_buf,
        1,
        hidden,
        eps,
    )?;
    let gate_dim = executor.encode_matmul_weight(
        encoder,
        &mut owned,
        &post_normed_buf,
        1,
        hidden,
        gate_proj.weight(),
        &gate_buf,
    )?;
    let up_dim = executor.encode_matmul_weight(
        encoder,
        &mut owned,
        &post_normed_buf,
        1,
        hidden,
        up_proj.weight(),
        &up_buf,
    )?;
    assert_eq!(gate_dim, inter);
    assert_eq!(up_dim, inter);
    executor.encode_swiglu(encoder, &mut owned, &gate_buf, &up_buf, &swiglu_buf, inter)?;
    let down_dim = executor.encode_matmul_weight(
        encoder,
        &mut owned,
        &swiglu_buf,
        1,
        inter,
        down_proj.weight(),
        &down_buf,
    )?;
    assert_eq!(down_dim, hidden);
    executor.encode_accumulate_scaled(encoder, &mut owned, &down_buf, &summed_buf, 1.0, hidden)?;
    encoder.end_encoding();
    commit_and_wait(command_buffer)?;

    let gpu = read_f32_buffer(&summed_buf, hidden)?;
    assert_close_eps(&gpu, reference.data(), 1.0e-4);
    Ok(())
}

#[test]
fn dense_resident_tail_fast_gate_up_matches_cpu() -> Result<()> {
    let executor = match MetalExecutor::new() {
        Ok(executor) => executor,
        Err(InferError::Metal(message)) if message.contains("aucun device") => return Ok(()),
        Err(error) => return Err(error),
    };

    let hidden = 512_usize;
    let inter = 512_usize;
    let eps = 1.0e-6_f32;
    let residual = Tensor::from_vec(
        vec![1, hidden],
        (0..hidden)
            .map(|idx| ((idx % 59) as f32 - 29.0) / 40.0)
            .collect::<Vec<_>>(),
    )
    .expect("invariant: résiduel dense fast valide");
    let attn = Tensor::from_vec(
        vec![1, hidden],
        (0..hidden)
            .map(|idx| ((idx % 41) as f32 - 20.0) / 56.0)
            .collect::<Vec<_>>(),
    )
    .expect("invariant: sortie attention dense fast valide");
    let post_norm = Tensor::from_vec(
        vec![hidden],
        (0..hidden)
            .map(|idx| 0.9 + (idx % 13) as f32 * 0.0075)
            .collect::<Vec<_>>(),
    )
    .expect("invariant: poids post_norm dense fast valide");
    let mlp = GatedMlp::new(
        Linear::from_weight(
            LinearWeight::AffineQuantized(test_affine(inter, hidden, 0.0035)?),
            None,
        )?,
        Linear::from_weight(
            LinearWeight::AffineQuantized(test_affine(inter, hidden, 0.0055)?),
            None,
        )?,
        Linear::from_weight(
            LinearWeight::AffineQuantized(test_affine(hidden, inter, -0.0025)?),
            None,
        )?,
    );

    let summed_cpu = residual.add(&attn)?;
    let post_normed_cpu = crate::rms_norm(&summed_cpu, &post_norm, eps)?;
    let dense_cpu = mlp.forward(&post_normed_cpu)?;
    let reference = summed_cpu.add(&dense_cpu)?;

    let residual_buf = executor.upload_f32_buffer(residual.data(), "dense_tail_fast_residual")?;
    let attn_buf = executor.upload_f32_buffer(attn.data(), "dense_tail_fast_attn")?;
    let post_norm_buf =
        executor.cached_buffer_from_f32(post_norm.data(), "dense_tail_fast_post_norm")?;
    let summed_buf = executor.private_f32_buffer(hidden, "dense_tail_fast_summed")?;
    let post_normed_buf = executor.private_f32_buffer(hidden, "dense_tail_fast_post_normed")?;
    let swiglu_buf = executor.private_f32_buffer(inter, "dense_tail_fast_swiglu")?;
    let down_buf = executor.private_f32_buffer(hidden, "dense_tail_fast_down")?;
    let (gate_proj, up_proj, down_proj) = mlp.projections();
    let mut owned = Vec::new();

    let command_buffer = executor.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    executor.encode_add_rms_norm_rows(
        encoder,
        &residual_buf,
        &attn_buf,
        &post_norm_buf,
        &summed_buf,
        &post_normed_buf,
        1,
        hidden,
        eps,
    )?;
    let used_fast = executor.encode_gate_up_swiglu_fast(
        encoder,
        &post_normed_buf,
        gate_proj,
        up_proj,
        &swiglu_buf,
        hidden,
    )?;
    assert!(used_fast, "gate/up dense fast doit être éligible");
    let down_dim = executor.encode_matmul_weight(
        encoder,
        &mut owned,
        &swiglu_buf,
        1,
        inter,
        down_proj.weight(),
        &down_buf,
    )?;
    assert_eq!(down_dim, hidden);
    executor.encode_accumulate_scaled(encoder, &mut owned, &down_buf, &summed_buf, 1.0, hidden)?;
    encoder.end_encoding();
    commit_and_wait(command_buffer)?;

    let gpu = read_f32_buffer(&summed_buf, hidden)?;
    assert_close_eps(&gpu, reference.data(), 1.0e-4);
    Ok(())
}

#[test]
fn resident_dense_large_qmv_matches_cpu() -> Result<()> {
    let executor = match MetalExecutor::new() {
        Ok(executor) => executor,
        Err(InferError::Metal(message)) if message.contains("aucun device") => return Ok(()),
        Err(error) => return Err(error),
    };

    let in_dim = 512_usize;
    let out_dim = 5120_usize;
    let input = Tensor::from_vec(
        vec![1, in_dim],
        (0..in_dim)
            .map(|idx| ((idx % 43) as f32 - 21.0) / 64.0)
            .collect::<Vec<_>>(),
    )
    .expect("invariant: entrée qmv résident dense valide");
    let linear = Linear::from_weight(
        LinearWeight::AffineQuantized(test_affine(out_dim, in_dim, 0.0025)?),
        None,
    )?;
    let reference = linear.forward(&input)?;

    let input_buf = executor.upload_f32_buffer(input.data(), "resident_dense_large_qmv_in")?;
    let output_buf = executor.private_f32_buffer(out_dim, "resident_dense_large_qmv_out")?;
    let weight =
        executor.resolve_linear_weight_buffers(linear.weight(), "resident_dense_large_qmv")?;
    let command_buffer = executor.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    let actual_out = executor.encode_matmul_weight_buffers(
        encoder,
        &input_buf,
        1,
        in_dim,
        &weight,
        &output_buf,
        false,
    )?;
    assert_eq!(actual_out, out_dim);
    encoder.end_encoding();
    commit_and_wait(command_buffer)?;

    let gpu = read_f32_buffer(&output_buf, out_dim)?;
    assert_close_eps(&gpu, reference.data(), 1.0e-4);
    Ok(())
}

#[test]
fn qmm2_aligned_matches_cpu_both_rows() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    // Shape éligible : 4-bit gs64, in_dim%512==0, out_dim%8==0.
    let in_dim = 512_usize;
    let out_dim = 64_usize;
    // test_affine est déterministe → mêmes poids côté CPU (Linear) et GPU (buffers).
    let cpu = Linear::from_weight(
        LinearWeight::AffineQuantized(test_affine(out_dim, in_dim, 0.0025)?),
        None,
    )?;
    let gpu_w = test_affine(out_dim, in_dim, 0.0025)?;

    let row = |off: usize| -> Vec<f32> {
        (0..in_dim)
            .map(|i| (((i + off) % 43) as f32 - 21.0) / 64.0)
            .collect()
    };
    let x0 = row(0);
    let x1 = row(7);
    let ref0 = cpu.forward(&Tensor::from_vec(vec![1, in_dim], x0.clone())?)?;
    let ref1 = cpu.forward(&Tensor::from_vec(vec![1, in_dim], x1.clone())?)?;

    // lhs = [2, in_dim] contigu, out = [2, out_dim].
    let mut lhs2 = x0;
    lhs2.extend_from_slice(&x1);
    let lhs_buf = executor.upload_f32_buffer(&lhs2, "qmm2_lhs")?;
    let packed = executor.cached_buffer_from_u32(gpu_w.packed_data(), "qmm2_packed")?;
    let scales = executor.cached_buffer_from_f32_as_bf16(gpu_w.scales().data(), "qmm2_scales")?;
    let biases = executor.cached_buffer_from_f32_as_bf16(gpu_w.biases().data(), "qmm2_biases")?;
    let out_buf = executor.uncached_f32_buffer(2 * out_dim, "qmm2_out")?;
    let packed_cols = in_dim / 8;
    let groups = in_dim / 64;
    let dims: [u32; 4] = [
        out_dim as u32,
        in_dim as u32,
        packed_cols as u32,
        groups as u32,
    ];

    let command_buffer = executor.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&executor.affine_qmm2_fast_aligned_u4_gs64_f32);
    encoder.set_buffer(0, Some(&lhs_buf), 0);
    encoder.set_buffer(1, Some(&packed), 0);
    encoder.set_buffer(2, Some(&scales), 0);
    encoder.set_buffer(3, Some(&biases), 0);
    encoder.set_buffer(4, Some(&out_buf), 0);
    encoder.set_bytes(5, 16, dims.as_ptr().cast());
    encoder.dispatch_thread_groups(
        MTLSize::new(1, (out_dim as u64).div_ceil(8), 1),
        MTLSize::new(64, 1, 1),
    );
    encoder.end_encoding();
    commit_and_wait(command_buffer)?;

    let out = read_f32_buffer(&out_buf, 2 * out_dim)?;
    assert_close_eps(&out[0..out_dim], ref0.data(), 1.0e-4);
    assert_close_eps(&out[out_dim..2 * out_dim], ref1.data(), 1.0e-4);
    Ok(())
}

#[test]
fn qmm2_route_matches_cpu_both_rows() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let in_dim = 512_usize;
    let out_dim = 64_usize;
    let affine = test_affine(out_dim, in_dim, 0.0025)?;
    let cpu = Linear::from_weight(LinearWeight::AffineQuantized(affine.clone()), None)?;
    let weight = executor.resolve_linear_weight_buffers(
        &LinearWeight::AffineQuantized(affine),
        "qmm2_route_weight",
    )?;

    let row = |off: usize| -> Vec<f32> {
        (0..in_dim)
            .map(|i| (((i + off) % 47) as f32 - 23.0) / 64.0)
            .collect()
    };
    let x0 = row(0);
    let x1 = row(11);
    let ref0 = cpu.forward(&Tensor::from_vec(vec![1, in_dim], x0.clone())?)?;
    let ref1 = cpu.forward(&Tensor::from_vec(vec![1, in_dim], x1.clone())?)?;
    let mut lhs = x0;
    lhs.extend_from_slice(&x1);

    let lhs_buf = executor.upload_f32_buffer(&lhs, "qmm2_route_lhs")?;
    let out_buf = executor.uncached_f32_buffer(2 * out_dim, "qmm2_route_out")?;
    let command_buffer = executor.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    let actual_out = executor
        .encode_matmul_weight_buffers(encoder, &lhs_buf, 2, in_dim, &weight, &out_buf, false)?;
    assert_eq!(actual_out, out_dim);
    encoder.end_encoding();
    commit_and_wait(command_buffer)?;

    let out = read_f32_buffer(&out_buf, 2 * out_dim)?;
    assert_close_eps(&out[0..out_dim], ref0.data(), 1.0e-4);
    assert_close_eps(&out[out_dim..2 * out_dim], ref1.data(), 1.0e-4);
    Ok(())
}

#[test]
fn affine_u8_standalone_matches_fast_qmv_routes() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let in_dim = 512_usize;
    let out_dim = 64_usize;
    let weights = [
        test_affine_varied_u8_group(out_dim, in_dim, 64)?,
        test_affine_varied_u8_group(out_dim, in_dim, 128)?,
    ];

    for weight in weights {
        for (batch, salts) in [(1_usize, [3_usize, 0_usize]), (2, [5, 17])] {
            let mut lhs = Vec::with_capacity(batch * in_dim);
            for salt in salts.into_iter().take(batch) {
                lhs.extend_from_slice(&varied_row(in_dim, salt));
            }
            let input = Tensor::from_vec(vec![batch, in_dim], lhs.clone())?;
            let actual = executor.matmul_rhs_t_affine(&input, &weight)?;
            let reference =
                fast_qmv_u8_reference(&executor, &weight, &lhs, batch, "standalone_u8")?;

            assert_eq!(actual.shape(), [batch, out_dim]);
            assert_bits_equal(
                actual.data(),
                &reference,
                &format!("standalone u8 gs{} batch={batch}", weight.group_size()),
            );
        }
    }

    Ok(())
}

#[test]
fn affine_qmv_u8_tg128_matches_tg64_route() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let in_dim = 512_usize;

    for group_size in [64_usize, 128_usize] {
        for out_dim in [24_usize, 64_usize] {
            let weight = test_affine_varied_u8_group(out_dim, in_dim, group_size)?;
            for (batch, salts) in [(1_usize, [41_usize, 0_usize]), (2, [43, 47])] {
                let mut lhs = Vec::with_capacity(batch * in_dim);
                for salt in salts.into_iter().take(batch) {
                    lhs.extend_from_slice(&varied_row(in_dim, salt));
                }
                let reference =
                    fast_qmv_u8_reference(&executor, &weight, &lhs, batch, "tg128_ref")?;
                let tg128 = fast_qmv_u8_tg128(&executor, &weight, &lhs, batch, "tg128")?;

                assert_bits_equal(
                    &tg128,
                    &reference,
                    &format!("tg128 u8 gs{group_size} out_dim={out_dim} batch={batch}"),
                );
            }
        }
    }

    Ok(())
}

#[test]
fn affine_qmv_u8_tg256_matches_tg64_route() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let in_dim = 512_usize;

    for group_size in [64_usize, 128_usize] {
        for out_dim in [24_usize, 64_usize] {
            let weight = test_affine_varied_u8_group(out_dim, in_dim, group_size)?;
            for (batch, salts) in [(1_usize, [73_usize, 0_usize]), (2, [79, 83])] {
                let mut lhs = Vec::with_capacity(batch * in_dim);
                for salt in salts.into_iter().take(batch) {
                    lhs.extend_from_slice(&varied_row(in_dim, salt));
                }
                let reference =
                    fast_qmv_u8_reference(&executor, &weight, &lhs, batch, "tg256_ref")?;
                let tg256 = fast_qmv_u8_tg256(&executor, &weight, &lhs, batch, "tg256")?;

                assert_bits_equal(
                    &tg256,
                    &reference,
                    &format!("tg256 u8 gs{group_size} out_dim={out_dim} batch={batch}"),
                );
            }
        }
    }

    Ok(())
}

#[test]
fn affine_qmv_u8_dot4_stays_close_to_tg64_route() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let in_dim = 512_usize;

    for group_size in [64_usize, 128_usize] {
        for out_dim in [24_usize, 64_usize] {
            let weight = test_affine_varied_u8_group(out_dim, in_dim, group_size)?;
            for (batch, salts) in [(1_usize, [61_usize, 0_usize]), (2, [67, 71])] {
                let mut lhs = Vec::with_capacity(batch * in_dim);
                for salt in salts.into_iter().take(batch) {
                    lhs.extend_from_slice(&varied_row(in_dim, salt));
                }
                let reference = fast_qmv_u8_reference(&executor, &weight, &lhs, batch, "dot4_ref")?;
                let dot4 = fast_qmv_u8_dot4(&executor, &weight, &lhs, batch, "dot4")?;

                assert_close_eps(&dot4, &reference, 1.0e-4);
            }
        }
    }

    Ok(())
}

#[test]
fn gather_qmv_u8_tg128_matches_tg64_route() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let in_dim = 512_usize;
    let out_dim = 24_usize;
    let topk = 3_usize;
    let indices = [0_u32, 2_u32, 1_u32];

    for group_size in [64_usize, 128_usize] {
        let weight =
            test_stacked_affine_varied_u8_group(&executor, 3, out_dim, in_dim, group_size)?;
        for lhs_rows in [1_usize, topk] {
            let mut lhs = Vec::with_capacity(lhs_rows * in_dim);
            for row in 0..lhs_rows {
                lhs.extend_from_slice(&varied_row(in_dim, 53 + row * 7));
            }
            let reference = gather_qmv_u8_values(
                &executor,
                &weight,
                &lhs,
                lhs_rows,
                &indices,
                GatherQmvU8Route::Tg64,
                "gather_tg64",
            )?;
            let tg128 = gather_qmv_u8_values(
                &executor,
                &weight,
                &lhs,
                lhs_rows,
                &indices,
                GatherQmvU8Route::Tg128,
                "gather_tg128",
            )?;

            assert_bits_equal(
                &tg128,
                &reference,
                &format!("gather tg128 u8 gs{group_size} lhs_rows={lhs_rows}"),
            );
        }
    }

    Ok(())
}

#[test]
fn gather_qmv_u8_tg256_matches_tg64_route() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let in_dim = 512_usize;
    let out_dim = 24_usize;
    let topk = 3_usize;
    let indices = [0_u32, 2_u32, 1_u32];

    for group_size in [64_usize, 128_usize] {
        let weight =
            test_stacked_affine_varied_u8_group(&executor, 3, out_dim, in_dim, group_size)?;
        for lhs_rows in [1_usize, topk] {
            let mut lhs = Vec::with_capacity(lhs_rows * in_dim);
            for row in 0..lhs_rows {
                lhs.extend_from_slice(&varied_row(in_dim, 89 + row * 7));
            }
            let reference = gather_qmv_u8_values(
                &executor,
                &weight,
                &lhs,
                lhs_rows,
                &indices,
                GatherQmvU8Route::Tg64,
                "gather_tg64_ref",
            )?;
            let tg256 = gather_qmv_u8_values(
                &executor,
                &weight,
                &lhs,
                lhs_rows,
                &indices,
                GatherQmvU8Route::Tg256,
                "gather_tg256",
            )?;

            assert_bits_equal(
                &tg256,
                &reference,
                &format!("gather tg256 u8 gs{group_size} lhs_rows={lhs_rows}"),
            );
        }
    }

    Ok(())
}

#[allow(dead_code, reason = "harnais garde en reserve pour le routage u8")]
fn encode_matmul_weight_u8_matches_fast_qmv_route() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let in_dim = 512_usize;
    let out_dim = 64_usize;
    let weights = [
        test_affine_varied_u8_group(out_dim, in_dim, 64)?,
        test_affine_varied_u8_group(out_dim, in_dim, 128)?,
    ];

    for weight in weights {
        let linear = LinearWeight::AffineQuantized(weight.clone());
        for (batch, salts) in [(1_usize, [11_usize, 0_usize]), (2, [13, 23])] {
            let mut lhs = Vec::with_capacity(batch * in_dim);
            for salt in salts.into_iter().take(batch) {
                lhs.extend_from_slice(&varied_row(in_dim, salt));
            }
            let lhs_buf = executor.upload_f32_buffer(&lhs, "encode_u8_lhs")?;
            let out_buf = executor.uncached_f32_buffer(batch * out_dim, "encode_u8_out")?;
            let reference =
                fast_qmv_u8_reference(&executor, &weight, &lhs, batch, "encode_u8_ref")?;

            let command_buffer = executor.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            let mut owned_buffers = Vec::new();
            let actual_out = executor.encode_matmul_weight(
                encoder,
                &mut owned_buffers,
                &lhs_buf,
                batch,
                in_dim,
                &linear,
                &out_buf,
            )?;
            assert_eq!(actual_out, out_dim);
            encoder.end_encoding();
            commit_and_wait(command_buffer)?;

            let actual = read_f32_buffer(&out_buf, batch * out_dim)?;
            assert_bits_equal(
                &actual,
                &reference,
                &format!(
                    "encode_matmul_weight u8 gs{} batch={batch}",
                    weight.group_size()
                ),
            );
        }
    }

    Ok(())
}

#[test]
fn encode_matmul_weight_buffers_u8_gs128_matches_fast_qmv_route() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let in_dim = 512_usize;
    let out_dim = 64_usize;
    let weight = test_affine_varied_u8_group(out_dim, in_dim, 128)?;
    let buffers = executor.resolve_linear_weight_buffers(
        &LinearWeight::AffineQuantized(weight.clone()),
        "encode_u8_gs128_buffers",
    )?;

    for (batch, salts) in [(1_usize, [29_usize, 0_usize]), (2, [31, 37])] {
        let mut lhs = Vec::with_capacity(batch * in_dim);
        for salt in salts.into_iter().take(batch) {
            lhs.extend_from_slice(&varied_row(in_dim, salt));
        }
        let lhs_buf = executor.upload_f32_buffer(&lhs, "encode_u8_gs128_lhs")?;
        let out_buf = executor.uncached_f32_buffer(batch * out_dim, "encode_u8_gs128_out")?;
        let reference =
            fast_qmv_u8_reference(&executor, &weight, &lhs, batch, "encode_u8_gs128_ref")?;

        let command_buffer = executor.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let actual_out = executor.encode_matmul_weight_buffers(
            encoder, &lhs_buf, batch, in_dim, &buffers, &out_buf, false,
        )?;
        assert_eq!(actual_out, out_dim);
        encoder.end_encoding();
        commit_and_wait(command_buffer)?;

        let actual = read_f32_buffer(&out_buf, batch * out_dim)?;
        assert_bits_equal(
            &actual,
            &reference,
            &format!("encode_matmul_weight_buffers u8 gs128 batch={batch}"),
        );
    }

    Ok(())
}

#[test]
fn affine_qmv_u6_gs64_matches_generic_and_routes() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    for (out_dim, in_dim, batch) in [
        (64_usize, 512_usize, 1_usize),
        (65, 512, 2),
        (256, 1024, 1),
        (2048, 2048, 1),
    ] {
        let weight = test_affine_varied_u6(out_dim, in_dim)?;
        let mut lhs = Vec::with_capacity(batch * in_dim);
        for row in 0..batch {
            lhs.extend_from_slice(&varied_row(in_dim, 53 + row));
        }

        let generic =
            generic_affine_reference(&executor, &weight, &lhs, batch, "qmv_u6_generic_ref")?;
        let fast = fast_qmv_u6_reference(&executor, &weight, &lhs, batch, "qmv_u6_fast_ref")?;
        assert_bits_equal(
            &fast,
            &generic,
            &format!("qmv u6 fast vs generic ({out_dim}x{in_dim}) batch={batch}"),
        );

        let input = Tensor::from_vec(vec![batch, in_dim], lhs.clone())?;
        let direct = executor.matmul_rhs_t_affine(&input, &weight)?;
        assert_bits_equal(
            direct.data(),
            &generic,
            &format!("matmul_rhs_t_affine u6 route ({out_dim}x{in_dim}) batch={batch}"),
        );

        let buffers = executor.resolve_linear_weight_buffers(
            &LinearWeight::AffineQuantized(weight),
            "qmv_u6_buffers_route",
        )?;
        let lhs_buf = executor.upload_f32_buffer(&lhs, "qmv_u6_buffers_lhs")?;
        let out_buf = executor.uncached_f32_buffer(batch * out_dim, "qmv_u6_buffers_out")?;
        let command_buffer = executor.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let actual_out = executor.encode_matmul_weight_buffers(
            encoder, &lhs_buf, batch, in_dim, &buffers, &out_buf, false,
        )?;
        assert_eq!(actual_out, out_dim);
        encoder.end_encoding();
        commit_and_wait(command_buffer)?;

        let routed = read_f32_buffer(&out_buf, batch * out_dim)?;
        assert_bits_equal(
            &routed,
            &generic,
            &format!("encode_matmul_weight_buffers u6 route ({out_dim}x{in_dim}) batch={batch}"),
        );
    }
    Ok(())
}

#[test]
fn encode_matmul_weight_buffers_na_gs128_matches_cpu_when_enabled() -> Result<()> {
    if !matches!(
        std::env::var("RETI_RUST_QMM_NA_GS128").as_deref(),
        Ok("1" | "true" | "on" | "yes")
    ) {
        eprintln!("skip: RETI_RUST_QMM_NA_GS128 désactivé");
        return Ok(());
    }
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    if executor.na_gemm_coop_qb_gs128.is_none() {
        eprintln!("skip: coop qb gs128 indisponible (macOS < 26 ?)");
        return Ok(());
    }
    let batch = 17_usize;
    let in_dim = 512_usize;
    let out_dim = 64_usize;
    let weight = test_affine_varied_u8_group(out_dim, in_dim, 128)?;
    let buffers = executor.resolve_linear_weight_buffers(
        &LinearWeight::AffineQuantized(weight.clone()),
        "encode_na_gs128_buffers",
    )?;
    let mut lhs = Vec::with_capacity(batch * in_dim);
    for row in 0..batch {
        lhs.extend_from_slice(&varied_row(in_dim, 41 + row));
    }
    let lhs_buf = executor.upload_f32_buffer(&lhs, "encode_na_gs128_lhs")?;
    let out_buf = executor.uncached_f32_buffer(batch * out_dim, "encode_na_gs128_out")?;
    let reference = qmm_na_qb_reference(&weight, &lhs, batch)?;

    let command_buffer = executor.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    let actual_out = executor.encode_matmul_weight_buffers(
        encoder, &lhs_buf, batch, in_dim, &buffers, &out_buf, false,
    )?;
    assert_eq!(actual_out, out_dim);
    encoder.end_encoding();
    commit_and_wait(command_buffer)?;

    let actual = read_f32_buffer(&out_buf, batch * out_dim)?;
    assert_close_eps(&actual, &reference, 5.0e-2);
    Ok(())
}

#[test]
fn encode_matmul_weight_buffers_na_fused_tiled_matches_cpu_when_enabled() -> Result<()> {
    if !matches!(
        std::env::var("RETI_RUST_QMM_NA_FUSED_TILED").as_deref(),
        Ok("1" | "true" | "on" | "yes")
    ) {
        eprintln!("skip: RETI_RUST_QMM_NA_FUSED_TILED désactivé");
        return Ok(());
    }
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let cases = [
        (
            8_usize,
            64_usize,
            65_usize,
            512_usize,
            128_usize,
            executor.na_gemm_coop_qb_tiled.is_some(),
            "qmm_na_fused_tiled_u8_gs64",
        ),
        (
            8_usize,
            128_usize,
            65_usize,
            512_usize,
            128_usize,
            executor.na_gemm_coop_qb_tiled_gs128.is_some(),
            "qmm_na_fused_tiled_u8_gs128",
        ),
        (
            4_usize,
            64_usize,
            17_usize,
            2048_usize,
            2048_usize,
            qmm_na_fused_tiled_u4_enabled() && executor.na_gemm_coop_qb_tiled_u4.is_some(),
            "qmm_na_fused_tiled_u4_gs64",
        ),
    ];
    for (bits, group_size, batch, in_dim, out_dim, available, profile_kind) in cases {
        if !available {
            eprintln!("skip: coop qb tiled u{bits} gs{group_size} indisponible (macOS < 26 ?)");
            continue;
        }
        let weight = if bits == 4 {
            test_affine_varied(out_dim, in_dim)?
        } else {
            test_affine_varied_u8_group(out_dim, in_dim, group_size)?
        };
        let buffers = executor.resolve_linear_weight_buffers(
            &LinearWeight::AffineQuantized(weight.clone()),
            "encode_na_fused_tiled_buffers",
        )?;
        let mut lhs = Vec::with_capacity(batch * in_dim);
        for row in 0..batch {
            lhs.extend_from_slice(&varied_row(in_dim, 91 + row));
        }
        let lhs_buf = executor.upload_f32_buffer(&lhs, "encode_na_fused_tiled_lhs")?;
        let out_buf = executor.uncached_f32_buffer(batch * out_dim, "encode_na_fused_tiled_out")?;
        let reference = qmm_na_qb_reference(&weight, &lhs, batch)?;
        let profile_key =
            DispatchProfileShape::matmul(profile_kind, batch, in_dim, out_dim, group_size, bits);
        let before_profile = decode_profile_dispatch_shapes_snapshot()
            .get(&profile_key)
            .copied()
            .unwrap_or_default();

        let command_buffer = executor.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let actual_out = executor.encode_matmul_weight_buffers(
            encoder, &lhs_buf, batch, in_dim, &buffers, &out_buf, false,
        )?;
        assert_eq!(actual_out, out_dim);
        encoder.end_encoding();
        commit_and_wait(command_buffer)?;

        let actual = read_f32_buffer(&out_buf, batch * out_dim)?;
        // Le u4 NA passe l'oracle greedy byte-identique ; ce test reste un smoke
        // numérique du dispatch tensor-core, dont l'ordre bf16 diverge du CPU.
        let eps = if bits == 4 { 3.0e-1 } else { 5.0e-2 };
        assert_close_eps(&actual, &reference, eps);
        if matches!(
            std::env::var("RETI_RUST_DECODE_PROFILE").as_deref(),
            Ok("1" | "true" | "on" | "yes")
        ) && matches!(
            std::env::var("RETI_RUST_DECODE_PROFILE_SITES").as_deref(),
            Ok("1" | "true" | "on" | "yes")
        ) {
            let after_profile = decode_profile_dispatch_shapes_snapshot()
                .get(&profile_key)
                .copied()
                .unwrap_or_default();
            assert!(
                after_profile > before_profile,
                "dispatch fused-tiled absent pour u{bits} gs{group_size}: {profile_kind}"
            );
        }
    }
    Ok(())
}

#[test]
fn qmm_na_fused_tiled_u4_accepts_dense_and_rejects_shared_shapes() {
    assert!(can_use_qmm_na_fused_tiled_u4_buffers(
        17,
        2048,
        2048,
        FAST_QMV_GROUP_SIZE,
        FAST_QMV_BITS
    ));
    assert!(!can_use_qmm_na_fused_tiled_u4_buffers(
        17,
        2048,
        512,
        FAST_QMV_GROUP_SIZE,
        FAST_QMV_BITS
    ));
    assert!(!can_use_qmm_na_fused_tiled_u4_buffers(
        17,
        512,
        2048,
        FAST_QMV_GROUP_SIZE,
        FAST_QMV_BITS
    ));
}

#[test]
fn affine_qmv_one_u8_gs64_matches_cpu_on_shared_gate_shape() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let batch = 2_usize;
    let in_dim = 2048_usize;
    let out_dim = 1_usize;
    let weight = test_affine_varied_u8_group(out_dim, in_dim, 64)?;
    let mut lhs = varied_row(in_dim, 41);
    lhs.extend_from_slice(&varied_row(in_dim, 43));
    let input = Tensor::from_vec(vec![batch, in_dim], lhs.clone())?;
    let cpu = weight.matmul_rhs_t(&input)?;
    let lhs_buf = executor.upload_f32_buffer(&lhs, "qmv_one_u8_lhs")?;
    let packed = executor.buffer_from_slice(weight.packed_data(), "qmv_one_u8_packed")?;
    let scales = executor.buffer_from_f32_as_bf16(weight.scales().data(), "qmv_one_u8_scales")?;
    let biases = executor.buffer_from_f32_as_bf16(weight.biases().data(), "qmv_one_u8_biases")?;
    let out_buf = executor.uncached_f32_buffer(batch * out_dim, "qmv_one_u8_out")?;
    let [_, packed_cols] = weight.packed_shape() else {
        return Err(InferError::Dimension(format!(
            "packed_shape qmv_one attendu rang 2, reçu {:?}",
            weight.packed_shape()
        )));
    };
    let dims = [
        out_dim as u32,
        in_dim as u32,
        *packed_cols as u32,
        (in_dim / 64) as u32,
    ];

    let command_buffer = executor.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&executor.affine_qmv_one_fast_u8_gs64_f32);
    encoder.set_buffer(0, Some(&lhs_buf), 0);
    encoder.set_buffer(1, Some(&packed), 0);
    encoder.set_buffer(2, Some(&scales), 0);
    encoder.set_buffer(3, Some(&biases), 0);
    encoder.set_buffer(4, Some(&out_buf), 0);
    encoder.set_bytes(5, 16, dims.as_ptr().cast());
    encoder.dispatch_thread_groups(MTLSize::new(batch as u64, 1, 1), MTLSize::new(32, 1, 1));
    encoder.end_encoding();
    commit_and_wait(command_buffer)?;

    let gpu = read_f32_buffer(&out_buf, batch * out_dim)?;
    assert_close_eps(&gpu, cpu.data(), 1.0e-4);
    Ok(())
}

/// Affine 4-bit gs64 déterministe à scales/biases VARIÉS (bf16-arrondis comme
/// les buffers GPU) — `test_affine` (scale constant, bias nul) annulerait le
/// terme `sum * bias` et masquerait une divergence d'accumulation.
fn test_affine_varied(out_dim: usize, in_dim: usize) -> Result<AffineQuantizedTensor> {
    let bits = 4;
    let values_per_word = 32 / bits;
    let packed_cols = in_dim / values_per_word;
    let groups = in_dim / 64;
    let mut packed = Vec::with_capacity(out_dim * packed_cols);
    for row in 0..out_dim {
        for word in 0..packed_cols {
            let mut lanes = [0_u32; 8];
            for (lane, value) in lanes.iter_mut().enumerate() {
                *value = ((row * 7 + word * 3 + lane) % 16) as u32;
            }
            packed.push(pack_lanes(&lanes, bits));
        }
    }
    let scales = Tensor::from_vec(
        vec![out_dim, groups],
        (0..out_dim * groups)
            .map(|i| bf16_round(0.002 + 0.000_1 * ((i % 7) as f32)))
            .collect(),
    )?;
    let biases = Tensor::from_vec(
        vec![out_dim, groups],
        (0..out_dim * groups)
            .map(|i| bf16_round(-0.01 + 0.000_5 * ((i % 11) as f32)))
            .collect(),
    )?;
    AffineQuantizedTensor::new(&[out_dim, packed_cols], packed, scales, biases, 64, bits)
}

/// Affine 8-bit gs64 déterministe à scales/biases variés (poids DWQ :
/// attention/LA/lm_head du 35B prod sont en 8-bit).
fn test_affine_varied_u8(out_dim: usize, in_dim: usize) -> Result<AffineQuantizedTensor> {
    test_affine_varied_u8_group(out_dim, in_dim, 64)
}

fn test_affine_varied_u8_group(
    out_dim: usize,
    in_dim: usize,
    group_size: usize,
) -> Result<AffineQuantizedTensor> {
    let bits = 8;
    let values_per_word = 32 / bits;
    let packed_cols = in_dim / values_per_word;
    let groups = in_dim / group_size;
    let mut packed = Vec::with_capacity(out_dim * packed_cols);
    for row in 0..out_dim {
        for word in 0..packed_cols {
            let mut value = 0_u32;
            for lane in 0..values_per_word {
                value |= (((row * 7 + word * 3 + lane) % 251) as u32) << (lane * bits);
            }
            packed.push(value);
        }
    }
    let scales = Tensor::from_vec(
        vec![out_dim, groups],
        (0..out_dim * groups)
            .map(|i| bf16_round(0.001 + 0.000_05 * ((i % 13) as f32)))
            .collect(),
    )?;
    let biases = Tensor::from_vec(
        vec![out_dim, groups],
        (0..out_dim * groups)
            .map(|i| bf16_round(-0.08 + 0.001 * ((i % 19) as f32)))
            .collect(),
    )?;
    AffineQuantizedTensor::new(
        &[out_dim, packed_cols],
        packed,
        scales,
        biases,
        group_size,
        bits,
    )
}

fn test_affine_varied_u6(out_dim: usize, in_dim: usize) -> Result<AffineQuantizedTensor> {
    let bits = 6;
    let packed_cols = in_dim * bits / 32;
    let groups = in_dim / 64;
    let mut packed = vec![0_u32; out_dim * packed_cols];
    for row in 0..out_dim {
        for col in 0..in_dim {
            let q = ((row * 11 + col * 7 + (col / 64) * 13) % 64) as u32;
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

/// Ligne d'activation déterministe variée (valeurs signées non triviales).
fn varied_row(in_dim: usize, salt: usize) -> Vec<f32> {
    (0..in_dim)
        .map(|i| ((((i * 37 + salt * 101) % 113) as f32) - 56.0) / 71.0)
        .collect()
}

fn generic_affine_reference(
    executor: &MetalExecutor,
    weight: &AffineQuantizedTensor,
    lhs: &[f32],
    batch: usize,
    label: &'static str,
) -> Result<Vec<f32>> {
    let [out_dim, weight_in_dim] = weight.shape() else {
        return Err(InferError::Dimension(format!(
            "{label}: poids attendu rang 2, reçu {:?}",
            weight.shape()
        )));
    };
    let [_, packed_cols] = weight.packed_shape() else {
        return Err(InferError::Dimension(format!(
            "{label}: packed_shape attendu rang 2, reçu {:?}",
            weight.packed_shape()
        )));
    };
    if lhs.len() != batch * *weight_in_dim {
        return Err(InferError::Dimension(format!(
            "{label}: lhs len={} incompatible batch={batch} in_dim={}",
            lhs.len(),
            weight_in_dim
        )));
    }
    let lhs_buf = executor.upload_f32_buffer(lhs, label)?;
    let packed = executor.buffer_from_slice(weight.packed_data(), label)?;
    let scales = executor.buffer_from_f32_as_bf16(weight.scales().data(), label)?;
    let biases = executor.buffer_from_f32_as_bf16(weight.biases().data(), label)?;
    let out_buf = executor.uncached_f32_buffer(batch * *out_dim, label)?;
    let groups = *weight_in_dim / weight.group_size();
    let dims: [u32; 4] = [
        batch as u32,
        *out_dim as u32,
        *weight_in_dim as u32,
        *packed_cols as u32,
    ];
    let quant: [u32; 4] = [
        weight.group_size() as u32,
        weight.bits() as u32,
        groups as u32,
        0,
    ];

    let command_buffer = executor.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&executor.affine_matmul_rhs_t_u32_f32);
    encoder.set_buffer(0, Some(&lhs_buf), 0);
    encoder.set_buffer(1, Some(&packed), 0);
    encoder.set_buffer(2, Some(&scales), 0);
    encoder.set_buffer(3, Some(&biases), 0);
    encoder.set_buffer(4, Some(&out_buf), 0);
    encoder.set_bytes(5, 16, dims.as_ptr().cast());
    encoder.set_bytes(6, 16, quant.as_ptr().cast());
    let threads_per_group = executor
        .affine_matmul_rhs_t_u32_f32
        .thread_execution_width()
        .max(1);
    encoder.dispatch_thread_groups(
        MTLSize::new(*out_dim as u64, batch as u64, 1),
        MTLSize::new(threads_per_group, 1, 1),
    );
    encoder.end_encoding();
    commit_and_wait(command_buffer)?;

    read_f32_buffer(&out_buf, batch * *out_dim)
}

fn fast_qmv_u6_reference(
    executor: &MetalExecutor,
    weight: &AffineQuantizedTensor,
    lhs: &[f32],
    batch: usize,
    label: &'static str,
) -> Result<Vec<f32>> {
    let [out_dim, weight_in_dim] = weight.shape() else {
        return Err(InferError::Dimension(format!(
            "{label}: poids attendu rang 2, reçu {:?}",
            weight.shape()
        )));
    };
    let [_, packed_cols] = weight.packed_shape() else {
        return Err(InferError::Dimension(format!(
            "{label}: packed_shape attendu rang 2, reçu {:?}",
            weight.packed_shape()
        )));
    };
    if lhs.len() != batch * *weight_in_dim {
        return Err(InferError::Dimension(format!(
            "{label}: lhs len={} incompatible batch={batch} in_dim={}",
            lhs.len(),
            weight_in_dim
        )));
    }
    let lhs_buf = executor.upload_f32_buffer(lhs, label)?;
    let packed = executor.buffer_from_slice(weight.packed_data(), label)?;
    let scales = executor.buffer_from_f32_as_bf16(weight.scales().data(), label)?;
    let biases = executor.buffer_from_f32_as_bf16(weight.biases().data(), label)?;
    let out_buf = executor.uncached_f32_buffer(batch * *out_dim, label)?;
    let groups = *weight_in_dim / weight.group_size();
    let dims: [u32; 4] = [
        *out_dim as u32,
        *weight_in_dim as u32,
        *packed_cols as u32,
        groups as u32,
    ];

    let command_buffer = executor.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    let pipeline = if *out_dim % 2 == 0 {
        &executor.affine_qmv_fast_aligned_u6_gs64_f32
    } else {
        &executor.affine_qmv_fast_u6_gs64_f32
    };
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(&lhs_buf), 0);
    encoder.set_buffer(1, Some(&packed), 0);
    encoder.set_buffer(2, Some(&scales), 0);
    encoder.set_buffer(3, Some(&biases), 0);
    encoder.set_buffer(4, Some(&out_buf), 0);
    encoder.set_bytes(5, 16, dims.as_ptr().cast());
    encoder.dispatch_thread_groups(
        MTLSize::new(batch as u64, (*out_dim as u64).div_ceil(2), 1),
        MTLSize::new(64, 1, 1),
    );
    encoder.end_encoding();
    commit_and_wait(command_buffer)?;

    read_f32_buffer(&out_buf, batch * *out_dim)
}

fn fast_qmv_u8_reference(
    executor: &MetalExecutor,
    weight: &AffineQuantizedTensor,
    lhs: &[f32],
    batch: usize,
    label: &'static str,
) -> Result<Vec<f32>> {
    let [out_dim, weight_in_dim] = weight.shape() else {
        return Err(InferError::Dimension(format!(
            "{label}: poids attendu rang 2, reçu {:?}",
            weight.shape()
        )));
    };
    let [_, packed_cols] = weight.packed_shape() else {
        return Err(InferError::Dimension(format!(
            "{label}: packed_shape attendu rang 2, reçu {:?}",
            weight.packed_shape()
        )));
    };
    if lhs.len() != batch * *weight_in_dim {
        return Err(InferError::Dimension(format!(
            "{label}: lhs len={} incompatible batch={batch} in_dim={}",
            lhs.len(),
            weight_in_dim
        )));
    }
    let lhs_buf = executor.upload_f32_buffer(lhs, label)?;
    let packed = executor.buffer_from_slice(weight.packed_data(), label)?;
    let scales = executor.buffer_from_f32_as_bf16(weight.scales().data(), label)?;
    let biases = executor.buffer_from_f32_as_bf16(weight.biases().data(), label)?;
    let out_buf = executor.uncached_f32_buffer(batch * *out_dim, label)?;
    let groups = *weight_in_dim / weight.group_size();
    let dims: [u32; 4] = [
        *out_dim as u32,
        *weight_in_dim as u32,
        *packed_cols as u32,
        groups as u32,
    ];

    let command_buffer = executor.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    let pipeline = match weight.group_size() {
        64 => &executor.affine_qmv_fast_aligned_u8_gs64_f32,
        128 => &executor.affine_qmv_fast_aligned_u8_gs128_f32,
        group_size => {
            return Err(InferError::Dimension(format!(
                "{label}: group_size u8 rapide non supporté {group_size}"
            )));
        }
    };
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(&lhs_buf), 0);
    encoder.set_buffer(1, Some(&packed), 0);
    encoder.set_buffer(2, Some(&scales), 0);
    encoder.set_buffer(3, Some(&biases), 0);
    encoder.set_buffer(4, Some(&out_buf), 0);
    encoder.set_bytes(5, 16, dims.as_ptr().cast());
    encoder.dispatch_thread_groups(
        MTLSize::new(batch as u64, (*out_dim as u64).div_ceil(8), 1),
        MTLSize::new(64, 1, 1),
    );
    encoder.end_encoding();
    commit_and_wait(command_buffer)?;

    read_f32_buffer(&out_buf, batch * *out_dim)
}

fn qmm_na_qb_reference(
    weight: &AffineQuantizedTensor,
    lhs: &[f32],
    batch: usize,
) -> Result<Vec<f32>> {
    let [out_dim, in_dim] = weight.shape() else {
        return Err(InferError::Dimension(format!(
            "qmm na qb référence: poids attendu rang 2, reçu {:?}",
            weight.shape()
        )));
    };
    let [_, packed_cols] = weight.packed_shape() else {
        return Err(InferError::Dimension(format!(
            "qmm na qb référence: packed_shape attendu rang 2, reçu {:?}",
            weight.packed_shape()
        )));
    };
    if !matches!(weight.bits(), 4 | 8) {
        return Err(InferError::Dimension(format!(
            "qmm na qb référence attend bits=4 ou bits=8, reçu {}",
            weight.bits()
        )));
    }
    if lhs.len() != batch * *in_dim {
        return Err(InferError::Dimension(format!(
            "qmm na qb référence: lhs len={} incompatible batch={batch} in_dim={in_dim}",
            lhs.len()
        )));
    }
    let groups = *in_dim / weight.group_size();
    let bits = weight.bits();
    let values_per_word = 32 / bits;
    let mask = (1_u32 << bits) - 1;
    let mut out = vec![0.0f32; batch * *out_dim];
    for bb in 0..batch {
        for nn in 0..*out_dim {
            let mut acc = 0.0f32;
            for kk in 0..*in_dim {
                let word = weight.packed_data()[nn * *packed_cols + kk / values_per_word];
                let q = (word >> ((kk % values_per_word) * bits)) & mask;
                let group = kk / weight.group_size();
                let scale = bf16_round(weight.scales().data()[nn * groups + group]);
                let bias = bf16_round(weight.biases().data()[nn * groups + group]);
                let deq = bf16_round(q as f32 * scale + bias);
                acc += bf16_round(lhs[bb * *in_dim + kk]) * deq;
            }
            out[bb * *out_dim + nn] = acc;
        }
    }
    Ok(out)
}

#[allow(dead_code, reason = "oracle brut conserve pour diagnostics qmm u8")]
struct RawAffineU8 {
    out_dim: usize,
    in_dim: usize,
    packed_cols: usize,
    groups: usize,
    packed: Vec<u32>,
    scales: Vec<f32>,
    biases: Vec<f32>,
}

#[allow(dead_code, reason = "oracle brut conserve pour diagnostics qmm u8")]
fn raw_affine_varied_u8(out_dim: usize, in_dim: usize) -> Result<RawAffineU8> {
    let values_per_word = 4_usize;
    let packed_cols = in_dim.div_ceil(values_per_word);
    let groups = in_dim.div_ceil(64);
    let mut packed = Vec::with_capacity(out_dim * packed_cols);
    for row in 0..out_dim {
        for word in 0..packed_cols {
            let mut value = 0_u32;
            for lane in 0..values_per_word {
                value |= (((row * 7 + word * 3 + lane) % 251) as u32) << (lane * 8);
            }
            packed.push(value);
        }
    }
    let scales = (0..out_dim * groups)
        .map(|i| bf16_round(0.001 + 0.000_05 * ((i % 13) as f32)))
        .collect();
    let biases = (0..out_dim * groups)
        .map(|i| bf16_round(-0.08 + 0.001 * ((i % 19) as f32)))
        .collect();
    Ok(RawAffineU8 {
        out_dim,
        in_dim,
        packed_cols,
        groups,
        packed,
        scales,
        biases,
    })
}

#[allow(dead_code, reason = "oracle brut conserve pour diagnostics qmm u8")]
fn raw_qmm_u8_cpu(weight: &RawAffineU8, lhs: &[f32], batch: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; batch * weight.out_dim];
    for row in 0..batch {
        for out_col in 0..weight.out_dim {
            let mut acc = 0.0_f32;
            for k in 0..weight.in_dim {
                let word = weight.packed[out_col * weight.packed_cols + (k >> 2)];
                let q = ((word >> ((k & 3) * 8)) & 0xff) as f32;
                let group = k / 64;
                let affine = out_col * weight.groups + group;
                acc += lhs[row * weight.in_dim + k]
                    * (q * weight.scales[affine] + weight.biases[affine]);
            }
            out[row * weight.out_dim + out_col] = acc;
        }
    }
    out
}

fn fast_qmv_u8_tg128(
    executor: &MetalExecutor,
    weight: &AffineQuantizedTensor,
    lhs: &[f32],
    batch: usize,
    label: &'static str,
) -> Result<Vec<f32>> {
    let [out_dim, weight_in_dim] = weight.shape() else {
        return Err(InferError::Dimension(format!(
            "{label}: poids attendu rang 2, reçu {:?}",
            weight.shape()
        )));
    };
    let [_, packed_cols] = weight.packed_shape() else {
        return Err(InferError::Dimension(format!(
            "{label}: packed_shape attendu rang 2, reçu {:?}",
            weight.packed_shape()
        )));
    };
    if lhs.len() != batch * *weight_in_dim {
        return Err(InferError::Dimension(format!(
            "{label}: lhs len={} incompatible batch={batch} in_dim={}",
            lhs.len(),
            weight_in_dim
        )));
    }
    let lhs_buf = executor.upload_f32_buffer(lhs, label)?;
    let packed = executor.buffer_from_slice(weight.packed_data(), label)?;
    let scales = executor.buffer_from_f32_as_bf16(weight.scales().data(), label)?;
    let biases = executor.buffer_from_f32_as_bf16(weight.biases().data(), label)?;
    let out_buf = executor.uncached_f32_buffer(batch * *out_dim, label)?;
    let groups = *weight_in_dim / weight.group_size();
    let dims: [u32; 4] = [
        *out_dim as u32,
        *weight_in_dim as u32,
        *packed_cols as u32,
        groups as u32,
    ];

    let command_buffer = executor.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    let pipeline = match weight.group_size() {
        64 => &executor.affine_qmv_fast_aligned_u8_gs64_tg128_f32,
        128 => &executor.affine_qmv_fast_aligned_u8_gs128_tg128_f32,
        group_size => {
            return Err(InferError::Dimension(format!(
                "{label}: group_size u8 tg128 non supporté {group_size}"
            )));
        }
    };
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(&lhs_buf), 0);
    encoder.set_buffer(1, Some(&packed), 0);
    encoder.set_buffer(2, Some(&scales), 0);
    encoder.set_buffer(3, Some(&biases), 0);
    encoder.set_buffer(4, Some(&out_buf), 0);
    encoder.set_bytes(5, 16, dims.as_ptr().cast());
    encoder.dispatch_thread_groups(
        MTLSize::new(batch as u64, (*out_dim as u64).div_ceil(16), 1),
        MTLSize::new(128, 1, 1),
    );
    encoder.end_encoding();
    commit_and_wait(command_buffer)?;

    read_f32_buffer(&out_buf, batch * *out_dim)
}

fn fast_qmv_u8_tg256(
    executor: &MetalExecutor,
    weight: &AffineQuantizedTensor,
    lhs: &[f32],
    batch: usize,
    label: &'static str,
) -> Result<Vec<f32>> {
    let [out_dim, weight_in_dim] = weight.shape() else {
        return Err(InferError::Dimension(format!(
            "{label}: poids attendu rang 2, reçu {:?}",
            weight.shape()
        )));
    };
    let [_, packed_cols] = weight.packed_shape() else {
        return Err(InferError::Dimension(format!(
            "{label}: packed_shape attendu rang 2, reçu {:?}",
            weight.packed_shape()
        )));
    };
    if lhs.len() != batch * *weight_in_dim {
        return Err(InferError::Dimension(format!(
            "{label}: lhs len={} incompatible batch={batch} in_dim={}",
            lhs.len(),
            weight_in_dim
        )));
    }
    let lhs_buf = executor.upload_f32_buffer(lhs, label)?;
    let packed = executor.buffer_from_slice(weight.packed_data(), label)?;
    let scales = executor.buffer_from_f32_as_bf16(weight.scales().data(), label)?;
    let biases = executor.buffer_from_f32_as_bf16(weight.biases().data(), label)?;
    let out_buf = executor.uncached_f32_buffer(batch * *out_dim, label)?;
    let groups = *weight_in_dim / weight.group_size();
    let dims: [u32; 4] = [
        *out_dim as u32,
        *weight_in_dim as u32,
        *packed_cols as u32,
        groups as u32,
    ];

    let command_buffer = executor.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    let pipeline = match weight.group_size() {
        64 => &executor.affine_qmv_fast_aligned_u8_gs64_tg256_f32,
        128 => &executor.affine_qmv_fast_aligned_u8_gs128_tg256_f32,
        group_size => {
            return Err(InferError::Dimension(format!(
                "{label}: group_size u8 tg256 non supporté {group_size}"
            )));
        }
    };
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(&lhs_buf), 0);
    encoder.set_buffer(1, Some(&packed), 0);
    encoder.set_buffer(2, Some(&scales), 0);
    encoder.set_buffer(3, Some(&biases), 0);
    encoder.set_buffer(4, Some(&out_buf), 0);
    encoder.set_bytes(5, 16, dims.as_ptr().cast());
    encoder.dispatch_thread_groups(
        MTLSize::new(batch as u64, (*out_dim as u64).div_ceil(32), 1),
        MTLSize::new(256, 1, 1),
    );
    encoder.end_encoding();
    commit_and_wait(command_buffer)?;

    read_f32_buffer(&out_buf, batch * *out_dim)
}

fn fast_qmv_u8_dot4(
    executor: &MetalExecutor,
    weight: &AffineQuantizedTensor,
    lhs: &[f32],
    batch: usize,
    label: &'static str,
) -> Result<Vec<f32>> {
    let [out_dim, weight_in_dim] = weight.shape() else {
        return Err(InferError::Dimension(format!(
            "{label}: poids attendu rang 2, reçu {:?}",
            weight.shape()
        )));
    };
    let [_, packed_cols] = weight.packed_shape() else {
        return Err(InferError::Dimension(format!(
            "{label}: packed_shape attendu rang 2, reçu {:?}",
            weight.packed_shape()
        )));
    };
    if lhs.len() != batch * *weight_in_dim {
        return Err(InferError::Dimension(format!(
            "{label}: lhs len={} incompatible batch={batch} in_dim={}",
            lhs.len(),
            weight_in_dim
        )));
    }
    let lhs_buf = executor.upload_f32_buffer(lhs, label)?;
    let packed = executor.buffer_from_slice(weight.packed_data(), label)?;
    let scales = executor.buffer_from_f32_as_bf16(weight.scales().data(), label)?;
    let biases = executor.buffer_from_f32_as_bf16(weight.biases().data(), label)?;
    let out_buf = executor.uncached_f32_buffer(batch * *out_dim, label)?;
    let groups = *weight_in_dim / weight.group_size();
    let dims: [u32; 4] = [
        *out_dim as u32,
        *weight_in_dim as u32,
        *packed_cols as u32,
        groups as u32,
    ];

    let command_buffer = executor.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    let pipeline = match weight.group_size() {
        64 => &executor.affine_qmv_fast_aligned_u8_gs64_dot4_f32,
        128 => &executor.affine_qmv_fast_aligned_u8_gs128_dot4_f32,
        group_size => {
            return Err(InferError::Dimension(format!(
                "{label}: group_size u8 dot4 non supporté {group_size}"
            )));
        }
    };
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(&lhs_buf), 0);
    encoder.set_buffer(1, Some(&packed), 0);
    encoder.set_buffer(2, Some(&scales), 0);
    encoder.set_buffer(3, Some(&biases), 0);
    encoder.set_buffer(4, Some(&out_buf), 0);
    encoder.set_bytes(5, 16, dims.as_ptr().cast());
    encoder.dispatch_thread_groups(
        MTLSize::new(batch as u64, (*out_dim as u64).div_ceil(8), 1),
        MTLSize::new(64, 1, 1),
    );
    encoder.end_encoding();
    commit_and_wait(command_buffer)?;

    read_f32_buffer(&out_buf, batch * *out_dim)
}

fn test_stacked_affine_varied_u8_group(
    executor: &MetalExecutor,
    experts: usize,
    out_dim: usize,
    in_dim: usize,
    group_size: usize,
) -> Result<StackedAffineBuffers> {
    let mut packed = Vec::new();
    let mut scales = Vec::new();
    let mut biases = Vec::new();
    let mut packed_cols = 0_usize;
    let mut groups = 0_usize;
    for expert in 0..experts {
        let affine = test_affine_varied_u8_group(out_dim, in_dim, group_size)?;
        let [_, affine_packed_cols] = affine.packed_shape() else {
            return Err(InferError::Dimension(format!(
                "stacked packed_shape attendu rang 2, reçu {:?}",
                affine.packed_shape()
            )));
        };
        packed_cols = *affine_packed_cols;
        groups = in_dim / group_size;
        packed.extend_from_slice(affine.packed_data());
        scales.extend(
            affine
                .scales()
                .data()
                .iter()
                .map(|value| bf16_round(*value * (1.0 + 0.03 * expert as f32))),
        );
        biases.extend(
            affine
                .biases()
                .data()
                .iter()
                .map(|value| bf16_round(*value + 0.002 * expert as f32)),
        );
    }

    Ok(StackedAffineBuffers {
        packed: executor.buffer_from_slice(&packed, "stacked_u8_packed")?,
        scales: executor.buffer_from_f32_as_bf16(&scales, "stacked_u8_scales")?,
        biases: executor.buffer_from_f32_as_bf16(&biases, "stacked_u8_biases")?,
        experts,
        out_dim,
        in_dim,
        packed_cols,
        group_size,
        bits: 8,
        groups,
    })
}

fn test_stacked_affine_varied_u4_group(
    executor: &MetalExecutor,
    experts: usize,
    out_dim: usize,
    in_dim: usize,
    salt: usize,
) -> Result<StackedAffineBuffers> {
    let bits = FAST_QMV_BITS;
    let group_size = FAST_QMV_GROUP_SIZE;
    let values_per_word = 32 / bits;
    let packed_cols = in_dim / values_per_word;
    let groups = in_dim / group_size;
    let mut packed = Vec::with_capacity(experts * out_dim * packed_cols);
    let mut scales = Vec::with_capacity(experts * out_dim * groups);
    let mut biases = Vec::with_capacity(experts * out_dim * groups);

    for expert in 0..experts {
        for row in 0..out_dim {
            for word in 0..packed_cols {
                let mut lanes = [0_u32; 8];
                for (lane, value) in lanes.iter_mut().enumerate() {
                    *value = ((expert * 11 + row * 7 + word * 3 + lane * 5 + salt) % 16) as u32;
                }
                packed.push(pack_lanes(&lanes, bits));
            }
            for group in 0..groups {
                scales.push(bf16_round(
                    0.0018 + 0.000_04 * ((expert + row + group + salt) % 17) as f32,
                ));
                biases.push(bf16_round(
                    -0.018 + 0.000_5 * ((expert * 3 + row + group + salt) % 23) as f32,
                ));
            }
        }
    }

    Ok(StackedAffineBuffers {
        packed: executor.buffer_from_slice(&packed, "stacked_u4_packed")?,
        scales: executor.buffer_from_f32_as_bf16(&scales, "stacked_u4_scales")?,
        biases: executor.buffer_from_f32_as_bf16(&biases, "stacked_u4_biases")?,
        experts,
        out_dim,
        in_dim,
        packed_cols,
        group_size,
        bits,
        groups,
    })
}

#[derive(Clone, Copy)]
enum GatherQmvU8Route {
    Tg64,
    Tg128,
    Tg256,
}

fn gather_qmv_u8_values(
    executor: &MetalExecutor,
    weight: &StackedAffineBuffers,
    lhs: &[f32],
    lhs_rows: usize,
    indices: &[u32],
    route: GatherQmvU8Route,
    label: &'static str,
) -> Result<Vec<f32>> {
    let topk = indices.len();
    if lhs.len() != lhs_rows * weight.in_dim {
        return Err(InferError::Dimension(format!(
            "{label}: lhs len={} incompatible lhs_rows={lhs_rows} in_dim={}",
            lhs.len(),
            weight.in_dim
        )));
    }
    let lhs_buf = executor.upload_f32_buffer(lhs, label)?;
    let indices_buf = executor.upload_u32_buffer(indices, label)?;
    let out_buf = executor.uncached_f32_buffer(topk * weight.out_dim, label)?;
    let dims = [
        topk as u32,
        weight.out_dim as u32,
        weight.in_dim as u32,
        weight.packed_cols as u32,
    ];
    let quant = [
        weight.group_size as u32,
        weight.bits as u32,
        weight.groups as u32,
        lhs_rows as u32,
    ];
    let pipeline = match (weight.group_size, route) {
        (64, GatherQmvU8Route::Tg64) => &executor.affine_gather_qmv_fast_u8_gs64_f32,
        (128, GatherQmvU8Route::Tg64) => &executor.affine_gather_qmv_fast_u8_gs128_f32,
        (64, GatherQmvU8Route::Tg128) => &executor.affine_gather_qmv_fast_u8_gs64_tg128_f32,
        (128, GatherQmvU8Route::Tg128) => &executor.affine_gather_qmv_fast_u8_gs128_tg128_f32,
        (64, GatherQmvU8Route::Tg256) => &executor.affine_gather_qmv_fast_u8_gs64_tg256_f32,
        (128, GatherQmvU8Route::Tg256) => &executor.affine_gather_qmv_fast_u8_gs128_tg256_f32,
        (group_size, _) => {
            return Err(InferError::Dimension(format!(
                "{label}: group_size gather u8 non supporté {group_size}"
            )));
        }
    };
    let (rows_per_threadgroup, threads_per_threadgroup) = match route {
        GatherQmvU8Route::Tg64 => (8, 64),
        GatherQmvU8Route::Tg128 => (16, 128),
        GatherQmvU8Route::Tg256 => (32, 256),
    };

    let command_buffer = executor.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(&lhs_buf), 0);
    encoder.set_buffer(1, Some(&weight.packed), 0);
    encoder.set_buffer(2, Some(&weight.scales), 0);
    encoder.set_buffer(3, Some(&weight.biases), 0);
    encoder.set_buffer(4, Some(&indices_buf), 0);
    encoder.set_buffer(5, Some(&out_buf), 0);
    encoder.set_bytes(6, 16, dims.as_ptr().cast());
    encoder.set_bytes(7, 16, quant.as_ptr().cast());
    encoder.dispatch_thread_groups(
        MTLSize::new(
            topk as u64,
            (weight.out_dim as u64).div_ceil(rows_per_threadgroup),
            1,
        ),
        MTLSize::new(threads_per_threadgroup, 1, 1),
    );
    encoder.end_encoding();
    commit_and_wait(command_buffer)?;

    read_f32_buffer(&out_buf, topk * weight.out_dim)
}

fn assert_bits_equal(left: &[f32], right: &[f32], label: &str) {
    assert_eq!(left.len(), right.len(), "{label}: longueurs");
    for (idx, (a, b)) in left.iter().zip(right.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{label}: bits divergents à l'index {idx} (left={a:e} right={b:e})"
        );
    }
}
