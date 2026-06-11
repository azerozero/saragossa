use super::*;

#[test]
fn moe_shared_route_overlap_buffers_are_disjoint() {
    let routed = [
        "moe_shared_router_logits",
        "moe_shared_indices",
        "moe_shared_scores",
        "moe_shared_gate",
        "moe_shared_up",
        "moe_shared_hidden",
        "moe_shared_down",
    ];
    let shared = [
        "moe_shared_gate_scalar",
        "moe_shared_proj_gate",
        "moe_shared_proj_up",
        "moe_shared_proj_hidden",
        "moe_shared_proj_down",
    ];
    let all = routed.into_iter().chain(shared);
    let unique: std::collections::HashSet<_> = all.collect();

    assert_eq!(unique.len(), 12);
    assert_ne!("moe_shared_down", "moe_shared_proj_down");
}

#[test]
fn full_attention_tail_moe_rejects_non_single_batch() -> Result<()> {
    let executor = match test_executor()? {
        Some(executor) => executor,
        None => return Ok(()),
    };
    let residual = Tensor::from_vec(vec![2, 8], vec![0.0; 16])?;
    let context = Tensor::from_vec(vec![1, 8], vec![0.0; 8])?;
    let o_proj = test_dense_linear(8, 8)?;
    let post_norm = Tensor::from_vec(vec![8], vec![1.0; 8])?;
    let router = test_dense_linear(2, 8)?;

    let err = executor
        .full_attention_tail_moe(
            &residual,
            &context,
            &o_proj,
            &post_norm,
            &router,
            &[],
            1,
            1.0e-6,
        )
        .expect_err("invariant: batch attention invalide rejeté");

    assert!(matches!(err, InferError::Dimension(_)));
    Ok(())
}

#[test]
fn full_attention_tail_moe_rejects_bad_norm_shape() -> Result<()> {
    let executor = match test_executor()? {
        Some(executor) => executor,
        None => return Ok(()),
    };
    let residual = Tensor::from_vec(vec![1, 8], vec![0.0; 8])?;
    let context = Tensor::from_vec(vec![1, 8], vec![0.0; 8])?;
    let o_proj = test_dense_linear(8, 8)?;
    let post_norm = Tensor::from_vec(vec![7], vec![1.0; 7])?;
    let router = test_dense_linear(2, 8)?;

    let err = executor
        .full_attention_tail_moe(
            &residual,
            &context,
            &o_proj,
            &post_norm,
            &router,
            &[],
            1,
            1.0e-6,
        )
        .expect_err("invariant: norm attention invalide rejetée");

    assert!(matches!(err, InferError::Dimension(_)));
    Ok(())
}

#[test]
fn moe_shared_rejects_input_dim_mismatch() -> Result<()> {
    let executor = match test_executor()? {
        Some(executor) => executor,
        None => return Ok(()),
    };
    let input = Tensor::from_vec(vec![1, 32], vec![0.0; 32])?;
    let router = test_dense_linear(2, 32)?;
    let experts = vec![
        test_expert(0.001, 0.0005, -0.0003)?,
        test_expert(0.0007, -0.0004, 0.0002)?,
    ];
    let shared_expert = test_expert(0.0008, 0.0002, -0.0001)?;
    let shared_gate = test_dense_linear(1, 32)?;

    let err = executor
        .moe_gated_router_topk_shared(&input, &router, &experts, 1, &shared_expert, &shared_gate)
        .expect_err("invariant: in_dim MoE shared invalide rejeté");

    assert!(matches!(err, InferError::Dimension(_)));
    Ok(())
}

#[test]
fn dense_matmul_matches_cpu() -> Result<()> {
    let executor = match test_executor()? {
        Some(executor) => executor,
        None => return Ok(()),
    };
    let x = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("invariant: shape valide");
    let w = Tensor::from_vec(vec![2, 3], vec![1.0, 0.0, 1.0, 0.0, 1.0, 1.0])
        .expect("invariant: shape valide");

    let cpu = x
        .matmul_rhs_t(&w)
        .expect("invariant: matmul CPU compatible");
    let gpu = executor
        .matmul_rhs_t_dense(&x, &w)
        .expect("invariant: matmul Metal compatible");

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
fn linear_attn_state_snapshot_restore_roundtrip() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let spec = LinearAttentionStepSpec {
        num_key_heads: 2,
        num_value_heads: 4,
        key_head_dim: 8,
        value_head_dim: 8,
        conv_kernel_dim: 4,
        rms_eps: 1.0e-6,
    };
    let conv_dim = 64;
    let conv_len = (spec.conv_kernel_dim - 1) * conv_dim;
    let ssm_len = spec.num_value_heads * spec.value_head_dim * spec.key_head_dim;
    let conv_seed: Vec<f32> = (0..conv_len).map(|i| 1.0 + i as f32).collect();
    let ssm_seed: Vec<f32> = (0..ssm_len).map(|i| 1000.0 + i as f32).collect();

    let mut state = None;
    executor.ensure_linear_attention_metal_state(
        &mut state, &conv_seed, &ssm_seed, conv_dim, conv_len, ssm_len, spec,
    )?;
    let state = state.expect("invariant: état linear-attn alloué");

    // Snapshot pré-verify, puis corruption en place (simule l'avancée du verify).
    let snapshot = executor.snapshot_linear_attn_state(&state)?;
    write_f32_buffer(&state.conv, &vec![0.0; conv_len])?;
    write_f32_buffer(&state.ssm, &vec![0.0; ssm_len])?;
    assert_eq!(read_f32_buffer(&state.conv, conv_len)?[1], 0.0);

    // Rollback : l'état doit revenir bit-pour-bit au snapshot.
    executor.restore_linear_attn_state(&state, &snapshot)?;
    assert_eq!(read_f32_buffer(&state.conv, conv_len)?, conv_seed);
    assert_eq!(read_f32_buffer(&state.ssm, ssm_len)?, ssm_seed);
    Ok(())
}

fn test_expert(gate_scale: f32, up_scale: f32, down_scale: f32) -> Result<GatedMlp> {
    Ok(GatedMlp::new(
        Linear::from_weight(
            LinearWeight::AffineQuantized(test_affine(64, 64, gate_scale)?),
            None,
        )?,
        Linear::from_weight(
            LinearWeight::AffineQuantized(test_affine(64, 64, up_scale)?),
            None,
        )?,
        Linear::from_weight(
            LinearWeight::AffineQuantized(test_affine(8, 64, down_scale)?),
            None,
        )?,
    ))
}

fn test_executor() -> Result<Option<MetalExecutor>> {
    match MetalExecutor::new() {
        Ok(executor) => Ok(Some(executor)),
        Err(InferError::Metal(message)) if message.contains("aucun device") => Ok(None),
        Err(error) => Err(error),
    }
}

fn test_dense_linear(out_dim: usize, in_dim: usize) -> Result<Linear> {
    let data = vec![0.0; out_dim * in_dim];
    Linear::new(Tensor::from_vec(vec![out_dim, in_dim], data)?, None)
}

fn test_affine(out_dim: usize, in_dim: usize, scale: f32) -> Result<AffineQuantizedTensor> {
    let bits = 4;
    let values_per_word = 32 / bits;
    let packed_cols = in_dim / values_per_word;
    let groups = in_dim / 64;
    let mut packed = Vec::with_capacity(out_dim * packed_cols);
    for row in 0..out_dim {
        for word in 0..packed_cols {
            let mut lanes = [0_u32; 8];
            for (lane, value) in lanes.iter_mut().enumerate() {
                *value = ((row + word + lane) % 15 + 1) as u32;
            }
            packed.push(pack_lanes(&lanes, bits));
        }
    }
    let scales = Tensor::from_vec(
        vec![out_dim, groups],
        vec![bf16_round(scale); out_dim * groups],
    )?;
    let biases = Tensor::from_vec(vec![out_dim, groups], vec![0.0; out_dim * groups])?;
    AffineQuantizedTensor::new(&[out_dim, packed_cols], packed, scales, biases, 64, bits)
}

fn pack_lanes(values: &[u32], bits: usize) -> u32 {
    values
        .iter()
        .enumerate()
        .fold(0_u32, |word, (idx, value)| word | (value << (idx * bits)))
}

/// Arrondit `v` à bf16 puis revient en f32 (RNE), identique à la conversion de
/// production des scales/biases. Les oracles GPU-vs-CPU utilisent des scales déjà
/// bf16-représentables : le GPU (qui lit les scales en bf16) et le CPU (qui calcule
/// en f32) partagent alors exactement la même valeur → tolérances inchangées.
fn bf16_round(v: f32) -> f32 {
    let bits = v.to_bits();
    let rounding = 0x7fff + ((bits >> 16) & 1);
    f32::from_bits(((bits + rounding) >> 16) << 16)
}

fn assert_close(left: &[f32], right: &[f32]) {
    assert_close_eps(left, right, 1.0e-5);
}

fn assert_close_eps(left: &[f32], right: &[f32], eps: f32) {
    assert_eq!(left.len(), right.len());
    for (idx, (a, b)) in left.iter().zip(right.iter()).enumerate() {
        assert!((a - b).abs() <= eps, "index={idx} left={a} right={b}");
    }
}
