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
fn scratch_namespace_isolates_label_keyed_buffers() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    // Namespace 0 (défaut) : même label+taille → même buffer mémoïsé.
    let base_a = executor.scratch_buffer(64, MetalBufferElement::F32, "ns_test_scratch")?;
    let base_b = executor.scratch_buffer(64, MetalBufferElement::F32, "ns_test_scratch")?;
    assert_eq!(base_a.contents(), base_b.contents());

    // Slot 1 : buffer DISJOINT du slot 0 (anti-aliasing inter-flux), mémoïsé
    // dans son propre namespace.
    let slot_1 = {
        let _guard = install_scratch_namespace(1);
        let first = executor.scratch_buffer(64, MetalBufferElement::F32, "ns_test_scratch")?;
        let second = executor.scratch_buffer(64, MetalBufferElement::F32, "ns_test_scratch")?;
        assert_eq!(first.contents(), second.contents());
        first
    };
    assert_ne!(base_a.contents(), slot_1.contents());

    // La garde RAII restaure le namespace précédent → on retombe sur le slot 0.
    let restored = executor.scratch_buffer(64, MetalBufferElement::F32, "ns_test_scratch")?;
    assert_eq!(base_a.contents(), restored.contents());
    Ok(())
}

#[test]
fn scratch_namespace_guard_restores_nested_scopes() {
    let _outer = install_scratch_namespace(7);
    assert_eq!(current_scratch_namespace(), 7);
    {
        let _inner = install_scratch_namespace(9);
        assert_eq!(current_scratch_namespace(), 9);
    }
    assert_eq!(current_scratch_namespace(), 7);
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
fn dense_qmv_fast_matches_dense_kernel_on_router_shape() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let batch = 2_usize;
    let out_dim = 256_usize;
    let in_dim = 2048_usize;
    let lhs: Vec<f32> = (0..batch * in_dim)
        .map(|idx| (((idx * 37 + 11) % 127) as f32 - 63.0) / 89.0)
        .collect();
    let rhs: Vec<f32> = (0..out_dim * in_dim)
        .map(|idx| (((idx * 19 + 7) % 131) as f32 - 65.0) / 97.0)
        .collect();
    let lhs_buf = executor.upload_f32_buffer(&lhs, "dense_fast_lhs")?;
    let rhs_buf = executor.cached_buffer_from_f32(&rhs, "dense_fast_rhs")?;
    let out_dense = executor.uncached_f32_buffer(batch * out_dim, "dense_fast_ref")?;
    let out_fast = executor.uncached_f32_buffer(batch * out_dim, "dense_fast_out")?;
    let dims = [batch as u32, out_dim as u32, in_dim as u32];

    let command_buffer = executor.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&executor.dense_matmul_rhs_t_f32);
    encoder.set_buffer(0, Some(&lhs_buf), 0);
    encoder.set_buffer(1, Some(&rhs_buf), 0);
    encoder.set_buffer(2, Some(&out_dense), 0);
    set_u32_bytes(encoder, 3, &dims, "dense_fast_ref_dims")?;
    encoder.dispatch_thread_groups(
        MTLSize::new(out_dim as u64, batch as u64, 1),
        MTLSize::new(32, 1, 1),
    );
    encoder.set_compute_pipeline_state(&executor.dense_qmv_fast_f32);
    encoder.set_buffer(0, Some(&lhs_buf), 0);
    encoder.set_buffer(1, Some(&rhs_buf), 0);
    encoder.set_buffer(2, Some(&out_fast), 0);
    set_u32_bytes(encoder, 3, &dims, "dense_fast_dims")?;
    encoder.dispatch_thread_groups(
        MTLSize::new(batch as u64, (out_dim as u64).div_ceil(8), 1),
        MTLSize::new(64, 1, 1),
    );
    encoder.end_encoding();
    commit_and_wait(command_buffer)?;

    let dense = read_f32_buffer(&out_dense, batch * out_dim)?;
    let fast = read_f32_buffer(&out_fast, batch * out_dim)?;
    assert_bits_equal(&fast, &dense, "dense qmv fast");
    Ok(())
}
