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
fn embedding_gather_from_index_applies_gemma_scale() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let table_values = [1.0_f32, -2.0, 3.0, -4.0, 0.5, 1.5, -2.5, 4.5];
    let table = executor.upload_f32_buffer(&table_values, "scaled_embedding_table")?;
    let index = executor.upload_u32_buffer(&[1], "scaled_embedding_index")?;
    let output = executor.uncached_f32_buffer(4, "scaled_embedding_output")?;
    let embedding = MetalEmbeddingWeightBuffers::Dense {
        table,
        vocab: 2,
        dim: 4,
    };

    let command_buffer = executor.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    executor.encode_embedding_from_index_buffers_scaled(
        encoder, &embedding, &index, &output, 4, 3.0, false,
    )?;
    encoder.end_encoding();
    commit_and_wait(command_buffer)?;

    assert_eq!(read_f32_buffer(&output, 4)?, vec![1.5, 4.5, -7.5, 13.5]);
    Ok(())
}

#[test]
fn embedding_gather_recast_is_opt_in_and_bf16_exact() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let table_values = [1.001_f32, -2.003, 3.007, -4.015];
    let table = executor.upload_f32_buffer(&table_values, "bf16_embedding_table")?;
    let index = executor.upload_u32_buffer(&[0], "bf16_embedding_index")?;
    let output = executor.uncached_f32_buffer(4, "bf16_embedding_output")?;
    let embedding = MetalEmbeddingWeightBuffers::Dense {
        table,
        vocab: 1,
        dim: 4,
    };

    let command_buffer = executor.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    executor.encode_embedding_from_index_buffers_scaled(
        encoder, &embedding, &index, &output, 4, 1.0, false,
    )?;
    encoder.end_encoding();
    commit_and_wait(command_buffer)?;
    assert_eq!(read_f32_buffer(&output, 4)?, table_values);

    let command_buffer = executor.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    executor.encode_embedding_from_index_buffers_scaled(
        encoder, &embedding, &index, &output, 4, 1.0, true,
    )?;
    encoder.end_encoding();
    commit_and_wait(command_buffer)?;

    let expected = table_values
        .into_iter()
        .map(|value| {
            let bits = value.to_bits();
            let rounding = 0x7fff_u32 + ((bits >> 16) & 1);
            f32::from_bits(bits.wrapping_add(rounding) & 0xffff_0000)
        })
        .collect::<Vec<_>>();
    assert_eq!(read_f32_buffer(&output, 4)?, expected);
    Ok(())
}

#[test]
fn quantized_embedding_batch_survives_cpu_payload_release() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let scales = Tensor::from_vec(vec![2, 1], vec![0.5, 0.25])?;
    let biases = Tensor::from_vec(vec![2, 1], vec![-1.0, 2.0])?;
    let weight = AffineQuantizedTensor::new(
        &[2, 1],
        vec![0x7654_3210, 0x0123_4567],
        scales,
        biases,
        8,
        4,
    )?;
    let mut embedding = EmbeddingWeight::AffineQuantized(weight);
    let token_ids = [1, 0];
    let expected = crate::embed_weight_tokens(&embedding, &token_ids)?;

    let before = executor.embed_weight_tokens(&embedding, &token_ids, 1.0, false)?;
    embedding.release_affine_cpu_data(&mut Vec::new());
    let after = executor.embed_weight_tokens(&embedding, &token_ids, 1.0, false)?;

    assert_eq!(before, expected);
    assert_eq!(after, expected);
    Ok(())
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

#[test]
fn native_bf16_expert_view_keeps_all_three_offsets() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let packed = executor.weight_buffer_from_u32(&[0x1111_1111, 0x2222_2222])?;
    let mut scales_bytes = Vec::with_capacity(4);
    scales_bytes.extend_from_slice(&0x3f00_u16.to_le_bytes());
    scales_bytes.extend_from_slice(&0x3f80_u16.to_le_bytes());
    let scales = executor.weight_buffer_from_bf16_bytes(&scales_bytes)?;
    let mut biases_bytes = Vec::with_capacity(4);
    biases_bytes.extend_from_slice(&0_u16.to_le_bytes());
    biases_bytes.extend_from_slice(&0x4000_u16.to_le_bytes());
    let biases = executor.weight_buffer_from_bf16_bytes(&biases_bytes)?;
    let weight = AffineQuantizedTensor::new_metal_shared_bf16(
        &[1, 1],
        packed.clone(),
        1,
        1,
        &[1, 1],
        scales.clone(),
        1,
        1,
        &[1, 1],
        biases.clone(),
        1,
        1,
        8,
        4,
    )?;

    assert_eq!(weight.row(0)?, vec![4.0; 8]);
    let resolved = executor.resolve_linear_weight_buffers(
        &LinearWeight::AffineQuantized(weight),
        "native_bf16_offset_test",
    )?;
    let MetalLinearWeightBuffers::AffineQuantized {
        packed: resolved_packed,
        packed_offset,
        scales: resolved_scales,
        scales_offset,
        biases: resolved_biases,
        biases_offset,
        ..
    } = resolved
    else {
        return Err(InferError::Config(
            "résolution affine attendue dans le test bf16".to_string(),
        ));
    };
    assert_eq!(resolved_packed.as_ptr(), packed.as_ptr());
    assert_eq!(resolved_scales.as_ptr(), scales.as_ptr());
    assert_eq!(resolved_biases.as_ptr(), biases.as_ptr());
    assert_eq!(packed_offset, std::mem::size_of::<u32>() as u64);
    assert_eq!(scales_offset, std::mem::size_of::<u16>() as u64);
    assert_eq!(biases_offset, std::mem::size_of::<u16>() as u64);
    Ok(())
}

#[test]
fn native_bf16_pair_resolves_as_storage_backed_split() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let make_linear = |word: u32, scale: u16| -> Result<(Linear, metal::Buffer)> {
        let packed = executor.weight_buffer_from_u32(&[word])?;
        let scales = executor.weight_buffer_from_bf16_bytes(&scale.to_le_bytes())?;
        let biases = executor.weight_buffer_from_bf16_bytes(&0_u16.to_le_bytes())?;
        let expected_scales = scales.clone();
        let weight = AffineQuantizedTensor::new_metal_shared_bf16(
            &[1, 1],
            packed,
            0,
            1,
            &[1, 1],
            scales,
            0,
            1,
            &[1, 1],
            biases,
            0,
            1,
            8,
            4,
        )?;
        Ok((Linear::new_quantized(weight, None)?, expected_scales))
    };
    let (first, first_scales) = make_linear(0x1111_1111, 0x3f00)?;
    let (second, second_scales) = make_linear(0x2222_2222, 0x3f80)?;

    let resolved = executor.resolve_linear_attn_pair_weights(
        &first,
        &second,
        "native_bf16_pair_concat",
        "native_bf16_pair_first",
        "native_bf16_pair_second",
    )?;
    let MetalLinearAttnResidentPairWeights::Split { first, second } = resolved else {
        return Err(InferError::Config(
            "une paire bf16 native ne doit pas créer de concat affine".to_string(),
        ));
    };
    let MetalLinearWeightBuffers::AffineQuantized {
        scales: resolved_first,
        ..
    } = first
    else {
        return Err(InferError::Config(
            "premier poids affine attendu".to_string(),
        ));
    };
    let MetalLinearWeightBuffers::AffineQuantized {
        scales: resolved_second,
        ..
    } = second
    else {
        return Err(InferError::Config(
            "second poids affine attendu".to_string(),
        ));
    };
    assert_eq!(resolved_first.as_ptr(), first_scales.as_ptr());
    assert_eq!(resolved_second.as_ptr(), second_scales.as_ptr());
    Ok(())
}

#[test]
fn native_bf16_file_load_writes_directly_into_storage() -> Result<()> {
    use std::io::Write;

    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let mut file = tempfile::NamedTempFile::new().map_err(|source| InferError::Io {
        path: std::env::temp_dir(),
        source,
    })?;
    file.write_all(&[0xaa, 0xbb, 0x00, 0x3f, 0x80, 0x3f])
        .map_err(|source| InferError::Io {
            path: file.path().to_path_buf(),
            source,
        })?;
    let buffer = executor.weight_buffer_from_bf16_file(file.path(), 2, 4)?;

    assert_eq!(read_u16_buffer(&buffer, 2)?, vec![0x3f00, 0x3f80]);
    Ok(())
}
