// Oracles sans poids de modèle pour les bornes des kernels.
#[test]
fn topk_corrupt_hidden_never_exports_sentinel() -> Result<()> {
    let Some(executor) = test_executor()? else { return Ok(()); };
    for (name, pipeline, width, rows, sigmoid) in [
        ("parallel", &executor.topk_softmax_f32, 32, false, false),
        ("wide", &executor.topk_softmax_wide_f32, 1024, false, false),
        ("topk8", &executor.topk8_softmax_256_f32, 32, false, false),
        ("topk8_rows", &executor.topk8_softmax_256_rows_f32, 32, false, false),
        ("sigmoid", &executor.topk_sigmoid_f32, 32, false, true),
        ("serial", &executor.topk_softmax_serial_f32, 32, false, false),
        ("rows", &executor.topk_softmax_rows_f32, 32, true, false),
        ("sigmoid_rows", &executor.topk_sigmoid_rows_f32, 32, true, true),
    ] {
        if pipeline.max_total_threads_per_threadgroup() < width { continue; }
        // Traverser le vrai matmul du routeur depuis un hidden corrompu.
        for invalid in [f32::NAN, f32::NEG_INFINITY] {
            let hidden = Tensor::from_vec(vec![1, 32], vec![invalid; 32])?;
            let router = Tensor::from_vec(vec![256, 32], vec![1.0; 256 * 32])?;
            let corrupt_logits = executor.matmul_rhs_t_dense(&hidden, &router)?;
            assert!(corrupt_logits.data().iter().all(|v| if invalid.is_nan() { v.is_nan() } else { *v == invalid }));
            for finite_count in [0, 2, 8] {
                let mut logits = corrupt_logits.data().to_vec();
                for i in 0..finite_count { logits[10 + i] = 0.0; }
                let input = executor.buffer_from_slice(&logits, "safety_logits")?;
                let indices = executor.uncached_u32_buffer(8, "safety_indices")?;
                let scores = executor.uncached_f32_buffer(8, "safety_scores")?;
                let cb = executor.queue.new_command_buffer();
                let enc = cb.new_compute_command_encoder();
                enc.set_compute_pipeline_state(pipeline);
                enc.set_buffer(0, Some(&input), 0);
                enc.set_buffer(1, Some(&indices), 0);
                enc.set_buffer(2, Some(&scores), 0);
                let dims = if rows { vec![1, 256, 8] } else { vec![256, 8] };
                set_u32_bytes(enc, 3, &dims, "safety_dims")?;
                enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(width, 1, 1));
                enc.end_encoding();
                commit_and_wait(cb)?;
                let indices = read_u32_buffer(&indices, 8)?;
                let scores = read_f32_buffer(&scores, 8)?;
                for slot in 0..8 {
                    assert!(indices[slot] < 256, "{name}: slot {slot}, {indices:?}");
                    assert!(scores[slot].is_finite(), "{name}: {scores:?}");
                    if slot < finite_count {
                        assert_eq!(indices[slot], (10 + slot) as u32, "{name}");
                        assert_eq!(scores[slot], 1.0 / finite_count as f32, "{name}");
                    } else {
                        assert_eq!(scores[slot], 0.0, "{name}: empty slot {slot}");
                        // Sigmoid(-inf) est un candidat de poids nul, pas une sentinelle.
                        if !sigmoid || invalid.is_nan() { assert_eq!(indices[slot], 0, "{name}"); }
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
fn qmv_barriers_have_no_lane_dependent_return() {
    // Une exécution GPU d'une barrière divergente a un comportement indéfini :
    // contrôler le contrat avant de lancer les oracles de queues partielles.
    let source = include_str!("../../kernels.metal");
    for name in [
        "affine_qmv_rms_fast_u4_gs64_f32",
        "affine_qmv_rms_fast_u8_gs64_f32",
        "affine_qmv_rms_fast_u8_gs128_f32",
        "affine_qkv_split_rms_qmv_fast_u4_gs64_f32",
        "affine_qkv_split_rms_qmv_fast_u3_gs64_f32",
        "affine_qkv_split_rms_qmv_fast_u8_gs64_f32",
        "affine_argmax_qmv_fast_u4_gs64_f32",
    ] {
        let body = source.split(&format!("kernel void {name}(")).nth(1).expect("kernel");
        let prologue = body.split("threadgroup_barrier").next().expect("barrier");
        assert!(!prologue.contains("return;"), "{name}: sortie avant barrière");
    }
}

#[test]
fn qmv_rms_partial_groups_match_exact_oracle() -> Result<()> {
    let Some(executor) = test_executor()? else { return Ok(()); };
    for (pipeline, bits, gs, split) in [
        (&executor.affine_qmv_rms_fast_u4_gs64_f32, 4, 64, false),
        (&executor.affine_qmv_rms_fast_u8_gs64_f32, 8, 64, false),
        (&executor.affine_qmv_rms_fast_u8_gs128_f32, 8, 128, false),
        (&executor.affine_qkv_split_rms_qmv_fast_u4_gs64_f32, 4, 64, true),
        (&executor.affine_qkv_split_rms_qmv_fast_u3_gs64_f32, 3, 64, true),
        (&executor.affine_qkv_split_rms_qmv_fast_u8_gs64_f32, 8, 64, true),
    ] {
        for n in 1usize..=17 {
            let k = 512;
            let packed_cols = k * bits / 32;
            let lhs = executor.buffer_from_slice(&vec![1.0f32; k], "tail_lhs")?;
            let packed = executor.buffer_from_slice(&vec![0u32; n * packed_cols], "tail_packed")?;
            let scales = executor.buffer_from_f32_as_bf16(&vec![0.0; n * k / gs], "tail_scales")?;
            let biases = executor.buffer_from_f32_as_bf16(&vec![-1.0; n * k / gs], "tail_biases")?;
            let out = executor.buffer_from_slice(&vec![123.0f32; n + 8], "tail_out")?;
            let q = executor.buffer_from_slice(&[123.0f32; 9], "tail_q")?;
            let gate = executor.buffer_from_slice(&[123.0f32; 9], "tail_gate")?;
            let cb = executor.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(pipeline);
            for (slot, buf) in [&lhs, &lhs, &packed, &scales, &biases, &out].iter().enumerate() {
                enc.set_buffer(slot as u64, Some(buf), 0);
            }
            let dims_slot = if split { 8 } else { 6 };
            set_u32_bytes(enc, dims_slot, &[n as u32, k as u32, packed_cols as u32, (k / gs) as u32], "tail_dims")?;
            if split {
                enc.set_buffer(6, Some(&q), 0);
                enc.set_buffer(7, Some(&gate), 0);
                set_u32_bytes(enc, 9, &[1, 1], "tail_q_dims")?;
            }
            set_f32_bytes(enc, if split { 10 } else { 7 }, &[0.0], "tail_eps")?;
            enc.dispatch_thread_groups(MTLSize::new(1, n.div_ceil(8) as u64, 1), MTLSize::new(64, 1, 1));
            enc.end_encoding();
            commit_and_wait(cb)?;
            let result = read_f32_buffer(&out, n + 8)?;
            for (row, value) in result.iter().enumerate() {
                let expected = if row < n && (!split || row >= 2) { -512.0 } else { 123.0 };
                assert_eq!(*value, expected, "bits={bits}, gs={gs}, split={split}, N={n}, row={row}");
            }
            if split {
                assert_eq!(read_f32_buffer(&q, 9)?, [&[-512.0][..], &[123.0; 8]].concat());
                let first = if n >= 2 { -512.0 } else { 123.0 };
                assert_eq!(read_f32_buffer(&gate, 9)?, [&[first][..], &[123.0; 8]].concat());
            }
        }
    }
    Ok(())
}

#[test]
fn qmv_argmax_partial_groups_ignore_padding() -> Result<()> {
    let Some(executor) = test_executor()? else { return Ok(()); };
    for n in 1usize..=17 {
        let k = 512;
        let lhs = executor.buffer_from_slice(&vec![1.0f32; k], "argmax_tail_lhs")?;
        let packed = executor.buffer_from_slice(&vec![0u32; n * k / 8], "argmax_tail_packed")?;
        let scales = executor.buffer_from_f32_as_bf16(&vec![0.0; n * k / 64], "argmax_tail_scales")?;
        let biases = executor.buffer_from_f32_as_bf16(&vec![-1.0; n * k / 64], "argmax_tail_biases")?;
        let groups = n.div_ceil(8);
        let values = executor.buffer_from_slice(&vec![123.0f32; groups + 1], "argmax_tail_values")?;
        let indices = executor.buffer_from_slice(&vec![123u32; groups + 1], "argmax_tail_indices")?;
        let cb = executor.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&executor.affine_argmax_qmv_fast_u4_gs64_f32);
        for (slot, buf) in [&lhs, &packed, &scales, &biases, &values, &indices].iter().enumerate() {
            enc.set_buffer(slot as u64, Some(buf), 0);
        }
        set_u32_bytes(enc, 6, &[n as u32, k as u32, (k / 8) as u32, (k / 64) as u32], "argmax_tail_dims")?;
        enc.dispatch_thread_groups(MTLSize::new(1, groups as u64, 1), MTLSize::new(64, 1, 1));
        enc.end_encoding();
        commit_and_wait(cb)?;
        let values = read_f32_buffer(&values, groups + 1)?;
        let indices = read_u32_buffer(&indices, groups + 1)?;
        for group in 0..groups {
            assert_eq!(values[group], -512.0, "N={n}");
            assert_eq!(indices[group], (group * 8) as u32, "N={n}");
        }
        assert_eq!(values[groups], 123.0);
        assert_eq!(indices[groups], 123);
    }
    Ok(())
}

#[test]
fn coop_qb_u8_gs64_masks_partial_m_tiles() -> Result<()> {
    let Some(executor) = test_executor()? else { return Ok(()); };
    let Some(pipeline) = &executor.na_gemm_coop_qb else { return Ok(()); };
    for m in [1usize, 15, 16, 17, 31] {
        let (n, k) = (32, 64);
        let padded_m = m.div_ceil(16) * 16;
        let mut input = vec![f32::NAN; padded_m * k];
        input[..m * k].fill(1.0);
        let lhs = executor.buffer_from_f32_as_bf16(&input, "coop_tail_a")?;
        let packed = executor.buffer_from_slice(&vec![0x01010101u32; n * k / 4], "coop_tail_packed")?;
        let scales = executor.buffer_from_f32_as_bf16(&vec![1.0; n], "coop_tail_scales")?;
        let biases = executor.buffer_from_f32_as_bf16(&vec![0.0; n], "coop_tail_biases")?;
        // La réserve rend l'ancien store hors de M observable sans faute GPU.
        let out = executor.buffer_from_slice(&vec![123.0f32; (padded_m + 1) * n], "coop_tail_out")?;
        let cb = executor.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipeline);
        for (slot, buf) in [&lhs, &packed, &scales, &biases, &out].iter().enumerate() {
            enc.set_buffer(slot as u64, Some(buf), 0);
        }
        set_u32_bytes(enc, 5, &[m as u32, n as u32, k as u32, 0], "coop_tail_dims")?;
        enc.dispatch_thread_groups(MTLSize::new(m.div_ceil(16) as u64, 1, 1), MTLSize::new(32, 1, 1));
        enc.end_encoding();
        commit_and_wait(cb)?;
        let output = read_f32_buffer(&out, (padded_m + 1) * n)?;
        assert!(output[..m * n].iter().all(|&v| v == k as f32), "M={m}");
        assert!(output[m * n..].iter().all(|&v| v == 123.0), "store après M={m}");
    }
    Ok(())
}
