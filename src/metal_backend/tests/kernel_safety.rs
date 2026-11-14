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
