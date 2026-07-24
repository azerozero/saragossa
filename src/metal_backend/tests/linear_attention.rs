type NormGateOutputs = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);
type ConvNormOutputs = (
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
);

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

#[test]
fn linear_attn_gated_delta_dk128_tg4_matches_scalar_threadgroups() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let spec = LinearAttentionStepSpec {
        num_key_heads: 2,
        num_value_heads: 4,
        key_head_dim: 128,
        value_head_dim: 128,
        conv_kernel_dim: 4,
        rms_eps: 1.0e-6,
    };
    let repeat = spec.num_value_heads / spec.num_key_heads;
    let key_dim = spec.num_key_heads * spec.key_head_dim;
    let value_dim = spec.num_value_heads * spec.value_head_dim;
    let conv_dim = key_dim * 2 + value_dim;
    let state_len = spec.num_value_heads * spec.value_head_dim * spec.key_head_dim;
    let conv_out: Vec<f32> = (0..conv_dim)
        .map(|idx| ((idx as f32) * 0.013).sin() * 0.5)
        .collect();
    let q_norm: Vec<f32> = (0..key_dim)
        .map(|idx| ((idx as f32) * 0.017).cos() * 0.25)
        .collect();
    let k_norm: Vec<f32> = (0..key_dim)
        .map(|idx| ((idx as f32) * 0.019).sin() * 0.25)
        .collect();
    let beta: Vec<f32> = (0..spec.num_value_heads)
        .map(|idx| 0.2 + idx as f32 * 0.11)
        .collect();
    let decay: Vec<f32> = (0..spec.num_value_heads)
        .map(|idx| 0.91 - idx as f32 * 0.03)
        .collect();
    let state_seed: Vec<f32> = (0..state_len)
        .map(|idx| ((idx as f32) * 0.007).cos() * 0.1)
        .collect();

    let conv_out_buffer = executor.upload_f32_buffer(&conv_out, "delta_dk128_conv_out")?;
    let q_norm_buffer = executor.upload_f32_buffer(&q_norm, "delta_dk128_q_norm")?;
    let k_norm_buffer = executor.upload_f32_buffer(&k_norm, "delta_dk128_k_norm")?;
    let beta_buffer = executor.upload_f32_buffer(&beta, "delta_dk128_beta")?;
    let decay_buffer = executor.upload_f32_buffer(&decay, "delta_dk128_decay")?;
    let dims = [
        spec.num_value_heads as u32,
        spec.value_head_dim as u32,
        spec.key_head_dim as u32,
        repeat as u32,
    ];

    let run_kernel = |pipeline: &ComputePipelineState,
                      dk128_tg_rows: Option<u64>|
     -> Result<(Vec<f32>, Vec<f32>)> {
        let ssm_buffer = executor.upload_f32_buffer(&state_seed, "delta_dk128_ssm")?;
        let y_buffer = executor.uncached_f32_buffer(value_dim, "delta_dk128_y")?;
        let command_buffer = executor.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&conv_out_buffer), 0);
        encoder.set_buffer(1, Some(&q_norm_buffer), 0);
        encoder.set_buffer(2, Some(&k_norm_buffer), 0);
        encoder.set_buffer(3, Some(&beta_buffer), 0);
        encoder.set_buffer(4, Some(&decay_buffer), 0);
        encoder.set_buffer(5, Some(&ssm_buffer), 0);
        encoder.set_buffer(6, Some(&y_buffer), 0);
        set_u32_bytes(encoder, 7, &dims, "delta_dk128_dims")?;
        if let Some(tg_rows) = dk128_tg_rows {
            encoder.dispatch_threads(
                MTLSize::new(32, spec.value_head_dim as u64, spec.num_value_heads as u64),
                MTLSize::new(32, tg_rows, 1),
            );
        } else {
            encoder.dispatch_thread_groups(
                MTLSize::new(spec.value_head_dim as u64, spec.num_value_heads as u64, 1),
                MTLSize::new(32, 1, 1),
            );
        }
        encoder.end_encoding();
        commit_and_wait(command_buffer)?;
        Ok((
            read_f32_buffer(&y_buffer, value_dim)?,
            read_f32_buffer(&ssm_buffer, state_len)?,
        ))
    };

    let (old_y, old_state) = run_kernel(&executor.linear_attn_gated_delta_f32, None)?;
    for tg_rows in [4, 8, 16, 32] {
        let (dk128_y, dk128_state) = run_kernel(
            &executor.linear_attn_gated_delta_dk128_tg4_f32,
            Some(tg_rows),
        )?;
        assert_bits_equal(&dk128_y, &old_y, "linear-attn dk128 y");
        assert_bits_equal(&dk128_state, &old_state, "linear-attn dk128 state");
    }

    let mut bf16_ref_state: Vec<f32> = state_seed.iter().map(|value| bf16_round(*value)).collect();
    let mut bf16_ref_y = vec![0.0_f32; value_dim];
    for value_head in 0..spec.num_value_heads {
        let key_head = value_head / repeat;
        let key_base = key_head * spec.key_head_dim;
        let d = decay[value_head];
        for value_col in 0..spec.value_head_dim {
            let value_index = value_head * spec.value_head_dim + value_col;
            let state_base = value_index * spec.key_head_dim;
            let mut kv_mem = 0.0_f32;
            for col in 0..spec.key_head_dim {
                let state_index = state_base + col;
                let decayed = bf16_ref_state[state_index] * d;
                bf16_ref_state[state_index] = decayed;
                kv_mem += decayed * k_norm[key_base + col];
            }
            let v = conv_out[key_dim * 2 + value_index];
            let delta = (v - kv_mem) * beta[value_head];
            let mut y = 0.0_f32;
            for col in 0..spec.key_head_dim {
                let state_index = state_base + col;
                let updated = bf16_ref_state[state_index] + delta * k_norm[key_base + col];
                bf16_ref_state[state_index] = updated;
                y += updated * q_norm[key_base + col];
            }
            bf16_ref_y[value_index] = y;
        }
    }
    for value in &mut bf16_ref_state {
        *value = bf16_round(*value);
    }

    let ssm_bf16_buffer = executor.buffer_from_f32_as_bf16(&state_seed, "delta_dk128_ssm_bf16")?;
    let y_bf16_buffer = executor.uncached_f32_buffer(value_dim, "delta_dk128_y_bf16")?;
    let command_buffer = executor.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&executor.linear_attn_gated_delta_dk128_bf16_tg4_f32);
    encoder.set_buffer(0, Some(&conv_out_buffer), 0);
    encoder.set_buffer(1, Some(&q_norm_buffer), 0);
    encoder.set_buffer(2, Some(&k_norm_buffer), 0);
    encoder.set_buffer(3, Some(&beta_buffer), 0);
    encoder.set_buffer(4, Some(&decay_buffer), 0);
    encoder.set_buffer(5, Some(&ssm_bf16_buffer), 0);
    encoder.set_buffer(6, Some(&y_bf16_buffer), 0);
    set_u32_bytes(encoder, 7, &dims, "delta_dk128_bf16_dims")?;
    encoder.dispatch_threads(
        MTLSize::new(32, spec.value_head_dim as u64, spec.num_value_heads as u64),
        MTLSize::new(32, 4, 1),
    );
    encoder.end_encoding();
    commit_and_wait(command_buffer)?;
    let bf16_y = read_f32_buffer(&y_bf16_buffer, value_dim)?;
    let bf16_state_words = read_u16_buffer(&ssm_bf16_buffer, state_len)?;
    let bf16_state: Vec<f32> = bf16_state_words
        .iter()
        .map(|word| f32::from_bits(u32::from(*word) << 16))
        .collect();
    assert_close_eps(&bf16_y, &bf16_ref_y, 5.0e-4);
    assert_close_eps(&bf16_state, &bf16_ref_state, 2.0e-4);
    Ok(())
}

#[test]
fn linear_attn_gated_delta_seq_dk128_matches_step_loop() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let spec = LinearAttentionStepSpec {
        num_key_heads: 2,
        num_value_heads: 4,
        key_head_dim: 128,
        value_head_dim: 128,
        conv_kernel_dim: 4,
        rms_eps: 1.0e-6,
    };
    let steps = 5_usize;
    let repeat = spec.num_value_heads / spec.num_key_heads;
    let key_dim = spec.num_key_heads * spec.key_head_dim;
    let value_dim = spec.num_value_heads * spec.value_head_dim;
    let conv_dim = key_dim * 2 + value_dim;
    let state_len = spec.num_value_heads * spec.value_head_dim * spec.key_head_dim;
    let conv_out: Vec<f32> = (0..steps * conv_dim)
        .map(|idx| ((idx as f32) * 0.011).sin() * 0.5)
        .collect();
    let q_norm: Vec<f32> = (0..steps * key_dim)
        .map(|idx| ((idx as f32) * 0.017).cos() * 0.25)
        .collect();
    let k_norm: Vec<f32> = (0..steps * key_dim)
        .map(|idx| ((idx as f32) * 0.019).sin() * 0.25)
        .collect();
    let beta: Vec<f32> = (0..steps * spec.num_value_heads)
        .map(|idx| 0.2 + (idx % spec.num_value_heads) as f32 * 0.07)
        .collect();
    let decay: Vec<f32> = (0..steps * spec.num_value_heads)
        .map(|idx| 0.91 - (idx % spec.num_value_heads) as f32 * 0.025)
        .collect();
    let state_seed: Vec<f32> = (0..state_len)
        .map(|idx| ((idx as f32) * 0.007).cos() * 0.1)
        .collect();
    let dims = [
        spec.num_value_heads as u32,
        spec.value_head_dim as u32,
        spec.key_head_dim as u32,
        repeat as u32,
    ];

    let conv_out_buffer = executor.upload_f32_buffer(&conv_out, "delta_seq_conv_out")?;
    let q_norm_buffer = executor.upload_f32_buffer(&q_norm, "delta_seq_q_norm")?;
    let k_norm_buffer = executor.upload_f32_buffer(&k_norm, "delta_seq_k_norm")?;
    let beta_buffer = executor.upload_f32_buffer(&beta, "delta_seq_beta")?;
    let decay_buffer = executor.upload_f32_buffer(&decay, "delta_seq_decay")?;

    let loop_ssm = executor.upload_f32_buffer(&state_seed, "delta_seq_loop_ssm")?;
    let loop_y = executor.uncached_f32_buffer(steps * value_dim, "delta_seq_loop_y")?;
    let command_buffer = executor.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    for step in 0..steps {
        encoder.set_compute_pipeline_state(&executor.linear_attn_gated_delta_dk128_tg4_f32);
        encoder.set_buffer(
            0,
            Some(&conv_out_buffer),
            byte_offset_f32(step * conv_dim, "delta seq loop conv")?,
        );
        encoder.set_buffer(
            1,
            Some(&q_norm_buffer),
            byte_offset_f32(step * key_dim, "delta seq loop q")?,
        );
        encoder.set_buffer(
            2,
            Some(&k_norm_buffer),
            byte_offset_f32(step * key_dim, "delta seq loop k")?,
        );
        encoder.set_buffer(
            3,
            Some(&beta_buffer),
            byte_offset_f32(step * spec.num_value_heads, "delta seq loop beta")?,
        );
        encoder.set_buffer(
            4,
            Some(&decay_buffer),
            byte_offset_f32(step * spec.num_value_heads, "delta seq loop decay")?,
        );
        encoder.set_buffer(5, Some(&loop_ssm), 0);
        encoder.set_buffer(
            6,
            Some(&loop_y),
            byte_offset_f32(step * value_dim, "delta seq loop y")?,
        );
        set_u32_bytes(encoder, 7, &dims, "delta_seq_loop_dims")?;
        encoder.dispatch_threads(
            MTLSize::new(32, spec.value_head_dim as u64, spec.num_value_heads as u64),
            MTLSize::new(32, 4, 1),
        );
    }
    encoder.end_encoding();
    commit_and_wait(command_buffer)?;

    let seq_ssm = executor.upload_f32_buffer(&state_seed, "delta_seq_ssm")?;
    let seq_y = executor.uncached_f32_buffer(steps * value_dim, "delta_seq_y")?;
    let command_buffer = executor.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&executor.linear_attn_gated_delta_seq_dk128_tg4_f32);
    encoder.set_buffer(0, Some(&conv_out_buffer), 0);
    encoder.set_buffer(1, Some(&q_norm_buffer), 0);
    encoder.set_buffer(2, Some(&k_norm_buffer), 0);
    encoder.set_buffer(3, Some(&beta_buffer), 0);
    encoder.set_buffer(4, Some(&decay_buffer), 0);
    encoder.set_buffer(5, Some(&seq_ssm), 0);
    encoder.set_buffer(6, Some(&seq_y), 0);
    set_u32_bytes(encoder, 7, &dims, "delta_seq_dims")?;
    set_u32_bytes(encoder, 8, &[steps as u32], "delta_seq_steps")?;
    encoder.dispatch_threads(
        MTLSize::new(32, spec.value_head_dim as u64, spec.num_value_heads as u64),
        MTLSize::new(32, 4, 1),
    );
    encoder.end_encoding();
    commit_and_wait(command_buffer)?;

    assert_close_eps(
        &read_f32_buffer(&seq_y, steps * value_dim)?,
        &read_f32_buffer(&loop_y, steps * value_dim)?,
        1.0e-5,
    );
    assert_close_eps(
        &read_f32_buffer(&seq_ssm, state_len)?,
        &read_f32_buffer(&loop_ssm, state_len)?,
        1.0e-5,
    );

    let mut bf16_ref_state: Vec<f32> = state_seed.iter().map(|value| bf16_round(*value)).collect();
    let mut bf16_ref_y = vec![0.0_f32; steps * value_dim];
    for step in 0..steps {
        let key_offset = step * key_dim;
        let value_offset = step * value_dim;
        let conv_offset = step * conv_dim;
        let gate_offset = step * spec.num_value_heads;
        for value_head in 0..spec.num_value_heads {
            let key_head = value_head / repeat;
            let key_base = key_head * spec.key_head_dim;
            let d = decay[gate_offset + value_head];
            let beta_v = beta[gate_offset + value_head];
            for value_col in 0..spec.value_head_dim {
                let value_index = value_head * spec.value_head_dim + value_col;
                let state_base = value_index * spec.key_head_dim;
                for col in 0..spec.key_head_dim {
                    bf16_ref_state[state_base + col] *= d;
                }

                let mut kv_mem = 0.0_f32;
                for col in 0..spec.key_head_dim {
                    kv_mem +=
                        bf16_ref_state[state_base + col] * k_norm[key_offset + key_base + col];
                }
                let v = conv_out[conv_offset + (2 * key_dim) + value_index];
                let delta = (v - kv_mem) * beta_v;
                for col in 0..spec.key_head_dim {
                    bf16_ref_state[state_base + col] += delta * k_norm[key_offset + key_base + col];
                }

                let mut y = 0.0_f32;
                for col in 0..spec.key_head_dim {
                    y += bf16_ref_state[state_base + col] * q_norm[key_offset + key_base + col];
                }
                bf16_ref_y[value_offset + value_index] = y;
            }
        }
    }
    for value in &mut bf16_ref_state {
        *value = bf16_round(*value);
    }

    let seq_bf16_ssm = executor.buffer_from_f32_as_bf16(&state_seed, "delta_seq_bf16_ssm")?;
    let seq_bf16_y = executor.uncached_f32_buffer(steps * value_dim, "delta_seq_bf16_y")?;
    let command_buffer = executor.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&executor.linear_attn_gated_delta_seq_dk128_bf16_tg4_f32);
    encoder.set_buffer(0, Some(&conv_out_buffer), 0);
    encoder.set_buffer(1, Some(&q_norm_buffer), 0);
    encoder.set_buffer(2, Some(&k_norm_buffer), 0);
    encoder.set_buffer(3, Some(&beta_buffer), 0);
    encoder.set_buffer(4, Some(&decay_buffer), 0);
    encoder.set_buffer(5, Some(&seq_bf16_ssm), 0);
    encoder.set_buffer(6, Some(&seq_bf16_y), 0);
    set_u32_bytes(encoder, 7, &dims, "delta_seq_bf16_dims")?;
    set_u32_bytes(encoder, 8, &[steps as u32], "delta_seq_bf16_steps")?;
    encoder.dispatch_threads(
        MTLSize::new(32, spec.value_head_dim as u64, spec.num_value_heads as u64),
        MTLSize::new(32, 4, 1),
    );
    encoder.end_encoding();
    commit_and_wait(command_buffer)?;

    let bf16_y = read_f32_buffer(&seq_bf16_y, steps * value_dim)?;
    let bf16_state_words = read_u16_buffer(&seq_bf16_ssm, state_len)?;
    let bf16_state: Vec<f32> = bf16_state_words
        .into_iter()
        .map(|word| f32::from_bits((word as u32) << 16))
        .collect();
    assert_close_eps(&bf16_y, &bf16_ref_y, 5.0e-4);
    assert_close_eps(&bf16_state, &bf16_ref_state, 2.0e-4);
    Ok(())
}

#[test]
fn linear_attn_conv_k4_matches_generic() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let conv_dim = 97_usize;
    let kernel = 4_usize;
    let qkv: Vec<f32> = (0..conv_dim)
        .map(|idx| ((idx as f32) * 0.019).sin())
        .collect();
    let conv_weight: Vec<f32> = (0..conv_dim * kernel)
        .map(|idx| ((idx as f32) * 0.013).cos() * 0.2)
        .collect();
    let state_seed: Vec<f32> = (0..conv_dim * (kernel - 1))
        .map(|idx| ((idx as f32) * 0.007).sin() * 0.5)
        .collect();
    let qkv_buffer = executor.upload_f32_buffer(&qkv, "conv_k4_qkv")?;
    let weight_buffer = executor.upload_f32_buffer(&conv_weight, "conv_k4_weight")?;
    let dims = [conv_dim as u32, kernel as u32];

    let run_kernel = |pipeline: &ComputePipelineState| -> Result<(Vec<f32>, Vec<f32>)> {
        let state_buffer = executor.upload_f32_buffer(&state_seed, "conv_k4_state")?;
        let out_buffer = executor.uncached_f32_buffer(conv_dim, "conv_k4_out")?;
        let command_buffer = executor.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&qkv_buffer), 0);
        encoder.set_buffer(1, Some(&weight_buffer), 0);
        encoder.set_buffer(2, Some(&state_buffer), 0);
        encoder.set_buffer(3, Some(&out_buffer), 0);
        set_u32_bytes(encoder, 4, &dims, "conv_k4_dims")?;
        encoder.dispatch_threads(MTLSize::new(conv_dim as u64, 1, 1), MTLSize::new(256, 1, 1));
        encoder.end_encoding();
        commit_and_wait(command_buffer)?;
        Ok((
            read_f32_buffer(&out_buffer, conv_dim)?,
            read_f32_buffer(&state_buffer, conv_dim * (kernel - 1))?,
        ))
    };

    let (generic_out, generic_state) = run_kernel(&executor.linear_attn_conv_silu_f32)?;
    let (k4_out, k4_state) = run_kernel(&executor.linear_attn_conv_silu_k4_f32)?;
    assert_bits_equal(&k4_out, &generic_out, "linear-attn conv k4 out");
    assert_bits_equal(&k4_state, &generic_state, "linear-attn conv k4 state");
    Ok(())
}

#[test]
fn linear_attn_norm_gates_dk128_matches_generic() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let key_heads = 3_usize;
    let value_heads = 5_usize;
    let key_head_dim = 128_usize;
    let key_dim = key_heads * key_head_dim;
    let conv_out_len = key_dim * 2 + value_heads * key_head_dim;
    let conv_out: Vec<f32> = (0..conv_out_len)
        .map(|idx| ((idx as f32) * 0.009).sin() * 0.4)
        .collect();
    let beta_input: Vec<f32> = (0..value_heads)
        .map(|idx| -0.5 + idx as f32 * 0.17)
        .collect();
    let gate_input: Vec<f32> = (0..value_heads)
        .map(|idx| -0.2 + idx as f32 * 0.11)
        .collect();
    let a_log: Vec<f32> = (0..value_heads)
        .map(|idx| -1.0 + idx as f32 * 0.03)
        .collect();
    let dt_bias: Vec<f32> = (0..value_heads)
        .map(|idx| 0.1 + idx as f32 * 0.02)
        .collect();
    let conv_out_buffer = executor.upload_f32_buffer(&conv_out, "norm_dk128_conv_out")?;
    let beta_input_buffer = executor.upload_f32_buffer(&beta_input, "norm_dk128_beta_in")?;
    let gate_input_buffer = executor.upload_f32_buffer(&gate_input, "norm_dk128_gate_in")?;
    let a_log_buffer = executor.upload_f32_buffer(&a_log, "norm_dk128_a_log")?;
    let dt_bias_buffer = executor.upload_f32_buffer(&dt_bias, "norm_dk128_dt_bias")?;
    let dims = [
        key_heads as u32,
        value_heads as u32,
        key_head_dim as u32,
        key_head_dim as u32,
    ];
    let inv = (key_head_dim as f32).powf(-0.5);
    let scales = [inv * inv, inv];

    let run_kernel =
        |pipeline: &ComputePipelineState| -> Result<NormGateOutputs> {
            let q_norm_buffer = executor.uncached_f32_buffer(key_dim, "norm_dk128_q")?;
            let k_norm_buffer = executor.uncached_f32_buffer(key_dim, "norm_dk128_k")?;
            let beta_buffer = executor.uncached_f32_buffer(value_heads, "norm_dk128_beta")?;
            let decay_buffer = executor.uncached_f32_buffer(value_heads, "norm_dk128_decay")?;
            let command_buffer = executor.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(&conv_out_buffer), 0);
            encoder.set_buffer(1, Some(&beta_input_buffer), 0);
            encoder.set_buffer(2, Some(&gate_input_buffer), 0);
            encoder.set_buffer(3, Some(&a_log_buffer), 0);
            encoder.set_buffer(4, Some(&dt_bias_buffer), 0);
            encoder.set_buffer(5, Some(&q_norm_buffer), 0);
            encoder.set_buffer(6, Some(&k_norm_buffer), 0);
            encoder.set_buffer(7, Some(&beta_buffer), 0);
            encoder.set_buffer(8, Some(&decay_buffer), 0);
            set_u32_bytes(encoder, 9, &dims, "norm_dk128_dims")?;
            set_f32_bytes(encoder, 10, &scales, "norm_dk128_scales")?;
            encoder.dispatch_thread_groups(
                MTLSize::new(value_heads as u64, 1, 1),
                MTLSize::new(32, 1, 1),
            );
            encoder.end_encoding();
            commit_and_wait(command_buffer)?;
            Ok((
                read_f32_buffer(&q_norm_buffer, key_dim)?,
                read_f32_buffer(&k_norm_buffer, key_dim)?,
                read_f32_buffer(&beta_buffer, value_heads)?,
                read_f32_buffer(&decay_buffer, value_heads)?,
            ))
        };

    let generic = run_kernel(&executor.linear_attn_norm_gates_f32)?;
    let dk128 = run_kernel(&executor.linear_attn_norm_gates_dk128_f32)?;
    assert_close_eps(&dk128.0, &generic.0, 1.0e-7);
    assert_close_eps(&dk128.1, &generic.1, 1.0e-7);
    assert_bits_equal(&dk128.2, &generic.2, "linear-attn norm dk128 beta");
    assert_bits_equal(&dk128.3, &generic.3, "linear-attn norm dk128 decay");
    Ok(())
}

#[test]
fn linear_attn_conv_norm_fused_matches_two_step() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let key_heads = 3_usize;
    let value_heads = 5_usize;
    let head_dim = 128_usize;
    let key_dim = key_heads * head_dim;
    let value_dim = value_heads * head_dim;
    let conv_dim = key_dim * 2 + value_dim;
    let kernel = 4_usize;
    let qkv: Vec<f32> = (0..conv_dim)
        .map(|idx| ((idx as f32) * 0.009).sin() * 0.3)
        .collect();
    let conv_weight: Vec<f32> = (0..conv_dim * kernel)
        .map(|idx| ((idx as f32) * 0.013).cos() * 0.2)
        .collect();
    let state_seed: Vec<f32> = (0..conv_dim * (kernel - 1))
        .map(|idx| ((idx as f32) * 0.007).sin() * 0.5)
        .collect();
    let beta_input: Vec<f32> = (0..value_heads)
        .map(|idx| -0.5 + idx as f32 * 0.17)
        .collect();
    let gate_input: Vec<f32> = (0..value_heads)
        .map(|idx| -0.2 + idx as f32 * 0.11)
        .collect();
    let a_log: Vec<f32> = (0..value_heads)
        .map(|idx| -1.0 + idx as f32 * 0.03)
        .collect();
    let dt_bias: Vec<f32> = (0..value_heads)
        .map(|idx| 0.1 + idx as f32 * 0.02)
        .collect();
    let qkv_buffer = executor.upload_f32_buffer(&qkv, "conv_norm_qkv")?;
    let weight_buffer = executor.upload_f32_buffer(&conv_weight, "conv_norm_weight")?;
    let beta_input_buffer = executor.upload_f32_buffer(&beta_input, "conv_norm_beta_in")?;
    let gate_input_buffer = executor.upload_f32_buffer(&gate_input, "conv_norm_gate_in")?;
    let a_log_buffer = executor.upload_f32_buffer(&a_log, "conv_norm_a_log")?;
    let dt_bias_buffer = executor.upload_f32_buffer(&dt_bias, "conv_norm_dt_bias")?;
    let conv_dims = [conv_dim as u32, kernel as u32];
    let norm_dims = [
        key_heads as u32,
        value_heads as u32,
        head_dim as u32,
        head_dim as u32,
    ];
    let inv = (head_dim as f32).powf(-0.5);
    let scales = [inv * inv, inv];

    let run_two_step =
        || -> Result<ConvNormOutputs> {
            let state_buffer = executor.upload_f32_buffer(&state_seed, "conv_norm_state_a")?;
            let conv_out_buffer = executor.uncached_f32_buffer(conv_dim, "conv_norm_out_a")?;
            let q_norm_buffer = executor.uncached_f32_buffer(key_dim, "conv_norm_q_a")?;
            let k_norm_buffer = executor.uncached_f32_buffer(key_dim, "conv_norm_k_a")?;
            let beta_buffer = executor.uncached_f32_buffer(value_heads, "conv_norm_beta_a")?;
            let decay_buffer = executor.uncached_f32_buffer(value_heads, "conv_norm_decay_a")?;
            let command_buffer = executor.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&executor.linear_attn_conv_silu_f32);
            encoder.set_buffer(0, Some(&qkv_buffer), 0);
            encoder.set_buffer(1, Some(&weight_buffer), 0);
            encoder.set_buffer(2, Some(&state_buffer), 0);
            encoder.set_buffer(3, Some(&conv_out_buffer), 0);
            set_u32_bytes(encoder, 4, &conv_dims, "conv_norm_conv_dims")?;
            encoder.dispatch_threads(MTLSize::new(conv_dim as u64, 1, 1), MTLSize::new(256, 1, 1));
            encoder.set_compute_pipeline_state(&executor.linear_attn_norm_gates_f32);
            encoder.set_buffer(0, Some(&conv_out_buffer), 0);
            encoder.set_buffer(1, Some(&beta_input_buffer), 0);
            encoder.set_buffer(2, Some(&gate_input_buffer), 0);
            encoder.set_buffer(3, Some(&a_log_buffer), 0);
            encoder.set_buffer(4, Some(&dt_bias_buffer), 0);
            encoder.set_buffer(5, Some(&q_norm_buffer), 0);
            encoder.set_buffer(6, Some(&k_norm_buffer), 0);
            encoder.set_buffer(7, Some(&beta_buffer), 0);
            encoder.set_buffer(8, Some(&decay_buffer), 0);
            set_u32_bytes(encoder, 9, &norm_dims, "conv_norm_norm_dims")?;
            set_f32_bytes(encoder, 10, &scales, "conv_norm_norm_scales")?;
            encoder.dispatch_thread_groups(
                MTLSize::new(value_heads as u64, 1, 1),
                MTLSize::new(32, 1, 1),
            );
            encoder.end_encoding();
            commit_and_wait(command_buffer)?;
            Ok((
                read_f32_buffer(&conv_out_buffer, conv_dim)?,
                read_f32_buffer(&q_norm_buffer, key_dim)?,
                read_f32_buffer(&k_norm_buffer, key_dim)?,
                read_f32_buffer(&beta_buffer, value_heads)?,
                read_f32_buffer(&decay_buffer, value_heads)?,
                read_f32_buffer(&state_buffer, conv_dim * (kernel - 1))?,
            ))
        };

    let run_fused = || -> Result<ConvNormOutputs> {
        let state_buffer = executor.upload_f32_buffer(&state_seed, "conv_norm_state_b")?;
        let conv_out_buffer = executor.uncached_f32_buffer(conv_dim, "conv_norm_out_b")?;
        let q_norm_buffer = executor.uncached_f32_buffer(key_dim, "conv_norm_q_b")?;
        let k_norm_buffer = executor.uncached_f32_buffer(key_dim, "conv_norm_k_b")?;
        let beta_buffer = executor.uncached_f32_buffer(value_heads, "conv_norm_beta_b")?;
        let decay_buffer = executor.uncached_f32_buffer(value_heads, "conv_norm_decay_b")?;
        let command_buffer = executor.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&executor.linear_attn_conv_norm_gates_k4_dk128_f32);
        encoder.set_buffer(0, Some(&qkv_buffer), 0);
        encoder.set_buffer(1, Some(&beta_input_buffer), 0);
        encoder.set_buffer(2, Some(&gate_input_buffer), 0);
        encoder.set_buffer(3, Some(&weight_buffer), 0);
        encoder.set_buffer(4, Some(&state_buffer), 0);
        encoder.set_buffer(5, Some(&a_log_buffer), 0);
        encoder.set_buffer(6, Some(&dt_bias_buffer), 0);
        encoder.set_buffer(7, Some(&conv_out_buffer), 0);
        encoder.set_buffer(8, Some(&q_norm_buffer), 0);
        encoder.set_buffer(9, Some(&k_norm_buffer), 0);
        encoder.set_buffer(10, Some(&beta_buffer), 0);
        encoder.set_buffer(11, Some(&decay_buffer), 0);
        set_u32_bytes(encoder, 12, &norm_dims, "conv_norm_fused_dims")?;
        set_f32_bytes(encoder, 13, &scales, "conv_norm_fused_scales")?;
        encoder.dispatch_thread_groups(
            MTLSize::new(value_heads as u64, 1, 1),
            MTLSize::new(32, 1, 1),
        );
        encoder.end_encoding();
        commit_and_wait(command_buffer)?;
        Ok((
            read_f32_buffer(&conv_out_buffer, conv_dim)?,
            read_f32_buffer(&q_norm_buffer, key_dim)?,
            read_f32_buffer(&k_norm_buffer, key_dim)?,
            read_f32_buffer(&beta_buffer, value_heads)?,
            read_f32_buffer(&decay_buffer, value_heads)?,
            read_f32_buffer(&state_buffer, conv_dim * (kernel - 1))?,
        ))
    };

    let two_step = run_two_step()?;
    let fused = run_fused()?;
    let value_offset = key_dim * 2;
    assert_bits_equal(
        &fused.0[value_offset..],
        &two_step.0[value_offset..],
        "linear-attn conv_norm fused v",
    );
    assert_close_eps(&fused.1, &two_step.1, 1.0e-7);
    assert_close_eps(&fused.2, &two_step.2, 1.0e-7);
    assert_bits_equal(&fused.3, &two_step.3, "linear-attn conv_norm beta");
    assert_bits_equal(&fused.4, &two_step.4, "linear-attn conv_norm decay");
    assert_bits_equal(&fused.5, &two_step.5, "linear-attn conv_norm state");
    Ok(())
}

#[test]
fn linear_attn_rms_gate_dv128_matches_generic() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let value_heads = 3_usize;
    let value_head_dim = 128_usize;
    let value_dim = value_heads * value_head_dim;
    let y: Vec<f32> = (0..value_dim)
        .map(|idx| ((idx as f32) * 0.011).sin() * 0.75)
        .collect();
    let z: Vec<f32> = (0..value_dim)
        .map(|idx| ((idx as f32) * 0.017).cos() * 0.5)
        .collect();
    let norm_weight: Vec<f32> = (0..value_head_dim)
        .map(|idx| 0.5 + (idx as f32) * 0.001)
        .collect();
    let y_buffer = executor.upload_f32_buffer(&y, "rms_dv128_y")?;
    let z_buffer = executor.upload_f32_buffer(&z, "rms_dv128_z")?;
    let norm_buffer = executor.upload_f32_buffer(&norm_weight, "rms_dv128_norm")?;
    let dims = [value_heads as u32, value_head_dim as u32];
    let eps = 1.0e-6_f32;

    let run_kernel = |pipeline: &ComputePipelineState| -> Result<Vec<f32>> {
        let out_buffer = executor.uncached_f32_buffer(value_dim, "rms_dv128_out")?;
        let command_buffer = executor.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&y_buffer), 0);
        encoder.set_buffer(1, Some(&z_buffer), 0);
        encoder.set_buffer(2, Some(&norm_buffer), 0);
        encoder.set_buffer(3, Some(&out_buffer), 0);
        set_u32_bytes(encoder, 4, &dims, "rms_dv128_dims")?;
        set_f32_bytes(encoder, 5, &[eps], "rms_dv128_eps")?;
        encoder.dispatch_thread_groups(
            MTLSize::new(value_heads as u64, 1, 1),
            MTLSize::new(32, 1, 1),
        );
        encoder.end_encoding();
        commit_and_wait(command_buffer)?;
        read_f32_buffer(&out_buffer, value_dim)
    };

    let generic = run_kernel(&executor.linear_attn_rms_gate_f32)?;
    let dv128 = run_kernel(&executor.linear_attn_rms_gate_dv128_f32)?;
    assert_close_eps(&dv128, &generic, 1.0e-7);
    Ok(())
}
