/// Sonde 5 : gated delta prefill, `seq_dk128` f32/bf16 contre NAX (hv32 et hv48).
#[test]
#[ignore = "kbench"]
fn kb_gated_delta() -> Result<()> {
    use crate::linear_attention::tests::{synthetic_gdn_inputs, DeterministicF32};
    use crate::linear_attention::LinearAttentionConfig;
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let reps = kb_reps();
    for hv in [32_usize, 48] {
        let config = LinearAttentionConfig {
            num_key_heads: 16,
            num_value_heads: hv,
            key_head_dim: 128,
            value_head_dim: 128,
            conv_kernel_dim: 4,
            rms_eps: 1.0e-6,
        };
        let spec = LinearAttentionStepSpec {
            num_key_heads: 16,
            num_value_heads: hv,
            key_head_dim: 128,
            value_head_dim: 128,
            conv_kernel_dim: 4,
            rms_eps: 1.0e-6,
            l2_qk_sigmoid_gate: false,
        };
        let key_dim = config.key_dim()?;
        let value_dim = config.value_dim()?;
        let conv_dim = 2 * key_dim + value_dim;
        let state_len = hv * 128 * 128;
        for steps in [512_usize, 4096, 8192] {
            let inputs = synthetic_gdn_inputs(config, steps, 0x6b62 + steps as u64);
            let mut rng = DeterministicF32::new(0xc0de);
            let mut conv_out = rng.fill(steps * conv_dim, -1.0, 1.0);
            for t in 0..steps {
                conv_out[t * conv_dim + 2 * key_dim..t * conv_dim + 2 * key_dim + value_dim]
                    .copy_from_slice(&inputs.v[t * value_dim..(t + 1) * value_dim]);
            }
            let state0 = rng.fill(state_len, -0.1, 0.1);
            let up = |d: &[f32]| executor.upload_f32_buffer(d, "kb_gd");
            let conv_buf = up(&conv_out)?;
            let q_buf = up(&inputs.q)?;
            let k_buf = up(&inputs.k)?;
            let beta_buf = up(&inputs.beta)?;
            let g_buf = up(&inputs.g)?;
            let state_f32 = executor.uncached_f32_buffer(state_len, "kb_st")?;
            let state0_buf = up(&state0)?;
            let state_bf16 = executor.private_bf16_buffer(state_len, "kb_st16")?;
            let y_buf = executor.uncached_f32_buffer(steps * value_dim, "kb_y")?;
            let modes: &[&str] = &["seq f32", "seq bf16", "nax"];
            let mut seq_reference: Option<(Vec<f32>, Vec<f32>)> = None;
            for &mode in modes {
                let mut ts = Vec::new();
                for _ in 0..reps {
                    // Réinitialise l'état hors chronométrage.
                    {
                        let cb = executor.queue.new_command_buffer();
                        let enc = cb.new_compute_command_encoder();
                        if mode == "seq bf16" {
                            executor.encode_f32_to_bf16(
                                enc,
                                &state0_buf,
                                &state_bf16,
                                state_len,
                            )?;
                        } else {
                            executor.encode_copy(enc, &state0_buf, &state_f32, state_len)?;
                        }
                        enc.end_encoding();
                        commit_and_wait(cb)?;
                    }
                    let cb = executor.queue.new_command_buffer();
                    let enc = cb.new_compute_command_encoder();
                    let guard = EncoderEndGuard::new(enc);
                    let r = match mode {
                        "nax" => executor
                            .encode_gated_delta_nax(
                                enc, &conv_buf, &q_buf, &k_buf, &beta_buf, &g_buf, &state_f32,
                                &y_buf, steps, spec,
                            )
                            .map(|ok| assert!(ok, "nax hors périmètre")),
                        "seq bf16" => executor.encode_linear_attn_gated_delta_seq_dk128(
                            enc,
                            &conv_buf,
                            &q_buf,
                            &k_buf,
                            &beta_buf,
                            &g_buf,
                            &state_bf16,
                            true,
                            &y_buf,
                            steps,
                            spec,
                        ),
                        _ => executor.encode_linear_attn_gated_delta_seq_dk128(
                            enc, &conv_buf, &q_buf, &k_buf, &beta_buf, &g_buf, &state_f32, false,
                            &y_buf, steps, spec,
                        ),
                    };
                    guard.end();
                    r?;
                    let t = std::time::Instant::now();
                    commit_and_wait(cb)?;
                    ts.push(t.elapsed().as_secs_f64());
                }
                if mode != "seq bf16" {
                    let y = read_f32_buffer(&y_buf, steps * value_dim)?;
                    let state = read_f32_buffer(&state_f32, state_len)?;
                    if let Some((ref seq_y, ref seq_state)) = seq_reference {
                        let gap = y
                            .iter()
                            .zip(seq_y)
                            .chain(state.iter().zip(seq_state))
                            .map(|(a, b)| (a - b).abs())
                            .fold(0.0_f32, f32::max);
                        eprintln!("KB gdn hv{hv} T={steps} écart NAX/seq {gap:e}");
                        // hv32 est le nouveau chemin : même borne que l'oracle CPU.
                        // hv48 garde son kernel historique à précision relâchée ;
                        // sa sonde longue reste une mesure, comme dans kbench.
                        if hv == 32 {
                            assert_close_eps(&y, seq_y, 5.0e-4);
                            assert_close_eps(&state, seq_state, 5.0e-4);
                        }
                    } else {
                        seq_reference = Some((y, state));
                    }
                }
                let (mn, md, mx) = kb_stats(ts);
                eprintln!(
                    "KB gdn hk16/hv{hv} T={steps} {mode}: min {:.3} med {:.3} max {:.3} ms, {:.2} us/token/couche (med)",
                    mn * 1e3,
                    md * 1e3,
                    mx * 1e3,
                    md * 1e6 / steps as f64
                );
            }
        }
    }
    Ok(())
}

#[test]
fn gated_delta_nax_unsupported_shape_or_pipeline_falls_back() -> Result<()> {
    let Some(mut executor) = test_executor()? else {
        return Ok(());
    };
    let buffer = executor.uncached_f32_buffer(1, "nax_rejected")?;
    let cb = executor.queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    let guard = EncoderEndGuard::new(enc);
    let run = |executor: &MetalExecutor, hk, hv, dk, dv| {
        executor.encode_gated_delta_nax(
            enc,
            &buffer,
            &buffer,
            &buffer,
            &buffer,
            &buffer,
            &buffer,
            &buffer,
            1,
            LinearAttentionStepSpec {
                num_key_heads: hk,
                num_value_heads: hv,
                key_head_dim: dk,
                value_head_dim: dv,
                conv_kernel_dim: 4,
                rms_eps: 1.0e-6,
                l2_qk_sigmoid_gate: false,
            },
        )
    };
    for (hk, hv, dk, dv) in [
        (8, 32, 128, 128),
        (16, 16, 128, 128),
        (16, 32, 64, 128),
        (16, 48, 128, 64),
    ] {
        assert!(!run(&executor, hk, hv, dk, dv)?);
    }
    executor.gated_delta_nax_hk16_hv32 = None;
    executor.gated_delta_nax_hk16_hv48 = None;
    assert!(!run(&executor, 16, 32, 128, 128)?);
    assert!(!run(&executor, 16, 48, 128, 128)?);
    guard.end();
    commit_and_wait(cb)
}
