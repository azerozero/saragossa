// Primitives Metal de Flash-Next (`qwen4_exp`) aux formes réelles du modèle.

/// DeltaNet Flash-Next pas à pas sur Metal (16 têtes K, 48 têtes V, dk = dv = 128,
/// conv k4) contre l'oracle naïf : sorties à chaque pas **et** état récurrent
/// final. La moitié des têtes clés a des q/k de très faible norme (Σx² ~ ε),
/// ce qui distingue `x/√(Σx²+ε)` de l'unité RMS `x/√(Σx²+d·ε)`.
#[test]
fn flash_next_linear_attn_resident_steps_match_naive_oracle_at_real_heads() -> Result<()> {
    use crate::linear_attention::tests::{naive_flash_next, DeterministicF32};
    use crate::linear_attention::LinearAttentionConfig;
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let config = LinearAttentionConfig {
        num_key_heads: 16,
        num_value_heads: 48,
        key_head_dim: 128,
        value_head_dim: 128,
        conv_kernel_dim: 4,
        rms_eps: 1.0e-6,
    };
    let (hidden, seq) = (64, 6);
    let (key_dim, value_dim) = (16 * 128, 48 * 128);
    let conv_dim = 2 * key_dim + value_dim;
    let mut rng = DeterministicF32::new(0xf1a5_4e57);
    let mut w_qkv = rng.fill(conv_dim * hidden, -0.12, 0.12);
    // Têtes clés 0..8 de q et de k : amplitude ~1e-4, Σx² comparable à ε.
    for part in [0, key_dim] {
        let rows = part..part + 8 * 128;
        for value in &mut w_qkv[rows.start * hidden..rows.end * hidden] {
            *value *= 1.0e-3;
        }
    }
    let w = [
        w_qkv,
        rng.fill(value_dim * hidden, -0.3, 0.3),
        rng.fill(48 * hidden, -0.2, 0.2),
        rng.fill(48 * hidden, -0.2, 0.2),
        rng.fill(hidden * value_dim, -0.02, 0.02),
    ];
    let conv = rng.fill(conv_dim * 4, -0.6, 0.6);
    let a_log = rng.fill(48, -0.5, 0.5);
    let dt_bias = rng.fill(48, -0.3, 0.3);
    let norm = rng.fill(128, 0.8, 1.2);
    let x = rng.fill(seq * hidden, -1.0, 1.0);
    let dense = |data: &[f32], rows: usize, cols: usize| -> Result<Linear> {
        Linear::from_weight(
            LinearWeight::Dense(Tensor::from_vec(vec![rows, cols], data.to_vec())?),
            None,
        )
    };
    let in_qkv = dense(&w[0], conv_dim, hidden)?;
    let in_z = dense(&w[1], value_dim, hidden)?;
    let in_b = dense(&w[2], 48, hidden)?;
    let in_a = dense(&w[3], 48, hidden)?;
    let out = dense(&w[4], hidden, value_dim)?;
    let conv_weight = Tensor::from_vec(vec![conv_dim, 1, 4], conv.clone())?;
    let a_log_t = Tensor::from_vec(vec![48], a_log.clone())?;
    let dt_bias_t = Tensor::from_vec(vec![48], dt_bias.clone())?;
    let norm_t = Tensor::from_vec(vec![128], norm.clone())?;
    let spec = LinearAttentionStepSpec {
        num_key_heads: 16,
        num_value_heads: 48,
        key_head_dim: 128,
        value_head_dim: 128,
        conv_kernel_dim: 4,
        rms_eps: 1.0e-6,
        l2_qk_sigmoid_gate: true,
    };
    let ssm_len = 48 * 128 * 128;
    let conv_seed = vec![0.0; 3 * conv_dim];
    let ssm_seed = vec![0.0; ssm_len];
    let mut state = None;
    let mut got = Vec::with_capacity(seq * hidden);
    for t in 0..seq {
        let row = Tensor::from_vec(vec![1, hidden], x[t * hidden..(t + 1) * hidden].to_vec())?;
        let y = executor.linear_attention_cached_step_resident(
            &row,
            &in_qkv,
            &in_z,
            &in_b,
            &in_a,
            &out,
            &conv_weight,
            &a_log_t,
            &dt_bias_t,
            &norm_t,
            &conv_seed,
            &ssm_seed,
            &mut state,
            spec,
        )?;
        got.extend_from_slice(y.data());
    }
    let (expected, expected_state) = naive_flash_next(
        &x, seq, hidden, &w, &conv, &a_log, &dt_bias, &norm, config, true,
    );
    let (out_err, _) = crate::qwen4_exp::math::relative_error(&got, &expected);
    assert!(out_err <= 1.0e-5, "sorties : erreur relative {out_err}");
    let state = state.expect("invariant: état alloué au premier pas");
    let ssm = read_f32_buffer(&state.ssm, ssm_len)?;
    let (state_err, _) = crate::qwen4_exp::math::relative_error(&ssm, &expected_state);
    assert!(state_err <= 1.0e-5, "état récurrent : erreur relative {state_err}");

    // L'unité RMS (variante Qwen) s'écarte nettement sur les têtes de faible norme.
    let mut legacy_state = None;
    let legacy_spec = LinearAttentionStepSpec {
        l2_qk_sigmoid_gate: false,
        ..spec
    };
    let row = Tensor::from_vec(vec![1, hidden], x[..hidden].to_vec())?;
    let legacy = executor.linear_attention_cached_step_resident(
        &row,
        &in_qkv,
        &in_z,
        &in_b,
        &in_a,
        &out,
        &conv_weight,
        &a_log_t,
        &dt_bias_t,
        &norm_t,
        &conv_seed,
        &ssm_seed,
        &mut legacy_state,
        legacy_spec,
    )?;
    let (legacy_err, _) =
        crate::qwen4_exp::math::relative_error(legacy.data(), &expected[..hidden]);
    assert!(legacy_err > 1.0e-2, "variante Qwen indiscernable : {legacy_err}");
    Ok(())
}

/// Matmuls résidents aux formes réelles de Flash-Next dont `in_dim` sort du cas
/// aligné 512, plus la tête u6 : routage attendu et sortie contre le CPU
/// (poids déquantifiés, accumulation f64).
#[test]
fn flash_next_resident_qmv_real_shapes_route_and_match_cpu() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let cases: [(&str, usize, usize, usize, AffineMatmulKernel); 4] = [
        // hc Down `[320, 10240]` : in 10240 % 512 = 0, qmv u8 fast ordinaire.
        ("hc_down", 320, 10_240, 8, AffineMatmulKernel::FastQmvU8),
        // hc Up `[10240, 320]` : in 320 % 512 ≠ 0, qmv u8 aligné 64.
        ("hc_up", 10_240, 320, 8, AffineMatmulKernel::FastQmvU8Align64),
        // Down de l'expert partagé `[2560, 640]` : aligné 64.
        ("shared_down", 2560, 640, 8, AffineMatmulKernel::FastQmvU8Align64),
        // lm_head `[248320, 2560]` en u6.
        ("lm_head", 248_320, 2560, 6, AffineMatmulKernel::FastQmvU6),
    ];
    for (label, out_dim, in_dim, bits, expected) in cases {
        let weight = if bits == 6 {
            test_affine_varied_u6(out_dim, in_dim)?
        } else {
            test_affine_varied_u8(out_dim, in_dim)?
        };
        let routed = executor.select_resident_affine_matmul_kernel(1, in_dim, out_dim, 64, bits, false);
        assert_eq!(
            std::mem::discriminant(&routed),
            std::mem::discriminant(&expected),
            "{label}: noyau {routed:?}, attendu {expected:?}"
        );
        let lhs = varied_row(in_dim, 211);
        let dense = weight.dequantize()?;
        let cpu: Vec<f32> = dense
            .data()
            .chunks_exact(in_dim)
            .map(|row| row.iter().zip(&lhs).map(|(w, x)| f64::from(*w) * f64::from(*x)).sum::<f64>() as f32)
            .collect();
        let buffers = executor
            .resolve_linear_weight_buffers(&LinearWeight::AffineQuantized(weight), "flash_next_qmv")?;
        let lhs_buf = executor.upload_f32_buffer(&lhs, "flash_next_qmv_lhs")?;
        let out_buf = executor.uncached_f32_buffer(out_dim, "flash_next_qmv_out")?;
        let command_buffer = executor.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let written =
            executor.encode_matmul_weight_buffers(encoder, &lhs_buf, 1, in_dim, &buffers, &out_buf, false)?;
        assert_eq!(written, out_dim, "{label}");
        encoder.end_encoding();
        commit_and_wait(command_buffer)?;
        let gpu = read_f32_buffer(&out_buf, out_dim)?;
        let (err, max) = crate::qwen4_exp::math::relative_error(&gpu, &cpu);
        assert!(err <= 1.0e-5, "{label}: erreur relative {err} (max {max})");
    }
    Ok(())
}
