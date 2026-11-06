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
    let seq = crate::runtime_flags::env_var("SARAGOSSA_FLASH_NEXT_STEPS")
        .and_then(|v| v.parse().ok())
        .unwrap_or(6);
    let hidden = 64;
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

/// Experts u4 distincts à faible coût : packed commun, échelles et biais
/// propres à chaque expert (bf16-représentables).
fn flash_next_expert_u4(base: &AffineQuantizedTensor, expert: usize) -> Result<Linear> {
    let factor = 1.0 + expert as f32 / 300.0;
    let scales: Vec<f32> = base.scales_f32().iter().map(|s| bf16_round(s * factor)).collect();
    let biases: Vec<f32> = base.biases_f32().iter().map(|b| bf16_round(b * factor)).collect();
    let shape = base.scales_shape().to_vec();
    Linear::from_weight(
        LinearWeight::AffineQuantized(AffineQuantizedTensor::new(
            base.packed_shape(),
            base.packed_data().to_vec(),
            Tensor::from_vec(shape.clone(), scales)?,
            Tensor::from_vec(shape, biases)?,
            64,
            4,
        )?),
        None,
    )
}

/// MoE résident Flash-Next : 512 experts, top-10, intermédiaire 640 (queue du
/// down hors alignement 512), routeur et gate partagée denses (copies bf16),
/// expert partagé, résiduel. Oracle CPU : softmax f32 sur les 512 logits, top-10
/// renormalisé (`norm_topk_prob`), experts déquantifiés, `σ(gate)·partagé`.
#[test]
fn flash_next_moe_512_top10_inter640_matches_cpu() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let (hidden, inter, experts_n, top_k) = (1024_usize, 640_usize, 512_usize, 10_usize);
    let bases = [
        test_affine_random_u4(inter, hidden, 31)?,
        test_affine_random_u4(inter, hidden, 32)?,
        test_affine_random_u4(hidden, inter, 33)?,
    ];
    let experts: Vec<GatedMlp> = (0..experts_n)
        .map(|e| {
            Ok(GatedMlp::new(
                flash_next_expert_u4(&bases[0], e)?,
                flash_next_expert_u4(&bases[1], e)?,
                flash_next_expert_u4(&bases[2], e)?,
            ))
        })
        .collect::<Result<_>>()?;
    let shared = GatedMlp::new(
        flash_next_expert_u4(&test_affine_random_u4(inter, hidden, 41)?, 0)?,
        flash_next_expert_u4(&test_affine_random_u4(inter, hidden, 42)?, 0)?,
        flash_next_expert_u4(&test_affine_random_u4(hidden, inter, 43)?, 0)?,
    );
    let mut rng = crate::linear_attention::tests::DeterministicF32::new(0x0512_0010);
    let router_w: Vec<f32> = rng
        .fill(experts_n * hidden, -0.05, 0.05)
        .into_iter()
        .map(bf16_round)
        .collect();
    let router = Linear::from_weight(
        LinearWeight::Dense(Tensor::from_vec(vec![experts_n, hidden], router_w.clone())?),
        None,
    )?;
    let gate_w: Vec<f32> = varied_row(hidden, 977).iter().map(|v| bf16_round(v * 0.02)).collect();
    let shared_gate = Linear::from_weight(
        LinearWeight::Dense(Tensor::from_vec(vec![1, hidden], gate_w.clone())?),
        None,
    )?;
    let weights = executor.resolve_moe_shared_weights(&router, &experts, &shared, &shared_gate)?;
    let x = rng.fill(hidden, -1.0, 1.0);
    let residual = varied_row(hidden, 733);

    // Oracle CPU.
    let dot = |w: &[f32], v: &[f32]| w.iter().zip(v).map(|(a, b)| f64::from(*a) * f64::from(*b)).sum::<f64>();
    let logits: Vec<f32> = router_w.chunks_exact(hidden).map(|w| dot(w, &x) as f32).collect();
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let probs: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
    let mut order: Vec<usize> = (0..experts_n).collect();
    order.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]).then(a.cmp(&b)));
    let margin = probs[order[top_k - 1]] - probs[order[top_k]];
    assert!(margin > 1.0e-4 * probs[order[0]], "fixture à quasi-égalité au seuil : {margin}");
    let chosen = &order[..top_k];
    let total: f32 = chosen.iter().map(|&e| probs[e]).sum();
    let row = Tensor::from_vec(vec![1, hidden], x.clone())?;
    let mlp = |m: &GatedMlp| -> Result<Vec<f32>> { Ok(m.forward(&row)?.data().to_vec()) };
    let mut expected = residual.clone();
    for &e in chosen {
        let y = mlp(&experts[e])?;
        let p = probs[e] / total;
        expected.iter_mut().zip(&y).for_each(|(o, v)| *o += p * v);
    }
    let s = 1.0 / (1.0 + (-(dot(&gate_w, &x) as f32)).exp());
    let y = mlp(&shared)?;
    expected.iter_mut().zip(&y).for_each(|(o, v)| *o += s * v);

    let input = executor.buffer_from_slice(&x, "flash_next_moe_in")?;
    let res = executor.buffer_from_slice(&residual, "flash_next_moe_res")?;
    let out = executor.uncached_f32_buffer(hidden, "flash_next_moe_out")?;
    let command_buffer = executor.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    let mut owned = Vec::new();
    executor.encode_moe_shared_buffers(
        encoder, &mut owned, &input, Some(&res), &out, hidden, &weights, top_k,
    )?;
    encoder.end_encoding();
    commit_and_wait(command_buffer)?;
    let gpu = read_f32_buffer(&out, hidden)?;
    let (err, max_abs) = crate::qwen4_exp::math::relative_error(&gpu, &expected);
    assert!(err <= 1.0e-5, "MoE 512/top-10 : erreur relative {err} (max {max_abs})");

    // Routage imposé (oracles) : le top-10 naturel, donné dans le désordre,
    // redonne la même sortie ; un autre ensemble suit le CPU sur cet ensemble.
    let run_forced = |experts: Vec<u32>| -> Result<Vec<f32>> {
        crate::metal_backend::force_next_topk(experts);
        let command_buffer = executor.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let mut owned = Vec::new();
        executor.encode_moe_shared_buffers(
            encoder, &mut owned, &input, Some(&res), &out, hidden, &weights, top_k,
        )?;
        encoder.end_encoding();
        commit_and_wait(command_buffer)?;
        assert!(!crate::metal_backend::forced_topk_pending(), "imposition non consommée");
        read_f32_buffer(&out, hidden)
    };
    let mut natural: Vec<u32> = chosen.iter().map(|&e| e as u32).collect();
    natural.reverse();
    assert_eq!(run_forced(natural)?, gpu, "top-10 naturel imposé");
    let other: Vec<usize> = order[..top_k - 1].iter().chain([&order[top_k + 5]]).copied().collect();
    let other_total: f32 = other.iter().map(|&e| probs[e]).sum();
    let mut expected_other = residual.clone();
    for &e in &other {
        let y = mlp(&experts[e])?;
        let p = probs[e] / other_total;
        expected_other.iter_mut().zip(&y).for_each(|(o, v)| *o += p * v);
    }
    let y = mlp(&shared)?;
    expected_other.iter_mut().zip(&y).for_each(|(o, v)| *o += s * v);
    let forced = run_forced(other.iter().map(|&e| e as u32).collect())?;
    let (err, _) = crate::qwen4_exp::math::relative_error(&forced, &expected_other);
    assert!(err <= 1.0e-5, "routage imposé : erreur relative {err}");
    let (gap, _) = crate::qwen4_exp::math::relative_error(&forced, &gpu);
    assert!(gap > 1.0e-4, "l'imposition n'a pas changé la sortie : {gap}");
    Ok(())
}

/// Segments du MoE Flash-Next aux formes réelles (512 experts u4, hidden 2560,
/// intermédiaire 640, expert partagé u8, routeur f32 dense) : route, gate/up,
/// down, combinaisons. Diagnostic de débit, machine libre.
#[test]
#[ignore = "diagnostic de débit, machine libre"]
fn bench_flash_next_moe_segments_at_real_dims() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let (hidden, inter, experts_n) = (2560_usize, 640_usize, 512_usize);
    let bases = [
        test_affine_random_u4(inter, hidden, 51)?,
        test_affine_random_u4(inter, hidden, 52)?,
        test_affine_random_u4(hidden, inter, 53)?,
    ];
    let experts: Vec<GatedMlp> = (0..experts_n)
        .map(|e| {
            Ok(GatedMlp::new(
                flash_next_expert_u4(&bases[0], e)?,
                flash_next_expert_u4(&bases[1], e)?,
                flash_next_expert_u4(&bases[2], e)?,
            ))
        })
        .collect::<Result<_>>()?;
    let u8_linear = |rows, cols| -> Result<Linear> {
        Linear::from_weight(LinearWeight::AffineQuantized(test_affine_varied_u8(rows, cols)?), None)
    };
    let shared = GatedMlp::new(u8_linear(inter, hidden)?, u8_linear(inter, hidden)?, u8_linear(hidden, inter)?);
    let mut rng = crate::linear_attention::tests::DeterministicF32::new(0x0b0e);
    let router = Linear::from_weight(
        LinearWeight::Dense(Tensor::from_vec(vec![experts_n, hidden], rng.fill(experts_n * hidden, -0.05, 0.05))?),
        None,
    )?;
    let shared_gate = Linear::from_weight(
        LinearWeight::Dense(Tensor::from_vec(vec![1, hidden], rng.fill(hidden, -0.02, 0.02))?),
        None,
    )?;
    let input = Tensor::from_vec(vec![1, hidden], rng.fill(hidden, -1.0, 1.0))?;
    let empty = || -> Result<()> { commit_and_wait(executor.queue.new_command_buffer()) };
    for _ in 0..8 {
        empty()?;
    }
    let started = std::time::Instant::now();
    for _ in 0..64 {
        empty()?;
    }
    let overhead_ms = started.elapsed().as_secs_f64() * 1e3 / 64.0;
    let report = executor.profile_moe_shared_segments(&input, &router, &experts, 10, &shared, &shared_gate, overhead_ms)?;
    eprintln!("{report}");
    Ok(())
}

/// Morceaux du MoE Flash-Next, 200 appels par command buffer (temps mural par
/// appel) : routeur f32 dense `[512, 2560]`, top-10 softmax, MoE complet.
#[test]
#[ignore = "diagnostic de débit, machine libre"]
fn bench_flash_next_moe_pieces_in_one_command_buffer() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let (hidden, inter, experts_n) = (2560_usize, 640_usize, 512_usize);
    let bases = [
        test_affine_random_u4(inter, hidden, 61)?,
        test_affine_random_u4(inter, hidden, 62)?,
        test_affine_random_u4(hidden, inter, 63)?,
    ];
    let experts: Vec<GatedMlp> = (0..experts_n)
        .map(|e| {
            Ok(GatedMlp::new(
                flash_next_expert_u4(&bases[0], e)?,
                flash_next_expert_u4(&bases[1], e)?,
                flash_next_expert_u4(&bases[2], e)?,
            ))
        })
        .collect::<Result<_>>()?;
    let u8_linear = |rows, cols| -> Result<Linear> {
        Linear::from_weight(LinearWeight::AffineQuantized(test_affine_varied_u8(rows, cols)?), None)
    };
    let shared = GatedMlp::new(u8_linear(inter, hidden)?, u8_linear(inter, hidden)?, u8_linear(hidden, inter)?);
    let mut rng = crate::linear_attention::tests::DeterministicF32::new(0x0b0f);
    let router_w = Tensor::from_vec(vec![experts_n, hidden], rng.fill(experts_n * hidden, -0.05, 0.05))?;
    let router = Linear::from_weight(LinearWeight::Dense(router_w), None)?;
    let shared_gate = Linear::from_weight(
        LinearWeight::Dense(Tensor::from_vec(vec![1, hidden], rng.fill(hidden, -0.02, 0.02))?),
        None,
    )?;
    let weights = executor.resolve_moe_shared_weights(&router, &experts, &shared, &shared_gate)?;
    let router_buffers = executor.resolve_linear_weight_buffers(router.weight(), "bench_router")?;
    let input = executor.buffer_from_slice(&rng.fill(hidden, -1.0, 1.0), "bench_moe_in")?;
    let logits = executor.uncached_f32_buffer(experts_n, "bench_moe_logits")?;
    let indices = executor.uncached_f32_buffer(16, "bench_moe_idx")?;
    let scores = executor.uncached_f32_buffer(16, "bench_moe_scores")?;
    let out = executor.uncached_f32_buffer(hidden, "bench_moe_out")?;
    let reps = 200;
    let time = |label: &str, encode: &dyn Fn(&metal::ComputeCommandEncoderRef, &mut Vec<metal::Buffer>) -> Result<()>| -> Result<()> {
        for warm in [true, false] {
            let command_buffer = executor.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            let mut owned = Vec::new();
            for _ in 0..reps {
                encode(encoder, &mut owned)?;
            }
            encoder.end_encoding();
            let started = std::time::Instant::now();
            commit_and_wait(command_buffer)?;
            if !warm {
                eprintln!("{label:28} {:8.2} µs/appel", started.elapsed().as_secs_f64() * 1e6 / reps as f64);
            }
        }
        Ok(())
    };
    time("routeur f32 [512, 2560]", &|e, _| {
        executor.encode_matmul_weight_buffers(e, &input, 1, hidden, &router_buffers, &logits, false).map(|_| ())
    })?;
    time("top-10 softmax (512)", &|e, owned| {
        executor.encode_topk_softmax(e, owned, &logits, &indices, &scores, experts_n, 10)
    })?;
    time("MoE complet", &|e, owned| {
        executor.encode_moe_shared_buffers(e, owned, &input, None, &out, hidden, &weights, 10)
    })?;
    Ok(())
}

/// Le top-k large rend les mêmes indices et scores, bit à bit, que le kernel
/// parallèle de référence : logits aléatoires, égalités massives (ordre par
/// indice), valeurs non finies, effectifs 257 à 1024 et top-k 1 à 16.
#[test]
fn topk_softmax_wide_matches_parallel_bitwise() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let run = |pipeline: &metal::ComputePipelineState, width: u64, logits: &[f32], topk: usize| -> Result<(Vec<u32>, Vec<f32>)> {
        let input = executor.buffer_from_slice(logits, "topk_wide_in")?;
        let indices = executor.uncached_f32_buffer(16, "topk_wide_idx")?;
        let scores = executor.uncached_f32_buffer(16, "topk_wide_scores")?;
        let command_buffer = executor.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&input), 0);
        encoder.set_buffer(1, Some(&indices), 0);
        encoder.set_buffer(2, Some(&scores), 0);
        set_u32_bytes(encoder, 3, &[logits.len() as u32, topk as u32], "topk_dims")?;
        encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(width, 1, 1));
        encoder.end_encoding();
        commit_and_wait(command_buffer)?;
        Ok((read_u32_buffer(&indices, topk)?, read_f32_buffer(&scores, topk)?))
    };
    let mut rng = crate::linear_attention::tests::DeterministicF32::new(0x70b5);
    let mut cases: Vec<(Vec<f32>, usize)> = Vec::new();
    for (count, topk) in [(512, 10), (257, 1), (1024, 16), (700, 8), (512, 16)] {
        cases.push((rng.fill(count, -4.0, 4.0), topk));
    }
    // Égalités massives au seuil : 40 logits identiques.
    let mut ties = rng.fill(512, -4.0, 0.0);
    for i in (3..512).step_by(12) {
        ties[i] = 1.5;
    }
    cases.push((ties, 10));
    // Valeurs non finies : NaN et −∞ jamais retenus, +∞ en tête.
    let mut odd = rng.fill(512, -1.0, 1.0);
    odd[7] = f32::NAN;
    odd[100] = f32::NEG_INFINITY;
    odd[300] = f32::INFINITY;
    cases.push((odd, 10));
    // Moins de candidats que de places : sentinelles identiques.
    let mut sparse = vec![f32::NEG_INFINITY; 300];
    sparse[5] = 0.5;
    sparse[250] = 0.25;
    cases.push((sparse, 4));
    for (case, (logits, topk)) in cases.iter().enumerate() {
        let reference = run(&executor.topk_softmax_f32, 32, logits, *topk)?;
        let wide = run(&executor.topk_softmax_wide_f32, 1024, logits, *topk)?;
        assert_eq!(wide.0, reference.0, "indices, cas {case}");
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&wide.1), bits(&reference.1), "scores, cas {case}");
    }
    Ok(())
}
