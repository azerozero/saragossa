use super::*;

fn test_executor() -> Result<Option<MetalExecutor>> {
    match MetalExecutor::new() {
        Ok(executor) => Ok(Some(executor)),
        Err(InferError::Metal(message)) if message.contains("aucun device") => Ok(None),
        Err(error) => Err(error),
    }
}

fn deterministic_value(seed: usize, index: usize, scale: f32) -> f32 {
    let mixed = seed
        .wrapping_mul(97)
        .wrapping_add(index.wrapping_mul(37))
        .wrapping_add(index.wrapping_mul(index).wrapping_mul(11));
    let centered = (mixed % 257) as f32 - 128.0;
    centered * scale / 128.0
}

fn deterministic_tensor(rows: usize, cols: usize, seed: usize, scale: f32) -> Result<Tensor> {
    Tensor::from_vec(
        vec![rows, cols],
        (0..rows * cols)
            .map(|index| deterministic_value(seed, index, scale))
            .collect(),
    )
}

fn deterministic_norm(dim: usize, seed: usize) -> Result<Tensor> {
    Tensor::from_vec(
        vec![dim],
        (0..dim)
            .map(|index| 0.9 + deterministic_value(seed, index, 0.12))
            .collect(),
    )
}

fn bf16_round(value: f32) -> f32 {
    let bits = value.to_bits();
    let rounding = 0x7fff + ((bits >> 16) & 1);
    f32::from_bits(((bits + rounding) >> 16) << 16)
}

fn deterministic_affine_weight(
    rows: usize,
    cols: usize,
    seed: usize,
    scale: f32,
) -> Result<AffineQuantizedTensor> {
    const BITS: usize = 8;
    const VALUES_PER_WORD: usize = 32 / BITS;
    if cols % VALUES_PER_WORD != 0 {
        return Err(InferError::Dimension(format!(
            "fixture affine cols={cols} non multiple de {VALUES_PER_WORD}"
        )));
    }
    let packed_cols = cols / VALUES_PER_WORD;
    let mut packed = Vec::with_capacity(rows * packed_cols);
    for row in 0..rows {
        for word in 0..packed_cols {
            let mut value = 0_u32;
            for lane in 0..VALUES_PER_WORD {
                let column = word * VALUES_PER_WORD + lane;
                let quant = ((seed + row * 17 + column * 31 + lane * 7) % 23 + 3) as u32;
                value |= quant << (lane * BITS);
            }
            packed.push(value);
        }
    }
    let scales = Tensor::from_vec(
        vec![rows, 1],
        (0..rows)
            .map(|index| bf16_round(scale * (0.7 + ((seed + index) % 5) as f32 * 0.05)))
            .collect(),
    )?;
    let biases = Tensor::from_vec(
        vec![rows, 1],
        (0..rows)
            .map(|index| bf16_round(deterministic_value(seed + 97, index, scale * 3.0)))
            .collect(),
    )?;
    AffineQuantizedTensor::new(&[rows, packed_cols], packed, scales, biases, cols, BITS)
}

fn deterministic_gemma_experts(
    count: usize,
    hidden: usize,
    inter: usize,
    seed: usize,
) -> Result<Vec<GatedMlp>> {
    let mut experts = Vec::with_capacity(count);
    for expert in 0..count {
        let gate = deterministic_affine_weight(inter, hidden, seed + expert * 101 + 1, 0.0035)?;
        let up = deterministic_affine_weight(inter, hidden, seed + expert * 101 + 17, 0.003)?;
        let down = deterministic_affine_weight(hidden, inter, seed + expert * 101 + 41, 0.0025)?;
        experts.push(
            GatedMlp::new(
                Linear::new_quantized(gate, None)?,
                Linear::new_quantized(up, None)?,
                Linear::new_quantized(down, None)?,
            )
            .with_activation(crate::Activation::GeluTanh),
        );
    }
    Ok(experts)
}

fn deterministic_qwen_experts(
    count: usize,
    hidden: usize,
    inter: usize,
    seed: usize,
) -> Result<Vec<GatedMlp>> {
    let mut experts = Vec::with_capacity(count);
    for expert in 0..count {
        let gate = deterministic_affine_weight(inter, hidden, seed + expert * 101 + 1, 0.0035)?;
        let up = deterministic_affine_weight(inter, hidden, seed + expert * 101 + 17, 0.003)?;
        let down = deterministic_affine_weight(hidden, inter, seed + expert * 101 + 41, 0.0025)?;
        experts.push(GatedMlp::new(
            Linear::new_quantized(gate, None)?,
            Linear::new_quantized(up, None)?,
            Linear::new_quantized(down, None)?,
        ));
    }
    Ok(experts)
}

fn zero_linear(rows: usize, cols: usize) -> Result<Linear> {
    Linear::new(Tensor::zeros(vec![rows, cols])?, None)
}

fn full_cache(caches: &[PrefillResidentLayerCache]) -> Result<(&Tensor, &Tensor)> {
    match caches {
        [PrefillResidentLayerCache::Full { key, value }] => Ok((key, value)),
        _ => Err(InferError::Dimension(format!(
            "oracle prefill attend un cache full unique, reçu {}",
            caches.len()
        ))),
    }
}

fn assert_bit_exact(actual: &Tensor, expected: &Tensor, label: &str) {
    assert_eq!(
        actual.shape(),
        expected.shape(),
        "{label}: formes différentes"
    );
    for (index, (&actual, &expected)) in actual.data().iter().zip(expected.data()).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "{label}[{index}] actual={actual} expected={expected}"
        );
    }
}

fn assert_close(actual: &Tensor, expected: &Tensor, tolerance: f32, label: &str) {
    assert_eq!(
        actual.shape(),
        expected.shape(),
        "{label}: formes différentes"
    );
    for (index, (&actual, &expected)) in actual.data().iter().zip(expected.data()).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{label}[{index}] actual={actual} expected={expected} tolérance={tolerance}"
        );
    }
}

/// Le tail dense Gemma batché résident reproduit sa référence CPU ligne par ligne.
#[test]
fn gemma_dense_tail_prefill_resident_matches_cpu() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    const SEQ: usize = 264;
    const HIDDEN: usize = 16;
    const INTER: usize = 24;
    const EPS: f32 = 1.0e-6;
    const LAYER_SCALE: f32 = 0.95;

    let hidden = deterministic_tensor(SEQ, HIDDEN, 7, 1.7)?;
    let pre_norm = deterministic_norm(HIDDEN, 19)?;
    let post_norm = deterministic_norm(HIDDEN, 31)?;
    let gate_proj = Linear::new(deterministic_tensor(INTER, HIDDEN, 43, 0.09)?, None)?;
    let up_proj = Linear::new(deterministic_tensor(INTER, HIDDEN, 59, 0.08)?, None)?;
    let down_proj = Linear::new(deterministic_tensor(HIDDEN, INTER, 71, 0.07)?, None)?;
    let layer_scalar = Tensor::from_vec(vec![1], vec![LAYER_SCALE])?;

    let ffn_input = crate::rms_norm(&hidden, &pre_norm, EPS)?;
    let gate = gate_proj.forward(&ffn_input)?;
    let up = up_proj.forward(&ffn_input)?;
    let geglu = crate::gelu_tanh(&gate).mul_elementwise(&up)?;
    let down = down_proj.forward(&geglu)?;
    let ffn_normed = crate::rms_norm(&down, &post_norm, EPS)?;
    let oracle = hidden.add(&ffn_normed)?.map(|value| value * LAYER_SCALE);

    let gpu = executor.gemma_dense_tail_prefill_resident(
        &hidden,
        PrefillMoeTail::GemmaDense {
            gate_proj: &gate_proj,
            up_proj: &up_proj,
            down_proj: &down_proj,
            pre_feedforward_norm: &pre_norm,
            post_feedforward_norm: &post_norm,
            layer_scalar: Some(&layer_scalar),
            inter_dim: INTER,
        },
        EPS,
    )?;

    // Near-tie documenté : l'oracle per-op garde RMSNorm et gelu_tanh côté CPU,
    // alors que cette brique résidente les évalue dans Metal (réductions par
    // threadgroup + fast-math `tanh`). Les écarts attendus restent au niveau ULP.
    let tolerance = 2.0e-5;
    for row in 0..SEQ {
        let actual = gpu.row_slice(row)?;
        let expected = oracle.row_slice(row)?;
        for (column, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "ligne={row} colonne={column} GPU={actual} CPU={expected} tolérance={tolerance}"
            );
        }
    }
    Ok(())
}

/// Le tail parallèle Gemma batché reproduit les cinq normes et le résidu CPU.
#[test]
fn gemma_parallel_tail_prefill_resident_matches_cpu() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    const SEQ: usize = 264;
    const HIDDEN: usize = 16;
    const DENSE_INTER: usize = 24;
    const MOE_INTER: usize = 16;
    const EXPERTS: usize = 4;
    const TOP_K: usize = 2;
    const EPS: f32 = 1.0e-6;
    const LAYER_SCALE: f32 = 0.95;

    let hidden = deterministic_tensor(SEQ, HIDDEN, 103, 1.3)?;
    let pre_ffn = deterministic_norm(HIDDEN, 109)?;
    let post_ffn_1 = deterministic_norm(HIDDEN, 127)?;
    let pre_ffn_2 = deterministic_norm(HIDDEN, 149)?;
    let post_ffn_2 = deterministic_norm(HIDDEN, 163)?;
    let post_ffn = deterministic_norm(HIDDEN, 181)?;
    let router_norm = deterministic_norm(HIDDEN, 191)?;
    let per_expert_scale = Tensor::from_vec(vec![EXPERTS], vec![0.85, 1.1, 0.7, 1.25])?;
    let dense_gate = Linear::new(deterministic_tensor(DENSE_INTER, HIDDEN, 197, 0.08)?, None)?;
    let dense_up = Linear::new(deterministic_tensor(DENSE_INTER, HIDDEN, 211, 0.07)?, None)?;
    let dense_down = Linear::new(deterministic_tensor(HIDDEN, DENSE_INTER, 223, 0.06)?, None)?;
    let router = Linear::new(deterministic_tensor(EXPERTS, HIDDEN, 239, 0.19)?, None)?;
    let experts = deterministic_gemma_experts(EXPERTS, HIDDEN, MOE_INTER, 251)?;
    let layer_scalar = Tensor::from_vec(vec![1], vec![LAYER_SCALE])?;

    let dense_input = crate::rms_norm(&hidden, &pre_ffn, EPS)?;
    let dense_gate_out = dense_gate.forward(&dense_input)?;
    let dense_up_out = dense_up.forward(&dense_input)?;
    let dense_geglu = crate::gelu_tanh(&dense_gate_out).mul_elementwise(&dense_up_out)?;
    let dense_raw = dense_down.forward(&dense_geglu)?;
    let dense_out = crate::rms_norm(&dense_raw, &post_ffn_1, EPS)?;
    let moe_input = crate::rms_norm(&hidden, &pre_ffn_2, EPS)?;
    let moe = crate::MoeMlp::new(router.clone(), experts.clone(), None, None, TOP_K)?
        .with_router_norm(router_norm.clone(), EPS)
        .with_per_expert_scale(per_expert_scale.clone());
    let moe_raw =
        moe.forward_with_router_source(&moe_input, &hidden, crate::ForwardRuntime::cpu())?;
    let moe_out = crate::rms_norm(&moe_raw, &post_ffn_2, EPS)?;
    let ffn_out = dense_out.add(&moe_out)?;
    let ffn_normed = crate::rms_norm(&ffn_out, &post_ffn, EPS)?;
    let oracle = hidden.add(&ffn_normed)?.map(|value| value * LAYER_SCALE);

    let gpu = executor.gemma_parallel_tail_prefill_resident(
        &hidden,
        PrefillMoeTail::GemmaParallel {
            dense_gate_proj: &dense_gate,
            dense_up_proj: &dense_up,
            dense_down_proj: &dense_down,
            pre_feedforward_norm: &pre_ffn,
            post_feedforward_norm_1: &post_ffn_1,
            router: &router,
            experts: &experts,
            top_k: TOP_K,
            router_norm: Some((&router_norm, EPS)),
            per_expert_scale: Some(&per_expert_scale),
            pre_feedforward_norm_2: &pre_ffn_2,
            post_feedforward_norm_2: &post_ffn_2,
            post_feedforward_norm: &post_ffn,
            layer_scalar: Some(&layer_scalar),
            dense_inter_dim: DENSE_INTER,
        },
        EPS,
    )?;

    let tolerance = 2.0e-5;
    for row in 0..SEQ {
        let actual = gpu.row_slice(row)?;
        let expected = oracle.row_slice(row)?;
        for (column, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "ligne={row} colonne={column} GPU={actual} CPU={expected} tolérance={tolerance}"
            );
        }
    }
    Ok(())
}

/// L'attention globale Gemma 4 batchée reproduit le chemin CPU complet.
#[test]
fn gemma_global_attention_prefill_matches_cpu() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    const SEQ: usize = 264;
    const HIDDEN: usize = 64;
    const HEAD_DIM: usize = 512;
    const ROPE_DIMS: usize = HEAD_DIM / 4;
    const EPS: f32 = 1.0e-6;

    let input = deterministic_tensor(SEQ, HIDDEN, 307, 0.18)?;
    let input_norm = deterministic_norm(HIDDEN, 311)?;
    let post_norm = deterministic_norm(HIDDEN, 313)?;
    let q_norm = deterministic_norm(HEAD_DIM, 317)?;
    let k_norm = deterministic_norm(HEAD_DIM, 331)?;
    let q_proj = Linear::new(deterministic_tensor(HEAD_DIM, HIDDEN, 337, 0.025)?, None)?;
    let k_proj = Linear::new(deterministic_tensor(HEAD_DIM, HIDDEN, 347, 0.023)?, None)?;
    let o_proj = Linear::new(deterministic_tensor(HIDDEN, HEAD_DIM, 349, 0.018)?, None)?;
    let dense_gate = zero_linear(8, HIDDEN)?;
    let dense_up = zero_linear(8, HIDDEN)?;
    let dense_down = zero_linear(HIDDEN, 8)?;
    let spec = PrefillAttentionSpec {
        seq: SEQ,
        hidden_dim: HIDDEN,
        q_heads: 1,
        kv_heads: 1,
        head_dim: HEAD_DIM,
        rope_dims: ROPE_DIMS,
        rope_frequency_dim: HEAD_DIM,
        rope_theta: 1_000_000.0,
        attn_scalar: 256.0,
        window: None,
        k_eq_v: true,
        value_norm: true,
        eps: EPS,
    };
    let layer = PrefillMoeLayer {
        input_norm: &input_norm,
        attention: PrefillAttentionLayer::Full {
            q_proj: &q_proj,
            k_proj: &k_proj,
            v_proj: &k_proj,
            o_proj: &o_proj,
            q_norm: &q_norm,
            k_norm: &k_norm,
            gated: false,
        },
        attention_spec: spec,
        post_norm: &post_norm,
        post_norm_before_residual: true,
        tail: PrefillMoeTail::Dense {
            gate_proj: &dense_gate,
            up_proj: &dense_up,
            down_proj: &dense_down,
        },
    };

    let normed = crate::rms_norm(&input, &input_norm, EPS)?;
    let q_raw = q_proj.forward(&normed)?;
    let k_raw = k_proj.forward(&normed)?;
    let (_, expected_key, expected_value, context) =
        crate::decoder::gemma_global_attention_prefill_oracle(
            &q_raw,
            &k_raw,
            &k_raw,
            &q_norm,
            &k_norm,
            crate::decoder::GemmaGlobalAttentionOracleSpec {
                q_heads: spec.q_heads,
                kv_heads: spec.kv_heads,
                head_dim: spec.head_dim,
                rope_dims: spec.rope_dims,
                rope_theta: spec.rope_theta,
                attn_scalar: spec.attn_scalar,
                eps: spec.eps,
                value_norm: spec.value_norm,
                window: None,
            },
        )?;
    let attention_output = o_proj.forward(&context)?;
    let expected = input.add(&crate::rms_norm(&attention_output, &post_norm, EPS)?)?;

    let (actual, caches) = executor.qwen_moe_prefill_resident(&input, &[layer], spec)?;
    let (actual_key, actual_value) = full_cache(&caches)?;
    // Near-tie RoPE f32 : la réduction Metal peut décaler K de quelques ULP
    // (écart mesuré 1,035e-4), sans modifier V ni la sortie d'attention.
    let key_tolerance = 2.0e-4;
    let tolerance = 1.0e-4;
    assert_close(actual_key, &expected_key, key_tolerance, "gemma global key");
    assert_close(
        actual_value,
        &expected_value,
        tolerance,
        "gemma global value",
    );
    assert_close(&actual, &expected, tolerance, "gemma global output");
    Ok(())
}

/// L'attention locale Gemma 4 applique une fenêtre propre à chaque requête.
#[test]
fn gemma_windowed_attention_prefill_matches_cpu() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    const SEQ: usize = 544;
    const WINDOW: usize = 512;
    const HIDDEN: usize = 16;
    const HEAD_DIM: usize = 256;
    const EPS: f32 = 1.0e-6;

    let input = deterministic_tensor(SEQ, HIDDEN, 601, 0.2)?;
    let input_norm = deterministic_norm(HIDDEN, 607)?;
    let post_norm = deterministic_norm(HIDDEN, 613)?;
    let q_norm = deterministic_norm(HEAD_DIM, 617)?;
    let k_norm = deterministic_norm(HEAD_DIM, 619)?;
    let q_proj = Linear::new(deterministic_tensor(HEAD_DIM, HIDDEN, 631, 0.03)?, None)?;
    let k_proj = Linear::new(deterministic_tensor(HEAD_DIM, HIDDEN, 641, 0.028)?, None)?;
    let v_proj = Linear::new(deterministic_tensor(HEAD_DIM, HIDDEN, 643, 0.026)?, None)?;
    let o_proj = Linear::new(deterministic_tensor(HIDDEN, HEAD_DIM, 647, 0.02)?, None)?;
    let dense_gate = zero_linear(8, HIDDEN)?;
    let dense_up = zero_linear(8, HIDDEN)?;
    let dense_down = zero_linear(HIDDEN, 8)?;
    let spec = PrefillAttentionSpec {
        seq: SEQ,
        hidden_dim: HIDDEN,
        q_heads: 1,
        kv_heads: 1,
        head_dim: HEAD_DIM,
        rope_dims: HEAD_DIM,
        rope_frequency_dim: HEAD_DIM,
        rope_theta: 10_000.0,
        attn_scalar: 256.0,
        window: Some(WINDOW),
        k_eq_v: false,
        value_norm: true,
        eps: EPS,
    };
    let layer = PrefillMoeLayer {
        input_norm: &input_norm,
        attention: PrefillAttentionLayer::Full {
            q_proj: &q_proj,
            k_proj: &k_proj,
            v_proj: &v_proj,
            o_proj: &o_proj,
            q_norm: &q_norm,
            k_norm: &k_norm,
            gated: false,
        },
        attention_spec: spec,
        post_norm: &post_norm,
        post_norm_before_residual: true,
        tail: PrefillMoeTail::Dense {
            gate_proj: &dense_gate,
            up_proj: &dense_up,
            down_proj: &dense_down,
        },
    };

    let normed = crate::rms_norm(&input, &input_norm, EPS)?;
    let q_raw = q_proj.forward(&normed)?;
    let k_raw = k_proj.forward(&normed)?;
    let v_raw = v_proj.forward(&normed)?;
    let oracle_spec = crate::decoder::GemmaGlobalAttentionOracleSpec {
        q_heads: spec.q_heads,
        kv_heads: spec.kv_heads,
        head_dim: spec.head_dim,
        rope_dims: spec.rope_dims,
        rope_theta: spec.rope_theta,
        attn_scalar: spec.attn_scalar,
        eps: spec.eps,
        value_norm: spec.value_norm,
        window: spec.window,
    };
    let (_, expected_key, expected_value, context) =
        crate::decoder::gemma_global_attention_prefill_oracle(
            &q_raw,
            &k_raw,
            &v_raw,
            &q_norm,
            &k_norm,
            oracle_spec,
        )?;
    let (_, _, _, causal_context) = crate::decoder::gemma_global_attention_prefill_oracle(
        &q_raw,
        &k_raw,
        &v_raw,
        &q_norm,
        &k_norm,
        crate::decoder::GemmaGlobalAttentionOracleSpec {
            window: None,
            ..oracle_spec
        },
    )?;
    let pos = SEQ - 1;
    let row_start = pos + 1 - WINDOW;
    let masked_delta = context
        .row_slice(pos)?
        .iter()
        .zip(causal_context.row_slice(pos)?)
        .map(|(windowed, causal)| (windowed - causal).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        masked_delta > 1.0e-5,
        "ligne pos={pos}: la fenêtre [{row_start}, {pos}] doit exclure [0, {row_start})"
    );
    let attention_output = o_proj.forward(&context)?;
    let expected = input.add(&crate::rms_norm(&attention_output, &post_norm, EPS)?)?;
    let causal_attention_output = o_proj.forward(&causal_context)?;
    let causal_expected =
        input.add(&crate::rms_norm(&causal_attention_output, &post_norm, EPS)?)?;
    let observable_delta = expected
        .row_slice(pos)?
        .iter()
        .zip(causal_expected.row_slice(pos)?)
        .map(|(windowed, causal)| (windowed - causal).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        observable_delta > 1.0e-4,
        "ligne pos={pos}: le préfixe masqué doit rester observable face à l'oracle causal"
    );

    let (actual, caches) = executor.qwen_moe_prefill_resident(&input, &[layer], spec)?;
    let (actual_key, actual_value) = full_cache(&caches)?;
    let tolerance = 1.0e-4;
    // NOTE: near-tie sur la clé — le RoPE des couches sliding applique une base
    // locale (theta=1e4) dont la rotation accumule un peu plus d'erreur f32 que
    // les couches globales (theta=1e6). Mesuré : 3/139264 éléments à ≤1,21e-4,
    // valeur et sortie restant sous 1e-4. Tolérance clé élargie en conséquence.
    let key_tolerance = 2.0e-4;
    assert_close(
        actual_key,
        &expected_key,
        key_tolerance,
        "gemma windowed key",
    );
    assert_close(
        actual_value,
        &expected_value,
        tolerance,
        "gemma windowed value",
    );
    assert_close(&actual, &expected, tolerance, "gemma windowed output");
    Ok(())
}

/// Une fenêtre couvrant la séquence reproduit exactement le causal historique.
#[test]
fn gemma_windowed_attention_prefill_full_window_is_causal_bit_exact() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    const SEQ: usize = 264;
    const WINDOW: usize = 512;
    const HIDDEN: usize = 8;
    const HEAD_DIM: usize = 256;

    let input = deterministic_tensor(SEQ, HIDDEN, 653, 0.25)?;
    let input_norm = deterministic_norm(HIDDEN, 659)?;
    let post_norm = deterministic_norm(HIDDEN, 661)?;
    let q_norm = deterministic_norm(HEAD_DIM, 673)?;
    let k_norm = deterministic_norm(HEAD_DIM, 677)?;
    let q_proj = Linear::new(deterministic_tensor(HEAD_DIM, HIDDEN, 683, 0.035)?, None)?;
    let k_proj = Linear::new(deterministic_tensor(HEAD_DIM, HIDDEN, 691, 0.032)?, None)?;
    let v_proj = Linear::new(deterministic_tensor(HEAD_DIM, HIDDEN, 701, 0.03)?, None)?;
    let o_proj = Linear::new(deterministic_tensor(HIDDEN, HEAD_DIM, 709, 0.02)?, None)?;
    let dense_gate = zero_linear(4, HIDDEN)?;
    let dense_up = zero_linear(4, HIDDEN)?;
    let dense_down = zero_linear(HIDDEN, 4)?;
    let causal_spec = PrefillAttentionSpec {
        seq: SEQ,
        hidden_dim: HIDDEN,
        q_heads: 1,
        kv_heads: 1,
        head_dim: HEAD_DIM,
        rope_dims: HEAD_DIM,
        rope_frequency_dim: HEAD_DIM,
        rope_theta: 10_000.0,
        attn_scalar: 256.0,
        window: None,
        k_eq_v: false,
        value_norm: true,
        eps: 1.0e-6,
    };
    let run = |attention_spec: PrefillAttentionSpec| {
        executor.qwen_moe_prefill_resident(
            &input,
            &[PrefillMoeLayer {
                input_norm: &input_norm,
                attention: PrefillAttentionLayer::Full {
                    q_proj: &q_proj,
                    k_proj: &k_proj,
                    v_proj: &v_proj,
                    o_proj: &o_proj,
                    q_norm: &q_norm,
                    k_norm: &k_norm,
                    gated: false,
                },
                attention_spec,
                post_norm: &post_norm,
                post_norm_before_residual: true,
                tail: PrefillMoeTail::Dense {
                    gate_proj: &dense_gate,
                    up_proj: &dense_up,
                    down_proj: &dense_down,
                },
            }],
            attention_spec,
        )
    };

    let (causal, causal_caches) = run(causal_spec)?;
    let (windowed, windowed_caches) = run(PrefillAttentionSpec {
        window: Some(WINDOW),
        ..causal_spec
    })?;
    let (causal_key, causal_value) = full_cache(&causal_caches)?;
    let (windowed_key, windowed_value) = full_cache(&windowed_caches)?;
    assert_bit_exact(&windowed, &causal, "gemma full-window output");
    assert_bit_exact(windowed_key, causal_key, "gemma full-window key");
    assert_bit_exact(windowed_value, causal_value, "gemma full-window value");
    Ok(())
}

/// L'alias Gemma K=V ignore toute projection V distincte avant le cache.
#[test]
fn gemma_global_attention_aliases_raw_k_as_v() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    const SEQ: usize = 4;
    const HIDDEN: usize = 8;
    const HEAD_DIM: usize = 8;

    let input = deterministic_tensor(SEQ, HIDDEN, 359, 0.4)?;
    let input_norm = deterministic_norm(HIDDEN, 367)?;
    let post_norm = deterministic_norm(HIDDEN, 373)?;
    let q_norm = deterministic_norm(HEAD_DIM, 379)?;
    let k_norm = deterministic_norm(HEAD_DIM, 383)?;
    let q_proj = Linear::new(deterministic_tensor(HEAD_DIM, HIDDEN, 389, 0.08)?, None)?;
    let k_proj = Linear::new(deterministic_tensor(HEAD_DIM, HIDDEN, 397, 0.07)?, None)?;
    let poison_v = Linear::new(deterministic_tensor(HEAD_DIM, HIDDEN, 401, 0.7)?, None)?;
    let o_proj = Linear::new(deterministic_tensor(HIDDEN, HEAD_DIM, 409, 0.06)?, None)?;
    let dense_gate = zero_linear(4, HIDDEN)?;
    let dense_up = zero_linear(4, HIDDEN)?;
    let dense_down = zero_linear(HIDDEN, 4)?;
    let base = PrefillAttentionSpec {
        seq: SEQ,
        hidden_dim: HIDDEN,
        q_heads: 1,
        kv_heads: 1,
        head_dim: HEAD_DIM,
        rope_dims: HEAD_DIM,
        rope_frequency_dim: HEAD_DIM,
        rope_theta: 1_000_000.0,
        attn_scalar: HEAD_DIM as f32,
        window: None,
        k_eq_v: true,
        value_norm: false,
        eps: 1.0e-6,
    };
    let run = |spec: PrefillAttentionSpec, v_proj: &Linear| {
        executor.qwen_moe_prefill_resident(
            &input,
            &[PrefillMoeLayer {
                input_norm: &input_norm,
                attention: PrefillAttentionLayer::Full {
                    q_proj: &q_proj,
                    k_proj: &k_proj,
                    v_proj,
                    o_proj: &o_proj,
                    q_norm: &q_norm,
                    k_norm: &k_norm,
                    gated: false,
                },
                attention_spec: spec,
                post_norm: &post_norm,
                post_norm_before_residual: false,
                tail: PrefillMoeTail::Dense {
                    gate_proj: &dense_gate,
                    up_proj: &dense_up,
                    down_proj: &dense_down,
                },
            }],
            base,
        )
    };

    let (_, alias_caches) = run(base, &poison_v)?;
    let (_, projected_k_caches) = run(
        PrefillAttentionSpec {
            k_eq_v: false,
            ..base
        },
        &k_proj,
    )?;
    let (_, alias_value) = full_cache(&alias_caches)?;
    let (_, projected_k_value) = full_cache(&projected_k_caches)?;
    assert_bit_exact(alias_value, projected_k_value, "gemma raw K alias V");
    Ok(())
}

/// Le scratch et le readback suivent les dimensions effectives de chaque couche.
#[test]
fn prefill_resident_uses_layer_specific_attention_dimensions() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    const SEQ: usize = 3;
    const HIDDEN: usize = 8;

    let input = deterministic_tensor(SEQ, HIDDEN, 479, 0.3)?;
    let input_norm = deterministic_norm(HIDDEN, 487)?;
    let post_norm = deterministic_norm(HIDDEN, 491)?;
    let dense_gate = zero_linear(4, HIDDEN)?;
    let dense_up = zero_linear(4, HIDDEN)?;
    let dense_down = zero_linear(HIDDEN, 4)?;
    let q_norm_1 = deterministic_norm(4, 499)?;
    let k_norm_1 = deterministic_norm(4, 503)?;
    let q_proj_1 = Linear::new(deterministic_tensor(8, HIDDEN, 509, 0.07)?, None)?;
    let k_proj_1 = Linear::new(deterministic_tensor(4, HIDDEN, 521, 0.06)?, None)?;
    let v_proj_1 = Linear::new(deterministic_tensor(4, HIDDEN, 523, 0.05)?, None)?;
    let o_proj_1 = Linear::new(deterministic_tensor(HIDDEN, 8, 541, 0.04)?, None)?;
    let q_norm_2 = deterministic_norm(2, 547)?;
    let k_norm_2 = deterministic_norm(2, 557)?;
    let q_proj_2 = Linear::new(deterministic_tensor(4, HIDDEN, 563, 0.07)?, None)?;
    let k_proj_2 = Linear::new(deterministic_tensor(2, HIDDEN, 569, 0.06)?, None)?;
    let v_proj_2 = Linear::new(deterministic_tensor(2, HIDDEN, 571, 0.05)?, None)?;
    let o_proj_2 = Linear::new(deterministic_tensor(HIDDEN, 4, 577, 0.04)?, None)?;
    let base = PrefillAttentionSpec {
        seq: SEQ,
        hidden_dim: HIDDEN,
        q_heads: 2,
        kv_heads: 1,
        head_dim: 4,
        rope_dims: 4,
        rope_frequency_dim: 4,
        rope_theta: 1_000_000.0,
        attn_scalar: 4.0,
        window: None,
        k_eq_v: false,
        value_norm: false,
        eps: 1.0e-6,
    };
    let second = PrefillAttentionSpec {
        head_dim: 2,
        rope_dims: 2,
        rope_frequency_dim: 2,
        attn_scalar: 2.0,
        ..base
    };
    let layers = [
        PrefillMoeLayer {
            input_norm: &input_norm,
            attention: PrefillAttentionLayer::Full {
                q_proj: &q_proj_1,
                k_proj: &k_proj_1,
                v_proj: &v_proj_1,
                o_proj: &o_proj_1,
                q_norm: &q_norm_1,
                k_norm: &k_norm_1,
                gated: false,
            },
            attention_spec: base,
            post_norm: &post_norm,
            post_norm_before_residual: false,
            tail: PrefillMoeTail::Dense {
                gate_proj: &dense_gate,
                up_proj: &dense_up,
                down_proj: &dense_down,
            },
        },
        PrefillMoeLayer {
            input_norm: &input_norm,
            attention: PrefillAttentionLayer::Full {
                q_proj: &q_proj_2,
                k_proj: &k_proj_2,
                v_proj: &v_proj_2,
                o_proj: &o_proj_2,
                q_norm: &q_norm_2,
                k_norm: &k_norm_2,
                gated: false,
            },
            attention_spec: second,
            post_norm: &post_norm,
            post_norm_before_residual: false,
            tail: PrefillMoeTail::Dense {
                gate_proj: &dense_gate,
                up_proj: &dense_up,
                down_proj: &dense_down,
            },
        },
    ];

    let (first_output, first_cache) =
        executor.qwen_moe_prefill_resident(&input, &layers[..1], base)?;
    let (sequential_output, second_cache) =
        executor.qwen_moe_prefill_resident(&first_output, &layers[1..], second)?;
    let (resident_output, caches) = executor.qwen_moe_prefill_resident(&input, &layers, base)?;
    assert_bit_exact(
        &resident_output,
        &sequential_output,
        "prefill multi-layout final output",
    );
    match caches.as_slice() {
        [PrefillResidentLayerCache::Full {
            key: first,
            value: first_value,
        }, PrefillResidentLayerCache::Full {
            key: second,
            value: second_value,
        }] => {
            assert_eq!(first.shape(), &[SEQ, 4]);
            assert_eq!(second.shape(), &[SEQ, 2]);
            let (expected_first, expected_first_value) = full_cache(&first_cache)?;
            let (expected_second, expected_second_value) = full_cache(&second_cache)?;
            assert_bit_exact(first, expected_first, "prefill multi-layout K couche 0");
            assert_bit_exact(
                first_value,
                expected_first_value,
                "prefill multi-layout V couche 0",
            );
            assert_bit_exact(second, expected_second, "prefill multi-layout K couche 1");
            assert_bit_exact(
                second_value,
                expected_second_value,
                "prefill multi-layout V couche 1",
            );
        }
        _ => {
            return Err(InferError::Dimension(
                "caches par couche absents".to_string(),
            ))
        }
    }
    Ok(())
}

/// Le spine par couche conserve le prefill Qwen historique bit pour bit.
#[test]
fn qwen3_moe_prefill_resident_matches_legacy_bit_exact() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    const SEQ: usize = 8;
    const HIDDEN: usize = 16;
    const HEAD_DIM: usize = 8;
    const Q_HEADS: usize = 2;
    const KV_HEADS: usize = 1;
    const EXPERTS: usize = 2;
    const TOP_K: usize = 1;

    let input = deterministic_tensor(SEQ, HIDDEN, 419, 0.5)?;
    let input_norm = deterministic_norm(HIDDEN, 421)?;
    let post_norm = deterministic_norm(HIDDEN, 431)?;
    let q_norm = deterministic_norm(HEAD_DIM, 433)?;
    let k_norm = deterministic_norm(HEAD_DIM, 439)?;
    let q_proj = Linear::new(deterministic_tensor(HIDDEN, HIDDEN, 443, 0.08)?, None)?;
    let k_proj = Linear::new(deterministic_tensor(HEAD_DIM, HIDDEN, 449, 0.07)?, None)?;
    let v_proj = Linear::new(deterministic_tensor(HEAD_DIM, HIDDEN, 457, 0.06)?, None)?;
    let o_proj = Linear::new(deterministic_tensor(HIDDEN, HIDDEN, 461, 0.05)?, None)?;
    let router = Linear::new(deterministic_tensor(EXPERTS, HIDDEN, 463, 0.1)?, None)?;
    let experts = deterministic_qwen_experts(EXPERTS, HIDDEN, 8, 467)?;
    let spec = PrefillAttentionSpec {
        seq: SEQ,
        hidden_dim: HIDDEN,
        q_heads: Q_HEADS,
        kv_heads: KV_HEADS,
        head_dim: HEAD_DIM,
        rope_dims: HEAD_DIM,
        rope_frequency_dim: HEAD_DIM,
        rope_theta: 1_000_000.0,
        attn_scalar: HEAD_DIM as f32,
        window: None,
        k_eq_v: false,
        value_norm: false,
        eps: 1.0e-6,
    };

    let (legacy, legacy_key, legacy_value) = executor.full_attention_prefill_tail_moe(
        &input,
        &input_norm,
        &q_proj,
        &k_proj,
        &v_proj,
        &o_proj,
        &q_norm,
        &k_norm,
        &post_norm,
        &router,
        &experts,
        TOP_K,
        spec,
    )?;
    let (resident, caches) = executor.qwen_moe_prefill_resident(
        &input,
        &[PrefillMoeLayer {
            input_norm: &input_norm,
            attention: PrefillAttentionLayer::Full {
                q_proj: &q_proj,
                k_proj: &k_proj,
                v_proj: &v_proj,
                o_proj: &o_proj,
                q_norm: &q_norm,
                k_norm: &k_norm,
                gated: false,
            },
            attention_spec: spec,
            post_norm: &post_norm,
            post_norm_before_residual: false,
            tail: PrefillMoeTail::Routed {
                router: &router,
                experts: &experts,
                top_k: TOP_K,
            },
        }],
        spec,
    )?;
    let (resident_key, resident_value) = full_cache(&caches)?;
    assert_bit_exact(&resident, &legacy, "qwen prefill output");
    assert_bit_exact(resident_key, &legacy_key, "qwen prefill key");
    assert_bit_exact(resident_value, &legacy_value, "qwen prefill value");
    Ok(())
}
