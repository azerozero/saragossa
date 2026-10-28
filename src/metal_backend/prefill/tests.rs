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

/// K ou V d'un cache de prefill en tenseur `[rows, kv_dim]`, vue partagée ou
/// copie relue.
fn kv_tensor(rows: &PrefillKvRows, kv_dim: usize) -> Result<Tensor> {
    let data = rows.data().to_vec();
    Tensor::from_vec(vec![data.len() / kv_dim.max(1), kv_dim], data)
}

/// Caches de prefill avec K/V en tenseurs, pour les comparaisons d'oracle.
enum TestCache<'a> {
    Full {
        key: Tensor,
        value: Tensor,
    },
    Linear {
        state: &'a LinearAttentionMetalState,
    },
}

fn test_caches(caches: &[PrefillResidentLayerCache]) -> Result<Vec<TestCache<'_>>> {
    caches
        .iter()
        .map(|cache| match cache {
            PrefillResidentLayerCache::Full { key, value, kv_dim } => Ok(TestCache::Full {
                key: kv_tensor(key, *kv_dim)?,
                value: kv_tensor(value, *kv_dim)?,
            }),
            PrefillResidentLayerCache::Linear { state } => Ok(TestCache::Linear { state }),
        })
        .collect()
}

fn full_cache(caches: &[PrefillResidentLayerCache]) -> Result<(Tensor, Tensor)> {
    match caches {
        [PrefillResidentLayerCache::Full { key, value, kv_dim }] => {
            Ok((kv_tensor(key, *kv_dim)?, kv_tensor(value, *kv_dim)?))
        }
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
    let (ref actual_key, ref actual_value) = full_cache(&caches)?;
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
    let (ref actual_key, ref actual_value) = full_cache(&caches)?;
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
    let (ref causal_key, ref causal_value) = full_cache(&causal_caches)?;
    let (ref windowed_key, ref windowed_value) = full_cache(&windowed_caches)?;
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
    let (_, ref alias_value) = full_cache(&alias_caches)?;
    let (_, ref projected_k_value) = full_cache(&projected_k_caches)?;
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
    // Captures : la sortie de la couche 0 est l'entrée de la couche 1 (= sortie
    // du run à une couche), celle de la couche 1 est la sortie finale.
    let (_, _, captures) = executor.qwen_moe_prefill_resident_with_captures(
        &input,
        &layers,
        base,
        &[0, 1],
        0,
        false,
    )?;
    let captures = captures.ok_or_else(|| InferError::Config("captures absentes".to_string()))?;
    assert_eq!(captures.shape(), &[SEQ, 2 * HIDDEN]);
    for row in 0..SEQ {
        let line = captures.row_slice(row)?;
        assert_eq!(
            &line[..HIDDEN],
            first_output.row_slice(row)?,
            "capture couche 0 ligne {row}"
        );
        assert_eq!(
            &line[HIDDEN..],
            resident_output.row_slice(row)?,
            "capture couche 1 ligne {row}"
        );
    }
    match test_caches(&caches)?.as_slice() {
        [TestCache::Full {
            key: first,
            value: first_value,
        }, TestCache::Full {
            key: second,
            value: second_value,
        }] => {
            assert_eq!(first.shape(), &[SEQ, 4]);
            assert_eq!(second.shape(), &[SEQ, 2]);
            let (ref expected_first, ref expected_first_value) = full_cache(&first_cache)?;
            let (ref expected_second, ref expected_second_value) = full_cache(&second_cache)?;
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
    let (ref resident_key, ref resident_value) = full_cache(&caches)?;
    assert_bit_exact(&resident, &legacy, "qwen prefill output");
    assert_bit_exact(resident_key, &legacy_key, "qwen prefill key");
    assert_bit_exact(resident_value, &legacy_value, "qwen prefill value");
    Ok(())
}

/// K/V full-attention identiques à l'octet entre deux prefills.
fn assert_full_kv_bit_exact(
    actual: &[PrefillResidentLayerCache],
    expected: &[PrefillResidentLayerCache],
    label: &str,
) -> Result<()> {
    for (a, b) in test_caches(actual)?.iter().zip(&test_caches(expected)?) {
        if let (
            TestCache::Full { key, value },
            TestCache::Full {
                key: key_ref,
                value: value_ref,
            },
        ) = (a, b)
        {
            assert_bit_exact(key, key_ref, label);
            assert_bit_exact(value, value_ref, label);
        }
    }
    Ok(())
}

/// Le prefill par morceaux (KV recopié, état linéaire reporté, RoPE et
/// attention décalées) reproduit le prefill en un seul morceau. Le miroir bf16
/// du K/V (`SARAGOSSA_RUST_PREFILL_KV_BF16_MIRROR`) est bit-identique à la
/// reconversion complète : morceaux de 367 lignes (non alignés), dernière
/// ligne seule (K/V des morceaux `kv_only` rattrapé) et prolongements.
#[test]
fn prefill_resident_chunked_matches_single_chunk() -> Result<()> {
    let Some(mut executor) = test_executor()? else {
        return Ok(());
    };
    executor.prefill_kv_bf16_mirror = false;
    // Le miroir n'est alloué qu'avec le NAX : on l'impose dès le départ,
    // quel que soit `SARAGOSSA_RUST_PREFILL_ATTN_NAX`, pour que les
    // assertions miroir comparent bien le chemin incrémental au chemin
    // courant, tous deux en NAX. Sans accélérateur, elles sont sautées.
    if executor.causal_attention_prefill_steel_nax_d256.is_none() {
        executor.causal_attention_prefill_steel_nax_d256 =
            super::super::pipelines::compile_steel_causal_nax_d256_attention(
                &executor.device,
                STEEL_ATTENTION_NAX_KERNELS,
                super::super::attention::NAX_CAUSAL_ATTN_BQ64,
            );
    }
    let nax = executor.causal_attention_prefill_steel_nax_d256.is_some();
    if !nax {
        eprintln!("assertions miroir bf16 sautées : NAX indisponible sur ce GPU");
    }
    const SEQ: usize = 1100;
    const HIDDEN: usize = 64;
    const Q_HEADS: usize = 16;
    const KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 256;
    const INTER: usize = 32;
    // Chemin de production 27B : dk128/dv128, noyau conv 4.
    let linear_spec = LinearAttentionStepSpec {
        num_key_heads: 1,
        num_value_heads: 2,
        key_head_dim: 128,
        value_head_dim: 128,
        conv_kernel_dim: 4,
        rms_eps: 1.0e-6,
    };
    let dims = LinearAttnResidentDims {
        in_dim: HIDDEN,
        conv_dim: 512,
        value_dim: 256,
        key_dim: 128,
    };

    let input = deterministic_tensor(SEQ, HIDDEN, 601, 0.5)?;
    let input_norm = deterministic_norm(HIDDEN, 607)?;
    let post_norm = deterministic_norm(HIDDEN, 613)?;
    let gate = Linear::new(deterministic_tensor(INTER, HIDDEN, 617, 0.1)?, None)?;
    let up = Linear::new(deterministic_tensor(INTER, HIDDEN, 619, 0.1)?, None)?;
    let down = Linear::new(deterministic_tensor(HIDDEN, INTER, 631, 0.1)?, None)?;
    let in_proj_qkv = Linear::new(deterministic_tensor(512, HIDDEN, 641, 0.08)?, None)?;
    let in_proj_z = Linear::new(deterministic_tensor(256, HIDDEN, 643, 0.08)?, None)?;
    let in_proj_b = Linear::new(deterministic_tensor(2, HIDDEN, 647, 0.08)?, None)?;
    let in_proj_a = Linear::new(deterministic_tensor(2, HIDDEN, 653, 0.08)?, None)?;
    let out_proj = Linear::new(deterministic_tensor(HIDDEN, 256, 659, 0.05)?, None)?;
    let conv_weight = Tensor::from_vec(
        vec![512, 4, 1],
        deterministic_tensor(512, 4, 661, 0.3)?.data().to_vec(),
    )?;
    let a_log = [0.1_f32, -0.2];
    let dt_bias = [0.05_f32, 0.15];
    let norm_weight = deterministic_norm(128, 673)?;
    let q_norm = deterministic_norm(HEAD_DIM, 677)?;
    let k_norm = deterministic_norm(HEAD_DIM, 683)?;
    let q_proj = Linear::new(
        deterministic_tensor(2 * Q_HEADS * HEAD_DIM, HIDDEN, 691, 0.06)?,
        None,
    )?;
    let k_proj = Linear::new(
        deterministic_tensor(KV_HEADS * HEAD_DIM, HIDDEN, 701, 0.06)?,
        None,
    )?;
    let v_proj = Linear::new(
        deterministic_tensor(KV_HEADS * HEAD_DIM, HIDDEN, 709, 0.06)?,
        None,
    )?;
    let o_proj = Linear::new(
        deterministic_tensor(HIDDEN, Q_HEADS * HEAD_DIM, 719, 0.02)?,
        None,
    )?;
    let spec = PrefillAttentionSpec {
        seq: SEQ,
        hidden_dim: HIDDEN,
        q_heads: Q_HEADS,
        kv_heads: KV_HEADS,
        head_dim: HEAD_DIM,
        rope_dims: 64,
        rope_frequency_dim: 64,
        rope_theta: 10_000_000.0,
        attn_scalar: HEAD_DIM as f32,
        window: None,
        k_eq_v: false,
        value_norm: false,
        eps: 1.0e-6,
    };
    let tail = PrefillMoeTail::Dense {
        gate_proj: &gate,
        up_proj: &up,
        down_proj: &down,
    };
    let make_layers = |spec: PrefillAttentionSpec| {
        [
            PrefillMoeLayer {
                input_norm: &input_norm,
                attention: PrefillAttentionLayer::Linear {
                    weights: LinearAttnResidentWeights {
                        in_proj_qkv: &in_proj_qkv,
                        in_proj_z: &in_proj_z,
                        in_proj_b: &in_proj_b,
                        in_proj_a: &in_proj_a,
                        out_proj: &out_proj,
                        conv_weight: &conv_weight,
                        a_log: &a_log,
                        dt_bias: &dt_bias,
                        norm_weight: norm_weight.data(),
                    },
                    spec: linear_spec,
                    dims,
                },
                attention_spec: spec,
                post_norm: &post_norm,
                post_norm_before_residual: false,
                tail,
            },
            PrefillMoeLayer {
                input_norm: &input_norm,
                attention: PrefillAttentionLayer::Full {
                    q_proj: &q_proj,
                    k_proj: &k_proj,
                    v_proj: &v_proj,
                    o_proj: &o_proj,
                    q_norm: &q_norm,
                    k_norm: &k_norm,
                    gated: true,
                },
                attention_spec: spec,
                post_norm: &post_norm,
                post_norm_before_residual: false,
                tail,
            },
        ]
    };
    let layers = make_layers(spec);

    let (single, single_caches, single_captures) =
        executor.qwen_moe_prefill_resident_with_captures(&input, &layers, spec, &[0], 0, false)?;
    // 512 → trois morceaux de 367/367/366 lignes.
    let (chunked, chunked_caches, chunked_captures) = executor
        .qwen_moe_prefill_resident_with_captures(&input, &layers, spec, &[0], 512, false)?;
    // Dernière ligne seule : la couche finale (attention complète) ne calcule
    // que son K/V sur les morceaux non finaux ; ligne et KV restent identiques.
    let (last, last_caches, _) =
        executor.qwen_moe_prefill_resident_with_captures(&input, &layers, spec, &[0], 512, true)?;
    let (single, chunked, last) = (
        single.into_tensor(spec.hidden_dim)?,
        chunked.into_tensor(spec.hidden_dim)?,
        last.into_tensor(spec.hidden_dim)?,
    );
    if nax {
        executor.prefill_kv_bf16_mirror = true;
        let (mirrored, mirrored_caches, mirrored_captures) = executor
            .qwen_moe_prefill_resident_with_captures(&input, &layers, spec, &[0], 512, false)?;
        let (mirrored_last, mirrored_last_caches, _) = executor
            .qwen_moe_prefill_resident_with_captures(&input, &layers, spec, &[0], 512, true)?;
        executor.prefill_kv_bf16_mirror = false;
        assert_bit_exact(
            &mirrored.into_tensor(spec.hidden_dim)?,
            &chunked,
            "miroir bf16 sortie",
        );
        assert_bit_exact(
            &mirrored_last.into_tensor(spec.hidden_dim)?,
            &last,
            "miroir bf16 dernière ligne",
        );
        if let (Some(a), Some(b)) = (mirrored_captures.as_ref(), chunked_captures.as_ref()) {
            assert_bit_exact(a, b, "miroir bf16 capture");
        }
        assert_full_kv_bit_exact(&mirrored_caches, &chunked_caches, "miroir bf16 K/V")?;
        assert_full_kv_bit_exact(
            &mirrored_last_caches,
            &last_caches,
            "miroir bf16 K/V dernière ligne",
        )?;
    }
    let hidden = last.data().len();
    assert_eq!(last.shape(), &[1, hidden]);
    assert_eq!(
        last.data(),
        &chunked.data()[chunked.data().len() - hidden..]
    );
    for (a, b) in test_caches(&last_caches)?
        .iter()
        .zip(&test_caches(&chunked_caches)?)
    {
        if let (
            TestCache::Full { key, value },
            TestCache::Full {
                key: key_ref,
                value: value_ref,
            },
        ) = (a, b)
        {
            assert_eq!(key.data(), key_ref.data());
            assert_eq!(value.data(), value_ref.data());
        }
    }
    let max_diff = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max)
    };
    let output_diff = max_diff(chunked.data(), single.data());
    eprintln!("prefill par morceaux: écart sortie {output_diff:e}");
    assert_close(&chunked, &single, 1.0e-4, "prefill par morceaux sortie");
    let (Some(single_captures), Some(chunked_captures)) = (single_captures, chunked_captures)
    else {
        return Err(InferError::Config("captures absentes".to_string()));
    };
    assert_close(
        &chunked_captures,
        &single_captures,
        1.0e-4,
        "prefill par morceaux capture",
    );
    match (
        test_caches(&single_caches)?.as_slice(),
        test_caches(&chunked_caches)?.as_slice(),
    ) {
        (
            [TestCache::Linear {
                state: single_state,
            }, TestCache::Full {
                key: single_key,
                value: single_value,
            }],
            [TestCache::Linear {
                state: chunked_state,
            }, TestCache::Full {
                key: chunked_key,
                value: chunked_value,
            }],
        ) => {
            assert_close(chunked_key, single_key, 1.0e-4, "prefill par morceaux K");
            assert_close(
                chunked_value,
                single_value,
                1.0e-4,
                "prefill par morceaux V",
            );
            let conv_single = read_f32_buffer(&single_state.conv, single_state.conv_len)?;
            let conv_chunked = read_f32_buffer(&chunked_state.conv, chunked_state.conv_len)?;
            assert!(max_diff(&conv_chunked, &conv_single) <= 1.0e-4, "état conv");
            if !single_state.ssm_bf16 {
                let ssm_single = read_f32_buffer(&single_state.ssm, single_state.ssm_len)?;
                let ssm_chunked = read_f32_buffer(&chunked_state.ssm, chunked_state.ssm_len)?;
                let ssm_diff = max_diff(&ssm_chunked, &ssm_single);
                eprintln!("prefill par morceaux: écart SSM {ssm_diff:e}");
                assert!(ssm_diff <= 1.0e-4, "état SSM écart {ssm_diff}");
            }
        }
        _ => {
            return Err(InferError::Dimension(
                "caches linéaire + full attendus".to_string(),
            ))
        }
    }

    // Prolongement d'un préfixe calculé à froid (KV recopié, état linéaire avancé
    // en place) : suffixe de 400 lignes, puis de 5 lignes, plus court qu'un bloc Steel.
    for prefix_rows in [700, SEQ - 5] {
        let rows_of = |tensor: &Tensor, from: usize, to: usize, dim: usize| {
            Tensor::from_vec(
                vec![to - from, dim],
                tensor.data()[from * dim..to * dim].to_vec(),
            )
        };
        let prefix_spec = PrefillAttentionSpec {
            seq: prefix_rows,
            ..spec
        };
        // L'état linéaire du préfixe avance en place : un préfixe frais par run.
        let mut runs = Vec::with_capacity(2);
        for mirror in [false, true] {
            executor.prefill_kv_bf16_mirror = mirror;
            let (_, prefix_caches) = executor.qwen_moe_prefill_resident(
                &rows_of(&input, 0, prefix_rows, HIDDEN)?,
                &make_layers(prefix_spec),
                prefix_spec,
            )?;
            let prefix = PrefillResidentPrefix {
                rows: prefix_rows,
                layers: prefix_caches
                    .iter()
                    .map(|cache| match cache {
                        PrefillResidentLayerCache::Full { key, value, .. } => {
                            PrefillResidentPrefixLayer::Full {
                                key: vec![key.data()],
                                value: vec![value.data()],
                            }
                        }
                        PrefillResidentLayerCache::Linear { state } => {
                            PrefillResidentPrefixLayer::Linear { state }
                        }
                    })
                    .collect(),
            };
            let (extended, extended_caches, _) = executor.qwen_moe_prefill_resident_from(
                &rows_of(&input, prefix_rows, SEQ, HIDDEN)?,
                &layers,
                spec,
                &[],
                512,
                Some(&prefix),
                false,
            )?;
            runs.push((extended.into_tensor(HIDDEN)?, extended_caches));
        }
        executor.prefill_kv_bf16_mirror = false;
        let (mirrored, mirrored_caches) = runs.pop().expect("invariant: deux runs");
        let (extended, extended_caches) = runs.pop().expect("invariant: deux runs");
        assert_bit_exact(&mirrored, &extended, "miroir bf16 prolongement");
        assert_full_kv_bit_exact(
            &mirrored_caches,
            &extended_caches,
            "miroir bf16 K/V prolongement",
        )?;
        assert_close(
            &extended,
            &rows_of(&single, prefix_rows, SEQ, HIDDEN)?,
            1.0e-4,
            "prolongement sortie",
        );
        match (
            test_caches(&single_caches)?.as_slice(),
            test_caches(&extended_caches)?.as_slice(),
        ) {
            (
                [TestCache::Linear {
                    state: single_state,
                }, TestCache::Full {
                    key: single_key,
                    value: single_value,
                }],
                [TestCache::Linear {
                    state: extended_state,
                }, TestCache::Full {
                    key: extended_key,
                    value: extended_value,
                }],
            ) => {
                let kv_dim = KV_HEADS * HEAD_DIM;
                assert_close(
                    extended_key,
                    &rows_of(single_key, prefix_rows, SEQ, kv_dim)?,
                    1.0e-4,
                    "prolongement K",
                );
                assert_close(
                    extended_value,
                    &rows_of(single_value, prefix_rows, SEQ, kv_dim)?,
                    1.0e-4,
                    "prolongement V",
                );
                let conv_single = read_f32_buffer(&single_state.conv, single_state.conv_len)?;
                let conv_extended = read_f32_buffer(&extended_state.conv, extended_state.conv_len)?;
                assert!(
                    max_diff(&conv_extended, &conv_single) <= 1.0e-4,
                    "prolongement conv"
                );
                // L'état SSM après la dernière ligne est celui que reprend le decode.
                let ssm_single = read_f32_buffer(&single_state.ssm, single_state.ssm_len)?;
                let ssm_extended = read_f32_buffer(&extended_state.ssm, extended_state.ssm_len)?;
                assert!(
                    max_diff(&ssm_extended, &ssm_single) <= 1.0e-4,
                    "prolongement ssm"
                );
            }
            _ => {
                return Err(InferError::Dimension(
                    "caches prolongés linéaire + full attendus".to_string(),
                ))
            }
        }
    }
    Ok(())
}
