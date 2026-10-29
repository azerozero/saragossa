//! Cas analytiques de l'oracle CPU flint : sinks, fenêtre 513 et NoPE.

use super::attention_ops::{
    cached_attention_one, causal_attention, full_attention_context_prefill,
    full_attention_from_tensors, AttentionLayout,
};
use super::*;

fn layout(heads: usize, window: Option<usize>) -> AttentionLayout {
    AttentionLayout {
        num_attention_heads: heads,
        num_key_value_heads: 1,
        head_dim: 2,
        rope_dims: 2,
        attn_scalar: 1.0,
        sliding_window: window,
    }
}

fn rows(n: usize, row: impl Fn(usize) -> [f32; 2]) -> Tensor {
    Tensor::from_vec(vec![n, 2], (0..n).flat_map(row).collect()).expect("invariant: lignes")
}

fn close(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected) {
        assert!((a - e).abs() <= tol, "{actual:?} vs {expected:?}");
    }
}

#[test]
fn sink_enters_max_and_denominator_once_per_head() {
    let layout = layout(2, None);
    let keys = [[1.0_f32, 0.0], [0.0, 1.0]];
    let values = keys;
    // Tête 0 : scores [2, 0], sink 3 (le sink relève le max).
    // Tête 1 : scores [0, 0], sink -1.
    let q = Tensor::from_vec(vec![1, 4], vec![2.0, 0.0, 0.0, 0.0]).expect("invariant: q");
    let sinks = [3.0_f32, -1.0];
    let d0 = 1.0 + (-1.0_f32).exp() + (-3.0_f32).exp();
    let p0 = [(-1.0_f32).exp() / d0, (-3.0_f32).exp() / d0];
    let d1 = (-1.0_f32).exp() + 2.0;
    let expected = [p0[0], p0[1], 1.0 / d1, 1.0 / d1];

    let mut cache = LayerKvCache::default();
    for (k, v) in keys.iter().zip(values.iter()) {
        cache
            .append(&rows(1, |_| *k), &rows(1, |_| *v), &layout)
            .expect("invariant: append");
    }
    let cached = cached_attention_one(&q, &mut cache, &layout, Some(&sinks))
        .expect("invariant: attention cache");
    close(cached.data(), &expected, 1.0e-6);

    // Même formule sur le chemin prefill (dernière ligne causale).
    let q_rows = Tensor::from_vec(vec![2, 4], vec![0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
        .expect("invariant: q prefill");
    let prefill = causal_attention(
        &q_rows,
        &rows(2, |i| keys[i]),
        &rows(2, |i| values[i]),
        &layout,
        Some(&sinks),
    )
    .expect("invariant: attention prefill");
    close(
        prefill.row_slice(1).expect("invariant: ligne"),
        &expected,
        1.0e-6,
    );
    // Un sink absorbe de la masse : la somme des poids reste < 1.
    assert!(expected[0] + expected[1] < 1.0);
}

#[test]
fn sliding_window_513_includes_current_token() {
    // q nul : poids uniformes, la sortie est la moyenne des valeurs visibles.
    let layout = layout(1, Some(513));
    let mean =
        |start: usize, end: usize| (start..=end).sum::<usize>() as f32 / (end - start + 1) as f32;
    let seq = 514;
    let q = rows(seq, |_| [0.0, 0.0]);
    let k = rows(seq, |i| [i as f32 * 1e-3, 1.0]);
    let v = rows(seq, |i| [i as f32, 1.0]);
    let out = causal_attention(&q, &k, &v, &layout, None).expect("invariant: prefill");
    let probe = |pos: usize| out.row_slice(pos).expect("invariant: ligne")[0];
    // 512 tokens visibles (0..=511), 513 (0..=512), puis 513 (1..=513).
    assert!((probe(511) - mean(0, 511)).abs() < 1e-3);
    assert!((probe(512) - mean(0, 512)).abs() < 1e-3);
    assert!((probe(513) - mean(1, 513)).abs() < 1e-3);

    // Chemin decode avec cache : 512, 513 puis 514 lignes en cache.
    let mut cache = LayerKvCache::default();
    let zero_q = rows(1, |_| [0.0, 0.0]);
    for (len, expected) in [
        (512, mean(0, 511)),
        (513, mean(0, 512)),
        (514, mean(1, 513)),
    ] {
        while cache.len() < len {
            let i = cache.len();
            cache
                .append(
                    &rows(1, |_| [0.0, 1.0]),
                    &rows(1, |_| [i as f32, 1.0]),
                    &layout,
                )
                .expect("invariant: append");
        }
        let out = cached_attention_one(&zero_q, &mut cache, &layout, None)
            .expect("invariant: decode fenêtré");
        assert!((out.data()[0] - expected).abs() < 1e-3, "len={len}");
    }
}

fn flint_config() -> CausalDecoderConfig {
    let config: ModelConfig =
        serde_json::from_value::<crate::config::RawModelConfig>(serde_json::json!({
            "model_type": "flint",
            "hidden_size": 4,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 2,
            "layer_types": ["full_attention", "sliding_attention"],
            "no_rope_layer_indices": [0],
            "sliding_window": 513,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "vocab_size": 4,
        }))
        .expect("invariant: JSON flint")
        .resolve()
        .expect("invariant: config flint");
    CausalDecoderConfig::from(&config)
}

#[test]
fn flint_layer_types_drive_window_and_nope() {
    let config = flint_config();
    assert_eq!(config.layer_sliding_window(0), None);
    assert_eq!(config.layer_sliding_window(1), Some(513));
    assert!(config.is_no_rope_layer(0) && !config.is_no_rope_layer(1));
    // Gemma 3 garde son motif historique même avec des `layer_types`.
    let gemma3 = CausalDecoderConfig {
        sliding_from_layer_types: false,
        ..config.clone()
    };
    assert_eq!(gemma3.layer_sliding_window(1), None);
}

#[test]
fn nope_layer_normalizes_qk_without_rotation() {
    let config = flint_config();
    let dense = |rows: usize, cols: usize, seed: f32| {
        DecoderTensor::Dense(
            Tensor::from_vec(
                vec![rows, cols],
                (0..rows * cols)
                    .map(|i| (i as f32 * 0.37 + seed).sin())
                    .collect(),
            )
            .expect("invariant: poids"),
        )
    };
    let attention_at = |layer: usize| {
        let mut tensors = HashMap::new();
        let p = format!("layers.{layer}.self_attn");
        tensors.insert(format!("{p}.q_proj.weight"), dense(4, 4, 0.1));
        tensors.insert(format!("{p}.k_proj.weight"), dense(2, 4, 0.2));
        tensors.insert(format!("{p}.v_proj.weight"), dense(2, 4, 0.3));
        tensors.insert(format!("{p}.o_proj.weight"), dense(4, 4, 0.4));
        tensors.insert(format!("{p}.q_norm.weight"), dense(1, 2, 0.5));
        tensors.insert(format!("{p}.k_norm.weight"), dense(1, 2, 0.6));
        tensors.insert(
            format!("{p}.softmax_offset"),
            DecoderTensor::Dense(
                Tensor::from_vec(vec![2], vec![0.0, 0.0]).expect("invariant: sinks"),
            ),
        );
        let AttentionBlock::Full(mut attention) =
            full_attention_from_tensors(&config, &mut tensors, &format!("layers.{layer}"), layer)
                .expect("invariant: attention flint")
        else {
            panic!("invariant: attention full attendue");
        };
        attention.no_rope = config.is_no_rope_layer(layer);
        attention
    };
    let normed =
        Tensor::from_vec(vec![1, 4], vec![0.3, -0.7, 0.2, 0.9]).expect("invariant: normed");
    let keys_at = |layer: usize, offset: usize| {
        let attention = attention_at(layer);
        let mut cache = LayerKvCache::default();
        full_attention_context_prefill(
            &config,
            &normed,
            &mut cache,
            offset,
            &attention,
            ForwardRuntime::cpu(),
        )
        .expect("invariant: prefill");
        cache.keys.to_vec()
    };
    // NoPE : K identique quelle que soit la position, mais normalisé.
    let nope = keys_at(0, 0);
    assert_eq!(nope, keys_at(0, 37));
    let rms = (nope.iter().map(|x| x * x).sum::<f32>() / 2.0).sqrt();
    assert!(rms > 0.0 && rms.is_finite());
    // Couche RoPE : la position change K.
    assert_ne!(keys_at(1, 0), keys_at(1, 37));
}
