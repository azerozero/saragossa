//! Prefill flint : attention avec sinks appris (pleine et fenêtrée) et routage
//! sigmoïde en lignes contre l'oracle CPU, et équivalence bit à bit du kernel
//! à sink neutre avec le kernel d128 sans sink de Qwen.

use super::*;
use crate::decoder::attention_probs_in_place;

/// GQA 8 et d = 128 comme flint, avec moins de têtes Q pour garder l'oracle
/// CPU rapide en debug : le kernel ne dépend du nombre de têtes que par la grille.
const Q_HEADS: usize = 8;
const KV_HEADS: usize = 1;
const HEAD_DIM: usize = 128;
const WINDOW: usize = 513;
/// Sink assez négatif pour un `exp` nul et un max inchangé, sans `-inf`
/// (les kernels sont compilés en fast-math).
const NEUTRAL_SINK: f32 = -1.0e30;

fn test_executor() -> Result<Option<MetalExecutor>> {
    match MetalExecutor::new() {
        Ok(executor) => Ok(Some(executor)),
        Err(InferError::Metal(message)) if message.contains("aucun device") => Ok(None),
        Err(error) => Err(error),
    }
}

fn pattern(len: usize, modulo: usize, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|i| ((i * 7919 % modulo) as f32 - modulo as f32 / 2.0) * scale)
        .collect()
}

/// Sinks variés ; la tête 3 domine les scores pour exercer le recalage quand
/// le sink porte le maximum.
fn sinks(heads: usize) -> Vec<f32> {
    (0..heads)
        .map(|head| {
            if head == 3 {
                8.0
            } else {
                head as f32 * 0.5 - 2.0
            }
        })
        .collect()
}

fn spec(
    seq: usize,
    q_heads: usize,
    kv_heads: usize,
    window: Option<usize>,
) -> PrefillAttentionSpec {
    PrefillAttentionSpec {
        seq,
        hidden_dim: q_heads * HEAD_DIM,
        q_heads,
        kv_heads,
        head_dim: HEAD_DIM,
        rope_dims: 0,
        rope_frequency_dim: 0,
        rope_theta: 10_000.0,
        attn_scalar: HEAD_DIM as f32,
        window,
        k_eq_v: false,
        value_norm: false,
        eps: 1.0e-6,
    }
}

/// Oracle CPU : scores `q·k/√d` sur la fenêtre causale, probabilités par
/// `attention_probs_in_place` (le noyau de l'oracle CPU flint), somme pondérée.
fn cpu_sink_attention(
    q: &[f32],
    keys: &[f32],
    values: &[f32],
    spec: PrefillAttentionSpec,
    sinks: &[f32],
) -> Vec<f32> {
    let (q_dim, kv_dim) = (spec.q_heads * HEAD_DIM, spec.kv_heads * HEAD_DIM);
    let group = spec.q_heads / spec.kv_heads;
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let mut out = vec![0.0_f32; spec.seq * q_dim];
    for pos in 0..spec.seq {
        let start = spec
            .window
            .map_or(0, |window| (pos + 1).saturating_sub(window));
        for head in 0..spec.q_heads {
            let kv_head = head / group;
            let query = &q[pos * q_dim + head * HEAD_DIM..][..HEAD_DIM];
            let mut scores = (start..=pos)
                .map(|row| {
                    let key = &keys[row * kv_dim + kv_head * HEAD_DIM..][..HEAD_DIM];
                    query.iter().zip(key).map(|(a, b)| a * b).sum::<f32>() * scale
                })
                .collect::<Vec<_>>();
            attention_probs_in_place(&mut scores, Some(sinks[head]));
            let dst = &mut out[pos * q_dim + head * HEAD_DIM..][..HEAD_DIM];
            for (offset, prob) in scores.iter().enumerate() {
                let value = &values[(start + offset) * kv_dim + kv_head * HEAD_DIM..][..HEAD_DIM];
                for (acc, v) in dst.iter_mut().zip(value) {
                    *acc += prob * v;
                }
            }
        }
    }
    out
}

/// Attention prefill GPU : kernel à sinks si `sinks`, sinon le d128 sans sink.
fn gpu_attention(
    executor: &MetalExecutor,
    (q, keys, values): (&[f32], &[f32], &[f32]),
    spec: PrefillAttentionSpec,
    sinks: Option<&[f32]>,
) -> Result<Vec<f32>> {
    let q_buffer = executor.upload_f32_buffer(q, "sink_test_q")?;
    let k_buffer = executor.upload_f32_buffer(keys, "sink_test_k")?;
    let v_buffer = executor.upload_f32_buffer(values, "sink_test_v")?;
    let out_buffer = executor.uncached_f32_buffer(q.len(), "sink_test_out")?;
    let sinks_buffer = sinks
        .map(|sinks| executor.upload_f32_buffer(sinks, "sink_test_sinks"))
        .transpose()?;
    let command_buffer = executor.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    let encoded = match &sinks_buffer {
        Some(sinks_buffer) => executor.encode_attention_prefill_sink(
            encoder,
            &q_buffer,
            &k_buffer,
            &v_buffer,
            &out_buffer,
            sinks_buffer,
            spec,
        ),
        None => executor.encode_causal_attention_prefill(
            encoder,
            &q_buffer,
            &k_buffer,
            &v_buffer,
            &out_buffer,
            spec,
        ),
    };
    encoder.end_encoding();
    encoded?;
    commit_and_wait(command_buffer)?;
    read_f32_buffer(&out_buffer, q.len())
}

fn inputs(seq: usize, q_heads: usize, kv_heads: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    (
        pattern(seq * q_heads * HEAD_DIM, 97, 0.02),
        pattern(seq * kv_heads * HEAD_DIM, 89, 0.02),
        pattern(seq * kv_heads * HEAD_DIM, 83, 0.01),
    )
}

/// Longueurs autour de la fenêtre flint (513, token courant compris) et
/// au-delà des seuils 256/2048 des kernels sans sink, en plein et fenêtré.
#[test]
fn prefill_sink_attention_matches_cpu_oracle() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let sinks = sinks(Q_HEADS);
    for seq in [1, 300, 513, 514, 1100] {
        let (q, keys, values) = inputs(seq, Q_HEADS, KV_HEADS);
        for window in [None, Some(WINDOW)] {
            let spec = spec(seq, Q_HEADS, KV_HEADS, window);
            let got = gpu_attention(&executor, (&q, &keys, &values), spec, Some(&sinks))?;
            let want = cpu_sink_attention(&q, &keys, &values, spec, &sinks);
            let gap = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            // Sorties d'ordre 1e-2 : seul l'ordre d'accumulation (softmax en
            // ligne, simd_sum) et `fast::exp` séparent le GPU du CPU.
            assert!(gap <= 1.0e-5, "seq={seq} window={window:?}: écart {gap}");
        }
    }
    Ok(())
}

/// Un sink neutre et aucune fenêtre rendent les octets exacts du kernel d128
/// sans sink de Qwen (`causal_attention_prefill_batch_long_d128_f32`, choisi
/// pour 32/4/128 entre 257 et 2048 positions), resté textuellement inchangé.
#[test]
fn prefill_sink_attention_neutral_is_bit_identical_to_d128() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    if !crate::runtime_flags::prefill_attn_batch_mid_30b_enabled() {
        return Ok(());
    }
    let (q_heads, kv_heads, seq) = (32, 4, 300);
    let (q, keys, values) = inputs(seq, q_heads, kv_heads);
    let spec = spec(seq, q_heads, kv_heads, None);
    let neutral = vec![NEUTRAL_SINK; q_heads];
    let with_sink = gpu_attention(&executor, (&q, &keys, &values), spec, Some(&neutral))?;
    let without = gpu_attention(&executor, (&q, &keys, &values), spec, None)?;
    assert!(
        with_sink
            .iter()
            .zip(&without)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "sink neutre != d128 sans sink"
    );
    Ok(())
}

/// Routage sigmoïde en lignes (prefill) contre `sigmoid_top_weights` : 256
/// experts top-6 comme flint, dont des scores saturés à 1,0 départagés par indice.
#[test]
fn topk_sigmoid_rows_matches_cpu_routing() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let (rows, experts, top_k) = (5, 256, 6);
    let mut logits = (0..rows * experts)
        .map(|i| ((i * 37) % 251) as f32 * 0.05 - 6.0)
        .collect::<Vec<_>>();
    for index in [200, 17, 90, 4] {
        logits[2 * experts + index] = 30.0 + index as f32;
    }
    let logits_buffer = executor.upload_f32_buffer(&logits, "topk_sigmoid_rows_logits")?;
    let indices_buffer = executor.uncached_u32_buffer(rows * top_k, "topk_sigmoid_rows_idx")?;
    let scores_buffer = executor.uncached_f32_buffer(rows * top_k, "topk_sigmoid_rows_w")?;
    let command_buffer = executor.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    let encoded = executor.encode_topk_sigmoid_rows(
        encoder,
        &logits_buffer,
        &indices_buffer,
        &scores_buffer,
        rows,
        experts,
        top_k,
    );
    encoder.end_encoding();
    encoded?;
    commit_and_wait(command_buffer)?;
    let indices = read_u32_buffer(&indices_buffer, rows * top_k)?;
    let scores = read_f32_buffer(&scores_buffer, rows * top_k)?;
    for row in 0..rows {
        let (want_indices, want_scores) =
            crate::mlp::sigmoid_top_weights(&logits[row * experts..][..experts], top_k);
        let got_indices = indices[row * top_k..][..top_k]
            .iter()
            .map(|&index| index as usize)
            .collect::<Vec<_>>();
        assert_eq!(got_indices, want_indices, "ligne {row}: sélection");
        for (got, want) in scores[row * top_k..][..top_k].iter().zip(&want_scores) {
            assert!((got - want).abs() <= 1.0e-6, "ligne {row}: {got} != {want}");
        }
    }
    Ok(())
}
