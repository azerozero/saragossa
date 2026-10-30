//! Prefill flint : attention avec sinks appris (pleine et fenêtrée, Steel
//! tuilé et kernel F3, en un bloc et par morceaux) et routage sigmoïde en
//! lignes contre l'oracle CPU, et équivalence bit à bit du kernel F3 à sink
//! neutre avec le kernel d128 sans sink de Qwen.

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
/// La requête locale `pos` est à la position absolue `q_off + pos` ; seules
/// les lignes `rows` sont calculées, les autres restent nulles.
fn cpu_sink_attention(
    q: &[f32],
    (keys, values): (&[f32], &[f32]),
    spec: PrefillAttentionSpec,
    sinks: &[f32],
    q_off: usize,
    rows: &[usize],
) -> Vec<f32> {
    let (q_dim, kv_dim) = (spec.q_heads * HEAD_DIM, spec.kv_heads * HEAD_DIM);
    let group = spec.q_heads / spec.kv_heads;
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let mut out = vec![0.0_f32; spec.seq * q_dim];
    for &pos in rows {
        let abs = q_off + pos;
        let start = spec
            .window
            .map_or(0, |window| (abs + 1).saturating_sub(window));
        for head in 0..spec.q_heads {
            let kv_head = head / group;
            let query = &q[pos * q_dim + head * HEAD_DIM..][..HEAD_DIM];
            let mut scores = (start..=abs)
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

/// Attention prefill GPU : kernel à sinks si `sinks` (Steel ou F3 selon
/// `executor.prefill_sink_steel`), sinon le d128 sans sink. Les `spec.seq`
/// requêtes sont les dernières des lignes de `keys`/`values`.
fn gpu_attention(
    executor: &MetalExecutor,
    (q, keys, values): (&[f32], &[f32], &[f32]),
    spec: PrefillAttentionSpec,
    sinks: Option<&[f32]>,
) -> Result<Vec<f32>> {
    let kv_rows = keys.len() / (spec.kv_heads * HEAD_DIM);
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
            kv_rows,
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

/// Lignes comparées à l'oracle CPU : toutes jusqu'à 1100, puis un
/// échantillon régulier et la fin (l'oracle CPU est en O(n²) en debug).
fn oracle_rows(seq: usize) -> Vec<usize> {
    if seq <= 1100 {
        return (0..seq).collect();
    }
    (0..seq).step_by(37).chain(seq - 3..seq).collect::<Vec<_>>()
}

fn row_gap(got: &[f32], want: &[f32], rows: &[usize], q_dim: usize) -> f32 {
    rows.iter()
        .flat_map(|&row| {
            got[row * q_dim..][..q_dim]
                .iter()
                .zip(&want[row * q_dim..][..q_dim])
        })
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max)
}

/// Le Steel d128 avec sinks doit compiler : sinon les tests ci-dessous ne
/// mesureraient que le kernel F3.
fn steel_executor() -> Result<Option<MetalExecutor>> {
    let executor = test_executor()?;
    if let Some(executor) = executor.as_ref() {
        assert!(
            executor.attention_prefill_steel_sink_d128_f32.is_some(),
            "pipeline Steel d128 avec sinks absent"
        );
    }
    Ok(executor)
}

/// Longueurs autour de la fenêtre flint (513, token courant compris), d'un
/// bord de tuile Steel (BQ32, BK16) et au-delà des seuils 256/2048 des kernels
/// sans sink, en plein et fenêtré, par le Steel et par le F3 : chacun contre
/// le CPU, et le Steel contre le F3 sur toutes les lignes.
#[test]
fn prefill_sink_attention_matches_cpu_oracle() -> Result<()> {
    let Some(mut executor) = steel_executor()? else {
        return Ok(());
    };
    let sinks = sinks(Q_HEADS);
    let q_dim = Q_HEADS * HEAD_DIM;
    for seq in [1, 300, 513, 514, 1100, 4000] {
        let (q, keys, values) = inputs(seq, Q_HEADS, KV_HEADS);
        let rows = oracle_rows(seq);
        for window in [None, Some(WINDOW)] {
            let spec = spec(seq, Q_HEADS, KV_HEADS, window);
            let want = cpu_sink_attention(&q, (&keys, &values), spec, &sinks, 0, &rows);
            let mut outputs = Vec::new();
            for steel in [false, true] {
                executor.prefill_sink_steel = steel;
                let got = gpu_attention(&executor, (&q, &keys, &values), spec, Some(&sinks))?;
                let gap = row_gap(&got, &want, &rows, q_dim);
                // Sorties d'ordre 1e-2 : seuls l'ordre d'accumulation (softmax
                // en ligne, simd_sum ou MMA) et `fast::exp`/`exp2` séparent le
                // GPU du CPU.
                assert!(
                    gap <= 1.0e-5,
                    "seq={seq} window={window:?} steel={steel}: écart CPU {gap}"
                );
                println!("seq={seq} window={window:?} steel={steel}: écart CPU {gap:.2e}");
                outputs.push(got);
            }
            let gap = row_gap(
                &outputs[0],
                &outputs[1],
                &(0..seq).collect::<Vec<_>>(),
                q_dim,
            );
            assert!(
                gap <= 1.0e-5,
                "seq={seq} window={window:?}: Steel vs F3 {gap}"
            );
            println!("seq={seq} window={window:?}: Steel vs F3 {gap:.2e}");
        }
    }
    Ok(())
}

/// Requêtes par morceaux (positions absolues `kv_rows - seq + pos`, K/V du
/// préfixe déjà écrit) contre le même calcul en un bloc, coupes alignées ou
/// non sur les tuiles Steel et sur la fenêtre : mêmes octets, puisque chaque
/// ligne refait les mêmes opérations (les tuiles K entièrement masquées
/// n'ajoutent que des zéros exacts au Steel).
#[test]
fn prefill_sink_attention_chunks_match_single_block() -> Result<()> {
    let Some(mut executor) = steel_executor()? else {
        return Ok(());
    };
    let sinks = sinks(Q_HEADS);
    let (q_dim, kv_dim) = (Q_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM);
    let seq = 1100;
    let (q, keys, values) = inputs(seq, Q_HEADS, KV_HEADS);
    for steel in [false, true] {
        executor.prefill_sink_steel = steel;
        for window in [None, Some(WINDOW)] {
            let single = gpu_attention(
                &executor,
                (&q, &keys, &values),
                spec(seq, Q_HEADS, KV_HEADS, window),
                Some(&sinks),
            )?;
            for chunk in [1, 256, 300, 513] {
                let mut chunked = Vec::with_capacity(single.len());
                for start in (0..seq).step_by(chunk) {
                    let end = (start + chunk).min(seq);
                    chunked.extend(gpu_attention(
                        &executor,
                        (
                            &q[start * q_dim..end * q_dim],
                            &keys[..end * kv_dim],
                            &values[..end * kv_dim],
                        ),
                        spec(end - start, Q_HEADS, KV_HEADS, window),
                        Some(&sinks),
                    )?);
                }
                assert!(
                    single
                        .iter()
                        .zip(&chunked)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "steel={steel} window={window:?} morceaux de {chunk} != un bloc"
                );
            }
        }
    }
    Ok(())
}

/// Un sink neutre et aucune fenêtre rendent, par le kernel F3, les octets
/// exacts du kernel d128 sans sink de Qwen
/// (`causal_attention_prefill_batch_long_d128_f32`, choisi pour 32/4/128 entre
/// 257 et 2048 positions), resté textuellement inchangé.
#[test]
fn prefill_sink_attention_neutral_is_bit_identical_to_d128() -> Result<()> {
    let Some(mut executor) = test_executor()? else {
        return Ok(());
    };
    if !crate::runtime_flags::prefill_attn_batch_mid_30b_enabled() {
        return Ok(());
    }
    executor.prefill_sink_steel = false;
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
