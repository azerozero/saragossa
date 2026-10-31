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

/// Arrondi bf16 (RNE) d'un K/V f32, élargi en f32 : ce que lit un kernel sur
/// le même K/V stocké en bf16.
fn bf16_rounded(data: &[f32]) -> Vec<f32> {
    crate::metal_backend::buffers::f32_slice_to_bf16(data)
        .into_iter()
        .map(|bits| f32::from_bits(u32::from(bits) << 16))
        .collect()
}

/// K ou V de test, en f32 ou arrondi et stocké en bf16.
fn upload_kv(
    executor: &MetalExecutor,
    data: &[f32],
    bf16: bool,
    label: &'static str,
) -> Result<metal::Buffer> {
    if !bf16 {
        return executor.upload_f32_buffer(data, label);
    }
    let buffer = executor.uncached_bf16_buffer(data.len(), label)?;
    write_u16_buffer_at(
        &buffer,
        0,
        &crate::metal_backend::buffers::f32_slice_to_bf16(data),
    )?;
    Ok(buffer)
}

/// Attention prefill GPU : kernel à sinks si `sinks` (Steel ou F3 selon
/// `executor.prefill_sink_steel`, K/V stockés en bf16 si `kv_bf16`), sinon le
/// d128 sans sink. Les `spec.seq` requêtes sont les dernières des lignes de
/// `keys`/`values`.
fn gpu_attention(
    executor: &MetalExecutor,
    (q, keys, values): (&[f32], &[f32], &[f32]),
    spec: PrefillAttentionSpec,
    sinks: Option<&[f32]>,
    kv_bf16: bool,
) -> Result<Vec<f32>> {
    let kv_rows = keys.len() / (spec.kv_heads * HEAD_DIM);
    let q_buffer = executor.upload_f32_buffer(q, "sink_test_q")?;
    let k_buffer = upload_kv(executor, keys, kv_bf16, "sink_test_k")?;
    let v_buffer = upload_kv(executor, values, kv_bf16, "sink_test_v")?;
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
            kv_bf16,
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

/// Les Steel d128 avec sinks (K/V f32 et bf16) doivent compiler : sinon les
/// tests ci-dessous ne mesureraient que le kernel F3.
fn steel_executor() -> Result<Option<MetalExecutor>> {
    let mut executor = test_executor()?;
    if let Some(executor) = executor.as_mut() {
        // Le NAX (opt-in) passerait devant le Steel bf16 : ces tests
        // mesurent le Steel et le F3 quel que soit l'environnement.
        executor.attention_prefill_nax_sink_d128 = None;
        executor.attention_prefill_nax_sink_d128_window = None;
        assert!(
            executor.attention_prefill_steel_sink_d128_f32.is_some(),
            "pipeline Steel d128 avec sinks absent"
        );
        assert!(
            executor.attention_prefill_steel_sink_d128_kv_bf16.is_some(),
            "pipeline Steel d128 avec sinks et K/V bf16 absent"
        );
    }
    Ok(executor)
}

/// Longueurs autour de la fenêtre flint (513, token courant compris), d'un
/// bord de tuile Steel (BQ32, BK16) et au-delà des seuils 256/2048 des kernels
/// sans sink, en plein et fenêtré, par le Steel et par le F3 : chacun contre
/// le CPU, et le Steel contre le F3 sur toutes les lignes. Deux régimes : les
/// scores doux de `inputs`, puis des scores ×10 avec des sinks ±8, où le
/// maximum passe tour à tour du sink aux scores et force le recalage.
/// K/V bf16 (sauf à 4000, pour la durée de l'oracle) : contre le CPU sur le
/// K/V arrondi, et bit à bit contre la variante f32 nourrie du même K/V
/// arrondi, puisque l'élargissement `float(bfloat)` est exact.
#[test]
fn prefill_sink_attention_matches_cpu_oracle() -> Result<()> {
    let Some(mut executor) = steel_executor()? else {
        return Ok(());
    };
    let hard_sinks = (0..Q_HEADS)
        .map(|head| if head % 2 == 0 { 8.0 } else { -8.0 })
        .collect::<Vec<_>>();
    let q_dim = Q_HEADS * HEAD_DIM;
    for (q_scale, sinks) in [(1.0_f32, sinks(Q_HEADS)), (10.0, hard_sinks)] {
        for seq in [1, 300, 513, 514, 1100, 4000] {
            let (q, keys, values) = inputs(seq, Q_HEADS, KV_HEADS);
            let q = q.iter().map(|value| value * q_scale).collect::<Vec<_>>();
            let rows = oracle_rows(seq);
            let (keys_bf16, values_bf16) = (bf16_rounded(&keys), bf16_rounded(&values));
            for (window, kv_bf16) in [None, Some(WINDOW)]
                .into_iter()
                .flat_map(|window| [(window, false), (window, true)])
                .filter(|&(_, kv_bf16)| !kv_bf16 || seq <= 1100)
            {
                let label = format!("q×{q_scale} seq={seq} window={window:?} kv_bf16={kv_bf16}");
                let spec = spec(seq, Q_HEADS, KV_HEADS, window);
                let (keys, values) = if kv_bf16 {
                    (&keys_bf16, &values_bf16)
                } else {
                    (&keys, &values)
                };
                let want = cpu_sink_attention(&q, (keys, values), spec, &sinks, 0, &rows);
                let mut outputs = Vec::new();
                for steel in [false, true] {
                    executor.prefill_sink_steel = steel;
                    let got =
                        gpu_attention(&executor, (&q, keys, values), spec, Some(&sinks), kv_bf16)?;
                    let gap = row_gap(&got, &want, &rows, q_dim);
                    // Sorties d'ordre 1e-2 : seuls l'ordre d'accumulation
                    // (softmax en ligne, simd_sum ou MMA) et `fast::exp`/`exp2`
                    // séparent le GPU du CPU.
                    assert!(gap <= 1.0e-5, "{label} steel={steel}: écart CPU {gap}");
                    println!("{label} steel={steel}: écart CPU {gap:.2e}");
                    if kv_bf16 {
                        let widened = gpu_attention(
                            &executor,
                            (&q, keys, values),
                            spec,
                            Some(&sinks),
                            false,
                        )?;
                        assert!(
                            got.iter()
                                .zip(&widened)
                                .all(|(a, b)| a.to_bits() == b.to_bits()),
                            "{label} steel={steel}: K/V bf16 != f32 arrondi"
                        );
                    }
                    outputs.push(got);
                }
                let all_rows = (0..seq).collect::<Vec<_>>();
                let gap = row_gap(&outputs[0], &outputs[1], &all_rows, q_dim);
                assert!(gap <= 1.0e-5, "{label}: Steel vs F3 {gap}");
                println!("{label}: Steel vs F3 {gap:.2e}");
            }
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
    for (steel, kv_bf16) in [(false, false), (true, false), (false, true), (true, true)] {
        executor.prefill_sink_steel = steel;
        for window in [None, Some(WINDOW)] {
            let single = gpu_attention(
                &executor,
                (&q, &keys, &values),
                spec(seq, Q_HEADS, KV_HEADS, window),
                Some(&sinks),
                kv_bf16,
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
                        kv_bf16,
                    )?);
                }
                assert!(
                    single
                        .iter()
                        .zip(&chunked)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "steel={steel} kv_bf16={kv_bf16} window={window:?} morceaux de {chunk} != un bloc"
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
    let with_sink = gpu_attention(&executor, (&q, &keys, &values), spec, Some(&neutral), false)?;
    let without = gpu_attention(&executor, (&q, &keys, &values), spec, None, false)?;
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

/// Oracle f64 de l'attention à sinks pleine (sans fenêtre), toutes lignes.
fn f64_sink_attention(
    q: &[f32],
    (keys, values): (&[f32], &[f32]),
    seq: usize,
    sinks: &[f32],
) -> Vec<f64> {
    let (q_dim, kv_dim) = (Q_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM);
    let group = Q_HEADS / KV_HEADS;
    let scale = 1.0 / (HEAD_DIM as f64).sqrt();
    let mut out = vec![0.0_f64; seq * q_dim];
    for pos in 0..seq {
        for head in 0..Q_HEADS {
            let kv_head = head / group;
            let query = &q[pos * q_dim + head * HEAD_DIM..][..HEAD_DIM];
            let scores = (0..=pos)
                .map(|row| {
                    let key = &keys[row * kv_dim + kv_head * HEAD_DIM..][..HEAD_DIM];
                    query
                        .iter()
                        .zip(key)
                        .map(|(a, b)| f64::from(*a) * f64::from(*b))
                        .sum::<f64>()
                        * scale
                })
                .collect::<Vec<_>>();
            let sink = f64::from(sinks[head]);
            let max = scores.iter().copied().fold(sink, f64::max);
            let weights = scores
                .iter()
                .map(|score| (score - max).exp())
                .collect::<Vec<_>>();
            let denom = (sink - max).exp() + weights.iter().sum::<f64>();
            let dst = &mut out[pos * q_dim + head * HEAD_DIM..][..HEAD_DIM];
            for (row, weight) in weights.iter().enumerate() {
                let value = &values[row * kv_dim + kv_head * HEAD_DIM..][..HEAD_DIM];
                for (acc, v) in dst.iter_mut().zip(value) {
                    *acc += weight / denom * f64::from(*v);
                }
            }
        }
    }
    out
}

/// Scores réels de grande amplitude (jusqu'à ~700 en nats) : Steel et F3
/// contre un oracle f64, en écart relatif par ligne. Mesuré le 2026-10-30 :
/// F3 ≤ 3,9e-6, Steel ≤ 1,5e-5 (l'échelle appliquée après le produit, en
/// base 2, perd un peu plus sur les grands scores), très en deçà de l'écart
/// GPU/CPU de 5,5e-3 relevé en sortie de la couche 0 de flint-int4.
#[test]
fn prefill_sink_attention_large_scores_match_f64() -> Result<()> {
    let Some(mut executor) = steel_executor()? else {
        return Ok(());
    };
    let seq = 600;
    let (q_dim, kv_dim) = (Q_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM);
    let sinks = sinks(Q_HEADS);
    for amp in [1.0_f32, 6.0, 20.0] {
        let q = pattern(seq * q_dim, 97, 0.04 * amp);
        let keys = pattern(seq * kv_dim, 89, 0.04 * amp);
        let values = pattern(seq * kv_dim, 83, 0.02);
        let want = f64_sink_attention(&q, (&keys, &values), seq, &sinks);
        for steel in [false, true] {
            executor.prefill_sink_steel = steel;
            let got = gpu_attention(
                &executor,
                (&q, &keys, &values),
                spec(seq, Q_HEADS, KV_HEADS, None),
                Some(&sinks),
                false,
            )?;
            let worst = (0..seq)
                .map(|pos| {
                    let got = &got[pos * q_dim..][..q_dim];
                    let want = &want[pos * q_dim..][..q_dim];
                    let num = got
                        .iter()
                        .zip(want)
                        .map(|(a, b)| (f64::from(*a) - b).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    num / want.iter().map(|b| b * b).sum::<f64>().sqrt().max(1e-30)
                })
                .fold(0.0_f64, f64::max);
            println!("amp={amp} steel={steel}: écart relatif f64 {worst:.2e}");
            assert!(worst <= 5.0e-5, "amp={amp} steel={steel}: {worst}");
        }
    }
    Ok(())
}

/// Pipeline NAX d128 avec sinks, fenêtré si `window` ; `None` sans
/// accélérateur (MPP, macOS 26).
fn nax_sink_pipeline(
    executor: &MetalExecutor,
    window: bool,
) -> Option<super::super::attention::NaxCausalAttention> {
    let nax = super::super::pipelines::compile_nax_sink_d128_attention(
        &executor.device,
        STEEL_ATTENTION_NAX_KERNELS,
        super::super::attention::NAX_SINK_ATTN_D128,
        window,
    );
    // Le GEMM NAX compile dès que MPP est là (macOS 26 et accélérateur) :
    // l'attention NAX doit alors compiler aussi, sinon le test échoue au
    // lieu de passer en silence.
    assert!(
        nax.is_some() || executor.na_gemm_bf16.is_none(),
        "MPP disponible mais le NAX d128 avec sinks (fenêtre={window}) ne compile pas"
    );
    if nax.is_none() {
        eprintln!("NAX d128 avec sinks non supporté (MPP absent) : test sauté");
    }
    nax
}

/// Écart absolu toléré du NAX d128 avec sinks (sorties d'ordre 1e-2) :
/// l'arrondi bf16 de P avant P·V domine. Mesuré le 2026-10-30 : pire 1,06e-4
/// contre le CPU comme contre le Steel bf16.
const NAX_SINK_TOL: f32 = 5.0e-4;

/// NAX d128 avec sinks à entrées bf16 identiques — Q arrondi comme le NAX le
/// convertit, K/V stockés en bf16 — contre l'oracle CPU et contre le Steel
/// d128 bf16 : un bloc de 1, 300, 513, 1100 et 4133 requêtes, puis des
/// morceaux finaux (1, 300, 1100 requêtes) derrière un préfixe de 4133 clés.
/// GQA 8 (une tête KV) en scores doux puis ×10 avec des sinks ±8 (le maximum
/// passe du sink aux scores), et GQA 2 sur quatre têtes KV en scores ×10 pour
/// les transitions de groupe (`kv_head_idx`). Seules différences attendues :
/// P arrondi en bf16 avant P·V et l'ordre des MMA.
#[test]
fn prefill_sink_nax_matches_steel_and_cpu() -> Result<()> {
    const KV_TOTAL: usize = 4133;
    let Some(mut executor) = steel_executor()? else {
        return Ok(());
    };
    let Some(nax) = nax_sink_pipeline(&executor, false) else {
        return Ok(());
    };
    let hard_sinks = (0..Q_HEADS)
        .map(|head| if head % 2 == 0 { 8.0 } else { -8.0 })
        .collect::<Vec<_>>();
    let q_dim = Q_HEADS * HEAD_DIM;
    let mut nax = Some(nax);
    for (kv_heads, q_scale, sinks) in [
        (KV_HEADS, 1.0_f32, sinks(Q_HEADS)),
        (KV_HEADS, 10.0, hard_sinks.clone()),
        (4, 10.0, hard_sinks),
    ] {
        let (q, keys, values) = inputs(KV_TOTAL, Q_HEADS, kv_heads);
        let q = bf16_rounded(&q.iter().map(|value| value * q_scale).collect::<Vec<_>>());
        let (keys, values) = (bf16_rounded(&keys), bf16_rounded(&values));
        // (requêtes, clés) : `kv_rows > seq` place les requêtes en fin de préfixe.
        for (seq, kv_rows) in [
            (1, 1),
            (300, 300),
            (513, 513),
            (1100, 1100),
            (KV_TOTAL, KV_TOTAL),
            (1, KV_TOTAL),
            (300, KV_TOTAL),
            (1100, KV_TOTAL),
        ] {
            let label = format!("kv_heads={kv_heads} q×{q_scale} q={seq} kv={kv_rows}");
            let kv_len = kv_rows * kv_heads * HEAD_DIM;
            let q_off = kv_rows - seq;
            let q = &q[q_off * q_dim..kv_rows * q_dim];
            let (keys, values) = (&keys[..kv_len], &values[..kv_len]);
            let spec = spec(seq, Q_HEADS, kv_heads, None);
            let rows = oracle_rows(seq);
            let want = cpu_sink_attention(q, (keys, values), spec, &sinks, q_off, &rows);
            let steel = gpu_attention(&executor, (q, keys, values), spec, Some(&sinks), true)?;
            executor.attention_prefill_nax_sink_d128 = nax.take();
            let got = gpu_attention(&executor, (q, keys, values), spec, Some(&sinks), true);
            nax = executor.attention_prefill_nax_sink_d128.take();
            let got = got?;
            let gap_cpu = row_gap(&got, &want, &rows, q_dim);
            let gap_steel = row_gap(&got, &steel, &(0..seq).collect::<Vec<_>>(), q_dim);
            println!("{label}: écart CPU {gap_cpu:.2e}, NAX vs Steel {gap_steel:.2e}");
            assert!(
                got.iter().all(|value| value.is_finite()),
                "{label}: NaN/inf"
            );
            assert!(gap_cpu <= NAX_SINK_TOL, "{label}: écart CPU {gap_cpu}");
            assert!(
                gap_steel <= NAX_SINK_TOL,
                "{label}: NAX vs Steel {gap_steel}"
            );
        }
    }
    Ok(())
}

/// NAX d128 fenêtré avec sinks (fenêtre flint de 513) à entrées bf16
/// identiques contre l'oracle CPU et contre le naïf bf16
/// (`attention_prefill_sink_bf16`, le chemin qu'il remplace). Un bloc de 1,
/// 300, 512, 513, 514, 1100 et 4133 requêtes ; puis des morceaux derrière un
/// préfixe comme le K/V borné de F6 : `q_off = 513` (après compaction, non
/// aligné sur BK = 32), `q_off = 1100` ou `3833` (avant compaction), avec des
/// queues non alignées sur BQ = 64. GQA 8 sur une tête KV en scores doux,
/// puis ×10 avec des sinks ±8, puis GQA 2 sur quatre têtes KV avec des sinks
/// ±30 (sink seul maître d'une tête sur deux).
#[test]
fn prefill_sink_nax_window_matches_naive_and_cpu() -> Result<()> {
    const KV_TOTAL: usize = 4133;
    let Some(mut executor) = steel_executor()? else {
        return Ok(());
    };
    let Some(nax) = nax_sink_pipeline(&executor, true) else {
        return Ok(());
    };
    let signed_sinks = |magnitude: f32| {
        (0..Q_HEADS)
            .map(|head| if head % 2 == 0 { magnitude } else { -magnitude })
            .collect::<Vec<_>>()
    };
    let q_dim = Q_HEADS * HEAD_DIM;
    let mut nax = Some(nax);
    for (kv_heads, q_scale, sinks) in [
        (KV_HEADS, 1.0_f32, sinks(Q_HEADS)),
        (KV_HEADS, 10.0, signed_sinks(8.0)),
        (4, 10.0, signed_sinks(30.0)),
    ] {
        let (q, keys, values) = inputs(KV_TOTAL, Q_HEADS, kv_heads);
        let q = bf16_rounded(&q.iter().map(|value| value * q_scale).collect::<Vec<_>>());
        let (keys, values) = (bf16_rounded(&keys), bf16_rounded(&values));
        // (requêtes, clés) : `kv_rows > seq` place les requêtes en fin de buffer.
        for (seq, kv_rows) in [
            (1, 1),
            (300, 300),
            (512, 512),
            (513, 513),
            (514, 514),
            (1100, 1100),
            (KV_TOTAL, KV_TOTAL),
            (1, 514),
            (300, 813),
            (1100, 1613),
            (77, 1177),
            (300, KV_TOTAL),
        ] {
            let label = format!("kv_heads={kv_heads} q×{q_scale} q={seq} kv={kv_rows}");
            let kv_len = kv_rows * kv_heads * HEAD_DIM;
            let q_off = kv_rows - seq;
            let q = &q[q_off * q_dim..kv_rows * q_dim];
            let (keys, values) = (&keys[..kv_len], &values[..kv_len]);
            let spec = spec(seq, Q_HEADS, kv_heads, Some(WINDOW));
            let rows = oracle_rows(seq);
            let want = cpu_sink_attention(q, (keys, values), spec, &sinks, q_off, &rows);
            let naive = gpu_attention(&executor, (q, keys, values), spec, Some(&sinks), true)?;
            executor.attention_prefill_nax_sink_d128_window = nax.take();
            let got = gpu_attention(&executor, (q, keys, values), spec, Some(&sinks), true);
            nax = executor.attention_prefill_nax_sink_d128_window.take();
            let got = got?;
            let gap_cpu = row_gap(&got, &want, &rows, q_dim);
            let gap_naive = row_gap(&got, &naive, &(0..seq).collect::<Vec<_>>(), q_dim);
            println!("{label}: écart CPU {gap_cpu:.2e}, NAX vs naïf {gap_naive:.2e}");
            assert!(
                got.iter().all(|value| value.is_finite()),
                "{label}: NaN/inf"
            );
            assert!(gap_cpu <= NAX_SINK_TOL, "{label}: écart CPU {gap_cpu}");
            assert!(
                gap_naive <= NAX_SINK_TOL,
                "{label}: NAX vs naïf {gap_naive}"
            );
        }
    }
    Ok(())
}

/// Microbench des couches pleines flint (32 têtes Q, 4 KV, d128, morceau de
/// 4096 requêtes derrière 16k-64k clés, K/V bf16) : Steel d128 bf16 contre
/// NAX, conversion de Q comprise, meilleur temps de 4 exécutions (GPU au repos).
#[test]
#[ignore = "microbench NAX d128 avec sinks (GPU au repos)"]
fn nax_sink_d128_chunk_microbench() -> Result<()> {
    let Some(mut executor) = steel_executor()? else {
        return Ok(());
    };
    let Some(nax) = nax_sink_pipeline(&executor, false) else {
        return Ok(());
    };
    const SEQ: usize = 4096;
    const FLINT_Q_HEADS: usize = 32;
    const FLINT_KV_HEADS: usize = 4;
    let spec = spec(SEQ, FLINT_Q_HEADS, FLINT_KV_HEADS, None);
    let kv_dim = FLINT_KV_HEADS * HEAD_DIM;
    let sinks = executor.upload_f32_buffer(&sinks(FLINT_Q_HEADS), "mb_sinks")?;
    let q = pattern(SEQ * FLINT_Q_HEADS * HEAD_DIM, 97, 0.02);
    let q_buf = executor.upload_f32_buffer(&q, "mb_q")?;
    let out = executor.uncached_f32_buffer(q.len(), "mb_out")?;
    let mut nax = Some(nax);
    for kv_rows in [16 * 1024, 32 * 1024, 64 * 1024] {
        let kv = pattern(kv_rows * kv_dim, 89, 0.02);
        let k_buf = upload_kv(&executor, &kv, true, "mb_k")?;
        let v_buf = upload_kv(&executor, &kv, true, "mb_v")?;
        let attended = (kv_rows - SEQ / 2) as f64;
        let flops = 4.0 * (FLINT_Q_HEADS * HEAD_DIM * SEQ) as f64 * attended;
        for label in ["steel bf16", "nax"] {
            if label == "nax" {
                executor.attention_prefill_nax_sink_d128 = nax.take();
            }
            let mut best = f64::MAX;
            for _ in 0..4 {
                let cb = executor.queue.new_command_buffer();
                let enc = cb.new_compute_command_encoder();
                let encoded = executor.encode_attention_prefill_sink(
                    enc, &q_buf, &k_buf, &v_buf, &out, &sinks, spec, kv_rows, true,
                );
                enc.end_encoding();
                encoded?;
                let t = std::time::Instant::now();
                commit_and_wait(cb)?;
                best = best.min(t.elapsed().as_secs_f64());
            }
            eprintln!(
                "[sinks d128] kv={kv_rows} q={SEQ} {label} : {:.1} ms, {:.1} TFLOP/s",
                best * 1.0e3,
                flops / best / 1.0e12
            );
        }
        nax = executor.attention_prefill_nax_sink_d128.take();
    }
    Ok(())
}
