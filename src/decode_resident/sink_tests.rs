//! Attention decode avec sinks appris (flint) contre l'oracle CPU, et
//! équivalence bit à bit des variantes avec sink neutre et sans sink.

use super::*;
use crate::decoder::attention_probs_in_place;

/// Géométrie flint : 32 têtes Q, 4 têtes KV (GQA 8), d = 128.
const Q_HEADS: usize = 32;
const KV_HEADS: usize = 4;
const HEAD_DIM: usize = 128;
/// Sink assez négatif pour un `exp` nul et un max inchangé, sans `-inf`
/// (les kernels sont compilés en fast-math).
const NEUTRAL_SINK: f32 = -1.0e30;

fn try_state() -> Result<Option<DecodeResidentState>> {
    match Device::system_default() {
        Some(device) => Ok(Some(DecodeResidentState::new(device)?)),
        None => Ok(None),
    }
}

fn pattern(len: usize, modulo: usize, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|i| ((i * 7919 % modulo) as f32 - modulo as f32 / 2.0) * scale)
        .collect()
}

/// Sinks variés : l'un domine les scores (8,0) pour exercer le recalage du
/// numérateur quand le sink relève le maximum.
fn sinks() -> Vec<f32> {
    (0..Q_HEADS)
        .map(|head| {
            if head == 3 {
                8.0
            } else {
                head as f32 * 0.25 - 4.0
            }
        })
        .collect()
}

fn bf16_round(values: &[f32]) -> Vec<f32> {
    values
        .iter()
        .map(|value| super::utils::bf16_round_f32(*value))
        .collect()
}

/// Oracle CPU : scores `q·k/√d` sur la fenêtre, probabilités par
/// `attention_probs_in_place` (le noyau de l'oracle CPU flint), somme pondérée.
fn cpu_sink_attention(
    q: &[f32],
    keys: &[f32],
    values: &[f32],
    len: usize,
    window_start: usize,
    sinks: &[f32],
) -> Vec<f32> {
    let kv_dim = KV_HEADS * HEAD_DIM;
    let group = Q_HEADS / KV_HEADS;
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let mut out = vec![0.0_f32; Q_HEADS * HEAD_DIM];
    for head in 0..Q_HEADS {
        let kv_head = head / group;
        let query = &q[head * HEAD_DIM..(head + 1) * HEAD_DIM];
        let mut scores = (window_start..len)
            .map(|row| {
                let key = &keys[row * kv_dim + kv_head * HEAD_DIM..][..HEAD_DIM];
                query.iter().zip(key).map(|(a, b)| a * b).sum::<f32>() * scale
            })
            .collect::<Vec<_>>();
        attention_probs_in_place(&mut scores, Some(sinks[head]));
        for (offset, prob) in scores.iter().enumerate() {
            let row = window_start + offset;
            let value = &values[row * kv_dim + kv_head * HEAD_DIM..][..HEAD_DIM];
            for (dst, v) in out[head * HEAD_DIM..].iter_mut().zip(value) {
                *dst += prob * v;
            }
        }
    }
    out
}

struct Case {
    q: Vec<f32>,
    keys: Vec<f32>,
    values: Vec<f32>,
}

fn case(len: usize) -> Case {
    let kv_dim = KV_HEADS * HEAD_DIM;
    Case {
        q: pattern(Q_HEADS * HEAD_DIM, 89, 0.02),
        keys: pattern(len * kv_dim, 101, 0.01),
        values: pattern(len * kv_dim, 97, 0.01),
    }
}

fn kv_state(
    state: &DecodeResidentState,
    case: &Case,
    len: usize,
    format: KvFormat,
    sinks: Option<&[f32]>,
) -> Result<FullAttentionMetalState> {
    let mut kv = state.full_attention_format_for_test(len, Q_HEADS, KV_HEADS, HEAD_DIM, format)?;
    kv.seed(&case.keys, &case.values, len)?;
    if let Some(sinks) = sinks {
        let buffer = state.device.new_buffer_with_data(
            sinks.as_ptr().cast(),
            std::mem::size_of_val(sinks) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        kv.set_sinks(state, buffer)?;
    }
    Ok(kv)
}

/// Écart max entre deux sorties finies : `f32::max` ignore les NaN, une sortie
/// entièrement NaN passerait sinon pour un écart nul.
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    assert!(
        a.iter().chain(b).all(|value| value.is_finite()),
        "sortie non finie"
    );
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// Couvre le 2-passes d128 (fenêtre nulle, court et au-delà du seuil 2048) et
/// fenêtré (fenêtre flint de 513 lignes), en f32 et bf16.
#[test]
fn sink_attention_matches_cpu_oracle() -> Result<()> {
    let Some(state) = try_state()? else {
        return Ok(());
    };
    let sinks = sinks();
    for (len, window_start) in [(1, 0), (300, 0), (3000, 0), (600, 600 - 513)] {
        let case = case(len);
        for format in [KvFormat::F32, KvFormat::Bf16] {
            let (keys, values, eps) = match format {
                KvFormat::Bf16 => (bf16_round(&case.keys), bf16_round(&case.values), 2.0e-5),
                _ => (case.keys.clone(), case.values.clone(), 2.0e-5),
            };
            let expected = cpu_sink_attention(&case.q, &keys, &values, len, window_start, &sinks);
            let kv = kv_state(&state, &case, len, format, Some(&sinks))?;
            let got = kv.attention_decode_windowed(&case.q, window_start)?;
            let gap = max_abs_diff(&got, &expected);
            assert!(
                gap <= eps,
                "len={len} fenêtre={window_start} {format:?}: écart {gap}"
            );
        }
    }
    Ok(())
}

/// Fenêtre flint de 513 lignes : le 2-passes sur la vue décalée du KV (choisi
/// en production pour d128/GQA 8) contre le naïf avec sink et l'oracle CPU.
/// L'ordre de réduction diffère (partitions entrelacées) : tolérance, pas bits.
// NOTE: Mesuré le 30 oct. 2026 (M5 Max, trois exécutions identiques) : pire
// écart 1,72e-8 contre le naïf et 1,68e-8 contre le CPU, en f32 comme en bf16.
// 1e-6 laisse ~60× de marge tout en restant assez serré pour qu'un décalage de
// vue K/V faux (lignes hors fenêtre) se voie ; les 2e-5 du test en fenêtre
// nulle couvrent 3 000 lignes et le sink dominant, pas ce cas.
#[test]
fn windowed_two_pass_sink_matches_naive_and_cpu_oracle() -> Result<()> {
    let Some(state) = try_state()? else {
        return Ok(());
    };
    let sinks = sinks();
    for len in [513_usize, 514, 600, 1100, 4000] {
        let window_start = len.saturating_sub(513);
        let case = case(len);
        for format in [KvFormat::F32, KvFormat::Bf16] {
            let (keys, values) = match format {
                KvFormat::Bf16 => (bf16_round(&case.keys), bf16_round(&case.values)),
                _ => (case.keys.clone(), case.values.clone()),
            };
            let expected = cpu_sink_attention(&case.q, &keys, &values, len, window_start, &sinks);
            let kv = kv_state(&state, &case, len, format, Some(&sinks))?;
            let got = kv.attention_decode_sink_two_pass(&case.q, window_start)?;
            let naive = kv.attention_decode_sink_naive(&case.q, window_start)?;
            let production = kv.attention_decode_windowed(&case.q, window_start)?;
            // La production prend le 2-passes (bits identiques), sauf si un flag
            // runtime la ramène au naïf : comparée alors à tolérance.
            assert!(max_abs_diff(&production, &got) <= 1.0e-6);
            let to_naive = max_abs_diff(&got, &naive);
            let to_cpu = max_abs_diff(&got, &expected);
            assert!(
                to_naive <= 1.0e-6 && to_cpu <= 1.0e-6,
                "len={len} fenêtre={window_start} {format:?}: écart naïf {to_naive}, CPU {to_cpu}"
            );
        }
    }
    Ok(())
}

/// Un sink neutre réduit les kernels avec sink à l'arithmétique historique :
/// `max(m, s) = m` et `+ exp(s - m) = + 0`. Les deux sorties sont égales bit à
/// bit, et les kernels sans sink (source distincte, inchangée) le restent par
/// construction pour Qwen et Gemma.
#[test]
fn neutral_sink_is_bit_identical_to_historical_kernels() -> Result<()> {
    let Some(state) = try_state()? else {
        return Ok(());
    };
    let neutral = vec![NEUTRAL_SINK; Q_HEADS];
    // 3000 ≥ seuil 2-passes (fenêtre nulle) ; fenêtre > 0 : l'historique est
    // le naïf, comparé au naïf avec sink (le 2-passes fenêtré n'existe qu'avec
    // sink et se compare à tolérance ci-dessus).
    for (len, window_start) in [(3000, 0), (600, 600 - 513)] {
        let case = case(len);
        for format in [KvFormat::F32, KvFormat::Bf16] {
            let historical = kv_state(&state, &case, len, format, None)?
                .attention_decode_windowed(&case.q, window_start)?;
            let sink_kv = kv_state(&state, &case, len, format, Some(&neutral))?;
            let with_sink = if window_start == 0 {
                sink_kv.attention_decode_windowed(&case.q, window_start)?
            } else {
                sink_kv.attention_decode_sink_naive(&case.q, window_start)?
            };
            assert!(
                historical
                    .iter()
                    .zip(&with_sink)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "len={len} fenêtre={window_start} {format:?}: sink neutre ≠ historique"
            );
        }
    }
    Ok(())
}

#[test]
fn sinks_refused_on_quantized_kv() -> Result<()> {
    let Some(state) = try_state()? else {
        return Ok(());
    };
    let case = case(64);
    let sinks = sinks();
    let err = kv_state(&state, &case, 64, KvFormat::Q8, Some(&sinks))
        .expect_err("sinks refusés sur KV quantifié");
    assert!(err.to_string().contains("sinks"), "{err}");
    Ok(())
}
