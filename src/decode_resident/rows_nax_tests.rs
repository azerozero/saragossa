//! Attention full en lignes de la vérification : préfixe NAX contre l'oracle
//! CPU f64, et coût mesuré des deux chemins à long contexte (Qwen 27B).

use super::rows::RowsNax;
use super::tests::{bf16_round_to_f32, lease_with, pseudo, read_f32, try_state};
use super::*;
use crate::metal_backend::commit_and_wait;

/// Géométrie Qwen3.8-27B : 24 têtes Q, 4 têtes KV (GQA 6), d256.
const Q_HEADS: usize = 24;
const KV_HEADS: usize = 4;
/// Couches full-attn du 27B : un tour de vérification les traverse toutes.
const LAYERS: usize = 16;

fn gpu_state() -> Result<DecodeResidentState> {
    try_state()?
        .ok_or_else(|| InferError::Metal("aucun device Metal : test GPU non exécuté".to_string()))
}

fn nax(state: &DecodeResidentState, head_dim: usize) -> Result<RowsNax> {
    RowsNax::compile(state.device(), head_dim)?
        .ok_or_else(|| InferError::Metal(format!("NAX d{head_dim} indisponible sur ce device")))
}

/// Oracle f64 de la ligne `row` (requêtes `q_prefix` sur `[0, prefix)`,
/// `q_diag` sur `[prefix, prefix + row]`), sortie `[q_head][d]`. `drop_last`
/// retire la position diagonale de la ligne (variante fausse du test).
#[allow(
    clippy::too_many_arguments,
    reason = "oracle CPU de test : requêtes, K/V et géométrie explicites"
)]
fn cpu_row_f64(
    q_prefix: &[f32],
    q_diag: &[f32],
    keys: &[f32],
    values: &[f32],
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    prefix: usize,
    row: usize,
    drop_last: bool,
) -> Vec<f64> {
    let kv_dim = kv_heads * head_dim;
    let gqa = q_heads / kv_heads;
    let scale = 1.0 / (head_dim as f64).sqrt();
    let len = prefix + row + usize::from(!drop_last);
    let mut out = vec![0.0_f64; q_heads * head_dim];
    for qh in 0..q_heads {
        let qs = qh * head_dim;
        let ks = (qh / gqa) * head_dim;
        let scores: Vec<f64> = (0..len)
            .map(|p| {
                let q = if p < prefix { q_prefix } else { q_diag };
                let k = &keys[p * kv_dim + ks..p * kv_dim + ks + head_dim];
                let dot: f64 = q[qs..qs + head_dim]
                    .iter()
                    .zip(k)
                    .map(|(&a, &b)| f64::from(a) * f64::from(b))
                    .sum();
                dot * scale
            })
            .collect();
        let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
        let sum: f64 = weights.iter().sum();
        for (p, w) in weights.iter().enumerate() {
            let v = &values[p * kv_dim + ks..p * kv_dim + ks + head_dim];
            for (o, &x) in out[qs..qs + head_dim].iter_mut().zip(v) {
                *o += w / sum * f64::from(x);
            }
        }
    }
    out
}

/// Un cas : `rows` lignes après `prefix` positions. Renvoie l'écart max
/// (entrées arrondies comme le candidat, Q f32 d'origine, 2-passes actuel)
/// et l'écart de l'oracle à sa variante sans dernière position diagonale.
fn run_nax_case(
    state: &DecodeResidentState,
    kernel: &RowsNax,
    (q_heads, kv_heads, head_dim): (usize, usize, usize),
    prefix: usize,
    rows: usize,
) -> Result<(f64, f64, f64, f64)> {
    let len = prefix + rows;
    let q_dim = q_heads * head_dim;
    let kv_dim = kv_heads * head_dim;
    let q: Vec<f32> = (0..rows * q_dim).map(|i| 30.0 * pseudo(5000 + i)).collect();
    let keys: Vec<f32> = (0..len * kv_dim).map(|i| pseudo(i + 3)).collect();
    // V diagonal ~50 contre ~0,1 au préfixe : une position diagonale oubliée
    // ou comptée deux fois dépasse nettement la tolérance, même à 4k clés.
    let values: Vec<f32> = (0..len * kv_dim)
        .map(|i| {
            let value = pseudo(i + 11);
            if i >= prefix * kv_dim {
                50.0 * (1.0 + value)
            } else {
                value
            }
        })
        .collect();
    let round = |data: &[f32]| {
        data.iter()
            .copied()
            .map(bf16_round_to_f32)
            .collect::<Vec<_>>()
    };
    let (keys_bf16, values_bf16, q_bf16) = (round(&keys), round(&values), round(&q));
    let mut kv =
        state.full_attention_format_for_test(len, q_heads, kv_heads, head_dim, KvFormat::Bf16)?;
    kv.seed(&keys, &values, len)?;
    let q_buf = lease_with(state, &q)?;
    let out_nax = state.scratch().lease(rows * q_dim, GpuElement::F32)?;
    let out_ref = state.scratch().lease(rows * q_dim, GpuElement::F32)?;
    let kernels = state.rows_kernels()?;
    let command_buffer = state.queue().new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    kv.encode_attention_rows_nax(
        encoder,
        kernel,
        q_buf.tensor().buffer(),
        out_nax.tensor().buffer(),
        rows,
    )?;
    kv.encode_attention_decode_rows_chunked(
        encoder,
        kernels,
        q_buf.tensor().buffer(),
        out_ref.tensor().buffer(),
        rows,
        None,
        None,
    )?;
    encoder.end_encoding();
    commit_and_wait(command_buffer)?;
    let (got, current) = (read_f32(out_nax.tensor()), read_f32(out_ref.tensor()));
    let label = format!("d{head_dim} {q_heads}/{kv_heads} prefix={prefix} rows={rows}");
    // `f32::max` ignore les NaN : une sortie non finie passerait l'écart à 0.
    assert!(
        got.iter().chain(&current).all(|value| value.is_finite()),
        "{label}: sortie non finie"
    );
    let (mut same_inputs, mut total, mut vs_current) = (0.0_f64, 0.0_f64, 0.0_f64);
    // Écart de l'oracle à sa variante sans la dernière position diagonale.
    let mut discriminant = 0.0_f64;
    for row in 0..rows {
        let range = row * q_dim..(row + 1) * q_dim;
        let q_row = &q[range.clone()];
        let rounded = cpu_row_f64(
            &q_bf16[range.clone()],
            q_row,
            &keys_bf16,
            &values_bf16,
            q_heads,
            kv_heads,
            head_dim,
            prefix,
            row,
            false,
        );
        let exact = cpu_row_f64(
            q_row,
            q_row,
            &keys_bf16,
            &values_bf16,
            q_heads,
            kv_heads,
            head_dim,
            prefix,
            row,
            false,
        );
        let wrong = cpu_row_f64(
            &q_bf16[range.clone()],
            q_row,
            &keys_bf16,
            &values_bf16,
            q_heads,
            kv_heads,
            head_dim,
            prefix,
            row,
            true,
        );
        discriminant = rounded
            .iter()
            .zip(&wrong)
            .map(|(a, b)| (a - b).abs())
            .fold(discriminant, f64::max);
        for (i, &value) in got[range.clone()].iter().enumerate() {
            let value = f64::from(value);
            same_inputs = same_inputs.max((value - rounded[i]).abs());
            total = total.max((value - exact[i]).abs());
            vs_current = vs_current.max((value - f64::from(current[range.start + i])).abs());
        }
    }
    eprintln!(
        "{label}: vs oracle arrondi {same_inputs:e}, vs oracle f32 {total:e}, \
         vs 2-passes {vs_current:e}, diagonale retirée {discriminant:e}"
    );
    Ok((same_inputs, total, vs_current, discriminant))
}

/// Préfixe NAX en lignes contre l'oracle CPU f64 : reste de bloc, préfixe
/// multiple exact des blocs (pas de reste), préfixe plus court qu'un bloc
/// (aucun bloc plein), lignes 1 à 16 (diagonale), d128 (2 tuiles Q) et d256.
#[test]
#[ignore = "GPU Metal requis : échoue explicitement sans device"]
fn verify_rows_nax_matches_cpu_oracle() -> Result<()> {
    let state = gpu_state()?;
    // Tolérance à entrées égales (K/V du KV bf16, Q du préfixe arrondi en bf16
    // comme le candidat, diagonale en Q f32) : reste l'arrondi du MMA PV et
    // l'ordre des réductions. Écart observé ≤ 2,3e-5 (sorties jusqu'à ~50 sur
    // la diagonale) : marge ×4. La variante sans dernière position diagonale
    // s'écarte d'au moins 1,7e-2, le test discrimine. L'écart à l'oracle en Q
    // f32 (dérive totale due à Q en bf16) est imprimé, pas borné.
    const EPS_SAME_INPUTS: f64 = 1.0e-4;
    let mut worst = (0.0_f64, 0.0_f64, 0.0_f64);
    for (geometry, cases) in [
        (
            (Q_HEADS, KV_HEADS, 256),
            &[
                (4133, 1),
                (4133, 2),
                (4133, 3),
                (4133, 6),
                (4133, 8),
                (4133, 16),
                (2048, 8),
                (100, 3),
                (5, 2),
            ][..],
        ),
        ((16, 2, 128), &[(4133, 8), (2048, 16), (100, 6), (5, 1)][..]),
    ] {
        let kernel = nax(&state, geometry.2)?;
        for &(prefix, rows) in cases {
            let (same, total, current, discriminant) =
                run_nax_case(&state, &kernel, geometry, prefix, rows)?;
            assert!(
                same <= EPS_SAME_INPUTS,
                "d{} prefix={prefix} rows={rows}: écart {same:e} > {EPS_SAME_INPUTS:e}",
                geometry.2
            );
            assert!(
                discriminant > 10.0 * EPS_SAME_INPUTS,
                "d{} prefix={prefix} rows={rows}: test non discriminant ({discriminant:e})",
                geometry.2
            );
            worst = (worst.0.max(same), worst.1.max(total), worst.2.max(current));
        }
    }
    eprintln!(
        "rows NAX : pire écart vs oracle arrondi {:e}, vs oracle f32 {:e}, vs 2-passes {:e}",
        worst.0, worst.1, worst.2
    );
    Ok(())
}

/// Bench (`--ignored --nocapture`) : temps d'un tour d'attention en lignes
/// (16 couches dans un command buffer) à `len` positions, bf16 dense, chemin
/// actuel puis préfixe NAX.
#[test]
#[ignore = "GPU Metal requis : bench manuel de l'attention en lignes"]
fn bench_rows_attention_long_context() -> Result<()> {
    let state = gpu_state()?;
    let kernels = state.rows_kernels()?;
    let head_dim = 256;
    let kernel = nax(&state, head_dim)?;
    let kv_dim = KV_HEADS * head_dim;
    let q_dim = Q_HEADS * head_dim;
    eprintln!(
        "pass1 max_total_threads_per_threadgroup = {}",
        kernels
            .two_pass_rows_d256_bf16
            .max_total_threads_per_threadgroup()
    );
    for len in [32_768, 65_536] {
        let keys: Vec<f32> = (0..len * kv_dim).map(|i| pseudo(i + 3)).collect();
        let values: Vec<f32> = (0..len * kv_dim).map(|i| pseudo(i + 11)).collect();
        let mut kv = state.full_attention_format_for_test(
            len,
            Q_HEADS,
            KV_HEADS,
            head_dim,
            KvFormat::Bf16,
        )?;
        kv.seed(&keys, &values, len)?;
        drop((keys, values));
        for rows in [1, 2, 3, 8, 16] {
            let q: Vec<f32> = (0..rows * q_dim).map(|i| pseudo(7 + i)).collect();
            let q_buf = lease_with(&state, &q)?;
            let out = state.scratch().lease(rows * q_dim, GpuElement::F32)?;
            let mut best = [f64::INFINITY; 2];
            for _ in 0..6 {
                for (path, best) in best.iter_mut().enumerate() {
                    let command_buffer = state.queue().new_command_buffer();
                    let encoder = command_buffer.new_compute_command_encoder();
                    for _ in 0..LAYERS {
                        if path == 0 {
                            kv.encode_attention_decode_rows(
                                encoder,
                                kernels,
                                q_buf.tensor().buffer(),
                                out.tensor().buffer(),
                                rows,
                                None,
                            )?;
                        } else {
                            kv.encode_attention_rows_nax(
                                encoder,
                                &kernel,
                                q_buf.tensor().buffer(),
                                out.tensor().buffer(),
                                rows,
                            )?;
                        }
                    }
                    encoder.end_encoding();
                    let started = std::time::Instant::now();
                    commit_and_wait(command_buffer)?;
                    *best = best.min(started.elapsed().as_secs_f64() * 1e3);
                }
            }
            eprintln!(
                "rows-attn len={len} N={rows}: actuel {:.2} ms, NAX {:.2} ms / {LAYERS} couches",
                best[0], best[1]
            );
        }
    }
    Ok(())
}
