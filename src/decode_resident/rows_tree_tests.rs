//! Chemin full-attn en lignes sous un arbre de vérification (DFlash) : attention
//! masquée par ancêtres (flash, 2-passes, NAX) et RoPE par profondeur, contre un
//! oracle CPU f64 et, pour une chaîne, au bit près contre le chemin historique.

use super::rows::RowsNax;
use super::tests::{bf16_round_to_f32, lease_with, pseudo, read_f32, try_state};
use super::*;
use crate::metal_backend::{commit_and_wait, install_verify_tree_scope, VerifyTree};
use std::rc::Rc;

const Q_HEADS: usize = 4;
const KV_HEADS: usize = 2;

fn gpu_state() -> Result<DecodeResidentState> {
    try_state()?
        .ok_or_else(|| InferError::Metal("aucun device Metal : test GPU non exécuté".to_string()))
}

/// Arbre de 9 lignes donné par ses parents.
fn tree(parents: &[usize]) -> Result<Rc<VerifyTree>> {
    let mut depths = vec![0usize; parents.len()];
    let mut masks = vec![1u32; parents.len()];
    for row in 1..parents.len() {
        depths[row] = depths[parents[row]] + 1;
        masks[row] = masks[parents[row]] | (1 << row);
    }
    Ok(Rc::new(VerifyTree::new(parents, &depths, &masks)?))
}

/// Chaîne, arbre touffu et arbre mêlant branches profondes et feuilles.
const TREES: [[usize; 9]; 3] = [
    [0, 0, 1, 2, 3, 4, 5, 6, 7],
    [0, 0, 0, 1, 1, 2, 3, 3, 5],
    [0, 0, 1, 2, 1, 4, 0, 6, 3],
];

/// Oracle f64 de la ligne `row` : préfixe `[0, context)` (requête `q_prefix`)
/// puis lignes d'arbre visibles selon `mask` (requête `q_tree`).
#[allow(
    clippy::too_many_arguments,
    reason = "oracle CPU de test : requêtes, K/V, masque et géométrie explicites"
)]
fn cpu_tree_row(
    q_prefix: &[f32],
    q_tree: &[f32],
    keys: &[f32],
    values: &[f32],
    head_dim: usize,
    context: usize,
    mask: u32,
    rows: usize,
) -> Vec<f64> {
    let kv_dim = KV_HEADS * head_dim;
    let gqa = Q_HEADS / KV_HEADS;
    let scale = 1.0 / (head_dim as f64).sqrt();
    let visible: Vec<usize> = (0..context)
        .chain(
            (0..rows)
                .filter(|j| mask >> j & 1 == 1)
                .map(|j| context + j),
        )
        .collect();
    let mut out = vec![0.0_f64; Q_HEADS * head_dim];
    for qh in 0..Q_HEADS {
        let qs = qh * head_dim;
        let ks = (qh / gqa) * head_dim;
        let scores: Vec<f64> = visible
            .iter()
            .map(|&p| {
                let q = if p < context { q_prefix } else { q_tree };
                let k = &keys[p * kv_dim + ks..p * kv_dim + ks + head_dim];
                q[qs..qs + head_dim]
                    .iter()
                    .zip(k)
                    .map(|(&a, &b)| f64::from(a) * f64::from(b))
                    .sum::<f64>()
                    * scale
            })
            .collect();
        let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
        let sum: f64 = weights.iter().sum();
        for (&p, w) in visible.iter().zip(&weights) {
            let v = &values[p * kv_dim + ks..p * kv_dim + ks + head_dim];
            for (o, &x) in out[qs..qs + head_dim].iter_mut().zip(v) {
                *o += w / sum * f64::from(x);
            }
        }
    }
    out
}

/// Sortie GPU, requêtes, K et V (arrondis bf16 si KV bf16) d'un cas.
type CaseRun = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

struct Case {
    head_dim: usize,
    format: KvFormat,
    context: usize,
}

/// Lance l'attention en lignes de `rows` lignes sous `tree` (ou sans arbre) et
/// rend la sortie ; `nax` force le NAX arbre (KV bf16).
fn run_case(
    state: &DecodeResidentState,
    case: &Case,
    tree: Option<Rc<VerifyTree>>,
    group: Option<usize>,
    nax: bool,
) -> Result<CaseRun> {
    let rows = 9;
    let Case {
        head_dim,
        format,
        context,
    } = *case;
    let total = context + rows;
    let q_dim = Q_HEADS * head_dim;
    let kv_dim = KV_HEADS * head_dim;
    // q amplifié : softmax piquée, une clé masquée comptée à tort se voit.
    let q: Vec<f32> = (0..rows * q_dim).map(|i| 30.0 * pseudo(7000 + i)).collect();
    let keys: Vec<f32> = (0..total * kv_dim).map(|i| pseudo(i + 3)).collect();
    let values: Vec<f32> = (0..total * kv_dim).map(|i| pseudo(i + 11)).collect();
    let mut kv =
        state.full_attention_format_for_test(total, Q_HEADS, KV_HEADS, head_dim, format)?;
    kv.seed(&keys, &values, total)?;
    let q_buf = lease_with(state, &q)?;
    let out = state.scratch().lease(rows * q_dim, GpuElement::F32)?;
    let kernels = state.rows_kernels()?;
    let nax_kernels = if nax {
        Some(
            RowsNax::compile(state.device(), head_dim, tree.is_some())?
                .ok_or_else(|| InferError::Metal("NAX indisponible".to_string()))?,
        )
    } else {
        None
    };
    let command_buffer = state.queue().new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    {
        let _scope = tree.map(install_verify_tree_scope);
        match &nax_kernels {
            Some(nax) => kv.encode_attention_rows_nax(
                encoder,
                nax,
                q_buf.tensor().buffer(),
                out.tensor().buffer(),
                rows,
            )?,
            None => kv.encode_attention_decode_rows_chunked(
                encoder,
                kernels,
                q_buf.tensor().buffer(),
                out.tensor().buffer(),
                rows,
                None,
                group,
            )?,
        }
    }
    encoder.end_encoding();
    commit_and_wait(command_buffer)?;
    let round = |data: &[f32]| -> Vec<f32> {
        if format == KvFormat::Bf16 {
            data.iter().copied().map(bf16_round_to_f32).collect()
        } else {
            data.to_vec()
        }
    };
    Ok((read_f32(out.tensor()), q, round(&keys), round(&values)))
}

#[test]
#[ignore = "GPU Metal requis : échoue explicitement sans device"]
fn full_attn_rows_tree_matches_cpu() -> Result<()> {
    let state = gpu_state()?;
    // Ordres de réduction différents du naïf f64 (exp rapide, max glissant) ;
    // écart observé imprimé en fin de test.
    const EPS: f64 = 1.0e-5;
    let mut worst = 0.0_f64;
    for head_dim in [128, 256] {
        for format in [KvFormat::F32, KvFormat::Bf16] {
            // Flash (court), 2-passes (≥ 2048) et bascule du nombre de blocs.
            for context in [5, 40, 2100, 4220] {
                let case = Case {
                    head_dim,
                    format,
                    context,
                };
                for group in [None, Some(1), Some(3)] {
                    let (chain_ref, ..) = run_case(&state, &case, None, group, false)?;
                    for (index, parents) in TREES.iter().enumerate() {
                        let tree = tree(parents)?;
                        let masks = tree.masks().to_vec();
                        let (got, q, keys, values) =
                            run_case(&state, &case, Some(tree), group, false)?;
                        let label = format!(
                            "d{head_dim} {format:?} contexte {context} groupe {group:?} arbre {parents:?}"
                        );
                        assert!(got.iter().all(|v| v.is_finite()), "{label}: non fini");
                        if index == 0 {
                            // Chaîne : bit à bit le chemin historique.
                            let bits =
                                |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                            assert_eq!(bits(&got), bits(&chain_ref), "{label}: chaîne");
                        }
                        let q_dim = Q_HEADS * head_dim;
                        let mut discriminant = 0.0_f64;
                        for (row, &mask) in masks.iter().enumerate() {
                            let range = row * q_dim..(row + 1) * q_dim;
                            let q_row = &q[range.clone()];
                            let expected = cpu_tree_row(
                                q_row, q_row, &keys, &values, head_dim, context, mask, 9,
                            );
                            let causal = cpu_tree_row(
                                q_row,
                                q_row,
                                &keys,
                                &values,
                                head_dim,
                                context,
                                (1u32 << (row + 1)) - 1,
                                9,
                            );
                            for (i, (&g, &e)) in got[range].iter().zip(&expected).enumerate() {
                                let delta = (f64::from(g) - e).abs();
                                worst = worst.max(delta);
                                assert!(delta <= EPS, "{label} ligne {row} [{i}]: {g} vs {e}");
                            }
                            discriminant = expected
                                .iter()
                                .zip(&causal)
                                .map(|(a, b)| (a - b).abs())
                                .fold(discriminant, f64::max);
                        }
                        // Le masque change réellement le résultat hors chaîne
                        // (contexte court : les lignes d'arbre pèsent).
                        if index > 0 && context < 100 {
                            assert!(discriminant > 100.0 * EPS, "{label}: masque sans effet");
                        }
                    }
                }
            }
        }
    }
    eprintln!("attention en lignes arbre : écart max à l'oracle f64 {worst:e}");
    Ok(())
}

#[test]
#[ignore = "GPU Metal requis : échoue explicitement sans device"]
fn full_attn_rows_tree_nax_matches_cpu() -> Result<()> {
    let state = gpu_state()?;
    // Préfixe en Q bf16 (comme `verify_rows_nax_matches_cpu_oracle`), diagonale
    // en Q f32 : oracle aux mêmes entrées, écart d'accumulation seul.
    const EPS: f64 = 1.0e-4;
    let mut worst = 0.0_f64;
    for head_dim in [128, 256] {
        for context in [2100, 4300] {
            let case = Case {
                head_dim,
                format: KvFormat::Bf16,
                context,
            };
            let (chain_ref, ..) = run_case(&state, &case, None, None, true)?;
            for (index, parents) in TREES.iter().enumerate() {
                let tree = tree(parents)?;
                let masks = tree.masks().to_vec();
                let (got, q, keys, values) = run_case(&state, &case, Some(tree), None, true)?;
                let label = format!("NAX d{head_dim} contexte {context} arbre {parents:?}");
                if index == 0 {
                    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                    assert_eq!(bits(&got), bits(&chain_ref), "{label}: chaîne");
                }
                let q_dim = Q_HEADS * head_dim;
                let q_bf16: Vec<f32> = q.iter().copied().map(bf16_round_to_f32).collect();
                for (row, &mask) in masks.iter().enumerate() {
                    let range = row * q_dim..(row + 1) * q_dim;
                    let expected = cpu_tree_row(
                        &q_bf16[range.clone()],
                        &q[range.clone()],
                        &keys,
                        &values,
                        head_dim,
                        context,
                        mask,
                        9,
                    );
                    for (i, (&g, &e)) in got[range].iter().zip(&expected).enumerate() {
                        let delta = (f64::from(g) - e).abs();
                        worst = worst.max(delta);
                        assert!(delta <= EPS, "{label} ligne {row} [{i}]: {g} vs {e}");
                    }
                }
            }
        }
    }
    eprintln!("attention NAX arbre : écart max à l'oracle f64 {worst:e}");
    Ok(())
}

/// RoPE en lignes sous un arbre : la ligne `r` tourne à `base + profondeur(r)`,
/// au bit près le kernel historique lancé ligne par ligne à cette position.
#[test]
#[ignore = "GPU Metal requis : échoue explicitement sans device"]
fn rope_rows_tree_rotates_at_depth() -> Result<()> {
    let state = gpu_state()?;
    let (heads, head_dim, rows, base) = (4usize, 256usize, 9usize, 4093usize);
    let width = heads * head_dim;
    let input: Vec<f32> = (0..rows * width).map(|i| pseudo(300 + i)).collect();
    let weight: Vec<f32> = (0..head_dim).map(|i| 1.0 + 0.1 * pseudo(i)).collect();
    let input_buf = lease_with(&state, &input)?;
    let weight_buf = lease_with(&state, &weight)?;
    let kernels = state.rows_kernels()?;
    let executor = crate::metal_backend::MetalExecutor::new()?;
    for parents in TREES {
        let tree = tree(&parents)?;
        let depths = tree.depths().to_vec();
        let got = state.scratch().lease(rows * width, GpuElement::F32)?;
        let expected = state.scratch().lease(rows * width, GpuElement::F32)?;
        let command_buffer = state.queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        {
            let _scope = install_verify_tree_scope(tree);
            state.encode_rms_norm_rope_decode_rows(
                encoder,
                kernels,
                input_buf.tensor().buffer(),
                0,
                width,
                weight_buf.tensor().buffer(),
                got.tensor().buffer(),
                width,
                heads,
                head_dim,
                64,
                64,
                base,
                rows,
                1.0e-6,
                1.0e7,
            )?;
        }
        for (row, &depth) in depths.iter().enumerate() {
            // Une ligne : sortie écrite à `row · width` via l'offset d'entrée et
            // un stride de sortie nul ; on recopie ensuite la ligne.
            let single = state.scratch().lease(width, GpuElement::F32)?;
            state.encode_rms_norm_rope_decode_rows(
                encoder,
                kernels,
                input_buf.tensor().buffer(),
                (row * width * 4) as u64,
                width,
                weight_buf.tensor().buffer(),
                single.tensor().buffer(),
                width,
                heads,
                head_dim,
                64,
                64,
                base + depth as usize,
                1,
                1.0e-6,
                1.0e7,
            )?;
            executor.encode_copy_with_offsets(
                encoder,
                single.tensor().buffer(),
                0,
                expected.tensor().buffer(),
                (row * width * 4) as u64,
                width,
            )?;
        }
        encoder.end_encoding();
        commit_and_wait(command_buffer)?;
        let bits = |v: Vec<f32>| v.into_iter().map(f32::to_bits).collect::<Vec<_>>();
        assert_eq!(
            bits(read_f32(got.tensor())),
            bits(read_f32(expected.tensor())),
            "RoPE arbre {parents:?}"
        );
    }
    Ok(())
}
