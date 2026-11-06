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

/// Arbre donné par ses parents (ordre topologique).
fn tree(parents: &[usize]) -> Result<Rc<VerifyTree>> {
    let mut depths = vec![0usize; parents.len()];
    let mut masks = vec![1u32; parents.len()];
    for row in 1..parents.len() {
        depths[row] = depths[parents[row]] + 1;
        masks[row] = masks[parents[row]] | (1 << row);
    }
    Ok(Rc::new(VerifyTree::new(parents, &depths, &masks)?))
}

/// Chaînes (9 et 16 lignes), arbres touffus et arbres mêlant branches
/// profondes et feuilles, jusqu'aux 16 lignes du journal (B = 15).
const TREES: [&[usize]; 5] = [
    &[0, 0, 1, 2, 3, 4, 5, 6, 7],
    &[0, 0, 0, 1, 1, 2, 3, 3, 5],
    &[0, 0, 1, 2, 1, 4, 0, 6, 3],
    &[0, 0, 0, 1, 1, 2, 3, 3, 5, 4, 6, 8, 9],
    &[0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
];

/// Vrai pour une chaîne (`parent = ligne − 1`) : comparée bit à bit au chemin
/// historique.
fn is_chain(parents: &[usize]) -> bool {
    parents
        .iter()
        .enumerate()
        .skip(1)
        .all(|(row, &parent)| parent + 1 == row)
}

/// Oracle f64 de la ligne `row` : préfixe `[0, context)` (requête `q_prefix`)
/// puis lignes d'arbre visibles selon `mask` (requête `q_tree`).
#[expect(
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
    rows: usize,
    tree: Option<Rc<VerifyTree>>,
    group: Option<usize>,
    nax: bool,
) -> Result<CaseRun> {
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
                    for parents in TREES {
                        let rows = parents.len();
                        let tree = tree(parents)?;
                        let masks = tree.masks().to_vec();
                        let (got, q, keys, values) =
                            run_case(&state, &case, rows, Some(tree), group, false)?;
                        let label = format!(
                            "d{head_dim} {format:?} contexte {context} groupe {group:?} arbre {parents:?}"
                        );
                        assert!(got.iter().all(|v| v.is_finite()), "{label}: non fini");
                        if is_chain(parents) {
                            // Chaîne : bit à bit le chemin historique.
                            let (chain_ref, ..) =
                                run_case(&state, &case, rows, None, group, false)?;
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
                                q_row, q_row, &keys, &values, head_dim, context, mask, rows,
                            );
                            let causal = cpu_tree_row(
                                q_row,
                                q_row,
                                &keys,
                                &values,
                                head_dim,
                                context,
                                (1u32 << (row + 1)) - 1,
                                rows,
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
                        if !is_chain(parents) && context < 100 {
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
            for parents in TREES {
                let rows = parents.len();
                let tree = tree(parents)?;
                let masks = tree.masks().to_vec();
                let (got, q, keys, values) = run_case(&state, &case, rows, Some(tree), None, true)?;
                let label = format!("NAX d{head_dim} contexte {context} arbre {parents:?}");
                if is_chain(parents) {
                    let (chain_ref, ..) = run_case(&state, &case, rows, None, None, true)?;
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
                        rows,
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
    let frequencies =
        crate::metal_backend::rope::create_rope_frequencies(state.device(), 1.0e7, 64)?;
    let (heads, head_dim, max_rows, base) = (4usize, 256usize, 16usize, 4093usize);
    let width = heads * head_dim;
    let input: Vec<f32> = (0..max_rows * width).map(|i| pseudo(300 + i)).collect();
    let weight: Vec<f32> = (0..head_dim).map(|i| 1.0 + 0.1 * pseudo(i)).collect();
    let input_buf = lease_with(&state, &input)?;
    let weight_buf = lease_with(&state, &weight)?;
    let kernels = state.rows_kernels()?;
    let executor = crate::metal_backend::MetalExecutor::new()?;
    for parents in TREES {
        let rows = parents.len();
        let tree = tree(parents)?;
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
                &frequencies,
                base,
                rows,
                1.0e-6,
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
                &frequencies,
                base + depth as usize,
                1,
                1.0e-6,
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

/// Compaction du KV après un tour en arbre : lignes du chemin ramenées au bout
/// du préfixe (oracle CPU), longueur, préfixe adopté intact, refus des chemins
/// invalides et du KV quantifié ; éligibilité arbre par format.
#[test]
#[ignore = "GPU Metal requis : échoue explicitement sans device"]
fn compact_tree_rows_matches_cpu_and_refuses_invalid_paths() -> Result<()> {
    let state = gpu_state()?;
    let head_dim = 128;
    let kv_dim = KV_HEADS * head_dim;
    let (context, rows) = (6usize, 9usize);
    let total = context + rows;
    let keys: Vec<f32> = (0..total * kv_dim).map(|i| pseudo(i + 21)).collect();
    let values: Vec<f32> = (0..total * kv_dim).map(|i| pseudo(i + 77)).collect();
    for format in [KvFormat::F32, KvFormat::Bf16] {
        let round = |data: &[f32]| -> Vec<f32> {
            if format == KvFormat::Bf16 {
                data.iter().copied().map(bf16_round_to_f32).collect()
            } else {
                data.to_vec()
            }
        };
        let (keys_ref, values_ref) = (round(&keys), round(&values));
        let make = || -> Result<FullAttentionMetalState> {
            let mut kv = state.full_attention_format_for_test(
                total + 2,
                Q_HEADS,
                KV_HEADS,
                head_dim,
                format,
            )?;
            kv.seed(&keys, &values, total)?;
            Ok(kv)
        };
        assert!(make()?.verify_tree_supported(), "{format:?} éligible");
        // Feuille sur la dernière ligne physique compris ([0, 2, 8], [0..9]).
        let paths: [&[usize]; 6] = [
            &[0],
            &[0, 2, 3],
            &[0, 1, 2, 3, 4, 5, 6, 7, 8],
            &[0, 5],
            &[0, 2, 8],
            &[0, 1, 4, 7, 8],
        ];
        for path in paths {
            let mut kv = make()?;
            kv.set_floor_for_test(context);
            kv.compact_tree_rows(context, path)?;
            assert_eq!(
                kv.len(),
                context + path.len(),
                "{format:?} {path:?}: longueur"
            );
            let (got_keys, got_values) = kv.read_rows(0, context + path.len())?;
            let source = (0..context).chain(path.iter().map(|&row| context + row));
            for (row, src) in source.enumerate() {
                let expect = |data: &[f32]| data[src * kv_dim..(src + 1) * kv_dim].to_vec();
                let range = row * kv_dim..(row + 1) * kv_dim;
                assert_eq!(
                    got_keys[range.clone()],
                    expect(&keys_ref)[..],
                    "{format:?} {path:?} K ligne {row}"
                );
                assert_eq!(
                    got_values[range],
                    expect(&values_ref)[..],
                    "{format:?} {path:?} V ligne {row}"
                );
            }
        }
        // Refus : chemin non croissant, hors des lignes valides, sous le préfixe
        // adopté, ne partant pas de la racine.
        for (first, path) in [
            (context, &[0usize, 3, 2][..]),
            (context, &[0, 9][..]),
            (context, &[1, 2][..]),
            (context - 1, &[0, 1][..]),
        ] {
            let mut kv = make()?;
            kv.set_floor_for_test(context);
            assert!(
                kv.compact_tree_rows(first, path).is_err(),
                "{format:?} {first} {path:?}"
            );
            assert_eq!(kv.len(), total, "refus sans effet");
        }
    }
    let mut quantized =
        state.full_attention_format_for_test(total, Q_HEADS, KV_HEADS, head_dim, KvFormat::Q8)?;
    quantized.seed(&keys, &values, total)?;
    assert!(!quantized.verify_tree_supported(), "Q8 refusé");
    assert!(quantized.compact_tree_rows(context, &[0, 2]).is_err());
    Ok(())
}

/// Attention en lignes de `rows` requêtes `q` sur le KV `keys`/`values`
/// (`len` positions), sous `tree` éventuel, hors NAX.
#[expect(
    clippy::too_many_arguments,
    reason = "cas de test : géométrie, KV, requêtes et arbre explicites"
)]
fn run_attention_rows(
    state: &DecodeResidentState,
    head_dim: usize,
    format: KvFormat,
    keys: &[f32],
    values: &[f32],
    len: usize,
    q: &[f32],
    rows: usize,
    tree: Option<Rc<VerifyTree>>,
) -> Result<Vec<f32>> {
    let mut kv = state.full_attention_format_for_test(len, Q_HEADS, KV_HEADS, head_dim, format)?;
    kv.seed(keys, values, len)?;
    let q_buf = lease_with(state, q)?;
    let out = state
        .scratch()
        .lease(rows * Q_HEADS * head_dim, GpuElement::F32)?;
    let kernels = state.rows_kernels()?;
    let command_buffer = state.queue().new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    {
        let _scope = tree.map(install_verify_tree_scope);
        kv.encode_attention_decode_rows_chunked(
            encoder,
            kernels,
            q_buf.tensor().buffer(),
            out.tensor().buffer(),
            rows,
            None,
            None,
        )?;
    }
    encoder.end_encoding();
    commit_and_wait(command_buffer)?;
    Ok(read_f32(out.tensor()))
}

/// Oracle O1 de l'arbre : la ligne d'un nœud, calculée sous l'arbre (KV du
/// préfixe puis de tous les nœuds, masque d'ancêtres), doit rendre les bits de
/// la même requête en dernière ligne d'une chaîne dont le KV ne contient que
/// le préfixe puis le chemin du nœud (ce que voit le pas AR). Diagnostic :
/// imprime les nœuds divergents.
#[test]
#[ignore = "GPU Metal requis : échoue explicitement sans device"]
fn full_attn_rows_tree_node_matches_path_chain_bits() -> Result<()> {
    let state = gpu_state()?;
    let mut report = Vec::new();
    for head_dim in [128, 256] {
        for format in [KvFormat::F32, KvFormat::Bf16] {
            for context in [5usize, 2100, 4220] {
                for parents in TREES {
                    let rows = parents.len();
                    let tree = tree(parents)?;
                    let kv_dim = KV_HEADS * head_dim;
                    let q_dim = Q_HEADS * head_dim;
                    let total = context + rows;
                    let keys: Vec<f32> = (0..total * kv_dim).map(|i| pseudo(i + 3)).collect();
                    let values: Vec<f32> = (0..total * kv_dim).map(|i| pseudo(i + 11)).collect();
                    let q: Vec<f32> = (0..rows * q_dim).map(|i| 30.0 * pseudo(7000 + i)).collect();
                    let got = run_attention_rows(
                        &state,
                        head_dim,
                        format,
                        &keys,
                        &values,
                        total,
                        &q,
                        rows,
                        Some(tree.clone()),
                    )?;
                    for node in 0..rows {
                        let mut path = vec![node];
                        while *path.last().expect("invariant: chemin non vide") != 0 {
                            let parent = parents[*path.last().expect("invariant: chemin")];
                            path.push(parent);
                        }
                        path.reverse();
                        let mut chain_keys = keys[..context * kv_dim].to_vec();
                        let mut chain_values = values[..context * kv_dim].to_vec();
                        let mut chain_q = Vec::new();
                        for &p in &path {
                            let at = (context + p) * kv_dim;
                            chain_keys.extend_from_slice(&keys[at..at + kv_dim]);
                            chain_values.extend_from_slice(&values[at..at + kv_dim]);
                            chain_q.extend_from_slice(&q[p * q_dim..(p + 1) * q_dim]);
                        }
                        let chain = run_attention_rows(
                            &state,
                            head_dim,
                            format,
                            &chain_keys,
                            &chain_values,
                            context + path.len(),
                            &chain_q,
                            path.len(),
                            None,
                        )?;
                        let last = (path.len() - 1) * q_dim;
                        let differing = got[node * q_dim..(node + 1) * q_dim]
                            .iter()
                            .zip(&chain[last..last + q_dim])
                            .filter(|(a, b)| a.to_bits() != b.to_bits())
                            .count();
                        if differing > 0 && report.len() < 12 {
                            report.push(format!(
                                "d{head_dim} {format:?} contexte {context} arbre {parents:?} nœud {node} (profondeur {}): {differing}/{q_dim}",
                                path.len() - 1
                            ));
                        }
                    }
                }
            }
        }
    }
    assert!(report.is_empty(), "arbre ≠ chaîne du chemin : {report:#?}");
    Ok(())
}
