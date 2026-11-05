//! Tests de la référence CPU `qwen4_exp`.
//!
//! N1 : modèle synthétique (`tests/fixtures/qwen4exp_mini`), converti par
//! `convert-qwen4exp` puis comparé aux sorties officielles transformers 5.20.2
//! (f32, sdpa). Les poids sont exactement représentables en q4e8 : la
//! conversion est sans perte, l'écart mesuré est celui du forward. Recette de
//! régénération : `RECIPE.json` (script `make_mini_fixture.py`, hors dépôt).

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use super::attention::{select_tokens, QsaDims, QsaWeights};
use super::math::relative_error;
use super::Qwen4ExpReference;
use crate::qwen_loader::qwen4_exp::Qwen4ExpCheckpoint;
use crate::qwen_loader::read_shard_header;
use crate::Tensor;

/// Empreintes figées de la fixture N1.
const FIXTURE_SHA256: &[(&str, &str)] = &[
    (
        "golden.safetensors",
        "25ef31ca87be856418e292ec68ad9760df194a7847140c572dd107f7f431a94c",
    ),
    (
        "source/model-00001-of-00001.safetensors",
        "cb887bbbf576a34540a3e0e9d0d9a183f06a098a559ca1239757927d47073996",
    ),
    (
        "source/config.json",
        "d7e9e183fc01c3959748e7458fd16c75e13fac7583d637efa9e279f187c0dc29",
    ),
    (
        "source/model.safetensors.index.json",
        "864262c3015b8ebbba55f7319557726f3dfeb60ea20338b90c6a6d5c10b370ab",
    ),
];

/// Tolérance par couche (norme relative), cible initiale du plan.
const LAYER_TOL: f64 = 1.0e-4;
/// Porte KL(oracle‖rust) sur les logits, moyenne et maximum par token.
const KL_TOL: f64 = 1.0e-3;

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen4exp_mini")
}

struct Golden {
    path: PathBuf,
}

impl Golden {
    fn raw(&self, name: &str) -> (Vec<usize>, Vec<u8>) {
        let header = read_shard_header(&self.path).expect("invariant: golden lisible");
        let entry = header
            .tensors
            .get(name)
            .unwrap_or_else(|| panic!("golden sans {name}"));
        let bytes = std::fs::read(&self.path).expect("invariant: golden");
        let start = header.data_start as usize + entry.data_offsets[0];
        let end = header.data_start as usize + entry.data_offsets[1];
        (entry.shape.clone(), bytes[start..end].to_vec())
    }

    fn f32(&self, name: &str) -> Tensor {
        let (shape, bytes) = self.raw(name);
        let data = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Tensor::from_vec(shape, data).expect("invariant: forme golden")
    }

    fn i64(&self, name: &str) -> Vec<i64> {
        let (_, bytes) = self.raw(name);
        bytes
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().expect("invariant: 8 octets")))
            .collect()
    }

    fn u8(&self, name: &str) -> (Vec<usize>, Vec<u8>) {
        self.raw(name)
    }
}

/// Convertit la source de la fixture et ouvre la référence.
#[cfg(feature = "devtools")]
fn mini_reference(dst: &Path) -> Qwen4ExpReference {
    use crate::devtools::convert::{convert_qwen4exp, ConvertOptions};
    convert_qwen4exp(&ConvertOptions {
        src: fixture().join("source"),
        dst: dst.to_path_buf(),
        source_revision: "fixture-qwen4exp-mini".to_string(),
        source_repo: None,
        converter_commit: None,
        only: None,
        verify: false,
        plan_only: false,
    })
    .expect("invariant: conversion de la fixture");
    Qwen4ExpReference::new(Qwen4ExpCheckpoint::open(dst).expect("invariant: checkpoint q4e8"))
}

fn assert_close(label: &str, rust: &Tensor, oracle: &Tensor, tol: f64) {
    assert_eq!(rust.shape(), oracle.shape(), "{label}: formes");
    let (rel, max) = relative_error(rust.data(), oracle.data());
    eprintln!("{label}: rel={rel:.3e} max_abs={max:.3e}");
    assert!(
        rel <= tol,
        "{label}: erreur relative {rel:.3e} > {tol:.0e} (max {max:.3e})"
    );
}

/// KL(oracle‖rust) par token via log-softmax stable : (moyenne, maximum).
fn kl_per_token(oracle: &Tensor, rust: &Tensor) -> (f64, f64) {
    let (tokens, vocab) = oracle.as_matrix().expect("invariant: logits 2D");
    let log_softmax = |row: &[f32]| -> Vec<f64> {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let sum: f64 = row.iter().map(|v| (*v as f64 - max).exp()).sum();
        row.iter().map(|v| *v as f64 - max - sum.ln()).collect()
    };
    let mut total = 0.0;
    let mut worst = 0.0_f64;
    for t in 0..tokens {
        let p = log_softmax(&oracle.data()[t * vocab..(t + 1) * vocab]);
        let q = log_softmax(&rust.data()[t * vocab..(t + 1) * vocab]);
        let kl: f64 = p.iter().zip(&q).map(|(a, b)| a.exp() * (a - b)).sum();
        total += kl;
        worst = worst.max(kl);
    }
    (total / tokens as f64, worst)
}

/// Sélection torch de la couche : tokens retenus par requête (masque de l'indexeur).
fn torch_selection(golden: &Golden, layer: usize) -> Vec<Vec<usize>> {
    let (shape, mask) = golden.u8(&format!("qsa_mask.{layer}"));
    let tokens = shape[0];
    (0..tokens)
        .map(|i| (0..tokens).filter(|&j| mask[i * tokens + j] != 0).collect())
        .collect()
}

/// Compare la sélection Rust à celle de torch : identiques, ou différentes
/// uniquement sur des blocs dont le score égale le seuil du top-k (écart
/// ≤ 1e-5), cas des ReLU nulles. Renvoie le nombre de requêtes à ex æquo.
fn check_selection_modulo_ties(
    layer: usize,
    trace: &super::QsaTrace,
    torch: &[Vec<usize>],
    keep: usize,
) -> usize {
    let blocks = |sel: &[usize]| -> BTreeSet<usize> { sel.iter().map(|j| j / 4).collect() };
    let mut ties = 0;
    for (i, (ours, theirs)) in trace.selected.iter().zip(torch).enumerate() {
        if ours == theirs {
            continue;
        }
        let scores = &trace.scores[i];
        let mut sorted = scores.clone();
        sorted.sort_by(|a, b| b.total_cmp(a));
        let threshold = sorted[keep - 1];
        let (a, b) = (blocks(ours), blocks(theirs));
        for block in a.symmetric_difference(&b) {
            let score = scores
                .get(*block)
                .unwrap_or_else(|| panic!("couche {layer} requête {i}: bloc {block} incomplet"));
            assert!(
                (score - threshold).abs() <= 1.0e-5,
                "couche {layer} requête {i}: bloc {block} score {score} ≠ seuil {threshold}"
            );
        }
        ties += 1;
    }
    ties
}

fn argmax(row: &[f32]) -> usize {
    row.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

#[test]
fn fixture_hashes_are_pinned() {
    for (file, expected) in FIXTURE_SHA256 {
        let bytes = std::fs::read(fixture().join(file)).expect("invariant: fichier de fixture");
        let got: String = Sha256::digest(&bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        assert_eq!(&got, expected, "{file}");
    }
}

#[cfg(feature = "devtools")]
#[test]
fn n1_mini_model_matches_transformers_layer_by_layer() {
    let dst = tempfile::tempdir().expect("invariant: dossier temporaire");
    let reference = mini_reference(dst.path());
    let golden = Golden {
        path: fixture().join("golden.safetensors"),
    };
    let ids: Vec<usize> = golden
        .i64("input_ids")
        .iter()
        .map(|&t| t as usize)
        .collect();
    let layers = reference.checkpoint().config.layer_kinds.len();

    // Embedding u8 exact : identique bit à bit à celui de torch.
    let embed = reference.embed(&ids).expect("invariant: embedding");
    let mut torch_input = embed.clone();
    let mut chained = embed;
    let mut tie_queries = 0;
    for layer in 0..layers {
        // Ablation locale : chaque couche part de l'entrée torch de la précédente.
        let own = reference
            .layer(layer, &torch_input, &ids)
            .expect("invariant: couche");
        let k = reference.checkpoint().config.top_k;
        let torch_routing = golden.i64(&format!("routing.{layer}"));
        for (t, picks) in own.routing.iter().enumerate() {
            let mut expected: Vec<usize> = torch_routing[t * k..(t + 1) * k]
                .iter()
                .map(|&e| e as usize)
                .collect();
            expected.sort_unstable();
            assert_eq!(picks, &expected, "routage couche {layer} token {t}");
        }
        let forced = own.qsa.as_ref().map(|trace| {
            let torch = torch_selection(&golden, layer);
            tie_queries +=
                check_selection_modulo_ties(layer, trace, &torch, reference.budget_blocks());
            torch
        });
        // Sorties comparées avec la sélection torch imposée (identique hors ex æquo).
        let local = match &forced {
            Some(sel) => reference
                .layer_with_selection(layer, &torch_input, &ids, Some(sel))
                .expect("invariant: couche imposée"),
            None => own,
        };
        chained = reference
            .layer_with_selection(layer, &chained, &ids, forced.as_deref())
            .expect("invariant: couche chaînée")
            .hidden;
        if layer + 1 < layers {
            let oracle = golden.f32(&format!("layer.{layer}"));
            assert_close(
                &format!("couche {layer} (locale)"),
                &local.hidden,
                &oracle,
                LAYER_TOL,
            );
            assert_close(
                &format!("couche {layer} (chaînée)"),
                &chained,
                &oracle,
                LAYER_TOL,
            );
            torch_input = oracle;
        } else {
            let final_local = reference
                .final_mixer(&local.hidden)
                .expect("invariant: mixer");
            assert_close(
                "mixer final (local)",
                &final_local,
                &golden.f32("final_mixer"),
                LAYER_TOL,
            );
        }
    }
    eprintln!("requêtes QSA à ex æquo (sélection torch imposée) : {tie_queries}");
    let final_hidden = reference.final_mixer(&chained).expect("invariant: mixer");
    assert_close(
        "mixer final (chaîné)",
        &final_hidden,
        &golden.f32("final_mixer"),
        LAYER_TOL,
    );
    let logits = reference.logits(&final_hidden).expect("invariant: logits");
    let oracle = golden.f32("logits");
    let (mean, worst) = kl_per_token(&oracle, &logits);
    eprintln!("logits: KL moyen {mean:.3e}, max {worst:.3e}");
    assert!(
        mean <= KL_TOL && worst <= KL_TOL,
        "KL {mean:.3e}/{worst:.3e}"
    );
    let vocab = logits.as_matrix().expect("invariant: 2D").1;
    for t in 0..ids.len() {
        assert_eq!(
            argmax(&logits.data()[t * vocab..(t + 1) * vocab]),
            argmax(&oracle.data()[t * vocab..(t + 1) * vocab]),
            "top-1 token {t}"
        );
    }
}

fn qsa_fixture(tokens: usize, budget: usize) -> (Tensor, QsaWeights, QsaDims) {
    let hidden = 8;
    let ihd = 4;
    let value = |i: usize| ((i as f32) * 0.731).sin();
    let mat = |rows: usize, cols: usize, seed: usize| {
        Tensor::from_vec(
            vec![rows, cols],
            (0..rows * cols).map(|i| value(i + seed)).collect(),
        )
        .expect("invariant: forme")
    };
    let w = QsaWeights {
        q_proj: mat(2 * 2 * 4, hidden, 1),
        k_proj: mat(4, hidden, 2),
        v_proj: mat(4, hidden, 3),
        o_proj: mat(hidden, 8, 4),
        q_norm: vec![0.0; 4],
        k_norm: vec![0.0; 4],
        index_qk: mat(2 * ihd, hidden, 5),
        index_q_norm: vec![0.0; ihd],
        index_k_norm: vec![0.0; ihd],
    };
    let d = QsaDims {
        heads: 2,
        kv_heads: 1,
        head_dim: 4,
        rotary_dim: 2,
        rope_theta: 10_000.0,
        index_heads: 1,
        index_head_dim: ihd,
        budget,
        compress: 4,
        eps: 1.0e-6,
    };
    (mat(tokens, hidden, 9), w, d)
}

#[test]
fn qsa_selection_size_tail_and_excludable_current_block() {
    let budget = 8;
    let tokens = 40;
    let (x, w, d) = qsa_fixture(tokens, budget);
    let freqs = super::math::inv_freq(d.rotary_dim, d.rope_theta);
    let selected = select_tokens(&x, &w, d, &freqs).expect("invariant: sélection");
    let mut excluded_current = false;
    for (i, sel) in selected.iter().enumerate() {
        let visible = i + 1;
        let expected = 4 * (visible / 4).min(budget / 4) + visible % 4;
        assert_eq!(sel.len(), expected, "taille requête {i}");
        // Queue incomplète toujours présente, tokens causaux et sans doublon.
        for j in (visible / 4) * 4..visible {
            assert!(sel.contains(&j), "queue {j} absente pour {i}");
        }
        assert!(sel.iter().all(|&j| j <= i));
        assert_eq!(sel.iter().collect::<BTreeSet<_>>().len(), sel.len());
        if visible % 4 == 0 && !sel.contains(&i) {
            excluded_current = true;
        }
    }
    assert!(
        excluded_current,
        "le bloc complet courant doit pouvoir être exclu sur cette fixture"
    );
}
