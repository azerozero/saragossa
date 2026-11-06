//! Tests de la référence CPU `qwen4_exp`.
//!
//! N1 : modèle synthétique (`tests/fixtures/qwen4exp_mini`), converti par
//! `convert-qwen4exp` puis comparé aux sorties officielles transformers 5.20.2
//! (f32, sdpa). Les poids sont exactement représentables en q4e8 : la
//! conversion est sans perte, l'écart mesuré est celui du forward.
//! Régénération : `tools/flashnext/make_mini_fixture.py` (voir `RECIPE.json`).
//!
//! N2 (ignoré) : vrais poids q4e8 contre les couches officielles
//! (`tools/flashnext/oracle_dump.py`).
//!
//! Quasi-égalités : un échange d'expert ou de bloc QSA n'est admis que s'il est
//! justifié par les scores **de l'oracle** (écart au seuil ≤ 1e-5) ; les scores
//! Rust de l'indexeur doivent en outre égaler ceux de torch.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use super::attention::{select_tokens, QsaDims, QsaWeights};
use super::math::relative_error;
use super::{QsaTrace, Qwen4ExpReference};
use crate::qwen_loader::qwen4_exp::{ngram_table_path, Qwen4ExpCheckpoint};
use crate::qwen_loader::read_shard_header;
use crate::Tensor;

/// Empreintes figées de la fixture N1.
const FIXTURE_SHA256: &[(&str, &str)] = &[
    (
        "golden.safetensors",
        "a759014da5315db43e28135f67f187e610e230a77a409dbc469670e060ad3f7c",
    ),
    (
        "source/model-00001-of-00001.safetensors",
        "cce4f5b94dfaf1dc3465fd5f042fd4e141104651bcccc6e178700bd0f2e97d09",
    ),
    (
        "source/config.json",
        "5b7ff4d9ec7c46c5bc6d8c6dfcb417308d308593cb5e9c85c9d66a838a6b56c2",
    ),
    (
        "source/model.safetensors.index.json",
        "69b7b126fbc4362a1b7ea4e1f8b7378187a566282914282e71473535e1bbbb60",
    ),
];

/// Tolérance par couche (norme relative), sur l'entrée torch de la couche précédente.
const LAYER_TOL: f64 = 1.0e-4;
/// Garde-fou des passes chaînées natives (dérive mesurée, non gatée).
const CHAINED_GUARD: f64 = 1.0e-2;
/// Porte KL(oracle‖rust) sur les logits, moyenne et maximum par token.
const KL_TOL: f64 = 1.0e-3;
/// Écart au seuil de l'oracle sous lequel un échange est une quasi-égalité.
const TIE_TOL: f32 = 1.0e-5;
/// Accord des scores Rust de l'indexeur avec ceux de torch.
const SCORE_TOL: f32 = 1.0e-4;

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
        // Lecture de la seule plage du tenseur (goldens N2 de plusieurs Go).
        let file = std::fs::File::open(&self.path).expect("invariant: golden");
        let mut bytes = vec![0_u8; entry.data_offsets[1] - entry.data_offsets[0]];
        std::os::unix::fs::FileExt::read_exact_at(
            &file,
            &mut bytes,
            header.data_start + entry.data_offsets[0] as u64,
        )
        .expect("invariant: lecture golden");
        (entry.shape.clone(), bytes)
    }

    fn has(&self, name: &str) -> bool {
        read_shard_header(&self.path)
            .expect("invariant: golden lisible")
            .tensors
            .contains_key(name)
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

    fn mask_selection(&self, name: &str) -> Vec<Vec<usize>> {
        let (shape, mask) = self.raw(name);
        let tokens = shape[0];
        (0..tokens)
            .map(|i| (0..tokens).filter(|&j| mask[i * tokens + j] != 0).collect())
            .collect()
    }

    /// Scores torch de l'indexeur par requête (aplatis + offsets).
    fn qsa_scores(&self, layer: usize) -> Vec<Vec<f32>> {
        let flat = self.f32(&format!("qsa_scores.{layer}"));
        let offsets = self.i64(&format!("qsa_offsets.{layer}"));
        offsets
            .windows(2)
            .map(|w| flat.data()[w[0] as usize..w[1] as usize].to_vec())
            .collect()
    }

    /// Sorties du routeur torch : (experts retenus triés, probabilités) par token.
    fn routing(&self, layer: usize, k: usize) -> (Vec<Vec<usize>>, Vec<Vec<f32>>) {
        let ids = self.i64(&format!("routing.{layer}"));
        let logits = self.f32(&format!("router_logits.{layer}"));
        let experts = logits.as_matrix().expect("invariant: 2D").1;
        let sets = ids
            .chunks_exact(k)
            .map(|row| {
                let mut row: Vec<usize> = row.iter().map(|&e| e as usize).collect();
                row.sort_unstable();
                row
            })
            .collect();
        let probs = logits
            .data()
            .chunks_exact(experts)
            .map(|row| {
                let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
                let sum: f64 = row.iter().map(|v| (*v as f64 - max).exp()).sum();
                row.iter()
                    .map(|v| ((*v as f64 - max).exp() / sum) as f32)
                    .collect()
            })
            .collect();
        (sets, probs)
    }
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

fn argmax(row: &[f32]) -> usize {
    row.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// k-ième plus grande valeur, `None` s'il y en a au plus `k` (pas de seuil).
fn kth_largest(values: &[f32], k: usize) -> Option<f32> {
    if values.len() <= k || k == 0 {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| b.total_cmp(a));
    Some(sorted[k - 1])
}

/// Justifie la sélection QSA Rust contre l'oracle :
/// 1. scores Rust de l'indexeur ≈ scores torch (`SCORE_TOL`) pour toute requête ;
/// 2. une sélection différente n'échange que des blocs dont le score **torch**
///    est au seuil torch du top-k (`TIE_TOL`).
///
/// Renvoie le nombre de requêtes à ex æquo, ou la première violation.
fn justify_qsa(
    rust: &QsaTrace,
    torch_selection: &[Vec<usize>],
    torch_scores: &[Vec<f32>],
    keep: usize,
    compress: usize,
) -> Result<usize, String> {
    let blocks = |sel: &[usize]| -> BTreeSet<usize> { sel.iter().map(|j| j / compress).collect() };
    let lengths = [
        rust.selected.len(),
        rust.scores.len(),
        torch_selection.len(),
        torch_scores.len(),
    ];
    if lengths.iter().any(|&n| n != lengths[0]) {
        return Err(format!("traces de longueurs différentes {lengths:?}"));
    }
    let mut ties = 0;
    for (i, ((ours, theirs), (rs, ts))) in rust
        .selected
        .iter()
        .zip(torch_selection)
        .zip(rust.scores.iter().zip(torch_scores))
        .enumerate()
    {
        if rs.len() != ts.len() {
            return Err(format!(
                "requête {i}: {} scores Rust, {} torch",
                rs.len(),
                ts.len()
            ));
        }
        for (b, (r, t)) in rs.iter().zip(ts).enumerate() {
            let within = (r - t).abs() <= SCORE_TOL * (1.0 + t.abs());
            if !within || !r.is_finite() || !t.is_finite() {
                return Err(format!("requête {i} bloc {b}: score Rust {r} ≠ torch {t}"));
            }
        }
        if ours == theirs {
            continue;
        }
        let threshold = kth_largest(ts, keep).ok_or_else(|| {
            format!(
                "requête {i}: sélection différente sans top-k ({} blocs)",
                ts.len()
            )
        })?;
        let (a, b) = (blocks(ours), blocks(theirs));
        for block in a.symmetric_difference(&b) {
            let score = ts
                .get(*block)
                .ok_or_else(|| format!("requête {i}: bloc {block} hors blocs complets"))?;
            // Même règle que le N2 résident : écart au seuil relatif à max(|seuil|, 1).
            if (score - threshold).abs() > TIE_TOL * threshold.abs().max(1.0) {
                return Err(format!(
                    "requête {i}: bloc {block} échangé, score torch {score} ≠ seuil {threshold}"
                ));
            }
        }
        ties += 1;
    }
    Ok(ties)
}

/// Justifie le routage Rust contre l'oracle : un expert échangé doit avoir une
/// probabilité **torch** au seuil torch du top-k. Renvoie les échanges
/// `(token, experts de la différence symétrique)`.
fn justify_routing(
    rust: &[Vec<usize>],
    torch_sets: &[Vec<usize>],
    torch_probs: &[Vec<f32>],
    k: usize,
) -> Result<Vec<(usize, Vec<usize>)>, String> {
    let mut swaps = Vec::new();
    for (t, (ours, theirs)) in rust.iter().zip(torch_sets).enumerate() {
        if ours == theirs {
            continue;
        }
        let probs = &torch_probs[t];
        let threshold = kth_largest(probs, k)
            .ok_or_else(|| format!("token {t}: routage différent sans seuil"))?;
        let a: BTreeSet<usize> = ours.iter().copied().collect();
        let b: BTreeSet<usize> = theirs.iter().copied().collect();
        let swapped: Vec<usize> = a.symmetric_difference(&b).copied().collect();
        for e in &swapped {
            let near = (probs[*e] - threshold).abs() <= TIE_TOL;
            if !near || !probs[*e].is_finite() {
                return Err(format!(
                    "token {t}: expert {e} échangé, probabilité torch {} ≠ seuil {threshold}",
                    probs[*e]
                ));
            }
        }
        swaps.push((t, swapped));
    }
    Ok(swaps)
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
    let k = reference.checkpoint().config.top_k;
    let (keep, compress) = (reference.budget_blocks(), reference.compress());

    // Embedding u8 exact : identique bit à bit à celui de torch.
    let embed = reference.embed(&ids).expect("invariant: embedding");
    let mut torch_input = embed.clone();
    let mut forced_chain = embed.clone();
    let mut native_chain = embed;
    for layer in 0..layers {
        let own = reference
            .layer(layer, &torch_input, &ids)
            .expect("invariant: couche");
        let (torch_sets, torch_probs) = golden.routing(layer, k);
        let swaps = justify_routing(&own.routing, &torch_sets, &torch_probs, k)
            .unwrap_or_else(|e| panic!("routage couche {layer}: {e}"));
        assert!(
            swaps.is_empty(),
            "N1 sans quasi-égalité de routage: {swaps:?}"
        );
        let forced = own.qsa.as_ref().map(|trace| {
            let torch = golden.mask_selection(&format!("qsa_mask.{layer}"));
            let ties = justify_qsa(trace, &torch, &golden.qsa_scores(layer), keep, compress)
                .unwrap_or_else(|e| panic!("QSA couche {layer}: {e}"));
            eprintln!("couche {layer}: {ties} requêtes QSA à ex æquo justifiées par l'oracle");
            torch
        });
        // Ablation : sorties comparées avec la sélection torch imposée.
        let local = match &forced {
            Some(sel) => reference
                .layer_with_selection(layer, &torch_input, &ids, Some(sel))
                .expect("invariant: couche imposée"),
            None => own,
        };
        let oracle = golden.f32(&format!("layer.{layer}"));
        assert_close(
            &format!("couche {layer} (locale)"),
            &local.hidden,
            &oracle,
            LAYER_TOL,
        );
        forced_chain = reference
            .layer_with_selection(layer, &forced_chain, &ids, forced.as_deref())
            .expect("invariant: couche chaînée imposée")
            .hidden;
        assert_close(
            &format!("couche {layer} (chaînée, sélection imposée)"),
            &forced_chain,
            &oracle,
            LAYER_TOL,
        );
        // Passe chaînée native : sélection propre de l'indexeur, mesurée.
        let native = reference
            .layer(layer, &native_chain, &ids)
            .expect("invariant: couche chaînée native");
        if let (Some(trace), Some(torch)) = (&native.qsa, &forced) {
            let differing = trace
                .selected
                .iter()
                .zip(torch)
                .filter(|(a, b)| a != b)
                .count();
            eprintln!("couche {layer}: passe native, {differing} sélections QSA ≠ torch");
        }
        native_chain = native.hidden;
        let (rel, _) = relative_error(native_chain.data(), oracle.data());
        eprintln!("couche {layer} (chaînée native): rel={rel:.3e}");
        assert!(
            rel <= CHAINED_GUARD,
            "passe native couche {layer}: {rel:.3e}"
        );
        torch_input = oracle;
    }
    let final_hidden = reference
        .final_mixer(&forced_chain)
        .expect("invariant: mixer");
    assert_close(
        "mixer final",
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
    let native_logits = reference
        .logits(
            &reference
                .final_mixer(&native_chain)
                .expect("invariant: mixer"),
        )
        .expect("invariant: logits natifs");
    let (mean, worst) = kl_per_token(&oracle, &native_logits);
    eprintln!("logits (passe native): KL moyen {mean:.3e}, max {worst:.3e}");
}

#[cfg(feature = "devtools")]
#[test]
fn n1_qsa_with_clear_margin_matches_exactly() {
    let dst = tempfile::tempdir().expect("invariant: dossier temporaire");
    let reference = mini_reference(dst.path());
    let golden = Golden {
        path: fixture().join("golden.safetensors"),
    };
    let last = reference.checkpoint().config.layer_kinds.len() - 1;
    let x = golden.f32("qsa_clear.x");
    let (out, trace) = reference
        .attention_layer(last, &x, None)
        .expect("invariant: attention");
    // Marge nette (> 1e-3) au seuil : la sélection doit être identique, sans
    // recours aux quasi-égalités.
    assert_eq!(trace.selected, golden.mask_selection("qsa_clear.mask"));
    assert_close(
        "QSA à marge nette",
        &out,
        &golden.f32("qsa_clear.out"),
        LAYER_TOL,
    );
}

#[cfg(feature = "devtools")]
#[test]
fn qsa_justification_rejects_all_zero_rust_scores() {
    let dst = tempfile::tempdir().expect("invariant: dossier temporaire");
    let reference = mini_reference(dst.path());
    let golden = Golden {
        path: fixture().join("golden.safetensors"),
    };
    let last = reference.checkpoint().config.layer_kinds.len() - 1;
    let torch = golden.mask_selection(&format!("qsa_mask.{last}"));
    let scores = golden.qsa_scores(last);
    let (keep, compress) = (reference.budget_blocks(), reference.compress());
    // Indexeur défaillant simulé : scores nuls, sélection « par indice ».
    let zeros: Vec<Vec<f32>> = scores.iter().map(|row| vec![0.0; row.len()]).collect();
    let selected = super::attention::select_from_scores(&zeros, selection_dims(keep, compress));
    let broken = QsaTrace {
        selected,
        scores: zeros,
    };
    assert!(
        justify_qsa(&broken, &torch, &scores, keep, compress).is_err(),
        "des scores Rust tous nuls doivent être rejetés"
    );
}

/// Dimensions QSA minimales pour `select_from_scores` (seuls budget et compress servent).
#[cfg(feature = "devtools")]
fn selection_dims(keep: usize, compress: usize) -> QsaDims {
    QsaDims {
        heads: 1,
        kv_heads: 1,
        head_dim: 2,
        rotary_dim: 2,
        rope_theta: 1.0,
        index_heads: 1,
        index_head_dim: 2,
        budget: keep * compress,
        compress,
        eps: 1.0e-6,
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
    let freqs = crate::rope::inv_freq(d.rotary_dim, d.rope_theta as f32);
    let selected = select_tokens(&x, &w, d, &freqs).expect("invariant: sélection");
    let c = d.compress;
    let mut excluded_current = false;
    for (i, sel) in selected.iter().enumerate() {
        let visible = i + 1;
        let expected = c * (visible / c).min(budget / c) + visible % c;
        assert_eq!(sel.len(), expected, "taille requête {i}");
        // Queue incomplète toujours présente, tokens causaux et sans doublon.
        for j in (visible / c) * c..visible {
            assert!(sel.contains(&j), "queue {j} absente pour {i}");
        }
        assert!(sel.iter().all(|&j| j <= i));
        assert_eq!(sel.iter().collect::<BTreeSet<_>>().len(), sel.len());
        if visible % c == 0 && !sel.contains(&i) {
            excluded_current = true;
        }
    }
    assert!(
        excluded_current,
        "le bloc complet courant doit pouvoir être exclu sur cette fixture"
    );
}

/// N2 : chemins des poids convertis et des goldens d'`oracle_dump.py`.
fn n2_paths() -> (PathBuf, PathBuf) {
    let dir = std::env::var_os("SARAGOSSA_QWEN4EXP_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("SARAGOSSA_QWEN4EXP_DIR requis"));
    let golden = std::env::var_os("SARAGOSSA_QWEN4EXP_GOLDEN")
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("SARAGOSSA_QWEN4EXP_GOLDEN requis"));
    (dir, golden)
}

fn sha256_file(path: &Path) -> String {
    use std::io::Read;
    let mut file = std::fs::File::open(path).expect("invariant: fichier");
    let mut sha = Sha256::new();
    let mut buf = vec![0_u8; 1 << 24];
    loop {
        let n = file.read(&mut buf).expect("invariant: lecture");
        if n == 0 {
            break;
        }
        sha.update(&buf[..n]);
    }
    sha.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

/// Fichiers de poids réellement ouverts : shards indexés, MTP, `ngram/meta.json`
/// et la table qu'il désigne (celle qu'ouvre le loader, pas un nom supposé).
fn opened_weight_files(dir: &Path) -> Result<Vec<String>, String> {
    let index: serde_json::Value = serde_json::from_slice(
        &std::fs::read(dir.join("model.safetensors.index.json")).expect("invariant: index"),
    )
    .expect("invariant: json");
    let mut names: BTreeSet<String> = index["weight_map"]
        .as_object()
        .expect("invariant: weight_map")
        .values()
        .filter_map(|v| v.as_str().map(str::to_string))
        .collect();
    if dir.join("mtp.safetensors").exists() {
        names.insert("mtp.safetensors".to_string());
    }
    let meta: serde_json::Value = serde_json::from_slice(
        &std::fs::read(dir.join("ngram/meta.json")).expect("invariant: ngram/meta.json"),
    )
    .expect("invariant: json");
    let file = meta["file"]
        .as_str()
        .ok_or_else(|| "ngram/meta.json sans file".to_string())?;
    // Même règle que le loader : la table hachée est celle réellement ouverte.
    ngram_table_path(dir, file).map_err(|e| e.to_string())?;
    names.insert("ngram/meta.json".to_string());
    names.insert(format!("ngram/{file}"));
    Ok(names.into_iter().collect())
}

/// Provenance N2 vérifiée avant l'inférence, **sur les fichiers réels** :
/// empreinte du golden, empreintes recalculées de la config et de chaque
/// fichier de poids ouvert, comparées à l'entrée propre au golden ; versions
/// et backend.
fn check_provenance(dir: &Path, golden_dir: &Path, file: &str) -> Result<(), String> {
    let provenance: serde_json::Value = serde_json::from_slice(
        &std::fs::read(golden_dir.join("provenance.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let entry = &provenance["goldens"][file];
    if entry.is_null() {
        return Err(format!("provenance sans entrée pour {file}"));
    }
    let golden_sha = sha256_file(&golden_dir.join(file));
    if entry["sha256"].as_str() != Some(golden_sha.as_str()) {
        return Err(format!("{file}: empreinte {golden_sha} ≠ provenance"));
    }
    let config_sha = sha256_file(&dir.join("config.json"));
    if entry["config_sha256"].as_str() != Some(config_sha.as_str()) {
        return Err(format!("config.json {config_sha} ≠ config du golden"));
    }
    let names = opened_weight_files(dir)?;
    let recorded = entry["weights"]
        .as_object()
        .ok_or_else(|| "provenance sans poids".to_string())?;
    if recorded.len() != names.len() {
        return Err(format!(
            "{} fichiers de poids ouverts, {} dans la provenance",
            names.len(),
            recorded.len()
        ));
    }
    for name in &names {
        let actual = sha256_file(&dir.join(name));
        if recorded.get(name).and_then(|v| v.as_str()) != Some(actual.as_str()) {
            return Err(format!("{name}: {actual} ≠ poids du golden"));
        }
    }
    for (key, expected) in [
        ("transformers", "5.20.2"),
        ("attn_implementation", "sdpa"),
        ("dtype", "float32"),
    ] {
        if entry[key] != expected {
            return Err(format!("{key} = {}, attendu {expected}", entry[key]));
        }
    }
    Ok(())
}

#[cfg(feature = "devtools")]
#[test]
fn provenance_check_rejects_modified_weights_config_or_golden() {
    let dst = tempfile::tempdir().expect("invariant: dossier temporaire");
    let _ = mini_reference(dst.path());
    let golden = tempfile::tempdir().expect("invariant: dossier temporaire");
    std::fs::write(golden.path().join("layers.safetensors"), b"golden").expect("invariant: golden");
    let weights: serde_json::Map<String, serde_json::Value> = opened_weight_files(dst.path())
        .expect("invariant: fichiers ouverts")
        .into_iter()
        .map(|name| {
            let sha = sha256_file(&dst.path().join(&name));
            (name, sha.into())
        })
        .collect();
    let provenance = serde_json::json!({"goldens": {"layers.safetensors": {
        "sha256": sha256_file(&golden.path().join("layers.safetensors")),
        "weights": weights,
        "config_sha256": sha256_file(&dst.path().join("config.json")),
        "transformers": "5.20.2",
        "attn_implementation": "sdpa",
        "dtype": "float32",
    }}});
    std::fs::write(
        golden.path().join("provenance.json"),
        provenance.to_string(),
    )
    .expect("invariant: provenance");
    check_provenance(dst.path(), golden.path(), "layers.safetensors")
        .expect("invariant: provenance cohérente acceptée");
    assert!(check_provenance(dst.path(), golden.path(), "full.safetensors").is_err());

    // Désaccords volontaires, un à la fois, à taille de fichier égale.
    let flip = |path: &Path| {
        let mut bytes = std::fs::read(path).expect("invariant: lecture");
        let last = bytes.len() - 1;
        bytes[last] ^= 0x01;
        std::fs::write(path, bytes).expect("invariant: écriture");
    };
    for target in [
        dst.path().join("model-00001-of-00002.safetensors"),
        dst.path().join("ngram/ngram_e4m3.bin"),
        dst.path().join("config.json"),
        golden.path().join("layers.safetensors"),
    ] {
        flip(&target);
        let result = check_provenance(dst.path(), golden.path(), "layers.safetensors");
        assert!(result.is_err(), "{} modifié accepté", target.display());
        flip(&target);
    }
    check_provenance(dst.path(), golden.path(), "layers.safetensors")
        .expect("invariant: état restauré accepté");

    // meta.json redirigé vers une autre table (copie altérée) : refusé.
    let other = dst.path().join("ngram/other_e4m3.bin");
    std::fs::copy(dst.path().join("ngram/ngram_e4m3.bin"), &other).expect("invariant: copie");
    flip(&other);
    let meta_path = dst.path().join("ngram/meta.json");
    let mut meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&meta_path).expect("invariant: meta"))
            .expect("invariant: json");
    meta["file"] = "other_e4m3.bin".into();
    std::fs::write(&meta_path, meta.to_string()).expect("invariant: meta redirigé");
    assert!(
        check_provenance(dst.path(), golden.path(), "layers.safetensors").is_err(),
        "meta.json redirigé accepté"
    );

    // Chemin absolu vers la même table : refusé avant tout hachage.
    meta["file"] = other.to_str().expect("invariant: chemin utf-8").into();
    std::fs::write(&meta_path, meta.to_string()).expect("invariant: meta absolu");
    let error = check_provenance(dst.path(), golden.path(), "layers.safetensors")
        .expect_err("meta.json absolu accepté");
    assert!(error.contains("hors de ngram/"), "{error}");
}

/// Erreur relative sur les `rows` premières lignes (préfixe causal), hors
/// lignes `skip` (quasi-égalités justifiées par l'oracle).
fn prefix_error(a: &Tensor, b: &Tensor, rows: usize, skip: &[usize]) -> f64 {
    let width = a.as_matrix().expect("invariant: 2D").1;
    let keep = |data: &[f32]| -> Vec<f32> {
        (0..rows)
            .filter(|t| !skip.contains(t))
            .flat_map(|t| data[t * width..(t + 1) * width].iter().copied())
            .collect()
    };
    relative_error(&keep(a.data()), &keep(b.data())).0
}

#[test]
#[ignore = "poids réels : SARAGOSSA_QWEN4EXP_DIR et SARAGOSSA_QWEN4EXP_GOLDEN"]
fn n2_layers_0_to_3_match_transformers_on_8k_tokens() {
    let (dir, golden_dir) = n2_paths();
    check_provenance(&dir, &golden_dir, "layers.safetensors")
        .unwrap_or_else(|e| panic!("provenance N2: {e}"));
    let reference =
        Qwen4ExpReference::new(Qwen4ExpCheckpoint::open(&dir).expect("invariant: checkpoint"));
    let golden = Golden {
        path: golden_dir.join("layers.safetensors"),
    };
    let ids: Vec<usize> = golden
        .i64("input_ids")
        .iter()
        .map(|&t| t as usize)
        .collect();
    let tokens = ids.len();
    let k = reference.checkpoint().config.top_k;
    let (keep, compress) = (reference.budget_blocks(), reference.compress());
    let embed = reference.embed(&ids).expect("invariant: embedding");
    let mut torch_input = embed.clone();
    let mut native_chain = embed;
    let mut prefixes: Vec<usize> = vec![64];
    prefixes.extend(2047..=2052);
    prefixes.extend([3000, tokens]);
    for layer in 0..4 {
        let started = std::time::Instant::now();
        let own = reference
            .layer(layer, &torch_input, &ids)
            .expect("invariant: couche");
        let (torch_sets, torch_probs) = golden.routing(layer, k);
        let swaps = justify_routing(&own.routing, &torch_sets, &torch_probs, k)
            .unwrap_or_else(|e| panic!("routage couche {layer}: {e}"));
        let near_ties: Vec<usize> = swaps.iter().map(|(t, _)| *t).collect();
        eprintln!("couche {layer}: échanges de routage justifiés {swaps:?}");
        let forced = own.qsa.as_ref().map(|trace| {
            let torch = golden.mask_selection(&format!("qsa_mask.{layer}"));
            let ties = justify_qsa(trace, &torch, &golden.qsa_scores(layer), keep, compress)
                .unwrap_or_else(|e| panic!("QSA couche {layer}: {e}"));
            for (i, sel) in trace.selected.iter().enumerate() {
                let visible = i + 1;
                let expected = compress * (visible / compress).min(keep) + visible % compress;
                assert_eq!(sel.len(), expected, "taille QSA requête {i}");
            }
            for v in 2047..=2052 {
                eprintln!(
                    "frontière V={v}: {} tokens retenus, identique à torch: {}",
                    trace.selected[v - 1].len(),
                    trace.selected[v - 1] == torch[v - 1]
                );
            }
            eprintln!("couche {layer}: {ties} requêtes QSA à ex æquo justifiées par l'oracle");
            torch
        });
        // Ablation : sélection torch imposée (identique à la sélection propre hors ex æquo).
        let local = match &forced {
            Some(sel) => reference
                .layer_with_selection(layer, &torch_input, &ids, Some(sel))
                .expect("invariant: couche imposée"),
            None => own,
        };
        // Passe chaînée native : sélection propre de l'indexeur, mesurée.
        let native = reference
            .layer(layer, &native_chain, &ids)
            .expect("invariant: couche chaînée native");
        let native_flips = native
            .routing
            .iter()
            .zip(&torch_sets)
            .filter(|(a, b)| a != b)
            .count();
        let native_qsa = match (&native.qsa, &forced) {
            (Some(trace), Some(torch)) => trace
                .selected
                .iter()
                .zip(torch)
                .filter(|(a, b)| a != b)
                .count(),
            _ => 0,
        };
        native_chain = native.hidden;
        let oracle = golden.f32(&format!("layer.{layer}"));
        for &rows in &prefixes {
            let local_err = prefix_error(&local.hidden, &oracle, rows, &near_ties);
            eprintln!(
                "couche {layer} préfixe {rows}: locale {local_err:.3e} (hors quasi-égalités), \
                 brute {:.3e}, chaînée native {:.3e}",
                prefix_error(&local.hidden, &oracle, rows, &[]),
                prefix_error(&native_chain, &oracle, rows, &[])
            );
            assert!(
                local_err <= LAYER_TOL,
                "couche {layer} préfixe {rows}: {local_err:.3e} > {LAYER_TOL:.0e}"
            );
        }
        eprintln!(
            "couche {layer}: passe native, {native_flips} routages et {native_qsa} sélections QSA ≠ torch, {:.0} s",
            started.elapsed().as_secs_f64()
        );
        let (rel, _) = relative_error(native_chain.data(), oracle.data());
        assert!(
            rel <= CHAINED_GUARD,
            "passe native couche {layer}: {rel:.3e}"
        );
        torch_input = oracle;
    }
}

#[test]
#[ignore = "poids réels : SARAGOSSA_QWEN4EXP_DIR et SARAGOSSA_QWEN4EXP_GOLDEN"]
fn n2_full_model_logits_match_transformers() {
    let (dir, golden_dir) = n2_paths();
    check_provenance(&dir, &golden_dir, "full.safetensors")
        .unwrap_or_else(|e| panic!("provenance N2: {e}"));
    let reference =
        Qwen4ExpReference::new(Qwen4ExpCheckpoint::open(&dir).expect("invariant: checkpoint"));
    let golden = Golden {
        path: golden_dir.join("full.safetensors"),
    };
    for prompt in 0..4 {
        let ids: Vec<usize> = golden
            .i64(&format!("prompt.{prompt}.ids"))
            .iter()
            .map(|&t| t as usize)
            .collect();
        let logits = reference.forward_logits(&ids).expect("invariant: forward");
        let oracle = golden.f32(&format!("prompt.{prompt}.logits"));
        let (mean, worst) = kl_per_token(&oracle, &logits);
        let (rel, _) = relative_error(logits.data(), oracle.data());
        eprintln!(
            "prompt {prompt} ({} tokens): KL moyen {mean:.3e}, max {worst:.3e}, rel {rel:.3e}",
            ids.len()
        );
        assert!(
            mean <= KL_TOL && worst <= KL_TOL,
            "prompt {prompt}: KL {mean:.3e}/{worst:.3e}"
        );
        let vocab = logits.as_matrix().expect("invariant: 2D").1;
        for t in 0..ids.len() {
            assert_eq!(
                argmax(&logits.data()[t * vocab..(t + 1) * vocab]),
                argmax(&oracle.data()[t * vocab..(t + 1) * vocab]),
                "prompt {prompt} top-1 token {t}"
            );
        }
    }
}

/// Experts torch imposés aux tokens dont la marge de probabilité torch au
/// seuil du top-k est ≤ `TIE_TOL`.
fn forced_torch_routing(
    sets: &[Vec<usize>],
    probs: &[Vec<f32>],
    k: usize,
) -> Vec<Option<Vec<usize>>> {
    probs
        .iter()
        .zip(sets)
        .map(|(probs, set)| {
            let margin = kth_largest(probs, k)
                .zip(kth_largest(probs, k + 1))
                .map_or(f32::INFINITY, |(a, b)| a - b);
            (margin <= TIE_TOL).then(|| set.clone())
        })
        .collect()
}

/// N2 long ré-ancré : chaque couche QSA Rust repart du résiduel d'entrée torch
/// (`qsa_input.{couche}`, 4096 tokens), sans dérive accumulée. Experts torch
/// imposés aux quasi-égalités torch, sélection QSA torch imposée. Portes par
/// couche : routage propre égal à torch ou justifié (`justify_routing`), scores
/// de l'indexeur à `SCORE_TOL` de torch et sélection propre justifiée à
/// `TIE_TOL` (`justify_qsa`), sortie à `LAYER_TOL` aux positions figées
/// (`qsa_output.{couche}`).
#[test]
#[ignore = "poids réels : SARAGOSSA_QWEN4EXP_DIR et SARAGOSSA_QWEN4EXP_GOLDEN"]
fn n2_long_qsa_layers_anchored_on_transformers() {
    let (dir, golden_dir) = n2_paths();
    check_provenance(&dir, &golden_dir, "long.safetensors")
        .unwrap_or_else(|e| panic!("provenance N2 long : {e}"));
    let reference =
        Qwen4ExpReference::new(Qwen4ExpCheckpoint::open(&dir).expect("invariant: checkpoint"));
    let golden = Golden {
        path: golden_dir.join("long.safetensors"),
    };
    let ids: Vec<usize> = golden
        .i64("input_ids")
        .iter()
        .map(|&t| t as usize)
        .collect();
    let positions: Vec<usize> = golden
        .i64("logit_positions")
        .iter()
        .map(|&p| p as usize)
        .collect();
    let config = &reference.checkpoint().config;
    let k = config.top_k;
    let (keep, compress) = (reference.budget_blocks(), reference.compress());
    let full_layers: Vec<usize> = (0..config.layer_kinds.len())
        .filter(|&l| golden.has(&format!("qsa_mask.{l}")))
        .collect();
    assert!(!full_layers.is_empty(), "golden long sans couche QSA");
    for &layer in &full_layers {
        let input = golden.f32(&format!("qsa_input.{layer}"));
        let (torch_sets, torch_probs) = golden.routing(layer, k);
        let forced_routing = forced_torch_routing(&torch_sets, &torch_probs, k);
        let torch_selection = golden.mask_selection(&format!("qsa_mask.{layer}"));
        let out = reference
            .layer_forced(
                layer,
                &input,
                &ids,
                Some(&torch_selection),
                Some(&forced_routing),
            )
            .expect("invariant: couche ré-ancrée");
        let swaps = justify_routing(&out.routing, &torch_sets, &torch_probs, k)
            .unwrap_or_else(|e| panic!("routage couche {layer} : {e}"));
        let trace = out.qsa.as_ref().expect("invariant: trace QSA");
        let ties = justify_qsa(
            trace,
            &torch_selection,
            &golden.qsa_scores(layer),
            keep,
            compress,
        )
        .unwrap_or_else(|e| panic!("QSA couche {layer} : {e}"));
        let width = out.hidden.as_matrix().expect("invariant: 2D").1;
        let rows: Vec<f32> = positions
            .iter()
            .flat_map(|&p| {
                out.hidden.data()[p * width..(p + 1) * width]
                    .iter()
                    .copied()
            })
            .collect();
        let picked =
            Tensor::from_vec(vec![positions.len(), width], rows).expect("invariant: lignes");
        eprintln!(
            "couche {layer} ré-ancrée : {} échanges de routage justifiés, {ties} ex æquo QSA justifiés",
            swaps.len()
        );
        assert_close(
            &format!("couche {layer} ré-ancrée"),
            &picked,
            &golden.f32(&format!("qsa_output.{layer}")),
            LAYER_TOL,
        );
    }
}

/// N2 long : référence CPU sur 4096 tokens (QSA creuse au-delà de 2051) contre
/// le golden `long.safetensors` de transformers, chaîne complète des 48 couches.
///
/// Contrôle de dérive : la chaîne Rust n'est jamais ré-ancrée, l'écart s'y
/// accumule sur 48 couches × 4096 tokens. Experts torch imposés aux
/// quasi-égalités torch (`TIE_TOL`), sélection QSA torch imposée (ablation de
/// l'étape 1). Le routage propre doit égaler torch ou être justifié
/// (`justify_routing`). Les scores et sélections QSA sont relevés sans porte :
/// la porte stricte est l'oracle ré-ancré
/// ([`n2_long_qsa_layers_anchored_on_transformers`]). Portes : KL ≤ 1e-3 et
/// top-1 identique aux positions figées.
#[test]
#[ignore = "poids réels : SARAGOSSA_QWEN4EXP_DIR et SARAGOSSA_QWEN4EXP_GOLDEN"]
fn n2_long_cpu_chain_matches_transformers_on_4k_tokens() {
    let (dir, golden_dir) = n2_paths();
    check_provenance(&dir, &golden_dir, "long.safetensors")
        .unwrap_or_else(|e| panic!("provenance N2 long : {e}"));
    let reference =
        Qwen4ExpReference::new(Qwen4ExpCheckpoint::open(&dir).expect("invariant: checkpoint"));
    let golden = Golden {
        path: golden_dir.join("long.safetensors"),
    };
    let ids: Vec<usize> = golden
        .i64("input_ids")
        .iter()
        .map(|&t| t as usize)
        .collect();
    let k = reference.checkpoint().config.top_k;
    let (keep, compress) = (reference.budget_blocks(), reference.compress());
    let started = std::time::Instant::now();
    let mut h = reference.embed(&ids).expect("invariant: embedding");
    let (mut forced_total, mut swaps_total, mut ties_total) = (0, 0, 0);
    // Couches QSA sorties des tolérances par dérive (relevé, sans porte).
    let mut drifted = Vec::new();
    for layer in 0..reference.checkpoint().config.layer_kinds.len() {
        let (torch_sets, torch_probs) = golden.routing(layer, k);
        let forced_routing = forced_torch_routing(&torch_sets, &torch_probs, k);
        forced_total += forced_routing.iter().filter(|f| f.is_some()).count();
        let torch_selection = golden
            .has(&format!("qsa_mask.{layer}"))
            .then(|| golden.mask_selection(&format!("qsa_mask.{layer}")));
        let out = reference
            .layer_forced(
                layer,
                &h,
                &ids,
                torch_selection.as_deref(),
                Some(&forced_routing),
            )
            .expect("invariant: couche CPU");
        let swaps = justify_routing(&out.routing, &torch_sets, &torch_probs, k)
            .unwrap_or_else(|e| panic!("routage couche {layer} : {e}"));
        swaps_total += swaps.len();
        if let (Some(trace), Some(torch)) = (out.qsa.as_ref(), torch_selection.as_ref()) {
            let torch_scores = golden.qsa_scores(layer);
            // Dérive des scores Rust/torch, relative à 1 + |torch| comme SCORE_TOL.
            let drift = trace
                .scores
                .iter()
                .zip(&torch_scores)
                .flat_map(|(rs, ts)| {
                    rs.iter()
                        .zip(ts)
                        .map(|(r, t)| (r - t).abs() / (1.0 + t.abs()))
                })
                .fold(0.0_f32, f32::max);
            let sparse = trace.scores.iter().filter(|row| row.len() > keep).count();
            // Toutes les couches sont relevées avant l'échec, pour situer la dérive.
            match justify_qsa(trace, torch, &torch_scores, keep, compress) {
                Ok(ties) => {
                    ties_total += ties;
                    eprintln!(
                        "couche {layer} : {sparse} requêtes en régime creux, {ties} ex æquo \
                         justifiés, dérive des scores {drift:.2e}"
                    );
                }
                Err(e) => {
                    eprintln!("couche {layer} : dérive des scores {drift:.2e}, {e}");
                    drifted.push(layer);
                }
            }
        }
        h = out.hidden;
    }
    let positions: Vec<usize> = golden
        .i64("logit_positions")
        .iter()
        .map(|&p| p as usize)
        .collect();
    let width = h.as_matrix().expect("invariant: 2D").1;
    let rows: Vec<f32> = positions
        .iter()
        .flat_map(|&p| h.data()[p * width..(p + 1) * width].iter().copied())
        .collect();
    let picked = Tensor::from_vec(vec![positions.len(), width], rows).expect("invariant: lignes");
    let logits = reference
        .logits(&reference.final_mixer(&picked).expect("invariant: mixer"))
        .expect("invariant: logits");
    let oracle = golden.f32("logits");
    let (mean, worst) = kl_per_token(&oracle, &logits);
    let vocab = logits.as_matrix().expect("invariant: 2D").1;
    let top1 = (0..positions.len())
        .filter(|&i| {
            argmax(&logits.data()[i * vocab..(i + 1) * vocab])
                != argmax(&oracle.data()[i * vocab..(i + 1) * vocab])
        })
        .count();
    eprintln!(
        "N2 long : {} tokens en {:.0} s ; experts imposés {forced_total}, échanges justifiés \
         {swaps_total}, ex æquo QSA {ties_total} ; KL moyen {mean:.2e}, max {worst:.2e}, \
         top-1 différents {top1} sur {}",
        ids.len(),
        started.elapsed().as_secs_f64(),
        positions.len()
    );
    assert!(
        mean <= KL_TOL && worst <= KL_TOL,
        "KL {mean:.2e} / {worst:.2e}"
    );
    eprintln!("couches QSA hors tolérance par dérive : {drifted:?}");
    assert_eq!(top1, 0, "top-1 différent");
}

/// Tête MTP CPU (transcription de `graph_mtp`) contre `mtp_golden.safetensors`
/// (`oracle_dump.py --mtp`) : couche QSA + MoE et mixer de tête officiels de
/// transformers sur les poids `mtp.*`, entrée fc transcrite ; paires
/// `(h_t, tok_{t+1})` des 4096 tokens du golden long, `h_t` partagé.
///
/// Portes : entrée fc et sortie de couche à `LAYER_TOL` aux positions figées ;
/// routage justifié et experts torch imposés aux quasi-égalités torch ;
/// scores de l'indexeur à `SCORE_TOL`, sélection justifiée à `TIE_TOL`
/// (sélection torch imposée pour la sortie) ; logits de tête KL ≤ 1e-3 et
/// top-1 identique. L'accord du brouillon avec l'argmax du tronc (golden long)
/// est publié : seul contrôle indépendant de l'entrée et de la tête, la
/// couche venant de transformers.
#[test]
#[ignore = "poids réels : SARAGOSSA_QWEN4EXP_DIR et SARAGOSSA_QWEN4EXP_GOLDEN"]
fn n2_mtp_head_matches_transformers_blocks() {
    let (dir, golden_dir) = n2_paths();
    check_provenance(&dir, &golden_dir, "mtp_golden.safetensors")
        .unwrap_or_else(|e| panic!("provenance MTP : {e}"));
    let reference =
        Qwen4ExpReference::new(Qwen4ExpCheckpoint::open(&dir).expect("invariant: checkpoint"));
    let golden = Golden {
        path: golden_dir.join("mtp_golden.safetensors"),
    };
    let long = Golden {
        path: golden_dir.join("long.safetensors"),
    };
    let h = golden.f32("mtp_h");
    let next: Vec<usize> = golden
        .i64("mtp_next_ids")
        .iter()
        .map(|&t| t as usize)
        .collect();
    let positions: Vec<usize> = golden
        .i64("mtp_positions")
        .iter()
        .map(|&p| p as usize)
        .collect();
    let width = h.as_matrix().expect("invariant: 2D").1;
    let pick = |t: &Tensor| -> Tensor {
        let rows: Vec<f32> = positions
            .iter()
            .flat_map(|&p| t.data()[p * width..(p + 1) * width].iter().copied())
            .collect();
        Tensor::from_vec(vec![positions.len(), width], rows).expect("invariant: lignes")
    };
    let started = std::time::Instant::now();
    let x = reference
        .mtp_input(&h, &next)
        .expect("invariant: entrée MTP");
    assert_close("entrée MTP", &pick(&x), &golden.f32("mtp_input"), LAYER_TOL);
    let k = reference.checkpoint().config.top_k;
    let (keep, compress) = (reference.budget_blocks(), reference.compress());
    let (torch_sets, torch_probs) = golden.routing(0, k);
    let forced_routing = forced_torch_routing(&torch_sets, &torch_probs, k);
    let torch_selection = golden.mask_selection("qsa_mask.0");
    let out = reference
        .mtp_layer(&x, Some(&torch_selection), Some(&forced_routing))
        .expect("invariant: couche MTP");
    let swaps = justify_routing(&out.routing, &torch_sets, &torch_probs, k)
        .unwrap_or_else(|e| panic!("routage MTP : {e}"));
    let trace = out.qsa.as_ref().expect("invariant: trace QSA");
    let ties = justify_qsa(
        trace,
        &torch_selection,
        &golden.qsa_scores(0),
        keep,
        compress,
    )
    .unwrap_or_else(|e| panic!("QSA MTP : {e}"));
    let picked = pick(&out.hidden);
    assert_close("sortie MTP", &picked, &golden.f32("mtp_output"), LAYER_TOL);
    let logits = reference
        .logits(&reference.mtp_head(&picked).expect("invariant: mixer MTP"))
        .expect("invariant: logits MTP");
    let oracle = golden.f32("logits");
    let (mean, worst) = kl_per_token(&oracle, &logits);
    let vocab = logits.as_matrix().expect("invariant: 2D").1;
    let row = |t: &Tensor, i: usize| argmax(&t.data()[i * vocab..(i + 1) * vocab]);
    let top1 = (0..positions.len())
        .filter(|&i| row(&logits, i) != row(&oracle, i))
        .count();
    // Accord du brouillon (ligne t, prédit tok_{t+2}) avec le tronc : argmax du
    // golden long à la position t+1, et texte réel tok_{t+2}.
    let trunk_positions: Vec<usize> = long
        .i64("logit_positions")
        .iter()
        .map(|&p| p as usize)
        .collect();
    let trunk = long.f32("logits");
    let (mut with_trunk, mut trunk_rows, mut with_text, mut text_rows) = (0, 0, 0, 0);
    for (i, &t) in positions.iter().enumerate() {
        let draft = row(&logits, i);
        if let Some(j) = trunk_positions.iter().position(|&p| p == t + 1) {
            trunk_rows += 1;
            with_trunk += usize::from(draft == row(&trunk, j));
        }
        if t + 1 < next.len() {
            text_rows += 1;
            with_text += usize::from(draft == next[t + 1]);
        }
    }
    eprintln!(
        "MTP : {} paires en {:.0} s ; {} échanges de routage justifiés, {ties} ex æquo QSA ; \
         KL moyen {mean:.2e}, max {worst:.2e}, top-1 différents {top1} sur {} ; accord du \
         brouillon avec l'argmax du tronc {with_trunk}/{trunk_rows}, avec le texte \
         {with_text}/{text_rows}",
        next.len(),
        started.elapsed().as_secs_f64(),
        swaps.len(),
        positions.len()
    );
    assert!(
        mean <= KL_TOL && worst <= KL_TOL,
        "KL {mean:.2e} / {worst:.2e}"
    );
    assert_eq!(top1, 0, "top-1 MTP différent de transformers");
}
