//! Oracles CPU du DFlash échantillonné, sans modèle : une cible synthétique
//! p(· | préfixe) sur 6 tokens remplace le vérifieur, chaque ligne de l'arbre
//! est tirée à l'aléa de son indice de sortie (`rng_state_for_draw`, comme la
//! tête `Sample`), et le parcours de production (`walk_tree_tour`, `commit_token`)
//! décide. La boucle externe reproduit celle de `generate_greedy_dflash_resident_tree`.
//!
//! Preuve testée : la ligne visitée à la profondeur `d` est fixée par les tirages
//! d'indices `< n + d` (arbre construit avant le tour, descente selon les cibles
//! précédentes) ; son tirage d'indice `n + d` suit donc p(· | préfixe) et vaut,
//! à logits identiques, celui de l'AR de même graine. D'où l'identité stricte de
//! la sortie avec l'AR, arrêts compris, et la loi jointe exacte (χ²).

use super::*;
use crate::sampling::{
    rng_state_for_draw, sample_from_token_distribution, sample_token_top_k_top_p,
    token_distribution_top_k_top_p,
};

const VOCAB: usize = 6;

fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Logits synthétiques dans [−2, 2], fonction du préfixe entier.
fn logits(prefix: &[usize]) -> Vec<f32> {
    let hash = prefix
        .iter()
        .fold(0xcbf2_9ce4_8422_2325_u64, |h, &t| mix(h ^ t as u64));
    (0..VOCAB)
        .map(|v| (mix(hash ^ ((v as u64 + 1) << 48)) >> 40) as f32 / 4_194_304.0 - 2.0)
        .collect()
}

fn sample_at(prefix: &[usize], options: &GenerationOptions, draw: usize) -> usize {
    let mut sampler = DeterministicSampler::new(rng_state_for_draw(options.seed, draw));
    sample_token_top_k_top_p(
        &logits(prefix),
        options.temperature,
        options.top_p,
        options.top_k,
        &mut sampler,
    )
    .expect("invariant: logits synthétiques valides")
}

/// Callback qui arrête après `limit` tokens émis (le dernier compris).
fn stop_after(limit: usize) -> impl FnMut(usize) -> bool {
    let mut seen = 0;
    move |_| {
        seen += 1;
        seen < limit
    }
}

/// AR échantillonné de référence : même boucle que l'AR résident (un tirage par
/// token, le premier au prefill, stop token exclu, stop sequence incluse).
fn ar(
    prompt: &[usize],
    max_new: usize,
    options: &GenerationOptions,
    callback: usize,
) -> Vec<usize> {
    let mut on_token = stop_after(callback);
    let mut generated = Vec::new();
    let mut token = sample_at(prompt, options, 0);
    for step in 0..max_new {
        if options.stop_token_ids.contains(&token) {
            break;
        }
        if !commit_token(&mut generated, token, options, &mut on_token) {
            break;
        }
        if step + 1 < max_new {
            let prefix = [prompt, generated.as_slice()].concat();
            token = sample_at(&prefix, options, generated.len());
        }
    }
    generated
}

#[derive(Default)]
struct Coverage {
    tours: usize,
    root_rejects: usize,
    deep_accepts: usize,
    leaf_bonus: usize,
    later_sibling: usize,
}

/// Arbre du tour, indépendant du flux d'aléas : candidats = top-3 de la cible
/// le long de sa chaîne argmax, perturbés par `tree_seed` pour rater parfois.
fn tour_tree(
    prefix: &[usize],
    anchor: usize,
    budget: usize,
    draft_limit: usize,
    max_depth: usize,
    tree_seed: u64,
) -> DFlashTree {
    let mut chain = prefix.to_vec();
    let mut candidates = Vec::new();
    for position in 0..draft_limit {
        let row = logits(&chain);
        let mut order: Vec<usize> = (0..VOCAB).collect();
        let noise = |v: usize| (mix(tree_seed ^ ((position * VOCAB + v) as u64)) >> 62) as f32;
        order.sort_by(|&a, &b| (row[b] + noise(b)).total_cmp(&(row[a] + noise(a))));
        chain.push(order[0]);
        order.truncate(3);
        candidates.push(order);
    }
    DFlashTree::best_first(
        anchor,
        &candidates,
        budget,
        max_depth,
        |position, previous| {
            Ok((0..3)
                .map(|c| {
                    let key = tree_seed ^ ((position as u64) << 32) ^ ((previous as u64) << 8) ^ c;
                    3.0 - c as f32 + (mix(key) >> 40) as f32 / 16_777_216.0
                })
                .collect())
        },
    )
    .expect("invariant: arbre synthétique valide")
}

/// Boucle DFlash en arbre sur la cible synthétique (budget 0 : chaîne).
fn dflash(
    prompt: &[usize],
    max_new: usize,
    options: &GenerationOptions,
    callback: usize,
    budget: usize,
    coverage: &mut Coverage,
) -> Vec<usize> {
    let draft_limit = 4.min(max_new);
    let mut on_token = stop_after(callback);
    let mut generated = Vec::new();
    let mut stats = SpeculativeStats::default();
    stats.proposed_by_position.resize(draft_limit, 0);
    stats.accepted_by_position.resize(draft_limit, 0);
    let mut segments = [std::time::Duration::ZERO; 5];
    let mut pending = sample_at(prompt, options, 0);
    while generated.len() < max_new {
        let primary = pending;
        if options.stop_token_ids.contains(&primary) {
            break;
        }
        if !commit_token(&mut generated, primary, options, &mut on_token)
            || generated.len() >= max_new
        {
            break;
        }
        let prefix = [prompt, generated.as_slice()].concat();
        let max_depth = draft_limit.min(max_new - generated.len());
        let tree_seed = mix(options.seed ^ (generated.len() as u64) << 20);
        let tree = if budget == 0 {
            let drafts: Vec<usize> = (0..max_depth)
                .scan(prefix.clone(), |chain, _| {
                    let token = crate::sampling::argmax(&logits(chain)).ok()?;
                    chain.push(token);
                    Some(token)
                })
                .collect();
            DFlashTree::chain(primary, &drafts).expect("invariant: chaîne valide")
        } else {
            tour_tree(&prefix, primary, budget, draft_limit, max_depth, tree_seed)
        };
        let n = generated.len();
        let targets: Vec<usize> = (0..tree.len())
            .map(|row| {
                let mut path = Vec::new();
                let mut node = row;
                while node != 0 {
                    path.push(tree.tokens()[node]);
                    node = tree.parents()[node];
                }
                path.reverse();
                let row_prefix = [prefix.as_slice(), &path].concat();
                sample_at(&row_prefix, options, n + tree.depths()[row])
            })
            .collect();
        let mut state = DFlashTourState {
            generated: &mut generated,
            stats: &mut stats,
            pending: &mut pending,
            segments: &mut segments,
            on_token: &mut on_token,
        };
        let Some(path) = walk_tree_tour(&tree, &targets, max_new, options, &mut state) else {
            break;
        };
        coverage.tours += 1;
        let last = *path.last().expect("invariant: chemin avec racine");
        let has_children = |node: usize| tree.parents()[node + 1..].contains(&node);
        if path.len() == 1 && has_children(0) && tree.child(0, pending).is_none() {
            coverage.root_rejects += 1;
        }
        if path.len() >= 3 {
            coverage.deep_accepts += 1;
        }
        if !has_children(last) {
            coverage.leaf_bonus += 1;
        }
        if path
            .windows(2)
            .any(|edge| (edge[0] + 1..edge[1]).any(|row| tree.parents()[row] == edge[0]))
        {
            coverage.later_sibling += 1;
        }
        if generated.len() >= max_new {
            break;
        }
    }
    generated
}

fn options(temperature: f32, top_p: f32, top_k: usize, seed: u64) -> GenerationOptions {
    GenerationOptions {
        temperature,
        top_p,
        top_k,
        seed,
        ..GenerationOptions::default()
    }
}

/// À graine fixée, la sortie DFlash (chaîne et arbres B = 1..15) est celle de
/// l'AR échantillonné, token pour token, avec stop token, stop sequence,
/// callback qui arrête en milieu de tour et limite de sortie en milieu d'arbre.
#[test]
fn sampled_tree_matches_ar_for_every_seed() {
    let prompt = [1, 4, 2];
    let mut coverage = Coverage::default();
    let configs = [
        (0.7, 1.0, 0),
        (1.3, 1.0, 0),
        (0.7, 0.9, 3),
        (1.0, 0.6, 2),
        (0.7, 1.0, 1),
    ];
    for (index, &(temperature, top_p, top_k)) in configs.iter().enumerate() {
        for seed in 0..400_u64 {
            let mut options = options(temperature, top_p, top_k, seed);
            // Arrêts variés selon la graine : stop token, stop sequence, callback.
            if seed % 3 == 1 {
                options.stop_token_ids = vec![5];
            }
            if seed % 4 == 2 {
                options.stop_sequences = vec![vec![2, 3]];
            }
            let callback = if seed % 5 == 3 {
                4 + seed as usize % 6
            } else {
                usize::MAX
            };
            let max_new = [12, 3, 7][seed as usize % 3];
            let expected = ar(&prompt, max_new, &options, callback);
            for budget in [0, 1, 4, 8, 15] {
                let got = dflash(&prompt, max_new, &options, callback, budget, &mut coverage);
                assert_eq!(
                    got, expected,
                    "config {index} graine {seed} B={budget} max_new={max_new}"
                );
            }
        }
    }
    // L'oracle exerce bien tous les cas du parcours.
    assert!(coverage.tours > 1000, "tours={}", coverage.tours);
    for (name, count) in [
        ("rejet racine", coverage.root_rejects),
        ("acceptation profonde", coverage.deep_accepts),
        ("bonus de feuille", coverage.leaf_bonus),
        ("enfant non premier", coverage.later_sibling),
    ] {
        assert!(count > 50, "{name}: {count}");
    }
}

/// Probabilité exacte d'une séquence sous la cible filtrée (même convention
/// top-k/top-p que le sampler).
fn sequence_probability(prompt: &[usize], sequence: &[usize], options: &GenerationOptions) -> f64 {
    let mut prefix = prompt.to_vec();
    let mut probability = 1.0_f64;
    for &token in sequence {
        let distribution = token_distribution_top_k_top_p(
            &logits(&prefix),
            options.temperature,
            options.top_p,
            options.top_k,
        )
        .expect("invariant: logits synthétiques valides");
        probability *= distribution
            .iter()
            .find(|(candidate, _)| *candidate == token)
            .map_or(0.0, |(_, p)| f64::from(*p));
        prefix.push(token);
    }
    probability
}

/// χ² de `counts` contre `expected` (probabilités), cases d'effectif attendu
/// < 5 regroupées ; rend (χ², degrés de liberté).
fn chi_square(counts: &[usize], expected: &[f64], total: usize) -> (f64, usize) {
    let mut chi2 = 0.0;
    let mut cells = 0_usize;
    let (mut pooled_observed, mut pooled_expected) = (0.0, 0.0);
    for (&observed, &probability) in counts.iter().zip(expected) {
        let mean = probability * total as f64;
        if mean < 5.0 {
            pooled_observed += observed as f64;
            pooled_expected += mean;
            continue;
        }
        chi2 += (observed as f64 - mean).powi(2) / mean;
        cells += 1;
    }
    if pooled_expected >= 5.0 {
        chi2 += (pooled_observed - pooled_expected).powi(2) / pooled_expected;
        cells += 1;
    }
    (chi2, cells.saturating_sub(1))
}

/// Loi jointe des 3 premiers tokens de l'arbre B = 8, 100 000 graines, contre la
/// loi exacte : χ² sous le seuil, aucune sortie de probabilité nulle.
#[test]
fn sampled_tree_joint_law_matches_target() {
    let prompt = [3, 0];
    let samples = 100_000;
    for (temperature, top_p, top_k, threshold) in [(1.0, 1.0, 0, 300.0), (0.7, 0.9, 3, 0.0)] {
        let base = options(temperature, top_p, top_k, 0);
        let cells: Vec<Vec<usize>> = (0..VOCAB.pow(3))
            .map(|cell| vec![cell / 36, cell / 6 % 6, cell % 6])
            .collect();
        let expected: Vec<f64> = cells
            .iter()
            .map(|cell| sequence_probability(&prompt, cell, &base))
            .collect();
        let mut counts = vec![0_usize; cells.len()];
        let mut coverage = Coverage::default();
        for seed in 0..samples {
            let options = GenerationOptions {
                seed: mix(seed as u64),
                ..base.clone()
            };
            let got = dflash(&prompt, 3, &options, usize::MAX, 8, &mut coverage);
            let cell = got[0] * 36 + got[1] * 6 + got[2];
            assert!(expected[cell] > 0.0, "séquence impossible {got:?}");
            counts[cell] += 1;
        }
        let (chi2, dof) = chi_square(&counts, &expected, samples);
        let threshold = if threshold > 0.0 {
            threshold
        } else {
            dof as f64 + 6.0 * (2.0 * dof as f64).sqrt()
        };
        eprintln!("dflash_sampled_chi2 T={temperature} top_k={top_k} chi2={chi2:.1} ddl={dof}");
        assert!(
            chi2 < threshold,
            "χ²={chi2:.1} ≥ {threshold:.1} (ddl {dof})"
        );
    }
}

/// Contrôle négatif (puissance de l'oracle) : le rejet récursif SpecInfer
/// appliqué à des enfants déterministes, avec q uniforme sur ces enfants comme
/// s'ils avaient été tirés de q, biaise la loi : p = (0,6 ; 0,2 ; 0,1 ; …),
/// enfants {1, 2}, q(1) = 0,5 → token 1 accepté avec 0,4 au lieu de 0,2. Le
/// même χ² dépasse 1000, le tirage puis descente retenu passe.
#[test]
fn specinfer_on_deterministic_children_fails_the_oracle() {
    let target: Vec<(usize, f32)> = vec![
        (0, 0.6),
        (1, 0.2),
        (2, 0.1),
        (3, 0.05),
        (4, 0.03),
        (5, 0.02),
    ];
    let children = [1_usize, 2];
    let samples = 100_000;
    let mut biased = vec![0_usize; VOCAB];
    let mut retained = vec![0_usize; VOCAB];
    for seed in 0..samples as u64 {
        let mut sampler = DeterministicSampler::new(mix(seed));
        let mut residual = target.clone();
        let mut token = None;
        for &child in &children {
            // SpecInfer avec remise : q uniforme sur les enfants, accepté avec
            // min(1, p/q), sinon p ← norm((p − q)+).
            let q = 1.0 / children.len() as f32;
            let p = residual
                .iter()
                .find(|(t, _)| *t == child)
                .map_or(0.0, |(_, p)| *p);
            let accept = (p / q).min(1.0);
            let coin =
                sample_from_token_distribution(&[(1, accept), (0, 1.0 - accept)], &mut sampler)
                    .expect("invariant: pièce valide");
            if coin == 1 {
                token = Some(child);
                break;
            }
            for (t, prob) in &mut residual {
                if children.contains(t) {
                    *prob = (*prob - q).max(0.0);
                }
            }
            let sum: f32 = residual.iter().map(|(_, p)| *p).sum();
            for (_, prob) in &mut residual {
                *prob /= sum;
            }
        }
        let token = match token {
            Some(token) => token,
            None => sample_from_token_distribution(&residual, &mut sampler)
                .expect("invariant: résidu valide"),
        };
        biased[token] += 1;
        // Schéma retenu : tirage selon p, descente si c'est un enfant ; le token
        // émis est le tirage dans tous les cas.
        let mut sampler = DeterministicSampler::new(mix(seed));
        retained[sample_from_token_distribution(&target, &mut sampler)
            .expect("invariant: cible valide")] += 1;
    }
    let expected: Vec<f64> = target.iter().map(|(_, p)| f64::from(*p)).collect();
    let (biased_chi2, _) = chi_square(&biased, &expected, samples);
    let (retained_chi2, dof) = chi_square(&retained, &expected, samples);
    eprintln!("specinfer_negative_control chi2={biased_chi2:.0} retenu={retained_chi2:.1}");
    assert!(biased_chi2 > 1000.0, "χ² du biais {biased_chi2:.1}");
    assert!(retained_chi2 < dof as f64 + 6.0 * (2.0 * dof as f64).sqrt());
}
