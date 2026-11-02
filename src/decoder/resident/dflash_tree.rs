//! Arbre de vérification DFlash 2 (`SARAGOSSA_RUST_DFLASH_TREE=B`) : construction
//! CPU best-first sur le treillis du sélecteur et parcours d'acceptation.
//!
//! Le sélecteur ne donne pas de probabilité cible : chaque ligne du treillis
//! (`top_k` scores d'une position sachant le token précédent) est normalisée par
//! un log-softmax sur ses seuls candidats, heuristique qui suffit à classer les
//! branches. Un nœud vaut la somme des log-softmax de son chemin ; les `B`
//! meilleurs nœuds sont développés dans l'ordre (score décroissant, puis
//! position, rang du candidat et parent croissants), ce qui reproduit l'analyse
//! hors ligne des traces `SARAGOSSA_RUST_DFLASH_TREE_TRACE`. Un nœud n'est
//! développé qu'après son parent : l'ordre de sortie est topologique (parent <
//! enfant), ce que supposent les kernels (ancêtres d'une ligne avant elle).

use crate::{InferError, Result};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Lignes maximales d'un arbre : le masque d'ancêtres tient dans un `u32`.
pub(in crate::decoder) const DFLASH_TREE_MAX_ROWS: usize = 32;

/// Arbre aplati en ordre topologique : la ligne 0 est l'ancre (token déjà
/// commité, racine), chaque autre ligne un draft dont le parent la précède.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::decoder) struct DFlashTree {
    tokens: Vec<usize>,
    parents: Vec<usize>,
    depths: Vec<usize>,
    masks: Vec<u32>,
}

/// Candidat de la file best-first : enfant `candidate` (rang dans la position
/// `position`) du nœud `parent`, de score cumulé `score`.
struct Frontier {
    score: f64,
    position: usize,
    candidate: usize,
    parent: usize,
}

impl PartialEq for Frontier {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Frontier {}

impl PartialOrd for Frontier {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Frontier {
    // Tas max : meilleur score d'abord, puis position, rang et parent les plus
    // petits (départage déterministe, identique au tuple du script hors ligne).
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| other.position.cmp(&self.position))
            .then_with(|| other.candidate.cmp(&self.candidate))
            .then_with(|| other.parent.cmp(&self.parent))
    }
}

/// Log-softmax en f64 sur les scores d'une ligne du treillis.
fn log_softmax(scores: &[f32]) -> Vec<f64> {
    let max = scores
        .iter()
        .map(|&score| f64::from(score))
        .fold(f64::NEG_INFINITY, f64::max);
    let sum: f64 = scores
        .iter()
        .map(|&score| (f64::from(score) - max).exp())
        .sum();
    let log_z = max + sum.ln();
    scores
        .iter()
        .map(|&score| f64::from(score) - log_z)
        .collect()
}

impl DFlashTree {
    /// Arbre dégénéré `ancre → drafts[0] → drafts[1] → …` : mêmes lignes que la
    /// vérification en chaîne (oracle de la plomberie arbre).
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si la chaîne dépasse [`DFLASH_TREE_MAX_ROWS`] lignes.
    #[cfg(test)]
    pub(in crate::decoder) fn chain(anchor: usize, drafts: &[usize]) -> Result<Self> {
        let mut tree = Self::root(anchor);
        for &token in drafts {
            tree.push(tree.len() - 1, token)?;
        }
        Ok(tree)
    }

    /// Construit l'arbre best-first de `budget` nœuds hors ancre, de profondeur
    /// au plus `max_depth` (et au plus `candidates.len()` positions).
    /// `candidates[p]` liste les tokens candidats de la position `p` (profondeur
    /// `p + 1`) ; `edge_scores(p, précédent)` rend leurs scores sachant le token
    /// précédent (ancre pour `p = 0`), dans le même ordre. Seuls les nœuds
    /// développés demandent leur ligne du treillis.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le budget dépasse [`DFLASH_TREE_MAX_ROWS`] − 1, si
    /// une ligne de scores n'a pas la taille de ses candidats, ou si
    /// `edge_scores` échoue.
    pub(in crate::decoder) fn best_first(
        anchor: usize,
        candidates: &[Vec<usize>],
        budget: usize,
        max_depth: usize,
        mut edge_scores: impl FnMut(usize, usize) -> Result<Vec<f32>>,
    ) -> Result<Self> {
        if budget >= DFLASH_TREE_MAX_ROWS {
            return Err(InferError::Config(format!(
                "arbre DFlash: budget {budget} ≥ {DFLASH_TREE_MAX_ROWS} lignes"
            )));
        }
        let positions = candidates.len().min(max_depth);
        let mut tree = Self::root(anchor);
        let mut heap = BinaryHeap::new();
        let mut expand = |heap: &mut BinaryHeap<Frontier>,
                          node: usize,
                          token: usize,
                          depth: usize,
                          base: f64|
         -> Result<()> {
            if depth >= positions {
                return Ok(());
            }
            let row = edge_scores(depth, token)?;
            if row.len() != candidates[depth].len() {
                return Err(InferError::Dimension(format!(
                    "arbre DFlash: {} scores pour {} candidats (position {depth})",
                    row.len(),
                    candidates[depth].len()
                )));
            }
            for (candidate, log_prob) in log_softmax(&row).into_iter().enumerate() {
                heap.push(Frontier {
                    score: base + log_prob,
                    position: depth,
                    candidate,
                    parent: node,
                });
            }
            Ok(())
        };
        expand(&mut heap, 0, anchor, 0, 0.0)?;
        while tree.len() <= budget {
            let Some(next) = heap.pop() else {
                break;
            };
            let token = candidates[next.position][next.candidate];
            let node = tree.push(next.parent, token)?;
            expand(&mut heap, node, token, next.position + 1, next.score)?;
        }
        Ok(tree)
    }

    fn root(anchor: usize) -> Self {
        Self {
            tokens: vec![anchor],
            parents: vec![0],
            depths: vec![0],
            masks: vec![1],
        }
    }

    /// Ajoute `token` sous `parent` et rend l'indice de la nouvelle ligne.
    fn push(&mut self, parent: usize, token: usize) -> Result<usize> {
        let row = self.tokens.len();
        if row >= DFLASH_TREE_MAX_ROWS || parent >= row {
            return Err(InferError::Dimension(format!(
                "arbre DFlash: ligne {row} sous le parent {parent} (max {DFLASH_TREE_MAX_ROWS})"
            )));
        }
        self.tokens.push(token);
        self.parents.push(parent);
        self.depths.push(self.depths[parent] + 1);
        self.masks.push(self.masks[parent] | (1u32 << row));
        Ok(row)
    }

    /// Lignes de l'arbre, ancre comprise.
    pub(in crate::decoder) fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Tokens des lignes (ligne 0 = ancre).
    pub(in crate::decoder) fn tokens(&self) -> &[usize] {
        &self.tokens
    }

    /// Parent de chaque ligne (0 pour la racine, sans signification).
    pub(in crate::decoder) fn parents(&self) -> &[usize] {
        &self.parents
    }

    /// Profondeur de chaque ligne (0 pour l'ancre) : décalage RoPE et taps conv.
    pub(in crate::decoder) fn depths(&self) -> &[usize] {
        &self.depths
    }

    /// Masque d'ancêtres de chaque ligne : bit `j` levé si la ligne `j` est la
    /// ligne elle-même ou l'un de ses ancêtres (lignes visibles en attention).
    pub(in crate::decoder) fn masks(&self) -> &[u32] {
        &self.masks
    }

    /// Enfant de `node` portant `token`, s'il existe (au plus un : les
    /// candidats d'une position sont distincts).
    pub(in crate::decoder) fn child(&self, node: usize, token: usize) -> Option<usize> {
        (node + 1..self.len()).find(|&row| self.parents[row] == node && self.tokens[row] == token)
    }

    /// Chemin accepté depuis la racine selon `targets` (argmax de chaque ligne) :
    /// on descend vers l'enfant dont le token égale la cible du nœud courant.
    /// Rend les lignes du chemin (racine comprise) ; la cible du dernier nœud est
    /// le token suivant. Sans arrêt ni stop token : la boucle de génération
    /// applique les siens en parcourant [`Self::child`].
    #[cfg(test)]
    pub(in crate::decoder) fn accepted_path(&self, targets: &[usize]) -> Vec<usize> {
        let mut path = vec![0];
        let mut node = 0;
        while let Some(child) = targets.get(node).and_then(|&t| self.child(node, t)) {
            path.push(child);
            node = child;
        }
        path
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Treillis synthétique déterministe : `k` candidats par position, scores
    /// pseudo-aléatoires dépendant de la position, du précédent et du candidat.
    fn lattice(
        positions: usize,
        k: usize,
        seed: u64,
    ) -> (Vec<Vec<usize>>, impl Fn(usize, usize) -> Vec<f32>) {
        let candidates = (0..positions)
            .map(|p| (0..k).map(|c| 100 * (p + 1) + c).collect())
            .collect();
        let score = move |p: usize, prev: usize| {
            (0..k)
                .map(|c| {
                    let mut x = seed ^ ((p as u64) << 40) ^ ((prev as u64) << 16) ^ c as u64;
                    x = x.wrapping_mul(0x9E37_79B9_7F4A_7C15);
                    x ^= x >> 29;
                    (x % 1000) as f32 / 100.0
                })
                .collect()
        };
        (candidates, score)
    }

    fn check_topology(tree: &DFlashTree) {
        for row in 1..tree.len() {
            let parent = tree.parents()[row];
            assert!(parent < row, "parent {parent} après la ligne {row}");
            assert_eq!(tree.depths()[row], tree.depths()[parent] + 1);
            assert_eq!(tree.masks()[row], tree.masks()[parent] | (1 << row));
        }
    }

    #[test]
    fn chain_tree_is_causal() -> Result<()> {
        let tree = DFlashTree::chain(7, &[1, 2, 3])?;
        assert_eq!(tree.tokens(), &[7, 1, 2, 3]);
        assert_eq!(tree.parents(), &[0, 0, 1, 2]);
        assert_eq!(tree.depths(), &[0, 1, 2, 3]);
        assert_eq!(tree.masks(), &[0b1, 0b11, 0b111, 0b1111]);
        assert_eq!(tree.accepted_path(&[1, 2, 9, 0]), vec![0, 1, 2]);
        assert_eq!(tree.accepted_path(&[1, 2, 3, 4]), vec![0, 1, 2, 3]);
        assert_eq!(tree.accepted_path(&[5, 2, 3, 4]), vec![0]);
        Ok(())
    }

    /// Une seule branche dominante (scores très piqués) : l'arbre best-first
    /// redevient la chaîne gloutonne du sélecteur.
    #[test]
    fn peaked_lattice_degenerates_to_greedy_chain() -> Result<()> {
        let candidates: Vec<Vec<usize>> = (0..5).map(|p| vec![10 + p, 20 + p, 30 + p]).collect();
        let tree = DFlashTree::best_first(1, &candidates, 5, 8, |_, _| Ok(vec![0.0, 50.0, -3.0]))?;
        assert_eq!(tree, DFlashTree::chain(1, &[20, 21, 22, 23, 24])?);
        Ok(())
    }

    #[test]
    fn budget_and_depth_are_respected() -> Result<()> {
        let (candidates, score) = lattice(8, 16, 3);
        for budget in [1, 4, 8, 15] {
            for max_depth in [1, 2, 8, 20] {
                let tree = DFlashTree::best_first(5, &candidates, budget, max_depth, |p, prev| {
                    Ok(score(p, prev))
                })?;
                assert_eq!(
                    tree.len(),
                    budget + 1,
                    "budget {budget} profondeur {max_depth}"
                );
                check_topology(&tree);
                let deepest = tree.depths().iter().copied().max().unwrap_or(0);
                assert!(deepest <= max_depth.min(8));
                // Frères distincts : un même token n'apparaît qu'une fois sous un parent.
                for row in 1..tree.len() {
                    assert_eq!(
                        tree.child(tree.parents()[row], tree.tokens()[row]),
                        Some(row)
                    );
                }
            }
        }
        // Profondeur 1 avec 3 candidats : l'arbre s'arrête à 3 nœuds.
        let (small, score) = lattice(4, 3, 1);
        let tree = DFlashTree::best_first(5, &small, 8, 1, |p, prev| Ok(score(p, prev)))?;
        assert_eq!(tree.len(), 4);
        assert!(DFlashTree::best_first(5, &small, 32, 4, |p, prev| Ok(score(p, prev))).is_err());
        Ok(())
    }

    /// Égalités de scores : départage par rang du candidat (le plus petit d'abord),
    /// résultat stable d'un appel à l'autre.
    #[test]
    fn ties_break_on_candidate_rank() -> Result<()> {
        let candidates: Vec<Vec<usize>> = (0..3).map(|p| vec![10 + p, 20 + p, 30 + p]).collect();
        let tree = DFlashTree::best_first(1, &candidates, 3, 3, |_, _| Ok(vec![1.0; 3]))?;
        // Les trois enfants de l'ancre (log 1/3) battent tout petit-enfant (log 1/9).
        assert_eq!(tree.tokens(), &[1, 10, 20, 30]);
        assert_eq!(tree.parents(), &[0, 0, 0, 0]);
        let again = DFlashTree::best_first(1, &candidates, 3, 3, |_, _| Ok(vec![1.0; 3]))?;
        assert_eq!(tree, again);
        Ok(())
    }

    /// Même token sous deux historiques différents : deux nœuds distincts.
    #[test]
    fn duplicate_tokens_keep_their_history() -> Result<()> {
        let candidates = vec![vec![4, 5], vec![4, 5]];
        let tree = DFlashTree::best_first(1, &candidates, 6, 2, |_, _| Ok(vec![0.0, 0.0]))?;
        assert_eq!(tree.len(), 7);
        check_topology(&tree);
        let fours = (1..tree.len())
            .filter(|&row| tree.tokens()[row] == 4)
            .count();
        assert_eq!(fours, 3);
        assert_eq!(tree.accepted_path(&[5, 0, 4]).len(), 3);
        Ok(())
    }

    /// Force brute : le chemin accepté est le plus long préfixe de la
    /// continuation `truth` présent dans l'arbre, pour des arbres et des
    /// continuations aléatoires.
    #[test]
    fn accepted_path_is_longest_prefix_in_tree() -> Result<()> {
        for seed in 0..40u64 {
            let (candidates, score) = lattice(6, 4, seed);
            let tree = DFlashTree::best_first(9, &candidates, 10, 6, |p, prev| Ok(score(p, prev)))?;
            check_topology(&tree);
            let truth: Vec<usize> = (0..6)
                .map(|p| candidates[p][((seed as usize) * 7 + p * 3) % 4])
                .collect();
            // Cibles par ligne : la ligne de profondeur d prédit truth[d].
            let targets: Vec<usize> = tree
                .depths()
                .iter()
                .map(|&d| truth.get(d).copied().unwrap_or(usize::MAX))
                .collect();
            let path = tree.accepted_path(&targets);
            let prefix_in_tree = |len: usize| {
                (1..tree.len()).any(|row| {
                    let mut tokens = Vec::new();
                    let mut node = row;
                    while node != 0 {
                        tokens.push(tree.tokens()[node]);
                        node = tree.parents()[node];
                    }
                    tokens.reverse();
                    tokens.len() == len && tokens[..] == truth[..len]
                })
            };
            let longest = (1..=6).rev().find(|&len| prefix_in_tree(len)).unwrap_or(0);
            assert_eq!(path.len() - 1, longest, "graine {seed}");
        }
        Ok(())
    }

    /// Rejeu hors ligne des traces `SARAGOSSA_RUST_DFLASH_TREE_TRACE` (JSONL) :
    /// tokens par tour de la chaîne observée et de l'arbre best-first à budget
    /// B, avec la même règle que l'analyse de référence (vérité = sortie greedy,
    /// tours censurés en fin ignorés). `SARAGOSSA_RUST_DFLASH_TREE_REPLAY` liste
    /// les traces séparées par `:` ; sans elle, le test ne fait rien.
    #[test]
    #[ignore = "traces DFlash requises (SARAGOSSA_RUST_DFLASH_TREE_REPLAY)"]
    fn replay_traces_tokens_per_tour() -> Result<()> {
        let Some(paths) = crate::runtime_flags::env_var("SARAGOSSA_RUST_DFLASH_TREE_REPLAY") else {
            eprintln!("skip rejeu arbre DFlash: SARAGOSSA_RUST_DFLASH_TREE_REPLAY absent");
            return Ok(());
        };
        let as_usize = |value: &serde_json::Value| -> Vec<usize> {
            value
                .as_array()
                .map(|items| {
                    items
                        .iter()
                        .filter_map(serde_json::Value::as_u64)
                        .filter_map(|v| usize::try_from(v).ok())
                        .collect()
                })
                .unwrap_or_default()
        };
        for path in paths.split(':').filter(|path| !path.is_empty()) {
            let text = std::fs::read_to_string(path).map_err(|source| InferError::Io {
                path: path.into(),
                source,
            })?;
            let lines = text
                .lines()
                .map(serde_json::from_str::<serde_json::Value>)
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|error| InferError::Config(format!("trace {path}: {error}")))?;
            let Some((last, tours)) = lines.split_last() else {
                continue;
            };
            let generated = as_usize(&last["generated"]);
            let budgets = [8usize, 10, 12, 15];
            let mut chain = 0usize;
            let mut trees = [0usize; 4];
            let mut counted = 0usize;
            for tour in tours {
                let anchor_index = tour["anchor_index"].as_u64().unwrap_or(0) as usize;
                let drafts = as_usize(&tour["drafts"]);
                let candidates: Vec<Vec<usize>> = tour["candidates"]
                    .as_array()
                    .map(|rows| rows.iter().map(as_usize).collect())
                    .unwrap_or_default();
                let lattice: Vec<Vec<f32>> = tour["scores"]
                    .as_array()
                    .map(|rows| {
                        rows.iter()
                            .map(|row| {
                                row.as_array()
                                    .map(|v| {
                                        v.iter()
                                            .filter_map(|x| x.as_f64())
                                            .map(|x| x as f32)
                                            .collect()
                                    })
                                    .unwrap_or_default()
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                let positions = candidates.len();
                if anchor_index + positions >= generated.len() || drafts.len() != positions {
                    continue;
                }
                let truth = &generated[anchor_index + 1..anchor_index + 1 + positions];
                counted += 1;
                chain += drafts.iter().zip(truth).take_while(|(d, t)| d == t).count() + 1;
                let anchor = generated[anchor_index];
                for (slot, &budget) in budgets.iter().enumerate() {
                    let tree = DFlashTree::best_first(
                        anchor,
                        &candidates,
                        budget,
                        positions,
                        |p, prev| {
                            let k = candidates[p].len();
                            let start = if p == 0 {
                                0
                            } else {
                                k * candidates[p - 1]
                                    .iter()
                                    .position(|&t| t == prev)
                                    .ok_or_else(|| {
                                        InferError::Dimension(format!(
                                            "précédent {prev} absent (position {p})"
                                        ))
                                    })?
                            };
                            Ok(lattice[p][start..start + k].to_vec())
                        },
                    )?;
                    let targets: Vec<usize> = tree
                        .depths()
                        .iter()
                        .map(|&d| truth.get(d).copied().unwrap_or(usize::MAX))
                        .collect();
                    trees[slot] += tree.accepted_path(&targets).len();
                }
            }
            let mean = |total: usize| total as f64 / counted.max(1) as f64;
            let gains = budgets
                .iter()
                .zip(trees)
                .map(|(budget, total)| {
                    format!(
                        "B={budget} {:.3} ({:+.1} %)",
                        mean(total),
                        100.0 * (mean(total) / mean(chain) - 1.0)
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");
            eprintln!(
                "{path}: {counted} tours, chaîne {:.3} tokens/tour, {gains}",
                mean(chain)
            );
        }
        Ok(())
    }
}
