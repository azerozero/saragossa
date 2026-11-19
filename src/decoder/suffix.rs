//! SuffixDecoding (arXiv 2411.04975) : arbre de suffixes CPU sur le prompt et la
//! sortie, sans modèle ni entraînement. Le nœud du plus long suffixe du contexte
//! déjà suivi d'une continuation donne le départ ; le brouillon suit l'enfant le
//! plus fréquent tant que la probabilité empirique cumulée reste au-dessus de
//! `min_prob`, sur au plus `profondeur de la correspondance` tokens (facteur
//! α = 1 du papier). Sortie en chaîne : la vérification en lignes existante
//! accepte exactement le préfixe que le greedy AR aurait émis.
//!
//! Avec `min_prob = 1`, le brouillon s'arrête là où les occurrences passées
//! divergent : c'est la garde « par preuve » du lookup de kimi-k3-in-c
//! (`spec_draft`), généralisée à toutes les occurrences au lieu des deux
//! dernières.

use std::collections::HashMap;

/// Profondeur maximale de l'arbre : correspondance plus continuation.
// ponytail: O(N·MAX_DEPTH) nœuds (≈ 400 Mo à 256k tokens de prompt) ; un vrai
// arbre de suffixes compressé (Ukkonen) si l'opt-in sert sur des prompts longs.
const MAX_DEPTH: usize = 24;
/// Correspondance la plus longue considérée : laisse au moins 8 tokens de
/// continuation sous `MAX_DEPTH`.
const MAX_MATCH: usize = 16;
/// Racine, jamais enfant : sert aussi de « pas d'enfant » dans `Node::best`.
const ROOT: usize = 0;

/// Réglages de la source suffixe (`runtime_flags::suffix_draft`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SuffixParams {
    /// Correspondance minimale (tokens) avant de proposer.
    pub min_match: usize,
    /// Probabilité empirique cumulée sous laquelle le brouillon s'arrête.
    pub min_prob: f32,
}

#[derive(Clone, Copy)]
struct Node {
    token: usize,
    /// Occurrences de la chaîne racine → nœud.
    count: usize,
    /// Enfant le plus fréquent (le plus récent à égalité), `ROOT` si aucun.
    best: usize,
}

pub(super) struct SuffixTree {
    params: SuffixParams,
    nodes: Vec<Node>,
    children: HashMap<(usize, usize), usize>,
    /// `active[k]` : nœud des `k` derniers tokens, `k < MAX_DEPTH`.
    active: Vec<usize>,
    scratch: Vec<usize>,
}

impl SuffixTree {
    pub(super) fn new(params: SuffixParams) -> Self {
        Self {
            params,
            nodes: vec![Node {
                token: 0,
                count: 0,
                best: ROOT,
            }],
            children: HashMap::new(),
            active: vec![ROOT],
            scratch: Vec::with_capacity(MAX_DEPTH),
        }
    }

    /// Ajoute un token : prolonge chaque suffixe actif (`MAX_DEPTH` insertions).
    pub(super) fn push(&mut self, token: usize) {
        let mut next = std::mem::take(&mut self.scratch);
        next.clear();
        next.push(ROOT);
        for (depth, &parent) in self.active.iter().enumerate() {
            let nodes = &mut self.nodes;
            let child = *self.children.entry((parent, token)).or_insert_with(|| {
                nodes.push(Node {
                    token,
                    count: 0,
                    best: ROOT,
                });
                nodes.len() - 1
            });
            nodes[child].count += 1;
            let best = nodes[parent].best;
            if best == ROOT || nodes[child].count >= nodes[best].count {
                nodes[parent].best = child;
            }
            if depth + 1 < MAX_DEPTH {
                next.push(child);
            }
        }
        self.scratch = std::mem::replace(&mut self.active, next);
    }

    /// Brouillon d'au plus `max_draft` tokens, `None` sans correspondance assez
    /// longue ou si le premier token n'atteint pas `min_prob`.
    pub(super) fn propose(&self, max_draft: usize) -> Option<Vec<usize>> {
        let (depth, &start) = self
            .active
            .iter()
            .enumerate()
            .take(MAX_MATCH + 1)
            .skip(self.params.min_match.max(1))
            .rev()
            .find(|(_, &node)| self.nodes[node].best != ROOT)?;
        let budget = max_draft.min(depth);
        let mut draft = Vec::with_capacity(budget);
        // L'occurrence qui finit le contexte n'a pas encore de suite.
        let mut total = self.nodes[start].count.saturating_sub(1).max(1);
        let mut node = start;
        let mut prob = 1.0_f64;
        while draft.len() < budget {
            let best = self.nodes[node].best;
            if best == ROOT {
                break;
            }
            #[expect(clippy::cast_precision_loss, reason = "compteurs ≪ 2^52")]
            let step = self.nodes[best].count as f64 / total as f64;
            prob *= step.min(1.0);
            if prob < f64::from(self.params.min_prob) {
                break;
            }
            draft.push(self.nodes[best].token);
            total = self.nodes[best].count;
            node = best;
        }
        (!draft.is_empty()).then_some(draft)
    }
}

#[cfg(test)]
mod tests {
    use super::{SuffixParams, SuffixTree};

    fn tree(tokens: &[usize], min_match: usize, min_prob: f32) -> SuffixTree {
        let mut tree = SuffixTree::new(SuffixParams {
            min_match,
            min_prob,
        });
        tokens.iter().for_each(|&token| tree.push(token));
        tree
    }

    #[test]
    fn proposes_frequent_continuation_with_adaptive_length() {
        // [1,2,3] vu deux fois suivi de [4,5,6,7] : chaîne complète, bornée par
        // la profondeur de correspondance (3) puis par `max_draft`.
        let seq = [1, 2, 3, 4, 5, 6, 7, 9, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3];
        assert_eq!(tree(&seq, 3, 0.1).propose(8), Some(vec![4, 5, 6]));
        assert_eq!(tree(&seq, 3, 0.1).propose(2), Some(vec![4, 5]));
        assert_eq!(tree(&seq, 4, 0.1).propose(8), None, "correspondance de 3");
        assert_eq!(tree(&seq, 3, 0.1).propose(0), None);
        assert_eq!(tree(&[1, 2], 1, 0.1).propose(4), None, "jamais suivi");

        // [5,6] suivi de 7 deux fois puis de 0 une fois : le plus fréquent gagne.
        let freq = [5, 6, 7, 1, 5, 6, 7, 2, 5, 6, 0, 3, 5, 6];
        assert_eq!(tree(&freq, 2, 0.1).propose(1), Some(vec![7]));
        // Garde par preuve : 2/3 < 1, rien.
        assert_eq!(tree(&freq, 2, 1.0).propose(1), None);

        // Divergence après accord : la garde coupe au premier désaccord.
        let fork = [5, 1, 2, 3, 4, 8, 0, 6, 1, 2, 3, 4, 9, 0, 7, 1, 2, 3];
        assert_eq!(tree(&fork, 3, 1.0).propose(8), Some(vec![4]));
        assert_eq!(tree(&fork, 3, 0.1).propose(8), Some(vec![4, 9, 0]));

        // Égalité de fréquence : la continuation la plus récente.
        let tie = [7, 8, 1, 0, 7, 8, 2, 0, 7, 8];
        assert_eq!(tree(&tie, 2, 0.1).propose(1), Some(vec![2]));
    }

    #[test]
    fn incremental_push_equals_batch_and_respects_depth() {
        let seq: Vec<usize> = (0..200).map(|i| (i * 7 + i / 13) % 11).collect();
        let whole = tree(&seq, 2, 0.1);
        let mut split = tree(&seq[..37], 2, 0.1);
        seq[37..].iter().for_each(|&token| split.push(token));
        assert_eq!(whole.nodes.len(), split.nodes.len());
        assert_eq!(whole.propose(8), split.propose(8));
        assert!(whole.active.len() <= super::MAX_DEPTH);
        // Chaque nœud de profondeur 1 compte les occurrences de son token.
        let ones = seq.iter().filter(|&&token| token == 1).count();
        let node = whole.children[&(super::ROOT, 1)];
        assert_eq!(whole.nodes[node].count, ones);
    }
}
