//! Spéculation par n-grammes sans drafter : quand les trois derniers tokens sont
//! déjà apparus dans le prompt ou la sortie, la continuation observée la plus
//! récente est proposée à la place des drafts de la tête MTP. Sans perte : le
//! tronc vérifie chaque token. Rentable sur les sorties qui recopient (édition
//! de code, citations), nul sur du texte libre, d'où la porte d'acceptation.
//! Actif par défaut sur prompt long seulement (`runtime_flags::mtp_lookup_draft_tokens`).
//! `runtime_flags::suffix_draft` remplace l'index n-grammes par l'arbre de
//! suffixes (`decoder::suffix`) ; repos et synchronisation restent communs.

use std::collections::HashMap;

use super::suffix::{SuffixParams, SuffixTree};

const N: usize = 3;
/// Longueur minimale de la correspondance (suffixe commun) : les 3-grammes seuls
/// accrochent trop souvent sur du texte libre (T1 et T2 perdaient 5 à 20 %).
const MIN_MATCH: usize = 6;

pub(super) struct NgramLookup {
    tokens: Vec<usize>,
    /// n-gramme -> index du token qui a suivi sa dernière occurrence.
    next_after: HashMap<[usize; N], usize>,
    /// Source suffixe à la place de `next_after` quand elle est demandée.
    suffix: Option<SuffixTree>,
    /// Prochain index de fin de n-gramme à indexer (la fin courante est indexée
    /// au moment de la proposition, après la consultation).
    indexed_end: usize,
    consumed_generated: usize,
    cooldown: usize,
    /// Repos à appliquer au prochain échec, doublé à chaque échec, remis à 8 au succès.
    next_cooldown: usize,
    /// Vérifications, drafts proposés et acceptés (`SARAGOSSA_RUST_LOOKUP_STATS=1`).
    stats: [usize; 3],
}

impl NgramLookup {
    pub(super) fn new(prompt: &[usize]) -> Self {
        Self::with_source(prompt, crate::runtime_flags::suffix_draft())
    }

    pub(super) fn with_source(prompt: &[usize], suffix: Option<SuffixParams>) -> Self {
        let mut lookup = Self {
            tokens: Vec::with_capacity(prompt.len() + 256),
            next_after: HashMap::new(),
            suffix: suffix.map(SuffixTree::new),
            indexed_end: N - 1,
            consumed_generated: 0,
            cooldown: 0,
            next_cooldown: 8,
            stats: [0; 3],
        };
        lookup.push(prompt);
        lookup
    }

    /// Ajoute les tokens générés non encore vus (`generated` est la sortie complète).
    pub(super) fn sync(&mut self, generated: &[usize]) {
        self.sync_from(0, generated);
    }

    /// [`Self::sync`] d'une sortie dont seule la fin `tail` (à partir de
    /// `offset`) est passée : les tokens avant `offset` sont déjà indexés.
    pub(super) fn sync_from(&mut self, offset: usize, tail: &[usize]) {
        let start = self
            .consumed_generated
            .saturating_sub(offset)
            .min(tail.len());
        self.consumed_generated = offset + tail.len();
        self.push(&tail[start..]);
    }

    fn push(&mut self, tokens: &[usize]) {
        self.tokens.extend_from_slice(tokens);
        if let Some(tree) = self.suffix.as_mut() {
            tokens.iter().for_each(|&token| tree.push(token));
            return;
        }
        // Indexe toutes les fins de n-gramme sauf la dernière, consultée d'abord.
        while self.indexed_end + 1 < self.tokens.len() {
            self.index_ending_at(self.indexed_end);
            self.indexed_end += 1;
        }
    }

    fn index_ending_at(&mut self, end: usize) {
        if end + 1 >= N {
            let key = key_at(&self.tokens, end);
            self.next_after.insert(key, end + 1);
        }
    }

    /// Continuation observée après la dernière occurrence antérieure du suffixe,
    /// au plus `max_draft` tokens ; `None` sans occurrence ou en période de repos.
    pub(super) fn propose(&mut self, max_draft: usize) -> Option<Vec<usize>> {
        if let Some(tree) = self.suffix.as_ref() {
            if max_draft == 0 {
                return None;
            }
            if self.cooldown > 0 {
                self.cooldown -= 1;
                return None;
            }
            return tree.propose(max_draft);
        }
        let len = self.tokens.len();
        if max_draft == 0 || len < N {
            return None;
        }
        let end = len - 1;
        let key = key_at(&self.tokens, end);
        let found = self.next_after.get(&key).copied();
        if self.indexed_end == end {
            self.next_after.insert(key, len);
            self.indexed_end += 1;
        }
        if self.cooldown > 0 {
            self.cooldown -= 1;
            return None;
        }
        let start = found?;
        if self.match_len(start, end) < MIN_MATCH {
            return None;
        }
        let stop = start.saturating_add(max_draft).min(len);
        (start < stop).then(|| self.tokens[start..stop].to_vec())
    }

    /// Longueur du suffixe commun entre l'occurrence qui finit en `start - 1` et
    /// celle qui finit en `end`.
    fn match_len(&self, start: usize, end: usize) -> usize {
        let mut len = 0;
        while len < start && self.tokens[start - 1 - len] == self.tokens[end - len] {
            len += 1;
        }
        len
    }

    /// Retour de la vérification : une tentative acceptée à moins de 75 % met le
    /// lookup au repos, de plus en plus longtemps à chaque échec (8, 16, ... 64
    /// pas). Une vérification large ne paie que presque entièrement acceptée.
    pub(super) fn observe(&mut self, proposed: usize, accepted: usize) {
        if proposed == 0 {
            return;
        }
        self.stats[0] += 1;
        self.stats[1] += proposed;
        self.stats[2] += accepted;
        if accepted * 4 < proposed * 3 {
            self.cooldown = self.next_cooldown;
            self.next_cooldown = (self.next_cooldown * 2).min(64);
        } else {
            self.next_cooldown = 8;
        }
    }
}

/// Bilan sur stderr à la fin de la génération, diagnostic opt-in des benchs.
impl Drop for NgramLookup {
    fn drop(&mut self) {
        if crate::runtime_flags::env_flag("SARAGOSSA_RUST_LOOKUP_STATS", false) {
            let [verifies, proposed, accepted] = self.stats;
            let source = if self.suffix.is_some() {
                "suffix"
            } else {
                "ngram"
            };
            eprintln!(
                "lookup_stats source={source} verifies={verifies} proposed={proposed} accepted={accepted}"
            );
        }
    }
}

fn key_at(tokens: &[usize], end: usize) -> [usize; N] {
    let mut key = [0; N];
    key.copy_from_slice(&tokens[end + 1 - N..=end]);
    key
}

#[cfg(test)]
mod tests {
    use super::NgramLookup;

    #[test]
    fn proposes_long_matches_and_rests_longer_after_each_miss() {
        // Suffixe [1..=6] déjà vu (6 tokens), continuation [9, 1, 2].
        let mut lookup = NgramLookup::new(&[1, 2, 3, 4, 5, 6, 9, 1, 2, 0, 1, 2, 3, 4, 5, 6]);
        assert_eq!(lookup.propose(4), Some(vec![9, 1, 2, 0]));
        lookup.sync(&[9, 7]);
        assert_eq!(lookup.propose(4), None, "[2,9,7] jamais vu");
        // [4,5,6] vu, mais correspondance de 3 tokens seulement : refusé.
        let mut short = NgramLookup::new(&[4, 5, 6, 8, 0, 4, 5, 6]);
        assert_eq!(short.propose(4), None);
        assert_eq!(NgramLookup::new(&[1, 2]).propose(4), None);
        assert_eq!(lookup.propose(0), None);

        let (mut whole, mut split) = (
            NgramLookup::new(&[1, 2, 3, 4, 5, 6, 7]),
            NgramLookup::new(&[1, 2, 3, 4, 5, 6, 7]),
        );
        whole.sync(&[1, 2, 3, 4, 5, 6]);
        split.sync(&[1, 2]);
        split.sync_from(1, &[2, 3, 4]);
        split.sync_from(3, &[4, 5, 6]);
        assert_eq!(
            whole.tokens, split.tokens,
            "sync par morceaux = sync entier"
        );
        assert_eq!(whole.propose(4), split.propose(4));

        let mut cold = NgramLookup::new(&[1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6]);
        assert_eq!(cold.propose(4), Some(vec![7, 1, 2, 3]));
        cold.observe(4, 3);
        assert_eq!(cold.cooldown, 0, "75 % acceptés : pas de repos");
        cold.observe(4, 2);
        assert_eq!(cold.cooldown, 8);
        cold.sync(&[7]);
        assert_eq!(cold.propose(4), None, "en repos malgré [..,6,7] déjà vu");
        assert_eq!(cold.cooldown, 7);
        cold.observe(4, 0);
        assert_eq!(cold.cooldown, 16, "repos doublé");
        cold.observe(4, 4);
        assert_eq!(cold.next_cooldown, 8, "succès : repos remis à 8");
    }

    #[test]
    fn suffix_source_syncs_and_rests_like_ngrams() {
        let params = Some(super::SuffixParams {
            min_match: 3,
            min_prob: 0.1,
        });
        let mut lookup = NgramLookup::with_source(&[1, 2, 3, 4, 5, 6, 9, 1, 2], params);
        assert_eq!(lookup.propose(4), None, "correspondance de 2");
        lookup.sync(&[3]);
        lookup.sync_from(0, &[3, 4]);
        assert_eq!(lookup.propose(8), Some(vec![5, 6, 9, 1]));
        lookup.observe(4, 0);
        assert_eq!(lookup.propose(8), None, "en repos");
    }
}
