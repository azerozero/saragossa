//! Cache de préfixe par blocs de `serve`.

use saragossa::runtime_flags::{
    serve_prefix_block_tokens, serve_prefix_blocks_per_session, serve_prefix_cache_blocks,
};
use saragossa::{CausalDecoderPromptMetalSnapshot, CausalDecoderPromptState};
use sha2::{Digest, Sha256};

/// Hash de bloc chaîné.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) struct BlockHash([u8; 32]);

impl BlockHash {
    /// Renvoie la racine du chaînage.
    #[must_use]
    pub(super) fn root() -> Self {
        Self([0; 32])
    }

    /// Renvoie la racine isolée pour une session.
    #[must_use]
    pub(super) fn for_session(session_key: Option<&str>) -> Self {
        let Some(key) = session_key.filter(|key| !key.is_empty()) else {
            return Self::root();
        };
        let mut hasher = Sha256::new();
        hasher.update(key.as_bytes());
        let digest = hasher.finalize();
        let mut hash = [0_u8; 32];
        hash.copy_from_slice(&digest);
        Self(hash)
    }

    /// Calcule le hash chaîné du bloc suivant.
    #[must_use]
    pub(super) fn chain(self, tokens: &[usize]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(self.0);
        for token in tokens {
            hasher.update((*token as u64).to_le_bytes());
        }
        let digest = hasher.finalize();
        let mut hash = [0_u8; 32];
        hash.copy_from_slice(&digest);
        Self(hash)
    }
}

/// Préfixe trouvé dans le cache par blocs.
#[derive(Clone, Debug)]
pub(super) struct PrefixBlockHit {
    /// Nombre de tokens réutilisables.
    pub(super) tokens: usize,
    /// Hash chaîné du dernier bloc réutilisé.
    pub(super) hash: BlockHash,
    /// Snapshot exact à la frontière du dernier bloc.
    pub(super) state: CausalDecoderPromptState,
    /// Snapshot Metal associé à cette frontière.
    pub(super) metal: CausalDecoderPromptMetalSnapshot,
}

#[derive(Clone, Debug)]
struct PrefixBlockEntry {
    session: BlockHash,
    hash: BlockHash,
    tokens: usize,
    state: CausalDecoderPromptState,
    metal: CausalDecoderPromptMetalSnapshot,
    bytes: usize,
}

/// Cache LRU de snapshots aux frontières de blocs.
#[derive(Clone, Debug)]
pub(super) struct BlockAwarePrefixCache {
    block_tokens: usize,
    capacity: usize,
    blocks_per_session: usize,
    entries: Vec<PrefixBlockEntry>,
}

impl BlockAwarePrefixCache {
    /// Construit le cache depuis les flags runtime.
    #[must_use]
    pub(super) fn from_runtime_flags() -> Self {
        Self::with_blocks_per_session(
            serve_prefix_block_tokens(),
            serve_prefix_cache_blocks(),
            serve_prefix_blocks_per_session(),
        )
    }

    /// Construit un cache de capacité fixe.
    #[must_use]
    pub(super) fn new(block_tokens: usize, capacity: usize) -> Self {
        Self::with_blocks_per_session(block_tokens, capacity, capacity)
    }

    /// Construit un cache avec un plafond de blocs par session.
    #[must_use]
    pub(super) fn with_blocks_per_session(
        block_tokens: usize,
        capacity: usize,
        blocks_per_session: usize,
    ) -> Self {
        Self {
            block_tokens: block_tokens.max(1),
            capacity,
            blocks_per_session,
            entries: Vec::new(),
        }
    }

    /// Renvoie la taille de bloc effective.
    #[must_use]
    pub(super) fn block_tokens(&self) -> usize {
        self.block_tokens
    }

    /// Recherche la plus longue chaîne de blocs préfixes.
    pub(super) fn match_prefix(
        &mut self,
        root: BlockHash,
        tokens: &[usize],
    ) -> Option<PrefixBlockHit> {
        if self.capacity == 0 {
            return None;
        }
        let mut previous = root;
        let mut best = None;
        for (index, block) in tokens.chunks_exact(self.block_tokens).enumerate() {
            let hash = previous.chain(block);
            let expected_tokens = (index + 1).saturating_mul(self.block_tokens);
            let Some(entry_index) = self.entries.iter().position(|entry| {
                entry.session == root && entry.hash == hash && entry.tokens == expected_tokens
            }) else {
                break;
            };
            let entry = self.entries.remove(entry_index);
            let hit = PrefixBlockHit {
                tokens: entry.tokens,
                hash: entry.hash,
                state: entry.state.clone(),
                metal: entry.metal.clone(),
            };
            self.entries.insert(0, entry);
            previous = hash;
            best = Some(hit);
        }
        best
    }

    /// Insère ou rafraîchit un snapshot de frontière de bloc.
    pub(super) fn insert(
        &mut self,
        session: BlockHash,
        hash: BlockHash,
        tokens: usize,
        state: CausalDecoderPromptState,
        metal: CausalDecoderPromptMetalSnapshot,
    ) -> usize {
        if self.capacity == 0 {
            return 0;
        }
        if let Some(index) = self.entries.iter().position(|entry| {
            entry.session == session && entry.hash == hash && entry.tokens == tokens
        }) {
            self.entries.remove(index);
        }
        let bytes = state
            .estimated_cpu_bytes()
            .saturating_add(metal.estimated_bytes());
        self.entries.insert(
            0,
            PrefixBlockEntry {
                session,
                hash,
                tokens,
                state,
                metal,
                bytes,
            },
        );
        while self.session_block_count(session) > self.blocks_per_session {
            if self.evict_lru_session_block(session).is_none() {
                break;
            }
        }
        while self.entries.len() > self.capacity {
            self.entries.pop();
        }
        bytes
    }

    /// Evince le bloc le moins récemment utilisé.
    pub(super) fn evict_lru_block(&mut self) -> Option<usize> {
        self.entries.pop().map(|entry| entry.bytes)
    }

    /// Renvoie l'empreinte estimée des snapshots.
    #[must_use]
    pub(super) fn estimated_bytes(&self) -> usize {
        self.entries.iter().map(|entry| entry.bytes).sum()
    }

    fn session_block_count(&self, session: BlockHash) -> usize {
        self.entries
            .iter()
            .filter(|entry| entry.session == session)
            .count()
    }

    fn evict_lru_session_block(&mut self, session: BlockHash) -> Option<usize> {
        let index = self
            .entries
            .iter()
            .rposition(|entry| entry.session == session)?;
        Some(self.entries.remove(index).bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn test_state(position: usize) -> CausalDecoderPromptState {
        let model = test_model();
        let prompt = (0..position).map(|index| index % 3).collect::<Vec<_>>();
        model
            .prefill_prompt_state_uncached(&prompt)
            .expect("invariant: prefill test valide")
    }

    fn test_model() -> saragossa::CausalDecoder {
        saragossa::CausalDecoder::from_tensors(
            test_weights(),
            saragossa::CausalDecoderConfig::default(),
        )
        .expect("invariant: modèle test valide")
    }

    fn test_weights() -> HashMap<String, saragossa::Tensor> {
        let mut tensors = HashMap::new();
        tensors.insert(
            "embed_tokens.weight".to_string(),
            saragossa::Tensor::from_vec(vec![3, 2], vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
                .expect("invariant: embedding valide"),
        );
        tensors.insert(
            "layers.0.input_layernorm.weight".to_string(),
            saragossa::Tensor::from_vec(vec![2], vec![1.0, 1.0]).expect("invariant: norm valide"),
        );
        tensors.insert(
            "norm.weight".to_string(),
            saragossa::Tensor::from_vec(vec![2], vec![1.0, 1.0]).expect("invariant: norm valide"),
        );
        for prefix in [
            "layers.0.self_attn.q_proj",
            "layers.0.self_attn.k_proj",
            "layers.0.self_attn.v_proj",
            "layers.0.self_attn.o_proj",
        ] {
            tensors.insert(
                format!("{prefix}.weight"),
                identity2().expect("invariant: identité valide"),
            );
        }
        tensors.insert(
            "lm_head.weight".to_string(),
            saragossa::Tensor::from_vec(vec![3, 2], vec![1.0, 0.0, -1.0, 0.0, 0.0, 1.0])
                .expect("invariant: lm_head valide"),
        );
        tensors
    }

    fn identity2() -> saragossa::Result<saragossa::Tensor> {
        saragossa::Tensor::from_vec(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0])
    }

    #[test]
    fn chained_hash_depends_on_previous_block() {
        let first = BlockHash::root().chain(&[1, 2]);
        let left = first.chain(&[3, 4]);
        let right = BlockHash::root().chain(&[3, 4]);

        assert_ne!(left, right);
    }

    #[test]
    fn session_roots_are_deterministic_and_disjoint() {
        let alpha = BlockHash::for_session(Some("alpha"));
        let alpha_again = BlockHash::for_session(Some("alpha"));
        let beta = BlockHash::for_session(Some("beta"));

        assert_eq!(alpha, alpha_again);
        assert_ne!(alpha, beta);
    }

    #[test]
    fn absent_session_keeps_global_root_byte_identity() {
        let tokens = [1, 2, 3];

        assert_eq!(BlockHash::for_session(None), BlockHash::root());
        assert_eq!(
            BlockHash::for_session(None).chain(&tokens),
            BlockHash::root().chain(&tokens)
        );
    }

    #[test]
    fn prefix_cache_matches_longest_chain() {
        let mut cache = BlockAwarePrefixCache::new(2, 8);
        let root = BlockHash::root();
        let first = root.chain(&[1, 2]);
        let second = first.chain(&[3, 4]);
        cache.insert(
            root,
            first,
            2,
            test_state(2),
            CausalDecoderPromptMetalSnapshot::default(),
        );
        cache.insert(
            root,
            second,
            4,
            test_state(4),
            CausalDecoderPromptMetalSnapshot::default(),
        );

        let hit = cache
            .match_prefix(root, &[1, 2, 3, 4, 5])
            .expect("invariant: préfixe présent");

        assert_eq!(hit.tokens, 4);
        assert_eq!(hit.hash, second);
        assert_eq!(hit.state.position(), 4);
    }

    #[test]
    fn prefix_cache_evicts_lru_block() {
        let mut cache = BlockAwarePrefixCache::new(1, 2);
        let root = BlockHash::root();
        let first = root.chain(&[1]);
        let second = first.chain(&[2]);
        let third = second.chain(&[3]);
        cache.insert(
            root,
            first,
            1,
            test_state(1),
            CausalDecoderPromptMetalSnapshot::default(),
        );
        cache.insert(
            root,
            second,
            2,
            test_state(2),
            CausalDecoderPromptMetalSnapshot::default(),
        );
        cache.insert(
            root,
            third,
            3,
            test_state(3),
            CausalDecoderPromptMetalSnapshot::default(),
        );

        assert!(cache.match_prefix(root, &[1]).is_none());
        assert_eq!(
            cache.match_prefix(root, &[1, 2, 3]).map(|hit| hit.tokens),
            None
        );
    }

    #[test]
    fn default_session_cap_keeps_global_lru_eviction_order() {
        let mut cache = BlockAwarePrefixCache::new(1, 2);
        let alpha = BlockHash::for_session(Some("agent-a"));
        let beta = BlockHash::for_session(Some("agent-b"));
        let beta_first = beta.chain(&[9]);
        let alpha_first = alpha.chain(&[1]);
        let alpha_second = alpha_first.chain(&[2]);
        cache.insert(
            beta,
            beta_first,
            1,
            test_state(1),
            CausalDecoderPromptMetalSnapshot::default(),
        );
        cache.insert(
            alpha,
            alpha_first,
            1,
            test_state(1),
            CausalDecoderPromptMetalSnapshot::default(),
        );
        cache.insert(
            alpha,
            alpha_second,
            2,
            test_state(2),
            CausalDecoderPromptMetalSnapshot::default(),
        );

        assert_eq!(
            cache
                .entries
                .iter()
                .map(|entry| entry.hash)
                .collect::<Vec<_>>(),
            vec![alpha_second, alpha_first]
        );
        assert!(cache.match_prefix(beta, &[9]).is_none());
    }

    #[test]
    fn reduced_session_cap_evicts_only_that_session_first() {
        let mut cache = BlockAwarePrefixCache::with_blocks_per_session(1, 2, 1);
        let alpha = BlockHash::for_session(Some("agent-a"));
        let beta = BlockHash::for_session(Some("agent-b"));
        let beta_first = beta.chain(&[9]);
        let alpha_first = alpha.chain(&[1]);
        let alpha_second = alpha_first.chain(&[2]);
        cache.insert(
            beta,
            beta_first,
            1,
            test_state(1),
            CausalDecoderPromptMetalSnapshot::default(),
        );
        cache.insert(
            alpha,
            alpha_first,
            1,
            test_state(1),
            CausalDecoderPromptMetalSnapshot::default(),
        );
        cache.insert(
            alpha,
            alpha_second,
            2,
            test_state(2),
            CausalDecoderPromptMetalSnapshot::default(),
        );

        assert_eq!(
            cache
                .entries
                .iter()
                .map(|entry| entry.hash)
                .collect::<Vec<_>>(),
            vec![alpha_second, beta_first]
        );
        assert_eq!(
            cache.match_prefix(beta, &[9]).map(|hit| hit.hash),
            Some(beta_first)
        );
        assert_eq!(cache.session_block_count(alpha), 1);
    }

    #[test]
    fn prefix_cache_does_not_match_across_sessions() {
        let mut cache = BlockAwarePrefixCache::new(2, 8);
        let alpha = BlockHash::for_session(Some("agent-a"));
        let beta = BlockHash::for_session(Some("agent-b"));
        let first = alpha.chain(&[1, 2]);
        let second = first.chain(&[3, 4]);
        cache.insert(
            alpha,
            first,
            2,
            test_state(2),
            CausalDecoderPromptMetalSnapshot::default(),
        );
        cache.insert(
            alpha,
            second,
            4,
            test_state(4),
            CausalDecoderPromptMetalSnapshot::default(),
        );

        assert_ne!(first, beta.chain(&[1, 2]));
        assert!(cache.match_prefix(beta, &[1, 2, 3, 4]).is_none());
        assert_eq!(
            cache
                .match_prefix(alpha, &[1, 2, 3, 4])
                .map(|hit| hit.tokens),
            Some(4)
        );
    }
}
