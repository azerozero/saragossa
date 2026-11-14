//! Préfixes VoiceDesign privés au modèle, bornés en nombre et en octets.

use super::TtsModel;
use crate::{CausalDecoderCache, Result, Tensor};
use std::collections::VecDeque;

const MAX_ENTRIES: usize = 4;
const MAX_BYTES: usize = 32 * 1024 * 1024;

#[cfg(test)]
mod tests;

#[derive(Debug)]
struct Entry {
    prefix: Tensor,
    cache: CausalDecoderCache,
    final_state: Tensor,
    bytes: usize,
}

#[derive(Debug, Default)]
pub(super) struct TtsPrefillMemo {
    entries: VecDeque<Entry>,
    bytes: usize,
}

impl TtsPrefillMemo {
    fn get(&mut self, prefix: &Tensor) -> Option<(CausalDecoderCache, Tensor)> {
        // Les embeddings encodent voix, langue et tokens spéciaux. Comparer
        // leurs bits évite les collisions de hash et distingue les zéros signés.
        let index = self.entries.iter().position(|entry| {
            entry.prefix.shape() == prefix.shape()
                && entry
                    .prefix
                    .data()
                    .iter()
                    .zip(prefix.data())
                    .all(|(a, b)| a.to_bits() == b.to_bits())
        })?;
        let entry = self.entries.remove(index)?;
        // Clone exclut l'arène résidente et partage seulement les segments K/V
        // immuables : la génération ne peut pas altérer le préfixe mémoïsé.
        let state = (entry.cache.clone(), entry.final_state.clone());
        self.entries.push_back(entry);
        Some(state)
    }

    fn insert(&mut self, prefix: Tensor, cache: CausalDecoderCache, final_state: Tensor) {
        // Deux requêtes peuvent calculer le même miss sans garder le mutex
        // pendant le forward. La seconde ne doit pas créer un doublon.
        if self.get(&prefix).is_some() {
            return;
        }
        let cache = cache.compact_tts_snapshot();
        let bytes = cache
            .estimated_cpu_bytes()
            .saturating_add(prefix.capacity_bytes())
            .saturating_add(final_state.capacity_bytes());
        if bytes > MAX_BYTES {
            return;
        }
        while self.entries.len() >= MAX_ENTRIES || self.bytes.saturating_add(bytes) > MAX_BYTES {
            let Some(oldest) = self.entries.pop_front() else {
                break;
            };
            self.bytes -= oldest.bytes;
        }
        self.bytes += bytes;
        self.entries.push_back(Entry {
            prefix,
            cache,
            final_state,
            bytes,
        });
    }
}

impl TtsModel {
    pub(super) fn prefill_talker(&self, input: &Tensor) -> Result<(CausalDecoderCache, Tensor)> {
        let _qmm_scope = crate::runtime_flags::install_tts_qmm_na_dequant_scope(
            crate::runtime_flags::tts_qmm_na_dequant_enabled(),
        );
        let full = || {
            #[cfg(test)]
            tests::FULL_PREFILLS.with(|count| count.set(count.get() + 1));
            self.talker.prefill_cache_from_embeddings(input)
        };
        if !crate::runtime_flags::tts_prefill_cache_enabled() || self.clone_ctx.is_some() {
            return full();
        }
        let (rows, hidden) = input.as_matrix()?;
        if rows < 2 {
            return full();
        }
        let prefix = Tensor::from_vec(
            vec![rows - 1, hidden],
            input.data()[..(rows - 1) * hidden].to_vec(),
        )?;
        let cached = match self.prefill_memo.lock() {
            Ok(mut memo) => memo.get(&prefix),
            Err(_) => return full(),
        };
        #[cfg(test)]
        let reused = cached.is_some();
        let (mut cache, _) = match cached {
            Some(state) => state,
            None => {
                let (cache, state) = match self.talker.prefill_cache_from_embeddings(&prefix) {
                    Ok(state) => state,
                    Err(_) => return full(),
                };
                if let Ok(mut memo) = self.prefill_memo.lock() {
                    memo.insert(prefix, cache.clone(), state.clone());
                }
                (cache, state)
            }
        };
        let last = Tensor::row(input.last_row()?.to_vec())?;
        match self.talker.extend_tts_prefill(&mut cache, &last) {
            Ok(state) => {
                #[cfg(test)]
                if reused {
                    tests::MEMO_REUSES.with(|count| count.set(count.get() + 1));
                }
                Ok((cache, state))
            }
            Err(_) => full(),
        }
    }
}
