//! Lectures de décision empaquetées : plusieurs séquences indépendantes dans
//! un seul prefill résident.
//!
//! Les opérations ligne à ligne (normes, résiduel, MLP) traitent toutes les
//! lignes d'un coup : leurs poids ne sont lus qu'une fois par couche. Le
//! mélange de positions (projections et récurrence gated-delta, RoPE,
//! attention non causale) passe par la phase d'attention habituelle, une fois
//! par séquence, avec ses positions et son propre état. Aucune ligne ne voit
//! donc une autre séquence : le résultat est celui de prefills séparés.

use super::resident::{f32_byte_offset, ResidentPrefillChunk, ResidentPrefillEncodeContext};
use super::*;

thread_local! {
    static DECISION_SEGMENTS: std::cell::RefCell<Vec<usize>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Exécute `run` avec, sur ce thread, un prefill de décision découpé en
/// séquences consécutives de longueurs `lengths`.
// ponytail: réglage par thread comme `with_prefill_kv_reserve`, plutôt qu'un
// paramètre traversant décodeur et prefill.
pub(crate) fn with_decision_segments<R>(lengths: &[usize], run: impl FnOnce() -> R) -> R {
    /// Rétablit le découpage précédent, panique comprise.
    struct Restore(Vec<usize>);
    impl Drop for Restore {
        fn drop(&mut self) {
            DECISION_SEGMENTS.with(|slot| *slot.borrow_mut() = std::mem::take(&mut self.0));
        }
    }
    let _restore = Restore(DECISION_SEGMENTS.with(|slot| slot.replace(lengths.to_vec())));
    run()
}

pub(super) fn decision_segments() -> Vec<usize> {
    DECISION_SEGMENTS.with(|slot| slot.borrow().clone())
}

impl MetalExecutor {
    /// Phase d'attention d'une couche, séquence par séquence. `caches[i]` est
    /// l'état (K/V ou conv/SSM, neuf) de la séquence `i` à cette couche.
    /// Renvoie la sortie empaquetée `[somme des longueurs, hidden]`.
    #[expect(
        clippy::too_many_arguments,
        reason = "même contrat que encode_resident_attention_phase"
    )]
    pub(super) fn encode_decision_segment_attention(
        &self,
        context: &mut ResidentPrefillEncodeContext<'_>,
        attention: PrefillAttentionLayer<'_>,
        scratch: &PrefillResidentAttentionScratch,
        normed: &BufferRef,
        caches: &[&PrefillResidentLayerCacheBuffer],
        segments: &[usize],
        spec: PrefillAttentionSpec,
        hidden_dim: usize,
        q_dim: usize,
        layer_index: usize,
    ) -> Result<Buffer> {
        let rows: usize = segments.iter().sum();
        let longest = segments.iter().copied().max().unwrap_or(0);
        let packed = self.private_f32_buffer(
            checked_len(rows, hidden_dim, "décision empaquetée")?,
            "decision_segments_output",
        )?;
        let input = self.private_f32_buffer(
            checked_len(longest, hidden_dim, "décision empaquetée")?,
            "decision_segments_input",
        )?;
        let mut first = 0;
        for (&len, cache) in segments.iter().zip(caches) {
            let floats = checked_len(len, hidden_dim, "décision empaquetée")?;
            let offset = f32_byte_offset(checked_len(first, hidden_dim, "décision empaquetée")?)?;
            // La première séquence commence au début du buffer : sans copie.
            let source: &BufferRef = if first == 0 {
                normed
            } else {
                self.run_resident_prefill_section(context, "segment_input", |encoder, _| {
                    self.encode_copy_with_offsets(encoder, normed, offset, &input, 0, floats)
                })?;
                &input
            };
            let output = self.encode_resident_attention_phase(
                context,
                attention,
                scratch.clone(),
                source,
                cache,
                ResidentPrefillChunk {
                    start: 0,
                    rows: len,
                    total: len,
                },
                PrefillAttentionSpec { seq: len, ..spec },
                hidden_dim,
                q_dim,
                layer_index,
                false,
                None,
            )?;
            self.run_resident_prefill_section(context, "segment_output", |encoder, _| {
                self.encode_copy_with_offsets(encoder, &output, 0, &packed, offset, floats)
            })?;
            first += len;
        }
        Ok(packed)
    }
}
