//! Boucle greedy « lookup seul » : propositions du lookup n-grammes, sans tête
//! MTP ni drafter, vérifiées en lignes sur l'arène résidente.

use super::super::lookup::NgramLookup;
use super::super::*;
use crate::decoder::ResidentVerifyHead;

/// Conditions de la boucle lookup indépendantes du modèle : drafts demandés,
/// greedy sans contrainte ni séquence d'arrêt, et pas de plafond de débit
/// (`throttled`) : la boucle ne temporise pas, elle le contournerait.
fn lookup_greedy_options_ok(drafts: usize, options: &GenerationOptions, throttled: bool) -> bool {
    drafts > 0
        && !throttled
        && options.temperature <= f32::EPSILON
        && options.token_constraint.is_none()
        && options.stop_sequences.is_empty()
}

impl CausalDecoder {
    /// Vrai si la boucle lookup greedy peut tourner sur ce cache : options
    /// éligibles (cf. [`lookup_greedy_options_ok`]), arène résidente prête et
    /// vérification en lignes de toutes les couches (rollback par captures).
    pub(in crate::decoder) fn lookup_greedy_eligible(
        &self,
        cache: &CausalDecoderCache,
        options: &GenerationOptions,
    ) -> bool {
        lookup_greedy_options_ok(
            crate::runtime_flags::lookup_greedy_draft_tokens(),
            options,
            crate::runtime_flags::decode_min_interval().is_some(),
        ) && cache
            .resident
            .as_ref()
            .is_some_and(|arena| self.resident_layer_batch_supported(arena))
    }

    /// Decode greedy depuis `first` (premier token, issu du prefill) : à chaque
    /// pas, le lookup propose des drafts tirés du prompt et de la sortie ; s'il
    /// en propose, `[token, drafts…]` est vérifié en lignes et le préfixe accepté
    /// plus le token de correction (ou bonus) sont émis. Sans proposition, decode
    /// résident : pipeliné si `pipelined` (interrompu dès que le lookup propose,
    /// les pas en vol restant acquis), sinon un pas à la fois. Sortie identique
    /// au greedy AR aux quasi-égalités près (la ligne vérifiée et le pas
    /// mono-ligne n'accumulent pas dans le même ordre).
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si la vérification ou le decode résident échoue, ou si
    /// une vérification rejetée ne rend pas de captures.
    #[expect(
        clippy::too_many_arguments,
        reason = "boucle de decode: cache, prompt, budget, options, mode et callback"
    )]
    pub(in crate::decoder) fn decode_tokens_resident_lookup(
        &self,
        cache: &mut CausalDecoderCache,
        prompt: &[usize],
        first: usize,
        max_new_tokens: usize,
        options: &GenerationOptions,
        mut pipelined: bool,
        on_token: &mut impl FnMut(usize) -> bool,
    ) -> Result<Vec<usize>> {
        let max_draft = crate::runtime_flags::lookup_greedy_draft_tokens();
        // `k` drafts vérifiés rendent `k + 1` tokens : jamais au-delà du budget.
        let draft_cap = |emitted: usize| max_draft.min(max_new_tokens.saturating_sub(emitted + 1));
        let mut lookup = NgramLookup::new(prompt);
        let mut sampler = DeterministicSampler::new(options.seed);
        let mut generated = Vec::with_capacity(max_new_tokens);
        // Tokens produits à émettre ; le dernier n'est pas encore injecté.
        let mut pending = vec![first];
        loop {
            for &next in &pending {
                if options.stop_token_ids.contains(&next) {
                    return Ok(generated);
                }
                generated.push(next);
                if !on_token(next) || generated.len() >= max_new_tokens {
                    return Ok(generated);
                }
            }
            let token = *pending
                .last()
                .ok_or_else(|| InferError::Config("lookup greedy: aucun token".into()))?;
            pending.clear();
            lookup.sync(&generated);
            // NOTE: `propose` a des effets de bord (index du dernier n-gramme,
            // décompte du repos) : un appel par position, comme la boucle MTP.
            if let Some(drafts) = lookup.propose(draft_cap(generated.len())) {
                pending = self.verify_lookup_drafts(cache, &mut lookup, token, &drafts)?;
                continue;
            }
            if !pipelined {
                pending.push(self.decode_token_resident(cache, token)?);
                continue;
            }
            // Le pipeline réémet `token` en tête, déjà émis ici.
            let offset = generated.len() - 1;
            let output = self.decode_tokens_resident_pipelined_until(
                cache,
                token,
                max_new_tokens - offset,
                options,
                &mut sampler,
                |tail| {
                    lookup.sync_from(offset, tail);
                    // Même appel que le pas mono-ligne (repos compris) ; la
                    // proposition est refaite au retour, après la vidange.
                    lookup.propose(draft_cap(offset + tail.len())).is_some()
                },
            )?;
            let Some(output) = output else {
                pipelined = false;
                pending.push(self.decode_token_resident(cache, token)?);
                continue;
            };
            pending = output.tokens.get(1..).unwrap_or_default().to_vec();
            if !output.drained {
                // Budget ou token d'arrêt : les tokens rendus sont les derniers
                // (le pipeline a déjà écarté le token d'arrêt).
                for next in pending {
                    generated.push(next);
                    if !on_token(next) {
                        break;
                    }
                }
                return Ok(generated);
            }
        }
    }

    /// Vérifie `[token, drafts…]` en lignes, ramène le cache au préfixe accepté
    /// et rend les tokens à émettre : drafts acceptés puis correction ou bonus.
    fn verify_lookup_drafts(
        &self,
        cache: &mut CausalDecoderCache,
        lookup: &mut NgramLookup,
        token: usize,
        drafts: &[usize],
    ) -> Result<Vec<usize>> {
        let mut rows = Vec::with_capacity(drafts.len() + 1);
        rows.push(token);
        rows.extend_from_slice(drafts);
        let output = self
            .next_final_states_resident_verify(
                cache,
                &rows,
                ResidentVerifyHead::Argmax,
                true,
                None,
                None,
            )?
            .ok_or_else(|| InferError::Config("lookup greedy: vérification absente".into()))?;
        let targets = output.tokens.as_deref().ok_or_else(|| {
            InferError::Config("lookup greedy: argmax de vérification absent".into())
        })?;
        let accepted = drafts
            .iter()
            .zip(targets)
            .take_while(|(draft, target)| draft == target)
            .count();
        lookup.observe(drafts.len(), accepted);
        if accepted < drafts.len() {
            let captures = output.captures.as_ref().ok_or_else(|| {
                InferError::Config("lookup greedy: captures de rollback absentes".into())
            })?;
            self.restore_mtp_resident_progress(cache, captures, accepted + 1)?;
        }
        let mut emitted = drafts[..accepted].to_vec();
        emitted.push(*targets.get(accepted).ok_or_else(|| {
            InferError::Dimension("lookup greedy: ligne de correction absente".into())
        })?);
        Ok(emitted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_options_refuse_throttle_sampling_and_constraints() {
        let greedy = GenerationOptions {
            stop_token_ids: vec![2],
            stop_sequences: Vec::new(),
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            seed: 0,
            token_constraint: None,
        };
        assert!(lookup_greedy_options_ok(2, &greedy, false));
        assert!(!lookup_greedy_options_ok(0, &greedy, false), "flag à 0");
        assert!(
            !lookup_greedy_options_ok(2, &greedy, true),
            "plafond de débit actif"
        );
        let sampled = GenerationOptions {
            temperature: 0.7,
            ..greedy.clone()
        };
        assert!(!lookup_greedy_options_ok(2, &sampled, false));
        let stop_sequence = GenerationOptions {
            stop_sequences: vec![vec![1, 2]],
            ..greedy
        };
        assert!(!lookup_greedy_options_ok(2, &stop_sequence, false));
    }
}
