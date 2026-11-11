//! Boucle DFlash avec drafter résident : proposition du bloc sur GPU
//! (`decode_resident::dflash`), vérification résidente en lignes avec captures
//! (rollback par troncature, sans rejeu), contexte du drafter nourri des hidden
//! states cibles commités. Même acceptance que la boucle CPU
//! `generate_greedy_dflash_batched_with_options` (oracle), coût par tour borné
//! par la vérification.

use super::super::*;
use super::dflash_tree::DFlashTree;
use crate::decode_resident::dflash::DFlashResident;
use crate::decode_resident::GpuTensor;
use crate::metal_backend::{commit_and_wait, read_f32_buffer, read_u32_buffer, MetalExecutor};
use crate::DFlashDraft;

/// Le loader valide les types et le nombre de couches avant cette décision.
fn drafter_window(types: &[String], window: Option<usize>) -> Option<usize> {
    types
        .iter()
        .all(|kind| kind == "sliding_attention")
        .then_some(window)
        .flatten()
        .filter(|&window| window > 0)
}

/// Valide la couverture avant l'allocation et calcule la réserve physique.
fn drafter_layout(
    prompt: usize,
    max_new: usize,
    block: usize,
    window: Option<usize>,
    captured_from: usize,
    captured_rows: usize,
    fixed_base: bool,
) -> Result<(usize, usize)> {
    let required_from = prompt.saturating_sub(window.unwrap_or(usize::MAX));
    let skip = required_from.max(captured_from);
    let covered = captured_from
        .checked_add(captured_rows)
        .is_some_and(|end| end >= prompt);
    if skip > required_from || !covered {
        return Err(InferError::Dimension(format!(
            "DFlash captures insuffisantes: origine={captured_from}, lignes={captured_rows}, suffixe requis={required_from}..{prompt}"
        )));
    }
    let base = if fixed_base && window.is_some() {
        skip
    } else {
        0
    };
    // NOTE: la base fixe supprime le coût du préfixe, mais la mémoire résiduelle
    // reste O(max_new). Ajouter un compactage seulement si un cas serve à très
    // grand max_new l'exige ; aucun déplacement KV dans cette version.
    let capacity = prompt
        .checked_sub(base)
        .and_then(|n| n.checked_add(max_new))
        .and_then(|n| n.checked_add(block))
        .and_then(|n| n.checked_add(1))
        .ok_or_else(|| InferError::Shape("DFlash capacité déborde".to_string()))?;
    Ok((skip, capacity))
}

/// Forme des drafts vérifiés à chaque tour.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::decoder) enum DFlashTreeMode {
    /// Chaîne gloutonne du sélecteur, vérifiée par le chemin historique.
    Off,
    /// Même chaîne, vérifiée par la plomberie arbre (oracle : sorties
    /// identiques à `Off`).
    #[cfg(test)]
    Chain,
    /// Arbre best-first de `B` nœuds (`SARAGOSSA_RUST_DFLASH_TREE=B`).
    BestFirst(usize),
}

impl DFlashTreeMode {
    /// Mode demandé par l'environnement et s'il est explicite : sans
    /// `SARAGOSSA_RUST_DFLASH_TREE`, l'arbre par défaut retombe sur la chaîne
    /// quand le modèle ne le supporte pas, au lieu d'échouer. `greedy` : requête
    /// à température nulle (le mode strict n'abaisse le défaut qu'en greedy).
    fn from_env(greedy: bool) -> Result<(Self, bool)> {
        Self::from_flag(
            crate::runtime_flags::dflash_tree_budget(),
            crate::runtime_flags::dflash_tree_default_budget(greedy),
        )
    }

    /// `None` : variable absente, arbre de `default_budget` nœuds ;
    /// `Some(Err(valeur))` : valeur invalide, refusée comme toute demande
    /// explicite hors périmètre.
    fn from_flag(
        flag: Option<std::result::Result<usize, String>>,
        default_budget: usize,
    ) -> Result<(Self, bool)> {
        match flag {
            None => Ok((Self::BestFirst(default_budget), false)),
            Some(Ok(0)) => Ok((Self::Off, true)),
            Some(Ok(budget)) => Ok((Self::BestFirst(budget), true)),
            Some(Err(value)) => Err(InferError::Config(format!(
                "SARAGOSSA_RUST_DFLASH_TREE={value:?} invalide (entier B ≥ 0)"
            ))),
        }
    }

    /// Lignes de vérification au plus, au-delà du bloc du drafter.
    fn budget(self) -> usize {
        match self {
            Self::Off => 0,
            #[cfg(test)]
            Self::Chain => 0,
            Self::BestFirst(budget) => budget,
        }
    }

    /// Lignes de vérification au plus d'un tour : `B + 1` pour l'arbre
    /// best-first (même si le bloc du drafter est plus long : l'arbre n'a que
    /// `B` nœuds), le bloc entier pour la chaîne passée par la plomberie arbre.
    #[cfg_attr(
        not(test),
        expect(
            unused_variables,
            reason = "le bloc ne compte que pour la chaîne de test"
        )
    )]
    fn verify_rows(self, draft_limit: usize) -> usize {
        match self {
            Self::Off => 0,
            #[cfg(test)]
            Self::Chain => draft_limit + 1,
            Self::BestFirst(budget) => budget + 1,
        }
    }
}

/// Raison du refus de la vérification en arbre (`None` : supportée), sur les
/// propriétés du modèle : couches glissantes, norme V (hors du chemin en lignes
/// fusionné), couches full-attn sans variante arbre (KV quantifié, sinks, chemin
/// en lignes fusionné désactivé), refus des états linéaires (cf.
/// `linear_tree_journal_refusal`).
fn dflash_tree_refusal(
    sliding: bool,
    value_norm: bool,
    full_supported: impl IntoIterator<Item = bool>,
    linear_refusals: impl IntoIterator<Item = Option<String>>,
) -> Option<String> {
    if sliding {
        return Some("couches à fenêtre glissante".to_string());
    }
    if value_norm {
        return Some("norme V hors du chemin en lignes fusionné".to_string());
    }
    if full_supported.into_iter().any(|supported| !supported) {
        return Some("KV quantifié, sinks ou chemin en lignes fusionné indisponible".to_string());
    }
    linear_refusals
        .into_iter()
        .flatten()
        .next()
        .map(|reason| format!("états linéaires : {reason}"))
}

/// Refus de l'arbre : erreur pour une demande explicite ; l'arbre par défaut
/// retombe sur la chaîne avec un diagnostic, pour qu'un bench ne mesure pas la
/// chaîne en croyant mesurer l'arbre.
fn tree_fallback(
    tree_mode: &mut DFlashTreeMode,
    explicit: bool,
    refusal: InferError,
) -> Result<()> {
    if explicit {
        return Err(refusal);
    }
    eprintln!("saragossa: arbre DFlash par défaut indisponible, chaîne ({refusal})");
    *tree_mode = DFlashTreeMode::Off;
    Ok(())
}

/// Lignes de hidden states cibles téléversées par command buffer (prefill du
/// contexte du drafter et append des lignes commitées).
const CONTEXT_UPLOAD_ROWS: usize = 64;

#[cfg(test)]
thread_local! {
    /// Lignes de contexte données au drafter, enregistrées si `Some` (tests :
    /// le rassemblement des hidden du chemin arbre contre la chaîne).
    pub(in crate::decoder) static DFLASH_APPENDED_CONTEXT: std::cell::RefCell<Option<Vec<f32>>> =
        const { std::cell::RefCell::new(None) };
}

struct DFlashResidentBuffers {
    /// Kernel skinny bf16 pour les matmuls du drafter (−20 % sur le bloc).
    skinny_bf16: bool,
    context_upload: GpuTensor,
    noise: GpuTensor,
    states: GpuTensor,
    index: GpuTensor,
    value: GpuTensor,
    projected: Option<GpuTensor>,
    block: usize,
    top_k: usize,
    rank: usize,
    context_cols: usize,
}

impl CausalDecoder {
    /// Génère en greedy avec le drafter DFlash résident et la vérification
    /// résidente en lignes. Sorties identiques à l'AR résident de même précision
    /// KV (vérifieur exact) ; face à la boucle CPU f32, l'écart vient des matmuls
    /// de la proposition et, au-delà du seuil greedy, du KV bf16 du vérifieur
    /// (`resident_kv_bf16`), qui peut trancher autrement un quasi-ex aequo.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si l'échantillonnage (température > 0) n'est pas permis
    /// (`SARAGOSSA_RUST_DFLASH_SAMPLED=0` ou paramètres hors du sampler GPU), si le decode résident
    /// complet n'est pas disponible pour ce modèle, ou si un encodage échoue.
    pub fn generate_greedy_dflash_resident_with_options(
        &self,
        prompt: &[usize],
        max_new_tokens: usize,
        options: &GenerationOptions,
        draft: &DFlashDraft,
        max_draft_tokens: usize,
    ) -> Result<SpeculativeOutput> {
        self.generate_greedy_dflash_resident_streaming_with_options(
            prompt,
            max_new_tokens,
            options,
            draft,
            max_draft_tokens,
            |_| true,
        )
    }

    /// Comme `generate_greedy_dflash_resident_with_options` mais émet chaque
    /// token via `on_token` (qui renvoie `false` pour arrêter). Tout token poussé
    /// est définitif (vérifieur exact), donc émis aussitôt : la séquence reste
    /// byte-identique à la variante non streamée, le callback ne fait
    /// qu'observer ou arrêter.
    ///
    /// # Errors
    ///
    /// Mêmes erreurs que `generate_greedy_dflash_resident_with_options`.
    pub fn generate_greedy_dflash_resident_streaming_with_options(
        &self,
        prompt: &[usize],
        max_new_tokens: usize,
        options: &GenerationOptions,
        draft: &DFlashDraft,
        max_draft_tokens: usize,
        on_token: impl FnMut(usize) -> bool,
    ) -> Result<SpeculativeOutput> {
        self.generate_greedy_dflash_resident_tree(
            prompt,
            max_new_tokens,
            options,
            draft,
            max_draft_tokens,
            DFlashTreeMode::from_env(options.temperature <= f32::EPSILON)?,
            on_token,
        )
    }

    /// Boucle DFlash résidente, vérification en chaîne ou en arbre (`tree`).
    #[expect(
        clippy::too_many_arguments,
        reason = "options de génération, drafter, mode d'arbre et callback de streaming"
    )]
    pub(in crate::decoder) fn generate_greedy_dflash_resident_tree(
        &self,
        prompt: &[usize],
        max_new_tokens: usize,
        options: &GenerationOptions,
        draft: &DFlashDraft,
        max_draft_tokens: usize,
        (tree_mode, tree_explicit): (DFlashTreeMode, bool),
        mut on_token: impl FnMut(usize) -> bool,
    ) -> Result<SpeculativeOutput> {
        // Arbre par défaut (non explicite) : toute raison de refus le ramène à la
        // chaîne ; une demande explicite reste refusée avec son erreur.
        let mut tree_mode = tree_mode;
        if let Some(reason) = dflash_sampling_refusal(options) {
            return Err(InferError::Config(format!(
                "decode DFlash résident: {reason}"
            )));
        }
        if max_draft_tokens == 0 {
            return Err(InferError::Config(
                "decode DFlash résident sans draft token".to_string(),
            ));
        }
        if prompt.is_empty() {
            return Err(InferError::Dimension("prompt token vide".to_string()));
        }
        if options.token_constraint.is_some() {
            // Les cibles viennent de l'argmax GPU, hors `mask_logits`/`accept_token`.
            return Err(InferError::Config(
                "decode DFlash résident sans contrainte de tokens (sélecteur greedy)".to_string(),
            ));
        }
        self.validate_dflash_draft(draft)?;
        if max_new_tokens == 0 {
            return Ok(SpeculativeOutput {
                tokens: Vec::new(),
                stats: SpeculativeStats::default(),
                loop_duration: Duration::default(),
            });
        }
        let Some(metal) = self.forward_runtime().metal_executor() else {
            return Err(InferError::Config(
                "decode DFlash résident sans exécuteur Metal".to_string(),
            ));
        };
        let draft_limit = max_draft_tokens
            .min(draft.info.block_size)
            .min(max_new_tokens);
        let block = draft_limit
            .checked_add(1)
            .ok_or_else(|| InferError::Shape("DFlash bloc déborde".to_string()))?;
        if let DFlashTreeMode::BestFirst(budget) = tree_mode {
            // Arbre du sélecteur DFlash 2, au plus `B + 1` lignes dans le journal
            // linéaire (16, vérifié couche par couche plus bas).
            if draft.selector.is_none() || budget >= super::dflash_tree::DFLASH_TREE_MAX_ROWS {
                tree_fallback(
                    &mut tree_mode,
                    tree_explicit,
                    InferError::Config(format!(
                        "decode DFlash résident: arbre B={budget} exige DFlash 2 (sélecteur) et B < {}",
                        super::dflash_tree::DFLASH_TREE_MAX_ROWS
                    )),
                )?;
            }
        }
        // La trace d'arbre décrit la chaîne gloutonne : refus avant d'ouvrir
        // (et tronquer) le fichier plutôt qu'une trace incohérente.
        if tree_mode != DFlashTreeMode::Off
            && crate::runtime_flags::dflash_tree_trace_path().is_some()
        {
            tree_fallback(
                &mut tree_mode,
                tree_explicit,
                InferError::Config(
                    "decode DFlash résident: SARAGOSSA_RUST_DFLASH_TREE et SARAGOSSA_RUST_DFLASH_TREE_TRACE sont exclusifs".to_string(),
                ),
            )?;
        }
        // Après les replis qui précèdent l'allocation : une chaîne repliée réserve
        // les lignes de la chaîne. Le repli sur le support du modèle vient après
        // l'allocation (il lit l'arène) et garde la réserve de l'arbre.
        let tree_budget = tree_mode.budget();
        let layer_ids = draft.info.target_layer_ids.as_slice();
        let hidden = self.final_norm.len();
        let context_cols = layer_ids
            .len()
            .checked_mul(hidden)
            .ok_or_else(|| InferError::Shape("DFlash contexte déborde".to_string()))?;

        // Toutes les couches fenêtrées : seules les `window` dernières positions du
        // prompt sont attendues, les précédentes ne sont jamais lues (ni gardées
        // dans le cache de préfixe).
        let window = drafter_window(&draft.info.layer_types, draft.info.sliding_window);
        let fixed_base = window.is_some() && crate::runtime_flags::dflash_drafter_base();
        let keep_rows = window.unwrap_or(usize::MAX);
        // Réserve du decode demandée dès le prefill : sans elle, la croissance
        // bornée au contexte du modèle peut laisser le K/V trop court pour être
        // adopté, d'où une recopie complète près de la fenêtre.
        let decode_reserve = max_new_tokens + draft_limit.max(tree_budget) + 1;
        let (mut cache, final_state, target_hidden, captured_from) =
            crate::metal_backend::with_prefill_kv_reserve(decode_reserve, || {
                self.prefill_cache_state_with_layer_capture_cached(
                    prompt,
                    layer_ids,
                    keep_rows,
                    Some(draft),
                )
            })?;
        let (captured_rows, captured_cols) = target_hidden.as_matrix()?;
        if captured_cols != context_cols {
            return Err(InferError::Dimension(
                "DFlash captures: largeur invalide".to_string(),
            ));
        }
        let (skip, capacity) = drafter_layout(
            prompt.len(),
            max_new_tokens,
            block,
            window,
            captured_from,
            captured_rows,
            fixed_base,
        )?;
        // Même règle KV que la vérification MTP (`resident_kv_bf16`) : bf16 au-delà du
        // seuil greedy, où la lecture du KV domine la vérification en lignes. Le
        // drafter garde son propre KV dense ; la boucle CPU de référence reste f32.
        let kv_bf16 = Self::resident_kv_bf16(options.temperature > f32::EPSILON, cache.position);
        if self.config.metal_unsupported_reason().is_some()
            || !self.supports_resident_full_decode()
            || !self.setup_resident_full_decode(&mut cache, decode_reserve, 0, kv_bf16)?
        {
            return Err(InferError::Config(
                "decode DFlash résident: decode résident complet indisponible".to_string(),
            ));
        }
        // Le rollback par troncature exige la vérification batchée (captures) ; le
        // repli séquentiel (`SARAGOSSA_RUST_PREFILL_MOE_ROWS=0`, ou couches full-attn
        // MoE sans `SARAGOSSA_RUST_MOE_VERIFY_ROWS=1`) n'en fournit pas.
        if !cache
            .resident
            .as_ref()
            .is_some_and(|arena| self.resident_layer_batch_supported(arena))
        {
            return Err(InferError::Config(
                "decode DFlash résident: vérification résidente batchée indisponible (couches en lignes requises)"
                    .to_string(),
            ));
        }
        if tree_mode != DFlashTreeMode::Off {
            if let Err(error) =
                self.check_dflash_tree_supported(&cache, tree_mode.verify_rows(draft_limit))
            {
                tree_fallback(&mut tree_mode, tree_explicit, error)?;
            }
        }
        let top_k = draft.selector.as_ref().map_or(1, |selector| selector.top_k);
        let rank = draft.selector.as_ref().map_or(0, |selector| selector.rank);
        let (mut drafter, buffers) = {
            let arena = cache.resident.as_mut().ok_or_else(|| {
                InferError::Config("decode DFlash résident: arène absente".to_string())
            })?;
            let drafter = DFlashResident::new(&arena.state, metal, draft, capacity)?;
            let state = &mut arena.state;
            let buffers = DFlashResidentBuffers {
                skinny_bf16: crate::runtime_flags::env_flag(
                    "SARAGOSSA_RUST_DFLASH_SKINNY_BF16",
                    true,
                ),
                context_upload: state
                    .persistent(CONTEXT_UPLOAD_ROWS * context_cols, GpuElement::F32)?,
                noise: state.persistent(block * hidden, GpuElement::F32)?,
                states: state.persistent(block * hidden, GpuElement::F32)?,
                index: state.persistent((block - 1) * top_k, GpuElement::U32)?,
                value: state.persistent((block - 1) * top_k, GpuElement::F32)?,
                projected: (rank > 0)
                    .then(|| state.persistent(block * rank, GpuElement::F32))
                    .transpose()?,
                block,
                top_k,
                rank,
                context_cols,
            };
            (drafter, buffers)
        };
        // Les captures ne couvrent que `captured_from..prompt.len()` : le contexte
        // du drafter commence au plus tôt là.
        if fixed_base {
            drafter.rebase(skip)?;
        } else {
            drafter.skip_context(skip)?;
        }
        self.dflash_resident_append_context(
            &mut cache,
            metal,
            &mut drafter,
            &buffers,
            &target_hidden,
            skip - captured_from,
            prompt.len() - captured_from,
        )?;
        // L'append attend la fin des commandes Metal : le KV du drafter suffit
        // désormais, sans garder les captures du prompt pendant le decode.
        drop(target_hidden);

        let mut sampler = DeterministicSampler::new(options.seed);
        let mut pending = self.sample_token_from_state(&final_state, options, &mut sampler)?;
        let mut generated = Vec::with_capacity(max_new_tokens);
        let mut stats = SpeculativeStats::default();
        stats.proposed_by_position.resize(draft_limit, 0);
        stats.accepted_by_position.resize(draft_limit, 0);
        let loop_started = Instant::now();
        // Profil par segment du tour (devtools) : draft, vérification, rollback,
        // append du contexte.
        let profile = crate::runtime_flags::env_flag("SARAGOSSA_RUST_DFLASH_PROFILE", false);
        // Bascule par acceptance observée : sous `switch_min` tokens par tour en
        // moyenne après `switch_warmup` tours, la boucle rend la main (le reste de
        // la génération revient à la boucle MTP, cf. harnais d'acceptance).
        // NOTE: le repli MTP repart avec une sortie vide et manquerait une stop
        // sequence à cheval sur la bascule : pas de bascule avec des séquences.
        // À T > 0 le repli MTP repartirait de la graine et rejouerait les aléas
        // déjà consommés : pas de bascule non plus.
        let switch_min = if options.stop_sequences.is_empty() && options.temperature <= f32::EPSILON
        {
            crate::runtime_flags::dflash_switch_min()
        } else {
            0.0
        };
        let switch_warmup = crate::runtime_flags::env_var("SARAGOSSA_RUST_DFLASH_SWITCH_WARMUP")
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(6)
            .max(1);
        let mut segments = [std::time::Duration::ZERO; 5];
        let mut tours = 0usize;
        let mut tree_trace = match crate::runtime_flags::dflash_tree_trace_path() {
            Some(path) if draft.selector.is_some() => Some((
                std::io::BufWriter::new(std::fs::File::create(&path).map_err(|source| {
                    InferError::Io {
                        path: path.clone(),
                        source,
                    }
                })?),
                path,
            )),
            _ => None,
        };

        while generated.len() < max_new_tokens {
            let primary = pending;
            if options.stop_token_ids.contains(&primary) {
                break;
            }
            if switch_min > 0.0
                && tours >= switch_warmup
                && (generated.len() as f64) < switch_min * tours as f64
            {
                stats.dflash_switched_at = Some(generated.len());
                break;
            }
            if !commit_token(&mut generated, primary, options, &mut on_token)
                || generated.len() >= max_new_tokens
            {
                break;
            }

            tours += 1;
            if tree_mode != DFlashTreeMode::Off {
                let tour = self.dflash_resident_tree_tour(
                    &mut cache,
                    metal,
                    &mut drafter,
                    &buffers,
                    draft,
                    primary,
                    draft_limit,
                    (tree_mode, max_new_tokens),
                    options,
                    &mut DFlashTourState {
                        generated: &mut generated,
                        stats: &mut stats,
                        pending: &mut pending,
                        segments: &mut segments,
                        on_token: &mut on_token,
                    },
                )?;
                if tour == TourEnd::Stop || generated.len() >= max_new_tokens {
                    break;
                }
                continue;
            }
            let mut segment_started = Instant::now();
            let (drafts, lattice) = self.dflash_resident_drafts(
                &mut cache,
                metal,
                &mut drafter,
                &buffers,
                draft,
                primary,
                draft_limit,
                tree_trace.is_some(),
            )?;
            if let (Some((writer, path)), Some(mut lattice)) = (tree_trace.as_mut(), lattice) {
                lattice["anchor_index"] = (generated.len() - 1).into();
                lattice["drafts"] = drafts.clone().into();
                write_trace_line(writer, path, &lattice)?;
            }
            segments[0] += segment_started.elapsed();
            segment_started = Instant::now();
            // Bloc de vérification = ancre + tous les drafts : la dernière ligne
            // fournit le token suivant quand tout est accepté, sans pas bonus
            // séquentiel (un forward complet, ~33 ms, par tour entièrement accepté).
            let mut verify_tokens = Vec::with_capacity(drafts.len() + 1);
            verify_tokens.push(primary);
            verify_tokens.extend(drafts.iter().copied());
            let output = self
                .next_final_states_resident_verify(
                    &mut cache,
                    &verify_tokens,
                    dflash_verify_head(options, generated.len()),
                    true,
                    Some(layer_ids),
                    None,
                )?
                .ok_or_else(|| {
                    InferError::Config(
                        "decode DFlash résident: vérification résidente indisponible".to_string(),
                    )
                })?;
            segments[1] += segment_started.elapsed();
            let targets = output.tokens.ok_or_else(|| {
                InferError::Config("decode DFlash résident: argmax résident absent".to_string())
            })?;
            let committed_hidden = output.target_hidden.ok_or_else(|| {
                InferError::Config("decode DFlash résident: capture cible absente".to_string())
            })?;
            let mut accepted_all = true;
            let mut stopped = false;
            let mut committed_len = 0usize;
            for (position, draft_token) in drafts.iter().copied().enumerate() {
                let target = *targets.get(position).ok_or_else(|| {
                    InferError::Dimension(format!(
                        "decode DFlash résident: {} cibles pour la position {position}",
                        targets.len()
                    ))
                })?;
                stats.verifications += 1;
                stats.proposed += 1;
                if let Some(slot) = stats.proposed_by_position.get_mut(position) {
                    *slot += 1;
                }
                committed_len = position + 1;
                if draft_token == target {
                    stats.accepted += 1;
                    if let Some(slot) = stats.accepted_by_position.get_mut(position) {
                        *slot += 1;
                    }
                    if options.stop_token_ids.contains(&draft_token) {
                        // Stop token accepté : exclu de la sortie comme en AR et
                        // MTP ; le tour suivant s'arrête sur ce `pending`.
                        pending = draft_token;
                        accepted_all = false;
                        break;
                    }
                    if !commit_token(&mut generated, draft_token, options, &mut on_token) {
                        stopped = true;
                        accepted_all = false;
                        break;
                    }
                    if generated.len() >= max_new_tokens {
                        accepted_all = false;
                        break;
                    }
                } else {
                    stats.rejected += 1;
                    pending = target;
                    accepted_all = false;
                    break;
                }
            }
            if stopped {
                break;
            }
            if accepted_all {
                pending = *targets.get(drafts.len()).ok_or_else(|| {
                    InferError::Dimension(format!(
                        "decode DFlash résident: {} cibles pour le token suivant",
                        targets.len()
                    ))
                })?;
                committed_len = verify_tokens.len();
            }
            if committed_len < verify_tokens.len() {
                stats.rollbacks += 1;
                let captures = output.captures.as_ref().ok_or_else(|| {
                    InferError::Config(
                        "decode DFlash résident: captures linéaires absentes pour le rollback"
                            .to_string(),
                    )
                })?;
                segment_started = Instant::now();
                self.restore_mtp_resident_progress(&mut cache, captures, committed_len)?;
                segments[2] += segment_started.elapsed();
            }
            segment_started = Instant::now();
            self.dflash_resident_append_context(
                &mut cache,
                metal,
                &mut drafter,
                &buffers,
                &committed_hidden,
                0,
                committed_len,
            )?;
            segments[3] += segment_started.elapsed();
            if generated.len() >= max_new_tokens {
                break;
            }
        }
        if let Some((mut writer, path)) = tree_trace {
            write_trace_line(
                &mut writer,
                &path,
                &serde_json::json!({ "prompt_len": prompt.len(), "generated": generated }),
            )?;
            std::io::Write::flush(&mut writer).map_err(|source| InferError::Io { path, source })?;
        }
        if profile && tours > 0 {
            let per_tour = |d: std::time::Duration| d.as_secs_f64() * 1e3 / tours as f64;
            // Arbre : construction CPU comptée à part (incluse dans draft).
            let tree = if tree_mode == DFlashTreeMode::Off {
                String::new()
            } else {
                format!(" tree_build={:.2}", per_tour(segments[4]))
            };
            eprintln!(
                "dflash_profile tours={tours} ms/tour draft={:.1} verify={:.1} rollback={:.1} append={:.1} total={:.1}{tree}",
                per_tour(segments[0]),
                per_tour(segments[1]),
                per_tour(segments[2]),
                per_tour(segments[3]),
                loop_started.elapsed().as_secs_f64() * 1e3 / tours as f64,
            );
        }

        Ok(SpeculativeOutput {
            tokens: generated,
            stats,
            loop_duration: loop_started.elapsed(),
        })
    }

    /// Un tour du drafter : embeddings ancre + masques, bloc résident, top-k et
    /// projection, parcours du sélecteur sur CPU. Renvoie `draft_limit` tokens et,
    /// si `lattice` (diagnostic d'arbre), les candidats et le treillis des scores.
    #[expect(
        clippy::too_many_arguments,
        reason = "cache résident, exécuteur, drafter, buffers persistants et draft"
    )]
    fn dflash_resident_drafts(
        &self,
        cache: &mut CausalDecoderCache,
        metal: &MetalExecutor,
        drafter: &mut DFlashResident,
        buffers: &DFlashResidentBuffers,
        draft: &DFlashDraft,
        primary: usize,
        draft_limit: usize,
        lattice: bool,
    ) -> Result<(Vec<usize>, Option<serde_json::Value>)> {
        let tokens = self.dflash_resident_block(cache, metal, drafter, buffers, draft, primary)?;
        let block = buffers.block;
        let count = (block - 1) * buffers.top_k;
        let mut trace = None;
        let drafts = match (draft.selector.as_ref(), buffers.projected.as_ref()) {
            (Some(selector), Some(projected)) => {
                let values = read_f32_buffer(buffers.value.buffer(), count)?;
                let candidates = (0..block - 1)
                    .map(|row| {
                        (0..buffers.top_k)
                            .map(|k| {
                                (
                                    tokens[row * buffers.top_k + k],
                                    values[row * buffers.top_k + k],
                                )
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                let projection = read_f32_buffer(projected.buffer(), block * buffers.rank)?;
                let projection = &projection[buffers.rank..];
                let drafts = selector.select_projected(projection, &candidates, primary)?;
                if lattice {
                    let scores = selector.transition_scores(projection, &candidates, primary)?;
                    let tokens: Vec<Vec<usize>> = candidates
                        .iter()
                        .map(|row| row.iter().map(|&(token, _)| token).collect())
                        .collect();
                    trace = Some(serde_json::json!({ "candidates": tokens, "scores": scores }));
                }
                drafts
            }
            // DFlash 1 : argmax par position (top-1).
            _ => tokens.iter().step_by(buffers.top_k).copied().collect(),
        };
        Ok((drafts.into_iter().take(draft_limit).collect(), trace))
    }

    /// Bloc du drafter sur GPU (embeddings ancre + masques, bloc résident, top-k
    /// et projection) ; rend les `top_k` tokens candidats de chaque position, les
    /// valeurs et la projection restant dans `buffers`.
    fn dflash_resident_block(
        &self,
        cache: &mut CausalDecoderCache,
        metal: &MetalExecutor,
        drafter: &mut DFlashResident,
        buffers: &DFlashResidentBuffers,
        draft: &DFlashDraft,
        primary: usize,
    ) -> Result<Vec<usize>> {
        let block = buffers.block;
        let mut noise_tokens = Vec::with_capacity(block);
        noise_tokens.push(primary);
        noise_tokens.resize(block, draft.info.mask_token_id);
        let noise = self.embed_scaled(&noise_tokens)?;
        let arena = cache.resident.as_mut().ok_or_else(|| {
            InferError::Config("decode DFlash résident: arène absente".to_string())
        })?;
        arena.state.upload(&buffers.noise, noise.data())?;
        let command_buffer = arena.state.queue().new_command_buffer();
        let _barrier_guard = crate::metal_backend::resident_concurrent_enabled()
            .then(crate::metal_backend::install_dispatch_barrier_scope);
        let _namespace_guard =
            crate::metal_backend::install_scratch_namespace(arena.state.scratch_namespace());
        let _skinny_guard =
            crate::metal_backend::install_qmm_skinny_bf16_scope(buffers.skinny_bf16);
        let encoder = new_resident_compute_encoder(command_buffer);
        let mut owned = Vec::new();
        drafter.encode_block(
            &arena.state,
            metal,
            encoder,
            &mut owned,
            buffers.noise.buffer(),
            block,
            buffers.states.buffer(),
        )?;
        drafter.encode_block_candidates(
            &arena.state,
            metal,
            encoder,
            &mut owned,
            buffers.states.buffer(),
            block,
            &arena.lm_head,
            buffers.top_k,
            buffers.index.buffer(),
            buffers.value.buffer(),
            buffers.projected.as_ref().map(|tensor| &**tensor.buffer()),
        )?;
        encoder.end_encoding();
        commit_and_wait(command_buffer)?;

        let count = (block - 1) * buffers.top_k;
        let indices = read_u32_buffer(buffers.index.buffer(), count)?;
        indices
            .iter()
            .map(|index| {
                usize::try_from(*index).map_err(|_| {
                    InferError::Metal(format!("drafter résident: index trop grand {index}"))
                })
            })
            .collect::<Result<Vec<_>>>()
    }

    /// Arbre best-first du tour : bloc du drafter puis construction CPU sur le
    /// treillis du sélecteur, une ligne de scores par nœud développé (le
    /// parcours glouton de la chaîne n'est pas calculé). Rend l'arbre et la durée
    /// de sa construction.
    #[expect(
        clippy::too_many_arguments,
        reason = "cache résident, exécuteur, drafter, buffers, draft et forme de l'arbre"
    )]
    fn dflash_resident_tree(
        &self,
        cache: &mut CausalDecoderCache,
        metal: &MetalExecutor,
        drafter: &mut DFlashResident,
        buffers: &DFlashResidentBuffers,
        draft: &DFlashDraft,
        primary: usize,
        budget: usize,
        max_depth: usize,
    ) -> Result<(DFlashTree, std::time::Duration)> {
        let tokens = self.dflash_resident_block(cache, metal, drafter, buffers, draft, primary)?;
        let started = Instant::now();
        let (Some(selector), Some(projected)) =
            (draft.selector.as_ref(), buffers.projected.as_ref())
        else {
            return Err(InferError::Config(
                "decode DFlash résident: arbre sans sélecteur DFlash 2".to_string(),
            ));
        };
        let positions = buffers.block - 1;
        let top_k = buffers.top_k;
        let values = read_f32_buffer(buffers.value.buffer(), positions * top_k)?;
        let candidates = (0..positions)
            .map(|row| {
                (0..top_k)
                    .map(|k| (tokens[row * top_k + k], values[row * top_k + k]))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let candidate_tokens = candidates
            .iter()
            .map(|row| row.iter().map(|&(token, _)| token).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let projection = read_f32_buffer(projected.buffer(), buffers.block * buffers.rank)?;
        // Ligne 0 de la projection : l'ancre, sans candidats (comme la chaîne).
        let projection = &projection[buffers.rank..];
        let rank = buffers.rank;
        let tree = DFlashTree::best_first(
            primary,
            &candidate_tokens,
            budget,
            max_depth,
            |position, previous| {
                selector.edge_scores_from(
                    &projection[position * rank..(position + 1) * rank],
                    previous,
                    &candidates[position],
                )
            },
        )?;
        Ok((tree, started.elapsed()))
    }

    /// Un tour en arbre : drafts, vérification des lignes de l'arbre, parcours
    /// d'acceptation depuis la racine, rollback le long du chemin accepté
    /// (toujours : la vérification laisse les états linéaires d'entrée intacts)
    /// et append au drafter des hidden des lignes du chemin.
    #[expect(
        clippy::too_many_arguments,
        reason = "cache résident, exécuteur, drafter, buffers, draft, forme et état du tour"
    )]
    fn dflash_resident_tree_tour(
        &self,
        cache: &mut CausalDecoderCache,
        metal: &MetalExecutor,
        drafter: &mut DFlashResident,
        buffers: &DFlashResidentBuffers,
        draft: &DFlashDraft,
        primary: usize,
        draft_limit: usize,
        (mode, max_new_tokens): (DFlashTreeMode, usize),
        options: &GenerationOptions,
        state: &mut DFlashTourState<'_, impl FnMut(usize) -> bool>,
    ) -> Result<TourEnd> {
        let mut segment_started = Instant::now();
        let tree = match mode {
            DFlashTreeMode::BestFirst(budget) => {
                // Pas de nœud au-delà des tokens encore permis.
                let max_depth = draft_limit.min(max_new_tokens - state.generated.len());
                let (tree, build) = self.dflash_resident_tree(
                    cache, metal, drafter, buffers, draft, primary, budget, max_depth,
                )?;
                state.segments[4] += build;
                tree
            }
            #[cfg(test)]
            DFlashTreeMode::Chain => {
                let (drafts, _) = self.dflash_resident_drafts(
                    cache,
                    metal,
                    drafter,
                    buffers,
                    draft,
                    primary,
                    draft_limit,
                    false,
                )?;
                DFlashTree::chain(primary, &drafts)?
            }
            DFlashTreeMode::Off => {
                return Err(InferError::Config(
                    "decode DFlash résident: tour en arbre sans arbre".to_string(),
                ));
            }
        };
        state.segments[0] += segment_started.elapsed();
        segment_started = Instant::now();
        let verify_tree = std::rc::Rc::new(crate::metal_backend::VerifyTree::new(
            tree.parents(),
            tree.depths(),
            tree.masks(),
        )?);
        let output = self
            .next_final_states_resident_verify(
                cache,
                tree.tokens(),
                dflash_verify_head(options, state.generated.len()),
                true,
                Some(draft.info.target_layer_ids.as_slice()),
                Some(verify_tree),
            )?
            .ok_or_else(|| {
                InferError::Config(
                    "decode DFlash résident: vérification en arbre indisponible".to_string(),
                )
            })?;
        state.segments[1] += segment_started.elapsed();
        let targets = output.tokens.as_ref().ok_or_else(|| {
            InferError::Config("decode DFlash résident: argmax résident absent".to_string())
        })?;
        let hidden = output.target_hidden.as_ref().ok_or_else(|| {
            InferError::Config("decode DFlash résident: capture cible absente".to_string())
        })?;
        let captures = output.captures.as_ref().ok_or_else(|| {
            InferError::Config(
                "decode DFlash résident: captures linéaires absentes pour le rollback".to_string(),
            )
        })?;
        if targets.len() != tree.len() {
            return Err(InferError::Dimension(format!(
                "decode DFlash résident: {} cibles pour {} lignes d'arbre",
                targets.len(),
                tree.len()
            )));
        }
        let Some(path) = walk_tree_tour(&tree, targets, max_new_tokens, options, state) else {
            return Ok(TourEnd::Stop);
        };
        if path.len() < tree.len() {
            state.stats.rollbacks += 1;
        }
        segment_started = Instant::now();
        self.restore_tree_resident_progress(cache, captures, &path)?;
        state.segments[2] += segment_started.elapsed();
        segment_started = Instant::now();
        // Contexte du drafter : hidden des lignes commitées (ancre comprise,
        // sans le `pending`), dans l'ordre du chemin.
        let (_, cols) = hidden.as_matrix()?;
        let mut gathered = Vec::with_capacity(path.len() * cols);
        for &row in &path {
            gathered.extend_from_slice(hidden.row_slice(row)?);
        }
        let gathered = Tensor::from_vec(vec![path.len(), cols], gathered)?;
        self.dflash_resident_append_context(
            cache,
            metal,
            drafter,
            buffers,
            &gathered,
            0,
            path.len(),
        )?;
        state.segments[3] += segment_started.elapsed();
        Ok(TourEnd::Continue)
    }

    /// Rollback d'un tour en arbre le long de `path` (lignes acceptées, racine
    /// comprise) : KV full-attn compacté au bout du préfixe, états linéaires
    /// rejoués depuis leur état d'entrée, position = base + `path.len()`.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le chemin est vide ou invalide, ou si la
    /// restauration GPU échoue.
    fn restore_tree_resident_progress(
        &self,
        cache: &mut CausalDecoderCache,
        captures: &ResidentVerifyCaptures,
        path: &[usize],
    ) -> Result<()> {
        cache.position = captures
            .base_position
            .checked_add(path.len())
            .ok_or_else(|| InferError::Dimension("position rollback arbre déborde".to_string()))?;
        for layer in &mut cache.layers {
            if let Some(full) = layer.full.as_mut() {
                full.compact_tree_rows(captures.base_position, path)?;
            }
        }
        let Some(metal) = self.forward_runtime().metal_executor() else {
            return Ok(());
        };
        let layers = cache
            .layers
            .iter()
            .zip(captures.linear.iter())
            .filter_map(|(layer, captures)| {
                Some((layer.linear.metal_state()?, captures.as_deref()?))
            })
            .collect::<Vec<_>>();
        metal.restore_linear_attn_tree(&layers, path)
    }

    /// Refus avant la boucle des configurations sans variante arbre : couches
    /// glissantes, KV quantifié ou à sinks, chemin en lignes fusionné désactivé,
    /// états linéaires hors du journal arbre pour `rows` lignes.
    fn check_dflash_tree_supported(&self, cache: &CausalDecoderCache, rows: usize) -> Result<()> {
        let sliding =
            (0..self.layers.len()).any(|index| self.config.layer_sliding_window(index).is_some());
        let full = cache.layers.iter().filter_map(|layer| layer.full.as_ref());
        let refusal = dflash_tree_refusal(
            sliding,
            self.config.attention_value_norm,
            full.clone().map(|full| full.verify_tree_supported()),
            cache.layers.iter().filter_map(|layer| {
                layer
                    .linear
                    .metal_state()
                    .map(|state| crate::metal_backend::linear_tree_journal_refusal(state, rows))
            }),
        );
        if let Some(reason) = refusal {
            return Err(InferError::Config(format!(
                "decode DFlash résident: vérification en arbre non supportée ({reason})"
            )));
        }
        // Compilation des kernels arbre ici, hors encodeur et hors du tour 1.
        let mut head_dims = full.map(|full| full.head_dim()).collect::<Vec<_>>();
        head_dims.sort_unstable();
        head_dims.dedup();
        cache
            .resident
            .as_ref()
            .ok_or_else(|| InferError::Config("decode DFlash résident: arène absente".to_string()))?
            .state
            .warm_verify_tree_kernels(&head_dims)
    }

    /// Projette les lignes `start..end` de `hidden` (`[rows, context_cols]`) dans
    /// le contexte du drafter, par tranches de [`CONTEXT_UPLOAD_ROWS`].
    #[expect(
        clippy::too_many_arguments,
        reason = "cache résident, exécuteur, drafter, buffers et plage de lignes"
    )]
    fn dflash_resident_append_context(
        &self,
        cache: &mut CausalDecoderCache,
        metal: &MetalExecutor,
        drafter: &mut DFlashResident,
        buffers: &DFlashResidentBuffers,
        hidden: &Tensor,
        start: usize,
        end: usize,
    ) -> Result<()> {
        let (rows, cols) = hidden.as_matrix()?;
        if cols != buffers.context_cols || end > rows || start > end {
            return Err(InferError::Dimension(format!(
                "drafter résident: contexte {:?}, lignes {start}..{end}, attendu [*, {}]",
                hidden.shape(),
                buffers.context_cols
            )));
        }
        #[cfg(test)]
        DFLASH_APPENDED_CONTEXT.with(|slot| {
            if let Some(rows) = slot.borrow_mut().as_mut() {
                rows.extend_from_slice(&hidden.data()[start * cols..end * cols]);
            }
        });
        let arena = cache.resident.as_mut().ok_or_else(|| {
            InferError::Config("decode DFlash résident: arène absente".to_string())
        })?;
        let mut row = start;
        while row < end {
            let chunk = (end - row).min(CONTEXT_UPLOAD_ROWS);
            arena.state.upload(
                &buffers.context_upload,
                &hidden.data()[row * cols..(row + chunk) * cols],
            )?;
            let command_buffer = arena.state.queue().new_command_buffer();
            let _barrier_guard = crate::metal_backend::resident_concurrent_enabled()
                .then(crate::metal_backend::install_dispatch_barrier_scope);
            let _namespace_guard =
                crate::metal_backend::install_scratch_namespace(arena.state.scratch_namespace());
            let _skinny_guard =
                crate::metal_backend::install_qmm_skinny_bf16_scope(buffers.skinny_bf16);
            let encoder = new_resident_compute_encoder(command_buffer);
            let mut owned = Vec::new();
            drafter.encode_context_append(
                &arena.state,
                metal,
                encoder,
                &mut owned,
                buffers.context_upload.buffer(),
                chunk,
            )?;
            encoder.end_encoding();
            commit_and_wait(command_buffer)?;
            row += chunk;
        }
        Ok(())
    }
}

/// Fin d'un tour en arbre.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TourEnd {
    Continue,
    /// Arrêt : callback de streaming, stop sequence ou limite de sortie.
    Stop,
}

/// État de la génération que modifie un tour en arbre.
struct DFlashTourState<'a, F> {
    generated: &'a mut Vec<usize>,
    stats: &'a mut SpeculativeStats,
    pending: &'a mut usize,
    segments: &'a mut [std::time::Duration; 5],
    on_token: &'a mut F,
}

/// Raison du refus de l'échantillonnage DFlash (`None` : greedy, ou
/// échantillonnage permis) : le flag opt-in et des paramètres que le sampler
/// GPU de l'AR sait tirer (sinon l'AR relit les logits et tire sur le CPU, et
/// l'identité à graine fixée avec l'AR se perdrait).
pub(in crate::decoder) fn dflash_sampling_refusal(
    options: &GenerationOptions,
) -> Option<&'static str> {
    if options.temperature <= f32::EPSILON {
        return None;
    }
    if !crate::runtime_flags::dflash_sampled_enabled() {
        return Some("échantillonnage (temperature > 0) coupé par SARAGOSSA_RUST_DFLASH_SAMPLED=0");
    }
    if !super::resident_sampling_on_device(options) {
        return Some(
            "échantillonnage hors du sampler GPU (top_k entre 1 et 32, ou top_k = 0 et top_p = 1)",
        );
    }
    None
}

/// Tête de vérification d'un tour dont l'ancre est le token d'indice
/// `generated - 1` : argmax en greedy ; en échantillonné, la ligne de
/// profondeur `d` produit le token d'indice `generated + d` et tire avec l'aléa
/// de cet indice, celui que l'AR de même graine utiliserait (un tirage par
/// token, le premier consommé au prefill).
fn dflash_verify_head(
    options: &GenerationOptions,
    generated: usize,
) -> crate::decoder::ResidentVerifyHead {
    if options.temperature <= f32::EPSILON {
        return crate::decoder::ResidentVerifyHead::Argmax;
    }
    crate::decoder::ResidentVerifyHead::Sample {
        temperature: options.temperature,
        top_p: options.top_p,
        top_k: options.top_k,
        seed: options.seed,
        first_draw: generated,
    }
}

/// Parcours d'acceptation d'un tour en arbre sur les cibles de chaque ligne
/// (argmax en greedy, tirage en échantillonné) : descente vers l'enfant dont le
/// token égale la cible du nœud courant, arrêt sur une feuille (bonus), un rejet
/// ou un stop token accepté, qui deviennent `pending`. Rend le chemin commité
/// (racine comprise), ou `None` si la génération s'arrête dans le tour
/// (callback, stop sequence, limite de sortie).
fn walk_tree_tour(
    tree: &DFlashTree,
    targets: &[usize],
    max_new_tokens: usize,
    options: &GenerationOptions,
    state: &mut DFlashTourState<'_, impl FnMut(usize) -> bool>,
) -> Option<Vec<usize>> {
    // Parcours : même acceptation que la chaîne, nœud par nœud ; les
    // statistiques par position comptent les profondeurs parcourues.
    let mut path = vec![0usize];
    let mut node = 0usize;
    loop {
        let target = targets[node];
        let depth = tree.depths()[node];
        if !tree.parents()[node + 1..].contains(&node) {
            // Feuille : sa cible devient le token suivant (bonus).
            *state.pending = target;
            break;
        }
        state.stats.verifications += 1;
        state.stats.proposed += 1;
        if let Some(slot) = state.stats.proposed_by_position.get_mut(depth) {
            *slot += 1;
        }
        let Some(child) = tree.child(node, target) else {
            state.stats.rejected += 1;
            *state.pending = target;
            break;
        };
        state.stats.accepted += 1;
        if let Some(slot) = state.stats.accepted_by_position.get_mut(depth) {
            *slot += 1;
        }
        if options.stop_token_ids.contains(&target) {
            // Stop token accepté : exclu de la sortie comme en chaîne ; le
            // tour suivant s'arrête sur ce `pending`.
            *state.pending = target;
            break;
        }
        // NOTE: arrêt (callback, stop sequence ou limite de sortie) : sortie
        // sans rollback ni append. Le cache est local à la génération et
        // détruit avec elle ; ses états (KV non compacté, états linéaires
        // d'entrée) ne sont plus relus.
        if !commit_token(state.generated, target, options, state.on_token) {
            return None;
        }
        if state.generated.len() >= max_new_tokens {
            return None;
        }
        path.push(child);
        node = child;
    }
    Some(path)
}

/// Pousse un token définitif, l'émet, puis dit si la génération continue :
/// `false` si le callback arrête ou si une stop sequence vient de se terminer
/// (incluse dans la sortie, comme en AR). La limite de sortie reste à
/// l'appelant ; les stop tokens, exclus, sont testés avant.
fn commit_token(
    generated: &mut Vec<usize>,
    token: usize,
    options: &GenerationOptions,
    on_token: &mut impl FnMut(usize) -> bool,
) -> bool {
    generated.push(token);
    on_token(token)
        && !crate::decoder::generation::generated_matches_stop_sequence(
            generated,
            &options.stop_sequences,
        )
}

/// Une ligne JSONL de la trace d'arbre (`SARAGOSSA_RUST_DFLASH_TREE_TRACE`).
fn write_trace_line(
    writer: &mut impl std::io::Write,
    path: &std::path::Path,
    value: &serde_json::Value,
) -> Result<()> {
    serde_json::to_writer(&mut *writer, value)
        .map_err(std::io::Error::from)
        .and_then(|()| writer.write_all(b"\n"))
        .map_err(|source| InferError::Io {
            path: path.to_path_buf(),
            source,
        })
}

#[cfg(test)]
#[path = "dflash_sampled_tests.rs"]
mod sampled_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drafter_base_layout_cpu() -> Result<()> {
        let (window, max_new, block) = (64, 128, 9);
        for prompt in [3usize, 63, 64, 65, 4096] {
            let skip = prompt.saturating_sub(window);
            for captured_from in [0, skip] {
                assert_eq!(
                    drafter_layout(
                        prompt,
                        max_new,
                        block,
                        Some(window),
                        captured_from,
                        prompt - captured_from,
                        true
                    )?,
                    (skip, prompt - skip + max_new + block + 1)
                );
                assert_eq!(
                    drafter_layout(
                        prompt,
                        max_new,
                        block,
                        Some(window),
                        captured_from,
                        prompt - captured_from,
                        false
                    )?,
                    (skip, prompt + max_new + block + 1)
                );
            }
            assert_eq!(
                drafter_layout(prompt, max_new, block, None, 0, prompt, true)?,
                (0, prompt + max_new + block + 1)
            );
        }
        Ok(())
    }

    #[test]
    fn drafter_base_refusals_cpu() {
        for (prompt, from, rows) in [(100, 37, 63), (100, 36, 63), (100, 101, 0), (3, 1, 2)] {
            for enabled in [false, true] {
                assert!(matches!(
                    drafter_layout(prompt, 8, 4, Some(64), from, rows, enabled),
                    Err(InferError::Dimension(_))
                ));
            }
        }
        assert!(matches!(
            drafter_layout(100, 8, 4, None, 1, 99, true),
            Err(InferError::Dimension(_))
        ));
        assert!(matches!(
            drafter_layout(100, 8, 4, Some(64), 36, usize::MAX, true),
            Err(InferError::Dimension(_))
        ));
        for (prompt, max_new, block) in [
            (usize::MAX, usize::MAX, 0),
            (1, usize::MAX, 1),
            (1, 0, usize::MAX),
            (1, 0, usize::MAX - 1),
        ] {
            assert!(matches!(
                drafter_layout(prompt, max_new, block, Some(64), 0, prompt, true),
                Err(InferError::Shape(_))
            ));
        }
    }

    #[test]
    fn drafter_base_eligibility_cpu() {
        let sliding = vec!["sliding_attention".to_string(); 2];
        assert_eq!(drafter_window(&sliding, Some(64)), Some(64));
        let mixed = vec![
            "sliding_attention".to_string(),
            "full_attention".to_string(),
        ];
        assert_eq!(drafter_window(&mixed, Some(64)), None);
    }

    #[test]
    fn verify_rows_follow_the_tree_budget() {
        // Bloc long (16 drafts) et B = 8 : 9 lignes, pas 17.
        assert_eq!(DFlashTreeMode::BestFirst(8).verify_rows(16), 9);
        assert_eq!(DFlashTreeMode::BestFirst(15).verify_rows(8), 16);
        assert_eq!(DFlashTreeMode::Chain.verify_rows(16), 17);
        assert_eq!(DFlashTreeMode::Off.verify_rows(8), 0);
    }

    /// Variable absente = arbre du budget par défaut, non explicite ; 0, B et
    /// valeur invalide sont explicites et l'emportent sur le défaut ; un refus
    /// replie seulement le défaut. Indépendant de l'environnement.
    #[test]
    fn tree_default_falls_back_and_explicit_refuses() {
        assert_eq!(
            DFlashTreeMode::from_flag(None, 8).ok(),
            Some((DFlashTreeMode::BestFirst(8), false))
        );
        assert_eq!(
            DFlashTreeMode::from_flag(None, 3).ok(),
            Some((DFlashTreeMode::BestFirst(3), false))
        );
        assert_eq!(
            DFlashTreeMode::from_flag(Some(Ok(8)), 3).ok(),
            Some((DFlashTreeMode::BestFirst(8), true))
        );
        assert_eq!(
            DFlashTreeMode::from_flag(Some(Ok(0)), 8).ok(),
            Some((DFlashTreeMode::Off, true))
        );
        assert_eq!(
            DFlashTreeMode::from_flag(Some(Ok(12)), 8).ok(),
            Some((DFlashTreeMode::BestFirst(12), true))
        );
        assert!(DFlashTreeMode::from_flag(Some(Err("huit".to_string())), 8).is_err());
        let refusal = || InferError::Config("refus".to_string());
        let mut mode = DFlashTreeMode::BestFirst(8);
        assert!(tree_fallback(&mut mode, false, refusal()).is_ok());
        assert_eq!(mode, DFlashTreeMode::Off);
        let mut mode = DFlashTreeMode::BestFirst(8);
        assert!(tree_fallback(&mut mode, true, refusal()).is_err());
        assert_eq!(mode, DFlashTreeMode::BestFirst(8));
    }

    #[test]
    fn commit_token_stops_after_emitting_a_stop_sequence() {
        let options = GenerationOptions {
            stop_sequences: vec![Vec::new(), vec![3, 4]],
            ..GenerationOptions::default()
        };
        let mut generated = Vec::new();
        let mut seen = Vec::new();
        let mut on_token = |token| {
            seen.push(token);
            true
        };
        let continues: Vec<bool> = [1, 3, 4]
            .into_iter()
            .map(|token| commit_token(&mut generated, token, &options, &mut on_token))
            .collect();
        assert_eq!(continues, [true, true, false]);
        assert_eq!(generated, [1, 3, 4]);
        assert_eq!(seen, [1, 3, 4]);
        let mut cancelled = Vec::new();
        assert!(!commit_token(
            &mut cancelled,
            1,
            &GenerationOptions::default(),
            &mut |_| false
        ));
        assert_eq!(cancelled, [1]);
    }

    #[test]
    fn tree_refusals_name_their_cause() {
        assert_eq!(
            dflash_tree_refusal(false, false, [true], [None, None]),
            None
        );
        let sliding = dflash_tree_refusal(true, false, [true], [None]);
        assert!(sliding.is_some_and(|r| r.contains("glissante")));
        let value_norm = dflash_tree_refusal(false, true, [true], [None]);
        assert!(value_norm.is_some_and(|r| r.contains("norme V")));
        let quantized = dflash_tree_refusal(false, false, [true, false], [None]);
        assert!(quantized.is_some_and(|r| r.contains("KV quantifié")));
        let linear = dflash_tree_refusal(
            false,
            false,
            [true],
            [None, Some("état SSM bf16".to_string())],
        );
        assert!(linear.is_some_and(|r| r.contains("SSM bf16")));
    }

    /// Refus linéaires au niveau de la forme : SSM bf16, dims, lignes au-delà
    /// du journal (message « B ≤ 15 »), journal plus grand que l'état.
    #[test]
    fn linear_tree_refusals() {
        use crate::metal_backend::{linear_tree_refusal, LinearTreeShape};
        let shape = LinearTreeShape {
            ssm_bf16: false,
            conv_kernel_dim: 4,
            key_head_dim: 128,
            value_head_dim: 128,
            num_value_heads: 48,
            ssm_len: 48 * 128 * 128,
        };
        assert_eq!(linear_tree_refusal(shape, 9, true), None);
        assert_eq!(linear_tree_refusal(shape, 16, true), None);
        assert!(linear_tree_refusal(shape, 17, true).is_some_and(|r| r.contains("B ≤ 15")));
        assert!(linear_tree_refusal(shape, 1, true).is_some());
        let bf16 = LinearTreeShape {
            ssm_bf16: true,
            ..shape
        };
        assert!(linear_tree_refusal(bf16, 9, true).is_some_and(|r| r.contains("bf16")));
        let k2 = LinearTreeShape {
            conv_kernel_dim: 2,
            ..shape
        };
        assert!(linear_tree_refusal(k2, 9, true).is_some());
        assert!(linear_tree_refusal(shape, 9, false).is_some());
        let small = LinearTreeShape {
            ssm_len: 1024,
            ..shape
        };
        assert!(linear_tree_refusal(small, 9, true).is_some_and(|r| r.contains("journal")));
    }

    #[test]
    fn tree_budget_flag_parses_integers_only() {
        use crate::runtime_flags::parse_dflash_tree_budget;
        assert_eq!(parse_dflash_tree_budget(" 8 "), Some(8));
        assert_eq!(parse_dflash_tree_budget("0"), Some(0));
        assert_eq!(parse_dflash_tree_budget("huit"), None);
        assert_eq!(parse_dflash_tree_budget("-1"), None);
    }
}
