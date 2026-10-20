//! Boucle DFlash avec drafter résident : proposition du bloc sur GPU
//! (`decode_resident::dflash`), vérification résidente en lignes avec captures
//! (rollback par troncature, sans rejeu), contexte du drafter nourri des hidden
//! states cibles commités. Même acceptance que la boucle CPU
//! `generate_greedy_dflash_batched_with_options` (oracle), coût par tour borné
//! par la vérification.

use super::super::*;
use crate::decode_resident::dflash::DFlashResident;
use crate::decode_resident::GpuTensor;
use crate::metal_backend::{commit_and_wait, read_f32_buffer, read_u32_buffer, MetalExecutor};
use crate::DFlashDraft;

/// Lignes de hidden states cibles téléversées par command buffer (prefill du
/// contexte du drafter et append des lignes commitées).
const CONTEXT_UPLOAD_ROWS: usize = 64;

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
    /// résidente en lignes. Sorties identiques à l'AR (vérifieur exact) et à la
    /// boucle CPU, à la précision des matmuls près pour la proposition.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le mode n'est pas greedy, si le decode résident
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
        mut on_token: impl FnMut(usize) -> bool,
    ) -> Result<SpeculativeOutput> {
        if options.temperature > f32::EPSILON {
            return Err(InferError::Config(
                "decode DFlash résident supporte uniquement greedy temperature=0".to_string(),
            ));
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
        let block = draft_limit + 1;
        let layer_ids = draft.info.target_layer_ids.as_slice();
        let hidden = self.final_norm.len();
        let context_cols = layer_ids
            .len()
            .checked_mul(hidden)
            .ok_or_else(|| InferError::Shape("DFlash contexte déborde".to_string()))?;

        let (mut cache, final_state, target_hidden) =
            self.prefill_cache_state_with_layer_capture(prompt, layer_ids)?;
        if !self.setup_resident_decode_from_prefill(&mut cache, max_new_tokens + draft_limit + 1)? {
            return Err(InferError::Config(
                "decode DFlash résident: decode résident complet indisponible".to_string(),
            ));
        }
        // Le rollback par troncature exige la vérification batchée (captures) ; le
        // repli séquentiel (`SARAGOSSA_RUST_PREFILL_MOE_ROWS=0`) n'en fournit pas.
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
        let top_k = draft.selector.as_ref().map_or(1, |selector| selector.top_k);
        let rank = draft.selector.as_ref().map_or(0, |selector| selector.rank);
        let capacity = prompt
            .len()
            .checked_add(max_new_tokens)
            .and_then(|value| value.checked_add(block + 1))
            .ok_or_else(|| InferError::Shape("DFlash capacité déborde".to_string()))?;
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
        // Toutes les couches fenêtrées : seules les `window` dernières positions du
        // prompt sont attendues, les précédentes ne sont jamais lues.
        let all_sliding = !draft.info.layer_types.is_empty()
            && draft
                .info
                .layer_types
                .iter()
                .all(|kind| kind == "sliding_attention");
        let skip = draft
            .info
            .sliding_window
            .filter(|_| all_sliding)
            .map_or(0, |window| prompt.len().saturating_sub(window));
        drafter.skip_context(skip)?;
        self.dflash_resident_append_context(
            &mut cache,
            metal,
            &mut drafter,
            &buffers,
            &target_hidden,
            skip,
            prompt.len(),
        )?;

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
        let switch_min = crate::runtime_flags::env_var("SARAGOSSA_RUST_DFLASH_SWITCH_MIN")
            .and_then(|value| value.trim().parse::<f64>().ok())
            .unwrap_or(0.0);
        let switch_warmup = crate::runtime_flags::env_var("SARAGOSSA_RUST_DFLASH_SWITCH_WARMUP")
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(6)
            .max(1);
        let mut segments = [std::time::Duration::ZERO; 4];
        let mut tours = 0usize;

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
            generated.push(primary);
            if !on_token(primary) || generated.len() >= max_new_tokens {
                break;
            }

            tours += 1;
            let mut segment_started = Instant::now();
            let drafts = self.dflash_resident_drafts(
                &mut cache,
                metal,
                &mut drafter,
                &buffers,
                draft,
                primary,
                draft_limit,
            )?;
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
                    true,
                    true,
                    Some(layer_ids),
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
                    generated.push(draft_token);
                    if !on_token(draft_token) {
                        stopped = true;
                        accepted_all = false;
                        break;
                    }
                    if generated.len() >= max_new_tokens
                        || options.stop_token_ids.contains(&draft_token)
                    {
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
            if generated.len() >= max_new_tokens
                || generated
                    .last()
                    .is_some_and(|token| options.stop_token_ids.contains(token))
            {
                break;
            }
        }
        if profile && tours > 0 {
            let per_tour = |d: std::time::Duration| d.as_secs_f64() * 1e3 / tours as f64;
            eprintln!(
                "dflash_profile tours={tours} ms/tour draft={:.1} verify={:.1} rollback={:.1} append={:.1} total={:.1}",
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
    /// projection, parcours du sélecteur sur CPU. Renvoie `draft_limit` tokens.
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
        let tokens = indices
            .iter()
            .map(|index| {
                usize::try_from(*index).map_err(|_| {
                    InferError::Metal(format!("drafter résident: index trop grand {index}"))
                })
            })
            .collect::<Result<Vec<_>>>()?;
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
                selector.select_projected(&projection[buffers.rank..], &candidates, primary)?
            }
            // DFlash 1 : argmax par position (top-1).
            _ => tokens.iter().step_by(buffers.top_k).copied().collect(),
        };
        Ok(drafts.into_iter().take(draft_limit).collect())
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
