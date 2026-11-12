use super::super::*;
#[cfg(all(target_os = "macos", feature = "metal"))]
use super::types::*;
#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::decoder::ResidentVerifyHead;

/// Vérification enregistrée par l'oracle O2 (tests seulement) : position de la
/// première ligne, tokens d'entrée, parents de l'arbre éventuel, états finaux
/// normés `[lignes, hidden]` et chemin réellement exécuté (`batched` : kernels
/// en lignes ; faux pour le repli séquentiel, un pas mono par token).
#[cfg(all(test, target_os = "macos", feature = "metal"))]
#[derive(Clone, Debug)]
pub(in crate::decoder) struct VerifyRecord {
    pub(in crate::decoder) base_position: usize,
    pub(in crate::decoder) tokens: Vec<usize>,
    pub(in crate::decoder) parents: Option<Vec<u32>>,
    pub(in crate::decoder) states: Vec<f32>,
    pub(in crate::decoder) batched: bool,
    /// Vérification par la tête `Skip` (repli sans argmax GPU, états seuls).
    pub(in crate::decoder) skip_head: bool,
}

#[cfg(all(test, target_os = "macos", feature = "metal"))]
thread_local! {
    /// Enregistreur des vérifications résidentes : `Some` pendant un oracle O2.
    pub(in crate::decoder) static VERIFY_RECORDER: std::cell::RefCell<Option<Vec<VerifyRecord>>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(all(test, target_os = "macos", feature = "metal"))]
pub(super) fn verify_recorder_active() -> bool {
    VERIFY_RECORDER.with(|slot| slot.borrow().is_some())
}

#[cfg(all(test, target_os = "macos", feature = "metal"))]
pub(super) fn record_verify(
    base_position: usize,
    tokens: &[usize],
    tree: Option<&crate::metal_backend::VerifyTree>,
    states: &[f32],
    batched: bool,
    skip_head: bool,
) {
    VERIFY_RECORDER.with(|slot| {
        if let Some(records) = slot.borrow_mut().as_mut() {
            records.push(VerifyRecord {
                base_position,
                tokens: tokens.to_vec(),
                parents: tree.map(|tree| tree.parents().to_vec()),
                states: states.to_vec(),
                batched,
                skip_head,
            });
        }
    });
}

impl CausalDecoder {
    /// Vérification MoE en lignes : la valeur explicite de
    /// `SARAGOSSA_RUST_MOE_VERIFY_ROWS`, sinon active dès qu'une tête MTP MoE est
    /// chargée. Ses cycles D1/D2 fusionnés n'existent qu'en lignes et portent tout
    /// le gain (×1,2 à ×1,6 contre l'AR, docs/perf/mtp-pairwise.md) ; sans tête,
    /// le défaut reste OFF (lookup seul : régression sur T0 et T7).
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(in crate::decoder) fn moe_verify_rows(&self) -> bool {
        crate::runtime_flags::moe_verify_rows_override().unwrap_or_else(|| {
            matches!(
                self.mtp.as_ref().map(|head| &head.layer.mlp),
                Some(FeedForward::Moe(_))
            )
        })
    }

    /// Vrai si toutes les couches ont leurs buffers résidents en lignes : la
    /// vérification batchée (captures, rollback par troncature) est possible.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(in crate::decoder) fn resident_layer_batch_supported(&self, arena: &ResidentArena) -> bool {
        self.layers.iter().enumerate().all(|(index, layer)| {
            match (self.plan.layers[index].resident_full, layer.mlp.as_ref()) {
                (true, Some(FeedForward::Dense(_))) => {
                    matches!(
                        arena.layers.get(index),
                        Some(ResidentLayerBuffers::FullDense(_))
                    )
                }
                (true, Some(FeedForward::Moe(_))) if self.moe_verify_rows() => {
                    matches!(
                        arena.layers.get(index),
                        Some(ResidentLayerBuffers::FullMoe(_))
                    )
                }
                (false, Some(FeedForward::Moe(_))) if prefill_moe_rows_enabled() => {
                    matches!(
                        arena.layers.get(index),
                        Some(ResidentLayerBuffers::LinearMoe(_))
                    )
                }
                (false, Some(FeedForward::Dense(_))) => {
                    matches!(
                        arena.layers.get(index),
                        Some(ResidentLayerBuffers::LinearDense(_))
                    )
                }
                _ => false,
            }
        })
    }

    /// Vérification résidente de `token_ids` en lignes. `tree` (DFlash) : les
    /// lignes forment cet arbre (positions par profondeur, attention et états
    /// linéaires par ancêtres) ; exige le chemin batché et des captures, laisse
    /// les états linéaires d'entrée intacts (rollback obligatoire, cf.
    /// `restore_tree_resident_progress`).
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(in crate::decoder) fn next_final_states_resident_verify(
        &self,
        cache: &mut CausalDecoderCache,
        token_ids: &[usize],
        head: ResidentVerifyHead,
        capture_linear: bool,
        capture_layer_ids: Option<&[usize]>,
        tree: Option<std::rc::Rc<crate::metal_backend::VerifyTree>>,
    ) -> Result<Option<ResidentVerifyOutput>> {
        if token_ids.is_empty() {
            return Ok(None);
        }
        if let Some(tree) = &tree {
            tree.check_rows(token_ids.len(), "vérification résidente")?;
            if !capture_linear {
                return Err(InferError::Config(
                    "vérification en arbre sans captures linéaires".to_string(),
                ));
            }
        }
        let Some(metal) = self.forward_runtime().metal_executor() else {
            return Ok(None);
        };
        let theta = self.config.rope_theta.ok_or_else(|| {
            InferError::Config("rope_theta manquant (verify résident)".to_string())
        })?;
        let eps = self.config.rms_eps;
        let hidden = self.final_norm.data().len();
        let linear_dims = if self.has_resident_linear_attention_layer() {
            let la_config = self.config.linear_attention_config()?;
            let la_spec = LinearAttentionStepSpec {
                num_key_heads: la_config.num_key_heads,
                num_value_heads: la_config.num_value_heads,
                key_head_dim: la_config.key_head_dim,
                value_head_dim: la_config.value_head_dim,
                conv_kernel_dim: la_config.conv_kernel_dim,
                rms_eps: la_config.rms_eps,
                l2_qk_sigmoid_gate: false,
            };
            let key_dim = la_config.key_dim()?;
            let value_dim = la_config.value_dim()?;
            let conv_dim = key_dim
                .checked_mul(2)
                .and_then(|twice| twice.checked_add(value_dim))
                .ok_or_else(|| {
                    InferError::Shape("conv_dim déborde (verify résident)".to_string())
                })?;
            Some((la_spec, key_dim, value_dim, conv_dim))
        } else {
            None
        };

        let CausalDecoderCache {
            layers,
            position: cache_position,
            resident,
        } = cache;
        let Some(arena) = resident.as_mut() else {
            return Ok(None);
        };

        let layer_batch_supported =
            token_ids.len() > 1 && self.resident_layer_batch_supported(arena);
        if matches!(head, ResidentVerifyHead::Sample { .. }) && !layer_batch_supported {
            return Err(InferError::Config(
                "vérification échantillonnée: couches en lignes et au moins deux lignes requises"
                    .to_string(),
            ));
        }
        if tree.is_some() && !layer_batch_supported {
            return Err(InferError::Config(
                "vérification en arbre: couches en lignes requises".to_string(),
            ));
        }
        if layer_batch_supported {
            let base_position = *cache_position;
            let embed = self.embed_scaled(token_ids)?;
            let (batch, embed_hidden) = embed.as_matrix()?;
            if batch != token_ids.len() || embed_hidden != hidden {
                return Err(InferError::Dimension(format!(
                    "embedding batch={batch} hidden={embed_hidden}, attendu batch={} hidden={hidden}",
                    token_ids.len()
                )));
            }
            let batch_elements = batch
                .checked_mul(hidden)
                .ok_or_else(|| InferError::Dimension("verify batch hidden déborde".to_string()))?;
            let batch_a = arena
                .state
                .scratch()
                .lease(batch_elements, GpuElement::F32)?;
            let batch_b = arena
                .state
                .scratch()
                .lease(batch_elements, GpuElement::F32)?;
            arena.state.upload(batch_a.tensor(), embed.data())?;

            let final_states = arena
                .state
                .scratch()
                .lease(batch_elements, GpuElement::F32)?;
            let target_capture_cols = capture_layer_ids
                .map(|layer_ids| layer_ids.len().saturating_mul(hidden))
                .unwrap_or(0);
            let target_captures = if target_capture_cols > 0 {
                let capture_elements = batch.checked_mul(target_capture_cols).ok_or_else(|| {
                    InferError::Dimension("capture hidden résident déborde".to_string())
                })?;
                Some(
                    arena
                        .state
                        .scratch()
                        .lease(capture_elements, GpuElement::F32)?,
                )
            } else {
                None
            };
            let linear_captures = if capture_linear {
                // Arbre : une capture par ligne (la feuille acceptée peut être la
                // dernière) ; chaîne : la dernière ligne vit dans l'état lui-même.
                let capture_rows = if tree.is_some() {
                    batch
                } else {
                    batch.saturating_sub(1)
                };
                if capture_rows == 0 {
                    None
                } else {
                    let linear_states = layers
                        .iter()
                        .map(|layer| layer.linear.metal_state())
                        .collect::<Vec<_>>();
                    Some(metal.lease_linear_attn_state_captures(
                        arena.state.scratch(),
                        &linear_states,
                        capture_rows,
                        tree.is_some(),
                    )?)
                }
            } else {
                None
            };
            let command_buffer = arena.state.queue().new_command_buffer();
            let _barrier_guard = crate::metal_backend::resident_concurrent_enabled()
                .then(crate::metal_backend::install_dispatch_barrier_scope);
            let _namespace_guard =
                crate::metal_backend::install_scratch_namespace(arena.state.scratch_namespace());
            let _tree_guard = tree
                .clone()
                .map(crate::metal_backend::install_verify_tree_scope);
            // Politique d'exactitude de la requête : posée par la boucle
            // spéculative (température nulle), quel que soit le type de tête
            // (`Skip` sous `SARAGOSSA_RUST_GPU_ARGMAX=0`) ; à défaut, seule la
            // tête argmax est greedy.
            let _greedy_guard = crate::metal_backend::install_verify_greedy_scope(
                crate::metal_backend::request_greedy()
                    .unwrap_or(matches!(head, ResidentVerifyHead::Argmax)),
            );
            let encoder = new_resident_compute_encoder(command_buffer);
            let mut owned: Vec<metal::Buffer> = Vec::new();
            let mut current = batch_a.tensor().buffer();
            let mut other = batch_b.tensor().buffer();

            for (index, layer) in self.layers.iter().enumerate() {
                let layer_cache = &mut layers[index];
                if self.plan.layers[index].resident_full {
                    // Toutes les lignes en un passage, comme le pas D2 fusionné :
                    // ligne par ligne, chaque ligne relisait les poids de la couche.
                    let dims = self.plan.layers[index]
                        .resident
                        .resident_windowed_full_attn_layer_dims(
                            hidden,
                            *cache_position,
                            eps,
                            theta,
                        )?;
                    // flint (MoE sigmoïde) est résolu en `FullDense` : seul le
                    // type des buffers résidents choisit la queue MoE shared.
                    if matches!(
                        arena.layers.get(index),
                        Some(ResidentLayerBuffers::FullMoe(_))
                    ) {
                        if crate::runtime_flags::verify_skip("full_layer") {
                            metal.encode_copy_with_offsets(
                                encoder,
                                current,
                                0,
                                other,
                                0,
                                batch * hidden,
                            )?;
                        } else {
                            self.encode_resident_full_moe_layer_rows(
                                metal,
                                arena,
                                layer_cache,
                                index,
                                encoder,
                                &mut owned,
                                dims,
                                batch,
                                current,
                                other,
                            )?;
                        }
                    } else {
                        self.encode_resident_full_dense_layer_rows(
                            metal,
                            arena,
                            layer_cache,
                            layer,
                            index,
                            encoder,
                            &mut owned,
                            dims,
                            batch,
                            current,
                            other,
                        )?;
                    }
                } else {
                    let Some((la_spec, key_dim, value_dim, conv_dim)) = linear_dims else {
                        return Err(InferError::Config(
                            "dims linear-attn résidentes absentes".to_string(),
                        ));
                    };
                    let res_dims = LinearAttnResidentDims {
                        in_dim: hidden,
                        conv_dim,
                        value_dim,
                        key_dim,
                    };
                    let captures = linear_captures
                        .as_ref()
                        .and_then(|(captures, _)| captures.get(index))
                        .and_then(|captures| captures.as_deref());
                    match layer.mlp.as_ref() {
                        Some(FeedForward::Moe(_)) => self.encode_resident_linear_layer_rows(
                            metal,
                            arena,
                            layer_cache,
                            layer,
                            index,
                            encoder,
                            &mut owned,
                            la_spec,
                            res_dims,
                            batch,
                            hidden,
                            eps,
                            current,
                            other,
                            captures,
                        )?,
                        Some(FeedForward::Dense(_)) => self
                            .encode_resident_linear_dense_layer_rows(
                                metal,
                                arena,
                                layer_cache,
                                layer,
                                index,
                                encoder,
                                &mut owned,
                                la_spec,
                                res_dims,
                                batch,
                                hidden,
                                eps,
                                current,
                                other,
                                captures,
                            )?,
                        _ => {
                            return Err(InferError::Config(
                                "MLP attendu (verify résident batché)".to_string(),
                            ));
                        }
                    }
                }
                if let (Some(layer_ids), Some(target_captures)) =
                    (capture_layer_ids, target_captures.as_ref())
                {
                    if let Some(slot) = resident_capture_slot(layer_ids, index) {
                        for row in 0..batch {
                            let source_offset = row
                                .checked_mul(hidden)
                                .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
                                .and_then(|value| u64::try_from(value).ok())
                                .ok_or_else(|| {
                                    InferError::Metal(
                                        "capture hidden source offset déborde".to_string(),
                                    )
                                })?;
                            let target_offset = row
                                .checked_mul(target_capture_cols)
                                .and_then(|value| value.checked_add(slot * hidden))
                                .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
                                .and_then(|value| u64::try_from(value).ok())
                                .ok_or_else(|| {
                                    InferError::Metal(
                                        "capture hidden target offset déborde".to_string(),
                                    )
                                })?;
                            metal.encode_copy_with_offsets(
                                encoder,
                                other,
                                source_offset,
                                target_captures.tensor().buffer(),
                                target_offset,
                                hidden,
                            )?;
                        }
                    }
                }
                std::mem::swap(&mut current, &mut other);
            }
            metal.encode_rms_norm_rows(
                encoder,
                current,
                &arena.final_norm,
                final_states.tensor().buffer(),
                batch,
                hidden,
                eps,
            )?;
            // Ablation devtools de la tête : aucun token rendu (le scratch des
            // indices n'est pas réinitialisé, il serait lu comme des tokens).
            let skip_head = crate::runtime_flags::verify_skip("head");
            let final_indices = if head == ResidentVerifyHead::Argmax && !skip_head {
                let indices = arena.state.scratch().lease(batch, GpuElement::U32)?;
                // lm_head projeté une fois pour toutes les lignes, argmax par ligne.
                metal.encode_lm_head_argmax_rows_buffers(
                    encoder,
                    final_states.tensor().buffer(),
                    &arena.lm_head,
                    indices.tensor().buffer(),
                    0,
                    0,
                    batch,
                    hidden,
                )?;
                Some(indices)
            } else if head == ResidentVerifyHead::Argmax {
                None
            } else if let ResidentVerifyHead::Sample {
                temperature,
                top_p,
                top_k,
                seed,
                first_draw,
            } = head
            {
                let indices = arena.state.scratch().lease(batch, GpuElement::U32)?;
                // NOTE: les lignes sœurs d'un arbre partagent l'état de leur
                // profondeur ; le parcours n'en visite qu'une par profondeur.
                let rng_states = (0..batch)
                    .map(|row| {
                        let depth = match &tree {
                            Some(tree) => tree.depths()[row] as usize,
                            None => row,
                        };
                        crate::sampling::rng_state_for_draw(seed, first_draw + depth)
                    })
                    .collect::<Vec<_>>();
                metal.encode_lm_head_sample_rows_buffers(
                    encoder,
                    final_states.tensor().buffer(),
                    &arena.lm_head,
                    indices.tensor().buffer(),
                    hidden,
                    (temperature, top_p, top_k),
                    &rng_states,
                )?;
                Some(indices)
            } else {
                None
            };
            let logits_readback = if head == ResidentVerifyHead::Logits {
                Some(metal.encode_lm_head_logits_rows_readback_buffers(
                    encoder,
                    final_states.tensor().buffer(),
                    &arena.lm_head,
                    batch,
                    hidden,
                )?)
            } else {
                None
            };
            encoder.end_encoding();
            crate::metal_backend::commit_and_wait(command_buffer)?;
            *cache_position += batch;
            let output = crate::metal_backend::read_f32_buffer(
                final_states.tensor().buffer(),
                batch_elements,
            )?;
            let tokens = final_indices
                .as_ref()
                .map(|indices| {
                    crate::metal_backend::read_u32_buffer(indices.tensor().buffer(), batch)?
                        .into_iter()
                        .map(|index| {
                            usize::try_from(index).map_err(|_| {
                                InferError::Metal(format!(
                                    "verify argmax index trop grand: {index}"
                                ))
                            })
                        })
                        .collect::<Result<Vec<_>>>()
                })
                .transpose()?;
            let logits = logits_readback
                .map(|(buffer, len)| {
                    let values = crate::metal_backend::read_f32_buffer(&buffer, len)?;
                    Tensor::from_vec(vec![batch, len / batch.max(1)], values)
                })
                .transpose()?;
            let target_hidden = target_captures
                .as_ref()
                .map(|captures| {
                    let values = crate::metal_backend::read_f32_buffer(
                        captures.tensor().buffer(),
                        batch * target_capture_cols,
                    )?;
                    Tensor::from_vec(vec![batch, target_capture_cols], values)
                })
                .transpose()?;
            #[cfg(test)]
            record_verify(
                base_position,
                token_ids,
                tree.as_deref(),
                &output,
                true,
                head == ResidentVerifyHead::Skip,
            );
            return Ok(Some(ResidentVerifyOutput {
                states: Tensor::from_vec(vec![batch, hidden], output)?,
                tokens,
                logits,
                captures: linear_captures.map(|linear| ResidentVerifyCaptures {
                    base_position,
                    linear: linear.0,
                    _linear_leases: linear.1,
                }),
                target_hidden,
            }));
        }

        let mut token_inputs = Vec::with_capacity(token_ids.len());
        for token_id in token_ids {
            let embed = self.embed_scaled(&[*token_id])?;
            let (_, embed_hidden) = embed.as_matrix()?;
            if embed_hidden != hidden {
                return Err(InferError::Dimension(format!(
                    "embedding hidden={embed_hidden}, attendu {hidden}"
                )));
            }
            let input = arena.state.persistent(hidden, GpuElement::F32)?;
            arena.state.upload(&input, embed.data())?;
            token_inputs.push(input);
        }

        let final_states = arena
            .state
            .scratch()
            .lease(token_ids.len() * hidden, GpuElement::F32)?;
        let target_capture_cols = capture_layer_ids
            .map(|layer_ids| layer_ids.len().saturating_mul(hidden))
            .unwrap_or(0);
        let target_captures = if target_capture_cols > 0 {
            let capture_elements = token_ids
                .len()
                .checked_mul(target_capture_cols)
                .ok_or_else(|| {
                    InferError::Dimension("capture hidden résident déborde".to_string())
                })?;
            Some(
                arena
                    .state
                    .scratch()
                    .lease(capture_elements, GpuElement::F32)?,
            )
        } else {
            None
        };
        let final_normed = arena.state.scratch().lease(hidden, GpuElement::F32)?;
        let command_buffer = arena.state.queue().new_command_buffer();
        let _barrier_guard = crate::metal_backend::resident_concurrent_enabled()
            .then(crate::metal_backend::install_dispatch_barrier_scope);
        let _namespace_guard =
            crate::metal_backend::install_scratch_namespace(arena.state.scratch_namespace());
        let encoder = new_resident_compute_encoder(command_buffer);
        let mut owned: Vec<metal::Buffer> = Vec::new();

        for (token_pos, input) in token_inputs.iter().enumerate() {
            let mut current = input.buffer();
            let mut other = arena.hidden_b.buffer();
            let position = *cache_position + token_pos;
            for (index, layer) in self.layers.iter().enumerate() {
                let layer_cache = &mut layers[index];
                if self.plan.layers[index].resident_full {
                    let dims = self.plan.layers[index]
                        .resident
                        .resident_windowed_full_attn_layer_dims(hidden, position, eps, theta)?;
                    if crate::runtime_flags::verify_skip("full_layer") {
                        metal.encode_copy_with_offsets(encoder, current, 0, other, 0, hidden)?;
                    } else {
                        self.encode_resident_full_layer(
                            metal,
                            arena,
                            layer_cache,
                            layer,
                            index,
                            encoder,
                            &mut owned,
                            dims,
                            current,
                            other,
                        )?;
                    }
                } else {
                    let Some((la_spec, key_dim, value_dim, conv_dim)) = linear_dims else {
                        return Err(InferError::Config(
                            "dims linear-attn résidentes absentes".to_string(),
                        ));
                    };
                    let res_dims = LinearAttnResidentDims {
                        in_dim: hidden,
                        conv_dim,
                        value_dim,
                        key_dim,
                    };
                    self.encode_resident_linear_layer(
                        metal,
                        arena,
                        layer_cache,
                        layer,
                        index,
                        encoder,
                        &mut owned,
                        la_spec,
                        res_dims,
                        hidden,
                        eps,
                        current,
                        other,
                    )?;
                }
                if let (Some(layer_ids), Some(target_captures)) =
                    (capture_layer_ids, target_captures.as_ref())
                {
                    if let Some(slot) = resident_capture_slot(layer_ids, index) {
                        let target_offset = token_pos
                            .checked_mul(target_capture_cols)
                            .and_then(|value| value.checked_add(slot * hidden))
                            .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
                            .and_then(|value| u64::try_from(value).ok())
                            .ok_or_else(|| {
                                InferError::Metal(
                                    "capture hidden target offset déborde".to_string(),
                                )
                            })?;
                        metal.encode_copy_with_offsets(
                            encoder,
                            other,
                            0,
                            target_captures.tensor().buffer(),
                            target_offset,
                            hidden,
                        )?;
                    }
                }
                std::mem::swap(&mut current, &mut other);
            }
            metal.encode_rms_norm_rows(
                encoder,
                current,
                &arena.final_norm,
                final_normed.tensor().buffer(),
                1,
                hidden,
                eps,
            )?;
            let final_offset = token_pos
                .checked_mul(hidden)
                .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
                .and_then(|value| u64::try_from(value).ok())
                .ok_or_else(|| InferError::Metal("verify final offset déborde".to_string()))?;
            metal.encode_copy_with_offsets(
                encoder,
                final_normed.tensor().buffer(),
                0,
                final_states.tensor().buffer(),
                final_offset,
                hidden,
            )?;
        }
        // Ablation devtools de la tête : aucun token rendu (voir le chemin en
        // lignes).
        let skip_head = crate::runtime_flags::verify_skip("head");
        let final_indices = if head == ResidentVerifyHead::Argmax && !skip_head {
            let indices = arena
                .state
                .scratch()
                .lease(token_ids.len(), GpuElement::U32)?;
            for row in 0..token_ids.len() {
                let input_offset = row
                    .checked_mul(hidden)
                    .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
                    .and_then(|value| u64::try_from(value).ok())
                    .ok_or_else(|| {
                        InferError::Metal("verify argmax input offset déborde".to_string())
                    })?;
                let index_offset = row
                    .checked_mul(std::mem::size_of::<u32>())
                    .and_then(|value| u64::try_from(value).ok())
                    .ok_or_else(|| {
                        InferError::Metal("verify argmax index offset déborde".to_string())
                    })?;
                metal.encode_lm_head_argmax_buffers_with_index_offset(
                    encoder,
                    &mut owned,
                    final_states.tensor().buffer(),
                    &arena.lm_head,
                    indices.tensor().buffer(),
                    input_offset,
                    index_offset,
                    hidden,
                )?;
            }
            Some(indices)
        } else {
            None
        };
        encoder.end_encoding();
        crate::metal_backend::commit_and_wait(command_buffer)?;
        #[cfg(test)]
        let base_position = *cache_position;
        *cache_position += token_ids.len();
        let output = crate::metal_backend::read_f32_buffer(
            final_states.tensor().buffer(),
            token_ids.len() * hidden,
        )?;
        #[cfg(test)]
        record_verify(
            base_position,
            token_ids,
            None,
            &output,
            false,
            head == ResidentVerifyHead::Skip,
        );
        let tokens = final_indices
            .as_ref()
            .map(|indices| {
                crate::metal_backend::read_u32_buffer(indices.tensor().buffer(), token_ids.len())?
                    .into_iter()
                    .map(|index| {
                        usize::try_from(index).map_err(|_| {
                            InferError::Metal(format!("verify argmax index trop grand: {index}"))
                        })
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?;
        let target_hidden = target_captures
            .as_ref()
            .map(|captures| {
                let values = crate::metal_backend::read_f32_buffer(
                    captures.tensor().buffer(),
                    token_ids.len() * target_capture_cols,
                )?;
                Tensor::from_vec(vec![token_ids.len(), target_capture_cols], values)
            })
            .transpose()?;
        Ok(Some(ResidentVerifyOutput {
            states: Tensor::from_vec(vec![token_ids.len(), hidden], output)?,
            tokens,
            // NOTE: le chemin séquentiel ne rend pas les logits ; l'appelant
            // retombe sur le lm_head CPU par ligne.
            logits: None,
            captures: None,
            target_hidden,
        }))
    }
}
