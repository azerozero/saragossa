use super::super::*;
use super::types::*;

impl CausalDecoder {
    /// Renvoie `true` si le decode résident COMPLET (1c) est applicable : un
    /// executor Metal, des dimensions GQA valides, un lm_head biasless (argmax
    /// GPU), et TOUTES les couches encodables en résident
    /// ([`DecoderLayer::supports_resident_full`]).
    ///
    /// Validation EN AMONT (réserve majeure 6) : le forward résident est
    /// tout-ou-rien — soit le command buffer unique est entièrement GPU, soit on
    /// retombe sur le per-op AVANT de commencer, jamais un readback CPU au milieu.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(in crate::decoder) fn supports_resident_full_decode(&self) -> bool {
        let unsupported = self.resident_full_decode_unsupported_reason();
        if let Some(reason) = unsupported.as_ref() {
            if crate::decoder::flags::trace_resident_enabled() {
                eprintln!("decode résident full désactivé: {reason}");
            }
            return false;
        }
        true
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn resident_full_decode_unsupported_reason(&self) -> Option<String> {
        if self.forward_runtime().metal_executor().is_none() {
            return Some("executor Metal absent".to_string());
        }
        if self.config.head_dim.is_none() {
            return Some("head_dim absent".to_string());
        }
        let q_heads = self.config.num_attention_heads;
        let kv_heads = self.config.num_key_value_heads;
        if q_heads == 0 || kv_heads == 0 || q_heads % kv_heads != 0 {
            return Some(format!(
                "GQA invalide q_heads={q_heads} kv_heads={kv_heads}"
            ));
        }
        if self.lm_head.bias().is_some() {
            return Some("lm_head biaisé".to_string());
        }
        // Le chemin full-attn résident fusionne TOUJOURS norm+RoPE à la position
        // (`encode_rms_norm_rope_decode`) → exiger rope_theta présent (sinon le
        // per-op ferait rms_norm sans RoPE, divergence).
        if self.config.rope_theta.is_none() {
            return Some("rope_theta absent".to_string());
        }
        // Les kernels résidents n'implémentent que le rotate-half : tout autre
        // appariement RoPE retombe sur le chemin per-op (qui le supporte).
        if self.config.rope_style != RopeStyle::Halves {
            return Some(format!(
                "rope_style {:?} non supporté",
                self.config.rope_style
            ));
        }
        // Les kernels résidents appliquent aussi des positions RoPE brutes :
        // exclure tout modèle à rope_scaling linear (defense-in-depth — Gemma
        // est déjà hors périmètre via sa double norme feed-forward par couche).
        if self.config.rope_position_scale.is_some() {
            return Some("rope_position_scale présent".to_string());
        }
        if !decode_resident_full_linear_enabled()
            && self
                .layers
                .iter()
                .enumerate()
                .any(|(index, _)| !self.config.is_full_attention_layer(index))
        {
            return Some(
                "couches linear-attn en decode résident full désactivées \
                 (RETI_RUST_DECODE_RESIDENT_FULL_LINEAR=0)"
                    .to_string(),
            );
        }
        self.layers.iter().enumerate().find_map(|(index, layer)| {
            if layer.supports_resident_full() {
                None
            } else {
                let reason =
                    resident_full_layer_unsupported_reason(layer).unwrap_or("raison inconnue");
                Some(format!("couche {index}: {reason}"))
            }
        })
    }

    /// Prépare le decode full-attn résident : alloue les buffers KV GPU des
    /// couches full-attn et les seed depuis le K/V (rope'd) du prefill (CPU,
    /// `keys`/`values`). Une couche sans KV prefill cohérent reste sur le chemin
    /// CPU (`full` laissé à `None`) → l'activation est sûre et incrémentale.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si la compilation du kernel résident, une allocation ou
    /// le seed échoue.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(in crate::decoder) fn setup_resident_decode(
        &self,
        cache: &mut CausalDecoderCache,
        max_new_tokens: usize,
        sampled: bool,
    ) -> Result<()> {
        let Some(metal) = self.forward_runtime().metal_executor() else {
            return Ok(());
        };
        let Some(head_dim) = self.config.head_dim else {
            return Ok(());
        };
        let q_heads = self.config.num_attention_heads;
        let kv_heads = self.config.num_key_value_heads;
        if q_heads == 0 || kv_heads == 0 || head_dim == 0 || q_heads % kv_heads != 0 {
            return Ok(());
        }
        let kv_dim = kv_heads * head_dim;
        let prefill_len = cache.position;
        let capacity = prefill_len
            .checked_add(max_new_tokens)
            .ok_or_else(|| InferError::Dimension("capacité KV résidente déborde".to_string()))?;
        if capacity == 0 {
            return Ok(());
        }
        let arena = DecodeResidentState::new(metal.device().clone())?;
        for (index, layer_cache) in cache.layers.iter_mut().enumerate() {
            if !self.config.is_full_attention_layer(index) {
                continue;
            }
            // Seed depuis le K/V (rope'd) du prefill ; si absent/incohérent, la
            // couche reste sur le chemin CPU (oracle), pas d'erreur.
            if layer_cache.keys.len() != prefill_len * kv_dim
                || layer_cache.values.len() != prefill_len * kv_dim
            {
                continue;
            }
            let mut full = arena.full_attention(capacity, q_heads, kv_heads, head_dim, sampled)?;
            full.seed(&layer_cache.keys, &layer_cache.values, prefill_len)?;
            layer_cache.full = Some(full);
        }
        Ok(())
    }

    /// Prépare le decode résident COMPLET (1c) : valide que TOUS les états par
    /// couche sont prêts (KV full-attn seedable depuis le prefill, état conv/ssm
    /// linear-attn peuplé par le prefill résident per-op), alloue l'arène + le
    /// ping-pong hidden + le buffer `u32` du token, puis seed les KV full-attn.
    ///
    /// Renvoie `Ok(false)` (tout-ou-rien, MAJEUR 6) si une précondition manque
    /// → l'appelant retombe sur le per-op SANS mutation laissée (validation AVANT
    /// seeding) ; `Ok(true)` si l'arène résidente est prête dans `cache.resident`.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si la compilation des kernels, une allocation ou un seed
    /// échoue.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(in crate::decoder) fn setup_resident_full_decode(
        &self,
        cache: &mut CausalDecoderCache,
        max_new_tokens: usize,
        sampled: bool,
    ) -> Result<bool> {
        self.setup_resident_full_decode_with_slot(cache, max_new_tokens, 0, sampled)
    }

    /// Variante light-batch de [`Self::setup_resident_full_decode`] : `slot`
    /// namespace le scratch label-keyed de l'exécuteur partagé pour ce flux
    /// (slot 0 = chemin mono-flux historique, clés inchangées).
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(in crate::decoder) fn setup_resident_full_decode_with_slot(
        &self,
        cache: &mut CausalDecoderCache,
        max_new_tokens: usize,
        slot: u64,
        sampled: bool,
    ) -> Result<bool> {
        let Some(metal) = self.forward_runtime().metal_executor() else {
            return Ok(false);
        };
        let Some(head_dim) = self.config.head_dim else {
            return Ok(false);
        };
        let q_heads = self.config.num_attention_heads;
        let kv_heads = self.config.num_key_value_heads;
        if q_heads == 0 || kv_heads == 0 || head_dim == 0 || q_heads % kv_heads != 0 {
            return Ok(false);
        }
        let hidden = self.final_norm.data().len();
        if hidden == 0 {
            return Ok(false);
        }
        let kv_dim = kv_heads * head_dim;
        let prefill_len = cache.position;
        let capacity = prefill_len
            .checked_add(max_new_tokens)
            .ok_or_else(|| InferError::Dimension("capacité KV résidente déborde".to_string()))?;
        if capacity == 0 {
            return Ok(false);
        }
        // Validation AVANT toute mutation (tout-ou-rien) : KV full-attn seedable et
        // état conv/ssm linear-attn présent (peuplé par le prefill résident per-op).
        for (index, layer_cache) in cache.layers.iter().enumerate() {
            if self.config.is_full_attention_layer(index) {
                if layer_cache.keys.len() != prefill_len * kv_dim
                    || layer_cache.values.len() != prefill_len * kv_dim
                {
                    if crate::decoder::flags::trace_resident_enabled() {
                        eprintln!(
                            "decode résident full setup désactivé: couche {index} KV full len keys={} values={} attendu={}",
                            layer_cache.keys.len(),
                            layer_cache.values.len(),
                            prefill_len * kv_dim
                        );
                    }
                    return Ok(false);
                }
            } else if layer_cache.linear.metal_state().is_none() {
                if crate::decoder::flags::trace_resident_enabled() {
                    eprintln!(
                        "decode résident full setup désactivé: couche {index} état linear-attn Metal absent"
                    );
                }
                return Ok(false);
            }
        }
        let mut layer_buffers = Vec::with_capacity(self.layers.len());
        for (index, layer) in self.layers.iter().enumerate() {
            if self.config.is_full_attention_layer(index) {
                let AttentionBlock::Full(attention) = &layer.attention else {
                    return Ok(false);
                };
                let (Some(q_norm), Some(k_norm)) =
                    (attention.q_norm.as_ref(), attention.k_norm.as_ref())
                else {
                    return Ok(false);
                };
                let Some(v_proj) = attention.v_proj.as_ref() else {
                    return Ok(false);
                };
                let Some(post_norm) = layer.post_attention_norm.as_ref() else {
                    return Ok(false);
                };
                match layer.mlp.as_ref() {
                    Some(FeedForward::Moe(mlp)) => {
                        let input_norm = metal.cached_buffer_from_f32(
                            layer.input_norm.data(),
                            "resident_full_input_norm",
                        )?;
                        let qkv_proj = if decode_resident_full_qkv_concat_enabled() {
                            match metal.resolve_concat_linear_weight_buffers(
                                &[
                                    attention.q_proj.weight(),
                                    attention.k_proj.weight(),
                                    v_proj.weight(),
                                ],
                                "resident_full_qkv_proj",
                            ) {
                                Ok(weights) => Some(weights),
                                Err(InferError::Dimension(_)) => None,
                                Err(error) => return Err(error),
                            }
                        } else {
                            None
                        };
                        let o_proj = metal.resolve_linear_weight_buffers(
                            attention.o_proj.weight(),
                            "resident_full_o_proj",
                        )?;
                        let q_norm =
                            metal.cached_buffer_from_f32(q_norm.data(), "resident_full_q_norm")?;
                        let k_norm =
                            metal.cached_buffer_from_f32(k_norm.data(), "resident_full_k_norm")?;
                        let post_norm = metal
                            .cached_buffer_from_f32(post_norm.data(), "resident_full_post_norm")?;
                        if let Some((router, experts, top_k, shared_expert, shared_gate)) =
                            mlp.shared_metal_parts()
                        {
                            layer_buffers.push(ResidentLayerBuffers::FullMoe(
                                ResidentFullMoeBuffers {
                                    input_norm,
                                    qkv_proj,
                                    q_proj: metal.resolve_linear_weight_buffers(
                                        attention.q_proj.weight(),
                                        "resident_full_q_proj",
                                    )?,
                                    k_proj: metal.resolve_linear_weight_buffers(
                                        attention.k_proj.weight(),
                                        "resident_full_k_proj",
                                    )?,
                                    v_proj: metal.resolve_linear_weight_buffers(
                                        v_proj.weight(),
                                        "resident_full_v_proj",
                                    )?,
                                    o_proj,
                                    q_norm,
                                    k_norm,
                                    post_norm,
                                    moe: metal.resolve_moe_shared_weights(
                                        router,
                                        experts,
                                        shared_expert,
                                        shared_gate,
                                    )?,
                                    top_k,
                                },
                            ));
                        } else if let Some((router, experts, top_k)) = mlp.metal_parts() {
                            let Some(qkv_proj) = qkv_proj else {
                                return Ok(false);
                            };
                            layer_buffers.push(ResidentLayerBuffers::FullRouted(
                                ResidentFullRoutedBuffers {
                                    input_norm,
                                    qkv_proj,
                                    o_proj,
                                    q_norm,
                                    k_norm,
                                    post_norm,
                                    moe: metal.resolve_moe_routed_weights(router, experts)?,
                                    top_k,
                                },
                            ));
                        } else {
                            return Ok(false);
                        }
                    }
                    Some(FeedForward::Dense(mlp)) => {
                        let (gate_proj, up_proj, down_proj) = mlp.projections();
                        let qkv_proj = if decode_resident_full_qkv_concat_enabled() {
                            match metal.resolve_concat_linear_weight_buffers(
                                &[
                                    attention.q_proj.weight(),
                                    attention.k_proj.weight(),
                                    v_proj.weight(),
                                ],
                                "resident_dense_full_qkv_proj",
                            ) {
                                Ok(weights) => Some(weights),
                                Err(InferError::Dimension(_)) => None,
                                Err(error) => return Err(error),
                            }
                        } else {
                            None
                        };
                        layer_buffers.push(ResidentLayerBuffers::FullDense(
                            ResidentFullDenseBuffers {
                                input_norm: metal.cached_buffer_from_f32(
                                    layer.input_norm.data(),
                                    "resident_dense_full_input_norm",
                                )?,
                                qkv_proj,
                                q_proj: metal.resolve_linear_weight_buffers(
                                    attention.q_proj.weight(),
                                    "resident_dense_full_q_proj",
                                )?,
                                k_proj: metal.resolve_linear_weight_buffers(
                                    attention.k_proj.weight(),
                                    "resident_dense_full_k_proj",
                                )?,
                                v_proj: metal.resolve_linear_weight_buffers(
                                    v_proj.weight(),
                                    "resident_dense_full_v_proj",
                                )?,
                                o_proj: metal.resolve_linear_weight_buffers(
                                    attention.o_proj.weight(),
                                    "resident_dense_full_o_proj",
                                )?,
                                q_norm: metal.cached_buffer_from_f32(
                                    q_norm.data(),
                                    "resident_dense_full_q_norm",
                                )?,
                                k_norm: metal.cached_buffer_from_f32(
                                    k_norm.data(),
                                    "resident_dense_full_k_norm",
                                )?,
                                post_norm: metal.cached_buffer_from_f32(
                                    post_norm.data(),
                                    "resident_dense_full_post_norm",
                                )?,
                                gate_proj: metal.resolve_linear_weight_buffers(
                                    gate_proj.weight(),
                                    "resident_dense_gate_proj",
                                )?,
                                up_proj: metal.resolve_linear_weight_buffers(
                                    up_proj.weight(),
                                    "resident_dense_up_proj",
                                )?,
                                down_proj: metal.resolve_linear_weight_buffers(
                                    down_proj.weight(),
                                    "resident_dense_down_proj",
                                )?,
                            },
                        ));
                    }
                    _ => layer_buffers.push(ResidentLayerBuffers::Other),
                }
            } else {
                let AttentionBlock::Linear(linear) = &layer.attention else {
                    return Ok(false);
                };
                let Some(post_norm) = layer.post_attention_norm.as_ref() else {
                    return Ok(false);
                };
                match layer.mlp.as_ref() {
                    Some(FeedForward::Moe(mlp)) => {
                        let Some((router, experts, top_k, shared_expert, shared_gate)) =
                            mlp.shared_metal_parts()
                        else {
                            return Ok(false);
                        };
                        layer_buffers.push(ResidentLayerBuffers::LinearMoe(
                            ResidentLinearMoeBuffers {
                                input_norm: metal.cached_buffer_from_f32(
                                    layer.input_norm.data(),
                                    "resident_linear_input_norm",
                                )?,
                                linear: metal.resolve_linear_attn_resident_dense_weights(
                                    linear.resident_weights(),
                                )?,
                                post_norm: metal.cached_buffer_from_f32(
                                    post_norm.data(),
                                    "resident_linear_post_norm",
                                )?,
                                moe: metal.resolve_moe_shared_weights(
                                    router,
                                    experts,
                                    shared_expert,
                                    shared_gate,
                                )?,
                                top_k,
                            },
                        ));
                    }
                    Some(FeedForward::Dense(mlp)) => {
                        let (gate_proj, up_proj, down_proj) = mlp.projections();
                        layer_buffers.push(ResidentLayerBuffers::LinearDense(
                            ResidentLinearDenseBuffers {
                                input_norm: metal.cached_buffer_from_f32(
                                    layer.input_norm.data(),
                                    "resident_dense_linear_input_norm",
                                )?,
                                linear: metal.resolve_linear_attn_resident_dense_weights(
                                    linear.resident_weights(),
                                )?,
                                post_norm: metal.cached_buffer_from_f32(
                                    post_norm.data(),
                                    "resident_dense_linear_post_norm",
                                )?,
                                gate_proj: metal.resolve_linear_weight_buffers(
                                    gate_proj.weight(),
                                    "resident_dense_gate_proj",
                                )?,
                                up_proj: metal.resolve_linear_weight_buffers(
                                    up_proj.weight(),
                                    "resident_dense_up_proj",
                                )?,
                                down_proj: metal.resolve_linear_weight_buffers(
                                    down_proj.weight(),
                                    "resident_dense_down_proj",
                                )?,
                            },
                        ));
                    }
                    _ => layer_buffers.push(ResidentLayerBuffers::Other),
                }
            }
        }
        let mut arena = DecodeResidentState::new(metal.device().clone())?;
        arena.set_scratch_namespace(slot);
        for (index, layer_cache) in cache.layers.iter_mut().enumerate() {
            if !self.config.is_full_attention_layer(index) {
                continue;
            }
            let mut full = arena.full_attention(capacity, q_heads, kv_heads, head_dim, sampled)?;
            full.seed(&layer_cache.keys, &layer_cache.values, prefill_len)?;
            layer_cache.full = Some(full);
        }
        let hidden_a = arena.persistent(hidden, GpuElement::F32)?;
        let hidden_b = arena.persistent(hidden, GpuElement::F32)?;
        let index = arena.persistent(1, GpuElement::U32)?;
        let mut index_ring = Vec::with_capacity(RESIDENT_PIPELINE_WINDOW);
        for _ in 0..RESIDENT_PIPELINE_WINDOW {
            index_ring.push(arena.persistent(1, GpuElement::U32)?);
        }
        let mtp = if let Some(head) = self.mtp.as_ref() {
            let mlp = match &head.layer.mlp {
                FeedForward::Dense(mlp) => mlp,
                _ => {
                    if crate::decoder::flags::trace_resident_enabled() {
                        eprintln!("decode résident full: arène MTP désactivée, tête MTP non dense");
                    }
                    cache.resident = Some(ResidentArena {
                        state: arena,
                        layers: layer_buffers,
                        embed_tokens: metal.resolve_embedding_weight_buffers(&self.embed_tokens)?,
                        final_norm: metal.cached_buffer_from_f32(
                            self.final_norm.data(),
                            "resident_final_norm",
                        )?,
                        lm_head: metal.resolve_linear_weight_buffers(
                            self.lm_head.weight(),
                            "resident_lm_head",
                        )?,
                        dense_tail_score: metal
                            .cached_buffer_from_f32(&[1.0], "resident_dense_tail_score")?,
                        hidden_a,
                        hidden_b,
                        index,
                        index_ring,
                        mtp: None,
                    });
                    return Ok(true);
                }
            };
            let (gate_proj, up_proj, down_proj) = mlp.projections();
            let q_norm =
                head.layer.attention.q_norm.as_ref().ok_or_else(|| {
                    InferError::Config("q_norm MTP manquant (résident)".to_string())
                })?;
            let k_norm =
                head.layer.attention.k_norm.as_ref().ok_or_else(|| {
                    InferError::Config("k_norm MTP manquant (résident)".to_string())
                })?;
            let v_proj =
                head.layer.attention.v_proj.as_ref().ok_or_else(|| {
                    InferError::Config("v_proj MTP manquant (résident)".to_string())
                })?;
            let qkv_proj = match metal.resolve_concat_linear_weight_buffers(
                &[
                    head.layer.attention.q_proj.weight(),
                    head.layer.attention.k_proj.weight(),
                    v_proj.weight(),
                ],
                "resident_mtp_qkv_proj",
            ) {
                Ok(weights) => Some(weights),
                Err(InferError::Dimension(_)) => None,
                Err(error) => return Err(error),
            };
            // KV de l'arène MTP : même dtype résolu que le KV principal (MTP est
            // greedy-only → `sampled` sera false quand la tête MTP tourne).
            let kv = arena.full_attention(capacity, q_heads, kv_heads, head_dim, sampled)?;
            #[cfg(feature = "devtools")]
            let append_oracle_kv =
                arena.full_attention(capacity, q_heads, kv_heads, head_dim, sampled)?;
            let hidden_a = arena.persistent(hidden, GpuElement::F32)?;
            let hidden_b = arena.persistent(hidden, GpuElement::F32)?;
            let index = arena.persistent(1, GpuElement::U32)?;
            let draft_indices = arena.persistent(max_new_tokens.max(1), GpuElement::U32)?;
            let verify_hidden_rows = arena.persistent(
                3usize.checked_mul(hidden).ok_or_else(|| {
                    InferError::Dimension("MTP verify hidden rows déborde".to_string())
                })?,
                GpuElement::F32,
            )?;
            let pending_append_indices = arena.persistent(2, GpuElement::U32)?;
            let embedding = arena.persistent(hidden, GpuElement::F32)?;
            let concat = arena.persistent(
                hidden.checked_mul(2).ok_or_else(|| {
                    InferError::Dimension("MTP concat hidden déborde".to_string())
                })?,
                GpuElement::F32,
            )?;
            let fc_out = arena.persistent(hidden, GpuElement::F32)?;
            Some(ResidentMtpArena {
                pre_fc_norm_embedding: metal.cached_buffer_from_f32(
                    head.pre_fc_norm_embedding.data(),
                    "resident_mtp_pre_fc_norm_embedding",
                )?,
                pre_fc_norm_hidden: metal.cached_buffer_from_f32(
                    head.pre_fc_norm_hidden.data(),
                    "resident_mtp_pre_fc_norm_hidden",
                )?,
                fc: metal.resolve_linear_weight_buffers(head.fc.weight(), "resident_mtp_fc")?,
                layer: ResidentFullDenseBuffers {
                    input_norm: metal.cached_buffer_from_f32(
                        head.layer.input_norm.data(),
                        "resident_mtp_input_norm",
                    )?,
                    qkv_proj,
                    q_proj: metal.resolve_linear_weight_buffers(
                        head.layer.attention.q_proj.weight(),
                        "resident_mtp_q_proj",
                    )?,
                    k_proj: metal.resolve_linear_weight_buffers(
                        head.layer.attention.k_proj.weight(),
                        "resident_mtp_k_proj",
                    )?,
                    v_proj: metal
                        .resolve_linear_weight_buffers(v_proj.weight(), "resident_mtp_v_proj")?,
                    o_proj: metal.resolve_linear_weight_buffers(
                        head.layer.attention.o_proj.weight(),
                        "resident_mtp_o_proj",
                    )?,
                    q_norm: metal.cached_buffer_from_f32(q_norm.data(), "resident_mtp_q_norm")?,
                    k_norm: metal.cached_buffer_from_f32(k_norm.data(), "resident_mtp_k_norm")?,
                    post_norm: metal.cached_buffer_from_f32(
                        head.layer.post_attention_norm.data(),
                        "resident_mtp_post_norm",
                    )?,
                    gate_proj: metal
                        .resolve_linear_weight_buffers(gate_proj.weight(), "resident_mtp_gate")?,
                    up_proj: metal
                        .resolve_linear_weight_buffers(up_proj.weight(), "resident_mtp_up")?,
                    down_proj: metal
                        .resolve_linear_weight_buffers(down_proj.weight(), "resident_mtp_down")?,
                },
                norm: metal.cached_buffer_from_f32(head.norm.data(), "resident_mtp_norm")?,
                draft_lm_head: metal.resolve_linear_weight_buffers(
                    self.mtp_draft_lm_head().weight(),
                    "resident_mtp_draft_lm_head",
                )?,
                kv,
                #[cfg(feature = "devtools")]
                append_oracle_kv,
                #[cfg(feature = "devtools")]
                append_oracle_len: 0,
                hidden_a,
                hidden_b,
                current_is_a: true,
                index,
                draft_indices,
                verify_hidden_rows,
                pending_append_indices,
                pending_append_start: 0,
                pending_append_count: 0,
                embedding,
                concat,
                fc_out,
            })
        } else {
            None
        };
        cache.resident = Some(ResidentArena {
            state: arena,
            hidden_a,
            hidden_b,
            index,
            index_ring,
            layers: layer_buffers,
            dense_tail_score: metal.cached_buffer_from_f32(&[1.0], "resident_dense_tail_score")?,
            embed_tokens: metal.resolve_embedding_weight_buffers(&self.embed_tokens)?,
            final_norm: metal
                .cached_buffer_from_f32(self.final_norm.data(), "resident_final_norm")?,
            lm_head: metal
                .resolve_linear_weight_buffers(self.lm_head.weight(), "resident_lm_head")?,
            mtp,
        });
        Ok(true)
    }
}
