use super::super::*;
#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::metal_backend::MetalExecutor;

impl CausalDecoder {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[expect(
        clippy::too_many_arguments,
        reason = "routage résident: état couche + arena + ping-pong + encoder"
    )]
    pub(super) fn encode_resident_full_layer(
        &self,
        metal: &MetalExecutor,
        arena: &ResidentArena,
        layer_cache: &mut LayerKvCache,
        layer: &DecoderLayer,
        index: usize,
        encoder: &metal::ComputeCommandEncoderRef,
        owned: &mut Vec<metal::Buffer>,
        dims: FullAttnLayerDims,
        layer_in: &metal::BufferRef,
        layer_out: &metal::BufferRef,
    ) -> Result<()> {
        let AttentionBlock::Full(attention) = &layer.attention else {
            return Err(InferError::Config(
                "couche full-attn attendue (decode résident)".to_string(),
            ));
        };
        if attention.q_norm.is_none() || attention.k_norm.is_none() {
            return Err(InferError::Config(
                "q_norm/k_norm manquant (decode résident)".to_string(),
            ));
        }
        if layer.post_attention_norm.is_none() {
            return Err(InferError::Config(
                "post_norm manquant (decode résident)".to_string(),
            ));
        }
        let kv = layer_cache.full.as_mut().ok_or_else(|| {
            InferError::Metal("KV full-attn résident absent (decode résident)".to_string())
        })?;
        match layer.mlp.as_ref() {
            Some(FeedForward::Moe(_)) => {
                match arena.layers.get(index).ok_or_else(|| {
                    InferError::Config(format!("poids résidents couche {index} absents"))
                })? {
                    ResidentLayerBuffers::FullMoe(resolved) => {
                        let weights = FullAttnLayerWeights {
                            input_norm: &resolved.input_norm,
                            qkv_proj: resolved.qkv_proj.as_ref(),
                            q_proj: &resolved.q_proj,
                            k_proj: &resolved.k_proj,
                            v_proj: &resolved.v_proj,
                            o_proj: &resolved.o_proj,
                            q_norm: &resolved.q_norm,
                            k_norm: &resolved.k_norm,
                            post_norm: &resolved.post_norm,
                            moe: &resolved.moe,
                            top_k: resolved.top_k,
                        };
                        arena.state.encode_full_attn_layer(
                            metal, encoder, owned, kv, weights, dims, layer_in, layer_out,
                        )
                    }
                    ResidentLayerBuffers::FullRouted(resolved) => {
                        let weights = FullAttnRoutedLayerWeights {
                            input_norm: &resolved.input_norm,
                            qkv_proj: &resolved.qkv_proj,
                            o_proj: &resolved.o_proj,
                            q_norm: &resolved.q_norm,
                            k_norm: &resolved.k_norm,
                            post_norm: &resolved.post_norm,
                            moe: &resolved.moe,
                            top_k: resolved.top_k,
                        };
                        arena.state.encode_full_attn_routed_layer(
                            metal, encoder, owned, kv, weights, dims, layer_in, layer_out,
                        )
                    }
                    _ => Err(InferError::Config(format!(
                        "poids full-attn MoE résidents absents couche {index}"
                    ))),
                }
            }
            Some(FeedForward::Dense(_)) => {
                let ResidentLayerBuffers::FullDense(resolved) =
                    arena.layers.get(index).ok_or_else(|| {
                        InferError::Config(format!("poids résidents couche {index} absents"))
                    })?
                else {
                    return Err(InferError::Config(format!(
                        "poids full-attn dense résidents absents couche {index}"
                    )));
                };
                let weights = FullAttnDenseLayerWeights {
                    input_norm: &resolved.input_norm,
                    qkv_proj: resolved.qkv_proj.as_ref(),
                    q_proj: &resolved.q_proj,
                    k_proj: &resolved.k_proj,
                    v_proj: &resolved.v_proj,
                    o_proj: &resolved.o_proj,
                    q_norm: &resolved.q_norm,
                    k_norm: &resolved.k_norm,
                    post_norm: &resolved.post_norm,
                    gate_proj: &resolved.gate_proj,
                    up_proj: &resolved.up_proj,
                    down_proj: &resolved.down_proj,
                    tail_score: &arena.dense_tail_score,
                };
                arena.state.encode_full_attn_dense_layer(
                    metal, encoder, owned, kv, weights, dims, layer_in, layer_out,
                )
            }
            None => Err(InferError::Config(
                "MLP attendu (decode résident)".to_string(),
            )),
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[expect(
        clippy::too_many_arguments,
        reason = "routage résident: état couche + arena + ping-pong + encoder"
    )]
    pub(super) fn encode_resident_linear_layer(
        &self,
        metal: &MetalExecutor,
        arena: &ResidentArena,
        layer_cache: &mut LayerKvCache,
        layer: &DecoderLayer,
        index: usize,
        encoder: &metal::ComputeCommandEncoderRef,
        owned: &mut Vec<metal::Buffer>,
        la_spec: LinearAttentionStepSpec,
        res_dims: LinearAttnResidentDims,
        hidden: usize,
        eps: f32,
        layer_in: &metal::BufferRef,
        layer_out: &metal::BufferRef,
    ) -> Result<()> {
        let AttentionBlock::Linear(_) = &layer.attention else {
            return Err(InferError::Config(
                "couche linear-attn attendue (decode résident)".to_string(),
            ));
        };
        if layer.post_attention_norm.is_none() {
            return Err(InferError::Config(
                "post_norm manquant (decode résident)".to_string(),
            ));
        }
        let state = layer_cache.linear.metal_state().ok_or_else(|| {
            InferError::Metal("état linear-attn résident absent (decode résident)".to_string())
        })?;
        match layer.mlp.as_ref() {
            Some(FeedForward::Moe(_)) => {
                let ResidentLayerBuffers::LinearMoe(resolved) =
                    arena.layers.get(index).ok_or_else(|| {
                        InferError::Config(format!("poids résidents couche {index} absents"))
                    })?
                else {
                    return Err(InferError::Config(format!(
                        "poids linear-attn MoE résidents absents couche {index}"
                    )));
                };
                let weights = LinearAttnLayerWeights {
                    input_norm: &resolved.input_norm,
                    linear: &resolved.linear,
                    post_norm: &resolved.post_norm,
                    moe: &resolved.moe,
                    top_k: resolved.top_k,
                };
                arena.state.encode_linear_attn_layer(
                    metal, encoder, owned, state, weights, la_spec, res_dims, hidden, eps,
                    layer_in, layer_out,
                )
            }
            Some(FeedForward::Dense(_)) => {
                let ResidentLayerBuffers::LinearDense(resolved) =
                    arena.layers.get(index).ok_or_else(|| {
                        InferError::Config(format!("poids résidents couche {index} absents"))
                    })?
                else {
                    return Err(InferError::Config(format!(
                        "poids linear-attn dense résidents absents couche {index}"
                    )));
                };
                let weights = LinearAttnDenseLayerWeights {
                    input_norm: &resolved.input_norm,
                    linear: &resolved.linear,
                    post_norm: &resolved.post_norm,
                    gate_proj: &resolved.gate_proj,
                    up_proj: &resolved.up_proj,
                    down_proj: &resolved.down_proj,
                    tail_score: &arena.dense_tail_score,
                };
                arena.state.encode_linear_attn_dense_layer(
                    metal, encoder, owned, state, weights, la_spec, res_dims, hidden, eps,
                    layer_in, layer_out,
                )
            }
            None => Err(InferError::Config(
                "MLP attendu (decode résident)".to_string(),
            )),
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[allow(
        dead_code,
        reason = "prototype full-attn batché gardé hors hot path après mesure défavorable"
    )]
    #[expect(
        clippy::too_many_arguments,
        reason = "routage full-attn résident batché: état couche + arena + ping-pong + encoder"
    )]
    pub(super) fn encode_resident_full_dense_layer_rows(
        &self,
        metal: &MetalExecutor,
        arena: &ResidentArena,
        layer_cache: &mut LayerKvCache,
        layer: &DecoderLayer,
        index: usize,
        encoder: &metal::ComputeCommandEncoderRef,
        owned: &mut Vec<metal::Buffer>,
        dims: FullAttnLayerDims,
        rows: usize,
        layer_in: &metal::BufferRef,
        layer_out: &metal::BufferRef,
    ) -> Result<()> {
        let AttentionBlock::Full(attention) = &layer.attention else {
            return Err(InferError::Config(
                "couche full-attn attendue (verify résident batché)".to_string(),
            ));
        };
        if attention.q_norm.is_none() || attention.k_norm.is_none() {
            return Err(InferError::Config(
                "q_norm/k_norm manquant (verify résident batché)".to_string(),
            ));
        }
        if !matches!(layer.mlp.as_ref(), Some(FeedForward::Dense(_))) {
            return Err(InferError::Config(
                "MLP dense attendu (verify full résident batché)".to_string(),
            ));
        }
        if layer.post_attention_norm.is_none() {
            return Err(InferError::Config(
                "post_norm manquant (verify full résident batché)".to_string(),
            ));
        }
        let kv = layer_cache.full.as_mut().ok_or_else(|| {
            InferError::Metal("KV full-attn résident absent (verify batché)".to_string())
        })?;
        let ResidentLayerBuffers::FullDense(resolved) = arena
            .layers
            .get(index)
            .ok_or_else(|| InferError::Config(format!("poids résidents couche {index} absents")))?
        else {
            return Err(InferError::Config(format!(
                "poids full-attn dense résidents absents couche {index}"
            )));
        };
        let weights = FullAttnDenseLayerWeights {
            input_norm: &resolved.input_norm,
            qkv_proj: resolved.qkv_proj.as_ref(),
            q_proj: &resolved.q_proj,
            k_proj: &resolved.k_proj,
            v_proj: &resolved.v_proj,
            o_proj: &resolved.o_proj,
            q_norm: &resolved.q_norm,
            k_norm: &resolved.k_norm,
            post_norm: &resolved.post_norm,
            gate_proj: &resolved.gate_proj,
            up_proj: &resolved.up_proj,
            down_proj: &resolved.down_proj,
            tail_score: &arena.dense_tail_score,
        };
        arena.state.encode_full_attn_dense_layer_rows(
            metal, encoder, owned, kv, weights, dims, rows, layer_in, layer_out,
        )
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[expect(
        clippy::too_many_arguments,
        reason = "routage résident batché: état couche + arena + ping-pong + encoder"
    )]
    pub(super) fn encode_resident_linear_layer_rows(
        &self,
        metal: &MetalExecutor,
        arena: &ResidentArena,
        layer_cache: &mut LayerKvCache,
        layer: &DecoderLayer,
        index: usize,
        encoder: &metal::ComputeCommandEncoderRef,
        owned: &mut Vec<metal::Buffer>,
        la_spec: LinearAttentionStepSpec,
        res_dims: LinearAttnResidentDims,
        rows: usize,
        hidden: usize,
        eps: f32,
        layer_in: &metal::BufferRef,
        layer_out: &metal::BufferRef,
        captures: Option<&[LinearAttentionMetalState]>,
    ) -> Result<()> {
        let AttentionBlock::Linear(_) = &layer.attention else {
            return Err(InferError::Config(
                "couche linear-attn attendue (verify résident batché)".to_string(),
            ));
        };
        if !matches!(layer.mlp.as_ref(), Some(FeedForward::Moe(_))) {
            return Err(InferError::Config(
                "MLP MoE attendu (verify résident batché)".to_string(),
            ));
        }
        if layer.post_attention_norm.is_none() {
            return Err(InferError::Config(
                "post_norm manquant (verify résident batché)".to_string(),
            ));
        }
        let state = layer_cache.linear.metal_state().ok_or_else(|| {
            InferError::Metal("état linear-attn résident absent (verify batché)".to_string())
        })?;
        let ResidentLayerBuffers::LinearMoe(resolved) = arena
            .layers
            .get(index)
            .ok_or_else(|| InferError::Config(format!("poids résidents couche {index} absents")))?
        else {
            return Err(InferError::Config(format!(
                "poids linear-attn MoE résidents absents couche {index}"
            )));
        };
        let weights = LinearAttnLayerWeights {
            input_norm: &resolved.input_norm,
            linear: &resolved.linear,
            post_norm: &resolved.post_norm,
            moe: &resolved.moe,
            top_k: resolved.top_k,
        };
        arena.state.encode_linear_attn_layer_rows(
            metal, encoder, owned, state, weights, la_spec, res_dims, rows, hidden, eps, layer_in,
            layer_out, captures,
        )
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[expect(
        clippy::too_many_arguments,
        reason = "routage résident batché: état couche + arena + ping-pong + encoder"
    )]
    pub(super) fn encode_resident_linear_dense_layer_rows(
        &self,
        metal: &MetalExecutor,
        arena: &ResidentArena,
        layer_cache: &mut LayerKvCache,
        layer: &DecoderLayer,
        index: usize,
        encoder: &metal::ComputeCommandEncoderRef,
        owned: &mut Vec<metal::Buffer>,
        la_spec: LinearAttentionStepSpec,
        res_dims: LinearAttnResidentDims,
        rows: usize,
        hidden: usize,
        eps: f32,
        layer_in: &metal::BufferRef,
        layer_out: &metal::BufferRef,
        captures: Option<&[LinearAttentionMetalState]>,
    ) -> Result<()> {
        let AttentionBlock::Linear(_) = &layer.attention else {
            return Err(InferError::Config(
                "couche linear-attn attendue (verify résident batché)".to_string(),
            ));
        };
        if !matches!(layer.mlp.as_ref(), Some(FeedForward::Dense(_))) {
            return Err(InferError::Config(
                "MLP dense attendu (verify résident batché)".to_string(),
            ));
        }
        if layer.post_attention_norm.is_none() {
            return Err(InferError::Config(
                "post_norm manquant (verify résident batché)".to_string(),
            ));
        }
        let state = layer_cache.linear.metal_state().ok_or_else(|| {
            InferError::Metal("état linear-attn résident absent (verify batché)".to_string())
        })?;
        let ResidentLayerBuffers::LinearDense(resolved) = arena
            .layers
            .get(index)
            .ok_or_else(|| InferError::Config(format!("poids résidents couche {index} absents")))?
        else {
            return Err(InferError::Config(format!(
                "poids linear-attn dense résidents absents couche {index}"
            )));
        };
        let weights = LinearAttnDenseLayerWeights {
            input_norm: &resolved.input_norm,
            linear: &resolved.linear,
            post_norm: &resolved.post_norm,
            gate_proj: &resolved.gate_proj,
            up_proj: &resolved.up_proj,
            down_proj: &resolved.down_proj,
            tail_score: &arena.dense_tail_score,
        };
        arena.state.encode_linear_attn_dense_layer_rows(
            metal, encoder, owned, state, weights, la_spec, res_dims, rows, hidden, eps, layer_in,
            layer_out, captures,
        )
    }

    /// Microbench (tranche 3, `RETI_RUST_GPU_COUNTERS`) isolant le **MoE** et le
    /// **surcoût commit/wait par command buffer** — les deux inconnues que la
    /// segmentation per-couche ne sépare pas (chaque couche = attn + MoE + 1 CB).
    ///
    /// `overhead` = K command buffers VIDES (commit/wait seul). `moe` = K MoE
    /// résidents (couche 0) via [`MetalExecutor::moe_gated_router_topk_shared`]
    /// (1 CB chacun). `MoE pur ≈ moe − overhead` ; permet de retrancher le MoE des
    /// temps de couche pour isoler le mécanisme d'attention. Mesure pure (aucun
    /// effet sur la génération). `None` si le modèle/backend ne s'y prête pas.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(in crate::decoder) fn profile_moe_and_overhead(
        &self,
        queue: &metal::CommandQueueRef,
    ) -> Option<String> {
        let metal = self.forward_runtime().metal_executor()?;
        let Some(FeedForward::Moe(mlp)) = self.layers.first()?.mlp.as_ref() else {
            return None;
        };
        let (router, experts, top_k, shared_expert, shared_gate) = mlp.shared_metal_parts()?;
        // Entrées VARIÉES (embeddings de tokens distincts) → le routeur sélectionne
        // des experts différents à chaque itération → lectures de poids FROIDES
        // (réalistes) au lieu d'un seul jeu d'experts cache-résident.
        let inputs: Vec<Tensor> = (0..64_usize)
            .filter_map(|token| embed_weight_tokens(&self.embed_tokens, &[token]).ok())
            .collect();
        if inputs.is_empty() {
            return None;
        }
        let iters = 256_u32;
        let warmup = 32_u32;
        let empty_cb = || -> Result<()> {
            let command_buffer = queue.new_command_buffer();
            command_buffer.new_compute_command_encoder().end_encoding();
            crate::metal_backend::commit_and_wait(command_buffer)
        };
        for _ in 0..warmup {
            empty_cb().ok()?;
        }
        let overhead_started = Instant::now();
        for _ in 0..iters {
            empty_cb().ok()?;
        }
        let overhead_ms = overhead_started.elapsed().as_secs_f64() * 1000.0 / f64::from(iters);
        let run_moe = |step: usize| {
            metal.moe_gated_router_topk_shared(
                &inputs[step % inputs.len()],
                router,
                experts,
                top_k,
                shared_expert,
                shared_gate,
            )
        };
        for step in 0..warmup as usize {
            run_moe(step).ok()?;
        }
        let moe_started = Instant::now();
        for step in 0..iters as usize {
            run_moe(step).ok()?;
        }
        let moe_ms = moe_started.elapsed().as_secs_f64() * 1000.0 / f64::from(iters);
        let moe_pure = (moe_ms - overhead_ms).max(0.0);
        let layers = self.layers.len() as f64;
        let split = if crate::decoder::flags::moe_micro_split_enabled() {
            match metal.profile_moe_shared_segments(
                inputs.first()?,
                router,
                experts,
                top_k,
                shared_expert,
                shared_gate,
                overhead_ms,
            ) {
                Ok(report) => format!(" | {report}"),
                Err(error) => format!(" | split MoE erreur: {error}"),
            }
        } else {
            String::new()
        };
        let linear_split = if crate::decoder::flags::linear_micro_split_enabled() {
            let report = (|| -> Option<String> {
                let la_config = self.config.linear_attention_config().ok()?;
                let la_spec = LinearAttentionStepSpec {
                    num_key_heads: la_config.num_key_heads,
                    num_value_heads: la_config.num_value_heads,
                    key_head_dim: la_config.key_head_dim,
                    value_head_dim: la_config.value_head_dim,
                    conv_kernel_dim: la_config.conv_kernel_dim,
                    rms_eps: la_config.rms_eps,
                };
                let key_dim = la_config.key_dim().ok()?;
                let value_dim = la_config.value_dim().ok()?;
                let conv_dim = key_dim.checked_mul(2)?.checked_add(value_dim)?;
                let input = inputs.first()?;
                let (_, hidden) = input.as_matrix().ok()?;
                let (layer_index, layer, attention) =
                    self.layers.iter().enumerate().find_map(|(index, layer)| {
                        let AttentionBlock::Linear(attention) = &layer.attention else {
                            return None;
                        };
                        Some((index, layer, attention.as_ref()))
                    })?;
                let dims = LinearAttnResidentDims {
                    in_dim: hidden,
                    conv_dim,
                    value_dim,
                    key_dim,
                };
                Some(
                    match metal.profile_linear_attn_dense_segments(
                        input,
                        Some((&layer.input_norm, self.config.rms_eps)),
                        attention.resident_weights(),
                        la_spec,
                        dims,
                        overhead_ms,
                    ) {
                        Ok(report) => format!(" | couche linear-attn {layer_index}: {report}"),
                        Err(error) => format!(" | split linear-attn erreur: {error}"),
                    },
                )
            })()
            .unwrap_or_else(|| " | split linear-attn indisponible".to_string());
            report
        } else {
            String::new()
        };
        Some(format!(
            "gpu microbench (couche 0, {iters} itér) : overhead/CB {overhead_ms:.3} ms | \
             MoE+CB {moe_ms:.3} ms | MoE pur ≈ {moe_pure:.3} ms/couche → \
             MoE total ≈ {total:.3} ms/token (×{n}){split}{linear_split}",
            total = moe_pure * layers,
            n = self.layers.len(),
        ))
    }
}
