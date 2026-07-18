//! Pas de decode duo (light-batch M=2, E2.2) : 2 flux dans UN command buffer.
//!
//! Les projections denses (qkv, o, in/out-proj linear-attn, lm_head) sont
//! batchées en qmm2 — les poids sont lus UNE fois pour les deux flux — tandis
//! que RoPE, append KV, attention, conv/SSM, MoE et argmax restent des
//! dispatches PAR FLUX sur l'état de chaque flux. La composition est BIT-exacte
//! vs le pas solo (oracles E2.2a) → byte-identité par flux.

#[cfg(all(target_os = "macos", feature = "metal"))]
use super::*;

#[cfg(all(target_os = "macos", feature = "metal"))]
impl CausalDecoder {
    /// Renvoie `true` si le pas duo est applicable au cache résident : couches
    /// exclusivement FullMoe/LinearMoe (35B-A3B), `attn_output_gate`, et TOUTES
    /// les projections batchées routables qmm2 (sinon le repli silencieux vers
    /// le matmul générique casserait la byte-identité).
    pub(super) fn supports_resident_duo(&self, cache: &CausalDecoderCache) -> bool {
        let Some(metal) = self.forward_runtime().metal_executor() else {
            return false;
        };
        let Some(arena) = cache.resident.as_ref() else {
            return false;
        };
        if !self.config.attn_output_gate || self.config.head_dim.is_none() {
            return false;
        }
        if self.final_norm.data().len() % 512 != 0 {
            return false;
        }
        if !metal.qmm2_eligible_weight(&arena.lm_head) {
            return false;
        }
        arena
            .layers
            .iter()
            .enumerate()
            .all(|(index, layer)| match layer {
                ResidentLayerBuffers::FullMoe(resolved) => {
                    self.config.is_resident_full_attention_layer(index)
                        && resolved
                            .qkv_proj
                            .as_ref()
                            .is_some_and(|qkv| metal.qmm2_eligible_weight(qkv))
                        && metal.qmm2_eligible_weight(&resolved.o_proj)
                }
                ResidentLayerBuffers::LinearMoe(resolved) => {
                    self.layers
                        .get(index)
                        .is_some_and(|layer| matches!(&layer.attention, AttentionBlock::Linear(_)))
                        && metal.qmm2_eligible_linear_attn(&resolved.linear)
                }
                _ => false,
            })
    }

    /// Décode UN token pour CHACUN des deux flux dans un command buffer unique :
    /// argmax on-device en greedy (`samples = None`), sampler GPU top-k/top-p
    /// PAR FLUX sinon (E2.4, `rng_state` de chaque flux). Renvoie
    /// `[token_a, token_b]`.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si une précondition duo manque (arène, état par
    /// couche, dimensions) ou si un encodage Metal échoue.
    pub(super) fn decode_tokens_resident_duo(
        &self,
        cache_a: &mut CausalDecoderCache,
        cache_b: &mut CausalDecoderCache,
        tokens: [usize; 2],
        samples: Option<[super::resident::ResidentSampleSpec; 2]>,
    ) -> Result<[usize; 2]> {
        let Some(metal) = self.forward_runtime().metal_executor() else {
            return Err(InferError::Metal(
                "decode duo sans executor Metal".to_string(),
            ));
        };
        let theta = self
            .config
            .rope_theta
            .ok_or_else(|| InferError::Config("rope_theta manquant (decode duo)".to_string()))?;
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
            };
            let key_dim = la_config.key_dim()?;
            let value_dim = la_config.value_dim()?;
            let conv_dim = key_dim
                .checked_mul(2)
                .and_then(|twice| twice.checked_add(value_dim))
                .ok_or_else(|| InferError::Shape("conv_dim déborde (decode duo)".to_string()))?;
            Some((la_spec, key_dim, value_dim, conv_dim))
        } else {
            None
        };

        let CausalDecoderCache {
            layers: layers_a,
            position: position_a,
            resident: resident_a,
        } = cache_a;
        let CausalDecoderCache {
            layers: layers_b,
            position: position_b,
            resident: resident_b,
        } = cache_b;
        let arena = resident_a
            .as_mut()
            .ok_or_else(|| InferError::Metal("arène résidente A absente (duo)".to_string()))?;
        let arena_b = resident_b
            .as_ref()
            .ok_or_else(|| InferError::Metal("arène résidente B absente (duo)".to_string()))?;
        let positions = [*position_a, *position_b];
        let slots = [
            arena.state.scratch_namespace(),
            arena_b.state.scratch_namespace(),
        ];

        // Embed des 2 tokens (CPU, input upload, scale Gemma appliqué comme le
        // solo) vers le duo ping-pong.
        let embed = self.embed_scaled(&tokens)?;
        let (batch, embed_hidden) = embed.as_matrix()?;
        if batch != 2 || embed_hidden != hidden {
            return Err(InferError::Dimension(format!(
                "embedding duo [{batch},{embed_hidden}], attendu [2,{hidden}]"
            )));
        }
        let duo_a = arena.state.scratch().lease(2 * hidden, GpuElement::F32)?;
        let duo_b = arena.state.scratch().lease(2 * hidden, GpuElement::F32)?;
        arena.state.upload(duo_a.tensor(), embed.data())?;

        let command_buffer = arena.state.queue().new_command_buffer();
        let _barrier_guard = crate::metal_backend::resident_concurrent_enabled()
            .then(crate::metal_backend::install_dispatch_barrier_scope);
        let _namespace_guard =
            crate::metal_backend::install_scratch_namespace(arena.state.scratch_namespace());
        // Diagnostic disjonction d'experts : arme la collecte des indices top-k
        // par couche MoE duo (drainée après le wait, AUCUN effet sur les valeurs).
        let expert_stats = super::flags::lightbatch_expert_stats_enabled();
        if expert_stats {
            crate::metal_backend::begin_expert_indices_collection();
        }
        let encoder = new_resident_compute_encoder(command_buffer);
        let encoder_guard = crate::metal_backend::EncoderEndGuard::new(encoder);
        let mut owned: Vec<metal::Buffer> = Vec::new();

        let mut current = &duo_a;
        let mut other = &duo_b;
        for index in 0..self.layers.len() {
            let layer_cache_a = &mut layers_a[index];
            let layer_cache_b = &mut layers_b[index];
            if self.config.is_resident_full_attention_layer(index) {
                let ResidentLayerBuffers::FullMoe(resolved) =
                    arena.layers.get(index).ok_or_else(|| {
                        InferError::Config(format!("poids résidents couche {index} absents"))
                    })?
                else {
                    return Err(InferError::Config(format!(
                        "duo: couche full-attn {index} non MoE shared"
                    )));
                };
                let kv_a = layer_cache_a
                    .full
                    .as_mut()
                    .ok_or_else(|| InferError::Metal("KV full-attn A absent (duo)".to_string()))?;
                let kv_b = layer_cache_b
                    .full
                    .as_mut()
                    .ok_or_else(|| InferError::Metal("KV full-attn B absent (duo)".to_string()))?;
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
                    pre_feedforward_norm: resolved.pre_feedforward_norm.as_ref(),
                    post_feedforward_norm: resolved.post_feedforward_norm.as_ref(),
                    layer_scalar: resolved.layer_scalar,
                    moe: &resolved.moe,
                    top_k: resolved.top_k,
                };
                let dims = self.config.resident_full_attn_layer_dims(
                    index,
                    hidden,
                    positions[0],
                    eps,
                    theta,
                )?;
                arena.state.encode_full_attn_moe_layer_duo(
                    metal,
                    encoder_guard.encoder(),
                    &mut owned,
                    [kv_a, kv_b],
                    weights,
                    dims,
                    positions,
                    slots,
                    current.tensor().buffer(),
                    other.tensor().buffer(),
                )?;
            } else {
                let Some((la_spec, key_dim, value_dim, conv_dim)) = linear_dims else {
                    return Err(InferError::Config(
                        "dims linear-attn duo absentes".to_string(),
                    ));
                };
                let ResidentLayerBuffers::LinearMoe(resolved) =
                    arena.layers.get(index).ok_or_else(|| {
                        InferError::Config(format!("poids résidents couche {index} absents"))
                    })?
                else {
                    return Err(InferError::Config(format!(
                        "duo: couche linear-attn {index} non MoE shared"
                    )));
                };
                let state_a = layer_cache_a.linear.metal_state().ok_or_else(|| {
                    InferError::Metal("état linear-attn A absent (duo)".to_string())
                })?;
                let state_b = layer_cache_b.linear.metal_state().ok_or_else(|| {
                    InferError::Metal("état linear-attn B absent (duo)".to_string())
                })?;
                let weights = LinearAttnLayerWeights {
                    input_norm: &resolved.input_norm,
                    linear: &resolved.linear,
                    post_norm: &resolved.post_norm,
                    moe: &resolved.moe,
                    top_k: resolved.top_k,
                };
                let res_dims = LinearAttnResidentDims {
                    in_dim: hidden,
                    conv_dim,
                    value_dim,
                    key_dim,
                };
                arena.state.encode_linear_attn_moe_layer_duo(
                    metal,
                    encoder_guard.encoder(),
                    &mut owned,
                    [state_a, state_b],
                    weights,
                    la_spec,
                    res_dims,
                    hidden,
                    eps,
                    slots,
                    current.tensor().buffer(),
                    other.tensor().buffer(),
                )?;
            }
            std::mem::swap(&mut current, &mut other);
        }

        // final_norm rows=2 (kernel du solo, bit-identique par row) → lm_head
        // qmm2 + argmax (greedy) ou sampler top-k/top-p (rng par flux) → 2 u32.
        let final2 = arena.state.scratch().lease(2 * hidden, GpuElement::F32)?;
        metal.encode_rms_norm_rows(
            encoder_guard.encoder(),
            current.tensor().buffer(),
            &arena.final_norm,
            final2.tensor().buffer(),
            2,
            hidden,
            eps,
        )?;
        let index2 = arena.state.scratch().lease(2, GpuElement::U32)?;
        match samples {
            Some(specs) => {
                let params = specs.map(|spec| crate::metal_backend::DuoSampleParams {
                    temperature: spec.temperature,
                    top_p: spec.top_p,
                    top_k: spec.top_k,
                    rng_state: spec.rng_state,
                });
                metal.encode_lm_head_sample_duo_buffers(
                    encoder_guard.encoder(),
                    final2.tensor().buffer(),
                    &arena.lm_head,
                    index2.tensor().buffer(),
                    hidden,
                    params,
                    slots,
                )?;
            }
            None => {
                metal.encode_lm_head_argmax_duo_buffers(
                    encoder_guard.encoder(),
                    final2.tensor().buffer(),
                    &arena.lm_head,
                    index2.tensor().buffer(),
                    hidden,
                    slots,
                )?;
            }
        }
        encoder_guard.end();
        crate::metal_backend::commit_and_wait(command_buffer)?;
        drop(owned);
        if expert_stats {
            if let Some(pairs) = crate::metal_backend::take_expert_indices_collection() {
                super::lightbatch::record_expert_stats(&pairs);
            }
        }

        let raw = crate::metal_backend::read_u32_buffer(index2.tensor().buffer(), 2)?;
        let [token_a, token_b] = raw.as_slice() else {
            return Err(InferError::Metal("decode duo sans 2 tokens".to_string()));
        };
        *position_a += 1;
        *position_b += 1;
        Ok([
            usize::try_from(*token_a)
                .map_err(|_| InferError::Metal(format!("token duo A hors usize: {token_a}")))?,
            usize::try_from(*token_b)
                .map_err(|_| InferError::Metal(format!("token duo B hors usize: {token_b}")))?,
        ])
    }
}
