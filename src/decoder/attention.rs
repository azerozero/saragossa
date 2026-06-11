//! Forward attention, couches decodeur et cache K/V.

use super::attention_ops::*;
use super::*;

impl DecoderLayer {
    pub(super) fn from_tensors(
        tensors: &mut HashMap<String, DecoderTensor>,
        layer_index: usize,
        config: &CausalDecoderConfig,
    ) -> Result<Self> {
        let prefix = format!("layers.{layer_index}");
        let input_norm = take_dense(tensors, &format!("{prefix}.input_layernorm.weight"))?;
        let attention = if config.is_full_attention_layer(layer_index) {
            full_attention_from_tensors(tensors, &prefix)?
        } else {
            linear_attention_from_tensors(tensors, &prefix)?
        };
        let (post_attention_norm, mlp) = optional_mlp_from(config, tensors, &prefix)?;
        Ok(Self {
            input_norm,
            attention,
            post_attention_norm,
            mlp,
        })
    }

    pub(super) fn forward(
        &self,
        config: &CausalDecoderConfig,
        x: &Tensor,
        runtime: ForwardRuntime<'_>,
    ) -> Result<Tensor> {
        let normed = rms_norm(x, &self.input_norm, config.rms_eps)?;
        let attn_out = match &self.attention {
            AttentionBlock::Full(attention) => {
                full_attention_forward(config, &normed, attention, runtime)?
            }
            AttentionBlock::Linear(linear) => {
                linear.forward_with_runtime(config.linear_attention_config()?, &normed, runtime)?
            }
        };
        let attention_state = x.add(&attn_out)?;
        match (&self.post_attention_norm, &self.mlp) {
            (Some(norm), Some(mlp)) => {
                let mlp_input = rms_norm(&attention_state, norm, config.rms_eps)?;
                attention_state.add(&mlp.forward_with_runtime(&mlp_input, runtime)?)
            }
            (None, None) => Ok(attention_state),
            _ => Err(InferError::Config(
                "bloc MLP partiellement initialisé".to_string(),
            )),
        }
    }

    pub(super) fn supports_batched_prefill(&self) -> bool {
        matches!(self.attention, AttentionBlock::Full(_))
    }

    /// Renvoie `true` si la couche peut être encodée ENTIÈREMENT en kernels
    /// résidents (chaînés, sans readback ni fallback CPU au milieu — réserve Codex
    /// MAJEUR 6). Validé EN AMONT par [`CausalDecoder::supports_resident_full_decode`]
    /// pour garantir le tout-ou-rien du command buffer unique de 1c.
    ///
    /// Exige : post-norm + tail MLP encodable (MoE shared-expert ou Dense
    /// gate/up/down biasless) ; pour full-attn, projections Q/K/V/O biasless et
    /// q_norm/k_norm présents (le kernel fusionné rms_norm+RoPE l'exige). Le
    /// shared-expert/router biasless est re-vérifié à l'encodage (`ensure_biasless`).
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(super) fn supports_resident_full(&self) -> bool {
        let Some(mlp) = self.mlp.as_ref() else {
            return false;
        };
        if self.post_attention_norm.is_none() {
            return false;
        }
        let mlp_supported = match mlp {
            FeedForward::Moe(mlp) => {
                mlp.shared_metal_parts().is_some() || mlp.metal_parts().is_some()
            }
            FeedForward::Dense(mlp) => {
                let (gate_proj, up_proj, down_proj) = mlp.projections();
                gate_proj.bias().is_none() && up_proj.bias().is_none() && down_proj.bias().is_none()
            }
        };
        if !mlp_supported {
            return false;
        }
        match &self.attention {
            AttentionBlock::Full(attention) => {
                attention.q_proj.bias().is_none()
                    && attention.k_proj.bias().is_none()
                    && attention.v_proj.bias().is_none()
                    && attention.o_proj.bias().is_none()
                    && attention.q_norm.is_some()
                    && attention.k_norm.is_some()
            }
            // Linear-attn : chemin résident conv/ssm déjà éprouvé (phases 1a/1b).
            AttentionBlock::Linear(_) => true,
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(super) fn prefill_moe_layer(&self) -> Option<crate::metal_backend::PrefillMoeLayer<'_>> {
        let AttentionBlock::Full(attention) = &self.attention else {
            return None;
        };
        let q_norm = attention.q_norm.as_ref()?;
        let k_norm = attention.k_norm.as_ref()?;
        let post_norm = self.post_attention_norm.as_ref()?;
        let Some(FeedForward::Moe(mlp)) = self.mlp.as_ref() else {
            return None;
        };
        let (router, experts, top_k) = mlp.metal_parts()?;
        Some(crate::metal_backend::PrefillMoeLayer {
            input_norm: &self.input_norm,
            q_proj: &attention.q_proj,
            k_proj: &attention.k_proj,
            v_proj: &attention.v_proj,
            o_proj: &attention.o_proj,
            q_norm,
            k_norm,
            post_norm,
            router,
            experts,
            top_k,
        })
    }

    pub(super) fn forward_prefill(
        &self,
        config: &CausalDecoderConfig,
        x: &Tensor,
        cache: &mut LayerKvCache,
        position_offset: usize,
        runtime: ForwardRuntime<'_>,
    ) -> Result<Tensor> {
        let profile = profile_layer_enabled();
        let total_started = profile.then(Instant::now);
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if position_offset == 0 && !config.attn_output_gate {
            if let (
                Some(metal),
                AttentionBlock::Full(attention),
                Some(post_norm),
                Some(FeedForward::Moe(mlp)),
                Some(q_norm),
                Some(k_norm),
                Some(rope_theta),
            ) = (
                runtime.metal_executor(),
                &self.attention,
                self.post_attention_norm.as_ref(),
                self.mlp.as_ref(),
                match &self.attention {
                    AttentionBlock::Full(attention) => attention.q_norm.as_ref(),
                    AttentionBlock::Linear(_) => None,
                },
                match &self.attention {
                    AttentionBlock::Full(attention) => attention.k_norm.as_ref(),
                    AttentionBlock::Linear(_) => None,
                },
                config.rope_theta,
            ) {
                if let Some((router, experts, top_k)) = mlp.metal_parts() {
                    let (seq, hidden_dim) = x.as_matrix()?;
                    let head_dim = config.head_dim.ok_or_else(|| {
                        InferError::Dimension("head_dim manquant pour prefill Metal".to_string())
                    })?;
                    let spec = crate::metal_backend::PrefillAttentionSpec {
                        seq,
                        hidden_dim,
                        q_heads: config.num_attention_heads,
                        kv_heads: config.num_key_value_heads,
                        head_dim,
                        rope_dims: config.rope_dims.unwrap_or(head_dim),
                        rope_theta,
                        eps: config.rms_eps,
                    };
                    match metal.full_attention_prefill_tail_moe(
                        x,
                        &self.input_norm,
                        &attention.q_proj,
                        &attention.k_proj,
                        &attention.v_proj,
                        &attention.o_proj,
                        q_norm,
                        k_norm,
                        post_norm,
                        router,
                        experts,
                        top_k,
                        spec,
                    ) {
                        Ok((output, key, value)) => {
                            let layout = AttentionLayout {
                                num_attention_heads: config.num_attention_heads,
                                num_key_value_heads: config.num_key_value_heads,
                                head_dim,
                                rope_dims: config.rope_dims.unwrap_or(head_dim),
                            };
                            cache.append_batch(&key, &value, &layout)?;
                            if let Some(total_started) = total_started {
                                eprintln!(
                                    "profile_layer prefill_gpu_total_us={}",
                                    total_started.elapsed().as_micros()
                                );
                            }
                            return Ok(output);
                        }
                        Err(error) => {
                            if trace_prefill_enabled() {
                                eprintln!("prefill gpu fallback: {error}");
                            }
                        }
                    }
                }
            }
        }
        let norm_started = profile.then(Instant::now);
        let normed = rms_norm(x, &self.input_norm, config.rms_eps)?;
        let norm_elapsed = norm_started.map(|started| started.elapsed());
        let attention_started = profile.then(Instant::now);
        let attn_out = match &self.attention {
            AttentionBlock::Full(attention) => {
                let context = full_attention_context_prefill(
                    config,
                    &normed,
                    cache,
                    position_offset,
                    attention,
                    runtime,
                )?;
                attention.o_proj.forward_with_runtime(&context, runtime)?
            }
            AttentionBlock::Linear(linear) => linear.forward_cached_batch_with_runtime(
                config.linear_attention_config()?,
                &normed,
                &mut cache.linear,
                runtime,
            )?,
        };
        let attention_elapsed = attention_started.map(|started| started.elapsed());
        let tail_started = profile.then(Instant::now);
        let attention_state = x.add(&attn_out)?;
        let output = match (&self.post_attention_norm, &self.mlp) {
            (Some(norm), Some(mlp)) => {
                let mlp_input = rms_norm(&attention_state, norm, config.rms_eps)?;
                attention_state.add(&mlp.forward_with_runtime(&mlp_input, runtime)?)
            }
            (None, None) => Ok(attention_state),
            _ => Err(InferError::Config(
                "bloc MLP partiellement initialisé".to_string(),
            )),
        }?;
        print_layer_profile(
            total_started,
            norm_elapsed,
            attention_elapsed,
            tail_started.map(|started| started.elapsed()),
        );
        Ok(output)
    }

    pub(super) fn forward_cached(
        &self,
        config: &CausalDecoderConfig,
        x: &Tensor,
        cache: &mut LayerKvCache,
        position: usize,
        runtime: ForwardRuntime<'_>,
    ) -> Result<Tensor> {
        let profile = profile_layer_enabled();
        let total_started = profile.then(Instant::now);
        let norm_started = profile.then(Instant::now);
        let normed = rms_norm(x, &self.input_norm, config.rms_eps)?;
        let norm_elapsed = norm_started.map(|started| started.elapsed());
        let attention_started = profile.then(Instant::now);
        let attn_out = match &self.attention {
            AttentionBlock::Full(attention) => {
                let context = full_attention_context_cached(
                    config, &normed, cache, position, attention, runtime,
                )?;
                let _attention_elapsed = attention_started.map(|started| started.elapsed());
                #[cfg(all(target_os = "macos", feature = "metal"))]
                if let (Some(metal), Some(norm), Some(FeedForward::Moe(mlp))) = (
                    runtime.metal_executor(),
                    self.post_attention_norm.as_ref(),
                    self.mlp.as_ref(),
                ) {
                    if let Some((router, experts, top_k)) = mlp.metal_parts() {
                        let tail_started = profile.then(Instant::now);
                        if let Ok(output) = metal.full_attention_tail_moe(
                            x,
                            &context,
                            &attention.o_proj,
                            norm,
                            router,
                            experts,
                            top_k,
                            config.rms_eps,
                        ) {
                            print_layer_profile(
                                total_started,
                                norm_elapsed,
                                _attention_elapsed,
                                tail_started.map(|started| started.elapsed()),
                            );
                            return Ok(output);
                        }
                    } else if let Some((router, experts, top_k, shared_expert, shared_gate)) =
                        mlp.shared_metal_parts()
                    {
                        let tail_started = profile.then(Instant::now);
                        if let Ok(output) = metal.full_attention_tail_moe_shared(
                            x,
                            &context,
                            &attention.o_proj,
                            norm,
                            router,
                            experts,
                            top_k,
                            shared_expert,
                            shared_gate,
                            config.rms_eps,
                        ) {
                            print_layer_profile(
                                total_started,
                                norm_elapsed,
                                _attention_elapsed,
                                tail_started.map(|started| started.elapsed()),
                            );
                            return Ok(output);
                        }
                    }
                }
                attention.o_proj.forward_with_runtime(&context, runtime)?
            }
            AttentionBlock::Linear(linear) => linear.forward_cached_with_runtime(
                config.linear_attention_config()?,
                &normed,
                &mut cache.linear,
                runtime,
            )?,
        };
        let attention_elapsed = attention_started.map(|started| started.elapsed());
        let tail_started = profile.then(Instant::now);
        let attention_state = x.add(&attn_out)?;
        let output = match (&self.post_attention_norm, &self.mlp) {
            (Some(norm), Some(mlp)) => {
                let mlp_input = rms_norm(&attention_state, norm, config.rms_eps)?;
                attention_state.add(&mlp.forward_with_runtime(&mlp_input, runtime)?)
            }
            (None, None) => Ok(attention_state),
            _ => Err(InferError::Config(
                "bloc MLP partiellement initialisé".to_string(),
            )),
        }?;
        print_layer_profile(
            total_started,
            norm_elapsed,
            attention_elapsed,
            tail_started.map(|started| started.elapsed()),
        );
        Ok(output)
    }
}
