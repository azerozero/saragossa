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
        let mut attention = if config.is_full_attention_layer(layer_index) {
            full_attention_from_tensors(config, tensors, &prefix, layer_index)?
        } else {
            linear_attention_from_tensors(tensors, &prefix)?
        };
        // Gemma 3 : base RoPE locale + fenêtre des couches locales + échelle
        // linéaire des positions des couches globales (rope_scaling ≥4B).
        if let AttentionBlock::Full(full) = &mut attention {
            full.rope_theta = config.layer_rope_theta_override(layer_index);
            full.rope_position_scale = config.layer_rope_position_scale(layer_index);
            full.sliding_window = config.layer_sliding_window(layer_index);
        }
        let (post_attention_norm, mlp) = optional_mlp_from(config, tensors, &prefix)?;
        let parallel_moe = if config.parallel_moe {
            optional_gemma4_parallel_moe_from(config, tensors, &prefix)?
        } else {
            None
        };
        // Gemma : double norme feed-forward (présence -> câblage Gemma au forward).
        let pre_feedforward_norm = take_optional_dense(
            tensors,
            &format!("{prefix}.pre_feedforward_layernorm.weight"),
        )?;
        let post_feedforward_norm = take_optional_dense(
            tensors,
            &format!("{prefix}.post_feedforward_layernorm.weight"),
        )?;
        let pre_feedforward_norm_2 = take_optional_dense(
            tensors,
            &format!("{prefix}.pre_feedforward_layernorm_2.weight"),
        )?;
        let post_feedforward_norm_1 = take_optional_dense(
            tensors,
            &format!("{prefix}.post_feedforward_layernorm_1.weight"),
        )?;
        let post_feedforward_norm_2 = take_optional_dense(
            tensors,
            &format!("{prefix}.post_feedforward_layernorm_2.weight"),
        )?;
        let layer_scalar = take_optional_dense(tensors, &format!("{prefix}.layer_scalar"))?;
        Ok(Self {
            input_norm,
            attention,
            post_attention_norm,
            mlp,
            parallel_moe,
            pre_feedforward_norm,
            post_feedforward_norm,
            pre_feedforward_norm_2,
            post_feedforward_norm_1,
            post_feedforward_norm_2,
            layer_scalar,
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
        self.combine_attention_and_mlp(config, x, &attn_out, runtime)
    }

    /// Recompose résiduel + norme(s) + MLP autour de la sortie d'attention.
    ///
    /// Deux câblages selon l'architecture : Qwen/Llama/Mistral (post-attention
    /// norm en entrée du MLP, deux résiduels) ou Gemma (norme *après* l'attention
    /// avant résiduel, puis double norme pre/post feed-forward) — sélectionné par
    /// la présence de `pre_feedforward_norm`. Le câblage standard reste identique
    /// (byte-identique pour les modèles non-Gemma).
    fn combine_attention_and_mlp(
        &self,
        config: &CausalDecoderConfig,
        x: &Tensor,
        attn_out: &Tensor,
        runtime: ForwardRuntime<'_>,
    ) -> Result<Tensor> {
        if let Some(pre_ffn) = &self.pre_feedforward_norm {
            let post_attn = self.post_attention_norm.as_ref().ok_or_else(|| {
                InferError::Config("Gemma sans post_attention_layernorm".to_string())
            })?;
            let post_ffn = self.post_feedforward_norm.as_ref().ok_or_else(|| {
                InferError::Config("Gemma sans post_feedforward_layernorm".to_string())
            })?;
            let mlp = self
                .mlp
                .as_ref()
                .ok_or_else(|| InferError::Config("Gemma sans MLP".to_string()))?;
            let normed_attn = rms_norm(attn_out, post_attn, config.rms_eps)?;
            let hidden = x.add(&normed_attn)?;
            let ffn_out = if let Some(parallel_moe) = &self.parallel_moe {
                let post_ffn_1 = self.post_feedforward_norm_1.as_ref().ok_or_else(|| {
                    InferError::Config("Gemma4 sans post_feedforward_layernorm_1".to_string())
                })?;
                let post_ffn_2 = self.post_feedforward_norm_2.as_ref().ok_or_else(|| {
                    InferError::Config("Gemma4 sans post_feedforward_layernorm_2".to_string())
                })?;
                let pre_ffn_2 = self.pre_feedforward_norm_2.as_ref().ok_or_else(|| {
                    InferError::Config("Gemma4 sans pre_feedforward_layernorm_2".to_string())
                })?;
                let dense_input = rms_norm(&hidden, pre_ffn, config.rms_eps)?;
                let dense_out = mlp.forward_with_runtime(&dense_input, runtime)?;
                let dense_out = rms_norm(&dense_out, post_ffn_1, config.rms_eps)?;
                let moe_input = rms_norm(&hidden, pre_ffn_2, config.rms_eps)?;
                let moe_out = parallel_moe.forward_with_runtime(&moe_input, runtime)?;
                let moe_out = rms_norm(&moe_out, post_ffn_2, config.rms_eps)?;
                dense_out.add(&moe_out)?
            } else {
                let ffn_input = rms_norm(&hidden, pre_ffn, config.rms_eps)?;
                mlp.forward_with_runtime(&ffn_input, runtime)?
            };
            let ffn_normed = rms_norm(&ffn_out, post_ffn, config.rms_eps)?;
            let mut output = hidden.add(&ffn_normed)?;
            if let Some(layer_scalar) = &self.layer_scalar {
                let scalar =
                    layer_scalar.data().first().copied().ok_or_else(|| {
                        InferError::Dimension("layer_scalar Gemma4 vide".to_string())
                    })?;
                output = output.map(|value| value * scalar);
            }
            return Ok(output);
        }
        let attention_state = x.add(attn_out)?;
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

    pub(super) fn supports_batched_prefill(&self, linear_enabled: bool) -> bool {
        match self.attention {
            AttentionBlock::Full(_) => true,
            AttentionBlock::Linear(_) => linear_enabled,
        }
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
        // Gemma (double norme FFN + GeGLU + embed scale) sort du périmètre des
        // kernels résidents : forcer le chemin générique pour rester correct.
        if self.pre_feedforward_norm.is_some() {
            return false;
        }
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
                    && attention
                        .v_proj
                        .as_ref()
                        .is_some_and(|v_proj| v_proj.bias().is_none())
                    && attention.o_proj.bias().is_none()
                    && attention.q_norm.is_some()
                    && attention.k_norm.is_some()
            }
            // Linear-attn : chemin résident conv/ssm déjà éprouvé (phases 1a/1b).
            AttentionBlock::Linear(_) => true,
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(super) fn prefill_moe_layer(
        &self,
        config: &CausalDecoderConfig,
    ) -> Option<crate::metal_backend::PrefillMoeLayer<'_>> {
        let post_norm = self.post_attention_norm.as_ref()?;
        let tail = match self.mlp.as_ref()? {
            FeedForward::Dense(mlp) => {
                if !prefill_dense_resident_enabled() {
                    return None;
                }
                let (gate_proj, up_proj, down_proj) = mlp.projections();
                crate::metal_backend::PrefillMoeTail::Dense {
                    gate_proj,
                    up_proj,
                    down_proj,
                }
            }
            FeedForward::Moe(mlp) => {
                if let Some((router, experts, top_k, shared_expert, shared_gate)) =
                    mlp.shared_metal_parts()
                {
                    crate::metal_backend::PrefillMoeTail::Shared {
                        router,
                        experts,
                        top_k,
                        shared_expert,
                        shared_gate,
                    }
                } else {
                    let (router, experts, top_k) = mlp.metal_parts()?;
                    crate::metal_backend::PrefillMoeTail::Routed {
                        router,
                        experts,
                        top_k,
                    }
                }
            }
        };
        let prefill_attention = match &self.attention {
            AttentionBlock::Full(attention) => {
                let q_norm = attention.q_norm.as_ref()?;
                let k_norm = attention.k_norm.as_ref()?;
                let v_proj = attention.v_proj.as_ref()?;
                crate::metal_backend::PrefillAttentionLayer::Full {
                    q_proj: &attention.q_proj,
                    k_proj: &attention.k_proj,
                    v_proj,
                    o_proj: &attention.o_proj,
                    q_norm,
                    k_norm,
                    gated: config.attn_output_gate,
                }
            }
            AttentionBlock::Linear(attention) => {
                let la_config = config.linear_attention_config().ok()?;
                let key_dim = la_config.key_dim().ok()?;
                let value_dim = la_config.value_dim().ok()?;
                let conv_dim = key_dim.checked_mul(2)?.checked_add(value_dim)?;
                crate::metal_backend::PrefillAttentionLayer::Linear {
                    weights: attention.resident_weights(),
                    spec: crate::metal_backend::LinearAttentionStepSpec {
                        num_key_heads: la_config.num_key_heads,
                        num_value_heads: la_config.num_value_heads,
                        key_head_dim: la_config.key_head_dim,
                        value_head_dim: la_config.value_head_dim,
                        conv_kernel_dim: la_config.conv_kernel_dim,
                        rms_eps: la_config.rms_eps,
                    },
                    dims: crate::metal_backend::LinearAttnResidentDims {
                        in_dim: self.input_norm.len(),
                        conv_dim,
                        value_dim,
                        key_dim,
                    },
                }
            }
        };
        Some(crate::metal_backend::PrefillMoeLayer {
            input_norm: &self.input_norm,
            attention: prefill_attention,
            post_norm,
            tail,
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
                // Le kernel prefill Metal applique une base RoPE unique
                // (rotate-half uniquement), des positions brutes et aucun masque
                // fenêtré : suivre l'override par couche et s'exclure si la couche
                // est sliding ou scalée (Gemma), ou si l'appariement n'est pas
                // Halves — fallback CPU correct juste en dessous.
                match &self.attention {
                    AttentionBlock::Full(attention) => {
                        attention.rope_theta.or(config.rope_theta).filter(|_| {
                            attention.sliding_window.is_none()
                                && attention.rope_position_scale.is_none()
                                && config.rope_style == RopeStyle::Halves
                        })
                    }
                    AttentionBlock::Linear(_) => None,
                },
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
                        attention.v_proj.as_ref().ok_or_else(|| {
                            InferError::Config("prefill Metal sans v_proj".to_string())
                        })?,
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
                                attn_scalar: config
                                    .query_pre_attn_scalar
                                    .unwrap_or(head_dim as f32),
                                sliding_window: None,
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
        // Variante GATED + SHARED-EXPERT (Qwen3.5/3.6 : `attn_output_gate`, MoE shared) :
        // GPU-ifie le prefill full-attn (attention causale batchée + MoE shared rows),
        // sinon l'attention retombe sur le CPU O(seq²) (goulot prefill à l'échelle).
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if position_offset == 0 && config.attn_output_gate {
            if let (
                Some(metal),
                AttentionBlock::Full(attention),
                Some(post_norm),
                Some(FeedForward::Moe(mlp)),
            ) = (
                runtime.metal_executor(),
                &self.attention,
                self.post_attention_norm.as_ref(),
                self.mlp.as_ref(),
            ) {
                if let (Some(q_norm), Some(k_norm), Some(v_proj), Some(rope_theta)) = (
                    attention.q_norm.as_ref(),
                    attention.k_norm.as_ref(),
                    attention.v_proj.as_ref(),
                    attention.rope_theta.or(config.rope_theta).filter(|_| {
                        attention.sliding_window.is_none()
                            && attention.rope_position_scale.is_none()
                            && config.rope_style == RopeStyle::Halves
                    }),
                ) {
                    if let Some((router, experts, top_k, shared_expert, shared_gate)) =
                        mlp.shared_metal_parts()
                    {
                        let (seq, hidden_dim) = x.as_matrix()?;
                        let head_dim = config.head_dim.ok_or_else(|| {
                            InferError::Dimension(
                                "head_dim manquant pour prefill Metal".to_string(),
                            )
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
                        match metal.full_attention_prefill_tail_moe_shared_gated(
                            x,
                            &self.input_norm,
                            &attention.q_proj,
                            &attention.k_proj,
                            v_proj,
                            &attention.o_proj,
                            q_norm,
                            k_norm,
                            post_norm,
                            router,
                            experts,
                            shared_expert,
                            shared_gate,
                            top_k,
                            spec,
                        ) {
                            Ok((output, key, value)) => {
                                let layout = AttentionLayout {
                                    num_attention_heads: config.num_attention_heads,
                                    num_key_value_heads: config.num_key_value_heads,
                                    head_dim,
                                    rope_dims: config.rope_dims.unwrap_or(head_dim),
                                    attn_scalar: config
                                        .query_pre_attn_scalar
                                        .unwrap_or(head_dim as f32),
                                    sliding_window: None,
                                };
                                cache.append_batch(&key, &value, &layout)?;
                                return Ok(output);
                            }
                            Err(error) => {
                                if trace_prefill_enabled() {
                                    eprintln!("prefill gpu gated fallback: {error}");
                                }
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
        let output = self.combine_attention_and_mlp(config, x, &attn_out, runtime)?;
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
        let output = self.combine_attention_and_mlp(config, x, &attn_out, runtime)?;
        print_layer_profile(
            total_started,
            norm_elapsed,
            attention_elapsed,
            tail_started.map(|started| started.elapsed()),
        );
        Ok(output)
    }
}
