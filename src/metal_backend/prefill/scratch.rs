//! Validation et allocation du scratch des prefills résidents.

use super::*;

#[expect(
    clippy::too_many_arguments,
    reason = "validation prefill: poids, dimensions et paramètres restent explicites"
)]
impl MetalExecutor {
    pub(super) fn allocate_prefill_tail_moe_scratch(
        &self,
        residual: &Tensor,
        shape: &PrefillTailMoeShape<'_>,
        spec: PrefillAttentionSpec,
        top_k: usize,
    ) -> Result<PrefillTailMoeScratch> {
        let hidden_dim = shape.hidden_dim;
        let q_dim = shape.q_dim;
        let kv_dim = shape.kv_dim;
        let total_topk = checked_len(spec.seq, top_k, "prefill topk total")?;
        let inter_dim = shape.stacked.gate.out_dim;
        Ok(PrefillTailMoeScratch {
            residual: self.upload_f32_buffer(residual.data(), "prefill_residual")?,
            input_norm: self.cached_buffer_from_f32(shape.norm_weight, "prefill_input_norm")?,
            q_norm: self.cached_buffer_from_f32(shape.q_norm_weight, "prefill_q_norm")?,
            k_norm: self.cached_buffer_from_f32(shape.k_norm_weight, "prefill_k_norm")?,
            post_norm: self.cached_buffer_from_f32(shape.post_norm_weight, "prefill_post_norm")?,
            normed: self.private_f32_buffer(
                checked_len(spec.seq, hidden_dim, "prefill normed")?,
                "prefill_normed",
            )?,
            q: self.private_f32_buffer(checked_len(spec.seq, q_dim, "prefill q")?, "prefill_q")?,
            k: self.new_f32_buffer(checked_len(spec.seq, kv_dim, "prefill k")?, "prefill_k")?,
            v: self.new_f32_buffer(checked_len(spec.seq, kv_dim, "prefill v")?, "prefill_v")?,
            q_rope: self.private_f32_buffer(
                checked_len(spec.seq, q_dim, "prefill q rope")?,
                "prefill_q_rope",
            )?,
            k_rope: self.new_f32_buffer(
                checked_len(spec.seq, kv_dim, "prefill k rope")?,
                "prefill_k_rope",
            )?,
            context: self.private_f32_buffer(
                checked_len(spec.seq, q_dim, "prefill context")?,
                "prefill_context",
            )?,
            o: self
                .private_f32_buffer(checked_len(spec.seq, hidden_dim, "prefill o")?, "prefill_o")?,
            attention_state: self.private_f32_buffer(
                checked_len(spec.seq, hidden_dim, "prefill attention state")?,
                "prefill_attention_state",
            )?,
            post_normed: self.private_f32_buffer(
                checked_len(spec.seq, hidden_dim, "prefill post normed")?,
                "prefill_post_normed",
            )?,
            router: self.new_f32_buffer(
                checked_len(spec.seq, shape.expert_count, "prefill router")?,
                "prefill_router",
            )?,
            output: self.new_f32_buffer(
                checked_len(spec.seq, hidden_dim, "prefill output")?,
                "prefill_output",
            )?,
            indices: self.private_u32_buffer(total_topk, "prefill_indices")?,
            scores: self.private_f32_buffer(total_topk, "prefill_scores")?,
            gate: self.private_f32_buffer(
                checked_len(total_topk, inter_dim, "prefill gate")?,
                "prefill_gate",
            )?,
            up: self.private_f32_buffer(
                checked_len(total_topk, inter_dim, "prefill up")?,
                "prefill_up",
            )?,
            hidden: self.private_f32_buffer(
                checked_len(total_topk, inter_dim, "prefill hidden")?,
                "prefill_hidden",
            )?,
            down: self.private_f32_buffer(
                checked_len(total_topk, hidden_dim, "prefill down")?,
                "prefill_down",
            )?,
        })
    }

    pub(super) fn allocate_prefill_resident_layer_scratch(
        &self,
        shape: &PrefillResidentLayerShape<'_>,
        spec: PrefillAttentionSpec,
        hidden_dim: usize,
        q_dim: usize,
        kv_dim: usize,
        top_k: usize,
    ) -> Result<PrefillResidentLayerScratch> {
        let hidden_len = checked_len(spec.seq, hidden_dim, "prefill résident hidden")?;
        let attention = match &shape.attention {
            PrefillResidentAttentionShape::Full {
                q_norm_weight,
                k_norm_weight,
                gated,
            } => {
                let q_len = checked_len(spec.seq, q_dim, "prefill résident q")?;
                let kv_len = checked_len(spec.seq, kv_dim, "prefill résident kv")?;
                let q2_len = checked_len(q_len, 2, "prefill résident q2")?;
                PrefillResidentAttentionScratch::Full(PrefillResidentFullAttentionScratch {
                    q_norm: self.cached_buffer_from_f32(q_norm_weight, "resident_q_norm")?,
                    k_norm: self.cached_buffer_from_f32(k_norm_weight, "resident_k_norm")?,
                    q2: if *gated {
                        Some(self.private_f32_buffer(q2_len, "resident_q2")?)
                    } else {
                        None
                    },
                    gate: if *gated {
                        Some(self.private_f32_buffer(q_len, "resident_attn_gate")?)
                    } else {
                        None
                    },
                    q: self.private_f32_buffer(q_len, "resident_q")?,
                    k: self.uncached_f32_buffer(kv_len, "resident_k")?,
                    v: self.uncached_f32_buffer(kv_len, "resident_v")?,
                    q_rope: self.private_f32_buffer(q_len, "resident_q_rope")?,
                    k_rope: self.uncached_f32_buffer(kv_len, "resident_k_rope")?,
                    context: self.private_f32_buffer(q_len, "resident_context")?,
                    gated_context: if *gated {
                        Some(self.private_f32_buffer(q_len, "resident_gated_context")?)
                    } else {
                        None
                    },
                    o: self.private_f32_buffer(hidden_len, "resident_o")?,
                })
            }
            PrefillResidentAttentionShape::Linear {
                spec: linear_spec,
                dims,
                conv_len,
                ssm_len,
                ..
            } => {
                let conv_seed = vec![0.0; *conv_len];
                let ssm_seed = vec![0.0; *ssm_len];
                let mut state = None;
                self.ensure_linear_attention_metal_state(
                    &mut state,
                    &conv_seed,
                    &ssm_seed,
                    dims.conv_dim,
                    *conv_len,
                    *ssm_len,
                    *linear_spec,
                )?;
                let state = state.ok_or_else(|| {
                    InferError::Metal("état prefill linear-attn absent".to_string())
                })?;
                PrefillResidentAttentionScratch::Linear(PrefillResidentLinearAttentionScratch {
                    output: self.private_f32_buffer(hidden_len, "resident_linear_o")?,
                    state,
                })
            }
        };
        let tail = match &shape.tail {
            PrefillResidentTailShape::Dense { inter_dim, .. } => {
                let inter_len = checked_len(spec.seq, *inter_dim, "resident dense inter")?;
                PrefillResidentTailScratch::Dense {
                    gate: self.private_f32_buffer(inter_len, "resident_dense_gate")?,
                    up: self.private_f32_buffer(inter_len, "resident_dense_up")?,
                    hidden: self.private_f32_buffer(inter_len, "resident_dense_hidden")?,
                    down: self.private_f32_buffer(hidden_len, "resident_dense_down")?,
                }
            }
            PrefillResidentTailShape::GemmaDense { inter_dim, .. } => {
                self.allocate_prefill_gemma_dense_tail_scratch(spec.seq, hidden_dim, *inter_dim)?
            }
            PrefillResidentTailShape::GemmaParallel {
                dense_inter_dim, ..
            } => self.allocate_prefill_gemma_parallel_tail_scratch(
                spec.seq,
                hidden_dim,
                *dense_inter_dim,
            )?,
            PrefillResidentTailShape::Routed {
                expert_count,
                stacked,
            } => {
                let total_topk = checked_len(spec.seq, top_k, "resident topk total")?;
                let inter_dim = stacked.gate.out_dim;
                PrefillResidentTailScratch::Routed {
                    router: self.private_f32_buffer(
                        checked_len(spec.seq, *expert_count, "resident router")?,
                        "resident_router",
                    )?,
                    indices: self.private_u32_buffer(total_topk, "resident_indices")?,
                    scores: self.private_f32_buffer(total_topk, "resident_scores")?,
                    gate: self.private_f32_buffer(
                        checked_len(total_topk, inter_dim, "resident gate")?,
                        "resident_gate",
                    )?,
                    up: self.private_f32_buffer(
                        checked_len(total_topk, inter_dim, "resident up")?,
                        "resident_up",
                    )?,
                    hidden: self.private_f32_buffer(
                        checked_len(total_topk, inter_dim, "resident hidden")?,
                        "resident_hidden",
                    )?,
                    down: self.private_f32_buffer(
                        checked_len(total_topk, hidden_dim, "resident down")?,
                        "resident_down",
                    )?,
                }
            }
            PrefillResidentTailShape::Shared { .. } => PrefillResidentTailScratch::Shared,
        };
        Ok(PrefillResidentLayerScratch {
            input_norm: self.cached_buffer_from_f32(shape.norm_weight, "resident_input_norm")?,
            post_norm: self.cached_buffer_from_f32(shape.post_norm_weight, "resident_post_norm")?,
            normed: self.private_f32_buffer(hidden_len, "resident_normed")?,
            attention,
            attention_state: self.private_f32_buffer(hidden_len, "resident_attention_state")?,
            post_normed: self.private_f32_buffer(hidden_len, "resident_post_normed")?,
            tail,
        })
    }

    pub(super) fn allocate_prefill_gemma_dense_tail_scratch(
        &self,
        rows: usize,
        hidden_dim: usize,
        inter_dim: usize,
    ) -> Result<PrefillResidentTailScratch> {
        let hidden_len = checked_len(rows, hidden_dim, "resident Gemma dense hidden")?;
        let inter_len = checked_len(rows, inter_dim, "resident Gemma dense inter")?;
        Ok(PrefillResidentTailScratch::GemmaDense {
            ffn_input: self.private_f32_buffer(hidden_len, "resident_gemma_ffn_input")?,
            gate: self.private_f32_buffer(inter_len, "resident_gemma_gate")?,
            up: self.private_f32_buffer(inter_len, "resident_gemma_up")?,
            geglu: self.private_f32_buffer(inter_len, "resident_gemma_geglu")?,
            down: self.private_f32_buffer(hidden_len, "resident_gemma_down")?,
            ffn_normed: self.private_f32_buffer(hidden_len, "resident_gemma_ffn_normed")?,
        })
    }

    pub(super) fn allocate_prefill_gemma_parallel_tail_scratch(
        &self,
        rows: usize,
        hidden_dim: usize,
        dense_inter_dim: usize,
    ) -> Result<PrefillResidentTailScratch> {
        let hidden_len = checked_len(rows, hidden_dim, "resident Gemma parallèle hidden")?;
        let dense_inter_len = checked_len(
            rows,
            dense_inter_dim,
            "resident Gemma parallèle dense inter",
        )?;
        Ok(PrefillResidentTailScratch::GemmaParallel {
            dense_input: self
                .private_f32_buffer(hidden_len, "resident_gemma_parallel_dense_input")?,
            dense_gate: self
                .private_f32_buffer(dense_inter_len, "resident_gemma_parallel_dense_gate")?,
            dense_up: self
                .private_f32_buffer(dense_inter_len, "resident_gemma_parallel_dense_up")?,
            dense_geglu: self
                .private_f32_buffer(dense_inter_len, "resident_gemma_parallel_dense_geglu")?,
            dense_down: self
                .private_f32_buffer(hidden_len, "resident_gemma_parallel_dense_down")?,
            dense_out: self.private_f32_buffer(hidden_len, "resident_gemma_parallel_dense_out")?,
            moe_input: self.private_f32_buffer(hidden_len, "resident_gemma_parallel_moe_input")?,
            moe_out: self.private_f32_buffer(hidden_len, "resident_gemma_parallel_moe_out")?,
            ffn_out: self.private_f32_buffer(hidden_len, "resident_gemma_parallel_ffn_out")?,
            ffn_normed: self
                .private_f32_buffer(hidden_len, "resident_gemma_parallel_ffn_normed")?,
        })
    }

    pub(super) fn check_prefill_tail_moe_shapes<'a>(
        &self,
        residual: &'a Tensor,
        input_norm: &'a Tensor,
        q_proj: &Linear,
        k_proj: &Linear,
        v_proj: &Linear,
        o_proj: &Linear,
        q_norm: &'a Tensor,
        k_norm: &'a Tensor,
        post_norm: &'a Tensor,
        router: &Linear,
        experts: &[GatedMlp],
        top_k: usize,
        spec: PrefillAttentionSpec,
    ) -> Result<PrefillTailMoeShape<'a>> {
        ensure_biasless(q_proj, "q_proj")?;
        ensure_biasless(k_proj, "k_proj")?;
        ensure_biasless(v_proj, "v_proj")?;
        ensure_biasless(o_proj, "o_proj")?;
        ensure_biasless(router, "router")?;
        if spec.seq == 0 || spec.seq > 256 {
            return Err(InferError::Dimension(format!(
                "prefill Metal seq={} non supporté",
                spec.seq
            )));
        }
        let (batch, hidden_dim) = residual.as_matrix()?;
        if batch != spec.seq || hidden_dim != spec.hidden_dim {
            return Err(InferError::Dimension(format!(
                "prefill residual=[{batch},{hidden_dim}], spec seq={} hidden={}",
                spec.seq, spec.hidden_dim
            )));
        }
        let q_dim = spec.q_heads * spec.head_dim;
        let kv_dim = spec.kv_heads * spec.head_dim;
        if linear_out_dim(q_proj.weight())? != q_dim
            || linear_out_dim(k_proj.weight())? != kv_dim
            || linear_out_dim(v_proj.weight())? != kv_dim
            || linear_out_dim(o_proj.weight())? != hidden_dim
        {
            return Err(InferError::Dimension(
                "prefill projections attention incompatibles".to_string(),
            ));
        }
        let norm_weight = dense_vector(input_norm, hidden_dim, "input_norm")?;
        let q_norm_weight = dense_vector(q_norm, spec.head_dim, "q_norm")?;
        let k_norm_weight = dense_vector(k_norm, spec.head_dim, "k_norm")?;
        let post_norm_weight = dense_vector(post_norm, hidden_dim, "post_norm")?;
        let expert_count = linear_out_dim(router.weight())?;
        ensure_valid_top_k(top_k, expert_count)?;
        if expert_count != experts.len() {
            return Err(InferError::Dimension(format!(
                "prefill router experts={expert_count}, poids experts={}",
                experts.len()
            )));
        }
        let stacked = self.stacked_moe_buffers(experts)?;
        if hidden_dim != stacked.gate.in_dim || hidden_dim != stacked.up.in_dim {
            return Err(InferError::Dimension(format!(
                "prefill hidden={hidden_dim}, gate_in={}, up_in={}",
                stacked.gate.in_dim, stacked.up.in_dim
            )));
        }
        if stacked.gate.out_dim != stacked.up.out_dim || stacked.down.in_dim != stacked.gate.out_dim
        {
            return Err(InferError::Dimension(format!(
                "prefill inter dims gate={} up={} down_in={}",
                stacked.gate.out_dim, stacked.up.out_dim, stacked.down.in_dim
            )));
        }
        Ok(PrefillTailMoeShape {
            hidden_dim,
            q_dim,
            kv_dim,
            norm_weight,
            q_norm_weight,
            k_norm_weight,
            post_norm_weight,
            expert_count,
            stacked,
        })
    }

    pub(super) fn check_prefill_resident_layer_shapes<'a>(
        &self,
        layer: PrefillMoeLayer<'a>,
        layer_index: usize,
        hidden_dim: usize,
        q_dim: usize,
        kv_dim: usize,
        spec: PrefillAttentionSpec,
    ) -> Result<PrefillResidentLayerShape<'a>> {
        let norm_weight = dense_vector(layer.input_norm, hidden_dim, "resident input_norm")?;
        let post_norm_weight = dense_vector(layer.post_norm, hidden_dim, "resident post_norm")?;
        let attention = match layer.attention {
            PrefillAttentionLayer::Full {
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                q_norm,
                k_norm,
                gated,
            } => {
                if spec.q_heads == 0
                    || spec.kv_heads == 0
                    || spec.head_dim == 0
                    || spec.q_heads % spec.kv_heads != 0
                    || spec.rope_dims == 0
                    || spec.rope_dims > spec.head_dim
                    || spec.rope_dims % 2 != 0
                    || spec.rope_frequency_dim < spec.rope_dims
                    || spec.rope_frequency_dim % 2 != 0
                {
                    return Err(InferError::Dimension(format!(
                        "prefill résident spec attention invalide couche {layer_index}: {spec:?}"
                    )));
                }
                ensure_biasless(q_proj, "q_proj")?;
                ensure_biasless(k_proj, "k_proj")?;
                ensure_biasless(v_proj, "v_proj")?;
                ensure_biasless(o_proj, "o_proj")?;
                let expected_q_proj_dim = if gated {
                    checked_len(q_dim, 2, "prefill résident q_proj gated")?
                } else {
                    q_dim
                };
                if linear_out_dim(q_proj.weight())? != expected_q_proj_dim
                    || linear_out_dim(k_proj.weight())? != kv_dim
                    || linear_out_dim(v_proj.weight())? != kv_dim
                    || linear_out_dim(o_proj.weight())? != hidden_dim
                {
                    return Err(InferError::Dimension(format!(
                        "prefill résident projections incompatibles couche {layer_index}"
                    )));
                }
                PrefillResidentAttentionShape::Full {
                    q_norm_weight: dense_vector(q_norm, spec.head_dim, "resident q_norm")?,
                    k_norm_weight: dense_vector(k_norm, spec.head_dim, "resident k_norm")?,
                    gated,
                }
            }
            PrefillAttentionLayer::Linear {
                weights,
                spec: linear_spec,
                dims,
            } => {
                ensure_biasless(weights.in_proj_qkv, "linear_attn.in_proj_qkv")?;
                ensure_biasless(weights.in_proj_z, "linear_attn.in_proj_z")?;
                ensure_biasless(weights.in_proj_b, "linear_attn.in_proj_b")?;
                ensure_biasless(weights.in_proj_a, "linear_attn.in_proj_a")?;
                ensure_biasless(weights.out_proj, "linear_attn.out_proj")?;
                if dims.in_dim != hidden_dim {
                    return Err(InferError::Dimension(format!(
                        "prefill résident linear in_dim={} hidden={} couche {layer_index}",
                        dims.in_dim, hidden_dim
                    )));
                }
                if linear_spec.num_key_heads == 0
                    || linear_spec.num_value_heads == 0
                    || linear_spec.key_head_dim == 0
                    || linear_spec.value_head_dim == 0
                    || linear_spec.conv_kernel_dim < 2
                    || linear_spec.num_value_heads % linear_spec.num_key_heads != 0
                {
                    return Err(InferError::Dimension(format!(
                        "prefill résident linear dims invalides couche {layer_index}: key_heads={}, value_heads={}, key_dim={}, value_dim={}, kernel={}",
                        linear_spec.num_key_heads,
                        linear_spec.num_value_heads,
                        linear_spec.key_head_dim,
                        linear_spec.value_head_dim,
                        linear_spec.conv_kernel_dim
                    )));
                }
                let expected_key_dim = checked_len(
                    linear_spec.num_key_heads,
                    linear_spec.key_head_dim,
                    "resident linear key_dim",
                )?;
                let expected_value_dim = checked_len(
                    linear_spec.num_value_heads,
                    linear_spec.value_head_dim,
                    "resident linear value_dim",
                )?;
                let expected_conv_dim = expected_key_dim
                    .checked_mul(2)
                    .and_then(|twice| twice.checked_add(expected_value_dim))
                    .ok_or_else(|| {
                        InferError::Shape("resident linear conv_dim trop grand".to_string())
                    })?;
                if dims.key_dim != expected_key_dim
                    || dims.value_dim != expected_value_dim
                    || dims.conv_dim != expected_conv_dim
                {
                    return Err(InferError::Dimension(format!(
                        "prefill résident linear dims incohérentes couche {layer_index}: key_dim={}/{}, value_dim={}/{}, conv_dim={}/{}",
                        dims.key_dim,
                        expected_key_dim,
                        dims.value_dim,
                        expected_value_dim,
                        dims.conv_dim,
                        expected_conv_dim
                    )));
                }
                expect_linear_shape(
                    weights.in_proj_qkv.weight(),
                    dims.conv_dim,
                    hidden_dim,
                    "linear_attn.in_proj_qkv",
                )?;
                expect_linear_shape(
                    weights.in_proj_z.weight(),
                    dims.value_dim,
                    hidden_dim,
                    "linear_attn.in_proj_z",
                )?;
                expect_linear_shape(
                    weights.in_proj_b.weight(),
                    linear_spec.num_value_heads,
                    hidden_dim,
                    "linear_attn.in_proj_b",
                )?;
                expect_linear_shape(
                    weights.in_proj_a.weight(),
                    linear_spec.num_value_heads,
                    hidden_dim,
                    "linear_attn.in_proj_a",
                )?;
                expect_linear_in(
                    weights.out_proj.weight(),
                    dims.value_dim,
                    "linear_attn.out_proj",
                )?;
                if linear_out_dim(weights.out_proj.weight())? != hidden_dim {
                    return Err(InferError::Dimension(format!(
                        "prefill résident linear out_dim incompatible couche {layer_index}"
                    )));
                }
                match weights.conv_weight.shape() {
                    [channels, kernel, one]
                        if *channels == dims.conv_dim
                            && *kernel == linear_spec.conv_kernel_dim
                            && *one == 1 => {}
                    [channels, one, kernel]
                        if *channels == dims.conv_dim
                            && *one == 1
                            && *kernel == linear_spec.conv_kernel_dim => {}
                    shape => {
                        return Err(InferError::Dimension(format!(
                            "linear_attn.conv1d.weight résident couche {layer_index} attendu [{}, {}, 1] ou [{}, 1, {}], reçu {shape:?}",
                            dims.conv_dim,
                            linear_spec.conv_kernel_dim,
                            dims.conv_dim,
                            linear_spec.conv_kernel_dim
                        )));
                    }
                }
                if weights.a_log.len() != linear_spec.num_value_heads
                    || weights.dt_bias.len() != linear_spec.num_value_heads
                    || weights.norm_weight.len() != linear_spec.value_head_dim
                {
                    return Err(InferError::Dimension(format!(
                        "prefill résident linear paramètres invalides couche {layer_index}: A_log={}, dt_bias={}, norm={}, attendu heads={} norm={}",
                        weights.a_log.len(),
                        weights.dt_bias.len(),
                        weights.norm_weight.len(),
                        linear_spec.num_value_heads,
                        linear_spec.value_head_dim
                    )));
                }
                let keep = linear_spec.conv_kernel_dim - 1;
                let conv_len = checked_len(keep, dims.conv_dim, "resident linear conv state")?;
                let ssm_len = checked_len(
                    dims.value_dim,
                    linear_spec.key_head_dim,
                    "resident linear ssm",
                )?;
                PrefillResidentAttentionShape::Linear {
                    spec: linear_spec,
                    dims,
                    conv_len,
                    ssm_len,
                }
            }
        };
        let tail = match layer.tail {
            PrefillMoeTail::Dense {
                gate_proj,
                up_proj,
                down_proj,
            } => {
                ensure_biasless(gate_proj, "dense.gate_proj")?;
                ensure_biasless(up_proj, "dense.up_proj")?;
                ensure_biasless(down_proj, "dense.down_proj")?;
                let inter_dim = linear_out_dim(gate_proj.weight())?;
                expect_linear_shape(gate_proj.weight(), inter_dim, hidden_dim, "dense.gate_proj")?;
                expect_linear_shape(up_proj.weight(), inter_dim, hidden_dim, "dense.up_proj")?;
                expect_linear_shape(down_proj.weight(), hidden_dim, inter_dim, "dense.down_proj")?;
                PrefillResidentTailShape::Dense {
                    gate_proj: self
                        .resolve_linear_weight_buffers(gate_proj.weight(), "resident_dense_gate")?,
                    up_proj: self
                        .resolve_linear_weight_buffers(up_proj.weight(), "resident_dense_up")?,
                    down_proj: self
                        .resolve_linear_weight_buffers(down_proj.weight(), "resident_dense_down")?,
                    inter_dim,
                }
            }
            PrefillMoeTail::GemmaDense {
                gate_proj,
                up_proj,
                down_proj,
                pre_feedforward_norm,
                post_feedforward_norm,
                layer_scalar,
                inter_dim,
            } => self.check_prefill_gemma_dense_tail_shape(
                hidden_dim,
                inter_dim,
                gate_proj,
                up_proj,
                down_proj,
                pre_feedforward_norm,
                post_feedforward_norm,
                layer_scalar,
            )?,
            PrefillMoeTail::GemmaParallel {
                dense_gate_proj,
                dense_up_proj,
                dense_down_proj,
                pre_feedforward_norm,
                post_feedforward_norm_1,
                router,
                experts,
                top_k,
                router_norm,
                per_expert_scale,
                pre_feedforward_norm_2,
                post_feedforward_norm_2,
                post_feedforward_norm,
                layer_scalar,
                dense_inter_dim,
            } => self.check_prefill_gemma_parallel_tail_shape(
                hidden_dim,
                dense_inter_dim,
                dense_gate_proj,
                dense_up_proj,
                dense_down_proj,
                pre_feedforward_norm,
                post_feedforward_norm_1,
                router,
                experts,
                top_k,
                router_norm,
                per_expert_scale,
                pre_feedforward_norm_2,
                post_feedforward_norm_2,
                post_feedforward_norm,
                layer_scalar,
            )?,
            PrefillMoeTail::Routed {
                router,
                experts,
                top_k,
            } => {
                ensure_biasless(router, "router")?;
                let expert_count = linear_out_dim(router.weight())?;
                ensure_valid_top_k(top_k, expert_count)?;
                if expert_count != experts.len() {
                    return Err(InferError::Dimension(format!(
                        "prefill résident router experts={expert_count}, poids experts={} couche {layer_index}",
                        experts.len()
                    )));
                }
                let stacked = self.stacked_moe_buffers(experts)?;
                if hidden_dim != stacked.gate.in_dim || hidden_dim != stacked.up.in_dim {
                    return Err(InferError::Dimension(format!(
                        "prefill résident hidden={hidden_dim}, gate_in={}, up_in={} couche {layer_index}",
                        stacked.gate.in_dim, stacked.up.in_dim
                    )));
                }
                if stacked.gate.out_dim != stacked.up.out_dim
                    || stacked.down.in_dim != stacked.gate.out_dim
                {
                    return Err(InferError::Dimension(format!(
                        "prefill résident inter dims gate={} up={} down_in={} couche {layer_index}",
                        stacked.gate.out_dim, stacked.up.out_dim, stacked.down.in_dim
                    )));
                }
                PrefillResidentTailShape::Routed {
                    expert_count,
                    stacked,
                }
            }
            PrefillMoeTail::Shared {
                router,
                experts,
                top_k,
                shared_expert,
                shared_gate,
            } => {
                let weights =
                    self.resolve_moe_shared_weights(router, experts, shared_expert, shared_gate)?;
                let shape = self.check_moe_shared_buffer_shapes(hidden_dim, &weights, top_k)?;
                if shape.out_dim != hidden_dim {
                    return Err(InferError::Dimension(format!(
                        "prefill résident MoE shared out_dim={} hidden={} couche {layer_index}",
                        shape.out_dim, hidden_dim
                    )));
                }
                PrefillResidentTailShape::Shared { weights }
            }
        };
        Ok(PrefillResidentLayerShape {
            norm_weight,
            post_norm_weight,
            attention,
            tail,
        })
    }

    pub(super) fn check_prefill_gemma_dense_tail_shape(
        &self,
        hidden_dim: usize,
        inter_dim: usize,
        gate_proj: &Linear,
        up_proj: &Linear,
        down_proj: &Linear,
        pre_feedforward_norm: &Tensor,
        post_feedforward_norm: &Tensor,
        layer_scalar: Option<&Tensor>,
    ) -> Result<PrefillResidentTailShape> {
        ensure_biasless(gate_proj, "gemma_dense.gate_proj")?;
        ensure_biasless(up_proj, "gemma_dense.up_proj")?;
        ensure_biasless(down_proj, "gemma_dense.down_proj")?;
        expect_linear_shape(
            gate_proj.weight(),
            inter_dim,
            hidden_dim,
            "gemma_dense.gate_proj",
        )?;
        expect_linear_shape(
            up_proj.weight(),
            inter_dim,
            hidden_dim,
            "gemma_dense.up_proj",
        )?;
        expect_linear_shape(
            down_proj.weight(),
            hidden_dim,
            inter_dim,
            "gemma_dense.down_proj",
        )?;
        let pre_feedforward_norm = dense_vector(
            pre_feedforward_norm,
            hidden_dim,
            "gemma_dense.pre_feedforward_norm",
        )?;
        let post_feedforward_norm = dense_vector(
            post_feedforward_norm,
            hidden_dim,
            "gemma_dense.post_feedforward_norm",
        )?;
        let layer_scalar = layer_scalar
            .map(|scalar| {
                dense_vector(scalar, 1, "gemma_dense.layer_scalar").and_then(|values| {
                    values.first().copied().ok_or_else(|| {
                        InferError::Dimension("layer_scalar Gemma4 vide".to_string())
                    })
                })
            })
            .transpose()?;
        Ok(PrefillResidentTailShape::GemmaDense {
            gate_proj: self
                .resolve_linear_weight_buffers(gate_proj.weight(), "resident_gemma_dense_gate")?,
            up_proj: self
                .resolve_linear_weight_buffers(up_proj.weight(), "resident_gemma_dense_up")?,
            down_proj: self
                .resolve_linear_weight_buffers(down_proj.weight(), "resident_gemma_dense_down")?,
            pre_feedforward_norm: self.cached_buffer_from_f32(
                pre_feedforward_norm,
                "resident_gemma_pre_feedforward_norm",
            )?,
            post_feedforward_norm: self.cached_buffer_from_f32(
                post_feedforward_norm,
                "resident_gemma_post_feedforward_norm",
            )?,
            layer_scalar,
            inter_dim,
        })
    }

    pub(super) fn check_prefill_gemma_parallel_tail_shape(
        &self,
        hidden_dim: usize,
        dense_inter_dim: usize,
        dense_gate_proj: &Linear,
        dense_up_proj: &Linear,
        dense_down_proj: &Linear,
        pre_feedforward_norm: &Tensor,
        post_feedforward_norm_1: &Tensor,
        router: &Linear,
        experts: &[GatedMlp],
        top_k: usize,
        router_norm: Option<(&Tensor, f32)>,
        per_expert_scale: Option<&Tensor>,
        pre_feedforward_norm_2: &Tensor,
        post_feedforward_norm_2: &Tensor,
        post_feedforward_norm: &Tensor,
        layer_scalar: Option<&Tensor>,
    ) -> Result<PrefillResidentTailShape> {
        let dense = self.check_prefill_gemma_dense_tail_shape(
            hidden_dim,
            dense_inter_dim,
            dense_gate_proj,
            dense_up_proj,
            dense_down_proj,
            pre_feedforward_norm,
            post_feedforward_norm_1,
            None,
        )?;
        let PrefillResidentTailShape::GemmaDense {
            gate_proj: dense_gate_proj,
            up_proj: dense_up_proj,
            down_proj: dense_down_proj,
            pre_feedforward_norm,
            post_feedforward_norm: post_feedforward_norm_1,
            ..
        } = dense
        else {
            return Err(InferError::Config(
                "validation Gemma parallèle dense incohérente".to_string(),
            ));
        };

        let moe = self.resolve_moe_routed_weights(router, experts)?;
        let moe_shape = self.check_moe_routed_buffer_shapes(hidden_dim, &moe, top_k)?;
        if moe_shape.out_dim != hidden_dim {
            return Err(InferError::Dimension(format!(
                "prefill Gemma parallèle MoE sort {}, attendu {hidden_dim}",
                moe_shape.out_dim
            )));
        }
        let router_norm = router_norm
            .map(|(weight, eps)| {
                if !eps.is_finite() || eps <= 0.0 {
                    return Err(InferError::Dimension(
                        "prefill Gemma parallèle router eps invalide".to_string(),
                    ));
                }
                let weight = dense_vector(weight, hidden_dim, "gemma_parallel.router_norm")?;
                Ok((
                    self.cached_buffer_from_f32(weight, "resident_gemma_parallel_router_norm")?,
                    eps,
                ))
            })
            .transpose()?;
        let per_expert_scale = per_expert_scale
            .map(|scale| {
                let scale = dense_vector(
                    scale,
                    moe_shape.expert_count,
                    "gemma_parallel.per_expert_scale",
                )?;
                self.cached_buffer_from_f32(scale, "resident_gemma_parallel_per_expert_scale")
            })
            .transpose()?;
        let pre_feedforward_norm_2 = dense_vector(
            pre_feedforward_norm_2,
            hidden_dim,
            "gemma_parallel.pre_feedforward_norm_2",
        )?;
        let post_feedforward_norm_2 = dense_vector(
            post_feedforward_norm_2,
            hidden_dim,
            "gemma_parallel.post_feedforward_norm_2",
        )?;
        let post_feedforward_norm = dense_vector(
            post_feedforward_norm,
            hidden_dim,
            "gemma_parallel.post_feedforward_norm",
        )?;
        let layer_scalar = layer_scalar
            .map(|scalar| {
                dense_vector(scalar, 1, "gemma_parallel.layer_scalar").and_then(|values| {
                    values.first().copied().ok_or_else(|| {
                        InferError::Dimension("layer_scalar Gemma4 vide".to_string())
                    })
                })
            })
            .transpose()?;

        Ok(PrefillResidentTailShape::GemmaParallel {
            dense_gate_proj,
            dense_up_proj,
            dense_down_proj,
            pre_feedforward_norm,
            post_feedforward_norm_1,
            moe,
            router_norm,
            per_expert_scale,
            pre_feedforward_norm_2: self.cached_buffer_from_f32(
                pre_feedforward_norm_2,
                "resident_gemma_parallel_pre_feedforward_norm_2",
            )?,
            post_feedforward_norm_2: self.cached_buffer_from_f32(
                post_feedforward_norm_2,
                "resident_gemma_parallel_post_feedforward_norm_2",
            )?,
            post_feedforward_norm: self.cached_buffer_from_f32(
                post_feedforward_norm,
                "resident_gemma_parallel_post_feedforward_norm",
            )?,
            layer_scalar,
            dense_inter_dim,
        })
    }
}
