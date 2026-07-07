//! Préfill full-attention et MoE résident.

use crate::runtime_flags::trace_prefill_enabled;

use super::kernel_timing::{time_prefill_pass, PrefillKernelTiming};
use super::*;

struct PrefillTailMoeShape<'a> {
    hidden_dim: usize,
    q_dim: usize,
    kv_dim: usize,
    norm_weight: &'a [f32],
    q_norm_weight: &'a [f32],
    k_norm_weight: &'a [f32],
    post_norm_weight: &'a [f32],
    expert_count: usize,
    stacked: StackedMoeBuffers,
}

struct PrefillResidentLayerShape<'a> {
    norm_weight: &'a [f32],
    post_norm_weight: &'a [f32],
    attention: PrefillResidentAttentionShape<'a>,
    tail: PrefillResidentTailShape,
}

enum PrefillResidentAttentionShape<'a> {
    Full {
        q_norm_weight: &'a [f32],
        k_norm_weight: &'a [f32],
        gated: bool,
    },
    Linear {
        spec: LinearAttentionStepSpec,
        dims: LinearAttnResidentDims,
        conv_len: usize,
        ssm_len: usize,
    },
}

struct PrefillTailMoeScratch {
    residual: Buffer,
    input_norm: Buffer,
    q_norm: Buffer,
    k_norm: Buffer,
    post_norm: Buffer,
    normed: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    q_rope: Buffer,
    k_rope: Buffer,
    context: Buffer,
    o: Buffer,
    attention_state: Buffer,
    post_normed: Buffer,
    router: Buffer,
    output: Buffer,
    indices: Buffer,
    scores: Buffer,
    gate: Buffer,
    up: Buffer,
    hidden: Buffer,
    down: Buffer,
}

struct PrefillResidentLayerScratch {
    input_norm: Buffer,
    post_norm: Buffer,
    normed: Buffer,
    attention: PrefillResidentAttentionScratch,
    attention_state: Buffer,
    post_normed: Buffer,
    tail: PrefillResidentTailScratch,
}

struct PrefillResidentFullAttentionScratch {
    q_norm: Buffer,
    k_norm: Buffer,
    q2: Option<Buffer>,
    gate: Option<Buffer>,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    q_rope: Buffer,
    k_rope: Buffer,
    context: Buffer,
    gated_context: Option<Buffer>,
    o: Buffer,
}

struct PrefillResidentLinearAttentionScratch {
    output: Buffer,
    state: LinearAttentionMetalState,
}

enum PrefillResidentAttentionScratch {
    Full(PrefillResidentFullAttentionScratch),
    Linear(PrefillResidentLinearAttentionScratch),
}

enum PrefillResidentTailShape {
    Dense {
        gate_proj: MetalLinearWeightBuffers,
        up_proj: MetalLinearWeightBuffers,
        down_proj: MetalLinearWeightBuffers,
        inter_dim: usize,
    },
    Routed {
        expert_count: usize,
        stacked: StackedMoeBuffers,
    },
    Shared {
        weights: MetalMoeSharedWeights,
    },
}

enum PrefillResidentTailScratch {
    Dense {
        gate: Buffer,
        up: Buffer,
        hidden: Buffer,
        down: Buffer,
    },
    Routed {
        router: Buffer,
        indices: Buffer,
        scores: Buffer,
        gate: Buffer,
        up: Buffer,
        hidden: Buffer,
        down: Buffer,
    },
    Shared,
}

enum PrefillResidentLayerCacheBuffer {
    Full { key: Buffer, value: Buffer },
    Linear { state: LinearAttentionMetalState },
}

#[derive(Default)]
struct PrefillSectionProfile {
    sections: HashMap<&'static str, PrefillSectionStat>,
}

#[derive(Default)]
struct PrefillSectionStat {
    encode_us: u128,
    wait_us: u128,
    count: u64,
}

impl PrefillSectionProfile {
    fn add(&mut self, label: &'static str, encode_us: u128, wait_us: u128) {
        let stat = self.sections.entry(label).or_default();
        stat.encode_us += encode_us;
        stat.wait_us += wait_us;
        stat.count += 1;
    }

    fn dump(&self) {
        let mut rows = self
            .sections
            .iter()
            .map(|(label, stat)| {
                (
                    *label,
                    stat.encode_us,
                    stat.wait_us,
                    stat.encode_us + stat.wait_us,
                    stat.count,
                )
            })
            .collect::<Vec<_>>();
        rows.sort_by(|left, right| right.3.cmp(&left.3));
        let total_us = rows.iter().map(|row| row.3).sum::<u128>();
        for (label, encode_us, wait_us, section_total_us, count) in rows {
            let pct = if total_us > 0 {
                100.0 * section_total_us as f64 / total_us as f64
            } else {
                0.0
            };
            eprintln!(
                "prefill_section label={label} count={count} encode_us={encode_us} wait_us={wait_us} total_us={section_total_us} pct={pct:.1}"
            );
        }
        eprintln!("prefill_section_total total_us={total_us}");
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "wrappers d'encodage Metal: buffers, dimensions et offsets restent explicites"
)]
impl MetalExecutor {
    fn allocate_prefill_tail_moe_scratch(
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

    fn allocate_prefill_resident_layer_scratch(
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

    fn check_prefill_tail_moe_shapes<'a>(
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

    fn check_prefill_resident_layer_shapes<'a>(
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

    pub(crate) fn full_attention_prefill_tail_moe(
        &self,
        residual: &Tensor,
        input_norm: &Tensor,
        q_proj: &Linear,
        k_proj: &Linear,
        v_proj: &Linear,
        o_proj: &Linear,
        q_norm: &Tensor,
        k_norm: &Tensor,
        post_norm: &Tensor,
        router: &Linear,
        experts: &[GatedMlp],
        top_k: usize,
        spec: PrefillAttentionSpec,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let shape = self.check_prefill_tail_moe_shapes(
            residual, input_norm, q_proj, k_proj, v_proj, o_proj, q_norm, k_norm, post_norm,
            router, experts, top_k, spec,
        )?;
        let hidden_dim = shape.hidden_dim;
        let q_dim = shape.q_dim;
        let kv_dim = shape.kv_dim;
        let expert_count = shape.expert_count;
        let scratch = self.allocate_prefill_tail_moe_scratch(residual, &shape, spec, top_k)?;
        let stacked = shape.stacked;
        let PrefillTailMoeScratch {
            residual: residual_buffer,
            input_norm: input_norm_buffer,
            q_norm: q_norm_buffer,
            k_norm: k_norm_buffer,
            post_norm: post_norm_buffer,
            normed: normed_buffer,
            q: q_buffer,
            k: k_buffer,
            v: v_buffer,
            q_rope: q_rope_buffer,
            k_rope: k_rope_buffer,
            context: context_buffer,
            o: o_buffer,
            attention_state: attention_state_buffer,
            post_normed: post_normed_buffer,
            router: router_buffer,
            output: output_buffer,
            indices: indices_buffer,
            scores: scores_buffer,
            gate: gate_buffer,
            up: up_buffer,
            hidden: hidden_buffer,
            down: down_buffer,
        } = scratch;
        let total_topk = checked_len(spec.seq, top_k, "prefill topk total")?;
        let inter_dim = stacked.gate.out_dim;
        let mut owned_buffers = Vec::new();

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let encoder_guard = EncoderEndGuard::new(encoder);
        self.encode_rms_norm_rows(
            encoder,
            &residual_buffer,
            &input_norm_buffer,
            &normed_buffer,
            spec.seq,
            hidden_dim,
            spec.eps,
        )?;
        self.encode_matmul_weight(
            encoder,
            &mut owned_buffers,
            &normed_buffer,
            spec.seq,
            hidden_dim,
            q_proj.weight(),
            &q_buffer,
        )?;
        self.encode_matmul_weight(
            encoder,
            &mut owned_buffers,
            &normed_buffer,
            spec.seq,
            hidden_dim,
            k_proj.weight(),
            &k_buffer,
        )?;
        self.encode_matmul_weight(
            encoder,
            &mut owned_buffers,
            &normed_buffer,
            spec.seq,
            hidden_dim,
            v_proj.weight(),
            &v_buffer,
        )?;
        self.encode_rms_norm_rope_heads(
            encoder,
            &q_buffer,
            &q_norm_buffer,
            &q_rope_buffer,
            spec,
            spec.q_heads,
        )?;
        self.encode_rms_norm_rope_heads(
            encoder,
            &k_buffer,
            &k_norm_buffer,
            &k_rope_buffer,
            spec,
            spec.kv_heads,
        )?;
        self.encode_causal_attention_prefill(
            encoder,
            &q_rope_buffer,
            &k_rope_buffer,
            &v_buffer,
            &context_buffer,
            spec,
        )?;
        self.encode_matmul_weight(
            encoder,
            &mut owned_buffers,
            &context_buffer,
            spec.seq,
            q_dim,
            o_proj.weight(),
            &o_buffer,
        )?;
        self.encode_add_rms_norm_rows(
            encoder,
            &residual_buffer,
            &o_buffer,
            &post_norm_buffer,
            &attention_state_buffer,
            &post_normed_buffer,
            spec.seq,
            hidden_dim,
            spec.eps,
        )?;
        self.encode_matmul_weight(
            encoder,
            &mut owned_buffers,
            &post_normed_buffer,
            spec.seq,
            hidden_dim,
            router.weight(),
            &router_buffer,
        )?;
        self.encode_topk_softmax_rows(
            encoder,
            &router_buffer,
            &indices_buffer,
            &scores_buffer,
            spec.seq,
            expert_count,
            top_k,
        )?;
        if !self.encode_gather_gate_up_swiglu(
            encoder,
            &mut owned_buffers,
            &post_normed_buffer,
            spec.seq,
            &stacked.gate,
            &stacked.up,
            &indices_buffer,
            total_topk,
            &hidden_buffer,
        )? {
            self.encode_gather_matmul(
                encoder,
                &mut owned_buffers,
                &post_normed_buffer,
                spec.seq,
                &stacked.gate,
                &indices_buffer,
                total_topk,
                &gate_buffer,
            )?;
            self.encode_gather_matmul(
                encoder,
                &mut owned_buffers,
                &post_normed_buffer,
                spec.seq,
                &stacked.up,
                &indices_buffer,
                total_topk,
                &up_buffer,
            )?;
            self.encode_swiglu(
                encoder,
                &mut owned_buffers,
                &gate_buffer,
                &up_buffer,
                &hidden_buffer,
                checked_len(total_topk, inter_dim, "prefill swiglu")?,
            )?;
        }
        self.encode_gather_matmul(
            encoder,
            &mut owned_buffers,
            &hidden_buffer,
            total_topk,
            &stacked.down,
            &indices_buffer,
            total_topk,
            &down_buffer,
        )?;
        self.encode_weighted_sum_add_grouped_topk(
            encoder,
            &mut owned_buffers,
            &down_buffer,
            &scores_buffer,
            &attention_state_buffer,
            &output_buffer,
            spec.seq,
            top_k,
            hidden_dim,
        )?;
        encoder_guard.end();
        commit_and_wait(command_buffer)?;

        let output = read_f32_buffer(&output_buffer, spec.seq * hidden_dim)?;
        let key = read_f32_buffer(&k_rope_buffer, spec.seq * kv_dim)?;
        let value = read_f32_buffer(&v_buffer, spec.seq * kv_dim)?;
        Ok((
            Tensor::from_vec(vec![spec.seq, hidden_dim], output)?,
            Tensor::from_vec(vec![spec.seq, kv_dim], key)?,
            Tensor::from_vec(vec![spec.seq, kv_dim], value)?,
        ))
    }

    pub(crate) fn encode_split_q_gate_rows(
        &self,
        encoder: &ComputeCommandEncoderRef,
        proj: &BufferRef,
        q: &BufferRef,
        gate: &BufferRef,
        seq: usize,
        q_heads: usize,
        head_dim: usize,
    ) -> Result<()> {
        let q_dim = checked_len(q_heads, head_dim, "split rows q_dim")?;
        self.encode_split_q_gate_rows_with_stride(
            encoder,
            proj,
            q,
            gate,
            seq,
            q_heads,
            head_dim,
            q_dim
                .checked_mul(2)
                .ok_or_else(|| InferError::Dimension("split rows stride déborde".to_string()))?,
        )
    }

    pub(crate) fn encode_split_q_gate_rows_with_stride(
        &self,
        encoder: &ComputeCommandEncoderRef,
        proj: &BufferRef,
        q: &BufferRef,
        gate: &BufferRef,
        seq: usize,
        q_heads: usize,
        head_dim: usize,
        row_stride: usize,
    ) -> Result<()> {
        let q_dim = checked_len(q_heads, head_dim, "split rows q_dim")?;
        let min_stride = q_dim
            .checked_mul(2)
            .ok_or_else(|| InferError::Dimension("split rows min stride déborde".to_string()))?;
        if row_stride < min_stride {
            return Err(InferError::Dimension(format!(
                "split rows stride {row_stride} < 2*q_dim {min_stride}",
            )));
        }
        let dims = [
            checked_u32(seq, "split rows seq")?,
            checked_u32(q_heads, "split rows q_heads")?,
            checked_u32(head_dim, "split rows head_dim")?,
            checked_u32(row_stride, "split rows stride")?,
        ];
        let total = checked_len(seq, q_dim, "split rows total")?;
        encoder.set_compute_pipeline_state(&self.split_q_gate_rows_f32);
        encoder.set_buffer(0, Some(proj), 0);
        encoder.set_buffer(1, Some(q), 0);
        encoder.set_buffer(2, Some(gate), 0);
        set_u32_bytes(encoder, 3, &dims, "split_q_gate_rows_dims")?;
        self.dispatch_1d(encoder, &self.split_q_gate_rows_f32, total)
    }

    pub(crate) fn encode_attn_gate_rows(
        &self,
        encoder: &ComputeCommandEncoderRef,
        ctx: &BufferRef,
        gate: &BufferRef,
        out: &BufferRef,
        n: usize,
    ) -> Result<()> {
        let len = checked_u32(n, "attn_gate_rows n")?;
        encoder.set_compute_pipeline_state(&self.attn_gate_rows_f32);
        encoder.set_buffer(0, Some(ctx), 0);
        encoder.set_buffer(1, Some(gate), 0);
        encoder.set_buffer(2, Some(out), 0);
        set_u32_bytes(encoder, 3, std::slice::from_ref(&len), "attn_gate_rows_n")?;
        self.dispatch_1d(encoder, &self.attn_gate_rows_f32, n)
    }

    /// Variante GATED + SHARED-EXPERT du prefill batché full-attn (Qwen3.5/3.6 :
    /// `attn_output_gate=true`, MoE à expert partagé). UN command buffer : input_norm +
    /// GEMM q(2·q_dim)/k/v + split q/gate + norm+RoPE q/k + attention causale batchée
    /// GPU + `ctx·σ(gate)` + o_proj + résiduel+post_norm + tail MoE shared (rows).
    /// Renvoie `(sortie [seq,hidden], k_roped [seq,kv_dim], v [seq,kv_dim])` pour le KV.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si une dimension déborde / l'exécution Metal échoue.
    #[expect(
        clippy::too_many_arguments,
        reason = "couche prefill gated+shared : poids attention + MoE shared + spec"
    )]
    pub(crate) fn full_attention_prefill_tail_moe_shared_gated(
        &self,
        residual: &Tensor,
        input_norm: &Tensor,
        q_proj: &Linear,
        k_proj: &Linear,
        v_proj: &Linear,
        o_proj: &Linear,
        q_norm: &Tensor,
        k_norm: &Tensor,
        post_norm: &Tensor,
        router: &Linear,
        experts: &[GatedMlp],
        shared_expert: &GatedMlp,
        shared_gate: &Linear,
        top_k: usize,
        spec: PrefillAttentionSpec,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let hidden_dim = spec.hidden_dim;
        let q_dim = checked_len(spec.q_heads, spec.head_dim, "gated prefill q_dim")?;
        let kv_dim = checked_len(spec.kv_heads, spec.head_dim, "gated prefill kv_dim")?;
        let q2_dim = q_dim
            .checked_mul(2)
            .ok_or_else(|| InferError::Dimension("gated prefill q2_dim déborde".to_string()))?;
        let (rseq, rhidden) = residual.as_matrix()?;
        if rseq != spec.seq || rhidden != hidden_dim {
            return Err(InferError::Dimension(format!(
                "gated prefill residual=[{rseq},{rhidden}], spec seq={} hidden={hidden_dim}",
                spec.seq
            )));
        }
        let moe_shared =
            self.resolve_moe_shared_weights(router, experts, shared_expert, shared_gate)?;

        let hidden_len = checked_len(spec.seq, hidden_dim, "gated prefill hidden")?;
        let q_len = checked_len(spec.seq, q_dim, "gated prefill q")?;
        let q2_len = checked_len(spec.seq, q2_dim, "gated prefill q2")?;
        let kv_len = checked_len(spec.seq, kv_dim, "gated prefill kv")?;

        let residual_buffer = self.upload_f32_buffer(residual.data(), "gated_prefill_residual")?;
        let input_norm_buffer =
            self.cached_buffer_from_f32(input_norm.data(), "gated_prefill_in_norm")?;
        let q_norm_buffer = self.cached_buffer_from_f32(q_norm.data(), "gated_prefill_q_norm")?;
        let k_norm_buffer = self.cached_buffer_from_f32(k_norm.data(), "gated_prefill_k_norm")?;
        let post_norm_buffer =
            self.cached_buffer_from_f32(post_norm.data(), "gated_prefill_post_norm")?;
        let normed = self.private_f32_buffer(hidden_len, "gated_prefill_normed")?;
        let q2 = self.private_f32_buffer(q2_len, "gated_prefill_q2")?;
        let q = self.private_f32_buffer(q_len, "gated_prefill_q")?;
        let gate = self.private_f32_buffer(q_len, "gated_prefill_gate")?;
        let k = self.private_f32_buffer(kv_len, "gated_prefill_k")?;
        let v = self.private_f32_buffer(kv_len, "gated_prefill_v")?;
        let q_rope = self.private_f32_buffer(q_len, "gated_prefill_q_rope")?;
        let k_rope = self.private_f32_buffer(kv_len, "gated_prefill_k_rope")?;
        let context = self.private_f32_buffer(q_len, "gated_prefill_context")?;
        let gated = self.private_f32_buffer(q_len, "gated_prefill_gated")?;
        let o = self.private_f32_buffer(hidden_len, "gated_prefill_o")?;
        let attention_state = self.private_f32_buffer(hidden_len, "gated_prefill_attn_state")?;
        let post_normed = self.private_f32_buffer(hidden_len, "gated_prefill_post_normed")?;
        let output = self.private_f32_buffer(hidden_len, "gated_prefill_output")?;
        let mut owned_buffers = Vec::new();

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let encoder_guard = EncoderEndGuard::new(encoder);
        self.encode_rms_norm_rows(
            encoder,
            &residual_buffer,
            &input_norm_buffer,
            &normed,
            spec.seq,
            hidden_dim,
            spec.eps,
        )?;
        self.encode_matmul_weight(
            encoder,
            &mut owned_buffers,
            &normed,
            spec.seq,
            hidden_dim,
            q_proj.weight(),
            &q2,
        )?;
        self.encode_matmul_weight(
            encoder,
            &mut owned_buffers,
            &normed,
            spec.seq,
            hidden_dim,
            k_proj.weight(),
            &k,
        )?;
        self.encode_matmul_weight(
            encoder,
            &mut owned_buffers,
            &normed,
            spec.seq,
            hidden_dim,
            v_proj.weight(),
            &v,
        )?;
        self.encode_split_q_gate_rows(
            encoder,
            &q2,
            &q,
            &gate,
            spec.seq,
            spec.q_heads,
            spec.head_dim,
        )?;
        self.encode_rms_norm_rope_heads(encoder, &q, &q_norm_buffer, &q_rope, spec, spec.q_heads)?;
        self.encode_rms_norm_rope_heads(encoder, &k, &k_norm_buffer, &k_rope, spec, spec.kv_heads)?;
        self.encode_causal_attention_prefill(encoder, &q_rope, &k_rope, &v, &context, spec)?;
        self.encode_attn_gate_rows(encoder, &context, &gate, &gated, q_len)?;
        self.encode_matmul_weight(
            encoder,
            &mut owned_buffers,
            &gated,
            spec.seq,
            q_dim,
            o_proj.weight(),
            &o,
        )?;
        self.encode_add_rms_norm_rows(
            encoder,
            &residual_buffer,
            &o,
            &post_norm_buffer,
            &attention_state,
            &post_normed,
            spec.seq,
            hidden_dim,
            spec.eps,
        )?;
        if moe_coop_enabled() && moe_shared.coop_compatible() {
            // MoE routé via le kernel gather_qmm porté : commit le CB d'attention
            // (post_normed/attention_state commités) puis MoE dans ses command buffers.
            encoder_guard.end();
            set_commit_label("fa_attn");
            commit_and_wait(command_buffer)?;
            self.moe_shared_rows_coop(
                &post_normed,
                Some(&attention_state),
                &output,
                spec.seq,
                hidden_dim,
                &moe_shared,
                top_k,
            )?;
        } else {
            self.encode_moe_shared_buffers_rows(
                encoder,
                &mut owned_buffers,
                &post_normed,
                Some(&attention_state),
                &output,
                spec.seq,
                hidden_dim,
                &moe_shared,
                top_k,
            )?;
            encoder_guard.end();
            set_commit_label("fa_layer_nocoop");
            commit_and_wait(command_buffer)?;
        }

        let output_vec = read_f32_buffer(&output, hidden_len)?;
        let key = read_f32_buffer(&k_rope, kv_len)?;
        let value = read_f32_buffer(&v, kv_len)?;
        Ok((
            Tensor::from_vec(vec![spec.seq, hidden_dim], output_vec)?,
            Tensor::from_vec(vec![spec.seq, kv_dim], key)?,
            Tensor::from_vec(vec![spec.seq, kv_dim], value)?,
        ))
    }

    fn run_prefill_profile_section<F>(
        &self,
        profile: &mut PrefillSectionProfile,
        label: &'static str,
        encode: F,
    ) -> Result<()>
    where
        F: FnOnce(&ComputeCommandEncoderRef, &mut Vec<Buffer>) -> Result<()>,
    {
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let encoder_guard = EncoderEndGuard::new(encoder);
        let mut owned_buffers = Vec::new();
        let encode_started = std::time::Instant::now();
        encode(encoder, &mut owned_buffers)?;
        let encode_us = encode_started.elapsed().as_micros();
        encoder_guard.end();
        let wait_started = std::time::Instant::now();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        let wait_us = wait_started.elapsed().as_micros();
        ensure_completed(command_buffer.status())?;
        profile.add(label, encode_us, wait_us);
        Ok(())
    }

    fn profile_f32_to_bf16_conversions(
        &self,
        profile: &mut PrefillSectionProfile,
        shapes: &[(usize, u64)],
    ) -> Result<()> {
        let Some(max_len) = shapes.iter().map(|(len, _)| *len).max() else {
            return Ok(());
        };
        let input = self.private_f32_buffer(max_len, "prefill_profile_bf16_input")?;
        let output = self.private_bf16_buffer(max_len, "prefill_profile_bf16_output")?;
        self.run_prefill_profile_section(profile, "f32_to_bf16", |encoder, _owned| {
            for (len, count) in shapes {
                for _ in 0..*count {
                    self.encode_f32_to_bf16(encoder, &input, &output, *len)?;
                }
            }
            Ok(())
        })?;
        reset_prefill_f32_to_bf16_shapes();
        Ok(())
    }

    pub(crate) fn qwen_moe_prefill_resident(
        &self,
        input: &Tensor,
        layers: &[PrefillMoeLayer<'_>],
        spec: PrefillAttentionSpec,
    ) -> Result<(Tensor, Vec<PrefillResidentLayerCache>)> {
        let profile_sections = prefill_profile_sections_enabled();
        if profile_sections {
            reset_prefill_f32_to_bf16_shapes();
        }
        let mut kernel_timing = if profile_sections {
            None
        } else {
            PrefillKernelTiming::try_new(&self.device, layers.len())
        };
        let profile_total_started = profile_sections.then(std::time::Instant::now);
        let mut section_profile = PrefillSectionProfile::default();
        let trace = !profile_sections && trace_prefill_enabled();
        let total_started = trace.then(std::time::Instant::now);
        let (seq, hidden_dim) = input.as_matrix()?;
        if seq != spec.seq || hidden_dim != spec.hidden_dim {
            return Err(InferError::Dimension(format!(
                "prefill résident input=[{seq},{hidden_dim}], spec seq={} hidden={}",
                spec.seq, spec.hidden_dim
            )));
        }
        if layers.is_empty() {
            return Err(InferError::Config(
                "prefill résident sans couche".to_string(),
            ));
        }
        let q_dim = spec.q_heads * spec.head_dim;
        let kv_dim = spec.kv_heads * spec.head_dim;
        let hidden_len = checked_len(spec.seq, hidden_dim, "prefill résident hidden")?;
        let kv_len = checked_len(spec.seq, kv_dim, "prefill résident kv")?;
        let input_buffer = self.upload_f32_buffer(input.data(), "resident_input")?;
        let hidden_a = self.private_f32_buffer(hidden_len, "resident_hidden_a")?;
        let hidden_b = self.private_f32_buffer(hidden_len, "resident_hidden_b")?;
        let mut current_buffer = input_buffer;
        let mut layer_cache_buffers = Vec::with_capacity(layers.len());
        let mut owned_buffers = Vec::new();

        let command_buffer = self.queue.new_command_buffer();
        let use_shared_encoder = kernel_timing
            .as_ref()
            .map_or(true, PrefillKernelTiming::uses_dispatch_boundary);
        let encoder = if use_shared_encoder {
            Some(command_buffer.new_compute_command_encoder())
        } else {
            None
        };
        let mut encoder_guard = encoder.map(EncoderEndGuard::new);
        let fallback_encoder = encoder_guard.as_ref().map(EncoderEndGuard::encoder);
        if profile_sections {
            let guard = encoder_guard
                .take()
                .expect("invariant: garde encodeur initialisée");
            guard.end();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            ensure_completed(command_buffer.status())?;
        }
        let encode_started = trace.then(std::time::Instant::now);
        macro_rules! run_section {
            ($label:literal, |$section_encoder:ident, $section_owned:ident| $body:block) => {{
                if profile_sections {
                    self.run_prefill_profile_section(
                        &mut section_profile,
                        $label,
                        |$section_encoder, $section_owned| $body,
                    )?;
                } else {
                    time_prefill_pass(
                        kernel_timing.as_mut(),
                        command_buffer,
                        fallback_encoder,
                        $label,
                        |$section_encoder| {
                            let $section_owned = &mut owned_buffers;
                            $body
                        },
                    )?;
                }
            }};
        }
        for (layer_index, layer) in layers.iter().enumerate() {
            let layer_shape = self.check_prefill_resident_layer_shapes(
                *layer,
                layer_index,
                hidden_dim,
                q_dim,
                kv_dim,
                spec,
            )?;
            let top_k = layer.tail.top_k();
            let scratch = self.allocate_prefill_resident_layer_scratch(
                &layer_shape,
                spec,
                hidden_dim,
                q_dim,
                kv_dim,
                top_k,
            )?;
            let tail_shape = layer_shape.tail;
            let PrefillResidentLayerScratch {
                input_norm: input_norm_buffer,
                post_norm: post_norm_buffer,
                normed: normed_buffer,
                attention: attention_scratch,
                attention_state: attention_state_buffer,
                post_normed: post_normed_buffer,
                tail: tail_scratch,
            } = scratch;
            let output_buffer = if layer_index % 2 == 0 {
                hidden_a.clone()
            } else {
                hidden_b.clone()
            };

            run_section!("input_norm", |encoder, _owned| {
                self.encode_rms_norm_rows(
                    encoder,
                    &current_buffer,
                    &input_norm_buffer,
                    &normed_buffer,
                    spec.seq,
                    hidden_dim,
                    spec.eps,
                )
            });
            let (attention_output_buffer, layer_cache_buffer) = match (
                layer.attention,
                attention_scratch,
            ) {
                (
                    PrefillAttentionLayer::Full {
                        q_proj,
                        k_proj,
                        v_proj,
                        o_proj,
                        gated,
                        ..
                    },
                    PrefillResidentAttentionScratch::Full(full_scratch),
                ) => {
                    let PrefillResidentFullAttentionScratch {
                        q_norm: q_norm_buffer,
                        k_norm: k_norm_buffer,
                        q2: q2_buffer,
                        gate: attn_gate_buffer,
                        q: q_buffer,
                        k: k_buffer,
                        v: v_buffer,
                        q_rope: q_rope_buffer,
                        k_rope: k_rope_buffer,
                        context: context_buffer,
                        gated_context: gated_context_buffer,
                        o: o_buffer,
                    } = full_scratch;
                    run_section!("qkv", |encoder, owned| {
                        if gated {
                            let (Some(q2_buffer), Some(attn_gate_buffer)) =
                                (q2_buffer.as_ref(), attn_gate_buffer.as_ref())
                            else {
                                return Err(InferError::Dimension(format!(
                                    "prefill résident full gated scratch incomplet couche {layer_index}"
                                )));
                            };
                            self.encode_matmul_weight(
                                encoder,
                                owned,
                                &normed_buffer,
                                spec.seq,
                                hidden_dim,
                                q_proj.weight(),
                                q2_buffer,
                            )?;
                            self.encode_split_q_gate_rows(
                                encoder,
                                q2_buffer,
                                &q_buffer,
                                attn_gate_buffer,
                                spec.seq,
                                spec.q_heads,
                                spec.head_dim,
                            )?;
                        } else {
                            self.encode_matmul_weight(
                                encoder,
                                owned,
                                &normed_buffer,
                                spec.seq,
                                hidden_dim,
                                q_proj.weight(),
                                &q_buffer,
                            )?;
                        }
                        self.encode_matmul_weight(
                            encoder,
                            owned,
                            &normed_buffer,
                            spec.seq,
                            hidden_dim,
                            k_proj.weight(),
                            &k_buffer,
                        )?;
                        self.encode_matmul_weight(
                            encoder,
                            owned,
                            &normed_buffer,
                            spec.seq,
                            hidden_dim,
                            v_proj.weight(),
                            &v_buffer,
                        )?;
                        self.encode_rms_norm_rope_heads(
                            encoder,
                            &q_buffer,
                            &q_norm_buffer,
                            &q_rope_buffer,
                            spec,
                            spec.q_heads,
                        )?;
                        self.encode_rms_norm_rope_heads(
                            encoder,
                            &k_buffer,
                            &k_norm_buffer,
                            &k_rope_buffer,
                            spec,
                            spec.kv_heads,
                        )
                    });
                    run_section!("causal_attention", |encoder, _owned| {
                        self.encode_causal_attention_prefill(
                            encoder,
                            &q_rope_buffer,
                            &k_rope_buffer,
                            &v_buffer,
                            &context_buffer,
                            spec,
                        )
                    });
                    let context_for_o = if gated {
                        let (Some(attn_gate_buffer), Some(gated_context_buffer)) =
                            (attn_gate_buffer.as_ref(), gated_context_buffer.as_ref())
                        else {
                            return Err(InferError::Dimension(format!(
                                "prefill résident full gated output scratch incomplet couche {layer_index}"
                            )));
                        };
                        run_section!("attn_gate", |encoder, _owned| {
                            self.encode_attn_gate_rows(
                                encoder,
                                &context_buffer,
                                attn_gate_buffer,
                                gated_context_buffer,
                                checked_len(spec.seq, q_dim, "resident gated context")?,
                            )
                        });
                        gated_context_buffer
                    } else {
                        &context_buffer
                    };
                    run_section!("o_proj", |encoder, owned| {
                        self.encode_matmul_weight(
                            encoder,
                            owned,
                            context_for_o,
                            spec.seq,
                            q_dim,
                            o_proj.weight(),
                            &o_buffer,
                        )?;
                        Ok(())
                    });
                    (
                        o_buffer,
                        PrefillResidentLayerCacheBuffer::Full {
                            key: k_rope_buffer,
                            value: v_buffer,
                        },
                    )
                }
                (
                    PrefillAttentionLayer::Linear {
                        weights,
                        spec: linear_spec,
                        dims,
                    },
                    PrefillResidentAttentionScratch::Linear(linear_scratch),
                ) => {
                    let PrefillResidentLinearAttentionScratch { output, state } = linear_scratch;
                    run_section!("linear_attention", |encoder, owned| {
                        self.encode_linear_attn_batch_resident(
                            encoder,
                            owned,
                            &normed_buffer,
                            &output,
                            spec.seq,
                            weights,
                            &state,
                            linear_spec,
                            dims,
                        )
                    });
                    (output, PrefillResidentLayerCacheBuffer::Linear { state })
                }
                _ => {
                    return Err(InferError::Dimension(format!(
                        "prefill résident scratch attention incohérent couche {layer_index}"
                    )));
                }
            };
            run_section!("o_postnorm", |encoder, _owned| {
                self.encode_add_rms_norm_rows(
                    encoder,
                    &current_buffer,
                    &attention_output_buffer,
                    &post_norm_buffer,
                    &attention_state_buffer,
                    &post_normed_buffer,
                    spec.seq,
                    hidden_dim,
                    spec.eps,
                )
            });
            match (layer.tail, tail_shape, tail_scratch) {
                (
                    PrefillMoeTail::Dense { .. },
                    PrefillResidentTailShape::Dense {
                        gate_proj,
                        up_proj,
                        down_proj,
                        inter_dim,
                    },
                    PrefillResidentTailScratch::Dense {
                        gate,
                        up,
                        hidden,
                        down,
                    },
                ) => {
                    run_section!("tail_dense", |encoder, owned| {
                        let inter_len = checked_len(spec.seq, inter_dim, "resident dense inter")?;
                        let gate_dim = self.encode_matmul_weight_buffers(
                            encoder,
                            &post_normed_buffer,
                            spec.seq,
                            hidden_dim,
                            &gate_proj,
                            &gate,
                            false,
                        )?;
                        let up_dim = self.encode_matmul_weight_buffers(
                            encoder,
                            &post_normed_buffer,
                            spec.seq,
                            hidden_dim,
                            &up_proj,
                            &up,
                            false,
                        )?;
                        if gate_dim != inter_dim || up_dim != inter_dim {
                            return Err(InferError::Dimension(format!(
                                "prefill résident dense gate/up sortent gate={gate_dim} up={up_dim}, attendu {inter_dim}"
                            )));
                        }
                        self.encode_swiglu(encoder, owned, &gate, &up, &hidden, inter_len)?;
                        let down_dim = self.encode_matmul_weight_buffers(
                            encoder, &hidden, spec.seq, inter_dim, &down_proj, &down, false,
                        )?;
                        if down_dim != hidden_dim {
                            return Err(InferError::Dimension(format!(
                                "prefill résident dense down sort {down_dim}, attendu {hidden_dim}"
                            )));
                        }
                        self.encode_copy(
                            encoder,
                            &attention_state_buffer,
                            &output_buffer,
                            hidden_len,
                        )?;
                        self.encode_accumulate_scaled(
                            encoder,
                            owned,
                            &down,
                            &output_buffer,
                            1.0,
                            hidden_len,
                        )
                    });
                }
                (
                    PrefillMoeTail::Routed { router, top_k, .. },
                    PrefillResidentTailShape::Routed {
                        expert_count,
                        stacked,
                    },
                    PrefillResidentTailScratch::Routed {
                        router: router_buffer,
                        indices: indices_buffer,
                        scores: scores_buffer,
                        gate: gate_buffer,
                        up: up_buffer,
                        hidden: hidden_buffer,
                        down: down_buffer,
                    },
                ) => {
                    let total_topk = checked_len(spec.seq, top_k, "resident topk total")?;
                    let inter_dim = stacked.gate.out_dim;
                    // Bascule routed-only coop gatée par un flag DÉDIÉ (défaut OFF) :
                    // l'oracle greedy 30B n'est pas qualifié (cf.
                    // `moe_routed_coop_prefill_enabled`). Le défaut garde le chemin
                    // gather-qmv par lignes, byte-identique à la base. Le routeur n'est
                    // résolu que dans la branche coop.
                    if moe_routed_coop_prefill_enabled()
                        && MetalMoeRoutedWeights::stacked_coop_compatible(&stacked)
                    {
                        let weights = MetalMoeRoutedWeights {
                            router: self.resolve_linear_weight_buffers(
                                router.weight(),
                                "resident_moe_router",
                            )?,
                            stacked,
                        };
                        run_section!("moe_routed_coop", |encoder, owned| {
                            self.encode_moe_routed_rows_coop(
                                encoder,
                                owned,
                                &post_normed_buffer,
                                Some(&attention_state_buffer),
                                &output_buffer,
                                &router_buffer,
                                &indices_buffer,
                                &scores_buffer,
                                &down_buffer,
                                spec.seq,
                                hidden_dim,
                                &weights,
                                top_k,
                            )
                        });
                    } else {
                        run_section!("router_topk", |encoder, owned| {
                            self.encode_matmul_weight(
                                encoder,
                                owned,
                                &post_normed_buffer,
                                spec.seq,
                                hidden_dim,
                                router.weight(),
                                &router_buffer,
                            )?;
                            self.encode_topk_softmax_rows(
                                encoder,
                                &router_buffer,
                                &indices_buffer,
                                &scores_buffer,
                                spec.seq,
                                expert_count,
                                top_k,
                            )
                        });
                        run_section!("moe_gate_up", |encoder, owned| {
                            if !self.encode_gather_gate_up_swiglu(
                                encoder,
                                owned,
                                &post_normed_buffer,
                                spec.seq,
                                &stacked.gate,
                                &stacked.up,
                                &indices_buffer,
                                total_topk,
                                &hidden_buffer,
                            )? {
                                self.encode_gather_matmul(
                                    encoder,
                                    owned,
                                    &post_normed_buffer,
                                    spec.seq,
                                    &stacked.gate,
                                    &indices_buffer,
                                    total_topk,
                                    &gate_buffer,
                                )?;
                                self.encode_gather_matmul(
                                    encoder,
                                    owned,
                                    &post_normed_buffer,
                                    spec.seq,
                                    &stacked.up,
                                    &indices_buffer,
                                    total_topk,
                                    &up_buffer,
                                )?;
                                self.encode_swiglu(
                                    encoder,
                                    owned,
                                    &gate_buffer,
                                    &up_buffer,
                                    &hidden_buffer,
                                    checked_len(total_topk, inter_dim, "resident swiglu")?,
                                )?;
                            }
                            Ok(())
                        });
                        run_section!("moe_down", |encoder, owned| {
                            self.encode_gather_matmul(
                                encoder,
                                owned,
                                &hidden_buffer,
                                total_topk,
                                &stacked.down,
                                &indices_buffer,
                                total_topk,
                                &down_buffer,
                            )
                        });
                        run_section!("moe_weighted_sum", |encoder, owned| {
                            self.encode_weighted_sum_add_grouped_topk(
                                encoder,
                                owned,
                                &down_buffer,
                                &scores_buffer,
                                &attention_state_buffer,
                                &output_buffer,
                                spec.seq,
                                top_k,
                                hidden_dim,
                            )
                        });
                    }
                }
                (
                    PrefillMoeTail::Shared { top_k, .. },
                    PrefillResidentTailShape::Shared { weights },
                    PrefillResidentTailScratch::Shared,
                ) => {
                    if moe_coop_enabled() && weights.coop_compatible() {
                        run_section!("moe_shared_coop", |encoder, owned| {
                            self.encode_moe_shared_rows_coop(
                                encoder,
                                owned,
                                &post_normed_buffer,
                                Some(&attention_state_buffer),
                                &output_buffer,
                                spec.seq,
                                hidden_dim,
                                &weights,
                                top_k,
                            )
                        });
                    } else {
                        run_section!("moe_shared_rows", |encoder, owned| {
                            self.encode_moe_shared_buffers_rows(
                                encoder,
                                owned,
                                &post_normed_buffer,
                                Some(&attention_state_buffer),
                                &output_buffer,
                                spec.seq,
                                hidden_dim,
                                &weights,
                                top_k,
                            )
                        });
                    }
                }
                _ => {
                    return Err(InferError::Dimension(format!(
                        "prefill résident tail MoE incohérent couche {layer_index}"
                    )));
                }
            }
            layer_cache_buffers.push(layer_cache_buffer);
            current_buffer = output_buffer;
        }
        let final_read_buffer = if private_scratch_enabled() {
            let shared = self.uncached_f32_buffer(hidden_len, "resident_final_output")?;
            run_section!("final_copy", |encoder, _owned| {
                self.encode_copy(encoder, &current_buffer, &shared, hidden_len)
            });
            shared
        } else {
            current_buffer.clone()
        };
        let encode_elapsed = encode_started.map(|started| started.elapsed());
        let wait_elapsed = if profile_sections {
            None
        } else {
            if let Some(guard) = encoder_guard.take() {
                guard.end();
            }
            if let Some(timing) = kernel_timing.as_ref() {
                timing.encode_resolve(command_buffer)?;
            }
            let wait_started = trace.then(std::time::Instant::now);
            command_buffer.commit();
            command_buffer.wait_until_completed();
            ensure_completed(command_buffer.status())?;
            if let Some(timing) = kernel_timing.as_ref() {
                if let Err(error) = timing.dump_report() {
                    eprintln!("gpu_timestamps report_error={error}");
                }
            }
            wait_started.map(|started| started.elapsed())
        };

        let read_started = trace.then(std::time::Instant::now);
        let profile_read_started = profile_sections.then(std::time::Instant::now);
        let output = read_f32_buffer(&final_read_buffer, hidden_len)?;
        let mut layer_caches = Vec::with_capacity(layer_cache_buffers.len());
        for cache_buffer in layer_cache_buffers {
            match cache_buffer {
                PrefillResidentLayerCacheBuffer::Full { key, value } => {
                    let key = read_f32_buffer(&key, kv_len)?;
                    let value = read_f32_buffer(&value, kv_len)?;
                    layer_caches.push(PrefillResidentLayerCache::Full {
                        key: Tensor::from_vec(vec![spec.seq, kv_dim], key)?,
                        value: Tensor::from_vec(vec![spec.seq, kv_dim], value)?,
                    });
                }
                PrefillResidentLayerCacheBuffer::Linear { state } => {
                    layer_caches.push(PrefillResidentLayerCache::Linear { state });
                }
            }
        }
        let read_elapsed = read_started.map(|started| started.elapsed());
        if let Some(started) = profile_read_started {
            section_profile.add("readback", started.elapsed().as_micros(), 0);
        }
        if let Some(total_started) = total_started {
            eprintln!(
                "prefill_resident profile encode_us={} wait_us={} read_us={} total_us={}",
                encode_elapsed.map_or(0, |elapsed| elapsed.as_micros()),
                wait_elapsed.map_or(0, |elapsed| elapsed.as_micros()),
                read_elapsed.map_or(0, |elapsed| elapsed.as_micros()),
                total_started.elapsed().as_micros()
            );
        }
        if let Some(profile_total_started) = profile_total_started {
            let conversion_shapes = take_prefill_f32_to_bf16_shapes();
            self.profile_f32_to_bf16_conversions(&mut section_profile, &conversion_shapes)?;
            eprintln!(
                "prefill_section_run seq={} layers={} total_wall_us={}",
                spec.seq,
                layers.len(),
                profile_total_started.elapsed().as_micros()
            );
            section_profile.dump();
        }
        Ok((
            Tensor::from_vec(vec![spec.seq, hidden_dim], output)?,
            layer_caches,
        ))
    }
}
