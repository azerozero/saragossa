//! Préfill full-attention et MoE résident.

use crate::decoder::flags::trace_prefill_enabled;

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
    q_norm_weight: &'a [f32],
    k_norm_weight: &'a [f32],
    post_norm_weight: &'a [f32],
    expert_count: usize,
    stacked: StackedMoeBuffers,
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
    indices: Buffer,
    scores: Buffer,
    gate: Buffer,
    up: Buffer,
    hidden: Buffer,
    down: Buffer,
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
        let q_len = checked_len(spec.seq, q_dim, "prefill résident q")?;
        let kv_len = checked_len(spec.seq, kv_dim, "prefill résident kv")?;
        let total_topk = checked_len(spec.seq, top_k, "resident topk total")?;
        let inter_dim = shape.stacked.gate.out_dim;
        Ok(PrefillResidentLayerScratch {
            input_norm: self.cached_buffer_from_f32(shape.norm_weight, "resident_input_norm")?,
            q_norm: self.cached_buffer_from_f32(shape.q_norm_weight, "resident_q_norm")?,
            k_norm: self.cached_buffer_from_f32(shape.k_norm_weight, "resident_k_norm")?,
            post_norm: self.cached_buffer_from_f32(shape.post_norm_weight, "resident_post_norm")?,
            normed: self.private_f32_buffer(hidden_len, "resident_normed")?,
            q: self.private_f32_buffer(q_len, "resident_q")?,
            k: self.uncached_f32_buffer(kv_len, "resident_k")?,
            v: self.uncached_f32_buffer(kv_len, "resident_v")?,
            q_rope: self.private_f32_buffer(q_len, "resident_q_rope")?,
            k_rope: self.uncached_f32_buffer(kv_len, "resident_k_rope")?,
            context: self.private_f32_buffer(q_len, "resident_context")?,
            o: self.private_f32_buffer(hidden_len, "resident_o")?,
            attention_state: self.private_f32_buffer(hidden_len, "resident_attention_state")?,
            post_normed: self.private_f32_buffer(hidden_len, "resident_post_normed")?,
            router: self.private_f32_buffer(
                checked_len(spec.seq, shape.expert_count, "resident router")?,
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
        ensure_biasless(layer.q_proj, "q_proj")?;
        ensure_biasless(layer.k_proj, "k_proj")?;
        ensure_biasless(layer.v_proj, "v_proj")?;
        ensure_biasless(layer.o_proj, "o_proj")?;
        ensure_biasless(layer.router, "router")?;
        if linear_out_dim(layer.q_proj.weight())? != q_dim
            || linear_out_dim(layer.k_proj.weight())? != kv_dim
            || linear_out_dim(layer.v_proj.weight())? != kv_dim
            || linear_out_dim(layer.o_proj.weight())? != hidden_dim
        {
            return Err(InferError::Dimension(format!(
                "prefill résident projections incompatibles couche {layer_index}"
            )));
        }
        let norm_weight = dense_vector(layer.input_norm, hidden_dim, "resident input_norm")?;
        let q_norm_weight = dense_vector(layer.q_norm, spec.head_dim, "resident q_norm")?;
        let k_norm_weight = dense_vector(layer.k_norm, spec.head_dim, "resident k_norm")?;
        let post_norm_weight = dense_vector(layer.post_norm, hidden_dim, "resident post_norm")?;
        let expert_count = linear_out_dim(layer.router.weight())?;
        ensure_valid_top_k(layer.top_k, expert_count)?;
        if expert_count != layer.experts.len() {
            return Err(InferError::Dimension(format!(
                "prefill résident router experts={expert_count}, poids experts={} couche {layer_index}",
                layer.experts.len()
            )));
        }
        let stacked = self.stacked_moe_buffers(layer.experts)?;
        if hidden_dim != stacked.gate.in_dim || hidden_dim != stacked.up.in_dim {
            return Err(InferError::Dimension(format!(
                "prefill résident hidden={hidden_dim}, gate_in={}, up_in={} couche {layer_index}",
                stacked.gate.in_dim, stacked.up.in_dim
            )));
        }
        if stacked.gate.out_dim != stacked.up.out_dim || stacked.down.in_dim != stacked.gate.out_dim
        {
            return Err(InferError::Dimension(format!(
                "prefill résident inter dims gate={} up={} down_in={} couche {layer_index}",
                stacked.gate.out_dim, stacked.up.out_dim, stacked.down.in_dim
            )));
        }
        Ok(PrefillResidentLayerShape {
            norm_weight,
            q_norm_weight,
            k_norm_weight,
            post_norm_weight,
            expert_count,
            stacked,
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

    pub(crate) fn qwen_moe_prefill_resident(
        &self,
        input: &Tensor,
        layers: &[PrefillMoeLayer<'_>],
        spec: PrefillAttentionSpec,
    ) -> Result<(Tensor, Vec<(Tensor, Tensor)>)> {
        let trace = trace_prefill_enabled();
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
        let mut key_value_buffers = Vec::with_capacity(layers.len());
        let mut owned_buffers = Vec::new();

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let encoder_guard = EncoderEndGuard::new(encoder);
        let encode_started = trace.then(std::time::Instant::now);
        for (layer_index, layer) in layers.iter().enumerate() {
            let layer_shape = self.check_prefill_resident_layer_shapes(
                *layer,
                layer_index,
                hidden_dim,
                q_dim,
                kv_dim,
                spec,
            )?;
            let expert_count = layer_shape.expert_count;
            let scratch = self.allocate_prefill_resident_layer_scratch(
                &layer_shape,
                spec,
                hidden_dim,
                q_dim,
                kv_dim,
                layer.top_k,
            )?;
            let stacked = layer_shape.stacked;
            let PrefillResidentLayerScratch {
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
                indices: indices_buffer,
                scores: scores_buffer,
                gate: gate_buffer,
                up: up_buffer,
                hidden: hidden_buffer,
                down: down_buffer,
            } = scratch;
            let total_topk = checked_len(spec.seq, layer.top_k, "resident topk total")?;
            let inter_dim = stacked.gate.out_dim;
            let output_buffer = if layer_index % 2 == 0 {
                hidden_a.clone()
            } else {
                hidden_b.clone()
            };

            self.encode_rms_norm_rows(
                encoder,
                &current_buffer,
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
                layer.q_proj.weight(),
                &q_buffer,
            )?;
            self.encode_matmul_weight(
                encoder,
                &mut owned_buffers,
                &normed_buffer,
                spec.seq,
                hidden_dim,
                layer.k_proj.weight(),
                &k_buffer,
            )?;
            self.encode_matmul_weight(
                encoder,
                &mut owned_buffers,
                &normed_buffer,
                spec.seq,
                hidden_dim,
                layer.v_proj.weight(),
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
                layer.o_proj.weight(),
                &o_buffer,
            )?;
            self.encode_add_rms_norm_rows(
                encoder,
                &current_buffer,
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
                layer.router.weight(),
                &router_buffer,
            )?;
            self.encode_topk_softmax_rows(
                encoder,
                &router_buffer,
                &indices_buffer,
                &scores_buffer,
                spec.seq,
                expert_count,
                layer.top_k,
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
                    checked_len(total_topk, inter_dim, "resident swiglu")?,
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
                layer.top_k,
                hidden_dim,
            )?;
            key_value_buffers.push((k_rope_buffer, v_buffer));
            current_buffer = output_buffer;
        }
        let final_read_buffer = if private_scratch_enabled() {
            let shared = self.uncached_f32_buffer(hidden_len, "resident_final_output")?;
            self.encode_copy(encoder, &current_buffer, &shared, hidden_len)?;
            shared
        } else {
            current_buffer.clone()
        };
        let encode_elapsed = encode_started.map(|started| started.elapsed());
        encoder_guard.end();
        let wait_started = trace.then(std::time::Instant::now);
        command_buffer.commit();
        command_buffer.wait_until_completed();
        let wait_elapsed = wait_started.map(|started| started.elapsed());
        ensure_completed(command_buffer.status())?;

        let read_started = trace.then(std::time::Instant::now);
        let output = read_f32_buffer(&final_read_buffer, hidden_len)?;
        let mut kv = Vec::with_capacity(key_value_buffers.len());
        for (key_buffer, value_buffer) in key_value_buffers {
            let key = read_f32_buffer(&key_buffer, kv_len)?;
            let value = read_f32_buffer(&value_buffer, kv_len)?;
            kv.push((
                Tensor::from_vec(vec![spec.seq, kv_dim], key)?,
                Tensor::from_vec(vec![spec.seq, kv_dim], value)?,
            ));
        }
        let read_elapsed = read_started.map(|started| started.elapsed());
        if let Some(total_started) = total_started {
            eprintln!(
                "prefill_resident profile encode_us={} wait_us={} read_us={} total_us={}",
                encode_elapsed.map_or(0, |elapsed| elapsed.as_micros()),
                wait_elapsed.map_or(0, |elapsed| elapsed.as_micros()),
                read_elapsed.map_or(0, |elapsed| elapsed.as_micros()),
                total_started.elapsed().as_micros()
            );
        }
        Ok((Tensor::from_vec(vec![spec.seq, hidden_dim], output)?, kv))
    }
}
