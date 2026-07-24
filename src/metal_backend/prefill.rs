//! Préfill full-attention et MoE résident.

use crate::runtime_flags::trace_prefill_enabled;

use super::kernel_timing::{time_prefill_pass, PrefillKernelTiming};
use super::*;

mod gemma_dense;
mod gemma_parallel;
mod global_attention;
mod resident;
mod scratch;
#[cfg(test)]
mod tests;

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
    GemmaDense {
        gate_proj: MetalLinearWeightBuffers,
        up_proj: MetalLinearWeightBuffers,
        down_proj: MetalLinearWeightBuffers,
        pre_feedforward_norm: Buffer,
        post_feedforward_norm: Buffer,
        layer_scalar: Option<f32>,
        inter_dim: usize,
    },
    GemmaParallel {
        dense_gate_proj: MetalLinearWeightBuffers,
        dense_up_proj: MetalLinearWeightBuffers,
        dense_down_proj: MetalLinearWeightBuffers,
        pre_feedforward_norm: Buffer,
        post_feedforward_norm_1: Buffer,
        moe: MetalMoeRoutedWeights,
        router_norm: Option<(Buffer, f32)>,
        per_expert_scale: Option<Buffer>,
        pre_feedforward_norm_2: Buffer,
        post_feedforward_norm_2: Buffer,
        post_feedforward_norm: Buffer,
        layer_scalar: Option<f32>,
        dense_inter_dim: usize,
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
    GemmaDense {
        ffn_input: Buffer,
        gate: Buffer,
        up: Buffer,
        geglu: Buffer,
        down: Buffer,
        ffn_normed: Buffer,
    },
    GemmaParallel {
        dense_input: Buffer,
        dense_gate: Buffer,
        dense_up: Buffer,
        dense_geglu: Buffer,
        dense_down: Buffer,
        dense_out: Buffer,
        moe_input: Buffer,
        moe_out: Buffer,
        ffn_out: Buffer,
        ffn_normed: Buffer,
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
    Full {
        key: Buffer,
        value: Buffer,
        kv_dim: usize,
    },
    Linear {
        state: LinearAttentionMetalState,
    },
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
        rows.sort_by_key(|row| std::cmp::Reverse(row.3));
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
}
