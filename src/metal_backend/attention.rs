//! Encodage Metal des couches full-attention en decode.

use super::attention_checks::TailMoeSharedShape;
use super::*;

const FULL_ATTN_PREFILL_TG_WIDTH: u64 = 256;
const FULL_ATTN_PREFILL_BATCH_LONG_TG_WIDTH: u64 = 32;
const STEEL_ATTN_BQ: usize = 32;
const STEEL_ATTN_BK: usize = 32;
const STEEL_ATTN_BD: usize = 64;
const STEEL_CAUSAL_ATTN_D256_BQ: usize = 32;
const STEEL_CAUSAL_ATTN_D256_BK: usize = 64;
const STEEL_CAUSAL_ATTN_D256_BD: usize = 256;

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct SteelAttnParams {
    b: i32,
    h: i32,
    d: i32,
    q_l: i32,
    k_l: i32,
    gqa_factor: i32,
    scale: f32,
    n_q: i32,
    n_k: i32,
    n_q_aligned: i32,
    n_k_aligned: i32,
    q_l_rem: i32,
    k_l_rem: i32,
    q_l_off: i32,
    q_strides: [i64; 3],
    k_strides: [i64; 3],
    v_strides: [i64; 3],
    o_strides: [i64; 3],
}

struct TailMoeSharedScratch {
    residual: Buffer,
    context: Buffer,
    norm: Buffer,
    o: Buffer,
    attention_state: Buffer,
    normed: Buffer,
    router: Buffer,
    indices: Buffer,
    scores: Buffer,
    gate: Buffer,
    up: Buffer,
    hidden: Buffer,
    down: Buffer,
    shared_gate: Buffer,
    shared_proj_gate: Buffer,
    shared_up: Buffer,
    shared_hidden: Buffer,
    shared_down: Buffer,
    output: Buffer,
}

pub(super) fn prefill_attn_batch_long_supported(spec: PrefillAttentionSpec) -> bool {
    spec.seq > 2048
        && matches!(
            (spec.q_heads, spec.kv_heads, spec.head_dim),
            (24, 4, 256) | (32, 4, 128) | (16, 2, 256)
        )
}

pub(super) fn prefill_attn_batch_mid_30b_supported(spec: PrefillAttentionSpec) -> bool {
    (257..=2048).contains(&spec.seq)
        && matches!((spec.q_heads, spec.kv_heads, spec.head_dim), (32, 4, 128))
}

pub(super) fn prefill_attn_batch_mid_35b_supported(spec: PrefillAttentionSpec) -> bool {
    (257..=2048).contains(&spec.seq)
        && matches!((spec.q_heads, spec.kv_heads, spec.head_dim), (16, 2, 256))
}

fn prefill_attn_batch_gqa8_d256_supported(spec: PrefillAttentionSpec) -> bool {
    matches!((spec.q_heads, spec.kv_heads, spec.head_dim), (16, 2, 256))
}

pub(super) fn prefill_attn_steel_d256_supported(spec: PrefillAttentionSpec) -> bool {
    spec.seq > 256
        && spec.q_heads == 16
        && spec.kv_heads == 2
        && spec.head_dim == STEEL_CAUSAL_ATTN_D256_BD
}

#[expect(
    clippy::too_many_arguments,
    reason = "wrappers d'encodage Metal: buffers, dimensions et offsets restent explicites"
)]
impl MetalExecutor {
    /// Calcule une attention prefill non causale `softmax(QK^T)V`.
    ///
    /// Les tenseurs sont au format aplati par ligne:
    /// `q=[seq, q_heads*head_dim]`, `k/v=[seq, kv_heads*head_dim]`.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si les formes sont incompatibles ou si Metal échoue.
    pub fn noncausal_attention_prefill(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        q_heads: usize,
        kv_heads: usize,
    ) -> Result<Tensor> {
        let (seq, q_dim) = q.as_matrix()?;
        let (k_seq, k_dim) = k.as_matrix()?;
        let (v_seq, v_dim) = v.as_matrix()?;
        if seq == 0 || seq > 2048 {
            return Err(InferError::Dimension(format!(
                "attention non causale Metal seq={seq}, attendu 1..=2048"
            )));
        }
        if k_seq != seq || v_seq != seq || k_dim != v_dim {
            return Err(InferError::Dimension(format!(
                "attention non causale q={:?}, k={:?}, v={:?}",
                q.shape(),
                k.shape(),
                v.shape()
            )));
        }
        if q_heads == 0 || kv_heads == 0 || q_heads % kv_heads != 0 {
            return Err(InferError::Dimension(format!(
                "attention non causale heads invalides q_heads={q_heads}, kv_heads={kv_heads}"
            )));
        }
        if q_dim % q_heads != 0 || k_dim % kv_heads != 0 {
            return Err(InferError::Dimension(format!(
                "attention non causale dims incompatibles q_dim={q_dim}, k_dim={k_dim}, q_heads={q_heads}, kv_heads={kv_heads}"
            )));
        }
        let head_dim = q_dim / q_heads;
        if k_dim / kv_heads != head_dim {
            return Err(InferError::Dimension(format!(
                "attention non causale head_dim q={}, kv={}",
                head_dim,
                k_dim / kv_heads
            )));
        }

        let q_buffer = self.upload_f32_buffer(q.data(), "noncausal_q")?;
        let k_buffer = self.upload_f32_buffer(k.data(), "noncausal_k")?;
        let v_buffer = self.upload_f32_buffer(v.data(), "noncausal_v")?;
        let output_len = checked_len(seq, q_dim, "sortie attention non causale")?;
        let output_buffer = self.device.new_buffer(
            byte_len::<f32>(output_len)?,
            MTLResourceOptions::StorageModeShared,
        );
        let spec = PrefillAttentionSpec {
            seq,
            hidden_dim: q_dim,
            q_heads,
            kv_heads,
            head_dim,
            rope_dims: head_dim,
            rope_frequency_dim: head_dim,
            rope_theta: 10_000.0,
            attn_scalar: head_dim as f32,
            window: None,
            k_eq_v: false,
            value_norm: false,
            eps: 0.0,
        };

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let encoder_guard = EncoderEndGuard::new(encoder);
        self.encode_noncausal_attention_prefill(
            encoder,
            &q_buffer,
            &k_buffer,
            &v_buffer,
            &output_buffer,
            spec,
        )?;
        encoder_guard.end();
        commit_and_wait(command_buffer)?;

        let output = read_f32_buffer(&output_buffer, output_len)?;
        Tensor::from_vec(vec![seq, q_dim], output)
    }

    /// Calcule une attention prefill causale `softmax(QK^T + masque)V` (tests).
    ///
    /// Sert à verrouiller la byte-identité des DEUX kernels causaux (court seq <=
    /// 256, long seq > 256) contre une référence CPU. Format aplati par ligne comme
    /// [`Self::noncausal_attention_prefill`]. Conservé comme API symétrique,
    /// exercé par `tests/attention.rs`.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si les formes sont incompatibles ou si Metal échoue.
    #[cfg(test)]
    pub(crate) fn causal_attention_prefill(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        q_heads: usize,
        kv_heads: usize,
    ) -> Result<Tensor> {
        let (seq, q_dim) = q.as_matrix()?;
        let (k_seq, k_dim) = k.as_matrix()?;
        let (v_seq, v_dim) = v.as_matrix()?;
        if seq == 0 || seq > 8192 {
            return Err(InferError::Dimension(format!(
                "attention causale Metal seq={seq}, attendu 1..=8192"
            )));
        }
        if k_seq != seq || v_seq != seq || k_dim != v_dim {
            return Err(InferError::Dimension(format!(
                "attention causale q={:?}, k={:?}, v={:?}",
                q.shape(),
                k.shape(),
                v.shape()
            )));
        }
        if q_heads == 0 || kv_heads == 0 || q_heads % kv_heads != 0 {
            return Err(InferError::Dimension(format!(
                "attention causale heads invalides q_heads={q_heads}, kv_heads={kv_heads}"
            )));
        }
        if q_dim % q_heads != 0 || k_dim % kv_heads != 0 {
            return Err(InferError::Dimension(format!(
                "attention causale dims incompatibles q_dim={q_dim}, k_dim={k_dim}, q_heads={q_heads}, kv_heads={kv_heads}"
            )));
        }
        let head_dim = q_dim / q_heads;
        if k_dim / kv_heads != head_dim {
            return Err(InferError::Dimension(format!(
                "attention causale head_dim q={}, kv={}",
                head_dim,
                k_dim / kv_heads
            )));
        }

        let q_buffer = self.upload_f32_buffer(q.data(), "causal_q")?;
        let k_buffer = self.upload_f32_buffer(k.data(), "causal_k")?;
        let v_buffer = self.upload_f32_buffer(v.data(), "causal_v")?;
        let output_len = checked_len(seq, q_dim, "sortie attention causale")?;
        let output_buffer = self.device.new_buffer(
            byte_len::<f32>(output_len)?,
            MTLResourceOptions::StorageModeShared,
        );
        let spec = PrefillAttentionSpec {
            seq,
            hidden_dim: q_dim,
            q_heads,
            kv_heads,
            head_dim,
            rope_dims: head_dim,
            rope_frequency_dim: head_dim,
            rope_theta: 10_000.0,
            attn_scalar: head_dim as f32,
            window: None,
            k_eq_v: false,
            value_norm: false,
            eps: 0.0,
        };

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let encoder_guard = EncoderEndGuard::new(encoder);
        self.encode_causal_attention_prefill(
            encoder,
            &q_buffer,
            &k_buffer,
            &v_buffer,
            &output_buffer,
            spec,
        )?;
        encoder_guard.end();
        commit_and_wait(command_buffer)?;

        let output = read_f32_buffer(&output_buffer, output_len)?;
        Tensor::from_vec(vec![seq, q_dim], output)
    }

    fn allocate_tail_moe_shared_scratch(
        &self,
        residual: &Tensor,
        context: &Tensor,
        shape: &TailMoeSharedShape<'_>,
        top_k: usize,
    ) -> Result<TailMoeSharedScratch> {
        let hidden_dim = shape.hidden_dim;
        let inter_dim = shape.inter_dim;
        let shared_inter_dim = shape.shared_inter_dim;
        Ok(TailMoeSharedScratch {
            residual: self.upload_f32_buffer(residual.data(), "tail_shared_residual")?,
            context: self.upload_f32_buffer(context.data(), "tail_shared_context")?,
            norm: self.cached_buffer_from_f32(shape.norm_weight.data(), "tail_shared_norm")?,
            o: self.private_f32_buffer(hidden_dim, "tail_shared_o")?,
            attention_state: self.private_f32_buffer(hidden_dim, "tail_shared_attention_state")?,
            normed: self.private_f32_buffer(hidden_dim, "tail_shared_normed")?,
            router: self.private_f32_buffer(shape.expert_count, "tail_shared_router_logits")?,
            indices: self.private_u32_buffer(top_k, "tail_shared_indices")?,
            scores: self.private_f32_buffer(top_k, "tail_shared_scores")?,
            gate: self.private_f32_buffer(
                checked_len(top_k, inter_dim, "tail shared gate")?,
                "tail_shared_gate",
            )?,
            up: self.private_f32_buffer(
                checked_len(top_k, inter_dim, "tail shared up")?,
                "tail_shared_up",
            )?,
            hidden: self.private_f32_buffer(
                checked_len(top_k, inter_dim, "tail shared hidden")?,
                "tail_shared_hidden",
            )?,
            down: self.private_f32_buffer(
                checked_len(top_k, hidden_dim, "tail shared down")?,
                "tail_shared_down",
            )?,
            shared_gate: self.private_f32_buffer(1, "tail_shared_gate_scalar")?,
            shared_proj_gate: self.private_f32_buffer(shared_inter_dim, "tail_shared_proj_gate")?,
            shared_up: self.private_f32_buffer(shared_inter_dim, "tail_shared_proj_up")?,
            shared_hidden: self.private_f32_buffer(shared_inter_dim, "tail_shared_proj_hidden")?,
            shared_down: self.private_f32_buffer(hidden_dim, "tail_shared_proj_down")?,
            output: self.new_f32_buffer(hidden_dim, "tail_shared_output")?,
        })
    }

    /// Fusionne la fin d'une couche Qwen MoE après le contexte d'attention.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si les projections ne sont pas biasless ou si les
    /// formes ne correspondent pas au chemin MoE empilé.
    pub(crate) fn full_attention_tail_moe(
        &self,
        residual: &Tensor,
        context: &Tensor,
        o_proj: &Linear,
        post_norm: &Tensor,
        router: &Linear,
        experts: &[GatedMlp],
        top_k: usize,
        eps: f32,
    ) -> Result<Tensor> {
        let shape = self
            .check_tail_moe_shapes(residual, context, o_proj, post_norm, router, experts, top_k)?;
        let batch = shape.batch;
        let hidden_dim = shape.hidden_dim;
        let context_dim = shape.context_dim;
        let norm_weight = shape.norm_weight;
        let expert_count = shape.expert_count;
        let inter_dim = shape.inter_dim;
        let stacked = shape.stacked;
        let residual_buffer = self.upload_f32_buffer(residual.data(), "tail_residual")?;
        let context_buffer = self.upload_f32_buffer(context.data(), "tail_context")?;
        let norm_weight_buffer = self.cached_buffer_from_f32(norm_weight.data(), "tail_norm")?;
        let o_buffer = self.private_f32_buffer(hidden_dim, "tail_o")?;
        let attention_state_buffer = self.private_f32_buffer(hidden_dim, "tail_attention_state")?;
        let normed_buffer = self.private_f32_buffer(hidden_dim, "tail_normed")?;
        let router_buffer = self.private_f32_buffer(expert_count, "tail_router_logits")?;
        let indices_buffer = self.private_u32_buffer(top_k, "tail_indices")?;
        let scores_buffer = self.private_f32_buffer(top_k, "tail_scores")?;
        let gate_buffer =
            self.private_f32_buffer(checked_len(top_k, inter_dim, "tail gate")?, "tail_gate")?;
        let up_buffer =
            self.private_f32_buffer(checked_len(top_k, inter_dim, "tail up")?, "tail_up")?;
        let hidden_buffer =
            self.private_f32_buffer(checked_len(top_k, inter_dim, "tail hidden")?, "tail_hidden")?;
        let down_buffer =
            self.private_f32_buffer(checked_len(top_k, hidden_dim, "tail down")?, "tail_down")?;
        let output_buffer = self.new_f32_buffer(hidden_dim, "tail_output")?;
        let mut owned_buffers = Vec::new();

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let encoder_guard = EncoderEndGuard::new(encoder);
        let projected_dim = self.encode_matmul_weight(
            encoder,
            &mut owned_buffers,
            &context_buffer,
            batch,
            context_dim,
            o_proj.weight(),
            &o_buffer,
        )?;
        if projected_dim != hidden_dim {
            return Err(InferError::Dimension(format!(
                "tail MoE o_proj sort {projected_dim}, attendu {hidden_dim}"
            )));
        }
        self.encode_add_rms_norm(
            encoder,
            &mut owned_buffers,
            &residual_buffer,
            &o_buffer,
            &norm_weight_buffer,
            &attention_state_buffer,
            &normed_buffer,
            hidden_dim,
            eps,
        )?;
        let router_out_dim = self.encode_matmul_weight(
            encoder,
            &mut owned_buffers,
            &normed_buffer,
            batch,
            hidden_dim,
            router.weight(),
            &router_buffer,
        )?;
        if router_out_dim != expert_count {
            return Err(InferError::Dimension(format!(
                "tail MoE routeur sort {router_out_dim}, attendu {expert_count}"
            )));
        }
        self.encode_topk_softmax(
            encoder,
            &mut owned_buffers,
            &router_buffer,
            &indices_buffer,
            &scores_buffer,
            expert_count,
            top_k,
        )?;
        if !self.encode_gather_gate_up_swiglu(
            encoder,
            &mut owned_buffers,
            &normed_buffer,
            1,
            &stacked.gate,
            &stacked.up,
            &indices_buffer,
            top_k,
            &hidden_buffer,
        )? {
            self.encode_gather_matmul(
                encoder,
                &mut owned_buffers,
                &normed_buffer,
                1,
                &stacked.gate,
                &indices_buffer,
                top_k,
                &gate_buffer,
            )?;
            self.encode_gather_matmul(
                encoder,
                &mut owned_buffers,
                &normed_buffer,
                1,
                &stacked.up,
                &indices_buffer,
                top_k,
                &up_buffer,
            )?;
            self.encode_swiglu(
                encoder,
                &mut owned_buffers,
                &gate_buffer,
                &up_buffer,
                &hidden_buffer,
                checked_len(top_k, inter_dim, "tail swiglu")?,
            )?;
        }
        self.encode_gather_matmul(
            encoder,
            &mut owned_buffers,
            &hidden_buffer,
            top_k,
            &stacked.down,
            &indices_buffer,
            top_k,
            &down_buffer,
        )?;
        self.encode_weighted_sum_add_topk(
            encoder,
            &mut owned_buffers,
            &down_buffer,
            &scores_buffer,
            &attention_state_buffer,
            &output_buffer,
            top_k,
            hidden_dim,
        )?;
        encoder_guard.end();
        commit_and_wait(command_buffer)?;

        let output = read_f32_buffer(&output_buffer, hidden_dim)?;
        Tensor::from_vec(vec![1, hidden_dim], output)
    }

    /// Fusionne la fin d'une couche Qwen MoE avec shared expert.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si les poids shared ou MoE ne correspondent pas aux
    /// formes attendues par le chemin Metal.
    pub(crate) fn full_attention_tail_moe_shared(
        &self,
        residual: &Tensor,
        context: &Tensor,
        o_proj: &Linear,
        post_norm: &Tensor,
        router: &Linear,
        experts: &[GatedMlp],
        top_k: usize,
        shared_expert: &GatedMlp,
        shared_gate: &Linear,
        eps: f32,
    ) -> Result<Tensor> {
        let shape = self.check_tail_moe_shared_shapes(
            residual,
            context,
            o_proj,
            post_norm,
            router,
            experts,
            top_k,
            shared_expert,
            shared_gate,
        )?;
        let batch = shape.batch;
        let hidden_dim = shape.hidden_dim;
        let context_dim = shape.context_dim;
        let expert_count = shape.expert_count;
        let inter_dim = shape.inter_dim;
        let shared_gate_proj = shape.shared_gate_proj;
        let shared_up_proj = shape.shared_up_proj;
        let shared_down_proj = shape.shared_down_proj;
        let shared_inter_dim = shape.shared_inter_dim;
        let scratch = self.allocate_tail_moe_shared_scratch(residual, context, &shape, top_k)?;
        let stacked = shape.stacked;
        let TailMoeSharedScratch {
            residual: residual_buffer,
            context: context_buffer,
            norm: norm_weight_buffer,
            o: o_buffer,
            attention_state: attention_state_buffer,
            normed: normed_buffer,
            router: router_buffer,
            indices: indices_buffer,
            scores: scores_buffer,
            gate: gate_buffer,
            up: up_buffer,
            hidden: hidden_buffer,
            down: down_buffer,
            shared_gate: shared_gate_buffer,
            shared_proj_gate: shared_proj_gate_buffer,
            shared_up: shared_up_buffer,
            shared_hidden: shared_hidden_buffer,
            shared_down: shared_down_buffer,
            output: output_buffer,
        } = scratch;
        let mut owned_buffers = Vec::new();

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let encoder_guard = EncoderEndGuard::new(encoder);
        let projected_dim = self.encode_matmul_weight(
            encoder,
            &mut owned_buffers,
            &context_buffer,
            batch,
            context_dim,
            o_proj.weight(),
            &o_buffer,
        )?;
        if projected_dim != hidden_dim {
            return Err(InferError::Dimension(format!(
                "tail MoE shared o_proj sort {projected_dim}, attendu {hidden_dim}"
            )));
        }
        self.encode_add_rms_norm(
            encoder,
            &mut owned_buffers,
            &residual_buffer,
            &o_buffer,
            &norm_weight_buffer,
            &attention_state_buffer,
            &normed_buffer,
            hidden_dim,
            eps,
        )?;
        let router_out_dim = self.encode_matmul_weight(
            encoder,
            &mut owned_buffers,
            &normed_buffer,
            batch,
            hidden_dim,
            router.weight(),
            &router_buffer,
        )?;
        if router_out_dim != expert_count {
            return Err(InferError::Dimension(format!(
                "tail MoE shared routeur sort {router_out_dim}, attendu {expert_count}"
            )));
        }
        self.encode_topk_softmax(
            encoder,
            &mut owned_buffers,
            &router_buffer,
            &indices_buffer,
            &scores_buffer,
            expert_count,
            top_k,
        )?;
        if !self.encode_gather_gate_up_swiglu(
            encoder,
            &mut owned_buffers,
            &normed_buffer,
            1,
            &stacked.gate,
            &stacked.up,
            &indices_buffer,
            top_k,
            &hidden_buffer,
        )? {
            self.encode_gather_matmul(
                encoder,
                &mut owned_buffers,
                &normed_buffer,
                1,
                &stacked.gate,
                &indices_buffer,
                top_k,
                &gate_buffer,
            )?;
            self.encode_gather_matmul(
                encoder,
                &mut owned_buffers,
                &normed_buffer,
                1,
                &stacked.up,
                &indices_buffer,
                top_k,
                &up_buffer,
            )?;
            self.encode_swiglu(
                encoder,
                &mut owned_buffers,
                &gate_buffer,
                &up_buffer,
                &hidden_buffer,
                checked_len(top_k, inter_dim, "tail shared swiglu")?,
            )?;
        }
        self.encode_gather_matmul(
            encoder,
            &mut owned_buffers,
            &hidden_buffer,
            top_k,
            &stacked.down,
            &indices_buffer,
            top_k,
            &down_buffer,
        )?;
        self.encode_weighted_sum_add_topk(
            encoder,
            &mut owned_buffers,
            &down_buffer,
            &scores_buffer,
            &attention_state_buffer,
            &output_buffer,
            top_k,
            hidden_dim,
        )?;
        let projected_gate_dim = self.encode_matmul_weight(
            encoder,
            &mut owned_buffers,
            &normed_buffer,
            batch,
            hidden_dim,
            shared_gate.weight(),
            &shared_gate_buffer,
        )?;
        if projected_gate_dim != 1 {
            return Err(InferError::Dimension(format!(
                "tail shared gate Metal sort {projected_gate_dim}, attendu 1"
            )));
        }
        let projected_shared_gate_dim = self.encode_matmul_weight(
            encoder,
            &mut owned_buffers,
            &normed_buffer,
            batch,
            hidden_dim,
            shared_gate_proj.weight(),
            &shared_proj_gate_buffer,
        )?;
        let projected_shared_up_dim = self.encode_matmul_weight(
            encoder,
            &mut owned_buffers,
            &normed_buffer,
            batch,
            hidden_dim,
            shared_up_proj.weight(),
            &shared_up_buffer,
        )?;
        if projected_shared_gate_dim != shared_inter_dim
            || projected_shared_up_dim != shared_inter_dim
        {
            return Err(InferError::Dimension(format!(
                "tail shared expert Metal proj gate={projected_shared_gate_dim}, up={projected_shared_up_dim}, attendu {shared_inter_dim}"
            )));
        }
        self.encode_swiglu(
            encoder,
            &mut owned_buffers,
            &shared_proj_gate_buffer,
            &shared_up_buffer,
            &shared_hidden_buffer,
            shared_inter_dim,
        )?;
        let projected_shared_down_dim = self.encode_matmul_weight(
            encoder,
            &mut owned_buffers,
            &shared_hidden_buffer,
            batch,
            shared_inter_dim,
            shared_down_proj.weight(),
            &shared_down_buffer,
        )?;
        if projected_shared_down_dim != hidden_dim {
            return Err(InferError::Dimension(format!(
                "tail shared expert Metal down sort {projected_shared_down_dim}, attendu {hidden_dim}"
            )));
        }
        self.encode_add_sigmoid_scaled(
            encoder,
            &shared_down_buffer,
            &shared_gate_buffer,
            &output_buffer,
            hidden_dim,
        )?;
        encoder_guard.end();
        commit_and_wait(command_buffer)?;

        let output = read_f32_buffer(&output_buffer, hidden_dim)?;
        Tensor::from_vec(vec![1, hidden_dim], output)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "kernel spécialisé full-attn: QKV concat + split q/gate"
    )]
    pub(crate) fn encode_full_attn_qkv_split_buffers(
        &self,
        encoder: &ComputeCommandEncoderRef,
        lhs_buffer: &BufferRef,
        in_dim: usize,
        weight: &MetalLinearWeightBuffers,
        qkv_output_buffer: &BufferRef,
        q_output_buffer: &BufferRef,
        gate_output_buffer: &BufferRef,
        q_heads: usize,
        head_dim: usize,
    ) -> Result<Option<usize>> {
        if !fused_attn_epilogue_enabled() {
            return Ok(None);
        }
        let MetalLinearWeightBuffers::AffineQuantized {
            packed,
            scales,
            biases,
            out_dim,
            in_dim: weight_in_dim,
            packed_cols,
            group_size,
            bits,
            groups,
        } = weight
        else {
            return Ok(None);
        };
        if in_dim != *weight_in_dim {
            return Err(InferError::Dimension(format!(
                "full-attn qkv split x=[1,{in_dim}] rhs=[{out_dim},{weight_in_dim}]"
            )));
        }
        let can_use_u4 = fast_affine_qmv_enabled(*out_dim)
            && *bits == FAST_QMV_BITS
            && *group_size == FAST_QMV_GROUP_SIZE
            && in_dim % 512 == 0;
        let can_use_u8 = full_qkv_split_rms_u8_enabled()
            && can_use_fast_affine_qmv_u8_buffers(1, in_dim, *out_dim, *group_size, *bits)
            && *group_size == FAST_QMV_GROUP_SIZE;
        if !can_use_u4 && !can_use_u8 {
            return Ok(None);
        }
        let q_dim = q_heads
            .checked_mul(head_dim)
            .ok_or_else(|| InferError::Dimension("full-attn q_dim déborde".to_string()))?;
        let q_gate_dim = q_dim
            .checked_mul(2)
            .ok_or_else(|| InferError::Dimension("full-attn q_gate_dim déborde".to_string()))?;
        if *out_dim < q_gate_dim {
            return Err(InferError::Dimension(format!(
                "full-attn qkv split out_dim={out_dim}, q_gate_dim={q_gate_dim}"
            )));
        }
        let fast_dims = [
            checked_u32(*out_dim, "full qkv split out_dim")?,
            checked_u32(in_dim, "full qkv split in_dim")?,
            checked_u32(*packed_cols, "full qkv split packed_cols")?,
            checked_u32(*groups, "full qkv split groups")?,
        ];
        let q_dims = [
            checked_u32(q_heads, "full qkv split q_heads")?,
            checked_u32(head_dim, "full qkv split head_dim")?,
        ];
        encoder.set_compute_pipeline_state(&self.affine_qkv_split_qmv_fast_u4_gs64_f32);
        encoder.set_buffer(0, Some(lhs_buffer), 0);
        encoder.set_buffer(1, Some(packed), 0);
        encoder.set_buffer(2, Some(scales), 0);
        encoder.set_buffer(3, Some(biases), 0);
        encoder.set_buffer(4, Some(qkv_output_buffer), 0);
        encoder.set_buffer(5, Some(q_output_buffer), 0);
        encoder.set_buffer(6, Some(gate_output_buffer), 0);
        set_u32_bytes(encoder, 7, &fast_dims, "full_qkv_split_dims")?;
        set_u32_bytes(encoder, 8, &q_dims, "full_qkv_split_q_dims")?;
        profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(
                1,
                checked_nsuint(out_dim.div_ceil(8), "full qkv split out groups")?,
                1,
            ),
            MTLSize::new(64, 1, 1),
        );
        post_dispatch_barrier(encoder);
        Ok(Some(*out_dim))
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "kernel spécialisé full-attn: rms_norm prologue + QKV concat + split q/gate"
    )]
    pub(crate) fn encode_full_attn_qkv_split_rms_buffers(
        &self,
        encoder: &ComputeCommandEncoderRef,
        lhs_buffer: &BufferRef,
        rms_weight_buffer: &BufferRef,
        rms_eps: f32,
        in_dim: usize,
        weight: &MetalLinearWeightBuffers,
        qkv_output_buffer: &BufferRef,
        q_output_buffer: &BufferRef,
        gate_output_buffer: &BufferRef,
        q_heads: usize,
        head_dim: usize,
    ) -> Result<Option<usize>> {
        if !fused_rms_prologue_enabled() || !fused_attn_epilogue_enabled() {
            return Ok(None);
        }
        let MetalLinearWeightBuffers::AffineQuantized {
            packed,
            scales,
            biases,
            out_dim,
            in_dim: weight_in_dim,
            packed_cols,
            group_size,
            bits,
            groups,
        } = weight
        else {
            return Ok(None);
        };
        if in_dim != *weight_in_dim {
            return Err(InferError::Dimension(format!(
                "full-attn qkv split rms x=[1,{in_dim}] rhs=[{out_dim},{weight_in_dim}]"
            )));
        }
        let can_use_u4 = fast_affine_qmv_enabled(*out_dim)
            && *bits == FAST_QMV_BITS
            && *group_size == FAST_QMV_GROUP_SIZE
            && in_dim % 512 == 0;
        let can_use_u8 = full_qkv_split_rms_u8_enabled()
            && can_use_fast_affine_qmv_u8_buffers(1, in_dim, *out_dim, *group_size, *bits)
            && *group_size == FAST_QMV_GROUP_SIZE;
        if !can_use_u4 && !can_use_u8 {
            return Ok(None);
        }
        let q_dim = q_heads
            .checked_mul(head_dim)
            .ok_or_else(|| InferError::Dimension("full-attn q_dim déborde".to_string()))?;
        let q_gate_dim = q_dim
            .checked_mul(2)
            .ok_or_else(|| InferError::Dimension("full-attn q_gate_dim déborde".to_string()))?;
        if *out_dim < q_gate_dim {
            return Err(InferError::Dimension(format!(
                "full-attn qkv split rms out_dim={out_dim}, q_gate_dim={q_gate_dim}"
            )));
        }
        let fast_dims = [
            checked_u32(*out_dim, "full qkv split rms out_dim")?,
            checked_u32(in_dim, "full qkv split rms in_dim")?,
            checked_u32(*packed_cols, "full qkv split rms packed_cols")?,
            checked_u32(*groups, "full qkv split rms groups")?,
        ];
        let q_dims = [
            checked_u32(q_heads, "full qkv split rms q_heads")?,
            checked_u32(head_dim, "full qkv split rms head_dim")?,
        ];
        let pipeline = if can_use_u4 {
            &self.affine_qkv_split_rms_qmv_fast_u4_gs64_f32
        } else {
            &self.affine_qkv_split_rms_qmv_fast_u8_gs64_f32
        };
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(lhs_buffer), 0);
        encoder.set_buffer(1, Some(rms_weight_buffer), 0);
        encoder.set_buffer(2, Some(packed), 0);
        encoder.set_buffer(3, Some(scales), 0);
        encoder.set_buffer(4, Some(biases), 0);
        encoder.set_buffer(5, Some(qkv_output_buffer), 0);
        encoder.set_buffer(6, Some(q_output_buffer), 0);
        encoder.set_buffer(7, Some(gate_output_buffer), 0);
        set_u32_bytes(encoder, 8, &fast_dims, "full_qkv_split_rms_dims")?;
        set_u32_bytes(encoder, 9, &q_dims, "full_qkv_split_rms_q_dims")?;
        set_f32_bytes(encoder, 10, &[rms_eps], "full_qkv_split_rms_eps")?;
        profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(
                1,
                checked_nsuint(out_dim.div_ceil(8), "full qkv split rms out groups")?,
                1,
            ),
            MTLSize::new(64, 1, 1),
        );
        post_dispatch_barrier(encoder);
        Ok(Some(*out_dim))
    }

    pub(crate) fn encode_full_attn_o_proj_gated_buffers(
        &self,
        encoder: &ComputeCommandEncoderRef,
        ctx_buffer: &BufferRef,
        gate_buffer: &BufferRef,
        in_dim: usize,
        weight: &MetalLinearWeightBuffers,
        output_buffer: &BufferRef,
    ) -> Result<Option<usize>> {
        if !fused_attn_epilogue_enabled() {
            return Ok(None);
        }
        let MetalLinearWeightBuffers::AffineQuantized {
            packed,
            scales,
            biases,
            out_dim,
            in_dim: weight_in_dim,
            packed_cols,
            group_size,
            bits,
            groups,
        } = weight
        else {
            return Ok(None);
        };
        if in_dim != *weight_in_dim {
            return Err(InferError::Dimension(format!(
                "full-attn o_proj gated x=[1,{in_dim}] rhs=[{out_dim},{weight_in_dim}]"
            )));
        }
        let can_use_u4 = fast_affine_qmv_enabled(*out_dim)
            && *bits == FAST_QMV_BITS
            && *group_size == FAST_QMV_GROUP_SIZE
            && in_dim % 512 == 0;
        let can_use_u8 = full_o_proj_gated_u8_enabled()
            && can_use_fast_affine_qmv_u8_buffers(1, in_dim, *out_dim, *group_size, *bits)
            && *group_size == FAST_QMV_GROUP_SIZE;
        if !can_use_u4 && !can_use_u8 {
            return Ok(None);
        }
        let fast_dims = [
            checked_u32(*out_dim, "full o gated out_dim")?,
            checked_u32(in_dim, "full o gated in_dim")?,
            checked_u32(*packed_cols, "full o gated packed_cols")?,
            checked_u32(*groups, "full o gated groups")?,
        ];
        let pipeline = if can_use_u4 {
            &self.affine_qmv_gated_input_fast_u4_gs64_f32
        } else {
            &self.affine_qmv_gated_input_fast_u8_gs64_f32
        };
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(ctx_buffer), 0);
        encoder.set_buffer(1, Some(gate_buffer), 0);
        encoder.set_buffer(2, Some(packed), 0);
        encoder.set_buffer(3, Some(scales), 0);
        encoder.set_buffer(4, Some(biases), 0);
        encoder.set_buffer(5, Some(output_buffer), 0);
        set_u32_bytes(encoder, 6, &fast_dims, "full_o_gated_dims")?;
        profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(
                1,
                checked_nsuint(out_dim.div_ceil(8), "full o gated out groups")?,
                1,
            ),
            MTLSize::new(64, 1, 1),
        );
        post_dispatch_barrier(encoder);
        Ok(Some(*out_dim))
    }

    pub(super) fn encode_rms_norm_rope_heads(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input_buffer: &BufferRef,
        weight_buffer: &BufferRef,
        output_buffer: &BufferRef,
        spec: PrefillAttentionSpec,
        heads: usize,
    ) -> Result<()> {
        let dims = [
            checked_u32(spec.seq, "rope seq")?,
            checked_u32(heads, "rope heads")?,
            checked_u32(spec.head_dim, "rope head_dim")?,
            checked_u32(spec.rope_dims, "rope dims")?,
        ];
        encoder.set_compute_pipeline_state(&self.rms_norm_rope_heads_f32);
        encoder.set_buffer(0, Some(input_buffer), 0);
        encoder.set_buffer(1, Some(weight_buffer), 0);
        encoder.set_buffer(2, Some(output_buffer), 0);
        set_u32_bytes(encoder, 3, &dims, "rms_rope_dims")?;
        set_f32_bytes(
            encoder,
            4,
            &[
                spec.eps,
                spec.rope_theta,
                spec.rope_frequency_dim as f32,
                0.0,
            ],
            "rms_rope_params",
        )?;
        profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(
                checked_nsuint(heads, "rope heads")?,
                checked_nsuint(spec.seq, "rope seq")?,
                1,
            ),
            MTLSize::new(FULL_ATTN_PREFILL_TG_WIDTH, 1, 1),
        );
        post_dispatch_barrier(encoder);
        Ok(())
    }

    pub(super) fn encode_causal_attention_prefill(
        &self,
        encoder: &ComputeCommandEncoderRef,
        q_buffer: &BufferRef,
        k_buffer: &BufferRef,
        v_buffer: &BufferRef,
        output_buffer: &BufferRef,
        spec: PrefillAttentionSpec,
    ) -> Result<()> {
        if spec.window.is_some() {
            return Err(InferError::Config(
                "prefill résident fenêtré réservé à la phase 4".to_string(),
            ));
        }
        let dims = [
            checked_u32(spec.seq, "attention seq")?,
            checked_u32(spec.q_heads, "attention q_heads")?,
            checked_u32(spec.kv_heads, "attention kv_heads")?,
            checked_u32(spec.head_dim, "attention head_dim")?,
        ];
        if self.encode_steel_causal_attention_prefill(
            encoder,
            q_buffer,
            k_buffer,
            v_buffer,
            output_buffer,
            spec,
        )? {
            return Ok(());
        }
        // Le kernel court `causal_attention_prefill_f32` reste figé pour seq <=
        // 256. La variante mid garde les scores en threadgroup jusqu'à 2048 pour
        // éviter les trois passes de recalcul du long sur les prompts usuels.
        // Au-delà, le long reste le fallback non borné ; il exige head_dim <= 256
        // (une colonne de sortie par thread).
        let (pipeline, tg_width, grid_heads, grid_rows) = if spec.seq <= 256 {
            (
                &self.causal_attention_prefill_f32,
                FULL_ATTN_PREFILL_TG_WIDTH,
                spec.q_heads,
                spec.seq,
            )
        } else if crate::runtime_flags::prefill_attn_batch_mid_30b_enabled()
            && prefill_attn_batch_mid_30b_supported(spec)
        {
            (
                &self.causal_attention_prefill_batch_long_d128_f32,
                FULL_ATTN_PREFILL_BATCH_LONG_TG_WIDTH,
                spec.q_heads,
                spec.seq,
            )
        } else if crate::runtime_flags::prefill_attn_batch_mid_30b_enabled()
            && prefill_attn_batch_mid_35b_supported(spec)
        {
            (
                &self.causal_attention_prefill_batch_gqa8x4_d256_f32,
                FULL_ATTN_PREFILL_TG_WIDTH,
                spec.kv_heads,
                spec.seq.div_ceil(4),
            )
        } else if spec.seq <= 2048 {
            (
                &self.causal_attention_prefill_mid_f32,
                FULL_ATTN_PREFILL_TG_WIDTH,
                spec.q_heads,
                spec.seq,
            )
        } else if crate::runtime_flags::prefill_attn_batch_long_enabled()
            && prefill_attn_batch_long_supported(spec)
        {
            match spec.head_dim {
                128 => (
                    &self.causal_attention_prefill_batch_long_d128_f32,
                    FULL_ATTN_PREFILL_BATCH_LONG_TG_WIDTH,
                    spec.q_heads,
                    spec.seq,
                ),
                256 if prefill_attn_batch_gqa8_d256_supported(spec) => (
                    &self.causal_attention_prefill_batch_gqa8x4_d256_f32,
                    FULL_ATTN_PREFILL_TG_WIDTH,
                    spec.kv_heads,
                    spec.seq.div_ceil(4),
                ),
                256 => (
                    &self.causal_attention_prefill_batch_long_d256_f32,
                    FULL_ATTN_PREFILL_BATCH_LONG_TG_WIDTH,
                    spec.q_heads,
                    spec.seq,
                ),
                _ => {
                    if spec.head_dim > 256 {
                        return Err(InferError::Dimension(format!(
                            "prefill causal résident seq={} head_dim={} non supporté (head_dim > 256)",
                            spec.seq, spec.head_dim
                        )));
                    }
                    (
                        &self.causal_attention_prefill_long_f32,
                        FULL_ATTN_PREFILL_TG_WIDTH,
                        spec.q_heads,
                        spec.seq,
                    )
                }
            }
        } else {
            if spec.head_dim > 256 {
                return Err(InferError::Dimension(format!(
                    "prefill causal résident seq={} head_dim={} non supporté (head_dim > 256)",
                    spec.seq, spec.head_dim
                )));
            }
            (
                &self.causal_attention_prefill_long_f32,
                FULL_ATTN_PREFILL_TG_WIDTH,
                spec.q_heads,
                spec.seq,
            )
        };
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(q_buffer), 0);
        encoder.set_buffer(1, Some(k_buffer), 0);
        encoder.set_buffer(2, Some(v_buffer), 0);
        encoder.set_buffer(3, Some(output_buffer), 0);
        set_u32_bytes(encoder, 4, &dims, "causal_prefill_dims")?;
        set_f32_bytes(
            encoder,
            5,
            &causal_prefill_scale_params(spec)?,
            "causal_prefill_scale_params",
        )?;
        profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(
                checked_nsuint(grid_heads, "attention grid_heads")?,
                checked_nsuint(grid_rows, "attention grid_rows")?,
                1,
            ),
            MTLSize::new(tg_width, 1, 1),
        );
        post_dispatch_barrier(encoder);
        Ok(())
    }

    /// Encode l'attention prefill causale bornée par une fenêtre glissante.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si la fenêtre ou les dimensions dépassent les kernels.
    pub(super) fn encode_windowed_attention_prefill(
        &self,
        encoder: &ComputeCommandEncoderRef,
        q_buffer: &BufferRef,
        k_buffer: &BufferRef,
        v_buffer: &BufferRef,
        output_buffer: &BufferRef,
        spec: PrefillAttentionSpec,
    ) -> Result<()> {
        let window = spec.window.ok_or_else(|| {
            InferError::Config("prefill fenêtré sans taille de fenêtre".to_string())
        })?;
        if window == 0 {
            return Err(InferError::Dimension(
                "prefill fenêtré avec fenêtre vide".to_string(),
            ));
        }
        let dims = [
            checked_u32(spec.seq, "attention fenêtrée seq")?,
            checked_u32(spec.q_heads, "attention fenêtrée q_heads")?,
            checked_u32(spec.kv_heads, "attention fenêtrée kv_heads")?,
            checked_u32(spec.head_dim, "attention fenêtrée head_dim")?,
            checked_u32(window, "attention fenêtrée window")?,
        ];
        let pipeline = if spec.seq <= 256 {
            &self.windowed_attention_prefill_f32
        } else if spec.seq <= 2048 {
            &self.windowed_attention_prefill_mid_f32
        } else {
            if spec.head_dim > 256 {
                return Err(InferError::Dimension(format!(
                    "prefill fenêtré résident seq={} head_dim={} non supporté (head_dim > 256)",
                    spec.seq, spec.head_dim
                )));
            }
            &self.windowed_attention_prefill_long_f32
        };
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(q_buffer), 0);
        encoder.set_buffer(1, Some(k_buffer), 0);
        encoder.set_buffer(2, Some(v_buffer), 0);
        encoder.set_buffer(3, Some(output_buffer), 0);
        set_u32_bytes(encoder, 4, &dims, "windowed_prefill_dims")?;
        set_f32_bytes(
            encoder,
            5,
            &causal_prefill_scale_params(spec)?,
            "windowed_prefill_scale_params",
        )?;
        profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(
                checked_nsuint(spec.q_heads, "attention fenêtrée grid_heads")?,
                checked_nsuint(spec.seq, "attention fenêtrée grid_rows")?,
                1,
            ),
            MTLSize::new(FULL_ATTN_PREFILL_TG_WIDTH, 1, 1),
        );
        post_dispatch_barrier(encoder);
        Ok(())
    }

    fn encode_steel_causal_attention_prefill(
        &self,
        encoder: &ComputeCommandEncoderRef,
        q_buffer: &BufferRef,
        k_buffer: &BufferRef,
        v_buffer: &BufferRef,
        output_buffer: &BufferRef,
        spec: PrefillAttentionSpec,
    ) -> Result<bool> {
        if !crate::runtime_flags::prefill_attn_steel_d256_enabled()
            || !prefill_attn_steel_d256_supported(spec)
        {
            return Ok(false);
        }
        let Some(pipeline) = self.causal_attention_prefill_steel_d256_f32.as_ref() else {
            return Ok(false);
        };
        let params = steel_attn_params(
            spec,
            STEEL_CAUSAL_ATTN_D256_BQ,
            STEEL_CAUSAL_ATTN_D256_BK,
            "steel causal d256 attention",
        )?;
        let grid_rows = spec.seq.div_ceil(STEEL_CAUSAL_ATTN_D256_BQ);
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(q_buffer), 0);
        encoder.set_buffer(1, Some(k_buffer), 0);
        encoder.set_buffer(2, Some(v_buffer), 0);
        encoder.set_buffer(3, Some(output_buffer), 0);
        encoder.set_bytes(
            4,
            std::mem::size_of::<SteelAttnParams>() as NSUInteger,
            (&params as *const SteelAttnParams).cast::<c_void>(),
        );
        profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(
                checked_nsuint(grid_rows, "steel causal d256 NQ")?,
                checked_nsuint(spec.q_heads, "steel causal d256 heads")?,
                1,
            ),
            MTLSize::new(32, 4, 1),
        );
        post_dispatch_barrier(encoder);
        Ok(true)
    }

    pub(super) fn encode_noncausal_attention_prefill(
        &self,
        encoder: &ComputeCommandEncoderRef,
        q_buffer: &BufferRef,
        k_buffer: &BufferRef,
        v_buffer: &BufferRef,
        output_buffer: &BufferRef,
        spec: PrefillAttentionSpec,
    ) -> Result<()> {
        if self.encode_steel_noncausal_attention_prefill(
            encoder,
            q_buffer,
            k_buffer,
            v_buffer,
            output_buffer,
            spec,
        )? {
            return Ok(());
        }
        let dims = [
            checked_u32(spec.seq, "attention seq")?,
            checked_u32(spec.q_heads, "attention q_heads")?,
            checked_u32(spec.kv_heads, "attention kv_heads")?,
            checked_u32(spec.head_dim, "attention head_dim")?,
        ];
        encoder.set_compute_pipeline_state(&self.noncausal_attention_prefill_f32);
        encoder.set_buffer(0, Some(q_buffer), 0);
        encoder.set_buffer(1, Some(k_buffer), 0);
        encoder.set_buffer(2, Some(v_buffer), 0);
        encoder.set_buffer(3, Some(output_buffer), 0);
        set_u32_bytes(encoder, 4, &dims, "noncausal_prefill_dims")?;
        profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(
                checked_nsuint(spec.q_heads, "attention q_heads")?,
                checked_nsuint(spec.seq, "attention seq")?,
                1,
            ),
            MTLSize::new(FULL_ATTN_PREFILL_TG_WIDTH, 1, 1),
        );
        post_dispatch_barrier(encoder);
        Ok(())
    }

    fn encode_steel_noncausal_attention_prefill(
        &self,
        encoder: &ComputeCommandEncoderRef,
        q_buffer: &BufferRef,
        k_buffer: &BufferRef,
        v_buffer: &BufferRef,
        output_buffer: &BufferRef,
        spec: PrefillAttentionSpec,
    ) -> Result<bool> {
        let Some(pipeline) = self.steel_attention_f32_bq32_bk32_bd64.as_ref() else {
            return Ok(false);
        };
        if spec.head_dim != STEEL_ATTN_BD {
            return Ok(false);
        }
        let n_q = spec.seq.div_ceil(STEEL_ATTN_BQ);
        let params = steel_attn_params(spec, STEEL_ATTN_BQ, STEEL_ATTN_BK, "steel attention")?;
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(q_buffer), 0);
        encoder.set_buffer(1, Some(k_buffer), 0);
        encoder.set_buffer(2, Some(v_buffer), 0);
        encoder.set_buffer(3, Some(output_buffer), 0);
        encoder.set_bytes(
            4,
            std::mem::size_of::<SteelAttnParams>() as NSUInteger,
            (&params as *const SteelAttnParams).cast::<c_void>(),
        );
        profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(
                checked_nsuint(n_q, "steel attention NQ")?,
                checked_nsuint(spec.q_heads, "steel attention heads")?,
                1,
            ),
            MTLSize::new(32, 4, 1),
        );
        post_dispatch_barrier(encoder);
        Ok(true)
    }
}

fn checked_i32(value: usize, label: &'static str) -> Result<i32> {
    i32::try_from(value)
        .map_err(|_| InferError::Dimension(format!("{label}={value} depasse la capacite i32")))
}

fn checked_i64(value: usize) -> Result<i64> {
    i64::try_from(value)
        .map_err(|_| InferError::Dimension(format!("{value} depasse la capacite i64")))
}

pub(super) fn steel_attn_params(
    spec: PrefillAttentionSpec,
    block_q: usize,
    block_k: usize,
    label: &'static str,
) -> Result<SteelAttnParams> {
    if spec.kv_heads == 0 || spec.q_heads == 0 || spec.q_heads % spec.kv_heads != 0 {
        return Err(InferError::Dimension(format!(
            "{label} heads invalides q_heads={} kv_heads={}",
            spec.q_heads, spec.kv_heads
        )));
    }
    let q_dim = checked_len(spec.q_heads, spec.head_dim, "steel attention q_dim")?;
    let kv_dim = checked_len(spec.kv_heads, spec.head_dim, "steel attention kv_dim")?;
    let n_q = spec.seq.div_ceil(block_q);
    let n_k = spec.seq.div_ceil(block_k);
    let n_q_aligned = spec.seq / block_q;
    let n_k_aligned = spec.seq / block_k;
    let scale = causal_prefill_scale_params(spec)?[0];
    Ok(SteelAttnParams {
        b: 1,
        h: checked_i32(spec.q_heads, "steel attention heads")?,
        d: checked_i32(spec.head_dim, "steel attention head_dim")?,
        q_l: checked_i32(spec.seq, "steel attention qL")?,
        k_l: checked_i32(spec.seq, "steel attention kL")?,
        gqa_factor: checked_i32(spec.q_heads / spec.kv_heads, "steel attention gqa")?,
        scale,
        n_q: checked_i32(n_q, label)?,
        n_k: checked_i32(n_k, label)?,
        n_q_aligned: checked_i32(n_q_aligned, label)?,
        n_k_aligned: checked_i32(n_k_aligned, label)?,
        q_l_rem: checked_i32(spec.seq - n_q_aligned * block_q, label)?,
        k_l_rem: checked_i32(spec.seq - n_k_aligned * block_k, label)?,
        q_l_off: 0,
        q_strides: [
            checked_i64(checked_len(
                spec.seq,
                q_dim,
                "steel attention Q batch stride",
            )?)?,
            checked_i64(spec.head_dim)?,
            checked_i64(q_dim)?,
        ],
        k_strides: [
            checked_i64(checked_len(
                spec.seq,
                kv_dim,
                "steel attention K batch stride",
            )?)?,
            checked_i64(spec.head_dim)?,
            checked_i64(kv_dim)?,
        ],
        v_strides: [
            checked_i64(checked_len(
                spec.seq,
                kv_dim,
                "steel attention V batch stride",
            )?)?,
            checked_i64(spec.head_dim)?,
            checked_i64(kv_dim)?,
        ],
        o_strides: [
            checked_i64(checked_len(
                spec.seq,
                q_dim,
                "steel attention O batch stride",
            )?)?,
            checked_i64(spec.head_dim)?,
            checked_i64(q_dim)?,
        ],
    })
}

pub(super) fn causal_prefill_scale_params(spec: PrefillAttentionSpec) -> Result<[f32; 2]> {
    if !spec.attn_scalar.is_finite() || spec.attn_scalar <= 0.0 {
        return Err(InferError::Dimension(format!(
            "attention attn_scalar={} invalide",
            spec.attn_scalar
        )));
    }
    let custom = f32::from(spec.attn_scalar != spec.head_dim as f32);
    Ok([spec.attn_scalar.sqrt().recip(), custom])
}
