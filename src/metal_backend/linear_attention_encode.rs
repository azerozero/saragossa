//! Kernels élémentaires Metal de linear-attention.

use super::*;

const LINEAR_ATTN_TG_WIDTH: u64 = 32;

#[expect(
    clippy::too_many_arguments,
    reason = "wrappers d'encodage Metal: buffers, dimensions et offsets restent explicites"
)]
impl MetalExecutor {
    pub(super) fn encode_linear_attn_conv(
        &self,
        encoder: &ComputeCommandEncoderRef,
        qkv_buffer: &BufferRef,
        conv_weight_buffer: &BufferRef,
        conv_state_buffer: &BufferRef,
        conv_out_buffer: &BufferRef,
        conv_dim: usize,
        kernel: usize,
    ) -> Result<()> {
        self.encode_linear_attn_conv_with_offset(
            encoder,
            qkv_buffer,
            0,
            conv_weight_buffer,
            conv_state_buffer,
            conv_out_buffer,
            conv_dim,
            kernel,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "signature d'encodage Metal: buffers + offset + dimensions"
    )]
    pub(super) fn encode_linear_attn_conv_with_offset(
        &self,
        encoder: &ComputeCommandEncoderRef,
        qkv_buffer: &BufferRef,
        qkv_offset: u64,
        conv_weight_buffer: &BufferRef,
        conv_state_buffer: &BufferRef,
        conv_out_buffer: &BufferRef,
        conv_dim: usize,
        kernel: usize,
    ) -> Result<()> {
        self.encode_linear_attn_conv_with_offsets(
            encoder,
            qkv_buffer,
            qkv_offset,
            conv_weight_buffer,
            conv_state_buffer,
            conv_out_buffer,
            0,
            conv_dim,
            kernel,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "signature d'encodage Metal: buffers + offsets + dimensions"
    )]
    pub(super) fn encode_linear_attn_conv_with_offsets(
        &self,
        encoder: &ComputeCommandEncoderRef,
        qkv_buffer: &BufferRef,
        qkv_offset: u64,
        conv_weight_buffer: &BufferRef,
        conv_state_buffer: &BufferRef,
        conv_out_buffer: &BufferRef,
        conv_out_offset: u64,
        conv_dim: usize,
        kernel: usize,
    ) -> Result<()> {
        let dims = [
            checked_u32(conv_dim, "linear-attn conv_dim")?,
            checked_u32(kernel, "linear-attn conv kernel")?,
        ];
        let use_k4 = linear_conv_k4_enabled() && kernel == 4;
        let pipeline = if use_k4 {
            &self.linear_attn_conv_silu_k4_f32
        } else {
            &self.linear_attn_conv_silu_f32
        };
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(qkv_buffer), qkv_offset);
        encoder.set_buffer(1, Some(conv_weight_buffer), 0);
        encoder.set_buffer(2, Some(conv_state_buffer), 0);
        encoder.set_buffer(3, Some(conv_out_buffer), conv_out_offset);
        set_u32_bytes(encoder, 4, &dims, "linear_attn_conv_dims")?;
        self.dispatch_1d(encoder, pipeline, conv_dim)
    }

    pub(super) fn encode_linear_attn_norm_gates(
        &self,
        encoder: &ComputeCommandEncoderRef,
        conv_out_buffer: &BufferRef,
        beta_input_buffer: &BufferRef,
        gate_input_buffer: &BufferRef,
        a_log_buffer: &BufferRef,
        dt_bias_buffer: &BufferRef,
        q_norm_buffer: &BufferRef,
        k_norm_buffer: &BufferRef,
        beta_buffer: &BufferRef,
        decay_buffer: &BufferRef,
        spec: LinearAttentionStepSpec,
    ) -> Result<()> {
        self.encode_linear_attn_norm_gates_with_offsets(
            encoder,
            conv_out_buffer,
            beta_input_buffer,
            0,
            gate_input_buffer,
            0,
            a_log_buffer,
            dt_bias_buffer,
            q_norm_buffer,
            k_norm_buffer,
            beta_buffer,
            decay_buffer,
            spec,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "signature d'encodage Metal: buffers + offsets + dimensions"
    )]
    pub(super) fn encode_linear_attn_norm_gates_with_offsets(
        &self,
        encoder: &ComputeCommandEncoderRef,
        conv_out_buffer: &BufferRef,
        beta_input_buffer: &BufferRef,
        beta_input_offset: u64,
        gate_input_buffer: &BufferRef,
        gate_input_offset: u64,
        a_log_buffer: &BufferRef,
        dt_bias_buffer: &BufferRef,
        q_norm_buffer: &BufferRef,
        k_norm_buffer: &BufferRef,
        beta_buffer: &BufferRef,
        decay_buffer: &BufferRef,
        spec: LinearAttentionStepSpec,
    ) -> Result<()> {
        self.encode_linear_attn_norm_gates_with_all_offsets(
            encoder,
            conv_out_buffer,
            0,
            beta_input_buffer,
            beta_input_offset,
            gate_input_buffer,
            gate_input_offset,
            a_log_buffer,
            dt_bias_buffer,
            q_norm_buffer,
            0,
            k_norm_buffer,
            0,
            beta_buffer,
            0,
            decay_buffer,
            0,
            spec,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "signature d'encodage Metal: buffers + offsets + dimensions"
    )]
    pub(super) fn encode_linear_attn_norm_gates_with_all_offsets(
        &self,
        encoder: &ComputeCommandEncoderRef,
        conv_out_buffer: &BufferRef,
        conv_out_offset: u64,
        beta_input_buffer: &BufferRef,
        beta_input_offset: u64,
        gate_input_buffer: &BufferRef,
        gate_input_offset: u64,
        a_log_buffer: &BufferRef,
        dt_bias_buffer: &BufferRef,
        q_norm_buffer: &BufferRef,
        q_norm_offset: u64,
        k_norm_buffer: &BufferRef,
        k_norm_offset: u64,
        beta_buffer: &BufferRef,
        beta_offset: u64,
        decay_buffer: &BufferRef,
        decay_offset: u64,
        spec: LinearAttentionStepSpec,
    ) -> Result<()> {
        let dims = [
            checked_u32(spec.num_key_heads, "linear-attn key heads")?,
            checked_u32(spec.num_value_heads, "linear-attn value heads")?,
            checked_u32(spec.key_head_dim, "linear-attn key head dim")?,
            checked_u32(spec.value_head_dim, "linear-attn value head dim")?,
        ];
        let inv = (spec.key_head_dim as f32).powf(-0.5);
        let scales = [inv * inv, inv];
        let groups = spec.num_key_heads.max(spec.num_value_heads);
        let use_dk128 = linear_norm_dk128_enabled() && spec.key_head_dim == 128;
        let pipeline = if use_dk128 {
            &self.linear_attn_norm_gates_dk128_f32
        } else {
            &self.linear_attn_norm_gates_f32
        };
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(conv_out_buffer), conv_out_offset);
        encoder.set_buffer(1, Some(beta_input_buffer), beta_input_offset);
        encoder.set_buffer(2, Some(gate_input_buffer), gate_input_offset);
        encoder.set_buffer(3, Some(a_log_buffer), 0);
        encoder.set_buffer(4, Some(dt_bias_buffer), 0);
        encoder.set_buffer(5, Some(q_norm_buffer), q_norm_offset);
        encoder.set_buffer(6, Some(k_norm_buffer), k_norm_offset);
        encoder.set_buffer(7, Some(beta_buffer), beta_offset);
        encoder.set_buffer(8, Some(decay_buffer), decay_offset);
        set_u32_bytes(encoder, 9, &dims, "linear_attn_norm_dims")?;
        set_f32_bytes(encoder, 10, &scales, "linear_attn_norm_scales")?;
        profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(checked_nsuint(groups, "linear-attn norm groups")?, 1, 1),
            MTLSize::new(LINEAR_ATTN_TG_WIDTH, 1, 1),
        );
        post_dispatch_barrier(encoder);
        Ok(())
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "signature d'encodage Metal: scan sequence + buffers explicites"
    )]
    pub(super) fn encode_linear_attn_gated_delta_seq_dk128(
        &self,
        encoder: &ComputeCommandEncoderRef,
        conv_out_buffer: &BufferRef,
        q_norm_buffer: &BufferRef,
        k_norm_buffer: &BufferRef,
        beta_buffer: &BufferRef,
        decay_buffer: &BufferRef,
        ssm_state_buffer: &BufferRef,
        ssm_bf16: bool,
        y_buffer: &BufferRef,
        steps: usize,
        spec: LinearAttentionStepSpec,
    ) -> Result<()> {
        if steps == 0 {
            return Err(InferError::Dimension(
                "linear-attn seq steps vide".to_string(),
            ));
        }
        if spec.key_head_dim != 128 {
            return Err(InferError::Dimension(format!(
                "linear-attn seq attend key_head_dim=128, reçu {}",
                spec.key_head_dim
            )));
        }
        let repeat = spec
            .num_value_heads
            .checked_div(spec.num_key_heads)
            .ok_or_else(|| InferError::Metal("linear-attn seq repeat nul".to_string()))?;
        let dims = [
            checked_u32(spec.num_value_heads, "linear-attn seq value heads")?,
            checked_u32(spec.value_head_dim, "linear-attn seq value dim")?,
            checked_u32(spec.key_head_dim, "linear-attn seq key dim")?,
            checked_u32(repeat, "linear-attn seq repeat")?,
        ];
        let pipeline = if ssm_bf16 {
            &self.linear_attn_gated_delta_seq_dk128_bf16_tg4_f32
        } else {
            &self.linear_attn_gated_delta_seq_dk128_tg4_f32
        };
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(conv_out_buffer), 0);
        encoder.set_buffer(1, Some(q_norm_buffer), 0);
        encoder.set_buffer(2, Some(k_norm_buffer), 0);
        encoder.set_buffer(3, Some(beta_buffer), 0);
        encoder.set_buffer(4, Some(decay_buffer), 0);
        encoder.set_buffer(5, Some(ssm_state_buffer), 0);
        encoder.set_buffer(6, Some(y_buffer), 0);
        set_u32_bytes(encoder, 7, &dims, "linear_attn_seq_delta_dims")?;
        set_u32_bytes(
            encoder,
            8,
            &[checked_u32(steps, "linear-attn seq steps")?],
            "linear_attn_seq_delta_steps",
        )?;
        profile_dispatch();
        let value_head_dim = checked_nsuint(spec.value_head_dim, "linear-attn seq value dim")?;
        let tg_rows = linear_delta_tg_rows().min(value_head_dim);
        encoder.dispatch_threads(
            MTLSize::new(
                LINEAR_ATTN_TG_WIDTH,
                value_head_dim,
                checked_nsuint(spec.num_value_heads, "linear-attn seq value heads")?,
            ),
            MTLSize::new(LINEAR_ATTN_TG_WIDTH, tg_rows, 1),
        );
        post_dispatch_barrier(encoder);
        Ok(())
    }

    /// Linear-attn gated-delta en forme CHUNKÉE (port chunked-DeltaNet brique 4).
    /// 1 threadgroup/value_head, boucle sur T/C chunks (16× moins d'étapes séquentielles).
    /// Exige `value_head_dim == 128` et `key_head_dim == 128`. Opt-in `RETI_RUST_LINEAR_CHUNKED`.
    ///
    /// Correspondance étapes du kernel ↔ équations (section « Forme chunkée » de la
    /// doc de module `crate::linear_attention`, chunk C=16, `S₀` = état d'entrée) :
    /// - boucle `gamma[i]` (tid 0) : decays cumulés intra-chunk `γ_i = ∏_{j≤i} g_j` ;
    /// - 1ʳᵉ passe par colonne `c` : `u[i]  = β_i·(v_i − γ_i·(S₀·k_i))` (delta contre S₀
    ///   décayé) et `qs[i] = S₀·q_i` (part de lecture venant de l'état d'entrée) ;
    /// - remplissage `A`/`P` (strided sur les threads) : couplages intra-chunk
    ///   `A_ij = β_i·(γ_i/γ_j)·(k_i·k_j)` (j<i) et `P_ij = (γ_i/γ_j)·(q_i·k_j)` (j≤i) ;
    /// - substitution avant : `Δ_i = u_i − Σ_{j<i} A_ij·Δ_j` (résout (I+A)·Δ = u,
    ///   équivalent au déroulé séquentiel de la delta rule) ;
    /// - sortie : `y_i = γ_i·qs_i + Σ_{j≤i} P_ij·Δ_j` ;
    /// - fin de chunk : `S ← γ_last·S₀ + Σ_j (γ_last/γ_j)·Δ_j·k_jᵀ` (état repris par
    ///   le chunk suivant, GQA : chaque tête de valeur lit q/k de sa tête clé).
    ///
    /// Oracle CPU naïf token-par-token : `naive_gdn_reference` dans
    /// `linear_attention/tests.rs`, comparé ici par
    /// `chunk_delta_seq_layout_gqa_matches_naive_oracle` (metal_backend/tests.rs).
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si dims non supportées ou NA indisponible.
    #[expect(
        clippy::too_many_arguments,
        reason = "wrapper d'encodage : buffers explicites"
    )]
    pub(super) fn encode_chunk_delta_seq_layout(
        &self,
        encoder: &ComputeCommandEncoderRef,
        conv_out_buffer: &BufferRef,
        q_norm_buffer: &BufferRef,
        k_norm_buffer: &BufferRef,
        beta_buffer: &BufferRef,
        decay_buffer: &BufferRef,
        ssm_state_buffer: &BufferRef,
        y_buffer: &BufferRef,
        steps: usize,
        spec: LinearAttentionStepSpec,
    ) -> Result<()> {
        if spec.key_head_dim != 128 || spec.value_head_dim != 128 {
            return Err(InferError::Dimension(format!(
                "chunk-delta layout exige key/value_head_dim=128, reçu k={} v={}",
                spec.key_head_dim, spec.value_head_dim
            )));
        }
        let repeat = spec
            .num_value_heads
            .checked_div(spec.num_key_heads)
            .ok_or_else(|| InferError::Metal("chunk-delta repeat nul".to_string()))?;
        // Préfère la variante TENSOR-CORES (KS/QS+state-update sur simdgroup_matrix,
        // ~3x mono-tête) ; repli sur le scalaire si indisponible.
        let pso = self
            .chunk_delta_seq_layout_tc
            .as_ref()
            .or(self.chunk_delta_seq_layout.as_ref())
            .ok_or_else(|| InferError::Config("chunk_delta_seq_layout indisponible".into()))?;
        let dims = [
            checked_u32(spec.num_value_heads, "chunk-delta vheads")?,
            checked_u32(spec.value_head_dim, "chunk-delta vhd")?,
            checked_u32(repeat, "chunk-delta repeat")?,
            checked_u32(steps, "chunk-delta steps")?,
        ];
        encoder.set_compute_pipeline_state(pso);
        encoder.set_buffer(0, Some(conv_out_buffer), 0);
        encoder.set_buffer(1, Some(q_norm_buffer), 0);
        encoder.set_buffer(2, Some(k_norm_buffer), 0);
        encoder.set_buffer(3, Some(beta_buffer), 0);
        encoder.set_buffer(4, Some(decay_buffer), 0);
        encoder.set_buffer(5, Some(ssm_state_buffer), 0);
        encoder.set_buffer(6, Some(y_buffer), 0);
        set_u32_bytes(encoder, 7, &dims, "chunk_delta_layout_dims")?;
        profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(
                checked_nsuint(spec.num_value_heads, "chunk-delta vheads grid")?,
                1,
                1,
            ),
            MTLSize::new(
                checked_nsuint(spec.value_head_dim, "chunk-delta vhd tg")?,
                1,
                1,
            ),
        );
        post_dispatch_barrier(encoder);
        Ok(())
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "fusion conv+norm linear-attn: tous les buffers du couple restent explicites"
    )]
    pub(super) fn encode_linear_attn_conv_norm_gates_k4_dk128_with_offsets(
        &self,
        encoder: &ComputeCommandEncoderRef,
        qkv_buffer: &BufferRef,
        qkv_offset: u64,
        beta_input_buffer: &BufferRef,
        beta_input_offset: u64,
        gate_input_buffer: &BufferRef,
        gate_input_offset: u64,
        conv_weight_buffer: &BufferRef,
        conv_state_buffer: &BufferRef,
        a_log_buffer: &BufferRef,
        dt_bias_buffer: &BufferRef,
        conv_out_buffer: &BufferRef,
        q_norm_buffer: &BufferRef,
        k_norm_buffer: &BufferRef,
        beta_buffer: &BufferRef,
        decay_buffer: &BufferRef,
        spec: LinearAttentionStepSpec,
    ) -> Result<()> {
        self.encode_linear_attn_conv_norm_gates_k4_dk128_with_all_offsets(
            encoder,
            qkv_buffer,
            qkv_offset,
            beta_input_buffer,
            beta_input_offset,
            gate_input_buffer,
            gate_input_offset,
            conv_weight_buffer,
            conv_state_buffer,
            a_log_buffer,
            dt_bias_buffer,
            conv_out_buffer,
            0,
            q_norm_buffer,
            0,
            k_norm_buffer,
            0,
            beta_buffer,
            0,
            decay_buffer,
            0,
            spec,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "fusion conv+norm linear-attn: buffers et offsets batched explicites"
    )]
    pub(super) fn encode_linear_attn_conv_norm_gates_k4_dk128_with_all_offsets(
        &self,
        encoder: &ComputeCommandEncoderRef,
        qkv_buffer: &BufferRef,
        qkv_offset: u64,
        beta_input_buffer: &BufferRef,
        beta_input_offset: u64,
        gate_input_buffer: &BufferRef,
        gate_input_offset: u64,
        conv_weight_buffer: &BufferRef,
        conv_state_buffer: &BufferRef,
        a_log_buffer: &BufferRef,
        dt_bias_buffer: &BufferRef,
        conv_out_buffer: &BufferRef,
        conv_out_offset: u64,
        q_norm_buffer: &BufferRef,
        q_norm_offset: u64,
        k_norm_buffer: &BufferRef,
        k_norm_offset: u64,
        beta_buffer: &BufferRef,
        beta_offset: u64,
        decay_buffer: &BufferRef,
        decay_offset: u64,
        spec: LinearAttentionStepSpec,
    ) -> Result<()> {
        let dims = [
            checked_u32(spec.num_key_heads, "linear-attn fused key heads")?,
            checked_u32(spec.num_value_heads, "linear-attn fused value heads")?,
            checked_u32(spec.key_head_dim, "linear-attn fused key dim")?,
            checked_u32(spec.value_head_dim, "linear-attn fused value dim")?,
        ];
        let inv = (spec.key_head_dim as f32).powf(-0.5);
        let scales = [inv * inv, inv];
        let groups = spec.num_key_heads.max(spec.num_value_heads);
        encoder.set_compute_pipeline_state(&self.linear_attn_conv_norm_gates_k4_dk128_f32);
        encoder.set_buffer(0, Some(qkv_buffer), qkv_offset);
        encoder.set_buffer(1, Some(beta_input_buffer), beta_input_offset);
        encoder.set_buffer(2, Some(gate_input_buffer), gate_input_offset);
        encoder.set_buffer(3, Some(conv_weight_buffer), 0);
        encoder.set_buffer(4, Some(conv_state_buffer), 0);
        encoder.set_buffer(5, Some(a_log_buffer), 0);
        encoder.set_buffer(6, Some(dt_bias_buffer), 0);
        encoder.set_buffer(7, Some(conv_out_buffer), conv_out_offset);
        encoder.set_buffer(8, Some(q_norm_buffer), q_norm_offset);
        encoder.set_buffer(9, Some(k_norm_buffer), k_norm_offset);
        encoder.set_buffer(10, Some(beta_buffer), beta_offset);
        encoder.set_buffer(11, Some(decay_buffer), decay_offset);
        set_u32_bytes(encoder, 12, &dims, "linear_attn_fused_conv_norm_dims")?;
        set_f32_bytes(encoder, 13, &scales, "linear_attn_fused_conv_norm_scales")?;
        profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(
                checked_nsuint(groups, "linear-attn fused conv norm groups")?,
                1,
                1,
            ),
            MTLSize::new(LINEAR_ATTN_TG_WIDTH, 1, 1),
        );
        post_dispatch_barrier(encoder);
        Ok(())
    }

    /// Brick #8 : conv+norm+gates BATCHÉ (1 dispatch grid `groups×batch` + finalize de
    /// conv_state) au lieu de la boucle per-token. Buffers contigus `[batch, *]`. La conv
    /// lit conv_state initial (taps p<0), puis `finalize` le met aux 3 derniers tokens.
    /// Sortie byte-identique au per-token. value_head_dim=key_head_dim=128.
    #[expect(
        clippy::too_many_arguments,
        reason = "signature d'encodage Metal: buffers + dimensions"
    )]
    pub(super) fn encode_linear_attn_conv_norm_gates_k4_dk128_batch(
        &self,
        encoder: &ComputeCommandEncoderRef,
        qkv_buffer: &BufferRef,
        beta_input_buffer: &BufferRef,
        gate_input_buffer: &BufferRef,
        conv_weight_buffer: &BufferRef,
        conv_state_buffer: &BufferRef,
        a_log_buffer: &BufferRef,
        dt_bias_buffer: &BufferRef,
        conv_out_buffer: &BufferRef,
        q_norm_buffer: &BufferRef,
        k_norm_buffer: &BufferRef,
        beta_buffer: &BufferRef,
        decay_buffer: &BufferRef,
        batch: usize,
        spec: LinearAttentionStepSpec,
    ) -> Result<()> {
        let dims = [
            checked_u32(spec.num_key_heads, "linear-attn batch key heads")?,
            checked_u32(spec.num_value_heads, "linear-attn batch value heads")?,
            checked_u32(spec.key_head_dim, "linear-attn batch key dim")?,
            checked_u32(spec.value_head_dim, "linear-attn batch value dim")?,
        ];
        let inv = (spec.key_head_dim as f32).powf(-0.5);
        let scales = [inv * inv, inv];
        let groups = spec.num_key_heads.max(spec.num_value_heads);
        let key_dim = checked_len(spec.num_key_heads, spec.key_head_dim, "lin batch key_dim")?;
        let value_dim = checked_len(
            spec.num_value_heads,
            spec.value_head_dim,
            "lin batch value_dim",
        )?;
        let conv_dim = 2 * key_dim + value_dim;
        let batch_u32 = checked_u32(batch, "linear-attn batch")?;
        encoder.set_compute_pipeline_state(&self.linear_attn_conv_norm_gates_k4_dk128_batch_f32);
        encoder.set_buffer(0, Some(qkv_buffer), 0);
        encoder.set_buffer(1, Some(beta_input_buffer), 0);
        encoder.set_buffer(2, Some(gate_input_buffer), 0);
        encoder.set_buffer(3, Some(conv_weight_buffer), 0);
        encoder.set_buffer(4, Some(conv_state_buffer), 0);
        encoder.set_buffer(5, Some(a_log_buffer), 0);
        encoder.set_buffer(6, Some(dt_bias_buffer), 0);
        encoder.set_buffer(7, Some(conv_out_buffer), 0);
        encoder.set_buffer(8, Some(q_norm_buffer), 0);
        encoder.set_buffer(9, Some(k_norm_buffer), 0);
        encoder.set_buffer(10, Some(beta_buffer), 0);
        encoder.set_buffer(11, Some(decay_buffer), 0);
        set_u32_bytes(encoder, 12, &dims, "linear_attn_batch_conv_norm_dims")?;
        set_f32_bytes(encoder, 13, &scales, "linear_attn_batch_conv_norm_scales")?;
        set_u32_bytes(
            encoder,
            14,
            &[batch_u32],
            "linear_attn_batch_conv_norm_batch",
        )?;
        profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(
                checked_nsuint(groups, "linear-attn batch conv norm groups")?,
                checked_nsuint(batch, "linear-attn batch conv norm grid")?,
                1,
            ),
            MTLSize::new(LINEAR_ATTN_TG_WIDTH, 1, 1),
        );
        // Barrière : la conv a lu conv_state initial ; finalize le met aux 3 derniers tokens.
        memory_barrier_buffers(encoder);
        let fin_dims = [
            checked_u32(conv_dim, "linear-attn finalize conv_dim")?,
            batch_u32,
        ];
        encoder.set_compute_pipeline_state(&self.linear_attn_conv_state_finalize_f32);
        encoder.set_buffer(0, Some(qkv_buffer), 0);
        encoder.set_buffer(1, Some(conv_state_buffer), 0);
        set_u32_bytes(encoder, 2, &fin_dims, "linear_attn_finalize_dims")?;
        profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(
                checked_nsuint(conv_dim.div_ceil(64), "lin finalize groups")?,
                1,
                1,
            ),
            MTLSize::new(64, 1, 1),
        );
        post_dispatch_barrier(encoder);
        Ok(())
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "signature d'encodage Metal: buffers + offsets + dimensions"
    )]
    pub(super) fn encode_linear_attn_norm_gates_inv_dk128_with_offsets(
        &self,
        encoder: &ComputeCommandEncoderRef,
        conv_out_buffer: &BufferRef,
        beta_input_buffer: &BufferRef,
        beta_input_offset: u64,
        gate_input_buffer: &BufferRef,
        gate_input_offset: u64,
        a_log_buffer: &BufferRef,
        dt_bias_buffer: &BufferRef,
        q_inv_buffer: &BufferRef,
        k_inv_buffer: &BufferRef,
        beta_buffer: &BufferRef,
        decay_buffer: &BufferRef,
        spec: LinearAttentionStepSpec,
    ) -> Result<()> {
        let dims = [
            checked_u32(spec.num_key_heads, "linear-attn inv key heads")?,
            checked_u32(spec.num_value_heads, "linear-attn inv value heads")?,
            checked_u32(spec.key_head_dim, "linear-attn inv key dim")?,
            checked_u32(spec.value_head_dim, "linear-attn inv value dim")?,
        ];
        let inv = (spec.key_head_dim as f32).powf(-0.5);
        let scales = [inv * inv, inv];
        let groups = spec.num_key_heads.max(spec.num_value_heads);
        encoder.set_compute_pipeline_state(&self.linear_attn_norm_gates_inv_dk128_f32);
        encoder.set_buffer(0, Some(conv_out_buffer), 0);
        encoder.set_buffer(1, Some(beta_input_buffer), beta_input_offset);
        encoder.set_buffer(2, Some(gate_input_buffer), gate_input_offset);
        encoder.set_buffer(3, Some(a_log_buffer), 0);
        encoder.set_buffer(4, Some(dt_bias_buffer), 0);
        encoder.set_buffer(5, Some(q_inv_buffer), 0);
        encoder.set_buffer(6, Some(k_inv_buffer), 0);
        encoder.set_buffer(7, Some(beta_buffer), 0);
        encoder.set_buffer(8, Some(decay_buffer), 0);
        set_u32_bytes(encoder, 9, &dims, "linear_attn_inv_norm_dims")?;
        set_f32_bytes(encoder, 10, &scales, "linear_attn_inv_norm_scales")?;
        profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(checked_nsuint(groups, "linear-attn inv norm groups")?, 1, 1),
            MTLSize::new(LINEAR_ATTN_TG_WIDTH, 1, 1),
        );
        post_dispatch_barrier(encoder);
        Ok(())
    }

    pub(super) fn encode_linear_attn_gated_delta(
        &self,
        encoder: &ComputeCommandEncoderRef,
        conv_out_buffer: &BufferRef,
        q_norm_buffer: &BufferRef,
        k_norm_buffer: &BufferRef,
        beta_buffer: &BufferRef,
        decay_buffer: &BufferRef,
        ssm_state_buffer: &BufferRef,
        ssm_bf16: bool,
        y_buffer: &BufferRef,
        spec: LinearAttentionStepSpec,
    ) -> Result<()> {
        let repeat = spec
            .num_value_heads
            .checked_div(spec.num_key_heads)
            .ok_or_else(|| InferError::Metal("linear-attn repeat nul".to_string()))?;
        let dims = [
            checked_u32(spec.num_value_heads, "linear-attn value heads")?,
            checked_u32(spec.value_head_dim, "linear-attn value head dim")?,
            checked_u32(spec.key_head_dim, "linear-attn key head dim")?,
            checked_u32(repeat, "linear-attn repeat")?,
        ];
        let use_dk128 = linear_delta_dk128_enabled()
            && spec.key_head_dim == 128
            && spec.value_head_dim % 4 == 0;
        if ssm_bf16 && !use_dk128 {
            return Err(InferError::Metal(
                "linear-attn SSM bf16 requiert le fast path dk128".to_string(),
            ));
        }
        let pipeline = if ssm_bf16 {
            &self.linear_attn_gated_delta_dk128_bf16_tg4_f32
        } else if use_dk128 {
            &self.linear_attn_gated_delta_dk128_tg4_f32
        } else {
            &self.linear_attn_gated_delta_f32
        };
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(conv_out_buffer), 0);
        encoder.set_buffer(1, Some(q_norm_buffer), 0);
        encoder.set_buffer(2, Some(k_norm_buffer), 0);
        encoder.set_buffer(3, Some(beta_buffer), 0);
        encoder.set_buffer(4, Some(decay_buffer), 0);
        encoder.set_buffer(5, Some(ssm_state_buffer), 0);
        encoder.set_buffer(6, Some(y_buffer), 0);
        set_u32_bytes(encoder, 7, &dims, "linear_attn_delta_dims")?;
        profile_dispatch();
        if use_dk128 {
            let value_head_dim = checked_nsuint(spec.value_head_dim, "linear-attn value dim")?;
            let tg_rows = linear_delta_tg_rows_for_state(ssm_bf16).min(value_head_dim);
            encoder.dispatch_threads(
                MTLSize::new(
                    LINEAR_ATTN_TG_WIDTH,
                    value_head_dim,
                    checked_nsuint(spec.num_value_heads, "linear-attn value heads")?,
                ),
                MTLSize::new(LINEAR_ATTN_TG_WIDTH, tg_rows, 1),
            );
        } else {
            encoder.dispatch_thread_groups(
                MTLSize::new(
                    checked_nsuint(spec.value_head_dim, "linear-attn value dim")?,
                    checked_nsuint(spec.num_value_heads, "linear-attn value heads")?,
                    1,
                ),
                MTLSize::new(LINEAR_ATTN_TG_WIDTH, 1, 1),
            );
        }
        post_dispatch_barrier(encoder);
        Ok(())
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "signature d'encodage Metal: buffers + dimensions"
    )]
    pub(super) fn encode_linear_attn_gated_delta_inv_dk128(
        &self,
        encoder: &ComputeCommandEncoderRef,
        conv_out_buffer: &BufferRef,
        q_inv_buffer: &BufferRef,
        k_inv_buffer: &BufferRef,
        beta_buffer: &BufferRef,
        decay_buffer: &BufferRef,
        ssm_state_buffer: &BufferRef,
        _ssm_bf16: bool,
        y_buffer: &BufferRef,
        spec: LinearAttentionStepSpec,
    ) -> Result<()> {
        let repeat = spec
            .num_value_heads
            .checked_div(spec.num_key_heads)
            .ok_or_else(|| InferError::Metal("linear-attn inv repeat nul".to_string()))?;
        let dims = [
            checked_u32(spec.num_value_heads, "linear-attn inv value heads")?,
            checked_u32(spec.value_head_dim, "linear-attn inv value dim")?,
            checked_u32(spec.key_head_dim, "linear-attn inv key dim")?,
            checked_u32(repeat, "linear-attn inv repeat")?,
        ];
        let value_head_dim = checked_nsuint(spec.value_head_dim, "linear-attn inv value dim")?;
        let tg_rows = linear_delta_tg_rows().min(value_head_dim);
        encoder.set_compute_pipeline_state(&self.linear_attn_gated_delta_inv_dk128_tg4_f32);
        encoder.set_buffer(0, Some(conv_out_buffer), 0);
        encoder.set_buffer(1, Some(q_inv_buffer), 0);
        encoder.set_buffer(2, Some(k_inv_buffer), 0);
        encoder.set_buffer(3, Some(beta_buffer), 0);
        encoder.set_buffer(4, Some(decay_buffer), 0);
        encoder.set_buffer(5, Some(ssm_state_buffer), 0);
        encoder.set_buffer(6, Some(y_buffer), 0);
        set_u32_bytes(encoder, 7, &dims, "linear_attn_inv_delta_dims")?;
        profile_dispatch();
        encoder.dispatch_threads(
            MTLSize::new(
                LINEAR_ATTN_TG_WIDTH,
                value_head_dim,
                checked_nsuint(spec.num_value_heads, "linear-attn inv value heads")?,
            ),
            MTLSize::new(LINEAR_ATTN_TG_WIDTH, tg_rows, 1),
        );
        post_dispatch_barrier(encoder);
        Ok(())
    }

    pub(super) fn encode_linear_attn_rms_gate(
        &self,
        encoder: &ComputeCommandEncoderRef,
        y_buffer: &BufferRef,
        z_buffer: &BufferRef,
        norm_weight_buffer: &BufferRef,
        gated_buffer: &BufferRef,
        spec: LinearAttentionStepSpec,
    ) -> Result<()> {
        self.encode_linear_attn_rms_gate_with_offset(
            encoder,
            y_buffer,
            z_buffer,
            0,
            norm_weight_buffer,
            gated_buffer,
            spec,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "signature d'encodage Metal: buffers + offset + dimensions"
    )]
    pub(super) fn encode_linear_attn_rms_gate_with_offset(
        &self,
        encoder: &ComputeCommandEncoderRef,
        y_buffer: &BufferRef,
        z_buffer: &BufferRef,
        z_offset: u64,
        norm_weight_buffer: &BufferRef,
        gated_buffer: &BufferRef,
        spec: LinearAttentionStepSpec,
    ) -> Result<()> {
        self.encode_linear_attn_rms_gate_with_offsets(
            encoder,
            y_buffer,
            0,
            z_buffer,
            z_offset,
            norm_weight_buffer,
            gated_buffer,
            0,
            spec,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "signature d'encodage Metal: buffers + offsets + dimensions"
    )]
    pub(super) fn encode_linear_attn_rms_gate_with_offsets(
        &self,
        encoder: &ComputeCommandEncoderRef,
        y_buffer: &BufferRef,
        y_offset: u64,
        z_buffer: &BufferRef,
        z_offset: u64,
        norm_weight_buffer: &BufferRef,
        gated_buffer: &BufferRef,
        gated_offset: u64,
        spec: LinearAttentionStepSpec,
    ) -> Result<()> {
        let dims = [
            checked_u32(spec.num_value_heads, "linear-attn value heads")?,
            checked_u32(spec.value_head_dim, "linear-attn value head dim")?,
        ];
        let use_dv128 = linear_rms_dv128_enabled() && spec.value_head_dim == 128;
        let pipeline = if use_dv128 {
            &self.linear_attn_rms_gate_dv128_f32
        } else {
            &self.linear_attn_rms_gate_f32
        };
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(y_buffer), y_offset);
        encoder.set_buffer(1, Some(z_buffer), z_offset);
        encoder.set_buffer(2, Some(norm_weight_buffer), 0);
        encoder.set_buffer(3, Some(gated_buffer), gated_offset);
        set_u32_bytes(encoder, 4, &dims, "linear_attn_rms_dims")?;
        set_f32_bytes(encoder, 5, &[spec.rms_eps], "linear_attn_rms_eps")?;
        profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(
                checked_nsuint(spec.num_value_heads, "linear-attn rms heads")?,
                1,
                1,
            ),
            MTLSize::new(LINEAR_ATTN_TG_WIDTH, 1, 1),
        );
        post_dispatch_barrier(encoder);
        Ok(())
    }

    /// Brick #8 : rms-gate BATCHÉ (un dispatch grid `value_heads×batch` au lieu d'une
    /// boucle per-token). Buffers contigus `[batch, value_dim]`. value_head_dim=128.
    pub(super) fn encode_linear_attn_rms_gate_batch(
        &self,
        encoder: &ComputeCommandEncoderRef,
        y_buffer: &BufferRef,
        z_buffer: &BufferRef,
        norm_weight_buffer: &BufferRef,
        gated_buffer: &BufferRef,
        batch: usize,
        spec: LinearAttentionStepSpec,
    ) -> Result<()> {
        let dims = [
            checked_u32(spec.num_value_heads, "linear-attn value heads")?,
            checked_u32(spec.value_head_dim, "linear-attn value head dim")?,
            checked_u32(batch, "linear-attn rms batch")?,
        ];
        encoder.set_compute_pipeline_state(&self.linear_attn_rms_gate_batch_dv128_f32);
        encoder.set_buffer(0, Some(y_buffer), 0);
        encoder.set_buffer(1, Some(z_buffer), 0);
        encoder.set_buffer(2, Some(norm_weight_buffer), 0);
        encoder.set_buffer(3, Some(gated_buffer), 0);
        set_u32_bytes(encoder, 4, &dims, "linear_attn_rms_batch_dims")?;
        set_f32_bytes(encoder, 5, &[spec.rms_eps], "linear_attn_rms_eps")?;
        profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(
                checked_nsuint(spec.num_value_heads, "linear-attn rms batch heads")?,
                checked_nsuint(batch, "linear-attn rms batch grid")?,
                1,
            ),
            MTLSize::new(LINEAR_ATTN_TG_WIDTH, 1, 1),
        );
        post_dispatch_barrier(encoder);
        Ok(())
    }
}
