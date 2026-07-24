//! Encodage des couches full-attn, linear-attn et MoE résidentes.

use super::utils::byte_offset_f32;
use super::*;

impl DecodeResidentState {
    fn encode_query_attention_scale(
        executor: &MetalExecutor,
        encoder: &ComputeCommandEncoderRef,
        owned: &mut Vec<Buffer>,
        query: &BufferRef,
        query_len: usize,
        head_dim: usize,
        attn_scalar: f32,
    ) -> Result<()> {
        if !attn_scalar.is_finite() || attn_scalar <= 0.0 {
            return Err(InferError::Config(format!(
                "scalaire d'attention résident invalide: {attn_scalar}"
            )));
        }
        let scale = ((head_dim as f32) / attn_scalar).sqrt();
        if scale.to_bits() != 1.0_f32.to_bits() {
            executor.encode_accumulate_scaled(
                encoder,
                owned,
                query,
                query,
                scale - 1.0,
                query_len,
            )?;
        }
        Ok(())
    }

    /// Encode UNE couche full-attn (decode résident 1c) à la position
    /// `dims.position` dans l'encoder PARTAGÉ : `layer_in [hidden]` → `layer_out
    /// [hidden]`, sans commit ni readback. Reproduit entièrement sur GPU le couple
    /// `full_attention_context_cached` puis `full_attention_tail_moe_shared`
    /// (modules `decoder` / `metal_backend`).
    ///
    /// Data flow (gotchas 1c) : (a) `attn_output_gate=true` → q_proj sort `2·q_dim`,
    /// split_q_gate AVANT le RoPE ; (b) norm+RoPE À LA POSITION du token
    /// ([`Self::encode_rms_norm_rope_decode`]), pas le prefill ; (c) append KV
    /// device puis attention dans le même encoder (R3, hazard-tracked) ; (d) gate
    /// de sortie APRÈS l'attention ; (e) le tail fusionne résiduel + post-norm +
    /// MoE + shared-expert vers `layer_out`. AUCUN chemin CPU au milieu (MAJEUR 6,
    /// garde-fou `supports_resident_full_decode` en amont).
    ///
    /// Liveness (R1) : les bails scratch sont loués au pool de l'arène et droppés
    /// en fin de couche ; le pool POSSÈDE les buffers (vivants jusqu'au wait), la
    /// couche suivante réutilise les slots, le hazard-tracking ordonne la
    /// réutilisation (prouvé micro-jalon `resident_scratch_drop_reuse_within_one_encoder`).
    ///
    /// # Errors
    ///
    /// Propage toute erreur d'encodage (dimension, overflow KV, Metal).
    #[expect(
        clippy::too_many_arguments,
        reason = "data flow d'une couche : exécuteur + KV + poids + dims + ping-pong"
    )]
    pub(crate) fn encode_full_attn_layer(
        &self,
        executor: &MetalExecutor,
        encoder: &ComputeCommandEncoderRef,
        owned: &mut Vec<Buffer>,
        kv: &mut FullAttentionMetalState,
        weights: FullAttnLayerWeights<'_>,
        dims: FullAttnLayerDims,
        layer_in: &BufferRef,
        layer_out: &BufferRef,
    ) -> Result<()> {
        let hidden = dims.hidden;
        let q_dim = dims
            .q_heads
            .checked_mul(dims.head_dim)
            .ok_or_else(|| InferError::Dimension("full-attn q_dim déborde".to_string()))?;
        let kv_dim = dims
            .kv_heads
            .checked_mul(dims.head_dim)
            .ok_or_else(|| InferError::Dimension("full-attn kv_dim déborde".to_string()))?;
        let q_gate_dim = q_dim
            .checked_mul(2)
            .ok_or_else(|| InferError::Dimension("full-attn q_gate_dim déborde".to_string()))?;
        let score_cells = dims
            .q_heads
            .checked_mul(kv.capacity())
            .ok_or_else(|| InferError::Dimension("full-attn scores débordent".to_string()))?;

        // Scratch d'UNE couche (loué, réutilisé sur toutes les couches — R1).
        let normed = self.scratch().lease(hidden, GpuElement::F32)?;
        let qkv_proj_dim = q_gate_dim
            .checked_add(kv_dim)
            .and_then(|value| value.checked_add(kv_dim))
            .ok_or_else(|| InferError::Dimension("full-attn qkv concat déborde".to_string()))?;
        let qkv_proj_out = self.scratch().lease(qkv_proj_dim, GpuElement::F32)?;
        let q_proj_out = self.scratch().lease(q_gate_dim, GpuElement::F32)?;
        let k_raw = self.scratch().lease(kv_dim, GpuElement::F32)?;
        let v_raw = self.scratch().lease(kv_dim, GpuElement::F32)?;
        let q_raw = self.scratch().lease(q_dim, GpuElement::F32)?;
        let gate = self.scratch().lease(q_dim, GpuElement::F32)?;
        let q_roped = self.scratch().lease(q_dim, GpuElement::F32)?;
        let k_roped = self.scratch().lease(kv_dim, GpuElement::F32)?;
        let scores = self.scratch().lease(score_cells, GpuElement::F32)?;
        let ctx = self.scratch().lease(q_dim, GpuElement::F32)?;
        let gated_ctx = self.scratch().lease(q_dim, GpuElement::F32)?;
        let o_out = self.scratch().lease(hidden, GpuElement::F32)?;
        let summed = self.scratch().lease(hidden, GpuElement::F32)?;
        let post_normed = self.scratch().lease(hidden, GpuElement::F32)?;

        // Projections concaténées Q (gated → 2·q_dim), K, V.
        let mut qkv_concat_used = false;
        if let Some(qkv_proj) = weights.qkv_proj {
            qkv_concat_used = executor
                .encode_full_attn_qkv_split_rms_buffers(
                    encoder,
                    layer_in,
                    weights.input_norm,
                    dims.eps,
                    hidden,
                    qkv_proj,
                    qkv_proj_out.tensor().buffer(),
                    q_raw.tensor().buffer(),
                    gate.tensor().buffer(),
                    dims.q_heads,
                    dims.head_dim,
                )?
                .is_some();
        }
        if !qkv_concat_used {
            executor.encode_rms_norm_rows(
                encoder,
                layer_in,
                weights.input_norm,
                normed.tensor().buffer(),
                1,
                hidden,
                dims.eps,
            )?;
            if let Some(qkv_proj) = weights.qkv_proj {
                qkv_concat_used = executor
                    .encode_full_attn_qkv_split_buffers(
                        encoder,
                        normed.tensor().buffer(),
                        hidden,
                        qkv_proj,
                        qkv_proj_out.tensor().buffer(),
                        q_raw.tensor().buffer(),
                        gate.tensor().buffer(),
                        dims.q_heads,
                        dims.head_dim,
                    )?
                    .is_some();
                if !qkv_concat_used {
                    executor.encode_matmul_weight_buffers(
                        encoder,
                        normed.tensor().buffer(),
                        1,
                        hidden,
                        qkv_proj,
                        qkv_proj_out.tensor().buffer(),
                        false,
                    )?;
                    self.encode_split_q_gate_with_offset(
                        encoder,
                        qkv_proj_out.tensor().buffer(),
                        0,
                        q_raw.tensor().buffer(),
                        gate.tensor().buffer(),
                        dims.q_heads,
                        dims.head_dim,
                    )?;
                    qkv_concat_used = true;
                }
            }
            if !qkv_concat_used {
                executor.encode_matmul_weight_buffers(
                    encoder,
                    normed.tensor().buffer(),
                    1,
                    hidden,
                    weights.q_proj,
                    q_proj_out.tensor().buffer(),
                    false,
                )?;
                executor.encode_matmul_weight_buffers(
                    encoder,
                    normed.tensor().buffer(),
                    1,
                    hidden,
                    weights.k_proj,
                    k_raw.tensor().buffer(),
                    false,
                )?;
                executor.encode_matmul_weight_buffers(
                    encoder,
                    normed.tensor().buffer(),
                    1,
                    hidden,
                    weights.v_proj,
                    v_raw.tensor().buffer(),
                    false,
                )?;
                self.encode_split_q_gate(
                    encoder,
                    q_proj_out.tensor().buffer(),
                    q_raw.tensor().buffer(),
                    gate.tensor().buffer(),
                    dims.q_heads,
                    dims.head_dim,
                )?;
            }
        }
        // norm + RoPE à la POSITION du token (single-query)
        self.encode_rms_norm_rope_decode_with_offset(
            encoder,
            q_raw.tensor().buffer(),
            0,
            weights.q_norm,
            q_roped.tensor().buffer(),
            dims.q_heads,
            dims.head_dim,
            dims.rope_dims,
            dims.rope_frequency_dim,
            dims.position,
            dims.eps,
            dims.theta,
        )?;
        Self::encode_query_attention_scale(
            executor,
            encoder,
            owned,
            q_roped.tensor().buffer(),
            q_dim,
            dims.head_dim,
            dims.attn_scalar,
        )?;
        if qkv_concat_used {
            let k_offset = byte_offset_f32(q_gate_dim, "full-attn k offset")?;
            let v_offset = byte_offset_f32(
                q_gate_dim.checked_add(kv_dim).ok_or_else(|| {
                    InferError::Dimension("full-attn v offset déborde".to_string())
                })?,
                "full-attn v offset",
            )?;
            self.encode_rms_norm_rope_decode_with_offset(
                encoder,
                qkv_proj_out.tensor().buffer(),
                k_offset,
                weights.k_norm,
                k_roped.tensor().buffer(),
                dims.kv_heads,
                dims.head_dim,
                dims.rope_dims,
                dims.rope_frequency_dim,
                dims.position,
                dims.eps,
                dims.theta,
            )?;
            kv.encode_append_kv_with_offsets(
                encoder,
                k_roped.tensor().buffer(),
                0,
                qkv_proj_out.tensor().buffer(),
                v_offset,
            )?;
        } else {
            self.encode_rms_norm_rope_decode(
                encoder,
                k_raw.tensor().buffer(),
                weights.k_norm,
                k_roped.tensor().buffer(),
                dims.kv_heads,
                dims.head_dim,
                dims.rope_dims,
                dims.rope_frequency_dim,
                dims.position,
                dims.eps,
                dims.theta,
            )?;
            kv.encode_append_kv(encoder, k_roped.tensor().buffer(), v_raw.tensor().buffer())?;
        }
        kv.encode_attention_decode_windowed(
            encoder,
            q_roped.tensor().buffer(),
            scores.tensor().buffer(),
            ctx.tensor().buffer(),
            dims.window_start,
        )?;
        let o_proj_fused = executor
            .encode_full_attn_o_proj_gated_buffers(
                encoder,
                ctx.tensor().buffer(),
                gate.tensor().buffer(),
                q_dim,
                weights.o_proj,
                o_out.tensor().buffer(),
            )?
            .is_some();
        if !o_proj_fused {
            // gate de sortie APRÈS l'attention
            self.encode_attn_gate(
                encoder,
                ctx.tensor().buffer(),
                gate.tensor().buffer(),
                gated_ctx.tensor().buffer(),
                q_dim,
            )?;
            // tail : o_proj + résiduel+post_norm fusionnés + MoE+shared, vers layer_out
            executor.encode_matmul_weight_buffers(
                encoder,
                gated_ctx.tensor().buffer(),
                1,
                q_dim,
                weights.o_proj,
                o_out.tensor().buffer(),
                false,
            )?;
        }
        if weights.pre_feedforward_norm.is_some() {
            executor.encode_rms_norm_rows(
                encoder,
                o_out.tensor().buffer(),
                weights.post_norm,
                post_normed.tensor().buffer(),
                1,
                hidden,
                dims.eps,
            )?;
            executor.encode_add_scaled(
                encoder,
                owned,
                layer_in,
                post_normed.tensor().buffer(),
                summed.tensor().buffer(),
                1.0,
                hidden,
            )?;
        } else {
            executor.encode_add_rms_norm_rows(
                encoder,
                layer_in,
                o_out.tensor().buffer(),
                weights.post_norm,
                summed.tensor().buffer(),
                post_normed.tensor().buffer(),
                1,
                hidden,
                dims.eps,
            )?;
        }
        executor.encode_moe_shared_buffers(
            encoder,
            owned,
            post_normed.tensor().buffer(),
            Some(summed.tensor().buffer()),
            layer_out,
            hidden,
            weights.moe,
            weights.top_k,
        )?;
        // Bails droppés ICI (fin de couche) ; pool propriétaire → buffers vivants
        // jusqu'au wait, slots réutilisés par la couche suivante (R1).
        Ok(())
    }

    /// Encode UNE couche full-attn MoE routed-only dans l'encoder partagé.
    ///
    /// Le préfixe attention/KV est identique au chemin shared+routed ; seul le
    /// tail MoE appelle le sous-ensemble routed sans shared-expert.
    ///
    /// Contrairement aux variantes shared et dense, ce chemin ne porte pas les
    /// poids Q/K/V séparés : le setup routed-only échoue avant l'encodage si le
    /// QKV concaténé n'est pas disponible, puis le decode retombe sur le chemin
    /// per-op global.
    #[expect(
        clippy::too_many_arguments,
        reason = "data flow d'une couche : exécuteur + KV + poids + dims + ping-pong"
    )]
    pub(crate) fn encode_full_attn_routed_layer(
        &self,
        executor: &MetalExecutor,
        encoder: &ComputeCommandEncoderRef,
        owned: &mut Vec<Buffer>,
        kv: &mut FullAttentionMetalState,
        weights: FullAttnRoutedLayerWeights<'_>,
        dims: FullAttnLayerDims,
        layer_in: &BufferRef,
        layer_out: &BufferRef,
    ) -> Result<()> {
        let hidden = dims.hidden;
        let q_dim = dims
            .q_heads
            .checked_mul(dims.head_dim)
            .ok_or_else(|| InferError::Dimension("full-attn routed q_dim déborde".to_string()))?;
        let kv_dim = dims
            .kv_heads
            .checked_mul(dims.head_dim)
            .ok_or_else(|| InferError::Dimension("full-attn routed kv_dim déborde".to_string()))?;
        let q_proj_dim = if dims.attn_output_gate {
            q_dim.checked_mul(2).ok_or_else(|| {
                InferError::Dimension("full-attn routed q_gate_dim déborde".to_string())
            })?
        } else {
            q_dim
        };
        let score_cells = dims.q_heads.checked_mul(kv.capacity()).ok_or_else(|| {
            InferError::Dimension("full-attn routed scores débordent".to_string())
        })?;

        let normed = self.scratch().lease(hidden, GpuElement::F32)?;
        let qkv_proj_dim = q_proj_dim
            .checked_add(kv_dim)
            .and_then(|value| value.checked_add(kv_dim))
            .ok_or_else(|| {
                InferError::Dimension("full-attn routed qkv concat déborde".to_string())
            })?;
        let qkv_proj_out = self.scratch().lease(qkv_proj_dim, GpuElement::F32)?;
        let q_raw = self.scratch().lease(q_dim, GpuElement::F32)?;
        let gate = self.scratch().lease(q_dim, GpuElement::F32)?;
        let q_roped = self.scratch().lease(q_dim, GpuElement::F32)?;
        let k_roped = self.scratch().lease(kv_dim, GpuElement::F32)?;
        let scores = self.scratch().lease(score_cells, GpuElement::F32)?;
        let ctx = self.scratch().lease(q_dim, GpuElement::F32)?;
        let gated_ctx = self.scratch().lease(q_dim, GpuElement::F32)?;
        let o_out = self.scratch().lease(hidden, GpuElement::F32)?;
        let summed = self.scratch().lease(hidden, GpuElement::F32)?;
        let post_normed = self.scratch().lease(hidden, GpuElement::F32)?;

        if dims.attn_output_gate {
            let qkv_split_fused = executor
                .encode_full_attn_qkv_split_rms_buffers(
                    encoder,
                    layer_in,
                    weights.input_norm,
                    dims.eps,
                    hidden,
                    weights.qkv_proj,
                    qkv_proj_out.tensor().buffer(),
                    q_raw.tensor().buffer(),
                    gate.tensor().buffer(),
                    dims.q_heads,
                    dims.head_dim,
                )?
                .is_some();
            if !qkv_split_fused {
                executor.encode_rms_norm_rows(
                    encoder,
                    layer_in,
                    weights.input_norm,
                    normed.tensor().buffer(),
                    1,
                    hidden,
                    dims.eps,
                )?;
                let qkv_epilogue_fused = executor
                    .encode_full_attn_qkv_split_buffers(
                        encoder,
                        normed.tensor().buffer(),
                        hidden,
                        weights.qkv_proj,
                        qkv_proj_out.tensor().buffer(),
                        q_raw.tensor().buffer(),
                        gate.tensor().buffer(),
                        dims.q_heads,
                        dims.head_dim,
                    )?
                    .is_some();
                if !qkv_epilogue_fused {
                    executor.encode_matmul_weight_buffers(
                        encoder,
                        normed.tensor().buffer(),
                        1,
                        hidden,
                        weights.qkv_proj,
                        qkv_proj_out.tensor().buffer(),
                        false,
                    )?;
                    self.encode_split_q_gate_with_offset(
                        encoder,
                        qkv_proj_out.tensor().buffer(),
                        0,
                        q_raw.tensor().buffer(),
                        gate.tensor().buffer(),
                        dims.q_heads,
                        dims.head_dim,
                    )?;
                }
            }
        } else {
            executor.encode_rms_norm_rows(
                encoder,
                layer_in,
                weights.input_norm,
                normed.tensor().buffer(),
                1,
                hidden,
                dims.eps,
            )?;
            executor.encode_matmul_weight_buffers(
                encoder,
                normed.tensor().buffer(),
                1,
                hidden,
                weights.qkv_proj,
                qkv_proj_out.tensor().buffer(),
                false,
            )?;
        }
        let k_offset = byte_offset_f32(q_proj_dim, "full-attn routed k offset")?;
        let v_offset = byte_offset_f32(
            q_proj_dim.checked_add(kv_dim).ok_or_else(|| {
                InferError::Dimension("full-attn routed v offset déborde".to_string())
            })?,
            "full-attn routed v offset",
        )?;
        let q_source_buffer = if dims.attn_output_gate {
            q_raw.tensor().buffer()
        } else {
            qkv_proj_out.tensor().buffer()
        };
        self.encode_rms_norm_rope_decode_with_offset(
            encoder,
            q_source_buffer,
            0,
            weights.q_norm,
            q_roped.tensor().buffer(),
            dims.q_heads,
            dims.head_dim,
            dims.rope_dims,
            dims.rope_frequency_dim,
            dims.position,
            dims.eps,
            dims.theta,
        )?;
        Self::encode_query_attention_scale(
            executor,
            encoder,
            owned,
            q_roped.tensor().buffer(),
            q_dim,
            dims.head_dim,
            dims.attn_scalar,
        )?;
        self.encode_rms_norm_rope_decode_with_offset(
            encoder,
            qkv_proj_out.tensor().buffer(),
            k_offset,
            weights.k_norm,
            k_roped.tensor().buffer(),
            dims.kv_heads,
            dims.head_dim,
            dims.rope_dims,
            dims.rope_frequency_dim,
            dims.position,
            dims.eps,
            dims.theta,
        )?;
        kv.encode_append_kv_with_offsets(
            encoder,
            k_roped.tensor().buffer(),
            0,
            qkv_proj_out.tensor().buffer(),
            v_offset,
        )?;
        kv.encode_attention_decode_windowed(
            encoder,
            q_roped.tensor().buffer(),
            scores.tensor().buffer(),
            ctx.tensor().buffer(),
            dims.window_start,
        )?;
        if dims.attn_output_gate {
            let o_proj_fused = executor
                .encode_full_attn_o_proj_gated_buffers(
                    encoder,
                    ctx.tensor().buffer(),
                    gate.tensor().buffer(),
                    q_dim,
                    weights.o_proj,
                    o_out.tensor().buffer(),
                )?
                .is_some();
            if !o_proj_fused {
                self.encode_attn_gate(
                    encoder,
                    ctx.tensor().buffer(),
                    gate.tensor().buffer(),
                    gated_ctx.tensor().buffer(),
                    q_dim,
                )?;
                executor.encode_matmul_weight_buffers(
                    encoder,
                    gated_ctx.tensor().buffer(),
                    1,
                    q_dim,
                    weights.o_proj,
                    o_out.tensor().buffer(),
                    false,
                )?;
            }
        } else {
            executor.encode_matmul_weight_buffers(
                encoder,
                ctx.tensor().buffer(),
                1,
                q_dim,
                weights.o_proj,
                o_out.tensor().buffer(),
                false,
            )?;
        }
        if weights.pre_feedforward_norm.is_some() {
            executor.encode_rms_norm_rows(
                encoder,
                o_out.tensor().buffer(),
                weights.post_norm,
                post_normed.tensor().buffer(),
                1,
                hidden,
                dims.eps,
            )?;
            executor.encode_add_scaled(
                encoder,
                owned,
                layer_in,
                post_normed.tensor().buffer(),
                summed.tensor().buffer(),
                1.0,
                hidden,
            )?;
        } else {
            executor.encode_add_rms_norm_rows(
                encoder,
                layer_in,
                o_out.tensor().buffer(),
                weights.post_norm,
                summed.tensor().buffer(),
                post_normed.tensor().buffer(),
                1,
                hidden,
                dims.eps,
            )?;
        }
        match (weights.pre_feedforward_norm, weights.post_feedforward_norm) {
            (Some(pre_feedforward_norm), Some(post_feedforward_norm)) => self
                .encode_gemma_moe_tail(
                    executor,
                    encoder,
                    owned,
                    summed.tensor().buffer(),
                    layer_out,
                    hidden,
                    dims.eps,
                    GemmaMoeTailWeights {
                        pre_feedforward_norm,
                        post_feedforward_norm,
                        layer_scalar: weights.layer_scalar,
                        moe: weights.moe,
                        top_k: weights.top_k,
                    },
                ),
            (None, None) => executor.encode_moe_routed_buffers(
                encoder,
                owned,
                post_normed.tensor().buffer(),
                Some(summed.tensor().buffer()),
                layer_out,
                hidden,
                weights.moe,
                weights.top_k,
            ),
            _ => Err(InferError::Config(
                "normes feed-forward Gemma résidentes partielles".to_string(),
            )),
        }
    }

    /// Encode UNE couche full-attn DENSE (decode résident 1c) dans l'encoder
    /// partagé. Le préfixe attention/KV est identique au chemin MoE, mais le tail
    /// exécute un MLP dense `gate/up/down` puis ajoute le résiduel `summed`.
    #[expect(
        clippy::too_many_arguments,
        reason = "data flow d'une couche : exécuteur + KV + poids + dims + ping-pong"
    )]
    pub(crate) fn encode_full_attn_dense_layer(
        &self,
        executor: &MetalExecutor,
        encoder: &ComputeCommandEncoderRef,
        owned: &mut Vec<Buffer>,
        kv: &mut FullAttentionMetalState,
        weights: FullAttnDenseLayerWeights<'_>,
        dims: FullAttnLayerDims,
        layer_in: &BufferRef,
        layer_out: &BufferRef,
    ) -> Result<()> {
        let hidden = dims.hidden;
        let q_dim = dims
            .q_heads
            .checked_mul(dims.head_dim)
            .ok_or_else(|| InferError::Dimension("full-attn dense q_dim déborde".to_string()))?;
        let kv_dim = dims
            .kv_heads
            .checked_mul(dims.head_dim)
            .ok_or_else(|| InferError::Dimension("full-attn dense kv_dim déborde".to_string()))?;
        let q_proj_dim = if dims.attn_output_gate {
            q_dim.checked_mul(2).ok_or_else(|| {
                InferError::Dimension("full-attn dense q_gate_dim déborde".to_string())
            })?
        } else {
            q_dim
        };
        let qkv_proj_dim = q_proj_dim
            .checked_add(kv_dim)
            .and_then(|value| value.checked_add(kv_dim))
            .ok_or_else(|| {
                InferError::Dimension("full-attn dense qkv concat déborde".to_string())
            })?;
        let score_cells = dims
            .q_heads
            .checked_mul(kv.capacity())
            .ok_or_else(|| InferError::Dimension("full-attn dense scores débordent".to_string()))?;

        let normed = self.scratch().lease(hidden, GpuElement::F32)?;
        let qkv_proj_out = self.scratch().lease(qkv_proj_dim, GpuElement::F32)?;
        let q_proj_out = self.scratch().lease(q_proj_dim, GpuElement::F32)?;
        let k_raw = self.scratch().lease(kv_dim, GpuElement::F32)?;
        let v_raw = self.scratch().lease(kv_dim, GpuElement::F32)?;
        let v_normed = self.scratch().lease(kv_dim, GpuElement::F32)?;
        let q_raw = self.scratch().lease(q_dim, GpuElement::F32)?;
        let gate = self.scratch().lease(q_dim, GpuElement::F32)?;
        let q_roped = self.scratch().lease(q_dim, GpuElement::F32)?;
        let k_roped = self.scratch().lease(kv_dim, GpuElement::F32)?;
        let scores = self.scratch().lease(score_cells, GpuElement::F32)?;
        let ctx = self.scratch().lease(q_dim, GpuElement::F32)?;
        let gated_ctx = self.scratch().lease(q_dim, GpuElement::F32)?;
        let o_out = self.scratch().lease(hidden, GpuElement::F32)?;
        let summed = self.scratch().lease(hidden, GpuElement::F32)?;
        let post_normed = self.scratch().lease(hidden, GpuElement::F32)?;

        let mut qkv_concat_used = false;
        if dims.attn_output_gate {
            if let Some(qkv_proj) = weights.qkv_proj {
                qkv_concat_used = executor
                    .encode_full_attn_qkv_split_rms_buffers(
                        encoder,
                        layer_in,
                        weights.input_norm,
                        dims.eps,
                        hidden,
                        qkv_proj,
                        qkv_proj_out.tensor().buffer(),
                        q_raw.tensor().buffer(),
                        gate.tensor().buffer(),
                        dims.q_heads,
                        dims.head_dim,
                    )?
                    .is_some();
            }
        }
        if !qkv_concat_used {
            executor.encode_rms_norm_rows(
                encoder,
                layer_in,
                weights.input_norm,
                normed.tensor().buffer(),
                1,
                hidden,
                dims.eps,
            )?;
            if dims.attn_output_gate {
                if let Some(qkv_proj) = weights.qkv_proj {
                    qkv_concat_used = executor
                        .encode_full_attn_qkv_split_buffers(
                            encoder,
                            normed.tensor().buffer(),
                            hidden,
                            qkv_proj,
                            qkv_proj_out.tensor().buffer(),
                            q_raw.tensor().buffer(),
                            gate.tensor().buffer(),
                            dims.q_heads,
                            dims.head_dim,
                        )?
                        .is_some();
                    if !qkv_concat_used {
                        executor.encode_matmul_weight_buffers(
                            encoder,
                            normed.tensor().buffer(),
                            1,
                            hidden,
                            qkv_proj,
                            qkv_proj_out.tensor().buffer(),
                            false,
                        )?;
                        self.encode_split_q_gate_with_offset(
                            encoder,
                            qkv_proj_out.tensor().buffer(),
                            0,
                            q_raw.tensor().buffer(),
                            gate.tensor().buffer(),
                            dims.q_heads,
                            dims.head_dim,
                        )?;
                        qkv_concat_used = true;
                    }
                }
            } else if weights.qkv_proj_without_gate {
                if let Some(qkv_proj) = weights.qkv_proj {
                    let projected = executor.encode_matmul_weight_buffers(
                        encoder,
                        normed.tensor().buffer(),
                        1,
                        hidden,
                        qkv_proj,
                        qkv_proj_out.tensor().buffer(),
                        false,
                    )?;
                    if projected != qkv_proj_dim {
                        return Err(InferError::Dimension(format!(
                            "full-attn dense qkv sort {projected}, attendu {qkv_proj_dim}"
                        )));
                    }
                    qkv_concat_used = true;
                }
            }
            if !qkv_concat_used {
                executor.encode_matmul_weight_buffers(
                    encoder,
                    normed.tensor().buffer(),
                    1,
                    hidden,
                    weights.q_proj,
                    q_proj_out.tensor().buffer(),
                    false,
                )?;
                executor.encode_matmul_weight_buffers(
                    encoder,
                    normed.tensor().buffer(),
                    1,
                    hidden,
                    weights.k_proj,
                    k_raw.tensor().buffer(),
                    false,
                )?;
                executor.encode_matmul_weight_buffers(
                    encoder,
                    normed.tensor().buffer(),
                    1,
                    hidden,
                    weights.v_proj,
                    v_raw.tensor().buffer(),
                    false,
                )?;
                if dims.attn_output_gate {
                    self.encode_split_q_gate(
                        encoder,
                        q_proj_out.tensor().buffer(),
                        q_raw.tensor().buffer(),
                        gate.tensor().buffer(),
                        dims.q_heads,
                        dims.head_dim,
                    )?;
                }
            }
        }
        let q_source_buffer = if dims.attn_output_gate {
            q_raw.tensor().buffer()
        } else if qkv_concat_used {
            qkv_proj_out.tensor().buffer()
        } else {
            q_proj_out.tensor().buffer()
        };
        self.encode_rms_norm_rope_decode_with_offset(
            encoder,
            q_source_buffer,
            0,
            weights.q_norm,
            q_roped.tensor().buffer(),
            dims.q_heads,
            dims.head_dim,
            dims.rope_dims,
            dims.rope_frequency_dim,
            dims.position,
            dims.eps,
            dims.theta,
        )?;
        Self::encode_query_attention_scale(
            executor,
            encoder,
            owned,
            q_roped.tensor().buffer(),
            q_dim,
            dims.head_dim,
            dims.attn_scalar,
        )?;
        if qkv_concat_used {
            let k_offset = byte_offset_f32(q_proj_dim, "full-attn dense k offset")?;
            let v_offset = byte_offset_f32(
                q_proj_dim.checked_add(kv_dim).ok_or_else(|| {
                    InferError::Dimension("full-attn dense v offset déborde".to_string())
                })?,
                "full-attn dense v offset",
            )?;
            self.encode_rms_norm_rope_decode_with_offset(
                encoder,
                qkv_proj_out.tensor().buffer(),
                k_offset,
                weights.k_norm,
                k_roped.tensor().buffer(),
                dims.kv_heads,
                dims.head_dim,
                dims.rope_dims,
                dims.rope_frequency_dim,
                dims.position,
                dims.eps,
                dims.theta,
            )?;
            if weights.value_norm {
                executor.encode_copy_with_offsets(
                    encoder,
                    qkv_proj_out.tensor().buffer(),
                    v_offset,
                    v_raw.tensor().buffer(),
                    0,
                    kv_dim,
                )?;
                executor.encode_rms_norm_heads_no_scale_rows(
                    encoder,
                    v_raw.tensor().buffer(),
                    v_normed.tensor().buffer(),
                    1,
                    dims.kv_heads,
                    dims.head_dim,
                    dims.eps,
                )?;
                kv.encode_append_kv(
                    encoder,
                    k_roped.tensor().buffer(),
                    v_normed.tensor().buffer(),
                )?;
            } else {
                kv.encode_append_kv_with_offsets(
                    encoder,
                    k_roped.tensor().buffer(),
                    0,
                    qkv_proj_out.tensor().buffer(),
                    v_offset,
                )?;
            }
        } else {
            self.encode_rms_norm_rope_decode(
                encoder,
                k_raw.tensor().buffer(),
                weights.k_norm,
                k_roped.tensor().buffer(),
                dims.kv_heads,
                dims.head_dim,
                dims.rope_dims,
                dims.rope_frequency_dim,
                dims.position,
                dims.eps,
                dims.theta,
            )?;
            if weights.value_norm {
                executor.encode_rms_norm_heads_no_scale_rows(
                    encoder,
                    v_raw.tensor().buffer(),
                    v_normed.tensor().buffer(),
                    1,
                    dims.kv_heads,
                    dims.head_dim,
                    dims.eps,
                )?;
                kv.encode_append_kv(
                    encoder,
                    k_roped.tensor().buffer(),
                    v_normed.tensor().buffer(),
                )?;
            } else {
                kv.encode_append_kv(encoder, k_roped.tensor().buffer(), v_raw.tensor().buffer())?;
            }
        }
        kv.encode_attention_decode_windowed(
            encoder,
            q_roped.tensor().buffer(),
            scores.tensor().buffer(),
            ctx.tensor().buffer(),
            dims.window_start,
        )?;
        if dims.attn_output_gate {
            let o_proj_fused = executor
                .encode_full_attn_o_proj_gated_buffers(
                    encoder,
                    ctx.tensor().buffer(),
                    gate.tensor().buffer(),
                    q_dim,
                    weights.o_proj,
                    o_out.tensor().buffer(),
                )?
                .is_some();
            if !o_proj_fused {
                self.encode_attn_gate(
                    encoder,
                    ctx.tensor().buffer(),
                    gate.tensor().buffer(),
                    gated_ctx.tensor().buffer(),
                    q_dim,
                )?;
                executor.encode_matmul_weight_buffers(
                    encoder,
                    gated_ctx.tensor().buffer(),
                    1,
                    q_dim,
                    weights.o_proj,
                    o_out.tensor().buffer(),
                    false,
                )?;
            }
        } else {
            executor.encode_matmul_weight_buffers(
                encoder,
                ctx.tensor().buffer(),
                1,
                q_dim,
                weights.o_proj,
                o_out.tensor().buffer(),
                false,
            )?;
        }
        if weights.pre_feedforward_norm.is_some() {
            executor.encode_rms_norm_rows(
                encoder,
                o_out.tensor().buffer(),
                weights.post_norm,
                post_normed.tensor().buffer(),
                1,
                hidden,
                dims.eps,
            )?;
            executor.encode_add_scaled(
                encoder,
                owned,
                layer_in,
                post_normed.tensor().buffer(),
                summed.tensor().buffer(),
                1.0,
                hidden,
            )?;
        } else {
            executor.encode_add_rms_norm_rows(
                encoder,
                layer_in,
                o_out.tensor().buffer(),
                weights.post_norm,
                summed.tensor().buffer(),
                post_normed.tensor().buffer(),
                1,
                hidden,
                dims.eps,
            )?;
        }
        if let Some(parallel_moe) = weights.parallel_moe {
            return self.encode_gemma_parallel_moe_tail(
                executor,
                encoder,
                owned,
                summed.tensor().buffer(),
                layer_out,
                1,
                hidden,
                dims.eps,
                parallel_moe,
            );
        }
        match (weights.pre_feedforward_norm, weights.post_feedforward_norm) {
            (Some(pre_feedforward_norm), Some(post_feedforward_norm)) => self
                .encode_gemma_dense_tail(
                    executor,
                    encoder,
                    owned,
                    summed.tensor().buffer(),
                    layer_out,
                    1,
                    hidden,
                    dims.eps,
                    weights.gate_proj,
                    weights.up_proj,
                    weights.down_proj,
                    GemmaDenseTailWeights {
                        pre_feedforward_norm,
                        post_feedforward_norm,
                        layer_scalar: weights.layer_scalar,
                    },
                ),
            (None, None) => self.encode_dense_tail(
                executor,
                encoder,
                owned,
                post_normed.tensor().buffer(),
                summed.tensor().buffer(),
                layer_out,
                hidden,
                weights.gate_proj,
                weights.up_proj,
                weights.down_proj,
                weights.tail_score,
            ),
            _ => Err(InferError::Config(
                "normes feed-forward Gemma résidentes partielles".to_string(),
            )),
        }
    }

    /// Encode une couche full-attn DENSE sur `rows` positions contiguës.
    ///
    /// Les projections QKV/O et le MLP dense sont batchés. Le RoPE, l'append KV
    /// et l'attention restent strictement ordonnés par position pour préserver le
    /// masque causal et l'état KV résident.
    #[expect(
        clippy::too_many_arguments,
        reason = "data flow batché d'une couche full-attn dense"
    )]
    pub(crate) fn encode_full_attn_dense_layer_rows(
        &self,
        executor: &MetalExecutor,
        encoder: &ComputeCommandEncoderRef,
        owned: &mut Vec<Buffer>,
        kv: &mut FullAttentionMetalState,
        weights: FullAttnDenseLayerWeights<'_>,
        dims: FullAttnLayerDims,
        rows: usize,
        layer_in: &BufferRef,
        layer_out: &BufferRef,
    ) -> Result<()> {
        if rows == 1 {
            return self.encode_full_attn_dense_layer(
                executor, encoder, owned, kv, weights, dims, layer_in, layer_out,
            );
        }
        let hidden = dims.hidden;
        let q_dim = dims.q_heads.checked_mul(dims.head_dim).ok_or_else(|| {
            InferError::Dimension("full-attn dense rows q_dim déborde".to_string())
        })?;
        let kv_dim = dims.kv_heads.checked_mul(dims.head_dim).ok_or_else(|| {
            InferError::Dimension("full-attn dense rows kv_dim déborde".to_string())
        })?;
        let q_proj_dim = if dims.attn_output_gate {
            q_dim.checked_mul(2).ok_or_else(|| {
                InferError::Dimension("full-attn dense rows q_gate_dim déborde".to_string())
            })?
        } else {
            q_dim
        };
        if !dims.attn_output_gate {
            return Err(InferError::Config(
                "full-attn dense batché attend attn_output_gate=true".to_string(),
            ));
        }
        let qkv_proj_dim = q_proj_dim
            .checked_add(kv_dim)
            .and_then(|value| value.checked_add(kv_dim))
            .ok_or_else(|| {
                InferError::Dimension("full-attn dense rows qkv concat déborde".to_string())
            })?;
        let batch_hidden = rows.checked_mul(hidden).ok_or_else(|| {
            InferError::Dimension("full-attn dense rows hidden déborde".to_string())
        })?;
        let batch_qkv = rows
            .checked_mul(qkv_proj_dim)
            .ok_or_else(|| InferError::Dimension("full-attn dense rows qkv déborde".to_string()))?;
        let batch_q_proj = rows.checked_mul(q_proj_dim).ok_or_else(|| {
            InferError::Dimension("full-attn dense rows q_proj déborde".to_string())
        })?;
        let batch_kv = rows
            .checked_mul(kv_dim)
            .ok_or_else(|| InferError::Dimension("full-attn dense rows kv déborde".to_string()))?;
        let batch_q = rows
            .checked_mul(q_dim)
            .ok_or_else(|| InferError::Dimension("full-attn dense rows q déborde".to_string()))?;
        let score_cells = dims.q_heads.checked_mul(kv.capacity()).ok_or_else(|| {
            InferError::Dimension("full-attn dense rows scores débordent".to_string())
        })?;

        let normed = self.scratch().lease(batch_hidden, GpuElement::F32)?;
        let qkv_proj_out = self.scratch().lease(batch_qkv, GpuElement::F32)?;
        let q_proj_out = self.scratch().lease(batch_q_proj, GpuElement::F32)?;
        let k_proj_out = self.scratch().lease(batch_kv, GpuElement::F32)?;
        let v_proj_out = self.scratch().lease(batch_kv, GpuElement::F32)?;
        let v_row = self.scratch().lease(kv_dim, GpuElement::F32)?;
        let v_normed = self.scratch().lease(kv_dim, GpuElement::F32)?;
        let q_raw = self.scratch().lease(batch_q, GpuElement::F32)?;
        let gate = self.scratch().lease(batch_q, GpuElement::F32)?;
        let q_roped = self.scratch().lease(q_dim, GpuElement::F32)?;
        let k_roped = self.scratch().lease(kv_dim, GpuElement::F32)?;
        let scores = self.scratch().lease(score_cells, GpuElement::F32)?;
        let ctx_row = self.scratch().lease(q_dim, GpuElement::F32)?;
        let ctx = self.scratch().lease(batch_q, GpuElement::F32)?;
        let gated_ctx = self.scratch().lease(batch_q, GpuElement::F32)?;
        let o_out = self.scratch().lease(batch_hidden, GpuElement::F32)?;
        let summed = self.scratch().lease(batch_hidden, GpuElement::F32)?;
        let post_normed = self.scratch().lease(batch_hidden, GpuElement::F32)?;

        executor.encode_rms_norm_rows(
            encoder,
            layer_in,
            weights.input_norm,
            normed.tensor().buffer(),
            rows,
            hidden,
            dims.eps,
        )?;
        if let Some(qkv_proj) = weights.qkv_proj {
            let out_dim = executor.encode_matmul_weight_buffers(
                encoder,
                normed.tensor().buffer(),
                rows,
                hidden,
                qkv_proj,
                qkv_proj_out.tensor().buffer(),
                false,
            )?;
            if out_dim != qkv_proj_dim {
                return Err(InferError::Dimension(format!(
                    "full-attn dense rows qkv sort {out_dim}, attendu {qkv_proj_dim}"
                )));
            }
        } else {
            let q_out_dim = executor.encode_matmul_weight_buffers(
                encoder,
                normed.tensor().buffer(),
                rows,
                hidden,
                weights.q_proj,
                q_proj_out.tensor().buffer(),
                false,
            )?;
            if q_out_dim != q_proj_dim {
                return Err(InferError::Dimension(format!(
                    "full-attn dense rows q_proj sort {q_out_dim}, attendu {q_proj_dim}"
                )));
            }
            let k_out_dim = executor.encode_matmul_weight_buffers(
                encoder,
                normed.tensor().buffer(),
                rows,
                hidden,
                weights.k_proj,
                k_proj_out.tensor().buffer(),
                false,
            )?;
            if k_out_dim != kv_dim {
                return Err(InferError::Dimension(format!(
                    "full-attn dense rows k_proj sort {k_out_dim}, attendu {kv_dim}"
                )));
            }
            let v_out_dim = executor.encode_matmul_weight_buffers(
                encoder,
                normed.tensor().buffer(),
                rows,
                hidden,
                weights.v_proj,
                v_proj_out.tensor().buffer(),
                false,
            )?;
            if v_out_dim != kv_dim {
                return Err(InferError::Dimension(format!(
                    "full-attn dense rows v_proj sort {v_out_dim}, attendu {kv_dim}"
                )));
            }
        }

        executor.encode_split_q_gate_rows_with_stride(
            encoder,
            if weights.qkv_proj.is_some() {
                qkv_proj_out.tensor().buffer()
            } else {
                q_proj_out.tensor().buffer()
            },
            q_raw.tensor().buffer(),
            gate.tensor().buffer(),
            rows,
            dims.q_heads,
            dims.head_dim,
            if weights.qkv_proj.is_some() {
                qkv_proj_dim
            } else {
                q_proj_dim
            },
        )?;

        let k_base = q_proj_dim;
        let v_base = q_proj_dim.checked_add(kv_dim).ok_or_else(|| {
            InferError::Dimension("full-attn dense rows v base déborde".to_string())
        })?;
        for row in 0..rows {
            let q_row_offset = row
                .checked_mul(q_dim)
                .ok_or_else(|| InferError::Dimension("full-attn dense q row déborde".to_string()))
                .and_then(|offset| byte_offset_f32(offset, "full-attn dense q row offset"))?;
            self.encode_rms_norm_rope_decode_with_offset(
                encoder,
                q_raw.tensor().buffer(),
                q_row_offset,
                weights.q_norm,
                q_roped.tensor().buffer(),
                dims.q_heads,
                dims.head_dim,
                dims.rope_dims,
                dims.rope_frequency_dim,
                dims.position + row,
                dims.eps,
                dims.theta,
            )?;
            Self::encode_query_attention_scale(
                executor,
                encoder,
                owned,
                q_roped.tensor().buffer(),
                q_dim,
                dims.head_dim,
                dims.attn_scalar,
            )?;
            let k_offset = if weights.qkv_proj.is_some() {
                let qkv_row = row.checked_mul(qkv_proj_dim).ok_or_else(|| {
                    InferError::Dimension("full-attn dense rows offset déborde".to_string())
                })?;
                byte_offset_f32(
                    qkv_row.checked_add(k_base).ok_or_else(|| {
                        InferError::Dimension("full-attn dense rows k offset déborde".to_string())
                    })?,
                    "full-attn dense rows k offset",
                )?
            } else {
                let k_row = row.checked_mul(kv_dim).ok_or_else(|| {
                    InferError::Dimension("full-attn dense rows split k offset déborde".to_string())
                })?;
                byte_offset_f32(k_row, "full-attn dense rows split k offset")?
            };
            self.encode_rms_norm_rope_decode_with_offset(
                encoder,
                if weights.qkv_proj.is_some() {
                    qkv_proj_out.tensor().buffer()
                } else {
                    k_proj_out.tensor().buffer()
                },
                k_offset,
                weights.k_norm,
                k_roped.tensor().buffer(),
                dims.kv_heads,
                dims.head_dim,
                dims.rope_dims,
                dims.rope_frequency_dim,
                dims.position + row,
                dims.eps,
                dims.theta,
            )?;
            let v_offset = if weights.qkv_proj.is_some() {
                let qkv_row = row.checked_mul(qkv_proj_dim).ok_or_else(|| {
                    InferError::Dimension("full-attn dense rows offset déborde".to_string())
                })?;
                byte_offset_f32(
                    qkv_row.checked_add(v_base).ok_or_else(|| {
                        InferError::Dimension("full-attn dense rows v offset déborde".to_string())
                    })?,
                    "full-attn dense rows v offset",
                )?
            } else {
                let v_row = row.checked_mul(kv_dim).ok_or_else(|| {
                    InferError::Dimension("full-attn dense rows split v offset déborde".to_string())
                })?;
                byte_offset_f32(v_row, "full-attn dense rows split v offset")?
            };
            let v_source = if weights.qkv_proj.is_some() {
                qkv_proj_out.tensor().buffer()
            } else {
                v_proj_out.tensor().buffer()
            };
            if weights.value_norm {
                executor.encode_copy_with_offsets(
                    encoder,
                    v_source,
                    v_offset,
                    v_row.tensor().buffer(),
                    0,
                    kv_dim,
                )?;
                executor.encode_rms_norm_heads_no_scale_rows(
                    encoder,
                    v_row.tensor().buffer(),
                    v_normed.tensor().buffer(),
                    1,
                    dims.kv_heads,
                    dims.head_dim,
                    dims.eps,
                )?;
                kv.encode_append_kv(
                    encoder,
                    k_roped.tensor().buffer(),
                    v_normed.tensor().buffer(),
                )?;
            } else {
                kv.encode_append_kv_with_offsets(
                    encoder,
                    k_roped.tensor().buffer(),
                    0,
                    v_source,
                    v_offset,
                )?;
            }
            kv.encode_attention_decode_windowed(
                encoder,
                q_roped.tensor().buffer(),
                scores.tensor().buffer(),
                ctx_row.tensor().buffer(),
                dims.window_start,
            )?;
            let gated_offset = row
                .checked_mul(q_dim)
                .ok_or_else(|| {
                    InferError::Dimension("full-attn dense rows gated ctx déborde".to_string())
                })
                .and_then(|offset| {
                    byte_offset_f32(offset, "full-attn dense rows gated ctx offset")
                })?;
            executor.encode_copy_with_offsets(
                encoder,
                ctx_row.tensor().buffer(),
                0,
                ctx.tensor().buffer(),
                gated_offset,
                q_dim,
            )?;
        }
        executor.encode_attn_gate_rows(
            encoder,
            ctx.tensor().buffer(),
            gate.tensor().buffer(),
            gated_ctx.tensor().buffer(),
            batch_q,
        )?;
        let o_dim = executor.encode_matmul_weight_buffers(
            encoder,
            gated_ctx.tensor().buffer(),
            rows,
            q_dim,
            weights.o_proj,
            o_out.tensor().buffer(),
            false,
        )?;
        if o_dim != hidden {
            return Err(InferError::Dimension(format!(
                "full-attn dense rows o_proj sort {o_dim}, attendu {hidden}"
            )));
        }
        if weights.pre_feedforward_norm.is_some() {
            executor.encode_rms_norm_rows(
                encoder,
                o_out.tensor().buffer(),
                weights.post_norm,
                post_normed.tensor().buffer(),
                rows,
                hidden,
                dims.eps,
            )?;
            executor.encode_add_scaled(
                encoder,
                owned,
                layer_in,
                post_normed.tensor().buffer(),
                summed.tensor().buffer(),
                1.0,
                batch_hidden,
            )?;
        } else {
            executor.encode_add_rms_norm_rows(
                encoder,
                layer_in,
                o_out.tensor().buffer(),
                weights.post_norm,
                summed.tensor().buffer(),
                post_normed.tensor().buffer(),
                rows,
                hidden,
                dims.eps,
            )?;
        }
        if let Some(parallel_moe) = weights.parallel_moe {
            return self.encode_gemma_parallel_moe_tail(
                executor,
                encoder,
                owned,
                summed.tensor().buffer(),
                layer_out,
                rows,
                hidden,
                dims.eps,
                parallel_moe,
            );
        }
        match (weights.pre_feedforward_norm, weights.post_feedforward_norm) {
            (Some(pre_feedforward_norm), Some(post_feedforward_norm)) => self
                .encode_gemma_dense_tail(
                    executor,
                    encoder,
                    owned,
                    summed.tensor().buffer(),
                    layer_out,
                    rows,
                    hidden,
                    dims.eps,
                    weights.gate_proj,
                    weights.up_proj,
                    weights.down_proj,
                    GemmaDenseTailWeights {
                        pre_feedforward_norm,
                        post_feedforward_norm,
                        layer_scalar: weights.layer_scalar,
                    },
                ),
            (None, None) => self.encode_dense_tail_rows(
                executor,
                encoder,
                owned,
                post_normed.tensor().buffer(),
                summed.tensor().buffer(),
                layer_out,
                rows,
                hidden,
                weights.gate_proj,
                weights.up_proj,
                weights.down_proj,
                weights.tail_score,
            ),
            _ => Err(InferError::Config(
                "normes feed-forward Gemma résidentes partielles".to_string(),
            )),
        }
    }

    /// Encode UNE couche linear-attn (decode résident 1c) dans l'encoder PARTAGÉ :
    /// `layer_in [hidden]` → `layer_out [hidden]`, sans commit ni readback.
    /// Reproduit `linear.forward_cached_with_runtime` + le tail MoE/shared
    /// (decoder.rs) sur GPU : `rms_norm` → [`MetalExecutor::encode_linear_attn_resident`]
    /// (conv/ssm résidents) → résiduel+post_norm fusionnés → MoE+shared vers
    /// `layer_out`.
    ///
    /// MAJEUR 5 (race SSM) : le kernel `linear_attn_gated_delta_f32` partitionne
    /// l'état SSM par cellule (un bloc disjoint par `value_index`, une cellule par
    /// lane) → aucun read-after-write inter-thread ; le RAW phase1→phase2 est
    /// intra-lane (ordre-programme). Même discipline disjointe que le conv. Pas de
    /// barrière nécessaire ; l'état n'est relu qu'au token suivant (CB séparé).
    ///
    /// # Errors
    ///
    /// Propage toute erreur d'encodage (dimension, Metal).
    #[expect(
        clippy::too_many_arguments,
        reason = "data flow d'une couche : exécuteur + état conv/ssm + poids + spec + ping-pong"
    )]
    pub(crate) fn encode_linear_attn_layer(
        &self,
        executor: &MetalExecutor,
        encoder: &ComputeCommandEncoderRef,
        owned: &mut Vec<Buffer>,
        state: &LinearAttentionMetalState,
        weights: LinearAttnLayerWeights<'_>,
        spec: LinearAttentionStepSpec,
        res_dims: LinearAttnResidentDims,
        hidden: usize,
        eps: f32,
        layer_in: &BufferRef,
        layer_out: &BufferRef,
    ) -> Result<()> {
        // Scratch d'UNE couche (réutilise les slots `hidden` des couches full-attn).
        let attn_out = self.scratch().lease(hidden, GpuElement::F32)?;
        let summed = self.scratch().lease(hidden, GpuElement::F32)?;
        let post_normed = self.scratch().lease(hidden, GpuElement::F32)?;

        executor.encode_linear_attn_resident_dense_buffers(
            encoder,
            layer_in,
            Some((weights.input_norm, eps)),
            attn_out.tensor().buffer(),
            weights.linear,
            state,
            spec,
            res_dims,
        )?;
        executor.encode_add_rms_norm_rows(
            encoder,
            layer_in,
            attn_out.tensor().buffer(),
            weights.post_norm,
            summed.tensor().buffer(),
            post_normed.tensor().buffer(),
            1,
            hidden,
            eps,
        )?;
        executor.encode_moe_shared_buffers(
            encoder,
            owned,
            post_normed.tensor().buffer(),
            Some(summed.tensor().buffer()),
            layer_out,
            hidden,
            weights.moe,
            weights.top_k,
        )?;
        Ok(())
    }

    /// Encode une couche linear-attn MoE sur `rows` positions contiguës.
    ///
    /// Les projections linear-attn sont batchées et le scan conv/SSM conserve
    /// l'ordre temporel. Le tail MoE shared utilise un routeur/top-k par ligne
    /// puis des gather sur `rows * top_k` slots.
    #[expect(
        clippy::too_many_arguments,
        reason = "data flow batché d'une couche linear-attn MoE résidente"
    )]
    pub(crate) fn encode_linear_attn_layer_rows(
        &self,
        executor: &MetalExecutor,
        encoder: &ComputeCommandEncoderRef,
        owned: &mut Vec<Buffer>,
        state: &LinearAttentionMetalState,
        weights: LinearAttnLayerWeights<'_>,
        spec: LinearAttentionStepSpec,
        res_dims: LinearAttnResidentDims,
        rows: usize,
        hidden: usize,
        eps: f32,
        layer_in: &BufferRef,
        layer_out: &BufferRef,
        captures: Option<&[LinearAttentionMetalState]>,
    ) -> Result<()> {
        if rows == 1 && captures.is_none() {
            return self.encode_linear_attn_layer(
                executor, encoder, owned, state, weights, spec, res_dims, hidden, eps, layer_in,
                layer_out,
            );
        }
        let elements = rows.checked_mul(hidden).ok_or_else(|| {
            InferError::Dimension("linear-attn MoE rows hidden déborde".to_string())
        })?;
        let attn_out = self.scratch().lease(elements, GpuElement::F32)?;
        let summed = self.scratch().lease(elements, GpuElement::F32)?;
        let post_normed = self.scratch().lease(elements, GpuElement::F32)?;

        executor.encode_linear_attn_resident_dense_buffers_rows(
            encoder,
            layer_in,
            Some((weights.input_norm, eps)),
            attn_out.tensor().buffer(),
            rows,
            weights.linear,
            state,
            captures,
            spec,
            res_dims,
        )?;
        executor.encode_add_rms_norm_rows(
            encoder,
            layer_in,
            attn_out.tensor().buffer(),
            weights.post_norm,
            summed.tensor().buffer(),
            post_normed.tensor().buffer(),
            rows,
            hidden,
            eps,
        )?;
        executor.encode_moe_shared_buffers_rows(
            encoder,
            owned,
            post_normed.tensor().buffer(),
            Some(summed.tensor().buffer()),
            layer_out,
            rows,
            hidden,
            weights.moe,
            weights.top_k,
        )
    }

    /// Encode UNE couche linear-attn DENSE (decode résident 1c) dans l'encoder
    /// partagé.
    #[expect(
        clippy::too_many_arguments,
        reason = "data flow d'une couche : exécuteur + état conv/ssm + poids + spec + ping-pong"
    )]
    pub(crate) fn encode_linear_attn_dense_layer(
        &self,
        executor: &MetalExecutor,
        encoder: &ComputeCommandEncoderRef,
        owned: &mut Vec<Buffer>,
        state: &LinearAttentionMetalState,
        weights: LinearAttnDenseLayerWeights<'_>,
        spec: LinearAttentionStepSpec,
        res_dims: LinearAttnResidentDims,
        hidden: usize,
        eps: f32,
        layer_in: &BufferRef,
        layer_out: &BufferRef,
    ) -> Result<()> {
        let attn_out = self.scratch().lease(hidden, GpuElement::F32)?;
        let summed = self.scratch().lease(hidden, GpuElement::F32)?;
        let post_normed = self.scratch().lease(hidden, GpuElement::F32)?;

        executor.encode_linear_attn_resident_dense_buffers(
            encoder,
            layer_in,
            Some((weights.input_norm, eps)),
            attn_out.tensor().buffer(),
            weights.linear,
            state,
            spec,
            res_dims,
        )?;
        executor.encode_add_rms_norm_rows(
            encoder,
            layer_in,
            attn_out.tensor().buffer(),
            weights.post_norm,
            summed.tensor().buffer(),
            post_normed.tensor().buffer(),
            1,
            hidden,
            eps,
        )?;
        self.encode_dense_tail(
            executor,
            encoder,
            owned,
            post_normed.tensor().buffer(),
            summed.tensor().buffer(),
            layer_out,
            hidden,
            weights.gate_proj,
            weights.up_proj,
            weights.down_proj,
            weights.tail_score,
        )
    }

    /// Encode une couche linear-attn DENSE sur `rows` positions contiguës.
    ///
    /// Les projections linear-attn et le MLP dense sont batchés; le scan
    /// conv/SSM reste séquentiel dans [`MetalExecutor`] pour conserver l'ordre
    /// temporel exact des états résidents.
    #[expect(
        clippy::too_many_arguments,
        reason = "data flow batché d'une couche dense résidente"
    )]
    pub(crate) fn encode_linear_attn_dense_layer_rows(
        &self,
        executor: &MetalExecutor,
        encoder: &ComputeCommandEncoderRef,
        owned: &mut Vec<Buffer>,
        state: &LinearAttentionMetalState,
        weights: LinearAttnDenseLayerWeights<'_>,
        spec: LinearAttentionStepSpec,
        res_dims: LinearAttnResidentDims,
        rows: usize,
        hidden: usize,
        eps: f32,
        layer_in: &BufferRef,
        layer_out: &BufferRef,
        captures: Option<&[LinearAttentionMetalState]>,
    ) -> Result<()> {
        if rows == 1 {
            if captures.is_none() {
                return self.encode_linear_attn_dense_layer(
                    executor, encoder, owned, state, weights, spec, res_dims, hidden, eps,
                    layer_in, layer_out,
                );
            }
            let elements = rows.checked_mul(hidden).ok_or_else(|| {
                InferError::Dimension("linear-attn dense rows hidden déborde".to_string())
            })?;
            let attn_out = self.scratch().lease(elements, GpuElement::F32)?;
            let summed = self.scratch().lease(elements, GpuElement::F32)?;
            let post_normed = self.scratch().lease(elements, GpuElement::F32)?;
            executor.encode_linear_attn_resident_dense_buffers_rows(
                encoder,
                layer_in,
                Some((weights.input_norm, eps)),
                attn_out.tensor().buffer(),
                rows,
                weights.linear,
                state,
                captures,
                spec,
                res_dims,
            )?;
            executor.encode_add_rms_norm_rows(
                encoder,
                layer_in,
                attn_out.tensor().buffer(),
                weights.post_norm,
                summed.tensor().buffer(),
                post_normed.tensor().buffer(),
                rows,
                hidden,
                eps,
            )?;
            return self.encode_dense_tail_rows(
                executor,
                encoder,
                owned,
                post_normed.tensor().buffer(),
                summed.tensor().buffer(),
                layer_out,
                rows,
                hidden,
                weights.gate_proj,
                weights.up_proj,
                weights.down_proj,
                weights.tail_score,
            );
        }
        let elements = rows.checked_mul(hidden).ok_or_else(|| {
            InferError::Dimension("linear-attn dense rows hidden déborde".to_string())
        })?;
        let attn_out = self.scratch().lease(elements, GpuElement::F32)?;
        let summed = self.scratch().lease(elements, GpuElement::F32)?;
        let post_normed = self.scratch().lease(elements, GpuElement::F32)?;

        executor.encode_linear_attn_resident_dense_buffers_rows(
            encoder,
            layer_in,
            Some((weights.input_norm, eps)),
            attn_out.tensor().buffer(),
            rows,
            weights.linear,
            state,
            captures,
            spec,
            res_dims,
        )?;
        executor.encode_add_rms_norm_rows(
            encoder,
            layer_in,
            attn_out.tensor().buffer(),
            weights.post_norm,
            summed.tensor().buffer(),
            post_normed.tensor().buffer(),
            rows,
            hidden,
            eps,
        )?;
        self.encode_dense_tail_rows(
            executor,
            encoder,
            owned,
            post_normed.tensor().buffer(),
            summed.tensor().buffer(),
            layer_out,
            rows,
            hidden,
            weights.gate_proj,
            weights.up_proj,
            weights.down_proj,
            weights.tail_score,
        )
    }
}
