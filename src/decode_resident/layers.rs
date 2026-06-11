//! Encodage des couches full-attn, linear-attn et MoE résidentes.

use super::utils::byte_offset_f32;
use super::*;

impl DecodeResidentState {
    /// Encode UNE couche full-attn (decode résident 1c) à la position
    /// `dims.position` dans l'encoder PARTAGÉ : `layer_in [hidden]` → `layer_out
    /// [hidden]`, sans commit ni readback. Reproduit entièrement sur GPU le couple
    /// `full_attention_context_cached` puis `full_attention_tail_moe_shared`
    /// (decoder.rs / metal_backend.rs).
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
            // input rms_norm
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
            }
            if !qkv_epilogue_fused {
                // désinterleave le gate AVANT le RoPE (attn_output_gate=true)
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
        let k_offset = byte_offset_f32(q_gate_dim, "full-attn k offset")?;
        let v_offset = byte_offset_f32(
            q_gate_dim
                .checked_add(kv_dim)
                .ok_or_else(|| InferError::Dimension("full-attn v offset déborde".to_string()))?,
            "full-attn v offset",
        )?;
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
            dims.position,
            dims.eps,
            dims.theta,
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
            dims.position,
            dims.eps,
            dims.theta,
        )?;
        // append KV device (K roped, V brut) à `len`, puis attention single-query (R3)
        kv.encode_append_kv_with_offsets(
            encoder,
            k_roped.tensor().buffer(),
            0,
            qkv_proj_out.tensor().buffer(),
            v_offset,
        )?;
        kv.encode_attention_decode(
            encoder,
            q_roped.tensor().buffer(),
            scores.tensor().buffer(),
            ctx.tensor().buffer(),
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
            dims.position,
            dims.eps,
            dims.theta,
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
        kv.encode_attention_decode(
            encoder,
            q_roped.tensor().buffer(),
            scores.tensor().buffer(),
            ctx.tensor().buffer(),
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
        executor.encode_moe_routed_buffers(
            encoder,
            owned,
            post_normed.tensor().buffer(),
            Some(summed.tensor().buffer()),
            layer_out,
            hidden,
            weights.moe,
            weights.top_k,
        )
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
            dims.position,
            dims.eps,
            dims.theta,
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
                dims.position,
                dims.eps,
                dims.theta,
            )?;
            kv.encode_append_kv(encoder, k_roped.tensor().buffer(), v_raw.tensor().buffer())?;
        }
        kv.encode_attention_decode(
            encoder,
            q_roped.tensor().buffer(),
            scores.tensor().buffer(),
            ctx.tensor().buffer(),
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
        let Some(qkv_proj) = weights.qkv_proj else {
            return Err(InferError::Config(
                "qkv concat requis pour full-attn dense batché".to_string(),
            ));
        };
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
        let batch_q = rows
            .checked_mul(q_dim)
            .ok_or_else(|| InferError::Dimension("full-attn dense rows q déborde".to_string()))?;
        let score_cells = dims.q_heads.checked_mul(kv.capacity()).ok_or_else(|| {
            InferError::Dimension("full-attn dense rows scores débordent".to_string())
        })?;

        let normed = self.scratch().lease(batch_hidden, GpuElement::F32)?;
        let qkv_proj_out = self.scratch().lease(batch_qkv, GpuElement::F32)?;
        let q_raw = self.scratch().lease(q_dim, GpuElement::F32)?;
        let gate_row = self.scratch().lease(q_dim, GpuElement::F32)?;
        let q_roped = self.scratch().lease(q_dim, GpuElement::F32)?;
        let k_roped = self.scratch().lease(kv_dim, GpuElement::F32)?;
        let scores = self.scratch().lease(score_cells, GpuElement::F32)?;
        let ctx = self.scratch().lease(q_dim, GpuElement::F32)?;
        let gated_ctx_row = self.scratch().lease(q_dim, GpuElement::F32)?;
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

        let k_base = q_proj_dim;
        let v_base = q_proj_dim.checked_add(kv_dim).ok_or_else(|| {
            InferError::Dimension("full-attn dense rows v base déborde".to_string())
        })?;
        for row in 0..rows {
            let qkv_row = row.checked_mul(qkv_proj_dim).ok_or_else(|| {
                InferError::Dimension("full-attn dense rows offset déborde".to_string())
            })?;
            let qkv_offset = byte_offset_f32(qkv_row, "full-attn dense rows q offset")?;
            self.encode_split_q_gate_with_offset(
                encoder,
                qkv_proj_out.tensor().buffer(),
                qkv_offset,
                q_raw.tensor().buffer(),
                gate_row.tensor().buffer(),
                dims.q_heads,
                dims.head_dim,
            )?;
            self.encode_rms_norm_rope_decode_with_offset(
                encoder,
                q_raw.tensor().buffer(),
                0,
                weights.q_norm,
                q_roped.tensor().buffer(),
                dims.q_heads,
                dims.head_dim,
                dims.rope_dims,
                dims.position + row,
                dims.eps,
                dims.theta,
            )?;
            let k_offset = byte_offset_f32(
                qkv_row.checked_add(k_base).ok_or_else(|| {
                    InferError::Dimension("full-attn dense rows k offset déborde".to_string())
                })?,
                "full-attn dense rows k offset",
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
                dims.position + row,
                dims.eps,
                dims.theta,
            )?;
            let v_offset = byte_offset_f32(
                qkv_row.checked_add(v_base).ok_or_else(|| {
                    InferError::Dimension("full-attn dense rows v offset déborde".to_string())
                })?,
                "full-attn dense rows v offset",
            )?;
            kv.encode_append_kv_with_offsets(
                encoder,
                k_roped.tensor().buffer(),
                0,
                qkv_proj_out.tensor().buffer(),
                v_offset,
            )?;
            kv.encode_attention_decode(
                encoder,
                q_roped.tensor().buffer(),
                scores.tensor().buffer(),
                ctx.tensor().buffer(),
            )?;
            self.encode_attn_gate(
                encoder,
                ctx.tensor().buffer(),
                gate_row.tensor().buffer(),
                gated_ctx_row.tensor().buffer(),
                q_dim,
            )?;
            let gated_offset =
                byte_offset_f32(row * q_dim, "full-attn dense rows gated ctx offset")?;
            executor.encode_copy_with_offsets(
                encoder,
                gated_ctx_row.tensor().buffer(),
                0,
                gated_ctx.tensor().buffer(),
                gated_offset,
                q_dim,
            )?;
        }
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

        executor.encode_linear_attn_resident_buffers(
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

    #[expect(
        clippy::too_many_arguments,
        reason = "tail dense résident: buffers + poids nécessaires à l'encodage"
    )]
    fn encode_dense_tail(
        &self,
        executor: &MetalExecutor,
        encoder: &ComputeCommandEncoderRef,
        owned: &mut Vec<Buffer>,
        post_normed: &BufferRef,
        summed: &BufferRef,
        layer_out: &BufferRef,
        hidden: usize,
        gate_proj: &MetalLinearWeightBuffers,
        up_proj: &MetalLinearWeightBuffers,
        down_proj: &MetalLinearWeightBuffers,
        tail_score: &BufferRef,
    ) -> Result<()> {
        let gate_in = executor.linear_weight_in_dim(gate_proj);
        let inter = executor.linear_weight_out_dim(gate_proj);
        if gate_in != hidden {
            return Err(InferError::Dimension(format!(
                "dense gate_proj attendu [inter,{hidden}], reçu [{inter},{gate_in}]"
            )));
        }
        let up_out = executor.linear_weight_out_dim(up_proj);
        let up_in = executor.linear_weight_in_dim(up_proj);
        if up_out != inter || up_in != hidden {
            return Err(InferError::Dimension(format!(
                "dense up_proj attendu [{inter},{hidden}], reçu [{up_out},{up_in}]"
            )));
        }
        let down_out = executor.linear_weight_out_dim(down_proj);
        let down_in = executor.linear_weight_in_dim(down_proj);
        if down_out != hidden || down_in != inter {
            return Err(InferError::Dimension(format!(
                "dense down_proj attendu [{hidden},{inter}], reçu [{down_out},{down_in}]"
            )));
        }
        let swiglu = self.scratch().lease(inter, GpuElement::F32)?;
        let down = self.scratch().lease(hidden, GpuElement::F32)?;

        if !executor.encode_gate_up_swiglu_fast_buffers(
            encoder,
            post_normed,
            gate_proj,
            up_proj,
            swiglu.tensor().buffer(),
            hidden,
        )? {
            let gate = self.scratch().lease(inter, GpuElement::F32)?;
            let up = self.scratch().lease(inter, GpuElement::F32)?;
            let gate_dim = executor.encode_matmul_weight_buffers(
                encoder,
                post_normed,
                1,
                hidden,
                gate_proj,
                gate.tensor().buffer(),
                false,
            )?;
            let up_dim = executor.encode_matmul_weight_buffers(
                encoder,
                post_normed,
                1,
                hidden,
                up_proj,
                up.tensor().buffer(),
                false,
            )?;
            if gate_dim != inter || up_dim != inter {
                return Err(InferError::Dimension(format!(
                    "dense gate/up sortent gate={gate_dim} up={up_dim}, attendu {inter}"
                )));
            }
            executor.encode_swiglu(
                encoder,
                owned,
                gate.tensor().buffer(),
                up.tensor().buffer(),
                swiglu.tensor().buffer(),
                inter,
            )?;
        }
        let down_dim = executor.encode_matmul_weight_buffers(
            encoder,
            swiglu.tensor().buffer(),
            1,
            inter,
            down_proj,
            down.tensor().buffer(),
            false,
        )?;
        if down_dim != hidden {
            return Err(InferError::Dimension(format!(
                "dense down sort {down_dim}, attendu {hidden}"
            )));
        }
        executor.encode_weighted_sum_add_topk(
            encoder,
            owned,
            down.tensor().buffer(),
            tail_score,
            summed,
            layer_out,
            1,
            hidden,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "tail dense résident batché: buffers + poids nécessaires à l'encodage"
    )]
    fn encode_dense_tail_rows(
        &self,
        executor: &MetalExecutor,
        encoder: &ComputeCommandEncoderRef,
        owned: &mut Vec<Buffer>,
        post_normed: &BufferRef,
        summed: &BufferRef,
        layer_out: &BufferRef,
        rows: usize,
        hidden: usize,
        gate_proj: &MetalLinearWeightBuffers,
        up_proj: &MetalLinearWeightBuffers,
        down_proj: &MetalLinearWeightBuffers,
        tail_score: &BufferRef,
    ) -> Result<()> {
        if rows == 1 {
            return self.encode_dense_tail(
                executor,
                encoder,
                owned,
                post_normed,
                summed,
                layer_out,
                hidden,
                gate_proj,
                up_proj,
                down_proj,
                tail_score,
            );
        }
        let gate_in = executor.linear_weight_in_dim(gate_proj);
        let inter = executor.linear_weight_out_dim(gate_proj);
        if gate_in != hidden {
            return Err(InferError::Dimension(format!(
                "dense gate_proj attendu [inter,{hidden}], reçu [{inter},{gate_in}]"
            )));
        }
        let up_out = executor.linear_weight_out_dim(up_proj);
        let up_in = executor.linear_weight_in_dim(up_proj);
        if up_out != inter || up_in != hidden {
            return Err(InferError::Dimension(format!(
                "dense up_proj attendu [{inter},{hidden}], reçu [{up_out},{up_in}]"
            )));
        }
        let down_out = executor.linear_weight_out_dim(down_proj);
        let down_in = executor.linear_weight_in_dim(down_proj);
        if down_out != hidden || down_in != inter {
            return Err(InferError::Dimension(format!(
                "dense down_proj attendu [{hidden},{inter}], reçu [{down_out},{down_in}]"
            )));
        }

        let inter_elements = rows
            .checked_mul(inter)
            .ok_or_else(|| InferError::Dimension("dense rows inter déborde".to_string()))?;
        let hidden_elements = rows
            .checked_mul(hidden)
            .ok_or_else(|| InferError::Dimension("dense rows hidden déborde".to_string()))?;
        let gate = self.scratch().lease(inter_elements, GpuElement::F32)?;
        let up = self.scratch().lease(inter_elements, GpuElement::F32)?;
        let swiglu = self.scratch().lease(inter_elements, GpuElement::F32)?;
        let down = self.scratch().lease(hidden_elements, GpuElement::F32)?;

        let gate_dim = executor.encode_matmul_weight_buffers(
            encoder,
            post_normed,
            rows,
            hidden,
            gate_proj,
            gate.tensor().buffer(),
            false,
        )?;
        let up_dim = executor.encode_matmul_weight_buffers(
            encoder,
            post_normed,
            rows,
            hidden,
            up_proj,
            up.tensor().buffer(),
            false,
        )?;
        if gate_dim != inter || up_dim != inter {
            return Err(InferError::Dimension(format!(
                "dense rows gate/up sortent gate={gate_dim} up={up_dim}, attendu {inter}"
            )));
        }
        executor.encode_swiglu(
            encoder,
            owned,
            gate.tensor().buffer(),
            up.tensor().buffer(),
            swiglu.tensor().buffer(),
            inter_elements,
        )?;
        let down_dim = executor.encode_matmul_weight_buffers(
            encoder,
            swiglu.tensor().buffer(),
            rows,
            inter,
            down_proj,
            down.tensor().buffer(),
            false,
        )?;
        if down_dim != hidden {
            return Err(InferError::Dimension(format!(
                "dense rows down sort {down_dim}, attendu {hidden}"
            )));
        }
        executor.encode_copy(encoder, summed, layer_out, hidden_elements)?;
        executor.encode_accumulate_scaled(
            encoder,
            owned,
            down.tensor().buffer(),
            layer_out,
            1.0,
            hidden_elements,
        )
    }
}
