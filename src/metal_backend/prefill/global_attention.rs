//! Primitives propres à l'attention globale Gemma 4.

use super::resident::ResidentPrefillEncodeContext;
use super::*;

const RMS_HEADS_TG_WIDTH: u64 = 256;

impl MetalExecutor {
    #[inline]
    #[expect(
        clippy::too_many_arguments,
        reason = "phase attention résidente: buffers, dimensions et index restent explicites"
    )]
    pub(super) fn encode_resident_attention_phase(
        &self,
        context: &mut ResidentPrefillEncodeContext<'_>,
        attention: PrefillAttentionLayer<'_>,
        attention_scratch: PrefillResidentAttentionScratch,
        normed_buffer: &BufferRef,
        spec: PrefillAttentionSpec,
        hidden_dim: usize,
        q_dim: usize,
        layer_index: usize,
    ) -> Result<(Buffer, PrefillResidentLayerCacheBuffer)> {
        match (attention, attention_scratch) {
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
                self.run_resident_prefill_section(context, "qkv", |encoder, owned| {
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
                            normed_buffer,
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
                            normed_buffer,
                            spec.seq,
                            hidden_dim,
                            q_proj.weight(),
                            &q_buffer,
                        )?;
                    }
                    self.encode_matmul_weight(
                        encoder,
                        owned,
                        normed_buffer,
                        spec.seq,
                        hidden_dim,
                        k_proj.weight(),
                        &k_buffer,
                    )?;
                    if spec.k_eq_v {
                        self.encode_copy(
                            encoder,
                            &k_buffer,
                            &v_buffer,
                            checked_len(
                                spec.seq,
                                checked_len(spec.kv_heads, spec.head_dim, "resident alias kv_dim")?,
                                "resident alias kv_len",
                            )?,
                        )?;
                    } else {
                        self.encode_matmul_weight(
                            encoder,
                            owned,
                            normed_buffer,
                            spec.seq,
                            hidden_dim,
                            v_proj.weight(),
                            &v_buffer,
                        )?;
                    }
                    if spec.value_norm {
                        self.encode_rms_norm_heads_no_scale(encoder, &v_buffer, &v_buffer, spec)?;
                    }
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
                })?;
                self.run_resident_prefill_section(
                    context,
                    if spec.window.is_some() {
                        "windowed_attention"
                    } else {
                        "causal_attention"
                    },
                    |encoder, _owned| {
                        if spec.window.is_some() {
                            self.encode_windowed_attention_prefill(
                                encoder,
                                &q_rope_buffer,
                                &k_rope_buffer,
                                &v_buffer,
                                &context_buffer,
                                spec,
                            )
                        } else {
                            self.encode_causal_attention_prefill(
                                encoder,
                                &q_rope_buffer,
                                &k_rope_buffer,
                                &v_buffer,
                                &context_buffer,
                                spec,
                            )
                        }
                    },
                )?;
                let context_for_o = if gated {
                    let (Some(attn_gate_buffer), Some(gated_context_buffer)) =
                        (attn_gate_buffer.as_ref(), gated_context_buffer.as_ref())
                    else {
                        return Err(InferError::Dimension(format!(
                            "prefill résident full gated output scratch incomplet couche {layer_index}"
                        )));
                    };
                    self.run_resident_prefill_section(context, "attn_gate", |encoder, _owned| {
                        self.encode_attn_gate_rows(
                            encoder,
                            &context_buffer,
                            attn_gate_buffer,
                            gated_context_buffer,
                            checked_len(spec.seq, q_dim, "resident gated context")?,
                        )
                    })?;
                    gated_context_buffer
                } else {
                    &context_buffer
                };
                self.run_resident_prefill_section(context, "o_proj", |encoder, owned| {
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
                })?;
                Ok((
                    o_buffer,
                    PrefillResidentLayerCacheBuffer::Full {
                        key: k_rope_buffer,
                        value: v_buffer,
                        kv_dim: checked_len(spec.kv_heads, spec.head_dim, "resident cache kv_dim")?,
                    },
                ))
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
                self.run_resident_prefill_section(
                    context,
                    "linear_attention",
                    |encoder, owned| {
                        self.encode_linear_attn_batch_resident(
                            encoder,
                            owned,
                            normed_buffer,
                            &output,
                            spec.seq,
                            weights,
                            &state,
                            linear_spec,
                            dims,
                        )
                    },
                )?;
                Ok((output, PrefillResidentLayerCacheBuffer::Linear { state }))
            }
            _ => Err(InferError::Dimension(format!(
                "prefill résident scratch attention incohérent couche {layer_index}"
            ))),
        }
    }

    pub(super) fn encode_rms_norm_heads_no_scale(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input_buffer: &BufferRef,
        output_buffer: &BufferRef,
        spec: PrefillAttentionSpec,
    ) -> Result<()> {
        self.encode_rms_norm_heads_no_scale_rows(
            encoder,
            input_buffer,
            output_buffer,
            spec.seq,
            spec.kv_heads,
            spec.head_dim,
            spec.eps,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "normalisation V par tête: encoder, buffers et dimensions explicites"
    )]
    pub(crate) fn encode_rms_norm_heads_no_scale_rows(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input_buffer: &BufferRef,
        output_buffer: &BufferRef,
        rows: usize,
        heads: usize,
        head_dim: usize,
        eps: f32,
    ) -> Result<()> {
        let dims = [
            checked_u32(rows, "value norm rows")?,
            checked_u32(heads, "value norm heads")?,
            checked_u32(head_dim, "value norm head_dim")?,
        ];
        encoder.set_compute_pipeline_state(&self.rms_norm_heads_no_scale_f32);
        encoder.set_buffer(0, Some(input_buffer), 0);
        encoder.set_buffer(1, Some(output_buffer), 0);
        set_u32_bytes(encoder, 2, &dims, "value_norm_dims")?;
        set_f32_bytes(encoder, 3, &[eps], "value_norm_eps")?;
        profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(
                checked_nsuint(heads, "value norm heads")?,
                checked_nsuint(rows, "value norm rows")?,
                1,
            ),
            MTLSize::new(RMS_HEADS_TG_WIDTH, 1, 1),
        );
        post_dispatch_barrier(encoder);
        Ok(())
    }
}
