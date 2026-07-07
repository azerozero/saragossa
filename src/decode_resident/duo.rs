//! Couches duo (light-batch M=2, E2.2) : projections denses batchées qmm2,
//! attention/RoPE/KV et MoE PAR FLUX sur l'état de chaque flux.
//!
//! Chaque composition dé-fusionnée est BIT-exacte vs le chemin solo fusionné
//! (oracles E2.2a) : `rms_norm_simd` rows=2 → qmm2 ≡ `qkv_split_rms` fusionné ;
//! `attn_gate` → qmm2 ≡ `gated_input` fusionné ; les dispatches par flux
//! (RoPE, append KV, attention, conv/SSM, MoE) sont les kernels du solo sur
//! l'état du flux → byte-identité par flux par composition.

use super::utils::byte_offset_f32;
use super::*;

impl DecodeResidentState {
    /// Encode UNE couche full-attn MoE duo dans l'encoder partagé :
    /// `layer_in [2, hidden]` → `layer_out [2, hidden]`. Les positions RoPE et
    /// les états KV sont PAR FLUX ; les projections qkv/o sont batchées qmm2.
    ///
    /// # Errors
    ///
    /// Propage toute erreur d'encodage (dimension, overflow KV, Metal).
    #[expect(
        clippy::too_many_arguments,
        reason = "data flow duo d'une couche : exécuteur + 2 KV + poids + dims + ping-pong"
    )]
    pub(crate) fn encode_full_attn_moe_layer_duo(
        &self,
        executor: &MetalExecutor,
        encoder: &ComputeCommandEncoderRef,
        owned: &mut Vec<Buffer>,
        kv: [&mut FullAttentionMetalState; 2],
        weights: FullAttnLayerWeights<'_>,
        dims: FullAttnLayerDims,
        positions: [usize; 2],
        slots: [u64; 2],
        layer_in: &BufferRef,
        layer_out: &BufferRef,
    ) -> Result<()> {
        if !dims.attn_output_gate {
            return Err(InferError::Config(
                "duo full-attn attend attn_output_gate=true".to_string(),
            ));
        }
        let hidden = dims.hidden;
        let q_dim = dims
            .q_heads
            .checked_mul(dims.head_dim)
            .ok_or_else(|| InferError::Dimension("duo q_dim déborde".to_string()))?;
        let kv_dim = dims
            .kv_heads
            .checked_mul(dims.head_dim)
            .ok_or_else(|| InferError::Dimension("duo kv_dim déborde".to_string()))?;
        let q_gate_dim = q_dim
            .checked_mul(2)
            .ok_or_else(|| InferError::Dimension("duo q_gate_dim déborde".to_string()))?;
        let qkv_proj_dim = q_gate_dim
            .checked_add(kv_dim)
            .and_then(|value| value.checked_add(kv_dim))
            .ok_or_else(|| InferError::Dimension("duo qkv concat déborde".to_string()))?;
        let qkv_proj = weights.qkv_proj.ok_or_else(|| {
            InferError::Config("duo full-attn exige un concat qkv complet".to_string())
        })?;

        let normed2 = self.scratch().lease(2 * hidden, GpuElement::F32)?;
        let qkv2 = self.scratch().lease(2 * qkv_proj_dim, GpuElement::F32)?;
        let gated2 = self.scratch().lease(2 * q_dim, GpuElement::F32)?;
        let o2 = self.scratch().lease(2 * hidden, GpuElement::F32)?;
        let summed2 = self.scratch().lease(2 * hidden, GpuElement::F32)?;
        let post2 = self.scratch().lease(2 * hidden, GpuElement::F32)?;

        // Norm d'entrée en miroir du solo (rms_simd si le solo fusionne le
        // prologue 4-bit, rms_norm_rows sinon — DWQ 8-bit) → qkv qmm2.
        executor.encode_rms_norm_duo_matching_solo(
            encoder,
            layer_in,
            weights.input_norm,
            normed2.tensor().buffer(),
            hidden,
            dims.eps,
            qkv_proj,
            true,
        )?;
        let projected = executor.encode_matmul_weight_buffers(
            encoder,
            normed2.tensor().buffer(),
            2,
            hidden,
            qkv_proj,
            qkv2.tensor().buffer(),
            false,
        )?;
        if projected != qkv_proj_dim {
            return Err(InferError::Dimension(format!(
                "duo qkv sort {projected}, attendu {qkv_proj_dim}"
            )));
        }

        for (stream, kv_state) in kv.into_iter().enumerate() {
            let _namespace = crate::metal_backend::install_scratch_namespace(slots[stream]);
            let row_base = stream
                .checked_mul(qkv_proj_dim)
                .ok_or_else(|| InferError::Dimension("duo qkv row déborde".to_string()))?;
            let qkv_offset = byte_offset_f32(row_base, "duo qkv row offset")?;
            let k_offset = byte_offset_f32(
                row_base
                    .checked_add(q_gate_dim)
                    .ok_or_else(|| InferError::Dimension("duo k offset déborde".to_string()))?,
                "duo k offset",
            )?;
            let v_offset = byte_offset_f32(
                row_base
                    .checked_add(q_gate_dim)
                    .and_then(|value| value.checked_add(kv_dim))
                    .ok_or_else(|| InferError::Dimension("duo v offset déborde".to_string()))?,
                "duo v offset",
            )?;
            let score_cells = dims
                .q_heads
                .checked_mul(kv_state.capacity())
                .ok_or_else(|| InferError::Dimension("duo scores débordent".to_string()))?;

            let q_raw = self.scratch().lease(q_dim, GpuElement::F32)?;
            let gate = self.scratch().lease(q_dim, GpuElement::F32)?;
            let q_roped = self.scratch().lease(q_dim, GpuElement::F32)?;
            let k_roped = self.scratch().lease(kv_dim, GpuElement::F32)?;
            let scores = self.scratch().lease(score_cells, GpuElement::F32)?;
            let ctx = self.scratch().lease(q_dim, GpuElement::F32)?;
            let gated_row = self.scratch().lease(q_dim, GpuElement::F32)?;

            self.encode_split_q_gate_with_offset(
                encoder,
                qkv2.tensor().buffer(),
                qkv_offset,
                q_raw.tensor().buffer(),
                gate.tensor().buffer(),
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
                positions[stream],
                dims.eps,
                dims.theta,
            )?;
            self.encode_rms_norm_rope_decode_with_offset(
                encoder,
                qkv2.tensor().buffer(),
                k_offset,
                weights.k_norm,
                k_roped.tensor().buffer(),
                dims.kv_heads,
                dims.head_dim,
                dims.rope_dims,
                positions[stream],
                dims.eps,
                dims.theta,
            )?;
            kv_state.encode_append_kv_with_offsets(
                encoder,
                k_roped.tensor().buffer(),
                0,
                qkv2.tensor().buffer(),
                v_offset,
            )?;
            kv_state.encode_attention_decode(
                encoder,
                q_roped.tensor().buffer(),
                scores.tensor().buffer(),
                ctx.tensor().buffer(),
            )?;
            self.encode_attn_gate(
                encoder,
                ctx.tensor().buffer(),
                gate.tensor().buffer(),
                gated_row.tensor().buffer(),
                q_dim,
            )?;
            let gated_offset = byte_offset_f32(
                stream
                    .checked_mul(q_dim)
                    .ok_or_else(|| InferError::Dimension("duo gated row déborde".to_string()))?,
                "duo gated row offset",
            )?;
            executor.encode_copy_with_offsets(
                encoder,
                gated_row.tensor().buffer(),
                0,
                gated2.tensor().buffer(),
                gated_offset,
                q_dim,
            )?;
        }

        let o_dim = executor.encode_matmul_weight_buffers(
            encoder,
            gated2.tensor().buffer(),
            2,
            q_dim,
            weights.o_proj,
            o2.tensor().buffer(),
            false,
        )?;
        if o_dim != hidden {
            return Err(InferError::Dimension(format!(
                "duo o_proj sort {o_dim}, attendu {hidden}"
            )));
        }
        executor.encode_add_rms_norm_rows(
            encoder,
            layer_in,
            o2.tensor().buffer(),
            weights.post_norm,
            summed2.tensor().buffer(),
            post2.tensor().buffer(),
            2,
            hidden,
            dims.eps,
        )?;
        self.encode_moe_tail_duo(
            executor,
            encoder,
            owned,
            post2.tensor().buffer(),
            summed2.tensor().buffer(),
            layer_out,
            hidden,
            weights.moe,
            weights.top_k,
            slots,
        )
    }

    /// Encode UNE couche linear-attn MoE duo : cœur conv/SSM par flux via
    /// [`MetalExecutor::encode_linear_attn_resident_duo_buffers`], tail MoE
    /// par flux.
    ///
    /// # Errors
    ///
    /// Propage toute erreur d'encodage (dimension, Metal).
    #[expect(
        clippy::too_many_arguments,
        reason = "data flow duo d'une couche : exécuteur + 2 états + poids + spec + ping-pong"
    )]
    pub(crate) fn encode_linear_attn_moe_layer_duo(
        &self,
        executor: &MetalExecutor,
        encoder: &ComputeCommandEncoderRef,
        owned: &mut Vec<Buffer>,
        states: [&LinearAttentionMetalState; 2],
        weights: LinearAttnLayerWeights<'_>,
        spec: LinearAttentionStepSpec,
        res_dims: LinearAttnResidentDims,
        hidden: usize,
        eps: f32,
        slots: [u64; 2],
        layer_in: &BufferRef,
        layer_out: &BufferRef,
    ) -> Result<()> {
        let attn2 = self.scratch().lease(2 * hidden, GpuElement::F32)?;
        let summed2 = self.scratch().lease(2 * hidden, GpuElement::F32)?;
        let post2 = self.scratch().lease(2 * hidden, GpuElement::F32)?;

        let full_linear = weights.linear.full.as_ref().ok_or_else(|| {
            InferError::Config(
                "linear-attn duo exige un concat complet des projections".to_string(),
            )
        })?;
        executor.encode_linear_attn_resident_duo_buffers(
            encoder,
            layer_in,
            (weights.input_norm, eps),
            attn2.tensor().buffer(),
            full_linear,
            states,
            slots,
            spec,
            res_dims,
        )?;
        executor.encode_add_rms_norm_rows(
            encoder,
            layer_in,
            attn2.tensor().buffer(),
            weights.post_norm,
            summed2.tensor().buffer(),
            post2.tensor().buffer(),
            2,
            hidden,
            eps,
        )?;
        self.encode_moe_tail_duo(
            executor,
            encoder,
            owned,
            post2.tensor().buffer(),
            summed2.tensor().buffer(),
            layer_out,
            hidden,
            weights.moe,
            weights.top_k,
            slots,
        )
    }

    /// Tail MoE duo : router + shared expert batchés qmm2 quand les poids sont
    /// éligibles (E2.3, kill-switch `RETI_RUST_LIGHTBATCH_MOE2=0`), sinon repli
    /// sur la composition solo par flux. Les deux variantes sont byte-identiques
    /// par flux ; la trace `RETI_RUST_TRACE_LIGHTBATCH=1` annonce le mode UNE fois.
    #[expect(
        clippy::too_many_arguments,
        reason = "tail MoE duo : exécuteur + duo buffers + poids + slots"
    )]
    fn encode_moe_tail_duo(
        &self,
        executor: &MetalExecutor,
        encoder: &ComputeCommandEncoderRef,
        owned: &mut Vec<Buffer>,
        post2: &BufferRef,
        summed2: &BufferRef,
        layer_out: &BufferRef,
        hidden: usize,
        moe: &MetalMoeSharedWeights,
        top_k: usize,
        slots: [u64; 2],
    ) -> Result<()> {
        let moe_duo = crate::runtime_flags::lightbatch_moe2_enabled()
            && executor.moe_shared_duo_eligible(moe);
        if crate::runtime_flags::trace_lightbatch_enabled() {
            static TRACED: OnceLock<()> = OnceLock::new();
            TRACED.get_or_init(|| {
                eprintln!(
                    "lightbatch: tail MoE {}",
                    if moe_duo {
                        "duo (router+shared qmm2)"
                    } else {
                        "par flux (composition solo)"
                    }
                );
            });
        }
        if moe_duo {
            return executor.encode_moe_shared_duo_buffers(
                encoder, owned, post2, summed2, layer_out, hidden, moe, top_k, slots,
            );
        }
        self.encode_moe_shared_per_stream(
            executor, encoder, owned, post2, summed2, layer_out, hidden, moe, top_k, slots,
        )
    }

    /// Tail MoE shared PAR FLUX : copies de la ligne du flux (entrée + résiduel)
    /// vers des scratch dédiés, MoE mono-token du solo (mêmes kernels, scratch
    /// exécuteur namespacé par slot), copie du résultat vers la ligne de sortie.
    /// Les copies sont bit-exactes ; le routage MoE de chaque flux est identique
    /// au solo.
    #[expect(
        clippy::too_many_arguments,
        reason = "tail MoE duo : exécuteur + duo buffers + poids + slots"
    )]
    fn encode_moe_shared_per_stream(
        &self,
        executor: &MetalExecutor,
        encoder: &ComputeCommandEncoderRef,
        owned: &mut Vec<Buffer>,
        post2: &BufferRef,
        summed2: &BufferRef,
        layer_out: &BufferRef,
        hidden: usize,
        moe: &MetalMoeSharedWeights,
        top_k: usize,
        slots: [u64; 2],
    ) -> Result<()> {
        for (stream, slot) in slots.into_iter().enumerate() {
            let _namespace = crate::metal_backend::install_scratch_namespace(slot);
            let row_offset = byte_offset_f32(
                stream
                    .checked_mul(hidden)
                    .ok_or_else(|| InferError::Dimension("duo moe row déborde".to_string()))?,
                "duo moe row offset",
            )?;
            let moe_in = self.scratch().lease(hidden, GpuElement::F32)?;
            let moe_res = self.scratch().lease(hidden, GpuElement::F32)?;
            let moe_out = self.scratch().lease(hidden, GpuElement::F32)?;
            executor.encode_copy_with_offsets(
                encoder,
                post2,
                row_offset,
                moe_in.tensor().buffer(),
                0,
                hidden,
            )?;
            executor.encode_copy_with_offsets(
                encoder,
                summed2,
                row_offset,
                moe_res.tensor().buffer(),
                0,
                hidden,
            )?;
            executor.encode_moe_shared_buffers(
                encoder,
                owned,
                moe_in.tensor().buffer(),
                Some(moe_res.tensor().buffer()),
                moe_out.tensor().buffer(),
                hidden,
                moe,
                top_k,
            )?;
            executor.encode_copy_with_offsets(
                encoder,
                moe_out.tensor().buffer(),
                0,
                layer_out,
                row_offset,
                hidden,
            )?;
        }
        Ok(())
    }
}
