//! Encodage Metal du MoE routed-only.

use super::*;

impl MetalExecutor {
    /// Encode le MoE routé seul dans un encoder partagé.
    ///
    /// Reprend le préfixe routed de [`Self::encode_moe_shared_buffers`] sans
    /// shared-expert. `residual = Some(buf)` fusionne `buf + MoE` via
    /// `weighted_sum_add`, comme le tail résident shared.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si une dimension est incompatible ou si un dispatch échoue.
    #[expect(
        clippy::too_many_arguments,
        reason = "mirror des poids MoE routed résolus (routeur + experts)"
    )]
    pub(crate) fn encode_moe_routed_buffers(
        &self,
        encoder: &ComputeCommandEncoderRef,
        owned_buffers: &mut Vec<Buffer>,
        input_buffer: &BufferRef,
        residual: Option<&BufferRef>,
        output_buffer: &BufferRef,
        in_dim: usize,
        weights: &MetalMoeRoutedWeights,
        top_k: usize,
    ) -> Result<()> {
        self.encode_moe_routed_buffers_with_activation(
            encoder,
            owned_buffers,
            input_buffer,
            residual,
            output_buffer,
            in_dim,
            weights,
            top_k,
            crate::Activation::Silu,
        )
    }

    /// Encode le MoE routé seul avec l'activation d'expert demandée.
    ///
    /// `Activation::Silu` garde strictement le chemin Qwen historique, y compris
    /// la fusion gate+up+SwiGLU quand elle est disponible. `Activation::GeluTanh`
    /// conserve le même routage/top-k/gather puis applique le kernel GeGLU.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si une dimension est incompatible ou si un dispatch échoue.
    #[expect(
        clippy::too_many_arguments,
        reason = "mirror des poids MoE routed résolus avec activation explicite"
    )]
    pub(crate) fn encode_moe_routed_buffers_with_activation(
        &self,
        encoder: &ComputeCommandEncoderRef,
        owned_buffers: &mut Vec<Buffer>,
        input_buffer: &BufferRef,
        residual: Option<&BufferRef>,
        output_buffer: &BufferRef,
        in_dim: usize,
        weights: &MetalMoeRoutedWeights,
        top_k: usize,
        activation: crate::Activation,
    ) -> Result<()> {
        self.encode_moe_routed_buffers_with_router_input_and_activation(
            encoder,
            owned_buffers,
            input_buffer,
            input_buffer,
            None,
            residual,
            output_buffer,
            in_dim,
            weights,
            top_k,
            activation,
        )
    }

    /// Encode le MoE routé avec une entrée routeur et des échelles d'expert.
    #[expect(
        clippy::too_many_arguments,
        reason = "Gemma sépare l'entrée du routeur de celle des experts"
    )]
    fn encode_moe_routed_buffers_with_router_input_and_activation(
        &self,
        encoder: &ComputeCommandEncoderRef,
        owned_buffers: &mut Vec<Buffer>,
        input_buffer: &BufferRef,
        router_input_buffer: &BufferRef,
        per_expert_scale: Option<&BufferRef>,
        residual: Option<&BufferRef>,
        output_buffer: &BufferRef,
        in_dim: usize,
        weights: &MetalMoeRoutedWeights,
        top_k: usize,
        activation: crate::Activation,
    ) -> Result<()> {
        let shape = self.check_moe_routed_buffer_shapes(in_dim, weights, top_k)?;
        let scratch = self.allocate_moe_routed_scratch(
            top_k,
            shape.expert_count,
            shape.inter_dim,
            shape.out_dim,
        )?;

        let router_out_dim = self.encode_matmul_weight_buffers(
            encoder,
            router_input_buffer,
            1,
            in_dim,
            &weights.router,
            &scratch.router,
            false,
        )?;
        if router_out_dim != shape.expert_count {
            return Err(InferError::Dimension(format!(
                "routeur MoE routed sort {router_out_dim}, attendu {}",
                shape.expert_count
            )));
        }
        self.encode_topk_softmax(
            encoder,
            owned_buffers,
            &scratch.router,
            &scratch.indices,
            &scratch.scores,
            shape.expert_count,
            top_k,
        )?;
        if let Some(scale) = per_expert_scale {
            self.encode_scale_topk_scores(
                encoder,
                &scratch.indices,
                &scratch.scores,
                scale,
                top_k,
            )?;
        }
        match activation {
            crate::Activation::Silu => {
                if !self.encode_gather_gate_up_swiglu(
                    encoder,
                    owned_buffers,
                    input_buffer,
                    1,
                    &weights.stacked.gate,
                    &weights.stacked.up,
                    &scratch.indices,
                    top_k,
                    &scratch.hidden,
                )? {
                    self.encode_gather_matmul(
                        encoder,
                        owned_buffers,
                        input_buffer,
                        1,
                        &weights.stacked.gate,
                        &scratch.indices,
                        top_k,
                        &scratch.gate,
                    )?;
                    self.encode_gather_matmul(
                        encoder,
                        owned_buffers,
                        input_buffer,
                        1,
                        &weights.stacked.up,
                        &scratch.indices,
                        top_k,
                        &scratch.up,
                    )?;
                    self.encode_swiglu(
                        encoder,
                        owned_buffers,
                        &scratch.gate,
                        &scratch.up,
                        &scratch.hidden,
                        checked_len(top_k, shape.inter_dim, "moe routed swiglu")?,
                    )?;
                }
            }
            crate::Activation::GeluTanh => {
                if !self.encode_gather_gate_up_geglu(
                    encoder,
                    owned_buffers,
                    input_buffer,
                    1,
                    &weights.stacked.gate,
                    &weights.stacked.up,
                    &scratch.indices,
                    top_k,
                    &scratch.hidden,
                )? {
                    self.encode_gather_matmul(
                        encoder,
                        owned_buffers,
                        input_buffer,
                        1,
                        &weights.stacked.gate,
                        &scratch.indices,
                        top_k,
                        &scratch.gate,
                    )?;
                    self.encode_gather_matmul(
                        encoder,
                        owned_buffers,
                        input_buffer,
                        1,
                        &weights.stacked.up,
                        &scratch.indices,
                        top_k,
                        &scratch.up,
                    )?;
                    self.encode_geglu_tanh(
                        encoder,
                        owned_buffers,
                        &scratch.gate,
                        &scratch.up,
                        &scratch.hidden,
                        checked_len(top_k, shape.inter_dim, "moe routed geglu")?,
                    )?;
                }
            }
        }
        self.encode_gather_matmul(
            encoder,
            owned_buffers,
            &scratch.hidden,
            top_k,
            &weights.stacked.down,
            &scratch.indices,
            top_k,
            &scratch.down,
        )?;
        match residual {
            Some(residual_buffer) => self.encode_weighted_sum_add_topk(
                encoder,
                owned_buffers,
                &scratch.down,
                &scratch.scores,
                residual_buffer,
                output_buffer,
                top_k,
                shape.out_dim,
            )?,
            None => self.encode_weighted_sum_topk(
                encoder,
                owned_buffers,
                &scratch.down,
                &scratch.scores,
                output_buffer,
                top_k,
                shape.out_dim,
            )?,
        }
        Ok(())
    }

    /// Encode le MoE routé batché avec l'activation d'expert demandée.
    ///
    /// Le chemin mono-ligne délègue à
    /// [`Self::encode_moe_routed_buffers_with_activation`] afin de préserver
    /// strictement l'encodage historique du decode. Les batches utilisent les
    /// mêmes kernels avec un top-k et une réduction groupés par ligne.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le batch est vide, si une dimension est incompatible
    /// ou si un dispatch échoue.
    #[expect(
        clippy::too_many_arguments,
        reason = "mirror batché des poids MoE routed avec activation explicite"
    )]
    pub(crate) fn encode_moe_routed_buffers_rows_with_router_input_and_activation(
        &self,
        encoder: &ComputeCommandEncoderRef,
        owned_buffers: &mut Vec<Buffer>,
        input_buffer: &BufferRef,
        router_input_buffer: &BufferRef,
        per_expert_scale: Option<&BufferRef>,
        residual: Option<&BufferRef>,
        output_buffer: &BufferRef,
        rows: usize,
        in_dim: usize,
        weights: &MetalMoeRoutedWeights,
        top_k: usize,
        activation: crate::Activation,
    ) -> Result<()> {
        if rows == 0 {
            return Err(InferError::Dimension(
                "MoE routed rows: batch vide".to_string(),
            ));
        }
        if rows == 1 {
            return self.encode_moe_routed_buffers_with_router_input_and_activation(
                encoder,
                owned_buffers,
                input_buffer,
                router_input_buffer,
                per_expert_scale,
                residual,
                output_buffer,
                in_dim,
                weights,
                top_k,
                activation,
            );
        }

        let shape = self.check_moe_routed_buffer_shapes(in_dim, weights, top_k)?;
        let total_topk = checked_len(rows, top_k, "moe routed rows topk total")?;
        let scratch = self.allocate_moe_routed_rows_scratch(
            rows,
            top_k,
            shape.expert_count,
            shape.inter_dim,
            shape.out_dim,
        )?;

        let router_out_dim = self.encode_matmul_weight_buffers(
            encoder,
            router_input_buffer,
            rows,
            in_dim,
            &weights.router,
            &scratch.router,
            false,
        )?;
        if router_out_dim != shape.expert_count {
            return Err(InferError::Dimension(format!(
                "routeur MoE routed rows sort {router_out_dim}, attendu {}",
                shape.expert_count
            )));
        }
        self.encode_topk_softmax_rows(
            encoder,
            &scratch.router,
            &scratch.indices,
            &scratch.scores,
            rows,
            shape.expert_count,
            top_k,
        )?;
        if let Some(scale) = per_expert_scale {
            self.encode_scale_topk_scores(
                encoder,
                &scratch.indices,
                &scratch.scores,
                scale,
                total_topk,
            )?;
        }

        match activation {
            crate::Activation::Silu => {
                if !self.encode_gather_gate_up_swiglu(
                    encoder,
                    owned_buffers,
                    input_buffer,
                    rows,
                    &weights.stacked.gate,
                    &weights.stacked.up,
                    &scratch.indices,
                    total_topk,
                    &scratch.hidden,
                )? {
                    self.encode_gather_matmul(
                        encoder,
                        owned_buffers,
                        input_buffer,
                        rows,
                        &weights.stacked.gate,
                        &scratch.indices,
                        total_topk,
                        &scratch.gate,
                    )?;
                    self.encode_gather_matmul(
                        encoder,
                        owned_buffers,
                        input_buffer,
                        rows,
                        &weights.stacked.up,
                        &scratch.indices,
                        total_topk,
                        &scratch.up,
                    )?;
                    self.encode_swiglu(
                        encoder,
                        owned_buffers,
                        &scratch.gate,
                        &scratch.up,
                        &scratch.hidden,
                        checked_len(total_topk, shape.inter_dim, "moe routed rows swiglu")?,
                    )?;
                }
            }
            crate::Activation::GeluTanh => {
                self.encode_gather_matmul(
                    encoder,
                    owned_buffers,
                    input_buffer,
                    rows,
                    &weights.stacked.gate,
                    &scratch.indices,
                    total_topk,
                    &scratch.gate,
                )?;
                self.encode_gather_matmul(
                    encoder,
                    owned_buffers,
                    input_buffer,
                    rows,
                    &weights.stacked.up,
                    &scratch.indices,
                    total_topk,
                    &scratch.up,
                )?;
                self.encode_geglu_tanh(
                    encoder,
                    owned_buffers,
                    &scratch.gate,
                    &scratch.up,
                    &scratch.hidden,
                    checked_len(total_topk, shape.inter_dim, "moe routed rows geglu")?,
                )?;
            }
        }
        self.encode_gather_matmul(
            encoder,
            owned_buffers,
            &scratch.hidden,
            total_topk,
            &weights.stacked.down,
            &scratch.indices,
            total_topk,
            &scratch.down,
        )?;
        match residual {
            Some(residual_buffer) => self.encode_weighted_sum_add_grouped_topk(
                encoder,
                owned_buffers,
                &scratch.down,
                &scratch.scores,
                residual_buffer,
                output_buffer,
                rows,
                top_k,
                shape.out_dim,
            ),
            None => self.encode_weighted_sum_grouped_topk(
                encoder,
                owned_buffers,
                &scratch.down,
                &scratch.scores,
                output_buffer,
                rows,
                top_k,
                shape.out_dim,
            ),
        }
    }

    /// Encode gate/up, activation et down des experts routés.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si une taille déborde ou si un dispatch échoue.
    pub(super) fn encode_moe_shared_routed_phase(
        &self,
        context: &mut MoeRoutedSharedPhase<'_>,
    ) -> Result<()> {
        let fused = self.encode_moe_shared_routed_gate_up_phase(context)?;
        if !fused {
            self.encode_swiglu(
                context.encoder,
                context.owned_buffers,
                context.gate,
                context.up,
                context.hidden,
                checked_len(context.slots, context.inter_dim, context.swiglu_label)?,
            )?;
        }
        self.encode_moe_shared_routed_down_phase(context)
    }

    /// Encode gate/up des experts routés et signale si l'activation est fusionnée.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si un gather ou un dispatch fusionné échoue.
    pub(super) fn encode_moe_shared_routed_gate_up_phase(
        &self,
        context: &mut MoeRoutedSharedPhase<'_>,
    ) -> Result<bool> {
        let fused = self.encode_gather_gate_up_swiglu(
            context.encoder,
            context.owned_buffers,
            context.input_buffer,
            context.rows,
            &context.stacked.gate,
            &context.stacked.up,
            context.indices,
            context.slots,
            context.hidden,
        )?;
        if !fused {
            self.encode_gather_matmul(
                context.encoder,
                context.owned_buffers,
                context.input_buffer,
                context.rows,
                &context.stacked.gate,
                context.indices,
                context.slots,
                context.gate,
            )?;
            self.encode_gather_matmul(
                context.encoder,
                context.owned_buffers,
                context.input_buffer,
                context.rows,
                &context.stacked.up,
                context.indices,
                context.slots,
                context.up,
            )?;
        }
        Ok(fused)
    }

    /// Encode la projection down des experts routés.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le gather down échoue.
    pub(super) fn encode_moe_shared_routed_down_phase(
        &self,
        context: &mut MoeRoutedSharedPhase<'_>,
    ) -> Result<()> {
        self.encode_gather_matmul(
            context.encoder,
            context.owned_buffers,
            context.hidden,
            context.slots,
            &context.stacked.down,
            context.indices,
            context.slots,
            context.down,
        )
    }
}
