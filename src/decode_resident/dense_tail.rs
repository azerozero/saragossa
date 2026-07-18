//! Encodage des tails FFN denses résidents.

use super::*;

struct DenseTailDims {
    inter: usize,
}

fn validate_dense_tail_dims(
    executor: &MetalExecutor,
    hidden: usize,
    gate_proj: &MetalLinearWeightBuffers,
    up_proj: &MetalLinearWeightBuffers,
    down_proj: &MetalLinearWeightBuffers,
) -> Result<DenseTailDims> {
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
    Ok(DenseTailDims { inter })
}

impl DecodeResidentState {
    #[expect(
        clippy::too_many_arguments,
        reason = "tail dense résident: buffers + poids nécessaires à l'encodage"
    )]
    pub(super) fn encode_dense_tail(
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
        let DenseTailDims { inter } =
            validate_dense_tail_dims(executor, hidden, gate_proj, up_proj, down_proj)?;
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
    pub(super) fn encode_dense_tail_rows(
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
        let DenseTailDims { inter } =
            validate_dense_tail_dims(executor, hidden, gate_proj, up_proj, down_proj)?;

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
        executor.encode_add_scaled(
            encoder,
            owned,
            summed,
            down.tensor().buffer(),
            layer_out,
            1.0,
            hidden_elements,
        )
    }

    /// Encode le tail dense Gemma 4 : pré-RMSNorm FFN, GeGLU tanh, post-RMSNorm.
    #[expect(
        clippy::too_many_arguments,
        reason = "tail Gemma dense: buffers + poids nécessaires à l'encodage"
    )]
    pub(super) fn encode_gemma_dense_tail(
        &self,
        executor: &MetalExecutor,
        encoder: &ComputeCommandEncoderRef,
        owned: &mut Vec<Buffer>,
        hidden_state: &BufferRef,
        layer_out: &BufferRef,
        rows: usize,
        hidden: usize,
        eps: f32,
        gate_proj: &MetalLinearWeightBuffers,
        up_proj: &MetalLinearWeightBuffers,
        down_proj: &MetalLinearWeightBuffers,
        gemma: GemmaDenseTailWeights<'_>,
    ) -> Result<()> {
        let DenseTailDims { inter } =
            validate_dense_tail_dims(executor, hidden, gate_proj, up_proj, down_proj)?;
        let inter_elements = rows
            .checked_mul(inter)
            .ok_or_else(|| InferError::Dimension("gemma dense inter déborde".to_string()))?;
        let hidden_elements = rows
            .checked_mul(hidden)
            .ok_or_else(|| InferError::Dimension("gemma dense hidden déborde".to_string()))?;
        let ffn_input = self.scratch().lease(hidden_elements, GpuElement::F32)?;
        let gate = self.scratch().lease(inter_elements, GpuElement::F32)?;
        let up = self.scratch().lease(inter_elements, GpuElement::F32)?;
        let geglu = self.scratch().lease(inter_elements, GpuElement::F32)?;
        let down = self.scratch().lease(hidden_elements, GpuElement::F32)?;
        let ffn_normed = self.scratch().lease(hidden_elements, GpuElement::F32)?;

        executor.encode_rms_norm_rows(
            encoder,
            hidden_state,
            gemma.pre_feedforward_norm,
            ffn_input.tensor().buffer(),
            rows,
            hidden,
            eps,
        )?;
        let gate_dim = executor.encode_matmul_weight_buffers(
            encoder,
            ffn_input.tensor().buffer(),
            rows,
            hidden,
            gate_proj,
            gate.tensor().buffer(),
            false,
        )?;
        let up_dim = executor.encode_matmul_weight_buffers(
            encoder,
            ffn_input.tensor().buffer(),
            rows,
            hidden,
            up_proj,
            up.tensor().buffer(),
            false,
        )?;
        if gate_dim != inter || up_dim != inter {
            return Err(InferError::Dimension(format!(
                "gemma dense gate/up sortent gate={gate_dim} up={up_dim}, attendu {inter}"
            )));
        }
        executor.encode_geglu_tanh(
            encoder,
            owned,
            gate.tensor().buffer(),
            up.tensor().buffer(),
            geglu.tensor().buffer(),
            inter_elements,
        )?;
        let down_dim = executor.encode_matmul_weight_buffers(
            encoder,
            geglu.tensor().buffer(),
            rows,
            inter,
            down_proj,
            down.tensor().buffer(),
            false,
        )?;
        if down_dim != hidden {
            return Err(InferError::Dimension(format!(
                "gemma dense down sort {down_dim}, attendu {hidden}"
            )));
        }
        executor.encode_rms_norm_rows(
            encoder,
            down.tensor().buffer(),
            gemma.post_feedforward_norm,
            ffn_normed.tensor().buffer(),
            rows,
            hidden,
            eps,
        )?;
        executor.encode_add_scaled(
            encoder,
            owned,
            hidden_state,
            ffn_normed.tensor().buffer(),
            layer_out,
            1.0,
            hidden_elements,
        )?;
        if let Some(scale) = gemma.layer_scalar {
            executor.encode_accumulate_scaled(
                encoder,
                owned,
                layer_out,
                layer_out,
                scale - 1.0,
                hidden_elements,
            )?;
        }
        Ok(())
    }
}
