//! Kernels élémentaires Metal du MoE.

use super::*;

const TOPK_SOFTMAX_TG_WIDTH: u64 = 32;

#[expect(
    clippy::too_many_arguments,
    reason = "wrappers d'encodage Metal: buffers, dimensions et offsets restent explicites"
)]
impl MetalExecutor {
    pub(super) fn encode_gather_gate_up_swiglu(
        &self,
        encoder: &ComputeCommandEncoderRef,
        _owned_buffers: &mut Vec<metal::Buffer>,
        lhs_buffer: &BufferRef,
        lhs_rows: usize,
        gate: &StackedAffineBuffers,
        up: &StackedAffineBuffers,
        indices_buffer: &BufferRef,
        topk: usize,
        output_buffer: &BufferRef,
    ) -> Result<bool> {
        if !valid_gather_lhs_rows(lhs_rows, topk) {
            return Err(InferError::Dimension(format!(
                "gather gate/up lhs_rows={lhs_rows}, topk={topk}"
            )));
        }
        if !can_use_fast_gather_pair_qmv(lhs_rows, gate, up) {
            return Ok(false);
        }
        let dims = [
            checked_u32(topk, "gate/up topk")?,
            checked_u32(gate.out_dim, "gate/up out_dim")?,
            checked_u32(gate.in_dim, "gate/up in_dim")?,
            checked_u32(gate.packed_cols, "gate/up packed_cols")?,
        ];
        let quant = [
            checked_u32(gate.group_size, "gate/up group_size")?,
            checked_u32(gate.bits, "gate/up bits")?,
            checked_u32(gate.groups, "gate/up groups")?,
            checked_u32(lhs_rows, "gate/up lhs_rows")?,
        ];
        let pipeline = if gate.bits == FAST_QMV_BITS {
            (
                &self.affine_gather_gate_up_swiglu_fast_u4_gs64_f32,
                8,
                "affine_gather_gate_up_swiglu_fast_u4_gs64_f32",
            )
        } else if gate.group_size == FAST_QMV_GROUP_SIZE {
            (
                &self.affine_gather_gate_up_swiglu_fast_u8_gs64_f32,
                8,
                "affine_gather_gate_up_swiglu_fast_u8_gs64_f32",
            )
        } else {
            (
                &self.affine_gather_gate_up_swiglu_fast_u8_gs128_f32,
                8,
                "affine_gather_gate_up_swiglu_fast_u8_gs128_f32",
            )
        };
        let (pipeline, rows_per_threadgroup, kernel_name) = pipeline;
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(lhs_buffer), 0);
        encoder.set_buffer(1, Some(&gate.packed), 0);
        encoder.set_buffer(2, Some(&gate.scales), 0);
        encoder.set_buffer(3, Some(&gate.biases), 0);
        encoder.set_buffer(4, Some(&up.packed), 0);
        encoder.set_buffer(5, Some(&up.scales), 0);
        encoder.set_buffer(6, Some(&up.biases), 0);
        encoder.set_buffer(7, Some(indices_buffer), 0);
        encoder.set_buffer(8, Some(output_buffer), 0);
        set_u32_bytes(encoder, 9, &dims, "gate_up_dims")?;
        set_u32_bytes(encoder, 10, &quant, "gate_up_quant")?;
        trace_dispatch_path(kernel_name, topk, gate.out_dim, gate.in_dim);
        profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(
                checked_nsuint(topk, "gate/up topk")?,
                checked_nsuint(
                    gate.out_dim.div_ceil(rows_per_threadgroup),
                    "gate/up out groups",
                )?,
                1,
            ),
            MTLSize::new(64, 1, 1),
        );
        post_dispatch_barrier(encoder);
        Ok(true)
    }

    /// Fond les deux QMV (gate_proj, up_proj) + le swiglu du **shared-expert**
    /// (single-row, batch=1) en UN dispatch via
    /// `affine_gate_up_swiglu_fast_*`. Remplace 2 QMV série + 1 swiglu par un seul
    /// micro-kernel (tranche 3 : le shared-expert est le poste latence-bound du
    /// MoE). Écrit `silu(gate·x)·(up·x)` dans `output_buffer [out_dim]`.
    ///
    /// Renvoie `Ok(false)` (l'appelant garde le chemin 3-dispatch, résultat
    /// inchangé) si `gate`/`up` ne sont pas tous deux `AffineQuantized` fast
    /// (u4/gs64 ou u8/gs64|gs128) avec `in_dim % 512 == 0` et des dimensions
    /// identiques.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si une dimension déborde ou si l'encodage échoue.
    pub(crate) fn encode_gate_up_swiglu_fast(
        &self,
        encoder: &ComputeCommandEncoderRef,
        lhs_buffer: &BufferRef,
        gate: &Linear,
        up: &Linear,
        output_buffer: &BufferRef,
        in_dim: usize,
    ) -> Result<bool> {
        let (LinearWeight::AffineQuantized(gate_w), LinearWeight::AffineQuantized(up_w)) =
            (gate.weight(), up.weight())
        else {
            return Ok(false);
        };
        let ([gate_out, gate_in], [up_out, up_in]) = (gate_w.shape(), up_w.shape()) else {
            return Ok(false);
        };
        let ([_, gate_packed_cols], [_, up_packed_cols]) =
            (gate_w.packed_shape(), up_w.packed_shape())
        else {
            return Ok(false);
        };
        let gate_bits = gate_w.bits();
        let gate_group_size = gate_w.group_size();
        let eligible_quant = gate_bits == up_w.bits()
            && gate_group_size == up_w.group_size()
            && ((gate_bits == FAST_QMV_BITS && gate_group_size == FAST_QMV_GROUP_SIZE)
                || (gate_bits == 8
                    && matches!(gate_group_size, FAST_QMV_GROUP_SIZE | 128)
                    && *gate_out % 8 == 0));
        let eligible = eligible_quant
            && in_dim % 512 == 0
            && *gate_in == in_dim
            && *up_in == in_dim
            && gate_out == up_out
            && gate_packed_cols == up_packed_cols;
        if !eligible {
            return Ok(false);
        }
        let out_dim = *gate_out;
        let groups = in_dim
            .checked_div(gate_group_size)
            .ok_or_else(|| InferError::Metal("shared gate/up group_size nul".to_string()))?;
        let dims = [
            checked_u32(out_dim, "shared gate/up out_dim")?,
            checked_u32(in_dim, "shared gate/up in_dim")?,
            checked_u32(*gate_packed_cols, "shared gate/up packed_cols")?,
            checked_u32(groups, "shared gate/up groups")?,
        ];
        let gate_packed =
            self.cached_buffer_from_u32(gate_w.packed_data(), "shared_gate_packed")?;
        let gate_scales =
            self.cached_buffer_from_f32_as_bf16(gate_w.scales().data(), "shared_gate_scales")?;
        let gate_biases =
            self.cached_buffer_from_f32_as_bf16(gate_w.biases().data(), "shared_gate_biases")?;
        let up_packed = self.cached_buffer_from_u32(up_w.packed_data(), "shared_up_packed")?;
        let up_scales =
            self.cached_buffer_from_f32_as_bf16(up_w.scales().data(), "shared_up_scales")?;
        let up_biases =
            self.cached_buffer_from_f32_as_bf16(up_w.biases().data(), "shared_up_biases")?;
        let (pipeline, kernel_name) = if gate_bits == FAST_QMV_BITS {
            (
                &self.affine_gate_up_swiglu_fast_u4_gs64_f32,
                "affine_gate_up_swiglu_fast_u4_gs64_f32",
            )
        } else if gate_group_size == FAST_QMV_GROUP_SIZE {
            (
                &self.affine_gate_up_swiglu_fast_u8_gs64_f32,
                "affine_gate_up_swiglu_fast_u8_gs64_f32",
            )
        } else {
            (
                &self.affine_gate_up_swiglu_fast_u8_gs128_f32,
                "affine_gate_up_swiglu_fast_u8_gs128_f32",
            )
        };
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(lhs_buffer), 0);
        encoder.set_buffer(1, Some(&gate_packed), 0);
        encoder.set_buffer(2, Some(&gate_scales), 0);
        encoder.set_buffer(3, Some(&gate_biases), 0);
        encoder.set_buffer(4, Some(&up_packed), 0);
        encoder.set_buffer(5, Some(&up_scales), 0);
        encoder.set_buffer(6, Some(&up_biases), 0);
        encoder.set_buffer(7, Some(output_buffer), 0);
        set_u32_bytes(encoder, 8, &dims, "shared_gate_up_dims")?;
        trace_dispatch_path(kernel_name, 1, out_dim, in_dim);
        profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(
                1,
                checked_nsuint(out_dim.div_ceil(8), "shared gate/up out groups")?,
                1,
            ),
            MTLSize::new(64, 1, 1),
        );
        post_dispatch_barrier(encoder);
        Ok(true)
    }

    pub(crate) fn encode_gate_up_swiglu_fast_buffers(
        &self,
        encoder: &ComputeCommandEncoderRef,
        lhs_buffer: &BufferRef,
        gate: &MetalLinearWeightBuffers,
        up: &MetalLinearWeightBuffers,
        output_buffer: &BufferRef,
        in_dim: usize,
    ) -> Result<bool> {
        let (
            MetalLinearWeightBuffers::AffineQuantized {
                packed: gate_packed,
                scales: gate_scales,
                biases: gate_biases,
                out_dim: gate_out,
                in_dim: gate_in,
                packed_cols: gate_packed_cols,
                group_size: gate_group_size,
                bits: gate_bits,
                groups,
            },
            MetalLinearWeightBuffers::AffineQuantized {
                packed: up_packed,
                scales: up_scales,
                biases: up_biases,
                out_dim: up_out,
                in_dim: up_in,
                packed_cols: up_packed_cols,
                group_size: up_group_size,
                bits: up_bits,
                ..
            },
        ) = (gate, up)
        else {
            return Ok(false);
        };
        let eligible_quant = *gate_bits == *up_bits
            && *gate_group_size == *up_group_size
            && ((*gate_bits == FAST_QMV_BITS && *gate_group_size == FAST_QMV_GROUP_SIZE)
                || (*gate_bits == 8
                    && matches!(*gate_group_size, FAST_QMV_GROUP_SIZE | 128)
                    && *gate_out % 8 == 0));
        let eligible = eligible_quant
            && in_dim % 512 == 0
            && *gate_in == in_dim
            && *up_in == in_dim
            && gate_out == up_out
            && gate_packed_cols == up_packed_cols;
        if !eligible {
            return Ok(false);
        }
        let dims = [
            checked_u32(*gate_out, "shared gate/up out_dim")?,
            checked_u32(in_dim, "shared gate/up in_dim")?,
            checked_u32(*gate_packed_cols, "shared gate/up packed_cols")?,
            checked_u32(*groups, "shared gate/up groups")?,
        ];
        let (pipeline, kernel_name) = if *gate_bits == FAST_QMV_BITS {
            (
                &self.affine_gate_up_swiglu_fast_u4_gs64_f32,
                "affine_gate_up_swiglu_fast_u4_gs64_f32",
            )
        } else if *gate_group_size == FAST_QMV_GROUP_SIZE {
            (
                &self.affine_gate_up_swiglu_fast_u8_gs64_f32,
                "affine_gate_up_swiglu_fast_u8_gs64_f32",
            )
        } else {
            (
                &self.affine_gate_up_swiglu_fast_u8_gs128_f32,
                "affine_gate_up_swiglu_fast_u8_gs128_f32",
            )
        };
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(lhs_buffer), 0);
        encoder.set_buffer(1, Some(gate_packed), 0);
        encoder.set_buffer(2, Some(gate_scales), 0);
        encoder.set_buffer(3, Some(gate_biases), 0);
        encoder.set_buffer(4, Some(up_packed), 0);
        encoder.set_buffer(5, Some(up_scales), 0);
        encoder.set_buffer(6, Some(up_biases), 0);
        encoder.set_buffer(7, Some(output_buffer), 0);
        set_u32_bytes(encoder, 8, &dims, "shared_gate_up_dims")?;
        trace_dispatch_path(kernel_name, 1, *gate_out, in_dim);
        profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(
                1,
                checked_nsuint(gate_out.div_ceil(8), "shared gate/up out groups")?,
                1,
            ),
            MTLSize::new(64, 1, 1),
        );
        post_dispatch_barrier(encoder);
        Ok(true)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "fusion QMV shared gate_proj + gate scalaire"
    )]
    pub(crate) fn encode_qmv_plus_shared_gate_fast_buffers(
        &self,
        encoder: &ComputeCommandEncoderRef,
        lhs_buffer: &BufferRef,
        weight: &MetalLinearWeightBuffers,
        shared_gate: &MetalLinearWeightBuffers,
        output_buffer: &BufferRef,
        shared_gate_buffer: &BufferRef,
        in_dim: usize,
    ) -> Result<bool> {
        let (
            MetalLinearWeightBuffers::AffineQuantized {
                packed,
                scales,
                biases,
                out_dim,
                in_dim: weight_in,
                packed_cols,
                group_size,
                bits,
                groups,
            },
            MetalLinearWeightBuffers::AffineQuantized {
                packed: shared_gate_packed,
                scales: shared_gate_scales,
                biases: shared_gate_biases,
                out_dim: shared_gate_out,
                in_dim: shared_gate_in,
                packed_cols: shared_gate_packed_cols,
                group_size: shared_gate_group_size,
                bits: shared_gate_bits,
                groups: shared_gate_groups,
            },
        ) = (weight, shared_gate)
        else {
            return Ok(false);
        };
        let eligible = *bits == 8
            && *shared_gate_bits == 8
            && *group_size == FAST_QMV_GROUP_SIZE
            && *shared_gate_group_size == FAST_QMV_GROUP_SIZE
            && in_dim % 512 == 0
            && *weight_in == in_dim
            && *shared_gate_in == in_dim
            && *shared_gate_out == 1
            && *out_dim % 8 == 0
            && packed_cols == shared_gate_packed_cols
            && groups == shared_gate_groups;
        if !eligible {
            return Ok(false);
        }
        let dims = [
            checked_u32(*out_dim, "shared gate qmv out_dim")?,
            checked_u32(in_dim, "shared gate qmv in_dim")?,
            checked_u32(*packed_cols, "shared gate qmv packed_cols")?,
            checked_u32(*groups, "shared gate qmv groups")?,
        ];
        encoder.set_compute_pipeline_state(&self.affine_qmv_plus_one_fast_aligned_u8_gs64_f32);
        encoder.set_buffer(0, Some(lhs_buffer), 0);
        encoder.set_buffer(1, Some(packed), 0);
        encoder.set_buffer(2, Some(scales), 0);
        encoder.set_buffer(3, Some(biases), 0);
        encoder.set_buffer(4, Some(shared_gate_packed), 0);
        encoder.set_buffer(5, Some(shared_gate_scales), 0);
        encoder.set_buffer(6, Some(shared_gate_biases), 0);
        encoder.set_buffer(7, Some(output_buffer), 0);
        encoder.set_buffer(8, Some(shared_gate_buffer), 0);
        set_u32_bytes(encoder, 9, &dims, "shared_gate_qmv_dims")?;
        trace_dispatch_path(
            "affine_qmv_plus_one_fast_aligned_u8_gs64_f32",
            1,
            out_dim.saturating_add(1),
            in_dim,
        );
        profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(
                1,
                checked_nsuint(
                    out_dim.div_ceil(8).saturating_add(1),
                    "shared gate qmv groups",
                )?,
                1,
            ),
            MTLSize::new(64, 1, 1),
        );
        post_dispatch_barrier(encoder);
        Ok(true)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "fusion shared-expert: deux projections + gate scalaire"
    )]
    pub(crate) fn encode_gate_up_swiglu_shared_gate_fast_buffers(
        &self,
        encoder: &ComputeCommandEncoderRef,
        lhs_buffer: &BufferRef,
        gate: &MetalLinearWeightBuffers,
        up: &MetalLinearWeightBuffers,
        shared_gate: &MetalLinearWeightBuffers,
        output_buffer: &BufferRef,
        shared_gate_buffer: &BufferRef,
        in_dim: usize,
    ) -> Result<bool> {
        let (
            MetalLinearWeightBuffers::AffineQuantized {
                packed: gate_packed,
                scales: gate_scales,
                biases: gate_biases,
                out_dim: gate_out,
                in_dim: gate_in,
                packed_cols: gate_packed_cols,
                group_size: gate_group_size,
                bits: gate_bits,
                groups,
            },
            MetalLinearWeightBuffers::AffineQuantized {
                packed: up_packed,
                scales: up_scales,
                biases: up_biases,
                out_dim: up_out,
                in_dim: up_in,
                packed_cols: up_packed_cols,
                group_size: up_group_size,
                bits: up_bits,
                ..
            },
            MetalLinearWeightBuffers::AffineQuantized {
                packed: shared_gate_packed,
                scales: shared_gate_scales,
                biases: shared_gate_biases,
                out_dim: shared_gate_out,
                in_dim: shared_gate_in,
                packed_cols: shared_gate_packed_cols,
                group_size: shared_gate_group_size,
                bits: shared_gate_bits,
                groups: shared_gate_groups,
            },
        ) = (gate, up, shared_gate)
        else {
            return Ok(false);
        };
        let eligible = *gate_bits == 8
            && *up_bits == 8
            && *shared_gate_bits == 8
            && matches!(*gate_group_size, FAST_QMV_GROUP_SIZE | 128)
            && *up_group_size == *gate_group_size
            && *shared_gate_group_size == *gate_group_size
            && in_dim % 512 == 0
            && *gate_in == in_dim
            && *up_in == in_dim
            && *shared_gate_in == in_dim
            && gate_out == up_out
            && *shared_gate_out == 1
            && *gate_out % 8 == 0
            && gate_packed_cols == up_packed_cols
            && gate_packed_cols == shared_gate_packed_cols
            && groups == shared_gate_groups;
        if !eligible {
            return Ok(false);
        }
        let dims = [
            checked_u32(*gate_out, "shared gate/up+scalar out_dim")?,
            checked_u32(in_dim, "shared gate/up+scalar in_dim")?,
            checked_u32(*gate_packed_cols, "shared gate/up+scalar packed_cols")?,
            checked_u32(*groups, "shared gate/up+scalar groups")?,
        ];
        let (pipeline, kernel_name) = match (*gate_group_size, *shared_gate_group_size) {
            (FAST_QMV_GROUP_SIZE, FAST_QMV_GROUP_SIZE) => (
                &self.affine_gate_up_swiglu_gate_fast_u8_gs64_f32,
                "affine_gate_up_swiglu_gate_fast_u8_gs64_f32",
            ),
            (128, 128) => (
                &self.affine_gate_up_swiglu_gate_fast_u8_gs128_f32,
                "affine_gate_up_swiglu_gate_fast_u8_gs128_f32",
            ),
            _ => return Ok(false),
        };
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(lhs_buffer), 0);
        encoder.set_buffer(1, Some(gate_packed), 0);
        encoder.set_buffer(2, Some(gate_scales), 0);
        encoder.set_buffer(3, Some(gate_biases), 0);
        encoder.set_buffer(4, Some(up_packed), 0);
        encoder.set_buffer(5, Some(up_scales), 0);
        encoder.set_buffer(6, Some(up_biases), 0);
        encoder.set_buffer(7, Some(shared_gate_packed), 0);
        encoder.set_buffer(8, Some(shared_gate_scales), 0);
        encoder.set_buffer(9, Some(shared_gate_biases), 0);
        encoder.set_buffer(10, Some(output_buffer), 0);
        encoder.set_buffer(11, Some(shared_gate_buffer), 0);
        set_u32_bytes(encoder, 12, &dims, "shared_gate_up_scalar_dims")?;
        trace_dispatch_path(kernel_name, 1, gate_out.saturating_add(1), in_dim);
        profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(
                1,
                checked_nsuint(
                    gate_out.div_ceil(8).saturating_add(1),
                    "shared gate/up+scalar out groups",
                )?,
                1,
            ),
            MTLSize::new(64, 1, 1),
        );
        post_dispatch_barrier(encoder);
        Ok(true)
    }

    /// Wrapper per-op (commit + readback) du fusé gate+up+swiglu shared-expert, pour
    /// la validation différentielle isolée (==CPU). Renvoie `silu(gate·x)·(up·x)`.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si l'entrée n'est pas batch=1, si gate/up ne sont pas
    /// fast-éligibles, ou si l'encodage/lecture échoue.
    #[cfg(test)]
    pub(crate) fn gate_up_swiglu_fast(
        &self,
        input: &Tensor,
        gate: &Linear,
        up: &Linear,
    ) -> Result<Tensor> {
        let (batch, in_dim) = input.as_matrix()?;
        if batch != 1 {
            return Err(InferError::Dimension(format!(
                "gate_up_swiglu_fast attend batch=1, reçu {batch}"
            )));
        }
        let out_dim = linear_out_dim(gate.weight())?;
        let input_buffer = self.upload_f32_buffer(input.data(), "shared_gate_up_input")?;
        let output_buffer = self.new_f32_buffer(out_dim, "shared_gate_up_output")?;
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        if !self.encode_gate_up_swiglu_fast(
            encoder,
            &input_buffer,
            gate,
            up,
            &output_buffer,
            in_dim,
        )? {
            encoder.end_encoding();
            return Err(InferError::Config(
                "gate_up_swiglu_fast: gate/up non fast-éligibles".to_string(),
            ));
        }
        encoder.end_encoding();
        commit_and_wait(command_buffer)?;
        let output = read_f32_buffer(&output_buffer, out_dim)?;
        Tensor::from_vec(vec![1, out_dim], output)
    }

    pub(super) fn encode_weighted_sum_topk(
        &self,
        encoder: &ComputeCommandEncoderRef,
        _owned_buffers: &mut Vec<metal::Buffer>,
        src_buffer: &BufferRef,
        scores_buffer: &BufferRef,
        output_buffer: &BufferRef,
        topk: usize,
        out_dim: usize,
    ) -> Result<()> {
        let dims = [
            checked_u32(topk, "weighted topk")?,
            checked_u32(out_dim, "weighted out_dim")?,
        ];
        encoder.set_compute_pipeline_state(&self.weighted_sum_topk_f32);
        encoder.set_buffer(0, Some(src_buffer), 0);
        encoder.set_buffer(1, Some(scores_buffer), 0);
        encoder.set_buffer(2, Some(output_buffer), 0);
        set_u32_bytes(encoder, 3, &dims, "weighted_dims")?;
        trace_dispatch_path("weighted_sum_topk_f32", 1, out_dim, topk);
        self.dispatch_1d(encoder, &self.weighted_sum_topk_f32, out_dim)
    }

    pub(super) fn encode_weighted_sum_grouped_topk(
        &self,
        encoder: &ComputeCommandEncoderRef,
        _owned_buffers: &mut Vec<metal::Buffer>,
        src_buffer: &BufferRef,
        scores_buffer: &BufferRef,
        output_buffer: &BufferRef,
        rows: usize,
        topk_per_row: usize,
        out_dim: usize,
    ) -> Result<()> {
        let len = checked_len(rows, out_dim, "weighted grouped output")?;
        let dims = [
            checked_u32(rows, "weighted grouped rows")?,
            checked_u32(topk_per_row, "weighted grouped topk")?,
            checked_u32(out_dim, "weighted grouped out_dim")?,
            0,
        ];
        encoder.set_compute_pipeline_state(&self.weighted_sum_grouped_topk_f32);
        encoder.set_buffer(0, Some(src_buffer), 0);
        encoder.set_buffer(1, Some(scores_buffer), 0);
        encoder.set_buffer(2, Some(output_buffer), 0);
        set_u32_bytes(encoder, 3, &dims, "weighted_grouped_dims")?;
        trace_dispatch_path("weighted_sum_grouped_topk_f32", rows, out_dim, topk_per_row);
        self.dispatch_1d(encoder, &self.weighted_sum_grouped_topk_f32, len)
    }

    pub(super) fn encode_weighted_sum_add_grouped_topk(
        &self,
        encoder: &ComputeCommandEncoderRef,
        _owned_buffers: &mut Vec<metal::Buffer>,
        src_buffer: &BufferRef,
        scores_buffer: &BufferRef,
        residual_buffer: &BufferRef,
        output_buffer: &BufferRef,
        rows: usize,
        topk_per_row: usize,
        out_dim: usize,
    ) -> Result<()> {
        let len = checked_len(rows, out_dim, "weighted grouped add output")?;
        let dims = [
            checked_u32(rows, "weighted grouped add rows")?,
            checked_u32(topk_per_row, "weighted grouped add topk")?,
            checked_u32(out_dim, "weighted grouped add out_dim")?,
            0,
        ];
        encoder.set_compute_pipeline_state(&self.weighted_sum_add_grouped_topk_f32);
        encoder.set_buffer(0, Some(src_buffer), 0);
        encoder.set_buffer(1, Some(scores_buffer), 0);
        encoder.set_buffer(2, Some(residual_buffer), 0);
        encoder.set_buffer(3, Some(output_buffer), 0);
        set_u32_bytes(encoder, 4, &dims, "weighted_grouped_add_dims")?;
        trace_dispatch_path(
            "weighted_sum_add_grouped_topk_f32",
            rows,
            out_dim,
            topk_per_row,
        );
        self.dispatch_1d(encoder, &self.weighted_sum_add_grouped_topk_f32, len)
    }

    pub(crate) fn encode_weighted_sum_add_topk(
        &self,
        encoder: &ComputeCommandEncoderRef,
        _owned_buffers: &mut Vec<metal::Buffer>,
        src_buffer: &BufferRef,
        scores_buffer: &BufferRef,
        residual_buffer: &BufferRef,
        output_buffer: &BufferRef,
        topk: usize,
        out_dim: usize,
    ) -> Result<()> {
        let dims = [
            checked_u32(topk, "weighted add topk")?,
            checked_u32(out_dim, "weighted add out_dim")?,
        ];
        encoder.set_compute_pipeline_state(&self.weighted_sum_add_topk_f32);
        encoder.set_buffer(0, Some(src_buffer), 0);
        encoder.set_buffer(1, Some(scores_buffer), 0);
        encoder.set_buffer(2, Some(residual_buffer), 0);
        encoder.set_buffer(3, Some(output_buffer), 0);
        set_u32_bytes(encoder, 4, &dims, "weighted_add_dims")?;
        trace_dispatch_path("weighted_sum_add_topk_f32", 1, out_dim, topk);
        self.dispatch_1d(encoder, &self.weighted_sum_add_topk_f32, out_dim)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "epilogue MoE shared: topk + residual + shared gate"
    )]
    pub(crate) fn encode_weighted_sum_add_shared_topk(
        &self,
        encoder: &ComputeCommandEncoderRef,
        src_buffer: &BufferRef,
        scores_buffer: &BufferRef,
        residual_buffer: &BufferRef,
        shared_buffer: &BufferRef,
        shared_gate_buffer: &BufferRef,
        output_buffer: &BufferRef,
        topk: usize,
        out_dim: usize,
    ) -> Result<()> {
        let dims = [
            checked_u32(topk, "weighted shared topk")?,
            checked_u32(out_dim, "weighted shared out_dim")?,
        ];
        encoder.set_compute_pipeline_state(&self.weighted_sum_add_shared_topk_f32);
        encoder.set_buffer(0, Some(src_buffer), 0);
        encoder.set_buffer(1, Some(scores_buffer), 0);
        encoder.set_buffer(2, Some(residual_buffer), 0);
        encoder.set_buffer(3, Some(shared_buffer), 0);
        encoder.set_buffer(4, Some(shared_gate_buffer), 0);
        encoder.set_buffer(5, Some(output_buffer), 0);
        set_u32_bytes(encoder, 6, &dims, "weighted_shared_dims")?;
        trace_dispatch_path("weighted_sum_add_shared_topk_f32", 1, out_dim, topk);
        self.dispatch_1d(encoder, &self.weighted_sum_add_shared_topk_f32, out_dim)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "fusion MoE down_proj + somme top-k + shared expert"
    )]
    pub(crate) fn encode_gather_down_weighted_shared_u8_gs64(
        &self,
        encoder: &ComputeCommandEncoderRef,
        lhs_buffer: &BufferRef,
        weight: &StackedAffineBuffers,
        indices_buffer: &BufferRef,
        scores_buffer: &BufferRef,
        residual_buffer: &BufferRef,
        shared_buffer: &BufferRef,
        shared_gate_buffer: &BufferRef,
        output_buffer: &BufferRef,
        topk: usize,
    ) -> Result<bool> {
        if !fused_moe_down_weighted_u8_enabled() {
            return Ok(false);
        }
        let eligible = weight.bits == 8
            && weight.group_size == FAST_QMV_GROUP_SIZE
            && weight.in_dim % 512 == 0
            && weight.out_dim % 8 == 0
            && topk > 0;
        if !eligible {
            return Ok(false);
        }
        let dims = [
            checked_u32(topk, "down weighted topk")?,
            checked_u32(weight.out_dim, "down weighted out_dim")?,
            checked_u32(weight.in_dim, "down weighted in_dim")?,
            checked_u32(weight.packed_cols, "down weighted packed_cols")?,
        ];
        let groups = checked_u32(weight.groups, "down weighted groups")?;
        encoder
            .set_compute_pipeline_state(&self.affine_gather_down_weighted_shared_fast_u8_gs64_f32);
        encoder.set_buffer(0, Some(lhs_buffer), 0);
        encoder.set_buffer(1, Some(&weight.packed), 0);
        encoder.set_buffer(2, Some(&weight.scales), 0);
        encoder.set_buffer(3, Some(&weight.biases), 0);
        encoder.set_buffer(4, Some(indices_buffer), 0);
        encoder.set_buffer(5, Some(scores_buffer), 0);
        encoder.set_buffer(6, Some(residual_buffer), 0);
        encoder.set_buffer(7, Some(shared_buffer), 0);
        encoder.set_buffer(8, Some(shared_gate_buffer), 0);
        encoder.set_buffer(9, Some(output_buffer), 0);
        set_u32_bytes(encoder, 10, &dims, "down_weighted_dims")?;
        set_u32_bytes(encoder, 11, &[groups], "down_weighted_groups")?;
        profile_dispatch_shape(DispatchProfileShape::gather(
            "gather_down_weighted_shared_u8_gs64",
            topk,
            topk,
            weight.in_dim,
            weight.out_dim,
            weight.group_size,
            weight.bits,
        ));
        trace_dispatch_path(
            "affine_gather_down_weighted_shared_fast_u8_gs64_f32",
            topk,
            weight.out_dim,
            weight.in_dim,
        );
        profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(
                checked_nsuint(weight.out_dim.div_ceil(8), "down weighted out groups")?,
                1,
                1,
            ),
            MTLSize::new(64, 1, 1),
        );
        post_dispatch_barrier_buffer(encoder, output_buffer);
        Ok(true)
    }

    pub(super) fn encode_add_sigmoid_scaled(
        &self,
        encoder: &ComputeCommandEncoderRef,
        src_buffer: &BufferRef,
        gate_buffer: &BufferRef,
        dst_buffer: &BufferRef,
        len: usize,
    ) -> Result<()> {
        let len_u32 = checked_u32(len, "sigmoid scaled add len")?;
        encoder.set_compute_pipeline_state(&self.add_sigmoid_scaled_f32);
        encoder.set_buffer(0, Some(src_buffer), 0);
        encoder.set_buffer(1, Some(gate_buffer), 0);
        encoder.set_buffer(2, Some(dst_buffer), 0);
        set_u32_bytes(encoder, 3, &[len_u32], "sigmoid_scaled_add_len")?;
        trace_dispatch_path("add_sigmoid_scaled_f32", 1, len, 0);
        self.dispatch_1d(encoder, &self.add_sigmoid_scaled_f32, len)
    }

    pub(super) fn encode_add_sigmoid_scaled_rows(
        &self,
        encoder: &ComputeCommandEncoderRef,
        src_buffer: &BufferRef,
        gate_buffer: &BufferRef,
        dst_buffer: &BufferRef,
        rows: usize,
        row_dim: usize,
    ) -> Result<()> {
        if rows == 0 || row_dim == 0 {
            return Err(InferError::Dimension(format!(
                "sigmoid scaled rows invalide rows={rows}, row_dim={row_dim}"
            )));
        }
        let len = checked_len(rows, row_dim, "sigmoid scaled rows len")?;
        let dims = [
            checked_u32(rows, "sigmoid scaled rows")?,
            checked_u32(row_dim, "sigmoid scaled row dim")?,
        ];
        encoder.set_compute_pipeline_state(&self.add_sigmoid_scaled_rows_f32);
        encoder.set_buffer(0, Some(src_buffer), 0);
        encoder.set_buffer(1, Some(gate_buffer), 0);
        encoder.set_buffer(2, Some(dst_buffer), 0);
        set_u32_bytes(encoder, 3, &dims, "sigmoid_scaled_rows_dims")?;
        trace_dispatch_path("add_sigmoid_scaled_rows_f32", rows, row_dim, 0);
        self.dispatch_1d(encoder, &self.add_sigmoid_scaled_rows_f32, len)
    }

    pub(super) fn encode_topk_softmax(
        &self,
        encoder: &ComputeCommandEncoderRef,
        _owned_buffers: &mut Vec<metal::Buffer>,
        logits_buffer: &BufferRef,
        indices_buffer: &BufferRef,
        scores_buffer: &BufferRef,
        count: usize,
        topk: usize,
    ) -> Result<()> {
        ensure_valid_top_k(topk, count)?;
        let dims = [
            checked_u32(count, "topk count")?,
            checked_u32(topk, "topk")?,
        ];
        let parallel = topk_parallel_enabled();
        let fast_topk8 = topk8_fast_enabled() && count == 256 && topk == 8;
        let (pipeline, kernel_name) = if fast_topk8 {
            (&self.topk8_softmax_256_f32, "topk8_softmax_256_f32")
        } else if parallel {
            (&self.topk_softmax_f32, "topk_softmax_f32")
        } else {
            (&self.topk_softmax_serial_f32, "topk_softmax_serial_f32")
        };
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(logits_buffer), 0);
        encoder.set_buffer(1, Some(indices_buffer), 0);
        encoder.set_buffer(2, Some(scores_buffer), 0);
        set_u32_bytes(encoder, 3, &dims, "topk_dims")?;
        trace_dispatch_path(kernel_name, 1, topk, count);
        profile_dispatch();
        if fast_topk8 || parallel {
            encoder.dispatch_thread_groups(
                MTLSize::new(1, 1, 1),
                MTLSize::new(TOPK_SOFTMAX_TG_WIDTH, 1, 1),
            );
        } else {
            let width = pipeline.thread_execution_width().max(1);
            encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(width, 1, 1));
        }
        post_dispatch_barrier(encoder);
        Ok(())
    }

    /// Microbench isolé du kernel top-k decode (router 256 -> top-8).
    ///
    /// Encode plusieurs dispatchs dans un seul command buffer : le temps mesuré
    /// est le wait GPU du paquet, pas l'encode CPU par dispatch.
    pub(crate) fn profile_topk_softmax_kernel(
        &self,
        count: usize,
        topk: usize,
        iters: usize,
    ) -> Result<String> {
        if iters == 0 {
            return Err(InferError::Dimension("topk bench iters nul".to_string()));
        }
        let logits: Vec<f32> = (0..count)
            .map(|idx| {
                let x = (idx as f32 * 0.017_453_292).sin();
                x * 3.0 + (idx % 7) as f32 * 0.01
            })
            .collect();
        let logits_buffer = self.buffer_from_f32(&logits, "topk_bench_logits")?;
        let indices_buffer = self.private_u32_buffer(topk, "topk_bench_indices")?;
        let scores_buffer = self.private_f32_buffer(topk, "topk_bench_scores")?;
        let mut owned = Vec::new();

        let warmup = self.queue.new_command_buffer();
        let warmup_encoder = warmup.new_compute_command_encoder();
        self.encode_topk_softmax(
            warmup_encoder,
            &mut owned,
            &logits_buffer,
            &indices_buffer,
            &scores_buffer,
            count,
            topk,
        )?;
        warmup_encoder.end_encoding();
        warmup.commit();
        warmup.wait_until_completed();
        ensure_completed(warmup.status())?;

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let encode_started = std::time::Instant::now();
        for _ in 0..iters {
            self.encode_topk_softmax(
                encoder,
                &mut owned,
                &logits_buffer,
                &indices_buffer,
                &scores_buffer,
                count,
                topk,
            )?;
        }
        let encode_us = encode_started.elapsed().as_secs_f64() * 1_000_000.0 / iters as f64;
        encoder.end_encoding();
        let wait_started = std::time::Instant::now();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        let wait_us = wait_started.elapsed().as_secs_f64() * 1_000_000.0 / iters as f64;
        ensure_completed(command_buffer.status())?;
        Ok(format!(
            "topk microbench count={count} topk={topk} iters={iters}: encode_us/dispatch={encode_us:.3} wait_us/dispatch={wait_us:.3}"
        ))
    }

    pub(super) fn encode_topk_softmax_rows(
        &self,
        encoder: &ComputeCommandEncoderRef,
        logits_buffer: &BufferRef,
        indices_buffer: &BufferRef,
        scores_buffer: &BufferRef,
        rows: usize,
        count: usize,
        topk: usize,
    ) -> Result<()> {
        if rows == 0 {
            return Err(InferError::Dimension(format!(
                "topk rows invalide rows={rows}, count={count}, topk={topk}"
            )));
        }
        ensure_valid_top_k(topk, count)?;
        let dims = [
            checked_u32(rows, "topk rows")?,
            checked_u32(count, "topk rows count")?,
            checked_u32(topk, "topk rows topk")?,
        ];
        encoder.set_compute_pipeline_state(&self.topk_softmax_rows_f32);
        encoder.set_buffer(0, Some(logits_buffer), 0);
        encoder.set_buffer(1, Some(indices_buffer), 0);
        encoder.set_buffer(2, Some(scores_buffer), 0);
        set_u32_bytes(encoder, 3, &dims, "topk_rows_dims")?;
        trace_dispatch_path("topk_softmax_rows_f32", rows, topk, count);
        self.dispatch_1d(encoder, &self.topk_softmax_rows_f32, rows)
    }
}
