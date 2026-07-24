//! Dispatch des matmuls à partir des poids hôtes.

use super::matmul_select::AffineMatmulKernel;
use super::*;

#[expect(
    clippy::too_many_arguments,
    reason = "dispatch Metal: buffers, dimensions et paramètres quantifiés restent explicites"
)]
impl MetalExecutor {
    #[inline]
    pub(super) fn encode_matmul_weight_inner(
        &self,
        encoder: &ComputeCommandEncoderRef,
        owned_buffers: &mut Vec<metal::Buffer>,
        lhs_buffer: &BufferRef,
        batch: usize,
        in_dim: usize,
        weight: &LinearWeight,
        output_buffer: &BufferRef,
        prefer_fast_affine: bool,
    ) -> Result<usize> {
        match weight {
            LinearWeight::Dense(weight) => self.encode_owned_dense_matmul(
                encoder,
                lhs_buffer,
                weight,
                output_buffer,
                batch,
                in_dim,
            ),
            LinearWeight::AffineQuantized(weight) => self.encode_owned_affine_matmul(
                encoder,
                owned_buffers,
                lhs_buffer,
                weight,
                output_buffer,
                batch,
                in_dim,
                prefer_fast_affine,
            ),
        }
    }

    #[inline]
    fn encode_owned_dense_matmul(
        &self,
        encoder: &ComputeCommandEncoderRef,
        lhs_buffer: &BufferRef,
        weight: &Tensor,
        output_buffer: &BufferRef,
        batch: usize,
        in_dim: usize,
    ) -> Result<usize> {
        let (out_dim, rhs_in_dim) = weight.as_matrix()?;
        if in_dim != rhs_in_dim {
            return Err(InferError::Dimension(format!(
                "matmul Metal encodé x=[{batch},{in_dim}] rhs=[{out_dim},{rhs_in_dim}]"
            )));
        }
        let rhs_buffer = self.cached_buffer_from_f32(weight.data(), "rhs")?;
        if can_use_dense_qmv_fast(batch, in_dim, out_dim) {
            self.encode_owned_dense_qmv_fast(
                encoder,
                lhs_buffer,
                &rhs_buffer,
                output_buffer,
                batch,
                in_dim,
                out_dim,
            )?;
        } else {
            self.encode_owned_dense_fallback(
                encoder,
                lhs_buffer,
                &rhs_buffer,
                output_buffer,
                batch,
                in_dim,
                out_dim,
            )?;
        }
        Ok(out_dim)
    }

    #[inline]
    fn encode_owned_dense_qmv_fast(
        &self,
        encoder: &ComputeCommandEncoderRef,
        lhs_buffer: &BufferRef,
        rhs_buffer: &BufferRef,
        output_buffer: &BufferRef,
        batch: usize,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<()> {
        let dims = [
            checked_u32(batch, "dense fast batch")?,
            checked_u32(out_dim, "dense fast out_dim")?,
            checked_u32(in_dim, "dense fast in_dim")?,
        ];
        encoder.set_compute_pipeline_state(&self.dense_qmv_fast_f32);
        encoder.set_buffer(0, Some(lhs_buffer), 0);
        encoder.set_buffer(1, Some(rhs_buffer), 0);
        encoder.set_buffer(2, Some(output_buffer), 0);
        set_u32_bytes(encoder, 3, &dims, "dense_fast_dims")?;
        trace_dispatch_path("dense_qmv_fast_f32", batch, out_dim, in_dim);
        profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(
                checked_nsuint(batch, "dense fast batch")?,
                checked_nsuint(out_dim.div_ceil(8), "dense fast out groups")?,
                1,
            ),
            MTLSize::new(64, 1, 1),
        );
        post_dispatch_barrier_buffer(encoder, output_buffer);
        Ok(())
    }

    #[inline]
    fn encode_owned_dense_fallback(
        &self,
        encoder: &ComputeCommandEncoderRef,
        lhs_buffer: &BufferRef,
        rhs_buffer: &BufferRef,
        output_buffer: &BufferRef,
        batch: usize,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<()> {
        let dims = [
            checked_u32(batch, "batch")?,
            checked_u32(out_dim, "out_dim")?,
            checked_u32(in_dim, "in_dim")?,
        ];
        encoder.set_compute_pipeline_state(&self.dense_matmul_rhs_t_f32);
        encoder.set_buffer(0, Some(lhs_buffer), 0);
        encoder.set_buffer(1, Some(rhs_buffer), 0);
        encoder.set_buffer(2, Some(output_buffer), 0);
        set_u32_bytes(encoder, 3, &dims, "dims")?;
        trace_dispatch_path("dense_matmul_rhs_t_f32", batch, out_dim, in_dim);
        self.dispatch_qmv(encoder, &self.dense_matmul_rhs_t_f32, out_dim, batch)
    }

    #[inline]
    fn encode_owned_affine_matmul(
        &self,
        encoder: &ComputeCommandEncoderRef,
        owned_buffers: &mut Vec<metal::Buffer>,
        lhs_buffer: &BufferRef,
        weight: &AffineQuantizedTensor,
        output_buffer: &BufferRef,
        batch: usize,
        in_dim: usize,
        prefer_fast_affine: bool,
    ) -> Result<usize> {
        let [out_dim, weight_in_dim] = weight.shape() else {
            return Err(InferError::Dimension(format!(
                "poids Metal quantifié attendu rang 2, reçu {:?}",
                weight.shape()
            )));
        };
        if in_dim != *weight_in_dim {
            return Err(InferError::Dimension(format!(
                "matmul Metal encodé quantifié x=[{batch},{in_dim}] rhs=[{out_dim},{weight_in_dim}]"
            )));
        }
        let [packed_rows, packed_cols] = weight.packed_shape() else {
            return Err(InferError::Dimension(format!(
                "packed_shape Metal attendu rang 2, reçu {:?}",
                weight.packed_shape()
            )));
        };
        if *packed_rows != *out_dim {
            return Err(InferError::Dimension(format!(
                "packed_rows={packed_rows} incompatible avec out_dim={out_dim}"
            )));
        }
        let groups = in_dim
            .checked_div(weight.group_size())
            .ok_or_else(|| InferError::Metal("group_size quantifié nul".to_string()))?;
        let packed_buffer = self.cached_buffer_from_u32(weight.packed_data(), "packed")?;
        let scales_buffer =
            self.cached_buffer_from_f32_as_bf16(weight.scales().data(), "scales")?;
        let biases_buffer =
            self.cached_buffer_from_f32_as_bf16(weight.biases().data(), "biases")?;
        let dims = [
            checked_u32(batch, "batch")?,
            checked_u32(*out_dim, "out_dim")?,
            checked_u32(in_dim, "in_dim")?,
            checked_u32(*packed_cols, "packed_cols")?,
        ];
        let quant = [
            checked_u32(weight.group_size(), "group_size")?,
            checked_u32(weight.bits(), "bits")?,
            checked_u32(groups, "groups")?,
            0,
        ];
        match self.select_owned_affine_matmul_kernel(batch, in_dim, weight, prefer_fast_affine) {
            AffineMatmulKernel::Qmm2 => self.encode_owned_affine_qmm2(
                encoder,
                lhs_buffer,
                &packed_buffer,
                &scales_buffer,
                &biases_buffer,
                output_buffer,
                batch,
                in_dim,
                *out_dim,
                *packed_cols,
                weight.group_size(),
                weight.bits(),
                groups,
            )?,
            AffineMatmulKernel::QmmNaFusedTiledU4 => {
                self.encode_affine_qmm_na_fused_tiled_u4_buffers(
                    encoder,
                    lhs_buffer,
                    &packed_buffer,
                    &scales_buffer,
                    &biases_buffer,
                    output_buffer,
                    batch,
                    in_dim,
                    *out_dim,
                )?;
            }
            AffineMatmulKernel::QmmNaFusedTiledU4Align64 => {
                self.encode_affine_qmm_na_fused_tiled_u4_align64_buffers(
                    encoder,
                    lhs_buffer,
                    &packed_buffer,
                    &scales_buffer,
                    &biases_buffer,
                    output_buffer,
                    batch,
                    in_dim,
                    *out_dim,
                )?;
            }
            AffineMatmulKernel::FastQmvU4 => self.encode_owned_affine_qmv_u4(
                encoder,
                lhs_buffer,
                &packed_buffer,
                &scales_buffer,
                &biases_buffer,
                output_buffer,
                batch,
                in_dim,
                *out_dim,
                *packed_cols,
                groups,
            )?,
            AffineMatmulKernel::FastQmvU4Align64 => self.encode_affine_qmv_align64_buffers(
                encoder,
                lhs_buffer,
                &packed_buffer,
                &scales_buffer,
                &biases_buffer,
                output_buffer,
                batch,
                in_dim,
                *out_dim,
                *packed_cols,
                groups,
                FAST_QMV_BITS,
            )?,
            AffineMatmulKernel::FastQmvU6 => self.encode_affine_qmv_u6_buffers(
                encoder,
                lhs_buffer,
                &packed_buffer,
                &scales_buffer,
                &biases_buffer,
                output_buffer,
                batch,
                in_dim,
                *out_dim,
                *packed_cols,
                groups,
            )?,
            AffineMatmulKernel::FastQmvOneU8 => self.encode_affine_qmv_one_u8_buffers(
                encoder,
                lhs_buffer,
                &packed_buffer,
                &scales_buffer,
                &biases_buffer,
                output_buffer,
                batch,
                in_dim,
                *packed_cols,
                groups,
            )?,
            AffineMatmulKernel::QmmNaFusedTiledU8 => {
                self.encode_affine_qmm_na_fused_tiled_u8_buffers(
                    encoder,
                    lhs_buffer,
                    &packed_buffer,
                    &scales_buffer,
                    &biases_buffer,
                    output_buffer,
                    batch,
                    in_dim,
                    *out_dim,
                    weight.group_size(),
                )?;
            }
            AffineMatmulKernel::QmmNaU8Gs128 => self.encode_affine_qmm_na_qb_u8_buffers(
                encoder,
                lhs_buffer,
                &packed_buffer,
                &scales_buffer,
                &biases_buffer,
                output_buffer,
                batch,
                in_dim,
                *out_dim,
                weight.group_size(),
            )?,
            AffineMatmulKernel::QmmNaU8 => self.encode_owned_affine_qmm_na_u8(
                encoder,
                owned_buffers,
                lhs_buffer,
                &packed_buffer,
                &scales_buffer,
                &biases_buffer,
                output_buffer,
                batch,
                in_dim,
                *out_dim,
                *packed_cols,
            )?,
            AffineMatmulKernel::FastQmvU8 => self.encode_owned_affine_qmv_u8(
                encoder,
                lhs_buffer,
                &packed_buffer,
                &scales_buffer,
                &biases_buffer,
                output_buffer,
                batch,
                in_dim,
                *out_dim,
                *packed_cols,
                weight.group_size(),
                groups,
            )?,
            AffineMatmulKernel::FastQmvU8Align64 => self.encode_affine_qmv_align64_buffers(
                encoder,
                lhs_buffer,
                &packed_buffer,
                &scales_buffer,
                &biases_buffer,
                output_buffer,
                batch,
                in_dim,
                *out_dim,
                *packed_cols,
                groups,
                8,
            )?,
            AffineMatmulKernel::Fallback => self.encode_owned_affine_fallback(
                encoder,
                lhs_buffer,
                &packed_buffer,
                &scales_buffer,
                &biases_buffer,
                output_buffer,
                batch,
                in_dim,
                *out_dim,
                &dims,
                &quant,
            )?,
        }
        Ok(*out_dim)
    }

    #[inline]
    fn encode_owned_affine_qmm2(
        &self,
        encoder: &ComputeCommandEncoderRef,
        lhs_buffer: &BufferRef,
        packed_buffer: &BufferRef,
        scales_buffer: &BufferRef,
        biases_buffer: &BufferRef,
        output_buffer: &BufferRef,
        batch: usize,
        in_dim: usize,
        out_dim: usize,
        packed_cols: usize,
        group_size: usize,
        bits: usize,
        groups: usize,
    ) -> Result<()> {
        let fast_dims = [
            checked_u32(out_dim, "qmm2 out_dim")?,
            checked_u32(in_dim, "qmm2 in_dim")?,
            checked_u32(packed_cols, "qmm2 packed_cols")?,
            checked_u32(groups, "qmm2 groups")?,
        ];
        let (pipeline, kernel_name) = if bits == FAST_QMV_BITS {
            (
                &self.affine_qmm2_fast_aligned_u4_gs64_f32,
                "affine_qmm2_fast_aligned_u4_gs64_f32",
            )
        } else if group_size == FAST_QMV_GROUP_SIZE {
            (
                &self.affine_qmm2_fast_aligned_u8_gs64_f32,
                "affine_qmm2_fast_aligned_u8_gs64_f32",
            )
        } else {
            (
                &self.affine_qmm2_fast_aligned_u8_gs128_f32,
                "affine_qmm2_fast_aligned_u8_gs128_f32",
            )
        };
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(lhs_buffer), 0);
        encoder.set_buffer(1, Some(packed_buffer), 0);
        encoder.set_buffer(2, Some(scales_buffer), 0);
        encoder.set_buffer(3, Some(biases_buffer), 0);
        encoder.set_buffer(4, Some(output_buffer), 0);
        set_u32_bytes(encoder, 5, &fast_dims, "qmm2_dims")?;
        trace_dispatch_path(kernel_name, batch, out_dim, in_dim);
        profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(
                1,
                checked_nsuint(out_dim.div_ceil(8), "qmm2 out groups")?,
                1,
            ),
            MTLSize::new(64, 1, 1),
        );
        post_dispatch_barrier_buffer(encoder, output_buffer);
        Ok(())
    }

    #[inline]
    fn encode_owned_affine_qmv_u4(
        &self,
        encoder: &ComputeCommandEncoderRef,
        lhs_buffer: &BufferRef,
        packed_buffer: &BufferRef,
        scales_buffer: &BufferRef,
        biases_buffer: &BufferRef,
        output_buffer: &BufferRef,
        batch: usize,
        in_dim: usize,
        out_dim: usize,
        packed_cols: usize,
        groups: usize,
    ) -> Result<()> {
        let fast_dims = [
            checked_u32(out_dim, "fast out_dim")?,
            checked_u32(in_dim, "fast in_dim")?,
            checked_u32(packed_cols, "fast packed_cols")?,
            checked_u32(groups, "fast groups")?,
        ];
        let (pipeline, kernel_name) = if out_dim % 8 == 0 {
            (
                &self.affine_qmv_fast_aligned_u4_gs64_f32,
                "affine_qmv_fast_aligned_u4_gs64_f32",
            )
        } else {
            (
                &self.affine_qmv_fast_u4_gs64_f32,
                "affine_qmv_fast_u4_gs64_f32",
            )
        };
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(lhs_buffer), 0);
        encoder.set_buffer(1, Some(packed_buffer), 0);
        encoder.set_buffer(2, Some(scales_buffer), 0);
        encoder.set_buffer(3, Some(biases_buffer), 0);
        encoder.set_buffer(4, Some(output_buffer), 0);
        set_u32_bytes(encoder, 5, &fast_dims, "fast_dims")?;
        trace_dispatch_path(kernel_name, batch, out_dim, in_dim);
        profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(
                checked_nsuint(batch, "batch")?,
                checked_nsuint(out_dim.div_ceil(8), "fast out groups")?,
                1,
            ),
            MTLSize::new(64, 1, 1),
        );
        post_dispatch_barrier_buffer(encoder, output_buffer);
        Ok(())
    }

    #[inline]
    fn encode_owned_affine_qmm_na_u8(
        &self,
        encoder: &ComputeCommandEncoderRef,
        owned_buffers: &mut Vec<metal::Buffer>,
        lhs_buffer: &BufferRef,
        packed_buffer: &BufferRef,
        scales_buffer: &BufferRef,
        biases_buffer: &BufferRef,
        output_buffer: &BufferRef,
        batch: usize,
        in_dim: usize,
        out_dim: usize,
        packed_cols: usize,
    ) -> Result<()> {
        // PREFILL : GEMM bf16 sur Neural Accelerators (matmul2d). Dé-quant
        // u8→bf16 transposée du poids + activations bf16 → tensor-cores.
        trace_dispatch_path("qmm_na", batch, out_dim, in_dim);
        let wt_bf16 =
            self.private_bf16_buffer(checked_len(in_dim, out_dim, "qmm na wt")?, "qmm_na_wt_bf16")?;
        self.encode_dequant_qweight_to_bf16_t(
            encoder,
            packed_buffer,
            scales_buffer,
            biases_buffer,
            &wt_bf16,
            out_dim,
            in_dim,
            packed_cols,
        )?;
        let lhs_len = checked_len(batch, in_dim, "qmm na lhs")?;
        let lhs_bf16 = self.private_bf16_buffer(lhs_len, "qmm_na_lhs_bf16")?;
        self.encode_f32_to_bf16(encoder, lhs_buffer, &lhs_bf16, lhs_len)?;
        self.encode_na_gemm(
            encoder,
            &lhs_bf16,
            &wt_bf16,
            output_buffer,
            batch,
            out_dim,
            in_dim,
        )?;
        owned_buffers.push(wt_bf16);
        owned_buffers.push(lhs_bf16);
        Ok(())
    }

    #[inline]
    fn encode_owned_affine_qmv_u8(
        &self,
        encoder: &ComputeCommandEncoderRef,
        lhs_buffer: &BufferRef,
        packed_buffer: &BufferRef,
        scales_buffer: &BufferRef,
        biases_buffer: &BufferRef,
        output_buffer: &BufferRef,
        batch: usize,
        in_dim: usize,
        out_dim: usize,
        packed_cols: usize,
        group_size: usize,
        groups: usize,
    ) -> Result<()> {
        let fast_dims = [
            checked_u32(out_dim, "fast u8 out_dim")?,
            checked_u32(in_dim, "fast u8 in_dim")?,
            checked_u32(packed_cols, "fast u8 packed_cols")?,
            checked_u32(groups, "fast u8 groups")?,
        ];
        let use_dot4 = qmv_u8_dot4_enabled();
        let use_tg256 = !use_dot4 && qmv_u8_tg256_enabled();
        let use_tg128 = !use_dot4 && !use_tg256 && qmv_u8_tg128_enabled();
        let rows_per_threadgroup = if use_tg256 {
            32
        } else if use_tg128 {
            16
        } else {
            8
        };
        let threads_per_threadgroup = if use_tg256 {
            256
        } else if use_tg128 {
            128
        } else {
            64
        };
        let (pipeline, kernel_name) = match (
            group_size == FAST_QMV_GROUP_SIZE,
            use_dot4,
            use_tg256,
            use_tg128,
        ) {
            (true, true, _, _) => (
                &self.affine_qmv_fast_aligned_u8_gs64_dot4_f32,
                "affine_qmv_fast_aligned_u8_gs64_dot4_f32",
            ),
            (false, true, _, _) => (
                &self.affine_qmv_fast_aligned_u8_gs128_dot4_f32,
                "affine_qmv_fast_aligned_u8_gs128_dot4_f32",
            ),
            (true, false, true, _) => (
                &self.affine_qmv_fast_aligned_u8_gs64_tg256_f32,
                "affine_qmv_fast_aligned_u8_gs64_tg256_f32",
            ),
            (false, false, true, _) => (
                &self.affine_qmv_fast_aligned_u8_gs128_tg256_f32,
                "affine_qmv_fast_aligned_u8_gs128_tg256_f32",
            ),
            (true, false, false, true) => (
                &self.affine_qmv_fast_aligned_u8_gs64_tg128_f32,
                "affine_qmv_fast_aligned_u8_gs64_tg128_f32",
            ),
            (false, false, false, true) => (
                &self.affine_qmv_fast_aligned_u8_gs128_tg128_f32,
                "affine_qmv_fast_aligned_u8_gs128_tg128_f32",
            ),
            (true, false, false, false) => (
                &self.affine_qmv_fast_aligned_u8_gs64_f32,
                "affine_qmv_fast_aligned_u8_gs64_f32",
            ),
            (false, false, false, false) => (
                &self.affine_qmv_fast_aligned_u8_gs128_f32,
                "affine_qmv_fast_aligned_u8_gs128_f32",
            ),
        };
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(lhs_buffer), 0);
        encoder.set_buffer(1, Some(packed_buffer), 0);
        encoder.set_buffer(2, Some(scales_buffer), 0);
        encoder.set_buffer(3, Some(biases_buffer), 0);
        encoder.set_buffer(4, Some(output_buffer), 0);
        set_u32_bytes(encoder, 5, &fast_dims, "fast_u8_dims")?;
        trace_dispatch_path(kernel_name, batch, out_dim, in_dim);
        profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(
                checked_nsuint(batch, "batch")?,
                checked_nsuint(out_dim.div_ceil(rows_per_threadgroup), "fast u8 out groups")?,
                1,
            ),
            MTLSize::new(threads_per_threadgroup, 1, 1),
        );
        post_dispatch_barrier_buffer(encoder, output_buffer);
        Ok(())
    }

    #[inline]
    fn encode_owned_affine_fallback(
        &self,
        encoder: &ComputeCommandEncoderRef,
        lhs_buffer: &BufferRef,
        packed_buffer: &BufferRef,
        scales_buffer: &BufferRef,
        biases_buffer: &BufferRef,
        output_buffer: &BufferRef,
        batch: usize,
        in_dim: usize,
        out_dim: usize,
        dims: &[u32; 4],
        quant: &[u32; 4],
    ) -> Result<()> {
        encoder.set_compute_pipeline_state(&self.affine_matmul_rhs_t_u32_f32);
        encoder.set_buffer(0, Some(lhs_buffer), 0);
        encoder.set_buffer(1, Some(packed_buffer), 0);
        encoder.set_buffer(2, Some(scales_buffer), 0);
        encoder.set_buffer(3, Some(biases_buffer), 0);
        encoder.set_buffer(4, Some(output_buffer), 0);
        set_u32_bytes(encoder, 5, dims, "dims")?;
        set_u32_bytes(encoder, 6, quant, "quant")?;
        trace_dispatch_path("affine_matmul_rhs_t_u32_f32", batch, out_dim, in_dim);
        self.dispatch_qmv(encoder, &self.affine_matmul_rhs_t_u32_f32, out_dim, batch)
    }

    pub(super) fn encode_gather_matmul(
        &self,
        encoder: &ComputeCommandEncoderRef,
        _owned_buffers: &mut Vec<metal::Buffer>,
        lhs_buffer: &BufferRef,
        lhs_rows: usize,
        weight: &StackedAffineBuffers,
        indices_buffer: &BufferRef,
        topk: usize,
        output_buffer: &BufferRef,
    ) -> Result<()> {
        if !valid_gather_lhs_rows(lhs_rows, topk) {
            return Err(InferError::Dimension(format!(
                "gather matmul lhs_rows={lhs_rows}, topk={topk}"
            )));
        }
        let dims = [
            checked_u32(topk, "topk")?,
            checked_u32(weight.out_dim, "out_dim")?,
            checked_u32(weight.in_dim, "in_dim")?,
            checked_u32(weight.packed_cols, "packed_cols")?,
        ];
        let quant = [
            checked_u32(weight.group_size, "group_size")?,
            checked_u32(weight.bits, "bits")?,
            checked_u32(weight.groups, "groups")?,
            checked_u32(lhs_rows, "lhs_rows")?,
        ];
        if can_use_fast_gather_qmv(lhs_rows, weight) {
            let use_u8_tg256 = weight.bits == 8 && qmv_u8_tg256_enabled();
            let use_u8_tg128 = weight.bits == 8 && !use_u8_tg256 && qmv_u8_tg128_enabled();
            let rows_per_threadgroup = if use_u8_tg256 {
                32
            } else if use_u8_tg128 {
                16
            } else {
                8
            };
            let threads_per_threadgroup = if use_u8_tg256 {
                256
            } else if use_u8_tg128 {
                128
            } else {
                64
            };
            let (pipeline, profile_label, kernel_name) = if weight.bits == FAST_QMV_BITS
                && weight.in_dim % 512 == 0
            {
                (
                    &self.affine_gather_qmv_fast_u4_gs64_f32,
                    "gather_qmv_u4_gs64",
                    "affine_gather_qmv_fast_u4_gs64_f32",
                )
            } else if weight.bits == 8 && weight.group_size == FAST_QMV_GROUP_SIZE && use_u8_tg128 {
                (
                    &self.affine_gather_qmv_fast_u8_gs64_tg128_f32,
                    "gather_qmv_u8_gs64_tg128",
                    "affine_gather_qmv_fast_u8_gs64_tg128_f32",
                )
            } else if weight.bits == 8 && weight.group_size == FAST_QMV_GROUP_SIZE && use_u8_tg256 {
                (
                    &self.affine_gather_qmv_fast_u8_gs64_tg256_f32,
                    "gather_qmv_u8_gs64_tg256",
                    "affine_gather_qmv_fast_u8_gs64_tg256_f32",
                )
            } else if weight.bits == 8 && weight.group_size == FAST_QMV_GROUP_SIZE {
                (
                    &self.affine_gather_qmv_fast_u8_gs64_f32,
                    "gather_qmv_u8_gs64",
                    "affine_gather_qmv_fast_u8_gs64_f32",
                )
            } else if weight.bits == 8 && use_u8_tg128 {
                (
                    &self.affine_gather_qmv_fast_u8_gs128_tg128_f32,
                    "gather_qmv_u8_gs128_tg128",
                    "affine_gather_qmv_fast_u8_gs128_tg128_f32",
                )
            } else if weight.bits == 8 && use_u8_tg256 {
                (
                    &self.affine_gather_qmv_fast_u8_gs128_tg256_f32,
                    "gather_qmv_u8_gs128_tg256",
                    "affine_gather_qmv_fast_u8_gs128_tg256_f32",
                )
            } else if weight.bits == 8 {
                (
                    &self.affine_gather_qmv_fast_u8_gs128_f32,
                    "gather_qmv_u8_gs128",
                    "affine_gather_qmv_fast_u8_gs128_f32",
                )
            } else {
                (
                    &self.affine_gather_qmv_tail_u4_gs64_f32,
                    "gather_qmv_u4_tail",
                    "affine_gather_qmv_tail_u4_gs64_f32",
                )
            };
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(lhs_buffer), 0);
            encoder.set_buffer(1, Some(&weight.packed), 0);
            encoder.set_buffer(2, Some(&weight.scales), 0);
            encoder.set_buffer(3, Some(&weight.biases), 0);
            encoder.set_buffer(4, Some(indices_buffer), 0);
            encoder.set_buffer(5, Some(output_buffer), 0);
            set_u32_bytes(encoder, 6, &dims, "gather_dims")?;
            set_u32_bytes(encoder, 7, &quant, "gather_quant")?;
            profile_dispatch_shape(DispatchProfileShape::gather(
                profile_label,
                lhs_rows,
                topk,
                weight.in_dim,
                weight.out_dim,
                weight.group_size,
                weight.bits,
            ));
            trace_dispatch_path(kernel_name, topk, weight.out_dim, weight.in_dim);
            profile_dispatch();
            encoder.dispatch_thread_groups(
                MTLSize::new(
                    checked_nsuint(topk, "gather topk")?,
                    checked_nsuint(
                        weight.out_dim.div_ceil(rows_per_threadgroup),
                        "gather fast out groups",
                    )?,
                    1,
                ),
                MTLSize::new(threads_per_threadgroup, 1, 1),
            );
            post_dispatch_barrier_buffer(encoder, output_buffer);
            Ok(())
        } else {
            encoder.set_compute_pipeline_state(&self.affine_gather_matmul_rhs_t_u32_f32);
            encoder.set_buffer(0, Some(lhs_buffer), 0);
            encoder.set_buffer(1, Some(&weight.packed), 0);
            encoder.set_buffer(2, Some(&weight.scales), 0);
            encoder.set_buffer(3, Some(&weight.biases), 0);
            encoder.set_buffer(4, Some(indices_buffer), 0);
            encoder.set_buffer(5, Some(output_buffer), 0);
            set_u32_bytes(encoder, 6, &dims, "gather_dims")?;
            set_u32_bytes(encoder, 7, &quant, "gather_quant")?;
            trace_dispatch_path(
                "affine_gather_matmul_rhs_t_u32_f32",
                topk,
                weight.out_dim,
                weight.in_dim,
            );
            self.dispatch_qmv(
                encoder,
                &self.affine_gather_matmul_rhs_t_u32_f32,
                weight.out_dim,
                topk,
            )
        }
    }
}
