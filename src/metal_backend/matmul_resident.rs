//! Dispatch des matmuls dont les poids sont déjà résidents sur le GPU.

use super::matmul_select::AffineMatmulKernel;
use super::*;

#[expect(
    clippy::too_many_arguments,
    reason = "dispatch Metal: buffers, dimensions et paramètres quantifiés restent explicites"
)]
impl MetalExecutor {
    /// Encode un matmul résident avec des buffers de poids pré-résolus.
    #[inline]
    pub(crate) fn encode_matmul_weight_buffers(
        &self,
        encoder: &ComputeCommandEncoderRef,
        lhs_buffer: &BufferRef,
        batch: usize,
        in_dim: usize,
        weight: &MetalLinearWeightBuffers,
        output_buffer: &BufferRef,
        prefer_fast_affine: bool,
    ) -> Result<usize> {
        match weight {
            MetalLinearWeightBuffers::Dense {
                rhs,
                out_dim,
                in_dim: rhs_in_dim,
                ..
            } => self.encode_resident_dense_matmul(
                encoder,
                lhs_buffer,
                rhs,
                output_buffer,
                batch,
                in_dim,
                *out_dim,
                *rhs_in_dim,
            ),
            MetalLinearWeightBuffers::AffineQuantized {
                packed,
                scales,
                biases,
                out_dim,
                in_dim: weight_in_dim,
                packed_cols,
                group_size,
                bits,
                groups,
            } => self.encode_resident_affine_matmul(
                encoder,
                lhs_buffer,
                packed,
                scales,
                biases,
                output_buffer,
                batch,
                in_dim,
                *out_dim,
                *weight_in_dim,
                *packed_cols,
                *group_size,
                *bits,
                *groups,
                prefer_fast_affine,
            ),
        }
    }

    #[inline]
    fn encode_resident_dense_matmul(
        &self,
        encoder: &ComputeCommandEncoderRef,
        lhs_buffer: &BufferRef,
        rhs: &BufferRef,
        output_buffer: &BufferRef,
        batch: usize,
        in_dim: usize,
        out_dim: usize,
        rhs_in_dim: usize,
    ) -> Result<usize> {
        if in_dim != rhs_in_dim {
            return Err(InferError::Dimension(format!(
                "matmul Metal résolu x=[{batch},{in_dim}] rhs=[{out_dim},{rhs_in_dim}]"
            )));
        }
        if can_use_dense_qmv_fast(batch, in_dim, out_dim) {
            self.encode_resident_dense_qmv_fast(
                encoder,
                lhs_buffer,
                rhs,
                output_buffer,
                batch,
                in_dim,
                out_dim,
            )?;
        } else {
            self.encode_resident_dense_fallback(
                encoder,
                lhs_buffer,
                rhs,
                output_buffer,
                batch,
                in_dim,
                out_dim,
            )?;
        }
        Ok(out_dim)
    }

    #[inline]
    fn encode_resident_dense_qmv_fast(
        &self,
        encoder: &ComputeCommandEncoderRef,
        lhs_buffer: &BufferRef,
        rhs: &BufferRef,
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
        encoder.set_buffer(1, Some(rhs), 0);
        encoder.set_buffer(2, Some(output_buffer), 0);
        set_u32_bytes(encoder, 3, &dims, "dense_fast_dims")?;
        profile_dispatch_shape(DispatchProfileShape::matmul(
            "dense_qmv_f32",
            batch,
            in_dim,
            out_dim,
            0,
            32,
        ));
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
    fn encode_resident_dense_fallback(
        &self,
        encoder: &ComputeCommandEncoderRef,
        lhs_buffer: &BufferRef,
        rhs: &BufferRef,
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
        encoder.set_buffer(1, Some(rhs), 0);
        encoder.set_buffer(2, Some(output_buffer), 0);
        set_u32_bytes(encoder, 3, &dims, "dims")?;
        trace_dispatch_path("dense_matmul_rhs_t_f32", batch, out_dim, in_dim);
        self.dispatch_qmv(encoder, &self.dense_matmul_rhs_t_f32, out_dim, batch)
    }

    #[inline]
    fn encode_resident_affine_matmul(
        &self,
        encoder: &ComputeCommandEncoderRef,
        lhs_buffer: &BufferRef,
        packed: &BufferRef,
        scales: &BufferRef,
        biases: &BufferRef,
        output_buffer: &BufferRef,
        batch: usize,
        in_dim: usize,
        out_dim: usize,
        weight_in_dim: usize,
        packed_cols: usize,
        group_size: usize,
        bits: usize,
        groups: usize,
        prefer_fast_affine: bool,
    ) -> Result<usize> {
        if in_dim != weight_in_dim {
            return Err(InferError::Dimension(format!(
                "matmul Metal résolu quantifié x=[{batch},{in_dim}] rhs=[{out_dim},{weight_in_dim}]"
            )));
        }
        let dims = [
            checked_u32(batch, "batch")?,
            checked_u32(out_dim, "out_dim")?,
            checked_u32(in_dim, "in_dim")?,
            checked_u32(packed_cols, "packed_cols")?,
        ];
        let quant = [
            checked_u32(group_size, "group_size")?,
            checked_u32(bits, "bits")?,
            checked_u32(groups, "groups")?,
            0,
        ];
        match self.select_resident_affine_matmul_kernel(
            batch,
            in_dim,
            out_dim,
            group_size,
            bits,
            prefer_fast_affine,
        ) {
            AffineMatmulKernel::Qmm2 => self.encode_resident_affine_qmm2(
                encoder,
                lhs_buffer,
                packed,
                scales,
                biases,
                output_buffer,
                batch,
                in_dim,
                out_dim,
                packed_cols,
                group_size,
                bits,
                groups,
            )?,
            AffineMatmulKernel::QmmNaFusedTiledU4 => {
                self.encode_affine_qmm_na_fused_tiled_u4_buffers(
                    encoder,
                    lhs_buffer,
                    packed,
                    scales,
                    biases,
                    output_buffer,
                    batch,
                    in_dim,
                    out_dim,
                )?;
            }
            AffineMatmulKernel::QmmNaFusedTiledU4Align64 => {
                self.encode_affine_qmm_na_fused_tiled_u4_align64_buffers(
                    encoder,
                    lhs_buffer,
                    packed,
                    scales,
                    biases,
                    output_buffer,
                    batch,
                    in_dim,
                    out_dim,
                )?;
            }
            AffineMatmulKernel::FastQmvU4 => self.encode_resident_affine_qmv_u4(
                encoder,
                lhs_buffer,
                packed,
                scales,
                biases,
                output_buffer,
                batch,
                in_dim,
                out_dim,
                packed_cols,
                group_size,
                bits,
                groups,
            )?,
            AffineMatmulKernel::FastQmvU4Align64 => self.encode_affine_qmv_align64_buffers(
                encoder,
                lhs_buffer,
                packed,
                scales,
                biases,
                output_buffer,
                batch,
                in_dim,
                out_dim,
                packed_cols,
                groups,
                FAST_QMV_BITS,
            )?,
            AffineMatmulKernel::FastQmvU6 => self.encode_affine_qmv_u6_buffers(
                encoder,
                lhs_buffer,
                packed,
                scales,
                biases,
                output_buffer,
                batch,
                in_dim,
                out_dim,
                packed_cols,
                groups,
            )?,
            AffineMatmulKernel::FastQmvOneU8 => self.encode_affine_qmv_one_u8_buffers(
                encoder,
                lhs_buffer,
                packed,
                scales,
                biases,
                output_buffer,
                batch,
                in_dim,
                packed_cols,
                groups,
            )?,
            AffineMatmulKernel::QmmNaFusedTiledU8 => {
                self.encode_affine_qmm_na_fused_tiled_u8_buffers(
                    encoder,
                    lhs_buffer,
                    packed,
                    scales,
                    biases,
                    output_buffer,
                    batch,
                    in_dim,
                    out_dim,
                    group_size,
                )?;
            }
            AffineMatmulKernel::QmmNaU8Gs128 => self.encode_affine_qmm_na_qb_u8_buffers(
                encoder,
                lhs_buffer,
                packed,
                scales,
                biases,
                output_buffer,
                batch,
                in_dim,
                out_dim,
                group_size,
            )?,
            AffineMatmulKernel::FastQmvU8 => self.encode_resident_affine_qmv_u8(
                encoder,
                lhs_buffer,
                packed,
                scales,
                biases,
                output_buffer,
                batch,
                in_dim,
                out_dim,
                packed_cols,
                group_size,
                bits,
                groups,
            )?,
            AffineMatmulKernel::FastQmvU8Align64 => self.encode_affine_qmv_align64_buffers(
                encoder,
                lhs_buffer,
                packed,
                scales,
                biases,
                output_buffer,
                batch,
                in_dim,
                out_dim,
                packed_cols,
                groups,
                8,
            )?,
            AffineMatmulKernel::Fallback | AffineMatmulKernel::QmmNaU8 => self
                .encode_resident_affine_fallback(
                    encoder,
                    lhs_buffer,
                    packed,
                    scales,
                    biases,
                    output_buffer,
                    batch,
                    in_dim,
                    out_dim,
                    &dims,
                    &quant,
                )?,
        }
        Ok(out_dim)
    }

    #[inline]
    fn encode_resident_affine_qmm2(
        &self,
        encoder: &ComputeCommandEncoderRef,
        lhs_buffer: &BufferRef,
        packed: &BufferRef,
        scales: &BufferRef,
        biases: &BufferRef,
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
        encoder.set_buffer(1, Some(packed), 0);
        encoder.set_buffer(2, Some(scales), 0);
        encoder.set_buffer(3, Some(biases), 0);
        encoder.set_buffer(4, Some(output_buffer), 0);
        set_u32_bytes(encoder, 5, &fast_dims, "qmm2_dims")?;
        profile_dispatch_shape(DispatchProfileShape::matmul(
            if bits == FAST_QMV_BITS {
                "affine_qmm2_u4_gs64"
            } else if group_size == FAST_QMV_GROUP_SIZE {
                "affine_qmm2_u8_gs64"
            } else {
                "affine_qmm2_u8_gs128"
            },
            batch,
            in_dim,
            out_dim,
            group_size,
            bits,
        ));
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
    fn encode_resident_affine_qmv_u4(
        &self,
        encoder: &ComputeCommandEncoderRef,
        lhs_buffer: &BufferRef,
        packed: &BufferRef,
        scales: &BufferRef,
        biases: &BufferRef,
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
        encoder.set_buffer(1, Some(packed), 0);
        encoder.set_buffer(2, Some(scales), 0);
        encoder.set_buffer(3, Some(biases), 0);
        encoder.set_buffer(4, Some(output_buffer), 0);
        set_u32_bytes(encoder, 5, &fast_dims, "fast_dims")?;
        profile_dispatch_shape(DispatchProfileShape::matmul(
            if out_dim % 8 == 0 {
                "affine_qmv_u4_aligned_gs64"
            } else {
                "affine_qmv_u4_tail_gs64"
            },
            batch,
            in_dim,
            out_dim,
            group_size,
            bits,
        ));
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
    fn encode_resident_affine_qmv_u8(
        &self,
        encoder: &ComputeCommandEncoderRef,
        lhs_buffer: &BufferRef,
        packed: &BufferRef,
        scales: &BufferRef,
        biases: &BufferRef,
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
        let (pipeline, profile_label, kernel_name) = match (
            group_size == FAST_QMV_GROUP_SIZE,
            use_dot4,
            use_tg256,
            use_tg128,
        ) {
            (true, true, _, _) => (
                &self.affine_qmv_fast_aligned_u8_gs64_dot4_f32,
                "affine_qmv_u8_gs64_dot4",
                "affine_qmv_fast_aligned_u8_gs64_dot4_f32",
            ),
            (false, true, _, _) => (
                &self.affine_qmv_fast_aligned_u8_gs128_dot4_f32,
                "affine_qmv_u8_gs128_dot4",
                "affine_qmv_fast_aligned_u8_gs128_dot4_f32",
            ),
            (true, false, true, _) => (
                &self.affine_qmv_fast_aligned_u8_gs64_tg256_f32,
                "affine_qmv_u8_gs64_tg256",
                "affine_qmv_fast_aligned_u8_gs64_tg256_f32",
            ),
            (false, false, true, _) => (
                &self.affine_qmv_fast_aligned_u8_gs128_tg256_f32,
                "affine_qmv_u8_gs128_tg256",
                "affine_qmv_fast_aligned_u8_gs128_tg256_f32",
            ),
            (true, false, false, true) => (
                &self.affine_qmv_fast_aligned_u8_gs64_tg128_f32,
                "affine_qmv_u8_gs64_tg128",
                "affine_qmv_fast_aligned_u8_gs64_tg128_f32",
            ),
            (false, false, false, true) => (
                &self.affine_qmv_fast_aligned_u8_gs128_tg128_f32,
                "affine_qmv_u8_gs128_tg128",
                "affine_qmv_fast_aligned_u8_gs128_tg128_f32",
            ),
            (true, false, false, false) => (
                &self.affine_qmv_fast_aligned_u8_gs64_f32,
                "affine_qmv_u8_gs64",
                "affine_qmv_fast_aligned_u8_gs64_f32",
            ),
            (false, false, false, false) => (
                &self.affine_qmv_fast_aligned_u8_gs128_f32,
                "affine_qmv_u8_gs128",
                "affine_qmv_fast_aligned_u8_gs128_f32",
            ),
        };
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(lhs_buffer), 0);
        encoder.set_buffer(1, Some(packed), 0);
        encoder.set_buffer(2, Some(scales), 0);
        encoder.set_buffer(3, Some(biases), 0);
        encoder.set_buffer(4, Some(output_buffer), 0);
        set_u32_bytes(encoder, 5, &fast_dims, "fast_u8_dims")?;
        profile_dispatch_shape(DispatchProfileShape::matmul(
            profile_label,
            batch,
            in_dim,
            out_dim,
            group_size,
            bits,
        ));
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
    fn encode_resident_affine_fallback(
        &self,
        encoder: &ComputeCommandEncoderRef,
        lhs_buffer: &BufferRef,
        packed: &BufferRef,
        scales: &BufferRef,
        biases: &BufferRef,
        output_buffer: &BufferRef,
        batch: usize,
        in_dim: usize,
        out_dim: usize,
        dims: &[u32; 4],
        quant: &[u32; 4],
    ) -> Result<()> {
        encoder.set_compute_pipeline_state(&self.affine_matmul_rhs_t_u32_f32);
        encoder.set_buffer(0, Some(lhs_buffer), 0);
        encoder.set_buffer(1, Some(packed), 0);
        encoder.set_buffer(2, Some(scales), 0);
        encoder.set_buffer(3, Some(biases), 0);
        encoder.set_buffer(4, Some(output_buffer), 0);
        set_u32_bytes(encoder, 5, dims, "dims")?;
        set_u32_bytes(encoder, 6, quant, "quant")?;
        trace_dispatch_path("affine_matmul_rhs_t_u32_f32", batch, out_dim, in_dim);
        self.dispatch_qmv(encoder, &self.affine_matmul_rhs_t_u32_f32, out_dim, batch)
    }
}
