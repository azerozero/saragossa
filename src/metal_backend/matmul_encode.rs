//! Encodage bas niveau des matmuls Metal.

use super::*;

const MATMUL_ROW_TG_WIDTH: u64 = 256;

#[expect(
    clippy::too_many_arguments,
    reason = "wrappers d'encodage Metal: buffers, dimensions et offsets restent explicites"
)]
impl MetalExecutor {
    pub(super) fn qmm_na_fused_tiled_available(&self, group_size: usize) -> bool {
        match group_size {
            FAST_QMV_GROUP_SIZE => self.na_gemm_coop_qb_tiled.is_some(),
            QMM_NA_GS128_GROUP_SIZE => self.na_gemm_coop_qb_tiled_gs128.is_some(),
            _ => false,
        }
    }

    pub(super) fn qmm_na_fused_tiled_u4_available(&self, group_size: usize) -> bool {
        group_size == FAST_QMV_GROUP_SIZE && self.na_gemm_coop_qb_tiled_u4.is_some()
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "kernel NA fused-tiled: buffers, dimensions et group_size explicites"
    )]
    pub(super) fn encode_affine_qmm_na_fused_tiled_u8_buffers(
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
        group_size: usize,
    ) -> Result<()> {
        if out_dim % 64 != 0 {
            return Err(InferError::Dimension(format!(
                "qmm na fused-tiled attend out_dim%64=0, reçu batch={batch} out_dim={out_dim}"
            )));
        }
        let (pso, label, profile_label) = match group_size {
            FAST_QMV_GROUP_SIZE => (
                self.na_gemm_coop_qb_tiled.as_ref(),
                "gemm_nax_coop_qb_tiled",
                "qmm_na_fused_tiled_u8_gs64",
            ),
            QMM_NA_GS128_GROUP_SIZE => (
                self.na_gemm_coop_qb_tiled_gs128.as_ref(),
                "gemm_nax_coop_qb_tiled_gs128",
                "qmm_na_fused_tiled_u8_gs128",
            ),
            other => {
                return Err(InferError::Dimension(format!(
                    "qmm na fused-tiled group_size non supporté {other}"
                )));
            }
        };
        let pso = pso.ok_or_else(|| InferError::Config(format!("{label}: NA indisponible")))?;
        let lhs_len = checked_len(batch, in_dim, "qmm na fused-tiled lhs")?;
        let lhs_bf16 = self.private_bf16_buffer(lhs_len, "qmm_na_fused_tiled_lhs_bf16")?;
        self.encode_f32_to_bf16(encoder, lhs_buffer, &lhs_bf16, lhs_len)?;
        let mnk = [
            checked_u32(batch, "qmm na fused-tiled batch")?,
            checked_u32(out_dim, "qmm na fused-tiled out_dim")?,
            checked_u32(in_dim, "qmm na fused-tiled in_dim")?,
        ];
        encoder.set_compute_pipeline_state(pso);
        encoder.set_buffer(0, Some(&lhs_bf16), 0);
        encoder.set_buffer(1, Some(packed_buffer), 0);
        encoder.set_buffer(2, Some(scales_buffer), 0);
        encoder.set_buffer(3, Some(biases_buffer), 0);
        encoder.set_buffer(4, Some(output_buffer), 0);
        encoder.set_bytes(5, 12, mnk.as_ptr().cast::<std::ffi::c_void>());
        profile_dispatch_shape(DispatchProfileShape::matmul(
            profile_label,
            batch,
            in_dim,
            out_dim,
            group_size,
            8,
        ));
        trace_dispatch_path(label, batch, out_dim, in_dim);
        profile_dispatch();
        let width = pso.thread_execution_width().max(1);
        encoder.dispatch_thread_groups(
            MTLSize::new(
                checked_nsuint(batch.div_ceil(64), "qmm na fused-tiled batch tiles")?,
                checked_nsuint(out_dim / 64, "qmm na fused-tiled out tiles")?,
                1,
            ),
            MTLSize::new(width * 4, 1, 1),
        );
        post_dispatch_barrier_buffer(encoder, output_buffer);
        Ok(())
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "kernel NA fused-tiled u4: buffers et dimensions explicites"
    )]
    pub(super) fn encode_affine_qmm_na_fused_tiled_u4_buffers(
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
    ) -> Result<()> {
        if out_dim % 64 != 0 {
            return Err(InferError::Dimension(format!(
                "qmm na fused-tiled u4 attend out_dim%64=0, reçu batch={batch} out_dim={out_dim}"
            )));
        }
        let label = "gemm_nax_coop_qb_tiled_u4";
        let pso = self
            .na_gemm_coop_qb_tiled_u4
            .as_ref()
            .ok_or_else(|| InferError::Config(format!("{label}: NA indisponible")))?;
        let lhs_len = checked_len(batch, in_dim, "qmm na fused-tiled u4 lhs")?;
        let lhs_bf16 = self.private_bf16_buffer(lhs_len, "qmm_na_fused_tiled_u4_lhs_bf16")?;
        self.encode_f32_to_bf16(encoder, lhs_buffer, &lhs_bf16, lhs_len)?;
        let mnk = [
            checked_u32(batch, "qmm na fused-tiled u4 batch")?,
            checked_u32(out_dim, "qmm na fused-tiled u4 out_dim")?,
            checked_u32(in_dim, "qmm na fused-tiled u4 in_dim")?,
        ];
        encoder.set_compute_pipeline_state(pso);
        encoder.set_buffer(0, Some(&lhs_bf16), 0);
        encoder.set_buffer(1, Some(packed_buffer), 0);
        encoder.set_buffer(2, Some(scales_buffer), 0);
        encoder.set_buffer(3, Some(biases_buffer), 0);
        encoder.set_buffer(4, Some(output_buffer), 0);
        encoder.set_bytes(5, 12, mnk.as_ptr().cast::<std::ffi::c_void>());
        profile_dispatch_shape(DispatchProfileShape::matmul(
            "qmm_na_fused_tiled_u4_gs64",
            batch,
            in_dim,
            out_dim,
            FAST_QMV_GROUP_SIZE,
            4,
        ));
        trace_dispatch_path(label, batch, out_dim, in_dim);
        profile_dispatch();
        let width = pso.thread_execution_width().max(1);
        encoder.dispatch_thread_groups(
            MTLSize::new(
                checked_nsuint(batch.div_ceil(64), "qmm na fused-tiled u4 batch tiles")?,
                checked_nsuint(out_dim / 64, "qmm na fused-tiled u4 out tiles")?,
                1,
            ),
            MTLSize::new(width * 4, 1, 1),
        );
        post_dispatch_barrier_buffer(encoder, output_buffer);
        Ok(())
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "kernel NA quantifié: buffers, dimensions et group_size explicites"
    )]
    pub(super) fn encode_affine_qmm_na_qb_u8_buffers(
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
        group_size: usize,
    ) -> Result<()> {
        if out_dim % 32 != 0 {
            return Err(InferError::Dimension(format!(
                "qmm na qb attend out_dim%32=0, reçu batch={batch} out_dim={out_dim}"
            )));
        }
        let (pso, label) = match group_size {
            FAST_QMV_GROUP_SIZE => (self.na_gemm_coop_qb.as_ref(), "gemm_nax_coop_qb"),
            QMM_NA_GS128_GROUP_SIZE => (
                self.na_gemm_coop_qb_gs128.as_ref(),
                "gemm_nax_coop_qb_gs128",
            ),
            other => {
                return Err(InferError::Dimension(format!(
                    "qmm na qb group_size non supporté {other}"
                )));
            }
        };
        let pso = pso.ok_or_else(|| InferError::Config(format!("{label}: NA indisponible")))?;
        let lhs_len = checked_len(batch, in_dim, "qmm na qb lhs")?;
        let lhs_bf16 = self.private_bf16_buffer(lhs_len, "qmm_na_qb_lhs_bf16")?;
        self.encode_f32_to_bf16(encoder, lhs_buffer, &lhs_bf16, lhs_len)?;
        let mnk = [
            checked_u32(batch, "qmm na qb batch")?,
            checked_u32(out_dim, "qmm na qb out_dim")?,
            checked_u32(in_dim, "qmm na qb in_dim")?,
        ];
        encoder.set_compute_pipeline_state(pso);
        encoder.set_buffer(0, Some(&lhs_bf16), 0);
        encoder.set_buffer(1, Some(packed_buffer), 0);
        encoder.set_buffer(2, Some(scales_buffer), 0);
        encoder.set_buffer(3, Some(biases_buffer), 0);
        encoder.set_buffer(4, Some(output_buffer), 0);
        encoder.set_bytes(5, 12, mnk.as_ptr().cast::<std::ffi::c_void>());
        profile_dispatch_shape(DispatchProfileShape::matmul(
            if group_size == QMM_NA_GS128_GROUP_SIZE {
                "qmm_na_qb_u8_gs128"
            } else {
                "qmm_na_qb_u8_gs64"
            },
            batch,
            in_dim,
            out_dim,
            group_size,
            8,
        ));
        trace_dispatch_path(label, batch, out_dim, in_dim);
        profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(
                checked_nsuint(batch.div_ceil(16), "qmm na qb batch tiles")?,
                checked_nsuint(out_dim / 32, "qmm na qb out tiles")?,
                1,
            ),
            MTLSize::new(32, 1, 1),
        );
        post_dispatch_barrier_buffer(encoder, output_buffer);
        Ok(())
    }

    pub(super) fn encode_affine_qmv_one_u8_buffers(
        &self,
        encoder: &ComputeCommandEncoderRef,
        lhs_buffer: &BufferRef,
        packed_buffer: &BufferRef,
        scales_buffer: &BufferRef,
        biases_buffer: &BufferRef,
        output_buffer: &BufferRef,
        batch: usize,
        in_dim: usize,
        packed_cols: usize,
        groups: usize,
    ) -> Result<()> {
        let dims = [
            1,
            checked_u32(in_dim, "qmv one u8 in_dim")?,
            checked_u32(packed_cols, "qmv one u8 packed_cols")?,
            checked_u32(groups, "qmv one u8 groups")?,
        ];
        encoder.set_compute_pipeline_state(&self.affine_qmv_one_fast_u8_gs64_f32);
        encoder.set_buffer(0, Some(lhs_buffer), 0);
        encoder.set_buffer(1, Some(packed_buffer), 0);
        encoder.set_buffer(2, Some(scales_buffer), 0);
        encoder.set_buffer(3, Some(biases_buffer), 0);
        encoder.set_buffer(4, Some(output_buffer), 0);
        set_u32_bytes(encoder, 5, &dims, "qmv_one_u8_dims")?;
        profile_dispatch_shape(DispatchProfileShape::matmul(
            "affine_qmv_one_u8_gs64",
            batch,
            in_dim,
            1,
            FAST_QMV_GROUP_SIZE,
            8,
        ));
        trace_dispatch_path("affine_qmv_one_fast_u8_gs64_f32", batch, 1, in_dim);
        profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(checked_nsuint(batch, "qmv one u8 batch")?, 1, 1),
            MTLSize::new(32, 1, 1),
        );
        post_dispatch_barrier_buffer(encoder, output_buffer);
        Ok(())
    }

    pub(super) fn encode_affine_qmv_u6_buffers(
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
        let dims = [
            checked_u32(out_dim, "qmv u6 out_dim")?,
            checked_u32(in_dim, "qmv u6 in_dim")?,
            checked_u32(packed_cols, "qmv u6 packed_cols")?,
            checked_u32(groups, "qmv u6 groups")?,
        ];
        let (pipeline, profile_label, kernel_name) = if out_dim % 2 == 0 {
            (
                &self.affine_qmv_fast_aligned_u6_gs64_f32,
                "affine_qmv_u6_aligned_gs64",
                "affine_qmv_fast_aligned_u6_gs64_f32",
            )
        } else {
            (
                &self.affine_qmv_fast_u6_gs64_f32,
                "affine_qmv_u6_tail_gs64",
                "affine_qmv_fast_u6_gs64_f32",
            )
        };
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(lhs_buffer), 0);
        encoder.set_buffer(1, Some(packed_buffer), 0);
        encoder.set_buffer(2, Some(scales_buffer), 0);
        encoder.set_buffer(3, Some(biases_buffer), 0);
        encoder.set_buffer(4, Some(output_buffer), 0);
        set_u32_bytes(encoder, 5, &dims, "qmv_u6_dims")?;
        profile_dispatch_shape(DispatchProfileShape::matmul(
            profile_label,
            batch,
            in_dim,
            out_dim,
            FAST_QMV_GROUP_SIZE,
            FAST_QMV_U6_BITS,
        ));
        trace_dispatch_path(kernel_name, batch, out_dim, in_dim);
        profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(
                checked_nsuint(batch, "qmv u6 batch")?,
                checked_nsuint(out_dim.div_ceil(2), "qmv u6 out groups")?,
                1,
            ),
            MTLSize::new(64, 1, 1),
        );
        post_dispatch_barrier_buffer(encoder, output_buffer);
        Ok(())
    }

    /// Encode le gather d'une ligne d'embedding depuis un token `u32` produit GPU.
    pub(crate) fn encode_embedding_from_index_buffers(
        &self,
        encoder: &ComputeCommandEncoderRef,
        embedding: &MetalEmbeddingWeightBuffers,
        index_buffer: &BufferRef,
        output_buffer: &BufferRef,
        expected_dim: usize,
    ) -> Result<()> {
        self.encode_embedding_from_index_buffers_with_offset(
            encoder,
            embedding,
            index_buffer,
            0,
            output_buffer,
            expected_dim,
        )
    }

    pub(crate) fn encode_embedding_from_index_buffers_with_offset(
        &self,
        encoder: &ComputeCommandEncoderRef,
        embedding: &MetalEmbeddingWeightBuffers,
        index_buffer: &BufferRef,
        index_offset: u64,
        output_buffer: &BufferRef,
        expected_dim: usize,
    ) -> Result<()> {
        match embedding {
            MetalEmbeddingWeightBuffers::Dense { table, vocab, dim } => {
                if *dim != expected_dim {
                    return Err(InferError::Dimension(format!(
                        "embedding dense dim={dim}, attendu {expected_dim}"
                    )));
                }
                let dims = [
                    checked_u32(*vocab, "embedding vocab")?,
                    checked_u32(*dim, "embedding dim")?,
                ];
                encoder.set_compute_pipeline_state(&self.embed_gather_dense_from_u32_f32);
                encoder.set_buffer(0, Some(table), 0);
                encoder.set_buffer(1, Some(index_buffer), index_offset);
                encoder.set_buffer(2, Some(output_buffer), 0);
                set_u32_bytes(encoder, 3, &dims, "embedding_dense_dims")?;
            }
            MetalEmbeddingWeightBuffers::AffineQuantized {
                packed,
                scales,
                biases,
                vocab,
                dim,
                packed_cols,
                group_size,
                bits,
                groups,
            } => {
                if *dim != expected_dim {
                    return Err(InferError::Dimension(format!(
                        "embedding quantifié dim={dim}, attendu {expected_dim}"
                    )));
                }
                let dims = [
                    checked_u32(*vocab, "embedding vocab")?,
                    checked_u32(*dim, "embedding dim")?,
                    checked_u32(*packed_cols, "embedding packed_cols")?,
                    checked_u32(*groups, "embedding groups")?,
                ];
                let quant = [
                    checked_u32(*group_size, "embedding group_size")?,
                    checked_u32(*bits, "embedding bits")?,
                    0,
                    0,
                ];
                encoder.set_compute_pipeline_state(&self.embed_gather_affine_from_u32_f32);
                encoder.set_buffer(0, Some(packed), 0);
                encoder.set_buffer(1, Some(scales), 0);
                encoder.set_buffer(2, Some(biases), 0);
                encoder.set_buffer(3, Some(index_buffer), index_offset);
                encoder.set_buffer(4, Some(output_buffer), 0);
                set_u32_bytes(encoder, 5, &dims, "embedding_affine_dims")?;
                set_u32_bytes(encoder, 6, &quant, "embedding_affine_quant")?;
            }
        }
        profile_dispatch();
        encoder.dispatch_threads(
            MTLSize::new(checked_nsuint(expected_dim, "embedding dim")?, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        post_dispatch_barrier_buffer(encoder, output_buffer);
        Ok(())
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "kernel spécialisé: rms_norm prologue + QMV fast"
    )]
    pub(crate) fn encode_matmul_weight_buffers_rms_prologue(
        &self,
        encoder: &ComputeCommandEncoderRef,
        lhs_buffer: &BufferRef,
        rms_weight_buffer: &BufferRef,
        rms_eps: f32,
        in_dim: usize,
        weight: &MetalLinearWeightBuffers,
        output_buffer: &BufferRef,
    ) -> Result<Option<usize>> {
        if !fused_rms_prologue_enabled() {
            return Ok(None);
        }
        let MetalLinearWeightBuffers::AffineQuantized {
            packed,
            scales,
            biases,
            out_dim,
            in_dim: weight_in_dim,
            packed_cols,
            group_size,
            bits,
            groups,
        } = weight
        else {
            return Ok(None);
        };
        if in_dim != *weight_in_dim {
            return Err(InferError::Dimension(format!(
                "matmul rms prologue x=[1,{in_dim}] rhs=[{out_dim},{weight_in_dim}]"
            )));
        }
        let can_use_u4 = fast_affine_qmv_enabled(*out_dim)
            && *bits == FAST_QMV_BITS
            && *group_size == FAST_QMV_GROUP_SIZE
            && in_dim % 512 == 0;
        let can_use_u8 =
            can_use_fast_affine_qmv_u8_buffers(1, in_dim, *out_dim, *group_size, *bits);
        if !can_use_u4 && !can_use_u8 {
            return Ok(None);
        }
        let fast_dims = [
            checked_u32(*out_dim, "rms qmv out_dim")?,
            checked_u32(in_dim, "rms qmv in_dim")?,
            checked_u32(*packed_cols, "rms qmv packed_cols")?,
            checked_u32(*groups, "rms qmv groups")?,
        ];
        let (pipeline, kernel_name) = if can_use_u4 {
            (
                &self.affine_qmv_rms_fast_u4_gs64_f32,
                "affine_qmv_rms_fast_u4_gs64_f32",
            )
        } else if *group_size == FAST_QMV_GROUP_SIZE {
            (
                &self.affine_qmv_rms_fast_u8_gs64_f32,
                "affine_qmv_rms_fast_u8_gs64_f32",
            )
        } else {
            (
                &self.affine_qmv_rms_fast_u8_gs128_f32,
                "affine_qmv_rms_fast_u8_gs128_f32",
            )
        };
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(lhs_buffer), 0);
        encoder.set_buffer(1, Some(rms_weight_buffer), 0);
        encoder.set_buffer(2, Some(packed), 0);
        encoder.set_buffer(3, Some(scales), 0);
        encoder.set_buffer(4, Some(biases), 0);
        encoder.set_buffer(5, Some(output_buffer), 0);
        set_u32_bytes(encoder, 6, &fast_dims, "rms_qmv_dims")?;
        set_f32_bytes(encoder, 7, &[rms_eps], "rms_qmv_eps")?;
        trace_dispatch_path(kernel_name, 1, *out_dim, in_dim);
        profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(
                1,
                checked_nsuint(out_dim.div_ceil(8), "rms qmv out groups")?,
                1,
            ),
            MTLSize::new(64, 1, 1),
        );
        post_dispatch_barrier_buffer(encoder, output_buffer);
        Ok(Some(*out_dim))
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "helper interne partagé par deux variantes d'encodage Metal"
    )]
    /// Dé-quantifie un poids u8 gs64 `[out_dim,in_dim]` en bf16 dense TRANSPOSÉ
    /// `[in_dim,out_dim]` (= `W^T`, layout `[K,N]` du GEMM NA), dans l'encodeur
    /// partagé. Pour le chemin prefill matmul2d.
    pub(super) fn encode_dequant_qweight_to_bf16_t(
        &self,
        encoder: &ComputeCommandEncoderRef,
        packed: &BufferRef,
        scales: &BufferRef,
        biases: &BufferRef,
        wt: &BufferRef,
        out_dim: usize,
        in_dim: usize,
        packed_cols: usize,
    ) -> Result<()> {
        let total = checked_len(out_dim, in_dim, "dequant na total")?;
        let dims = [
            checked_u32(out_dim, "dequant na out_dim")?,
            checked_u32(in_dim, "dequant na in_dim")?,
            checked_u32(packed_cols, "dequant na packed_cols")?,
            0,
        ];
        encoder.set_compute_pipeline_state(&self.dequant_u8_to_bf16_t_gs64);
        encoder.set_buffer(0, Some(packed), 0);
        encoder.set_buffer(1, Some(scales), 0);
        encoder.set_buffer(2, Some(biases), 0);
        encoder.set_buffer(3, Some(wt), 0);
        set_u32_bytes(encoder, 4, &dims, "dequant_na_dims")?;
        trace_dispatch_path("dequant_u8_to_bf16_t_gs64", out_dim, in_dim, packed_cols);
        self.dispatch_1d(encoder, &self.dequant_u8_to_bf16_t_gs64, total)
    }

    pub(crate) fn encode_copy(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input_buffer: &BufferRef,
        output_buffer: &BufferRef,
        len: usize,
    ) -> Result<()> {
        self.encode_copy_with_offsets(encoder, input_buffer, 0, output_buffer, 0, len)
    }

    pub(crate) fn encode_copy_with_offsets(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input_buffer: &BufferRef,
        input_offset: u64,
        output_buffer: &BufferRef,
        output_offset: u64,
        len: usize,
    ) -> Result<()> {
        let len_u32 = checked_u32(len, "copy len")?;
        encoder.set_compute_pipeline_state(&self.copy_f32);
        encoder.set_buffer(0, Some(input_buffer), input_offset);
        encoder.set_buffer(1, Some(output_buffer), output_offset);
        set_u32_bytes(encoder, 2, &[len_u32], "copy_len")?;
        self.dispatch_1d(encoder, &self.copy_f32, len)
    }

    pub(crate) fn encode_copy_u16(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input_buffer: &BufferRef,
        output_buffer: &BufferRef,
        len: usize,
    ) -> Result<()> {
        self.encode_copy_u16_with_offsets(encoder, input_buffer, 0, output_buffer, 0, len)
    }

    pub(crate) fn encode_copy_u16_with_offsets(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input_buffer: &BufferRef,
        input_offset: u64,
        output_buffer: &BufferRef,
        output_offset: u64,
        len: usize,
    ) -> Result<()> {
        let len_u32 = checked_u32(len, "copy u16 len")?;
        encoder.set_compute_pipeline_state(&self.copy_u16);
        encoder.set_buffer(0, Some(input_buffer), input_offset);
        encoder.set_buffer(1, Some(output_buffer), output_offset);
        set_u32_bytes(encoder, 2, &[len_u32], "copy_u16_len")?;
        self.dispatch_1d(encoder, &self.copy_u16, len)
    }

    /// Encode `rms_norm` par ligne (`rows × dim`) vers `output_buffer` résident.
    /// Exposé `pub(crate)` pour l'input/final norm du decode résident (1c).
    pub(crate) fn encode_rms_norm_rows(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input_buffer: &BufferRef,
        weight_buffer: &BufferRef,
        output_buffer: &BufferRef,
        rows: usize,
        dim: usize,
        eps: f32,
    ) -> Result<()> {
        encoder.set_compute_pipeline_state(&self.rms_norm_rows_f32);
        encoder.set_buffer(0, Some(input_buffer), 0);
        encoder.set_buffer(1, Some(weight_buffer), 0);
        encoder.set_buffer(2, Some(output_buffer), 0);
        set_u32_bytes(
            encoder,
            3,
            &[checked_u32(dim, "rms rows dim")?],
            "rms_rows_dim",
        )?;
        set_f32_bytes(encoder, 4, &[eps], "rms_rows_eps")?;
        trace_dispatch_path("rms_norm_rows_f32", rows, dim, 0);
        profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(checked_nsuint(rows, "rms rows")?, 1, 1),
            MTLSize::new(MATMUL_ROW_TG_WIDTH, 1, 1),
        );
        post_dispatch_barrier_buffer(encoder, output_buffer);
        Ok(())
    }

    /// Encode `rms_norm` par ligne en reproduisant BIT-À-BIT le prologue rms des
    /// kernels fusionnés (`affine_qmv_rms_fast`, `affine_qkv_split_rms_qmv_fast`) :
    /// 1 simdgroup par row, accumulation séquentielle 16 valeurs/thread.
    /// Chemin duo light-batch (E2.2) : rms_simd → qmm2 == fused(rms+qmv) en bits.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si `dim % 512 != 0` (précondition des kernels fusionnés).
    pub(crate) fn encode_rms_norm_simd_rows(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input_buffer: &BufferRef,
        weight_buffer: &BufferRef,
        output_buffer: &BufferRef,
        rows: usize,
        dim: usize,
        eps: f32,
    ) -> Result<()> {
        if dim % 512 != 0 {
            return Err(InferError::Dimension(format!(
                "rms_norm_simd exige dim % 512 == 0, reçu {dim}"
            )));
        }
        encoder.set_compute_pipeline_state(&self.rms_norm_simd_rows_f32);
        encoder.set_buffer(0, Some(input_buffer), 0);
        encoder.set_buffer(1, Some(weight_buffer), 0);
        encoder.set_buffer(2, Some(output_buffer), 0);
        set_u32_bytes(
            encoder,
            3,
            &[checked_u32(dim, "rms simd rows dim")?],
            "rms_simd_rows_dim",
        )?;
        set_f32_bytes(encoder, 4, &[eps], "rms_simd_rows_eps")?;
        trace_dispatch_path("rms_norm_simd_rows_f32", rows, dim, 0);
        profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(checked_nsuint(rows, "rms simd rows")?, 1, 1),
            MTLSize::new(32, 1, 1),
        );
        post_dispatch_barrier_buffer(encoder, output_buffer);
        Ok(())
    }

    /// Encode `summed = left + right` puis `normed = rms_norm(summed)` par ligne
    /// (résiduel + post-norm fusionnés), vers buffers résidents. Exposé
    /// `pub(crate)` pour le tail du decode résident (1c).
    pub(crate) fn encode_add_rms_norm_rows(
        &self,
        encoder: &ComputeCommandEncoderRef,
        left_buffer: &BufferRef,
        right_buffer: &BufferRef,
        weight_buffer: &BufferRef,
        summed_buffer: &BufferRef,
        normed_buffer: &BufferRef,
        rows: usize,
        dim: usize,
        eps: f32,
    ) -> Result<()> {
        encoder.set_compute_pipeline_state(&self.add_rms_norm_rows_f32);
        encoder.set_buffer(0, Some(left_buffer), 0);
        encoder.set_buffer(1, Some(right_buffer), 0);
        encoder.set_buffer(2, Some(weight_buffer), 0);
        encoder.set_buffer(3, Some(summed_buffer), 0);
        encoder.set_buffer(4, Some(normed_buffer), 0);
        set_u32_bytes(
            encoder,
            5,
            &[checked_u32(dim, "add rms rows dim")?],
            "add_rms_rows_dim",
        )?;
        set_f32_bytes(encoder, 6, &[eps], "add_rms_rows_eps")?;
        trace_dispatch_path("add_rms_norm_rows_f32", rows, dim, 0);
        profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(checked_nsuint(rows, "add rms rows")?, 1, 1),
            MTLSize::new(MATMUL_ROW_TG_WIDTH, 1, 1),
        );
        post_dispatch_barrier(encoder);
        Ok(())
    }

    pub(crate) fn encode_swiglu(
        &self,
        encoder: &ComputeCommandEncoderRef,
        _owned_buffers: &mut Vec<metal::Buffer>,
        gate_buffer: &BufferRef,
        up_buffer: &BufferRef,
        output_buffer: &BufferRef,
        len: usize,
    ) -> Result<()> {
        let len_u32 = checked_u32(len, "swiglu len")?;
        encoder.set_compute_pipeline_state(&self.swiglu_f32);
        encoder.set_buffer(0, Some(gate_buffer), 0);
        encoder.set_buffer(1, Some(up_buffer), 0);
        encoder.set_buffer(2, Some(output_buffer), 0);
        set_u32_bytes(encoder, 3, &[len_u32], "swiglu_len")?;
        trace_dispatch_path("swiglu_f32", len, 1, 0);
        self.dispatch_1d(encoder, &self.swiglu_f32, len)
    }

    /// Encode `out = gelu_tanh(gate) * up` élément par élément.
    pub(crate) fn encode_geglu_tanh(
        &self,
        encoder: &ComputeCommandEncoderRef,
        _owned_buffers: &mut Vec<metal::Buffer>,
        gate_buffer: &BufferRef,
        up_buffer: &BufferRef,
        output_buffer: &BufferRef,
        len: usize,
    ) -> Result<()> {
        let len_u32 = checked_u32(len, "geglu tanh len")?;
        encoder.set_compute_pipeline_state(&self.geglu_tanh_f32);
        encoder.set_buffer(0, Some(gate_buffer), 0);
        encoder.set_buffer(1, Some(up_buffer), 0);
        encoder.set_buffer(2, Some(output_buffer), 0);
        set_u32_bytes(encoder, 3, &[len_u32], "geglu_tanh_len")?;
        trace_dispatch_path("geglu_tanh_f32", len, 1, 0);
        self.dispatch_1d(encoder, &self.geglu_tanh_f32, len)
    }

    pub(crate) fn encode_accumulate_scaled(
        &self,
        encoder: &ComputeCommandEncoderRef,
        _owned_buffers: &mut Vec<metal::Buffer>,
        src_buffer: &BufferRef,
        dst_buffer: &BufferRef,
        scale: f32,
        len: usize,
    ) -> Result<()> {
        let len_u32 = checked_u32(len, "acc len")?;
        encoder.set_compute_pipeline_state(&self.accumulate_scaled_f32);
        encoder.set_buffer(0, Some(src_buffer), 0);
        encoder.set_buffer(1, Some(dst_buffer), 0);
        set_f32_bytes(encoder, 2, &[scale], "acc_scale")?;
        set_u32_bytes(encoder, 3, &[len_u32], "acc_len")?;
        trace_dispatch_path("accumulate_scaled_f32", 1, len, 0);
        self.dispatch_1d(encoder, &self.accumulate_scaled_f32, len)
    }

    pub(crate) fn encode_add_scaled(
        &self,
        encoder: &ComputeCommandEncoderRef,
        _owned_buffers: &mut Vec<metal::Buffer>,
        left_buffer: &BufferRef,
        right_buffer: &BufferRef,
        output_buffer: &BufferRef,
        scale: f32,
        len: usize,
    ) -> Result<()> {
        let len_u32 = checked_u32(len, "add scaled len")?;
        encoder.set_compute_pipeline_state(&self.add_scaled_f32);
        encoder.set_buffer(0, Some(left_buffer), 0);
        encoder.set_buffer(1, Some(right_buffer), 0);
        encoder.set_buffer(2, Some(output_buffer), 0);
        set_f32_bytes(encoder, 3, &[scale], "add_scaled_scale")?;
        set_u32_bytes(encoder, 4, &[len_u32], "add_scaled_len")?;
        trace_dispatch_path("add_scaled_f32", 1, len, 0);
        self.dispatch_1d(encoder, &self.add_scaled_f32, len)
    }

    #[track_caller]
    pub(super) fn dispatch_qmv(
        &self,
        encoder: &ComputeCommandEncoderRef,
        pipeline: &ComputePipelineState,
        out_dim: usize,
        batch: usize,
    ) -> Result<()> {
        let threads_per_group = self.qmv_thread_group_size(pipeline);
        profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(
                checked_nsuint(out_dim, "out_dim")?,
                checked_nsuint(batch, "batch")?,
                1,
            ),
            MTLSize::new(threads_per_group, 1, 1),
        );
        post_dispatch_barrier(encoder);
        Ok(())
    }

    #[track_caller]
    pub(super) fn dispatch_1d(
        &self,
        encoder: &ComputeCommandEncoderRef,
        pipeline: &ComputePipelineState,
        len: usize,
    ) -> Result<()> {
        let width = pipeline.thread_execution_width().max(1);
        let threads = checked_nsuint(len, "threads")?;
        profile_dispatch();
        encoder.dispatch_threads(MTLSize::new(threads, 1, 1), MTLSize::new(width, 1, 1));
        post_dispatch_barrier(encoder);
        Ok(())
    }
}
