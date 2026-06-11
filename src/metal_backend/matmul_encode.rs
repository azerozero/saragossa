//! Encodage bas niveau des matmuls Metal.

use super::*;

const MATMUL_ROW_TG_WIDTH: u64 = 256;

#[expect(
    clippy::too_many_arguments,
    reason = "wrappers d'encodage Metal: buffers, dimensions et offsets restent explicites"
)]
impl MetalExecutor {
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
        post_dispatch_barrier(encoder);
        Ok(())
    }

    /// Encode un matmul résident avec des buffers de poids pré-résolus.
    #[expect(
        clippy::too_many_arguments,
        reason = "signature d'encodage Metal: encoder + buffers + dimensions"
    )]
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
            } => {
                if in_dim != *rhs_in_dim {
                    return Err(InferError::Dimension(format!(
                        "matmul Metal résolu x=[{batch},{in_dim}] rhs=[{out_dim},{rhs_in_dim}]"
                    )));
                }
                let dims = [
                    checked_u32(batch, "batch")?,
                    checked_u32(*out_dim, "out_dim")?,
                    checked_u32(in_dim, "in_dim")?,
                ];
                encoder.set_compute_pipeline_state(&self.dense_matmul_rhs_t_f32);
                encoder.set_buffer(0, Some(lhs_buffer), 0);
                encoder.set_buffer(1, Some(rhs), 0);
                encoder.set_buffer(2, Some(output_buffer), 0);
                set_u32_bytes(encoder, 3, &dims, "dims")?;
                self.dispatch_qmv(encoder, &self.dense_matmul_rhs_t_f32, *out_dim, batch)?;
                Ok(*out_dim)
            }
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
            } => {
                if in_dim != *weight_in_dim {
                    return Err(InferError::Dimension(format!(
                        "matmul Metal résolu quantifié x=[{batch},{in_dim}] rhs=[{out_dim},{weight_in_dim}]"
                    )));
                }
                let dims = [
                    checked_u32(batch, "batch")?,
                    checked_u32(*out_dim, "out_dim")?,
                    checked_u32(in_dim, "in_dim")?,
                    checked_u32(*packed_cols, "packed_cols")?,
                ];
                let quant = [
                    checked_u32(*group_size, "group_size")?,
                    checked_u32(*bits, "bits")?,
                    checked_u32(*groups, "groups")?,
                    0,
                ];
                if can_use_fast_affine_qmm2_buffers(batch, in_dim, *out_dim, *group_size, *bits) {
                    let fast_dims = [
                        checked_u32(*out_dim, "qmm2 out_dim")?,
                        checked_u32(in_dim, "qmm2 in_dim")?,
                        checked_u32(*packed_cols, "qmm2 packed_cols")?,
                        checked_u32(*groups, "qmm2 groups")?,
                    ];
                    encoder.set_compute_pipeline_state(&self.affine_qmm2_fast_aligned_u4_gs64_f32);
                    encoder.set_buffer(0, Some(lhs_buffer), 0);
                    encoder.set_buffer(1, Some(packed), 0);
                    encoder.set_buffer(2, Some(scales), 0);
                    encoder.set_buffer(3, Some(biases), 0);
                    encoder.set_buffer(4, Some(output_buffer), 0);
                    set_u32_bytes(encoder, 5, &fast_dims, "qmm2_dims")?;
                    profile_dispatch();
                    encoder.dispatch_thread_groups(
                        MTLSize::new(
                            1,
                            checked_nsuint(out_dim.div_ceil(8), "qmm2 out groups")?,
                            1,
                        ),
                        MTLSize::new(64, 1, 1),
                    );
                    post_dispatch_barrier(encoder);
                } else if (fast_affine_qmv_enabled(*out_dim) || prefer_fast_affine)
                    && batch > 0
                    && *bits == FAST_QMV_BITS
                    && *group_size == FAST_QMV_GROUP_SIZE
                    && in_dim % 512 == 0
                {
                    let fast_dims = [
                        checked_u32(*out_dim, "fast out_dim")?,
                        checked_u32(in_dim, "fast in_dim")?,
                        checked_u32(*packed_cols, "fast packed_cols")?,
                        checked_u32(*groups, "fast groups")?,
                    ];
                    let pipeline = if *out_dim % 8 == 0 {
                        &self.affine_qmv_fast_aligned_u4_gs64_f32
                    } else {
                        &self.affine_qmv_fast_u4_gs64_f32
                    };
                    encoder.set_compute_pipeline_state(pipeline);
                    encoder.set_buffer(0, Some(lhs_buffer), 0);
                    encoder.set_buffer(1, Some(packed), 0);
                    encoder.set_buffer(2, Some(scales), 0);
                    encoder.set_buffer(3, Some(biases), 0);
                    encoder.set_buffer(4, Some(output_buffer), 0);
                    set_u32_bytes(encoder, 5, &fast_dims, "fast_dims")?;
                    profile_dispatch();
                    encoder.dispatch_thread_groups(
                        MTLSize::new(
                            checked_nsuint(batch, "batch")?,
                            checked_nsuint(out_dim.div_ceil(8), "fast out groups")?,
                            1,
                        ),
                        MTLSize::new(64, 1, 1),
                    );
                    post_dispatch_barrier(encoder);
                } else if fast_affine_qmv_enabled(*out_dim)
                    && batch > 0
                    && *bits == 8
                    && *group_size == FAST_QMV_GROUP_SIZE
                    && in_dim % 512 == 0
                    && *out_dim % 8 == 0
                {
                    let fast_dims = [
                        checked_u32(*out_dim, "fast u8 out_dim")?,
                        checked_u32(in_dim, "fast u8 in_dim")?,
                        checked_u32(*packed_cols, "fast u8 packed_cols")?,
                        checked_u32(*groups, "fast u8 groups")?,
                    ];
                    encoder.set_compute_pipeline_state(&self.affine_qmv_fast_aligned_u8_gs64_f32);
                    encoder.set_buffer(0, Some(lhs_buffer), 0);
                    encoder.set_buffer(1, Some(packed), 0);
                    encoder.set_buffer(2, Some(scales), 0);
                    encoder.set_buffer(3, Some(biases), 0);
                    encoder.set_buffer(4, Some(output_buffer), 0);
                    set_u32_bytes(encoder, 5, &fast_dims, "fast_u8_dims")?;
                    profile_dispatch();
                    encoder.dispatch_thread_groups(
                        MTLSize::new(
                            checked_nsuint(batch, "batch")?,
                            checked_nsuint(out_dim.div_ceil(8), "fast u8 out groups")?,
                            1,
                        ),
                        MTLSize::new(64, 1, 1),
                    );
                    post_dispatch_barrier(encoder);
                } else {
                    encoder.set_compute_pipeline_state(&self.affine_matmul_rhs_t_u32_f32);
                    encoder.set_buffer(0, Some(lhs_buffer), 0);
                    encoder.set_buffer(1, Some(packed), 0);
                    encoder.set_buffer(2, Some(scales), 0);
                    encoder.set_buffer(3, Some(biases), 0);
                    encoder.set_buffer(4, Some(output_buffer), 0);
                    set_u32_bytes(encoder, 5, &dims, "dims")?;
                    set_u32_bytes(encoder, 6, &quant, "quant")?;
                    self.dispatch_qmv(encoder, &self.affine_matmul_rhs_t_u32_f32, *out_dim, batch)?;
                }
                Ok(*out_dim)
            }
        }
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
        if !fast_affine_qmv_enabled(*out_dim)
            || *bits != FAST_QMV_BITS
            || *group_size != FAST_QMV_GROUP_SIZE
            || in_dim % 512 != 0
        {
            return Ok(None);
        }
        let fast_dims = [
            checked_u32(*out_dim, "rms qmv out_dim")?,
            checked_u32(in_dim, "rms qmv in_dim")?,
            checked_u32(*packed_cols, "rms qmv packed_cols")?,
            checked_u32(*groups, "rms qmv groups")?,
        ];
        encoder.set_compute_pipeline_state(&self.affine_qmv_rms_fast_u4_gs64_f32);
        encoder.set_buffer(0, Some(lhs_buffer), 0);
        encoder.set_buffer(1, Some(rms_weight_buffer), 0);
        encoder.set_buffer(2, Some(packed), 0);
        encoder.set_buffer(3, Some(scales), 0);
        encoder.set_buffer(4, Some(biases), 0);
        encoder.set_buffer(5, Some(output_buffer), 0);
        set_u32_bytes(encoder, 6, &fast_dims, "rms_qmv_dims")?;
        set_f32_bytes(encoder, 7, &[rms_eps], "rms_qmv_eps")?;
        profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(
                1,
                checked_nsuint(out_dim.div_ceil(8), "rms qmv out groups")?,
                1,
            ),
            MTLSize::new(64, 1, 1),
        );
        post_dispatch_barrier(encoder);
        Ok(Some(*out_dim))
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "helper interne partagé par deux variantes d'encodage Metal"
    )]
    pub(super) fn encode_matmul_weight_inner(
        &self,
        encoder: &ComputeCommandEncoderRef,
        lhs_buffer: &BufferRef,
        batch: usize,
        in_dim: usize,
        weight: &LinearWeight,
        output_buffer: &BufferRef,
        prefer_fast_affine: bool,
    ) -> Result<usize> {
        match weight {
            LinearWeight::Dense(weight) => {
                let (out_dim, rhs_in_dim) = weight.as_matrix()?;
                if in_dim != rhs_in_dim {
                    return Err(InferError::Dimension(format!(
                        "matmul Metal encodé x=[{batch},{in_dim}] rhs=[{out_dim},{rhs_in_dim}]"
                    )));
                }
                let rhs_buffer = self.cached_buffer_from_f32(weight.data(), "rhs")?;
                let dims = [
                    checked_u32(batch, "batch")?,
                    checked_u32(out_dim, "out_dim")?,
                    checked_u32(in_dim, "in_dim")?,
                ];
                encoder.set_compute_pipeline_state(&self.dense_matmul_rhs_t_f32);
                encoder.set_buffer(0, Some(lhs_buffer), 0);
                encoder.set_buffer(1, Some(&rhs_buffer), 0);
                encoder.set_buffer(2, Some(output_buffer), 0);
                set_u32_bytes(encoder, 3, &dims, "dims")?;
                self.dispatch_qmv(encoder, &self.dense_matmul_rhs_t_f32, out_dim, batch)?;
                Ok(out_dim)
            }
            LinearWeight::AffineQuantized(weight) => {
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
                if can_use_fast_affine_qmm2(batch, in_dim, weight) {
                    let fast_dims = [
                        checked_u32(*out_dim, "qmm2 out_dim")?,
                        checked_u32(in_dim, "qmm2 in_dim")?,
                        checked_u32(*packed_cols, "qmm2 packed_cols")?,
                        checked_u32(groups, "qmm2 groups")?,
                    ];
                    encoder.set_compute_pipeline_state(&self.affine_qmm2_fast_aligned_u4_gs64_f32);
                    encoder.set_buffer(0, Some(lhs_buffer), 0);
                    encoder.set_buffer(1, Some(&packed_buffer), 0);
                    encoder.set_buffer(2, Some(&scales_buffer), 0);
                    encoder.set_buffer(3, Some(&biases_buffer), 0);
                    encoder.set_buffer(4, Some(output_buffer), 0);
                    set_u32_bytes(encoder, 5, &fast_dims, "qmm2_dims")?;
                    profile_dispatch();
                    encoder.dispatch_thread_groups(
                        MTLSize::new(
                            1,
                            checked_nsuint(out_dim.div_ceil(8), "qmm2 out groups")?,
                            1,
                        ),
                        MTLSize::new(64, 1, 1),
                    );
                    post_dispatch_barrier(encoder);
                } else if can_use_fast_affine_qmv(batch, in_dim, weight)
                    || (prefer_fast_affine && can_use_fast_affine_qmv_shape(batch, in_dim, weight))
                {
                    let fast_dims = [
                        checked_u32(*out_dim, "fast out_dim")?,
                        checked_u32(in_dim, "fast in_dim")?,
                        checked_u32(*packed_cols, "fast packed_cols")?,
                        checked_u32(groups, "fast groups")?,
                    ];
                    let pipeline = if *out_dim % 8 == 0 {
                        &self.affine_qmv_fast_aligned_u4_gs64_f32
                    } else {
                        &self.affine_qmv_fast_u4_gs64_f32
                    };
                    encoder.set_compute_pipeline_state(pipeline);
                    encoder.set_buffer(0, Some(lhs_buffer), 0);
                    encoder.set_buffer(1, Some(&packed_buffer), 0);
                    encoder.set_buffer(2, Some(&scales_buffer), 0);
                    encoder.set_buffer(3, Some(&biases_buffer), 0);
                    encoder.set_buffer(4, Some(output_buffer), 0);
                    set_u32_bytes(encoder, 5, &fast_dims, "fast_dims")?;
                    profile_dispatch();
                    encoder.dispatch_thread_groups(
                        MTLSize::new(
                            checked_nsuint(batch, "batch")?,
                            checked_nsuint(out_dim.div_ceil(8), "fast out groups")?,
                            1,
                        ),
                        MTLSize::new(64, 1, 1),
                    );
                    post_dispatch_barrier(encoder);
                } else {
                    encoder.set_compute_pipeline_state(&self.affine_matmul_rhs_t_u32_f32);
                    encoder.set_buffer(0, Some(lhs_buffer), 0);
                    encoder.set_buffer(1, Some(&packed_buffer), 0);
                    encoder.set_buffer(2, Some(&scales_buffer), 0);
                    encoder.set_buffer(3, Some(&biases_buffer), 0);
                    encoder.set_buffer(4, Some(output_buffer), 0);
                    set_u32_bytes(encoder, 5, &dims, "dims")?;
                    set_u32_bytes(encoder, 6, &quant, "quant")?;
                    self.dispatch_qmv(encoder, &self.affine_matmul_rhs_t_u32_f32, *out_dim, batch)?;
                }
                Ok(*out_dim)
            }
        }
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
            let pipeline = if weight.in_dim % 512 == 0 {
                &self.affine_gather_qmv_fast_u4_gs64_f32
            } else {
                &self.affine_gather_qmv_tail_u4_gs64_f32
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
            profile_dispatch();
            encoder.dispatch_thread_groups(
                MTLSize::new(
                    checked_nsuint(topk, "gather topk")?,
                    checked_nsuint(weight.out_dim.div_ceil(8), "gather fast out groups")?,
                    1,
                ),
                MTLSize::new(64, 1, 1),
            );
            post_dispatch_barrier(encoder);
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
            self.dispatch_qmv(
                encoder,
                &self.affine_gather_matmul_rhs_t_u32_f32,
                weight.out_dim,
                topk,
            )
        }
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
        profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(checked_nsuint(rows, "rms rows")?, 1, 1),
            MTLSize::new(MATMUL_ROW_TG_WIDTH, 1, 1),
        );
        post_dispatch_barrier(encoder);
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
        self.dispatch_1d(encoder, &self.swiglu_f32, len)
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
        self.dispatch_1d(encoder, &self.accumulate_scaled_f32, len)
    }

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
