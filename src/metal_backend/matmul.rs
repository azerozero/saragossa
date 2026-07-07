//! Matmuls Metal et résolution des poids linéaires.

use super::*;

/// Active le chemin GEMM bf16 de l'encodeur Whisper (matmul2d Neural Accelerators)
/// par défaut si disponible. `RETI_STT_F32=1` force l'ancien chemin f32 ;
/// `RETI_STT_BF16=0` garde aussi un kill-switch compatible avec les anciens benches.
/// bf16-input / accumulation f32.
pub(crate) fn whisper_bf16_gemm_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| {
        if std::env::var_os("RETI_STT_F32").is_some() {
            return false;
        }
        !matches!(
            std::env::var("RETI_STT_BF16").as_deref(),
            Ok("0" | "false" | "off" | "no")
        )
    })
}

/// Active le QMV decode Whisper avec poids bf16 row-major pour les projections
/// M=1 où le microbench montre un gain. `RETI_STT_F32=1` force le chemin f32.
pub(crate) fn whisper_decode_bf16_qmv_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| {
        if std::env::var_os("RETI_STT_F32").is_some() {
            return false;
        }
        !matches!(
            std::env::var("RETI_STT_DECODE_BF16_QMV").as_deref(),
            Ok("0" | "false" | "off" | "no")
        )
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "wrappers d'encodage Metal: buffers, dimensions et offsets restent explicites"
)]
impl MetalExecutor {
    /// Multiplie `input` par la transposée du poids dense logique `[out,in]`.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si les dimensions sont incompatibles ou si Metal échoue.
    pub fn matmul_rhs_t_dense(&self, input: &Tensor, rhs_out_in: &Tensor) -> Result<Tensor> {
        let (batch, in_dim) = input.as_matrix()?;
        let (out_dim, rhs_in_dim) = rhs_out_in.as_matrix()?;
        if batch == 0 || in_dim == 0 || out_dim == 0 {
            return Err(InferError::Dimension(format!(
                "matmul Metal dimensions nulles x=[{batch},{in_dim}] rhs=[{out_dim},{rhs_in_dim}]"
            )));
        }
        if in_dim != rhs_in_dim {
            return Err(InferError::Dimension(format!(
                "matmul Metal x=[{batch},{in_dim}] rhs_t_source=[{out_dim},{rhs_in_dim}]"
            )));
        }

        let lhs_buffer = self.upload_f32_buffer(input.data(), "input")?;
        let rhs_buffer = self.cached_buffer_from_f32(rhs_out_in.data(), "rhs")?;
        let output_len = checked_len(batch, out_dim, "sortie matmul Metal")?;
        let output_buffer = self.device.new_buffer(
            byte_len::<f32>(output_len)?,
            MTLResourceOptions::StorageModeShared,
        );
        let dims = [
            checked_u32(batch, "batch")?,
            checked_u32(out_dim, "out_dim")?,
            checked_u32(in_dim, "in_dim")?,
        ];
        let dims_buffer = self.buffer_from_u32(&dims, "dims")?;

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let encoder_guard = EncoderEndGuard::new(encoder);
        encoder.set_compute_pipeline_state(&self.dense_matmul_rhs_t_f32);
        encoder.set_buffer(0, Some(&lhs_buffer), 0);
        encoder.set_buffer(1, Some(&rhs_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_buffer(3, Some(&dims_buffer), 0);
        let threads_per_group = self.qmv_thread_group_size(&self.dense_matmul_rhs_t_f32);
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
        encoder_guard.end();
        commit_and_wait(command_buffer)?;

        let output = read_f32_buffer(&output_buffer, output_len)?;
        Tensor::from_vec(vec![batch, out_dim], output)
    }

    /// Encode le matmul dense **qmv** `out = lhs · rhs^T` dans un encoder PARTAGÉ
    /// (buffers résidents, zéro readback) — cœur de [`Self::matmul_rhs_t_dense`]
    /// réutilisé par le decode résident. MÊME kernel `dense_matmul_rhs_t_f32` ⇒
    /// byte-identique au decode per-op (qui passe par `matmul_rhs_t_dense` en
    /// batch=1). `out` peut être lié à un offset (append KV in-place).
    pub(crate) fn encode_dense_qmv(
        &self,
        encoder: &ComputeCommandEncoderRef,
        lhs: &BufferRef,
        rhs: &BufferRef,
        out: &BufferRef,
        out_offset_bytes: u64,
        batch: usize,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<()> {
        let dims = [
            checked_u32(batch, "qmv batch")?,
            checked_u32(out_dim, "qmv out_dim")?,
            checked_u32(in_dim, "qmv in_dim")?,
        ];
        encoder.set_compute_pipeline_state(&self.dense_matmul_rhs_t_f32);
        encoder.set_buffer(0, Some(lhs), 0);
        encoder.set_buffer(1, Some(rhs), 0);
        encoder.set_buffer(2, Some(out), out_offset_bytes);
        encoder.set_bytes(
            3,
            std::mem::size_of::<[u32; 3]>() as u64,
            dims.as_ptr().cast::<std::ffi::c_void>(),
        );
        let threads_per_group = self.qmv_thread_group_size(&self.dense_matmul_rhs_t_f32);
        profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(
                checked_nsuint(out_dim, "qmv out_dim")?,
                checked_nsuint(batch, "qmv batch")?,
                1,
            ),
            MTLSize::new(threads_per_group, 1, 1),
        );
        post_dispatch_barrier(encoder);
        Ok(())
    }

    /// QMV dense batch=1 avec poids bf16 row-major `[out_dim,in_dim]`.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si une dimension déborde ou si Metal échoue.
    pub(crate) fn encode_dense_qmv_rhs_bf16(
        &self,
        encoder: &ComputeCommandEncoderRef,
        lhs: &BufferRef,
        rhs_bf16: &BufferRef,
        out: &BufferRef,
        batch: usize,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<()> {
        let dims = [
            checked_u32(batch, "bf16 qmv batch")?,
            checked_u32(out_dim, "bf16 qmv out_dim")?,
            checked_u32(in_dim, "bf16 qmv in_dim")?,
        ];
        encoder.set_compute_pipeline_state(&self.dense_qmv_rhs_bf16_f32);
        encoder.set_buffer(0, Some(lhs), 0);
        encoder.set_buffer(1, Some(rhs_bf16), 0);
        encoder.set_buffer(2, Some(out), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<[u32; 3]>() as u64,
            dims.as_ptr().cast::<std::ffi::c_void>(),
        );
        let threads_per_group = self.qmv_thread_group_size(&self.dense_qmv_rhs_bf16_f32);
        profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(
                checked_nsuint(out_dim, "bf16 qmv out_dim")?,
                checked_nsuint(batch, "bf16 qmv batch")?,
                1,
            ),
            MTLSize::new(threads_per_group, 1, 1),
        );
        post_dispatch_barrier(encoder);
        Ok(())
    }

    /// Variante GEMM tuilée de [`Self::matmul_rhs_t_dense`] pour `batch ≫ 1`
    /// (prefill encodeur Whisper). Le kernel qmv re-lit la ligne de poids par
    /// ligne de batch (memory-bound) ; ici les tuiles 16×16 sont réutilisées en
    /// mémoire threadgroup. Accumulation in-order (≈ `dot` CPU), pas l'ordre
    /// `simd_sum` du qmv → résultats f32 voisins mais non bit-à-bit identiques.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si les dimensions sont incompatibles ou si Metal échoue.
    pub(crate) fn matmul_rhs_t_dense_tiled(
        &self,
        input: &Tensor,
        rhs_out_in: &Tensor,
    ) -> Result<Tensor> {
        let (batch, in_dim) = input.as_matrix()?;
        let (out_dim, rhs_in_dim) = rhs_out_in.as_matrix()?;
        if batch == 0 || in_dim == 0 || out_dim == 0 {
            return Err(InferError::Dimension(format!(
                "gemm Metal dimensions nulles x=[{batch},{in_dim}] rhs=[{out_dim},{rhs_in_dim}]"
            )));
        }
        if in_dim != rhs_in_dim {
            return Err(InferError::Dimension(format!(
                "gemm Metal x=[{batch},{in_dim}] rhs_t_source=[{out_dim},{rhs_in_dim}]"
            )));
        }

        let lhs_buffer = self.upload_f32_buffer(input.data(), "gemm_input")?;
        let rhs_buffer = self.cached_buffer_from_f32(rhs_out_in.data(), "gemm_rhs")?;
        let output_len = checked_len(batch, out_dim, "sortie gemm Metal")?;
        let output_buffer = self.device.new_buffer(
            byte_len::<f32>(output_len)?,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let encoder_guard = EncoderEndGuard::new(encoder);
        self.encode_dense_gemm(
            encoder,
            &lhs_buffer,
            &rhs_buffer,
            &output_buffer,
            batch,
            out_dim,
            in_dim,
        )?;
        encoder_guard.end();
        commit_and_wait(command_buffer)?;

        let output = read_f32_buffer(&output_buffer, output_len)?;
        Tensor::from_vec(vec![batch, out_dim], output)
    }

    /// Encode le GEMM dense tuilé `out = lhs · rhs^T` dans un encoder PARTAGÉ
    /// (buffers résidents, zéro readback) — cœur de [`Self::matmul_rhs_t_dense_tiled`]
    /// réutilisé par l'encodeur résident. MÊME kernel `dense_gemm_rhs_t_f32` ⇒
    /// résultats byte-identiques au chemin standalone.
    pub(crate) fn encode_dense_gemm(
        &self,
        encoder: &ComputeCommandEncoderRef,
        lhs: &BufferRef,
        rhs: &BufferRef,
        out: &BufferRef,
        batch: usize,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<()> {
        let dims = [
            checked_u32(batch, "batch")?,
            checked_u32(out_dim, "out_dim")?,
            checked_u32(in_dim, "in_dim")?,
        ];
        // GEMM dense f32 byte-identique. Le chemin bf16 (encodeur) passe par le
        // matmul2d NA résident (`encode_na_gemm`), pas par ce dispatch.
        encoder.set_compute_pipeline_state(&self.dense_gemm_rhs_t_f32);
        encoder.set_buffer(0, Some(lhs), 0);
        encoder.set_buffer(1, Some(rhs), 0);
        encoder.set_buffer(2, Some(out), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<[u32; 3]>() as u64,
            dims.as_ptr().cast::<std::ffi::c_void>(),
        );
        const TILE: NSUInteger = 64; // tuile de sortie 64×64 (micro-bloc 4×4 / thread)
        const THREADS: NSUInteger = 16; // 16×16 = 256 threads
        let groups = MTLSize::new(
            (checked_nsuint(out_dim, "out_dim")? + TILE - 1) / TILE,
            (checked_nsuint(batch, "batch")? + TILE - 1) / TILE,
            1,
        );
        profile_dispatch();
        encoder.dispatch_thread_groups(groups, MTLSize::new(THREADS, THREADS, 1));
        post_dispatch_barrier(encoder);
        Ok(())
    }

    /// Multiplie `input` par un poids affine compact MLX `[out,in]`.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si les dimensions ou paramètres quantifiés divergent.
    pub fn matmul_rhs_t_affine(
        &self,
        input: &Tensor,
        weight: &AffineQuantizedTensor,
    ) -> Result<Tensor> {
        let (batch, in_dim) = input.as_matrix()?;
        let [out_dim, weight_in_dim] = weight.shape() else {
            return Err(InferError::Dimension(format!(
                "poids Metal quantifié attendu rang 2, reçu {:?}",
                weight.shape()
            )));
        };
        let [packed_rows, packed_cols] = weight.packed_shape() else {
            return Err(InferError::Dimension(format!(
                "packed_shape Metal attendu rang 2, reçu {:?}",
                weight.packed_shape()
            )));
        };
        if batch == 0 || in_dim == 0 || *out_dim == 0 {
            return Err(InferError::Dimension(format!(
                "matmul Metal quantifié dimensions nulles x=[{batch},{in_dim}] rhs=[{out_dim},{weight_in_dim}]"
            )));
        }
        if in_dim != *weight_in_dim || *packed_rows != *out_dim {
            return Err(InferError::Dimension(format!(
                "matmul Metal quantifié x=[{batch},{in_dim}] rhs=[{out_dim},{weight_in_dim}] packed={:?}",
                weight.packed_shape()
            )));
        }
        let groups = in_dim
            .checked_div(weight.group_size())
            .ok_or_else(|| InferError::Metal("group_size quantifié nul".to_string()))?;
        if groups * weight.group_size() != in_dim {
            return Err(InferError::Dimension(format!(
                "in_dim={in_dim} non divisible par group_size={}",
                weight.group_size()
            )));
        }

        let lhs_buffer = self.upload_f32_buffer(input.data(), "input")?;
        let packed_buffer = self.cached_buffer_from_u32(weight.packed_data(), "packed")?;
        let scales_buffer =
            self.cached_buffer_from_f32_as_bf16(weight.scales().data(), "scales")?;
        let biases_buffer =
            self.cached_buffer_from_f32_as_bf16(weight.biases().data(), "biases")?;
        let output_len = checked_len(batch, *out_dim, "sortie matmul Metal quantifiée")?;
        let output_buffer = self.device.new_buffer(
            byte_len::<f32>(output_len)?,
            MTLResourceOptions::StorageModeShared,
        );
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
        let dims_buffer = self.buffer_from_u32(&dims, "dims")?;
        let quant_buffer = self.buffer_from_u32(&quant, "quant")?;
        let mut owned_buffers = Vec::new();

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let encoder_guard = EncoderEndGuard::new(encoder);
        if can_use_fast_affine_qmm2(batch, in_dim, weight)
            || can_use_fast_affine_qmm2_u8(batch, in_dim, weight)
        {
            let fast_dims = [
                checked_u32(*out_dim, "qmm2 out_dim")?,
                checked_u32(in_dim, "qmm2 in_dim")?,
                checked_u32(*packed_cols, "qmm2 packed_cols")?,
                checked_u32(groups, "qmm2 groups")?,
            ];
            let fast_dims_buffer = self.buffer_from_u32(&fast_dims, "qmm2_dims")?;
            owned_buffers.push(fast_dims_buffer.clone());
            let pipeline = if weight.bits() == FAST_QMV_BITS {
                &self.affine_qmm2_fast_aligned_u4_gs64_f32
            } else if weight.group_size() == FAST_QMV_GROUP_SIZE {
                &self.affine_qmm2_fast_aligned_u8_gs64_f32
            } else {
                &self.affine_qmm2_fast_aligned_u8_gs128_f32
            };
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(&lhs_buffer), 0);
            encoder.set_buffer(1, Some(&packed_buffer), 0);
            encoder.set_buffer(2, Some(&scales_buffer), 0);
            encoder.set_buffer(3, Some(&biases_buffer), 0);
            encoder.set_buffer(4, Some(&output_buffer), 0);
            encoder.set_buffer(5, Some(&fast_dims_buffer), 0);
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
        } else if can_use_fast_affine_qmv(batch, in_dim, weight) {
            let fast_dims = [
                checked_u32(*out_dim, "fast out_dim")?,
                checked_u32(in_dim, "fast in_dim")?,
                checked_u32(*packed_cols, "fast packed_cols")?,
                checked_u32(groups, "fast groups")?,
            ];
            let fast_dims_buffer = self.buffer_from_u32(&fast_dims, "fast_dims")?;
            owned_buffers.push(fast_dims_buffer.clone());
            let pipeline = if *out_dim % 8 == 0 {
                &self.affine_qmv_fast_aligned_u4_gs64_f32
            } else {
                &self.affine_qmv_fast_u4_gs64_f32
            };
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(&lhs_buffer), 0);
            encoder.set_buffer(1, Some(&packed_buffer), 0);
            encoder.set_buffer(2, Some(&scales_buffer), 0);
            encoder.set_buffer(3, Some(&biases_buffer), 0);
            encoder.set_buffer(4, Some(&output_buffer), 0);
            encoder.set_buffer(5, Some(&fast_dims_buffer), 0);
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
        } else if can_use_fast_affine_qmv_u6(batch, in_dim, weight) {
            let fast_dims = [
                checked_u32(*out_dim, "fast u6 out_dim")?,
                checked_u32(in_dim, "fast u6 in_dim")?,
                checked_u32(*packed_cols, "fast u6 packed_cols")?,
                checked_u32(groups, "fast u6 groups")?,
            ];
            let fast_dims_buffer = self.buffer_from_u32(&fast_dims, "fast_u6_dims")?;
            owned_buffers.push(fast_dims_buffer.clone());
            let pipeline = if *out_dim % 2 == 0 {
                &self.affine_qmv_fast_aligned_u6_gs64_f32
            } else {
                &self.affine_qmv_fast_u6_gs64_f32
            };
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(&lhs_buffer), 0);
            encoder.set_buffer(1, Some(&packed_buffer), 0);
            encoder.set_buffer(2, Some(&scales_buffer), 0);
            encoder.set_buffer(3, Some(&biases_buffer), 0);
            encoder.set_buffer(4, Some(&output_buffer), 0);
            encoder.set_buffer(5, Some(&fast_dims_buffer), 0);
            profile_dispatch();
            encoder.dispatch_thread_groups(
                MTLSize::new(
                    checked_nsuint(batch, "batch")?,
                    checked_nsuint(out_dim.div_ceil(2), "fast u6 out groups")?,
                    1,
                ),
                MTLSize::new(64, 1, 1),
            );
            post_dispatch_barrier(encoder);
        } else if can_use_fast_affine_qmv_u8(batch, in_dim, weight) {
            let fast_dims = [
                checked_u32(*out_dim, "fast u8 out_dim")?,
                checked_u32(in_dim, "fast u8 in_dim")?,
                checked_u32(*packed_cols, "fast u8 packed_cols")?,
                checked_u32(groups, "fast u8 groups")?,
            ];
            let fast_dims_buffer = self.buffer_from_u32(&fast_dims, "fast_u8_dims")?;
            owned_buffers.push(fast_dims_buffer.clone());
            let pipeline = match weight.group_size() == FAST_QMV_GROUP_SIZE {
                true => &self.affine_qmv_fast_aligned_u8_gs64_f32,
                false => &self.affine_qmv_fast_aligned_u8_gs128_f32,
            };
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(&lhs_buffer), 0);
            encoder.set_buffer(1, Some(&packed_buffer), 0);
            encoder.set_buffer(2, Some(&scales_buffer), 0);
            encoder.set_buffer(3, Some(&biases_buffer), 0);
            encoder.set_buffer(4, Some(&output_buffer), 0);
            encoder.set_buffer(5, Some(&fast_dims_buffer), 0);
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
            encoder.set_buffer(0, Some(&lhs_buffer), 0);
            encoder.set_buffer(1, Some(&packed_buffer), 0);
            encoder.set_buffer(2, Some(&scales_buffer), 0);
            encoder.set_buffer(3, Some(&biases_buffer), 0);
            encoder.set_buffer(4, Some(&output_buffer), 0);
            encoder.set_buffer(5, Some(&dims_buffer), 0);
            encoder.set_buffer(6, Some(&quant_buffer), 0);
            let threads_per_group = self.qmv_thread_group_size(&self.affine_matmul_rhs_t_u32_f32);
            profile_dispatch();
            encoder.dispatch_thread_groups(
                MTLSize::new(
                    checked_nsuint(*out_dim, "out_dim")?,
                    checked_nsuint(batch, "batch")?,
                    1,
                ),
                MTLSize::new(threads_per_group, 1, 1),
            );
            post_dispatch_barrier(encoder);
        }
        encoder_guard.end();
        commit_and_wait(command_buffer)?;

        let output = read_f32_buffer(&output_buffer, output_len)?;
        Tensor::from_vec(vec![batch, *out_dim], output)
    }

    /// Projette trois couches linéaires indépendantes dans une seule commande.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si une projection porte un biais ou si les dimensions
    /// sont incompatibles avec l'entrée.
    pub(crate) fn project_three_biasless(
        &self,
        input: &Tensor,
        first: &Linear,
        second: &Linear,
        third: &Linear,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        ensure_biasless(first, "first")?;
        ensure_biasless(second, "second")?;
        ensure_biasless(third, "third")?;
        let (batch, in_dim) = input.as_matrix()?;
        let first_dim = linear_out_dim(first.weight())?;
        let second_dim = linear_out_dim(second.weight())?;
        let third_dim = linear_out_dim(third.weight())?;
        let input_buffer = self.upload_f32_buffer(input.data(), "project3_input")?;
        let first_buffer = self.new_f32_buffer(
            checked_len(batch, first_dim, "project3 first")?,
            "project3_first",
        )?;
        let second_buffer = self.new_f32_buffer(
            checked_len(batch, second_dim, "project3 second")?,
            "project3_second",
        )?;
        let third_buffer = self.new_f32_buffer(
            checked_len(batch, third_dim, "project3 third")?,
            "project3_third",
        )?;
        let mut owned_buffers = Vec::new();
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let encoder_guard = EncoderEndGuard::new(encoder);
        self.encode_matmul_weight(
            encoder,
            &mut owned_buffers,
            &input_buffer,
            batch,
            in_dim,
            first.weight(),
            &first_buffer,
        )?;
        self.encode_matmul_weight(
            encoder,
            &mut owned_buffers,
            &input_buffer,
            batch,
            in_dim,
            second.weight(),
            &second_buffer,
        )?;
        self.encode_matmul_weight(
            encoder,
            &mut owned_buffers,
            &input_buffer,
            batch,
            in_dim,
            third.weight(),
            &third_buffer,
        )?;
        encoder_guard.end();
        commit_and_wait(command_buffer)?;

        let first = read_f32_buffer(&first_buffer, batch * first_dim)?;
        let second = read_f32_buffer(&second_buffer, batch * second_dim)?;
        let third = read_f32_buffer(&third_buffer, batch * third_dim)?;
        Ok((
            Tensor::from_vec(vec![batch, first_dim], first)?,
            Tensor::from_vec(vec![batch, second_dim], second)?,
            Tensor::from_vec(vec![batch, third_dim], third)?,
        ))
    }

    /// Projette quatre couches linéaires indépendantes dans une seule commande.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si une projection porte un biais ou si les dimensions
    /// sont incompatibles avec l'entrée.
    pub(crate) fn project_four_biasless(
        &self,
        input: &Tensor,
        first: &Linear,
        second: &Linear,
        third: &Linear,
        fourth: &Linear,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        ensure_biasless(first, "first")?;
        ensure_biasless(second, "second")?;
        ensure_biasless(third, "third")?;
        ensure_biasless(fourth, "fourth")?;
        let (batch, in_dim) = input.as_matrix()?;
        let first_dim = linear_out_dim(first.weight())?;
        let second_dim = linear_out_dim(second.weight())?;
        let third_dim = linear_out_dim(third.weight())?;
        let fourth_dim = linear_out_dim(fourth.weight())?;
        let input_buffer = self.upload_f32_buffer(input.data(), "project4_input")?;
        let first_buffer = self.new_f32_buffer(
            checked_len(batch, first_dim, "project4 first")?,
            "project4_first",
        )?;
        let second_buffer = self.new_f32_buffer(
            checked_len(batch, second_dim, "project4 second")?,
            "project4_second",
        )?;
        let third_buffer = self.new_f32_buffer(
            checked_len(batch, third_dim, "project4 third")?,
            "project4_third",
        )?;
        let fourth_buffer = self.new_f32_buffer(
            checked_len(batch, fourth_dim, "project4 fourth")?,
            "project4_fourth",
        )?;
        let mut owned_buffers = Vec::new();
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let encoder_guard = EncoderEndGuard::new(encoder);
        self.encode_matmul_weight(
            encoder,
            &mut owned_buffers,
            &input_buffer,
            batch,
            in_dim,
            first.weight(),
            &first_buffer,
        )?;
        self.encode_matmul_weight(
            encoder,
            &mut owned_buffers,
            &input_buffer,
            batch,
            in_dim,
            second.weight(),
            &second_buffer,
        )?;
        self.encode_matmul_weight(
            encoder,
            &mut owned_buffers,
            &input_buffer,
            batch,
            in_dim,
            third.weight(),
            &third_buffer,
        )?;
        self.encode_matmul_weight(
            encoder,
            &mut owned_buffers,
            &input_buffer,
            batch,
            in_dim,
            fourth.weight(),
            &fourth_buffer,
        )?;
        encoder_guard.end();
        commit_and_wait(command_buffer)?;

        let first = read_f32_buffer(&first_buffer, batch * first_dim)?;
        let second = read_f32_buffer(&second_buffer, batch * second_dim)?;
        let third = read_f32_buffer(&third_buffer, batch * third_dim)?;
        let fourth = read_f32_buffer(&fourth_buffer, batch * fourth_dim)?;
        Ok((
            Tensor::from_vec(vec![batch, first_dim], first)?,
            Tensor::from_vec(vec![batch, second_dim], second)?,
            Tensor::from_vec(vec![batch, third_dim], third)?,
            Tensor::from_vec(vec![batch, fourth_dim], fourth)?,
        ))
    }

    /// Encode un matmul `[batch,in_dim] · weightᵀ` (dense ou quantifié) vers
    /// `output_buffer` résident, sans commit ni readback. Exposé `pub(crate)` pour
    /// le chaînage des projections du decode résident (`decode_resident.rs`, 1c).
    pub(crate) fn encode_matmul_weight(
        &self,
        encoder: &ComputeCommandEncoderRef,
        owned_buffers: &mut Vec<metal::Buffer>,
        lhs_buffer: &BufferRef,
        batch: usize,
        in_dim: usize,
        weight: &LinearWeight,
        output_buffer: &BufferRef,
    ) -> Result<usize> {
        self.encode_matmul_weight_inner(
            encoder,
            owned_buffers,
            lhs_buffer,
            batch,
            in_dim,
            weight,
            output_buffer,
            false,
        )
    }

    /// Résout les buffers Metal d'un poids linéaire une fois par session résidente.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si la forme du poids est invalide ou si la création du
    /// buffer Metal échoue.
    pub(crate) fn resolve_linear_weight_buffers(
        &self,
        weight: &LinearWeight,
        label: &'static str,
    ) -> Result<MetalLinearWeightBuffers> {
        match weight {
            LinearWeight::Dense(weight) => {
                let (out_dim, in_dim) = weight.as_matrix()?;
                Ok(MetalLinearWeightBuffers::Dense {
                    rhs: self.cached_buffer_from_f32(weight.data(), label)?,
                    rhs_bf16: None,
                    out_dim,
                    in_dim,
                })
            }
            LinearWeight::AffineQuantized(weight) => {
                let [out_dim, in_dim] = weight.shape() else {
                    return Err(InferError::Dimension(format!(
                        "poids Metal quantifié attendu rang 2, reçu {:?}",
                        weight.shape()
                    )));
                };
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
                Ok(MetalLinearWeightBuffers::AffineQuantized {
                    packed: self.cached_buffer_from_u32(weight.packed_data(), label)?,
                    scales: self.cached_buffer_from_f32_as_bf16(weight.scales().data(), label)?,
                    biases: self.cached_buffer_from_f32_as_bf16(weight.biases().data(), label)?,
                    out_dim: *out_dim,
                    in_dim: *in_dim,
                    packed_cols: *packed_cols,
                    group_size: weight.group_size(),
                    bits: weight.bits(),
                    groups,
                })
            }
        }
    }

    pub(crate) fn resolve_concat_linear_weight_buffers(
        &self,
        weights: &[&LinearWeight],
        label: &'static str,
    ) -> Result<MetalLinearWeightBuffers> {
        let Some(first) = weights.first() else {
            return Err(InferError::Dimension(format!(
                "{label}: liste de poids vide"
            )));
        };
        match first {
            LinearWeight::Dense(first_weight) => {
                let (_, in_dim) = first_weight.as_matrix()?;
                let mut out_dim = 0usize;
                let mut data = Vec::new();
                for weight in weights {
                    let LinearWeight::Dense(weight) = weight else {
                        return Err(InferError::Dimension(format!(
                            "{label}: mélange dense/quantifié non supporté"
                        )));
                    };
                    let (rows, cols) = weight.as_matrix()?;
                    if cols != in_dim {
                        return Err(InferError::Dimension(format!(
                            "{label}: in_dim incompatible {cols} != {in_dim}"
                        )));
                    }
                    out_dim = out_dim.checked_add(rows).ok_or_else(|| {
                        InferError::Dimension(format!("{label}: out_dim concat déborde"))
                    })?;
                    data.extend_from_slice(weight.data());
                }
                Ok(MetalLinearWeightBuffers::Dense {
                    rhs: self.buffer_from_slice(&data, label)?,
                    rhs_bf16: None,
                    out_dim,
                    in_dim,
                })
            }
            LinearWeight::AffineQuantized(first_weight) => {
                let [_, in_dim] = first_weight.shape() else {
                    return Err(InferError::Dimension(format!(
                        "{label}: poids quantifié attendu rang 2, reçu {:?}",
                        first_weight.shape()
                    )));
                };
                let [_, packed_cols] = first_weight.packed_shape() else {
                    return Err(InferError::Dimension(format!(
                        "{label}: packed_shape attendu rang 2, reçu {:?}",
                        first_weight.packed_shape()
                    )));
                };
                let group_size = first_weight.group_size();
                let bits = first_weight.bits();
                let groups = in_dim
                    .checked_div(group_size)
                    .ok_or_else(|| InferError::Metal(format!("{label}: group_size nul")))?;
                let mut out_dim = 0usize;
                let mut packed = Vec::new();
                let mut scales = Vec::new();
                let mut biases = Vec::new();
                for weight in weights {
                    let LinearWeight::AffineQuantized(weight) = weight else {
                        return Err(InferError::Dimension(format!(
                            "{label}: mélange dense/quantifié non supporté"
                        )));
                    };
                    let [rows, cols] = weight.shape() else {
                        return Err(InferError::Dimension(format!(
                            "{label}: poids quantifié attendu rang 2, reçu {:?}",
                            weight.shape()
                        )));
                    };
                    let [packed_rows, cols_packed] = weight.packed_shape() else {
                        return Err(InferError::Dimension(format!(
                            "{label}: packed_shape attendu rang 2, reçu {:?}",
                            weight.packed_shape()
                        )));
                    };
                    if *cols != *in_dim
                        || cols_packed != packed_cols
                        || weight.group_size() != group_size
                        || weight.bits() != bits
                    {
                        return Err(InferError::Dimension(format!(
                            "{label}: poids concat incompatibles cols={cols}/{in_dim} packed_cols={cols_packed}/{packed_cols} group_size={}/{} bits={}/{}",
                            weight.group_size(),
                            group_size,
                            weight.bits(),
                            bits
                        )));
                    }
                    if packed_rows != rows {
                        return Err(InferError::Dimension(format!(
                            "{label}: packed_rows={packed_rows} incompatible avec rows={rows}"
                        )));
                    }
                    out_dim = out_dim.checked_add(*rows).ok_or_else(|| {
                        InferError::Dimension(format!("{label}: out_dim concat déborde"))
                    })?;
                    packed.extend_from_slice(weight.packed_data());
                    scales.extend_from_slice(weight.scales().data());
                    biases.extend_from_slice(weight.biases().data());
                }
                Ok(MetalLinearWeightBuffers::AffineQuantized {
                    packed: self.buffer_from_slice(&packed, label)?,
                    scales: self.buffer_from_f32_as_bf16(&scales, label)?,
                    biases: self.buffer_from_f32_as_bf16(&biases, label)?,
                    out_dim,
                    in_dim: *in_dim,
                    packed_cols: *packed_cols,
                    group_size,
                    bits,
                    groups,
                })
            }
        }
    }
}
