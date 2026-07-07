//! Ops encode-form de l'encodeur Whisper résident : LayerNorm, GELU, biais.
//!
//! Le LLM résident a RMSNorm/SwiGLU ; Whisper a LayerNorm (moyenne + biais) et
//! GELU exact. Ces kernels reproduisent `norm.rs::layer_norm` et
//! `activation.rs::gelu_scalar`. Chaque op existe en deux formes : `encode_*`
//! (dans un command buffer PARTAGÉ, zéro readback — pour le chemin résident) et
//! un wrapper standalone (commit+wait+readback — pour les tests `==CPU`).

use super::*;
use std::ffi::c_void;

impl MetalExecutor {
    /// Encode une LayerNorm ligne par ligne : `out = (x-µ)/σ · weight + bias`.
    pub(crate) fn encode_layer_norm_rows(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input: &BufferRef,
        weight: &BufferRef,
        bias: &BufferRef,
        out: &BufferRef,
        rows: usize,
        dim: usize,
        eps: f32,
    ) -> Result<()> {
        let dim_u32 = checked_u32(dim, "layer_norm dim")?;
        encoder.set_compute_pipeline_state(&self.layer_norm_rows_f32);
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(weight), 0);
        encoder.set_buffer(2, Some(bias), 0);
        encoder.set_buffer(3, Some(out), 0);
        encoder.set_bytes(4, 4, std::ptr::from_ref(&dim_u32).cast::<c_void>());
        encoder.set_bytes(5, 4, std::ptr::from_ref(&eps).cast::<c_void>());
        profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(checked_nsuint(rows, "layer_norm rows")?, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        post_dispatch_barrier(encoder);
        Ok(())
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "signature d'encodage Metal: buffers + dimensions"
    )]
    pub(crate) fn encode_layer_norm_rows_bf16out(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input: &BufferRef,
        weight: &BufferRef,
        bias: &BufferRef,
        out: &BufferRef,
        out_bf16: &BufferRef,
        rows: usize,
        dim: usize,
        eps: f32,
    ) -> Result<()> {
        let dim_u32 = checked_u32(dim, "layer_norm dim")?;
        encoder.set_compute_pipeline_state(&self.layer_norm_rows_f32_bf16out);
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(weight), 0);
        encoder.set_buffer(2, Some(bias), 0);
        encoder.set_buffer(3, Some(out), 0);
        encoder.set_buffer(4, Some(out_bf16), 0);
        encoder.set_bytes(5, 4, std::ptr::from_ref(&dim_u32).cast::<c_void>());
        encoder.set_bytes(6, 4, std::ptr::from_ref(&eps).cast::<c_void>());
        profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(checked_nsuint(rows, "layer_norm rows")?, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        post_dispatch_barrier(encoder);
        Ok(())
    }

    /// Encode `summed = left + right` puis `normed = LayerNorm(summed)` (fusion
    /// résiduel + norm).
    #[expect(
        clippy::too_many_arguments,
        reason = "signature d'encodage Metal: buffers + dimensions"
    )]
    pub(crate) fn encode_add_layer_norm_rows(
        &self,
        encoder: &ComputeCommandEncoderRef,
        left: &BufferRef,
        right: &BufferRef,
        weight: &BufferRef,
        bias: &BufferRef,
        summed: &BufferRef,
        normed: &BufferRef,
        rows: usize,
        dim: usize,
        eps: f32,
    ) -> Result<()> {
        let dim_u32 = checked_u32(dim, "add_layer_norm dim")?;
        encoder.set_compute_pipeline_state(&self.add_layer_norm_rows_f32);
        encoder.set_buffer(0, Some(left), 0);
        encoder.set_buffer(1, Some(right), 0);
        encoder.set_buffer(2, Some(weight), 0);
        encoder.set_buffer(3, Some(bias), 0);
        encoder.set_buffer(4, Some(summed), 0);
        encoder.set_buffer(5, Some(normed), 0);
        encoder.set_bytes(6, 4, std::ptr::from_ref(&dim_u32).cast::<c_void>());
        encoder.set_bytes(7, 4, std::ptr::from_ref(&eps).cast::<c_void>());
        profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(checked_nsuint(rows, "add_layer_norm rows")?, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        post_dispatch_barrier(encoder);
        Ok(())
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "signature d'encodage Metal: buffers + dimensions"
    )]
    pub(crate) fn encode_add_layer_norm_rows_bf16out(
        &self,
        encoder: &ComputeCommandEncoderRef,
        left: &BufferRef,
        right: &BufferRef,
        weight: &BufferRef,
        bias: &BufferRef,
        summed: &BufferRef,
        normed: &BufferRef,
        normed_bf16: &BufferRef,
        rows: usize,
        dim: usize,
        eps: f32,
    ) -> Result<()> {
        let dim_u32 = checked_u32(dim, "add_layer_norm dim")?;
        encoder.set_compute_pipeline_state(&self.add_layer_norm_rows_f32_bf16out);
        encoder.set_buffer(0, Some(left), 0);
        encoder.set_buffer(1, Some(right), 0);
        encoder.set_buffer(2, Some(weight), 0);
        encoder.set_buffer(3, Some(bias), 0);
        encoder.set_buffer(4, Some(summed), 0);
        encoder.set_buffer(5, Some(normed), 0);
        encoder.set_buffer(6, Some(normed_bf16), 0);
        encoder.set_bytes(7, 4, std::ptr::from_ref(&dim_u32).cast::<c_void>());
        encoder.set_bytes(8, 4, std::ptr::from_ref(&eps).cast::<c_void>());
        profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(checked_nsuint(rows, "add_layer_norm rows")?, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        post_dispatch_barrier(encoder);
        Ok(())
    }

    /// Encode la GELU exacte élément par élément `out = gelu(input)`.
    pub(crate) fn encode_gelu(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input: &BufferRef,
        out: &BufferRef,
        len: usize,
    ) -> Result<()> {
        let len_u32 = checked_u32(len, "gelu len")?;
        encoder.set_compute_pipeline_state(&self.gelu_f32);
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(out), 0);
        encoder.set_bytes(2, 4, std::ptr::from_ref(&len_u32).cast::<c_void>());
        let width = self.gelu_f32.thread_execution_width().max(1);
        profile_dispatch();
        encoder.dispatch_threads(
            MTLSize::new(checked_nsuint(len, "gelu len")?, 1, 1),
            MTLSize::new(width, 1, 1),
        );
        post_dispatch_barrier(encoder);
        Ok(())
    }

    pub(crate) fn encode_gelu_bf16out(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input: &BufferRef,
        out: &BufferRef,
        out_bf16: &BufferRef,
        len: usize,
    ) -> Result<()> {
        let len_u32 = checked_u32(len, "gelu len")?;
        encoder.set_compute_pipeline_state(&self.gelu_f32_bf16out);
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(out), 0);
        encoder.set_buffer(2, Some(out_bf16), 0);
        encoder.set_bytes(3, 4, std::ptr::from_ref(&len_u32).cast::<c_void>());
        let width = self.gelu_f32_bf16out.thread_execution_width().max(1);
        profile_dispatch();
        encoder.dispatch_threads(
            MTLSize::new(checked_nsuint(len, "gelu len")?, 1, 1),
            MTLSize::new(width, 1, 1),
        );
        post_dispatch_barrier(encoder);
        Ok(())
    }

    /// Encode l'ajout du biais ligne à ligne (in-place) `data[r,c] += bias[c]`.
    /// `data` peut être lié à un offset (épilogue d'un append KV résident).
    pub(crate) fn encode_add_row_bias(
        &self,
        encoder: &ComputeCommandEncoderRef,
        data: &BufferRef,
        data_offset_bytes: u64,
        bias: &BufferRef,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        let dims = [
            checked_u32(rows, "add_row_bias rows")?,
            checked_u32(cols, "add_row_bias cols")?,
        ];
        let total = checked_len(rows, cols, "add_row_bias total")?;
        encoder.set_compute_pipeline_state(&self.add_row_bias_f32);
        encoder.set_buffer(0, Some(data), data_offset_bytes);
        encoder.set_buffer(1, Some(bias), 0);
        encoder.set_bytes(2, 8, dims.as_ptr().cast::<c_void>());
        let width = self.add_row_bias_f32.thread_execution_width().max(1);
        profile_dispatch();
        encoder.dispatch_threads(
            MTLSize::new(checked_nsuint(total, "add_row_bias total")?, 1, 1),
            MTLSize::new(width, 1, 1),
        );
        post_dispatch_barrier(encoder);
        Ok(())
    }

    /// Encode l'im2col conv1d : `input [frames, in_ch]` → `output [out_frames,
    /// in_ch·kernel]` (zéro-pad), pour un GEMM tuilé conv1d résident.
    #[expect(
        clippy::too_many_arguments,
        reason = "signature im2col: buffers + dimensions conv"
    )]
    pub(crate) fn encode_im2col(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input: &BufferRef,
        output: &BufferRef,
        out_frames: usize,
        in_ch: usize,
        kernel: usize,
        frames: usize,
        stride: usize,
        padding: usize,
    ) -> Result<()> {
        let dims = [
            checked_u32(out_frames, "im2col out_frames")?,
            checked_u32(in_ch, "im2col in_ch")?,
            checked_u32(kernel, "im2col kernel")?,
            checked_u32(frames, "im2col frames")?,
        ];
        let params = [
            checked_u32(stride, "im2col stride")?,
            checked_u32(padding, "im2col padding")?,
        ];
        let total = checked_len(out_frames, in_ch * kernel, "im2col total")?;
        encoder.set_compute_pipeline_state(&self.im2col_f32);
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(output), 0);
        encoder.set_bytes(2, 16, dims.as_ptr().cast::<c_void>());
        encoder.set_bytes(3, 8, params.as_ptr().cast::<c_void>());
        let width = self.im2col_f32.thread_execution_width().max(1);
        profile_dispatch();
        encoder.dispatch_threads(
            MTLSize::new(checked_nsuint(total, "im2col total")?, 1, 1),
            MTLSize::new(width, 1, 1),
        );
        post_dispatch_barrier(encoder);
        Ok(())
    }

    /// Encode la conversion f32 → bf16 `[n]` (+ barrière).
    pub(crate) fn encode_f32_to_bf16(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input: &BufferRef,
        out_bf16: &BufferRef,
        n: usize,
    ) -> Result<()> {
        let n_u32 = checked_u32(n, "f32_to_bf16 n")?;
        record_prefill_f32_to_bf16_shape(n);
        encoder.set_compute_pipeline_state(&self.f32_to_bf16);
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(out_bf16), 0);
        encoder.set_bytes(2, 4, std::ptr::from_ref(&n_u32).cast::<c_void>());
        let width = self.f32_to_bf16.thread_execution_width().max(1);
        trace_dispatch_path("f32_to_bf16", n, 1, 0);
        profile_dispatch();
        encoder.dispatch_threads(
            MTLSize::new(checked_nsuint(n, "f32_to_bf16 n")?, 1, 1),
            MTLSize::new(width, 1, 1),
        );
        post_dispatch_barrier(encoder);
        Ok(())
    }

    /// GEMM Neural Accelerators standalone `out[M,N] = lhs[M,K]·rhs[N,K]^T`
    /// (bf16-input / f32-accum, = `matmul2d`). `lhs`/`rhs` f32 → convertis bf16 ;
    /// `rhs` est transposé en `[K,N]` (= `rhs^T`) pour le matmul standard prouvé.
    /// Renvoie `None` si la NA n'est pas dispo. Réservé au test de correctness
    /// (le chemin prod passe par `cached_rhs_t_bf16` + `encode_na_gemm` résident).
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si les dimensions divergent ou si Metal échoue.
    #[cfg(test)]
    pub(crate) fn na_gemm(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Option<Tensor>> {
        let Some(pso) = self.na_gemm_bf16.clone() else {
            return Ok(None);
        };
        let (m, k) = lhs.as_matrix()?;
        let (n, k2) = rhs.as_matrix()?;
        if k != k2 {
            return Err(InferError::Dimension(format!(
                "na_gemm: lhs[{m},{k}] rhs[{n},{k2}]"
            )));
        }
        // rhs [N,K] → rhs^T [K,N] (CPU).
        let rhs_data = rhs.data();
        let mut rhs_t = vec![0.0_f32; k * n];
        for nn in 0..n {
            for kk in 0..k {
                rhs_t[kk * n + nn] = rhs_data[nn * k + kk];
            }
        }
        let a_f32 = self.upload_f32_buffer(lhs.data(), "na_a_f32")?;
        let bt_f32 = self.upload_f32_buffer(&rhs_t, "na_bt_f32")?;
        let a_bf16 = self
            .device
            .new_buffer((m * k * 2) as u64, MTLResourceOptions::StorageModeShared);
        let b_bf16 = self
            .device
            .new_buffer((k * n * 2) as u64, MTLResourceOptions::StorageModeShared);
        let out = self.device.new_buffer(
            byte_len::<f32>(checked_len(m, n, "na out")?)?,
            MTLResourceOptions::StorageModeShared,
        );
        let mnk = [
            checked_u32(m, "na m")?,
            checked_u32(n, "na n")?,
            checked_u32(k, "na k")?,
        ];

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let guard = EncoderEndGuard::new(encoder);
        self.encode_f32_to_bf16(encoder, &a_f32, &a_bf16, m * k)?;
        self.encode_f32_to_bf16(encoder, &bt_f32, &b_bf16, k * n)?;
        encoder.set_compute_pipeline_state(&pso);
        encoder.set_buffer(0, Some(&a_bf16), 0);
        encoder.set_buffer(1, Some(&b_bf16), 0);
        encoder.set_buffer(2, Some(&out), 0);
        encoder.set_bytes(3, 12, mnk.as_ptr().cast::<c_void>());
        let width = pso.thread_execution_width().max(1);
        profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(m.div_ceil(64) as u64, n.div_ceil(32) as u64, 1),
            MTLSize::new(width * 4, 1, 1),
        );
        post_dispatch_barrier(encoder);
        guard.end();
        commit_and_wait(command_buffer)?;
        Ok(Some(Tensor::from_vec(
            vec![m, n],
            read_f32_buffer(&out, m * n)?,
        )?))
    }

    /// Construit (et cache par ptr source) le poids transposé bf16 `rhs^T [K,N]`
    /// depuis `rhs [N,K]` f32, pour le GEMM NA résident. Transpose + arrondi bf16
    /// (RTNE) une seule fois ; réutilisé sur toutes les transcriptions.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si `rhs` n'est pas une matrice ou si Metal échoue.
    pub(crate) fn cached_rhs_t_bf16(&self, rhs: &Tensor) -> Result<Buffer> {
        let (n, k) = rhs.as_matrix()?;
        self.cached_rhs_t_bf16_matrix(rhs.data(), n, k, rhs.data().as_ptr() as usize)
    }

    /// Construit (et cache par clé stable) le poids transposé bf16 `rhs^T [K,N]`
    /// depuis des données f32 interprétées comme `rhs [N,K]`.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si la forme ne correspond pas aux données ou si Metal
    /// échoue.
    pub(crate) fn cached_rhs_t_bf16_matrix(
        &self,
        data: &[f32],
        n: usize,
        k: usize,
        key: usize,
    ) -> Result<Buffer> {
        let expected = checked_len(n, k, "rhs bf16 matrix")?;
        if data.len() != expected {
            return Err(InferError::Shape(format!(
                "rhs bf16 matrix: len={}, attendu {expected} pour [{n},{k}]",
                data.len()
            )));
        }
        if let Some(buf) = self
            .bf16_rhs_t_cache
            .lock()
            .expect("invariant: bf16_rhs_t_cache lock")
            .get(&key)
        {
            return Ok(buf.clone());
        }
        // rhs [N,K] → rhs^T [K,N] + arrondi bf16 (RTNE) en un passage.
        let mut bf = vec![0u16; checked_len(k, n, "rhs_t bf16")?];
        for nn in 0..n {
            for kk in 0..k {
                let bits = data[nn * k + kk].to_bits();
                bf[kk * n + nn] =
                    ((bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) & 0xffff) as u16;
            }
        }
        let buffer = self.device.new_buffer_with_data(
            bf.as_ptr().cast::<c_void>(),
            byte_len::<u16>(bf.len())?,
            MTLResourceOptions::StorageModeShared,
        );
        self.bf16_rhs_t_cache
            .lock()
            .expect("invariant: bf16_rhs_t_cache lock")
            .insert(key, buffer.clone());
        Ok(buffer)
    }

    /// Encode le GEMM NA résident `out[M,N] = lhs_bf16[M,K]·rhs_t_bf16[K,N]`
    /// (matmul2d, accum f32) dans un command buffer partagé. Opérandes déjà bf16.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si la NA est indisponible ou si les dims débordent.
    #[expect(
        clippy::too_many_arguments,
        reason = "GEMM bas niveau (dims + buffers)"
    )]
    pub(crate) fn encode_na_gemm(
        &self,
        encoder: &ComputeCommandEncoderRef,
        lhs_bf16: &BufferRef,
        rhs_t_bf16: &BufferRef,
        out: &BufferRef,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        let use_bn128 = !matches!(
            std::env::var("RETI_STT_NAX_BN128").as_deref(),
            Ok("0" | "false" | "off" | "no")
        ) && n % 128 == 0;
        let pso = if use_bn128 {
            self.na_gemm_bf16_bn128
                .as_ref()
                .or(self.na_gemm_bf16.as_ref())
        } else {
            self.na_gemm_bf16.as_ref()
        }
        .ok_or_else(|| InferError::Config("encode_na_gemm: NA indisponible".into()))?;
        let tile_n = if use_bn128 && self.na_gemm_bf16_bn128.is_some() {
            128
        } else {
            32
        };
        let simdgroups = if tile_n == 128 { 8 } else { 4 };
        let mnk = [
            checked_u32(m, "na m")?,
            checked_u32(n, "na n")?,
            checked_u32(k, "na k")?,
        ];
        encoder.set_compute_pipeline_state(pso);
        encoder.set_buffer(0, Some(lhs_bf16), 0);
        encoder.set_buffer(1, Some(rhs_t_bf16), 0);
        encoder.set_buffer(2, Some(out), 0);
        encoder.set_bytes(3, 12, mnk.as_ptr().cast::<c_void>());
        let width = pso.thread_execution_width().max(1);
        trace_dispatch_path(
            if tile_n == 128 {
                "gemm_nax_bf16_bn128"
            } else {
                "gemm_nax"
            },
            m,
            n,
            k,
        );
        profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(m.div_ceil(64) as u64, n.div_ceil(tile_n) as u64, 1),
            MTLSize::new(width * simdgroups, 1, 1),
        );
        post_dispatch_barrier(encoder);
        Ok(())
    }

    /// Encode l'attention single-query `out = softmax(q·Kᵀ·scale)·V` (decode
    /// Whisper, self ou cross), une tête par threadgroup.
    pub(crate) fn encode_whisper_attn_decode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        q: &BufferRef,
        keys: &BufferRef,
        values: &BufferRef,
        out: &BufferRef,
        heads: usize,
        head_dim: usize,
        len: usize,
    ) -> Result<()> {
        let dims = [
            checked_u32(heads, "attn heads")?,
            checked_u32(head_dim, "attn head_dim")?,
            checked_u32(len, "attn len")?,
            0u32,
        ];
        let use_vec64 = head_dim == 64;
        let pipeline = if use_vec64 {
            &self.whisper_attn_decode_vec64_f32
        } else {
            &self.whisper_attn_decode_f32
        };
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(q), 0);
        encoder.set_buffer(1, Some(keys), 0);
        encoder.set_buffer(2, Some(values), 0);
        encoder.set_buffer(3, Some(out), 0);
        encoder.set_bytes(4, 16, dims.as_ptr().cast::<c_void>());
        profile_dispatch();
        let threads = if use_vec64 { 1024 } else { 256 };
        encoder.dispatch_thread_groups(
            MTLSize::new(checked_nsuint(heads, "attn heads")?, 1, 1),
            MTLSize::new(threads, 1, 1),
        );
        post_dispatch_barrier(encoder);
        Ok(())
    }

    /// Frontend conv mel Whisper RÉSIDENT : `conv1(im2col)→GELU→conv2(im2col)→GELU
    /// →+positions` dans UN command buffer (zéro readback), via le GEMM tuilé.
    /// `nlc` = mel transposé `[frames, num_mel_bins]` (frames-major). Renvoie
    /// `h [out_frames2, d_model]` prêt pour l'encodeur. Conv en GEMM ⇒ ordre
    /// d'accumulation différent du conv CPU (drift ~1e-6, vérifié au golden).
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si une dimension déborde ou si Metal échoue.
    pub(crate) fn encode_whisper_conv(
        &self,
        nlc: &Tensor,
        w: &WhisperConvWeights,
    ) -> Result<Tensor> {
        let resident = self.encode_whisper_conv_resident(nlc, w)?;
        Tensor::from_vec(
            vec![resident.rows, resident.cols],
            read_f32_buffer(
                &resident.buffer,
                checked_len(resident.rows, resident.cols, "conv out")?,
            )?,
        )
    }

    /// Frontend conv mel Whisper en buffer GPU résident.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si une dimension déborde ou si Metal échoue.
    pub(crate) fn encode_whisper_conv_resident(
        &self,
        nlc: &Tensor,
        w: &WhisperConvWeights,
    ) -> Result<WhisperResidentMatrix> {
        let (frames1, in_ch1) = nlc.as_matrix()?;
        if in_ch1 != w.num_mel_bins {
            return Err(InferError::Dimension(format!(
                "conv résident: nlc in_ch={in_ch1}, attendu {}",
                w.num_mel_bins
            )));
        }
        let d = w.d_model;
        let k = w.kernel;
        let pad = 1usize;
        // conv1 stride 1, conv2 stride 2 (Whisper).
        let out1 = (frames1 + 2 * pad)
            .checked_sub(k)
            .ok_or_else(|| InferError::Dimension("conv1 kernel > entrée".to_string()))?
            + 1;
        let out2 = (out1 + 2 * pad)
            .checked_sub(k)
            .ok_or_else(|| InferError::Dimension("conv2 kernel > entrée".to_string()))?
            / 2
            + 1;
        let cols1 = in_ch1 * k;
        let cols2 = d * k;
        let n1 = checked_len(out1, d, "conv n1")?;
        let n2 = checked_len(out2, d, "conv n2")?;

        let nlc_buf = self.upload_f32_buffer(nlc.data(), "conv_nlc")?;
        let im1 = self.new_f32_buffer(checked_len(out1, cols1, "conv im1")?, "conv_im1")?;
        let x1 = self.new_f32_buffer(n1, "conv_x1")?;
        let g1 = self.new_f32_buffer(n1, "conv_g1")?;
        let im2 = self.new_f32_buffer(checked_len(out2, cols2, "conv im2")?, "conv_im2")?;
        let h = self.new_f32_buffer(n2, "conv_h")?;
        let g2 = self.new_f32_buffer(n2, "conv_g2")?;
        let mut owned: Vec<Buffer> = Vec::new();
        let use_na = super::whisper_bf16_gemm_enabled()
            && !matches!(
                std::env::var("RETI_STT_CONV_NAX").as_deref(),
                Ok("0" | "false" | "off" | "no")
            )
            && w.conv1_weight_na.is_some()
            && w.conv2_weight_na.is_some()
            && self.na_gemm_bf16.is_some();

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let guard = EncoderEndGuard::new(encoder);

        // conv1 : im2col → GEMM(+biais) → GELU.
        self.encode_im2col(encoder, &nlc_buf, &im1, out1, in_ch1, k, frames1, 1, pad)?;
        if let (true, Some(conv1_weight_na)) = (use_na, w.conv1_weight_na.as_ref()) {
            let im1_bf16 =
                self.new_bf16_buffer(checked_len(out1, cols1, "conv_im1_bf16")?, "conv_im1_bf16")?;
            self.encode_f32_to_bf16(encoder, &im1, &im1_bf16, out1 * cols1)?;
            self.encode_na_gemm(encoder, &im1_bf16, conv1_weight_na, &x1, out1, d, cols1)?;
        } else {
            self.encode_dense_gemm(encoder, &im1, &w.conv1_weight, &x1, out1, d, cols1)?;
        }
        self.encode_add_row_bias(encoder, &x1, 0, &w.conv1_bias, out1, d)?;
        self.encode_gelu(encoder, &x1, &g1, n1)?;
        // conv2 : im2col(g1) → GEMM(+biais) → GELU.
        self.encode_im2col(encoder, &g1, &im2, out2, d, k, out1, 2, pad)?;
        if let (true, Some(conv2_weight_na)) = (use_na, w.conv2_weight_na.as_ref()) {
            let im2_bf16 =
                self.new_bf16_buffer(checked_len(out2, cols2, "conv_im2_bf16")?, "conv_im2_bf16")?;
            self.encode_f32_to_bf16(encoder, &im2, &im2_bf16, out2 * cols2)?;
            self.encode_na_gemm(encoder, &im2_bf16, conv2_weight_na, &h, out2, d, cols2)?;
        } else {
            self.encode_dense_gemm(encoder, &im2, &w.conv2_weight, &h, out2, d, cols2)?;
        }
        self.encode_add_row_bias(encoder, &h, 0, &w.conv2_bias, out2, d)?;
        self.encode_gelu(encoder, &h, &g2, n2)?;
        // + positions (g2 += positions[0..out2]).
        self.encode_accumulate_scaled(encoder, &mut owned, &w.positions, &g2, 1.0, n2)?;

        guard.end();
        commit_and_wait(command_buffer)?;
        Ok(WhisperResidentMatrix {
            buffer: g2,
            rows: out2,
            cols: d,
        })
    }

    /// Encodeur Whisper RÉSIDENT : déroule les `n` couches dans UN command buffer,
    /// buffers GPU résidents, ZÉRO readback entre ops (vs ~7 `commit_and_wait`/
    /// couche du chemin per-op). Math f32 inchangée (mêmes GEMM/attention ; seuls
    /// LayerNorm/GELU/add migrent GPU) ⇒ transcription byte-identique au golden.
    ///
    /// `h_input` = sortie conv + positions `[seq, d_model]` (conv reste CPU,
    /// un upload). Renvoie les features encodeur `[seq, d_model]` (un readback).
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si une dimension déborde ou si Metal échoue.
    pub(crate) fn encode_whisper_encoder(
        &self,
        h_input: &Tensor,
        enc: &WhisperResidentEncoder,
    ) -> Result<Tensor> {
        let (seq, dim) = h_input.as_matrix()?;
        let h = self.upload_f32_buffer(h_input.data(), "we_h")?;
        let resident = self.encode_whisper_encoder_buffer(&h, seq, dim, enc)?;
        Tensor::from_vec(
            vec![resident.rows, resident.cols],
            read_f32_buffer(
                &resident.buffer,
                checked_len(resident.rows, resident.cols, "we out")?,
            )?,
        )
    }

    fn encode_whisper_encoder_buffer(
        &self,
        h: &BufferRef,
        seq: usize,
        dim: usize,
        enc: &WhisperResidentEncoder,
    ) -> Result<WhisperResidentMatrix> {
        if dim != enc.d_model {
            return Err(InferError::Dimension(format!(
                "encodeur résident: h dim={dim}, attendu {}",
                enc.d_model
            )));
        }
        if enc.heads == 0 || dim % enc.heads != 0 {
            return Err(InferError::Dimension(format!(
                "encodeur résident: d_model={dim} non divisible par heads={}",
                enc.heads
            )));
        }
        let head_dim = dim / enc.heads;
        let ffn = enc.ffn_dim;
        let nd = checked_len(seq, dim, "we nd")?;
        let nf = checked_len(seq, ffn, "we nf")?;
        let eps = enc.eps;

        // Buffers résidents (mémoïsés par label ; réutilisés entre énoncés).
        let normed = self.new_f32_buffer(nd, "we_normed")?;
        let q = self.new_f32_buffer(nd, "we_q")?;
        let k = self.new_f32_buffer(nd, "we_k")?;
        let v = self.new_f32_buffer(nd, "we_v")?;
        let ctx = self.new_f32_buffer(nd, "we_ctx")?;
        let proj = self.new_f32_buffer(nd, "we_proj")?;
        let f1 = self.new_f32_buffer(nf, "we_f1")?;
        let g = self.new_f32_buffer(nf, "we_g")?;

        // Chemin bf16 matmul2d NA (`RETI_STT_BF16`) : opérandes lhs convertis bf16,
        // poids déjà cachés en bf16 (`weight_na`). Accumulation f32 → transcription
        // préservée (vérifié golden). Sinon : GEMM dense f32 byte-identique.
        let na = self.na_gemm_bf16.is_some()
            && super::whisper_bf16_gemm_enabled()
            && enc.layers.iter().all(|l| {
                l.q.weight_na.is_some()
                    && l.k.weight_na.is_some()
                    && l.v.weight_na.is_some()
                    && l.o.weight_na.is_some()
                    && l.fc1.weight_na.is_some()
                    && l.fc2.weight_na.is_some()
            });
        let normed_bf16 = self.new_bf16_buffer(nd, "we_normed_bf16")?;
        let ctx_bf16 = self.new_bf16_buffer(nd, "we_ctx_bf16")?;
        let g_bf16 = self.new_bf16_buffer(nf, "we_g_bf16")?;

        let attn_spec = PrefillAttentionSpec {
            seq,
            hidden_dim: dim,
            q_heads: enc.heads,
            kv_heads: enc.heads,
            head_dim,
            rope_dims: 0,
            rope_theta: 0.0,
            eps: 0.0,
        };

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let guard = EncoderEndGuard::new(encoder);

        let first = enc
            .layers
            .first()
            .ok_or_else(|| InferError::Config("encodeur résident: zéro couche".to_string()))?;
        if na {
            self.encode_layer_norm_rows_bf16out(
                encoder,
                &h,
                &first.self_ln.weight,
                &first.self_ln.bias,
                &normed,
                &normed_bf16,
                seq,
                dim,
                eps,
            )?;
        } else {
            self.encode_layer_norm_rows(
                encoder,
                &h,
                &first.self_ln.weight,
                &first.self_ln.bias,
                &normed,
                seq,
                dim,
                eps,
            )?;
        }

        for (i, layer) in enc.layers.iter().enumerate() {
            // Self-attention : q/k/v depuis `normed`.
            self.encode_proj_gemm(
                encoder,
                na,
                &normed,
                &normed_bf16,
                &layer.q,
                &q,
                seq,
                dim,
                dim,
            )?;
            self.encode_proj_bias(encoder, &q, layer.q.bias.as_ref(), seq, dim)?;
            self.encode_proj_gemm(
                encoder,
                na,
                &normed,
                &normed_bf16,
                &layer.k,
                &k,
                seq,
                dim,
                dim,
            )?;
            self.encode_proj_bias(encoder, &k, layer.k.bias.as_ref(), seq, dim)?;
            self.encode_proj_gemm(
                encoder,
                na,
                &normed,
                &normed_bf16,
                &layer.v,
                &v,
                seq,
                dim,
                dim,
            )?;
            self.encode_proj_bias(encoder, &v, layer.v.bias.as_ref(), seq, dim)?;
            self.encode_noncausal_attention_prefill(encoder, &q, &k, &v, &ctx, attn_spec)?;
            if na {
                self.encode_f32_to_bf16(encoder, &ctx, &ctx_bf16, nd)?;
            }
            self.encode_proj_gemm(encoder, na, &ctx, &ctx_bf16, &layer.o, &proj, seq, dim, dim)?;
            self.encode_proj_bias(encoder, &proj, layer.o.bias.as_ref(), seq, dim)?;
            // h += attn ; normed = final_ln(h).
            if na {
                self.encode_add_layer_norm_rows_bf16out(
                    encoder,
                    &h,
                    &proj,
                    &layer.final_ln.weight,
                    &layer.final_ln.bias,
                    &h,
                    &normed,
                    &normed_bf16,
                    seq,
                    dim,
                    eps,
                )?;
            } else {
                self.encode_add_layer_norm_rows(
                    encoder,
                    &h,
                    &proj,
                    &layer.final_ln.weight,
                    &layer.final_ln.bias,
                    &h,
                    &normed,
                    seq,
                    dim,
                    eps,
                )?;
            }
            // FFN : fc1 → gelu → fc2.
            self.encode_proj_gemm(
                encoder,
                na,
                &normed,
                &normed_bf16,
                &layer.fc1,
                &f1,
                seq,
                ffn,
                dim,
            )?;
            self.encode_proj_bias(encoder, &f1, layer.fc1.bias.as_ref(), seq, ffn)?;
            if na {
                self.encode_gelu_bf16out(encoder, &f1, &g, &g_bf16, nf)?;
            } else {
                self.encode_gelu(encoder, &f1, &g, nf)?;
            }
            self.encode_proj_gemm(encoder, na, &g, &g_bf16, &layer.fc2, &proj, seq, dim, ffn)?;
            self.encode_proj_bias(encoder, &proj, layer.fc2.bias.as_ref(), seq, dim)?;
            // h += ff ; normed = norm de la couche suivante (ou LayerNorm encodeur final).
            let next = if i + 1 < enc.layers.len() {
                &enc.layers[i + 1].self_ln
            } else {
                &enc.encoder_ln
            };
            if na {
                self.encode_add_layer_norm_rows_bf16out(
                    encoder,
                    &h,
                    &proj,
                    &next.weight,
                    &next.bias,
                    &h,
                    &normed,
                    &normed_bf16,
                    seq,
                    dim,
                    eps,
                )?;
            } else {
                self.encode_add_layer_norm_rows(
                    encoder,
                    &h,
                    &proj,
                    &next.weight,
                    &next.bias,
                    &h,
                    &normed,
                    seq,
                    dim,
                    eps,
                )?;
            }
        }

        guard.end();
        commit_and_wait(command_buffer)?;
        // `normed` porte le LayerNorm encodeur final (dernière fusion add+norm).
        Ok(WhisperResidentMatrix {
            buffer: normed,
            rows: seq,
            cols: dim,
        })
    }

    /// GEMM d'une projection encodeur : matmul2d NA (`na`, lhs bf16 + `weight_na`)
    /// ou GEMM dense f32 byte-identique. `out[m,n] = lhs[m,k]·weight[n,k]^T`.
    #[expect(
        clippy::too_many_arguments,
        reason = "GEMM bas niveau (dims + buffers)"
    )]
    fn encode_proj_gemm(
        &self,
        encoder: &ComputeCommandEncoderRef,
        na: bool,
        lhs_f32: &BufferRef,
        lhs_bf16: &BufferRef,
        proj: &WhisperResidentProj,
        out: &BufferRef,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        if na {
            let wna = proj
                .weight_na
                .as_ref()
                .ok_or_else(|| InferError::Config("encodeur NA: weight_na manquant".to_string()))?;
            self.encode_na_gemm(encoder, lhs_bf16, wna, out, m, n, k)
        } else {
            self.encode_dense_gemm(encoder, lhs_f32, &proj.weight, out, m, n, k)
        }
    }

    fn encode_decode_ffn_qmv(
        &self,
        encoder: &ComputeCommandEncoderRef,
        lhs: &BufferRef,
        proj: &WhisperResidentProj,
        out: &BufferRef,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<()> {
        if super::whisper_decode_bf16_qmv_enabled() {
            if let Some(weight_bf16) = proj.weight_bf16.as_ref() {
                return self.encode_dense_qmv_rhs_bf16(
                    encoder,
                    lhs,
                    weight_bf16,
                    out,
                    1,
                    out_dim,
                    in_dim,
                );
            }
        }
        self.encode_dense_qmv(encoder, lhs, &proj.weight, out, 0, 1, out_dim, in_dim)
    }

    /// Ajoute le biais si présent (k_proj est biasless dans Whisper).
    fn encode_proj_bias(
        &self,
        encoder: &ComputeCommandEncoderRef,
        data: &BufferRef,
        bias: Option<&Buffer>,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        if let Some(bias) = bias {
            self.encode_add_row_bias(encoder, data, 0, bias, rows, cols)?;
        }
        Ok(())
    }

    /// Decode Whisper RÉSIDENT d'UN token : couches + head greedy dans UN command
    /// buffer, KV self **résident** (append device-side à `kv.len`), KV cross
    /// **résident** (statique), zéro readback entre ops, un seul readback final de
    /// l'id token (`u32`). Cela supprime le readback complet des logits et le
    /// deuxième command buffer `decoder_head` du chemin per-op. Incrémente
    /// `kv.len`.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si une dimension déborde, si la capacité KV est atteinte
    /// ou si Metal échoue.
    pub(crate) fn encode_whisper_decode_step(
        &self,
        h_input: &Tensor,
        dec: &WhisperResidentDecoder,
        kv: &mut WhisperDecodeKv,
        emit_argmax: bool,
    ) -> Result<Option<u32>> {
        let (rows, dim) = h_input.as_matrix()?;
        if rows != 1 || dim != dec.d_model {
            return Err(InferError::Dimension(format!(
                "decode résident: h=[{rows},{dim}], attendu [1,{}]",
                dec.d_model
            )));
        }
        if kv.len >= kv.capacity {
            return Err(InferError::Dimension(format!(
                "decode résident: capacité KV {} atteinte",
                kv.capacity
            )));
        }
        let n = dec.layers.len();
        if kv.self_keys.len() != n || kv.cross_keys.len() != n {
            return Err(InferError::Dimension(
                "decode résident: nombre de couches KV incohérent".to_string(),
            ));
        }
        let head_dim = dim / dec.heads;
        let ffn = dec.ffn_dim;
        let eps = dec.eps;
        let self_len = kv.len + 1; // append puis attention sur [0..=len]
        let kv_off = u64::try_from(kv.len * dim * std::mem::size_of::<f32>()).map_err(|_| {
            InferError::Dimension("decode résident: offset KV hors u64".to_string())
        })?;

        let h = self.upload_f32_buffer(h_input.data(), "wd_h")?;
        let normed = self.new_f32_buffer(dim, "wd_normed")?;
        let q = self.new_f32_buffer(dim, "wd_q")?;
        let ctx = self.new_f32_buffer(dim, "wd_ctx")?;
        let proj = self.new_f32_buffer(dim, "wd_proj")?;
        let cross_q = self.new_f32_buffer(dim, "wd_cross_q")?;
        let f1 = self.new_f32_buffer(ffn, "wd_f1")?;
        let g = self.new_f32_buffer(ffn, "wd_g")?;
        let next_index = if emit_argmax {
            Some(self.new_u32_buffer(1, "wd_next_index")?)
        } else {
            None
        };
        let mut owned: Vec<Buffer> = Vec::new();

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let guard = EncoderEndGuard::new(encoder);

        let first = &dec.layers[0];
        self.encode_layer_norm_rows(
            encoder,
            &h,
            &first.self_ln.weight,
            &first.self_ln.bias,
            &normed,
            1,
            dim,
            eps,
        )?;

        for (i, layer) in dec.layers.iter().enumerate() {
            let self_keys = &kv.self_keys[i];
            let self_values = &kv.self_values[i];
            // Self-attn : q→scratch, k/v→append device-side dans le KV à la ligne `len`.
            self.encode_dense_qmv(encoder, &normed, &layer.q.weight, &q, 0, 1, dim, dim)?;
            self.encode_proj_bias(encoder, &q, layer.q.bias.as_ref(), 1, dim)?;
            self.encode_dense_qmv(
                encoder,
                &normed,
                &layer.k.weight,
                self_keys,
                kv_off,
                1,
                dim,
                dim,
            )?;
            self.encode_dense_qmv(
                encoder,
                &normed,
                &layer.v.weight,
                self_values,
                kv_off,
                1,
                dim,
                dim,
            )?;
            if let Some(bias) = layer.v.bias.as_ref() {
                self.encode_add_row_bias(encoder, self_values, kv_off, bias, 1, dim)?;
            }
            self.encode_whisper_attn_decode(
                encoder,
                &q,
                self_keys,
                self_values,
                &ctx,
                dec.heads,
                head_dim,
                self_len,
            )?;
            self.encode_dense_qmv(encoder, &ctx, &layer.o.weight, &proj, 0, 1, dim, dim)?;
            self.encode_proj_bias(encoder, &proj, layer.o.bias.as_ref(), 1, dim)?;
            // h += attn ; normed = cross_ln(h).
            self.encode_add_layer_norm_rows(
                encoder,
                &h,
                &proj,
                &layer.cross_ln.weight,
                &layer.cross_ln.bias,
                &h,
                &normed,
                1,
                dim,
                eps,
            )?;
            // Cross-attn sur le KV statique.
            self.encode_dense_qmv(
                encoder,
                &normed,
                &layer.cross_q.weight,
                &cross_q,
                0,
                1,
                dim,
                dim,
            )?;
            self.encode_proj_bias(encoder, &cross_q, layer.cross_q.bias.as_ref(), 1, dim)?;
            self.encode_whisper_attn_decode(
                encoder,
                &cross_q,
                &kv.cross_keys[i],
                &kv.cross_values[i],
                &ctx,
                dec.heads,
                head_dim,
                kv.cross_len,
            )?;
            self.encode_dense_qmv(encoder, &ctx, &layer.cross_o.weight, &proj, 0, 1, dim, dim)?;
            self.encode_proj_bias(encoder, &proj, layer.cross_o.bias.as_ref(), 1, dim)?;
            // h += cross ; normed = final_ln(h).
            self.encode_add_layer_norm_rows(
                encoder,
                &h,
                &proj,
                &layer.final_ln.weight,
                &layer.final_ln.bias,
                &h,
                &normed,
                1,
                dim,
                eps,
            )?;
            // FFN.
            self.encode_decode_ffn_qmv(encoder, &normed, &layer.fc1, &f1, ffn, dim)?;
            self.encode_proj_bias(encoder, &f1, layer.fc1.bias.as_ref(), 1, ffn)?;
            self.encode_gelu(encoder, &f1, &g, ffn)?;
            self.encode_decode_ffn_qmv(encoder, &g, &layer.fc2, &proj, dim, ffn)?;
            self.encode_proj_bias(encoder, &proj, layer.fc2.bias.as_ref(), 1, dim)?;
            // h += ff ; normed = self_ln de la couche suivante (sinon add nu : le
            // LayerNorm décodeur final est appliqué par l'appelant `decoder_head`).
            if i + 1 < n {
                self.encode_add_layer_norm_rows(
                    encoder,
                    &h,
                    &proj,
                    &dec.layers[i + 1].self_ln.weight,
                    &dec.layers[i + 1].self_ln.bias,
                    &h,
                    &normed,
                    1,
                    dim,
                    eps,
                )?;
            } else {
                self.encode_accumulate_scaled(encoder, &mut owned, &proj, &h, 1.0, dim)?;
            }
        }
        if let Some(next_index) = next_index.as_ref() {
            self.encode_layer_norm_rows(
                encoder,
                &h,
                &dec.final_ln.weight,
                &dec.final_ln.bias,
                &normed,
                1,
                dim,
                eps,
            )?;
            self.encode_lm_head_argmax_buffers(
                encoder,
                &mut owned,
                &normed,
                &dec.lm_head,
                next_index,
                dim,
            )?;
        }

        guard.end();
        commit_and_wait(command_buffer)?;
        kv.len += 1;
        match next_index {
            Some(next_index) => read_u32_buffer(&next_index, 1)?
                .into_iter()
                .next()
                .map(Some)
                .ok_or_else(|| InferError::Metal("decode résident: argmax sans token".to_string())),
            None => Ok(None),
        }
    }

    /// Prépare le KV decode Whisper directement sur GPU : self KV vide et cross
    /// KV projeté depuis `audio_features` sans readback hôte intermédiaire.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si les formes divergent ou si Metal échoue.
    pub(crate) fn build_whisper_decode_kv_resident(
        &self,
        audio_features: &Tensor,
        dec: &WhisperResidentDecoder,
        capacity: usize,
    ) -> Result<WhisperDecodeKv> {
        let (frames, dim) = audio_features.as_matrix()?;
        if dim != dec.d_model {
            return Err(InferError::Dimension(format!(
                "decode KV résident: audio dim={dim}, attendu {}",
                dec.d_model
            )));
        }
        let audio = self.upload_f32_buffer(audio_features.data(), "wd_audio_features")?;
        self.build_whisper_decode_kv_resident_from_buffer(&audio, frames, dim, dec, capacity)
    }

    fn build_whisper_decode_kv_resident_from_buffer(
        &self,
        audio: &BufferRef,
        frames: usize,
        dim: usize,
        dec: &WhisperResidentDecoder,
        capacity: usize,
    ) -> Result<WhisperDecodeKv> {
        if dim != dec.d_model {
            return Err(InferError::Dimension(format!(
                "decode KV résident: audio dim={dim}, attendu {}",
                dec.d_model
            )));
        }
        let kv_len = checked_len(frames, dim, "decode cross kv")?;
        let self_len = checked_len(capacity, dim, "decode self kv")?;
        let mut self_keys = Vec::with_capacity(dec.layers.len());
        let mut self_values = Vec::with_capacity(dec.layers.len());
        let mut cross_keys = Vec::with_capacity(dec.layers.len());
        let mut cross_values = Vec::with_capacity(dec.layers.len());

        for _ in &dec.layers {
            self_keys.push(self.alloc_resident_f32(self_len)?);
            self_values.push(self.alloc_resident_f32(self_len)?);
            cross_keys.push(self.alloc_resident_f32(kv_len)?);
            cross_values.push(self.alloc_resident_f32(kv_len)?);
        }
        let na = self.na_gemm_bf16.is_some()
            && super::whisper_bf16_gemm_enabled()
            && dec.layers.iter().all(|layer| {
                layer.cross_k.weight_na.is_some() && layer.cross_v.weight_na.is_some()
            });
        let audio_bf16 = self.new_bf16_buffer(kv_len, "wd_audio_bf16")?;

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let guard = EncoderEndGuard::new(encoder);
        if na {
            self.encode_f32_to_bf16(encoder, &audio, &audio_bf16, kv_len)?;
        }
        for (index, layer) in dec.layers.iter().enumerate() {
            self.encode_proj_gemm(
                encoder,
                na,
                &audio,
                &audio_bf16,
                &layer.cross_k,
                &cross_keys[index],
                frames,
                dim,
                dim,
            )?;
            self.encode_proj_bias(
                encoder,
                &cross_keys[index],
                layer.cross_k.bias.as_ref(),
                frames,
                dim,
            )?;
            self.encode_proj_gemm(
                encoder,
                na,
                &audio,
                &audio_bf16,
                &layer.cross_v,
                &cross_values[index],
                frames,
                dim,
                dim,
            )?;
            self.encode_proj_bias(
                encoder,
                &cross_values[index],
                layer.cross_v.bias.as_ref(),
                frames,
                dim,
            )?;
        }
        guard.end();
        commit_and_wait(command_buffer)?;

        Ok(WhisperDecodeKv {
            self_keys,
            self_values,
            cross_keys,
            cross_values,
            cross_len: frames,
            len: 0,
            capacity,
        })
    }

    /// Alloue un buffer résident `[len]` f32 (non mémoïsé) pour l'état KV décodeur.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si `len` déborde.
    pub(crate) fn alloc_resident_f32(&self, len: usize) -> Result<Buffer> {
        Ok(self
            .device
            .new_buffer(byte_len::<f32>(len)?, MTLResourceOptions::StorageModeShared))
    }
}

/// Poids GPU (mémoïsés) du frontend conv mel Whisper.
pub(crate) struct WhisperConvWeights {
    /// conv1 `[d_model, num_mel_bins·kernel]` (mémoire conv1_weight `[out,in,k]`).
    pub conv1_weight: Buffer,
    pub conv1_weight_na: Option<Buffer>,
    pub conv1_bias: Buffer,
    /// conv2 `[d_model, d_model·kernel]`.
    pub conv2_weight: Buffer,
    pub conv2_weight_na: Option<Buffer>,
    pub conv2_bias: Buffer,
    /// Table de positions encodeur `[≥out_frames2, d_model]`.
    pub positions: Buffer,
    pub num_mel_bins: usize,
    pub d_model: usize,
    pub kernel: usize,
}

/// Matrice f32 Whisper déjà matérialisée dans un buffer Metal.
pub(crate) struct WhisperResidentMatrix {
    pub buffer: Buffer,
    pub rows: usize,
    pub cols: usize,
}

/// Poids GPU (mémoïsés) d'une norme Whisper.
pub(crate) struct WhisperResidentNorm {
    pub weight: Buffer,
    pub bias: Buffer,
}

/// Poids GPU d'une projection dense Whisper (biais optionnel).
pub(crate) struct WhisperResidentProj {
    pub weight: Buffer,
    pub bias: Option<Buffer>,
    /// Poids transposé bf16 `rhs^T [K,N]` pour le GEMM NA (Some si `RETI_STT_BF16`).
    pub weight_na: Option<Buffer>,
    /// Poids bf16 row-major `[N,K]` pour le QMV decode M=1.
    pub weight_bf16: Option<Buffer>,
}

/// Poids GPU d'une couche encodeur Whisper.
pub(crate) struct WhisperResidentLayer {
    pub self_ln: WhisperResidentNorm,
    pub q: WhisperResidentProj,
    pub k: WhisperResidentProj,
    pub v: WhisperResidentProj,
    pub o: WhisperResidentProj,
    pub final_ln: WhisperResidentNorm,
    pub fc1: WhisperResidentProj,
    pub fc2: WhisperResidentProj,
}

/// Poids GPU de l'encodeur Whisper résident (préparés une fois par `whisper.rs`).
pub(crate) struct WhisperResidentEncoder {
    pub layers: Vec<WhisperResidentLayer>,
    pub encoder_ln: WhisperResidentNorm,
    pub d_model: usize,
    pub heads: usize,
    pub ffn_dim: usize,
    pub eps: f32,
}

/// Poids GPU d'une couche décodeur Whisper (self-attn + cross-attn + FFN).
pub(crate) struct WhisperDecodeLayer {
    pub self_ln: WhisperResidentNorm,
    pub q: WhisperResidentProj,
    pub k: WhisperResidentProj,
    pub v: WhisperResidentProj,
    pub o: WhisperResidentProj,
    pub cross_ln: WhisperResidentNorm,
    pub cross_q: WhisperResidentProj,
    pub cross_k: WhisperResidentProj,
    pub cross_v: WhisperResidentProj,
    pub cross_o: WhisperResidentProj,
    pub final_ln: WhisperResidentNorm,
    pub fc1: WhisperResidentProj,
    pub fc2: WhisperResidentProj,
}

/// Poids GPU du décodeur Whisper résident (préparés une fois par `whisper.rs`).
pub(crate) struct WhisperResidentDecoder {
    pub layers: Vec<WhisperDecodeLayer>,
    pub final_ln: WhisperResidentNorm,
    pub lm_head: MetalLinearWeightBuffers,
    pub d_model: usize,
    pub heads: usize,
    pub ffn_dim: usize,
    pub eps: f32,
}

/// État KV résident du décodeur (self append-only + cross statique), mutable
/// entre tokens. Buffers GPU mémoïsés, alloués une fois par énoncé.
pub(crate) struct WhisperDecodeKv {
    pub self_keys: Vec<Buffer>,
    pub self_values: Vec<Buffer>,
    pub cross_keys: Vec<Buffer>,
    pub cross_values: Vec<Buffer>,
    pub cross_len: usize,
    pub len: usize,
    pub capacity: usize,
}
