//! Conversions bf16, GEMM NAX et biais partagés par les modalités.

use super::*;
use std::ffi::c_void;
#[cfg(any(feature = "stt", test))]
use std::sync::PoisonError;

impl MetalExecutor {
    /// Encode la conversion f32 → bf16 `[n]` (+ barrière).
    pub(crate) fn encode_f32_to_bf16(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input: &BufferRef,
        out_bf16: &BufferRef,
        n: usize,
    ) -> Result<()> {
        self.encode_f32_to_bf16_at(encoder, input, out_bf16, 0, n)
    }

    /// Encode la conversion f32 → bf16 des `n` éléments à partir de
    /// l'élément `start` des deux buffers (+ barrière).
    pub(crate) fn encode_f32_to_bf16_at(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input: &BufferRef,
        out_bf16: &BufferRef,
        start: usize,
        n: usize,
    ) -> Result<()> {
        self.encode_f32_to_bf16_offsets(encoder, input, start, out_bf16, start, n)
    }

    /// Comme [`Self::encode_f32_to_bf16_at`], avec des éléments de départ
    /// distincts en entrée (f32) et en sortie (bf16).
    pub(crate) fn encode_f32_to_bf16_offsets(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input: &BufferRef,
        input_start: usize,
        out_bf16: &BufferRef,
        out_start: usize,
        n: usize,
    ) -> Result<()> {
        let n_u32 = checked_u32(n, "f32_to_bf16 n")?;
        record_prefill_f32_to_bf16_shape(n);
        encoder.set_compute_pipeline_state(&self.f32_to_bf16);
        encoder.set_buffer(0, Some(input), byte_len::<f32>(input_start)?);
        encoder.set_buffer(1, Some(out_bf16), byte_len::<u16>(out_start)?);
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
    /// Conservé comme façade standalone symétrique des encodeurs résidents,
    /// exercé par `tests/na_gemm_delta.rs`.
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
            0,
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
        encoder.set_bytes(3, 16, mnk.as_ptr().cast::<c_void>());
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
    #[cfg(feature = "stt")]
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
    #[cfg(any(feature = "stt", test))]
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
        // Le cache ne reçoit que des buffers entièrement construits, sans
        // mutation de leur contenu sous verrou : l'empoisonnement les laisse valides.
        if let Some(buf) = self
            .bf16_rhs_t_cache
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
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
            .unwrap_or_else(PoisonError::into_inner)
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
            crate::runtime_flags::env_var("SARAGOSSA_STT_NAX_BN128").as_deref(),
            Some("0" | "false" | "off" | "no")
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
            0,
        ];
        encoder.set_compute_pipeline_state(pso);
        encoder.set_buffer(0, Some(lhs_bf16), 0);
        encoder.set_buffer(1, Some(rhs_t_bf16), 0);
        encoder.set_buffer(2, Some(out), 0);
        encoder.set_bytes(3, 16, mnk.as_ptr().cast::<c_void>());
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

    #[cfg(any(feature = "stt", feature = "diar", feature = "tts"))]
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
}
