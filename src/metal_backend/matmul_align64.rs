//! Encodage des qmv affines gs64 à entrée alignée sur 64 valeurs.

use super::*;

#[expect(
    clippy::too_many_arguments,
    reason = "dispatch Metal: buffers et dimensions quantifiées restent explicites"
)]
impl MetalExecutor {
    /// Encode un qmv u4/u8 gs64 dont l'entrée est alignée sur 64 valeurs.
    #[inline]
    pub(super) fn encode_affine_qmv_align64_buffers(
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
        bits: usize,
    ) -> Result<()> {
        let (pipeline, profile_label, kernel_name) = match bits {
            FAST_QMV_BITS => (
                &self.affine_qmv_fast_u4_gs64_align64_f32,
                "affine_qmv_u4_gs64_align64",
                "affine_qmv_fast_u4_gs64_align64_f32",
            ),
            8 => (
                &self.affine_qmv_fast_u8_gs64_align64_f32,
                "affine_qmv_u8_gs64_align64",
                "affine_qmv_fast_u8_gs64_align64_f32",
            ),
            _ => {
                return Err(InferError::Config(format!(
                    "qmv align64 attend des poids u4 ou u8, reçu u{bits}"
                )));
            }
        };
        let fast_dims = [
            checked_u32(out_dim, "qmv align64 out_dim")?,
            checked_u32(in_dim, "qmv align64 in_dim")?,
            checked_u32(packed_cols, "qmv align64 packed_cols")?,
            checked_u32(groups, "qmv align64 groups")?,
        ];
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(lhs_buffer), 0);
        encoder.set_buffer(1, Some(packed_buffer), 0);
        encoder.set_buffer(2, Some(scales_buffer), 0);
        encoder.set_buffer(3, Some(biases_buffer), 0);
        encoder.set_buffer(4, Some(output_buffer), 0);
        set_u32_bytes(encoder, 5, &fast_dims, "qmv_align64_dims")?;
        profile_dispatch_shape(DispatchProfileShape::matmul(
            profile_label,
            batch,
            in_dim,
            out_dim,
            FAST_QMV_GROUP_SIZE,
            bits,
        ));
        trace_dispatch_path(kernel_name, batch, out_dim, in_dim);
        profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(
                checked_nsuint(batch, "qmv align64 batch")?,
                checked_nsuint(out_dim.div_ceil(8), "qmv align64 out groups")?,
                1,
            ),
            MTLSize::new(64, 1, 1),
        );
        post_dispatch_barrier_buffer(encoder, output_buffer);
        Ok(())
    }
}
