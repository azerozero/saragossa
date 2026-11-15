//! Dispatch des poids à micro-échelles, sans matérialisation dense des poids.

use super::*;
use crate::{MicroMode, MicroQuantizedTensor};
use metal::CompileOptions;

#[derive(Clone, Debug)]
pub(crate) struct MicroWeightBuffers {
    pub packed: Buffer,
    pub scales: Buffer,
    pub out_dim: usize,
    pub in_dim: usize,
    pub mode: MicroMode,
}

impl MetalExecutor {
    pub(super) fn micro_pipeline(&self, name: &str) -> Result<ComputePipelineState> {
        let mut cache = self
            .qmvt_pipelines
            .lock()
            .map_err(|_| InferError::Metal("cache micro empoisonné".into()))?;
        if let Some(pipeline) = cache.get(name) {
            return Ok(pipeline.clone());
        }
        let mut library = self
            .micro_library
            .lock()
            .map_err(|_| InferError::Metal("bibliothèque micro empoisonnée".into()))?;
        if library.is_none() {
            let source = format!(
                "{}\n{}",
                include_str!("na_gemm.metal"),
                include_str!("micro.metal")
            );
            let options = CompileOptions::new();
            options.set_fast_math_enabled(false);
            *library = Some(
                self.device
                    .new_library_with_source(&source, &options)
                    .map_err(InferError::Metal)?,
            );
        }
        let library = library
            .as_ref()
            .ok_or_else(|| InferError::Metal("bibliothèque micro absente".into()))?;
        let pipeline = super::pipelines::pipeline(library, &self.device, name)?;
        cache.insert(name.to_owned(), pipeline.clone());
        Ok(pipeline)
    }

    pub(super) fn resolve_micro(
        &self,
        weight: &MicroQuantizedTensor,
    ) -> Result<MicroWeightBuffers> {
        // Les échelles U8 restent compactes ; le cache réutilise le stockage
        // immuable des poids entre les tokens.
        let scales = self.cached_buffer(
            MetalBufferSource::Micro {
                weight_id: weight.weight_id(),
                scales: true,
            },
            weight.scales().len(),
            MetalBufferElement::U8,
            "micro_scales",
            || self.buffer_from_slice(weight.scales(), "micro_scales"),
        )?;
        Ok(MicroWeightBuffers {
            packed: self.cached_buffer(
                MetalBufferSource::Micro {
                    weight_id: weight.weight_id(),
                    scales: false,
                },
                weight.packed().len(),
                MetalBufferElement::U32,
                "micro_packed",
                || self.buffer_from_u32(weight.packed(), "micro_packed"),
            )?,
            scales,
            out_dim: weight.shape()[0],
            in_dim: weight.shape()[1],
            mode: weight.mode(),
        })
    }

    /// Produit Metal avec codes micro-échelles conservés compacts.
    ///
    /// # Errors
    /// Refuse les dimensions incompatibles ou une indisponibilité Metal/NAX.
    pub fn matmul_rhs_t_micro(
        &self,
        input: &Tensor,
        weight: &MicroQuantizedTensor,
    ) -> Result<Tensor> {
        let (batch, in_dim) = input.as_matrix()?;
        let buffers = self.resolve_micro(weight)?;
        let lhs = self.upload_f32_buffer(input.data(), "micro_input")?;
        let len = checked_len(batch, buffers.out_dim, "micro output")?;
        let output = self.new_f32_buffer(len, "micro_output")?;
        let cb = self.queue.new_command_buffer();
        let encoder = cb.new_compute_command_encoder();
        let guard = EncoderEndGuard::new(encoder);
        self.encode_micro(encoder, &lhs, &output, batch, in_dim, &buffers)?;
        guard.end();
        commit_and_wait(cb)?;
        Tensor::from_vec(vec![batch, buffers.out_dim], read_f32_buffer(&output, len)?)
    }

    pub(super) fn encode_micro(
        &self,
        encoder: &ComputeCommandEncoderRef,
        lhs: &BufferRef,
        output: &BufferRef,
        batch: usize,
        in_dim: usize,
        weight: &MicroWeightBuffers,
    ) -> Result<usize> {
        if in_dim != weight.in_dim || batch == 0 {
            return Err(InferError::Dimension("matmul micro incompatible".into()));
        }
        let use_gemm = batch >= 16 && in_dim % 64 == 0;
        let name = format!(
            "micro_{}_{}",
            if use_gemm { "gemm" } else { "qmv" },
            weight.mode.name()
        );
        let pipeline = self.micro_pipeline(&name)?;
        let bf16;
        let input = if use_gemm {
            bf16 =
                self.new_bf16_buffer(checked_len(batch, in_dim, "micro lhs")?, "micro_lhs_bf16")?;
            self.encode_f32_to_bf16(encoder, lhs, &bf16, batch * in_dim)?;
            &bf16
        } else {
            lhs
        };
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(&weight.packed), 0);
        encoder.set_buffer(2, Some(&weight.scales), 0);
        encoder.set_buffer(4, Some(output), 0);
        set_u32_bytes(
            encoder,
            5,
            &[
                checked_u32(batch, "micro M")?,
                checked_u32(weight.out_dim, "micro N")?,
                checked_u32(in_dim, "micro K")?,
            ],
            "micro_dims",
        )?;
        profile_dispatch();
        if use_gemm {
            encoder.dispatch_thread_groups(
                MTLSize::new(
                    batch.div_ceil(64) as u64,
                    weight.out_dim.div_ceil(64) as u64,
                    1,
                ),
                MTLSize::new(128, 1, 1),
            );
        } else {
            encoder.dispatch_thread_groups(
                MTLSize::new(weight.out_dim.div_ceil(4) as u64, batch as u64, 1),
                MTLSize::new(128, 1, 1),
            );
        }
        post_dispatch_barrier(encoder);
        Ok(weight.out_dim)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "gather résident avec offsets, formes et échelle explicites"
    )]
    pub(super) fn encode_micro_embedding(
        &self,
        encoder: &ComputeCommandEncoderRef,
        weight: &MicroWeightBuffers,
        indices: &BufferRef,
        offset: u64,
        output: &BufferRef,
        dim: usize,
        tokens: usize,
        scale: f32,
        recast: bool,
    ) -> Result<()> {
        if dim != weight.in_dim {
            return Err(InferError::Dimension("embedding micro incompatible".into()));
        }
        let pipeline = self.micro_pipeline(&format!("micro_embed_{}", weight.mode.name()))?;
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&weight.packed), 0);
        encoder.set_buffer(1, Some(&weight.scales), 0);
        encoder.set_buffer(2, Some(indices), offset);
        encoder.set_buffer(3, Some(output), 0);
        set_u32_bytes(
            encoder,
            4,
            &[
                checked_u32(dim, "micro embedding dim")?,
                checked_u32(weight.out_dim, "micro vocab")?,
            ],
            "micro_embed_dims",
        )?;
        set_f32_bytes(encoder, 5, &[scale], "micro_embed_scale")?;
        set_u32_bytes(encoder, 6, &[u32::from(recast)], "micro_embed_recast")?;
        profile_dispatch();
        encoder.dispatch_threads(
            MTLSize::new(dim as u64, tokens as u64, 1),
            MTLSize::new(256, 1, 1),
        );
        post_dispatch_barrier_buffer(encoder, output);
        Ok(())
    }
}
