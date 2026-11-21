//! Profil par kernel de la section `decoder.decoder` streaming (tests ignorés).
//!
//! Chaque opération passe dans son propre command buffer pour lire sa durée
//! GPU ; la chaîne et l'état streaming sont ceux de `encode_tail_stream`.

use super::{BufPool, CodecGpu, CodecGpuStream, GpuNlc};
use crate::metal_backend::{
    command_buffer_gpu_ns, commit_and_wait, install_dispatch_barrier_scope,
};
use crate::{InferError, Result};
use metal::{ComputeCommandEncoderRef, MTLResourceOptions};
use std::ffi::c_void;

/// Durée GPU, travail et trafic d'une opération du profil.
#[derive(Debug, Clone)]
pub(crate) struct KernelTime {
    pub(crate) label: String,
    pub(crate) flop: f64,
    pub(crate) bytes: f64,
    pub(crate) ns: f64,
}

const F32: f64 = 4.0;

impl CodecGpu {
    fn timed<T>(&self, f: impl FnOnce(&ComputeCommandEncoderRef) -> Result<T>) -> Result<(T, f64)> {
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let _barrier_scope = install_dispatch_barrier_scope();
        let out = f(encoder)?;
        encoder.end_encoding();
        commit_and_wait(command_buffer)?;
        Ok((out, command_buffer_gpu_ns(command_buffer) as f64))
    }

    /// Durée GPU de la section streaming en un seul command buffer.
    pub(crate) fn time_tail_stream(
        &self,
        state: &mut CodecGpuStream,
        hidden_time: usize,
        hidden_channels: usize,
        hidden: &[f32],
    ) -> Result<f64> {
        let mut pool = BufPool::new(self.device.clone(), MTLResourceOptions::StorageModeShared);
        let input = self.upload_nlc(hidden_time, hidden_channels, hidden)?;
        let (_, ns) =
            self.timed(|encoder| self.encode_tail_stream(state, encoder, &mut pool, input))?;
        Ok(ns)
    }

    fn upload_nlc(&self, time: usize, channels: usize, data: &[f32]) -> Result<GpuNlc> {
        if data.len() != time * channels || data.is_empty() {
            return Err(InferError::Dimension(
                "profil codec: entrée invalide".to_string(),
            ));
        }
        Ok(GpuNlc {
            buffer: self.device.new_buffer_with_data(
                data.as_ptr().cast::<c_void>(),
                std::mem::size_of_val(data) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            pool_index: None,
            time,
            channels,
        })
    }

    /// Même chaîne que `encode_tail_stream`, une opération par command buffer.
    pub(crate) fn profile_tail_stream(
        &self,
        state: &mut CodecGpuStream,
        hidden_time: usize,
        hidden_channels: usize,
        hidden: &[f32],
    ) -> Result<Vec<KernelTime>> {
        let mut pool = BufPool::new(self.device.clone(), MTLResourceOptions::StorageModeShared);
        let mut rec = Vec::new();
        let input = self.upload_nlc(hidden_time, hidden_channels, hidden)?;
        let mut cur = self.profile_conv(
            &mut rec,
            state,
            &mut pool,
            &input,
            "decoder.decoder.0.conv",
            1,
            false,
            None,
            "dec0",
        )?;
        pool.release(input);
        let blocks = self.upsample_rates.len();
        for block in 1..=blocks {
            let stride = *self
                .upsample_rates
                .get(block - 1)
                .ok_or_else(|| InferError::Config(format!("profil: bloc {block} absent")))?;
            let snaked = self.profile_snake(
                &mut rec,
                &mut pool,
                &cur,
                &format!("decoder.decoder.{block}.block.0"),
                &format!("b{block} snake"),
            )?;
            pool.release(cur);
            let name = format!("decoder.decoder.{block}.block.1.conv");
            let conv = self
                .transposes
                .get(&name)
                .ok_or_else(|| InferError::MissingWeight(format!("profil transpose {name}")))?;
            let (out, ns) = self.timed(|encoder| {
                self.transpose_stream(state, encoder, &mut pool, &snaked, &name, stride)
            })?;
            let taps = conv.kernel.div_ceil(stride);
            rec.push(KernelTime {
                label: format!(
                    "b{block} transpose k{} s{stride} {}→{}",
                    conv.kernel, conv.inner, conv.out_dim
                ),
                flop: 2.0 * (out.time * out.channels * conv.inner * taps) as f64,
                bytes: F32
                    * (snaked.len() + conv.out_dim * conv.inner * conv.kernel + out.len()) as f64,
                ns,
            });
            pool.release(snaked);
            cur = out;
            for (slot, dilation) in [(2_usize, 1_usize), (3, 3), (4, 9)] {
                let prefix = format!("decoder.decoder.{block}.block.{slot}");
                let h = self.profile_snake(
                    &mut rec,
                    &mut pool,
                    &cur,
                    &format!("{prefix}.act1"),
                    &format!("b{block} snake"),
                )?;
                let h2 = self.profile_conv(
                    &mut rec,
                    state,
                    &mut pool,
                    &h,
                    &format!("{prefix}.conv1.conv"),
                    dilation,
                    false,
                    None,
                    &format!("b{block} conv1 d{dilation}"),
                )?;
                pool.release(h);
                let h3 = self.profile_snake(
                    &mut rec,
                    &mut pool,
                    &h2,
                    &format!("{prefix}.act2"),
                    &format!("b{block} snake"),
                )?;
                pool.release(h2);
                let out = self.profile_conv(
                    &mut rec,
                    state,
                    &mut pool,
                    &h3,
                    &format!("{prefix}.conv2.conv"),
                    1,
                    true,
                    Some(&cur),
                    &format!("b{block} conv2+res"),
                )?;
                pool.release(h3);
                pool.release(cur);
                cur = out;
            }
        }
        let snaked = self.profile_snake(
            &mut rec,
            &mut pool,
            &cur,
            &format!("decoder.decoder.{}", blocks + 1),
            "tail snake",
        )?;
        pool.release(cur);
        let wav = self.profile_conv(
            &mut rec,
            state,
            &mut pool,
            &snaked,
            &format!("decoder.decoder.{}.conv", blocks + 2),
            1,
            false,
            None,
            "tail conv",
        )?;
        pool.release(snaked);
        pool.release(wav);
        Ok(rec)
    }

    fn profile_snake(
        &self,
        rec: &mut Vec<KernelTime>,
        pool: &mut BufPool,
        x: &GpuNlc,
        name: &str,
        label: &str,
    ) -> Result<GpuNlc> {
        let (out, ns) = self.timed(|encoder| self.snake(encoder, pool, x, name))?;
        rec.push(KernelTime {
            label: format!("{label} c{}", x.channels),
            flop: 0.0,
            bytes: F32 * 2.0 * x.len() as f64,
            ns,
        });
        Ok(out)
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "profil de test: état, pool et forme de la conv"
    )]
    fn profile_conv(
        &self,
        rec: &mut Vec<KernelTime>,
        state: &mut CodecGpuStream,
        pool: &mut BufPool,
        x: &GpuNlc,
        name: &str,
        dilation: usize,
        nopad: bool,
        residual: Option<&GpuNlc>,
        label: &str,
    ) -> Result<GpuNlc> {
        let conv = self
            .convs
            .get(name)
            .ok_or_else(|| InferError::MissingWeight(format!("profil conv {name}")))?;
        let (out, ns) = self.timed(|encoder| {
            self.conv_stream(state, encoder, pool, x, name, dilation, nopad, residual)
        })?;
        rec.push(KernelTime {
            label: format!("{label} k{} {}→{}", conv.kernel, conv.inner, conv.out_dim),
            flop: 2.0 * (out.time * out.channels * conv.inner * conv.kernel) as f64,
            bytes: F32 * (x.len() + conv.out_dim * conv.inner * conv.kernel + out.len()) as f64,
            ns,
        });
        Ok(out)
    }
}
