//! Forward GPU résident du vocodeur codec TTS (boucle `decoder.decoder`).
//!
//! Le codec CPU (`tts_codec.rs`) est compute-bound : ~94 % du temps part dans
//! trois convolutions (`causal_conv1d`, `causal_transpose1d`, `conv1d_nopad`) aux
//! résolutions montant jusqu'à 1920×N. Ce module porte cette section sur GPU en
//! gardant les tenseurs **résidents** : un seul command buffer, zéro readback
//! intermédiaire (barrières mémoire entre dispatches dépendants), une unique
//! lecture finale du PCM. Les activations Snake-β et l'addition résiduelle sont
//! aussi sur GPU pour ne jamais repasser par le CPU dans la section chaude.
//!
//! Le reste (RVQ, `pre_transformer`, boucle d'upsampling/`convnext`) reste CPU :
//! il est marginal (~5 %) et éviterait sinon de porter LayerNorm/Linear/GELU.
//!
//! Bibliothèque Metal **séparée** (kernels ci-dessous, compilés indépendamment) :
//! aucun kernel partagé avec le LLM n'est touché. La numérique GPU diffère de la
//! réduction scalaire CPU au niveau de l'arrondi f32 (FMA / ordre) ; le gate est
//! une tolérance audio (cf. `tts_codec` tests), pas l'octet-à-octet.

#![cfg(all(target_os = "macos", feature = "metal"))]

use crate::metal_backend::{
    commit_and_wait, install_dispatch_barrier_scope, post_dispatch_barrier, read_f32_buffer,
};
use crate::tts::TtsDecoderConfig;
use crate::{InferError, Result, Tensor};
use metal::{
    Buffer, CommandQueue, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device,
    MTLResourceOptions, MTLSize,
};
use std::collections::HashMap;
use std::ffi::c_void;

const CODEC_KERNELS: &str = r#"
#include <metal_stdlib>
using namespace metal;

// conv1d gather (causal ET valid/nopad) : un thread par (out_ch, out_time).
// Poids réorganisé [out_dim, k, in_per_group] (mêmes valeurs que la version CPU).
// p = [out_time, out_dim, in_per_group, k, dilation, pad, out_per_group, channels, x_time, has_bias, out_start]
kernel void codec_conv1d_f32(
    device const float* x      [[buffer(0)]],
    device const float* w      [[buffer(1)]],
    device const float* bias   [[buffer(2)]],
    device float* out          [[buffer(3)]],
    constant uint* p           [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint out_time = p[0];
    uint out_dim = p[1];
    uint oc = gid.x;
    uint t = gid.y;
    if (oc >= out_dim || t >= out_time) { return; }
    uint in_per_group = p[2];
    uint ksize = p[3];
    uint dilation = p[4];
    uint pad = p[5];
    uint out_per_group = p[6];
    uint channels = p[7];
    uint x_time = p[8];
    uint has_bias = p[9];
    uint out_start = p[10];
    uint group = oc / out_per_group;
    uint in_start = group * in_per_group;
    float acc = (has_bias != 0u) ? bias[oc] : 0.0f;
    for (uint k = 0; k < ksize; ++k) {
        int src = int(t + out_start) + int(k * dilation) - int(pad);
        if (src < 0 || src >= int(x_time)) { continue; }
        uint xbase = uint(src) * channels + in_start;
        uint wbase = (oc * ksize + k) * in_per_group;
        for (uint ic = 0; ic < in_per_group; ++ic) {
            acc += x[xbase + ic] * w[wbase + ic];
        }
    }
    out[t * out_dim + oc] = acc;
}

// conv transposée 1d gather : un thread par (out_ch, out_time).
// Poids réorganisé [out_dim, k, in_dim]. p = [out_time, out_dim, in_dim, k, stride, x_time, has_bias, out_start]
kernel void codec_transpose1d_f32(
    device const float* x      [[buffer(0)]],
    device const float* w      [[buffer(1)]],
    device const float* bias   [[buffer(2)]],
    device float* out          [[buffer(3)]],
    constant uint* p           [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint out_time = p[0];
    uint out_dim = p[1];
    uint oc = gid.x;
    uint out_row = gid.y;
    if (oc >= out_dim || out_row >= out_time) { return; }
    uint pp = out_row + p[7];
    uint in_dim = p[2];
    uint ksize = p[3];
    uint stride = p[4];
    uint x_time = p[5];
    uint has_bias = p[6];
    uint t_max = min(pp / stride, x_time - 1u);
    uint t_min = (pp >= ksize) ? ((pp - ksize) / stride + 1u) : 0u;
    float acc = 0.0f;
    for (uint t = t_min; t <= t_max; ++t) {
        uint k = pp - t * stride;
        uint xbase = t * in_dim;
        uint wbase = (oc * ksize + k) * in_dim;
        for (uint ic = 0; ic < in_dim; ++ic) {
            acc += x[xbase + ic] * w[wbase + ic];
        }
    }
    if (has_bias != 0u) { acc += bias[oc]; }
    out[out_row * out_dim + oc] = acc;
}

// Snake-β : un thread par élément. a_exp/b_exp = exp(alpha)/exp(beta) par canal,
// précalculés sur CPU (identiques à la version CPU). p = [total, channels]
kernel void codec_snake_beta_f32(
    device const float* x      [[buffer(0)]],
    device const float* a_exp  [[buffer(1)]],
    device const float* b_exp  [[buffer(2)]],
    device float* out          [[buffer(3)]],
    constant uint* p           [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = p[0];
    if (gid >= total) { return; }
    uint channels = p[1];
    uint ch = gid % channels;
    float v = x[gid];
    float s = sin(v * a_exp[ch]);
    out[gid] = v + (s * s) / (b_exp[ch] + 1.0e-9f);
}

// Addition élémentaire (raccord résiduel). p = [total]
kernel void codec_add_f32(
    device const float* a      [[buffer(0)]],
    device const float* b      [[buffer(1)]],
    device float* out          [[buffer(2)]],
    constant uint* p           [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = p[0];
    if (gid >= total) { return; }
    out[gid] = a[gid] + b[gid];
}

// Concatène un contexte `[context_time, channels]` et le nouveau bloc `[time, channels]`.
// p = [total, context_total]
kernel void codec_concat_rows_f32(
    device const float* context [[buffer(0)]],
    device const float* x       [[buffer(1)]],
    device float* out           [[buffer(2)]],
    constant uint* p            [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = p[0];
    if (gid >= total) { return; }
    uint context_total = p[1];
    if (gid < context_total) {
        out[gid] = context[gid];
    } else {
        out[gid] = x[gid - context_total];
    }
}

// Copie les dernières lignes d'un tenseur `[time, channels]` vers un contexte.
// p = [total, channels, start_row]
kernel void codec_tail_rows_f32(
    device const float* x [[buffer(0)]],
    device float* out     [[buffer(1)]],
    constant uint* p      [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = p[0];
    if (gid >= total) { return; }
    uint channels = p[1];
    uint start_row = p[2];
    uint row = gid / channels;
    uint ch = gid - row * channels;
    out[gid] = x[(start_row + row) * channels + ch];
}
"#;

const TG_2D: (u64, u64) = (32, 8);
const TG_1D: u64 = 256;

/// Métadonnées + buffer GPU d'une convolution (poids réorganisé une fois).
#[derive(Debug)]
struct GpuConv {
    wkc: Buffer,
    bias: Option<Buffer>,
    out_dim: usize,
    inner: usize,
    kernel: usize,
}

/// Buffers GPU d'une activation Snake-β (`exp(alpha)`/`exp(beta)` par canal).
#[derive(Debug)]
struct GpuSnake {
    a_exp: Buffer,
    b_exp: Buffer,
    channels: usize,
}

/// Forward GPU résident du vocodeur (section `decoder.decoder` + tail).
#[derive(Debug)]
pub(crate) struct CodecGpu {
    device: Device,
    queue: CommandQueue,
    conv_pipeline: ComputePipelineState,
    transpose_pipeline: ComputePipelineState,
    snake_pipeline: ComputePipelineState,
    add_pipeline: ComputePipelineState,
    concat_pipeline: ComputePipelineState,
    tail_pipeline: ComputePipelineState,
    convs: HashMap<String, GpuConv>,
    transposes: HashMap<String, GpuConv>,
    snakes: HashMap<String, GpuSnake>,
    upsample_rates: Vec<usize>,
}

/// État streaming du codec GPU : contextes causaux par couche conv/transpose.
#[derive(Debug, Default)]
pub(crate) struct CodecGpuStream {
    contexts: HashMap<String, GpuContext>,
}

#[derive(Debug)]
struct GpuContext {
    buffer: Buffer,
    time: usize,
    channels: usize,
}

/// Tenseur GPU `[time, channels]` (row-major), éventuellement issu du pool.
struct GpuNlc {
    buffer: Buffer,
    pool_index: Option<usize>,
    time: usize,
    channels: usize,
}

impl GpuNlc {
    fn len(&self) -> usize {
        self.time * self.channels
    }
}

/// Emplacement réutilisable du pool d'intermédiaires.
struct PoolSlot {
    buffer: Buffer,
    len: usize,
    busy: bool,
}

/// Pool de buffers intermédiaires (réemploi par longueur exacte, anti-aliasing
/// par drapeau `busy`). Toutes les passes étant sérialisées par des barrières
/// mémoire, réutiliser un slot libéré est sûr : la dernière lecture a été encodée
/// avant la nouvelle écriture.
struct BufPool {
    device: Device,
    options: MTLResourceOptions,
    slots: Vec<PoolSlot>,
}

impl BufPool {
    fn new(device: Device, options: MTLResourceOptions) -> Self {
        Self {
            device,
            options,
            slots: Vec::new(),
        }
    }

    fn acquire(&mut self, time: usize, channels: usize) -> Result<GpuNlc> {
        let len = time
            .checked_mul(channels)
            .filter(|n| *n > 0)
            .ok_or_else(|| InferError::Metal("buffer codec GPU de taille nulle".to_string()))?;
        let free = self.slots.iter().position(|s| !s.busy && s.len == len);
        let index = match free {
            Some(index) => index,
            None => {
                let bytes = u64::try_from(len * std::mem::size_of::<f32>())
                    .map_err(|_| InferError::Metal("taille buffer codec hors u64".to_string()))?;
                self.slots.push(PoolSlot {
                    buffer: self.device.new_buffer(bytes, self.options),
                    len,
                    busy: false,
                });
                self.slots.len() - 1
            }
        };
        self.slots[index].busy = true;
        Ok(GpuNlc {
            buffer: self.slots[index].buffer.clone(),
            pool_index: Some(index),
            time,
            channels,
        })
    }

    fn release(&mut self, tensor: GpuNlc) {
        if let Some(index) = tensor.pool_index {
            if let Some(slot) = self.slots.get_mut(index) {
                slot.busy = false;
            }
        }
    }
}

impl CodecGpu {
    /// Crée un état vide pour décoder un flux codec par suffixes.
    pub(crate) fn new_stream(&self) -> CodecGpuStream {
        CodecGpuStream::default()
    }

    /// Construit le forward GPU : compile les kernels et téléverse une fois tous
    /// les poids de la section résidente (`decoder.decoder.*`).
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si Metal est indisponible, si la compilation échoue ou
    /// si un poids attendu est absent / mal formé.
    pub(crate) fn new(weights: &HashMap<String, Tensor>, cfg: &TtsDecoderConfig) -> Result<Self> {
        let device = Device::system_default()
            .ok_or_else(|| InferError::Metal("aucun device Metal pour le codec".to_string()))?;
        let queue = device.new_command_queue();
        let conv_pipeline = compile(&device, "codec_conv1d_f32")?;
        let transpose_pipeline = compile(&device, "codec_transpose1d_f32")?;
        let snake_pipeline = compile(&device, "codec_snake_beta_f32")?;
        let add_pipeline = compile(&device, "codec_add_f32")?;
        let concat_pipeline = compile(&device, "codec_concat_rows_f32")?;
        let tail_pipeline = compile(&device, "codec_tail_rows_f32")?;

        let upsample_rates = cfg
            .upsample_rates
            .iter()
            .map(|rate| usize::try_from(*rate))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|_| InferError::Config("upsample_rate codec négatif".to_string()))?;

        let mut gpu = Self {
            device,
            queue,
            conv_pipeline,
            transpose_pipeline,
            snake_pipeline,
            add_pipeline,
            concat_pipeline,
            tail_pipeline,
            convs: HashMap::new(),
            transposes: HashMap::new(),
            snakes: HashMap::new(),
            upsample_rates,
        };
        gpu.register_weights(weights)?;
        Ok(gpu)
    }

    /// Enregistre (réorganise + téléverse) les poids de toute la section résidente.
    fn register_weights(&mut self, weights: &HashMap<String, Tensor>) -> Result<()> {
        self.register_conv(weights, "decoder.decoder.0.conv")?;
        let blocks = self.upsample_rates.len();
        for block in 1..=blocks {
            self.register_snake(weights, &format!("decoder.decoder.{block}.block.0"))?;
            self.register_transpose(weights, &format!("decoder.decoder.{block}.block.1.conv"))?;
            for slot in [2_usize, 3, 4] {
                let prefix = format!("decoder.decoder.{block}.block.{slot}");
                self.register_snake(weights, &format!("{prefix}.act1"))?;
                self.register_conv(weights, &format!("{prefix}.conv1.conv"))?;
                self.register_snake(weights, &format!("{prefix}.act2"))?;
                self.register_conv(weights, &format!("{prefix}.conv2.conv"))?;
            }
        }
        let tail_snake = blocks + 1;
        let tail_conv = blocks + 2;
        self.register_snake(weights, &format!("decoder.decoder.{tail_snake}"))?;
        self.register_conv(weights, &format!("decoder.decoder.{tail_conv}.conv"))?;
        Ok(())
    }

    fn register_conv(&mut self, weights: &HashMap<String, Tensor>, prefix: &str) -> Result<()> {
        let weight = weight_of(weights, &format!("{prefix}.weight"))?;
        let [out_dim, inner, kernel] = conv_shape(weight, prefix)?;
        // Réorganisation [out, in, k] -> [out, k, in] (identique à `causal_conv1d`).
        let mut wkc = vec![0.0_f32; out_dim * inner * kernel];
        let wdata = weight.data();
        for oc in 0..out_dim {
            for kk in 0..kernel {
                for ic in 0..inner {
                    wkc[(oc * kernel + kk) * inner + ic] = wdata[(oc * inner + ic) * kernel + kk];
                }
            }
        }
        let bias = self.upload_bias(weights, prefix)?;
        let conv = GpuConv {
            wkc: self.upload(&wkc)?,
            bias,
            out_dim,
            inner,
            kernel,
        };
        self.convs.insert(prefix.to_string(), conv);
        Ok(())
    }

    fn register_transpose(
        &mut self,
        weights: &HashMap<String, Tensor>,
        prefix: &str,
    ) -> Result<()> {
        let weight = weight_of(weights, &format!("{prefix}.weight"))?;
        let [in_dim, out_dim, kernel] = conv_shape(weight, prefix)?;
        // Réorganisation [in, out, k] -> [out, k, in] (identique à `causal_transpose1d`).
        let mut wkc = vec![0.0_f32; out_dim * in_dim * kernel];
        let wdata = weight.data();
        for oc in 0..out_dim {
            for kk in 0..kernel {
                for ic in 0..in_dim {
                    wkc[(oc * kernel + kk) * in_dim + ic] =
                        wdata[(ic * out_dim + oc) * kernel + kk];
                }
            }
        }
        let bias = self.upload_bias(weights, prefix)?;
        let conv = GpuConv {
            wkc: self.upload(&wkc)?,
            bias,
            out_dim,
            inner: in_dim,
            kernel,
        };
        self.transposes.insert(prefix.to_string(), conv);
        Ok(())
    }

    fn register_snake(&mut self, weights: &HashMap<String, Tensor>, prefix: &str) -> Result<()> {
        let alpha = channel_vec(weight_of(weights, &format!("{prefix}.alpha"))?)?;
        let beta = channel_vec(weight_of(weights, &format!("{prefix}.beta"))?)?;
        if alpha.len() != beta.len() {
            return Err(InferError::Dimension(format!(
                "snake {prefix}: alpha={} beta={}",
                alpha.len(),
                beta.len()
            )));
        }
        let a_exp: Vec<f32> = alpha.iter().map(|v| v.exp()).collect();
        let b_exp: Vec<f32> = beta.iter().map(|v| v.exp()).collect();
        let snake = GpuSnake {
            channels: a_exp.len(),
            a_exp: self.upload(&a_exp)?,
            b_exp: self.upload(&b_exp)?,
        };
        self.snakes.insert(prefix.to_string(), snake);
        Ok(())
    }

    fn upload_bias(
        &self,
        weights: &HashMap<String, Tensor>,
        prefix: &str,
    ) -> Result<Option<Buffer>> {
        match weights.get(&format!("{prefix}.bias")) {
            Some(bias) => Ok(Some(self.upload(channel_slice(bias)?)?)),
            None => Ok(None),
        }
    }

    fn upload(&self, data: &[f32]) -> Result<Buffer> {
        if data.is_empty() {
            return Err(InferError::Metal("upload codec GPU vide".to_string()));
        }
        let bytes = u64::try_from(std::mem::size_of_val(data))
            .map_err(|_| InferError::Metal("upload codec hors u64".to_string()))?;
        Ok(self.device.new_buffer_with_data(
            data.as_ptr().cast::<c_void>(),
            bytes,
            MTLResourceOptions::StorageModeShared,
        ))
    }

    /// Décode la section résidente (de `decoder.decoder.0` au tail) sur GPU et
    /// renvoie le PCM mono **non clampé** (le clamp `[-1,1]` reste côté CPU, comme
    /// `decode_codes`). `hidden` est la sortie CPU de la boucle d'upsampling.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si une dimension est incompatible ou si Metal échoue.
    pub(crate) fn decode_tail(
        &self,
        hidden_time: usize,
        hidden_channels: usize,
        hidden: &[f32],
    ) -> Result<Vec<f32>> {
        if hidden.len() != hidden_time * hidden_channels {
            return Err(InferError::Dimension(format!(
                "codec GPU hidden [{hidden_time},{hidden_channels}] != {} valeurs",
                hidden.len()
            )));
        }
        let options = MTLResourceOptions::StorageModeShared;
        let mut pool = BufPool::new(self.device.clone(), options);
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        // Barrière mémoire après chaque dispatch : la chaîne est séquentielle, les
        // intermédiaires restent résidents, aucune lecture CPU avant la fin.
        let _barrier_scope = install_dispatch_barrier_scope();

        let input_bytes = u64::try_from(std::mem::size_of_val(hidden))
            .map_err(|_| InferError::Metal("hidden codec hors u64".to_string()))?;
        let input = GpuNlc {
            buffer: self.device.new_buffer_with_data(
                hidden.as_ptr().cast::<c_void>(),
                input_bytes,
                options,
            ),
            pool_index: None,
            time: hidden_time,
            channels: hidden_channels,
        };

        // decoder.decoder.0 : conv causale 1×1 (dilation 1).
        let mut cur = self.conv(
            encoder,
            &mut pool,
            &input,
            "decoder.decoder.0.conv",
            1,
            false,
        )?;
        pool.release(input);

        let blocks = self.upsample_rates.len();
        for block in 1..=blocks {
            let stride = self.upsample_rates[block - 1];
            let snaked = self.snake(
                encoder,
                &mut pool,
                &cur,
                &format!("decoder.decoder.{block}.block.0"),
            )?;
            pool.release(cur);
            cur = self.transpose(
                encoder,
                &mut pool,
                &snaked,
                &format!("decoder.decoder.{block}.block.1.conv"),
                stride,
            )?;
            pool.release(snaked);
            for (slot, dilation) in [(2_usize, 1_usize), (3, 3), (4, 9)] {
                let prefix = format!("decoder.decoder.{block}.block.{slot}");
                cur = self.residual_unit(encoder, &mut pool, cur, &prefix, dilation)?;
            }
        }

        let tail_snake = blocks + 1;
        let tail_conv = blocks + 2;
        let snaked = self.snake(
            encoder,
            &mut pool,
            &cur,
            &format!("decoder.decoder.{tail_snake}"),
        )?;
        pool.release(cur);
        let wav = self.conv(
            encoder,
            &mut pool,
            &snaked,
            &format!("decoder.decoder.{tail_conv}.conv"),
            1,
            false,
        )?;
        pool.release(snaked);

        if wav.channels != 1 {
            return Err(InferError::Dimension(format!(
                "codec GPU wav attendu mono, reçu {} canaux",
                wav.channels
            )));
        }
        encoder.end_encoding();
        commit_and_wait(command_buffer)?;
        read_f32_buffer(&wav.buffer, wav.len())
    }

    /// Unité résiduelle : `out = x + conv2(snake(conv1(snake(x))))`.
    fn residual_unit(
        &self,
        encoder: &ComputeCommandEncoderRef,
        pool: &mut BufPool,
        x: GpuNlc,
        prefix: &str,
        dilation: usize,
    ) -> Result<GpuNlc> {
        let h = self.snake(encoder, pool, &x, &format!("{prefix}.act1"))?;
        let h2 = self.conv(
            encoder,
            pool,
            &h,
            &format!("{prefix}.conv1.conv"),
            dilation,
            false,
        )?;
        pool.release(h);
        let h3 = self.snake(encoder, pool, &h2, &format!("{prefix}.act2"))?;
        pool.release(h2);
        let h4 = self.conv(encoder, pool, &h3, &format!("{prefix}.conv2.conv"), 1, true)?;
        pool.release(h3);
        let out = self.add(encoder, pool, &x, &h4)?;
        pool.release(h4);
        pool.release(x);
        Ok(out)
    }

    /// conv1d causale (`nopad=false`, `pad=(k-1)·dilation`, `out_time=x_time`) ou
    /// valide (`nopad=true`, `pad=0`, `out_time=x_time-k+1`). groupes = 1.
    fn conv(
        &self,
        encoder: &ComputeCommandEncoderRef,
        pool: &mut BufPool,
        x: &GpuNlc,
        prefix: &str,
        dilation: usize,
        nopad: bool,
    ) -> Result<GpuNlc> {
        let conv = self
            .convs
            .get(prefix)
            .ok_or_else(|| InferError::MissingWeight(format!("conv GPU {prefix}")))?;
        if x.channels != conv.inner {
            return Err(InferError::Dimension(format!(
                "conv GPU {prefix} attendu in={}, reçu {}",
                conv.inner, x.channels
            )));
        }
        let (pad, out_time) = if nopad {
            if conv.kernel > x.time {
                return Err(InferError::Dimension(format!(
                    "conv valid GPU {prefix} kernel={} > time={}",
                    conv.kernel, x.time
                )));
            }
            (0_usize, x.time - conv.kernel + 1)
        } else {
            ((conv.kernel - 1) * dilation, x.time)
        };
        let out = pool.acquire(out_time, conv.out_dim)?;
        let params = [
            u32_of(out_time)?,
            u32_of(conv.out_dim)?,
            u32_of(conv.inner)?,
            u32_of(conv.kernel)?,
            u32_of(dilation)?,
            u32_of(pad)?,
            u32_of(conv.out_dim)?, // out_per_group (groupes = 1)
            u32_of(x.channels)?,
            u32_of(x.time)?,
            u32::from(conv.bias.is_some()),
            0,
        ];
        encoder.set_compute_pipeline_state(&self.conv_pipeline);
        encoder.set_buffer(0, Some(&x.buffer), 0);
        encoder.set_buffer(1, Some(&conv.wkc), 0);
        encoder.set_buffer(2, conv.bias.as_deref(), 0);
        encoder.set_buffer(3, Some(&out.buffer), 0);
        set_params(encoder, 4, &params);
        dispatch_2d(encoder, conv.out_dim, out_time);
        post_dispatch_barrier(encoder);
        Ok(out)
    }

    /// conv transposée 1d causale (groupes = 1), `out_time = x_time·stride`.
    fn transpose(
        &self,
        encoder: &ComputeCommandEncoderRef,
        pool: &mut BufPool,
        x: &GpuNlc,
        prefix: &str,
        stride: usize,
    ) -> Result<GpuNlc> {
        let conv = self
            .transposes
            .get(prefix)
            .ok_or_else(|| InferError::MissingWeight(format!("transpose GPU {prefix}")))?;
        if x.channels != conv.inner {
            return Err(InferError::Dimension(format!(
                "transpose GPU {prefix} attendu in={}, reçu {}",
                conv.inner, x.channels
            )));
        }
        if stride == 0 || conv.kernel < stride {
            return Err(InferError::Dimension(format!(
                "transpose GPU {prefix} stride={stride} kernel={}",
                conv.kernel
            )));
        }
        // out_time = (x_time-1)·stride + kernel - (kernel - stride) = x_time·stride
        // (identique au trim causal de `causal_transpose1d`).
        let raw_time = (x.time - 1) * stride + conv.kernel;
        let out_time = raw_time - (conv.kernel - stride);
        let out = pool.acquire(out_time, conv.out_dim)?;
        let params = [
            u32_of(out_time)?,
            u32_of(conv.out_dim)?,
            u32_of(conv.inner)?,
            u32_of(conv.kernel)?,
            u32_of(stride)?,
            u32_of(x.time)?,
            u32::from(conv.bias.is_some()),
            0,
        ];
        encoder.set_compute_pipeline_state(&self.transpose_pipeline);
        encoder.set_buffer(0, Some(&x.buffer), 0);
        encoder.set_buffer(1, Some(&conv.wkc), 0);
        encoder.set_buffer(2, conv.bias.as_deref(), 0);
        encoder.set_buffer(3, Some(&out.buffer), 0);
        set_params(encoder, 4, &params);
        dispatch_2d(encoder, conv.out_dim, out_time);
        post_dispatch_barrier(encoder);
        Ok(out)
    }

    fn snake(
        &self,
        encoder: &ComputeCommandEncoderRef,
        pool: &mut BufPool,
        x: &GpuNlc,
        prefix: &str,
    ) -> Result<GpuNlc> {
        let snake = self
            .snakes
            .get(prefix)
            .ok_or_else(|| InferError::MissingWeight(format!("snake GPU {prefix}")))?;
        if x.channels != snake.channels {
            return Err(InferError::Dimension(format!(
                "snake GPU {prefix} canaux={} reçu {}",
                snake.channels, x.channels
            )));
        }
        let out = pool.acquire(x.time, x.channels)?;
        let params = [u32_of(out.len())?, u32_of(x.channels)?];
        encoder.set_compute_pipeline_state(&self.snake_pipeline);
        encoder.set_buffer(0, Some(&x.buffer), 0);
        encoder.set_buffer(1, Some(&snake.a_exp), 0);
        encoder.set_buffer(2, Some(&snake.b_exp), 0);
        encoder.set_buffer(3, Some(&out.buffer), 0);
        set_params(encoder, 4, &params);
        dispatch_1d(encoder, out.len());
        post_dispatch_barrier(encoder);
        Ok(out)
    }

    fn add(
        &self,
        encoder: &ComputeCommandEncoderRef,
        pool: &mut BufPool,
        a: &GpuNlc,
        b: &GpuNlc,
    ) -> Result<GpuNlc> {
        if a.time != b.time || a.channels != b.channels {
            return Err(InferError::Dimension(format!(
                "add GPU [{},{}] vs [{},{}]",
                a.time, a.channels, b.time, b.channels
            )));
        }
        let out = pool.acquire(a.time, a.channels)?;
        let params = [u32_of(out.len())?];
        encoder.set_compute_pipeline_state(&self.add_pipeline);
        encoder.set_buffer(0, Some(&a.buffer), 0);
        encoder.set_buffer(1, Some(&b.buffer), 0);
        encoder.set_buffer(2, Some(&out.buffer), 0);
        set_params(encoder, 3, &params);
        dispatch_1d(encoder, out.len());
        post_dispatch_barrier(encoder);
        Ok(out)
    }

    /// Décode un suffixe de `hidden` en réutilisant les contextes causaux.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si une dimension est incompatible ou si Metal échoue.
    pub(crate) fn decode_tail_streaming(
        &self,
        state: &mut CodecGpuStream,
        hidden_time: usize,
        hidden_channels: usize,
        hidden: &[f32],
    ) -> Result<Vec<f32>> {
        if hidden_time == 0 {
            return Ok(Vec::new());
        }
        if hidden.len() != hidden_time * hidden_channels {
            return Err(InferError::Dimension(format!(
                "codec GPU streaming hidden [{hidden_time},{hidden_channels}] != {} valeurs",
                hidden.len()
            )));
        }
        let options = MTLResourceOptions::StorageModeShared;
        let mut pool = BufPool::new(self.device.clone(), options);
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let _barrier_scope = install_dispatch_barrier_scope();

        let input_bytes = u64::try_from(std::mem::size_of_val(hidden))
            .map_err(|_| InferError::Metal("hidden codec streaming hors u64".to_string()))?;
        let input = GpuNlc {
            buffer: self.device.new_buffer_with_data(
                hidden.as_ptr().cast::<c_void>(),
                input_bytes,
                options,
            ),
            pool_index: None,
            time: hidden_time,
            channels: hidden_channels,
        };

        let mut cur = self.conv_stream(
            state,
            encoder,
            &mut pool,
            &input,
            "decoder.decoder.0.conv",
            1,
            false,
        )?;
        pool.release(input);

        let blocks = self.upsample_rates.len();
        for block in 1..=blocks {
            let stride = self.upsample_rates[block - 1];
            let snaked = self.snake(
                encoder,
                &mut pool,
                &cur,
                &format!("decoder.decoder.{block}.block.0"),
            )?;
            pool.release(cur);
            cur = self.transpose_stream(
                state,
                encoder,
                &mut pool,
                &snaked,
                &format!("decoder.decoder.{block}.block.1.conv"),
                stride,
            )?;
            pool.release(snaked);
            for (slot, dilation) in [(2_usize, 1_usize), (3, 3), (4, 9)] {
                let prefix = format!("decoder.decoder.{block}.block.{slot}");
                cur =
                    self.residual_unit_stream(state, encoder, &mut pool, cur, &prefix, dilation)?;
            }
        }

        let tail_snake = blocks + 1;
        let tail_conv = blocks + 2;
        let snaked = self.snake(
            encoder,
            &mut pool,
            &cur,
            &format!("decoder.decoder.{tail_snake}"),
        )?;
        pool.release(cur);
        let wav = self.conv_stream(
            state,
            encoder,
            &mut pool,
            &snaked,
            &format!("decoder.decoder.{tail_conv}.conv"),
            1,
            false,
        )?;
        pool.release(snaked);

        if wav.channels != 1 {
            return Err(InferError::Dimension(format!(
                "codec GPU streaming wav attendu mono, reçu {} canaux",
                wav.channels
            )));
        }
        encoder.end_encoding();
        commit_and_wait(command_buffer)?;
        read_f32_buffer(&wav.buffer, wav.len())
    }

    fn residual_unit_stream(
        &self,
        state: &mut CodecGpuStream,
        encoder: &ComputeCommandEncoderRef,
        pool: &mut BufPool,
        x: GpuNlc,
        prefix: &str,
        dilation: usize,
    ) -> Result<GpuNlc> {
        let h = self.snake(encoder, pool, &x, &format!("{prefix}.act1"))?;
        let h2 = self.conv_stream(
            state,
            encoder,
            pool,
            &h,
            &format!("{prefix}.conv1.conv"),
            dilation,
            false,
        )?;
        pool.release(h);
        let h3 = self.snake(encoder, pool, &h2, &format!("{prefix}.act2"))?;
        pool.release(h2);
        let h4 = self.conv_stream(
            state,
            encoder,
            pool,
            &h3,
            &format!("{prefix}.conv2.conv"),
            1,
            true,
        )?;
        pool.release(h3);
        let out = self.add(encoder, pool, &x, &h4)?;
        pool.release(h4);
        pool.release(x);
        Ok(out)
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "chemin codec GPU chaud: état, buffers et paramètres de convolution restent explicites"
    )]
    fn conv_stream(
        &self,
        state: &mut CodecGpuStream,
        encoder: &ComputeCommandEncoderRef,
        pool: &mut BufPool,
        x: &GpuNlc,
        prefix: &str,
        dilation: usize,
        nopad: bool,
    ) -> Result<GpuNlc> {
        let conv = self
            .convs
            .get(prefix)
            .ok_or_else(|| InferError::MissingWeight(format!("conv GPU {prefix}")))?;
        if x.channels != conv.inner {
            return Err(InferError::Dimension(format!(
                "conv GPU streaming {prefix} attendu in={}, reçu {}",
                conv.inner, x.channels
            )));
        }
        if nopad {
            if conv.kernel != 1 {
                return Err(InferError::Config(format!(
                    "codec streaming: conv valid non causale {prefix} kernel={}",
                    conv.kernel
                )));
            }
            return self.conv(encoder, pool, x, prefix, dilation, true);
        }

        let keep_rows = (conv.kernel - 1).checked_mul(dilation).ok_or_else(|| {
            InferError::Dimension(format!("codec streaming contexte déborde {prefix}"))
        })?;
        let key = format!("conv:{prefix}");
        let (local, context_rows, release_local) =
            self.context_input(encoder, pool, state, &key, x, keep_rows)?;
        self.copy_tail_context(encoder, state, &key, &local, keep_rows)?;
        let out =
            self.conv_with_start(encoder, pool, &local, prefix, dilation, false, context_rows)?;
        if release_local {
            pool.release(local);
        }
        Ok(out)
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "primitive codec GPU chaude: offset et paramètres de convolution suivent le dispatch Metal"
    )]
    fn conv_with_start(
        &self,
        encoder: &ComputeCommandEncoderRef,
        pool: &mut BufPool,
        x: &GpuNlc,
        prefix: &str,
        dilation: usize,
        nopad: bool,
        out_start: usize,
    ) -> Result<GpuNlc> {
        let conv = self
            .convs
            .get(prefix)
            .ok_or_else(|| InferError::MissingWeight(format!("conv GPU {prefix}")))?;
        if x.channels != conv.inner {
            return Err(InferError::Dimension(format!(
                "conv GPU {prefix} attendu in={}, reçu {}",
                conv.inner, x.channels
            )));
        }
        let (pad, out_time) = if nopad {
            if conv.kernel > x.time {
                return Err(InferError::Dimension(format!(
                    "conv valid GPU {prefix} kernel={} > time={}",
                    conv.kernel, x.time
                )));
            }
            (0_usize, x.time - conv.kernel + 1)
        } else {
            ((conv.kernel - 1) * dilation, x.time - out_start)
        };
        let out = pool.acquire(out_time, conv.out_dim)?;
        let params = [
            u32_of(out_time)?,
            u32_of(conv.out_dim)?,
            u32_of(conv.inner)?,
            u32_of(conv.kernel)?,
            u32_of(dilation)?,
            u32_of(pad)?,
            u32_of(conv.out_dim)?,
            u32_of(x.channels)?,
            u32_of(x.time)?,
            u32::from(conv.bias.is_some()),
            u32_of(out_start)?,
        ];
        encoder.set_compute_pipeline_state(&self.conv_pipeline);
        encoder.set_buffer(0, Some(&x.buffer), 0);
        encoder.set_buffer(1, Some(&conv.wkc), 0);
        encoder.set_buffer(2, conv.bias.as_deref(), 0);
        encoder.set_buffer(3, Some(&out.buffer), 0);
        set_params(encoder, 4, &params);
        dispatch_2d(encoder, conv.out_dim, out_time);
        post_dispatch_barrier(encoder);
        Ok(out)
    }

    fn transpose_stream(
        &self,
        state: &mut CodecGpuStream,
        encoder: &ComputeCommandEncoderRef,
        pool: &mut BufPool,
        x: &GpuNlc,
        prefix: &str,
        stride: usize,
    ) -> Result<GpuNlc> {
        let conv = self
            .transposes
            .get(prefix)
            .ok_or_else(|| InferError::MissingWeight(format!("transpose GPU {prefix}")))?;
        if x.channels != conv.inner {
            return Err(InferError::Dimension(format!(
                "transpose GPU streaming {prefix} attendu in={}, reçu {}",
                conv.inner, x.channels
            )));
        }
        let keep_rows = conv.kernel.saturating_sub(1);
        let key = format!("transpose:{prefix}");
        let (local, context_rows, release_local) =
            self.context_input(encoder, pool, state, &key, x, keep_rows)?;
        self.copy_tail_context(encoder, state, &key, &local, keep_rows)?;
        let out = self.transpose_with_start(
            encoder,
            pool,
            &local,
            prefix,
            stride,
            context_rows.checked_mul(stride).ok_or_else(|| {
                InferError::Dimension(format!("codec streaming transpose offset déborde {prefix}"))
            })?,
            x.time.checked_mul(stride).ok_or_else(|| {
                InferError::Dimension(format!("codec streaming transpose temps déborde {prefix}"))
            })?,
        )?;
        if release_local {
            pool.release(local);
        }
        Ok(out)
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "primitive codec GPU chaude: fenêtre de sortie et stride suivent le dispatch Metal"
    )]
    fn transpose_with_start(
        &self,
        encoder: &ComputeCommandEncoderRef,
        pool: &mut BufPool,
        x: &GpuNlc,
        prefix: &str,
        stride: usize,
        out_start: usize,
        out_time: usize,
    ) -> Result<GpuNlc> {
        let conv = self
            .transposes
            .get(prefix)
            .ok_or_else(|| InferError::MissingWeight(format!("transpose GPU {prefix}")))?;
        if x.channels != conv.inner {
            return Err(InferError::Dimension(format!(
                "transpose GPU {prefix} attendu in={}, reçu {}",
                conv.inner, x.channels
            )));
        }
        if stride == 0 || conv.kernel < stride {
            return Err(InferError::Dimension(format!(
                "transpose GPU {prefix} stride={stride} kernel={}",
                conv.kernel
            )));
        }
        let out = pool.acquire(out_time, conv.out_dim)?;
        let params = [
            u32_of(out_time)?,
            u32_of(conv.out_dim)?,
            u32_of(conv.inner)?,
            u32_of(conv.kernel)?,
            u32_of(stride)?,
            u32_of(x.time)?,
            u32::from(conv.bias.is_some()),
            u32_of(out_start)?,
        ];
        encoder.set_compute_pipeline_state(&self.transpose_pipeline);
        encoder.set_buffer(0, Some(&x.buffer), 0);
        encoder.set_buffer(1, Some(&conv.wkc), 0);
        encoder.set_buffer(2, conv.bias.as_deref(), 0);
        encoder.set_buffer(3, Some(&out.buffer), 0);
        set_params(encoder, 4, &params);
        dispatch_2d(encoder, conv.out_dim, out_time);
        post_dispatch_barrier(encoder);
        Ok(out)
    }

    fn context_input(
        &self,
        encoder: &ComputeCommandEncoderRef,
        pool: &mut BufPool,
        state: &CodecGpuStream,
        key: &str,
        x: &GpuNlc,
        keep_rows: usize,
    ) -> Result<(GpuNlc, usize, bool)> {
        let Some(context) = state.contexts.get(key) else {
            let borrowed = GpuNlc {
                buffer: x.buffer.clone(),
                pool_index: None,
                time: x.time,
                channels: x.channels,
            };
            return Ok((borrowed, 0, false));
        };
        if context.channels != x.channels {
            return Err(InferError::Dimension(format!(
                "codec streaming contexte {key} canaux={} reçu {}",
                context.channels, x.channels
            )));
        }
        let context_time = context.time.min(keep_rows);
        let total_time = context_time.checked_add(x.time).ok_or_else(|| {
            InferError::Dimension(format!("codec streaming contexte déborde {key}"))
        })?;
        let out = pool.acquire(total_time, x.channels)?;
        let params = [u32_of(out.len())?, u32_of(context_time * x.channels)?];
        encoder.set_compute_pipeline_state(&self.concat_pipeline);
        encoder.set_buffer(0, Some(&context.buffer), 0);
        encoder.set_buffer(1, Some(&x.buffer), 0);
        encoder.set_buffer(2, Some(&out.buffer), 0);
        set_params(encoder, 3, &params);
        dispatch_1d(encoder, out.len());
        post_dispatch_barrier(encoder);
        Ok((out, context_time, true))
    }

    fn copy_tail_context(
        &self,
        encoder: &ComputeCommandEncoderRef,
        state: &mut CodecGpuStream,
        key: &str,
        source: &GpuNlc,
        keep_rows: usize,
    ) -> Result<()> {
        if keep_rows == 0 {
            state.contexts.remove(key);
            return Ok(());
        }
        let keep_time = keep_rows.min(source.time);
        if keep_time == 0 {
            state.contexts.remove(key);
            return Ok(());
        }
        let len = keep_time.checked_mul(source.channels).ok_or_else(|| {
            InferError::Dimension(format!("codec streaming contexte taille déborde {key}"))
        })?;
        let bytes = u64::try_from(len * std::mem::size_of::<f32>())
            .map_err(|_| InferError::Metal("contexte codec hors u64".to_string()))?;
        let buffer = self
            .device
            .new_buffer(bytes, MTLResourceOptions::StorageModeShared);
        let params = [
            u32_of(len)?,
            u32_of(source.channels)?,
            u32_of(source.time - keep_time)?,
        ];
        encoder.set_compute_pipeline_state(&self.tail_pipeline);
        encoder.set_buffer(0, Some(&source.buffer), 0);
        encoder.set_buffer(1, Some(&buffer), 0);
        set_params(encoder, 2, &params);
        dispatch_1d(encoder, len);
        post_dispatch_barrier(encoder);
        state.contexts.insert(
            key.to_string(),
            GpuContext {
                buffer,
                time: keep_time,
                channels: source.channels,
            },
        );
        Ok(())
    }
}

fn compile(device: &Device, name: &str) -> Result<ComputePipelineState> {
    let options = CompileOptions::new();
    options.set_fast_math_enabled(true);
    let library = device
        .new_library_with_source(CODEC_KERNELS, &options)
        .map_err(|error| InferError::Metal(format!("compilation codec {name}: {error}")))?;
    let function = library
        .get_function(name, None)
        .map_err(|error| InferError::Metal(format!("fonction codec {name}: {error}")))?;
    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|error| InferError::Metal(format!("pipeline codec {name}: {error}")))
}

fn set_params(encoder: &ComputeCommandEncoderRef, index: u64, params: &[u32]) {
    let bytes = std::mem::size_of_val(params) as u64;
    encoder.set_bytes(index, bytes, params.as_ptr().cast::<c_void>());
}

fn dispatch_2d(encoder: &ComputeCommandEncoderRef, dim_x: usize, dim_y: usize) {
    let (tgx, tgy) = TG_2D;
    let groups = MTLSize::new(ceil_div(dim_x as u64, tgx), ceil_div(dim_y as u64, tgy), 1);
    encoder.dispatch_thread_groups(groups, MTLSize::new(tgx, tgy, 1));
}

fn dispatch_1d(encoder: &ComputeCommandEncoderRef, total: usize) {
    let groups = MTLSize::new(ceil_div(total as u64, TG_1D), 1, 1);
    encoder.dispatch_thread_groups(groups, MTLSize::new(TG_1D, 1, 1));
}

fn ceil_div(value: u64, divisor: u64) -> u64 {
    value.div_ceil(divisor).max(1)
}

fn u32_of(value: usize) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| InferError::Metal(format!("dimension codec hors u32: {value}")))
}

fn weight_of<'a>(weights: &'a HashMap<String, Tensor>, key: &str) -> Result<&'a Tensor> {
    weights
        .get(key)
        .ok_or_else(|| InferError::MissingWeight(key.to_string()))
}

fn conv_shape(weight: &Tensor, name: &str) -> Result<[usize; 3]> {
    match weight.shape() {
        [a, b, c] => Ok([*a, *b, *c]),
        shape => Err(InferError::Dimension(format!(
            "poids conv codec {name} attendu rang 3, reçu {shape:?}"
        ))),
    }
}

fn channel_slice(tensor: &Tensor) -> Result<&[f32]> {
    match tensor.shape() {
        [n] | [1, n] => Ok(&tensor.data()[..*n]),
        shape => Err(InferError::Dimension(format!(
            "paramètre canal codec incompatible shape={shape:?}"
        ))),
    }
}

fn channel_vec(tensor: &Tensor) -> Result<Vec<f32>> {
    Ok(channel_slice(tensor)?.to_vec())
}
