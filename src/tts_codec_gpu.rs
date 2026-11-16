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
    Buffer, CommandQueue, ComputeCommandEncoderRef, ComputePipelineState, Device,
    MTLResourceOptions, MTLSize,
};
use std::collections::HashMap;
use std::ffi::c_void;

mod prefix;
#[cfg(test)]
mod profile;
#[cfg(test)]
mod tests;
#[cfg(test)]
pub(crate) use profile::KernelTime;

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

// conv1d et conv transposée en tuiles (im2col implicite) : threadgroup de 128
// threads, tuile (16·TM) lignes × 32 canaux de sortie, TM×4 sorties par thread,
// entrées et poids par tranches de 16 canaux d'entrée en mémoire threadgroup.
// TM = 4 (64 lignes) par défaut, TM = 1 (16 lignes) quand la grille serait trop
// petite pour occuper le GPU.
// Chaque sortie garde l'accumulation du kernel gather (biais d'abord pour la conv,
// à la fin pour la transposée ; prises puis canaux dans l'ordre croissant) : une
// ligne hors bornes charge 0 au lieu d'être sautée, ce qui laisse l'accumulateur
// inchangé tant qu'il n'est pas un zéro signé.
// p = [out_time, out_dim, inner, ksize, dilation, pad, x_time, has_bias, out_start,
//      transpose, stride, taps, has_residual]
// Conv : ligne de sortie `row`, prise k lit x[row + out_start + k·dilation - pad].
// Transposée (phase r = grille z) : ligne q = out_start/stride + i, prise j lit
// x[q - (taps-1) + j] avec le poids k = r + (taps-1-j)·stride, sortie i·stride + r.
#define CT_BN 32u
#define CT_BK 16u
template <uint TM>
inline void codec_conv_tiled(
    device const float* x,
    device const float* w,
    device const float* bias,
    device float* out,
    constant uint* p,
    device const float* residual,
    uint3 tg,
    uint tid,
    threadgroup float* xs,
    threadgroup float* ws
) {
    constexpr uint BM = 16u * TM;
    constexpr uint XS = BM + 1u;
    constexpr uint WS = CT_BN + 1u;
    const uint out_time = p[0];
    const uint out_dim = p[1];
    const uint inner = p[2];
    const uint ksize = p[3];
    const uint dilation = p[4];
    const uint pad = p[5];
    const int x_time = int(p[6]);
    const bool has_bias = p[7] != 0u;
    const uint out_start = p[8];
    const bool transpose = p[9] != 0u;
    const uint stride = p[10];
    const uint taps = p[11];
    const bool has_residual = p[12] != 0u;
    const uint phase = tg.z;
    const uint oc0 = tg.x * CT_BN;
    const uint i0 = tg.y * BM;
    const uint tx = tid & 7u;
    const uint ty = tid >> 3u;
    float acc[TM][4];
    for (uint j = 0u; j < 4u; ++j) {
        const float b0 = (!transpose && has_bias) ? bias[oc0 + tx + 8u * j] : 0.0f;
        for (uint i = 0u; i < TM; ++i) { acc[i][j] = b0; }
    }
    const uint q0 = transpose ? out_start / stride : 0u;
    for (uint tap = 0u; tap < taps; ++tap) {
        const uint k = transpose ? phase + (taps - 1u - tap) * stride : tap;
        for (uint ic0 = 0u; ic0 < inner; ic0 += CT_BK) {
            for (uint e = 0u; e < 2u * TM; ++e) {
                const uint lin = tid + 128u * e;
                const uint r = lin / CT_BK;
                const uint c = lin % CT_BK;
                const int src = transpose
                    ? int(q0 + i0 + r) - int(taps - 1u) + int(tap)
                    : int(i0 + r + out_start + k * dilation) - int(pad);
                const bool ok = src >= 0 && src < x_time;
                xs[c * XS + r] = ok ? x[uint(src) * inner + ic0 + c] : 0.0f;
            }
            for (uint e = 0u; e < 4u; ++e) {
                const uint lin = tid + 128u * e;
                const uint n = lin / CT_BK;
                const uint c = lin % CT_BK;
                ws[c * WS + n] = w[((oc0 + n) * ksize + k) * inner + ic0 + c];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint kk = 0u; kk < CT_BK; ++kk) {
                float xv[TM];
                float wv[4];
                for (uint i = 0u; i < TM; ++i) { xv[i] = xs[kk * XS + ty + 16u * i]; }
                for (uint j = 0u; j < 4u; ++j) { wv[j] = ws[kk * WS + tx + 8u * j]; }
                for (uint i = 0u; i < TM; ++i) {
                    for (uint j = 0u; j < 4u; ++j) { acc[i][j] += xv[i] * wv[j]; }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    for (uint i = 0u; i < TM; ++i) {
        const uint li = i0 + ty + 16u * i;
        const uint row = transpose ? li * stride + phase : li;
        if (row >= out_time) { continue; }
        for (uint j = 0u; j < 4u; ++j) {
            const uint oc = oc0 + tx + 8u * j;
            float v = acc[i][j];
            if (transpose && has_bias) { v += bias[oc]; }
            if (has_residual) { v = residual[row * out_dim + oc] + v; }
            out[row * out_dim + oc] = v;
        }
    }
}

#define CODEC_CONV_TILED_KERNEL(NAME, TM)                                         \
kernel void NAME(                                                                 \
    device const float* x        [[buffer(0)]],                                   \
    device const float* w        [[buffer(1)]],                                   \
    device const float* bias     [[buffer(2)]],                                   \
    device float* out            [[buffer(3)]],                                   \
    constant uint* p             [[buffer(4)]],                                   \
    device const float* residual [[buffer(5)]],                                   \
    uint3 tg [[threadgroup_position_in_grid]],                                    \
    uint tid [[thread_index_in_threadgroup]]                                      \
) {                                                                               \
    threadgroup float xs[CT_BK * (16u * TM + 1u)];                                \
    threadgroup float ws[CT_BK * (CT_BN + 1u)];                                   \
    codec_conv_tiled<TM>(x, w, bias, out, p, residual, tg, tid, xs, ws);          \
}
CODEC_CONV_TILED_KERNEL(codec_conv_tiled_f32, 4u)
CODEC_CONV_TILED_KERNEL(codec_conv_tiled_small_f32, 1u)

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
/// Tuile de `codec_conv_tiled_f32` : lignes, canaux de sortie, canaux
/// d'entrée par tranche, threads (mêmes valeurs que `CT_BM`/`CT_BN`/`CT_BK`).
const TILE_M: u64 = 64;
const TILE_N: usize = 32;
const TILE_K: usize = 16;
const TILE_THREADS: u64 = 128;
/// Seuil de grille (threadgroups en tuiles de 64 lignes) sous lequel les tuiles
/// de 16 lignes prennent le relais : 2 par cœur sur un M5 Max à 40 cœurs.
const TILE_SMALL_BELOW: u64 = 80;

/// Métadonnées + buffer GPU d'une convolution (poids réorganisé une fois).
#[derive(Debug)]
struct GpuConv {
    wkc: Buffer,
    bias: Option<Buffer>,
    out_dim: usize,
    inner: usize,
    kernel: usize,
    /// Groupes de la convolution (1, ou `out_dim` en depthwise).
    groups: usize,
}

impl GpuConv {
    fn in_channels(&self) -> usize {
        self.inner * self.groups
    }
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
    tiled_pipeline: ComputePipelineState,
    tiled_small_pipeline: ComputePipelineState,
    /// Convolutions en tuiles (`SARAGOSSA_TTS_CODEC_CONV_TILED`).
    conv_tiled: bool,
    convs: HashMap<String, GpuConv>,
    transposes: HashMap<String, GpuConv>,
    snakes: HashMap<String, GpuSnake>,
    upsample_rates: Vec<usize>,
    /// Préfixe GPU (RVQ → upsampling), `None` : préfixe CPU.
    prefix: Option<prefix::GpuPrefix>,
}

/// État streaming du codec GPU : contextes causaux par couche conv/transpose.
#[derive(Debug, Default)]
pub(crate) struct CodecGpuStream {
    contexts: HashMap<String, GpuContext>,
    /// Cache K/V résident du `pre_transformer`, par couche (préfixe GPU).
    kv: Vec<Option<prefix::GpuKv>>,
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
    /// Bascule les convolutions en tuiles (oracles gather contre tuiles).
    #[cfg(test)]
    pub(crate) fn set_conv_tiled(&mut self, enabled: bool) {
        self.conv_tiled = enabled;
    }

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
        let upsample_rates = cfg
            .upsample_rates
            .iter()
            .map(|rate| usize::try_from(*rate))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|_| InferError::Config("upsample_rate codec négatif".to_string()))?;
        let mut gpu = Self::without_weights(upsample_rates)?;
        gpu.register_weights(weights)?;
        if crate::runtime_flags::tts_codec_gpu_prefix_enabled() {
            gpu.register_prefix(weights, cfg)?;
        }
        Ok(gpu)
    }

    /// Device, file et pipelines compilés, sans aucun poids enregistré.
    fn without_weights(upsample_rates: Vec<usize>) -> Result<Self> {
        let device = Device::system_default()
            .ok_or_else(|| InferError::Metal("aucun device Metal pour le codec".to_string()))?;
        let queue = device.new_command_queue();
        let conv_pipeline = compile(&device, "codec_conv1d_f32")?;
        let transpose_pipeline = compile(&device, "codec_transpose1d_f32")?;
        let snake_pipeline = compile(&device, "codec_snake_beta_f32")?;
        let add_pipeline = compile(&device, "codec_add_f32")?;
        let concat_pipeline = compile(&device, "codec_concat_rows_f32")?;
        let tail_pipeline = compile(&device, "codec_tail_rows_f32")?;
        let tiled_pipeline = compile(&device, "codec_conv_tiled_f32")?;
        let tiled_small_pipeline = compile(&device, "codec_conv_tiled_small_f32")?;
        Ok(Self {
            device,
            queue,
            conv_pipeline,
            transpose_pipeline,
            snake_pipeline,
            add_pipeline,
            concat_pipeline,
            tail_pipeline,
            tiled_pipeline,
            tiled_small_pipeline,
            conv_tiled: crate::runtime_flags::tts_codec_conv_tiled_enabled(),
            convs: HashMap::new(),
            transposes: HashMap::new(),
            snakes: HashMap::new(),
            upsample_rates,
            prefix: None,
        })
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
            groups: 1,
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
            groups: 1,
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
            None,
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
            None,
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
            None,
        )?;
        pool.release(h);
        let h3 = self.snake(encoder, pool, &h2, &format!("{prefix}.act2"))?;
        pool.release(h2);
        // conv2 + raccord résiduel en un dispatch (`x + conv2(...)`).
        let out = self.conv(
            encoder,
            pool,
            &h3,
            &format!("{prefix}.conv2.conv"),
            1,
            true,
            Some(&x),
        )?;
        pool.release(h3);
        pool.release(x);
        Ok(out)
    }

    /// conv1d causale (`nopad=false`, `pad=(k-1)·dilation`, `out_time=x_time`) ou
    /// valide (`nopad=true`, `pad=0`, `out_time=x_time-k+1`). groupes = 1.
    #[allow(
        clippy::too_many_arguments,
        reason = "primitive codec GPU chaude: forme et résiduel suivent le dispatch Metal"
    )]
    fn conv(
        &self,
        encoder: &ComputeCommandEncoderRef,
        pool: &mut BufPool,
        x: &GpuNlc,
        prefix: &str,
        dilation: usize,
        nopad: bool,
        residual: Option<&GpuNlc>,
    ) -> Result<GpuNlc> {
        self.conv_with_start(encoder, pool, x, prefix, dilation, nopad, 0, residual)
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
        // out_time = (x_time-1)·stride + kernel - (kernel - stride) = x_time·stride
        // (identique au trim causal de `causal_transpose1d`).
        let out_time = x.time.checked_mul(stride).ok_or_else(|| {
            InferError::Dimension(format!("transpose GPU {prefix} temps déborde"))
        })?;
        self.transpose_with_start(encoder, pool, x, prefix, stride, 0, out_time)
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
        let wav = self.encode_tail_stream(state, encoder, &mut pool, input)?;
        encoder.end_encoding();
        commit_and_wait(command_buffer)?;
        read_f32_buffer(&wav.buffer, wav.len())
    }

    /// Décode `time` trames depuis leurs sommes RVQ (`first` : premier
    /// quantificateur, `rest` : les suivants, `[time, codebook_dim]`) : préfixe
    /// et section `decoder.decoder` dans un command buffer. PCM non clampé.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le préfixe GPU est inactif, si une forme est
    /// incohérente ou si Metal échoue.
    pub(crate) fn decode_rvq_streaming(
        &self,
        state: &mut CodecGpuStream,
        time: usize,
        first: &[f32],
        rest: &[f32],
    ) -> Result<Vec<f32>> {
        if time == 0 {
            return Ok(Vec::new());
        }
        let options = MTLResourceOptions::StorageModeShared;
        let mut pool = BufPool::new(self.device.clone(), options);
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let _barrier_scope = install_dispatch_barrier_scope();
        let hidden = self.encode_prefix_stream(state, encoder, &mut pool, time, first, rest)?;
        let wav = self.encode_tail_stream(state, encoder, &mut pool, hidden)?;
        encoder.end_encoding();
        commit_and_wait(command_buffer)?;
        read_f32_buffer(&wav.buffer, wav.len())
    }

    /// Encode la section `decoder.decoder` streaming depuis `input` et renvoie
    /// le PCM mono résident.
    fn encode_tail_stream(
        &self,
        state: &mut CodecGpuStream,
        encoder: &ComputeCommandEncoderRef,
        pool: &mut BufPool,
        input: GpuNlc,
    ) -> Result<GpuNlc> {
        let mut cur = self.conv_stream(
            state,
            encoder,
            pool,
            &input,
            "decoder.decoder.0.conv",
            1,
            false,
            None,
        )?;
        pool.release(input);

        let blocks = self.upsample_rates.len();
        for block in 1..=blocks {
            let stride = self.upsample_rates[block - 1];
            let snaked = self.snake(
                encoder,
                pool,
                &cur,
                &format!("decoder.decoder.{block}.block.0"),
            )?;
            pool.release(cur);
            cur = self.transpose_stream(
                state,
                encoder,
                pool,
                &snaked,
                &format!("decoder.decoder.{block}.block.1.conv"),
                stride,
            )?;
            pool.release(snaked);
            for (slot, dilation) in [(2_usize, 1_usize), (3, 3), (4, 9)] {
                let prefix = format!("decoder.decoder.{block}.block.{slot}");
                cur = self.residual_unit_stream(state, encoder, pool, cur, &prefix, dilation)?;
            }
        }

        let tail_snake = blocks + 1;
        let tail_conv = blocks + 2;
        let snaked = self.snake(
            encoder,
            pool,
            &cur,
            &format!("decoder.decoder.{tail_snake}"),
        )?;
        pool.release(cur);
        let wav = self.conv_stream(
            state,
            encoder,
            pool,
            &snaked,
            &format!("decoder.decoder.{tail_conv}.conv"),
            1,
            false,
            None,
        )?;
        pool.release(snaked);

        if wav.channels != 1 {
            return Err(InferError::Dimension(format!(
                "codec GPU streaming wav attendu mono, reçu {} canaux",
                wav.channels
            )));
        }
        Ok(wav)
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
            None,
        )?;
        pool.release(h);
        let h3 = self.snake(encoder, pool, &h2, &format!("{prefix}.act2"))?;
        pool.release(h2);
        let out = self.conv_stream(
            state,
            encoder,
            pool,
            &h3,
            &format!("{prefix}.conv2.conv"),
            1,
            true,
            Some(&x),
        )?;
        pool.release(h3);
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
        residual: Option<&GpuNlc>,
    ) -> Result<GpuNlc> {
        let conv = self
            .convs
            .get(prefix)
            .ok_or_else(|| InferError::MissingWeight(format!("conv GPU {prefix}")))?;
        if x.channels != conv.in_channels() {
            return Err(InferError::Dimension(format!(
                "conv GPU streaming {prefix} attendu in={}, reçu {}",
                conv.in_channels(),
                x.channels
            )));
        }
        if nopad {
            if conv.kernel != 1 {
                return Err(InferError::Config(format!(
                    "codec streaming: conv valid non causale {prefix} kernel={}",
                    conv.kernel
                )));
            }
            return self.conv_with_start(encoder, pool, x, prefix, dilation, true, 0, residual);
        }

        let keep_rows = (conv.kernel - 1).checked_mul(dilation).ok_or_else(|| {
            InferError::Dimension(format!("codec streaming contexte déborde {prefix}"))
        })?;
        let key = format!("conv:{prefix}");
        let (local, context_rows, release_local) =
            self.context_input(encoder, pool, state, &key, x, keep_rows)?;
        self.copy_tail_context(encoder, state, &key, &local, keep_rows)?;
        let out = self.conv_with_start(
            encoder,
            pool,
            &local,
            prefix,
            dilation,
            false,
            context_rows,
            residual,
        )?;
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
        residual: Option<&GpuNlc>,
    ) -> Result<GpuNlc> {
        let conv = self
            .convs
            .get(prefix)
            .ok_or_else(|| InferError::MissingWeight(format!("conv GPU {prefix}")))?;
        if x.channels != conv.in_channels() {
            return Err(InferError::Dimension(format!(
                "conv GPU {prefix} attendu in={}, reçu {}",
                conv.in_channels(),
                x.channels
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
        if let Some(res) = residual {
            if res.time != out_time || res.channels != conv.out_dim {
                return Err(InferError::Dimension(format!(
                    "conv GPU {prefix} résiduel [{},{}] != [{out_time},{}]",
                    res.time, res.channels, conv.out_dim
                )));
            }
        }
        let out = pool.acquire(out_time, conv.out_dim)?;
        if self.tiled_eligible(conv) {
            let params = [
                u32_of(out_time)?,
                u32_of(conv.out_dim)?,
                u32_of(conv.inner)?,
                u32_of(conv.kernel)?,
                u32_of(dilation)?,
                u32_of(pad)?,
                u32_of(x.time)?,
                u32::from(conv.bias.is_some()),
                u32_of(out_start)?,
                0,
                1,
                u32_of(conv.kernel)?,
                u32::from(residual.is_some()),
            ];
            self.dispatch_tiled(encoder, x, conv, &out, residual, &params, out_time, 1);
            return Ok(out);
        }
        let params = [
            u32_of(out_time)?,
            u32_of(conv.out_dim)?,
            u32_of(conv.inner)?,
            u32_of(conv.kernel)?,
            u32_of(dilation)?,
            u32_of(pad)?,
            u32_of(conv.out_dim / conv.groups)?,
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
        match residual {
            Some(res) => {
                let sum = self.add(encoder, pool, res, &out)?;
                pool.release(out);
                Ok(sum)
            }
            None => Ok(out),
        }
    }

    /// Le kernel en tuiles couvre la conv : groupes = 1, canaux de sortie
    /// multiples de 32 et d'entrée multiples de 16.
    fn tiled_eligible(&self, conv: &GpuConv) -> bool {
        self.conv_tiled
            && conv.groups == 1
            && conv.out_dim % TILE_N == 0
            && conv.inner % TILE_K == 0
    }

    /// Moins de 2 threadgroups par cœur en tuiles de 64 lignes : tuiles de 16
    /// lignes, quatre fois plus nombreuses (même accumulation par sortie).
    fn tiled_small(&self, conv: &GpuConv, rows: usize, phases: usize) -> bool {
        let col_tiles = (conv.out_dim / TILE_N) as u64 * phases as u64;
        col_tiles * ceil_div(rows as u64, TILE_M) < TILE_SMALL_BELOW
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "dispatch Metal: buffers liés et grille de la tuile"
    )]
    fn dispatch_tiled(
        &self,
        encoder: &ComputeCommandEncoderRef,
        x: &GpuNlc,
        conv: &GpuConv,
        out: &GpuNlc,
        residual: Option<&GpuNlc>,
        params: &[u32; 13],
        rows: usize,
        phases: usize,
    ) {
        let (pipeline, tile_m) = if self.tiled_small(conv, rows, phases) {
            (&self.tiled_small_pipeline, TILE_M / 4)
        } else {
            (&self.tiled_pipeline, TILE_M)
        };
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&x.buffer), 0);
        encoder.set_buffer(1, Some(&conv.wkc), 0);
        encoder.set_buffer(2, conv.bias.as_deref(), 0);
        encoder.set_buffer(3, Some(&out.buffer), 0);
        set_params(encoder, 4, params);
        encoder.set_buffer(5, residual.map(|res| &*res.buffer), 0);
        let groups = MTLSize::new(
            (conv.out_dim / TILE_N) as u64,
            ceil_div(rows as u64, tile_m),
            phases as u64,
        );
        encoder.dispatch_thread_groups(groups, MTLSize::new(TILE_THREADS, 1, 1));
        post_dispatch_barrier(encoder);
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
        if self.tiled_eligible(conv)
            && conv.kernel % stride == 0
            && out_start % stride == 0
            && out_time % stride == 0
        {
            let params = [
                u32_of(out_time)?,
                u32_of(conv.out_dim)?,
                u32_of(conv.inner)?,
                u32_of(conv.kernel)?,
                1,
                0,
                u32_of(x.time)?,
                u32::from(conv.bias.is_some()),
                u32_of(out_start)?,
                1,
                u32_of(stride)?,
                u32_of(conv.kernel / stride)?,
                0,
            ];
            self.dispatch_tiled(
                encoder,
                x,
                conv,
                &out,
                None,
                &params,
                out_time / stride,
                stride,
            );
            return Ok(out);
        }
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
    let options = crate::metal_backend::compile_options(true);
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
    // Une seule colonne (conv finale mono) : threadgroups en colonne, sinon
    // 31 threads sur 32 sortent aussitôt. Même calcul par thread.
    let (tgx, tgy) = if dim_x == 1 { (1, 256) } else { TG_2D };
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
