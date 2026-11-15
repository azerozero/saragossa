//! Préfixe du codec TTS sur GPU (`SARAGOSSA_TTS_CODEC_GPU_PREFIX`).
//!
//! Le CPU ne garde que la somme des embeddings RVQ (16 lignes de 512 par
//! trame). Les projections RVQ, `pre_conv`, les 8 couches du
//! `pre_transformer` (KV résident par couche) et les deux blocs d'upsampling
//! (transposée, ConvNeXt) sont encodés dans le command buffer de la section
//! `decoder.decoder` : un seul aller-retour par lot de trames.
//!
//! Numérique : chaque sortie dépend des mêmes entrées, dans le même ordre,
//! quel que soit le découpage en lots ; le PCM streaming reste donc identique
//! au PCM d'un seul appel. Face au préfixe CPU, seuls diffèrent les arrondis
//! des réductions (normes, softmax, produits scalaires) : l'écart est mesuré
//! par `codec_gpu_prefix_matches_cpu_prefix`.

use super::{
    channel_slice, dispatch_1d, dispatch_2d, set_params, u32_of, weight_of, BufPool, CodecGpu,
    CodecGpuStream, GpuConv, GpuNlc,
};
use crate::metal_backend::post_dispatch_barrier;
use crate::tts::TtsDecoderConfig;
use crate::{InferError, Result, Tensor};
use metal::{
    Buffer, ComputeCommandEncoderRef, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
};
use std::collections::HashMap;

/// Compilés sans fast-math : mêmes formules que le CPU, arrondis IEEE.
const PREFIX_KERNELS: &str = r#"
#include <metal_stdlib>
using namespace metal;

// RMSNorm par ligne. p = [channels, eps_bits]
kernel void codec_rms_norm_f32(
    device const float* x [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device float* out     [[buffer(2)]],
    constant uint* p      [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]
) {
    #pragma METAL fp contract(off)
    threadgroup float partial[256];
    const uint ch = p[0];
    const float eps = as_type<float>(p[1]);
    device const float* xs = x + row * ch;
    float sum = 0.0f;
    for (uint c = tid; c < ch; c += 256u) { sum += xs[c] * xs[c]; }
    partial[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) { partial[tid] += partial[tid + stride]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv = 1.0f / sqrt(partial[0] / float(ch) + eps);
    for (uint c = tid; c < ch; c += 256u) { out[row * ch + c] = xs[c] * inv * w[c]; }
}

// LayerNorm par ligne. p = [channels, eps_bits]
kernel void codec_layer_norm_f32(
    device const float* x [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device const float* b [[buffer(2)]],
    device float* out     [[buffer(3)]],
    constant uint* p      [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]
) {
    #pragma METAL fp contract(off)
    threadgroup float partial[256];
    const uint ch = p[0];
    const float eps = as_type<float>(p[1]);
    device const float* xs = x + row * ch;
    float sum = 0.0f;
    for (uint c = tid; c < ch; c += 256u) { sum += xs[c]; }
    partial[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) { partial[tid] += partial[tid + stride]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float mean = partial[0] / float(ch);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float var = 0.0f;
    for (uint c = tid; c < ch; c += 256u) {
        const float centered = xs[c] - mean;
        var += centered * centered;
    }
    partial[tid] = var;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) { partial[tid] += partial[tid + stride]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv = 1.0f / sqrt(partial[0] / float(ch) + eps);
    for (uint c = tid; c < ch; c += 256u) {
        out[row * ch + c] = (xs[c] - mean) * inv * w[c] + b[c];
    }
}

// RoPE rotate-half en place. Le dénominateur base^(2i/d) vient du CPU.
// p = [rows, heads, head_dim, position]
kernel void codec_rope_f32(
    device float* x            [[buffer(0)]],
    device const float* denom  [[buffer(1)]],
    constant uint* p           [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    #pragma METAL fp contract(off)
    const uint heads = p[1];
    const uint pairs = p[2] / 2u;
    if (gid >= p[0] * heads * pairs) { return; }
    const uint pair = gid % pairs;
    const uint head = (gid / pairs) % heads;
    const uint row = gid / (pairs * heads);
    const float angle = float(p[3] + row) / denom[pair];
    const float c = cos(angle);
    const float s = sin(angle);
    const uint left_idx = row * heads * p[2] + head * p[2] + pair;
    const uint right_idx = left_idx + pairs;
    const float left = x[left_idx];
    const float right = x[right_idx];
    x[left_idx] = left * c - right * s;
    x[right_idx] = left * s + right * c;
}

// Scores causaux q·k × scale. Grille (clé, ligne·tête).
// p = [rows, heads, kv_repeat, head_dim, past, stride, scale_bits]
kernel void codec_attn_scores_f32(
    device const float* q  [[buffer(0)]],
    device const float* k  [[buffer(1)]],
    device float* scores   [[buffer(2)]],
    constant uint* p       [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    #pragma METAL fp contract(off)
    const uint key = gid.x;
    const uint rh = gid.y;
    const uint heads = p[1];
    if (rh >= p[0] * heads) { return; }
    const uint row = rh / heads;
    const uint head = rh % heads;
    if (key > p[4] + row) { return; }
    const uint hd = p[3];
    const uint kv_head = head / p[2];
    const uint kv_dim = (heads / p[2]) * hd;
    device const float* qr = q + row * heads * hd + head * hd;
    device const float* kr = k + key * kv_dim + kv_head * hd;
    float dot = 0.0f;
    for (uint d = 0u; d < hd; ++d) { dot += qr[d] * kr[d]; }
    scores[rh * p[5] + key] = dot * as_type<float>(p[6]);
}

// Softmax en place des clés 0..=past+ligne. p = [heads, past, stride]
kernel void codec_attn_softmax_f32(
    device float* scores [[buffer(0)]],
    constant uint* p     [[buffer(1)]],
    uint tid [[thread_index_in_threadgroup]],
    uint rh [[threadgroup_position_in_grid]]
) {
    #pragma METAL fp contract(off)
    threadgroup float partial[256];
    const uint n = p[1] + rh / p[0] + 1u;
    device float* row = scores + rh * p[2];
    float best = -INFINITY;
    for (uint i = tid; i < n; i += 256u) { best = max(best, row[i]); }
    partial[tid] = best;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) { partial[tid] = max(partial[tid], partial[tid + stride]); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float top = partial[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = 0.0f;
    for (uint i = tid; i < n; i += 256u) {
        const float e = exp(row[i] - top);
        row[i] = e;
        sum += e;
    }
    partial[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) { partial[tid] += partial[tid + stride]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float total = partial[0];
    if (total > 0.0f) {
        for (uint i = tid; i < n; i += 256u) { row[i] = row[i] / total; }
    }
}

// Contexte Σ score·v. Grille (tête·dim, ligne).
// p = [rows, heads, kv_repeat, head_dim, past, stride]
kernel void codec_attn_values_f32(
    device const float* scores [[buffer(0)]],
    device const float* v      [[buffer(1)]],
    device float* out          [[buffer(2)]],
    constant uint* p           [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    #pragma METAL fp contract(off)
    const uint heads = p[1];
    const uint hd = p[3];
    const uint col = gid.x;
    const uint row = gid.y;
    if (col >= heads * hd || row >= p[0]) { return; }
    const uint head = col / hd;
    const uint dim = col % hd;
    const uint kv_dim = (heads / p[2]) * hd;
    const uint v_col = (head / p[2]) * hd + dim;
    device const float* sr = scores + (row * heads + head) * p[5];
    const uint n = p[4] + row + 1u;
    float acc = 0.0f;
    for (uint key = 0u; key < n; ++key) { acc += sr[key] * v[key * kv_dim + v_col]; }
    out[row * heads * hd + col] = acc;
}

// silu(gate) · up. p = [total]
kernel void codec_silu_mul_f32(
    device const float* gate [[buffer(0)]],
    device const float* up   [[buffer(1)]],
    device float* out        [[buffer(2)]],
    constant uint* p         [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    #pragma METAL fp contract(off)
    if (gid >= p[0]) { return; }
    const float g = gate[gid];
    out[gid] = (g / (1.0f + exp(-g))) * up[gid];
}

// GELU erf (approximation d'Abramowitz-Stegun du CPU), en place. p = [total]
kernel void codec_gelu_f32(
    device float* x  [[buffer(0)]],
    constant uint* p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    #pragma METAL fp contract(off)
    if (gid >= p[0]) { return; }
    const float value = x[gid];
    const float z = value * 0.70710677f;
    const float sign = (z < 0.0f) ? -1.0f : 1.0f;
    const float a = fabs(z);
    const float t = 1.0f / (1.0f + 0.3275911f * a);
    const float y = 1.0f - (((((1.0614054f * t - 1.4531521f) * t) + 1.4214138f) * t - 0.28449672f) * t
        + 0.2548296f) * t * exp(-a * a);
    x[gid] = 0.5f * value * (1.0f + sign * y);
}

// a + b · w[canal]. p = [total, channels]
kernel void codec_add_scaled_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device const float* w [[buffer(2)]],
    device float* out     [[buffer(3)]],
    constant uint* p      [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    #pragma METAL fp contract(off)
    if (gid >= p[0]) { return; }
    out[gid] = a[gid] + b[gid] * w[gid % p[1]];
}

// dst[offset + i] = src[i]. p = [n, offset]
kernel void codec_copy_f32(
    device const float* src [[buffer(0)]],
    device float* dst       [[buffer(1)]],
    constant uint* p        [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p[0]) { return; }
    dst[p[1] + gid] = src[gid];
}
"#;

const PIPELINES: [&str; 10] = [
    "codec_rms_norm_f32",
    "codec_layer_norm_f32",
    "codec_rope_f32",
    "codec_attn_scores_f32",
    "codec_attn_softmax_f32",
    "codec_attn_values_f32",
    "codec_silu_mul_f32",
    "codec_gelu_f32",
    "codec_add_scaled_f32",
    "codec_copy_f32",
];

/// Pipelines, poids vectoriels et dimensions du préfixe GPU.
#[derive(Debug)]
pub(super) struct GpuPrefix {
    pipelines: Vec<ComputePipelineState>,
    vectors: HashMap<String, Buffer>,
    rope_denominators: Buffer,
    layers: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rms_eps: f32,
    upsampling: Vec<usize>,
}

/// Cache K/V résident d'une couche du `pre_transformer`.
#[derive(Debug)]
pub(super) struct GpuKv {
    keys: Buffer,
    values: Buffer,
    capacity: usize,
    len: usize,
}

enum Kernel {
    RmsNorm,
    LayerNorm,
    Rope,
    Scores,
    Softmax,
    Values,
    SiluMul,
    Gelu,
    AddScaled,
    Copy,
}

fn usize_of(value: i32, what: &str) -> Result<usize> {
    usize::try_from(value).map_err(|_| InferError::Config(format!("{what} codec négatif: {value}")))
}

impl GpuPrefix {
    fn pipeline(&self, kernel: Kernel) -> &ComputePipelineState {
        &self.pipelines[kernel as usize]
    }
}

impl CodecGpu {
    /// Compile les kernels du préfixe et téléverse ses poids.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si un poids manque, si une forme est incohérente ou
    /// si Metal échoue.
    pub(super) fn register_prefix(
        &mut self,
        weights: &HashMap<String, Tensor>,
        cfg: &TtsDecoderConfig,
    ) -> Result<()> {
        let pipelines = compile_prefix(&self.device)?;
        let layers = usize_of(cfg.num_hidden_layers, "layers")?;
        let heads = usize_of(cfg.num_attention_heads, "heads")?;
        let kv_heads = usize_of(cfg.num_key_value_heads, "kv heads")?;
        let head_dim = usize_of(cfg.head_dim, "head_dim")?;
        if kv_heads == 0 || heads % kv_heads != 0 || head_dim % 2 != 0 {
            return Err(InferError::Dimension(format!(
                "codec GPU: têtes {heads}/{kv_heads} ou head_dim {head_dim} invalides"
            )));
        }
        let latent = usize_of(cfg.latent_dim, "latent_dim")?;
        let upsampling = cfg
            .upsampling_ratios
            .iter()
            .map(|ratio| usize_of(*ratio, "upsampling ratio"))
            .collect::<Result<Vec<_>>>()?;

        for name in ["decoder.quantizer.rvq_first", "decoder.quantizer.rvq_rest"] {
            self.register_linear(weights, &format!("{name}.output_proj"))?;
        }
        self.register_conv(weights, "decoder.pre_conv.conv")?;
        let mut vectors = HashMap::new();
        let mut vector = |key: String| -> Result<()> {
            let buffer = self.upload(channel_slice(weight_of(weights, &key)?)?)?;
            vectors.insert(key, buffer);
            Ok(())
        };
        let transformer = "decoder.pre_transformer";
        for layer in 0..layers {
            let prefix = format!("{transformer}.layers.{layer}");
            vector(format!("{prefix}.input_layernorm.weight"))?;
            vector(format!("{prefix}.post_attention_layernorm.weight"))?;
            vector(format!("{prefix}.self_attn_layer_scale.scale"))?;
            vector(format!("{prefix}.mlp_layer_scale.scale"))?;
        }
        vector(format!("{transformer}.norm.weight"))?;
        for idx in 0..upsampling.len() {
            let block = format!("decoder.upsample.{idx}.1");
            vector(format!("{block}.norm.weight"))?;
            vector(format!("{block}.norm.bias"))?;
            vector(format!("{block}.gamma"))?;
        }
        self.register_linear(weights, &format!("{transformer}.input_proj"))?;
        self.register_linear(weights, &format!("{transformer}.output_proj"))?;
        for layer in 0..layers {
            let prefix = format!("{transformer}.layers.{layer}");
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                self.register_linear(weights, &format!("{prefix}.self_attn.{proj}"))?;
            }
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                self.register_linear(weights, &format!("{prefix}.mlp.{proj}"))?;
            }
        }
        for idx in 0..upsampling.len() {
            self.register_transpose(weights, &format!("decoder.upsample.{idx}.0.conv"))?;
            let block = format!("decoder.upsample.{idx}.1");
            self.register_conv(weights, &format!("{block}.dwconv.conv"))?;
            if let Some(conv) = self.convs.get_mut(&format!("{block}.dwconv.conv")) {
                if conv.inner != 1 || conv.out_dim != latent {
                    return Err(InferError::Dimension(format!(
                        "codec GPU: dwconv {block} attendu depthwise {latent}"
                    )));
                }
                conv.groups = latent;
            }
            self.register_linear(weights, &format!("{block}.pwconv1"))?;
            self.register_linear(weights, &format!("{block}.pwconv2"))?;
        }
        // Mêmes flottants que `rope_heads_at` : base.powf(2i/d) en f32.
        let denominators = (0..head_dim / 2)
            .map(|pair| cfg.rope_theta.powf((2 * pair) as f32 / head_dim as f32))
            .collect::<Vec<_>>();
        let rope_denominators = self.upload(&denominators)?;
        self.prefix = Some(GpuPrefix {
            pipelines,
            vectors,
            rope_denominators,
            layers,
            heads,
            kv_heads,
            head_dim,
            rms_eps: cfg.rms_norm_eps,
            upsampling,
        });
        Ok(())
    }

    /// Linéaire `[out, in]` (biais optionnel) enregistrée comme conv k=1.
    fn register_linear(&mut self, weights: &HashMap<String, Tensor>, prefix: &str) -> Result<()> {
        let weight = weight_of(weights, &format!("{prefix}.weight"))?;
        let (out_dim, inner) = match weight.shape() {
            [out_dim, inner] | [out_dim, inner, 1] => (*out_dim, *inner),
            shape => {
                return Err(InferError::Dimension(format!(
                    "linéaire codec {prefix} attendue [out,in], reçue {shape:?}"
                )));
            }
        };
        let bias = self.upload_bias(weights, prefix)?;
        let conv = GpuConv {
            wkc: self.upload(weight.data())?,
            bias,
            out_dim,
            inner,
            kernel: 1,
            groups: 1,
        };
        self.convs.insert(prefix.to_string(), conv);
        Ok(())
    }

    /// Indique si le préfixe GPU est actif.
    pub(crate) fn has_prefix(&self) -> bool {
        self.prefix.is_some()
    }

    /// Encode le préfixe depuis les sommes RVQ `[time, codebook_dim]` (premier
    /// quantificateur, puis les suivants) et renvoie l'entrée de la section
    /// `decoder.decoder`, résidente.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le préfixe est inactif, si une forme est
    /// incohérente ou si un encodage échoue.
    pub(super) fn encode_prefix_stream(
        &self,
        state: &mut CodecGpuStream,
        encoder: &ComputeCommandEncoderRef,
        pool: &mut BufPool,
        time: usize,
        first: &[f32],
        rest: &[f32],
    ) -> Result<GpuNlc> {
        let prefix = self
            .prefix
            .as_ref()
            .ok_or_else(|| InferError::Metal("préfixe codec GPU inactif".to_string()))?;
        let upload = |data: &[f32]| -> Result<GpuNlc> {
            let channels = data.len().checked_div(time).unwrap_or(0);
            if channels == 0 || channels * time != data.len() {
                return Err(InferError::Dimension(format!(
                    "somme RVQ de {} valeurs pour {time} trames",
                    data.len()
                )));
            }
            Ok(GpuNlc {
                buffer: self.upload(data)?,
                pool_index: None,
                time,
                channels,
            })
        };
        let first = upload(first)?;
        let rest = upload(rest)?;
        let projected_first = self.conv_with_start(
            encoder,
            pool,
            &first,
            "decoder.quantizer.rvq_first.output_proj",
            1,
            false,
            0,
        )?;
        let projected_rest = self.conv_with_start(
            encoder,
            pool,
            &rest,
            "decoder.quantizer.rvq_rest.output_proj",
            1,
            false,
            0,
        )?;
        let mut hidden = self.add(encoder, pool, &projected_first, &projected_rest)?;
        pool.release(projected_first);
        pool.release(projected_rest);
        hidden = self.swap(pool, hidden, |pool, x| {
            self.conv_stream(state, encoder, pool, x, "decoder.pre_conv.conv", 1, false)
        })?;
        hidden = self.swap(pool, hidden, |pool, x| {
            self.pre_transformer(prefix, state, encoder, pool, x)
        })?;
        for (idx, stride) in prefix.upsampling.iter().copied().enumerate() {
            hidden = self.swap(pool, hidden, |pool, x| {
                self.transpose_stream(
                    state,
                    encoder,
                    pool,
                    x,
                    &format!("decoder.upsample.{idx}.0.conv"),
                    stride,
                )
            })?;
            let block = format!("decoder.upsample.{idx}.1");
            let h = self.conv_stream(
                state,
                encoder,
                pool,
                &hidden,
                &format!("{block}.dwconv.conv"),
                1,
                false,
            )?;
            let normed = self.layer_norm(prefix, encoder, pool, &h, &block)?;
            pool.release(h);
            let up = self.linear(encoder, pool, &normed, &format!("{block}.pwconv1"))?;
            pool.release(normed);
            self.elementwise(
                encoder,
                prefix.pipeline(Kernel::Gelu),
                &[&up.buffer],
                up.len(),
            )?;
            let down = self.linear(encoder, pool, &up, &format!("{block}.pwconv2"))?;
            pool.release(up);
            let out = self.add_scaled(
                prefix,
                encoder,
                pool,
                &hidden,
                &down,
                &format!("{block}.gamma"),
            )?;
            pool.release(down);
            pool.release(hidden);
            hidden = out;
        }
        Ok(hidden)
    }

    /// Applique `step` à `x` puis libère `x`.
    fn swap(
        &self,
        pool: &mut BufPool,
        x: GpuNlc,
        step: impl FnOnce(&mut BufPool, &GpuNlc) -> Result<GpuNlc>,
    ) -> Result<GpuNlc> {
        let out = step(pool, &x)?;
        pool.release(x);
        Ok(out)
    }

    fn pre_transformer(
        &self,
        prefix: &GpuPrefix,
        state: &mut CodecGpuStream,
        encoder: &ComputeCommandEncoderRef,
        pool: &mut BufPool,
        x: &GpuNlc,
    ) -> Result<GpuNlc> {
        let transformer = "decoder.pre_transformer";
        let mut h = self.linear(encoder, pool, x, &format!("{transformer}.input_proj"))?;
        if state.kv.len() < prefix.layers {
            state.kv.resize_with(prefix.layers, || None);
        }
        for layer in 0..prefix.layers {
            let name = format!("{transformer}.layers.{layer}");
            let normed = self.rms_norm(
                prefix,
                encoder,
                pool,
                &h,
                &format!("{name}.input_layernorm.weight"),
            )?;
            let attn = self.attention(
                prefix,
                state,
                layer,
                encoder,
                pool,
                &normed,
                &format!("{name}.self_attn"),
            )?;
            pool.release(normed);
            let summed = self.add_scaled(
                prefix,
                encoder,
                pool,
                &h,
                &attn,
                &format!("{name}.self_attn_layer_scale.scale"),
            )?;
            pool.release(attn);
            pool.release(h);
            h = summed;
            let normed = self.rms_norm(
                prefix,
                encoder,
                pool,
                &h,
                &format!("{name}.post_attention_layernorm.weight"),
            )?;
            let gate = self.linear(encoder, pool, &normed, &format!("{name}.mlp.gate_proj"))?;
            let up = self.linear(encoder, pool, &normed, &format!("{name}.mlp.up_proj"))?;
            pool.release(normed);
            let act = pool.acquire(gate.time, gate.channels)?;
            self.elementwise(
                encoder,
                prefix.pipeline(Kernel::SiluMul),
                &[&gate.buffer, &up.buffer, &act.buffer],
                act.len(),
            )?;
            pool.release(gate);
            pool.release(up);
            let ff = self.linear(encoder, pool, &act, &format!("{name}.mlp.down_proj"))?;
            pool.release(act);
            let summed = self.add_scaled(
                prefix,
                encoder,
                pool,
                &h,
                &ff,
                &format!("{name}.mlp_layer_scale.scale"),
            )?;
            pool.release(ff);
            pool.release(h);
            h = summed;
        }
        let normed = self.rms_norm(
            prefix,
            encoder,
            pool,
            &h,
            &format!("{transformer}.norm.weight"),
        )?;
        pool.release(h);
        let out = self.linear(
            encoder,
            pool,
            &normed,
            &format!("{transformer}.output_proj"),
        )?;
        pool.release(normed);
        Ok(out)
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "attention codec: état KV de la couche, encodeur et pool"
    )]
    fn attention(
        &self,
        prefix: &GpuPrefix,
        state: &mut CodecGpuStream,
        layer: usize,
        encoder: &ComputeCommandEncoderRef,
        pool: &mut BufPool,
        x: &GpuNlc,
        name: &str,
    ) -> Result<GpuNlc> {
        let q = self.linear(encoder, pool, x, &format!("{name}.q_proj"))?;
        let k = self.linear(encoder, pool, x, &format!("{name}.k_proj"))?;
        let v = self.linear(encoder, pool, x, &format!("{name}.v_proj"))?;
        let kv_dim = prefix.kv_heads * prefix.head_dim;
        if q.channels != prefix.heads * prefix.head_dim
            || k.channels != kv_dim
            || v.channels != kv_dim
        {
            return Err(InferError::Dimension(format!(
                "attention codec {name}: q={} k={} v={}",
                q.channels, k.channels, v.channels
            )));
        }
        let past = state.kv[layer].as_ref().map_or(0, |kv| kv.len);
        let rows = x.time;
        for (tensor, heads) in [(&q, prefix.heads), (&k, prefix.kv_heads)] {
            let pairs = prefix.head_dim / 2;
            let params = [
                u32_of(rows)?,
                u32_of(heads)?,
                u32_of(prefix.head_dim)?,
                u32_of(past)?,
            ];
            encoder.set_compute_pipeline_state(prefix.pipeline(Kernel::Rope));
            encoder.set_buffer(0, Some(&tensor.buffer), 0);
            encoder.set_buffer(1, Some(&prefix.rope_denominators), 0);
            set_params(encoder, 2, &params);
            dispatch_1d(encoder, rows * heads * pairs);
            post_dispatch_barrier(encoder);
        }
        let kv = self.append_kv(prefix, &mut state.kv[layer], encoder, &k, &v)?;
        pool.release(k);
        pool.release(v);
        let keys = past + rows;
        let scores = pool.acquire(rows * prefix.heads, keys)?;
        let scale = (prefix.head_dim as f32).powf(-0.5);
        let kv_repeat = prefix.heads / prefix.kv_heads;
        let params = [
            u32_of(rows)?,
            u32_of(prefix.heads)?,
            u32_of(kv_repeat)?,
            u32_of(prefix.head_dim)?,
            u32_of(past)?,
            u32_of(keys)?,
            scale.to_bits(),
        ];
        encoder.set_compute_pipeline_state(prefix.pipeline(Kernel::Scores));
        encoder.set_buffer(0, Some(&q.buffer), 0);
        encoder.set_buffer(1, Some(&kv.keys), 0);
        encoder.set_buffer(2, Some(&scores.buffer), 0);
        set_params(encoder, 3, &params);
        dispatch_2d(encoder, keys, rows * prefix.heads);
        post_dispatch_barrier(encoder);
        pool.release(q);

        let softmax = [u32_of(prefix.heads)?, u32_of(past)?, u32_of(keys)?];
        encoder.set_compute_pipeline_state(prefix.pipeline(Kernel::Softmax));
        encoder.set_buffer(0, Some(&scores.buffer), 0);
        set_params(encoder, 1, &softmax);
        encoder.dispatch_thread_groups(
            MTLSize::new(u64::from(u32_of(rows * prefix.heads)?), 1, 1),
            MTLSize::new(256, 1, 1),
        );
        post_dispatch_barrier(encoder);

        let ctx = pool.acquire(rows, prefix.heads * prefix.head_dim)?;
        encoder.set_compute_pipeline_state(prefix.pipeline(Kernel::Values));
        encoder.set_buffer(0, Some(&scores.buffer), 0);
        encoder.set_buffer(1, Some(&kv.values), 0);
        encoder.set_buffer(2, Some(&ctx.buffer), 0);
        set_params(encoder, 3, &params[..6]);
        dispatch_2d(encoder, ctx.channels, rows);
        post_dispatch_barrier(encoder);
        pool.release(scores);
        let out = self.linear(encoder, pool, &ctx, &format!("{name}.o_proj"))?;
        pool.release(ctx);
        Ok(out)
    }

    /// Ajoute les lignes K/V du lot au cache de la couche (capacité doublée
    /// au besoin, anciennes lignes recopiées sur le GPU).
    fn append_kv<'a>(
        &self,
        prefix: &GpuPrefix,
        slot: &'a mut Option<GpuKv>,
        encoder: &ComputeCommandEncoderRef,
        k: &GpuNlc,
        v: &GpuNlc,
    ) -> Result<&'a GpuKv> {
        let row = k.channels;
        let past = slot.as_ref().map_or(0, |kv| kv.len);
        let needed = past + k.time;
        if slot.as_ref().is_none_or(|kv| kv.capacity < needed) {
            let capacity = needed.next_power_of_two().max(64);
            let bytes = u64::try_from(capacity * row * std::mem::size_of::<f32>())
                .map_err(|_| InferError::Metal("cache KV codec hors u64".to_string()))?;
            let fresh = GpuKv {
                keys: self
                    .device
                    .new_buffer(bytes, MTLResourceOptions::StorageModeShared),
                values: self
                    .device
                    .new_buffer(bytes, MTLResourceOptions::StorageModeShared),
                capacity,
                len: past,
            };
            if let Some(old) = slot.as_ref() {
                for (src, dst) in [(&old.keys, &fresh.keys), (&old.values, &fresh.values)] {
                    self.copy(prefix, encoder, src, dst, past * row, 0)?;
                }
            }
            *slot = Some(fresh);
        }
        let kv = slot
            .as_mut()
            .ok_or_else(|| InferError::Metal("cache KV codec absent".to_string()))?;
        for (src, dst) in [(&k.buffer, &kv.keys), (&v.buffer, &kv.values)] {
            self.copy(prefix, encoder, src, dst, k.len(), past * row)?;
        }
        kv.len = needed;
        Ok(kv)
    }

    fn copy(
        &self,
        prefix: &GpuPrefix,
        encoder: &ComputeCommandEncoderRef,
        src: &Buffer,
        dst: &Buffer,
        len: usize,
        offset: usize,
    ) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        encoder.set_compute_pipeline_state(prefix.pipeline(Kernel::Copy));
        encoder.set_buffer(0, Some(src), 0);
        encoder.set_buffer(1, Some(dst), 0);
        set_params(encoder, 2, &[u32_of(len)?, u32_of(offset)?]);
        dispatch_1d(encoder, len);
        post_dispatch_barrier(encoder);
        Ok(())
    }

    fn linear(
        &self,
        encoder: &ComputeCommandEncoderRef,
        pool: &mut BufPool,
        x: &GpuNlc,
        prefix: &str,
    ) -> Result<GpuNlc> {
        self.conv_with_start(encoder, pool, x, prefix, 1, false, 0)
    }

    fn rms_norm(
        &self,
        prefix: &GpuPrefix,
        encoder: &ComputeCommandEncoderRef,
        pool: &mut BufPool,
        x: &GpuNlc,
        weight: &str,
    ) -> Result<GpuNlc> {
        let weight = vector(prefix, weight)?;
        let out = pool.acquire(x.time, x.channels)?;
        encoder.set_compute_pipeline_state(prefix.pipeline(Kernel::RmsNorm));
        encoder.set_buffer(0, Some(&x.buffer), 0);
        encoder.set_buffer(1, Some(weight), 0);
        encoder.set_buffer(2, Some(&out.buffer), 0);
        set_params(encoder, 3, &[u32_of(x.channels)?, prefix.rms_eps.to_bits()]);
        dispatch_rows(encoder, x.time)?;
        Ok(out)
    }

    fn layer_norm(
        &self,
        prefix: &GpuPrefix,
        encoder: &ComputeCommandEncoderRef,
        pool: &mut BufPool,
        x: &GpuNlc,
        block: &str,
    ) -> Result<GpuNlc> {
        let weight = vector(prefix, &format!("{block}.norm.weight"))?;
        let bias = vector(prefix, &format!("{block}.norm.bias"))?;
        let out = pool.acquire(x.time, x.channels)?;
        encoder.set_compute_pipeline_state(prefix.pipeline(Kernel::LayerNorm));
        encoder.set_buffer(0, Some(&x.buffer), 0);
        encoder.set_buffer(1, Some(weight), 0);
        encoder.set_buffer(2, Some(bias), 0);
        encoder.set_buffer(3, Some(&out.buffer), 0);
        set_params(encoder, 4, &[u32_of(x.channels)?, 1.0e-6_f32.to_bits()]);
        dispatch_rows(encoder, x.time)?;
        Ok(out)
    }

    /// `a + b · w[canal]` : multiplication par canal puis résiduel, comme
    /// `mul_channel` suivi de `add` côté CPU.
    fn add_scaled(
        &self,
        prefix: &GpuPrefix,
        encoder: &ComputeCommandEncoderRef,
        pool: &mut BufPool,
        a: &GpuNlc,
        b: &GpuNlc,
        weight: &str,
    ) -> Result<GpuNlc> {
        if a.time != b.time || a.channels != b.channels {
            return Err(InferError::Dimension(format!(
                "add_scaled GPU [{},{}] vs [{},{}]",
                a.time, a.channels, b.time, b.channels
            )));
        }
        let weight = vector(prefix, weight)?;
        let out = pool.acquire(a.time, a.channels)?;
        encoder.set_compute_pipeline_state(prefix.pipeline(Kernel::AddScaled));
        encoder.set_buffer(0, Some(&a.buffer), 0);
        encoder.set_buffer(1, Some(&b.buffer), 0);
        encoder.set_buffer(2, Some(weight), 0);
        encoder.set_buffer(3, Some(&out.buffer), 0);
        set_params(encoder, 4, &[u32_of(out.len())?, u32_of(a.channels)?]);
        dispatch_1d(encoder, out.len());
        post_dispatch_barrier(encoder);
        Ok(out)
    }

    /// Kernel élément par élément : `buffers` liés dans l'ordre, puis `[total]`.
    fn elementwise(
        &self,
        encoder: &ComputeCommandEncoderRef,
        pipeline: &ComputePipelineState,
        buffers: &[&Buffer],
        total: usize,
    ) -> Result<()> {
        encoder.set_compute_pipeline_state(pipeline);
        for (index, buffer) in (0..).zip(buffers) {
            encoder.set_buffer(index, Some(buffer), 0);
        }
        set_params(encoder, buffers.len() as u64, &[u32_of(total)?]);
        dispatch_1d(encoder, total);
        post_dispatch_barrier(encoder);
        Ok(())
    }
}

fn vector<'a>(prefix: &'a GpuPrefix, key: &str) -> Result<&'a Buffer> {
    prefix
        .vectors
        .get(key)
        .ok_or_else(|| InferError::MissingWeight(format!("codec GPU {key}")))
}

fn dispatch_rows(encoder: &ComputeCommandEncoderRef, rows: usize) -> Result<()> {
    encoder.dispatch_thread_groups(
        MTLSize::new(u64::from(u32_of(rows)?), 1, 1),
        MTLSize::new(256, 1, 1),
    );
    post_dispatch_barrier(encoder);
    Ok(())
}

fn compile_prefix(device: &Device) -> Result<Vec<ComputePipelineState>> {
    let options = crate::metal_backend::compile_options(false);
    let library = device
        .new_library_with_source(PREFIX_KERNELS, &options)
        .map_err(|error| InferError::Metal(format!("compilation préfixe codec: {error}")))?;
    PIPELINES
        .iter()
        .map(|name| {
            let function = library
                .get_function(name, None)
                .map_err(|error| InferError::Metal(format!("fonction codec {name}: {error}")))?;
            device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|error| InferError::Metal(format!("pipeline codec {name}: {error}")))
        })
        .collect()
}

const _: () = assert!(PIPELINES.len() == Kernel::Copy as usize + 1);
