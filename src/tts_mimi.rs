//! Encodeur Mimi du speech tokenizer Qwen3-TTS.

use crate::tts::{SafetensorPayload, TtsCodecConfig, TtsEncoderConfig};
use crate::{InferError, Result, Tensor};
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::Path;

/// Encode le PCM 24 kHz en codes RVQ `[1, valid_q, frames]`.
#[derive(Debug)]
pub struct TtsMimiEncoder {
    cfg: TtsEncoderConfig,
    valid_quantizers: usize,
    weights: HashMap<String, Tensor>,
}

#[derive(Clone, Debug)]
struct Nlc {
    time: usize,
    channels: usize,
    data: Vec<f32>,
}

impl TtsMimiEncoder {
    /// Charge l'encodeur Mimi depuis `speech_tokenizer/model.safetensors`.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si les poids ou la config encodeur sont absents.
    pub fn load(model_dir: impl AsRef<Path>, codec_config: &TtsCodecConfig) -> Result<Self> {
        let cfg = codec_config.encoder_config.clone().ok_or_else(|| {
            InferError::Config("encoder_config TTS absent du snapshot Base".to_string())
        })?;
        let valid_quantizers = usize_from_i32(
            codec_config.encoder_valid_num_quantizers,
            "encoder_valid_num_quantizers",
        )?;
        let payload = SafetensorPayload::open(
            &model_dir
                .as_ref()
                .join("speech_tokenizer/model.safetensors"),
        )?;
        Self::load_from_payload(&payload, cfg, valid_quantizers)
    }

    pub(crate) fn load_from_payload(
        payload: &SafetensorPayload,
        cfg: TtsEncoderConfig,
        valid_quantizers: usize,
    ) -> Result<Self> {
        let mut weights = HashMap::new();
        let mut cluster = HashMap::new();
        let mut embsum = HashMap::new();

        for name in payload.names() {
            if !name.starts_with("encoder.") {
                continue;
            }
            if let Some(base) = name.strip_suffix("._codebook.cluster_usage") {
                cluster.insert(
                    format!("{base}.codebook"),
                    payload.read_dense_tensor(&name)?,
                );
            } else if let Some(base) = name.strip_suffix("._codebook.embedding_sum") {
                embsum.insert(
                    format!("{base}.codebook"),
                    payload.read_dense_tensor(&name)?,
                );
            } else if let Some(base) = name.strip_suffix(".codebook.cluster_usage") {
                cluster.insert(
                    format!("{base}.codebook"),
                    payload.read_dense_tensor(&name)?,
                );
            } else if let Some(base) = name.strip_suffix(".codebook.embed_sum") {
                embsum.insert(
                    format!("{base}.codebook"),
                    payload.read_dense_tensor(&name)?,
                );
            } else if name.ends_with(".codebook.initialized") {
                continue;
            } else {
                weights.insert(name.clone(), payload.read_dense_tensor(&name)?);
            }
        }

        for (base, usage) in cluster {
            let sum = embsum
                .get(&base)
                .ok_or_else(|| InferError::MissingWeight(format!("{base}.embedding_sum Mimi")))?;
            weights.insert(format!("{base}.embed.weight"), codebook_embed(&usage, sum)?);
        }

        Ok(Self {
            cfg,
            valid_quantizers,
            weights,
        })
    }

    /// Encode une référence PCM mono 24 kHz en codes RVQ.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si un poids manque ou si une forme est incompatible.
    pub fn encode_pcm_24k(&self, pcm_24k: &[f32]) -> Result<Vec<Vec<i32>>> {
        let audio = Nlc::new(pcm_24k.len(), 1, pcm_24k.to_vec())?;
        let xs = self.seanet_encode(&audio)?;
        let xs = self.encoder_transformer(&xs)?;
        let enc_frame_rate =
            self.cfg.sampling_rate as f32 / product(&self.cfg.upsampling_ratios) as f32;
        let down_stride = (enc_frame_rate / self.cfg.frame_rate).round();
        if down_stride < 1.0 {
            return Err(InferError::Config(format!(
                "stride downsample Mimi invalide: {down_stride}"
            )));
        }
        let down = self.streamable_conv1d(
            &xs,
            "encoder.downsample.conv.weight",
            None,
            down_stride as usize,
            1,
            1,
            self.cfg.use_causal_conv,
            PadMode::Edge,
        )?;
        self.split_rvq_encode(&down)
    }

    fn seanet_encode(&self, x: &Nlc) -> Result<Nlc> {
        let mut h = self.streamable_conv1d(
            x,
            "encoder.encoder.layers.0.conv.weight",
            Some("encoder.encoder.layers.0.conv.bias"),
            1,
            1,
            1,
            self.cfg.use_causal_conv,
            PadMode::Constant,
        )?;

        let ratios_rev = self
            .cfg
            .upsampling_ratios
            .iter()
            .rev()
            .map(|v| usize_from_i32(*v, "upsampling_ratio"))
            .collect::<Result<Vec<_>>>()?;
        let resnet_idx = [1_usize, 4, 7, 10];
        let down_idx = [3_usize, 6, 9, 12];
        for (layer, ratio) in ratios_rev.iter().copied().enumerate() {
            h = self.seanet_resnet_block(
                &h,
                &format!("encoder.encoder.layers.{}", resnet_idx[layer]),
                1,
            )?;
            let hd = h.map(elu_scalar);
            h = self.streamable_conv1d(
                &hd,
                &format!("encoder.encoder.layers.{}.conv.weight", down_idx[layer]),
                Some(&format!(
                    "encoder.encoder.layers.{}.conv.bias",
                    down_idx[layer]
                )),
                ratio,
                1,
                1,
                true,
                PadMode::Constant,
            )?;
        }

        let h = h.map(elu_scalar);
        self.streamable_conv1d(
            &h,
            "encoder.encoder.layers.14.conv.weight",
            Some("encoder.encoder.layers.14.conv.bias"),
            1,
            1,
            1,
            self.cfg.use_causal_conv,
            PadMode::Constant,
        )
    }

    fn seanet_resnet_block(&self, x: &Nlc, prefix: &str, dilation: usize) -> Result<Nlc> {
        let residual = x.clone();
        let h = x.map(elu_scalar);
        let h = self.streamable_conv1d(
            &h,
            &format!("{prefix}.block.1.conv.weight"),
            Some(&format!("{prefix}.block.1.conv.bias")),
            1,
            dilation,
            1,
            self.cfg.use_causal_conv,
            PadMode::Constant,
        )?;
        let h = h.map(elu_scalar);
        let h = self.streamable_conv1d(
            &h,
            &format!("{prefix}.block.3.conv.weight"),
            Some(&format!("{prefix}.block.3.conv.bias")),
            1,
            1,
            1,
            self.cfg.use_causal_conv,
            PadMode::Constant,
        )?;
        residual.add(&h)
    }

    fn encoder_transformer(&self, x: &Nlc) -> Result<Nlc> {
        let mut h = x.clone();
        for layer in 0..usize_from_i32(self.cfg.num_hidden_layers, "Mimi hidden layers")? {
            let p = format!("encoder.encoder_transformer.layers.{layer}");
            let residual = h.clone();
            let n1 = self.layer_norm(
                &h,
                &format!("{p}.input_layernorm.weight"),
                &format!("{p}.input_layernorm.bias"),
                1.0e-5,
            )?;
            let attn = self.enc_attention(&n1, &format!("{p}.self_attn"))?;
            let attn = self.mul_channel(&attn, &format!("{p}.self_attn_layer_scale.scale"))?;
            h = residual.add(&attn)?;

            let residual = h.clone();
            let n2 = self.layer_norm(
                &h,
                &format!("{p}.post_attention_layernorm.weight"),
                &format!("{p}.post_attention_layernorm.bias"),
                1.0e-5,
            )?;
            let ff = self
                .linear(&n2, &format!("{p}.mlp.fc1"))?
                .map(gelu_erf_scalar);
            let ff = self.linear(&ff, &format!("{p}.mlp.fc2"))?;
            let ff = self.mul_channel(&ff, &format!("{p}.mlp_layer_scale.scale"))?;
            h = residual.add(&ff)?;
        }
        Ok(h)
    }

    fn enc_attention(&self, x: &Nlc, prefix: &str) -> Result<Nlc> {
        let head_dim = usize_from_i32(self.cfg.head_dim, "Mimi head_dim")?;
        let n_heads = usize_from_i32(self.cfg.num_attention_heads, "Mimi heads")?;
        let n_kv = usize_from_i32(self.cfg.num_key_value_heads, "Mimi kv heads")?;
        if n_heads != n_kv {
            return Err(InferError::Config(format!(
                "Mimi GQA non porté: heads={n_heads}, kv_heads={n_kv}"
            )));
        }
        let q = rope_heads(
            &self.linear(x, &format!("{prefix}.q_proj"))?,
            n_heads,
            head_dim,
            self.cfg.rope_theta,
        )?;
        let k = rope_heads(
            &self.linear(x, &format!("{prefix}.k_proj"))?,
            n_heads,
            head_dim,
            self.cfg.rope_theta,
        )?;
        let v = self.linear(x, &format!("{prefix}.v_proj"))?;
        let scale = (head_dim as f32).powf(-0.5);
        let mut out = Nlc::zeros(x.time, n_heads * head_dim);

        for time in 0..x.time {
            for head in 0..n_heads {
                let mut scores = vec![0.0_f32; time + 1];
                for key_time in 0..=time {
                    let mut acc = 0.0_f32;
                    for dim in 0..head_dim {
                        acc += q.get(time, head * head_dim + dim)
                            * k.get(key_time, head * head_dim + dim);
                    }
                    scores[key_time] = acc * scale;
                }
                softmax_in_place(&mut scores);
                for dim in 0..head_dim {
                    let mut acc = 0.0_f32;
                    for (key_time, score) in scores.iter().copied().enumerate() {
                        acc += score * v.get(key_time, head * head_dim + dim);
                    }
                    out.set(time, head * head_dim + dim, acc);
                }
            }
        }
        self.linear(&out, &format!("{prefix}.o_proj"))
    }

    fn split_rvq_encode(&self, xs: &Nlc) -> Result<Vec<Vec<i32>>> {
        let semantic = usize_from_i32(self.cfg.num_semantic_quantizers, "semantic quantizers")?;
        let total = usize_from_i32(self.cfg.num_quantizers, "Mimi quantizers")?;
        let mut codes = self.rvq_encode_one(
            xs,
            "encoder.quantizer.semantic_residual_vector_quantizer",
            semantic,
        )?;
        if total > semantic {
            let rest = self.rvq_encode_one(
                xs,
                "encoder.quantizer.acoustic_residual_vector_quantizer",
                total - semantic,
            )?;
            for (left, right) in codes.iter_mut().zip(rest) {
                left.extend(right);
            }
        }
        for frame in &mut codes {
            frame.truncate(self.valid_quantizers);
        }
        Ok(codes)
    }

    fn rvq_encode_one(&self, xs: &Nlc, prefix: &str, n_q: usize) -> Result<Vec<Vec<i32>>> {
        let proj = self.conv1x1(xs, &format!("{prefix}.input_proj.weight"))?;
        let mut residual = proj.clone();
        let mut codes = vec![Vec::with_capacity(n_q); proj.time];
        for q in 0..n_q {
            let emb = self.weight(&format!("{prefix}.layers.{q}.codebook.embed.weight"))?;
            let (bins, dim) = matrix_shape(emb)?;
            if residual.channels != dim {
                return Err(InferError::Dimension(format!(
                    "RVQ {prefix} dim résiduelle={} codebook={dim}",
                    residual.channels
                )));
            }
            let c2 = codebook_half_norms(emb, bins, dim);
            let indices = residual
                .data
                .par_chunks(dim)
                .map(|row| nearest_code(row, emb, bins, dim, &c2))
                .collect::<Vec<_>>();
            let mut next = residual.data.clone();
            for (time, idx) in indices.iter().copied().enumerate() {
                let idx_i32 = i32::try_from(idx)
                    .map_err(|_| InferError::Config(format!("code RVQ hors i32: {idx}")))?;
                codes[time].push(idx_i32);
                let emb_row = emb.row_slice(idx)?;
                let start = time * dim;
                subtract_residual_row(&mut next[start..start + dim], emb_row);
            }
            residual = Nlc::new(proj.time, dim, next)?;
        }
        Ok(codes)
    }

    fn streamable_conv1d(
        &self,
        x: &Nlc,
        weight_key: &str,
        bias_key: Option<&str>,
        stride: usize,
        dilation: usize,
        groups: usize,
        causal: bool,
        pad_mode: PadMode,
    ) -> Result<Nlc> {
        let weight = self.weight(weight_key)?;
        let [out_dim, in_per_group, kernel] = conv_weight_shape(weight, weight_key)?;
        if stride == 0 || dilation == 0 || groups == 0 {
            return Err(InferError::Config(format!(
                "conv Mimi paramètres invalides stride={stride} dilation={dilation} groups={groups}"
            )));
        }
        if x.channels != in_per_group * groups || out_dim % groups != 0 {
            return Err(InferError::Dimension(format!(
                "conv {weight_key} incompatible x_channels={} groups={groups} weight={:?}",
                x.channels,
                weight.shape()
            )));
        }
        let ksize = (kernel - 1)
            .checked_mul(dilation)
            .and_then(|v| v.checked_add(1))
            .ok_or_else(|| InferError::Shape(format!("kernel conv trop grand {weight_key}")))?;
        let padding_total = ksize.checked_sub(stride).ok_or_else(|| {
            InferError::Dimension(format!(
                "padding conv négatif {weight_key}: ksize={ksize}, stride={stride}"
            ))
        })?;
        let extra = extra_padding(x.time, ksize, stride, padding_total);
        let (pad_left, pad_right) = if causal {
            (padding_total, extra)
        } else {
            let pr = padding_total / 2;
            (padding_total - pr, pr + extra)
        };
        let padded = pad_time(x, pad_left, pad_right, pad_mode)?;
        let out_time = conv_out_time(padded.time, ksize, stride)?;
        let bias = match bias_key {
            Some(key) => Some(self.weight(key)?),
            None => None,
        };
        let out_per_group = out_dim / groups;
        let mut out = vec![0.0_f32; out_time * out_dim];
        out.par_chunks_mut(out_dim)
            .enumerate()
            .try_for_each(|(time, out_row)| -> Result<()> {
                for out_ch in 0..out_dim {
                    let group = out_ch / out_per_group;
                    let in_start = group * in_per_group;
                    let mut acc = bias.map_or(Ok(0.0), |b| channel_value(b, out_ch))?;
                    for k in 0..kernel {
                        let src_time = time * stride + k * dilation;
                        for in_ch in 0..in_per_group {
                            let weight_idx = (out_ch * in_per_group + in_ch) * kernel + k;
                            acc +=
                                padded.get(src_time, in_start + in_ch) * weight.data()[weight_idx];
                        }
                    }
                    out_row[out_ch] = acc;
                }
                Ok(())
            })?;
        Nlc::new(out_time, out_dim, out)
    }

    fn linear(&self, x: &Nlc, prefix: &str) -> Result<Nlc> {
        let weight = self.weight(&format!("{prefix}.weight"))?;
        let (out_dim, in_dim) = matrix_shape(weight)?;
        if x.channels != in_dim {
            return Err(InferError::Dimension(format!(
                "linear {prefix} attendu in={in_dim}, reçu {}",
                x.channels
            )));
        }
        let bias = self.opt(&format!("{prefix}.bias"));
        let mut out = vec![0.0_f32; x.time * out_dim];
        out.par_chunks_mut(out_dim)
            .enumerate()
            .try_for_each(|(time, out_row)| -> Result<()> {
                let row = x.row(time);
                for out_ch in 0..out_dim {
                    let mut acc = dot(row, weight.row_slice(out_ch)?);
                    if let Some(bias) = bias {
                        acc += channel_value(bias, out_ch)?;
                    }
                    out_row[out_ch] = acc;
                }
                Ok(())
            })?;
        Nlc::new(x.time, out_dim, out)
    }

    fn conv1x1(&self, x: &Nlc, weight_key: &str) -> Result<Nlc> {
        let weight = self.weight(weight_key)?;
        match weight.shape() {
            [out_dim, in_dim, 1] if *in_dim == x.channels => {
                let mut out = vec![0.0_f32; x.time * out_dim];
                out.par_chunks_mut(*out_dim)
                    .enumerate()
                    .for_each(|(time, out_row)| {
                        let row = x.row(time);
                        for (out_ch, value) in out_row.iter_mut().enumerate() {
                            let start = out_ch * in_dim;
                            *value = dot(row, &weight.data()[start..start + in_dim]);
                        }
                    });
                Nlc::new(x.time, *out_dim, out)
            }
            shape => Err(InferError::Dimension(format!(
                "conv1x1 {weight_key} attendu [out,{},1], reçu {shape:?}",
                x.channels
            ))),
        }
    }

    fn layer_norm(&self, x: &Nlc, weight_key: &str, bias_key: &str, eps: f32) -> Result<Nlc> {
        let weight = self.weight(weight_key)?;
        let bias = self.weight(bias_key)?;
        let mut out = vec![0.0_f32; x.data.len()];
        out.par_chunks_mut(x.channels).enumerate().try_for_each(
            |(time, out_row)| -> Result<()> {
                let row = x.row(time);
                let mean = row.iter().sum::<f32>() / x.channels as f32;
                let var = row
                    .iter()
                    .map(|value| {
                        let centered = value - mean;
                        centered * centered
                    })
                    .sum::<f32>()
                    / x.channels as f32;
                let inv = 1.0 / (var + eps).sqrt();
                for ch in 0..x.channels {
                    out_row[ch] = (row[ch] - mean) * inv * channel_value(weight, ch)?
                        + channel_value(bias, ch)?;
                }
                Ok(())
            },
        )?;
        Nlc::new(x.time, x.channels, out)
    }

    fn mul_channel(&self, x: &Nlc, weight_key: &str) -> Result<Nlc> {
        let weight = self.weight(weight_key)?;
        let mut out = x.clone();
        out.data
            .par_chunks_mut(x.channels)
            .try_for_each(|row| -> Result<()> {
                for (ch, value) in row.iter_mut().enumerate() {
                    *value *= channel_value(weight, ch)?;
                }
                Ok(())
            })?;
        Ok(out)
    }

    fn weight(&self, key: &str) -> Result<&Tensor> {
        self.weights
            .get(key)
            .ok_or_else(|| InferError::MissingWeight(key.to_string()))
    }

    fn opt(&self, key: &str) -> Option<&Tensor> {
        self.weights.get(key)
    }
}

impl Nlc {
    fn new(time: usize, channels: usize, data: Vec<f32>) -> Result<Self> {
        if time
            .checked_mul(channels)
            .ok_or_else(|| InferError::Shape("NLC Mimi trop grand".to_string()))?
            != data.len()
        {
            return Err(InferError::Shape(format!(
                "NLC Mimi [{time},{channels}] incompatible avec {} valeurs",
                data.len()
            )));
        }
        Ok(Self {
            time,
            channels,
            data,
        })
    }

    fn zeros(time: usize, channels: usize) -> Self {
        Self {
            time,
            channels,
            data: vec![0.0; time * channels],
        }
    }

    fn row(&self, time: usize) -> &[f32] {
        let start = time * self.channels;
        &self.data[start..start + self.channels]
    }

    fn get(&self, time: usize, channel: usize) -> f32 {
        self.data[time * self.channels + channel]
    }

    fn set(&mut self, time: usize, channel: usize, value: f32) {
        self.data[time * self.channels + channel] = value;
    }

    fn add(&self, rhs: &Self) -> Result<Self> {
        if self.time != rhs.time || self.channels != rhs.channels {
            return Err(InferError::Dimension(format!(
                "NLC Mimi add gauche=[{},{}] droite=[{},{}]",
                self.time, self.channels, rhs.time, rhs.channels
            )));
        }
        let data = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(left, right)| left + right)
            .collect();
        Self::new(self.time, self.channels, data)
    }

    fn map(&self, f: impl Fn(f32) -> f32 + Sync + Send) -> Self {
        Self {
            time: self.time,
            channels: self.channels,
            data: self.data.par_iter().copied().map(f).collect(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum PadMode {
    Constant,
    Edge,
}

fn pad_time(x: &Nlc, left: usize, right: usize, mode: PadMode) -> Result<Nlc> {
    if left == 0 && right == 0 {
        return Ok(x.clone());
    }
    let out_time = left
        .checked_add(x.time)
        .and_then(|v| v.checked_add(right))
        .ok_or_else(|| InferError::Shape("pad Mimi trop grand".to_string()))?;
    let mut out = vec![0.0_f32; out_time * x.channels];
    for time in 0..out_time {
        let src = if time < left {
            match mode {
                PadMode::Constant => None,
                PadMode::Edge => Some(0),
            }
        } else if time >= left + x.time {
            match mode {
                PadMode::Constant => None,
                PadMode::Edge => x.time.checked_sub(1),
            }
        } else {
            Some(time - left)
        };
        if let Some(src_time) = src {
            let dst = time * x.channels;
            out[dst..dst + x.channels].copy_from_slice(x.row(src_time));
        }
    }
    Nlc::new(out_time, x.channels, out)
}

fn extra_padding(length: usize, ksize: usize, stride: usize, padding_total: usize) -> usize {
    let len_f = length as f32;
    let nframes = ((len_f + padding_total as f32 - ksize as f32).max(0.0)) / stride as f32 + 1.0;
    let ideal = (nframes.ceil() as usize - 1) * stride + ksize - padding_total;
    ideal.saturating_sub(length)
}

fn conv_out_time(padded_time: usize, ksize: usize, stride: usize) -> Result<usize> {
    if padded_time < ksize {
        return Ok(0);
    }
    Ok(1 + (padded_time - ksize) / stride)
}

fn codebook_embed(usage: &Tensor, sum: &Tensor) -> Result<Tensor> {
    let rows = match usage.shape() {
        [n] => *n,
        shape => {
            return Err(InferError::Dimension(format!(
                "cluster_usage Mimi attendu [N], reçu {shape:?}"
            )));
        }
    };
    let (sum_rows, cols) = matrix_shape(sum)?;
    if rows != sum_rows {
        return Err(InferError::Dimension(format!(
            "cluster_usage Mimi rows={rows} incompatible embedding_sum rows={sum_rows}"
        )));
    }
    let mut out = vec![0.0_f32; rows * cols];
    for row in 0..rows {
        let denom = usage.data()[row].max(1.0e-5);
        for col in 0..cols {
            out[row * cols + col] = sum.data()[row * cols + col] / denom;
        }
    }
    Tensor::from_vec(vec![rows, cols], out)
}

/// Précalcule `0.5 * ||code||²` par ligne du codebook RVQ.
///
/// Terme constant de l'astuce de distance utilisée par [`nearest_code`] :
/// évite de recalculer la norme du codebook à chaque appel.
fn codebook_half_norms(embed: &Tensor, bins: usize, dim: usize) -> Vec<f32> {
    (0..bins)
        .into_par_iter()
        .map(|row| {
            embed.data()[row * dim..(row + 1) * dim]
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                * 0.5
        })
        .collect()
}

/// Trouve l'entrée du codebook RVQ la plus proche de `row` (recherche du
/// plus proche voisin, distance euclidienne).
///
/// Minimise `0.5·‖code‖² − row·code`, proportionnel à `‖row−code‖²` à une
/// constante additive près (`‖row‖²`, indépendante du bin) et un facteur 2 —
/// équivalent à l'argmin de distance euclidienne complète sans le recalculer.
/// En cas d'égalité stricte de distance, conserve le premier bin rencontré
/// (index le plus petit), car la comparaison est `<` et non `<=`.
fn nearest_code(row: &[f32], embed: &Tensor, bins: usize, dim: usize, c2: &[f32]) -> usize {
    let mut best = 0_usize;
    let mut best_dist = f32::INFINITY;
    for (bin, c2_value) in c2.iter().copied().enumerate().take(bins) {
        let emb = &embed.data()[bin * dim..(bin + 1) * dim];
        let dist = c2_value - dot(row, emb);
        if dist < best_dist {
            best = bin;
            best_dist = dist;
        }
    }
    best
}

/// Soustrait le code retenu du résidu RVQ à cet étage (`résidu -= code`),
/// l'étage suivant quantifie l'erreur restante.
fn subtract_residual_row(residual: &mut [f32], code: &[f32]) {
    for (value, code_value) in residual.iter_mut().zip(code.iter()) {
        *value -= code_value;
    }
}

fn matrix_shape(tensor: &Tensor) -> Result<(usize, usize)> {
    tensor.as_matrix()
}

fn conv_weight_shape(weight: &Tensor, name: &str) -> Result<[usize; 3]> {
    match weight.shape() {
        [a, b, c] => Ok([*a, *b, *c]),
        shape => Err(InferError::Dimension(format!(
            "poids conv Mimi {name} attendu rang 3, reçu {shape:?}"
        ))),
    }
}

fn channel_value(tensor: &Tensor, idx: usize) -> Result<f32> {
    match tensor.shape() {
        [n] if idx < *n => Ok(tensor.data()[idx]),
        [1, n] if idx < *n => Ok(tensor.data()[idx]),
        shape => Err(InferError::Dimension(format!(
            "paramètre Mimi idx={idx} incompatible shape={shape:?}"
        ))),
    }
}

fn rope_heads(x: &Nlc, heads: usize, head_dim: usize, base: f32) -> Result<Nlc> {
    if x.channels != heads * head_dim {
        return Err(InferError::Dimension(format!(
            "RoPE Mimi channels={} attendu {}",
            x.channels,
            heads * head_dim
        )));
    }
    let mut out = x.clone();
    let pairs = head_dim / 2;
    for time in 0..x.time {
        for pair in 0..pairs {
            let angle = time as f32 / base.powf((2 * pair) as f32 / head_dim as f32);
            let cos = angle.cos();
            let sin = angle.sin();
            for head in 0..heads {
                let start = head * head_dim;
                let left_idx = start + pair;
                let right_idx = start + pairs + pair;
                let left = x.get(time, left_idx);
                let right = x.get(time, right_idx);
                out.set(time, left_idx, left * cos - right * sin);
                out.set(time, right_idx, left * sin + right * cos);
            }
        }
    }
    Ok(out)
}

fn softmax_in_place(values: &mut [f32]) {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for value in values.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }
    if sum > 0.0 {
        for value in values.iter_mut() {
            *value /= sum;
        }
    }
}

fn product(values: &[i32]) -> i32 {
    values.iter().product()
}

fn usize_from_i32(value: i32, what: &str) -> Result<usize> {
    usize::try_from(value).map_err(|_| InferError::Config(format!("{what} négatif: {value}")))
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right.iter())
        .map(|(left, right)| left * right)
        .sum()
}

fn elu_scalar(value: f32) -> f32 {
    if value >= 0.0 {
        value
    } else {
        value.exp() - 1.0
    }
}

fn gelu_erf_scalar(value: f32) -> f32 {
    0.5 * value * (1.0 + erf_approx(value * std::f32::consts::FRAC_1_SQRT_2))
}

fn erf_approx(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_4 * t - 1.453_152_1) * t) + 1.421_413_8) * t - 0.284_496_72) * t
            + 0.254_829_6)
            * t
            * (-x * x).exp();
    sign * y
}

#[cfg(test)]
mod tests;
