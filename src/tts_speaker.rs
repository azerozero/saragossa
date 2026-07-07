//! Encodeur speaker ECAPA-TDNN du clone Qwen3-TTS Base.

use crate::tts::{SafetensorPayload, TtsModelConfig, TtsSpeakerEncoderConfig};
use crate::{InferError, Result, Tensor};
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::Path;

/// Encode le log-mel speaker en x-vector `[1, enc_dim]`.
#[derive(Debug)]
pub struct TtsSpeakerEncoder {
    cfg: TtsSpeakerEncoderConfig,
    weights: HashMap<String, Tensor>,
}

#[derive(Clone, Debug)]
struct Ncl {
    channels: usize,
    time: usize,
    data: Vec<f32>,
}

impl TtsSpeakerEncoder {
    /// Charge les poids `speaker_encoder.*` depuis `model.safetensors`.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le snapshot n'est pas Base ou si un poids manque.
    pub fn load(model_dir: impl AsRef<Path>, model_config: &TtsModelConfig) -> Result<Self> {
        let cfg = model_config.speaker_encoder_config.clone().ok_or_else(|| {
            InferError::Config("speaker_encoder_config TTS absent du snapshot Base".to_string())
        })?;
        let payload = SafetensorPayload::open(&model_dir.as_ref().join("model.safetensors"))?;
        let mut weights = HashMap::new();
        for name in payload.names() {
            if name.starts_with("speaker_encoder.") {
                weights.insert(name.clone(), payload.read_dense_tensor(&name)?);
            }
        }
        if weights.is_empty() {
            return Err(InferError::MissingWeight("speaker_encoder.*".to_string()));
        }
        Ok(Self { cfg, weights })
    }

    /// Calcule le x-vector ECAPA depuis un log-mel `[1, frames, 128]`.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le mel ou les poids sont incompatibles.
    pub fn embed_mel(&self, mel_nlc: &Tensor) -> Result<Tensor> {
        let mut x = Ncl::from_mel_nlc(mel_nlc)?;
        let channels = [512_usize, 512, 512, 512, 1536];
        let kernels = [5_usize, 3, 3, 3, 1];
        let dilations = [1_usize, 2, 3, 4, 1];
        let scale = 8_usize;

        let mut hidden = Vec::new();
        x = self.tdnn_block(&x, "speaker_encoder.blocks.0", kernels[0], dilations[0])?;
        hidden.push(x.clone());
        for i in 1..channels.len() - 1 {
            x = self.se_res2net_block(
                &x,
                &format!("speaker_encoder.blocks.{i}"),
                scale,
                kernels[i],
                dilations[i],
            )?;
            hidden.push(x.clone());
        }

        let cat = Ncl::concat_channels(&hidden[1..])?;
        let x = self.tdnn_block(&cat, "speaker_encoder.mfa", kernels[4], dilations[4])?;
        let pooled = self.attentive_stats_pooling(&x, channels[channels.len() - 1])?;
        let emb = self.conv1x1_ncl(&pooled, "speaker_encoder.fc")?;
        if emb.time != 1 {
            return Err(InferError::Dimension(format!(
                "x-vector speaker attendu time=1, reçu {}",
                emb.time
            )));
        }
        let enc_dim = usize_from_i32(self.cfg.enc_dim, "speaker enc_dim")?;
        if emb.channels != enc_dim {
            return Err(InferError::Dimension(format!(
                "x-vector speaker attendu {enc_dim}, reçu {}",
                emb.channels
            )));
        }
        Tensor::from_vec(vec![1, emb.channels], emb.data)
    }

    fn se_res2net_block(
        &self,
        x: &Ncl,
        prefix: &str,
        scale: usize,
        kernel: usize,
        dilation: usize,
    ) -> Result<Ncl> {
        let residual = x.clone();
        let h = self.tdnn_block(x, &format!("{prefix}.tdnn1"), 1, 1)?;
        let h = self.res2net_block(
            &h,
            &format!("{prefix}.res2net_block"),
            scale,
            kernel,
            dilation,
        )?;
        let h = self.tdnn_block(&h, &format!("{prefix}.tdnn2"), 1, 1)?;
        let h = self.se_block(&h, &format!("{prefix}.se_block"))?;
        residual.add(&h)
    }

    fn res2net_block(
        &self,
        x: &Ncl,
        prefix: &str,
        scale: usize,
        kernel: usize,
        dilation: usize,
    ) -> Result<Ncl> {
        if scale == 0 || x.channels % scale != 0 {
            return Err(InferError::Dimension(format!(
                "Res2Net split invalide channels={} scale={scale}",
                x.channels
            )));
        }
        let chunks = x.split_channels(scale)?;
        let mut outputs = Vec::with_capacity(scale);
        let mut acc: Option<Ncl> = None;
        for (idx, chunk) in chunks.iter().enumerate() {
            let out = if idx == 0 {
                chunk.clone()
            } else if idx == 1 {
                self.tdnn_block(
                    chunk,
                    &format!("{prefix}.blocks.{}", idx - 1),
                    kernel,
                    dilation,
                )?
            } else {
                let prev = acc
                    .as_ref()
                    .ok_or_else(|| InferError::Config("Res2Net speaker acc absent".to_string()))?;
                let inp = chunk.add(prev)?;
                self.tdnn_block(
                    &inp,
                    &format!("{prefix}.blocks.{}", idx - 1),
                    kernel,
                    dilation,
                )?
            };
            acc = Some(out.clone());
            outputs.push(out);
        }
        Ncl::concat_channels(&outputs)
    }

    fn se_block(&self, x: &Ncl, prefix: &str) -> Result<Ncl> {
        let mean = x.mean_time();
        let se = self
            .conv1x1_ncl(&mean, &format!("{prefix}.conv1"))?
            .map(relu_scalar);
        let se = self
            .conv1x1_ncl(&se, &format!("{prefix}.conv2"))?
            .map(sigmoid_scalar);
        x.mul_broadcast_time(&se)
    }

    fn attentive_stats_pooling(&self, x: &Ncl, channels: usize) -> Result<Ncl> {
        if x.channels != channels {
            return Err(InferError::Dimension(format!(
                "ASP attendu channels={channels}, reçu {}",
                x.channels
            )));
        }
        let mean = x.mean_time();
        let std = x.std_time(1.0e-12)?;
        let cat =
            Ncl::concat_channels(&[x.clone(), mean.repeat_time(x.time), std.repeat_time(x.time)])?;
        let att = self
            .tdnn_block(&cat, "speaker_encoder.asp.tdnn", 1, 1)?
            .map(|value| value.tanh());
        let att = self.conv1x1_ncl(&att, "speaker_encoder.asp.conv")?;
        let att = att.softmax_time();
        let wmean = x.weighted_mean_time(&att)?;
        let centered = x.sub_broadcast_time(&wmean)?;
        let wvar = centered.square().weighted_mean_time(&att)?;
        let wstd = wvar.map(|value| value.max(1.0e-12).sqrt());
        Ncl::concat_channels(&[wmean, wstd])
    }

    fn tdnn_block(&self, x: &Ncl, prefix: &str, kernel: usize, dilation: usize) -> Result<Ncl> {
        self.tdnn_conv(x, &format!("{prefix}.conv"), kernel, dilation)
            .map(|out| out.map(relu_scalar))
    }

    fn tdnn_conv(&self, x: &Ncl, prefix: &str, kernel: usize, dilation: usize) -> Result<Ncl> {
        let pad = (kernel - 1)
            .checked_mul(dilation)
            .map(|v| v / 2)
            .ok_or_else(|| InferError::Shape("padding TDNN trop grand".to_string()))?;
        let padded = x.reflect_pad_time(pad);
        let weight = self.weight(&format!("{prefix}.weight"))?;
        let [out_dim, weight_kernel, in_dim] = speaker_conv_shape(weight, prefix)?;
        if weight_kernel != kernel || in_dim != x.channels {
            return Err(InferError::Dimension(format!(
                "TDNN {prefix} attendu kernel={kernel} in={}, poids={:?}",
                x.channels,
                weight.shape()
            )));
        }
        let bias = self.opt(&format!("{prefix}.bias"));
        let mut out = vec![0.0_f32; out_dim * x.time];
        out.par_chunks_mut(x.time)
            .enumerate()
            .try_for_each(|(out_ch, row)| -> Result<()> {
                for (time, value) in row.iter_mut().enumerate() {
                    let mut acc = bias.map_or(Ok(0.0), |b| channel_value(b, out_ch))?;
                    for k in 0..kernel {
                        let src_time = time + k * dilation;
                        for in_ch in 0..in_dim {
                            let weight_idx = (out_ch * kernel + k) * in_dim + in_ch;
                            acc += padded.get(in_ch, src_time) * weight.data()[weight_idx];
                        }
                    }
                    *value = acc;
                }
                Ok(())
            })?;
        Ncl::new(out_dim, x.time, out)
    }

    fn conv1x1_ncl(&self, x: &Ncl, prefix: &str) -> Result<Ncl> {
        let weight = self.weight(&format!("{prefix}.weight"))?;
        let [out_dim, kernel, in_dim] = speaker_conv_shape(weight, prefix)?;
        if kernel != 1 || in_dim != x.channels {
            return Err(InferError::Dimension(format!(
                "conv1x1 speaker {prefix} attendu [out,1,{}], reçu {:?}",
                x.channels,
                weight.shape()
            )));
        }
        let bias = self.opt(&format!("{prefix}.bias"));
        let mut out = vec![0.0_f32; out_dim * x.time];
        out.par_chunks_mut(x.time)
            .enumerate()
            .try_for_each(|(out_ch, row)| -> Result<()> {
                let w = &weight.data()[out_ch * in_dim..(out_ch + 1) * in_dim];
                for (time, value) in row.iter_mut().enumerate() {
                    let mut acc = bias.map_or(Ok(0.0), |b| channel_value(b, out_ch))?;
                    for (in_ch, weight) in w.iter().copied().enumerate() {
                        acc += x.get(in_ch, time) * weight;
                    }
                    *value = acc;
                }
                Ok(())
            })?;
        Ncl::new(out_dim, x.time, out)
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

impl Ncl {
    fn new(channels: usize, time: usize, data: Vec<f32>) -> Result<Self> {
        if channels
            .checked_mul(time)
            .ok_or_else(|| InferError::Shape("NCL speaker trop grand".to_string()))?
            != data.len()
        {
            return Err(InferError::Shape(format!(
                "NCL speaker [{channels},{time}] incompatible avec {} valeurs",
                data.len()
            )));
        }
        Ok(Self {
            channels,
            time,
            data,
        })
    }

    fn from_mel_nlc(mel: &Tensor) -> Result<Self> {
        let shape = mel.shape();
        let [batch, time, channels] = shape else {
            return Err(InferError::Dimension(format!(
                "mel speaker attendu [1,T,128], reçu {shape:?}"
            )));
        };
        if *batch != 1 || *channels != 128 {
            return Err(InferError::Dimension(format!(
                "mel speaker attendu [1,T,128], reçu {shape:?}"
            )));
        }
        let mut data = vec![0.0_f32; channels * time];
        for t in 0..*time {
            let src = t * channels;
            for ch in 0..*channels {
                data[ch * time + t] = mel.data()[src + ch];
            }
        }
        Self::new(*channels, *time, data)
    }

    fn get(&self, channel: usize, time: usize) -> f32 {
        self.data[channel * self.time + time]
    }

    fn row(&self, channel: usize) -> &[f32] {
        let start = channel * self.time;
        &self.data[start..start + self.time]
    }

    fn add(&self, rhs: &Self) -> Result<Self> {
        self.binary_same_shape(rhs, "add", |left, right| left + right)
    }

    fn square(&self) -> Self {
        self.map(|value| value * value)
    }

    fn map(&self, f: impl Fn(f32) -> f32 + Sync + Send) -> Self {
        Self {
            channels: self.channels,
            time: self.time,
            data: self.data.par_iter().copied().map(f).collect(),
        }
    }

    fn split_channels(&self, chunks: usize) -> Result<Vec<Self>> {
        let chunk = self.channels / chunks;
        (0..chunks)
            .map(|idx| {
                let start = idx * chunk * self.time;
                let end = start + chunk * self.time;
                Self::new(chunk, self.time, self.data[start..end].to_vec())
            })
            .collect()
    }

    fn concat_channels(parts: &[Self]) -> Result<Self> {
        let first = parts
            .first()
            .ok_or_else(|| InferError::Dimension("concat speaker vide".to_string()))?;
        let time = first.time;
        let channels = parts.iter().try_fold(0_usize, |acc, part| {
            if part.time != time {
                return Err(InferError::Dimension(format!(
                    "concat speaker time incompatible {} != {time}",
                    part.time
                )));
            }
            acc.checked_add(part.channels)
                .ok_or_else(|| InferError::Shape("concat speaker trop grand".to_string()))
        })?;
        let mut data = Vec::with_capacity(channels * time);
        for part in parts {
            data.extend_from_slice(&part.data);
        }
        Self::new(channels, time, data)
    }

    fn reflect_pad_time(&self, pad: usize) -> Self {
        if pad == 0 || self.time < pad + 1 {
            return self.clone();
        }
        let out_time = self.time + 2 * pad;
        let mut out = vec![0.0_f32; self.channels * out_time];
        for ch in 0..self.channels {
            for i in 0..pad {
                out[ch * out_time + i] = self.get(ch, pad - i);
            }
            let dst = ch * out_time + pad;
            out[dst..dst + self.time].copy_from_slice(self.row(ch));
            for i in 0..pad {
                out[ch * out_time + pad + self.time + i] = self.get(ch, self.time - 2 - i);
            }
        }
        Self {
            channels: self.channels,
            time: out_time,
            data: out,
        }
    }

    fn mean_time(&self) -> Self {
        let mut out = vec![0.0_f32; self.channels];
        out.par_iter_mut().enumerate().for_each(|(ch, value)| {
            *value = self.row(ch).iter().sum::<f32>() / self.time as f32;
        });
        Self {
            channels: self.channels,
            time: 1,
            data: out,
        }
    }

    fn std_time(&self, eps: f32) -> Result<Self> {
        let mean = self.mean_time();
        let mut out = vec![0.0_f32; self.channels];
        out.par_iter_mut().enumerate().for_each(|(ch, value)| {
            let m = mean.get(ch, 0);
            let var = self
                .row(ch)
                .iter()
                .map(|sample| {
                    let centered = sample - m;
                    centered * centered
                })
                .sum::<f32>()
                / self.time as f32;
            *value = (var + eps).sqrt();
        });
        Self::new(self.channels, 1, out)
    }

    fn repeat_time(&self, time: usize) -> Self {
        let mut out = vec![0.0_f32; self.channels * time];
        for ch in 0..self.channels {
            for t in 0..time {
                out[ch * time + t] = self.get(ch, 0);
            }
        }
        Self {
            channels: self.channels,
            time,
            data: out,
        }
    }

    fn mul_broadcast_time(&self, rhs: &Self) -> Result<Self> {
        self.binary_broadcast_time(rhs, "mul", |left, right| left * right)
    }

    fn sub_broadcast_time(&self, rhs: &Self) -> Result<Self> {
        self.binary_broadcast_time(rhs, "sub", |left, right| left - right)
    }

    fn weighted_mean_time(&self, weights: &Self) -> Result<Self> {
        if self.channels != weights.channels || self.time != weights.time {
            return Err(InferError::Dimension(format!(
                "weighted mean speaker shape gauche=[{},{}] droite=[{},{}]",
                self.channels, self.time, weights.channels, weights.time
            )));
        }
        let mut out = vec![0.0_f32; self.channels];
        out.par_iter_mut().enumerate().for_each(|(ch, value)| {
            *value = self
                .row(ch)
                .iter()
                .zip(weights.row(ch).iter())
                .map(|(sample, weight)| sample * weight)
                .sum();
        });
        Self::new(self.channels, 1, out)
    }

    fn softmax_time(&self) -> Self {
        let mut out = self.clone();
        out.data
            .par_chunks_mut(self.time)
            .for_each(softmax_in_place);
        out
    }

    fn binary_same_shape(
        &self,
        rhs: &Self,
        name: &str,
        f: impl Fn(f32, f32) -> f32 + Sync + Send,
    ) -> Result<Self> {
        if self.channels != rhs.channels || self.time != rhs.time {
            return Err(InferError::Dimension(format!(
                "NCL speaker {name} gauche=[{},{}] droite=[{},{}]",
                self.channels, self.time, rhs.channels, rhs.time
            )));
        }
        Ok(Self {
            channels: self.channels,
            time: self.time,
            data: self
                .data
                .par_iter()
                .zip(rhs.data.par_iter())
                .map(|(left, right)| f(*left, *right))
                .collect(),
        })
    }

    fn binary_broadcast_time(
        &self,
        rhs: &Self,
        name: &str,
        f: impl Fn(f32, f32) -> f32 + Sync + Send,
    ) -> Result<Self> {
        if self.channels != rhs.channels || rhs.time != 1 {
            return Err(InferError::Dimension(format!(
                "NCL speaker {name} broadcast gauche=[{},{}] droite=[{},{}]",
                self.channels, self.time, rhs.channels, rhs.time
            )));
        }
        let mut out = self.clone();
        out.data
            .par_chunks_mut(self.time)
            .enumerate()
            .for_each(|(ch, row)| {
                let value = rhs.get(ch, 0);
                for sample in row {
                    *sample = f(*sample, value);
                }
            });
        Ok(out)
    }
}

fn speaker_conv_shape(weight: &Tensor, name: &str) -> Result<[usize; 3]> {
    match weight.shape() {
        [out, kernel, input] => Ok([*out, *kernel, *input]),
        shape => Err(InferError::Dimension(format!(
            "poids conv speaker {name} attendu [out,k,in], reçu {shape:?}"
        ))),
    }
}

fn channel_value(tensor: &Tensor, idx: usize) -> Result<f32> {
    match tensor.shape() {
        [n] if idx < *n => Ok(tensor.data()[idx]),
        [1, n] if idx < *n => Ok(tensor.data()[idx]),
        shape => Err(InferError::Dimension(format!(
            "paramètre speaker idx={idx} incompatible shape={shape:?}"
        ))),
    }
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

fn usize_from_i32(value: i32, what: &str) -> Result<usize> {
    usize::try_from(value).map_err(|_| InferError::Config(format!("{what} négatif: {value}")))
}

fn relu_scalar(value: f32) -> f32 {
    value.max(0.0)
}

fn sigmoid_scalar(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

#[cfg(test)]
mod tests;
