//! Frontend et encodeur Whisper portés dans le noyau `saragossa`.
//!
//! Ce module couvre le frontend encodeur: conv1d mel, positions apprises,
//! self-attention non causale et FFN GELU. Les couches linéaires passent par
//! [`Linear::forward_with_runtime`], donc les matmuls denses réutilisent le
//! backend Metal existant quand un [`ForwardRuntime::metal`] est fourni.

mod decoder;

pub use decoder::{WhisperDecoder, WhisperModel};

use crate::{
    gelu, layer_norm, load_float_tensors, ForwardRuntime, InferError, Linear, Result, Tensor,
};
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

const LAYER_NORM_EPS: f32 = 1.0e-5;
const SAMPLE_RATE: usize = 16_000;
const N_FFT: usize = 400;
const HOP: usize = 160;
const N_FREQS: usize = N_FFT / 2 + 1;
const CHUNK_SECONDS: usize = 30;
const N_SAMPLES: usize = SAMPLE_RATE * CHUNK_SECONDS;
const N_FRAMES: usize = N_SAMPLES / HOP;
const MEL_FILTERS_80: &[u8] = include_bytes!("../assets/melfilters.bytes");
const MEL_FILTERS_128: &[u8] = include_bytes!("../assets/melfilters128.bytes");

/// Sous-ensemble utile du `config.json` Whisper HF.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct WhisperConfig {
    pub d_model: usize,
    pub encoder_attention_heads: usize,
    pub encoder_layers: usize,
    pub decoder_attention_heads: usize,
    pub decoder_layers: usize,
    pub num_mel_bins: usize,
    pub max_target_positions: usize,
    pub vocab_size: usize,
}

impl WhisperConfig {
    /// Charge la configuration Whisper depuis `config.json`.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le fichier est absent, invalide ou incohérent.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = std::fs::File::open(path).map_err(|source| InferError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let cfg: Self = serde_json::from_reader(file).map_err(|source| InferError::Json {
            path: path.to_path_buf(),
            source,
        })?;
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<()> {
        if self.d_model == 0 || self.encoder_attention_heads == 0 || self.encoder_layers == 0 {
            return Err(InferError::Config(format!(
                "config Whisper invalide: d_model={} heads={} layers={}",
                self.d_model, self.encoder_attention_heads, self.encoder_layers
            )));
        }
        if self.decoder_attention_heads == 0
            || self.decoder_layers == 0
            || self.max_target_positions == 0
            || self.vocab_size == 0
        {
            return Err(InferError::Config(format!(
                "config decodeur Whisper invalide: heads={} layers={} max_target_positions={} vocab_size={}",
                self.decoder_attention_heads,
                self.decoder_layers,
                self.max_target_positions,
                self.vocab_size
            )));
        }
        if self.d_model % self.encoder_attention_heads != 0 {
            return Err(InferError::Config(format!(
                "d_model={} non divisible par encoder_attention_heads={}",
                self.d_model, self.encoder_attention_heads
            )));
        }
        if self.d_model % self.decoder_attention_heads != 0 {
            return Err(InferError::Config(format!(
                "d_model={} non divisible par decoder_attention_heads={}",
                self.d_model, self.decoder_attention_heads
            )));
        }
        if !matches!(self.num_mel_bins, 80 | 128) {
            return Err(InferError::Config(format!(
                "num_mel_bins={} non supporte (80 ou 128 attendus)",
                self.num_mel_bins
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub(super) struct WhisperLayerNorm {
    pub(super) weight: Tensor,
    pub(super) bias: Tensor,
}

#[derive(Clone, Debug)]
pub(super) struct WhisperSelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
}

impl WhisperSelfAttention {
    /// Projette la requête `q_proj·x`. `forward_batched` : batch>1 (prefill cross,
    /// encodeur) → GEMM tuilé ; batch==1 (decode token) → qmv byte-identique.
    pub(super) fn project_q(&self, x: &Tensor, runtime: ForwardRuntime<'_>) -> Result<Tensor> {
        self.q_proj.forward_batched(x, runtime)
    }

    /// Projette la clé `k_proj·x`.
    pub(super) fn project_k(&self, x: &Tensor, runtime: ForwardRuntime<'_>) -> Result<Tensor> {
        self.k_proj.forward_batched(x, runtime)
    }

    /// Projette la valeur `v_proj·x`.
    pub(super) fn project_v(&self, x: &Tensor, runtime: ForwardRuntime<'_>) -> Result<Tensor> {
        self.v_proj.forward_batched(x, runtime)
    }

    /// Applique la projection de sortie `out_proj·ctx`.
    pub(super) fn project_out(&self, x: &Tensor, runtime: ForwardRuntime<'_>) -> Result<Tensor> {
        self.out_proj.forward_batched(x, runtime)
    }

    /// Accès aux couches linéaires (résolution des buffers GPU résidents).
    pub(super) fn linears(&self) -> (&Linear, &Linear, &Linear, &Linear) {
        (&self.q_proj, &self.k_proj, &self.v_proj, &self.out_proj)
    }
}

#[derive(Clone, Debug)]
struct WhisperEncoderLayer {
    self_attn_layer_norm: WhisperLayerNorm,
    self_attn: WhisperSelfAttention,
    final_layer_norm: WhisperLayerNorm,
    fc1: Linear,
    fc2: Linear,
}

/// Encodeur audio Whisper.
#[derive(Clone, Debug)]
pub struct WhisperEncoder {
    config: WhisperConfig,
    conv1_weight: Tensor,
    conv1_bias: Tensor,
    conv2_weight: Tensor,
    conv2_bias: Tensor,
    positions: Tensor,
    layers: Vec<WhisperEncoderLayer>,
    layer_norm: WhisperLayerNorm,
}

impl WhisperEncoder {
    /// Charge l'encodeur depuis un dossier HF local.
    ///
    /// Le chargeur accepte `model.safetensors` et `weights.safetensors`, comme le
    /// port mlx-rs existant.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si la config ou les poids nécessaires sont absents.
    pub fn from_model_dir(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let config = WhisperConfig::from_file(model_dir.join("config.json"))?;
        let weights = whisper_weights_path(model_dir)?;
        Self::from_tensors(config, load_float_tensors(weights)?)
    }

    /// Construit l'encodeur depuis les tenseurs HF déjà chargés.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si un poids obligatoire manque ou si une forme est
    /// incompatible avec la configuration.
    pub fn from_tensors(
        config: WhisperConfig,
        mut tensors: HashMap<String, Tensor>,
    ) -> Result<Self> {
        Self::from_tensor_map(config, &mut tensors)
    }

    pub(super) fn from_tensor_map(
        config: WhisperConfig,
        tensors: &mut HashMap<String, Tensor>,
    ) -> Result<Self> {
        config.validate()?;
        let conv1_weight = take(tensors, "model.encoder.conv1.weight")?;
        let conv1_bias = take(tensors, "model.encoder.conv1.bias")?;
        let conv2_weight = take(tensors, "model.encoder.conv2.weight")?;
        let conv2_bias = take(tensors, "model.encoder.conv2.bias")?;
        let positions = take(tensors, "model.encoder.embed_positions.weight")?;
        let mut layers = Vec::with_capacity(config.encoder_layers);
        for layer in 0..config.encoder_layers {
            let prefix = format!("model.encoder.layers.{layer}");
            layers.push(WhisperEncoderLayer {
                self_attn_layer_norm: take_layer_norm(
                    tensors,
                    &format!("{prefix}.self_attn_layer_norm"),
                )?,
                self_attn: WhisperSelfAttention {
                    q_proj: take_linear(tensors, &format!("{prefix}.self_attn.q_proj"))?,
                    k_proj: take_linear(tensors, &format!("{prefix}.self_attn.k_proj"))?,
                    v_proj: take_linear(tensors, &format!("{prefix}.self_attn.v_proj"))?,
                    out_proj: take_linear(tensors, &format!("{prefix}.self_attn.out_proj"))?,
                },
                final_layer_norm: take_layer_norm(tensors, &format!("{prefix}.final_layer_norm"))?,
                fc1: take_linear(tensors, &format!("{prefix}.fc1"))?,
                fc2: take_linear(tensors, &format!("{prefix}.fc2"))?,
            });
        }
        let layer_norm = take_layer_norm(tensors, "model.encoder.layer_norm")?;
        let encoder = Self {
            config,
            conv1_weight,
            conv1_bias,
            conv2_weight,
            conv2_bias,
            positions,
            layers,
            layer_norm,
        };
        encoder.validate_shapes()?;
        Ok(encoder)
    }

    /// Encode un mel-spectrogramme Whisper `[num_mel_bins, frames]`.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si les dimensions sont invalides ou si une operation
    /// interne échoue.
    pub fn encode_mel(&self, mel: &Tensor, runtime: ForwardRuntime<'_>) -> Result<Tensor> {
        let [mel_bins, _frames] = mel.shape() else {
            return Err(InferError::Dimension(format!(
                "mel Whisper attendu [n_mels, frames], reçu {:?}",
                mel.shape()
            )));
        };
        if *mel_bins != self.config.num_mel_bins {
            return Err(InferError::Dimension(format!(
                "mel bins={} incompatible avec config {}",
                mel_bins, self.config.num_mel_bins
            )));
        }

        let timing = std::env::var_os("RETI_STT_TIMING").is_some();
        let tc = std::time::Instant::now();

        // Frontend conv : RÉSIDENT (GPU, im2col + GEMM tuilé, un command buffer) par
        // défaut sur Metal ; `RETI_STT_CONV_PEROP=1` rebascule CPU rayon. Conv en
        // GEMM ⇒ drift ~1e-6 vs CPU (vérifié au golden).
        #[cfg(all(target_os = "macos", feature = "metal"))]
        let h_resident: Option<Tensor> = match runtime.metal_executor() {
            Some(metal) if std::env::var_os("RETI_STT_CONV_PEROP").is_none() => {
                let nlc = transpose_mel_to_nlc(mel)?;
                let conv = self.build_conv_weights(metal)?;
                Some(metal.encode_whisper_conv(&nlc, &conv)?)
            }
            _ => None,
        };
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        let h_resident: Option<Tensor> = None;

        let mut h = match h_resident {
            Some(h) => h,
            None => {
                let x = conv1d_mel_hf(mel, &self.conv1_weight, &self.conv1_bias, 1, 1)?;
                let x = gelu(&x);
                let mut h = conv1d_nlc_hf(&x, &self.conv2_weight, &self.conv2_bias, 2, 1)?;
                h = gelu(&h);
                add_position_embeddings(&h, &self.positions)?
            }
        };
        if timing {
            eprintln!("[enc] conv1+conv2: {} ms", tc.elapsed().as_millis());
        }

        // Chemin RÉSIDENT (défaut sur Metal) : les couches dans un command buffer,
        // zéro readback (vs ~7 syncs/couche). Mêmes GEMM/attention f32 ⇒
        // byte-identique ; seuls LayerNorm/GELU/add migrent GPU (drift ~1e-7,
        // vérifié au golden). `RETI_STT_ENCODER_PEROP=1` rebascule sur le per-op.
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if let Some(metal) = runtime.metal_executor() {
            if std::env::var_os("RETI_STT_ENCODER_PEROP").is_none() {
                let enc = self.build_resident(metal)?;
                let tr = std::time::Instant::now();
                let out = metal.encode_whisper_encoder(&h, &enc)?;
                if timing {
                    eprintln!("[enc] résident 32 couches: {} ms", tr.elapsed().as_millis());
                }
                return Ok(out);
            }
        }

        let (mut t_attn, mut t_ffn) = (0u128, 0u128);
        for layer in &self.layers {
            let ta = std::time::Instant::now();
            let residual = h.clone();
            let normed = layer.self_attn_layer_norm.forward(&h)?;
            let attn = self_attention_forward(&normed, &layer.self_attn, &self.config, runtime)?;
            h = residual.add(&attn)?;
            t_attn += ta.elapsed().as_micros();

            let tf = std::time::Instant::now();
            let residual = h.clone();
            let normed = layer.final_layer_norm.forward(&h)?;
            let ff = feed_forward(&normed, &layer.fc1, &layer.fc2, runtime)?;
            h = residual.add(&ff)?;
            t_ffn += tf.elapsed().as_micros();
        }
        if timing {
            eprintln!(
                "[enc] {} layers: attn={} ms, ffn={} ms",
                self.layers.len(),
                t_attn / 1000,
                t_ffn / 1000
            );
        }

        self.layer_norm.forward(&h)
    }

    /// Prépare les poids GPU (mémoïsés) de l'encodeur résident depuis les `Linear`/
    /// `WhisperLayerNorm` chargés. Les buffers sont mémoïsés par adresse → un seul
    /// upload, réutilisés entre énoncés.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn build_resident(
        &self,
        metal: &crate::MetalExecutor,
    ) -> Result<crate::metal_backend::WhisperResidentEncoder> {
        use crate::metal_backend::{
            WhisperResidentEncoder, WhisperResidentLayer, WhisperResidentNorm, WhisperResidentProj,
        };
        let na = crate::metal_backend::whisper_bf16_gemm_enabled();
        let proj = |linear: &Linear| -> Result<WhisperResidentProj> {
            let tensor = match linear.weight() {
                crate::LinearWeight::Dense(tensor) => tensor,
                crate::LinearWeight::AffineQuantized(_) => {
                    return Err(InferError::Config(
                        "encodeur résident: projection non dense".to_string(),
                    ));
                }
            };
            let weight = metal.cached_buffer_from_f32(tensor.data(), "whisper_proj_w")?;
            // Poids transposé bf16 (rhs^T) pour le matmul2d NA, construit une fois.
            let weight_na = if na {
                Some(metal.cached_rhs_t_bf16(tensor)?)
            } else {
                None
            };
            let bias = match linear.bias() {
                Some(bias) => Some(metal.cached_buffer_from_f32(bias.data(), "whisper_proj_b")?),
                None => None,
            };
            Ok(WhisperResidentProj {
                weight,
                bias,
                weight_na,
                weight_bf16: None,
            })
        };
        let norm = |ln: &WhisperLayerNorm| -> Result<WhisperResidentNorm> {
            Ok(WhisperResidentNorm {
                weight: metal.cached_buffer_from_f32(ln.weight.data(), "whisper_ln_w")?,
                bias: metal.cached_buffer_from_f32(ln.bias.data(), "whisper_ln_b")?,
            })
        };

        let mut layers = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            layers.push(WhisperResidentLayer {
                self_ln: norm(&layer.self_attn_layer_norm)?,
                q: proj(&layer.self_attn.q_proj)?,
                k: proj(&layer.self_attn.k_proj)?,
                v: proj(&layer.self_attn.v_proj)?,
                o: proj(&layer.self_attn.out_proj)?,
                final_ln: norm(&layer.final_layer_norm)?,
                fc1: proj(&layer.fc1)?,
                fc2: proj(&layer.fc2)?,
            });
        }
        let ffn_dim = self
            .layers
            .first()
            .and_then(|l| l.fc1.weight().shape().first().copied())
            .unwrap_or(0);
        Ok(WhisperResidentEncoder {
            layers,
            encoder_ln: norm(&self.layer_norm)?,
            d_model: self.config.d_model,
            heads: self.config.encoder_attention_heads,
            ffn_dim,
            eps: LAYER_NORM_EPS,
        })
    }

    /// Prépare les poids GPU (mémoïsés) du frontend conv mel résident. conv1/conv2
    /// `[out,in,k]` sont relus tels quels comme `[out, in·k]` pour le GEMM.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn build_conv_weights(
        &self,
        metal: &crate::MetalExecutor,
    ) -> Result<crate::metal_backend::WhisperConvWeights> {
        let (conv1_out, conv1_in, kernel) = match self.conv1_weight.shape() {
            [out, input, k] => (*out, *input, *k),
            shape => {
                return Err(InferError::Dimension(format!(
                    "conv1_weight attendu rang 3, reçu {shape:?}"
                )));
            }
        };
        let (conv2_out, conv2_in, conv2_kernel) = match self.conv2_weight.shape() {
            [out, input, k] => (*out, *input, *k),
            shape => {
                return Err(InferError::Dimension(format!(
                    "conv2_weight attendu rang 3, reçu {shape:?}"
                )));
            }
        };
        if conv1_out != self.config.d_model || conv1_in != self.config.num_mel_bins {
            return Err(InferError::Dimension(format!(
                "conv1_weight [{conv1_out},{conv1_in},{kernel}] incompatible avec d_model={} num_mel_bins={}",
                self.config.d_model, self.config.num_mel_bins
            )));
        }
        if conv2_out != self.config.d_model
            || conv2_in != self.config.d_model
            || conv2_kernel != kernel
        {
            return Err(InferError::Dimension(format!(
                "conv2_weight [{conv2_out},{conv2_in},{conv2_kernel}] incompatible avec d_model={} kernel={kernel}",
                self.config.d_model
            )));
        }
        let conv1_k = self
            .config
            .num_mel_bins
            .checked_mul(kernel)
            .ok_or_else(|| {
                InferError::Dimension("conv1_weight: num_mel_bins*kernel déborde".to_string())
            })?;
        let conv2_k = self.config.d_model.checked_mul(kernel).ok_or_else(|| {
            InferError::Dimension("conv2_weight: d_model*kernel déborde".to_string())
        })?;
        let na = crate::metal_backend::whisper_bf16_gemm_enabled();
        let conv1_weight_na = if na {
            Some(metal.cached_rhs_t_bf16_matrix(
                self.conv1_weight.data(),
                self.config.d_model,
                conv1_k,
                self.conv1_weight.data().as_ptr() as usize,
            )?)
        } else {
            None
        };
        let conv2_weight_na = if na {
            Some(metal.cached_rhs_t_bf16_matrix(
                self.conv2_weight.data(),
                self.config.d_model,
                conv2_k,
                self.conv2_weight.data().as_ptr() as usize,
            )?)
        } else {
            None
        };
        Ok(crate::metal_backend::WhisperConvWeights {
            conv1_weight: metal
                .cached_buffer_from_f32(self.conv1_weight.data(), "whisper_conv1_w")?,
            conv1_weight_na,
            conv1_bias: metal.cached_buffer_from_f32(self.conv1_bias.data(), "whisper_conv1_b")?,
            conv2_weight: metal
                .cached_buffer_from_f32(self.conv2_weight.data(), "whisper_conv2_w")?,
            conv2_weight_na,
            conv2_bias: metal.cached_buffer_from_f32(self.conv2_bias.data(), "whisper_conv2_b")?,
            positions: metal.cached_buffer_from_f32(self.positions.data(), "whisper_enc_pos")?,
            num_mel_bins: self.config.num_mel_bins,
            d_model: self.config.d_model,
            kernel,
        })
    }

    /// Encode un signal PCM mono 16 kHz en features audio Whisper.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le frontend mel ou l'encodeur échoue.
    pub fn encode_samples(&self, samples: &[f32], runtime: ForwardRuntime<'_>) -> Result<Tensor> {
        let timing = std::env::var_os("RETI_STT_TIMING").is_some();
        let t_mel = std::time::Instant::now();
        let mel = self.log_mel_spectrogram(samples)?;
        if timing {
            eprintln!("[stt] mel: {} ms", t_mel.elapsed().as_millis());
        }
        let t_encode_mel = std::time::Instant::now();
        let encoded = self.encode_mel(&mel, runtime)?;
        if timing {
            eprintln!(
                "[stt] encode_mel: {} ms",
                t_encode_mel.elapsed().as_millis()
            );
        }
        Ok(encoded)
    }

    /// Calcule le log-mel Whisper `[num_mel_bins, 3000]` depuis du PCM mono
    /// 16 kHz. Le signal est pad/tronqué à 30 secondes comme l'implémentation
    /// Whisper de référence.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si les assets mel ou la forme de sortie sont invalides.
    pub fn log_mel_spectrogram(&self, samples: &[f32]) -> Result<Tensor> {
        log_mel_spectrogram(samples, self.config.num_mel_bins)
    }

    fn validate_shapes(&self) -> Result<()> {
        validate_conv_weight(
            &self.conv1_weight,
            self.config.d_model,
            self.config.num_mel_bins,
            "model.encoder.conv1.weight",
        )?;
        validate_vector(
            &self.conv1_bias,
            self.config.d_model,
            "model.encoder.conv1.bias",
        )?;
        validate_conv_weight(
            &self.conv2_weight,
            self.config.d_model,
            self.config.d_model,
            "model.encoder.conv2.weight",
        )?;
        validate_vector(
            &self.conv2_bias,
            self.config.d_model,
            "model.encoder.conv2.bias",
        )?;
        let [_, pos_dim] = self.positions.shape() else {
            return Err(InferError::Dimension(format!(
                "model.encoder.embed_positions.weight attendu rang 2, reçu {:?}",
                self.positions.shape()
            )));
        };
        if *pos_dim != self.config.d_model {
            return Err(InferError::Dimension(format!(
                "positions dim={} incompatible avec d_model={}",
                pos_dim, self.config.d_model
            )));
        }
        Ok(())
    }
}

/// Calcule le log-mel Whisper `[num_mel_bins, 3000]` depuis du PCM mono 16 kHz.
///
/// # Errors
///
/// Renvoie une erreur si `num_mel_bins` n'est pas supporté.
pub fn log_mel_spectrogram(samples: &[f32], num_mel_bins: usize) -> Result<Tensor> {
    use rustfft::num_complex::Complex;

    let filters = mel_filter_bank(num_mel_bins)?;
    let hann = hann_window();
    let mut pcm = samples.to_vec();
    if pcm.len() < N_SAMPLES {
        pcm.resize(N_SAMPLES, 0.0);
    } else {
        pcm.truncate(N_SAMPLES);
    }

    let pad = N_FFT / 2;
    let mut padded = Vec::with_capacity(pcm.len() + 2 * pad);
    for index in 0..pad {
        padded.push(pcm[pad - index]);
    }
    padded.extend_from_slice(&pcm);
    for index in 0..pad {
        padded.push(pcm[pcm.len() - 2 - index]);
    }

    let fft = fft_400();
    let mut mel = vec![0.0_f32; num_mel_bins * N_FRAMES];
    let mut buf = vec![Complex::new(0.0_f32, 0.0_f32); N_FFT];
    let mut power = [0.0_f32; N_FREQS];
    for frame in 0..N_FRAMES {
        let start = frame * HOP;
        for index in 0..N_FFT {
            let sample = padded.get(start + index).copied().unwrap_or(0.0);
            buf[index] = Complex::new(sample * hann[index], 0.0);
        }
        fft.process(&mut buf);
        for freq in 0..N_FREQS {
            let value = buf[freq];
            power[freq] = value.re * value.re + value.im * value.im;
        }
        if std::env::var_os("RETI_STT_DENSE_MEL").is_some() {
            for mel_bin in 0..num_mel_bins {
                let mut acc = 0.0_f32;
                for freq in 0..N_FREQS {
                    acc += filters.dense_weight(mel_bin, freq) * power[freq];
                }
                mel[mel_bin * N_FRAMES + frame] = acc;
            }
        } else {
            for mel_bin in 0..num_mel_bins {
                let mut acc = 0.0_f32;
                for entry in filters.row(mel_bin) {
                    acc += entry.weight * power[entry.freq];
                }
                mel[mel_bin * N_FRAMES + frame] = acc;
            }
        }
    }

    let mut max_val = f32::MIN;
    for value in &mut mel {
        let logged = value.max(1.0e-10).log10();
        *value = logged;
        max_val = max_val.max(logged);
    }
    let floor = max_val - 8.0;
    for value in &mut mel {
        *value = (value.max(floor) + 4.0) / 4.0;
    }
    Tensor::from_vec(vec![num_mel_bins, N_FRAMES], mel)
}

#[derive(Clone, Copy)]
struct MelFilterEntry {
    freq: usize,
    weight: f32,
}

struct MelFilterBank {
    dense: Vec<f32>,
    entries: Vec<MelFilterEntry>,
    ranges: Vec<std::ops::Range<usize>>,
}

impl MelFilterBank {
    fn row(&self, mel_bin: usize) -> &[MelFilterEntry] {
        let range = self.ranges[mel_bin].clone();
        &self.entries[range]
    }

    fn dense_weight(&self, mel_bin: usize, freq: usize) -> f32 {
        self.dense[mel_bin * N_FREQS + freq]
    }
}

fn mel_filter_bank(num_mel_bins: usize) -> Result<&'static MelFilterBank> {
    static MEL_FILTERS_80_CACHE: OnceLock<MelFilterBank> = OnceLock::new();
    static MEL_FILTERS_128_CACHE: OnceLock<MelFilterBank> = OnceLock::new();
    let filters = match num_mel_bins {
        80 => MEL_FILTERS_80_CACHE.get_or_init(|| parse_sparse_mel_filters(MEL_FILTERS_80, 80)),
        128 => MEL_FILTERS_128_CACHE.get_or_init(|| parse_sparse_mel_filters(MEL_FILTERS_128, 128)),
        _ => {
            return Err(InferError::Config(format!(
                "num_mel_bins={num_mel_bins} non supporte pour Whisper"
            )));
        }
    };
    let expected = num_mel_bins;
    let actual = filters.ranges.len();
    if actual != expected {
        return Err(InferError::Shape(format!(
            "filterbank mel {num_mel_bins} lignes={actual}, attendu {expected}"
        )));
    }
    let expected_weights = num_mel_bins * N_FREQS;
    let actual_weights = filters.dense.len();
    if actual_weights != expected_weights {
        return Err(InferError::Shape(format!(
            "filterbank mel {num_mel_bins} poids={actual_weights}, attendu {expected_weights}"
        )));
    }
    Ok(filters)
}

fn parse_sparse_mel_filters(bytes: &[u8], num_mel_bins: usize) -> MelFilterBank {
    let dense = bytes
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect::<Vec<_>>();
    let mut entries = Vec::new();
    let mut ranges = Vec::with_capacity(num_mel_bins);
    for mel_bin in 0..num_mel_bins {
        let start = entries.len();
        for freq in 0..N_FREQS {
            if let Some(&weight) = dense.get(mel_bin * N_FREQS + freq) {
                if weight != 0.0 {
                    entries.push(MelFilterEntry { freq, weight });
                }
            }
        }
        ranges.push(start..entries.len());
    }
    MelFilterBank {
        dense,
        entries,
        ranges,
    }
}

fn hann_window() -> &'static [f32] {
    static HANN: OnceLock<Vec<f32>> = OnceLock::new();
    HANN.get_or_init(|| {
        (0..N_FFT)
            .map(|index| {
                let phase = (2.0 * std::f32::consts::PI * index as f32) / N_FFT as f32;
                0.5 * (1.0 - phase.cos())
            })
            .collect()
    })
    .as_slice()
}

fn fft_400() -> &'static Arc<dyn rustfft::Fft<f32>> {
    static FFT: OnceLock<Arc<dyn rustfft::Fft<f32>>> = OnceLock::new();
    FFT.get_or_init(|| {
        let mut planner = rustfft::FftPlanner::<f32>::new();
        planner.plan_fft_forward(N_FFT)
    })
}

impl WhisperLayerNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        layer_norm(x, &self.weight, &self.bias, LAYER_NORM_EPS)
    }
}

fn whisper_weights_path(model_dir: &Path) -> Result<PathBuf> {
    for name in ["model.safetensors", "weights.safetensors"] {
        let path = model_dir.join(name);
        if path.is_file() {
            return Ok(path);
        }
    }
    Err(InferError::MissingArtifact {
        path: model_dir.to_path_buf(),
        what: "model.safetensors ou weights.safetensors",
    })
}

fn take(tensors: &mut HashMap<String, Tensor>, key: &str) -> Result<Tensor> {
    tensors
        .remove(key)
        .ok_or_else(|| InferError::MissingWeight(key.to_string()))
}

fn take_optional(tensors: &mut HashMap<String, Tensor>, key: &str) -> Option<Tensor> {
    tensors.remove(key)
}

fn take_linear(tensors: &mut HashMap<String, Tensor>, prefix: &str) -> Result<Linear> {
    let weight = take(tensors, &format!("{prefix}.weight"))?;
    let bias = take_optional(tensors, &format!("{prefix}.bias"));
    Linear::new(weight, bias)
}

fn take_layer_norm(
    tensors: &mut HashMap<String, Tensor>,
    prefix: &str,
) -> Result<WhisperLayerNorm> {
    Ok(WhisperLayerNorm {
        weight: take(tensors, &format!("{prefix}.weight"))?,
        bias: take(tensors, &format!("{prefix}.bias"))?,
    })
}

fn validate_conv_weight(weight: &Tensor, out_dim: usize, in_dim: usize, key: &str) -> Result<()> {
    match weight.shape() {
        [out, input, kernel] if *out == out_dim && *input == in_dim && *kernel > 0 => Ok(()),
        shape => Err(InferError::Dimension(format!(
            "{key} attendu [{out_dim},{in_dim},k>0], reçu {shape:?}"
        ))),
    }
}

fn validate_vector(tensor: &Tensor, dim: usize, key: &str) -> Result<()> {
    match tensor.shape() {
        [n] if *n == dim => Ok(()),
        [1, n] if *n == dim => Ok(()),
        shape => Err(InferError::Dimension(format!(
            "{key} attendu [{dim}] ou [1,{dim}], reçu {shape:?}"
        ))),
    }
}

/// Transpose le mel `[num_mel_bins, frames]` (channels-major) en `[frames,
/// num_mel_bins]` (frames-major) pour le frontend conv résident.
#[cfg(all(target_os = "macos", feature = "metal"))]
fn transpose_mel_to_nlc(mel: &Tensor) -> Result<Tensor> {
    let [channels, frames] = mel.shape() else {
        return Err(InferError::Dimension(format!(
            "mel attendu [channels, frames], reçu {:?}",
            mel.shape()
        )));
    };
    let (channels, frames) = (*channels, *frames);
    let data = mel.data();
    let mut nlc = vec![0.0_f32; frames * channels];
    for frame in 0..frames {
        for channel in 0..channels {
            nlc[frame * channels + channel] = data[channel * frames + frame];
        }
    }
    Tensor::from_vec(vec![frames, channels], nlc)
}

fn conv1d_mel_hf(
    mel: &Tensor,
    weight_out_in_k: &Tensor,
    bias: &Tensor,
    stride: usize,
    padding: usize,
) -> Result<Tensor> {
    let [in_channels, frames] = mel.shape() else {
        return Err(InferError::Dimension(format!(
            "mel attendu [channels, frames], reçu {:?}",
            mel.shape()
        )));
    };
    let mut nlc = vec![0.0_f32; frames * in_channels];
    for frame in 0..*frames {
        for channel in 0..*in_channels {
            nlc[frame * *in_channels + channel] = mel.data()[channel * *frames + frame];
        }
    }
    conv1d_hf_impl(
        &nlc,
        *frames,
        *in_channels,
        weight_out_in_k,
        bias,
        stride,
        padding,
    )
}

fn conv1d_nlc_hf(
    input: &Tensor,
    weight_out_in_k: &Tensor,
    bias: &Tensor,
    stride: usize,
    padding: usize,
) -> Result<Tensor> {
    let (frames, in_channels) = input.as_matrix()?;
    conv1d_hf_impl(
        input.data(),
        frames,
        in_channels,
        weight_out_in_k,
        bias,
        stride,
        padding,
    )
}

fn conv1d_hf_impl(
    input_nlc: &[f32],
    frames: usize,
    in_channels: usize,
    weight_out_in_k: &Tensor,
    bias: &Tensor,
    stride: usize,
    padding: usize,
) -> Result<Tensor> {
    let [out_channels, weight_in, kernel] = weight_out_in_k.shape() else {
        return Err(InferError::Dimension(format!(
            "conv1d weight attendu [out,in,k], reçu {:?}",
            weight_out_in_k.shape()
        )));
    };
    if *weight_in != in_channels {
        return Err(InferError::Dimension(format!(
            "conv1d input channels={in_channels}, weight in={weight_in}"
        )));
    }
    validate_vector(bias, *out_channels, "conv1d bias")?;
    if stride == 0 {
        return Err(InferError::Dimension("conv1d stride nul".to_string()));
    }
    let padded = frames + 2 * padding;
    let out_frames = padded.checked_sub(*kernel).ok_or_else(|| {
        InferError::Dimension("conv1d kernel plus large que l'entrée".to_string())
    })? / stride
        + 1;
    // Parallélisé par frame de sortie (chaque sortie est une réduction
    // indépendante, MÊME ordre d'accumulation k→in_ch ⇒ byte-identique au
    // séquentiel). conv2 (1280×3×1280 MAC/frame) dominait l'encodeur en CPU
    // mono-thread (~7,4 GMAC).
    let (out_channels, kernel) = (*out_channels, *kernel);
    let bias = bias.data();
    let weight = weight_out_in_k.data();
    let mut out = vec![0.0_f32; out_frames * out_channels];
    out.par_chunks_mut(out_channels)
        .enumerate()
        .for_each(|(t_out, row)| {
            for (out_ch, slot) in row.iter_mut().enumerate() {
                let mut acc = bias[out_ch];
                for k in 0..kernel {
                    let padded_t = t_out * stride + k;
                    if padded_t < padding || padded_t >= frames + padding {
                        continue;
                    }
                    let t_in = padded_t - padding;
                    for in_ch in 0..in_channels {
                        let x = input_nlc[t_in * in_channels + in_ch];
                        let w = weight[(out_ch * in_channels + in_ch) * kernel + k];
                        acc += x * w;
                    }
                }
                *slot = acc;
            }
        });
    Tensor::from_vec(vec![out_frames, out_channels], out)
}

fn add_position_embeddings(x: &Tensor, positions: &Tensor) -> Result<Tensor> {
    let (seq, dim) = x.as_matrix()?;
    let [pos_seq, pos_dim] = positions.shape() else {
        return Err(InferError::Dimension(format!(
            "positions attendues rang 2, reçu {:?}",
            positions.shape()
        )));
    };
    if *pos_dim != dim || *pos_seq < seq {
        return Err(InferError::Dimension(format!(
            "positions shape={:?} incompatible avec x=[{seq},{dim}]",
            positions.shape()
        )));
    }
    let mut out = x.data().to_vec();
    for row in 0..seq {
        let base = row * dim;
        for col in 0..dim {
            out[base + col] += positions.data()[base + col];
        }
    }
    Tensor::from_vec(vec![seq, dim], out)
}

fn self_attention_forward(
    x: &Tensor,
    attn: &WhisperSelfAttention,
    config: &WhisperConfig,
    runtime: ForwardRuntime<'_>,
) -> Result<Tensor> {
    attention_forward(x, x, attn, config.encoder_attention_heads, false, runtime)
}

pub(super) fn attention_forward(
    query: &Tensor,
    key_value: &Tensor,
    attn: &WhisperSelfAttention,
    heads: usize,
    causal: bool,
    runtime: ForwardRuntime<'_>,
) -> Result<Tensor> {
    let q = attn.q_proj.forward_batched(query, runtime)?;
    let k = attn.k_proj.forward_batched(key_value, runtime)?;
    let v = attn.v_proj.forward_batched(key_value, runtime)?;
    let context = multi_head_attention(&q, &k, &v, heads, causal, runtime)?;
    attn.out_proj.forward_batched(&context, runtime)
}

pub(super) fn multi_head_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    heads: usize,
    causal: bool,
    runtime: ForwardRuntime<'_>,
) -> Result<Tensor> {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    if let Some(metal) = runtime.metal_executor() {
        if !causal && q.shape() == k.shape() && k.shape() == v.shape() {
            return metal.noncausal_attention_prefill(q, k, v, heads, heads);
        }
    }
    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    let _ = runtime;
    let (q_seq, dim) = q.as_matrix()?;
    let (kv_seq, k_dim) = k.as_matrix()?;
    if v.shape() != k.shape() || k_dim != dim {
        return Err(InferError::Dimension(format!(
            "attention q={:?}, k={:?}, v={:?}",
            q.shape(),
            k.shape(),
            v.shape()
        )));
    }
    if heads == 0 || dim % heads != 0 {
        return Err(InferError::Dimension(format!(
            "attention dim={dim} heads={heads}"
        )));
    }
    let head_dim = dim / heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0.0_f32; q_seq * dim];
    let mut scores = vec![0.0_f32; kv_seq];
    for head in 0..heads {
        let head_base = head * head_dim;
        for row_q in 0..q_seq {
            let mut max_score = f32::NEG_INFINITY;
            for row_k in 0..kv_seq {
                if causal && row_k > row_q {
                    scores[row_k] = f32::NEG_INFINITY;
                    continue;
                }
                let mut dot = 0.0_f32;
                for col in 0..head_dim {
                    dot += q.data()[row_q * dim + head_base + col]
                        * k.data()[row_k * dim + head_base + col];
                }
                let score = dot * scale;
                scores[row_k] = score;
                max_score = max_score.max(score);
            }
            let mut denom = 0.0_f32;
            for score in scores.iter_mut().take(kv_seq) {
                if *score == f32::NEG_INFINITY {
                    continue;
                }
                *score = (*score - max_score).exp();
                denom += *score;
            }
            for col in 0..head_dim {
                let mut acc = 0.0_f32;
                for (row_v, prob) in scores.iter().take(kv_seq).enumerate() {
                    if *prob == f32::NEG_INFINITY {
                        continue;
                    }
                    acc += (*prob / denom) * v.data()[row_v * dim + head_base + col];
                }
                out[row_q * dim + head_base + col] = acc;
            }
        }
    }
    Tensor::from_vec(vec![q_seq, dim], out)
}

fn feed_forward(
    x: &Tensor,
    fc1: &Linear,
    fc2: &Linear,
    runtime: ForwardRuntime<'_>,
) -> Result<Tensor> {
    let h = fc1.forward_batched(x, runtime)?;
    fc2.forward_batched(&gelu(&h), runtime)
}

#[cfg(test)]
mod tests;
