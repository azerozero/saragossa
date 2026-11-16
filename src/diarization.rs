//! Diarisation hors ligne `nvidia/Nemotron-3-Diarization` (Streaming Sortformer).
//!
//! Poids publiés par NVIDIA sous licence OpenMDW-1.1 : usage, modification et
//! redistribution autorisés, à condition de conserver la licence et ses
//! notices avec les poids. Ce module n'embarque aucun poids.
//!
//! Chaîne, d'après transformers 5.22 (`modeling_nemotron3_diarization.py`) :
//!
//! 1. log-mel NeMo 128 bandes, trames de 10 ms (`frontend`) ;
//! 2. empilement de 8 trames (pas de conv2D) puis projection `1024 → 512` ;
//! 3. découpage en morceaux de 340 trames de 80 ms (+40 d'anticipation),
//!    précédés du cache des locuteurs et de la FIFO (`cache`) ;
//! 4. LayerNorm d'entrée, 31 couches Transformer pré-norme (d = 512, 8 têtes,
//!    RoPE rotate-half θ = 10⁴, positions remises à zéro à chaque morceau,
//!    FFN GELU 2048), LayerNorm finale ;
//! 5. projection `512 → 192`, conv1d sous-pixel (k = 3) qui remonte à 10 ms,
//!    tête `ReLU → dense → ReLU → 8 logits`, sigmoïde ;
//! 6. segments : probabilité > 0,5 par locuteur, sans fusion ni durée minimale.
//!
//! Sur Metal, les 31 couches passent par l'encodeur résident de Whisper (un
//! command buffer par morceau) avec la RoPE en place sur q et k.

mod cache;
mod frontend;

use crate::{
    gelu, layer_norm, load_float_tensors, ForwardRuntime, InferError, Linear, Result, Tensor,
};
use cache::{CachePolicy, SpeakerCache};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::Path;

const LAYER_NORM_EPS: f32 = 1.0e-5;
/// Seuil d'activité de `Nemotron3DiarizationProcessor.extract_speaker_dict`.
pub const DEFAULT_THRESHOLD: f32 = 0.5;

#[derive(Deserialize)]
struct RawConfig {
    audio_config: RawAudioConfig,
    head_config: RawHeadConfig,
    streaming_config: RawStreamingConfig,
    chunk_length: usize,
    chunk_right_context: usize,
    #[serde(default)]
    chunk_left_context: usize,
    fifo_length: usize,
    speaker_cache_update_period: usize,
}

#[derive(Deserialize)]
struct RawAudioConfig {
    hidden_act: String,
    hidden_size: usize,
    intermediate_size: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    num_hidden_layers: usize,
    num_mel_bins: usize,
    subsampling_factor: usize,
    rope_parameters: RawRope,
}

#[derive(Deserialize)]
struct RawRope {
    rope_theta: f32,
    #[serde(default = "one")]
    partial_rotary_factor: f32,
}

fn one() -> f32 {
    1.0
}

#[derive(Deserialize)]
struct RawHeadConfig {
    hidden_size: usize,
    num_speakers: usize,
}

#[derive(Deserialize)]
struct RawStreamingConfig {
    speaker_cache_length: usize,
    speaker_cache_silence_frames_per_speaker: usize,
    prediction_score_threshold: f32,
    latest_frames_score_boost: f32,
    strong_boost_rate: f32,
    weak_boost_rate: f32,
    min_positive_scores_rate: f32,
}

/// Configuration utile de `config.json`.
#[derive(Clone, Debug)]
pub struct DiarizationConfig {
    pub hidden_size: usize,
    pub heads: usize,
    pub layers: usize,
    pub num_mel_bins: usize,
    pub subsampling: usize,
    pub rope_theta: f32,
    pub head_hidden: usize,
    pub num_speakers: usize,
    pub chunk_length: usize,
    pub chunk_right_context: usize,
    pub chunk_left_context: usize,
    cache: CachePolicy,
}

impl DiarizationConfig {
    /// Lit et valide `config.json`.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le fichier manque, est invalide ou décrit une
    /// variante non prise en charge (activation, GQA, RoPE partielle).
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|source| InferError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let raw: RawConfig = serde_json::from_slice(&bytes)
            .map_err(|e| InferError::Config(format!("{}: {e}", path.display())))?;
        let a = &raw.audio_config;
        let unsupported = |what: String| Err(InferError::Config(format!("diarisation: {what}")));
        if a.hidden_act != "gelu" {
            return unsupported(format!("hidden_act={} (gelu attendu)", a.hidden_act));
        }
        if a.num_key_value_heads != a.num_attention_heads {
            return unsupported("GQA non prise en charge".into());
        }
        if a.rope_parameters.partial_rotary_factor != 1.0 {
            return unsupported("RoPE partielle non prise en charge".into());
        }
        if a.num_attention_heads == 0
            || a.hidden_size % a.num_attention_heads != 0
            || (a.hidden_size / a.num_attention_heads) % 2 != 0
            || a.intermediate_size == 0
            || a.subsampling_factor == 0
            || raw.chunk_length == 0
            || raw.head_config.num_speakers == 0
        {
            return unsupported("dimensions incohérentes".into());
        }
        let s = &raw.streaming_config;
        let speakers = raw.head_config.num_speakers;
        let budget = (s.speaker_cache_length / speakers)
            .checked_sub(s.speaker_cache_silence_frames_per_speaker)
            .ok_or_else(|| {
                InferError::Config("diarisation: cache des locuteurs trop court".into())
            })? as f32;
        Ok(Self {
            hidden_size: a.hidden_size,
            heads: a.num_attention_heads,
            layers: a.num_hidden_layers,
            num_mel_bins: a.num_mel_bins,
            subsampling: a.subsampling_factor,
            rope_theta: a.rope_parameters.rope_theta,
            head_hidden: raw.head_config.hidden_size,
            num_speakers: speakers,
            chunk_length: raw.chunk_length,
            chunk_right_context: raw.chunk_right_context,
            chunk_left_context: raw.chunk_left_context,
            cache: CachePolicy {
                cache_length: s.speaker_cache_length,
                fifo_length: raw.fifo_length,
                update_period: raw.speaker_cache_update_period,
                silence_frames: s.speaker_cache_silence_frames_per_speaker,
                score_threshold: s.prediction_score_threshold,
                latest_boost: s.latest_frames_score_boost,
                min_positive: (budget * s.min_positive_scores_rate).floor() as usize,
                strong_boosted: (budget * s.strong_boost_rate).floor() as usize,
                weak_boosted: (budget * s.weak_boost_rate).floor() as usize,
                num_speakers: speakers,
            },
        })
    }
}

struct Norm {
    weight: Tensor,
    bias: Tensor,
}

impl Norm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        layer_norm(x, &self.weight, &self.bias, LAYER_NORM_EPS)
    }
}

struct Layer {
    ln1: Norm,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    ln2: Norm,
    fc1: Linear,
    fc2: Linear,
}

/// Modèle de diarisation chargé en f32.
pub struct DiarizationModel {
    config: DiarizationConfig,
    embed: Linear,
    input_ln: Norm,
    layers: Vec<Layer>,
    final_ln: Norm,
    proj: Linear,
    /// Conv1d `[out, in, 3]` relue comme `[out, in * 3]` pour un GEMM im2col.
    upsampler: Linear,
    dense: Linear,
    out_proj: Linear,
    silence: Vec<f32>,
}

/// Intervalle de parole d'un locuteur, en trames de 10 ms (`end` exclus).
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct SpeakerSegment {
    pub speaker: usize,
    pub start_frame: usize,
    pub end_frame: usize,
}

impl SpeakerSegment {
    /// Début en secondes.
    #[must_use]
    pub fn start_seconds(&self) -> f64 {
        self.start_frame as f64 * FRAME_SECONDS
    }

    /// Fin en secondes.
    #[must_use]
    pub fn end_seconds(&self) -> f64 {
        self.end_frame as f64 * FRAME_SECONDS
    }
}

const FRAME_SECONDS: f64 = frontend::HOP as f64 / frontend::SAMPLE_RATE as f64;

/// Probabilités par trame de 10 ms et segments seuillés.
#[derive(Clone, Debug)]
pub struct Diarization {
    /// `[frames, num_speakers]`, locuteurs dans l'ordre d'arrivée.
    pub probabilities: Tensor,
    pub segments: Vec<SpeakerSegment>,
}

fn take(tensors: &mut HashMap<String, Tensor>, key: &str) -> Result<Tensor> {
    tensors
        .remove(key)
        .ok_or_else(|| InferError::MissingWeight(key.to_string()))
}

fn take_linear(tensors: &mut HashMap<String, Tensor>, prefix: &str) -> Result<Linear> {
    let weight = take(tensors, &format!("{prefix}.weight"))?;
    let bias = tensors.remove(&format!("{prefix}.bias"));
    Linear::new(weight, bias)
}

fn take_norm(tensors: &mut HashMap<String, Tensor>, prefix: &str) -> Result<Norm> {
    Ok(Norm {
        weight: take(tensors, &format!("{prefix}.weight"))?,
        bias: take(tensors, &format!("{prefix}.bias"))?,
    })
}

fn expect_shape(tensor: &Tensor, shape: &[usize], key: &str) -> Result<()> {
    if tensor.shape() == shape {
        Ok(())
    } else {
        Err(InferError::Shape(format!(
            "{key}: attendu {shape:?}, reçu {:?}",
            tensor.shape()
        )))
    }
}

impl DiarizationModel {
    /// Charge `config.json` et `model.safetensors` depuis un dossier local.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si un fichier, un poids ou une forme manque.
    pub fn from_model_dir(model_dir: impl AsRef<Path>) -> Result<Self> {
        let dir = model_dir.as_ref();
        let config = DiarizationConfig::from_file(dir.join("config.json"))?;
        let weights = dir.join("model.safetensors");
        if !weights.is_file() {
            return Err(InferError::MissingArtifact {
                path: weights,
                what: "model.safetensors",
            });
        }
        Self::from_tensors(config, load_float_tensors(weights)?)
    }

    /// Construit le modèle depuis des tenseurs déjà chargés (clés 🤗).
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si un poids manque ou n'a pas la forme attendue.
    pub fn from_tensors(config: DiarizationConfig, mut t: HashMap<String, Tensor>) -> Result<Self> {
        let (d, sub, mels) = (config.hidden_size, config.subsampling, config.num_mel_bins);
        let p = "model.audio_tower";
        let embed_w = take(&mut t, &format!("{p}.embedder.projection.weight"))?;
        expect_shape(&embed_w, &[d, sub * mels], "embedder.projection")?;
        let mut layers = Vec::with_capacity(config.layers);
        for i in 0..config.layers {
            let l = format!("{p}.layers.{i}");
            layers.push(Layer {
                ln1: take_norm(&mut t, &format!("{l}.layer_norm1"))?,
                q: take_linear(&mut t, &format!("{l}.self_attn.q_proj"))?,
                k: take_linear(&mut t, &format!("{l}.self_attn.k_proj"))?,
                v: take_linear(&mut t, &format!("{l}.self_attn.v_proj"))?,
                o: take_linear(&mut t, &format!("{l}.self_attn.o_proj"))?,
                ln2: take_norm(&mut t, &format!("{l}.layer_norm2"))?,
                fc1: take_linear(&mut t, &format!("{l}.mlp.fc1"))?,
                fc2: take_linear(&mut t, &format!("{l}.mlp.fc2"))?,
            });
        }
        let hh = config.head_hidden;
        let up_w = take(&mut t, "model.upsampler.conv.weight")?;
        expect_shape(&up_w, &[hh * sub, hh, 3], "upsampler.conv")?;
        let up_w = Tensor::from_vec(vec![hh * sub, hh * 3], up_w.data().to_vec())?;
        let up_b = take(&mut t, "model.upsampler.conv.bias")?;
        let silence = take(&mut t, "silence_embeds")?;
        expect_shape(&silence, &[d], "silence_embeds")?;
        let out_proj = take_linear(&mut t, "classifier.out_proj")?;
        if out_proj.weight().shape() != [config.num_speakers, hh] {
            return Err(InferError::Shape("classifier.out_proj".into()));
        }
        Ok(Self {
            embed: Linear::new(embed_w, None)?,
            input_ln: take_norm(&mut t, &format!("{p}.input_layer_norm"))?,
            layers,
            final_ln: take_norm(&mut t, &format!("{p}.layer_norm"))?,
            proj: take_linear(&mut t, "model.proj")?,
            upsampler: Linear::new(up_w, Some(up_b))?,
            dense: take_linear(&mut t, "classifier.dense")?,
            out_proj,
            silence: silence.data().to_vec(),
            config,
        })
    }

    /// Configuration du modèle.
    #[must_use]
    pub fn config(&self) -> &DiarizationConfig {
        &self.config
    }

    /// Diarise un signal mono 16 kHz : probabilités puis segments au seuil 0,5.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si l'audio est vide ou si un calcul échoue.
    pub fn diarize(&self, samples: &[f32], runtime: ForwardRuntime<'_>) -> Result<Diarization> {
        let probabilities = self.speaker_probabilities(samples, runtime)?;
        let segments = speaker_segments(&probabilities, DEFAULT_THRESHOLD)?;
        Ok(Diarization {
            probabilities,
            segments,
        })
    }

    /// Probabilités d'activité `[frames, num_speakers]`, une trame par 10 ms
    /// (`floor(len / 160)` trames, la trame de centrage masquée exclue).
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si l'audio est vide ou si un calcul échoue.
    pub fn speaker_probabilities(
        &self,
        samples: &[f32],
        runtime: ForwardRuntime<'_>,
    ) -> Result<Tensor> {
        let c = &self.config;
        let (d, sub, s) = (c.hidden_size, c.subsampling, c.num_speakers);
        let mel = frontend::log_mel(samples, c.num_mel_bins)?;
        let stacked = frontend::stack_frames(&mel.data, mel.frames_total, c.num_mel_bins, sub);
        let groups = mel.frames_total.div_ceil(sub);
        let stacked = Tensor::from_vec(vec![groups, sub * c.num_mel_bins], stacked)?;
        let embeds = self.embed.forward_batched(&stacked, runtime)?;
        // Une trame encodeur est valide si sa première trame mel l'est. Seule la
        // dernière peut ne pas l'être : elle reste une requête (la conv1d de la
        // tête lit sa voisine) mais sort des clés de l'attention.
        let frames_enc = groups;
        let valid_enc = (0..groups)
            .take_while(|g| g * sub < mel.frames_valid)
            .count();
        if valid_enc == 0 {
            return Err(InferError::Dimension(
                "diarisation: audio plus court qu'une trame".into(),
            ));
        }
        let embeds = embeds.data();

        #[cfg(all(target_os = "macos", feature = "metal"))]
        let resident = match runtime.metal_executor() {
            Some(metal) => Some(self.build_resident(metal)?),
            None => None,
        };

        let mut cache = SpeakerCache::new(c.cache.clone(), d);
        let mut probs = Vec::with_capacity(frames_enc * sub * s);
        let mut start = 0;
        while start < frames_enc {
            let end = (start + c.chunk_length).min(frames_enc);
            let context_start = start.saturating_sub(c.chunk_left_context);
            let lookback = start - context_start;
            let chunk_end = (end + c.chunk_right_context).min(frames_enc);
            let mut input = cache.prefix();
            let cached = input.len() / d;
            input.extend_from_slice(&embeds[context_start * d..chunk_end * d]);
            let rows = input.len() / d;
            // La trame masquée, si présente, est toujours la dernière de l'entrée.
            let key_rows = rows - chunk_end.saturating_sub(valid_enc);
            let input = Tensor::from_vec(vec![rows, d], input)?;
            let normed = self.input_ln.forward(&input)?;
            #[cfg(all(target_os = "macos", feature = "metal"))]
            let hidden = match (runtime.metal_executor(), resident.as_ref()) {
                (Some(metal), Some(enc)) => {
                    metal.encode_whisper_encoder_keys(&normed, enc, key_rows)?
                }
                _ => self.encode_cpu(normed, key_rows, runtime)?,
            };
            #[cfg(not(all(target_os = "macos", feature = "metal")))]
            let hidden = self.encode_cpu(normed, key_rows, runtime)?;
            let step_probs = self.head(&hidden, runtime)?;
            // Moyenne des 8 sous-trames : probabilités par trame encodeur du cache.
            let mut pooled: Vec<f32> = step_probs
                .chunks_exact(sub * s)
                .flat_map(|group| {
                    (0..s)
                        .map(move |k| (0..sub).map(|j| group[j * s + k]).sum::<f32>() / sub as f32)
                })
                .collect();
            // Probabilités nulles sur la trame masquée, comme `probs * mask`.
            pooled[key_rows * s..].fill(0.0);
            cache.update(input.data(), &pooled, &self.silence, end - start, lookback);
            let first = (cached + lookback) * sub * s;
            probs.extend_from_slice(&step_probs[first..first + (end - start) * sub * s]);
            start = end;
        }
        probs.truncate(mel.frames_valid * s);
        Tensor::from_vec(vec![mel.frames_valid, s], probs)
    }

    /// Oracle CPU des 31 couches (ou per-op Metal pour les GEMM) ; seules les
    /// `key_rows` premières trames servent de clés.
    fn encode_cpu(
        &self,
        mut h: Tensor,
        key_rows: usize,
        runtime: ForwardRuntime<'_>,
    ) -> Result<Tensor> {
        let heads = self.config.heads;
        let head_dim = self.config.hidden_size / heads;
        let inv_freq = crate::rope::inv_freq(head_dim, self.config.rope_theta);
        for layer in &self.layers {
            let x = layer.ln1.forward(&h)?;
            let mut q = layer.q.forward_batched(&x, runtime)?;
            let mut k = layer.k.forward_batched(&x, runtime)?;
            let v = layer.v.forward_batched(&x, runtime)?;
            rope_in_place(&mut q, heads, &inv_freq)?;
            rope_in_place(&mut k, heads, &inv_freq)?;
            let (rows, dim) = k.as_matrix()?;
            let (k, v) = if key_rows < rows {
                (
                    Tensor::from_vec(vec![key_rows, dim], k.data()[..key_rows * dim].to_vec())?,
                    Tensor::from_vec(vec![key_rows, dim], v.data()[..key_rows * dim].to_vec())?,
                )
            } else {
                (k, v)
            };
            let ctx = crate::whisper::multi_head_attention(&q, &k, &v, heads, false, runtime)?;
            h = h.add(&layer.o.forward_batched(&ctx, runtime)?)?;
            let x = layer.ln2.forward(&h)?;
            let ff = layer
                .fc2
                .forward_batched(&gelu(&layer.fc1.forward_batched(&x, runtime)?), runtime)?;
            h = h.add(&ff)?;
        }
        self.final_ln.forward(&h)
    }

    /// `[rows, d]` → sigmoïdes `[rows * 8, S]` : proj, conv1d sous-pixel, tête.
    fn head(&self, hidden: &Tensor, runtime: ForwardRuntime<'_>) -> Result<Vec<f32>> {
        let (rows, _) = hidden.as_matrix()?;
        let hh = self.config.head_hidden;
        let sub = self.config.subsampling;
        let x = self.proj.forward_batched(hidden, runtime)?;
        let cols = conv3_im2col(x.data(), rows, hh);
        let up = self
            .upsampler
            .forward_batched(&Tensor::from_vec(vec![rows, hh * 3], cols)?, runtime)?;
        // `[rows, sub * hh]` relu comme `[rows * sub, hh]` (reshape de la référence).
        let up = Tensor::from_vec(
            vec![rows * sub, hh],
            up.data().iter().map(|v| v.max(0.0)).collect(),
        )?;
        let x = self
            .dense
            .forward_batched(&up, runtime)?
            .map(|v| v.max(0.0));
        let logits = self.out_proj.forward_batched(&x, runtime)?;
        Ok(logits
            .data()
            .iter()
            .map(|&v| 1.0 / (1.0 + (-v).exp()))
            .collect())
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn build_resident(
        &self,
        metal: &crate::MetalExecutor,
    ) -> Result<crate::metal_backend::WhisperResidentEncoder> {
        use crate::metal_backend::{
            WhisperResidentEncoder, WhisperResidentLayer, WhisperResidentNorm, WhisperResidentProj,
        };
        let proj = |linear: &Linear| -> Result<WhisperResidentProj> {
            let crate::LinearWeight::Dense(tensor) = linear.weight() else {
                return Err(InferError::Config(
                    "diarisation résidente: projection non dense".into(),
                ));
            };
            Ok(WhisperResidentProj {
                weight: metal.cached_buffer_from_f32(tensor.data(), "diar_proj_w")?,
                bias: match linear.bias() {
                    Some(b) => Some(metal.cached_buffer_from_f32(b.data(), "diar_proj_b")?),
                    None => None,
                },
                // NOTE: GEMM f32 ; le chemin bf16 de Whisper n'est pas activé ici.
                weight_na: None,
                weight_bf16: None,
            })
        };
        let norm = |n: &Norm| -> Result<WhisperResidentNorm> {
            Ok(WhisperResidentNorm {
                weight: metal.cached_buffer_from_f32(n.weight.data(), "diar_ln_w")?,
                bias: metal.cached_buffer_from_f32(n.bias.data(), "diar_ln_b")?,
            })
        };
        let layers = self
            .layers
            .iter()
            .map(|l| {
                Ok(WhisperResidentLayer {
                    self_ln: norm(&l.ln1)?,
                    q: proj(&l.q)?,
                    k: proj(&l.k)?,
                    v: proj(&l.v)?,
                    o: proj(&l.o)?,
                    final_ln: norm(&l.ln2)?,
                    fc1: proj(&l.fc1)?,
                    fc2: proj(&l.fc2)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(WhisperResidentEncoder {
            layers,
            encoder_ln: norm(&self.final_ln)?,
            d_model: self.config.hidden_size,
            heads: self.config.heads,
            ffn_dim: self.layers.first().map_or(0, |l| l.fc1.weight().shape()[0]),
            eps: LAYER_NORM_EPS,
            rope_theta: Some(self.config.rope_theta),
        })
    }
}

/// RoPE rotate-half (convention 🤗 `rotate_half`), positions `0..rows`.
fn rope_in_place(x: &mut Tensor, heads: usize, inv_freq: &[f32]) -> Result<()> {
    let (rows, dim) = x.as_matrix()?;
    let head_dim = dim / heads;
    let half = head_dim / 2;
    let mut data = x.data().to_vec();
    for pos in 0..rows {
        for head in 0..heads {
            let base = pos * dim + head * head_dim;
            for (i, f) in inv_freq.iter().enumerate().take(half) {
                let (sin, cos) = (pos as f32 * f).sin_cos();
                let (a, b) = (data[base + i], data[base + i + half]);
                data[base + i] = a * cos - b * sin;
                data[base + i + half] = b * cos + a * sin;
            }
        }
    }
    *x = Tensor::from_vec(vec![rows, dim], data)?;
    Ok(())
}

/// im2col d'une conv1d k = 3, padding 1 : ligne `t` = `[x[t-1+k][c]]` rangée
/// `c * 3 + k`, l'ordre mémoire du poids `[out, in, k]`.
fn conv3_im2col(x: &[f32], rows: usize, channels: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; rows * channels * 3];
    for t in 0..rows {
        for k in 0..3 {
            let Some(src) = (t + k).checked_sub(1).filter(|&s| s < rows) else {
                continue;
            };
            for c in 0..channels {
                out[(t * channels + c) * 3 + k] = x[src * channels + c];
            }
        }
    }
    out
}

/// Segments `probabilité > threshold` par locuteur, triés par début puis
/// locuteur (`extract_speaker_dict` de la référence).
///
/// # Errors
///
/// Renvoie une erreur si `probabilities` n'est pas une matrice.
pub fn speaker_segments(probabilities: &Tensor, threshold: f32) -> Result<Vec<SpeakerSegment>> {
    let (frames, speakers) = probabilities.as_matrix()?;
    let p = probabilities.data();
    let mut segments = Vec::new();
    for speaker in 0..speakers {
        let mut open = None;
        for t in 0..=frames {
            let active = t < frames && p[t * speakers + speaker] > threshold;
            match (active, open) {
                (true, None) => open = Some(t),
                (false, Some(start_frame)) => {
                    segments.push(SpeakerSegment {
                        speaker,
                        start_frame,
                        end_frame: t,
                    });
                    open = None;
                }
                _ => {}
            }
        }
    }
    segments.sort_by_key(|s| (s.start_frame, s.speaker));
    Ok(segments)
}

/// Segments au format RTTM (`SPEAKER uri 1 début durée <NA> <NA> speaker_k <NA> <NA>`).
#[must_use]
pub fn to_rttm(segments: &[SpeakerSegment], uri: &str) -> String {
    let mut out = String::new();
    for s in segments {
        let _ = writeln!(
            out,
            "SPEAKER {uri} 1 {:.2} {:.2} <NA> <NA> speaker_{} <NA> <NA>",
            s.start_seconds(),
            s.end_seconds() - s.start_seconds(),
            s.speaker
        );
    }
    out
}

/// Segments en JSON : `[{"speaker", "start", "end"}]`, secondes.
#[must_use]
pub fn to_json(segments: &[SpeakerSegment]) -> serde_json::Value {
    serde_json::Value::Array(
        segments
            .iter()
            .map(|s| {
                serde_json::json!({
                    "speaker": s.speaker,
                    "start": (s.start_seconds() * 100.0).round() / 100.0,
                    "end": (s.end_seconds() * 100.0).round() / 100.0,
                })
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests;
