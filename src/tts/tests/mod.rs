use super::*;
use crate::sampling::DeterministicSampler;
use crate::tts_codec::TtsCodec;
use crate::{InferError, Result};
use safetensors::{serialize, Dtype, View};
use std::borrow::Cow;
use std::path::{Path, PathBuf};
use std::time::Instant;

mod assets;
mod codec;
mod golden;
mod guards;

/// Charge le codec TTS réel depuis le snapshot VoiceDesign en cache HF.
fn load_voicedesign_codec() -> Option<TtsCodec> {
    let codec = (|| {
        let model_dir = local_tts_snapshot(
            "RETI_QWEN3_TTS_VOICEDESIGN_DIR",
            "models--mlx-community--Qwen3-TTS-12Hz-1.7B-VoiceDesign-6bit",
        )?;
        let codec_weights = model_dir.join("speech_tokenizer/model.safetensors");
        let codec_config_path = model_dir.join("speech_tokenizer/config.json");
        let payload = SafetensorPayload::open(&codec_weights).ok()?;
        let codec_config: TtsCodecConfig = read_json(&codec_config_path).ok()?;
        TtsCodec::load(&payload, &codec_config).ok()
    })();
    crate::test_support::require_real_model(codec, "codec du snapshot Qwen3-TTS VoiceDesign")
}

/// Génère `n` frames de codes RVQ déterministes (16 quantizers, valeurs bornées).
///
/// Les valeurs restent < 500, donc valides pour tous les codebooks (≥ 2048
/// lignes). Le contenu importe peu pour mesurer le coût du décodeur : seul le
/// flux de features compte.
fn synthetic_codes(n: usize, quantizers: usize) -> Vec<Vec<i32>> {
    (0..n)
        .map(|time| {
            (0..quantizers)
                .map(|q| ((time * 31 + q * 17 + 7) % 500) as i32)
                .collect()
        })
        .collect()
}

/// Hash FNV-1a 64 bits sur les octets bruts des échantillons PCM.
fn fnv1a_f32(samples: &[f32]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for sample in samples {
        for byte in sample.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

/// Statistiques de dérive entre deux trains PCM de même longueur :
/// `(max_abs_diff, rms_diff, mean_abs_diff, signal_max_abs)`.
fn drift_stats(reference: &[f32], candidate: &[f32]) -> (f32, f32, f32, f32) {
    let mut max_abs = 0.0_f32;
    let mut sum_sq = 0.0_f64;
    let mut sum_abs = 0.0_f64;
    let mut signal_max = 0.0_f32;
    for (r, c) in reference.iter().zip(candidate.iter()) {
        let diff = (r - c).abs();
        max_abs = max_abs.max(diff);
        sum_sq += f64::from(diff) * f64::from(diff);
        sum_abs += f64::from(diff);
        signal_max = signal_max.max(r.abs());
    }
    let n = reference.len().max(1) as f64;
    (
        max_abs,
        (sum_sq / n).sqrt() as f32,
        (sum_abs / n) as f32,
        signal_max,
    )
}

fn clone_reference_assets() -> Result<(Vec<u8>, String)> {
    let wav = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../voices/reti-fr.wav");
    let txt = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../voices/reti-fr.txt");
    let wav_bytes = std::fs::read(&wav).map_err(|source| InferError::Io {
        path: wav.clone(),
        source,
    })?;
    let ref_text = std::fs::read_to_string(&txt).map_err(|source| InferError::Io {
        path: txt.clone(),
        source,
    })?;
    Ok((wav_bytes, ref_text))
}

/// Reconstruit la séquence d'ids ICL clone côté metal-rs (réf tronquée ⊕ cible
/// tronquée) — miroir exact de la construction de `live_clone_icl_inputs_*`.
fn clone_icl_rust_ids(rust: &TtsModel, text: &str) -> Result<Vec<i32>> {
    let ctx = rust
        .clone_ctx
        .as_ref()
        .ok_or_else(|| InferError::Config("ctx clone absent".to_string()))?;
    let target_chat = format!("<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n");
    let target_ids = rust.encode_ids(&target_chat)?;
    Ok(ctx
        .ref_text_ids
        .iter()
        .chain(&target_ids[3..target_ids.len() - 5])
        .copied()
        .collect::<Vec<_>>())
}

fn local_tts_snapshot(env_var: &str, cache_name: &str) -> Option<PathBuf> {
    if let Ok(path) = std::env::var(env_var) {
        let path = PathBuf::from(path);
        if path.join("config.json").is_file()
            && path.join("speech_tokenizer/model.safetensors").is_file()
        {
            return Some(path);
        }
    }
    let snapshot = crate::hf_resolve::hf_cache_dir_from_env().and_then(|hub| {
        let snapshots = hub.join(cache_name).join("snapshots");
        std::fs::read_dir(snapshots)
            .ok()?
            .flatten()
            .map(|entry| entry.path())
            .find(|path| {
                path.join("config.json").is_file()
                    && path.join("speech_tokenizer/model.safetensors").is_file()
            })
    });
    crate::test_support::require_real_model(snapshot, cache_name)
}

fn argmax_index(values: &[f32]) -> Result<usize> {
    values
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| {
            left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(idx, _)| idx)
        .ok_or_else(|| InferError::Dimension("argmax sur logits vides".to_string()))
}

fn max_abs_same_len(left: &[f32], right: &[f32]) -> Result<f32> {
    if left.len() != right.len() {
        return Err(InferError::Dimension(format!(
            "longueurs incompatibles: gauche={} droite={}",
            left.len(),
            right.len()
        )));
    }
    Ok(left
        .iter()
        .zip(right.iter())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f32, f32::max))
}

fn write(path: PathBuf, content: &str) -> Result<()> {
    std::fs::write(&path, content).map_err(|source| InferError::Io { path, source })
}

fn write_safetensors(path: &Path, names: &[&str]) -> Result<()> {
    let tensors = names
        .iter()
        .map(|name| {
            (
                *name,
                F32View {
                    shape: vec![1],
                    data: 1.0_f32.to_le_bytes().to_vec(),
                },
            )
        })
        .collect::<Vec<_>>();
    let buffer = serialize(tensors, None).map_err(|source| InferError::Safetensors {
        path: path.to_path_buf(),
        source,
    })?;
    std::fs::write(path, buffer).map_err(|source| InferError::Io {
        path: path.to_path_buf(),
        source,
    })
}

struct F32View {
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl View for F32View {
    fn dtype(&self) -> Dtype {
        Dtype::F32
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(&self.data)
    }

    fn data_len(&self) -> usize {
        self.data.len()
    }
}

fn model_config_json(kind: &str, clone: bool) -> String {
    let speaker = if clone {
        r#","speaker_encoder_config":{"enc_dim":2048}"#
    } else {
        ""
    };
    format!(
        r#"{{
          "talker_config":{{
            "head_dim":64,
            "hidden_size":128,
            "intermediate_size":256,
            "num_attention_heads":2,
            "num_key_value_heads":1,
            "num_hidden_layers":1,
            "num_code_groups":16,
            "rms_norm_eps":0.000001,
            "rope_theta":1000000.0,
            "vocab_size":3072,
            "text_vocab_size":151936,
            "text_hidden_size":128,
            "rope_scaling":{{"interleaved":true,"mrope_section":[16,24,24]}},
            "code_predictor_config":{{
              "head_dim":64,
              "hidden_size":64,
              "intermediate_size":128,
              "num_attention_heads":1,
              "num_key_value_heads":1,
              "num_hidden_layers":1,
              "num_code_groups":16,
              "rms_norm_eps":0.000001,
              "rope_theta":1000000.0,
              "vocab_size":3072
            }},
            "codec_bos_id":1,
            "codec_eos_token_id":2,
            "codec_pad_id":0,
            "codec_think_id":3,
            "codec_nothink_id":4,
            "codec_think_bos_id":5,
            "codec_think_eos_id":6,
            "codec_language_id":{{"french":42}}
          }},
          "quantization":{{"group_size":64,"bits":6}},
          "tts_bos_token_id":10,
          "tts_eos_token_id":11,
          "tts_pad_token_id":12,
          "tts_model_type":"{kind}"
          {speaker}
        }}"#
    )
}

fn codec_config_json(clone: bool) -> String {
    let encoder = if clone {
        r#","encoder_config":{
          "audio_channels":1,
          "codebook_dim":8,
          "codebook_size":16,
          "compress":2,
          "dilation_growth_rate":2,
          "head_dim":64,
          "hidden_size":64,
          "intermediate_size":128,
          "kernel_size":7,
          "last_kernel_size":7,
          "layer_scale_initial_scale":0.01,
          "num_attention_heads":1,
          "num_filters":32,
          "num_hidden_layers":1,
          "num_key_value_heads":1,
          "num_quantizers":16,
          "num_residual_layers":1,
          "num_semantic_quantizers":1,
          "residual_kernel_size":3,
          "rope_theta":10000.0,
          "sampling_rate":24000,
          "sliding_window":128,
          "_frame_rate":12.5,
          "upsampling_ratios":[8,6,5,4]
        }"#
    } else {
        ""
    };
    format!(
        r#"{{
          "decoder_config":{{
            "latent_dim":64,
            "codebook_dim":8,
            "codebook_size":16,
            "decoder_dim":64,
            "hidden_size":64,
            "intermediate_size":128,
            "head_dim":64,
            "num_attention_heads":1,
            "num_key_value_heads":1,
            "num_hidden_layers":1,
            "num_quantizers":16,
            "num_semantic_quantizers":1,
            "rms_norm_eps":0.000001,
            "rope_theta":10000.0,
            "layer_scale_initial_scale":0.01,
            "upsample_rates":[8,5,4,3],
            "upsampling_ratios":[2,2]
          }},
          "decode_upsample_rate":1920,
          "output_sample_rate":24000,
          "encoder_valid_num_quantizers":16
          {encoder}
        }}"#
    )
}
