//! Contrat local des checkpoints Qwen3-TTS pour le port Rust pur.
//!
//! Ce module ne synthétise pas encore l'audio. Il pose la frontière testable du
//! port TTS : configs typées, chemins tokenizer BPE, payloads talker/codec
//! et forward talker mesurable contre l'oracle mlx-rs.

use crate::{
    catalog::read_safetensors_keys,
    decoder::DecoderTensor,
    safetensor::bytes_to_dense_f32,
    sampling::{sample_token_top_k_top_p, DeterministicSampler},
    tts_codec::{TtsCodec, TtsCodecStreamState},
    tts_mimi::TtsMimiEncoder,
    tts_speaker::TtsSpeakerEncoder,
    AffineQuantizedTensor, CausalDecoder, CausalDecoderCache, CausalDecoderConfig, InferError,
    LinearWeight, Result, Tensor,
};
use safetensors::Dtype;
use serde::Deserialize;
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::{AddedToken, Tokenizer};

pub const DEFAULT_INSTRUCT: &str = "Voix féminine française Réti-01 : claire, posée et professionnelle, articulation nette, ton neutre et rassurant, débit efficace.";
const CLONE_GENERATION_HARD_CAP: usize = 800;
const CLONE_DEFAULT_MIN_FRAMES: usize = 75;
const CLONE_DEFAULT_FRAMES_PER_TOKEN: usize = 6;
const CLONE_SAMPLE_TEMPERATURE: f32 = 0.9;
const CLONE_SAMPLE_TOP_K: usize = 50;
const CLONE_SAMPLE_TOP_P: f32 = 1.0;
const CLONE_SAMPLE_REPETITION_PENALTY: f32 = 1.05;
const DEFAULT_REPEAT_FRAME_STOP: usize = 8;

const QWEN3_TTS_SPECIAL_TOKENS: &[(&str, bool)] = &[
    ("<|endoftext|>", true),
    ("<|im_start|>", true),
    ("<|im_end|>", true),
    ("<|object_ref_start|>", true),
    ("<|object_ref_end|>", true),
    ("<|box_start|>", true),
    ("<|box_end|>", true),
    ("<|quad_start|>", true),
    ("<|quad_end|>", true),
    ("<|vision_start|>", true),
    ("<|vision_end|>", true),
    ("<|vision_pad|>", true),
    ("<|image_pad|>", true),
    ("<|video_pad|>", true),
    ("<tool_call>", false),
    ("</tool_call>", false),
    ("<|fim_prefix|>", false),
    ("<|fim_middle|>", false),
    ("<|fim_suffix|>", false),
    ("<|fim_pad|>", false),
    ("<|repo_name|>", false),
    ("<|file_sep|>", false),
    ("<tool_response>", false),
    ("</tool_response>", false),
    ("<think>", false),
    ("</think>", false),
    ("<|audio_start|>", true),
    ("<|audio_end|>", true),
    ("<tts_pad>", true),
    ("<tts_text_bos>", true),
    ("<tts_text_eod>", true),
    ("<tts_text_bos_single>", true),
    ("<|audio_pad|>", true),
];

#[derive(Debug, Clone)]
pub struct TtsAssets {
    /// Répertoire racine du snapshot Qwen3-TTS local.
    pub model_dir: PathBuf,
    /// Config racine du talker Qwen3-TTS.
    pub model_config: TtsModelConfig,
    /// Config du speech tokenizer.
    pub codec_config: TtsCodecConfig,
    /// Vocabulaire BPE Qwen3-TTS (`vocab.json`).
    pub vocab_path: PathBuf,
    /// Merges BPE Qwen3-TTS (`merges.txt`).
    pub merges_path: PathBuf,
    /// Poids racine (`model.safetensors`) : talker + éventuellement speaker encoder.
    pub talker_weights: PathBuf,
    /// Poids du speech tokenizer (`speech_tokenizer/model.safetensors`).
    pub codec_weights: PathBuf,
    /// Catalogue des clés du safetensors racine.
    pub talker_catalog: TtsTalkerCatalog,
    /// Catalogue des clés du speech tokenizer.
    pub codec_catalog: TtsCodecCatalog,
}

impl TtsAssets {
    /// Charge le contrat d'un snapshot Qwen3-TTS local sans charger les poids.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si un artefact obligatoire manque, si une config JSON
    /// est invalide, ou si les safetensors ne ressemblent pas à un checkpoint TTS.
    pub fn load_local(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        if !model_dir.is_dir() {
            return Err(InferError::MissingArtifact {
                path: model_dir.to_path_buf(),
                what: "tts model_dir",
            });
        }

        let config_path = model_dir.join("config.json");
        let vocab_path = model_dir.join("vocab.json");
        let merges_path = model_dir.join("merges.txt");
        let talker_weights = model_dir.join("model.safetensors");
        let speech_dir = model_dir.join("speech_tokenizer");
        let codec_config_path = speech_dir.join("config.json");
        let codec_weights = speech_dir.join("model.safetensors");

        require_file(&config_path, "config.json")?;
        require_file(&vocab_path, "vocab.json")?;
        require_file(&merges_path, "merges.txt")?;
        require_file(&talker_weights, "model.safetensors")?;
        require_file(&codec_config_path, "speech_tokenizer/config.json")?;
        require_file(&codec_weights, "speech_tokenizer/model.safetensors")?;

        let model_config = read_json(&config_path)?;
        let codec_config = read_json(&codec_config_path)?;
        let talker_catalog = TtsTalkerCatalog::from_path(&talker_weights)?;
        let codec_catalog = TtsCodecCatalog::from_path(&codec_weights)?;

        if !talker_catalog.has_talker_weights {
            return Err(InferError::MissingArtifact {
                path: talker_weights,
                what: "talker.* weights",
            });
        }
        if !codec_catalog.has_decoder_weights {
            return Err(InferError::MissingArtifact {
                path: codec_weights,
                what: "speech_tokenizer decoder weights",
            });
        }

        Ok(Self {
            model_dir: model_dir.to_path_buf(),
            model_config,
            codec_config,
            vocab_path,
            merges_path,
            talker_weights,
            codec_weights,
            talker_catalog,
            codec_catalog,
        })
    }

    #[must_use]
    pub fn model_kind(&self) -> TtsModelKind {
        self.model_config.model_kind()
    }

    #[must_use]
    pub fn clone_capable(&self) -> bool {
        self.model_config.speaker_encoder_config.is_some()
            && self.codec_config.encoder_config.is_some()
            && self.talker_catalog.has_speaker_encoder_weights
            && self.codec_catalog.has_encoder_weights
    }
}

#[derive(Debug)]
pub struct TtsModel {
    pub assets: TtsAssets,
    tokenizer: Tokenizer,
    text_embedding: Tensor,
    codec_embedding: Tensor,
    text_projection_fc1: crate::Linear,
    text_projection_fc2: crate::Linear,
    talker: CausalDecoder,
    code_predictor_projection: crate::Linear,
    code_predictor: CausalDecoder,
    code_predictor_heads: Vec<crate::Linear>,
    code_predictor_embeddings: Vec<Tensor>,
    codec: TtsCodec,
    codec_payload: TtsPayloadSummary,
    clone_ctx: Option<TtsCloneContext>,
}

enum TtsStreamDecodeState {
    Incremental(TtsCodecStreamState),
    FullPrefixDelta { emitted: usize },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TtsPayloadSummary {
    pub talker_tensor_count: usize,
    pub codec_tensor_count: usize,
    pub codec_payload_bytes: u64,
    pub codec_payload_bytes_read: u64,
    pub codec_payload_checksum: u64,
}

#[derive(Debug)]
pub struct TtsForwardOutput {
    pub cache: CausalDecoderCache,
    pub logits: Tensor,
    pub final_state: Tensor,
}

#[derive(Debug)]
pub struct TtsSynthesisOutput {
    pub codes: Vec<Vec<i32>>,
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

#[derive(Clone, Copy, Debug)]
struct TtsSampleParams {
    temperature: f32,
    top_k: usize,
    top_p: f32,
    repetition_penalty: f32,
    seed: u64,
}

#[derive(Debug)]
struct PreparedVoiceDesign {
    input: Tensor,
    trailing: Tensor,
    tts_pad: Tensor,
}

#[derive(Debug)]
struct TtsCloneContext {
    ref_codes: Vec<Vec<i32>>,
    speaker_embed: Tensor,
    ref_text_ids: Vec<i32>,
    ref_codec_embed: Option<Tensor>,
    mode: TtsCloneMode,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TtsCloneMode {
    Icl,
    XVectorOnly,
}

impl TtsCloneMode {
    fn is_xvec_only(self) -> bool {
        matches!(self, Self::XVectorOnly)
    }

    fn label(self) -> &'static str {
        match self {
            Self::Icl => "clone-icl",
            Self::XVectorOnly => "clone-xvec-only",
        }
    }
}

impl TtsModel {
    /// Charge les vrais payloads talker/codec Qwen3-TTS depuis un snapshot local.
    ///
    /// Le codec est validé en lisant tous ses payloads par offsets pour prouver
    /// que le checkpoint complet est disponible ; le forward implémenté ici ne
    /// consomme encore que le talker.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le snapshot est incomplet ou si un poids attendu est
    /// absent/incompatible.
    pub fn load_local(model_dir: impl AsRef<Path>) -> Result<Self> {
        let assets = TtsAssets::load_local(model_dir)?;
        let tokenizer = load_qwen3_tts_tokenizer(&assets.model_dir)?;
        let talker_header = SafetensorPayload::open(&assets.talker_weights)?;
        let codec_header = SafetensorPayload::open(&assets.codec_weights)?;

        let mut tensors = HashMap::new();
        let mut cp_tensors = HashMap::new();
        let q_group = usize::try_from(assets.model_config.quantization.group_size)
            .map_err(|_| InferError::Config("group_size TTS négatif".to_string()))?;
        let q_bits = usize::try_from(assets.model_config.quantization.bits)
            .map_err(|_| InferError::Config("bits TTS négatif".to_string()))?;

        let text_embedding =
            talker_header.read_dense_tensor("talker.model.text_embedding.weight")?;
        let codec_embedding =
            talker_header.read_dense_tensor("talker.model.codec_embedding.weight")?;
        let text_projection_fc1 = read_linear_layer(
            &talker_header,
            "talker.text_projection.linear_fc1",
            q_group,
            q_bits,
        )?;
        let text_projection_fc2 = read_linear_layer(
            &talker_header,
            "talker.text_projection.linear_fc2",
            q_group,
            q_bits,
        )?;
        let code_predictor_projection = read_linear_layer(
            &talker_header,
            "talker.code_predictor.small_to_mtp_projection",
            q_group,
            q_bits,
        )?;
        let mut code_predictor_embeddings = Vec::new();
        for index in 0..(assets.model_config.talker_config.num_code_groups - 1) {
            code_predictor_embeddings.push(talker_header.read_dense_tensor(&format!(
                "talker.code_predictor.model.codec_embedding.{index}.weight"
            ))?);
        }

        tensors.insert(
            "embed_tokens.weight".to_string(),
            DecoderTensor::Dense(text_embedding.clone()),
        );
        tensors.insert(
            "norm.weight".to_string(),
            DecoderTensor::Dense(talker_header.read_dense_tensor("talker.model.norm.weight")?),
        );
        insert_linear(
            &mut tensors,
            &talker_header,
            "talker.codec_head",
            "lm_head",
            q_group,
            q_bits,
        )?;

        for layer in 0..assets.model_config.talker_config.num_hidden_layers {
            let source = format!("talker.model.layers.{layer}");
            let target = format!("layers.{layer}");
            copy_dense(
                &mut tensors,
                &talker_header,
                &format!("{source}.input_layernorm.weight"),
                &format!("{target}.input_layernorm.weight"),
            )?;
            copy_dense(
                &mut tensors,
                &talker_header,
                &format!("{source}.post_attention_layernorm.weight"),
                &format!("{target}.post_attention_layernorm.weight"),
            )?;
            copy_dense(
                &mut tensors,
                &talker_header,
                &format!("{source}.self_attn.q_norm.weight"),
                &format!("{target}.self_attn.q_norm.weight"),
            )?;
            copy_dense(
                &mut tensors,
                &talker_header,
                &format!("{source}.self_attn.k_norm.weight"),
                &format!("{target}.self_attn.k_norm.weight"),
            )?;
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                insert_linear(
                    &mut tensors,
                    &talker_header,
                    &format!("{source}.self_attn.{proj}"),
                    &format!("{target}.self_attn.{proj}"),
                    q_group,
                    q_bits,
                )?;
            }
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                insert_linear(
                    &mut tensors,
                    &talker_header,
                    &format!("{source}.mlp.{proj}"),
                    &format!("{target}.mlp.{proj}"),
                    q_group,
                    q_bits,
                )?;
            }
        }
        let cp_cfg = &assets.model_config.talker_config.code_predictor_config;
        let cp_hidden = usize::try_from(cp_cfg.hidden_size)
            .map_err(|_| InferError::Config("hidden_size code_predictor négatif".to_string()))?;
        cp_tensors.insert(
            "embed_tokens.weight".to_string(),
            DecoderTensor::Dense(Tensor::zeros(vec![1, cp_hidden])?),
        );
        cp_tensors.insert(
            "norm.weight".to_string(),
            DecoderTensor::Dense(
                talker_header.read_dense_tensor("talker.code_predictor.model.norm.weight")?,
            ),
        );
        insert_linear(
            &mut cp_tensors,
            &talker_header,
            "talker.code_predictor.lm_head.0",
            "lm_head",
            q_group,
            q_bits,
        )?;
        let mut code_predictor_heads = Vec::new();
        for index in 0..(assets.model_config.talker_config.num_code_groups - 1) {
            code_predictor_heads.push(read_linear_layer(
                &talker_header,
                &format!("talker.code_predictor.lm_head.{index}"),
                q_group,
                q_bits,
            )?);
        }
        for layer in 0..cp_cfg.num_hidden_layers {
            let source = format!("talker.code_predictor.model.layers.{layer}");
            let target = format!("layers.{layer}");
            copy_dense(
                &mut cp_tensors,
                &talker_header,
                &format!("{source}.input_layernorm.weight"),
                &format!("{target}.input_layernorm.weight"),
            )?;
            copy_dense(
                &mut cp_tensors,
                &talker_header,
                &format!("{source}.post_attention_layernorm.weight"),
                &format!("{target}.post_attention_layernorm.weight"),
            )?;
            copy_dense(
                &mut cp_tensors,
                &talker_header,
                &format!("{source}.self_attn.q_norm.weight"),
                &format!("{target}.self_attn.q_norm.weight"),
            )?;
            copy_dense(
                &mut cp_tensors,
                &talker_header,
                &format!("{source}.self_attn.k_norm.weight"),
                &format!("{target}.self_attn.k_norm.weight"),
            )?;
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                insert_linear(
                    &mut cp_tensors,
                    &talker_header,
                    &format!("{source}.self_attn.{proj}"),
                    &format!("{target}.self_attn.{proj}"),
                    q_group,
                    q_bits,
                )?;
            }
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                insert_linear(
                    &mut cp_tensors,
                    &talker_header,
                    &format!("{source}.mlp.{proj}"),
                    &format!("{target}.mlp.{proj}"),
                    q_group,
                    q_bits,
                )?;
            }
        }

        let decoder_config = CausalDecoderConfig {
            rms_eps: assets.model_config.talker_config.rms_norm_eps,
            rope_theta: Some(assets.model_config.talker_config.rope_theta),
            num_hidden_layers: usize::try_from(assets.model_config.talker_config.num_hidden_layers)
                .map_err(|_| InferError::Config("num_hidden_layers TTS négatif".to_string()))?,
            num_attention_heads: usize::try_from(
                assets.model_config.talker_config.num_attention_heads,
            )
            .map_err(|_| InferError::Config("num_attention_heads TTS négatif".to_string()))?,
            num_key_value_heads: usize::try_from(
                assets.model_config.talker_config.num_key_value_heads,
            )
            .map_err(|_| InferError::Config("num_key_value_heads TTS négatif".to_string()))?,
            num_global_key_value_heads: None,
            head_dim: Some(
                usize::try_from(assets.model_config.talker_config.head_dim)
                    .map_err(|_| InferError::Config("head_dim TTS négatif".to_string()))?,
            ),
            global_head_dim: None,
            rope_dims: Some(
                usize::try_from(assets.model_config.talker_config.head_dim)
                    .map_err(|_| InferError::Config("head_dim TTS négatif".to_string()))?,
            ),
            rope_full_dims: None,
            rope_sliding_dims: None,
            attn_output_gate: false,
            layer_types: Vec::new(),
            full_attention_interval: None,
            linear_num_value_heads: None,
            linear_num_key_heads: None,
            linear_key_head_dim: None,
            linear_value_head_dim: None,
            linear_conv_kernel_dim: None,
            num_experts: None,
            num_experts_per_tok: 1,
            moe_intermediate_size: 0,
            shared_expert_intermediate_size: 0,
            embed_scale: None,
            rope_local_base_freq: None,
            rope_full_base_freq: None,
            rope_position_scale: None,
            sliding_window: None,
            sliding_window_pattern: None,
            attention_k_eq_v: false,
            attention_value_norm: false,
            parallel_moe: false,
            final_logit_softcapping: None,
            query_pre_attn_scalar: None,
            activation: crate::Activation::Silu,
            rope_style: crate::RopeStyle::Halves,
            is_gemma4: false,
        };
        let mut talker = CausalDecoder::from_decoder_tensors(tensors, decoder_config)?;
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            talker = talker.with_metal_runtime()?;
        }
        let cp_config =
            CausalDecoderConfig {
                rms_eps: cp_cfg.rms_norm_eps,
                rope_theta: Some(cp_cfg.rope_theta),
                num_hidden_layers: usize::try_from(cp_cfg.num_hidden_layers).map_err(|_| {
                    InferError::Config("num_hidden_layers code_predictor négatif".to_string())
                })?,
                num_attention_heads: usize::try_from(cp_cfg.num_attention_heads).map_err(|_| {
                    InferError::Config("num_attention_heads code_predictor négatif".to_string())
                })?,
                num_key_value_heads: usize::try_from(cp_cfg.num_key_value_heads).map_err(|_| {
                    InferError::Config("num_key_value_heads code_predictor négatif".to_string())
                })?,
                num_global_key_value_heads: None,
                head_dim: Some(usize::try_from(cp_cfg.head_dim).map_err(|_| {
                    InferError::Config("head_dim code_predictor négatif".to_string())
                })?),
                global_head_dim: None,
                rope_dims: Some(usize::try_from(cp_cfg.head_dim).map_err(|_| {
                    InferError::Config("head_dim code_predictor négatif".to_string())
                })?),
                rope_full_dims: None,
                rope_sliding_dims: None,
                attn_output_gate: false,
                layer_types: Vec::new(),
                full_attention_interval: None,
                linear_num_value_heads: None,
                linear_num_key_heads: None,
                linear_key_head_dim: None,
                linear_value_head_dim: None,
                linear_conv_kernel_dim: None,
                num_experts: None,
                num_experts_per_tok: 1,
                moe_intermediate_size: 0,
                shared_expert_intermediate_size: 0,
                embed_scale: None,
                rope_local_base_freq: None,
                rope_full_base_freq: None,
                rope_position_scale: None,
                sliding_window: None,
                sliding_window_pattern: None,
                attention_k_eq_v: false,
                attention_value_norm: false,
                parallel_moe: false,
                final_logit_softcapping: None,
                query_pre_attn_scalar: None,
                activation: crate::Activation::Silu,
                rope_style: crate::RopeStyle::Halves,
                is_gemma4: false,
            };
        let mut code_predictor = CausalDecoder::from_decoder_tensors(cp_tensors, cp_config)?;
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            code_predictor = code_predictor.with_metal_runtime()?;
        }

        let codec = TtsCodec::load(&codec_header, &assets.codec_config)?;

        let codec_read = codec_header.read_payload_summary()?;
        let codec_payload = TtsPayloadSummary {
            talker_tensor_count: talker_header.entries.len(),
            codec_tensor_count: codec_header.entries.len(),
            codec_payload_bytes: codec_read.bytes,
            codec_payload_bytes_read: codec_read.bytes_read,
            codec_payload_checksum: codec_read.checksum,
        };

        Ok(Self {
            assets,
            tokenizer,
            text_embedding,
            codec_embedding,
            text_projection_fc1,
            text_projection_fc2,
            talker,
            code_predictor_projection,
            code_predictor,
            code_predictor_heads,
            code_predictor_embeddings,
            codec,
            codec_payload,
            clone_ctx: None,
        })
    }

    /// Charge Qwen3-TTS Base avec une référence clone Base/ICL.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le snapshot ou la référence voix sont invalides.
    pub fn load_clone_local(
        model_dir: impl AsRef<Path>,
        ref_wav: &[u8],
        ref_text: &str,
    ) -> Result<Self> {
        Self::load_clone_local_with_mode(model_dir, ref_wav, Some(ref_text), TtsCloneMode::Icl)
    }

    /// Charge Qwen3-TTS Base avec une empreinte locuteur seule (`x_vector_only_mode`).
    ///
    /// Ce mode garde une voix stable à partir du WAV de référence, sans injecter
    /// transcript + codes audio de référence dans chaque prompt.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le snapshot ou la référence voix sont invalides.
    pub fn load_clone_xvec_local(model_dir: impl AsRef<Path>, ref_wav: &[u8]) -> Result<Self> {
        Self::load_clone_local_with_mode(model_dir, ref_wav, None, TtsCloneMode::XVectorOnly)
    }

    fn load_clone_local_with_mode(
        model_dir: impl AsRef<Path>,
        ref_wav: &[u8],
        ref_text: Option<&str>,
        mode: TtsCloneMode,
    ) -> Result<Self> {
        let mut model = Self::load_local(model_dir)?;
        if model.assets.model_kind() != TtsModelKind::Base || !model.assets.clone_capable() {
            return Err(InferError::Config(format!(
                "snapshot TTS clone attendu Base clone-capable, reçu {:?}",
                model.assets.model_kind()
            )));
        }
        let ref_text = ref_text.unwrap_or("").trim();
        if !mode.is_xvec_only() && ref_text.is_empty() {
            return Err(InferError::Config("transcript ref clone vide".to_string()));
        }
        let pcm = crate::tts_clone::load_wav_24k(ref_wav)?;
        if pcm.is_empty() {
            return Err(InferError::Config("ref WAV clone vide".to_string()));
        }
        let mel = crate::tts_clone::log_mel_24k(&pcm)?;
        let speaker = TtsSpeakerEncoder::load(&model.assets.model_dir, &model.assets.model_config)?;
        let speaker_embed = speaker.embed_mel(&mel)?;
        let (ref_codes, ref_text_ids, ref_codec_embed) = if mode.is_xvec_only() {
            (Vec::new(), Vec::new(), None)
        } else {
            let mimi = TtsMimiEncoder::load(&model.assets.model_dir, &model.assets.codec_config)?;
            let ref_codes = mimi.encode_pcm_24k(&pcm)?;
            let ref_chat = format!("<|im_start|>assistant\n{ref_text}<|im_end|>\n");
            let ref_ids = model.encode_ids(&ref_chat)?;
            if ref_ids.len() < 5 {
                return Err(InferError::Config(
                    "prompt ref clone trop court".to_string(),
                ));
            }
            let ref_text_ids = ref_ids[3..ref_ids.len() - 2].to_vec();
            let ref_codec_embed = model.ref_codec_embed_sum(&ref_codes)?;
            (ref_codes, ref_text_ids, Some(ref_codec_embed))
        };
        model.clone_ctx = Some(TtsCloneContext {
            ref_codes,
            speaker_embed,
            ref_text_ids,
            ref_codec_embed,
            mode,
        });
        Ok(model)
    }

    #[must_use]
    pub fn payload_summary(&self) -> &TtsPayloadSummary {
        &self.codec_payload
    }

    /// Exécute le forward talker autoregressif sur le préfixe VoiceDesign.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si la tokenisation, la préparation des embeddings ou le
    /// forward échoue.
    pub fn forward_voicedesign_prefix(&self, text: &str) -> Result<TtsForwardOutput> {
        let prepared = self.prepare_voicedesign_inputs(text)?;
        let (cache, final_state) = self.talker.prefill_cache_from_embeddings(&prepared.input)?;
        let logits = self.talker.logits_from_final_state(&final_state)?;
        Ok(TtsForwardOutput {
            cache,
            logits,
            final_state,
        })
    }

    /// Synthétise un texte en PCM f32 mono avec décodage greedy déterministe.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si la génération des codes ou le codec échoue.
    /// Renvoie la fréquence d'échantillonnage du codec (Hz) — pour le playback
    /// streaming où le taux est requis avant la fin de la synthèse.
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.codec.sample_rate()
    }

    pub fn synthesize_greedy(&self, text: &str, max_frames: usize) -> Result<TtsSynthesisOutput> {
        let codes = self.generate_codes_greedy(text, max_frames)?;
        let samples = self.decode_codes_for_mode(&codes)?;
        Ok(TtsSynthesisOutput {
            codes,
            samples,
            sample_rate: self.codec.sample_rate(),
        })
    }

    /// Synthétise avec la politique de décodage adaptée au mode.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si la génération des codes ou le codec échoue.
    pub fn synthesize_default(&self, text: &str, max_frames: usize) -> Result<TtsSynthesisOutput> {
        let codes = if self.clone_ctx.is_some() {
            self.generate_codes_clone_sampled(text, max_frames)?
        } else {
            self.generate_codes_greedy(text, max_frames)?
        };
        let samples = self.decode_codes_for_mode(&codes)?;
        Ok(TtsSynthesisOutput {
            codes,
            samples,
            sample_rate: self.codec.sample_rate(),
        })
    }

    /// Synthétise en streaming avec la politique de décodage adaptée au mode.
    ///
    /// VoiceDesign conserve le chemin greedy historique. Le clone Base/ICL passe
    /// par le sampling + cap adaptatif utilisés par [`Self::synthesize_default`],
    /// ce qui évite de réintroduire le chemin greedy clone qui boucle.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si la génération, le codec, ou `on_audio` échoue.
    pub fn synthesize_default_streaming<F>(
        &self,
        text: &str,
        max_frames: usize,
        on_audio: F,
    ) -> Result<TtsSynthesisOutput>
    where
        F: FnMut(&[f32]) -> Result<()>,
    {
        if self.clone_ctx.is_some() {
            let target_tokens = self.encode_ids(text)?.len();
            let effective_max_frames = clone_effective_frame_cap(max_frames, target_tokens);
            let params = TtsSampleParams {
                temperature: CLONE_SAMPLE_TEMPERATURE,
                top_k: CLONE_SAMPLE_TOP_K,
                top_p: CLONE_SAMPLE_TOP_P,
                repetition_penalty: CLONE_SAMPLE_REPETITION_PENALTY,
                seed: clone_sample_seed(),
            };
            let trace = tts_generation_trace_enabled();
            return self.synthesize_streaming_from_frames(
                |on_frame| {
                    self.generate_codes_clone_sampled_trace(
                        text,
                        max_frames,
                        effective_max_frames,
                        params,
                        trace,
                        on_frame,
                    )
                },
                on_audio,
            );
        }
        self.synthesize_greedy_streaming(text, max_frames, on_audio)
    }

    /// Synthétise en STREAMING : `on_audio(&[f32])` est appelé avec chaque nouveau
    /// segment PCM dès qu'un lot de frames est généré puis décodé, pour un premier
    /// son immédiat (TTFA basse). La concaténation des segments est **byte-identique**
    /// à [`Self::synthesize_greedy`] : le codec étant entièrement causal, décoder le
    /// préfixe [0..N] puis émettre le delta reproduit exactement l'audio batch.
    ///
    /// Stratégie « préfixe croissant, émission du delta » : on re-décode le préfixe
    /// courant par lots à taille croissante (1er lot petit → TTFA basse ; ×2 ensuite
    /// → coût total O(N) borné, le codec GPU étant rapide). Le `TtsSynthesisOutput`
    /// renvoyé agrège tout l'audio émis (== batch).
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si la génération, le codec, ou `on_audio` échoue.
    pub fn synthesize_greedy_streaming<F>(
        &self,
        text: &str,
        max_frames: usize,
        on_audio: F,
    ) -> Result<TtsSynthesisOutput>
    where
        F: FnMut(&[f32]) -> Result<()>,
    {
        self.synthesize_streaming_from_frames(
            |on_frame| {
                self.generate_codes_greedy_trace(
                    text,
                    max_frames,
                    tts_generation_trace_enabled(),
                    on_frame,
                )
            },
            on_audio,
        )
    }

    fn synthesize_streaming_from_frames<F, G>(
        &self,
        generate_codes: G,
        mut on_audio: F,
    ) -> Result<TtsSynthesisOutput>
    where
        F: FnMut(&[f32]) -> Result<()>,
        G: FnOnce(&mut dyn FnMut(&[Vec<i32>]) -> Result<()>) -> Result<Vec<Vec<i32>>>,
    {
        let first_lot = tts_stream_first_lot();
        let mut all_samples: Vec<f32> = Vec::new();
        let mut threshold = first_lot;
        let mut decode_state = self.new_stream_decode_state();
        let profile = tts_internal_profile_enabled();
        let profile_started = Instant::now();
        let mut codec_calls = 0_usize;
        let mut codec_total = Duration::ZERO;
        let mut first_codec = None;
        let mut first_emit = None;
        let mut first_emit_frames = 0_usize;
        let mut on_frame = |generated: &[Vec<i32>]| -> Result<()> {
            if generated.len() < threshold {
                return Ok(());
            }
            let codec_started = Instant::now();
            let pcm = self.decode_codes_for_mode_streaming(&mut decode_state, generated)?;
            let codec_elapsed = codec_started.elapsed();
            if profile {
                codec_calls += 1;
                codec_total += codec_elapsed;
                first_codec.get_or_insert(codec_elapsed);
                if !pcm.is_empty() && first_emit.is_none() {
                    first_emit = Some(profile_started.elapsed());
                    first_emit_frames = generated.len();
                }
            }
            if !pcm.is_empty() {
                on_audio(&pcm)?;
                all_samples.extend_from_slice(&pcm);
            }
            // Lots à taille croissante (×2) : O(N) total, peu de dispatches.
            threshold = threshold.saturating_mul(2).max(generated.len() + 1);
            Ok(())
        };
        let codes = generate_codes(&mut on_frame)?;
        // Flush final : décode le suffixe non couvert par le dernier lot.
        let final_flush_started = Instant::now();
        let pcm = self.decode_codes_for_mode_streaming(&mut decode_state, &codes)?;
        let final_flush = final_flush_started.elapsed();
        if profile {
            codec_calls += 1;
            codec_total += final_flush;
            first_codec.get_or_insert(final_flush);
            if !pcm.is_empty() && first_emit.is_none() {
                first_emit = Some(profile_started.elapsed());
                first_emit_frames = codes.len();
            }
        }
        if !pcm.is_empty() {
            on_audio(&pcm)?;
            all_samples.extend_from_slice(&pcm);
        }
        if profile {
            eprintln!(
                "perf tts.internal.stream_codec first_lot={} frames={} samples={} codec_calls={} codec_ms={:.3} first_codec_ms={:.3} first_emit_ms={:.3} first_emit_frames={} final_flush_ms={:.3} total_ms={:.3}",
                first_lot,
                codes.len(),
                all_samples.len(),
                codec_calls,
                codec_total.as_secs_f64() * 1e3,
                first_codec.unwrap_or_default().as_secs_f64() * 1e3,
                first_emit.unwrap_or_default().as_secs_f64() * 1e3,
                first_emit_frames,
                final_flush.as_secs_f64() * 1e3,
                profile_started.elapsed().as_secs_f64() * 1e3,
            );
        }
        Ok(TtsSynthesisOutput {
            codes,
            samples: all_samples,
            sample_rate: self.codec.sample_rate(),
        })
    }

    fn decode_codes_for_mode(&self, codes: &[Vec<i32>]) -> Result<Vec<f32>> {
        let Some(ctx) = self.clone_ctx.as_ref() else {
            return self.codec.decode_codes(codes);
        };
        if codes.is_empty() {
            return Ok(Vec::new());
        }
        if ctx.mode.is_xvec_only() {
            return self.codec.decode_codes(codes);
        }
        let mut prefixed = ctx.ref_codes.clone();
        prefixed.extend_from_slice(codes);
        let mut samples = self.codec.decode_codes(&prefixed)?;
        let skip = ctx
            .ref_codes
            .len()
            .checked_mul(usize_from_i32(
                self.assets.codec_config.decode_upsample_rate,
                "decode_upsample_rate",
            )?)
            .ok_or_else(|| InferError::Shape("trim clone TTS trop grand".to_string()))?;
        if skip >= samples.len() {
            return Ok(Vec::new());
        }
        samples.drain(0..skip);
        Ok(samples)
    }

    fn new_stream_decode_state(&self) -> TtsStreamDecodeState {
        match self.clone_ctx.as_ref() {
            Some(ctx) if !ctx.mode.is_xvec_only() => {
                TtsStreamDecodeState::FullPrefixDelta { emitted: 0 }
            }
            _ => TtsStreamDecodeState::Incremental(self.codec.new_stream_state()),
        }
    }

    fn decode_codes_for_mode_streaming(
        &self,
        state: &mut TtsStreamDecodeState,
        codes: &[Vec<i32>],
    ) -> Result<Vec<f32>> {
        match state {
            TtsStreamDecodeState::Incremental(codec_state) => {
                self.codec.decode_codes_streaming(codec_state, codes)
            }
            TtsStreamDecodeState::FullPrefixDelta { emitted } => {
                let pcm = self.decode_codes_for_mode(codes)?;
                if pcm.len() < *emitted {
                    return Err(InferError::Dimension(format!(
                        "streaming TTS PCM régressif: {} < {}",
                        pcm.len(),
                        *emitted
                    )));
                }
                let delta = pcm[*emitted..].to_vec();
                *emitted = pcm.len();
                Ok(delta)
            }
        }
    }

    fn prepare_inputs(&self, text: &str) -> Result<PreparedVoiceDesign> {
        match self.clone_ctx.as_ref() {
            Some(ctx) => self.prepare_icl_inputs(text, ctx),
            None => self.prepare_voicedesign_inputs(text),
        }
    }

    fn prepare_icl_inputs(&self, text: &str, ctx: &TtsCloneContext) -> Result<PreparedVoiceDesign> {
        let cfg = &self.assets.model_config.talker_config;
        let target_chat =
            format!("<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n");
        let target_ids = self.encode_ids(&target_chat)?;
        if target_ids.len() < 9 {
            return Err(InferError::Config(
                "prompt cible clone trop court".to_string(),
            ));
        }
        let text_ids = &target_ids[3..target_ids.len() - 5];
        let mut combined_ids = Vec::with_capacity(ctx.ref_text_ids.len() + text_ids.len());
        combined_ids.extend_from_slice(&ctx.ref_text_ids);
        combined_ids.extend_from_slice(text_ids);

        let tts_ids = [
            self.assets.model_config.tts_bos_token_id,
            self.assets.model_config.tts_eos_token_id,
            self.assets.model_config.tts_pad_token_id,
        ];
        let tts_embeds = self.text_embed(&tts_ids)?;
        let tts_bos = tts_embeds.row_slice(0)?.to_vec();
        let tts_eos = tts_embeds.row_slice(1)?.to_vec();
        let tts_pad = tts_embeds.row_slice(2)?.to_vec();

        let mut text_embed = self.text_embed(&combined_ids)?;
        let hidden = self.hidden_dim()?;
        let mut text_rows = text_embed.into_data();
        text_rows.extend_from_slice(&tts_eos);
        text_embed = Tensor::from_vec(vec![text_rows.len() / hidden, hidden], text_rows)?;

        let codec_bos = self.codec_embed(&[cfg.codec_bos_id])?;
        let mut codec_icl_rows = codec_bos.into_data();
        if let Some(ref_codec) = ctx.ref_codec_embed.as_ref() {
            codec_icl_rows.extend_from_slice(ref_codec.data());
        }
        let codec_icl =
            Tensor::from_vec(vec![codec_icl_rows.len() / hidden, hidden], codec_icl_rows)?;

        let codec_pad = self.codec_embed(&[cfg.codec_pad_id])?;
        let codec_pad = codec_pad.as_row()?.to_vec();
        let mut icl_rows =
            Vec::with_capacity((text_embed.shape()[0] + codec_icl.shape()[0]) * hidden);
        for row in 0..text_embed.shape()[0] {
            let mut item = text_embed.row_slice(row)?.to_vec();
            add_into(&mut item, &codec_pad);
            icl_rows.extend_from_slice(&item);
        }
        for row in 0..codec_icl.shape()[0] {
            let mut item = codec_icl.row_slice(row)?.to_vec();
            add_into(&mut item, &tts_pad);
            icl_rows.extend_from_slice(&item);
        }

        let lang_id = cfg.codec_language_id.get("french").copied();
        let codec_prefill: Vec<i32> = match lang_id {
            Some(lid) => vec![
                cfg.codec_think_id,
                cfg.codec_think_bos_id,
                lid,
                cfg.codec_think_eos_id,
            ],
            None => vec![
                cfg.codec_nothink_id,
                cfg.codec_think_bos_id,
                cfg.codec_think_eos_id,
            ],
        };
        let mut codec_prefix_rows = self.codec_embed(&codec_prefill)?.into_data();
        codec_prefix_rows.extend_from_slice(ctx.speaker_embed.as_row()?);
        codec_prefix_rows.extend_from_slice(
            self.codec_embed(&[cfg.codec_pad_id, cfg.codec_bos_id])?
                .data(),
        );
        let codec_prefix = Tensor::from_vec(
            vec![codec_prefix_rows.len() / hidden, hidden],
            codec_prefix_rows,
        )?;

        let role_embed = self.text_embed(&target_ids[..3])?;
        let prefix_len = codec_prefix.shape()[0];
        if prefix_len < 2 {
            return Err(InferError::Config(
                "préfixe codec clone trop court".to_string(),
            ));
        }
        let mut rows = Vec::new();
        push_rows(&mut rows, &role_embed, 0, role_embed.shape()[0])?;
        for row in 0..(prefix_len - 1) {
            let mut item = if row + 1 == prefix_len - 1 {
                tts_bos.clone()
            } else {
                tts_pad.clone()
            };
            add_into(&mut item, codec_prefix.row_slice(row)?);
            rows.extend_from_slice(&item);
        }
        rows.extend_from_slice(&icl_rows);

        Ok(PreparedVoiceDesign {
            input: Tensor::from_vec(vec![rows.len() / hidden, hidden], rows)?,
            trailing: Tensor::row(tts_pad.clone())?,
            tts_pad: Tensor::row(tts_pad)?,
        })
    }

    fn ref_codec_embed_sum(&self, ref_codes: &[Vec<i32>]) -> Result<Tensor> {
        let n_groups = usize::try_from(self.assets.model_config.talker_config.num_code_groups)
            .map_err(|_| InferError::Config("num_code_groups TTS négatif".to_string()))?;
        if ref_codes.iter().any(|frame| frame.len() < n_groups) {
            return Err(InferError::Dimension(
                "ref_codes clone sans tous les codebooks".to_string(),
            ));
        }
        let cb0 = ref_codes.iter().map(|frame| frame[0]).collect::<Vec<_>>();
        let mut acc = self.codec_embed(&cb0)?;
        for codebook in 1..n_groups {
            let ids = ref_codes
                .iter()
                .map(|frame| frame[codebook])
                .collect::<Vec<_>>();
            acc = acc.add(&self.code_predictor_embed(codebook - 1, &ids)?)?;
        }
        Ok(acc)
    }

    fn prepare_voicedesign_inputs(&self, text: &str) -> Result<PreparedVoiceDesign> {
        let cfg = &self.assets.model_config.talker_config;
        let chat = format!("<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n");
        let ids = self.encode_ids(&chat)?;
        if ids.len() < 9 {
            return Err(InferError::Config(format!(
                "prompt TTS trop court après tokenisation: {} tokens",
                ids.len()
            )));
        }
        let text_embed = self.text_embed(&ids)?;

        let tts_ids = [
            self.assets.model_config.tts_bos_token_id,
            self.assets.model_config.tts_eos_token_id,
            self.assets.model_config.tts_pad_token_id,
        ];
        let tts_embeds = self.text_embed(&tts_ids)?;
        let tts_bos = tts_embeds.row_slice(0)?.to_vec();
        let tts_pad = tts_embeds.row_slice(2)?.to_vec();

        let lang_id = cfg.codec_language_id.get("french").copied();
        let codec_prefill: Vec<i32> = match lang_id {
            Some(lid) => vec![
                cfg.codec_think_id,
                cfg.codec_think_bos_id,
                lid,
                cfg.codec_think_eos_id,
            ],
            None => vec![
                cfg.codec_nothink_id,
                cfg.codec_think_bos_id,
                cfg.codec_think_eos_id,
            ],
        };
        let mut codec_ids = codec_prefill;
        codec_ids.push(cfg.codec_pad_id);
        codec_ids.push(cfg.codec_bos_id);
        let codec_embed = self.codec_embed(&codec_ids)?;
        let codec_len = codec_embed.shape()[0];
        let hidden = self.hidden_dim()?;
        if codec_len < 2 {
            return Err(InferError::Config(
                "préfixe codec TTS trop court".to_string(),
            ));
        }

        let instruct = DEFAULT_INSTRUCT;
        let instruct_chat = format!("<|im_start|>user\n{instruct}<|im_end|>\n");
        let instruct_ids = self.encode_ids(&instruct_chat)?;
        let instruct_embed = self.text_embed(&instruct_ids)?;

        let mut rows = Vec::new();
        push_rows(&mut rows, &instruct_embed, 0, instruct_embed.shape()[0])?;
        push_rows(&mut rows, &text_embed, 0, 3)?;

        for i in 0..(codec_len - 1) {
            let mut row = if i + 1 == codec_len - 1 {
                tts_bos.clone()
            } else {
                tts_pad.clone()
            };
            add_into(&mut row, codec_embed.row_slice(i)?);
            rows.extend_from_slice(&row);
        }

        let mut first_text = text_embed.row_slice(3)?.to_vec();
        add_into(&mut first_text, codec_embed.row_slice(codec_len - 1)?);
        rows.extend_from_slice(&first_text);

        let trailing_start = 4;
        let trailing_end = ids
            .len()
            .checked_sub(5)
            .ok_or_else(|| InferError::Config("prompt TTS trop court pour trailing".to_string()))?;
        let mut trailing_rows = Vec::new();
        if trailing_start < trailing_end {
            push_rows(
                &mut trailing_rows,
                &text_embed,
                trailing_start,
                trailing_end,
            )?;
        }
        trailing_rows.extend_from_slice(tts_embeds.row_slice(1)?);

        Ok(PreparedVoiceDesign {
            input: Tensor::from_vec(vec![rows.len() / hidden, hidden], rows)?,
            trailing: Tensor::from_vec(vec![trailing_rows.len() / hidden, hidden], trailing_rows)?,
            tts_pad: Tensor::row(tts_pad)?,
        })
    }

    fn text_embed(&self, ids: &[i32]) -> Result<Tensor> {
        let raw = gather_rows_i32(&self.text_embedding, ids)?;
        self.text_projection_fc2
            .forward(&crate::silu(&self.text_projection_fc1.forward(&raw)?))
    }

    fn codec_embed(&self, ids: &[i32]) -> Result<Tensor> {
        gather_rows_i32(&self.codec_embedding, ids)
    }

    fn code_predictor_embed(&self, codebook: usize, ids: &[i32]) -> Result<Tensor> {
        let table = self
            .code_predictor_embeddings
            .get(codebook)
            .ok_or_else(|| {
                InferError::MissingWeight(format!(
                    "code_predictor.model.codec_embedding.{codebook}"
                ))
            })?;
        gather_rows_i32(table, ids)
    }

    fn generate_codes_greedy(&self, text: &str, max_frames: usize) -> Result<Vec<Vec<i32>>> {
        self.generate_codes_greedy_trace(
            text,
            max_frames,
            tts_generation_trace_enabled(),
            &mut |_| Ok(()),
        )
    }

    fn generate_codes_clone_sampled(&self, text: &str, max_frames: usize) -> Result<Vec<Vec<i32>>> {
        let target_tokens = self.encode_ids(text)?.len();
        let effective_max_frames = clone_effective_frame_cap(max_frames, target_tokens);
        let params = TtsSampleParams {
            temperature: CLONE_SAMPLE_TEMPERATURE,
            top_k: CLONE_SAMPLE_TOP_K,
            top_p: CLONE_SAMPLE_TOP_P,
            repetition_penalty: CLONE_SAMPLE_REPETITION_PENALTY,
            seed: clone_sample_seed(),
        };
        self.generate_codes_clone_sampled_trace(
            text,
            max_frames,
            effective_max_frames,
            params,
            tts_generation_trace_enabled(),
            &mut |_| Ok(()),
        )
    }

    fn generate_codes_clone_sampled_trace(
        &self,
        text: &str,
        requested_max_frames: usize,
        effective_max_frames: usize,
        params: TtsSampleParams,
        trace: bool,
        on_frame: &mut dyn FnMut(&[Vec<i32>]) -> Result<()>,
    ) -> Result<Vec<Vec<i32>>> {
        let cfg = &self.assets.model_config.talker_config;
        let mode = self
            .clone_ctx
            .as_ref()
            .map_or("clone-sampled", |ctx| ctx.mode.label());
        if trace {
            eprintln!(
                "qwen3-tts rust diag: start mode={mode} requested_max_frames={requested_max_frames} effective_max_frames={effective_max_frames} hard_cap={CLONE_GENERATION_HARD_CAP} temp={} top_k={} top_p={} rep_penalty={} seed={}",
                params.temperature,
                params.top_k,
                params.top_p,
                params.repetition_penalty,
                params.seed,
            );
        }

        let started = Instant::now();
        let prepared = self.prepare_inputs(text)?;
        if trace {
            let ref_frames = self.clone_ctx.as_ref().map_or(0, |ctx| ctx.ref_codes.len());
            eprintln!(
                "qwen3-tts rust diag: prepared input_rows={} trailing_rows={} ref_frames={} elapsed_ms={}",
                prepared.input.shape()[0],
                prepared.trailing.shape()[0],
                ref_frames,
                started.elapsed().as_millis()
            );
        }
        let n_groups = usize::try_from(cfg.num_code_groups)
            .map_err(|_| InferError::Config("num_code_groups TTS négatif".to_string()))?;
        if n_groups == 0 {
            return Err(InferError::Config("num_code_groups TTS nul".to_string()));
        }
        let eos = cfg.codec_eos_token_id;
        let vocab = cfg.vocab_size;
        let suppress_start = vocab.checked_sub(1024).ok_or_else(|| {
            InferError::Config(format!("vocab TTS trop petit pour suppression: {vocab}"))
        })?;
        let suppress = (suppress_start..vocab)
            .filter(|token| *token != eos)
            .collect::<Vec<_>>();

        let hidden = self.hidden_dim()?;
        if trace {
            eprintln!(
                "qwen3-tts rust diag: prefill start hidden={hidden} groups={n_groups} eos={eos} vocab={vocab} suppress_start={suppress_start}"
            );
        }
        let prefill_started = Instant::now();
        let (mut talker_cache, mut final_state) =
            self.talker.prefill_cache_from_embeddings(&prepared.input)?;
        if trace {
            eprintln!(
                "qwen3-tts rust diag: prefill done elapsed_ms={}",
                prefill_started.elapsed().as_millis()
            );
        }
        let talker_resident = self
            .talker
            .setup_resident_decode_from_prefill(&mut talker_cache, effective_max_frames)?;
        if trace {
            eprintln!("qwen3-tts rust diag: talker resident={talker_resident}");
        }
        let mut cp_resident_cache = if n_groups > 1 {
            self.code_predictor
                .new_resident_decode_cache(n_groups + 1)?
        } else {
            None
        };
        if trace {
            eprintln!(
                "qwen3-tts rust diag: cp resident={}",
                cp_resident_cache.is_some()
            );
        }

        let mut generated = Vec::with_capacity(effective_max_frames);
        let mut trailing_idx = 0_usize;
        let repeat_stop = tts_repeat_frame_stop();
        let mut repeat_run = 1_usize;
        let mut cb0_history = Vec::with_capacity(effective_max_frames);
        let mut sampler = DeterministicSampler::new(params.seed);

        for step in 0..effective_max_frames {
            let logits = self.talker.logits_from_final_state(&final_state)?;
            let tok0 = sample_talker_token(
                logits.as_row()?,
                &params,
                &suppress,
                &cb0_history,
                Some(eos),
                &mut sampler,
            )?;
            if trace {
                eprintln!("qwen3-tts rust diag: frame={step} cb0={tok0}");
            }
            if tok0 == eos {
                if trace {
                    eprintln!(
                        "qwen3-tts rust diag: eos frame={step} generated_frames={}",
                        generated.len()
                    );
                }
                break;
            }

            let codes_frame =
                self.predict_codebooks(tok0, &final_state, n_groups, cp_resident_cache.as_mut())?;

            let text_embed = if trailing_idx < prepared.trailing.shape()[0] {
                let row = Tensor::row(prepared.trailing.row_slice(trailing_idx)?.to_vec())?;
                trailing_idx += 1;
                row
            } else {
                prepared.tts_pad.clone()
            };
            let mut codec_embed = self.codec_embed(&[tok0])?;
            for (codebook, code) in codes_frame.iter().skip(1).copied().enumerate() {
                codec_embed = codec_embed.add(&self.code_predictor_embed(codebook, &[code])?)?;
            }
            let next_input = text_embed.add(&codec_embed)?;
            final_state = if talker_resident {
                self.talker
                    .decode_step_resident_from_embedding(&mut talker_cache, &next_input)?
                    .ok_or_else(|| {
                        InferError::Metal(
                            "decode résident talker indisponible en cours de frame".to_string(),
                        )
                    })?
            } else {
                self.talker
                    .next_state_from_embedding(&mut talker_cache, &next_input)?
            };

            let repeated_frame = generated.last().is_some_and(|prev| prev == &codes_frame);
            repeat_run = if repeated_frame {
                repeat_run.saturating_add(1)
            } else {
                1
            };
            cb0_history.push(tok0);
            generated.push(codes_frame);
            on_frame(&generated)?;
            if trace {
                let last = generated
                    .last()
                    .ok_or_else(|| InferError::Dimension("frame TTS manquante".to_string()))?;
                eprintln!(
                    "qwen3-tts rust diag: frame={step} done generated_frames={} trailing_idx={} codes={last:?}",
                    generated.len(),
                    trailing_idx
                );
            }
            if repeat_frame_stop_tripped(repeat_run, repeat_stop) {
                if trace {
                    eprintln!(
                        "qwen3-tts rust diag: stop repeat_frame run={repeat_run} threshold={repeat_stop} generated_frames={}",
                        generated.len()
                    );
                }
                break;
            }
        }
        if trace {
            eprintln!(
                "qwen3-tts rust diag: stop generated_frames={} reached_cap={} requested_max_frames={} effective_max_frames={effective_max_frames}",
                generated.len(),
                generated.len() == effective_max_frames,
                requested_max_frames
            );
        }
        Ok(generated)
    }

    /// Sélectionne cb0 (token talker) depuis `final_state` : argmax greedy GPU
    /// (quantifié + suppression de la plage `[suppress_start, vocab)` sauf `eos`)
    /// si Metal est dispo, sinon fallback CPU `logits_from_final_state` +
    /// `greedy_talker_token`. Byte-identique entre les deux chemins.
    fn talker_cb0(
        &self,
        final_state: &Tensor,
        suppress_start: i32,
        eos: i32,
        suppress: &[i32],
    ) -> Result<i32> {
        let suppress_start_usize = usize::try_from(suppress_start)
            .map_err(|_| InferError::Config("suppress_start TTS négatif".to_string()))?;
        let eos_usize =
            usize::try_from(eos).map_err(|_| InferError::Config("eos TTS négatif".to_string()))?;
        match self
            .talker
            .talker_greedy_token(final_state, suppress_start_usize, eos_usize)?
        {
            Some(token) => i32::try_from(token)
                .map_err(|_| InferError::Config(format!("cb0 talker hors i32: {token}"))),
            None => {
                let logits = self.talker.logits_from_final_state(final_state)?;
                greedy_talker_token(logits.as_row()?, suppress)
            }
        }
    }

    /// Prédit les codes d'un frame : `tok0` (talker) suivi des `n_groups-1`
    /// codebooks du code_predictor, en decode KV-caché.
    ///
    /// Si `cp_resident` est `Some`, chaque pas du code_predictor passe par le
    /// decode GPU résident (1 pas = 1 command buffer) ; sinon le decode reste
    /// per-op. Les deux chemins sont algorithmiquement identiques (préfixe
    /// `[final_state, e0]` puis un pas par codebook).
    fn predict_codebooks(
        &self,
        tok0: i32,
        final_state: &Tensor,
        n_groups: usize,
        cp_resident: Option<&mut CausalDecoderCache>,
    ) -> Result<Vec<i32>> {
        let mut codes_frame = vec![tok0];
        if n_groups <= 1 {
            return Ok(codes_frame);
        }
        let e0 = self.codec_embed(&[tok0])?;
        match cp_resident {
            Some(cp_cache) => {
                // position 0 : final_state (seed KV, pas de code). Puis un pas par
                // codebook avec la tête FUSIONNÉE on-device (matmul tête + argmax
                // greedy dans le command buffer du pas → readback d'1 u32, plus de
                // cp_state relu ni de matmul de tête CPU). KV remis à zéro par frame.
                self.code_predictor.reset_resident_decode_cache(cp_cache)?;
                let fs_proj = self.code_predictor_projection.forward(final_state)?;
                self.code_predictor
                    .decode_step_resident_from_embedding(cp_cache, &fs_proj)?
                    .ok_or_else(|| {
                        InferError::Metal("cp résident: pas final_state absent".to_string())
                    })?;
                let mut next_input = self.code_predictor_projection.forward(&e0)?;
                for code_idx in 0..(n_groups - 1) {
                    let head = self.code_predictor_heads.get(code_idx).ok_or_else(|| {
                        InferError::MissingWeight(format!("code_predictor.lm_head.{code_idx}"))
                    })?;
                    let token = self
                        .code_predictor
                        .decode_token_resident_from_embedding_head(cp_cache, &next_input, head)?
                        .ok_or_else(|| {
                            InferError::Metal("cp résident: token absent".to_string())
                        })?;
                    let code = i32::try_from(token)
                        .map_err(|_| InferError::Config(format!("code cp hors i32: {token}")))?;
                    codes_frame.push(code);
                    if code_idx + 1 < n_groups - 1 {
                        let embed = self.code_predictor_embed(code_idx, &[code])?;
                        next_input = self.code_predictor_projection.forward(&embed)?;
                    }
                }
            }
            None => {
                let hidden = self.hidden_dim()?;
                let mut prefix_rows = Vec::with_capacity(2 * hidden);
                prefix_rows.extend_from_slice(final_state.as_row()?);
                prefix_rows.extend_from_slice(e0.as_row()?);
                let prefix = Tensor::from_vec(vec![2, hidden], prefix_rows)?;
                let cp_prefix = self.code_predictor_projection.forward(&prefix)?;
                let (mut cp_cache, mut cp_state) = self
                    .code_predictor
                    .prefill_cache_from_embeddings(&cp_prefix)?;
                for code_idx in 0..(n_groups - 1) {
                    let head = self.code_predictor_heads.get(code_idx).ok_or_else(|| {
                        InferError::MissingWeight(format!("code_predictor.lm_head.{code_idx}"))
                    })?;
                    let code = greedy_token(head.forward(&cp_state)?.as_row()?, &[])?;
                    codes_frame.push(code);
                    if code_idx + 1 < n_groups - 1 {
                        let embed = self.code_predictor_embed(code_idx, &[code])?;
                        let cp_in = self.code_predictor_projection.forward(&embed)?;
                        cp_state = self
                            .code_predictor
                            .next_state_from_embedding(&mut cp_cache, &cp_in)?;
                    }
                }
            }
        }
        Ok(codes_frame)
    }

    fn generate_codes_greedy_trace(
        &self,
        text: &str,
        max_frames: usize,
        trace: bool,
        on_frame: &mut dyn FnMut(&[Vec<i32>]) -> Result<()>,
    ) -> Result<Vec<Vec<i32>>> {
        let cfg = &self.assets.model_config.talker_config;
        let mode = self
            .clone_ctx
            .as_ref()
            .map_or("voicedesign", |ctx| ctx.mode.label());
        let effective_max_frames = if self.clone_ctx.is_some() {
            max_frames.min(CLONE_GENERATION_HARD_CAP)
        } else {
            max_frames
        };
        if trace {
            eprintln!(
                "qwen3-tts rust diag: start mode={mode} requested_max_frames={max_frames} effective_max_frames={effective_max_frames} hard_cap={CLONE_GENERATION_HARD_CAP}"
            );
        }

        let started = Instant::now();
        let prepared = self.prepare_inputs(text)?;
        if trace {
            let ref_frames = self.clone_ctx.as_ref().map_or(0, |ctx| ctx.ref_codes.len());
            eprintln!(
                "qwen3-tts rust diag: prepared input_rows={} trailing_rows={} ref_frames={} elapsed_ms={}",
                prepared.input.shape()[0],
                prepared.trailing.shape()[0],
                ref_frames,
                started.elapsed().as_millis()
            );
        }
        let n_groups = usize::try_from(cfg.num_code_groups)
            .map_err(|_| InferError::Config("num_code_groups TTS négatif".to_string()))?;
        if n_groups == 0 {
            return Err(InferError::Config("num_code_groups TTS nul".to_string()));
        }
        let eos = cfg.codec_eos_token_id;
        let vocab = cfg.vocab_size;
        let suppress_start = vocab.checked_sub(1024).ok_or_else(|| {
            InferError::Config(format!("vocab TTS trop petit pour suppression: {vocab}"))
        })?;
        let suppress = (suppress_start..vocab)
            .filter(|token| *token != eos)
            .collect::<Vec<_>>();

        let hidden = self.hidden_dim()?;
        if trace {
            eprintln!(
                "qwen3-tts rust diag: prefill start hidden={hidden} groups={n_groups} eos={eos} vocab={vocab} suppress_start={suppress_start}"
            );
        }
        let prefill_started = Instant::now();
        let (mut talker_cache, mut final_state) =
            self.talker.prefill_cache_from_embeddings(&prepared.input)?;
        if trace {
            eprintln!(
                "qwen3-tts rust diag: prefill done elapsed_ms={}",
                prefill_started.elapsed().as_millis()
            );
        }
        // Decode résident GPU du talker : un pas = un command buffer (vs ~28 couches
        // per-op). Seed le KV GPU depuis le cache du prefill ; si indisponible, on
        // reste sur le per-op (`next_state_from_embedding`). Tout-ou-rien.
        let talker_resident = self
            .talker
            .setup_resident_decode_from_prefill(&mut talker_cache, effective_max_frames)?;
        if trace {
            eprintln!("qwen3-tts rust diag: talker resident={talker_resident}");
        }
        // Code predictor : arène résidente allouée une fois, remise à zéro par
        // frame (séquence courte de num_code_groups+1 positions).
        let mut cp_resident_cache = if n_groups > 1 {
            self.code_predictor
                .new_resident_decode_cache(n_groups + 1)?
        } else {
            None
        };
        if trace {
            eprintln!(
                "qwen3-tts rust diag: cp resident={}",
                cp_resident_cache.is_some()
            );
        }
        let mut generated = Vec::with_capacity(effective_max_frames);
        let mut trailing_idx = 0_usize;
        let repeat_stop = tts_repeat_frame_stop();
        let mut repeat_run = 1_usize;

        for step in 0..effective_max_frames {
            // cb0 talker : argmax greedy GPU (quantifié + suppression) on-device,
            // sinon fallback CPU. Réplique greedy_talker_token byte-identique.
            let tok0 = self.talker_cb0(&final_state, suppress_start, eos, &suppress)?;
            if trace {
                eprintln!("qwen3-tts rust diag: frame={step} cb0={tok0}");
            }
            if tok0 == eos {
                if trace {
                    eprintln!(
                        "qwen3-tts rust diag: eos frame={step} generated_frames={}",
                        generated.len()
                    );
                }
                break;
            }

            let codes_frame =
                self.predict_codebooks(tok0, &final_state, n_groups, cp_resident_cache.as_mut())?;

            let text_embed = if trailing_idx < prepared.trailing.shape()[0] {
                let row = Tensor::row(prepared.trailing.row_slice(trailing_idx)?.to_vec())?;
                trailing_idx += 1;
                row
            } else {
                prepared.tts_pad.clone()
            };
            let mut codec_embed = self.codec_embed(&[tok0])?;
            for (codebook, code) in codes_frame.iter().skip(1).copied().enumerate() {
                codec_embed = codec_embed.add(&self.code_predictor_embed(codebook, &[code])?)?;
            }
            let next_input = text_embed.add(&codec_embed)?;
            final_state = if talker_resident {
                self.talker
                    .decode_step_resident_from_embedding(&mut talker_cache, &next_input)?
                    .ok_or_else(|| {
                        InferError::Metal(
                            "decode résident talker indisponible en cours de frame".to_string(),
                        )
                    })?
            } else {
                self.talker
                    .next_state_from_embedding(&mut talker_cache, &next_input)?
            };
            let repeated_frame = generated.last().is_some_and(|prev| prev == &codes_frame);
            repeat_run = if repeated_frame {
                repeat_run.saturating_add(1)
            } else {
                1
            };
            generated.push(codes_frame);
            // Hook streaming : permet au décodage codec incrémental d'émettre
            // l'audio du préfixe au fil de la génération (byte-identique par
            // causalité). No-op pour le chemin batch.
            on_frame(&generated)?;
            if trace {
                let last = generated
                    .last()
                    .ok_or_else(|| InferError::Dimension("frame TTS manquante".to_string()))?;
                eprintln!(
                    "qwen3-tts rust diag: frame={step} done generated_frames={} trailing_idx={} codes={last:?}",
                    generated.len(),
                    trailing_idx
                );
            }
            if repeat_frame_stop_tripped(repeat_run, repeat_stop) {
                if trace {
                    eprintln!(
                        "qwen3-tts rust diag: stop repeat_frame run={repeat_run} threshold={repeat_stop} generated_frames={}",
                        generated.len()
                    );
                }
                break;
            }
        }
        if trace {
            eprintln!(
                "qwen3-tts rust diag: stop generated_frames={} reached_cap={} requested_max_frames={} effective_max_frames={effective_max_frames}",
                generated.len(),
                generated.len() == effective_max_frames,
                max_frames
            );
        }
        Ok(generated)
    }

    #[cfg(test)]
    fn probe_greedy_logits(
        &self,
        text: &str,
        max_frames: usize,
        target_frame: usize,
        target_group: usize,
    ) -> Result<(Vec<Vec<i32>>, Vec<f32>)> {
        let cfg = &self.assets.model_config.talker_config;
        let prepared = self.prepare_inputs(text)?;
        let n_groups = usize::try_from(cfg.num_code_groups)
            .map_err(|_| InferError::Config("num_code_groups TTS négatif".to_string()))?;
        if target_group >= n_groups {
            return Err(InferError::Config(format!(
                "target_group TTS hors bornes: {target_group}"
            )));
        }
        let eos = cfg.codec_eos_token_id;
        let vocab = cfg.vocab_size;
        let suppress_start = vocab.checked_sub(1024).ok_or_else(|| {
            InferError::Config(format!("vocab TTS trop petit pour suppression: {vocab}"))
        })?;
        let suppress = (suppress_start..vocab)
            .filter(|token| *token != eos)
            .collect::<Vec<_>>();
        let hidden = self.hidden_dim()?;
        let (mut talker_cache, mut final_state) =
            self.talker.prefill_cache_from_embeddings(&prepared.input)?;
        let mut logits = self.talker.logits_from_final_state(&final_state)?;
        let mut generated = Vec::with_capacity(max_frames);
        let mut trailing_idx = 0_usize;

        for frame_idx in 0..max_frames {
            if frame_idx == target_frame && target_group == 0 {
                return Ok((generated, logits.as_row()?.to_vec()));
            }
            let tok0 = greedy_talker_token(logits.as_row()?, &suppress)?;
            if tok0 == eos {
                break;
            }

            let mut codes_frame = vec![tok0];
            let e0 = self.codec_embed(&[tok0])?;
            let mut cp_rows = Vec::with_capacity((n_groups + 1) * hidden);
            cp_rows.extend_from_slice(final_state.as_row()?);
            cp_rows.extend_from_slice(e0.as_row()?);

            for code_idx in 0..(n_groups - 1) {
                if code_idx > 0 {
                    let prev = codes_frame[code_idx];
                    let embed = self.code_predictor_embed(code_idx - 1, &[prev])?;
                    cp_rows.extend_from_slice(embed.as_row()?);
                }
                let cp_input =
                    Tensor::from_vec(vec![cp_rows.len() / hidden, hidden], cp_rows.clone())?;
                let cp_input = self.code_predictor_projection.forward(&cp_input)?;
                let (_cp_cache, cp_state) = self
                    .code_predictor
                    .prefill_cache_from_embeddings(&cp_input)?;
                let head = self.code_predictor_heads.get(code_idx).ok_or_else(|| {
                    InferError::MissingWeight(format!("code_predictor.lm_head.{code_idx}"))
                })?;
                let code_logits = head.forward(&cp_state)?;
                let group = code_idx + 1;
                if frame_idx == target_frame && group == target_group {
                    return Ok((generated, code_logits.as_row()?.to_vec()));
                }
                codes_frame.push(greedy_token(code_logits.as_row()?, &[])?);
            }

            let text_embed = if trailing_idx < prepared.trailing.shape()[0] {
                let row = Tensor::row(prepared.trailing.row_slice(trailing_idx)?.to_vec())?;
                trailing_idx += 1;
                row
            } else {
                prepared.tts_pad.clone()
            };
            let mut codec_embed = self.codec_embed(&[tok0])?;
            for (codebook, code) in codes_frame.iter().skip(1).copied().enumerate() {
                codec_embed = codec_embed.add(&self.code_predictor_embed(codebook, &[code])?)?;
            }
            let next_input = text_embed.add(&codec_embed)?;
            final_state = self
                .talker
                .next_state_from_embedding(&mut talker_cache, &next_input)?;
            logits = self.talker.logits_from_final_state(&final_state)?;
            generated.push(codes_frame);
        }
        Err(InferError::Dimension(format!(
            "point probe TTS non atteint: frame={target_frame}, group={target_group}"
        )))
    }

    fn encode_ids(&self, text: &str) -> Result<Vec<i32>> {
        let enc = self
            .tokenizer
            .encode(text, false)
            .map_err(|err| InferError::Tokenizer {
                path: self.assets.model_dir.clone(),
                message: err.to_string(),
            })?;
        enc.get_ids()
            .iter()
            .map(|id| {
                i32::try_from(*id)
                    .map_err(|_| InferError::Config(format!("token id TTS hors i32: {id}")))
            })
            .collect()
    }

    fn hidden_dim(&self) -> Result<usize> {
        usize::try_from(self.assets.model_config.talker_config.hidden_size)
            .map_err(|_| InferError::Config("hidden_size TTS négatif".to_string()))
    }
}

fn load_qwen3_tts_tokenizer(dir: &Path) -> Result<Tokenizer> {
    let tokenizer_json = dir.join("tokenizer.json");
    if tokenizer_json.is_file() {
        return Tokenizer::from_file(&tokenizer_json).map_err(|err| InferError::Tokenizer {
            path: tokenizer_json,
            message: err.to_string(),
        });
    }
    let vocab = dir.join("vocab.json");
    let merges = dir.join("merges.txt");
    require_file(&vocab, "vocab.json")?;
    require_file(&merges, "merges.txt")?;
    let vocab_str = vocab
        .to_str()
        .ok_or_else(|| InferError::Config(format!("chemin vocab non UTF-8: {vocab:?}")))?;
    let merges_str = merges
        .to_str()
        .ok_or_else(|| InferError::Config(format!("chemin merges non UTF-8: {merges:?}")))?;
    let bpe = BPE::from_file(vocab_str, merges_str)
        .build()
        .map_err(|err| InferError::Tokenizer {
            path: dir.to_path_buf(),
            message: format!("build BPE Qwen3-TTS: {err}"),
        })?;
    let mut tokenizer = Tokenizer::new(bpe);
    tokenizer.with_pre_tokenizer(Some(ByteLevel::new(false, true, true)));
    tokenizer.with_decoder(Some(ByteLevel::new(false, true, true)));
    let added = QWEN3_TTS_SPECIAL_TOKENS
        .iter()
        .map(|(content, special)| AddedToken::from(*content, *special))
        .collect::<Vec<_>>();
    tokenizer.add_special_tokens(&added);
    Ok(tokenizer)
}

fn tts_generation_trace_enabled() -> bool {
    std::env::var("RETI_TTS_TRACE_FRAMES")
        .map(|value| {
            let value = value.trim();
            !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
        })
        .unwrap_or(false)
}

fn tts_internal_profile_enabled() -> bool {
    std::env::var("RETI_TTS_INTERNAL_PROFILE")
        .map(|value| {
            let value = value.trim();
            !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
        })
        .unwrap_or(false)
}

/// Taille (en frames) du PREMIER lot de décodage streaming : petit = TTFA basse.
/// Les lots suivants croissent (×2) → coût total O(N). Réglable via
/// `RETI_TTS_STREAM_LOT` (défaut 4 frames ≈ 320 ms d'audio). Borné ≥ 1.
fn tts_stream_first_lot() -> usize {
    std::env::var("RETI_TTS_STREAM_LOT")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(4)
}

/// Arrêt anti-boucle du talker TTS : une frame codec complète identique répétée
/// plusieurs fois indique une dérive de décodage greedy, pas une prosodie utile.
/// `0` désactive le garde-fou pour les diagnostics.
fn tts_repeat_frame_stop() -> usize {
    tts_repeat_frame_stop_from_env(std::env::var("RETI_TTS_REPEAT_FRAME_STOP").ok().as_deref())
}

fn tts_repeat_frame_stop_from_env(value: Option<&str>) -> usize {
    value
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(DEFAULT_REPEAT_FRAME_STOP)
}

fn repeat_frame_stop_tripped(repeat_run: usize, threshold: usize) -> bool {
    threshold > 0 && repeat_run >= threshold
}

fn read_linear_layer(
    payload: &SafetensorPayload,
    source_prefix: &str,
    group_size: usize,
    bits: usize,
) -> Result<crate::Linear> {
    let weight = read_linear_weight(payload, source_prefix, group_size, bits)?;
    let bias_key = format!("{source_prefix}.bias");
    let bias = if payload.contains(&bias_key) {
        Some(payload.read_dense_tensor(&bias_key)?)
    } else {
        None
    };
    crate::Linear::from_weight(weight, bias)
}

fn insert_linear(
    tensors: &mut HashMap<String, DecoderTensor>,
    payload: &SafetensorPayload,
    source_prefix: &str,
    target_prefix: &str,
    group_size: usize,
    bits: usize,
) -> Result<()> {
    tensors.insert(
        format!("{target_prefix}.weight"),
        DecoderTensor::LinearWeight(read_linear_weight(
            payload,
            source_prefix,
            group_size,
            bits,
        )?),
    );
    let bias_key = format!("{source_prefix}.bias");
    if payload.contains(&bias_key) {
        tensors.insert(
            format!("{target_prefix}.bias"),
            DecoderTensor::Dense(payload.read_dense_tensor(&bias_key)?),
        );
    }
    Ok(())
}

fn read_linear_weight(
    payload: &SafetensorPayload,
    source_prefix: &str,
    group_size: usize,
    bits: usize,
) -> Result<LinearWeight> {
    let weight_key = format!("{source_prefix}.weight");
    let scales_key = format!("{source_prefix}.scales");
    if payload.contains(&scales_key) {
        let packed = payload.read_u32_tensor(&weight_key)?;
        let scales = payload.read_dense_tensor(&scales_key)?;
        let biases = payload.read_dense_tensor(&format!("{source_prefix}.biases"))?;
        let packed_shape = payload.entry(&weight_key)?.shape.clone();
        return Ok(LinearWeight::AffineQuantized(AffineQuantizedTensor::new(
            &packed_shape,
            packed,
            scales,
            biases,
            group_size,
            bits,
        )?));
    }
    Ok(LinearWeight::Dense(payload.read_dense_tensor(&weight_key)?))
}

fn copy_dense(
    tensors: &mut HashMap<String, DecoderTensor>,
    payload: &SafetensorPayload,
    source: &str,
    target: &str,
) -> Result<()> {
    tensors.insert(
        target.to_string(),
        DecoderTensor::Dense(payload.read_dense_tensor(source)?),
    );
    Ok(())
}

fn gather_rows_i32(table: &Tensor, ids: &[i32]) -> Result<Tensor> {
    let mut rows = Vec::with_capacity(ids.len());
    for id in ids {
        let id = usize::try_from(*id)
            .map_err(|_| InferError::Dimension(format!("id embedding négatif: {id}")))?;
        rows.push(id);
    }
    crate::embed_tokens(table, &rows)
}

fn push_rows(out: &mut Vec<f32>, tensor: &Tensor, start: usize, end: usize) -> Result<()> {
    if start > end || end > tensor.shape()[0] {
        return Err(InferError::Dimension(format!(
            "slice rows invalide [{start}..{end}] pour {:?}",
            tensor.shape()
        )));
    }
    for row in start..end {
        out.extend_from_slice(tensor.row_slice(row)?);
    }
    Ok(())
}

fn add_into(left: &mut [f32], right: &[f32]) {
    for (left, right) in left.iter_mut().zip(right.iter()) {
        *left += *right;
    }
}

fn usize_from_i32(value: i32, what: &str) -> Result<usize> {
    usize::try_from(value).map_err(|_| InferError::Config(format!("{what} négatif: {value}")))
}

fn clone_effective_frame_cap(max_frames: usize, target_tokens: usize) -> usize {
    max_frames.min(CLONE_GENERATION_HARD_CAP).min(
        CLONE_DEFAULT_MIN_FRAMES.max(target_tokens.saturating_mul(CLONE_DEFAULT_FRAMES_PER_TOKEN)),
    )
}

fn clone_sample_seed() -> u64 {
    std::env::var("RETI_TTS_CLONE_SAMPLE_SEED")
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or(0)
}

fn sample_talker_token(
    logits: &[f32],
    params: &TtsSampleParams,
    suppress: &[i32],
    history: &[i32],
    eos: Option<i32>,
    sampler: &mut DeterministicSampler,
) -> Result<i32> {
    if logits.is_empty() {
        return Err(InferError::Dimension(
            "sampling TTS sur logits vides".to_string(),
        ));
    }
    let mut adjusted = logits.to_vec();
    for token in suppress {
        if let Ok(idx) = usize::try_from(*token) {
            if let Some(value) = adjusted.get_mut(idx) {
                *value = f32::NEG_INFINITY;
            }
        }
    }
    apply_talker_repetition_penalty(&mut adjusted, history, eos, params.repetition_penalty);
    let sampled = sample_token_top_k_top_p(
        &adjusted,
        params.temperature,
        params.top_p,
        params.top_k,
        sampler,
    )?;
    i32::try_from(sampled).map_err(|_| InferError::Config(format!("token TTS hors i32: {sampled}")))
}

fn apply_talker_repetition_penalty(
    logits: &mut [f32],
    history: &[i32],
    eos: Option<i32>,
    repetition_penalty: f32,
) {
    if !repetition_penalty.is_finite() || repetition_penalty <= 1.0 {
        return;
    }
    let eos_idx = eos.and_then(|token| usize::try_from(token).ok());
    let mut seen = Vec::new();
    for token in history {
        let Ok(idx) = usize::try_from(*token) else {
            continue;
        };
        if Some(idx) == eos_idx || idx >= logits.len() || seen.contains(&idx) {
            continue;
        }
        seen.push(idx);
        let value = logits[idx];
        if !value.is_finite() {
            continue;
        }
        logits[idx] = if value < 0.0 {
            value * repetition_penalty
        } else {
            value / repetition_penalty
        };
    }
}

fn greedy_token(logits: &[f32], suppress: &[i32]) -> Result<i32> {
    if logits.is_empty() {
        return Err(InferError::Dimension(
            "argmax TTS sur logits vides".to_string(),
        ));
    }
    let mut best = 0_usize;
    let mut best_value = f32::NEG_INFINITY;
    'outer: for (idx, value) in logits.iter().copied().enumerate() {
        for token in suppress {
            if usize::try_from(*token).ok() == Some(idx) {
                continue 'outer;
            }
        }
        if value > best_value {
            best = idx;
            best_value = value;
        }
    }
    i32::try_from(best).map_err(|_| InferError::Config(format!("token TTS hors i32: {best}")))
}

fn greedy_talker_token(logits: &[f32], suppress: &[i32]) -> Result<i32> {
    if logits.is_empty() {
        return Err(InferError::Dimension(
            "argmax TTS sur logits vides".to_string(),
        ));
    }
    let mut best = 0_usize;
    let mut best_value = f32::NEG_INFINITY;
    'outer: for (idx, value) in logits.iter().copied().enumerate() {
        for token in suppress {
            if usize::try_from(*token).ok() == Some(idx) {
                continue 'outer;
            }
        }
        let value = mlx_greedy_logit(value);
        if value > best_value {
            best = idx;
            best_value = value;
        }
    }
    i32::try_from(best).map_err(|_| InferError::Config(format!("token TTS hors i32: {best}")))
}

fn mlx_greedy_logit(value: f32) -> f32 {
    if value.is_finite() {
        (value * 4.0).floor() * 0.25
    } else {
        value
    }
}

#[derive(Debug)]
pub(crate) struct SafetensorPayload {
    path: PathBuf,
    data_start: u64,
    entries: HashMap<String, PayloadEntry>,
}

#[derive(Clone, Debug)]
struct PayloadEntry {
    dtype: Dtype,
    shape: Vec<usize>,
    offsets: [u64; 2],
}

#[derive(Clone, Copy, Debug)]
struct PayloadReadSummary {
    bytes: u64,
    bytes_read: u64,
    checksum: u64,
}

/// Offset basis FNV-1a 64 bits (constante officielle de l'algorithme FNV).
const FNV1A64_OFFSET_BASIS: u64 = 0xcbf29ce484222325;

/// Prime FNV-1a 64 bits (constante officielle de l'algorithme FNV).
const FNV1A64_PRIME: u64 = 0x100000001b3;

/// Combine un octet dans un hash FNV-1a 64 bits (`(hash XOR octet) * prime`).
///
/// Sert de checksum de non-régression sur les octets bruts du payload codec
/// TTS (détecte une corruption/dérive de poids) : ce n'est PAS un usage
/// cryptographique, FNV n'offre aucune résistance aux collisions adverses.
fn fnv1a64_update(hash: u64, byte: u8) -> u64 {
    (hash ^ u64::from(byte)).wrapping_mul(FNV1A64_PRIME)
}

impl SafetensorPayload {
    pub(crate) fn open(path: &Path) -> Result<Self> {
        let mut file = std::fs::File::open(path).map_err(|source| InferError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let mut len_bytes = [0_u8; 8];
        file.read_exact(&mut len_bytes)
            .map_err(|source| InferError::Io {
                path: path.to_path_buf(),
                source,
            })?;
        let header_len = u64::from_le_bytes(len_bytes);
        if header_len == 0 || header_len > 128 * 1024 * 1024 {
            return Err(InferError::SafetensorsHeader {
                path: path.to_path_buf(),
                message: format!("taille header invalide: {header_len}"),
            });
        }
        let header_len_usize =
            usize::try_from(header_len).map_err(|_| InferError::SafetensorsHeader {
                path: path.to_path_buf(),
                message: format!("taille header non représentable: {header_len}"),
            })?;
        let mut header = vec![0_u8; header_len_usize];
        file.read_exact(&mut header)
            .map_err(|source| InferError::Io {
                path: path.to_path_buf(),
                source,
            })?;
        let json: serde_json::Value =
            serde_json::from_slice(&header).map_err(|source| InferError::Json {
                path: path.to_path_buf(),
                source,
            })?;
        let object = json
            .as_object()
            .ok_or_else(|| InferError::SafetensorsHeader {
                path: path.to_path_buf(),
                message: "header JSON non objet".to_string(),
            })?;
        let mut entries = HashMap::new();
        for (name, value) in object {
            if name == "__metadata__" {
                continue;
            }
            let raw: RawPayloadEntry =
                serde_json::from_value(value.clone()).map_err(|source| InferError::Json {
                    path: path.to_path_buf(),
                    source,
                })?;
            entries.insert(
                name.clone(),
                PayloadEntry {
                    dtype: parse_dtype(&raw.dtype).ok_or_else(|| {
                        InferError::Config(format!(
                            "dtype safetensors TTS non supporté pour {name}: {}",
                            raw.dtype
                        ))
                    })?,
                    shape: raw.shape,
                    offsets: raw.data_offsets,
                },
            );
        }
        Ok(Self {
            path: path.to_path_buf(),
            data_start: 8 + header_len,
            entries,
        })
    }

    pub(crate) fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    pub(crate) fn names(&self) -> Vec<String> {
        self.entries.keys().cloned().collect()
    }

    fn entry(&self, name: &str) -> Result<&PayloadEntry> {
        self.entries
            .get(name)
            .ok_or_else(|| InferError::MissingWeight(name.to_string()))
    }

    fn read_payload_summary(&self) -> Result<PayloadReadSummary> {
        let mut bytes = 0_u64;
        let mut bytes_read = 0_u64;
        let mut checksum = FNV1A64_OFFSET_BASIS;
        let mut names = self.entries.keys().collect::<Vec<_>>();
        names.sort();
        for name in names {
            let entry = self.entry(name)?;
            let entry_bytes = self.read_entry_bytes(entry)?;
            let len = u64::try_from(entry_bytes.len()).map_err(|_| {
                InferError::Shape("entrée safetensors codec trop grande".to_string())
            })?;
            bytes = bytes.checked_add(len).ok_or_else(|| {
                InferError::Shape("payload safetensors codec trop grand".to_string())
            })?;
            bytes_read = bytes_read.checked_add(len).ok_or_else(|| {
                InferError::Shape("payload safetensors codec lu trop grand".to_string())
            })?;
            for byte in entry_bytes {
                checksum = fnv1a64_update(checksum, byte);
            }
        }
        Ok(PayloadReadSummary {
            bytes,
            bytes_read,
            checksum,
        })
    }

    pub(crate) fn read_dense_tensor(&self, name: &str) -> Result<Tensor> {
        let entry = self.entry(name)?;
        let bytes = self.read_entry_bytes(entry)?;
        Tensor::from_vec(
            entry.shape.clone(),
            bytes_to_dense_f32(&bytes, entry.dtype, name)?,
        )
    }

    pub(crate) fn read_u32_tensor(&self, name: &str) -> Result<Vec<u32>> {
        let entry = self.entry(name)?;
        if entry.dtype != Dtype::U32 {
            return Err(InferError::UnsupportedDtype {
                name: name.to_string(),
                dtype: entry.dtype,
            });
        }
        let bytes = self.read_entry_bytes(entry)?;
        bytes_to_u32(&bytes, name)
    }

    fn read_entry_bytes(&self, entry: &PayloadEntry) -> Result<Vec<u8>> {
        let len = entry.offsets[1]
            .checked_sub(entry.offsets[0])
            .ok_or_else(|| InferError::Shape("offset safetensors inversé".to_string()))?;
        let offset = self
            .data_start
            .checked_add(entry.offsets[0])
            .ok_or_else(|| InferError::Shape("offset safetensors absolu trop grand".to_string()))?;
        let len_usize = usize::try_from(len)
            .map_err(|_| InferError::Shape(format!("entrée safetensors trop grande: {len}")))?;
        let mut file = std::fs::File::open(&self.path).map_err(|source| InferError::Io {
            path: self.path.clone(),
            source,
        })?;
        file.seek(SeekFrom::Start(offset))
            .map_err(|source| InferError::Io {
                path: self.path.clone(),
                source,
            })?;
        let mut bytes = vec![0_u8; len_usize];
        file.read_exact(&mut bytes)
            .map_err(|source| InferError::Io {
                path: self.path.clone(),
                source,
            })?;
        Ok(bytes)
    }
}

#[derive(Debug, Deserialize)]
struct RawPayloadEntry {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

fn parse_dtype(dtype: &str) -> Option<Dtype> {
    match dtype {
        "F32" => Some(Dtype::F32),
        "F16" => Some(Dtype::F16),
        "BF16" => Some(Dtype::BF16),
        "U32" => Some(Dtype::U32),
        "F8_E4M3" => Some(Dtype::F8_E4M3),
        "F8_E5M2" => Some(Dtype::F8_E5M2),
        _ => None,
    }
}

fn bytes_to_u32(bytes: &[u8], name: &str) -> Result<Vec<u32>> {
    let chunks = bytes.chunks_exact(4);
    if !chunks.remainder().is_empty() {
        return Err(InferError::Shape(format!(
            "tensor {name} U32 avec {} octets non multiple de 4",
            bytes.len()
        )));
    }
    chunks
        .map(|chunk| {
            let arr = <[u8; 4]>::try_from(chunk)
                .map_err(|_| InferError::Shape(format!("chunk U32 invalide pour {name}")))?;
            Ok(u32::from_le_bytes(arr))
        })
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TtsTalkerCatalog {
    keys: Vec<String>,
    pub tensor_count: usize,
    pub has_talker_weights: bool,
    pub has_speaker_encoder_weights: bool,
}

impl TtsTalkerCatalog {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let keys = read_safetensors_keys(path)?;
        Ok(Self::from_keys(keys))
    }

    #[must_use]
    pub fn from_keys(mut keys: Vec<String>) -> Self {
        keys.sort();
        let has_talker_weights = keys.iter().any(|key| key.starts_with("talker."));
        let has_speaker_encoder_weights =
            keys.iter().any(|key| key.starts_with("speaker_encoder."));
        Self {
            tensor_count: keys.len(),
            keys,
            has_talker_weights,
            has_speaker_encoder_weights,
        }
    }

    #[must_use]
    pub fn keys(&self) -> &[String] {
        &self.keys
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TtsCodecCatalog {
    keys: Vec<String>,
    pub tensor_count: usize,
    pub has_decoder_weights: bool,
    pub has_encoder_weights: bool,
    pub has_codebook_stats: bool,
}

impl TtsCodecCatalog {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let keys = read_safetensors_keys(path)?;
        Ok(Self::from_keys(keys))
    }

    #[must_use]
    pub fn from_keys(mut keys: Vec<String>) -> Self {
        keys.sort();
        let has_decoder_weights = keys.iter().any(is_decoder_weight_key);
        let has_encoder_weights = keys.iter().any(is_encoder_weight_key);
        let has_codebook_stats = keys.iter().any(|key| {
            key.ends_with("._codebook.cluster_usage")
                || key.ends_with("._codebook.embedding_sum")
                || key.ends_with(".codebook.cluster_usage")
                || key.ends_with(".codebook.embed_sum")
        });
        Self {
            tensor_count: keys.len(),
            keys,
            has_decoder_weights,
            has_encoder_weights,
            has_codebook_stats,
        }
    }

    #[must_use]
    pub fn keys(&self) -> &[String] {
        &self.keys
    }
}

fn is_decoder_weight_key(key: &String) -> bool {
    key.starts_with("decoder.")
        || key.starts_with("model.decoder.")
        || key.contains(".decode.")
        || key.contains("semantic_model")
}

fn is_encoder_weight_key(key: &String) -> bool {
    key.starts_with("encoder.")
        || key.starts_with("model.encoder.")
        || key.contains(".encode.")
        || key.contains("seanet_encoder")
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TtsModelKind {
    VoiceDesign,
    Base,
    CustomVoice,
    Other(String),
}

#[derive(Debug, Clone, Deserialize)]
pub struct TtsQuantConfig {
    pub group_size: i32,
    pub bits: i32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TtsCodePredictorConfig {
    pub head_dim: i32,
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub num_hidden_layers: i32,
    pub num_code_groups: i32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub vocab_size: i32,
    #[serde(default = "default_code_predictor_max_pos")]
    pub max_position_embeddings: i32,
    #[serde(default)]
    pub attention_bias: bool,
}

fn default_code_predictor_max_pos() -> i32 {
    65_536
}

#[derive(Debug, Clone, Deserialize)]
pub struct TtsRopeScaling {
    #[serde(default)]
    pub interleaved: bool,
    pub mrope_section: Vec<i32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TtsTalkerConfig {
    pub head_dim: i32,
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub num_hidden_layers: i32,
    pub num_code_groups: i32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub vocab_size: i32,
    pub text_vocab_size: i32,
    pub text_hidden_size: i32,
    #[serde(default = "default_talker_max_pos")]
    pub max_position_embeddings: i32,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    pub rope_scaling: TtsRopeScaling,
    pub code_predictor_config: TtsCodePredictorConfig,
    pub codec_bos_id: i32,
    pub codec_eos_token_id: i32,
    pub codec_pad_id: i32,
    pub codec_think_id: i32,
    pub codec_nothink_id: i32,
    pub codec_think_bos_id: i32,
    pub codec_think_eos_id: i32,
    #[serde(default)]
    pub codec_language_id: HashMap<String, i32>,
}

fn default_talker_max_pos() -> i32 {
    32_768
}

fn default_hidden_act() -> String {
    "silu".to_string()
}

#[derive(Debug, Clone, Deserialize)]
pub struct TtsSpeakerEncoderConfig {
    pub enc_dim: i32,
    #[serde(default = "default_speaker_sr")]
    pub sample_rate: i32,
}

fn default_speaker_sr() -> i32 {
    24_000
}

#[derive(Debug, Clone, Deserialize)]
pub struct TtsModelConfig {
    pub talker_config: TtsTalkerConfig,
    pub quantization: TtsQuantConfig,
    pub tts_bos_token_id: i32,
    pub tts_eos_token_id: i32,
    pub tts_pad_token_id: i32,
    #[serde(default = "default_tts_model_type")]
    pub tts_model_type: String,
    #[serde(default)]
    pub speaker_encoder_config: Option<TtsSpeakerEncoderConfig>,
}

impl TtsModelConfig {
    #[must_use]
    pub fn model_kind(&self) -> TtsModelKind {
        match self.tts_model_type.as_str() {
            "voice_design" => TtsModelKind::VoiceDesign,
            "base" => TtsModelKind::Base,
            "custom_voice" => TtsModelKind::CustomVoice,
            other => TtsModelKind::Other(other.to_string()),
        }
    }
}

fn default_tts_model_type() -> String {
    "voice_design".to_string()
}

#[derive(Debug, Clone, Deserialize)]
pub struct TtsDecoderConfig {
    pub latent_dim: i32,
    pub codebook_dim: i32,
    pub codebook_size: i32,
    pub decoder_dim: i32,
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub head_dim: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub num_hidden_layers: i32,
    pub num_quantizers: i32,
    pub num_semantic_quantizers: i32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub layer_scale_initial_scale: f32,
    #[serde(default = "default_codec_max_pos")]
    pub max_position_embeddings: i32,
    #[serde(default)]
    pub attention_bias: bool,
    pub upsample_rates: Vec<i32>,
    pub upsampling_ratios: Vec<i32>,
}

fn default_codec_max_pos() -> i32 {
    8_000
}

#[derive(Debug, Clone, Deserialize)]
pub struct TtsEncoderConfig {
    pub audio_channels: i32,
    pub codebook_dim: i32,
    pub codebook_size: i32,
    pub compress: i32,
    pub dilation_growth_rate: i32,
    pub head_dim: i32,
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub kernel_size: i32,
    pub last_kernel_size: i32,
    pub layer_scale_initial_scale: f32,
    pub num_attention_heads: i32,
    pub num_filters: i32,
    pub num_hidden_layers: i32,
    pub num_key_value_heads: i32,
    pub num_quantizers: i32,
    pub num_residual_layers: i32,
    pub num_semantic_quantizers: i32,
    pub residual_kernel_size: i32,
    pub rope_theta: f32,
    pub sampling_rate: i32,
    pub sliding_window: i32,
    #[serde(rename = "_frame_rate")]
    pub frame_rate: f32,
    pub upsampling_ratios: Vec<i32>,
    #[serde(default = "default_true")]
    pub use_causal_conv: bool,
    #[serde(default)]
    pub use_conv_shortcut: bool,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize)]
pub struct TtsCodecConfig {
    pub decoder_config: TtsDecoderConfig,
    pub decode_upsample_rate: i32,
    pub output_sample_rate: i32,
    pub encoder_valid_num_quantizers: i32,
    #[serde(default)]
    pub encoder_config: Option<TtsEncoderConfig>,
}

fn require_file(path: &Path, what: &'static str) -> Result<()> {
    if path.is_file() {
        Ok(())
    } else {
        Err(InferError::MissingArtifact {
            path: path.to_path_buf(),
            what,
        })
    }
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let file = std::fs::File::open(path).map_err(|source| InferError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::from_reader(file).map_err(|source| InferError::Json {
        path: path.to_path_buf(),
        source,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::{serialize, Dtype, View};
    use std::borrow::Cow;

    /// Charge le codec TTS réel depuis le snapshot VoiceDesign en cache HF.
    fn load_voicedesign_codec() -> Option<TtsCodec> {
        let model_dir = local_tts_snapshot(
            "RETI_QWEN3_TTS_VOICEDESIGN_DIR",
            "models--mlx-community--Qwen3-TTS-12Hz-1.7B-VoiceDesign-6bit",
        )?;
        let codec_weights = model_dir.join("speech_tokenizer/model.safetensors");
        let codec_config_path = model_dir.join("speech_tokenizer/config.json");
        let payload = SafetensorPayload::open(&codec_weights).ok()?;
        let codec_config: TtsCodecConfig = read_json(&codec_config_path).ok()?;
        TtsCodec::load(&payload, &codec_config).ok()
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

    #[test]
    fn repeat_frame_stop_env_defaults_and_allows_disable() {
        assert_eq!(
            tts_repeat_frame_stop_from_env(None),
            DEFAULT_REPEAT_FRAME_STOP
        );
        assert_eq!(
            tts_repeat_frame_stop_from_env(Some("bad")),
            DEFAULT_REPEAT_FRAME_STOP
        );
        assert_eq!(tts_repeat_frame_stop_from_env(Some(" 12 ")), 12);
        assert_eq!(tts_repeat_frame_stop_from_env(Some("0")), 0);
    }

    #[test]
    fn repeat_frame_stop_trips_only_at_threshold() {
        assert!(!repeat_frame_stop_tripped(32, 0));
        assert!(!repeat_frame_stop_tripped(7, 8));
        assert!(repeat_frame_stop_tripped(8, 8));
        assert!(repeat_frame_stop_tripped(9, 8));
    }

    #[test]
    fn clone_frame_cap_matches_legacy_icl_bound() {
        assert_eq!(clone_effective_frame_cap(160, 4), 75);
        assert_eq!(clone_effective_frame_cap(160, 40), 160);
        assert_eq!(
            clone_effective_frame_cap(1_000, 200),
            CLONE_GENERATION_HARD_CAP
        );
    }

    #[test]
    fn sampled_talker_penalizes_repeated_cb0() -> Result<()> {
        let params = TtsSampleParams {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 2.0,
            seed: 0,
        };
        let mut sampler = DeterministicSampler::new(0);
        let token = sample_talker_token(&[0.0, 10.0, 9.0], &params, &[], &[1], None, &mut sampler)?;

        assert_eq!(token, 2);
        Ok(())
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

    /// Vecteurs de référence officiels FNV-1a 64 bits (Fowler/Noll/Vo :
    /// offset basis `0xcbf29ce484222325`, prime `0x100000001b3`) : la chaîne
    /// vide reste l'offset basis (aucun octet traité) et `"a"` = premier
    /// octet non trivial (`0x61`). Recalculés indépendamment (Python, à
    /// partir de la seule définition `hash = (hash XOR octet) * prime`) avant
    /// d'être codés en dur, pour ne pas piéger le test avec le code qu'il vérifie.
    #[test]
    fn fnv1a64_update_matches_official_reference_vectors() {
        assert_eq!(FNV1A64_OFFSET_BASIS, 0xcbf29ce484222325);

        let empty = FNV1A64_OFFSET_BASIS;
        assert_eq!(empty, 0xcbf29ce484222325);

        let a = fnv1a64_update(FNV1A64_OFFSET_BASIS, b'a');
        assert_eq!(a, 0xaf63dc4c8601ec8c);

        let foobar = b"foobar".iter().fold(FNV1A64_OFFSET_BASIS, |hash, &byte| {
            fnv1a64_update(hash, byte)
        });
        assert_eq!(foobar, 0x85944171f73967e8);

        let abc = b"abc".iter().fold(FNV1A64_OFFSET_BASIS, |hash, &byte| {
            fnv1a64_update(hash, byte)
        });
        assert_eq!(abc, 0xe71fa2190541574b);
    }

    /// Le checksum de `read_payload_summary` hache les octets bruts LE du
    /// tenseur (pas ses u32/f32 en tant que mots), donc le vecteur attendu se
    /// recalcule à la main sur les 4 octets `1.0_f32.to_le_bytes()` écrits
    /// par `write_safetensors`.
    #[test]
    fn payload_summary_checksum_hashes_raw_tensor_bytes_fnv1a64() -> Result<()> {
        let tmp = tempfile::tempdir().map_err(|source| InferError::Io {
            path: PathBuf::from("tempdir"),
            source,
        })?;
        let path = tmp.path().join("codec.safetensors");
        write_safetensors(&path, &["only"])?;

        let payload = SafetensorPayload::open(&path)?;
        let summary = payload.read_payload_summary()?;

        let expected = 1.0_f32
            .to_le_bytes()
            .iter()
            .fold(FNV1A64_OFFSET_BASIS, |hash, &byte| {
                fnv1a64_update(hash, byte)
            });
        assert_eq!(summary.bytes, 4);
        assert_eq!(summary.bytes_read, 4);
        assert_eq!(summary.checksum, expected);
        assert_eq!(summary.checksum, 0x4b72477f9c5c2f98);
        Ok(())
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

    /// Parité audio codec **GPU vs CPU** sur codes synthétiques (160 + 256 frames).
    ///
    /// Gate = tolérance audio (max_abs ≤ 0,50, rms ≤ 0,10), PAS l'octet-à-octet :
    /// la réduction GPU diffère du scalaire CPU au niveau de l'arrondi f32. Écrit
    /// la dérive mesurée par échantillon dans `/tmp/tts_codec_drift.md`.
    #[test]
    #[ignore = "parité: codec GPU vs CPU (cache HF + Metal requis)"]
    fn codec_gpu_cpu_parity_synthetic() -> Result<()> {
        let Some(codec) = load_voicedesign_codec() else {
            eprintln!("skip: snapshot Qwen3-TTS VoiceDesign absent du cache HF");
            return Ok(());
        };
        if !codec.gpu_active() {
            eprintln!("skip: forward GPU codec inactif (Metal absent ou RETI_TTS_CODEC_GPU=0)");
            return Ok(());
        }
        let mut report = String::from("# Parité codec GPU vs CPU — codes synthétiques\n\n");
        for &n in &[160_usize, 256] {
            let codes = synthetic_codes(n, 16);
            let gpu = codec.decode_codes(&codes)?;
            let cpu = codec.decode_codes_cpu(&codes)?;
            assert_eq!(gpu.len(), cpu.len(), "longueur PCM GPU != CPU (N={n})");
            let (max_abs, rms, mean_abs, signal_max) = drift_stats(&cpu, &gpu);
            report.push_str(&format!(
                "## N = {n} frames ({} échantillons)\n\
- signal_max_abs (CPU): {signal_max:.6}\n\
- **max_abs_diff**: {max_abs:.3e}\n- **rms_diff**: {rms:.3e}\n- mean_abs_diff: {mean_abs:.3e}\n\
- fnv_cpu: {:#018x}\n- fnv_gpu: {:#018x}\n\n",
                gpu.len(),
                fnv1a_f32(&cpu),
                fnv1a_f32(&gpu),
            ));
            eprintln!(
                "codec parité N={n}: max_abs={max_abs:.3e} rms={rms:.3e} mean_abs={mean_abs:.3e}"
            );
            assert!(max_abs <= 0.5, "max_abs_diff {max_abs} > 0.5 (N={n})");
            assert!(rms <= 0.1, "rms_diff {rms} > 0.1 (N={n})");
        }
        std::fs::write("/tmp/tts_codec_drift.md", &report).map_err(|source| InferError::Io {
            path: PathBuf::from("/tmp/tts_codec_drift.md"),
            source,
        })?;
        eprintln!("dérive écrite dans /tmp/tts_codec_drift.md");
        Ok(())
    }

    /// Parité audio codec **e2e VoiceDesign** : codes réels (talker greedy) décodés
    /// GPU vs CPU, gate tolérance audio (max_abs ≤ 0,50, rms ≤ 0,10).
    #[test]
    #[ignore = "parité: codec GPU vs CPU sur codes réels VoiceDesign (cache HF + Metal requis)"]
    fn codec_gpu_cpu_parity_e2e() -> Result<()> {
        let Some(model_dir) = local_tts_snapshot(
            "RETI_QWEN3_TTS_VOICEDESIGN_DIR",
            "models--mlx-community--Qwen3-TTS-12Hz-1.7B-VoiceDesign-6bit",
        ) else {
            eprintln!("skip: snapshot Qwen3-TTS VoiceDesign absent du cache HF");
            return Ok(());
        };
        let rust = TtsModel::load_local(model_dir)?;
        if !rust.codec.gpu_active() {
            eprintln!("skip: forward GPU codec inactif");
            return Ok(());
        }
        let text = "Bonjour, ceci est un test de parité du décodeur codec sur des codes réels.";
        let codes = rust.generate_codes_greedy(text, 128)?;
        let gpu = rust.codec.decode_codes(&codes)?;
        let cpu = rust.codec.decode_codes_cpu(&codes)?;
        assert_eq!(gpu.len(), cpu.len(), "longueur PCM GPU != CPU e2e");
        let (max_abs, rms, mean_abs, signal_max) = drift_stats(&cpu, &gpu);
        eprintln!(
            "codec parité e2e: frames={} samples={} signal_max={signal_max:.4} max_abs={max_abs:.3e} rms={rms:.3e} mean_abs={mean_abs:.3e}",
            codes.len(),
            gpu.len(),
        );
        assert!(max_abs <= 0.5, "max_abs_diff e2e {max_abs} > 0.5");
        assert!(rms <= 0.1, "rms_diff e2e {rms} > 0.1");
        Ok(())
    }

    /// Parité codec streaming : suffixes incrémentaux vs préfixe complet.
    #[test]
    #[ignore = "parité: codec streaming incrémental vs batch (cache HF + Metal requis)"]
    fn codec_streaming_incremental_matches_full_prefix_on_representative_reply() -> Result<()> {
        let Some(model_dir) = local_tts_snapshot(
            "RETI_QWEN3_TTS_VOICEDESIGN_DIR",
            "models--mlx-community--Qwen3-TTS-12Hz-1.7B-VoiceDesign-6bit",
        ) else {
            eprintln!("skip: snapshot Qwen3-TTS VoiceDesign absent du cache HF");
            return Ok(());
        };
        let rust = TtsModel::load_local(model_dir)?;
        if !rust.codec.gpu_active() {
            eprintln!("skip: forward GPU codec inactif");
            return Ok(());
        }
        let text = "D'accord, je vérifie l'état du projet et je te donne le point utile. La priorité est de garder la réponse courte pour lancer l'audio plus vite.";
        let codes = rust.generate_codes_greedy(text, 160)?;
        let full = rust.codec.decode_codes(&codes)?;
        let mut state = rust.codec.new_stream_state();
        let mut streamed = Vec::new();
        let mut end = 0_usize;
        let mut next = 4_usize;
        while end < codes.len() {
            let target = next.min(codes.len());
            let chunk = rust
                .codec
                .decode_codes_streaming(&mut state, &codes[..target])?;
            streamed.extend_from_slice(&chunk);
            end = target;
            next = next.saturating_mul(2).max(end + 1);
        }

        assert_eq!(streamed.len(), full.len(), "longueur streaming != batch");
        let (max_abs, rms, mean_abs, signal_max) = drift_stats(&full, &streamed);
        let report = format!(
            "# Parité codec streaming incrémental\n\nframes={}\nsamples={}\n\
signal_max_abs={signal_max:.6}\nmax_abs_diff={max_abs:.3e}\nrms_diff={rms:.3e}\n\
mean_abs_diff={mean_abs:.3e}\n",
            codes.len(),
            full.len(),
        );
        std::fs::write("/tmp/tts_codec_streaming_incremental.md", &report).map_err(|source| {
            InferError::Io {
                path: PathBuf::from("/tmp/tts_codec_streaming_incremental.md"),
                source,
            }
        })?;
        eprintln!(
            "codec streaming incrémental: frames={} max_abs={max_abs:.3e} rms={rms:.3e} mean_abs={mean_abs:.3e}",
            codes.len(),
        );
        assert!(max_abs <= 0.5, "max_abs_diff streaming {max_abs} > 0.5");
        assert!(rms <= 0.1, "rms_diff streaming {rms} > 0.1");
        Ok(())
    }

    /// Parité streaming : `synthesize_greedy_streaming` (préfixe croissant +
    /// codec incrémental) doit produire un audio dans la tolérance codec de
    /// `synthesize_greedy` (batch).
    /// Mesure aussi un proxy de TTFA : temps jusqu'au 1er chunk émis vs synthèse totale.
    #[test]
    #[ignore = "parité+TTFA: streaming TTS ~= batch (cache HF + Metal requis)"]
    fn tts_streaming_matches_batch() -> Result<()> {
        let Some(model_dir) = local_tts_snapshot(
            "RETI_QWEN3_TTS_VOICEDESIGN_DIR",
            "models--mlx-community--Qwen3-TTS-12Hz-1.7B-VoiceDesign-6bit",
        ) else {
            eprintln!("skip: snapshot Qwen3-TTS VoiceDesign absent du cache HF");
            return Ok(());
        };
        let rust = TtsModel::load_local(model_dir)?;
        let text = "Bonjour, ceci est un test de streaming incrémental du décodeur TTS.";
        let max_frames = 128;

        // Batch (oracle) + chrono total.
        let batch_start = Instant::now();
        let batch = rust.synthesize_greedy(text, max_frames)?;
        let batch_ms = batch_start.elapsed().as_secs_f64() * 1e3;

        // Streaming : agrège les chunks + capture le temps jusqu'au 1er chunk (TTFA proxy).
        let stream_start = Instant::now();
        let mut streamed: Vec<f32> = Vec::new();
        let mut first_chunk_ms: Option<f64> = None;
        let mut chunks = 0_usize;
        let out = rust.synthesize_greedy_streaming(text, max_frames, |chunk| {
            if first_chunk_ms.is_none() {
                first_chunk_ms = Some(stream_start.elapsed().as_secs_f64() * 1e3);
            }
            chunks += 1;
            streamed.extend_from_slice(chunk);
            Ok(())
        })?;
        let stream_ms = stream_start.elapsed().as_secs_f64() * 1e3;

        // Parité audio (callback agrégé == out.samples ~= batch).
        assert_eq!(
            streamed.len(),
            batch.samples.len(),
            "longueur streaming != batch"
        );
        assert_eq!(
            out.samples.len(),
            batch.samples.len(),
            "longueur out != batch"
        );
        let (max_abs, rms, mean_abs, signal_max) = drift_stats(&batch.samples, &streamed);
        let (out_max_abs, out_rms, _, _) = drift_stats(&batch.samples, &out.samples);
        assert!(max_abs <= 0.5, "max_abs_diff streaming {max_abs} > 0.5");
        assert!(rms <= 0.1, "rms_diff streaming {rms} > 0.1");
        assert!(out_max_abs <= 0.5, "max_abs_diff out {out_max_abs} > 0.5");
        assert!(out_rms <= 0.1, "rms_diff out {out_rms} > 0.1");
        let ttfa = first_chunk_ms.unwrap_or(stream_ms);
        let report = format!(
            "# TTFA streaming TTS\n\nframes={}\nsamples={}\nchunks={chunks}\n\
ttfa_stream_ms={ttfa:.1}\nttfa_batch_ms={batch_ms:.1}\nstream_total_ms={stream_ms:.1}\n\
speedup_ttfa={:.2}\nsignal_max_abs={signal_max:.6}\nmax_abs_diff={max_abs:.3e}\n\
rms_diff={rms:.3e}\nmean_abs_diff={mean_abs:.3e}\ncodec_tolerance=oui\n",
            batch.codes.len(),
            batch.samples.len(),
            batch_ms / ttfa.max(1.0),
        );
        std::fs::write("/tmp/tts_streaming_ttfa.md", &report).map_err(|source| InferError::Io {
            path: PathBuf::from("/tmp/tts_streaming_ttfa.md"),
            source,
        })?;
        eprintln!(
            "streaming: frames={} chunks={chunks} TTFA={ttfa:.0}ms (batch TTFA={batch_ms:.0}ms, ×{:.1}) total={stream_ms:.0}ms max_abs={max_abs:.3e} rms={rms:.3e}",
            batch.codes.len(),
            batch_ms / ttfa.max(1.0),
        );
        Ok(())
    }

    #[test]
    #[ignore = "perf: profil CPU du codec TTS sur N frames croissant (cache HF requis)"]
    fn codec_perf_profile() -> Result<()> {
        let Some(codec) = load_voicedesign_codec() else {
            eprintln!("skip: snapshot Qwen3-TTS VoiceDesign absent du cache HF");
            return Ok(());
        };
        let quantizers = 16_usize;
        let frame_counts: &[usize] = match std::env::var("CODEC_PROFILE_N") {
            Ok(spec) if !spec.is_empty() => Box::leak(
                spec.split(',')
                    .filter_map(|s| s.trim().parse::<usize>().ok())
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            ),
            _ => &[2, 16, 64],
        };
        let mut report = String::new();
        report.push_str("# Profil CPU codec TTS (decode_codes)\n\n");
        report.push_str(
            "Temps total et par sous-étape (ms), mesuré via `decode_codes_profiled`.\n\n",
        );
        for &n in frame_counts {
            let codes = synthetic_codes(n, quantizers);
            // Préchauffe le cache pour le plus petit N afin d'éviter le coût
            // d'amorçage (allocateur, pages) dans la mesure publiée.
            let (samples, timings) = codec.decode_codes_profiled(&codes)?;
            let total: std::time::Duration = timings.iter().map(|(_, d)| *d).sum();
            // Agrège par étiquette (les blocs décodeur partagent un label).
            let mut agg: Vec<(&'static str, std::time::Duration)> = Vec::new();
            for (label, dur) in &timings {
                if let Some(slot) = agg.iter_mut().find(|(l, _)| l == label) {
                    slot.1 += *dur;
                } else {
                    agg.push((label, *dur));
                }
            }
            report.push_str(&format!(
                "## N = {n} frames ({} échantillons PCM, {:.3} s audio)\n\n",
                samples.len(),
                samples.len() as f32 / codec.sample_rate() as f32
            ));
            report.push_str(&format!(
                "- **total**: {:.2} ms\n",
                total.as_secs_f64() * 1e3
            ));
            for (label, dur) in &agg {
                report.push_str(&format!(
                    "  - {label}: {:.2} ms ({:.1} %)\n",
                    dur.as_secs_f64() * 1e3,
                    dur.as_secs_f64() / total.as_secs_f64() * 100.0
                ));
            }
            report.push('\n');
            eprintln!(
                "codec profil N={n}: total={:.1}ms samples={}",
                total.as_secs_f64() * 1e3,
                samples.len()
            );
        }
        std::fs::write("/tmp/codec_perf_profile.md", &report).map_err(|source| InferError::Io {
            path: PathBuf::from("/tmp/codec_perf_profile.md"),
            source,
        })?;
        eprintln!("profil écrit dans /tmp/codec_perf_profile.md");
        Ok(())
    }

    /// Mesure le rtf de la génération TTS (talker + code_predictor) et du codec.
    ///
    /// Harnais avant/après pour le chantier decode résident : sépare le temps de
    /// `generate_codes_greedy` (la cible) du décodage codec→PCM (déjà parallélisé),
    /// avec préchauffe et cooldown. Étiquette via `RTF_LABEL` (défaut `baseline`),
    /// nombre de répétitions via `RTF_REPEATS` (défaut 3), texte via `RTF_TEXT`.
    /// Cible temps réel : rtf < 0.3 (codec 12.5 Hz ⇒ < 24 ms/frame génération).
    #[test]
    #[ignore = "perf: rtf génération TTS VoiceDesign avant/après (cache HF requis, GPU idle)"]
    fn perf_voicedesign_rtf() -> Result<()> {
        const DEFAULT_TEXT: &str = "Bonjour, ceci est un test de synthèse vocale pour \
mesurer le débit en temps réel du décodeur. Nous générons plusieurs phrases afin \
d'obtenir un nombre de frames représentatif et une mesure stable.";
        let text = std::env::var("RTF_TEXT").unwrap_or_else(|_| DEFAULT_TEXT.to_string());
        let label = std::env::var("RTF_LABEL").unwrap_or_else(|_| "baseline".to_string());
        let repeats = std::env::var("RTF_REPEATS")
            .ok()
            .and_then(|spec| spec.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(3);
        let max_frames = std::env::var("RTF_MAX_FRAMES")
            .ok()
            .and_then(|spec| spec.parse::<usize>().ok())
            .unwrap_or(400);

        let Some(model_dir) = local_tts_snapshot(
            "RETI_QWEN3_TTS_VOICEDESIGN_DIR",
            "models--mlx-community--Qwen3-TTS-12Hz-1.7B-VoiceDesign-6bit",
        ) else {
            eprintln!("skip: snapshot Qwen3-TTS VoiceDesign absent du cache HF");
            return Ok(());
        };
        let rust = TtsModel::load_local(model_dir)?;
        let sample_rate = rust.codec.sample_rate() as f64;

        // Préchauffe (allocateur, pages, pipelines Metal) hors mesure.
        let _ = rust.synthesize_greedy(&text, 8)?;

        let mut gen_ms = Vec::with_capacity(repeats);
        let mut codec_ms = Vec::with_capacity(repeats);
        let mut frames = 0_usize;
        let mut audio_s = 0.0_f64;
        for run in 0..repeats {
            let gen_start = Instant::now();
            let codes = rust.generate_codes_greedy(&text, max_frames)?;
            let gen = gen_start.elapsed();
            let codec_start = Instant::now();
            let samples = rust.decode_codes_for_mode(&codes)?;
            let codec = codec_start.elapsed();
            frames = codes.len();
            audio_s = samples.len() as f64 / sample_rate;
            gen_ms.push(gen.as_secs_f64() * 1e3);
            codec_ms.push(codec.as_secs_f64() * 1e3);
            eprintln!(
                "rtf[{label}] run={run} frames={frames} audio={audio_s:.3}s gen={:.1}ms codec={:.1}ms",
                gen.as_secs_f64() * 1e3,
                codec.as_secs_f64() * 1e3
            );
            // Cooldown GPU entre les runs (sauf le dernier).
            if run + 1 < repeats {
                std::thread::sleep(std::time::Duration::from_secs(2));
            }
        }
        let median = |values: &[f64]| -> f64 {
            let mut sorted = values.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            sorted[sorted.len() / 2]
        };
        let gen_med = median(&gen_ms);
        let codec_med = median(&codec_ms);
        let gen_min = gen_ms.iter().copied().fold(f64::INFINITY, f64::min);
        let ms_per_frame = if frames > 0 {
            gen_med / frames as f64
        } else {
            0.0
        };
        let rtf_gen = if audio_s > 0.0 {
            gen_med / 1e3 / audio_s
        } else {
            0.0
        };
        let rtf_total = if audio_s > 0.0 {
            (gen_med + codec_med) / 1e3 / audio_s
        } else {
            0.0
        };
        let report = format!(
            "# rtf TTS génération — {label}\n\n\
text={text:?}\nframes={frames}\naudio_s={audio_s:.3}\nrepeats={repeats}\n\
gen_ms_median={gen_med:.2}\ngen_ms_min={gen_min:.2}\ncodec_ms_median={codec_med:.2}\n\
ms_per_frame_gen={ms_per_frame:.3}\nrtf_gen={rtf_gen:.4}\nrtf_total={rtf_total:.4}\n\
cible_rtf=0.3000\ncible_ms_per_frame=24.000\n"
        );
        let path = format!("/tmp/tts_rtf_{label}.md");
        std::fs::write(&path, &report).map_err(|source| InferError::Io {
            path: PathBuf::from(&path),
            source,
        })?;
        eprintln!("rtf[{label}] => gen_median={gen_med:.1}ms ms/frame={ms_per_frame:.2} rtf_gen={rtf_gen:.4} rtf_total={rtf_total:.4} (écrit {path})");
        Ok(())
    }

    /// Empreinte audio de référence du codec (160 frames synthétiques).
    ///
    /// Capturée sur le code scalaire d'origine (commit base `3aa5112`) AVANT
    /// l'optimisation. Toute évolution du codec doit la préserver à l'octet près
    /// (ou justifier une nouvelle baseline). Voir `codec_emit_golden` pour
    /// recalculer la valeur.
    const CODEC_GOLDEN_HASH: u64 = 0xcc45_71e4_1a09_84c1;
    const CODEC_GOLDEN_LEN: usize = 307_200;

    #[test]
    #[ignore = "parité: sortie codec CPU byte-identique vs baseline scalaire (cache HF requis)"]
    fn codec_parity_golden() -> Result<()> {
        let Some(codec) = load_voicedesign_codec() else {
            eprintln!("skip: snapshot Qwen3-TTS VoiceDesign absent du cache HF");
            return Ok(());
        };
        let codes = synthetic_codes(160, 16);
        // Chemin CPU explicitement : il reste l'oracle byte-identique. Le forward
        // GPU (`decode_codes`) ne diffère qu'à l'arrondi f32 (cf.
        // `codec_gpu_cpu_parity_*`, gate tolérance audio).
        let samples = codec.decode_codes_cpu(&codes)?;
        assert_eq!(
            samples.len(),
            CODEC_GOLDEN_LEN,
            "longueur PCM codec changée"
        );
        assert_eq!(
            fnv1a_f32(&samples),
            CODEC_GOLDEN_HASH,
            "sortie PCM codec non byte-identique vs baseline scalaire"
        );
        Ok(())
    }

    #[test]
    #[ignore = "perf: capture l'empreinte audio de référence (cache HF requis)"]
    fn codec_emit_golden() -> Result<()> {
        let Some(codec) = load_voicedesign_codec() else {
            eprintln!("skip: snapshot Qwen3-TTS VoiceDesign absent du cache HF");
            return Ok(());
        };
        let codes = synthetic_codes(160, 16);
        let start = std::time::Instant::now();
        let samples = codec.decode_codes(&codes)?;
        let elapsed = start.elapsed();
        let hash = fnv1a_f32(&samples);
        let max_abs = samples.iter().copied().fold(0.0_f32, |m, s| m.max(s.abs()));
        let meta = format!(
            "frames=160\nelapsed_ms={:.2}\nlen={}\nhash={hash:#018x}\nmax_abs={max_abs:.8}\nfirst={:?}\nlast={:?}\n",
            elapsed.as_secs_f64() * 1e3,
            samples.len(),
            &samples[..samples.len().min(4)],
            &samples[samples.len().saturating_sub(4)..],
        );
        eprint!("{meta}");
        std::fs::write("/tmp/codec_golden_meta.txt", &meta).map_err(|source| InferError::Io {
            path: PathBuf::from("/tmp/codec_golden_meta.txt"),
            source,
        })?;
        Ok(())
    }

    #[test]
    fn parses_voicedesign_config_defaults() -> Result<()> {
        let cfg: TtsModelConfig = serde_json::from_str(&model_config_json("voice_design", false))
            .map_err(|source| InferError::Json {
            path: PathBuf::from("inline"),
            source,
        })?;
        assert_eq!(cfg.model_kind(), TtsModelKind::VoiceDesign);
        assert_eq!(cfg.talker_config.hidden_act, "silu");
        assert_eq!(
            cfg.talker_config.codec_language_id.get("french").copied(),
            Some(42)
        );
        assert!(cfg.speaker_encoder_config.is_none());
        Ok(())
    }

    #[test]
    fn catalog_detects_clone_capable_weights() {
        let talker = TtsTalkerCatalog::from_keys(vec![
            "speaker_encoder.blocks.0.weight".to_string(),
            "talker.model.text_embedding.weight".to_string(),
        ]);
        let codec = TtsCodecCatalog::from_keys(vec![
            "decoder.model.layers.0.weight".to_string(),
            "encoder.model.layers.0.weight".to_string(),
            "rvq.layers.0._codebook.cluster_usage".to_string(),
            "rvq.layers.0._codebook.embedding_sum".to_string(),
        ]);
        assert!(talker.has_talker_weights);
        assert!(talker.has_speaker_encoder_weights);
        assert!(codec.has_decoder_weights);
        assert!(codec.has_encoder_weights);
        assert!(codec.has_codebook_stats);
    }

    #[test]
    fn loads_local_tts_assets_without_loading_payloads() -> Result<()> {
        let tmp = tempfile::tempdir().map_err(|source| InferError::Io {
            path: PathBuf::from("tempdir"),
            source,
        })?;
        let root = tmp.path();
        let speech = root.join("speech_tokenizer");
        std::fs::create_dir_all(&speech).map_err(|source| InferError::Io {
            path: speech.clone(),
            source,
        })?;
        write(root.join("config.json"), &model_config_json("base", true))?;
        write(root.join("vocab.json"), "{}")?;
        write(root.join("merges.txt"), "#version: 0.2\n")?;
        write(speech.join("config.json"), &codec_config_json(true))?;
        write_safetensors(
            &root.join("model.safetensors"),
            &[
                "speaker_encoder.blocks.0.weight",
                "talker.model.text_embedding.weight",
            ],
        )?;
        write_safetensors(
            &speech.join("model.safetensors"),
            &[
                "decoder.model.layers.0.weight",
                "encoder.model.layers.0.weight",
                "rvq.layers.0._codebook.cluster_usage",
                "rvq.layers.0._codebook.embedding_sum",
            ],
        )?;

        let assets = TtsAssets::load_local(root)?;
        assert_eq!(assets.model_kind(), TtsModelKind::Base);
        assert!(assets.clone_capable());
        assert_eq!(assets.talker_catalog.tensor_count, 2);
        assert_eq!(assets.codec_catalog.tensor_count, 4);
        Ok(())
    }

    #[test]
    fn rejects_missing_talker_weights() -> Result<()> {
        let tmp = tempfile::tempdir().map_err(|source| InferError::Io {
            path: PathBuf::from("tempdir"),
            source,
        })?;
        let root = tmp.path();
        let speech = root.join("speech_tokenizer");
        std::fs::create_dir_all(&speech).map_err(|source| InferError::Io {
            path: speech.clone(),
            source,
        })?;
        write(
            root.join("config.json"),
            &model_config_json("voice_design", false),
        )?;
        write(root.join("vocab.json"), "{}")?;
        write(root.join("merges.txt"), "#version: 0.2\n")?;
        write(speech.join("config.json"), &codec_config_json(false))?;
        write_safetensors(&root.join("model.safetensors"), &["not_talker.weight"])?;
        write_safetensors(
            &speech.join("model.safetensors"),
            &["decoder.model.layers.0.weight"],
        )?;

        let err = TtsAssets::load_local(root).expect_err("invariant: poids talker absents");
        assert!(err.to_string().contains("talker.* weights"));
        Ok(())
    }

    #[test]
    #[ignore = "live: charge le contrat header-only d'un snapshot Qwen3-TTS VoiceDesign"]
    fn live_loads_voicedesign_snapshot_contract() -> Result<()> {
        let Some(model_dir) = local_tts_snapshot(
            "RETI_QWEN3_TTS_VOICEDESIGN_DIR",
            "models--mlx-community--Qwen3-TTS-12Hz-1.7B-VoiceDesign-6bit",
        ) else {
            eprintln!("skip: snapshot Qwen3-TTS VoiceDesign absent du cache HF");
            return Ok(());
        };
        let assets = TtsAssets::load_local(model_dir)?;
        assert_eq!(assets.model_kind(), TtsModelKind::VoiceDesign);
        assert!(!assets.clone_capable());
        assert!(assets.talker_catalog.tensor_count > 0);
        assert!(assets.codec_catalog.tensor_count > 0);
        Ok(())
    }

    #[test]
    #[ignore = "live: charge les payloads Qwen3-TTS VoiceDesign et exécute le talker"]
    fn live_loads_voicedesign_payloads_and_forwards_talker() -> Result<()> {
        let Some(model_dir) = local_tts_snapshot(
            "RETI_QWEN3_TTS_VOICEDESIGN_DIR",
            "models--mlx-community--Qwen3-TTS-12Hz-1.7B-VoiceDesign-6bit",
        ) else {
            eprintln!("skip: snapshot Qwen3-TTS VoiceDesign absent du cache HF");
            return Ok(());
        };
        let model = TtsModel::load_local(model_dir)?;
        let out = model.forward_voicedesign_prefix("Bonjour.")?;
        let summary = model.payload_summary();
        eprintln!(
            "tts payloads: talker_tensors={} codec_tensors={} codec_payload_bytes={} codec_payload_bytes_read={} codec_payload_checksum={:#x} logits_shape={:?}",
            summary.talker_tensor_count,
            summary.codec_tensor_count,
            summary.codec_payload_bytes,
            summary.codec_payload_bytes_read,
            summary.codec_payload_checksum,
            out.logits.shape()
        );
        assert!(summary.talker_tensor_count > 0);
        assert!(summary.codec_tensor_count > 0);
        assert!(summary.codec_payload_bytes > 0);
        assert_eq!(
            summary.codec_payload_bytes_read,
            summary.codec_payload_bytes
        );
        // Le snapshot est un poids réel (HF, live-only) : son checksum FNV-1a
        // 64 n'est pas un vecteur connu à figer ici. L'égalité exacte sur
        // vecteur de référence vit dans les tests isolés
        // `fnv1a64_update_matches_official_reference_vectors` et
        // `payload_summary_checksum_hashes_raw_tensor_bytes_fnv1a64`
        // (payload synthétique déterministe) ; ce test-ci reste un smoke
        // check de non-nullité sur données live.
        assert_ne!(summary.codec_payload_checksum, 0);
        assert_eq!(
            out.logits.shape(),
            &[
                1,
                model.assets.model_config.talker_config.vocab_size as usize
            ]
        );
        Ok(())
    }

    /// Talker VoiceDesign metal-rs ≡ golden mlx-rs figé (sans mlx-rs). Même critère :
    /// argmax identique + `max_abs<=0.20`.
    #[test]
    #[ignore = "golden: charge Qwen3-TTS VoiceDesign (cache HF) pour le talker metal-rs"]
    fn golden_voicedesign_talker_logits_matches_fixture() -> Result<()> {
        const TEXT: &str = "Bonjour.";
        const MAX_ABS_TOLERANCE: f32 = 0.20;

        let Some(model_dir) = local_tts_snapshot(
            "RETI_QWEN3_TTS_VOICEDESIGN_DIR",
            "models--mlx-community--Qwen3-TTS-12Hz-1.7B-VoiceDesign-6bit",
        ) else {
            eprintln!("skip: snapshot Qwen3-TTS VoiceDesign absent du cache HF");
            return Ok(());
        };
        let rust = TtsModel::load_local(model_dir)?;
        let rust_logits = rust
            .forward_voicedesign_prefix(TEXT)?
            .logits
            .as_row()?
            .to_vec();

        let (_, golden) = crate::golden::read_f32("voicedesign_talker_logits")?;
        if rust_logits.len() != golden.len() {
            return Err(InferError::Dimension(format!(
                "logits TTS len rust={} golden={}",
                rust_logits.len(),
                golden.len()
            )));
        }
        let max_abs = max_abs_same_len(&rust_logits, &golden)?;
        assert_eq!(argmax_index(&rust_logits)?, argmax_index(&golden)?);
        assert!(
            max_abs <= MAX_ABS_TOLERANCE,
            "drift talker TTS max_abs={max_abs} > {MAX_ABS_TOLERANCE}"
        );
        Ok(())
    }

    /// Pipeline TTS VoiceDesign metal-rs ≡ golden mlx-rs figé (sans mlx-rs). Mêmes
    /// invariants : rate 24 kHz, longueurs égales, codes frame 0 + 6 premiers groupes
    /// frame 1 identiques, `max_abs<=0.5`, `rms<=0.1`.
    #[test]
    #[ignore = "golden: charge Qwen3-TTS VoiceDesign (cache HF) pour la synthèse metal-rs"]
    fn golden_voicedesign_e2e_audio_matches_fixture() -> Result<()> {
        const TEXT: &str = "Bonjour, test réel.";
        const MAX_FRAMES: usize = 2;
        const PROBE_GROUP: usize = 6;
        const MAX_ABS_TOLERANCE: f32 = 0.50;
        const RMS_TOLERANCE: f32 = 0.10;

        let Some(model_dir) = local_tts_snapshot(
            "RETI_QWEN3_TTS_VOICEDESIGN_DIR",
            "models--mlx-community--Qwen3-TTS-12Hz-1.7B-VoiceDesign-6bit",
        ) else {
            eprintln!("skip: snapshot Qwen3-TTS VoiceDesign absent du cache HF");
            return Ok(());
        };
        let rust = TtsModel::load_local(model_dir)?;
        let rust_audio = rust.synthesize_greedy(TEXT, MAX_FRAMES)?;

        let (_, golden_samples) = crate::golden::read_f32("voicedesign_e2e_samples")?;
        let (codes_shape, codes_flat) = crate::golden::read_i32("voicedesign_e2e_codes")?;
        let frames = codes_shape[0];
        let groups = codes_shape[1];
        let golden_codes: Vec<Vec<i32>> = (0..frames)
            .map(|frame| codes_flat[frame * groups..(frame + 1) * groups].to_vec())
            .collect();

        let common = rust_audio.samples.len().min(golden_samples.len());
        if common == 0 {
            return Err(InferError::Dimension("audio TTS e2e vide".to_string()));
        }
        let max_abs = rust_audio
            .samples
            .iter()
            .zip(golden_samples.iter())
            .take(common)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f32, f32::max);
        let rms = (rust_audio
            .samples
            .iter()
            .zip(golden_samples.iter())
            .take(common)
            .map(|(left, right)| {
                let diff = left - right;
                diff * diff
            })
            .sum::<f32>()
            / common as f32)
            .sqrt();

        assert_eq!(rust_audio.sample_rate, 24_000);
        assert_eq!(rust_audio.samples.len(), golden_samples.len());
        assert_eq!(rust_audio.codes.first(), golden_codes.first());
        assert_eq!(
            &rust_audio.codes[1][..PROBE_GROUP],
            &golden_codes[1][..PROBE_GROUP]
        );
        assert!(
            max_abs <= MAX_ABS_TOLERANCE,
            "drift audio TTS max_abs={max_abs} > {MAX_ABS_TOLERANCE}"
        );
        assert!(
            rms <= RMS_TOLERANCE,
            "drift audio TTS rms={rms} > {RMS_TOLERANCE}"
        );
        Ok(())
    }

    /// Inputs ICL clone metal-rs ≡ golden mlx-rs figé (sans mlx-rs). Mêmes critères :
    /// ids identiques + `max_abs<=0.18` sur input/trailing/tts_pad.
    #[test]
    #[ignore = "golden: charge Qwen3-TTS Base clone (cache HF) pour prepare_icl metal-rs"]
    fn golden_clone_icl_inputs_matches_fixture() -> Result<()> {
        const TEXT: &str = "Bonjour, ceci est un test.";
        const MAX_ABS_TOLERANCE: f32 = 0.18;

        let Some(model_dir) = local_tts_snapshot(
            "RETI_QWEN3_TTS_BASE_DIR",
            "models--mlx-community--Qwen3-TTS-12Hz-1.7B-Base-6bit",
        ) else {
            eprintln!("skip: snapshot Qwen3-TTS Base absent du cache HF");
            return Ok(());
        };
        let (wav_bytes, ref_text) = clone_reference_assets()?;
        let rust = TtsModel::load_clone_local(&model_dir, &wav_bytes, &ref_text)?;
        let prepared = rust.prepare_inputs(TEXT)?;
        let rust_ids = clone_icl_rust_ids(&rust, TEXT)?;

        let (_, golden_ids) = crate::golden::read_i32("clone_icl_ids")?;
        let (_, golden_input) = crate::golden::read_f32("clone_icl_input")?;
        let (_, golden_trailing) = crate::golden::read_f32("clone_icl_trailing")?;
        let (_, golden_tts_pad) = crate::golden::read_f32("clone_icl_tts_pad")?;

        assert_eq!(rust_ids, golden_ids);
        let input_max_abs = max_abs_same_len(prepared.input.data(), &golden_input)?;
        let trailing_max_abs = max_abs_same_len(prepared.trailing.data(), &golden_trailing)?;
        let tts_pad_max_abs = max_abs_same_len(prepared.tts_pad.data(), &golden_tts_pad)?;
        assert!(
            input_max_abs <= MAX_ABS_TOLERANCE,
            "drift input ICL max_abs={input_max_abs} > {MAX_ABS_TOLERANCE}"
        );
        assert!(
            trailing_max_abs <= MAX_ABS_TOLERANCE,
            "drift trailing ICL max_abs={trailing_max_abs} > {MAX_ABS_TOLERANCE}"
        );
        assert!(
            tts_pad_max_abs <= MAX_ABS_TOLERANCE,
            "drift tts_pad ICL max_abs={tts_pad_max_abs} > {MAX_ABS_TOLERANCE}"
        );
        Ok(())
    }

    /// Premier logit clone metal-rs ≡ golden mlx-rs figé (sans mlx-rs). Mêmes critères :
    /// cb0 identique + `max_abs<=0.56`.
    #[test]
    #[ignore = "golden: charge Qwen3-TTS Base clone (cache HF) pour le 1er cb0 metal-rs"]
    fn golden_clone_first_cb0_matches_fixture() -> Result<()> {
        const TEXT: &str = "Bonjour, ceci est un test.";
        const MAX_ABS_TOLERANCE: f32 = 0.56;

        let Some(model_dir) = local_tts_snapshot(
            "RETI_QWEN3_TTS_BASE_DIR",
            "models--mlx-community--Qwen3-TTS-12Hz-1.7B-Base-6bit",
        ) else {
            eprintln!("skip: snapshot Qwen3-TTS Base absent du cache HF");
            return Ok(());
        };
        let (wav_bytes, ref_text) = clone_reference_assets()?;
        let rust = TtsModel::load_clone_local(&model_dir, &wav_bytes, &ref_text)?;
        let (_prefix, rust_logits) = rust.probe_greedy_logits(TEXT, 1, 0, 0)?;
        let cfg = &rust.assets.model_config.talker_config;
        let suppress_start = cfg.vocab_size.checked_sub(1024).ok_or_else(|| {
            InferError::Config(format!(
                "vocab TTS trop petit pour suppression: {}",
                cfg.vocab_size
            ))
        })?;
        let suppress = (suppress_start..cfg.vocab_size)
            .filter(|token| *token != cfg.codec_eos_token_id)
            .collect::<Vec<_>>();
        let rust_cb0 = greedy_talker_token(&rust_logits, &suppress)?;

        let (_, golden_logits) = crate::golden::read_f32("clone_first_cb0_logits")?;
        let (_, golden_token) = crate::golden::read_i32("clone_first_cb0_token")?;
        let max_abs = max_abs_same_len(&rust_logits, &golden_logits)?;
        assert_eq!(rust_cb0, golden_token[0]);
        assert!(
            max_abs <= MAX_ABS_TOLERANCE,
            "drift first cb0 logits max_abs={max_abs} > {MAX_ABS_TOLERANCE}"
        );
        Ok(())
    }

    /// Pipeline clone e2e metal-rs ≡ golden mlx-rs figé (sans mlx-rs). Mêmes tolérances :
    /// `max_abs<=0.95`, `rms<=0.25`.
    #[test]
    #[ignore = "golden: charge Qwen3-TTS Base clone (cache HF) pour la synthèse metal-rs"]
    fn golden_clone_e2e_audio_matches_fixture() -> Result<()> {
        const TEXT: &str = "Bonjour, ceci est un test.";
        const MAX_FRAMES: usize = 2;
        const MAX_ABS_TOLERANCE: f32 = 0.95;
        const RMS_TOLERANCE: f32 = 0.25;

        let Some(model_dir) = local_tts_snapshot(
            "RETI_QWEN3_TTS_BASE_DIR",
            "models--mlx-community--Qwen3-TTS-12Hz-1.7B-Base-6bit",
        ) else {
            eprintln!("skip: snapshot Qwen3-TTS Base absent du cache HF");
            return Ok(());
        };
        let (wav_bytes, ref_text) = clone_reference_assets()?;
        let rust = TtsModel::load_clone_local(&model_dir, &wav_bytes, &ref_text)?;
        let rust_audio = rust.synthesize_greedy(TEXT, MAX_FRAMES)?;

        let (_, golden_samples) = crate::golden::read_f32("clone_e2e_samples")?;
        let common = rust_audio.samples.len().min(golden_samples.len());
        if common == 0 {
            return Err(InferError::Dimension("audio clone e2e vide".to_string()));
        }
        let max_abs = rust_audio
            .samples
            .iter()
            .zip(golden_samples.iter())
            .take(common)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f32, f32::max);
        let rms = (rust_audio
            .samples
            .iter()
            .zip(golden_samples.iter())
            .take(common)
            .map(|(left, right)| {
                let diff = left - right;
                diff * diff
            })
            .sum::<f32>()
            / common as f32)
            .sqrt();
        assert!(
            max_abs <= MAX_ABS_TOLERANCE,
            "drift clone max_abs={max_abs} > {MAX_ABS_TOLERANCE}"
        );
        assert!(
            rms <= RMS_TOLERANCE,
            "drift clone rms={rms} > {RMS_TOLERANCE}"
        );
        Ok(())
    }

    /// Charge le couple (WAV, texte) de référence du clone (reti-fr) — partagé par les
    /// tests `golden_clone_*`.
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
        let target_chat =
            format!("<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n");
        let target_ids = rust.encode_ids(&target_chat)?;
        Ok(ctx
            .ref_text_ids
            .iter()
            .chain(&target_ids[3..target_ids.len() - 5])
            .copied()
            .collect::<Vec<_>>())
    }

    #[test]
    #[ignore = "live: charge le snapshot Qwen3-TTS Base pour diagnostiquer le cap de frames clone"]
    fn live_clone_generation_diagnoses_frame_cap() -> Result<()> {
        const TEXT: &str = "Bonjour, ceci est un test.";
        const MAX_FRAMES: usize = 2;

        let Some(model_dir) = local_tts_snapshot(
            "RETI_QWEN3_TTS_BASE_DIR",
            "models--mlx-community--Qwen3-TTS-12Hz-1.7B-Base-6bit",
        ) else {
            eprintln!("skip: snapshot Qwen3-TTS Base absent du cache HF");
            return Ok(());
        };
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
        let rust = TtsModel::load_clone_local(&model_dir, &wav_bytes, &ref_text)?;
        let prepared = rust.prepare_inputs(TEXT)?;
        let ref_frames = rust.clone_ctx.as_ref().map_or(0, |ctx| ctx.ref_codes.len());
        let codes = rust.generate_codes_greedy_trace(TEXT, MAX_FRAMES, true, &mut |_| Ok(()))?;
        let reached_eos = codes.len() < MAX_FRAMES;
        let metrics = format!(
            "text={TEXT:?}\nrequested_max_frames={MAX_FRAMES}\nhard_cap={CLONE_GENERATION_HARD_CAP}\neffective_max_frames={}\ninput_rows={}\ntrailing_rows={}\nref_frames={ref_frames}\ngenerated_frames={}\nreached_eos={reached_eos}\nfirst_frame={:?}\n",
            MAX_FRAMES.min(CLONE_GENERATION_HARD_CAP),
            prepared.input.shape()[0],
            prepared.trailing.shape()[0],
            codes.len(),
            codes.first()
        );
        eprint!("qwen3-tts clone generation diagnostic:\n{metrics}");
        std::fs::write("/tmp/qwen3_tts_clone_generation_diag.txt", &metrics).map_err(|source| {
            InferError::Io {
                path: PathBuf::from("/tmp/qwen3_tts_clone_generation_diag.txt"),
                source,
            }
        })?;
        assert!(
            codes.len() <= MAX_FRAMES.min(CLONE_GENERATION_HARD_CAP),
            "le clone a dépassé le cap de frames"
        );
        Ok(())
    }

    #[test]
    #[ignore = "live: charge le contrat header-only d'un snapshot Qwen3-TTS Base"]
    fn live_loads_base_snapshot_contract() -> Result<()> {
        let Some(model_dir) = local_tts_snapshot(
            "RETI_QWEN3_TTS_BASE_DIR",
            "models--mlx-community--Qwen3-TTS-12Hz-1.7B-Base-6bit",
        ) else {
            eprintln!("skip: snapshot Qwen3-TTS Base absent du cache HF");
            return Ok(());
        };
        let assets = TtsAssets::load_local(model_dir)?;
        assert_eq!(assets.model_kind(), TtsModelKind::Base);
        assert!(assets.clone_capable());
        assert!(assets.talker_catalog.has_speaker_encoder_weights);
        assert!(assets.codec_catalog.has_encoder_weights);
        Ok(())
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
        let home = std::env::var("HOME").ok()?;
        let snapshots = PathBuf::from(home)
            .join(".cache/huggingface/hub")
            .join(cache_name)
            .join("snapshots");
        let entries = std::fs::read_dir(snapshots).ok()?;
        for entry in entries.flatten() {
            let path = entry.path();
            if path.join("config.json").is_file()
                && path.join("speech_tokenizer/model.safetensors").is_file()
            {
                return Some(path);
            }
        }
        None
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
}
