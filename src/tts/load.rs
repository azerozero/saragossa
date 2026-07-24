use super::{
    copy_dense, insert_linear, load_qwen3_tts_tokenizer, read_linear_layer, SafetensorPayload,
    TtsAssets, TtsCloneContext, TtsCloneMode, TtsModel, TtsModelKind, TtsPayloadSummary,
};
use crate::decoder::DecoderTensor;
use crate::tts_codec::TtsCodec;
use crate::tts_mimi::TtsMimiEncoder;
use crate::tts_speaker::TtsSpeakerEncoder;
use crate::{CausalDecoder, CausalDecoderConfig, InferError, MemoryGuard, Result, Tensor};
use std::collections::HashMap;
use std::path::Path;

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
            is_qwen: false,
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
                is_qwen: false,
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

    /// Attache une garde mémoire aux décodeurs internes.
    #[must_use]
    pub fn with_memory_guard(mut self, guard: MemoryGuard) -> Self {
        self.talker = self.talker.with_memory_guard(guard.clone());
        self.code_predictor = self.code_predictor.with_memory_guard(guard);
        self
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

    pub(super) fn load_clone_local_with_mode(
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
}
