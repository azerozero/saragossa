use super::TtsAssets;
use crate::tts_codec::{TtsCodec, TtsCodecStreamState};
use crate::{CausalDecoder, CausalDecoderCache, Tensor};
use tokenizers::Tokenizer;

#[derive(Debug)]
pub struct TtsModel {
    pub assets: TtsAssets,
    pub(super) tokenizer: Tokenizer,
    pub(super) text_embedding: Tensor,
    pub(super) codec_embedding: Tensor,
    pub(super) text_projection_fc1: crate::Linear,
    pub(super) text_projection_fc2: crate::Linear,
    pub(super) talker: CausalDecoder,
    pub(super) code_predictor_projection: crate::Linear,
    pub(super) code_predictor: CausalDecoder,
    pub(super) code_predictor_heads: Vec<crate::Linear>,
    pub(super) code_predictor_embeddings: Vec<Tensor>,
    pub(super) codec: TtsCodec,
    pub(super) codec_payload: TtsPayloadSummary,
    pub(super) clone_ctx: Option<TtsCloneContext>,
    pub(super) voice_design_instruction: String,
    pub(super) speaker_id: Option<i32>,
    pub(super) generation_config: Option<super::config::TtsGenerationConfig>,
    pub(super) seed: Option<u64>,
    // Le cache appartient aux poids chargés : aucun état ne survit au modèle.
    pub(super) prefill_memo: std::sync::Mutex<super::prefill::TtsPrefillMemo>,
}

pub(super) enum TtsStreamDecodeState {
    Incremental(TtsCodecStreamState),
    /// `exact` : préfixe codec CPU (glouton).
    FullPrefixDelta {
        emitted: usize,
        exact: bool,
    },
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
    /// Le budget effectif a été consommé sans observer EOS. Le PCM reste utile,
    /// mais peut être incomplet. `false` n'exclut pas l'arrêt anti-répétition.
    pub frame_limit_reached: bool,
    /// Cause effective de la fin de génération.
    pub stop_reason: TtsStopReason,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct TtsSampleParams {
    pub(super) temperature: f32,
    pub(super) top_k: usize,
    pub(super) top_p: f32,
    pub(super) repetition_penalty: f32,
    pub(super) seed: u64,
}

#[derive(Debug)]
pub(super) struct PreparedVoiceDesign {
    pub(super) input: Tensor,
    pub(super) trailing: Tensor,
    pub(super) tts_pad: Tensor,
}

#[derive(Debug)]
pub(super) struct TtsCloneContext {
    pub(super) ref_codes: Vec<Vec<i32>>,
    pub(super) speaker_embed: Tensor,
    pub(super) ref_text_ids: Vec<i32>,
    pub(super) ref_codec_embed: Option<Tensor>,
    pub(super) mode: TtsCloneMode,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TtsCloneMode {
    Icl,
    XVectorOnly,
}

impl TtsCloneMode {
    pub(super) fn is_xvec_only(self) -> bool {
        matches!(self, Self::XVectorOnly)
    }

    pub(super) fn label(self) -> &'static str {
        match self {
            Self::Icl => "clone-icl",
            Self::XVectorOnly => "clone-xvec-only",
        }
    }
}

/// Cause effective de la fin de génération TTS.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TtsStopReason {
    Eos,
    Repeat,
    Cap,
}

impl TtsStopReason {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Eos => "eos",
            Self::Repeat => "repeat",
            Self::Cap => "cap",
        }
    }
}

pub(super) struct GeneratedCodes {
    pub codes: Vec<Vec<i32>>,
    pub stop_reason: TtsStopReason,
}
