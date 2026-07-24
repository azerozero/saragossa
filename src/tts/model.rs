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
}

pub(super) enum TtsStreamDecodeState {
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
