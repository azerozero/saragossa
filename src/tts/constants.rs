pub const DEFAULT_INSTRUCT: &str = "Voix féminine française Réti-01 : claire, posée et professionnelle, articulation nette, ton neutre et rassurant, débit efficace.";
pub(super) const CLONE_GENERATION_HARD_CAP: usize = 800;
pub(super) const CLONE_DEFAULT_MIN_FRAMES: usize = 75;
pub(super) const CLONE_DEFAULT_FRAMES_PER_TOKEN: usize = 6;
pub(super) const CLONE_SAMPLE_TEMPERATURE: f32 = 0.9;
pub(super) const CLONE_SAMPLE_TOP_K: usize = 50;
pub(super) const CLONE_SAMPLE_TOP_P: f32 = 1.0;
pub(super) const CLONE_SAMPLE_REPETITION_PENALTY: f32 = 1.05;
pub(super) const DEFAULT_REPEAT_FRAME_STOP: usize = 8;

pub(super) const QWEN3_TTS_SPECIAL_TOKENS: &[(&str, bool)] = &[
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
