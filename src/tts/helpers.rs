use super::{
    CLONE_DEFAULT_FRAMES_PER_TOKEN, CLONE_DEFAULT_MIN_FRAMES, CLONE_GENERATION_HARD_CAP,
    DEFAULT_REPEAT_FRAME_STOP,
};
use crate::{InferError, Result, Tensor};

pub(super) fn tts_generation_trace_enabled() -> bool {
    std::env::var("RETI_TTS_TRACE_FRAMES")
        .map(|value| {
            let value = value.trim();
            !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
        })
        .unwrap_or(false)
}

pub(super) fn tts_internal_profile_enabled() -> bool {
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
pub(super) fn tts_stream_first_lot() -> usize {
    std::env::var("RETI_TTS_STREAM_LOT")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(4)
}

/// Arrêt anti-boucle du talker TTS : une frame codec complète identique répétée
/// plusieurs fois indique une dérive de décodage greedy, pas une prosodie utile.
/// `0` désactive le garde-fou pour les diagnostics.
pub(super) fn tts_repeat_frame_stop() -> usize {
    tts_repeat_frame_stop_from_env(std::env::var("RETI_TTS_REPEAT_FRAME_STOP").ok().as_deref())
}

pub(super) fn tts_repeat_frame_stop_from_env(value: Option<&str>) -> usize {
    value
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(DEFAULT_REPEAT_FRAME_STOP)
}

pub(super) fn repeat_frame_stop_tripped(repeat_run: usize, threshold: usize) -> bool {
    threshold > 0 && repeat_run >= threshold
}

pub(super) fn gather_rows_i32(table: &Tensor, ids: &[i32]) -> Result<Tensor> {
    let mut rows = Vec::with_capacity(ids.len());
    for id in ids {
        let id = usize::try_from(*id)
            .map_err(|_| InferError::Dimension(format!("id embedding négatif: {id}")))?;
        rows.push(id);
    }
    crate::embed_tokens(table, &rows)
}

pub(super) fn push_rows(
    out: &mut Vec<f32>,
    tensor: &Tensor,
    start: usize,
    end: usize,
) -> Result<()> {
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

pub(super) fn add_into(left: &mut [f32], right: &[f32]) {
    for (left, right) in left.iter_mut().zip(right.iter()) {
        *left += *right;
    }
}

pub(super) fn usize_from_i32(value: i32, what: &str) -> Result<usize> {
    usize::try_from(value).map_err(|_| InferError::Config(format!("{what} négatif: {value}")))
}

pub(super) fn clone_effective_frame_cap(max_frames: usize, target_tokens: usize) -> usize {
    max_frames.min(CLONE_GENERATION_HARD_CAP).min(
        CLONE_DEFAULT_MIN_FRAMES.max(target_tokens.saturating_mul(CLONE_DEFAULT_FRAMES_PER_TOKEN)),
    )
}

pub(super) fn clone_sample_seed() -> u64 {
    std::env::var("RETI_TTS_CLONE_SAMPLE_SEED")
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or(0)
}
