use super::TtsSampleParams;
use crate::sampling::{sample_token_top_k_top_p, DeterministicSampler};
use crate::{InferError, Result};

pub(super) fn sample_talker_token(
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

pub(super) fn greedy_token(logits: &[f32], suppress: &[i32]) -> Result<i32> {
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

pub(super) fn greedy_talker_token(logits: &[f32], suppress: &[i32]) -> Result<i32> {
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
