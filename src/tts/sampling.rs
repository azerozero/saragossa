use super::TtsSampleParams;
use crate::sampling::{
    sample_from_token_distribution, sample_token_top_k_top_p, token_distribution_top_k_top_p,
    DeterministicSampler,
};
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
    let filtered = params.temperature > f32::EPSILON
        && ((params.top_k > 0 && params.top_k < adjusted.len()) || params.top_p < 1.0);
    let eos_idx = eos
        .and_then(|token| usize::try_from(token).ok())
        .filter(|idx| filtered && *idx < adjusted.len());
    let sampled = match eos_idx {
        // NOTE: la référence Qwen3-TTS (mlx_audio `_sample_token`) réinjecte le
        // logit EOS après top-k/top-p. Sans cela, un EOS hors du top-k n'est
        // jamais tiré et la génération court jusqu'au plafond de trames.
        Some(eos_idx) => {
            let mut distribution = token_distribution_top_k_top_p(
                &adjusted,
                params.temperature,
                params.top_p,
                params.top_k,
            )?;
            keep_eos_in_distribution(&mut distribution, &adjusted, eos_idx, params.temperature);
            sample_from_token_distribution(&distribution, sampler)?
        }
        None => sample_token_top_k_top_p(
            &adjusted,
            params.temperature,
            params.top_p,
            params.top_k,
            sampler,
        )?,
    };
    i32::try_from(sampled).map_err(|_| InferError::Config(format!("token TTS hors i32: {sampled}")))
}

/// Ajoute EOS à une distribution filtrée normalisée, avec le poids qu'il aurait
/// dans un softmax restreint au filtre plus EOS (même température).
fn keep_eos_in_distribution(
    distribution: &mut Vec<(usize, f32)>,
    logits: &[f32],
    eos: usize,
    temperature: f32,
) {
    if distribution.iter().any(|(token, _)| *token == eos) {
        return;
    }
    let max = distribution
        .iter()
        .map(|(token, _)| logits[*token])
        .fold(f32::NEG_INFINITY, f32::max);
    let kept = distribution
        .iter()
        .map(|(token, _)| ((logits[*token] - max) / temperature).exp())
        .sum::<f32>();
    let eos_weight = ((logits[eos] - max) / temperature).exp();
    if kept > 0.0 && eos_weight.is_finite() && eos_weight > 0.0 {
        // La distribution reçue somme à 1 : le poids EOS est relatif au filtre.
        distribution.push((eos, eos_weight / kept));
    }
}

fn apply_talker_repetition_penalty(
    logits: &mut [f32],
    history: &[i32],
    eos: Option<i32>,
    repetition_penalty: f32,
) {
    if !repetition_penalty.is_finite() || repetition_penalty <= 0.0 || repetition_penalty == 1.0 {
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

#[cfg(test)]
mod penalty_tests {
    use super::*;

    #[test]
    fn repetition_penalty_is_sign_aware_unique_and_preserves_eos() {
        let mut logits = [4.0, -4.0, 3.0, 2.0];
        apply_talker_repetition_penalty(&mut logits, &[0, 0, 1, 3], Some(3), 2.0);
        assert_eq!(logits, [2.0, -8.0, 3.0, 2.0]);
        apply_talker_repetition_penalty(&mut logits, &[0, 1], Some(3), 0.5);
        assert_eq!(logits, [4.0, -4.0, 3.0, 2.0]);
    }

    #[test]
    fn eos_outside_top_k_keeps_its_softmax_weight() -> Result<()> {
        let params = TtsSampleParams {
            temperature: 1.0,
            top_k: 1,
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed: 3,
        };
        // Filtre {0} plus EOS (2) : P(EOS) = e^-1 / (1 + e^-1) ≈ 0,27.
        let logits = [10.0, 9.5, 9.0];
        let mut distribution = token_distribution_top_k_top_p(&logits, 1.0, 1.0, 1)?;
        keep_eos_in_distribution(&mut distribution, &logits, 2, 1.0);
        assert_eq!(distribution.len(), 2);
        assert!((distribution[1].1 - (-1.0_f32).exp()).abs() < 1e-6);
        let mut sampler = DeterministicSampler::new(params.seed);
        let draws = (0..400)
            .map(|_| sample_talker_token(&logits, &params, &[], &[], Some(2), &mut sampler))
            .collect::<Result<Vec<_>>>()?;
        let eos = draws.iter().filter(|token| **token == 2).count();
        assert!(draws.iter().all(|token| *token == 0 || *token == 2));
        assert!((70..150).contains(&eos), "EOS tiré {eos} fois sur 400");
        // Sans EOS désigné, le top-k reste strict.
        let mut sampler = DeterministicSampler::new(params.seed);
        assert_eq!(
            sample_talker_token(&logits, &params, &[], &[], None, &mut sampler)?,
            0
        );
        Ok(())
    }

    #[test]
    fn seeded_sampling_repeats_and_penalty_precedes_top_k() -> Result<()> {
        let params = TtsSampleParams {
            temperature: 0.9,
            top_k: 1,
            top_p: 1.0,
            repetition_penalty: 2.0,
            seed: 11,
        };
        let mut sampler = DeterministicSampler::new(params.seed);
        assert_eq!(
            sample_talker_token(&[4.0, 3.0, 100.0], &params, &[2], &[0], None, &mut sampler)?,
            1
        );
        let params = TtsSampleParams {
            top_k: 0,
            repetition_penalty: 1.0,
            ..params
        };
        let draw = |seed| -> Result<Vec<i32>> {
            let mut sampler = DeterministicSampler::new(seed);
            (0..32)
                .map(|_| sample_talker_token(&[0.0; 8], &params, &[], &[], None, &mut sampler))
                .collect()
        };
        assert_eq!(draw(11)?, draw(11)?);
        assert_ne!(draw(11)?, draw(12)?);
        Ok(())
    }
}
