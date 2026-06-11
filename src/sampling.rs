//! Échantillonnage déterministe et filtrage des logits.

use crate::{InferError, Result};

#[derive(Clone, Debug)]
/// Génère un flux pseudo-aléatoire déterministe pour le sampling.
pub struct DeterministicSampler {
    state: u64,
}

impl DeterministicSampler {
    /// Crée un sampler déterministe depuis une graine.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_unit_f32(&mut self) -> f32 {
        let mantissa = (self.next_u64() >> 40) as u32;
        mantissa as f32 / (1_u32 << 24) as f32
    }

    /// Renvoie l'état interne courant.
    pub fn state(&self) -> u64 {
        self.state
    }

    /// Avance le flux d'un tirage.
    pub fn advance(&mut self) {
        let _ = self.next_u64();
    }
}

/// Renvoie l'indice du maximum, avec tie-break au premier maximum.
///
/// # Errors
///
/// Renvoie une erreur si la liste est vide.
pub fn argmax(xs: &[f32]) -> Result<usize> {
    if xs.is_empty() {
        return Err(InferError::Dimension("argmax sur logits vides".to_string()));
    }
    let mut best = 0_usize;
    let mut best_value = f32::NEG_INFINITY;
    for (idx, value) in xs.iter().copied().enumerate() {
        if value > best_value {
            best = idx;
            best_value = value;
        }
    }
    Ok(best)
}

/// Échantillonne un token avec température et nucleus sampling.
///
/// # Errors
///
/// Renvoie une erreur si les logits ou paramètres sont invalides.
pub fn sample_token(
    logits: &[f32],
    temperature: f32,
    top_p: f32,
    sampler: &mut DeterministicSampler,
) -> Result<usize> {
    if temperature <= f32::EPSILON {
        return argmax(logits);
    }
    if logits.is_empty() {
        return Err(InferError::Dimension(
            "sampling sur logits vides".to_string(),
        ));
    }
    if !temperature.is_finite() || temperature < 0.0 {
        return Err(InferError::Config(
            "temperature sampling invalide".to_string(),
        ));
    }
    if !top_p.is_finite() || top_p <= 0.0 || top_p > 1.0 {
        return Err(InferError::Config("top_p sampling invalide".to_string()));
    }

    let mut indexed_probs: Vec<(usize, f32)> = softmax(logits, temperature)
        .into_iter()
        .enumerate()
        .collect();
    indexed_probs.sort_by(|(_, left), (_, right)| right.total_cmp(left));

    let mut nucleus = Vec::new();
    let mut cumulative = 0.0_f32;
    for item in indexed_probs {
        cumulative += item.1;
        nucleus.push(item);
        if cumulative >= top_p {
            break;
        }
    }

    let sum = nucleus.iter().map(|(_, prob)| *prob).sum::<f32>();
    if sum <= f32::EPSILON {
        return argmax(logits);
    }

    let roll = sampler.next_unit_f32() * sum;
    let mut acc = 0.0_f32;
    for (token, prob) in nucleus {
        acc += prob;
        if roll <= acc {
            return Ok(token);
        }
    }
    argmax(logits)
}

/// Échantillonne avec température, top-k puis top-p.
///
/// # Errors
///
/// Renvoie une erreur si les logits ou paramètres sont invalides.
pub fn sample_token_top_k_top_p(
    logits: &[f32],
    temperature: f32,
    top_p: f32,
    top_k: usize,
    sampler: &mut DeterministicSampler,
) -> Result<usize> {
    if top_k == 0 || top_k >= logits.len() {
        return sample_token(logits, temperature, top_p, sampler);
    }
    if temperature <= f32::EPSILON {
        return argmax(logits);
    }
    if logits.is_empty() {
        return Err(InferError::Dimension(
            "sampling sur logits vides".to_string(),
        ));
    }
    if !temperature.is_finite() || temperature < 0.0 {
        return Err(InferError::Config(
            "temperature sampling invalide".to_string(),
        ));
    }
    if !top_p.is_finite() || top_p <= 0.0 || top_p > 1.0 {
        return Err(InferError::Config("top_p sampling invalide".to_string()));
    }

    let mut indexed_probs: Vec<(usize, f32)> = softmax(logits, temperature)
        .into_iter()
        .enumerate()
        .collect();
    indexed_probs.sort_by(|(left_idx, left), (right_idx, right)| {
        right.total_cmp(left).then_with(|| left_idx.cmp(right_idx))
    });
    indexed_probs.truncate(top_k);

    let top_sum = indexed_probs.iter().map(|(_, prob)| *prob).sum::<f32>();
    if top_sum <= f32::EPSILON {
        return argmax(logits);
    }

    let mut nucleus = Vec::new();
    let mut cumulative = 0.0_f32;
    for (rank, item) in indexed_probs.into_iter().enumerate() {
        cumulative += item.1 / top_sum;
        nucleus.push(item);
        if rank == 0 || cumulative < top_p {
            continue;
        }
        break;
    }

    let sum = nucleus.iter().map(|(_, prob)| *prob).sum::<f32>();
    if sum <= f32::EPSILON {
        return argmax(logits);
    }
    let roll = sampler.next_unit_f32() * sum;
    let mut acc = 0.0_f32;
    for (token, prob) in nucleus {
        acc += prob;
        if roll <= acc {
            return Ok(token);
        }
    }
    argmax(logits)
}

/// Calcule une distribution softmax stable.
pub fn softmax(logits: &[f32], temperature: f32) -> Vec<f32> {
    let temperature = temperature.max(0.000_1);
    let max = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |a, b| a.max(b));
    let mut sum = 0.0_f32;
    let mut out = Vec::with_capacity(logits.len());
    for logit in logits {
        let value = ((*logit - max) / temperature).exp();
        sum += value;
        out.push(value);
    }
    if sum <= f32::EPSILON {
        return vec![1.0 / logits.len() as f32; logits.len()];
    }
    for value in &mut out {
        *value /= sum;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_returns_first_max() {
        assert_eq!(
            argmax(&[1.0, 3.0, 3.0, 2.0]).expect("invariant: logits non vides"),
            1
        );
    }

    #[test]
    fn argmax_rejects_empty_logits() {
        let err = argmax(&[]).expect_err("invariant: logits vides rejetés");
        assert!(matches!(err, InferError::Dimension(_)));
    }

    #[test]
    fn sampling_is_seed_deterministic() {
        let logits = [0.0, 1.0, 2.0, 3.0];
        let mut left = DeterministicSampler::new(42);
        let mut right = DeterministicSampler::new(42);

        let left_tokens = (0..8)
            .map(|_| sample_token(&logits, 0.8, 1.0, &mut left))
            .collect::<Result<Vec<_>>>()
            .expect("invariant: sampling valide");
        let right_tokens = (0..8)
            .map(|_| sample_token(&logits, 0.8, 1.0, &mut right))
            .collect::<Result<Vec<_>>>()
            .expect("invariant: sampling valide");

        assert_eq!(left_tokens, right_tokens);
    }

    #[test]
    fn sampling_top_p_keeps_best_token_when_nucleus_is_tight() {
        let mut sampler = DeterministicSampler::new(7);
        let token = sample_token(&[0.0, 1.0, 8.0], 1.0, 0.1, &mut sampler)
            .expect("invariant: sampling valide");

        assert_eq!(token, 2);
    }

    #[test]
    fn softmax_sums_to_one() {
        let probs = softmax(&[0.0, 1.0, 2.0], 1.0);
        let sum = probs.iter().sum::<f32>();
        assert!((sum - 1.0).abs() < 1.0e-6);
        assert!(probs[2] > probs[1]);
    }

    #[test]
    fn sampling_top_k_limits_candidates() {
        let mut sampler = DeterministicSampler::new(7);
        let token = sample_token_top_k_top_p(&[8.0, 7.0, 6.0, 5.0], 1.0, 1.0, 2, &mut sampler)
            .expect("invariant: sampling valide");

        assert!(token <= 1, "token={token}");
    }

    #[test]
    fn sampler_advance_matches_one_draw() {
        let mut left = DeterministicSampler::new(42);
        let mut right = DeterministicSampler::new(42);

        left.advance();
        let _ = right.next_unit_f32();

        assert_eq!(left.state(), right.state());
    }
}
