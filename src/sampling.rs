//! Échantillonnage déterministe et filtrage des logits.

use crate::{InferError, Result};

/// Distribution discrète filtrée sous forme `(token, probabilité normalisée)`.
pub type TokenDistribution = Vec<(usize, f32)>;

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

    let distribution = token_distribution_top_k_top_p(logits, temperature, top_p, 0)?;
    sample_from_token_distribution(&distribution, sampler)
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

    let distribution = token_distribution_top_k_top_p(logits, temperature, top_p, top_k)?;
    sample_from_token_distribution(&distribution, sampler)
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

/// Matérialise la distribution top-k/top-p utilisée par le sampler.
///
/// # Errors
///
/// Renvoie une erreur si les logits ou paramètres sont invalides.
pub fn token_distribution_top_k_top_p(
    logits: &[f32],
    temperature: f32,
    top_p: f32,
    top_k: usize,
) -> Result<TokenDistribution> {
    if logits.is_empty() {
        return Err(InferError::Dimension(
            "distribution sur logits vides".to_string(),
        ));
    }
    if temperature <= f32::EPSILON {
        return Ok(vec![(argmax(logits)?, 1.0)]);
    }
    if !temperature.is_finite() || temperature < 0.0 {
        return Err(InferError::Config(
            "temperature sampling invalide".to_string(),
        ));
    }
    if !top_p.is_finite() || top_p <= 0.0 || top_p > 1.0 {
        return Err(InferError::Config("top_p sampling invalide".to_string()));
    }

    let mut indexed_probs: TokenDistribution = softmax(logits, temperature)
        .into_iter()
        .enumerate()
        .collect();

    if top_k == 0 || top_k >= logits.len() {
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
        return normalize_distribution(nucleus, logits);
    }

    indexed_probs.sort_by(|(left_idx, left), (right_idx, right)| {
        right.total_cmp(left).then_with(|| left_idx.cmp(right_idx))
    });
    indexed_probs.truncate(top_k);

    let top_sum = indexed_probs.iter().map(|(_, prob)| *prob).sum::<f32>();
    if top_sum <= f32::EPSILON {
        return Ok(vec![(argmax(logits)?, 1.0)]);
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
    normalize_distribution(nucleus, logits)
}

/// Échantillonne une distribution déjà filtrée.
///
/// # Errors
///
/// Renvoie une erreur si la distribution est vide ou dégénérée.
pub fn sample_from_token_distribution(
    distribution: &[(usize, f32)],
    sampler: &mut DeterministicSampler,
) -> Result<usize> {
    if distribution.is_empty() {
        return Err(InferError::Dimension(
            "sampling sur distribution vide".to_string(),
        ));
    }
    let sum = distribution.iter().map(|(_, prob)| *prob).sum::<f32>();
    if sum <= f32::EPSILON {
        return Err(InferError::Config(
            "distribution sampling dégénérée".to_string(),
        ));
    }
    let roll = sampler.next_unit_f32() * sum;
    let mut acc = 0.0_f32;
    for (token, prob) in distribution {
        acc += *prob;
        if roll <= acc {
            return Ok(*token);
        }
    }
    distribution
        .last()
        .map(|(token, _)| *token)
        .ok_or_else(|| InferError::Dimension("distribution vide".to_string()))
}

/// Résultat d'un pas de sampling spéculatif exact.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LosslessSpeculativeSample {
    /// Indique si le token draft est accepté.
    pub accepted: bool,
    /// Token à émettre ou rejouer côté cible.
    pub token: usize,
    /// Probabilité d'acceptation `min(1, p(t)/q(t))`.
    pub accept_probability: f32,
}

/// Applique l'accept/reject lossless de speculative sampling.
///
/// # Errors
///
/// Renvoie une erreur si les distributions sont invalides.
pub fn lossless_speculative_sample(
    target: &[(usize, f32)],
    draft: &[(usize, f32)],
    draft_token: usize,
    sampler: &mut DeterministicSampler,
) -> Result<LosslessSpeculativeSample> {
    validate_distribution(target, "target")?;
    validate_distribution(draft, "draft")?;
    let p = distribution_probability(target, draft_token);
    let q = distribution_probability(draft, draft_token);
    let accept_probability = if q <= f32::EPSILON {
        if p > 0.0 {
            1.0
        } else {
            0.0
        }
    } else {
        (p / q).min(1.0)
    };
    let accepted = sampler.next_unit_f32() <= accept_probability;
    if accepted {
        return Ok(LosslessSpeculativeSample {
            accepted: true,
            token: draft_token,
            accept_probability,
        });
    }

    let residual = residual_distribution(target, draft)?;
    let token = sample_from_token_distribution(&residual, sampler)?;
    Ok(LosslessSpeculativeSample {
        accepted: false,
        token,
        accept_probability,
    })
}

fn normalize_distribution(
    mut distribution: TokenDistribution,
    fallback_logits: &[f32],
) -> Result<TokenDistribution> {
    let sum = distribution.iter().map(|(_, prob)| *prob).sum::<f32>();
    if sum <= f32::EPSILON {
        return Ok(vec![(argmax(fallback_logits)?, 1.0)]);
    }
    for (_, prob) in &mut distribution {
        *prob /= sum;
    }
    Ok(distribution)
}

fn validate_distribution(distribution: &[(usize, f32)], name: &str) -> Result<()> {
    if distribution.is_empty() {
        return Err(InferError::Dimension(format!("distribution {name} vide")));
    }
    let mut sum = 0.0_f32;
    for (_, prob) in distribution {
        if !prob.is_finite() || *prob < 0.0 {
            return Err(InferError::Config(format!(
                "distribution {name} contient une probabilité invalide"
            )));
        }
        sum += *prob;
    }
    if sum <= f32::EPSILON {
        return Err(InferError::Config(format!("distribution {name} dégénérée")));
    }
    Ok(())
}

fn distribution_probability(distribution: &[(usize, f32)], token: usize) -> f32 {
    distribution
        .iter()
        .find_map(|(candidate, prob)| (*candidate == token).then_some(*prob))
        .unwrap_or(0.0)
}

fn residual_distribution(
    target: &[(usize, f32)],
    draft: &[(usize, f32)],
) -> Result<TokenDistribution> {
    let mut residual = Vec::new();
    for (token, p) in target {
        let q = distribution_probability(draft, *token);
        let prob = (*p - q).max(0.0);
        if prob > f32::EPSILON {
            residual.push((*token, prob));
        }
    }
    let sum = residual.iter().map(|(_, prob)| *prob).sum::<f32>();
    if sum <= f32::EPSILON {
        return Ok(target.to_vec());
    }
    for (_, prob) in &mut residual {
        *prob /= sum;
    }
    Ok(residual)
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

    #[test]
    fn lossless_speculative_sampling_matches_target_distribution() {
        let target = vec![(0, 0.10), (1, 0.35), (2, 0.55)];
        let draft = vec![(0, 0.60), (1, 0.20), (2, 0.20)];
        let samples = 20_000usize;
        let mut counts = [0usize; 3];

        for seed in 0..samples {
            let mut sampler = DeterministicSampler::new(seed as u64);
            let draft_token = sample_from_token_distribution(&draft, &mut sampler)
                .expect("invariant: distribution draft valide");
            let sample = lossless_speculative_sample(&target, &draft, draft_token, &mut sampler)
                .expect("invariant: distributions valides");
            counts[sample.token] += 1;
        }

        for (token, expected) in target {
            let observed = counts[token] as f32 / samples as f32;
            assert!(
                (observed - expected).abs() < 0.015,
                "token={token} observed={observed} expected={expected}"
            );
        }
    }

    #[test]
    fn lossless_speculative_sampling_rejects_from_residual() {
        let target = vec![(0, 0.0), (1, 0.25), (2, 0.75)];
        let draft = vec![(0, 1.0)];
        let mut sampler = DeterministicSampler::new(1);

        let sample = lossless_speculative_sample(&target, &draft, 0, &mut sampler)
            .expect("invariant: distributions valides");

        assert!(!sample.accepted);
        assert_ne!(sample.token, 0);
    }
}
