//! KL(référence ‖ candidat) et accord top-1 en teacher forcing, portage de
//! `tools/quant/kl_eval.py`.
//!
//! Même formule que le Python, `Σ p · (ln(p + 1e-9) − ln(q + 1e-9))` moyennée
//! sur les positions, mais calculée en f64 au lieu du f32 MLX. Sur logits
//! identiques, l'écart au Python se réduit donc à l'arrondi f32 de MLX ; les
//! logits viennent ici du moteur et non de mlx-lm, l'écart de bout en bout
//! reste à mesurer sur un vrai modèle.

use std::fs;
use std::path::Path;

use super::kl_ref::{normalize, top1};
use crate::{InferError, Result, Tensor};

/// Jeu figé de `kl_eval.py` : changer les prompts change les valeurs de KL.
pub const DEFAULT_PROMPTS: [&str; 5] = [
    "Explique en détail pourquoi le ciel est bleu et comment fonctionne la diffusion de Rayleigh.",
    "Résume les causes principales de la Révolution française de 1789.",
    "def fibonacci(n):\n    # renvoie le n-ieme terme\n    if n < 2:\n        return n\n    a, b = 0, 1\n",
    "Un train part de Paris à 14h00 à 120 km/h vers Lyon (450 km). À quelle heure arrive-t-il ?",
    "Écris une fonction Python qui vérifie si une chaîne est un palindrome, en ignorant la casse.",
];

const EPSILON: f64 = 1e-9;

fn invalid(message: impl Into<String>) -> InferError {
    InferError::Config(format!("kl-eval: {}", message.into()))
}

/// Un prompt par ligne non blanche, coupures de `str.splitlines()` ; jeu figé
/// sans fichier.
///
/// # Errors
///
/// Refuse un fichier illisible ou sans prompt exploitable.
pub fn load_prompts(path: Option<&Path>) -> Result<Vec<String>> {
    let Some(path) = path else {
        return Ok(DEFAULT_PROMPTS.map(str::to_string).to_vec());
    };
    let text = fs::read_to_string(path).map_err(|source| InferError::Io {
        path: path.into(),
        source,
    })?;
    let prompts: Vec<String> = text
        .split([
            '\n', '\r', '\x0b', '\x0c', '\x1c', '\x1d', '\x1e', '\u{85}', '\u{2028}', '\u{2029}',
        ])
        .filter(|line| !line.trim().is_empty())
        .map(str::to_string)
        .collect();
    if prompts.is_empty() {
        return Err(invalid(format!(
            "aucun prompt exploitable dans {}",
            path.display()
        )));
    }
    Ok(prompts)
}

fn probabilities(logits: &[f32]) -> Result<Vec<f64>> {
    Ok(normalize(logits)?.into_iter().map(f64::exp).collect())
}

/// KL(référence ‖ candidat) moyenne et accord top-1 sur les positions d'un
/// prompt, à partir des logits bruts `[T, V]` des deux modèles.
///
/// # Errors
///
/// Refuse des formes différentes ou vides et des logits non finis.
pub fn kl_and_agreement(reference: &Tensor, candidate: &Tensor) -> Result<(f64, f64)> {
    let (positions, _) = reference.as_matrix()?;
    if positions == 0 || reference.shape() != candidate.shape() {
        return Err(invalid("formes [T, V] différentes ou vides"));
    }
    let (mut divergence, mut agreement) = (0.0, 0.0);
    for row in 0..positions {
        let (reference, candidate) = (reference.row_slice(row)?, candidate.row_slice(row)?);
        let p = probabilities(reference)?;
        let q = probabilities(candidate)?;
        divergence += p
            .iter()
            .zip(&q)
            .map(|(&p, &q)| p * ((p + EPSILON).ln() - (q + EPSILON).ln()))
            .sum::<f64>();
        agreement += f64::from(top1(reference) == top1(candidate));
    }
    Ok((divergence / positions as f64, agreement / positions as f64))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix(rows: &[&[f32]]) -> Tensor {
        let data = rows.iter().flat_map(|row| row.iter().copied()).collect();
        Tensor::from_vec([rows.len(), rows[0].len()], data).expect("invariant: matrice")
    }

    /// Valeurs de `kl_and_agreement` de `kl_eval.py` réécrit en Python pur
    /// (f64, `math.exp`/`math.log`), sur deux positions et quatre classes.
    #[test]
    fn parity_with_kl_eval_formula() {
        let reference = matrix(&[&[1.0, 2.0, 0.5, -1.0], &[0.0, 0.0, 3.0, 1.0]]);
        let candidate = matrix(&[&[1.2, 1.8, 0.4, -0.5], &[0.1, 0.0, 2.5, 1.5]]);
        let (kl, agreement) =
            kl_and_agreement(&reference, &candidate).expect("invariant: métriques");
        assert!((kl - 0.043_577_677_686_705_035).abs() < 1e-15, "{kl}");
        assert_eq!(agreement, 1.0);
        let (same, agreement) =
            kl_and_agreement(&reference, &reference).expect("invariant: identité");
        assert_eq!((same, agreement), (0.0, 1.0));
        let swapped = matrix(&[&[2.0, 1.0, 0.5, -1.0], &[0.0, 0.0, 3.0, 1.0]]);
        let (_, agreement) = kl_and_agreement(&reference, &swapped).expect("invariant: accord");
        assert_eq!(agreement, 0.5);
    }

    #[test]
    fn rejects_mismatched_shapes_and_non_finite_logits() {
        let row = matrix(&[&[0.0, 1.0]]);
        assert!(kl_and_agreement(&row, &matrix(&[&[0.0]])).is_err());
        assert!(kl_and_agreement(&row, &matrix(&[&[0.0, 1.0], &[0.0, 1.0]])).is_err());
        assert!(kl_and_agreement(&row, &matrix(&[&[f32::NAN, 0.0]])).is_err());
    }

    #[test]
    fn prompts_follow_splitlines_and_skip_blank_lines() {
        assert_eq!(load_prompts(None).expect("invariant: jeu figé").len(), 5);
        let dir = tempfile::tempdir().expect("invariant: répertoire temporaire");
        let path = dir.path().join("prompts.txt");
        fs::write(&path, "un\r\n  \ndeux\u{2028} trois \n").expect("invariant: fixture");
        assert_eq!(
            load_prompts(Some(&path)).expect("invariant: prompts"),
            ["un", "deux", " trois "]
        );
        fs::write(&path, "\n \n").expect("invariant: fixture");
        assert!(load_prompts(Some(&path)).is_err());
    }
}
