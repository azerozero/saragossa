//! Débruitage multi-cœur opt-in : segments avec recouvrement, recollés en fondu.
use std::thread;

use super::{DeepFilterNet3, SAMPLE_RATE};
use crate::{InferError, Result};

/// Chauffe de l'état récurrent avant la zone utile : 1 s, soit 100 trames.
const WARMUP: usize = SAMPLE_RATE as usize;
/// Durée du fondu enchaîné centré sur chaque frontière : 0,5 s.
const FADE: usize = WARMUP / 2;
/// Un segment plus court que ceci ne rembourse pas la chauffe ni le fondu.
const MIN_SEGMENT: usize = 4 * WARMUP;

impl DeepFilterNet3 {
    /// Comme [`Self::enhance`], sur `threads` cœurs.
    ///
    /// `threads <= 1`, ou un signal trop court pour être découpé, emprunte le
    /// chemin mono-cœur : la sortie est alors identique bit à bit. Au-delà,
    /// chaque segment repart d'un état vierge chauffé sur `WARMUP` (1 s) d'échantillons
    /// de contexte, donc la sortie est proche du mono-cœur sans lui être égale.
    ///
    /// # Errors
    /// Propage les erreurs du traitement des trames.
    pub fn enhance_parallel(&self, audio: &[f32], threads: usize) -> Result<Vec<f32>> {
        let n = audio.len();
        let parts = threads.min(n / MIN_SEGMENT);
        if parts <= 1 {
            return self.enhance(audio);
        }
        // Frontières utiles b_0 = 0 < b_1 < … < b_parts = n.
        let bounds: Vec<usize> = (0..=parts).map(|i| i * n / parts).collect();
        let ranges: Vec<(usize, usize)> = (0..parts)
            .map(|i| {
                (
                    bounds[i].saturating_sub(WARMUP),
                    (bounds[i + 1] + WARMUP).min(n),
                )
            })
            .collect();
        let results: Vec<Result<Vec<f32>>> = thread::scope(|scope| {
            let handles: Vec<_> = ranges
                .iter()
                .map(|&(lo, hi)| scope.spawn(move || self.enhance(&audio[lo..hi])))
                .collect();
            handles
                .into_iter()
                .map(|h| {
                    h.join().unwrap_or_else(|_| {
                        Err(InferError::Config("thread de débruitage mort".into()))
                    })
                })
                .collect()
        });
        // Poids complémentaires : la montée en b_i vaut 1 moins la descente en b_i.
        let ramp =
            |t: usize, b: usize| ((t + FADE / 2).saturating_sub(b) as f32 / FADE as f32).min(1.0);
        let mut out = vec![0.0_f32; n];
        for (i, result) in results.into_iter().enumerate() {
            let seg = result?;
            let (lo, hi) = ranges[i];
            let from = if i == 0 {
                0
            } else {
                bounds[i].saturating_sub(FADE / 2)
            };
            let to = if i + 1 == parts {
                n
            } else {
                (bounds[i + 1] + FADE / 2).min(n)
            };
            for t in from.max(lo)..to.min(hi) {
                let up = if i == 0 { 1.0 } else { ramp(t, bounds[i]) };
                let down = if i + 1 == parts {
                    1.0
                } else {
                    1.0 - ramp(t, bounds[i + 1])
                };
                out[t] += seg[t - lo] * up * down;
            }
        }
        Ok(out)
    }
}
