//! Décodage horodaté Whisper : règles de timestamps et segments `verbose_json`.
//!
//! Les règles reprennent `ApplyTimestampRules` de l'implémentation OpenAI :
//! un timestamp ouvre le texte (au plus 1 s), les timestamps vont par paires
//! croissantes, et la masse des timestamps l'emporte sur le meilleur token texte.
//! Les tokens spéciaux entre EOT et `<|0.00|>` restent supprimés.

use serde::Serialize;

/// Tokens `<|t.tt|>` par seconde (pas de 20 ms) ; diviser évite `120 × 0,02 ≠ 2,4`.
const TIMESTAMPS_PER_SECOND: f32 = 50.0;
/// Premier timestamp autorisé au plus à 1 s (`max_initial_timestamp`).
const MAX_INITIAL_TIMESTAMP_INDEX: u32 = 50;

/// Segment horodaté au format `verbose_json` d'OpenAI.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct WhisperSegment {
    pub id: usize,
    /// Début de la fenêtre de 30 s, en trames mel de 10 ms.
    pub seek: usize,
    pub start: f32,
    pub end: f32,
    pub text: String,
    pub tokens: Vec<u32>,
    pub temperature: f32,
    /// Moyenne des logprobs de la fenêtre, EOT compris, comme OpenAI.
    pub avg_logprob: f32,
    /// Octets UTF-8 du texte de la fenêtre sur sa taille zlib : une valeur
    /// haute signale une boucle de répétition.
    pub compression_ratio: f32,
    /// Probabilité de `<|nospeech|>` après `<|startoftranscript|>`.
    pub no_speech_prob: f32,
}

/// Transcription horodatée complète.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct WhisperTranscript {
    pub text: String,
    pub language: String,
    pub duration: f32,
    pub segments: Vec<WhisperSegment>,
}

/// Contraintes de timestamps et accumulateurs d'une fenêtre.
pub(super) struct TimestampRules {
    eot: u32,
    ts_begin: u32,
    nospeech: Option<u32>,
    languages: std::ops::Range<u32>,
    pub(super) sum_logprob: f64,
    pub(super) no_speech_prob: f32,
    pub(super) language: Option<u32>,
}

impl TimestampRules {
    pub(super) fn new(
        eot: u32,
        ts_begin: u32,
        nospeech: Option<u32>,
        languages: std::ops::Range<u32>,
    ) -> Self {
        Self {
            eot,
            ts_begin,
            nospeech,
            languages,
            sum_logprob: 0.0,
            no_speech_prob: 0.0,
            language: None,
        }
    }

    /// Lit les logits après `<|startoftranscript|>` : langue et absence de parole.
    pub(super) fn observe_sot(&mut self, logits: &[f32]) {
        let probs = log_softmax(logits);
        if let Some(prob) = self.nospeech.and_then(|id| probs.get(id as usize)) {
            self.no_speech_prob = prob.exp();
        }
        self.language = self
            .languages
            .clone()
            .filter(|&id| (id as usize) < probs.len())
            .max_by(|&a, &b| probs[a as usize].total_cmp(&probs[b as usize]));
    }

    /// Applique les règles, choisit le token greedy et cumule son logprob.
    pub(super) fn select(&mut self, mut logits: Vec<f32>, generated: &[u32]) -> u32 {
        let vocab = logits.len();
        let ts = (self.ts_begin as usize).min(vocab);
        let eot = (self.eot as usize).min(ts);
        logits[(eot + 1).min(ts)..ts].fill(f32::NEG_INFINITY);
        let is_ts = |token: &u32| *token >= self.ts_begin;
        let last = generated.last().is_some_and(is_ts);
        let penultimate = generated.len() < 2 || is_ts(&generated[generated.len() - 2]);
        if last {
            if penultimate {
                logits[ts..].fill(f32::NEG_INFINITY);
            } else {
                logits[..eot].fill(f32::NEG_INFINITY);
            }
        }
        if let Some(&previous) = generated.iter().rev().find(|token| is_ts(token)) {
            // Un timestamp fermant peut se répéter pour ouvrir le segment suivant.
            let floor = if last && !penultimate {
                previous
            } else {
                previous + 1
            };
            logits[ts..(floor as usize).min(vocab)].fill(f32::NEG_INFINITY);
        }
        if generated.is_empty() {
            logits[..ts].fill(f32::NEG_INFINITY);
            let limit = ts + MAX_INITIAL_TIMESTAMP_INDEX as usize + 1;
            logits[limit.min(vocab)..].fill(f32::NEG_INFINITY);
        }
        let logprobs = log_softmax(&logits);
        let timestamp_mass = log_sum_exp(&logprobs[ts..]);
        let best_text = logprobs[..ts]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        if timestamp_mass > best_text {
            logits[..ts].fill(f32::NEG_INFINITY);
        }
        let next = argmax_index(&logits);
        self.sum_logprob += f64::from(logprobs[next]);
        // invariant: `next < vocab` et le vocab Whisper tient en u32.
        next as u32
    }
}

/// Découpe les tokens d'une fenêtre en `(début, fin, tokens texte)`.
///
/// Un texte sans timestamp ouvrant démarre à la dernière fin connue ; un texte
/// non fermé se termine à la fin de la fenêtre. Les temps sont bornés à la
/// durée réelle, Whisper voyant un audio complété à 30 s.
pub(super) fn parse_segments(
    tokens: &[u32],
    eot: u32,
    ts_begin: u32,
    window_seconds: f32,
) -> Vec<(f32, f32, Vec<u32>)> {
    let time = |token: u32| ((token - ts_begin) as f32 / TIMESTAMPS_PER_SECOND).min(window_seconds);
    let mut segments = Vec::new();
    let mut start = None;
    let mut last_end = 0.0;
    let mut text = Vec::new();
    for &token in tokens {
        if token >= ts_begin {
            match start {
                Some(open) if !text.is_empty() => {
                    let end = time(token).max(open);
                    segments.push((open, end, std::mem::take(&mut text)));
                    last_end = end;
                    start = None;
                }
                _ => start = Some(time(token)),
            }
        } else if token < eot {
            if start.is_none() {
                start = Some(last_end);
            }
            text.push(token);
        }
    }
    if let (Some(open), false) = (start, text.is_empty()) {
        segments.push((open, window_seconds.max(open), text));
    }
    segments
}

/// Ratio de compression zlib d'OpenAI : `len(texte) / len(zlib(texte))`.
pub(super) fn compression_ratio(text: &str) -> f32 {
    use std::io::Write;
    let bytes = text.as_bytes();
    if bytes.is_empty() {
        return 0.0;
    }
    let mut encoder = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
    // Écrire dans un `Vec` n'échoue pas ; sinon le ratio neutre vaut 1.
    let compressed = encoder
        .write_all(bytes)
        .and_then(|()| encoder.finish())
        .map_or(bytes.len(), |out| out.len());
    bytes.len() as f32 / compressed.max(1) as f32
}

fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let norm = log_sum_exp(logits);
    logits.iter().map(|&value| value - norm).collect()
}

fn log_sum_exp(values: &[f32]) -> f32 {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if max == f32::NEG_INFINITY {
        return max;
    }
    max + values.iter().map(|&v| (v - max).exp()).sum::<f32>().ln()
}

fn argmax_index(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |best, (index, &value)| {
            if value > best.1 {
                (index, value)
            } else {
                best
            }
        })
        .0
}

#[cfg(test)]
mod tests {
    use super::*;

    const EOT: u32 = 10;
    const TS: u32 = 20;
    const VOCAB: usize = 120;

    fn rules() -> TimestampRules {
        TimestampRules::new(EOT, TS, Some(12), 13..16)
    }

    fn logits_favoring(token: usize) -> Vec<f32> {
        let mut logits = vec![0.0; VOCAB];
        logits[token] = 10.0;
        logits
    }

    #[test]
    fn compression_ratio_flags_repetition() {
        assert_eq!(compression_ratio(""), 0.0);
        let plain = compression_ratio("Le romantisme présentait un élément culturel.");
        let looped = compression_ratio(&"merci beaucoup ".repeat(40));
        assert!(plain < 1.5, "{plain}");
        assert!(looped > 2.4, "{looped}");
    }

    #[test]
    fn segments_follow_timestamp_pairs() {
        let tokens = [TS, 1, 2, TS + 120, TS + 120, 3, TS + 250];
        assert_eq!(
            parse_segments(&tokens, EOT, TS, 30.0),
            vec![(0.0, 2.4, vec![1, 2]), (2.4, 5.0, vec![3])]
        );
    }

    #[test]
    fn single_closing_timestamp_and_dangling_text_keep_every_token() {
        let tokens = [TS + 10, 1, TS + 50, 2, TS + 90, 3];
        assert_eq!(
            parse_segments(&tokens, EOT, TS, 4.0),
            vec![
                (0.2, 1.0, vec![1]),
                (1.0, 1.8, vec![2]),
                (1.8, 4.0, vec![3])
            ]
        );
        // Sans timestamp, tout le texte couvre la fenêtre ; les spéciaux sont omis.
        assert_eq!(
            parse_segments(&[1, EOT + 1, 2], EOT, TS, 3.0),
            vec![(0.0, 3.0, vec![1, 2])]
        );
        // Un timestamp au-delà de l'audio réel est borné.
        assert_eq!(
            parse_segments(&[TS, 1, TS + 1000], EOT, TS, 3.0),
            vec![(0.0, 3.0, vec![1])]
        );
    }

    #[test]
    fn first_token_is_an_early_timestamp() {
        let mut rules = rules();
        assert_eq!(rules.select(logits_favoring(1), &[]), TS);
        assert_eq!(
            rules.select(logits_favoring(TS as usize + 30), &[]),
            TS + 30
        );
        // Au-delà d'une seconde, le premier timestamp est interdit.
        assert_eq!(rules.select(logits_favoring(TS as usize + 80), &[]), TS);
    }

    #[test]
    fn timestamps_pair_up_and_never_go_back() {
        let mut rules = rules();
        // Après texte, un timestamp plus ancien que le dernier est interdit.
        let mut logits = logits_favoring(1);
        logits[TS as usize + 3] = 20.0;
        assert_eq!(rules.select(logits, &[TS + 5, 1]), 1);
        // Timestamp seul (fermant) : texte interdit, répétition permise.
        let next = rules.select(logits_favoring(2), &[TS, 1, TS + 7]);
        assert!(next == EOT || next >= TS + 7);
        // Paire complète : le token suivant est du texte.
        let next = rules.select(logits_favoring(TS as usize + 30), &[TS, 1, TS + 7, TS + 7]);
        assert!(next < TS);
        // Les spéciaux (langue, nospeech) ne sortent jamais.
        let mut logits = logits_favoring(14);
        logits[1] = 8.0;
        assert_eq!(rules.select(logits, &[TS, 1]), 1);
    }

    #[test]
    fn timestamp_mass_beats_best_text_token() {
        let mut rules = rules();
        let mut logits = vec![0.0; VOCAB];
        logits[1] = 3.0;
        logits[TS as usize + 1..].fill(2.5);
        assert!(rules.select(logits, &[TS, 1]) > TS);
        assert!(rules.sum_logprob < 0.0);
    }

    #[test]
    fn sot_logits_give_language_and_no_speech_probability() {
        let mut rules = rules();
        let mut logits = vec![0.0; VOCAB];
        logits[12] = 50.0;
        logits[14] = 5.0;
        rules.observe_sot(&logits);
        assert!(rules.no_speech_prob > 0.99);
        assert_eq!(rules.language, Some(14));
    }
}
