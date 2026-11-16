//! Transcription Whisper en flux : deltas de texte et entrée audio incrémentale.
//!
//! [`WhisperStream`] reçoit le PCM par morceaux. L'audio courant forme une
//! fenêtre d'au plus 30 s : chaque morceau relance l'encodeur résident sur la
//! fenêtre (Whisper n'a pas d'encodeur causal), et la stabilisation suit
//! LocalAgreement-2 (whisper_streaming) : les mots communs aux deux dernières
//! hypothèses deviennent `committed`, le reste est `partial`.
//!
//! La fenêtre se ferme dès que la coupe batch est connue, c'est-à-dire quand
//! 30 s + 10 ms d'audio sont présents : [`super::segments::audio_segments`] ne
//! lit rien au-delà. Le texte final est donc celui du batch sur le même PCM, et
//! la latence finale ne dépend que de la dernière fenêtre ouverte.
//!
//! Une session est bornée en mémoire et en calcul (voir [`StreamLimit`]) :
//! morceaux d'au plus 30 s, 30 min d'audio, un nombre fini de morceaux et
//! d'inférences. Un morceau qui apporte moins de 250 ms depuis la dernière
//! hypothèse ne relance pas l'inférence.

use super::{segments::audio_segments, HOP, N_SAMPLES, SAMPLE_RATE};
use crate::{InferError, Result};

/// Plus grand morceau accepté : une fenêtre Whisper.
pub const MAX_CHUNK_SAMPLES: usize = N_SAMPLES;
/// Audio total par session : 30 min, soit au plus 62 fenêtres closes.
pub const MAX_SESSION_SAMPLES: usize = 30 * 60 * SAMPLE_RATE;
/// Audio nouveau minimal avant une nouvelle hypothèse partielle.
pub const MIN_PARTIAL_SAMPLES: usize = SAMPLE_RATE / 4;
/// Morceaux par session : un toutes les 100 ms sur la durée maximale.
pub const MAX_SESSION_CHUNKS: usize = MAX_SESSION_SAMPLES / (SAMPLE_RATE / 10);
/// Inférences par session : partiels espacés de 250 ms, fermetures et final.
pub const MAX_SESSION_INFERENCES: usize = MAX_SESSION_SAMPLES / MIN_PARTIAL_SAMPLES + 128;
/// Capacité gardée par la fenêtre après une fermeture (deux fenêtres).
const RETAINED_WINDOW_SAMPLES: usize = 2 * N_SAMPLES;

/// Admission refusée avant tout calcul.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamLimit {
    ChunkTooLarge,
    SessionTooLong,
    TooManyChunks,
    ComputeBudget,
}

impl StreamLimit {
    #[must_use]
    pub fn message(self) -> &'static str {
        match self {
            Self::ChunkTooLarge => "morceau audio > 30 s",
            Self::SessionTooLong => "session STT > 30 min d'audio",
            Self::TooManyChunks => "quota de morceaux de la session épuisé",
            Self::ComputeBudget => "budget d'inférence de la session épuisé",
        }
    }
}

/// Transforme les décodages successifs d'un préfixe en deltas append-only.
///
/// Les blancs finaux et un UTF-8 incomplet (`U+FFFD`) attendent le token
/// suivant, pour que la concaténation des deltas égale le texte final rogné.
pub(super) struct TextDeltas {
    emitted: String,
    separator: bool,
}

impl TextDeltas {
    /// `separator` préfixe le premier delta d'un espace (fenêtre non initiale).
    pub(super) fn new(separator: bool) -> Self {
        Self {
            emitted: String::new(),
            separator,
        }
    }

    pub(super) fn update(&mut self, decoded: &str) -> Option<String> {
        let visible = decoded.trim();
        if visible.ends_with('\u{FFFD}')
            || visible.len() <= self.emitted.len()
            || !visible.starts_with(self.emitted.as_str())
        {
            return None;
        }
        let mut delta = visible[self.emitted.len()..].to_string();
        if self.emitted.is_empty() && self.separator {
            delta.insert(0, ' ');
        }
        self.emitted = visible.to_string();
        Some(delta)
    }
}

/// État visible après un morceau audio.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WhisperStreamUpdate {
    /// Texte stabilisé : fenêtres closes, puis accord LocalAgreement-2.
    pub committed: String,
    /// Suite de la dernière hypothèse, encore susceptible de changer.
    pub partial: String,
}

/// Session STT à entrée incrémentale (PCM mono 16 kHz).
#[derive(Debug, Default)]
pub struct WhisperStream {
    window: Vec<f32>,
    closed: Vec<String>,
    committed: Vec<String>,
    previous: Vec<String>,
    /// Hypothèse calculée sur exactement `window.len()` échantillons.
    last: Option<(usize, String)>,
    /// Dernier état rendu ; son `partial` resert quand l'inférence est sautée.
    last_update: WhisperStreamUpdate,
    received: usize,
    chunks: usize,
    inferences: usize,
}

impl WhisperStream {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Échantillons reçus depuis l'ouverture de la session.
    #[must_use]
    pub fn received_samples(&self) -> usize {
        self.received
    }

    /// Inférences consommées depuis l'ouverture.
    #[must_use]
    pub fn inferences(&self) -> usize {
        self.inferences
    }

    /// Vérifie qu'un morceau de `samples` échantillons peut être accepté.
    ///
    /// # Errors
    ///
    /// Renvoie la limite atteinte ; la session reste inchangée.
    pub fn admit(&self, samples: usize) -> std::result::Result<(), StreamLimit> {
        if samples > MAX_CHUNK_SAMPLES {
            return Err(StreamLimit::ChunkTooLarge);
        }
        if self.received + samples > MAX_SESSION_SAMPLES {
            return Err(StreamLimit::SessionTooLong);
        }
        if self.chunks >= MAX_SESSION_CHUNKS {
            return Err(StreamLimit::TooManyChunks);
        }
        // Pire cas d'un morceau : deux fermetures et une hypothèse ou le final.
        if self.inferences + 3 > MAX_SESSION_INFERENCES {
            return Err(StreamLimit::ComputeBudget);
        }
        Ok(())
    }

    /// Ajoute un morceau, ferme les fenêtres complètes et met à jour l'hypothèse.
    ///
    /// Moins de 250 ms d'audio nouveau (morceau vide compris) renvoie le
    /// dernier état sans inférence.
    ///
    /// `transcribe` est la transcription batch d'au plus 30 s.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le morceau dépasse une limite de session
    /// ([`Self::admit`]), ou propage l'échec de `transcribe`.
    pub fn push(
        &mut self,
        chunk: &[f32],
        transcribe: &mut dyn FnMut(&[f32]) -> Result<String>,
    ) -> Result<WhisperStreamUpdate> {
        self.append(chunk, transcribe)?;
        let fresh = match &self.last {
            Some((len, _)) => self.window.len().saturating_sub(*len),
            None => self.window.len(),
        };
        if fresh < MIN_PARTIAL_SAMPLES {
            // Une fermeture a pu compléter `committed` sans nouvelle hypothèse.
            let partial = match self.last {
                Some(_) => std::mem::take(&mut self.last_update.partial),
                None => String::new(),
            };
            return Ok(self.snapshot(partial));
        }
        self.inferences += 1;
        let hypothesis = transcribe(&self.window[..self.window.len().min(N_SAMPLES)])?;
        let words: Vec<String> = hypothesis.split_whitespace().map(str::to_owned).collect();
        let agreed = self
            .previous
            .iter()
            .zip(&words)
            .take_while(|(a, b)| a == b)
            .count();
        if agreed > self.committed.len() && words.starts_with(&self.committed) {
            self.committed = words[..agreed].to_vec();
        }
        let partial = words[self.committed.len().min(words.len())..].join(" ");
        self.previous = words;
        self.last = Some((self.window.len(), hypothesis));
        Ok(self.snapshot(partial))
    }

    fn snapshot(&mut self, partial: String) -> WhisperStreamUpdate {
        let committed = self
            .closed
            .iter()
            .map(String::as_str)
            .chain(self.committed.iter().map(String::as_str))
            .collect::<Vec<_>>()
            .join(" ");
        self.last_update = WhisperStreamUpdate { committed, partial };
        self.last_update.clone()
    }

    /// Ajoute le PCM et transcrit chaque fenêtre dont la coupe batch est connue.
    fn append(
        &mut self,
        chunk: &[f32],
        transcribe: &mut dyn FnMut(&[f32]) -> Result<String>,
    ) -> Result<()> {
        self.admit(chunk.len())
            .map_err(|limit| InferError::Config(limit.message().to_string()))?;
        self.window.extend_from_slice(chunk);
        self.received += chunk.len();
        self.chunks += 1;
        while self.window.len() > N_SAMPLES + HOP {
            let cut = audio_segments(&self.window[..=N_SAMPLES + HOP])
                .first()
                .map_or(N_SAMPLES, |range| range.end);
            self.inferences += 1;
            let text = transcribe(&self.window[..cut])?;
            if !text.is_empty() {
                self.closed.push(text);
            }
            self.window.drain(..cut);
            // `drain` garde la capacité : un pic ne doit pas rester alloué.
            self.window.shrink_to(RETAINED_WINDOW_SAMPLES);
            self.committed.clear();
            self.previous.clear();
            self.last = None;
        }
        Ok(())
    }

    /// Ajoute un dernier morceau (éventuellement vide), termine la session et
    /// renvoie le texte final, identique au batch sur tout le PCM reçu.
    ///
    /// Sans audio depuis le dernier [`Self::push`], l'hypothèse déjà décodée
    /// sur la même fenêtre est réutilisée : le decode greedy est déterministe.
    ///
    /// # Errors
    ///
    /// Propage l'échec de `transcribe`.
    pub fn finish(
        mut self,
        chunk: &[f32],
        transcribe: &mut dyn FnMut(&[f32]) -> Result<String>,
    ) -> Result<String> {
        self.append(chunk, transcribe)?;
        let mut texts = self.closed;
        match self.last {
            Some((len, text)) if len == self.window.len() && len <= N_SAMPLES => texts.push(text),
            _ => {
                for range in audio_segments(&self.window) {
                    if !range.is_empty() {
                        self.inferences += 1;
                        texts.push(transcribe(&self.window[range])?);
                    }
                }
            }
        }
        texts.retain(|text| !text.is_empty());
        Ok(texts.join(" "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deltas_concatenate_to_the_trimmed_text() {
        let mut deltas = TextDeltas::new(false);
        let steps = [
            " Bon",
            " Bonjour",
            " Bonjour ",
            " Bonjour \u{FFFD}",
            " Bonjour à",
            " Bonjour à tous.",
        ];
        let out: String = steps.iter().filter_map(|s| deltas.update(s)).collect();
        assert_eq!(out, "Bonjour à tous.");
        let mut next = TextDeltas::new(true);
        assert_eq!(next.update(" Suite").as_deref(), Some(" Suite"));
        assert_eq!(next.update(" Suite.").as_deref(), Some("."));
    }

    /// Transcription factice : un mot par seconde d'audio, `w<seconde absolue>`.
    fn fake(samples: &[f32]) -> Result<String> {
        let base = samples.first().map_or(0, |&s| s as usize);
        Ok((0..samples.len() / 16_000)
            .map(|second| format!("w{}", base + second))
            .collect::<Vec<_>>()
            .join(" "))
    }

    /// PCM dont chaque échantillon vaut sa seconde absolue (énergie croissante).
    fn pcm(seconds: usize) -> Vec<f32> {
        (0..seconds * 16_000).map(|i| (i / 16_000) as f32).collect()
    }

    fn batch(samples: &[f32]) -> Result<String> {
        let mut texts = Vec::new();
        for range in audio_segments(samples) {
            texts.push(fake(&samples[range])?);
        }
        texts.retain(|text| !text.is_empty());
        Ok(texts.join(" "))
    }

    #[test]
    fn local_agreement_commits_words_seen_twice() -> Result<()> {
        let mut stream = WhisperStream::new();
        let mut hypotheses = ["a b", "a b c", "a x c d", "a x c d"].into_iter();
        let mut transcribe = |_: &[f32]| Ok(hypotheses.next().unwrap_or_default().to_string());
        let first = stream.push(&[0.0; MIN_PARTIAL_SAMPLES], &mut transcribe)?;
        assert_eq!(
            (first.committed.as_str(), first.partial.as_str()),
            ("", "a b")
        );
        let second = stream.push(&[0.0; MIN_PARTIAL_SAMPLES], &mut transcribe)?;
        assert_eq!(
            (second.committed.as_str(), second.partial.as_str()),
            ("a b", "c")
        );
        // Une hypothèse qui contredit l'accord ne retire rien du texte stabilisé.
        let third = stream.push(&[0.0; MIN_PARTIAL_SAMPLES], &mut transcribe)?;
        assert_eq!(
            (third.committed.as_str(), third.partial.as_str()),
            ("a b", "c d")
        );
        Ok(())
    }

    #[test]
    fn long_stream_matches_batch_cuts_and_reuses_last_hypothesis() -> Result<()> {
        let samples = pcm(75);
        let expected = batch(&samples)?;
        for chunk in [1_600, 16_000, 7_777, N_SAMPLES] {
            let mut stream = WhisperStream::new();
            let calls = std::cell::Cell::new(0);
            let mut counted = |s: &[f32]| {
                calls.set(calls.get() + 1);
                fake(s)
            };
            for part in samples.chunks(chunk) {
                stream.push(part, &mut counted)?;
                assert!(stream.window.len() <= N_SAMPLES + HOP);
            }
            let before = calls.get();
            assert_eq!(stream.received_samples(), samples.len());
            assert_eq!(stream.finish(&[], &mut counted)?, expected);
            // Un dernier morceau < 250 ms saute l'hypothèse : le final la recalcule.
            let tail = samples.len() % chunk;
            if chunk >= MIN_PARTIAL_SAMPLES && (tail == 0 || tail >= MIN_PARTIAL_SAMPLES) {
                assert_eq!(calls.get(), before, "dernière hypothèse réutilisée");
            }
            // Partiels espacés d'au moins 250 ms, plus fermetures et final.
            assert!(calls.get() <= samples.len() / MIN_PARTIAL_SAMPLES + 4);
        }
        Ok(())
    }

    #[test]
    fn empty_and_short_chunks_skip_inference_but_keep_state() -> Result<()> {
        let mut stream = WhisperStream::new();
        let calls = std::cell::Cell::new(0);
        let mut counted = |s: &[f32]| {
            calls.set(calls.get() + 1);
            Ok(format!("mot{}", s.len()))
        };
        assert_eq!(
            stream.push(&[], &mut counted)?,
            WhisperStreamUpdate::default()
        );
        assert_eq!(calls.get(), 0, "fenêtre vide : aucune inférence");
        let first = stream.push(&[0.0; MIN_PARTIAL_SAMPLES], &mut counted)?;
        assert_eq!(calls.get(), 1);
        for _ in 0..3 {
            assert_eq!(stream.push(&[], &mut counted)?, first);
            assert_eq!(stream.push(&[0.0; 100], &mut counted)?, first);
        }
        assert_eq!(
            calls.get(),
            1,
            "morceaux vides ou < 250 ms : pas d'inférence"
        );
        stream.push(&[0.0; MIN_PARTIAL_SAMPLES], &mut counted)?;
        assert_eq!((calls.get(), stream.inferences()), (2, 2));
        Ok(())
    }

    #[test]
    fn admission_limits_reject_before_any_change() {
        let mut stream = WhisperStream::new();
        let mut never = |_: &[f32]| -> Result<String> { panic!("aucune inférence attendue") };
        assert_eq!(stream.admit(MAX_CHUNK_SAMPLES), Ok(()));
        assert_eq!(
            stream.admit(MAX_CHUNK_SAMPLES + 1),
            Err(StreamLimit::ChunkTooLarge)
        );
        assert!(stream
            .push(&vec![0.0; MAX_CHUNK_SAMPLES + 1], &mut never)
            .is_err());
        assert_eq!((stream.received_samples(), stream.window.len()), (0, 0));
        stream.received = MAX_SESSION_SAMPLES - 10;
        assert_eq!(stream.admit(11), Err(StreamLimit::SessionTooLong));
        assert_eq!(stream.admit(10), Ok(()));
        stream.chunks = MAX_SESSION_CHUNKS;
        assert_eq!(stream.admit(0), Err(StreamLimit::TooManyChunks));
        stream.chunks = 0;
        stream.inferences = MAX_SESSION_INFERENCES - 2;
        assert_eq!(stream.admit(0), Err(StreamLimit::ComputeBudget));
        assert!(stream.finish(&[], &mut never).is_err());
    }

    #[test]
    fn closed_windows_release_retained_memory() -> Result<()> {
        let mut stream = WhisperStream::new();
        let samples = pcm(150);
        for part in samples.chunks(MAX_CHUNK_SAMPLES) {
            stream.push(part, &mut fake)?;
            assert!(stream.window.len() <= N_SAMPLES + HOP);
            assert!(stream.window.capacity() <= RETAINED_WINDOW_SAMPLES);
        }
        assert_eq!(stream.finish(&[], &mut fake)?, batch(&samples)?);
        Ok(())
    }

    #[test]
    fn finish_after_window_overflow_splits_like_batch() -> Result<()> {
        // 30 s + 5 ms : la coupe batch dépasse la fenêtre de partial.
        let samples: Vec<f32> = pcm(31)[..N_SAMPLES + 80].to_vec();
        let mut stream = WhisperStream::new();
        stream.push(&samples[..1600], &mut fake)?;
        assert_eq!(
            stream.finish(&samples[1600..], &mut fake)?,
            batch(&samples)?
        );
        Ok(())
    }
}
