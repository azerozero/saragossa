//! Diarisation incrémentale : audio par morceaux, segments émis au fil de l'eau.
//!
//! Même calcul que `DiarizationModel::speaker_probabilities` (mêmes morceaux,
//! même cache des locuteurs). L'état reste borné : fenêtre de 512 échantillons,
//! moins de 8 trames mel en attente, embeddings du morceau courant et de son
//! contexte, cache et FIFO, un début de segment par locuteur. Un morceau
//! n'est traité qu'une fois son anticipation reçue ; `finish` vide le reste.

use super::cache::SpeakerCache;
use super::frontend::{self, MelStream};
use super::{DiarizationModel, Resident, SpeakerSegment, DEFAULT_THRESHOLD};
use crate::{ForwardRuntime, InferError, Result, Tensor};

/// Nouveautés d'un appel : probabilités `[n, num_speakers]` des trames de 10 ms
/// suivantes et segments clos (ordre de fermeture, pas de début).
#[derive(Clone, Debug, Default)]
pub struct StreamUpdate {
    pub probabilities: Vec<f32>,
    pub segments: Vec<SpeakerSegment>,
}

/// Session de diarisation incrémentale sur un signal mono 16 kHz.
pub struct DiarizationStream<'m, 'r> {
    model: &'m DiarizationModel,
    runtime: ForwardRuntime<'r>,
    resident: Resident,
    mel: MelStream,
    pending_mel: Vec<f32>,
    embeds: Vec<f32>,
    /// Indice (en trames encodeur) de la première trame de `embeds`.
    base: usize,
    /// Prochain morceau à traiter, en trames encodeur.
    start: usize,
    cache: SpeakerCache,
    emitted: usize,
    open: Vec<Option<usize>>,
}

impl<'m, 'r> DiarizationStream<'m, 'r> {
    /// Ouvre une session.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si l'encodeur résident ne peut pas être construit.
    pub fn new(model: &'m DiarizationModel, runtime: ForwardRuntime<'r>) -> Result<Self> {
        let c = &model.config;
        Ok(Self {
            model,
            runtime,
            resident: model.resident_for(runtime)?,
            mel: MelStream::new(c.num_mel_bins),
            pending_mel: Vec::new(),
            embeds: Vec::new(),
            base: 0,
            start: 0,
            cache: SpeakerCache::new(c.cache.clone(), c.hidden_size),
            emitted: 0,
            open: vec![None; c.num_speakers],
        })
    }

    /// Ajoute des échantillons et traite les morceaux désormais complets.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si un calcul échoue.
    pub fn push(&mut self, samples: &[f32]) -> Result<StreamUpdate> {
        self.mel.push(samples);
        let rows = self.mel.ready(false);
        self.absorb(rows)
    }

    /// Termine le signal : trame masquée, derniers morceaux, segments ouverts.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si l'audio total est plus court qu'une trame.
    pub fn finish(mut self) -> Result<StreamUpdate> {
        let rows = self.mel.ready(true);
        let valid = self.mel.frames_valid();
        if valid == 0 {
            return Err(InferError::Dimension(
                "diarisation: audio plus court qu'une trame".into(),
            ));
        }
        let c = &self.model.config;
        self.pending_mel.extend_from_slice(&rows);
        // Trame de centrage masquée (zéros) puis groupe final complété de zéros.
        let n_mels = c.num_mel_bins;
        self.pending_mel
            .resize(self.pending_mel.len() + n_mels, 0.0);
        let frames = self.pending_mel.len() / n_mels;
        let stacked = frontend::stack_frames(&self.pending_mel, frames, n_mels, c.subsampling);
        self.pending_mel.clear();
        self.embed(stacked)?;
        let mut update = self.run_chunks(true, valid.div_ceil(c.subsampling))?;
        for (speaker, open) in self.open.iter().enumerate() {
            if let Some(start_frame) = *open {
                update.segments.push(SpeakerSegment {
                    speaker,
                    start_frame,
                    end_frame: self.emitted,
                });
            }
        }
        Ok(update)
    }

    fn embed(&mut self, stacked: Vec<f32>) -> Result<()> {
        let width = self.model.config.subsampling * self.model.config.num_mel_bins;
        let groups = stacked.len() / width;
        if groups > 0 {
            let t = Tensor::from_vec(vec![groups, width], stacked)?;
            let e = self.model.embed.forward_batched(&t, self.runtime)?;
            self.embeds.extend_from_slice(e.data());
        }
        Ok(())
    }

    fn absorb(&mut self, rows: Vec<f32>) -> Result<StreamUpdate> {
        let c = &self.model.config;
        let width = c.subsampling * c.num_mel_bins;
        self.pending_mel.extend_from_slice(&rows);
        let full = self.pending_mel.len() / width * width;
        let stacked: Vec<f32> = self.pending_mel.drain(..full).collect();
        self.embed(stacked)?;
        self.run_chunks(false, usize::MAX)
    }

    fn run_chunks(&mut self, finished: bool, valid_enc: usize) -> Result<StreamUpdate> {
        let m = self.model;
        let c = &m.config;
        let (d, s) = (c.hidden_size, c.num_speakers);
        let mut update = StreamUpdate::default();
        loop {
            let avail = self.base + self.embeds.len() / d;
            let start = self.start;
            if (finished && start >= avail)
                || (!finished && avail < start + c.chunk_length + c.chunk_right_context)
            {
                break;
            }
            let end = (start + c.chunk_length).min(avail);
            let context_start = start.saturating_sub(c.chunk_left_context);
            let chunk_end = (end + c.chunk_right_context).min(avail);
            let mut probs = m.encode_chunk(
                &mut self.cache,
                &self.embeds[(context_start - self.base) * d..(chunk_end - self.base) * d],
                (start - context_start, end - start),
                chunk_end.saturating_sub(valid_enc),
                &self.resident,
                self.runtime,
            )?;
            if finished {
                probs.truncate((self.mel.frames_valid() - self.emitted) * s);
            }
            self.track(&probs, &mut update.segments);
            update.probabilities.extend_from_slice(&probs);
            self.start = end;
            let keep = end.saturating_sub(c.chunk_left_context);
            self.embeds.drain(..(keep - self.base) * d);
            self.base = keep;
        }
        Ok(update)
    }

    /// Ouvre et ferme les segments sur les nouvelles trames.
    fn track(&mut self, probs: &[f32], closed: &mut Vec<SpeakerSegment>) {
        let s = self.open.len();
        for row in probs.chunks_exact(s) {
            for (speaker, p) in row.iter().enumerate() {
                match (*p > DEFAULT_THRESHOLD, self.open[speaker]) {
                    (true, None) => self.open[speaker] = Some(self.emitted),
                    (false, Some(start_frame)) => {
                        closed.push(SpeakerSegment {
                            speaker,
                            start_frame,
                            end_frame: self.emitted,
                        });
                        self.open[speaker] = None;
                    }
                    _ => {}
                }
            }
            self.emitted += 1;
        }
    }
}
