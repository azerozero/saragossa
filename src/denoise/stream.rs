//! Adaptation des morceaux arbitraires aux hops DFN3, sans historique audio.
use super::{DeepFilterNet3, DfStream, FFT_SIZE, HOP_SIZE};
use crate::{InferError, Result};

/// Flux mono 48 kHz, compensé du retard, indépendant du découpage des appels.
/// Seul le vecteur rendu dépend de la taille du morceau fourni par l'appelant.
pub struct DenoiseStream<'a> {
    buffered: Buffered<DfStream<'a>>,
}

impl<'a> DenoiseStream<'a> {
    /// Ouvre un état GRU/STFT indépendant, partageant les poids du modèle.
    #[must_use]
    pub fn new(model: &'a DeepFilterNet3) -> Self {
        Self::from_stream(model.stream())
    }

    pub(crate) fn from_stream(stream: DfStream<'a>) -> Self {
        Self {
            buffered: Buffered::new(stream),
        }
    }

    #[cfg_attr(
        not(all(target_os = "macos", feature = "metal")),
        allow(dead_code, reason = "seul le chemin Metal relit le flux")
    )]
    pub(crate) fn stream_mut(&mut self) -> &mut DfStream<'a> {
        &mut self.buffered.processor
    }

    /// Fournit un morceau et restitue immédiatement les échantillons disponibles.
    ///
    /// # Errors
    /// Refuse un appel après `finish` et propage les erreurs du modèle.
    pub fn push(&mut self, samples: &[f32]) -> Result<Vec<f32>> {
        self.buffered.push(samples)
    }

    /// Vide le retard une seule fois, sans ajouter d'échantillons au signal.
    /// Les appels suivants rendent un vecteur vide.
    ///
    /// # Errors
    /// Propage les erreurs du modèle.
    pub fn finish(&mut self) -> Result<Vec<f32>> {
        self.buffered.finish()
    }
}

trait Hops {
    fn hop(&mut self, input: &[f32], output: &mut [f32]) -> Result<()>;
    fn tail(&mut self) -> Vec<f32>;
}

impl Hops for DfStream<'_> {
    fn hop(&mut self, input: &[f32], output: &mut [f32]) -> Result<()> {
        self.process_hop(input, output)
    }
    fn tail(&mut self) -> Vec<f32> {
        self.finish()
    }
}

struct Buffered<P> {
    processor: P,
    pending: [f32; HOP_SIZE],
    used: usize,
    skip: usize,
    owed: usize,
    finished: bool,
}

impl<P: Hops> Buffered<P> {
    fn new(processor: P) -> Self {
        Self {
            processor,
            pending: [0.0; HOP_SIZE],
            used: 0,
            skip: DfStream::DELAY_SAMPLES,
            owed: 0,
            finished: false,
        }
    }

    fn emit(&mut self, hop: &[f32], out: &mut Vec<f32>) {
        let skip = self.skip.min(hop.len());
        self.skip -= skip;
        let count = self.owed.min(hop.len() - skip);
        out.extend_from_slice(&hop[skip..skip + count]);
        self.owed -= count;
    }

    fn push(&mut self, mut input: &[f32]) -> Result<Vec<f32>> {
        if self.finished {
            return Err(InferError::Config("flux de débruitage clos".into()));
        }
        let mut out = Vec::new();
        while !input.is_empty() {
            let n = input.len().min(HOP_SIZE - self.used);
            self.pending[self.used..self.used + n].copy_from_slice(&input[..n]);
            self.used += n;
            self.owed += n;
            input = &input[n..];
            if self.used == HOP_SIZE {
                let mut hop = [0.0; HOP_SIZE];
                self.processor.hop(&self.pending, &mut hop)?;
                self.used = 0;
                self.emit(&hop, &mut out);
            }
        }
        Ok(out)
    }

    fn finish(&mut self) -> Result<Vec<f32>> {
        if self.finished {
            return Ok(Vec::new());
        }
        self.finished = true;
        let mut out = Vec::new();
        // Même padding que le batch historique : ceil((n + FFT_SIZE) / HOP).
        let hops = FFT_SIZE / HOP_SIZE + usize::from(self.used > 0);
        self.pending[self.used..].fill(0.0);
        for _ in 0..hops {
            let mut hop = [0.0; HOP_SIZE];
            self.processor.hop(&self.pending, &mut hop)?;
            self.emit(&hop, &mut out);
            self.pending.fill(0.0);
        }
        self.used = 0;
        let tail = self.processor.tail();
        self.emit(&tail, &mut out);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Identity;
    impl Hops for Identity {
        fn hop(&mut self, input: &[f32], output: &mut [f32]) -> Result<()> {
            output.copy_from_slice(input);
            Ok(())
        }
        fn tail(&mut self) -> Vec<f32> {
            vec![0.0; 2 * HOP_SIZE]
        }
    }

    #[test]
    fn one_hour_keeps_fixed_buffer_and_bounded_debt() {
        // Même adaptateur de production, noyau substitué pour ne pas exécuter
        // 360 000 inférences : les états GRU/STFT sont de taille fixe.
        let mut stream = Buffered::new(Identity);
        let chunk = [0.125; 4800];
        let mut produced = 0;
        for _ in 0..36_000 {
            produced += stream.push(&chunk).unwrap().len();
            assert!(stream.used < HOP_SIZE);
            assert!(stream.owed <= DfStream::DELAY_SAMPLES + HOP_SIZE);
            assert_eq!(stream.pending.len(), HOP_SIZE);
        }
        produced += stream.finish().unwrap().len();
        assert_eq!(produced, 3600 * 48_000);
        assert!(stream.finish().unwrap().is_empty());
        assert!(stream.push(&[0.0]).is_err());
    }
}
