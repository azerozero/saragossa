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

    /// Comme [`Self::new`], le réseau tournant sur `threads` cœurs par blocs de
    /// hops. La sortie reste identique bit à bit au chemin mono-cœur ; seule la
    /// latence change, un bloc n'étant rendu qu'une fois complet.
    #[must_use]
    pub fn with_threads(model: &'a DeepFilterNet3, threads: usize) -> Self {
        let mut stream = model.stream();
        // Variante faible latence : convolutions internes à état, pas multi-cœur non porté.
        stream.threads = if model.net.is_std() {
            threads.max(1)
        } else {
            1
        };
        Self::from_stream(stream)
    }

    #[cfg_attr(
        not(gpu_metal),
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

/// Hops traités d'un bloc en mode multi-cœur : 2,56 s de signal.
const BLOCK_HOPS: usize = 256;

trait Hops {
    fn hop(&mut self, input: &[f32], output: &mut [f32]) -> Result<()>;
    fn tail(&mut self) -> Vec<f32>;
    /// Échantillons de retard à retirer en tête de sortie.
    fn delay(&self) -> usize {
        DfStream::DELAY_SAMPLES
    }
    /// Nombre de hops entiers que [`Self::hops`] veut traiter d'un coup (1 : hop par hop).
    fn block(&self) -> usize {
        1
    }
    /// Plusieurs hops consécutifs, même résultat que `hop` sur chacun.
    fn hops(&mut self, input: &[f32], output: &mut [f32]) -> Result<()> {
        let mut pairs = input
            .chunks_exact(HOP_SIZE)
            .zip(output.chunks_exact_mut(HOP_SIZE));
        pairs.try_for_each(|(i, o)| self.hop(i, o))
    }
}

impl Hops for DfStream<'_> {
    fn hop(&mut self, input: &[f32], output: &mut [f32]) -> Result<()> {
        self.process_hop(input, output)
    }
    fn block(&self) -> usize {
        if self.threads > 1 {
            BLOCK_HOPS
        } else {
            1
        }
    }
    fn hops(&mut self, input: &[f32], output: &mut [f32]) -> Result<()> {
        if self.threads <= 1 {
            let mut pairs = input
                .chunks_exact(HOP_SIZE)
                .zip(output.chunks_exact_mut(HOP_SIZE));
            return pairs.try_for_each(|(i, o)| self.process_hop(i, o));
        }
        self.process_hops(input, output)
    }
    fn tail(&mut self) -> Vec<f32> {
        self.finish()
    }
    fn delay(&self) -> usize {
        self.delay_samples()
    }
}

struct Buffered<P> {
    processor: P,
    /// Un bloc de `processor.block()` hops ; `used` premiers échantillons valides.
    pending: Vec<f32>,
    /// Sorties de `run`, réutilisées d'un appel à l'autre.
    hops: Vec<f32>,
    used: usize,
    skip: usize,
    owed: usize,
    finished: bool,
}

impl<P: Hops> Buffered<P> {
    fn new(processor: P) -> Self {
        Self {
            pending: vec![0.0; processor.block() * HOP_SIZE],
            skip: processor.delay(),
            processor,
            hops: Vec::new(),
            used: 0,
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

    /// Traite des hops entiers et restitue chacun dans l'ordre.
    fn run(&mut self, input: &[f32], out: &mut Vec<f32>) -> Result<()> {
        let mut hops = std::mem::take(&mut self.hops);
        hops.resize(input.len(), 0.0);
        let result = self.processor.hops(input, &mut hops);
        if result.is_ok() {
            for hop in hops.chunks_exact(HOP_SIZE) {
                self.emit(hop, out);
            }
        }
        self.hops = hops;
        result
    }

    fn push(&mut self, mut input: &[f32]) -> Result<Vec<f32>> {
        if self.finished {
            return Err(InferError::Config("flux de débruitage clos".into()));
        }
        let mut out = Vec::new();
        while !input.is_empty() {
            let n = input.len().min(self.pending.len() - self.used);
            self.pending[self.used..self.used + n].copy_from_slice(&input[..n]);
            self.used += n;
            self.owed += n;
            input = &input[n..];
            if self.used == self.pending.len() {
                let block = std::mem::take(&mut self.pending);
                self.used = 0;
                let result = self.run(&block, &mut out);
                self.pending = block;
                result?;
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
        let hops = FFT_SIZE / HOP_SIZE + self.used.div_ceil(HOP_SIZE);
        let mut rest = self.pending[..self.used].to_vec();
        rest.resize(hops * HOP_SIZE, 0.0);
        self.used = 0;
        self.run(&rest, &mut out)?;
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
