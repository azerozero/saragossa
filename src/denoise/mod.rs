//! Débruitage temps réel DeepFilterNet3 (48 kHz, pas de 10 ms), sur CPU.
//!
//! Port Rust du modèle de Schröter et al. (2023), poids officiels
//! `Rikorose/DeepFilterNet` sous double licence MIT / Apache-2.0. Voir
//! `docs/voice-isolation.md` pour la spécification et les mesures.
//!
//! Le flux est causal avec une anticipation de deux trames : chaque hop de
//! 480 échantillons fourni à [`DfStream::process_hop`] restitue le hop retardé
//! de [`DfStream::DELAY_SAMPLES`] échantillons (30 ms, plus le tamponnage d'un hop).

mod dsp;
mod exact_math;
#[cfg(all(target_os = "macos", feature = "metal"))]
mod gpu;
#[cfg(all(test, target_os = "macos", feature = "metal"))]
mod gpu_probe;
mod layers;
mod network;
mod resample;
mod stream;
pub use resample::StreamResampler;
pub mod wav;
pub use stream::DenoiseStream;
#[cfg(test)]
mod tests;

use std::path::Path;

use rustfft::num_complex::Complex32;
use rustfft::FftPlanner;

pub use crate::audio_resample::resample;
use dsp::{Analysis, FeatureState, Synthesis, FREQ_BINS, NB_DF, NB_ERB};
pub use dsp::{FFT_SIZE, HOP_SIZE, SAMPLE_RATE};
use layers::Math;
use network::{NetState, Network, ORDER};

use crate::{InferError, Result};

/// Anticipation des convolutions d'entrée (`conv_lookahead`), en trames.
const CONV_LOOKAHEAD: usize = 2;
/// Anticipation du filtre profond (`df_lookahead`), en trames.
const DF_LOOKAHEAD: usize = 2;

/// Décode un WAV complet ; préférer [`wav::WavStream`] pour les fichiers longs.
///
/// # Errors
/// Refuse les en-têtes et les échantillons invalides.
pub fn decode_wav(bytes: &[u8]) -> Result<(Vec<f32>, u32)> {
    let mut reader = wav::WavStream::new(std::io::Cursor::new(bytes)).map_err(InferError::Audio)?;
    let rate = reader.sample_rate();
    let mut output = Vec::new();
    loop {
        let chunk = reader.read_chunk().map_err(InferError::Audio)?;
        if chunk.is_empty() {
            break;
        }
        output.extend(chunk);
    }
    reader
        .finish_input()
        .map_err(|e| InferError::Audio(crate::wire_audio::AudioParseError(e.to_string())))?;
    Ok((output, rate))
}

/// Modèle DeepFilterNet3 chargé.
pub struct DeepFilterNet3 {
    net: Network,
    widths: Vec<usize>,
    window: Vec<f32>,
}

impl DeepFilterNet3 {
    /// Charge `dfn3.safetensors` (state dict officiel converti par `tools/voice/convert_dfn3.py`).
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si un poids manque, a une forme inattendue, ou si la
    /// banque ERB du checkpoint ne correspond pas à celle de libDF.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let weights = crate::safetensor::load_f32_tensors(path)?;
        let net = Network::load(weights, FREQ_BINS)?;
        let widths = dsp::erb_widths();
        let mut start = 0;
        for (band, &width) in widths.iter().enumerate() {
            for f in 0..FREQ_BINS {
                let inside = (start..start + width).contains(&f);
                let fb = net.erb_fb[f * NB_ERB + band];
                let inv = net.erb_inv[band * FREQ_BINS + f];
                let ok = if inside {
                    (fb - 1.0 / width as f32).abs() < 1e-6 && inv == 1.0
                } else {
                    fb == 0.0 && inv == 0.0
                };
                if !ok {
                    return Err(InferError::Config(format!(
                        "banque ERB DeepFilterNet3 incohérente (bande {band}, bin {f})"
                    )));
                }
            }
            start += width;
        }
        Ok(Self {
            net,
            widths,
            window: dsp::vorbis_window(),
        })
    }

    /// Débruite un signal mono 48 kHz complet, sans retard : équivalent de `df.enhance(pad=True)`.
    ///
    /// # Errors
    ///
    /// Propage les erreurs du traitement des trames.
    pub fn enhance(&self, audio: &[f32]) -> Result<Vec<f32>> {
        let mut stream = DenoiseStream::new(self);
        let mut out = stream.push(audio)?;
        out.extend(stream.finish()?);
        Ok(out)
    }

    /// Comme [`Self::enhance`], avec le réseau en lot sur GPU Metal (hors ligne).
    ///
    /// Sortie identique au bit à [`Self::enhance_common`] : `exp`, `tanh` et la
    /// sigmoïde y suivent `exact_math` au lieu de la libm, ce qui l'écarte de
    /// quelques ulp de [`Self::enhance`].
    ///
    /// # Errors
    /// Erreurs Metal (pas de GPU, compilation, exécution).
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn enhance_gpu(&self, audio: &[f32]) -> Result<Vec<f32>> {
        // Passe 1 : features de chaque pas, sans réseau ; mêmes trames que le flux.
        let mut record = self.stream();
        record.steps = Steps::Record(Vec::new(), Vec::new());
        let mut stream = DenoiseStream::from_stream(record);
        stream.push(audio)?;
        stream.finish()?;
        let Steps::Record(erb, cplx) =
            std::mem::replace(&mut stream.stream_mut().steps, Steps::Live)
        else {
            return Err(InferError::Config("enregistrement DFN3 perdu".into()));
        };
        let (mask, coefs) = gpu::run(&self.net, &erb, &cplx)?;
        // Passe 2 : mêmes trames, sorties du réseau rejouées.
        let mut replay = self.stream();
        replay.steps = Steps::Replay {
            mask,
            coefs,
            next: 0,
        };
        let mut stream = DenoiseStream::from_stream(replay);
        let mut out = stream.push(audio)?;
        out.extend(stream.finish()?);
        Ok(out)
    }

    /// [`Self::enhance`] en variante `Math::Common` : référence CPU du chemin GPU.
    ///
    /// # Errors
    /// Propage les erreurs du traitement des trames.
    pub fn enhance_common(&self, audio: &[f32]) -> Result<Vec<f32>> {
        let mut inner = self.stream();
        inner.state.math = Math::Common;
        let mut stream = DenoiseStream::from_stream(inner);
        let mut out = stream.push(audio)?;
        out.extend(stream.finish()?);
        Ok(out)
    }

    /// [`Self::enhance`] en variante `Math::Libm` (historique, kill-switch
    /// `SARAGOSSA_DENOISE_LIBM`), indépendamment du flag.
    ///
    /// # Errors
    /// Propage les erreurs du traitement des trames.
    pub fn enhance_libm(&self, audio: &[f32]) -> Result<Vec<f32>> {
        let mut inner = self.stream();
        inner.state.math = Math::Libm;
        let mut stream = DenoiseStream::from_stream(inner);
        let mut out = stream.push(audio)?;
        out.extend(stream.finish()?);
        Ok(out)
    }

    /// Ouvre un flux temps réel.
    #[must_use]
    pub fn stream(&self) -> DfStream<'_> {
        let mut planner = FftPlanner::new();
        DfStream {
            model: self,
            analysis: Analysis::new(&mut planner, &self.window),
            synthesis: Synthesis::new(&mut planner, &self.window),
            features: FeatureState::new(self.widths.clone()),
            state: self.net.new_state(),
            spec_ring: vec![vec![Complex32::default(); FREQ_BINS]; 2 * DF_LOOKAHEAD + 1],
            frames: 0,
            last_lsnr: 0.0,
            steps: Steps::Live,
            threads: 1,
        }
    }
}

/// Origine des sorties du réseau d'un flux.
#[cfg_attr(
    not(all(target_os = "macos", feature = "metal")),
    allow(dead_code, reason = "seul le chemin Metal enregistre et rejoue")
)]
enum Steps {
    /// Réseau CPU pas à pas.
    Live,
    /// Collecte les features `[pas][NB_ERB]` et `[pas][2·NB_DF]` sans calculer le réseau.
    Record(Vec<f32>, Vec<f32>),
    /// Rejoue des sorties calculées en lot : masques `[pas][NB_ERB]`, coefficients `[pas][NB_DF·2·ORDER]`.
    Replay {
        mask: Vec<f32>,
        coefs: Vec<f32>,
        next: usize,
    },
}

/// Flux DeepFilterNet3 : un hop de 10 ms en entrée, un hop retardé en sortie.
pub struct DfStream<'a> {
    model: &'a DeepFilterNet3,
    analysis: Analysis,
    synthesis: Synthesis,
    features: FeatureState,
    state: NetState,
    /// Spectres `t - 4 ..= t`, soit `τ - 2 ..= τ + 2` pour la trame de sortie τ = t − 2.
    spec_ring: Vec<Vec<Complex32>>,
    frames: usize,
    last_lsnr: f32,
    steps: Steps,
    /// Cœurs de [`DfStream::process_hops`] ; 1 garde le chemin trame par trame.
    threads: usize,
}

impl DfStream<'_> {
    /// Retard entrée → sortie : `N − H` (STFT) + deux trames d'anticipation.
    pub const DELAY_SAMPLES: usize = FFT_SIZE - HOP_SIZE + CONV_LOOKAHEAD * HOP_SIZE;

    /// SNR local estimé (dB) de la dernière trame produite ; 0 sur le chemin
    /// GPU (`DeepFilterNet3::enhance_gpu`), qui ne calcule pas le LSNR.
    #[must_use]
    pub fn last_lsnr(&self) -> f32 {
        self.last_lsnr
    }

    /// Traite un hop de `HOP_SIZE` échantillons et écrit le hop de sortie.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si `input` ou `output` n'ont pas `HOP_SIZE` échantillons.
    pub fn process_hop(&mut self, input: &[f32], output: &mut [f32]) -> Result<()> {
        if input.len() != HOP_SIZE || output.len() != HOP_SIZE {
            return Err(InferError::Dimension(format!(
                "hop DeepFilterNet3 de {HOP_SIZE} échantillons attendu, reçu {} → {}",
                input.len(),
                output.len()
            )));
        }
        self.spec_ring.rotate_left(1);
        let (mut erb, mut cplx) = ([0.0_f32; NB_ERB], [0.0_f32; 2 * NB_DF]);
        if let Some(slot) = self.spec_ring.last_mut() {
            self.analysis.process(input, slot);
            self.features.process(slot, &mut erb, &mut cplx);
        }
        self.frames = (self.frames + 1).min(CONV_LOOKAHEAD + 1);
        // `pad_feat` ignore les deux premières trames de features.
        if self.frames <= CONV_LOOKAHEAD {
            output.fill(0.0);
            return Ok(());
        }
        self.emit(&erb, &cplx, output);
        Ok(())
    }

    /// Vide l'anticipation en fin de signal (features et spectres nuls, comme le padding PyTorch).
    #[must_use]
    pub fn finish(&mut self) -> Vec<f32> {
        let mut out = vec![0.0; CONV_LOOKAHEAD * HOP_SIZE];
        let zeros = ([0.0_f32; NB_ERB], [0.0_f32; 2 * NB_DF]);
        for hop in out.chunks_exact_mut(HOP_SIZE) {
            self.spec_ring.rotate_left(1);
            if let Some(slot) = self.spec_ring.last_mut() {
                slot.fill(Complex32::default());
            }
            self.emit(&zeros.0, &zeros.1, hop);
        }
        out
    }

    fn emit(&mut self, erb: &[f32], cplx: &[f32], output: &mut [f32]) {
        let live;
        let (mask, coefs): (&[f32], &[f32]) = match &mut self.steps {
            Steps::Live => {
                live = self.model.net.step(&mut self.state, erb, cplx);
                self.last_lsnr = live.lsnr;
                (&live.mask, &live.coefs)
            }
            Steps::Record(e, c) => {
                e.extend_from_slice(erb);
                c.extend_from_slice(cplx);
                output.fill(0.0);
                return;
            }
            Steps::Replay { mask, coefs, next } => {
                let (i, width) = (*next, NB_DF * 2 * ORDER);
                *next += 1;
                (
                    mask.get(i * NB_ERB..(i + 1) * NB_ERB).expect(
                        "invariant: le rejeu suit les trames de l'enregistrement, même audio",
                    ),
                    coefs.get(i * width..(i + 1) * width).expect(
                        "invariant: le rejeu suit les trames de l'enregistrement, même audio",
                    ),
                )
            }
        };
        render(
            &self.model.net,
            &mut self.synthesis,
            &self.spec_ring,
            (mask, coefs),
            output,
        );
    }

    /// Comme `process_hop` sur chaque hop de `input`, le réseau tournant sur
    /// `threads` cœurs (voir `network::batch`). Sortie identique bit à bit.
    /// L'enregistrement et le rejeu du chemin GPU restent hop par hop.
    fn process_hops(&mut self, input: &[f32], output: &mut [f32]) -> Result<()> {
        if !matches!(self.steps, Steps::Live) {
            let mut pairs = input
                .chunks_exact(HOP_SIZE)
                .zip(output.chunks_exact_mut(HOP_SIZE));
            return pairs.try_for_each(|(i, o)| self.process_hop(i, o));
        }
        let (mut erbs, mut cplxs, mut rings, mut emits) = (vec![], vec![], vec![], vec![]);
        for hop in input.chunks_exact(HOP_SIZE) {
            self.spec_ring.rotate_left(1);
            let (mut erb, mut cplx) = ([0.0_f32; NB_ERB], [0.0_f32; 2 * NB_DF]);
            if let Some(slot) = self.spec_ring.last_mut() {
                self.analysis.process(hop, slot);
                self.features.process(slot, &mut erb, &mut cplx);
            }
            self.frames = (self.frames + 1).min(CONV_LOOKAHEAD + 1);
            let emit = self.frames > CONV_LOOKAHEAD;
            if emit {
                erbs.extend(erb);
                cplxs.extend(cplx);
                rings.push(self.spec_ring.clone());
            }
            emits.push(emit);
        }
        let net = &self.model.net;
        let steps = net.step_batch(&mut self.state, &erbs, &cplxs, self.threads);
        let mut steps = steps.iter().zip(&rings);
        for (out, emit) in output.chunks_exact_mut(HOP_SIZE).zip(emits) {
            let next = if emit { steps.next() } else { None };
            match next {
                Some((step, ring)) => {
                    self.last_lsnr = step.lsnr;
                    render(
                        net,
                        &mut self.synthesis,
                        ring,
                        (&step.mask, &step.coefs),
                        out,
                    );
                }
                None => out.fill(0.0),
            }
        }
        Ok(())
    }
}

/// Applique masque ERB et filtre profond au spectre central de `ring`, puis synthétise le hop.
fn render(
    net: &Network,
    synthesis: &mut Synthesis,
    ring: &[Vec<Complex32>],
    (mask, coefs): (&[f32], &[f32]),
    output: &mut [f32],
) {
    let center = &ring[DF_LOOKAHEAD];
    let inv = &net.erb_inv;
    let mut spec = vec![Complex32::default(); FREQ_BINS];
    for f in NB_DF..FREQ_BINS {
        let gain: f32 = (0..NB_ERB).map(|b| mask[b] * inv[b * FREQ_BINS + f]).sum();
        spec[f] = center[f] * gain;
    }
    for (f, slot) in spec.iter_mut().enumerate().take(NB_DF) {
        let mut acc = Complex32::default();
        for (n, frame) in ring.iter().enumerate().take(ORDER) {
            let c = &coefs[(f * ORDER + n) * 2..][..2];
            acc += frame[f] * Complex32::new(c[0], c[1]);
        }
        *slot = acc;
    }
    synthesis.process(&spec, output);
}
