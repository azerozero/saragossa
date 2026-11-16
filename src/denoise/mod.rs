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
#[cfg(all(test, target_os = "macos", feature = "metal"))]
mod gpu_probe;
mod layers;
mod network;
#[cfg(test)]
mod tests;

use std::path::Path;

use rustfft::num_complex::Complex32;
use rustfft::FftPlanner;

use dsp::{Analysis, FeatureState, Synthesis, FREQ_BINS, NB_DF, NB_ERB};
pub use dsp::{FFT_SIZE, HOP_SIZE, SAMPLE_RATE};
use network::{NetState, Network, ORDER};

use crate::{InferError, Result};

/// Anticipation des convolutions d'entrée (`conv_lookahead`), en trames.
const CONV_LOOKAHEAD: usize = 2;
/// Anticipation du filtre profond (`df_lookahead`), en trames.
const DF_LOOKAHEAD: usize = 2;

/// Durée maximale d'un traitement hors ligne (cinq minutes).
pub const MAX_DURATION_SECS: u64 = 300;
/// Borne de la sortie mono rééchantillonnée à 48 kHz.
pub const MAX_AUDIO_SAMPLES: usize = MAX_DURATION_SECS as usize * SAMPLE_RATE as usize;

/// Vérifie la durée et la taille à 48 kHz sans allouer de tampon audio.
///
/// # Errors
///
/// Refuse les fréquences invalides, les débordements et les signaux trop longs.
pub fn validate_input_length(samples: u64, rate: u32) -> crate::wire_audio::Result<usize> {
    use crate::wire_audio::{validate_wav_limits, AudioParseError};
    validate_wav_limits(rate, 1, 0)?;
    if samples > MAX_DURATION_SECS * u64::from(rate) {
        return Err(AudioParseError(format!(
            "audio enhance dépasse la durée maximale de {MAX_DURATION_SECS} s (5 min)"
        )));
    }
    let output = samples
        .checked_mul(u64::from(SAMPLE_RATE))
        .map(|n| n.div_ceil(u64::from(rate)))
        .filter(|&n| n <= MAX_AUDIO_SAMPLES as u64)
        .ok_or_else(|| {
            AudioParseError(format!(
                "audio enhance dépasse {MAX_AUDIO_SAMPLES} échantillons rééchantillonnés"
            ))
        })?;
    Ok(output as usize)
}

/// Décode un WAV pour le débruitage, en bornant les tailles avant le décodage.
///
/// # Errors
///
/// Refuse les WAV invalides ou dont la durée ou la taille rééchantillonnée dépasse les limites.
pub fn decode_wav(bytes: &[u8]) -> Result<(Vec<f32>, u32)> {
    crate::wire_audio::wav_to_mono_validated(bytes, |samples, rate| {
        validate_input_length(samples, rate).map(|_| ())
    })
    .map_err(InferError::Audio)
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
    /// Refuse les signaux de plus de cinq minutes avant toute allocation audio.
    pub fn enhance(&self, audio: &[f32]) -> Result<Vec<f32>> {
        validate_input_length(audio.len() as u64, SAMPLE_RATE).map_err(InferError::Audio)?;
        let padded_len = (audio.len() + FFT_SIZE).div_ceil(HOP_SIZE) * HOP_SIZE;
        let mut stream = self.stream();
        let mut padded = audio.to_vec();
        padded.resize(padded_len, 0.0);
        let frames = padded.len() / HOP_SIZE;
        let mut out = Vec::with_capacity((frames + CONV_LOOKAHEAD) * HOP_SIZE);
        let mut hop = [0.0_f32; HOP_SIZE];
        for chunk in padded.chunks_exact(HOP_SIZE) {
            stream.process_hop(chunk, &mut hop)?;
            out.extend_from_slice(&hop);
        }
        out.extend(stream.finish());
        let start = DfStream::DELAY_SAMPLES;
        Ok(out[start..start + audio.len()].to_vec())
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
        }
    }
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
}

impl DfStream<'_> {
    /// Retard entrée → sortie : `N − H` (STFT) + deux trames d'anticipation.
    pub const DELAY_SAMPLES: usize = FFT_SIZE - HOP_SIZE + CONV_LOOKAHEAD * HOP_SIZE;

    /// SNR local estimé (dB) de la dernière trame produite.
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
        self.frames += 1;
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
        let step = self.model.net.step(&mut self.state, erb, cplx);
        self.last_lsnr = step.lsnr;
        let center = &self.spec_ring[DF_LOOKAHEAD];
        let inv = &self.model.net.erb_inv;
        let mut spec = vec![Complex32::default(); FREQ_BINS];
        for f in NB_DF..FREQ_BINS {
            let gain: f32 = (0..NB_ERB)
                .map(|b| step.mask[b] * inv[b * FREQ_BINS + f])
                .sum();
            spec[f] = center[f] * gain;
        }
        for (f, slot) in spec.iter_mut().enumerate().take(NB_DF) {
            let mut acc = Complex32::default();
            for n in 0..ORDER {
                let c = &step.coefs[(f * ORDER + n) * 2..][..2];
                acc += self.spec_ring[n][f] * Complex32::new(c[0], c[1]);
            }
            *slot = acc;
        }
        self.synthesis.process(&spec, output);
    }
}
