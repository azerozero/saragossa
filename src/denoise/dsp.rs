//! STFT, iSTFT et features ERB de DeepFilterNet3, portés de `libDF/src/lib.rs`.
//!
//! Le STFT de libDF est « temps réel » : chaque trame concatène les `N - H`
//! derniers échantillons et le nouveau hop de `H` échantillons, sans centrage.
//! La normalisation `1 / (N² / 2H)` est appliquée en analyse seulement ; la
//! synthèse est une iFFT non normalisée, fenêtrée puis sommée par overlap-add.

use std::sync::Arc;

use rustfft::num_complex::Complex32;
use rustfft::{Fft, FftPlanner};

/// Fréquence d'échantillonnage du modèle.
pub const SAMPLE_RATE: u32 = 48_000;
/// Taille de FFT et de fenêtre (20 ms).
pub const FFT_SIZE: usize = 960;
/// Pas d'analyse (10 ms).
pub const HOP_SIZE: usize = 480;
/// Nombre de bins de fréquence.
pub const FREQ_BINS: usize = FFT_SIZE / 2 + 1;
/// Nombre de bandes ERB.
pub const NB_ERB: usize = 32;
/// Nombre de bins basses fréquences filtrés par deep filtering.
pub const NB_DF: usize = 96;
/// Coefficient du lissage exponentiel des normalisations : `round(exp(-H/sr/τ), 3)` avec τ = 1 s.
const NORM_ALPHA: f32 = 0.99;
const MIN_NB_ERB_FREQS: usize = 2;

/// Fenêtre de Vorbis `sin(π/2 · sin²(π(n + ½)/N))`, de somme des carrés constante à 50 %.
pub(crate) fn vorbis_window() -> Vec<f32> {
    let half = (FFT_SIZE / 2) as f64;
    (0..FFT_SIZE)
        .map(|i| {
            let s = (0.5 * std::f64::consts::PI * (i as f64 + 0.5) / half).sin();
            (0.5 * std::f64::consts::PI * s * s).sin() as f32
        })
        .collect()
}

fn freq2erb(freq_hz: f32) -> f32 {
    9.265 * (freq_hz / (24.7 * 9.265)).ln_1p()
}

fn erb2freq(n_erb: f32) -> f32 {
    24.7 * 9.265 * ((n_erb / 9.265).exp() - 1.0)
}

/// Largeurs des bandes ERB, en bins, calculées comme `libDF::erb_fb`.
pub(crate) fn erb_widths() -> Vec<usize> {
    let freq_width = SAMPLE_RATE as f32 / FFT_SIZE as f32;
    let erb_low = freq2erb(0.0);
    let erb_high = freq2erb((SAMPLE_RATE / 2) as f32);
    let step = (erb_high - erb_low) / NB_ERB as f32;
    let mut widths = vec![0_usize; NB_ERB];
    let (mut prev, mut over) = (0_i64, 0_i64);
    for (i, width) in widths.iter_mut().enumerate() {
        let fb = (erb2freq(erb_low + (i + 1) as f32 * step) / freq_width).round() as i64;
        let mut nb = fb - prev - over;
        if nb < MIN_NB_ERB_FREQS as i64 {
            over = MIN_NB_ERB_FREQS as i64 - nb;
            nb = MIN_NB_ERB_FREQS as i64;
        } else {
            over = 0;
        }
        *width = nb.max(0) as usize;
        prev = fb;
    }
    // libDF ajoute le bin de Nyquist à la dernière bande puis retire l'excédent.
    widths[NB_ERB - 1] += 1;
    let total: usize = widths.iter().sum();
    widths[NB_ERB - 1] = widths[NB_ERB - 1].saturating_sub(total.saturating_sub(FREQ_BINS));
    widths
}

/// Analyse STFT incrémentale : un hop de `HOP_SIZE` échantillons donne une trame.
pub(crate) struct Analysis {
    fft: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
    mem: Vec<f32>,
    buf: Vec<Complex32>,
}

impl Analysis {
    pub(crate) fn new(planner: &mut FftPlanner<f32>, window: &[f32]) -> Self {
        Self {
            fft: planner.plan_fft_forward(FFT_SIZE),
            window: window.to_vec(),
            mem: vec![0.0; FFT_SIZE - HOP_SIZE],
            buf: vec![Complex32::default(); FFT_SIZE],
        }
    }

    /// Ecrit les `FREQ_BINS` bins de la trame courante dans `out`.
    pub(crate) fn process(&mut self, hop: &[f32], out: &mut [Complex32]) {
        let split = FFT_SIZE - HOP_SIZE;
        for (i, slot) in self.buf.iter_mut().enumerate() {
            let x = if i < split {
                self.mem[i]
            } else {
                hop[i - split]
            };
            *slot = Complex32::new(x * self.window[i], 0.0);
        }
        self.mem.copy_from_slice(hop);
        self.fft.process(&mut self.buf);
        let norm = (2 * HOP_SIZE) as f32 / (FFT_SIZE * FFT_SIZE) as f32;
        for (o, &x) in out.iter_mut().zip(&self.buf[..FREQ_BINS]) {
            *o = x * norm;
        }
    }
}

/// Synthèse iSTFT incrémentale par overlap-add.
pub(crate) struct Synthesis {
    fft: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
    mem: Vec<f32>,
    buf: Vec<Complex32>,
}

impl Synthesis {
    pub(crate) fn new(planner: &mut FftPlanner<f32>, window: &[f32]) -> Self {
        Self {
            fft: planner.plan_fft_inverse(FFT_SIZE),
            window: window.to_vec(),
            mem: vec![0.0; FFT_SIZE - HOP_SIZE],
            buf: vec![Complex32::default(); FFT_SIZE],
        }
    }

    /// Reconstruit `HOP_SIZE` échantillons depuis le demi-spectre `spec`.
    pub(crate) fn process(&mut self, spec: &[Complex32], out: &mut [f32]) {
        // Spectre hermitien : la partie imaginaire du DC et de Nyquist ne change
        // pas la partie réelle de la sortie, comme le `c2r` de realfft.
        self.buf[..FREQ_BINS].copy_from_slice(spec);
        for (k, &c) in spec
            .iter()
            .enumerate()
            .take(FFT_SIZE - FREQ_BINS + 1)
            .skip(1)
        {
            self.buf[FFT_SIZE - k] = c.conj();
        }
        self.fft.process(&mut self.buf);
        for (i, o) in out.iter_mut().enumerate() {
            *o = self.buf[i].re * self.window[i] + self.mem[i];
        }
        for i in 0..FFT_SIZE - HOP_SIZE {
            self.mem[i] = self.buf[HOP_SIZE + i].re * self.window[HOP_SIZE + i];
        }
    }
}

/// Normalisations exponentielles des features d'entrée (`band_mean_norm_erb`, `band_unit_norm`).
pub(crate) struct FeatureState {
    widths: Vec<usize>,
    mean: Vec<f32>,
    unit: Vec<f32>,
}

impl FeatureState {
    pub(crate) fn new(widths: Vec<usize>) -> Self {
        let lin = |a: f32, b: f32, n: usize| {
            (0..n)
                .map(|i| a + i as f32 * (b - a) / (n - 1) as f32)
                .collect::<Vec<_>>()
        };
        Self {
            widths,
            mean: lin(-60.0, -90.0, NB_ERB),
            unit: lin(0.001, 0.0001, NB_DF),
        }
    }

    /// Calcule `erb[NB_ERB]` (dB normalisés) et `cplx[2][NB_DF]` (plans réel puis imaginaire).
    pub(crate) fn process(&mut self, spec: &[Complex32], erb: &mut [f32], cplx: &mut [f32]) {
        let mut start = 0;
        for (band, &width) in self.widths.iter().enumerate() {
            let k = 1.0 / width as f32;
            let power: f32 = spec[start..start + width]
                .iter()
                .map(|c| (c.re * c.re + c.im * c.im) * k)
                .sum();
            start += width;
            let db = (power + 1e-10).log10() * 10.0;
            let s = &mut self.mean[band];
            *s = db * (1.0 - NORM_ALPHA) + *s * NORM_ALPHA;
            erb[band] = (db - *s) / 40.0;
        }
        let (re, im) = cplx.split_at_mut(NB_DF);
        for f in 0..NB_DF {
            let s = &mut self.unit[f];
            *s = spec[f].norm() * (1.0 - NORM_ALPHA) + *s * NORM_ALPHA;
            let d = s.sqrt();
            re[f] = spec[f].re / d;
            im[f] = spec[f].im / d;
        }
    }
}
