//! Front-end log-mel NeMo de Nemotron-3-Diarization et empilement des trames.
//!
//! Reproduit `NemotronAsrStreamingFeatureExtractor` (transformers 5.22) en mode
//! hors ligne : préaccentuation 0,97, STFT 512 points centrée (zéros), fenêtre
//! de Hann symétrique de 400 points centrée dans la trame, puissance, banc mel
//! Slaney 128 bandes (0-8 kHz, normalisation Slaney de librosa), `ln(x + 2^-24)`,
//! sans normalisation par bande. La trame `floor(L / hop)` produite par le
//! centrage est masquée (mise à zéro), comme dans la référence.

use crate::{InferError, Result};
use rayon::prelude::*;
use rustfft::num_complex::Complex;
use std::sync::{Arc, OnceLock};

pub(super) const SAMPLE_RATE: usize = 16_000;
pub(super) const HOP: usize = 160;
const N_FFT: usize = 512;
const WIN: usize = 400;
const N_FREQS: usize = N_FFT / 2 + 1;
const PREEMPHASIS: f32 = 0.97;
const LOG_GUARD: f32 = 1.0 / 16_777_216.0; // 2^-24

/// Log-mel `[frames_total, n_mels]` (trames majeures) et nombre de trames valides.
pub(super) struct MelFeatures {
    pub data: Vec<f32>,
    pub frames_total: usize,
    pub frames_valid: usize,
}

pub(super) fn log_mel(samples: &[f32], n_mels: usize) -> Result<MelFeatures> {
    if samples.is_empty() {
        return Err(InferError::Dimension("diarisation: audio vide".into()));
    }
    let filters = build_mel_filters(n_mels);
    // Bandes creuses : chaque fréquence touche au plus deux filtres triangulaires.
    let rows: Vec<(usize, &[f32])> = filters
        .chunks_exact(N_FREQS)
        .map(|row| {
            let first = row.iter().position(|w| *w != 0.0).unwrap_or(0);
            let last = row.iter().rposition(|w| *w != 0.0).map_or(first, |i| i + 1);
            (first, &row[first..last])
        })
        .collect();
    let window = window();
    let pad = N_FFT / 2;
    let mut padded = vec![0.0_f32; samples.len() + 2 * pad];
    padded[pad] = samples[0];
    for i in 1..samples.len() {
        padded[pad + i] = samples[i] - PREEMPHASIS * samples[i - 1];
    }
    let frames_total = 1 + samples.len() / HOP;
    let frames_valid = samples.len() / HOP;
    let fft = fft_512();
    let mut data = vec![0.0_f32; frames_total * n_mels];
    data.par_chunks_mut(n_mels)
        .take(frames_valid)
        .enumerate()
        .for_each_init(
            || vec![Complex::new(0.0_f32, 0.0); N_FFT],
            |buf, (frame, row)| {
                let start = frame * HOP;
                for (i, slot) in buf.iter_mut().enumerate() {
                    *slot = Complex::new(padded[start + i] * window[i], 0.0);
                }
                fft.process(buf);
                let mut power = [0.0_f32; N_FREQS];
                for (p, value) in power.iter_mut().zip(buf.iter()) {
                    // La référence prend le module puis le remet au carré.
                    let magnitude = (value.re * value.re + value.im * value.im).sqrt();
                    *p = magnitude * magnitude;
                }
                for (out, (first, weights)) in row.iter_mut().zip(&rows) {
                    let acc: f32 = weights
                        .iter()
                        .zip(&power[*first..])
                        .map(|(w, p)| w * p)
                        .sum();
                    *out = (acc + LOG_GUARD).ln();
                }
            },
        );
    Ok(MelFeatures {
        data,
        frames_total,
        frames_valid,
    })
}

/// Empile `factor` trames consécutives (zéros pour le groupe incomplet final) :
/// `[ceil(frames / factor), factor * n_mels]`, ligne `t` = trames `t*factor..`.
pub(super) fn stack_frames(mel: &[f32], frames: usize, n_mels: usize, factor: usize) -> Vec<f32> {
    let groups = frames.div_ceil(factor);
    let mut out = vec![0.0_f32; groups * factor * n_mels];
    out[..frames * n_mels].copy_from_slice(&mel[..frames * n_mels]);
    out
}

fn window() -> &'static [f32; N_FFT] {
    static WINDOW: OnceLock<[f32; N_FFT]> = OnceLock::new();
    WINDOW.get_or_init(|| {
        // torch.hann_window(400, periodic=False), centrée par torch.stft dans 512.
        let mut w = [0.0_f32; N_FFT];
        let offset = (N_FFT - WIN) / 2;
        for i in 0..WIN {
            let phase = 2.0 * std::f64::consts::PI * i as f64 / (WIN - 1) as f64;
            w[offset + i] = (0.5 - 0.5 * phase.cos()) as f32;
        }
        w
    })
}

fn fft_512() -> &'static Arc<dyn rustfft::Fft<f32>> {
    static FFT: OnceLock<Arc<dyn rustfft::Fft<f32>>> = OnceLock::new();
    FFT.get_or_init(|| rustfft::FftPlanner::<f32>::new().plan_fft_forward(N_FFT))
}

/// Banc mel `librosa.filters.mel(sr=16000, n_fft=512, norm="slaney")` en f32,
/// `[n_mels, 257]`, calculé en f64 avec les deux arrondis f32 de librosa.
pub(super) fn build_mel_filters(n_mels: usize) -> Vec<f32> {
    let hz_to_mel = |f: f64| {
        let f_sp = 200.0 / 3.0;
        let logstep = 6.4_f64.ln() / 27.0;
        if f >= 1000.0 {
            15.0 + (f / 1000.0).ln() / logstep
        } else {
            f / f_sp
        }
    };
    let mel_to_hz = |m: f64| {
        let f_sp = 200.0 / 3.0;
        let logstep = 6.4_f64.ln() / 27.0;
        if m >= 15.0 {
            1000.0 * (logstep * (m - 15.0)).exp()
        } else {
            f_sp * m
        }
    };
    let fmax = SAMPLE_RATE as f64 / 2.0;
    let (mel_min, mel_max) = (hz_to_mel(0.0), hz_to_mel(fmax));
    let points = n_mels + 2;
    let mel_f: Vec<f64> = (0..points)
        .map(|i| mel_to_hz(mel_min + (mel_max - mel_min) * i as f64 / (points - 1) as f64))
        .collect();
    let fft_freqs: Vec<f64> = (0..N_FREQS)
        .map(|i| i as f64 * SAMPLE_RATE as f64 / N_FFT as f64)
        .collect();
    let mut weights = vec![0.0_f32; n_mels * N_FREQS];
    for m in 0..n_mels {
        let enorm = 2.0 / (mel_f[m + 2] - mel_f[m]);
        for (k, &freq) in fft_freqs.iter().enumerate() {
            let lower = -(mel_f[m] - freq) / (mel_f[m + 1] - mel_f[m]);
            let upper = (mel_f[m + 2] - freq) / (mel_f[m + 2] - mel_f[m + 1]);
            let w = lower.min(upper).max(0.0) as f32;
            weights[m * N_FREQS + k] = (f64::from(w) * enorm) as f32;
        }
    }
    weights
}
