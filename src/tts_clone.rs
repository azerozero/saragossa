//! Prétraitement clone Base/ICL pour Qwen3-TTS.

use crate::{InferError, Result, Tensor};
use rustfft::{num_complex::Complex, FftPlanner};

const N_FFT: usize = 1024;
const HOP: usize = 256;
const WIN: usize = 1024;
const N_MELS: usize = 128;
const SAMPLE_RATE: f32 = 24_000.0;
const FMIN: f32 = 0.0;
const FMAX: f32 = 12_000.0;
const N_FREQS: usize = N_FFT / 2 + 1;

/// Décode un WAV en PCM f32 mono 24 kHz.
///
/// # Errors
///
/// Renvoie une erreur si le WAV est invalide ou si le format est unsupported.
pub fn load_wav_24k(bytes: &[u8]) -> Result<Vec<f32>> {
    let (mono, sample_rate) = decode_wav_mono(bytes)?;
    Ok(resample(&mono, sample_rate, 24_000))
}

/// Rééchantillonne un signal mono par polyphase rationnel.
#[must_use]
pub fn resample(samples: &[f32], from_hz: u32, to_hz: u32) -> Vec<f32> {
    if from_hz == to_hz || samples.is_empty() {
        return samples.to_vec();
    }
    let g = gcd(from_hz, to_hz);
    let up = (to_hz / g) as usize;
    let down = (from_hz / g) as usize;
    let max_rate = up.max(down);
    let half_len = 10 * max_rate;
    let numtaps = 2 * half_len + 1;
    let cutoff = 1.0 / max_rate as f64;
    let mut h = firwin_lowpass(numtaps, cutoff, 5.0);
    for value in &mut h {
        *value *= up as f64;
    }
    upfirdn(samples, &h, up, down, numtaps)
}

/// Calcule le log-mel speaker `[1, frames, 128]`.
///
/// # Errors
///
/// Renvoie une erreur si la forme de sortie déborde.
pub fn log_mel_24k(samples: &[f32]) -> Result<Tensor> {
    let fb = mel_filterbank();
    let pad = (N_FFT - HOP) / 2;
    let padded = reflect_pad(samples, pad);
    let hann = (0..WIN)
        .map(|n| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * n as f32 / WIN as f32).cos()))
        .collect::<Vec<_>>();
    let n_frames = if padded.len() >= N_FFT {
        1 + (padded.len() - N_FFT) / HOP
    } else {
        0
    };

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N_FFT);
    let mut buf = vec![Complex::new(0.0, 0.0); N_FFT];
    let mut mel_out = vec![0.0_f32; n_frames * N_MELS];
    for frame in 0..n_frames {
        let start = frame * HOP;
        for i in 0..N_FFT {
            let sample = padded.get(start + i).copied().unwrap_or(0.0);
            buf[i] = Complex::new(sample * hann[i], 0.0);
        }
        fft.process(&mut buf);
        for mel in 0..N_MELS {
            let row = &fb[mel * N_FREQS..(mel + 1) * N_FREQS];
            let mut acc = 0.0_f32;
            for freq in 0..N_FREQS {
                let c = buf[freq];
                let mag = (c.re * c.re + c.im * c.im + 1.0e-9).sqrt();
                acc += row[freq] * mag;
            }
            mel_out[frame * N_MELS + mel] = acc.max(1.0e-5).ln();
        }
    }
    Tensor::from_vec(vec![1, n_frames, N_MELS], mel_out)
}

fn decode_wav_mono(bytes: &[u8]) -> Result<(Vec<f32>, u32)> {
    let mut reader = hound::WavReader::new(std::io::Cursor::new(bytes))
        .map_err(|err| InferError::Config(format!("parse WAV clone: {err}")))?;
    let spec = reader.spec();
    let channels = spec.channels.max(1) as usize;
    let interleaved = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|err| InferError::Config(format!("lecture WAV f32 clone: {err}")))?,
        hound::SampleFormat::Int => {
            if spec.bits_per_sample == 0 || spec.bits_per_sample > 32 {
                return Err(InferError::Config(format!(
                    "bits WAV clone unsupported: {}",
                    spec.bits_per_sample
                )));
            }
            let max = (1_i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|sample| sample.map(|value| value as f32 / max))
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|err| InferError::Config(format!("lecture WAV int clone: {err}")))?
        }
    };
    let mono = if channels == 1 {
        interleaved
    } else {
        interleaved
            .chunks_exact(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect()
    };
    Ok((mono, spec.sample_rate))
}

fn upfirdn(samples: &[f32], h: &[f64], up: usize, down: usize, numtaps: usize) -> Vec<f32> {
    let n_in = samples.len();
    let n_out = n_in.checked_mul(up).map(|v| v.div_ceil(down)).unwrap_or(0);
    let pad = (numtaps - 1) / 2;
    let mut out = Vec::with_capacity(n_out);
    for i in 0..n_out {
        let center = i * down + pad;
        let kstart = center % up;
        let mut k = kstart;
        let mut acc = 0.0_f64;
        while k < numtaps && k <= center {
            let j = center - k;
            let sample_idx = j / up;
            if sample_idx < n_in {
                acc += h[k] * samples[sample_idx] as f64;
            }
            k += up;
        }
        out.push(acc as f32);
    }
    out
}

fn firwin_lowpass(numtaps: usize, cutoff: f64, beta: f64) -> Vec<f64> {
    let win = kaiser_window(numtaps, beta);
    let center = (numtaps - 1) as f64 / 2.0;
    let mut h = (0..numtaps)
        .map(|i| {
            let m = i as f64 - center;
            let arg = cutoff * m;
            let sinc = if arg == 0.0 {
                1.0
            } else {
                (std::f64::consts::PI * arg).sin() / (std::f64::consts::PI * arg)
            };
            cutoff * sinc * win[i]
        })
        .collect::<Vec<_>>();
    let sum = h.iter().sum::<f64>();
    if sum.abs() > 1.0e-12 {
        for value in &mut h {
            *value /= sum;
        }
    }
    h
}

fn kaiser_window(n: usize, beta: f64) -> Vec<f64> {
    if n == 1 {
        return vec![1.0];
    }
    let denom = bessel_i0(beta);
    let nm1 = (n - 1) as f64;
    (0..n)
        .map(|i| {
            let r = 2.0 * i as f64 / nm1 - 1.0;
            bessel_i0(beta * (1.0 - r * r).max(0.0).sqrt()) / denom
        })
        .collect()
}

fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0_f64;
    let mut term = 1.0_f64;
    let half_x = x / 2.0;
    for k in 1..50 {
        term *= (half_x / k as f64) * (half_x / k as f64);
        sum += term;
        if term < 1.0e-12 * sum {
            break;
        }
    }
    sum
}

fn reflect_pad(samples: &[f32], pad: usize) -> Vec<f32> {
    if pad == 0 || samples.len() < pad + 1 {
        return samples.to_vec();
    }
    let mut out = Vec::with_capacity(samples.len() + 2 * pad);
    for i in (1..=pad).rev() {
        out.push(samples[i]);
    }
    out.extend_from_slice(samples);
    let n = samples.len();
    for i in (n - pad - 1..n - 1).rev() {
        out.push(samples[i]);
    }
    out
}

fn mel_filterbank() -> Vec<f32> {
    let fft_freqs = (0..N_FREQS)
        .map(|i| i as f32 * SAMPLE_RATE / N_FFT as f32)
        .collect::<Vec<_>>();
    let mel_min = hz_to_mel(FMIN);
    let mel_max = hz_to_mel(FMAX);
    let mel_points = (0..N_MELS + 2)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (N_MELS + 1) as f32)
        .collect::<Vec<_>>();
    let hz_points = mel_points
        .iter()
        .map(|mel| mel_to_hz(*mel))
        .collect::<Vec<_>>();

    let mut fb = vec![0.0_f32; N_MELS * N_FREQS];
    for mel in 0..N_MELS {
        let left = hz_points[mel];
        let center = hz_points[mel + 1];
        let right = hz_points[mel + 2];
        for (freq_idx, freq) in fft_freqs.iter().copied().enumerate() {
            let lower = (freq - left) / (center - left);
            let upper = (right - freq) / (right - center);
            fb[mel * N_FREQS + freq_idx] = lower.min(upper).max(0.0);
        }
        let enorm = 2.0 / (right - left);
        for freq_idx in 0..N_FREQS {
            fb[mel * N_FREQS + freq_idx] *= enorm;
        }
    }
    fb
}

fn hz_to_mel(freq: f32) -> f32 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0_f32;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = 6.4_f32.ln() / 27.0;
    if freq >= min_log_hz {
        min_log_mel + (freq / min_log_hz).ln() / logstep
    } else {
        freq / f_sp
    }
}

fn mel_to_hz(mel: f32) -> f32 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0_f32;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = 6.4_f32.ln() / 27.0;
    if mel >= min_log_mel {
        min_log_hz * ((mel - min_log_mel) * logstep).exp()
    } else {
        f_sp * mel
    }
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_preserves_same_rate() {
        let samples = vec![0.1, -0.2, 0.3, 0.4];
        assert_eq!(resample(&samples, 24_000, 24_000), samples);
    }

    #[test]
    fn resample_32k_to_24k_has_expected_length() {
        let samples = vec![0.0_f32; 32_000];
        let out = resample(&samples, 32_000, 24_000);
        assert_eq!(out.len(), (samples.len() * 3).div_ceil(4));
    }

    #[test]
    fn mel_empty_input_has_zero_frames() -> Result<()> {
        let mel = log_mel_24k(&[])?;
        assert_eq!(mel.shape(), &[1, 0, 128]);
        Ok(())
    }

    /// Pré-traitement clone metal-rs ≡ golden mlx-rs figé (sans mlx-rs, sans modèle :
    /// décode/mel pur sur reti-fr.wav). Mêmes tolérances que `live_clone_preprocess_*`.
    #[test]
    fn golden_clone_preprocess_matches_fixture() -> Result<()> {
        let wav =
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../voices/reti-fr.wav");
        // NOTE: la fixture voix vit dans le repo reti, pas dans la crate — le
        // golden reste actif chez reti et se saute proprement en standalone.
        if !wav.exists() {
            return Ok(());
        }
        let bytes = std::fs::read(&wav).map_err(|source| InferError::Io {
            path: wav.clone(),
            source,
        })?;

        let rust_pcm = load_wav_24k(&bytes)?;
        let (_, golden_pcm) = crate::golden::read_f32("clone_pcm24k")?;
        assert_eq!(rust_pcm.len(), golden_pcm.len());
        let pcm_max_abs = max_abs(&rust_pcm, &golden_pcm);

        let rust_mel = log_mel_24k(&rust_pcm)?;
        let (_, golden_mel) = crate::golden::read_f32("clone_mel24k")?;
        assert_eq!(rust_mel.data().len(), golden_mel.len());
        let mel_max_abs = max_abs(rust_mel.data(), &golden_mel);
        assert!(pcm_max_abs <= 1.0e-7, "pcm_max_abs={pcm_max_abs}");
        assert!(mel_max_abs <= 1.0e-6, "mel_max_abs={mel_max_abs}");
        Ok(())
    }

    fn max_abs(left: &[f32], right: &[f32]) -> f32 {
        left.iter()
            .zip(right.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max)
    }
}
