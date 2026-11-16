//! Rééchantillonnage polyphase commun aux voix TTS et aux fixtures STT.

use crate::{InferError, Result};

/// Rééchantillonne le PCM avec un filtre passe-bas polyphase.
///
/// # Errors
///
/// Refuse une fréquence hors de la plage WAV ou une taille de sortie qui déborde.
pub fn resample(samples: &[f32], from_hz: u32, to_hz: u32) -> Result<Vec<f32>> {
    crate::wire_audio::validate_wav_limits(from_hz, 1, 0).map_err(InferError::Audio)?;
    crate::wire_audio::validate_wav_limits(to_hz, 1, 0).map_err(InferError::Audio)?;
    if from_hz == to_hz || samples.is_empty() {
        return Ok(samples.to_vec());
    }
    let g = gcd(from_hz, to_hz);
    let up = (to_hz / g) as usize;
    let down = (from_hz / g) as usize;
    let n_out = samples
        .len()
        .checked_mul(up)
        .map(|n| n.div_ceil(down))
        .filter(|&n| n <= isize::MAX as usize / std::mem::size_of::<f32>())
        .ok_or_else(|| InferError::Dimension("taille audio rééchantillonnée trop grande".into()))?;
    let max_rate = up.max(down);
    let half_len = 10 * max_rate;
    let numtaps = 2 * half_len + 1;
    let cutoff = 1.0 / max_rate as f64;
    let mut h = firwin_lowpass(numtaps, cutoff, 5.0);
    for value in &mut h {
        *value *= up as f64;
    }
    Ok(upfirdn(samples, &h, up, down, numtaps, n_out))
}

fn upfirdn(
    samples: &[f32],
    h: &[f64],
    up: usize,
    down: usize,
    numtaps: usize,
    n_out: usize,
) -> Vec<f32> {
    let n_in = samples.len();
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

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}
