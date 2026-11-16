//! Filtre polyphase du batch TTS, avec historique et phase bornés.
use crate::{InferError, Result};
use std::collections::VecDeque;

/// Rééchantillonneur mono FIR conservant sa phase entre morceaux.
pub struct StreamResampler {
    taps: Vec<f64>,
    samples: VecDeque<f32>,
    up: usize,
    down: usize,
    center: usize,
    finished: bool,
}

impl StreamResampler {
    /// Construit le même filtre Kaiser que `tts_clone::resample`.
    ///
    /// # Errors
    /// Refuse une fréquence hors de la plage WAV.
    pub fn new(from: u32, to: u32) -> Result<Self> {
        for rate in [from, to] {
            crate::wire_audio::validate_wav_limits(rate, 1, 0).map_err(InferError::Audio)?;
        }
        let gcd = crate::tts_clone::gcd(from, to);
        let (up, down) = ((to / gcd) as usize, (from / gcd) as usize);
        let half = if up == down { 0 } else { 10 * up.max(down) };
        let mut taps =
            crate::tts_clone::firwin_lowpass(2 * half + 1, 1.0 / up.max(down) as f64, 5.0);
        for h in &mut taps {
            *h *= up as f64;
        }
        Ok(Self {
            taps,
            samples: VecDeque::new(),
            up,
            down,
            center: half,
            finished: false,
        })
    }

    /// Rend la sortie disponible, en attendant l'anticipation du FIR.
    ///
    /// # Errors
    /// Refuse un appel après la fin du flux.
    pub fn push(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        if self.finished {
            return Err(InferError::Config("rééchantillonneur clos".into()));
        }
        if self.up == self.down {
            return Ok(input.to_vec());
        }
        let mut out = Vec::new();
        for &sample in input {
            self.samples.push_back(sample);
            while self.center / self.up < self.samples.len() {
                out.push(self.emit());
            }
        }
        Ok(out)
    }

    /// Complète le filtre avec des zéros ; la longueur reste ceil(n × to/from).
    pub fn finish(&mut self) -> Vec<f32> {
        if self.finished {
            return Vec::new();
        }
        self.finished = true;
        let mut out = Vec::new();
        let pad = (self.taps.len() - 1) / 2;
        while self.center < self.samples.len() * self.up + pad {
            out.push(self.emit());
        }
        out
    }

    fn emit(&mut self) -> f32 {
        let mut acc = 0.0;
        for k in (self.center % self.up..self.taps.len().min(self.center + 1)).step_by(self.up) {
            if let Some(sample) = self.samples.get((self.center - k) / self.up) {
                acc += self.taps[k] * f64::from(*sample);
            }
        }
        self.center += self.down;
        let discard = self
            .center
            .saturating_sub(self.taps.len() - 1)
            .div_ceil(self.up)
            .min(self.samples.len());
        self.samples.drain(..discard);
        self.center -= discard * self.up;
        acc as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn polyphase_is_bit_exact_and_bounded() {
        for from in [8000, 16000, 44100, 48000, 96000, 192000] {
            for len in [0, 1, 17, 1501] {
                let input: Vec<_> = (0..len).map(|i| (i as f32 * 0.2).sin()).collect();
                let reference = crate::tts_clone::resample(&input, from, 48000).unwrap();
                let mut stream = StreamResampler::new(from, 48000).unwrap();
                let mut out = Vec::new();
                for chunk in input.chunks(7) {
                    out.extend(stream.push(chunk).unwrap());
                    assert!(stream.samples.len() <= stream.taps.len().div_ceil(stream.up));
                }
                out.extend(stream.finish());
                assert_eq!(
                    out.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    reference.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    "{from}, {len}"
                );
            }
        }
    }
}
