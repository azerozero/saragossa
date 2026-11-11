use super::{HOP, N_SAMPLES, SAMPLE_RATE};
use std::ops::Range;

/// Partition sans perte ni recouvrement du PCM 16 kHz, limitée à 30 s.
pub(super) fn audio_segments(samples: &[f32]) -> Vec<Range<usize>> {
    let mut segments = Vec::new();
    let mut start = 0;
    while samples.len() - start > N_SAMPLES {
        let limit = start + N_SAMPLES;
        let mut end = limit;
        let mut minimum = f64::INFINITY;
        // Une fenêtre de 20 ms évite de confondre un passage par zéro avec une
        // pause. Chercher seulement en amont garantit la limite de l'encodeur.
        for boundary in ((limit - SAMPLE_RATE)..=limit).step_by(HOP) {
            let window = &samples[boundary - HOP..(boundary + HOP).min(samples.len())];
            let energy =
                window.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>() / window.len() as f64;
            // En cas d'égalité, garder la coupe la plus tardive : le silence
            // uniforme ne doit pas multiplier les segments inutilement.
            if energy <= minimum {
                minimum = energy;
                end = boundary;
            }
        }
        segments.push(start..end);
        start = end;
    }
    segments.push(start..samples.len());
    segments
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_and_exactly_thirty_seconds_stay_whole() {
        for len in [0, 1, N_SAMPLES - 1, N_SAMPLES] {
            let segments = audio_segments(&vec![0.5; len]);
            assert_eq!(segments.len(), 1);
            assert_eq!(segments[0], 0..len);
        }
    }

    #[test]
    fn thirty_seconds_plus_one_keeps_last_sample() {
        assert_eq!(
            audio_segments(&vec![0.5; N_SAMPLES + 1]),
            vec![0..N_SAMPLES, N_SAMPLES..N_SAMPLES + 1]
        );
    }

    #[test]
    fn silence_and_constant_audio_cover_every_sample_once() {
        for amplitude in [0.0, 0.5] {
            let len = 3 * N_SAMPLES + 1;
            let segments = audio_segments(&vec![amplitude; len]);
            assert_eq!(segments.first().map(|r| r.start), Some(0));
            assert_eq!(segments.last().map(|r| r.end), Some(len));
            assert_eq!(segments.len(), 4);
            assert!(segments
                .iter()
                .all(|r| !r.is_empty() && r.len() <= N_SAMPLES));
            assert!(segments.windows(2).all(|pair| pair[0].end == pair[1].start));
        }
    }

    #[test]
    fn cut_follows_local_energy_minimum_even_without_silence() {
        for quiet in [0.0, 0.1] {
            let mut samples = vec![0.5; 2 * N_SAMPLES];
            let pause = N_SAMPLES - SAMPLE_RATE / 2;
            samples[pause - 2 * HOP..pause + 2 * HOP].fill(quiet);
            // Un zéro isolé plus tard ne doit pas gagner contre une vraie pause.
            samples[N_SAMPLES - HOP] = 0.0;
            let segments = audio_segments(&samples);
            assert_eq!(segments[0].end, pause + HOP);
            assert!(segments
                .iter()
                .all(|r| !r.is_empty() && r.len() <= N_SAMPLES));
            assert_eq!(
                segments.iter().map(|r| r.len()).sum::<usize>(),
                samples.len()
            );
            assert!(segments.windows(2).all(|pair| pair[0].end == pair[1].start));
        }
    }
}
