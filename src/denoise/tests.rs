use super::dsp::{self, Analysis, Synthesis, FREQ_BINS};
use super::*;

#[test]
fn erb_widths_cover_all_bins_with_minimum_two() {
    let widths = dsp::erb_widths();
    assert_eq!(widths.len(), NB_ERB);
    assert_eq!(widths.iter().sum::<usize>(), FREQ_BINS);
    assert!(widths.iter().all(|&w| w >= 2));
}

#[test]
fn stft_round_trip_is_identity_after_one_hop() {
    let window = dsp::vorbis_window();
    let mut planner = FftPlanner::new();
    let mut analysis = Analysis::new(&mut planner, &window);
    let mut synthesis = Synthesis::new(&mut planner, &window);
    let signal: Vec<f32> = (0..HOP_SIZE * 8)
        .map(|i| ((i * 7919) % 1000) as f32 / 500.0 - 1.0)
        .collect();
    let mut spec = vec![Complex32::default(); FREQ_BINS];
    let mut out = Vec::new();
    let mut hop = [0.0_f32; HOP_SIZE];
    for chunk in signal.chunks_exact(HOP_SIZE) {
        analysis.process(chunk, &mut spec);
        synthesis.process(&spec, &mut hop);
        out.extend_from_slice(&hop);
    }
    // Reconstruction parfaite retardée de N − H : Σ w² = 1 à 50 % de recouvrement.
    let delay = FFT_SIZE - HOP_SIZE;
    let err = signal[..signal.len() - delay]
        .iter()
        .zip(&out[delay..])
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(err < 1e-5, "erreur de reconstruction {err}");
}

/// Oracle contre la référence PyTorch officielle (`tools/voice/dfn3_ref.py`).
///
/// Tolérances figées avant la première comparaison : f32 partout, mais ordres
/// de sommation différents (8 accumulateurs contre BLAS) sur des produits de
/// 3 072 termes et cinq GRU récurrents sur ~300 trames. L'erreur relative
/// attendue est de l'ordre de 1e-5 ; une erreur de disposition des poids donne
/// au mieux −20 dB. Seuils : features ≤ 1e-4 en absolu, masque ≤ 1e-3, sortie
/// audio ≥ 60 dB de SNR contre la référence et ≤ 1e-3 en écart absolu.
#[test]
#[ignore = "requiert SARAGOSSA_DFN3_DIR (dfn3.safetensors + golden.safetensors)"]
fn dfn3_matches_pytorch_reference() {
    let dir = std::path::PathBuf::from(
        std::env::var("SARAGOSSA_DFN3_DIR").expect("invariant: test ignoré sans la variable"),
    );
    let model = DeepFilterNet3::load(dir.join("dfn3.safetensors")).unwrap();
    let golden = crate::safetensor::load_f32_tensors(dir.join("golden.safetensors")).unwrap();
    let audio = golden["audio"].data();
    let want = golden["out"].data();

    // Features : rejoue l'analyse sur le signal paddé comme la référence.
    let mut planner = FftPlanner::new();
    let mut analysis = Analysis::new(&mut planner, &model.window);
    let mut features = dsp::FeatureState::new(model.widths.clone());
    let mut padded = audio.to_vec();
    padded.resize(audio.len() + FFT_SIZE, 0.0);
    let (ref_erb, ref_cplx) = (golden["erb_feat"].data(), golden["spec_feat"].data());
    let mut spec = vec![Complex32::default(); FREQ_BINS];
    let (mut erb, mut cplx) = ([0.0_f32; NB_ERB], [0.0_f32; 2 * NB_DF]);
    let mut feat_err = 0.0_f32;
    for (t, chunk) in padded.chunks_exact(HOP_SIZE).enumerate() {
        analysis.process(chunk, &mut spec);
        features.process(&spec, &mut erb, &mut cplx);
        for b in 0..NB_ERB {
            feat_err = feat_err.max((erb[b] - ref_erb[t * NB_ERB + b]).abs());
        }
        for f in 0..NB_DF {
            let r = &ref_cplx[(t * NB_DF + f) * 2..][..2];
            feat_err = feat_err.max((cplx[f] - r[0]).abs());
            feat_err = feat_err.max((cplx[NB_DF + f] - r[1]).abs());
        }
    }

    // Masque : pas réseau par pas réseau.
    let ref_mask = golden["mask"].data();
    let mut stream = model.stream();
    let mut mask_err = 0.0_f32;
    let mut tau = 0;
    let mut analysis = Analysis::new(&mut planner, &model.window);
    let mut feats = dsp::FeatureState::new(model.widths.clone());
    for (t, chunk) in padded.chunks_exact(HOP_SIZE).enumerate() {
        analysis.process(chunk, &mut spec);
        feats.process(&spec, &mut erb, &mut cplx);
        if t < CONV_LOOKAHEAD {
            continue;
        }
        let step = model.net.step(&mut stream.state, &erb, &cplx);
        for b in 0..NB_ERB {
            mask_err = mask_err.max((step.mask[b] - ref_mask[tau * NB_ERB + b]).abs());
        }
        tau += 1;
    }

    let got = model.enhance(audio).unwrap();
    assert_eq!(got.len(), want.len());
    let (mut num, mut den, mut max_abs) = (0.0_f64, 0.0_f64, 0.0_f32);
    for (g, w) in got.iter().zip(want) {
        num += f64::from(w * w);
        den += f64::from((g - w) * (g - w));
        max_abs = max_abs.max((g - w).abs());
    }
    let snr = 10.0 * (num / den.max(1e-30)).log10();
    eprintln!(
        "dfn3 oracle: feat {feat_err:.2e} mask {mask_err:.2e} out {max_abs:.2e} SNR {snr:.1} dB"
    );
    assert!(feat_err <= 1e-4, "features {feat_err}");
    assert!(mask_err <= 1e-3, "masque {mask_err}");
    assert!(
        max_abs <= 1e-3 && snr >= 60.0,
        "sortie {max_abs} / {snr} dB"
    );
}

/// Latence CPU par hop de 10 ms (p50/p99/max) sur le clip doré, un seul thread.
#[test]
#[ignore = "mesure ; requiert SARAGOSSA_DFN3_DIR"]
fn dfn3_hop_latency() {
    let dir = std::path::PathBuf::from(
        std::env::var("SARAGOSSA_DFN3_DIR").expect("invariant: test ignoré sans la variable"),
    );
    let model = DeepFilterNet3::load(dir.join("dfn3.safetensors")).unwrap();
    let golden = crate::safetensor::load_f32_tensors(dir.join("golden.safetensors")).unwrap();
    let audio = golden["audio"].data();
    let mut times = Vec::new();
    let mut hop = [0.0_f32; HOP_SIZE];
    for _ in 0..3 {
        let mut stream = model.stream();
        for chunk in audio.chunks_exact(HOP_SIZE) {
            let t0 = std::time::Instant::now();
            stream.process_hop(chunk, &mut hop).unwrap();
            times.push(t0.elapsed().as_secs_f64() * 1e6);
        }
    }
    times.sort_by(f64::total_cmp);
    let q = |p: f64| times[((times.len() - 1) as f64 * p) as usize];
    eprintln!(
        "dfn3 CPU hop µs: p50 {:.0} p99 {:.0} max {:.0} (n={}) RTF {:.4}",
        q(0.5),
        q(0.99),
        q(1.0),
        times.len(),
        q(0.5) / 10_000.0
    );
}

mod synthetic;

#[test]
fn enhance_preserves_partial_hops_and_matches_streaming() {
    let model = synthetic::model();
    for len in [
        0,
        1,
        HOP_SIZE - 1,
        HOP_SIZE,
        HOP_SIZE + 1,
        2 * HOP_SIZE - 1,
        2 * HOP_SIZE + 17,
    ] {
        let input: Vec<f32> = (0..len)
            .map(|i| if i % 2 == 0 { 0.5 } else { -0.5 })
            .collect();
        let output = model.enhance(&input).unwrap();
        assert_eq!(output.len(), len);
        assert!(output.iter().all(|x| x.is_finite()));
        // Oracle par flux : conserve le hop partiel et vide explicitement le retard.
        let mut stream = model.stream();
        let mut reference = Vec::new();
        let mut hop = [0.0; HOP_SIZE];
        for chunk in input.chunks(HOP_SIZE) {
            let mut padded = [0.0; HOP_SIZE];
            padded[..chunk.len()].copy_from_slice(chunk);
            stream.process_hop(&padded, &mut hop).unwrap();
            reference.extend_from_slice(&hop);
        }
        for _ in 0..FFT_SIZE / HOP_SIZE {
            stream.process_hop(&[0.0; HOP_SIZE], &mut hop).unwrap();
            reference.extend_from_slice(&hop);
        }
        reference.extend(stream.finish());
        assert_eq!(
            output,
            reference[DfStream::DELAY_SAMPLES..DfStream::DELAY_SAMPLES + len]
        );
        if len > HOP_SIZE {
            assert!(
                output.last().unwrap().abs() > 0.01,
                "fin de signal perdue: {len}"
            );
        }
    }
}

#[test]
fn arbitrary_chunks_are_bit_exact_against_historical_batch() {
    let model = synthetic::recurrent_model();
    for len in [0, 1, 479, 480, 481, 960, 5777] {
        let input: Vec<f32> = (0..len)
            .map(|i| ((i * 7919 % 997) as f32 - 498.0) / 997.0)
            .collect();
        // Oracle indépendant : ancien batch, padding et compensation explicites.
        let mut padded = input.clone();
        padded.resize((len + FFT_SIZE).div_ceil(HOP_SIZE) * HOP_SIZE, 0.0);
        let mut raw = model.stream();
        let mut reference = Vec::new();
        for chunk in padded.chunks(HOP_SIZE) {
            let mut hop = [0.0; HOP_SIZE];
            raw.process_hop(chunk, &mut hop).unwrap();
            reference.extend(hop);
        }
        reference.extend(raw.finish());
        let want: Vec<u32> = reference[DfStream::DELAY_SAMPLES..DfStream::DELAY_SAMPLES + len]
            .iter()
            .map(|s| s.to_bits())
            .collect();
        for (size, threads) in [1, 7, 480, 4800, usize::MAX]
            .into_iter()
            .flat_map(|size| [(size, 1), (size, 3)])
        {
            let mut stream = DenoiseStream::with_threads(&model, threads);
            let mut result = Vec::new();
            let mut pos = 0;
            let mut seed = 42_u32;
            while pos < input.len() {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                let n = if size == usize::MAX {
                    1 + seed as usize % 1753
                } else {
                    size
                };
                let end = (pos + n).min(input.len());
                result.extend(stream.push(&input[pos..end]).unwrap());
                pos = end;
            }
            result.extend(stream.finish().unwrap());
            assert_eq!(
                result.iter().map(|s| s.to_bits()).collect::<Vec<_>>(),
                want,
                "len={len}, chunk={size}, threads={threads}"
            );
        }
    }
}

#[test]
fn enhance_resampling_preserves_validation_without_tts() {
    for (from, to) in [(0, SAMPLE_RATE), (SAMPLE_RATE, 0)] {
        assert!(matches!(resample(&[], from, to), Err(InferError::Audio(_))));
    }
    let signal = vec![0.25; 160];
    assert_eq!(resample(&signal, SAMPLE_RATE, SAMPLE_RATE).unwrap(), signal);
    let converted = resample(&signal, 16_000, SAMPLE_RATE).unwrap();
    assert_eq!(converted.len(), 480);
    assert!(converted.iter().all(|value| value.is_finite()));
}

/// Le mode multi-cœur exact parallélise tout sauf la récurrence des GRU :
/// la sortie doit être identique bit à bit au chemin trame par trame, sur
/// plusieurs blocs et un bloc final partiel.
#[test]
fn multicore_stream_is_bit_exact() {
    let model = synthetic::recurrent_model();
    // 6,5 s : deux blocs entiers de 2,56 s et un reste.
    let input: Vec<f32> = (0..6_u64 * 48_000 + 24_000)
        .map(|i| ((i * 7919 % 997) as f32 - 498.0) / 997.0 * 0.5)
        .collect();
    let bits = |v: Vec<f32>| v.into_iter().map(f32::to_bits).collect::<Vec<_>>();
    let mono = bits(model.enhance(&input).unwrap());
    let cores = std::thread::available_parallelism().map_or(1, usize::from);
    for threads in [1, 2, 4, 8, cores] {
        let mut stream = DenoiseStream::with_threads(&model, threads);
        let mut out = Vec::new();
        for chunk in input.chunks(10_007) {
            out.extend(stream.push(chunk).unwrap());
        }
        out.extend(stream.finish().unwrap());
        assert_eq!(bits(out), mono, "threads={threads}");
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn gpu_enhance_is_bit_exact_against_cpu_common_math() {
    if metal::Device::system_default().is_none() {
        return;
    }
    let model = synthetic::recurrent_model();
    // ~200 pas : traverse trois frontières de morceau GPU (64 pas en test).
    let input: Vec<f32> = (0..200 * HOP_SIZE + 17)
        .map(|i| ((i * 7919 % 997) as f32 - 498.0) / 997.0 * 0.5)
        .collect();
    let cpu = model.enhance_common(&input).unwrap();
    let gpu = model.enhance_gpu(&input).unwrap();
    assert_eq!(cpu.len(), input.len());
    let bits = |v: &[f32]| v.iter().map(|s| s.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&gpu), bits(&cpu));
}

/// Sonde de l'étape 1 : boucle GRU persistante, µs par pas (hidden 256).
#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
#[ignore = "mesure GPU ; lancer sous le verrou GPU"]
fn dfn3_gpu_gru_step_probe() {
    let model = synthetic::recurrent_model();
    for seqs in [1, 2] {
        let us = super::gpu::gru_probe(&model.net, 29_300, seqs).unwrap();
        eprintln!("dfn3 GRU persistant, {seqs} séquence(s) : {us:.2} µs/pas");
    }
}

/// Vitesse et identité sur un clip long réel : CPU historique, CPU commun, GPU.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
#[ignore = "requiert SARAGOSSA_DFN3_DIR et SARAGOSSA_DFN3_LONG_WAV ; GPU sous verrou"]
fn dfn3_gpu_long_file_bit_exact_and_speed() {
    let dir = std::path::PathBuf::from(
        std::env::var("SARAGOSSA_DFN3_DIR").expect("invariant: test ignoré sans la variable"),
    );
    let wav =
        std::env::var("SARAGOSSA_DFN3_LONG_WAV").expect("invariant: test ignoré sans la variable");
    let model = DeepFilterNet3::load(dir.join("dfn3.safetensors")).unwrap();
    let (raw, rate) = decode_wav(&std::fs::read(wav).unwrap()).unwrap();
    let audio = resample(&raw, rate, SAMPLE_RATE).unwrap();
    let time = |f: &dyn Fn() -> Vec<f32>| {
        let t0 = std::time::Instant::now();
        let out = f();
        (out, t0.elapsed().as_secs_f64())
    };
    let (gpu, t_gpu) = time(&|| model.enhance_gpu(&audio).unwrap());
    let (common, t_common) = time(&|| model.enhance_common(&audio).unwrap());
    let (libm, t_libm) = time(&|| model.enhance(&audio).unwrap());
    let same = gpu
        .iter()
        .zip(&common)
        .filter(|(a, b)| a.to_bits() == b.to_bits())
        .count();
    let differ = libm
        .iter()
        .zip(&common)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    let signal: f64 = libm.iter().map(|&s| f64::from(s).powi(2)).sum();
    let error: f64 = libm
        .iter()
        .zip(&common)
        .map(|(&a, &b)| f64::from(a - b).powi(2))
        .sum();
    let max_abs = libm
        .iter()
        .zip(&common)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    eprintln!(
        "dfn3 {:.1} s : GPU {t_gpu:.2} s, CPU commun {t_common:.2} s, CPU libm {t_libm:.2} s ; \
         GPU = commun sur {same}/{} échantillons ; commun vs libm : {differ} échantillons différents, SNR {:.1} dB, max {max_abs:.2e}",
        audio.len() as f64 / f64::from(SAMPLE_RATE),
        gpu.len(),
        10.0 * (signal / error.max(1e-30)).log10()
    );
    assert_eq!(gpu.len(), common.len());
    assert_eq!(same, gpu.len(), "GPU différent du CPU commun");
}
