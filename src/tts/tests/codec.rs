use super::*;
use crate::tts::payload::{fnv1a64_update, FNV1A64_OFFSET_BASIS};

/// Parité audio codec **GPU vs CPU** sur codes synthétiques (160 + 256 frames).
///
/// Gate = tolérance audio (max_abs ≤ 0,50, rms ≤ 0,10), PAS l'octet-à-octet :
/// la réduction GPU diffère du scalaire CPU au niveau de l'arrondi f32. Écrit
/// la dérive mesurée par échantillon dans `/tmp/tts_codec_drift.md`.
#[test]
#[ignore = "parité: codec GPU vs CPU (cache HF + Metal requis)"]
fn codec_gpu_cpu_parity_synthetic() -> Result<()> {
    let Some(codec) = load_voicedesign_codec() else {
        eprintln!("skip: snapshot Qwen3-TTS VoiceDesign absent du cache HF");
        return Ok(());
    };
    if crate::test_support::require_real_model(
        codec.gpu_active().then_some(()),
        "codec GPU actif (Metal et RETI_TTS_CODEC_GPU)",
    )
    .is_none()
    {
        eprintln!("skip: forward GPU codec inactif (Metal absent ou RETI_TTS_CODEC_GPU=0)");
        return Ok(());
    }
    let mut report = String::from("# Parité codec GPU vs CPU — codes synthétiques\n\n");
    for &n in &[160_usize, 256] {
        let codes = synthetic_codes(n, 16);
        let gpu = codec.decode_codes(&codes)?;
        let cpu = codec.decode_codes_cpu(&codes)?;
        assert_eq!(gpu.len(), cpu.len(), "longueur PCM GPU != CPU (N={n})");
        let (max_abs, rms, mean_abs, signal_max) = drift_stats(&cpu, &gpu);
        report.push_str(&format!(
            "## N = {n} frames ({} échantillons)\n\
- signal_max_abs (CPU): {signal_max:.6}\n\
- **max_abs_diff**: {max_abs:.3e}\n- **rms_diff**: {rms:.3e}\n- mean_abs_diff: {mean_abs:.3e}\n\
- fnv_cpu: {:#018x}\n- fnv_gpu: {:#018x}\n\n",
            gpu.len(),
            fnv1a_f32(&cpu),
            fnv1a_f32(&gpu),
        ));
        eprintln!(
            "codec parité N={n}: max_abs={max_abs:.3e} rms={rms:.3e} mean_abs={mean_abs:.3e}"
        );
        assert!(max_abs <= 0.5, "max_abs_diff {max_abs} > 0.5 (N={n})");
        assert!(rms <= 0.1, "rms_diff {rms} > 0.1 (N={n})");
    }
    std::fs::write("/tmp/tts_codec_drift.md", &report).map_err(|source| InferError::Io {
        path: PathBuf::from("/tmp/tts_codec_drift.md"),
        source,
    })?;
    eprintln!("dérive écrite dans /tmp/tts_codec_drift.md");
    Ok(())
}

/// Parité audio codec **e2e VoiceDesign** : codes réels (talker greedy) décodés
/// GPU vs CPU, gate tolérance audio (max_abs ≤ 0,50, rms ≤ 0,10).
#[test]
#[ignore = "parité: codec GPU vs CPU sur codes réels VoiceDesign (cache HF + Metal requis)"]
fn codec_gpu_cpu_parity_e2e() -> Result<()> {
    let Some(model_dir) = local_tts_snapshot(
        "RETI_QWEN3_TTS_VOICEDESIGN_DIR",
        "models--mlx-community--Qwen3-TTS-12Hz-1.7B-VoiceDesign-6bit",
    ) else {
        eprintln!("skip: snapshot Qwen3-TTS VoiceDesign absent du cache HF");
        return Ok(());
    };
    let rust = TtsModel::load_local(model_dir)?;
    if crate::test_support::require_real_model(
        rust.codec.gpu_active().then_some(()),
        "codec GPU actif (Metal et RETI_TTS_CODEC_GPU)",
    )
    .is_none()
    {
        eprintln!("skip: forward GPU codec inactif");
        return Ok(());
    }
    let text = "Bonjour, ceci est un test de parité du décodeur codec sur des codes réels.";
    let codes = rust.generate_codes_greedy(text, 128)?;
    let gpu = rust.codec.decode_codes(&codes)?;
    let cpu = rust.codec.decode_codes_cpu(&codes)?;
    assert_eq!(gpu.len(), cpu.len(), "longueur PCM GPU != CPU e2e");
    let (max_abs, rms, mean_abs, signal_max) = drift_stats(&cpu, &gpu);
    eprintln!(
        "codec parité e2e: frames={} samples={} signal_max={signal_max:.4} max_abs={max_abs:.3e} rms={rms:.3e} mean_abs={mean_abs:.3e}",
        codes.len(),
        gpu.len(),
    );
    assert!(max_abs <= 0.5, "max_abs_diff e2e {max_abs} > 0.5");
    assert!(rms <= 0.1, "rms_diff e2e {rms} > 0.1");
    Ok(())
}

/// Parité codec streaming : suffixes incrémentaux vs préfixe complet.
#[test]
#[ignore = "parité: codec streaming incrémental vs batch (cache HF + Metal requis)"]
fn codec_streaming_incremental_matches_full_prefix_on_representative_reply() -> Result<()> {
    let Some(model_dir) = local_tts_snapshot(
        "RETI_QWEN3_TTS_VOICEDESIGN_DIR",
        "models--mlx-community--Qwen3-TTS-12Hz-1.7B-VoiceDesign-6bit",
    ) else {
        eprintln!("skip: snapshot Qwen3-TTS VoiceDesign absent du cache HF");
        return Ok(());
    };
    let rust = TtsModel::load_local(model_dir)?;
    if crate::test_support::require_real_model(
        rust.codec.gpu_active().then_some(()),
        "codec GPU actif (Metal et RETI_TTS_CODEC_GPU)",
    )
    .is_none()
    {
        eprintln!("skip: forward GPU codec inactif");
        return Ok(());
    }
    let text = "D'accord, je vérifie l'état du projet et je te donne le point utile. La priorité est de garder la réponse courte pour lancer l'audio plus vite.";
    let codes = rust.generate_codes_greedy(text, 160)?;
    let full = rust.codec.decode_codes(&codes)?;
    let mut state = rust.codec.new_stream_state();
    let mut streamed = Vec::new();
    let mut end = 0_usize;
    let mut next = 4_usize;
    while end < codes.len() {
        let target = next.min(codes.len());
        let chunk = rust
            .codec
            .decode_codes_streaming(&mut state, &codes[..target])?;
        streamed.extend_from_slice(&chunk);
        end = target;
        next = next.saturating_mul(2).max(end + 1);
    }

    assert_eq!(streamed.len(), full.len(), "longueur streaming != batch");
    let (max_abs, rms, mean_abs, signal_max) = drift_stats(&full, &streamed);
    let report = format!(
        "# Parité codec streaming incrémental\n\nframes={}\nsamples={}\n\
signal_max_abs={signal_max:.6}\nmax_abs_diff={max_abs:.3e}\nrms_diff={rms:.3e}\n\
mean_abs_diff={mean_abs:.3e}\n",
        codes.len(),
        full.len(),
    );
    std::fs::write("/tmp/tts_codec_streaming_incremental.md", &report).map_err(|source| {
        InferError::Io {
            path: PathBuf::from("/tmp/tts_codec_streaming_incremental.md"),
            source,
        }
    })?;
    eprintln!(
        "codec streaming incrémental: frames={} max_abs={max_abs:.3e} rms={rms:.3e} mean_abs={mean_abs:.3e}",
        codes.len(),
    );
    assert!(max_abs <= 0.5, "max_abs_diff streaming {max_abs} > 0.5");
    assert!(rms <= 0.1, "rms_diff streaming {rms} > 0.1");
    Ok(())
}

/// Parité streaming : `synthesize_greedy_streaming` (préfixe croissant +
/// codec incrémental) doit produire un audio dans la tolérance codec de
/// `synthesize_greedy` (batch).
/// Mesure aussi un proxy de TTFA : temps jusqu'au 1er chunk émis vs synthèse totale.
#[test]
#[ignore = "parité+TTFA: streaming TTS ~= batch (cache HF + Metal requis)"]
fn tts_streaming_matches_batch() -> Result<()> {
    let Some(model_dir) = local_tts_snapshot(
        "RETI_QWEN3_TTS_VOICEDESIGN_DIR",
        "models--mlx-community--Qwen3-TTS-12Hz-1.7B-VoiceDesign-6bit",
    ) else {
        eprintln!("skip: snapshot Qwen3-TTS VoiceDesign absent du cache HF");
        return Ok(());
    };
    let rust = TtsModel::load_local(model_dir)?;
    let text = "Bonjour, ceci est un test de streaming incrémental du décodeur TTS.";
    let max_frames = 128;

    // Batch (oracle) + chrono total.
    let batch_start = Instant::now();
    let batch = rust.synthesize_greedy(text, max_frames)?;
    let batch_ms = batch_start.elapsed().as_secs_f64() * 1e3;

    // Streaming : agrège les chunks + capture le temps jusqu'au 1er chunk (TTFA proxy).
    let stream_start = Instant::now();
    let mut streamed: Vec<f32> = Vec::new();
    let mut first_chunk_ms: Option<f64> = None;
    let mut chunks = 0_usize;
    let out = rust.synthesize_greedy_streaming(text, max_frames, |chunk| {
        if first_chunk_ms.is_none() {
            first_chunk_ms = Some(stream_start.elapsed().as_secs_f64() * 1e3);
        }
        chunks += 1;
        streamed.extend_from_slice(chunk);
        Ok(())
    })?;
    let stream_ms = stream_start.elapsed().as_secs_f64() * 1e3;

    // Parité audio (callback agrégé == out.samples ~= batch).
    assert_eq!(
        streamed.len(),
        batch.samples.len(),
        "longueur streaming != batch"
    );
    assert_eq!(
        out.samples.len(),
        batch.samples.len(),
        "longueur out != batch"
    );
    let (max_abs, rms, mean_abs, signal_max) = drift_stats(&batch.samples, &streamed);
    let (out_max_abs, out_rms, _, _) = drift_stats(&batch.samples, &out.samples);
    assert!(max_abs <= 0.5, "max_abs_diff streaming {max_abs} > 0.5");
    assert!(rms <= 0.1, "rms_diff streaming {rms} > 0.1");
    assert!(out_max_abs <= 0.5, "max_abs_diff out {out_max_abs} > 0.5");
    assert!(out_rms <= 0.1, "rms_diff out {out_rms} > 0.1");
    let ttfa = first_chunk_ms.unwrap_or(stream_ms);
    let report = format!(
        "# TTFA streaming TTS\n\nframes={}\nsamples={}\nchunks={chunks}\n\
ttfa_stream_ms={ttfa:.1}\nttfa_batch_ms={batch_ms:.1}\nstream_total_ms={stream_ms:.1}\n\
speedup_ttfa={:.2}\nsignal_max_abs={signal_max:.6}\nmax_abs_diff={max_abs:.3e}\n\
rms_diff={rms:.3e}\nmean_abs_diff={mean_abs:.3e}\ncodec_tolerance=oui\n",
        batch.codes.len(),
        batch.samples.len(),
        batch_ms / ttfa.max(1.0),
    );
    std::fs::write("/tmp/tts_streaming_ttfa.md", &report).map_err(|source| InferError::Io {
        path: PathBuf::from("/tmp/tts_streaming_ttfa.md"),
        source,
    })?;
    eprintln!(
        "streaming: frames={} chunks={chunks} TTFA={ttfa:.0}ms (batch TTFA={batch_ms:.0}ms, ×{:.1}) total={stream_ms:.0}ms max_abs={max_abs:.3e} rms={rms:.3e}",
        batch.codes.len(),
        batch_ms / ttfa.max(1.0),
    );
    Ok(())
}

#[test]
#[ignore = "perf: profil CPU du codec TTS sur N frames croissant (cache HF requis)"]
fn codec_perf_profile() -> Result<()> {
    let Some(codec) = load_voicedesign_codec() else {
        eprintln!("skip: snapshot Qwen3-TTS VoiceDesign absent du cache HF");
        return Ok(());
    };
    let quantizers = 16_usize;
    let frame_counts: &[usize] = match std::env::var("CODEC_PROFILE_N") {
        Ok(spec) if !spec.is_empty() => Box::leak(
            spec.split(',')
                .filter_map(|s| s.trim().parse::<usize>().ok())
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        ),
        _ => &[2, 16, 64],
    };
    let mut report = String::new();
    report.push_str("# Profil CPU codec TTS (decode_codes)\n\n");
    report.push_str("Temps total et par sous-étape (ms), mesuré via `decode_codes_profiled`.\n\n");
    for &n in frame_counts {
        let codes = synthetic_codes(n, quantizers);
        // Préchauffe le cache pour le plus petit N afin d'éviter le coût
        // d'amorçage (allocateur, pages) dans la mesure publiée.
        let (samples, timings) = codec.decode_codes_profiled(&codes)?;
        let total: std::time::Duration = timings.iter().map(|(_, d)| *d).sum();
        // Agrège par étiquette (les blocs décodeur partagent un label).
        let mut agg: Vec<(&'static str, std::time::Duration)> = Vec::new();
        for (label, dur) in &timings {
            if let Some(slot) = agg.iter_mut().find(|(l, _)| l == label) {
                slot.1 += *dur;
            } else {
                agg.push((label, *dur));
            }
        }
        report.push_str(&format!(
            "## N = {n} frames ({} échantillons PCM, {:.3} s audio)\n\n",
            samples.len(),
            samples.len() as f32 / codec.sample_rate() as f32
        ));
        report.push_str(&format!(
            "- **total**: {:.2} ms\n",
            total.as_secs_f64() * 1e3
        ));
        for (label, dur) in &agg {
            report.push_str(&format!(
                "  - {label}: {:.2} ms ({:.1} %)\n",
                dur.as_secs_f64() * 1e3,
                dur.as_secs_f64() / total.as_secs_f64() * 100.0
            ));
        }
        report.push('\n');
        eprintln!(
            "codec profil N={n}: total={:.1}ms samples={}",
            total.as_secs_f64() * 1e3,
            samples.len()
        );
    }
    std::fs::write("/tmp/codec_perf_profile.md", &report).map_err(|source| InferError::Io {
        path: PathBuf::from("/tmp/codec_perf_profile.md"),
        source,
    })?;
    eprintln!("profil écrit dans /tmp/codec_perf_profile.md");
    Ok(())
}

/// Mesure le rtf de la génération TTS (talker + code_predictor) et du codec.
///
/// Harnais avant/après pour le chantier decode résident : sépare le temps de
/// `generate_codes_greedy` (la cible) du décodage codec→PCM (déjà parallélisé),
/// avec préchauffe et cooldown. Étiquette via `RTF_LABEL` (défaut `baseline`),
/// nombre de répétitions via `RTF_REPEATS` (défaut 3), texte via `RTF_TEXT`.
/// Cible temps réel : rtf < 0.3 (codec 12.5 Hz ⇒ < 24 ms/frame génération).
#[test]
#[ignore = "perf: rtf génération TTS VoiceDesign avant/après (cache HF requis, GPU idle)"]
fn perf_voicedesign_rtf() -> Result<()> {
    const DEFAULT_TEXT: &str = "Bonjour, ceci est un test de synthèse vocale pour \
mesurer le débit en temps réel du décodeur. Nous générons plusieurs phrases afin \
d'obtenir un nombre de frames représentatif et une mesure stable.";
    let text = std::env::var("RTF_TEXT").unwrap_or_else(|_| DEFAULT_TEXT.to_string());
    let label = std::env::var("RTF_LABEL").unwrap_or_else(|_| "baseline".to_string());
    let repeats = std::env::var("RTF_REPEATS")
        .ok()
        .and_then(|spec| spec.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(3);
    let max_frames = std::env::var("RTF_MAX_FRAMES")
        .ok()
        .and_then(|spec| spec.parse::<usize>().ok())
        .unwrap_or(400);

    let Some(model_dir) = local_tts_snapshot(
        "RETI_QWEN3_TTS_VOICEDESIGN_DIR",
        "models--mlx-community--Qwen3-TTS-12Hz-1.7B-VoiceDesign-6bit",
    ) else {
        eprintln!("skip: snapshot Qwen3-TTS VoiceDesign absent du cache HF");
        return Ok(());
    };
    let rust = TtsModel::load_local(model_dir)?;
    let sample_rate = rust.codec.sample_rate() as f64;

    // Préchauffe (allocateur, pages, pipelines Metal) hors mesure.
    let _ = rust.synthesize_greedy(&text, 8)?;

    let mut gen_ms = Vec::with_capacity(repeats);
    let mut codec_ms = Vec::with_capacity(repeats);
    let mut frames = 0_usize;
    let mut audio_s = 0.0_f64;
    for run in 0..repeats {
        let gen_start = Instant::now();
        let codes = rust.generate_codes_greedy(&text, max_frames)?;
        let gen = gen_start.elapsed();
        let codec_start = Instant::now();
        let samples = rust.decode_codes_for_mode(&codes)?;
        let codec = codec_start.elapsed();
        frames = codes.len();
        audio_s = samples.len() as f64 / sample_rate;
        gen_ms.push(gen.as_secs_f64() * 1e3);
        codec_ms.push(codec.as_secs_f64() * 1e3);
        eprintln!(
            "rtf[{label}] run={run} frames={frames} audio={audio_s:.3}s gen={:.1}ms codec={:.1}ms",
            gen.as_secs_f64() * 1e3,
            codec.as_secs_f64() * 1e3
        );
        // Cooldown GPU entre les runs (sauf le dernier).
        if run + 1 < repeats {
            std::thread::sleep(std::time::Duration::from_secs(2));
        }
    }
    let median = |values: &[f64]| -> f64 {
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        sorted[sorted.len() / 2]
    };
    let gen_med = median(&gen_ms);
    let codec_med = median(&codec_ms);
    let gen_min = gen_ms.iter().copied().fold(f64::INFINITY, f64::min);
    let ms_per_frame = if frames > 0 {
        gen_med / frames as f64
    } else {
        0.0
    };
    let rtf_gen = if audio_s > 0.0 {
        gen_med / 1e3 / audio_s
    } else {
        0.0
    };
    let rtf_total = if audio_s > 0.0 {
        (gen_med + codec_med) / 1e3 / audio_s
    } else {
        0.0
    };
    let report = format!(
        "# rtf TTS génération — {label}\n\n\
text={text:?}\nframes={frames}\naudio_s={audio_s:.3}\nrepeats={repeats}\n\
gen_ms_median={gen_med:.2}\ngen_ms_min={gen_min:.2}\ncodec_ms_median={codec_med:.2}\n\
ms_per_frame_gen={ms_per_frame:.3}\nrtf_gen={rtf_gen:.4}\nrtf_total={rtf_total:.4}\n\
cible_rtf=0.3000\ncible_ms_per_frame=24.000\n"
    );
    let path = format!("/tmp/tts_rtf_{label}.md");
    std::fs::write(&path, &report).map_err(|source| InferError::Io {
        path: PathBuf::from(&path),
        source,
    })?;
    eprintln!("rtf[{label}] => gen_median={gen_med:.1}ms ms/frame={ms_per_frame:.2} rtf_gen={rtf_gen:.4} rtf_total={rtf_total:.4} (écrit {path})");
    Ok(())
}

/// Empreinte audio de référence du codec (160 frames synthétiques).
///
/// Capturée sur le code scalaire d'origine (commit base `3aa5112`) AVANT
/// l'optimisation. Toute évolution du codec doit la préserver à l'octet près
/// (ou justifier une nouvelle baseline). Voir `codec_emit_golden` pour
/// recalculer la valeur.
const CODEC_GOLDEN_HASH: u64 = 0xcc45_71e4_1a09_84c1;
const CODEC_GOLDEN_LEN: usize = 307_200;

#[test]
#[ignore = "parité: sortie codec CPU byte-identique vs baseline scalaire (cache HF requis)"]
fn codec_parity_golden() -> Result<()> {
    let Some(codec) = load_voicedesign_codec() else {
        eprintln!("skip: snapshot Qwen3-TTS VoiceDesign absent du cache HF");
        return Ok(());
    };
    let codes = synthetic_codes(160, 16);
    // Chemin CPU explicitement : il reste l'oracle byte-identique. Le forward
    // GPU (`decode_codes`) ne diffère qu'à l'arrondi f32 (cf.
    // `codec_gpu_cpu_parity_*`, gate tolérance audio).
    let samples = codec.decode_codes_cpu(&codes)?;
    assert_eq!(
        samples.len(),
        CODEC_GOLDEN_LEN,
        "longueur PCM codec changée"
    );
    assert_eq!(
        fnv1a_f32(&samples),
        CODEC_GOLDEN_HASH,
        "sortie PCM codec non byte-identique vs baseline scalaire"
    );
    Ok(())
}

#[test]
#[ignore = "perf: capture l'empreinte audio de référence (cache HF requis)"]
fn codec_emit_golden() -> Result<()> {
    let Some(codec) = load_voicedesign_codec() else {
        eprintln!("skip: snapshot Qwen3-TTS VoiceDesign absent du cache HF");
        return Ok(());
    };
    let codes = synthetic_codes(160, 16);
    let start = std::time::Instant::now();
    let samples = codec.decode_codes(&codes)?;
    let elapsed = start.elapsed();
    let hash = fnv1a_f32(&samples);
    let max_abs = samples.iter().copied().fold(0.0_f32, |m, s| m.max(s.abs()));
    let meta = format!(
        "frames=160\nelapsed_ms={:.2}\nlen={}\nhash={hash:#018x}\nmax_abs={max_abs:.8}\nfirst={:?}\nlast={:?}\n",
        elapsed.as_secs_f64() * 1e3,
        samples.len(),
        &samples[..samples.len().min(4)],
        &samples[samples.len().saturating_sub(4)..],
    );
    eprint!("{meta}");
    std::fs::write("/tmp/codec_golden_meta.txt", &meta).map_err(|source| InferError::Io {
        path: PathBuf::from("/tmp/codec_golden_meta.txt"),
        source,
    })?;
    Ok(())
}

/// Vecteurs de référence officiels FNV-1a 64 bits (Fowler/Noll/Vo :
/// offset basis `0xcbf29ce484222325`, prime `0x100000001b3`) : la chaîne
/// vide reste l'offset basis (aucun octet traité) et `"a"` = premier
/// octet non trivial (`0x61`). Recalculés indépendamment (Python, à
/// partir de la seule définition `hash = (hash XOR octet) * prime`) avant
/// d'être codés en dur, pour ne pas piéger le test avec le code qu'il vérifie.
#[test]
fn fnv1a64_update_matches_official_reference_vectors() {
    assert_eq!(FNV1A64_OFFSET_BASIS, 0xcbf29ce484222325);

    let empty = FNV1A64_OFFSET_BASIS;
    assert_eq!(empty, 0xcbf29ce484222325);

    let a = fnv1a64_update(FNV1A64_OFFSET_BASIS, b'a');
    assert_eq!(a, 0xaf63dc4c8601ec8c);

    let foobar = b"foobar".iter().fold(FNV1A64_OFFSET_BASIS, |hash, &byte| {
        fnv1a64_update(hash, byte)
    });
    assert_eq!(foobar, 0x85944171f73967e8);

    let abc = b"abc".iter().fold(FNV1A64_OFFSET_BASIS, |hash, &byte| {
        fnv1a64_update(hash, byte)
    });
    assert_eq!(abc, 0xe71fa2190541574b);
}

/// Le checksum de `read_payload_summary` hache les octets bruts LE du
/// tenseur (pas ses u32/f32 en tant que mots), donc le vecteur attendu se
/// recalcule à la main sur les 4 octets `1.0_f32.to_le_bytes()` écrits
/// par `write_safetensors`.
#[test]
fn payload_summary_checksum_hashes_raw_tensor_bytes_fnv1a64() -> Result<()> {
    let tmp = tempfile::tempdir().map_err(|source| InferError::Io {
        path: PathBuf::from("tempdir"),
        source,
    })?;
    let path = tmp.path().join("codec.safetensors");
    write_safetensors(&path, &["only"])?;

    let payload = SafetensorPayload::open(&path)?;
    let summary = payload.read_payload_summary()?;

    let expected = 1.0_f32
        .to_le_bytes()
        .iter()
        .fold(FNV1A64_OFFSET_BASIS, |hash, &byte| {
            fnv1a64_update(hash, byte)
        });
    assert_eq!(summary.bytes, 4);
    assert_eq!(summary.bytes_read, 4);
    assert_eq!(summary.checksum, expected);
    assert_eq!(summary.checksum, 0x4b72477f9c5c2f98);
    Ok(())
}
