use super::*;
use std::path::PathBuf;

/// Conv1d k = 3, padding 1, écrite naïvement sur `[out, in, k]`.
fn naive_conv3(x: &[f32], rows: usize, cin: usize, w: &[f32], b: &[f32], cout: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; rows * cout];
    for t in 0..rows {
        for o in 0..cout {
            let mut acc = b[o];
            for c in 0..cin {
                for k in 0..3 {
                    let src = t as isize + k as isize - 1;
                    if (0..rows as isize).contains(&src) {
                        acc += x[src as usize * cin + c] * w[(o * cin + c) * 3 + k];
                    }
                }
            }
            out[t * cout + o] = acc;
        }
    }
    out
}

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5
}

#[test]
fn subpixel_conv_im2col_matches_naive_conv1d() -> Result<()> {
    let (rows, cin, cout) = (7, 5, 12);
    let mut seed = 7;
    let x: Vec<f32> = (0..rows * cin).map(|_| lcg(&mut seed)).collect();
    let w: Vec<f32> = (0..cout * cin * 3).map(|_| lcg(&mut seed)).collect();
    let b: Vec<f32> = (0..cout).map(|_| lcg(&mut seed)).collect();
    let linear = Linear::new(
        Tensor::from_vec(vec![cout, cin * 3], w.clone())?,
        Some(Tensor::from_vec(vec![cout], b.clone())?),
    )?;
    let cols = Tensor::from_vec(vec![rows, cin * 3], conv3_im2col(&x, rows, cin))?;
    let got = linear.forward(&cols)?;
    let want = naive_conv3(&x, rows, cin, &w, &b, cout);
    for (g, w) in got.data().iter().zip(&want) {
        assert!((g - w).abs() < 1e-5, "{g} vs {w}");
    }
    Ok(())
}

#[test]
fn feature_stacking_pads_last_group_with_zeros() {
    // 3 trames de 2 bandes, facteur 2 : [[f0 f1], [f2 0]].
    let mel = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    assert_eq!(
        frontend::stack_frames(&mel, 3, 2, 2),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0]
    );
}

#[test]
fn mel_filters_are_slaney_normalized_triangles() {
    let filters = frontend::build_mel_filters(128);
    assert_eq!(filters.len(), 128 * 257);
    for row in filters.chunks_exact(257) {
        assert!(row.iter().all(|w| *w >= 0.0));
        assert!(row.iter().any(|w| *w > 0.0));
    }
}

#[test]
fn rope_matches_reference_rotate_half() -> Result<()> {
    let (rows, heads, head_dim) = (3, 2, 8);
    let mut seed = 3;
    let data: Vec<f32> = (0..rows * heads * head_dim)
        .map(|_| lcg(&mut seed))
        .collect();
    let mut x = Tensor::from_vec(vec![rows, heads * head_dim], data.clone())?;
    rope_in_place(&mut x, heads, &crate::rope::inv_freq(head_dim, 10_000.0))?;
    for pos in 0..rows {
        for h in 0..heads {
            let at = pos * heads * head_dim + h * head_dim;
            let want = crate::rope::reference_rotate_half(
                &data[at..at + head_dim],
                head_dim,
                head_dim,
                10_000.0,
                pos,
            );
            for (g, w) in x.data()[at..at + head_dim].iter().zip(&want) {
                assert!((g - w).abs() < 1e-6);
            }
        }
    }
    Ok(())
}

#[test]
fn segments_follow_threshold_and_sort_by_start() -> Result<()> {
    // 2 locuteurs, 5 trames.
    let p = Tensor::from_vec(
        vec![5, 2],
        vec![0.9, 0.1, 0.9, 0.6, 0.2, 0.6, 0.6, 0.4, 0.6, 0.5],
    )?;
    let segs = speaker_segments(&p, 0.5)?;
    let as_tuples: Vec<_> = segs
        .iter()
        .map(|s| (s.speaker, s.start_frame, s.end_frame))
        .collect();
    assert_eq!(as_tuples, vec![(0, 0, 2), (1, 1, 3), (0, 3, 5)]);
    assert_eq!(
        to_rttm(&segs[..1], "a"),
        "SPEAKER a 1 0.00 0.02 <NA> <NA> speaker_0 <NA> <NA>\n"
    );
    Ok(())
}

fn policy() -> CachePolicy {
    CachePolicy {
        cache_length: 8,
        fifo_length: 2,
        update_period: 3,
        silence_frames: 1,
        score_threshold: 0.25,
        latest_boost: 0.05,
        min_positive: 1,
        strong_boosted: 2,
        weak_boosted: 4,
        num_speakers: 2,
    }
}

fn small_model_fixture() -> Result<(DiarizationConfig, HashMap<String, Tensor>)> {
    let config = DiarizationConfig {
        hidden_size: 8,
        heads: 2,
        layers: 0,
        num_mel_bins: 2,
        subsampling: 2,
        rope_theta: 10_000.0,
        head_hidden: 4,
        num_speakers: 2,
        chunk_length: 4,
        chunk_right_context: 1,
        chunk_left_context: 0,
        cache: policy(),
    };
    let mut tensors = HashMap::new();
    for (name, shape) in [
        ("model.audio_tower.embedder.projection.weight", vec![8, 4]),
        ("model.audio_tower.input_layer_norm.weight", vec![8]),
        ("model.audio_tower.input_layer_norm.bias", vec![8]),
        ("model.audio_tower.layer_norm.weight", vec![8]),
        ("model.audio_tower.layer_norm.bias", vec![8]),
        ("model.proj.weight", vec![4, 8]),
        ("model.upsampler.conv.weight", vec![8, 4, 3]),
        ("model.upsampler.conv.bias", vec![8]),
        ("classifier.dense.weight", vec![4, 4]),
        ("classifier.out_proj.weight", vec![2, 4]),
        ("silence_embeds", vec![8]),
    ] {
        let values = vec![0.0; shape.iter().product()];
        tensors.insert(name.into(), Tensor::from_vec(shape, values)?);
    }
    Ok((config, tensors))
}

#[test]
fn from_tensors_rejects_incompatible_head_projection() -> Result<()> {
    let (config, tensors) = small_model_fixture()?;
    let model = DiarizationModel::from_tensors(config.clone(), tensors.clone())?;
    let hidden = Tensor::from_vec(vec![3, 8], vec![0.0; 24])?;
    assert_eq!(model.head(&hidden, ForwardRuntime::cpu())?.len(), 12);
    for shape in [vec![3, 8], vec![5, 8], vec![4, 7], vec![4, 9]] {
        let mut invalid = tensors.clone();
        let values = vec![0.0; shape.iter().product()];
        invalid.insert("model.proj.weight".into(), Tensor::from_vec(shape, values)?);
        assert!(matches!(
            DiarizationModel::from_tensors(config.clone(), invalid),
            Err(InferError::Shape(message)) if message.contains("model.proj.weight")
        ));
    }
    Ok(())
}

#[test]
fn from_tensors_revalidates_mutated_config() -> Result<()> {
    let (config, tensors) = small_model_fixture()?;
    DiarizationModel::from_tensors(config.clone(), tensors.clone())?;
    for (field, value) in [
        ("heads", 0),
        ("heads", 3), // hidden_size non divisible par heads.
        ("heads", 8), // Dimension par tête impaire, incompatible avec RoPE.
        ("chunk_length", 0),
        ("subsampling", 0),
        ("hidden_size", 0),
        ("num_mel_bins", 0),
        ("head_hidden", 0),
        ("num_speakers", 0),
        ("num_speakers", 3), // Le cache doit rester cohérent avec la tête.
    ] {
        let mut invalid = config.clone();
        match field {
            "heads" => invalid.heads = value,
            "chunk_length" => invalid.chunk_length = value,
            "subsampling" => invalid.subsampling = value,
            "hidden_size" => invalid.hidden_size = value,
            "num_mel_bins" => invalid.num_mel_bins = value,
            "head_hidden" => invalid.head_hidden = value,
            "num_speakers" => invalid.num_speakers = value,
            _ => {
                return Err(InferError::Config(format!(
                    "champ de test inconnu: {field}"
                )))
            }
        }
        assert!(
            matches!(
                DiarizationModel::from_tensors(invalid, tensors.clone()),
                Err(InferError::Config(_))
            ),
            "{field}={value}"
        );
    }
    Ok(())
}

#[test]
fn cache_compression_keeps_length_and_speaker_order() {
    // 12 trames de dimension 1 (valeur = indice) ; locuteur 0 sur 0..6, 1 sur 6..12.
    let input: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let probs: Vec<f32> = (0..12)
        .flat_map(|t| if t < 6 { [0.9, 0.05] } else { [0.05, 0.9] })
        .collect();
    let mut cache = SpeakerCache::new(policy(), 1);
    cache.update(&input, &probs, &[-1.0], 12, 0);
    // FIFO : 2 dernières trames ; cache compressé à 8 trames, 4 créneaux par
    // locuteur dont un de silence (-1) en fin de groupe.
    let prefix = cache.prefix();
    assert_eq!(prefix.len(), 10);
    assert_eq!(&prefix[8..], &[10.0, 11.0]);
    let cached = &prefix[..8];
    let silences = cached.iter().filter(|v| **v < 0.0).count();
    assert_eq!(silences, 2);
    let speech: Vec<f32> = cached.iter().copied().filter(|v| *v >= 0.0).collect();
    assert!(speech.windows(2).all(|w| w[0] < w[1]), "{cached:?}");
}

/// Deux couches aléatoires pour l'encodeur Metal (d = 128, 2 têtes de 64 : Steel d64).
#[cfg(gpu_metal)]
fn metal_test_model(qkv_bias: bool) -> Result<DiarizationModel> {
    let (d, heads, ffn) = (128, 2, 256);
    let mut seed = 11;
    let mut rnd =
        |n: usize, scale: f32| -> Vec<f32> { (0..n).map(|_| lcg(&mut seed) * scale).collect() };
    let mut t = HashMap::new();
    let p = "model.audio_tower";
    let mut put = |k: String, shape: Vec<usize>, v: Vec<f32>| -> Result<()> {
        t.insert(k, Tensor::from_vec(shape, v)?);
        Ok(())
    };
    let ones = |n: usize| vec![1.0_f32; n];
    put(
        format!("{p}.embedder.projection.weight"),
        vec![d, 16],
        rnd(d * 16, 0.5),
    )?;
    for n in ["input_layer_norm", "layer_norm"] {
        put(format!("{p}.{n}.weight"), vec![d], ones(d))?;
        put(format!("{p}.{n}.bias"), vec![d], rnd(d, 0.1))?;
    }
    for i in 0..2 {
        let l = format!("{p}.layers.{i}");
        for n in ["layer_norm1", "layer_norm2"] {
            put(format!("{l}.{n}.weight"), vec![d], ones(d))?;
            put(format!("{l}.{n}.bias"), vec![d], rnd(d, 0.1))?;
        }
        for n in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            put(
                format!("{l}.self_attn.{n}.weight"),
                vec![d, d],
                rnd(d * d, 0.2),
            )?;
        }
        put(format!("{l}.self_attn.o_proj.bias"), vec![d], rnd(d, 0.1))?;
        if qkv_bias {
            for n in ["q_proj", "v_proj"] {
                put(format!("{l}.self_attn.{n}.bias"), vec![d], rnd(d, 0.1))?;
            }
        }
        put(
            format!("{l}.mlp.fc1.weight"),
            vec![ffn, d],
            rnd(ffn * d, 0.2),
        )?;
        put(format!("{l}.mlp.fc1.bias"), vec![ffn], rnd(ffn, 0.1))?;
        put(
            format!("{l}.mlp.fc2.weight"),
            vec![d, ffn],
            rnd(ffn * d, 0.2),
        )?;
        put(format!("{l}.mlp.fc2.bias"), vec![d], rnd(d, 0.1))?;
    }
    let h = 16;
    put("model.proj.weight".into(), vec![h, d], rnd(h * d, 0.2))?;
    put("model.proj.bias".into(), vec![h], rnd(h, 0.1))?;
    put(
        "model.upsampler.conv.weight".into(),
        vec![h * 8, h, 3],
        rnd(h * 8 * h * 3, 0.2),
    )?;
    put(
        "model.upsampler.conv.bias".into(),
        vec![h * 8],
        rnd(h * 8, 0.1),
    )?;
    put(
        "classifier.dense.weight".into(),
        vec![h, h],
        rnd(h * h, 0.3),
    )?;
    put("classifier.dense.bias".into(), vec![h], rnd(h, 0.1))?;
    put(
        "classifier.out_proj.weight".into(),
        vec![2, h],
        rnd(2 * h, 0.5),
    )?;
    put("classifier.out_proj.bias".into(), vec![2], rnd(2, 0.1))?;
    put("silence_embeds".into(), vec![d], rnd(d, 0.1))?;
    let config = DiarizationConfig {
        hidden_size: d,
        heads,
        layers: 2,
        num_mel_bins: 2,
        subsampling: 8,
        rope_theta: 10_000.0,
        head_hidden: h,
        num_speakers: 2,
        chunk_length: 64,
        chunk_right_context: 4,
        chunk_left_context: 0,
        cache: policy(),
    };
    DiarizationModel::from_tensors(config, t)
}

#[cfg(gpu_metal)]
#[test]
fn metal_rope_encoder_matches_cpu_oracle() -> Result<()> {
    let Ok(metal) = crate::MetalExecutor::new() else {
        return Ok(());
    };
    let rows = 37;
    let d = 128;
    let model = metal_test_model(false)?;
    let mut seed = 1;
    let data: Vec<f32> = (0..rows * d).map(|_| lcg(&mut seed) * 2.0).collect();
    let input = Tensor::from_vec(vec![rows, d], data)?;
    let normed = model.input_ln.forward(&input)?;
    let cpu = model.encode_cpu(normed.clone(), rows - 1, ForwardRuntime::cpu())?;
    let enc = model.build_resident(&metal)?;
    let gpu = metal.encode_whisper_encoder_keys(&normed, &enc, rows - 1)?;
    let max = cpu
        .data()
        .iter()
        .zip(gpu.data())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert!(max < 1e-3, "écart Metal/CPU {max}");
    Ok(())
}

/// Résident fusionné contre résident séparé, biais q/v non nuls. La séparation
/// est une copie ; seul l'ordre d'accumulation du GEMM `[3d, d]` peut différer
/// (autre tuile sur N = 3d). 1e-4 est une borne a priori, non mesurée : l'écart
/// observé est journalisé pour la resserrer après le run de nuit.
#[cfg(gpu_metal)]
#[test]
#[ignore = "GPU Metal"]
fn metal_fused_qkv_matches_separate_projections() -> Result<()> {
    let Ok(metal) = crate::MetalExecutor::new() else {
        return Ok(());
    };
    let (d, rows) = (128, 37);
    // Explicite : la variable d'environnement ne doit pas choisir la branche.
    let model = metal_test_model(true)?.with_fused_qkv(false)?;
    let mut seed = 3;
    let data: Vec<f32> = (0..rows * d).map(|_| lcg(&mut seed)).collect();
    let normed = model
        .input_ln
        .forward(&Tensor::from_vec(vec![rows, d], data)?)?;
    let separate = model.build_resident(&metal)?;
    assert!(separate.layers.iter().all(|l| l.qkv.is_none()));
    let want = metal.encode_whisper_encoder_keys(&normed, &separate, rows - 1)?;
    let model = model.with_fused_qkv(true)?;
    let fused = model.build_resident(&metal)?;
    assert!(fused.layers.iter().all(|l| l.qkv.is_some()));
    let got = metal.encode_whisper_encoder_keys(&normed, &fused, rows - 1)?;
    let max = got
        .data()
        .iter()
        .zip(want.data())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    eprintln!("écart max fusionné/séparé : {max:e}");
    assert!(max < 1e-4, "écart fusionné/séparé {max}");
    Ok(())
}

/// Poids réels : `SARAGOSSA_DIAR_MODEL_DIR` (dossier 🤗) et
/// `SARAGOSSA_DIAR_GOLDEN_DIR` (`<nom>.wav`, `<nom>.probs.f32`, `<nom>.rttm`
/// produits par la référence torch f32, et `tolerance.json` figé avant).
#[test]
#[ignore = "poids réels et goldens locaux"]
fn real_weights_match_torch_reference() -> Result<()> {
    let (Some(model_dir), Some(golden)) = (
        crate::runtime_flags::env_var_os("SARAGOSSA_DIAR_MODEL_DIR"),
        crate::runtime_flags::env_var_os("SARAGOSSA_DIAR_GOLDEN_DIR"),
    ) else {
        return Ok(());
    };
    let golden = PathBuf::from(golden);
    let model = DiarizationModel::from_model_dir(PathBuf::from(model_dir))?;
    let read = |p: PathBuf| std::fs::read(&p).map_err(|source| InferError::Io { path: p, source });
    let tolerance: serde_json::Value =
        serde_json::from_slice(&read(golden.join("tolerance.json"))?)
            .map_err(|e| InferError::Config(e.to_string()))?;
    let tol = |k: &str| tolerance["tolerance"][k].as_f64().unwrap_or(0.0) as f32;
    #[cfg(gpu_metal)]
    let metal = crate::MetalExecutor::new()?;
    #[cfg(gpu_metal)]
    let runtime = ForwardRuntime::metal(&metal);
    #[cfg(not(gpu_metal))]
    let runtime = ForwardRuntime::cpu();
    for name in ["fleurs_en_us", "fleurs_fr_fr", "voices8"] {
        let samples =
            crate::wire_audio::wav_to_mono_16k(&read(golden.join(format!("{name}.wav")))?)
                .map_err(|e| InferError::Config(e.to_string()))?;
        let out = model.diarize(&samples, runtime)?;
        let want: Vec<f32> = read(golden.join(format!("{name}.probs.f32")))?
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        assert_eq!(out.probabilities.len(), want.len(), "{name}: trames");
        let diffs: Vec<f32> = out
            .probabilities
            .data()
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .collect();
        let max = diffs.iter().copied().fold(0.0, f32::max);
        let mean = diffs.iter().sum::<f32>() / diffs.len() as f32;
        // Écart max par morceau de 27,2 s : localise une dérive du cache.
        let per_chunk: Vec<String> = diffs
            .chunks(2720 * model.config.num_speakers)
            .map(|c| format!("{:.1e}", c.iter().copied().fold(0.0, f32::max)))
            .collect();
        eprintln!("{name}: max {max:e} mean {mean:e} par morceau {per_chunk:?}");
        assert!(
            max <= tol("max_abs") && mean <= tol("mean_abs"),
            "{name}: hors tolérance"
        );
        let rttm =
            String::from_utf8_lossy(&read(golden.join(format!("{name}.rttm")))?).into_owned();
        if name != "voices8" {
            assert_eq!(to_rttm(&out.segments, name), rttm, "{name}: segments");
            continue;
        }
        // NOTE: sur voices8, la 2e compression du cache départage deux scores
        // distants de 1e-7 (trames 574 et 587) ; la référence torch bascule de
        // même sous un bruit de 2e-6. Qualification ajoutée après la mesure :
        // seules les trames à moins de 0,01 du seuil peuvent changer d'état.
        let flips: Vec<usize> = (0..want.len())
            .filter(|&i| (out.probabilities.data()[i] > 0.5) != (want[i] > 0.5))
            .collect();
        eprintln!("{name}: {} bascule(s) au seuil", flips.len());
        assert!(
            flips.len() <= 2 && flips.iter().all(|&i| (want[i] - 0.5).abs() < 0.01),
            "{name}: bascules hors quasi-égalité {flips:?}"
        );
    }
    Ok(())
}

/// Poids réels, streaming : `DiarizationStream` sur l'audio golden avec plusieurs
/// tailles de push, comparé au batch (segments identiques, même calcul) et à la
/// référence torch (mêmes tolérances que le test batch, RTTM hors `voices8`).
/// Mêmes variables que `real_weights_match_torch_reference`.
#[test]
#[ignore = "poids réels et goldens locaux"]
fn real_weights_streaming_matches_batch_and_torch() -> Result<()> {
    let (Some(model_dir), Some(golden)) = (
        crate::runtime_flags::env_var_os("SARAGOSSA_DIAR_MODEL_DIR"),
        crate::runtime_flags::env_var_os("SARAGOSSA_DIAR_GOLDEN_DIR"),
    ) else {
        return Ok(());
    };
    let golden = PathBuf::from(golden);
    let model = DiarizationModel::from_model_dir(PathBuf::from(model_dir))?;
    let read = |p: PathBuf| std::fs::read(&p).map_err(|source| InferError::Io { path: p, source });
    let tolerance: serde_json::Value =
        serde_json::from_slice(&read(golden.join("tolerance.json"))?)
            .map_err(|e| InferError::Config(e.to_string()))?;
    let tol = |k: &str| tolerance["tolerance"][k].as_f64().unwrap_or(0.0) as f32;
    #[cfg(gpu_metal)]
    let metal = crate::MetalExecutor::new()?;
    #[cfg(gpu_metal)]
    let runtime = ForwardRuntime::metal(&metal);
    #[cfg(not(gpu_metal))]
    let runtime = ForwardRuntime::cpu();
    for name in ["fleurs_en_us", "fleurs_fr_fr", "voices8"] {
        let samples =
            crate::wire_audio::wav_to_mono_16k(&read(golden.join(format!("{name}.wav")))?)
                .map_err(|e| InferError::Config(e.to_string()))?;
        let batch = model.diarize(&samples, runtime)?;
        let want: Vec<f32> = read(golden.join(format!("{name}.probs.f32")))?
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let rttm =
            String::from_utf8_lossy(&read(golden.join(format!("{name}.rttm")))?).into_owned();
        // 10 ms, 100 ms, valeurs non alignées sur les trames, puis tout d'un bloc.
        for push in [160, 1_600, 4_097, 16_000 + 7, 48_000, samples.len()] {
            let mut stream = DiarizationStream::new(&model, runtime)?;
            let (mut probs, mut segs) = (Vec::new(), Vec::new());
            for part in samples.chunks(push) {
                let u = stream.push(part)?;
                probs.extend(u.probabilities);
                segs.extend(u.segments);
            }
            let u = stream.finish()?;
            probs.extend(u.probabilities);
            segs.extend(u.segments);
            segs.sort_by_key(|s| (s.start_frame, s.speaker));
            assert_eq!(probs.len(), want.len(), "{name} push {push}: trames");
            let diffs = |a: &[f32], b: &[f32]| -> (f32, f32) {
                let d: Vec<f32> = a.iter().zip(b).map(|(x, y)| (x - y).abs()).collect();
                (
                    d.iter().copied().fold(0.0, f32::max),
                    d.iter().sum::<f32>() / d.len() as f32,
                )
            };
            let (bmax, _) = diffs(&probs, batch.probabilities.data());
            let (tmax, tmean) = diffs(&probs, &want);
            eprintln!("{name} push {push}: batch max {bmax:e}, torch max {tmax:e} mean {tmean:e}");
            assert!(bmax < 1e-4, "{name} push {push}: écart au batch {bmax}");
            assert_eq!(segs, batch.segments, "{name} push {push}: segments batch");
            assert!(
                tmax <= tol("max_abs") && tmean <= tol("mean_abs"),
                "{name} push {push}: hors tolérance torch"
            );
            if name != "voices8" {
                assert_eq!(to_rttm(&segs, name), rttm, "{name} push {push}: RTTM");
            }
        }
    }
    Ok(())
}

/// Débit sur une heure : `voices8.wav` du dossier golden répété jusqu'à 3600 s.
#[cfg(gpu_metal)]
#[test]
#[ignore = "poids réels, mesure de débit"]
fn real_weights_hour_throughput() -> Result<()> {
    let (Some(model_dir), Some(golden)) = (
        crate::runtime_flags::env_var_os("SARAGOSSA_DIAR_MODEL_DIR"),
        crate::runtime_flags::env_var_os("SARAGOSSA_DIAR_GOLDEN_DIR"),
    ) else {
        return Ok(());
    };
    let path = PathBuf::from(golden).join("voices8.wav");
    let bytes = std::fs::read(&path).map_err(|source| InferError::Io { path, source })?;
    let clip = crate::wire_audio::wav_to_mono_16k(&bytes)
        .map_err(|e| InferError::Config(e.to_string()))?;
    let samples: Vec<f32> = clip.iter().copied().cycle().take(3600 * 16_000).collect();
    let model = DiarizationModel::from_model_dir(PathBuf::from(model_dir))?;
    let metal = crate::MetalExecutor::new()?;
    let runtime = ForwardRuntime::metal(&metal);
    model.diarize(&samples[..30 * 16_000], runtime)?;
    let t = std::time::Instant::now();
    let mel = frontend::log_mel(&samples, model.config.num_mel_bins)?;
    let mel_s = t.elapsed().as_secs_f64();
    drop(mel);
    let t = std::time::Instant::now();
    let out = model.diarize(&samples, runtime)?;
    eprintln!(
        "1 h d'audio : {:.2} s au total (mel CPU {mel_s:.2} s), {} segments",
        t.elapsed().as_secs_f64(),
        out.segments.len()
    );
    Ok(())
}

/// Petit modèle f32 aléatoire (2 couches, chunk de 4 trames, contexte gauche 1).
fn random_model() -> Result<DiarizationModel> {
    let (d, ffn, h, sub, mels) = (16, 32, 4, 2, 4);
    let mut seed = 5;
    let mut t = HashMap::new();
    let mut put = |k: String, shape: Vec<usize>, scale: f32| -> Result<()> {
        let n = shape.iter().product();
        let v = (0..n).map(|_| lcg(&mut seed) * scale).collect();
        t.insert(k, Tensor::from_vec(shape, v)?);
        Ok(())
    };
    let p = "model.audio_tower";
    put(
        format!("{p}.embedder.projection.weight"),
        vec![d, sub * mels],
        0.3,
    )?;
    for n in ["input_layer_norm", "layer_norm"] {
        put(format!("{p}.{n}.weight"), vec![d], 1.0)?;
        put(format!("{p}.{n}.bias"), vec![d], 0.1)?;
    }
    for i in 0..2 {
        let l = format!("{p}.layers.{i}");
        for n in ["layer_norm1", "layer_norm2"] {
            put(format!("{l}.{n}.weight"), vec![d], 1.0)?;
            put(format!("{l}.{n}.bias"), vec![d], 0.1)?;
        }
        for n in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            put(format!("{l}.self_attn.{n}.weight"), vec![d, d], 0.3)?;
        }
        put(format!("{l}.self_attn.q_proj.bias"), vec![d], 0.1)?;
        put(format!("{l}.self_attn.v_proj.bias"), vec![d], 0.1)?;
        put(format!("{l}.mlp.fc1.weight"), vec![ffn, d], 0.3)?;
        put(format!("{l}.mlp.fc2.weight"), vec![d, ffn], 0.3)?;
    }
    put("model.proj.weight".into(), vec![h, d], 0.3)?;
    put(
        "model.upsampler.conv.weight".into(),
        vec![h * sub, h, 3],
        0.5,
    )?;
    put("model.upsampler.conv.bias".into(), vec![h * sub], 0.1)?;
    put("classifier.dense.weight".into(), vec![h, h], 0.8)?;
    put("classifier.out_proj.weight".into(), vec![2, h], 2.0)?;
    put("classifier.out_proj.bias".into(), vec![2], 0.1)?;
    put("silence_embeds".into(), vec![d], 0.1)?;
    let config = DiarizationConfig {
        hidden_size: d,
        heads: 2,
        layers: 2,
        num_mel_bins: mels,
        subsampling: sub,
        rope_theta: 10_000.0,
        head_hidden: h,
        num_speakers: 2,
        chunk_length: 4,
        chunk_right_context: 2,
        chunk_left_context: 1,
        cache: policy(),
    };
    DiarizationModel::from_tensors(config, t)
}

fn noise(n: usize) -> Vec<f32> {
    let mut seed = 99;
    (0..n)
        .map(|i| lcg(&mut seed) * (1.0 + (i / 4000 % 3) as f32))
        .collect()
}

#[test]
fn streaming_matches_batch_for_any_push_size() -> Result<()> {
    let model = random_model()?;
    let rt = ForwardRuntime::cpu();
    for len in [16_000 + 77, 16_000 * 3 + 5, 161] {
        let samples = noise(len);
        let batch = model.diarize(&samples, rt)?;
        let mut want = batch.segments.clone();
        want.sort_by_key(|s| (s.start_frame, s.speaker));
        for push in [1, 333, 4096, len] {
            let mut stream = DiarizationStream::new(&model, rt)?;
            let (mut probs, mut segs) = (Vec::new(), Vec::new());
            for part in samples.chunks(push) {
                let u = stream.push(part)?;
                probs.extend(u.probabilities);
                segs.extend(u.segments);
            }
            let u = stream.finish()?;
            probs.extend(u.probabilities);
            segs.extend(u.segments);
            assert_eq!(
                probs.len(),
                batch.probabilities.len(),
                "len {len} push {push}"
            );
            let max = probs
                .iter()
                .zip(batch.probabilities.data())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            assert!(max < 1e-5, "len {len} push {push}: écart {max}");
            segs.sort_by_key(|s| (s.start_frame, s.speaker));
            assert_eq!(segs, want, "len {len} push {push}");
        }
    }
    Ok(())
}

#[test]
fn streaming_rejects_audio_shorter_than_a_frame() -> Result<()> {
    let model = random_model()?;
    let mut stream = DiarizationStream::new(&model, ForwardRuntime::cpu())?;
    stream.push(&noise(100))?;
    assert!(stream.finish().is_err());
    Ok(())
}

#[test]
fn fused_qkv_matches_separate_projections() -> Result<()> {
    let model = random_model()?;
    let rt = ForwardRuntime::cpu();
    let samples = noise(16_000 * 2);
    let want = model.diarize(&samples, rt)?;
    let fused = model.with_fused_qkv(true)?;
    assert!(fused.layers.iter().all(|l| l.qkv.is_some()));
    let got = fused.diarize(&samples, rt)?;
    let max = got
        .probabilities
        .data()
        .iter()
        .zip(want.probabilities.data())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert!(max < 1e-5, "écart {max}");
    assert_eq!(got.segments, want.segments);
    Ok(())
}
