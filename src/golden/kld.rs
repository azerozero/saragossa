//! KL sur corpus contre mlx-lm float32, rapportée au bruit propre de mlx-lm bf16.
//! Protocole et calibration : tools/goldens/README.md, section « KL sur corpus ».
use super::external::*;
use serde::Deserialize;

#[derive(Deserialize)]
struct Meta {
    model: String,
    chunks: Vec<Vec<usize>>,
    scored: usize,
    top_k: usize,
    mlx_bf16_noise: Stats,
}

#[derive(Debug, Deserialize, Clone, Copy)]
struct Stats {
    positions: usize,
    mean_kl: f64,
    p99_delta_p: f64,
    max_delta_p: f64,
    same_top: f64,
}

/// saragossa bf16 peut s'écarter de la référence float32 jusqu'à 1,5 fois le
/// bruit de mlx-lm bf16 ; au-delà, l'écart vient du moteur et non de la précision.
const MAX_NOISE_RATIO: f64 = 1.5;

struct Reference {
    meta: Meta,
    /// Par position notée : ids top-K puis logprobs top-K float32.
    rows: Vec<(Vec<usize>, Vec<f64>)>,
}

fn load(name: &str) -> TestResult<Reference> {
    let meta: Meta = json(&format!("kld/{name}_meta"))?;
    assert_eq!(meta.model, name);
    let (header, bytes) = read(&format!("kld/{name}"))?;
    assert_eq!(header.dtype, "f32");
    assert_eq!(header.source["float32_control"], true);
    let positions = meta.chunks.len() * meta.scored;
    assert_eq!(header.shape, [positions, 2, meta.top_k]);
    assert_eq!(meta.mlx_bf16_noise.positions, positions);
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let rows = values
        .chunks_exact(2 * meta.top_k)
        .map(|row| {
            let (ids, logprobs) = row.split_at(meta.top_k);
            (
                ids.iter().map(|v| *v as usize).collect(),
                logprobs.iter().map(|v| f64::from(*v)).collect(),
            )
        })
        .collect();
    // Morceaux de largeur variable en régime chat (gemma4) : voir kld.py.
    assert!(meta.chunks.iter().all(|c| c.len() > meta.scored));
    Ok(Reference { meta, rows })
}

/// KL et Δp max sur la partition top-K de référence + masse restante.
fn metrics(logits: &[f32], ids: &[usize], reference: &[f64]) -> (f64, f64) {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let log_z = max
        + logits
            .iter()
            .map(|v| (f64::from(*v) - max).exp())
            .sum::<f64>()
            .ln();
    let p: Vec<f64> = reference.iter().map(|v| v.exp()).collect();
    let q: Vec<f64> = ids
        .iter()
        .map(|id| (f64::from(logits[*id]) - log_z).exp())
        .collect();
    let rest = (
        (1.0 - p.iter().sum::<f64>()).max(0.0),
        (1.0 - q.iter().sum::<f64>()).max(0.0),
    );
    let mut kl = 0.0;
    let mut delta = 0.0_f64;
    for (p, q) in p.into_iter().zip(q).chain([rest]) {
        if p > 0.0 {
            kl += p * (p / q.max(f64::MIN_POSITIVE)).ln();
        }
        delta = delta.max((p - q).abs());
    }
    (kl, delta)
}

fn summarize(kls: &[f64], deltas: &mut [f64], same: usize) -> Stats {
    deltas.sort_by(f64::total_cmp);
    Stats {
        positions: kls.len(),
        mean_kl: kls.iter().sum::<f64>() / kls.len() as f64,
        p99_delta_p: deltas[(0.99 * (deltas.len() - 1) as f64) as usize],
        max_delta_p: deltas[deltas.len() - 1],
        same_top: same as f64 / kls.len() as f64,
    }
}

fn within_noise(engine: Stats, noise: Stats) -> bool {
    engine.mean_kl <= MAX_NOISE_RATIO * noise.mean_kl
        && engine.p99_delta_p <= MAX_NOISE_RATIO * noise.p99_delta_p
        // Un ratio borne mal un bruit déjà élevé ; le top-1 ne doit jamais régresser.
        && engine.same_top >= noise.same_top
}

#[test]
fn external_kld_references_are_consistent() -> TestResult {
    for name in CHAT_MODELS.iter().map(|(name, _)| *name).chain([
        "gemma4_long_1024",
        "gemma4_long_2048",
        "gemma4_long_4100",
        "mistral7b",
    ]) {
        let reference = load(name)?;
        for (ids, logprobs) in &reference.rows {
            assert!(logprobs.windows(2).all(|v| v[0] >= v[1]) && logprobs[0] <= 0.0);
            let unique: std::collections::HashSet<_> = ids.iter().collect();
            assert_eq!(unique.len(), ids.len());
        }
    }
    Ok(())
}

#[test]
fn external_kld_gemma4_long_protocol() -> TestResult {
    for length in [1024, 2048, 4100] {
        let name = format!("gemma4_long_{length}");
        let reference = load(&name)?;
        let (header, _) = read(&format!("kld/{name}"))?;
        let (meta_header, _) = read(&format!("kld/{name}_meta"))?;
        assert_eq!(header.source, meta_header.source);
        assert_eq!(header.source["versions"]["mlx"], "0.32.4");
        assert_eq!(header.source["versions"]["mlx-lm"], "0.32.2");
        assert!(!header.source["weight_sha256"]
            .as_object()
            .ok_or("empreintes des poids longues absentes")?
            .is_empty());
        assert_eq!(reference.meta.scored, 128);
        assert_eq!(reference.meta.top_k, 64);
        assert_eq!(reference.meta.chunks.len(), 1);
        let chunk = &reference.meta.chunks[0];
        assert_eq!(chunk.len(), length + 128);
        for ((ids, _), token) in reference.rows.iter().zip(&chunk[length..]) {
            assert_eq!(ids[0], *token, "la continuation doit être greedy f32");
        }
    }
    Ok(())
}

#[test]
fn external_kld_ratio_gate() {
    let noise = Stats {
        positions: 1,
        mean_kl: 0.002,
        p99_delta_p: 0.06,
        max_delta_p: 0.2,
        same_top: 0.98,
    };
    assert!(within_noise(noise, noise));
    let worse_kl = Stats {
        mean_kl: 0.0031,
        ..noise
    };
    assert!(!within_noise(worse_kl, noise));
    let worse_delta = Stats {
        p99_delta_p: 0.091,
        ..noise
    };
    assert!(!within_noise(worse_delta, noise));
    let worse_top = Stats {
        same_top: 0.97,
        ..noise
    };
    assert!(!within_noise(worse_top, noise));
    let logits = [0.0_f32; 4];
    let (kl, delta) = metrics(&logits, &[0, 1], &[0.25_f64.ln(), 0.25_f64.ln()]);
    assert!(kl.abs() < 1e-12 && delta < 1e-12);
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn compare(name: &str, env: &str) -> TestResult {
    compare_with_prefill(name, env, |model, prompt| model.prefill_cache(prompt))
}

/// Même référence, provenance et seuils que l'oracle externe ; les deux chemins
/// doivent réellement s'exécuter. Une absence de poids ou un refus résident
/// fait échouer cette qualification même sans SARAGOSSA_REQUIRE_REAL_MODELS.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub(crate) fn compare_gemma4_prefill(resident: bool) -> TestResult {
    let env = "SARAGOSSA_GEMMA4_DIR";
    local(env)?.ok_or("SARAGOSSA_GEMMA4_DIR requis pour cet oracle")?;
    let _flag = crate::runtime_flags::override_prefill_resident_gemma4_for_test(resident);
    eprintln!("kld/gemma4: prefill résident={resident}");
    compare_with_prefill("gemma4", env, |model, prompt| {
        let (cache, state) = if resident {
            model
                .prefill_cache_state_metal_resident_for_test(prompt)?
                .ok_or_else(|| {
                    crate::InferError::Config("oracle: prefill résident refusé".into())
                })?
        } else {
            model.prefill_cache_state_batched_for_test(prompt)?
        };
        Ok((cache, model.logits_from_final_state(&state)?))
    })
}

/// Référence indépendante mlx-lm f32 et bruit bf16 calibré sur le même prompt.
/// Le prefill candidat est forcé sans repli ; le decode et les tokens de la
/// référence sont communs aux deux chemins. Les seuils restent ceux du golden.
#[cfg(all(target_os = "macos", feature = "metal"))]
fn compare_gemma4_long_reference(name: &str, resident: bool) -> TestResult {
    let env = "SARAGOSSA_GEMMA4_DIR";
    local(env)?.ok_or("SARAGOSSA_GEMMA4_DIR requis pour cet oracle")?;
    let _flag = crate::runtime_flags::override_prefill_resident_gemma4_for_test(resident);
    eprintln!("kld/{name}: prefill résident={resident}");
    compare_with_prefill(name, env, |model, prompt| {
        crate::MetalExecutor::take_long_d512_dispatches_for_test();
        let (cache, state) = if resident {
            model
                .prefill_cache_state_metal_resident_for_test(prompt)?
                .ok_or_else(|| {
                    crate::InferError::Config("oracle MLX long: résident refusé".into())
                })?
        } else {
            model.prefill_cache_state_batched_for_test(prompt)?
        };
        let trace = crate::MetalExecutor::take_long_d512_dispatches_for_test();
        eprintln!(
            "kld/{name}: prompt={} résident={resident} trace={trace:?}",
            prompt.len()
        );
        assert_eq!(
            trace.len(),
            if resident && prompt.len() > 2048 {
                5
            } else {
                0
            }
        );
        assert!(trace
            .iter()
            .all(|&(seq, dim)| seq == prompt.len() && dim == 512));
        Ok((cache, model.logits_from_final_state(&state)?))
    })
}

#[cfg(all(target_os = "macos", feature = "metal"))]
macro_rules! long_oracle {
    ($test:ident, $name:literal, $resident:literal) => {
        #[test]
        #[ignore = "oracle MLX long q4a8-ls : poids locaux et verrou GPU, borne >= 4100 ; échec top-1 connu à 4100 sur les deux chemins (docs/perf/gemma4-prefill.md)"]
        fn $test() -> TestResult {
            compare_gemma4_long_reference($name, $resident)
        }
    };
}

#[cfg(all(target_os = "macos", feature = "metal"))]
long_oracle!(
    external_kld_gemma4_long_1024_per_op,
    "gemma4_long_1024",
    false
);
#[cfg(all(target_os = "macos", feature = "metal"))]
long_oracle!(
    external_kld_gemma4_long_1024_resident,
    "gemma4_long_1024",
    true
);
#[cfg(all(target_os = "macos", feature = "metal"))]
long_oracle!(
    external_kld_gemma4_long_2048_per_op,
    "gemma4_long_2048",
    false
);
#[cfg(all(target_os = "macos", feature = "metal"))]
long_oracle!(
    external_kld_gemma4_long_2048_resident,
    "gemma4_long_2048",
    true
);
#[cfg(all(target_os = "macos", feature = "metal"))]
long_oracle!(
    external_kld_gemma4_long_4100_per_op,
    "gemma4_long_4100",
    false
);
#[cfg(all(target_os = "macos", feature = "metal"))]
long_oracle!(
    external_kld_gemma4_long_4100_resident,
    "gemma4_long_4100",
    true
);

/// Équivalent interne record/compare : référence per-op en mémoire, puis
/// replay forcé des mêmes tokens après un prefill résident sans repli.
/// La partition top-K + reste et le p99 des Δp maximaux par position sont
/// exactement ceux du golden, contrairement aux métriques plein vocab CLI.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub(crate) fn compare_gemma4_long_prefill(
    model: &crate::CausalDecoder,
    prompt: &[usize],
) -> TestResult {
    assert!((4096..=4128).contains(&prompt.len()), "prompt long attendu");
    compare_gemma4_prefill_at_length(model, prompt)
}

/// Même oracle et mêmes seuils, à une longueur totale contrôlée par le test.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub(crate) fn compare_gemma4_prefill_at_length(
    model: &crate::CausalDecoder,
    prompt: &[usize],
) -> TestResult {
    let meta: Meta = json("kld/gemma4_meta")?;
    assert_eq!(meta.model, "gemma4");
    assert!(
        prompt.len() <= crate::runtime_flags::prefill_resident_gemma4_max_tokens(),
        "relever explicitement SARAGOSSA_RUST_PREFILL_RESIDENT_GEMMA4_MAX_TOKENS pour cet oracle"
    );
    let n = meta.scored;
    let mut rows = Vec::with_capacity(n);
    let mut forced = Vec::with_capacity(n);
    crate::MetalExecutor::take_long_d512_dispatches_for_test();
    {
        let _flag = crate::runtime_flags::override_prefill_resident_gemma4_for_test(false);
        let (mut cache, state) = model.prefill_cache_state_batched_for_test(prompt)?;
        let mut logits = model.logits_from_final_state(&state)?;
        for position in 0..n {
            let values = logits.data();
            assert!(values.iter().all(|v| v.is_finite()));
            let token = crate::argmax(values)?;
            let max = f64::from(values[token]);
            let log_z = max
                + values
                    .iter()
                    .map(|&v| (f64::from(v) - max).exp())
                    .sum::<f64>()
                    .ln();
            let mut ids: Vec<usize> = (0..values.len()).collect();
            ids.sort_by(|&a, &b| values[b].total_cmp(&values[a]).then(a.cmp(&b)));
            ids.truncate(meta.top_k);
            let logprobs: Vec<f64> = ids
                .iter()
                .map(|&id| f64::from(values[id]) - log_z)
                .collect();
            rows.push((ids, logprobs));
            forced.push(token);
            if position + 1 < n {
                logits = model.next_logits_cached(&mut cache, token)?;
            }
        }
    }
    assert!(
        crate::MetalExecutor::take_long_d512_dispatches_for_test().is_empty(),
        "la référence doit utiliser le per-op (attention CPU f32)"
    );
    eprintln!(
        "kld/gemma4-long: référence per-op enregistrée, prompt={} positions={n}",
        prompt.len()
    );
    let _flag = crate::runtime_flags::override_prefill_resident_gemma4_for_test(true);
    // Diagnostic limité au prefill candidat : référence et decode inchangés.
    let strict = crate::runtime_flags::env_var("SARAGOSSA_GEMMA4_ORACLE_GEMM_F32")
        .is_some_and(|value| value == "1");
    let global_ablation = crate::runtime_flags::env_var("SARAGOSSA_GEMMA4_ORACLE_GLOBAL_PER_OP");
    let (mut cache, state) = crate::metal_backend::with_verify_strict_override(strict, || {
        let _greedy = crate::metal_backend::install_verify_greedy_scope(strict);
        assert_eq!(crate::metal_backend::verify_strict_active(), strict);
        if let Some(mode) = &global_ablation {
            assert!(mode == "1" || mode == "control");
            model
                .prefill_gemma4_global_ablation_for_test(prompt, mode == "1")
                .map(Some)
        } else {
            model.prefill_cache_state_metal_resident_for_test(prompt)
        }
    })?
    .ok_or("oracle long: prefill résident refusé")?;
    let trace = crate::MetalExecutor::take_long_d512_dispatches_for_test();
    eprintln!("kld/gemma4-long: trace causal_attention_prefill_long_d512_f32 {trace:?}");
    assert_eq!(
        !trace.is_empty(),
        prompt.len() > 2048 && global_ablation.as_deref() != Some("1"),
        "dispatch long d512 incohérent avec la longueur"
    );
    assert!(trace
        .iter()
        .all(|&(seq, dim)| seq == prompt.len() && dim == 512));
    let mut logits = model.logits_from_final_state(&state)?;
    let (mut kls, mut deltas, mut same) = (Vec::new(), Vec::new(), 0);
    for (position, ((ids, logprobs), &token)) in rows.iter().zip(&forced).enumerate() {
        assert!(logits.data().iter().all(|v| v.is_finite()));
        let (kl, delta) = metrics(logits.data(), ids, logprobs);
        kls.push(kl);
        deltas.push(delta);
        same += usize::from(crate::argmax(logits.data())? == token);
        if position + 1 < n {
            logits = model.next_logits_cached(&mut cache, token)?;
        }
    }
    let engine = summarize(&kls, &mut deltas, same);
    let noise = meta.mlx_bf16_noise;
    eprintln!("kld/gemma4-long: résident vs per-op {engine:?}");
    eprintln!("kld/gemma4-long: mlx bf16 {noise:?}");
    eprintln!(
        "kld/gemma4-long: ratio KL {:.6}, ratio p99 Δp {:.6}, limite {MAX_NOISE_RATIO}",
        engine.mean_kl / noise.mean_kl,
        engine.p99_delta_p / noise.p99_delta_p
    );
    assert!(
        within_noise(engine, noise),
        "kld/gemma4-long: hors du bruit bf16"
    );
    Ok(())
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn compare_with_prefill(
    name: &str,
    env: &str,
    prefill: impl Fn(
        &crate::CausalDecoder,
        &[usize],
    ) -> crate::Result<(crate::CausalDecoderCache, crate::Tensor)>,
) -> TestResult {
    let Some(path) = local(env)? else {
        return Ok(());
    };
    check_model(&format!("kld/{name}"), &path)?;
    let reference = load(name)?;
    let meta = &reference.meta;
    let assets = crate::ModelAssets::load_local(path)?;
    let executor = crate::MetalExecutor::new()?;
    let model =
        crate::load_causal_decoder_for_metal(&assets, &executor)?.with_metal_executor(executor)?;
    let (mut kls, mut deltas, mut same) = (Vec::new(), Vec::new(), 0);
    // NOTE: export opt-in par position (« kl top1 » par ligne) pour les
    // comparaisons appariées entre variantes ; sans effet sur les seuils.
    let mut tops = Vec::new();
    let mut rows = reference.rows.iter();
    for chunk in &meta.chunks {
        // Même découpe que kld.py : prefill du contexte, puis decode token par token.
        let start = chunk.len() - meta.scored;
        let (mut cache, mut logits) = prefill(&model, &chunk[..start])?;
        for position in start..chunk.len() {
            let (ids, logprobs) = rows.next().ok_or("référence trop courte")?;
            let values = logits.data();
            assert!(values.iter().all(|v| v.is_finite()), "logits non finis");
            let (kl, delta) = metrics(values, ids, logprobs);
            kls.push(kl);
            deltas.push(delta);
            let actual = crate::argmax(values)?;
            if actual != ids[0] {
                let margin = logprobs[0] - logprobs[1];
                eprintln!(
                    "kld/{name}/{}: greedy={actual}/{} marge_ref={margin:.6}",
                    kls.len() - 1,
                    ids[0]
                );
            }
            same += usize::from(actual == ids[0]);
            tops.push(actual == ids[0]);
            if position + 1 < chunk.len() {
                logits = model.next_logits_cached(&mut cache, chunk[position])?;
            }
        }
    }
    if let Ok(dump) = std::env::var("SARAGOSSA_TEST_KLD_DUMP") {
        let lines: String = kls
            .iter()
            .zip(&tops)
            .map(|(kl, top)| format!("{kl:e} {}\n", u8::from(*top)))
            .collect();
        std::fs::write(dump, lines)?;
    }
    let engine = summarize(&kls, &mut deltas, same);
    let noise = meta.mlx_bf16_noise;
    eprintln!("kld/{name}: saragossa {engine:?}");
    eprintln!("kld/{name}: mlx bf16  {noise:?}");
    eprintln!(
        "kld/{name}: ratio KL {:.6}, ratio p99 Δp {:.6}",
        engine.mean_kl / noise.mean_kl,
        engine.p99_delta_p / noise.p99_delta_p
    );
    assert!(
        within_noise(engine, noise),
        "kld/{name}: hors du bruit bf16"
    );
    Ok(())
}

#[cfg(all(target_os = "macos", feature = "metal"))]
macro_rules! oracle {
    ($test:ident, $name:literal, $env:literal) => {
        #[test]
        #[ignore = "oracle GPU externe: poids locaux requis, lancer sous verrou"]
        fn $test() -> TestResult {
            compare($name, $env)
        }
    };
}
#[cfg(all(target_os = "macos", feature = "metal"))]
oracle!(external_kld_qwen38, "qwen38", "SARAGOSSA_QWEN38_DIR");
#[cfg(all(target_os = "macos", feature = "metal"))]
oracle!(
    external_kld_qwen36_moe,
    "qwen36_moe",
    "SARAGOSSA_QWEN36_MOE_DIR"
);
#[cfg(all(target_os = "macos", feature = "metal"))]
oracle!(external_kld_gemma3, "gemma3", "SARAGOSSA_GEMMA3_DIR");
#[cfg(all(target_os = "macos", feature = "metal"))]
oracle!(external_kld_gemma4, "gemma4", "SARAGOSSA_GEMMA4_DIR");
#[cfg(all(target_os = "macos", feature = "metal"))]
oracle!(
    external_kld_mistral7b,
    "mistral7b",
    "SARAGOSSA_MISTRAL7B_DIR"
);
