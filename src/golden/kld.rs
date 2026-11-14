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
    for (name, _) in CHAT_MODELS {
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
    let mut rows = reference.rows.iter();
    for chunk in &meta.chunks {
        // Même découpe que kld.py : prefill du contexte, puis decode token par token.
        let start = chunk.len() - meta.scored;
        let (mut cache, mut logits) = prefill(&model, &chunk[..start])?;
        for position in start..chunk.len() {
            let (ids, logprobs) = rows.next().ok_or("référence trop courte")?;
            let values = logits.data();
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
            if position + 1 < chunk.len() {
                logits = model.next_logits_cached(&mut cache, chunk[position])?;
            }
        }
    }
    let engine = summarize(&kls, &mut deltas, same);
    let noise = meta.mlx_bf16_noise;
    eprintln!("kld/{name}: saragossa {engine:?}");
    eprintln!("kld/{name}: mlx bf16  {noise:?}");
    eprintln!(
        "kld/{name}: ratio KL {:.2}, ratio p99 Δp {:.2}",
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
