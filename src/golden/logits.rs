use super::external::*;
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    model: String,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    prompt_ids: Vec<usize>,
    steps: Vec<Step>,
}

#[derive(Deserialize)]
struct Step {
    token: usize,
    top_ids: Vec<usize>,
    logprobs: Vec<f64>,
}

fn validate(fixture: &Fixture) {
    assert_eq!(fixture.cases.len(), 6);
    for case in &fixture.cases {
        assert!(!case.prompt_ids.is_empty());
        assert_eq!(case.steps.len(), 16);
        for step in &case.steps {
            assert_eq!(step.top_ids.len(), 20);
            assert_eq!(step.logprobs.len(), 20);
            assert_eq!(step.token, step.top_ids[0]);
            assert!(step.logprobs.iter().all(|v| v.is_finite() && *v <= 0.0));
            assert!(step.logprobs.windows(2).all(|v| v[0] >= v[1]));
            assert!(step.logprobs.iter().map(|v| v.exp()).sum::<f64>() <= 1.00001);
            let unique: std::collections::HashSet<_> = step.top_ids.iter().collect();
            assert_eq!(unique.len(), 20);
        }
    }
}

/// KL sur une partition de 21 catégories : top-20 de référence + masse restante.
/// Ce n'est ni la KL du vocabulaire entier, ni une renormalisation du top-20.
fn metrics(logits: &[f32], step: &Step) -> (f64, f64) {
    assert!(logits.iter().all(|v| v.is_finite()));
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let log_z = max
        + logits
            .iter()
            .map(|v| (f64::from(*v) - max).exp())
            .sum::<f64>()
            .ln();
    let p: Vec<f64> = step.logprobs.iter().map(|v| v.exp()).collect();
    let q: Vec<f64> = step
        .top_ids
        .iter()
        .map(|id| (f64::from(logits[*id]) - log_z).exp())
        .collect();
    let mut kl = 0.0;
    let mut delta = 0.0_f64;
    for (p, q) in p.iter().copied().zip(q.iter().copied()).chain([(
        (1.0 - p.iter().sum::<f64>()).max(0.0),
        (1.0 - q.iter().sum::<f64>()).max(0.0),
    )]) {
        if p > 0.0 {
            kl += p * (p / q.max(f64::MIN_POSITIVE)).ln();
        }
        delta = delta.max((p - q).abs());
    }
    (kl, delta)
}

#[test]
fn external_logits_inventory_is_complete() -> TestResult {
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root().join("logits/inventory.json"))?)?;
    let expected: Vec<_> = CHAT_MODELS.iter().map(|(name, _)| *name).collect();
    let captures = manifest["captures"]
        .as_object()
        .ok_or("inventaire absent")?;
    assert_eq!(captures.len(), expected.len());
    for (name, _) in CHAT_MODELS {
        assert_eq!(captures[name], "captured");
        let fixture: Fixture = json(&format!("logits/{name}"))?;
        assert_eq!(fixture.model, name);
        validate(&fixture);
    }
    Ok(())
}

#[test]
fn external_logits_metrics_detect_tail_and_top_errors() {
    let logits = [0.0_f32; 21];
    let step = Step {
        token: 0,
        top_ids: (0..20).collect(),
        logprobs: vec![-21.0_f64.ln(); 20],
    };
    let (kl, delta) = metrics(&logits, &step);
    assert!(kl.abs() < 1e-12 && delta < 1e-12);
    let mut changed = logits;
    changed[20] = 2.0;
    let (kl, delta) = metrics(&changed, &step);
    assert!(kl > 0.02 && delta > 0.02);
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn compare(name: &str, env: &str) -> TestResult {
    let path = local(env)?;
    let stem = format!("logits/{name}");
    check_model(&stem, &path)?;
    let fixture: Fixture = json(&stem)?;
    validate(&fixture);
    let assets = crate::ModelAssets::load_local(path)?;
    let executor = crate::MetalExecutor::new()?;
    let model =
        crate::load_causal_decoder_for_metal(&assets, &executor)?.with_metal_executor(executor)?;
    let mut failures = Vec::new();
    for (case_index, case) in fixture.cases.iter().enumerate() {
        let (mut cache, mut logits) = model.prefill_cache(&case.prompt_ids)?;
        for (index, step) in case.steps.iter().enumerate() {
            let values = logits.data();
            let actual = crate::argmax(values)?;
            let (kl, delta) = metrics(values, step);
            let margin = step.logprobs[0] - step.logprobs[1];
            eprintln!("{name}/{case_index}/{index}: greedy={actual}/{} margin={margin:.6} near_tie={} KL21={kl:.6} delta_p={delta:.6}", step.token, margin <= 0.01);
            if actual != step.token || kl > 0.02 || delta > 0.02 {
                failures.push(format!("{case_index}/{index}: greedy={actual}/{} margin={margin} near_tie={} KL21={kl} delta_p={delta}", step.token, margin <= 0.01));
            }
            // Même préfixe : une divergence ne doit pas polluer toutes les mesures suivantes.
            if index + 1 < case.steps.len() {
                logits = model.next_logits_cached(&mut cache, step.token)?;
            }
        }
    }
    assert!(failures.is_empty(), "{name}: {}", failures.join("\n"));
    Ok(())
}

#[cfg(all(target_os = "macos", feature = "metal"))]
macro_rules! oracle {
    ($test:ident, $name:literal, $env:literal) => {
        #[test]
        #[ignore = "oracle GPU externe: poids locaux et capture mlx-lm requis, lancer sous verrou"]
        fn $test() -> TestResult {
            compare($name, $env)
        }
    };
}
#[cfg(all(target_os = "macos", feature = "metal"))]
oracle!(external_logits_qwen38, "qwen38", "SARAGOSSA_QWEN38_DIR");
#[cfg(all(target_os = "macos", feature = "metal"))]
oracle!(
    external_logits_qwen36_moe,
    "qwen36_moe",
    "SARAGOSSA_QWEN36_MOE_DIR"
);
#[cfg(all(target_os = "macos", feature = "metal"))]
oracle!(external_logits_gemma3, "gemma3", "SARAGOSSA_GEMMA3_DIR");
#[cfg(all(target_os = "macos", feature = "metal"))]
oracle!(external_logits_gemma4, "gemma4", "SARAGOSSA_GEMMA4_DIR");
