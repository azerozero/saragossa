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

const FLOAT32_MODELS: [&str; 2] = ["gemma4", "qwen36_moe"];

fn load_fixture(name: &str, float32: bool) -> TestResult<Fixture> {
    let stem = format!("logits/{name}{}", if float32 { "_f32" } else { "" });
    let (header, bytes) = read(&stem)?;
    assert_eq!(header.dtype, "utf8");
    if float32 {
        assert_eq!(header.source["float32_control"], true, "{stem}: provenance");
    }
    let fixture: Fixture = serde_json::from_slice(&bytes)?;
    assert_eq!(fixture.model, name);
    validate(&fixture);
    Ok(fixture)
}

/// Une capture en attente ne peut manquer que si les deux fichiers sont absents.
fn capture_present(stem: &std::path::Path, status: &str) -> TestResult<bool> {
    assert!(matches!(status, "captured" | "pending_capture"));
    let header = stem.with_extension("json").try_exists()?;
    let payload = stem.with_extension("bin").try_exists()?;
    assert_eq!(header, payload, "{}: capture incomplète", stem.display());
    assert!(
        header || status == "pending_capture",
        "capture obligatoire absente"
    );
    Ok(header)
}

#[test]
fn external_logits_inventory_is_complete() -> TestResult {
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root().join("logits/inventory.json"))?)?;
    let captures = manifest["captures"]
        .as_object()
        .ok_or("inventaire absent")?;
    assert_eq!(captures.len(), CHAT_MODELS.len() + FLOAT32_MODELS.len());
    for (name, _) in CHAT_MODELS {
        assert_eq!(captures[name], "captured");
        load_fixture(name, false)?;
    }
    for name in FLOAT32_MODELS {
        let stem = format!("{name}_f32");
        let status = captures[&stem].as_str().ok_or("statut f32 absent")?;
        if capture_present(&root().join("logits").join(&stem), status)? {
            load_fixture(name, true)?;
        } else {
            eprintln!("logits/{stem}: capture float32 en attente");
        }
    }
    Ok(())
}

#[test]
fn external_logits_pending_capture_requires_both_files_absent() -> TestResult {
    let directory = tempfile::tempdir()?;
    let stem = directory.path().join("control_f32");
    assert!(!capture_present(&stem, "pending_capture")?);
    assert!(std::panic::catch_unwind(|| capture_present(&stem, "captured")).is_err());
    for extension in ["json", "bin"] {
        std::fs::write(stem.with_extension(extension), [])?;
        assert!(std::panic::catch_unwind(|| capture_present(&stem, "pending_capture")).is_err());
        std::fs::remove_file(stem.with_extension(extension))?;
    }
    std::fs::write(stem.with_extension("json"), [])?;
    std::fs::write(stem.with_extension("bin"), [])?;
    assert!(capture_present(&stem, "pending_capture")?);
    assert!(capture_present(&stem, "captured")?);
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

enum Comparison {
    Strict,
    Float32,
    KnownBf16 { max_mismatches: usize },
}

impl Comparison {
    fn accepts(&self, mismatches: usize, metric_failures: usize) -> bool {
        match self {
            Self::Strict | Self::Float32 => mismatches == 0 && metric_failures == 0,
            Self::KnownBf16 { max_mismatches } => mismatches <= *max_mismatches,
        }
    }
}

#[test]
fn external_logits_comparison_keeps_strict_gates_and_bounds_bf16() {
    for comparison in [Comparison::Strict, Comparison::Float32] {
        assert!(comparison.accepts(0, 0));
        assert!(!comparison.accepts(1, 0));
        assert!(!comparison.accepts(0, 1));
    }
    for max_mismatches in [7, 2] {
        let comparison = Comparison::KnownBf16 { max_mismatches };
        assert!(comparison.accepts(0, 0));
        assert!(comparison.accepts(max_mismatches, 96));
        assert!(!comparison.accepts(max_mismatches + 1, 0));
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn compare(name: &str, env: &str, comparison: Comparison) -> TestResult {
    let Some(path) = local(env)? else {
        return Ok(());
    };
    let float32 = matches!(comparison, Comparison::Float32);
    let stem = format!("logits/{name}{}", if float32 { "_f32" } else { "" });
    check_model(&stem, &path)?;
    let fixture = load_fixture(name, float32)?;
    let assets = crate::ModelAssets::load_local(path)?;
    let executor = crate::MetalExecutor::new()?;
    let model =
        crate::load_causal_decoder_for_metal(&assets, &executor)?.with_metal_executor(executor)?;
    let mut failures = Vec::new();
    let mut mismatches = 0;
    let mut metric_failures = 0;
    let mut near_ties = 0;
    let mut max_kl = 0.0_f64;
    let mut max_delta = 0.0_f64;
    for (case_index, case) in fixture.cases.iter().enumerate() {
        let (mut cache, mut logits) = model.prefill_cache(&case.prompt_ids)?;
        for (index, step) in case.steps.iter().enumerate() {
            let values = logits.data();
            let actual = crate::argmax(values)?;
            let (kl, delta) = metrics(values, step);
            let margin = step.logprobs[0] - step.logprobs[1];
            let mismatch = actual != step.token;
            mismatches += usize::from(mismatch);
            metric_failures += usize::from(kl > 0.02 || delta > 0.02);
            near_ties += usize::from(margin <= 0.01);
            max_kl = max_kl.max(kl);
            max_delta = max_delta.max(delta);
            eprintln!("{stem}/{case_index}/{index}: greedy={actual}/{} mismatch={mismatch} margin={margin:.6} near_tie={} KL21={kl:.6} delta_p={delta:.6}", step.token, margin <= 0.01);
            if actual != step.token || kl > 0.02 || delta > 0.02 {
                failures.push(format!("{case_index}/{index}: greedy={actual}/{} margin={margin} near_tie={} KL21={kl} delta_p={delta}", step.token, margin <= 0.01));
            }
            // Même préfixe : une divergence ne doit pas polluer toutes les mesures suivantes.
            if index + 1 < case.steps.len() {
                logits = model.next_logits_cached(&mut cache, step.token)?;
            }
        }
    }
    eprintln!("{stem}: greedy_mismatches={mismatches}/96 near_ties={near_ties}/96 metric_failures={metric_failures}/96 max_KL21={max_kl:.6} max_delta_p={max_delta:.6}");
    assert!(
        comparison.accepts(mismatches, metric_failures),
        "{stem}: {}",
        failures.join("\n")
    );
    Ok(())
}

#[cfg(all(target_os = "macos", feature = "metal"))]
macro_rules! oracle {
    ($test:ident, $name:literal, $env:literal, $comparison:expr) => {
        #[test]
        #[ignore = "oracle GPU externe: poids locaux et capture mlx-lm requis, lancer sous verrou"]
        fn $test() -> TestResult {
            compare($name, $env, $comparison)
        }
    };
}
#[cfg(all(target_os = "macos", feature = "metal"))]
oracle!(
    external_logits_qwen38,
    "qwen38",
    "SARAGOSSA_QWEN38_DIR",
    Comparison::Strict
);
#[cfg(all(target_os = "macos", feature = "metal"))]
oracle!(
    external_logits_qwen36_moe,
    "qwen36_moe",
    "SARAGOSSA_QWEN36_MOE_DIR",
    Comparison::KnownBf16 { max_mismatches: 2 }
);
#[cfg(all(target_os = "macos", feature = "metal"))]
oracle!(
    external_logits_gemma3,
    "gemma3",
    "SARAGOSSA_GEMMA3_DIR",
    Comparison::Strict
);
#[cfg(all(target_os = "macos", feature = "metal"))]
oracle!(
    external_logits_gemma4,
    "gemma4",
    "SARAGOSSA_GEMMA4_DIR",
    Comparison::KnownBf16 { max_mismatches: 7 }
);
#[cfg(all(target_os = "macos", feature = "metal"))]
oracle!(
    external_logits_gemma4_f32,
    "gemma4",
    "SARAGOSSA_GEMMA4_DIR",
    Comparison::Float32
);
#[cfg(all(target_os = "macos", feature = "metal"))]
oracle!(
    external_logits_qwen36_moe_f32,
    "qwen36_moe",
    "SARAGOSSA_QWEN36_MOE_DIR",
    Comparison::Float32
);
