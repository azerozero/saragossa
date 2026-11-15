//! Oracle et chronométrage local sur poids réels ; lancer sous le verrou GPU.
//!
//! Usage : `decision_bench MODEL FIXTURES [REPETITIONS]`, où `FIXTURES` est
//! `tests/fixtures/decision`. Écrit une ligne JSON par longueur sur stdout.
use saragossa::decision::DecisionModel;
use serde_json::{json, Value};
use std::{error::Error, path::Path, time::Instant};

fn read(path: &Path) -> Result<Value, Box<dyn Error>> {
    Ok(serde_json::from_slice(&std::fs::read(path)?)?)
}

/// Écart absolu maximal et KL(référence ‖ saragossa) maximale par question.
fn drift(a: &[Vec<f32>], b: &Value) -> Result<(f64, f64), Box<dyn Error>> {
    let b: Vec<Vec<f32>> = serde_json::from_value(b.clone())?;
    if a.len() != b.len() || a.iter().zip(&b).any(|(x, y)| x.len() != y.len()) {
        return Err("formes de probabilités différentes".into());
    }
    let (mut abs, mut kl) = (0.0_f64, 0.0_f64);
    for (x, y) in a.iter().zip(&b) {
        let mut question = 0.0;
        for (&p, &r) in x.iter().zip(y) {
            abs = abs.max(f64::from((p - r).abs()));
            if r > 0.0 {
                question += f64::from(r) * (f64::from(r) / f64::from(p).max(1e-30)).ln();
            }
        }
        kl = kl.max(question);
    }
    Ok((abs, kl))
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if !(3..=4).contains(&args.len()) {
        return Err("usage: decision_bench MODEL FIXTURES [REPETITIONS]".into());
    }
    let repetitions: usize = args.get(3).map_or(Ok(21), |v| v.parse())?;
    if repetitions == 0 {
        return Err("au moins une répétition".into());
    }
    let fixtures = Path::new(&args[2]);
    let model = DecisionModel::load(&args[1], true)?;
    let corpus = read(&fixtures.join("reference-f32.json"))?;
    let (mut corpus_max, mut corpus_kl) = (0.0_f64, 0.0_f64);
    for record in corpus["records"].as_array().ok_or("records absents")? {
        let e = model.encode(&record["request"])?;
        if json!(e.tokens) != record["tokens"] || json!(e.markers) != record["markers"] {
            return Err("encodage différent de la référence".into());
        }
        let p = model.probabilities(&e.tokens, &e.markers, &e.counts)?;
        let (abs, kl) = drift(&p, &record["probabilities"])?;
        corpus_max = corpus_max.max(abs);
        corpus_kl = corpus_kl.max(kl);
    }
    println!(
        "{}",
        json!({"corpus_max_abs": corpus_max, "corpus_max_kl": corpus_kl})
    );
    for case in read(&fixtures.join("reference-long.json"))?
        .as_array()
        .ok_or("cas longs absents")?
    {
        let tokens: Vec<usize> = serde_json::from_value(case["tokens"].clone())?;
        let markers: Vec<usize> = serde_json::from_value(case["markers"].clone())?;
        let counts: Vec<usize> = serde_json::from_value(case["counts"].clone())?;
        for _ in 0..3 {
            model.probabilities(&tokens, &markers, &counts)?;
        }
        let mut ms = Vec::with_capacity(repetitions);
        let mut p = Vec::new();
        for _ in 0..repetitions {
            let start = Instant::now();
            p = model.probabilities(&tokens, &markers, &counts)?;
            ms.push(start.elapsed().as_secs_f64() * 1e3);
        }
        let mut sorted = ms.clone();
        sorted.sort_by(f64::total_cmp);
        let p95 = sorted[(sorted.len() * 95).div_ceil(100).saturating_sub(1)];
        let (abs, kl) = drift(&p, &case["probabilities"])?;
        println!(
            "{}",
            json!({"length": tokens.len(), "p50_ms": sorted[sorted.len() / 2], "p95_ms": p95,
                   "max_abs": abs, "max_kl": kl, "ms": ms})
        );
    }
    Ok(())
}
