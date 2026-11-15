//! Lecture de lettres H2O sur poids réels ; lancer sous le verrou GPU.
//!
//! Usage : `letters_bench MODEL FIXTURES [REPETITIONS]`, `FIXTURES` valant
//! `tests/fixtures/decision/h2o`. Une ligne JSON par mesure sur stdout :
//! écart au corpus, lectures bidirectionnelles de 128/512/2048 tokens, puis
//! les questions de l'état de 4000 tokens en causal, avec et sans cache de
//! préfixe (égalité au bit comptée).
use saragossa::decision::letters::{AttentionMode, LetterModel};
use serde_json::{json, Value};
use std::{error::Error, path::Path, time::Instant};

fn read(path: &Path) -> Result<Value, Box<dyn Error>> {
    Ok(serde_json::from_slice(&std::fs::read(path)?)?)
}

fn max_abs(a: &[Vec<f32>], b: &Value) -> Result<f64, Box<dyn Error>> {
    let b: Vec<Vec<f32>> = serde_json::from_value(b.clone())?;
    if a.len() != b.len() || a.iter().zip(&b).any(|(x, y)| x.len() != y.len()) {
        return Err("formes de probabilités différentes".into());
    }
    Ok(a.iter()
        .flatten()
        .zip(b.iter().flatten())
        .map(|(x, y)| f64::from((x - y).abs()))
        .fold(0.0, f64::max))
}

fn stats(mut ms: Vec<f64>) -> Value {
    let raw = ms.clone();
    ms.sort_by(f64::total_cmp);
    let p95 = ms[(ms.len() * 95).div_ceil(100).saturating_sub(1)];
    json!({"p50_ms": ms[ms.len() / 2], "p95_ms": p95, "ms": raw})
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if !(3..=4).contains(&args.len()) {
        return Err("usage: letters_bench MODEL FIXTURES [REPETITIONS]".into());
    }
    let repetitions: usize = args.get(3).map_or(Ok(21), |v| v.parse())?;
    if repetitions == 0 {
        return Err("au moins une répétition".into());
    }
    let fixtures = Path::new(&args[2]);
    let model = LetterModel::load(&args[1], true)?;
    let footprint_mib = || saragossa::memory_guard::process_footprint_bytes().map(|b| b >> 20);
    let mut corpus = [0.0_f64; 2];
    for record in read(&fixtures.join("reference-f32.json"))?["records"]
        .as_array()
        .ok_or("records absents")?
    {
        let mode = model.mode(&record["request"])?;
        let reads = model.encode(&record["request"])?;
        let drift = max_abs(
            &model.probabilities(&reads, mode, true)?,
            &record["probabilities"],
        )?;
        let slot = usize::from(mode == AttentionMode::Causal);
        corpus[slot] = corpus[slot].max(drift);
    }
    println!(
        "{}",
        json!({"corpus_bidirectional_max_abs": corpus[0], "corpus_causal_max_abs": corpus[1],
               "footprint_mib": footprint_mib()})
    );
    // Cas longs reconstruits comme la référence : `Delivery` + ` parcel` × n.
    let question = json!({"type":"noul","instructions":"Does the text mention a parcel delivery?"});
    let long = read(&fixtures.join("reference-long.json"))?;
    for case in long.as_array().ok_or("cas longs absents")? {
        let want: Vec<usize> = serde_json::from_value(case["tokens"].clone())?;
        let base = model.encode(&json!({"state":"Delivery","questions":{"q":question}}))?;
        let pad = want.len().saturating_sub(base[0].tokens.len());
        let state = format!("Delivery{}", " parcel".repeat(pad));
        let reads = model.encode(&json!({"state":state,"questions":{"q":question}}))?;
        if reads[0].tokens != want {
            return Err("encodage long différent de la référence".into());
        }
        let mode = AttentionMode::Bidirectional;
        for _ in 0..3 {
            model.probabilities(&reads, mode, true)?;
        }
        let (mut ms, mut p) = (Vec::with_capacity(repetitions), Vec::new());
        for _ in 0..repetitions {
            let start = Instant::now();
            p = model.probabilities(&reads, mode, true)?;
            ms.push(start.elapsed().as_secs_f64() * 1e3);
        }
        let mut line = stats(ms);
        line["length"] = json!(want.len());
        line["max_abs"] = json!(max_abs(&p, &case["probabilities"])?);
        println!("{line}");
    }
    let cache = read(&fixtures.join("reference-cache.json"))?;
    let mut request = cache["request"].clone();
    request["attention_mode"] = json!("causal");
    let reads = model.encode(&request)?;
    let prefix = cache["prefix_tokens"].as_array().map_or(0, Vec::len);
    let mode = AttentionMode::Causal;
    model.probabilities(&reads[..2], mode, true)?;
    let mut cold = Vec::new();
    // ABBA : sans cache, avec, avec, sans.
    for share in [false, true, true, false] {
        let start = Instant::now();
        let p = model.probabilities(&reads, mode, share)?;
        let ms = start.elapsed().as_secs_f64() * 1e3;
        let mut line = json!({"causal": if share { "cache" } else { "cold" },
            "questions": reads.len(), "prefix": prefix, "total_ms": ms,
            "per_question_ms": ms / reads.len() as f64, "footprint_mib": footprint_mib()});
        if share {
            let flat = p.iter().flatten().zip(cold.iter().flatten());
            let (mut equal, mut total, mut gap) = (0, 0, 0.0_f32);
            for (a, b) in flat {
                let (a, b): (&f32, &f32) = (a, b);
                equal += usize::from(a.to_bits() == b.to_bits());
                total += 1;
                gap = gap.max((a - b).abs());
            }
            line["bit_equal"] = json!(format!("{equal}/{total}"));
            line["max_abs_vs_cold"] = json!(gap);
        } else {
            cold = p;
        }
        println!("{line}");
    }
    Ok(())
}
