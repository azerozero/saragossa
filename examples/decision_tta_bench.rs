//! Chronométrage TTA=1 et TTA=4 de `decide` sur poids THX-02 réels ; lancer
//! sous le verrou GPU.
//!
//! Usage : `decision_tta_bench MODEL [REPETITIONS]`. Une question `choice` à
//! quatre options, état allongé jusqu'à 128, 512 et 2048 tokens par lecture.
//! Une ligne JSON par (longueur, tta) sur stdout, tokenisation comprise.
//! `SARAGOSSA_DECISION_PACK_TOKENS=0` mesure les lectures en prefills séparés.
use saragossa::decision::DecisionModel;
use serde_json::{json, Value};
use std::{error::Error, time::Instant};

fn request(state: &str) -> Value {
    json!({"state": state, "questions": {"next": {"type": "choice",
        "instructions": "What should happen next?",
        "criteria": {"replace": "Send a replacement", "refund": "Issue a refund",
                     "wait": "Ask the customer to wait", "escalate": "Escalate to a manager"}}}})
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if !(2..=3).contains(&args.len()) {
        return Err("usage: decision_tta_bench MODEL [REPETITIONS]".into());
    }
    let repetitions: usize = args.get(2).map_or(Ok(21), |v| v.parse())?;
    if repetitions == 0 {
        return Err("au moins une répétition".into());
    }
    let model = DecisionModel::load(&args[1], true)?;
    for length in [128_usize, 512, 2048] {
        // Même allongement que les oracles longs : « parcel » répété.
        let base = model.encode(&request("Delivery"))?[0].tokens.len();
        let state = format!("Delivery{}", " parcel".repeat(length.saturating_sub(base)));
        let got = model.encode(&request(&state))?[0].tokens.len();
        if got != length {
            return Err(format!("{got} tokens au lieu de {length}").into());
        }
        for tta in [1, 4] {
            let mut req = request(&state);
            req["tta"] = json!(tta);
            for _ in 0..3 {
                model.decide(&req)?;
            }
            let mut ms = Vec::with_capacity(repetitions);
            let mut answer = Value::Null;
            for _ in 0..repetitions {
                let start = Instant::now();
                answer = model.decide(&req)?;
                ms.push(start.elapsed().as_secs_f64() * 1e3);
            }
            let mut sorted = ms.clone();
            sorted.sort_by(f64::total_cmp);
            let p95 = sorted[(sorted.len() * 95).div_ceil(100).saturating_sub(1)];
            println!(
                "{}",
                json!({"length": length, "tta": tta, "p50_ms": sorted[sorted.len() / 2],
                       "p95_ms": p95, "probabilities": answer["answers"]["next"]["probabilities"],
                       "ms": ms})
            );
        }
    }
    Ok(())
}
