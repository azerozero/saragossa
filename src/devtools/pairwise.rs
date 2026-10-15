//! Campagne locale : mêmes poids/processus, contrôle D1 committed, aucun env muté.
use crate::runtime_flags::{
    self as flags,
    pairwise::{self, Case, Guard, Mode},
};
use crate::{
    CausalDecoder, GenerationOptions, InferError, Result, SpeculativeOutput, SpeculativeStats,
};
use serde::Deserialize;
use serde_json::json;
use std::time::{Duration, Instant};

#[derive(Deserialize)]
struct Row {
    id: String,
    #[serde(flatten)]
    case: Case,
}
#[derive(Deserialize)]
struct Matrix {
    rows: Vec<Row>,
}
struct Run {
    instrumented: bool,
    output: SpeculativeOutput,
    total: Duration,
    seen: Vec<usize>,
    arrivals: Vec<u128>,
    sampled_metal_max: u64,
    profile: [u64; 4],
    hits: [u64; 7],
    io: [u64; 4],
}

#[allow(
    clippy::too_many_arguments,
    reason = "harnais : entrées, variante et instrumentation explicites"
)]
fn one(
    decoder: &CausalDecoder,
    prompt: &[usize],
    options: &GenerationOptions,
    case: Case,
    budget: usize,
    trace: bool,
    cancel: Option<usize>,
    device: &metal::DeviceRef,
) -> Result<Run> {
    let _guard = Guard::set(case, trace)?;
    let (cb, wait, read, dispatch) = crate::metal_backend::decode_profile_snapshot();
    let started = Instant::now();
    let mut seen = Vec::new();
    let mut arrivals = Vec::new();
    let mut sampled_metal_max = 0;
    let mut callback = |token| {
        seen.push(token);
        if trace {
            arrivals.push(started.elapsed().as_nanos());
            sampled_metal_max = sampled_metal_max.max(device.current_allocated_size());
        }
        cancel.is_none_or(|limit| seen.len() < limit)
    };
    let output = if case.mode == Mode::Ar {
        let ar = if trace || cancel.is_some() {
            decoder.generate_greedy_streaming_with_options(
                prompt,
                budget,
                options,
                &mut callback,
            )?
        } else {
            decoder.generate_greedy_timed_with_options(prompt, budget, options)?
        };
        SpeculativeOutput {
            tokens: ar.tokens,
            stats: SpeculativeStats::default(),
            loop_duration: ar.timings.decode,
        }
    } else if trace || cancel.is_some() {
        decoder.generate_greedy_mtp_streaming_with_options(
            prompt,
            budget,
            options,
            case.depth(),
            &mut callback,
        )?
    } else {
        decoder.generate_greedy_mtp_batched_with_options(prompt, budget, options, case.depth())?
    };
    let total = started.elapsed();
    let (cb1, wait1, read1, dispatch1) = crate::metal_backend::decode_profile_snapshot();
    Ok(Run {
        instrumented: trace,
        output,
        total,
        seen,
        arrivals,
        sampled_metal_max,
        profile: [cb1 - cb, wait1 - wait, read1 - read, dispatch1 - dispatch],
        hits: pairwise::hits(),
        io: pairwise::io(),
    })
}

fn exact(run: &Run, expected: &[usize], id: &str) -> Result<()> {
    if run.output.tokens != expected {
        return Err(InferError::Config(format!(
            "pairwise {id}: oracle divergent, got={:?} expected={expected:?}",
            run.output.tokens
        )));
    }
    Ok(())
}
fn routes(run: &Run, case: Case, id: &str) -> Result<()> {
    let expected = [
        case.hidden,
        case.embedding,
        case.overlap,
        case.mode == Mode::Block2,
        case.p1,
        case.p2,
        case.p3,
    ];
    for (index, enabled) in expected.into_iter().enumerate() {
        if enabled != (run.hits[index] > 0) {
            return Err(InferError::Config(format!("pairwise {id}: chemin{index} enabled={enabled} hits={} (aucun fallback compté comme activation)", run.hits[index])));
        }
    }
    Ok(())
}
fn benchmark_control(name: &str, rows: &[&Row]) -> Result<Case> {
    match name {
        "d1" => Ok(Case::default()),
        "ar" if !rows.is_empty() && rows.iter().all(|r| r.case.mode == Mode::Ar) => Ok(Case {
            mode: Mode::Ar,
            ..Case::default()
        }),
        _ => Err(InferError::Config(
            "contrôle pairwise : d1 obligatoire, ou ar pour un A/B secondaire exclusivement AR"
                .into(),
        )),
    }
}

fn benchmark_order(pair: usize, reverse: bool) -> [bool; 2] {
    let candidate_first = (pair % 2 != 0) ^ reverse;
    [candidate_first, !candidate_first]
}

fn measurement(run: &Run, id: &str, phase: &str, pair: usize, role: &str) -> serde_json::Value {
    let s = &run.output.stats;
    json!({"id":id,"phase":phase,"pair":pair,"role":role,"instrumented":run.instrumented,
        "tokens":run.output.tokens,"loop_ns":run.output.loop_duration.as_nanos(),"total_ns":run.total.as_nanos(),
        "stats":{"proposed":s.proposed,"accepted":s.accepted,"rejected":s.rejected,"verifications":s.verifications,
            "rollbacks":s.rollbacks,"by_position":s.proposed_by_position,"accepted_by_position":s.accepted_by_position},
        "profile_including_prefill":run.instrumented.then_some(run.profile),
        "hits":run.instrumented.then_some(run.hits),
        "tracked_shared_copy_calls_bytes_including_prefill":run.instrumented.then_some(run.io),
        "callback_arrivals_ns":run.instrumented.then_some(&run.arrivals),
        "sampled_live_metal_max_not_peak":run.instrumented.then_some(run.sampled_metal_max)})
}
fn report(run: &Run, id: &str, phase: &str, pair: usize, role: &str) {
    eprintln!("pairwise {}", measurement(run, id, phase, pair, role));
}

pub(super) fn run(
    decoder: &CausalDecoder,
    prompt: &[usize],
    budget: usize,
    options: &GenerationOptions,
) -> Result<()> {
    if options.temperature.abs() > f32::EPSILON
        || options.token_constraint.is_some()
        || !options.stop_sequences.is_empty()
        || budget < 9
        || flags::prefix_cache_enabled()
        || flags::mtp_history_policy() != flags::MtpHistoryPolicy::Committed
        || flags::gpu_counters_enabled()
        || flags::decode_profile_enabled()
        || flags::env_flag("SARAGOSSA_RUST_MTP_STEP_PROFILE", false)
        || flags::env_flag("SARAGOSSA_RUST_TRACE_COMPONENTS", false)
    {
        return Err(InferError::Config("pairwise exige greedy, budget>=9, committed, prefix-cache et instrumentation externes OFF".into()));
    }
    let path = flags::env_var("SARAGOSSA_RUST_PAIRWISE_MATRIX")
        .ok_or_else(|| InferError::Config("PAIRWISE_MATRIX requis".into()))?;
    let text = std::fs::read_to_string(path)
        .map_err(|error| InferError::Config(format!("lecture matrice : {error}")))?;
    let matrix: Matrix = serde_json::from_str(&text)
        .map_err(|error| InferError::Config(format!("matrice JSON : {error}")))?;
    let ids = flags::env_var("SARAGOSSA_RUST_PAIRWISE_IDS")
        .ok_or_else(|| InferError::Config("PAIRWISE_IDS requis pour borner le lot".into()))?;
    let mut selected = Vec::new();
    for id in ids.split(',') {
        let row = matrix
            .rows
            .iter()
            .find(|r| r.id == id)
            .ok_or_else(|| InferError::Config(format!("cas inconnu {id}")))?;
        if selected.iter().any(|r: &&Row| r.id == id) {
            return Err(InferError::Config("cas dupliqué".into()));
        }
        row.case.validate()?;
        selected.push(row);
    }
    let action = flags::env_var("SARAGOSSA_RUST_PAIRWISE_ACTION").unwrap_or_else(|| "smoke".into());
    if !matches!(action.as_str(), "smoke" | "bench" | "metrics" | "edges") {
        return Err(InferError::Config("action pairwise inconnue".into()));
    }
    let pairs: usize = flags::env_var("SARAGOSSA_RUST_PAIRWISE_PAIRS")
        .unwrap_or_else(|| "6".into())
        .parse()
        .map_err(|_| InferError::Config("PAIRWISE_PAIRS invalide".into()))?;
    if !(1..=6).contains(&pairs) || selected.len() > 4 {
        return Err(InferError::Config(
            "lot limité à4cas et6paires ; borner aussi la durée à60s".into(),
        ));
    }
    let control_mode =
        flags::env_var("SARAGOSSA_RUST_PAIRWISE_CONTROL").unwrap_or_else(|| "d1".into());
    let control = benchmark_control(&control_mode, &selected)?;
    // Les longs prefills imposent parfois un lot d'une paire ; inverser le lot suivant.
    let reverse_order = flags::env_flag("SARAGOSSA_RUST_PAIRWISE_REVERSE_ORDER", false);
    if control.mode == Mode::Ar && action != "bench" {
        return Err(InferError::Config(
            "contrôle AR réservé au benchmark secondaire AR".into(),
        ));
    }
    let device = metal::Device::system_default()
        .ok_or_else(|| InferError::Metal("aucun device pour le harnais".into()))?;
    let counter_sets: Vec<String> = device
        .counter_sets()
        .iter()
        .map(|set| set.name().to_owned())
        .collect();
    eprintln!(
        "pairwise {}",
        json!({"phase":"start","ids":ids,"action":action,"budget":budget,"prompt_tokens":prompt.len(),"control_mode":control_mode,"reverse_order":reverse_order,"harness_version":4,"counter_sets":counter_sets,"device":device.name()})
    );
    let base = Case::default();
    let oracle = one(
        decoder,
        prompt,
        options,
        Case {
            mode: Mode::Ar,
            ..base
        },
        budget,
        false,
        None,
        &device,
    )?;
    if oracle.output.tokens.len() != budget {
        return Err(InferError::Config(
            "oracle AR arrêté avant le budget de mesure".into(),
        ));
    }
    report(&oracle, "AR", "oracle", 0, "reference");
    let mut baseline_stats = None;
    let mut inactive = Vec::new();
    for row in &selected {
        for case in [control, row.case] {
            let warm = one(decoder, prompt, options, case, 4, false, None, &device)?;
            exact(&warm, &oracle.output.tokens[..4], "warmup")?;
        }
        if action == "bench" {
            let mut candidate_stats = None;
            for pair in 0..pairs {
                for candidate in benchmark_order(pair, reverse_order) {
                    let case = if candidate { row.case } else { control };
                    let run = one(decoder, prompt, options, case, budget, false, None, &device)?;
                    exact(&run, &oracle.output.tokens, &row.id)?;
                    let stats = if candidate {
                        &mut candidate_stats
                    } else {
                        &mut baseline_stats
                    };
                    if let Some(previous) = stats {
                        if *previous != run.output.stats {
                            return Err(InferError::Config(format!("stats instables {}", row.id)));
                        }
                    }
                    *stats = Some(run.output.stats.clone());
                    report(
                        &run,
                        &row.id,
                        "bench",
                        pair + 1,
                        if candidate { "candidate" } else { "control" },
                    );
                }
            }
        } else if action == "edges" {
            for limit in [0, 1, 2, 3, 4, 9] {
                let run = one(
                    decoder, prompt, options, row.case, limit, false, None, &device,
                )?;
                exact(&run, &oracle.output.tokens[..limit], &row.id)?;
                report(&run, &row.id, "budget", limit, "candidate");
            }
            for at in 0..4 {
                let mut stopped = options.clone();
                let eos = oracle.output.tokens[at];
                stopped.stop_token_ids.push(eos);
                let end = oracle
                    .output
                    .tokens
                    .iter()
                    .position(|t| *t == eos)
                    .ok_or_else(|| InferError::Config("EOS oracle absent".into()))?;
                let run = one(decoder, prompt, &stopped, row.case, 9, false, None, &device)?;
                exact(&run, &oracle.output.tokens[..end], &row.id)?;
                report(&run, &row.id, "eos", at, "candidate");
            }
            for at in 1..=4 {
                let run = one(
                    decoder,
                    prompt,
                    options,
                    row.case,
                    9,
                    true,
                    Some(at),
                    &device,
                )?;
                if run.seen != oracle.output.tokens[..at]
                    || !oracle.output.tokens.starts_with(&run.output.tokens)
                {
                    return Err(InferError::Config(format!(
                        "callback après annulation ou tokens divergents {}",
                        row.id
                    )));
                }
                report(&run, &row.id, "cancel", at, "candidate");
            }
        } else {
            let run = one(
                decoder, prompt, options, row.case, budget, true, None, &device,
            )?;
            exact(&run, &oracle.output.tokens, &row.id)?;
            if run.seen != run.output.tokens {
                return Err(InferError::Config(
                    "stream différent des tokens retournés".into(),
                ));
            }
            report(&run, &row.id, &action, 0, "candidate");
            if let Err(error) = routes(&run, row.case, &row.id) {
                eprintln!(
                    "pairwise {}",
                    json!({"phase":"case_inactive","id":row.id,"action":action,"error":error.to_string()})
                );
                inactive.push(row.id.as_str());
                continue;
            }
        }
        eprintln!(
            "pairwise {}",
            json!({"phase":"case_complete","id":row.id,"action":action})
        );
    }
    if !inactive.is_empty() {
        return Err(InferError::Config(format!(
            "pairwise chemins non activés : {inactive:?}"
        )));
    }
    eprintln!(
        "pairwise {}",
        json!({"phase":"complete","ids":ids,"action":action,"pairs":pairs})
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unmeasured_resources_are_null_not_zero() {
        let mut run = Run {
            instrumented: false,
            output: SpeculativeOutput {
                tokens: vec![1],
                stats: SpeculativeStats::default(),
                loop_duration: Duration::from_millis(1),
            },
            total: Duration::from_millis(2),
            seen: vec![],
            arrivals: vec![],
            sampled_metal_max: 0,
            profile: [0; 4],
            hits: [0; 7],
            io: [0; 4],
        };
        let value = measurement(&run, "PW00", "bench", 1, "control");
        for key in [
            "profile_including_prefill",
            "hits",
            "tracked_shared_copy_calls_bytes_including_prefill",
            "callback_arrivals_ns",
            "sampled_live_metal_max_not_peak",
        ] {
            assert!(
                value[key].is_null(),
                "{key} must be unknown outside an instrumented run"
            );
        }
        run.instrumented = true;
        assert_eq!(
            measurement(&run, "PW00", "metrics", 0, "control")["profile_including_prefill"],
            json!([0, 0, 0, 0])
        );
        run.hits[4] = 1;
        assert!(routes(&run, Case::default(), "PW00").is_ok());
        let p3 = Case {
            p3: true,
            ..Case::default()
        };
        assert!(routes(&run, p3, "inactive").is_err());
        run.hits[6] = 1;
        assert!(routes(&run, p3, "active").is_ok());
    }

    #[test]
    fn single_pair_batches_can_counterbalance_order() {
        assert_eq!(benchmark_order(0, false), [false, true]);
        assert_eq!(benchmark_order(0, true), [true, false]);
        assert_eq!(benchmark_order(1, false), [true, false]);
        assert_eq!(benchmark_order(1, true), [false, true]);
    }

    #[test]
    fn secondary_ar_control_never_replaces_the_mtp_reference() {
        let mtp = Row {
            id: "D1".into(),
            case: Case::default(),
        };
        let ar = Row {
            id: "AR".into(),
            case: Case {
                mode: Mode::Ar,
                ..Case::default()
            },
        };
        assert_eq!(
            benchmark_control("d1", &[&mtp, &ar]).expect("invariant: D1 control"),
            Case::default()
        );
        assert_eq!(
            benchmark_control("ar", &[&ar])
                .expect("invariant: secondary AR control")
                .mode,
            Mode::Ar
        );
        assert!(benchmark_control("ar", &[&mtp]).is_err());
        assert!(benchmark_control("ar", &[]).is_err());
        assert!(benchmark_control("fresh", &[&mtp]).is_err());
    }

    #[test]
    fn matrix_schema_and_admissibility_are_explicit() {
        let row: Row = serde_json::from_str(r#"{"id":"PW00","mode":"d1","hidden":false,"embedding":false,"overlap":false,"p1":true,"p2":false,"p3":false,"status":"planned"}"#)
            .expect("invariant: fixture JSON");
        assert_eq!(row.case, Case::default());
        assert!(Case {
            mode: Mode::Ar,
            hidden: true,
            ..row.case
        }
        .validate()
        .is_err());
        assert!(Case {
            mode: Mode::Block2,
            ..row.case
        }
        .validate()
        .is_err());
        assert!(Case {
            mode: Mode::Block2,
            embedding: true,
            ..row.case
        }
        .validate()
        .is_ok());
        assert!(serde_json::from_str::<Row>(r#"{"id":"PW00","mode":"inconnu"}"#).is_err());
    }
}
