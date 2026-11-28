//! Identité au bit entre un prefill à froid et l'extension d'un préfixe en
//! cache (cache de préfixe du décodeur, états de prompt de serve).

use super::*;

fn max_abs_gap(left: &[f32], right: &[f32]) -> f32 {
    if left.len() != right.len() {
        return f32::INFINITY;
    }
    left.iter()
        .zip(right)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

fn top_margin(logits: &[f32]) -> f32 {
    let (mut best, mut second) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for &value in logits {
        if value > best {
            second = best;
            best = value;
        } else if value > second {
            second = value;
        }
    }
    best - second
}

fn first_divergence(left: &[usize], right: &[usize]) -> Option<usize> {
    left.iter()
        .zip(right)
        .position(|(a, b)| a != b)
        .or_else(|| (left.len() != right.len()).then(|| left.len().min(right.len())))
}

/// Exige l'identité au bit entre un prefill à froid de `P + S` et le prefill
/// de `P` prolongé de `S` par le cache de préfixe du décodeur, en AR (état
/// final, logits, greedy), MTP et DFlash (greedy).
/// Cas `p:s` dans `SARAGOSSA_RUST_PREFIX_ALIGN_CASES`, modes dans
/// `SARAGOSSA_RUST_PREFIX_ALIGN_MODES` (`ar,etats,mtp,dflash` ; `etats` :
/// primitive d'extension d'un état de prompt, tenue à l'identité seulement
/// alignée).
#[test]
#[ignore = "modèle réel requis"]
#[cfg(all(gpu_metal, feature = "devtools"))]
fn prefix_extension_matches_cold_prefill_real_model() -> Result<()> {
    let Some(model_dir) = crate::runtime_flags::env_var_os("SARAGOSSA_RUST_PREFIX_ALIGN_MODEL")
        .map(std::path::PathBuf::from)
        .filter(|path| path.is_dir())
    else {
        eprintln!("skip: SARAGOSSA_RUST_PREFIX_ALIGN_MODEL absent");
        return Ok(());
    };
    let env = |name: &str, default: &str| {
        crate::runtime_flags::env_var(name).unwrap_or_else(|| default.to_string())
    };
    let cases: Vec<(usize, usize)> = env(
        "SARAGOSSA_RUST_PREFIX_ALIGN_CASES",
        "1000:20,1037:20,1024:20,1000:100,1037:100,1024:100,1040:100,1037:40,1024:64,2000:5,3000:1200",
    )
    .split(',')
    .filter_map(|case| {
        let (p, s) = case.trim().split_once(':')?;
        Some((p.parse().ok()?, s.parse().ok()?))
    })
    .collect();
    let modes = env("SARAGOSSA_RUST_PREFIX_ALIGN_MODES", "ar,etats");
    let texts = env(
        "SARAGOSSA_RUST_PREFIX_ALIGN_TEXTS",
        "code4k-1,code4k-2,code4k-3",
    );
    let new_tokens: usize = env("SARAGOSSA_RUST_PREFIX_ALIGN_TOKENS", "48")
        .parse()
        .unwrap_or(48);

    let assets = crate::ModelAssets::load_local(&model_dir)?;
    // Chargement de production (poids bf16 gardés en bf16) : l'alignement
    // automatique dépend des projections réellement chargées.
    let executor = crate::MetalExecutor::new()?;
    let mut model =
        crate::load_causal_decoder_for_metal(&assets, &executor)?.with_metal_executor(executor)?;
    if modes.contains("mtp") || modes.contains("dflash") {
        let _path = assets
            .mtp
            .path
            .clone()
            .ok_or_else(|| InferError::Config("tête MTP absente".into()))?;
        model = model.with_mtp_weights(&assets.mtp)?;
    }
    let draft = if modes.contains("dflash") {
        model = model.with_dflash_draft_from_env(&assets.config)?;
        model.dflash.take()
    } else {
        None
    };
    let options = GenerationOptions {
        temperature: 0.0,
        top_p: 1.0,
        ..GenerationOptions::default()
    };
    let fixtures =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/rowsexact");
    let mut failures = Vec::new();
    println!("alignement={:?}", model.prefix_resume_align());
    println!("mode\ttexte\tP\tS\tfinal_max\tlogits_max\tkv_couche\tdiv_token");
    for name in texts.split(',').map(str::trim) {
        // `a+b` : textes concaténés, pour dépasser un morceau de prefill.
        let text = name
            .split('+')
            .map(|part| {
                std::fs::read_to_string(fixtures.join(format!("{part}.txt")))
                    .map_err(|error| InferError::Config(format!("{part}: {error}")))
            })
            .collect::<Result<Vec<_>>>()?
            .join("\n\n");
        let ids = assets
            .encode_prompt(&crate::render_qwen_chatml(
                &[crate::ChatTemplateMessage::new("user", &text)],
                true,
                false,
            ))?
            .into_iter()
            .map(|id| id as usize)
            .collect::<Vec<_>>();
        for &(p, s) in &cases {
            let Some(full) = ids.get(..p + s) else {
                continue;
            };
            let aligned = model
                .prefix_resume_align()
                .is_some_and(|(align, tail)| p % align == 0 && s >= tail);
            if modes.contains("ar") {
                // Cache de préfixe du décodeur : prefill à froid de P (ancre
                // comprise) puis hit prolongé jusqu'à P + S, contre un prefill à
                // froid d'un seul tenant, sans cache.
                let cold = model.prefill_prompt_state_uncached(full)?;
                model.clear_prefix_cache();
                let split = model.prefill_cache_state(full)?;
                model.clear_prefix_cache();
                model.prefill_cache_state(&full[..p])?;
                let warm = model.prefill_cache_state(full)?;
                // Hit exact : l'entrée du prompt complet, sans queue recalculée.
                let (exact_cache, exact_state, exact_reused) =
                    model.prefill_cache_state_reused(full)?;
                model.clear_prefix_cache();
                let exact_gap = max_abs_gap(cold.final_state.data(), exact_state.data());
                let exact_kv = (0..model.layers.len()).all(|index| {
                    exact_cache.layer_kv_for_test(index) == cold.cache.layer_kv_for_test(index)
                });
                if exact_reused != full.len() || exact_gap != 0.0 || !exact_kv {
                    failures.push(format!(
                        "ar exact {name} {p}:{s} repris={exact_reused} écart={exact_gap:e}"
                    ));
                }
                let split_gap = max_abs_gap(cold.final_state.data(), split.1.data());
                let final_gap = max_abs_gap(cold.final_state.data(), warm.1.data());
                let logits_gap = max_abs_gap(
                    model.logits_from_final_state(&cold.final_state)?.data(),
                    model.logits_from_final_state(&warm.1)?.data(),
                );
                let cold_logits = model.logits_from_final_state(&cold.final_state)?;
                let warm = CausalDecoderPromptState::new(warm.0, warm.1);
                let cold_tokens = model
                    .generate_greedy_timed_from_prompt_state_with_options(
                        cold,
                        Duration::ZERO,
                        new_tokens,
                        &options,
                    )?
                    .tokens;
                let warm_tokens = model
                    .generate_greedy_timed_from_prompt_state_with_options(
                        warm,
                        Duration::ZERO,
                        new_tokens,
                        &options,
                    )?
                    .tokens;
                let divergence = first_divergence(&cold_tokens, &warm_tokens);
                // Marge top-1/top-2 du froid au token divergent : une
                // quasi-égalité, ou un vrai changement de choix ?
                let margin = match divergence {
                    None => None,
                    Some(0) => Some(top_margin(cold_logits.data())),
                    Some(index) => model
                        .resident_forced_logits_for_test(full, &cold_tokens, false)?
                        .get(index - 1)
                        .map(|row| top_margin(row)),
                };
                println!(
                    "ar\t{name}\t{p}\t{s}\t{final_gap:e}\t{logits_gap:e}\tfroid_scindé={split_gap:e}\t{divergence:?}\tmarge={margin:?}"
                );
                if final_gap != 0.0 || split_gap != 0.0 || divergence.is_some() {
                    failures.push(format!("ar {name} {p}:{s}"));
                }
            }
            if modes.contains("etats") {
                // Primitive d'état de prompt (serve) : prolonge exactement P.
                let cold = model.prefill_prompt_state_uncached(full)?;
                let mut warm = model.prefill_prompt_state_uncached(&full[..p])?;
                model.extend_prompt_state(&mut warm, &full[p..])?;
                let final_gap = max_abs_gap(cold.final_state.data(), warm.final_state.data());
                let logits_gap = max_abs_gap(
                    model.logits_from_final_state(&cold.final_state)?.data(),
                    model.logits_from_final_state(&warm.final_state)?.data(),
                );
                let kv_layer = (0..model.layers.len()).find(|&index| {
                    match (
                        cold.cache.layer_kv_for_test(index),
                        warm.cache.layer_kv_for_test(index),
                    ) {
                        (Some((ck, cv, _)), Some((wk, wv, _))) => {
                            !ck.is_empty() && (ck != wk || cv != wv)
                        }
                        _ => false,
                    }
                });
                let cold_tokens = model
                    .generate_greedy_timed_from_prompt_state_with_options(
                        cold,
                        Duration::ZERO,
                        new_tokens,
                        &options,
                    )?
                    .tokens;
                let warm_tokens = model
                    .generate_greedy_timed_from_prompt_state_with_options(
                        warm,
                        Duration::ZERO,
                        new_tokens,
                        &options,
                    )?
                    .tokens;
                let divergence = first_divergence(&cold_tokens, &warm_tokens);
                println!(
                    "etats\t{name}\t{p}\t{s}\t{final_gap:e}\t{logits_gap:e}\t{kv_layer:?}\t{divergence:?}"
                );
                // Hors alignement, la primitive n'est pas tenue à l'identité :
                // serve et le cache de préfixe ne la prolongent qu'alignée.
                if aligned && (final_gap != 0.0 || divergence.is_some()) {
                    failures.push(format!("etats {name} {p}:{s}"));
                }
            }
            for (mode, with_draft) in [("mtp", false), ("dflash", true)] {
                if !modes.contains(mode) || (with_draft && draft.is_none()) {
                    continue;
                }
                model.dflash = if with_draft { draft.clone() } else { None };
                let run = |prompt: &[usize], max_new: usize| {
                    model
                        .generate_greedy_mtp_batched_with_options(prompt, max_new, &options, 2)
                        .map(|output| output.tokens)
                };
                model.clear_prefix_cache();
                let cold_tokens = run(full, new_tokens)?;
                model.clear_prefix_cache();
                run(&full[..p], 1)?;
                // Entrée reprise par la requête prolongée : sans elle, le run
                // « chaud » serait un second prefill à froid.
                let reusable = model
                    .prefix_cache
                    .lock()
                    .map_err(|_| InferError::Config("cache préfixe empoisonné".into()))?
                    .entries
                    .iter()
                    .filter(|entry| {
                        entry.tokens.len() < full.len() && full.starts_with(&entry.tokens)
                    })
                    .map(|entry| entry.tokens.len())
                    .max();
                let warm_tokens = run(full, new_tokens)?;
                // Hit exact sur le prompt complet.
                let exact_tokens = run(full, new_tokens)?;
                model.clear_prefix_cache();
                if first_divergence(&cold_tokens, &exact_tokens).is_some() {
                    failures.push(format!("{mode} exact {name} {p}:{s}"));
                }
                let divergence = first_divergence(&cold_tokens, &warm_tokens);
                println!("{mode}\t{name}\t{p}\t{s}\tentrée={reusable:?}\t-\t-\t{divergence:?}");
                if reusable.is_none() {
                    failures.push(format!("{mode} {name} {p}:{s} sans entrée"));
                }
                if divergence.is_some() {
                    failures.push(format!("{mode} {name} {p}:{s}"));
                }
            }
            model.dflash = None;
        }
    }
    println!("écarts: {} ({failures:?})", failures.len());
    if !failures.is_empty() {
        return Err(InferError::Config(format!(
            "extensions ≠ froid: {failures:?}"
        )));
    }
    Ok(())
}
