//! Interface CLI du moteur d'inférence Rust.

#![deny(unsafe_code)]

mod bench_serve;
mod hf_resolve;
mod run_repl;
mod serve;

use std::env;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use saragossa::{
    load_causal_decoder, render_gemma4_chat, render_gemma_chat, render_qwen_chatml,
    verify_decoder_contract, CausalDecoder, ChatTemplateMessage, DecoderContract,
    GenerationOptions, ModelAssets,
};

mod doctor_bench;

type CliResult<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Debug)]
struct CliArgs {
    model_dir: PathBuf,
    prompt: Option<String>,
    prompt_tokens: Option<usize>,
    prompt_b: Option<String>,
    max_tokens: usize,
    temperature: f32,
    seed: u64,
    check_only: bool,
    load_only: bool,
    prefill_only: bool,
    prefill_repeat: Option<usize>,
    top_k: usize,
    top_p: f32,
    top_k_explicit: bool,
    top_p_explicit: bool,
    metrics: bool,
    raw: bool,
    ignore_stop_tokens: bool,
    backend: RuntimeKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RuntimeKind {
    Cpu,
    Metal,
}

#[derive(Debug)]
struct CliError(String);

impl Display for CliError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for CliError {}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("saragossa: {error}");
            ExitCode::from(2)
        }
    }
}

fn run() -> CliResult<()> {
    let raw_args: Vec<String> = env::args().skip(1).collect();
    if let Some(command) = raw_args.first() {
        match command.as_str() {
            "doctor" => return doctor_bench::run_doctor(raw_args.into_iter().skip(1)),
            "bench" => return doctor_bench::run_bench(raw_args.into_iter().skip(1)),
            "bench-serve" => return bench_serve::run(raw_args.into_iter().skip(1)),
            "list" => return hf_resolve::run_list(raw_args.into_iter().skip(1)),
            "run" => return run_repl::run(raw_args.into_iter().skip(1)),
            "serve" => return serve::run(raw_args.into_iter().skip(1)),
            _ if command.starts_with('-') => {}
            other => {
                print_help();
                return Err(cli_error(format!("commande inconnue: {other}")));
            }
        }
    }
    if raw_args
        .iter()
        .any(|arg| matches!(arg.as_str(), "--help" | "-h"))
    {
        print_help();
        return Ok(());
    }
    let mut args = CliArgs::parse(raw_args)?;
    if let Some(preset) = saragossa::runtime_preset_for_model_dir(&args.model_dir) {
        args.apply_generation_preset(preset);
        let _ = saragossa::apply_runtime_preset_for_model_dir(&args.model_dir);
    }
    let assets = ModelAssets::load_local(&args.model_dir)?;
    if args.check_only {
        let contract = verify_decoder_contract(&assets)?;
        print_contract(&contract);
        if let Some(draft_dir) = env::var_os("RETI_RUST_DFLASH_DRAFT_DIR") {
            let draft = saragossa::devtools::load_dflash_draft_weights_for_target(
                &assets.config,
                PathBuf::from(draft_dir),
            )?;
            saragossa::devtools::print_dflash_draft(&draft);
        }
        return Ok(());
    }
    let load_started = Instant::now();
    let mut decoder = load_decoder_with_runtime(&assets, args.backend)?;
    if saragossa::devtools::mtp_acceptance_enabled() {
        let path = assets.mtp.path.as_ref().ok_or_else(|| {
            cli_error("RETI_RUST_MTP_ACCEPTANCE=1 mais aucun sidecar MTP détecté")
        })?;
        decoder = decoder.with_mtp_sidecar(path)?;
        if let Some(path) = env::var_os("RETI_RUST_MTP_DRAFT_LM_HEAD") {
            decoder = decoder.with_mtp_draft_lm_head_sidecar(PathBuf::from(path))?;
        }
    }
    let load_elapsed = load_started.elapsed();
    if args.load_only {
        println!("ok loaded");
        return Ok(());
    }
    let generated_prompt;
    let prompt = if let Some(target_tokens) = args.prompt_tokens {
        generated_prompt = generate_benchmark_prompt(&assets, target_tokens, args.raw)?;
        generated_prompt.as_str()
    } else {
        args.prompt.as_deref().ok_or_else(|| {
            cli_error("--prompt ou --prompt-tokens requis hors --check/--load-only")
        })?
    };
    let prompt_ids = encode_prompt_ids(&assets, prompt, args.raw)?;
    let warmup_elapsed = warmup_decoder(&decoder, &prompt_ids, args.backend)?;
    if saragossa::devtools::lightbatch_acceptance_enabled() {
        let prompt_b = args
            .prompt_b
            .as_ref()
            .ok_or_else(|| cli_error("RETI_RUST_LIGHTBATCH_ACCEPTANCE=1 requiert --prompt-b"))?;
        let prompt_b_ids = encode_prompt_ids(&assets, prompt_b, args.raw)?;
        let options = generation_options(&args, &assets, args.top_p);
        return Ok(saragossa::devtools::run_lightbatch_acceptance(
            &assets,
            &decoder,
            &prompt_ids,
            &prompt_b_ids,
            args.max_tokens,
            &options,
            load_elapsed,
            warmup_elapsed,
        )?);
    }
    if args.prefill_only {
        // Bench serveur-chaud : N prefills réels, sans prefix-cache, dans le
        // même process après un seul load/warmup. `RETI_RUST_PREFILL_ITERS`
        // reste accepté pour rejouer les anciens protocoles.
        let prefill_repeat = args.prefill_repeat.or_else(env_prefill_repeat);
        if let Some(repeat) = prefill_repeat {
            println!(
                "prefill_repeat prompt_tokens={} load_ms={} warmup_ms={} repeat={}",
                prompt_ids.len(),
                load_elapsed.as_millis(),
                warmup_elapsed.as_millis(),
                repeat
            );
            for pass in 0..repeat {
                let started = Instant::now();
                let (_, _logits) = decoder.prefill_cache_uncached(&prompt_ids)?;
                let elapsed = started.elapsed();
                let seconds = elapsed.as_secs_f64();
                let tok_s = if seconds > 0.0 {
                    prompt_ids.len() as f64 / seconds
                } else {
                    0.0
                };
                println!(
                    "prefill_repeat pass={pass} prefill_ms={} prefill_tok_s={tok_s:.1}",
                    elapsed.as_millis()
                );
            }
            return Ok(());
        }
        #[cfg(all(target_os = "macos", feature = "metal"))]
        let profile_snapshot = prefill_profile_enabled().then(|| {
            (
                saragossa::metal_backend::decode_profile_snapshot(),
                saragossa::metal_backend::decode_profile_dispatch_sites_snapshot(),
            )
        });
        let prefill_started = Instant::now();
        let (_, logits) = decoder.prefill_cache(&prompt_ids)?;
        let prefill_elapsed = prefill_started.elapsed();
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if let Some(((cb0, wait0, read0, dispatch0), site0)) = profile_snapshot {
            let (cb1, wait1, read1, dispatch1) =
                saragossa::metal_backend::decode_profile_snapshot();
            eprintln!(
                "prefill profile total_ms={} wait_ms={} read_ms={} cmd_buffers={} dispatches={}",
                prefill_elapsed.as_millis(),
                wait1.saturating_sub(wait0) / 1_000_000,
                read1.saturating_sub(read0) / 1_000_000,
                cb1.saturating_sub(cb0),
                dispatch1.saturating_sub(dispatch0)
            );
            let mut deltas = saragossa::metal_backend::decode_profile_dispatch_sites_snapshot()
                .into_iter()
                .filter_map(|(site, count1)| {
                    let count0 = site0.get(&site).copied().unwrap_or(0);
                    let delta = count1.saturating_sub(count0);
                    (delta > 0).then_some((site, delta))
                })
                .collect::<Vec<_>>();
            deltas.sort_by(|(_, left), (_, right)| right.cmp(left));
            for (site, count) in deltas.into_iter().take(16) {
                eprintln!(
                    "prefill dispatch_site count={count} at {}:{}:{}",
                    compact_source_path(site.file),
                    site.line,
                    site.column
                );
            }
        }
        saragossa::metal_backend::dump_commit_components();
        let row = logits.as_row()?;
        println!(
            "ok prefill tokens={} vocab={} load_ms={} warmup_ms={} prefill_ms={} top_k={}",
            prompt_ids.len(),
            row.len(),
            load_elapsed.as_millis(),
            warmup_elapsed.as_millis(),
            prefill_elapsed.as_millis(),
            format_top_k(row, args.top_k)
        );
        return Ok(());
    }
    if saragossa::devtools::mtp_acceptance_enabled() {
        let options = generation_options(&args, &assets, 1.0);
        return Ok(saragossa::devtools::run_mtp_acceptance(
            &assets,
            &decoder,
            &prompt_ids,
            args.max_tokens,
            &options,
            load_elapsed,
            warmup_elapsed,
        )?);
    }
    if saragossa::devtools::dflash_acceptance_enabled() {
        let draft_dir = env::var_os("RETI_RUST_DFLASH_DRAFT_DIR").ok_or_else(|| {
            cli_error("RETI_RUST_DFLASH_ACCEPTANCE=1 requiert RETI_RUST_DFLASH_DRAFT_DIR")
        })?;
        let draft = saragossa::devtools::load_dflash_draft_weights_for_target(
            &assets.config,
            PathBuf::from(draft_dir),
        )?;
        let options = generation_options(&args, &assets, 1.0);
        return Ok(saragossa::devtools::run_dflash_acceptance(
            &assets,
            &decoder,
            &draft,
            &prompt_ids,
            args.max_tokens,
            &options,
            load_elapsed,
            warmup_elapsed,
        )?);
    }
    #[cfg(all(target_os = "macos", feature = "metal"))]
    if saragossa::devtools::resident_linear_xray_enabled() {
        let options = generation_options(&args, &assets, args.top_p);
        return Ok(saragossa::devtools::run_resident_linear_xray(
            &assets,
            &decoder,
            &prompt_ids,
            &options,
        )?);
    }
    #[cfg(all(target_os = "macos", feature = "metal"))]
    let decode_profile_snapshot = prefill_profile_enabled().then(|| {
        (
            saragossa::metal_backend::decode_profile_snapshot(),
            saragossa::metal_backend::decode_profile_dispatch_shapes_snapshot(),
        )
    });
    let generate_started = Instant::now();
    let output = decoder.generate_greedy_timed_with_options(
        &prompt_ids,
        args.max_tokens,
        &generation_options(&args, &assets, args.top_p),
    )?;
    let generate_elapsed = generate_started.elapsed();
    let timings = output.timings;
    let generated = output.tokens;
    let generated_len = generated.len();
    let generated = generated
        .into_iter()
        .map(|id| {
            u32::try_from(id).map_err(|_| cli_error(format!("token généré hors plage: {id}")))
        })
        .collect::<CliResult<Vec<_>>>()?;
    let text = assets.decode_tokens(&generated, true)?;
    println!("{}", text.trim());
    // Diagnostic oracle : dump des IDs générés pour un diff token-à-token exact
    // entre deux configurations (ex. chemin NA ON vs OFF). Env-gated, hors chemin prod.
    if env::var_os("RETI_RUST_ORACLE_DUMP_IDS").is_some() {
        let ids = generated
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(",");
        eprintln!("oracle_ids {ids}");
    }
    if args.metrics {
        let seconds = generate_elapsed.as_secs_f64();
        let tok_s = if seconds > 0.0 {
            generated_len as f64 / seconds
        } else {
            0.0
        };
        let decode_seconds = timings.decode.as_secs_f64();
        let decode_tok_s = if decode_seconds > 0.0 {
            timings.decode_tokens as f64 / decode_seconds
        } else {
            0.0
        };
        let prefill_seconds = timings.prefill.as_secs_f64();
        let prefill_tok_s = if prefill_seconds > 0.0 {
            prompt_ids.len() as f64 / prefill_seconds
        } else {
            0.0
        };
        eprintln!(
            "metrics prompt_tokens={} tokens={} load_ms={} warmup_ms={} prefill_ms={} prefill_tok_s={:.3} decode_tokens={} decode_ms={} generate_ms={} tok_s={:.3} decode_tok_s={:.3}",
            prompt_ids.len(),
            generated_len,
            load_elapsed.as_millis(),
            warmup_elapsed.as_millis(),
            timings.prefill.as_millis(),
            prefill_tok_s,
            timings.decode_tokens,
            timings.decode.as_millis(),
            generate_elapsed.as_millis(),
            tok_s,
            decode_tok_s
        );
    }
    #[cfg(all(target_os = "macos", feature = "metal"))]
    if let Some(((cb0, wait0, read0, dispatch0), shape0)) = decode_profile_snapshot {
        let (cb1, wait1, read1, dispatch1) = saragossa::metal_backend::decode_profile_snapshot();
        eprintln!(
            "decode profile total_ms={} wait_ms={} read_ms={} cmd_buffers={} dispatches={}",
            generate_elapsed.as_millis(),
            wait1.saturating_sub(wait0) / 1_000_000,
            read1.saturating_sub(read0) / 1_000_000,
            cb1.saturating_sub(cb0),
            dispatch1.saturating_sub(dispatch0)
        );
        let mut deltas = saragossa::metal_backend::decode_profile_dispatch_shapes_snapshot()
            .into_iter()
            .filter_map(|(shape, count1)| {
                let count0 = shape0.get(&shape).copied().unwrap_or(0);
                let delta = count1.saturating_sub(count0);
                (delta > 0).then_some((shape, delta))
            })
            .collect::<Vec<_>>();
        deltas.sort_by(|(_, left), (_, right)| right.cmp(left));
        for (shape, count) in deltas.into_iter().take(24) {
            eprintln!(
                "decode dispatch_shape count={} kind={} batch={} lhs_rows={} topk={} in_dim={} out_dim={} group_size={} bits={}",
                count,
                shape.kind,
                shape.batch,
                shape.lhs_rows,
                shape.topk,
                shape.in_dim,
                shape.out_dim,
                shape.group_size,
                shape.bits
            );
        }
    }
    Ok(())
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn compact_source_path(path: &'static str) -> &'static str {
    path.strip_prefix(concat!(env!("CARGO_MANIFEST_DIR"), "/"))
        .unwrap_or(path)
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn prefill_profile_enabled() -> bool {
    saragossa::runtime_flags::env_flag("RETI_RUST_DECODE_PROFILE", false)
}

/// Encode un prompt en ids. `--raw` : complétion brute (pas de template de
/// chat), utile pour tester un modèle hors familles connues (Llama/Mistral) —
/// tokens spéciaux du tokenizer activés (BOS Llama). Gemma 3 : tours
/// `<start_of_turn>` ; Gemma 4 : tours `<|turn>` avec `<bos>` littéral.
/// Sinon : ChatML Qwen, byte-identique.
fn encode_prompt_ids(assets: &ModelAssets, prompt: &str, raw: bool) -> CliResult<Vec<usize>> {
    if raw {
        assets.encode_prompt_with_special(prompt)?
    } else if assets.config.is_gemma4() {
        let templated =
            render_gemma4_chat(&[ChatTemplateMessage::new("user", prompt)], true, false);
        assets.encode_prompt(&templated)?
    } else if assets.config.is_gemma() {
        let templated = render_gemma_chat(&[ChatTemplateMessage::new("user", prompt)], true);
        assets.encode_prompt_with_special(&templated)?
    } else {
        let templated =
            render_qwen_chatml(&[ChatTemplateMessage::new("user", prompt)], true, false);
        assets.encode_prompt(&templated)?
    }
    .into_iter()
    .map(|id| usize::try_from(id).map_err(|_| cli_error(format!("token id hors plage: {id}"))))
    .collect::<CliResult<Vec<_>>>()
}

fn generation_options(args: &CliArgs, assets: &ModelAssets, top_p: f32) -> GenerationOptions {
    let mut options = GenerationOptions {
        temperature: args.temperature,
        top_p,
        top_k: args.top_k,
        seed: args.seed,
        ..GenerationOptions::default()
    };
    if !args.ignore_stop_tokens {
        options.stop_token_ids = assets.stop_token_ids();
    }
    options
}

fn generate_benchmark_prompt(
    assets: &ModelAssets,
    target_tokens: usize,
    raw: bool,
) -> CliResult<String> {
    if target_tokens == 0 {
        return Err(cli_error("--prompt-tokens doit être > 0"));
    }
    let unique_prefix = "BENCH-reti ";
    let filler = "The quick brown fox jumps over the lazy dog. \
In the realm of artificial intelligence, large language models \
have demonstrated remarkable capabilities across diverse tasks. ";
    let mut text = format!("{}{}", unique_prefix, filler.repeat(target_tokens / 10 + 1));
    let mut tokens = encode_benchmark_text(assets, &text, raw)?;
    if tokens.len() < target_tokens {
        text = format!("{}{}", unique_prefix, filler.repeat(target_tokens / 5 + 1));
        tokens = encode_benchmark_text(assets, &text, raw)?;
    }
    tokens.truncate(target_tokens);
    let token_ids = tokens
        .into_iter()
        .map(|id| u32::try_from(id).map_err(|_| cli_error(format!("token id hors plage: {id}"))))
        .collect::<CliResult<Vec<_>>>()?;
    Ok(assets.decode_tokens(&token_ids, false)?)
}

fn encode_benchmark_text(assets: &ModelAssets, text: &str, raw: bool) -> CliResult<Vec<usize>> {
    let ids = if raw {
        assets.encode_prompt_with_special(text)?
    } else {
        assets.encode_prompt(text)?
    };
    ids.into_iter()
        .map(|id| usize::try_from(id).map_err(|_| cli_error(format!("token id hors plage: {id}"))))
        .collect::<CliResult<Vec<_>>>()
}

fn warmup_decoder(
    decoder: &CausalDecoder,
    prompt_ids: &[usize],
    backend: RuntimeKind,
) -> CliResult<Duration> {
    if backend != RuntimeKind::Metal || !warmup_enabled() {
        return Ok(Duration::ZERO);
    }
    if prompt_ids.is_empty() {
        return Err(cli_error("prompt token vide"));
    }
    let started = Instant::now();
    let passes = warmup_passes();
    let warmup_len = prompt_ids.len().min(warmup_prompt_tokens());
    let warmup_prompt = &prompt_ids[..warmup_len];
    for _ in 0..passes {
        let _ = decoder.prefill_cache_uncached(warmup_prompt)?;
    }
    Ok(started.elapsed())
}

fn warmup_enabled() -> bool {
    !env::var("RETI_RUST_WARMUP").is_ok_and(|value| {
        value == "0" || value.eq_ignore_ascii_case("false") || value.eq_ignore_ascii_case("off")
    })
}

fn warmup_passes() -> usize {
    env::var("RETI_RUST_WARMUP_PASSES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|passes| *passes > 0)
        .unwrap_or(2)
}

fn warmup_prompt_tokens() -> usize {
    env::var("RETI_RUST_WARMUP_PROMPT_TOKENS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|tokens| *tokens > 0)
        .unwrap_or(32)
}

fn env_prefill_repeat() -> Option<usize> {
    env::var("RETI_RUST_PREFILL_ITERS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|repeat| *repeat > 1)
}

impl CliArgs {
    fn parse(args: impl IntoIterator<Item = String>) -> CliResult<Self> {
        let mut model_dir = None;
        let mut prompt = None;
        let mut prompt_tokens = None;
        let mut prompt_b = None;
        let mut max_tokens = 32_usize;
        let mut temperature = 0.0_f32;
        let mut seed = 0_u64;
        let mut check_only = false;
        let mut load_only = false;
        let mut prefill_only = false;
        let mut prefill_repeat = None;
        let mut top_k = 0_usize;
        let mut top_p = 1.0_f32;
        let mut top_k_explicit = false;
        let mut top_p_explicit = false;
        let mut metrics = false;
        let mut raw = false;
        let mut ignore_stop_tokens = false;
        let mut backend = RuntimeKind::default_backend();
        let mut iter = args.into_iter();

        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--model-dir" => model_dir = Some(next_value(&mut iter, "--model-dir")?.into()),
                "--prompt" => prompt = Some(next_value(&mut iter, "--prompt")?),
                "--prompt-tokens" => {
                    prompt_tokens = Some(next_value(&mut iter, "--prompt-tokens")?.parse()?);
                }
                "--prompt-b" => prompt_b = Some(next_value(&mut iter, "--prompt-b")?),
                "--prompt-file" => {
                    let path = next_value(&mut iter, "--prompt-file")?;
                    prompt = Some(std::fs::read_to_string(&path).map_err(|error| {
                        cli_error(format!("lecture --prompt-file {path}: {error}"))
                    })?);
                }
                "--check" => check_only = true,
                "--load-only" => load_only = true,
                "--prefill-only" => prefill_only = true,
                "--prefill-repeat" => {
                    let repeat = next_value(&mut iter, "--prefill-repeat")?
                        .parse::<usize>()
                        .map_err(|e| cli_error(format!("--prefill-repeat invalide: {e}")))?;
                    if repeat == 0 {
                        return Err(cli_error("--prefill-repeat doit être > 0"));
                    }
                    prefill_repeat = Some(repeat);
                }
                "--metrics" => metrics = true,
                "--raw" => raw = true,
                "--ignore-stop-tokens" => ignore_stop_tokens = true,
                "--backend" | "--runtime" => {
                    backend = RuntimeKind::parse(&next_value(&mut iter, "--backend")?)?;
                }
                "--top-k" => {
                    top_k = next_value(&mut iter, "--top-k")?.parse()?;
                    top_k_explicit = true;
                }
                "--top-p" => {
                    top_p = next_value(&mut iter, "--top-p")?.parse()?;
                    top_p_explicit = true;
                }
                "--max-tokens" => {
                    max_tokens = next_value(&mut iter, "--max-tokens")?.parse()?;
                }
                "--temperature" => {
                    temperature = next_value(&mut iter, "--temperature")?.parse()?;
                }
                "--seed" => {
                    seed = next_value(&mut iter, "--seed")?.parse()?;
                }
                other => return Err(cli_error(format!("argument inconnu: {other}"))),
            }
        }

        let model_dir = model_dir.ok_or_else(|| cli_error("--model-dir requis"))?;
        let mode_count = [check_only, load_only, prefill_only]
            .into_iter()
            .filter(|enabled| *enabled)
            .count();
        if mode_count > 1 {
            return Err(cli_error(
                "--check, --load-only et --prefill-only sont exclusifs",
            ));
        }
        if prefill_repeat.is_some() && !prefill_only {
            return Err(cli_error("--prefill-repeat requiert --prefill-only"));
        }
        if !check_only && !load_only && prompt.is_none() && prompt_tokens.is_none() {
            return Err(cli_error(
                "--prompt ou --prompt-tokens requis hors --check/--load-only",
            ));
        }
        if max_tokens == 0 {
            return Err(cli_error("--max-tokens doit être > 0"));
        }
        if !temperature.is_finite() || temperature < 0.0 {
            return Err(cli_error("--temperature doit être un flottant positif"));
        }
        Ok(Self {
            model_dir,
            prompt,
            prompt_tokens,
            prompt_b,
            max_tokens,
            temperature,
            seed,
            check_only,
            load_only,
            prefill_only,
            prefill_repeat,
            top_k,
            top_p,
            top_k_explicit,
            top_p_explicit,
            metrics,
            raw,
            ignore_stop_tokens,
            backend,
        })
    }

    fn apply_generation_preset(&mut self, preset: saragossa::RuntimePreset) {
        if self.temperature <= f32::EPSILON {
            return;
        }
        if !self.top_k_explicit {
            self.top_k = preset.sampling_top_k;
        }
        if !self.top_p_explicit {
            self.top_p = preset.sampling_top_p;
        }
    }
}

impl RuntimeKind {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn default_backend() -> Self {
        Self::Metal
    }

    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    fn default_backend() -> Self {
        Self::Cpu
    }

    fn parse(value: &str) -> CliResult<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "cpu" | "rust" => Ok(Self::Cpu),
            "metal" | "gpu" | "rust-metal" => Ok(Self::Metal),
            other => Err(cli_error(format!(
                "backend inconnu: {other} — attendu cpu ou metal"
            ))),
        }
    }
}

fn next_value(iter: &mut impl Iterator<Item = String>, flag: &'static str) -> CliResult<String> {
    iter.next()
        .ok_or_else(|| cli_error(format!("valeur manquante pour {flag}")))
}

fn cli_error(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(CliError(message.into()))
}

fn print_help() {
    println!(
        "Usage: saragossa run <chemin|org/repo> [--backend cpu|metal] [--max-tokens N] [--temperature T] [--top-k N] [--top-p P] [--seed N]\n       saragossa list\n       saragossa <doctor|bench|serve|bench-serve> [options]\n       saragossa --model-dir <dir> (--check | --load-only | --prefill-only (--prompt <text>|--prompt-tokens N) | --prompt <text> | --prompt-tokens N) [--backend cpu|metal] [--raw] [--ignore-stop-tokens] [--top-k N] [--top-p P] [--max-tokens N] [--temperature T] [--seed N] [--metrics] [--prefill-repeat N]\nDefault backend: metal when available, cpu otherwise.\n`run` accepte un chemin local existant ou un id Hugging Face org/repo, télécharge les artefacts absents puis ouvre un REPL chat. `list` affiche les snapshots HF locaux contenant config.json.\nSet RETI_RUST_DFLASH_DRAFT_DIR during --check to validate a DFlash draft checkpoint. Set RETI_RUST_DFLASH_ACCEPTANCE=1 with RETI_RUST_DFLASH_DRAFT_DIR to run AR vs DFlash acceptance."
    );
}

fn load_decoder_with_runtime(
    assets: &ModelAssets,
    backend: RuntimeKind,
) -> CliResult<CausalDecoder> {
    match backend {
        RuntimeKind::Cpu => Ok(load_causal_decoder(assets)?),
        RuntimeKind::Metal => load_decoder_metal(assets),
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn load_decoder_metal(assets: &ModelAssets) -> CliResult<CausalDecoder> {
    let executor = saragossa::MetalExecutor::new()?;
    Ok(load_causal_decoder(assets)?.with_metal_executor(executor))
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
fn load_decoder_metal(_assets: &ModelAssets) -> CliResult<CausalDecoder> {
    Err(cli_error(
        "backend metal indisponible dans ce build — recompile avec --features metal",
    ))
}

fn print_contract(contract: &DecoderContract) {
    println!(
        "ok shards={} tensors={} required={} optional={} present={} mtp_declared={} mtp_weights={} mtp_tensors={}",
        contract.shard_count,
        contract.catalog_tensors,
        contract.required_specs,
        contract.optional_specs,
        contract.present_specs,
        contract.mtp_declared,
        contract.mtp_weights_present,
        contract.mtp_tensor_count
    );
}

fn format_top_k(logits: &[f32], k: usize) -> String {
    let mut indexed = logits
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .collect::<Vec<_>>();
    indexed.sort_by(|(_, left), (_, right)| right.total_cmp(left));
    indexed
        .into_iter()
        .take(k)
        .map(|(idx, value)| format!("{idx}:{value:.6}"))
        .collect::<Vec<_>>()
        .join(",")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_args_parse_required_values_and_defaults() {
        let args = CliArgs::parse([
            "--model-dir".to_string(),
            "models/tiny".to_string(),
            "--prompt".to_string(),
            "bonjour".to_string(),
        ])
        .expect("invariant: args minimaux valides");

        assert_eq!(args.model_dir, PathBuf::from("models/tiny"));
        assert_eq!(args.prompt.as_deref(), Some("bonjour"));
        assert_eq!(args.prompt_tokens, None);
        assert_eq!(args.max_tokens, 32);
        assert_eq!(args.temperature, 0.0);
        assert_eq!(args.seed, 0);
        assert!(!args.check_only);
        assert!(!args.load_only);
        assert!(!args.prefill_only);
        assert_eq!(args.prefill_repeat, None);
        assert_eq!(args.top_k, 0);
        assert!(!args.metrics);
        assert!(!args.ignore_stop_tokens);
        assert_eq!(args.backend, RuntimeKind::default_backend());
    }

    #[test]
    fn cli_direct_backend_default_matches_platform_runtime() {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        assert_eq!(RuntimeKind::default_backend(), RuntimeKind::Metal);

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        assert_eq!(RuntimeKind::default_backend(), RuntimeKind::Cpu);
    }

    #[test]
    fn cli_args_parse_sampling_overrides() {
        let args = CliArgs::parse([
            "--model-dir".to_string(),
            "models/tiny".to_string(),
            "--prompt".to_string(),
            "bonjour".to_string(),
            "--max-tokens".to_string(),
            "4".to_string(),
            "--temperature".to_string(),
            "0.7".to_string(),
            "--seed".to_string(),
            "42".to_string(),
        ])
        .expect("invariant: args sampling valides");

        assert_eq!(args.max_tokens, 4);
        assert!((args.temperature - 0.7).abs() < f32::EPSILON);
        assert_eq!(args.seed, 42);
    }

    #[test]
    fn qwen36_oq8_generation_preset_sets_sampling_defaults() {
        let mut args = CliArgs::parse([
            "--model-dir".to_string(),
            "models/Qwen3.6-35B-A3B-oQ8".to_string(),
            "--prompt".to_string(),
            "bonjour".to_string(),
            "--temperature".to_string(),
            "0.7".to_string(),
        ])
        .expect("invariant: args sampling valides");
        let preset = saragossa::runtime_preset_for_model_dir(&args.model_dir)
            .expect("invariant: preset qwen36 oQ8 detecte");

        args.apply_generation_preset(preset);

        assert_eq!(args.top_k, 20);
        assert!((args.top_p - 0.95).abs() < f32::EPSILON);
    }

    #[test]
    fn qwen36_oq8_generation_preset_preserves_explicit_top_k_zero() {
        let mut args = CliArgs::parse([
            "--model-dir".to_string(),
            "models/Qwen3.6-35B-A3B-oQ8".to_string(),
            "--prompt".to_string(),
            "bonjour".to_string(),
            "--temperature".to_string(),
            "0.7".to_string(),
            "--top-k".to_string(),
            "0".to_string(),
        ])
        .expect("invariant: args sampling valides");
        let preset = saragossa::runtime_preset_for_model_dir(&args.model_dir)
            .expect("invariant: preset qwen36 oQ8 detecte");

        args.apply_generation_preset(preset);

        assert_eq!(args.top_k, 0);
        assert!((args.top_p - 0.95).abs() < f32::EPSILON);
    }

    #[test]
    fn cli_args_accept_check_without_prompt() {
        let args = CliArgs::parse([
            "--model-dir".to_string(),
            "models/tiny".to_string(),
            "--check".to_string(),
        ])
        .expect("invariant: check sans prompt valide");

        assert_eq!(args.model_dir, PathBuf::from("models/tiny"));
        assert!(args.prompt.is_none());
        assert!(args.check_only);
        assert!(!args.load_only);
    }

    #[test]
    fn cli_args_accept_load_only_without_prompt() {
        let args = CliArgs::parse([
            "--model-dir".to_string(),
            "models/tiny".to_string(),
            "--load-only".to_string(),
        ])
        .expect("invariant: load-only sans prompt valide");

        assert_eq!(args.model_dir, PathBuf::from("models/tiny"));
        assert!(args.prompt.is_none());
        assert!(!args.check_only);
        assert!(args.load_only);
        assert!(!args.prefill_only);
    }

    #[test]
    fn cli_args_accept_prefill_only_with_prompt() {
        let args = CliArgs::parse([
            "--model-dir".to_string(),
            "models/tiny".to_string(),
            "--prefill-only".to_string(),
            "--prompt".to_string(),
            "bonjour".to_string(),
            "--prefill-repeat".to_string(),
            "3".to_string(),
            "--top-k".to_string(),
            "4".to_string(),
        ])
        .expect("invariant: prefill-only valide");

        assert!(args.prefill_only);
        assert_eq!(args.prefill_repeat, Some(3));
        assert_eq!(args.prompt.as_deref(), Some("bonjour"));
        assert_eq!(args.top_k, 4);
    }

    #[test]
    fn cli_args_reject_prefill_repeat_without_prefill_only() {
        let error = CliArgs::parse([
            "--model-dir".to_string(),
            "models/tiny".to_string(),
            "--prompt".to_string(),
            "bonjour".to_string(),
            "--prefill-repeat".to_string(),
            "2".to_string(),
        ])
        .expect_err("invariant: repeat exige prefill-only");

        assert!(
            error.to_string().contains("--prefill-repeat requiert"),
            "{error}"
        );
    }

    #[test]
    fn cli_args_accept_prompt_tokens_without_prompt() {
        let args = CliArgs::parse([
            "--model-dir".to_string(),
            "models/tiny".to_string(),
            "--prompt-tokens".to_string(),
            "32768".to_string(),
            "--raw".to_string(),
        ])
        .expect("invariant: prompt-tokens valide");

        assert!(args.prompt.is_none());
        assert_eq!(args.prompt_tokens, Some(32_768));
        assert!(args.raw);
    }

    #[test]
    fn cli_args_accept_metrics() {
        let args = CliArgs::parse([
            "--model-dir".to_string(),
            "models/tiny".to_string(),
            "--prompt".to_string(),
            "bonjour".to_string(),
            "--metrics".to_string(),
        ])
        .expect("invariant: metrics valides");

        assert!(args.metrics);
    }

    #[test]
    fn cli_args_accept_ignore_stop_tokens() {
        let args = CliArgs::parse([
            "--model-dir".to_string(),
            "models/tiny".to_string(),
            "--prompt".to_string(),
            "bonjour".to_string(),
            "--ignore-stop-tokens".to_string(),
        ])
        .expect("invariant: ignore-stop-tokens valide");

        assert!(args.ignore_stop_tokens);
    }

    #[test]
    fn cli_args_parse_metal_backend() {
        let args = CliArgs::parse([
            "--model-dir".to_string(),
            "models/tiny".to_string(),
            "--prompt".to_string(),
            "bonjour".to_string(),
            "--backend".to_string(),
            "metal".to_string(),
        ])
        .expect("invariant: backend metal accepté");

        assert_eq!(args.backend, RuntimeKind::Metal);
    }

    #[test]
    fn cli_args_reject_unknown_backend() {
        let err = CliArgs::parse([
            "--model-dir".to_string(),
            "models/tiny".to_string(),
            "--prompt".to_string(),
            "bonjour".to_string(),
            "--backend".to_string(),
            "mlx".to_string(),
        ])
        .expect_err("invariant: backend inconnu rejeté");

        assert!(err.to_string().contains("backend inconnu"));
    }

    #[test]
    fn cli_args_reject_exclusive_modes_together() {
        let err = CliArgs::parse([
            "--model-dir".to_string(),
            "models/tiny".to_string(),
            "--check".to_string(),
            "--load-only".to_string(),
        ])
        .expect_err("invariant: modes exclusifs");
        assert!(err.to_string().contains("exclusifs"));
    }

    #[test]
    fn cli_args_reject_missing_model_dir() {
        let err = CliArgs::parse(["--prompt".to_string(), "bonjour".to_string()])
            .expect_err("invariant: model_dir requis");
        assert!(err.to_string().contains("--model-dir requis"));
    }

    #[test]
    fn cli_args_reject_missing_prompt_outside_check() {
        let err = CliArgs::parse(["--model-dir".to_string(), "models/tiny".to_string()])
            .expect_err("invariant: prompt requis");
        assert!(err
            .to_string()
            .contains("--prompt ou --prompt-tokens requis hors --check/--load-only"));
    }

    #[test]
    fn cli_args_accept_zero_top_k() {
        let args = CliArgs::parse([
            "--model-dir".to_string(),
            "models/tiny".to_string(),
            "--prefill-only".to_string(),
            "--prompt".to_string(),
            "bonjour".to_string(),
            "--top-k".to_string(),
            "0".to_string(),
        ])
        .expect("invariant: top-k nul accepté");
        assert_eq!(args.top_k, 0);
    }

    #[test]
    fn top_k_formatter_orders_finite_logits() {
        assert_eq!(
            format_top_k(&[0.5, f32::NAN, 2.0, 1.0], 2),
            "2:2.000000,3:1.000000"
        );
    }
}
