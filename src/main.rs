//! Interface CLI du moteur d'inférence Rust.

use std::env;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use saragossa::{
    load_qwen_causal_decoder, render_qwen_chatml, verify_qwen_decoder_contract, CausalDecoder,
    ChatTemplateMessage, GenerationOptions, ModelAssets, QwenDecoderContract,
};

type CliResult<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Debug)]
struct CliArgs {
    model_dir: PathBuf,
    prompt: Option<String>,
    max_tokens: usize,
    temperature: f32,
    seed: u64,
    check_only: bool,
    load_only: bool,
    prefill_only: bool,
    top_k: usize,
    top_p: f32,
    metrics: bool,
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
            eprintln!("reti-rust-infer: {error}");
            ExitCode::from(2)
        }
    }
}

fn run() -> CliResult<()> {
    let raw_args: Vec<String> = env::args().skip(1).collect();
    if raw_args
        .iter()
        .any(|arg| matches!(arg.as_str(), "--help" | "-h"))
    {
        print_help();
        return Ok(());
    }
    let args = CliArgs::parse(raw_args)?;
    let assets = ModelAssets::load_local(&args.model_dir)?;
    if args.check_only {
        let contract = verify_qwen_decoder_contract(&assets)?;
        print_contract(&contract);
        return Ok(());
    }
    let load_started = Instant::now();
    let mut decoder = load_decoder_with_runtime(&assets, args.backend)?;
    if mtp_acceptance_enabled() {
        let path = assets.mtp.path.as_ref().ok_or_else(|| {
            cli_error("RETI_RUST_MTP_ACCEPTANCE=1 mais aucun sidecar MTP détecté")
        })?;
        decoder = decoder.with_mtp_sidecar(path)?;
    }
    let load_elapsed = load_started.elapsed();
    if args.load_only {
        println!("ok loaded");
        return Ok(());
    }
    let prompt = args
        .prompt
        .as_ref()
        .ok_or_else(|| cli_error("--prompt requis hors --check/--load-only/--prefill-only"))?;
    let prompt = render_qwen_chatml(&[ChatTemplateMessage::new("user", prompt)], true, false);
    let prompt_ids = assets
        .encode_prompt(&prompt)?
        .into_iter()
        .map(|id| usize::try_from(id).map_err(|_| cli_error(format!("token id hors plage: {id}"))))
        .collect::<CliResult<Vec<_>>>()?;
    let warmup_elapsed = warmup_decoder(&decoder, &prompt_ids, args.backend)?;
    if args.prefill_only {
        let prefill_started = Instant::now();
        let (_, logits) = decoder.prefill_cache(&prompt_ids)?;
        let prefill_elapsed = prefill_started.elapsed();
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
    if mtp_acceptance_enabled() {
        return run_mtp_acceptance(
            &assets,
            &decoder,
            &prompt_ids,
            &args,
            load_elapsed,
            warmup_elapsed,
        );
    }
    let generate_started = Instant::now();
    let output = decoder.generate_greedy_timed_with_options(
        &prompt_ids,
        args.max_tokens,
        &GenerationOptions {
            stop_token_ids: assets.stop_token_ids(),
            temperature: args.temperature,
            top_p: args.top_p,
            top_k: args.top_k,
            seed: args.seed,
        },
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
        eprintln!(
            "metrics tokens={} load_ms={} warmup_ms={} prefill_ms={} decode_tokens={} decode_ms={} generate_ms={} tok_s={:.3} decode_tok_s={:.3}",
            generated_len,
            load_elapsed.as_millis(),
            warmup_elapsed.as_millis(),
            timings.prefill.as_millis(),
            timings.decode_tokens,
            timings.decode.as_millis(),
            generate_elapsed.as_millis(),
            tok_s,
            decode_tok_s
        );
    }
    Ok(())
}

fn run_mtp_acceptance(
    assets: &ModelAssets,
    decoder: &CausalDecoder,
    prompt_ids: &[usize],
    args: &CliArgs,
    load_elapsed: Duration,
    warmup_elapsed: Duration,
) -> CliResult<()> {
    let options = GenerationOptions {
        stop_token_ids: assets.stop_token_ids(),
        temperature: args.temperature,
        top_p: 1.0,
        top_k: args.top_k,
        seed: args.seed,
    };
    let max_draft = std::env::var("RETI_RUST_MTP_MAX_DRAFT")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(2);
    let ar_started = Instant::now();
    let ar = decoder.generate_greedy_cached_with_options(prompt_ids, args.max_tokens, &options)?;
    let ar_elapsed = ar_started.elapsed();
    let spec_started = Instant::now();
    let spec = decoder.generate_greedy_mtp_batched_with_options(
        prompt_ids,
        args.max_tokens,
        &options,
        max_draft,
    )?;
    let spec_elapsed = spec_started.elapsed();
    let tokens_equal = ar == spec.tokens;
    let generated = spec
        .tokens
        .iter()
        .copied()
        .map(|id| {
            u32::try_from(id).map_err(|_| cli_error(format!("token généré hors plage: {id}")))
        })
        .collect::<CliResult<Vec<_>>>()?;
    let text = assets.decode_tokens(&generated, true)?;
    println!("{}", text.trim());

    let avg_accepted = if spec.stats.verifications > 0 {
        spec.stats.accepted as f64 / spec.stats.verifications as f64
    } else {
        0.0
    };
    eprintln!(
        "mtp_acceptance tokens_equal={} generated={} max_draft={} proposed={} accepted={} rejected={} verifications={} avg_accepted_per_verify={:.3} load_ms={} warmup_ms={} ar_ms={} spec_ms={}",
        tokens_equal,
        spec.tokens.len(),
        max_draft,
        spec.stats.proposed,
        spec.stats.accepted,
        spec.stats.rejected,
        spec.stats.verifications,
        avg_accepted,
        load_elapsed.as_millis(),
        warmup_elapsed.as_millis(),
        ar_elapsed.as_millis(),
        spec_elapsed.as_millis(),
    );
    for position in 0..max_draft {
        let proposed = spec
            .stats
            .proposed_by_position
            .get(position)
            .copied()
            .unwrap_or(0);
        let accepted = spec
            .stats
            .accepted_by_position
            .get(position)
            .copied()
            .unwrap_or(0);
        let rate = if proposed > 0 {
            accepted as f64 / proposed as f64
        } else {
            0.0
        };
        eprintln!(
            "mtp_acceptance_pos{} proposed={} accepted={} rate={:.3}",
            position + 1,
            proposed,
            accepted,
            rate
        );
    }
    if !tokens_equal {
        return Err(cli_error(
            "oracle AR==MTP échoué: tokens divergents en greedy",
        ));
    }
    Ok(())
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
    for _ in 0..passes {
        let _ = decoder.prefill_cache(prompt_ids)?;
    }
    Ok(started.elapsed())
}

fn warmup_enabled() -> bool {
    !env::var("RETI_RUST_WARMUP").is_ok_and(|value| {
        value == "0" || value.eq_ignore_ascii_case("false") || value.eq_ignore_ascii_case("off")
    })
}

fn mtp_acceptance_enabled() -> bool {
    env::var("RETI_RUST_MTP_ACCEPTANCE").is_ok_and(|value| {
        value != "0" && !value.eq_ignore_ascii_case("false") && !value.eq_ignore_ascii_case("off")
    })
}

fn warmup_passes() -> usize {
    env::var("RETI_RUST_WARMUP_PASSES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|passes| *passes > 0)
        .unwrap_or(2)
}

impl CliArgs {
    fn parse(args: impl IntoIterator<Item = String>) -> CliResult<Self> {
        let mut model_dir = None;
        let mut prompt = None;
        let mut max_tokens = 32_usize;
        let mut temperature = 0.0_f32;
        let mut seed = 0_u64;
        let mut check_only = false;
        let mut load_only = false;
        let mut prefill_only = false;
        let mut top_k = 0_usize;
        let mut top_p = 1.0_f32;
        let mut metrics = false;
        let mut backend = RuntimeKind::Cpu;
        let mut iter = args.into_iter();

        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--model-dir" => model_dir = Some(next_value(&mut iter, "--model-dir")?.into()),
                "--prompt" => prompt = Some(next_value(&mut iter, "--prompt")?),
                "--prompt-file" => {
                    let path = next_value(&mut iter, "--prompt-file")?;
                    prompt = Some(std::fs::read_to_string(&path).map_err(|error| {
                        cli_error(format!("lecture --prompt-file {path}: {error}"))
                    })?);
                }
                "--check" => check_only = true,
                "--load-only" => load_only = true,
                "--prefill-only" => prefill_only = true,
                "--metrics" => metrics = true,
                "--backend" | "--runtime" => {
                    backend = RuntimeKind::parse(&next_value(&mut iter, "--backend")?)?;
                }
                "--top-k" => {
                    top_k = next_value(&mut iter, "--top-k")?.parse()?;
                }
                "--top-p" => {
                    top_p = next_value(&mut iter, "--top-p")?.parse()?;
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
        if !check_only && !load_only && prompt.is_none() {
            return Err(cli_error(
                "--prompt requis hors --check/--load-only/--prefill-only",
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
            max_tokens,
            temperature,
            seed,
            check_only,
            load_only,
            prefill_only,
            top_k,
            top_p,
            metrics,
            backend,
        })
    }
}

impl RuntimeKind {
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
        "Usage: reti-rust-infer --model-dir <dir> (--check | --load-only | --prefill-only --prompt <text> | --prompt <text>) [--backend cpu|metal] [--top-k N] [--max-tokens N] [--temperature T] [--seed N] [--metrics]"
    );
}

fn load_decoder_with_runtime(
    assets: &ModelAssets,
    backend: RuntimeKind,
) -> CliResult<CausalDecoder> {
    match backend {
        RuntimeKind::Cpu => Ok(load_qwen_causal_decoder(assets)?),
        RuntimeKind::Metal => load_decoder_metal(assets),
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn load_decoder_metal(assets: &ModelAssets) -> CliResult<CausalDecoder> {
    let executor = saragossa::MetalExecutor::new()?;
    Ok(load_qwen_causal_decoder(assets)?.with_metal_executor(executor))
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
fn load_decoder_metal(_assets: &ModelAssets) -> CliResult<CausalDecoder> {
    Err(cli_error(
        "backend metal indisponible dans ce build — recompile avec --features metal",
    ))
}

fn print_contract(contract: &QwenDecoderContract) {
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
        assert_eq!(args.max_tokens, 32);
        assert_eq!(args.temperature, 0.0);
        assert_eq!(args.seed, 0);
        assert!(!args.check_only);
        assert!(!args.load_only);
        assert!(!args.prefill_only);
        assert_eq!(args.top_k, 0);
        assert!(!args.metrics);
        assert_eq!(args.backend, RuntimeKind::Cpu);
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
            "--top-k".to_string(),
            "4".to_string(),
        ])
        .expect("invariant: prefill-only valide");

        assert!(args.prefill_only);
        assert_eq!(args.prompt.as_deref(), Some("bonjour"));
        assert_eq!(args.top_k, 4);
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
            .contains("--prompt requis hors --check/--load-only/--prefill-only"));
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
