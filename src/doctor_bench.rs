//! Sous-commandes d'introspection et de benchmark du binaire saragossa.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use super::{
    cli_error, encode_prompt_ids, generate_benchmark_prompt, load_decoder_with_runtime, CliResult,
    RuntimeKind,
};
use saragossa::{runtime_flags::env_flag, GenerationOptions, ModelAssets, ModelConfig};

const MEASURED_CEILING_DATE: &str = "mesure 2026-06-28";
const DEFAULT_STT_MS: f64 = 127.0;
const DEFAULT_TTS_TTFA_MS: f64 = 4310.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OutputFormat {
    Markdown,
    Json,
}

impl OutputFormat {
    fn parse(value: &str) -> CliResult<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "md" | "markdown" => Ok(Self::Markdown),
            "json" => Ok(Self::Json),
            other => Err(cli_error(format!(
                "format inconnu: {other} - attendu md ou json"
            ))),
        }
    }
}

#[derive(Debug)]
struct DoctorArgs {
    model_dirs: Vec<PathBuf>,
    model_roots: Vec<PathBuf>,
    format: OutputFormat,
}

#[derive(Debug)]
struct BenchArgs {
    model_dir: PathBuf,
    backend: RuntimeKind,
    format: OutputFormat,
    repeats: usize,
    warmups: usize,
    pp_tokens: Vec<usize>,
    decode_tokens: usize,
    raw: bool,
    prompt: Option<String>,
    stt_ms: f64,
    tts_ttfa_ms: f64,
}

#[derive(Debug)]
struct ModelSummary {
    path: PathBuf,
    name: String,
    role: String,
    model_type: String,
    arch: String,
    quant: String,
    layers: Option<usize>,
    full_layers: Option<usize>,
    linear_layers: Option<usize>,
    context: Option<usize>,
    head_dim: Option<usize>,
    kv_heads: Option<usize>,
    weight_bytes: u64,
    kv_bytes_at_context: Option<u64>,
    tensor_count: usize,
    shard_count: usize,
    mtp_present: bool,
    mtp_tensors: usize,
}

#[derive(Debug)]
struct BenchResult {
    model: String,
    quant: String,
    backend: String,
    pp: Vec<BenchRow>,
    tg: BenchRow,
    ttft_ms: Stats,
    tpot_ms: Stats,
    voice: VoiceChain,
}

#[derive(Debug)]
struct BenchRow {
    test: String,
    tokens: usize,
    samples: Vec<f64>,
    stats: Stats,
}

#[derive(Clone, Copy, Debug, Default)]
struct Stats {
    mean: f64,
    stddev: f64,
    median: f64,
}

#[derive(Debug)]
struct VoiceChain {
    stt_ms: f64,
    llm_ttft_ms: f64,
    decode_tpot_ms: f64,
    tts_ttfa_ms: f64,
    e2e_first_audio_ms: f64,
}

/// Exécute `saragossa doctor`.
pub(crate) fn run_doctor(args: impl IntoIterator<Item = String>) -> CliResult<()> {
    let args = DoctorArgs::parse(args)?;
    let model_dirs = resolve_model_dirs(&args)?;
    if model_dirs.is_empty() {
        return Err(cli_error(
            "aucun modele detecte: passe --model-dir <dir> ou --model-root <dir>",
        ));
    }
    let mut models = Vec::with_capacity(model_dirs.len());
    for dir in model_dirs {
        models.push(load_model_summary(&dir)?);
    }
    let memory = host_memory_bytes();
    match args.format {
        OutputFormat::Markdown => print_doctor_markdown(&models, memory),
        OutputFormat::Json => print_doctor_json(&models, memory),
    }
    Ok(())
}

/// Exécute `saragossa bench`.
pub(crate) fn run_bench(args: impl IntoIterator<Item = String>) -> CliResult<()> {
    let args = BenchArgs::parse(args)?;
    if let Some(preset) = saragossa::runtime_preset_for_model_dir(&args.model_dir) {
        let _ = saragossa::apply_runtime_preset_for_model_dir(&args.model_dir);
        if args.format == OutputFormat::Markdown {
            eprintln!(
                "bench preset model={} sampling_top_k={} sampling_top_p={:.2}",
                args.model_dir.display(),
                preset.sampling_top_k,
                preset.sampling_top_p
            );
        }
    }
    let assets = ModelAssets::load_local(&args.model_dir)?;
    let model = load_model_summary(&args.model_dir)?;
    let decoder = load_decoder_with_runtime(&assets, args.backend)?;
    let result = run_bench_loaded(&assets, &decoder, &model, &args)?;
    match args.format {
        OutputFormat::Markdown => print_bench_markdown(&result),
        OutputFormat::Json => print_bench_json(&result)?,
    }
    Ok(())
}

impl DoctorArgs {
    fn parse(args: impl IntoIterator<Item = String>) -> CliResult<Self> {
        let mut model_dirs = Vec::new();
        let mut model_roots = Vec::new();
        let mut format = OutputFormat::Markdown;
        let mut iter = args.into_iter();
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--help" | "-h" => {
                    print_doctor_help();
                    return Err(cli_error("help"));
                }
                "--model-dir" => model_dirs.push(next_value(&mut iter, "--model-dir")?.into()),
                "--model-root" => model_roots.push(next_value(&mut iter, "--model-root")?.into()),
                "--format" => format = OutputFormat::parse(&next_value(&mut iter, "--format")?)?,
                other => return Err(cli_error(format!("argument doctor inconnu: {other}"))),
            }
        }
        Ok(Self {
            model_dirs,
            model_roots,
            format,
        })
    }
}

impl BenchArgs {
    fn parse(args: impl IntoIterator<Item = String>) -> CliResult<Self> {
        let mut model_dir = None;
        let mut backend = RuntimeKind::default_backend();
        let mut format = OutputFormat::Markdown;
        let mut repeats = 3_usize;
        let mut warmups = 1_usize;
        let mut pp_tokens = Vec::new();
        let mut decode_tokens = 128_usize;
        let mut raw = false;
        let mut prompt = None;
        let mut stt_ms = DEFAULT_STT_MS;
        let mut tts_ttfa_ms = DEFAULT_TTS_TTFA_MS;
        let mut iter = args.into_iter();
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--help" | "-h" => {
                    print_bench_help();
                    return Err(cli_error("help"));
                }
                "--model-dir" => model_dir = Some(next_value(&mut iter, "--model-dir")?.into()),
                "--backend" | "--runtime" => {
                    backend = RuntimeKind::parse(&next_value(&mut iter, "--backend")?)?;
                }
                "--format" => format = OutputFormat::parse(&next_value(&mut iter, "--format")?)?,
                "--repeats" => repeats = next_value(&mut iter, "--repeats")?.parse()?,
                "--warmups" => warmups = next_value(&mut iter, "--warmups")?.parse()?,
                "--pp-tokens" => pp_tokens.push(next_value(&mut iter, "--pp-tokens")?.parse()?),
                "--decode-tokens" | "--max-tokens" => {
                    decode_tokens = next_value(&mut iter, "--decode-tokens")?.parse()?;
                }
                "--prompt" => prompt = Some(next_value(&mut iter, "--prompt")?),
                "--prompt-file" => {
                    let path = next_value(&mut iter, "--prompt-file")?;
                    prompt = Some(std::fs::read_to_string(&path).map_err(|source| {
                        cli_error(format!("lecture --prompt-file {path}: {source}"))
                    })?);
                }
                "--raw" => raw = true,
                "--stt-ms" => stt_ms = next_value(&mut iter, "--stt-ms")?.parse()?,
                "--tts-ttfa-ms" => {
                    tts_ttfa_ms = next_value(&mut iter, "--tts-ttfa-ms")?.parse()?;
                }
                other => return Err(cli_error(format!("argument bench inconnu: {other}"))),
            }
        }
        let model_dir = model_dir.ok_or_else(|| cli_error("bench requiert --model-dir <dir>"))?;
        if repeats == 0 {
            return Err(cli_error("--repeats doit etre > 0"));
        }
        if decode_tokens == 0 {
            return Err(cli_error("--decode-tokens doit etre > 0"));
        }
        if pp_tokens.is_empty() {
            pp_tokens = vec![264, 8192, 32768];
        }
        Ok(Self {
            model_dir,
            backend,
            format,
            repeats,
            warmups,
            pp_tokens,
            decode_tokens,
            raw,
            prompt,
            stt_ms,
            tts_ttfa_ms,
        })
    }
}

fn next_value(iter: &mut impl Iterator<Item = String>, flag: &'static str) -> CliResult<String> {
    iter.next()
        .ok_or_else(|| cli_error(format!("valeur manquante pour {flag}")))
}

fn resolve_model_dirs(args: &DoctorArgs) -> CliResult<Vec<PathBuf>> {
    let mut dirs = BTreeSet::new();
    for dir in &args.model_dirs {
        dirs.insert(dir.clone());
    }
    let mut roots = args.model_roots.clone();
    if roots.is_empty() {
        if let Ok(value) = std::env::var("SARAGOSSA_MODEL_ROOTS") {
            roots.extend(
                value
                    .split(':')
                    .filter(|part| !part.trim().is_empty())
                    .map(PathBuf::from),
            );
        }
        let cwd_models = PathBuf::from("models");
        if cwd_models.is_dir() {
            roots.push(cwd_models);
        }
    }
    for root in roots {
        if !root.is_dir() {
            continue;
        }
        for entry in std::fs::read_dir(&root).map_err(|source| {
            cli_error(format!("lecture model-root {}: {source}", root.display()))
        })? {
            let path = entry
                .map_err(|source| {
                    cli_error(format!(
                        "lecture entree model-root {}: {source}",
                        root.display()
                    ))
                })?
                .path();
            if path.join("config.json").is_file() {
                dirs.insert(path);
            }
        }
    }
    Ok(dirs.into_iter().collect())
}

fn load_model_summary(path: &Path) -> CliResult<ModelSummary> {
    let config_path = path.join("config.json");
    let raw = read_json(&config_path)?;
    let normalized = ModelConfig::from_file(&config_path).ok();
    let assets = ModelAssets::load_local(path).ok();
    let config = assets
        .as_ref()
        .map(|assets| assets.config.clone())
        .or_else(|| normalized.clone());
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("<modele>")
        .to_string();
    let role = infer_role(&name, &raw);
    let model_type = config
        .as_ref()
        .map(|config| config.model_type.clone())
        .or_else(|| json_string(&raw, &["model_type"]))
        .unwrap_or_else(|| "unknown".to_string());
    let weight_bytes = safetensor_bytes(path)?;
    let shard_count = list_safetensors(path)?.len();
    let tensor_count = match &assets {
        Some(assets) => assets.catalog.tensor_count() + assets.mtp.tensor_count,
        None => count_safetensor_keys(path)?,
    };
    let mtp_present = assets
        .as_ref()
        .is_some_and(|assets| assets.mtp.is_available());
    let mtp_tensors = assets.as_ref().map_or(0, |assets| assets.mtp.tensor_count);
    let context = config_context(&raw);
    let (layers, full_layers, linear_layers, head_dim, kv_heads, arch, kv_bytes_at_context) =
        match &config {
            Some(config) => {
                let full = count_full_layers(config);
                let linear = config.num_hidden_layers.saturating_sub(full);
                let arch = describe_arch(config, full, linear);
                let kv = context.map(|context| estimate_kv_bytes(config, context, full));
                (
                    Some(config.num_hidden_layers),
                    Some(full),
                    Some(linear),
                    Some(config.head_dim()),
                    Some(config.num_key_value_heads),
                    arch,
                    kv,
                )
            }
            None => (None, None, None, None, None, "unknown".to_string(), None),
        };
    let quant = config
        .as_ref()
        .map(describe_quant)
        .unwrap_or_else(|| describe_quant_raw(&raw));
    Ok(ModelSummary {
        path: path.to_path_buf(),
        name,
        role,
        model_type,
        arch,
        quant,
        layers,
        full_layers,
        linear_layers,
        context,
        head_dim,
        kv_heads,
        weight_bytes,
        kv_bytes_at_context,
        tensor_count,
        shard_count,
        mtp_present,
        mtp_tensors,
    })
}

fn read_json(path: &Path) -> CliResult<Value> {
    let file = std::fs::File::open(path)
        .map_err(|source| cli_error(format!("lecture {}: {source}", path.display())))?;
    serde_json::from_reader(file)
        .map_err(|source| cli_error(format!("json {}: {source}", path.display())))
}

fn list_safetensors(path: &Path) -> CliResult<Vec<PathBuf>> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(path)
        .map_err(|source| cli_error(format!("lecture {}: {source}", path.display())))?
    {
        let entry = entry
            .map_err(|source| cli_error(format!("lecture entree {}: {source}", path.display())))?;
        let item = entry.path();
        if item.extension().and_then(|ext| ext.to_str()) == Some("safetensors") {
            files.push(item);
        }
    }
    files.sort();
    Ok(files)
}

fn safetensor_bytes(path: &Path) -> CliResult<u64> {
    let mut total = 0_u64;
    for file in list_safetensors(path)? {
        total = total.saturating_add(
            std::fs::metadata(&file)
                .map_err(|source| cli_error(format!("metadata {}: {source}", file.display())))?
                .len(),
        );
    }
    Ok(total)
}

fn count_safetensor_keys(path: &Path) -> CliResult<usize> {
    let mut total = 0_usize;
    for file in list_safetensors(path)? {
        total = total.saturating_add(saragossa::catalog::read_safetensors_keys(file)?.len());
    }
    Ok(total)
}

fn infer_role(name: &str, raw: &Value) -> String {
    let lower = name.to_ascii_lowercase();
    let model_type = json_string(raw, &["model_type"]).unwrap_or_default();
    if lower.contains("whisper") || model_type.contains("whisper") {
        "STT Whisper".to_string()
    } else if lower.contains("tts") || model_type.contains("tts") {
        "TTS Qwen3".to_string()
    } else if lower.contains("35b") {
        "LLM voix".to_string()
    } else if lower.contains("27b") {
        "LLM agent".to_string()
    } else {
        "LLM".to_string()
    }
}

fn describe_arch(config: &ModelConfig, full_layers: usize, linear_layers: usize) -> String {
    let attention = if config.is_hybrid() {
        format!("hybride {linear_layers} linear / {full_layers} full")
    } else {
        format!("{full_layers} full")
    };
    if config.is_moe() {
        let experts = config.num_experts.unwrap_or(0);
        let active = config
            .num_experts_per_tok
            .or(config.top_k_experts)
            .unwrap_or(0);
        format!("MoE {experts} experts top-{active}; {attention}")
    } else {
        format!("dense FFN; {attention}")
    }
}

fn describe_quant(config: &ModelConfig) -> String {
    match &config.quantization {
        Some(quant) => {
            let bits = quant
                .bits
                .map(|bits| bits.to_string())
                .unwrap_or_else(|| "?".to_string());
            let group = quant
                .group_size
                .map(|group| group.to_string())
                .unwrap_or_else(|| "?".to_string());
            let method = quant
                .quant_method
                .as_deref()
                .or(quant.fmt.as_deref())
                .or_else(|| quant.extra.get("mode").and_then(Value::as_str))
                .unwrap_or("affine");
            format!("{bits}-bit g{group} {method}")
        }
        None => "dense".to_string(),
    }
}

fn describe_quant_raw(raw: &Value) -> String {
    let quant = raw
        .get("quantization_config")
        .or_else(|| raw.get("quantization"))
        .or_else(|| {
            raw.get("text_config")
                .and_then(|v| v.get("quantization_config"))
        })
        .or_else(|| raw.get("text_config").and_then(|v| v.get("quantization")));
    if let Some(quant) = quant {
        let bits = quant
            .get("bits")
            .and_then(Value::as_u64)
            .map(|bits| bits.to_string())
            .unwrap_or_else(|| "?".to_string());
        let group = quant
            .get("group_size")
            .and_then(Value::as_u64)
            .map(|group| group.to_string())
            .unwrap_or_else(|| "?".to_string());
        return format!("{bits}-bit g{group}");
    }
    json_string(raw, &["torch_dtype"]).unwrap_or_else(|| "dense/unknown".to_string())
}

fn count_full_layers(config: &ModelConfig) -> usize {
    (0..config.num_hidden_layers)
        .filter(|layer| config.is_full_attention_layer(*layer))
        .count()
}

fn estimate_kv_bytes(config: &ModelConfig, context: usize, full_layers: usize) -> u64 {
    let per_token = config
        .num_key_value_heads
        .saturating_mul(config.head_dim())
        .saturating_mul(2)
        .saturating_mul(2);
    (full_layers as u64)
        .saturating_mul(context as u64)
        .saturating_mul(per_token as u64)
}

fn config_context(raw: &Value) -> Option<usize> {
    for path in [
        &["max_position_embeddings"][..],
        &["max_sequence_length"][..],
        &["seq_length"][..],
        &["text_config", "max_position_embeddings"][..],
        &["text_config", "max_sequence_length"][..],
        &["text_config", "seq_length"][..],
    ] {
        if let Some(value) = json_usize(raw, path) {
            return Some(value);
        }
    }
    None
}

fn json_string(raw: &Value, path: &[&str]) -> Option<String> {
    let mut cursor = raw;
    for key in path {
        cursor = cursor.get(*key)?;
    }
    cursor.as_str().map(ToString::to_string)
}

fn json_usize(raw: &Value, path: &[&str]) -> Option<usize> {
    let mut cursor = raw;
    for key in path {
        cursor = cursor.get(*key)?;
    }
    cursor
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
}

fn host_memory_bytes() -> Option<u64> {
    command_stdout("sysctl", &["-n", "hw.memsize"]).and_then(|text| text.trim().parse().ok())
}

fn macos_version() -> Option<String> {
    command_stdout("sw_vers", &["-productVersion"]).map(|text| text.trim().to_string())
}

fn command_stdout(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).to_string())
}

fn runtime_flag(name: &str, default: bool) -> bool {
    env_flag(name, default)
}

fn run_bench_loaded(
    assets: &ModelAssets,
    decoder: &saragossa::CausalDecoder,
    model: &ModelSummary,
    args: &BenchArgs,
) -> CliResult<BenchResult> {
    let mut pp = Vec::new();
    for target in &args.pp_tokens {
        let prompt = generate_benchmark_prompt(assets, *target, args.raw)?;
        let prompt_ids = encode_prompt_ids(assets, &prompt, args.raw)?;
        let mut samples = Vec::with_capacity(args.repeats);
        for run in 0..args.warmups + args.repeats {
            let started = Instant::now();
            let _ = decoder.prefill_cache_uncached(&prompt_ids)?;
            let elapsed = started.elapsed();
            if run >= args.warmups {
                samples.push(rate(prompt_ids.len(), elapsed));
            }
        }
        pp.push(BenchRow {
            test: format!("pp{target}"),
            tokens: prompt_ids.len(),
            stats: stats(&samples),
            samples,
        });
    }

    let base_prompt = args.prompt.clone().unwrap_or_else(default_voice_prompt);
    let options = GenerationOptions {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        seed: 0,
        stop_token_ids: Vec::new(),
        stop_sequences: Vec::new(),
    };
    let mut tg_samples = Vec::with_capacity(args.repeats);
    let mut ttft_samples = Vec::with_capacity(args.repeats);
    let mut tpot_samples = Vec::with_capacity(args.repeats);
    let mut prompt_len = 0_usize;
    for run in 0..args.warmups + args.repeats {
        let prompt = format!("{base_prompt}\n\n[bench-repeat:{run}]");
        let prompt_ids = encode_prompt_ids(assets, &prompt, args.raw)?;
        prompt_len = prompt_ids.len();
        let output = decoder.generate_greedy_timed_with_options(
            &prompt_ids,
            args.decode_tokens,
            &options,
        )?;
        if run >= args.warmups {
            let decode_tok_s = rate(output.timings.decode_tokens, output.timings.decode);
            let tpot = millis_per_token(output.timings.decode, output.timings.decode_tokens);
            tg_samples.push(decode_tok_s);
            ttft_samples.push(output.timings.prefill.as_secs_f64() * 1_000.0);
            tpot_samples.push(tpot);
        }
    }
    let ttft_stats = stats(&ttft_samples);
    let tpot_stats = stats(&tpot_samples);
    let voice = VoiceChain {
        stt_ms: args.stt_ms,
        llm_ttft_ms: ttft_stats.median,
        decode_tpot_ms: tpot_stats.median,
        tts_ttfa_ms: args.tts_ttfa_ms,
        e2e_first_audio_ms: args.stt_ms + ttft_stats.median + tpot_stats.median + args.tts_ttfa_ms,
    };
    Ok(BenchResult {
        model: model.name.clone(),
        quant: model.quant.clone(),
        backend: backend_name(args.backend).to_string(),
        pp,
        tg: BenchRow {
            test: "tg".to_string(),
            tokens: prompt_len,
            stats: stats(&tg_samples),
            samples: tg_samples,
        },
        ttft_ms: ttft_stats,
        tpot_ms: tpot_stats,
        voice,
    })
}

fn default_voice_prompt() -> String {
    "Tu es Reti, assistant vocal local. Reponds en une phrase courte et utile: donne le prochain geste concret pour reduire la latence voix.".to_string()
}

fn rate(tokens: usize, elapsed: Duration) -> f64 {
    let seconds = elapsed.as_secs_f64();
    if seconds > 0.0 {
        tokens as f64 / seconds
    } else {
        0.0
    }
}

fn millis_per_token(elapsed: Duration, tokens: usize) -> f64 {
    if tokens > 0 {
        elapsed.as_secs_f64() * 1_000.0 / tokens as f64
    } else {
        0.0
    }
}

fn stats(samples: &[f64]) -> Stats {
    if samples.is_empty() {
        return Stats::default();
    }
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let variance = if samples.len() > 1 {
        samples
            .iter()
            .map(|sample| {
                let diff = *sample - mean;
                diff * diff
            })
            .sum::<f64>()
            / (samples.len() - 1) as f64
    } else {
        0.0
    };
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let median = sorted[sorted.len() / 2];
    Stats {
        mean,
        stddev: variance.sqrt(),
        median,
    }
}

fn print_doctor_markdown(models: &[ModelSummary], memory: Option<u64>) {
    println!("# saragossa doctor");
    println!();
    println!("## MODELES");
    println!(
        "| nom | role | arch | quant | couches | contexte | head_dim | MTP | tenseurs | poids |"
    );
    println!("| --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: |");
    for model in models {
        println!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            model.name,
            model.role,
            model.arch,
            model.quant,
            opt_usize(model.layers),
            opt_usize(model.context),
            opt_usize(model.head_dim),
            if model.mtp_present { "oui" } else { "non" },
            model.tensor_count,
            format_bytes(model.weight_bytes)
        );
    }
    println!();
    println!("## CHARGE / RESIDENT");
    println!("Processus saragossa: one-shot CLI, pas de daemon resident; % GPU live indisponible sans serveur.");
    println!(
        "| modele | footprint poids | KV max contexte | decode resident | prefix-cache | GPU % |"
    );
    println!("| --- | ---: | ---: | --- | --- | --- |");
    for model in models {
        println!(
            "| {} | {} | {} | {} | {} | n/a |",
            model.name,
            format_bytes(model.weight_bytes),
            model
                .kv_bytes_at_context
                .map(format_bytes)
                .unwrap_or_else(|| "n/a".to_string()),
            resident_status(model),
            flag_text(runtime_flag("RETI_RUST_PREFIX_CACHE", true)),
        );
    }
    println!();
    println!("## SANTE");
    for (status, name, detail) in health_checks() {
        println!("- {status} **{name}** - {detail}");
    }
    println!();
    println!("### FIT-CHECK");
    print_fit_check(models, memory);
    println!();
    println!("## PLAFONDS MESURES / RESTE");
    println!(
        "| surface | plafond mesure | reste honnete |\n| --- | --- | --- |\n| decode 35B resident | 151 t/s ({MEASURED_CEILING_DATE}), mur ~510 GB/s atteint | MTP/fusion spec fermes sauf nouvelle preuve |\n| prefill voix | ~209 ms = 127 ms fixe + 0.31 ms/token ({MEASURED_CEILING_DATE}) | residence/warmup avant 4-bit |\n| budget voix | <500 ms fin-parole -> premier token/audio | STT/TTS dominent selon chemin |\n| 4-bit NA voix | mesure plus lente et drift qualite ({MEASURED_CEILING_DATE}) | pas levier par defaut |\n| chunked linear-attn | dead end mesure: recurrent deja equivalent oMLX | ne pas reactiver sans nouvelle preuve |"
    );
    println!();
    println!("## POURQUOI RETI / POURQUOI PAS");
    println!("| Utilise saragossa pour | Pas pour |");
    println!("| --- | --- |");
    println!("| voix temps-reel FR locale | multi-GPU |");
    println!("| Apple Silicon / Metal / NA tensor cores | CUDA ou non-Apple |");
    println!("| decode resident pur Rust, zero Python | serving batch massif type vLLM/TGI |");
    println!("| MoE A3B rapide single-stream | modeles non portes / configs inconnues |");
    println!("| chaine STT + LLM + TTS integree | debit agrege multi-tenant |");
    println!("| reproductibilite byte-id quand le chemin l'exige | quantifs experimentales sans oracle |");
}

fn print_doctor_json(models: &[ModelSummary], memory: Option<u64>) {
    let value = json!({
        "models": models.iter().map(model_json).collect::<Vec<_>>(),
        "health": health_checks().into_iter().map(|(status, name, detail)| {
            json!({"status": status, "name": name, "detail": detail})
        }).collect::<Vec<_>>(),
        "fit_check": fit_check_json(models, memory),
        "measured_ceilings": {
            "date": MEASURED_CEILING_DATE,
            "decode_35b_tok_s": 151.0,
            "decode_bandwidth_gb_s": 510.0,
            "voice_prefill_ms": 209.0,
            "voice_prefill_fixed_ms": 127.0,
            "voice_prefill_ms_per_token": 0.31
        }
    });
    println!("{value}");
}

fn model_json(model: &ModelSummary) -> Value {
    json!({
        "name": model.name,
        "path": model.path,
        "role": model.role,
        "model_type": model.model_type,
        "arch": model.arch,
        "quant": model.quant,
        "layers": model.layers,
        "full_layers": model.full_layers,
        "linear_layers": model.linear_layers,
        "context": model.context,
        "head_dim": model.head_dim,
        "kv_heads": model.kv_heads,
        "weight_bytes": model.weight_bytes,
        "kv_bytes_at_context": model.kv_bytes_at_context,
        "tensor_count": model.tensor_count,
        "shard_count": model.shard_count,
        "mtp_present": model.mtp_present,
        "mtp_tensors": model.mtp_tensors
    })
}

fn health_checks() -> Vec<(&'static str, &'static str, String)> {
    let metal = cfg!(all(target_os = "macos", feature = "metal"));
    let macos = macos_version().unwrap_or_else(|| "unknown".to_string());
    vec![
        (
            check(metal),
            "backend rust-metal",
            if metal {
                format!("build Metal actif, macOS {macos}")
            } else {
                "build sans feature metal".to_string()
            },
        ),
        (
            check(metal && macos_major_at_least(26)),
            "NA matmul2d",
            format!(
                "macOS {macos}, RETI_RUST_QMM_NA={}",
                flag_text(runtime_flag("RETI_RUST_QMM_NA", true))
            ),
        ),
        (
            check(runtime_flag("RETI_RUST_DECODE_RESIDENT_FULL", true)),
            "decode resident full",
            format!(
                "full={}, linear={}",
                flag_text(runtime_flag("RETI_RUST_DECODE_RESIDENT_FULL", true)),
                flag_text(runtime_flag("RETI_RUST_DECODE_RESIDENT_FULL_LINEAR", false))
            ),
        ),
        (
            check(runtime_flag("RETI_RUST_PREFILL_RESIDENT", true)),
            "prefill resident",
            format!(
                "RETI_RUST_PREFILL_RESIDENT={}",
                flag_text(runtime_flag("RETI_RUST_PREFILL_RESIDENT", true))
            ),
        ),
        (
            check(metal),
            "bf16 Metal",
            "chemins bf16/NA disponibles quand Metal est actif".to_string(),
        ),
        (
            check(runtime_flag("RETI_RUST_GPU_SAMPLER", true)),
            "sampler GPU",
            format!(
                "RETI_RUST_GPU_SAMPLER={}",
                flag_text(runtime_flag("RETI_RUST_GPU_SAMPLER", true))
            ),
        ),
        (
            check(runtime_flag("RETI_RUST_LIGHTBATCH_QMM2", true)),
            "light-batch",
            format!(
                "qmm2={}, moe2={}",
                flag_text(runtime_flag("RETI_RUST_LIGHTBATCH_QMM2", true)),
                flag_text(runtime_flag("RETI_RUST_LIGHTBATCH_MOE2", true))
            ),
        ),
        (
            check(runtime_flag("RETI_RUST_PREFIX_CACHE", true)),
            "prefix-cache",
            format!(
                "enabled={}, cap={}",
                flag_text(runtime_flag("RETI_RUST_PREFIX_CACHE", true)),
                std::env::var("RETI_RUST_PREFIX_CACHE_CAP").unwrap_or_else(|_| "32".to_string())
            ),
        ),
        (
            "⚠",
            "ANE",
            "non cable: Metal GPU/NA uniquement pour saragossa".to_string(),
        ),
    ]
}

fn check(ok: bool) -> &'static str {
    if ok {
        "✓"
    } else {
        "✗"
    }
}

fn macos_major_at_least(major: u64) -> bool {
    macos_version()
        .and_then(|version| version.split('.').next().and_then(|head| head.parse().ok()))
        .is_some_and(|actual: u64| actual >= major)
}

fn resident_status(model: &ModelSummary) -> String {
    let full = runtime_flag("RETI_RUST_DECODE_RESIDENT_FULL", true);
    let linear = runtime_flag("RETI_RUST_DECODE_RESIDENT_FULL_LINEAR", false);
    if model.linear_layers.unwrap_or(0) > 0 && !linear {
        format!("partiel: full={}, linear=off", flag_text(full))
    } else {
        flag_text(full).to_string()
    }
}

fn print_fit_check(models: &[ModelSummary], memory: Option<u64>) {
    let Some(memory) = memory else {
        println!("⚠ memoire unifiee inconnue (sysctl hw.memsize indisponible).");
        return;
    };
    println!("Memoire unifiee detectee: {}.", format_bytes(memory));
    println!("| modele | poids + KV contexte | part memoire | verdict |");
    println!("| --- | ---: | ---: | --- |");
    let mut aggregate = 0_u64;
    for model in models {
        let total = model
            .weight_bytes
            .saturating_add(model.kv_bytes_at_context.unwrap_or(0));
        aggregate = aggregate.saturating_add(total);
        println!(
            "| {} | {} | {:.1}% | {} |",
            model.name,
            format_bytes(total),
            100.0 * total as f64 / memory as f64,
            fit_verdict(total, memory)
        );
    }
    if models.len() > 1 {
        println!(
            "| tous modeles listes | {} | {:.1}% | {} |",
            format_bytes(aggregate),
            100.0 * aggregate as f64 / memory as f64,
            fit_verdict(aggregate, memory)
        );
    }
}

fn fit_check_json(models: &[ModelSummary], memory: Option<u64>) -> Value {
    let rows = models
        .iter()
        .map(|model| {
            let total = model
                .weight_bytes
                .saturating_add(model.kv_bytes_at_context.unwrap_or(0));
            json!({
                "model": model.name,
                "bytes": total,
                "memory_fraction": memory.map(|memory| total as f64 / memory as f64),
                "verdict": memory.map(|memory| fit_verdict(total, memory))
            })
        })
        .collect::<Vec<_>>();
    json!({"unified_memory_bytes": memory, "rows": rows})
}

fn fit_verdict(bytes: u64, memory: u64) -> &'static str {
    let ratio = bytes as f64 / memory as f64;
    if ratio >= 1.0 {
        "✗ risque swap/OOM"
    } else if ratio >= 0.85 {
        "⚠ marge faible, swap possible"
    } else {
        "✓ tient en memoire"
    }
}

fn print_bench_markdown(result: &BenchResult) {
    println!("# saragossa bench");
    println!();
    println!(
        "model={} quant={} backend={}",
        result.model, result.quant, result.backend
    );
    println!();
    println!("## Throughput");
    println!("| model | quant | backend | test | tokens | t/s ± stddev | samples |");
    println!("| --- | --- | --- | --- | ---: | ---: | --- |");
    for row in &result.pp {
        print_bench_row(result, row);
    }
    print_bench_row(result, &result.tg);
    println!();
    println!("## Latence");
    println!("| metrique | ms | stddev |");
    println!("| --- | ---: | ---: |");
    println!(
        "| TTFT | {:.1} | {:.1} |",
        result.ttft_ms.median, result.ttft_ms.stddev
    );
    println!(
        "| TPOT | {:.2} | {:.2} |",
        result.tpot_ms.median, result.tpot_ms.stddev
    );
    println!();
    println!("## Chaine voix E2E");
    println!("| STT | prefill TTFT | decode TPOT | TTS TTFA | e2e premier audio |");
    println!("| ---: | ---: | ---: | ---: | ---: |");
    println!(
        "| {:.1} ms | {:.1} ms | {:.2} ms | {:.1} ms | {:.1} ms |",
        result.voice.stt_ms,
        result.voice.llm_ttft_ms,
        result.voice.decode_tpot_ms,
        result.voice.tts_ttfa_ms,
        result.voice.e2e_first_audio_ms
    );
    println!();
    println!(
        "Note: STT et TTS TTFA sont des plafonds mesures injectables via --stt-ms/--tts-ttfa-ms; LLM TTFT/TPOT sont mesures dans ce run."
    );
}

fn print_bench_row(result: &BenchResult, row: &BenchRow) {
    println!(
        "| {} | {} | {} | {} | {} | {:.2} ± {:.2} | {} |",
        result.model,
        result.quant,
        result.backend,
        row.test,
        row.tokens,
        row.stats.mean,
        row.stats.stddev,
        row.samples
            .iter()
            .map(|sample| format!("{sample:.2}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
}

fn print_bench_json(result: &BenchResult) -> CliResult<()> {
    let value = json!({
        "model": result.model,
        "quant": result.quant,
        "backend": result.backend,
        "pp": result.pp.iter().map(bench_row_json).collect::<Vec<_>>(),
        "tg": bench_row_json(&result.tg),
        "latency": {
            "ttft_ms": stats_json(result.ttft_ms),
            "tpot_ms": stats_json(result.tpot_ms)
        },
        "voice_chain": {
            "stt_ms": result.voice.stt_ms,
            "llm_ttft_ms": result.voice.llm_ttft_ms,
            "decode_tpot_ms": result.voice.decode_tpot_ms,
            "tts_ttfa_ms": result.voice.tts_ttfa_ms,
            "e2e_first_audio_ms": result.voice.e2e_first_audio_ms
        }
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&value)
            .map_err(|source| cli_error(format!("serialisation json bench: {source}")))?
    );
    Ok(())
}

fn bench_row_json(row: &BenchRow) -> Value {
    json!({
        "test": row.test,
        "tokens": row.tokens,
        "tok_s": stats_json(row.stats),
        "samples": row.samples
    })
}

fn stats_json(stats: Stats) -> Value {
    json!({
        "mean": stats.mean,
        "stddev": stats.stddev,
        "median": stats.median
    })
}

fn opt_usize(value: Option<usize>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "n/a".to_string())
}

fn flag_text(value: bool) -> &'static str {
    if value {
        "on"
    } else {
        "off"
    }
}

fn backend_name(backend: RuntimeKind) -> &'static str {
    match backend {
        RuntimeKind::Cpu => "cpu",
        RuntimeKind::Metal => "metal",
    }
}

fn format_bytes(bytes: u64) -> String {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.1} GiB", bytes as f64 / GIB)
    } else if bytes >= 1024 * 1024 {
        format!("{:.1} MiB", bytes as f64 / MIB)
    } else {
        format!("{bytes} B")
    }
}

fn print_doctor_help() {
    println!(
        "Usage: saragossa doctor [--model-dir DIR ...] [--model-root DIR ...] [--format md|json]"
    );
}

fn print_bench_help() {
    println!(
        "Usage: saragossa bench --model-dir DIR [--backend metal|cpu] [--format md|json] [--repeats N] [--warmups N] [--pp-tokens N ...] [--decode-tokens N] [--prompt TEXT|--prompt-file PATH] [--raw] [--stt-ms MS] [--tts-ttfa-ms MS]"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn doctor_args_accept_repeated_model_dirs() {
        let args = DoctorArgs::parse([
            "--model-dir".to_string(),
            "models/a".to_string(),
            "--model-dir".to_string(),
            "models/b".to_string(),
            "--format".to_string(),
            "json".to_string(),
        ])
        .expect("invariant: args doctor valides");

        assert_eq!(args.model_dirs.len(), 2);
        assert_eq!(args.format, OutputFormat::Json);
    }

    #[test]
    fn bench_args_default_long_pp_suite() {
        let args = BenchArgs::parse([
            "--model-dir".to_string(),
            "models/a".to_string(),
            "--repeats".to_string(),
            "2".to_string(),
        ])
        .expect("invariant: args bench valides");

        assert_eq!(args.pp_tokens, vec![264, 8192, 32768]);
        assert_eq!(args.repeats, 2);
        assert_eq!(args.decode_tokens, 128);
    }

    #[test]
    fn stats_reports_sample_stddev() {
        let got = stats(&[10.0, 12.0, 14.0]);

        assert!((got.mean - 12.0).abs() < f64::EPSILON);
        assert!((got.stddev - 2.0).abs() < f64::EPSILON);
        assert!((got.median - 12.0).abs() < f64::EPSILON);
    }
}
