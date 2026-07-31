//! Mesure le cache scratch GPU sur plusieurs générations longues dans un process.
//!
//! Usage :
//!   cargo run -p saragossa --release --features metal \
//!     --example scratch_cache_32k -- <model_dir> [prompt_tokens] [max_tokens] [repeats]

use std::path::Path;
use std::time::Instant;

use saragossa::{
    apply_runtime_preset_for_model_dir, load_causal_decoder_for_metal, GenerationOptions,
    MetalExecutor, ModelAssets,
};

fn synth_ids(len: usize) -> Vec<usize> {
    (0..len)
        .map(|index| (index * 7 + 13) % 30_000 + 1)
        .collect()
}

fn parse_positive(value: Option<String>, default: usize, label: &str) -> Result<usize, String> {
    let Some(value) = value else {
        return Ok(default);
    };
    value
        .parse::<usize>()
        .ok()
        .filter(|parsed| *parsed > 0)
        .ok_or_else(|| format!("{label} doit être un entier > 0"))
}

fn token_md5(tokens: &[usize]) -> String {
    let mut context = md5::Context::new();
    for token in tokens {
        context.consume(token.to_le_bytes());
    }
    format!("{:x}", context.compute())
}

fn token_ids(tokens: &[usize]) -> String {
    tokens
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model_dir = args
        .next()
        .ok_or("usage: scratch_cache_32k <model_dir> [prompt_tokens] [max_tokens] [repeats]")?;
    let prompt_tokens = parse_positive(args.next(), 32_768, "prompt_tokens")?;
    let max_tokens = parse_positive(args.next(), 64, "max_tokens")?;
    let repeats = parse_positive(args.next(), 3, "repeats")?;

    eprintln!("chargement {model_dir} …");
    if let Some(preset) = apply_runtime_preset_for_model_dir(Path::new(&model_dir)) {
        eprintln!("preset runtime {} ({})", preset.name, preset.opt_profile);
    }
    let assets = ModelAssets::load_local(&model_dir)?;
    let executor = MetalExecutor::new()?;
    let decoder =
        load_causal_decoder_for_metal(&assets, &executor)?.with_metal_executor(executor)?;
    let options = GenerationOptions {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        seed: 0,
        ..GenerationOptions::default()
    };

    eprintln!("warmup (256 tokens, 2 générations de 8 tokens) …");
    let warm = synth_ids(256);
    let warm_started = Instant::now();
    let _ = decoder.generate_greedy_timed_with_options(&warm, 8, &options)?;
    let _ = decoder.generate_greedy_timed_with_options(&warm, 8, &options)?;
    eprintln!("warmup fait en {} ms", warm_started.elapsed().as_millis());

    let prompt = synth_ids(prompt_tokens);
    let mut reference = None;
    println!(
        "# pass\tprompt_tokens\tgenerated\tprefill_ms\tprefill_tok_s\tdecode_ms\tdecode_tok_s\ttotal_ms\ttokens_equal\ttokens_md5"
    );
    for pass in 0..repeats {
        let started = Instant::now();
        let output = decoder.generate_greedy_timed_with_options(&prompt, max_tokens, &options)?;
        let total = started.elapsed();
        let prefill_seconds = output.timings.prefill.as_secs_f64();
        let decode_seconds = output.timings.decode.as_secs_f64();
        let prefill_tok_s = if prefill_seconds > 0.0 {
            prompt_tokens as f64 / prefill_seconds
        } else {
            0.0
        };
        let decode_tok_s = if decode_seconds > 0.0 {
            output.timings.decode_tokens as f64 / decode_seconds
        } else {
            0.0
        };
        let tokens_equal = reference
            .as_ref()
            .is_none_or(|expected: &Vec<usize>| expected == &output.tokens);
        let md5 = token_md5(&output.tokens);
        println!(
            "{pass}\t{prompt_tokens}\t{}\t{}\t{prefill_tok_s:.3}\t{}\t{decode_tok_s:.3}\t{}\t{tokens_equal}\t{md5}",
            output.tokens.len(),
            output.timings.prefill.as_millis(),
            output.timings.decode.as_millis(),
            total.as_millis(),
        );
        eprintln!(
            "oracle_ids pass={pass} tokens_equal={tokens_equal} ids={}",
            token_ids(&output.tokens)
        );
        if reference.is_none() {
            reference = Some(output.tokens);
        }
    }
    Ok(())
}
