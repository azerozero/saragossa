//! Oracle greedy et mesure à longueur exacte, à lancer sous verrou GPU.
#[cfg(gpu_metal)]
mod benchmark {
    use saragossa::decoder::{
        configure_prefix_disk_cache, flush_prefix_disk_cache, PrefixDiskSession,
    };
    use saragossa::{CausalDecoder, GenerationOptions, ModelAssets};
    use std::path::PathBuf;
    use std::time::{Duration, Instant};

    pub fn run() -> Result<(), Box<dyn std::error::Error>> {
        let args: Vec<_> = std::env::args().skip(1).collect();
        if args.len() != 6 {
            return Err(
            "usage: prefix_ssd MODEL_DIR CACHE_DIR TOKENS baseline|cold|warm ar|mtp OUTPUT_JSON"
                .into(),
        );
        }
        let count: usize = args[2].parse()?;
        if count == 0 || count > 1_048_576 {
            return Err("TOKENS hors limites".into());
        }
        let phase = args[3].as_str();
        if !matches!(phase, "baseline" | "cold" | "warm")
            || !matches!(args[4].as_str(), "ar" | "mtp")
        {
            return Err("phase ou mode invalide".into());
        }
        if phase != "baseline" {
            configure_prefix_disk_cache(&PathBuf::from(&args[1]), 32 << 30, 4096)?;
        }
        saragossa::force_resident_full_linear_decode();
        let assets = ModelAssets::load_local(&args[0])?;
        let seed = assets.encode_prompt("Le cache conserve exactement les états du préfixe. ")?;
        if seed.is_empty() {
            return Err("tokenizer vide".into());
        }
        let prompt: Vec<usize> = seed
            .iter()
            .cycle()
            .take(count)
            .map(|&id| id as usize)
            .collect();
        let mut decoder: CausalDecoder =
            saragossa::load_causal_decoder(&assets)?.with_metal_runtime()?;
        if args[4] == "mtp" {
            decoder = decoder.with_mtp_weights(&assets.mtp)?;
            #[cfg(all(gpu_metal, feature = "devtools"))]
            {
                decoder = decoder.with_dflash_draft_from_env(&assets.config)?;
            }
        }
        {
            let _warmup = saragossa::runtime_flags::StateSnapshotWarmup::enter();
            let _ = decoder.prefill_prompt_state_uncached(&prompt[..count.min(32)])?;
        }
        decoder.clear_prefix_cache();
        let _session = PrefixDiskSession::enter(Some("prefix-ssd-benchmark"));
        let options = GenerationOptions::default();
        let started = Instant::now();
        let (tokens, prefill_ms, reused) = if args[4] == "ar" {
            let loaded = decoder.load_persistent_prompt(&prompt)?;
            if loaded.is_some() != (phase == "warm") {
                return Err("hit SSD différent de la phase attendue".into());
            }
            let reused = if loaded.is_some() { count } else { 0 };
            let state = match loaded {
                Some(state) => state,
                None => decoder.prefill_prompt_state_uncached(&prompt)?,
            };
            let prefill = started.elapsed();
            let output = decoder.generate_greedy_timed_from_prompt_state_with_options(
                state, prefill, 32, &options,
            )?;
            (output.tokens, prefill.as_secs_f64() * 1000.0, reused)
        } else {
            let output = decoder.generate_greedy_mtp_streaming_with_options(
                &prompt,
                32,
                &options,
                2,
                |_| true,
            )?;
            (
                output.tokens,
                started
                    .elapsed()
                    .saturating_sub(output.loop_duration)
                    .as_secs_f64()
                    * 1000.0,
                output.stats.reused_prefix_tokens,
            )
        };
        let total_ms = started.elapsed().as_secs_f64() * 1000.0;
        flush_prefix_disk_cache(Duration::from_secs(600))?;
        if args[4] == "mtp" && ((reused > 0) != (phase == "warm")) {
            return Err("réutilisation MTP différente de la phase attendue".into());
        }
        let report = serde_json::json!({"phase":phase, "mode":args[4], "prompt_tokens":count, "generated_tokens":tokens,
        "prefill_or_reload_ms":prefill_ms, "generation_total_ms":total_ms, "reused_prefix_tokens":reused});
        std::fs::write(&args[5], serde_json::to_vec_pretty(&report)?)?;
        println!("{report}");
        Ok(())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(gpu_metal)]
    {
        benchmark::run()
    }
    #[cfg(not(gpu_metal))]
    {
        Err("Cet oracle nécessite macOS et la feature metal".into())
    }
}
