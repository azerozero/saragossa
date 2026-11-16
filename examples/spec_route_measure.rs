//! Worker JSONL du routage spéculatif : AR, MTP, DFlash et routage par classe
//! de prompt, tous par les API publiques de production.
//! Chaque ligne stdin contient {mode, prompt, context_tokens?, tokens}.
//! Le drafter vient de `SARAGOSSA_RUST_DFLASH_DRAFT_DIR` ; les poids restent
//! chargés et chaque requête repart d'un préfixe froid.

#[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use saragossa::{
        load_causal_decoder_for_metal, spec_route::prefers_dflash, GenerationOptions,
        KernelProfile, MetalExecutor, ModelAssets, SpeculativeOutput, SpeculativeStats,
    };
    use serde::Deserialize;
    use serde_json::json;
    use std::{
        io::{self, BufRead, Write},
        path::Path,
        time::Instant,
    };

    #[derive(Deserialize)]
    struct Request {
        mode: String,
        prompt: String,
        #[serde(default)]
        context_tokens: usize,
        tokens: usize,
    }
    const BACKGROUND: &str = "Le journal décrit les opérations quotidiennes du port. Les équipes vérifient les livraisons, consignent les incidents et préparent le travail du lendemain.\n";
    let model = std::env::args().nth(1).ok_or("model_dir requis")?;
    let assets = ModelAssets::load_local(&model)?;
    let executor =
        MetalExecutor::new_with_kernel_profile(KernelProfile::for_model_dir(Path::new(&model)))?;
    let decoder = load_causal_decoder_for_metal(&assets, &executor)?
        .with_metal_executor(executor)?
        .with_mtp_sidecar(assets.mtp.path.as_ref().ok_or("sidecar MTP requis")?)?
        .with_dflash_draft_from_env(&assets.config)?;
    if decoder.dflash_draft().is_none() {
        return Err("SARAGOSSA_RUST_DFLASH_DRAFT_DIR requis".into());
    }
    let options = GenerationOptions::default();
    println!("{}", json!({"ready": true}));
    io::stdout().flush()?;
    for line in io::stdin().lock().lines() {
        let request: Request = serde_json::from_str(&line?)?;
        let user = assets.encode_prompt(&request.prompt)?;
        let head = assets.encode_prompt("<|im_start|>user\n")?;
        let tail =
            assets.encode_prompt("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")?;
        let mut prompt = head;
        let mut user_text = String::new();
        if request.context_tokens > 0 {
            let reserved = prompt.len() + user.len() + tail.len();
            let fill = request
                .context_tokens
                .checked_sub(reserved)
                .ok_or("contexte trop court")?;
            let background = assets.encode_prompt(BACKGROUND)?;
            prompt.extend(background.iter().copied().cycle().take(fill));
            // Le classifieur ne lit que la fin du dernier paragraphe : une copie
            // du fond suffit pour lui présenter la même fin de texte.
            user_text.push_str(BACKGROUND);
        }
        user_text.push_str(&request.prompt);
        prompt.extend(user);
        prompt.extend(tail);
        let prompt: Vec<usize> = prompt.into_iter().map(|id| id as usize).collect();
        let started = Instant::now();
        let mut first = None;
        let mut seen = Vec::new();
        let mut callback = |token| {
            first.get_or_insert_with(|| started.elapsed());
            seen.push(token);
            true
        };
        let routed = match request.mode.as_str() {
            "ar" => None,
            "mtp" => Some(false),
            "dflash" => Some(true),
            "route" => Some(prefers_dflash([("user", user_text.as_str())], false)),
            _ => return Err("mode inconnu".into()),
        };
        let output = match routed {
            None => {
                let ar = decoder.generate_greedy_streaming_with_options(
                    &prompt,
                    request.tokens,
                    &options,
                    &mut callback,
                )?;
                SpeculativeOutput {
                    tokens: ar.tokens,
                    stats: SpeculativeStats::default(),
                    loop_duration: ar.timings.decode,
                }
            }
            Some(allow_dflash) => decoder.generate_greedy_mtp_routed(
                &prompt,
                request.tokens,
                &options,
                2,
                allow_dflash,
                &mut callback,
            )?,
        };
        let elapsed = started.elapsed();
        if seen != output.tokens || output.tokens.len() != request.tokens {
            return Err("IDs streamés ou budget incorrects".into());
        }
        println!(
            "{}",
            json!({
                "mode": request.mode, "prompt_tokens": prompt.len(), "ids": output.tokens,
                "routed": routed.map(|dflash| if dflash { "dflash" } else { "mtp" }),
                "ttft_ms": first.ok_or("aucun token")?.as_secs_f64() * 1000.0,
                "total_ms": elapsed.as_secs_f64() * 1000.0,
                "decode_ms": output.loop_duration.as_secs_f64() * 1000.0,
                "decode_tok_s": request.tokens as f64 / output.loop_duration.as_secs_f64(),
                "accepted": output.stats.accepted, "proposed": output.stats.proposed,
                "verifications": output.stats.verifications,
                "stats": format!("{:?}", output.stats),
            })
        );
        io::stdout().flush()?;
    }
    Ok(())
}

#[cfg(not(all(target_os = "macos", feature = "metal", feature = "devtools")))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err("cet exemple nécessite macOS et les features metal,devtools".into())
}
