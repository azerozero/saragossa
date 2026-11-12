//! Capture locale Gemma 4 MoE : `dump_layers MODEL IDS_JSON OUT [cpu|metal]`.

use saragossa::{CausalDecoder, ModelAssets};
use std::path::Path;

fn load(assets: &ModelAssets, backend: &str) -> Result<CausalDecoder, Box<dyn std::error::Error>> {
    match backend {
        "cpu" => Ok(saragossa::load_causal_decoder(assets)?),
        #[cfg(all(target_os = "macos", feature = "metal"))]
        "metal" => {
            let executor = saragossa::MetalExecutor::new()?;
            Ok(saragossa::load_causal_decoder_for_metal(assets, &executor)?
                .with_metal_executor(executor)?)
        }
        _ => Err("backend indisponible (cpu, ou metal sur macOS avec feature metal)".into()),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(3..=4).contains(&args.len()) {
        return Err("usage: dump_layers MODEL IDS_JSON OUT [cpu|metal] (défaut metal)".into());
    }
    let ids: Vec<usize> = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    if ids.is_empty() || Path::new(&args[2]).exists() {
        return Err("prompt vide ou répertoire de sortie déjà existant".into());
    }
    let assets = ModelAssets::load_local(&args[0])?;
    if !assets.config.is_gemma4() || !assets.config.enable_moe_block {
        return Err("ce harnais exige un checkpoint Gemma 4 MoE local".into());
    }
    let backend = args.get(3).map_or("metal", String::as_str);
    let decoder = load(&assets, backend)?;
    let logits = decoder.dump_gemma4_layers(&ids, Path::new(&args[2]))?;
    eprintln!(
        "capture {backend} terminée ; greedy={}",
        saragossa::argmax(logits.data())?
    );
    Ok(())
}
