//! Oracle PCM f32 : snapshot local, voix (`-` pour VoiceDesign), corpus JSON.
//! Le corpus associe un nom à un texte. Aucun téléchargement n'est effectué.

use saragossa::TtsModel;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 3 {
        return Err("usage: tts_pcm_digest SNAPSHOT VOIX|- CORPUS.json".into());
    }
    let corpus: BTreeMap<String, String> = serde_json::from_slice(&std::fs::read(&args[2])?)?;
    let mut model = TtsModel::load_local(&args[0])?;
    model.set_voice((args[1] != "-").then_some(args[1].as_str()))?;
    model.set_seed(Some(0));
    for (name, text) in corpus {
        let cap = text.chars().count().saturating_mul(2).clamp(160, 800);
        for greedy in [true, false] {
            let mut digest = Sha256::new();
            let mut samples = 0;
            let mut consume = |pcm: &[f32]| {
                for value in pcm {
                    digest.update(value.to_le_bytes());
                }
                samples += pcm.len();
                Ok(())
            };
            let output = if greedy {
                model.synthesize_greedy_streaming(&text, cap, &mut consume)?
            } else {
                model.synthesize_default_streaming(&text, cap, &mut consume)?
            };
            if samples == 0 || samples != output.samples.len() {
                return Err("PCM streaming vide ou incomplet".into());
            }
            println!(
                "{}",
                serde_json::json!({
                    "name": name, "voice": args[1], "greedy": greedy,
                    "samples": samples, "frames": output.codes.len(),
                    "stop": output.stop_reason.as_str(),
                    "sha256": digest.finalize().iter().map(|byte| format!("{byte:02x}")).collect::<String>(),
                })
            );
        }
    }
    Ok(())
}
