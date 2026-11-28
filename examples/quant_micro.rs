//! Mesure locale sans golden : `quant_micro capture|score|bench MODEL ARTIFACT`.

#[cfg(gpu_metal)]
mod macos {
    use saragossa::{GenerationOptions, ModelAssets};
    use serde::{Deserialize, Serialize};
    use sha2::{Digest, Sha256};
    use std::{fs, path::Path};

    /// sha2 0.11 renvoie un `hybrid_array::Array` sans `LowerHex`.
    fn hex(digest: impl AsRef<[u8]>) -> String {
        digest
            .as_ref()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }

    type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

    #[derive(Deserialize, Serialize)]
    struct Protocol {
        chunks: Vec<Vec<usize>>,
        scored: usize,
        top_k: usize,
    }
    #[derive(Deserialize, Serialize)]
    struct Reference {
        tokenizer_sha256: String,
        vocab_sha256: String,
        config_sha256: String,
        protocol: Protocol,
        rows: Vec<(Vec<usize>, Vec<f64>)>,
    }

    fn hash(path: &Path) -> Result<String> {
        Ok(hex(Sha256::digest(fs::read(path)?)))
    }

    fn partition(values: &[f32], top: usize) -> Result<(Vec<usize>, Vec<f64>)> {
        if values.iter().any(|v| !v.is_finite()) {
            return Err("logits non finis".into());
        }
        let top = top.min(values.len());
        if top == 0 {
            return Err("partition KL vide".into());
        }
        let mut ids: Vec<_> = (0..values.len()).collect();
        let compare = |a: &usize, b: &usize| values[*b].total_cmp(&values[*a]).then(a.cmp(b));
        ids.select_nth_unstable_by(top - 1, compare);
        ids.truncate(top);
        ids.sort_by(compare);
        let max = f64::from(values[ids[0]]);
        let log_z = max
            + values
                .iter()
                .map(|&v| (f64::from(v) - max).exp())
                .sum::<f64>()
                .ln();
        ids.truncate(top);
        let probs = ids.iter().map(|&i| f64::from(values[i]) - log_z).collect();
        Ok((ids, probs))
    }

    pub(super) fn main() -> Result {
        let args: Vec<_> = std::env::args().skip(1).collect();
        if args.len() != 3 || !matches!(args[0].as_str(), "capture" | "score" | "bench") {
            return Err("usage: quant_micro capture|score|bench MODEL ARTIFACT".into());
        }
        let path = Path::new(&args[1]);
        let output = Path::new(&args[2]);
        if args[0] != "score" && output.exists() {
            return Err("artefact de sortie déjà présent".into());
        }
        let protocol: Protocol = serde_json::from_slice(include_bytes!(
            "../tests/fixtures/goldens/kld/qwen38_meta.bin"
        ))?;
        let tokenizer_sha256 = hash(&path.join("tokenizer.json"))?;
        let tokenizer_json: serde_json::Value =
            serde_json::from_slice(&fs::read(path.join("tokenizer.json"))?)?;
        // Clés triées : mlx_lm réordonne les champs des added_tokens sans
        // changer leur contenu.
        let mut vocab = serde_json::json!({"vocab":tokenizer_json["model"]["vocab"],"added_tokens":tokenizer_json["added_tokens"]});
        vocab.sort_all_objects();
        let vocab_sha256 = hex(Sha256::digest(serde_json::to_vec(&vocab)?));
        // Les IDs du protocole sont ceux du banc MLX. Les conversions mlx_lm
        // réécrivent le pré-tokenizer : on compare donc le vocabulaire, sans
        // re-tokeniser le corpus avec le tokenizer du candidat.
        let reference: Option<Reference> = if args[0] == "score" {
            Some(serde_json::from_slice(&fs::read(output)?)?)
        } else {
            None
        };
        if let Some(reference) = &reference {
            if reference.vocab_sha256 != vocab_sha256
                || reference.protocol.chunks != protocol.chunks
            {
                return Err("tokenizers référence/candidat différents".into());
            }
            if reference.protocol.scored != protocol.scored
                || reference.protocol.top_k != protocol.top_k
                || reference.rows.len() != protocol.chunks.len() * protocol.scored
                || reference.rows.iter().any(|(ids, probs)| {
                    ids.len() != protocol.top_k
                        || probs.len() != protocol.top_k
                        || probs.iter().any(|p| !p.is_finite() || *p > 0.0)
                })
            {
                return Err("référence KL incohérente".into());
            }
        }
        let assets = ModelAssets::load_local(path)?;
        let executor = saragossa::MetalExecutor::new()?;
        let model = saragossa::load_causal_decoder_for_metal(&assets, &executor)?
            .with_metal_executor(executor)?;
        if args[0] == "bench" {
            let corpus: Vec<usize> = protocol.chunks.iter().flatten().copied().collect();
            // Mêmes variables que le mlx_bench.py du banc : longueurs, tokens
            // décodés et essais (le premier est la chauffe).
            let env =
                |name: &str, default: &str| std::env::var(name).unwrap_or_else(|_| default.into());
            let lengths = env("BENCH_LENGTHS", "1024,4096")
                .split(',')
                .map(str::parse)
                .collect::<std::result::Result<Vec<usize>, _>>()?;
            let decode: usize = env("BENCH_DECODE", "64").parse()?;
            let trials: usize = env("BENCH_TRIALS", "4").parse()?;
            let mut results = Vec::new();
            for n in lengths {
                let prompt: Vec<_> = corpus.iter().copied().cycle().take(n).collect();
                for trial in 0..trials {
                    let started = std::time::Instant::now();
                    let mut emissions = Vec::new();
                    let generated = model.generate_greedy_streaming_with_options(
                        &prompt,
                        decode,
                        &GenerationOptions::default(),
                        |_| {
                            emissions.push(started.elapsed().as_secs_f64());
                            true
                        },
                    )?;
                    let first = *emissions.first().ok_or("aucun token généré")?;
                    let last = *emissions.last().ok_or("aucun token généré")?;
                    let decode_tokens = generated.tokens.len().saturating_sub(1);
                    let row = serde_json::json!({"length":n,"trial":trial,"warmup":trial==0,"prompt":prompt,"tokens":generated.tokens,"prefill_seconds":first,"decode_seconds":last-first,"decode_tokens":decode_tokens});
                    eprintln!("BENCH length={n} trial={trial} prefill_s={first} decode_s={} decode_tokens={decode_tokens}", last-first);
                    results.push(row);
                }
            }
            fs::write(output, serde_json::to_vec(&results)?)?;
            return Ok(());
        }
        let protocol = reference.as_ref().map_or(&protocol, |r| &r.protocol);
        let mut rows = Vec::new();
        let (mut kls, mut deltas, mut same) = (Vec::new(), Vec::new(), 0usize);
        for (chunk_index, chunk) in protocol.chunks.iter().enumerate() {
            let start = chunk
                .len()
                .checked_sub(protocol.scored)
                .filter(|&n| n > 0)
                .ok_or("chunk KL invalide")?;
            let (mut cache, mut logits) = model.prefill_cache(&chunk[..start])?;
            for position in start..chunk.len() {
                let values = logits.data();
                let row = partition(values, protocol.top_k)?;
                if let Some(reference) = &reference {
                    let (ids, logprobs) = reference
                        .rows
                        .get(kls.len())
                        .ok_or("référence trop courte")?;
                    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
                    let log_z = max
                        + values
                            .iter()
                            .map(|&v| (f64::from(v) - max).exp())
                            .sum::<f64>()
                            .ln();
                    let p: Vec<_> = logprobs.iter().map(|v| v.exp()).collect();
                    let q: Vec<_> = ids
                        .iter()
                        .map(|&id| {
                            values
                                .get(id)
                                .map(|&v| (f64::from(v) - log_z).exp())
                                .ok_or_else(|| "ID de référence hors vocabulaire".into())
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let rest = (
                        (1.0 - p.iter().sum::<f64>()).max(0.0),
                        (1.0 - q.iter().sum::<f64>()).max(0.0),
                    );
                    let (mut kl, mut delta) = (0.0, 0.0_f64);
                    for (p, q) in p.into_iter().zip(q).chain([rest]) {
                        if p > 0.0 {
                            kl += p * (p / q.max(f64::MIN_POSITIVE)).ln();
                        }
                        delta = delta.max((p - q).abs());
                    }
                    kls.push(kl);
                    deltas.push(delta);
                    same += usize::from(row.0[0] == ids[0]);
                } else {
                    rows.push(row);
                }
                if position + 1 < chunk.len() {
                    logits = model.next_logits_cached(&mut cache, chunk[position])?;
                }
            }
            eprintln!(
                "KL chunk {}/{} terminé",
                chunk_index + 1,
                protocol.chunks.len()
            );
        }
        if reference.is_some() {
            deltas.sort_by(f64::total_cmp);
            let positions = kls.len();
            if positions == 0 {
                return Err("corpus vide".into());
            }
            println!(
                "{}",
                serde_json::json!({"positions":positions,"mean_kl":kls.iter().sum::<f64>() / positions as f64,"p99_delta_p":deltas[(0.99 * (positions - 1) as f64) as usize],"max_delta_p":deltas[positions-1],"same_top":same as f64 / positions as f64})
            );
        } else {
            let artifact = Reference {
                tokenizer_sha256,
                vocab_sha256,
                config_sha256: hash(&path.join("config.json"))?,
                protocol: Protocol {
                    chunks: protocol.chunks.clone(),
                    scored: protocol.scored,
                    top_k: protocol.top_k,
                },
                rows,
            };
            fs::write(output, serde_json::to_vec(&artifact)?)?;
        }
        Ok(())
    }
}

#[cfg(gpu_metal)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    macos::main()
}

#[cfg(not(gpu_metal))]
fn main() {
    eprintln!("quant_micro exige macOS avec la feature metal");
}
