//! Oracles CPU : aucune construction de décodeur ni de poids.
use super::*;
use proptest::prelude::*;
use saragossa::{ModelConfig, MtpWeightsInfo, RustTokenizer, WeightCatalog};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

fn assets(tokenizer: RustTokenizer, family: &str) -> ModelAssets {
    let temp = tempfile::tempdir().expect("invariant: tempdir");
    let config = temp.path().join("config.json");
    std::fs::write(
        &config,
        json!({
            "model_type": family, "hidden_size": 2, "num_hidden_layers": 1,
            "num_attention_heads": 1, "num_key_value_heads": 1, "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0, "vocab_size": tokenizer.vocab_size(), "eos_token_id": 256
        })
        .to_string(),
    )
    .expect("invariant: config");
    ModelAssets {
        model_dir: temp.path().to_owned(),
        config: ModelConfig::from_file(config).expect("invariant: config"),
        tokenizer,
        shards: vec![],
        catalog: WeightCatalog::from_shards(&[]).expect("invariant: catalogue vide"),
        mtp: MtpWeightsInfo::default(),
    }
}

pub(crate) fn synthetic() -> ModelAssets {
    // Alphabet réversible GPT-2/ByteLevel, un token par octet.
    let mut next = 256;
    let alphabet: Vec<char> = (0..=255)
        .map(|byte| {
            let code = if (33..=126).contains(&byte)
                || (161..=172).contains(&byte)
                || (174..=255).contains(&byte)
            {
                byte
            } else {
                let code = next;
                next += 1;
                code
            };
            char::from_u32(code).expect("invariant: alphabet ByteLevel")
        })
        .collect();
    let mut vocab = serde_json::Map::new();
    for (id, c) in alphabet.iter().enumerate() {
        vocab.insert(c.to_string(), json!(id));
    }
    vocab.insert("<special>".into(), json!(256));
    vocab.insert(
        [0xef, 0xbf, 0xbd].iter().map(|b| alphabet[*b]).collect(),
        json!(257),
    );
    // Stop et octet UTF-8 incomplet dans le même token : arrêt immédiat requis.
    vocab.insert(format!("STOP{}", alphabet[0xc3]), json!(258));
    let tokenizer: tokenizers::Tokenizer = serde_json::from_value(json!({
        "version":"1.0", "added_tokens":[{"id":256,"content":"<special>",
        "single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],
        "decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},
        "model":{"type":"WordLevel","vocab":vocab,"unk_token":"<special>"}
    })).expect("invariant: tokenizer");
    let temp = tempfile::tempdir().expect("invariant: tempdir");
    let path = temp.path().join("tokenizer.json");
    tokenizer.save(&path, false).expect("invariant: sauvegarde");
    assets(
        RustTokenizer::from_file(path).expect("invariant: tokenizer"),
        "qwen3",
    )
}

fn bytes(text: &str) -> Vec<u32> {
    text.bytes().map(u32::from).collect()
}

fn raw_oracle(tokenizer: &RustTokenizer, ids: &[u32]) {
    for skip in [false, true] {
        let mut stream = tokenizer
            .byte_level_decode_stream(skip)
            .expect("ByteLevel qualifié");
        let mut stable = String::new();
        for (i, id) in ids.iter().enumerate() {
            if let Some(delta) = stream.step(*id).expect("invariant: step") {
                stable.push_str(&delta);
            }
            let full = tokenizer
                .decode(&ids[..=i], skip)
                .expect("invariant: oracle");
            // À TOUT point de coupe, même au milieu d'un caractère invalide.
            assert_eq!(
                format!("{stable}{}", stream.pending_text().expect("queue")),
                full,
                "skip={skip}, ids={:?}",
                &ids[..=i]
            );
        }
    }
}

fn serve_oracle(assets: &ModelAssets, ids: &[u32], stops: &[String]) -> usize {
    let mut stream = StreamingTextDetokenizer::new(assets, stops, ids.len());
    let mut stop = TextStop::new(assets, stops, ids.len());
    let mut output = String::new();
    let mut full = String::new();
    for (i, id) in ids.iter().enumerate() {
        stream
            .push_token(*id as usize, &mut |event| {
                if let CompletionStreamEvent::Delta(delta) = event {
                    output.push_str(delta);
                }
                Ok(())
            })
            .expect("invariant: streaming");
        full = decode_completion_tokens(assets, &ids[..=i]).expect("invariant: oracle");
        let text = strip_empty_think_streaming(&full);
        let stopped = contains_stop(text, stops);
        assert_eq!(stream.stopped(), stopped, "token {i}: {text:?}");
        assert_eq!(stop.push(*id as usize), !stopped, "TextStop token {i}");
        let visible = streaming_visible_text(&full, stops);
        if !visible.ends_with('\u{fffd}') {
            assert_eq!(output, visible, "token {i}");
        }
        assert!(visible.starts_with(&output));
        if stopped {
            break;
        }
    }
    stop.finish().expect("invariant: TextStop");
    // finish_completion ne retire qu'un bloc think COMPLET.
    let text = full
        .strip_prefix(saragossa::QWEN_EMPTY_THINK_BLOCK)
        .unwrap_or(&full);
    let content = strip_after_first_stop(text, stops);
    stream
        .finish(content, &mut |event| {
            if let CompletionStreamEvent::Delta(delta) = event {
                output.push_str(delta);
            }
            Ok(())
        })
        .expect("invariant: finish");
    assert_eq!(output, content);
    stream.fallback_decodes
}

#[test]
fn only_qualified_stream_degradation_enables_diagnostics() {
    let qualified = synthetic();
    let gemma = assets(synthetic().tokenizer, "gemma4");
    for (assets, expected_degraded) in [(&qualified, true), (&gemma, false)] {
        let mut stream = StreamingTextDetokenizer::new(assets, &[], 65);
        for _ in 0..64 {
            stream.push_token(257, &mut |_| Ok(())).expect("step");
            assert!(!stream.degraded);
        }
        stream.push_token(257, &mut |_| Ok(())).expect("step");
        assert_eq!(stream.degraded, expected_degraded);
        assert_eq!(
            stream.fallback_decodes,
            if expected_degraded { 1 } else { 65 }
        );
    }
}

#[test]
fn bytelevel_utf8_specials_leading_spaces_and_all_finish_boundaries() {
    let assets = synthetic();
    for ids in [
        bytes("  café 👩‍💻 中文\n"),
        vec![0xc3, 256, 0xa9, 257, 256],
        vec![256, 256, 32, 65, 256],
        vec![258, 0xa9],
        bytes(saragossa::QWEN_EMPTY_THINK_BLOCK),
        bytes(&format!("{}Bonjour", saragossa::QWEN_EMPTY_THINK_BLOCK)),
    ] {
        raw_oracle(&assets.tokenizer, &ids);
        for cut in 0..=ids.len() {
            assert_eq!(serve_oracle(&assets, &ids[..cut], &[]), 0);
            serve_oracle(
                &assets,
                &ids[..cut],
                &["STOP".into(), "é".into(), "\n\n".into()],
            );
        }
    }
    // Les suites invisibles ne font pas grossir la fenêtre.
    assert_eq!(serve_oracle(&assets, &vec![256; 512], &[]), 0);
    assert!(serve_oracle(&assets, &vec![257; 80], &[]) > 0);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]
    #[test]
    fn arbitrary_bytes_match_full_decode(ids in prop::collection::vec(0u32..260, 0..100)) {
        let assets = synthetic();
        raw_oracle(&assets.tokenizer, &ids);
        serve_oracle(&assets, &ids, &["STOP".into(), "é".into(), "�".into()]);
        serve_oracle(&assets, &ids, &[]);
    }
    #[test]
    fn stops_cross_every_byte_boundary(prefix in "[a-z ]{0,30}", stop in "[A-Zé🌍]{1,8}") {
        let assets = synthetic();
        let ids = bytes(&format!("{prefix}{stop}ignored"));
        serve_oracle(&assets, &ids, &[stop]);
    }
}

#[test]
fn stops_at_zero_overlapping_and_pending_utf8() {
    let assets = synthetic();
    for (ids, stops) in [
        (bytes("STOPignored"), vec!["STOP".into()]),
        (
            bytes("abcabcEND"),
            vec!["bc".into(), "abc".into(), "".into()],
        ),
        (vec![258, 0xa9], vec!["STOP".into()]),
        (bytes("hello é🌍done"), vec!["é🌍".into()]),
        (bytes("hello ST"), vec!["STOP".into()]),
    ] {
        assert_eq!(serve_oracle(&assets, &ids, &stops), 0);
    }
}

fn local_fixture(family: &str) -> Option<(ModelAssets, Vec<Vec<u32>>)> {
    use sha2::{Digest, Sha256};
    let file = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(format!("tests/fixtures/tokenizer/{family}.jsonl"));
    let data = std::fs::read_to_string(file).expect("invariant: golden");
    let mut rows = data
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).expect("golden JSON"));
    let header = rows.next().expect("en-tête");
    let variable = format!("SARAGOSSA_TOKENIZER_{}", family.to_uppercase());
    let explicit = std::env::var_os(&variable).map(PathBuf::from);
    let path = explicit.clone().or_else(|| {
        let hub = std::env::var_os("HF_HUB_CACHE")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HF_HOME").map(|p| PathBuf::from(p).join("hub")))
            .or_else(|| {
                std::env::var_os("HOME").map(|p| PathBuf::from(p).join(".cache/huggingface/hub"))
            })?;
        let source = header["source"].as_str()?;
        if let Some(name) = source.strip_prefix("local:") {
            for repo in std::fs::read_dir(hub).ok()?.flatten() {
                if !repo
                    .file_name()
                    .to_string_lossy()
                    .ends_with(&format!("--{name}"))
                {
                    continue;
                }
                for entry in std::fs::read_dir(repo.path().join("snapshots"))
                    .ok()?
                    .flatten()
                {
                    let path = entry.path().join("tokenizer.json");
                    if path.is_file() {
                        return Some(path);
                    }
                }
            }
            None
        } else {
            let (repo, revision) = source.split_once('@')?;
            Some(hub.join(format!(
                "models--{}/snapshots/{revision}/tokenizer.json",
                repo.replace('/', "--")
            )))
        }
    });
    let Some(path) = path.filter(|p| explicit.is_some() || p.is_file()) else {
        eprintln!("SKIP streaming tokenizer {family}: définir {variable}");
        return None;
    };
    assert_eq!(
        format!(
            "{:x}",
            Sha256::digest(std::fs::read(&path).expect("fixture locale"))
        ),
        header["tokenizer_sha256"].as_str().expect("SHA")
    );
    let ids = rows
        .flat_map(|row| {
            ["ids", "ids_special"].map(|key| serde_json::from_value(row[key].clone()).expect("ids"))
        })
        .collect();
    Some((
        assets(
            RustTokenizer::from_file(path).expect("tokenizer local"),
            if family == "qwen3" { "qwen3" } else { family },
        ),
        ids,
    ))
}

#[test]
fn local_qwen_and_gemma_goldens_all_cuts_and_random_ids() {
    use proptest::test_runner::{Config, TestRunner};
    for family in ["qwen3", "gemma3", "gemma4"] {
        let Some((assets, sequences)) = local_fixture(family) else {
            continue;
        };
        let qualified = family == "qwen3";
        assert_eq!(
            assets.tokenizer.byte_level_decode_stream(true).is_some(),
            qualified
        );
        if family == "gemma4" {
            // Oracle indépendant : chaque marqueur ferme un segment Metaspace.
            let segment = assets.tokenizer.encode(" suite é").expect("segment");
            let decoded = assets.tokenizer.decode(&segment, true).expect("segment");
            let mut ids = segment.clone();
            let mut expected = decoded.clone();
            for marker in [
                "<|tool_call>",
                "<tool_call|>",
                "<|\"|>",
                "<|channel>",
                "<channel|>",
            ] {
                ids.push(
                    assets
                        .tokenizer
                        .token_to_id(marker)
                        .expect("contrôle Gemma 4"),
                );
                ids.extend_from_slice(&segment);
                expected.push_str(marker);
                expected.push_str(&decoded);
            }
            assert_eq!(
                decode_completion_tokens(&assets, &ids).expect("contrôles"),
                expected
            );
            assert_eq!(serve_oracle(&assets, &ids, &[]), ids.len());
        }
        for ids in &sequences {
            if qualified {
                raw_oracle(&assets.tokenizer, ids);
            }
            for cut in 0..=ids.len() {
                let fallbacks = serve_oracle(&assets, &ids[..cut], &[]);
                if qualified {
                    assert_eq!(fallbacks, 0);
                } else {
                    assert_eq!(fallbacks, cut);
                }
            }
            serve_oracle(&assets, ids, &["é".into(), "\n".into(), "user".into()]);
        }
        let strategy = prop::collection::vec(0..assets.tokenizer.vocab_size() as u32, 0..80);
        let mut runner = TestRunner::new(Config {
            cases: 128,
            ..Config::default()
        });
        runner
            .run(&strategy, |ids| {
                if qualified {
                    raw_oracle(&assets.tokenizer, &ids);
                }
                serve_oracle(&assets, &ids, &["STOP".into()]);
                Ok(())
            })
            .expect("invariant: propriété locale");
        eprintln!(
            "PASS streaming {family}: {} séquences, tous les points de coupe, 128 aléatoires",
            sequences.len()
        );
    }
}

/// Microbench manuel reproductible du chemin serve, avec tokenizer local SHA
/// vérifié et sans GPU. Chaque paire alterne l'ordre ; préparation hors mesure.
#[test]
#[ignore = "microbench CPU release ; SARAGOSSA_TOKENIZER_QWEN3 requis"]
fn detokenizer_cpu_bench() {
    use std::{hint::black_box, time::Instant};
    let (assets, _) = local_fixture("qwen3").expect("fixture Qwen requise");
    let corpus = assets
        .tokenizer
        .encode("Bonjour, le moteur produit du texte en français. Hello world! 中文 👩‍💻\n")
        .expect("invariant: corpus");
    let stops = vec![
        "<STOP jamais produit>".to_string(),
        "END OF RESPONSE".to_string(),
    ];
    for length in [200, 1024, 8192, 32768] {
        let ids: Vec<u32> = corpus.iter().copied().cycle().take(length + 128).collect();
        let mut samples = [Vec::new(), Vec::new()];
        for repetition in 0..10 {
            for mode in if repetition % 2 == 0 { [0, 1] } else { [1, 0] } {
                // À 200 tokens : réponse entière. Sinon fenêtre des 128 derniers
                // pas après N tokens, soit le coût marginal de l'audit X1.
                let (warm, end) = if length == 200 {
                    (0, 200)
                } else {
                    (length, length + 128)
                };
                let duration = if mode == 0 {
                    // Copie du chemin parent : decode complet, contains, find,
                    // suffixe, strip_prefix et copie du snapshot émis.
                    let mut generated = ids[..warm].to_vec();
                    let mut emitted = streaming_visible_text(
                        &decode_completion_tokens(&assets, &generated).expect("oracle"),
                        &stops,
                    )
                    .to_owned();
                    let start = Instant::now();
                    for id in &ids[warm..end] {
                        generated.push(*id);
                        let decoded =
                            decode_completion_tokens(&assets, &generated).expect("oracle");
                        black_box(contains_stop(strip_empty_think_streaming(&decoded), &stops));
                        let visible = streaming_visible_text(&decoded, &stops);
                        let delta = visible.strip_prefix(&emitted).expect("préfixe");
                        if !delta.is_empty() && !visible.ends_with('�') {
                            black_box(delta);
                            emitted = visible.to_owned();
                        }
                    }
                    start.elapsed()
                } else {
                    let mut stream = StreamingTextDetokenizer::new(&assets, &stops, ids.len());
                    for id in &ids[..warm] {
                        stream
                            .push_token(*id as usize, &mut |_| Ok(()))
                            .expect("warm");
                    }
                    let start = Instant::now();
                    for id in &ids[warm..end] {
                        stream
                            .push_token(*id as usize, &mut |event| {
                                if let CompletionStreamEvent::Delta(delta) = event {
                                    black_box(delta);
                                }
                                Ok(())
                            })
                            .expect("step");
                    }
                    let elapsed = start.elapsed();
                    assert_eq!(stream.fallback_decodes, 0);
                    assert_eq!(
                        stream.decoded,
                        decode_completion_tokens(&assets, &ids[..end]).expect("oracle")
                    );
                    elapsed
                };
                if repetition > 0 {
                    samples[mode].push(duration.as_secs_f64() * 1000.0 / (end - warm) as f64);
                }
            }
        }
        for sample in &mut samples {
            sample.sort_by(f64::total_cmp);
        }
        eprintln!(
            "CPU N={length} ms/token avant={:.6} après={:.6} après_min={:.6} après_max={:.6}",
            samples[0][4], samples[1][4], samples[1][0], samples[1][8]
        );
    }
}
