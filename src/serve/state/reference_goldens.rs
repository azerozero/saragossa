//! Exerce prompt_encoding lui-même, sans charger ni exécuter les poids LLM.
use super::*;

#[allow(dead_code)]
#[path = "../../golden/external.rs"]
mod external;
use external::*;

fn template_model(path: &std::path::Path) -> TestResult<LoadedModel> {
    let assets = ModelAssets {
        model_dir: path.to_path_buf(),
        config: saragossa::ModelConfig::from_file(path.join("config.json"))?,
        tokenizer: saragossa::RustTokenizer::from_file(path.join("tokenizer.json"))?,
        shards: Vec::new(),
        catalog: saragossa::WeightCatalog::from_shards(&[])?,
        mtp: Default::default(),
    };
    // Sentinelle minimale : prompt_encoding ne consulte jamais le décodeur.
    let mut weights: std::collections::HashMap<String, saragossa::Tensor> = [
        (
            "embed_tokens.weight".to_string(),
            saragossa::Tensor::from_vec(vec![1, 1], vec![1.0])?,
        ),
        (
            "norm.weight".to_string(),
            saragossa::Tensor::from_vec(vec![1], vec![1.0])?,
        ),
        (
            "lm_head.weight".to_string(),
            saragossa::Tensor::from_vec(vec![1, 1], vec![1.0])?,
        ),
    ]
    .into_iter()
    .collect();
    weights.insert(
        "layers.0.input_layernorm.weight".into(),
        saragossa::Tensor::from_vec(vec![1], vec![1.0])?,
    );
    for projection in ["q_proj", "k_proj", "v_proj", "o_proj"] {
        weights.insert(
            format!("layers.0.self_attn.{projection}.weight"),
            saragossa::Tensor::from_vec(vec![1, 1], vec![1.0])?,
        );
    }
    let decoder = CausalDecoder::from_tensors(
        weights,
        saragossa::CausalDecoderConfig {
            rope_theta: None,
            ..Default::default()
        },
    )?;
    Ok(LoadedModel {
        id: "reference".into(),
        assets,
        decoder,
        preset: None,
        mtp_active: false,
        json_token_catalog: OnceLock::new(),
        prefix_cache: BlockAwarePrefixCache::from_runtime_flags(),
    })
}

#[test]
fn external_serve_template_fixtures_parse() -> TestResult {
    for (name, _) in CHAT_MODELS {
        let fixture: ChatFixture = json(&format!("chat_templates/{name}"))?;
        assert_eq!(fixture.model, name);
        assert!(!fixture.cases.is_empty());
    }
    Ok(())
}

#[test]
#[ignore = "CPU sans poids: SARAGOSSA_QWEN38_DIR, SARAGOSSA_QWEN36_MOE_DIR, SARAGOSSA_GEMMA3_DIR, SARAGOSSA_GEMMA4_DIR"]
fn external_serve_templates_match_transformers() -> TestResult {
    let mut failures = Vec::new();
    let mut compared = 0;
    for (name, env) in CHAT_MODELS {
        let path = local(env)
            .ok_or_else(|| format!("{env} requis pour comparer réellement les références"))?;
        let stem = format!("chat_templates/{name}");
        check_model(&stem, &path)?;
        let loaded = template_model(&path)?;
        let fixture: ChatFixture = json(&stem)?;
        for case in fixture.cases {
            // serve ouvre toujours le tour de génération et refuse thinking=true.
            if !case.add_generation_prompt || case.enable_thinking == Some(true) {
                continue;
            }
            let thinking = case
                .enable_thinking
                .map(|b| format!(r#", "chat_template_kwargs":{{"enable_thinking":{b}}}"#))
                .unwrap_or_default();
            let wire = format!(
                r#"{{"model":"reference","messages":{},"tools":{}{thinking}}}"#,
                case.messages.get(),
                case.tools.get()
            );
            let request = ChatCompletionRequest::parse(wire.as_bytes())?;
            let label = format!("{name}/{}", case.name);
            compared += 1;
            match (loaded.prompt_encoding(&request), case.token_ids) {
                (Ok(actual), Some(expected)) => {
                    let expected: Vec<usize> = expected.into_iter().map(|v| v as usize).collect();
                    if actual.tokens != expected {
                        failures.push(format!(
                            "{label}: Rust tokens={:?}; référence={expected:?}",
                            actual.tokens
                        ));
                    }
                }
                (Err(ServeError::Http(error)), None)
                    if error.contains("Conversation roles must alternate") => {}
                (Err(error), None) => {
                    failures.push(format!("{label}: mauvaise catégorie de rejet: {error}"))
                }
                (Ok(_), None) => failures.push(format!(
                    "{label}: Rust accepte, référence rejette {:?}",
                    case.error
                )),
                (Err(error), Some(_)) => {
                    failures.push(format!("{label}: Rust rejette {error}, référence accepte"))
                }
            }
        }
    }
    assert_eq!(compared, 8);
    assert!(
        failures.is_empty(),
        "{} écarts serve:\n{}",
        failures.len(),
        failures.join("\n")
    );
    Ok(())
}
