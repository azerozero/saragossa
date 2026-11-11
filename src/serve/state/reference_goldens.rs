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

fn compare() -> TestResult<Vec<String>> {
    let mut failures = Vec::new();
    for (name, env) in CHAT_MODELS {
        let path = local(env)?;
        let stem = format!("chat_templates/{name}");
        check_model(&stem, &path)?;
        let loaded = template_model(&path)?;
        let fixture: ChatFixture = json(&stem)?;
        for case in fixture.cases {
            // serve ouvre toujours le tour de génération et refuse thinking=true.
            if !case.add_generation_prompt || case.enable_thinking == Some(true) {
                continue;
            }
            let mut wire = serde_json::json!({"model":"reference", "messages":case.messages, "tools":case.tools});
            if let Some(thinking) = case.enable_thinking {
                wire["chat_template_kwargs"] = serde_json::json!({"enable_thinking":thinking});
            }
            let request: ChatCompletionRequest = serde_json::from_value(wire)?;
            let label = format!("{name}/{}", case.name);
            let mismatch = match (loaded.prompt_encoding(&request), case.token_ids) {
                (Ok(actual), Some(expected)) => {
                    let expected: Vec<usize> = expected.into_iter().map(|v| v as usize).collect();
                    (actual.tokens != expected).then(|| {
                        format!(
                            "{label}: Rust tokens={:?}; référence={expected:?}",
                            actual.tokens
                        )
                    })
                }
                (Err(error), None) if is_role_rejection(&error, case.error.as_deref()) => None,
                (Err(error), None) => {
                    // Une panne de tokenizer/IO ou un autre HTTP 400 n'est pas un rejet de rôle.
                    return Err(format!("{label}: catégorie de rejet inattendue: {error}").into());
                }
                (Ok(_), None) => Some(format!(
                    "{label}: Rust accepte, référence rejette {:?}",
                    case.error
                )),
                (Err(error), Some(_)) => {
                    return Err(format!("{label}: Rust rejette {error}, référence accepte").into());
                }
            };
            if let Some(detail) = mismatch {
                eprintln!("{detail}");
                failures.push(label);
            }
        }
    }
    Ok(failures)
}

fn is_role_rejection(error: &ServeError, expected: Option<&str>) -> bool {
    expected == Some(ROLE_ALTERNATION_ERROR)
        && matches!(error, ServeError::Http(message)
            if message == "Conversation roles must alternate user/assistant/user/assistant/...")
}

#[test]
fn external_serve_rejection_requires_role_category() {
    let message = "Conversation roles must alternate user/assistant/user/assistant/...";
    assert!(is_role_rejection(
        &ServeError::Http(message.into()),
        Some(ROLE_ALTERNATION_ERROR)
    ));
    assert!(!is_role_rejection(
        &ServeError::Args(message.into()),
        Some(ROLE_ALTERNATION_ERROR)
    ));
    assert!(!is_role_rejection(
        &ServeError::Http("tokenizer absent".into()),
        Some(ROLE_ALTERNATION_ERROR)
    ));
    assert!(!is_role_rejection(&ServeError::Http(message.into()), None));
}

const KNOWN_DIFFERENCES: &[&str] = &[
    "qwen38/tools/generation=True/thinking=False",
    "qwen36_moe/multi/generation=True/thinking=False",
    "qwen36_moe/tools/generation=True/thinking=False",
    "gemma3/tools/generation=True/thinking=None",
    "gemma4/tools/generation=True/thinking=False",
];

#[test]
#[ignore = "tokenizers locaux requis; 5/8 écarts serve connus; tools/goldens/README.md#suivi-des-ecarts"]
fn external_serve_templates_known_differences_are_stable() -> TestResult {
    assert_known_differences(&compare()?, KNOWN_DIFFERENCES);
    Ok(())
}

#[test]
#[ignore = "parité stricte, tokenizers locaux requis; 5/8 écarts serve connus; tools/goldens/README.md#suivi-des-ecarts"]
fn external_serve_templates_strict_parity_matches_transformers() -> TestResult {
    assert_known_differences(&compare()?, &[]);
    Ok(())
}
