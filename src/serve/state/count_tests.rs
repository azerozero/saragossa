//! Prétraitement CPU avec tokenizer synthétique et aucun fichier de poids valide.
use super::*;
use serde_json::{json, Value};

fn fixture(family: &str) -> (tempfile::TempDir, ServeState) {
    let dir = tempfile::tempdir().expect("invariant: dossier temporaire");
    std::fs::write(dir.path().join("config.json"), json!({
        "model_type":family,"hidden_size":2,"num_hidden_layers":1,"num_attention_heads":1,
        "num_key_value_heads":1,"rms_norm_eps":1e-6,"rope_theta":10000.0,"vocab_size":3,"eos_token_id":2
    }).to_string()).expect("invariant: config synthétique");
    let mut tokenizer: tokenizers::Tokenizer = serde_json::from_value(json!({
        "version":"1.0","truncation":null,"padding":null,"added_tokens":[],
        "normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,
        "model":{"type":"WordLevel","vocab":{"<unk>":0,"Bonjour":1,"Paris":2},"unk_token":"<unk>"}
    })).expect("invariant: tokenizer synthétique");
    // Le BOS synthétique rend observable le chemin spécial Gemma 3.
    tokenizer.with_post_processor(Some(
        tokenizers::processors::template::TemplateProcessing::builder()
            .try_single("<unk> $A")
            .expect("invariant: template BOS")
            .special_tokens(vec![("<unk>", 0)])
            .build()
            .expect("invariant: post-processeur"),
    ));
    tokenizer
        .save(dir.path().join("tokenizer.json"), false)
        .expect("invariant: sauvegarde");
    std::fs::write(
        dir.path().join("chat_template.jinja"),
        if family == "qwen3_8" {
            "<|im_start|> Reasoning effort is set to"
        } else {
            "<|im_start|> enable_thinking last_query_index"
        },
    )
    .expect("invariant: profil synthétique");
    // Un chargement ModelAssets ou décodeur échouerait sur ce fichier.
    std::fs::write(dir.path().join("model.safetensors"), b"poison").expect("invariant: sentinelle");
    let args = ServeArgs::parse([
        "--model".to_owned(),
        format!("fixture={}", dir.path().display()),
    ])
    .expect("invariant: args");
    (dir, ServeState::new(&args))
}

fn post(state: &mut ServeState, route: &str, body: &Value) -> (String, Value) {
    let body = body.to_string();
    let wire = format!(
        "POST {route} HTTP/1.1\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    let request = crate::serve::http::read_request(
        &mut std::io::Cursor::new(wire),
        Instant::now() + Duration::from_secs(5),
    )
    .expect("invariant: lecture HTTP")
    .expect("invariant: requête HTTP");
    let mut response = Vec::new();
    if let Err(error) = crate::serve::http::handle_job(&mut response, state, &request) {
        assert!(matches!(error, ServeError::Http(_)), "{error}");
    }
    let wire = String::from_utf8(response).expect("invariant: UTF-8");
    let (headers, body) = wire.split_once("\r\n\r\n").expect("invariant: HTTP");
    (
        headers.to_owned(),
        serde_json::from_str(body).expect("invariant: JSON"),
    )
}

#[test]
fn count_equals_messages_and_chat_usage_for_every_family_with_and_without_tools() {
    for family in ["qwen3_6", "qwen3_8", "gemma3", "gemma4", "flint"] {
        for tools in [false, true] {
            let (dir, mut state) = fixture(family);
            let mut body = json!({"model":"fixture","system":"Consigne", "messages":[
                {"role":"user","content":"Bonjour 🌍"},
                {"role":"assistant","content":"Paris"},
                {"role":"user","content":"Suite"}]});
            let mut chat_body = json!({"model":"fixture","max_tokens":1,"messages":[
                {"role":"system","content":"Consigne"},
                {"role":"user","content":"Bonjour 🌍"},
                {"role":"assistant","content":"Paris"},
                {"role":"user","content":"Suite"}]});
            if tools {
                body["tools"] = json!([{"name":"weather","description":"Météo", "input_schema":{"type":"object"}}]);
                chat_body["tools"] = json!([{"type":"function","function":{"name":"weather","description":"Météo","parameters":{"type":"object"}}}]);
            }
            let (count_status, count) = post(&mut state, "/v1/messages/count_tokens", &body);
            assert_eq!(state.loaded_count(), 0);
            assert_eq!(state.clock, 0);
            let mut loaded = super::json_tests::tiny_loaded_model();
            loaded.assets.config =
                saragossa::ModelConfig::from_file(dir.path().join("config.json"))
                    .expect("invariant: config synthétique");
            loaded.assets.tokenizer =
                saragossa::RustTokenizer::from_file(dir.path().join("tokenizer.json"))
                    .expect("invariant: tokenizer synthétique");
            let request: ChatCompletionRequest =
                serde_json::from_value(chat_body.clone()).expect("invariant: requête Chat");
            if family != "gemma3" || !tools {
                assert!(
                    count_status.starts_with("HTTP/1.1 200"),
                    "{family}: {count}"
                );
                // Oracle indépendant du helper partagé, avec profil attendu et BOS.
                let messages = request.template_messages();
                let declarations: Vec<String> = request
                    .active_tools()
                    .expect("invariant: outils")
                    .iter()
                    .map(|tool| tool.json.clone())
                    .collect();
                let rendered = match family {
                    "flint" => saragossa::render_flint_chat(&messages, &declarations, true),
                    "gemma4" => saragossa::chat_template::render_gemma4_chat_with_tools(
                        &messages,
                        &declarations,
                        true,
                        false,
                    ),
                    "gemma3" => {
                        saragossa::render_gemma_chat(&messages, true).expect("invariant: Gemma 3")
                    }
                    _ => saragossa::chat_template::render_qwen_chatml_profile(
                        &messages,
                        &declarations,
                        true,
                        false,
                        if family == "qwen3_8" {
                            saragossa::chat_template::QwenProfile::Qwen38
                        } else {
                            saragossa::chat_template::QwenProfile::Qwen36
                        },
                    ),
                };
                let tokenizer = &loaded.assets.tokenizer;
                let expected = if family == "gemma3" {
                    let ids = tokenizer
                        .encode_with_special_tokens(&rendered)
                        .expect("invariant: BOS");
                    assert_eq!(
                        ids.len(),
                        tokenizer
                            .encode(&rendered)
                            .expect("invariant: sans BOS")
                            .len()
                            + 1
                    );
                    ids
                } else {
                    tokenizer.encode(&rendered).expect("invariant: encodage")
                };
                assert_eq!(
                    loaded
                        .prompt_encoding(&request)
                        .expect("invariant: prompt")
                        .tokens,
                    tokens_to_usize(&expected).expect("invariant: ids"),
                    "{family}, tools={tools}"
                );
                assert_eq!(
                    count["input_tokens"],
                    expected.len(),
                    "{family}, tools={tools}"
                );
            } else {
                assert!(count_status.starts_with("HTTP/1.1 400"), "{count}");
            }
            state.models[0].loaded = Some(loaded);
            body["max_tokens"] = json!(1);
            let (messages_status, messages) = post(&mut state, "/v1/messages", &body);
            let (chat_status, chat) = post(&mut state, "/v1/chat/completions", &chat_body);
            if family == "gemma3" && tools {
                assert!(messages_status.starts_with("HTTP/1.1 400"), "{messages}");
                assert!(chat_status.starts_with("HTTP/1.1 400"), "{chat}");
            } else {
                assert!(
                    messages_status.starts_with("HTTP/1.1 200"),
                    "{family}: {messages}"
                );
                assert!(chat_status.starts_with("HTTP/1.1 200"), "{family}: {chat}");
                assert_eq!(
                    count["input_tokens"], messages["usage"]["input_tokens"],
                    "{family}, tools={tools}"
                );
                assert_eq!(
                    count["input_tokens"], chat["usage"]["prompt_tokens"],
                    "{family}, tools={tools}"
                );
                assert_eq!(
                    count["input_tokens"],
                    state
                        .count_tokens(&request)
                        .expect("invariant: comptage à chaud")
                );
            }
        }
    }
}

#[test]
fn responses_uses_chat_tool_policy_before_loading_weights() {
    for family in ["qwen3_5", "qwen3_6", "qwen3_8", "gemma3", "gemma4", "flint"] {
        let (dir, mut state) = fixture(family);
        let config = saragossa::ModelConfig::from_file(dir.path().join("config.json"))
            .expect("invariant: config");
        for extra in [
            json!({"tools":[{"type":"function","function":{"name":"weather","parameters":{"type":"object"}}}]}),
            json!({"tools":[{"type":"function","function":{"name":"weather"}}],"tool_choice":"none"}),
            json!({"messages":[{"role":"assistant","tool_calls":[{"id":"a","type":"function","function":{"name":"weather","arguments":"{}"}}]},
                {"role":"tool","tool_call_id":"a","content":"soleil"}]}),
        ] {
            let mut body =
                json!({"model":"fixture","messages":[{"role":"user","content":"Bonjour"}]});
            body.as_object_mut()
                .expect("invariant: objet")
                .extend(extra.as_object().expect("invariant: objet").clone());
            let chat: ChatCompletionRequest =
                serde_json::from_value(body).expect("invariant: Chat");
            let expected = output_parser_for_config(&config, &chat);
            let actual = state.response_output_parser(&chat);
            match (expected, actual) {
                (Ok(expected), Ok(actual)) => {
                    assert_eq!(expected.is_some(), actual.is_some(), "{family}")
                }
                (Err(expected), Err(actual)) => {
                    assert_eq!(expected.to_string(), actual.to_string())
                }
                _ => panic!("invariant: politique outils divergente pour {family}"),
            }
        }
        if family == "gemma3" {
            let (status, _) = post(
                &mut state,
                "/v1/responses",
                &json!({"model":"fixture","input":"Bonjour",
                "tools":[{"type":"function","name":"weather","parameters":{"type":"object"}}]}),
            );
            assert!(status.starts_with("HTTP/1.1 400"));
        }
        assert_eq!(state.loaded_count(), 0);
    }
}
