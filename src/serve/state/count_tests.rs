//! Prétraitement CPU avec tokenizer synthétique et aucun fichier de poids valide.
use super::*;
use serde_json::{json, Value};

fn fixture(family: &str) -> (tempfile::TempDir, ServeState) {
    let dir = tempfile::tempdir().expect("invariant: dossier temporaire");
    std::fs::write(dir.path().join("config.json"), json!({
        "model_type":family,"hidden_size":2,"num_hidden_layers":1,"num_attention_heads":1,
        "num_key_value_heads":1,"rms_norm_eps":1e-6,"rope_theta":10000.0,"vocab_size":3,"eos_token_id":2
    }).to_string()).expect("invariant: config synthétique");
    let tokenizer: tokenizers::Tokenizer = serde_json::from_value(json!({
        "version":"1.0","truncation":null,"padding":null,"added_tokens":[],
        "normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,
        "model":{"type":"WordLevel","vocab":{"<unk>":0,"Bonjour":1,"Paris":2},"unk_token":"<unk>"}
    })).expect("invariant: tokenizer synthétique");
    tokenizer
        .save(dir.path().join("tokenizer.json"), false)
        .expect("invariant: sauvegarde");
    // Un chargement ModelAssets ou décodeur échouerait sur ce fichier.
    std::fs::write(dir.path().join("model.safetensors"), b"poison").expect("invariant: sentinelle");
    let args = ServeArgs::parse([
        "--model".to_owned(),
        format!("fixture={}", dir.path().display()),
    ])
    .expect("invariant: args");
    (dir, ServeState::new(&args))
}

#[test]
fn counts_exact_rendered_messages_without_loading_decoder_or_weights() {
    for family in ["qwen3", "gemma3", "gemma4", "flint"] {
        let (dir, state) = fixture(family);
        let body = json!({"model":"fixture","system":[{"type":"text","text":"Consigne"}],
            "messages":[{"role":"user","content":"Bonjour 🌍"},{"role":"assistant","content":"Paris"},
                {"role":"user","content":"Suite"}],
            "tools":[{"name":"weather","description":"Météo","input_schema":{"type":"object"}}]});
        let bytes = serde_json::to_vec(&body).expect("invariant: JSON");
        let mut wire = Vec::new();
        crate::serve::anthropic::handle_count_tokens(&mut wire, &state, &bytes)
            .expect("invariant: comptage CPU");
        let wire = String::from_utf8(wire).expect("invariant: UTF-8");
        let result: Value =
            serde_json::from_str(wire.split_once("\r\n\r\n").expect("invariant: HTTP").1)
                .expect("invariant: JSON");
        let mut generation = body.clone();
        generation["max_tokens"] = json!(4);
        let (chat, _, _) = crate::serve::anthropic::anthropic_chat(
            &serde_json::to_vec(&generation).expect("invariant: JSON"),
            None,
        )
        .expect("invariant: même conversion Messages");
        let config = saragossa::ModelConfig::from_file(dir.path().join("config.json"))
            .expect("invariant: config");
        let tokenizer = saragossa::RustTokenizer::from_file(dir.path().join("tokenizer.json"))
            .expect("invariant: tokenizer");
        let tokens = encode_chat_prompt(&config, &tokenizer, &chat)
            .expect("invariant: prétraitement génération");
        assert_eq!(result, json!({"input_tokens":tokens.len()}));
        assert!(tokens.len() > 10);
        assert_eq!(state.loaded_count(), 0);
        assert_eq!(state.clock, 0);
        let mut no_tools = body;
        no_tools["tools"] = json!([]);
        let mut small_wire = Vec::new();
        crate::serve::anthropic::handle_count_tokens(
            &mut small_wire,
            &state,
            &serde_json::to_vec(&no_tools).expect("invariant: JSON"),
        )
        .expect("invariant: comptage sans outils");
        let small_wire = String::from_utf8(small_wire).expect("invariant: UTF-8");
        let small: Value = serde_json::from_str(
            small_wire
                .split_once("\r\n\r\n")
                .expect("invariant: HTTP")
                .1,
        )
        .expect("invariant: JSON");
        assert!(small["input_tokens"].as_u64() < result["input_tokens"].as_u64());
    }
}

#[test]
fn count_matches_actual_tiny_cpu_completion_usage() {
    let (_dir, mut state) = fixture("qwen3");
    state.models[0].loaded = Some(super::json_tests::tiny_loaded_model());
    let body = json!({"model":"fixture", "max_tokens":2, "system":"Consigne",
        "messages":[{"role":"user","content":"Bonjour"}],
        "tools":[{"name":"weather","input_schema":{"type":"object"}}]});
    let (chat, _, _) = crate::serve::anthropic::anthropic_chat(
        &serde_json::to_vec(&body).expect("invariant: JSON"),
        None,
    )
    .expect("invariant: conversion");
    let count = state.count_tokens(&chat).expect("invariant: comptage");
    let completion = state
        .complete(chat, None)
        .expect("invariant: génération CPU synthétique");
    assert_eq!(count, completion.usage.prompt_tokens());
}

#[test]
fn responses_refuses_tools_on_other_families_before_loading_weights() {
    let (_dir, mut state) = fixture("qwen3");
    let body = json!({"model":"fixture", "input":"Bonjour",
        "tools":[{"type":"function","name":"weather","parameters":{"type":"object"}}]});
    let error = crate::serve::responses::handle(
        &mut Vec::new(),
        &mut state,
        &serde_json::to_vec(&body).expect("invariant: JSON"),
    )
    .expect_err("invariant: outils non supportés");
    assert!(matches!(error, ServeError::NotImplemented(_)));
    assert_eq!(state.loaded_count(), 0);
}
