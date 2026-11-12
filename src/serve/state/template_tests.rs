use super::*;
use saragossa::chat_template::QwenProfile;
use serde::Deserialize;
use serde_json::value::RawValue;

fn config(model_type: &str) -> saragossa::ModelConfig {
    serde_json::from_value(serde_json::json!({"model_type":model_type,"hidden_size":1,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,"rms_norm_eps":0.000001,"rope_theta":10000.0,"vocab_size":1})).expect("synthetic config")
}

#[test]
fn serve_renders_all_eight_reference_cases_without_weights() {
    #[derive(Deserialize)]
    struct Fixture {
        cases: Vec<Case>,
    }
    #[derive(Deserialize)]
    struct Case {
        name: String,
        messages: Box<RawValue>,
        tools: Box<RawValue>,
        add_generation_prompt: bool,
        enable_thinking: Option<bool>,
        rendered: Option<String>,
        error: Option<String>,
    }
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/goldens/chat_templates");
    let mut count = 0;
    for (model, model_type, profile) in [
        ("qwen36_moe", "qwen3_5_moe", QwenProfile::Qwen36),
        ("qwen38", "qwen3_5", QwenProfile::Qwen38),
        ("gemma3", "gemma3", QwenProfile::Qwen36),
        ("gemma4", "gemma4", QwenProfile::Qwen36),
    ] {
        let fixture: Fixture = serde_json::from_slice(
            &std::fs::read(root.join(format!("{model}.bin"))).expect("fixture"),
        )
        .expect("json");
        for case in fixture
            .cases
            .into_iter()
            .filter(|c| c.add_generation_prompt && c.enable_thinking != Some(true))
        {
            let thinking = case
                .enable_thinking
                .map(|b| format!(r#", "chat_template_kwargs":{{"enable_thinking":{b}}}"#))
                .unwrap_or_default();
            let wire = format!(
                r#"{{"model":"test","messages":{},"tools":{}{thinking}}}"#,
                case.messages.get(),
                case.tools.get()
            );
            let request = ChatCompletionRequest::parse(wire.as_bytes()).expect("request");
            let actual = render_chat_prompt(&config(model_type), profile, &request);
            match (actual, case.rendered, case.error) {
                (Ok(text), Some(expected), None) => assert_eq!(
                    if model == "gemma3" {
                        format!("<bos>{text}")
                    } else {
                        text
                    },
                    expected,
                    "{model}/{}",
                    case.name
                ),
                (Err(ServeError::Http(error)), None, Some(expected)) => {
                    assert_eq!(format!("TemplateError: {error}"), expected)
                }
                other => panic!("{model}/{}: {other:?}", case.name),
            }
            count += 1;
        }
    }
    assert_eq!(count, 8);
}

#[test]
fn unsupported_tools_fail_and_none_removes_declarations() {
    let wire = br#"{"model":"test","messages":[{"role":"user","content":"Bonjour"}],"tools":[{"type":"function","function":{"name":"weather","parameters":{"type":"object"}}}]}"#;
    let mut request = ChatCompletionRequest::parse(wire).expect("request");
    for family in ["gemma3", "llama", "qwen3", "qwen3_next"] {
        let error = output_parser_for_config(&config(family), &request)
            .err()
            .expect("unsupported");
        assert!(error.to_string().contains("outils non pris en charge"));
    }
    request.tool_choice = Some(serde_json::json!("none"));
    assert!(output_parser_for_config(&config("gemma3"), &request)
        .expect("none")
        .is_none());
    request.chat_template_kwargs =
        Some(serde_json::from_value(serde_json::json!({"enable_thinking":true})).expect("kwargs"));
    assert!(request.validate(1024).is_err());
}

#[test]
fn no_active_tools_never_selects_a_framed_parser() {
    for family in ["qwen3_5", "qwen3_6", "qwen3_8", "gemma4", "gemma3", "llama"] {
        for (tools, choice) in [
            (serde_json::json!([]), "auto"),
            (
                serde_json::json!([{"type":"function","function":{"name":"weather","parameters":{"type":"object"}}}]),
                "none",
            ),
        ] {
            let request = serde_json::json!({"model":"test","messages":[{"role":"user","content":"test"}],"tools":tools,"tool_choice":choice});
            let request =
                ChatCompletionRequest::parse(&serde_json::to_vec(&request).expect("request"))
                    .expect("parse");
            assert!(
                output_parser_for_config(&config(family), &request)
                    .expect("inactive tools")
                    .is_none(),
                "{family}/{choice}"
            );
        }
    }
}
