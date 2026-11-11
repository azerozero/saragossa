//! Références transformers 5.22.2 copiées de test/goldens-reference.
use crate::chat_template::*;
use serde::Deserialize;
use serde_json::value::RawValue;
use sha2::{Digest, Sha256};

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    messages: Vec<Message>,
    tools: Vec<Box<RawValue>>,
    add_generation_prompt: bool,
    enable_thinking: Option<bool>,
    rendered: Option<String>,
    error: Option<String>,
}

#[derive(Deserialize)]
struct Message {
    role: String,
    content: Option<String>,
    reasoning_content: Option<String>,
    name: Option<String>,
    tool_call_id: Option<String>,
    #[serde(default)]
    tool_calls: Vec<Box<RawValue>>,
}

#[test]
fn templates_match_all_28_transformers_references() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/goldens/chat_templates");
    let mut count = 0;
    for model in ["qwen36_moe", "qwen38", "gemma3", "gemma4"] {
        let bytes = std::fs::read(root.join(format!("{model}.bin"))).expect("fixture");
        let header: serde_json::Value = serde_json::from_slice(
            &std::fs::read(root.join(format!("{model}.json"))).expect("header"),
        )
        .expect("json");
        assert_eq!(
            format!("{:x}", Sha256::digest(&bytes)),
            header["payload_sha256"].as_str().expect("digest")
        );
        let fixture: Fixture = serde_json::from_slice(&bytes).expect("fixture JSON");
        for case in fixture.cases {
            let messages: Vec<_> = case
                .messages
                .into_iter()
                .map(|m| ChatTemplateMessage {
                    role: m.role,
                    content: m.content,
                    reasoning_content: m.reasoning_content,
                    tool_calls: m.tool_calls.iter().map(|v| v.get().to_string()).collect(),
                    name: m.name,
                    tool_call_id: m.tool_call_id,
                })
                .collect();
            let tools: Vec<_> = case.tools.iter().map(|v| v.get().to_string()).collect();
            let thinking = case.enable_thinking.unwrap_or(false);
            let actual = match model {
                "gemma3" => render_gemma_chat(&messages, case.add_generation_prompt)
                    .map(|s| format!("<bos>{s}")),
                "gemma4" => Ok(render_gemma4_chat_with_tools(
                    &messages,
                    &tools,
                    case.add_generation_prompt,
                    thinking,
                )),
                _ => Ok(render_qwen_chatml_profile(
                    &messages,
                    &tools,
                    case.add_generation_prompt,
                    thinking,
                    if model == "qwen38" {
                        QwenProfile::Qwen38
                    } else {
                        QwenProfile::Qwen36
                    },
                )),
            };
            match (actual, case.rendered, case.error) {
                (Ok(actual), Some(expected), None) => {
                    assert_eq!(actual, expected, "{model}/{}", case.name)
                }
                (Err(error), None, Some(expected)) => assert_eq!(
                    format!("TemplateError: {error}"),
                    expected,
                    "{model}/{}",
                    case.name
                ),
                other => panic!("{model}/{}: résultat inattendu {other:?}", case.name),
            }
            count += 1;
        }
    }
    assert_eq!(count, 28);
}
