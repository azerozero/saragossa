use super::external::*;
use crate::chat_template::*;
use serde::Deserialize;
use serde_json::value::RawValue;

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
fn external_chat_fixtures_parse() -> TestResult {
    for name in CHAT_MODELS
        .iter()
        .map(|(name, _)| *name)
        .chain(["qwen3_tts"])
    {
        let fixture: ChatFixture = json(&format!("chat_templates/{name}"))?;
        assert_eq!(fixture.model, name);
        if fixture.unavailable.is_some() {
            assert_eq!(name, "qwen3_tts");
            assert!(fixture.cases.is_empty());
            continue;
        }
        assert!(fixture.cases.len() >= 4);
        for case in fixture.cases {
            let _: Vec<Message> = serde_json::from_str(case.messages.get())?;
            let _: Vec<Box<RawValue>> = serde_json::from_str(case.tools.get())?;
            assert!(case.rendered.is_some() ^ case.error.is_some());
            if case.rendered.is_some() {
                assert!(!case.token_ids.ok_or("tokens absents")?.is_empty());
            }
        }
    }
    Ok(())
}

fn compare() -> TestResult<Vec<String>> {
    let mut differences = Vec::new();
    let mut compared = 0;
    for (name, _) in CHAT_MODELS {
        let fixture: ChatFixture = json(&format!("chat_templates/{name}"))?;
        for case in fixture.cases {
            let wire_messages: Vec<Message> = serde_json::from_str(case.messages.get())?;
            let wire_tools: Vec<Box<RawValue>> = serde_json::from_str(case.tools.get())?;
            let messages: Vec<_> = wire_messages
                .into_iter()
                .map(|m| ChatTemplateMessage {
                    assistant_blocks: None,
                    role: m.role,
                    content: m.content,
                    reasoning_content: m.reasoning_content,
                    tool_calls: m.tool_calls.iter().map(|v| v.get().to_string()).collect(),
                    name: m.name,
                    tool_call_id: m.tool_call_id,
                })
                .collect();
            let tools: Vec<_> = wire_tools.iter().map(|v| v.get().to_string()).collect();
            let thinking = case.enable_thinking.unwrap_or(false);
            let actual = match name {
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
                    if name == "qwen38" {
                        QwenProfile::Qwen38
                    } else {
                        QwenProfile::Qwen36
                    },
                )),
            };
            let label = format!("{name}/{}", case.name);
            let mismatch = match (actual, &case.rendered, &case.error) {
                (Ok(actual), Some(expected), None) => difference(&label, &actual, expected),
                (Err(error), None, Some(expected)) => {
                    difference(&label, &format!("TemplateError: {error}"), expected)
                }
                (actual, _, _) => Some(format!("{label}: résultat inattendu {actual:?}")),
            };
            compared += 1;
            if let Some(detail) = mismatch {
                eprintln!("{detail}");
                differences.push(label);
            }
        }
    }
    assert_eq!(compared, 28);
    Ok(differences)
}

// Suivi séparé des références externes : toute apparition OU disparition échoue.
// Les corrections de main résolvent les 21 anciens écarts, références inchangées.
const KNOWN_DIFFERENCES: &[&str] = &[];

#[test]
fn external_chat_known_differences_are_stable() -> TestResult {
    assert_known_differences(&compare()?, KNOWN_DIFFERENCES);
    Ok(())
}

#[test]
fn external_chat_strict_parity_matches_transformers() -> TestResult {
    assert_known_differences(&compare()?, &[]);
    Ok(())
}
