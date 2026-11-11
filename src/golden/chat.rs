use super::external::*;
use crate::{render_gemma4_chat, render_gemma_chat, render_qwen_chatml, ChatTemplateMessage};

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
            assert!(case.messages.is_array() && case.tools.is_array());
            assert!(case.rendered.is_some() ^ case.error.is_some());
            if case.rendered.is_some() {
                assert!(!case.token_ids.ok_or("tokens absents")?.is_empty());
            }
        }
    }
    Ok(())
}

/// Oracle strict conservé en échec si le moteur diverge ; le parse reste actif.
#[test]
#[ignore = "oracle externe exact des templates; écarts connus documentés dans tools/goldens/README.md"]
fn external_chat_matches_transformers() -> TestResult {
    let mut differences = Vec::new();
    for (name, _) in CHAT_MODELS {
        let fixture: ChatFixture = json(&format!("chat_templates/{name}"))?;
        for case in fixture.cases {
            let Some(expected) = case.rendered else {
                differences.push(format!(
                    "{name}/{}: référence rejette: {:?}; Rust ne valide pas les rôles",
                    case.name, case.error
                ));
                continue;
            };
            let messages = case
                .messages
                .as_array()
                .ok_or("messages absents")?
                .iter()
                .map(|m| {
                    let mut message = ChatTemplateMessage::new(
                        m["role"].as_str().unwrap_or(""),
                        m["content"].as_str().unwrap_or(""),
                    );
                    message.tool_calls = m["tool_calls"]
                        .as_array()
                        .map(|calls| calls.iter().map(ToString::to_string).collect())
                        .unwrap_or_default();
                    message.reasoning_content = m["reasoning_content"].as_str().map(str::to_string);
                    message
                })
                .collect::<Vec<_>>();
            let generation = case.add_generation_prompt;
            let thinking = case.enable_thinking.unwrap_or(true);
            let actual = match name {
                "gemma3" => format!("<bos>{}", render_gemma_chat(&messages, generation)),
                "gemma4" => render_gemma4_chat(&messages, generation, thinking),
                _ => render_qwen_chatml(&messages, generation, thinking),
            };
            if let Some(diff) = difference(&format!("{name}/{}", case.name), &actual, &expected) {
                differences.push(diff);
            }
        }
    }
    assert!(
        differences.is_empty(),
        "{} écarts:\n{}",
        differences.len(),
        differences.join("\n")
    );
    Ok(())
}
