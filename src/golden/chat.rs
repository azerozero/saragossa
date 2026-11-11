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

fn compare() -> TestResult<Vec<String>> {
    let mut differences = Vec::new();
    for (name, _) in CHAT_MODELS {
        let fixture: ChatFixture = json(&format!("chat_templates/{name}"))?;
        for case in fixture.cases {
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
            let label = format!("{name}/{}", case.name);
            // Toujours appeler le rendu de production, même si la référence rejette.
            // L'API actuelle est infallible : un rendu ne vaut jamais un rejet.
            let mismatch = match (&case.rendered, &case.error) {
                (Some(expected), None) => difference(&label, &actual, expected),
                (None, Some(error)) => {
                    assert_eq!(error, ROLE_ALTERNATION_ERROR);
                    Some(format!(
                        "{label}: Rust rend {actual:?}, référence rejette {error}"
                    ))
                }
                _ => return Err(format!("{label}: résultat de référence invalide").into()),
            };
            if let Some(detail) = mismatch {
                eprintln!("{detail}");
                differences.push(label);
            }
        }
    }
    Ok(differences)
}

// Suivi séparé des références externes : toute apparition OU disparition échoue.
const KNOWN_DIFFERENCES: &[&str] = &[
    "qwen38/multi/generation=False/thinking=False",
    "qwen38/multi/generation=False/thinking=True",
    "qwen38/multi/generation=True/thinking=False",
    "qwen38/multi/generation=True/thinking=True",
    "qwen38/tools/generation=False/thinking=False",
    "qwen38/tools/generation=False/thinking=True",
    "qwen38/tools/generation=True/thinking=False",
    "qwen38/tools/generation=True/thinking=True",
    "qwen36_moe/multi/generation=True/thinking=True",
    "qwen36_moe/tools/generation=False/thinking=False",
    "qwen36_moe/tools/generation=False/thinking=True",
    "qwen36_moe/tools/generation=True/thinking=False",
    "qwen36_moe/tools/generation=True/thinking=True",
    "gemma3/tools/generation=False/thinking=None",
    "gemma3/tools/generation=True/thinking=None",
    "gemma4/multi/generation=False/thinking=True",
    "gemma4/multi/generation=True/thinking=True",
    "gemma4/tools/generation=False/thinking=False",
    "gemma4/tools/generation=False/thinking=True",
    "gemma4/tools/generation=True/thinking=False",
    "gemma4/tools/generation=True/thinking=True",
];

#[test]
fn external_chat_known_differences_are_stable() -> TestResult {
    assert_known_differences(&compare()?, KNOWN_DIFFERENCES);
    Ok(())
}

#[test]
#[ignore = "parité stricte: 21/28 écarts connus; tools/goldens/README.md#suivi-des-ecarts"]
fn external_chat_strict_parity_matches_transformers() -> TestResult {
    assert_known_differences(&compare()?, &[]);
    Ok(())
}
