//! Rendu des outils Qwen au format du tokenizer.
use super::{ChatTemplateMessage, JsonObject, JsonValue, QWEN_EMPTY_THINK_BLOCK};

/// Rend ChatML avec les déclarations et l'historique structuré des outils.
pub fn render_qwen_chatml_with_tools(
    messages: &[ChatTemplateMessage],
    tools: &[String],
    add_generation_prompt: bool,
    enable_thinking: bool,
) -> String {
    let mut out = String::new();
    let mut turns = messages;
    if !tools.is_empty() {
        out.push_str("<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>\n");
        for tool in tools {
            match serde_json::from_str::<JsonValue>(tool) {
                Ok(value) => out.push_str(&value.to_python_json()),
                Err(_) => out.push_str(tool),
            }
            out.push('\n');
        }
        out.push_str(include_str!("qwen_tool_instructions.txt").trim_end());
        if let Some((first, rest)) = messages.split_first().filter(|(m, _)| m.role == "system") {
            out.push_str("\n\n");
            out.push_str(first.content.as_deref().unwrap_or(""));
            turns = rest;
        }
        out.push_str("<|im_end|>\n");
    }
    for (index, message) in turns.iter().enumerate() {
        let content = message.content.as_deref().unwrap_or("");
        if message.role == "tool" {
            if index == 0 || turns[index - 1].role != "tool" {
                out.push_str("<|im_start|>user");
            }
            out.push_str("\n<tool_response>\n");
            out.push_str(content);
            out.push_str("\n</tool_response>");
            if turns.get(index + 1).is_none_or(|m| m.role != "tool") {
                out.push_str("<|im_end|>\n");
            }
            continue;
        }
        out.push_str("<|im_start|>");
        out.push_str(&message.role);
        out.push('\n');
        out.push_str(content);
        if message.role == "assistant" {
            for (index, call) in message.tool_calls.iter().enumerate() {
                if index > 0 || !content.is_empty() {
                    out.push('\n');
                }
                push_call(&mut out, call);
            }
        }
        out.push_str("<|im_end|>\n");
    }
    if add_generation_prompt {
        out.push_str("<|im_start|>assistant\n");
        if !enable_thinking {
            out.push_str(QWEN_EMPTY_THINK_BLOCK);
        }
    }
    out
}

fn push_call(out: &mut String, raw: &str) {
    let Some(call) = JsonObject::parse(raw) else {
        return;
    };
    let function = match call.get("function") {
        Some(JsonValue::Object(function)) => function,
        _ => &call,
    };
    let name = function
        .get("name")
        .and_then(JsonValue::as_str)
        .unwrap_or("");
    out.push_str("<tool_call>\n<function=");
    out.push_str(name);
    out.push_str(">\n");
    let parsed;
    let arguments = match function.get("arguments") {
        Some(JsonValue::Object(object)) => Some(object),
        Some(JsonValue::String(text)) => {
            parsed = JsonObject::parse(text);
            parsed.as_ref()
        }
        _ => None,
    };
    for (key, value) in arguments.iter().flat_map(|object| object.iter()) {
        out.push_str("<parameter=");
        out.push_str(key);
        out.push_str(">\n");
        match value.as_str() {
            Some(text) => out.push_str(text),
            None => out.push_str(&value.to_python_json()),
        }
        out.push_str("\n</parameter>\n");
    }
    out.push_str("</function>\n</tool_call>");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groups_results_and_preserves_whitespace() {
        let messages = [
            ChatTemplateMessage::new("tool", " a "),
            ChatTemplateMessage::new("tool", "b"),
        ];
        assert_eq!(render_qwen_chatml_with_tools(&messages, &[], false, false),
            "<|im_start|>user\n<tool_response>\n a \n</tool_response>\n<tool_response>\nb\n</tool_response><|im_end|>\n");
    }

    #[test]
    fn qwen36_tools_match_reference_without_thinking() {
        // RawValue conserve l'ordre des clés de la référence.
        #[derive(serde::Deserialize)]
        struct Case {
            messages: Vec<Box<serde_json::value::RawValue>>,
            tools: Vec<Box<serde_json::value::RawValue>>,
            rendered: String,
            enable_thinking: bool,
            add_generation_prompt: bool,
        }
        #[derive(serde::Deserialize)]
        struct Fixture {
            cases: Vec<Case>,
        }
        let fixture: Fixture = serde_json::from_slice(include_bytes!(
            "../../tests/fixtures/goldens/chat_templates/qwen36_moe.bin"
        ))
        .expect("fixture");
        for case in fixture.cases.into_iter().filter(|c| !c.enable_thinking) {
            let messages: Vec<_> = case
                .messages
                .iter()
                .map(|raw| {
                    #[derive(serde::Deserialize)]
                    struct Message {
                        role: String,
                        content: String,
                        #[serde(default)]
                        tool_calls: Vec<Box<serde_json::value::RawValue>>,
                    }
                    let m: Message = serde_json::from_str(raw.get()).expect("message");
                    let mut message = ChatTemplateMessage::new(m.role, m.content);
                    message.tool_calls = m.tool_calls.iter().map(|v| v.get().to_string()).collect();
                    message
                })
                .collect();
            let tools: Vec<_> = case.tools.iter().map(|v| v.get().to_string()).collect();
            assert_eq!(
                render_qwen_chatml_with_tools(&messages, &tools, case.add_generation_prompt, false),
                case.rendered
            );
        }
    }
}
