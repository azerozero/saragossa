//! Déclarations, appels et résultats dans les tours Gemma 4.
use super::ChatTemplateMessage;
use serde_json::Value;
use std::collections::HashMap;

/// Rend le template Gemma 4 avec les outils structurés du client.
pub fn render_gemma4_chat_with_tools(
    messages: &[ChatTemplateMessage],
    tools: &[String],
    add_generation_prompt: bool,
    enable_thinking: bool,
) -> String {
    let mut out = String::from("<bos>");
    let mut turns = messages;
    let system = messages
        .first()
        .filter(|m| matches!(m.role.as_str(), "system" | "developer"));
    if system.is_some() || !tools.is_empty() {
        out.push_str("<|turn>system\n");
        if let Some(system) = system {
            out.push_str(system.content.as_deref().unwrap_or("").trim());
            turns = &messages[1..];
        }
        for raw in tools {
            if let Ok(tool) = serde_json::from_str::<Value>(raw) {
                let function = tool.get("function").unwrap_or(&tool);
                let name = function["name"].as_str().unwrap_or("");
                out.push_str("<|tool>declaration:");
                out.push_str(name);
                out.push_str(&declaration(function));
                out.push_str("<tool|>");
            }
        }
        out.push_str("<turn|>\n");
    }
    let last_user = turns.iter().rposition(|m| m.role == "user");
    let mut previous_role = "";
    let mut pending_tool = false;
    for (index, message) in turns.iter().enumerate() {
        if message.role == "tool" {
            continue;
        }
        pending_tool = false;
        let assistant = message.role == "assistant";
        if !(assistant && previous_role == "assistant") {
            out.push_str("<|turn>");
            out.push_str(if assistant { "model" } else { &message.role });
            out.push('\n');
        }
        previous_role = &message.role;
        if assistant && !message.tool_calls.is_empty() && last_user.is_none_or(|last| index > last)
        {
            if let Some(reasoning) = message
                .reasoning_content
                .as_deref()
                .filter(|s| !s.is_empty())
            {
                out.push_str("<|channel>thought\n");
                out.push_str(reasoning);
                out.push_str("\n<channel|>");
            }
        }
        let mut names = HashMap::new();
        for raw in &message.tool_calls {
            if let Ok(call) = serde_json::from_str::<Value>(raw) {
                let function = call.get("function").unwrap_or(&call);
                let name = function["name"].as_str().unwrap_or("");
                if let Some(id) = call["id"].as_str() {
                    names.insert(id.to_string(), name.to_string());
                }
                let arguments = match &function["arguments"] {
                    Value::String(text) => serde_json::from_str(text).unwrap_or(Value::Null),
                    value => value.clone(),
                };
                out.push_str("<|tool_call>call:");
                out.push_str(name);
                out.push_str(&format_value(&arguments, false));
                out.push_str("<tool_call|>");
            }
        }
        let mut responses = false;
        if !message.tool_calls.is_empty() {
            for result in turns[index + 1..].iter().take_while(|m| m.role == "tool") {
                let name = result
                    .tool_call_id
                    .as_ref()
                    .and_then(|id| names.get(id).map(String::as_str))
                    .or(result.name.as_deref())
                    .unwrap_or("unknown");
                out.push_str("<|tool_response>response:");
                out.push_str(name);
                out.push_str("{value:");
                out.push_str(&format_value(
                    &Value::String(result.content.clone().unwrap_or_default()),
                    false,
                ));
                out.push_str("}<tool_response|>");
                responses = true;
            }
        }
        let content = message.content.as_deref().unwrap_or("");
        let content = if assistant {
            strip_thinking(content)
        } else {
            content.trim().to_string()
        };
        out.push_str(&content);
        if !message.tool_calls.is_empty() && !responses {
            out.push_str("<|tool_response>");
            pending_tool = true;
        } else if responses {
            pending_tool = true;
            if !content.is_empty() {
                out.push_str("<turn|>\n");
            }
        } else {
            out.push_str("<turn|>\n");
        }
    }
    if add_generation_prompt && !pending_tool {
        out.push_str("<|turn>model\n");
        if !enable_thinking {
            out.push_str("<|channel>thought\n<channel|>");
        }
    }
    out
}

fn strip_thinking(text: &str) -> String {
    text.split("<channel|>")
        .map(|part| part.split("<|channel>").next().unwrap_or(""))
        .collect::<String>()
        .trim()
        .to_string()
}

fn format_value(value: &Value, escape_keys: bool) -> String {
    match value {
        Value::String(text) => format!("<|\"|>{text}<|\"|>"),
        Value::Array(items) => format!(
            "[{}]",
            items
                .iter()
                .map(|v| format_value(v, escape_keys))
                .collect::<Vec<_>>()
                .join(",")
        ),
        Value::Object(object) => format!(
            "{{{}}}",
            object
                .iter()
                .map(|(key, value)| {
                    let key = if escape_keys {
                        format!("<|\"|>{key}<|\"|>")
                    } else {
                        key.clone()
                    };
                    format!("{key}:{}", format_value(value, escape_keys))
                })
                .collect::<Vec<_>>()
                .join(",")
        ),
        Value::Null => "None".into(),
        _ => value.to_string(),
    }
}

fn nonempty(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(v) => *v,
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
        Value::Number(n) => n.as_f64() != Some(0.0),
    }
}

fn typed(value: &Value) -> Value {
    match value {
        Value::String(s) => Value::String(s.to_uppercase()),
        Value::Array(a) => Value::Array(a.iter().map(typed).collect()),
        _ => Value::String(String::new()),
    }
}

fn properties(value: &Value) -> String {
    let Some(object) = value.as_object() else {
        return String::new();
    };
    object
        .iter()
        .map(|(key, value)| format!("{key}:{}", parameter(value)))
        .collect::<Vec<_>>()
        .join(",")
}

fn parameter(value: &Value) -> String {
    let mut fields = Vec::new();
    if nonempty(&value["description"]) {
        fields.push(format!(
            "description:{}",
            format_value(&value["description"], true)
        ));
    }
    let kind = value["type"].as_str().unwrap_or("").to_uppercase();
    if kind == "STRING" && nonempty(&value["enum"]) {
        fields.push(format!("enum:{}", format_value(&value["enum"], true)));
    }
    if kind == "ARRAY" {
        if let Some(items) = value["items"].as_object().filter(|o| !o.is_empty()) {
            let items = items
                .iter()
                .filter(|(_, v)| !v.is_null())
                .map(|(key, v)| match key.as_str() {
                    "properties" => format!("properties:{{{}}}", properties(v)),
                    "type" => format!("type:{}", format_value(&typed(v), true)),
                    _ => format!("{key}:{}", format_value(v, true)),
                })
                .collect::<Vec<_>>()
                .join(",");
            fields.push(format!("items:{{{items}}}"));
        }
    }
    if nonempty(&value["nullable"]) {
        fields.push("nullable:true".into());
    }
    if kind == "OBJECT" {
        let props = value
            .get("properties")
            .filter(|v| v.is_object())
            .cloned()
            .unwrap_or_else(|| {
                Value::Object(
                    value
                        .as_object()
                        .into_iter()
                        .flatten()
                        .filter(|(k, _)| {
                            !["description", "type", "properties", "required", "nullable"]
                                .contains(&k.as_str())
                        })
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect(),
                )
            });
        fields.push(format!("properties:{{{}}}", properties(&props)));
        if nonempty(&value["required"]) {
            fields.push(format!(
                "required:{}",
                format_value(&value["required"], true)
            ));
        }
    }
    fields.push(format!("type:<|\"|>{kind}<|\"|>"));
    format!("{{{}}}", fields.join(","))
}

fn declaration(function: &Value) -> String {
    let description = function["description"].as_str().unwrap_or("");
    let mut out = format!("{{description:<|\"|>{description}<|\"|>");
    let params = &function["parameters"];
    if nonempty(params) {
        let mut fields = Vec::new();
        if nonempty(&params["properties"]) {
            fields.push(format!(
                "properties:{{{}}}",
                properties(&params["properties"])
            ));
        }
        if nonempty(&params["required"]) {
            fields.push(format!(
                "required:{}",
                format_value(&params["required"], true)
            ));
        }
        if nonempty(&params["type"]) {
            fields.push(format!(
                "type:{}",
                format_value(&typed(&params["type"]), true)
            ));
        }
        out.push_str(&format!(",parameters:{{{}}}", fields.join(",")));
    }
    if let Some(response) = function.get("response") {
        let mut fields = Vec::new();
        if nonempty(&response["description"]) {
            fields.push(format!(
                "description:{}",
                format_value(&response["description"], true)
            ));
        }
        if response["type"]
            .as_str()
            .is_some_and(|t| t.eq_ignore_ascii_case("object"))
        {
            fields.push("type:<|\"|>OBJECT<|\"|>".into());
        }
        out.push_str(&format!(",response:{{{}}}", fields.join(",")));
    }
    out.push('}');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn resolves_result_name_from_call_id_and_continues_turn() {
        let mut call = ChatTemplateMessage::new("assistant", "");
        call.tool_calls.push(
            r#"{"id":"abc","function":{"name":"weather","arguments":{"city":"Paris"}}}"#.into(),
        );
        let mut result = ChatTemplateMessage::new("tool", "18");
        result.tool_call_id = Some("abc".into());
        let rendered = render_gemma4_chat_with_tools(
            &[
                call,
                result,
                ChatTemplateMessage::new("assistant", "Dix-huit"),
            ],
            &[],
            false,
            false,
        );
        assert_eq!(rendered, "<bos><|turn>model\n<|tool_call>call:weather{city:<|\"|>Paris<|\"|>}<tool_call|><|tool_response>response:weather{value:<|\"|>18<|\"|>}<tool_response|>Dix-huit<turn|>\n");
    }
}

#[cfg(test)]
mod tool_turn_tests {
    use super::*;
    #[test]
    fn final_tool_result_continues_without_generation_prefix() {
        let mut call = ChatTemplateMessage::new("assistant", "");
        call.tool_calls
            .push(r#"{"id":"c","function":{"name":"f","arguments":{}}}"#.into());
        let mut response = ChatTemplateMessage::new("tool", "ok");
        response.tool_call_id = Some("c".into());
        response.name = Some("wrong".into());
        let rendered = render_gemma4_chat_with_tools(&[call.clone(), response], &[], true, false);
        assert_eq!(rendered, "<bos><|turn>model\n<|tool_call>call:f{}<tool_call|><|tool_response>response:f{value:<|\"|>ok<|\"|>}<tool_response|>");
        let pending = render_gemma4_chat_with_tools(&[call], &[], true, false);
        assert_eq!(
            pending,
            "<bos><|turn>model\n<|tool_call>call:f{}<tool_call|><|tool_response>"
        );
    }
    #[test]
    fn declaration_filters_schema_fields_and_renders_nested_types() {
        let tool = serde_json::json!({"name":"f","description":"D","parameters":{"type":"object","additionalProperties":false,"properties":{"items":{"type":"array","items":{"type":"object","properties":{"value":{"type":"string","enum":["a","b"],"default":"a"}},"required":["value"]},"nullable":true}},"required":[]}});
        assert_eq!(declaration(&tool), "{description:<|\"|>D<|\"|>,parameters:{properties:{items:{items:{properties:{value:{enum:[<|\"|>a<|\"|>,<|\"|>b<|\"|>],type:<|\"|>STRING<|\"|>}},required:[<|\"|>value<|\"|>],type:<|\"|>OBJECT<|\"|>},nullable:true,type:<|\"|>ARRAY<|\"|>}},type:<|\"|>OBJECT<|\"|>}}");
    }
}
