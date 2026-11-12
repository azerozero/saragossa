//! Conversion du profil Responses sans stockage vers le chat du moteur.

use serde_json::{json, Value};

use super::super::error::{ServeError, ServeResult};
use super::super::protocol::ChatCompletionRequest;

pub(super) struct Request {
    pub chat: ChatCompletionRequest,
    pub options: Value,
    pub uses_tools: bool,
}

fn invalid(field: &str) -> ServeError {
    ServeError::Http(format!("Responses: {field} invalide"))
}

fn string<'a>(value: &'a Value, key: &str) -> ServeResult<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| invalid(key))
}

fn text(value: &Value) -> ServeResult<String> {
    if let Some(text) = value.as_str() {
        return Ok(text.to_owned());
    }
    let parts = value.as_array().ok_or_else(|| invalid("content"))?;
    let mut result = String::new();
    for part in parts {
        if !matches!(part["type"].as_str(), Some("input_text" | "output_text")) {
            return Err(ServeError::not_implemented(
                "Responses: contenu autre que texte",
            ));
        }
        result.push_str(
            part["text"]
                .as_str()
                .ok_or_else(|| invalid("content.text"))?,
        );
    }
    Ok(result)
}

impl Request {
    pub fn parse(body: &[u8], cap: usize) -> ServeResult<Self> {
        let value: Value = serde_json::from_slice(body).map_err(|_| invalid("JSON"))?;
        let fields = value.as_object().ok_or_else(|| invalid("corps"))?;
        for (key, val) in fields {
            match key.as_str() {
                "previous_response_id" | "include" | "background" if !val.is_null() => {
                    return Err(ServeError::not_implemented(format!("Responses: {key}")));
                }
                "store" if val == &Value::Bool(true) => {
                    return Err(ServeError::not_implemented("Responses: store=true"));
                }
                "store" if !val.is_null() && val != &Value::Bool(false) => {
                    return Err(invalid("store"))
                }
                "model"
                | "input"
                | "instructions"
                | "max_output_tokens"
                | "temperature"
                | "top_p"
                | "stream"
                | "tools"
                | "tool_choice"
                | "parallel_tool_calls"
                | "store"
                | "previous_response_id"
                | "include"
                | "background" => {}
                _ => return Err(ServeError::not_implemented(format!("Responses: {key}"))),
            }
        }
        string(&value, "model")?;
        let mut messages = Vec::new();
        if let Some(instructions) = value.get("instructions").filter(|v| !v.is_null()) {
            let instructions = instructions
                .as_str()
                .ok_or_else(|| invalid("instructions"))?;
            messages.push(json!({"role":"system", "content":instructions}));
        }
        let mut uses_tools = false;
        let mut calls = std::collections::HashSet::new();
        // Le template flint représente les retours par position. Les call_id
        // servent à les réordonner selon les appels, même s'ils arrivent à l'envers.
        let mut pending = std::collections::VecDeque::new();
        let mut results = std::collections::HashMap::new();
        match value.get("input") {
            Some(Value::String(input)) => messages.push(json!({"role":"user", "content":input})),
            Some(Value::Array(items)) => {
                for item in items {
                    let kind = match item.get("type") {
                        None => "message",
                        Some(Value::String(kind)) => kind.as_str(),
                        _ => return Err(invalid("input.type")),
                    };
                    match kind {
                        "message" => {
                            if !pending.is_empty() {
                                return Err(invalid("function_call_output manquant"));
                            }
                            let role = string(item, "role")?;
                            if !matches!(role, "user" | "assistant" | "system" | "developer") {
                                return Err(invalid("role"));
                            }
                            messages
                                .push(json!({"role":if role == "developer" {"system"} else {role},
                            "content":text(&item["content"])?}));
                        }
                        "function_call" => {
                            if !results.is_empty() {
                                return Err(invalid("retours d'outils incomplets"));
                            }
                            uses_tools = true;
                            let id = string(item, "call_id")?;
                            if !calls.insert(id.to_owned()) {
                                return Err(invalid("call_id dupliqué"));
                            }
                            pending.push_back(id.to_owned());
                            let name = string(item, "name")?;
                            let arguments = string(item, "arguments")?;
                            let parsed: Value = serde_json::from_str(arguments)
                                .map_err(|_| invalid("arguments"))?;
                            if !parsed.is_object() {
                                return Err(invalid("arguments"));
                            }
                            let call = json!({"id":id,"type":"function","function":{"name":name,"arguments":arguments}});
                            if let Some(last) =
                                messages.last_mut().filter(|m| m["tool_calls"].is_array())
                            {
                                if let Some(previous) = last["tool_calls"].as_array_mut() {
                                    previous.push(call);
                                }
                            } else {
                                messages.push(
                                    json!({"role":"assistant","content":null,"tool_calls":[call]}),
                                );
                            }
                        }
                        "function_call_output" => {
                            uses_tools = true;
                            let id = string(item, "call_id")?;
                            if !pending.iter().any(|pending| pending == id)
                                || results.contains_key(id)
                            {
                                return Err(invalid("call_id du retour"));
                            }
                            results.insert(id.to_owned(), text(&item["output"])?);
                            if results.len() == pending.len() {
                                for id in pending.drain(..) {
                                    let content = results
                                        .remove(&id)
                                        .ok_or_else(|| invalid("call_id du retour"))?;
                                    messages.push(json!({"role":"tool","content":content}));
                                }
                            }
                        }
                        _ => {
                            return Err(ServeError::not_implemented("Responses: type d'item input"))
                        }
                    }
                }
            }
            _ => return Err(invalid("input")),
        }
        if !pending.is_empty() {
            return Err(invalid("function_call_output manquant"));
        }
        let mut tools = Vec::new();
        if let Some(declared) = value.get("tools") {
            for tool in declared.as_array().ok_or_else(|| invalid("tools"))? {
                if tool["type"] != "function" {
                    return Err(ServeError::not_implemented(
                        "Responses: seuls les outils function sont supportés",
                    ));
                }
                for key in tool.as_object().ok_or_else(|| invalid("tools"))?.keys() {
                    if !matches!(
                        key.as_str(),
                        "type" | "name" | "description" | "parameters" | "strict"
                    ) {
                        return Err(ServeError::not_implemented(format!(
                            "Responses: tools.{key}"
                        )));
                    }
                }
                if tool
                    .get("description")
                    .is_some_and(|v| !v.is_null() && !v.is_string())
                {
                    return Err(invalid("tools.description"));
                }
                string(tool, "name")?;
                if tool
                    .get("strict")
                    .is_some_and(|v| !v.is_null() && v != &Value::Bool(false))
                {
                    return Err(ServeError::not_implemented("Responses: outils stricts"));
                }
                if tool
                    .get("parameters")
                    .is_some_and(|v| !v.is_null() && !v.is_object())
                {
                    return Err(invalid("tools.parameters"));
                }
                let mut function = tool.clone();
                if let Some(object) = function.as_object_mut() {
                    object.remove("type");
                }
                tools.push(json!({"type":"function", "function":function}));
            }
        }
        uses_tools |= !tools.is_empty();
        let mut chat = json!({"model":value["model"],"messages":messages,"tools":tools});
        for key in [
            "temperature",
            "top_p",
            "stream",
            "parallel_tool_calls",
            "tool_choice",
        ] {
            if let Some(v) = value.get(key) {
                chat[key] = v.clone();
            }
        }
        if let Some(max) = value.get("max_output_tokens").filter(|v| !v.is_null()) {
            if max.as_u64().is_none_or(|v| v == 0) {
                return Err(invalid("max_output_tokens"));
            }
            chat["max_tokens"] = max.clone();
        }
        for (key, upper) in [("temperature", 2.0), ("top_p", 1.0)] {
            if let Some(v) = value.get(key).filter(|v| !v.is_null()) {
                if v.as_f64().is_none_or(|n| n < 0.0 || n > upper) {
                    return Err(invalid(key));
                }
            }
        }
        let mut chat = ChatCompletionRequest::parse(
            &serde_json::to_vec(&chat).map_err(|e| ServeError::json("Responses chat", e))?,
        )?;
        // Les valeurs effectives exposées dans Response restent indépendantes
        // des presets locaux de sampling du modèle.
        chat.temperature.get_or_insert(0.0);
        chat.top_p.get_or_insert(1.0);
        chat.top_k = Some(0);
        chat.validate(cap)?;
        chat.active_tools()?;
        Ok(Self {
            chat,
            options: value,
            uses_tools,
        })
    }
}
