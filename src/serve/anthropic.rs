//! Shim Anthropic Messages pour `saragossa serve`.

use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::error::{ServeError, ServeResult};
use super::http::{send_json, write_headers};
use super::protocol::{json_bytes, ChatCompletionRequest, StopSpec, Usage, WireMessage};
use super::state::{ServeState, ServedCompletion};
use super::streaming::CompletionStreamEvent;

mod live;

use self::live::AnthropicLiveBlocks;

/// Sert `POST /v1/messages`.
pub(super) fn handle_anthropic_messages<S: Write>(
    stream: &mut S,
    state: &mut ServeState,
    body: &[u8],
) -> ServeResult<()> {
    let request = parse_anthropic_messages_request(body)?;
    let stream_enabled = request.stream;
    let chat = request.to_chat_request()?;
    if stream_enabled {
        send_anthropic_sse_streaming(stream, state, chat)
    } else {
        let completion = state.complete(chat)?;
        let response = AnthropicMessageResponse::from_completion(&completion);
        send_json(stream, 200, &response, completion.metric_headers())
    }
}

pub(super) fn send_anthropic_error<S: Write>(
    stream: &mut S,
    status: u16,
    message: &str,
) -> ServeResult<()> {
    let body = json!({
        "type": "error",
        "error": {
            "type": anthropic_error_type(status),
            "message": message
        }
    });
    send_json(stream, status, &body, Vec::new())
}

#[cfg(test)]
fn handle_anthropic_messages_with_completion<S, F>(
    stream: &mut S,
    body: &[u8],
    complete: F,
) -> ServeResult<()>
where
    S: Write,
    F: FnOnce(ChatCompletionRequest) -> ServeResult<ServedCompletion>,
{
    let request = parse_anthropic_messages_request(body)?;
    let chat = request.to_chat_request()?;
    let completion = complete(chat)?;
    let response = AnthropicMessageResponse::from_completion(&completion);
    send_json(stream, 200, &response, completion.metric_headers())
}

fn parse_anthropic_messages_request(body: &[u8]) -> ServeResult<AnthropicMessagesRequest> {
    serde_json::from_slice(body)
        .map_err(|e| ServeError::json("désérialisation messages Anthropic", e))
}

#[derive(Clone, Debug, Deserialize)]
struct AnthropicMessagesRequest {
    model: String,
    max_tokens: Option<usize>,
    messages: Vec<AnthropicInputMessage>,
    #[serde(default)]
    system: Option<AnthropicContent>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    stop_sequences: Vec<String>,
    #[serde(default)]
    tools: Vec<Value>,
}

impl AnthropicMessagesRequest {
    fn to_chat_request(&self) -> ServeResult<ChatCompletionRequest> {
        let max_tokens = self
            .max_tokens
            .ok_or_else(|| ServeError::Http("max_tokens Anthropic requis".to_string()))?;
        let mut messages = Vec::new();
        let system = system_with_tools(self.system.as_ref(), &self.tools)?;
        if !system.is_empty() {
            messages.push(wire_message("system", system));
        }
        for message in &self.messages {
            messages.extend(message.to_wire_messages()?);
        }
        let mut stop_sequences = self.stop_sequences.clone();
        if !self.tools.is_empty() && !stop_sequences.iter().any(|stop| stop == "</tool_call>") {
            stop_sequences.push("</tool_call>".to_string());
        }
        let stop = if stop_sequences.is_empty() {
            None
        } else {
            Some(StopSpec::Many(stop_sequences))
        };
        Ok(ChatCompletionRequest {
            model: self.model.clone(),
            messages,
            stream: false,
            max_tokens: Some(max_tokens),
            max_completion_tokens: None,
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: None,
            stop,
            response_format: None,
        })
    }
}

#[derive(Clone, Debug, Deserialize)]
struct AnthropicInputMessage {
    role: String,
    #[serde(default)]
    content: Option<AnthropicContent>,
}

impl AnthropicInputMessage {
    fn to_wire_messages(&self) -> ServeResult<Vec<WireMessage>> {
        let role = self.role.trim().to_ascii_lowercase();
        match role.as_str() {
            "assistant" => Ok(vec![wire_message(
                "assistant",
                assistant_content_to_text(self.content.as_ref()),
            )]),
            "user" => user_content_to_wire_messages(self.content.as_ref()),
            _ => Err(ServeError::Http(format!(
                "rôle Anthropic non supporté: {}",
                self.role
            ))),
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
enum AnthropicContent {
    Text(String),
    Blocks(Vec<Value>),
}

#[derive(Debug, Serialize)]
struct AnthropicMessageResponse {
    id: String,
    #[serde(rename = "type")]
    kind: &'static str,
    role: &'static str,
    model: String,
    content: Vec<AnthropicContentBlock>,
    stop_reason: &'static str,
    stop_sequence: Option<String>,
    usage: AnthropicUsage,
}

impl AnthropicMessageResponse {
    fn from_completion(completion: &ServedCompletion) -> Self {
        let output = AssistantOutput::parse(&completion.content);
        Self {
            id: response_id(),
            kind: "message",
            role: "assistant",
            model: completion.model.clone(),
            content: output.blocks,
            stop_reason: anthropic_stop_reason(completion, output.has_tool_use),
            stop_sequence: anthropic_stop_sequence(completion, output.has_tool_use),
            usage: AnthropicUsage::from_usage(completion.usage),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type")]
enum AnthropicContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
}

#[derive(Debug)]
struct AssistantOutput {
    blocks: Vec<AnthropicContentBlock>,
    has_tool_use: bool,
}

#[derive(Debug, Serialize)]
struct AnthropicUsage {
    input_tokens: usize,
    output_tokens: usize,
}

impl AnthropicUsage {
    fn from_usage(usage: Usage) -> Self {
        Self {
            input_tokens: usage.prompt_tokens(),
            output_tokens: usage.completion_tokens(),
        }
    }
}

impl AssistantOutput {
    fn parse(raw: &str) -> Self {
        let mut blocks = Vec::new();
        let mut rest = strip_think_blocks(raw);
        let mut index = 0_usize;
        loop {
            let Some(start) = rest.find("<tool_call") else {
                push_text_block(&mut blocks, rest.trim());
                break;
            };
            push_text_block(&mut blocks, rest[..start].trim());
            let after_tag = &rest[start..];
            let Some(tag_end) = after_tag.find('>') else {
                push_text_block(&mut blocks, after_tag.trim());
                break;
            };
            let after_open = &after_tag[tag_end + 1..];
            let (inner, after_close) = match after_open.find("</tool_call>") {
                Some(end) => (
                    &after_open[..end],
                    &after_open[end + "</tool_call>".len()..],
                ),
                None => (after_open, ""),
            };
            match parse_tool_call_block(inner, index) {
                Some(block) => {
                    blocks.push(block);
                    index += 1;
                }
                None => push_text_block(&mut blocks, after_tag.trim()),
            }
            if after_close.is_empty() {
                break;
            }
            rest = after_close.to_string();
        }
        if blocks.is_empty() {
            blocks.push(AnthropicContentBlock::Text {
                text: String::new(),
            });
        }
        let has_tool_use = blocks
            .iter()
            .any(|block| matches!(block, AnthropicContentBlock::ToolUse { .. }));
        Self {
            blocks,
            has_tool_use,
        }
    }
}

fn push_text_block(blocks: &mut Vec<AnthropicContentBlock>, text: &str) {
    if text.is_empty() {
        return;
    }
    blocks.push(AnthropicContentBlock::Text {
        text: text.to_string(),
    });
}

fn parse_tool_call_block(inner: &str, index: usize) -> Option<AnthropicContentBlock> {
    let value: Value = serde_json::from_str(inner.trim()).ok()?;
    let function = value.get("function").unwrap_or(&value);
    let name = function.get("name")?.as_str()?.trim();
    if name.is_empty() {
        return None;
    }
    let id = value
        .get("id")
        .and_then(Value::as_str)
        .filter(|id| !id.trim().is_empty())
        .map_or_else(|| format!("toolu_saragossa_{index}"), ToString::to_string);
    let input = tool_input_from_call(function);
    Some(AnthropicContentBlock::ToolUse {
        id,
        name: name.to_string(),
        input,
    })
}

fn tool_input_from_call(function: &Value) -> Value {
    for key in ["input", "arguments"] {
        if let Some(input) = function.get(key).and_then(tool_input_object) {
            return input;
        }
    }
    json!({})
}

fn tool_input_object(value: &Value) -> Option<Value> {
    match value {
        Value::Object(_) => Some(value.clone()),
        Value::String(text) => serde_json::from_str::<Value>(text)
            .ok()
            .filter(Value::is_object),
        _ => None,
    }
}

fn user_content_to_wire_messages(
    content: Option<&AnthropicContent>,
) -> ServeResult<Vec<WireMessage>> {
    let Some(AnthropicContent::Blocks(blocks)) = content else {
        return Ok(vec![wire_message("user", content_to_text(content))]);
    };
    let mut messages = Vec::new();
    let mut text = String::new();
    for block in blocks {
        if let Some(result) = tool_result_text(block)? {
            if !text.is_empty() {
                messages.push(wire_message("user", std::mem::take(&mut text)));
            }
            messages.push(wire_message("tool", result));
        } else {
            push_text_part(&mut text, &block_to_text(block));
        }
    }
    if !text.is_empty() || messages.is_empty() {
        messages.push(wire_message("user", text));
    }
    Ok(messages)
}

fn wire_message(role: &str, content: String) -> WireMessage {
    WireMessage {
        role: role.to_string(),
        content: Some(Value::String(content)),
    }
}

fn system_with_tools(content: Option<&AnthropicContent>, tools: &[Value]) -> ServeResult<String> {
    let system = system_to_text(content);
    let tools = render_tools_block(tools)?;
    Ok(join_text_parts([system, tools]))
}

fn system_to_text(content: Option<&AnthropicContent>) -> String {
    match content {
        Some(AnthropicContent::Blocks(blocks)) => blocks
            .iter()
            .map(block_to_text)
            .map(|text| filter_system_text(&text))
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join("\n\n"),
        _ => filter_system_text(&content_to_text(content)),
    }
}

fn filter_system_text(text: &str) -> String {
    // Claude Code transmet parfois cet en-tête comme bloc system synthétique;
    // le modèle local ne doit pas le traiter comme instruction utilisateur.
    text.lines()
        .filter(|line| {
            !line
                .trim()
                .to_ascii_lowercase()
                .starts_with("x-anthropic-billing-header:")
        })
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

fn content_to_text(content: Option<&AnthropicContent>) -> String {
    match content {
        Some(AnthropicContent::Text(text)) => text.clone(),
        Some(AnthropicContent::Blocks(blocks)) => blocks_to_text(blocks),
        None => String::new(),
    }
}

fn content_value_to_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Array(items) => blocks_to_text(items),
        Value::Object(_) => block_to_text(value),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

fn assistant_content_to_text(content: Option<&AnthropicContent>) -> String {
    match content {
        Some(AnthropicContent::Blocks(blocks)) => {
            let mut text = String::new();
            for block in blocks {
                push_text_part(&mut text, &block_to_text(block));
            }
            text
        }
        _ => content_to_text(content),
    }
}

fn blocks_to_text(blocks: &[Value]) -> String {
    join_text_parts(blocks.iter().map(block_to_text))
}

fn join_text_parts(parts: impl IntoIterator<Item = String>) -> String {
    parts
        .into_iter()
        .filter(|part| !part.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn push_text_part(target: &mut String, part: &str) {
    if part.trim().is_empty() {
        return;
    }
    if !target.is_empty() {
        target.push_str("\n\n");
    }
    target.push_str(part);
}

fn block_to_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Object(object) => match object.get("type").and_then(Value::as_str) {
            Some("text") => object
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            Some("tool_result") => object
                .get("content")
                .map(content_value_to_text)
                .unwrap_or_default(),
            Some("tool_use") => tool_use_to_text(value),
            _ => object
                .get("text")
                .and_then(Value::as_str)
                .map_or_else(|| value.to_string(), ToString::to_string),
        },
        Value::Array(items) => items.iter().map(block_to_text).collect(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

fn tool_result_text(value: &Value) -> ServeResult<Option<String>> {
    let Some(object) = value.as_object() else {
        return Ok(None);
    };
    if object.get("type").and_then(Value::as_str) != Some("tool_result") {
        return Ok(None);
    }
    let tool_use_id = object
        .get("tool_use_id")
        .or_else(|| object.get("id"))
        .and_then(Value::as_str)
        .filter(|id| !id.trim().is_empty())
        .ok_or_else(|| ServeError::Http("tool_result sans tool_use_id".to_string()))?;
    Ok(Some(
        json!({
            "tool_use_id": tool_use_id,
            "content": object
                .get("content")
                .map(content_value_to_text)
                .unwrap_or_default()
        })
        .to_string(),
    ))
}

fn tool_use_to_text(value: &Value) -> String {
    let Some(object) = value.as_object() else {
        return String::new();
    };
    let name = object
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let arguments = object.get("input").cloned().unwrap_or_else(|| json!({}));
    let payload = json!({"name": name, "arguments": arguments});
    format!("<tool_call>{payload}</tool_call>")
}

fn render_tools_block(tools: &[Value]) -> ServeResult<String> {
    if tools.is_empty() {
        return Ok(String::new());
    }
    let mut rendered = Vec::with_capacity(tools.len());
    let mut names = Vec::with_capacity(tools.len());
    for (index, tool) in tools.iter().enumerate() {
        let object = tool
            .as_object()
            .ok_or_else(|| ServeError::Http(format!("tools[{index}] doit être un objet")))?;
        let name = object
            .get("name")
            .and_then(Value::as_str)
            .filter(|name| !name.trim().is_empty())
            .ok_or_else(|| ServeError::Http(format!("tools[{index}] sans name")))?;
        let parameters = object
            .get("input_schema")
            .or_else(|| object.get("parameters"))
            .cloned()
            .unwrap_or_else(|| json!({"type": "object"}));
        names.push(name.to_string());
        rendered.push(json!({
            "type": "function",
            "function": {
                "name": name,
                "description": object
                    .get("description")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
                "parameters": parameters
            }
        }));
    }
    Ok(format!(
        "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n\
         You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{}\n</tools>\n\n\
         Available function names: {}.\n\n\
         If a function is needed, output only one or more <tool_call> blocks. \
         Each block must contain a JSON object exactly like this shape:\n\
         <tool_call>{{\"name\":\"FUNCTION_NAME\",\"arguments\":{{}}}}</tool_call>\n\
         Replace FUNCTION_NAME with one available function name and fill arguments from the function schema. \
         Do not add prose, markdown, placeholder names, or arrays of name/value pairs.",
        rendered
            .iter()
            .map(Value::to_string)
            .collect::<Vec<_>>()
            .join("\n"),
        names.join(", ")
    ))
}

fn strip_think_blocks(raw: &str) -> String {
    const OPEN: &str = "<think>";
    const CLOSE: &str = "</think>";
    let Some(start) = raw.find(OPEN) else {
        return raw.to_string();
    };
    let Some(end_rel) = raw[start..].find(CLOSE) else {
        return raw.to_string();
    };
    let mut out = String::with_capacity(raw.len());
    out.push_str(&raw[..start]);
    out.push_str(&raw[start + end_rel + CLOSE.len()..]);
    out
}

fn send_anthropic_sse_streaming<S: Write>(
    stream: &mut S,
    state: &mut ServeState,
    chat: ChatCompletionRequest,
) -> ServeResult<()> {
    let mut stream_started = false;
    let mut live_blocks = AnthropicLiveBlocks::new();
    let message_id = response_id();
    let result = state.complete_streaming(chat, |event| match event {
        CompletionStreamEvent::Start(start) => {
            stream_started = true;
            write_headers(
                stream,
                200,
                "OK",
                "text/event-stream; charset=utf-8",
                None,
                start.metric_headers(),
            )?;
            write_anthropic_sse_event(
                stream,
                "message_start",
                &json!({
                    "type": "message_start",
                    "message": {
                        "id": message_id.as_str(),
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": start.model.as_str(),
                        "stop_reason": null,
                        "stop_sequence": null,
                        "usage": {
                            "input_tokens": start.prompt_tokens,
                            "output_tokens": 0
                        }
                    }
                }),
            )?;
            stream
                .flush()
                .map_err(|e| ServeError::io("flush SSE Anthropic", e))
        }
        CompletionStreamEvent::Delta(delta) => {
            if delta.is_empty() {
                return Ok(());
            }
            live_blocks.push_text_delta(stream, delta)?;
            stream
                .flush()
                .map_err(|e| ServeError::io("flush SSE Anthropic", e))
        }
    });
    let completion = match result {
        Ok(completion) => completion,
        Err(error) if stream_started => {
            eprintln!("saragossa serve stream Anthropic interrompu: {error}");
            return Ok(());
        }
        Err(error) => return Err(error),
    };
    let has_tool_use = live_blocks.finish(stream, &completion)?;
    write_anthropic_sse_event(
        stream,
        "message_delta",
        &json!({
            "type": "message_delta",
            "delta": {
                "stop_reason": anthropic_stop_reason(&completion, has_tool_use),
                "stop_sequence": anthropic_stop_sequence(&completion, has_tool_use)
            },
            "usage": {
                "output_tokens": completion.usage.completion_tokens()
            }
        }),
    )?;
    write_anthropic_sse_event(stream, "message_stop", &json!({"type": "message_stop"}))?;
    stream
        .flush()
        .map_err(|e| ServeError::io("flush SSE Anthropic", e))
}

fn write_anthropic_content_block<S: Write>(
    stream: &mut S,
    index: usize,
    block: &AnthropicContentBlock,
) -> ServeResult<()> {
    match block {
        AnthropicContentBlock::Text { text } => {
            write_anthropic_sse_event(
                stream,
                "content_block_start",
                &json!({
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {"type": "text", "text": ""}
                }),
            )?;
            if !text.is_empty() {
                write_anthropic_sse_event(
                    stream,
                    "content_block_delta",
                    &json!({
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {"type": "text_delta", "text": text}
                    }),
                )?;
            }
        }
        AnthropicContentBlock::ToolUse { id, name, input } => {
            write_anthropic_sse_event(
                stream,
                "content_block_start",
                &json!({
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {
                        "type": "tool_use",
                        "id": id,
                        "name": name,
                        "input": {}
                    }
                }),
            )?;
            write_anthropic_sse_event(
                stream,
                "content_block_delta",
                &json!({
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": input.to_string()
                    }
                }),
            )?;
        }
    }
    write_anthropic_sse_event(
        stream,
        "content_block_stop",
        &json!({"type": "content_block_stop", "index": index}),
    )
}

fn write_anthropic_sse_event<S: Write, T: Serialize>(
    stream: &mut S,
    event: &str,
    value: &T,
) -> ServeResult<()> {
    let mut bytes = b"event: ".to_vec();
    bytes.extend_from_slice(event.as_bytes());
    bytes.extend_from_slice(b"\ndata: ");
    bytes.extend(json_bytes(value)?);
    bytes.extend_from_slice(b"\n\n");
    stream
        .write_all(&bytes)
        .map_err(|e| ServeError::io("écriture SSE Anthropic", e))
}

fn anthropic_stop_reason(completion: &ServedCompletion, has_tool_use: bool) -> &'static str {
    if has_tool_use {
        "tool_use"
    } else if completion.finish_reason == "length" {
        "max_tokens"
    } else if completion.matched_stop.is_some() {
        "stop_sequence"
    } else {
        "end_turn"
    }
}

fn anthropic_stop_sequence(completion: &ServedCompletion, has_tool_use: bool) -> Option<String> {
    if anthropic_stop_reason(completion, has_tool_use) == "stop_sequence" {
        completion.matched_stop.clone()
    } else {
        None
    }
}

fn anthropic_error_type(status: u16) -> &'static str {
    match status {
        400 => "invalid_request_error",
        401 => "authentication_error",
        404 => "not_found_error",
        429 => "rate_limit_error",
        _ => "api_error",
    }
}

fn response_id() -> String {
    let created = match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(duration) => duration.as_secs(),
        Err(_) => 0,
    };
    format!("msg_{created}")
}

#[cfg(test)]
mod tests;
