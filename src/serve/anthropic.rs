//! Shim Anthropic Messages pour `saragossa serve`.

use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::error::{ServeError, ServeResult};
use super::flint_output::FlintEvent;
use super::http::{send_json, write_headers};
use super::protocol::{
    json_bytes, normalize_session_key, ChatCompletionRequest, StopSpec, Usage, WireMessage,
};
use super::state::{ServeState, ServedCompletion};
use super::streaming::{CompletionSink, CompletionStreamEvent};
use super::tool_output::ToolOutputParser;

mod live;

use self::live::AnthropicLiveBlocks;

/// Sert `POST /v1/messages`.
pub(super) fn handle_anthropic_messages<S: Write>(
    stream: &mut S,
    state: &mut ServeState,
    session_header: Option<&str>,
    body: &[u8],
) -> ServeResult<()> {
    let (chat, session_key, stream_enabled) = anthropic_chat(body, session_header)?;
    chat.validate(state.max_tokens_cap())?;
    let parser = state.output_parser(&chat)?;
    if stream_enabled {
        send_anthropic_sse_streaming(stream, state, chat, session_key.as_deref(), parser)
    } else {
        let completion = state.complete(chat, session_key.as_deref())?;
        let response = AnthropicMessageResponse::with_parser(&completion, parser);
        send_json(stream, 200, &response, completion.metric_headers())
    }
}

/// Même conversion Messages et même encodage que la génération, sans budget de sortie.
pub(super) fn handle_count_tokens<S: Write>(
    stream: &mut S,
    state: &ServeState,
    body: &[u8],
) -> ServeResult<()> {
    validate_count_body(body)?;
    let mut request = parse_anthropic_messages_request(body)?;
    request.max_tokens = Some(0);
    let chat = request.to_chat_request()?;
    chat.validate(0)?;
    let count = state.count_tokens(&chat)?;
    send_json(stream, 200, &json!({"input_tokens": count}), Vec::new())
}

/// Le comptage ne doit pas transformer une modalité inconnue en texte JSON.
fn validate_count_body(body: &[u8]) -> ServeResult<()> {
    let value: Value = serde_json::from_slice(body)
        .map_err(|_| ServeError::Http("comptage Anthropic: JSON invalide".into()))?;
    for key in ["thinking", "tool_choice", "output_config", "output_format"] {
        if value.get(key).is_some_and(|v| !v.is_null()) {
            return Err(ServeError::not_implemented(format!(
                "comptage Anthropic: {key}"
            )));
        }
    }
    fn content(value: &Value, system: bool) -> ServeResult<()> {
        if value.is_string() || value.is_null() {
            return Ok(());
        }
        let blocks = value
            .as_array()
            .ok_or_else(|| ServeError::Http("contenu Anthropic invalide".into()))?;
        for block in blocks {
            match block["type"].as_str() {
                Some("text") if block["text"].is_string() => {}
                Some("tool_use")
                    if !system && block["name"].is_string() && block["input"].is_object() => {}
                Some("tool_result") if !system => content(&block["content"], true)?,
                _ => {
                    return Err(ServeError::not_implemented(
                        "comptage Anthropic: bloc non textuel ou invalide",
                    ))
                }
            }
        }
        Ok(())
    }
    content(&value["system"], true)?;
    if let Some(messages) = value["messages"].as_array() {
        for message in messages {
            content(&message["content"], false)?;
        }
    }
    Ok(())
}

/// Convertit un corps Messages en requête chat : (chat, clé de session,
/// streaming demandé).
pub(super) fn anthropic_chat(
    body: &[u8],
    session_header: Option<&str>,
) -> ServeResult<(ChatCompletionRequest, Option<String>, bool)> {
    let request = parse_anthropic_messages_request(body)?;
    let session_key = request.session_key(session_header);
    Ok((request.to_chat_request()?, session_key, request.stream))
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
    logit_bias: Option<Value>,
    #[serde(default)]
    stop_sequences: Vec<String>,
    #[serde(default)]
    tools: Vec<Value>,
    #[serde(default)]
    tool_choice: Option<Value>,
    #[serde(default)]
    metadata: AnthropicMetadata,
}

impl AnthropicMessagesRequest {
    fn session_key(&self, header: Option<&str>) -> Option<String> {
        header
            .and_then(normalize_session_key)
            .or_else(|| {
                self.metadata
                    .user_id
                    .as_deref()
                    .and_then(normalize_session_key)
            })
            .map(ToString::to_string)
    }

    fn to_chat_request(&self) -> ServeResult<ChatCompletionRequest> {
        let max_tokens = self
            .max_tokens
            .ok_or_else(|| ServeError::Http("max_tokens Anthropic requis".to_string()))?;
        let mut messages = Vec::new();
        let system = system_to_text(self.system.as_ref());
        if !system.is_empty() {
            messages.push(wire_message("system", system));
        }
        for message in &self.messages {
            messages.extend(message.to_wire_messages()?);
        }
        let stop_sequences = self.stop_sequences.clone();
        let stop = if stop_sequences.is_empty() {
            None
        } else {
            Some(StopSpec::Many(stop_sequences))
        };
        Ok(ChatCompletionRequest {
            model: self.model.clone(),
            messages,
            stream: false,
            n: None,
            max_tokens: Some(max_tokens),
            max_completion_tokens: None,
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: None,
            stop,
            response_format: None,
            user: None,
            tools: Some(convert_tools(&self.tools)?),
            tool_choice: anthropic_tool_choice(self.tool_choice.as_ref())?,
            parallel_tool_calls: self
                .tool_choice
                .as_ref()
                .and_then(|v| v["disable_parallel_tool_use"].as_bool())
                .map(|disable| !disable),
            seed: None,
            chat_template_kwargs: None,
            stream_options: None,
            min_p: None,
            logit_bias: self.logit_bias.clone(),
            presence_penalty: None,
            frequency_penalty: None,
            repetition_penalty: None,
        })
    }
}

#[derive(Clone, Debug, Default, Deserialize)]
struct AnthropicMetadata {
    #[serde(default)]
    user_id: Option<String>,
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
            "assistant" => Ok(vec![assistant_message(self.content.as_ref())?]),
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
    #[cfg(test)]
    fn from_completion(completion: &ServedCompletion) -> Self {
        Self::with_parser(completion, None)
    }

    fn with_parser(completion: &ServedCompletion, parser: Option<ToolOutputParser>) -> Self {
        let output = match parser {
            Some(parser) => AssistantOutput::from_events(parser.parse_events(&completion.content)),
            None => AssistantOutput::literal(&completion.content),
        };
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
    fn literal(raw: &str) -> Self {
        Self {
            blocks: vec![AnthropicContentBlock::Text {
                text: raw.to_owned(),
            }],
            has_tool_use: false,
        }
    }

    fn from_events(events: Vec<FlintEvent>) -> Self {
        let mut blocks = Vec::new();
        let mut has_tool_use = false;
        for event in events {
            match event {
                FlintEvent::Content(text) if !text.is_empty() => {
                    if let Some(AnthropicContentBlock::Text { text: previous }) = blocks.last_mut()
                    {
                        previous.push_str(&text);
                    } else {
                        blocks.push(AnthropicContentBlock::Text { text });
                    }
                }
                FlintEvent::ToolCall(call) => {
                    has_tool_use = true;
                    blocks.push(AnthropicContentBlock::ToolUse {
                        id: call.id,
                        name: call.function.name,
                        input: serde_json::from_str(&call.function.arguments)
                            .unwrap_or_else(|_| json!({})),
                    });
                }
                _ => {}
            }
        }
        if blocks.is_empty() {
            return Self::literal("");
        }
        Self {
            blocks,
            has_tool_use,
        }
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
            let mut message = wire_message("tool", result);
            message.tool_call_id = block
                .get("tool_use_id")
                .or_else(|| block.get("id"))
                .and_then(Value::as_str)
                .map(str::to_string);
            message.name = block
                .get("name")
                .and_then(Value::as_str)
                .map(str::to_string);
            messages.push(message);
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
        assistant_blocks: None,
        role: role.to_string(),
        content: Some(Value::String(content)),
        reasoning_content: None,
        tool_calls: None,
        name: None,
        tool_call_id: None,
    }
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

fn assistant_message(content: Option<&AnthropicContent>) -> ServeResult<WireMessage> {
    let Some(AnthropicContent::Blocks(blocks)) = content else {
        return Ok(wire_message("assistant", content_to_text(content)));
    };
    let mut message = wire_message("assistant", String::new());
    let mut text = String::new();
    let mut calls = Vec::new();
    let mut ordered = Vec::new();
    use saragossa::chat_template::AssistantBlock;
    for block in blocks {
        if block["type"] == "tool_use" {
            let id = block["id"]
                .as_str()
                .filter(|s| !s.is_empty())
                .ok_or_else(|| ServeError::Http("tool_use sans id".into()))?;
            let name = block["name"]
                .as_str()
                .filter(|s| !s.is_empty())
                .ok_or_else(|| ServeError::Http("tool_use sans name".into()))?;
            let input = block
                .get("input")
                .filter(|v| v.is_object())
                .ok_or_else(|| ServeError::Http("tool_use input doit être un objet".into()))?;
            let raw = json!({"id":id,"type":"function","function":{"name":name,"arguments":input}})
                .to_string();
            ordered.push(AssistantBlock::ToolCall(calls.len()));
            calls.push(
                serde_json::value::RawValue::from_string(raw)
                    .map_err(|e| ServeError::json("tool_use", e))?,
            );
        } else {
            let part = block_to_text(block);
            push_text_part(&mut text, &part);
            ordered.push(AssistantBlock::Text(part));
        }
    }
    message.assistant_blocks = Some(ordered);
    message.content = Some(Value::String(text));
    message.tool_calls = Some(calls);
    Ok(message)
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
    object
        .get("tool_use_id")
        .or_else(|| object.get("id"))
        .and_then(Value::as_str)
        .filter(|id| !id.trim().is_empty())
        .ok_or_else(|| ServeError::Http("tool_result sans tool_use_id".to_string()))?;
    Ok(Some(
        object
            .get("content")
            .map(content_value_to_text)
            .unwrap_or_default(),
    ))
}

fn convert_tools(tools: &[Value]) -> ServeResult<Vec<super::protocol::DeclaredTool>> {
    tools.iter().enumerate().map(|(index, tool)| {
        let name = tool.get("name").and_then(Value::as_str).filter(|s| !s.trim().is_empty())
            .ok_or_else(|| ServeError::Http(format!("tools[{index}] sans name")))?;
        let parameters = tool.get("input_schema").or_else(|| tool.get("parameters")).cloned().unwrap_or_else(|| json!({"type":"object"}));
        let value = json!({"type":"function","function":{"name":name,"description":tool.get("description").and_then(Value::as_str).unwrap_or(""),"parameters":parameters}});
        Ok(super::protocol::DeclaredTool { json: value.to_string(), value })
    }).collect()
}

fn anthropic_tool_choice(choice: Option<&Value>) -> ServeResult<Option<Value>> {
    let Some(choice) = choice else {
        return Ok(None);
    };
    match choice.get("type").and_then(Value::as_str) {
        Some("auto") => Ok(Some(json!("auto"))),
        Some("none") => Ok(Some(json!("none"))),
        Some("any" | "tool") => Err(ServeError::Http(
            "tool_choice forcé non pris en charge ; utiliser auto".into(),
        )),
        _ => Err(ServeError::Http("tool_choice Anthropic invalide".into())),
    }
}

fn send_anthropic_sse_streaming<S: Write>(
    stream: &mut S,
    state: &mut ServeState,
    chat: ChatCompletionRequest,
    session_key: Option<&str>,
    parser: Option<ToolOutputParser>,
) -> ServeResult<()> {
    let mut sink = AnthropicSse::with_parser(stream, parser);
    let result = state.complete_streaming(chat, session_key, |event| sink.event(event));
    sink.finish(result)
}

/// Réponse SSE Anthropic Messages d'un client.
pub(super) struct AnthropicSse<'a, S> {
    stream: &'a mut S,
    started: bool,
    live_blocks: AnthropicLiveBlocks,
    message_id: String,
    parser: Option<ToolOutputParser>,
}

impl<'a, S: Write> AnthropicSse<'a, S> {
    pub(super) fn with_parser(stream: &'a mut S, parser: Option<ToolOutputParser>) -> Self {
        Self {
            parser,
            stream,
            started: false,
            live_blocks: AnthropicLiveBlocks::new(),
            message_id: response_id(),
        }
    }

    fn flush(&mut self) -> ServeResult<()> {
        self.stream
            .flush()
            .map_err(|e| ServeError::io("flush SSE Anthropic", e))
    }
}

impl<S: Write> CompletionSink for AnthropicSse<'_, S> {
    fn event(&mut self, event: CompletionStreamEvent<'_>) -> ServeResult<()> {
        match event {
            CompletionStreamEvent::Start(start) => {
                self.started = true;
                write_headers(
                    self.stream,
                    200,
                    "OK",
                    "text/event-stream; charset=utf-8",
                    None,
                    start.metric_headers(),
                )?;
                write_anthropic_sse_event(
                    self.stream,
                    "message_start",
                    &json!({
                        "type": "message_start",
                        "message": {
                            "id": self.message_id.as_str(),
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
            }
            CompletionStreamEvent::Delta(delta) => {
                if delta.is_empty() {
                    return Ok(());
                }
                if let Some(parser) = self.parser.as_mut() {
                    for event in parser.push(delta) {
                        self.live_blocks.common_event(self.stream, event)?;
                    }
                } else {
                    self.live_blocks.push_text_delta(self.stream, delta)?;
                }
            }
            CompletionStreamEvent::TerminalError(error) => {
                write_anthropic_sse_event(
                    self.stream,
                    "error",
                    &json!({
                        "type": "error",
                        "error": {
                            "type": error.error_type,
                            "message": error.message
                        }
                    }),
                )?;
            }
        }
        self.flush()
    }

    fn finish(&mut self, result: ServeResult<ServedCompletion>) -> ServeResult<()> {
        let completion = match result {
            Ok(completion) => completion,
            Err(error) if self.started => {
                eprintln!("saragossa serve stream Anthropic interrompu: {error}");
                return Ok(());
            }
            Err(error) => return Err(error),
        };
        let has_tool_use = if let Some(parser) = self.parser.as_mut() {
            for event in parser.finish() {
                self.live_blocks.common_event(self.stream, event)?;
            }
            self.live_blocks.finish(self.stream, &completion)?;
            parser.has_tool_calls()
        } else {
            self.live_blocks.finish(self.stream, &completion)?
        };
        write_anthropic_sse_event(
            self.stream,
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
        write_anthropic_sse_event(
            self.stream,
            "message_stop",
            &json!({"type": "message_stop"}),
        )?;
        self.flush()
    }
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
