use std::time::Duration;

use saragossa::decoder::GenerationTimings;
use serde_json::{json, Value};

use super::*;

#[test]
fn request_translates_blocks_system_and_stop_sequences() {
    let request: AnthropicMessagesRequest = serde_json::from_str(
        r#"{
            "model":"reti-35b",
            "max_tokens":64,
            "system":[
                {"type":"text","text":"x-anthropic-billing-header: skip"},
                {"type":"text","text":"system prompt"}
            ],
            "stop_sequences":["</stop>"],
            "temperature":0.2,
            "top_p":0.8,
            "messages":[
                {"role":"user","content":[
                    {"type":"text","text":"run"},
                    {"type":"tool_result","tool_use_id":"toolu_1","content":[{"type":"text","text":"ok"}]}
                ]},
                {"role":"assistant","content":[
                    {"type":"text","text":"done"},
                    {"type":"tool_use","id":"toolu_2","name":"Bash","input":{"command":"pwd"}}
                ]}
            ]
        }"#,
    )
    .expect("invariant: JSON Anthropic valide");

    let chat = request
        .to_chat_request()
        .expect("invariant: requête traduisible");

    assert_eq!(chat.model, "reti-35b");
    assert_eq!(chat.max_tokens(), 64);
    assert_eq!(chat.temperature, Some(0.2));
    assert_eq!(chat.top_p, Some(0.8));
    assert_eq!(chat.stop_texts(), vec!["</stop>".to_string()]);
    let messages = chat.template_messages();
    assert_eq!(messages[0].role, "system");
    assert_eq!(messages[0].content.as_deref(), Some("system prompt"));
    assert_eq!(messages[1].role, "user");
    assert_eq!(messages[1].content.as_deref(), Some("run"));
    assert_eq!(messages[2].role, "tool");
    assert_eq!(
        messages[2].content.as_deref(),
        Some(r#"{"content":"ok","tool_use_id":"toolu_1"}"#)
    );
    assert_eq!(messages[3].role, "assistant");
    let assistant = messages[3]
        .content
        .as_deref()
        .expect("invariant: contenu assistant");
    assert!(assistant.starts_with("done\n\n<tool_call>"));
    assert!(assistant.contains(r#""name":"Bash""#));
    assert!(assistant.contains(r#""command":"pwd""#));
}

#[test]
fn request_injects_tools_and_tool_stop_sequence() {
    let request: AnthropicMessagesRequest = serde_json::from_str(
        r#"{
            "model":"reti-35b",
            "max_tokens":64,
            "messages":[{"role":"user","content":"liste"}],
            "tools":[{
                "name":"Bash",
                "description":"Run a shell command",
                "input_schema":{
                    "type":"object",
                    "properties":{"command":{"type":"string"}}
                }
            }]
        }"#,
    )
    .expect("invariant: JSON Anthropic valide");

    let chat = request
        .to_chat_request()
        .expect("invariant: requête traduisible");
    let messages = chat.template_messages();

    assert_eq!(messages[0].role, "system");
    let system = messages[0]
        .content
        .as_deref()
        .expect("invariant: system injecté");
    assert!(system.contains("<tools>"));
    assert!(system.contains(r#""name":"Bash""#));
    assert!(system.contains("<tool_call>"));
    assert!(chat.stop_texts().contains(&"</tool_call>".to_string()));
}

#[test]
fn response_serializes_anthropic_message_shape() {
    let completion = fake_completion("reti-35b", "bonjour", "stop", None, 9, 3);

    let value = serde_json::to_value(AnthropicMessageResponse::from_completion(&completion))
        .expect("invariant: réponse sérialisable");

    assert_eq!(value["type"], "message");
    assert_eq!(value["role"], "assistant");
    assert_eq!(value["model"], "reti-35b");
    assert_eq!(
        value["content"],
        json!([{"type": "text", "text": "bonjour"}])
    );
    assert_eq!(value["stop_reason"], "end_turn");
    assert!(value["stop_sequence"].is_null());
    assert_eq!(
        value["usage"],
        json!({"input_tokens": 9, "output_tokens": 3})
    );
}

#[test]
fn response_maps_matched_stop_sequence() {
    let completion = fake_completion(
        "reti-35b",
        "avant",
        "stop",
        Some("</stop>".to_string()),
        2,
        2,
    );

    let value = serde_json::to_value(AnthropicMessageResponse::from_completion(&completion))
        .expect("invariant: réponse sérialisable");

    assert_eq!(value["stop_reason"], "stop_sequence");
    assert_eq!(value["stop_sequence"], "</stop>");
}

#[test]
fn response_maps_tool_call_to_anthropic_tool_use() {
    let completion = fake_completion(
        "reti-35b",
        r#"Je lance l'outil.
<tool_call>{"name":"Bash","arguments":{"command":"pwd"}}</tool_call>"#,
        "stop",
        None,
        4,
        8,
    );

    let value = serde_json::to_value(AnthropicMessageResponse::from_completion(&completion))
        .expect("invariant: réponse sérialisable");

    assert_eq!(value["stop_reason"], "tool_use");
    assert!(value["stop_sequence"].is_null());
    assert_eq!(
        value["content"][0],
        json!({"type":"text","text":"Je lance l'outil."})
    );
    assert_eq!(value["content"][1]["type"], "tool_use");
    assert_eq!(value["content"][1]["name"], "Bash");
    assert_eq!(value["content"][1]["input"], json!({"command": "pwd"}));
}

#[test]
fn output_parser_accepts_tool_call_closed_by_stop_sequence() {
    let output = AssistantOutput::parse(
        r#"<tool_call>{"name":"Read","arguments":{"file_path":"README.md"}}"#,
    );

    assert!(output.has_tool_use);
    let value = serde_json::to_value(&output.blocks[0]).expect("invariant: bloc sérialisable");
    assert_eq!(value["type"], "tool_use");
    assert_eq!(value["name"], "Read");
    assert_eq!(value["input"], json!({"file_path": "README.md"}));
}

#[test]
fn sse_events_follow_anthropic_order_and_fields() {
    let completion = fake_completion("reti-35b", "salut", "length", None, 5, 4);
    let mut stream = Vec::new();

    send_anthropic_sse(&mut stream, &completion).expect("invariant: SSE sérialisable");

    let body = http_body(&stream);
    let events = sse_events(body);
    assert_eq!(
        events
            .iter()
            .map(|(event, _)| event.as_str())
            .collect::<Vec<_>>(),
        vec![
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop"
        ]
    );
    assert_eq!(events[0].1["type"], "message_start");
    assert_eq!(events[0].1["message"]["content"], json!([]));
    assert_eq!(
        events[2].1["delta"],
        json!({"type": "text_delta", "text": "salut"})
    );
    assert_eq!(events[4].1["delta"]["stop_reason"], "max_tokens");
    assert_eq!(events[4].1["usage"], json!({"output_tokens": 4}));
    assert_eq!(events[5].1["type"], "message_stop");
}

#[test]
fn sse_events_stream_tool_use_blocks() {
    let completion = fake_completion(
        "reti-35b",
        r#"<tool_call>{"name":"Bash","arguments":{"command":"pwd"}}</tool_call>"#,
        "stop",
        None,
        5,
        4,
    );
    let mut stream = Vec::new();

    send_anthropic_sse(&mut stream, &completion).expect("invariant: SSE sérialisable");

    let events = sse_events(http_body(&stream));
    assert_eq!(
        events
            .iter()
            .map(|(event, _)| event.as_str())
            .collect::<Vec<_>>(),
        vec![
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop"
        ]
    );
    assert_eq!(events[1].1["content_block"]["type"], "tool_use");
    assert_eq!(events[1].1["content_block"]["name"], "Bash");
    assert_eq!(events[2].1["delta"]["type"], "input_json_delta");
    assert_eq!(events[2].1["delta"]["partial_json"], r#"{"command":"pwd"}"#);
    assert_eq!(events[4].1["delta"]["stop_reason"], "tool_use");
}

#[test]
fn anthropic_error_uses_messages_error_shape() {
    let mut stream = Vec::new();

    send_anthropic_error(&mut stream, 400, "requête invalide")
        .expect("invariant: erreur sérialisable");

    let value: Value = serde_json::from_str(http_body(&stream)).expect("invariant: JSON erreur");
    assert_eq!(value["type"], "error");
    assert_eq!(value["error"]["type"], "invalid_request_error");
    assert_eq!(value["error"]["message"], "requête invalide");
}

#[test]
fn handler_smoke_returns_anthropic_json_with_mock_completion() {
    let body = br#"{
        "model":"reti-35b",
        "max_tokens":4,
        "messages":[{"role":"user","content":"Bonjour"}]
    }"#;
    let mut stream = Vec::new();

    handle_anthropic_messages_with_completion(&mut stream, body, |request| {
        assert_eq!(request.model, "reti-35b");
        assert_eq!(request.max_tokens(), 4);
        Ok(fake_completion("reti-35b", "Salut", "stop", None, 3, 1))
    })
    .expect("invariant: handler Anthropic avec génération mockée");

    let response = String::from_utf8(stream).expect("invariant: réponse UTF-8");
    assert!(response.starts_with("HTTP/1.1 200 OK"));
    let value: Value =
        serde_json::from_str(http_body(response.as_bytes())).expect("invariant: JSON réponse");
    assert_eq!(value["type"], "message");
    assert_eq!(value["content"][0]["text"], "Salut");
    assert_eq!(
        value["usage"],
        json!({"input_tokens": 3, "output_tokens": 1})
    );
}

fn fake_completion(
    model: &str,
    content: &str,
    finish_reason: &'static str,
    matched_stop: Option<String>,
    input_tokens: usize,
    output_tokens: usize,
) -> ServedCompletion {
    ServedCompletion {
        model: model.to_string(),
        content: content.to_string(),
        finish_reason,
        matched_stop,
        usage: Usage::new(input_tokens, output_tokens),
        prompt_tokens: input_tokens,
        reused_prefix_tokens: 0,
        timings: GenerationTimings::default(),
        total: Duration::from_millis(1),
    }
}

fn http_body(bytes: &[u8]) -> &str {
    let text = std::str::from_utf8(bytes).expect("invariant: HTTP UTF-8");
    text.split_once("\r\n\r\n")
        .map(|(_, body)| body)
        .expect("invariant: séparation headers/body")
}

fn sse_events(body: &str) -> Vec<(String, Value)> {
    body.split("\n\n")
        .filter(|frame| !frame.is_empty())
        .map(|frame| {
            let mut lines = frame.lines();
            let event = lines
                .next()
                .and_then(|line| line.strip_prefix("event: "))
                .expect("invariant: ligne event SSE")
                .to_string();
            let data = lines
                .next()
                .and_then(|line| line.strip_prefix("data: "))
                .expect("invariant: ligne data SSE");
            let value = serde_json::from_str(data).expect("invariant: data JSON SSE");
            (event, value)
        })
        .collect()
}
