use std::io::Write;
use std::time::Duration;

use saragossa::decoder::GenerationTimings;
use serde_json::{json, Value};

use super::*;

#[test]
fn request_ignores_logit_bias() {
    for bias in [json!({"42": -100}), json!({}), json!(null), json!(false)] {
        let body = json!({
            "model": "m", "max_tokens": 1, "messages": [], "logit_bias": bias
        });
        let (chat, _, _) = anthropic_chat(body.to_string().as_bytes(), None)
            .expect("invariant: requête Anthropic valide");
        assert!(chat.logit_bias.is_none());
        assert_eq!(chat.validate(1).expect("invariant: champ ignoré"), 1);
    }
}

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
    assert_eq!(messages[2].content.as_deref(), Some("ok"));
    assert_eq!(messages[2].tool_call_id.as_deref(), Some("toolu_1"));
    assert_eq!(messages[3].role, "assistant");
    assert_eq!(messages[3].content.as_deref(), Some("done"));
    let call: Value = serde_json::from_str(&messages[3].tool_calls[0]).expect("call");
    assert_eq!(call["id"], "toolu_2");
    assert_eq!(call["function"]["name"], "Bash");
    assert_eq!(call["function"]["arguments"], json!({"command":"pwd"}));
}

#[test]
fn request_preserves_tools_without_injecting_a_generic_prompt_or_stop() {
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

    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].role, "user");
    assert_eq!(messages[0].content.as_deref(), Some("liste"));
    let tools = chat.active_tools().expect("tools");
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].value["function"]["name"], "Bash");
    assert_eq!(
        tools[0].value["function"]["description"],
        "Run a shell command"
    );
    assert_eq!(
        tools[0].value["function"]["parameters"]["properties"]["command"]["type"],
        "string"
    );
    assert!(chat.stop_texts().is_empty());
}

#[test]
fn request_uses_metadata_user_id_as_session_fallback() {
    let request: AnthropicMessagesRequest = serde_json::from_str(
        r#"{
            "model":"reti-35b",
            "max_tokens":64,
            "metadata":{"user_id":" agent-a "},
            "messages":[{"role":"user","content":"salut"}]
        }"#,
    )
    .expect("invariant: JSON Anthropic valide");

    assert_eq!(request.session_key(None).as_deref(), Some("agent-a"));
    assert_eq!(request.session_key(Some("   ")).as_deref(), Some("agent-a"));
}

#[test]
fn request_session_header_overrides_metadata_user_id() {
    let request: AnthropicMessagesRequest = serde_json::from_str(
        r#"{
            "model":"reti-35b",
            "max_tokens":64,
            "metadata":{"user_id":"agent-a"},
            "messages":[{"role":"user","content":"salut"}]
        }"#,
    )
    .expect("invariant: JSON Anthropic valide");

    assert_eq!(
        request.session_key(Some(" agent-b ")).as_deref(),
        Some("agent-b")
    );
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
fn live_blocks_stream_text_deltas() {
    let completion = fake_completion("reti-35b", "salut", "stop", None, 5, 2);
    let mut blocks = AnthropicLiveBlocks::new();
    let mut stream = Vec::new();

    blocks
        .push_text_delta(&mut stream, "sal")
        .expect("invariant: delta texte sérialisable");
    blocks
        .push_text_delta(&mut stream, "ut")
        .expect("invariant: delta texte sérialisable");
    let has_tool_use = blocks
        .finish(&mut stream, &completion)
        .expect("invariant: fin bloc texte sérialisable");

    assert!(!has_tool_use);
    let events = sse_events(http_bodyless(&stream));
    assert_eq!(
        events
            .iter()
            .map(|(event, _)| event.as_str())
            .collect::<Vec<_>>(),
        vec![
            "content_block_start",
            "content_block_delta",
            "content_block_delta",
            "content_block_stop"
        ]
    );
    assert_eq!(events[1].1["delta"]["text"], "sal");
    assert_eq!(events[2].1["delta"]["text"], "ut");
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

fn send_anthropic_sse<S: Write>(stream: &mut S, completion: &ServedCompletion) -> ServeResult<()> {
    let output = AssistantOutput::literal(&completion.content);
    write_headers(
        stream,
        200,
        "OK",
        "text/event-stream; charset=utf-8",
        None,
        completion.metric_headers(),
    )?;
    let message_id = response_id();
    write_anthropic_sse_event(
        stream,
        "message_start",
        &json!({
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": completion.model.as_str(),
                "stop_reason": null,
                "stop_sequence": null,
                "usage": {
                    "input_tokens": completion.usage.prompt_tokens(),
                    "output_tokens": 0
                }
            }
        }),
    )?;
    for (index, block) in output.blocks.iter().enumerate() {
        write_anthropic_content_block(stream, index, block)?;
    }
    write_anthropic_sse_event(
        stream,
        "message_delta",
        &json!({
            "type": "message_delta",
            "delta": {
                "stop_reason": anthropic_stop_reason(completion, output.has_tool_use),
                "stop_sequence": anthropic_stop_sequence(completion, output.has_tool_use)
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

fn http_body(bytes: &[u8]) -> &str {
    let text = std::str::from_utf8(bytes).expect("invariant: HTTP UTF-8");
    text.split_once("\r\n\r\n")
        .map(|(_, body)| body)
        .expect("invariant: séparation headers/body")
}

fn http_bodyless(bytes: &[u8]) -> &str {
    std::str::from_utf8(bytes).expect("invariant: SSE UTF-8")
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

#[test]
fn common_qwen_and_gemma_parser_produce_anthropic_blocks_and_sse() {
    use super::super::tool_output::ToolFamily;
    for (family, text) in [
        (ToolFamily::Qwen, "Texte<tool_call><function=weather><parameter=city>Paris</parameter></function></tool_call>"),
        (ToolFamily::Gemma4, "Texte<|tool_call>call:weather{city:<|\"|>Paris<|\"|>}<tool_call|>"),
    ] {
        let tools = vec![json!({"type":"function","function":{"name":"weather","parameters":{"properties":{"city":{"type":"string"}}}}})];
        let completion = fake_completion("test", text, "stop", None, 3, 4);
        let response = AnthropicMessageResponse::with_parser(&completion, Some(ToolOutputParser::new(family, tools.clone(), true)));
        let value = serde_json::to_value(response).expect("response");
        assert_eq!(value["stop_reason"], "tool_use");
        assert_eq!(value["content"][0]["text"], "Texte");
        assert_eq!(value["content"][1]["input"], json!({"city":"Paris"}));
        let mut bytes = Vec::new();
        let mut sink = AnthropicSse::with_parser(&mut bytes, Some(ToolOutputParser::new(family, tools, true)));
        for ch in text.chars() { sink.event(CompletionStreamEvent::Delta(&ch.to_string())).expect("delta"); }
        sink.finish(Ok(completion)).expect("finish");
        let events = sse_events(std::str::from_utf8(&bytes).expect("utf8"));
        let tool = events.iter().find(|(_, v)| v["content_block"]["type"] == "tool_use").expect("tool block");
        let id = tool.1["content_block"]["id"].as_str().expect("id");
        assert!(!id.is_empty());
        assert_eq!(tool.1["content_block"]["name"], "weather");
        assert!(events.iter().any(|(_, v)| v["delta"]["partial_json"] == r#"{"city":"Paris"}"#));
        assert!(events.iter().any(|(_, v)| v["delta"]["stop_reason"] == "tool_use"));
        // L'identifiant émis se retrouve intact dans l'historique du tour suivant.
        let request = json!({"model":"test","max_tokens":5,"messages":[{"role":"assistant","content":[{"type":"tool_use","id":id,"name":"weather","input":{"city":"Paris"}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":id,"content":"18"}]}]});
        let (chat, _, _) = anthropic_chat(&serde_json::to_vec(&request).expect("wire"), None).expect("convert");
        assert_eq!(chat.messages[1].tool_call_id.as_deref(), Some(id));
        let call: Value = serde_json::from_str(chat.messages[0].tool_calls.as_ref().expect("calls")[0].get()).expect("call");
        assert_eq!(call["id"], id);
    }
}

#[test]
fn anthropic_tool_choices_are_explicit() {
    assert_eq!(
        anthropic_tool_choice(Some(&json!({"type":"none"}))).expect("none"),
        Some(json!("none"))
    );
    assert!(anthropic_tool_choice(Some(&json!({"type":"any"}))).is_err());
    assert!(anthropic_tool_choice(Some(&json!({"type":"tool","name":"weather"}))).is_err());
    assert!(assistant_message(Some(&AnthropicContent::Blocks(vec![
        json!({"type":"tool_use","name":"x","input":{}})
    ])))
    .is_err());
    assert!(tool_result_text(&json!({"type":"tool_result","content":"18"})).is_err());
}

fn streamed_blocks(bytes: &[u8]) -> (Value, Value) {
    let mut blocks: Vec<Value> = Vec::new();
    let mut inputs: Vec<String> = Vec::new();
    let mut stop = Value::Null;
    for (kind, value) in sse_events(http_bodyless(bytes)) {
        let index = value["index"].as_u64().unwrap_or(0) as usize;
        match kind.as_str() {
            "content_block_start" => {
                assert_eq!(index, blocks.len());
                blocks.push(value["content_block"].clone());
                inputs.push(String::new());
            }
            "content_block_delta" if value["delta"]["type"] == "text_delta" => {
                let previous = blocks[index]["text"].as_str().expect("text").to_owned();
                blocks[index]["text"] =
                    json!(previous + value["delta"]["text"].as_str().expect("delta"));
            }
            "content_block_delta" => {
                inputs[index].push_str(value["delta"]["partial_json"].as_str().expect("input"))
            }
            "content_block_stop" if blocks[index]["type"] == "tool_use" => {
                blocks[index]["input"] =
                    serde_json::from_str(&inputs[index]).expect("complete JSON");
            }
            "message_delta" => stop = value["delta"]["stop_reason"].clone(),
            _ => {}
        }
    }
    (json!(blocks), stop)
}

#[test]
fn absent_parser_never_interprets_tools_in_either_transport() {
    for text in [
        r#"<tool_call>{"name":"weather","arguments":{}}"#,
        r#"Avant<tool_call>{"name":"weather","arguments":{}}</tool_call>Après"#,
        "<tool_call><function=weather></function></tool_call>",
        "<|tool_call>call:weather{}<tool_call|>",
        "<think>literal</think>texte",
    ] {
        let completion = fake_completion("test", text, "stop", None, 1, 2);
        let response =
            serde_json::to_value(AnthropicMessageResponse::with_parser(&completion, None))
                .expect("response");
        assert_eq!(response["content"], json!([{"type":"text","text":text}]));
        assert_eq!(response["stop_reason"], "end_turn");
        for split in text.char_indices().map(|(i, _)| i).chain([text.len()]) {
            let mut bytes = Vec::new();
            let mut sink = AnthropicSse::with_parser(&mut bytes, None);
            sink.event(CompletionStreamEvent::Delta(&text[..split]))
                .expect("delta");
            sink.event(CompletionStreamEvent::Delta(&text[split..]))
                .expect("delta");
            sink.finish(Ok(fake_completion("test", text, "stop", None, 1, 2)))
                .expect("finish");
            let (blocks, stop) = streamed_blocks(&bytes);
            assert_eq!(blocks, response["content"]);
            assert_eq!(stop, response["stop_reason"]);
        }
    }
}

#[test]
fn anthropic_preserves_interleaved_text_and_calls_in_both_transports() {
    use super::super::tool_output::ToolFamily;
    for (family, call) in [
        (
            ToolFamily::Qwen,
            "<tool_call><function=weather><parameter=city>Paris</parameter></function></tool_call>",
        ),
        (
            ToolFamily::Gemma4,
            "<|tool_call>call:weather{city:<|\"|>Paris<|\"|>}<tool_call|>",
        ),
    ] {
        let tools = vec![
            json!({"function":{"name":"weather","parameters":{"properties":{"city":{"type":"string"}}}}}),
        ];
        let text = format!("Avant {call} Entre {call} Après");
        let completion = fake_completion("test", &text, "stop", None, 1, 2);
        let response = serde_json::to_value(AnthropicMessageResponse::with_parser(
            &completion,
            Some(ToolOutputParser::new(family, tools.clone(), true)),
        ))
        .expect("response");
        assert_eq!(response["content"].as_array().expect("blocks").len(), 5);
        assert_eq!(response["content"][0]["text"], "Avant ");
        assert_eq!(response["content"][2]["text"], " Entre ");
        assert_eq!(response["content"][4]["text"], " Après");
        for split in text.char_indices().map(|(i, _)| i).chain([text.len()]) {
            let mut bytes = Vec::new();
            let mut sink = AnthropicSse::with_parser(
                &mut bytes,
                Some(ToolOutputParser::new(family, tools.clone(), true)),
            );
            sink.event(CompletionStreamEvent::Delta(&text[..split]))
                .expect("delta");
            sink.event(CompletionStreamEvent::Delta(&text[split..]))
                .expect("delta");
            sink.finish(Ok(fake_completion("test", &text, "stop", None, 1, 2)))
                .expect("finish");
            let (mut blocks, stop) = streamed_blocks(&bytes);
            for index in [1, 3] {
                assert!(!blocks[index]["id"].as_str().expect("id").is_empty());
                blocks[index]["id"] = response["content"][index]["id"].clone();
            }
            assert_eq!(blocks, response["content"], "split {split}");
            assert_eq!(stop, "tool_use");
        }
    }
}

#[test]
fn none_absent_tools_and_unsupported_families_disable_api_interpretation() {
    use super::super::state::output_parser_for_config;
    let text = r#"<tool_call>{"name":"weather","arguments":{}}"#;
    for family in ["qwen3_5", "qwen3_6", "qwen3_8", "gemma4", "gemma3", "llama"] {
        let config: saragossa::ModelConfig = serde_json::from_value(json!({"model_type":family,"hidden_size":1,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,"rms_norm_eps":0.000001,"rope_theta":10000.0,"vocab_size":1})).expect("config");
        for tools in [
            json!([]),
            json!([{"name":"weather","input_schema":{"type":"object"}}]),
        ] {
            let request = json!({"model":"test","max_tokens":10,"messages":[{"role":"user","content":"exemple"}],"tools":tools,"tool_choice":{"type":if tools.as_array().expect("tools").is_empty() {"auto"} else {"none"}}});
            let (chat, _, _) = anthropic_chat(&serde_json::to_vec(&request).expect("wire"), None)
                .expect("convert");
            let completion = fake_completion("test", text, "stop", None, 1, 2);
            let response = serde_json::to_value(AnthropicMessageResponse::with_parser(
                &completion,
                output_parser_for_config(&config, &chat).expect("parser"),
            ))
            .expect("response");
            assert_eq!(response["content"], json!([{"type":"text","text":text}]));
            let mut bytes = Vec::new();
            let mut sink = AnthropicSse::with_parser(
                &mut bytes,
                output_parser_for_config(&config, &chat).expect("parser"),
            );
            for ch in text.chars() {
                sink.event(CompletionStreamEvent::Delta(&ch.to_string()))
                    .expect("delta");
            }
            sink.finish(Ok(completion)).expect("finish");
            let (blocks, stop) = streamed_blocks(&bytes);
            assert_eq!(blocks, response["content"], "{family}");
            assert_eq!(stop, "end_turn");
        }
    }
}

#[test]
fn empty_visible_output_has_the_same_block_in_both_transports() {
    use super::super::tool_output::ToolFamily;
    for text in ["", "<think>raisonnement</think>"] {
        let tools = vec![json!({"function":{"name":"weather"}})];
        let completion = fake_completion("test", text, "stop", None, 1, 2);
        let response = serde_json::to_value(AnthropicMessageResponse::with_parser(
            &completion,
            Some(ToolOutputParser::new(ToolFamily::Qwen, tools.clone(), true)),
        ))
        .expect("response");
        let mut bytes = Vec::new();
        let mut sink = AnthropicSse::with_parser(
            &mut bytes,
            Some(ToolOutputParser::new(ToolFamily::Qwen, tools, true)),
        );
        sink.event(CompletionStreamEvent::Delta(text))
            .expect("delta");
        sink.finish(Ok(completion)).expect("finish");
        let (blocks, stop) = streamed_blocks(&bytes);
        assert_eq!(blocks, response["content"]);
        assert_eq!(stop, response["stop_reason"]);
    }
}

#[test]
fn complete_anthropic_response_round_trips_interleaved_history() {
    use super::super::tool_output::ToolFamily;
    use saragossa::chat_template::{
        render_flint_chat, render_gemma4_chat_with_tools, render_qwen_chatml_profile, QwenProfile,
    };
    for (family, call) in [
        (
            ToolFamily::Qwen,
            "<tool_call><function=weather><parameter=city>Paris</parameter></function></tool_call>",
        ),
        (
            ToolFamily::Gemma4,
            "<|tool_call>call:weather{city:<|\"|>Paris<|\"|>}<tool_call|>",
        ),
    ] {
        let tools = vec![
            json!({"function":{"name":"weather","parameters":{"properties":{"city":{"type":"string"}}}}}),
        ];
        let text = format!("Avant {call} Entre {call} Après");
        let response = serde_json::to_value(AnthropicMessageResponse::with_parser(
            &fake_completion("test", &text, "stop", None, 1, 2),
            Some(ToolOutputParser::new(family, tools, true)),
        ))
        .expect("response");
        assert_eq!(response["content"].as_array().expect("blocks").len(), 5);
        let request = json!({"model":"test", "max_tokens":10, "messages":[
            {"role":"user","content":"Météo ?"},
            response,
            {"role":"user","content":[
                {"type":"tool_result","tool_use_id":response["content"][3]["id"],"content":"18"},
                {"type":"tool_result","tool_use_id":response["content"][1]["id"],"content":"12"}
            ]}
        ]});
        let (chat, _, _) =
            anthropic_chat(&serde_json::to_vec(&request).expect("wire"), None).expect("convert");
        let messages = chat.template_messages();
        let rendered = match family {
            ToolFamily::Qwen => {
                render_qwen_chatml_profile(&messages, &[], true, false, QwenProfile::Qwen36)
            }
            ToolFamily::Gemma4 => render_gemma4_chat_with_tools(&messages, &[], true, false),
            ToolFamily::Flint => render_flint_chat(&messages, &[], true),
        };
        let open = if family == ToolFamily::Gemma4 {
            "<|tool_call>"
        } else {
            "<tool_call>"
        };
        let flint = render_flint_chat(&messages, &[], false);
        assert!(flint.find("Avant ").expect("before") < flint.find("<tool_call>").expect("call"));
        assert!(flint.find("<tool_call>").expect("call") < flint.find(" Entre ").expect("between"));
        assert!(
            flint.find(" Entre ").expect("between") < flint.rfind("<tool_call>").expect("call")
        );
        assert!(flint.rfind("<tool_call>").expect("call") < flint.find(" Après").expect("after"));
        let positions = [
            rendered.find("Avant ").expect("before"),
            rendered.find(open).expect("first call"),
            rendered.find(" Entre ").expect("between"),
            rendered.rfind(open).expect("second call"),
            rendered.find(" Après").expect("after"),
        ];
        assert!(positions.windows(2).all(|p| p[0] < p[1]), "{rendered}");
        assert_eq!(rendered.matches(open).count(), 2);
        assert_eq!(
            rendered
                .matches(if family == ToolFamily::Qwen {
                    "<|im_start|>assistant"
                } else if family == ToolFamily::Gemma4 {
                    "<|turn>model"
                } else {
                    "<|assistant|>"
                })
                .count(),
            if family == ToolFamily::Gemma4 { 1 } else { 2 }
        );
    }
}
