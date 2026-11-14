use std::time::Duration;

use super::*;
use crate::serve::protocol::Usage;
use crate::serve::streaming::{StreamTerminalError, StreamingCompletionStart};

fn request(extra: Value) -> Request {
    let mut body = json!({"model":"fixture","input":"Bonjour"});
    body.as_object_mut()
        .expect("invariant: objet")
        .extend(extra.as_object().expect("invariant: options").clone());
    Request::parse(&serde_json::to_vec(&body).expect("invariant: JSON"), 512)
        .expect("invariant: requête valide")
}

fn completion(reason: &'static str) -> ServedCompletion {
    ServedCompletion {
        model: "fixture".into(),
        content: "Bonjour 🌍".into(),
        finish_reason: reason,
        matched_stop: None,
        usage: Usage::new(12, 3),
        prompt_tokens: 12,
        reused_prefix_tokens: 4,
        timings: saragossa::decoder::GenerationTimings::default(),
        total: Duration::ZERO,
    }
}

fn start() -> StreamingCompletionStart {
    StreamingCompletionStart {
        model: "fixture".into(),
        prompt_tokens: 12,
        reused_prefix_tokens: 4,
        prefill: Duration::ZERO,
    }
}

fn events(bytes: Vec<u8>) -> Vec<Value> {
    let wire = String::from_utf8(bytes).expect("invariant: UTF-8");
    assert_eq!(wire.matches("HTTP/1.1").count(), 1);
    assert!(!wire.contains("[DONE]"));
    let events: Vec<Value> = wire
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .map(|data| serde_json::from_str(data).expect("invariant: événement JSON"))
        .collect();
    for (index, event) in events.iter().enumerate() {
        assert_eq!(event["sequence_number"], index);
        assert!(wire.contains(&format!(
            "event: {}\n",
            event["type"].as_str().expect("invariant: type")
        )));
    }
    events
}

#[test]
fn converts_roles_instructions_and_text_parts() {
    let req = request(json!({"instructions":"règle", "input":[
        {"role":"developer","content":"contrainte"},
        {"role":"system","content":"système"},
        {"role":"user","content":[{"type":"input_text","text":"é"}]},
        {"type":"message","role":"assistant","content":[{"type":"output_text","text":"🌍","annotations":[]}]}
    ],"max_output_tokens":7,"temperature":0.5,"top_p":0.8,"store":false}));
    let messages = req.chat.template_messages();
    assert_eq!(messages.len(), 5);
    assert_eq!(messages[0].content.as_deref(), Some("règle"));
    assert_eq!(messages[1].role, "system");
    assert_eq!(messages[4].content.as_deref(), Some("🌍"));
    assert_eq!(req.chat.max_tokens(), 7);
}

#[test]
fn converts_function_history_and_checks_call_ids() {
    let req = request(
        json!({"tools":[{"type":"function","name":"weather","parameters":{"type":"object"},"strict":false}],
        "input":[{"type":"function_call","id":"fc_1","call_id":"call_1","name":"weather","arguments":"{\"city\":\"Paris\"}"},
        {"type":"function_call_output","call_id":"call_1","output":"soleil"}]}),
    );
    let messages = req.chat.template_messages();
    assert_eq!(messages[0].tool_calls.len(), 1);
    assert!(messages[0].tool_calls[0].contains("call_1"));
    assert_eq!(messages[1].role, "tool");
    assert_eq!(
        req.chat.active_tools().expect("invariant: outils")[0].value["function"]["name"],
        "weather"
    );
    for input in [
        json!([{"type":"function_call_output","call_id":"absent","output":"x"}]),
        json!([{"type":"function_call","call_id":"c","name":"f","arguments":"{}"}]),
    ] {
        let bytes =
            serde_json::to_vec(&json!({"model":"fixture","input":input})).expect("invariant: JSON");
        assert!(Request::parse(&bytes, 512).is_err());
    }
}

#[test]
fn rejects_unsupported_and_invalid_options_before_loading() {
    for extra in [
        json!({"previous_response_id":"r"}),
        json!({"store":true}),
        json!({"background":false}),
        json!({"include":[]}),
        json!({"tools":[{"type":"web_search"}]}),
        json!({"input":[{"role":"user","content":[{"type":"input_image","image_url":"x"}]}]}),
        json!({"tool_choice":"required"}),
        json!({"temperature":3}),
        json!({"max_output_tokens":0}),
        json!({"max_output_tokens":513}),
        json!({"input":[{"role":"bogus","content":"secret"}]}),
        json!({"reasoning":{"effort":"high"}}),
    ] {
        let mut body = json!({"model":"fixture","input":"Bonjour"});
        body.as_object_mut()
            .expect("invariant: objet")
            .extend(extra.as_object().expect("invariant: options").clone());
        assert!(
            Request::parse(&serde_json::to_vec(&body).expect("invariant: JSON"), 512).is_err(),
            "{body}"
        );
    }
}

#[test]
fn non_stream_shape_and_usage() {
    let mut response = Response::new(&request(json!({})));
    response.value["output"] = output("Bonjour 🌍", None, "stop");
    response.complete(&completion("stop"));
    assert_eq!(response.value["object"], "response");
    assert_eq!(response.value["status"], "completed");
    assert_eq!(
        response.value["output"][0]["content"][0]["text"],
        "Bonjour 🌍"
    );
    assert_eq!(response.value["usage"]["total_tokens"], 15);
    assert_eq!(
        response.value["usage"]["input_tokens_details"]["cached_tokens"],
        4
    );
}

#[test]
fn text_sse_order_identity_unicode_and_usage() {
    let mut bytes = Vec::new();
    let mut sink = ResponsesSse::new(&mut bytes, Response::new(&request(json!({}))), None);
    sink.event(CompletionStreamEvent::Start(&start()))
        .expect("invariant: start");
    for delta in ["Bon", "jour ", "🌍"] {
        sink.event(CompletionStreamEvent::Delta(delta))
            .expect("invariant: delta");
    }
    sink.finish(Ok(completion("stop"))).expect("invariant: fin");
    let events = events(bytes);
    assert_eq!(
        events
            .iter()
            .map(|v| v["type"].as_str().expect("invariant: type"))
            .collect::<Vec<_>>(),
        [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.delta",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed"
        ]
    );
    assert_eq!(events[0]["response"]["id"], events[10]["response"]["id"]);
    assert_eq!(events[4]["item_id"], events[9]["item"]["id"]);
    assert_eq!(events[7]["text"], "Bonjour 🌍");
    assert_eq!(events[10]["response"]["output"][0], events[9]["item"]);
    assert_eq!(events[10]["response"]["usage"]["output_tokens"], 3);
}

#[test]
fn function_sse_buffers_fragmented_xml_and_preserves_call_identity() {
    use crate::serve::tool_output::ToolFamily;
    for (family, text) in [
        (ToolFamily::Flint, "<tool_call>weather<arg_key>city</arg_key><arg_value>París 🌍</arg_value></tool_call>"),
        (ToolFamily::Qwen, "<tool_call><function=weather><parameter=city>París 🌍</parameter></function></tool_call>"),
        (ToolFamily::Gemma4, "<|tool_call>call:weather{city:<|\"|>París 🌍<|\"|>}<tool_call|>"),
    ] {
        let tool = json!({"type":"function","function":{"name":"weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}});
        let parser = || ToolOutputParser::new(family, vec![tool.clone()], true);
        let non_stream = output(text, Some(parser()), "stop");
        assert_eq!(non_stream.as_array().expect("invariant: output").len(), 1);
        assert_eq!(non_stream[0]["type"], "function_call");
        assert_eq!(non_stream[0]["name"], "weather");
        assert_eq!(non_stream[0]["arguments"], r#"{"city":"París 🌍"}"#);
        let mut bytes = Vec::new();
        let mut sink = ResponsesSse::new(&mut bytes, Response::new(&request(json!({}))), Some(parser()));
        sink.event(CompletionStreamEvent::Start(&start())).expect("invariant: start");
        for ch in text.chars() {
            sink.event(CompletionStreamEvent::Delta(&ch.to_string())).expect("invariant: delta");
        }
        sink.finish(Ok(completion("stop"))).expect("invariant: fin");
        let events = events(bytes);
        assert_eq!(events.iter().map(|v| v["type"].as_str().expect("invariant: type")).collect::<Vec<_>>(), [
            "response.created", "response.in_progress", "response.output_item.added",
            "response.function_call_arguments.delta", "response.function_call_arguments.done",
            "response.output_item.done", "response.completed"
        ]);
        assert_eq!(events[2]["item"]["arguments"], "");
        assert_eq!(events[3]["delta"], non_stream[0]["arguments"]);
        assert_eq!(events[3]["delta"], events[4]["arguments"]);
        assert_eq!(events[2]["item"]["call_id"], events[5]["item"]["call_id"]);
        assert_eq!(events[6]["response"]["output"][0], events[5]["item"]);
    }
}

#[test]
fn terminal_failure_is_once_and_prestart_error_stays_http() {
    let mut bytes = Vec::new();
    let mut sink = ResponsesSse::new(&mut bytes, Response::new(&request(json!({}))), None);
    assert!(sink
        .finish(Err(ServeError::Http("fixture".into())))
        .is_err());
    assert!(!sink.started);
    sink.event(CompletionStreamEvent::Start(&start()))
        .expect("invariant: start");
    sink.event(CompletionStreamEvent::TerminalError(&StreamTerminalError {
        message: "fixture",
        error_type: "server_error",
    }))
    .expect("invariant: erreur");
    sink.finish(Err(ServeError::Http("fixture".into())))
        .expect("invariant: déjà terminé");
    let events = events(bytes);
    assert_eq!(events.len(), 3);
    assert_eq!(events[2]["type"], "response.failed");
    assert_eq!(events[2]["response"]["status"], "failed");
}

#[test]
fn length_uses_official_incomplete_event() {
    let mut bytes = Vec::new();
    let mut sink = ResponsesSse::new(&mut bytes, Response::new(&request(json!({}))), None);
    sink.event(CompletionStreamEvent::Start(&start()))
        .expect("invariant: start");
    sink.finish(Ok(completion("length")))
        .expect("invariant: fin");
    let events = events(bytes);
    let last = events.last().expect("invariant: terminal");
    assert_eq!(last["type"], "response.incomplete");
    assert_eq!(
        last["response"]["incomplete_details"]["reason"],
        "max_output_tokens"
    );
}

#[test]
fn reorders_parallel_results_by_call_id_for_positional_template() {
    let req = request(json!({"input":[
        {"type":"function_call","call_id":"a","name":"first","arguments":"{}"},
        {"type":"function_call","call_id":"b","name":"second","arguments":"{}"},
        {"type":"function_call_output","call_id":"b","output":"second result"},
        {"type":"function_call_output","call_id":"a","output":"first result"},
        {"role":"user","content":"Continue"}
    ]}));
    let messages = req.chat.template_messages();
    assert_eq!(messages[0].tool_calls.len(), 2);
    assert_eq!(messages[1].tool_call_id.as_deref(), Some("a"));
    assert_eq!(messages[2].tool_call_id.as_deref(), Some("b"));
    let rendered =
        saragossa::chat_template::render_gemma4_chat_with_tools(&messages, &[], true, false);
    assert!(rendered.contains("response:first{"));
    assert!(rendered.contains("response:second{"));
    assert!(!rendered.contains("response:unknown{"));
    assert_eq!(messages[1].content.as_deref(), Some("first result"));
    assert_eq!(messages[2].content.as_deref(), Some("second result"));
}

#[test]
fn tools_are_explicitly_non_strict_in_every_response() {
    let req = request(json!({"tools":[
        {"type":"function","name":"implicit"},
        {"type":"function","name":"explicit","strict":false},
        {"type":"function","name":"nullable","strict":null}
    ]}));
    let response = Response::new(&req);
    for tool in response.value["tools"]
        .as_array()
        .expect("invariant: outils")
    {
        assert_eq!(tool["strict"], false);
    }
    for tool in req.chat.active_tools().expect("invariant: outils") {
        assert_eq!(tool.value["function"]["strict"], false);
    }
    let mut bytes = Vec::new();
    let mut sink = ResponsesSse::new(&mut bytes, response, None);
    sink.event(CompletionStreamEvent::Start(&start()))
        .expect("invariant: start");
    sink.finish(Ok(completion("stop"))).expect("invariant: fin");
    for event in events(bytes)
        .iter()
        .filter(|event| event.get("response").is_some())
    {
        for tool in event["response"]["tools"]
            .as_array()
            .expect("invariant: outils")
        {
            assert_eq!(tool["strict"], false);
        }
    }
}

#[test]
fn strict_tools_return_http_501_with_indexed_parameter() {
    let body = json!({"model":"fixture","input":"Bonjour","tools":[
        {"type":"function","name":"ordinary"},
        {"type":"function","name":"strict","strict":true}
    ]});
    let error = Request::parse(&serde_json::to_vec(&body).expect("invariant: JSON"), 512)
        .err()
        .expect("invariant: refus strict");
    // Le chemin de réplication doit aussi conserver le paramètre structuré.
    for error in [error.replicate(), error] {
        let mut bytes = Vec::new();
        assert!(super::super::http::reply_on_error(&mut bytes, Err(error)).is_err());
        let wire = String::from_utf8(bytes).expect("invariant: UTF-8");
        assert!(wire.starts_with("HTTP/1.1 501 "));
        let (_, body) = wire.split_once("\r\n\r\n").expect("invariant: corps HTTP");
        let body: Value = serde_json::from_str(body).expect("invariant: JSON");
        assert_eq!(body["error"]["param"], "tools[1].strict");
        assert!(body["error"]["message"]
            .as_str()
            .expect("invariant: message")
            .contains("génération contrainte"));
    }
}

fn tool_fixtures() -> [(crate::serve::tool_output::ToolFamily, &'static str); 3] {
    use crate::serve::tool_output::ToolFamily;
    [
        (
            ToolFamily::Flint,
            "<tool_call>weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>",
        ),
        (
            ToolFamily::Qwen,
            "<tool_call><function=weather><parameter=city>Paris</parameter></function></tool_call>",
        ),
        (
            ToolFamily::Gemma4,
            "<|tool_call>call:weather{city:<|\"|>Paris<|\"|>}<tool_call|>",
        ),
    ]
}

fn fixture_parser(family: crate::serve::tool_output::ToolFamily) -> ToolOutputParser {
    ToolOutputParser::new(
        family,
        vec![json!({"type":"function","function":{
            "name":"weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}
        }})],
        true,
    )
}

fn streamed_output(
    content: &str,
    family: crate::serve::tool_output::ToolFamily,
    reason: &'static str,
) -> Vec<Value> {
    let mut bytes = Vec::new();
    let mut sink = ResponsesSse::new(
        &mut bytes,
        Response::new(&request(json!({}))),
        Some(fixture_parser(family)),
    );
    sink.event(CompletionStreamEvent::Start(&start()))
        .expect("invariant: start");
    for ch in content.chars() {
        sink.event(CompletionStreamEvent::Delta(&ch.to_string()))
            .expect("invariant: delta");
    }
    sink.finish(Ok(completion(reason))).expect("invariant: fin");
    events(bytes)
}

fn without_ids(mut items: Value) -> Value {
    for item in items.as_array_mut().expect("invariant: output") {
        let object = item.as_object_mut().expect("invariant: item");
        object.remove("id");
        object.remove("call_id");
    }
    items
}

#[test]
fn json_and_sse_keep_interleaved_output_order_and_closed_statuses() {
    for (family, call) in tool_fixtures() {
        let second_call = call.replace("Paris", "Lyon");
        for text in [
            format!("Avant 🌍{call}Après{second_call}Fin"),
            format!("Avant{call}"),
            format!("{call}Après"),
            "Texte seul".into(),
            String::new(),
        ] {
            for reason in ["stop", "length"] {
                let events = streamed_output(&text, family, reason);
                let terminal = &events.last().expect("invariant: terminal")["response"];
                let mut response = Response::new(&request(json!({})));
                response.value["output"] = output(&text, Some(fixture_parser(family)), reason);
                response.complete(&completion(reason));
                assert_eq!(
                    without_ids(response.value["output"].clone()),
                    without_ids(terminal["output"].clone()),
                    "{family:?}, {text}, {reason}"
                );
                for event in events
                    .iter()
                    .filter(|e| e["type"] == "response.output_item.done")
                {
                    let index = event["output_index"].as_u64().expect("invariant: index") as usize;
                    assert_eq!(event["item"], terminal["output"][index]);
                }
                let items = terminal["output"].as_array().expect("invariant: items");
                for (index, item) in items.iter().enumerate() {
                    let incomplete =
                        reason == "length" && index + 1 == items.len() && item["type"] == "message";
                    assert_eq!(
                        item["status"],
                        if incomplete {
                            "incomplete"
                        } else {
                            "completed"
                        }
                    );
                }
                if text.starts_with("Avant 🌍")
                    && family != crate::serve::tool_output::ToolFamily::Flint
                {
                    assert_eq!(items.len(), 5);
                    assert_eq!(items[0]["content"][0]["text"], "Avant 🌍");
                    assert_eq!(items[1]["type"], "function_call");
                    assert_eq!(items[2]["content"][0]["text"], "Après");
                    assert_eq!(items[3]["type"], "function_call");
                    assert_eq!(items[4]["content"][0]["text"], "Fin");
                }
            }
        }
    }
    assert_eq!(output("Texte", None, "length")[0]["status"], "incomplete");
}

#[test]
fn complete_sse_history_replays_with_assistant_text_and_call_ids() {
    use saragossa::chat_template::{
        render_flint_chat, render_gemma4_chat_with_tools, render_qwen_chatml_with_tools,
        AssistantBlock,
    };
    for (family, call) in tool_fixtures()
        .into_iter()
        .filter(|(family, _)| *family != crate::serve::tool_output::ToolFamily::Flint)
    {
        let second_call = call.replace("Paris", "Lyon");
        let events = streamed_output(
            &format!("Avant{call}Milieu{second_call}Après"),
            family,
            "stop",
        );
        let output = events.last().expect("invariant: terminal")["response"]["output"]
            .as_array()
            .expect("invariant: output");
        let mut input = vec![json!({"role":"user","content":"Météo ?"})];
        input.extend(output.iter().cloned());
        // Le client rejoue tous les items SSE, puis rend les résultats à l'envers.
        input.push(json!({"type":"function_call_output","call_id":output[3]["call_id"],"output":"second résultat"}));
        input.push(json!({"type":"function_call_output","call_id":output[1]["call_id"],"output":"premier résultat"}));
        let req = request(json!({"input":input}));
        let messages = req.chat.template_messages();
        assert_eq!(messages.len(), 4);
        assert_eq!(
            messages[1].assistant_blocks,
            Some(vec![
                AssistantBlock::Text("Avant".into()),
                AssistantBlock::ToolCall(0),
                AssistantBlock::Text("Milieu".into()),
                AssistantBlock::ToolCall(1),
                AssistantBlock::Text("Après".into())
            ])
        );
        assert_eq!(
            messages[2].tool_call_id.as_deref(),
            output[1]["call_id"].as_str()
        );
        assert_eq!(
            messages[3].tool_call_id.as_deref(),
            output[3]["call_id"].as_str()
        );
        for rendered in [
            render_flint_chat(&messages, &[], true),
            render_qwen_chatml_with_tools(&messages, &[], true, false),
            render_gemma4_chat_with_tools(&messages, &[], true, false),
        ] {
            let positions: Vec<_> = [
                "Avant",
                "weather",
                "Milieu",
                "weather",
                "Après",
                "premier résultat",
                "second résultat",
            ]
            .iter()
            .scan(0, |offset, needle| {
                let found = rendered[*offset..]
                    .find(needle)
                    .expect("invariant: bloc rendu dans l'ordre");
                *offset += found + needle.len();
                Some(*offset)
            })
            .collect();
            assert_eq!(positions.len(), 7);
            assert!(!rendered.contains("response:unknown{"));
        }
    }
}

#[test]
fn intermediate_assistant_keeps_history_validation() {
    let call = json!({"type":"function_call","call_id":"c","name":"weather","arguments":"{}"});
    let assistant = json!({"role":"assistant","content":"En attente"});
    let result = json!({"type":"function_call_output","call_id":"c","output":"soleil"});
    for input in [
        json!([call, assistant]),
        json!([call, assistant, {"role":"user","content":"suite"}, result]),
        json!([call, assistant, {"type":"function_call_output","call_id":"autre","output":"x"}]),
        json!([call, assistant, result, result]),
        json!([call, assistant, call, result]),
    ] {
        let body =
            serde_json::to_vec(&json!({"model":"fixture","input":input})).expect("invariant: JSON");
        assert!(Request::parse(&body, 512).is_err());
    }
}
