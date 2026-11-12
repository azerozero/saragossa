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
    assert!(req.uses_tools);
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
    response.value["output"] = output("Bonjour 🌍", None);
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
    let tool = json!({"type":"function","function":{"name":"weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}});
    let parser = FlintOutputParser::new(vec![tool], true);
    let mut bytes = Vec::new();
    let mut sink = ResponsesSse::new(&mut bytes, Response::new(&request(json!({}))), Some(parser));
    sink.event(CompletionStreamEvent::Start(&start()))
        .expect("invariant: start");
    for ch in "<tool_call>weather<arg_key>city</arg_key><arg_value>París 🌍</arg_value></tool_call>"
        .chars()
    {
        sink.event(CompletionStreamEvent::Delta(&ch.to_string()))
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
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "response.output_item.done",
            "response.completed"
        ]
    );
    assert_eq!(events[2]["item"]["arguments"], "");
    assert_eq!(events[3]["delta"], events[4]["arguments"]);
    assert_eq!(events[2]["item"]["call_id"], events[5]["item"]["call_id"]);
    assert_eq!(events[6]["response"]["output"][0], events[5]["item"]);
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
    assert_eq!(messages[1].content.as_deref(), Some("first result"));
    assert_eq!(messages[2].content.as_deref(), Some("second result"));
}
