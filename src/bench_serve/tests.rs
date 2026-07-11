use std::collections::BTreeMap;
use std::time::Duration;

use serde_json::Value;

use super::*;

fn socket_args() -> BenchServeArgs {
    BenchServeArgs::parse([
        "--socket".to_string(),
        "/tmp/test.sock".to_string(),
        "--model".to_string(),
        "m".to_string(),
    ])
    .expect("invariant: args bench valides")
}

#[test]
fn build_chat_request_sets_stream_body_and_bearer() {
    let mut args = socket_args();
    args.api_key = Some("secret".to_string());

    let request = build_chat_request(&args, 3).expect("invariant: requete construite");
    let text = String::from_utf8(request).expect("invariant: HTTP UTF-8");
    let (head, body) = text
        .split_once("\r\n\r\n")
        .expect("invariant: separateur headers present");

    assert!(head.contains("POST /v1/chat/completions HTTP/1.1"));
    assert!(head.contains("Authorization: Bearer secret"));
    assert!(head.contains("Accept: text/event-stream"));
    let value: Value = serde_json::from_str(body).expect("invariant: body JSON");
    assert_eq!(value["model"], "m");
    assert_eq!(value["stream"], true);
    assert_eq!(value["temperature"], 0.0);
    assert_eq!(value["max_tokens"], DEFAULT_MAX_TOKENS);
    assert!(value["messages"][0]["content"]
        .as_str()
        .is_some_and(|prompt| prompt.contains("Bench request 3.")));
}

#[test]
fn aggregates_compute_interpolated_p50_p95_and_mean() {
    let values = [10.0, 20.0, 30.0, 40.0];

    assert_eq!(percentile(&values, 0.50), Some(25.0));
    assert_eq!(percentile(&values, 0.95), Some(38.5));
    assert_eq!(mean(&values), Some(25.0));
}

#[test]
fn summary_uses_successful_requests_only() {
    let reports = vec![
        RequestReport {
            index: 0,
            worker_count: 2,
            ok: true,
            status: Some(200),
            ttft_ms: Some(10.0),
            decode_tok_s: Some(100.0),
            total_ms: 50.0,
            generated_tokens: 10,
            x_saragossa: BTreeMap::new(),
            error: None,
        },
        RequestReport {
            index: 1,
            worker_count: 2,
            ok: false,
            status: Some(500),
            ttft_ms: None,
            decode_tok_s: None,
            total_ms: 20.0,
            generated_tokens: 0,
            x_saragossa: BTreeMap::new(),
            error: Some("HTTP 500".to_string()),
        },
    ];

    let summary = BenchSummary::from_results(&reports, Duration::from_millis(70));

    assert_eq!(summary.requested, 2);
    assert_eq!(summary.succeeded, 1);
    assert_eq!(summary.failed, 1);
    assert_eq!(summary.concurrency, 2);
    assert_eq!(summary.ttft_p50_ms, 10.0);
    assert_eq!(summary.decode_tok_s_mean, 100.0);
    assert_eq!(summary.total_ms, 70.0);
}
