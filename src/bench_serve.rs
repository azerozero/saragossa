//! Client de bench pour un `saragossa serve` deja lance.

mod sse;
#[cfg(test)]
mod tests;

use std::collections::BTreeMap;
use std::env;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::str;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[cfg(unix)]
use std::os::unix::net::UnixStream;

use serde::Serialize;

use super::CliResult;
use sse::SseParser;

const DEFAULT_SOCKET: &str = "/tmp/saragossa-serve.sock";
const DEFAULT_MODEL: &str = "reti-35b";
const DEFAULT_REQUESTS: usize = 8;
const DEFAULT_CONCURRENCY: usize = 1;
const DEFAULT_PROMPT_TOKENS: usize = 256;
const DEFAULT_MAX_TOKENS: usize = 128;
const DEFAULT_TIMEOUT_SECS: u64 = 600;
const API_KEY_ENV: &str = "SARAGOSSA_API_KEY";

/// Lance le bench HTTP/SSE local.
pub(super) fn run(args: impl IntoIterator<Item = String>) -> CliResult<()> {
    let raw_args = args.into_iter().collect::<Vec<_>>();
    if raw_args
        .iter()
        .any(|arg| matches!(arg.as_str(), "--help" | "-h"))
    {
        println!("{}", help_text());
        return Ok(());
    }
    let args = BenchServeArgs::parse(raw_args).map_err(boxed_error)?;
    let started = Instant::now();
    let results = run_requests(&args).map_err(boxed_error)?;
    let total = started.elapsed();
    let timestamp = unix_timestamp_secs();
    let summary = BenchSummary::from_results(&results, total);
    let report = BenchReport::new(&args, summary, results, timestamp);
    let report_path = write_report(&report).map_err(boxed_error)?;
    print_summary(&report.summary, &report_path);
    if report.summary.succeeded == 0 {
        return Err(boxed_error(BenchServeError::new(
            "aucune requete reussie pendant le bench",
        )));
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct BenchServeArgs {
    target: BenchTarget,
    api_key: Option<String>,
    model: String,
    requests: usize,
    concurrency: usize,
    prompt_tokens: usize,
    max_tokens: usize,
    timeout: Duration,
}

impl BenchServeArgs {
    fn parse(args: impl IntoIterator<Item = String>) -> BenchResult<Self> {
        let mut socket = PathBuf::from(DEFAULT_SOCKET);
        let mut socket_seen = false;
        let mut url = None;
        let mut api_key = env::var(API_KEY_ENV).ok().filter(|value| !value.is_empty());
        let mut model = DEFAULT_MODEL.to_string();
        let mut requests = DEFAULT_REQUESTS;
        let mut concurrency = DEFAULT_CONCURRENCY;
        let mut prompt_tokens = DEFAULT_PROMPT_TOKENS;
        let mut max_tokens = DEFAULT_MAX_TOKENS;
        let mut timeout = Duration::from_secs(DEFAULT_TIMEOUT_SECS);
        let mut iter = args.into_iter();

        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--socket" => {
                    if url.is_some() {
                        return Err(BenchServeError::new(
                            "--socket et --url sont mutuellement exclusifs",
                        ));
                    }
                    socket = next_value(&mut iter, "--socket")?.into();
                    socket_seen = true;
                }
                "--url" => {
                    if socket_seen {
                        return Err(BenchServeError::new(
                            "--socket et --url sont mutuellement exclusifs",
                        ));
                    }
                    url = Some(normalize_tcp_addr(&next_value(&mut iter, "--url")?)?);
                }
                "--api-key" | "--bearer-token" => {
                    api_key = Some(validate_api_key(&next_value(&mut iter, "--api-key")?)?);
                }
                "--model" => model = next_value(&mut iter, "--model")?,
                "--requests" => requests = parse_positive_usize("--requests", &mut iter)?,
                "--concurrency" => {
                    concurrency = parse_positive_usize("--concurrency", &mut iter)?;
                }
                "--prompt-tokens" => {
                    prompt_tokens = parse_positive_usize("--prompt-tokens", &mut iter)?;
                }
                "--max-tokens" => max_tokens = parse_positive_usize("--max-tokens", &mut iter)?,
                "--timeout-secs" => {
                    timeout = Duration::from_secs(parse_positive_u64("--timeout-secs", &mut iter)?);
                }
                other => {
                    return Err(BenchServeError::new(format!(
                        "argument bench-serve inconnu: {other}"
                    )));
                }
            }
        }

        if model.trim().is_empty() {
            return Err(BenchServeError::new("--model ne peut pas etre vide"));
        }
        let api_key = match api_key {
            Some(value) => Some(validate_api_key(&value)?),
            None => None,
        };
        let target = if let Some(addr) = url {
            if api_key.is_none() {
                return Err(BenchServeError::new(format!(
                    "--url requiert --api-key TOKEN ou {API_KEY_ENV}=TOKEN"
                )));
            }
            BenchTarget::Tcp(addr)
        } else {
            BenchTarget::Unix(socket)
        };
        Ok(Self {
            target,
            api_key,
            model,
            requests,
            concurrency,
            prompt_tokens,
            max_tokens,
            timeout,
        })
    }

    fn target_label(&self) -> String {
        match &self.target {
            BenchTarget::Tcp(addr) => format!("tcp://{addr}"),
            BenchTarget::Unix(path) => format!("unix://{}", path.display()),
        }
    }
}

#[derive(Clone, Debug)]
enum BenchTarget {
    Tcp(String),
    Unix(PathBuf),
}

trait BenchStream: Read + Write {}

impl<T: Read + Write> BenchStream for T {}

#[derive(Debug)]
struct BenchServeError {
    message: String,
    status: Option<u16>,
}

impl BenchServeError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            status: None,
        }
    }

    fn with_status(status: u16, message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            status: Some(status),
        }
    }
}

impl Display for BenchServeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl Error for BenchServeError {}

type BenchResult<T> = std::result::Result<T, BenchServeError>;

#[derive(Debug, Serialize)]
struct BenchReport {
    kind: &'static str,
    timestamp: u64,
    target: String,
    model: String,
    requests: usize,
    concurrency: usize,
    prompt_tokens: usize,
    max_tokens: usize,
    summary: BenchSummary,
    results: Vec<RequestReport>,
}

impl BenchReport {
    fn new(
        args: &BenchServeArgs,
        summary: BenchSummary,
        results: Vec<RequestReport>,
        timestamp: u64,
    ) -> Self {
        Self {
            kind: "saragossa.bench_serve",
            timestamp,
            target: args.target_label(),
            model: args.model.clone(),
            requests: args.requests,
            concurrency: args.concurrency,
            prompt_tokens: args.prompt_tokens,
            max_tokens: args.max_tokens,
            summary,
            results,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
struct BenchSummary {
    requested: usize,
    succeeded: usize,
    failed: usize,
    concurrency: usize,
    ttft_p50_ms: f64,
    ttft_p95_ms: f64,
    decode_tok_s_mean: f64,
    total_ms: f64,
}

impl BenchSummary {
    fn from_results(results: &[RequestReport], total: Duration) -> Self {
        let successes = results
            .iter()
            .filter(|result| result.ok)
            .collect::<Vec<_>>();
        let ttfts = successes
            .iter()
            .filter_map(|result| result.ttft_ms)
            .collect::<Vec<_>>();
        let decode_rates = successes
            .iter()
            .filter_map(|result| result.decode_tok_s)
            .collect::<Vec<_>>();
        let concurrency = results
            .iter()
            .map(|result| result.worker_count)
            .max()
            .unwrap_or(0);
        Self {
            requested: results.len(),
            succeeded: successes.len(),
            failed: results.len().saturating_sub(successes.len()),
            concurrency,
            ttft_p50_ms: percentile(&ttfts, 0.50).unwrap_or(0.0),
            ttft_p95_ms: percentile(&ttfts, 0.95).unwrap_or(0.0),
            decode_tok_s_mean: mean(&decode_rates).unwrap_or(0.0),
            total_ms: duration_ms(total),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
struct RequestReport {
    index: usize,
    worker_count: usize,
    ok: bool,
    status: Option<u16>,
    ttft_ms: Option<f64>,
    decode_tok_s: Option<f64>,
    total_ms: f64,
    generated_tokens: usize,
    x_saragossa: BTreeMap<String, String>,
    error: Option<String>,
}

impl RequestReport {
    fn failure(index: usize, worker_count: usize, total: Duration, error: BenchServeError) -> Self {
        Self {
            index,
            worker_count,
            ok: false,
            status: error.status,
            ttft_ms: None,
            decode_tok_s: None,
            total_ms: duration_ms(total),
            generated_tokens: 0,
            x_saragossa: BTreeMap::new(),
            error: Some(error.to_string()),
        }
    }
}

struct ResponseMetrics {
    status: u16,
    ttft_ms: f64,
    decode_tok_s: f64,
    total_ms: f64,
    generated_tokens: usize,
    x_saragossa: BTreeMap<String, String>,
}

fn run_requests(args: &BenchServeArgs) -> BenchResult<Vec<RequestReport>> {
    let worker_count = args.concurrency.min(args.requests);
    let args = Arc::new(args.clone());
    let next = Arc::new(AtomicUsize::new(0));
    let mut handles = Vec::with_capacity(worker_count);
    for _ in 0..worker_count {
        let args = Arc::clone(&args);
        let next = Arc::clone(&next);
        handles.push(thread::spawn(move || {
            let mut reports = Vec::new();
            loop {
                let index = next.fetch_add(1, Ordering::SeqCst);
                if index >= args.requests {
                    break;
                }
                reports.push(run_one_request(&args, index, worker_count));
            }
            reports
        }));
    }

    let mut results = Vec::with_capacity(args.requests);
    for handle in handles {
        let mut reports = handle
            .join()
            .map_err(|_| BenchServeError::new("thread bench interrompu"))?;
        results.append(&mut reports);
    }
    results.sort_by_key(|result| result.index);
    Ok(results)
}

fn run_one_request(args: &BenchServeArgs, index: usize, worker_count: usize) -> RequestReport {
    let started = Instant::now();
    match execute_request(args, index, started) {
        Ok(metrics) => RequestReport {
            index,
            worker_count,
            ok: true,
            status: Some(metrics.status),
            ttft_ms: Some(metrics.ttft_ms),
            decode_tok_s: Some(metrics.decode_tok_s),
            total_ms: metrics.total_ms,
            generated_tokens: metrics.generated_tokens,
            x_saragossa: metrics.x_saragossa,
            error: None,
        },
        Err(error) => RequestReport::failure(index, worker_count, started.elapsed(), error),
    }
}

fn execute_request(
    args: &BenchServeArgs,
    index: usize,
    started: Instant,
) -> BenchResult<ResponseMetrics> {
    let mut stream = open_connection(&args.target, args.timeout)?;
    let request = build_chat_request(args, index)?;
    stream
        .write_all(&request)
        .map_err(|e| BenchServeError::new(format!("ecriture requete HTTP: {e}")))?;
    stream
        .flush()
        .map_err(|e| BenchServeError::new(format!("flush requete HTTP: {e}")))?;
    read_sse_response(&mut stream, started)
}

fn open_connection(target: &BenchTarget, timeout: Duration) -> BenchResult<Box<dyn BenchStream>> {
    match target {
        BenchTarget::Tcp(addr) => {
            let stream = TcpStream::connect(addr)
                .map_err(|e| BenchServeError::new(format!("connexion TCP {addr}: {e}")))?;
            stream
                .set_read_timeout(Some(timeout))
                .map_err(|e| BenchServeError::new(format!("timeout lecture TCP: {e}")))?;
            stream
                .set_write_timeout(Some(timeout))
                .map_err(|e| BenchServeError::new(format!("timeout ecriture TCP: {e}")))?;
            Ok(Box::new(stream))
        }
        BenchTarget::Unix(path) => open_unix_connection(path, timeout),
    }
}

#[cfg(unix)]
fn open_unix_connection(path: &Path, timeout: Duration) -> BenchResult<Box<dyn BenchStream>> {
    let stream = UnixStream::connect(path)
        .map_err(|e| BenchServeError::new(format!("connexion socket {}: {e}", path.display())))?;
    stream
        .set_read_timeout(Some(timeout))
        .map_err(|e| BenchServeError::new(format!("timeout lecture socket: {e}")))?;
    stream
        .set_write_timeout(Some(timeout))
        .map_err(|e| BenchServeError::new(format!("timeout ecriture socket: {e}")))?;
    Ok(Box::new(stream))
}

#[cfg(not(unix))]
fn open_unix_connection(_path: &Path, _timeout: Duration) -> BenchResult<Box<dyn BenchStream>> {
    Err(BenchServeError::new(
        "transport socket Unix indisponible sur cette plateforme",
    ))
}

fn build_chat_request(args: &BenchServeArgs, index: usize) -> BenchResult<Vec<u8>> {
    let body = serde_json::json!({
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": synthetic_prompt(args.prompt_tokens, index),
            }
        ],
        "stream": true,
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
    });
    let body = serde_json::to_vec(&body)
        .map_err(|e| BenchServeError::new(format!("serialisation JSON requete: {e}")))?;
    let mut request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\n\
         Host: saragossa\r\n\
         Content-Type: application/json\r\n\
         Accept: text/event-stream\r\n\
         Connection: close\r\n\
         Content-Length: {}\r\n",
        body.len()
    )
    .into_bytes();
    if let Some(api_key) = args.api_key.as_deref() {
        request.extend_from_slice(b"Authorization: Bearer ");
        request.extend_from_slice(api_key.as_bytes());
        request.extend_from_slice(b"\r\n");
    }
    request.extend_from_slice(b"\r\n");
    request.extend_from_slice(&body);
    Ok(request)
}

fn read_sse_response<S: Read + ?Sized>(
    stream: &mut S,
    started: Instant,
) -> BenchResult<ResponseMetrics> {
    let mut buffer = Vec::new();
    let mut chunk = [0_u8; 8192];
    let mut head = None;
    let mut parser = SseParser::default();
    let mut first_sse = None;
    let mut first_content = None;
    let mut generated = String::new();
    let mut error_body = Vec::new();

    loop {
        let read = stream
            .read(&mut chunk)
            .map_err(|e| BenchServeError::new(format!("lecture reponse HTTP: {e}")))?;
        if read == 0 {
            break;
        }
        let now = Instant::now();
        if head.is_none() {
            buffer.extend_from_slice(&chunk[..read]);
            if let Some(index) = find_http_header_end(&buffer) {
                let header_bytes = &buffer[..index];
                let header_text = str::from_utf8(header_bytes)
                    .map_err(|e| BenchServeError::new(format!("headers non UTF-8: {e}")))?;
                let parsed = parse_response_headers(header_text)?;
                let body_start = index + 4;
                let body = buffer[body_start..].to_vec();
                buffer.clear();
                let success = parsed.status == 200;
                head = Some(parsed);
                if success {
                    consume_sse_events(
                        &mut parser,
                        &body,
                        now,
                        &mut first_sse,
                        &mut first_content,
                        &mut generated,
                    )?;
                } else {
                    error_body.extend_from_slice(&body);
                }
            }
            continue;
        }
        if head.as_ref().is_some_and(|parsed| parsed.status == 200) {
            consume_sse_events(
                &mut parser,
                &chunk[..read],
                now,
                &mut first_sse,
                &mut first_content,
                &mut generated,
            )?;
        } else {
            error_body.extend_from_slice(&chunk[..read]);
        }
    }

    let Some(head) = head else {
        return Err(BenchServeError::new("reponse HTTP vide"));
    };
    if head.status != 200 {
        return Err(BenchServeError::with_status(
            head.status,
            format!("HTTP {}: {}", head.status, compact_body(&error_body)),
        ));
    }
    let Some(first_sse) = first_sse else {
        return Err(BenchServeError::with_status(
            head.status,
            "aucun chunk SSE recu",
        ));
    };
    let finished = Instant::now();
    let x_saragossa = x_saragossa_headers(&head.headers);
    let generated_tokens = x_saragossa
        .get("x-saragossa-decode-tokens")
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or_else(|| approximate_tokens(&generated));
    let decode_started = first_content.unwrap_or(first_sse);
    let decode_secs = finished
        .saturating_duration_since(decode_started)
        .as_secs_f64();
    let decode_tok_s = if decode_secs > 0.0 {
        generated_tokens as f64 / decode_secs
    } else {
        0.0
    };
    Ok(ResponseMetrics {
        status: head.status,
        ttft_ms: duration_ms(first_sse.saturating_duration_since(started)),
        decode_tok_s,
        total_ms: duration_ms(finished.saturating_duration_since(started)),
        generated_tokens,
        x_saragossa,
    })
}

fn consume_sse_events(
    parser: &mut SseParser,
    bytes: &[u8],
    observed_at: Instant,
    first_sse: &mut Option<Instant>,
    first_content: &mut Option<Instant>,
    generated: &mut String,
) -> BenchResult<()> {
    for event in parser.push(bytes)? {
        if first_sse.is_none() {
            *first_sse = Some(observed_at);
        }
        if event.done {
            continue;
        }
        if let Some(content) = event.content {
            if first_content.is_none() && !content.is_empty() {
                *first_content = Some(observed_at);
            }
            generated.push_str(&content);
        }
    }
    Ok(())
}

#[derive(Debug)]
struct ParsedHead {
    status: u16,
    headers: BTreeMap<String, String>,
}

fn parse_response_headers(header_text: &str) -> BenchResult<ParsedHead> {
    let mut lines = header_text.split("\r\n");
    let status_line = lines
        .next()
        .ok_or_else(|| BenchServeError::new("ligne de statut absente"))?;
    let mut parts = status_line.split_whitespace();
    let version = parts
        .next()
        .ok_or_else(|| BenchServeError::new("version HTTP absente"))?;
    if !version.starts_with("HTTP/") {
        return Err(BenchServeError::new(format!(
            "version HTTP invalide: {version}"
        )));
    }
    let status = parts
        .next()
        .ok_or_else(|| BenchServeError::new("statut HTTP absent"))?
        .parse::<u16>()
        .map_err(|e| BenchServeError::new(format!("statut HTTP invalide: {e}")))?;
    let mut headers = BTreeMap::new();
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let Some((name, value)) = line.split_once(':') else {
            return Err(BenchServeError::new(format!("header invalide: {line}")));
        };
        headers.insert(
            name.trim().to_ascii_lowercase(),
            value.trim_start().trim_end().to_string(),
        );
    }
    Ok(ParsedHead { status, headers })
}

fn find_http_header_end(buffer: &[u8]) -> Option<usize> {
    buffer.windows(4).position(|window| window == b"\r\n\r\n")
}

fn x_saragossa_headers(headers: &BTreeMap<String, String>) -> BTreeMap<String, String> {
    headers
        .iter()
        .filter(|(name, _)| name.starts_with("x-saragossa-"))
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect()
}

fn write_report(report: &BenchReport) -> BenchResult<PathBuf> {
    fs::create_dir_all("bench")
        .map_err(|e| BenchServeError::new(format!("creation dossier bench: {e}")))?;
    let path = PathBuf::from(format!("bench/serve-{}.json", report.timestamp));
    let bytes = serde_json::to_vec_pretty(report)
        .map_err(|e| BenchServeError::new(format!("serialisation rapport JSON: {e}")))?;
    fs::write(&path, bytes)
        .map_err(|e| BenchServeError::new(format!("ecriture {}: {e}", path.display())))?;
    Ok(path)
}

fn print_summary(summary: &BenchSummary, report_path: &Path) {
    println!("{:<24}value", "metric");
    println!("{:<24}{}", "requests", summary.requested);
    println!("{:<24}{}", "succeeded", summary.succeeded);
    println!("{:<24}{}", "failed", summary.failed);
    println!("{:<24}{}", "concurrency", summary.concurrency);
    println!("{:<24}{:.2}", "ttft_p50_ms", summary.ttft_p50_ms);
    println!("{:<24}{:.2}", "ttft_p95_ms", summary.ttft_p95_ms);
    println!(
        "{:<24}{:.2}",
        "decode_tok_s_mean", summary.decode_tok_s_mean
    );
    println!("{:<24}{:.2}", "total_ms", summary.total_ms);
    println!("{:<24}{}", "json", report_path.display());
}

fn synthetic_prompt(target_words: usize, index: usize) -> String {
    let words = [
        "local", "bench", "serve", "latency", "decode", "queue", "token", "request", "model",
        "prompt", "stream", "health", "metric", "system", "memory", "cache",
    ];
    let mut parts = Vec::with_capacity(target_words.saturating_add(8));
    parts.push(format!("Bench request {index}."));
    parts.push("Answer with one concise paragraph.".to_string());
    for offset in 0..target_words {
        let word = words
            .get((index + offset) % words.len())
            .expect("invariant: vocabulaire non vide");
        parts.push((*word).to_string());
    }
    parts.join(" ")
}

fn approximate_tokens(text: &str) -> usize {
    let words = text.split_whitespace().count();
    if words > 0 {
        words
    } else {
        usize::from(!text.is_empty())
    }
}

fn percentile(values: &[f64], quantile: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    if sorted.len() == 1 {
        return sorted.first().copied();
    }
    let position = quantile.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lower_index = position.floor() as usize;
    let upper_index = position.ceil() as usize;
    let lower = sorted
        .get(lower_index)
        .copied()
        .expect("invariant: indice percentile borne");
    let upper = sorted
        .get(upper_index)
        .copied()
        .expect("invariant: indice percentile borne");
    if lower_index == upper_index {
        Some(lower)
    } else {
        Some(lower + (upper - lower) * (position - lower_index as f64))
    }
}

fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    Some(values.iter().sum::<f64>() / values.len() as f64)
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn normalize_tcp_addr(value: &str) -> BenchResult<String> {
    let trimmed = value.trim();
    if trimmed.starts_with("https://") {
        return Err(BenchServeError::new("--url ne supporte que HTTP local"));
    }
    let without_scheme = trimmed.strip_prefix("http://").unwrap_or(trimmed);
    let addr = without_scheme
        .split_once('/')
        .map(|(addr, _)| addr)
        .unwrap_or(without_scheme)
        .trim();
    if addr.is_empty() || !addr.contains(':') {
        return Err(BenchServeError::new(
            "--url attend une adresse 127.0.0.1:PORT",
        ));
    }
    Ok(addr.to_string())
}

fn validate_api_key(value: &str) -> BenchResult<String> {
    if value.contains(['\r', '\n']) {
        return Err(BenchServeError::new("--api-key contient un retour ligne"));
    }
    if value.is_empty() {
        return Err(BenchServeError::new("--api-key ne peut pas etre vide"));
    }
    Ok(value.to_string())
}

fn parse_positive_usize(
    flag: &'static str,
    iter: &mut impl Iterator<Item = String>,
) -> BenchResult<usize> {
    let value = next_value(iter, flag)?;
    let parsed = value
        .parse::<usize>()
        .map_err(|e| BenchServeError::new(format!("{flag} invalide: {e}")))?;
    if parsed == 0 {
        return Err(BenchServeError::new(format!("{flag} doit etre > 0")));
    }
    Ok(parsed)
}

fn parse_positive_u64(
    flag: &'static str,
    iter: &mut impl Iterator<Item = String>,
) -> BenchResult<u64> {
    let value = next_value(iter, flag)?;
    let parsed = value
        .parse::<u64>()
        .map_err(|e| BenchServeError::new(format!("{flag} invalide: {e}")))?;
    if parsed == 0 {
        return Err(BenchServeError::new(format!("{flag} doit etre > 0")));
    }
    Ok(parsed)
}

fn next_value(iter: &mut impl Iterator<Item = String>, flag: &'static str) -> BenchResult<String> {
    iter.next()
        .ok_or_else(|| BenchServeError::new(format!("valeur manquante pour {flag}")))
}

fn compact_body(body: &[u8]) -> String {
    let text = String::from_utf8_lossy(body);
    let mut compact = text.trim().replace(['\r', '\n', '\t'], " ");
    compact.truncate(240);
    compact
}

fn unix_timestamp_secs() -> u64 {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(duration) => duration.as_secs(),
        Err(_) => 0,
    }
}

fn help_text() -> String {
    format!(
        "Usage: saragossa bench-serve [--socket {DEFAULT_SOCKET} | --url 127.0.0.1:PORT] \\
         [--api-key TOKEN] [--model {DEFAULT_MODEL}] [--requests {DEFAULT_REQUESTS}] \\
         [--concurrency {DEFAULT_CONCURRENCY}] [--prompt-tokens {DEFAULT_PROMPT_TOKENS}] \\
         [--max-tokens {DEFAULT_MAX_TOKENS}] [--timeout-secs {DEFAULT_TIMEOUT_SECS}]\n\
         Bench un serveur saragossa serve deja lance. Le transport par defaut \\
         est la socket Unix de serve. --url active TCP local et requiert \\
         --api-key ou {API_KEY_ENV}. La sortie stdout resume p50/p95 TTFT, \\
         debit decode moyen et total; le rapport detaille est ecrit dans bench/."
    )
}

fn boxed_error(error: BenchServeError) -> Box<dyn Error> {
    Box::new(error)
}
