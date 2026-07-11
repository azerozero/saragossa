//! Serveur HTTP local minimal pour l'API OpenAI-compatible.

use std::fs;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::str;
use std::time::{Duration, Instant};

#[cfg(unix)]
use std::os::unix::fs::{FileTypeExt, PermissionsExt};
#[cfg(unix)]
use std::os::unix::net::{UnixListener, UnixStream};

use serde::Serialize;

use super::anthropic::{handle_anthropic_messages, send_anthropic_error};
use super::audio::{handle_speech, handle_transcription};
use super::embeddings::handle_embeddings;
use super::error::{ServeError, ServeResult};
use super::protocol::{
    json_bytes, sse_done, sse_event, ChatCompletionChunk, ChatCompletionRequest,
    ChatCompletionResponse, HealthResponse,
};
use super::state::ServeState;
use super::streaming::CompletionStreamEvent;

const MAX_HEADER_BYTES: usize = 64 * 1024;
const MAX_BODY_BYTES: usize = 16 * 1024 * 1024;

/// Sert en HTTP TCP local.
pub(super) fn serve_tcp(
    addr: &str,
    api_key: &str,
    state: &mut ServeState,
    read_timeout: Duration,
) -> ServeResult<()> {
    let listener =
        TcpListener::bind(addr).map_err(|e| ServeError::io(format!("bind TCP {addr}"), e))?;
    eprintln!("saragossa serve TCP listening on http://{addr}/v1");
    eprintln!("saragossa serve TCP bearer auth enabled");
    for stream in listener.incoming() {
        match stream {
            Ok(mut stream) => {
                if let Err(error) = configure_tcp_stream(&stream, read_timeout) {
                    eprintln!("saragossa serve request error: {error}");
                    continue;
                }
                let deadline = Instant::now() + read_timeout;
                if let Err(error) = handle_connection(&mut stream, state, Some(api_key), deadline) {
                    eprintln!("saragossa serve request error: {error}");
                }
            }
            Err(error) => return Err(ServeError::io("accept TCP", error)),
        }
    }
    Ok(())
}

/// Sert en HTTP sur socket Unix avec permissions propriétaire.
#[cfg(unix)]
pub(super) fn serve_unix(
    path: &Path,
    state: &mut ServeState,
    read_timeout: Duration,
) -> ServeResult<()> {
    prepare_socket_path(path)?;
    let listener = UnixListener::bind(path)
        .map_err(|e| ServeError::io(format!("bind socket {}", path.display()), e))?;
    let permissions = fs::Permissions::from_mode(0o600);
    fs::set_permissions(path, permissions)
        .map_err(|e| ServeError::io(format!("chmod 0600 {}", path.display()), e))?;
    eprintln!(
        "saragossa serve Unix socket listening on {} (0600)",
        path.display()
    );
    for stream in listener.incoming() {
        match stream {
            Ok(mut stream) => {
                if let Err(error) = configure_unix_stream(&stream, read_timeout) {
                    eprintln!("saragossa serve request error: {error}");
                    continue;
                }
                let deadline = Instant::now() + read_timeout;
                if let Err(error) = handle_connection(&mut stream, state, None, deadline) {
                    eprintln!("saragossa serve request error: {error}");
                }
            }
            Err(error) => return Err(ServeError::io("accept socket Unix", error)),
        }
    }
    Ok(())
}

/// Echec explicite hors Unix.
#[cfg(not(unix))]
pub(super) fn serve_unix(
    _path: &Path,
    _state: &mut ServeState,
    _read_timeout: Duration,
) -> ServeResult<()> {
    Err(ServeError::args(
        "transport socket Unix indisponible sur cette plateforme",
    ))
}

fn configure_tcp_stream(stream: &TcpStream, read_timeout: Duration) -> ServeResult<()> {
    stream
        .set_read_timeout(Some(read_timeout))
        .map_err(|e| ServeError::io("set read timeout TCP", e))?;
    stream
        .set_write_timeout(Some(read_timeout))
        .map_err(|e| ServeError::io("set write timeout TCP", e))
}

#[cfg(unix)]
fn configure_unix_stream(stream: &UnixStream, read_timeout: Duration) -> ServeResult<()> {
    stream
        .set_read_timeout(Some(read_timeout))
        .map_err(|e| ServeError::io("set read timeout socket Unix", e))?;
    stream
        .set_write_timeout(Some(read_timeout))
        .map_err(|e| ServeError::io("set write timeout socket Unix", e))
}

#[cfg(unix)]
fn prepare_socket_path(path: &Path) -> ServeResult<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_socket() => fs::remove_file(path)
            .map_err(|e| ServeError::io(format!("remove stale socket {}", path.display()), e)),
        Ok(_) => Err(ServeError::args(format!(
            "{} existe déjà et n'est pas une socket",
            path.display()
        ))),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(ServeError::io(
            format!("stat socket {}", path.display()),
            error,
        )),
    }
}

fn handle_connection<S: Read + Write>(
    stream: &mut S,
    state: &mut ServeState,
    api_key: Option<&str>,
    read_deadline: Instant,
) -> ServeResult<()> {
    let request = match read_request(stream, read_deadline) {
        Ok(Some(request)) => request,
        Ok(None) => return Ok(()),
        Err(error) => {
            let _ = send_error(stream, error_status(&error), &error.to_string());
            return Err(error);
        }
    };
    if let Err(error) = route_request(stream, state, api_key, request) {
        let _ = send_error(stream, error_status(&error), &error.to_string());
        return Err(error);
    }
    Ok(())
}

fn route_request<S: Write>(
    stream: &mut S,
    state: &mut ServeState,
    api_key: Option<&str>,
    request: HttpRequest,
) -> ServeResult<()> {
    let path = request
        .path
        .split('?')
        .next()
        .unwrap_or(request.path.as_str());
    if request.method == "GET" && path == "/health" {
        return send_json(stream, 200, &HealthResponse::ok(), Vec::new());
    }
    if let Some(expected) = api_key {
        if !is_authorized(&request, expected) {
            if path == "/v1/messages" {
                return send_anthropic_error(stream, 401, "authentification bearer requise");
            }
            return send_error(stream, 401, "authentification bearer requise");
        }
    }
    match (request.method.as_str(), path) {
        ("GET", "/v1/models") => send_json(stream, 200, &state.models_response(), Vec::new()),
        ("POST", "/v1/chat/completions") => handle_chat(stream, state, &request),
        ("POST", "/v1/messages") => match handle_anthropic_messages(stream, state, &request.body) {
            Ok(()) => Ok(()),
            Err(error) => send_anthropic_error(stream, error_status(&error), &error.to_string()),
        },
        ("POST", "/v1/audio/transcriptions") => handle_transcription(
            stream,
            state.audio_mut(),
            request.header("content-type"),
            &request.body,
        ),
        ("POST", "/v1/audio/speech") => handle_speech(stream, state.audio_mut(), &request.body),
        ("POST", "/v1/embeddings") => {
            handle_embeddings(stream, state.embeddings_mut(), &request.body)
        }
        _ => send_error(stream, 404, "endpoint inconnu"),
    }
}

fn handle_chat<S: Write>(
    stream: &mut S,
    state: &mut ServeState,
    request: &HttpRequest,
) -> ServeResult<()> {
    let chat: ChatCompletionRequest = serde_json::from_slice(&request.body)
        .map_err(|e| ServeError::json("désérialisation chat/completions", e))?;
    chat.max_tokens_capped(state.max_tokens_cap())?;
    let stream_enabled = chat.stream;
    if stream_enabled {
        send_sse_streaming(stream, state, chat)
    } else {
        let completion = state.complete(chat)?;
        let headers = completion.metric_headers();
        let response = ChatCompletionResponse::new(
            &completion.model,
            completion.content,
            completion.finish_reason,
            completion.usage,
        );
        send_json(stream, 200, &response, headers)
    }
}

fn send_sse_streaming<S: Write>(
    stream: &mut S,
    state: &mut ServeState,
    chat: ChatCompletionRequest,
) -> ServeResult<()> {
    let mut stream_started = false;
    let mut stream_model = String::new();
    let result = state.complete_streaming(chat, |event| match event {
        CompletionStreamEvent::Start(start) => {
            stream_model = start.model.clone();
            stream_started = true;
            write_headers(
                stream,
                200,
                "OK",
                "text/event-stream; charset=utf-8",
                None,
                start.metric_headers(),
            )?;
            write_sse_event(stream, &ChatCompletionChunk::role(&start.model))?;
            stream.flush().map_err(|e| ServeError::io("flush SSE", e))
        }
        CompletionStreamEvent::Delta(delta) => {
            if delta.is_empty() {
                return Ok(());
            }
            write_sse_event(
                stream,
                &ChatCompletionChunk::content(&stream_model, delta.to_string()),
            )?;
            stream.flush().map_err(|e| ServeError::io("flush SSE", e))
        }
    });
    let completion = match result {
        Ok(completion) => completion,
        Err(error) if stream_started => {
            eprintln!("saragossa serve stream interrompu: {error}");
            return Ok(());
        }
        Err(error) => return Err(error),
    };
    write_sse_event(
        stream,
        &ChatCompletionChunk::done(&completion.model, completion.finish_reason),
    )?;
    stream
        .write_all(&sse_done())
        .map_err(|e| ServeError::io("écriture SSE [DONE]", e))?;
    stream.flush().map_err(|e| ServeError::io("flush SSE", e))
}

fn write_sse_event<S: Write, T: Serialize>(stream: &mut S, value: &T) -> ServeResult<()> {
    stream
        .write_all(&sse_event(value)?)
        .map_err(|e| ServeError::io("écriture SSE", e))
}

pub(super) fn send_json<S: Write, T: Serialize>(
    stream: &mut S,
    status: u16,
    value: &T,
    extra_headers: Vec<(&'static str, String)>,
) -> ServeResult<()> {
    let body = json_bytes(value)?;
    write_headers(
        stream,
        status,
        reason_phrase(status),
        "application/json; charset=utf-8",
        Some(body.len()),
        extra_headers,
    )?;
    stream
        .write_all(&body)
        .map_err(|e| ServeError::io("écriture réponse JSON", e))?;
    stream
        .flush()
        .map_err(|e| ServeError::io("flush réponse JSON", e))
}

fn send_error<S: Write>(stream: &mut S, status: u16, message: &str) -> ServeResult<()> {
    let body = serde_json::json!({
        "error": {
            "message": message,
            "type": "saragossa_error"
        }
    });
    send_json(stream, status, &body, Vec::new())
}

pub(super) fn write_headers<S: Write>(
    stream: &mut S,
    status: u16,
    reason: &str,
    content_type: &str,
    content_length: Option<usize>,
    extra_headers: Vec<(&'static str, String)>,
) -> ServeResult<()> {
    let mut head = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\nConnection: close\r\n"
    );
    if let Some(length) = content_length {
        head.push_str(&format!("Content-Length: {length}\r\n"));
    }
    for (name, value) in extra_headers {
        head.push_str(name);
        head.push_str(": ");
        head.push_str(&sanitize_header_value(&value));
        head.push_str("\r\n");
    }
    head.push_str("\r\n");
    stream
        .write_all(head.as_bytes())
        .map_err(|e| ServeError::io("écriture headers HTTP", e))
}

fn sanitize_header_value(value: &str) -> String {
    value.replace(['\r', '\n'], " ")
}

fn error_status(error: &ServeError) -> u16 {
    match error {
        ServeError::UnknownModel(_) => 404,
        ServeError::Args(_) | ServeError::Json { .. } | ServeError::Http(_) => 400,
        ServeError::Memory(_) => 503,
        ServeError::Io { .. } | ServeError::Inference(_) => 500,
    }
}

fn reason_phrase(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
        500 => "Internal Server Error",
        _ => "OK",
    }
}

fn is_authorized(request: &HttpRequest, expected: &str) -> bool {
    let Some(header) = request.header("authorization") else {
        return false;
    };
    let Some(token) = header.strip_prefix("Bearer ") else {
        return false;
    };
    constant_time_eq(token.as_bytes(), expected.as_bytes())
}

fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    let mut diff = left.len() ^ right.len();
    let max_len = left.len().max(right.len());
    for index in 0..max_len {
        let a = left.get(index).copied().unwrap_or(0);
        let b = right.get(index).copied().unwrap_or(0);
        diff |= usize::from(a ^ b);
    }
    diff == 0
}

#[derive(Debug)]
struct HttpRequest {
    method: String,
    path: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl HttpRequest {
    fn header(&self, name: &str) -> Option<&str> {
        let target = name.to_ascii_lowercase();
        self.headers
            .iter()
            .find(|(header, _)| header == &target)
            .map(|(_, value)| value.as_str())
    }
}

fn read_request<S: Read>(stream: &mut S, deadline: Instant) -> ServeResult<Option<HttpRequest>> {
    let mut buffer = Vec::new();
    let mut chunk = [0_u8; 1024];
    let header_end = loop {
        if let Some(index) = find_header_end(&buffer) {
            break index;
        }
        if buffer.len() > MAX_HEADER_BYTES {
            return Err(ServeError::Http("headers trop grands".to_string()));
        }
        // NOTE: le timeout socket ne borne que CHAQUE read ; sans deadline
        // absolue, un client au goutte-à-goutte épingle le serveur mono-thread.
        if Instant::now() >= deadline {
            return Err(ServeError::Http(
                "délai de lecture de la requête dépassé".to_string(),
            ));
        }
        let read = stream
            .read(&mut chunk)
            .map_err(|e| ServeError::io("lecture requête HTTP", e))?;
        if read == 0 {
            if buffer.is_empty() {
                return Ok(None);
            }
            return Err(ServeError::Http(
                "connexion coupée avant la fin des headers".to_string(),
            ));
        }
        buffer.extend_from_slice(&chunk[..read]);
    };
    let header_bytes = &buffer[..header_end];
    let header_text = str::from_utf8(header_bytes)
        .map_err(|e| ServeError::Http(format!("headers non UTF-8: {e}")))?;
    let (method, path, headers) = parse_headers(header_text)?;
    let content_length = content_length(&headers)?;
    if content_length > MAX_BODY_BYTES {
        return Err(ServeError::Http("body trop grand".to_string()));
    }
    let body_start = header_end + 4;
    let mut body = buffer[body_start..].to_vec();
    while body.len() < content_length {
        if Instant::now() >= deadline {
            return Err(ServeError::Http(
                "délai de lecture de la requête dépassé".to_string(),
            ));
        }
        let read = stream
            .read(&mut chunk)
            .map_err(|e| ServeError::io("lecture body HTTP", e))?;
        if read == 0 {
            return Err(ServeError::Http("body HTTP tronqué".to_string()));
        }
        body.extend_from_slice(&chunk[..read]);
    }
    body.truncate(content_length);
    Ok(Some(HttpRequest {
        method,
        path,
        headers,
        body,
    }))
}

fn parse_headers(header_text: &str) -> ServeResult<(String, String, Vec<(String, String)>)> {
    let mut lines = header_text.split("\r\n");
    let request_line = lines
        .next()
        .ok_or_else(|| ServeError::Http("ligne de requête absente".to_string()))?;
    let mut parts = request_line.split_whitespace();
    let method = parts
        .next()
        .ok_or_else(|| ServeError::Http("méthode absente".to_string()))?
        .to_string();
    let path = parts
        .next()
        .ok_or_else(|| ServeError::Http("chemin absent".to_string()))?
        .to_string();
    let version = parts
        .next()
        .ok_or_else(|| ServeError::Http("version HTTP absente".to_string()))?;
    if !version.starts_with("HTTP/") {
        return Err(ServeError::Http(format!(
            "version HTTP invalide: {version}"
        )));
    }
    let mut headers = Vec::new();
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let Some((name, value)) = line.split_once(':') else {
            return Err(ServeError::Http(format!("header invalide: {line}")));
        };
        headers.push((
            name.trim().to_ascii_lowercase(),
            value.trim_start().trim_end().to_string(),
        ));
    }
    Ok((method, path, headers))
}

fn content_length(headers: &[(String, String)]) -> ServeResult<usize> {
    let Some((_, value)) = headers.iter().find(|(name, _)| name == "content-length") else {
        return Ok(0);
    };
    value
        .parse::<usize>()
        .map_err(|e| ServeError::Http(format!("Content-Length invalide: {e}")))
}

fn find_header_end(buffer: &[u8]) -> Option<usize> {
    buffer.windows(4).position(|window| window == b"\r\n\r\n")
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;
    use std::net::TcpStream;
    use std::time::Duration;

    #[cfg(unix)]
    use std::os::unix::net::UnixStream;

    use super::super::args::ServeArgs;
    use super::super::state::ServeState;
    use super::*;

    fn far_deadline() -> Instant {
        Instant::now() + Duration::from_secs(30)
    }

    /// Client au goutte-à-goutte : un octet utile toutes les 5 ms, jamais de
    /// fin de headers.
    struct SlowLorisStream;

    impl Read for SlowLorisStream {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            std::thread::sleep(Duration::from_millis(5));
            if buf.is_empty() {
                return Ok(0);
            }
            buf[0] = b'A';
            Ok(1)
        }
    }

    #[test]
    fn read_request_cuts_slow_loris_at_absolute_deadline() {
        let mut stream = SlowLorisStream;

        let error = read_request(&mut stream, Instant::now() + Duration::from_millis(20))
            .expect_err("invariant: client au goutte-à-goutte coupé à la deadline");

        assert!(error.to_string().contains("délai"));
    }

    #[test]
    fn parses_request_with_body() {
        let raw = b"POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 2\r\n\r\n{}";
        let mut cursor = Cursor::new(raw.to_vec());

        let request = read_request(&mut cursor, far_deadline())
            .expect("invariant: requête lisible")
            .expect("invariant: requête présente");

        assert_eq!(request.method, "POST");
        assert_eq!(request.path, "/v1/chat/completions");
        assert_eq!(request.body, b"{}");
    }

    #[test]
    fn bearer_auth_uses_authorization_header() {
        let request = HttpRequest {
            method: "GET".to_string(),
            path: "/v1/models".to_string(),
            headers: vec![("authorization".to_string(), "Bearer secret".to_string())],
            body: Vec::new(),
        };

        assert!(is_authorized(&request, "secret"));
        assert!(!is_authorized(&request, "other"));
    }

    #[test]
    fn health_route_skips_bearer_without_leaking_models() {
        let args = ServeArgs::parse([
            "--model".to_string(),
            "alpha=/m/a".to_string(),
            "--model".to_string(),
            "beta=/m/b".to_string(),
        ])
        .expect("invariant: args valides");
        let mut state = ServeState::new(&args);
        let raw = b"GET /health HTTP/1.1\r\n\r\n";
        let mut stream = Cursor::new(raw.to_vec());

        handle_connection(&mut stream, &mut state, Some("secret"), far_deadline())
            .expect("invariant: health non authentifie");

        let response = String::from_utf8(stream.into_inner()).expect("invariant: reponse UTF-8");
        assert!(response.contains("HTTP/1.1 200 OK"));
        assert!(response.contains(r#""status":"ok""#));
        // La liste des modèles vit derrière le bearer (/v1/models) : l'endpoint
        // exempté d'auth ne doit pas la répéter.
        assert!(!response.contains("alpha"));
        assert!(!response.contains("models"));
    }

    #[test]
    fn bearer_still_protects_other_routes() {
        let args = ServeArgs::parse(Vec::<String>::new()).expect("invariant: args valides");
        let mut state = ServeState::new(&args);
        let raw = b"GET /v1/models HTTP/1.1\r\n\r\n";
        let mut stream = Cursor::new(raw.to_vec());

        handle_connection(&mut stream, &mut state, Some("secret"), far_deadline())
            .expect("invariant: erreur auth serialisee");

        let response = String::from_utf8(stream.into_inner()).expect("invariant: reponse UTF-8");
        assert!(response.contains("HTTP/1.1 401 Unauthorized"));
        assert!(response.contains("authentification bearer requise"));
    }

    #[test]
    fn accepted_tcp_stream_gets_read_and_write_timeout() {
        let listener = match TcpListener::bind("127.0.0.1:0") {
            Ok(listener) => listener,
            Err(error) if error.kind() == std::io::ErrorKind::PermissionDenied => return,
            Err(error) => panic!("invariant: port local disponible: {error}"),
        };
        let addr = listener
            .local_addr()
            .expect("invariant: adresse listener disponible");
        let client = TcpStream::connect(addr).expect("invariant: connexion locale possible");
        let (stream, _) = listener.accept().expect("invariant: accept local possible");
        let timeout = Duration::from_secs(7);

        configure_tcp_stream(&stream, timeout).expect("invariant: timeout TCP applicable");

        assert_eq!(
            stream
                .read_timeout()
                .expect("invariant: timeout TCP lisible"),
            Some(timeout)
        );
        assert_eq!(
            stream
                .write_timeout()
                .expect("invariant: timeout écriture TCP lisible"),
            Some(timeout)
        );
        drop(client);
    }

    #[cfg(unix)]
    #[test]
    fn accepted_unix_stream_gets_read_and_write_timeout() {
        let (stream, peer) = UnixStream::pair().expect("invariant: paire Unix locale possible");
        let timeout = Duration::from_secs(7);

        configure_unix_stream(&stream, timeout).expect("invariant: timeout Unix applicable");

        assert_eq!(
            stream
                .read_timeout()
                .expect("invariant: timeout Unix lisible"),
            Some(timeout)
        );
        assert_eq!(
            stream
                .write_timeout()
                .expect("invariant: timeout écriture Unix lisible"),
            Some(timeout)
        );
        drop(peer);
    }

    #[test]
    fn chat_request_above_max_tokens_cap_returns_400() {
        let args = ServeArgs::parse(["--max-tokens-cap".to_string(), "4".to_string()])
            .expect("invariant: args valides");
        let mut state = ServeState::new(&args);
        let body = br#"{"model":"reti-35b","messages":[],"max_tokens":5}"#;
        let raw = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            str::from_utf8(body).expect("invariant: JSON test UTF-8")
        );
        let mut stream = Cursor::new(raw.into_bytes());

        let error = handle_connection(&mut stream, &mut state, None, far_deadline())
            .expect_err("invariant: max_tokens au-dessus du cap refusé");

        assert!(error.to_string().contains("plafond serveur 4"));
        let response = String::from_utf8(stream.into_inner()).expect("invariant: réponse UTF-8");
        assert!(response.contains("HTTP/1.1 400 Bad Request"));
        assert!(response.contains("plafond serveur 4"));
    }

    #[test]
    fn audio_speech_route_without_model_returns_400() {
        let args = ServeArgs::parse(Vec::<String>::new()).expect("invariant: args valides");
        let mut state = ServeState::new(&args);
        let body = br#"{"model":"reti-tts","input":"Bonjour"}"#;
        let raw = format!(
            "POST /v1/audio/speech HTTP/1.1\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            str::from_utf8(body).expect("invariant: JSON test UTF-8")
        );
        let mut stream = Cursor::new(raw.into_bytes());

        let error = handle_connection(&mut stream, &mut state, None, far_deadline())
            .expect_err("invariant: TTS non configuré refusé");

        assert!(error.to_string().contains("TTS non configuré"));
        let response = String::from_utf8(stream.into_inner()).expect("invariant: réponse UTF-8");
        assert!(response.contains("HTTP/1.1 400 Bad Request"));
        assert!(response.contains("TTS non configuré"));
    }

    #[test]
    fn audio_transcription_route_without_model_returns_400() {
        let args = ServeArgs::parse(Vec::<String>::new()).expect("invariant: args valides");
        let mut state = ServeState::new(&args);
        let boundary = "BND";
        let mut body = Vec::new();
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(
            b"Content-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n\r\n",
        );
        body.extend_from_slice(b"RIFF0000WAVE");
        body.extend_from_slice(b"\r\n");
        body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
        let raw_head = format!(
            "POST /v1/audio/transcriptions HTTP/1.1\r\nContent-Type: multipart/form-data; boundary={boundary}\r\nContent-Length: {}\r\n\r\n",
            body.len()
        );
        let mut raw = raw_head.into_bytes();
        raw.extend_from_slice(&body);
        let mut stream = Cursor::new(raw);

        let error = handle_connection(&mut stream, &mut state, None, far_deadline())
            .expect_err("invariant: STT non configuré ou WAV refusé");

        // Le WAV factice est rejeté avant le chargement modèle : dans les deux cas
        // la route est bien câblée et renvoie une erreur 400.
        let response = String::from_utf8_lossy(&stream.into_inner()).to_string();
        assert!(response.contains("HTTP/1.1 400 Bad Request"));
        assert!(error.to_string().contains("WAV") || error.to_string().contains("STT"));
    }

    #[test]
    fn embeddings_route_without_model_returns_400() {
        let args = ServeArgs::parse(Vec::<String>::new()).expect("invariant: args valides");
        let mut state = ServeState::new(&args);
        let body = br#"{"model":"e5","input":"bonjour"}"#;
        let raw = format!(
            "POST /v1/embeddings HTTP/1.1\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            str::from_utf8(body).expect("invariant: JSON test UTF-8")
        );
        let mut stream = Cursor::new(raw.into_bytes());

        let error = handle_connection(&mut stream, &mut state, None, far_deadline())
            .expect_err("invariant: embeddings non configuré refusé");

        assert!(error.to_string().contains("embeddings non configuré"));
        let response = String::from_utf8(stream.into_inner()).expect("invariant: réponse UTF-8");
        assert!(response.contains("HTTP/1.1 400 Bad Request"));
        assert!(response.contains("embeddings non configuré"));
    }

    #[test]
    fn anthropic_messages_errors_use_anthropic_shape() {
        let args = ServeArgs::parse(Vec::<String>::new()).expect("invariant: args valides");
        let mut state = ServeState::new(&args);
        let body = br#"{"model":"reti-35b","messages":[{"role":"user","content":"Bonjour"}]}"#;
        let raw = format!(
            "POST /v1/messages HTTP/1.1\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            str::from_utf8(body).expect("invariant: JSON test UTF-8")
        );
        let mut stream = Cursor::new(raw.into_bytes());

        handle_connection(&mut stream, &mut state, None, far_deadline())
            .expect("invariant: erreur Anthropic sérialisée");

        let response = String::from_utf8(stream.into_inner()).expect("invariant: réponse UTF-8");
        assert!(response.contains("HTTP/1.1 400 Bad Request"));
        let response_start = response
            .rfind("HTTP/1.1 400 Bad Request")
            .expect("invariant: réponse HTTP présente");
        let body = response[response_start..]
            .split_once("\r\n\r\n")
            .map(|(_, body)| body)
            .expect("invariant: body HTTP présent");
        let value: serde_json::Value = serde_json::from_str(body).expect("invariant: JSON erreur");
        assert_eq!(value["type"], "error");
        assert_eq!(value["error"]["type"], "invalid_request_error");
        assert!(value["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("max_tokens")));
    }
}
