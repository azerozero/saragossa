//! Serveur HTTP local minimal pour l'API OpenAI-compatible.

#[cfg(unix)]
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

use super::anthropic::{
    anthropic_chat, handle_anthropic_messages, send_anthropic_error, AnthropicSse,
};
use super::audio::{handle_speech, handle_transcription};
use super::dispatch::{self, Front};
use super::embeddings::handle_embeddings;
use super::error::{ServeError, ServeResult};
use super::flint_output::FlintEvent;
use super::protocol::{
    created_now, json_bytes, sse_done, sse_event, AssistantMessage, ChatCompletionChunk,
    ChatCompletionRequest, ChatCompletionResponse, HealthResponse, ModelInfo, ModelsResponse,
    ResponseFormatMode, StreamErrorEvent,
};
use super::state::{ServeState, ServedCompletion};
use super::streaming::{CompletionSink, CompletionStreamEvent};
use super::tool_output::ToolOutputParser;

const MAX_HEADER_BYTES: usize = 64 * 1024;
const MAX_BODY_BYTES: usize = 16 * 1024 * 1024;
const SESSION_HEADER: &str = "x-saragossa-session";

type ParsedHeaders = (String, String, Vec<(String, String)>);

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
    let front = Front::new(Some(api_key.to_string()), state.model_ids());
    dispatch::serve(state, front, read_timeout, move |dispatcher| {
        // NOTE: sans TCP_NODELAY, Nagle et l'ACK retardé regroupent les petits
        // événements SSE d'un flux ; un échec de réglage laisse le défaut.
        let incoming = listener.incoming().map(|stream| {
            stream.inspect(|stream| {
                let _ = stream.set_nodelay(true);
            })
        });
        dispatcher.accept_all(incoming, "accept TCP")
    })
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
    let front = Front::new(None, state.model_ids());
    dispatch::serve(state, front, read_timeout, move |dispatcher| {
        dispatcher.accept_all(listener.incoming(), "accept socket Unix")
    })
}

/// Echec explicite hors Unix.
#[cfg(not(unix))]
pub(super) fn serve_unix(
    _path: &Path,
    _state: &mut ServeState,
    _read_timeout: Duration,
) -> ServeResult<()> {
    Err(ServeError::args(
        "transport socket Unix indisponible; utilisez --port et --api-key pour TCP",
    ))
}

pub(super) fn configure_tcp_stream(stream: &TcpStream, read_timeout: Duration) -> ServeResult<()> {
    stream
        .set_read_timeout(Some(read_timeout))
        .map_err(|e| ServeError::io("set read timeout TCP", e))?;
    stream
        .set_write_timeout(Some(read_timeout))
        .map_err(|e| ServeError::io("set write timeout TCP", e))
}

#[cfg(unix)]
pub(super) fn configure_unix_stream(
    stream: &UnixStream,
    read_timeout: Duration,
) -> ServeResult<()> {
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

/// Lit avec `read` ; une requête illisible reçoit sa réponse d'erreur ici.
pub(super) fn read_or_reject<S: Read + Write, T>(
    stream: &mut S,
    read: impl FnOnce(&mut S) -> ServeResult<T>,
) -> ServeResult<T> {
    read(stream).inspect_err(|error| {
        let _ = send_failure(stream, error, false);
    })
}

fn is_anthropic_route(path: &str) -> bool {
    matches!(path, "/v1/messages" | "/v1/messages/count_tokens")
}

/// Erreur au format de la route : Anthropic pour `/v1/messages`, comme les
/// autres erreurs de cette route.
pub(super) fn send_route_error<S: Write>(
    stream: &mut S,
    request: &HttpRequest,
    status: u16,
    message: &str,
) -> ServeResult<()> {
    if is_anthropic_route(request.route()) {
        send_anthropic_error(stream, status, message)
    } else {
        send_error(stream, status, message)
    }
}

/// Tente d'envoyer une réponse d'erreur avant de propager l'échec.
pub(super) fn reply_on_error<S: Write>(stream: &mut S, result: ServeResult<()>) -> ServeResult<()> {
    result.inspect_err(|error| {
        let _ = send_failure(stream, error, false);
    })
}

/// Routes indépendantes de `ServeState`, servies par le thread de connexion
/// même pendant une génération : santé, bearer et liste des modèles. `None`
/// laisse la requête au thread moteur.
pub(super) fn route_front<S: Write>(
    stream: &mut S,
    front: &Front,
    request: &HttpRequest,
) -> Option<ServeResult<()>> {
    let path = request.route();
    if request.method == "GET" && path == "/health" {
        return Some(send_json(stream, 200, &HealthResponse::ok(), Vec::new()));
    }
    if let Some(expected) = front.api_key.as_deref() {
        if !is_authorized(request, expected) {
            if is_anthropic_route(path) {
                return Some(send_anthropic_error(
                    stream,
                    401,
                    "authentification bearer requise",
                ));
            }
            return Some(send_error(stream, 401, "authentification bearer requise"));
        }
    }
    if request.method == "GET" && path == "/v1/models" {
        let models = ModelsResponse::from_ids(&front.model_ids);
        return Some(send_json(stream, 200, &models, Vec::new()));
    }
    if request.method == "GET" {
        if let Some(id) = path.strip_prefix("/v1/models/") {
            let id = match decode_model_id(id) {
                Ok(id) => id,
                Err(error) => return Some(send_failure(stream, &error, false)),
            };
            return Some(if front.model_ids.iter().any(|model| model == &id) {
                send_json(stream, 200, &ModelInfo::new(&id), Vec::new())
            } else {
                send_failure(stream, &ServeError::UnknownModel(id), false)
            });
        }
    }
    None
}

/// Décode le suffixe après routage : `%2F` et `%3F` appartiennent à l'alias,
/// `+` reste littéral et `%25` ne déclenche jamais un second décodage.
fn decode_model_id(encoded: &str) -> ServeResult<String> {
    let invalid = || ServeError::Http("encodage URL du modèle invalide".to_string());
    let mut decoded = Vec::with_capacity(encoded.len());
    let mut bytes = encoded.bytes();
    while let Some(byte) = bytes.next() {
        if byte == b'%' {
            let high = bytes.next().and_then(|b| char::from(b).to_digit(16));
            let low = bytes.next().and_then(|b| char::from(b).to_digit(16));
            match (high, low) {
                (Some(high), Some(low)) => decoded.push((high * 16 + low) as u8),
                _ => return Err(invalid()),
            }
        } else {
            decoded.push(byte);
        }
    }
    String::from_utf8(decoded).map_err(|_| invalid())
}

/// Traite sur le thread moteur une requête déjà authentifiée par
/// [`route_front`].
pub(super) fn handle_job<S: Write>(
    stream: &mut S,
    state: &mut ServeState,
    request: &HttpRequest,
) -> ServeResult<()> {
    let result = route_engine(stream, state, request);
    reply_on_error(stream, result)
}

fn route_engine<S: Write>(
    stream: &mut S,
    state: &mut ServeState,
    request: &HttpRequest,
) -> ServeResult<()> {
    match (request.method.as_str(), request.route()) {
        ("POST", "/v1/chat/completions") => handle_chat(stream, state, request),
        ("POST", "/v1/messages/count_tokens") => {
            match super::anthropic::handle_count_tokens(stream, state, &request.body) {
                Ok(()) => Ok(()),
                Err(error) => send_failure(stream, &error, true),
            }
        }
        ("POST", "/v1/responses") => super::responses::handle(stream, state, &request.body),
        ("POST", "/v1/messages") => match handle_anthropic_messages(
            stream,
            state,
            request.header(SESSION_HEADER),
            &request.body,
        ) {
            Ok(()) => Ok(()),
            Err(error) => send_failure(stream, &error, true),
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
    let chat = ChatCompletionRequest::parse(&request.body)?;
    chat.validate(state.max_tokens_cap())?;
    let session_key = chat.session_key(request.header(SESSION_HEADER));
    let parser = state.output_parser(&chat)?;
    let stream_enabled = chat.stream;
    if stream_enabled {
        send_sse_streaming(stream, state, chat, session_key, parser)
    } else {
        let completion = state.complete(chat, session_key.as_deref())?;
        let headers = completion.metric_headers();
        let mut finish_reason = completion.finish_reason;
        let message = if let Some(parser) = parser {
            let output = parser.parse(&completion.content);
            finish_reason = flint_finish_reason(finish_reason, !output.tool_calls.is_empty());
            let content = if output.content.is_empty() && !output.tool_calls.is_empty() {
                None
            } else {
                Some(output.content)
            };
            AssistantMessage::new(content, output.reasoning, output.tool_calls)
        } else {
            AssistantMessage::text(completion.content)
        };
        let response = ChatCompletionResponse::new(
            &completion.model,
            message,
            finish_reason,
            completion.usage,
        );
        send_json(stream, 200, &response, headers)
    }
}

/// Requête chat admissible à un lot.
pub(super) struct BatchChat {
    chat: ChatCompletionRequest,
    anthropic: bool,
}

impl BatchChat {
    /// Modèle visé : un lot ne réunit que des requêtes du même modèle.
    pub(super) fn model(&self) -> &str {
        &self.chat.model
    }
}

/// Analyse `request` pour un lot : chat OpenAI ou Messages Anthropic streamé,
/// valide, greedy, sans sortie structurée. `None` laisse la requête au chemin
/// solo, qui rapporte ses erreurs avant tout chargement. `mtp_model` dit si un
/// modèle est chargé avec son decode MTP actif.
///
/// NOTE: une requête avec session garde le solo et son cache de préfixe par
/// blocs ; dans un lot, elle perdrait le TTFT de son tour suivant. Sauf quand
/// le solo la servirait en MTP (modèle MTP et `mtp::routes_request`, `stop`
/// compris) : ce chemin ignore déjà la session (cf. `complete_streaming`), le
/// lot ne perd rien ; ses stops texte arrêtent le flux (`drive_batch`).
pub(super) fn batch_chat(
    request: &HttpRequest,
    max_tokens_cap: usize,
    mtp_model: impl Fn(&str) -> bool,
) -> Option<BatchChat> {
    let header = request.header(SESSION_HEADER);
    let (chat, session_key, stream, anthropic) = match (request.method.as_str(), request.route()) {
        ("POST", "/v1/chat/completions") => {
            let chat: ChatCompletionRequest = serde_json::from_slice(&request.body).ok()?;
            let session_key = chat.session_key(header);
            let stream = chat.stream;
            (chat, session_key, stream, false)
        }
        ("POST", "/v1/messages") => {
            let (chat, session_key, stream) = anthropic_chat(&request.body, header).ok()?;
            (chat, session_key, stream, true)
        }
        _ => return None,
    };
    // `batchable` garantit greedy et texte libre ; le solo prend alors le MTP
    // si le modèle l'a et que ses options l'admettent.
    let session_ignored = || {
        mtp_model(&chat.model)
            // Lot greedy seulement (`batchable`) : la règle d'échantillonnage est sans objet.
            && super::mtp::routes_request(&chat, ResponseFormatMode::Text, false)
    };
    (stream && chat.batchable(max_tokens_cap) && (session_key.is_none() || session_ignored()))
        .then_some(BatchChat { chat, anthropic })
}

/// Sert ensemble des requêtes acceptées par [`batch_chat`] pour un même
/// modèle : chaque client reçoit le même flux SSE que sur le chemin solo, ou
/// son erreur au format de sa route.
pub(super) fn handle_batch<S: Write>(state: &mut ServeState, jobs: &mut [(&mut S, &HttpRequest)]) {
    let cap = state.max_tokens_cap();
    let mut batched = Vec::with_capacity(jobs.len());
    let mut solo = Vec::new();
    for (index, (_, request)) in jobs.iter().enumerate() {
        match batch_chat(request, cap, |model| state.model_routes_mtp(model)) {
            Some(chat) => batched.push((index, chat)),
            None => solo.push(index),
        }
    }
    let mut requests = Vec::with_capacity(batched.len());
    let mut sinks: Vec<Box<dyn CompletionSink + '_>> = Vec::with_capacity(batched.len());
    let mut slots = Vec::with_capacity(batched.len());
    let mut batched = batched.into_iter().peekable();
    for (index, (stream, _)) in jobs.iter_mut().enumerate() {
        let Some((_, batch)) = batched.next_if(|(slot, _)| *slot == index) else {
            continue;
        };
        let stream: &mut S = stream;
        let parser = match state.output_parser(&batch.chat) {
            Ok(parser) => parser,
            Err(error) => {
                let _ = if batch.anthropic {
                    send_anthropic_error(stream, error_status(&error), &error.to_string())
                } else {
                    send_error(stream, error_status(&error), &error.to_string())
                };
                continue;
            }
        };
        if batch.anthropic {
            sinks.push(Box::new(AnthropicSse::with_parser(stream, parser)));
        } else {
            sinks.push(Box::new(OpenAiSse::new(
                stream,
                parser,
                batch.chat.include_usage(),
            )));
        }
        requests.push(batch.chat);
        slots.push(index);
    }
    let (outcomes, _) = state.complete_streaming_batch(requests, &mut sinks);
    drop(sinks);
    for (index, outcome) in slots.into_iter().zip(outcomes) {
        if let Err(error) = outcome {
            let (stream, request) = &mut jobs[index];
            eprintln!("saragossa serve request error: {error}");
            let _ = send_failure(*stream, &error, is_anthropic_route(request.route()));
        }
    }
    // NOTE: l'analyse est déterministe, ce cas ne survient pas ; le solo
    // reste la réponse sûre s'il survenait.
    for index in solo {
        let (stream, request) = &mut jobs[index];
        if let Err(error) = handle_job(*stream, state, request) {
            eprintln!("saragossa serve request error: {error}");
        }
    }
}

/// `stop` devient `tool_calls` quand le tour a appelé des outils (SGLang).
fn flint_finish_reason(finish_reason: &'static str, has_tool_calls: bool) -> &'static str {
    if has_tool_calls && finish_reason == "stop" {
        "tool_calls"
    } else {
        finish_reason
    }
}

fn write_flint_events<S: Write>(
    stream: &mut S,
    id: &str,
    created: u64,
    include_usage: bool,
    model: &str,
    events: Vec<FlintEvent>,
) -> ServeResult<()> {
    for event in events {
        let chunk = match event {
            FlintEvent::Reasoning(text) => ChatCompletionChunk::reasoning(model, text),
            FlintEvent::Content(text) => ChatCompletionChunk::content(model, text),
            FlintEvent::ToolCall(call) => ChatCompletionChunk::tool_call(model, call),
        };
        write_sse_event(
            stream,
            &chunk
                .with_identity(id, created)
                .with_usage_option(include_usage),
        )?;
    }
    Ok(())
}

fn send_sse_streaming<S: Write>(
    stream: &mut S,
    state: &mut ServeState,
    chat: ChatCompletionRequest,
    session_key: Option<String>,
    flint: Option<ToolOutputParser>,
) -> ServeResult<()> {
    let mut sink = OpenAiSse::new(stream, flint, chat.include_usage());
    let result = state.complete_streaming(chat, session_key.as_deref(), |event| sink.event(event));
    sink.finish(result)
}

/// Réponse SSE OpenAI `chat.completion.chunk` d'un client.
struct OpenAiSse<'a, S> {
    stream: &'a mut S,
    flint: Option<ToolOutputParser>,
    started: bool,
    model: String,
    /// `stream_options.include_usage` : chunk d'usage avant `[DONE]`.
    include_usage: bool,
    /// Identifiant commun à tous les chunks du flux.
    id: String,
    /// Timestamp capturé une seule fois à l’ouverture du flux.
    created: u64,
}

impl<'a, S: Write> OpenAiSse<'a, S> {
    fn new(stream: &'a mut S, flint: Option<ToolOutputParser>, include_usage: bool) -> Self {
        Self {
            stream,
            flint,
            started: false,
            model: String::new(),
            include_usage,
            id: ChatCompletionChunk::stream_id(),
            created: created_now(),
        }
    }

    fn flush(&mut self) -> ServeResult<()> {
        self.stream
            .flush()
            .map_err(|e| ServeError::io("flush SSE", e))
    }
}

impl<S: Write> CompletionSink for OpenAiSse<'_, S> {
    fn event(&mut self, event: CompletionStreamEvent<'_>) -> ServeResult<()> {
        match event {
            CompletionStreamEvent::Start(start) => {
                self.model = start.model.clone();
                self.started = true;
                write_headers(
                    self.stream,
                    200,
                    "OK",
                    "text/event-stream; charset=utf-8",
                    None,
                    start.metric_headers(),
                )?;
                write_sse_event(
                    self.stream,
                    &ChatCompletionChunk::role(&start.model)
                        .with_identity(&self.id, self.created)
                        .with_usage_option(self.include_usage),
                )?;
            }
            CompletionStreamEvent::Delta(delta) => {
                if delta.is_empty() {
                    return Ok(());
                }
                match self.flint.as_mut() {
                    Some(parser) => {
                        write_flint_events(
                            self.stream,
                            &self.id,
                            self.created,
                            self.include_usage,
                            &self.model,
                            parser.push(delta),
                        )?;
                    }
                    None => write_sse_event(
                        self.stream,
                        &ChatCompletionChunk::content(&self.model, delta.to_string())
                            .with_identity(&self.id, self.created)
                            .with_usage_option(self.include_usage),
                    )?,
                }
            }
            CompletionStreamEvent::TerminalError(error) => {
                write_sse_event(
                    self.stream,
                    &StreamErrorEvent::new(error.message, error.error_type),
                )?;
            }
        }
        self.flush()
    }

    fn finish(&mut self, result: ServeResult<ServedCompletion>) -> ServeResult<()> {
        let completion = match result {
            Ok(completion) => completion,
            Err(error) if self.started => {
                eprintln!("saragossa serve stream interrompu: {error}");
                return Ok(());
            }
            Err(error) => return Err(error),
        };
        let mut finish_reason = completion.finish_reason;
        if let Some(parser) = self.flint.as_mut() {
            write_flint_events(
                self.stream,
                &self.id,
                self.created,
                self.include_usage,
                &completion.model,
                parser.finish(),
            )?;
            finish_reason = flint_finish_reason(finish_reason, parser.has_tool_calls());
        }
        write_sse_event(
            self.stream,
            &ChatCompletionChunk::done(&completion.model, finish_reason)
                .with_identity(&self.id, self.created)
                .with_usage_option(self.include_usage),
        )?;
        if self.include_usage {
            write_sse_event(
                self.stream,
                &ChatCompletionChunk::usage(&completion.model, completion.usage)
                    .with_identity(&self.id, self.created),
            )?;
        }
        self.stream
            .write_all(&sse_done())
            .map_err(|e| ServeError::io("écriture SSE [DONE]", e))?;
        self.flush()
    }
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

/// Conserve les détails typés que les clients utilisent pour traiter l'erreur.
fn send_failure<S: Write>(stream: &mut S, error: &ServeError, anthropic: bool) -> ServeResult<()> {
    let status = error_status(error);
    let anthropic = anthropic || matches!(error, ServeError::BodyTooLarge { anthropic: true });
    let (code, param) = match error {
        ServeError::UnknownModel(_) => (Some("model_not_found"), Some("model")),
        ServeError::InvalidParameter { param, .. } => (None, Some(*param)),
        ServeError::NotImplementedParameter { param, .. } => (None, Some(param.as_str())),
        _ => {
            return if anthropic {
                send_anthropic_error(stream, status, &error.to_string())
            } else {
                send_error(stream, status, &error.to_string())
            };
        }
    };
    let mut body = serde_json::json!({"error": {
        "message": error.to_string(),
        "type": if anthropic && status == 404 { "not_found_error" } else { "invalid_request_error" }
    }});
    if anthropic {
        body["type"] = "error".into();
    } else if let Some(param) = param {
        body["error"]["param"] = param.into();
    }
    if let Some(code) = code {
        body["error"]["code"] = code.into();
    }
    send_json(stream, status, &body, Vec::new())
}

pub(super) fn send_error<S: Write>(stream: &mut S, status: u16, message: &str) -> ServeResult<()> {
    let body = serde_json::json!({
        "error": {
            "message": message,
            "type": if status == 400 { "invalid_request_error" } else { "saragossa_error" }
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
        ServeError::Args(_)
        | ServeError::Json { .. }
        | ServeError::Http(_)
        | ServeError::BodyTooLarge { .. }
        | ServeError::InvalidParameter { .. }
        | ServeError::IncompleteJson(_) => 400,
        ServeError::NotImplemented(_) | ServeError::NotImplementedParameter { .. } => 501,
        ServeError::Memory(_) => 503,
        ServeError::Io { .. } | ServeError::Inference(_) | ServeError::Batch(_) => 500,
    }
}

fn reason_phrase(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
        501 => "Not Implemented",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "OK",
    }
}

fn is_authorized(request: &HttpRequest, expected: &str) -> bool {
    if is_anthropic_route(request.route()) {
        if let Some(key) = request.header("x-api-key") {
            if constant_time_eq(key.as_bytes(), expected.as_bytes()) {
                return true;
            }
        }
    }
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
pub(super) struct HttpRequest {
    pub(super) method: String,
    path: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl HttpRequest {
    /// Chemin sans query string.
    pub(super) fn route(&self) -> &str {
        self.path.split('?').next().unwrap_or(self.path.as_str())
    }

    fn header(&self, name: &str) -> Option<&str> {
        let target = name.to_ascii_lowercase();
        self.headers
            .iter()
            .find(|(header, _)| header == &target)
            .map(|(_, value)| value.as_str())
    }
}

/// Lit headers puis corps d'un trait (chemin des tests sans threads).
#[cfg(test)]
pub(super) fn read_request<S: Read>(
    stream: &mut S,
    deadline: Instant,
) -> ServeResult<Option<HttpRequest>> {
    let Some((mut request, content_length)) = read_head(stream, deadline)? else {
        return Ok(None);
    };
    read_body(stream, &mut request, content_length, deadline)?;
    Ok(Some(request))
}

/// Lit les headers seuls : le corps reste sur la socket pour qu'un client non
/// authentifié ne fasse jamais allouer ses 16 Mio. Le corps de la requête
/// rendue ne contient que les octets déjà reçus ; la longueur annoncée suit.
pub(super) fn read_head<S: Read>(
    stream: &mut S,
    deadline: Instant,
) -> ServeResult<Option<(HttpRequest, usize)>> {
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
        return Err(ServeError::BodyTooLarge {
            anthropic: is_anthropic_route(path.split('?').next().unwrap_or_default()),
        });
    }
    let body = buffer[header_end + 4..].to_vec();
    Ok(Some((
        HttpRequest {
            method,
            path,
            headers,
            body,
        },
        content_length,
    )))
}

/// Complète le corps annoncé par `Content-Length` (déjà borné par
/// [`read_head`]).
pub(super) fn read_body<S: Read>(
    stream: &mut S,
    request: &mut HttpRequest,
    content_length: usize,
    deadline: Instant,
) -> ServeResult<()> {
    let mut chunk = [0_u8; 1024];
    let body = &mut request.body;
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
    Ok(())
}

fn parse_headers(header_text: &str) -> ServeResult<ParsedHeaders> {
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

    /// Chemin complet d'une connexion sans threads : front puis moteur.
    fn handle_connection<S: Read + Write>(
        stream: &mut S,
        state: &mut ServeState,
        api_key: Option<&str>,
        read_deadline: Instant,
    ) -> ServeResult<()> {
        let Some(request) = read_or_reject(stream, |s| read_request(s, read_deadline))? else {
            return Ok(());
        };
        let front = Front::new(api_key.map(str::to_string), state.model_ids());
        if let Some(result) = route_front(stream, &front, &request) {
            return reply_on_error(stream, result);
        }
        handle_job(stream, state, &request)
    }

    fn far_deadline() -> Instant {
        Instant::now() + Duration::from_secs(30)
    }

    #[test]
    fn new_endpoints_reject_before_loading_with_protocol_envelopes() {
        let args = super::super::args::ServeArgs::parse([
            "--model".to_owned(),
            "fixture=/nonexistent/fixture".to_owned(),
        ])
        .expect("invariant: args");
        let mut state = ServeState::new(&args);
        for (route, body, status) in [
            (
                "/v1/messages/count_tokens",
                r#"{"model":"absent","messages":[]}"#,
                404,
            ),
            (
                "/v1/messages/count_tokens",
                r#"{"model":"fixture","messages":[{"role":"user","content":[{"type":"image","source":{}}]}]}"#,
                501,
            ),
            ("/v1/messages/count_tokens", "{", 400),
            (
                "/v1/responses",
                r#"{"model":"fixture","input":"x","store":true}"#,
                501,
            ),
            (
                "/v1/responses",
                r#"{"model":"fixture","input":"x","include":[]}"#,
                501,
            ),
            ("/v1/responses", r#"{"model":"absent","input":"x"}"#, 404),
            ("/v1/responses", r#"{"model":"fixture","input":42}"#, 400),
        ] {
            let raw = format!(
                "POST {route} HTTP/1.1\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            );
            let request = read_request(
                &mut std::io::Cursor::new(raw),
                std::time::Instant::now() + std::time::Duration::from_secs(5),
            )
            .expect("invariant: lecture")
            .expect("invariant: requête");
            let mut wire = Vec::new();
            let _ = handle_job(&mut wire, &mut state, &request);
            let wire = String::from_utf8(wire).expect("invariant: UTF-8");
            assert!(wire.starts_with(&format!("HTTP/1.1 {status} ")), "{wire}");
            assert_eq!(wire.matches("HTTP/1.1").count(), 1);
            let value: serde_json::Value =
                serde_json::from_str(wire.split_once("\r\n\r\n").expect("invariant: HTTP").1)
                    .expect("invariant: JSON");
            assert_eq!(value.get("type").is_some(), is_anthropic_route(route));
        }
    }

    #[test]
    fn count_tokens_auth_and_oversized_body_keep_anthropic_envelope() {
        let raw = "POST /v1/messages/count_tokens?x=1 HTTP/1.1\r\nx-api-key: secret\r\nContent-Length: 0\r\n\r\n";
        let request = read_request(
            &mut std::io::Cursor::new(raw),
            std::time::Instant::now() + std::time::Duration::from_secs(5),
        )
        .expect("invariant: lecture")
        .expect("invariant: requête");
        assert!(is_authorized(&request, "secret"));
        let front = Front::new(Some("other".into()), vec![]);
        let mut wire = Vec::new();
        route_front(&mut wire, &front, &request)
            .expect("invariant: refus")
            .expect("invariant: écriture");
        assert!(String::from_utf8(wire)
            .expect("invariant: UTF-8")
            .contains(r#""type":"authentication_error""#));
        let raw = "POST /v1/messages/count_tokens HTTP/1.1\r\nContent-Length: 16777217\r\n\r\n";
        let error = read_request(
            &mut std::io::Cursor::new(raw),
            std::time::Instant::now() + std::time::Duration::from_secs(5),
        )
        .expect_err("invariant: trop grand");
        assert!(matches!(
            error,
            ServeError::BodyTooLarge { anthropic: true }
        ));
    }

    #[test]
    fn model_lookup_decodes_path_once_and_rejects_invalid_encoding() {
        let ids = ["mon modèle", "日本語", "a/b?c#d+e%f", "%20", "a+b"];
        let front = Front::new(None, ids.iter().map(|id| (*id).to_string()).collect());
        for (suffix, status, expected) in [
            ("mon%20mod%C3%A8le", 200, "mon modèle"),
            ("%E6%97%A5%E6%9C%AC%E8%AA%9E", 200, "日本語"),
            ("a%2fb%3Fc%23d%2Be%25f?ignored=1", 200, "a/b?c#d+e%f"),
            ("%2520", 200, "%20"),
            ("a+b", 200, "a+b"),
            ("mon%2520mod%C3%A8le", 404, ""),
            ("%", 400, ""),
            ("%2", 400, ""),
            ("%GG", 400, ""),
            ("%FF", 400, ""),
            ("%C3%28", 400, ""),
        ] {
            let raw = format!("GET /v1/models/{suffix} HTTP/1.1\r\n\r\n");
            let request = read_request(&mut Cursor::new(raw), far_deadline())
                .expect("invariant: requête lisible")
                .expect("invariant: requête présente");
            let mut reply = Vec::new();
            route_front(&mut reply, &front, &request)
                .expect("invariant: route modèle traitée")
                .expect("invariant: réponse écrite");
            let reply = String::from_utf8(reply).expect("invariant: réponse UTF-8");
            assert!(
                reply.starts_with(&format!("HTTP/1.1 {status} ")),
                "{suffix}: {reply}"
            );
            let (_, body) = reply.split_once("\r\n\r\n").expect("invariant: corps HTTP");
            let body: serde_json::Value =
                serde_json::from_str(body).expect("invariant: corps JSON");
            if status == 200 {
                assert_eq!(body["id"], expected);
            } else {
                assert_eq!(body["error"]["type"], "invalid_request_error");
            }
        }
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
    fn chat_session_header_is_available_for_request_namespace() {
        let body = br#"{"model":"reti-35b","messages":[],"user":"body-user"}"#;
        let raw = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nX-Saragossa-Session: header-user\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            str::from_utf8(body).expect("invariant: JSON test UTF-8")
        );
        let mut cursor = Cursor::new(raw.into_bytes());

        let request = read_request(&mut cursor, far_deadline())
            .expect("invariant: requête lisible")
            .expect("invariant: requête présente");
        let chat: ChatCompletionRequest =
            serde_json::from_slice(&request.body).expect("invariant: JSON chat valide");

        assert_eq!(
            chat.session_key(request.header(SESSION_HEADER)).as_deref(),
            Some("header-user")
        );
    }

    fn post(route: &str, body: &str) -> HttpRequest {
        let raw = format!(
            "POST {route} HTTP/1.1\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        read_request(&mut Cursor::new(raw.into_bytes()), far_deadline())
            .expect("invariant: requête lisible")
            .expect("invariant: requête présente")
    }

    fn admit(request: &HttpRequest) -> Option<BatchChat> {
        batch_chat(request, 1024, |_| false)
    }

    /// Seules les requêtes chat streamées greedy en texte libre entrent dans
    /// un lot ; le reste garde le chemin solo, qui rapporte ses erreurs.
    #[test]
    fn batch_chat_admits_only_streamed_greedy_text() {
        let chat = |extra: &str| {
            post(
                "/v1/chat/completions",
                &format!(r#"{{"model":"m","messages":[],"stream":true{extra}}}"#),
            )
        };

        let admitted = admit(&chat("")).expect("invariant: chat streamé greedy admis");
        assert_eq!(admitted.model(), "m");
        assert!(!admitted.anthropic);
        assert!(admit(&chat(r#","temperature":0"#)).is_some());
        assert!(admit(&chat(r#","temperature":0.7"#)).is_none());
        assert!(admit(&chat(r#","response_format":{"type":"json_object"}"#)).is_none());
        assert!(admit(&chat(r#","max_tokens":0"#)).is_none());
        assert!(admit(&chat(r#","tool_choice":"required""#)).is_none());
        // Invalide : le solo répond 400 sans charger le modèle.
        assert!(admit(&chat(r#","max_tokens":1025"#)).is_none());
        assert!(admit(&chat(&format!(r#","stop":{:?}"#, vec![""; 17]))).is_none());
        // Avec session : solo, pour garder le cache de préfixe par blocs.
        assert!(admit(&chat(r#","user":"alice""#)).is_none());
        assert!(admit(&post(
            "/v1/chat/completions",
            r#"{"model":"m","messages":[]}"#
        ))
        .is_none());
        assert!(admit(&post("/v1/chat/completions", "{")).is_none());
        assert!(admit(&post("/v1/embeddings", r#"{"model":"m","input":"x"}"#)).is_none());
    }

    /// Avec session, le lot n'admet que ce que le solo servirait en MTP, qui
    /// ignore déjà la session : modèle MTP chargé, `stop` compris.
    #[test]
    fn batch_chat_admits_session_only_when_solo_would_route_mtp() {
        let chat = |extra: &str| {
            post(
                "/v1/chat/completions",
                &format!(r#"{{"model":"m","messages":[],"stream":true,"user":"alice"{extra}}}"#),
            )
        };
        let mtp = |model: &str| model == "m";

        assert!(batch_chat(&chat(""), 1024, mtp).is_some());
        assert!(batch_chat(&chat(""), 1024, |_| false).is_none());
        assert!(batch_chat(&chat(r#","stop":"fin""#), 1024, mtp).is_some());
        assert!(batch_chat(&chat(r#","stop":"fin""#), 1024, |_| false).is_none());
        let body = r#"{"model":"m","max_tokens":8,"stream":true,"messages":[{"role":"user","content":"x"}]}"#;
        let raw = format!(
            "POST /v1/messages HTTP/1.1\r\n{SESSION_HEADER}: s1\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        let anthropic = read_request(&mut Cursor::new(raw.into_bytes()), far_deadline())
            .expect("invariant: requête lisible")
            .expect("invariant: requête présente");
        assert!(batch_chat(&anthropic, 1024, |_| false).is_none());
        assert!(batch_chat(&anthropic, 1024, mtp).is_some());
    }

    #[test]
    fn batch_chat_admits_streamed_anthropic_messages() {
        let body = r#"{"model":"m","max_tokens":8,"stream":true,"messages":[{"role":"user","content":"x"}]}"#;

        let admitted =
            admit(&post("/v1/messages", body)).expect("invariant: Messages streamé admis");

        assert_eq!(admitted.model(), "m");
        assert!(admitted.anthropic);
        assert!(admit(&post("/v1/messages", &body.replace("true", "false"))).is_none());
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

    fn response_json(bytes: Vec<u8>, status: u16) -> serde_json::Value {
        let response = String::from_utf8(bytes).expect("invariant: réponse UTF-8");
        assert!(
            response.starts_with(&format!("HTTP/1.1 {status} ")),
            "{response}"
        );
        serde_json::from_str(
            response
                .split_once("\r\n\r\n")
                .expect("invariant: corps HTTP")
                .1,
        )
        .expect("invariant: JSON réponse")
    }

    #[test]
    fn chat_rejects_multiple_choices_but_accepts_one_or_absent() {
        let args = ServeArgs::parse(Vec::<String>::new()).expect("invariant: args valides");
        let mut state = ServeState::new(&args);
        for n in [None, Some(1), Some(2), Some(0)] {
            for streaming in [false, true] {
                let mut body =
                    serde_json::json!({"model":"missing", "messages":[], "stream":streaming});
                if let Some(n) = n {
                    body["n"] = n.into();
                }
                let request = post("/v1/chat/completions", &body.to_string());
                let mut response = Vec::new();
                handle_job(&mut response, &mut state, &request)
                    .expect_err("invariant: pas de modèle chargé");
                let rejected = n.is_some_and(|n| n != 1);
                let value = response_json(response, if rejected { 400 } else { 404 });
                if rejected {
                    assert_eq!(value["error"]["type"], "invalid_request_error");
                    assert_eq!(value["error"]["param"], "n");
                    assert!(batch_chat(&request, 4096, |_| false).is_none());
                } else {
                    assert_eq!(value["error"]["code"], "model_not_found");
                }
            }
        }
    }

    #[test]
    fn auxiliary_only_startup_keeps_llm_routes_explicitly_unavailable() {
        for flags in [
            vec!["--embed-model"],
            vec!["--stt-model"],
            vec!["--tts-model"],
            vec!["--stt-model", "--tts-model", "--embed-model"],
        ] {
            let mut args = ServeArgs::parse(
                flags
                    .into_iter()
                    .flat_map(|flag| [flag.to_string(), "unused".into()]),
            )
            .expect("invariant: options auxiliaires valides");
            super::super::model_selection::select_if_missing(&mut args)
                .expect("invariant: aucun sélecteur LLM pour les modèles auxiliaires");
            super::super::model_selection::resolve_models(&mut args)
                .expect("invariant: aucun modèle distant à résoudre");
            let mut state = ServeState::new(&args);
            state.preload().expect("invariant: aucun LLM à précharger");
            assert!(state.model_ids().is_empty());
            for route in ["/v1/chat/completions", "/v1/messages"] {
                for streaming in [false, true] {
                    let request = post(
                        route,
                        &serde_json::json!({"model":"missing", "messages":[],
                        "max_tokens":1, "stream":streaming})
                        .to_string(),
                    );
                    let mut response = Vec::new();
                    let _ = handle_job(&mut response, &mut state, &request);
                    let value = response_json(response, 404);
                    assert!(value["error"]["message"]
                        .as_str()
                        .is_some_and(|s| s.contains("modèle non chargé")));
                    assert_eq!(value["error"]["code"], "model_not_found");
                    assert_eq!(value.get("type").is_some(), route == "/v1/messages");
                }
            }
        }
    }

    #[test]
    fn unknown_model_has_code_in_both_envelopes_before_streaming() {
        let args = ServeArgs::parse(["--model".into(), "known=unused".into()])
            .expect("invariant: modèle déclaré sans chargement");
        let mut state = ServeState::new(&args);
        for route in ["/v1/chat/completions", "/v1/messages"] {
            for streaming in [false, true] {
                let request = post(
                    route,
                    &serde_json::json!({"model":"missing", "messages":[],
                    "max_tokens":1, "stream":streaming})
                    .to_string(),
                );
                let mut response = Vec::new();
                let _ = handle_job(&mut response, &mut state, &request);
                let value = response_json(response, 404);
                assert_eq!(value["error"]["code"], "model_not_found");
                if route == "/v1/messages" {
                    assert_eq!(value["type"], "error");
                    assert_eq!(value["error"]["type"], "not_found_error");
                } else {
                    assert_eq!(value["error"]["type"], "invalid_request_error");
                    assert_eq!(value["error"]["param"], "model");
                    assert!(value.get("type").is_none());
                }
            }
        }
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
    fn chat_json_schema_response_format_returns_501() {
        let args = ServeArgs::parse(Vec::<String>::new()).expect("invariant: args valides");
        let mut state = ServeState::new(&args);
        let body =
            br#"{"model":"reti-35b","messages":[],"response_format":{"type":"json_schema"}}"#;
        let raw = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            str::from_utf8(body).expect("invariant: JSON test UTF-8")
        );
        let mut stream = Cursor::new(raw.into_bytes());

        let error = handle_connection(&mut stream, &mut state, None, far_deadline())
            .expect_err("invariant: json_schema refusé");

        assert!(error.to_string().contains("json_schema"));
        let response = String::from_utf8(stream.into_inner()).expect("invariant: réponse UTF-8");
        assert!(response.contains("HTTP/1.1 501 Not Implemented"));
        assert!(response.contains("json_schema"));
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

    #[test]
    fn unsupported_sampling_returns_permanent_400() {
        for (field, value) in [
            ("min_p", 0.1),
            ("presence_penalty", 0.5),
            ("frequency_penalty", -1.0),
            ("repetition_penalty", 1.1),
        ] {
            let args = ServeArgs::parse(Vec::<String>::new()).expect("invariant: args valides");
            let mut state = ServeState::new(&args);
            let body = serde_json::json!({"model":"m", "messages":[], "max_tokens":1,
                field: value})
            .to_string();
            let raw = format!(
                "POST /v1/chat/completions HTTP/1.1\r\nContent-Length: {}\r\n\r\n{}",
                body.len(),
                body
            );
            let mut stream = Cursor::new(raw.into_bytes());
            let error = handle_connection(&mut stream, &mut state, None, far_deadline())
                .expect_err("invariant: paramètre non implémenté");
            assert!(error.to_string().contains(field));
            let response = String::from_utf8(stream.into_inner()).expect("invariant: UTF-8");
            let start = response
                .rfind("HTTP/1.1 400 Bad Request")
                .expect("invariant: erreur permanente 400");
            let (_, body) = response[start..]
                .split_once("\r\n\r\n")
                .expect("invariant: corps HTTP");
            let value: serde_json::Value = serde_json::from_str(body).expect("invariant: JSON");
            assert_eq!(value["error"]["type"], "invalid_request_error");
        }
    }

    #[test]
    fn sse_identity_is_shared_by_text_reasoning_tools_finish_and_usage() {
        for flint in [false, true] {
            let mut bytes = Vec::new();
            let parser = flint.then(|| {
                ToolOutputParser::new(
                    super::super::tool_output::ToolFamily::Flint,
                    vec![serde_json::json!({
                        "type":"function", "function":{"name":"heure"}
                    })],
                    true,
                )
            });
            let mut sink = OpenAiSse::new(&mut bytes, parser, true);
            // Une date indépendante de l'horloge révèle tout chunk qui
            // recréerait ses métadonnées, sans attente dans le test.
            sink.created = 42;
            let start = super::super::streaming::StreamingCompletionStart {
                model: "m".into(),
                prompt_tokens: 1,
                reused_prefix_tokens: 0,
                prefill: Duration::ZERO,
            };
            sink.event(CompletionStreamEvent::Start(&start))
                .expect("invariant: début SSE");
            let delta = if flint {
                "<think>raison</think>texte<tool_call>heure</tool_call>"
            } else {
                "texte"
            };
            sink.event(CompletionStreamEvent::Delta(delta))
                .expect("invariant: delta SSE");
            sink.finish(Ok(ServedCompletion {
                model: "m".into(),
                content: String::new(),
                finish_reason: "stop",
                matched_stop: None,
                usage: super::super::protocol::Usage::new(1, 1),
                prompt_tokens: 1,
                reused_prefix_tokens: 0,
                timings: saragossa::GenerationTimings::default(),
                total: Duration::ZERO,
            }))
            .expect("invariant: fin SSE");
            let response = String::from_utf8(bytes).expect("invariant: SSE UTF-8");
            let frames: Vec<serde_json::Value> = response
                .lines()
                .filter_map(|line| line.strip_prefix("data: "))
                .filter(|data| *data != "[DONE]")
                .map(|data| serde_json::from_str(data).expect("invariant: chunk JSON"))
                .collect();
            assert_eq!(frames.len(), if flint { 6 } else { 4 });
            for chunk in &frames {
                assert_eq!(chunk["id"], frames[0]["id"]);
                assert_eq!(chunk["created"], 42);
            }
            if flint {
                assert!(frames
                    .iter()
                    .any(|c| c["choices"][0]["delta"]["reasoning_content"].is_string()));
                assert!(frames
                    .iter()
                    .any(|c| c["choices"][0]["delta"]["tool_calls"].is_array()));
            }
            assert_eq!(response.matches("data: [DONE]").count(), 1);
        }
    }

    #[test]
    fn openai_stream_converts_qwen_and_gemma_calls_without_leaking_markers() {
        use super::super::tool_output::ToolFamily;
        for (family, text) in [
            (ToolFamily::Qwen, "Avant <tool_call><function=weather><parameter=city>Paris</parameter></function></tool_call>"),
            (ToolFamily::Gemma4, "Avant <|tool_call>call:weather{city:<|\"|>Paris<|\"|>}<tool_call|>"),
        ] {
            let tools = vec![serde_json::json!({"type":"function","function":{"name":"weather","parameters":{"properties":{"city":{"type":"string"}}}}})];
            let expected = ToolOutputParser::new(family, tools.clone(), true).parse(text);
            let mut bytes = Vec::new();
            let mut sink = OpenAiSse::new(&mut bytes, Some(ToolOutputParser::new(family, tools, true)), false);
            let start = super::super::streaming::StreamingCompletionStart { model:"m".into(), prompt_tokens:7, reused_prefix_tokens:0, prefill:Duration::ZERO };
            sink.event(CompletionStreamEvent::Start(&start)).expect("start");
            for ch in text.chars() { sink.event(CompletionStreamEvent::Delta(&ch.to_string())).expect("delta"); }
            sink.finish(Ok(ServedCompletion { model:"m".into(), content:text.into(), finish_reason:"stop", matched_stop:None, usage:super::super::protocol::Usage::new(7,3), prompt_tokens:7, reused_prefix_tokens:0, timings:saragossa::GenerationTimings::default(), total:Duration::ZERO })).expect("finish");
            let response = String::from_utf8(bytes).expect("utf8");
            let frames: Vec<serde_json::Value> = response.split_once("\r\n\r\n").expect("headers").1.split("\n\n").filter_map(|f| f.strip_prefix("data: ")).filter(|f| *f != "[DONE]").map(|f| serde_json::from_str(f).expect("frame")).collect();
            let content: String = frames.iter().filter_map(|f| f["choices"][0]["delta"]["content"].as_str()).collect();
            assert_eq!(content, expected.content);
            let calls: Vec<_> = frames.iter().filter_map(|f| f["choices"][0]["delta"]["tool_calls"].as_array()).flatten().collect();
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0]["function"]["arguments"], expected.tool_calls[0].function.arguments);
            assert_eq!(calls[0]["function"]["name"], "weather");
            assert_eq!(calls[0]["index"], 0);
            assert!(calls[0]["id"].as_str().is_some_and(|s| !s.is_empty()));
            assert_eq!(frames.last().expect("finish")["choices"][0]["finish_reason"], "tool_calls");
            assert!(frames.iter().all(|f| f["id"] == frames[0]["id"]));
        }
    }

    /// `stream_options.include_usage` : chunk d'usage entre le chunk de fin
    /// et `[DONE]` ; sans l'option, le flux reste inchangé.
    #[test]
    fn openai_sse_writes_usage_chunk_only_on_request() {
        let completion = || ServedCompletion {
            model: "m".to_string(),
            content: "ok".to_string(),
            finish_reason: "stop",
            matched_stop: None,
            usage: super::super::protocol::Usage::new(7, 3),
            prompt_tokens: 7,
            reused_prefix_tokens: 0,
            timings: saragossa::GenerationTimings::default(),
            total: Duration::from_millis(1),
        };
        let frames = |include_usage: bool| -> Vec<serde_json::Value> {
            let mut bytes = Vec::new();
            let mut sink = OpenAiSse::new(&mut bytes, None, include_usage);
            sink.created = 42;
            let start = super::super::streaming::StreamingCompletionStart {
                model: "m".into(),
                prompt_tokens: 7,
                reused_prefix_tokens: 0,
                prefill: Duration::ZERO,
            };
            sink.event(CompletionStreamEvent::Start(&start))
                .expect("invariant: début HTTP en mémoire");
            sink.event(CompletionStreamEvent::Delta("ok"))
                .expect("invariant: delta en mémoire");
            sink.finish(Ok(completion()))
                .expect("invariant: écriture en mémoire");
            let response = String::from_utf8(bytes).expect("invariant: SSE UTF-8");
            assert!(response.starts_with("HTTP/1.1 200 OK\r\n"));
            assert!(response.ends_with("data: [DONE]\n\n"));
            response
                .split_once("\r\n\r\n")
                .expect("invariant: corps HTTP")
                .1
                .split("\n\n")
                .filter_map(|frame| frame.strip_prefix("data: "))
                .filter(|data| *data != "[DONE]")
                .map(|data| serde_json::from_str(data).expect("invariant: data JSON"))
                .collect()
        };

        let with_usage = frames(true);
        assert_eq!(with_usage.len(), 4);
        assert_eq!(with_usage[0]["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(with_usage[1]["choices"][0]["delta"]["content"], "ok");
        assert_eq!(with_usage[2]["choices"][0]["finish_reason"], "stop");
        for chunk in &with_usage[..3] {
            assert_eq!(chunk.get("usage"), Some(&serde_json::Value::Null));
            assert_eq!(chunk["id"], with_usage[3]["id"]);
            assert_eq!(chunk["created"], 42);
        }
        assert_eq!(with_usage[3]["created"], 42);
        assert_eq!(with_usage[3]["choices"], serde_json::json!([]));
        assert_eq!(with_usage[3]["usage"]["prompt_tokens"], 7);
        assert_eq!(with_usage[3]["usage"]["completion_tokens"], 3);
        assert_eq!(frames(false).len(), 3);
        for chunk in [
            ChatCompletionChunk::role("m"),
            ChatCompletionChunk::content("m", "ok".into()),
            ChatCompletionChunk::reasoning("m", "r".into()),
        ] {
            let value = serde_json::to_value(chunk.with_usage_option(true))
                .expect("invariant: chunk sérialisable");
            assert_eq!(value.get("usage"), Some(&serde_json::Value::Null));
        }
        let non_stream: ChatCompletionRequest = serde_json::from_str(
            r#"{"model":"m","messages":[],"stream_options":{"include_usage":"ignored"}}"#,
        )
        .expect("invariant: options non streamées ignorées");
        assert!(non_stream.validate(1024).is_ok());
        assert!(!non_stream.include_usage());
        for (options, expected) in [
            ("", false),
            (r#", "stream_options": null"#, false),
            (r#", "stream_options": {}"#, false),
            (r#", "stream_options": {"include_usage": null}"#, false),
            (r#", "stream_options": {"include_usage": true}"#, true),
            (r#", "stream_options": {"include_usage": false}"#, false),
        ] {
            for stream in [false, true] {
                let http = post(
                    "/v1/chat/completions",
                    &format!(r#"{{"model":"m","messages":[],"stream":{stream}{options}}}"#),
                );
                let request = ChatCompletionRequest::parse(&http.body)
                    .expect("invariant: options tolérées sur HTTP");
                assert!(request.validate(1024).is_ok());
                let chunks = frames(request.include_usage());
                assert_eq!(chunks.len(), if stream && expected { 4 } else { 3 });
                for chunk in &chunks[..3] {
                    assert_eq!(chunk.get("usage").is_some(), stream && expected);
                }
            }
        }
    }
}
