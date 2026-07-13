//! Types JSON OpenAI compatibles pour `/v1`.

use std::time::{SystemTime, UNIX_EPOCH};

use saragossa::ChatTemplateMessage;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::error::{ServeError, ServeResult};

/// Requête `/v1/chat/completions`.
#[derive(Clone, Debug, Deserialize)]
pub(super) struct ChatCompletionRequest {
    /// Identifiant du modèle demandé.
    pub(super) model: String,
    /// Historique role-tagged.
    pub(super) messages: Vec<WireMessage>,
    /// Active le flux SSE.
    #[serde(default)]
    pub(super) stream: bool,
    /// Budget de génération.
    #[serde(default)]
    pub(super) max_tokens: Option<usize>,
    /// Alias OpenAI récent du budget.
    #[serde(default)]
    pub(super) max_completion_tokens: Option<usize>,
    /// Température de sampling.
    #[serde(default)]
    pub(super) temperature: Option<f32>,
    /// Nucleus sampling.
    #[serde(default)]
    pub(super) top_p: Option<f32>,
    /// Extension locale pour le top-k.
    #[serde(default)]
    pub(super) top_k: Option<usize>,
    /// Stop textuel OpenAI.
    #[serde(default)]
    pub(super) stop: Option<StopSpec>,
    /// Format de réponse OpenAI.
    #[serde(default)]
    pub(super) response_format: Option<ResponseFormat>,
    /// Identifiant utilisateur OpenAI, utilisé comme session de repli.
    #[serde(default)]
    pub(super) user: Option<String>,
}

impl ChatCompletionRequest {
    /// Renvoie le budget effectif.
    pub(super) fn max_tokens(&self) -> usize {
        self.max_tokens
            .or(self.max_completion_tokens)
            .unwrap_or(512)
    }

    /// Renvoie le budget effectif après plafond serveur.
    pub(super) fn max_tokens_capped(&self, cap: usize) -> ServeResult<usize> {
        let requested = self.max_tokens();
        if requested > cap {
            return Err(ServeError::Http(format!(
                "max_tokens {requested} dépasse le plafond serveur {cap}"
            )));
        }
        Ok(requested)
    }

    /// Convertit les messages wire en messages de template.
    pub(super) fn template_messages(&self) -> Vec<ChatTemplateMessage> {
        self.messages
            .iter()
            .map(|message| ChatTemplateMessage {
                role: message.role.clone(),
                content: Some(message.content_text()),
            })
            .collect()
    }

    /// Renvoie les chaînes stop demandées.
    pub(super) fn stop_texts(&self) -> Vec<String> {
        match &self.stop {
            Some(StopSpec::One(value)) => vec![value.clone()],
            Some(StopSpec::Many(values)) => values.clone(),
            None => Vec::new(),
        }
    }

    /// Renvoie le mode de sortie structuré demandé.
    pub(super) fn response_format_mode(&self) -> ServeResult<ResponseFormatMode> {
        let Some(format) = &self.response_format else {
            return Ok(ResponseFormatMode::Text);
        };
        match format.kind.as_str() {
            "text" => Ok(ResponseFormatMode::Text),
            "json_object" => Ok(ResponseFormatMode::JsonObject),
            "json_schema" => Err(ServeError::not_implemented(
                "response_format json_schema n'est pas encore supporté (v1: json_object)",
            )),
            other => Err(ServeError::Http(format!(
                "response_format type inconnu: {other}"
            ))),
        }
    }

    /// Dérive la clé de session effective.
    pub(super) fn session_key(&self, header: Option<&str>) -> Option<String> {
        header
            .and_then(normalize_session_key)
            .or_else(|| self.user.as_deref().and_then(normalize_session_key))
            .map(ToString::to_string)
    }
}

pub(super) fn normalize_session_key(value: &str) -> Option<&str> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed)
    }
}

/// Mode de sortie structuré supporté par `serve`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ResponseFormatMode {
    /// Comportement texte existant.
    Text,
    /// Objet JSON racine contraint par le sampler.
    JsonObject,
}

/// Objet `response_format` OpenAI minimal.
#[derive(Clone, Debug, Deserialize)]
pub(super) struct ResponseFormat {
    /// Type demandé (`text`, `json_object`, `json_schema`).
    #[serde(rename = "type")]
    kind: String,
}

/// Message OpenAI minimal.
#[derive(Clone, Debug, Deserialize)]
pub(super) struct WireMessage {
    /// Rôle du message.
    pub(super) role: String,
    /// Contenu OpenAI: string ou segments.
    #[serde(default)]
    pub(super) content: Option<Value>,
}

impl WireMessage {
    fn content_text(&self) -> String {
        match &self.content {
            Some(Value::String(text)) => text.clone(),
            Some(Value::Array(items)) => items
                .iter()
                .filter_map(text_part)
                .collect::<Vec<_>>()
                .join(""),
            Some(Value::Null) | None => String::new(),
            Some(other) => other.to_string(),
        }
    }
}

fn text_part(value: &Value) -> Option<String> {
    if let Some(text) = value.as_str() {
        return Some(text.to_string());
    }
    let object = value.as_object()?;
    match object.get("type").and_then(Value::as_str) {
        Some("text") | Some("input_text") => object
            .get("text")
            .and_then(Value::as_str)
            .map(ToString::to_string),
        _ => None,
    }
}

/// Champ `stop` OpenAI.
#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub(super) enum StopSpec {
    /// Une seule chaîne stop.
    One(String),
    /// Plusieurs chaînes stop.
    Many(Vec<String>),
}

/// Réponse `/v1/models`.
#[derive(Debug, Serialize)]
pub(super) struct ModelsResponse {
    object: &'static str,
    data: Vec<ModelInfo>,
}

impl ModelsResponse {
    /// Construit une réponse models.
    pub(super) fn new(models: Vec<ModelInfo>) -> Self {
        Self {
            object: "list",
            data: models,
        }
    }
}

/// Réponse `/health`.
///
/// Statut seul, volontairement : la liste des modèles vit derrière le bearer
/// (`/v1/models`) — l'exemption d'auth du health-check ne doit rien fuiter.
#[derive(Debug, Serialize)]
pub(super) struct HealthResponse {
    status: &'static str,
}

impl HealthResponse {
    /// Construit la réponse de santé nominale.
    pub(super) fn ok() -> Self {
        Self { status: "ok" }
    }
}

/// Entrée de modèle OpenAI.
#[derive(Clone, Debug, Serialize)]
pub(super) struct ModelInfo {
    id: String,
    object: &'static str,
    created: u64,
    owned_by: &'static str,
}

impl ModelInfo {
    /// Construit une entrée de modèle.
    pub(super) fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            object: "model",
            created: created_now(),
            owned_by: "reti",
        }
    }
}

/// Usage OpenAI.
#[derive(Clone, Copy, Debug, Serialize)]
pub(super) struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

impl Usage {
    /// Construit un usage token.
    pub(super) fn new(prompt_tokens: usize, completion_tokens: usize) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        }
    }

    /// Renvoie les tokens d'entrée.
    pub(super) fn prompt_tokens(&self) -> usize {
        self.prompt_tokens
    }

    /// Renvoie les tokens de sortie.
    pub(super) fn completion_tokens(&self) -> usize {
        self.completion_tokens
    }
}

/// Réponse non-streaming.
#[derive(Debug, Serialize)]
pub(super) struct ChatCompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChatChoice>,
    usage: Usage,
}

impl ChatCompletionResponse {
    /// Construit une réponse complète.
    pub(super) fn new(model: &str, content: String, finish_reason: &str, usage: Usage) -> Self {
        Self {
            id: response_id("chatcmpl"),
            object: "chat.completion",
            created: created_now(),
            model: model.to_string(),
            choices: vec![ChatChoice {
                index: 0,
                message: Some(AssistantMessage {
                    role: "assistant",
                    content,
                }),
                delta: None,
                finish_reason: Some(finish_reason.to_string()),
            }],
            usage,
        }
    }
}

#[derive(Debug, Serialize)]
struct AssistantMessage {
    role: &'static str,
    content: String,
}

#[derive(Debug, Serialize)]
struct ChatChoice {
    index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    message: Option<AssistantMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    delta: Option<DeltaMessage>,
    finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
struct DeltaMessage {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

/// Chunk SSE OpenAI.
#[derive(Debug, Serialize)]
pub(super) struct ChatCompletionChunk {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChatChoice>,
}

impl ChatCompletionChunk {
    /// Construit le chunk de rôle initial.
    pub(super) fn role(model: &str) -> Self {
        Self::chunk(
            model,
            DeltaMessage {
                role: Some("assistant"),
                content: None,
            },
            None,
        )
    }

    /// Construit un chunk de contenu.
    pub(super) fn content(model: &str, content: String) -> Self {
        Self::chunk(
            model,
            DeltaMessage {
                role: None,
                content: Some(content),
            },
            None,
        )
    }

    /// Construit le chunk final.
    pub(super) fn done(model: &str, finish_reason: &str) -> Self {
        Self {
            id: response_id("chatcmpl"),
            object: "chat.completion.chunk",
            created: created_now(),
            model: model.to_string(),
            choices: vec![ChatChoice {
                index: 0,
                message: None,
                delta: Some(DeltaMessage {
                    role: None,
                    content: None,
                }),
                finish_reason: Some(finish_reason.to_string()),
            }],
        }
    }

    fn chunk(model: &str, delta: DeltaMessage, finish_reason: Option<String>) -> Self {
        Self {
            id: response_id("chatcmpl"),
            object: "chat.completion.chunk",
            created: created_now(),
            model: model.to_string(),
            choices: vec![ChatChoice {
                index: 0,
                message: None,
                delta: Some(delta),
                finish_reason,
            }],
        }
    }
}

/// Evénement d'erreur terminale pour le streaming OpenAI.
#[derive(Debug, Serialize)]
pub(super) struct StreamErrorEvent<'a> {
    error: StreamErrorPayload<'a>,
}

impl<'a> StreamErrorEvent<'a> {
    /// Construit une erreur terminale SSE.
    pub(super) fn new(message: &'a str, error_type: &'a str) -> Self {
        Self {
            error: StreamErrorPayload {
                message,
                kind: error_type,
            },
        }
    }
}

#[derive(Debug, Serialize)]
struct StreamErrorPayload<'a> {
    message: &'a str,
    #[serde(rename = "type")]
    kind: &'a str,
}

/// Sérialise en JSON bytes.
pub(super) fn json_bytes<T: Serialize>(value: &T) -> ServeResult<Vec<u8>> {
    serde_json::to_vec(value).map_err(|e| ServeError::json("sérialisation réponse", e))
}

/// Sérialise un chunk SSE.
pub(super) fn sse_event<T: Serialize>(value: &T) -> ServeResult<Vec<u8>> {
    let mut bytes = b"data: ".to_vec();
    bytes.extend(json_bytes(value)?);
    bytes.extend_from_slice(b"\n\n");
    Ok(bytes)
}

/// Renvoie l'événement SSE final.
pub(super) fn sse_done() -> Vec<u8> {
    b"data: [DONE]\n\n".to_vec()
}

fn created_now() -> u64 {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(duration) => duration.as_secs(),
        Err(_) => 0,
    }
}

fn response_id(prefix: &str) -> String {
    format!("{prefix}-{}", created_now())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_accepts_string_stop() {
        let req: ChatCompletionRequest = serde_json::from_str(
            r#"{"model":"reti-35b","messages":[],"stop":"\n\n","max_tokens":4}"#,
        )
        .expect("invariant: JSON valide");

        assert_eq!(req.stop_texts(), vec!["\n\n".to_string()]);
        assert_eq!(req.max_tokens(), 4);
        assert_eq!(
            req.max_tokens_capped(4)
                .expect("invariant: max_tokens sous plafond"),
            4
        );
    }

    #[test]
    fn request_rejects_tokens_above_cap() {
        let req: ChatCompletionRequest =
            serde_json::from_str(r#"{"model":"reti-35b","messages":[],"max_tokens":4097}"#)
                .expect("invariant: JSON valide");

        let error = req
            .max_tokens_capped(4096)
            .expect_err("invariant: cap dépassé");
        assert!(error.to_string().contains("plafond serveur 4096"));
    }

    #[test]
    fn request_extracts_segmented_text() {
        let req: ChatCompletionRequest = serde_json::from_str(
            r#"{"model":"reti-35b","messages":[{"role":"user","content":[{"type":"text","text":"bon"},{"type":"text","text":"jour"}]}]}"#,
        )
        .expect("invariant: JSON valide");

        let messages = req.template_messages();
        assert_eq!(messages[0].content.as_deref(), Some("bonjour"));
    }

    #[test]
    fn request_accepts_user_as_session_fallback() {
        let req: ChatCompletionRequest =
            serde_json::from_str(r#"{"model":"reti-35b","messages":[],"user":" agent-a "}"#)
                .expect("invariant: JSON valide");

        assert_eq!(req.session_key(None).as_deref(), Some("agent-a"));
    }

    #[test]
    fn request_session_header_overrides_user() {
        let req: ChatCompletionRequest =
            serde_json::from_str(r#"{"model":"reti-35b","messages":[],"user":"agent-a"}"#)
                .expect("invariant: JSON valide");

        assert_eq!(req.session_key(Some("agent-b")).as_deref(), Some("agent-b"));
    }

    #[test]
    fn chunk_serializes_openai_shape() {
        let bytes = json_bytes(&ChatCompletionChunk::content(
            "reti-35b",
            "salut".to_string(),
        ))
        .expect("invariant: chunk sérialisable");
        let value: Value = serde_json::from_slice(&bytes).expect("invariant: JSON chunk sérialisé");

        assert_eq!(value["object"], "chat.completion.chunk");
        assert_eq!(value["choices"][0]["delta"]["content"], "salut");
    }

    #[test]
    fn stream_error_event_serializes_openai_error_shape() {
        let bytes = sse_event(&StreamErrorEvent::new(
            "json incomplet (max_tokens)",
            "incomplete_json",
        ))
        .expect("invariant: erreur SSE sérialisable");
        let text = String::from_utf8(bytes).expect("invariant: SSE UTF-8");
        let data = text
            .strip_prefix("data: ")
            .and_then(|value| value.strip_suffix("\n\n"))
            .expect("invariant: frame SSE data");
        let value: Value = serde_json::from_str(data).expect("invariant: data JSON");

        assert_eq!(value["error"]["type"], "incomplete_json");
        assert!(value["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("max_tokens")));
    }
}
