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
    /// Outils OpenAI (`{"type":"function","function":{...}}`), rendus et
    /// parsés pour flint seulement.
    #[serde(default)]
    pub(super) tools: Option<Vec<Value>>,
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
                reasoning_content: message.reasoning_content.clone(),
                tool_calls: message.tool_calls.clone().unwrap_or_default(),
            })
            .collect()
    }

    /// Renvoie les outils déclarés par la requête.
    pub(super) fn tools(&self) -> &[Value] {
        self.tools.as_deref().unwrap_or_default()
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
            "json_lines" => Ok(ResponseFormatMode::JsonLines),
            "json_schema" => Err(ServeError::not_implemented(
                "response_format json_schema n'est pas encore supporté (v1: json_object, extension saragossa json_lines)",
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
    /// Suite NDJSON saragossa, un objet JSON racine complet par ligne.
    JsonLines,
}

/// Objet `response_format` OpenAI minimal.
#[derive(Clone, Debug, Deserialize)]
pub(super) struct ResponseFormat {
    /// Type demandé (`text`, `json_object`, `json_lines`, `json_schema`).
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
    /// Raisonnement d'un tour assistant passé.
    #[serde(default)]
    pub(super) reasoning_content: Option<String>,
    /// Appels d'outils d'un tour assistant passé.
    #[serde(default)]
    pub(super) tool_calls: Option<Vec<Value>>,
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
    pub(super) fn new(
        model: &str,
        message: AssistantMessage,
        finish_reason: &str,
        usage: Usage,
    ) -> Self {
        Self {
            id: response_id("chatcmpl"),
            object: "chat.completion",
            created: created_now(),
            model: model.to_string(),
            choices: vec![ChatChoice {
                index: 0,
                message: Some(message),
                delta: None,
                finish_reason: Some(finish_reason.to_string()),
            }],
            usage,
        }
    }
}

/// Message assistant d'une réponse non-streaming.
#[derive(Debug, Serialize)]
pub(super) struct AssistantMessage {
    role: &'static str,
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tool_calls: Vec<ToolCall>,
}

impl AssistantMessage {
    /// Message texte seul.
    pub(super) fn text(content: String) -> Self {
        Self::new(Some(content), None, Vec::new())
    }

    /// Message complet : `content` vaut `null` quand seuls des appels d'outils
    /// ont été produits, comme l'API OpenAI.
    pub(super) fn new(
        content: Option<String>,
        reasoning_content: Option<String>,
        tool_calls: Vec<ToolCall>,
    ) -> Self {
        Self {
            role: "assistant",
            content,
            reasoning_content,
            tool_calls,
        }
    }
}

/// Appel d'outil OpenAI (`tool_calls[]`).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub(super) struct ToolCall {
    /// Position de l'appel, présente dans les deltas streamés seulement.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) index: Option<usize>,
    /// Identifiant de l'appel, repris par le message `tool` de réponse.
    pub(super) id: String,
    /// Toujours `function`.
    #[serde(rename = "type")]
    pub(super) kind: &'static str,
    /// Fonction appelée.
    pub(super) function: ToolCallFunction,
}

/// Fonction d'un appel d'outil OpenAI.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub(super) struct ToolCallFunction {
    /// Nom de la fonction.
    pub(super) name: String,
    /// Arguments sérialisés en JSON (objet).
    pub(super) arguments: String,
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

#[derive(Debug, Default, Serialize)]
struct DeltaMessage {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCall>>,
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
                ..DeltaMessage::default()
            },
            None,
        )
    }

    /// Construit un chunk de contenu.
    pub(super) fn content(model: &str, content: String) -> Self {
        Self::chunk(
            model,
            DeltaMessage {
                content: Some(content),
                ..DeltaMessage::default()
            },
            None,
        )
    }

    /// Construit un chunk de raisonnement.
    pub(super) fn reasoning(model: &str, reasoning: String) -> Self {
        Self::chunk(
            model,
            DeltaMessage {
                reasoning_content: Some(reasoning),
                ..DeltaMessage::default()
            },
            None,
        )
    }

    /// Construit un chunk portant un appel d'outil complet.
    pub(super) fn tool_call(model: &str, call: ToolCall) -> Self {
        Self::chunk(
            model,
            DeltaMessage {
                tool_calls: Some(vec![call]),
                ..DeltaMessage::default()
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
                delta: Some(DeltaMessage::default()),
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
    fn request_accepts_json_lines_response_format() {
        let req: ChatCompletionRequest = serde_json::from_str(
            r#"{"model":"reti-35b","messages":[],"response_format":{"type":"json_lines"}}"#,
        )
        .expect("invariant: JSON valide");

        assert_eq!(
            req.response_format_mode()
                .expect("invariant: json_lines supporté"),
            ResponseFormatMode::JsonLines
        );
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
    fn request_carries_tools_reasoning_and_tool_calls_to_template() {
        let req: ChatCompletionRequest = serde_json::from_str(
            r#"{"model":"flint","tools":[{"type":"function","function":{"name":"heure"}}],
                "messages":[{"role":"assistant","content":null,"reasoning_content":"r",
                "tool_calls":[{"id":"c0","type":"function","function":{"name":"heure","arguments":"{}"}}]},
                {"role":"tool","tool_call_id":"c0","content":"12:00"}]}"#,
        )
        .expect("invariant: JSON valide");

        let messages = req.template_messages();
        assert_eq!(req.tools().len(), 1);
        assert_eq!(messages[0].reasoning_content.as_deref(), Some("r"));
        assert_eq!(messages[0].tool_calls[0]["function"]["name"], "heure");
    }

    #[test]
    fn tool_call_message_serializes_null_content() {
        let call = ToolCall {
            index: None,
            id: "c0".to_string(),
            kind: "function",
            function: ToolCallFunction {
                name: "heure".to_string(),
                arguments: "{}".to_string(),
            },
        };
        let message = AssistantMessage::new(None, Some("r".to_string()), vec![call]);
        let value: Value =
            serde_json::from_slice(&json_bytes(&message).expect("invariant: message sérialisable"))
                .expect("invariant: JSON");

        assert_eq!(value["content"], Value::Null);
        assert_eq!(value["reasoning_content"], "r");
        assert_eq!(value["tool_calls"][0]["type"], "function");
        assert_eq!(value["tool_calls"][0]["function"]["arguments"], "{}");
        assert!(value["tool_calls"][0].get("index").is_none());
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
