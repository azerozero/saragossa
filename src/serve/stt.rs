//! Routes STT de `saragossa serve`.
//!
//! - `POST /v1/audio/transcriptions` : `multipart/form-data` compatible OpenAI
//!   (`file` WAV, `language`, `response_format`, `stream`,
//!   `timestamp_granularities[]`). `json` et `text` décodent sans timestamps ;
//!   `verbose_json` ajoute `language`, `duration` et `segments[]` issus des
//!   tokens `<|t.tt|>`. `stream=true` répond en SSE : un
//!   `transcript.text.delta` par morceau de texte décodé, puis
//!   `transcript.text.done` avec le texte complet, identique à `json`.
//! - `POST /v1/audio/transcriptions/stream` : session à entrée incrémentale.
//!   Le corps est du PCM s16le mono 16 kHz brut ; `X-Saragossa-STT-Session`
//!   reprend une session (absent : nouvelle session), `X-Saragossa-STT-Language`
//!   la configure à l'ouverture et `X-Saragossa-STT-Final: true` la termine. Chaque
//!   réponse JSON porte `committed` (stabilisé par LocalAgreement-2) et
//!   `partial` ; la réponse finale porte `text`, égal au batch du même PCM. Une
//!   requête par morceau convient au serveur HTTP séquentiel (pas de corps
//!   chunked) et à un client bloquant ; voir `docs/stt-streaming.md`.

use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;

use saragossa::wire_audio::{extract_boundary, text_field};
use saragossa::WhisperStream;
use serde::Serialize;
use serde_json::json;

use super::audio::{parse_multipart, send_text, wav_to_mono_16k, AudioState};
use super::error::{ServeError, ServeResult};
use super::http::{send_json, write_headers};
use super::protocol::sse_event;

pub(super) const STT_SESSION_HEADER: &str = "x-saragossa-stt-session";
pub(super) const STT_FINAL_HEADER: &str = "x-saragossa-stt-final";
pub(super) const STT_LANGUAGE_HEADER: &str = "x-saragossa-stt-language";
/// Au plus 31 s de PCM par session : la borne tient la mémoire sous ~16 Mo.
const MAX_STT_SESSIONS: usize = 8;

/// Sessions ouvertes ; la plus ancienne est évincée au-delà de la borne.
// ponytail: identifiants séquentiels, protégés seulement par le bearer du
// serveur ; passer à des identifiants aléatoires si le serveur sort du local.
#[derive(Default)]
pub(super) struct SttSessions {
    next_id: u64,
    open: HashMap<String, SttSession>,
}

struct SttSession {
    stream: WhisperStream,
    language: String,
    /// Horloge logique : `Instant` peut égaliser deux accès rapprochés.
    last_used: u64,
}

impl SttSessions {
    fn tick(&mut self) -> u64 {
        self.next_id += 1;
        self.next_id
    }

    fn open(&mut self, language: String) -> (String, SttSession) {
        if self.open.len() >= MAX_STT_SESSIONS {
            let oldest = self
                .open
                .iter()
                .min_by_key(|(_, session)| session.last_used)
                .map(|(id, _)| id.clone());
            if let Some(id) = oldest {
                self.open.remove(&id);
            }
        }
        let id = self.tick();
        let session = SttSession {
            stream: WhisperStream::new(),
            language,
            last_used: id,
        };
        (format!("stt-{id}"), session)
    }
}

/// Événements SSE de transcription, au format OpenAI récent.
#[derive(Serialize)]
#[serde(tag = "type")]
enum TranscriptEvent<'a> {
    #[serde(rename = "transcript.text.delta")]
    Delta { delta: &'a str },
    #[serde(rename = "transcript.text.done")]
    Done { text: &'a str },
}

/// Réponse d'un morceau de session incrémentale.
#[derive(Debug, Serialize)]
struct StreamChunkResponse {
    session: String,
    #[serde(rename = "final")]
    is_final: bool,
    committed: String,
    partial: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    audio_seconds: f32,
}

/// En-têtes de la session incrémentale.
pub(super) struct StreamChunkHeaders<'a> {
    pub(super) session: Option<&'a str>,
    pub(super) is_final: bool,
    pub(super) language: Option<&'a str>,
}

/// Sert `POST /v1/audio/transcriptions`.
pub(super) fn handle_transcription<S: Write>(
    stream: &mut S,
    audio: &mut AudioState,
    content_type: Option<&str>,
    body: &[u8],
) -> ServeResult<()> {
    let content_type = content_type
        .ok_or_else(|| ServeError::Http("Content-Type multipart/form-data requis".to_string()))?;
    let boundary = extract_boundary(content_type).ok_or_else(|| {
        ServeError::Http("boundary multipart absente du Content-Type".to_string())
    })?;
    let fields = parse_multipart(body, &boundary)?;
    let file = fields
        .iter()
        .find(|field| field.name == "file")
        .ok_or_else(|| ServeError::Http("champ multipart 'file' requis".to_string()))?;
    let language = text_field(&fields, "language").unwrap_or_else(|| "auto".to_string());
    let format = text_field(&fields, "response_format")
        .unwrap_or_default()
        .to_ascii_lowercase();
    let verbose = format == "verbose_json";
    let streaming = text_field(&fields, "stream").is_some_and(|v| v.eq_ignore_ascii_case("true"));
    check_granularities(
        fields
            .iter()
            .filter(|field| field.name.starts_with("timestamp_granularities"))
            .map(|field| {
                String::from_utf8_lossy(&field.data)
                    .trim()
                    .to_ascii_lowercase()
            }),
        verbose,
    )?;
    if streaming && verbose {
        return Err(ServeError::InvalidParameter {
            param: "stream",
            message: "incompatible avec response_format=verbose_json",
        });
    }
    let samples = wav_to_mono_16k(&file.data)?;
    if streaming {
        return send_transcription_sse(stream, audio, &samples, &language);
    }
    if verbose {
        let transcript = audio.with_stt(|model, runtime| {
            Ok(model.transcribe_verbose(&samples, &language, runtime)?)
        })?;
        return send_json(stream, 200, &transcript, Vec::new());
    }
    let text = audio.transcribe(&samples, &language)?;
    if format == "text" {
        return send_text(stream, &text);
    }
    send_json(stream, 200, &json!({ "text": text }), Vec::new())
}

fn check_granularities(values: impl Iterator<Item = String>, verbose: bool) -> ServeResult<()> {
    const PARAM: &str = "timestamp_granularities[]";
    for value in values {
        match value.as_str() {
            "segment" if verbose => {}
            "segment" => {
                return Err(ServeError::InvalidParameter {
                    param: PARAM,
                    message: "exige response_format=verbose_json",
                })
            }
            // Les mots exigeraient l'alignement DTW des têtes cross-attention.
            "word" => {
                return Err(ServeError::NotImplementedParameter {
                    param: PARAM.to_string(),
                    message: "word non disponible, utiliser segment",
                })
            }
            _ => {
                return Err(ServeError::InvalidParameter {
                    param: PARAM,
                    message: "valeurs acceptées: segment",
                })
            }
        }
    }
    Ok(())
}

fn sse_frame(event: &TranscriptEvent<'_>) -> ServeResult<Vec<u8>> {
    sse_event(event)
}

// Le modèle est chargé avant les en-têtes : une erreur de configuration reste
// une réponse HTTP. Après les en-têtes, l'échec devient un événement `error`.
fn send_transcription_sse<S: Write>(
    stream: &mut S,
    audio: &mut AudioState,
    samples: &[f32],
    language: &str,
) -> ServeResult<()> {
    audio.with_stt(|model, runtime| {
        write_headers(
            stream,
            200,
            "OK",
            "text/event-stream; charset=utf-8",
            None,
            Vec::new(),
        )?;
        let result = (|| -> ServeResult<()> {
            let (text, _) =
                model.transcribe_with_deltas(samples, language, runtime, &mut |delta| {
                    write_frame(stream, &TranscriptEvent::Delta { delta })
                })?;
            write_frame(stream, &TranscriptEvent::Done { text: &text })?;
            Ok(())
        })();
        if let Err(error) = result {
            eprintln!("saragossa serve STT stream interrompu: {error}");
            let frame =
                sse_event(&json!({"type": "error", "error": {"message": error.to_string()}}))?;
            let _ = stream.write_all(&frame).and_then(|()| stream.flush());
        }
        Ok(())
    })
}

fn write_frame<S: Write>(stream: &mut S, event: &TranscriptEvent<'_>) -> saragossa::Result<()> {
    let io = |source| saragossa::InferError::Io {
        path: PathBuf::from("/v1/audio/transcriptions"),
        source,
    };
    let frame = sse_frame(event).map_err(|e| io(std::io::Error::other(e.to_string())))?;
    stream.write_all(&frame).map_err(io)?;
    stream.flush().map_err(io)
}

/// Sert `POST /v1/audio/transcriptions/stream`.
pub(super) fn handle_transcription_stream<S: Write>(
    stream: &mut S,
    audio: &mut AudioState,
    headers: &StreamChunkHeaders<'_>,
    body: &[u8],
) -> ServeResult<()> {
    let chunk = pcm_s16le_to_f32(body)?;
    let (id, mut session) = match headers.session {
        Some(id) => {
            let session =
                audio
                    .stt_sessions
                    .open
                    .remove(id)
                    .ok_or(ServeError::InvalidParameter {
                        param: "X-Saragossa-STT-Session",
                        message: "session STT inconnue, close ou évincée",
                    })?;
            (id.to_string(), session)
        }
        None => audio
            .stt_sessions
            .open(headers.language.unwrap_or("auto").to_string()),
    };
    session.last_used = audio.stt_sessions.tick();
    let language = session.language.clone();
    // En cas d'erreur la session est abandonnée : son état peut être partiel.
    let (response, session) = audio.with_stt(|model, runtime| {
        let mut transcribe = |samples: &[f32]| {
            model
                .transcribe(samples, &language, runtime)
                .map(|(text, _)| text)
        };
        let received = session.stream.received_samples() + chunk.len();
        let audio_seconds = received as f32 / 16_000.0;
        if headers.is_final {
            let text = session.stream.finish(&chunk, &mut transcribe)?;
            let response = StreamChunkResponse {
                session: id.clone(),
                is_final: true,
                committed: text.clone(),
                partial: String::new(),
                text: Some(text),
                audio_seconds,
            };
            return Ok((response, None));
        }
        let update = session.stream.push(&chunk, &mut transcribe)?;
        let response = StreamChunkResponse {
            session: id.clone(),
            is_final: false,
            committed: update.committed,
            partial: update.partial,
            text: None,
            audio_seconds,
        };
        Ok((response, Some(session)))
    })?;
    if let Some(session) = session {
        audio.stt_sessions.open.insert(id.clone(), session);
    }
    send_json(
        stream,
        200,
        &response,
        vec![("X-Saragossa-STT-Session", id)],
    )
}

fn pcm_s16le_to_f32(body: &[u8]) -> ServeResult<Vec<f32>> {
    if body.len() % 2 != 0 {
        return Err(ServeError::Http(
            "PCM s16le attendu: nombre d'octets impair".to_string(),
        ));
    }
    Ok(body
        .chunks_exact(2)
        .map(|pair| f32::from(i16::from_le_bytes([pair[0], pair[1]])) / 32_768.0)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::super::args::ServeArgs;
    use super::*;

    #[test]
    fn sse_events_follow_openai_transcript_format() -> ServeResult<()> {
        assert_eq!(
            sse_frame(&TranscriptEvent::Delta {
                delta: " Bon\"jour"
            })?,
            b"data: {\"type\":\"transcript.text.delta\",\"delta\":\" Bon\\\"jour\"}\n\n"
        );
        assert_eq!(
            sse_frame(&TranscriptEvent::Done { text: "Bonjour." })?,
            b"data: {\"type\":\"transcript.text.done\",\"text\":\"Bonjour.\"}\n\n"
        );
        Ok(())
    }

    #[test]
    fn granularities_accept_segment_only_with_verbose_json() {
        let values = |list: &[&str]| list.iter().map(|v| v.to_string()).collect::<Vec<_>>();
        assert!(check_granularities(values(&[]).into_iter(), false).is_ok());
        assert!(check_granularities(values(&["segment"]).into_iter(), true).is_ok());
        assert!(matches!(
            check_granularities(values(&["segment"]).into_iter(), false),
            Err(ServeError::InvalidParameter { .. })
        ));
        assert!(matches!(
            check_granularities(values(&["segment", "word"]).into_iter(), true),
            Err(ServeError::NotImplementedParameter { .. })
        ));
        assert!(matches!(
            check_granularities(values(&["phrase"]).into_iter(), true),
            Err(ServeError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn pcm_chunks_decode_s16le_and_reject_odd_length() -> ServeResult<()> {
        let body = [0x00, 0x80, 0xff, 0x7f, 0x00, 0x00];
        assert_eq!(pcm_s16le_to_f32(&body)?, [-1.0, 32_767.0 / 32_768.0, 0.0]);
        assert!(pcm_s16le_to_f32(&body[..3]).is_err());
        Ok(())
    }

    #[test]
    fn stream_session_without_model_fails_before_creating_state() {
        let args = ServeArgs::parse(Vec::<String>::new()).expect("invariant: args valides");
        let mut audio = AudioState::new(&args);
        let mut response = Vec::new();
        let headers = StreamChunkHeaders {
            session: None,
            is_final: false,
            language: Some("fr"),
        };
        let error = handle_transcription_stream(&mut response, &mut audio, &headers, &[0; 320])
            .expect_err("invariant: STT non configuré refusé");
        assert!(error.to_string().contains("STT non configuré"));
        assert!(audio.stt_sessions.open.is_empty());
        let unknown = StreamChunkHeaders {
            session: Some("stt-404"),
            ..headers
        };
        assert!(matches!(
            handle_transcription_stream(&mut response, &mut audio, &unknown, &[]),
            Err(ServeError::InvalidParameter { .. })
        ));
        assert!(response.is_empty());
    }

    #[test]
    fn sessions_evict_the_least_recently_used() {
        let mut sessions = SttSessions::default();
        for _ in 0..MAX_STT_SESSIONS + 2 {
            let (id, session) = sessions.open("fr".to_string());
            sessions.open.insert(id, session);
        }
        assert_eq!(sessions.open.len(), MAX_STT_SESSIONS);
        assert!(!sessions.open.contains_key("stt-1"));
        assert!(!sessions.open.contains_key("stt-2"));
        assert!(sessions
            .open
            .contains_key(&format!("stt-{}", MAX_STT_SESSIONS + 2)));
    }
}
