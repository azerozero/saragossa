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
use std::time::{Duration, Instant};

use saragossa::whisper::{StreamLimit, MAX_CHUNK_SAMPLES};
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
/// Sessions simultanées ; chaque fenêtre garde au plus 60 s de PCM f32
/// (3,8 Mo) et ses fenêtres closes au plus 30 min de texte.
const MAX_STT_SESSIONS: usize = 8;
/// Une session inactive plus longtemps est fermée, avec son texte retenu.
const STT_SESSION_IDLE_TTL: Duration = Duration::from_secs(300);

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
    /// Dernier accès, pour l'expiration.
    touched: Instant,
}

impl SttSessions {
    fn tick(&mut self) -> u64 {
        self.next_id += 1;
        self.next_id
    }

    fn evict_idle(&mut self, now: Instant) {
        self.open
            .retain(|_, session| now.duration_since(session.touched) < STT_SESSION_IDLE_TTL);
    }

    fn open(&mut self, language: String, now: Instant) -> (String, SttSession) {
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
            touched: now,
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
        write_transcription_sse(stream, |on_delta| {
            model
                .transcribe_with_deltas(samples, language, runtime, on_delta)
                .map(|(text, _)| text)
        })
    })
}

/// Écrit la réponse SSE ; `transcribe` émet les deltas puis rend le texte.
/// Un client parti fait échouer l'écriture d'un delta, ce qui arrête le décodage.
fn write_transcription_sse<S: Write>(
    stream: &mut S,
    transcribe: impl FnOnce(&mut dyn FnMut(&str) -> saragossa::Result<()>) -> saragossa::Result<String>,
) -> ServeResult<()> {
    write_headers(
        stream,
        200,
        "OK",
        "text/event-stream; charset=utf-8",
        None,
        Vec::new(),
    )?;
    let result = (|| -> ServeResult<()> {
        let text = transcribe(&mut |delta| write_frame(stream, &TranscriptEvent::Delta { delta }))?;
        write_frame(stream, &TranscriptEvent::Done { text: &text })?;
        Ok(())
    })();
    if let Err(error) = result {
        eprintln!("saragossa serve STT stream interrompu: {error}");
        let frame = sse_event(&json!({"type": "error", "error": {"message": error.to_string()}}))?;
        let _ = stream.write_all(&frame).and_then(|()| stream.flush());
    }
    Ok(())
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
    // Refus avant toute copie : un corps maximal (256 Mio) ferait 524 s d'audio.
    if body.len() > MAX_CHUNK_SAMPLES * 2 {
        return Err(limit_error(StreamLimit::ChunkTooLarge));
    }
    let chunk = pcm_s16le_to_f32(body)?;
    audio.with_stt_sessions(|model, runtime, sessions| {
        serve_stream_chunk(
            stream,
            sessions,
            headers,
            &chunk,
            Instant::now(),
            &mut |samples, language| {
                model
                    .transcribe(samples, language, runtime)
                    .map(|(text, _)| text)
            },
        )
    })
}

fn limit_error(limit: StreamLimit) -> ServeError {
    ServeError::InvalidParameter {
        param: "body",
        message: limit.message(),
    }
}

/// Avance une session et écrit sa réponse, sans dépendre du modèle.
///
/// L'admission précède toute modification : un refus laisse la session
/// intacte. Une erreur d'inférence abandonne la session, dont l'état peut
/// être partiel.
fn serve_stream_chunk<S: Write>(
    stream: &mut S,
    sessions: &mut SttSessions,
    headers: &StreamChunkHeaders<'_>,
    chunk: &[f32],
    now: Instant,
    transcribe: &mut dyn FnMut(&[f32], &str) -> saragossa::Result<String>,
) -> ServeResult<()> {
    sessions.evict_idle(now);
    let (id, mut session) = match headers.session {
        Some(id) => {
            let existing = sessions.open.get(id).ok_or(ServeError::InvalidParameter {
                param: "X-Saragossa-STT-Session",
                message: "session STT inconnue, close, expirée ou évincée",
            })?;
            existing.stream.admit(chunk.len()).map_err(limit_error)?;
            let session = sessions
                .open
                .remove(id)
                .ok_or(ServeError::InvalidParameter {
                    param: "X-Saragossa-STT-Session",
                    message: "session STT inconnue, close, expirée ou évincée",
                })?;
            (id.to_string(), session)
        }
        None => {
            WhisperStream::new()
                .admit(chunk.len())
                .map_err(limit_error)?;
            sessions.open(headers.language.unwrap_or("auto").to_string(), now)
        }
    };
    session.last_used = sessions.tick();
    session.touched = now;
    let language = session.language.clone();
    let mut transcribe = |samples: &[f32]| transcribe(samples, &language);
    let audio_seconds = (session.stream.received_samples() + chunk.len()) as f32 / 16_000.0;
    let response = if headers.is_final {
        let text = session.stream.finish(chunk, &mut transcribe)?;
        StreamChunkResponse {
            session: id.clone(),
            is_final: true,
            committed: text.clone(),
            partial: String::new(),
            text: Some(text),
            audio_seconds,
        }
    } else {
        let update = session.stream.push(chunk, &mut transcribe)?;
        sessions.open.insert(id.clone(), session);
        StreamChunkResponse {
            session: id.clone(),
            is_final: false,
            committed: update.committed,
            partial: update.partial,
            text: None,
            audio_seconds,
        }
    };
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

    // Le déchargement d'inactivité libère les modèles mais garde les sessions :
    // une pause plus longue que le délai ne doit pas rendre « session inconnue ».
    #[test]
    fn idle_unload_keeps_stt_sessions() {
        let args = ServeArgs::parse(["--model", "a=/x/a", "--backend", "cpu"].map(String::from))
            .expect("args");
        let mut audio = super::super::audio::AudioState::new(&args);
        let (id, session) = audio
            .stt_sessions
            .lock()
            .expect("invariant: sessions saines")
            .open("fr".into(), Instant::now());
        audio
            .stt_sessions
            .lock()
            .expect("invariant: sessions saines")
            .open
            .insert(id.clone(), session);
        assert!(!audio.unload_models(), "rien de chargé");
        assert!(audio
            .stt_sessions
            .lock()
            .expect("invariant: sessions saines")
            .open
            .contains_key(&id));
        audio.invalidate();
        assert!(
            audio
                .stt_sessions
                .lock()
                .expect("invariant: sessions saines")
                .open
                .is_empty(),
            "panique : tout part"
        );
    }

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
        assert!(audio
            .stt_sessions
            .lock()
            .expect("invariant: sessions saines")
            .open
            .is_empty());
        // Un corps hors limite est refusé avant toute copie ou chargement.
        let huge = vec![0; MAX_CHUNK_SAMPLES * 2 + 2];
        assert!(matches!(
            handle_transcription_stream(&mut response, &mut audio, &headers, &huge),
            Err(ServeError::InvalidParameter { .. })
        ));
        assert!(response.is_empty());
    }

    fn headers(session: Option<&str>, is_final: bool) -> StreamChunkHeaders<'_> {
        StreamChunkHeaders {
            session,
            is_final,
            language: Some("fr"),
        }
    }

    /// Transcription factice : un mot par tranche de 250 ms, compte les appels.
    fn fake(
        calls: &std::cell::Cell<usize>,
    ) -> impl FnMut(&[f32], &str) -> saragossa::Result<String> + '_ {
        move |samples, language| {
            calls.set(calls.get() + 1);
            assert_eq!(language, "fr");
            Ok(vec!["mot"; samples.len() / 4_000].join(" "))
        }
    }

    fn json_body(response: &[u8]) -> serde_json::Value {
        let text = String::from_utf8_lossy(response);
        let (head, body) = text
            .split_once("\r\n\r\n")
            .expect("invariant: en-têtes HTTP");
        assert!(head.starts_with("HTTP/1.1 200 OK"), "{head}");
        serde_json::from_str(body).expect("invariant: corps JSON")
    }

    #[test]
    fn stream_session_http_round_trip_reaches_final_text() -> ServeResult<()> {
        let calls = std::cell::Cell::new(0);
        let mut transcribe = fake(&calls);
        let mut sessions = SttSessions::default();
        let now = Instant::now();
        let quarter = vec![0.0; 4_000];
        let mut response = Vec::new();
        serve_stream_chunk(
            &mut response,
            &mut sessions,
            &headers(None, false),
            &quarter,
            now,
            &mut transcribe,
        )?;
        let first = json_body(&response);
        assert_eq!(first["session"], "stt-1");
        assert_eq!(first["final"], false);
        assert_eq!(first["partial"], "mot");
        assert!(String::from_utf8_lossy(&response).contains("X-Saragossa-STT-Session: stt-1"));
        let mut response = Vec::new();
        serve_stream_chunk(
            &mut response,
            &mut sessions,
            &headers(Some("stt-1"), false),
            &quarter,
            now,
            &mut transcribe,
        )?;
        assert_eq!(json_body(&response)["committed"], "mot");
        // Morceau vide : réponse identique, sans inférence.
        let before = calls.get();
        let mut response = Vec::new();
        serve_stream_chunk(
            &mut response,
            &mut sessions,
            &headers(Some("stt-1"), false),
            &[],
            now,
            &mut transcribe,
        )?;
        assert_eq!(calls.get(), before);
        let unchanged = json_body(&response);
        assert_eq!(
            (unchanged["committed"].clone(), unchanged["partial"].clone()),
            ("mot".into(), "mot".into())
        );
        let mut response = Vec::new();
        serve_stream_chunk(
            &mut response,
            &mut sessions,
            &headers(Some("stt-1"), true),
            &quarter,
            now,
            &mut transcribe,
        )?;
        let done = json_body(&response);
        assert_eq!(
            (done["final"].clone(), done["text"].clone()),
            (true.into(), "mot mot mot".into())
        );
        assert!(sessions.open.is_empty(), "session close après le final");
        let error = serve_stream_chunk(
            &mut Vec::new(),
            &mut sessions,
            &headers(Some("stt-1"), false),
            &[],
            now,
            &mut transcribe,
        );
        assert!(matches!(error, Err(ServeError::InvalidParameter { .. })));
        Ok(())
    }

    #[test]
    fn rejected_chunk_keeps_session_and_skips_inference() -> ServeResult<()> {
        let calls = std::cell::Cell::new(0);
        let mut transcribe = fake(&calls);
        let mut sessions = SttSessions::default();
        let now = Instant::now();
        serve_stream_chunk(
            &mut Vec::new(),
            &mut sessions,
            &headers(None, false),
            &[0.0; 4_000],
            now,
            &mut transcribe,
        )?;
        let too_big = vec![0.0; MAX_CHUNK_SAMPLES + 1];
        for session in [None, Some("stt-1")] {
            let error = serve_stream_chunk(
                &mut Vec::new(),
                &mut sessions,
                &headers(session, false),
                &too_big,
                now,
                &mut transcribe,
            );
            assert!(matches!(
                error,
                Err(ServeError::InvalidParameter { param: "body", .. })
            ));
        }
        assert_eq!(calls.get(), 1);
        assert_eq!(sessions.open.len(), 1);
        assert_eq!(sessions.open["stt-1"].stream.received_samples(), 4_000);
        Ok(())
    }

    /// Écrivain qui échoue après `budget` octets, comme un client parti.
    struct Disconnecting {
        written: Vec<u8>,
        budget: usize,
    }

    impl Write for Disconnecting {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            if self.written.len() + buf.len() > self.budget {
                return Err(std::io::ErrorKind::BrokenPipe.into());
            }
            self.written.extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn sse_round_trip_and_client_disconnect_stops_decoding() -> ServeResult<()> {
        let mut response = Vec::new();
        write_transcription_sse(&mut response, |on_delta| {
            on_delta("Bon")?;
            on_delta("jour.")?;
            Ok("Bonjour.".to_string())
        })?;
        let text = String::from_utf8_lossy(&response);
        assert!(text.starts_with("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream"));
        assert!(text.ends_with(
            "data: {\"type\":\"transcript.text.delta\",\"delta\":\"jour.\"}\n\n\
             data: {\"type\":\"transcript.text.done\",\"text\":\"Bonjour.\"}\n\n"
        ));
        let mut gone = Disconnecting {
            written: Vec::new(),
            budget: 160,
        };
        let mut deltas = 0;
        write_transcription_sse(&mut gone, |on_delta| {
            for _ in 0..100 {
                deltas += 1;
                on_delta("mot ")?;
            }
            Ok(String::new())
        })?;
        assert!(
            deltas < 100,
            "le décodage s'arrête à la première écriture refusée"
        );
        Ok(())
    }

    #[test]
    fn session_survives_response_write_failure() -> ServeResult<()> {
        let calls = std::cell::Cell::new(0);
        let mut transcribe = fake(&calls);
        let mut sessions = SttSessions::default();
        let mut gone = Disconnecting {
            written: Vec::new(),
            budget: 0,
        };
        let error = serve_stream_chunk(
            &mut gone,
            &mut sessions,
            &headers(None, false),
            &[0.0; 4_000],
            Instant::now(),
            &mut transcribe,
        );
        assert!(matches!(error, Err(ServeError::Io { .. })));
        // L'état est avancé : le client peut reprendre avec l'identifiant suivant.
        assert_eq!(sessions.open["stt-1"].stream.received_samples(), 4_000);
        Ok(())
    }

    #[test]
    fn access_refreshes_lru_and_idle_sessions_expire() -> ServeResult<()> {
        let calls = std::cell::Cell::new(0);
        let mut transcribe = fake(&calls);
        let mut sessions = SttSessions::default();
        let now = Instant::now();
        for _ in 0..MAX_STT_SESSIONS {
            serve_stream_chunk(
                &mut Vec::new(),
                &mut sessions,
                &headers(None, false),
                &[],
                now,
                &mut transcribe,
            )?;
        }
        // stt-1 est le plus ancien ; un accès le rend le plus récent.
        serve_stream_chunk(
            &mut Vec::new(),
            &mut sessions,
            &headers(Some("stt-1"), false),
            &[],
            now,
            &mut transcribe,
        )?;
        serve_stream_chunk(
            &mut Vec::new(),
            &mut sessions,
            &headers(None, false),
            &[],
            now,
            &mut transcribe,
        )?;
        assert!(sessions.open.contains_key("stt-1"));
        assert!(
            !sessions.open.contains_key("stt-2"),
            "le moins récent est évincé"
        );
        assert_eq!(sessions.open.len(), MAX_STT_SESSIONS);
        let later = now + STT_SESSION_IDLE_TTL;
        let error = serve_stream_chunk(
            &mut Vec::new(),
            &mut sessions,
            &headers(Some("stt-1"), false),
            &[],
            later,
            &mut transcribe,
        );
        assert!(matches!(error, Err(ServeError::InvalidParameter { .. })));
        assert!(sessions.open.is_empty(), "sessions inactives expirées");
        Ok(())
    }

    /// Requête complète sur le thread moteur, sans modèle : seules les
    /// validations faites avant le chargement peuvent répondre.
    fn route(head: &str, body: &[u8]) -> (ServeResult<()>, String) {
        let mut raw = format!("{head}Content-Length: {}\r\n\r\n", body.len()).into_bytes();
        raw.extend_from_slice(body);
        let request = super::super::http::read_request(
            &mut std::io::Cursor::new(raw),
            Instant::now() + Duration::from_secs(30),
        )
        .expect("invariant: requête HTTP valide")
        .expect("invariant: requête présente");
        let args = ServeArgs::parse(Vec::<String>::new()).expect("invariant: args valides");
        let mut state = super::super::state::ServeState::new(&args);
        let mut response = Vec::new();
        let result = super::super::http::handle_job(&mut response, &mut state, &request);
        (result, String::from_utf8_lossy(&response).into_owned())
    }

    fn form(fields: &[(&str, &str)]) -> Vec<u8> {
        let mut body = Vec::new();
        for (name, value) in fields {
            body.extend_from_slice(
                format!(
                    "--B\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
                )
                .as_bytes(),
            );
        }
        body.extend_from_slice(b"--B--\r\n");
        body
    }

    #[test]
    fn routes_validate_stt_parameters_before_loading_the_model() {
        let multipart = "POST /v1/audio/transcriptions HTTP/1.1\r\nContent-Type: multipart/form-data; boundary=B\r\n";
        for (fields, status) in [
            (
                vec![
                    ("file", "x"),
                    ("response_format", "verbose_json"),
                    ("timestamp_granularities[]", "word"),
                ],
                "501",
            ),
            (
                vec![("file", "x"), ("timestamp_granularities[]", "segment")],
                "400",
            ),
            (
                vec![
                    ("file", "x"),
                    ("response_format", "verbose_json"),
                    ("stream", "true"),
                ],
                "400",
            ),
        ] {
            let (result, response) = route(multipart, &form(&fields));
            assert!(result.is_err());
            assert!(
                response.starts_with(&format!("HTTP/1.1 {status} ")),
                "{fields:?}: {response}"
            );
        }
        let stream =
            "POST /v1/audio/transcriptions/stream HTTP/1.1\r\nX-Saragossa-STT-Final: true\r\n";
        let (_, response) = route(stream, &[0; 3]);
        assert!(
            response.starts_with("HTTP/1.1 400 ") && response.contains("impair"),
            "{response}"
        );
        let (_, response) = route(stream, &vec![0; MAX_CHUNK_SAMPLES * 2 + 2]);
        assert!(
            response.starts_with("HTTP/1.1 400 ") && response.contains("30 s"),
            "{response}"
        );
    }

    #[test]
    fn sessions_evict_the_least_recently_used() {
        let mut sessions = SttSessions::default();
        for _ in 0..MAX_STT_SESSIONS + 2 {
            let (id, session) = sessions.open("fr".to_string(), Instant::now());
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
