//! Endpoints audio OpenAI-compatibles de `saragossa serve`.
//!
//! Deux routes purement additives :
//! - `POST /v1/audio/transcriptions` : STT Whisper, requête `multipart/form-data`
//!   (`file` = WAV, plus `model`/`language`/`response_format` optionnels), réponse
//!   JSON `{"text": ...}` (ou texte brut si `response_format=text`).
//! - `POST /v1/audio/speech` : TTS Qwen3, requête JSON `{"model", "input", "voice"?}`,
//!   réponse bytes WAV (`Content-Type: audio/wav`).
//!
//! Les modèles sont opt-in (`--stt-model`, `--tts-model`) et chargés paresseusement
//! au premier appel : sans configuration, la route renvoie une erreur 400 claire.

use std::io::{Cursor, Write};
use std::path::PathBuf;

use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use serde::Deserialize;
use serde_json::json;

#[cfg(all(target_os = "macos", feature = "metal"))]
use saragossa::MetalExecutor;
use saragossa::{ForwardRuntime, TtsModel, TtsSynthesisOutput, WhisperModel};

use super::args::ServeArgs;
use super::error::{ServeError, ServeResult};
use super::http::{send_json, write_headers};
use crate::RuntimeKind;

/// Fréquence d'échantillonnage attendue par l'encodeur Whisper.
const WHISPER_SAMPLE_RATE: u32 = 16_000;
/// Cap par défaut du nombre de frames codec générées par la TTS.
const DEFAULT_TTS_MAX_FRAMES: usize = 160;
/// Surcharge du cap de frames TTS pour les entrées longues.
const TTS_MAX_FRAMES_ENV: &str = "SARAGOSSA_TTS_MAX_FRAMES";
const STT_NOT_CONFIGURED: &str =
    "endpoint STT non configuré: relancez saragossa serve avec --stt-model <dir>";
const TTS_NOT_CONFIGURED: &str =
    "endpoint TTS non configuré: relancez saragossa serve avec --tts-model <dir>";

/// Etat audio du serveur : slots STT/TTS chargés à la demande.
pub(super) struct AudioState {
    backend: RuntimeKind,
    stt: SttSlot,
    tts: TtsSlot,
}

struct SttSlot {
    path: Option<PathBuf>,
    model: Option<WhisperModel>,
    #[cfg(all(target_os = "macos", feature = "metal"))]
    metal: Option<MetalExecutor>,
}

struct TtsSlot {
    path: Option<PathBuf>,
    model: Option<TtsModel>,
}

impl AudioState {
    /// Construit l'état audio depuis les chemins CLI, sans charger les poids.
    pub(super) fn new(args: &ServeArgs) -> Self {
        Self {
            backend: args.backend,
            stt: SttSlot {
                path: args.stt_model.clone(),
                model: None,
                #[cfg(all(target_os = "macos", feature = "metal"))]
                metal: None,
            },
            tts: TtsSlot {
                path: args.tts_model.clone(),
                model: None,
            },
        }
    }

    /// Transcrit des échantillons mono 16 kHz avec le backend Whisper.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le modèle STT n'est pas configuré ou si l'inférence échoue.
    pub(super) fn transcribe(&mut self, samples: &[f32], lang: &str) -> ServeResult<String> {
        self.ensure_stt_loaded()?;
        let runtime = self.stt.forward_runtime(self.backend);
        let model = self
            .stt
            .model
            .as_ref()
            .ok_or_else(|| ServeError::args("modèle STT absent après chargement"))?;
        let (text, _lang) = model.transcribe(samples, lang, runtime)?;
        Ok(text)
    }

    /// Synthétise du texte en audio avec le backend Qwen3-TTS.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le modèle TTS n'est pas configuré ou si la synthèse échoue.
    pub(super) fn synthesize(&mut self, text: &str) -> ServeResult<TtsSynthesisOutput> {
        self.ensure_tts_loaded()?;
        let model = self
            .tts
            .model
            .as_ref()
            .ok_or_else(|| ServeError::args("modèle TTS absent après chargement"))?;
        let output = model.synthesize_default(text, tts_max_frames())?;
        Ok(output)
    }

    fn ensure_stt_loaded(&mut self) -> ServeResult<()> {
        if self.stt.model.is_some() {
            return Ok(());
        }
        let path = self
            .stt
            .path
            .clone()
            .ok_or_else(|| ServeError::args(STT_NOT_CONFIGURED))?;
        if !path.is_dir() {
            return Err(ServeError::args(format!(
                "dossier modèle STT introuvable: {}",
                path.display()
            )));
        }
        eprintln!("saragossa serve loading STT model path={}", path.display());
        let model = WhisperModel::from_model_dir(&path)?;
        // L'exécuteur Metal reste vivant dans le slot : `ForwardRuntime::metal`
        // emprunte cette référence à chaque transcription.
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if self.backend == RuntimeKind::Metal && self.stt.metal.is_none() {
            self.stt.metal = Some(MetalExecutor::new()?);
        }
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        if self.backend == RuntimeKind::Metal {
            return Err(ServeError::args(
                "backend metal STT indisponible dans ce build — recompile avec --features metal",
            ));
        }
        self.stt.model = Some(model);
        Ok(())
    }

    fn ensure_tts_loaded(&mut self) -> ServeResult<()> {
        if self.tts.model.is_some() {
            return Ok(());
        }
        let path = self
            .tts
            .path
            .clone()
            .ok_or_else(|| ServeError::args(TTS_NOT_CONFIGURED))?;
        if !path.is_dir() {
            return Err(ServeError::args(format!(
                "dossier modèle TTS introuvable: {}",
                path.display()
            )));
        }
        eprintln!("saragossa serve loading TTS model path={}", path.display());
        let model = TtsModel::load_local(&path)?;
        self.tts.model = Some(model);
        Ok(())
    }
}

impl SttSlot {
    fn forward_runtime(&self, backend: RuntimeKind) -> ForwardRuntime<'_> {
        match backend {
            RuntimeKind::Cpu => ForwardRuntime::cpu(),
            RuntimeKind::Metal => {
                #[cfg(all(target_os = "macos", feature = "metal"))]
                {
                    if let Some(metal) = self.metal.as_ref() {
                        return ForwardRuntime::metal(metal);
                    }
                }
                ForwardRuntime::cpu()
            }
        }
    }
}

fn tts_max_frames() -> usize {
    std::env::var(TTS_MAX_FRAMES_ENV)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_TTS_MAX_FRAMES)
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
    let response_format = text_field(&fields, "response_format").unwrap_or_default();
    let samples = wav_to_mono_16k(&file.data)?;
    let text = audio.transcribe(&samples, &language)?;
    if response_format.eq_ignore_ascii_case("text") {
        return send_text(stream, &text);
    }
    send_json(stream, 200, &json!({ "text": text }), Vec::new())
}

/// Sert `POST /v1/audio/speech`.
pub(super) fn handle_speech<S: Write>(
    stream: &mut S,
    audio: &mut AudioState,
    body: &[u8],
) -> ServeResult<()> {
    let request: SpeechRequest = serde_json::from_slice(body)
        .map_err(|e| ServeError::json("désérialisation audio/speech", e))?;
    let input = request.input.trim();
    if input.is_empty() {
        return Err(ServeError::Http(
            "champ 'input' requis pour audio/speech".to_string(),
        ));
    }
    let output = audio.synthesize(input)?;
    let wav = mono_f32_to_wav(&output.samples, output.sample_rate)?;
    write_headers(stream, 200, "OK", "audio/wav", Some(wav.len()), Vec::new())?;
    stream
        .write_all(&wav)
        .map_err(|e| ServeError::io("écriture réponse WAV", e))?;
    stream
        .flush()
        .map_err(|e| ServeError::io("flush réponse WAV", e))
}

/// Requête `/v1/audio/speech`. Seul `input` est consommé : le backend expose une
/// voix VoiceDesign unique, donc `model`/`voice` (ignorés) sont acceptés sans effet.
#[derive(Debug, Deserialize)]
struct SpeechRequest {
    /// Texte à synthétiser.
    input: String,
}

fn send_text<S: Write>(stream: &mut S, text: &str) -> ServeResult<()> {
    let body = text.as_bytes();
    write_headers(
        stream,
        200,
        "OK",
        "text/plain; charset=utf-8",
        Some(body.len()),
        Vec::new(),
    )?;
    stream
        .write_all(body)
        .map_err(|e| ServeError::io("écriture réponse texte", e))?;
    stream
        .flush()
        .map_err(|e| ServeError::io("flush réponse texte", e))
}

/// Un champ de formulaire `multipart/form-data`.
#[derive(Debug)]
struct MultipartField {
    name: String,
    #[allow(
        dead_code,
        reason = "conservé pour le diagnostic des parties multipart"
    )]
    filename: Option<String>,
    data: Vec<u8>,
}

fn text_field(fields: &[MultipartField], name: &str) -> Option<String> {
    let value = fields
        .iter()
        .find(|field| field.name == name)
        .map(|field| String::from_utf8_lossy(&field.data).trim().to_string())?;
    (!value.is_empty()).then_some(value)
}

/// Extrait la valeur `boundary` d'un Content-Type multipart.
fn extract_boundary(content_type: &str) -> Option<String> {
    for part in content_type.split(';') {
        let Some((name, value)) = part.trim().split_once('=') else {
            continue;
        };
        if name.trim().eq_ignore_ascii_case("boundary") {
            let value = value.trim().trim_matches('"');
            if !value.is_empty() {
                return Some(value.to_string());
            }
        }
    }
    None
}

/// Parse un corps `multipart/form-data` à la main (le serveur n'a pas de crate HTTP).
fn parse_multipart(body: &[u8], boundary: &str) -> ServeResult<Vec<MultipartField>> {
    let delimiter = format!("--{boundary}");
    let delimiter = delimiter.as_bytes();
    let closing = format!("\r\n--{boundary}");
    let closing = closing.as_bytes();
    let mut index = find_sub(body, delimiter)
        .ok_or_else(|| ServeError::Http("délimiteur multipart absent".to_string()))?;
    let mut fields = Vec::new();
    loop {
        let after_delim = index + delimiter.len();
        // `--` après le délimiteur marque la frontière de clôture.
        if body[after_delim..].starts_with(b"--") {
            break;
        }
        let part_start = if body[after_delim..].starts_with(b"\r\n") {
            after_delim + 2
        } else {
            return Err(ServeError::Http(
                "délimiteur multipart sans fin de ligne".to_string(),
            ));
        };
        let next = find_sub(&body[part_start..], closing).ok_or_else(|| {
            ServeError::Http("partie multipart sans délimiteur suivant".to_string())
        })?;
        fields.push(parse_part(&body[part_start..part_start + next])?);
        // Repositionne sur le délimiteur suivant (saute le `\r\n` de tête).
        index = part_start + next + 2;
    }
    Ok(fields)
}

fn parse_part(part: &[u8]) -> ServeResult<MultipartField> {
    let split = find_sub(part, b"\r\n\r\n")
        .ok_or_else(|| ServeError::Http("partie multipart sans en-têtes".to_string()))?;
    let headers = std::str::from_utf8(&part[..split])
        .map_err(|e| ServeError::Http(format!("en-têtes multipart non UTF-8: {e}")))?;
    let data = part[split + 4..].to_vec();
    let mut name = None;
    let mut filename = None;
    for line in headers.split("\r\n") {
        let Some((header, value)) = line.split_once(':') else {
            continue;
        };
        if header.trim().eq_ignore_ascii_case("content-disposition") {
            (name, filename) = parse_content_disposition(value);
        }
    }
    let name = name.ok_or_else(|| ServeError::Http("Content-Disposition sans name".to_string()))?;
    Ok(MultipartField {
        name,
        filename,
        data,
    })
}

fn parse_content_disposition(value: &str) -> (Option<String>, Option<String>) {
    let mut name = None;
    let mut filename = None;
    for part in value.split(';') {
        let Some((key, raw)) = part.trim().split_once('=') else {
            continue;
        };
        let raw = raw.trim().trim_matches('"').to_string();
        match key.trim().to_ascii_lowercase().as_str() {
            "name" => name = Some(raw),
            "filename" => filename = Some(raw),
            _ => {}
        }
    }
    (name, filename)
}

fn find_sub(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

/// Décode un WAV en échantillons mono f32 @ 16 kHz (format Whisper).
fn wav_to_mono_16k(bytes: &[u8]) -> ServeResult<Vec<f32>> {
    let mut reader = WavReader::new(Cursor::new(bytes))
        .map_err(|e| ServeError::Http(format!("WAV invalide: {e}")))?;
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        SampleFormat::Int => match spec.bits_per_sample {
            16 => reader
                .samples::<i16>()
                .map(|sample| f32::from(sample.unwrap_or(0)) / f32::from(i16::MAX))
                .collect(),
            24 | 32 => reader
                .samples::<i32>()
                .map(|sample| sample.unwrap_or(0) as f32 / i32::MAX as f32)
                .collect(),
            other => {
                return Err(ServeError::Http(format!(
                    "profondeur WAV non supportée: {other} bits"
                )))
            }
        },
        SampleFormat::Float => reader
            .samples::<f32>()
            .map(|sample| sample.unwrap_or(0.0))
            .collect(),
    };
    let mono = to_mono(&samples, spec.channels as usize);
    Ok(downsample(&mono, spec.sample_rate, WHISPER_SAMPLE_RATE))
}

/// Encode des échantillons mono f32 en WAV PCM i16 en mémoire.
fn mono_f32_to_wav(samples: &[f32], sample_rate: u32) -> ServeResult<Vec<u8>> {
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut cursor = Cursor::new(Vec::new());
    {
        let mut writer = WavWriter::new(&mut cursor, spec)
            .map_err(|e| ServeError::Http(format!("initialisation WAV: {e}")))?;
        for &sample in samples {
            let value = (sample.clamp(-1.0, 1.0) * f32::from(i16::MAX)) as i16;
            writer
                .write_sample(value)
                .map_err(|e| ServeError::Http(format!("écriture sample WAV: {e}")))?;
        }
        writer
            .finalize()
            .map_err(|e| ServeError::Http(format!("finalisation WAV: {e}")))?;
    }
    Ok(cursor.into_inner())
}

/// Replie des canaux entrelacés en mono par moyenne.
fn to_mono(samples: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return samples.to_vec();
    }
    samples
        .chunks(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

/// Rééchantillonne linéairement vers `target_rate` (sur/sous-échantillonnage).
fn downsample(samples: &[f32], input_rate: u32, target_rate: u32) -> Vec<f32> {
    if input_rate == target_rate || samples.is_empty() {
        return samples.to_vec();
    }
    if input_rate < target_rate {
        let ratio = f64::from(target_rate) / f64::from(input_rate);
        let out_len = (samples.len() as f64 * ratio) as usize;
        return (0..out_len)
            .map(|i| samples[((i as f64 / ratio) as usize).min(samples.len() - 1)])
            .collect();
    }
    let ratio = f64::from(input_rate) / f64::from(target_rate);
    let out_len = (samples.len() as f64 / ratio) as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let start = (i as f64 * ratio) as usize;
        let end = (((i + 1) as f64) * ratio) as usize;
        let slice = &samples[start..end.min(samples.len())];
        if !slice.is_empty() {
            out.push(slice.iter().sum::<f32>() / slice.len() as f32);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::super::args::ServeArgs;
    use super::*;

    fn multipart_body(boundary: &str, file: &[u8], language: Option<&str>) -> Vec<u8> {
        let mut body = Vec::new();
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(
            b"Content-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n",
        );
        body.extend_from_slice(b"Content-Type: audio/wav\r\n\r\n");
        body.extend_from_slice(file);
        body.extend_from_slice(b"\r\n");
        if let Some(language) = language {
            body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
            body.extend_from_slice(b"Content-Disposition: form-data; name=\"language\"\r\n\r\n");
            body.extend_from_slice(language.as_bytes());
            body.extend_from_slice(b"\r\n");
        }
        body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
        body
    }

    #[test]
    fn extract_boundary_reads_content_type_param() {
        assert_eq!(
            extract_boundary("multipart/form-data; boundary=abc123").as_deref(),
            Some("abc123")
        );
        assert_eq!(
            extract_boundary("multipart/form-data; boundary=\"quoted\"").as_deref(),
            Some("quoted")
        );
        assert_eq!(extract_boundary("application/json"), None);
    }

    #[test]
    fn parse_multipart_extracts_file_and_text_fields() {
        let body = multipart_body("XBOUND", b"RIFFDATA", Some("fr"));

        let fields = parse_multipart(&body, "XBOUND").expect("invariant: multipart parsable");

        let file = fields
            .iter()
            .find(|field| field.name == "file")
            .expect("invariant: champ file présent");
        assert_eq!(file.data, b"RIFFDATA");
        assert_eq!(file.filename.as_deref(), Some("a.wav"));
        assert_eq!(text_field(&fields, "language").as_deref(), Some("fr"));
    }

    #[test]
    fn parse_multipart_preserves_binary_payload_with_crlf() {
        let payload = vec![0x00, 0x0d, 0x0a, 0xff, 0x52, 0x49];
        let body = multipart_body("BND", &payload, None);

        let fields = parse_multipart(&body, "BND").expect("invariant: multipart binaire parsable");

        let file = fields
            .iter()
            .find(|field| field.name == "file")
            .expect("invariant: champ file présent");
        assert_eq!(file.data, payload);
    }

    #[test]
    fn parse_multipart_missing_delimiter_errors() {
        let error = parse_multipart(b"pas de delimiteur ici", "BND")
            .expect_err("invariant: délimiteur requis");

        assert!(error.to_string().contains("délimiteur multipart absent"));
    }

    #[test]
    fn wav_roundtrips_through_mono_16k() {
        let samples: Vec<f32> = (0..1600).map(|i| (i as f32 * 0.05).sin() * 0.5).collect();
        let wav = mono_f32_to_wav(&samples, 16_000).expect("invariant: WAV encodé");
        assert_eq!(&wav[..4], b"RIFF");

        let decoded = wav_to_mono_16k(&wav).expect("invariant: WAV décodé");

        assert_eq!(decoded.len(), samples.len());
        for (a, b) in samples.iter().zip(decoded.iter()) {
            assert!((a - b).abs() < 1e-3, "diff trop grand: {a} vs {b}");
        }
    }

    #[test]
    fn wav_decode_downmixes_and_resamples() {
        // Stéréo 8 kHz → mono 16 kHz : la longueur double (upsampling ×2).
        let spec = WavSpec {
            channels: 2,
            sample_rate: 8_000,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut cursor = Cursor::new(Vec::new());
        {
            let mut writer = WavWriter::new(&mut cursor, spec).expect("invariant: writer WAV");
            for _ in 0..800 {
                writer.write_sample(1000_i16).expect("invariant: L");
                writer.write_sample(-1000_i16).expect("invariant: R");
            }
            writer.finalize().expect("invariant: finalize");
        }

        let decoded = wav_to_mono_16k(&cursor.into_inner()).expect("invariant: WAV décodé");

        // 800 frames @ 8 kHz → 1600 échantillons @ 16 kHz, moyenne L/R ≈ 0.
        assert_eq!(decoded.len(), 1600);
        assert!(decoded.iter().all(|sample| sample.abs() < 1e-3));
    }

    #[test]
    fn transcription_without_model_returns_clear_error() {
        let args = ServeArgs::parse(Vec::<String>::new()).expect("invariant: args valides");
        let mut audio = AudioState::new(&args);
        let mut stream = Cursor::new(Vec::new());
        let wav = mono_f32_to_wav(&[0.1_f32; 1600], 16_000).expect("invariant: WAV");
        let body = multipart_body("BND", &wav, None);

        let error = handle_transcription(
            &mut stream,
            &mut audio,
            Some("multipart/form-data; boundary=BND"),
            &body,
        )
        .expect_err("invariant: STT non configuré refusé");

        assert!(error.to_string().contains("STT non configuré"));
    }

    #[test]
    fn speech_without_model_returns_clear_error() {
        let args = ServeArgs::parse(Vec::<String>::new()).expect("invariant: args valides");
        let mut audio = AudioState::new(&args);
        let mut stream = Cursor::new(Vec::new());
        let body = br#"{"model":"reti-tts","input":"Bonjour"}"#;

        let error = handle_speech(&mut stream, &mut audio, body)
            .expect_err("invariant: TTS non configuré refusé");

        assert!(error.to_string().contains("TTS non configuré"));
    }

    #[test]
    fn speech_rejects_empty_input() {
        let args = ServeArgs::parse(Vec::<String>::new()).expect("invariant: args valides");
        let mut audio = AudioState::new(&args);
        let mut stream = Cursor::new(Vec::new());
        let body = br#"{"model":"reti-tts","input":"   "}"#;

        let error =
            handle_speech(&mut stream, &mut audio, body).expect_err("invariant: input vide refusé");

        assert!(error.to_string().contains("'input' requis"));
    }
}
