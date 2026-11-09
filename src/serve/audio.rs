//! Endpoints audio OpenAI-compatibles de `saragossa serve`.
//!
//! Deux routes purement additives :
//! - `POST /v1/audio/transcriptions` : STT Whisper, requête `multipart/form-data`
//!   (`file` = WAV, plus `model`/`language`/`response_format` optionnels), réponse
//!   JSON `{"text": ...}` (ou texte brut si `response_format=text`).
//! - `POST /v1/audio/speech` : TTS Qwen3, requête JSON `{"model", "input", "voice"?}`,
//!   réponse WAV par défaut, ou PCM s16le mono chunked avec `stream: true`.
//!
//! Les modèles sont opt-in (`--stt-model`, `--tts-model`) et chargés paresseusement
//! au premier appel : sans configuration, la route renvoie une erreur 400 claire.

use std::io::{Cursor, Write};
use std::path::PathBuf;

use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use serde::Deserialize;
use serde_json::json;

use saragossa::tts::TTS_GENERATION_HARD_CAP;
#[cfg(all(target_os = "macos", feature = "metal"))]
use saragossa::MetalExecutor;
use saragossa::{ForwardRuntime, TtsModel, TtsSynthesisOutput, WhisperModel};

use super::args::ServeArgs;
use super::error::{ServeError, ServeResult};
use super::http::{send_json, write_headers};
use crate::RuntimeKind;

/// Fréquence d'échantillonnage attendue par l'encodeur Whisper.
const WHISPER_SAMPLE_RATE: u32 = 16_000;
/// De la téléphonie à la prise de son haute résolution ; borne l'amplification à ×2.
const WAV_SAMPLE_RATE_RANGE: std::ops::RangeInclusive<u32> = 8_000..=192_000;
/// Valeurs entrelacées, tous canaux confondus : au plus 32 Mo décodés en f32.
const MAX_WAV_SAMPLES: u64 = 8_000_000;
/// Budget minimal conservé pour les clauses courtes.
const MIN_TTS_MAX_FRAMES: usize = 160;
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

// Seule la synthèse est substituable : les tests conservent le choix WAV/PCM
// du gestionnaire de production, sans charger de poids ni initialiser Metal.
pub(super) trait SpeechSynthesis {
    fn synthesize(&mut self, text: &str, max_frames: usize) -> ServeResult<TtsSynthesisOutput>;
    fn stream_sample_rate(&mut self) -> ServeResult<u32>;
    fn synthesize_streaming(
        &mut self,
        text: &str,
        max_frames: usize,
        on_audio: &mut dyn FnMut(&[f32]) -> saragossa::Result<()>,
    ) -> ServeResult<TtsSynthesisOutput>;
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

impl SpeechSynthesis for AudioState {
    fn synthesize(&mut self, text: &str, max_frames: usize) -> ServeResult<TtsSynthesisOutput> {
        self.ensure_tts_loaded()?;
        let model = self
            .tts
            .model
            .as_ref()
            .ok_or_else(|| ServeError::args("modèle TTS absent après chargement"))?;
        Ok(model.synthesize_default(text, max_frames)?)
    }

    fn stream_sample_rate(&mut self) -> ServeResult<u32> {
        self.ensure_tts_loaded()?;
        let model = self
            .tts
            .model
            .as_ref()
            .ok_or_else(|| ServeError::args("modèle TTS absent après chargement"))?;
        Ok(model.sample_rate())
    }

    fn synthesize_streaming(
        &mut self,
        text: &str,
        max_frames: usize,
        on_audio: &mut dyn FnMut(&[f32]) -> saragossa::Result<()>,
    ) -> ServeResult<TtsSynthesisOutput> {
        let model = self
            .tts
            .model
            .as_ref()
            .ok_or_else(|| ServeError::args("modèle TTS absent après chargement"))?;
        Ok(model.synthesize_default_streaming(text, max_frames, on_audio)?)
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

fn tts_max_frames(text: &str, override_value: Option<&str>) -> usize {
    // Nom canonique depuis l'origine : pas d'alias `RETI_` à honorer ici.
    override_value
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or_else(|| {
            text.chars()
                .count()
                .saturating_mul(2)
                .max(MIN_TTS_MAX_FRAMES)
        })
        .min(TTS_GENERATION_HARD_CAP)
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
    audio: &mut impl SpeechSynthesis,
    body: &[u8],
) -> ServeResult<()> {
    handle_speech_with(
        stream,
        body,
        std::env::var(TTS_MAX_FRAMES_ENV).ok().as_deref(),
        |stream, text, max_frames, streaming| {
            if !streaming {
                return send_speech(stream, &audio.synthesize(text, max_frames)?);
            }
            // Le chargement peut encore produire une erreur HTTP avant les en-têtes.
            let sample_rate = audio.stream_sample_rate()?;
            send_speech_stream(stream, sample_rate, |on_audio| {
                audio.synthesize_streaming(text, max_frames, on_audio)
            })
        },
    )
}

// L'injection permet de vérifier les arguments transmis par la route sans
// charger de poids ni initialiser Metal.
fn handle_speech_with<S: Write>(
    stream: &mut S,
    body: &[u8],
    override_value: Option<&str>,
    synthesize: impl FnOnce(&mut S, &str, usize, bool) -> ServeResult<()>,
) -> ServeResult<()> {
    let request: SpeechRequest = serde_json::from_slice(body)
        .map_err(|e| ServeError::json("désérialisation audio/speech", e))?;
    let input = request.input.trim();
    if input.is_empty() {
        return Err(ServeError::Http(
            "champ 'input' requis pour audio/speech".to_string(),
        ));
    }
    synthesize(
        stream,
        input,
        tts_max_frames(input, override_value),
        request.stream,
    )
}

fn send_speech<S: Write>(stream: &mut S, output: &TtsSynthesisOutput) -> ServeResult<()> {
    let wav = mono_f32_to_wav(&output.samples, output.sample_rate)?;
    let headers = vec![(
        "X-Saragossa-TTS-Frame-Limit-Reached",
        output.frame_limit_reached.to_string(),
    )];
    write_headers(stream, 200, "OK", "audio/wav", Some(wav.len()), headers)?;
    stream
        .write_all(&wav)
        .map_err(|e| ServeError::io("écriture réponse WAV", e))?;
    stream
        .flush()
        .map_err(|e| ServeError::io("flush réponse WAV", e))
}

// L'écriture reste synchrone sur le thread moteur : pas de producteur détaché
// ni de file audio non bornée. Les sockets ont déjà un timeout d'écriture.
fn send_speech_stream<S: Write>(
    stream: &mut S,
    sample_rate: u32,
    synthesize: impl FnOnce(
        &mut dyn FnMut(&[f32]) -> saragossa::Result<()>,
    ) -> ServeResult<TtsSynthesisOutput>,
) -> ServeResult<()> {
    let result = (|| {
        write_headers(
            stream,
            200,
            "OK",
            "audio/pcm",
            None,
            vec![
                ("Transfer-Encoding", "chunked".into()),
                ("X-Saragossa-Audio-Sample-Rate", sample_rate.to_string()),
                ("Trailer", "X-Saragossa-TTS-Frame-Limit-Reached".into()),
            ],
        )?;
        stream
            .flush()
            .map_err(|e| ServeError::io("flush en-têtes PCM", e))?;
        let output = synthesize(&mut |samples| {
            write_pcm_chunk(stream, samples).map_err(|source| saragossa::InferError::Io {
                path: PathBuf::from("/v1/audio/speech"),
                source,
            })
        })?;
        // Le budget atteint n'est connu qu'après la génération : trailer HTTP.
        write!(
            stream,
            "0\r\nX-Saragossa-TTS-Frame-Limit-Reached: {}\r\n\r\n",
            output.frame_limit_reached
        )
        .and_then(|()| stream.flush())
        .map_err(|e| ServeError::io("fin réponse PCM", e))
    })();
    if let Err(error) = result {
        // Une réponse a pu commencer, même si l'écriture des en-têtes a échoué.
        // Fermer sans chunk terminal signale la troncature ; pas de second HTTP.
        eprintln!("saragossa serve audio stream interrompu: {error}");
    }
    Ok(())
}

fn write_pcm_chunk<S: Write>(stream: &mut S, samples: &[f32]) -> std::io::Result<()> {
    if samples.is_empty() {
        return Ok(());
    }
    let pcm: Vec<u8> = samples
        .iter()
        .flat_map(|&sample| pcm_i16(sample).to_le_bytes())
        .collect();
    write!(stream, "{:x}\r\n", pcm.len())?;
    stream.write_all(&pcm)?;
    stream.write_all(b"\r\n")?;
    stream.flush()
}

fn pcm_i16(sample: f32) -> i16 {
    (sample.clamp(-1.0, 1.0) * f32::from(i16::MAX)) as i16
}

/// Requête `/v1/audio/speech`. Le backend expose une voix VoiceDesign unique,
/// donc `model`/`voice` (ignorés) sont acceptés sans effet.
#[derive(Debug, Deserialize)]
struct SpeechRequest {
    /// Texte à synthétiser.
    input: String,
    /// Extension explicite : PCM s16le mono chunked au lieu du WAV complet.
    #[serde(default)]
    stream: bool,
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
    let invalid_wav = |e| ServeError::Http(format!("WAV invalide: {e}"));
    let mut cursor = Cursor::new(bytes);
    let file_len = hound::read_wave_header(&mut cursor).map_err(invalid_wav)?;
    if file_len > bytes.len() as u64 {
        return Err(ServeError::Http(
            "WAV tronqué: taille RIFF supérieure au corps".into(),
        ));
    }
    let reader = WavReader::new(Cursor::new(bytes)).map_err(invalid_wav)?;
    let spec = reader.spec();
    let samples_len = u64::from(reader.len());
    let data_start = reader.into_inner().position() as usize;
    // Le curseur de Hound suit immédiatement la taille du bloc data.
    let size_bytes = bytes
        .get(data_start.saturating_sub(4)..data_start)
        .and_then(|raw| raw.try_into().ok())
        .ok_or_else(|| ServeError::Http("WAV sans taille data".into()))?;
    let data_len = u32::from_le_bytes(size_bytes);
    // Hound déduit la longueur de l'itérateur de l'en-tête, sans vérifier le corps.
    // Vérifier les octets stockés (y compris le padding des samples) avant collect.
    if data_start as u64 + u64::from(data_len) > file_len {
        return Err(ServeError::Http(
            "WAV tronqué: bloc data supérieur au corps RIFF".into(),
        ));
    }
    validate_wav_limits(spec.sample_rate, spec.channels, samples_len)?;
    let mut reader = WavReader::new(Cursor::new(bytes)).map_err(invalid_wav)?;
    let samples: Vec<f32> = match spec.sample_format {
        SampleFormat::Int => match spec.bits_per_sample {
            16 => reader
                .samples::<i16>()
                .map(|sample| sample.map(|value| f32::from(value) / f32::from(i16::MAX)))
                .collect::<Result<Vec<_>, _>>(),
            bits @ (24 | 32) => {
                // hound rend la valeur 24 bits telle quelle dans un i32, sans la décaler.
                let full_scale = ((1_u64 << (bits - 1)) - 1) as f32;
                reader
                    .samples::<i32>()
                    .map(|sample| sample.map(|value| value as f32 / full_scale))
                    .collect()
            }
            other => {
                return Err(ServeError::Http(format!(
                    "profondeur WAV non supportée: {other} bits"
                )))
            }
        },
        SampleFormat::Float => reader.samples::<f32>().collect(),
    }
    .map_err(invalid_wav)?;
    let mono = to_mono(&samples, spec.channels as usize);
    downsample(&mono, spec.sample_rate, WHISPER_SAMPLE_RATE)
}

fn validate_wav_limits(sample_rate: u32, channels: u16, samples: u64) -> ServeResult<()> {
    if !WAV_SAMPLE_RATE_RANGE.contains(&sample_rate) {
        return Err(ServeError::Http(format!(
            "taux WAV hors plage 8000–192000 Hz: {sample_rate}"
        )));
    }
    if channels == 0 {
        return Err(ServeError::Http("WAV sans canal".into()));
    }
    if samples > MAX_WAV_SAMPLES {
        return Err(ServeError::Http(format!(
            "WAV dépasse {MAX_WAV_SAMPLES} échantillons"
        )));
    }
    Ok(())
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
            let value = pcm_i16(sample);
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
fn downsample(samples: &[f32], input_rate: u32, target_rate: u32) -> ServeResult<Vec<f32>> {
    validate_wav_limits(input_rate, 1, samples.len() as u64)?;
    validate_wav_limits(target_rate, 1, 0)?;
    if input_rate == target_rate || samples.is_empty() {
        return Ok(samples.to_vec());
    }
    if input_rate < target_rate {
        let ratio = f64::from(target_rate) / f64::from(input_rate);
        let out_len = (samples.len() as f64 * ratio) as usize;
        validate_wav_limits(target_rate, 1, out_len as u64)?;
        return Ok((0..out_len)
            .map(|i| samples[((i as f64 / ratio) as usize).min(samples.len() - 1)])
            .collect());
    }
    let ratio = f64::from(input_rate) / f64::from(target_rate);
    let out_len = (samples.len() as f64 / ratio) as usize;
    validate_wav_limits(target_rate, 1, out_len as u64)?;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let start = (i as f64 * ratio) as usize;
        let end = (((i + 1) as f64) * ratio) as usize;
        let slice = &samples[start..end.min(samples.len())];
        if !slice.is_empty() {
            out.push(slice.iter().sum::<f32>() / slice.len() as f32);
        }
    }
    Ok(out)
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
    fn tts_budget_tracks_unicode_text_and_respects_bounded_override() {
        assert_eq!(tts_max_frames("", None), 160);
        assert_eq!(tts_max_frames("Bonjour.", None), 160);
        assert_eq!(tts_max_frames(&"é".repeat(200), None), 400);
        assert_eq!(tts_max_frames(&"a".repeat(320), None), 640);
        assert_eq!(tts_max_frames(&"a".repeat(10_000), None), 800);
        for invalid in ["", "0", "-1", "bad", "99999999999999999999999999999"] {
            assert_eq!(tts_max_frames(&"a".repeat(200), Some(invalid)), 400);
        }
        assert_eq!(tts_max_frames(&"a".repeat(200), Some(" 12 ")), 12);
        assert_eq!(tts_max_frames("Bonjour.", Some("640")), 640);
        assert_eq!(tts_max_frames("Bonjour.", Some("2000")), 800);
        assert_eq!(
            tts_max_frames("Bonjour.", Some(&usize::MAX.to_string())),
            800
        );
    }

    #[test]
    fn speech_passes_cleaned_text_and_budget_to_synthesis() -> ServeResult<()> {
        for (text, override_value, expected_frames) in [
            ("Bonjour.".to_string(), None, 160),
            ("é".repeat(200), None, 400),
            ("a".repeat(500), None, 800),
            ("é".repeat(200), Some("160"), 160),
            ("é".repeat(200), Some("12"), 12),
            ("é".repeat(200), Some("2000"), 800),
            ("é".repeat(200), Some("0"), 400),
        ] {
            let body = json!({ "input": format!("  {text}\n") }).to_string();
            let mut response = Vec::new();
            let mut called = false;
            handle_speech_with(
                &mut response,
                body.as_bytes(),
                override_value,
                |stream, input, max_frames, streaming| {
                    called = true;
                    assert!(!streaming);
                    assert_eq!(input, text);
                    assert_eq!(max_frames, expected_frames);
                    send_speech(
                        stream,
                        &TtsSynthesisOutput {
                            codes: Vec::new(),
                            samples: vec![0.0],
                            sample_rate: 24_000,
                            frame_limit_reached: false,
                        },
                    )
                },
            )?;
            assert!(called);
            assert!(response.starts_with(b"HTTP/1.1 200 OK\r\n"));
        }
        Ok(())
    }

    #[test]
    fn speech_returns_wav_and_explicit_limit_header() -> ServeResult<()> {
        for reached in [false, true] {
            let output = TtsSynthesisOutput {
                codes: vec![vec![1; 16]],
                samples: vec![0.0, 0.25, -0.25],
                sample_rate: 24_000,
                frame_limit_reached: reached,
            };
            let mut response = Vec::new();
            send_speech(&mut response, &output)?;
            let split = response
                .windows(4)
                .position(|w| w == b"\r\n\r\n")
                .expect("invariant: réponse HTTP avec en-têtes");
            let headers = String::from_utf8_lossy(&response[..split]);
            assert!(headers.starts_with("HTTP/1.1 200 OK"));
            assert!(headers.contains("Content-Type: audio/wav"));
            assert!(headers.contains(&format!("X-Saragossa-TTS-Frame-Limit-Reached: {reached}")));
            assert_eq!(
                &response[split + 4..],
                mono_f32_to_wav(&output.samples, 24_000)?
            );
        }
        Ok(())
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

    fn pcm_wav(rate: u32, channels: u16, bits: u16, samples: u32) -> Vec<u8> {
        let block_align = channels * (bits / 8);
        let data_len = samples * u32::from(bits / 8);
        let mut wav = b"RIFF".to_vec();
        wav.extend_from_slice(&(36 + data_len).to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt \x10\0\0\0\x01\0");
        wav.extend_from_slice(&channels.to_le_bytes());
        wav.extend_from_slice(&rate.to_le_bytes());
        wav.extend_from_slice(&(rate * u32::from(block_align)).to_le_bytes());
        wav.extend_from_slice(&block_align.to_le_bytes());
        wav.extend_from_slice(&bits.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_len.to_le_bytes());
        wav.resize(44 + data_len as usize, 0);
        wav
    }

    fn assert_wav_rejected(wav: &[u8], expected: &str) {
        let error = wav_to_mono_16k(wav).expect_err("invariant: WAV hostile refusé");
        assert!(
            matches!(&error, ServeError::Http(message) if message.contains(expected)),
            "{error}"
        );

        // Exerce l'appelant du thread moteur : l'erreur doit précéder le STT,
        // puis devenir une réponse 400, sans device ni interception de panique.
        let body = multipart_body("BND", wav, None);
        let mut raw = format!(
            "POST /v1/audio/transcriptions HTTP/1.1\r\nContent-Type: multipart/form-data; boundary=BND\r\nContent-Length: {}\r\n\r\n",
            body.len()
        ).into_bytes();
        raw.extend_from_slice(&body);
        let request = super::super::http::read_request(
            &mut Cursor::new(raw),
            std::time::Instant::now() + std::time::Duration::from_secs(30),
        )
        .expect("invariant: requête HTTP valide")
        .expect("invariant: requête présente");
        let args = ServeArgs::parse(Vec::<String>::new()).expect("invariant: args valides");
        let mut state = super::super::state::ServeState::new(&args);
        let mut response = Vec::new();
        let error = super::super::http::handle_job(&mut response, &mut state, &request)
            .expect_err("invariant: WAV refusé avant inférence");
        assert!(matches!(error, ServeError::Http(_)));
        assert!(response.starts_with(b"HTTP/1.1 400 Bad Request\r\n"));
        assert!(state.audio_mut().stt.model.is_none());
    }

    #[test]
    fn wav_rejects_invalid_rates() {
        for rate in [0, 1, 7_999, 192_001] {
            assert_wav_rejected(&pcm_wav(rate, 1, 16, 8), "taux WAV");
        }
        assert_wav_rejected(&pcm_wav(1, 1, 16, 512_000), "taux WAV");
    }

    #[test]
    fn wav_rejects_zero_channels() {
        assert_wav_rejected(&pcm_wav(16_000, 0, 16, 8), "zero channels");
    }

    #[test]
    fn wav_rejects_lying_headers_and_truncation() {
        let wav = pcm_wav(16_000, 1, 16, 8);
        let mut lying = wav.clone();
        lying[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_wav_rejected(&lying, "taille RIFF");
        lying = wav.clone();
        lying[40..44].copy_from_slice(&(u32::MAX - 1).to_le_bytes());
        assert_wav_rejected(&lying, "bloc data");
        assert_wav_rejected(&wav[..wav.len() - 1], "tronqué");
        // La taille totale suffit ici ; seul le contrôle depuis le début de
        // data détecte le sample manquant, même si RIFF a été réajusté.
        lying = wav[..wav.len() - 1].to_vec();
        let riff_len = (lying.len() - 8) as u32;
        lying[4..8].copy_from_slice(&riff_len.to_le_bytes());
        assert_wav_rejected(&lying, "bloc data");
    }

    #[test]
    fn wav_rejects_unreadable_samples_in_every_format() {
        for (format, bits, storage_bits) in [(1_u16, 16_u16, 24), (1, 24, 40), (3, 32, 40)] {
            let mut wav = pcm_wav(16_000, 1, storage_bits, 2);
            wav[20..22].copy_from_slice(&format.to_le_bytes());
            wav[34..36].copy_from_slice(&bits.to_le_bytes());
            assert_wav_rejected(&wav, "sample has more bits");
        }
    }

    #[test]
    fn wav_rejects_unsupported_8_bit_pcm() {
        assert_wav_rejected(&pcm_wav(16_000, 1, 8, 8), "non supportée: 8 bits");
    }

    #[test]
    fn wav_sample_limit_is_inclusive() {
        let limit = MAX_WAV_SAMPLES as u32;
        let decoded = wav_to_mono_16k(&pcm_wav(32_000, 2, 16, limit))
            .expect("invariant: plafond exact accepté");
        assert_eq!(decoded.len(), limit as usize / 4);
        assert!(decoded.iter().all(|&sample| sample == 0.0));
        assert_wav_rejected(&pcm_wav(32_000, 1, 16, limit + 1), "échantillons");
    }

    #[test]
    fn downsample_checks_limits_before_copy_or_resampling() {
        for rate in [0, 1, 7_999, 192_001] {
            for samples in [&[][..], &[1.0][..]] {
                assert!(matches!(
                    downsample(samples, rate, 16_000),
                    Err(ServeError::Http(_))
                ));
                assert!(matches!(
                    downsample(samples, 16_000, rate),
                    Err(ServeError::Http(_))
                ));
            }
        }
        let too_long = vec![0.0; (MAX_WAV_SAMPLES + 1) as usize];
        assert!(matches!(
            downsample(&too_long, 8_000, 8_000),
            Err(ServeError::Http(_))
        ));
        assert!(matches!(
            downsample(&too_long[..400_000], 8_000, 192_000),
            Err(ServeError::Http(_))
        ));
    }

    #[test]
    fn downsample_preserves_valid_samples() -> ServeResult<()> {
        assert_eq!(
            downsample(&[1.0, -1.0], 8_000, 16_000)?,
            [1.0, 1.0, -1.0, -1.0]
        );
        assert_eq!(
            downsample(&[1.0, -1.0, 0.5, 0.5], 32_000, 16_000)?,
            [0.0, 0.5]
        );
        assert_eq!(downsample(&[1.0, -1.0], 16_000, 16_000)?, [1.0, -1.0]);
        let decoded = wav_to_mono_16k(&pcm_wav(192_000, 1, 16, 12))?;
        assert_eq!(decoded, [0.0]);
        Ok(())
    }

    #[test]
    fn wav_preserves_supported_sample_formats() -> ServeResult<()> {
        for bits in [16, 24, 32] {
            let mut wav = pcm_wav(16_000, 1, bits, 1);
            wav[44..].fill(0xff);
            wav[44 + usize::from(bits / 8) - 1] = 0x7f;
            assert_eq!(wav_to_mono_16k(&wav)?, [1.0]);
        }
        let mut wav = pcm_wav(16_000, 1, 32, 1);
        wav[20..22].copy_from_slice(&3_u16.to_le_bytes());
        wav[44..].copy_from_slice(&0.5_f32.to_le_bytes());
        assert_eq!(wav_to_mono_16k(&wav)?, [0.5]);
        Ok(())
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
        for streaming in [false, true] {
            let body = json!({"input": "Bonjour", "stream": streaming}).to_string();
            let error = handle_speech(&mut stream, &mut audio, body.as_bytes())
                .expect_err("invariant: TTS non configuré refusé");
            assert!(error.to_string().contains("TTS non configuré"));
            assert!(stream.get_ref().is_empty());
        }
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

#[cfg(test)]
mod stream_tests;
