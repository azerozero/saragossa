//! Débruitage duplex sur le thread de connexion, indépendant du moteur Metal.
use super::dispatch::{Reservation, ServeStream};
use super::enhance_body::{Body, Multipart};
use super::error::{ServeError, ServeResult};
use super::http::{reply_on_error, send_error, HttpRequest};
use saragossa::denoise::wav::{pcm16_stream_header, WavStream};
use saragossa::denoise::{DeepFilterNet3, DenoiseStream, StreamResampler, SAMPLE_RATE};
use saragossa::runtime_flags::{enhance_idle_timeout, enhance_max_streams};
use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex};

#[cfg(feature = "ws")]
pub(super) mod pcm;
#[cfg(feature = "ws")]
pub(super) mod ws;

pub(super) struct EnhanceService {
    path: Option<PathBuf>,
    model: Mutex<Option<Arc<DeepFilterNet3>>>,
    active: Arc<AtomicUsize>,
}

impl EnhanceService {
    pub(super) fn new(path: Option<PathBuf>) -> Self {
        Self {
            path,
            model: Mutex::new(None),
            active: Arc::new(AtomicUsize::new(0)),
        }
    }
    fn model(&self) -> ServeResult<Arc<DeepFilterNet3>> {
        let mut loaded = self
            .model
            .lock()
            .map_err(|_| ServeError::args("verrou DFN3 indisponible"))?;
        if let Some(model) = loaded.as_ref() {
            return Ok(Arc::clone(model));
        }
        let path = self.path.as_ref().ok_or_else(|| {
            ServeError::args(
                "endpoint enhance non configuré: --enhance-model <dfn3.safetensors> requis",
            )
        })?;
        let model = Arc::new(DeepFilterNet3::load(path)?);
        *loaded = Some(Arc::clone(&model));
        Ok(model)
    }

    pub(super) fn handle<S: ServeStream>(
        &self,
        socket: &mut S,
        request: &mut HttpRequest,
        length: usize,
    ) -> ServeResult<()> {
        // Même réservation RAII que les connexions : aucun état évincé
        // pendant un flux actif, place rendue à chaque sortie, erreur comprise.
        let Some(_slot) = Reservation::acquire(&self.active, 1, enhance_max_streams()) else {
            return send_error(socket, 503, "trop de flux enhance simultanés");
        };
        socket.configure(enhance_idle_timeout())?;
        serve(socket, request, length, |wav, started| {
            // Validation WAV avant tout chargement : fréquence invalide => 400.
            let model = self.model()?;
            let mut denoise = DenoiseStream::new(&model);
            pump(wav, started, |chunk, final_chunk| {
                if final_chunk {
                    denoise.finish()
                } else {
                    denoise.push(chunk)
                }
            })
        })
    }
}

enum Envelope<'a, S> {
    Raw(Body<'a, S>),
    Form(Multipart<Body<'a, S>>),
}
impl<S: Read + Write> Read for Envelope<'_, S> {
    fn read(&mut self, b: &mut [u8]) -> io::Result<usize> {
        match self {
            Self::Raw(r) => r.read(b),
            Self::Form(r) => r.read(b),
        }
    }
}
impl<S: Read + Write> Write for Envelope<'_, S> {
    fn write(&mut self, b: &[u8]) -> io::Result<usize> {
        match self {
            Self::Raw(r) => r.write(b),
            Self::Form(r) => r.write(b),
        }
    }
    fn flush(&mut self) -> io::Result<()> {
        match self {
            Self::Raw(r) => r.flush(),
            Self::Form(r) => r.flush(),
        }
    }
}

fn serve<S: Read + Write>(
    socket: &mut S,
    request: &mut HttpRequest,
    length: usize,
    process: impl FnOnce(&mut WavStream<Envelope<'_, S>>, &mut bool) -> ServeResult<()>,
) -> ServeResult<()> {
    let mut started = false;
    let result = (|| {
        let content_type = request.header("content-type").unwrap_or("").to_string();
        let chunked = request.header("transfer-encoding").is_some();
        let body = Body::new(socket, request.take_prefix(), length, chunked);
        let media_type = content_type
            .split(';')
            .next()
            .unwrap_or("")
            .trim()
            .to_ascii_lowercase();
        let envelope = if matches!(
            media_type.as_str(),
            "audio/wav" | "audio/x-wav" | "application/octet-stream"
        ) {
            Envelope::Raw(body)
        } else if media_type == "multipart/form-data" {
            let boundary = saragossa::wire_audio::extract_boundary(&content_type)
                .ok_or_else(|| ServeError::Http("boundary multipart absente".into()))?;
            Envelope::Form(
                Multipart::new(body, &boundary).map_err(|e| ServeError::Http(e.to_string()))?,
            )
        } else {
            return Err(ServeError::Http(
                "Content-Type audio/wav ou multipart/form-data requis".into(),
            ));
        };
        let mut wav = WavStream::new(envelope).map_err(|e| ServeError::Http(e.to_string()))?;
        process(&mut wav, &mut started)?;
        wav.finish_input()
            .map_err(|e| ServeError::io("fin du corps enhance", e))?;
        if let Envelope::Form(form) = wav.get_mut() {
            form.finish()
                .map_err(|e| ServeError::io("fin multipart enhance", e))?;
        }
        wav.get_mut()
            .write_all(b"0\r\n\r\n")
            .map_err(|e| ServeError::io("fin réponse enhance", e))?;
        wav.get_mut()
            .flush()
            .map_err(|e| ServeError::io("flush enhance", e))
    })();
    // Après le 200, fermeture sans chunk terminal : le client détecte le flux
    // tronqué. Ne jamais concaténer une deuxième réponse HTTP dans le WAV.
    if started {
        result
    } else {
        reply_on_error(socket, result)
    }
}

fn pump<R: Read + Write>(
    wav: &mut WavStream<R>,
    started: &mut bool,
    mut transform: impl FnMut(&[f32], bool) -> saragossa::Result<Vec<f32>>,
) -> ServeResult<()> {
    let mut resampler = StreamResampler::new(wav.sample_rate(), SAMPLE_RATE)?;
    *started = true;
    wav.get_mut().write_all(b"HTTP/1.1 200 OK\r\nContent-Type: audio/wav\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n")
        .map_err(|e| ServeError::io("en-têtes enhance", e))?;
    write_chunk(
        wav.get_mut(),
        &pcm16_stream_header().map_err(|e| ServeError::io("en-tête WAV", e))?,
    )?;
    loop {
        let chunk = wav
            .read_chunk()
            .map_err(|e| ServeError::Http(e.to_string()))?;
        if chunk.is_empty() {
            break;
        }
        write_samples(wav.get_mut(), &transform(&resampler.push(&chunk)?, false)?)?;
    }
    write_samples(wav.get_mut(), &transform(&resampler.finish(), false)?)?;
    write_samples(wav.get_mut(), &transform(&[], true)?)
}

fn write_samples(writer: &mut impl Write, samples: &[f32]) -> ServeResult<()> {
    let bytes: Vec<_> = samples
        .iter()
        .flat_map(|s| super::audio::pcm_i16(*s).to_le_bytes())
        .collect();
    write_chunk(writer, &bytes)
}
fn write_chunk(writer: &mut impl Write, bytes: &[u8]) -> ServeResult<()> {
    if bytes.is_empty() {
        return Ok(());
    }
    (|| -> io::Result<()> {
        write!(writer, "{:x}\r\n", bytes.len())?;
        writer.write_all(bytes)?;
        writer.write_all(b"\r\n")?;
        writer.flush()
    })()
    .map_err(|e| ServeError::io("écriture chunk enhance", e))
}

#[cfg(test)]
mod tests;

#[cfg(all(test, feature = "ws"))]
mod test_support;

#[cfg(all(test, feature = "ws"))]
mod latency;
