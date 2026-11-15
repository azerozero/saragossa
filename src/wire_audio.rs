//! Décodage des entrées audio du transport, sans modèle ni runtime GPU.
use hound::{SampleFormat, WavReader};
use std::io::Cursor;

/// Entrée WAV ou multipart invalide.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct AudioParseError(pub String);

/// Résultat de décodage audio.
pub type Result<T> = std::result::Result<T, AudioParseError>;

/// Fréquence PCM attendue par Whisper.
pub const WHISPER_SAMPLE_RATE: u32 = 16_000;
/// Fréquences acceptées pour les entrées WAV.
pub const WAV_SAMPLE_RATE_RANGE: std::ops::RangeInclusive<u32> = 8_000..=192_000;
/// Borne des valeurs PCM entrelacées avant allocation.
pub const MAX_WAV_SAMPLES: u64 = 8_000_000;

/// Un champ de formulaire `multipart/form-data`.
#[derive(Debug)]
pub struct MultipartField {
    pub name: String,
    #[allow(
        dead_code,
        reason = "conservé pour le diagnostic des parties multipart"
    )]
    pub filename: Option<String>,
    pub data: Vec<u8>,
}

pub fn text_field(fields: &[MultipartField], name: &str) -> Option<String> {
    let value = fields
        .iter()
        .find(|field| field.name == name)
        .map(|field| String::from_utf8_lossy(&field.data).trim().to_string())?;
    (!value.is_empty()).then_some(value)
}

/// Extrait la valeur `boundary` d'un Content-Type multipart.
pub fn extract_boundary(content_type: &str) -> Option<String> {
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
pub fn parse_multipart(body: &[u8], boundary: &str) -> Result<Vec<MultipartField>> {
    let delimiter = format!("--{boundary}");
    let delimiter = delimiter.as_bytes();
    let closing = format!("\r\n--{boundary}");
    let closing = closing.as_bytes();
    let mut index = find_sub(body, delimiter)
        .ok_or_else(|| AudioParseError("délimiteur multipart absent".to_string()))?;
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
            return Err(AudioParseError(
                "délimiteur multipart sans fin de ligne".to_string(),
            ));
        };
        let next = find_sub(&body[part_start..], closing).ok_or_else(|| {
            AudioParseError("partie multipart sans délimiteur suivant".to_string())
        })?;
        fields.push(parse_part(&body[part_start..part_start + next])?);
        // Repositionne sur le délimiteur suivant (saute le `\r\n` de tête).
        index = part_start + next + 2;
    }
    Ok(fields)
}

fn parse_part(part: &[u8]) -> Result<MultipartField> {
    let split = find_sub(part, b"\r\n\r\n")
        .ok_or_else(|| AudioParseError("partie multipart sans en-têtes".to_string()))?;
    let headers = std::str::from_utf8(&part[..split])
        .map_err(|e| AudioParseError(format!("en-têtes multipart non UTF-8: {e}")))?;
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
    let name = name.ok_or_else(|| AudioParseError("Content-Disposition sans name".to_string()))?;
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
pub fn wav_to_mono_16k(bytes: &[u8]) -> Result<Vec<f32>> {
    let invalid_wav = |e| AudioParseError(format!("WAV invalide: {e}"));
    let mut cursor = Cursor::new(bytes);
    let file_len = hound::read_wave_header(&mut cursor).map_err(invalid_wav)?;
    if file_len > bytes.len() as u64 {
        return Err(AudioParseError(
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
        .ok_or_else(|| AudioParseError("WAV sans taille data".into()))?;
    let data_len = u32::from_le_bytes(size_bytes);
    // Hound déduit la longueur de l'itérateur de l'en-tête, sans vérifier le corps.
    // Vérifier les octets stockés (y compris le padding des samples) avant collect.
    if data_start as u64 + u64::from(data_len) > file_len {
        return Err(AudioParseError(
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
                .collect::<std::result::Result<Vec<_>, _>>(),
            bits @ (24 | 32) => {
                // hound rend la valeur 24 bits telle quelle dans un i32, sans la décaler.
                let full_scale = ((1_u64 << (bits - 1)) - 1) as f32;
                reader
                    .samples::<i32>()
                    .map(|sample| sample.map(|value| value as f32 / full_scale))
                    .collect()
            }
            other => {
                return Err(AudioParseError(format!(
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

pub fn validate_wav_limits(sample_rate: u32, channels: u16, samples: u64) -> Result<()> {
    if !WAV_SAMPLE_RATE_RANGE.contains(&sample_rate) {
        return Err(AudioParseError(format!(
            "taux WAV hors plage 8000–192000 Hz: {sample_rate}"
        )));
    }
    if channels == 0 {
        return Err(AudioParseError("WAV sans canal".into()));
    }
    if samples > MAX_WAV_SAMPLES {
        return Err(AudioParseError(format!(
            "WAV dépasse {MAX_WAV_SAMPLES} échantillons"
        )));
    }
    Ok(())
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
pub fn downsample(samples: &[f32], input_rate: u32, target_rate: u32) -> Result<Vec<f32>> {
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
