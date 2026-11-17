//! WAV incrémental : PCM 16/24/32 et f32, RIFF/RF64, longueur inconnue.
use crate::wire_audio::{validate_wav_limits, AudioParseError, Result};
use std::io::{self, BufReader, Read};

const HEADER_LIMIT: u64 = 64 * 1024;
/// Nombre maximal de trames mono décodées par lecture.
pub const CHUNK_SAMPLES: usize = 480;

fn invalid(message: impl ToString) -> AudioParseError {
    AudioParseError(message.to_string())
}
fn read<const N: usize>(r: &mut impl Read) -> Result<[u8; N]> {
    let mut b = [0; N];
    r.read_exact(&mut b).map_err(invalid)?;
    Ok(b)
}
fn u16le(b: &[u8]) -> u16 {
    u16::from_le_bytes([b[0], b[1]])
}
fn u32le(b: &[u8]) -> u32 {
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}
fn u64le(b: &[u8]) -> u64 {
    u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
}

/// Lecteur WAV à mémoire constante ; ne lit jamais le signal entier.
pub struct WavStream<R> {
    reader: BufReader<R>,
    rate: u32,
    channels: u16,
    bits: u16,
    float: bool,
    remaining: Option<u64>,
    trailing: Option<u64>,
}

impl<R: Read> WavStream<R> {
    /// Lit et valide l'en-tête avant d'exposer les échantillons.
    ///
    /// # Errors
    /// Refuse les formats, tailles, fréquences et alignements incohérents.
    pub fn new(reader: R) -> Result<Self> {
        let mut reader = BufReader::with_capacity(8192, reader);
        let header = read::<12>(&mut reader)?;
        let rf64 = &header[..4] == b"RF64";
        if (!rf64 && &header[..4] != b"RIFF") || &header[8..] != b"WAVE" {
            return Err(invalid("en-tête RIFF/WAVE requis"));
        }
        let riff = u32le(&header[4..]);
        let mut riff_end = (riff != u32::MAX).then_some(u64::from(riff) + 8);
        let mut data64 = None;
        let mut spec = None;
        let mut position = 12_u64;
        loop {
            if position + 8 > HEADER_LIMIT {
                return Err(invalid("en-tête WAV trop grand"));
            }
            let chunk = read::<8>(&mut reader)?;
            let size = u32le(&chunk[4..]);
            position += 8;
            if &chunk[..4] == b"data" {
                let (rate, channels, bits, float) = spec.ok_or_else(|| invalid("WAV sans fmt"))?;
                let length = if size == u32::MAX {
                    data64
                } else {
                    Some(u64::from(size))
                };
                let align = u64::from(channels) * u64::from(bits / 8);
                if let Some(n) = length {
                    if n % align != 0
                        || riff_end
                            .is_some_and(|end| position > end || n > end.saturating_sub(position))
                    {
                        return Err(invalid("taille data WAV incohérente"));
                    }
                }
                if length.is_none() && riff_end.is_some() {
                    return Err(invalid(
                        "longueur data inconnue dans un RIFF de taille connue",
                    ));
                }
                let trailing = riff_end.zip(length).map(|(end, n)| end - position - n);
                if rf64 && data64.is_none() {
                    return Err(invalid("RF64 sans ds64"));
                }
                return Ok(Self {
                    reader,
                    rate,
                    channels,
                    bits,
                    float,
                    remaining: length,
                    trailing,
                });
            }
            let padded = u64::from(size) + u64::from(size % 2);
            if position + padded > HEADER_LIMIT
                || riff_end.is_some_and(|end| position + padded > end)
            {
                return Err(invalid("en-tête WAV trop grand ou tronqué"));
            }
            match &chunk[..4] {
                b"fmt " => {
                    if spec.is_some() || !(16..=40).contains(&size) {
                        return Err(invalid("fmt WAV invalide"));
                    }
                    let mut fmt = [0_u8; 40];
                    reader
                        .read_exact(&mut fmt[..size as usize])
                        .map_err(invalid)?;
                    let mut format = u16le(&fmt);
                    let channels = u16le(&fmt[2..]);
                    let rate = u32le(&fmt[4..]);
                    let bits = u16le(&fmt[14..]);
                    validate_wav_limits(rate, channels, 0)?;
                    if format == 0xfffe {
                        const GUID_SUFFIX: [u8; 14] =
                            [0, 0, 0, 0, 16, 0, 128, 0, 0, 170, 0, 56, 155, 113];
                        if size != 40
                            || u16le(&fmt[16..]) != 22
                            || u16le(&fmt[18..]) != bits
                            || fmt[26..40] != GUID_SUFFIX
                        {
                            return Err(invalid("WAV extensible non supporté"));
                        }
                        format = u16le(&fmt[24..]);
                    }
                    if !matches!((format, bits), (1, 16 | 24 | 32) | (3, 32)) {
                        return Err(invalid("format WAV attendu : PCM 16/24/32 ou float32"));
                    }
                    let align = u32::from(channels) * u32::from(bits / 8);
                    if u32::from(u16le(&fmt[12..])) != align
                        || u64::from(u32le(&fmt[8..])) != u64::from(rate) * u64::from(align)
                    {
                        return Err(invalid("alignement ou débit WAV incohérent"));
                    }
                    spec = Some((rate, channels, bits, format == 3));
                }
                b"ds64" if rf64 => {
                    if size < 28 || data64.is_some() {
                        return Err(invalid("ds64 invalide"));
                    }
                    let ds = read::<28>(&mut reader)?;
                    riff_end = Some(
                        u64le(&ds)
                            .checked_add(8)
                            .ok_or_else(|| invalid("taille RF64 débordée"))?,
                    );
                    data64 = Some(u64le(&ds[8..]));
                    discard(&mut reader, u64::from(size - 28))?;
                }
                _ => discard(&mut reader, u64::from(size))?,
            }
            if size % 2 != 0 {
                read::<1>(&mut reader)?;
            }
            position += padded;
        }
    }

    /// Fréquence d'entrée validée.
    pub fn sample_rate(&self) -> u32 {
        self.rate
    }
    /// Transport sous-jacent pour les écritures duplex.
    pub fn get_mut(&mut self) -> &mut R {
        self.reader.get_mut()
    }

    /// Consomme les chunks annexes jusqu'à la fin du corps, sans accumulation.
    ///
    /// # Errors
    /// Propage les erreurs de transport.
    pub fn finish_input(&mut self) -> io::Result<()> {
        let trailing = io::copy(&mut self.reader, &mut io::sink())?;
        if self.trailing.is_some_and(|expected| trailing < expected) {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "WAV tronqué après data",
            ));
        }
        self.trailing = None;
        Ok(())
    }

    /// Décode au plus 480 trames, moyennées en mono.
    ///
    /// # Errors
    /// Refuse une fin prématurée, un échantillon non fini ou une erreur d'E/S.
    pub fn read_chunk(&mut self) -> Result<Vec<f32>> {
        let mut out = Vec::with_capacity(CHUNK_SAMPLES);
        let bytes = usize::from(self.bits / 8);
        for _ in 0..CHUNK_SAMPLES {
            if self.remaining == Some(0) {
                break;
            }
            let mut sum = 0.0;
            for channel in 0..self.channels {
                let mut raw = [0_u8; 4];
                let count = self.reader.read(&mut raw[..1]).map_err(invalid)?;
                if count == 0 {
                    if channel == 0 && self.remaining.is_none() {
                        self.remaining = Some(0);
                        return Ok(out);
                    }
                    return Err(invalid("WAV tronqué dans les échantillons"));
                }
                self.reader
                    .read_exact(&mut raw[1..bytes])
                    .map_err(invalid)?;
                if let Some(left) = &mut self.remaining {
                    *left -= bytes as u64;
                }
                let value = if self.float {
                    f32::from_le_bytes(raw)
                } else {
                    let value = match self.bits {
                        16 => i32::from(i16::from_le_bytes([raw[0], raw[1]])),
                        24 => i32::from_le_bytes([0, raw[0], raw[1], raw[2]]) >> 8,
                        _ => i32::from_le_bytes(raw),
                    };
                    value as f32 / ((1_u64 << (self.bits - 1)) - 1) as f32
                };
                if !value.is_finite() {
                    return Err(invalid("échantillon WAV non fini"));
                }
                sum += value;
            }
            let sample = sum / f32::from(self.channels);
            if !sample.is_finite() {
                return Err(invalid("moyenne WAV non finie"));
            }
            out.push(sample);
        }
        Ok(out)
    }
}

fn discard(reader: &mut impl Read, mut n: u64) -> Result<()> {
    let mut scratch = [0_u8; 4096];
    while n > 0 {
        let take = n.min(scratch.len() as u64) as usize;
        reader.read_exact(&mut scratch[..take]).map_err(invalid)?;
        n -= take as u64;
    }
    Ok(())
}

/// En-tête mono float32 48 kHz de 80 octets. `None` utilise les longueurs
/// RIFF/data 0xffffffff des flux non seekable ; au-delà de 4 Gio, RF64/ds64.
///
/// # Errors
/// Refuse un compteur dont la taille en octets déborde u64.
pub fn output_header(samples: Option<u64>) -> io::Result<Vec<u8>> {
    header(samples, 4, 3)
}

/// En-tête WAV mono PCM16 48 kHz de longueur inconnue pour les réponses HTTP.
///
/// # Errors
/// Propage une erreur de construction de l'en-tête.
pub fn pcm16_stream_header() -> io::Result<Vec<u8>> {
    header(None, 2, 1)
}

fn header(samples: Option<u64>, sample_bytes: u16, format: u16) -> io::Result<Vec<u8>> {
    let data = samples
        .map(|n| {
            n.checked_mul(u64::from(sample_bytes))
                .filter(|n| *n <= u64::MAX - 72)
                .ok_or_else(|| io::Error::other("taille WAV débordée"))
        })
        .transpose()?;
    let rf64 = data.is_some_and(|n| n > u64::from(u32::MAX) - 72);
    let mut out = Vec::with_capacity(80);
    out.extend(if rf64 { b"RF64" } else { b"RIFF" });
    out.extend(
        data.filter(|_| !rf64)
            .map(|n| (n + 72) as u32)
            .unwrap_or(u32::MAX)
            .to_le_bytes(),
    );
    out.extend(b"WAVE");
    out.extend(if rf64 { b"ds64" } else { b"JUNK" });
    out.extend(28_u32.to_le_bytes());
    out.extend(data.map(|n| n + 72).unwrap_or(0).to_le_bytes());
    out.extend(data.unwrap_or(0).to_le_bytes());
    out.extend(samples.unwrap_or(0).to_le_bytes());
    out.extend(0_u32.to_le_bytes());
    out.extend(b"fmt ");
    out.extend(16_u32.to_le_bytes());
    out.extend(format.to_le_bytes());
    out.extend(1_u16.to_le_bytes());
    out.extend(48_000_u32.to_le_bytes());
    out.extend((48_000 * u32::from(sample_bytes)).to_le_bytes());
    out.extend(sample_bytes.to_le_bytes());
    out.extend((8 * sample_bytes).to_le_bytes());
    out.extend(b"data");
    out.extend(
        data.filter(|_| !rf64)
            .map(|n| n as u32)
            .unwrap_or(u32::MAX)
            .to_le_bytes(),
    );
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn truncated_riff_tail_is_rejected() {
        let mut wav = output_header(Some(1)).unwrap();
        wav[4..8].copy_from_slice(&100_u32.to_le_bytes());
        wav.extend(0.25_f32.to_le_bytes());
        let mut reader = WavStream::new(io::Cursor::new(wav)).unwrap();
        assert_eq!(reader.read_chunk().unwrap(), [0.25]);
        assert!(reader.finish_input().is_err());
    }

    #[test]
    fn headers_round_trip_and_rf64_has_no_four_gib_limit() {
        for size in [None, Some(1), Some(u64::from(u32::MAX))] {
            let mut wav = output_header(size).unwrap();
            assert_eq!(wav.len(), 80);
            wav.extend(0.25_f32.to_le_bytes());
            let mut reader = WavStream::new(io::Cursor::new(wav)).unwrap();
            if size == Some(u64::from(u32::MAX)) {
                assert!(reader.read_chunk().is_err());
            } else {
                assert_eq!(reader.read_chunk().unwrap(), [0.25]);
                assert!(reader.read_chunk().unwrap().is_empty());
            }
        }
        assert!(output_header(Some(u64::MAX)).is_err());
    }
}
