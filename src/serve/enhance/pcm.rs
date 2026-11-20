//! PCM mono commun aux transports duplex, sans accumulation de l'audio.
use super::*;

pub(super) const MAX_FRAME: usize = 16 * 1024;

#[derive(Clone, Copy, Debug)]
pub(super) struct Config {
    pub rate: u32,
    pub float: bool,
}
impl Config {
    pub fn parse(query: &str) -> ServeResult<Self> {
        let mut rate = None;
        let mut float = None;
        for field in query.split('&') {
            match field.split_once('=') {
                Some(("rate", value)) if rate.is_none() => {
                    rate = Some(value.parse::<u32>().map_err(|_| invalid("rate invalide"))?);
                }
                Some(("format", value)) if float.is_none() => {
                    float = Some(match value {
                        "s16" => false,
                        "f32" => true,
                        _ => return Err(invalid("format s16 ou f32 requis")),
                    });
                }
                _ => return Err(invalid("paramètre inconnu ou répété")),
            }
        }
        let rate = rate
            .filter(|r| (8000..=192000).contains(r))
            .ok_or_else(|| invalid("rate requis entre 8000 et 192000"))?;
        Ok(Self {
            rate,
            float: float.ok_or_else(|| invalid("format requis"))?,
        })
    }
}
pub(super) fn invalid(message: &str) -> ServeError {
    ServeError::Http(message.into())
}

pub(super) struct Pcm<'a> {
    config: Config,
    resampler: StreamResampler,
    denoise: DenoiseStream<'a>,
    pending: Vec<u8>,
}
impl<'a> Pcm<'a> {
    pub fn new(model: &'a DeepFilterNet3, config: Config) -> ServeResult<Self> {
        Ok(Self {
            config,
            resampler: StreamResampler::new(config.rate, SAMPLE_RATE)?,
            denoise: DenoiseStream::new(model),
            pending: Vec::with_capacity(4),
        })
    }
    pub fn push(&mut self, bytes: &[u8]) -> ServeResult<Vec<u8>> {
        if bytes.len() > MAX_FRAME {
            return Err(invalid("trame PCM trop grande"));
        }
        self.pending.extend_from_slice(bytes);
        let width = if self.config.float { 4 } else { 2 };
        let count = self.pending.len() / width * width;
        let mut samples = Vec::with_capacity(count / width);
        for b in self.pending[..count].chunks_exact(width) {
            let sample = if self.config.float {
                f32::from_le_bytes([b[0], b[1], b[2], b[3]])
            } else {
                f32::from(i16::from_le_bytes([b[0], b[1]])) / f32::from(i16::MAX)
            };
            if !sample.is_finite() {
                return Err(invalid("PCM non fini"));
            }
            samples.push(sample);
        }
        self.pending.drain(..count);
        let samples = self.resampler.push(&samples)?;
        let output = self.denoise.push(&samples)?;
        Ok(self.encode(&output))
    }
    pub fn finish(&mut self) -> ServeResult<Vec<u8>> {
        if !self.pending.is_empty() {
            return Err(invalid("échantillon PCM tronqué"));
        }
        let mut output = self.denoise.push(&self.resampler.finish())?;
        output.extend(self.denoise.finish()?);
        Ok(self.encode(&output))
    }
    fn encode(&self, samples: &[f32]) -> Vec<u8> {
        if self.config.float {
            samples.iter().flat_map(|s| s.to_le_bytes()).collect()
        } else {
            samples
                .iter()
                .flat_map(|s| super::super::audio::pcm_i16(*s).to_le_bytes())
                .collect()
        }
    }
}
/// Origin d'un navigateur sur la machine locale, quel que soit le port.
pub(super) fn loopback_origin(origin: &str) -> bool {
    let Some(rest) = origin
        .strip_prefix("http://")
        .or_else(|| origin.strip_prefix("https://"))
    else {
        return false;
    };
    let (host, port) = if let Some(inner) = rest.strip_prefix('[') {
        let Some((host, tail)) = inner.split_once(']') else {
            return false;
        };
        (host, tail.strip_prefix(':').unwrap_or(tail))
    } else {
        rest.split_once(':').unwrap_or((rest, ""))
    };
    port.bytes().all(|b| b.is_ascii_digit())
        && (host == "localhost"
            || host == "::1"
            || host
                .parse::<std::net::Ipv4Addr>()
                .is_ok_and(|ip| ip.is_loopback()))
}
