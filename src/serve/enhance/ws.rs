//! RFC 6455 synchrone : messages bornés, masque client et contrôle de flux socket.
use super::pcm::{invalid, loopback_origin, Config, Pcm, MAX_FRAME};
use super::*;
use base64::{engine::general_purpose::STANDARD, Engine};
use std::io::Cursor;
use std::time::{Duration, Instant};

impl EnhanceService {
    pub(in crate::serve) fn handle_ws<S: ServeStream>(
        &self,
        socket: &mut S,
        request: &mut HttpRequest,
    ) -> ServeResult<()> {
        // Un navigateur joint toujours Origin : une page tierce ne doit pas piloter l'endpoint local.
        if request
            .header("origin")
            .is_some_and(|o| !loopback_origin(o))
        {
            return send_error(socket, 403, "origine WebSocket non loopback refusée");
        }
        let Some(_slot) = Reservation::acquire(&self.active, 1, enhance_max_streams()) else {
            return send_error(socket, 503, "trop de flux enhance simultanés");
        };
        socket.configure(enhance_idle_timeout())?;
        let prepared = (|| {
            let accept = handshake(request)?;
            let config = Config::parse(request.query())?;
            Ok((accept, config, self.model()?))
        })();
        let (accept, config, model) = match prepared {
            Ok(value) => value,
            Err(error) => return reply_on_error(socket, Err(error)),
        };
        let mut pcm = match Pcm::new(&model, config) {
            Ok(pcm) => pcm,
            Err(error) => return reply_on_error(socket, Err(error)),
        };
        write!(socket, "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: {accept}\r\n")
            .map_err(|e| ServeError::io("handshake WS", e))?;
        // Le navigateur exige qu'un sous-protocole proposé soit renvoyé tel quel.
        let offered = request.header("sec-websocket-protocol").unwrap_or("");
        let chosen = offered
            .split(',')
            .map(str::trim)
            .find(|p| *p == "saragossa")
            .or_else(|| {
                offered
                    .split(',')
                    .map(str::trim)
                    .find(|p| p.starts_with("bearer."))
            });
        if let Some(protocol) = chosen {
            write!(socket, "Sec-WebSocket-Protocol: {protocol}\r\n")
                .map_err(|e| ServeError::io("protocole WS", e))?;
        }
        socket
            .write_all(b"\r\n")
            .and_then(|()| socket.flush())
            .map_err(|e| ServeError::io("handshake WS", e))?;
        let mut wire = Wire {
            prefix: Cursor::new(request.take_prefix()),
            socket,
        };
        session(&mut wire, &mut pcm, enhance_idle_timeout())
    }
}
/// Pings ou pongs tolérés entre deux messages de données.
const MAX_CONTROL_RUN: usize = 32;
/// Code de fermeture RFC 6455 pour une erreur PCM : 1007 donnée invalide, 1011 interne.
fn close_code(error: &ServeError) -> u16 {
    match error {
        ServeError::Args(_) | ServeError::Http(_) | ServeError::InvalidParameter { .. } => 1007,
        _ => 1011,
    }
}
/// Lecteur dont l'échéance est fixée pour toute une trame, pas par `read()`.
struct Until<'a, R>(&'a mut R, Instant);
impl<R: Read> Read for Until<'_, R> {
    fn read(&mut self, b: &mut [u8]) -> io::Result<usize> {
        if Instant::now() >= self.1 {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "trame WS trop lente",
            ));
        }
        self.0.read(b)
    }
}
fn handshake(request: &HttpRequest) -> ServeResult<String> {
    let token = |name: &str, expected: &str| {
        request.header(name).is_some_and(|v| {
            v.split(',')
                .any(|s| s.trim().eq_ignore_ascii_case(expected))
        })
    };
    if request.method != "GET"
        || !token("upgrade", "websocket")
        || !token("connection", "upgrade")
        || request.header("sec-websocket-version") != Some("13")
        || request.header_count("sec-websocket-key") != 1
        || request.header_count("sec-websocket-version") != 1
        || request.header("content-length").is_some_and(|v| v != "0")
        || request.header("transfer-encoding").is_some()
    {
        return Err(invalid("handshake WebSocket invalide"));
    }
    let key = request
        .header("sec-websocket-key")
        .ok_or_else(|| invalid("clé WS absente"))?;
    if STANDARD
        .decode(key)
        .map_err(|_| invalid("clé WS base64 invalide"))?
        .len()
        != 16
    {
        return Err(invalid("clé WS de 16 octets requise"));
    }
    let mut sha = sha1_smol::Sha1::new();
    sha.update(key.as_bytes());
    sha.update(b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11");
    Ok(STANDARD.encode(sha.digest().bytes()))
}
struct Wire<'a, S> {
    prefix: Cursor<Vec<u8>>,
    socket: &'a mut S,
}
impl<S: Read> Read for Wire<'_, S> {
    fn read(&mut self, b: &mut [u8]) -> io::Result<usize> {
        let n = self.prefix.read(b)?;
        if n != 0 {
            Ok(n)
        } else {
            self.socket.read(b)
        }
    }
}
impl<S: Write> Write for Wire<'_, S> {
    fn write(&mut self, b: &[u8]) -> io::Result<usize> {
        self.socket.write(b)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.socket.flush()
    }
}
fn session(socket: &mut (impl Read + Write), pcm: &mut Pcm<'_>, idle: Duration) -> ServeResult<()> {
    let mut fragmented = false;
    let mut message = Vec::new();
    let mut controls = 0;
    loop {
        let frame = read_frame(&mut Until(socket, Instant::now() + idle));
        let (fin, opcode, bytes) = match frame {
            Ok(frame) => frame,
            Err((code, error)) => {
                let _ = write_frame(socket, 8, &code.to_be_bytes());
                return Err(ServeError::io("trame WS", error));
            }
        };
        match opcode {
            8 => {
                if !valid_close(&bytes) {
                    write_frame(socket, 8, &1002_u16.to_be_bytes())?;
                    return Err(invalid("close WS invalide"));
                }
                return write_frame(socket, 8, &bytes);
            }
            9 | 10 => {
                controls += 1;
                if controls > MAX_CONTROL_RUN {
                    write_frame(socket, 8, &1008_u16.to_be_bytes())?;
                    return Err(invalid("trop de pings WS sans audio"));
                }
                if opcode == 9 {
                    write_frame(socket, 10, &bytes)?;
                }
            }
            2 if !fragmented => {
                controls = 0;
                message.extend(bytes);
                fragmented = !fin;
            }
            0 if fragmented => {
                controls = 0;
                message.extend(bytes);
                fragmented = !fin;
            }
            _ => {
                write_frame(socket, 8, &1002_u16.to_be_bytes())?;
                return Err(invalid("opcode ou fragmentation WS invalide"));
            }
        }
        if message.len() > MAX_FRAME {
            write_frame(socket, 8, &1009_u16.to_be_bytes())?;
            return Err(invalid("message WS trop grand"));
        }
        if opcode <= 2 && fin {
            if message.is_empty() {
                let output = pcm.finish().inspect_err(|e| fail(socket, e))?;
                if !output.is_empty() {
                    write_frame(socket, 2, &output)?;
                }
                write_frame(socket, 2, &[])?;
                return write_frame(socket, 8, &1000_u16.to_be_bytes());
            }
            let output = pcm.push(&message).inspect_err(|e| fail(socket, e))?;
            if !output.is_empty() {
                write_frame(socket, 2, &output)?;
            }
            message.clear();
        }
    }
}
fn fail(socket: &mut impl Write, error: &ServeError) {
    let _ = write_frame(socket, 8, &close_code(error).to_be_bytes());
}
fn valid_close(bytes: &[u8]) -> bool {
    if bytes.is_empty() {
        return true;
    }
    if bytes.len() < 2 {
        return false;
    }
    let code = u16::from_be_bytes([bytes[0], bytes[1]]);
    (matches!(code, 1000..=1003 | 1007..=1014 | 3000..=4999))
        && std::str::from_utf8(&bytes[2..]).is_ok()
}
fn read_frame(reader: &mut impl Read) -> Result<(bool, u8, Vec<u8>), (u16, io::Error)> {
    fn read(reader: &mut impl Read) -> io::Result<(bool, u8, Vec<u8>)> {
        let bad = || io::Error::new(io::ErrorKind::InvalidData, "cadrage WS invalide");
        let mut head = [0; 2];
        reader.read_exact(&mut head)?;
        let fin = head[0] & 0x80 != 0;
        let opcode = head[0] & 15;
        if head[0] & 0x70 != 0 || head[1] & 0x80 == 0 {
            return Err(bad());
        }
        let short = head[1] & 127;
        let len = match short {
            126 => {
                let mut b = [0; 2];
                reader.read_exact(&mut b)?;
                u64::from(u16::from_be_bytes(b))
            }
            127 => {
                let mut b = [0; 8];
                reader.read_exact(&mut b)?;
                u64::from_be_bytes(b)
            }
            n => u64::from(n),
        };
        if (short == 126 && len < 126)
            || (short == 127 && (len < 65536 || len >> 63 != 0))
            || (opcode >= 8 && (!fin || len > 125))
        {
            return Err(bad());
        }
        if len > MAX_FRAME as u64 {
            return Err(io::Error::new(
                io::ErrorKind::FileTooLarge,
                "trame WS trop grande",
            ));
        }
        let mut mask = [0; 4];
        reader.read_exact(&mut mask)?;
        let mut bytes = vec![0; len as usize];
        reader.read_exact(&mut bytes)?;
        for (i, b) in bytes.iter_mut().enumerate() {
            *b ^= mask[i % 4];
        }
        Ok((fin, opcode, bytes))
    }
    read(reader).map_err(|e| {
        (
            if e.kind() == io::ErrorKind::FileTooLarge {
                1009
            } else {
                1002
            },
            e,
        )
    })
}
pub(super) fn write_frame(writer: &mut impl Write, opcode: u8, bytes: &[u8]) -> ServeResult<()> {
    (|| -> io::Result<()> {
        writer.write_all(&[0x80 | opcode])?;
        match bytes.len() {
            n @ 0..=125 => writer.write_all(&[n as u8])?,
            n @ 126..=65535 => {
                writer.write_all(&[126])?;
                writer.write_all(&(n as u16).to_be_bytes())?;
            }
            n => {
                writer.write_all(&[127])?;
                writer.write_all(&(n as u64).to_be_bytes())?;
            }
        }
        writer.write_all(bytes)?;
        writer.flush()
    })()
    .map_err(|e| ServeError::io("écriture WS", e))
}

#[cfg(test)]
pub(super) mod tests;
