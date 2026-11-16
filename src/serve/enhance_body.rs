//! Cadrage duplex borné des corps HTTP et de la partie WAV multipart.
use std::collections::VecDeque;
use std::io::{self, BufReader, Cursor, Read, Write};

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

/// Corps HTTP déchunké. Une lecture ne dépasse jamais le chunk courant.
pub(super) struct Body<'a, S> {
    pub(super) socket: &'a mut S,
    prefix: Cursor<Vec<u8>>,
    remaining: u64,
    chunked: bool,
    separator: bool,
    done: bool,
}

impl<'a, S: Read + Write> Body<'a, S> {
    pub(super) fn new(socket: &'a mut S, prefix: Vec<u8>, length: usize, chunked: bool) -> Self {
        Self {
            socket,
            prefix: Cursor::new(prefix),
            remaining: if chunked { 0 } else { length as u64 },
            chunked,
            separator: false,
            done: false,
        }
    }
    fn raw_read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        if self.prefix.position() < self.prefix.get_ref().len() as u64 {
            self.prefix.read(bytes)
        } else {
            self.socket.read(bytes)
        }
    }
    fn line(&mut self) -> io::Result<Vec<u8>> {
        let mut line = Vec::new();
        loop {
            let mut byte = [0];
            if self.raw_read(&mut byte)? == 0 {
                return Err(invalid("corps HTTP tronqué"));
            }
            line.push(byte[0]);
            if line.ends_with(b"\r\n") {
                line.truncate(line.len() - 2);
                return Ok(line);
            }
            if line.len() >= 1024 {
                return Err(invalid("ligne chunked trop longue"));
            }
        }
    }
}

impl<S: Read + Write> Read for Body<'_, S> {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        if bytes.is_empty() || self.done {
            return Ok(0);
        }
        if self.chunked && self.remaining == 0 {
            if self.separator && !self.line()?.is_empty() {
                return Err(invalid("séparateur chunked invalide"));
            }
            let line = self.line()?;
            let size = line.split(|b| *b == b';').next().unwrap_or_default();
            if size.is_empty() || !size.iter().all(u8::is_ascii_hexdigit) {
                return Err(invalid("taille chunked invalide"));
            }
            self.remaining = u64::from_str_radix(
                std::str::from_utf8(size).map_err(|_| invalid("taille chunked invalide"))?,
                16,
            )
            .map_err(|_| invalid("taille chunked débordée"))?;
            self.separator = true;
            if self.remaining == 0 {
                let mut total = 0;
                loop {
                    let trailer = self.line()?;
                    total += trailer.len() + 2;
                    if total > 65536 {
                        return Err(invalid("trailers trop grands"));
                    }
                    if trailer.is_empty() {
                        break;
                    }
                    if !trailer.contains(&b':') {
                        return Err(invalid("trailer invalide"));
                    }
                }
                self.done = true;
                return Ok(0);
            }
        }
        if self.remaining == 0 {
            self.done = true;
            return Ok(0);
        }
        let cap = self.remaining.min(bytes.len() as u64) as usize;
        let n = self.raw_read(&mut bytes[..cap])?;
        if n == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "corps HTTP tronqué",
            ));
        }
        self.remaining -= n as u64;
        Ok(n)
    }
}
impl<S: Write> Write for Body<'_, S> {
    fn write(&mut self, b: &[u8]) -> io::Result<usize> {
        self.socket.write(b)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.socket.flush()
    }
}

/// Le délimiteur de la partie reste dans une fenêtre de 74 octets au plus.
pub(super) struct Multipart<R> {
    inner: BufReader<R>,
    delimiter: Vec<u8>,
    pending: VecDeque<u8>,
    done: bool,
    metadata_budget: usize,
}
impl<R: Read + Write> Multipart<R> {
    pub(super) fn new(inner: R, boundary: &str) -> io::Result<Self> {
        if boundary.is_empty()
            || boundary.len() > 70
            || !boundary.bytes().all(|b| b.is_ascii_graphic() && b != b'"')
        {
            return Err(invalid("boundary multipart invalide"));
        }
        let mut this = Self {
            inner: BufReader::with_capacity(8192, inner),
            delimiter: format!("\r\n--{boundary}").into_bytes(),
            pending: VecDeque::new(),
            done: false,
            metadata_budget: 65536,
        };
        let opening = format!("--{boundary}\r\n");
        let mut bytes = vec![0; opening.len()];
        this.inner.read_exact(&mut bytes)?;
        if bytes != opening.as_bytes() {
            return Err(invalid("délimiteur multipart absent"));
        }
        loop {
            if this.file_header()? {
                return Ok(this);
            }
            this.skip_field()?;
            if this.suffix()? == *b"--" {
                return Err(invalid("champ multipart file requis"));
            }
            this.done = false;
        }
    }
    fn charge_metadata(&mut self) -> io::Result<()> {
        self.metadata_budget = self
            .metadata_budget
            .checked_sub(1)
            .ok_or_else(|| invalid("métadonnées multipart trop grandes"))?;
        Ok(())
    }
    fn file_header(&mut self) -> io::Result<bool> {
        let mut headers = Vec::new();
        while !headers.ends_with(b"\r\n\r\n") {
            self.charge_metadata()?;
            let mut byte = [0];
            self.inner.read_exact(&mut byte)?;
            headers.push(byte[0]);
        }
        let text =
            std::str::from_utf8(&headers).map_err(|_| invalid("en-tête multipart invalide"))?;
        Ok(text
            .lines()
            .filter_map(|line| line.split_once(':'))
            .filter(|(key, _)| key.eq_ignore_ascii_case("content-disposition"))
            .any(|(_, value)| {
                value
                    .split(';')
                    .filter_map(|p| p.trim().split_once('='))
                    .any(|(key, value)| key == "name" && value.trim_matches('"') == "file")
            }))
    }
    fn skip_field(&mut self) -> io::Result<()> {
        let mut byte = [0];
        while self.read(&mut byte)? != 0 {
            self.charge_metadata()?;
        }
        Ok(())
    }
    fn suffix(&mut self) -> io::Result<[u8; 2]> {
        let mut suffix = [0; 2];
        self.inner.read_exact(&mut suffix)?;
        if suffix != *b"--" && suffix != *b"\r\n" {
            return Err(invalid("fin multipart invalide"));
        }
        Ok(suffix)
    }
    pub(super) fn finish(&mut self) -> io::Result<()> {
        io::copy(self, &mut io::sink())?;
        while self.suffix()? != *b"--" {
            if self.file_header()? {
                return Err(invalid("plusieurs fichiers multipart"));
            }
            self.done = false;
            self.skip_field()?;
        }
        let mut tail = Vec::new();
        self.inner.by_ref().take(3).read_to_end(&mut tail)?;
        if !tail.is_empty() && tail != b"\r\n" {
            return Err(invalid("suffixe multipart invalide"));
        }
        Ok(())
    }
}
impl<R: Read + Write> Read for Multipart<R> {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if out.is_empty() || self.done {
            return Ok(0);
        }
        let mut n = 0;
        while n < out.len() {
            while self.pending.len() < self.delimiter.len() {
                let mut byte = [0];
                self.inner.read_exact(&mut byte)?;
                self.pending.push_back(byte[0]);
            }
            if self
                .pending
                .iter()
                .copied()
                .eq(self.delimiter.iter().copied())
            {
                self.pending.clear();
                self.done = true;
                break;
            }
            if let Some(byte) = self.pending.pop_front() {
                out[n] = byte;
                n += 1;
            }
            // Ne pas attendre 8 Kio supplémentaires : rend ce qui est déjà
            // reçu au lecteur WAV, qui choisit la taille du prochain hop.
            if self.inner.buffer().is_empty() {
                break;
            }
        }
        Ok(n)
    }
}
impl<R: Read + Write> Write for Multipart<R> {
    fn write(&mut self, b: &[u8]) -> io::Result<usize> {
        self.inner.get_mut().write(b)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.inner.get_mut().flush()
    }
}
