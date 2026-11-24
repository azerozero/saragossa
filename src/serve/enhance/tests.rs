use super::*;
use crate::serve::http::read_head;
use saragossa::denoise::wav::output_header;
use std::io::{BufRead, BufReader, Cursor};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::Ordering;
use std::thread;
use std::time::{Duration, Instant};

fn pair() -> (TcpStream, TcpStream) {
    let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
    let (server, _) = listener.accept().unwrap();
    client.configure(Duration::from_secs(5)).unwrap();
    server.configure(Duration::from_secs(5)).unwrap();
    (client, server)
}
fn headers(reader: &mut impl BufRead) -> String {
    let mut result = String::new();
    loop {
        let mut line = String::new();
        assert!(reader.read_line(&mut line).unwrap() > 0);
        result.push_str(&line);
        if line == "\r\n" {
            return result;
        }
    }
}
fn chunk(reader: &mut impl BufRead) -> Vec<u8> {
    let mut line = String::new();
    reader.read_line(&mut line).unwrap();
    let n = usize::from_str_radix(line.trim(), 16).unwrap();
    let mut bytes = vec![0; n];
    reader.read_exact(&mut bytes).unwrap();
    let mut crlf = [0; 2];
    reader.read_exact(&mut crlf).unwrap();
    assert_eq!(&crlf, b"\r\n");
    bytes
}
fn echo<S: Read + Write>(
    socket: &mut S,
    request: &mut HttpRequest,
    length: usize,
) -> ServeResult<()> {
    serve(socket, request, length, |wav, started| {
        pump(wav, started, |input, _| Ok(input.to_vec()))
    })
}

#[test]
fn http_duplex_emits_audio_before_upload_finishes() {
    for multipart in [false, true] {
        for chunked in [false, true] {
            let (mut client, mut server) = pair();
            let worker = thread::spawn(move || {
                let (mut request, length) =
                    read_head(&mut server, Instant::now() + Duration::from_secs(5))
                        .unwrap()
                        .unwrap();
                echo(&mut server, &mut request, length)
            });
            let samples = [0.25_f32; 480 * 4];
            let mut wav = output_header(Some(samples.len() as u64)).unwrap();
            wav.extend(samples.iter().flat_map(|s| s.to_le_bytes()));
            let mut body = Vec::new();
            if multipart {
                body.extend(b"--BND\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\ndfn3\r\n--BND\r\nContent-Disposition: form-data; name=\"file\"; filename=\"in.wav\"\r\n\r\n");
            }
            body.extend(wav);
            if multipart {
                body.extend(b"\r\n--BND\r\nContent-Disposition: form-data; name=\"unused\"\r\n\r\nvalue\r\n--BND--\r\n");
            }
            let ty = match (multipart, chunked) {
                (true, false) => "multipart/form-data; boundary=BND",
                (true, true) => "Multipart/Form-Data; Boundary=BND",
                (false, false) => "audio/wav",
                (false, true) => "Audio/Wav",
            };
            let framing = if chunked {
                "Transfer-Encoding: chunked".to_string()
            } else {
                format!("Content-Length: {}", body.len())
            };
            write!(
                client,
                "POST /v1/audio/enhance HTTP/1.1\r\nContent-Type: {ty}\r\n{framing}\r\n\r\n"
            )
            .unwrap();
            // Garde plus de la moitié du corps côté client jusqu'à la réponse.
            let split = 3000;
            if chunked {
                write!(client, "{split:x}\r\n").unwrap();
            }
            client.write_all(&body[..split]).unwrap();
            if chunked {
                client.write_all(b"\r\n").unwrap();
            }
            let mut response = BufReader::new(client.try_clone().unwrap());
            let head = headers(&mut response);
            assert!(head.starts_with("HTTP/1.1 200"));
            assert!(head.contains("Transfer-Encoding: chunked"));
            let mut output = chunk(&mut response);
            let first_pcm = chunk(&mut response);
            assert!(!first_pcm.is_empty());
            output.extend(first_pcm);
            if chunked {
                write!(client, "{:x}\r\n", body.len() - split).unwrap();
            }
            client.write_all(&body[split..]).unwrap();
            if chunked {
                client.write_all(b"\r\n0\r\n\r\n").unwrap();
            }
            loop {
                let bytes = chunk(&mut response);
                if bytes.is_empty() {
                    break;
                }
                output.extend(bytes);
            }
            worker.join().unwrap().unwrap();
            let quantized: Vec<_> = samples
                .iter()
                .map(|s| f32::from(crate::serve::audio::pcm_i16(*s)) / f32::from(i16::MAX))
                .collect();
            assert_eq!(
                saragossa::denoise::decode_wav(&output).unwrap().0,
                quantized
            );
        }
    }
}

#[test]
fn invalid_rate_is_400_before_model_loading_and_large_lengths_are_admitted() {
    for rate in [0_u32, 7999, 192001] {
        let service = EnhanceService::new(Some(PathBuf::from("absent.safetensors")));
        let (mut client, mut server) = pair();
        let mut wav = output_header(None).unwrap();
        wav[60..64].copy_from_slice(&rate.to_le_bytes());
        write!(client, "POST /v1/audio/enhance HTTP/1.1\r\nContent-Type: audio/wav\r\nContent-Length: 99999999999\r\n\r\n").unwrap();
        client.write_all(&wav).unwrap();
        let (mut request, n) = read_head(&mut server, Instant::now() + Duration::from_secs(1))
            .unwrap()
            .unwrap();
        let error = service.handle(&mut server, &mut request, n).unwrap_err();
        assert!(error.to_string().contains("taux WAV"));
        assert!(service.model.lock().unwrap().is_none());
        assert_eq!(service.active.load(Ordering::Acquire), 0);
        assert!(headers(&mut BufReader::new(client)).starts_with("HTTP/1.1 400"));
    }
}

#[test]
fn excess_stream_returns_503_and_slots_are_reusable() {
    let service = EnhanceService::new(None);
    let slot = Reservation::acquire(
        &service.active,
        enhance_max_streams(),
        enhance_max_streams(),
    )
    .unwrap();
    let (mut client, mut server) = pair();
    client
        .write_all(b"POST /v1/audio/enhance HTTP/1.1\r\nContent-Length: 900000000\r\n\r\n")
        .unwrap();
    let (mut request, n) = read_head(&mut server, Instant::now() + Duration::from_secs(1))
        .unwrap()
        .unwrap();
    service.handle(&mut server, &mut request, n).unwrap();
    assert!(headers(&mut BufReader::new(client)).starts_with("HTTP/1.1 503"));
    drop(slot);
    assert!(Reservation::acquire(&service.active, 1, enhance_max_streams()).is_some());
}

#[test]
fn read_inactivity_closes_without_a_second_response_or_terminal_chunk() {
    let (mut client, mut server) = pair();
    server.configure(Duration::from_millis(80)).unwrap();
    let worker = thread::spawn(move || {
        let (mut request, n) = read_head(&mut server, Instant::now() + Duration::from_secs(1))
            .unwrap()
            .unwrap();
        let start = Instant::now();
        assert!(echo(&mut server, &mut request, n).is_err());
        assert!(start.elapsed() < Duration::from_secs(3));
    });
    client.write_all(b"POST /v1/audio/enhance HTTP/1.1\r\nContent-Type: audio/wav\r\nTransfer-Encoding: chunked\r\n\r\n50\r\n").unwrap();
    client.write_all(&output_header(None).unwrap()).unwrap();
    client.write_all(b"\r\n").unwrap();
    let mut response = Vec::new();
    client.read_to_end(&mut response).unwrap();
    worker.join().unwrap();
    assert_eq!(response.windows(8).filter(|w| *w == b"HTTP/1.1").count(), 1);
    assert!(!response.ends_with(b"0\r\n\r\n"));
}

#[test]
fn write_inactivity_is_bounded_on_a_real_socket() {
    let (_client, mut server) = pair();
    server.configure(Duration::from_millis(80)).unwrap();
    let start = Instant::now();
    let bytes = vec![0; 1024 * 1024];
    let mut failed = false;
    for _ in 0..64 {
        if write_chunk(&mut server, &bytes).is_err() {
            failed = true;
            break;
        }
    }
    assert!(
        failed,
        "le client ne lit pas, le timeout doit interrompre l'écriture"
    );
    assert!(start.elapsed() < Duration::from_secs(5));
}

#[test]
fn backpressure_stops_reads_with_a_bounded_prefix() {
    struct SlowSink {
        input: io::Chain<Cursor<Vec<u8>>, io::Take<io::Repeat>>,
        read: usize,
        written: usize,
    }
    impl Read for SlowSink {
        fn read(&mut self, b: &mut [u8]) -> io::Result<usize> {
            let n = self.input.read(b)?;
            self.read += n;
            Ok(n)
        }
    }
    impl Write for SlowSink {
        fn write(&mut self, b: &[u8]) -> io::Result<usize> {
            if self.written > 200 {
                return Err(io::Error::new(io::ErrorKind::TimedOut, "lecteur lent"));
            }
            self.written += b.len();
            Ok(b.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let input =
        Cursor::new(output_header(None).unwrap()).chain(io::repeat(0).take(3600 * 48000 * 4));
    let mut wav = WavStream::new(SlowSink {
        input,
        read: 0,
        written: 0,
    })
    .unwrap();
    let mut started = false;
    assert!(pump(&mut wav, &mut started, |x, _| Ok(x.to_vec())).is_err());
    assert!(wav.get_mut().read <= 8192 + 80);
}

#[test]
fn chunk_framing_is_incremental_and_rejects_truncation() {
    struct Duplex {
        input: Cursor<Vec<u8>>,
        output: Vec<u8>,
    }
    impl Read for Duplex {
        fn read(&mut self, b: &mut [u8]) -> io::Result<usize> {
            self.input.read(b)
        }
    }
    impl Write for Duplex {
        fn write(&mut self, b: &[u8]) -> io::Result<usize> {
            self.output.extend(b);
            Ok(b.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    for (raw, valid) in [
        (
            b"1\r\na\r\n2;ext=yes\r\nbc\r\n0\r\nX-Test: yes\r\n\r\n".as_slice(),
            true,
        ),
        (b"3\r\nab".as_slice(), false),
        (b"+3\r\nabc\r\n0\r\n\r\n".as_slice(), false),
    ] {
        let mut socket = Duplex {
            input: Cursor::new(raw.to_vec()),
            output: Vec::new(),
        };
        let mut body = Body::new(&mut socket, Vec::new(), 0, true);
        let mut out = Vec::new();
        let result = body.read_to_end(&mut out);
        assert_eq!(result.is_ok(), valid);
        if valid {
            assert_eq!(out, b"abc");
        }
    }
}

#[test]
fn enhance_without_model_returns_clear_error() {
    let service = EnhanceService::new(None);
    let (mut client, mut server) = pair();
    let wav = output_header(Some(0)).unwrap();
    write!(
        client,
        "POST /v1/audio/enhance HTTP/1.1\r\nContent-Type: audio/wav\r\nContent-Length: {}\r\n\r\n",
        wav.len()
    )
    .unwrap();
    client.write_all(&wav).unwrap();
    let (mut request, length) = read_head(&mut server, Instant::now() + Duration::from_secs(1))
        .unwrap()
        .unwrap();
    let error = service
        .handle(&mut server, &mut request, length)
        .unwrap_err();
    assert!(error.to_string().contains("enhance non configuré"));
    assert!(headers(&mut BufReader::new(client)).starts_with("HTTP/1.1 400"));
}

#[cfg(any(feature = "ws", feature = "h2", feature = "webtransport"))]
#[test]
fn idle_dfn3_waits_for_the_end_of_the_stream() {
    let service = EnhanceService::new(None);
    *service.model.lock().unwrap() = Some(super::test_support::model());
    let start = Instant::now();
    let at = |ms| move || start + Duration::from_millis(ms);
    assert!(!service.unload_idle(0, at(600_000)), "0 : désactivé");
    let slot = service.reserve().unwrap();
    let held = service.model().unwrap();
    assert!(!service.unload_idle(1, at(5_000)), "flux actif : on garde");
    drop(slot);
    // Fin datée par la réservation : délai compté depuis elle.
    let ended = *service.usage.lock().unwrap();
    let after = |ms| move || ended + Duration::from_millis(ms);
    assert!(!service.unload_idle(1, after(999)));
    assert!(service.unload_idle(1, after(1_000)));
    assert!(service.model.lock().unwrap().is_none());
    // L'Arc déjà cloné reste utilisable après le vidage du slot.
    let _stream = DenoiseStream::new(&held);
    assert!(!service.unload_idle(1, at(600_000)), "déjà vide");
}

#[cfg(any(feature = "ws", feature = "h2", feature = "webtransport"))]
#[test]
fn idle_dfn3_counts_from_a_stream_ended_between_two_passes() {
    // Délai 1 s, modèle pris à ~0, flux fini à 0,24 s, passage suivant à 1,02 s.
    let service = EnhanceService::new(None);
    *service.model.lock().unwrap() = Some(super::test_support::model());
    let start = Instant::now();
    let slot = service.reserve().unwrap();
    let held = service.model().unwrap();
    thread::sleep(Duration::from_millis(240));
    drop((held, slot));
    let at = |ms| move || start + Duration::from_millis(ms);
    assert!(
        !service.unload_idle(1, at(1_020)),
        "0,78 s d'inactivité seulement"
    );
    assert!(service.unload_idle(1, at(1_400)));
}

#[cfg(any(feature = "ws", feature = "h2", feature = "webtransport"))]
#[test]
fn idle_dfn3_keeps_a_model_whose_loading_outlasted_the_delay() {
    let service = EnhanceService::new(Some(PathBuf::from("lent.safetensors")));
    // Chemin h2 : réservation, puis chargement plus long que le délai d'1 s.
    let slot = service.reserve().unwrap();
    let held = thread::scope(|scope| {
        let loader = scope.spawn(|| {
            service.model_with(|_| {
                thread::sleep(Duration::from_millis(1500));
                Ok(super::test_support::model())
            })
        });
        thread::sleep(Duration::from_millis(200));
        // Bloqué derrière le chargement ; l'horloge est lue après.
        assert!(!service.unload_idle(1, Instant::now));
        loader.join().unwrap().unwrap()
    });
    // Flux bref après un long chargement : aucun déchargement immédiat.
    drop((held, slot));
    assert!(!service.unload_idle(1, Instant::now));
    assert!(!service.unload_idle(1, Instant::now));
    thread::sleep(Duration::from_millis(1100));
    assert!(service.unload_idle(1, Instant::now));
}
