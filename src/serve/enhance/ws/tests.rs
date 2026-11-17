use super::super::test_support::*;
use super::*;
use crate::serve::http::read_head;
use base64::Engine;
use std::io::BufReader;
use std::time::{Duration, Instant};

/// Clé d'exemple RFC 6455, encodée à l'exécution pour ne pas figer de littéral.
pub(crate) fn key() -> String {
    base64::engine::general_purpose::STANDARD.encode(b"the sample nonce")
}
pub(crate) fn request() -> Vec<u8> {
    format!("GET /v1/audio/enhance/ws?rate=48000&format=f32 HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Version: 13\r\nSec-WebSocket-Key: {}\r\n\r\n", key()).into_bytes()
}
pub(crate) fn masked(writer: &mut impl Write, opcode: u8, bytes: &[u8]) {
    writer.write_all(&[0x80 | opcode]).unwrap();
    if bytes.len() < 126 {
        writer.write_all(&[0x80 | bytes.len() as u8]).unwrap();
    } else {
        writer.write_all(&[0xfe]).unwrap();
        writer
            .write_all(&(bytes.len() as u16).to_be_bytes())
            .unwrap();
    }
    writer.write_all(&[1, 2, 3, 4]).unwrap();
    writer
        .write_all(
            &bytes
                .iter()
                .enumerate()
                .map(|(i, b)| b ^ [1, 2, 3, 4][i % 4])
                .collect::<Vec<_>>(),
        )
        .unwrap();
}
pub(crate) fn frame(reader: &mut impl Read) -> (u8, Vec<u8>) {
    let mut head = [0; 2];
    reader.read_exact(&mut head).unwrap();
    assert_eq!(head[1] & 0x80, 0);
    let len = match head[1] {
        126 => {
            let mut b = [0; 2];
            reader.read_exact(&mut b).unwrap();
            u16::from_be_bytes(b) as usize
        }
        127 => {
            let mut b = [0; 8];
            reader.read_exact(&mut b).unwrap();
            u64::from_be_bytes(b) as usize
        }
        n => n as usize,
    };
    let mut bytes = vec![0; len];
    reader.read_exact(&mut bytes).unwrap();
    (head[0] & 15, bytes)
}
#[test]
fn duplex_is_bit_exact_with_batch_and_ping_pong() {
    for (rate, float) in [(48000, true), (16000, false)] {
        let service = service();
        let (mut client, mut server) = pair();
        let worker = std::thread::spawn(move || {
            let (mut req, _) = read_head(&mut server, Instant::now() + Duration::from_secs(5))
                .unwrap()
                .unwrap();
            service.handle_ws(&mut server, &mut req).unwrap();
        });
        let config = Config { rate, float };
        let input = signal(5777);
        let bytes: Vec<u8> = if float {
            input.iter().flat_map(|s| s.to_le_bytes()).collect()
        } else {
            input
                .iter()
                .flat_map(|s| crate::serve::audio::pcm_i16(*s).to_le_bytes())
                .collect()
        };
        let input = if float {
            input
        } else {
            bytes
                .chunks_exact(2)
                .map(|b| f32::from(i16::from_le_bytes([b[0], b[1]])) / 32767.0)
                .collect()
        };
        let req = String::from_utf8(request()).unwrap().replace(
            "48000&format=f32",
            &format!("{rate}&format={}", if float { "f32" } else { "s16" }),
        );
        client.write_all(req.as_bytes()).unwrap();
        // Pipeline une trame avec le handshake : le préfixe HTTP doit être conservé.
        masked(&mut client, 9, b"ping");
        let mut reader = BufReader::new(client.try_clone().unwrap());
        let head = headers(&mut reader);
        assert!(head.starts_with("HTTP/1.1 101"));
        assert!(head.contains("s3pPLMBiTxaQ9kYGzzhZRbK+xOo="));
        assert_eq!(frame(&mut reader), (10, b"ping".to_vec()));
        masked(&mut client, 2, &bytes[..8000]);
        let (op, mut output) = frame(&mut reader);
        assert_eq!(op, 2);
        assert!(!output.is_empty());
        for chunk in bytes[8000..].chunks(4093) {
            masked(&mut client, 2, chunk);
        }
        masked(&mut client, 2, &[]);
        loop {
            let (op, b) = frame(&mut reader);
            if op == 8 {
                break;
            }
            assert_eq!(op, 2);
            output.extend(b);
        }
        worker.join().unwrap();
        assert_eq!(output, oracle(&input, config));
    }
}
#[test]
fn framing_mask_size_control_and_fragment_rules() {
    for raw in [
        vec![0x82, 0],
        vec![0xc2, 0x80],
        vec![0x89, 0xfe, 0, 126],
        vec![0x82, 0xfe, 0, 1],
    ] {
        assert_eq!(read_frame(&mut Cursor::new(raw)).unwrap_err().0, 1002);
    }
    assert_eq!(
        read_frame(&mut Cursor::new([0x82, 0xfe, 0x40, 1]))
            .unwrap_err()
            .0,
        1009
    );
    assert!(!valid_close(&[1]));
    assert!(!valid_close(&1005_u16.to_be_bytes()));
    assert!(!valid_close(&[3, 232, 255]));
}
#[test]
fn capacity_is_shared_with_chunked() {
    let service = EnhanceService::new(None);
    let _slot = Reservation::acquire(
        &service.active,
        enhance_max_streams(),
        enhance_max_streams(),
    )
    .unwrap();
    let (mut client, mut server) = pair();
    client.write_all(&request()).unwrap();
    let (mut req, _) = read_head(&mut server, Instant::now() + Duration::from_secs(1))
        .unwrap()
        .unwrap();
    service.handle_ws(&mut server, &mut req).unwrap();
    assert!(headers(&mut BufReader::new(client)).starts_with("HTTP/1.1 503"));
}
#[test]
fn idle_and_backpressure_close_the_socket() {
    let (mut client, mut server) = pair();
    server.configure(Duration::from_millis(80)).unwrap();
    let model = model();
    let worker = std::thread::spawn(move || {
        let mut pcm = Pcm::new(
            &model,
            Config {
                rate: 48000,
                float: true,
            },
        )
        .unwrap();
        assert!(session(&mut server, &mut pcm, Duration::from_millis(80)).is_err());
    });
    assert_eq!(frame(&mut client).0, 8);
    worker.join().unwrap();
    let (_client, mut server) = pair();
    server.configure(Duration::from_millis(80)).unwrap();
    let start = Instant::now();
    assert!((0..1024).any(|_| write_frame(&mut server, 2, &[0; 16384]).is_err()));
    assert!(start.elapsed() < Duration::from_secs(5));
}

#[test]
fn oversized_frame_and_unmasked_client_are_closed_on_wire() {
    for (bad, expected) in [(vec![0x82, 0xfe, 0x40, 1], 1009_u16), (vec![0x82, 0], 1002)] {
        let service = service();
        let (mut client, mut server) = pair();
        let worker = std::thread::spawn(move || {
            let (mut req, _) = read_head(&mut server, Instant::now() + Duration::from_secs(5))
                .unwrap()
                .unwrap();
            assert!(service.handle_ws(&mut server, &mut req).is_err());
            assert_eq!(service.active.load(std::sync::atomic::Ordering::Acquire), 0);
        });
        client.write_all(&request()).unwrap();
        let mut reader = BufReader::new(client.try_clone().unwrap());
        headers(&mut reader);
        client.write_all(&bad).unwrap();
        assert_eq!(frame(&mut reader), (8, expected.to_be_bytes().to_vec()));
        worker.join().unwrap();
    }
}
#[test]
fn fragmented_message_accepts_interleaved_ping() {
    let service = service();
    let (mut client, mut server) = pair();
    let worker = std::thread::spawn(move || {
        let (mut req, _) = read_head(&mut server, Instant::now() + Duration::from_secs(5))
            .unwrap()
            .unwrap();
        service.handle_ws(&mut server, &mut req).unwrap();
    });
    client.write_all(&request()).unwrap();
    let mut reader = BufReader::new(client.try_clone().unwrap());
    headers(&mut reader);
    let mut bytes = Vec::new();
    masked(&mut bytes, 2, &[0; 1920]);
    bytes[0] &= 0x7f;
    client.write_all(&bytes).unwrap();
    masked(&mut client, 9, b"p");
    assert_eq!(frame(&mut reader), (10, b"p".to_vec()));
    masked(&mut client, 0, &[0; 5760]);
    assert_eq!(frame(&mut reader).1.len(), 1920);
    masked(&mut client, 8, &1000_u16.to_be_bytes());
    assert_eq!(frame(&mut reader), (8, 1000_u16.to_be_bytes().to_vec()));
    worker.join().unwrap();
}
#[test]
fn invalid_handshake_and_query_precede_model_loading() {
    for raw in [
        String::from_utf8(request())
            .unwrap()
            .replace("rate=48000", "rate=0"),
        String::from_utf8(request())
            .unwrap()
            .replace(&key(), "AA=="),
        String::from_utf8(request())
            .unwrap()
            .replace("Sec-WebSocket-Version: 13", "Sec-WebSocket-Version: 12"),
    ] {
        let service = EnhanceService::new(None);
        let (mut client, mut server) = pair();
        client.write_all(raw.as_bytes()).unwrap();
        let (mut req, _) = read_head(&mut server, Instant::now() + Duration::from_secs(5))
            .unwrap()
            .unwrap();
        assert!(service.handle_ws(&mut server, &mut req).is_err());
        assert!(headers(&mut BufReader::new(client)).starts_with("HTTP/1.1 400"));
        assert!(service.model.lock().unwrap().is_none());
    }
}

struct Drip(Vec<u8>);
impl Read for Drip {
    fn read(&mut self, b: &mut [u8]) -> io::Result<usize> {
        std::thread::sleep(Duration::from_millis(20));
        if self.0.is_empty() {
            return Ok(0);
        }
        b[0] = self.0.remove(0);
        Ok(1)
    }
}
impl Write for Drip {
    fn write(&mut self, b: &[u8]) -> io::Result<usize> {
        Ok(b.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
fn pcm_config(model: &DeepFilterNet3) -> Pcm<'_> {
    Pcm::new(
        model,
        Config {
            rate: 48000,
            float: true,
        },
    )
    .unwrap()
}
#[test]
fn byte_per_read_client_is_evicted_by_frame_deadline() {
    let mut wire = Vec::new();
    masked(&mut wire, 2, &[0; 64]);
    let model = model();
    let mut pcm = pcm_config(&model);
    let start = Instant::now();
    // Chaque read() arrive avant 100 ms : seule l'échéance de trame peut expulser.
    let result = session(&mut Drip(wire), &mut pcm, Duration::from_millis(100));
    assert!(result.is_err());
    assert!(start.elapsed() < Duration::from_secs(2));
}
#[test]
fn ping_flood_without_audio_is_closed_1008() {
    struct Flood(Cursor<Vec<u8>>, Vec<u8>);
    impl Read for Flood {
        fn read(&mut self, b: &mut [u8]) -> io::Result<usize> {
            self.0.read(b)
        }
    }
    impl Write for Flood {
        fn write(&mut self, b: &[u8]) -> io::Result<usize> {
            self.1.extend_from_slice(b);
            Ok(b.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let mut wire = Vec::new();
    for _ in 0..(MAX_CONTROL_RUN + 2) {
        masked(&mut wire, 9, b"p");
    }
    let model = model();
    let mut pcm = pcm_config(&model);
    let mut flood = Flood(Cursor::new(wire), Vec::new());
    assert!(session(&mut flood, &mut pcm, Duration::from_secs(5)).is_err());
    let tail = &flood.1[flood.1.len() - 4..];
    assert_eq!(tail, [0x88, 2, 0x03, 0xf0]);
}
fn upgrade(service: &EnhanceService, raw: String) -> String {
    let (mut client, mut server) = pair();
    client.write_all(raw.as_bytes()).unwrap();
    let (mut req, _) = read_head(&mut server, Instant::now() + Duration::from_secs(5))
        .unwrap()
        .unwrap();
    // Le client ferme proprement pour que la session se termine.
    let worker = std::thread::scope(|scope| {
        let handle = scope.spawn(|| service.handle_ws(&mut server, &mut req));
        let mut reader = BufReader::new(client.try_clone().unwrap());
        let head = headers(&mut reader);
        if head.starts_with("HTTP/1.1 101") {
            masked(&mut client, 8, &1000_u16.to_be_bytes());
        }
        let _ = handle.join().unwrap();
        head
    });
    worker
}
#[test]
fn foreign_origin_is_refused_and_loopback_origin_accepted() {
    let service = service();
    let with = |origin: &str| {
        String::from_utf8(request()).unwrap().replace(
            "Host: localhost\r\n",
            &format!("Host: localhost\r\nOrigin: {origin}\r\n"),
        )
    };
    for bad in [
        "https://evil.example",
        "http://127.0.0.1.evil.example",
        "http://192.168.1.2:3000",
        "null",
    ] {
        assert!(
            upgrade(&service, with(bad)).starts_with("HTTP/1.1 403"),
            "{bad}"
        );
    }
    for good in [
        "http://127.0.0.1:5173",
        "http://localhost",
        "https://localhost:8443",
        "http://[::1]:3000",
    ] {
        assert!(
            upgrade(&service, with(good)).starts_with("HTTP/1.1 101"),
            "{good}"
        );
    }
    assert!(upgrade(&service, String::from_utf8(request()).unwrap()).starts_with("HTTP/1.1 101"));
}
#[test]
fn bearer_subprotocol_is_authenticated_and_echoed() {
    let raw = |protocols: &str| {
        String::from_utf8(request()).unwrap().replace(
            "\r\n\r\n",
            &format!("\r\nSec-WebSocket-Protocol: {protocols}\r\n\r\n"),
        )
    };
    let parse = |raw: String| {
        let (mut client, mut server) = pair();
        client.write_all(raw.as_bytes()).unwrap();
        read_head(&mut server, Instant::now() + Duration::from_secs(5))
            .unwrap()
            .unwrap()
            .0
    };
    let good = parse(raw("bearer.secret"));
    assert!(crate::serve::http::is_authorized(&good, "secret"));
    assert!(!crate::serve::http::is_authorized(&good, "other"));
    let none = parse(String::from_utf8(request()).unwrap());
    assert!(!crate::serve::http::is_authorized(&none, "secret"));
    let service = service();
    let head = upgrade(&service, raw("bearer.secret"));
    assert!(
        head.contains("Sec-WebSocket-Protocol: bearer.secret"),
        "{head}"
    );
    let head = upgrade(&service, raw("saragossa, bearer.secret"));
    assert!(head.contains("Sec-WebSocket-Protocol: saragossa"), "{head}");
}
#[test]
fn pcm_errors_map_to_close_codes() {
    assert_eq!(close_code(&invalid("x")), 1007);
    assert_eq!(
        close_code(&ServeError::io("x", io::Error::other("x"))),
        1011
    );
}
