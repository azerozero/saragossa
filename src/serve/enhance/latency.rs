//! Mesures locales ABBA, exclues des gates fonctionnelles.
use super::test_support::{headers, pair, service};
use super::*;
use std::io::{BufRead, BufReader};
use std::time::{Duration, Instant};

pub(super) struct Timing {
    pub first_byte: Duration,
    pub first_pcm: Duration,
    pub hops: Vec<Duration>,
}
pub(super) fn low_load() {
    let output = std::process::Command::new("python3")
        .args(["-I", "-c", "import os;print(os.getloadavg()[0])"])
        .output()
        .unwrap();
    assert!(output.status.success());
    let load: f64 = std::str::from_utf8(&output.stdout)
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    assert!(load < 2.0, "mesure annulée: load1={load} >= 2");
}
fn chunk(reader: &mut impl BufRead) -> Vec<u8> {
    let mut line = String::new();
    reader.read_line(&mut line).unwrap();
    let n = usize::from_str_radix(line.trim(), 16).unwrap();
    let mut b = vec![0; n];
    reader.read_exact(&mut b).unwrap();
    let mut crlf = [0; 2];
    reader.read_exact(&mut crlf).unwrap();
    assert_eq!(crlf, *b"\r\n");
    b
}
pub(super) fn tcp_trial(ws: bool) -> Timing {
    let service = service();
    let (mut client, mut server) = pair();
    client.set_nodelay(true).unwrap();
    server.set_nodelay(true).unwrap();
    let worker = std::thread::spawn(move || {
        let (mut request, n) =
            super::super::http::read_head(&mut server, Instant::now() + Duration::from_secs(5))
                .unwrap()
                .unwrap();
        if ws {
            service.handle_ws(&mut server, &mut request)
        } else {
            service.handle(&mut server, &mut request, n)
        }
        .unwrap();
    });
    let mut reader = BufReader::new(client.try_clone().unwrap());
    let start = Instant::now();
    if ws {
        client.write_all(&super::ws::tests::request()).unwrap();
    } else {
        client.write_all(b"POST /v1/audio/enhance HTTP/1.1\r\nContent-Type: audio/wav\r\nTransfer-Encoding: chunked\r\n\r\n").unwrap();
        write_chunk(
            &mut client,
            &saragossa::denoise::wav::output_header(None).unwrap(),
        )
        .unwrap();
    }
    assert!(!reader.fill_buf().unwrap().is_empty());
    let first_byte = start.elapsed();
    headers(&mut reader);
    if !ws {
        chunk(&mut reader);
    }
    let start = Instant::now();
    let mut hops = Vec::new();
    let mut first_pcm = Duration::ZERO;
    for hop in 0..64 {
        let sent = Instant::now();
        let b: Vec<u8> = (0..480)
            .flat_map(|i| ((i as f32 * 0.03).sin() * 0.2).to_le_bytes())
            .collect();
        if ws {
            super::ws::tests::masked(&mut client, 2, &b);
        } else {
            write_chunk(&mut client, &b).unwrap();
        }
        if hop >= 3 {
            let b = if ws {
                super::ws::tests::frame(&mut reader).1
            } else {
                chunk(&mut reader)
            };
            assert!(!b.is_empty());
            if hop == 3 {
                first_pcm = start.elapsed();
            } else {
                hops.push(sent.elapsed());
            }
        }
    }
    if ws {
        super::ws::tests::masked(&mut client, 2, &[]);
        while super::ws::tests::frame(&mut reader).0 != 8 {}
    } else {
        client.write_all(b"0\r\n\r\n").unwrap();
        while !chunk(&mut reader).is_empty() {}
    }
    worker.join().unwrap();
    Timing {
        first_byte,
        first_pcm,
        hops,
    }
}
pub(super) fn abba(name: &str, mut candidate: impl FnMut() -> Timing) {
    // Warm-up hors mesure, mêmes poids et mêmes hops pour chaque transport.
    low_load();
    tcp_trial(false);
    candidate();
    let mut base = Vec::new();
    let mut other = Vec::new();
    let blocks = std::env::var("SARAGOSSA_ABBA_BLOCKS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(6);
    for _ in 0..blocks {
        for is_candidate in [false, true, true, false] {
            low_load();
            let result = if is_candidate {
                candidate()
            } else {
                tcp_trial(false)
            };
            low_load();
            if is_candidate {
                other.push(result);
            } else {
                base.push(result);
            }
        }
    }
    for (label, trials) in [("chunked", base), (name, other)] {
        fn percentile(mut values: Vec<Duration>, p: usize) -> u128 {
            values.sort();
            values[(values.len() - 1) * p / 100].as_micros()
        }
        let first = trials.iter().map(|t| t.first_byte).collect();
        let pcm = trials.iter().map(|t| t.first_pcm).collect();
        let hops: Vec<_> = trials.into_iter().flat_map(|t| t.hops).collect();
        eprintln!(
            "{label}: first_response_p50={}us first_pcm_p50={}us hop_p50={}us hop_p95={}us hop_p99={}us",
            percentile(first, 50),
            percentile(pcm, 50),
            percentile(hops.clone(), 50),
            percentile(hops.clone(), 95),
            percentile(hops, 99)
        );
    }
}
#[test]
#[ignore = "mesure ABBA uniquement sous load1 < 2"]
fn ws_abba_latency() {
    abba("ws", || tcp_trial(true));
}
