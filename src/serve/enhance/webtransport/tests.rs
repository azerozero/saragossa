use super::super::test_support::{model, oracle, signal};
use super::*;
use sha2::{Digest, Sha256};

struct Fixture {
    client: web_transport_quinn::Client,
    front: Arc<Front>,
    url: url::Url,
    task: tokio::task::JoinHandle<()>,
    endpoint: quinn::Endpoint,
}
impl Drop for Fixture {
    fn drop(&mut self) {
        self.task.abort();
        self.endpoint.close(0_u32.into(), b"test termine");
    }
}
async fn setup() -> Fixture {
    let addr = "127.0.0.1:0".parse().unwrap();
    let identity = super::super::certificate::identity(addr).unwrap();
    assert_eq!(
        identity.info.server_certificate_hashes[0].value,
        Sha256::digest(&identity.der).to_vec()
    );
    let mut roots = rustls::RootCertStore::empty();
    roots.add(identity.der).unwrap();
    let endpoint = quinn::Endpoint::server(identity.config, addr).unwrap();
    let mut tls = rustls::ClientConfig::builder_with_provider(Arc::new(
        rustls::crypto::ring::default_provider(),
    ))
    .with_protocol_versions(&[&rustls::version::TLS13])
    .unwrap()
    .with_root_certificates(roots)
    .with_no_client_auth();
    tls.alpn_protocols = vec![b"h3".to_vec()];
    let crypto = quinn::crypto::rustls::QuicClientConfig::try_from(tls).unwrap();
    let mut config = quinn::ClientConfig::new(Arc::new(crypto));
    let mut transport = quinn::TransportConfig::default();
    transport
        .stream_receive_window(16384_u32.into())
        .receive_window(65536_u32.into());
    config.transport_config(Arc::new(transport));
    let client = web_transport_quinn::Client::new(quinn::Endpoint::client(addr).unwrap(), config);
    let front = Arc::new(Front::new(Some("test-key".into()), vec![]));
    *front.enhance.model.lock().unwrap() = Some(model());
    let url = format!(
        "https://127.0.0.1:{}/v1/audio/enhance/wt?token=test-key&rate=48000&format=f32",
        endpoint.local_addr().unwrap().port()
    )
    .parse()
    .unwrap();
    let task = tokio::spawn(listen(
        endpoint.clone(),
        Arc::clone(&front),
        Arc::new(AtomicUsize::new(0)),
        tokio::sync::watch::channel(false).1,
    ));
    Fixture {
        client,
        front,
        url,
        task,
        endpoint,
    }
}
async fn read_frame(recv: &mut RecvStream) -> Vec<u8> {
    let mut len = [0; 4];
    deadline(recv.read_exact(&mut len)).await.unwrap();
    let len = u32::from_le_bytes(len) as usize;
    assert!(len <= MAX_FRAME);
    let mut bytes = vec![0; len];
    deadline(recv.read_exact(&mut bytes)).await.unwrap();
    bytes
}
#[tokio::test]
async fn duplex_matches_batch_for_both_formats() {
    for (rate, float) in [(48000, true), (16000, false)] {
        let mut fixture = setup().await;
        fixture.url.set_query(Some(&format!(
            "token=test-key&rate={rate}&format={}",
            if float { "f32" } else { "s16" }
        )));
        let session = fixture.client.connect(fixture.url.clone()).await.unwrap();
        let (mut send, mut recv) = session.open_bi().await.unwrap();
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
        write_frame(&mut send, &bytes[..8000]).await.unwrap();
        let mut output = read_frame(&mut recv).await;
        assert!(!output.is_empty());
        let producer = tokio::spawn(async move {
            for b in bytes[8000..].chunks(4093) {
                write_frame(&mut send, b).await.unwrap();
            }
            write_frame(&mut send, &[]).await.unwrap();
            send.finish().unwrap();
        });
        loop {
            let b = read_frame(&mut recv).await;
            if b.is_empty() {
                break;
            }
            output.extend(b);
        }
        assert!(deadline(recv.read(&mut [0; 1])).await.unwrap().is_none());
        producer.await.unwrap();
        assert_eq!(output, oracle(&input, Config { rate, float }));
    }
}
#[tokio::test]
async fn shared_admission_returns_http_503_before_accepting_audio() {
    let fixture = setup().await;
    let _slot = Reservation::acquire(
        &fixture.front.enhance.active,
        enhance_max_streams(),
        enhance_max_streams(),
    )
    .unwrap();
    let e = fixture
        .client
        .connect(fixture.url.clone())
        .await
        .unwrap_err();
    assert!(e.to_string().contains("got: Some(503)"), "{e}");
}
#[tokio::test]
async fn invalid_token_and_rate_are_rejected() {
    let mut fixture = setup().await;
    for (query, status) in [
        (
            "token=wrong&rate=48000&format=f32",
            http::StatusCode::UNAUTHORIZED,
        ),
        (
            "token=test-key&rate=0&format=f32",
            http::StatusCode::BAD_REQUEST,
        ),
    ] {
        fixture.url.set_query(Some(query));
        let error = fixture
            .client
            .connect(fixture.url.clone())
            .await
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains(&format!("got: Some({})", status.as_u16())),
            "{error}"
        );
    }
}
#[tokio::test]
async fn oversized_frame_and_partial_sample_reset_the_session() {
    for bytes in [
        (MAX_FRAME as u32 + 1).to_le_bytes().to_vec(),
        vec![1, 0, 0, 0, 42, 0, 0, 0, 0],
    ] {
        let fixture = setup().await;
        let session = fixture.client.connect(fixture.url.clone()).await.unwrap();
        let (mut send, mut recv) = session.open_bi().await.unwrap();
        send.write_all(&bytes).await.unwrap();
        send.finish().unwrap();
        assert!(deadline(recv.read(&mut [0; 4])).await.is_err());
    }
}
#[tokio::test]
async fn idle_client_releases_slot() {
    let fixture = setup().await;
    let session = fixture.client.connect(fixture.url.clone()).await.unwrap();
    let (mut send, mut recv) = session.open_bi().await.unwrap();
    // Rend le stream visible côté serveur, puis abandonne le corps annoncé.
    send.write_all(&4_u32.to_le_bytes()).await.unwrap();
    let result = tokio::time::timeout(
        // Marge proportionnelle : indépendante de la charge de la machine.
        enhance_idle_timeout() * 4,
        recv.read(&mut [0; 4]),
    )
    .await
    .unwrap();
    assert!(result.is_err());
    for _ in 0..100 {
        if fixture
            .front
            .enhance
            .active
            .load(std::sync::atomic::Ordering::Acquire)
            == 0
        {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    panic!("place DFN3 non rendue");
}
#[tokio::test]
async fn certificate_metadata_is_exposed_by_authenticated_http_route() {
    let front = Front::new(Some("test-key".into()), vec![]);
    let endpoint = bind(
        std::net::UdpSocket::bind("127.0.0.1:0").unwrap(),
        &front,
        false,
    )
    .unwrap();
    let raw =
        b"GET /v1/audio/enhance/certificate HTTP/1.1\r\nAuthorization: Bearer test-key\r\n\r\n";
    let (request, _) = crate::serve::http::read_head(
        &mut std::io::Cursor::new(raw),
        std::time::Instant::now() + std::time::Duration::from_secs(1),
    )
    .unwrap()
    .unwrap();
    let mut response = Vec::new();
    crate::serve::http::route_front(&mut response, &front, &request)
        .unwrap()
        .unwrap();
    let response = String::from_utf8(response).unwrap();
    assert!(response.starts_with("HTTP/1.1 200"));
    let body: serde_json::Value =
        serde_json::from_str(response.split_once("\r\n\r\n").unwrap().1).unwrap();
    assert_eq!(
        body["serverCertificateHashes"][0]["value"]
            .as_array()
            .unwrap()
            .len(),
        32
    );
    assert!(body["url"]
        .as_str()
        .unwrap()
        .contains(&endpoint.local_addr().unwrap().port().to_string()));
    endpoint.close(0_u32.into(), b"fin");
}
#[cfg(feature = "ws")]
#[test]
#[ignore = "mesure ABBA uniquement sous load1 < 2"]
fn webtransport_abba_latency() {
    use super::super::latency::{abba, Timing};
    use std::time::{Duration, Instant};
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    abba("webtransport", || {
        runtime.block_on(async {
            let fixture = setup().await;
            let start = Instant::now();
            let session = fixture.client.connect(fixture.url.clone()).await.unwrap();
            let first_byte = start.elapsed();
            let (mut send, mut recv) = session.open_bi().await.unwrap();
            let start = Instant::now();
            let mut first_pcm = Duration::ZERO;
            let mut hops = vec![];
            for hop in 0..64 {
                let sent = Instant::now();
                let bytes: Vec<_> = (0..480)
                    .flat_map(|i| ((i as f32 * 0.03).sin() * 0.2).to_le_bytes())
                    .collect();
                write_frame(&mut send, &bytes).await.unwrap();
                if hop >= 3 {
                    assert!(!read_frame(&mut recv).await.is_empty());
                    if hop == 3 {
                        first_pcm = start.elapsed();
                    } else {
                        hops.push(sent.elapsed());
                    }
                }
            }
            write_frame(&mut send, &[]).await.unwrap();
            send.finish().unwrap();
            while !read_frame(&mut recv).await.is_empty() {}
            Timing {
                first_byte,
                first_pcm,
                hops,
            }
        })
    });
}

#[tokio::test]
async fn unread_output_applies_backpressure_and_releases_slot() {
    let fixture = setup().await;
    let session = fixture.client.connect(fixture.url.clone()).await.unwrap();
    let (mut send, _recv) = session.open_bi().await.unwrap();
    let producer = tokio::spawn(async move {
        for _ in 0..128 {
            if write_frame(&mut send, &[0; MAX_FRAME]).await.is_err() {
                return;
            }
        }
        panic!("le flux a consommé plus que les fenêtres sans lecteur");
    });
    tokio::time::timeout(
        // Marge proportionnelle : indépendante de la charge de la machine.
        enhance_idle_timeout() * 4,
        session.closed(),
    )
    .await
    .unwrap();
    producer.await.unwrap();
    for _ in 0..100 {
        if fixture
            .front
            .enhance
            .active
            .load(std::sync::atomic::Ordering::Acquire)
            == 0
        {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    panic!("réservation retenue après contre-pression");
}
