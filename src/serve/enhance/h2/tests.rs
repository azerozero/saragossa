use super::super::test_support::{model, oracle, signal};
use super::*;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

pub(super) async fn setup() -> (
    protocol::client::SendRequest<Bytes>,
    tokio::task::JoinHandle<()>,
    tokio::task::JoinHandle<ServeResult<()>>,
    Arc<Front>,
) {
    let front = Arc::new(Front::new(Some("test-key".into()), vec![]));
    *front.enhance.model.lock().unwrap() = Some(model());
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let socket = TcpStream::connect(listener.local_addr().unwrap())
        .await
        .unwrap();
    let (server, _) = listener.accept().await.unwrap();
    let server_front = Arc::clone(&front);
    let server = tokio::spawn(connection(server, server_front));
    let (client, connection) = protocol::client::handshake(socket).await.unwrap();
    let driver = tokio::spawn(async move {
        let _ = connection.await;
    });
    (client, driver, server, front)
}
fn request(query: &str) -> Request<()> {
    Request::builder()
        .method("POST")
        .uri(format!("http://localhost/v1/audio/enhance/h2?{query}"))
        .header("authorization", "Bearer test-key")
        .body(())
        .unwrap()
}
#[tokio::test]
async fn duplex_matches_batch_for_resampled_pcm() {
    for (rate, float) in [(48000, true), (16000, false)] {
        let (mut client, driver, server, _) = setup().await;
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
        let (response, mut send) = client
            .send_request(
                request(&format!(
                    "rate={rate}&format={}",
                    if float { "f32" } else { "s16" }
                )),
                false,
            )
            .unwrap();
        let response = deadline(response).await.unwrap();
        assert_eq!(response.status(), 200);
        let mut recv = response.into_body();
        send_bytes(&mut send, bytes[..8000].to_vec(), false)
            .await
            .unwrap();
        let first = recv.data().await.unwrap().unwrap();
        assert!(!first.is_empty());
        let mut output = first.to_vec();
        recv.flow_control().release_capacity(first.len()).unwrap();
        let producer = tokio::spawn(async move {
            for chunk in bytes[8000..].chunks(4093) {
                send_bytes(&mut send, chunk.to_vec(), false).await.unwrap();
            }
            send_bytes(&mut send, vec![], true).await.unwrap();
        });
        while let Some(frame) = recv.data().await {
            let frame = frame.unwrap();
            output.extend_from_slice(&frame);
            recv.flow_control().release_capacity(frame.len()).unwrap();
        }
        producer.await.unwrap();
        assert_eq!(output, oracle(&input, config));
        driver.abort();
        server.abort();
    }
}
#[tokio::test]
async fn admission_auth_validation_and_shared_503() {
    let (mut client, driver, server, front) = setup().await;
    let slot = Reservation::acquire(
        &front.enhance.active,
        enhance_max_streams(),
        enhance_max_streams(),
    )
    .unwrap();
    let (response, _send) = client
        .send_request(request("rate=48000&format=f32"), false)
        .unwrap();
    assert_eq!(response.await.unwrap().status(), 503);
    drop(slot);
    let (response, _send) = client
        .send_request(request("rate=0&format=f32"), false)
        .unwrap();
    assert_eq!(response.await.unwrap().status(), 400);
    let mut req = request("rate=48000&format=f32");
    req.headers_mut().remove("authorization");
    let (response, _send) = client.send_request(req, false).unwrap();
    assert_eq!(response.await.unwrap().status(), 401);
    assert_eq!(
        front
            .enhance
            .active
            .load(std::sync::atomic::Ordering::Acquire),
        0
    );
    driver.abort();
    server.abort();
}
#[tokio::test]
async fn idle_stream_is_reset_and_slot_released() {
    let (mut client, driver, server, front) = setup().await;
    let (response, _send) = client
        .send_request(request("rate=48000&format=f32"), false)
        .unwrap();
    let mut recv = response.await.unwrap().into_body();
    let result = tokio::time::timeout(
        enhance_idle_timeout() + std::time::Duration::from_secs(5),
        recv.data(),
    )
    .await
    .unwrap();
    assert!(result.unwrap().is_err());
    for _ in 0..100 {
        if front
            .enhance
            .active
            .load(std::sync::atomic::Ordering::Acquire)
            == 0
        {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert_eq!(
        front
            .enhance
            .active
            .load(std::sync::atomic::Ordering::Acquire),
        0
    );
    driver.abort();
    server.abort();
}
#[tokio::test]
async fn oversized_wire_frame_produces_frame_size_error() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let mut client = TcpStream::connect(listener.local_addr().unwrap())
        .await
        .unwrap();
    let (socket, _) = listener.accept().await.unwrap();
    let server = tokio::spawn(connection(
        socket,
        Arc::new(Front::new(Some("test-key".into()), vec![])),
    ));
    client
        .write_all(b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n")
        .await
        .unwrap();
    client
        .write_all(&[0, 0, 0, 4, 0, 0, 0, 0, 0])
        .await
        .unwrap();
    client
        .write_all(&[0, 64, 1, 0, 0, 0, 0, 0, 1])
        .await
        .unwrap();
    client.write_all(&vec![0; 16385]).await.unwrap();
    loop {
        let mut head = [0; 9];
        deadline(client.read_exact(&mut head)).await.unwrap();
        let n = (usize::from(head[0]) << 16) | (usize::from(head[1]) << 8) | usize::from(head[2]);
        assert!(n <= 65536);
        let mut bytes = vec![0; n];
        client.read_exact(&mut bytes).await.unwrap();
        if head[3] == 7 {
            assert_eq!(&bytes[4..8], &6_u32.to_be_bytes());
            break;
        }
    }
    assert!(server.await.unwrap().is_err());
}
#[cfg(feature = "ws")]
#[test]
#[ignore = "mesure ABBA uniquement sous load1 < 2"]
fn h2_abba_latency() {
    use super::super::latency::{abba, Timing};
    use std::time::{Duration, Instant};
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    abba("h2", || {
        runtime.block_on(async {
            let (mut client, driver, server, _) = setup().await;
            let start = Instant::now();
            let (response, mut send) = client
                .send_request(request("rate=48000&format=f32"), false)
                .unwrap();
            let mut recv = response.await.unwrap().into_body();
            let first_byte = start.elapsed();
            let start = Instant::now();
            let mut first_pcm = Duration::ZERO;
            let mut hops = vec![];
            for hop in 0..64 {
                let sent = Instant::now();
                let bytes = (0..480)
                    .flat_map(|i| ((i as f32 * 0.03).sin() * 0.2).to_le_bytes())
                    .collect();
                send_bytes(&mut send, bytes, false).await.unwrap();
                if hop >= 3 {
                    let bytes = recv.data().await.unwrap().unwrap();
                    assert!(!bytes.is_empty());
                    recv.flow_control().release_capacity(bytes.len()).unwrap();
                    if hop == 3 {
                        first_pcm = start.elapsed();
                    } else {
                        hops.push(sent.elapsed());
                    }
                }
            }
            send_bytes(&mut send, vec![], true).await.unwrap();
            while let Some(b) = recv.data().await {
                let b = b.unwrap();
                recv.flow_control().release_capacity(b.len()).unwrap();
            }
            driver.abort();
            server.abort();
            Timing {
                first_byte,
                first_pcm,
                hops,
            }
        })
    });
}

#[tokio::test]
async fn backpressure_waits_for_window_and_times_out() {
    let (mut client, driver, server, front) = setup().await;
    let (response, mut send) = client
        .send_request(request("rate=48000&format=f32"), false)
        .unwrap();
    let mut recv = response.await.unwrap().into_body();
    // La fenêtre réponse vaut 65535. On ne libère jamais sa capacité.
    let producer = tokio::spawn(async move {
        for _ in 0..32 {
            if send_bytes(&mut send, vec![0; MAX_FRAME], false)
                .await
                .is_err()
            {
                return;
            }
        }
        panic!("le serveur a consommé une entrée non bornée sans lecteur");
    });
    let result = tokio::time::timeout(
        enhance_idle_timeout() + std::time::Duration::from_secs(5),
        async {
            let mut total = 0;
            while let Some(frame) = recv.data().await {
                match frame {
                    Ok(bytes) => total += bytes.len(),
                    Err(_) => return total,
                }
            }
            panic!("le flux doit être réinitialisé sans END_STREAM");
        },
    )
    .await
    .unwrap();
    assert!(result <= 65535);
    producer.await.unwrap();
    for _ in 0..100 {
        if front
            .enhance
            .active
            .load(std::sync::atomic::Ordering::Acquire)
            == 0
        {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert_eq!(
        front
            .enhance
            .active
            .load(std::sync::atomic::Ordering::Acquire),
        0
    );
    driver.abort();
    server.abort();
}
