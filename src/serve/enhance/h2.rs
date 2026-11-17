use ::h2 as protocol;
// HTTP/2 prior knowledge h2c sur un listener opt-in, avec fenêtres bornées.
use super::super::dispatch::Front;
use super::async_io::{deadline, error, Worker};
use super::pcm::{Config, MAX_FRAME};
use super::*;
use bytes::Bytes;
use http::{Request, Response};
use std::future::poll_fn;
use tokio::net::{TcpListener, TcpStream};
use tokio::task::JoinSet;

pub(super) async fn listen(
    listener: TcpListener,
    front: Arc<Front>,
    connections: Arc<AtomicUsize>,
) {
    let mut tasks = JoinSet::new();
    loop {
        tokio::select! {
            accepted = listener.accept() => {
                let (socket, _) = match accepted {
                    Ok(accepted) => accepted,
                    Err(e) => {
                        // NOTE: EMFILE ou ECONNABORTED sont transitoires, le listener survit.
                        eprintln!("enhance h2c accept: {e}");
                        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                        continue;
                    }
                };
                let Some(slot) = Reservation::acquire(&connections, 1, saragossa::runtime_flags::serve_max_connections()) else {
                    // Avant SETTINGS, aucune réponse HTTP/2 ne peut être envoyée.
                    drop(socket); continue;
                };
                let front = Arc::clone(&front);
                tasks.spawn(async move {
                    let _slot = slot;
                    if let Err(e) = connection(socket, front).await { eprintln!("{e}"); }
                });
            }
            _ = tasks.join_next(), if !tasks.is_empty() => {}
        }
    }
}
pub(super) async fn connection(socket: TcpStream, front: Arc<Front>) -> ServeResult<()> {
    socket.set_nodelay(true).map_err(error)?;
    let mut conn = deadline(
        protocol::server::Builder::new()
            .initial_window_size(MAX_FRAME as u32)
            .initial_connection_window_size((MAX_FRAME * 2) as u32)
            .max_frame_size(MAX_FRAME as u32)
            .max_header_list_size(16 * 1024)
            .max_concurrent_streams(64)
            .handshake::<_, Bytes>(socket),
    )
    .await?;
    let mut tasks = JoinSet::new();
    loop {
        let incoming = if tasks.is_empty() {
            tokio::time::timeout(enhance_idle_timeout(), conn.accept())
                .await
                .map_err(error)?
        } else {
            tokio::select! {
                request = conn.accept() => request,
                _ = tasks.join_next() => continue,
            }
        };
        let Some(incoming) = incoming else {
            break;
        };
        let (request, response) = incoming.map_err(error)?;
        let front = Arc::clone(&front);
        tasks.spawn(async move {
            if let Err(e) = request_stream(request, response, front).await {
                eprintln!("{e}");
            }
        });
    }
    Ok(())
}
async fn reject(
    response: &mut protocol::server::SendResponse<Bytes>,
    status: u16,
) -> ServeResult<()> {
    response
        .send_response(
            Response::builder().status(status).body(()).map_err(error)?,
            true,
        )
        .map_err(error)?;
    Ok(())
}
async fn request_stream(
    request: Request<protocol::RecvStream>,
    mut response: protocol::server::SendResponse<Bytes>,
    front: Arc<Front>,
) -> ServeResult<()> {
    if request.method() != "POST" || request.uri().path() != "/v1/audio/enhance/h2" {
        return reject(&mut response, 404).await;
    }
    if !front.api_key.as_deref().is_some_and(|key| {
        request
            .headers()
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
            .is_some_and(|v| super::super::http::constant_time_eq(v.as_bytes(), key.as_bytes()))
    }) {
        return reject(&mut response, 401).await;
    }
    let config = match Config::parse(request.uri().query().unwrap_or("")) {
        Ok(config) => config,
        Err(_) => return reject(&mut response, 400).await,
    };
    let Some(slot) = Reservation::acquire(&front.enhance.active, 1, enhance_max_streams()) else {
        return reject(&mut response, 503).await;
    };
    let loader = Arc::clone(&front);
    let model = match deadline(tokio::task::spawn_blocking(move || loader.enhance.model())).await? {
        Ok(model) => model,
        Err(e) => return reject(&mut response, super::super::http::error_status(&e)).await,
    };
    let mut worker = Worker::new(model, config, slot);
    let mut send = response
        .send_response(
            Response::builder()
                .status(200)
                .header("content-type", "application/octet-stream")
                .header("x-audio-rate", "48000")
                .body(())
                .map_err(error)?,
            false,
        )
        .map_err(error)?;
    let mut recv = request.into_body();
    let result = async {
        loop {
            let frame = tokio::time::timeout(enhance_idle_timeout(), recv.data())
                .await
                .map_err(error)?;
            let Some(frame) = frame else {
                break;
            };
            let frame = frame.map_err(error)?;
            let n = frame.len();
            if n > MAX_FRAME {
                return Err(error("trame h2 trop grande"));
            }
            let output = worker.process(Some(frame.to_vec())).await?;
            // Libère la fenêtre entrante seulement après consommation de la sortie.
            send_bytes(&mut send, output, false).await?;
            recv.flow_control().release_capacity(n).map_err(error)?;
        }
        if let Some(trailers) = deadline(recv.trailers()).await? {
            if !trailers.is_empty() {
                return Err(error("trailers PCM inattendus"));
            }
        }
        let output = worker.process(None).await?;
        send_bytes(&mut send, output, true).await
    }
    .await;
    if result.is_err() {
        send.send_reset(protocol::Reason::CANCEL);
    }
    result
}
pub(super) async fn send_bytes(
    send: &mut protocol::SendStream<Bytes>,
    bytes: Vec<u8>,
    end: bool,
) -> ServeResult<()> {
    let mut bytes = Bytes::from(bytes);
    while !bytes.is_empty() {
        send.reserve_capacity(bytes.len());
        let capacity = if send.capacity() > 0 {
            send.capacity()
        } else {
            tokio::time::timeout(enhance_idle_timeout(), poll_fn(|cx| send.poll_capacity(cx)))
                .await
                .map_err(error)?
                .ok_or_else(|| error("flux h2 fermé"))?
                .map_err(error)?
        };
        if capacity == 0 {
            continue;
        }
        let n = capacity.min(bytes.len()).min(MAX_FRAME);
        send.send_data(bytes.split_to(n), false).map_err(error)?;
    }
    if end {
        send.send_data(Bytes::new(), true).map_err(error)?;
    }
    Ok(())
}
#[cfg(test)]
mod tests;
