//! Une session HTTP/3 WebTransport par flux DFN3, un stream bidirectionnel fiable.
use super::super::dispatch::Front;
use super::async_io::{announced, deadline, drained, draining, error, Drain, Worker};
use super::pcm::{Config, MAX_FRAME};
use super::*;
use std::net::SocketAddr;
use std::pin::Pin;
use std::task::{Context, Poll};
use web_transport_quinn::{RecvStream, SendStream};

/// Endpoint sur `socket` déjà liée, ou héritée d'un relais (`inherited`), au
/// loopback. Une socket héritée n'est lue qu'après l'annonce « prêt » : avant,
/// les datagrammes restent au parent. Appel depuis un runtime tokio.
pub(super) fn bind(
    socket: std::net::UdpSocket,
    front: &Front,
    inherited: bool,
) -> ServeResult<quinn::Endpoint> {
    let addr = socket.local_addr().map_err(error)?;
    super::async_io::require_loopback(addr, "SARAGOSSA_ENHANCE_WEBTRANSPORT_ADDR")?;
    let identity = super::certificate::identity(addr)?;
    let runtime: Arc<dyn quinn::Runtime> = Arc::new(quinn::TokioRuntime);
    let socket = Arc::new(Relayed {
        inner: runtime.wrap_udp_socket(socket).map_err(error)?,
        gated: inherited,
        waker: std::sync::Mutex::new(None),
    });
    if inherited {
        let gate = Arc::clone(&socket);
        tokio::spawn(async move {
            while !announced() {
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
            if let Some(waker) = gate.waker.lock().ok().and_then(|mut w| w.take()) {
                waker.wake();
            }
        });
    }
    let endpoint = quinn::Endpoint::new_with_abstract_socket(
        quinn::EndpointConfig::default(),
        Some(identity.config),
        socket,
        runtime,
    )
    .map_err(error)?;
    *front
        .enhance
        .certificate
        .lock()
        .map_err(|_| error("verrou certificat"))? = Some(identity.info);
    Ok(endpoint)
}
/// Socket UDP partagée en relais. Successeur : lue seulement après son
/// annonce « prêt ». Parent : plus lue une fois le successeur prêt (les
/// datagrammes lui reviennent), mais émet encore ses CONNECTION_CLOSE.
#[derive(Debug)]
struct Relayed {
    inner: Arc<dyn quinn::AsyncUdpSocket>,
    gated: bool,
    /// Réveil du driver quand l'annonce lève la barrière.
    waker: std::sync::Mutex<Option<std::task::Waker>>,
}
impl quinn::AsyncUdpSocket for Relayed {
    fn create_io_poller(self: Arc<Self>) -> Pin<Box<dyn quinn::UdpPoller>> {
        Arc::clone(&self.inner).create_io_poller()
    }
    fn try_send(&self, transmit: &quinn::udp::Transmit) -> std::io::Result<()> {
        self.inner.try_send(transmit)
    }
    fn poll_recv(
        &self,
        cx: &mut Context,
        bufs: &mut [std::io::IoSliceMut<'_>],
        meta: &mut [quinn::udp::RecvMeta],
    ) -> Poll<std::io::Result<usize>> {
        if draining() {
            return Poll::Pending;
        }
        if self.gated && !announced() {
            if let Ok(mut waker) = self.waker.lock() {
                *waker = Some(cx.waker().clone());
            }
            // NOTE: relu après l'inscription du réveil, sinon une annonce
            // entre les deux lectures serait perdue.
            if !announced() {
                return Poll::Pending;
            }
        }
        self.inner.poll_recv(cx, bufs, meta)
    }
    fn local_addr(&self) -> std::io::Result<SocketAddr> {
        self.inner.local_addr()
    }
    fn max_transmit_segments(&self) -> usize {
        self.inner.max_transmit_segments()
    }
    fn max_receive_segments(&self) -> usize {
        self.inner.max_receive_segments()
    }
    fn may_fragment(&self) -> bool {
        self.inner.may_fragment()
    }
}
/// Accepte jusqu'au relais ; ferme alors toutes les sessions (H3_NO_ERROR),
/// qui ne migrent pas : les clients se reconnectent chez le successeur.
pub(super) async fn listen(
    endpoint: quinn::Endpoint,
    front: Arc<Front>,
    connections: Arc<AtomicUsize>,
    mut drain: Drain,
) {
    let mut tasks = tokio::task::JoinSet::new();
    let rotation = tokio::time::sleep(std::time::Duration::from_secs(12 * 86400));
    tokio::pin!(rotation);
    loop {
        tokio::select! {
            incoming = endpoint.accept() => {
                let Some(incoming) = incoming else { break; };
                let Some(slot) = Reservation::acquire(&connections, 1, saragossa::runtime_flags::serve_max_connections()) else {
                    incoming.refuse(); continue;
                };
                let front = Arc::clone(&front);
                tasks.spawn(async move {
                    let _slot = slot;
                    let result = async {
                        let conn = deadline(incoming.accept().map_err(error)?).await?;
                        session(conn, front).await
                    }.await;
                    if let Err(e) = result { eprintln!("{e}"); }
                });
            }
            _ = tasks.join_next(), if !tasks.is_empty() => {},
            _ = drained(&mut drain) => {
                endpoint.close(quinn::VarInt::from_u32(0x100), b"relais");
                break;
            }
            _ = &mut rotation => {
                let renewed = endpoint.local_addr().map_err(error).and_then(super::certificate::identity);
                match renewed {
                    Ok(identity) => {
                        if let Ok(mut info) = front.enhance.certificate.lock() {
                            endpoint.set_server_config(Some(identity.config));
                            *info = Some(identity.info);
                        }
                    }
                    Err(e) => eprintln!("renouvellement certificat: {e}"),
                }
                rotation.as_mut().reset(tokio::time::Instant::now() + std::time::Duration::from_secs(12 * 86400));
            }
        }
    }
    if draining() {
        // Laisse partir les CONNECTION_CLOSE avant l'arrêt du runtime.
        let _ = tokio::time::timeout(std::time::Duration::from_secs(2), endpoint.wait_idle()).await;
    }
}
pub(super) async fn session(conn: quinn::Connection, front: Arc<Front>) -> ServeResult<()> {
    let request = deadline(web_transport_quinn::Request::accept(conn)).await?;
    if request.url.path() != "/v1/audio/enhance/wt" {
        return reject(request, http::StatusCode::NOT_FOUND).await;
    }
    // Origin comme WS : une page tierce ne doit pas piloter l'endpoint local.
    if request
        .headers
        .get("origin")
        .is_some_and(|o| !o.to_str().is_ok_and(super::pcm::loopback_origin))
    {
        return reject(request, http::StatusCode::FORBIDDEN).await;
    }
    // Jeton : en-tête Authorization, sinon sous-protocole `bearer.<jeton>`, sinon
    // `?token=` (les navigateurs ne posent aucun en-tête sur WebTransport ; le
    // sous-protocole y est préférable). Jamais journalisé : l'URL n'est pas loguée.
    let mut query = Vec::new();
    let mut query_token = None;
    for (key, value) in request.url.query_pairs() {
        if key == "token" && query_token.is_none() {
            query_token = Some(value.into_owned());
        } else {
            query.push(format!("{key}={value}"));
        }
    }
    let header_token = request
        .headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(super::super::http::bearer_token);
    let protocol = request
        .protocols
        .iter()
        .find(|p| p.starts_with("bearer."))
        .cloned();
    let token = header_token
        .or(protocol.as_deref().and_then(|p| p.strip_prefix("bearer.")))
        .or(query_token.as_deref());
    if !front.api_key.as_deref().is_some_and(|expected| {
        token.is_some_and(|token| {
            super::super::http::constant_time_eq(token.as_bytes(), expected.as_bytes())
        })
    }) {
        return reject(request, http::StatusCode::UNAUTHORIZED).await;
    }
    let config = match Config::parse(&query.join("&")) {
        Ok(config) => config,
        Err(_) => return reject(request, http::StatusCode::BAD_REQUEST).await,
    };
    let Some(slot) = front.enhance.reserve() else {
        return reject(request, http::StatusCode::SERVICE_UNAVAILABLE).await;
    };
    let loader = Arc::clone(&front);
    let model = match deadline(tokio::task::spawn_blocking(move || loader.enhance.model())).await? {
        Ok(model) => model,
        Err(e) => {
            let status = http::StatusCode::from_u16(super::super::http::error_status(&e))
                .unwrap_or(http::StatusCode::INTERNAL_SERVER_ERROR);
            return reject(request, status).await;
        }
    };
    let mut worker = Worker::new(model, config, slot);
    let response = match protocol {
        Some(p) => web_transport_quinn::proto::ConnectResponse::OK.with_protocol(p),
        None => web_transport_quinn::proto::ConnectResponse::OK,
    };
    let session = deadline(request.respond(response)).await?;
    let result = async {
        let (mut send, mut recv) = deadline(session.accept_bi()).await?;
        let result = pump(&mut send, &mut recv, &mut worker).await;
        if result.is_err() {
            let _ = send.reset(1);
            let _ = recv.stop(1);
        }
        result
    }
    .await;
    session.close(if result.is_ok() { 0 } else { 1 }, b"fin du flux");
    result
}
/// Laisse le client lire le statut avant de fermer la connexion QUIC.
async fn reject(
    request: web_transport_quinn::Request,
    status: http::StatusCode,
) -> ServeResult<()> {
    let conn = request.conn().clone();
    deadline(request.reject(status)).await?;
    // Un client qui ne ferme pas est libéré par le délai d'inactivité.
    let _ = tokio::time::timeout(std::time::Duration::from_secs(2), conn.closed()).await;
    Ok(())
}
async fn pump(
    send: &mut SendStream,
    recv: &mut RecvStream,
    worker: &mut Worker,
) -> ServeResult<()> {
    loop {
        let mut length = [0; 4];
        deadline(recv.read_exact(&mut length)).await?;
        let length = u32::from_le_bytes(length) as usize;
        if length > MAX_FRAME {
            return Err(error("trame WebTransport trop grande"));
        }
        if length == 0 {
            break;
        }
        let mut bytes = vec![0; length];
        deadline(recv.read_exact(&mut bytes)).await?;
        let output = worker.process(Some(bytes)).await?;
        if !output.is_empty() {
            write_frame(send, &output).await?;
        }
    }
    // FIN est obligatoire après le terminateur : pas de suffixe audio ignoré.
    if deadline(recv.read(&mut [0; 1])).await?.is_some() {
        return Err(error("données après fin PCM"));
    }
    let output = worker.process(None).await?;
    if !output.is_empty() {
        write_frame(send, &output).await?;
    }
    write_frame(send, &[]).await?;
    send.finish().map_err(error)?;
    deadline(send.stopped()).await?;
    Ok(())
}
pub(super) async fn write_frame(send: &mut SendStream, bytes: &[u8]) -> ServeResult<()> {
    // Le rééchantillonnage 8 -> 48 kHz peut agrandir la réponse ; chaque trame
    // réseau reste bornée, indépendamment de la taille du vecteur du worker.
    if bytes.is_empty() {
        return deadline(send.write_all(&0_u32.to_le_bytes())).await;
    }
    for chunk in bytes.chunks(MAX_FRAME) {
        deadline(send.write_all(&(chunk.len() as u32).to_le_bytes())).await?;
        deadline(send.write_all(chunk)).await?;
    }
    Ok(())
}
#[cfg(test)]
mod tests;
