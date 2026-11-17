//! Une session HTTP/3 WebTransport par flux DFN3, un stream bidirectionnel fiable.
use super::super::dispatch::Front;
use super::async_io::{deadline, error, Worker};
use super::pcm::{Config, MAX_FRAME};
use super::*;
use std::net::SocketAddr;
use web_transport_quinn::{RecvStream, SendStream};

pub(super) fn bind(addr: SocketAddr, front: &Front) -> ServeResult<quinn::Endpoint> {
    let identity = super::certificate::identity(addr)?;
    let endpoint = quinn::Endpoint::server(identity.config, addr).map_err(error)?;
    let mut info = identity.info;
    // Le port 0 est utile aux tests et aux démarrages avec allocation automatique.
    if addr.port() == 0 {
        info.url = info.url.replace(
            ":0/",
            &format!(":{}/", endpoint.local_addr().map_err(error)?.port()),
        );
    }
    *front
        .enhance
        .certificate
        .lock()
        .map_err(|_| error("verrou certificat"))? = Some(info);
    Ok(endpoint)
}
pub(super) async fn listen(
    endpoint: quinn::Endpoint,
    front: Arc<Front>,
    connections: Arc<AtomicUsize>,
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
}
pub(super) async fn session(conn: quinn::Connection, front: Arc<Front>) -> ServeResult<()> {
    let request = deadline(web_transport_quinn::Request::accept(conn)).await?;
    if request.url.path() != "/v1/audio/enhance/wt" {
        return reject(request, http::StatusCode::NOT_FOUND).await;
    }
    let mut token = None;
    let mut query = Vec::new();
    for (key, value) in request.url.query_pairs() {
        if key == "token" && token.is_none() {
            token = Some(value.into_owned());
        } else {
            query.push(format!("{key}={value}"));
        }
    }
    if !front.api_key.as_deref().is_some_and(|expected| {
        token.as_deref().is_some_and(|token| {
            super::super::http::constant_time_eq(token.as_bytes(), expected.as_bytes())
        })
    }) {
        return reject(request, http::StatusCode::UNAUTHORIZED).await;
    }
    let config = match Config::parse(&query.join("&")) {
        Ok(config) => config,
        Err(_) => return reject(request, http::StatusCode::BAD_REQUEST).await,
    };
    let Some(slot) = Reservation::acquire(&front.enhance.active, 1, enhance_max_streams()) else {
        return reject(request, http::StatusCode::SERVICE_UNAVAILABLE).await;
    };
    let loader = Arc::clone(&front);
    let model = match deadline(tokio::task::spawn_blocking(move || loader.enhance.model())).await? {
        Ok(model) => model,
        Err(_) => return reject(request, http::StatusCode::BAD_REQUEST).await,
    };
    let mut worker = Worker::new(model, config, slot);
    let session = deadline(request.ok()).await?;
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
