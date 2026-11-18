//! Pont borné entre E/S async et calcul DFN3 CPU sur un worker par flux admis.
use super::pcm::{Config, Pcm};
use super::*;
use std::future::Future;
use tokio::sync::mpsc;

pub(super) fn error(error: impl std::fmt::Display) -> ServeError {
    ServeError::args(format!("transport enhance: {error}"))
}
/// Passe à `true` quand un relais SIGUSR2 a un successeur prêt.
pub(super) type Drain = tokio::sync::watch::Receiver<bool>;
/// Se résout à la demande de vidage ; jamais si l'émetteur a disparu.
pub(super) async fn drained(drain: &mut Drain) {
    if drain.wait_for(|d| *d).await.is_err() {
        std::future::pending::<()>().await;
    }
}
/// Vrai une fois le successeur prêt : ce processus vide ses transports.
pub(super) fn draining() -> bool {
    #[cfg(unix)]
    return super::super::relay::draining();
    #[cfg(not(unix))]
    false
}
/// Vrai une fois ce processus annoncé prêt au parent (toujours hors Unix).
pub(super) fn announced() -> bool {
    #[cfg(unix)]
    return super::super::relay::announced();
    #[cfg(not(unix))]
    true
}
/// Refuse une socket héritée qui n'écoute pas sur l'adresse configurée ; un
/// port 0 configuré accepte le port réel attribué au premier lancement.
pub(super) fn same_addr(
    local: std::io::Result<std::net::SocketAddr>,
    addr: std::net::SocketAddr,
) -> ServeResult<()> {
    let local = local.map_err(error)?;
    if local == addr || (addr.port() == 0 && local.ip() == addr.ip()) {
        Ok(())
    } else {
        Err(error(format!(
            "socket héritée sur {local}, attendue sur {addr}"
        )))
    }
}
/// Certificat auto-signé et API locale : seuls les listeners loopback sont admis.
pub(super) fn require_loopback(addr: std::net::SocketAddr, name: &str) -> ServeResult<()> {
    if addr.ip().is_loopback() {
        Ok(())
    } else {
        Err(error(format!(
            "{name}={addr} refusée : l'écoute n'est admise que sur le loopback (127.0.0.0/8 ou [::1])"
        )))
    }
}
pub(super) async fn deadline<T, E: std::fmt::Display>(
    future: impl Future<Output = Result<T, E>>,
) -> ServeResult<T> {
    tokio::time::timeout(enhance_idle_timeout(), future)
        .await
        .map_err(error)?
        .map_err(error)
}
pub(super) struct Worker {
    _slot: Arc<Reservation>,
    input: mpsc::Sender<Option<Vec<u8>>>,
    output: mpsc::Receiver<ServeResult<Vec<u8>>>,
}
impl Worker {
    pub fn new(model: Arc<DeepFilterNet3>, config: Config, slot: Reservation) -> Self {
        let (input, mut receiver) = mpsc::channel::<Option<Vec<u8>>>(1);
        let (sender, output) = mpsc::channel(1);
        let slot = Arc::new(slot);
        let worker_slot = Arc::clone(&slot);
        tokio::task::spawn_blocking(move || {
            let _slot = worker_slot;
            let mut pcm = match Pcm::new(&model, config) {
                Ok(pcm) => pcm,
                Err(e) => {
                    let _ = sender.blocking_send(Err(e));
                    return;
                }
            };
            while let Some(bytes) = receiver.blocking_recv() {
                let finished = bytes.is_none();
                let result = match bytes {
                    Some(bytes) => pcm.push(&bytes),
                    None => pcm.finish(),
                };
                let failed = result.is_err();
                if sender.blocking_send(result).is_err() || failed || finished {
                    break;
                }
            }
        });
        Self {
            input,
            output,
            _slot: slot,
        }
    }
    pub async fn process(&mut self, bytes: Option<Vec<u8>>) -> ServeResult<Vec<u8>> {
        deadline(self.input.send(bytes)).await?;
        tokio::time::timeout(enhance_idle_timeout(), self.output.recv())
            .await
            .map_err(error)?
            .ok_or_else(|| error("worker interrompu"))?
    }
}

/// Le runtime et ses listeners sont détruits lorsque le serveur principal sort.
pub(in crate::serve) struct Runtime {
    stop: Option<tokio::sync::oneshot::Sender<()>>,
    thread: Option<std::thread::JoinHandle<()>>,
}
impl Drop for Runtime {
    fn drop(&mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}
pub(in crate::serve) fn start(
    front: Arc<super::super::dispatch::Front>,
    connections: Arc<AtomicUsize>,
) -> ServeResult<Runtime> {
    let (stop, stopped) = tokio::sync::oneshot::channel();
    let (ready, startup) = std::sync::mpsc::sync_channel(1);
    let thread = std::thread::Builder::new()
        .name("saragossa-enhance-io".into())
        .spawn(move || {
            let runtime = match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                Ok(runtime) => runtime,
                Err(e) => {
                    let _ = ready.send(Err(error(e)));
                    return;
                }
            };
            runtime.block_on(async move {
                let (drain_sender, drain) = tokio::sync::watch::channel(false);
                tokio::spawn(async move {
                    while !draining() {
                        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                    }
                    let _ = drain_sender.send(true);
                });
                let result: ServeResult<Vec<tokio::task::JoinHandle<()>>> = async {
                    let mut listeners = Vec::new();
                    #[cfg(feature = "h2")]
                    if let Some(addr) = saragossa::runtime_flags::enhance_h2_addr() {
                        if front.api_key.is_none() {
                            return Err(error("h2 requiert SARAGOSSA_API_KEY"));
                        }
                        let parsed: std::net::SocketAddr = addr.parse().map_err(error)?;
                        super::async_io::require_loopback(parsed, "SARAGOSSA_ENHANCE_H2_ADDR")?;
                        #[cfg(unix)]
                        let inherited =
                            super::super::relay::inherited_named("h2", libc::SOCK_STREAM)?
                                .map(std::net::TcpListener::from);
                        #[cfg(not(unix))]
                        let inherited: Option<std::net::TcpListener> = None;
                        let listener = match inherited {
                            Some(listener) => {
                                same_addr(listener.local_addr(), parsed)?;
                                listener.set_nonblocking(true).map_err(error)?;
                                tokio::net::TcpListener::from_std(listener).map_err(error)?
                            }
                            None => tokio::net::TcpListener::bind(parsed).await.map_err(error)?,
                        };
                        #[cfg(unix)]
                        super::super::relay::share(
                            "h2",
                            std::os::fd::AsRawFd::as_raw_fd(&listener),
                        );
                        eprintln!("enhance h2c: {}", listener.local_addr().map_err(error)?);
                        listeners.push(tokio::spawn(super::h2::listen(
                            listener,
                            Arc::clone(&front),
                            Arc::clone(&connections),
                            drain.clone(),
                        )));
                    }
                    #[cfg(feature = "webtransport")]
                    if let Some(addr) = saragossa::runtime_flags::enhance_webtransport_addr() {
                        if front.api_key.is_none() {
                            return Err(error("WebTransport requiert SARAGOSSA_API_KEY"));
                        }
                        let parsed: std::net::SocketAddr = addr.parse().map_err(error)?;
                        require_loopback(parsed, "SARAGOSSA_ENHANCE_WEBTRANSPORT_ADDR")?;
                        #[cfg(unix)]
                        let inherited =
                            super::super::relay::inherited_named("wt", libc::SOCK_DGRAM)?
                                .map(std::net::UdpSocket::from);
                        #[cfg(not(unix))]
                        let inherited: Option<std::net::UdpSocket> = None;
                        let adopted = inherited.is_some();
                        let socket = match inherited {
                            Some(socket) => {
                                same_addr(socket.local_addr(), parsed)?;
                                socket
                            }
                            None => std::net::UdpSocket::bind(parsed).map_err(error)?,
                        };
                        #[cfg(unix)]
                        super::super::relay::share("wt", std::os::fd::AsRawFd::as_raw_fd(&socket));
                        let endpoint = super::webtransport::bind(socket, &front, adopted)?;
                        eprintln!(
                            "enhance WebTransport: {}",
                            endpoint.local_addr().map_err(error)?
                        );
                        listeners.push(tokio::spawn(super::webtransport::listen(
                            endpoint,
                            Arc::clone(&front),
                            Arc::clone(&connections),
                            drain.clone(),
                        )));
                    }
                    Ok(listeners)
                }
                .await;
                let listeners = match result {
                    Ok(listeners) => {
                        let _ = ready.send(Ok(()));
                        listeners
                    }
                    Err(e) => {
                        let _ = ready.send(Err(e));
                        return;
                    }
                };
                let _ = stopped.await;
                // NOTE: en relais, connexions h2 et fermetures QUIC finissent
                // avant l'arrêt ; le délai de vidage du parent borne l'attente.
                if draining() {
                    for listener in listeners {
                        let _ = listener.await;
                    }
                }
            });
            // Les canaux des sessions sont lâchés lors de l'annulation des tâches.
            runtime.shutdown_timeout(std::time::Duration::from_secs(1));
        })
        .map_err(|e| ServeError::io("thread transports", e))?;
    let guard = Runtime {
        stop: Some(stop),
        thread: Some(thread),
    };
    startup.recv().map_err(error)??;
    Ok(guard)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn inherited_address_matches_configured_port_zero() {
        let real = "127.0.0.1:4433".parse().unwrap();
        assert!(same_addr(Ok(real), real).is_ok());
        assert!(same_addr(Ok(real), "127.0.0.1:0".parse().unwrap()).is_ok());
        assert!(same_addr(Ok(real), "127.0.0.2:0".parse().unwrap()).is_err());
        assert!(same_addr(Ok(real), "127.0.0.1:4434".parse().unwrap()).is_err());
    }
    #[test]
    fn only_loopback_addresses_are_accepted() {
        for ok in ["127.0.0.1:4433", "127.0.0.2:0", "[::1]:4433"] {
            assert!(require_loopback(ok.parse().unwrap(), "X").is_ok(), "{ok}");
        }
        for refused in ["0.0.0.0:4433", "192.168.1.10:4433", "[::]:4433"] {
            assert!(
                require_loopback(refused.parse().unwrap(), "X").is_err(),
                "{refused}"
            );
        }
    }
}
