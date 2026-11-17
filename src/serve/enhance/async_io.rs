//! Pont borné entre E/S async et calcul DFN3 CPU sur un worker par flux admis.
use super::pcm::{Config, Pcm};
use super::*;
use std::future::Future;
use tokio::sync::mpsc;

pub(super) fn error(error: impl std::fmt::Display) -> ServeError {
    ServeError::args(format!("transport enhance: {error}"))
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
                let result: ServeResult<()> = async {
                    #[cfg(feature = "h2")]
                    if let Some(addr) = saragossa::runtime_flags::enhance_h2_addr() {
                        if front.api_key.is_none() {
                            return Err(error("h2 requiert SARAGOSSA_API_KEY"));
                        }
                        let listener = tokio::net::TcpListener::bind(&addr).await.map_err(error)?;
                        eprintln!("enhance h2c: {}", listener.local_addr().map_err(error)?);
                        tokio::spawn(super::h2::listen(
                            listener,
                            Arc::clone(&front),
                            Arc::clone(&connections),
                        ));
                    }
                    #[cfg(feature = "webtransport")]
                    if let Some(addr) = saragossa::runtime_flags::enhance_webtransport_addr() {
                        if front.api_key.is_none() {
                            return Err(error("WebTransport requiert SARAGOSSA_API_KEY"));
                        }
                        let endpoint =
                            super::webtransport::bind(addr.parse().map_err(error)?, &front)?;
                        eprintln!(
                            "enhance WebTransport: {}",
                            endpoint.local_addr().map_err(error)?
                        );
                        tokio::spawn(super::webtransport::listen(
                            endpoint,
                            Arc::clone(&front),
                            Arc::clone(&connections),
                        ));
                    }
                    Ok(())
                }
                .await;
                let failed = result.is_err();
                let _ = ready.send(result);
                if !failed {
                    let _ = stopped.await;
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
