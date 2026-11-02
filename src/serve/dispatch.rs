//! Concurrence d'E/S de `saragossa serve`.
//!
//! Un thread d'accept confie chaque connexion à un thread de connexion qui lit
//! la requête et sert les routes statiques (`/health`, `/v1/models`). Le reste
//! part dans une file bornée vers le thread moteur, seul propriétaire de
//! [`ServeState`] : l'inférence reste sérielle, seules les E/S sont parallèles.

use std::io::{self, Read, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{sync_channel, SyncSender, TrySendError};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use saragossa::runtime_flags::{serve_max_connections, serve_queue_depth};

use super::error::{ServeError, ServeResult};
use super::http::{
    handle_job, read_or_reject, reply_on_error, route_front, send_error, HttpRequest,
};
use super::state::ServeState;

/// Contexte figé au démarrage, lu par les threads de connexion sans toucher à
/// l'état moteur.
pub(super) struct Front {
    pub(super) api_key: Option<String>,
    pub(super) model_ids: Vec<String>,
}

impl Front {
    pub(super) fn new(api_key: Option<String>, model_ids: Vec<String>) -> Self {
        Self { api_key, model_ids }
    }
}

/// Requête lue, en attente du thread moteur avec la socket de sa réponse.
struct Job<S> {
    request: HttpRequest,
    stream: S,
}

type Configure<S> = fn(&S, Duration) -> ServeResult<()>;

/// Distribue les connexions acceptées vers des threads de connexion plafonnés.
pub(super) struct Dispatcher<S> {
    front: Arc<Front>,
    jobs: SyncSender<Job<S>>,
    active: Arc<AtomicUsize>,
    max_connections: usize,
    read_timeout: Duration,
    configure: Configure<S>,
}

impl<S: Read + Write + Send + 'static> Dispatcher<S> {
    /// Accepte jusqu'à l'erreur du listener, fatale comme avant.
    pub(super) fn accept_all(
        self,
        incoming: impl Iterator<Item = io::Result<S>>,
        context: &'static str,
    ) -> ServeResult<()> {
        for stream in incoming {
            self.dispatch(stream.map_err(|e| ServeError::io(context, e))?);
        }
        Ok(())
    }

    /// Ne bloque jamais l'accept : au-delà du plafond, 503 immédiat.
    fn dispatch(&self, mut stream: S) {
        if let Err(error) = (self.configure)(&stream, self.read_timeout) {
            eprintln!("saragossa serve request error: {error}");
            return;
        }
        let Some(slot) = ConnectionSlot::acquire(&self.active, self.max_connections) else {
            // NOTE: la requête n'est pas lue ; un client TCP qui a déjà écrit
            // peut recevoir un RST au lieu du 503. Acceptable en surcharge.
            let _ = send_error(&mut stream, 503, "trop de connexions simultanées");
            return;
        };
        let front = Arc::clone(&self.front);
        let jobs = self.jobs.clone();
        let deadline = Instant::now() + self.read_timeout;
        let spawned = thread::Builder::new()
            .name("saragossa-serve-conn".to_string())
            .spawn(move || {
                let _slot = slot;
                if let Err(error) = handle_front(stream, &front, &jobs, deadline) {
                    eprintln!("saragossa serve request error: {error}");
                }
            });
        // NOTE: un échec de spawn rend la fermeture, donc le slot et la socket.
        if let Err(error) = spawned {
            eprintln!("saragossa serve request error: thread de connexion: {error}");
        }
    }
}

/// Sert les connexions : accept dans un thread dédié, inférence sur le thread
/// appelant, qui doit être celui qui a construit `state`.
pub(super) fn serve<S, A>(
    state: &mut ServeState,
    front: Front,
    read_timeout: Duration,
    configure: Configure<S>,
    accept: A,
) -> ServeResult<()>
where
    S: Read + Write + Send + 'static,
    A: FnOnce(Dispatcher<S>) -> ServeResult<()> + Send + 'static,
{
    let (jobs, queue) = sync_channel(serve_queue_depth());
    let dispatcher = Dispatcher {
        front: Arc::new(front),
        jobs,
        active: Arc::new(AtomicUsize::new(0)),
        max_connections: serve_max_connections(),
        read_timeout,
        configure,
    };
    // NOTE: thread détaché plutôt que scope : une panique moteur doit tuer le
    // process comme avant, pas attendre un accept qui ne finit jamais.
    let acceptor = thread::Builder::new()
        .name("saragossa-serve-accept".to_string())
        .spawn(move || accept(dispatcher))
        .map_err(|e| ServeError::io("thread d'accept", e))?;
    // La boucle s'arrête quand l'accept et toutes les connexions ont lâché
    // leur émetteur.
    for Job {
        request,
        mut stream,
    } in queue
    {
        if let Err(error) = handle_job(&mut stream, state, request) {
            eprintln!("saragossa serve request error: {error}");
        }
    }
    acceptor
        .join()
        .map_err(|_| ServeError::io("thread d'accept", io::Error::other("panique")))?
}

/// Lit la requête, sert les routes statiques, sinon met en file sans bloquer.
fn handle_front<S: Read + Write>(
    mut stream: S,
    front: &Front,
    jobs: &SyncSender<Job<S>>,
    deadline: Instant,
) -> ServeResult<()> {
    let Some(request) = read_or_reject(&mut stream, deadline)? else {
        return Ok(());
    };
    if let Some(result) = route_front(&mut stream, front, &request) {
        return reply_on_error(&mut stream, result);
    }
    match jobs.try_send(Job { request, stream }) {
        Ok(()) => Ok(()),
        Err(TrySendError::Full(mut job)) => {
            send_error(&mut job.stream, 503, "file d'inférence pleine, réessayer")
        }
        Err(TrySendError::Disconnected(mut job)) => {
            send_error(&mut job.stream, 503, "moteur d'inférence arrêté")
        }
    }
}

/// Place réservée dans le plafond de connexions, rendue au drop.
struct ConnectionSlot(Arc<AtomicUsize>);

impl ConnectionSlot {
    fn acquire(active: &Arc<AtomicUsize>, max: usize) -> Option<Self> {
        active
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |count| {
                (count < max).then_some(count + 1)
            })
            .ok()
            .map(|_| Self(Arc::clone(active)))
    }
}

impl Drop for ConnectionSlot {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

#[cfg(all(test, unix))]
mod tests {
    use std::os::unix::net::UnixStream;
    use std::sync::mpsc::Receiver;

    use super::super::args::ServeArgs;
    use super::super::http::configure_unix_stream;
    use super::*;

    const CHAT: &[u8] = b"POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 2\r\n\r\n{}";
    const HEALTH: &[u8] = b"GET /health HTTP/1.1\r\n\r\n";

    /// Dispatcher de test ; la file n'est jamais vidée, comme un moteur
    /// occupé par une génération longue.
    fn dispatcher(
        max_connections: usize,
        queue: usize,
    ) -> (Dispatcher<UnixStream>, Receiver<Job<UnixStream>>) {
        let (jobs, rx) = sync_channel(queue);
        let dispatcher = Dispatcher {
            front: Arc::new(Front::new(None, vec!["alpha".to_string()])),
            jobs,
            active: Arc::new(AtomicUsize::new(0)),
            max_connections,
            read_timeout: Duration::from_secs(5),
            configure: configure_unix_stream,
        };
        (dispatcher, rx)
    }

    /// Ouvre une connexion servie par `dispatcher`, envoie `raw` s'il y en a.
    fn connect(dispatcher: &Dispatcher<UnixStream>, raw: &[u8]) -> UnixStream {
        let (mut client, server) = UnixStream::pair().expect("invariant: paire Unix locale");
        client
            .set_read_timeout(Some(Duration::from_secs(5)))
            .expect("invariant: timeout client");
        client.write_all(raw).expect("invariant: écriture client");
        dispatcher.dispatch(server);
        client
    }

    fn response(mut client: UnixStream) -> String {
        let mut response = String::new();
        client
            .read_to_string(&mut response)
            .expect("invariant: réponse lisible avant le timeout");
        response
    }

    #[test]
    fn health_answers_while_engine_busy_and_client_slow() {
        let (dispatcher, rx) = dispatcher(4, 1);
        let _queued = connect(&dispatcher, CHAT);
        let _slow = connect(&dispatcher, b"GET /hea");

        let health = response(connect(&dispatcher, HEALTH));

        assert!(health.starts_with("HTTP/1.1 200 OK"));
        assert!(health.contains(r#""status":"ok""#));
        let job = rx.recv().expect("invariant: chat en file moteur");
        assert_eq!(job.request.route(), "/v1/chat/completions");
    }

    #[test]
    fn models_served_by_connection_thread() {
        let (dispatcher, _rx) = dispatcher(4, 1);

        let models = response(connect(&dispatcher, b"GET /v1/models HTTP/1.1\r\n\r\n"));

        assert!(models.starts_with("HTTP/1.1 200 OK"));
        assert!(models.contains(r#""id":"alpha""#));
    }

    #[test]
    fn full_queue_returns_503() {
        let (dispatcher, _rx) = dispatcher(4, 1);
        let first = connect(&dispatcher, CHAT);
        // La file est pleine une fois le premier job déposé.
        while dispatcher.active.load(Ordering::Acquire) > 0 {
            thread::yield_now();
        }

        let second = response(connect(&dispatcher, CHAT));

        assert!(second.starts_with("HTTP/1.1 503 Service Unavailable"));
        assert!(second.contains("file d'inférence pleine"));
        drop(first);
    }

    #[test]
    fn connection_cap_returns_503_without_reading() {
        let (dispatcher, _rx) = dispatcher(1, 1);
        let _slow = connect(&dispatcher, b"");

        let rejected = response(connect(&dispatcher, b""));

        assert!(rejected.starts_with("HTTP/1.1 503 Service Unavailable"));
        assert!(rejected.contains("trop de connexions"));
    }

    #[test]
    fn engine_runs_jobs_on_calling_thread() {
        let args = ServeArgs::parse(["--model".to_string(), "alpha=/nonexistent/a".to_string()])
            .expect("invariant: args valides");
        let mut state = ServeState::new(&args);
        let (mut client, server) = UnixStream::pair().expect("invariant: paire Unix locale");
        // Le chargement du modèle échoue, mais passe d'abord par la garde.
        let body = br#"{"model":"alpha","messages":[]}"#;
        let raw = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nContent-Length: {}\r\n\r\n",
            body.len()
        );
        client
            .write_all(&[raw.as_bytes(), body].concat())
            .expect("invariant: écriture client");
        let front = Front::new(None, state.model_ids());

        // L'accept rend la main après une connexion : la boucle moteur traite
        // le job sur ce thread (la garde debug de ServeState le vérifie) puis
        // s'arrête quand plus aucun émetteur ne vit.
        serve(
            &mut state,
            front,
            Duration::from_secs(5),
            configure_unix_stream,
            move |dispatcher| dispatcher.accept_all(std::iter::once(Ok(server)), "test"),
        )
        .expect("invariant: arrêt propre quand l'accept se termine");

        let mut response = String::new();
        client
            .read_to_string(&mut response)
            .expect("invariant: réponse moteur");
        assert!(response.starts_with("HTTP/1.1 "));
        assert!(!response.starts_with("HTTP/1.1 200"));
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "thread moteur")]
    fn decoder_call_off_engine_thread_is_caught() {
        let args = ServeArgs::parse(["--model".to_string(), "alpha=/nonexistent/a".to_string()])
            .expect("invariant: args valides");
        let state = ServeState::new(&args);

        let outcome = thread::spawn(move || {
            let mut state = state;
            let _ = state.preload();
        })
        .join();

        if let Err(panic) = outcome {
            std::panic::resume_unwind(panic);
        }
    }
}
