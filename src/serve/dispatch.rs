//! Concurrence d'E/S de `saragossa serve`.
//!
//! Un thread d'accept confie chaque connexion à un thread de connexion qui lit
//! la requête et sert les routes statiques (`/health`, `/v1/models`). Le reste
//! part dans une file bornée vers le thread moteur, seul propriétaire de
//! [`ServeState`] : l'inférence reste sérielle, seules les E/S sont parallèles.

use std::io::{self, Read, Write};
use std::net::TcpStream;
#[cfg(unix)]
use std::os::unix::net::UnixStream;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{sync_channel, SyncSender, TrySendError};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use saragossa::runtime_flags::{
    serve_max_connections, serve_max_inflight_bytes, serve_queue_depth,
};

use super::error::{ServeError, ServeResult};
#[cfg(unix)]
use super::http::configure_unix_stream;
use super::http::{
    configure_tcp_stream, handle_job, read_body, read_head, read_or_reject, reply_on_error,
    route_front, send_error, send_route_error, HttpRequest,
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

/// Socket servie : TCP ou Unix.
pub(super) trait ServeStream: Read + Write + Send + 'static {
    /// Applique les timeouts de lecture et d'écriture.
    fn configure(&self, read_timeout: Duration) -> ServeResult<()>;
    /// Bascule le mode non bloquant (sonde de client parti).
    fn set_nonblocking(&self, nonblocking: bool) -> io::Result<()>;
}

impl ServeStream for TcpStream {
    fn configure(&self, read_timeout: Duration) -> ServeResult<()> {
        configure_tcp_stream(self, read_timeout)
    }

    fn set_nonblocking(&self, nonblocking: bool) -> io::Result<()> {
        TcpStream::set_nonblocking(self, nonblocking)
    }
}

#[cfg(unix)]
impl ServeStream for UnixStream {
    fn configure(&self, read_timeout: Duration) -> ServeResult<()> {
        configure_unix_stream(self, read_timeout)
    }

    fn set_nonblocking(&self, nonblocking: bool) -> io::Result<()> {
        UnixStream::set_nonblocking(self, nonblocking)
    }
}

/// Requête lue, en attente du thread moteur avec la socket de sa réponse. La
/// réservation d'octets suit le corps jusqu'à la fin du traitement.
struct Job<S> {
    request: HttpRequest,
    stream: S,
    _bytes: Reservation,
}

/// Plafonds de la façade d'E/S.
struct Limits {
    max_connections: usize,
    max_inflight_bytes: usize,
    read_timeout: Duration,
}

/// Distribue les connexions acceptées vers des threads de connexion plafonnés.
pub(super) struct Dispatcher<S> {
    front: Arc<Front>,
    jobs: SyncSender<Job<S>>,
    connections: Arc<AtomicUsize>,
    inflight_bytes: Arc<AtomicUsize>,
    limits: Arc<Limits>,
}

impl<S: ServeStream> Dispatcher<S> {
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
        if let Err(error) = stream.configure(self.limits.read_timeout) {
            eprintln!("saragossa serve request error: {error}");
            return;
        }
        let Some(slot) = Reservation::acquire(&self.connections, 1, self.limits.max_connections)
        else {
            // NOTE: la requête n'est pas lue, sa route est inconnue : erreur au
            // format OpenAI. Un client TCP qui a déjà écrit peut recevoir un
            // RST au lieu du 503, acceptable en surcharge.
            let _ = send_error(&mut stream, 503, "trop de connexions simultanées");
            return;
        };
        let connection = Connection {
            front: Arc::clone(&self.front),
            jobs: self.jobs.clone(),
            inflight_bytes: Arc::clone(&self.inflight_bytes),
            limits: Arc::clone(&self.limits),
        };
        let spawned = thread::Builder::new()
            .name("saragossa-serve-conn".to_string())
            .spawn(move || {
                // NOTE: rendu au drop, y compris si ce thread panique ; la
                // panique ne coûte que cette connexion.
                let _slot = slot;
                if let Err(error) = connection.handle(stream) {
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
    accept: A,
) -> ServeResult<()>
where
    S: ServeStream,
    A: FnOnce(Dispatcher<S>) -> ServeResult<()> + Send + 'static,
{
    let (jobs, queue) = sync_channel(serve_queue_depth());
    let dispatcher = Dispatcher {
        front: Arc::new(front),
        jobs,
        connections: Arc::new(AtomicUsize::new(0)),
        inflight_bytes: Arc::new(AtomicUsize::new(0)),
        limits: Arc::new(Limits {
            max_connections: serve_max_connections(),
            max_inflight_bytes: serve_max_inflight_bytes(),
            read_timeout,
        }),
    };
    // NOTE: thread détaché plutôt que scope : une panique moteur doit tuer le
    // process comme avant, pas attendre un accept qui ne finit jamais.
    let acceptor = thread::Builder::new()
        .name("saragossa-serve-accept".to_string())
        .spawn(move || accept(dispatcher))
        .map_err(|e| ServeError::io("thread d'accept", e))?;
    // La boucle s'arrête quand l'accept et toutes les connexions ont lâché
    // leur émetteur.
    for job in queue {
        run_job(job, state);
    }
    acceptor
        .join()
        .map_err(|_| ServeError::io("thread d'accept", io::Error::other("panique")))?
}

/// Exécute un job sur le thread moteur ; `false` si son client est parti.
fn run_job<S: ServeStream>(mut job: Job<S>, state: &mut ServeState) -> bool {
    if peer_gone(&mut job.stream) {
        eprintln!(
            "saragossa serve requête abandonnée par le client, ignorée: {} {}",
            job.request.method,
            job.request.route()
        );
        return false;
    }
    if let Err(error) = handle_job(&mut job.stream, state, job.request) {
        eprintln!("saragossa serve request error: {error}");
    }
    true
}

/// Le client a-t-il fermé sa socket pendant l'attente en file ? La requête
/// est déjà lue en entier : un octet de plus serait du pipelining, ignoré
/// avec `Connection: close`, donc le consommer est sans conséquence.
fn peer_gone<S: ServeStream>(stream: &mut S) -> bool {
    if stream.set_nonblocking(true).is_err() {
        return false;
    }
    let gone = match stream.read(&mut [0_u8; 1]) {
        Ok(read) => read == 0,
        Err(error) => !matches!(
            error.kind(),
            io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
        ),
    };
    // NOTE: sans retour au mode bloquant, l'écriture de la réponse
    // échouerait en WouldBlock dès que le tampon socket est plein.
    if stream.set_nonblocking(false).is_err() {
        return true;
    }
    gone
}

/// Contexte d'un thread de connexion.
struct Connection<S> {
    front: Arc<Front>,
    jobs: SyncSender<Job<S>>,
    inflight_bytes: Arc<AtomicUsize>,
    limits: Arc<Limits>,
}

impl<S: ServeStream> Connection<S> {
    /// Lit les headers, sert les routes statiques et le refus bearer avant de
    /// lire le corps, puis met en file sans bloquer.
    fn handle(&self, mut stream: S) -> ServeResult<()> {
        let deadline = Instant::now() + self.limits.read_timeout;
        let Some((mut request, content_length)) =
            read_or_reject(&mut stream, |s| read_head(s, deadline))?
        else {
            return Ok(());
        };
        if let Some(result) = route_front(&mut stream, &self.front, &request) {
            return reply_on_error(&mut stream, result);
        }
        let Some(bytes) = Reservation::acquire(
            &self.inflight_bytes,
            content_length,
            self.limits.max_inflight_bytes,
        ) else {
            return send_route_error(
                &mut stream,
                &request,
                503,
                "budget mémoire des requêtes en cours épuisé, réessayer",
            );
        };
        read_or_reject(&mut stream, |s| {
            read_body(s, &mut request, content_length, deadline)
        })?;
        let job = Job {
            request,
            stream,
            _bytes: bytes,
        };
        match self.jobs.try_send(job) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(mut job)) => send_route_error(
                &mut job.stream,
                &job.request,
                503,
                "file d'inférence pleine, réessayer",
            ),
            Err(TrySendError::Disconnected(mut job)) => send_route_error(
                &mut job.stream,
                &job.request,
                503,
                "moteur d'inférence arrêté",
            ),
        }
    }
}

/// Part réservée d'un compteur plafonné (connexions, octets), rendue au drop.
struct Reservation {
    counter: Arc<AtomicUsize>,
    amount: usize,
}

impl Reservation {
    fn acquire(counter: &Arc<AtomicUsize>, amount: usize, max: usize) -> Option<Self> {
        counter
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |count| {
                count.checked_add(amount).filter(|total| *total <= max)
            })
            .ok()
            .map(|_| Self {
                counter: Arc::clone(counter),
                amount,
            })
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        self.counter.fetch_sub(self.amount, Ordering::AcqRel);
    }
}

#[cfg(all(test, unix))]
mod tests {
    use std::os::unix::net::UnixStream;
    use std::sync::mpsc::Receiver;

    use super::super::args::ServeArgs;
    use super::*;

    const CHAT: &[u8] = b"POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 2\r\n\r\n{}";
    const HEALTH: &[u8] = b"GET /health HTTP/1.1\r\n\r\n";

    /// Dispatcher de test ; la file n'est jamais vidée, comme un moteur
    /// occupé par une génération longue.
    fn dispatcher(
        max_connections: usize,
        queue: usize,
    ) -> (Dispatcher<UnixStream>, Receiver<Job<UnixStream>>) {
        dispatcher_with(None, max_connections, usize::MAX, queue)
    }

    fn dispatcher_with(
        api_key: Option<&str>,
        max_connections: usize,
        max_inflight_bytes: usize,
        queue: usize,
    ) -> (Dispatcher<UnixStream>, Receiver<Job<UnixStream>>) {
        let (jobs, rx) = sync_channel(queue);
        let dispatcher = Dispatcher {
            front: Arc::new(Front::new(
                api_key.map(str::to_string),
                vec!["alpha".to_string()],
            )),
            jobs,
            connections: Arc::new(AtomicUsize::new(0)),
            inflight_bytes: Arc::new(AtomicUsize::new(0)),
            limits: Arc::new(Limits {
                max_connections,
                max_inflight_bytes,
                read_timeout: Duration::from_secs(5),
            }),
        };
        (dispatcher, rx)
    }

    /// Attend que tous les threads de connexion aient rendu la main.
    fn settle(dispatcher: &Dispatcher<UnixStream>) {
        while dispatcher.connections.load(Ordering::Acquire) > 0 {
            thread::yield_now();
        }
    }

    fn json_body(response: &str) -> serde_json::Value {
        let (_, body) = response
            .split_once("\r\n\r\n")
            .expect("invariant: body HTTP présent");
        serde_json::from_str(body).expect("invariant: body JSON")
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
        settle(&dispatcher);

        let second = response(connect(&dispatcher, CHAT));

        assert!(second.starts_with("HTTP/1.1 503 Service Unavailable"));
        assert!(second.contains("file d'inférence pleine"));
        drop(first);
    }

    #[test]
    fn full_queue_on_messages_uses_anthropic_shape() {
        let (dispatcher, _rx) = dispatcher(4, 1);
        let messages = b"POST /v1/messages HTTP/1.1\r\nContent-Length: 2\r\n\r\n{}";
        let _first = connect(&dispatcher, messages);
        settle(&dispatcher);

        let second = response(connect(&dispatcher, messages));

        assert!(second.starts_with("HTTP/1.1 503 Service Unavailable"));
        let value = json_body(&second);
        assert_eq!(value["type"], "error");
        assert_eq!(value["error"]["type"], "api_error");
        assert!(value["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("file d'inférence pleine")));
    }

    #[test]
    fn bearer_rejected_before_reading_body() {
        let (dispatcher, rx) = dispatcher_with(Some("secret"), 4, usize::MAX, 1);
        // Corps annoncé mais jamais envoyé : lire avant l'auth attendrait la
        // deadline (5 s) au lieu de répondre 401.
        let head = b"POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 16000000\r\n\r\n";
        let started = Instant::now();

        let openai = response(connect(&dispatcher, head));
        let anthropic = response(connect(
            &dispatcher,
            b"POST /v1/messages HTTP/1.1\r\nContent-Length: 16000000\r\n\r\n",
        ));

        assert!(started.elapsed() < Duration::from_secs(4));
        assert!(openai.starts_with("HTTP/1.1 401 Unauthorized"));
        assert!(openai.contains("authentification bearer requise"));
        assert_eq!(json_body(&anthropic)["type"], "error");
        assert_eq!(
            json_body(&anthropic)["error"]["type"],
            "authentication_error"
        );
        assert_eq!(dispatcher.inflight_bytes.load(Ordering::Acquire), 0);
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn inflight_bytes_budget_returns_503() {
        let (dispatcher, rx) = dispatcher_with(None, 4, 3, 4);
        let _first = connect(&dispatcher, CHAT);
        settle(&dispatcher);

        let second = response(connect(&dispatcher, CHAT));

        assert!(second.starts_with("HTTP/1.1 503 Service Unavailable"));
        assert!(second.contains("budget"));
        assert_eq!(dispatcher.inflight_bytes.load(Ordering::Acquire), 2);
        drop(rx.recv().expect("invariant: premier job en file"));
        assert_eq!(dispatcher.inflight_bytes.load(Ordering::Acquire), 0);
    }

    #[test]
    fn peer_gone_detects_closed_client_and_restores_blocking() {
        let (client, mut server) = UnixStream::pair().expect("invariant: paire Unix locale");
        server
            .set_read_timeout(Some(Duration::from_millis(50)))
            .expect("invariant: timeout serveur");

        assert!(!peer_gone(&mut server));
        // Mode bloquant restauré : la lecture attend le timeout au lieu de
        // rendre WouldBlock immédiatement.
        let started = Instant::now();
        assert!(server.read(&mut [0_u8; 1]).is_err());
        assert!(started.elapsed() >= Duration::from_millis(40));

        drop(client);
        assert!(peer_gone(&mut server));
    }

    #[test]
    fn engine_skips_job_of_departed_client() {
        let args = ServeArgs::parse(Vec::<String>::new()).expect("invariant: args valides");
        let mut state = ServeState::new(&args);
        let (dispatcher, rx) = dispatcher(4, 2);
        // `settle` entre les deux connexions fixe l'ordre de la file.
        let departed = connect(&dispatcher, CHAT);
        settle(&dispatcher);
        let present = connect(&dispatcher, CHAT);
        settle(&dispatcher);
        drop(departed);

        let first = rx.recv().expect("invariant: job du client parti");
        let second = rx.recv().expect("invariant: job du client présent");

        assert!(!run_job(first, &mut state));
        assert!(run_job(second, &mut state));
        assert!(response(present).starts_with("HTTP/1.1 400"));
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
