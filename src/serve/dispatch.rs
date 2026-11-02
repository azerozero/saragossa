//! Concurrence d'E/S de `saragossa serve`.
//!
//! Un thread d'accept confie chaque connexion à un thread de connexion qui lit
//! la requête et sert les routes statiques (`/health`, `/v1/models`). Le reste
//! part dans une file bornée vers le thread moteur, seul propriétaire de
//! [`ServeState`] : l'inférence reste sérielle, seules les E/S sont parallèles,
//! sauf les lots opt-in (`SARAGOSSA_SERVE_BATCH=1`, voir [`run_engine`]) qui
//! décodent ensemble plusieurs requêtes chat streamées greedy.

use std::collections::VecDeque;
use std::io::{self, Read, Write};
use std::net::TcpStream;
#[cfg(unix)]
use std::os::unix::net::UnixStream;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TrySendError};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use saragossa::runtime_flags::{
    serve_batch_enabled, serve_max_connections, serve_max_inflight_bytes, serve_max_streams,
    serve_queue_depth,
};

use super::error::{ServeError, ServeResult};
#[cfg(unix)]
use super::http::configure_unix_stream;
use super::http::{
    batch_chat, configure_tcp_stream, handle_batch, handle_job, read_body, read_head,
    read_or_reject, reply_on_error, route_front, send_error, send_route_error, HttpRequest,
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
}

impl ServeStream for TcpStream {
    fn configure(&self, read_timeout: Duration) -> ServeResult<()> {
        configure_tcp_stream(self, read_timeout)
    }
}

#[cfg(unix)]
impl ServeStream for UnixStream {
    fn configure(&self, read_timeout: Duration) -> ServeResult<()> {
        configure_unix_stream(self, read_timeout)
    }
}

/// Requête lue, en attente du thread moteur avec la socket de sa réponse. La
/// réservation d'octets suit le corps jusqu'à la fin du traitement ; la place
/// d'attente est rendue quand le moteur prend le job.
struct Job<S> {
    request: HttpRequest,
    stream: S,
    _bytes: Reservation,
    waiting: Reservation,
}

/// Plafonds de la façade d'E/S.
struct Limits {
    max_connections: usize,
    max_inflight_bytes: usize,
    /// Jobs en attente du moteur, canal et file locale des lots confondus.
    queue_depth: usize,
    read_timeout: Duration,
}

/// Distribue les connexions acceptées vers des threads de connexion plafonnés.
pub(super) struct Dispatcher<S> {
    front: Arc<Front>,
    jobs: SyncSender<Job<S>>,
    connections: Arc<AtomicUsize>,
    inflight_bytes: Arc<AtomicUsize>,
    waiting: Arc<AtomicUsize>,
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
            waiting: Arc::clone(&self.waiting),
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
    let queue_depth = serve_queue_depth();
    let (jobs, queue) = sync_channel(queue_depth);
    let dispatcher = Dispatcher {
        front: Arc::new(front),
        jobs,
        connections: Arc::new(AtomicUsize::new(0)),
        inflight_bytes: Arc::new(AtomicUsize::new(0)),
        waiting: Arc::new(AtomicUsize::new(0)),
        limits: Arc::new(Limits {
            max_connections: serve_max_connections(),
            max_inflight_bytes: serve_max_inflight_bytes(),
            queue_depth,
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
    let max_streams = if serve_batch_enabled() {
        serve_max_streams()
    } else {
        1
    };
    run_engine(&queue, state, max_streams);
    acceptor
        .join()
        .map_err(|_| ServeError::io("thread d'accept", io::Error::other("panique")))?
}

/// Job en attente du moteur, avec le modèle visé s'il est admissible à un
/// lot (`None` : chemin solo).
///
/// NOTE: seul l'identifiant du modèle est gardé, pas la requête décodée : la
/// file locale ne retient que le corps brut déjà compté dans le budget
/// d'octets. Le lot ré-analyse le corps au moment de servir.
struct Queued<S> {
    job: Job<S>,
    model: Option<String>,
}

impl<S> Queued<S> {
    fn new(job: Job<S>, batching: Option<usize>, state: &ServeState) -> Self {
        let model = batching
            .and_then(|cap| batch_chat(&job.request, cap, |model| state.model_routes_mtp(model)))
            .map(|chat| chat.model().to_string());
        Self { job, model }
    }

    fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }
}

/// Boucle du thread moteur. Avec `max_streams > 1` (B2), un job chat éligible
/// réunit les jobs éligibles du même modèle déjà en file, sans attendre ; les
/// autres restent dans `pending`, dans leur ordre, pour les tours suivants.
///
/// NOTE: l'admission n'a lieu qu'entre deux générations ; un job arrivé
/// pendant un lot attend la fin de tout le lot.
fn run_engine<S: ServeStream>(
    queue: &Receiver<Job<S>>,
    state: &mut ServeState,
    max_streams: usize,
) {
    // Plafond `max_tokens` pour la validation d'admission, `None` sans lots.
    let batching = (max_streams > 1).then(|| state.max_tokens_cap());
    // NOTE: les jobs de `pending` gardent leur place d'attente : canal et
    // file locale restent ensemble sous `SARAGOSSA_SERVE_QUEUE`.
    let pending_cap = serve_queue_depth();
    let mut pending: VecDeque<Queued<S>> = VecDeque::new();
    loop {
        let leader = match pending.pop_front() {
            Some(queued) => queued,
            None => match queue.recv() {
                Ok(job) => Queued::new(job, batching, state),
                Err(_) => break,
            },
        };
        let Some(model) = leader
            .model()
            .map(str::to_string)
            .filter(|model| state.batch_ready(model))
        else {
            run_job(leader.job, state);
            continue;
        };
        drain_queue(queue, &mut pending, pending_cap, |job| {
            Queued::new(job, batching, state)
        });
        let others = take_batch(
            &mut pending,
            &model,
            max_streams.saturating_sub(1),
            Queued::model,
        );
        if others.is_empty() {
            run_job(leader.job, state);
            continue;
        }
        let mut batch = vec![leader];
        batch.extend(others);
        run_batch(batch, state);
    }
}

/// Déplace sans bloquer les jobs déjà en file vers `pending`, jusqu'à `cap`.
fn drain_queue<J, T>(
    queue: &Receiver<J>,
    pending: &mut VecDeque<T>,
    cap: usize,
    wrap: impl Fn(J) -> T,
) {
    while pending.len() < cap {
        match queue.try_recv() {
            Ok(job) => pending.push_back(wrap(job)),
            Err(_) => break,
        }
    }
}

/// Retire de `pending` au plus `max` éléments du modèle `model`, dans l'ordre ;
/// les autres y restent dans leur ordre.
fn take_batch<T>(
    pending: &mut VecDeque<T>,
    model: &str,
    max: usize,
    model_of: impl Fn(&T) -> Option<&str>,
) -> Vec<T> {
    let mut taken = Vec::new();
    let mut kept = VecDeque::with_capacity(pending.len());
    for item in pending.drain(..) {
        if taken.len() < max && model_of(&item) == Some(model) {
            taken.push(item);
        } else {
            kept.push_back(item);
        }
    }
    *pending = kept;
    taken
}

/// Exécute un lot de jobs éligibles du même modèle.
fn run_batch<S: ServeStream>(batch: Vec<Queued<S>>, state: &mut ServeState) {
    let mut jobs: Vec<(S, HttpRequest, Reservation)> = batch
        .into_iter()
        .map(|queued| {
            let Job {
                request,
                stream,
                _bytes,
                waiting,
            } = queued.job;
            drop(waiting);
            (stream, request, _bytes)
        })
        .collect();
    let mut sockets: Vec<(&mut S, &HttpRequest)> = jobs
        .iter_mut()
        .map(|(stream, request, _)| (stream, &*request))
        .collect();
    handle_batch(state, &mut sockets);
}

/// Exécute un job sur le thread moteur.
///
/// NOTE: un client parti pendant l'attente en file n'est pas détecté avant
/// l'exécution : une sonde de lecture confond fermeture complète et
/// demi-fermeture (`shutdown(Write)` après la requête, client toujours à
/// l'écoute), toutes deux vues comme EOF. En streaming, les premières
/// écritures SSE échouent vite (EPIPE) et coupent la génération.
fn run_job<S: ServeStream>(job: Job<S>, state: &mut ServeState) {
    let Job {
        request,
        mut stream,
        _bytes,
        waiting,
    } = job;
    drop(waiting);
    if let Err(error) = handle_job(&mut stream, state, &request) {
        eprintln!("saragossa serve request error: {error}");
    }
}

/// Contexte d'un thread de connexion.
struct Connection<S> {
    front: Arc<Front>,
    jobs: SyncSender<Job<S>>,
    inflight_bytes: Arc<AtomicUsize>,
    waiting: Arc<AtomicUsize>,
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
        let Some(waiting) = Reservation::acquire(&self.waiting, 1, self.limits.queue_depth) else {
            return send_route_error(
                &mut stream,
                &request,
                503,
                "file d'inférence pleine, réessayer",
            );
        };
        let job = Job {
            request,
            stream,
            _bytes: bytes,
            waiting,
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
        // NOTE: boucle compare-exchange plutôt que `fetch_update`, déprécié au
        // profit de `try_update` absent de la MSRV 1.85.
        let mut count = counter.load(Ordering::Acquire);
        loop {
            let total = count.checked_add(amount).filter(|total| *total <= max)?;
            match counter.compare_exchange_weak(count, total, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => {
                    return Some(Self {
                        counter: Arc::clone(counter),
                        amount,
                    })
                }
                Err(current) => count = current,
            }
        }
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
            waiting: Arc::new(AtomicUsize::new(0)),
            limits: Arc::new(Limits {
                max_connections,
                max_inflight_bytes,
                queue_depth: queue,
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

    /// Le lot prend les éléments du même modèle dans l'ordre, sous le
    /// plafond ; les autres restent en file locale dans leur ordre.
    #[test]
    fn take_batch_keeps_set_aside_jobs_in_order() {
        let mut pending: VecDeque<(u32, Option<&str>)> = [
            (1, Some("a")),
            (2, Some("b")),
            (3, None),
            (4, Some("a")),
            (5, Some("a")),
            (6, Some("b")),
        ]
        .into_iter()
        .collect();

        let taken = take_batch(&mut pending, "a", 2, |item| item.1);

        assert_eq!(taken.iter().map(|item| item.0).collect::<Vec<_>>(), [1, 4]);
        assert_eq!(
            pending.iter().map(|item| item.0).collect::<Vec<_>>(),
            [2, 3, 5, 6]
        );
        assert!(take_batch(&mut pending, "a", 0, |item| item.1).is_empty());
        assert_eq!(pending.len(), 4);
    }

    #[test]
    fn drain_queue_stops_at_cap_without_blocking() {
        let (tx, rx) = sync_channel(8);
        for job in 0..5 {
            tx.send(job).expect("invariant: file ouverte");
        }
        let mut pending = VecDeque::from([100]);

        drain_queue(&rx, &mut pending, 3, |job| job);

        assert_eq!(pending, [100, 0, 1]);
        // File vide mais ouverte : rend la main au lieu d'attendre.
        drain_queue(&rx, &mut pending, 10, |job| job);
        assert_eq!(pending, [100, 0, 1, 2, 3, 4]);
    }

    /// Lots actifs sur un modèle non chargeable : chaque job éligible retombe
    /// sur le chemin solo et reçoit sa réponse, aucun n'est perdu.
    #[test]
    fn batching_engine_answers_every_job_when_model_not_batchable() {
        let args = ServeArgs::parse(["--model".to_string(), "alpha=/nonexistent/a".to_string()])
            .expect("invariant: args valides");
        let mut state = ServeState::new(&args);
        let (tx, rx) = sync_channel(4);
        let body = br#"{"model":"alpha","messages":[],"stream":true}"#;
        let raw = [
            format!(
                "POST /v1/chat/completions HTTP/1.1\r\nContent-Length: {}\r\n\r\n",
                body.len()
            )
            .as_bytes(),
            body,
        ]
        .concat();
        let mut clients = Vec::new();
        for _ in 0..3 {
            let (client, server) = UnixStream::pair().expect("invariant: paire Unix locale");
            let mut cursor = std::io::Cursor::new(raw.clone());
            let request = super::super::http::read_request(
                &mut cursor,
                Instant::now() + Duration::from_secs(5),
            )
            .expect("invariant: requête lisible")
            .expect("invariant: requête présente");
            let counter = Arc::new(AtomicUsize::new(0));
            tx.send(Job {
                request,
                stream: server,
                _bytes: Reservation::acquire(&counter, 0, 1).expect("invariant: réservation vide"),
                waiting: Reservation::acquire(&counter, 0, 1).expect("invariant: réservation vide"),
            })
            .expect("invariant: file ouverte");
            clients.push(client);
        }
        drop(tx);

        run_engine(&rx, &mut state, 8);

        for client in clients {
            let answer = response(client);
            assert!(answer.starts_with("HTTP/1.1 "), "réponse absente: {answer}");
            assert!(!answer.starts_with("HTTP/1.1 200"));
        }
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
    fn engine_serves_half_closed_client() {
        let args = ServeArgs::parse(Vec::<String>::new()).expect("invariant: args valides");
        let mut state = ServeState::new(&args);
        let (dispatcher, rx) = dispatcher(4, 2);
        let client = connect(&dispatcher, CHAT);
        settle(&dispatcher);
        // Demi-fermeture : la requête est envoyée, le client écoute encore.
        client
            .shutdown(std::net::Shutdown::Write)
            .expect("invariant: demi-fermeture locale");

        run_job(rx.recv().expect("invariant: job en file"), &mut state);

        assert!(response(client).starts_with("HTTP/1.1 400"));
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
