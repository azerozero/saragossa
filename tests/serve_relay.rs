//! Relais SIGUSR2 de `saragossa serve` sans modèle : aucune connexion refusée
//! pendant la bascule, ancien processus terminé ; échec de l'enfant sans effet.
//! Les tests se synchronisent sur des événements (lignes de journal, sortie
//! du processus, requêtes abouties), jamais sur une attente fixe.
#![cfg(all(unix, feature = "serve"))]

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::{Duration, Instant};

/// Lance `serve` en TCP ; renvoie le processus, le port et ses lignes stderr.
fn serve(extra: &[&str]) -> (Child, u16, mpsc::Receiver<String>) {
    let (child, port, _, rx) = spawn_serve(extra, false);
    (child, port, rx)
}

/// Lance `serve` sur des sockets liées par le test et transmises comme par un
/// relais (`SARAGOSSA_LISTEN_FDS`) : aucun port libéré puis repris entre-temps.
/// Avec `enhance`, ajoute le listener h2 (et la socket WebTransport si
/// compilée) ; renvoie aussi le port h2.
fn spawn_serve(extra: &[&str], enhance: bool) -> (Child, u16, Option<u16>, mpsc::Receiver<String>) {
    use std::os::fd::{AsRawFd, OwnedFd};
    use std::os::unix::process::CommandExt;
    let bind = || std::net::TcpListener::bind("127.0.0.1:0").expect("écoute");
    let main = bind();
    let port = main.local_addr().expect("adresse").port();
    let mut listen = main.as_raw_fd().to_string();
    let mut owned: Vec<OwnedFd> = vec![main.into()];
    let mut command = Command::new(env!("CARGO_BIN_EXE_saragossa"));
    let mut h2_port = None;
    if enhance {
        let h2 = bind();
        let addr = h2.local_addr().expect("adresse h2");
        h2_port = Some(addr.port());
        command.env("SARAGOSSA_ENHANCE_H2_ADDR", addr.to_string());
        listen.push_str(&format!(",h2={}", h2.as_raw_fd()));
        owned.push(h2.into());
        #[cfg(feature = "webtransport")]
        {
            let wt = std::net::UdpSocket::bind("127.0.0.1:0").expect("socket wt");
            let addr = wt.local_addr().expect("adresse wt");
            command.env("SARAGOSSA_ENHANCE_WEBTRANSPORT_ADDR", addr.to_string());
            listen.push_str(&format!(",wt={}", wt.as_raw_fd()));
            owned.push(wt.into());
        }
    }
    let fds: Vec<i32> = owned.iter().map(AsRawFd::as_raw_fd).collect();
    command
        .args(["serve", "--port", &port.to_string(), "--api-key", "k"])
        .args(["--model", "t=/nonexistent", "--backend", "cpu"])
        .args(["--set", "serve.drain_timeout_secs=60"])
        .args(extra)
        .env("SARAGOSSA_LISTEN_FDS", listen)
        // NOTE: un enfant de relais orphelin après un échec garderait sinon
        // la sortie de cargo ouverte, et le test ne finirait jamais.
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    // SAFETY: entre fork et exec, seul `fcntl` (async-signal-safe) retire
    // FD_CLOEXEC aux sockets transmises, sur un vecteur déjà alloué.
    unsafe {
        command.pre_exec(move || {
            for &fd in &fds {
                if libc::fcntl(fd, libc::F_SETFD, 0) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
            }
            Ok(())
        });
    }
    let mut child = command.spawn().expect("lancement");
    // NOTE: le test lâche ses copies ; serve garde les sockets.
    drop(owned);
    let (lines, rx) = mpsc::channel();
    let stderr = child.stderr.take().expect("stderr");
    thread::spawn(move || {
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            eprintln!("[serve] {line}");
            let _ = lines.send(line);
        }
    });
    let deadline = Instant::now() + Duration::from_secs(20);
    while !health(port) {
        assert!(Instant::now() < deadline, "serve ne répond pas");
        assert!(
            child.try_wait().expect("try_wait").is_none(),
            "serve sorti au démarrage"
        );
        thread::sleep(Duration::from_millis(50));
    }
    (child, port, h2_port, rx)
}

fn health(port: u16) -> bool {
    let Ok(mut stream) = TcpStream::connect(("127.0.0.1", port)) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
    let mut reply = String::new();
    stream
        .write_all(b"GET /health HTTP/1.1\r\nHost: x\r\n\r\n")
        .and_then(|()| stream.read_to_string(&mut reply))
        .is_ok()
        && reply.starts_with("HTTP/1.1 200")
}

fn signal(name: &str, pid: u32) {
    let status = Command::new("kill")
        .args([name, &pid.to_string()])
        .status()
        .expect("kill");
    assert!(status.success());
}

fn wait_line(rx: &mpsc::Receiver<String>, needle: &str) -> String {
    loop {
        let line = rx
            .recv_timeout(Duration::from_secs(30))
            .unwrap_or_else(|_| panic!("ligne « {needle} » absente"));
        if line.contains(needle) {
            return line;
        }
    }
}

fn child_pid(rx: &mpsc::Receiver<String>) -> u32 {
    parse_child_pid(&wait_line(rx, "prêt, vidage")).expect("pid de l'enfant")
}

fn parse_child_pid(line: &str) -> Option<u32> {
    line.split("enfant pid ")
        .nth(1)
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|pid| pid.parse().ok())
}

fn wait_exit(child: &mut Child, within: Duration) {
    let deadline = Instant::now() + within;
    while child.try_wait().expect("try_wait").is_none() {
        assert!(Instant::now() < deadline, "l'ancien processus ne sort pas");
        thread::sleep(Duration::from_millis(50));
    }
}

/// Attend que `counter` dépasse `beyond` : des requêtes ont abouti depuis.
fn wait_served(counter: &AtomicUsize, beyond: usize, what: &str) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while counter.load(Ordering::Relaxed) <= beyond {
        assert!(Instant::now() < deadline, "{what}");
        thread::sleep(Duration::from_millis(10));
    }
}

/// Clients `/health` en boucle jusqu'à `stop` ; renvoie (réussis, échoués).
fn health_clients(
    port: u16,
    stop: &Arc<AtomicBool>,
) -> (
    Vec<thread::JoinHandle<()>>,
    Arc<AtomicUsize>,
    Arc<AtomicUsize>,
) {
    let (ok, failed) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
    let clients = (0..4)
        .map(|_| {
            let (stop, ok, failed) = (stop.clone(), ok.clone(), failed.clone());
            thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    let counter = if health(port) { &ok } else { &failed };
                    counter.fetch_add(1, Ordering::Relaxed);
                }
            })
        })
        .collect();
    (clients, ok, failed)
}

#[test]
fn relay_hands_over_without_refusing_connections() {
    let (mut old, port, rx) = serve(&[]);
    let stop = Arc::new(AtomicBool::new(false));
    let (clients, ok, failed) = health_clients(port, &stop);
    wait_served(&ok, 0, "clients inactifs avant le relais");
    signal("-USR2", old.id());
    let new_pid = child_pid(&rx);
    // NOTE: vidage à 60 s : sortir avant 10 s prouve que le parent n'est
    // pas resté bloqué dans un accept pris par l'enfant.
    wait_exit(&mut old, Duration::from_secs(10));
    let served_before = ok.load(Ordering::Relaxed);
    wait_served(&ok, served_before, "le nouveau ne sert pas");
    stop.store(true, Ordering::Relaxed);
    for client in clients {
        client.join().expect("client");
    }
    let alive = health(port);
    signal("-TERM", new_pid);
    assert_eq!(failed.load(Ordering::Relaxed), 0, "connexions refusées");
    assert!(
        ok.load(Ordering::Relaxed) > served_before,
        "le nouveau ne sert pas"
    );
    assert!(alive);
    assert_ne!(new_pid, old.id());
}

#[test]
fn failed_child_keeps_parent_serving() {
    let (mut server, port, rx) = serve(&["--upgrade-binary", "/usr/bin/false"]);
    signal("-USR2", server.id());
    wait_line(&rx, "service maintenu");
    let still_running = server.try_wait().expect("try_wait").is_none();
    let alive = health(port);
    let _ = server.kill();
    let _ = server.wait();
    assert!(still_running && alive);
}

/// Mode séquentiel forcé : le moteur de l'enfant attend la sortie du parent,
/// la requête reçue entre-temps attend en file au lieu d'être refusée.
///
/// NOTE: le test tient lui-même le parent en vie par une connexion muette et
/// le libère en la fermant. Comparer l'arrivée de la réponse à `try_wait`
/// serait une course : l'enfant voit son parent changer (fin de `exit`, après
/// la libération de la mémoire) un peu avant que le zombie soit visible à
/// `waitpid`.
#[test]
fn sequential_relay_queues_until_parent_exits() {
    let (mut old, port, rx) = serve(&["--set", "serve.relay_mode=sequential"]);
    // Connexion muette : le parent la sert jusqu'à sa fermeture par le test
    // (délai de lecture 30 s, vidage 60 s).
    let idle = TcpStream::connect(("127.0.0.1", port)).expect("connexion muette");
    // La file d'accept est FIFO et seul le parent accepte encore : une
    // réponse sur une connexion ouverte après prouve que la muette est chez
    // lui (sous charge, une attente fixe la laissait à l'enfant).
    assert!(health(port), "parent muet");
    let stop = Arc::new(AtomicBool::new(false));
    let (clients, _ok, failed) = health_clients(port, &stop);
    signal("-USR2", old.id());
    wait_line(&rx, "relais séquentiel: l'enfant attendra");
    // Après « accept arrêté » le parent n'accepte plus : la requête suivante
    // est forcément chez l'enfant (sinon le parent la servirait lui-même).
    // Les deux lignes viennent de deux threads, dans un ordre quelconque.
    let (mut new_pid, mut stopped) = (None, false);
    while new_pid.is_none() || !stopped {
        let line = rx
            .recv_timeout(Duration::from_secs(30))
            .expect("relais ou arrêt de l'accept absent");
        stopped |= line.contains("relais: accept arrêté");
        if line.contains("prêt, vidage") {
            new_pid = parse_child_pid(&line);
        }
    }
    let new_pid = new_pid.expect("pid de l'enfant");
    let mut chat = TcpStream::connect(("127.0.0.1", port)).expect("connexion chat");
    let body = r#"{"model":"t","messages":[{"role":"user","content":"x"}]}"#;
    write!(
        chat,
        "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer k\r\n\
         Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
    .expect("requête chat");
    // Tant que le parent vit, la requête reste en file : rien à lire.
    chat.set_read_timeout(Some(Duration::from_secs(1)))
        .expect("délai");
    let mut early = [0u8; 1];
    let early = chat.read(&mut early);
    let parent_alive = old.try_wait().expect("try_wait").is_none();
    // Le parent sort une fois sa dernière connexion fermée ; la requête en
    // file est alors servie par l'enfant.
    drop(idle);
    wait_line(&rx, "sorti, moteur actif");
    wait_exit(&mut old, Duration::from_secs(10));
    chat.set_read_timeout(Some(Duration::from_secs(30)))
        .expect("délai");
    let mut reply = String::new();
    let _ = chat.read_to_string(&mut reply);
    stop.store(true, Ordering::Relaxed);
    for client in clients {
        client.join().expect("client");
    }
    signal("-TERM", new_pid);
    assert!(parent_alive, "parent sorti avant la libération du test");
    assert!(
        early.is_err(),
        "le moteur de l'enfant a répondu avant la sortie du parent: {early:?}"
    );
    assert!(reply.starts_with("HTTP/1.1 "), "réponse: {reply:?}");
    assert!(!reply.contains("file d'inférence pleine"), "{reply}");
    assert_eq!(failed.load(Ordering::Relaxed), 0, "connexions refusées");
}

/// Relais avec le listener h2 (et WebTransport s'il est compilé) : aucune
/// requête h2 perdue, la connexion h2 tenue par le parent reçoit GOAWAY.
#[cfg(feature = "h2")]
#[test]
fn relay_with_h2_refuses_nothing() {
    let (mut old, port, h2_port, rx) = spawn_serve(&[], true);
    let h2_port = h2_port.expect("port h2");
    // Connexion h2 tenue par le parent, pilotée sur son propre thread pour
    // répondre au PING du GOAWAY en deux temps.
    let (connected, connected_rx) = mpsc::channel();
    let (check, check_rx) = tokio::sync::oneshot::channel::<()>();
    let held = thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime");
        runtime.block_on(async move {
            let tcp = tokio::net::TcpStream::connect(("127.0.0.1", h2_port))
                .await
                .expect("connexion h2");
            let (mut client, conn) = h2::client::handshake(tcp).await.expect("handshake h2");
            let conn = tokio::spawn(conn);
            connected.send(()).expect("signal");
            let _ = check_rx.await;
            let ended = tokio::time::timeout(Duration::from_secs(5), conn).await;
            let request = http::Request::post("http://x/v1/audio/enhance/h2")
                .body(())
                .expect("requête");
            let refused = client.send_request(request, true).err();
            (ended.is_ok(), refused.is_some_and(|e| e.is_go_away()))
        })
    });
    connected_rx
        .recv_timeout(Duration::from_secs(10))
        .expect("connexion h2 tenue");
    let stop = Arc::new(AtomicBool::new(false));
    let (mut clients, _ok, failed) = health_clients(port, &stop);
    let (h2_ok, h2_failed) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
    for _ in 0..2 {
        let (stop, ok, failed) = (stop.clone(), h2_ok.clone(), h2_failed.clone());
        clients.push(thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("runtime");
            while !stop.load(Ordering::Relaxed) {
                let counter = if runtime.block_on(h2_request(h2_port)) {
                    &ok
                } else {
                    &failed
                };
                counter.fetch_add(1, Ordering::Relaxed);
            }
        }));
    }
    wait_served(&h2_ok, 0, "clients h2 inactifs avant le relais");
    signal("-USR2", old.id());
    #[cfg(feature = "webtransport")]
    wait_line(&rx, "socket wt héritée");
    let new_pid = child_pid(&rx);
    wait_exit(&mut old, Duration::from_secs(10));
    let served_before = h2_ok.load(Ordering::Relaxed);
    wait_served(&h2_ok, served_before, "l'enfant ne sert pas h2");
    stop.store(true, Ordering::Relaxed);
    for client in clients {
        client.join().expect("client");
    }
    let _ = check.send(());
    let goaway = held.join().expect("connexion tenue");
    signal("-TERM", new_pid);
    assert_eq!(
        failed.load(Ordering::Relaxed),
        0,
        "connexions HTTP/1 refusées"
    );
    assert_eq!(h2_failed.load(Ordering::Relaxed), 0, "requêtes h2 perdues");
    assert!(
        h2_ok.load(Ordering::Relaxed) > served_before,
        "l'enfant ne sert pas h2"
    );
    assert_eq!(goaway, (true, true), "connexion h2 du parent sans GOAWAY");
}

/// Une requête h2 sur une connexion neuve ; un GOAWAY reçu avant d'ouvrir le
/// flux est rejoué sur une autre connexion, comme le fait un client HTTP/2.
#[cfg(feature = "h2")]
async fn h2_request(port: u16) -> bool {
    for _ in 0..3 {
        match h2_once(port).await {
            Ok(()) => return true,
            Err(e) if e.is_go_away() || e.reason() == Some(h2::Reason::REFUSED_STREAM) => {}
            Err(e) => {
                eprintln!("requête h2 échouée: {e}");
                return false;
            }
        }
    }
    false
}

#[cfg(feature = "h2")]
async fn h2_once(port: u16) -> Result<(), h2::Error> {
    let Ok(tcp) = tokio::net::TcpStream::connect(("127.0.0.1", port)).await else {
        return Err(h2::Error::from(h2::Reason::CONNECT_ERROR));
    };
    let (client, conn) = h2::client::handshake(tcp).await?;
    let driver = tokio::spawn(conn);
    let mut client = client.ready().await?;
    let request = http::Request::post("http://x/v1/audio/enhance/h2")
        .header("authorization", "Bearer k")
        .body(())
        .expect("requête");
    let (response, _) = client.send_request(request, true)?;
    let _status = response.await?.status();
    drop(client);
    let _ = driver.await;
    Ok(())
}
