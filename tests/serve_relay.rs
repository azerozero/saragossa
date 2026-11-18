//! Relais SIGUSR2 de `saragossa serve` sans modèle : aucune connexion refusée
//! pendant la bascule, ancien processus terminé ; échec de l'enfant sans effet.
#![cfg(all(unix, feature = "serve"))]

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::{Duration, Instant};

/// Lance `serve` en TCP ; renvoie le processus, le port et ses lignes stderr.
/// NOTE: le port libre peut être pris entre-temps (ports source des clients de
/// l'autre test) ; un `serve` sorti au démarrage est relancé sur un autre port.
fn serve(extra: &[&str]) -> (Child, u16, mpsc::Receiver<String>) {
    serve_with(extra, &[])
}

fn serve_with(extra: &[&str], envs: &[(&str, String)]) -> (Child, u16, mpsc::Receiver<String>) {
    for _ in 0..5 {
        if let Some(started) = try_serve(extra, envs) {
            return started;
        }
    }
    panic!("serve ne démarre pas");
}

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .and_then(|l| l.local_addr())
        .expect("port libre")
        .port()
}

fn try_serve(
    extra: &[&str],
    envs: &[(&str, String)],
) -> Option<(Child, u16, mpsc::Receiver<String>)> {
    let port = free_port();
    let mut child = Command::new(env!("CARGO_BIN_EXE_saragossa"))
        .args(["serve", "--port", &port.to_string(), "--api-key", "k"])
        .args(["--model", "t=/nonexistent", "--backend", "cpu"])
        .args(["--set", "serve.drain_timeout_secs=60"])
        .args(extra)
        .envs(envs.iter().map(|(k, v)| (k, v)))
        .stderr(Stdio::piped())
        .spawn()
        .expect("lancement");
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
        if child.try_wait().expect("try_wait").is_some() {
            return None;
        }
        thread::sleep(Duration::from_millis(50));
    }
    Some((child, port, rx))
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
    wait_line(rx, "prêt, vidage")
        .split("enfant pid ")
        .nth(1)
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|pid| pid.parse().ok())
        .expect("pid de l'enfant")
}

fn wait_exit(child: &mut Child, within: Duration) {
    let deadline = Instant::now() + within;
    while child.try_wait().expect("try_wait").is_none() {
        assert!(Instant::now() < deadline, "l'ancien processus ne sort pas");
        thread::sleep(Duration::from_millis(50));
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
    thread::sleep(Duration::from_millis(300));
    signal("-USR2", old.id());
    let new_pid = child_pid(&rx);
    // NOTE: vidage à 60 s : sortir avant 10 s prouve que le parent n'est
    // pas resté bloqué dans un accept pris par l'enfant.
    wait_exit(&mut old, Duration::from_secs(10));
    let served_before = ok.load(Ordering::Relaxed);
    thread::sleep(Duration::from_millis(500));
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
#[test]
fn sequential_relay_queues_until_parent_exits() {
    let (mut old, port, rx) = serve(&[
        "--set",
        "serve.relay_mode=sequential",
        "--set",
        "serve.read_timeout_secs=3",
    ]);
    // Connexion muette : le parent la garde jusqu'à son délai de lecture (3 s).
    let idle = TcpStream::connect(("127.0.0.1", port)).expect("connexion muette");
    thread::sleep(Duration::from_millis(300));
    let stop = Arc::new(AtomicBool::new(false));
    let (clients, _ok, failed) = health_clients(port, &stop);
    signal("-USR2", old.id());
    wait_line(&rx, "relais séquentiel: l'enfant attendra");
    let new_pid = child_pid(&rx);
    let mut chat = TcpStream::connect(("127.0.0.1", port)).expect("connexion chat");
    let body = r#"{"model":"t","messages":[{"role":"user","content":"x"}]}"#;
    write!(
        chat,
        "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer k\r\n\
         Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
    .expect("requête chat");
    chat.set_read_timeout(Some(Duration::from_secs(30)))
        .expect("délai");
    let mut reply = String::new();
    let _ = chat.read_to_string(&mut reply);
    let parent_gone = old.try_wait().expect("try_wait").is_some();
    wait_exit(&mut old, Duration::from_secs(10));
    stop.store(true, Ordering::Relaxed);
    for client in clients {
        client.join().expect("client");
    }
    drop(idle);
    signal("-TERM", new_pid);
    assert!(reply.starts_with("HTTP/1.1 "), "réponse: {reply:?}");
    assert!(!reply.contains("file d'inférence pleine"), "{reply}");
    assert!(
        parent_gone,
        "le moteur de l'enfant a répondu avant la sortie du parent"
    );
    assert_eq!(failed.load(Ordering::Relaxed), 0, "connexions refusées");
}

/// Relais avec le listener h2 (et WebTransport s'il est compilé) : aucune
/// requête h2 perdue, la connexion h2 tenue par le parent reçoit GOAWAY.
#[cfg(feature = "h2")]
#[test]
fn relay_with_h2_refuses_nothing() {
    let h2_port = free_port();
    #[allow(unused_mut)]
    let mut envs = vec![("SARAGOSSA_ENHANCE_H2_ADDR", format!("127.0.0.1:{h2_port}"))];
    #[cfg(feature = "webtransport")]
    envs.push((
        "SARAGOSSA_ENHANCE_WEBTRANSPORT_ADDR",
        format!("127.0.0.1:{}", free_port()),
    ));
    let (mut old, port, rx) = serve_with(&[], &envs);
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
    thread::sleep(Duration::from_millis(300));
    signal("-USR2", old.id());
    #[cfg(feature = "webtransport")]
    wait_line(&rx, "socket wt héritée");
    let new_pid = child_pid(&rx);
    wait_exit(&mut old, Duration::from_secs(10));
    let served_before = h2_ok.load(Ordering::Relaxed);
    thread::sleep(Duration::from_millis(500));
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
