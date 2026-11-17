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
    for _ in 0..5 {
        if let Some(started) = try_serve(extra) {
            return started;
        }
    }
    panic!("serve ne démarre pas");
}

fn try_serve(extra: &[&str]) -> Option<(Child, u16, mpsc::Receiver<String>)> {
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .and_then(|l| l.local_addr())
        .expect("port libre")
        .port();
    let mut child = Command::new(env!("CARGO_BIN_EXE_saragossa"))
        .args(["serve", "--port", &port.to_string(), "--api-key", "k"])
        .args(["--model", "t=/nonexistent", "--backend", "cpu"])
        .args(["--set", "serve.drain_timeout_secs=60"])
        .args(extra)
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

#[test]
fn relay_hands_over_without_refusing_connections() {
    let (mut old, port, rx) = serve(&[]);
    let stop = Arc::new(AtomicBool::new(false));
    let (ok, failed) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
    let clients: Vec<_> = (0..4)
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
    thread::sleep(Duration::from_millis(300));
    signal("-USR2", old.id());
    let line = wait_line(&rx, "prêt, vidage");
    let new_pid: u32 = line
        .split("enfant pid ")
        .nth(1)
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|pid| pid.parse().ok())
        .expect("pid de l'enfant");
    let deadline = Instant::now() + Duration::from_secs(10);
    while old.try_wait().expect("try_wait").is_none() {
        // NOTE: vidage à 60 s : sortir avant 10 s prouve que le parent n'est
        // pas resté bloqué dans un accept pris par l'enfant.
        assert!(Instant::now() < deadline, "l'ancien processus ne sort pas");
        thread::sleep(Duration::from_millis(50));
    }
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
