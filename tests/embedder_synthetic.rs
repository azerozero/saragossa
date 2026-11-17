#![cfg(feature = "embed")]
#[path = "support/mini_embedder.rs"]
mod mini_embedder;

#[test]
fn golden_synthetic_embedder_cpu() {
    let dir = tempfile::tempdir().expect("invariant: dossier temporaire");
    mini_embedder::write_model(dir.path());
    let model =
        saragossa::TextEmbedder::load_local(dir.path()).expect("invariant: mini BERT valide");
    for (text, passage, mixed) in [
        ("rouge", false, false),
        ("bleu", true, false),
        ("bleu", false, true),
        ("rouge", true, true),
    ] {
        let got = if passage {
            model.embed_passage(text)
        } else {
            model.embed_query(text)
        }
        .expect("invariant: vocabulaire synthétique");
        mini_embedder::assert_golden(&got, mixed, passage);
    }
    let uneven = model
        .embed_query("bleu bleu")
        .expect("invariant: trois tokens synthétiques");
    mini_embedder::assert_uneven_golden(&uneven);
}

#[cfg(all(unix, feature = "devtools"))]
#[test]
fn embeddings_http_cpu_matches_synthetic_golden() {
    use std::io::{Read, Write};
    use std::os::unix::net::UnixStream;
    use std::process::{Child, Command, Stdio};
    use std::time::{Duration, Instant};
    struct Server(Child);
    impl Drop for Server {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
    // Un socket privé évite la course entre réservation et bind d’un port TCP.
    let dir = tempfile::tempdir().expect("invariant: dossier temporaire");
    mini_embedder::write_model(dir.path());
    let socket = dir.path().join("http.sock");
    let mut server = Server(
        Command::new(env!("CARGO_BIN_EXE_saragossa"))
            .args(["serve", "--backend", "cpu", "--embed-model"])
            .arg(dir.path())
            .arg("--socket")
            .arg(&socket)
            .env("HF_HUB_OFFLINE", "1")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .spawn()
            .expect("invariant: binaire de test"),
    );
    let deadline = Instant::now() + Duration::from_secs(15);
    while !socket.exists() {
        assert!(
            server
                .0
                .try_wait()
                .expect("invariant: processus enfant")
                .is_none(),
            "serveur arrêté"
        );
        assert!(Instant::now() < deadline, "démarrage serveur expiré");
        std::thread::sleep(Duration::from_millis(20));
    }
    for (input, count) in [
        (serde_json::json!("bleu"), 1),
        (serde_json::json!(["bleu", "rouge"]), 2),
    ] {
        let body = serde_json::json!({"model":"synthetic", "input":input}).to_string();
        let mut stream = UnixStream::connect(&socket).expect("invariant: socket local");
        stream
            .set_read_timeout(Some(Duration::from_secs(15)))
            .expect("invariant: timeout");
        write!(stream, "POST /v1/embeddings HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body.len(), body)
            .expect("invariant: écriture requête");
        let mut response = String::new();
        stream
            .read_to_string(&mut response)
            .expect("invariant: réponse HTTP");
        let (headers, body) = response
            .split_once("\r\n\r\n")
            .expect("invariant: en-têtes HTTP");
        assert!(headers.starts_with("HTTP/1.1 200 "), "{headers}");
        assert!(headers
            .to_ascii_lowercase()
            .contains("content-type: application/json"));
        let value: serde_json::Value = serde_json::from_str(body).expect("invariant: réponse JSON");
        assert_eq!(value["model"], "synthetic");
        assert_eq!(value["object"], "list");
        let data = value["data"].as_array().expect("invariant: tableau data");
        assert_eq!(data.len(), count);
        for (i, item) in data.iter().enumerate() {
            assert_eq!(item["index"], i);
            assert_eq!(item["object"], "embedding");
            let vector: Vec<f32> = serde_json::from_value(item["embedding"].clone())
                .expect("invariant: vecteur float");
            mini_embedder::assert_golden(&vector, i == 1, true);
        }
    }
}
