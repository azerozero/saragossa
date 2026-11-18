//! `saragossa config check|print` : codes de sortie et sortie effective, sans modèle.

use std::process::Command;

fn run(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_saragossa"))
        .arg("config")
        .args(args)
        .env_remove("SARAGOSSA_SERVE_QUEUE")
        .output()
        .expect("lancement")
}

#[test]
fn check_and_print() {
    let dir = std::env::temp_dir().join(format!("saragossa-config-cli-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("dossier");
    let good = dir.join("good.toml");
    let bad = dir.join("bad.toml");
    std::fs::write(
        &good,
        "extends = \"low-mem\"\n[serve]\nread_timeout_secs = 9\n",
    )
    .expect("good");
    std::fs::write(&bad, "[serve]\nqueue = 1\nnope = 2\n").expect("bad");
    let (good, bad) = (good.to_str().expect("utf8"), bad.to_str().expect("utf8"));

    assert!(run(&["check", good]).status.success());
    let out = run(&["check", bad]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("line 3, column 1"));
    assert!(!run(&["check", "/nonexistent.toml"]).status.success());

    let out = run(&["print", good, "--set", "serve.queue=5"]);
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "{text}");
    assert!(text.contains("serve.queue = 5  # --set"), "{text}");
    assert!(
        text.contains("serve.model_pool = 1  # preset low-mem"),
        "{text}"
    );
    assert!(
        text.contains("serve.max_tokens_cap = 4096  # défaut"),
        "{text}"
    );
    // Env vide ou invalide : pas d'origine env ; alias hérité valide : origine env.
    let print_with = |name: &str, value: &str| {
        let out = Command::new(env!("CARGO_BIN_EXE_saragossa"))
            .args(["config", "print"])
            .env_remove("SARAGOSSA_SERVE_QUEUE")
            .env(name, value)
            .output()
            .expect("lancement");
        String::from_utf8_lossy(&out.stdout).into_owned()
    };
    assert!(print_with("SARAGOSSA_SERVE_QUEUE", "").contains("serve.queue = 32  # défaut"));
    assert!(print_with("SARAGOSSA_SERVE_QUEUE", "abc").contains("serve.queue = 32  # défaut"));
    assert!(print_with("RETI_SERVE_QUEUE", "7").contains("serve.queue = 7  # env RETI_SERVE_QUEUE"));
    let _ = std::fs::remove_dir_all(dir);
}
