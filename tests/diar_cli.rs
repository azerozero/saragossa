//! Disponibilité de la sous-commande suivant la modalité compilée.
use std::process::Command;

#[test]
fn diarize_command_follows_compiled_feature() {
    for args in [
        vec!["diarize", "--help"],
        vec!["diarize", "--model-dir", "unused", "--help"],
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_saragossa"))
            .args(args)
            .output()
            .expect("invariant: binaire construit");
        if cfg!(feature = "diar") {
            assert!(output.status.success());
            assert!(String::from_utf8_lossy(&output.stdout).contains("saragossa diarize"));
        } else {
            assert_eq!(output.status.code(), Some(2));
            assert!(String::from_utf8_lossy(&output.stderr).contains("--features diar"));
        }
    }
}
