//! Le prétraitement RGB n'ouvre pas encore le transport média CLI.
use std::process::Command;

#[test]
fn vision_cli_refuses_media_before_loading_a_model() {
    for flag in ["--image", "--video"] {
        let output = Command::new(env!("CARGO_BIN_EXE_saragossa"))
            .args(["run", "unused", flag, "unused"])
            .output()
            .expect("invariant: binaire construit");
        assert_eq!(output.status.code(), Some(2));
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(
            error.contains(&format!("argument run inconnu: {flag}")),
            "{error}"
        );
    }
}
