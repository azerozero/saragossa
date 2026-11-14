#![cfg(all(target_os = "macos", feature = "metal"))]

use saragossa::{InferError, KernelProfile, MetalExecutor};

#[test]
fn removed_chunked_flag_propagates_through_public_constructors(
) -> Result<(), Box<dyn std::error::Error>> {
    const CHILD: &str = "SARAGOSSA_TEST_CHUNKED_REFUSAL_CHILD";
    if std::env::var_os(CHILD).is_some() {
        // Les deux entrées alimentent les runtimes per-op et résidents. Le refus
        // doit précéder l'accès au device, même si aucun GPU n'est disponible.
        for result in [
            MetalExecutor::new(),
            MetalExecutor::new_with_kernel_profile(KernelProfile::for_model_dir(
                std::path::Path::new("models/test"),
            )),
        ] {
            assert!(
                matches!(result, Err(InferError::Config(ref message))
                    if message.contains("SARAGOSSA_RUST_LINEAR_CHUNKED")
                        && message.contains("chemin retiré")
                        && message.contains("supprimer le flag")),
                "le constructeur doit propager le refus typé du flag retiré"
            );
        }
        return Ok(());
    }

    // Un processus par valeur : aucun changement d'environnement global ni
    // partage des OnceLock avec les autres tests exécutés en parallèle.
    for flag in ["SARAGOSSA_RUST_LINEAR_CHUNKED", "RETI_RUST_LINEAR_CHUNKED"] {
        for value in ["1", "true", "on"] {
            let output = std::process::Command::new(std::env::current_exe()?)
                .args([
                    "--exact",
                    "removed_chunked_flag_propagates_through_public_constructors",
                    "--nocapture",
                ])
                .env(CHILD, "1")
                .env_remove("SARAGOSSA_RUST_LINEAR_CHUNKED")
                .env_remove("RETI_RUST_LINEAR_CHUNKED")
                .env_remove("SARAGOSSA_RUST_TRACE_LINEAR_ATTN")
                .env_remove("RETI_RUST_TRACE_LINEAR_ATTN")
                .env(flag, value)
                .output()?;
            assert!(
                output.status.success(),
                "{flag}={value}:\n{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }
    Ok(())
}
