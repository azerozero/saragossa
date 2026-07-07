use std::{env, fs, path::PathBuf};

// Préambule Metal **steel-attention** (encodeur Whisper large-v3-turbo) FIGÉ.
//
// Historiquement généré au build par `vendor/mlx/.../make_compiled_preamble.sh`
// (préprocesseur sur les headers MLX). Depuis le retrait de mlx-rs / vendor/mlx,
// on EMBARQUE la sortie générée une fois pour toutes dans
// `assets/steel_attention.metal` (déterministe, byte-identique, validée par les
// golden STT). `build.rs` ne fait plus que la recopier dans `OUT_DIR`, d'où
// `metal_backend/mod.rs` l'`include_str!`. Régénération (jamais nécessaire en
// pratique) = relancer le script MLX amont sur `utils.h` + `steel_attention.h`.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_METAL");
    println!("cargo:rerun-if-changed=assets/steel_attention.metal");

    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    let output = out_dir.join("steel_attention.metal");
    let target = env::var("TARGET")?;
    // Hors feature `metal` (ou hors Apple Silicon) le kernel n'est jamais
    // compilé : un fichier vide suffit à satisfaire l'`include_str!`.
    if env::var_os("CARGO_FEATURE_METAL").is_none() || !target.contains("apple-darwin") {
        fs::write(output, "")?;
        return Ok(());
    }

    let frozen =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR")?).join("assets/steel_attention.metal");
    fs::copy(&frozen, &output).map_err(|e| {
        format!(
            "copie du préambule Metal figé {} → {}: {e}",
            frozen.display(),
            output.display()
        )
    })?;
    Ok(())
}
