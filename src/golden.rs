//! Support des fixtures « golden » de parité metal-rs ↔ référence figée.
//!
//! Les oracles cross-moteur STT/TTS/clone comparent metal-rs à une **référence
//! figée** stockée dans `tests/golden/` (vecteurs/codes/transcription capturés
//! depuis l'ancienne référence mlx-rs). Les tests `golden_*` tournent **sans
//! mlx-rs** : la comparaison live versus mlx-rs a été supprimée.
//!
//! Format : `<name>.bin` (octets little-endian bruts) + `<name>.json` (en-tête
//! shape/dtype + provenance SHA mlx-rs/date/tolérance). Les chaînes (transcription)
//! sont stockées en `<name>.txt`. Pas d'alignement supposé à la lecture (on
//! reconstruit chaque scalaire via `from_le_bytes`).

#![cfg(test)]
// Les helpers d'écriture/cohérence (`write_*`, `coherence_*`) ne sont plus
// appelés dans le build par défaut (la comparaison live versus mlx-rs est
// supprimée). Les helpers de lecture servent aux tests `golden_*`. On tait le
// lint `dead_code` plutôt que de supprimer du code utile à la maintenance.
#![allow(dead_code)]

use crate::error::{InferError, Result};
use crate::runtime_flags::env_flag;
use std::path::PathBuf;

/// SHA reti (HEAD à la capture) — le fork mlx-rs vendoré est figé sur MLX 0.31.2.
pub const MLX_SHA: &str = "be456e2";
/// Date de capture des goldens (ISO).
pub const CAPTURED: &str = "2026-06-13";

/// Indique si l'on est en mode capture (`RETI_GOLDEN_CAPTURE=1`) — écrit les
/// fixtures depuis la référence mlx-rs au lieu de les comparer.
pub fn capturing() -> bool {
    env_flag("RETI_GOLDEN_CAPTURE", false)
}

/// Racine versionnée des fixtures (`crates/saragossa/tests/golden`).
pub fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden")
}

/// En-tête JSON co-localisé décrivant un fixture binaire (provenance + forme).
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Header {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub tolerance: String,
    pub source: String,
    pub mlx_sha: String,
    pub captured: String,
}

fn io_err(path: &std::path::Path, source: std::io::Error) -> InferError {
    InferError::Io {
        path: path.to_path_buf(),
        source,
    }
}

fn write_header(name: &str, dtype: &str, shape: &[usize], tol: &str, source: &str) -> Result<()> {
    let header = Header {
        dtype: dtype.to_string(),
        shape: shape.to_vec(),
        tolerance: tol.to_string(),
        source: source.to_string(),
        mlx_sha: MLX_SHA.to_string(),
        captured: CAPTURED.to_string(),
    };
    let path = dir().join(format!("{name}.json"));
    let json = serde_json::to_string_pretty(&header)
        .map_err(|err| InferError::Config(format!("sérialisation header golden {name}: {err}")))?;
    std::fs::create_dir_all(dir()).map_err(|source| io_err(&dir(), source))?;
    std::fs::write(&path, format!("{json}\n")).map_err(|source| io_err(&path, source))
}

fn read_header(name: &str) -> Result<Header> {
    let path = dir().join(format!("{name}.json"));
    let raw = std::fs::read_to_string(&path).map_err(|source| io_err(&path, source))?;
    serde_json::from_str(&raw)
        .map_err(|err| InferError::Config(format!("lecture header golden {name}: {err}")))
}

/// Écrit un fixture f32 (`<name>.bin` + `<name>.json`).
pub fn write_f32(name: &str, shape: &[usize], data: &[f32], tol: &str, source: &str) -> Result<()> {
    let path = dir().join(format!("{name}.bin"));
    std::fs::create_dir_all(dir()).map_err(|source| io_err(&dir(), source))?;
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for value in data {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(&path, &bytes).map_err(|source| io_err(&path, source))?;
    write_header(name, "f32", shape, tol, source)
}

/// Lit un fixture f32 ; renvoie `(shape, data)`.
pub fn read_f32(name: &str) -> Result<(Vec<usize>, Vec<f32>)> {
    let header = read_header(name)?;
    let path = dir().join(format!("{name}.bin"));
    let bytes = std::fs::read(&path).map_err(|source| io_err(&path, source))?;
    let data = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect::<Vec<_>>();
    Ok((header.shape, data))
}

/// Écrit un fixture i32 (`<name>.bin` + `<name>.json`).
pub fn write_i32(name: &str, shape: &[usize], data: &[i32], tol: &str, source: &str) -> Result<()> {
    let path = dir().join(format!("{name}.bin"));
    std::fs::create_dir_all(dir()).map_err(|source| io_err(&dir(), source))?;
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for value in data {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(&path, &bytes).map_err(|source| io_err(&path, source))?;
    write_header(name, "i32", shape, tol, source)
}

/// Lit un fixture i32 ; renvoie `(shape, data)`.
pub fn read_i32(name: &str) -> Result<(Vec<usize>, Vec<i32>)> {
    let header = read_header(name)?;
    let path = dir().join(format!("{name}.bin"));
    let bytes = std::fs::read(&path).map_err(|source| io_err(&path, source))?;
    let data = bytes
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect::<Vec<_>>();
    Ok((header.shape, data))
}

/// Écrit un fixture texte (`<name>.txt`) + en-tête.
pub fn write_text(name: &str, text: &str, tol: &str, source: &str) -> Result<()> {
    let path = dir().join(format!("{name}.txt"));
    std::fs::create_dir_all(dir()).map_err(|source| io_err(&dir(), source))?;
    std::fs::write(&path, text).map_err(|source| io_err(&path, source))?;
    write_header(name, "utf8", &[text.len()], tol, source)
}

/// Lit un fixture texte (`<name>.txt`), saut de ligne final retiré.
///
/// On retire les `\n`/`\r` terminaux pour rester robuste au hook `end-of-file-fixer`
/// (prek) qui ajoute un saut de ligne final au fichier versionné — sans incidence
/// sur l'égalité exacte de la transcription (jamais terminée par un espace).
pub fn read_text(name: &str) -> Result<String> {
    let path = dir().join(format!("{name}.txt"));
    let raw = std::fs::read_to_string(&path).map_err(|source| io_err(&path, source))?;
    Ok(raw.trim_end_matches(['\n', '\r']).to_string())
}

/// Vérifie la cohérence d'un vecteur f32 par rapport à la référence figée
/// (longueur + `max_abs` borné par `max_abs_tol`).
pub fn coherence_f32(name: &str, data: &[f32], max_abs_tol: f32) -> Result<()> {
    let (_, golden) = read_f32(name)?;
    assert_eq!(
        golden.len(),
        data.len(),
        "cohérence golden↔mlx {name}: len golden={} mlx={}",
        golden.len(),
        data.len()
    );
    let max_abs = golden
        .iter()
        .zip(data.iter())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_abs <= max_abs_tol,
        "cohérence golden↔mlx {name}: max_abs={max_abs} > {max_abs_tol}"
    );
    Ok(())
}

/// Vérifie la cohérence d'un vecteur i32 par rapport à la référence figée (égalité exacte).
pub fn coherence_i32(name: &str, data: &[i32]) -> Result<()> {
    let (_, golden) = read_i32(name)?;
    assert_eq!(
        golden, data,
        "cohérence golden↔mlx {name}: codes i32 divergents"
    );
    Ok(())
}

/// Vérifie la cohérence d'une chaîne par rapport à la référence figée (égalité exacte).
pub fn coherence_text(name: &str, text: &str) -> Result<()> {
    let golden = read_text(name)?;
    assert_eq!(golden, text, "cohérence golden↔mlx {name}: texte divergent");
    Ok(())
}
