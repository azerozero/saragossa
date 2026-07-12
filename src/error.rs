//! Types d'erreur partagés par le moteur d'inférence.

use safetensors::{Dtype, SafeTensorError};
use std::path::PathBuf;
use thiserror::Error;

use crate::memory_guard::MemoryGuardError;

/// Représente le résultat standard du moteur d'inférence.
pub type Result<T> = std::result::Result<T, InferError>;

#[derive(Debug, Error)]
/// Décrit les erreurs produites par le moteur d'inférence.
pub enum InferError {
    /// Signale une forme de tenseur invalide.
    #[error("forme tensor invalide: {0}")]
    Shape(String),

    /// Signale des dimensions incompatibles.
    #[error("dimension incompatible: {0}")]
    Dimension(String),

    /// Signale un poids absent du modèle.
    #[error("poids manquant: {0}")]
    MissingWeight(String),

    /// Signale une configuration modèle invalide.
    #[error("config modèle invalide: {0}")]
    Config(String),

    /// Signale un artefact modèle manquant.
    #[error("artefact modèle manquant {what}: {path}")]
    MissingArtifact { path: PathBuf, what: &'static str },

    /// Signale un dtype non supporté.
    #[error("dtype non supporté pour {name}: {dtype:?}")]
    UnsupportedDtype { name: String, dtype: Dtype },

    /// Signale une erreur de tokenizer.
    #[error("tokenizer {path}: {message}")]
    Tokenizer { path: PathBuf, message: String },

    /// Signale une erreur JSON.
    #[error("JSON {path}: {source}")]
    Json {
        path: PathBuf,
        source: serde_json::Error,
    },

    /// Signale une erreur safetensors.
    #[error("safetensors {path}: {source}")]
    Safetensors {
        path: PathBuf,
        source: SafeTensorError,
    },

    /// Signale une erreur d'en-tête safetensors.
    #[error("safetensors header {path}: {message}")]
    SafetensorsHeader { path: PathBuf, message: String },

    /// Signale une erreur Metal.
    #[error("metal: {0}")]
    Metal(String),

    /// Signale un refus de la garde mémoire.
    #[error(transparent)]
    MemoryGuard(MemoryGuardError),

    /// Signale une erreur d'entrée-sortie.
    #[error("I/O {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
}
