//! Erreurs typées du serveur OpenAI local.

use std::io;
use thiserror::Error;

/// Décrit un échec de la sous-commande `serve`.
#[derive(Debug, Error)]
pub(super) enum ServeError {
    /// Argument CLI invalide.
    #[error("{0}")]
    Args(String),
    /// Erreur d'entrée-sortie.
    #[error("I/O {context}: {source}")]
    Io {
        /// Contexte de l'appel.
        context: String,
        /// Source bas niveau.
        source: io::Error,
    },
    /// Erreur JSON.
    #[error("JSON {context}: {source}")]
    Json {
        /// Contexte de sérialisation ou désérialisation.
        context: String,
        /// Source serde.
        source: serde_json::Error,
    },
    /// Requête HTTP invalide.
    #[error("HTTP invalide: {0}")]
    Http(String),
    /// Modèle demandé absent.
    #[error("modèle inconnu: {0}")]
    UnknownModel(String),
    /// Erreur d'inférence.
    #[error("inférence: {0}")]
    Inference(#[from] saragossa::InferError),
}

impl ServeError {
    /// Construit une erreur d'arguments.
    pub(super) fn args(message: impl Into<String>) -> Self {
        Self::Args(message.into())
    }

    /// Construit une erreur I/O contextualisée.
    pub(super) fn io(context: impl Into<String>, source: io::Error) -> Self {
        Self::Io {
            context: context.into(),
            source,
        }
    }

    /// Construit une erreur JSON contextualisée.
    pub(super) fn json(context: impl Into<String>, source: serde_json::Error) -> Self {
        Self::Json {
            context: context.into(),
            source,
        }
    }
}

/// Résultat local du serveur.
pub(super) type ServeResult<T> = std::result::Result<T, ServeError>;
