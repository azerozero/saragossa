//! Erreurs typées du serveur OpenAI local.

use std::io;
use thiserror::Error;

/// Décrit un échec de la sous-commande `serve`.
#[derive(Debug, Error)]
pub(crate) enum ServeError {
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
    /// Génération JSON guidée terminée avant fermeture de l'objet racine.
    #[error("{0}")]
    IncompleteJson(String),
    /// Fonctionnalité connue mais non livrée.
    #[error("non implémenté: {0}")]
    NotImplemented(String),
    /// Modèle demandé absent.
    #[error("modèle inconnu: {0}")]
    UnknownModel(String),
    /// Ressources mémoire insuffisantes.
    #[error("ressource mémoire: {0}")]
    Memory(String),
    /// Echec d'un lot multi-flux, rapporté à chacun de ses clients.
    #[error("lot multi-flux: {0}")]
    Batch(String),
    /// Erreur d'inférence.
    #[error("inférence: {0}")]
    Inference(#[from] saragossa::InferError),
}

impl ServeError {
    /// Construit une erreur d'arguments.
    pub(crate) fn args(message: impl Into<String>) -> Self {
        Self::Args(message.into())
    }

    /// Construit une erreur de JSON guidé incomplet.
    pub(crate) fn incomplete_json(message: impl Into<String>) -> Self {
        Self::IncompleteJson(message.into())
    }

    /// Renvoie le message si l'erreur signale un JSON guidé incomplet.
    pub(crate) fn incomplete_json_message(&self) -> Option<&str> {
        match self {
            Self::IncompleteJson(message) => Some(message),
            _ => None,
        }
    }

    /// Construit une erreur I/O contextualisée.
    pub(crate) fn io(context: impl Into<String>, source: io::Error) -> Self {
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

    /// Construit une erreur de pression mémoire.
    pub(super) fn memory(message: impl Into<String>) -> Self {
        Self::Memory(message.into())
    }

    /// Copie de l'erreur pour chaque client d'un lot : même statut HTTP que
    /// le solo (503 mémoire, 404 modèle…) ; les sources non clonables
    /// gardent leur message.
    pub(super) fn replicate(&self) -> Self {
        match self {
            Self::Args(message) => Self::Args(message.clone()),
            Self::Http(message) => Self::Http(message.clone()),
            Self::Json { .. } => Self::Http(self.to_string()),
            Self::IncompleteJson(message) => Self::IncompleteJson(message.clone()),
            Self::NotImplemented(message) => Self::NotImplemented(message.clone()),
            Self::UnknownModel(model) => Self::UnknownModel(model.clone()),
            Self::Memory(message) => Self::Memory(message.clone()),
            Self::Batch(message) => Self::Batch(message.clone()),
            Self::Io { .. } | Self::Inference(_) => Self::Batch(self.to_string()),
        }
    }

    /// Construit une erreur 501.
    pub(super) fn not_implemented(message: impl Into<String>) -> Self {
        Self::NotImplemented(message.into())
    }
}

/// Résultat local du serveur.
pub(crate) type ServeResult<T> = std::result::Result<T, ServeError>;
