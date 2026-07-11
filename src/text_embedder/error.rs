//! Frontière d'erreur de l'embedder texte pur Rust (PATRON `thiserror`, cf.
//! `docs/rust-guidelines.md` R3) : l'appelant distingue un échec **réseau/cache**
//! (résolution HF côté reti — retry ou repli) d'un échec **structurel**
//! (config/tokenizer/poids corrompus — re-tenter ne changera rien). Pas de
//! variante d'inférence : le forward CPU est infaillible une fois les shapes
//! validées au chargement.

use thiserror::Error;

/// Erreur de l'embedder sémantique texte pur Rust.
#[derive(Debug, Error)]
pub enum TextEmbedError {
    /// Téléchargement / résolution d'un artefact HF (config, tokenizer, poids).
    /// Construite côté appelant (reti résout le snapshot HF) : cause typique,
    /// réseau absent au 1ᵉʳ usage ou cache incomplet.
    #[error("téléchargement HF '{what}': {source}")]
    Download {
        /// Artefact concerné (ex. `model.safetensors`).
        what: String,
        /// Cause sous-jacente (résolveur HF).
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// Lecture / parsing du `config.json` du modèle.
    #[error("config du modèle: {0}")]
    Config(String),

    /// Chargement du tokenizer (`tokenizer.json`) ou échec d'encodage.
    #[error("tokenizer: {0}")]
    Tokenizer(String),

    /// Chargement des poids (`model.safetensors`) : clé absente, shape ou
    /// dtype inattendu.
    #[error("chargement des poids: {0}")]
    Weights(String),
}

impl TextEmbedError {
    /// Construit une erreur de téléchargement/résolution HF.
    pub fn download(
        what: impl Into<String>,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::Download {
            what: what.into(),
            source: Box::new(source),
        }
    }

    /// Indique si un retry / un repli a un sens. Source unique de vérité pour
    /// la politique de résilience de l'appelant (PATRON `is_retryable`, R3).
    ///
    /// Seul un échec de **téléchargement** est transitoire ; un échec de
    /// config / poids / tokenizer est structurel.
    #[must_use]
    pub fn is_retryable(&self) -> bool {
        matches!(self, TextEmbedError::Download { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_download_is_retryable() {
        let dl = TextEmbedError::download(
            "model.safetensors",
            std::io::Error::new(std::io::ErrorKind::ConnectionReset, "réseau coupé"),
        );
        assert!(dl.is_retryable());
        assert!(!TextEmbedError::Config("x".into()).is_retryable());
        assert!(!TextEmbedError::Tokenizer("x".into()).is_retryable());
        assert!(!TextEmbedError::Weights("x".into()).is_retryable());
    }
}
