//! Embedder sémantique **texte** multilingue (`intfloat/multilingual-e5-small`,
//! BERT 384-dim, **MIT**) en **Rust pur f32 CPU**, pour la mémoire de reti.
//!
//! Distinct de [`crate::embedding`] (embeddings de tokens du LLM) : ici on
//! produit un vecteur sémantique par **phrase** (mean-pooling masqué +
//! L2-normalize), surface [`TextEmbedder::embed_query`] /
//! [`TextEmbedder::embed_passage`] → `[f32; TEXT_EMBED_DIM]`. Le forward tourne
//! sur le **CPU** (rayon par ligne de tokens) : aucune queue Metal, aucune
//! dépendance C/C++. La résolution du snapshot HF est laissée à l'appelant
//! (même idiome que [`crate::WhisperModel::from_model_dir`] /
//! [`crate::TtsModel::load_local`]) — d'où [`TextEmbedder::load_local`].
//!
//! ## Concurrence
//!
//! Tout est CPU et `&self` : [`TextEmbedder`] est `Send + Sync`, aucun verrou à
//! prendre par l'appelant. Le parallélisme interne passe par le pool rayon
//! global, partagé avec le reste de `saragossa`.

mod config;
mod error;
mod math;
mod model;
mod weights;

pub use error::TextEmbedError;
pub use model::{TextEmbedder, DEFAULT_TEXT_EMBED_REPO, TEXT_EMBED_DIM};

#[cfg(test)]
mod tests {
    use super::*;

    /// [`TextEmbedder`] doit rester partageable entre tâches sans verrou (contrat
    /// de frontière du port CPU).
    #[test]
    fn embedder_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<TextEmbedder>();
    }
}
