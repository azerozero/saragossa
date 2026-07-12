//! Contraintes de génération guidée appliquées côté CPU.

mod json;

pub use json::{JsonTokenCatalog, JsonTokenConstraint};

use crate::Result;

/// Filtre opt-in appliqué aux logits avant le sampling CPU.
pub trait TokenConstraint: Send + Sync {
    /// Masque les logits non admissibles pour l'état courant.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si aucun token ne peut prolonger la contrainte.
    fn mask_logits(&self, logits: &mut [f32]) -> Result<()>;

    /// Enregistre le token choisi par le sampler.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le token choisi viole la contrainte.
    fn accept_token(&self, token: usize) -> Result<()>;

    /// Indique si l'état courant autorise une fin de génération.
    fn is_finished(&self) -> bool;
}
