//! Qualification conservatrice du DecodeStream de tokenizers.

use super::RustTokenizer;
use crate::{InferError, Result};
use tokenizers::{
    decoders::DecoderWrapper, models::ModelWrapper, normalizers::NormalizerWrapper,
    pre_tokenizers::PreTokenizerWrapper, processors::PostProcessorWrapper, DecodeStream,
};

/// Flux ByteLevel exact. Les caractères encodés représentent des octets ; leur
/// concaténation puis `from_utf8_lossy` est le décodage complet. Un delta ne
/// finissant pas par U+FFFD ferme une frontière stable. Aucun cleanup ni règle
/// d'espace de tête ne s'applique à cette famille de décodeurs.
/// Les added tokens ordinaires passent aussi par ByteLevel : si un caractère
/// sort de son alphabet, le token entier contribue ses octets UTF-8 bruts.
/// Cette règle reste locale au token et conserve les mêmes frontières stables.
pub struct ByteLevelDecodeStream<'a> {
    tokenizer: &'a RustTokenizer,
    stream: DecodeStream<
        'a,
        ModelWrapper,
        NormalizerWrapper,
        PreTokenizerWrapper,
        PostProcessorWrapper,
        DecoderWrapper,
    >,
    pending: Vec<u32>,
    skip_special_tokens: bool,
}

pub(super) fn new(
    tokenizer: &RustTokenizer,
    skip_special_tokens: bool,
) -> Option<ByteLevelDecodeStream<'_>> {
    matches!(
        tokenizer.inner.get_decoder(),
        Some(DecoderWrapper::ByteLevel(_))
    )
    .then(|| ByteLevelDecodeStream {
        tokenizer,
        stream: tokenizer.inner.decode_stream(skip_special_tokens),
        pending: Vec::new(),
        skip_special_tokens,
    })
}

impl ByteLevelDecodeStream<'_> {
    /// Renvoie le nouveau texte stable, sans les octets UTF-8 encore incomplets.
    ///
    /// # Errors
    /// Propage une erreur ou une divergence du préfixe de DecodeStream.
    pub fn step(&mut self, id: u32) -> Result<Option<String>> {
        // Les ids absents et les spéciaux ignorés ne contribuent aucun octet.
        // Les retirer évite de faire croître la fenêtre sur une suite d'EOS.
        let Some(token) = self.tokenizer.inner.id_to_token(id) else {
            return Ok(None);
        };
        if self.skip_special_tokens
            && self
                .tokenizer
                .inner
                .get_added_vocabulary()
                .is_special_token(&token)
        {
            return Ok(None);
        }
        self.pending.push(id);
        let delta = self
            .stream
            .step(id)
            .map_err(|error| InferError::Tokenizer {
                path: self.tokenizer.path.clone(),
                message: error.to_string(),
            })?;
        if delta.is_some() {
            self.pending.clear();
        }
        Ok(delta)
    }

    /// Snapshot de la seule queue retenue : permet de trouver un stop même
    /// lorsque le dernier token se termine par un fragment UTF-8 ou U+FFFD.
    ///
    /// # Errors
    /// Propage les erreurs de décodage de la fenêtre.
    pub fn pending_text(&self) -> Result<String> {
        self.tokenizer
            .decode(&self.pending, self.skip_special_tokens)
    }

    /// Taille de la queue, pour borner le travail avant un repli complet compté.
    #[must_use]
    pub fn pending_tokens(&self) -> usize {
        self.pending.len()
    }
}
