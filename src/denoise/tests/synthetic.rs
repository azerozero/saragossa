//! Modèles synthétiques utilisés par l'oracle historique.
use super::*;
#[path = "../../../tests/support/dfn3_weights.rs"]
mod fixture;
pub(super) fn model() -> DeepFilterNet3 {
    build(false)
}
pub(super) fn recurrent_model() -> DeepFilterNet3 {
    build(true)
}
fn build(recurrent: bool) -> DeepFilterNet3 {
    let widths = dsp::erb_widths();
    let weights = fixture::weights(recurrent, &widths)
        .into_iter()
        .map(|(name, (shape, data))| (name, crate::Tensor::from_vec(shape, data).unwrap()))
        .collect();
    DeepFilterNet3 {
        net: Network::load(weights, FREQ_BINS).unwrap(),
        widths,
        window: dsp::vorbis_window(),
    }
}
