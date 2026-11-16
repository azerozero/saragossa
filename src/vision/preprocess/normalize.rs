//! Normalisation f32 fusionnée du processeur Qwen, puis duplication temporelle.

use super::{allocate, invalid, Result, RgbImage};

/// Deux frames identiques normalisées, contiguës dans l'ordre [T,C,H,W].
#[derive(Debug)]
pub struct TemporalImage {
    pub(super) height: u32,
    pub(super) width: u32,
    pub(super) pixels: Vec<f32>,
}

impl TemporalImage {
    /// Dimensions [2,3,H,W] ; les frames sont répétées après normalisation.
    pub fn shape(&self) -> [usize; 4] {
        [2, 3, self.height as usize, self.width as usize]
    }

    /// Pixels f32 dans l'ordre TCHW.
    pub fn pixels(&self) -> &[f32] {
        &self.pixels
    }
}

pub(super) fn normalize_and_repeat(image: &RgbImage) -> Result<TemporalImage> {
    let frame_len = image.pixels.len();
    let len = frame_len
        .checked_mul(2)
        .ok_or_else(|| invalid("taille temporelle non représentable"))?;
    let mut pixels = allocate::<f32>(len)?;
    let plane_len = frame_len / 3;
    let (first, second) = pixels.split_at_mut(frame_len);
    for (position, rgb) in image.pixels.chunks_exact(3).enumerate() {
        for c in 0..3 {
            // Forme fusionnée employée par Transformers 5.8.1, équivalente
            // à (pixel / 255 - 0,5) / 0,5 à l'arrondi f32 près.
            first[c * plane_len + position] = (f32::from(rgb[c]) - 127.5) / 127.5;
        }
    }
    second.copy_from_slice(first);
    Ok(TemporalImage {
        height: image.height,
        width: image.width,
        pixels,
    })
}
