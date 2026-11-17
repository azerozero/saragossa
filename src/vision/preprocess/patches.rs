//! Permutation des patchs Qwen : groupes spatiaux 2 × 2, puis C,T,H,W.

use super::{allocate, invalid, temporal_len, ImageProfile, Result, RgbImage, TemporalImage};

/// Entrée du ViT : une ligne de 1 536 valeurs par patch 16 × 16 × 2.
#[derive(Debug)]
pub struct PreprocessedImage {
    /// Matrice aplatie [grid_h × grid_w, 3 × 2 × 16 × 16].
    pub pixel_values: Vec<f32>,
    /// Grille [1,H/16,W/16], avant fusion spatiale 2 × 2.
    pub grid_thw: [usize; 3],
}

impl TemporalImage {
    /// Regroupe les lignes par [bloc_h,bloc_w,intra_h,intra_w].
    /// Chaque ligne conserve [C,T,patch_h,patch_w], et non le layout MLX.
    /// Vérifie le plafond du profil avant allocation ; les deux tampons f32
    /// coexistants occupent au plus 48 octets par pixel autorisé.
    pub fn into_patches(self, profile: ImageProfile) -> Result<PreprocessedImage> {
        let len = temporal_len(self.height, self.width, profile)?;
        let height = self.height as usize;
        let width = self.width as usize;
        if height % 32 != 0 || width % 32 != 0 {
            return Err(invalid("les patchs exigent des dimensions multiples de 32"));
        }
        let plane = height * width;
        let frame = plane * 3;
        let mut pixel_values = allocate::<f32>(len)?;
        let mut destination = 0;
        for block_h in 0..height / 32 {
            for block_w in 0..width / 32 {
                for intra_h in 0..2 {
                    for intra_w in 0..2 {
                        let y = block_h * 32 + intra_h * 16;
                        let x = block_w * 32 + intra_w * 16;
                        for c in 0..3 {
                            for t in 0..2 {
                                for row in 0..16 {
                                    let source = t * frame + c * plane + (y + row) * width + x;
                                    pixel_values[destination..destination + 16]
                                        .copy_from_slice(&self.pixels[source..source + 16]);
                                    destination += 16;
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(PreprocessedImage {
            pixel_values,
            grid_thw: [1, height / 16, width / 16],
        })
    }
}

/// Prétraitement complet d'une image déjà décodée : resize, normalisation,
/// répétition temporelle et permutation des patchs. Aucun GPU ni poids requis.
pub fn preprocess(image: &RgbImage, profile: ImageProfile) -> Result<PreprocessedImage> {
    image
        .resized(profile)?
        .normalize_and_repeat(profile)?
        .into_patches(profile)
}
