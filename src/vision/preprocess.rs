//! Prétraitement Qwen sur pixels RGB déjà décodés, sans appareil Metal.

use crate::{InferError, Result};

mod normalize;
mod patches;
mod resize;

pub use normalize::TemporalImage;
pub use patches::{preprocess, PreprocessedImage};

/// Image RGB uint8 contiguë, dans l'ordre [H,W,C], sans padding.
#[derive(Debug)]
pub struct RgbImage {
    height: u32,
    width: u32,
    pixels: Vec<u8>,
}

impl RgbImage {
    /// Valide les dimensions et la longueur d'un tampon RGB déjà décodé.
    pub fn from_rgb8(height: u32, width: u32, pixels: Vec<u8>) -> Result<Self> {
        if pixels.len() != rgb_len(height, width)? {
            return Err(invalid("longueur RGB différente de H × W × 3"));
        }
        Ok(Self {
            height,
            width,
            pixels,
        })
    }

    /// Hauteur en pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Largeur en pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Pixels RGB dans l'ordre HWC.
    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }

    /// Applique les dimensions Qwen puis le bicubique antialiasé uint8 de PIL.
    pub fn resized(&self, profile: ImageProfile) -> Result<Self> {
        let (height, width) = smart_resize(self.height, self.width, profile)?;
        resize::bicubic(self, height, width)
    }

    /// Normalise les canaux RGB en f32 puis répète l'image sur T=2.
    pub fn normalize_and_repeat(&self) -> Result<TemporalImage> {
        normalize::normalize_and_repeat(self)
    }
}

fn rgb_len(height: u32, width: u32) -> Result<usize> {
    if height == 0 || width == 0 {
        return Err(invalid("dimensions nulles"));
    }
    (height as usize)
        .checked_mul(width as usize)
        .and_then(|n| n.checked_mul(3))
        .ok_or_else(|| invalid("taille RGB non représentable"))
}

fn allocate<T: Default + Clone>(len: usize) -> Result<Vec<T>> {
    let mut data = Vec::new();
    data.try_reserve_exact(len)
        .map_err(|_| invalid("allocation du tampon impossible"))?;
    data.resize(len, T::default());
    Ok(data)
}

/// Profil de budget image ; d3 utilise sa limite runtime, pas celle du fichier.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImageProfile {
    /// Qwen3.8-27B : 65 536 à 16 777 216 pixels.
    Qwen38,
    /// Qwen3.6-35B : 65 536 à 16 777 216 pixels.
    Qwen36,
    /// THX-02 : 65 536 à 1 638 400 pixels.
    Thx02,
    /// d3-flash : 65 536 à 1 638 400 pixels.
    D3Flash,
}

impl ImageProfile {
    /// Bornes (minimum, maximum) du contrat de prétraitement.
    pub const fn pixel_bounds(self) -> (u32, u32) {
        let maximum = match self {
            Self::Qwen38 | Self::Qwen36 => 16_777_216,
            Self::Thx02 | Self::D3Flash => 1_638_400,
        };
        (65_536, maximum)
    }
}

fn invalid(message: &str) -> InferError {
    InferError::Dimension(format!("prétraitement vision: {message}"))
}

/// Retourne (hauteur, largeur), multiples de 32, selon `smart_resize` Qwen.
///
/// Les branches floor/ceil utilisent les dimensions originales, après le test
/// du budget sur les dimensions arrondies au demi pair. Aucun second ajustement
/// des bornes n'est appliqué : il changerait les grilles du processeur Python.
pub fn smart_resize(height: u32, width: u32, profile: ImageProfile) -> Result<(u32, u32)> {
    if height == 0 || width == 0 {
        return Err(invalid("dimensions nulles"));
    }
    let h = f64::from(height);
    let w = f64::from(width);
    if h.max(w) / h.min(w) > 200.0 {
        return Err(invalid("ratio supérieur à 200"));
    }
    let (min_pixels, max_pixels) = profile.pixel_bounds();
    let mut resized_h = (h / 32.0).round_ties_even() * 32.0;
    let mut resized_w = (w / 32.0).round_ties_even() * 32.0;
    if resized_h * resized_w > f64::from(max_pixels) {
        let beta = (h * w / f64::from(max_pixels)).sqrt();
        resized_h = ((h / beta / 32.0).floor() * 32.0).max(32.0);
        resized_w = ((w / beta / 32.0).floor() * 32.0).max(32.0);
    } else if resized_h * resized_w < f64::from(min_pixels) {
        let beta = (f64::from(min_pixels) / (h * w)).sqrt();
        resized_h = (h * beta / 32.0).ceil() * 32.0;
        resized_w = (w * beta / 32.0).ceil() * 32.0;
    }
    Ok((resized_h as u32, resized_w as u32))
}

#[cfg(test)]
mod fixtures;

#[cfg(test)]
mod patch_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vision_resize_rounds_ties_to_even_before_testing_bounds() -> Result<()> {
        for (h, w, expected) in [
            (256, 256, (256, 256)),
            (272, 304, (256, 320)),
            (304, 336, (320, 320)),
            (240, 272, (256, 256)),
            (256, 255, (256, 256)),
            (255, 256, (256, 256)),
            (257, 257, (256, 256)),
            (1, 1, (256, 256)),
            (1, 200, (32, 3648)),
            (200, 1, (3648, 32)),
        ] {
            assert_eq!(smart_resize(h, w, ImageProfile::Qwen38)?, expected);
        }
        Ok(())
    }

    #[test]
    fn vision_resize_honors_each_model_budget() -> Result<()> {
        for profile in [ImageProfile::Qwen38, ImageProfile::Qwen36] {
            assert_eq!(smart_resize(4096, 4096, profile)?, (4096, 4096));
            assert_eq!(smart_resize(4128, 4128, profile)?, (4096, 4096));
            assert_eq!(smart_resize(6000, 3000, profile)?, (5792, 2880));
        }
        for profile in [ImageProfile::Thx02, ImageProfile::D3Flash] {
            assert_eq!(smart_resize(1280, 1280, profile)?, (1280, 1280));
            assert_eq!(smart_resize(1312, 1312, profile)?, (1280, 1280));
            assert_eq!(smart_resize(6000, 3000, profile)?, (1792, 896));
        }
        Ok(())
    }

    #[test]
    fn vision_resize_rejects_invalid_sizes_and_handles_large_dimensions() -> Result<()> {
        for (h, w) in [(0, 1), (1, 0), (0, 0), (1, 201), (201, 1)] {
            assert!(smart_resize(h, w, ImageProfile::Qwen38).is_err());
        }
        assert_eq!(
            smart_resize(u32::MAX, u32::MAX, ImageProfile::Qwen38)?,
            (4096, 4096)
        );
        Ok(())
    }
}
