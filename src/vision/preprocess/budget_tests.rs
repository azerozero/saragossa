use super::*;

const PROFILES: [ImageProfile; 4] = [
    ImageProfile::Qwen38,
    ImageProfile::Qwen36,
    ImageProfile::Thx02,
    ImageProfile::D3Flash,
];

#[test]
fn vision_normalization_and_patches_accept_the_budget_boundary() -> Result<()> {
    let profile = ImageProfile::Thx02;
    let maximum = profile.pixel_bounds().1;
    // Le plus petit plafond suffit à exercer les vraies allocations autorisées.
    let below = RgbImage::from_rgb8(1, maximum - 1, vec![0; (maximum as usize - 1) * 3])?;
    assert_eq!(
        below.normalize_and_repeat(profile)?.pixels().len(),
        (maximum as usize - 1) * 6
    );
    let at = RgbImage::from_rgb8(1280, 1280, vec![255; maximum as usize * 3])?;
    let patches = at.normalize_and_repeat(profile)?.into_patches(profile)?;
    assert_eq!(patches.pixel_values.len(), maximum as usize * 6);
    assert!(patches.pixel_values.iter().all(|&value| value == 1.0));
    Ok(())
}

#[test]
fn vision_temporal_budget_checks_exact_pixel_and_byte_boundaries() -> Result<()> {
    for profile in PROFILES {
        let maximum = profile.pixel_bounds().1;
        // Vérifier les grandes bornes sans réserver de tampon de cette taille.
        for pixels in [maximum - 1, maximum] {
            let len = temporal_len(1, pixels, profile)?;
            assert_eq!(len, pixels as usize * 6);
            assert_eq!(len * size_of::<f32>() * 2, pixels as usize * 48);
        }
        assert!(matches!(
            temporal_len(1, maximum + 1, profile),
            Err(InferError::VisionPixelBudget { pixels, max_pixels })
                if pixels == u64::from(maximum) + 1 && max_pixels == u64::from(maximum)
        ));
        let side = if maximum == 16_777_216 { 4096 } else { 1280 };
        let (h, w) = smart_resize(side + 32, side + 32, profile)?;
        assert_eq!(temporal_len(h, w, profile)?, maximum as usize * 6);
    }
    Ok(())
}

#[test]
fn vision_public_stages_reject_oversized_dimensions_before_touching_buffers() {
    for profile in PROFILES {
        let maximum = profile.pixel_bounds().1;
        let side = if maximum == 16_777_216 { 4096 } else { 1280 };
        for (height, width) in [(1, maximum + 1), (side, side + 32), (8192, 8192)] {
            // Sentinelles internes sans stockage : le refus doit précéder
            // toute allocation de sortie et tout accès aux pixels d'entrée.
            let rgb = RgbImage {
                height,
                width,
                pixels: Vec::new(),
            };
            let temporal = TemporalImage {
                height,
                width,
                pixels: Vec::new(),
            };
            for result in [
                rgb.normalize_and_repeat(profile).map(|_| ()),
                temporal.into_patches(profile).map(|_| ()),
            ] {
                assert!(matches!(result,
                    Err(InferError::VisionPixelBudget { pixels, max_pixels })
                        if pixels == u64::from(height) * u64::from(width)
                            && max_pixels == u64::from(maximum)
                ));
            }
        }
    }
}

#[test]
fn vision_patch_budget_cannot_be_bypassed_by_normalizing_with_larger_profile() {
    let image = TemporalImage {
        height: 4096,
        width: 4096,
        pixels: Vec::new(),
    };
    assert!(matches!(
        image.into_patches(ImageProfile::Thx02),
        Err(InferError::VisionPixelBudget {
            pixels: 16_777_216,
            max_pixels: 1_638_400
        })
    ));
}
