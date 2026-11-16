use super::*;

#[test]
fn vision_patch_rows_follow_spatial_groups_then_channels_and_time() -> Result<()> {
    // Chaque coordonnée et chaque canal porte sa propre information : une
    // transposition H/W, C/T ou un parcours raster des patchs devient visible.
    let mut pixels = Vec::new();
    for y in 0..64 {
        for x in 0..96 {
            pixels.extend_from_slice(&[y as u8, x as u8, (y + x) as u8]);
        }
    }
    let image = RgbImage::from_rgb8(64, 96, pixels)?;
    let patches = image.normalize_and_repeat()?.into_patches()?;
    assert_eq!(patches.grid_thw, [1, 4, 6]);
    assert_eq!(patches.pixel_values.len(), 24 * 1536);
    let origins = [
        (0, 0),
        (0, 16),
        (16, 0),
        (16, 16),
        (0, 32),
        (0, 48),
        (16, 32),
        (16, 48),
        (0, 64),
        (0, 80),
        (16, 64),
        (16, 80),
        (32, 0),
        (32, 16),
        (48, 0),
        (48, 16),
        (32, 32),
        (32, 48),
        (48, 32),
        (48, 48),
        (32, 64),
        (32, 80),
        (48, 64),
        (48, 80),
    ];
    for (patch, &(y, x)) in origins.iter().enumerate() {
        for offset in 0..1536 {
            let c = offset / 512;
            let py = offset / 16 % 16;
            let px = offset % 16;
            let value = match c {
                0 => y + py,
                1 => x + px,
                _ => y + py + x + px,
            };
            let expected = (value as f64 / 255.0 - 0.5) / 0.5;
            assert!(
                (f64::from(patches.pixel_values[patch * 1536 + offset]) - expected).abs() <= 1e-6
            );
        }
    }
    Ok(())
}

#[test]
fn vision_patch_order_rejects_unaligned_images() -> Result<()> {
    for (h, w) in [(31, 32), (32, 31), (16, 16), (33, 64)] {
        let image = RgbImage::from_rgb8(h, w, vec![0; (h * w * 3) as usize])?;
        assert!(image.normalize_and_repeat()?.into_patches().is_err());
    }
    Ok(())
}
