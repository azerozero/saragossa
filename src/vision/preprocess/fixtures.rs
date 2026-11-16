//! Portes CPU exécutées sur les sorties indépendantes Python/PIL versionnées.

use super::*;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::path::PathBuf;

#[derive(Deserialize)]
struct Fixture {
    name: String,
    height: u32,
    width: u32,
    pattern: String,
    profile: String,
    resized: [u32; 2],
    file: String,
    sha256: String,
}

#[derive(Deserialize)]
struct ResizeCase {
    profile: String,
    input: [u32; 2],
    output: [u32; 2],
}

#[derive(Deserialize)]
struct Manifest {
    fixtures: Vec<Fixture>,
    filter_probes: Vec<Fixture>,
    smart_resize: Vec<ResizeCase>,
    normalized_sha256: String,
}

fn profile(name: &str) -> ImageProfile {
    match name {
        "Qwen38" => ImageProfile::Qwen38,
        "Qwen36" => ImageProfile::Qwen36,
        "Thx02" => ImageProfile::Thx02,
        "D3Flash" => ImageProfile::D3Flash,
        _ => panic!("profil de fixture inconnu: {name}"),
    }
}

fn directory() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/vision/preprocess")
}

fn manifest() -> Manifest {
    serde_json::from_slice(&std::fs::read(directory().join("manifest.json")).unwrap()).unwrap()
}

fn synthetic(fixture: &Fixture) -> RgbImage {
    let mut pixels = Vec::new();
    for y in 0..fixture.height {
        for x in 0..fixture.width {
            for c in 0..3 {
                let value = match fixture.pattern.as_str() {
                    "ramp" => x * 13 + y * 7 + c * 73,
                    "checker" => ((x + y + c) % 2) * 255,
                    "colors" => c * 127 + (x / 16 + y / 16) % 2,
                    "impulse" => u32::from(x == fixture.width / 2 && y == fixture.height / 2) * 255,
                    _ => panic!("motif de fixture inconnu"),
                };
                pixels.push((value % 256) as u8);
            }
        }
    }
    RgbImage::from_rgb8(fixture.height, fixture.width, pixels).unwrap()
}

fn golden(fixture: &Fixture) -> Vec<u8> {
    let data = std::fs::read(directory().join(&fixture.file)).unwrap();
    assert_eq!(sha256(&data), fixture.sha256);
    data
}

fn sha256(data: &[u8]) -> String {
    Sha256::digest(data)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn normalized_levels() -> Vec<f32> {
    let data = std::fs::read(directory().join("normalized-f32.bin")).unwrap();
    assert_eq!(data.len(), 256 * 4);
    assert_eq!(sha256(&data), manifest().normalized_sha256);
    data.chunks_exact(4)
        .map(|bytes| {
            f32::from_le_bytes(bytes.try_into().expect("invariant: chunk de quatre octets"))
        })
        .collect()
}

#[test]
fn vision_normalization_matches_all_python_levels_and_repeats_afterward() -> Result<()> {
    let pixels: Vec<u8> = (0..=255)
        .flat_map(|value| [value, 255 - value, 127])
        .collect();
    let image = RgbImage::from_rgb8(16, 16, pixels)?;
    let actual = image.normalize_and_repeat()?;
    assert_eq!(actual.shape(), [2, 3, 16, 16]);
    let frame_len = 3 * 16 * 16;
    assert_eq!(&actual.pixels()[..frame_len], &actual.pixels()[frame_len..]);
    let levels = normalized_levels();
    let mut max_error = 0_f32;
    for (position, rgb) in image.pixels().chunks_exact(3).enumerate() {
        for c in 0..3 {
            let value = actual.pixels()[c * 256 + position];
            max_error = max_error.max((value - levels[rgb[c] as usize]).abs());
            let mathematical = (f64::from(rgb[c]) / 255.0 - 0.5) / 0.5;
            assert!((f64::from(value) - mathematical).abs() <= 1e-6);
        }
    }
    eprintln!("normalisation 256 niveaux: max_abs={max_error:e}");
    assert!(max_error <= 1e-6);
    Ok(())
}

#[test]
fn vision_normalized_no_resize_fixtures_pass_f32_gate() -> Result<()> {
    let levels = normalized_levels();
    let mut count = 0;
    for fixture in manifest().fixtures {
        if fixture.resized != [fixture.height, fixture.width] {
            continue;
        }
        count += 1;
        let source = synthetic(&fixture);
        let image = source.resized(profile(&fixture.profile))?;
        let actual = image.normalize_and_repeat()?;
        let rgb = golden(&fixture);
        let plane = (fixture.height * fixture.width) as usize;
        let mut max_error = 0_f32;
        for t in 0..2 {
            for (position, pixel) in rgb.chunks_exact(3).enumerate() {
                for c in 0..3 {
                    let value = actual.pixels()[(t * 3 + c) * plane + position];
                    assert!(value.is_finite());
                    max_error = max_error.max((value - levels[pixel[c] as usize]).abs());
                }
            }
        }
        eprintln!("{} normalisé: max_abs={max_error:e}", fixture.name);
        assert!(max_error <= 1e-6, "{}", fixture.name);
    }
    assert_eq!(count, 5);
    Ok(())
}

#[test]
fn vision_smart_resize_matches_installed_python() -> Result<()> {
    let reference = manifest();
    assert_eq!(reference.smart_resize.len(), 60);
    for case in reference.smart_resize {
        let actual = smart_resize(case.input[0], case.input[1], profile(&case.profile))?;
        assert_eq!([actual.0, actual.1], case.output, "{:?}", case.input);
    }
    Ok(())
}

#[test]
fn vision_bicubic_passes_uint8_gate_on_twelve_python_fixtures() -> Result<()> {
    let reference = manifest();
    assert_eq!(reference.fixtures.len(), 12);
    for fixture in reference.fixtures {
        let input = synthetic(&fixture);
        let actual = input.resized(profile(&fixture.profile))?;
        assert_eq!([actual.height(), actual.width()], fixture.resized);
        let expected = golden(&fixture);
        assert_eq!(actual.pixels().len(), expected.len());
        let error = actual
            .pixels()
            .iter()
            .zip(&expected)
            .map(|(&a, &b)| a.abs_diff(b))
            .max()
            .unwrap();
        eprintln!("{}: max_abs_uint8={error}", fixture.name);
        if fixture.resized == [fixture.height, fixture.width] {
            assert_eq!(actual.pixels(), input.pixels());
            assert_eq!(error, 0, "{}", fixture.name);
        } else {
            assert!(error <= 1, "{}: {error} niveaux uint8", fixture.name);
        }
    }
    Ok(())
}

#[test]
fn vision_bicubic_antialiases_strong_reduction_and_handles_single_pixel_axes() -> Result<()> {
    let reference = manifest();
    assert_eq!(reference.filter_probes.len(), 6);
    for fixture in reference.filter_probes {
        let actual = resize::bicubic(&synthetic(&fixture), fixture.resized[0], fixture.resized[1])?;
        let expected = golden(&fixture);
        assert_eq!(actual.pixels().len(), expected.len());
        let error = actual
            .pixels()
            .iter()
            .zip(&expected)
            .map(|(&a, &b)| a.abs_diff(b))
            .max()
            .unwrap();
        eprintln!("{}: max_abs_uint8={error}", fixture.name);
        assert!(error <= 1, "{}: {error} niveaux uint8", fixture.name);
    }
    Ok(())
}

#[test]
fn vision_rgb_rejects_malformed_buffers() {
    for (h, w, bytes) in [(0, 0, 0), (1, 1, 2), (1, 1, 4), (u32::MAX, u32::MAX, 0)] {
        assert!(RgbImage::from_rgb8(h, w, vec![0; bytes]).is_err());
    }
    assert!(allocate::<f32>(usize::MAX).is_err());
}
