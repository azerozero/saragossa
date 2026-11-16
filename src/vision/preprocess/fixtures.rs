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
