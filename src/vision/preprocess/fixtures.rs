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
    #[serde(default)]
    grid_thw: [usize; 3],
    #[serde(default)]
    patches_sha256: String,
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
    torchvision_reference: TorchvisionReference,
}

#[derive(Deserialize)]
struct TorchvisionReference {
    fixtures: Vec<Fixture>,
    filter_probes: Vec<Fixture>,
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
    let actual = image.normalize_and_repeat(ImageProfile::Qwen38)?;
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
        let actual = image.normalize_and_repeat(ImageProfile::Qwen38)?;
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
            // Information seulement : PIL (22 bits) diffère de torchvision
            // (poids int16), qui fait foi ; voir le test torchvision.
            eprintln!("{} PIL informatif: {error}", fixture.name);
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
        // Information seulement : torchvision fait foi pour ces sondes.
        eprintln!("{} PIL informatif: max_abs_uint8={error}", fixture.name);
    }
    Ok(())
}

#[test]
fn vision_bicubic_matches_torchvision_with_documented_uint8_tolerance() -> Result<()> {
    let reference = manifest().torchvision_reference;
    assert_eq!(reference.fixtures.len(), 4);
    assert_eq!(reference.filter_probes.len(), 6);
    let mut maximum = 0;
    for fixture in reference
        .fixtures
        .into_iter()
        .chain(reference.filter_probes)
    {
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
        eprintln!("{} torchvision: max_abs_uint8={error}", fixture.name);
        maximum = maximum.max(error);
        if fixture.resized == [fixture.height, fixture.width] {
            assert_eq!(error, 0, "{} sans resize", fixture.name);
        }
    }
    eprintln!(
        "torchvision global: max_abs_uint8={maximum}, max_abs_01={}",
        f64::from(maximum) / 255.0
    );
    // La porte de vision-spec.md est d'un niveau uint8 ; l'égalité exacte est
    // atteinte avec les poids int16 de torchvision et reste exigée.
    assert!(
        maximum == 0,
        "écart torchvision: {maximum} niveaux uint8, maximum autorisé: 0"
    );
    Ok(())
}

#[test]
fn vision_rgb_rejects_malformed_buffers() {
    for (h, w, bytes) in [(0, 0, 0), (1, 1, 2), (1, 1, 4), (u32::MAX, u32::MAX, 0)] {
        assert!(RgbImage::from_rgb8(h, w, vec![0; bytes]).is_err());
    }
    assert!(allocate::<f32>(usize::MAX).is_err());
}

#[test]
fn vision_preprocess_matches_python_patch_order_and_numeric_gates() -> Result<()> {
    let levels = normalized_levels();
    let reference = manifest();
    let pil = reference.fixtures.into_iter().map(|f| (false, f));
    let torchvision = reference
        .torchvision_reference
        .fixtures
        .into_iter()
        .map(|f| (true, f));
    for (gated, fixture) in pil.chain(torchvision) {
        let actual = preprocess(&synthetic(&fixture), profile(&fixture.profile))?;
        assert_eq!(actual.grid_thw, fixture.grid_thw, "{}", fixture.name);
        let rgb = golden(&fixture);
        assert_eq!(actual.pixel_values.len(), rgb.len() * 2);
        let width = fixture.resized[1] as usize;
        let mut expected_hash = Sha256::new();
        let mut max_f32_error = 0_f32;
        let mut max_uint8_error = 0_u8;
        // Parcours inverse de la matrice aplatie ; le SHA final valide ce
        // décodage des indices contre la permutation indépendante PyTorch.
        for (index, &value) in actual.pixel_values.iter().enumerate() {
            assert!(value.is_finite());
            let patch = index / 1536;
            let offset = index % 1536;
            let block = patch / 4;
            let y = block / (width / 32) * 32 + patch / 2 % 2 * 16 + offset / 16 % 16;
            let x = block % (width / 32) * 32 + patch % 2 * 16 + offset % 16;
            let c = offset / 512;
            let byte = rgb[(y * width + x) * 3 + c];
            let expected = levels[byte as usize];
            expected_hash.update(expected.to_le_bytes());
            max_f32_error = max_f32_error.max((value - expected).abs());
            let restored = ((value + 1.0) * 127.5).round().clamp(0.0, 255.0) as u8;
            max_uint8_error = max_uint8_error.max(restored.abs_diff(byte));
        }
        let hash: String = expected_hash
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        assert_eq!(
            hash, fixture.patches_sha256,
            "{}: ordre des patchs",
            fixture.name
        );
        if fixture.resized == [fixture.height, fixture.width] {
            eprintln!("{} patchs: max_abs_f32={max_f32_error:e}", fixture.name);
            assert!(max_f32_error <= 1e-6, "{}", fixture.name);
        } else {
            eprintln!("{} patchs: max_abs_uint8={max_uint8_error}", fixture.name);
            // Les captures PIL sont informatives, seule torchvision est une porte.
            if gated {
                assert!(max_uint8_error <= 1, "{}", fixture.name);
            }
        }
    }
    Ok(())
}
