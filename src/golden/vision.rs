//! Oracle LLM sur le vocabulaire entier ; seuils figés avant comparaison Rust.
use super::external::TestResult;
#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::vision::prompt::SuppliedImage;
#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::{CausalDecoder, Tensor};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

#[derive(Deserialize)]
struct Inputs {
    cases: Vec<Input>,
}
#[derive(Deserialize)]
struct Input {
    input_ids: Vec<usize>,
    image_grid_thw: Vec<[usize; 3]>,
    image_spans: Vec<Span>,
    forced_decode_ids: Vec<usize>,
}
#[derive(Deserialize)]
struct Span {
    start: usize,
    length: usize,
    embedding_file: String,
}
#[derive(Deserialize)]
struct Calibration {
    schema_version: u32,
    mrope_inputs_sha256: String,
    config_sha256: String,
    embedding_sha256: Vec<SourceHash>,
    cases: Vec<CalibratedCase>,
}
#[derive(Deserialize)]
struct SourceHash {
    path: String,
    sha256: String,
}
#[derive(Deserialize)]
struct CalibratedCase {
    shape: [usize; 2],
    payloads: Payloads,
    noise: Metrics,
    threshold: Metrics,
}
#[derive(Deserialize)]
struct Payloads {
    f32: Payload,
    bf16: Payload,
}
#[derive(Deserialize)]
struct Payload {
    file: String,
    sha256: String,
}
#[derive(Deserialize)]
struct Metrics {
    #[serde(deserialize_with = "exact_f64s")]
    rms: Vec<f64>,
    #[serde(deserialize_with = "exact_f64s")]
    max_abs: Vec<f64>,
    #[serde(deserialize_with = "exact_f64s")]
    max_probability: Vec<f64>,
}

fn exact_f64s<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<Vec<f64>, D::Error> {
    // Conserver l'arrondi IEEE du décimal Python. Le parseur rapide JSON peut
    // perdre un ulp et casser l'égalité exacte seuil = 2 × bruit.
    Vec::<Box<serde_json::value::RawValue>>::deserialize(deserializer)?
        .into_iter()
        .map(|number| {
            number
                .get()
                .parse::<f64>()
                .map_err(serde::de::Error::custom)
        })
        .collect()
}

#[test]
fn calibration_metrics_keep_decimal_roundtrip_bits() -> TestResult {
    let metrics: Metrics = serde_json::from_str(
        r#"{"rms":[0.17336344718933105],"max_abs":[0.3467268943786621],"max_probability":[0.0]}"#,
    )?;
    assert_eq!(metrics.rms[0], 0.17336344718933105_f64);
    assert_eq!(metrics.max_abs[0], 2.0 * metrics.rms[0]);
    Ok(())
}

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/vision/Qwen3.8-27B-BF16")
}
fn read_payload(root: &Path, payload: &Payload) -> TestResult<Vec<f32>> {
    let bytes = std::fs::read(root.join(&payload.file))?;
    assert_eq!(crate::hex(Sha256::digest(&bytes)), payload.sha256);
    assert_eq!(bytes.len() % 4, 0);
    Ok(bytes
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().expect("invariant: four bytes")))
        .collect())
}
fn validate(case: &CalibratedCase) {
    assert_eq!(case.shape[0], 17);
    for (noise, threshold) in [
        (&case.noise.rms, &case.threshold.rms),
        (&case.noise.max_abs, &case.threshold.max_abs),
        (&case.noise.max_probability, &case.threshold.max_probability),
    ] {
        assert_eq!(noise.len(), 17);
        assert_eq!(threshold.len(), 17);
        for (&n, &t) in noise.iter().zip(threshold) {
            assert!(n.is_finite() && n >= 0.0);
            assert_eq!(t, 2.0 * n, "seuil modifié après calibration");
        }
    }
}
fn softmax(values: &[f32]) -> Vec<f64> {
    assert!(values.iter().all(|v| v.is_finite()));
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let exps: Vec<_> = values.iter().map(|&v| (f64::from(v) - max).exp()).collect();
    let z: f64 = exps.iter().sum();
    exps.into_iter().map(|v| v / z).collect()
}
fn measure(reference: &[f32], actual: &[f32]) -> [f64; 3] {
    assert_eq!(actual.len(), reference.len());
    let mut square = 0.0f64;
    let mut max = 0.0f64;
    for (&a, &b) in actual.iter().zip(reference) {
        let error = (f64::from(a) - f64::from(b)).abs();
        assert!(error.is_finite());
        square += error * error;
        max = max.max(error);
    }
    let rms = (square / actual.len() as f64).sqrt();
    let probability = softmax(actual)
        .iter()
        .zip(softmax(reference))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    [rms, max, probability]
}
fn compare(reference: &[f32], actual: &[f32], thresholds: &Metrics, step: usize) {
    let [rms, max, probability] = measure(reference, actual);
    eprintln!("vision step={step} rms={rms} max={max} probability={probability}");
    assert!(
        rms <= thresholds.rms[step],
        "RMS {rms} > {}",
        thresholds.rms[step]
    );
    assert!(
        max <= thresholds.max_abs[step],
        "max {max} > {}",
        thresholds.max_abs[step]
    );
    assert!(
        probability <= thresholds.max_probability[step],
        "probability {probability} > {}",
        thresholds.max_probability[step]
    );
}

#[test]
fn calibration_is_frozen() -> TestResult {
    let path = root().join("llm-calibration.json");
    let calibration: Calibration = serde_json::from_slice(&std::fs::read(path)?)?;
    assert_eq!(calibration.schema_version, 1);
    assert_eq!(calibration.cases.len(), 6);
    assert_eq!(calibration.config_sha256.len(), 64);
    assert_eq!(calibration.embedding_sha256.len(), 3);
    for entry in &calibration.embedding_sha256 {
        assert_eq!(
            crate::hex(Sha256::digest(std::fs::read(root().join(&entry.path))?)),
            entry.sha256
        );
    }
    assert_eq!(
        calibration.mrope_inputs_sha256,
        crate::hex(Sha256::digest(std::fs::read(
            root().join("mrope-inputs.json")
        )?))
    );
    for case in calibration.cases {
        validate(&case);
        let reference = read_payload(&root(), &case.payloads.f32)?;
        let noisy = read_payload(&root(), &case.payloads.bf16)?;
        for payload in [&reference, &noisy] {
            assert_eq!(payload.len(), case.shape.iter().product::<usize>());
        }
        for (step, (reference, noisy)) in reference
            .chunks_exact(case.shape[1])
            .zip(noisy.chunks_exact(case.shape[1]))
            .enumerate()
        {
            for (actual, recorded) in measure(reference, noisy).into_iter().zip([
                case.noise.rms[step],
                case.noise.max_abs[step],
                case.noise.max_probability[step],
            ]) {
                // Tolérance de sommation f64 NumPy/Rust, pas de modification
                // du seuil d'acceptation des logits Rust.
                assert!((actual - recorded).abs() <= 1e-12 * recorded.abs().max(1.0));
            }
        }
    }
    Ok(())
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
#[ignore = "poids locaux, calibration préalable et verrou GPU requis"]
fn vision_llm_logits_against_frozen_transformers() -> TestResult {
    calibration_is_frozen()?;
    let calibration: Calibration =
        serde_json::from_slice(&std::fs::read(root().join("llm-calibration.json"))?)?;
    let inputs: Inputs = serde_json::from_slice(&std::fs::read(root().join("mrope-inputs.json"))?)?;
    assert_eq!(inputs.cases.len(), 6);
    assert_eq!(calibration.cases.len(), 6);
    let model_path = PathBuf::from(std::env::var("SARAGOSSA_QWEN38_BF16_DIR")?);
    assert_eq!(
        crate::hex(Sha256::digest(std::fs::read(
            model_path.join("config.json")
        )?)),
        calibration.config_sha256,
        "la calibration BF16 exige le checkpoint natif, sans quantification"
    );
    let assets = crate::ModelAssets::load_local(model_path)?;
    let executor = crate::MetalExecutor::new()?;
    let model: CausalDecoder =
        crate::load_causal_decoder_for_metal(&assets, &executor)?.with_metal_executor(executor)?;
    for (index, (case, calibrated)) in inputs.cases.iter().zip(&calibration.cases).enumerate() {
        validate(calibrated);
        let reference = read_payload(&root(), &calibrated.payloads.f32)?;
        let embeddings: Vec<_> = case
            .image_spans
            .iter()
            .map(|span| -> TestResult<Tensor> {
                let bytes = std::fs::read(root().join(&span.embedding_file))?;
                let values: Vec<_> = bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().expect("invariant: four bytes")))
                    .collect();
                Ok(Tensor::from_vec(
                    vec![span.length, values.len() / span.length],
                    values,
                )?)
            })
            .collect::<TestResult<_>>()?;
        let images: Vec<_> = case
            .image_spans
            .iter()
            .zip(&case.image_grid_thw)
            .zip(&embeddings)
            .map(|((span, &grid), embedding)| SuppliedImage {
                start: span.start,
                grid,
                embeddings: embedding,
                content_hash: [0; 32],
                parameters: b"versioned-fixture",
            })
            .collect();
        let pad = case.input_ids[case.image_spans[0].start];
        for (resident, chunk) in [(false, usize::MAX), (true, usize::MAX), (true, 37)] {
            eprintln!("vision case={index} resident={resident} chunk={chunk}");
            let (mut cache, logits) =
                model.prefill_supplied_vision(&case.input_ids, pad, &images, chunk, resident)?;
            compare(
                &reference[..calibrated.shape[1]],
                logits.data(),
                &calibrated.threshold,
                0,
            );
            let actual =
                model.vision_forced_logits(&mut cache, &case.forced_decode_ids, resident)?;
            for (step, logits) in actual.iter().enumerate() {
                let start = (step + 1) * calibrated.shape[1];
                compare(
                    &reference[start..start + calibrated.shape[1]],
                    logits,
                    &calibrated.threshold,
                    step + 1,
                );
            }
        }
    }
    Ok(())
}
