//! Tests synthétiques de `saragossa convert` (aucun poids réel, CPU seul).

use std::collections::BTreeMap;
use std::path::Path;

use safetensors::{serialize_to_file, tensor::TensorView, Dtype, SafeTensors};

use super::{convert_hf, HfConvertOptions};
use crate::quantization::{bf16_bits, bf16_value, quantize_affine_bitstream_bf16};

const SHARDS: [&str; 2] = [
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
];

fn tensors() -> [(usize, &'static str, Vec<usize>); 10] {
    [
        (0, "model.language_model.embed_tokens.weight", vec![16, 128]),
        (
            0,
            "model.language_model.layers.0.mlp.up_proj.weight",
            vec![8, 128],
        ),
        (
            0,
            "model.language_model.layers.0.input_layernorm.weight",
            vec![128],
        ),
        (
            0,
            "model.language_model.layers.0.linear_attn.conv1d.weight",
            vec![8, 1, 4],
        ),
        (0, "model.visual.blocks.0.attn.qkv.weight", vec![8, 128]),
        (
            1,
            "model.language_model.layers.0.mlp.down_proj.weight",
            vec![8, 96],
        ),
        (1, "lm_head.weight", vec![16, 128]),
        (1, "mtp.fc.weight", vec![8, 256]),
        (1, "mtp.layers.0.mlp.down_proj.weight", vec![8, 128]),
        (1, "mtp.norm.weight", vec![128]),
    ]
}

fn values(shape: &[usize], seed: usize) -> Vec<f32> {
    (0..shape.iter().product::<usize>())
        .map(|i| bf16_value(bf16_bits(((i + seed * 37) as f32 * 0.173).sin() * 0.5)))
        .collect()
}

fn write_source(dir: &Path) {
    let mut weight_map = serde_json::Map::new();
    for (shard, file) in SHARDS.iter().enumerate() {
        let owned: Vec<(String, Vec<usize>, Vec<u8>)> = tensors()
            .into_iter()
            .enumerate()
            .filter(|(_, (s, _, _))| *s == shard)
            .map(|(seed, (_, name, shape))| {
                let bytes = values(&shape, seed)
                    .iter()
                    .flat_map(|v| bf16_bits(*v).to_le_bytes())
                    .collect();
                weight_map.insert(name.to_string(), (*file).into());
                (name.to_string(), shape, bytes)
            })
            .collect();
        let views = owned.iter().map(|(name, shape, bytes)| {
            (
                name.clone(),
                TensorView::new(Dtype::BF16, shape.clone(), bytes)
                    .expect("invariant: vue BF16 valide"),
            )
        });
        serialize_to_file(views, None, &dir.join(file)).expect("invariant: shard écrit");
    }
    let index = serde_json::json!({ "metadata": {}, "weight_map": weight_map });
    std::fs::write(dir.join("model.safetensors.index.json"), index.to_string())
        .expect("invariant: index écrit");
    std::fs::write(dir.join("config.json"), r#"{"model_type":"qwen3_5"}"#)
        .expect("invariant: config écrite");
    std::fs::write(dir.join("tokenizer.json"), "{\"pre_tokenizer\":null}\n")
        .expect("invariant: tokenizer écrit");
}

fn options(src: &Path, dst: &Path) -> HfConvertOptions {
    HfConvertOptions {
        src: src.to_path_buf(),
        dst: dst.to_path_buf(),
        bits: 4,
        minmax: false,
        keep_8bit: Vec::new(),
        keep_bf16: Vec::new(),
        verify: false,
    }
}

/// Tous les fichiers de sortie (journal compris), par nom.
fn snapshot(dir: &Path) -> BTreeMap<String, Vec<u8>> {
    let mut out = BTreeMap::new();
    for sub in [dir.to_path_buf(), dir.join(".convert")] {
        for entry in std::fs::read_dir(&sub).expect("invariant: sortie lisible") {
            let path = entry.expect("invariant: entrée").path();
            if path.is_file() {
                let name = path.strip_prefix(dir).expect("invariant: préfixe");
                out.insert(
                    name.display().to_string(),
                    std::fs::read(&path).expect("invariant: fichier lisible"),
                );
            }
        }
    }
    out
}

fn tensor_bytes(dir: &Path, file: &str, name: &str) -> (Dtype, Vec<usize>, Vec<u8>) {
    let bytes = std::fs::read(dir.join(file)).expect("invariant: shard lisible");
    let st = SafeTensors::deserialize(&bytes).expect("invariant: shard valide");
    let view = st.tensor(name).expect("invariant: tenseur présent");
    (view.dtype(), view.shape().to_vec(), view.data().to_vec())
}

fn config(dir: &Path) -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(dir.join("config.json")).expect("invariant: config"))
        .expect("invariant: config JSON")
}

#[test]
fn minmax_is_bit_identical_to_bitstream_and_deterministic() {
    let root = tempfile::tempdir().expect("invariant: dossier temporaire");
    let src = root.path().join("src");
    std::fs::create_dir(&src).expect("invariant: src");
    write_source(&src);
    let mut runs = Vec::new();
    for name in ["a", "b"] {
        let dst = root.path().join(name);
        let mut opts = options(&src, &dst);
        opts.minmax = true;
        let report = convert_hf(&opts).expect("invariant: conversion min-max");
        assert_eq!(report.quantized, BTreeMap::from([(4, 5)]));
        assert_eq!(report.copied, 5);
        runs.push(snapshot(&dst));
    }
    assert_eq!(runs[0], runs[1], "sortie non déterministe");
    let dst = root.path().join("a");
    for (seed, (shard, name, shape)) in tensors().into_iter().enumerate() {
        let file = SHARDS[shard];
        let data = values(&shape, seed);
        let quantized = shape.len() == 2 && shape[1] % 64 == 0 && !name.contains("visual");
        if !quantized {
            let (dtype, out_shape, bytes) = tensor_bytes(&dst, file, name);
            let expected: Vec<u8> = data
                .iter()
                .flat_map(|v| bf16_bits(*v).to_le_bytes())
                .collect();
            assert_eq!(
                (dtype, out_shape, bytes),
                (Dtype::BF16, shape, expected),
                "{name}"
            );
            continue;
        }
        let (rows, cols) = (shape[0], shape[1]);
        let mut packed = vec![0_u32; rows * cols / 8];
        let mut scales = vec![0_u16; rows * cols / 64];
        let mut biases = vec![0_u16; rows * cols / 64];
        quantize_affine_bitstream_bf16(&data, cols, 64, 4, &mut packed, &mut scales, &mut biases)
            .expect("invariant: référence min-max");
        let base = name.trim_end_matches(".weight");
        let get = |suffix: &str| tensor_bytes(&dst, file, &format!("{base}.{suffix}")).2;
        assert_eq!(
            get("weight"),
            bytemuck::cast_slice::<u32, u8>(&packed),
            "{name}"
        );
        assert_eq!(
            get("scales"),
            bytemuck::cast_slice::<u16, u8>(&scales),
            "{name}"
        );
        assert_eq!(
            get("biases"),
            bytemuck::cast_slice::<u16, u8>(&biases),
            "{name}"
        );
    }
    // Tête MTP restée dans le shard source, tokenizer recopié à l'octet.
    let index: serde_json::Value = serde_json::from_slice(
        &std::fs::read(dst.join("model.safetensors.index.json")).expect("invariant: index"),
    )
    .expect("invariant: index JSON");
    assert_eq!(index["weight_map"]["mtp.fc.scales"], SHARDS[1]);
    assert_eq!(
        std::fs::read(dst.join("tokenizer.json")).expect("invariant: tokenizer"),
        std::fs::read(src.join("tokenizer.json")).expect("invariant: tokenizer source")
    );
    let quant = &config(&dst)["quantization"];
    assert_eq!(
        (quant["bits"].as_u64(), quant["group_size"].as_u64()),
        (Some(4), Some(64))
    );
}

#[test]
fn least_squares_lowers_error_and_honours_keep_lists_and_resume() {
    let root = tempfile::tempdir().expect("invariant: dossier temporaire");
    let src = root.path().join("src");
    std::fs::create_dir(&src).expect("invariant: src");
    write_source(&src);
    let rel_err = |dst: &Path| -> BTreeMap<String, f64> {
        let manifest: serde_json::Value = serde_json::from_slice(
            &std::fs::read(dst.join("convert_manifest.json")).expect("invariant: manifeste"),
        )
        .expect("invariant: manifeste JSON");
        manifest["tensors"]
            .as_array()
            .expect("invariant: tenseurs")
            .iter()
            .filter_map(|t| Some((t["source"].as_str()?.to_string(), t["rel_err"].as_f64()?)))
            .collect()
    };
    let minmax = root.path().join("minmax");
    let mut opts = options(&src, &minmax);
    opts.minmax = true;
    convert_hf(&opts).expect("invariant: min-max");
    let ls = root.path().join("ls");
    convert_hf(&options(&src, &ls)).expect("invariant: moindres carrés");
    let (before, after) = (rel_err(&minmax), rel_err(&ls));
    assert_eq!(before.len(), 5);
    for (name, err) in &after {
        assert!(*err < before[name], "{name}: {err} ≥ {}", before[name]);
    }

    let mixed = root.path().join("mixed");
    let mut opts = options(&src, &mixed);
    opts.keep_8bit = vec!["mlp.up_proj".to_string()];
    opts.keep_bf16 = vec!["lm_head".to_string()];
    let report = convert_hf(&opts).expect("invariant: mixte");
    assert_eq!(report.quantized, BTreeMap::from([(4, 3), (8, 1)]));
    let up = "model.language_model.layers.0.mlp.up_proj";
    assert_eq!(
        tensor_bytes(&mixed, SHARDS[0], &format!("{up}.weight")).1,
        [8, 32]
    );
    assert_eq!(
        tensor_bytes(&mixed, SHARDS[1], "lm_head.weight").0,
        Dtype::BF16
    );
    let quant = &config(&mixed)["quantization"];
    assert_eq!(quant[up]["bits"].as_u64(), Some(8));
    assert_eq!(quant["lm_head"], false);

    let again = convert_hf(&opts).expect("invariant: reprise");
    assert_eq!(again.units_resumed.len(), 2);
    assert!(again.units_written.is_empty());
    opts.bits = 8;
    assert!(
        convert_hf(&opts).is_err(),
        "reprise avec d'autres options acceptée"
    );
}
