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
    write_source_with(dir, false, "mtp.");
}

/// `split_mtp` déplace `mtp.norm.weight` dans le premier shard ; `mtp`
/// remplace le préfixe `mtp.` (alias `language_model.mtp.`).
fn write_source_with(dir: &Path, split_mtp: bool, mtp: &str) {
    let mut weight_map = serde_json::Map::new();
    for (shard, file) in SHARDS.iter().enumerate() {
        let owned: Vec<(String, Vec<usize>, Vec<u8>)> = tensors()
            .into_iter()
            .enumerate()
            .filter(|(_, (s, name, _))| {
                let s = if split_mtp && *name == "mtp.norm.weight" {
                    0
                } else {
                    *s
                };
                s == shard
            })
            .map(|(seed, (_, name, shape))| {
                let name = match name.strip_prefix("mtp.") {
                    Some(rest) => format!("{mtp}{rest}"),
                    None => name.to_string(),
                };
                let name = name.as_str();
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

/// Bascule le bit de poids faible de chaque valeur bf16 (valeurs finies,
/// en-tête et taille intacts) : mêmes noms et formes, autres poids.
fn perturb_data(path: &Path) {
    let mut bytes = std::fs::read(path).expect("invariant: shard source");
    let header = u64::from_le_bytes(bytes[..8].try_into().expect("invariant: en-tête"));
    let start = 8 + usize::try_from(header).expect("invariant: en-tête borné");
    for low in bytes[start..].iter_mut().step_by(2) {
        *low ^= 1;
    }
    std::fs::write(path, bytes).expect("invariant: shard réécrit");
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
        imatrix: None,
        bits_for: Vec::new(),
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

#[test]
fn mtp_head_is_regrouped_and_stays_u4() {
    for prefix in ["mtp.", "language_model.mtp."] {
        let root = tempfile::tempdir().expect("invariant: dossier temporaire");
        let src = root.path().join("src");
        std::fs::create_dir(&src).expect("invariant: src");
        write_source_with(&src, true, prefix);
        let dst = root.path().join("u8");
        let mut opts = options(&src, &dst);
        opts.bits = 8;
        convert_hf(&opts).expect("invariant: conversion u8");
        let index: serde_json::Value = serde_json::from_slice(
            &std::fs::read(dst.join("model.safetensors.index.json")).expect("invariant: index"),
        )
        .expect("invariant: index JSON");
        let map = index["weight_map"]
            .as_object()
            .expect("invariant: weight_map");
        let mtp: Vec<_> = map.iter().filter(|(k, _)| k.starts_with(prefix)).collect();
        assert_eq!(mtp.len(), 7, "{prefix}: {mtp:?}");
        assert!(mtp.iter().all(|(_, file)| *file == SHARDS[1]), "{mtp:?}");
        // u4 : 256 colonnes → 32 mots ; le tronc suit --bits 8 (128 → 32 mots).
        let fc = format!("{prefix}fc");
        assert_eq!(
            tensor_bytes(&dst, SHARDS[1], &format!("{fc}.weight")).1,
            [8, 32]
        );
        let up = "model.language_model.layers.0.mlp.up_proj.weight";
        assert_eq!(tensor_bytes(&dst, SHARDS[0], up).1, [8, 32]);
        assert_eq!(config(&dst)["quantization"][&fc]["bits"].as_u64(), Some(4));

        let mut opts = options(&src, &root.path().join("refus"));
        opts.keep_8bit = vec![prefix.to_string()];
        let error = convert_hf(&opts).expect_err("tête MTP 8 bits acceptée");
        assert!(error.to_string().contains("MTP"), "{error}");
    }
}

#[test]
fn resume_refuses_rewritten_source_weights() {
    let root = tempfile::tempdir().expect("invariant: dossier temporaire");
    let src = root.path().join("src");
    std::fs::create_dir(&src).expect("invariant: src");
    write_source(&src);
    let dst = root.path().join("dst");
    convert_hf(&options(&src, &dst)).expect("invariant: première conversion");
    // Mêmes noms, formes, index et config : seules les valeurs changent.
    // Mêmes octets d'en-tête et mêmes dates : seul le contenu trahit le changement.
    let dates: Vec<_> = SHARDS
        .iter()
        .map(|file| {
            std::fs::metadata(src.join(file))
                .and_then(|m| m.modified())
                .expect("invariant: date source")
        })
        .collect();
    for (file, date) in SHARDS.iter().zip(dates) {
        perturb_data(&src.join(file));
        std::fs::File::options()
            .write(true)
            .open(src.join(file))
            .and_then(|f| f.set_modified(date))
            .expect("invariant: date restaurée");
    }
    let error = convert_hf(&options(&src, &dst)).expect_err("reprise sur d'autres poids acceptée");
    assert!(error.to_string().contains("shards source"), "{error}");
}

/// Imatrix fixture : un vecteur par linéaire quantifié hors embedding et MTP.
fn write_imatrix(path: &Path, value: impl Fn(usize) -> f32) {
    let vectors: Vec<(&str, Vec<u8>)> = [
        "language_model.model.layers.0.mlp.up_proj",
        "language_model.lm_head",
    ]
    .into_iter()
    .map(|name| {
        let bytes = (0..128).flat_map(|j| value(j).to_le_bytes()).collect();
        (name, bytes)
    })
    .collect();
    let views: Vec<(&str, TensorView<'_>)> = vectors
        .iter()
        .map(|(name, bytes)| {
            let view =
                TensorView::new(Dtype::F32, vec![128], bytes).expect("invariant: vecteur valide");
            (*name, view)
        })
        .collect();
    serialize_to_file(views, None, path).expect("invariant: imatrix écrite");
}

/// Imatrix uniforme : shards identiques à l'octet au fit non pondéré ; imatrix
/// non uniforme : autres codes ; `--bits-for` produit du u5 déclaré dans la config.
#[test]
fn imatrix_uniform_is_identity_and_bits_for_overrides() {
    let root = tempfile::tempdir().expect("invariant: dossier temporaire");
    let src = root.path().join("src");
    std::fs::create_dir(&src).expect("invariant: src");
    write_source(&src);
    let shards = |dir: &Path| -> Vec<Vec<u8>> {
        SHARDS
            .iter()
            .map(|f| std::fs::read(dir.join(f)).expect("invariant: shard"))
            .collect()
    };
    let plain = root.path().join("plain");
    convert_hf(&options(&src, &plain)).expect("invariant: sans imatrix");

    let ones = root.path().join("ones.safetensors");
    write_imatrix(&ones, |_| 1.0);
    let uniform = root.path().join("uniform");
    let mut opts = options(&src, &uniform);
    opts.imatrix = Some(ones);
    convert_hf(&opts).expect("invariant: imatrix uniforme");
    assert_eq!(shards(&uniform), shards(&plain));

    let skewed = root.path().join("skewed.safetensors");
    // Importance variable dans chaque groupe (constante par groupe = même optimum).
    write_imatrix(&skewed, |j| if j % 3 == 0 { 100.0 } else { 0.01 });
    let weighted = root.path().join("weighted");
    let mut opts = options(&src, &weighted);
    opts.imatrix = Some(skewed);
    opts.bits_for = vec![("mlp.up_proj".to_string(), 5)];
    convert_hf(&opts).expect("invariant: imatrix pondérée");
    let up = "model.language_model.layers.0.mlp.up_proj";
    assert_eq!(
        tensor_bytes(&weighted, SHARDS[0], &format!("{up}.weight")).1,
        [8, 20]
    );
    assert_eq!(
        config(&weighted)["quantization"][up]["bits"].as_u64(),
        Some(5)
    );
    let lm_head = |dir: &Path| {
        ["weight", "scales", "biases"]
            .map(|suffix| tensor_bytes(dir, SHARDS[1], &format!("lm_head.{suffix}")))
    };
    assert_ne!(lm_head(&weighted), lm_head(&plain));

    let missing = root.path().join("missing.safetensors");
    serialize_to_file(Vec::<(&str, TensorView<'_>)>::new(), None, &missing)
        .expect("invariant: imatrix vide");
    let mut opts = options(&src, &root.path().join("refused"));
    opts.imatrix = Some(missing);
    assert!(convert_hf(&opts).is_err(), "module sans imatrix accepté");
}

/// Source MoE d'un seul shard (experts fusionnés transformers, tête MTP MoE).
fn write_moe_source(dir: &Path) -> Vec<(&'static str, Vec<usize>, Vec<f32>)> {
    let tensors: Vec<(&str, Vec<usize>)> = vec![
        (
            "model.language_model.layers.0.mlp.experts.gate_up_proj",
            vec![3, 128, 128],
        ),
        (
            "model.language_model.layers.0.mlp.experts.down_proj",
            vec![3, 128, 64],
        ),
        (
            "model.language_model.layers.0.mlp.gate.weight",
            vec![3, 128],
        ),
        ("mtp.layers.0.mlp.experts.gate_up_proj", vec![3, 128, 128]),
        ("mtp.fc.weight", vec![8, 256]),
    ];
    let owned: Vec<_> = tensors
        .into_iter()
        .enumerate()
        .map(|(seed, (name, shape))| {
            let data = values(&shape, seed);
            (name, shape, data)
        })
        .collect();
    let bytes: Vec<Vec<u8>> = owned
        .iter()
        .map(|(_, _, data)| {
            data.iter()
                .flat_map(|v| bf16_bits(*v).to_le_bytes())
                .collect()
        })
        .collect();
    let views = owned.iter().zip(&bytes).map(|((name, shape, _), bytes)| {
        (
            (*name).to_string(),
            TensorView::new(Dtype::BF16, shape.clone(), bytes).expect("invariant: vue BF16"),
        )
    });
    let file = "model.safetensors";
    serialize_to_file(views, None, &dir.join(file)).expect("invariant: shard écrit");
    let weight_map: serde_json::Map<_, _> = owned
        .iter()
        .map(|(name, _, _)| ((*name).to_string(), file.into()))
        .collect();
    let index = serde_json::json!({ "metadata": {}, "weight_map": weight_map });
    std::fs::write(dir.join("model.safetensors.index.json"), index.to_string())
        .expect("invariant: index écrit");
    std::fs::write(dir.join("config.json"), r#"{"model_type":"qwen3_5_moe"}"#)
        .expect("invariant: config écrite");
    std::fs::write(dir.join("tokenizer.json"), "{}\n").expect("invariant: tokenizer écrit");
    owned
}

/// Référence min-max d'une matrice `[rows, cols]` : (codes, scales, biais).
fn minmax_ref(data: &[f32], cols: usize, bits: usize) -> [Vec<u8>; 3] {
    let rows = data.len() / cols;
    let mut packed = vec![0_u32; rows * cols * bits / 32];
    let mut scales = vec![0_u16; rows * cols / 64];
    let mut biases = vec![0_u16; rows * cols / 64];
    quantize_affine_bitstream_bf16(data, cols, 64, bits, &mut packed, &mut scales, &mut biases)
        .expect("invariant: référence min-max");
    [
        bytemuck::cast_slice(&packed).to_vec(),
        bytemuck::cast_slice(&scales).to_vec(),
        bytemuck::cast_slice(&biases).to_vec(),
    ]
}

#[test]
fn fused_moe_experts_become_switch_mlp_stacks_and_mtp_head_stays_bf16() {
    let root = tempfile::tempdir().expect("invariant: dossier temporaire");
    let src = root.path().join("src");
    std::fs::create_dir(&src).expect("invariant: src");
    let source = write_moe_source(&src);
    let dst = root.path().join("dst");
    let mut opts = options(&src, &dst);
    opts.minmax = true;
    let report = convert_hf(&opts).expect("invariant: conversion MoE");
    // gate_up + down → 2 piles u4, routeur u8 ; tête MTP recopiée.
    assert_eq!(report.quantized, BTreeMap::from([(4, 2), (8, 1)]));
    assert_eq!(report.copied, 2);
    let file = "model.safetensors";
    let layer = "model.language_model.layers.0.mlp.switch_mlp";
    let check = |base: &str, shape: [usize; 3], data: &[f32]| {
        let reference = minmax_ref(data, shape[2], 4);
        for (suffix, expected) in ["weight", "scales", "biases"].iter().zip(&reference) {
            let (_, out_shape, bytes) = tensor_bytes(&dst, file, &format!("{base}.{suffix}"));
            assert_eq!(&bytes, expected, "{base}.{suffix}");
            assert_eq!(out_shape[..2], shape[..2], "{base}.{suffix}");
        }
    };
    let gate_up = &source[0].2;
    let (inter, hidden) = (64, 128);
    let half = |second: bool| -> Vec<f32> {
        gate_up
            .chunks(2 * inter * hidden)
            .flat_map(|expert| {
                let (gate, up) = expert.split_at(inter * hidden);
                if second { up } else { gate }.to_vec()
            })
            .collect()
    };
    check(
        &format!("{layer}.gate_proj"),
        [3, inter, hidden],
        &half(false),
    );
    check(&format!("{layer}.up_proj"), [3, inter, hidden], &half(true));
    check(
        &format!("{layer}.down_proj"),
        [3, hidden, inter],
        &source[1].2,
    );
    let router = "model.language_model.layers.0.mlp.gate";
    assert_eq!(
        tensor_bytes(&dst, file, &format!("{router}.weight")).2,
        minmax_ref(&source[2].2, 128, 8)[0]
    );
    for (name, shape, data) in &source[3..] {
        let expected: Vec<u8> = data
            .iter()
            .flat_map(|v| bf16_bits(*v).to_le_bytes())
            .collect();
        assert_eq!(
            tensor_bytes(&dst, file, name),
            (Dtype::BF16, shape.clone(), expected),
            "{name}"
        );
    }
    assert_eq!(
        config(&dst)["quantization"][router]["bits"].as_u64(),
        Some(8)
    );

    // `--keep-bf16 .` : référence BF16, piles d'experts recopiées sans perte.
    let bf16 = root.path().join("bf16");
    let mut opts = options(&src, &bf16);
    opts.keep_bf16 = vec![".".to_string()];
    let report = convert_hf(&opts).expect("invariant: conversion bf16");
    assert!(report.quantized.is_empty(), "{:?}", report.quantized);
    let as_bytes =
        |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| bf16_bits(*x).to_le_bytes()).collect() };
    for (proj, shape, data) in [
        ("gate_proj", vec![3, inter, hidden], half(false)),
        ("up_proj", vec![3, inter, hidden], half(true)),
        ("down_proj", vec![3, hidden, inter], source[1].2.clone()),
    ] {
        let name = format!("{layer}.{proj}.weight");
        assert_eq!(
            tensor_bytes(&bf16, file, &name),
            (Dtype::BF16, shape, as_bytes(&data)),
            "{name}"
        );
        assert_eq!(
            config(&bf16)["quantization"][&format!("{layer}.{proj}")],
            false
        );
    }
}

/// Source mono-fichier sans index (Llama 3.2 1B : `model.safetensors` seul,
/// embeddings liés) : conversion, reprise à l'identique, refus après altération.
#[test]
fn single_file_source_without_index_converts_and_resumes() {
    let root = tempfile::tempdir().expect("invariant: dossier temporaire");
    let src = root.path().join("src");
    std::fs::create_dir(&src).expect("invariant: src");
    let owned: Vec<(&str, Vec<usize>, Vec<u8>)> = [
        ("model.embed_tokens.weight", vec![16, 128]),
        ("model.layers.0.self_attn.q_proj.weight", vec![8, 128]),
        ("model.layers.0.input_layernorm.weight", vec![128]),
    ]
    .into_iter()
    .enumerate()
    .map(|(seed, (name, shape))| {
        let bytes = values(&shape, seed)
            .iter()
            .flat_map(|v| bf16_bits(*v).to_le_bytes())
            .collect();
        (name, shape, bytes)
    })
    .collect();
    let views = owned.iter().map(|(name, shape, bytes)| {
        (
            (*name).to_string(),
            TensorView::new(Dtype::BF16, shape.clone(), bytes).expect("invariant: vue BF16"),
        )
    });
    let single = src.join("model.safetensors");
    serialize_to_file(views, None, &single).expect("invariant: fichier écrit");
    std::fs::write(src.join("config.json"), r#"{"model_type":"llama"}"#)
        .expect("invariant: config écrite");
    std::fs::write(src.join("tokenizer.json"), "{\"pre_tokenizer\":null}\n")
        .expect("invariant: tokenizer écrit");

    let dst = root.path().join("dst");
    let first = convert_hf(&options(&src, &dst)).expect("invariant: conversion mono-fichier");
    assert_eq!(first.units_written, ["model.safetensors"]);
    let index: serde_json::Value = serde_json::from_slice(
        &std::fs::read(dst.join("model.safetensors.index.json")).expect("invariant: index"),
    )
    .expect("invariant: index JSON");
    assert_eq!(
        index["weight_map"]["model.embed_tokens.weight"],
        "model.safetensors"
    );
    assert_eq!(
        tensor_bytes(&dst, "model.safetensors", "model.embed_tokens.weight").1,
        [16, 16]
    );
    let before = snapshot(&dst);
    let again = convert_hf(&options(&src, &dst)).expect("invariant: reprise");
    assert_eq!(again.units_resumed, ["model.safetensors"]);
    assert_eq!(snapshot(&dst), before);

    perturb_data(&single);
    let error = convert_hf(&options(&src, &dst)).expect_err("reprise sur d'autres poids acceptée");
    assert!(error.to_string().contains("shards source"), "{error}");
}
