//! Tests synthétiques du convertisseur (aucun poids réel).

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::path::Path;

use safetensors::{serialize_to_file, Dtype, View};

use super::ngram::{head_vocab_sizes, layer_multipliers, NGRAM_SEED};
use super::rules::{classify, expected_source_names, ExpertGroup, LayerType, Proj, Role};
use super::{convert_qwen4exp, ConvertOptions};
use crate::quantization::bf16_bits;
use crate::qwen_loader::read_shard_header;
use crate::safetensor::f8_e4m3_to_f32;

const H: usize = 128;
/// 256 lignes : les experts gate/up couvrent deux blocs d'échelle FP8 (128).
const I: usize = 256;
const E: usize = 2;
const VOCAB: usize = 64;
const PARTS: usize = 12;
const NGRAM_WIDTH: usize = 32;
const REVISION: &str = "0000000000000000000000000000000000000000";

/// Altération d'une source synthétique.
type Mutation = Box<dyn Fn(&mut BTreeMap<String, Raw>)>;

struct Raw {
    dtype: Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl View for &Raw {
    fn dtype(&self) -> Dtype {
        self.dtype
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(&self.data)
    }
    fn data_len(&self) -> usize {
        self.data.len()
    }
}

fn bf16(shape: &[usize], seed: usize) -> Raw {
    let count: usize = shape.iter().product();
    let data = (0..count)
        .flat_map(|i| {
            let x = ((i + seed * 31) as f32 * 0.173).sin() * 0.5;
            bf16_bits(x).to_le_bytes()
        })
        .collect();
    Raw {
        dtype: Dtype::BF16,
        shape: shape.to_vec(),
        data,
    }
}

/// Octets E4M3 déterministes sans NaN (`0x7f`/`0xff` exclus).
fn e4m3_bytes(count: usize, seed: usize) -> Vec<u8> {
    (0..count)
        .map(|i| {
            let byte = ((i * 37 + seed * 101) % 254) as u8;
            if byte >= 0x7f {
                byte + 1
            } else {
                byte
            }
        })
        .collect()
}

fn i64s(values: &[i64]) -> Raw {
    Raw {
        dtype: Dtype::I64,
        shape: vec![values.len()],
        data: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    }
}

fn layer_types() -> Vec<LayerType> {
    vec![LayerType::Linear, LayerType::Full]
}

fn ngram_params() -> (Vec<i64>, Vec<i64>, Vec<i64>, usize) {
    let mult = layer_multipliers(1000, 3, 0, NGRAM_SEED);
    let sizes = head_vocab_sizes(50, 2);
    let offsets = vec![0, sizes[0]];
    let total = (sizes[0] + sizes[1]) as usize;
    let padded = total.div_ceil(PARTS) * PARTS;
    (mult, sizes, offsets, padded / PARTS)
}

fn tensor_for(name: &str, seed: usize) -> Raw {
    let shape: Vec<usize> = match name {
        n if n.ends_with("embed_tokens.weight") || n == "lm_head.weight" => vec![VOCAB, H],
        n if n.ends_with("block_inject_weight.weight") => vec![4, H],
        n if n.ends_with("conv1d.weight") => vec![H, 1, 4],
        n if n.ends_with("A_log") || n.ends_with("dt_bias") => vec![4],
        n if n.ends_with("in_proj_a.weight") || n.ends_with("in_proj_b.weight") => vec![4, H],
        n if n.ends_with("mlp.gate.weight") => vec![E, H],
        n if n.ends_with("shared_expert_gate.weight") => vec![1, H],
        n if n.ends_with("norm.weight") || n.ends_with("layernorm.weight") => vec![H],
        n if n.ends_with("norm_conv.weight")
            || n.ends_with("norm_key.weight")
            || n.ends_with("norm_query.weight")
            || n.ends_with("norm_embedding.weight")
            || n.ends_with("norm_hidden.weight") =>
        {
            vec![H]
        }
        _ => vec![64, H],
    };
    bf16(&shape, seed)
}

/// Écrit un checkpoint `qwen4_exp` miniature, réparti sur deux shards.
fn write_source(dir: &Path, mutate: impl FnOnce(&mut BTreeMap<String, Raw>)) {
    let mut tensors: BTreeMap<String, Raw> = BTreeMap::new();
    for (seed, name) in expected_source_names(&layer_types(), Some(LayerType::Full), 1)
        .into_iter()
        .enumerate()
    {
        let raw = if name.ends_with("weight_scale") {
            Raw {
                dtype: Dtype::BF16,
                shape: vec![1],
                data: bf16_bits(0.01).to_le_bytes().to_vec(),
            }
        } else if name.ends_with("layer_multipliers") {
            i64s(&ngram_params().0)
        } else if name.ends_with("ngram_heads_vocab_sizes") {
            i64s(&ngram_params().1)
        } else if name.ends_with("ngram_heads_offsets") {
            i64s(&ngram_params().2)
        } else {
            tensor_for(&name, seed)
        };
        tensors.insert(name, raw);
    }
    let rows_per_part = ngram_params().3;
    for part in 0..PARTS {
        tensors.insert(
            format!("model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{part}.weight"),
            Raw {
                dtype: Dtype::F8_E4M3,
                shape: vec![rows_per_part, NGRAM_WIDTH],
                data: e4m3_bytes(rows_per_part * NGRAM_WIDTH, part),
            },
        );
    }
    let prefixes = [
        "model.language_model.layers.0.".to_string(),
        "model.language_model.layers.1.".to_string(),
        "mtp.layers.0.".to_string(),
    ];
    for (group, prefix) in prefixes.iter().enumerate() {
        for expert in 0..E {
            for (proj, shape) in [
                ("gate_proj", [I, H]),
                ("up_proj", [I, H]),
                ("down_proj", [H, I]),
            ] {
                let base = format!("{prefix}mlp.experts.{expert}.{proj}");
                tensors.insert(
                    format!("{base}.weight"),
                    Raw {
                        dtype: Dtype::F8_E4M3,
                        shape: shape.to_vec(),
                        data: e4m3_bytes(shape[0] * shape[1], group * 10 + expert),
                    },
                );
                let scale = [shape[0].div_ceil(128), shape[1].div_ceil(128)];
                let value = 0.001 * (1 + expert + group) as f32;
                // Échelle distincte par bloc : un mauvais indexage des blocs se voit.
                let block_scale = |block: usize| value * (1 + block) as f32;
                tensors.insert(
                    format!("{base}.weight_scale_inv"),
                    Raw {
                        dtype: Dtype::BF16,
                        shape: scale.to_vec(),
                        data: (0..scale[0] * scale[1])
                            .flat_map(|block| bf16_bits(block_scale(block)).to_le_bytes())
                            .collect(),
                    },
                );
            }
        }
    }
    tensors.insert(
        "model.visual.patch_embed.proj.weight".to_string(),
        bf16(&[4, 4], 7),
    );
    mutate(&mut tensors);

    let names: Vec<String> = tensors.keys().cloned().collect();
    let (first, second) = names.split_at(names.len() / 2);
    let mut weight_map = serde_json::Map::new();
    for (file, part) in [
        ("model-00001-of-00002.safetensors", first),
        ("model-00002-of-00002.safetensors", second),
    ] {
        let views: Vec<(&str, &Raw)> = part.iter().map(|n| (n.as_str(), &tensors[n])).collect();
        serialize_to_file(views, None, &dir.join(file)).expect("invariant: écriture du shard test");
        for name in part {
            weight_map.insert(name.clone(), file.into());
        }
    }
    let index = serde_json::json!({ "metadata": {}, "weight_map": weight_map });
    std::fs::write(dir.join("model.safetensors.index.json"), index.to_string())
        .expect("invariant: index test");
    let config = serde_json::json!({
        "model_type": "qwen4_exp",
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "vision_config": { "depth": 1 },
        "image_token_id": 1,
        "quantization_config": { "quant_method": "fp8" },
        "text_config": {
            "num_hidden_layers": 2,
            "layer_types": ["linear_attention", "full_attention"],
            "num_experts": E,
            "mtp_num_hidden_layers": 1,
            "mtp": { "layer_types": ["full_attention"] },
            "split_ngram_parts": PARTS,
            "vocab_size": 1000,
            "ngram_size": 3,
            "heads_per_ngram": 1,
            "ngram_vocab_size_base": 50,
            "make_ngram_vocab_size_divisible_by": PARTS,
            "ple_layer_ids": [2],
        }
    });
    std::fs::write(dir.join("config.json"), config.to_string()).expect("invariant: config test");
    for side in ["tokenizer.json", "tokenizer_config.json", "LICENSE"] {
        std::fs::write(dir.join(side), side).expect("invariant: fichier annexe test");
    }
}

fn options(src: &Path, dst: &Path) -> ConvertOptions {
    ConvertOptions {
        src: src.to_path_buf(),
        dst: dst.to_path_buf(),
        source_revision: REVISION.to_string(),
        source_repo: Some("test/qwen4-exp-mini".to_string()),
        converter_commit: Some("cafe".to_string()),
        only: None,
        verify: false,
        plan_only: false,
    }
}

fn read_tensor(path: &Path, name: &str) -> (Vec<usize>, Vec<u8>) {
    let header = read_shard_header(path).expect("invariant: en-tête de sortie");
    let entry = &header.tensors[name];
    let bytes = std::fs::read(path).expect("invariant: shard de sortie");
    let start = header.data_start as usize + entry.data_offsets[0];
    let end = header.data_start as usize + entry.data_offsets[1];
    (entry.shape.clone(), bytes[start..end].to_vec())
}

fn u32s(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn bf16s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
        .collect()
}

fn find_shard(dst: &Path, name: &str) -> std::path::PathBuf {
    for entry in std::fs::read_dir(dst).expect("invariant: sortie lisible") {
        let path = entry.expect("invariant: entrée").path();
        if path.extension().is_some_and(|e| e == "safetensors")
            && read_shard_header(&path)
                .expect("invariant: en-tête")
                .tensors
                .contains_key(name)
        {
            return path;
        }
    }
    panic!("tenseur {name} introuvable");
}

#[test]
fn classify_rejects_unknown_and_non_canonical_indices() {
    assert!(matches!(
        classify(
            "model.language_model.layers.3.mlp.experts.17.down_proj.weight",
            1
        ),
        Ok(Role::Expert {
            group: ExpertGroup::Trunk(3),
            expert: 17,
            proj: Proj::Down
        })
    ));
    assert_eq!(
        classify(
            "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_100.weight",
            1
        )
        .ok(),
        Some(Role::NgramShard(100))
    );
    assert_eq!(
        classify("lm_head.weight", 1).ok(),
        Some(Role::Affine { bits: 6 })
    );
    assert_eq!(
        classify("model.visual.blocks.0.attn.qkv.weight", 1).ok(),
        Some(Role::Vision)
    );
    for bad in [
        "model.language_model.layers.3.mlp.experts.017.down_proj.weight",
        "model.language_model.layers.3.some_norm.weight",
        "model.language_model.layers.2.ple.key_proj.weight",
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_1.weight.x",
        "mtp.layers.1.self_attn.q_proj.weight",
        "other.weight",
    ] {
        assert!(classify(bad, 1).is_err(), "{bad}");
    }
}

#[test]
fn converts_synthetic_checkpoint_end_to_end() {
    let src = tempfile::tempdir().expect("invariant: dossier temporaire");
    let dst = tempfile::tempdir().expect("invariant: dossier temporaire");
    write_source(src.path(), |_| {});
    let report = convert_qwen4exp(&options(src.path(), dst.path())).expect("invariant: conversion");
    assert!(report.finalized);
    assert_eq!(report.counts.experts, 3 * 3 * E);
    assert_eq!(report.counts.vision_excluded, 1);
    assert_eq!(report.counts.i64, 3);

    // Config : quantification MLX, vision et FP8 retirés.
    let config: serde_json::Value = serde_json::from_slice(
        &std::fs::read(dst.path().join("config.json")).expect("invariant: config"),
    )
    .expect("invariant: json");
    assert!(config.get("vision_config").is_none());
    assert!(config.get("quantization_config").is_none());
    assert_eq!(config["quantization"]["bits"], 4);
    assert_eq!(config["quantization"]["language_model.lm_head"]["bits"], 6);
    assert_eq!(
        config["quantization"]["language_model.model.layers.0.linear_attn.in_proj_qkv"]["bits"],
        8
    );

    // lm_head u6 : [VOCAB, H·6/32] et scales [VOCAB, H/64].
    let shard = find_shard(dst.path(), "language_model.lm_head.weight");
    let (shape, _) = read_tensor(&shard, "language_model.lm_head.weight");
    assert_eq!(shape, vec![VOCAB, H * 6 / 32]);
    let (shape, _) = read_tensor(&shard, "language_model.lm_head.scales");
    assert_eq!(shape, vec![VOCAB, H / 64]);

    // Expert 1 de la couche 0 : la restitution affine colle à la source FP8.
    let base = "language_model.model.layers.0.mlp.switch_mlp.gate_proj";
    let shard = find_shard(dst.path(), &format!("{base}.weight"));
    let (shape, packed) = read_tensor(&shard, &format!("{base}.weight"));
    assert_eq!(shape, vec![E, I, H * 4 / 32]);
    let (_, scales) = read_tensor(&shard, &format!("{base}.scales"));
    let (_, biases) = read_tensor(&shard, &format!("{base}.biases"));
    let (packed, scales, biases) = (u32s(&packed), bf16s(&scales), bf16s(&biases));
    let source = e4m3_bytes(I * H, 1);
    // Expert 1, couche 0 : échelles 0,002 × (1 + bloc de lignes), une colonne de blocs.
    let src_scale =
        |row: usize| f32::from_bits(u32::from(bf16_bits(0.002 * (1 + row / 128) as f32)) << 16);
    let words_per_expert = I * H / 8;
    let groups_per_expert = I * H / 64;
    for (idx, byte) in source.iter().enumerate() {
        let w = f8_e4m3_to_f32(*byte).expect("invariant: octet fini") * src_scale(idx / H);
        let word = packed[words_per_expert + idx / 8];
        let code = (word >> ((idx % 8) * 4)) & 0xf;
        let g = groups_per_expert + idx / 64;
        let r = code as f32 * scales[g] + biases[g];
        assert!(
            (w - r).abs() <= scales[g] * 0.5 + 1.0e-7,
            "idx={idx} w={w} r={r}"
        );
    }

    // I64 conservés à l'octet ; vision absente.
    let name = "language_model.model.layers.1.ple.ple_embedding.layer_multipliers";
    let (_, bytes) = read_tensor(&find_shard(dst.path(), name), name);
    let values: Vec<i64> = bytes
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().expect("invariant: 8 octets")))
        .collect();
    assert_eq!(values, ngram_params().0);

    // N-gram : concaténation compacte dans l'ordre numérique (shard_10 après shard_9).
    let table = std::fs::read(dst.path().join("ngram/ngram_e4m3.bin")).expect("invariant: table");
    let rows = ngram_params().3;
    let expected: Vec<u8> = (0..PARTS)
        .flat_map(|part| e4m3_bytes(rows * NGRAM_WIDTH, part))
        .collect();
    assert_eq!(table, expected);
    let meta: serde_json::Value = serde_json::from_slice(
        &std::fs::read(dst.path().join("ngram/meta.json")).expect("invariant: meta"),
    )
    .expect("invariant: json");
    assert_eq!(meta["hash"]["seed"], NGRAM_SEED);
    assert_eq!(meta["segments"][10]["offset"], 10 * rows * NGRAM_WIDTH);
    assert!(meta["formula"]
        .as_str()
        .is_some_and(|f| f.contains("weight_scale")));

    let manifest: serde_json::Value = serde_json::from_slice(
        &std::fs::read(dst.path().join("convert_manifest.json")).expect("invariant: manifeste"),
    )
    .expect("invariant: json");
    assert_eq!(
        manifest["excluded_vision"][0],
        "model.visual.patch_embed.proj.weight"
    );
    assert_eq!(manifest["source"]["revision"], REVISION);
    assert_eq!(manifest["source"]["repo"], "test/qwen4-exp-mini");
    assert_eq!(manifest["converter"]["commit"], "cafe");
    assert_eq!(manifest["conventions"]["ple_layer"], 1);
    assert!(manifest["conventions"]["norms"]["gain_w"]
        .as_str()
        .is_some_and(|s| s.contains("linear_attn.norm")));
}

#[test]
fn resume_after_partial_run_is_byte_identical() {
    let src = tempfile::tempdir().expect("invariant: dossier temporaire");
    write_source(src.path(), |_| {});
    let fresh = tempfile::tempdir().expect("invariant: dossier temporaire");
    let fresh_report =
        convert_qwen4exp(&options(src.path(), fresh.path())).expect("invariant: conversion");

    let resumed = tempfile::tempdir().expect("invariant: dossier temporaire");
    let mut partial = options(src.path(), resumed.path());
    partial.only = Some(vec![
        "model-00001-of-00002".to_string(),
        "ngram-003".to_string(),
    ]);
    let first = convert_qwen4exp(&partial).expect("invariant: pilote");
    assert!(!first.finalized);
    assert!(!resumed.path().join("config.json").exists());
    // Interruption simulée : un shard temporaire orphelin est écrasé.
    std::fs::write(
        resumed.path().join("model-00002-of-00002.safetensors.tmp"),
        b"x",
    )
    .expect("invariant: orphelin");
    let mut full = options(src.path(), resumed.path());
    full.verify = true;
    let second = convert_qwen4exp(&full).expect("invariant: reprise");
    assert!(second
        .units_resumed
        .contains(&"model-00001-of-00002".to_string()));
    assert!(second.units_resumed.contains(&"ngram-003".to_string()));
    let digest = |files: &[super::OutputFile]| {
        files
            .iter()
            .map(|f| (f.name.clone(), f.sha256.clone()))
            .collect::<BTreeMap<_, _>>()
    };
    assert_eq!(digest(&second.files), digest(&fresh_report.files));
}

#[test]
fn resume_refuses_changed_identity() {
    let src = tempfile::tempdir().expect("invariant: dossier temporaire");
    let dst = tempfile::tempdir().expect("invariant: dossier temporaire");
    write_source(src.path(), |_| {});
    let mut partial = options(src.path(), dst.path());
    partial.only = Some(vec!["ngram-000".to_string()]);
    convert_qwen4exp(&partial).expect("invariant: pilote");
    let mut other = partial.clone();
    other.source_revision = "1111111111111111111111111111111111111111".to_string();
    assert!(convert_qwen4exp(&other).is_err());
}

#[test]
fn destination_lock_blocks_a_second_writer() {
    let src = tempfile::tempdir().expect("invariant: dossier temporaire");
    let dst = tempfile::tempdir().expect("invariant: dossier temporaire");
    write_source(src.path(), |_| {});
    std::fs::create_dir_all(dst.path().join(".convert/lock")).expect("invariant: verrou");
    assert!(convert_qwen4exp(&options(src.path(), dst.path())).is_err());
}

#[test]
fn rejects_corrupt_or_incomplete_sources() {
    let cases: Vec<(&str, &str, Mutation)> = vec![
        (
            "expert manquant",
            "poids ou échelle absent",
            Box::new(|t| {
                t.remove("model.language_model.layers.0.mlp.experts.1.up_proj.weight");
            }),
        ),
        (
            "échelle absente",
            "poids ou échelle absent",
            Box::new(|t| {
                t.remove("mtp.layers.0.mlp.experts.0.down_proj.weight_scale_inv");
            }),
        ),
        (
            "échelle mal formée",
            "échelle [3, 1]",
            Box::new(|t| {
                t.insert(
                    "model.language_model.layers.1.mlp.experts.0.gate_proj.weight_scale_inv"
                        .to_string(),
                    bf16(&[3, 1], 0),
                );
            }),
        ),
        (
            "dense manquant",
            "o_proj",
            Box::new(|t| {
                t.remove("model.language_model.layers.1.self_attn.o_proj.weight");
            }),
        ),
        (
            "tenseur hors type de couche",
            "inattendu",
            Box::new(|t| {
                t.insert(
                    "model.language_model.layers.1.linear_attn.in_proj_z.weight".to_string(),
                    bf16(&[64, H], 0),
                );
            }),
        ),
        (
            "shard n-gram manquant",
            "shards n-gram",
            Box::new(|t| {
                t.remove(
                    "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_11.weight",
                );
            }),
        ),
        (
            "seed incohérent",
            "multiplicateurs",
            Box::new(|t| {
                t.insert(
                    "model.language_model.layers.1.ple.ple_embedding.layer_multipliers".to_string(),
                    i64s(&[1, 3, 5]),
                );
            }),
        ),
        (
            "NaN E4M3 dans un expert",
            "NaN (0x7f)",
            Box::new(|t| {
                if let Some(raw) =
                    t.get_mut("model.language_model.layers.0.mlp.experts.0.gate_proj.weight")
                {
                    raw.data[5] = 0x7f;
                }
            }),
        ),
        (
            "NaN E4M3 dans la table",
            "octets NaN",
            Box::new(|t| {
                if let Some(raw) = t.get_mut(
                    "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_4.weight",
                ) {
                    raw.data[9] = 0xff;
                }
            }),
        ),
    ];
    for (label, needle, mutate) in cases {
        let src = tempfile::tempdir().expect("invariant: dossier temporaire");
        let dst = tempfile::tempdir().expect("invariant: dossier temporaire");
        write_source(src.path(), |t| mutate(t));
        let error = convert_qwen4exp(&options(src.path(), dst.path()))
            .err()
            .unwrap_or_else(|| panic!("{label} accepté"));
        assert!(
            error.to_string().contains(needle),
            "{label}: erreur inattendue {error}"
        );
    }
}

#[test]
fn real_checkpoint_hash_goldens() {
    // Valeurs lues dans le checkpoint officiel (rev 236dfdf2) : seed 1234,
    // vocabulaire 248 320, base 20 000 000, 16 têtes.
    assert_eq!(
        layer_multipliers(248_320, 3, 0, NGRAM_SEED),
        vec![23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071]
    );
    let sizes = head_vocab_sizes(20_000_000, 16);
    assert_eq!(sizes[..4], [20_000_003, 20_000_023, 20_000_033, 20_000_047]);
    assert_eq!(sizes[15], 20_000_171);
    assert_eq!(
        (sizes.iter().sum::<i64>() as u64).div_ceil(128) * 128,
        320_001_536
    );
}

fn digests(files: &[super::OutputFile]) -> BTreeMap<String, String> {
    files
        .iter()
        .map(|f| (f.name.clone(), f.sha256.clone()))
        .collect()
}

/// Altère `ngram_e4m3.bin` après une conversion complète, puis relance sans
/// `--verify` : la reprise doit refaire les segments touchés et retrouver les
/// empreintes d'origine.
fn ngram_damage_is_repaired(damage: impl FnOnce(&Path)) {
    let src = tempfile::tempdir().expect("invariant: dossier temporaire");
    let dst = tempfile::tempdir().expect("invariant: dossier temporaire");
    write_source(src.path(), |_| {});
    let first = convert_qwen4exp(&options(src.path(), dst.path())).expect("invariant: conversion");
    damage(&dst.path().join("ngram/ngram_e4m3.bin"));
    let second = convert_qwen4exp(&options(src.path(), dst.path())).expect("invariant: reprise");
    assert!(
        second.units_written.iter().any(|u| u.starts_with("ngram-")),
        "aucun segment refait: {:?}",
        second.units_written
    );
    assert_eq!(digests(&second.files), digests(&first.files));
}

#[test]
fn resume_repairs_deleted_ngram_file_without_verify() {
    ngram_damage_is_repaired(|path| std::fs::remove_file(path).expect("invariant: suppression"));
}

#[test]
fn resume_repairs_truncated_ngram_file_without_verify() {
    ngram_damage_is_repaired(|path| {
        let file = std::fs::OpenOptions::new()
            .write(true)
            .open(path)
            .expect("invariant: ouverture");
        let len = file.metadata().expect("invariant: taille").len();
        file.set_len(len / 3).expect("invariant: troncature");
    });
}

#[test]
fn resume_repairs_same_size_ngram_corruption_without_verify() {
    ngram_damage_is_repaired(|path| {
        let mut bytes = std::fs::read(path).expect("invariant: lecture");
        let at = bytes.len() / 2;
        bytes[at] ^= 0x01;
        std::fs::write(path, bytes).expect("invariant: écriture");
    });
}

#[test]
fn finalize_invalidates_corrupt_shard_then_resume_repairs_it() {
    let src = tempfile::tempdir().expect("invariant: dossier temporaire");
    let dst = tempfile::tempdir().expect("invariant: dossier temporaire");
    write_source(src.path(), |_| {});
    let first = convert_qwen4exp(&options(src.path(), dst.path())).expect("invariant: conversion");
    let shard = dst.path().join("model-00001-of-00002.safetensors");
    let mut bytes = std::fs::read(&shard).expect("invariant: lecture");
    let last = bytes.len() - 1;
    bytes[last] ^= 0x01;
    std::fs::write(&shard, bytes).expect("invariant: écriture");
    // Même taille : la reprise sans --verify saute l'unité, la finalisation la
    // rejette et l'invalide.
    let error = convert_qwen4exp(&options(src.path(), dst.path()))
        .err()
        .unwrap_or_else(|| panic!("shard corrompu accepté"));
    assert!(error.to_string().contains("invalidée"), "{error}");
    let third = convert_qwen4exp(&options(src.path(), dst.path())).expect("invariant: réparation");
    assert!(third
        .units_written
        .contains(&"model-00001-of-00002".to_string()));
    assert_eq!(digests(&third.files), digests(&first.files));
}

#[test]
fn refuses_in_place_conversion_even_through_a_symlink() {
    let src = tempfile::tempdir().expect("invariant: dossier temporaire");
    write_source(src.path(), |_| {});
    let config = std::fs::read(src.path().join("config.json")).expect("invariant: config");
    let error = convert_qwen4exp(&options(src.path(), src.path()))
        .err()
        .unwrap_or_else(|| panic!("conversion sur place acceptée"));
    assert!(error.to_string().contains("sur place"), "{error}");
    let alias = tempfile::tempdir().expect("invariant: dossier temporaire");
    let link = alias.path().join("alias");
    std::os::unix::fs::symlink(src.path(), &link).expect("invariant: lien");
    let error = convert_qwen4exp(&options(src.path(), &link))
        .err()
        .unwrap_or_else(|| panic!("conversion via alias acceptée"));
    assert!(error.to_string().contains("sur place"), "{error}");
    // Alias par composant `.` non normalisé.
    let dotted = link.join(".");
    assert!(convert_qwen4exp(&options(src.path(), &dotted)).is_err());
    assert_eq!(
        std::fs::read(src.path().join("config.json")).expect("invariant: config"),
        config,
        "la config source ne doit pas changer"
    );
    assert!(!src.path().join(".convert").exists());
}

/// Réécrit l'en-tête JSON d'un shard source (données inchangées).
fn rewrite_header(path: &Path, edit: impl FnOnce(&mut serde_json::Map<String, serde_json::Value>)) {
    let bytes = std::fs::read(path).expect("invariant: shard");
    let n = u64::from_le_bytes(bytes[..8].try_into().expect("invariant: 8 octets")) as usize;
    let mut header: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(&bytes[8..8 + n]).expect("invariant: en-tête");
    edit(&mut header);
    let mut json = serde_json::to_vec(&header).expect("invariant: json");
    while json.len() % 8 != 0 {
        json.push(b' ');
    }
    let mut out = (json.len() as u64).to_le_bytes().to_vec();
    out.extend(json);
    out.extend_from_slice(&bytes[8 + n..]);
    std::fs::write(path, out).expect("invariant: réécriture");
}

fn plan_error(src: &Path) -> String {
    let dst = tempfile::tempdir().expect("invariant: dossier temporaire");
    let mut opts = options(src, dst.path());
    opts.plan_only = true;
    convert_qwen4exp(&opts)
        .err()
        .unwrap_or_else(|| panic!("source invalide acceptée"))
        .to_string()
}

fn first_tensor(header: &serde_json::Map<String, serde_json::Value>, dtype: &str) -> String {
    header
        .iter()
        .find(|(k, v)| *k != "__metadata__" && v["dtype"] == dtype)
        .map(|(k, _)| k.clone())
        .expect("invariant: tenseur du dtype")
}

#[test]
fn rejects_inconsistent_source_layouts() {
    type Edit = fn(&mut serde_json::Map<String, serde_json::Value>);
    let cases: [(&str, Edit); 3] = [
        ("attendus", |h| {
            let name = first_tensor(h, "BF16");
            let shape = h[&name]["shape"]
                .as_array()
                .expect("invariant: forme")
                .clone();
            let mut shape: Vec<u64> = shape.iter().filter_map(|d| d.as_u64()).collect();
            shape[0] -= 1;
            h[&name]["shape"] = shape.into();
        }),
        ("chevauchent", |h| {
            let name = first_tensor(h, "BF16");
            let start = h[&name]["data_offsets"][0]
                .as_u64()
                .expect("invariant: offset");
            let end = h[&name]["data_offsets"][1]
                .as_u64()
                .expect("invariant: offset");
            let other = h
                .iter()
                .find(|(k, v)| {
                    *k != "__metadata__"
                        && *k != &name
                        && v["data_offsets"][1].as_u64() > Some(start + 2)
                        && v["data_offsets"][0].as_u64() >= Some(end)
                })
                .map(|(k, _)| k.clone())
                .expect("invariant: voisin");
            let o_start = h[&other]["data_offsets"][0]
                .as_u64()
                .expect("invariant: offset");
            let o_end = h[&other]["data_offsets"][1]
                .as_u64()
                .expect("invariant: offset");
            // Même longueur, décalé sur `name`.
            let len = o_end - o_start;
            h[&other]["data_offsets"] = vec![start, start + len].into();
        }),
        ("hors fichier", |h| {
            let max = h
                .values()
                .filter_map(|v| v["data_offsets"][1].as_u64())
                .max()
                .expect("invariant: offsets");
            h.insert(
                "model.visual.extra.weight".to_string(),
                serde_json::json!({"dtype": "BF16", "shape": [8], "data_offsets": [max, max + 16]}),
            );
        }),
    ];
    for (needle, edit) in cases {
        let src = tempfile::tempdir().expect("invariant: dossier temporaire");
        write_source(src.path(), |_| {});
        let shard = src.path().join("model-00001-of-00002.safetensors");
        rewrite_header(&shard, edit);
        if needle == "hors fichier" {
            // Le nouvel en-tête doit figurer à l'index pour atteindre la validation.
            let index_path = src.path().join("model.safetensors.index.json");
            let mut index: serde_json::Value =
                serde_json::from_slice(&std::fs::read(&index_path).expect("invariant: index"))
                    .expect("invariant: json");
            index["weight_map"]["model.visual.extra.weight"] =
                "model-00001-of-00002.safetensors".into();
            std::fs::write(&index_path, index.to_string()).expect("invariant: index");
        }
        let error = plan_error(src.path());
        assert!(error.contains(needle), "{needle}: {error}");
    }
}

#[test]
fn rejects_non_finite_or_non_positive_ngram_scale() {
    for bits in [0x7fc0_u16, 0x7f80, 0xff80, 0x0000, 0xbf80] {
        let src = tempfile::tempdir().expect("invariant: dossier temporaire");
        write_source(src.path(), |t| {
            if let Some(raw) = t.get_mut(
                "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale",
            ) {
                raw.data = bits.to_le_bytes().to_vec();
            }
        });
        let error = plan_error(src.path());
        assert!(
            error.contains("échelle n-gram invalide"),
            "0x{bits:04x}: {error}"
        );
    }
}

#[test]
fn degenerate_configs_are_errors_not_panics() {
    for (key, value) in [
        ("num_experts", 0),
        ("split_ngram_parts", 0),
        ("ngram_size", 0),
        ("make_ngram_vocab_size_divisible_by", 0),
        ("heads_per_ngram", 0),
        ("num_hidden_layers", 0),
    ] {
        let src = tempfile::tempdir().expect("invariant: dossier temporaire");
        write_source(src.path(), |_| {});
        let path = src.path().join("config.json");
        let mut config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).expect("invariant: config"))
                .expect("invariant: json");
        config["text_config"][key] = value.into();
        std::fs::write(&path, config.to_string()).expect("invariant: config");
        let error = plan_error(src.path());
        assert!(!error.is_empty(), "{key}");
    }
    for ids in [
        serde_json::json!([]),
        serde_json::json!([0]),
        serde_json::json!([3]),
        serde_json::json!([1, 2]),
        serde_json::json!([2, -1]),
        serde_json::json!([2, "invalide"]),
        serde_json::json!(["2"]),
        serde_json::json!([-1]),
        serde_json::json!([1.5]),
        serde_json::json!(2),
    ] {
        let src = tempfile::tempdir().expect("invariant: dossier temporaire");
        write_source(src.path(), |_| {});
        let path = src.path().join("config.json");
        let mut config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).expect("invariant: config"))
                .expect("invariant: json");
        config["text_config"]["ple_layer_ids"] = ids.clone();
        std::fs::write(&path, config.to_string()).expect("invariant: config");
        assert!(plan_error(src.path()).contains("ple_layer_ids"), "{ids}");
    }
    let src = tempfile::tempdir().expect("invariant: dossier temporaire");
    write_source(src.path(), |t| {
        t.insert("lm_head.weight".to_string(), bf16(&[VOCAB, 0], 0));
    });
    assert!(plan_error(src.path()).contains("matrice vide"));
}

#[test]
fn lm_head_u6_restores_source_values() {
    let src = tempfile::tempdir().expect("invariant: dossier temporaire");
    let dst = tempfile::tempdir().expect("invariant: dossier temporaire");
    write_source(src.path(), |_| {});
    convert_qwen4exp(&options(src.path(), dst.path())).expect("invariant: conversion");
    let (_, source) = read_tensor(&find_shard(src.path(), "lm_head.weight"), "lm_head.weight");
    let source = bf16s(&source);
    let shard = find_shard(dst.path(), "language_model.lm_head.weight");
    let (_, words) = read_tensor(&shard, "language_model.lm_head.weight");
    let (_, scales) = read_tensor(&shard, "language_model.lm_head.scales");
    let (_, biases) = read_tensor(&shard, "language_model.lm_head.biases");
    let (words, scales, biases) = (u32s(&words), bf16s(&scales), bf16s(&biases));
    let packed_cols = H * 6 / 32;
    let groups = H / 64;
    for row in [0, 17, VOCAB - 1] {
        for col in 0..H {
            // Décodage bitstream indépendant : bits lus un par un, LSB-first.
            let code = (0..6).fold(0_u32, |acc, bit| {
                let pos = col * 6 + bit;
                acc | (((words[row * packed_cols + pos / 32] >> (pos % 32)) & 1) << bit)
            });
            let g = row * groups + col / 64;
            let restored = code as f32 * scales[g] + biases[g];
            let w = source[row * H + col];
            assert!(
                (w - restored).abs() <= scales[g] * 0.5 + 1.0e-7,
                "ligne {row} col {col}: {w} vs {restored}"
            );
        }
    }
}

#[test]
fn rejects_duplicate_tensor_across_shards() {
    let src = tempfile::tempdir().expect("invariant: dossier temporaire");
    write_source(src.path(), |_| {});
    let first = src.path().join("model-00001-of-00002.safetensors");
    let second = src.path().join("model-00002-of-00002.safetensors");
    let header = read_shard_header(&first).expect("invariant: en-tête");
    let (name, entry) = header
        .tensors
        .iter()
        .find(|(_, e)| e.dtype == Dtype::BF16)
        .map(|(n, e)| (n.clone(), e.clone()))
        .expect("invariant: tenseur bf16");
    let len = entry.data_offsets[1] - entry.data_offsets[0];
    // Ajoute une copie valide (plage libre en fin de données) dans l'autre shard.
    let data_len = std::fs::read(&second).expect("invariant: shard").len()
        - read_shard_header(&second)
            .expect("invariant: en-tête")
            .data_start as usize;
    rewrite_header(&second, |h| {
        h.insert(
            name.clone(),
            serde_json::json!({
                "dtype": "BF16",
                "shape": entry.shape,
                "data_offsets": [data_len, data_len + len],
            }),
        );
    });
    let mut bytes = std::fs::read(&second).expect("invariant: shard");
    bytes.extend(std::iter::repeat_n(0_u8, len));
    std::fs::write(&second, bytes).expect("invariant: shard");
    assert!(plan_error(src.path()).contains("incohérents"));
}

#[test]
fn rejects_out_of_range_expert_f32_dense_and_non_finite_block_scale() {
    let cases: Vec<(&str, &str, Mutation)> = vec![
        (
            "expert ≥ num_experts",
            "hors configuration",
            Box::new(|t| {
                let weight = Raw {
                    dtype: Dtype::F8_E4M3,
                    shape: vec![I, H],
                    data: e4m3_bytes(I * H, 0),
                };
                t.insert(
                    format!("model.language_model.layers.0.mlp.experts.{E}.gate_proj.weight"),
                    weight,
                );
            }),
        ),
        (
            "dense F32",
            "BF16",
            Box::new(|t| {
                let count = 64 * H;
                t.insert(
                    "model.language_model.layers.0.linear_attn.out_proj.weight".to_string(),
                    Raw {
                        dtype: Dtype::F32,
                        shape: vec![64, H],
                        data: (0..count)
                            .flat_map(|i| (i as f32 * 1e-3).to_le_bytes())
                            .collect(),
                    },
                );
            }),
        ),
        (
            "weight_scale_inv non finie",
            "non finie",
            Box::new(|t| {
                if let Some(raw) = t.get_mut(
                    "model.language_model.layers.0.mlp.experts.1.gate_proj.weight_scale_inv",
                ) {
                    raw.data[2..4].copy_from_slice(&0x7f80_u16.to_le_bytes());
                }
            }),
        ),
    ];
    for (label, needle, mutate) in cases {
        let src = tempfile::tempdir().expect("invariant: dossier temporaire");
        let dst = tempfile::tempdir().expect("invariant: dossier temporaire");
        write_source(src.path(), |t| mutate(t));
        let error = convert_qwen4exp(&options(src.path(), dst.path()))
            .err()
            .unwrap_or_else(|| panic!("{label} accepté"));
        assert!(error.to_string().contains(needle), "{label}: {error}");
    }
}

#[test]
fn refuses_parent_component_behind_missing_directory() {
    let src = tempfile::tempdir().expect("invariant: dossier temporaire");
    write_source(src.path(), |_| {});
    let config = std::fs::read(src.path().join("config.json")).expect("invariant: config");
    // `src/inexistant/..` : create_dir_all ramènerait la destination sur la source.
    let sneaky = src.path().join("inexistant").join("..");
    let error = convert_qwen4exp(&options(src.path(), &sneaky))
        .err()
        .unwrap_or_else(|| panic!("destination ambiguë acceptée"));
    assert!(error.to_string().contains("ambiguë"), "{error}");
    assert!(!src.path().join("inexistant").exists());
    assert_eq!(
        std::fs::read(src.path().join("config.json")).expect("invariant: config"),
        config
    );
    // Un `..` dans la partie existante est résolu par le système et accepté.
    let dst = tempfile::tempdir().expect("invariant: dossier temporaire");
    std::fs::create_dir(dst.path().join("x")).expect("invariant: dossier");
    let mut opts = options(src.path(), &dst.path().join("x").join("..").join("y"));
    opts.plan_only = true;
    assert!(convert_qwen4exp(&opts).is_ok());
}

#[test]
fn corrupt_shard_header_is_invalidated_then_repaired() {
    let src = tempfile::tempdir().expect("invariant: dossier temporaire");
    let dst = tempfile::tempdir().expect("invariant: dossier temporaire");
    write_source(src.path(), |_| {});
    let first = convert_qwen4exp(&options(src.path(), dst.path())).expect("invariant: conversion");
    let shard = dst.path().join("model-00002-of-00002.safetensors");
    let mut bytes = std::fs::read(&shard).expect("invariant: lecture");
    // En-tête JSON illisible, même taille de fichier.
    bytes[8] = b'#';
    std::fs::write(&shard, bytes).expect("invariant: écriture");
    let error = convert_qwen4exp(&options(src.path(), dst.path()))
        .err()
        .unwrap_or_else(|| panic!("en-tête corrompu accepté"));
    assert!(error.to_string().contains("invalidée"), "{error}");
    let third = convert_qwen4exp(&options(src.path(), dst.path())).expect("invariant: réparation");
    assert!(third
        .units_written
        .contains(&"model-00002-of-00002".to_string()));
    assert_eq!(digests(&third.files), digests(&first.files));
}

#[test]
fn verify_repairs_corrupt_shard_and_segment_in_one_run() {
    let src = tempfile::tempdir().expect("invariant: dossier temporaire");
    let dst = tempfile::tempdir().expect("invariant: dossier temporaire");
    write_source(src.path(), |_| {});
    let first = convert_qwen4exp(&options(src.path(), dst.path())).expect("invariant: conversion");
    let shard = dst.path().join("model-00001-of-00002.safetensors");
    let mut bytes = std::fs::read(&shard).expect("invariant: lecture");
    let last = bytes.len() - 1;
    bytes[last] ^= 0x01;
    std::fs::write(&shard, bytes).expect("invariant: écriture");
    let table = dst.path().join("ngram/ngram_e4m3.bin");
    let mut bytes = std::fs::read(&table).expect("invariant: lecture");
    bytes[0] ^= 0x01;
    std::fs::write(&table, bytes).expect("invariant: écriture");
    let mut verify = options(src.path(), dst.path());
    verify.verify = true;
    let second = convert_qwen4exp(&verify).expect("invariant: réparation --verify");
    assert!(second
        .units_written
        .contains(&"model-00001-of-00002".to_string()));
    assert!(second.units_written.contains(&"ngram-000".to_string()));
    assert!(second.finalized);
    assert_eq!(digests(&second.files), digests(&first.files));
}
