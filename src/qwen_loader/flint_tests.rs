//! Tests du chargeur et de l'oracle CPU flint, sur un modèle synthétique.

use super::*;
use safetensors::{serialize, View};
use std::borrow::Cow;

const HIDDEN: usize = 64;
const HEADS: usize = 2;
const HEAD_DIM: usize = 32;
const KV_DIM: usize = HEAD_DIM;
const VOCAB: usize = 8;
const EXPERTS: usize = 4;
const INTER: usize = 64;

struct Fixture {
    dtype: Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl View for Fixture {
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

/// Générateur déterministe des octets empaquetés (pas de dépendance rand).
struct Lcg(u64);

impl Lcg {
    fn byte(&mut self) -> u8 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u8
    }
}

fn bf16(shape: Vec<usize>, values: impl IntoIterator<Item = f32>) -> Fixture {
    Fixture {
        dtype: Dtype::BF16,
        shape,
        data: values
            .into_iter()
            .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
            .collect(),
    }
}

/// Ajoute un poids affine `[.., rows, HIDDEN]` empaqueté en `U8` (ou `U32`).
fn push_affine(
    out: &mut Vec<(String, Fixture)>,
    rng: &mut Lcg,
    name: &str,
    lead: &[usize],
    bits: usize,
    packed_u8: bool,
) {
    let row_bytes = HIDDEN * bits / 8;
    let rows = lead.iter().product::<usize>();
    let data = (0..rows * row_bytes)
        .map(|_| rng.byte())
        .collect::<Vec<_>>();
    let (dtype, last) = if packed_u8 {
        (Dtype::U8, row_bytes)
    } else {
        (Dtype::U32, row_bytes / 4)
    };
    let mut shape = lead.to_vec();
    shape.push(last);
    out.push((format!("{name}.weight"), Fixture { dtype, shape, data }));
    let levels = ((1_u32 << bits) - 1) as f32;
    let mut affine_shape = lead.to_vec();
    affine_shape.push(HIDDEN / 64);
    let groups = rows * (HIDDEN / 64);
    // Poids centrés dans [-0,3 ; 0,3] quel que soit `bits`.
    out.push((
        format!("{name}.scales"),
        bf16(affine_shape.clone(), vec![0.6 / levels; groups]),
    ));
    out.push((
        format!("{name}.biases"),
        bf16(affine_shape, vec![-0.3; groups]),
    ));
}

fn flint_tensors(packed_u8: bool) -> Vec<(String, Fixture)> {
    let mut rng = Lcg(7);
    let mut out = Vec::new();
    push_affine(
        &mut out,
        &mut rng,
        "model.embed_tokens",
        &[VOCAB],
        8,
        packed_u8,
    );
    push_affine(&mut out, &mut rng, "lm_head", &[VOCAB], 4, packed_u8);
    out.push((
        "model.norm.weight".into(),
        bf16(vec![HIDDEN], vec![1.0; HIDDEN]),
    ));
    for layer in 0..3 {
        let p = format!("model.layers.{layer}");
        for norm in ["input_layernorm", "post_attention_layernorm"] {
            out.push((
                format!("{p}.{norm}.weight"),
                bf16(vec![HIDDEN], vec![1.0; HIDDEN]),
            ));
        }
        for (proj, rows) in [
            ("q_proj", HEADS * HEAD_DIM),
            ("k_proj", KV_DIM),
            ("v_proj", KV_DIM),
        ] {
            push_affine(
                &mut out,
                &mut rng,
                &format!("{p}.self_attn.{proj}"),
                &[rows],
                4,
                packed_u8,
            );
        }
        push_affine(
            &mut out,
            &mut rng,
            &format!("{p}.self_attn.o_proj"),
            &[HIDDEN],
            4,
            packed_u8,
        );
        for norm in ["q_norm", "k_norm"] {
            out.push((
                format!("{p}.self_attn.{norm}.weight"),
                bf16(vec![HEAD_DIM], vec![1.0; HEAD_DIM]),
            ));
        }
        out.push((
            format!("{p}.self_attn.softmax_offset"),
            bf16(vec![HEADS], [0.5, -1.0]),
        ));
        if layer == 0 {
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                push_affine(
                    &mut out,
                    &mut rng,
                    &format!("{p}.mlp.{proj}"),
                    &[INTER],
                    4,
                    packed_u8,
                );
            }
        } else {
            out.push((
                format!("{p}.mlp.gate.weight"),
                bf16(
                    vec![EXPERTS, HIDDEN],
                    (0..EXPERTS * HIDDEN).map(|i| ((i % 13) as f32 - 6.0) * 0.05),
                ),
            ));
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                push_affine(
                    &mut out,
                    &mut rng,
                    &format!("{p}.mlp.switch_mlp.{proj}"),
                    &[EXPERTS, INTER],
                    4,
                    packed_u8,
                );
                push_affine(
                    &mut out,
                    &mut rng,
                    &format!("{p}.mlp.shared_expert.{proj}"),
                    &[INTER],
                    4,
                    packed_u8,
                );
            }
        }
    }
    out
}

fn flint_config() -> ModelConfig {
    let raw = serde_json::json!({
        "model_type": "flint",
        "hidden_size": HIDDEN,
        "num_hidden_layers": 3,
        "num_attention_heads": HEADS,
        "num_key_value_heads": 1,
        "head_dim": HEAD_DIM,
        "intermediate_size": INTER,
        "moe_intermediate_size": INTER,
        "shared_expert_intermediate_size": INTER,
        "num_experts": EXPERTS,
        "num_experts_per_tok": 2,
        "mlp_only_layers": [0],
        "no_rope_layer_indices": [0],
        "layer_types": ["full_attention", "sliding_attention", "sliding_attention"],
        "sliding_window": 2,
        "router_score_function": "sigmoid",
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "vocab_size": VOCAB,
        "eos_token_id": [7],
        "tie_word_embeddings": false,
        "quantization": {"bits": 4, "group_size": 64},
    });
    serde_json::from_value::<crate::config::RawModelConfig>(raw)
        .expect("invariant: config flint JSON valide")
        .resolve()
        .expect("invariant: config flint résolue")
}

fn write_shard(tensors: Vec<(String, Fixture)>) -> tempfile::NamedTempFile {
    let tmp = tempfile::NamedTempFile::new().expect("invariant: fichier temporaire");
    let buffer = serialize(tensors, None).expect("invariant: safetensors sérialisable");
    std::fs::write(tmp.path(), buffer).expect("invariant: écriture safetensors");
    tmp
}

fn load(tensors: Vec<(String, Fixture)>) -> Result<CausalDecoder> {
    let tmp = write_shard(tensors);
    let shards = [tmp.path().to_path_buf()];
    let catalog = WeightCatalog::from_shards(&shards)?;
    let config = flint_config();
    verify_decoder_contract_from_shards(&config, &shards, &catalog)?;
    load_causal_decoder_from_shards(&config, &shards, &catalog)
}

#[test]
fn flint_config_decodes_per_layer_contract() {
    let config = flint_config();
    assert!(config.is_flint());
    assert!(config.is_mlp_only_layer(0) && !config.is_mlp_only_layer(1));
    assert_eq!(
        config.router_score().ok(),
        Some(crate::RouterScore::Sigmoid)
    );
    assert!(!config.shared_expert_gated());
    let decoder = crate::CausalDecoderConfig::from(&config);
    assert_eq!(decoder.no_rope_layers, vec![0]);
    assert!(decoder.metal_unsupported_reason().is_some());
}

#[test]
fn synthetic_flint_loads_and_generates_on_cpu() {
    let model = load(flint_tensors(true)).expect("invariant: flint synthétique chargeable");
    let logits = model.next_logits(&[1, 2, 3]).expect("invariant: forward");
    assert_eq!(logits.shape(), &[1, VOCAB]);
    assert!(logits.data().iter().all(|value| value.is_finite()));
    // Le chemin avec cache (prefill puis decode, fenêtre 2 dépassée) doit
    // rendre les mêmes logits que le forward sans cache.
    let mut cache = model.empty_cache();
    let mut cached = None;
    for token in [1, 2, 3] {
        cached = Some(
            model
                .next_logits_cached(&mut cache, token)
                .expect("invariant: decode avec cache"),
        );
    }
    let cached = cached.expect("invariant: trois pas");
    for (a, b) in logits.data().iter().zip(cached.data()) {
        assert!((a - b).abs() < 1.0e-4, "cache {b} vs sans cache {a}");
    }
    let generated = model
        .generate_greedy(&[1, 2, 3], 4)
        .expect("invariant: génération greedy CPU");
    assert_eq!(generated.len(), 4);
    assert!(generated.iter().all(|token| *token < VOCAB));
}

#[test]
fn u8_packing_matches_u32_packing() {
    let u8_model = load(flint_tensors(true)).expect("invariant: flint U8");
    let u32_model = load(flint_tensors(false)).expect("invariant: flint U32");
    let prompt = [4, 0, 5, 6];
    assert_eq!(
        u8_model
            .next_logits(&prompt)
            .expect("invariant: forward U8"),
        u32_model
            .next_logits(&prompt)
            .expect("invariant: forward U32")
    );
}

#[test]
fn flint_rejects_missing_sink_and_orphan_tensors() {
    let mut missing = flint_tensors(true);
    missing.retain(|(name, _)| name != "model.layers.1.self_attn.softmax_offset");
    let err = load(missing).expect_err("sink manquant refusé");
    assert!(err.to_string().contains("softmax_offset"), "{err}");

    for orphan in [
        "model.layers.1.mlp.gate.expert_bias",
        "model.layers.1.mlp.shared_expert_gate.weight",
    ] {
        let mut tensors = flint_tensors(true);
        tensors.push((orphan.to_string(), bf16(vec![1], [0.0])));
        let err = load(tensors).expect_err("tenseur orphelin refusé");
        assert!(err.to_string().contains("non consommés"), "{err}");
    }

    let mut dense_moe = flint_tensors(true);
    dense_moe.retain(|(name, _)| !name.starts_with("model.layers.0.mlp."));
    assert!(load(dense_moe).is_err(), "couche mlp_only sans MLP dense");
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn flint_refuses_metal_runtime_explicitly() {
    let model = load(flint_tensors(true)).expect("invariant: flint synthétique");
    let err = model
        .with_metal_runtime()
        .expect_err("runtime Metal refusé pour flint");
    assert!(err.to_string().contains("backend CPU"), "{err}");
}

/// Lit un `.npy` v1 f32 little-endian (en-tête puis données brutes).
fn read_npy_f32(path: &Path) -> (Vec<usize>, Vec<f32>) {
    let bytes = std::fs::read(path).expect("invariant: .npy lisible");
    assert_eq!(&bytes[..6], b"\x93NUMPY", "magie npy");
    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header =
        std::str::from_utf8(&bytes[10..10 + header_len]).expect("invariant: en-tête ascii");
    assert!(header.contains("'<f4'"), "dtype attendu <f4: {header}");
    assert!(header.contains("'fortran_order': False"), "{header}");
    let shape_start = header.find("'shape': (").expect("invariant: shape") + 10;
    let shape_end = shape_start
        + header[shape_start..]
            .find(')')
            .expect("invariant: shape fermée");
    let shape = header[shape_start..shape_end]
        .split(',')
        .filter(|dim| !dim.trim().is_empty())
        .map(|dim| dim.trim().parse::<usize>().expect("invariant: dimension"))
        .collect::<Vec<_>>();
    let data = bytes[10 + header_len..]
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect::<Vec<_>>();
    assert_eq!(data.len(), shape.iter().product::<usize>());
    (shape, data)
}

fn compare_rows(label: &str, ours: &[f32], reference: &[f32]) -> (f32, f32, bool) {
    assert_eq!(ours.len(), reference.len());
    assert!(
        ours.iter().all(|value| value.is_finite()),
        "{label}: logits non finis"
    );
    let diffs = ours.iter().zip(reference).map(|(a, b)| (a - b).abs());
    let max = diffs.clone().fold(0.0_f32, f32::max);
    let mean = diffs.sum::<f32>() / ours.len() as f32;
    let argmax = |row: &[f32]| {
        row.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(index, _)| index)
    };
    let same = argmax(ours) == argmax(reference);
    // Décalage moyen signé : un décalage constant ne change pas le softmax.
    let shift = ours.iter().zip(reference).map(|(a, b)| a - b).sum::<f32>() / ours.len() as f32;
    println!(
        "{label}: max_abs={max:.4} mean_abs={mean:.5} shift={shift:.4} argmax ours={:?} ref={:?}",
        argmax(ours),
        argmax(reference)
    );
    (max, mean, same)
}

/// Poids réels flint-int4 contre les logits de mlx-lm épinglé (4cfc6478).
///
/// `SARAGOSSA_FLINT_DIR` et `SARAGOSSA_FLINT_ORACLE_DIR` surchargent les chemins.
#[test]
#[ignore = "poids réels flint-int4 (22,6 Go) et oracle mlx-lm requis"]
fn flint_int4_real_weights_match_mlx_oracle() {
    let model_dir = std::env::var_os("SARAGOSSA_FLINT_DIR").map_or_else(
        || PathBuf::from("/Users/ludwig/workspace/saragossa-oss/models/flint-int4"),
        PathBuf::from,
    );
    let oracle_dir = std::env::var_os("SARAGOSSA_FLINT_ORACLE_DIR").map_or_else(
        || {
            PathBuf::from(
                "/private/tmp/claude-501/-Users-ludwig-workspace-saragossa-oss/\
                 47d98124-b7be-4a39-ab25-3be149b935b4/scratchpad/flint",
            )
        },
        PathBuf::from,
    );
    let oracle: serde_json::Value = serde_json::from_slice(
        &std::fs::read(oracle_dir.join("oracle.json")).expect("invariant: oracle.json"),
    )
    .expect("invariant: oracle JSON");
    let ids = |key: &str| {
        oracle[key]
            .as_array()
            .unwrap_or_else(|| panic!("oracle sans {key}"))
            .iter()
            .map(|id| id.as_u64().expect("invariant: id entier") as usize)
            .collect::<Vec<_>>()
    };
    let prompt_ids = ids("ids");
    let greedy_ids = ids("greedy");
    let prompt = oracle["prompt"].as_str().expect("invariant: prompt texte");

    let started = std::time::Instant::now();
    let assets = ModelAssets::load_local(&model_dir).expect("invariant: assets flint");
    let ours = assets
        .tokenizer
        .encode(prompt)
        .expect("invariant: tokenisation")
        .into_iter()
        .map(|id| id as usize)
        .collect::<Vec<_>>();
    println!("tokens ours={ours:?} mlx={prompt_ids:?}");
    let model = load_causal_decoder(&assets).expect("invariant: flint-int4 chargeable");
    println!("chargement: {:.1?}", started.elapsed());

    // Logits du prompt position par position via le chemin avec cache.
    let (shape, prompt_logits) = read_npy_f32(&oracle_dir.join("oracle_prompt_logits.npy"));
    assert_eq!(shape, vec![prompt_ids.len(), assets.config.vocab_size]);
    let vocab = assets.config.vocab_size;
    let started = std::time::Instant::now();
    let mut cache = model.empty_cache();
    let mut worst = 0.0_f32;
    let mut argmax_ok = true;
    let mut last_row = Vec::new();
    for (position, token) in prompt_ids.iter().enumerate() {
        let logits = model
            .next_logits_cached(&mut cache, *token)
            .expect("invariant: pas avec cache");
        let reference = &prompt_logits[position * vocab..(position + 1) * vocab];
        let (max, _, same) = compare_rows(&format!("prompt[{position}]"), logits.data(), reference);
        worst = worst.max(max);
        argmax_ok &= same;
        last_row = logits.data().to_vec();
    }
    println!("prompt: {:.1?}", started.elapsed());

    // Greedy avec cache, en forçant les tokens de la référence pour mesurer la
    // dérive sans divergence de trajectoire.
    let (_, greedy_logits) = read_npy_f32(&oracle_dir.join("oracle_greedy_logits.npy"));
    let steps = std::env::var("SARAGOSSA_FLINT_STEPS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(4)
        .min(greedy_ids.len());
    let started = std::time::Instant::now();
    let mut ours_greedy = Vec::new();
    // Le premier token greedy sort des logits de la dernière position du prompt.
    let mut row = last_row;
    for step in 0..steps {
        if step > 0 {
            row = model
                .next_logits_cached(&mut cache, greedy_ids[step - 1])
                .expect("invariant: pas greedy")
                .data()
                .to_vec();
        }
        let reference = &greedy_logits[step * vocab..(step + 1) * vocab];
        let (max, _, same) = compare_rows(&format!("greedy[{step}]"), &row, reference);
        worst = worst.max(max);
        argmax_ok &= same;
        let token = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(index, _)| index)
            .expect("invariant: vocab non vide");
        ours_greedy.push(token);
    }
    println!("greedy: {:.1?}", started.elapsed());
    let text = assets
        .tokenizer
        .decode(
            &ours_greedy.iter().map(|id| *id as u32).collect::<Vec<_>>(),
            false,
        )
        .expect("invariant: décodage");
    println!("greedy ours={ours_greedy:?} ref={:?}", &greedy_ids[..steps]);
    println!("texte: {prompt}{text:?}");
    println!("écart max global: {worst:.4}");
    assert_eq!(ours, prompt_ids, "tokenizer divergent de mlx");
    assert_eq!(ours_greedy, greedy_ids[..steps], "tokens greedy divergents");
    assert!(argmax_ok, "argmax divergent sur au moins une position");
}
