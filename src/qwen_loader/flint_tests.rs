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
fn router_defaults_to_sigmoid_for_flint_only() {
    let mut config = flint_config();
    config.router_score_function = None;
    assert_eq!(
        config.router_score().ok(),
        Some(crate::RouterScore::Sigmoid)
    );
    config.model_type = "qwen3_moe".to_string();
    assert_eq!(
        config.router_score().ok(),
        Some(crate::RouterScore::Softmax)
    );
    config.router_score_function = Some("softplus".to_string());
    assert!(config.router_score().is_err());
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

    // Biais dense chargé (spec optionnelle) mais jamais assemblé sur une
    // couche MoE : refusé après assemblage, pas ignoré.
    let mut stray_bias = flint_tensors(true);
    stray_bias.push((
        "model.layers.1.mlp.gate_proj.bias".to_string(),
        bf16(vec![INTER], vec![0.0; INTER]),
    ));
    let err = load(stray_bias).expect_err("biais non assemblé refusé");
    assert!(err.to_string().contains("non assemblés"), "{err}");

    // Scales/biases affines d'un poids dense : orphelins.
    let mut dense_scales = flint_tensors(true);
    dense_scales.push((
        "model.layers.1.mlp.gate.scales".to_string(),
        bf16(vec![EXPERTS, 1], vec![1.0; EXPERTS]),
    ));
    let err = load(dense_scales).expect_err("scales d'un poids dense refusées");
    assert!(err.to_string().contains("non consommés"), "{err}");
}

/// Decode résident complet (sinks, NoPE, fenêtre dépassée, MoE sigmoïde et
/// expert partagé sans gate) contre l'oracle CPU, sur prefill CPU commun.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn synthetic_flint_resident_decode_matches_cpu_oracle() {
    let cpu = load(flint_tensors(true)).expect("invariant: flint synthétique");
    let gpu = load(flint_tensors(true))
        .expect("invariant: flint synthétique")
        .with_metal_runtime()
        .expect("invariant: runtime Metal attaché pour le decode résident");
    assert!(gpu.supports_resident_full_decode(), "decode résident flint");

    let prompt = [1, 2, 3];
    let forced = [4, 0, 5, 6, 3, 2];
    let mut cache = cpu.empty_cache();
    let mut expected = Vec::new();
    for (step, token) in prompt.iter().chain(&forced).enumerate() {
        let logits = cpu
            .next_logits_cached(&mut cache, *token)
            .expect("invariant: oracle CPU");
        if step >= prompt.len() {
            expected.push(logits.data().to_vec());
        }
    }
    // KV f32 : seul l'ordre d'accumulation diffère du CPU (écart absolu).
    // KV bf16 : l'arrondi relatif 2^-9 des K/V, propagé par trois couches,
    // borne l'écart à 1 % de la plus grande magnitude de logit.
    for (kv_bf16, rel) in [(false, None), (true, Some(1.0e-2_f32))] {
        let rows = gpu
            .resident_forced_logits_for_test(&prompt, &forced, kv_bf16)
            .expect("invariant: decode résident flint");
        for (step, (got, want)) in rows.iter().zip(&expected).enumerate() {
            let gap = got
                .iter()
                .zip(want)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            let scale = want.iter().fold(0.0_f32, |acc, value| acc.max(value.abs()));
            let eps = rel.map_or(1.0e-4, |rel| rel * scale);
            assert!(
                gap <= eps,
                "pas {step} kv_bf16={kv_bf16}: écart {gap} > {eps}"
            );
        }
    }
    assert_eq!(
        gpu.generate_greedy(&prompt, 8)
            .expect("invariant: greedy GPU"),
        cpu.generate_greedy(&prompt, 8)
            .expect("invariant: greedy CPU"),
        "tokens greedy résident != CPU"
    );
    let err = gpu
        .generate_greedy_lightbatch_with_options(
            &[prompt.to_vec()],
            2,
            &crate::GenerationOptions::default(),
        )
        .expect_err("light-batch refusé pour flint");
    assert!(err.to_string().contains("light-batch"), "{err}");
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

/// Écarts d'une ligne de logits : max et moyenne absolus, après retrait du
/// décalage constant (sans effet sur le softmax), et argmax identique.
struct RowGap {
    max: f32,
    mean: f32,
    same_argmax: bool,
}

fn argmax(row: &[f32]) -> Option<usize> {
    row.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(index, _)| index)
}

fn compare_rows(label: &str, ours: &[f32], reference: &[f32]) -> RowGap {
    assert_eq!(ours.len(), reference.len());
    assert!(
        ours.iter().all(|value| value.is_finite()),
        "{label}: logits non finis"
    );
    let len = ours.len() as f32;
    let shift = ours.iter().zip(reference).map(|(a, b)| a - b).sum::<f32>() / len;
    let centered = ours
        .iter()
        .zip(reference)
        .map(|(a, b)| (a - b - shift).abs());
    let max = centered.clone().fold(0.0_f32, f32::max);
    let mean = centered.sum::<f32>() / len;
    let same_argmax = argmax(ours) == argmax(reference);
    println!(
        "{label}: max={max:.4} mean={mean:.5} shift={shift:.4} argmax ours={:?} ref={:?}",
        argmax(ours),
        argmax(reference)
    );
    RowGap {
        max,
        mean,
        same_argmax,
    }
}

fn oracle_ids(oracle: &serde_json::Value, key: &str) -> Vec<usize> {
    oracle[key]
        .as_array()
        .unwrap_or_else(|| panic!("invariant: oracle sans {key}"))
        .iter()
        .map(|id| id.as_u64().expect("invariant: id entier") as usize)
        .collect()
}

/// Compare chaque ligne de `ours` (positions du prompt puis pas greedy) aux
/// logits d'un oracle ; renvoie le pire écart `(max, mean)` et l'accord des argmax.
fn compare_oracle(
    name: &str,
    dir: &Path,
    ours: &[Vec<f32>],
    prompt_len: usize,
) -> (f32, f32, bool) {
    let (_, prompt) = read_npy_f32(&dir.join("oracle_prompt_logits.npy"));
    let (_, greedy) = read_npy_f32(&dir.join("oracle_greedy_logits.npy"));
    let vocab = ours[0].len();
    let (mut max, mut mean, mut argmax_ok) = (0.0_f32, 0.0_f32, true);
    for (index, row) in ours.iter().enumerate() {
        let (label, reference) = if index < prompt_len {
            (
                format!("{name} prompt[{index}]"),
                &prompt[index * vocab..(index + 1) * vocab],
            )
        } else {
            let step = index + 1 - prompt_len;
            (
                format!("{name} greedy[{step}]"),
                &greedy[step * vocab..(step + 1) * vocab],
            )
        };
        let gap = compare_rows(&label, row, reference);
        max = max.max(gap.max);
        mean = mean.max(gap.mean);
        argmax_ok &= gap.same_argmax;
    }
    println!("{name}: pire max={max:.4}, pire moyenne={mean:.5}");
    (max, mean, argmax_ok)
}

/// Poids réels flint-int4 contre les logits de mlx-lm épinglé (4cfc6478).
///
/// Exige `SARAGOSSA_FLINT_MODEL_DIR` (poids MLX int4) et
/// `SARAGOSSA_FLINT_ORACLE_DIR` (oracle mlx en float32 : `oracle.json`,
/// `oracle_prompt_logits.npy`, `oracle_greedy_logits.npy`) ;
/// `SARAGOSSA_FLINT_ORACLE_BF16_DIR`, optionnel, ajoute l'oracle bf16 pour
/// information. Sans les deux premières, le test s'arrête avec un message.
#[test]
#[ignore = "poids réels flint-int4 (22,6 Go) et oracle mlx-lm requis"]
fn flint_int4_real_weights_match_mlx_oracle() {
    let (Some(model_dir), Some(oracle_dir)) = (
        std::env::var_os("SARAGOSSA_FLINT_MODEL_DIR").map(PathBuf::from),
        std::env::var_os("SARAGOSSA_FLINT_ORACLE_DIR").map(PathBuf::from),
    ) else {
        eprintln!(
            "test flint réel sauté : définir SARAGOSSA_FLINT_MODEL_DIR et \
             SARAGOSSA_FLINT_ORACLE_DIR"
        );
        return;
    };
    let oracle: serde_json::Value = serde_json::from_slice(
        &std::fs::read(oracle_dir.join("oracle.json")).expect("invariant: oracle.json"),
    )
    .expect("invariant: oracle JSON");
    let prompt_ids = oracle_ids(&oracle, "ids");
    let greedy_ids = oracle_ids(&oracle, "greedy");
    let prompt = oracle["prompt"].as_str().expect("invariant: prompt texte");

    let started = std::time::Instant::now();
    let assets = ModelAssets::load_local(&model_dir).expect("invariant: assets flint");
    let ours_ids = assets
        .tokenizer
        .encode(prompt)
        .expect("invariant: tokenisation")
        .into_iter()
        .map(|id| id as usize)
        .collect::<Vec<_>>();
    println!("tokens ours={ours_ids:?} mlx={prompt_ids:?}");
    assert_eq!(ours_ids, prompt_ids, "tokenizer divergent de mlx");
    let model = load_causal_decoder(&assets).expect("invariant: flint-int4 chargeable");
    println!("chargement: {:.1?}", started.elapsed());

    // Prompt position par position puis toute la continuation de référence,
    // en forçant ses tokens pour mesurer la dérive sans divergence. La ligne
    // `prompt_len - 1` porte aussi le premier pas greedy.
    let started = std::time::Instant::now();
    let mut cache = model.empty_cache();
    let mut rows = Vec::new();
    for token in prompt_ids.iter().chain(&greedy_ids[..greedy_ids.len() - 1]) {
        rows.push(
            model
                .next_logits_cached(&mut cache, *token)
                .expect("invariant: pas avec cache")
                .data()
                .to_vec(),
        );
    }
    println!("prompt + greedy: {:.1?}", started.elapsed());
    let ours_greedy = rows[prompt_ids.len() - 1..]
        .iter()
        .map(|row| argmax(row).expect("invariant: vocab non vide"))
        .collect::<Vec<_>>();
    let text = assets
        .tokenizer
        .decode(
            &ours_greedy.iter().map(|id| *id as u32).collect::<Vec<_>>(),
            false,
        )
        .expect("invariant: décodage");
    println!("texte: {prompt}{text:?}");

    if let Some(bf16_dir) = std::env::var_os("SARAGOSSA_FLINT_ORACLE_BF16_DIR") {
        compare_oracle("mlx-bf16", Path::new(&bf16_dir), &rows, prompt_ids.len());
    }
    let (max, mean, argmax_ok) = compare_oracle("mlx-f32", &oracle_dir, &rows, prompt_ids.len());
    assert_eq!(ours_greedy, greedy_ids, "tokens greedy divergents");
    assert!(argmax_ok, "argmax divergent sur au moins une position");
    assert!(max < MAX_GAP_F32, "écart max {max} ≥ {MAX_GAP_F32}");
    assert!(mean < MEAN_GAP_F32, "écart moyen {mean} ≥ {MEAN_GAP_F32}");
}

// Seuils contre mlx float32, écarts centrés (décalage constant retiré), sur
// les 5 positions du prompt et les 16 pas greedy.
// NOTE: Mesuré le 29 oct. 2026 (CPU f32, mlx-lm 4cfc6478 en float32) : pire
// max 4,16 et pire moyenne 0,70 (greedy[15]) ; la position 0 est à 0,004 près.
// L'écart croît par sauts non monotones, signe probable d'experts départagés
// différemment sur des quasi-égalités de scores (ordre d'accumulation f32),
// pas d'une erreur de formule. Les seuils laissent ~1,5× de marge et restent
// 3 à 4× sous l'écart propre de mlx bf16 contre mlx f32 (17,0 et 2,6).
const MAX_GAP_F32: f32 = 6.0;
const MEAN_GAP_F32: f32 = 1.0;

/// Decode résident Metal de flint-int4 (chargement Metal du CLI, prefill CPU)
/// contre l'oracle mlx-lm float32 : 16 tokens greedy identiques par le chemin
/// de production, puis logits GPU sur la continuation de référence forcée.
///
/// Mêmes variables d'environnement que [`flint_int4_real_weights_match_mlx_oracle`].
#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
#[ignore = "poids réels flint-int4 (22,6 Go) et oracle mlx-lm requis"]
fn flint_int4_resident_decode_matches_mlx_oracle() {
    let (Some(model_dir), Some(oracle_dir)) = (
        std::env::var_os("SARAGOSSA_FLINT_MODEL_DIR").map(PathBuf::from),
        std::env::var_os("SARAGOSSA_FLINT_ORACLE_DIR").map(PathBuf::from),
    ) else {
        eprintln!(
            "test flint Metal sauté : définir SARAGOSSA_FLINT_MODEL_DIR et \
             SARAGOSSA_FLINT_ORACLE_DIR"
        );
        return;
    };
    let oracle: serde_json::Value = serde_json::from_slice(
        &std::fs::read(oracle_dir.join("oracle.json")).expect("invariant: oracle.json"),
    )
    .expect("invariant: oracle JSON");
    let prompt_ids = oracle_ids(&oracle, "ids");
    let greedy_ids = oracle_ids(&oracle, "greedy");

    let started = std::time::Instant::now();
    let assets = ModelAssets::load_local(&model_dir).expect("invariant: assets flint");
    let executor = crate::MetalExecutor::new().expect("invariant: device Metal");
    // Même séquence que `saragossa run` (main.rs, `load_decoder_metal`).
    let model = crate::load_causal_decoder_for_metal(&assets, &executor)
        .expect("invariant: flint-int4 chargeable en Metal")
        .with_metal_executor(executor)
        .expect("invariant: runtime Metal attaché");
    println!("chargement Metal: {:.1?}", started.elapsed());
    assert!(
        model.supports_resident_full_decode(),
        "decode résident complet attendu pour flint"
    );

    let output = model
        .generate_greedy_timed_with_options(
            &prompt_ids,
            greedy_ids.len(),
            &crate::GenerationOptions::default(),
        )
        .expect("invariant: génération résidente");
    let timings = &output.timings;
    let decode_secs = timings.decode.as_secs_f64();
    println!(
        "prefill (CPU) {} tokens: {:.1?} ; decode résident {} tokens: {:.1?} → {:.1} tok/s",
        prompt_ids.len(),
        timings.prefill,
        timings.decode_tokens,
        timings.decode,
        timings.decode_tokens as f64 / decode_secs.max(f64::EPSILON)
    );
    println!("tokens résidents={:?}", output.tokens);
    assert_eq!(
        output.tokens, greedy_ids,
        "tokens greedy résidents divergents"
    );

    // Logits GPU des pas 1..16 : continuation de référence forcée, comparée
    // ligne à ligne aux logits mlx (la ligne 0 vient du prefill CPU).
    let rows = model
        .resident_forced_logits_for_test(&prompt_ids, &greedy_ids[..greedy_ids.len() - 1], false)
        .expect("invariant: logits résidents");
    let (_, reference) = read_npy_f32(&oracle_dir.join("oracle_greedy_logits.npy"));
    let vocab = rows[0].len();
    let (mut max, mut mean, mut argmax_ok) = (0.0_f32, 0.0_f32, true);
    for (index, row) in rows.iter().enumerate() {
        let step = index + 1;
        let gap = compare_rows(
            &format!("résident greedy[{step}]"),
            row,
            &reference[step * vocab..(step + 1) * vocab],
        );
        max = max.max(gap.max);
        mean = mean.max(gap.mean);
        argmax_ok &= gap.same_argmax;
    }
    println!("résident contre mlx-f32: pire max={max:.4}, pire moyenne={mean:.5}");
    // Le decode résident calcule en f32 (KV f32 en greedy court) la même
    // formule que l'oracle CPU ; seuls l'ordre d'accumulation et les kernels
    // diffèrent. Les seuils CPU contre mlx f32 s'appliquent donc tels quels.
    assert!(argmax_ok, "argmax résident divergent sur au moins un pas");
    assert!(max < MAX_GAP_F32, "écart max {max} ≥ {MAX_GAP_F32}");
    assert!(mean < MEAN_GAP_F32, "écart moyen {mean} ≥ {MEAN_GAP_F32}");
}
