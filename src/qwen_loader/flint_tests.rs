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
        // Prefill MoE qmv : l'écart absolu 1e-4 ne tolère pas l'arrondi bf16
        // du coop groupé, couvert par `synthetic_flint_moe_coop_matches_qmv`.
        let rows = crate::metal_backend::with_flint_moe_coop_override(false, || {
            gpu.resident_forced_logits_for_test(&prompt, &forced, kv_bf16)
        })
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
    let steps_before = resident_decode_steps();
    assert_eq!(
        gpu.generate_greedy(&prompt, 8)
            .expect("invariant: greedy GPU"),
        cpu.generate_greedy(&prompt, 8)
            .expect("invariant: greedy CPU"),
        "tokens greedy résident != CPU"
    );
    assert!(
        resident_decode_steps() - steps_before >= 7,
        "greedy GPU passé par le decode résident"
    );
}

/// Light-batch flint : time-slicing (le pas duo ne connaît pas flint), deux
/// flux, chacun identique au flux seul, et chaque pas par le decode résident.
/// Le flux de 600 tokens dépasse de loin la fenêtre synthétique de 2 ; il
/// tient en un morceau de prefill résident (4096 par défaut).
#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn synthetic_flint_lightbatch_matches_solo_on_resident_decode() {
    let gpu = load(flint_tensors(true))
        .expect("invariant: flint synthétique")
        .with_metal_runtime()
        .expect("invariant: runtime Metal attaché");
    let options = crate::GenerationOptions::default();
    let short = vec![1, 2, 3];
    let long = (0..600)
        .map(|index| (index * 5 + 1) % VOCAB)
        .collect::<Vec<_>>();
    let max_new = 8;
    let steps = resident_decode_steps();
    let batch = gpu
        .generate_greedy_lightbatch_with_options(&[short.clone(), long.clone()], max_new, &options)
        .expect("invariant: light-batch flint");
    assert_eq!(
        resident_decode_steps() - steps,
        batch
            .iter()
            .map(|output| output.timings.decode_tokens)
            .sum::<usize>(),
        "chaque pas du light-batch doit passer par le decode résident"
    );
    for (output, prompt) in batch.iter().zip([&short, &long]) {
        assert_eq!(output.tokens.len(), max_new);
        let solo = gpu
            .generate_greedy_timed_with_options(prompt, max_new, &options)
            .expect("invariant: solo flint")
            .tokens;
        assert_eq!(
            output.tokens,
            solo,
            "flux de {} tokens != solo",
            prompt.len()
        );
    }
}

/// Prefill résident Metal (sinks, NoPE, fenêtre 2 dépassée, MLP dense puis MoE
/// sigmoïde avec expert partagé sans gate) contre le prefill per-op CPU sur
/// toutes les positions, puis greedy (prefill Metal + decode résident) contre
/// le CPU.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn synthetic_flint_resident_prefill_matches_cpu_oracle() {
    let cpu = load(flint_tensors(true)).expect("invariant: flint synthétique");
    let gpu = load(flint_tensors(true))
        .expect("invariant: flint synthétique")
        .with_metal_runtime()
        .expect("invariant: runtime Metal attaché");
    let prompt = [1, 2, 3, 4, 0, 5, 6, 7, 3];
    let per_op = gpu
        .prefill_hidden_for_test(&prompt, None)
        .expect("invariant: prefill per-op flint")
        .expect("invariant: hidden per-op");
    // MoE qmv : oracle f32 exact ; le coop groupé (bf16) a son propre test.
    let resident = crate::metal_backend::with_flint_moe_coop_override(false, || {
        gpu.prefill_hidden_for_test(
            &prompt,
            Some(crate::runtime_flags::prefill_resident_chunk_tokens()),
        )
    })
    .expect("invariant: prefill résident flint")
    .expect("prefill résident flint attendu");
    assert_eq!(per_op.shape(), resident.shape());
    let gap = per_op
        .data()
        .iter()
        .zip(resident.data())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    // Hidden d'ordre 1 : seul l'ordre d'accumulation f32 sépare GPU et CPU.
    assert!(gap <= 1.0e-4, "hidden résident vs per-op: écart {gap}");
    assert_eq!(
        gpu.generate_greedy(&prompt, 6)
            .expect("invariant: greedy GPU"),
        cpu.generate_greedy(&prompt, 6)
            .expect("invariant: greedy CPU"),
        "tokens greedy (prefill Metal) != CPU"
    );
}

/// MoE routé groupé (coop, tensor-core) contre le qmv par paire, sur le
/// prefill résident flint u4 gs64 (routeur sigmoïde, expert partagé sans
/// gate). Le coop arrondit en bf16 les activations d'entrée et l'intermédiaire
/// SwiGLU ; le qmv garde du f32 : l'écart relatif attendu est de l'ordre de
/// quelques 2^-9 par couche MoE.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn synthetic_flint_moe_coop_matches_qmv() {
    let executor = crate::MetalExecutor::new().expect("invariant: device Metal");
    // Support détecté hors du compteur : la bibliothèque NA (MPP) contient les
    // pipelines coop groupés ; si elle compile, le coop doit s'exécuter.
    let supported = executor.na_library_available_for_test();
    let gpu = load(flint_tensors(true))
        .expect("invariant: flint synthétique")
        .with_metal_executor(executor)
        .expect("invariant: runtime Metal attaché");
    let (qmv, qmv_trace) = traced_coop_prefill(&gpu, false);
    let (coop, coop_trace) = traced_coop_prefill(&gpu, true);
    // Kill-switch flint coupé : aucune couche éligible.
    assert_eq!(qmv_trace, (0, 0), "qmv: couches coop inattendues");
    if !supported {
        // Fixture u4 gs64 : ses deux couches MoE restent éligibles et
        // retombent sur le qmv faute de pipelines.
        assert_eq!(coop_trace, (0, 2), "repli qmv attendu sans bibliothèque NA");
        eprintln!("skip: bibliothèque NA indisponible (macOS < 26 ?)");
        return;
    }
    assert_eq!(
        coop_trace,
        (2, 0),
        "coop non exécuté sur les deux couches MoE"
    );
    let gap = max_gap(qmv.data(), coop.data());
    let scale = qmv.data().iter().fold(0.0_f32, |acc, v| acc.max(v.abs()));
    eprintln!("flint coop vs qmv: écart {gap} échelle {scale}");
    // Mesuré : 0,038 pour une échelle de 16,7 (2,3e-3 relatif).
    assert!(
        gap <= 1.0e-2 * scale,
        "coop vs qmv: écart {gap} échelle {scale}"
    );
    let greedy = |coop| {
        crate::metal_backend::with_flint_moe_coop_override(coop, || {
            gpu.generate_greedy(&SYNTHETIC_COOP_PROMPT, 6)
                .expect("invariant: greedy GPU")
        })
    };
    assert_eq!(greedy(true), greedy(false), "greedy coop != qmv");
}

/// Repli exécuté : sans le pipeline scatter u4 du coop groupé, le prefill
/// résident flint doit passer par le qmv sur ses deux couches MoE et rendre
/// exactement les hidden du qmv.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn synthetic_flint_moe_coop_falls_back_without_pipelines() {
    let mut executor = crate::MetalExecutor::new().expect("invariant: device Metal");
    executor.drop_routed_coop_scatter_u4_for_test();
    let gpu = load(flint_tensors(true))
        .expect("invariant: flint synthétique")
        .with_metal_executor(executor)
        .expect("invariant: runtime Metal attaché");
    let (fallback, fallback_trace) = traced_coop_prefill(&gpu, true);
    let (qmv, qmv_trace) = traced_coop_prefill(&gpu, false);
    assert_eq!(
        fallback_trace,
        (0, 2),
        "repli qmv attendu sur les deux couches MoE"
    );
    assert_eq!(qmv_trace, (0, 0), "qmv: couches coop inattendues");
    assert_eq!(fallback.data(), qmv.data(), "repli != qmv");
}

#[cfg(all(target_os = "macos", feature = "metal"))]
const SYNTHETIC_COOP_PROMPT: [usize; 17] = [1, 2, 3, 4, 0, 5, 6, 7, 3, 2, 6, 1, 0, 4, 5, 7, 6];

/// Prefill résident du prompt synthétique, bascule flint coop forcée, et
/// couches MoE routées (coop exécuté, repli qmv faute de pipelines) comptées.
#[cfg(all(target_os = "macos", feature = "metal"))]
fn traced_coop_prefill(gpu: &CausalDecoder, coop: bool) -> (Tensor, (usize, usize)) {
    crate::metal_backend::with_flint_moe_coop_override(coop, || {
        let (ran, fallback) = crate::metal_backend::routed_coop_trace_for_test();
        let hidden = gpu
            .prefill_hidden_for_test(
                &SYNTHETIC_COOP_PROMPT,
                Some(crate::runtime_flags::prefill_resident_chunk_tokens()),
            )
            .expect("invariant: prefill résident flint")
            .expect("prefill résident flint attendu");
        let (ran_after, fallback_after) = crate::metal_backend::routed_coop_trace_for_test();
        (hidden, (ran_after - ran, fallback_after - fallback))
    })
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn max_gap(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

/// Prefill résident par morceaux (sinks à décalage : requêtes locales,
/// K/V, causalité et fenêtre absolus) contre le même prefill en un morceau,
/// sur des coupes qui tombent dans la fenêtre de 2 et autour.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn synthetic_flint_chunked_prefill_matches_single_chunk() {
    let gpu = load(flint_tensors(true))
        .expect("invariant: flint synthétique")
        .with_metal_runtime()
        .expect("invariant: runtime Metal attaché");
    let prompt = [1, 2, 3, 4, 0, 5, 6, 7, 3, 2, 6];
    let hidden = |chunk_tokens| {
        gpu.prefill_hidden_for_test(&prompt, Some(chunk_tokens))
            .expect("invariant: prefill résident flint")
            .expect("prefill résident flint attendu")
    };
    let single = hidden(0);
    for chunk_tokens in [1, 2, 3, 5] {
        let chunked = hidden(chunk_tokens);
        let gap = max_gap(single.data(), chunked.data());
        // Même arithmétique par ligne ; seuls les GEMM, dont la tuile dépend
        // du nombre de lignes, peuvent réordonner une somme.
        assert!(gap <= 1.0e-5, "morceaux de {chunk_tokens}: écart {gap}");
    }
}

/// Prolongement résident d'un préfixe (reliquat sous le seuil de 32 tokens
/// des autres modèles) contre le prefill complet, puis greedy depuis l'état
/// prolongé contre le CPU.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn synthetic_flint_prefix_extension_matches_full_prefill() {
    let cpu = load(flint_tensors(true)).expect("invariant: flint synthétique");
    let gpu = load(flint_tensors(true))
        .expect("invariant: flint synthétique")
        .with_metal_runtime()
        .expect("invariant: runtime Metal attaché");
    let prompt = [1, 2, 3, 4, 0, 5, 6, 7, 3, 2, 6];
    assert!(
        gpu.prefix_extend_as_cheap_as_cold(prompt.len()),
        "prolongement résident attendu pour flint"
    );
    let full = gpu
        .prefill_prompt_state_uncached(&prompt)
        .expect("invariant: prefill complet");
    for split in [1, 4, 9, 10] {
        let mut state = gpu
            .prefill_prompt_state_uncached(&prompt[..split])
            .expect("invariant: prefill du préfixe");
        gpu.extend_prompt_state(&mut state, &prompt[split..])
            .expect("invariant: prolongement résident");
        assert_eq!(state.position(), prompt.len());
        let gap = max_gap(
            full.final_state_for_test().data(),
            state.final_state_for_test().data(),
        );
        assert!(gap <= 1.0e-5, "préfixe de {split}: écart {gap}");
        let tokens = gpu
            .generate_greedy_timed_from_prompt_state_with_options(
                state,
                std::time::Duration::ZERO,
                4,
                &crate::GenerationOptions::default(),
            )
            .expect("invariant: greedy depuis l'état prolongé")
            .tokens;
        assert_eq!(
            tokens,
            cpu.generate_greedy(&prompt, 4)
                .expect("invariant: greedy CPU"),
            "greedy après préfixe de {split} != CPU"
        );
    }
}

/// Stockage bf16 du K/V de prefill (seuil forcé à 0, flag flint forcé)
/// contre le stockage f32 : prefill par morceaux, dont les morceaux relisent
/// le K/V arrondi par les jumeaux bf16 des kernels à sinks (pleine et
/// fenêtrée), puis prolongement d'un préfixe et greedy. La dérive des hidden
/// est chiffrée ; les tokens greedy doivent rester ceux du CPU. Sans le flag flint,
/// le seuil seul ne change aucun octet.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn synthetic_flint_prefill_kv_bf16_store_drift() {
    let cpu = load(flint_tensors(true)).expect("invariant: flint synthétique");
    let gpu = |min_rows, sinks| {
        let mut executor = crate::MetalExecutor::new().expect("invariant: executor Metal");
        executor.set_prefill_kv_bf16_store_min_rows_for_test(min_rows);
        executor.set_prefill_kv_bf16_store_sinks_for_test(sinks);
        load(flint_tensors(true))
            .expect("invariant: flint synthétique")
            .with_metal_executor(executor)
            .expect("invariant: runtime Metal attaché")
    };
    let (f32_store, bf16_store) = (gpu(None, false), gpu(Some(0), true));
    let opted_out = gpu(Some(0), false);
    let prompt = [1, 2, 3, 4, 0, 5, 6, 7, 3, 2, 6];
    for chunk_tokens in [1, 3, 5] {
        let hidden = |gpu: &CausalDecoder| {
            gpu.prefill_hidden_for_test(&prompt, Some(chunk_tokens))
                .expect("invariant: prefill résident flint")
                .expect("prefill résident flint attendu")
        };
        let reference = hidden(&f32_store);
        assert_eq!(
            max_gap(reference.data(), hidden(&opted_out).data()),
            0.0,
            "morceaux de {chunk_tokens}: stockage bf16 pris sans le flag flint"
        );
        let gap = max_gap(reference.data(), hidden(&bf16_store).data());
        let scale = reference
            .data()
            .iter()
            .fold(0.0_f32, |acc, value| acc.max(value.abs()));
        println!(
            "morceaux de {chunk_tokens}: dérive K/V bf16 {gap:.2e} (max |hidden| {scale:.2e}, relatif {:.2e})",
            gap / scale
        );
        // Écart nul : le stockage bf16 n'a pas été pris (tous les morceaux, le
        // premier compris, relisent le K/V depuis le cache). Mesuré le
        // 2026-10-30 : 5,8e-3 relatif à toutes les coupes, l'ordre de l'arrondi
        // bf16 (2⁻⁸) des K/V.
        assert!(
            gap > 0.0 && gap / scale <= 1.0e-2,
            "morceaux de {chunk_tokens}: dérive {gap}"
        );
    }
    let want = cpu
        .generate_greedy(&prompt, 4)
        .expect("invariant: greedy CPU");
    for split in [1, 4, 10] {
        let mut state = bf16_store
            .prefill_prompt_state_uncached(&prompt[..split])
            .expect("invariant: prefill du préfixe");
        bf16_store
            .extend_prompt_state(&mut state, &prompt[split..])
            .expect("invariant: prolongement résident");
        let tokens = bf16_store
            .generate_greedy_timed_from_prompt_state_with_options(
                state,
                std::time::Duration::ZERO,
                4,
                &crate::GenerationOptions::default(),
            )
            .expect("invariant: greedy depuis l'état prolongé")
            .tokens;
        assert_eq!(
            tokens, want,
            "greedy K/V bf16 après préfixe de {split} != CPU"
        );
    }
}

/// Adoption par l'arène de decode du K/V bf16 de flint (flag flint forcé, seuil
/// 0) : couche pleine à sinks et couches fenêtrées (fenêtre 2, dépassée par
/// le prompt et le decode). L'arène tient la réserve du prefill de chaque
/// couche pendant le decode, et les tokens échantillonnés (arène bf16) sont
/// ceux du semis par copie. Sans le flag flint, le K/V reste f32 : rien n'est
/// adoptable.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn synthetic_flint_arena_adopts_prefill_kv_bf16() {
    let gpu = |sinks| {
        let mut executor = crate::MetalExecutor::new().expect("invariant: executor Metal");
        executor.set_prefill_kv_bf16_store_min_rows_for_test(Some(0));
        executor.set_prefill_kv_bf16_store_sinks_for_test(sinks);
        executor.set_prefill_window_kv_for_test(false);
        load(flint_tensors(true))
            .expect("invariant: flint synthétique")
            .with_metal_executor(executor)
            .expect("invariant: runtime Metal attaché")
    };
    let prompt = [1, 2, 3, 4, 0, 5, 6, 7, 3, 2, 6];
    let opted_out = crate::metal_backend::with_prefill_kv_layout_override(3, 0, || {
        gpu(false).prefill_prompt_state_uncached(&prompt)
    })
    .expect("invariant: prefill résident flint");
    assert!(
        opted_out.kv_bf16_prefixes_for_test().is_empty(),
        "K/V bf16 sans le flag flint"
    );
    if !crate::decode_resident::kv_bf16_dense_by_default() {
        return;
    }
    // Échantillonné : l'arène est bf16 quelle que soit la longueur du prompt.
    let options = crate::GenerationOptions {
        temperature: 0.8,
        seed: 7,
        ..crate::GenerationOptions::default()
    };
    let bf16 = gpu(true);
    let (copied, tokens, adopted) =
        bf16.decode_with_and_without_adoption_for_test(&prompt, 3, 6, &options);
    assert!(
        adopted.iter().all(|&adopted| adopted),
        "arène sans adoption du K/V bf16 de flint: {adopted:?}"
    );
    assert_eq!(tokens.len(), 6);
    assert_eq!(tokens, copied, "adoption != semis par copie");
    // K/V borné (défaut) : les couches glissantes sèment leur queue par
    // copie, seule la couche pleine est adoptée ; mêmes tokens.
    let mut executor = crate::MetalExecutor::new().expect("invariant: executor Metal");
    executor.set_prefill_kv_bf16_store_min_rows_for_test(Some(0));
    executor.set_prefill_kv_bf16_store_sinks_for_test(true);
    let window = load(flint_tensors(true))
        .expect("invariant: flint synthétique")
        .with_metal_executor(executor)
        .expect("invariant: runtime Metal attaché");
    let (window_copied, window_tokens, window_adopted) =
        window.decode_with_and_without_adoption_for_test(&prompt, 3, 6, &options);
    // Couche 0 pleine adoptée, couches 1 et 2 glissantes bornées semées
    // par copie de leur queue.
    assert_eq!(
        window_adopted,
        vec![true, false, false],
        "adoption par couche (pleine, glissante, glissante)"
    );
    assert_eq!(window_tokens, window_copied, "K/V borné: adoption != copie");
    assert_eq!(window_tokens, tokens, "K/V borné != K/V plein");
}

/// K/V des couches glissantes borné à la fenêtre (2 lignes ici) contre le
/// K/V plein du kill-switch, en f32 puis en bf16 : hidden de toutes les
/// positions et tokens (greedy, arène f32 ; échantillonné, arène bf16)
/// bit-identiques, sur des morceaux de 1 (une compaction par morceau), 2
/// (dernier morceau plus court que la fenêtre), 3 et 5 lignes, puis sur un
/// tour 2 dont le préfixe est plus court ou plus long que la fenêtre. Les
/// couches glissantes ne gardent que leur fenêtre, la couche pleine tout.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn synthetic_flint_window_kv_matches_full_kv_bit_for_bit() {
    let gpu = |window, bf16| {
        let mut executor = crate::MetalExecutor::new().expect("invariant: executor Metal");
        executor.set_prefill_window_kv_for_test(window);
        executor.set_prefill_kv_bf16_store_min_rows_for_test(Some(0));
        executor.set_prefill_kv_bf16_store_sinks_for_test(bf16);
        load(flint_tensors(true))
            .expect("invariant: flint synthétique")
            .with_metal_executor(executor)
            .expect("invariant: runtime Metal attaché")
    };
    let bits = |tensor: &crate::Tensor| -> Vec<u32> {
        tensor.data().iter().map(|value| value.to_bits()).collect()
    };
    let decode = |gpu: &CausalDecoder, state, options: &crate::GenerationOptions| {
        gpu.generate_greedy_timed_from_prompt_state_with_options(
            state,
            std::time::Duration::ZERO,
            6,
            options,
        )
        .expect("invariant: decode résident")
        .tokens
    };
    let sampled = crate::GenerationOptions {
        temperature: 0.8,
        seed: 7,
        ..crate::GenerationOptions::default()
    };
    let prompt = [1, 2, 3, 4, 0, 5, 6, 7, 3, 2, 6];
    let len = prompt.len();
    for bf16 in [false, true] {
        let (bounded, full) = (gpu(true, bf16), gpu(false, bf16));
        let state = |gpu: &CausalDecoder, chunk, split: usize| {
            crate::metal_backend::with_prefill_kv_layout_override(chunk, 0, || {
                let mut state = gpu.prefill_prompt_state_uncached(&prompt[..split])?;
                gpu.extend_prompt_state(&mut state, &prompt[split..])?;
                Ok::<_, crate::InferError>(state)
            })
            .expect("invariant: prefill résident flint")
        };
        for chunk in [1, 2, 3, 5] {
            let hidden = |gpu: &CausalDecoder| {
                gpu.prefill_hidden_for_test(&prompt, Some(chunk))
                    .expect("invariant: prefill résident flint")
                    .expect("prefill résident flint attendu")
            };
            assert_eq!(
                bits(&hidden(&bounded)),
                bits(&hidden(&full)),
                "bf16={bf16} morceaux de {chunk}: hidden"
            );
        }
        // Préfixe complet (tour 1), puis plus court (1) ou plus long (4, 9)
        // que la fenêtre.
        for (chunk, split) in [
            (1, len),
            (2, len),
            (3, len),
            (5, len),
            (3, 1),
            (3, 4),
            (2, 9),
        ] {
            let label = format!("bf16={bf16} morceaux de {chunk}, préfixe {split}");
            let (kept, all) = (state(&bounded, chunk, split), state(&full, chunk, split));
            assert_eq!(
                kept.kv_rows_for_test(),
                vec![(0, len), (len - 2, 2), (len - 2, 2)],
                "{label}: K/V borné"
            );
            assert_eq!(all.kv_rows_for_test(), vec![(0, len); 3], "{label}");
            assert_eq!(
                bits(kept.final_state_for_test()),
                bits(all.final_state_for_test()),
                "{label}: hidden final"
            );
            for options in [&crate::GenerationOptions::default(), &sampled] {
                assert_eq!(
                    decode(&bounded, kept.clone(), options),
                    decode(&full, all.clone(), options),
                    "{label}: tokens (température {})",
                    options.temperature
                );
            }
        }
    }
}

/// Empreinte du decode résident flint pour comparer deux chemins bit à bit :
/// bits des logits forcés (K/V f32 puis bf16, la dernière ligne remplit la
/// capacité de l'arène), puis tokens échantillonnés avec arène adoptée et
/// semée par copie, en K/V plein puis borné (base non nulle des couches
/// glissantes).
#[cfg(all(target_os = "macos", feature = "metal"))]
fn flint_decode_fingerprint() -> Vec<Vec<u32>> {
    let prompt = [1, 2, 3, 4, 0, 5, 6, 7, 3, 2, 6];
    let forced = [4, 0, 5, 6, 3, 2];
    let gpu = load(flint_tensors(true))
        .expect("invariant: flint synthétique")
        .with_metal_runtime()
        .expect("invariant: runtime Metal attaché");
    let mut fingerprint = Vec::new();
    for kv_bf16 in [false, true] {
        let rows = crate::metal_backend::with_flint_moe_coop_override(false, || {
            gpu.resident_forced_logits_for_test(&prompt, &forced, kv_bf16)
        })
        .expect("invariant: decode résident flint");
        fingerprint.extend(
            rows.iter()
                .map(|row| row.iter().map(|value| value.to_bits()).collect()),
        );
    }
    if !crate::decode_resident::kv_bf16_dense_by_default() {
        return fingerprint;
    }
    let options = crate::GenerationOptions {
        temperature: 0.8,
        seed: 7,
        ..crate::GenerationOptions::default()
    };
    for window in [false, true] {
        let mut executor = crate::MetalExecutor::new().expect("invariant: executor Metal");
        executor.set_prefill_kv_bf16_store_min_rows_for_test(Some(0));
        executor.set_prefill_kv_bf16_store_sinks_for_test(true);
        executor.set_prefill_window_kv_for_test(window);
        let gpu = load(flint_tensors(true))
            .expect("invariant: flint synthétique")
            .with_metal_executor(executor)
            .expect("invariant: runtime Metal attaché");
        let (copied, adopted, flags) =
            gpu.decode_with_and_without_adoption_for_test(&prompt, 3, 6, &options);
        assert!(flags[0], "couche pleine adoptée (window={window})");
        for tokens in [copied, adopted] {
            fingerprint.push(tokens.iter().map(|&token| token as u32).collect());
        }
    }
    fingerprint
}

/// QKV concaténé de flint : un qmv au lieu de trois par couche, chaque ligne
/// de sortie calculée à l'identique. Empreinte du decode bit-identique au
/// chemin à trois projections, et chemin effectivement emprunté.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn synthetic_flint_qkv_concat_is_bit_identical() {
    let hits = || crate::decode_resident::QKV_CONCAT_NO_GATE_HITS.with(std::cell::Cell::get);
    let start = hits();
    let split = crate::runtime_flags::with_qkv_concat_override(false, flint_decode_fingerprint);
    assert_eq!(hits(), start, "QKV concaténé employé malgré le kill-switch");
    let fused = crate::runtime_flags::with_qkv_concat_override(true, flint_decode_fingerprint);
    // Trois couches par pas, six pas forcés, K/V f32 puis bf16 au moins.
    assert!(hits() - start >= 2 * 3 * 6, "QKV concaténé non emprunté");
    assert_eq!(fused, split, "QKV concaténé != trois projections");
}

/// Rotation de K et append K/V fusionnés dans le decode de flint : un
/// dispatch au lieu de trois par couche. Empreinte du decode bit-identique
/// au chemin historique, et chemin effectivement emprunté.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn synthetic_flint_rope_append_kv_is_bit_identical() {
    let hits = || crate::decode_resident::ROPE_APPEND_KV_HITS.with(std::cell::Cell::get);
    let start = hits();
    let split = crate::runtime_flags::with_rope_append_kv_override(false, flint_decode_fingerprint);
    assert_eq!(
        hits(),
        start,
        "append fusionné employé malgré le kill-switch"
    );
    let fused = crate::runtime_flags::with_rope_append_kv_override(true, flint_decode_fingerprint);
    assert!(hits() - start >= 2 * 3 * 6, "append fusionné non emprunté");
    assert_eq!(fused, split, "append fusionné != rotation puis append");
}

/// Le scratch du prefill résident flint passe par le scope de purge (marquage
/// volatil, éviction LRU au-delà de `SARAGOSSA_RUST_SCRATCH_CAP_GB`) après
/// chaque prefill, quelle que soit sa taille, et reste sous ce plafond.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn synthetic_flint_resident_prefill_scratch_is_released() {
    let gpu = load(flint_tensors(true))
        .expect("invariant: flint synthétique")
        .with_metal_runtime()
        .expect("invariant: runtime Metal attaché");
    let mut releases = gpu
        .scratch_stats_for_test()
        .expect("invariant: stats scratch")
        .2;
    for len in [3, 9, 17, 40] {
        let prompt = (0..len).map(|i| i % VOCAB).collect::<Vec<_>>();
        gpu.generate_greedy(&prompt, 1)
            .expect("invariant: prefill Metal flint");
        let releases_before = releases;
        let (entries, bytes, after) = gpu
            .scratch_stats_for_test()
            .expect("invariant: stats scratch");
        releases = after;
        assert!(entries > 0, "prefill résident sans scratch (len={len})");
        assert!(
            releases > releases_before,
            "scratch jamais purgé (len={len})"
        );
        assert!(
            bytes <= crate::runtime_flags::scratch_cap_bytes(),
            "scratch au-delà du plafond (len={len}): {bytes}"
        );
    }
}

/// Au-delà du plafond de prefill flint, refus explicite avant tout calcul
/// plutôt qu'un échec mémoire suivi d'un repli CPU : même règle pour la
/// vérification préalable de serve, le prefill à froid et le prolongement.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn synthetic_flint_refuses_prompt_beyond_prefill_cap() {
    let gpu = load(flint_tensors(true))
        .expect("invariant: flint synthétique")
        .with_metal_runtime()
        .expect("invariant: runtime Metal attaché");
    let cap = gpu
        .flint_prefill_max_tokens()
        .expect("invariant: plafond flint sous Metal");
    assert!((4096..=262_144).contains(&cap), "plafond {cap}");
    gpu.ensure_prompt_within_prefill_cap(cap)
        .expect("prompt au plafond accepté");
    let prompt = vec![1; cap + 1];
    let refused = |err: crate::InferError| {
        assert!(
            err.to_string()
                .contains("SARAGOSSA_RUST_FLINT_PREFILL_MAX_TOKENS"),
            "{err}"
        );
    };
    refused(
        gpu.ensure_prompt_within_prefill_cap(prompt.len())
            .expect_err("vérification préalable refusée"),
    );
    refused(
        gpu.generate_greedy(&prompt, 1)
            .expect_err("prefill à froid refusé"),
    );
    let mut state = gpu
        .prefill_prompt_state_uncached(&prompt[..4])
        .expect("invariant: prefill du préfixe");
    refused(
        gpu.extend_prompt_state(&mut state, &prompt[4..])
            .expect_err("prolongement refusé"),
    );
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
// Position du prompt de l'oracle où mlx bf16 s'écarte lui-même de mlx f32
// (16,94 / 2,62 contre au plus 1,30 / 0,23 ailleurs) : quasi-égalité de
// routage sensible à l'arrondi bf16, que le MoE coop groupé rencontre aussi.
#[cfg(all(target_os = "macos", feature = "metal"))]
const NEAR_TIE_POSITION: usize = 3;

/// Chemin Metal de flint-int4 (chargement Metal du CLI) contre l'oracle mlx-lm
/// float32 et contre l'oracle CPU F1 :
/// - prefill résident : logits des 5 positions du prompt contre mlx ;
/// - 16 tokens greedy identiques par le chemin de production (prefill Metal,
///   decode résident), avec temps de prefill CPU et Metal et débit du decode ;
/// - logits GPU sur la continuation de référence forcée ;
/// - prompt synthétique de 1 000 tokens (fenêtre de 513 dépassée) : logits du
///   prefill résident contre le prefill per-op CPU à cinq positions.
///
/// Mêmes variables d'environnement que [`flint_int4_real_weights_match_mlx_oracle`].
#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
#[ignore = "poids réels flint-int4 (22,6 Go) et oracle mlx-lm requis"]
fn flint_int4_metal_matches_mlx_oracle() {
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

    // 1. Logits du prompt : prefill résident (et per-op CPU, pour le temps
    //    « avant ») contre mlx f32. Le premier prefill Metal paie aussi la
    //    résolution des poids GPU ; le temps retenu est celui du greedy.
    //    MoE qmv (f32) ici ; le coop groupé (bf16), défaut, est mesuré après.
    let (cpu_hidden, metal_hidden, cpu_time, metal_time) =
        crate::metal_backend::with_flint_moe_coop_override(false, || {
            timed_prefills(&model, &prompt_ids)
        });
    println!(
        "prompt {} tokens : prefill CPU per-op {cpu_time:.1?}, prefill Metal à froid {metal_time:.1?}",
        prompt_ids.len()
    );
    let all_rows = (0..prompt_ids.len()).collect::<Vec<_>>();
    let metal_rows = model
        .logits_of_hidden_rows_for_test(&metal_hidden, &all_rows)
        .expect("invariant: logits du prefill Metal");
    let cpu_rows = model
        .logits_of_hidden_rows_for_test(&cpu_hidden, &all_rows)
        .expect("invariant: logits du prefill CPU");
    for (index, (metal, cpu)) in metal_rows.iter().zip(&cpu_rows).enumerate() {
        compare_rows(&format!("Metal vs CPU prompt[{index}]"), metal, cpu);
    }
    let (max, mean, argmax_ok) = compare_oracle(
        "prefill Metal mlx-f32",
        &oracle_dir,
        &metal_rows,
        prompt_ids.len(),
    );
    // Le prefill résident qmv calcule en f32 la formule de l'oracle CPU ; seuls
    // l'ordre d'accumulation et les kernels diffèrent : seuils CPU contre mlx.
    assert!(argmax_ok, "argmax du prefill Metal divergent");
    assert!(
        max < MAX_GAP_F32,
        "prefill: écart max {max} ≥ {MAX_GAP_F32}"
    );
    assert!(
        mean < MEAN_GAP_F32,
        "prefill: écart moyen {mean} ≥ {MEAN_GAP_F32}"
    );
    // MoE coop groupé (défaut) : activations et intermédiaire SwiGLU arrondis
    // en bf16. Mêmes seuils f32 que le qmv à toutes les positions, sauf la
    // position 3, qualifiée par la trace des experts choisis.
    let resident_with_router = |coop: bool| {
        crate::metal_backend::with_flint_moe_coop_override(coop, || {
            crate::metal_backend::with_router_capture(|| {
                model.prefill_hidden_for_test(
                    &prompt_ids,
                    Some(crate::runtime_flags::prefill_resident_chunk_tokens()),
                )
            })
        })
        .expect("invariant: capture du routeur")
    };
    let (coop_hidden, coop_router) = resident_with_router(true);
    let (_, qmv_router) = resident_with_router(false);
    let coop_hidden = coop_hidden
        .expect("invariant: prefill résident coop")
        .expect("prefill résident Metal attendu pour flint");
    let coop_rows = model
        .logits_of_hidden_rows_for_test(&coop_hidden, &all_rows)
        .expect("invariant: logits du prefill coop");
    let (_, mlx_prompt) = read_npy_f32(&oracle_dir.join("oracle_prompt_logits.npy"));
    let vocab = coop_rows[0].len();
    let mut near_tie_gap = (0.0_f32, 0.0_f32);
    for (index, row) in coop_rows.iter().enumerate() {
        let gap = compare_rows(
            &format!("prefill Metal coop mlx-f32 prompt[{index}]"),
            row,
            &mlx_prompt[index * vocab..(index + 1) * vocab],
        );
        assert!(
            gap.same_argmax,
            "argmax du prefill coop divergent [{index}]"
        );
        if index == NEAR_TIE_POSITION {
            near_tie_gap = (gap.max, gap.mean);
        } else {
            assert!(
                gap.max < MAX_GAP_F32 && gap.mean < MEAN_GAP_F32,
                "prefill coop [{index}]: écart {}/{} ≥ {MAX_GAP_F32}/{MEAN_GAP_F32}",
                gap.max,
                gap.mean
            );
        }
    }
    // Trace du routage : chaque couple (couche, position) dont le top-k
    // diffère entre coop et qmv doit être une quasi-égalité, c'est-à-dire un
    // écart entre le k-ième et le (k+1)-ième logit du qmv inférieur à la
    // perturbation coop des logits de routeur de cette ligne.
    // NOTE: Mesuré le 30 oct. 2026 sur les indices du kernel : 21 écarts sur 47 couches × 5 positions,
    // tous des quasi-égalités. Premier écart couche MoE 10, position 4 (marge
    // 0,00056, perturbation 0,011) ; position 3 dès la couche 24 (0,0015 pour
    // 0,021), puis perturbation de 8 à 22 aux couches 44-46, d'où l'écart de
    // 14,4 / 2,1 de cette position.
    assert_eq!(coop_router.len(), qmv_router.len(), "couches MoE capturées");
    let top_k = qmv_router[0].indices.len() / prompt_ids.len();
    let experts = qmv_router[0].logits.len() / prompt_ids.len();
    let ranked = |logits: &[f32]| {
        let mut order = (0..logits.len()).collect::<Vec<_>>();
        order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
        order
    };
    let mut flipped_positions = vec![false; prompt_ids.len()];
    for (layer, (coop, qmv)) in coop_router.iter().zip(&qmv_router).enumerate() {
        for (position, flipped) in flipped_positions.iter_mut().enumerate() {
            // Écart qualifié sur les indices choisis par le kernel ; les
            // logits ne servent qu'à mesurer marge et perturbation.
            let chosen = |capture: &crate::metal_backend::RouterCapture| {
                let mut set = capture.indices[position * top_k..(position + 1) * top_k].to_vec();
                set.sort_unstable();
                set
            };
            let (coop_set, qmv_set) = (chosen(coop), chosen(qmv));
            if coop_set == qmv_set {
                continue;
            }
            let span = position * experts..(position + 1) * experts;
            let (coop_logits, qmv_logits) = (&coop.logits[span.clone()], &qmv.logits[span]);
            let qmv_order = ranked(qmv_logits);
            let margin = qmv_logits[qmv_order[top_k - 1]] - qmv_logits[qmv_order[top_k]];
            let perturbation = coop_logits
                .iter()
                .zip(qmv_logits)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            println!(
                "routage coop≠qmv : couche MoE {layer}, position {position}, \
                 marge qmv {margin:.5}, perturbation {perturbation:.5}, \
                 coop {coop_set:?}, qmv {qmv_set:?}"
            );
            assert!(
                margin <= perturbation,
                "routage couche {layer} position {position} sans quasi-égalité : \
                 marge {margin} > perturbation {perturbation}"
            );
            *flipped = true;
        }
    }
    // Position 3 : sans écart de routage, seuils f32 ; avec, l'écart reste
    // sous celui de mlx bf16 contre mlx f32 à cette même position.
    let (max, mean) = near_tie_gap;
    if flipped_positions[NEAR_TIE_POSITION] {
        assert!(
            max < MAX_GAP_MLX_BF16 && mean < MEAN_GAP_MLX_BF16,
            "prefill coop [3]: écart {max}/{mean} ≥ {MAX_GAP_MLX_BF16}/{MEAN_GAP_MLX_BF16}"
        );
    } else {
        assert!(
            max < MAX_GAP_F32 && mean < MEAN_GAP_F32,
            "prefill coop [3]: écart {max}/{mean} sans écart de routage"
        );
    }

    // 2. Greedy de production : prefill Metal (à chaud) puis decode résident.
    let steps_before = resident_decode_steps();
    let output = model
        .generate_greedy_timed_with_options(
            &prompt_ids,
            greedy_ids.len(),
            &crate::GenerationOptions::default(),
        )
        .expect("invariant: génération résidente");
    assert_eq!(
        resident_decode_steps() - steps_before,
        output.timings.decode_tokens,
        "chaque pas de decode doit passer par le decode résident Metal"
    );
    let timings = &output.timings;
    let decode_secs = timings.decode.as_secs_f64();
    println!(
        "prefill Metal {} tokens: {:.1?} ; decode résident {} tokens: {:.1?} → {:.1} tok/s",
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

    // 3. Logits GPU des pas 1..16 : continuation de référence forcée, comparée
    //    ligne à ligne aux logits mlx (la ligne 0 vient du prefill).
    //    Prefill MoE qmv, pour garder les seuils f32.
    let rows = crate::metal_backend::with_flint_moe_coop_override(false, || {
        model.resident_forced_logits_for_test(
            &prompt_ids,
            &greedy_ids[..greedy_ids.len() - 1],
            false,
        )
    })
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
    assert!(argmax_ok, "argmax résident divergent sur au moins un pas");
    assert!(max < MAX_GAP_F32, "écart max {max} ≥ {MAX_GAP_F32}");
    assert!(mean < MEAN_GAP_F32, "écart moyen {mean} ≥ {MEAN_GAP_F32}");

    // 4. Prompt long synthétique contre le prefill per-op CPU (oracle F1) :
    //    positions de part et d'autre de la fenêtre de 513 tokens.
    //    MoE qmv (f32) ; le coop est mesuré à 16k par `flint_int4_moe_coop_matches_qmv`.
    let long_ids = synthetic_long_prompt(&assets, 1000);
    let (cpu_hidden, metal_hidden, cpu_time, metal_time) =
        crate::metal_backend::with_flint_moe_coop_override(false, || {
            timed_prefills(&model, &long_ids)
        });
    println!(
        "prompt long {} tokens : prefill CPU per-op {cpu_time:.1?}, prefill Metal {metal_time:.1?} → {:.0} tok/s",
        long_ids.len(),
        long_ids.len() as f64 / metal_time.as_secs_f64().max(f64::EPSILON)
    );
    let positions = [0, 511, 512, 513, long_ids.len() - 1];
    let metal_rows = model
        .logits_of_hidden_rows_for_test(&metal_hidden, &positions)
        .expect("invariant: logits du prefill Metal long");
    let cpu_rows = model
        .logits_of_hidden_rows_for_test(&cpu_hidden, &positions)
        .expect("invariant: logits du prefill CPU long");
    let (mut max, mut mean, mut same_argmax) = (0.0_f32, 0.0_f32, 0);
    for ((position, metal), cpu) in positions.iter().zip(&metal_rows).zip(&cpu_rows) {
        let gap = compare_rows(&format!("long Metal vs CPU [{position}]"), metal, cpu);
        max = max.max(gap.max);
        mean = mean.max(gap.mean);
        same_argmax += usize::from(gap.same_argmax);
    }
    println!(
        "long Metal vs CPU : pire max={max:.4}, pire moyenne={mean:.5}, argmax identiques {same_argmax}/{}",
        positions.len()
    );
    // NOTE: Même formule f32 des deux côtés ; seuls l'ordre d'accumulation et
    // d'éventuels experts départagés autrement sur des quasi-égalités les
    // séparent, comme CPU contre mlx f32 : mêmes seuils, argmax seulement
    // affichés (une quasi-égalité peut les intervertir sans erreur).
    assert!(max < MAX_GAP_F32, "long: écart max {max} ≥ {MAX_GAP_F32}");
    assert!(
        mean < MEAN_GAP_F32,
        "long: écart moyen {mean} ≥ {MEAN_GAP_F32}"
    );
}

/// Diagnostic de dérive : écrit dans `SARAGOSSA_FLINT_DUMP` (f32 LE) les
/// captures du flux résiduel en sortie des couches 0 (pleine, MLP dense),
/// 1 (pleine, MLP dense), 5 (pleine, après routage MoE) et 48 du prompt
/// synthétique de 1000 tokens, puis les logits de sa dernière ligne. Deux
/// exécutions sous des flags différents (`SARAGOSSA_RUST_FLINT_PREFILL_STEEL`,
/// morceaux `SARAGOSSA_FLINT_DUMP_CHUNK`) se comparent ensuite hors processus.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
#[ignore = "poids réels flint-int4 (22,6 Go) requis ; diagnostic, pas un gate"]
fn flint_int4_prefill_capture_dump() {
    let (Some(model_dir), Some(dump)) = (
        std::env::var_os("SARAGOSSA_FLINT_MODEL_DIR").map(PathBuf::from),
        std::env::var_os("SARAGOSSA_FLINT_DUMP").map(PathBuf::from),
    ) else {
        eprintln!("dump flint sauté : définir SARAGOSSA_FLINT_MODEL_DIR et SARAGOSSA_FLINT_DUMP");
        return;
    };
    let chunk_tokens = std::env::var("SARAGOSSA_FLINT_DUMP_CHUNK")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .unwrap_or(0);
    let assets = ModelAssets::load_local(&model_dir).expect("invariant: assets flint");
    let executor = crate::MetalExecutor::new().expect("invariant: device Metal");
    let model = crate::load_causal_decoder_for_metal(&assets, &executor)
        .expect("invariant: flint-int4 chargeable en Metal")
        .with_metal_executor(executor)
        .expect("invariant: runtime Metal attaché");
    let prompt = synthetic_long_prompt(&assets, 1000);
    let captures = model
        .prefill_captures_for_test(&prompt, &[0, 1, 5, 48], chunk_tokens)
        .expect("invariant: prefill résident flint")
        .expect("captures attendues");
    let hidden = model
        .prefill_hidden_for_test(&prompt, Some(chunk_tokens))
        .expect("invariant: prefill résident flint")
        .expect("hidden attendu");
    let logits = model
        .logits_of_hidden_rows_for_test(&hidden, &[prompt.len() - 1])
        .expect("invariant: logits");
    let bytes = captures
        .data()
        .iter()
        .chain(&logits[0])
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    std::fs::write(&dump, bytes).expect("invariant: écriture du dump");
    // Oracle CPU per-op des couches 0 et 1 (attention pleine, MLP dense),
    // sur demande : ~20 s.
    if let Some(cpu_dump) = std::env::var_os("SARAGOSSA_FLINT_DUMP_CPU") {
        let outputs = model
            .per_op_layer_outputs_for_test(&prompt, 2)
            .expect("invariant: forward per-op");
        let bytes = outputs
            .iter()
            .flat_map(|output| output.data().iter())
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        std::fs::write(cpu_dump, bytes).expect("invariant: écriture du dump CPU");
    }
}

/// Mesure du prefill résident flint (chemin AR, dernière ligne seulement) à
/// 1k, 4k et 16k tokens, une exécution par longueur après un échauffement.
/// L'état « avant » se mesure avec le même binaire :
/// `SARAGOSSA_RUST_FLINT_PREFILL_STEEL=0 SARAGOSSA_RUST_PREFILL_RESIDENT_CHUNK=0`
/// (kernel F3 seul, un seul morceau). `SARAGOSSA_FLINT_BENCH_LENS` remplace les
/// longueurs (liste séparée par des virgules).
#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
#[ignore = "poids réels flint-int4 (22,6 Go) requis ; mesure, pas un gate"]
fn flint_int4_prefill_ttft_bench() {
    let Some(model_dir) = std::env::var_os("SARAGOSSA_FLINT_MODEL_DIR").map(PathBuf::from) else {
        eprintln!("bench flint sauté : définir SARAGOSSA_FLINT_MODEL_DIR");
        return;
    };
    let lens = std::env::var("SARAGOSSA_FLINT_BENCH_LENS")
        .ok()
        .map(|raw| {
            raw.split(',')
                .filter_map(|len| len.trim().parse::<usize>().ok())
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| vec![1024, 4096, 16384]);
    let assets = ModelAssets::load_local(&model_dir).expect("invariant: assets flint");
    let executor = crate::MetalExecutor::new().expect("invariant: device Metal");
    let model = crate::load_causal_decoder_for_metal(&assets, &executor)
        .expect("invariant: flint-int4 chargeable en Metal")
        .with_metal_executor(executor)
        .expect("invariant: runtime Metal attaché");
    let warmup = synthetic_long_prompt(&assets, 64);
    model
        .prefill_cache_state_metal_resident_for_test(&warmup)
        .expect("invariant: échauffement");
    println!(
        "plafond de prefill flint: {:?} tokens",
        model.flint_prefill_max_tokens()
    );
    for len in lens {
        let prompt = synthetic_long_prompt(&assets, len);
        let started = std::time::Instant::now();
        let state = model
            .prefill_cache_state_metal_resident_for_test(&prompt)
            .expect("invariant: prefill résident flint");
        let elapsed = started.elapsed();
        assert!(state.is_some(), "prefill résident flint attendu");
        drop(state);
        println!(
            "flint prefill {len} tokens: {:.2} s, {:.0} tok/s",
            elapsed.as_secs_f64(),
            len as f64 / elapsed.as_secs_f64().max(f64::EPSILON)
        );
    }
}

/// Stockage bf16 du K/V de prefill sur flint-int4 réel : prompt synthétique
/// de `SARAGOSSA_FLINT_BF16_LEN` tokens (16 400 par défaut, au-delà du seuil de
/// 16 384), prefill résident par morceaux avec K/V f32 puis bf16 (seuil et
/// flag flint forcés par override), format du cache vérifié (tout bf16 ou
/// tout f32), logits comparés aux frontières réelles des morceaux (écarts
/// centrés, argmax) et 16 tokens greedy de production identiques. Une seule
/// copie des poids en mémoire à la fois.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
#[ignore = "poids réels flint-int4 (22,6 Go) requis ; mesure de dérive, pas un gate"]
fn flint_int4_prefill_kv_bf16_store_drift() {
    let Some(model_dir) = std::env::var_os("SARAGOSSA_FLINT_MODEL_DIR").map(PathBuf::from) else {
        eprintln!("test flint bf16 sauté : définir SARAGOSSA_FLINT_MODEL_DIR");
        return;
    };
    let len = std::env::var("SARAGOSSA_FLINT_BF16_LEN")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(16_400);
    let assets = ModelAssets::load_local(&model_dir).expect("invariant: assets flint");
    let prompt = synthetic_long_prompt(&assets, len);
    let target = crate::runtime_flags::prefill_resident_chunk_tokens();
    // Morceaux rééquilibrés comme le prefill résident : 16 400 tokens en
    // cible 4096 font cinq morceaux de 3280.
    let chunk = len.div_ceil(len.div_ceil(target.max(1)));
    // Dernière ligne du morceau précédent, première et milieu de chaque
    // morceau, puis les dernières lignes.
    let mut rows = (0..len)
        .step_by(chunk)
        .flat_map(|start| [start.saturating_sub(1), start, start + chunk / 2])
        .chain(len.saturating_sub(4)..len)
        .filter(|&row| row < len)
        .collect::<Vec<_>>();
    rows.sort_unstable();
    rows.dedup();
    println!("{len} tokens : morceaux de {chunk}, lignes {rows:?}");
    let run = |bf16: bool| {
        let mut executor = crate::MetalExecutor::new().expect("invariant: device Metal");
        executor.set_prefill_kv_bf16_store_min_rows_for_test(bf16.then_some(0));
        executor.set_prefill_kv_bf16_store_sinks_for_test(bf16);
        let model = crate::load_causal_decoder_for_metal(&assets, &executor)
            .expect("invariant: flint-int4 chargeable en Metal")
            .with_metal_executor(executor)
            .expect("invariant: runtime Metal attaché");
        let label = if bf16 { "bf16" } else { "f32" };
        // Chemin exécuté : le cache rendu doit être tout bf16 ou tout f32,
        // sinon la comparaison serait f32 contre f32 sans le dire.
        let (cache, _) = model
            .prefill_cache_state_metal_resident_for_test(&prompt)
            .expect("invariant: prefill résident")
            .expect("prefill résident Metal attendu pour flint");
        let (bf16_layers, layers) = cache.bf16_kv_layers_for_test();
        drop(cache);
        println!("K/V {label}: {bf16_layers}/{layers} couches stockées en bf16");
        assert!(layers > 0, "aucune couche full-attention");
        assert_eq!(
            bf16_layers,
            if bf16 { layers } else { 0 },
            "K/V {label}: format du cache inattendu"
        );
        let started = std::time::Instant::now();
        let hidden = model
            .prefill_hidden_for_test(&prompt, Some(target))
            .expect("invariant: prefill résident")
            .expect("prefill résident Metal attendu pour flint");
        println!(
            "K/V {label}: prefill {len} tokens en {:.1?}",
            started.elapsed()
        );
        let logits = model
            .logits_of_hidden_rows_for_test(&hidden, &rows)
            .expect("invariant: logits du prefill");
        drop(hidden);
        let tokens = model
            .generate_greedy_timed_with_options(&prompt, 16, &crate::GenerationOptions::default())
            .expect("invariant: génération résidente")
            .tokens;
        (logits, tokens)
    };
    let (f32_logits, f32_tokens) = run(false);
    let (bf16_logits, bf16_tokens) = run(true);
    let (mut max, mut mean, mut argmax_ok) = (0.0_f32, 0.0_f32, 0_usize);
    for (row, (bf16, f32)) in rows.iter().zip(bf16_logits.iter().zip(&f32_logits)) {
        let gap = compare_rows(&format!("ligne {row}"), bf16, f32);
        max = max.max(gap.max);
        mean = mean.max(gap.mean);
        argmax_ok += usize::from(gap.same_argmax);
    }
    println!(
        "dérive K/V bf16 vs f32 sur {} lignes : max {max:.4}, pire moyenne {mean:.5}, argmax {argmax_ok}/{}",
        rows.len(),
        rows.len()
    );
    println!("greedy f32={f32_tokens:?} bf16={bf16_tokens:?}");
    assert_eq!(bf16_tokens, f32_tokens, "greedy K/V bf16 != f32");
    assert_eq!(argmax_ok, rows.len(), "argmax K/V bf16 divergent");
    // NOTE: Mesuré le 30 oct. 2026 (16 400 tokens, cinq morceaux de 3280,
    // 49/49 couches en bf16) : aux frontières (dernière et première ligne
    // de chaque morceau) et aux milieux, max 0,64 à 2,02 et moyenne 0,11 à
    // 0,34 (pire : ligne 3280, début du 2e morceau) ; argmax 18/18, greedy
    // identique. Hors frontières, une mesure précédente relevait 4,06/0,77
    // (ligne 2048) et 7,32/1,34 (ligne 16 384) : la dérive dépend de la
    // position, pas du découpage.
    // Bornes larges d'un garde-fou de mesure, pas un budget de qualité : la
    // qualité est validée en serve (intégrité 4/4 sur 7 contextes de 32k à
    // 115k, cf. `flint_prefill_kv_bf16_store_enabled`).
    assert!(
        max < MAX_GAP_MLX_BF16,
        "écart max {max} ≥ {MAX_GAP_MLX_BF16}"
    );
    assert!(
        mean < MEAN_GAP_MLX_BF16,
        "écart moyen {mean} ≥ {MEAN_GAP_MLX_BF16}"
    );
}

/// NAX fenêtré avec sinks (35 couches glissantes) contre le naïf bf16 sur
/// flint-int4 réel : prompt synthétique de `SARAGOSSA_FLINT_BF16_LEN` tokens
/// (16 400 par défaut), K/V stocké en bf16 dans les deux exécutions (seul
/// cas où le NAX fenêtré s'applique), logits aux frontières et milieux des
/// morceaux, argmax et 16 tokens greedy. Le chemin pris est journalisé et
/// vérifié (pipeline présent ou retiré) ; des logits bit à bit égaux
/// signaleraient un NAX jamais exécuté.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
#[ignore = "poids réels flint-int4 (22,6 Go) requis ; mesure de dérive, pas un gate"]
fn flint_int4_prefill_nax_window_drift() {
    let Some(model_dir) = std::env::var_os("SARAGOSSA_FLINT_MODEL_DIR").map(PathBuf::from) else {
        eprintln!("test flint NAX fenêtré sauté : définir SARAGOSSA_FLINT_MODEL_DIR");
        return;
    };
    let len = std::env::var("SARAGOSSA_FLINT_BF16_LEN")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(16_400);
    let assets = ModelAssets::load_local(&model_dir).expect("invariant: assets flint");
    let prompt = synthetic_long_prompt(&assets, len);
    let target = crate::runtime_flags::prefill_resident_chunk_tokens();
    let chunk = len.div_ceil(len.div_ceil(target.max(1)));
    let mut rows = (0..len)
        .step_by(chunk)
        .flat_map(|start| [start.saturating_sub(1), start, start + chunk / 2])
        .chain(len.saturating_sub(4)..len)
        .filter(|&row| row < len)
        .collect::<Vec<_>>();
    rows.sort_unstable();
    rows.dedup();
    println!("{len} tokens : morceaux de {chunk}, lignes {rows:?}");
    let run = |nax: bool| {
        let mut executor = crate::MetalExecutor::new().expect("invariant: device Metal");
        executor.set_prefill_kv_bf16_store_min_rows_for_test(Some(0));
        executor.set_prefill_kv_bf16_store_sinks_for_test(true);
        let present = executor.keep_prefill_attn_nax_window_for_test(nax);
        println!("NAX fenêtré demandé={nax}, pipeline présent={present}");
        assert_eq!(present, nax, "NAX fenêtré absent (MPP ? flag ?)");
        let model = crate::load_causal_decoder_for_metal(&assets, &executor)
            .expect("invariant: flint-int4 chargeable en Metal")
            .with_metal_executor(executor)
            .expect("invariant: runtime Metal attaché");
        let started = std::time::Instant::now();
        let hidden = model
            .prefill_hidden_for_test(&prompt, Some(target))
            .expect("invariant: prefill résident")
            .expect("prefill résident Metal attendu pour flint");
        println!(
            "NAX fenêtré={nax}: prefill {len} tokens en {:.1?}",
            started.elapsed()
        );
        let logits = model
            .logits_of_hidden_rows_for_test(&hidden, &rows)
            .expect("invariant: logits du prefill");
        drop(hidden);
        let tokens = model
            .generate_greedy_timed_with_options(&prompt, 16, &crate::GenerationOptions::default())
            .expect("invariant: génération résidente")
            .tokens;
        (logits, tokens)
    };
    let (naive_logits, naive_tokens) = run(false);
    let (nax_logits, nax_tokens) = run(true);
    let (mut max, mut mean, mut argmax_ok) = (0.0_f32, 0.0_f32, 0_usize);
    for (row, (nax, naive)) in rows.iter().zip(nax_logits.iter().zip(&naive_logits)) {
        let gap = compare_rows(&format!("ligne {row}"), nax, naive);
        max = max.max(gap.max);
        mean = mean.max(gap.mean);
        argmax_ok += usize::from(gap.same_argmax);
    }
    println!(
        "dérive NAX fenêtré vs naïf sur {} lignes : max {max:.4}, pire moyenne {mean:.5}, argmax {argmax_ok}/{}",
        rows.len(),
        rows.len()
    );
    println!("greedy naïf={naive_tokens:?} nax={nax_tokens:?}");
    assert!(max > 0.0, "logits identiques : NAX fenêtré non exécuté ?");
    assert_eq!(nax_tokens, naive_tokens, "greedy NAX fenêtré != naïf");
    assert_eq!(argmax_ok, rows.len(), "argmax NAX fenêtré divergent");
    // NOTE: Mesuré le 31 oct. 2026 (16 400 tokens, cinq morceaux de 3280) :
    // max 3,57 (ligne 9840), pire moyenne 0,34, argmax 18/18, greedy
    // identique. Bornes propres à cette dérive (bf16 contre bf16), reprises
    // des seuils f32 de l'oracle mlx et non des bornes mlx bf16 (17/2,6),
    // trop larges pour détecter une régression du masque de fenêtre.
    assert!(max < MAX_GAP_F32, "écart max {max} ≥ {MAX_GAP_F32}");
    assert!(mean < MEAN_GAP_F32, "écart moyen {mean} ≥ {MEAN_GAP_F32}");
}

// Écart propre de mlx bf16 contre mlx f32 sur l'oracle flint (cf. les seuils
// f32 ci-dessus) : garde-fou large de la dérive d'un K/V de prefill stocké en
// bf16, pas un budget transférable.
#[cfg(all(target_os = "macos", feature = "metal"))]
const MAX_GAP_MLX_BF16: f32 = 17.0;
#[cfg(all(target_os = "macos", feature = "metal"))]
const MEAN_GAP_MLX_BF16: f32 = 2.6;

/// MoE routé groupé (coop) contre qmv sur flint-int4 réel : prompt
/// synthétique de `SARAGOSSA_FLINT_COOP_LEN` tokens (16 384 par défaut),
/// logits aux frontières et milieux des morceaux, argmax et 16 tokens greedy.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
#[ignore = "poids réels flint-int4 (22,6 Go) requis ; mesure de dérive, pas un gate"]
fn flint_int4_moe_coop_matches_qmv() {
    let Some(model_dir) = std::env::var_os("SARAGOSSA_FLINT_MODEL_DIR").map(PathBuf::from) else {
        eprintln!("test flint coop sauté : définir SARAGOSSA_FLINT_MODEL_DIR");
        return;
    };
    let len = std::env::var("SARAGOSSA_FLINT_COOP_LEN")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(16_384);
    let assets = ModelAssets::load_local(&model_dir).expect("invariant: assets flint");
    let prompt = synthetic_long_prompt(&assets, len);
    let target = crate::runtime_flags::prefill_resident_chunk_tokens();
    let chunk = len.div_ceil(len.div_ceil(target.max(1)));
    let mut rows = (0..len)
        .step_by(chunk)
        .flat_map(|start| [start.saturating_sub(1), start, start + chunk / 2])
        .chain(len.saturating_sub(4)..len)
        .filter(|&row| row < len)
        .collect::<Vec<_>>();
    rows.sort_unstable();
    rows.dedup();
    let executor = crate::MetalExecutor::new().expect("invariant: device Metal");
    let model = crate::load_causal_decoder_for_metal(&assets, &executor)
        .expect("invariant: flint-int4 chargeable en Metal")
        .with_metal_executor(executor)
        .expect("invariant: runtime Metal attaché");
    let run = |coop: bool| {
        crate::metal_backend::with_flint_moe_coop_override(coop, || {
            let started = std::time::Instant::now();
            let hidden = model
                .prefill_hidden_for_test(&prompt, Some(target))
                .expect("invariant: prefill résident")
                .expect("prefill résident Metal attendu pour flint");
            println!(
                "MoE coop={coop}: prefill {len} tokens en {:.1?}",
                started.elapsed()
            );
            let logits = model
                .logits_of_hidden_rows_for_test(&hidden, &rows)
                .expect("invariant: logits du prefill");
            drop(hidden);
            let tokens = model
                .generate_greedy_timed_with_options(
                    &prompt,
                    16,
                    &crate::GenerationOptions::default(),
                )
                .expect("invariant: génération résidente")
                .tokens;
            (logits, tokens)
        })
    };
    let (qmv_logits, qmv_tokens) = run(false);
    let (coop_logits, coop_tokens) = run(true);
    let (mut max, mut mean, mut argmax_ok) = (0.0_f32, 0.0_f32, 0_usize);
    for (row, (coop, qmv)) in rows.iter().zip(coop_logits.iter().zip(&qmv_logits)) {
        let gap = compare_rows(&format!("ligne {row}"), coop, qmv);
        max = max.max(gap.max);
        mean = mean.max(gap.mean);
        argmax_ok += usize::from(gap.same_argmax);
    }
    println!(
        "dérive MoE coop vs qmv sur {} lignes : max {max:.4}, pire moyenne {mean:.5}, argmax {argmax_ok}/{}",
        rows.len(),
        rows.len()
    );
    println!("greedy qmv={qmv_tokens:?} coop={coop_tokens:?}");
    assert_eq!(coop_tokens, qmv_tokens, "greedy MoE coop != qmv");
    assert_eq!(argmax_ok, rows.len(), "argmax MoE coop divergent");
    // Garde-fou de mesure, pas un budget de qualité (cf. MAX_GAP_MLX_BF16).
    assert!(
        max < MAX_GAP_MLX_BF16,
        "écart max {max} ≥ {MAX_GAP_MLX_BF16}"
    );
    assert!(
        mean < MEAN_GAP_MLX_BF16,
        "écart moyen {mean} ≥ {MEAN_GAP_MLX_BF16}"
    );
}

/// Hidden non normés du prefill per-op CPU puis du prefill résident Metal, et
/// leurs durées.
#[cfg(all(target_os = "macos", feature = "metal"))]
fn timed_prefills(
    model: &CausalDecoder,
    prompt: &[usize],
) -> (Tensor, Tensor, std::time::Duration, std::time::Duration) {
    let started = std::time::Instant::now();
    let cpu = model
        .prefill_hidden_for_test(prompt, None)
        .expect("invariant: prefill per-op");
    let cpu_time = started.elapsed();
    let started = std::time::Instant::now();
    let metal = model
        .prefill_hidden_for_test(
            prompt,
            Some(crate::runtime_flags::prefill_resident_chunk_tokens()),
        )
        .expect("invariant: prefill résident");
    let metal_time = started.elapsed();
    let metal = metal.expect("prefill résident Metal attendu pour flint");
    (
        cpu.expect("invariant: hidden per-op"),
        metal,
        cpu_time,
        metal_time,
    )
}

/// Texte synthétique déterministe tokenisé, tronqué à `len` tokens.
#[cfg(all(target_os = "macos", feature = "metal"))]
fn synthetic_long_prompt(assets: &ModelAssets, len: usize) -> Vec<usize> {
    const SUBJECTS: [&str; 5] = [
        "The engineer",
        "A small robot",
        "The archive",
        "Our team",
        "The river",
    ];
    const VERBS: [&str; 5] = ["records", "inspects", "rewrites", "measures", "follows"];
    const OBJECTS: [&str; 5] = [
        "the cache layout",
        "a sliding window",
        "every expert route",
        "the attention sink",
        "an old ledger",
    ];
    let mut text = String::new();
    let mut index = 0_usize;
    loop {
        text.push_str(&format!(
            "{} {} {} on day {}, then writes note {} about it.\n",
            SUBJECTS[index % 5],
            VERBS[(index / 5) % 5],
            OBJECTS[(index / 25) % 5],
            index % 31,
            index * 7 % 101
        ));
        index += 1;
        let ids = assets
            .tokenizer
            .encode(&text)
            .expect("invariant: tokenisation");
        if ids.len() >= len {
            return ids[..len].iter().map(|&id| id as usize).collect();
        }
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn resident_decode_steps() -> usize {
    crate::decoder::resident_decode_steps_for_test()
}

/// K/V des couches glissantes borné à la fenêtre sur flint-int4 réel contre
/// le K/V plein du kill-switch, en f32 puis en bf16 (stockage forcé) : prompt
/// de `SARAGOSSA_FLINT_BF16_LEN` tokens (16 400 par défaut, cinq morceaux),
/// hidden de toutes les positions et 16 tokens greedy bit-identiques, couches
/// glissantes réduites à leur fenêtre. Une seule copie des poids à la fois.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
#[ignore = "poids réels flint-int4 (22,6 Go) requis"]
fn flint_int4_window_kv_matches_full_kv() {
    let Some(model_dir) = std::env::var_os("SARAGOSSA_FLINT_MODEL_DIR").map(PathBuf::from) else {
        eprintln!("test flint fenêtre sauté : définir SARAGOSSA_FLINT_MODEL_DIR");
        return;
    };
    let len = std::env::var("SARAGOSSA_FLINT_BF16_LEN")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(16_400);
    let assets = ModelAssets::load_local(&model_dir).expect("invariant: assets flint");
    let prompt = synthetic_long_prompt(&assets, len);
    let chunk = crate::runtime_flags::prefill_resident_chunk_tokens();
    let run = |window: bool, bf16: bool| {
        let mut executor = crate::MetalExecutor::new().expect("invariant: device Metal");
        executor.set_prefill_window_kv_for_test(window);
        executor.set_prefill_kv_bf16_store_min_rows_for_test(bf16.then_some(0));
        executor.set_prefill_kv_bf16_store_sinks_for_test(bf16);
        let model = crate::load_causal_decoder_for_metal(&assets, &executor)
            .expect("invariant: flint-int4 chargeable en Metal")
            .with_metal_executor(executor)
            .expect("invariant: runtime Metal attaché");
        let started = std::time::Instant::now();
        let hidden = model
            .prefill_hidden_for_test(&prompt, Some(chunk))
            .expect("invariant: prefill résident")
            .expect("prefill résident Metal attendu pour flint");
        let hidden = hidden
            .data()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>();
        let state = model
            .prefill_prompt_state_uncached(&prompt)
            .expect("invariant: prefill résident");
        let rows = state.kv_rows_for_test();
        let tokens = model
            .generate_greedy_timed_from_prompt_state_with_options(
                state,
                std::time::Duration::ZERO,
                16,
                &crate::GenerationOptions::default(),
            )
            .expect("invariant: génération résidente")
            .tokens;
        println!(
            "fenêtre={window} bf16={bf16}: {:.1?}, greedy {tokens:?}",
            started.elapsed()
        );
        (hidden, rows, tokens)
    };
    for bf16 in [false, true] {
        let (window_hidden, window_rows, window_tokens) = run(true, bf16);
        let (full_hidden, full_rows, full_tokens) = run(false, bf16);
        let bounded = window_rows
            .iter()
            .filter(|(_, stored)| *stored == 513)
            .count();
        println!(
            "bf16={bf16}: {bounded}/{} couches réduites à 513 lignes",
            window_rows.len()
        );
        assert!(
            bounded > 0
                && full_rows
                    .iter()
                    .all(|&(base, stored)| (base, stored) == (0, len))
        );
        assert!(window_rows
            .iter()
            .all(|&(base, stored)| base + stored == len && (base == 0 || stored == 513)));
        assert!(
            window_hidden == full_hidden,
            "bf16={bf16}: hidden K/V borné != K/V plein"
        );
        assert_eq!(window_tokens, full_tokens, "bf16={bf16}: greedy");
    }
}

/// Adoption du K/V bf16 de flint-int4 réel par l'arène de decode : prompt de
/// `SARAGOSSA_FLINT_BF16_LEN` tokens (16 400 par défaut, au-delà des seuils
/// de stockage et d'arène bf16), flag flint forcé, 16 tokens greedy identiques
/// au semis par copie et réserve tenue par l'arène de toutes les couches.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
#[ignore = "poids réels flint-int4 (22,6 Go) requis"]
fn flint_int4_arena_adopts_prefill_kv_bf16() {
    let Some(model_dir) = std::env::var_os("SARAGOSSA_FLINT_MODEL_DIR").map(PathBuf::from) else {
        eprintln!("test flint adoption sauté : définir SARAGOSSA_FLINT_MODEL_DIR");
        return;
    };
    let len = std::env::var("SARAGOSSA_FLINT_BF16_LEN")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(16_400);
    let assets = ModelAssets::load_local(&model_dir).expect("invariant: assets flint");
    let prompt = synthetic_long_prompt(&assets, len);
    let mut executor = crate::MetalExecutor::new().expect("invariant: device Metal");
    executor.set_prefill_kv_bf16_store_min_rows_for_test(Some(0));
    executor.set_prefill_kv_bf16_store_sinks_for_test(true);
    // Couches glissantes bornées à leur fenêtre : queue recopiée, l'adoption
    // de toutes les couches n'existe que sous le kill-switch.
    executor.set_prefill_window_kv_for_test(false);
    let model = crate::load_causal_decoder_for_metal(&assets, &executor)
        .expect("invariant: flint-int4 chargeable en Metal")
        .with_metal_executor(executor)
        .expect("invariant: runtime Metal attaché");
    let chunk = crate::runtime_flags::prefill_resident_chunk_tokens();
    let started = std::time::Instant::now();
    let (copied, tokens, adopted) = model.decode_with_and_without_adoption_for_test(
        &prompt,
        chunk,
        16,
        &crate::GenerationOptions::default(),
    );
    println!(
        "{len} tokens en {:.1?} : copie {copied:?}, adoption {tokens:?}",
        started.elapsed()
    );
    assert!(
        adopted.iter().all(|&adopted| adopted),
        "arène sans adoption du K/V bf16 de flint-int4: {adopted:?}"
    );
    assert_eq!(tokens, copied, "greedy adopté != semis par copie");
}
