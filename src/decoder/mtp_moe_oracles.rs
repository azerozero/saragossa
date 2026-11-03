//! Oracles ignorés de la tête MTP MoE officielle du 35B-A3B (portes G0-G2).
//!
//! Variables : `SARAGOSSA_RUST_MTP_VERIFIER_TEST_MODEL` (tronc 35B),
//! `SARAGOSSA_RUST_MTP_SIDECAR` (tête officielle) et, pour les mesures
//! d'accord, `SARAGOSSA_RUST_MTP_ORACLE_PROMPTS` (JSON `{id: prompt}`).

use super::mtp::{MtpLoadContext, MtpMoeQuant};
use super::*;
use safetensors::SafeTensors;
use std::path::PathBuf;

/// Prompts de l'accord du draft : ≥ 1024 positions générées (Fable 4).
const AGREEMENT_PROMPTS: [&str; 3] = ["T1", "T2", "T7"];
const AGREEMENT_NEW_TOKENS: usize = 400;

fn model_dir() -> Option<PathBuf> {
    let dir = crate::runtime_flags::env_var_os("SARAGOSSA_RUST_MTP_VERIFIER_TEST_MODEL")
        .map(PathBuf::from)
        .filter(|path| path.is_dir());
    crate::test_support::require_real_model(
        dir,
        "SARAGOSSA_RUST_MTP_VERIFIER_TEST_MODEL pointant vers le 35B-A3B",
    )
}

fn sidecar_path(assets: &crate::ModelAssets) -> Result<PathBuf> {
    assets.mtp.path.clone().ok_or_else(|| {
        InferError::Config("SARAGOSSA_RUST_MTP_SIDECAR requis (tête officielle)".to_string())
    })
}

fn rms(values: &[f32]) -> f32 {
    let sum = values
        .iter()
        .map(|v| f64::from(*v) * f64::from(*v))
        .sum::<f64>();
    (sum / values.len().max(1) as f64).sqrt() as f32
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot = a
        .iter()
        .zip(b)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum::<f64>();
    let na = a.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    let nb = b.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    (dot / (na * nb).max(f64::MIN_POSITIVE)) as f32
}

fn median(mut values: Vec<f32>) -> f32 {
    values.sort_by(f32::total_cmp);
    values.get(values.len() / 2).copied().unwrap_or(f32::NAN)
}

fn dense_of(weight: &LinearWeight) -> Result<Tensor> {
    match weight {
        LinearWeight::Dense(tensor) => Ok(tensor.clone()),
        LinearWeight::AffineQuantized(tensor) => tensor.dequantize(),
    }
}

/// Erreur relative de quantification affine gs64 `‖W − Ŵ‖ / ‖W‖`.
fn quant_error(weight: &Tensor, bits: usize) -> Result<f32> {
    let restored = crate::AffineQuantizedTensor::quantize(weight, 64, bits)?.dequantize()?;
    let diff = weight
        .data()
        .iter()
        .zip(restored.data())
        .map(|(a, b)| a - b)
        .collect::<Vec<_>>();
    Ok(rms(&diff) / rms(weight.data()).max(f32::MIN_POSITIVE))
}

/// G0 (format et numérique) : tenseurs finis, RMS dans un rapport [¼, 4] avec
/// la couche full-attn MoE 39 du tronc (hors `fc`, sans homologue), erreur de
/// quantification bornée (u4 < 0,15, u8 < 0,02 en relatif). Normes rapportées
/// après `1 + weight`.
#[test]
#[ignore = "requiert le 35B-A3B et SARAGOSSA_RUST_MTP_SIDECAR"]
fn mtp_moe_head_weights_are_plausible() -> Result<()> {
    let Some(dir) = model_dir() else {
        return Ok(());
    };
    let assets = crate::ModelAssets::load_local(&dir)?;
    let bytes = std::fs::read(sidecar_path(&assets)?)
        .map_err(|source| InferError::Config(format!("lecture sidecar: {source}")))?;
    let st = SafeTensors::deserialize(&bytes)
        .map_err(|source| InferError::Config(format!("parse sidecar: {source}")))?;
    let tensor = |name: &str| -> Result<Tensor> {
        let view = st
            .tensor(name)
            .map_err(|source| InferError::Config(format!("{name}: {source}")))?;
        Tensor::from_vec(
            view.shape().to_vec(),
            crate::safetensor::bytes_to_dense_f32(view.data(), view.dtype(), name)?,
        )
    };
    let mut names = st.names();
    names.sort();
    for name in &names {
        let t = tensor(name)?;
        assert!(t.data().iter().all(|v| v.is_finite()), "{name} non fini");
        let shown = if name.ends_with("norm.weight") {
            let mean = t.data().iter().sum::<f32>() / t.len() as f32;
            format!(" moyenne après +1 = {:.4}", mean + 1.0)
        } else {
            String::new()
        };
        eprintln!("G0 {name} {:?} rms={:.5}{shown}", t.shape(), rms(t.data()));
    }

    let trunk = crate::load_causal_decoder(&assets)?;
    let layer = trunk
        .layers
        .get(39)
        .ok_or_else(|| InferError::Config("couche 39 absente".to_string()))?;
    let AttentionBlock::Full(attention) = &layer.attention else {
        return Err(InferError::Config("couche 39 non full-attn".to_string()));
    };
    let Some(FeedForward::Moe(moe)) = layer.mlp.as_ref() else {
        return Err(InferError::Config("couche 39 non MoE".to_string()));
    };
    let (router, experts, _) = moe.routed_parts_for_test();
    let (trunk_gate, trunk_up, trunk_down) = experts[0].projections();
    let v_proj = attention
        .v_proj
        .as_ref()
        .ok_or_else(|| InferError::Config("v_proj couche 39 absent".to_string()))?;
    let gate_up = tensor("mtp.layers.0.mlp.experts.gate_up_proj")?;
    let down = tensor("mtp.layers.0.mlp.experts.down_proj")?;
    let half = gate_up.shape()[1] / 2 * gate_up.shape()[2];
    let expert_down = down.shape()[1] * down.shape()[2];
    let pairs: [(&str, Vec<f32>, &LinearWeight); 7] = [
        (
            "q_proj",
            tensor("mtp.layers.0.self_attn.q_proj.weight")?
                .data()
                .to_vec(),
            attention.q_proj.weight(),
        ),
        (
            "k_proj",
            tensor("mtp.layers.0.self_attn.k_proj.weight")?
                .data()
                .to_vec(),
            attention.k_proj.weight(),
        ),
        (
            "v_proj",
            tensor("mtp.layers.0.self_attn.v_proj.weight")?
                .data()
                .to_vec(),
            v_proj.weight(),
        ),
        (
            "o_proj",
            tensor("mtp.layers.0.self_attn.o_proj.weight")?
                .data()
                .to_vec(),
            attention.o_proj.weight(),
        ),
        (
            "router",
            tensor("mtp.layers.0.mlp.gate.weight")?.data().to_vec(),
            router.weight(),
        ),
        (
            "expert0.gate",
            gate_up.data()[..half].to_vec(),
            trunk_gate.weight(),
        ),
        (
            "expert0.up",
            gate_up.data()[half..2 * half].to_vec(),
            trunk_up.weight(),
        ),
    ];
    let mut checked = pairs
        .iter()
        .map(|(label, head, trunk)| Ok((*label, rms(head), rms(dense_of(trunk)?.data()))))
        .collect::<Result<Vec<_>>>()?;
    checked.push((
        "expert0.down",
        rms(&down.data()[..expert_down]),
        rms(dense_of(trunk_down.weight())?.data()),
    ));
    for (label, head, trunk) in checked {
        let ratio = head / trunk;
        eprintln!("G0 rms {label}: tête={head:.5} tronc39={trunk:.5} rapport={ratio:.3}");
        assert!(
            (0.25..=4.0).contains(&ratio),
            "{label}: rapport RMS {ratio}"
        );
    }

    for (name, bits, bound) in [
        ("mtp.fc.weight", 8, 0.02),
        ("mtp.layers.0.self_attn.q_proj.weight", 8, 0.02),
        ("mtp.layers.0.self_attn.o_proj.weight", 8, 0.02),
        ("mtp.layers.0.mlp.shared_expert.down_proj.weight", 4, 0.15),
        ("mtp.layers.0.mlp.gate.weight", 8, 0.02),
        ("mtp.layers.0.mlp.shared_expert_gate.weight", 8, 0.02),
    ] {
        let error = quant_error(&tensor(name)?, bits)?;
        eprintln!("G0 quant u{bits} {name}: erreur relative {error:.4}");
        assert!(error < bound, "{name}: erreur de quantification {error}");
    }
    let expert0 = Tensor::from_vec(
        vec![gate_up.shape()[1], gate_up.shape()[2]],
        gate_up.data()[..2 * half].to_vec(),
    )?;
    let error = quant_error(&expert0, 4)?;
    eprintln!("G0 quant u4 expert0.gate_up: erreur relative {error:.4}");
    assert!(error < 0.15);
    Ok(())
}

/// Prompts du JSON `{id: texte}` désigné par `SARAGOSSA_RUST_MTP_ORACLE_PROMPTS`.
pub(super) fn oracle_prompts(
    assets: &crate::ModelAssets,
    ids: &[&str],
) -> Option<Vec<(String, Vec<usize>)>> {
    let path = crate::runtime_flags::env_var_os("SARAGOSSA_RUST_MTP_ORACLE_PROMPTS")?;
    let text = std::fs::read_to_string(path).ok()?;
    let prompts: std::collections::HashMap<String, String> = serde_json::from_str(&text).ok()?;
    ids.iter()
        .map(|id| {
            let prompt = prompts.get(*id)?;
            let templated = crate::render_qwen_chatml(
                &[crate::ChatTemplateMessage::new("user", prompt)],
                true,
                false,
            );
            let tokens = assets
                .encode_prompt(&templated)
                .ok()?
                .into_iter()
                .map(|id| usize::try_from(id).ok())
                .collect::<Option<Vec<_>>>()?;
            Some(((*id).to_string(), tokens))
        })
        .collect()
}

/// Hidden post-tête de chaque position `p` (entrée : état final normé `p`,
/// token `p + 1`), runtime CPU, historique committed depuis la position 0.
fn head_states(
    model: &CausalDecoder,
    head: &MtpHead,
    states: &Tensor,
    sequence: &[usize],
) -> Result<Vec<Vec<f32>>> {
    let mut cache = LayerKvCache::default();
    (0..sequence.len() - 1)
        .map(|position| {
            let row = Tensor::row(states.row_slice(position)?.to_vec())?;
            let post = model.mtp_head_forward(
                head,
                &row,
                sequence[position + 1],
                position,
                &mut cache,
                ForwardRuntime::cpu(),
            )?;
            Ok(post.data().to_vec())
        })
        .collect()
}

/// Argmax du lm_head partagé (même projection pour toutes les têtes).
fn head_tokens(model: &CausalDecoder, posts: &[Vec<f32>]) -> Result<Vec<usize>> {
    let hidden = model.final_norm.data().len();
    let mut tokens = Vec::with_capacity(posts.len());
    for chunk in posts.chunks(64) {
        let rows = Tensor::from_vec(vec![chunk.len(), hidden], chunk.concat())?;
        let logits = model
            .lm_head
            .forward_with_runtime(&rows, model.forward_runtime())?;
        let (count, vocab) = logits.as_matrix()?;
        for row in 0..count {
            let values = &logits.data()[row * vocab..(row + 1) * vocab];
            let best = values
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map_or(0, |(index, _)| index);
            tokens.push(best);
        }
    }
    Ok(tokens)
}

/// G0 (qualité du draft) et G1 (perte de quantification) : accord top-1 du
/// draft profondeur 1 avec l'AR greedy sur ≥ 1024 positions générées, tête
/// f32 non quantifiée contre tête affine, bascules appariées comptées.
#[test]
#[ignore = "requiert Metal, le 35B-A3B, SARAGOSSA_RUST_MTP_SIDECAR et les prompts"]
fn mtp_moe_draft_agreement_and_quantization_loss() -> Result<()> {
    let Some(dir) = model_dir() else {
        return Ok(());
    };
    let assets = crate::ModelAssets::load_local(&dir)?;
    let Some(prompts) = oracle_prompts(&assets, &AGREEMENT_PROMPTS) else {
        eprintln!("skip : SARAGOSSA_RUST_MTP_ORACLE_PROMPTS absent ou incomplet");
        return Ok(());
    };
    let sidecar = sidecar_path(&assets)?;
    let model = crate::load_causal_decoder(&assets)?.with_metal_runtime()?;
    let cpu = MtpLoadContext::cpu();
    let dense = MtpHead::from_sidecar_with(&sidecar, &model.config, &cpu, MtpMoeQuant::Dense)?;
    let affine = MtpHead::from_sidecar_with(&sidecar, &model.config, &cpu, MtpMoeQuant::Affine)?;
    let options = GenerationOptions {
        stop_token_ids: Vec::new(),
        stop_sequences: Vec::new(),
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        seed: 0,
        token_constraint: None,
    };
    let (mut positions, mut dense_ok, mut affine_ok) = (0usize, 0usize, 0usize);
    let (mut lost, mut gained) = (0usize, 0usize);
    let mut cosines = Vec::new();
    for (id, prompt) in prompts {
        let generated =
            model.generate_greedy_cached_with_options(&prompt, AGREEMENT_NEW_TOKENS, &options)?;
        let sequence = [prompt.as_slice(), generated.as_slice()].concat();
        let raw = model
            .prefill_hidden_for_test(&sequence, Some(0))?
            .ok_or_else(|| InferError::Config("prefill résident indisponible".to_string()))?;
        let states = rms_norm(&raw, &model.final_norm, model.config.rms_eps)?;
        let dense_posts = head_states(&model, &dense, &states, &sequence)?;
        let affine_posts = head_states(&model, &affine, &states, &sequence)?;
        let dense_tokens = head_tokens(&model, &dense_posts)?;
        let affine_tokens = head_tokens(&model, &affine_posts)?;
        // Le draft de la position p propose le token p + 2 : on ne compte que
        // les cibles générées par l'AR.
        let first = prompt.len() - 2;
        let (mut local_dense, mut local_positions) = (0usize, 0usize);
        for p in first..sequence.len() - 2 {
            let target = sequence[p + 2];
            let d = dense_tokens[p] == target;
            let a = affine_tokens[p] == target;
            local_positions += 1;
            local_dense += usize::from(d);
            dense_ok += usize::from(d);
            affine_ok += usize::from(a);
            lost += usize::from(d && !a);
            gained += usize::from(!d && a);
            cosines.push(cosine(&dense_posts[p], &affine_posts[p]));
        }
        positions += local_positions;
        eprintln!(
            "G1 {id}: positions={local_positions} accord_f32={:.4}",
            local_dense as f32 / local_positions as f32
        );
    }
    let dense_rate = dense_ok as f32 / positions as f32;
    let affine_rate = affine_ok as f32 / positions as f32;
    let cos = median(cosines);
    eprintln!(
        "G0/G1 positions={positions} accord_f32={dense_rate:.4} accord_u4={affine_rate:.4} \
         bascules perdues={lost} gagnées={gained} cosinus_médian={cos:.5}"
    );
    assert!(positions >= 1024, "G1 : {positions} positions < 1024");
    assert!(dense_rate >= 0.5, "G0 : accord f32 {dense_rate} < 0,5");
    assert!(
        dense_rate - affine_rate <= 0.02,
        "G1 : perte de quantification {} points",
        100.0 * (dense_rate - affine_rate)
    );
    Ok(())
}

/// Écart des logits du routeur entre le 8e et le 9e expert : sous un seuil,
/// CPU et GPU peuvent router différemment sans bug.
fn router_margin(head: &MtpHead, mlp_input: &Tensor) -> Result<f32> {
    let FeedForward::Moe(moe) = &head.layer.mlp else {
        return Err(InferError::Config("tête MTP non MoE".to_string()));
    };
    let (router, _, top_k) = moe.routed_parts_for_test();
    let mut logits = router
        .forward_with_runtime(mlp_input, ForwardRuntime::cpu())?
        .data()
        .to_vec();
    logits.sort_by(|a, b| b.total_cmp(a));
    Ok(logits[top_k - 1] - logits[top_k])
}

/// G2 (oracle GPU) : bloc MTP MoE résident contre le forward CPU de la même
/// tête affine, position par position sur un texte généré. Cosinus médian
/// ≥ 0,999 ; hors quasi-égalités du routeur (écart 8e/9e < 2e-2, comptées, ≤ 25 %),
/// écart relatif ≤ 2e-2 et argmax identique sauf quasi-égalité de logits.
#[test]
#[ignore = "requiert Metal, le 35B-A3B, SARAGOSSA_RUST_MTP_SIDECAR et les prompts"]
fn mtp_moe_resident_block_matches_cpu_oracle() -> Result<()> {
    const POSITIONS: usize = 160;
    // Routeur u8 : échelles f32 côté CPU, bf16 côté GPU (Astra 9) ; l'écart
    // observé des logits reste sous 1,4e-2 sur les positions divergentes.
    const ROUTER_TIE: f32 = 2e-2;
    let Some(dir) = model_dir() else {
        return Ok(());
    };
    let assets = crate::ModelAssets::load_local(&dir)?;
    let Some(prompts) = oracle_prompts(&assets, &["T1"]) else {
        eprintln!("skip : SARAGOSSA_RUST_MTP_ORACLE_PROMPTS absent ou incomplet");
        return Ok(());
    };
    let sidecar = sidecar_path(&assets)?;
    let model = crate::load_causal_decoder(&assets)?
        .with_metal_runtime()?
        .with_mtp_sidecar(&sidecar)?;
    let head = model
        .mtp
        .as_ref()
        .ok_or_else(|| InferError::Config("tête MTP absente".to_string()))?;
    let options = GenerationOptions {
        stop_token_ids: Vec::new(),
        stop_sequences: Vec::new(),
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        seed: 0,
        token_constraint: None,
    };
    let (_, prompt) = &prompts[0];
    let generated = model.generate_greedy_cached_with_options(prompt, POSITIONS, &options)?;
    let sequence = [prompt.as_slice(), generated.as_slice()].concat();
    let raw = model
        .prefill_hidden_for_test(&sequence, Some(0))?
        .ok_or_else(|| InferError::Config("prefill résident indisponible".to_string()))?;
    let states = rms_norm(&raw, &model.final_norm, model.config.rms_eps)?;
    let hidden = model.final_norm.data().len();

    let (mut cache, _) = model.prefill_cache_state(&sequence[..1])?;
    assert!(model.setup_resident_full_decode(&mut cache, sequence.len() + 4, 0, false)?);
    let mut cpu_cache = LayerKvCache::default();
    let (mut cosines, mut ties, mut flips, mut max_rel) = (Vec::new(), 0usize, 0usize, 0f32);
    let first = sequence.len() - 1 - POSITIONS.min(sequence.len() - 1);
    for position in 0..sequence.len() - 1 {
        let row = Tensor::row(states.row_slice(position)?.to_vec())?;
        let (cpu_post, mlp_input) = model.mtp_head_forward_parts(
            head,
            &row,
            sequence[position + 1],
            position,
            &mut cpu_cache,
            ForwardRuntime::cpu(),
        )?;
        assert!(model.start_mtp_resident_draft(&mut cache, &row, Some(position))?);
        let gpu_token =
            model.next_mtp_draft_resident(&mut cache, sequence[position + 1], position)?;
        let arena = cache
            .resident
            .as_ref()
            .and_then(|arena| arena.mtp.as_ref())
            .ok_or_else(|| InferError::Config("arène MTP absente".to_string()))?;
        // Le draft écrit `hidden_b` puis bascule `current_is_a`.
        let gpu_post = crate::metal_backend::read_f32_buffer(arena.hidden_b.buffer(), hidden)?;
        if position < first {
            continue;
        }
        let cos = cosine(cpu_post.data(), &gpu_post);
        cosines.push(cos);
        let margin = router_margin(head, &mlp_input)?;
        let tie = margin < ROUTER_TIE;
        ties += usize::from(tie);
        let diff = cpu_post
            .data()
            .iter()
            .zip(&gpu_post)
            .map(|(a, b)| a - b)
            .collect::<Vec<_>>();
        let rel = rms(&diff) / rms(cpu_post.data()).max(f32::MIN_POSITIVE);
        if !tie {
            max_rel = max_rel.max(rel);
        }
        let logits = model
            .lm_head
            .forward_with_runtime(&cpu_post, model.forward_runtime())?;
        let mut sorted = logits.data().to_vec();
        let cpu_token = logits
            .data()
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map_or(0, |(index, _)| index);
        sorted.sort_by(|a, b| b.total_cmp(a));
        let logit_tie = sorted[0] - sorted[1] < 1e-2;
        if cpu_token != gpu_token && !logit_tie && !tie {
            flips += 1;
        }
        if rel > 2e-2 {
            eprintln!(
                "G2 position {position}: cos={cos:.6} rel={rel:.4e} marge_routeur={margin:.3e}"
            );
        }
    }
    let count = cosines.len();
    let cos = median(cosines);
    eprintln!(
        "G2 positions={count} cosinus_médian={cos:.6} max_rel_hors_égalités={max_rel:.4e} \
         égalités_routeur={ties} bascules_argmax={flips}"
    );
    assert!(cos >= 0.999, "G2 : cosinus médian {cos}");
    assert!(max_rel <= 2e-2, "G2 : écart relatif {max_rel}");
    assert!(
        ties * 4 <= count,
        "G2 : {ties} quasi-égalités du routeur sur {count}"
    );
    assert_eq!(
        flips, 0,
        "G2 : argmax GPU/CPU différents hors quasi-égalité"
    );
    Ok(())
}
