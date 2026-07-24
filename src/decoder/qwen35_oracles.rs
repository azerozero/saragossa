//! Oracles ignorés Qwen3.6-35B : précision couche 0 et CPU↔GPU réel.

use super::*;
use crate::MoeMlp;
use serde::Deserialize;
use std::path::PathBuf;

const SKY_FLIP_INPUT_INDEX: usize = 18;
const COOP_MAX_ABS_TOLERANCE: f32 = 5.0e-3;
const COOP_MEAN_ABS_TOLERANCE: f32 = 5.0e-4;

#[derive(Deserialize)]
struct OmlxPrecisionCapture {
    prompt_ids: Vec<usize>,
    targets: Vec<usize>,
    predict_index: usize,
    input_token: usize,
    target_token: usize,
    expert_indices: Vec<usize>,
    captures: OmlxBoundaries,
}

#[derive(Deserialize)]
struct OmlxBoundaries {
    embed: Vec<f32>,
    input_norm: Vec<f32>,
    q_proj: Vec<f32>,
    k_proj: Vec<f32>,
    v_proj: Vec<f32>,
    attention: Vec<f32>,
    o_proj: Vec<f32>,
    post_norm: Vec<f32>,
    mlp_gate: Vec<f32>,
    mlp_up: Vec<f32>,
    mlp_down: Vec<f32>,
}

#[derive(Deserialize)]
struct OmlxTeacherCapture {
    token_count: usize,
    cases: Vec<OmlxTeacherCase>,
}

#[derive(Deserialize)]
struct OmlxTeacherCase {
    prompt: String,
    prompt_ids: Vec<usize>,
    targets: Vec<usize>,
    positions: Vec<OmlxTeacherPosition>,
}

#[derive(Deserialize)]
struct OmlxTeacherPosition {
    target: usize,
    top3: Vec<usize>,
    margin: f32,
}

struct TeacherCase {
    name: &'static str,
    prompt: Vec<usize>,
    forced: Vec<usize>,
    cpu_predictions: Vec<usize>,
}

fn required_path(name: &str) -> Result<PathBuf> {
    std::env::var(name)
        .map(PathBuf::from)
        .map_err(|error| InferError::Config(format!("{name} absent: {error}")))
}

fn read_json<T: serde::de::DeserializeOwned>(path: &std::path::Path) -> Result<T> {
    serde_json::from_slice(&std::fs::read(path).map_err(|source| InferError::Io {
        path: path.to_path_buf(),
        source,
    })?)
    .map_err(|source| InferError::Json {
        path: path.to_path_buf(),
        source,
    })
}

fn bf16_round(value: f32) -> f32 {
    let bits = value.to_bits();
    let rounding = 0x7fff_u32 + ((bits >> 16) & 1);
    f32::from_bits(bits.wrapping_add(rounding) & 0xffff_0000)
}

fn diff_stats(actual: &[f32], expected: &[f32]) -> Result<(f32, f32)> {
    if actual.len() != expected.len() {
        return Err(InferError::Dimension(format!(
            "capture actual={} référence={}",
            actual.len(),
            expected.len()
        )));
    }
    let mut maximum = 0.0_f32;
    let mut sum = 0.0_f64;
    for (left, right) in actual.iter().zip(expected) {
        let delta = (left - right).abs();
        maximum = maximum.max(delta);
        sum += f64::from(delta);
    }
    Ok((maximum, (sum / actual.len().max(1) as f64) as f32))
}

fn report_boundary(name: &str, actual: &[f32], expected: &[f32]) -> Result<()> {
    let recast = actual.iter().copied().map(bf16_round).collect::<Vec<_>>();
    let (strict_max, strict_mean) = diff_stats(actual, expected)?;
    let (bf16_max, bf16_mean) = diff_stats(&recast, expected)?;
    let direction = if bf16_mean < strict_mean {
        "rapproche"
    } else if bf16_mean > strict_mean {
        "éloigne"
    } else {
        "neutre"
    };
    println!(
        "precision_boundary name={name} strict_max={strict_max:.9e} strict_mean={strict_mean:.9e} bf16_max={bf16_max:.9e} bf16_mean={bf16_mean:.9e} recast={direction}"
    );
    Ok(())
}

fn argmax(values: &[f32]) -> Result<usize> {
    values
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index)
        .ok_or_else(|| InferError::Dimension("argmax sur logits vides".to_string()))
}

fn top_two(values: &[f32]) -> Result<[(usize, f32); 2]> {
    let mut first = None;
    let mut second = None;
    for (index, value) in values.iter().copied().enumerate() {
        if first.is_none_or(|(_, current)| value.total_cmp(&current).is_gt()) {
            second = first;
            first = Some((index, value));
        } else if second.is_none_or(|(_, current)| value.total_cmp(&current).is_gt()) {
            second = Some((index, value));
        }
    }
    match (first, second) {
        (Some(first), Some(second)) => Ok([first, second]),
        _ => Err(InferError::Dimension(
            "top-2 sur moins de deux logits".to_string(),
        )),
    }
}

fn layer0_mlp_input(model: &CausalDecoder, tokens: &[usize]) -> Result<Tensor> {
    let layer = model
        .layers
        .first()
        .ok_or_else(|| InferError::Config("Qwen35 sans couche 0".to_string()))?;
    let hidden = model.embed_scaled(tokens)?;
    let normed = rms_norm(&hidden, &layer.input_norm, model.config.rms_eps)?;
    let attention = match &layer.attention {
        AttentionBlock::Linear(linear) => linear.forward_with_runtime(
            model.config.linear_attention_config()?,
            &normed,
            ForwardRuntime::cpu(),
        )?,
        AttentionBlock::Full(_) => {
            return Err(InferError::Config(
                "Qwen35 couche 0 attendue linear-attn".to_string(),
            ));
        }
    };
    let after_attention = hidden.add(&attention)?;
    let post_norm = layer
        .post_attention_norm
        .as_ref()
        .ok_or_else(|| InferError::Config("post RMSNorm couche 0 absent".to_string()))?;
    rms_norm(&after_attention, post_norm, model.config.rms_eps)
}

fn layer0_moe(model: &CausalDecoder) -> Result<&MoeMlp> {
    let mlp = model
        .layers
        .first()
        .and_then(|layer| layer.mlp.as_ref())
        .ok_or_else(|| InferError::Config("MLP couche 0 absent".to_string()))?;
    match mlp {
        FeedForward::Moe(moe) => Ok(moe),
        FeedForward::Dense(_) => Err(InferError::Config(
            "Qwen35 couche 0 attendue MoE".to_string(),
        )),
    }
}

/// Localise la première frontière couche 0 face aux captures oMLX bf16.
#[test]
#[ignore = "charge Qwen35 réel; définir QWEN35_MODEL et QWEN35_OMLX_CAPTURE"]
fn qwen35_layer0_precision_boundary_vs_omlx() -> Result<()> {
    let model_dir = required_path("QWEN35_MODEL")?;
    let reference_path = required_path("QWEN35_OMLX_CAPTURE")?;
    let reference: OmlxPrecisionCapture = read_json(&reference_path)?;
    if reference.targets.len() <= reference.predict_index
        || reference.targets.len() <= SKY_FLIP_INPUT_INDEX
        || reference.predict_index != SKY_FLIP_INPUT_INDEX + 1
        || reference.input_token != reference.targets[SKY_FLIP_INPUT_INDEX]
        || reference.target_token != 11
        || reference.targets[reference.predict_index] != reference.target_token
    {
        return Err(InferError::Config(
            "capture oMLX ciel/19 incohérente".to_string(),
        ));
    }

    let assets = crate::ModelAssets::load_local(&model_dir)?;
    let model = crate::load_causal_decoder(&assets)?;
    let mut state = model.prefill_prompt_state_uncached(&reference.prompt_ids)?;
    model.extend_prompt_state(&mut state, &reference.targets[..SKY_FLIP_INPUT_INDEX])?;
    let input = model.embed_scaled(&[reference.input_token])?;
    let layer = model
        .layers
        .first()
        .ok_or_else(|| InferError::Config("Qwen35 sans couche 0".to_string()))?;
    let input_norm = rms_norm(&input, &layer.input_norm, model.config.rms_eps)?;
    let mut linear_cache = state
        .cache
        .layers
        .first()
        .ok_or_else(|| InferError::Config("cache couche 0 absent".to_string()))?
        .linear
        .clone();
    let linear = match &layer.attention {
        AttentionBlock::Linear(linear) => linear,
        AttentionBlock::Full(_) => {
            return Err(InferError::Config(
                "Qwen35 couche 0 attendue linear-attn".to_string(),
            ));
        }
    };
    let linear_capture = linear.precision_capture_cached(
        model.config.linear_attention_config()?,
        &input_norm,
        &mut linear_cache,
    )?;
    let after_attention = input.add(&linear_capture.o_proj)?;
    let post_norm = layer
        .post_attention_norm
        .as_ref()
        .ok_or_else(|| InferError::Config("post RMSNorm couche 0 absent".to_string()))?;
    let post_normed = rms_norm(&after_attention, post_norm, model.config.rms_eps)?;
    let mlp_capture = layer0_moe(&model)?.precision_capture(&post_normed)?;
    if mlp_capture.expert_indices != reference.expert_indices {
        return Err(InferError::Config(format!(
            "experts couche 0 reti={:?} oMLX={:?}",
            mlp_capture.expert_indices, reference.expert_indices
        )));
    }

    for (name, actual, expected) in [
        ("embed", input.data(), reference.captures.embed.as_slice()),
        (
            "rmsnorm_input",
            input_norm.data(),
            reference.captures.input_norm.as_slice(),
        ),
        (
            "q_proj",
            linear_capture.q_proj.data(),
            reference.captures.q_proj.as_slice(),
        ),
        (
            "k_proj",
            linear_capture.k_proj.data(),
            reference.captures.k_proj.as_slice(),
        ),
        (
            "v_proj",
            linear_capture.v_proj.data(),
            reference.captures.v_proj.as_slice(),
        ),
        (
            "attention",
            linear_capture.attention.data(),
            reference.captures.attention.as_slice(),
        ),
        (
            "o_proj",
            linear_capture.o_proj.data(),
            reference.captures.o_proj.as_slice(),
        ),
        (
            "rmsnorm_post",
            post_normed.data(),
            reference.captures.post_norm.as_slice(),
        ),
        (
            "mlp_gate",
            mlp_capture.gate.data(),
            reference.captures.mlp_gate.as_slice(),
        ),
        (
            "mlp_up",
            mlp_capture.up.data(),
            reference.captures.mlp_up.as_slice(),
        ),
        (
            "mlp_down",
            mlp_capture.down.data(),
            reference.captures.mlp_down.as_slice(),
        ),
    ] {
        report_boundary(name, actual, expected)?;
    }
    Ok(())
}

/// Mesure l'accord teacher-forced du chemin résident face aux 153 tokens oMLX.
#[test]
#[ignore = "charge Qwen35 Metal réel; définir QWEN35_MODEL et QWEN35_OMLX_TEACHER"]
fn qwen35_teacher_forced_vs_omlx() -> Result<()> {
    const EXPECTED_LENGTHS: [usize; 5] = [11, 48, 42, 4, 48];

    let model_dir = required_path("QWEN35_MODEL")?;
    let reference_path = required_path("QWEN35_OMLX_TEACHER")?;
    let reference: OmlxTeacherCapture = read_json(&reference_path)?;
    let lengths = reference
        .cases
        .iter()
        .map(|case| case.targets.len())
        .collect::<Vec<_>>();
    if reference.token_count != 153 || lengths != EXPECTED_LENGTHS {
        return Err(InferError::Config(format!(
            "corpus teacher oMLX incohérent: total={} longueurs={lengths:?}",
            reference.token_count
        )));
    }

    let assets = crate::ModelAssets::load_local(&model_dir)?;
    let model = crate::load_causal_decoder(&assets)?.with_metal_runtime()?;
    let mut agreements = 0_usize;
    let mut visited = 0_usize;
    for (case_index, case) in reference.cases.iter().enumerate() {
        if case.positions.len() != case.targets.len()
            || case
                .positions
                .iter()
                .zip(&case.targets)
                .any(|(position, target)| position.target != *target)
        {
            return Err(InferError::Config(format!(
                "cas teacher oMLX {} incohérent",
                case_index + 1
            )));
        }
        let mut state = model.prefill_prompt_state_uncached(&case.prompt_ids)?;
        model.setup_resident_decode(&mut state.cache, case.targets.len() + 1, false)?;
        let mut case_agreements = 0_usize;
        for (position_index, (target, omlx)) in case.targets.iter().zip(&case.positions).enumerate()
        {
            let logits = model.logits_from_final_state(&state.final_state)?;
            let [reti_first, reti_second] = top_two(logits.as_row()?)?;
            visited += 1;
            if reti_first.0 == *target {
                agreements += 1;
                case_agreements += 1;
            } else {
                println!(
                    "qwen35_teacher_flip case={} position={} target={} reti={} reti_second={} reti_margin={:.9} omlx_top3={:?} omlx_margin={:.9}",
                    case_index + 1,
                    position_index,
                    target,
                    reti_first.0,
                    reti_second.0,
                    reti_first.1 - reti_second.1,
                    omlx.top3,
                    omlx.margin
                );
            }
            model.extend_prompt_state(&mut state, &[*target])?;
        }
        println!(
            "qwen35_teacher_case case={} prompt={:?} agreement={}/{}",
            case_index + 1,
            case.prompt,
            case_agreements,
            case.targets.len()
        );
    }
    println!(
        "qwen35_teacher_summary embed_bf16={} agreement={agreements}/{visited} percent={:.6}",
        qwen_embed_bf16_enabled(),
        agreements as f64 * 100.0 / visited.max(1) as f64
    );
    Ok(())
}

fn teacher_cases(assets: &crate::ModelAssets) -> Result<Vec<TeacherCase>> {
    let specifications = [
        (
            "ciel",
            "Résume pourquoi le ciel paraît bleu en une phrase.",
            " La lumière bleue est davantage diffusée par l'atmosphère.",
        ),
        (
            "rust",
            "Donne une règle simple pour écrire du Rust fiable.",
            " Propager les erreurs explicitement évite les arrêts brutaux.",
        ),
    ];
    let mut cases = Vec::new();
    for (name, prompt, continuation) in specifications {
        let rendered = crate::render_qwen_chatml(
            &[crate::ChatTemplateMessage::new("user", prompt)],
            true,
            false,
        );
        let prompt = assets
            .encode_prompt_with_special(&rendered)?
            .into_iter()
            .map(|value| value as usize)
            .collect::<Vec<_>>();
        let forced = assets
            .encode_prompt(continuation)?
            .into_iter()
            .map(|value| value as usize)
            .take(8)
            .collect::<Vec<_>>();
        if forced.len() < 4 {
            return Err(InferError::Config(format!(
                "corpus {name}: continuation trop courte"
            )));
        }
        cases.push(TeacherCase {
            name,
            prompt,
            forced,
            cpu_predictions: Vec::new(),
        });
    }
    Ok(cases)
}

fn fill_cpu_predictions(model: &CausalDecoder, cases: &mut [TeacherCase]) -> Result<()> {
    for case in cases {
        let (mut cache, _) = model.prefill_cache_state_tokenwise(&case.prompt)?;
        for token in &case.forced {
            let logits = model.next_logits_cached(&mut cache, *token)?;
            case.cpu_predictions.push(argmax(logits.as_row()?)?);
        }
    }
    Ok(())
}

fn assert_predictions(path: &str, case: &TeacherCase, actual: &[usize]) -> Result<()> {
    if actual != case.cpu_predictions {
        let first = actual
            .iter()
            .zip(&case.cpu_predictions)
            .position(|(gpu, cpu)| gpu != cpu)
            .unwrap_or_else(|| actual.len().min(case.cpu_predictions.len()));
        return Err(InferError::Config(format!(
            "oracle {path}/{} rouge à {first}: gpu={:?} cpu={:?}",
            case.name,
            actual.get(first),
            case.cpu_predictions.get(first)
        )));
    }
    println!(
        "qwen35_oracle path={path} case={} tokens={} status=vert",
        case.name,
        actual.len()
    );
    Ok(())
}

/// Compare les quatre chemins Qwen35 Metal à une référence CPU teacher-forced.
#[test]
#[ignore = "charge Qwen35 CPU+Metal réel; définir QWEN35_MODEL"]
fn qwen35_cpu_gpu_teacher_forced_oracle_matrix() -> Result<()> {
    let model_dir = required_path("QWEN35_MODEL")?;
    let assets = crate::ModelAssets::load_local(&model_dir)?;
    let mut cases = teacher_cases(&assets)?;

    let cpu = crate::load_causal_decoder(&assets)?;
    fill_cpu_predictions(&cpu, &mut cases)?;
    let coop_input = layer0_mlp_input(&cpu, &cases[0].prompt)?;
    let coop_cpu = layer0_moe(&cpu)?.routed_only_for_test(&coop_input)?;
    drop(cpu);

    let gpu = crate::load_causal_decoder(&assets)?.with_metal_runtime()?;
    let runtime = gpu
        .forward_runtime()
        .metal_executor()
        .ok_or_else(|| InferError::Metal("executor Metal Qwen35 absent".to_string()))?;

    for case in &cases {
        let (mut cache, _) = gpu.prefill_cache_state_tokenwise(&case.prompt)?;
        let mut predictions = Vec::new();
        for token in &case.forced {
            let logits = gpu.next_logits_cached(&mut cache, *token)?;
            predictions.push(argmax(logits.as_row()?)?);
        }
        assert_predictions("per_op_routed", case, &predictions)?;
    }

    let (router, experts, top_k) = layer0_moe(&gpu)?.routed_parts_for_test();
    let coop_gpu =
        runtime.moe_routed_rows_coop_real_for_test(&coop_input, router, experts, top_k)?;
    let (coop_max, coop_mean) = diff_stats(coop_gpu.data(), coop_cpu.data())?;
    println!(
        "qwen35_oracle path=moe_routed_rows_coop rows={} max_abs={coop_max:.9e} mean_abs={coop_mean:.9e}",
        coop_input.as_matrix()?.0
    );
    if coop_max > COOP_MAX_ABS_TOLERANCE || coop_mean > COOP_MEAN_ABS_TOLERANCE {
        return Err(InferError::Config(format!(
            "oracle moe_routed_rows_coop rouge: max_abs={coop_max:e} (tol={COOP_MAX_ABS_TOLERANCE:e}) mean_abs={coop_mean:e} (tol={COOP_MEAN_ABS_TOLERANCE:e})"
        )));
    }

    for case in &cases {
        let (mut cache, _) = gpu.prefill_cache_state_tokenwise(&case.prompt)?;
        if !gpu.setup_resident_full_decode(&mut cache, case.forced.len() + 1, 0, false)? {
            return Err(InferError::Config(
                "setup résident-full Qwen35 refusé".to_string(),
            ));
        }
        let output = gpu
            .next_final_states_resident_verify(&mut cache, &case.forced, true, false, None)?
            .ok_or_else(|| InferError::Config("verify résident-full non exercé".to_string()))?;
        let predictions = output
            .tokens
            .ok_or_else(|| InferError::Config("argmax résident-full absent".to_string()))?;
        assert_predictions("resident_full", case, &predictions)?;
    }

    let [first, second] = cases.as_slice() else {
        return Err(InferError::Config(
            "oracle duo attend exactement deux cas".to_string(),
        ));
    };
    let (mut cache_a, _) = gpu.prefill_cache_state_tokenwise(&first.prompt)?;
    let (mut cache_b, _) = gpu.prefill_cache_state_tokenwise(&second.prompt)?;
    if !gpu.setup_resident_full_decode_with_slot(
        &mut cache_a,
        first.forced.len() + 1,
        0,
        0,
        false,
    )? || !gpu.setup_resident_full_decode_with_slot(
        &mut cache_b,
        second.forced.len() + 1,
        0,
        1,
        false,
    )? {
        return Err(InferError::Config("setup duo Qwen35 refusé".to_string()));
    }
    if !gpu.supports_resident_duo(&cache_a) || !gpu.supports_resident_duo(&cache_b) {
        return Err(InferError::Config(
            "chemin duo Qwen35 non exercé".to_string(),
        ));
    }
    let count = first.forced.len().min(second.forced.len());
    let mut duo_a = Vec::with_capacity(count);
    let mut duo_b = Vec::with_capacity(count);
    for index in 0..count {
        let [next_a, next_b] = gpu.decode_tokens_resident_duo(
            &mut cache_a,
            &mut cache_b,
            [first.forced[index], second.forced[index]],
            None,
        )?;
        duo_a.push(next_a);
        duo_b.push(next_b);
    }
    assert_predictions("duo_lightbatch", first, &duo_a)?;
    assert_predictions("duo_lightbatch", second, &duo_b)?;
    Ok(())
}
