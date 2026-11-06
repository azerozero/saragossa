//! Oracles CPU des briques résidentes Flash-Next, aux dimensions réelles.

use metal::{Buffer, MTLResourceOptions};

use super::*;
use crate::decode_resident::utils::bf16_round_f32;
use crate::linear_attention::tests::DeterministicF32;
use crate::metal_backend::{
    commit_and_wait, read_f32_buffer, MetalExecutor, MetalLinearWeightBuffers,
};
use crate::qwen4_exp::math::relative_error;
use crate::qwen4_exp::residual::{combine, GatedResidual};
use crate::{AffineQuantizedTensor, InferError, LinearWeight, Result, Tensor};

/// Longueur d'un oracle de brique : `default`, ou `SARAGOSSA_FLASH_NEXT_STEPS`
/// pour la bisection longue (jusqu'à 2051 positions).
fn oracle_steps(default: usize) -> usize {
    crate::runtime_flags::env_var("SARAGOSSA_FLASH_NEXT_STEPS")
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn executor() -> Result<Option<MetalExecutor>> {
    match MetalExecutor::new() {
        Ok(executor) => Ok(Some(executor)),
        Err(InferError::Metal(message)) if message.contains("aucun device") => Ok(None),
        Err(error) => Err(error),
    }
}

fn buffer(executor: &MetalExecutor, data: &[f32]) -> Buffer {
    executor.device().new_buffer_with_data(
        data.as_ptr().cast(),
        std::mem::size_of_val(data) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn zeros(executor: &MetalExecutor, len: usize) -> Buffer {
    buffer(executor, &vec![0.0; len])
}

/// Poids u8 gs64 (format q4e8 des projections hc) et leur déquantification,
/// seule vérité commune au CPU et au GPU.
fn u8_weight(
    executor: &MetalExecutor,
    rng: &mut DeterministicF32,
    rows: usize,
    cols: usize,
    scale: f32,
) -> Result<(MetalLinearWeightBuffers, Tensor)> {
    let dense = Tensor::from_vec(vec![rows, cols], rng.fill(rows * cols, -scale, scale))?;
    let raw = AffineQuantizedTensor::quantize(&dense, 64, 8)?;
    // Échelles et biais bf16, comme dans le checkpoint : le GPU les lit en bf16.
    let bf16 = |v: &[f32]| -> Vec<f32> { v.iter().map(|x| bf16_round_f32(*x)).collect() };
    let quantized = AffineQuantizedTensor::new(
        raw.packed_shape(),
        raw.packed_data().to_vec(),
        Tensor::from_vec(raw.scales_shape().to_vec(), bf16(&raw.scales_f32()))?,
        Tensor::from_vec(raw.biases_shape().to_vec(), bf16(&raw.biases_f32()))?,
        64,
        8,
    )?;
    let reference = quantized.dequantize()?;
    let gpu = executor.resolve_linear_weight_buffers(
        &LinearWeight::AffineQuantized(quantized),
        "flash_next_hc_test",
    )?;
    Ok((gpu, reference))
}

/// Hyper-connection aux dimensions réelles (4 flux × 2560, rang 320) : entrée
/// de branche, injection et combinaison contre `GatedResidual` CPU ; mixer
/// final (sans injection) inclus.
#[test]
fn flash_next_hc_mix_and_combine_match_cpu_at_real_dims() -> Result<()> {
    let Some(executor) = executor()? else {
        return Ok(());
    };
    let kernels = FlashNextKernels::new(executor.device())?;
    let dims = HcDims {
        hidden: 2560,
        hc: 4,
        lowrank: 320,
        eps: 1.0e-6,
    };
    let width = dims.width();
    let mut rng = DeterministicF32::new(0x4ec0_0004);
    let norm = rng.fill(width, -0.3, 0.3);
    let (down_gpu, down) = u8_weight(&executor, &mut rng, dims.lowrank, width, 0.05)?;
    let (up_gpu, up) = u8_weight(&executor, &mut rng, width, dims.lowrank, 0.2)?;
    let inject_w = Tensor::from_vec(vec![dims.hc, width], rng.fill(dims.hc * width, -0.02, 0.02))?;
    let inject_gpu = executor
        .resolve_linear_weight_buffers(&LinearWeight::Dense(inject_w.clone()), "flash_next_inj")?;
    let h_data = rng.fill(width, -2.0, 2.0);
    let f_data = rng.fill(dims.hidden, -1.0, 1.0);
    let h = Tensor::from_vec(vec![1, width], h_data.clone())?;
    let f = Tensor::from_vec(vec![1, dims.hidden], f_data.clone())?;

    let scratch = HcScratch {
        normed: zeros(&executor, width),
        low: zeros(&executor, dims.lowrank),
        up: zeros(&executor, width),
        inject: zeros(&executor, dims.hc),
    };
    for with_inject in [true, false] {
        let cpu = GatedResidual {
            norm: norm.clone(),
            down: down.clone(),
            up: up.clone(),
            inject: with_inject.then(|| inject_w.clone()),
        };
        let mixed = cpu.mix(&h, dims.hc, dims.hidden, dims.eps)?;
        let gpu_weights = HcWeights {
            norm: buffer(&executor, &norm),
            down: down_gpu.clone(),
            up: up_gpu.clone(),
            inject: with_inject.then(|| inject_gpu.clone()),
        };
        let h_buf = buffer(&executor, &h_data);
        let f_buf = buffer(&executor, &f_data);
        let x_buf = zeros(&executor, dims.hidden);
        let out_buf = zeros(&executor, width);
        let queue = executor.device().new_command_queue();
        let command_buffer = queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encode_hc_mix(
            &kernels,
            &executor,
            encoder,
            &gpu_weights,
            &scratch,
            dims,
            &h_buf,
            &x_buf,
        )?;
        if with_inject {
            encode_hc_combine(&kernels, encoder, &scratch, dims, &h_buf, &f_buf, &out_buf);
        }
        encoder.end_encoding();
        commit_and_wait(command_buffer)?;

        let x = read_f32_buffer(&x_buf, dims.hidden)?;
        let (err, max) = relative_error(&x, mixed.input.data());
        assert!(
            err <= 1.0e-5,
            "entrée de branche (injection {with_inject}) : {err} (max {max})"
        );
        if let Some(inject) = mixed.inject {
            let expected = combine(&h, &f, &inject, dims.hc)?;
            let out = read_f32_buffer(&out_buf, width)?;
            let (err, max) = relative_error(&out, expected.data());
            assert!(err <= 1.0e-6, "combinaison : {err} (max {max})");
        }
    }
    Ok(())
}

/// PLE résident pas à pas (4 flux × 2560, table de 2560 octets, conv k4
/// dilatée 3) contre `ple` CPU sans cache sur 14 tokens : l'anneau de 9
/// colonnes fait le tour. Un reset rejoue ensuite les premiers pas à l'identique.
#[test]
fn flash_next_ple_steps_match_cpu_sequence_and_reset() -> Result<()> {
    use crate::qwen4_exp::ngram::NgramContext;
    use crate::qwen4_exp::ple::{ple, PleWeights as CpuPle};
    use crate::safetensor::f8_e4m3_to_f32;
    let Some(executor) = executor()? else {
        return Ok(());
    };
    let kernels = FlashNextKernels::new(executor.device())?;
    let dims = HcDims {
        hidden: 2560,
        hc: 4,
        lowrank: 320,
        eps: 1.0e-6,
    };
    let (width, ple_dim, tokens, scale) = (dims.width(), 2560, 14, 3.0e-3_f32);
    let mut rng = DeterministicF32::new(0x0091_e000);
    let (key_gpu, key) = u8_weight(&executor, &mut rng, width, ple_dim, 0.05)?;
    let (value_gpu, value) = u8_weight(&executor, &mut rng, dims.hidden, ple_dim, 0.05)?;
    let norm_key = rng.fill(width, -0.2, 0.2);
    let norm_query = rng.fill(width, -0.2, 0.2);
    let norm_conv = rng.fill(width, -0.2, 0.2);
    let conv = rng.fill(width * 4, -0.5, 0.5);
    let lut: Vec<f32> = (0..=255_u8)
        .map(|b| f8_e4m3_to_f32(b).map_or(f32::NAN, |v| v * scale))
        .collect();
    // Octets sans NaN (0x7F, 0xFF), dont des zéros signés.
    let bytes: Vec<u8> = (0..tokens * ple_dim)
        .map(|i| {
            let b = (rng.next_in(0.0, 254.0) as u8).wrapping_add(i as u8 % 3);
            if b & 0x7f == 0x7f {
                0x80
            } else {
                b
            }
        })
        .collect();
    let emb: Vec<f32> = bytes.iter().map(|&b| lut[usize::from(b)]).collect();
    let h_all = rng.fill(tokens * width, -2.0, 2.0);
    let cpu_w = CpuPle {
        key_proj: key,
        value_proj: value,
        norm_key: norm_key.clone(),
        norm_query: norm_query.clone(),
        norm_conv: norm_conv.clone(),
        conv: conv.clone(),
        kernel: 4,
        dilation: 3,
    };
    let expected = ple(
        &Tensor::from_vec(vec![tokens, width], h_all.clone())?,
        &Tensor::from_vec(vec![tokens, ple_dim], emb)?,
        &cpu_w,
        dims.hc,
        dims.hidden,
        dims.eps,
    )?;

    let weights = PleWeights {
        key_proj: key_gpu,
        value_proj: value_gpu,
        norm_key: buffer(&executor, &norm_key),
        norm_query: buffer(&executor, &norm_query),
        norm_conv: buffer(&executor, &norm_conv),
        conv: buffer(&executor, &conv),
        lut: buffer(&executor, &lut),
        kernel: 4,
        dilation: 3,
        ple_dim,
    };
    let scratch = PleScratch {
        emb: zeros(&executor, ple_dim),
        key: zeros(&executor, width),
        key_normed: zeros(&executor, width),
        value: zeros(&executor, dims.hidden),
        query: zeros(&executor, width),
        gated: zeros(&executor, width),
        normed: zeros(&executor, width),
    };
    let mut state = PleState {
        bytes: executor
            .device()
            .new_buffer(ple_dim as u64, MTLResourceOptions::StorageModeShared),
        ring: zeros(&executor, 9 * width),
        context: NgramContext::new(3, 0),
        step: 0,
    };
    let queue = executor.device().new_command_queue();
    let run = |state: &mut PleState, t: usize| -> Result<Vec<f32>> {
        state.stage_bytes(&bytes[t * ple_dim..(t + 1) * ple_dim])?;
        let h = buffer(&executor, &h_all[t * width..(t + 1) * width]);
        let out = zeros(&executor, width);
        let command_buffer = queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encode_ple(
            &kernels, &executor, encoder, &weights, state, &scratch, dims, &h, &out,
        )?;
        encoder.end_encoding();
        commit_and_wait(command_buffer)?;
        read_f32_buffer(&out, width)
    };
    let mut first = Vec::new();
    for t in 0..tokens {
        let got = run(&mut state, t)?;
        let want = &expected.data()[t * width..(t + 1) * width];
        let reference: Vec<f32> = want
            .iter()
            .zip(&h_all[t * width..(t + 1) * width])
            .map(|(p, h)| h + p)
            .collect();
        let (err, max) = relative_error(&got, &reference);
        assert!(err <= 1.0e-5, "PLE pas {t} : {err} (max {max})");
        if t < 3 {
            first.push(got);
        }
    }
    state.reset(3, 0);
    for (t, before) in first.iter().enumerate() {
        assert_eq!(&run(&mut state, t)?, before, "pas {t} après reset");
    }
    Ok(())
}

/// Attention QSA résidente en régime dense (24 têtes, 2 KV, d256, RoPE 64,
/// θ = 1e7, `q_proj` entrelacé) pas à pas contre `qsa` CPU sans cache sur
/// 40 tokens ; puis refus explicite d'une position ≥ 2051.
#[test]
fn flash_next_qsa_dense_steps_match_cpu_sequence() -> Result<()> {
    use crate::decode_resident::DecodeResidentState;
    use crate::qwen4_exp::attention::{qsa, QsaDims, QsaWeights as CpuQsa};
    let Some(executor) = executor()? else {
        return Ok(());
    };
    let arena = DecodeResidentState::new(executor.device().clone())?;
    let dims = QsaDecodeDims {
        hidden: 2560,
        heads: 24,
        kv_heads: 2,
        head_dim: 256,
        rotary_dim: 64,
        theta: 1.0e7,
        eps: 1.0e-6,
        dense_max: dense_max_visible(2048, 4),
    };
    let (tokens, q_dim, kv_dim) = (oracle_steps(40), 24 * 256, 2 * 256);
    let mut rng = DeterministicF32::new(0x05a0_0024);
    let (q_gpu, q_w) = u8_weight(&executor, &mut rng, 2 * q_dim, dims.hidden, 0.04)?;
    let (k_gpu, k_w) = u8_weight(&executor, &mut rng, kv_dim, dims.hidden, 0.04)?;
    let (v_gpu, v_w) = u8_weight(&executor, &mut rng, kv_dim, dims.hidden, 0.04)?;
    let (o_gpu, o_w) = u8_weight(&executor, &mut rng, dims.hidden, q_dim, 0.02)?;
    let q_norm = rng.fill(256, -0.3, 0.3);
    let k_norm = rng.fill(256, -0.3, 0.3);
    let x = rng.fill(tokens * dims.hidden, -1.0, 1.0);
    let cpu_w = CpuQsa {
        q_proj: q_w,
        k_proj: k_w,
        v_proj: v_w,
        o_proj: o_w,
        q_norm: q_norm.clone(),
        k_norm: k_norm.clone(),
        index_qk: Tensor::from_vec(
            vec![5 * 128, dims.hidden],
            rng.fill(5 * 128 * 2560, -0.02, 0.02),
        )?,
        index_q_norm: vec![0.0; 128],
        index_k_norm: vec![0.0; 128],
    };
    let cpu_dims = QsaDims {
        heads: 24,
        kv_heads: 2,
        head_dim: 256,
        rotary_dim: 64,
        rope_theta: 1.0e7,
        index_heads: 4,
        index_head_dim: 128,
        budget: 2048,
        compress: 4,
        eps: 1.0e-6,
    };
    let expected = qsa(
        &Tensor::from_vec(vec![tokens, dims.hidden], x.clone())?,
        &cpu_w,
        cpu_dims,
        None,
    )?
    .output;

    let one_plus = |w: &[f32]| -> Vec<f32> { w.iter().map(|v| 1.0 + v).collect() };
    let weights = QsaLayerWeights {
        q_proj: q_gpu,
        k_proj: k_gpu,
        v_proj: v_gpu,
        o_proj: o_gpu,
        q_norm: buffer(&executor, &one_plus(&q_norm)),
        k_norm: buffer(&executor, &one_plus(&k_norm)),
    };
    let capacity = tokens.max(64);
    let scratch = QsaScratch {
        q_gate: zeros(&executor, 2 * q_dim),
        q_raw: zeros(&executor, q_dim),
        gate: zeros(&executor, q_dim),
        q_roped: zeros(&executor, q_dim),
        k_raw: zeros(&executor, kv_dim),
        k_roped: zeros(&executor, kv_dim),
        v: zeros(&executor, kv_dim),
        scores: zeros(&executor, 24 * capacity),
        ctx: zeros(&executor, q_dim),
        gated: zeros(&executor, q_dim),
    };
    let mut kv = arena.full_attention_dense(capacity, 24, 2, 256, false)?;
    let queue = executor.device().new_command_queue();
    for t in 0..tokens {
        let x_buf = buffer(&executor, &x[t * dims.hidden..(t + 1) * dims.hidden]);
        let out = zeros(&executor, dims.hidden);
        let command_buffer = queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encode_qsa_dense(
            &arena, &executor, encoder, &mut kv, &weights, &scratch, dims, t, &x_buf, &out,
        )?;
        encoder.end_encoding();
        commit_and_wait(command_buffer)?;
        let got = read_f32_buffer(&out, dims.hidden)?;
        let want = &expected.data()[t * dims.hidden..(t + 1) * dims.hidden];
        let (err, max) = relative_error(&got, want);
        assert!(err <= 1.0e-5, "QSA dense pas {t} : {err} (max {max})");
    }
    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    let x_buf = zeros(&executor, dims.hidden);
    let refused = encode_qsa_dense(
        &arena,
        &executor,
        encoder,
        &mut kv,
        &weights,
        &scratch,
        dims,
        dims.dense_max,
        &x_buf,
        &x_buf,
    );
    encoder.end_encoding();
    assert!(refused.is_err(), "position 2051 acceptée");
    Ok(())
}

/// Le KV Flash-Next reste f32 sous un override de format (bf16 ou quantifié).
#[cfg(feature = "devtools")]
#[test]
fn flash_next_kv_format_is_pinned_to_f32() -> Result<()> {
    use crate::decode_resident::{with_kv_format_override, DecodeResidentState, KvFormat};
    let Some(executor) = executor()? else {
        return Ok(());
    };
    let arena = DecodeResidentState::new(executor.device().clone())?;
    for format in [KvFormat::Bf16, KvFormat::Q8] {
        let kv = with_kv_format_override(format, || {
            arena.full_attention_with_format(16, 24, 2, 256, KvFormat::F32, None)
        })?;
        assert_eq!(kv.format(), KvFormat::F32, "override {format:?}");
    }
    Ok(())
}

/// Identifiants `input_ids` du golden N2 `layers.safetensors` (texte réel).
fn n2_input_ids(golden: &std::path::Path) -> Vec<usize> {
    let path = golden.join("layers.safetensors");
    let header = crate::qwen_loader::read_shard_header(&path).expect("invariant: golden lisible");
    let entry = header
        .tensors
        .get("input_ids")
        .expect("invariant: input_ids");
    let mut bytes = vec![0_u8; entry.data_offsets[1] - entry.data_offsets[0]];
    use std::os::unix::fs::FileExt;
    std::fs::File::open(&path)
        .expect("invariant: golden")
        .read_exact_at(&mut bytes, header.data_start + entry.data_offsets[0] as u64)
        .expect("invariant: lecture input_ids");
    bytes
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().expect("invariant: 8 octets")) as usize)
        .collect()
}

/// KL(p‖q) en f64 entre deux vecteurs de logits.
fn kl_logits(p: &[f32], q: &[f32]) -> f64 {
    let log_softmax = |x: &[f32]| -> Vec<f64> {
        let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let sum: f64 = x.iter().map(|&v| (f64::from(v) - max).exp()).sum();
        x.iter().map(|&v| f64::from(v) - max - sum.ln()).collect()
    };
    let (lp, lq) = (log_softmax(p), log_softmax(q));
    lp.iter().zip(&lq).map(|(a, b)| a.exp() * (a - b)).sum()
}

/// Mode d'une passe N2 résidente.
#[derive(Clone, Copy, Debug, PartialEq)]
enum RoutingMode {
    /// Routage propre du GPU (témoin, sans porte dure).
    Native,
    /// Experts du CPU imposés aux quasi-égalités (marge ≤ 1e-5).
    Forced,
    /// Comme `Forced`, plus un mauvais expert imposé à `(token, couche)` de
    /// marge nette : la porte doit passer au rouge.
    Mutation(usize, usize),
}

/// Mesures d'un token : écart relatif par couche, KL des logits, argmax.
struct TokenReport {
    layer_errors: Vec<f64>,
    kl: f64,
    swap: Option<(usize, usize, f64)>,
}

/// Référence CPU d'une séquence : résiduels, logits, routage et marges.
struct CpuChain {
    layers: Vec<crate::Tensor>,
    logits: crate::Tensor,
    routing: Vec<Vec<Vec<usize>>>,
    margins: Vec<Vec<f32>>,
    width: usize,
    vocab: usize,
}

/// Marge de quasi-égalité du routeur (règle de l'étape 1).
const ROUTING_TIE: f32 = 1.0e-5;

fn run_resident_pass(
    dir: &std::path::Path,
    ids: &[usize],
    cpu: &CpuChain,
    mode: RoutingMode,
    steps: usize,
) -> (Vec<TokenReport>, usize) {
    use crate::qwen_loader::qwen4_exp::Qwen4ExpCheckpoint;
    let guard = crate::MemoryGuard::process();
    let mut session = FlashNextSession::open(
        Qwen4ExpCheckpoint::open(dir).expect("invariant: checkpoint"),
        ids.len(),
        Some(&guard),
    )
    .expect("invariant: session résidente");
    let layers = cpu.layers.len();
    let (width, vocab) = (cpu.width, cpu.vocab);
    let mut imposed = 0;
    let mut reports = Vec::with_capacity(steps);
    for (t, &token) in ids.iter().enumerate().take(steps) {
        if mode != RoutingMode::Native {
            session.forced_routing = (0..layers)
                .map(|layer| {
                    let set = &cpu.routing[layer][t];
                    if mode == RoutingMode::Mutation(t, layer) {
                        // Un expert retenu remplacé par le premier expert non retenu.
                        let outsider = (0..)
                            .find(|e| !set.contains(e))
                            .expect("invariant: expert libre");
                        let mut wrong: Vec<u32> = set.iter().map(|&e| e as u32).collect();
                        wrong[0] = outsider as u32;
                        Some(wrong)
                    } else if cpu.margins[layer][t] <= ROUTING_TIE {
                        Some(set.iter().map(|&e| e as u32).collect())
                    } else {
                        None
                    }
                })
                .collect();
            imposed += session
                .forced_routing
                .iter()
                .filter(|f| f.is_some())
                .count();
        }
        let out = session
            .step(token, true, true)
            .expect("invariant: pas résident");
        let got_layers = out.layers.expect("invariant: résiduels");
        let layer_errors = got_layers
            .iter()
            .enumerate()
            .map(|(index, got)| {
                relative_error(got, &cpu.layers[index].data()[t * width..(t + 1) * width]).0
            })
            .collect();
        let got = out.logits.expect("invariant: logits");
        let want = &cpu.logits.data()[t * vocab..(t + 1) * vocab];
        let cpu_top = want
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1).then(b.0.cmp(&a.0)))
            .map(|(i, _)| i)
            .expect("invariant: vocabulaire non vide");
        let gpu_top = out.argmax as usize;
        let swap = (gpu_top != cpu_top).then(|| {
            let max = want[cpu_top];
            let sum: f64 = want.iter().map(|&v| f64::from(v - max).exp()).sum();
            (
                cpu_top,
                gpu_top,
                (1.0 - f64::from(want[gpu_top] - max).exp()) / sum,
            )
        });
        reports.push(TokenReport {
            layer_errors,
            kl: kl_logits(want, &got),
            swap,
        });
    }
    (reports, imposed)
}

/// Synthèse d'une passe sur un préfixe.
struct Summary {
    worst: f64,
    kl: f64,
    /// `(token, argmax CPU, argmax GPU, écart de probabilité)`.
    swaps: Vec<(usize, usize, usize, f64)>,
    bad: usize,
    first: Option<(usize, usize, f64)>,
}

/// Synthèse d'une passe sur le préfixe `[0, prefix)` : écart max par couche,
/// KL max, argmax différents, tokens au-delà de 1e-3 (premier compris).
fn summarize(reports: &[TokenReport], prefix: usize) -> Summary {
    let reports = &reports[..prefix.min(reports.len())];
    let worst = reports
        .iter()
        .flat_map(|r| r.layer_errors.iter().copied())
        .fold(0.0_f64, f64::max);
    let kl = reports.iter().map(|r| r.kl).fold(0.0_f64, f64::max);
    let swaps = reports
        .iter()
        .enumerate()
        .filter_map(|(t, r)| r.swap.map(|(a, b, gap)| (t, a, b, gap)))
        .collect();
    let bad: Vec<(usize, usize, f64)> = reports
        .iter()
        .enumerate()
        .filter_map(|(t, r)| {
            r.layer_errors
                .iter()
                .position(|&e| e > 1.0e-3)
                .map(|layer| (t, layer, r.layer_errors[layer]))
        })
        .collect();
    Summary {
        worst,
        kl,
        swaps,
        bad: bad.len(),
        first: bad.first().copied(),
    }
}

/// N2 résident : decode GPU token par token (teacher forcing sur le texte du
/// golden) contre la référence CPU f32 sans cache, chaîne native des deux côtés.
/// Longueur : `SARAGOSSA_QWEN4EXP_TOKENS` (2051 par défaut).
///
/// Une quasi-égalité du routeur (marge ≤ 1e-5) fait basculer un expert ; la
/// divergence passe ensuite aux tokens suivants par l'état récurrent et le KV.
/// La porte porte donc sur la **passe à routage imposé** (experts du CPU aux
/// quasi-égalités), à 64, 256 et 2051 tokens : écart relatif ≤ 1e-3 par couche
/// et par token, KL max ≤ 1e-3, argmax identique sauf quasi-égalité (écart de
/// probabilité ≤ 1e-5). La passe native est mesurée comme témoin. Une
/// mutation (mauvais expert imposé à marge nette) doit faire échouer la porte.
#[test]
#[ignore = "poids réels : SARAGOSSA_QWEN4EXP_DIR et SARAGOSSA_QWEN4EXP_GOLDEN"]
fn n2_resident_decode_matches_cpu_reference() {
    use crate::qwen4_exp::Qwen4ExpReference;
    use crate::qwen_loader::qwen4_exp::Qwen4ExpCheckpoint;
    let dir = std::path::PathBuf::from(
        std::env::var_os("SARAGOSSA_QWEN4EXP_DIR").expect("SARAGOSSA_QWEN4EXP_DIR requis"),
    );
    let golden =
        std::env::var_os("SARAGOSSA_QWEN4EXP_GOLDEN").expect("SARAGOSSA_QWEN4EXP_GOLDEN requis");
    let tokens: usize = std::env::var("SARAGOSSA_QWEN4EXP_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(dense_max_visible(2048, 4));
    let ids: Vec<usize> = n2_input_ids(std::path::Path::new(&golden))[..tokens].to_vec();

    let started = std::time::Instant::now();
    let reference =
        Qwen4ExpReference::new(Qwen4ExpCheckpoint::open(&dir).expect("invariant: checkpoint"));
    let config = &reference.checkpoint().config;
    let mut h = reference.embed(&ids).expect("invariant: embedding");
    let (mut layers, mut routing, mut margins) = (Vec::new(), Vec::new(), Vec::new());
    for index in 0..config.layer_kinds.len() {
        let out = reference
            .layer(index, &h, &ids)
            .expect("invariant: couche CPU");
        h = out.hidden;
        layers.push(h.clone());
        routing.push(out.routing);
        margins.push(out.routing_margins);
    }
    let cpu = CpuChain {
        logits: reference
            .logits(&reference.final_mixer(&h).expect("invariant: mixer"))
            .expect("invariant: logits CPU"),
        width: config.hc_width(),
        vocab: config.vocab,
        layers,
        routing,
        margins,
    };
    eprintln!(
        "référence CPU : {tokens} tokens en {:.0} s",
        started.elapsed().as_secs_f64()
    );
    let ties: usize = cpu
        .margins
        .iter()
        .flatten()
        .filter(|&&m| m <= ROUTING_TIE)
        .count();
    eprintln!("quasi-égalités du routeur CPU (marge ≤ 1e-5) : {ties}");

    let mut prefixes: Vec<usize> = [64, 256, tokens]
        .into_iter()
        .filter(|&p| p <= tokens)
        .collect();
    prefixes.dedup();
    let (native, _) = run_resident_pass(&dir, &ids, &cpu, RoutingMode::Native, tokens);
    for &prefix in &prefixes {
        let Summary {
            worst,
            kl,
            swaps,
            bad,
            first,
        } = summarize(&native, prefix);
        eprintln!(
            "native  {prefix:4} : écart max {worst:.2e}, KL max {kl:.2e}, argmax ≠ {}, \
             tokens > 1e-3 : {bad}, premier (token, couche, écart) {first:?}",
            swaps.len()
        );
    }
    drop(native);
    let (forced, imposed) = run_resident_pass(&dir, &ids, &cpu, RoutingMode::Forced, tokens);
    eprintln!("impositions (token × couche) : {imposed}");
    let mut red = Vec::new();
    for &prefix in &prefixes {
        let Summary {
            worst,
            kl,
            swaps,
            bad,
            first,
        } = summarize(&forced, prefix);
        eprintln!(
            "imposée {prefix:4} : écart max {worst:.2e}, KL max {kl:.2e}, argmax ≠ {swaps:?}, \
             tokens > 1e-3 : {bad}, premier {first:?}"
        );
        if worst > 1.0e-3 || kl > 1.0e-3 || swaps.iter().any(|s| s.3 > 1.0e-5) {
            red.push(prefix);
        }
    }
    let worst_by_layer: Vec<String> = (0..cpu.layers.len())
        .map(|layer| {
            let e = forced
                .iter()
                .map(|r| r.layer_errors[layer])
                .fold(0.0_f64, f64::max);
            format!("{e:.1e}")
        })
        .collect();
    eprintln!(
        "imposée, écart max par couche : {}",
        worst_by_layer.join(" ")
    );
    drop(forced);

    // Mutation : (token ≥ 8, couche) de plus grande marge CPU parmi les 64 premiers.
    let (token, layer) = (8..tokens.min(64))
        .flat_map(|t| (0..cpu.margins.len()).map(move |l| (t, l)))
        .max_by(|a, b| cpu.margins[a.1][a.0].total_cmp(&cpu.margins[b.1][b.0]))
        .expect("invariant: au moins 9 tokens");
    eprintln!("mutation : marge CPU {:.2e}", cpu.margins[layer][token]);
    let steps = (token + 8).min(tokens);
    let (mutated, _) =
        run_resident_pass(&dir, &ids, &cpu, RoutingMode::Mutation(token, layer), steps);
    let Summary {
        worst, kl, first, ..
    } = summarize(&mutated, steps);
    eprintln!("mutation (token {token}, couche {layer}) : écart max {worst:.2e}, KL max {kl:.2e}, premier {first:?}");
    assert!(worst > 1.0e-3, "mutation non détectée par la porte");
    assert!(red.is_empty(), "passe imposée rouge aux préfixes {red:?}");
}

/// Débit AR résident : contexte de 1024 tokens (texte du golden), puis decode
/// glouton chronométré token par token (temps mural, argmax relu). Affiche
/// médiane, p10/p90 et le temps CPU de préparation (embedding + PLE). À lancer
/// seulement machine libre.
#[test]
#[ignore = "débit : poids réels, machine libre"]
fn bench_resident_decode_after_1k_context() {
    use crate::qwen_loader::qwen4_exp::Qwen4ExpCheckpoint;
    let dir = std::path::PathBuf::from(
        std::env::var_os("SARAGOSSA_QWEN4EXP_DIR").expect("SARAGOSSA_QWEN4EXP_DIR requis"),
    );
    let golden =
        std::env::var_os("SARAGOSSA_QWEN4EXP_GOLDEN").expect("SARAGOSSA_QWEN4EXP_GOLDEN requis");
    let (context, generated) = (1024, 256);
    let ids = n2_input_ids(std::path::Path::new(&golden));
    let mut session = FlashNextSession::open(
        Qwen4ExpCheckpoint::open(&dir).expect("invariant: checkpoint"),
        context + generated,
        Some(&crate::MemoryGuard::process()),
    )
    .expect("invariant: session");
    let mut next = 0;
    for &token in &ids[..context] {
        next = session
            .step(token, false, false)
            .expect("invariant: contexte")
            .argmax as usize;
    }
    let mut walls = Vec::with_capacity(generated);
    let mut hosts = Vec::with_capacity(generated);
    for _ in 0..generated {
        let started = std::time::Instant::now();
        let out = session.step(next, false, false).expect("invariant: decode");
        walls.push(started.elapsed().as_secs_f64() * 1e3);
        hosts.push(out.host.as_secs_f64() * 1e3);
        next = out.argmax as usize;
    }
    let pct = |v: &mut Vec<f64>, q: f64| {
        v.sort_by(f64::total_cmp);
        v[((v.len() - 1) as f64 * q).round() as usize]
    };
    let (p10, p50, p90) = (
        pct(&mut walls, 0.1),
        pct(&mut walls, 0.5),
        pct(&mut walls, 0.9),
    );
    let host = pct(&mut hosts, 0.5);
    eprintln!(
        "decode après {context} tokens, {generated} pas : médiane {p50:.2} ms ({:.1} tok/s), \
         p10 {p10:.2}, p90 {p90:.2} ; préparation CPU médiane {host:.3} ms",
        1e3 / p50
    );
}

/// Validation à l'ouverture : la configuration de Flash-Next est acceptée
/// (borne dense 2051, calculée depuis budget et compression) ; chaque variante
/// non portée est refusée avant toute allocation.
#[test]
fn resident_config_validation_accepts_flash_next_and_refuses_variants() -> Result<()> {
    use super::session::validate_resident_config;
    use crate::qwen_loader::qwen4_exp::Qwen4ExpConfig;
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/qwen4exp_mini/source/config.json");
    let json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path).expect("invariant: config de la fixture"))
            .map_err(|e| InferError::Config(e.to_string()))?;
    let mut base = Qwen4ExpConfig::from_json(&json)?;
    // Géométrie résidente de Flash-Next sur la fixture.
    (base.linear_key_dim, base.linear_value_dim, base.conv_kernel) = (128, 128, 4);
    (base.linear_key_heads, base.linear_value_heads) = (16, 48);
    (base.indexer_budget, base.indexer_compress) = (2048, 4);
    base.top_k = base.top_k.min(base.experts);
    assert_eq!(validate_resident_config(&base)?, 2051);
    type Edit = fn(&mut Qwen4ExpConfig);
    let variants: [(&str, Edit); 6] = [
        ("norm_topk_prob", |c| c.norm_topk_prob = false),
        ("gate SiLU", |c| c.sigmoid_output_gate = false),
        ("d_k 64", |c| c.linear_key_dim = 64),
        ("têtes V", |c| c.linear_value_heads = 47),
        ("top-k 17", |c| (c.top_k, c.experts) = (17, 512)),
        ("budget", |c| c.indexer_budget = 2047),
    ];
    for (label, edit) in variants {
        let mut config = base.clone();
        edit(&mut config);
        assert!(
            validate_resident_config(&config).is_err(),
            "{label} accepté"
        );
    }
    Ok(())
}
