//! Oracles CPU des briques résidentes Flash-Next, aux dimensions réelles.

use metal::{Buffer, MTLResourceOptions};

use super::attention::{
    dense_max_visible, encode_qsa_dense, QsaDecodeDims, QsaLayerWeights, QsaScratch,
};
use super::hc::{encode_hc_combine, encode_hc_mix, HcDims, HcScratch, HcWeights};
use super::kernels::FlashNextKernels;
use super::ple::{encode_ple, PleScratch, PleState, PleWeights};
use super::session::FlashNextSession;
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
    let inject_gpu = buffer(&executor, inject_w.data());
    let h_data = rng.fill(width, -2.0, 2.0);
    let f_data = rng.fill(dims.hidden, -1.0, 1.0);
    let h = Tensor::from_vec(vec![1, width], h_data.clone())?;
    let f = Tensor::from_vec(vec![1, dims.hidden], f_data.clone())?;

    let scratch = HcScratch {
        normed: zeros(&executor, width),
        low: zeros(&executor, dims.lowrank),
        up: zeros(&executor, width),
        inject: zeros(&executor, dims.hc * super::hc::INJECT_CHUNKS),
        down_partial: zeros(&executor, dims.lowrank * super::hc::DOWN_CHUNKS),
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
    /// Routage propre du GPU.
    Native,
    /// Experts du CPU imposés aux points de marge de probabilité ≤ seuil.
    Forced(f32),
    /// Comme `Forced(seuil)`, plus un mauvais expert imposé à `(token, couche)`.
    Mutation(usize, usize, f32),
}

/// Mesures d'un token.
struct TokenReport {
    /// Écart relatif par couche (NaN si une sortie n'est pas finie).
    layer_errors: Vec<f64>,
    kl: f64,
    /// `(argmax CPU, argmax GPU, écart de probabilité oracle)` s'ils diffèrent.
    swap: Option<(usize, usize, f64)>,
    /// Experts routés par le GPU `[couche]`, triés.
    routing: Vec<Vec<usize>>,
    /// Points `(couche)` imposés à ce token.
    imposed: Vec<usize>,
}

/// Référence CPU d'une séquence.
struct CpuChain {
    layers: Vec<crate::Tensor>,
    logits: crate::Tensor,
    /// Experts routés `[couche][token]`, triés.
    routing: Vec<Vec<Vec<usize>>>,
    /// Marge de probabilité au seuil du top-k `[couche][token]`.
    margins: Vec<Vec<f32>>,
    /// `(k-ième, (k+1)-ième)` experts `[couche][token]`.
    boundary: Vec<Vec<(usize, usize)>>,
    width: usize,
    vocab: usize,
}

/// Seuils de marge de probabilité essayés, du plus strict au plus large.
const TIE_THRESHOLDS: [f32; 3] = [1.0e-7, 1.0e-6, 1.0e-5];
/// Marge maximale admise pour la première bascule native.
const NATIVE_FIRST_TIE: f32 = 1.0e-6;

/// Écart relatif, NaN si l'une des entrées n'est pas finie.
fn finite_relative_error(got: &[f32], want: &[f32]) -> f64 {
    if got.len() != want.len() || got.iter().chain(want).any(|v| !v.is_finite()) {
        return f64::NAN;
    }
    relative_error(got, want).0
}

/// Vrai si `value` est finie et ≤ `limit` (NaN et infinis refusés).
fn within<T: Into<f64>>(value: T, limit: f64) -> bool {
    let value = value.into();
    value.is_finite() && value <= limit
}

/// Maximum qui propage NaN (`f64::max` l'ignorerait).
fn nan_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

fn run_resident_pass(
    dir: &std::path::Path,
    ids: &[usize],
    cpu: &CpuChain,
    mode: RoutingMode,
    steps: usize,
) -> Vec<TokenReport> {
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
    let mut reports = Vec::with_capacity(steps);
    for (t, &token) in ids.iter().enumerate().take(steps) {
        let threshold = match mode {
            RoutingMode::Native => None,
            RoutingMode::Forced(th) | RoutingMode::Mutation(_, _, th) => Some(th),
        };
        session.forced_routing = (0..layers)
            .map(|layer| {
                let set = &cpu.routing[layer][t];
                if matches!(mode, RoutingMode::Mutation(mt, ml, _) if (mt, ml) == (t, layer)) {
                    // Un expert retenu remplacé par le premier expert non retenu.
                    let outsider = (0..)
                        .find(|e| !set.contains(e))
                        .expect("invariant: expert libre");
                    let mut wrong: Vec<u32> = set.iter().map(|&e| e as u32).collect();
                    wrong[0] = outsider as u32;
                    Some(wrong)
                } else if threshold.is_some_and(|th| cpu.margins[layer][t] <= th) {
                    Some(set.iter().map(|&e| e as u32).collect())
                } else {
                    None
                }
            })
            .collect();
        let imposed = (0..layers)
            .filter(|&l| session.forced_routing[l].is_some())
            .collect();
        let out = session
            .step(token, true, true)
            .expect("invariant: pas résident");
        let got_layers = out.layers.expect("invariant: résiduels");
        let layer_errors = got_layers
            .iter()
            .enumerate()
            .map(|(index, got)| {
                finite_relative_error(got, &cpu.layers[index].data()[t * width..(t + 1) * width])
            })
            .collect();
        let got = out.logits.expect("invariant: logits");
        let want = &cpu.logits.data()[t * vocab..(t + 1) * vocab];
        let kl = if got.iter().all(|v| v.is_finite()) {
            kl_logits(want, &got)
        } else {
            f64::NAN
        };
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
        let routing = out
            .routing
            .expect("invariant: routage GPU")
            .into_iter()
            .map(|set| {
                let mut set: Vec<usize> = set.into_iter().map(|e| e as usize).collect();
                set.sort_unstable();
                set
            })
            .collect();
        reports.push(TokenReport {
            layer_errors,
            kl,
            swap,
            routing,
            imposed,
        });
    }
    reports
}

/// Synthèse d'une passe sur un préfixe.
struct Summary {
    /// NaN si une valeur n'est pas finie.
    worst: f64,
    kl: f64,
    /// `(token, argmax CPU, argmax GPU, écart de probabilité)`.
    swaps: Vec<(usize, usize, usize, f64)>,
    bad: usize,
    first: Option<(usize, usize, f64)>,
    /// Points non imposés où le GPU route autrement que le CPU.
    routing_mismatches: usize,
}

/// Synthèse d'une passe sur le préfixe `[0, prefix)`.
fn summarize(reports: &[TokenReport], cpu: &CpuChain, prefix: usize) -> Summary {
    let reports = &reports[..prefix.min(reports.len())];
    let worst = reports
        .iter()
        .flat_map(|r| r.layer_errors.iter().copied())
        .fold(0.0_f64, nan_max);
    let kl = reports.iter().map(|r| r.kl).fold(0.0_f64, nan_max);
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
                .position(|&e| !within(e, 1.0e-3))
                .map(|layer| (t, layer, r.layer_errors[layer]))
        })
        .collect();
    let routing_mismatches = reports
        .iter()
        .enumerate()
        .map(|(t, r)| {
            (0..r.routing.len())
                .filter(|&l| !r.imposed.contains(&l) && r.routing[l] != cpu.routing[l][t])
                .count()
        })
        .sum();
    Summary {
        worst,
        kl,
        swaps,
        bad: bad.len(),
        first: bad.first().copied(),
        routing_mismatches,
    }
}

/// Porte d'une passe imposée : écart ≤ 1e-3, KL ≤ 1e-3, valeurs finies,
/// argmax identique sauf quasi-égalité (≤ 1e-5), routage identique aux points
/// non imposés.
fn forced_gate_green(summary: &Summary) -> bool {
    within(summary.worst, 1.0e-3)
        && within(summary.kl, 1.0e-3)
        && summary.swaps.iter().all(|s| within(s.3, 1.0e-5))
        && summary.routing_mismatches == 0
}

/// Contrôle de la passe native : jusqu'à la première bascule de routage
/// `(t₀, l₀)`, écart ≤ 1e-3 (fini) et argmax identique ; la bascule tombe sur
/// une marge de probabilité ≤ [`NATIVE_FIRST_TIE`] et n'échange que la paire
/// au seuil (k-ième ↔ (k+1)-ième expert du CPU). Renvoie `(t₀, l₀, marge)`.
fn native_check(
    reports: &[TokenReport],
    cpu: &CpuChain,
) -> std::result::Result<Option<(usize, usize, f32)>, String> {
    let first = reports.iter().enumerate().find_map(|(t, r)| {
        (0..r.routing.len())
            .find(|&l| r.routing[l] != cpu.routing[l][t])
            .map(|l| (t, l))
    });
    let (end_token, end_layer) = first.unwrap_or((reports.len(), 0));
    for (t, r) in reports.iter().enumerate().take(end_token + 1) {
        let layers = if t == end_token {
            end_layer
        } else {
            r.layer_errors.len()
        };
        if let Some(l) = (0..layers).find(|&l| !within(r.layer_errors[l], 1.0e-3)) {
            return Err(format!(
                "écart {:.2e} au token {t}, couche {l}, avant la première bascule",
                r.layer_errors[l]
            ));
        }
        if t < end_token && r.swap.is_some() {
            return Err(format!(
                "argmax différent au token {t}, avant la première bascule"
            ));
        }
    }
    let Some((t, l)) = first else {
        return Ok(None);
    };
    let margin = cpu.margins[l][t];
    if !within(margin, f64::from(NATIVE_FIRST_TIE)) {
        return Err(format!(
            "première bascule (token {t}, couche {l}) à marge {margin:.2e}"
        ));
    }
    let (kth, next) = cpu.boundary[l][t];
    let mut expected: Vec<usize> = cpu.routing[l][t]
        .iter()
        .map(|&e| if e == kth { next } else { e })
        .collect();
    expected.sort_unstable();
    if reports[t].routing[l] != expected {
        return Err(format!(
            "bascule (token {t}, couche {l}) hors de la paire au seuil : GPU {:?}, CPU {:?}",
            reports[t].routing[l], cpu.routing[l][t]
        ));
    }
    Ok(Some((t, l, margin)))
}

/// Vrai si toutes les sorties et métriques d'une passe sont finies.
fn all_finite(reports: &[TokenReport]) -> bool {
    reports
        .iter()
        .all(|r| r.kl.is_finite() && r.layer_errors.iter().all(|e| e.is_finite()))
}

/// Toutes les bascules de routage d'une passe : `(token, couche, paire au
/// seuil ?, marge CPU)`.
fn routing_swaps(reports: &[TokenReport], cpu: &CpuChain) -> Vec<(usize, usize, bool, f32)> {
    let mut out = Vec::new();
    for (t, r) in reports.iter().enumerate() {
        for l in 0..r.routing.len() {
            if r.routing[l] != cpu.routing[l][t] {
                let (kth, next) = cpu.boundary[l][t];
                let mut pair: Vec<usize> = cpu.routing[l][t]
                    .iter()
                    .map(|&e| if e == kth { next } else { e })
                    .collect();
                pair.sort_unstable();
                out.push((t, l, r.routing[l] == pair, cpu.margins[l][t]));
            }
        }
    }
    out
}

/// La porte refuse les NaN et un écart tardif, que `f64::max` aurait masqués.
#[test]
fn n2_gate_refuses_late_nan_and_native_check_refuses_early_drift() {
    let cpu = CpuChain {
        layers: vec![crate::Tensor::from_vec(vec![1], vec![0.0]).expect("invariant: tenseur")],
        logits: crate::Tensor::from_vec(vec![1], vec![0.0]).expect("invariant: tenseur"),
        routing: vec![vec![vec![1, 2]; 4]],
        margins: vec![vec![1.0; 4]],
        boundary: vec![vec![(2, 3); 4]],
        width: 1,
        vocab: 1,
    };
    let report = |err: f64, kl: f64| TokenReport {
        layer_errors: vec![err],
        kl,
        swap: None,
        routing: vec![vec![1, 2]],
        imposed: Vec::new(),
    };
    let clean: Vec<TokenReport> = (0..4).map(|_| report(1.0e-6, 1.0e-9)).collect();
    assert!(forced_gate_green(&summarize(&clean, &cpu, 4)));
    let mut late_nan: Vec<TokenReport> = (0..3).map(|_| report(1.0e-6, 1.0e-9)).collect();
    late_nan.push(report(f64::NAN, 1.0e-9));
    assert!(
        !forced_gate_green(&summarize(&late_nan, &cpu, 4)),
        "NaN d'écart accepté"
    );
    let mut late_kl: Vec<TokenReport> = (0..3).map(|_| report(1.0e-6, 1.0e-9)).collect();
    late_kl.push(report(1.0e-6, f64::NAN));
    assert!(
        !forced_gate_green(&summarize(&late_kl, &cpu, 4)),
        "NaN de KL accepté"
    );
    assert!(native_check(&late_nan, &cpu).is_err(), "NaN natif accepté");
    assert!(all_finite(&clean) && !all_finite(&late_nan) && !all_finite(&late_kl));
    let mut wrong = (0..4).map(|_| report(1.0e-6, 1.0e-9)).collect::<Vec<_>>();
    wrong[2].routing[0] = vec![1, 9];
    assert!(
        native_check(&wrong, &cpu).is_err(),
        "bascule à marge nette acceptée"
    );
}

/// N2 résident : decode GPU token par token (teacher forcing sur le texte du
/// golden) contre la référence CPU f32 sans cache. Longueur :
/// `SARAGOSSA_QWEN4EXP_TOKENS` (2051 par défaut).
///
/// - Passe native contrôlée jusqu'à la première bascule ([`native_check`]),
///   témoin publié au-delà.
/// - Passe imposée : experts du CPU aux points de marge de probabilité ≤ seuil,
///   seuil le plus strict de [`TIE_THRESHOLDS`] qui garde la porte verte
///   ([`forced_gate_green`]) à 64, 256 et 2051 tokens.
/// - Mutation : un mauvais expert imposé à marge nette fait échouer la porte.
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
    let (mut layers, mut routing, mut margins, mut boundary) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for index in 0..config.layer_kinds.len() {
        let out = reference
            .layer(index, &h, &ids)
            .expect("invariant: couche CPU");
        h = out.hidden;
        layers.push(h.clone());
        routing.push(out.routing);
        margins.push(out.routing_margins);
        boundary.push(out.routing_boundary);
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
        boundary,
    };
    assert!(
        cpu.logits.data().iter().all(|v| v.is_finite()),
        "logits CPU non finis"
    );
    eprintln!(
        "référence CPU : {tokens} tokens en {:.0} s",
        started.elapsed().as_secs_f64()
    );
    for th in TIE_THRESHOLDS {
        let n = cpu.margins.iter().flatten().filter(|&&m| m <= th).count();
        eprintln!(
            "points de marge de probabilité ≤ {th:.0e} : {n} sur {}",
            tokens * cpu.layers.len()
        );
    }

    let mut prefixes: Vec<usize> = [64, 256, tokens]
        .into_iter()
        .filter(|&p| p <= tokens)
        .collect();
    prefixes.dedup();
    let native = run_resident_pass(&dir, &ids, &cpu, RoutingMode::Native, tokens);
    let native_first = native_check(&native, &cpu);
    // Toute la séquence native : sorties et métriques finies.
    let native_finite = all_finite(&native);
    let swaps = routing_swaps(&native, &cpu);
    eprintln!(
        "native : première bascule {native_first:?} ; bascules {} (paire au seuil : {}), 10 premières {:?}",
        swaps.len(),
        swaps.iter().filter(|s| s.2).count(),
        &swaps[..swaps.len().min(10)]
    );
    for &prefix in &prefixes {
        let s = summarize(&native, &cpu, prefix);
        eprintln!(
            "native  {prefix:4} : écart max {:.2e}, KL max {:.2e}, argmax ≠ {}, tokens > 1e-3 : {}, premier {:?}",
            s.worst, s.kl, s.swaps.len(), s.bad, s.first
        );
    }
    drop(native);

    let mut chosen = None;
    for th in TIE_THRESHOLDS {
        let forced = run_resident_pass(&dir, &ids, &cpu, RoutingMode::Forced(th), tokens);
        let mut green = true;
        for &prefix in &prefixes {
            let s = summarize(&forced, &cpu, prefix);
            eprintln!(
                "imposée ≤ {th:.0e}, {prefix:4} : écart max {:.2e}, KL max {:.2e}, argmax ≠ {:?}, \
                 tokens > 1e-3 : {}, premier {:?}, routage ≠ hors imposition : {}",
                s.worst, s.kl, s.swaps, s.bad, s.first, s.routing_mismatches
            );
            green &= forced_gate_green(&s);
        }
        let by_layer: Vec<String> = (0..cpu.layers.len())
            .map(|l| {
                format!(
                    "{:.1e}",
                    forced.iter().map(|r| r.layer_errors[l]).fold(0.0, nan_max)
                )
            })
            .collect();
        eprintln!(
            "imposée ≤ {th:.0e}, écart max par couche : {}",
            by_layer.join(" ")
        );
        if green {
            chosen = Some(th);
            break;
        }
    }
    let threshold = chosen.unwrap_or(TIE_THRESHOLDS[TIE_THRESHOLDS.len() - 1]);
    eprintln!("seuil retenu : {chosen:?}");

    // Mutations : un mauvais expert imposé au point de plus grande marge CPU,
    // tôt (tokens 8 à 63) puis tard (64 tokens après la première bascule
    // native), dans la passe imposée au seuil retenu.
    let after = native_first
        .as_ref()
        .ok()
        .and_then(|f| f.map(|(t, _, _)| t + 1))
        .unwrap_or(8);
    for range in [8..tokens.min(64), after..tokens.min(after + 64)] {
        let Some((token, layer)) = range
            .flat_map(|t| (0..cpu.margins.len()).map(move |l| (t, l)))
            .max_by(|a, b| cpu.margins[a.1][a.0].total_cmp(&cpu.margins[b.1][b.0]))
        else {
            continue;
        };
        let steps = (token + 8).min(tokens);
        let mutated = run_resident_pass(
            &dir,
            &ids,
            &cpu,
            RoutingMode::Mutation(token, layer, threshold),
            steps,
        );
        let s = summarize(&mutated, &cpu, steps);
        eprintln!(
            "mutation (token {token}, couche {layer}, marge {:.2e}) : écart max {:.2e}, KL max {:.2e}, premier {:?}",
            cpu.margins[layer][token], s.worst, s.kl, s.first
        );
        assert!(
            !forced_gate_green(&s),
            "mutation au token {token} non détectée par la porte"
        );
    }
    assert!(chosen.is_some(), "passe imposée rouge à tous les seuils");
    native_first.unwrap_or_else(|e| panic!("passe native : {e}"));
    assert!(native_finite, "passe native : valeur non finie");
}

/// Débit AR résident : `SARAGOSSA_FLASH_NEXT_BENCH_RUNS` répétitions (3 par
/// défaut) d'un contexte de 1024 tokens (texte du golden, reset entre les
/// répétitions) puis de 256 pas gloutons chronométrés. Publie, par
/// répétition, le débit mural `tokens / durée totale` (porte des 45 tok/s,
/// évaluée sur la plus lente), la médiane, p10/p90 et le maximum par pas, et le
/// temps CPU de préparation (embedding + PLE). À lancer seulement machine libre.
#[test]
#[ignore = "débit : poids réels, machine libre"]
fn bench_resident_decode_after_1k_context() {
    use crate::qwen_loader::qwen4_exp::Qwen4ExpCheckpoint;
    let dir = std::path::PathBuf::from(
        std::env::var_os("SARAGOSSA_QWEN4EXP_DIR").expect("SARAGOSSA_QWEN4EXP_DIR requis"),
    );
    let golden =
        std::env::var_os("SARAGOSSA_QWEN4EXP_GOLDEN").expect("SARAGOSSA_QWEN4EXP_GOLDEN requis");
    let runs: usize = crate::runtime_flags::env_var("SARAGOSSA_FLASH_NEXT_BENCH_RUNS")
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    assert!(
        runs > 0,
        "SARAGOSSA_FLASH_NEXT_BENCH_RUNS doit valoir au moins 1"
    );
    let (context, generated) = (1024, 256);
    let ids = n2_input_ids(std::path::Path::new(&golden));
    let mut session = FlashNextSession::open(
        Qwen4ExpCheckpoint::open(&dir).expect("invariant: checkpoint"),
        context + generated,
        Some(&crate::MemoryGuard::process()),
    )
    .expect("invariant: session");
    let pct = |v: &[f64], q: f64| {
        let mut v = v.to_vec();
        v.sort_by(f64::total_cmp);
        v[((v.len() - 1) as f64 * q).round() as usize]
    };
    let mut slowest = f64::INFINITY;
    for run in 0..runs {
        session.reset().expect("invariant: reset");
        let mut next = 0;
        for &token in &ids[..context] {
            next = session
                .step(token, false, false)
                .expect("invariant: contexte")
                .argmax as usize;
        }
        let mut walls = Vec::with_capacity(generated);
        let mut hosts = Vec::with_capacity(generated);
        let mut encodes = Vec::with_capacity(generated);
        let started = std::time::Instant::now();
        for _ in 0..generated {
            let step = std::time::Instant::now();
            let out = session.step(next, false, false).expect("invariant: decode");
            walls.push(step.elapsed().as_secs_f64() * 1e3);
            hosts.push(out.host.as_secs_f64() * 1e3);
            encodes.push(out.encode.as_secs_f64() * 1e3);
            next = out.argmax as usize;
        }
        let throughput = generated as f64 / started.elapsed().as_secs_f64();
        slowest = slowest.min(throughput);
        eprintln!(
            "répétition {run} : {throughput:.1} tok/s ({generated} tokens / durée totale) ; \
             par pas médiane {:.2} ms, p10 {:.2}, p90 {:.2}, max {:.2} ; préparation CPU médiane {:.3} ms, encodage médian {:.3} ms",
            pct(&walls, 0.5),
            pct(&walls, 0.1),
            pct(&walls, 0.9),
            pct(&walls, 1.0),
            pct(&hosts, 0.5),
            pct(&encodes, 0.5)
        );
    }
    eprintln!("porte 45 tok/s à 1k : répétition la plus lente à {slowest:.1} tok/s");
    assert!(slowest.is_finite(), "débit non mesuré : {slowest}");
    assert!(
        slowest >= 45.0,
        "porte de débit rouge : {slowest:.1} tok/s < 45"
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

/// Reprise après échec : au token 6, une lecture n-gram puis un encodage
/// (après la couche 20) échouent ; la session garde sa position, restaure KV
/// et PLE, puis redonne au bit près les logits d'une session sans échec. Un
/// reset rejoue ensuite le début de la séquence à l'identique.
#[test]
#[ignore = "poids réels : SARAGOSSA_QWEN4EXP_DIR et SARAGOSSA_QWEN4EXP_GOLDEN"]
fn resident_session_recovers_after_ple_and_encode_failures() {
    use crate::qwen_loader::qwen4_exp::Qwen4ExpCheckpoint;
    let dir = std::path::PathBuf::from(
        std::env::var_os("SARAGOSSA_QWEN4EXP_DIR").expect("SARAGOSSA_QWEN4EXP_DIR requis"),
    );
    let golden =
        std::env::var_os("SARAGOSSA_QWEN4EXP_GOLDEN").expect("SARAGOSSA_QWEN4EXP_GOLDEN requis");
    let ids: Vec<usize> = n2_input_ids(std::path::Path::new(&golden))[..16].to_vec();
    let open = || {
        FlashNextSession::open(
            Qwen4ExpCheckpoint::open(&dir).expect("invariant: checkpoint"),
            ids.len(),
            Some(&crate::MemoryGuard::process()),
        )
        .expect("invariant: session")
    };
    let expected: Vec<Vec<f32>> = {
        let mut session = open();
        // Budget de session : allocations propres, plus le scratch interne de
        // l'exécuteur alloué au premier pas, sous `session_state_bytes`.
        let before = session.device_allocated_bytes();
        let first = session
            .step(ids[0], true, false)
            .expect("invariant: premier pas");
        let lazy = session.device_allocated_bytes().saturating_sub(before);
        let budget = super::session::session_state_bytes(
            &Qwen4ExpCheckpoint::open(&dir)
                .expect("invariant: checkpoint")
                .config,
            ids.len(),
        );
        let used = session.owned_state_bytes() + lazy;
        eprintln!("session : {used} octets alloués (dont {lazy} au premier pas), budget {budget}");
        assert!(
            used <= budget,
            "allocations de session {used} > budget {budget}"
        );
        session.reset().expect("invariant: reset");
        drop(first);
        ids.iter()
            .map(|&t| {
                session
                    .step(t, true, false)
                    .expect("invariant: pas")
                    .logits
                    .expect("invariant: logits")
            })
            .collect()
    };
    let mut session = open();
    for (t, &token) in ids.iter().enumerate() {
        if t == 6 {
            session.fail_next_ngram_read = true;
            assert!(
                session.step(token, true, false).is_err(),
                "échec PLE ignoré"
            );
            assert_eq!(session.position(), 6);
            session.fail_encode_after_layer = Some(20);
            assert!(
                session.step(token, true, false).is_err(),
                "échec d'encodage ignoré"
            );
            assert_eq!(session.position(), 6);
        }
        let got = session
            .step(token, true, false)
            .expect("invariant: pas repris");
        assert!(
            got.logits.expect("invariant: logits") == expected[t],
            "token {t}"
        );
    }
    // Reset : la séquence rejouée depuis le début redonne les mêmes logits.
    session.reset().expect("invariant: reset");
    for (t, &token) in ids.iter().enumerate().take(4) {
        let got = session
            .step(token, true, false)
            .expect("invariant: pas après reset");
        assert!(
            got.logits.expect("invariant: logits") == expected[t],
            "token {t} après reset"
        );
    }
    // Reset en échec après quelques couches, capacité non saturée (4 / 16) :
    // le pas suivant est refusé par l'invalidation, pas par la capacité ; un
    // reset complet réhabilite la session.
    session.fail_reset_after_layer = Some(3);
    assert!(session.reset().is_err(), "échec de reset ignoré");
    let refused = session
        .step(ids[4], true, false)
        .err()
        .expect("pas accepté après un reset partiel");
    assert!(
        refused.to_string().contains("session invalidée"),
        "refus inattendu : {refused}"
    );
    session.reset().expect("invariant: reset complet");
    for (t, &token) in ids.iter().enumerate().take(2) {
        let got = session
            .step(token, true, false)
            .expect("invariant: pas après reset complet");
        assert!(
            got.logits.expect("invariant: logits") == expected[t],
            "token {t} après reset complet"
        );
    }
}

/// Garde mémoire du chargement : un plafond égal à l'état résident final est
/// refusé, car une étape a un pic transitoire (copies CPU des triplets,
/// empilement des experts) ; un plafond égal au pic est accepté. La simulation
/// suit l'ordre du chargement réel, sur la fixture N1 convertie.
#[cfg(feature = "devtools")]
#[test]
fn load_plan_guard_refuses_transient_peak_above_final_resident() -> Result<()> {
    use super::weights::load_plan;
    use crate::devtools::convert::{convert_qwen4exp, ConvertOptions};
    use crate::qwen_loader::qwen4_exp::Qwen4ExpCheckpoint;
    use crate::{MemoryGuard, MemoryLimits};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;
    let dst = tempfile::tempdir().expect("invariant: dossier temporaire");
    convert_qwen4exp(&ConvertOptions {
        src: std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/qwen4exp_mini/source"),
        dst: dst.path().to_path_buf(),
        source_revision: "fixture".to_string(),
        source_repo: None,
        converter_commit: None,
        only: None,
        verify: false,
        plan_only: false,
    })?;
    let ckpt = Qwen4ExpCheckpoint::open(dst.path())?;
    let plan = load_plan(&ckpt);
    assert_eq!(plan.len(), 3 + ckpt.config.layer_kinds.len());
    let total: u64 = plan.iter().map(|s| s.resident).sum();
    let mut before = 0;
    let mut peak = 0;
    for stage in &plan {
        peak = peak.max(before + stage.resident + stage.transient);
        before += stage.resident;
    }
    assert!(peak > total, "aucun pic transitoire : {peak} ≤ {total}");
    let simulate = |cap: u64| -> Result<()> {
        let current = Arc::new(AtomicU64::new(0));
        let probe = Arc::clone(&current);
        let guard = MemoryGuard::from_limits(
            true,
            MemoryLimits {
                static_cap: Some(cap),
                host_memory: None,
                headroom_bytes: 0,
                metal_cap: None,
            },
            Arc::new(move || Some(probe.load(Ordering::Relaxed))),
        );
        for stage in &plan {
            stage.check(Some(&guard))?;
            current.fetch_add(stage.resident, Ordering::Relaxed);
        }
        Ok(())
    };
    assert!(
        matches!(simulate(total), Err(InferError::MemoryGuard(_))),
        "plafond au résident final accepté malgré le pic"
    );
    simulate(peak)
}

/// Profil GPU par brique après un contexte de 1024 tokens (128 pas gloutons).
/// À lancer avec `SARAGOSSA_FLASH_NEXT_PROFILE=1` (temps par brique) et
/// `SARAGOSSA_RUST_DECODE_PROFILE=1` (dispatches par pas), machine libre.
#[cfg(feature = "devtools")]
#[test]
#[ignore = "profil : poids réels, machine libre"]
fn profile_resident_decode_after_1k_context() {
    use crate::qwen_loader::qwen4_exp::Qwen4ExpCheckpoint;
    let dir = std::path::PathBuf::from(
        std::env::var_os("SARAGOSSA_QWEN4EXP_DIR").expect("SARAGOSSA_QWEN4EXP_DIR requis"),
    );
    let golden =
        std::env::var_os("SARAGOSSA_QWEN4EXP_GOLDEN").expect("SARAGOSSA_QWEN4EXP_GOLDEN requis");
    let (context, generated) = (1024, 128);
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
    session
        .profiler_mut()
        .expect("SARAGOSSA_FLASH_NEXT_PROFILE=1 requis")
        .clear();
    let started = std::time::Instant::now();
    for _ in 0..generated {
        next = session
            .step(next, false, false)
            .expect("invariant: decode")
            .argmax as usize;
    }
    let wall = started.elapsed().as_secs_f64() * 1e3 / generated as f64;
    let (rows, dispatches) = session.profiler_mut().expect("invariant: profil").report();
    let gpu: f64 = rows.iter().map(|r| r.ms_per_step).sum();
    eprintln!("profil après {context} tokens, {generated} pas : {wall:.2} ms/pas mural (profilé), GPU {gpu:.2} ms, {dispatches:.0} dispatches/pas");
    for row in rows {
        eprintln!(
            "  {:24} {:7.3} ms/pas  {:5.1} %  {:5.1} encodeurs/pas",
            row.brick.label(),
            row.ms_per_step,
            100.0 * row.ms_per_step / gpu,
            row.encoders_per_step
        );
    }
}

/// Micro-bench des morceaux d'une hyper-connection aux formes réelles
/// (4 × 2560, rang 320) : chaque morceau encodé 200 fois dans un command
/// buffer, temps mural par appel. Diagnostic du profil, machine libre.
#[test]
#[ignore = "diagnostic de débit, machine libre"]
fn bench_hc_pieces_at_real_dims() -> Result<()> {
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
    let mut rng = DeterministicF32::new(7);
    let (down, _) = u8_weight(&executor, &mut rng, dims.lowrank, width, 0.05)?;
    let (up, _) = u8_weight(&executor, &mut rng, width, dims.lowrank, 0.2)?;
    let inject = buffer(&executor, &rng.fill(4 * width, -0.02, 0.02));
    let norm = buffer(&executor, &rng.fill(width, -0.3, 0.3));
    let h = buffer(&executor, &rng.fill(width, -2.0, 2.0));
    let scratch = HcScratch {
        normed: zeros(&executor, width),
        low: zeros(&executor, dims.lowrank),
        up: zeros(&executor, width),
        inject: zeros(&executor, 4 * super::hc::INJECT_CHUNKS),
        down_partial: zeros(&executor, dims.lowrank * super::hc::DOWN_CHUNKS),
    };
    let x = zeros(&executor, dims.hidden);
    let out = zeros(&executor, width);
    let queue = executor.device().new_command_queue();
    let reps = 200;
    let time = |label: &str,
                encode: &dyn Fn(&metal::ComputeCommandEncoderRef) -> Result<()>|
     -> Result<()> {
        for warm in [true, false] {
            let command_buffer = queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            for _ in 0..reps {
                encode(encoder)?;
            }
            encoder.end_encoding();
            let started = std::time::Instant::now();
            commit_and_wait(command_buffer)?;
            if !warm {
                eprintln!(
                    "{label:28} {:8.2} µs/appel",
                    started.elapsed().as_secs_f64() * 1e6 / reps as f64
                );
            }
        }
        Ok(())
    };
    time("norme groupée (1+w)", &|e| {
        super::ple::encode_rms_norm_1p(&kernels, e, &h, &norm, &scratch.normed, dims);
        Ok(())
    })?;
    time("Down u8 [320, 10240]", &|e| {
        executor
            .encode_matmul_weight_buffers(e, &scratch.normed, 1, width, &down, &scratch.low, true)
            .map(|_| ())
    })?;
    time("Up u8 [10240, 320]", &|e| {
        executor
            .encode_matmul_weight_buffers(e, &scratch.low, 1, dims.lowrank, &up, &scratch.up, true)
            .map(|_| ())
    })?;
    let weights = HcWeights {
        norm: norm.clone(),
        down: down.clone(),
        up: up.clone(),
        inject: Some(inject.clone()),
    };
    let no_inject = HcWeights {
        norm: norm.clone(),
        down: down.clone(),
        up: up.clone(),
        inject: None,
    };
    time("mix sans injection", &|e| {
        encode_hc_mix(&kernels, &executor, e, &no_inject, &scratch, dims, &h, &x)
    })?;
    time("mix complet (6 dispatches)", &|e| {
        encode_hc_mix(&kernels, &executor, e, &weights, &scratch, dims, &h, &x)
    })?;
    time("combine", &|e| {
        encode_hc_combine(&kernels, e, &scratch, dims, &h, &x, &out);
        Ok(())
    })?;
    Ok(())
}

/// Injection en tranches : la somme des partiels couvre toute la ligne, y
/// compris quand la largeur n'est pas divisible par le nombre de tranches.
#[test]
fn hc_inject_partials_cover_widths_not_divisible_by_chunks() -> Result<()> {
    use super::hc::INJECT_CHUNKS;
    let Some(executor) = executor()? else {
        return Ok(());
    };
    let kernels = FlashNextKernels::new(executor.device())?;
    let mut rng = DeterministicF32::new(0x1c);
    for width in [10_240_usize, 10_248, 1_000, 37] {
        let normed = rng.fill(width, -1.0, 1.0);
        let w = rng.fill(4 * width, -0.5, 0.5);
        let partial = zeros(&executor, 4 * INJECT_CHUNKS);
        let queue = executor.device().new_command_queue();
        let command_buffer = queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&kernels.hc_inject_partial);
        encoder.set_buffer(0, Some(&buffer(&executor, &normed)), 0);
        encoder.set_buffer(1, Some(&buffer(&executor, &w)), 0);
        encoder.set_buffer(2, Some(&partial), 0);
        super::set_value(encoder, 3, &[width as u32, INJECT_CHUNKS as u32]);
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(INJECT_CHUNKS as u64, 4, 1),
            metal::MTLSize::new(256, 1, 1),
        );
        encoder.end_encoding();
        commit_and_wait(command_buffer)?;
        let got = read_f32_buffer(&partial, 4 * INJECT_CHUNKS)?;
        for row in 0..4 {
            let sum: f32 = got[row * INJECT_CHUNKS..(row + 1) * INJECT_CHUNKS]
                .iter()
                .sum();
            let want: f64 = (0..width)
                .map(|d| f64::from(w[row * width + d]) * f64::from(normed[d]))
                .sum();
            assert!(
                (f64::from(sum) - want).abs() <= 1.0e-4 * want.abs().max(1.0),
                "largeur {width}, ligne {row} : {sum} contre {want}"
            );
        }
    }
    Ok(())
}
