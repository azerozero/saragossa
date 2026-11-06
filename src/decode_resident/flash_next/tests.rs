//! Oracles CPU des briques résidentes Flash-Next, aux dimensions réelles.

use metal::{Buffer, MTLResourceOptions};

use super::attention::{dense_max_visible, encode_qsa, QsaDecodeDims, QsaLayerWeights, QsaScratch};
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
    // Down dense f32 : repli sur le qmv standard (une seule tranche, SiLU en place).
    let down_dense_gpu = executor.resolve_linear_weight_buffers(
        &LinearWeight::Dense(down.clone()),
        "flash_next_down_dense",
    )?;
    for (with_inject, dense_down) in [(true, false), (false, false), (true, true)] {
        let cpu = GatedResidual {
            norm: norm.clone(),
            down: down.clone(),
            up: up.clone(),
            inject: with_inject.then(|| inject_w.clone()),
        };
        let mixed = cpu.mix(&h, dims.hc, dims.hidden, dims.eps)?;
        let gpu_weights = HcWeights {
            norm: buffer(&executor, &norm),
            down: if dense_down {
                down_dense_gpu.clone()
            } else {
                down_gpu.clone()
            },
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
            "entrée de branche (injection {with_inject}, Down dense {dense_down}) : {err} (max {max})"
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
    let kernels = FlashNextKernels::new(executor.device())?;
    let dims = QsaDecodeDims {
        hidden: 2560,
        heads: 24,
        kv_heads: 2,
        head_dim: 256,
        rotary_dim: 64,
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
        freqs: buffer(&executor, &crate::qwen4_exp::math::inv_freq(64, 1.0e7)),
        indexer: None,
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
        encode_qsa(
            &arena, &kernels, &executor, encoder, &mut kv, &weights, &scratch, dims, None, t,
            &x_buf, &out,
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
    let refused = encode_qsa(
        &arena,
        &kernels,
        &executor,
        encoder,
        &mut kv,
        &weights,
        &scratch,
        dims,
        None,
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

/// Seuils d'imposition : marge de probabilité du routage, marge relative QSA.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Ties {
    routing: f32,
    qsa: f32,
}

/// Mode d'une passe N2 résidente.
#[derive(Clone, Copy, Debug, PartialEq)]
enum RoutingMode {
    /// Routage et sélection propres du GPU.
    Native,
    /// Comme `Native`, plus un mauvais bloc QSA imposé à `(token, couche)`
    /// (puissance de l'oracle ré-ancré).
    NativeQsaMutation(usize, usize),
    /// Choix du CPU imposés aux points de marge ≤ seuils.
    Forced(Ties),
    /// Comme `Forced`, plus un mauvais expert imposé à `(token, couche)`.
    Mutation(usize, usize, Ties),
    /// Comme `Forced`, plus un mauvais bloc QSA imposé à `(token, couche)`.
    QsaMutation(usize, usize, Ties),
}

/// Sélection QSA GPU d'une couche : blocs retenus, scores, blocs natifs.
type GpuQsa = (Vec<usize>, Vec<f32>, Vec<usize>);

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
    /// Blocs QSA retenus par le GPU et scores de son indexeur `[couche]`
    /// (régime creux seulement).
    /// `(blocs retenus, scores, blocs natifs avant imposition)`.
    qsa: Vec<Option<GpuQsa>>,
    /// Points QSA imposés dont la sélection native n'est pas une quasi-égalité.
    qsa_native_unjustified: usize,
    /// Couches QSA dont la sélection a été imposée à ce token.
    qsa_imposed: Vec<usize>,
    /// Passes natives : `(couche QSA, entrée GPU, sortie GPU)`.
    anchors: Vec<(usize, Vec<f32>, Vec<f32>)>,
}

/// Blocs retenus au token `t` d'une trace QSA en régime creux.
fn trace_blocks(
    trace: &crate::qwen4_exp::QsaTrace,
    t: usize,
    keep: usize,
    compress: usize,
) -> Option<Vec<usize>> {
    (trace.scores[t].len() > keep).then(|| {
        trace.selected[t][..keep * compress]
            .chunks_exact(compress)
            .map(|c| c[0] / compress)
            .collect()
    })
}

/// Score du dernier bloc retenu (seuil du top-k QSA) et marge relative au
/// premier bloc écarté : `(s_keep − s_keep+1) / max(|s_keep|, 1)`.
fn row_threshold(row: &[f32], keep: usize) -> Option<(f32, f32)> {
    if row.len() <= keep {
        return None;
    }
    let mut sorted = row.to_vec();
    sorted.sort_by(|a, b| b.total_cmp(a));
    let threshold = sorted[keep - 1];
    Some((
        threshold,
        (threshold - sorted[keep]) / threshold.abs().max(1.0),
    ))
}

/// Vrai si `gpu` ne diffère de `want` que par des blocs dont le score `row`
/// est à `tie` (relatif à `max(|seuil|, 1)`) du seuil.
fn swap_justified(row: &[f32], want: &[usize], gpu: &[usize], keep: usize, tie: f32) -> bool {
    let Some((threshold, _)) = row_threshold(row, keep) else {
        return false;
    };
    // Sélection GPU bien formée : `keep` blocs distincts, triés, complets.
    // Sinon un doublon (`[0, 0]` contre `[0, 1]` à scores nuls) passerait.
    if gpu.len() != keep
        || gpu.windows(2).any(|w| w[0] >= w[1])
        || gpu.last().is_some_and(|&b| b >= row.len())
    {
        return false;
    }
    gpu.iter()
        .filter(|b| !want.contains(b))
        .chain(want.iter().filter(|b| !gpu.contains(b)))
        .all(|&b| b < row.len() && (row[b] - threshold).abs() <= tie * threshold.abs().max(1.0))
}

/// Écart max des scores GPU aux scores `want`, dans l'unité des marges QSA :
/// `max_b |g_b − w_b| / max(|seuil|, 1)` (NaN si longueurs ou valeurs
/// incohérentes).
fn score_error(gpu: &[f32], want: &[f32], keep: usize) -> f64 {
    let Some((threshold, _)) = row_threshold(want, keep) else {
        // CPU dense : le GPU ne doit pas avoir de scores creux.
        return if gpu.is_empty() { 0.0 } else { f64::NAN };
    };
    if gpu.len() != want.len() {
        return f64::NAN;
    }
    let scale = f64::from(threshold.abs().max(1.0));
    gpu.iter()
        .zip(want)
        .map(|(g, w)| f64::from((g - w).abs()) / scale)
        .fold(0.0, nan_max)
}

/// Blocs retenus par le CPU au token `t` d'une couche QSA en régime creux.
fn cpu_blocks(cpu: &CpuChain, layer: usize, t: usize) -> Option<Vec<usize>> {
    trace_blocks(cpu.qsa[layer].as_ref()?, t, cpu.keep, cpu.compress)
}

/// Seuil et marge relative QSA du CPU au token `t` ([`row_threshold`]).
fn qsa_threshold(cpu: &CpuChain, layer: usize, t: usize) -> Option<(f32, f32)> {
    row_threshold(&cpu.qsa[layer].as_ref()?.scores[t], cpu.keep)
}

/// [`swap_justified`] sur la chaîne CPU.
fn qsa_swap_justified(cpu: &CpuChain, layer: usize, t: usize, gpu: &[usize], tie: f32) -> bool {
    let (Some(want), Some(trace)) = (cpu_blocks(cpu, layer, t), cpu.qsa[layer].as_ref()) else {
        return false;
    };
    swap_justified(&trace.scores[t], &want, gpu, cpu.keep, tie)
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
    /// Trace QSA (tokens retenus, scores de blocs) des couches QSA.
    qsa: Vec<Option<crate::qwen4_exp::QsaTrace>>,
    /// Blocs retenus en régime creux (`budget / compress`).
    keep: usize,
    compress: usize,
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
        let ties = match mode {
            RoutingMode::Native | RoutingMode::NativeQsaMutation(..) => None,
            RoutingMode::Forced(ties)
            | RoutingMode::Mutation(_, _, ties)
            | RoutingMode::QsaMutation(_, _, ties) => Some(ties),
        };
        let (threshold, qsa_tie) = (ties.map(|t| t.routing), ties.map(|t| t.qsa));
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
        session.forced_selection = (0..layers)
            .map(|layer| {
                if matches!(
                    mode,
                    RoutingMode::QsaMutation(mt, ml, _) | RoutingMode::NativeQsaMutation(mt, ml)
                        if (mt, ml) == (t, layer)
                ) {
                    // Un bloc retenu remplacé par le premier bloc complet écarté.
                    let blocks = cpu_blocks(cpu, layer, t)?;
                    let outsider = (0..).find(|b| !blocks.contains(b))?;
                    let mut wrong: Vec<u32> = blocks.iter().map(|&b| b as u32).collect();
                    wrong[0] = outsider as u32;
                    wrong.sort_unstable();
                    return Some(wrong);
                }
                let th = qsa_tie?;
                let (_, margin) = qsa_threshold(cpu, layer, t)?;
                (margin <= th).then(|| {
                    cpu_blocks(cpu, layer, t)
                        .unwrap_or_default()
                        .into_iter()
                        .map(|b| b as u32)
                        .collect()
                })
            })
            .collect();
        let qsa_imposed: Vec<usize> = (0..layers)
            .filter(|&l| session.forced_selection[l].is_some())
            .collect();
        let qsa_imposed_now = qsa_imposed.clone();
        let out = session
            .step(token, true, true)
            .expect("invariant: pas résident");
        let got_layers = out.layers.expect("invariant: résiduels");
        // Passes natives : entrée (sortie GPU de la couche précédente) et
        // sortie de chaque couche QSA, pour l'oracle ré-ancré.
        // La couche 0 n'est pas QSA (vérifié par l'appelant) : son entrée
        // serait l'embedding. En mutation, seule la couche mutée est gardée.
        let anchors = match mode {
            RoutingMode::Native | RoutingMode::NativeQsaMutation(..) => (1..layers)
                .filter(|&l| cpu.qsa[l].is_some())
                .filter(|&l| !matches!(mode, RoutingMode::NativeQsaMutation(_, ml) if ml != l))
                .map(|l| (l, got_layers[l - 1].clone(), got_layers[l].clone()))
                .collect(),
            _ => Vec::new(),
        };
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
        let qsa = out
            .qsa
            .expect("invariant: QSA GPU")
            .into_iter()
            .map(|entry| {
                entry.map(|(blocks, scores, native)| {
                    let sorted = |v: Vec<u32>| {
                        let mut v: Vec<usize> = v.into_iter().map(|b| b as usize).collect();
                        v.sort_unstable();
                        v
                    };
                    (sorted(blocks), scores, sorted(native))
                })
            })
            .collect::<Vec<_>>();
        // Aux points imposés, la sélection native du GPU doit être une
        // quasi-égalité du CPU (sinon l'imposition masquerait un défaut).
        let qsa_native_unjustified = qsa
            .iter()
            .enumerate()
            .filter(|(l, entry)| {
                qsa_tie.is_some_and(|th| {
                    session.forced_selection.len() == layers
                        && entry.as_ref().is_some_and(|(_, _, native)| {
                            qsa_imposed_now.contains(l)
                                && !qsa_swap_justified(cpu, *l, t, native, th.max(1.0e-5))
                        })
                })
            })
            .count();
        reports.push(TokenReport {
            layer_errors,
            kl,
            swap,
            routing,
            imposed,
            qsa,
            qsa_imposed,
            qsa_native_unjustified,
            anchors,
        });
    }
    reports
}

/// Seuil de l'oracle ré-ancré : quasi-égalités (routage, sélection QSA) et
/// écart des scores de l'indexeur, dans l'unité des marges.
const ANCHORED_TIE: f32 = 1.0e-5;
/// Écart relatif admis sur la sortie d'une couche QSA ré-ancrée. La couche 0,
/// aux entrées identiques, s'écarte déjà de 1,1e-5 (N2 de l'étape 4) : 1e-5
/// serait sous le bruit des kernels, 1e-4 reste dix fois sous la porte de
/// bout en bout (1e-3). Une mutation numérique de 2e-4 doit la faire rougir.
const ANCHORED_OUTPUT_TOL: f64 = 1.0e-4;

/// Bilan de l'oracle ré-ancré sur une couche QSA.
#[derive(Debug)]
struct AnchoredLayer {
    layer: usize,
    /// Écart max des scores GPU/CPU ([`score_error`]).
    score_error: f64,
    /// Sélections différentes, dont non justifiées à [`ANCHORED_TIE`].
    selection_swaps: usize,
    selection_unjustified: usize,
    /// Routages différents, dont non justifiés (marge > seuil ou autre
    /// échange que la paire au seuil).
    routing_swaps: usize,
    routing_unjustified: usize,
    /// Écart relatif max de la sortie (NaN si non finie).
    output_error: f64,
}

impl AnchoredLayer {
    fn green(&self) -> bool {
        // Scores à la moitié du seuil : une erreur ε par score ne peut alors
        // échanger que des blocs à 2ε ≤ ANCHORED_TIE du seuil, que la
        // justification couvre.
        within(self.score_error, f64::from(ANCHORED_TIE) / 2.0)
            && self.selection_unjustified == 0
            && self.routing_unjustified == 0
            && within(self.output_error, ANCHORED_OUTPUT_TOL)
    }
}

/// Oracle local ré-ancré : chaque couche QSA CPU repart, sur toute la
/// séquence, des entrées GPU de la passe native (`anchors`), sélection et
/// routage GPU imposés. Les choix propres du CPU (trace et routage naturels)
/// justifient ceux du GPU ; la sortie compare les kernels seuls, sans dérive
/// accumulée.
fn anchored_qsa_layers(
    reference: &crate::qwen4_exp::Qwen4ExpReference,
    ids: &[usize],
    reports: &[TokenReport],
    keep: usize,
    compress: usize,
    only: Option<usize>,
) -> Vec<AnchoredLayer> {
    let tokens = reports.len();
    let Some(first) = reports.first() else {
        return Vec::new();
    };
    let layers: Vec<usize> = first
        .anchors
        .iter()
        .map(|a| a.0)
        .filter(|&l| only.is_none_or(|o| o == l))
        .collect();
    let mut summary = Vec::new();
    for &layer in &layers {
        let slot = first
            .anchors
            .iter()
            .position(|a| a.0 == layer)
            .expect("invariant: ancre de la couche");
        let width = first.anchors[slot].1.len();
        let input: Vec<f32> = reports
            .iter()
            .flat_map(|r| r.anchors[slot].1.iter().copied())
            .collect();
        let input = crate::Tensor::from_vec(vec![tokens, width], input)
            .expect("invariant: entrées ré-ancrées");
        // Sélection GPU effectivement utilisée : blocs retenus puis queue en
        // régime creux, tous les tokens visibles en régime dense.
        let selection: Vec<Vec<usize>> = reports
            .iter()
            .enumerate()
            .map(|(t, r)| match &r.qsa[layer] {
                Some((blocks, _, _)) => {
                    let complete = (t + 1) / compress;
                    blocks
                        .iter()
                        .flat_map(|&b| b * compress..(b + 1) * compress)
                        .chain(complete * compress..=t)
                        .collect()
                }
                None => (0..=t).collect(),
            })
            .collect();
        let routing: Vec<Option<Vec<usize>>> = reports
            .iter()
            .map(|r| Some(r.routing[layer].clone()))
            .collect();
        let out = reference
            .layer_forced(
                layer,
                &input,
                &ids[..tokens],
                Some(&selection),
                Some(&routing),
            )
            .expect("invariant: couche CPU ré-ancrée");
        let trace = out.qsa.as_ref().expect("invariant: trace QSA");
        let mut entry = AnchoredLayer {
            layer,
            score_error: 0.0,
            selection_swaps: 0,
            selection_unjustified: 0,
            routing_swaps: 0,
            routing_unjustified: 0,
            output_error: 0.0,
        };
        for (t, r) in reports.iter().enumerate() {
            let gpu_routing = &r.routing[layer];
            if *gpu_routing != out.routing[t] {
                entry.routing_swaps += 1;
                let (kth, next) = out.routing_boundary[t];
                let pair_only = gpu_routing
                    .iter()
                    .filter(|e| !out.routing[t].contains(e))
                    .eq([&next])
                    && out.routing[t]
                        .iter()
                        .filter(|e| !gpu_routing.contains(e))
                        .eq([&kth]);
                if !(pair_only && out.routing_margins[t] <= ANCHORED_TIE) {
                    entry.routing_unjustified += 1;
                }
            }
            if let Some((blocks, scores, _)) = &r.qsa[layer] {
                entry.score_error = nan_max(
                    entry.score_error,
                    score_error(scores, &trace.scores[t], keep),
                );
                let want = trace_blocks(trace, t, keep, compress).unwrap_or_default();
                if *blocks != want {
                    entry.selection_swaps += 1;
                    if !swap_justified(&trace.scores[t], &want, blocks, keep, ANCHORED_TIE) {
                        entry.selection_unjustified += 1;
                    }
                }
            } else if trace.scores[t].len() > keep {
                // Régime creux côté CPU sans capture GPU : incohérence.
                entry.selection_unjustified += 1;
            }
            let row = &out.hidden.data()[t * width..(t + 1) * width];
            entry.output_error = nan_max(
                entry.output_error,
                finite_relative_error(&r.anchors[slot].2, row),
            );
        }
        eprintln!("ré-ancrée, couche {} : {entry:?}", entry.layer);
        summary.push(entry);
    }
    summary
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
    /// Points QSA non imposés où le GPU retient d'autres blocs que le CPU.
    qsa_mismatches: usize,
    /// Points QSA imposés dont la sélection native n'est pas justifiée.
    qsa_native_unjustified: usize,
}

/// Points QSA non imposés où le GPU retient d'autres blocs que le CPU :
/// `(token, couche, marge CPU au seuil)`.
fn qsa_mismatch_points(reports: &[TokenReport], cpu: &CpuChain) -> Vec<(usize, usize, f32)> {
    reports
        .iter()
        .enumerate()
        .flat_map(|(t, r)| {
            (0..r.qsa.len())
                .filter(move |&l| {
                    !r.qsa_imposed.contains(&l)
                        && r.qsa[l].as_ref().map(|(b, _, _)| b) != cpu_blocks(cpu, l, t).as_ref()
                })
                .map(move |l| (t, l, qsa_threshold(cpu, l, t).map_or(f32::NAN, |(_, m)| m)))
        })
        .collect()
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
    let qsa_mismatches = reports
        .iter()
        .enumerate()
        .map(|(t, r)| {
            (0..r.qsa.len())
                .filter(|&l| {
                    !r.qsa_imposed.contains(&l)
                        && r.qsa[l].as_ref().map(|(b, _, _)| b) != cpu_blocks(cpu, l, t).as_ref()
                })
                .count()
        })
        .sum();
    let qsa_native_unjustified = reports.iter().map(|r| r.qsa_native_unjustified).sum();
    Summary {
        worst,
        kl,
        swaps,
        bad: bad.len(),
        first: bad.first().copied(),
        routing_mismatches,
        qsa_mismatches,
        qsa_native_unjustified,
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
        && summary.qsa_mismatches == 0
        && summary.qsa_native_unjustified == 0
}

/// Contrôle de la passe native : jusqu'à la première bascule de routage
/// `(t₀, l₀)`, écart ≤ 1e-3 (fini) et argmax identique ; la bascule tombe sur
/// une marge de probabilité ≤ [`NATIVE_FIRST_TIE`] et n'échange que la paire
/// au seuil (k-ième ↔ (k+1)-ième expert du CPU). Renvoie `(t₀, l₀, marge)`.
fn native_check(
    reports: &[TokenReport],
    cpu: &CpuChain,
) -> std::result::Result<Option<(usize, usize, f32)>, String> {
    // Première divergence : sélection QSA (calculée avant le MoE de la couche)
    // ou routage, dans l'ordre (token, couche).
    let qsa_diff = |t: usize, r: &TokenReport, l: usize| {
        r.qsa[l].as_ref().map(|(b, _, _)| b) != cpu_blocks(cpu, l, t).as_ref()
    };
    let first = reports.iter().enumerate().find_map(|(t, r)| {
        (0..r.routing.len())
            .find(|&l| qsa_diff(t, r, l) || r.routing[l] != cpu.routing[l][t])
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
    if qsa_diff(t, &reports[t], l) {
        let gpu = reports[t].qsa[l]
            .as_ref()
            .map(|(b, _, _)| b.clone())
            .unwrap_or_default();
        if !qsa_swap_justified(cpu, l, t, &gpu, NATIVE_FIRST_TIE) {
            return Err(format!(
                "première divergence QSA (token {t}, couche {l}) hors quasi-égalité"
            ));
        }
        let (_, margin) = qsa_threshold(cpu, l, t).unwrap_or((0.0, f32::NAN));
        return Ok(Some((t, l, margin)));
    }
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
        qsa: vec![None],
        keep: 512,
        compress: 4,
        width: 1,
        vocab: 1,
    };
    let report = |err: f64, kl: f64| TokenReport {
        layer_errors: vec![err],
        kl,
        swap: None,
        routing: vec![vec![1, 2]],
        imposed: Vec::new(),
        qsa: vec![None],
        qsa_imposed: Vec::new(),
        qsa_native_unjustified: 0,
        anchors: Vec::new(),
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
        .unwrap_or(4096);
    let ids: Vec<usize> = n2_input_ids(std::path::Path::new(&golden))[..tokens].to_vec();

    let started = std::time::Instant::now();
    let reference =
        Qwen4ExpReference::new(Qwen4ExpCheckpoint::open(&dir).expect("invariant: checkpoint"));
    let config = &reference.checkpoint().config;
    let mut h = reference.embed(&ids).expect("invariant: embedding");
    let (mut layers, mut routing, mut margins, mut boundary, mut qsa) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for index in 0..config.layer_kinds.len() {
        let out = reference
            .layer(index, &h, &ids)
            .expect("invariant: couche CPU");
        h = out.hidden;
        layers.push(h.clone());
        routing.push(out.routing);
        margins.push(out.routing_margins);
        boundary.push(out.routing_boundary);
        qsa.push(out.qsa);
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
        qsa,
        keep: config.indexer_budget / config.indexer_compress,
        compress: config.indexer_compress,
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
    assert!(cpu.qsa[0].is_none(), "couche 0 QSA : entrée non ancrable");
    let native = run_resident_pass(&dir, &ids, &cpu, RoutingMode::Native, tokens);
    let native_first = native_check(&native, &cpu);
    // Erreur des scores GPU de l'indexeur avant la première divergence (calibre
    // la règle de quasi-égalité QSA) et points QSA par seuil.
    let horizon = native_first
        .as_ref()
        .ok()
        .and_then(|f| f.map(|(t, _, _)| t))
        .unwrap_or(tokens);
    let mut early_error = 0.0_f64;
    for (t, r) in native[..horizon].iter().enumerate() {
        for (l, entry) in r.qsa.iter().enumerate() {
            if let (Some((_, gpu, _)), Some(trace)) = (entry, cpu.qsa[l].as_ref()) {
                early_error = nan_max(early_error, relative_error(gpu, &trace.scores[t]).0);
            }
        }
    }
    eprintln!("scores de l'indexeur GPU/CPU avant la première divergence : écart relatif max {early_error:.2e}");
    for th in TIE_THRESHOLDS {
        let n = (0..tokens)
            .flat_map(|t| (0..cpu.layers.len()).map(move |l| (t, l)))
            .filter(|&(t, l)| qsa_threshold(&cpu, l, t).is_some_and(|(_, m)| m <= th))
            .count();
        eprintln!("points QSA de marge relative ≤ {th:.0e} : {n}");
    }
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
    // Oracle local ré-ancré (porte stricte) : chaque couche QSA CPU repart des
    // entrées GPU de la passe native, sans dérive accumulée.
    let anchored = anchored_qsa_layers(&reference, &ids, &native, cpu.keep, cpu.compress, None);
    let anchored_green = !anchored.is_empty() && anchored.iter().all(AnchoredLayer::green);
    drop(native);

    // Contrôle de dérive de bout en bout. Le seuil QSA est fixé avant la
    // validation : 4 × l'écart max des scores GPU/CPU sur la première moitié
    // du régime creux (passe de calibration, routage et QSA imposés au seuil
    // le plus large), puis validé sur toute la séquence sans réglage.
    let widest = TIE_THRESHOLDS[TIE_THRESHOLDS.len() - 1];
    let sparse_start = dense_max_visible(cpu.keep * cpu.compress, cpu.compress).min(tokens);
    let mid = sparse_start + (tokens - sparse_start) / 2;
    let calibration = run_resident_pass(
        &dir,
        &ids,
        &cpu,
        RoutingMode::Forced(Ties {
            routing: widest,
            qsa: widest,
        }),
        mid,
    );
    let mut calibration_error = 0.0_f64;
    for (t, r) in calibration.iter().enumerate().skip(sparse_start) {
        for (l, entry) in r.qsa.iter().enumerate() {
            if let (Some((_, gpu, _)), Some(trace)) = (entry, cpu.qsa[l].as_ref()) {
                calibration_error = nan_max(
                    calibration_error,
                    score_error(gpu, &trace.scores[t], cpu.keep),
                );
            }
        }
    }
    drop(calibration);
    let qsa_tie = (4.0 * calibration_error) as f32;
    eprintln!(
        "calibration QSA (tokens {sparse_start}..{mid}) : écart max des scores {calibration_error:.2e}, \
         seuil QSA {qsa_tie:.2e}"
    );
    assert!(qsa_tie.is_finite(), "calibration QSA non finie");

    let mut chosen = None;
    for th in TIE_THRESHOLDS {
        let ties = Ties {
            routing: th,
            qsa: qsa_tie,
        };
        let forced = run_resident_pass(&dir, &ids, &cpu, RoutingMode::Forced(ties), tokens);
        let mut green = true;
        for &prefix in &prefixes {
            let s = summarize(&forced, &cpu, prefix);
            eprintln!(
                "imposée ≤ {th:.0e}, {prefix:4} : écart max {:.2e}, KL max {:.2e}, argmax ≠ {:?}, \
                 tokens > 1e-3 : {}, premier {:?}, routage ≠ hors imposition : {}, \
                 sélection QSA ≠ hors imposition : {}, natives injustifiées : {}",
                s.worst,
                s.kl,
                s.swaps,
                s.bad,
                s.first,
                s.routing_mismatches,
                s.qsa_mismatches,
                s.qsa_native_unjustified
            );
            green &= forced_gate_green(&s);
        }
        // Scores de l'indexeur GPU contre CPU sur toute la passe imposée (régime
        // creux compris ; les entrées suivent le CPU à la dérive près).
        let mut forced_score_error = 0.0_f64;
        let mut qsa_imposed_points = 0;
        for (t, r) in forced.iter().enumerate() {
            qsa_imposed_points += r.qsa_imposed.len();
            for (l, entry) in r.qsa.iter().enumerate() {
                if let (Some((_, gpu, _)), Some(trace)) = (entry, cpu.qsa[l].as_ref()) {
                    forced_score_error =
                        nan_max(forced_score_error, relative_error(gpu, &trace.scores[t]).0);
                }
            }
        }
        let sparse_points = forced
            .iter()
            .map(|r| r.qsa.iter().filter(|e| e.is_some()).count())
            .sum::<usize>()
            .max(1);
        eprintln!(
            "imposée ≤ {th:.0e} : scores de l'indexeur GPU/CPU écart relatif max \
             {forced_score_error:.2e}, {qsa_imposed_points} sélections QSA imposées sur \
             {sparse_points} points creux ({:.2} %)",
            100.0 * qsa_imposed_points as f64 / sparse_points as f64
        );
        green &= within(forced_score_error, 1.0e-3);
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
        let mismatches = qsa_mismatch_points(&forced, &cpu);
        eprintln!(
            "imposée ≤ {th:.0e} : sélections QSA ≠ hors imposition (token, couche, marge) {:?}",
            &mismatches[..mismatches.len().min(40)]
        );
        if green {
            chosen = Some(th);
            break;
        }
    }
    let ties = Ties {
        routing: chosen.unwrap_or(widest),
        qsa: qsa_tie,
    };
    eprintln!("seuil de routage retenu : {chosen:?}");

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
            RoutingMode::Mutation(token, layer, ties),
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
    // Mutations QSA : un mauvais bloc imposé au point de plus grande marge QSA
    // relative, au début du régime creux puis sur les 128 derniers tokens.
    let late = tokens.saturating_sub(128).max(sparse_start + 64);
    for range in [sparse_start..tokens.min(sparse_start + 64), late..tokens] {
        if range.is_empty() {
            continue;
        }
        let (token, layer) = range
            .flat_map(|t| (0..cpu.layers.len()).map(move |l| (t, l)))
            .filter_map(|(t, l)| qsa_threshold(&cpu, l, t).map(|(_, m)| (t, l, m)))
            .max_by(|a, b| a.2.total_cmp(&b.2))
            .map(|(t, l, _)| (t, l))
            .expect("invariant: une couche QSA en régime creux");
        let steps = (token + 8).min(tokens);
        let mutated = run_resident_pass(
            &dir,
            &ids,
            &cpu,
            RoutingMode::QsaMutation(token, layer, ties),
            steps,
        );
        let s = summarize(&mutated, &cpu, steps);
        eprintln!(
            "mutation QSA (token {token}, couche {layer}, marge {:.2e}) : écart max {:.2e}, KL max {:.2e}, premier {:?}",
            qsa_threshold(&cpu, layer, token).map_or(f32::NAN, |(_, m)| m),
            s.worst,
            s.kl,
            s.first
        );
        assert!(
            !forced_gate_green(&s),
            "mutation QSA au token {token} non détectée"
        );
    }
    // Puissance de l'oracle ré-ancré : un mauvais bloc imposé en passe native,
    // au dernier token et sur la couche QSA de plus grande marge, doit le
    // faire rougir.
    if tokens > sparse_start {
        let token = tokens - 1;
        let layer = (0..cpu.layers.len())
            .filter_map(|l| qsa_threshold(&cpu, l, token).map(|(_, m)| (l, m)))
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(l, _)| l)
            .expect("invariant: une couche QSA en régime creux");
        let mut mutated = run_resident_pass(
            &dir,
            &ids,
            &cpu,
            RoutingMode::NativeQsaMutation(token, layer),
            tokens,
        );
        let check = anchored_qsa_layers(
            &reference,
            &ids,
            &mutated,
            cpu.keep,
            cpu.compress,
            Some(layer),
        );
        // Mutation numérique : la sortie GPU du token précédent, sur cette
        // couche, est mise à l'échelle de 1 + 2e-4 (sélections intactes avant
        // `token`) ; seule la porte de sortie doit rougir.
        let mut scaled: Vec<TokenReport> = mutated.drain(..token).collect();
        for v in &mut scaled[token - 1].anchors[0].2 {
            *v *= 1.0 + 2.0e-4;
        }
        let numeric = anchored_qsa_layers(
            &reference,
            &ids,
            &scaled,
            cpu.keep,
            cpu.compress,
            Some(layer),
        );
        assert!(
            numeric
                .iter()
                .any(|c| !within(c.output_error, ANCHORED_OUTPUT_TOL)
                    && c.selection_unjustified == 0
                    && c.routing_unjustified == 0),
            "mutation numérique non détectée par l'oracle ré-ancré : {numeric:?}"
        );
        // Rouge par la branche sélection, le reste vert : c'est bien la
        // mutation qui est vue, pas une panne du harnais.
        assert!(
            check.iter().any(|c| c.selection_unjustified > 0
                && within(c.score_error, f64::from(ANCHORED_TIE) / 2.0)
                && within(c.output_error, ANCHORED_OUTPUT_TOL)),
            "mutation QSA (token {token}, couche {layer}) non détectée par l'oracle ré-ancré : {check:?}"
        );
    }
    assert!(anchored_green, "oracle ré-ancré rouge : {anchored:?}");
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
    (base.head_dim, base.rotary_dim, base.indexer_head_dim) = (256, 64, 128);
    base.top_k = base.top_k.min(base.experts);
    assert_eq!(validate_resident_config(&base)?, 2051);
    type Edit = fn(&mut Qwen4ExpConfig);
    let variants: [(&str, Edit); 8] = [
        ("norm_topk_prob", |c| c.norm_topk_prob = false),
        ("gate SiLU", |c| c.sigmoid_output_gate = false),
        ("d_k 64", |c| c.linear_key_dim = 64),
        ("têtes V", |c| c.linear_value_heads = 47),
        ("top-k 17", |c| (c.top_k, c.experts) = (17, 512)),
        ("budget", |c| c.indexer_budget = 2047),
        ("attention d 320", |c| c.head_dim = 320),
        ("indexeur d 48", |c| c.indexer_head_dim = 48),
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
    // Rejeu sans tête puis point de reprise : après la reprise, les mêmes
    // tokens redonnent les mêmes logits au bit près ; un point d'avant un reset
    // est refusé.
    session.reset().expect("invariant: reset");
    session.replay(ids[0]).expect("invariant: rejeu");
    let point = session
        .checkpoint(None)
        .expect("invariant: point de reprise");
    let after: Vec<Vec<f32>> = (1..4)
        .map(|t| {
            session
                .step(ids[t], true, false)
                .expect("invariant: pas")
                .logits
                .expect("invariant: logits")
        })
        .collect();
    assert!(
        after[0] == expected[1],
        "rejeu sans tête : logits du token 1"
    );
    session.restore(&point).expect("invariant: reprise");
    for (i, t) in (1..4).enumerate() {
        let again = session
            .step(ids[t], true, false)
            .expect("invariant: pas repris")
            .logits
            .expect("invariant: logits");
        assert!(again == after[i], "token {t} après reprise");
    }
    // Reprise à une position non alignée (1) après avoir franchi un bloc :
    // les clés brutes du bloc 0 sont restaurées, la clé poolée recalculée
    // est identique (couche QSA 3, sans effet sur les logits en régime dense).
    let qsa_layer = 3;
    session.restore(&point).expect("invariant: reprise");
    for &token in &ids[1..6] {
        session.step(token, false, false).expect("invariant: pas");
    }
    let pooled_first = session
        .indexer_probe(qsa_layer, 1)
        .expect("invariant: sonde")
        .pooled;
    session
        .restore(&point)
        .expect("invariant: reprise après un bloc");
    for &token in &ids[1..4] {
        session.step(token, false, false).expect("invariant: pas");
    }
    let pooled_again = session
        .indexer_probe(qsa_layer, 1)
        .expect("invariant: sonde")
        .pooled;
    assert!(
        pooled_again == pooled_first,
        "clé poolée du bloc 0 après reprise non alignée"
    );
    // Filiation : un point pris sur une branche abandonnée est refusé ; la base
    // de la branche courante reste réutilisable.
    let later = session
        .checkpoint(None)
        .expect("invariant: point ultérieur");
    session
        .restore(&point)
        .expect("invariant: reprise à la base");
    session
        .step(ids[5], false, false)
        .expect("invariant: autre branche");
    assert!(
        session.restore(&later).is_err(),
        "point d'une branche abandonnée accepté"
    );
    session
        .restore(&point)
        .expect("invariant: base réutilisable");
    session.reset().expect("invariant: reset");
    assert!(
        session.restore(&point).is_err(),
        "point d'avant un reset accepté"
    );
    for (t, &token) in ids.iter().enumerate().take(4) {
        let got = session.step(token, true, false).expect("invariant: pas");
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
    time("Down u8 qmv standard (repli)", &|e| {
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

/// Top-512 de l'indexeur (radix exact) : à scores identiques, mêmes blocs que
/// `select_for_query`, égalités comprises — aléatoire, tous nuls, −0,0 et +0,0
/// mêlés, égalités massives au seuil, 511 / 512 / 513 / 8192 / 65 536 blocs.
#[test]
fn flash_next_index_topk_matches_cpu_selection() -> Result<()> {
    use crate::qwen4_exp::attention::{select_for_query, QsaDims};
    let Some(executor) = executor()? else {
        return Ok(());
    };
    let kernels = FlashNextKernels::new(executor.device())?;
    let d = QsaDims {
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
    let keep = 512;
    let mut rng = DeterministicF32::new(0x70_9512);
    let mut cases: Vec<Vec<f32>> = Vec::new();
    for count in [513_usize, 8192, 65_536] {
        cases.push(rng.fill(count, 0.0, 3.0));
    }
    cases.push(vec![0.0; 2000]);
    cases.push(
        (0..2000)
            .map(|i| if i % 3 == 0 { -0.0 } else { 0.0 })
            .collect(),
    );
    let mut ties = rng.fill(4000, 0.0, 1.0);
    for i in (0..4000).step_by(5) {
        ties[i] = 0.75;
    }
    cases.push(ties);
    let mut quantized: Vec<f32> = rng
        .fill(6000, 0.0, 1.0)
        .iter()
        .map(|v| (v * 8.0).floor() / 8.0)
        .collect();
    quantized[17] = 0.0;
    cases.push(quantized);
    let queue = executor.device().new_command_queue();
    for (case, scores) in cases.iter().enumerate() {
        let count = scores.len();
        let selected = executor.device().new_buffer(
            (keep * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let command_buffer = queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&kernels.index_topk);
        encoder.set_buffer(0, Some(&buffer(&executor, scores)), 0);
        encoder.set_buffer(1, Some(&selected), 0);
        super::set_value(encoder, 2, &[count as u32, keep as u32]);
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(1, 1, 1),
            metal::MTLSize::new(1024, 1, 1),
        );
        encoder.end_encoding();
        commit_and_wait(command_buffer)?;
        let got: Vec<usize> = crate::metal_backend::read_u32_buffer(&selected, keep)?
            .into_iter()
            .map(|b| b as usize)
            .collect();
        // Blocs retenus par le CPU (tokens triés → blocs), avec `+ 0.0` comme `query_block_scores`.
        let normalized: Vec<f32> = scores.iter().map(|s| s + 0.0).collect();
        let tokens = select_for_query(&normalized, count * 4, d);
        let want: Vec<usize> = tokens.chunks_exact(4).map(|c| c[0] / 4).collect();
        assert_eq!(got, want, "cas {case} ({count} blocs)");
    }
    Ok(())
}

/// Couche QSA résidente en régime creux, pas à pas contre `qsa` CPU sur 200
/// tokens avec un budget réduit (16 blocs de 4 : creux dès 68 tokens visibles)
/// et un indexeur aux formes réelles (4 têtes × 128, RoPE 64). Sorties à 1e-5 ;
/// une sélection GPU différente n'est admise que si chaque bloc échangé a un
/// score CPU à 1e-5 près du score du dernier bloc retenu (quasi-égalité).
#[test]
fn flash_next_qsa_sparse_steps_match_cpu_sequence() -> Result<()> {
    use super::attention::{IndexerDims, IndexerScratch, IndexerState, IndexerWeights, QsaSparse};
    use crate::decode_resident::{DecodeResidentState, KvFormat};
    use crate::qwen4_exp::attention::{
        block_scores, qsa, select_from_scores, QsaDims, QsaWeights as CpuQsa,
    };
    use crate::qwen4_exp::math::inv_freq;
    let Some(executor) = executor()? else {
        return Ok(());
    };
    let arena = DecodeResidentState::new(executor.device().clone())?;
    let kernels = FlashNextKernels::new(executor.device())?;
    let (budget, compress, tokens) = (64_usize, 4_usize, oracle_steps(200));
    let dims = QsaDecodeDims {
        hidden: 512,
        heads: 4,
        kv_heads: 2,
        head_dim: 128,
        rotary_dim: 64,
        eps: 1.0e-6,
        dense_max: dense_max_visible(budget, compress),
    };
    let (q_dim, kv_dim, ihd, iheads) = (4 * 128, 2 * 128, 128, 4);
    let mut rng = DeterministicF32::new(0x5a_9e5e);
    let (q_gpu, q_w) = u8_weight(&executor, &mut rng, 2 * q_dim, dims.hidden, 0.06)?;
    let (k_gpu, k_w) = u8_weight(&executor, &mut rng, kv_dim, dims.hidden, 0.06)?;
    let (v_gpu, v_w) = u8_weight(&executor, &mut rng, kv_dim, dims.hidden, 0.06)?;
    let (o_gpu, o_w) = u8_weight(&executor, &mut rng, dims.hidden, q_dim, 0.04)?;
    let (iqk_gpu, iqk_w) = u8_weight(&executor, &mut rng, (iheads + 1) * ihd, dims.hidden, 0.06)?;
    let norms: Vec<Vec<f32>> = (0..4).map(|_| rng.fill(128, -0.3, 0.3)).collect();
    let x = rng.fill(tokens * dims.hidden, -1.0, 1.0);
    let cpu_w = CpuQsa {
        q_proj: q_w,
        k_proj: k_w,
        v_proj: v_w,
        o_proj: o_w,
        q_norm: norms[0].clone(),
        k_norm: norms[1].clone(),
        index_qk: iqk_w,
        index_q_norm: norms[2].clone(),
        index_k_norm: norms[3].clone(),
    };
    let cpu_dims = QsaDims {
        heads: 4,
        kv_heads: 2,
        head_dim: 128,
        rotary_dim: 64,
        rope_theta: 1.0e7,
        index_heads: iheads,
        index_head_dim: ihd,
        budget,
        compress,
        eps: 1.0e-6,
    };
    let input = Tensor::from_vec(vec![tokens, dims.hidden], x.clone())?;
    let expected = qsa(&input, &cpu_w, cpu_dims, None)?.output;
    let freqs = inv_freq(64, 1.0e7);
    let cpu_scores = block_scores(&input, &cpu_w, cpu_dims, &freqs)?;
    let cpu_selected = select_from_scores(&cpu_scores, cpu_dims);

    let one_plus = |w: &[f32]| -> Vec<f32> { w.iter().map(|v| 1.0 + v).collect() };
    let weights = QsaLayerWeights {
        q_proj: q_gpu,
        k_proj: k_gpu,
        v_proj: v_gpu,
        o_proj: o_gpu,
        q_norm: buffer(&executor, &one_plus(&norms[0])),
        k_norm: buffer(&executor, &one_plus(&norms[1])),
        freqs: buffer(&executor, &freqs),
        indexer: Some(IndexerWeights {
            qk_proj: iqk_gpu,
            q_norm: buffer(&executor, &one_plus(&norms[2])),
            k_norm: buffer(&executor, &one_plus(&norms[3])),
        }),
    };
    let scratch = QsaScratch {
        q_gate: zeros(&executor, 2 * q_dim),
        q_raw: zeros(&executor, q_dim),
        gate: zeros(&executor, q_dim),
        q_roped: zeros(&executor, q_dim),
        k_raw: zeros(&executor, kv_dim),
        k_roped: zeros(&executor, kv_dim),
        v: zeros(&executor, kv_dim),
        scores: zeros(&executor, 4 * tokens),
        ctx: zeros(&executor, q_dim),
        gated: zeros(&executor, q_dim),
    };
    let state = IndexerState {
        pooled: zeros(&executor, (tokens / compress + 1) * ihd),
        raw: zeros(&executor, compress * ihd),
    };
    let keep = budget / compress;
    let indexer_scratch = IndexerScratch {
        qk: zeros(&executor, (iheads + 1) * ihd),
        q: zeros(&executor, iheads * ihd),
        scores: zeros(&executor, tokens / compress + 1),
        selected: executor.device().new_buffer(
            (keep * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ),
    };
    let mut kv = arena.full_attention_with_format(tokens, 4, 2, 128, KvFormat::F32, None)?;
    let mut gathered =
        arena.full_attention_with_format(dims.dense_max, 4, 2, 128, KvFormat::F32, None)?;
    let queue = executor.device().new_command_queue();
    let (mut sparse_steps, mut justified) = (0, 0);
    for t in 0..tokens {
        let x_buf = buffer(&executor, &x[t * dims.hidden..(t + 1) * dims.hidden]);
        let out = zeros(&executor, dims.hidden);
        let command_buffer = queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encode_qsa(
            &arena,
            &kernels,
            &executor,
            encoder,
            &mut kv,
            &weights,
            &scratch,
            dims,
            Some(QsaSparse {
                weights: weights.indexer.as_ref().expect("invariant: indexeur"),
                state: &state,
                scratch: &indexer_scratch,
                gathered: &mut gathered,
                dims: IndexerDims {
                    heads: iheads,
                    head_dim: ihd,
                    compress,
                    keep,
                },
                forced: None,
                native_capture: None,
            }),
            t,
            &x_buf,
            &out,
        )?;
        encoder.end_encoding();
        commit_and_wait(command_buffer)?;
        let complete = (t + 1) / compress;
        if complete > keep {
            sparse_steps += 1;
            let got: Vec<usize> =
                crate::metal_backend::read_u32_buffer(&indexer_scratch.selected, keep)?
                    .into_iter()
                    .map(|b| b as usize)
                    .collect();
            let want: Vec<usize> = cpu_selected[t][..keep * compress]
                .chunks_exact(compress)
                .map(|c| c[0] / compress)
                .collect();
            // Scores GPU de l'indexeur contre le CPU (mêmes entrées à 1e-5 près).
            let gpu_scores = read_f32_buffer(&indexer_scratch.scores, complete)?;
            let (score_err, _) = relative_error(&gpu_scores, &cpu_scores[t]);
            assert!(score_err <= 1.0e-5, "scores pas {t} : {score_err}");
            if got != want {
                let row = &cpu_scores[t];
                let threshold = want.iter().map(|&b| row[b]).fold(f32::INFINITY, f32::min);
                for b in got
                    .iter()
                    .filter(|b| !want.contains(b))
                    .chain(want.iter().filter(|b| !got.contains(b)))
                {
                    assert!(
                        (row[*b] - threshold).abs() <= 1.0e-5 * threshold.abs().max(1.0),
                        "pas {t} : bloc {b} échangé hors quasi-égalité ({} contre {threshold})",
                        row[*b]
                    );
                }
                justified += 1;
                continue;
            }
        }
        let got = read_f32_buffer(&out, dims.hidden)?;
        let want = &expected.data()[t * dims.hidden..(t + 1) * dims.hidden];
        let (err, max) = relative_error(&got, want);
        assert!(err <= 1.0e-5, "QSA pas {t} : {err} (max {max})");
    }
    eprintln!(
        "régime creux : {sparse_steps} pas, {justified} sélections justifiées par quasi-égalité"
    );
    assert!(
        sparse_steps > 100,
        "régime creux trop court : {sparse_steps}"
    );
    assert!(
        justified * 10 <= sparse_steps,
        "trop de quasi-égalités : {justified}"
    );
    Ok(())
}

/// Débit AR résident après 32 768 tokens de contexte (corpus figé
/// `SARAGOSSA_FLASH_NEXT_CORPUS` : `ids.u32` et `provenance.json`), QSA
/// creuse : un seul rejeu du contexte (sans lm_head), point de reprise, puis
/// `SARAGOSSA_FLASH_NEXT_BENCH_RUNS` répétitions (3) de 256 pas gloutons
/// restaurées depuis ce point. Pendant le rejeu, tous les 4096 tokens, la
/// dernière couche QSA est sondée : scores de l'indexeur recalculés sur CPU
/// depuis ses requêtes et clés poolées GPU (1e-5), sélection identique à
/// `select_for_query` sur ces scores, et clés poolées du bloc qui vient de se
/// compléter recalculées depuis ses clés brutes (1e-5). Porte : ≥ 40 tok/s sur
/// la plus lente (256 / durée totale). Machine libre.
#[test]
#[ignore = "débit : poids réels, machine libre"]
fn bench_resident_decode_after_32k_context() {
    use crate::qwen4_exp::attention::{
        pooled_block_keys, query_block_scores, select_for_query, QsaDims,
    };
    use crate::qwen4_exp::math::inv_freq;
    use crate::qwen_loader::qwen4_exp::{Qwen4ExpCheckpoint, Qwen4ExpLayerKind};
    let dir = std::path::PathBuf::from(
        std::env::var_os("SARAGOSSA_QWEN4EXP_DIR").expect("SARAGOSSA_QWEN4EXP_DIR requis"),
    );
    let corpus = std::path::PathBuf::from(
        std::env::var_os("SARAGOSSA_FLASH_NEXT_CORPUS")
            .expect("SARAGOSSA_FLASH_NEXT_CORPUS requis"),
    );
    let runs: usize = crate::runtime_flags::env_var("SARAGOSSA_FLASH_NEXT_BENCH_RUNS")
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    assert!(
        runs > 0,
        "SARAGOSSA_FLASH_NEXT_BENCH_RUNS doit valoir au moins 1"
    );
    let ids: Vec<usize> = std::fs::read(corpus.join("ids.u32"))
        .expect("invariant: corpus")
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().expect("invariant: 4 octets")) as usize)
        .collect();
    let (context, generated) = (32_768, 256);
    assert!(ids.len() >= context, "corpus de {} tokens", ids.len());
    let ckpt = Qwen4ExpCheckpoint::open(&dir).expect("invariant: checkpoint");
    let c = ckpt.config.clone();
    let probe_layer = c
        .layer_kinds
        .iter()
        .rposition(|k| *k == Qwen4ExpLayerKind::Full)
        .expect("invariant: une couche QSA");
    let d = QsaDims {
        heads: c.heads,
        kv_heads: c.kv_heads,
        head_dim: c.head_dim,
        rotary_dim: c.rotary_dim,
        rope_theta: c.rope_theta,
        index_heads: c.indexer_heads,
        index_head_dim: c.indexer_head_dim,
        budget: c.indexer_budget,
        compress: c.indexer_compress,
        eps: c.rms_eps,
    };
    let freqs = inv_freq(c.rotary_dim, c.rope_theta);
    let k_norm = ckpt
        .floats(&format!(
            "{}self_attn.indexer.k_layernorm.weight",
            Qwen4ExpCheckpoint::layer_prefix(probe_layer)
        ))
        .expect("invariant: k_layernorm");
    let q_norm = ckpt
        .floats(&format!(
            "{}self_attn.indexer.q_layernorm.weight",
            Qwen4ExpCheckpoint::layer_prefix(probe_layer)
        ))
        .expect("invariant: q_layernorm");
    let mut session = FlashNextSession::open(
        ckpt,
        context + generated,
        Some(&crate::MemoryGuard::process()),
    )
    .expect("invariant: session");
    let started = std::time::Instant::now();
    let mut probes = 0;
    let mut worst_q = 0.0_f64;
    for (t, &token) in ids[..context - 1].iter().enumerate() {
        let probe = t > 2051 && (t + 1) % 4096 == 0;
        if !probe {
            session.replay(token).expect("invariant: rejeu");
            continue;
        }
        // Pas sondé : t ≡ 3 (mod 4), le bloc t/4 vient de se compléter.
        let out = session
            .step(token, false, true)
            .expect("invariant: pas sondé");
        let (blocks, scores, _) = out.qsa.expect("invariant: QSA")[probe_layer]
            .clone()
            .expect("invariant: régime creux");
        let complete = (t + 1) / d.compress;
        let probe_data = session
            .indexer_probe(probe_layer, complete)
            .expect("invariant: sonde");
        let (q, pooled, raw) = (&probe_data.q, &probe_data.pooled, &probe_data.raw);
        // Requêtes de l'indexeur : q_layernorm (1+w) puis RoPE à la position t, sur CPU.
        let ihd = d.index_head_dim;
        let mut cpu_q = crate::qwen4_exp::math::rms_norm_1p(
            &probe_data.qk[..d.index_heads * ihd],
            ihd,
            ihd,
            &q_norm,
            d.eps,
        );
        for head in cpu_q.chunks_exact_mut(ihd) {
            crate::qwen4_exp::math::rope_in_place(head, &freqs, t);
        }
        let (q_err, _) = relative_error(q, &cpu_q);
        worst_q = worst_q.max(q_err);
        assert!(
            q_err <= 1.0e-5,
            "token {t} : requêtes de l'indexeur (RoPE) {q_err}"
        );
        let cpu_scores = query_block_scores(q, pooled, complete, d);
        let (err, _) = relative_error(&scores, &cpu_scores);
        assert!(err <= 1.0e-5, "token {t} : scores de l'indexeur {err}");
        let tokens = select_for_query(&scores, t + 1, d);
        let mut want: Vec<u32> = tokens[..d.budget]
            .chunks_exact(d.compress)
            .map(|c| (c[0] / d.compress) as u32)
            .collect();
        want.sort_unstable();
        let mut got = blocks.clone();
        got.sort_unstable();
        assert_eq!(got, want, "token {t} : sélection");
        // Bloc complété : moyenne, k_layernorm et RoPE à la position 4·b recalculées.
        let block = complete - 1;
        let mut raw_seq = vec![0.0_f32; complete * d.compress * d.index_head_dim];
        raw_seq[block * d.compress * d.index_head_dim..].copy_from_slice(raw);
        let cpu_pooled = pooled_block_keys(&raw_seq, complete, &k_norm, d, &freqs);
        let range = block * d.index_head_dim..complete * d.index_head_dim;
        let (err, _) = relative_error(&pooled[range.clone()], &cpu_pooled[range]);
        assert!(
            err <= 1.0e-5,
            "token {t} : clé poolée du bloc {block} {err}"
        );
        probes += 1;
    }
    let mut next = session
        .step(ids[context - 1], false, false)
        .expect("invariant: dernier token du contexte")
        .argmax as usize;
    eprintln!(
        "rejeu de {context} tokens en {:.0} s, {probes} sondes de l'indexeur conformes \
         (écart max des requêtes après RoPE {worst_q:.2e})",
        started.elapsed().as_secs_f64()
    );
    let point = session
        .checkpoint(Some(&crate::MemoryGuard::process()))
        .expect("invariant: point de reprise");
    let first = next;
    let pct = |v: &[f64], q: f64| {
        let mut v = v.to_vec();
        v.sort_by(f64::total_cmp);
        v[((v.len() - 1) as f64 * q).round() as usize]
    };
    let mut slowest = f64::INFINITY;
    for run in 0..runs {
        session.restore(&point).expect("invariant: reprise");
        next = first;
        let mut walls = Vec::with_capacity(generated);
        let started = std::time::Instant::now();
        for _ in 0..generated {
            let step = std::time::Instant::now();
            next = session
                .step(next, false, false)
                .expect("invariant: decode")
                .argmax as usize;
            walls.push(step.elapsed().as_secs_f64() * 1e3);
        }
        let throughput = generated as f64 / started.elapsed().as_secs_f64();
        slowest = slowest.min(throughput);
        eprintln!(
            "répétition {run} : {throughput:.1} tok/s ({generated} tokens / durée totale) ; \
             par pas médiane {:.2} ms, p10 {:.2}, p90 {:.2}, max {:.2}",
            pct(&walls, 0.5),
            pct(&walls, 0.1),
            pct(&walls, 0.9),
            pct(&walls, 1.0)
        );
    }
    eprintln!("porte 40 tok/s à 32k : répétition la plus lente à {slowest:.1} tok/s");
    assert!(slowest.is_finite(), "débit non mesuré : {slowest}");
    assert!(
        slowest >= 40.0,
        "porte de débit rouge : {slowest:.1} tok/s < 40"
    );
}

/// Capacité maximale (262 144 tokens) : la session s'ouvre sous la garde
/// mémoire, ses allocations (états, KV f32, clés poolées, KV compact, scratch,
/// plus le scratch du premier pas) restent sous `session_state_bytes`, et un
/// pas s'exécute. Le pic mémoire se lit avec `/usr/bin/time -l`. Le remplissage
/// à 256k exige le prefill résident (étape suivante).
#[test]
#[ignore = "poids réels : SARAGOSSA_QWEN4EXP_DIR"]
fn resident_session_allocates_max_capacity() {
    use super::session::{session_state_bytes, MAX_CAPACITY};
    use crate::qwen_loader::qwen4_exp::Qwen4ExpCheckpoint;
    let dir = std::path::PathBuf::from(
        std::env::var_os("SARAGOSSA_QWEN4EXP_DIR").expect("SARAGOSSA_QWEN4EXP_DIR requis"),
    );
    let config = Qwen4ExpCheckpoint::open(&dir)
        .expect("invariant: checkpoint")
        .config;
    let mut session = FlashNextSession::open(
        Qwen4ExpCheckpoint::open(&dir).expect("invariant: checkpoint"),
        MAX_CAPACITY,
        Some(&crate::MemoryGuard::process()),
    )
    .expect("invariant: session à 256k");
    let before = session.device_allocated_bytes();
    session
        .step(0, false, false)
        .expect("invariant: premier pas");
    let lazy = session.device_allocated_bytes().saturating_sub(before);
    let used = session.owned_state_bytes() + lazy;
    let budget = session_state_bytes(&config, MAX_CAPACITY);
    eprintln!(
        "session {MAX_CAPACITY} tokens : {:.2} Go alloués hors poids (dont {:.1} Mo au premier pas), budget {:.2} Go",
        used as f64 / 1e9,
        lazy as f64 / 1e6,
        budget as f64 / 1e9
    );
    assert!(used <= budget, "allocations {used} > budget {budget}");
}

/// RoPE résidente de Flash-Next (`encode_norm_rope` : q/k de l'attention et
/// requêtes de l'indexeur) contre `rms_norm_1p` + `rope_in_place` CPU aux
/// positions longues : l'angle `position·inv_freq` doit rester exact à 1e-5
/// jusqu'à 262 143. Le kernel générique de l'arène (fast-math, `pow`) sert de
/// témoin : il dépasse 1e-4 dès 32k.
#[test]
fn flash_next_rope_decode_matches_cpu_at_long_positions() -> Result<()> {
    use super::attention::encode_norm_rope;
    use crate::decode_resident::DecodeResidentState;
    use crate::qwen4_exp::math::{inv_freq, rms_norm_1p, rope_in_place};
    let Some(executor) = executor()? else {
        return Ok(());
    };
    let arena = DecodeResidentState::new(executor.device().clone())?;
    let kernels = FlashNextKernels::new(executor.device())?;
    let queue = executor.device().new_command_queue();
    let mut report = Vec::new();
    // Indexeur (4 × 128) puis attention (24 × 256), RoPE sur 64 dimensions.
    for (heads, d) in [(4_usize, 128_usize), (24, 256)] {
        let (rotary, theta) = (64_usize, 1.0e7_f32);
        let mut rng = DeterministicF32::new(0x20e);
        let x = rng.fill(heads * d, -1.0, 1.0);
        let w = rng.fill(d, -0.3, 0.3);
        let gains: Vec<f32> = w.iter().map(|v| 1.0 + v).collect();
        let freqs = inv_freq(rotary, f64::from(theta));
        let (x_buf, gains_buf, freqs_buf) = (
            buffer(&executor, &x),
            buffer(&executor, &gains),
            buffer(&executor, &freqs),
        );
        for position in [5_usize, 2051, 32_767, 131_071, 262_143] {
            let (out, witness) = (zeros(&executor, heads * d), zeros(&executor, heads * d));
            let command_buffer = queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encode_norm_rope(
                &kernels, encoder, &x_buf, &gains_buf, &freqs_buf, &out, heads, d, rotary,
                position, 1.0e-6,
            )?;
            arena.encode_rms_norm_rope_decode(
                encoder, &x_buf, &gains_buf, &witness, heads, d, rotary, rotary, position, 1.0e-6,
                theta,
            )?;
            encoder.end_encoding();
            commit_and_wait(command_buffer)?;
            let mut want = rms_norm_1p(&x, d, d, &w, 1.0e-6);
            for head in want.chunks_exact_mut(d) {
                rope_in_place(head, &freqs, position);
            }
            let (err, _) = relative_error(&read_f32_buffer(&out, heads * d)?, &want);
            let (arena_err, _) = relative_error(&read_f32_buffer(&witness, heads * d)?, &want);
            report.push((d, position, err, arena_err));
        }
    }
    eprintln!(
        "RoPE résidente contre CPU (d, position, écart Flash-Next, écart arène) : {report:?}"
    );
    for (d, position, err, _) in &report {
        assert!(*err <= 1.0e-5, "d {d}, position {position} : {err}");
    }
    // Le témoin montre que le test discrimine l'angle imprécis.
    assert!(
        report.iter().any(|r| r.3 > 1.0e-4),
        "le kernel de l'arène ne s'écarte plus aux positions longues"
    );
    Ok(())
}
