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
    affine_weight(executor, rng, rows, cols, scale, 8)
}

/// Poids affine gs64 de `bits` bits, échelles et biais arrondis en bf16.
fn affine_weight(
    executor: &MetalExecutor,
    rng: &mut DeterministicF32,
    rows: usize,
    cols: usize,
    scale: f32,
    bits: usize,
) -> Result<(MetalLinearWeightBuffers, Tensor)> {
    let quantized = affine_quantized(rng, rows, cols, scale, bits)?;
    let reference = quantized.dequantize()?;
    let gpu = executor.resolve_linear_weight_buffers(
        &LinearWeight::AffineQuantized(quantized),
        "flash_next_hc_test",
    )?;
    Ok((gpu, reference))
}

/// Poids affine gs64 aléatoire, échelles et biais arrondis en bf16.
fn affine_quantized(
    rng: &mut DeterministicF32,
    rows: usize,
    cols: usize,
    scale: f32,
    bits: usize,
) -> Result<AffineQuantizedTensor> {
    let dense = Tensor::from_vec(vec![rows, cols], rng.fill(rows * cols, -scale, scale))?;
    let raw = AffineQuantizedTensor::quantize(&dense, 64, bits)?;
    // Échelles et biais bf16, comme dans le checkpoint : le GPU les lit en bf16.
    let bf16 = |v: &[f32]| -> Vec<f32> { v.iter().map(|x| bf16_round_f32(*x)).collect() };
    AffineQuantizedTensor::new(
        raw.packed_shape(),
        raw.packed_data().to_vec(),
        Tensor::from_vec(raw.scales_shape().to_vec(), bf16(&raw.scales_f32()))?,
        Tensor::from_vec(raw.biases_shape().to_vec(), bf16(&raw.biases_f32()))?,
        64,
        bits,
    )
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
            &kernels,
            &executor,
            encoder,
            &weights,
            state,
            &state.bytes.clone(),
            &scratch,
            dims,
            &h,
            &out,
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
        query_offset: 0,
        index_query_offset: 0,
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
        freqs: buffer(&executor, &crate::rope::inv_freq(64, 1.0e7)),
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

/// Mode d'une passe N2 résidente.
#[derive(Clone, Copy, Debug, PartialEq)]
enum RoutingMode {
    /// Routage et sélection propres du GPU.
    Native,
    /// Comme `Native`, plus un mauvais bloc QSA imposé à `(token, couche)`
    /// (puissance de l'oracle ré-ancré).
    NativeQsaMutation(usize, usize),
    /// Experts du CPU imposés aux points de marge de probabilité ≤ seuil ;
    /// sélection QSA propre du GPU.
    Forced(f32),
    /// Comme `Forced`, plus un mauvais expert imposé à `(token, couche)`.
    Mutation(usize, usize, f32),
}

/// Sélection QSA GPU d'une couche : blocs retenus, scores, blocs natifs.
type GpuQsa = (Vec<usize>, Vec<f32>);

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
        let threshold = match mode {
            RoutingMode::Native | RoutingMode::NativeQsaMutation(..) => None,
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
        session.forced_selection = (0..layers)
            .map(|layer| {
                if !matches!(mode, RoutingMode::NativeQsaMutation(mt, ml) if (mt, ml) == (t, layer))
                {
                    return None;
                }
                // Un bloc retenu remplacé par le premier bloc complet écarté.
                let blocks = cpu_blocks(cpu, layer, t)?;
                let outsider = (0..).find(|b| !blocks.contains(b))?;
                let mut wrong: Vec<u32> = blocks.iter().map(|&b| b as u32).collect();
                wrong[0] = outsider as u32;
                wrong.sort_unstable();
                Some(wrong)
            })
            .collect();
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
                entry.map(|(blocks, scores)| {
                    let sorted = |v: Vec<u32>| {
                        let mut v: Vec<usize> = v.into_iter().map(|b| b as usize).collect();
                        v.sort_unstable();
                        v
                    };
                    (sorted(blocks), scores)
                })
            })
            .collect::<Vec<_>>();
        reports.push(TokenReport {
            layer_errors,
            kl,
            swap,
            routing,
            imposed,
            qsa,
            anchors,
        });
    }
    reports
}

/// Seuil de l'oracle ré-ancré : quasi-égalités du routage et écart des scores
/// de l'indexeur, dans l'unité des marges. Les scores GPU et CPU, aux mêmes
/// entrées, diffèrent par l'ordre d'accumulation f32 (qmv u8 sur 2560, puis
/// 4 têtes × d128) : 8,1e-6 mesuré au plus.
const ANCHORED_TIE: f32 = 1.0e-5;
/// Justification des échanges de sélection : une erreur ε par score peut
/// échanger des blocs à 2ε du seuil.
const ANCHORED_SWAP_TIE: f32 = 2.0 * ANCHORED_TIE;
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
        within(self.score_error, f64::from(ANCHORED_TIE))
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
                Some((blocks, _)) => {
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
            if let Some((blocks, scores)) = &r.qsa[layer] {
                entry.score_error = nan_max(
                    entry.score_error,
                    score_error(scores, &trace.scores[t], keep),
                );
                let want = trace_blocks(trace, t, keep, compress).unwrap_or_default();
                if *blocks != want {
                    entry.selection_swaps += 1;
                    if !swap_justified(&trace.scores[t], &want, blocks, keep, ANCHORED_SWAP_TIE) {
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
}

/// Points QSA où le GPU retient d'autres blocs que le CPU :
/// `(token, couche, marge CPU au seuil)`.
fn qsa_mismatch_points(reports: &[TokenReport], cpu: &CpuChain) -> Vec<(usize, usize, f32)> {
    reports
        .iter()
        .enumerate()
        .flat_map(|(t, r)| {
            (0..r.qsa.len())
                .filter(move |&l| {
                    r.qsa[l].as_ref().map(|(b, _)| b) != cpu_blocks(cpu, l, t).as_ref()
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
    let qsa_mismatches = qsa_mismatch_points(reports, cpu).len();
    Summary {
        worst,
        kl,
        swaps,
        bad: bad.len(),
        first: bad.first().copied(),
        routing_mismatches,
        qsa_mismatches,
    }
}

/// Porte d'une passe imposée : écart ≤ 1e-3, KL ≤ 1e-3, valeurs finies,
/// argmax identique sauf quasi-égalité (≤ 1e-5), routage identique aux points
/// non imposés. Évaluée sur le régime dense, où la clause QSA est vraie par
/// construction : elle garde la porte honnête si un préfixe creux y entrait.
fn forced_gate_green(summary: &Summary) -> bool {
    within(summary.worst, 1.0e-3)
        && within(summary.kl, 1.0e-3)
        && summary.swaps.iter().all(|s| within(s.3, 1.0e-5))
        && summary.routing_mismatches == 0
        && summary.qsa_mismatches == 0
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
        r.qsa[l].as_ref().map(|(b, _)| b) != cpu_blocks(cpu, l, t).as_ref()
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
            .map(|(b, _)| b.clone())
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
/// `SARAGOSSA_QWEN4EXP_TOKENS` (4096 par défaut, régime QSA creux au-delà de
/// 2051).
///
/// - Passe native contrôlée jusqu'à la première bascule ([`native_check`]).
/// - Oracle ré-ancré, porte stricte ([`anchored_qsa_layers`]) : chaque couche
///   QSA CPU repart des entrées GPU de la passe native ; mutations de
///   sélection et numérique.
/// - Bout en bout, experts du CPU imposés aux quasi-égalités, sélection QSA
///   propre du GPU : régime dense vert au seuil le plus strict de
///   [`TIE_THRESHOLDS`] ([`forced_gate_green`]), aucun argmax différent au
///   seuil 1e-6 sur toute la séquence ; dérive publiée sans porte au-delà.
/// - Mutations de routage : un mauvais expert imposé à marge nette fait
///   échouer la porte du régime dense.
#[test]
#[ignore = "poids réels : SARAGOSSA_QWEN4EXP_DIR et SARAGOSSA_QWEN4EXP_GOLDEN"]
fn n2_resident_decode_matches_cpu_reference() {
    use crate::qwen4_exp::Qwen4ExpReference;
    use crate::qwen_loader::qwen4_exp::Qwen4ExpCheckpoint;
    let dir = std::path::PathBuf::from(
        std::env::var_os("SARAGOSSA_QWEN4EXP_DIR")
            .expect("invariant: SARAGOSSA_QWEN4EXP_DIR requis"),
    );
    let golden = std::env::var_os("SARAGOSSA_QWEN4EXP_GOLDEN")
        .expect("invariant: SARAGOSSA_QWEN4EXP_GOLDEN requis");
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
    // L'étape 5 se valide en régime creux : au moins un pas au-delà de 2051.
    assert!(
        tokens > dense_max_visible(cpu.keep * cpu.compress, cpu.compress),
        "N2 sans pas creux : SARAGOSSA_QWEN4EXP_TOKENS = {tokens}"
    );
    let native = run_resident_pass(&dir, &ids, &cpu, RoutingMode::Native, tokens);
    let native_first = native_check(&native, &cpu);
    // Points QSA par marge relative au seuil du top-512.
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

    // Bout en bout : experts du CPU imposés aux quasi-égalités, sélection QSA
    // propre du GPU. Au-delà du régime dense, la dérive des scores (1e-3 à 4k,
    // comme entre le CPU et torch) rend une imposition QSA stricte vide de
    // sens : la sélection y est un contrôle publié, la porte stricte est
    // l'oracle ré-ancré. Portes : régime dense comme à l'étape 4 (seuil de
    // routage le plus strict qui la garde verte), et aucun argmax différent
    // sur toute la séquence au seuil de routage 1e-6.
    let sparse_start = dense_max_visible(cpu.keep * cpu.compress, cpu.compress).min(tokens);
    let dense_prefixes: Vec<usize> = prefixes.iter().map(|&p| p.min(sparse_start)).collect();
    let mut chosen = None;
    let mut argmax_at_1e6 = None;
    for th in TIE_THRESHOLDS {
        let forced = run_resident_pass(&dir, &ids, &cpu, RoutingMode::Forced(th), tokens);
        let dense_green = dense_prefixes
            .iter()
            .all(|&p| forced_gate_green(&summarize(&forced, &cpu, p)));
        let s = summarize(&forced, &cpu, tokens);
        let mut score_drift = 0.0_f64;
        for (t, r) in forced.iter().enumerate() {
            for (l, entry) in r.qsa.iter().enumerate() {
                if let (Some((_, gpu)), Some(trace)) = (entry, cpu.qsa[l].as_ref()) {
                    score_drift =
                        nan_max(score_drift, score_error(gpu, &trace.scores[t], cpu.keep));
                }
            }
        }
        let imposed: usize = forced.iter().map(|r| r.imposed.len()).sum();
        let mismatches = qsa_mismatch_points(&forced, &cpu);
        eprintln!(
            "imposée ≤ {th:.0e} : régime dense (préfixes {dense_prefixes:?}) {}, \
             {tokens} tokens : écart max {:.2e}, KL max {:.2e}, argmax ≠ {:?}, premier {:?}, \
             routage imposé {imposed} points sur {} ({:.2} %), routage ≠ hors imposition {}, \
             sélections QSA ≠ {} (10 premières {:?}), dérive des scores {score_drift:.2e}",
            if dense_green { "vert" } else { "rouge" },
            s.worst,
            s.kl,
            s.swaps,
            s.first,
            tokens * cpu.layers.len(),
            100.0 * imposed as f64 / (tokens * cpu.layers.len()) as f64,
            s.routing_mismatches,
            mismatches.len(),
            &mismatches[..mismatches.len().min(10)]
        );
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
        if th == 1.0e-6 {
            argmax_at_1e6 = Some((s.swaps.clone(), all_finite(&forced)));
        }
        if dense_green && chosen.is_none() {
            chosen = Some(th);
        }
    }
    eprintln!("seuil de routage retenu (régime dense) : {chosen:?}");
    let threshold = chosen.unwrap_or(TIE_THRESHOLDS[TIE_THRESHOLDS.len() - 1]);

    // Mutations : un mauvais expert imposé au point de plus grande marge CPU,
    // tôt (tokens 8 à 63) puis tard (64 tokens après la première bascule
    // native), dans la passe imposée au seuil retenu.
    let after = native_first
        .as_ref()
        .ok()
        .and_then(|f| f.map(|(t, _, _)| t + 1))
        .unwrap_or(8);
    for (label, range) in [
        ("précoce", 8..tokens.min(64)),
        ("tardive", after..sparse_start.min(after + 64)),
    ] {
        let Some((token, layer)) = range
            .flat_map(|t| (0..cpu.margins.len()).map(move |l| (t, l)))
            .max_by(|a, b| cpu.margins[a.1][a.0].total_cmp(&cpu.margins[b.1][b.0]))
        else {
            eprintln!("mutation {label} non exécutée : plage vide en régime dense");
            continue;
        };
        let steps = (token + 8).min(sparse_start);
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
                && within(c.score_error, f64::from(ANCHORED_TIE))
                && within(c.output_error, ANCHORED_OUTPUT_TOL)),
            "mutation QSA (token {token}, couche {layer}) non détectée par l'oracle ré-ancré : {check:?}"
        );
    }
    assert!(anchored_green, "oracle ré-ancré rouge : {anchored:?}");
    assert!(chosen.is_some(), "régime dense rouge à tous les seuils");
    let (swaps, finite) = argmax_at_1e6.expect("invariant: passe à 1e-6");
    assert!(
        swaps.is_empty() && finite,
        "argmax différents au seuil de routage 1e-6 : {swaps:?}"
    );
    native_first.unwrap_or_else(|e| panic!("passe native : {e}"));
    assert!(native_finite, "passe native : valeur non finie");
}

/// Empreinte au bit près du decode résident, pour prouver qu'un refactor ne
/// change aucun bit : rejoue 2048 tokens du golden N2 sans tête, puis écrit, pour
/// 8 pas (dont les derniers en régime QSA creux), l'argmax, les logits et les
/// résiduels hc de chaque couche dans `SARAGOSSA_FLASH_NEXT_BITS_OUT`. Deux
/// binaires (avant / après) doivent produire des fichiers identiques.
#[test]
#[ignore = "poids réels : SARAGOSSA_QWEN4EXP_DIR, SARAGOSSA_QWEN4EXP_GOLDEN, SARAGOSSA_FLASH_NEXT_BITS_OUT"]
fn flash_next_step_bits_dump() {
    use crate::qwen_loader::qwen4_exp::Qwen4ExpCheckpoint;
    use std::io::Write;
    let dir = std::path::PathBuf::from(
        std::env::var_os("SARAGOSSA_QWEN4EXP_DIR")
            .expect("invariant: SARAGOSSA_QWEN4EXP_DIR requis"),
    );
    let golden = std::env::var_os("SARAGOSSA_QWEN4EXP_GOLDEN")
        .expect("invariant: SARAGOSSA_QWEN4EXP_GOLDEN requis");
    let out = std::env::var_os("SARAGOSSA_FLASH_NEXT_BITS_OUT")
        .expect("invariant: SARAGOSSA_FLASH_NEXT_BITS_OUT requis");
    let (context, steps) = (2048, 8);
    let ids = n2_input_ids(std::path::Path::new(&golden));
    let mut session = FlashNextSession::open(
        Qwen4ExpCheckpoint::open(&dir).expect("invariant: checkpoint"),
        context + steps,
        Some(&crate::MemoryGuard::process()),
    )
    .expect("invariant: session");
    for &token in &ids[..context] {
        session.replay(token).expect("invariant: rejeu");
    }
    let mut file = std::fs::File::create(&out).expect("invariant: fichier de sortie");
    for &token in &ids[context..context + steps] {
        let step = session.step(token, true, true).expect("invariant: pas");
        file.write_all(&step.argmax.to_le_bytes())
            .expect("invariant: écriture");
        let values = step
            .logits
            .into_iter()
            .flatten()
            .chain(step.layers.into_iter().flatten().flatten());
        for v in values {
            file.write_all(&v.to_le_bytes())
                .expect("invariant: écriture");
        }
    }
    eprintln!("empreinte écrite : {} pas après {context} tokens", steps);
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
        std::env::var_os("SARAGOSSA_QWEN4EXP_DIR")
            .expect("invariant: SARAGOSSA_QWEN4EXP_DIR requis"),
    );
    let golden = std::env::var_os("SARAGOSSA_QWEN4EXP_GOLDEN")
        .expect("invariant: SARAGOSSA_QWEN4EXP_GOLDEN requis");
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
        std::env::var_os("SARAGOSSA_QWEN4EXP_DIR")
            .expect("invariant: SARAGOSSA_QWEN4EXP_DIR requis"),
    );
    let golden = std::env::var_os("SARAGOSSA_QWEN4EXP_GOLDEN")
        .expect("invariant: SARAGOSSA_QWEN4EXP_GOLDEN requis");
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
        .expect("invariant: pas accepté après un reset partiel");
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
        std::env::var_os("SARAGOSSA_QWEN4EXP_DIR")
            .expect("invariant: SARAGOSSA_QWEN4EXP_DIR requis"),
    );
    let golden = std::env::var_os("SARAGOSSA_QWEN4EXP_GOLDEN")
        .expect("invariant: SARAGOSSA_QWEN4EXP_GOLDEN requis");
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
        .expect("invariant: SARAGOSSA_FLASH_NEXT_PROFILE=1 requis")
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
            "  {:24} {:7.3} ms/pas  {:5.1} %  {:5.1} encodeurs/pas  {:6.0} dispatches/pas",
            row.brick.label(),
            row.ms_per_step,
            100.0 * row.ms_per_step / gpu,
            row.encoders_per_step,
            row.dispatches_per_step
        );
    }
    profile_rows_rest(&session).expect("invariant: profil du reste");
    profile_rows_deltanet_moe(&session).expect("invariant: profil DeltaNet et MoE");
}

/// Étape 6 D, mesures in situ sur les vrais poids (toutes les couches, donc
/// lues depuis la DRAM), en vérification greedy stricte : couche DeltaNet
/// entière contre ses matmuls en R lignes (reste in situ = scan, conv, normes,
/// portes), et MoE entier en R lignes par le kernel existant contre le MoE
/// mono, avec l'exactitude au bit de chaque ligne. Chauffe, puis minimum de 3.
/// Les états DeltaNet avancent : la session n'est plus utilisable ensuite.
#[cfg(feature = "devtools")]
fn profile_rows_deltanet_moe(session: &FlashNextSession) -> Result<()> {
    use crate::metal_backend::{
        install_verify_greedy_scope, with_verify_strict_override,
        MetalLinearAttnResidentPairWeights,
    };
    let (executor, _, _, hidden) = session.head_parts();
    let parts = session.deltanet_and_moe();
    let queue = executor.device().new_command_queue();
    let best_of = |encode: &dyn Fn(
        &metal::ComputeCommandEncoderRef,
        &mut Vec<Buffer>,
    ) -> Result<()>|
     -> Result<f64> {
        let mut best = f64::INFINITY;
        for round in 0..4 {
            let mut owned = Vec::new();
            let command_buffer = queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encode(encoder, &mut owned)?;
            encoder.end_encoding();
            let started = std::time::Instant::now();
            commit_and_wait(command_buffer)?;
            if round > 0 {
                best = best.min(started.elapsed().as_secs_f64());
            }
        }
        Ok(best)
    };
    // Comparaison publiée en trois classes : identique au bit, identique au
    // signe des zéros près, différente.
    #[derive(Default)]
    struct Exactness {
        bits: usize,
        signed_zero: usize,
        different: usize,
    }
    impl Exactness {
        fn add(&mut self, a: &[f32], b: &[f32]) {
            let same = |zero: bool| {
                a.iter()
                    .zip(b)
                    .all(|(x, y)| x.to_bits() == y.to_bits() || (zero && *x == 0.0 && *y == 0.0))
            };
            if same(false) {
                self.bits += 1;
            } else if same(true) {
                self.signed_zero += 1;
            } else {
                self.different += 1;
            }
        }
        fn report(&self) -> String {
            format!(
                "{} identiques au bit, {} au signe des zéros près, {} différentes",
                self.bits, self.signed_zero, self.different
            )
        }
    }
    let value_dim = parts.dims.value_dim;
    let mut rng = DeterministicF32::new(0x6f);
    let x_host = rng.fill(4 * hidden, -1.0, 1.0);
    let v_host = rng.fill(4 * value_dim, -1.0, 1.0);
    let x = buffer(executor, &x_host);
    let v = buffer(executor, &v_host);
    let x_rows: Vec<_> = (0..4)
        .map(|r| buffer(executor, &x_host[r * hidden..(r + 1) * hidden]))
        .collect();
    let v_rows: Vec<_> = (0..4)
        .map(|r| buffer(executor, &v_host[r * value_dim..(r + 1) * value_dim]))
        .collect();
    // Matmuls d'une couche DeltaNet par leurs poids séparés : (poids, entrée en
    // lignes, entrées par ligne, dimension d'entrée).
    fn projections(
        w: &crate::metal_backend::MetalLinearAttnResidentDenseWeights,
    ) -> Vec<(&MetalLinearWeightBuffers, bool)> {
        let mut list = Vec::new();
        for pair in [&w.qkv_z, &w.beta_gate] {
            match pair {
                MetalLinearAttnResidentPairWeights::Concat(weight) => list.push((weight, true)),
                MetalLinearAttnResidentPairWeights::Split { first, second } => {
                    list.push((first, true));
                    list.push((second, true));
                }
            }
        }
        list.push((&w.out_proj, false));
        list
    }
    let max_out = 4 * (parts.dims.conv_dim + 2 * value_dim + hidden);
    let out = zeros(executor, max_out);
    let _greedy = install_verify_greedy_scope(true);
    with_verify_strict_override(true, || -> Result<()> {
        let layer_out = zeros(executor, hidden);
        let t_dn = best_of(&|encoder, _| {
            for (weights, state) in &parts.linear {
                executor.encode_linear_attn_resident_dense_buffers(
                    encoder, &x_rows[0], None, &layer_out, weights, state, parts.spec, parts.dims,
                )?;
            }
            Ok(())
        })?;
        let mut line = format!(
            "DeltaNet in situ ({} couches) : couche entière {:.3} ms",
            parts.linear.len(),
            t_dn * 1e3
        );
        let mut mono = 0.0;
        for rows in 1..=4 {
            let t = best_of(&|encoder, _| {
                for (weights, _) in &parts.linear {
                    for (weight, on_x) in projections(weights) {
                        let (input, in_dim) = if on_x { (&x, hidden) } else { (&v, value_dim) };
                        executor
                            .encode_matmul_weight_buffers(
                                encoder, input, rows, in_dim, weight, &out, true,
                            )
                            .map(|_| ())?;
                    }
                }
                Ok(())
            })?;
            if rows == 1 {
                mono = t;
                line += &format!(
                    " ; matmuls R1 {:.3} ms, reste in situ {:.3} ms",
                    t * 1e3,
                    (t_dn - t) * 1e3
                );
            } else {
                line += &format!(" ; matmuls R{rows} {:.3} ms (ρ {:.2})", t * 1e3, t / mono);
            }
        }
        // Exactitude des projections de toutes les couches, R = 2..4 : chaque
        // ligne contre son matmul mono.
        let mut exact = Exactness::default();
        for (weights, _) in &parts.linear {
            for (weight, on_x) in projections(weights) {
                let (input, rows_in, in_dim) = if on_x {
                    (&x, &x_rows, hidden)
                } else {
                    (&v, &v_rows, value_dim)
                };
                for rows in 2..=4 {
                    let command_buffer = queue.new_command_buffer();
                    let encoder = command_buffer.new_compute_command_encoder();
                    let out_dim = executor.encode_matmul_weight_buffers(
                        encoder, input, rows, in_dim, weight, &out, true,
                    )?;
                    let singles: Vec<_> = (0..rows).map(|_| zeros(executor, out_dim)).collect();
                    for (row, single) in rows_in.iter().zip(&singles) {
                        executor.encode_matmul_weight_buffers(
                            encoder, row, 1, in_dim, weight, single, true,
                        )?;
                    }
                    encoder.end_encoding();
                    commit_and_wait(command_buffer)?;
                    let all = read_f32_buffer(&out, rows * out_dim)?;
                    for (r, single) in singles.iter().enumerate() {
                        exact.add(
                            &read_f32_buffer(single, out_dim)?,
                            &all[r * out_dim..(r + 1) * out_dim],
                        );
                    }
                }
            }
        }
        eprintln!(
            "{line} ; lignes (toutes couches, R = 2..4) : {}",
            exact.report()
        );

        // MoE : mono (chemin du pas) puis lignes (kernel de vérification).
        let moe_out = zeros(executor, 4 * hidden);
        let residual = zeros(executor, 4 * hidden);
        let t_moe = best_of(&|encoder, owned| {
            for weights in &parts.moe {
                executor.encode_moe_shared_buffers(
                    encoder,
                    owned,
                    &x_rows[0],
                    None,
                    &moe_out,
                    hidden,
                    weights,
                    parts.top_k,
                )?;
            }
            Ok(())
        })?;
        let mut line = format!(
            "MoE in situ ({} couches) : mono {:.3} ms",
            parts.moe.len(),
            t_moe * 1e3
        );
        for rows in 2..=4 {
            let t = best_of(&|encoder, owned| {
                for weights in &parts.moe {
                    executor.encode_moe_shared_buffers_verify_rows(
                        encoder,
                        owned,
                        &x,
                        &residual,
                        &moe_out,
                        rows,
                        hidden,
                        weights,
                        parts.top_k,
                    )?;
                }
                Ok(())
            })?;
            line += &format!(" ; R{rows} {:.3} ms (ρ {:.2})", t * 1e3, t / t_moe);
        }
        // Exactitude, R = 2..4, couche par couche : chaque ligne contre le MoE
        // mono du pas (sans résiduel, comme en decode).
        let mut exact = Exactness::default();
        for weights in &parts.moe {
            for rows in 2..=4 {
                let mut owned = Vec::new();
                let singles: Vec<_> = (0..rows).map(|_| zeros(executor, hidden)).collect();
                let command_buffer = queue.new_command_buffer();
                let encoder = command_buffer.new_compute_command_encoder();
                executor.encode_moe_shared_buffers_verify_rows(
                    encoder,
                    &mut owned,
                    &x,
                    &residual,
                    &moe_out,
                    rows,
                    hidden,
                    weights,
                    parts.top_k,
                )?;
                for (row, single) in x_rows.iter().zip(&singles) {
                    executor.encode_moe_shared_buffers(
                        encoder,
                        &mut owned,
                        row,
                        None,
                        single,
                        hidden,
                        weights,
                        parts.top_k,
                    )?;
                }
                encoder.end_encoding();
                commit_and_wait(command_buffer)?;
                let all = read_f32_buffer(&moe_out, rows * hidden)?;
                for (r, single) in singles.iter().enumerate() {
                    exact.add(
                        &read_f32_buffer(single, hidden)?,
                        &all[r * hidden..(r + 1) * hidden],
                    );
                }
            }
        }
        eprintln!(
            "{line} ; lignes ({} couches, R = 2..4) : {}",
            parts.moe.len(),
            exact.report()
        );
        Ok(())
    })
}

/// Étape 6 D, reste du pas hors matmuls : latence d'un dispatch suivi d'une
/// barrière (part amortie par un kernel en lignes, qui garde un dispatch pour
/// R lignes), et argmax de tête en lignes sur le vrai lm_head u6, contre R
/// argmax mono (ρ et égalité des indices). Chaque mesure : chauffe, puis
/// minimum de 3 command buffers.
#[cfg(feature = "devtools")]
fn profile_rows_rest(session: &FlashNextSession) -> Result<()> {
    let (executor, kernels, lm_head, hidden) = session.head_parts();
    let queue = executor.device().new_command_queue();
    let best_of = |encode: &dyn Fn(&metal::ComputeCommandEncoderRef) -> Result<()>| -> Result<f64> {
        let mut best = f64::INFINITY;
        for round in 0..4 {
            let command_buffer = queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encode(encoder)?;
            encoder.end_encoding();
            let started = std::time::Instant::now();
            commit_and_wait(command_buffer)?;
            if round > 0 {
                best = best.min(started.elapsed().as_secs_f64());
            }
        }
        Ok(best)
    };
    // Latence d'un petit dispatch (copie de 1 à 10 240 mots) et de sa barrière.
    let source = zeros(executor, 4 * 10_240);
    let target = zeros(executor, 4 * 10_240);
    let count = 2000;
    for len in [1_usize, 10_240, 4 * 10_240] {
        let elapsed = best_of(&|encoder| {
            for _ in 0..count {
                encoder.set_compute_pipeline_state(&kernels.copy_u32);
                encoder.set_buffer(0, Some(&source), 0);
                encoder.set_buffer(1, Some(&target), 0);
                super::set_value(encoder, 2, &[len as u32, 0]);
                super::dispatch_1d(encoder, len);
                crate::metal_backend::post_dispatch_barrier(encoder);
            }
            Ok(())
        })?;
        eprintln!(
            "dispatch + barrière, copie de {len} mots : {:.2} µs",
            elapsed * 1e6 / count as f64
        );
    }
    // Argmax de tête : R mono contre une passe en lignes.
    let mut rng = DeterministicF32::new(0x6e);
    let input = buffer(executor, &rng.fill(4 * hidden, -1.0, 1.0));
    let rows_in: Vec<_> = (0..4)
        .map(|r| -> Result<Buffer> {
            let row = zeros(executor, hidden);
            let command_buffer = queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            executor.encode_copy_with_offsets(
                encoder,
                &input,
                (r * hidden * 4) as u64,
                &row,
                0,
                hidden,
            )?;
            encoder.end_encoding();
            commit_and_wait(command_buffer)?;
            Ok(row)
        })
        .collect::<Result<_>>()?;
    let indices = zeros(executor, 4);
    let singles: Vec<_> = (0..4).map(|_| zeros(executor, 1)).collect();
    let mono = best_of(&|encoder| {
        let mut owned = Vec::new();
        executor.encode_lm_head_argmax_buffers(
            encoder,
            &mut owned,
            &rows_in[0],
            lm_head,
            &singles[0],
            hidden,
        )
    })?;
    let mut line = format!("argmax de tête (u6 réel) : mono {:.2} ms", mono * 1e3);
    for rows in 2..=4 {
        let batched = best_of(&|encoder| {
            executor.encode_lm_head_argmax_rows_buffers(
                encoder, &input, lm_head, &indices, 0, 0, rows, hidden,
            )
        })?;
        let command_buffer = queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let mut owned = Vec::new();
        for r in 0..rows {
            executor.encode_lm_head_argmax_buffers(
                encoder,
                &mut owned,
                &rows_in[r],
                lm_head,
                &singles[r],
                hidden,
            )?;
        }
        encoder.end_encoding();
        commit_and_wait(command_buffer)?;
        let all = crate::metal_backend::read_u32_buffer(&indices, rows)?;
        let mut same = true;
        for (r, single) in singles.iter().take(rows).enumerate() {
            same &= crate::metal_backend::read_u32_buffer(single, 1)?[0] == all[r];
        }
        line += &format!(
            " ; R{rows} ρ {:.2}{}",
            batched / mono,
            if same {
                " mêmes indices"
            } else {
                " indices DIFFÉRENTS"
            }
        );
    }
    eprintln!("{line}");
    Ok(())
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
    use crate::rope::inv_freq;
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
        query_offset: 0,
        index_query_offset: 0,
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
    use crate::qwen_loader::qwen4_exp::{Qwen4ExpCheckpoint, Qwen4ExpLayerKind};
    use crate::rope::inv_freq;
    let dir = std::path::PathBuf::from(
        std::env::var_os("SARAGOSSA_QWEN4EXP_DIR")
            .expect("invariant: SARAGOSSA_QWEN4EXP_DIR requis"),
    );
    let corpus = std::path::PathBuf::from(
        std::env::var_os("SARAGOSSA_FLASH_NEXT_CORPUS")
            .expect("invariant: SARAGOSSA_FLASH_NEXT_CORPUS requis"),
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
    let freqs = inv_freq(c.rotary_dim, c.rope_theta as f32);
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
        let (blocks, scores) = out.qsa.expect("invariant: QSA")[probe_layer]
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

/// Troncature de l'historique MTP : un dépassement d'historique est signalé
/// avant l'absence d'instantané ; une copie de restauration des clés brutes en
/// échec invalide la session (longueur déjà réduite) : troncature, même à
/// longueur inchangée, instantané, pas MTP et pas du tronc sont refusés jusqu'à
/// un `reset`, qui la réhabilite.
#[test]
#[ignore = "poids réels : SARAGOSSA_QWEN4EXP_DIR"]
fn mtp_failed_raw_restore_poisons_session() {
    use super::session::MtpInput;
    use crate::qwen_loader::qwen4_exp::Qwen4ExpCheckpoint;
    let dir = std::path::PathBuf::from(
        std::env::var_os("SARAGOSSA_QWEN4EXP_DIR")
            .expect("invariant: SARAGOSSA_QWEN4EXP_DIR requis"),
    );
    let mut session = FlashNextSession::open(
        Qwen4ExpCheckpoint::open(&dir).expect("invariant: checkpoint"),
        64,
        Some(&crate::MemoryGuard::process()),
    )
    .expect("invariant: session");
    let h = vec![0.0; session.hc_width()];
    let append = |session: &mut FlashNextSession, count: usize| {
        for token in 0..count {
            session
                .mtp_step(MtpInput::Host(&h), token + 1, false, false)
                .expect("invariant: pas MTP");
        }
    };
    // 6 lignes : bloc en cours non vide (compress = 4), la troncature à 6
    // exige donc la restauration, où l'échec est injecté.
    append(&mut session, 6);
    session.mtp_snapshot_raw().expect("invariant: instantané");
    append(&mut session, 2);
    let overflow = session.mtp_truncate(9).expect_err("invariant: dépassement");
    assert!(overflow.to_string().contains("au-delà"), "{overflow}");
    session.fail_mtp_raw_restore = true;
    let failed = session
        .mtp_truncate(6)
        .expect_err("invariant: échec injecté");
    session.fail_mtp_raw_restore = false;
    assert!(failed.to_string().contains("injecté"), "{failed}");
    assert_eq!(
        session.mtp_rows(),
        Some(6),
        "longueur réduite avant la copie"
    );
    assert!(
        session.mtp_truncate(6).is_err(),
        "troncature à longueur inchangée"
    );
    assert!(session.mtp_snapshot_raw().is_err(), "instantané");
    assert!(
        session
            .mtp_step(MtpInput::Host(&h), 1, false, false)
            .is_err(),
        "pas MTP"
    );
    assert!(session.step(1, false, false).is_err(), "pas du tronc");
    session.reset().expect("invariant: reset");
    session
        .step(1, false, false)
        .expect("invariant: pas après reset");
    assert_eq!(session.mtp_rows(), Some(0), "historique MTP remis à zéro");
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
        std::env::var_os("SARAGOSSA_QWEN4EXP_DIR")
            .expect("invariant: SARAGOSSA_QWEN4EXP_DIR requis"),
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
    // Tête MTP : un pas (scratch paresseux de la couche et du lm_head).
    if session.mtp_rows().is_some() {
        session
            .mtp_step(
                super::session::MtpInput::Host(&vec![0.0; config.hc_width()]),
                0,
                false,
                false,
            )
            .expect("invariant: premier pas MTP");
    }
    let lazy = session.device_allocated_bytes().saturating_sub(before);
    let used = session.owned_state_bytes() + lazy;
    let budget = session_state_bytes(&config, MAX_CAPACITY);
    // Point de reprise (MTP comprise) : accepté sous la garde du process,
    // refusé sous un plafond inférieur à ses copies.
    session
        .checkpoint(Some(&crate::MemoryGuard::process()))
        .expect("invariant: point de reprise sous la garde");
    let tiny = crate::MemoryGuard::from_limits(
        true,
        crate::memory_guard::MemoryLimits {
            static_cap: Some(1),
            host_memory: None,
            headroom_bytes: 0,
            metal_cap: None,
        },
        std::sync::Arc::new(|| Some(0)),
    );
    assert!(
        session.checkpoint(Some(&tiny)).is_err(),
        "point de reprise accepté sous un plafond de 1 octet"
    );
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
/// jusqu'à 262 143. La copie figée de l'ancien kernel d'arène (fast-math,
/// `position / pow`) sert de témoin : elle dépasse 1e-4 dès 32k.
#[test]
fn flash_next_rope_decode_matches_cpu_at_long_positions() -> Result<()> {
    use super::attention::encode_norm_rope;
    use crate::qwen4_exp::math::{rms_norm_1p, rope_in_place};
    use crate::rope::inv_freq;
    let Some(executor) = executor()? else {
        return Ok(());
    };
    let legacy = crate::decode_resident::kernels::compile_kernel(
        executor.device(),
        crate::decode_resident::tests::LEGACY_ROPE_DECODE_KERNEL,
        "rms_norm_rope_heads_decode_f32",
    )?;
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
        let freqs = inv_freq(rotary, theta);
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
            // Témoin : ancien kernel, `frequency_dim` passé en octets (slot 5).
            encoder.set_compute_pipeline_state(&legacy);
            encoder.set_buffer(0, Some(&x_buf), 0);
            encoder.set_buffer(1, Some(&gains_buf), 0);
            encoder.set_buffer(2, Some(&witness), 0);
            super::set_value(
                encoder,
                3,
                &[heads as u32, d as u32, rotary as u32, position as u32],
            );
            super::set_value(encoder, 4, &[1.0e-6_f32, theta]);
            super::set_value(encoder, 5, &(rotary as u32));
            encoder.dispatch_thread_groups(
                metal::MTLSize::new(heads as u64, 1, 1),
                metal::MTLSize::new(256, 1, 1),
            );
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
        "RoPE résidente contre CPU (d, position, écart Flash-Next, écart du témoin figé) : {report:?}"
    );
    for (d, position, err, _) in &report {
        assert!(*err <= 1.0e-5, "d {d}, position {position} : {err}");
    }
    // Le témoin montre que le test discrimine l'angle imprécis.
    assert!(
        report.iter().any(|r| r.3 > 1.0e-4),
        "le témoin figé ne s'écarte plus aux positions longues"
    );
    Ok(())
}

/// Tenseur d'un golden safetensors : `(forme, octets)`.
fn golden_tensor(path: &std::path::Path, name: &str) -> (Vec<usize>, Vec<u8>) {
    use std::os::unix::fs::FileExt;
    let header = crate::qwen_loader::read_shard_header(path).expect("invariant: golden lisible");
    let entry = header
        .tensors
        .get(name)
        .unwrap_or_else(|| panic!("golden sans {name}"));
    let mut bytes = vec![0_u8; entry.data_offsets[1] - entry.data_offsets[0]];
    std::fs::File::open(path)
        .expect("invariant: golden")
        .read_exact_at(&mut bytes, header.data_start + entry.data_offsets[0] as u64)
        .expect("invariant: lecture golden");
    (entry.shape.clone(), bytes)
}

fn golden_f32(path: &std::path::Path, name: &str) -> Vec<f32> {
    golden_tensor(path, name)
        .1
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn golden_i64(path: &std::path::Path, name: &str) -> Vec<usize> {
    golden_tensor(path, name)
        .1
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().expect("invariant: 8 octets")) as usize)
        .collect()
}

/// Comparaison à un golden : longueurs égales et valeurs finies exigées
/// ([`finite_relative_error`]), NaN sinon.
fn golden_error(got: &[f32], want: &[f32]) -> f64 {
    if got.len() != want.len() {
        return f64::NAN;
    }
    finite_relative_error(got, want)
}

/// N2 de la tête MTP résidente contre la tête CPU (`qwen4_exp/mtp.rs`), sur
/// les 4095 paires `(h_t, tok_{t+1})` de `mtp_golden.safetensors` (dont 2044
/// en régime QSA creux, frontières 2047-2052 comprises). Chaque ligne GPU part
/// du même `h` que le CPU : la couche est ré-ancrée par construction.
///
/// - Portes contre le CPU, à chaque ligne : entrée fc et sortie hc ≤ 1e-4,
///   scores ≤ 1e-5, échanges de sélection justifiés à 2e-5, routage à la
///   paire au seuil (marge ≤ 1e-5), valeurs finies ; logits aux positions
///   figées : KL ≤ 1e-3, argmax renvoyé identique sauf quasi-égalité ≤ 1e-5.
/// - Triangulation avec transformers aux positions figées : sortie hc
///   (1e-4), scores (1e-4 relatif à `1 + |torch|`), sélection et routage
///   justifiés par les marges torch (1e-5).
/// - Accord du brouillon avec l'argmax du tronc (golden long) ≥ ½.
/// - Mutations (rouges) : gains d'entrée doublés ou absents, paires
///   décalées, flux permutés, requêtes seules décalées (attention, indexeur),
///   historique figé, expert, bloc ; gain doublé aussi contre transformers
///   (une erreur commune au CPU et au GPU y reste visible).
/// - Reprise : point de reprise, drafts parasites à travers une frontière de
///   bloc, restauration, puis continuation exacte.
#[test]
#[ignore = "poids réels : SARAGOSSA_QWEN4EXP_DIR et SARAGOSSA_QWEN4EXP_GOLDEN"]
fn n2_resident_mtp_head_matches_cpu() {
    use super::session::MtpInput;
    use crate::qwen4_exp::Qwen4ExpReference;
    use crate::qwen_loader::qwen4_exp::Qwen4ExpCheckpoint;
    let dir = std::path::PathBuf::from(
        std::env::var_os("SARAGOSSA_QWEN4EXP_DIR")
            .expect("invariant: SARAGOSSA_QWEN4EXP_DIR requis"),
    );
    let golden_dir = std::path::PathBuf::from(
        std::env::var_os("SARAGOSSA_QWEN4EXP_GOLDEN")
            .expect("invariant: SARAGOSSA_QWEN4EXP_GOLDEN requis"),
    );
    let golden = golden_dir.join("mtp_golden.safetensors");
    let next = golden_i64(&golden, "mtp_next_ids");
    let positions = golden_i64(&golden, "mtp_positions");
    let h = golden_f32(&golden, "mtp_h");
    let rows = next.len();
    let reference =
        Qwen4ExpReference::new(Qwen4ExpCheckpoint::open(&dir).expect("invariant: checkpoint"));
    let config = reference.checkpoint().config.clone();
    let (width, vocab, keep, compress, top_k) = (
        config.hc_width(),
        config.vocab,
        reference.budget_blocks(),
        reference.compress(),
        config.top_k,
    );
    let sparse_start = dense_max_visible(keep * compress, compress);
    // Formes : 4095 paires, positions dans l'historique, frontières couvertes.
    assert_eq!(rows, 4095, "paires MTP");
    assert_eq!(h.len(), rows * width, "h MTP");
    assert!(!positions.is_empty() && positions.iter().all(|&p| p < rows));
    let shape = |name: &str| golden_tensor(&golden, name).0;
    assert_eq!(shape("mtp_h"), vec![rows, width], "forme mtp_h");
    assert_eq!(
        shape("mtp_input"),
        vec![positions.len(), width],
        "forme mtp_input"
    );
    assert_eq!(
        shape("mtp_output"),
        vec![positions.len(), width],
        "forme mtp_output"
    );
    assert_eq!(
        shape("routing.0"),
        vec![rows, config.top_k],
        "forme du routage"
    );
    assert_eq!(
        shape("router_logits.0"),
        vec![rows, config.experts],
        "forme des logits du routeur"
    );
    // Frontière du régime creux parmi les positions figées des logits.
    assert!(
        positions
            .iter()
            .any(|&p| p + 1 >= dense_max_visible(keep * compress, compress) && p < rows),
        "aucune position figée en régime creux"
    );
    assert!(rows > sparse_start + 4, "régime creux non couvert");
    let started = std::time::Instant::now();
    let h_tensor = Tensor::from_vec(vec![rows, width], h.clone()).expect("invariant: forme");
    let x = reference
        .mtp_input(&h_tensor, &next)
        .expect("invariant: entrée MTP CPU");
    let cpu = reference
        .mtp_layer(&x, None, None)
        .expect("invariant: couche MTP CPU");
    let trace = cpu.qsa.as_ref().expect("invariant: trace QSA");
    let picked: Vec<f32> = positions
        .iter()
        .flat_map(|&p| {
            cpu.hidden.data()[p * width..(p + 1) * width]
                .iter()
                .copied()
        })
        .collect();
    let cpu_logits = reference
        .logits(
            &reference
                .mtp_head(
                    &Tensor::from_vec(vec![positions.len(), width], picked)
                        .expect("invariant: lignes"),
                )
                .expect("invariant: mixer MTP"),
        )
        .expect("invariant: logits MTP CPU");
    eprintln!(
        "référence CPU MTP : {rows} paires en {:.0} s",
        started.elapsed().as_secs_f64()
    );
    let cpu_row = |j: usize| &cpu.hidden.data()[j * width..(j + 1) * width];
    let mut session = FlashNextSession::open(
        Qwen4ExpCheckpoint::open(&dir).expect("invariant: checkpoint"),
        rows,
        Some(&crate::MemoryGuard::process()),
    )
    .expect("invariant: session");

    // Passe native, une ligne d'historique par paire.
    let (mut in_err, mut out_err, mut score_err, mut kl_max) = (0.0_f64, 0.0, 0.0, 0.0_f64);
    let (mut sel_swaps, mut sel_bad, mut route_swaps, mut route_bad, mut argmax_bad) =
        (0, 0, 0, 0, 0);
    let (mut gpu_out, mut gpu_in, mut gpu_routing, mut gpu_qsa, mut drafts) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for j in 0..rows {
        let at = positions.iter().position(|&p| p == j);
        let step = session
            .mtp_step(
                MtpInput::Host(&h[j * width..(j + 1) * width]),
                next[j],
                at.is_some(),
                true,
            )
            .expect("invariant: pas MTP");
        in_err = nan_max(
            in_err,
            golden_error(&step.input, &x.data()[j * width..(j + 1) * width]),
        );
        let hidden = step.hidden.expect("invariant: sortie hc");
        out_err = nan_max(out_err, golden_error(&hidden, cpu_row(j)));
        let mut routing: Vec<usize> = step
            .routing
            .expect("invariant: routage")
            .into_iter()
            .map(|e| e as usize)
            .collect();
        routing.sort_unstable();
        if routing != cpu.routing[j] {
            route_swaps += 1;
            let (kth, next_e) = cpu.routing_boundary[j];
            let pair = routing
                .iter()
                .filter(|e| !cpu.routing[j].contains(e))
                .eq([&next_e])
                && cpu.routing[j]
                    .iter()
                    .filter(|e| !routing.contains(e))
                    .eq([&kth]);
            if !(pair && cpu.routing_margins[j] <= ANCHORED_TIE) {
                route_bad += 1;
            }
        }
        let qsa = step.qsa.map(|(blocks, scores)| {
            let mut blocks: Vec<usize> = blocks.into_iter().map(|b| b as usize).collect();
            blocks.sort_unstable();
            (blocks, scores)
        });
        match &qsa {
            Some((blocks, scores)) => {
                score_err = nan_max(score_err, score_error(scores, &trace.scores[j], keep));
                let want = trace_blocks(trace, j, keep, compress).unwrap_or_default();
                if *blocks != want {
                    sel_swaps += 1;
                    if !swap_justified(&trace.scores[j], &want, blocks, keep, ANCHORED_SWAP_TIE) {
                        sel_bad += 1;
                    }
                }
            }
            None if trace.scores[j].len() > keep => sel_bad += 1,
            None => {}
        }
        if let Some(i) = at {
            let got = step.logits.expect("invariant: logits");
            let want = &cpu_logits.data()[i * vocab..(i + 1) * vocab];
            let kl = if got.len() == vocab && got.iter().all(|v| v.is_finite()) {
                kl_logits(want, &got)
            } else {
                f64::NAN
            };
            kl_max = nan_max(kl_max, kl);
            let top = want
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1).then(b.0.cmp(&a.0)))
                .map(|(t, _)| t)
                .expect("invariant: vocabulaire");
            let gpu_top = step.argmax as usize;
            if gpu_top != top {
                let max = want[top];
                let sum: f64 = want.iter().map(|&v| f64::from(v - max).exp()).sum();
                let gap = (1.0 - f64::from(want[gpu_top] - max).exp()) / sum;
                if !within(gap, 1.0e-5) {
                    argmax_bad += 1;
                }
            }
            gpu_out.extend_from_slice(&hidden);
            gpu_in.extend_from_slice(&step.input);
            gpu_routing.push(routing);
            gpu_qsa.push(qsa);
            drafts.push(gpu_top);
        }
    }

    // Triangulation avec transformers aux positions figées.
    let torch_err = golden_error(&gpu_out, &golden_f32(&golden, "mtp_output"));
    let torch_in_err = golden_error(&gpu_in, &golden_f32(&golden, "mtp_input"));
    let torch_ids = golden_i64(&golden, "routing.0");
    let torch_logits = golden_f32(&golden, "router_logits.0");
    let experts = config.experts;
    let torch_scores = {
        let flat = golden_f32(&golden, "qsa_scores.0");
        let offsets = golden_i64(&golden, "qsa_offsets.0");
        offsets
            .windows(2)
            .map(|w| flat[w[0]..w[1]].to_vec())
            .collect::<Vec<_>>()
    };
    let (mask_shape, mask) = golden_tensor(&golden, "qsa_mask.0");
    assert_eq!(mask_shape, vec![rows, rows], "masque torch");
    let (mut torch_route_bad, mut torch_sel_bad, mut torch_score_err) = (0, 0, 0.0_f64);
    for (i, &p) in positions.iter().enumerate() {
        // Routage : écart justifié par la marge de probabilité torch au seuil.
        let logits_p = &torch_logits[p * experts..(p + 1) * experts];
        let max = logits_p.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = logits_p.iter().map(|v| (v - max).exp()).collect();
        let sum: f32 = exp.iter().sum();
        let probs: Vec<f32> = exp.iter().map(|v| v / sum).collect();
        let mut torch_set: Vec<usize> = torch_ids[p * top_k..(p + 1) * top_k].to_vec();
        torch_set.sort_unstable();
        if gpu_routing[i] != torch_set {
            // Chaque expert échangé doit être au seuil torch du top-k.
            let mut sorted = probs.clone();
            sorted.sort_by(|a, b| b.total_cmp(a));
            let threshold = sorted[top_k - 1];
            let swapped = gpu_routing[i]
                .iter()
                .filter(|e| !torch_set.contains(e))
                .chain(torch_set.iter().filter(|e| !gpu_routing[i].contains(e)));
            if !swapped.into_iter().all(|&e| {
                within(probs[e] - threshold, 1.0e-5) && within(threshold - probs[e], 1.0e-5)
            }) {
                torch_route_bad += 1;
            }
        }
        // Scores et sélection de l'indexeur contre torch.
        if let Some((blocks, scores)) = &gpu_qsa[i] {
            let ts = &torch_scores[p];
            if ts.len() != scores.len() {
                torch_score_err = f64::NAN;
            }
            for (g, t) in scores.iter().zip(ts) {
                torch_score_err =
                    nan_max(torch_score_err, f64::from((g - t).abs() / (1.0 + t.abs())));
            }
            let torch_blocks: Vec<usize> = (0..ts.len())
                .filter(|&b| mask[p * rows + b * compress] != 0)
                .collect();
            if *blocks != torch_blocks && !swap_justified(ts, &torch_blocks, blocks, keep, 1.0e-5) {
                torch_sel_bad += 1;
            }
        }
    }
    // Accord du brouillon (ligne p, prédit tok_{p+2}) avec l'argmax du tronc.
    let long = golden_dir.join("long.safetensors");
    let trunk_positions = golden_i64(&long, "logit_positions");
    let trunk = golden_f32(&long, "logits");
    let (mut agree, mut denominator) = (0, 0);
    for (i, &p) in positions.iter().enumerate() {
        if let Some(k) = trunk_positions.iter().position(|&t| t == p + 1) {
            denominator += 1;
            let row = &trunk[k * vocab..(k + 1) * vocab];
            let top = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1).then(b.0.cmp(&a.0)))
                .map(|(t, _)| t)
                .expect("invariant: vocabulaire");
            agree += usize::from(drafts[i] == top);
        }
    }
    eprintln!(
        "MTP résidente : entrée {in_err:.2e}, sortie {out_err:.2e}, scores {score_err:.2e}, \
         sélections ≠ {sel_swaps} (injustifiées {sel_bad}), routages ≠ {route_swaps} \
         (injustifiés {route_bad}), KL max {kl_max:.2e}, argmax injustifiés {argmax_bad} ; \
         transformers : entrée {torch_in_err:.2e}, sortie {torch_err:.2e}, scores {torch_score_err:.2e}, routages \
         injustifiés {torch_route_bad}, sélections injustifiées {torch_sel_bad} ; accord avec \
         le tronc {agree}/{denominator}"
    );
    let green = within(in_err, ANCHORED_OUTPUT_TOL)
        && within(out_err, ANCHORED_OUTPUT_TOL)
        && within(score_err, f64::from(ANCHORED_TIE))
        && within(kl_max, 1.0e-3)
        && within(torch_err, ANCHORED_OUTPUT_TOL)
        && within(torch_in_err, ANCHORED_OUTPUT_TOL)
        && within(torch_score_err, 1.0e-4)
        && sel_bad == 0
        && route_bad == 0
        && argmax_bad == 0
        && torch_route_bad == 0
        && torch_sel_bad == 0
        && denominator > 0
        && 2 * agree >= denominator;

    // Mutations, sur la même session remise à zéro : `hook(session, j)` avant
    // chaque pas, entrée `(h_j, tok)` éventuellement altérée par `pair(j)`.
    type Hook<'a> = &'a dyn Fn(&mut FlashNextSession, usize);
    type Pair<'a> = &'a dyn Fn(usize) -> (Vec<f32>, usize);
    let plain = |j: usize| (h[j * width..(j + 1) * width].to_vec(), next[j]);
    // Chaque mutation est jugée par la porte de l'observable qu'elle touche :
    // `Watch::Output` (sortie hc, 1e-4), `Watch::Scores` (scores de l'indexeur
    // contre le CPU, 1e-5, avec un écart muté d'au moins 2 × la porte) ou
    // `Watch::Selection` (top-512 basculé hors quasi-égalité). Les trois
    // mesures sont publiées pour chaque mutation.
    #[derive(Clone, Copy, PartialEq)]
    enum Watch {
        Output,
        Scores,
        Selection,
    }
    struct Mutated {
        output: f64,
        scores: f64,
        selection_bad: usize,
        /// Sélections malformées (doublons, hors bornes, taille ≠ keep) ou
        /// scores non finis : une bascule n'est pas prouvée par celles-ci.
        malformed: usize,
        sparse_rows: usize,
        torch: f64,
    }
    let run = |session: &mut FlashNextSession,
               label: &str,
               steps: usize,
               watch: Watch,
               hook: Hook<'_>,
               pair: Pair<'_>|
     -> Mutated {
        session.reset().expect("invariant: reset");
        let mut m = Mutated {
            output: 0.0,
            scores: 0.0,
            selection_bad: 0,
            malformed: 0,
            sparse_rows: 0,
            torch: 0.0,
        };
        let mut torch = Vec::new();
        for j in 0..steps {
            hook(session, j);
            let (hj, tok) = pair(j);
            let step = session
                .mtp_step(MtpInput::Host(&hj), tok, false, true)
                .expect("invariant: pas muté");
            let hidden = step.hidden.expect("invariant: sortie hc");
            m.output = nan_max(m.output, golden_error(&hidden, cpu_row(j)));
            if let Some((blocks, scores)) = step.qsa {
                m.sparse_rows += 1;
                m.scores = nan_max(m.scores, score_error(&scores, &trace.scores[j], keep));
                let mut blocks: Vec<usize> = blocks.into_iter().map(|b| b as usize).collect();
                blocks.sort_unstable();
                let well_formed = blocks.len() == keep
                    && blocks.windows(2).all(|w| w[0] < w[1])
                    && blocks.last().is_some_and(|&b| b < scores.len())
                    && scores.iter().all(|v| v.is_finite());
                if !well_formed {
                    m.malformed += 1;
                }
                let want = trace_blocks(trace, j, keep, compress).unwrap_or_default();
                if well_formed
                    && blocks != want
                    && !swap_justified(&trace.scores[j], &want, &blocks, keep, ANCHORED_SWAP_TIE)
                {
                    m.selection_bad += 1;
                }
            }
            if positions.contains(&j) {
                torch.extend_from_slice(&hidden);
            }
        }
        session.set_query_offsets(0, 0);
        let n = torch.len() / width;
        m.torch = golden_error(&torch, &golden_f32(&golden, "mtp_output")[..n * width]);
        eprintln!(
            "mutation {label} : sortie {:.2e}, scores {:.2e} (natif {score_err:.2e}), \
             sélections injustifiées {} (malformées {}) sur {} lignes creuses",
            m.output, m.scores, m.selection_bad, m.malformed, m.sparse_rows
        );
        let detected = match watch {
            Watch::Output => !within(m.output, ANCHORED_OUTPUT_TOL),
            Watch::Scores => {
                m.sparse_rows > 0
                    && m.scores.is_finite()
                    && m.scores >= 2.0 * f64::from(ANCHORED_TIE)
            }
            Watch::Selection => m.selection_bad > 0 && m.malformed == 0,
        };
        assert!(detected, "mutation {label} non détectée");
        m
    };
    let gains = |session: &FlashNextSession| {
        let (e, hd) = session.mtp_norm_buffers().expect("invariant: tête MTP");
        (
            read_f32_buffer(e, config.hidden).expect("invariant: gains"),
            read_f32_buffer(hd, width).expect("invariant: gains"),
        )
    };
    let (gain_e, gain_h) = gains(&session);
    let set_gains = |session: &FlashNextSession, e: &[f32], hd: &[f32]| {
        let (eb, hb) = session.mtp_norm_buffers().expect("invariant: tête MTP");
        crate::metal_backend::write_f32_buffer(eb, e).expect("invariant: écriture");
        crate::metal_backend::write_f32_buffer(hb, hd).expect("invariant: écriture");
    };
    let doubled = |w: &[f32]| w.iter().map(|v| 2.0 * v + 1.0).collect::<Vec<_>>();
    let probe = 64;
    let none: Hook<'_> = &|_, _| {};
    let torch_doubled = run(
        &mut session,
        "gain doublé (flux)",
        probe,
        Watch::Output,
        &|s, j| {
            if j == 0 {
                set_gains(s, &gain_e, &doubled(&gain_h));
            }
        },
        &plain,
    );
    assert!(
        !within(torch_doubled.torch, ANCHORED_OUTPUT_TOL),
        "gain doublé invisible contre transformers ({:.2e})",
        torch_doubled.torch
    );
    run(
        &mut session,
        "gain absent (flux)",
        probe,
        Watch::Output,
        &|s, j| {
            if j == 0 {
                set_gains(s, &gain_e, &vec![0.0; width]);
            }
        },
        &plain,
    );
    run(
        &mut session,
        "gain doublé (embedding)",
        probe,
        Watch::Output,
        &|s, j| {
            if j == 0 {
                set_gains(s, &doubled(&gain_e), &gain_h);
            }
        },
        &plain,
    );
    set_gains(&session, &gain_e, &gain_h);
    run(
        &mut session,
        "paires décalées",
        probe,
        Watch::Output,
        none,
        &|j| (h[j * width..(j + 1) * width].to_vec(), next[j + 1]),
    );
    run(
        &mut session,
        "flux permutés",
        probe,
        Watch::Output,
        none,
        &|j| {
            let mut hj = h[j * width..(j + 1) * width].to_vec();
            hj.rotate_left(config.hidden);
            (hj, next[j])
        },
    );
    run(
        &mut session,
        "requêtes d'attention décalées",
        probe,
        Watch::Output,
        &|s, j| {
            if j == 0 {
                s.set_query_offsets(1, 0);
            }
        },
        &plain,
    );
    let b_row = sparse_start + 4;
    // Requêtes de l'indexeur seules décalées, sur 256 lignes creuses : d'une
    // position (scores, porte 1e-5 inchangée), puis de 1024 positions (le
    // top-512 bascule, porte de sélection).
    let index_rows = sparse_start + 256;
    let shifted = run(
        &mut session,
        "requêtes d'indexeur décalées d'une position",
        index_rows,
        Watch::Scores,
        &|s, j| {
            if j == 0 {
                s.set_query_offsets(0, 1);
            }
        },
        &plain,
    );
    assert!(
        shifted.sparse_rows >= 200,
        "lignes creuses sondées : {}",
        shifted.sparse_rows
    );
    run(
        &mut session,
        "requêtes d'indexeur décalées de 1024 positions",
        index_rows,
        Watch::Selection,
        &|s, j| {
            if j == 0 {
                s.set_query_offsets(0, 1024);
            }
        },
        &plain,
    );
    run(
        &mut session,
        "historique figé",
        probe,
        Watch::Output,
        &|s, j| {
            if j > 0 {
                s.mtp_truncate(0).expect("invariant: troncature");
            }
        },
        &plain,
    );
    let e_row = 8;
    run(
        &mut session,
        "mauvais expert",
        e_row + 1,
        Watch::Output,
        &|s, j| {
            if j == e_row {
                let set = &cpu.routing[e_row];
                let outsider = (0..)
                    .find(|e| !set.contains(e))
                    .expect("invariant: expert libre");
                let mut wrong: Vec<u32> = set.iter().map(|&e| e as u32).collect();
                wrong[0] = outsider as u32;
                s.mtp_forced_routing = Some(wrong);
            }
        },
        &plain,
    );
    run(
        &mut session,
        "mauvais bloc",
        b_row + 1,
        Watch::Output,
        &|s, j| {
            if j == b_row {
                let blocks =
                    trace_blocks(trace, b_row, keep, compress).expect("invariant: régime creux");
                let outsider = (0..)
                    .find(|b| !blocks.contains(b))
                    .expect("invariant: bloc libre");
                let mut wrong: Vec<u32> = blocks.iter().map(|&b| b as u32).collect();
                wrong[0] = outsider as u32;
                wrong.sort_unstable();
                s.mtp_forced_selection = Some(wrong);
            }
        },
        &plain,
    );

    // Reprise : point à `cp` lignes, trois drafts parasites qui complètent un
    // bloc (`cp + 2` termine le bloc `(cp + 2) / 4`), restauration, puis
    // continuation exacte jusqu'à `cp + 8`.
    session.reset().expect("invariant: reset");
    let cp = sparse_start - 2;
    assert_eq!(
        (cp + 2) % compress,
        compress - 1,
        "le bloc doit se compléter"
    );
    for j in 0..cp {
        session
            .mtp_step(
                MtpInput::Host(&h[j * width..(j + 1) * width]),
                next[j],
                false,
                false,
            )
            .expect("invariant: historique");
    }
    let checkpoint = session
        .checkpoint(None)
        .expect("invariant: point de reprise");
    for j in cp..cp + 3 {
        let parasite = &h[(rows - 1 - j) * width..(rows - j) * width];
        session
            .mtp_step(MtpInput::Host(parasite), next[rows - 1 - j], false, false)
            .expect("invariant: draft parasite");
    }
    session
        .restore(&checkpoint)
        .expect("invariant: restauration");
    let mut resumed = 0.0_f64;
    for j in cp..cp + 8 {
        let step = session
            .mtp_step(
                MtpInput::Host(&h[j * width..(j + 1) * width]),
                next[j],
                false,
                true,
            )
            .expect("invariant: continuation");
        resumed = nan_max(
            resumed,
            golden_error(&step.hidden.expect("invariant: sortie"), cpu_row(j)),
        );
    }
    // Draft enchaîné : entrée résidente (sortie du pas précédent) contre la
    // même entrée relue sur l'hôte, au bit près.
    // Copie réelle : le buffer de sortie de la session est réécrit par le pas
    // hôte, un alias fausserait la comparaison.
    let host_copy = read_f32_buffer(
        session
            .mtp_hidden_buffer()
            .expect("invariant: sortie résidente"),
        width,
    )
    .expect("invariant: relecture");
    let chained = metal::Device::system_default()
        .expect("invariant: device Metal")
        .new_buffer_with_data(
            host_copy.as_ptr().cast(),
            std::mem::size_of_val(host_copy.as_slice()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
    let before = session
        .checkpoint(None)
        .expect("invariant: point de reprise");
    let token = next[cp + 8];
    let from_host = session
        .mtp_step(MtpInput::Host(&host_copy), token, false, true)
        .expect("invariant: draft hôte");
    session.restore(&before).expect("invariant: restauration");
    let from_buffer = session
        .mtp_step(MtpInput::Resident(&chained), token, false, true)
        .expect("invariant: draft résident");
    assert_eq!(from_host.argmax, from_buffer.argmax, "draft enchaîné");
    let bits = |v: Option<Vec<f32>>| -> Vec<u32> {
        v.expect("invariant: sortie")
            .iter()
            .map(|x| x.to_bits())
            .collect()
    };
    assert_eq!(
        bits(from_host.hidden),
        bits(from_buffer.hidden),
        "entrée résidente ≠ entrée hôte"
    );
    eprintln!("reprise après drafts parasites : écart de sortie {resumed:.2e}");
    assert!(
        within(resumed, ANCHORED_OUTPUT_TOL),
        "continuation après reprise : {resumed:.2e}"
    );
    assert!(green, "N2 MTP résidente rouge");
}

/// Vérification en pas mono contre l'AR, au bit près : après un contexte de
/// `P` tokens du golden N2 (`P` = 2046..2049, les quatre alignements modulo 4
/// autour de l'entrée du régime creux), `verify(t_0..t_3)` doit rendre pour
/// chaque ligne l'argmax et le résiduel hc avant mixer de l'AR (mêmes bits).
/// Puis, pour chaque `k` de 0 à 3 : `accept(k)` suivi d'un pas sur un token
/// divergent doit égaler l'AR `t_0..t_k` puis ce token (argmax et résiduels de
/// toutes les couches) : le retour arrière (DeltaNet, KV, clés brutes, PLE)
/// est exact. Une seconde vérification, ou un pas, avant `accept` est refusé.
#[test]
#[ignore = "poids réels : SARAGOSSA_QWEN4EXP_DIR et SARAGOSSA_QWEN4EXP_GOLDEN"]
fn resident_verify_rows_and_accept_match_ar_bitwise() {
    use crate::qwen_loader::qwen4_exp::Qwen4ExpCheckpoint;
    let dir = std::path::PathBuf::from(
        std::env::var_os("SARAGOSSA_QWEN4EXP_DIR")
            .expect("invariant: SARAGOSSA_QWEN4EXP_DIR requis"),
    );
    let golden = std::env::var_os("SARAGOSSA_QWEN4EXP_GOLDEN")
        .expect("invariant: SARAGOSSA_QWEN4EXP_GOLDEN requis");
    let ids = n2_input_ids(std::path::Path::new(&golden));
    let rows = super::session::VERIFY_ROWS;
    let open = || {
        FlashNextSession::open(
            Qwen4ExpCheckpoint::open(&dir).expect("invariant: checkpoint"),
            2049 + rows + 2,
            Some(&crate::MemoryGuard::process()),
        )
        .expect("invariant: session")
    };
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
    // Une seule session (deux copies des poids dépasseraient le plafond) : les
    // références AR et les vérifications repartent du même point de reprise.
    let mut session = open();
    // Token divergent : différent de tous les tokens vérifiés.
    let divergent = 13;
    for context in 2046..2050 {
        let tokens = &ids[context..context + rows];
        assert!(!tokens.contains(&divergent));
        session.reset().expect("invariant: reset");
        for &token in &ids[..context] {
            session.replay(token).expect("invariant: rejeu");
        }
        let base = session
            .checkpoint(None)
            .expect("invariant: point de reprise");
        // Références AR : pas t_0..t_3 (résiduel avant mixer = sortie de la
        // dernière couche), puis pour chaque k, t_0..t_k et le token divergent.
        let mut want = Vec::new();
        for &token in tokens {
            let step = session.step(token, false, true).expect("invariant: pas AR");
            let layers = step.layers.expect("invariant: résiduels");
            want.push((step.argmax, layers[layers.len() - 1].clone()));
        }
        let mut continuations = Vec::new();
        for k in 0..rows {
            session.restore(&base).expect("invariant: reprise");
            for &token in &tokens[..=k] {
                session
                    .step(token, false, false)
                    .expect("invariant: pas AR");
            }
            let digest = session.state_digest().expect("invariant: état AR");
            let step = session
                .step(divergent, false, true)
                .expect("invariant: continuation AR");
            continuations.push((
                step.argmax,
                step.layers.expect("invariant: résiduels"),
                digest,
            ));
        }
        for (k, (argmax, reference, digest)) in continuations.iter().enumerate() {
            session.restore(&base).expect("invariant: reprise");
            let got = session.verify(tokens).expect("invariant: vérification");
            assert!(
                session.verify(tokens).is_err(),
                "seconde vérification acceptée"
            );
            assert!(
                session.step(tokens[0], false, false).is_err(),
                "pas accepté en attente"
            );
            for (r, (row, (want_argmax, residual))) in got.iter().zip(&want).enumerate() {
                assert_eq!(row.argmax, *want_argmax, "P {context}, ligne {r} : argmax");
                assert_eq!(
                    bits(&row.residual),
                    bits(residual),
                    "P {context}, ligne {r} : résiduel"
                );
            }
            assert!(session.accept(rows).is_err(), "acceptation hors bornes");
            session.accept(k).expect("invariant: acceptation");
            assert_eq!(session.position(), context + k + 1);
            // Oracle d'état direct : DeltaNet, préfixes KV et blocs poolés valides,
            // clés brutes du bloc en cours, anneau, pas et contexte PLE.
            assert!(
                session.state_digest().expect("invariant: état") == *digest,
                "P {context}, k {k} : état après acceptation ≠ AR"
            );
            let after = session
                .step(divergent, false, true)
                .expect("invariant: continuation");
            assert_eq!(after.argmax, *argmax, "P {context}, k {k} : argmax");
            let layers = after.layers.expect("invariant: résiduels");
            for (layer, (x, y)) in layers.iter().zip(reference).enumerate() {
                assert_eq!(bits(x), bits(y), "P {context}, k {k}, couche {layer}");
            }
        }
        eprintln!("vérification P {context} : 4 lignes et k = 0..3 identiques à l'AR");
        if context != 2047 {
            continue;
        }
        // Mutations (k = 1, P = 2047) : chacune doit écarter l'état ou les
        // sorties de l'AR.
        use super::session::VerifyMutation;
        for mutation in [
            VerifyMutation::CaptureNext,
            VerifyMutation::NoTruncate,
            VerifyMutation::StaleContext,
            VerifyMutation::PleStepOff,
            VerifyMutation::AliasFirstRow,
            VerifyMutation::WrongResidual,
        ] {
            session.restore(&base).expect("invariant: reprise");
            session.verify_mutation = Some(mutation);
            let got = session
                .verify(tokens)
                .expect("invariant: vérification mutée");
            session.accept(1).expect("invariant: acceptation mutée");
            session.verify_mutation = None;
            let rows_differ = got.iter().zip(&want).any(|(row, (argmax, residual))| {
                row.argmax != *argmax || bits(&row.residual) != bits(residual)
            });
            let state_differs =
                session.state_digest().expect("invariant: état") != continuations[1].2;
            eprintln!("mutation {mutation:?} : sorties ≠ {rows_differ}, état ≠ {state_differs}");
            assert!(
                rows_differ || state_differs,
                "mutation {mutation:?} non détectée"
            );
        }
        // Échec d'encodage injecté sur la dernière ligne : rien n'est engagé.
        session.restore(&base).expect("invariant: reprise");
        let before = session.state_digest().expect("invariant: état");
        session.fail_verify_row = Some(rows - 1);
        assert!(session.verify(tokens).is_err(), "échec injecté ignoré");
        session.fail_verify_row = None;
        assert!(
            session.state_digest().expect("invariant: état") == before,
            "état modifié par une vérification non soumise"
        );
        // Refus : pas d'acceptation sans vérification ; pendant une transaction,
        // ni rejeu, ni pas MTP, ni point de reprise, ni reprise.
        assert!(session.accept(0).is_err(), "acceptation sans vérification");
        assert!(
            session.accept(usize::MAX).is_err(),
            "acceptation hors bornes"
        );
        assert!(session.verify(&[]).is_err(), "vérification vide");
        assert!(
            session.verify(&[usize::MAX]).is_err(),
            "token hors vocabulaire"
        );
        assert!(session.verify(&ids[..rows + 1]).is_err(), "trop de lignes");
        session
            .verify(&tokens[..2])
            .expect("invariant: deux lignes");
        assert!(session.replay(tokens[0]).is_err(), "rejeu en attente");
        assert!(
            session.checkpoint(None).is_err(),
            "point de reprise en attente"
        );
        assert!(session.restore(&base).is_err(), "reprise en attente");
        if session.mtp_rows().is_some() {
            assert!(
                session
                    .mtp_step(
                        super::session::MtpInput::Host(&want[0].1),
                        tokens[0],
                        false,
                        false
                    )
                    .is_err(),
                "pas MTP en attente"
            );
        }
        session
            .accept(0)
            .expect("invariant: acceptation d'une ligne");
        assert!(
            session.state_digest().expect("invariant: état") == continuations[0].2,
            "deux lignes, k = 0 : état ≠ AR"
        );
        // Une seule ligne : pas de capture ni de restauration.
        session.restore(&base).expect("invariant: reprise");
        let one = session.verify(&tokens[..1]).expect("invariant: une ligne");
        assert_eq!(one[0].argmax, want[0].0, "une ligne : argmax");
        session.accept(0).expect("invariant: acceptation");
        assert!(
            session.state_digest().expect("invariant: état") == continuations[0].2,
            "une ligne : état ≠ AR"
        );
        // `reset` pendant une transaction la lève et rend la session utilisable.
        session.verify(tokens).expect("invariant: vérification");
        session.reset().expect("invariant: reset en attente");
        session
            .step(ids[0], false, false)
            .expect("invariant: pas après reset");
    }
}

/// Identité stricte du décodage spéculatif MTP (pas mono) contre l'AR glouton :
/// 8 prompts réels (golden N2 de 64, 512, 2040 et 3000 tokens, wikitext figé de
/// 1024 et 2600 tokens, deux fichiers de code tokenisés) × 256 jetons, aux
/// profondeurs 1, 2 et 3 : mêmes jetons, et mêmes résiduels avant mixer au bit
/// près sur les positions engagées communes. Le prompt de 2040 tokens traverse
/// 2047-2052 pendant la génération. Cas limites : drafts forcés faux (rejet
/// total), jeton d'arrêt rencontré en cours de route, capacité atteinte en fin
/// de génération. L'acceptation (jetons par cycle) est publiée.
#[test]
#[ignore = "poids réels : SARAGOSSA_QWEN4EXP_DIR, SARAGOSSA_QWEN4EXP_GOLDEN, SARAGOSSA_FLASH_NEXT_CORPUS"]
fn speculative_greedy_matches_ar_bitwise() {
    use super::session::MtpInput;
    use super::speculative::{
        generate_ar, generate_speculative, Generation, SpecMutation, SpecOptions,
    };
    use crate::qwen_loader::qwen4_exp::Qwen4ExpCheckpoint;
    let none = SpecOptions::default();
    let dir = std::path::PathBuf::from(
        std::env::var_os("SARAGOSSA_QWEN4EXP_DIR")
            .expect("invariant: SARAGOSSA_QWEN4EXP_DIR requis"),
    );
    let golden = std::env::var_os("SARAGOSSA_QWEN4EXP_GOLDEN")
        .expect("invariant: SARAGOSSA_QWEN4EXP_GOLDEN requis");
    let corpus = std::path::PathBuf::from(
        std::env::var_os("SARAGOSSA_FLASH_NEXT_CORPUS")
            .expect("invariant: SARAGOSSA_FLASH_NEXT_CORPUS requis"),
    );
    let ids = n2_input_ids(std::path::Path::new(&golden));
    let wiki: Vec<usize> = std::fs::read(corpus.join("ids.u32"))
        .expect("invariant: corpus")
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as usize)
        .collect();
    let tokenizer =
        tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("invariant: tokenizer");
    let code = |name: &str, len: usize| -> Vec<usize> {
        let text = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/rowsexact")
                .join(name),
        )
        .expect("invariant: fixture de code");
        let ids: Vec<usize> = tokenizer
            .encode(text, false)
            .expect("invariant: tokenisation")
            .get_ids()
            .iter()
            .map(|&t| t as usize)
            .collect();
        ids[..len.min(ids.len())].to_vec()
    };
    let prompts: Vec<(&str, Vec<usize>)> = vec![
        ("golden 64", ids[..64].to_vec()),
        ("golden 512", ids[..512].to_vec()),
        ("golden 2040", ids[..2040].to_vec()),
        ("golden 3000", ids[..3000].to_vec()),
        ("wikitext 1024", wiki[..1024].to_vec()),
        ("wikitext 2600", wiki[4096..4096 + 2600].to_vec()),
        ("code4k-1", code("code4k-1.txt", 1024)),
        ("code4k-2", code("code4k-2.txt", 1500)),
    ];
    let generated = 256;
    let capacity = 3000 + generated + 4;
    let mut session = FlashNextSession::open(
        Qwen4ExpCheckpoint::open(&dir).expect("invariant: checkpoint"),
        capacity,
        Some(&crate::MemoryGuard::process()),
    )
    .expect("invariant: session");
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
    let same = |label: &str, ar: &Generation, spec: &Generation| {
        assert_eq!(spec.tokens, ar.tokens, "{label} : jetons");
        // `residuals[i]` : résiduel de la position qui produit `tokens[i]`.
        assert_eq!(
            spec.residuals.len(),
            ar.residuals.len(),
            "{label} : résiduels"
        );
        for (i, (a, b)) in ar.residuals.iter().zip(&spec.residuals).enumerate() {
            assert_eq!(bits(a), bits(b), "{label} : résiduel {i}");
        }
    };
    // Jetons, résiduels et état final du tronc (position, KV, DeltaNet, PLE)
    // égaux à l'AR : le dernier jeton émis n'est consommé ni par l'un ni par
    // l'autre.
    let check = |session: &mut FlashNextSession,
                 label: &str,
                 prompt: &[usize],
                 budget: usize,
                 depth: usize,
                 stop: &[usize],
                 options: &SpecOptions<'_>|
     -> Generation {
        let ar = generate_ar(session, prompt, budget, stop).expect("invariant: AR");
        let ar_state = session.state_digest().expect("invariant: état AR");
        let spec = generate_speculative(session, prompt, budget, depth, stop, options)
            .expect("invariant: spéculatif");
        same(label, &ar, &spec);
        assert!(
            session.state_digest().expect("invariant: état") == ar_state,
            "{label} : état final du tronc"
        );
        spec
    };
    let started = std::time::Instant::now();
    for (label, prompt) in &prompts {
        let ar = generate_ar(&mut session, prompt, generated, &[]).expect("invariant: AR");
        let ar_state = session.state_digest().expect("invariant: état AR");
        for depth in 1..=3 {
            let spec = generate_speculative(&mut session, prompt, generated, depth, &[], &none)
                .expect("invariant: spéculatif");
            same(&format!("{label}, profondeur {depth}"), &ar, &spec);
            assert!(
                session.state_digest().expect("invariant: état") == ar_state,
                "{label}, profondeur {depth} : état final du tronc"
            );
            let cycles = spec.accepted.len().max(1);
            let per_cycle =
                spec.accepted.iter().map(|k| k + 1).sum::<usize>() as f64 / cycles as f64;
            eprintln!(
                "{label}, profondeur {depth} : identique ; {per_cycle:.2} jetons par cycle sur {cycles} cycles"
            );
        }
    }
    // Rejet total : chaque draft forcé faux.
    let (_, prompt) = &prompts[1];
    let wrong = |_: usize, _: usize, _: usize, draft: usize| (draft + 1) % 1000;
    let forced = SpecOptions {
        overrides: Some(&wrong),
        ..SpecOptions::default()
    };
    let spec = check(&mut session, "rejet total", prompt, 64, 3, &[], &forced);
    // Un draft forcé peut retomber par hasard sur la cible : compte publié.
    eprintln!(
        "rejet forcé : {} drafts acceptés sur {} cycles",
        spec.accepted.iter().sum::<usize>(),
        spec.accepted.len()
    );
    // Arrêt : jeton d'arrêt pris au milieu de la génération AR.
    let full = generate_ar(&mut session, prompt, 96, &[]).expect("invariant: AR");
    let stop = [full.tokens[40]];
    let spec = check(&mut session, "arrêt", prompt, 96, 3, &stop, &none);
    assert!(stop.contains(spec.tokens.last().expect("invariant: jetons")));
    // Acceptation complète forcée (drafts = jetons AR) : chaque cycle se
    // termine sur un draft accepté, y compris en fin de budget.
    let reference = generate_ar(&mut session, prompt, 64, &[]).expect("invariant: AR");
    let oracle = |_: usize, emitted: usize, rank: usize, _: usize| {
        *reference
            .tokens
            .get(emitted + 1 + rank)
            .expect("invariant: draft d'oracle dans la sortie AR")
    };
    let all = SpecOptions {
        overrides: Some(&oracle),
        ..SpecOptions::default()
    };
    for budget in 2..=8 {
        let spec = check(
            &mut session,
            &format!("acceptation complète, budget {budget}"),
            prompt,
            budget,
            3,
            &[],
            &all,
        );
        // Cycles attendus : 3 drafts acceptés tant que le budget le permet,
        // le dernier cycle réduit à `restant − 2`, le dernier jeton émis seul.
        let mut expected = Vec::new();
        let mut left = budget;
        while left >= 2 {
            let k = (left - 2).min(3);
            expected.push(k);
            left -= k + 1;
        }
        assert_eq!(spec.accepted, expected, "budget {budget} : acceptations");
    }
    // Capacité : génération jusqu'à l'avant-dernière puis la dernière position.
    let (_, prompt) = &prompts[3];
    for budget in [capacity - prompt.len(), capacity - prompt.len() + 1] {
        check(
            &mut session,
            &format!("capacité, budget {budget}"),
            prompt,
            budget,
            3,
            &[],
            &none,
        );
        assert_eq!(
            session.mtp_rows(),
            Some(session.position() - 1),
            "historique MTP en fin de génération"
        );
    }

    // Oracle de l'historique MTP : le D_1 de chaque cycle (position p) doit
    // égaler le draft d'un rejeu séquentiel des paires AR vérifiées
    // (h_j, x_{j+1}). Une troncature oubliée ou des paires bâties sur un mauvais
    // résiduel laissent les jetons identiques (le tronc vérifie toujours) mais
    // changent les drafts.
    let reference_drafts = |session: &mut FlashNextSession, sequence: &[usize]| -> Vec<usize> {
        session.reset().expect("invariant: reset");
        let mut drafts = Vec::with_capacity(sequence.len());
        for j in 0..sequence.len() - 1 {
            let row = session
                .verify(&sequence[j..=j])
                .expect("invariant: pas de référence");
            session.accept(0).expect("invariant: acceptation");
            let step = session
                .mtp_step(
                    MtpInput::Host(&row[0].residual),
                    sequence[j + 1],
                    false,
                    false,
                )
                .expect("invariant: pas MTP de référence");
            drafts.push(step.argmax as usize);
        }
        drafts
    };
    let first_drafts = |session: &mut FlashNextSession,
                        prompt: &[usize],
                        mutation: Option<SpecMutation>|
     -> (Generation, Vec<Option<usize>>) {
        // D_1 par cycle : un cycle de fin de budget peut n'avoir aucun draft.
        let seen = std::cell::RefCell::new(Vec::new());
        let record = |cycle: usize, _: usize, rank: usize, draft: usize| {
            if rank == 0 {
                let mut seen = seen.borrow_mut();
                if seen.len() <= cycle {
                    seen.resize(cycle + 1, None);
                }
                seen[cycle] = Some(draft);
            }
            draft
        };
        let options = SpecOptions {
            overrides: Some(&record),
            mutation,
            ..SpecOptions::default()
        };
        let out = generate_speculative(session, prompt, generated, 3, &[], &options)
            .expect("invariant: spéculatif");
        (out, seen.into_inner())
    };
    for index in [1, 6] {
        let (label, prompt) = &prompts[index];
        let (spec, drafts) = first_drafts(&mut session, prompt, None);
        let mut sequence = prompt.clone();
        sequence.extend_from_slice(&spec.tokens);
        let reference = reference_drafts(&mut session, &sequence);
        // Position p du cycle c : p_0 = |prompt| − 1, p_{c+1} = p_c + 1 + k_c.
        let mut p = prompt.len() - 1;
        let mut mismatches = 0;
        for (c, &k) in spec.accepted.iter().enumerate() {
            mismatches += usize::from(
                drafts
                    .get(c)
                    .copied()
                    .flatten()
                    .is_some_and(|draft| draft != reference[p]),
            );
            p += 1 + k;
        }
        eprintln!(
            "{label} : D_1 de {} cycles contre le rejeu des paires AR, {mismatches} écarts",
            spec.accepted.len()
        );
        assert_eq!(mismatches, 0, "{label} : drafts ≠ historique AR");
        for mutation in [SpecMutation::NoTruncate, SpecMutation::StaleHistory] {
            let (mutated, drafts) = first_drafts(&mut session, prompt, Some(mutation));
            assert_eq!(
                mutated.tokens, spec.tokens,
                "{label} : jetons sous {mutation:?}"
            );
            let mut p = prompt.len() - 1;
            let mut mismatches = 0;
            for (c, &k) in mutated.accepted.iter().enumerate() {
                mismatches += usize::from(
                    drafts
                        .get(c)
                        .copied()
                        .flatten()
                        .is_some_and(|draft| draft != reference[p]),
                );
                p += 1 + k;
            }
            eprintln!("{label}, mutation {mutation:?} : {mismatches} écarts de D_1");
            assert!(
                mismatches > 0,
                "{label} : mutation {mutation:?} non détectée"
            );
        }
    }

    // Oracle d'état de l'historique MTP (KV, clés poolées et brutes) contre le
    // rejeu séquentiel des paires AR, en régime dense (512) et creux (3000).
    // Les drafts enchaînés écrivent l'anneau des clés brutes : sans
    // restauration, un bloc est poolé avec la clé d'une ligne retirée.
    let mtp_state =
        |session: &mut FlashNextSession| session.mtp_state_digest().expect("invariant: état MTP");
    // Prompt creux : 64 jetons, pour que la mutation NoTruncate (lignes de
    // drafts jamais retirées) tienne dans la capacité de la session.
    for (index, budget) in [(1, generated), (3, 64)] {
        let (label, prompt) = &prompts[index];
        let spec = generate_speculative(&mut session, prompt, budget, 3, &[], &none)
            .expect("invariant: spéculatif");
        let state = mtp_state(&mut session);
        let mut sequence = prompt.clone();
        sequence.extend_from_slice(&spec.tokens);
        // Le dernier jeton émis n'est pas consommé : l'historique s'arrête à
        // la paire de l'avant-dernier.
        reference_drafts(&mut session, &sequence[..sequence.len() - 1]);
        let reference = mtp_state(&mut session);
        assert!(state == reference, "{label} : état MTP ≠ rejeu séquentiel");
        for mutation in [
            SpecMutation::NoRawRestore,
            SpecMutation::NoTruncate,
            SpecMutation::StaleHistory,
        ] {
            let options = SpecOptions {
                mutation: Some(mutation),
                ..SpecOptions::default()
            };
            generate_speculative(&mut session, prompt, budget, 3, &[], &options)
                .expect("invariant: spéculatif muté");
            assert!(
                mtp_state(&mut session) != reference,
                "{label} : mutation {mutation:?} non détectée par l'état MTP"
            );
        }
        eprintln!("{label} : état MTP égal au rejeu séquentiel, 3 mutations détectées");
    }

    // Échec d'encodage de la vérification au 3ᵉ cycle, après les drafts :
    // l'historique MTP revient à ses lignes vérifiées.
    let (_, prompt) = &prompts[1];
    let failing = SpecOptions {
        fail_verify_cycle: Some(3),
        ..SpecOptions::default()
    };
    assert!(generate_speculative(&mut session, prompt, 64, 3, &[], &failing).is_err());
    assert_eq!(
        session.mtp_rows(),
        Some(session.position()),
        "historique MTP après un échec de vérification"
    );
    // Échec de la reconstruction de l'historique après deux cycles d'acceptation
    // complète : au cycle 2, l'historique propre a `|prompt| + 8` lignes et le
    // tronc est à `|prompt| + 12`. Échec à la 2ᵉ paire sur un préfixe aligné
    // (512), puis à la 3ᵉ sur un préfixe congru à 3 modulo 4 (515) : la
    // reconstruction complète le bloc en cours et écrit une clé brute du bloc
    // suivant avant d'échouer ; l'anneau doit être restauré.
    for (length, pair) in [(512, 1), (515, 2)] {
        let prompt = &ids[..length];
        let reference = generate_ar(&mut session, prompt, 64, &[]).expect("invariant: AR");
        let oracle = |_: usize, emitted: usize, rank: usize, _: usize| {
            *reference
                .tokens
                .get(emitted + 1 + rank)
                .expect("invariant: draft d'oracle dans la sortie AR")
        };
        let failing = SpecOptions {
            overrides: Some(&oracle),
            fail_rebuild: Some((2, pair)),
            ..SpecOptions::default()
        };
        let error = generate_speculative(&mut session, prompt, 64, 3, &[], &failing)
            .err()
            .expect("invariant: échec injecté");
        assert!(
            error.to_string().contains("injecté"),
            "{length} : erreur inattendue {error}"
        );
        assert_eq!(
            session.mtp_rows(),
            Some(length + 8),
            "{length} : lignes MTP"
        );
        assert_eq!(
            session.position(),
            length + 12,
            "{length} : position du tronc"
        );
        let state = mtp_state(&mut session);
        let mut sequence = prompt.to_vec();
        sequence.extend_from_slice(&reference.tokens);
        reference_drafts(&mut session, &sequence[..=length + 8]);
        assert!(
            mtp_state(&mut session) == state,
            "{length} : historique MTP après un échec de reconstruction"
        );
    }
    let (_, prompt) = &prompts[1];

    // Draft d'arrêt accepté : jeton d'arrêt = un D_1 accepté dont c'est la
    // première occurrence dans la sortie.
    let (spec, drafts) = first_drafts(&mut session, prompt, None);
    let mut emitted = 0;
    let mut accepted_stop = None;
    for (c, &k) in spec.accepted.iter().enumerate() {
        let index = emitted + 1;
        let draft = drafts.get(c).copied().flatten();
        if let Some(draft) =
            draft.filter(|&d| k >= 1 && spec.tokens.iter().position(|&t| t == d) == Some(index))
        {
            accepted_stop = Some(draft);
            break;
        }
        emitted += 1 + k;
    }
    let stop = [accepted_stop.expect("invariant: un draft accepté en première occurrence")];
    check(
        &mut session,
        "draft d'arrêt accepté",
        prompt,
        generated,
        3,
        &stop,
        &none,
    );
    // Draft d'arrêt rejeté : un jeton jamais émis, imposé comme draft.
    let full = generate_ar(&mut session, prompt, 64, &[]).expect("invariant: AR");
    let never = (0..)
        .find(|t| !full.tokens.contains(t))
        .expect("invariant: jeton libre");
    let stop = [never];
    let reject = |cycle: usize, _: usize, rank: usize, draft: usize| {
        if cycle == 2 && rank == 0 {
            never
        } else {
            draft
        }
    };
    let options = SpecOptions {
        overrides: Some(&reject),
        ..SpecOptions::default()
    };
    let spec = check(
        &mut session,
        "draft d'arrêt rejeté",
        prompt,
        64,
        3,
        &stop,
        &options,
    );
    same("draft d'arrêt rejeté (sans arrêt)", &full, &spec);
    // Profondeur nulle (AR pur) et budgets de 0 à 3 jetons.
    for (depth, budget) in [(0, 32), (3, 0), (3, 1), (3, 2), (3, 3)] {
        check(
            &mut session,
            &format!("profondeur {depth}, budget {budget}"),
            prompt,
            budget,
            depth,
            &[],
            &none,
        );
    }
    assert!(generate_speculative(&mut session, prompt, 8, usize::MAX, &[], &none).is_err());
    eprintln!(
        "identité spéculative : tout vert en {:.0} s",
        started.elapsed().as_secs_f64()
    );
}

/// Étape 6 D, mesure (2) : acceptation d'un draft restreint à un
/// sous-vocabulaire, sur C à profondeur 3. Le sous-vocabulaire (les 32 000
/// jetons les plus fréquents, ou tous les jetons vus, plus EOS) est choisi sur
/// un corpus distinct des prompts d'évaluation : wikitext `corpus32k`,
/// `code4k-2`, `code4k-3`, `chat-1` à `chat-6`. Évaluation : golden 512
/// (prose) et `code4k-1` (code, 1024 jetons), 256 jetons, drafts pleins puis
/// restreints ; identité à l'AR vérifiée. Publié sans porte.
#[test]
#[ignore = "mesure : poids réels, machine libre"]
fn measure_subvocab_draft_acceptance() {
    use super::speculative::{generate_ar, generate_speculative, Generation, SpecOptions};
    use crate::qwen_loader::qwen4_exp::Qwen4ExpCheckpoint;
    let dir = std::path::PathBuf::from(
        std::env::var_os("SARAGOSSA_QWEN4EXP_DIR")
            .expect("invariant: SARAGOSSA_QWEN4EXP_DIR requis"),
    );
    let golden = std::env::var_os("SARAGOSSA_QWEN4EXP_GOLDEN")
        .expect("invariant: SARAGOSSA_QWEN4EXP_GOLDEN requis");
    let corpus = std::path::PathBuf::from(
        std::env::var_os("SARAGOSSA_FLASH_NEXT_CORPUS")
            .expect("invariant: SARAGOSSA_FLASH_NEXT_CORPUS requis"),
    );
    let tokenizer =
        tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("invariant: tokenizer");
    let fixture = |name: &str| -> Vec<usize> {
        let text = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/rowsexact")
                .join(name),
        )
        .expect("invariant: fixture");
        tokenizer
            .encode(text, false)
            .expect("invariant: tokenisation")
            .get_ids()
            .iter()
            .map(|&t| t as usize)
            .collect()
    };
    let ckpt = Qwen4ExpCheckpoint::open(&dir).expect("invariant: checkpoint");
    let (vocab, eos) = (ckpt.config.vocab, ckpt.config.eos as usize);
    // Fréquences sur le corpus distinct des prompts d'évaluation.
    let mut counts = vec![0usize; vocab];
    let mut seen_tokens = 0usize;
    let wiki: Vec<usize> = std::fs::read(corpus.join("ids.u32"))
        .expect("invariant: corpus")
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as usize)
        .collect();
    let mut sources = vec![wiki];
    for name in ["code4k-2.txt", "code4k-3.txt"]
        .into_iter()
        .map(String::from)
        .chain((1..=6).map(|i| format!("chat-{i}.txt")))
    {
        sources.push(fixture(&name));
    }
    for &token in sources.iter().flatten() {
        if let Some(count) = counts.get_mut(token) {
            *count += 1;
            seen_tokens += 1;
        }
    }
    let mut ranked: Vec<usize> = (0..vocab).filter(|&t| counts[t] > 0).collect();
    ranked.sort_by(|a, b| counts[*b].cmp(&counts[*a]).then(a.cmp(b)));
    ranked.truncate(32_000);
    let mut mask = vec![false; vocab];
    for &token in &ranked {
        mask[token] = true;
    }
    mask[eos] = true;
    let size = mask.iter().filter(|&&keep| keep).count();
    eprintln!("sous-vocabulaire : {size} jetons ({seen_tokens} jetons de corpus)");

    let ids = n2_input_ids(std::path::Path::new(&golden));
    let code = fixture("code4k-1.txt");
    let prompts = [
        ("prose (golden 512)", ids[..512].to_vec()),
        ("code (code4k-1)", code[..1024.min(code.len())].to_vec()),
    ];
    let generated = 256;
    let mut session = FlashNextSession::open(
        ckpt,
        1024 + generated + 4,
        Some(&crate::MemoryGuard::process()),
    )
    .expect("invariant: session");
    let per_cycle = |g: &Generation| {
        g.accepted.iter().map(|k| k + 1).sum::<usize>() as f64 / g.accepted.len().max(1) as f64
    };
    let restricted = SpecOptions {
        draft_vocab: Some(&mask),
        ..SpecOptions::default()
    };
    for (label, prompt) in &prompts {
        let ar = generate_ar(&mut session, prompt, generated, &[]).expect("invariant: AR");
        let covered = ar.tokens.iter().filter(|&&t| mask[t]).count();
        let full = generate_speculative(
            &mut session,
            prompt,
            generated,
            3,
            &[],
            &SpecOptions::default(),
        )
        .expect("invariant: spéculatif");
        let sub = generate_speculative(&mut session, prompt, generated, 3, &[], &restricted)
            .expect("invariant: spéculatif restreint");
        assert_eq!(full.tokens, ar.tokens, "{label} : drafts pleins ≠ AR");
        assert_eq!(sub.tokens, ar.tokens, "{label} : drafts restreints ≠ AR");
        eprintln!(
            "{label} : sortie AR couverte {covered}/{} ; profondeur 3 : {:.2} jetons par cycle (pleins), {:.2} (restreints)",
            ar.tokens.len(),
            per_cycle(&full),
            per_cycle(&sub)
        );
    }
}

/// Débit du décodage spéculatif en pas mono (étape 6 C), publié sans porte :
/// sans lignes, une vérification de R lignes coûte R pas du tronc, d'où un
/// débit attendu sous l'AR. AR et spéculatif (profondeurs 1 à 3) alternés,
/// 2 répétitions, 256 jetons après un prompt de prose (golden 512) puis de
/// code (code4k-1, 1024 tokens) ; débit = jetons / durée après le prompt.
#[test]
#[ignore = "débit : poids réels, machine libre"]
fn bench_speculative_mono_vs_ar() {
    use super::speculative::{generate_ar, generate_speculative};
    use crate::qwen_loader::qwen4_exp::Qwen4ExpCheckpoint;
    let dir = std::path::PathBuf::from(
        std::env::var_os("SARAGOSSA_QWEN4EXP_DIR")
            .expect("invariant: SARAGOSSA_QWEN4EXP_DIR requis"),
    );
    let golden = std::env::var_os("SARAGOSSA_QWEN4EXP_GOLDEN")
        .expect("invariant: SARAGOSSA_QWEN4EXP_GOLDEN requis");
    let ids = n2_input_ids(std::path::Path::new(&golden));
    let tokenizer =
        tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("invariant: tokenizer");
    let code_text = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/rowsexact/code4k-1.txt"),
    )
    .expect("invariant: fixture");
    let code: Vec<usize> = tokenizer
        .encode(code_text, false)
        .expect("invariant: tokenisation")
        .get_ids()
        .iter()
        .take(1024)
        .map(|&t| t as usize)
        .collect();
    let generated = 256;
    let mut session = FlashNextSession::open(
        Qwen4ExpCheckpoint::open(&dir).expect("invariant: checkpoint"),
        1024 + generated + 4,
        Some(&crate::MemoryGuard::process()),
    )
    .expect("invariant: session");
    for (label, prompt) in [("prose", ids[..512].to_vec()), ("code", code)] {
        for repeat in 0..2 {
            let ar = generate_ar(&mut session, &prompt, generated, &[]).expect("invariant: AR");
            let ar_rate = ar.tokens.len() as f64 / ar.decode.as_secs_f64();
            let mut line = format!("{label}, répétition {repeat} : AR {ar_rate:.1} tok/s");
            for depth in 1..=3 {
                let spec = generate_speculative(
                    &mut session,
                    &prompt,
                    generated,
                    depth,
                    &[],
                    &super::speculative::SpecOptions::default(),
                )
                .expect("invariant: spéculatif");
                assert_eq!(spec.tokens, ar.tokens, "{label} : jetons");
                let rate = spec.tokens.len() as f64 / spec.decode.as_secs_f64();
                let per_cycle = spec.accepted.iter().map(|k| k + 1).sum::<usize>() as f64
                    / spec.accepted.len().max(1) as f64;
                line += &format!(
                    " ; d{depth} {rate:.1} tok/s ({per_cycle:.2} jetons/cycle, ×{:.2})",
                    rate / ar_rate
                );
            }
            eprintln!("{line}");
        }
    }
    // Coût d'un pas MTP rapporté au pas AR (modèle de gain de l'étape 6 D).
    let ar = generate_ar(&mut session, &ids[..512], 64, &[]).expect("invariant: AR");
    let t_ar = ar.decode.as_secs_f64() / ar.tokens.len() as f64;
    let h = vec![0.0_f32; session_width(&session)];
    let started = std::time::Instant::now();
    let steps = 64;
    for &token in &ids[..steps] {
        session
            .mtp_step(super::session::MtpInput::Host(&h), token, false, false)
            .expect("invariant: pas MTP");
    }
    let t_mtp = started.elapsed().as_secs_f64() / steps as f64;
    eprintln!(
        "pas AR {:.2} ms, pas MTP {:.2} ms : T_mtp/T_ar = {:.3}",
        t_ar * 1e3,
        t_mtp * 1e3,
        t_mtp / t_ar
    );
}

/// Profil de l'étape 6 D (sans porte, poids synthétiques) : pour chaque matmul
/// du tronc Flash-Next à ses formes et bits réels, un lot de R lignes
/// (R = 1..4) contre R pas mono, en mode normal puis en vérification greedy
/// stricte. Publie l'exactitude au bit près de chaque ligne et
/// ρ_brique(R) = durée du lot / durée d'un pas mono. Chaque command buffer lit
/// des copies distinctes du poids (≥ 512 Mio, buffers GPU vérifiés distincts,
/// une lecture par copie) pour mesurer depuis la DRAM comme le decode, et non
/// depuis le SLC ; chauffe puis minimum de 3 ; débit mono publié en Go/s. Le
/// hc Down de production (qmv split-K) est mesuré en mono à part. Le lm_head
/// u6 (kernel argmax en lignes à part) est hors profil. Les experts sont
/// mesurés isolément, témoins du kernel MoE fusionné du decode.
#[test]
#[ignore = "profil GPU : machine libre"]
fn profile_rows_matmuls_flash_next_shapes() -> Result<()> {
    use crate::metal_backend::{install_verify_greedy_scope, with_verify_strict_override};
    let Some(executor) = executor()? else {
        return Ok(());
    };
    let queue = executor.device().new_command_queue();
    // (nom, sorties, entrées, bits, nombre par pas de tronc)
    // bits 32 : poids dense f32, témoin (2× les octets) des linéaires bf16.
    let shapes: [(&str, usize, usize, usize, usize); 15] = [
        ("attention q_proj (q + gate)", 12288, 2560, 8, 12),
        ("attention k/v_proj", 512, 2560, 8, 24),
        ("attention o_proj", 2560, 6144, 8, 12),
        ("DeltaNet in_proj_qkv", 10240, 2560, 8, 36),
        ("DeltaNet in_proj_z", 6144, 2560, 8, 36),
        ("DeltaNet out_proj", 2560, 6144, 8, 36),
        ("hc Up", 10240, 320, 8, 96),
        ("expert gate/up (u4)", 640, 2560, 4, 960),
        ("expert down (u4)", 2560, 640, 4, 480),
        ("expert partagé gate/up", 640, 2560, 8, 96),
        ("expert partagé down", 2560, 640, 8, 48),
        ("hc Down (split-K en mono)", 320, 10240, 8, 97),
        ("indexeur index_qk_proj", 640, 2560, 8, 12),
        ("routeur MoE (bf16)", 512, 2560, 32, 48),
        ("DeltaNet in_proj_a/b (bf16)", 48, 2560, 32, 72),
    ];
    let run = |copies: usize,
               encode: &dyn Fn(&metal::ComputeCommandEncoderRef, usize) -> Result<()>|
     -> Result<f64> {
        let command_buffer = queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        for copy in 0..copies {
            encode(encoder, copy)?;
        }
        encoder.end_encoding();
        let started = std::time::Instant::now();
        commit_and_wait(command_buffer)?;
        Ok(started.elapsed().as_secs_f64() / copies as f64)
    };
    for (name, out_dim, in_dim, bits, per_step) in shapes {
        let mut rng = DeterministicF32::new(0x6d + out_dim as u64);
        let linear = if bits == 32 {
            LinearWeight::Dense(Tensor::from_vec(
                vec![out_dim, in_dim],
                rng.fill(out_dim * in_dim, -0.05, 0.05),
            )?)
        } else {
            LinearWeight::AffineQuantized(affine_quantized(&mut rng, out_dim, in_dim, 0.05, bits)?)
        };
        // Octets lus par pas : codes, plus échelles et biais bf16 par groupe.
        let bytes = if bits == 32 {
            out_dim * in_dim * 4
        } else {
            out_dim * in_dim * bits / 8 + out_dim * (in_dim / 64) * 4
        };
        let copies = (512_usize << 20).div_ceil(bytes).clamp(4, 4096);
        let (_linears, weights) = distinct_weights(&executor, &linear, copies)?;
        let weight = &weights[0];
        let x = rng.fill(4 * in_dim, -1.0, 1.0);
        let input = buffer(&executor, &x);
        let mono_in: Vec<_> = (0..4)
            .map(|r| buffer(&executor, &x[r * in_dim..(r + 1) * in_dim]))
            .collect();
        for strict in [false, true] {
            let _greedy = install_verify_greedy_scope(strict);
            with_verify_strict_override(strict, || -> Result<()> {
                let mono_out = zeros(&executor, out_dim);
                let batched_out = zeros(&executor, 4 * out_dim);
                let timed = |rows: usize| -> Result<f64> {
                    let (src, dst) = if rows == 1 {
                        (&mono_in[0], &mono_out)
                    } else {
                        (&input, &batched_out)
                    };
                    run(copies, &|encoder, copy| {
                        executor
                            .encode_matmul_weight_buffers(
                                encoder,
                                src,
                                rows,
                                in_dim,
                                &weights[copy],
                                dst,
                                true,
                            )
                            .map(|_| ())
                    })
                };
                // Chauffe, puis minimum de 3 tours alternés mono / lots.
                for rows in 1..=4 {
                    timed(rows)?;
                }
                let mut best = [f64::INFINITY; 5];
                for _ in 0..3 {
                    for (rows, best) in best.iter_mut().enumerate().skip(1) {
                        *best = best.min(timed(rows)?);
                    }
                }
                let mono = best[1];
                let mut line = format!(
                    "{name} [{out_dim}×{in_dim}, u{bits}, ×{per_step}/pas] {} : mono {:.1} µs ({:.0} Go/s, {copies} copies)",
                    if strict { "strict" } else { "normal" },
                    mono * 1e6,
                    bytes as f64 / mono / 1e9
                );
                for (rows, &batched) in best.iter().enumerate().skip(2) {
                    let out = zeros(&executor, rows * out_dim);
                    // Exactitude : chaque ligne contre son pas mono.
                    let command_buffer = queue.new_command_buffer();
                    let encoder = command_buffer.new_compute_command_encoder();
                    let singles: Vec<_> = (0..rows).map(|_| zeros(&executor, out_dim)).collect();
                    for r in 0..rows {
                        executor.encode_matmul_weight_buffers(
                            encoder,
                            &mono_in[r],
                            1,
                            in_dim,
                            weight,
                            &singles[r],
                            true,
                        )?;
                    }
                    executor.encode_matmul_weight_buffers(
                        encoder, &input, rows, in_dim, weight, &out, true,
                    )?;
                    encoder.end_encoding();
                    commit_and_wait(command_buffer)?;
                    let all = read_f32_buffer(&out, rows * out_dim)?;
                    let mut exact = true;
                    for (r, single) in singles.iter().enumerate() {
                        let mono = read_f32_buffer(single, out_dim)?;
                        exact &= mono
                            .iter()
                            .zip(&all[r * out_dim..(r + 1) * out_dim])
                            .all(|(a, b)| a.to_bits() == b.to_bits());
                    }
                    line += &format!(
                        " ; R{rows} ρ {:.2}{}",
                        batched / mono,
                        if exact { " exact" } else { " NON exact" }
                    );
                }
                eprintln!("{line}");
                Ok(())
            })?;
        }
    }
    // hc Down de production : qmv u8 split-K de `encode_hc_mix` (même
    // dispatch), mono ; la réduction des partiels reste dans le reste du pas.
    let kernels = FlashNextKernels::new(executor.device())?;
    let (out_dim, in_dim) = (320, 10_240);
    let mut rng = DeterministicF32::new(0x6e);
    let linear =
        LinearWeight::AffineQuantized(affine_quantized(&mut rng, out_dim, in_dim, 0.05, 8)?);
    let bytes = out_dim * in_dim + out_dim * (in_dim / 64) * 4;
    let copies = (512_usize << 20).div_ceil(bytes);
    let (_linears, weights) = distinct_weights(&executor, &linear, copies)?;
    let normed = buffer(&executor, &rng.fill(in_dim, -1.0, 1.0));
    let partial = zeros(&executor, out_dim * super::hc::DOWN_CHUNKS);
    let encode = |encoder: &metal::ComputeCommandEncoderRef, copy: usize| -> Result<()> {
        let MetalLinearWeightBuffers::AffineQuantized {
            packed,
            packed_offset,
            scales,
            scales_offset,
            biases,
            biases_offset,
            packed_cols,
            groups,
            ..
        } = &weights[copy]
        else {
            return Err(InferError::Config(
                "hc Down : poids non quantifié".to_string(),
            ));
        };
        encoder.set_compute_pipeline_state(&kernels.qmv_u8_splitk);
        encoder.set_buffer(0, Some(&normed), 0);
        encoder.set_buffer(1, Some(packed), *packed_offset);
        encoder.set_buffer(2, Some(scales), *scales_offset);
        encoder.set_buffer(3, Some(biases), *biases_offset);
        encoder.set_buffer(4, Some(&partial), 0);
        super::set_value(
            encoder,
            5,
            &[
                out_dim as u32,
                in_dim as u32,
                *packed_cols as u32,
                *groups as u32,
            ],
        );
        super::set_value(encoder, 6, &(super::hc::DOWN_CHUNKS as u32));
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(super::hc::DOWN_CHUNKS as u64, out_dim.div_ceil(8) as u64, 1),
            metal::MTLSize::new(256, 1, 1),
        );
        crate::metal_backend::post_dispatch_barrier(encoder);
        Ok(())
    };
    run(copies, &encode)?;
    let mut best = f64::INFINITY;
    for _ in 0..3 {
        best = best.min(run(copies, &encode)?);
    }
    eprintln!(
        "hc Down de production (split-K) [{out_dim}×{in_dim}, u8, ×97/pas] : mono {:.1} µs ({:.0} Go/s, {copies} copies)",
        best * 1e6,
        bytes as f64 / best / 1e9
    );
    Ok(())
}

/// `copies` poids GPU distincts de même contenu que `linear`. Le cache de
/// buffers est indexé par `weight_id` (affine, conservé par `clone`) et par
/// adresse (dense) : chaque copie est un tenseur neuf, rendu avec ses buffers
/// pour rester vivant pendant la mesure. Les adresses GPU sont vérifiées
/// distinctes.
fn distinct_weights(
    executor: &MetalExecutor,
    linear: &LinearWeight,
    copies: usize,
) -> Result<(Vec<LinearWeight>, Vec<MetalLinearWeightBuffers>)> {
    let linears = (0..copies)
        .map(|_| -> Result<LinearWeight> {
            Ok(match linear {
                LinearWeight::Dense(t) => {
                    LinearWeight::Dense(Tensor::from_vec(t.shape().to_vec(), t.data().to_vec())?)
                }
                LinearWeight::DenseBf16(t) => LinearWeight::Dense(t.to_tensor()?),
                LinearWeight::AffineQuantized(q) => {
                    LinearWeight::AffineQuantized(AffineQuantizedTensor::new(
                        q.packed_shape(),
                        q.packed_data().to_vec(),
                        Tensor::from_vec(q.scales_shape().to_vec(), q.scales_f32().to_vec())?,
                        Tensor::from_vec(q.biases_shape().to_vec(), q.biases_f32().to_vec())?,
                        q.group_size(),
                        q.bits(),
                    )?)
                }
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let weights = linears
        .iter()
        .map(|l| executor.resolve_linear_weight_buffers(l, "flash_next_profile"))
        .collect::<Result<Vec<_>>>()?;
    let address = |w: &MetalLinearWeightBuffers| match w {
        MetalLinearWeightBuffers::DenseBf16 { rhs, .. }
        | MetalLinearWeightBuffers::Dense { rhs, .. } => rhs.gpu_address(),
        MetalLinearWeightBuffers::AffineQuantized { packed, .. } => packed.gpu_address(),
    };
    let distinct: std::collections::HashSet<u64> = weights.iter().map(address).collect();
    if distinct.len() != copies {
        return Err(InferError::Metal(format!(
            "profil : {} buffers distincts pour {copies} copies",
            distinct.len()
        )));
    }
    Ok((linears, weights))
}

/// Largeur hc d'une session (oracles).
fn session_width(session: &FlashNextSession) -> usize {
    session.hc_width()
}
