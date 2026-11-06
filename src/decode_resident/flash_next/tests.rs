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
