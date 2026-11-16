//! Oracles synthétiques et mesures des formats à micro-échelles.
use super::*;
use crate::{MicroMode, MicroQuantizedTensor};

fn weight(mode: MicroMode, n: usize, k: usize) -> Result<MicroQuantizedTensor> {
    let lanes = 32 / mode.bits();
    let words = (0..n * k / lanes)
        .map(|i| {
            (0..lanes).fold(0u32, |word, j| {
                let code = if mode.bits() == 4 {
                    ((i * lanes + j) % 16) as u32
                } else {
                    ((i * lanes + j) % 256) as u32
                };
                word | (code << (j * mode.bits()))
            })
        })
        .collect();
    let scales = (0..n * k / mode.group_size())
        .map(|i| {
            if mode == MicroMode::Nvfp4 {
                [48, 56, 64][i % 3]
            } else {
                [126, 127, 128][i % 3]
            }
        })
        .collect();
    MicroQuantizedTensor::new([n, k], words, scales, mode)
}

#[test]
#[ignore = "oracle micro-échelles : nécessite Metal, lancer sous verrou GPU"]
fn micro_dequant_and_matmul_bit_exact() -> Result<()> {
    let metal = MetalExecutor::new()?;
    for mode in [MicroMode::Mxfp4, MicroMode::Nvfp4, MicroMode::Mxfp8] {
        let w = weight(mode, 67, 256)?;
        // Le setup résident replie les concats incompatibles uniquement sur
        // Dimension ; des overrides MLX par projection doivent rester valides.
        let source = LinearWeight::MicroScaled(w.clone());
        let other_mode = if mode == MicroMode::Mxfp4 {
            MicroMode::Nvfp4
        } else {
            MicroMode::Mxfp4
        };
        let different_mode = LinearWeight::MicroScaled(weight(other_mode, 3, 256)?);
        let dense = LinearWeight::Dense(Tensor::from_vec(vec![3, 256], vec![0.0; 3 * 256])?);
        for other in [&different_mode, &dense] {
            for _ in 0..2 {
                assert!(matches!(
                    metal.resolve_concat_linear_weight_buffers(
                        &[&source, other],
                        "micro_mixed_concat"
                    ),
                    Err(InferError::Dimension(_))
                ));
            }
        }
        let buffers = metal.resolve_micro(&w)?;
        let expected = w.dequantize()?;
        let out = metal.new_f32_buffer(expected.len(), "micro_dequant_test")?;
        let pipeline = metal.micro_pipeline(&format!("micro_dequant_{}", mode.name()))?;
        let cb = metal.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&buffers.packed), 0);
        enc.set_buffer(1, Some(&buffers.scales), 0);
        enc.set_buffer(2, Some(&out), 0);
        set_u32_bytes(enc, 3, &[expected.len() as u32], "micro_count")?;
        enc.dispatch_threads(
            MTLSize::new(expected.len() as u64, 1, 1),
            MTLSize::new(128, 1, 1),
        );
        enc.end_encoding();
        commit_and_wait(cb)?;
        let actual = read_f32_buffer(&out, expected.len())?;
        assert_eq!(
            actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            expected
                .data()
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            "{} dequant",
            mode.name()
        );
        // Entrées dyadiques et sommes petites : aucune dépendance à l'ordre
        // d'accumulation. Les lots 1/3 exercent QMV, 16/65 le GEMM NAX et ses bords.
        for m in [1, 3, 16, 65] {
            let input = Tensor::from_vec(
                vec![m, 256],
                (0..m * 256).map(|i| ((i % 3) as f32 - 1.0) * 0.5).collect(),
            )?;
            let cpu = w.matmul_rhs_t(&input)?;
            let gpu = metal.matmul_rhs_t_micro(&input, &w)?;
            assert_eq!(
                gpu.data().iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                cpu.data().iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                "{} M={m}",
                mode.name()
            );
        }
        // Stockage partagé : même contenu CPU, buffers Metal réutilisés tels quels.
        let mut shared = w.clone();
        shared.move_to_metal_shared(&metal)?;
        assert_eq!(shared, w);
        let views = shared.metal_views().expect("invariant: poids déplacé");
        let resolved = metal.resolve_micro(&shared)?;
        assert_eq!(resolved.packed.gpu_address(), views.0.gpu_address());
        assert_eq!(resolved.scales.gpu_address(), views.1.gpu_address());
        let input = Tensor::from_vec(
            vec![16, 256],
            (0..16 * 256)
                .map(|i| ((i % 3) as f32 - 1.0) * 0.5)
                .collect(),
        )?;
        assert_eq!(
            metal.matmul_rhs_t_micro(&input, &shared)?.data(),
            w.matmul_rhs_t(&input)?.data()
        );
        // K court non multiple de 64 : repli QMV sûr, même pour un lot long.
        let w = weight(mode, 5, mode.group_size())?;
        let x = Tensor::from_vec(
            vec![17, mode.group_size()],
            vec![0.5; 17 * mode.group_size()],
        )?;
        assert_eq!(
            metal.matmul_rhs_t_micro(&x, &w)?.data(),
            w.matmul_rhs_t(&x)?.data()
        );
    }
    Ok(())
}

/// MLP micro fusionné (gate+up+SwiGLU, H bf16, down sur H bf16) égal au bit
/// au chemin séparé du prefill : gate, up, `swiglu_f32`, conversion bf16, down.
/// Entrées non dyadiques et lots 16/65/130 : bords M, exp et arrondis exercés.
#[test]
#[ignore = "oracle MLP micro : nécessite Metal, lancer sous verrou GPU"]
fn micro_fused_mlp_bit_exact_vs_unfused() -> Result<()> {
    let metal = MetalExecutor::new()?;
    let (hidden, inter) = (256_usize, 192_usize);
    for mode in [MicroMode::Mxfp4, MicroMode::Nvfp4, MicroMode::Mxfp8] {
        let weights = [
            weight(mode, inter, hidden)?,
            weight(mode, inter, hidden)?,
            weight(mode, hidden, inter)?,
        ];
        let [g, u, d] = [
            metal.resolve_micro(&weights[0])?,
            metal.resolve_micro(&weights[1])?,
            metal.resolve_micro(&weights[2])?,
        ];
        let [gate, up, down] =
            [g.clone(), u.clone(), d.clone()].map(MetalLinearWeightBuffers::MicroScaled);
        for m in [16_usize, 65, 130] {
            assert!(metal.fused_dense_mlp_supported(m, hidden, &gate, &up, &down, inter));
            let x: Vec<f32> = (0..m * hidden)
                .map(|i| ((i * 7919 % 1013) as f32 / 1013.0 - 0.5) * 0.37)
                .collect();
            let lhs = metal.upload_f32_buffer(&x, "micro_mlp_lhs")?;
            let fused = metal.new_f32_buffer(m * hidden, "micro_mlp_fused")?;
            let gate_out = metal.new_f32_buffer(m * inter, "micro_mlp_gate")?;
            let up_out = metal.new_f32_buffer(m * inter, "micro_mlp_up")?;
            let h = metal.new_f32_buffer(m * inter, "micro_mlp_h")?;
            let split = metal.new_f32_buffer(m * hidden, "micro_mlp_split")?;
            let cb = metal.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            metal.encode_fused_dense_mlp(enc, &lhs, m, hidden, &gate, &up, &down, inter, &fused)?;
            metal.encode_micro(enc, &lhs, &gate_out, m, hidden, &g)?;
            metal.encode_micro(enc, &lhs, &up_out, m, hidden, &u)?;
            metal.encode_swiglu(enc, &mut Vec::new(), &gate_out, &up_out, &h, m * inter)?;
            metal.encode_micro(enc, &h, &split, m, inter, &d)?;
            enc.end_encoding();
            commit_and_wait(cb)?;
            let bits = |b: &Buffer, n| -> Result<Vec<u32>> {
                Ok(read_f32_buffer(b, n)?.iter().map(|v| v.to_bits()).collect())
            };
            let (fused, split) = (bits(&fused, m * hidden)?, bits(&split, m * hidden)?);
            let first = fused.iter().zip(&split).position(|(a, b)| a != b);
            let h_g = read_f32_buffer(&gate_out, m * inter)?;
            let h_u = read_f32_buffer(&up_out, m * inter)?;
            let g_max = h_g.iter().fold(0f32, |a, v| a.max(v.abs()));
            assert!(
                first.is_none(),
                "{} M={m} premier écart {first:?} max|gate|={g_max} u0={}",
                mode.name(),
                h_u[0]
            );
        }
    }
    Ok(())
}

#[test]
#[ignore = "oracle des échelles : nécessite Metal, lancer sous verrou GPU"]
fn micro_all_scale_bytes_bit_exact() -> Result<()> {
    let metal = MetalExecutor::new()?;
    for mode in [MicroMode::Mxfp4, MicroMode::Nvfp4, MicroMode::Mxfp8] {
        let k = mode.group_size();
        // Valeur 1, plus petit/grand code non nul et leurs opposés : couvre
        // sous-normaux, signes et débordements, sans comparer des payloads NaN.
        let codes = if mode.bits() == 4 {
            [2, 1, 7, 9, 15]
        } else {
            [56, 1, 127, 129, 255]
        };
        for code in codes {
            let word = if mode.bits() == 4 {
                0x1111_1111 * code
            } else {
                0x0101_0101 * code
            };
            let w = MicroQuantizedTensor::new(
                [256, k],
                vec![word; 256 * k * mode.bits() / 32],
                (0..=255).collect(),
                mode,
            )?;
            let buffers = metal.resolve_micro(&w)?;
            let expected = w.dequantize()?;
            let out = metal.new_f32_buffer(expected.len(), "micro_all_scales")?;
            let pipeline = metal.micro_pipeline(&format!("micro_dequant_{}", mode.name()))?;
            let cb = metal.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_buffer(0, Some(&buffers.packed), 0);
            enc.set_buffer(1, Some(&buffers.scales), 0);
            enc.set_buffer(2, Some(&out), 0);
            set_u32_bytes(enc, 3, &[expected.len() as u32], "micro_count")?;
            enc.dispatch_threads(
                MTLSize::new(expected.len() as u64, 1, 1),
                MTLSize::new(128, 1, 1),
            );
            enc.end_encoding();
            commit_and_wait(cb)?;
            for (i, (gpu, cpu)) in read_f32_buffer(&out, expected.len())?
                .iter()
                .zip(expected.data())
                .enumerate()
            {
                assert_eq!(
                    gpu.to_bits(),
                    cpu.to_bits(),
                    "{} code={code} scale={} column={}",
                    mode.name(),
                    i / k,
                    i % k
                );
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "mesure micro-échelles : GPU au repos, lancer sous verrou"]
fn micro_bandwidth_and_gemm_probe() -> Result<()> {
    let metal = MetalExecutor::new()?;
    eprintln!("micro_device={}", metal.device.name());
    for (n, k) in [(17408, 5120), (5120, 17408), (248320, 5120)] {
        for mode in [MicroMode::Mxfp4, MicroMode::Nvfp4, MicroMode::Mxfp8] {
            let w = weight(mode, n, k)?;
            let buffers = metal.resolve_micro(&w)?;
            for m in if n == 248320 {
                vec![1]
            } else {
                vec![1, 1024, 4096]
            } {
                let lhs = metal.buffer_from_f32(&vec![0.125; m * k], "micro_bench_input")?;
                let out = metal.uncached_f32_buffer(m * n, "micro_bench_output")?;
                let mut times = Vec::new();
                let repeats = if m == 1 { 16 } else { 2 };
                for trial in 0..9 {
                    let cb = metal.queue.new_command_buffer();
                    let enc = cb.new_compute_command_encoder();
                    for _ in 0..repeats {
                        metal.encode_micro(enc, &lhs, &out, m, k, &buffers)?;
                    }
                    enc.end_encoding();
                    let start = std::time::Instant::now();
                    commit_and_wait(cb)?;
                    if trial >= 2 {
                        times.push(start.elapsed().as_secs_f64() / repeats as f64);
                    }
                }
                times.sort_by(f64::total_cmp);
                let sec = times[3];
                let gb = (w.packed().len() * 4 + w.scales().len()) as f64 / sec / 1e9;
                eprintln!("MICRO mode={} M={m} N={n} K={k} ms={:.6} GBps={gb:.3} pct500={:.2} TFLOPS={:.4}", mode.name(), sec * 1e3, gb / 5.0, 2.0 * m as f64 * n as f64 * k as f64 / sec / 1e12);
                assert!(read_f32_buffer(&out, m * n)?.iter().all(|x| x.is_finite()));
            }
        }
        if n != 248320 {
            let w = LinearWeight::AffineQuantized(test_affine(n, k, 0.125)?);
            let buffers = metal.resolve_linear_weight_buffers(&w, "micro_u4_reference")?;
            for m in [1024, 4096] {
                let lhs = metal.buffer_from_f32(&vec![0.125; m * k], "micro_u4_input")?;
                let out = metal.uncached_f32_buffer(m * n, "micro_u4_output")?;
                let mut times = Vec::new();
                for trial in 0..9 {
                    let cb = metal.queue.new_command_buffer();
                    let enc = cb.new_compute_command_encoder();
                    for _ in 0..2 {
                        metal
                            .encode_matmul_weight_buffers(enc, &lhs, m, k, &buffers, &out, true)?;
                    }
                    enc.end_encoding();
                    let start = std::time::Instant::now();
                    commit_and_wait(cb)?;
                    if trial >= 2 {
                        times.push(start.elapsed().as_secs_f64() / 2.0);
                    }
                }
                times.sort_by(f64::total_cmp);
                eprintln!(
                    "MICRO mode=u4 M={m} N={n} K={k} ms={:.6} TFLOPS={:.4}",
                    times[3] * 1e3,
                    2.0 * m as f64 * n as f64 * k as f64 / times[3] / 1e12
                );
            }
        }
    }
    Ok(())
}

/// Bits égaux, ou NaN des deux côtés : les payloads NaN ne sont pas comparés.
fn same_bits(a: &[f32], b: &[f32]) -> Option<usize> {
    a.iter()
        .zip(b)
        .position(|(x, y)| x.to_bits() != y.to_bits() && !(x.is_nan() && y.is_nan()))
}

/// MLP micro fusionné, compilé en fast-math, contre le chemin séparé sur les
/// 256 octets d'échelle de gate et up (sous-normaux, débordements, NaN).
#[test]
#[ignore = "oracle MLP micro : nécessite Metal, lancer sous verrou GPU"]
fn micro_fused_mlp_all_scale_bytes_bit_exact() -> Result<()> {
    let metal = MetalExecutor::new()?;
    let (hidden, inter, m) = (256_usize, 128_usize, 16_usize);
    for mode in [MicroMode::Mxfp4, MicroMode::Nvfp4, MicroMode::Mxfp8] {
        let extreme = |seed: usize| -> Result<MicroQuantizedTensor> {
            let base = weight(mode, inter, hidden)?;
            let scales = (0..inter * hidden / mode.group_size())
                .map(|i| ((i * 37 + seed) % 256) as u8)
                .collect();
            MicroQuantizedTensor::new([inter, hidden], base.packed().to_vec(), scales, mode)
        };
        let (g, u) = (
            metal.resolve_micro(&extreme(0)?)?,
            metal.resolve_micro(&extreme(11)?)?,
        );
        let d = metal.resolve_micro(&weight(mode, hidden, inter)?)?;
        let [gate, up, down] =
            [g.clone(), u.clone(), d.clone()].map(MetalLinearWeightBuffers::MicroScaled);
        assert!(metal.fused_dense_mlp_supported(m, hidden, &gate, &up, &down, inter));
        let x: Vec<f32> = (0..m * hidden)
            .map(|i| ((i * 7919 % 1013) as f32 / 1013.0 - 0.5) * 0.37)
            .collect();
        let lhs = metal.upload_f32_buffer(&x, "micro_mlp_x_lhs")?;
        let fused = metal.new_f32_buffer(m * hidden, "micro_mlp_x_fused")?;
        let gate_out = metal.new_f32_buffer(m * inter, "micro_mlp_x_gate")?;
        let up_out = metal.new_f32_buffer(m * inter, "micro_mlp_x_up")?;
        let h = metal.new_f32_buffer(m * inter, "micro_mlp_x_h")?;
        let split = metal.new_f32_buffer(m * hidden, "micro_mlp_x_split")?;
        let cb = metal.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        metal.encode_fused_dense_mlp(enc, &lhs, m, hidden, &gate, &up, &down, inter, &fused)?;
        metal.encode_micro(enc, &lhs, &gate_out, m, hidden, &g)?;
        metal.encode_micro(enc, &lhs, &up_out, m, hidden, &u)?;
        metal.encode_swiglu(enc, &mut Vec::new(), &gate_out, &up_out, &h, m * inter)?;
        metal.encode_micro(enc, &h, &split, m, inter, &d)?;
        enc.end_encoding();
        commit_and_wait(cb)?;
        let (fused, split) = (
            read_f32_buffer(&fused, m * hidden)?,
            read_f32_buffer(&split, m * hidden)?,
        );
        assert_eq!(
            same_bits(&fused, &split),
            None,
            "{} échelles extrêmes",
            mode.name()
        );
    }
    Ok(())
}

/// QMV micro à quatre lignes égal au bit au QMV à une ligne : lots 1 et 3,
/// N non multiple de 16, tous les octets d'échelle.
#[test]
#[ignore = "oracle QMV micro : nécessite Metal, lancer sous verrou GPU"]
fn micro_qmv_rows4_bit_exact_vs_rows1() -> Result<()> {
    let metal = MetalExecutor::new()?;
    for mode in [MicroMode::Mxfp4, MicroMode::Nvfp4, MicroMode::Mxfp8] {
        let (n, k) = (67_usize, 512_usize);
        let base = weight(mode, n, k)?;
        let scales = (0..n * k / mode.group_size())
            .map(|i| (i % 256) as u8)
            .collect();
        for w in [
            base.clone(),
            MicroQuantizedTensor::new([n, k], base.packed().to_vec(), scales, mode)?,
        ] {
            for m in [1_usize, 3] {
                let x = Tensor::from_vec(
                    vec![m, k],
                    (0..m * k)
                        .map(|i| ((i * 31 % 97) as f32 / 97.0 - 0.5) * 0.3)
                        .collect(),
                )?;
                let one = super::flags::with_micro_qmv_rows4_override(false, || {
                    metal.matmul_rhs_t_micro(&x, &w)
                })?;
                let four = super::flags::with_micro_qmv_rows4_override(true, || {
                    metal.matmul_rhs_t_micro(&x, &w)
                })?;
                assert_eq!(
                    same_bits(four.data(), one.data()),
                    None,
                    "{} M={m}",
                    mode.name()
                );
            }
        }
    }
    Ok(())
}

/// Sonde du déroulement `NAX_KK_UNROLL` kernel par kernel, aux formes du 27B :
/// même source compilée avec et sans la macro.
#[test]
#[ignore = "sonde de performance : nécessite Metal, lancer sous verrou GPU"]
fn nax_kk_unroll_probe() -> Result<()> {
    let metal = MetalExecutor::new()?;
    let src = format!(
        "{}\n{}",
        include_str!("../na_gemm.metal"),
        include_str!("../micro.metal")
    );
    let compile = |prelude: &str| -> Result<metal::Library> {
        metal
            .device
            .new_library_with_source(
                &format!("{prelude}{src}"),
                &super::pipelines::compile_options(true),
            )
            .map_err(InferError::Metal)
    };
    let libs = [
        ("unroll", compile("")?),
        ("plain", compile("#define NAX_KK_UNROLL\n")?),
    ];
    let m = 1024_usize;
    let time = |encode: &dyn Fn(&ComputeCommandEncoderRef)| -> Result<f64> {
        let mut times = Vec::new();
        for trial in 0..9 {
            let cb = metal.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            encode(enc);
            encode(enc);
            enc.end_encoding();
            let start = std::time::Instant::now();
            commit_and_wait(cb)?;
            if trial >= 2 {
                times.push(start.elapsed().as_secs_f64() / 2.0);
            }
        }
        times.sort_by(f64::total_cmp);
        Ok(times[3])
    };
    for (n, k) in [(17408, 5120), (5120, 17408), (1024, 5120)] {
        let u4 = |seed: f32| -> Result<_> {
            let w = LinearWeight::AffineQuantized(test_affine(n, k, seed)?);
            match metal.resolve_linear_weight_buffers(&w, "unroll_u4")? {
                MetalLinearWeightBuffers::AffineQuantized {
                    packed,
                    scales,
                    biases,
                    ..
                } => Ok((packed, scales, biases)),
                _ => Err(InferError::Config("u4 attendu".into())),
            }
        };
        let (gp, gs, gb) = u4(0.125)?;
        let (up, us, ub) = u4(0.25)?;
        let mx = metal.resolve_micro(&weight(MicroMode::Mxfp4, n, k)?)?;
        let mx2 = metal.resolve_micro(&weight(MicroMode::Mxfp4, n, k)?)?;
        let a32 = metal.buffer_from_f32(&vec![0.125; m * k], "unroll_a32")?;
        let a16 = metal.private_bf16_buffer(m * k, "unroll_a16")?;
        let cb = metal.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        metal.encode_f32_to_bf16(enc, &a32, &a16, m * k)?;
        enc.end_encoding();
        commit_and_wait(cb)?;
        let out = metal.uncached_f32_buffer(m * n, "unroll_out")?;
        let mnk = [m as u32, n as u32, k as u32, 0];
        let grid = MTLSize::new(m.div_ceil(64) as u64, n.div_ceil(64) as u64, 1);
        let tg = MTLSize::new(128, 1, 1);
        for (label, lib) in &libs {
            let p = |name: &str| super::pipelines::pipeline(lib, &metal.device, name);
            let kernels: [(&str, ComputePipelineState); 5] = [
                ("tiled_u4", p("gemm_nax_coop_qb_tiled_u4")?),
                ("tiled_u4_x3", p("gemm_nax_coop_qb_tiled_u4_x3")?),
                (
                    "gate_up_swiglu_u4",
                    p("gemm_nax_coop_qb_tiled_u4_gate_up_swiglu")?,
                ),
                ("micro_gemm_mxfp4", p("micro_gemm_mxfp4")?),
                ("micro_gate_up_mxfp4", p("micro_gemm_gate_up_swiglu_mxfp4")?),
            ];
            for (name, pso) in &kernels {
                let sec = time(&|enc: &ComputeCommandEncoderRef| {
                    enc.set_compute_pipeline_state(pso);
                    match *name {
                        "tiled_u4" | "tiled_u4_x3" => {
                            enc.set_buffer(
                                0,
                                Some(if *name == "tiled_u4" { &a16 } else { &a32 }),
                                0,
                            );
                            enc.set_buffer(1, Some(&gp), 0);
                            enc.set_buffer(2, Some(&gs), 0);
                            enc.set_buffer(3, Some(&gb), 0);
                            enc.set_buffer(4, Some(&out), 0);
                            enc.set_bytes(5, 16, mnk.as_ptr().cast());
                        }
                        "gate_up_swiglu_u4" => {
                            enc.set_buffer(0, Some(&a16), 0);
                            for (i, b) in [&gp, &gs, &gb, &up, &us, &ub].into_iter().enumerate() {
                                enc.set_buffer(1 + i as u64, Some(b), 0);
                            }
                            enc.set_buffer(7, Some(&out), 0);
                            enc.set_bytes(8, 16, mnk.as_ptr().cast());
                        }
                        "micro_gemm_mxfp4" => {
                            enc.set_buffer(0, Some(&a16), 0);
                            enc.set_buffer(1, Some(&mx.packed), 0);
                            enc.set_buffer(2, Some(&mx.scales), 0);
                            enc.set_buffer(4, Some(&out), 0);
                            enc.set_bytes(5, 16, mnk.as_ptr().cast());
                        }
                        _ => {
                            enc.set_buffer(0, Some(&a16), 0);
                            enc.set_buffer(1, Some(&mx.packed), 0);
                            enc.set_buffer(2, Some(&mx.scales), 0);
                            enc.set_buffer(3, Some(&mx2.packed), 0);
                            enc.set_buffer(4, Some(&mx2.scales), 0);
                            enc.set_buffer(5, Some(&out), 0);
                            enc.set_bytes(6, 16, mnk.as_ptr().cast());
                        }
                    }
                    enc.dispatch_thread_groups(grid, tg);
                })?;
                let flops = if name.contains("gate_up") { 4.0 } else { 2.0 } * (m * n * k) as f64;
                eprintln!(
                    "UNROLL {label} {name} M={m} N={n} K={k} ms={:.4} TFLOPS={:.2}",
                    sec * 1e3,
                    flops / sec / 1e12
                );
            }
        }
    }
    Ok(())
}
