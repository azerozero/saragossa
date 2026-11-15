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
