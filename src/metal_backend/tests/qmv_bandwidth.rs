// Sonde courte sans modèle : A/B scalaire/aligné, ordre alterné, deux warmups
// et sept mesures. Les temps incluent commit/wait, amortis sur 32 dispatches.
// Ce contrôle local ne remplace pas une distribution sur GPU au repos.
#[test]
#[ignore = "sonde bande passante qmv u5/u6, GPU au repos"]
fn qmv_aligned_bandwidth_probe() -> Result<()> {
    let metal = MetalExecutor::new()?;
    eprintln!("device={}", metal.device.name());
    for bits in [5, 6] {
        for (n, k) in [(5120, 17408), (17408, 5120), (248320, 5120)] {
            let cols = k * bits / 32;
            let packed =
                metal.buffer_from_slice(&vec![0xa937_7ac5_u32; n * cols], "probe_packed")?;
            let scales = metal.buffer_from_slice(&vec![0x3a80_u16; n * k / 64], "probe_scales")?;
            let biases = metal.buffer_from_slice(&vec![0xbc80_u16; n * k / 64], "probe_biases")?;
            let lhs = metal.upload_f32_buffer(&vec![0.125; k], "probe_lhs")?;
            let out = metal.uncached_f32_buffer(n, "probe_out")?;
            let dims = [n as u32, k as u32, cols as u32, (k / 64) as u32];
            let mut samples = [Vec::new(), Vec::new()];
            for trial in 0..9 {
                for mode in [trial % 2, 1 - trial % 2] {
                    let (pipeline, rows) = match (bits, mode) {
                        (5, 0) => (&metal.affine_qmv_fast_u5_gs64_f32, 2),
                        (5, _) => (&metal.affine_qmv_fast_aligned_u5_gs64_f32, 8),
                        (6, 0) => (&metal.affine_qmv_fast_u6_gs64_f32, 2),
                        _ => (&metal.affine_qmv_fast_aligned_u6_gs64_f32, 8),
                    };
                    let cb = metal.queue.new_command_buffer();
                    let enc = cb.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(pipeline);
                    enc.set_buffer(0, Some(&lhs), 0);
                    enc.set_buffer(1, Some(&packed), 0);
                    enc.set_buffer(2, Some(&scales), 0);
                    enc.set_buffer(3, Some(&biases), 0);
                    enc.set_buffer(4, Some(&out), 0);
                    set_u32_bytes(enc, 5, &dims, "probe_dims")?;
                    for _ in 0..32 {
                        enc.dispatch_thread_groups(
                            MTLSize::new(1, (n / rows) as u64, 1),
                            MTLSize::new(64, 1, 1),
                        );
                        post_dispatch_barrier_buffer(enc, &out);
                    }
                    enc.end_encoding();
                    let start = std::time::Instant::now();
                    commit_and_wait(cb)?;
                    if trial >= 2 {
                        samples[mode].push(start.elapsed().as_secs_f64() / 32.0);
                    }
                }
            }
            assert!(read_f32_buffer(&out, n)?
                .iter()
                .all(|v| v.is_finite() && *v != 0.0));
            for (mode, values) in samples.iter_mut().enumerate() {
                values.sort_by(f64::total_cmp);
                eprintln!(
                    "u{bits} N={n} K={k} aligned={} median={:.3}ms p95={:.3}ms packed={:.0}GB/s",
                    mode == 1,
                    values[3] * 1e3,
                    values[6] * 1e3,
                    (n * cols * 4) as f64 / values[3] / 1e9
                );
            }
        }
    }
    Ok(())
}
