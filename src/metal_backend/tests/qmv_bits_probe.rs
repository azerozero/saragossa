// Sonde qmv par largeur de bits via le vrai sélecteur résident
// (`encode_matmul_weight_buffers`) : chaque forme et chaque M passe par le
// kernel que le decode/verify dispatcherait. Les poids tournent sur plusieurs
// copies (≥ 192 Mo) pour que le SLC ne serve pas les petites formes.
#[test]
#[ignore = "sonde bande passante qmv u2..u8 par M, GPU au repos"]
fn qmv_bits_bandwidth_probe() -> Result<()> {
    let metal = MetalExecutor::new()?;
    eprintln!("device={}", metal.device.name());
    let bits_list = [2_usize, 3, 4, 5, 6, 8];
    let batches = [1_usize, 2, 4, 8, 16];
    const REPS: usize = 16;
    for (n, k) in [(5120, 17408), (17408, 5120), (6144, 5120), (248320, 5120)] {
        let groups = k / 64;
        let mut weights = Vec::new();
        for &bits in &bits_list {
            let cols = k * bits / 32;
            let bytes = n * cols * 4 + 4 * n * groups;
            let copies = (192_usize << 20).div_ceil(bytes).clamp(1, 8);
            let mut seed = 0x9e37_79b9_u32 ^ bits as u32;
            let words: Vec<u32> = (0..n * cols)
                .map(|_| {
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    seed
                })
                .collect();
            let mut set = Vec::new();
            for _ in 0..copies {
                set.push(MetalLinearWeightBuffers::AffineQuantized {
                    packed: metal.buffer_from_slice(&words, "probe_packed")?,
                    packed_offset: 0,
                    scales: metal.buffer_from_slice(&vec![0x3a80_u16; n * groups], "probe_s")?,
                    scales_offset: 0,
                    biases: metal.buffer_from_slice(&vec![0xbc80_u16; n * groups], "probe_b")?,
                    biases_offset: 0,
                    out_dim: n,
                    in_dim: k,
                    packed_cols: cols,
                    group_size: 64,
                    bits,
                    groups,
                });
            }
            weights.push((bits, bytes, set));
        }
        let lhs = metal.upload_f32_buffer(&vec![0.125; 16 * k], "probe_lhs")?;
        let out = metal.uncached_f32_buffer(16 * n, "probe_out")?;
        for &m in &batches {
            let mut samples = vec![Vec::new(); weights.len()];
            let mut kernels = vec![String::new(); weights.len()];
            for trial in 0..9 {
                let order: Vec<usize> = if trial % 2 == 0 {
                    (0..weights.len()).collect()
                } else {
                    (0..weights.len()).rev().collect()
                };
                for idx in order {
                    let (_, _, set) = &weights[idx];
                    super::profiling::OFFSET_TEST_DISPATCHES.with(|t| *t.borrow_mut() = Some(Vec::new()));
                    let cb = metal.queue.new_command_buffer();
                    let enc = cb.new_compute_command_encoder();
                    for rep in 0..REPS {
                        metal.encode_matmul_weight_buffers(
                            enc,
                            &lhs,
                            m,
                            k,
                            &set[rep % set.len()],
                            &out,
                            false,
                        )?;
                        post_dispatch_barrier_buffer(enc, &out);
                    }
                    enc.end_encoding();
                    let start = std::time::Instant::now();
                    commit_and_wait(cb)?;
                    let elapsed = start.elapsed().as_secs_f64() / REPS as f64;
                    let trace = super::profiling::OFFSET_TEST_DISPATCHES
                        .with(|t| t.borrow_mut().take())
                        .unwrap_or_default();
                    kernels[idx] = trace.first().cloned().unwrap_or_default();
                    if trial >= 2 {
                        samples[idx].push(elapsed);
                    }
                }
            }
            for (idx, values) in samples.iter_mut().enumerate() {
                values.sort_by(f64::total_cmp);
                let (bits, bytes, _) = &weights[idx];
                let kernel = kernels[idx].split(' ').next().unwrap_or("");
                eprintln!(
                    "PROBE bits={bits} N={n} K={k} M={m} median_us={:.1} p95_us={:.1} gbs={:.0} kernel={kernel}",
                    values[3] * 1e6,
                    values[6] * 1e6,
                    *bytes as f64 / values[3] / 1e9
                );
            }
        }
    }
    Ok(())
}
