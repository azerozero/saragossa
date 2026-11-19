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

/// Médiane (s) de `reps` dispatches identiques par command buffer, deux
/// warmups puis sept mesures.
fn u4_probe_median(
    metal: &MetalExecutor,
    pipeline: &ComputePipelineState,
    buffers: &[&metal::Buffer],
    consts: &[(u64, &[u32])],
    grid: MTLSize,
    reps: usize,
) -> Result<f64> {
    let mut samples = Vec::new();
    for trial in 0..9 {
        let cb = metal.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipeline);
        for (index, buffer) in buffers.iter().enumerate() {
            enc.set_buffer(index as u64, Some(buffer), 0);
        }
        for (index, values) in consts {
            set_u32_bytes(enc, *index, values, "probe_consts")?;
        }
        let out = buffers[buffers.len() - 1];
        for _ in 0..reps {
            enc.dispatch_thread_groups(grid, MTLSize::new(64, 1, 1));
            post_dispatch_barrier_buffer(enc, out);
        }
        enc.end_encoding();
        let start = std::time::Instant::now();
        commit_and_wait(cb)?;
        if trial >= 2 {
            samples.push(start.elapsed().as_secs_f64() / reps as f64);
        }
    }
    samples.sort_by(f64::total_cmp);
    Ok(samples[3])
}

// Débit des qmv u4 gs64 du decode M=1 (27B-4bit dense, 35B-A3B MoE), octets
// packed + scales + biases. Même protocole que la sonde u5/u6.
#[test]
#[ignore = "sonde bande passante qmv u4 dense et MoE, GPU au repos"]
fn qmv_u4_bandwidth_probe() -> Result<()> {
    let metal = MetalExecutor::new()?;
    eprintln!("device={}", metal.device.name());
    let quant = |n: usize, k: usize| -> Result<[metal::Buffer; 3]> {
        Ok([
            metal.buffer_from_slice(&vec![0xa937_7ac5_u32; n * k / 8], "probe_packed")?,
            metal.buffer_from_slice(&vec![0x3a80_u16; n * k / 64], "probe_scales")?,
            metal.buffer_from_slice(&vec![0xbc80_u16; n * k / 64], "probe_biases")?,
        ])
    };
    // Dense : formes du 27B puis du 35B (tête, attention, expert partagé).
    for (n, k) in [
        (5120, 17408),
        (17408, 5120),
        (10240, 5120),
        (6144, 5120),
        (5120, 6144),
        (248_320, 5120),
        (248_320, 2048),
        (8192, 2048),
        (2048, 4096),
        (512, 2048),
        (2048, 512),
    ] {
        let [packed, scales, biases] = quant(n, k)?;
        let lhs = metal.upload_f32_buffer(&vec![0.125; k], "probe_lhs")?;
        let out = metal.uncached_f32_buffer(n, "probe_out")?;
        let dims = [n as u32, k as u32, (k / 8) as u32, (k / 64) as u32];
        let reps = if n * k > 1 << 26 { 8 } else { 64 };
        let t = u4_probe_median(
            &metal,
            &metal.affine_qmv_fast_aligned_u4_gs64_f32,
            &[&lhs, &packed, &scales, &biases, &out],
            &[(5, &dims)],
            MTLSize::new(1, (n / 8) as u64, 1),
            reps,
        )?;
        eprintln!(
            "qmv_u4 N={n} K={k} median={:.4}ms {:.0}GB/s",
            t * 1e3,
            (n * k * 9 / 16) as f64 / t / 1e9
        );
    }
    // MoE 35B-A3B : 256 experts, top-8, gate/up 512×2048 puis down 2048×512.
    let (experts, topk) = (256usize, 8usize);
    let indices: Vec<u32> = (0..topk as u32).map(|i| i * 31 % 256).collect();
    let indices = metal.buffer_from_slice(&indices, "probe_indices")?;
    let (n, k) = (512usize, 2048usize);
    let gate = quant(experts * n, k)?;
    let up = quant(experts * n, k)?;
    let lhs = metal.upload_f32_buffer(&vec![0.125; k], "probe_lhs")?;
    let out = metal.uncached_f32_buffer(topk * n, "probe_out")?;
    let dims = [topk as u32, n as u32, k as u32, (k / 8) as u32];
    let q = [64, 4, (k / 64) as u32, 1];
    let t = u4_probe_median(
        &metal,
        &metal.affine_gather_gate_up_swiglu_fast_u4_gs64_f32,
        &[
            &lhs, &gate[0], &gate[1], &gate[2], &up[0], &up[1], &up[2], &indices, &out,
        ],
        &[(9, &dims), (10, &q), (11, &[0])],
        MTLSize::new(topk as u64, (n / 8) as u64, 1),
        256,
    )?;
    eprintln!(
        "gather_gate_up_u4 top{topk} N={n} K={k} median={:.4}ms {:.0}GB/s",
        t * 1e3,
        (2 * topk * n * k * 9 / 16) as f64 / t / 1e9
    );
    let (n, k) = (2048usize, 512usize);
    let down = quant(experts * n, k)?;
    let lhs = metal.upload_f32_buffer(&vec![0.125; topk * k], "probe_lhs")?;
    let out = metal.uncached_f32_buffer(topk * n, "probe_out")?;
    let dims = [topk as u32, n as u32, k as u32, (k / 8) as u32];
    let q = [64, 4, (k / 64) as u32, topk as u32];
    let t = u4_probe_median(
        &metal,
        &metal.affine_gather_qmv_fast_u4_gs64_f32,
        &[&lhs, &down[0], &down[1], &down[2], &indices, &out],
        &[(6, &dims), (7, &q)],
        MTLSize::new(topk as u64, (n / 8) as u64, 1),
        256,
    )?;
    eprintln!(
        "gather_qmv_u4 top{topk} N={n} K={k} median={:.4}ms {:.0}GB/s",
        t * 1e3,
        (topk * n * k * 9 / 16) as f64 / t / 1e9
    );
    Ok(())
}

// NOTE: la sonde précédente garde une seule copie des poids (servie par le SLC
// sous ~48 Mo) et mesure 8 dispatches par command buffer pour les grosses
// formes (latence commit/wait non amortie). Ici, des copies tournantes
// dépassent 1 Go et chaque command buffer dure ≥ 20 ms : débit DRAM du decode.
#[test]
#[ignore = "sonde bande passante qmv u4 en DRAM, GPU au repos"]
fn qmv_u4_dram_bandwidth_probe() -> Result<()> {
    let metal = MetalExecutor::new()?;
    for (n, k) in [
        (5120, 17408),
        (17408, 5120),
        (10240, 5120),
        (6144, 5120),
        (5120, 6144),
        (248_320, 5120),
    ] {
        let bytes = n * k * 9 / 16;
        let copies = (1usize << 30).div_ceil(bytes).max(2);
        let mut weights = Vec::new();
        for _ in 0..copies {
            weights.push([
                metal.buffer_from_slice(&vec![0xa937_7ac5_u32; n * k / 8], "probe_packed")?,
                metal.buffer_from_slice(&vec![0x3a80_u16; n * k / 64], "probe_scales")?,
                metal.buffer_from_slice(&vec![0xbc80_u16; n * k / 64], "probe_biases")?,
            ]);
        }
        let lhs = metal.upload_f32_buffer(&vec![0.125; k], "probe_lhs")?;
        let out = metal.uncached_f32_buffer(n, "probe_out")?;
        let dims = [n as u32, k as u32, (k / 8) as u32, (k / 64) as u32];
        let reps = (10_000_000_000 / bytes).max(copies) / copies * copies;
        let mut samples = Vec::new();
        for trial in 0..9 {
            let cb = metal.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&metal.affine_qmv_fast_aligned_u4_gs64_f32);
            enc.set_buffer(0, Some(&lhs), 0);
            enc.set_buffer(4, Some(&out), 0);
            set_u32_bytes(enc, 5, &dims, "probe_dims")?;
            for rep in 0..reps {
                let [p, s, b] = &weights[rep % copies];
                enc.set_buffer(1, Some(p), 0);
                enc.set_buffer(2, Some(s), 0);
                enc.set_buffer(3, Some(b), 0);
                enc.dispatch_thread_groups(
                    MTLSize::new(1, (n / 8) as u64, 1),
                    MTLSize::new(64, 1, 1),
                );
                post_dispatch_barrier_buffer(enc, &out);
            }
            enc.end_encoding();
            let start = std::time::Instant::now();
            commit_and_wait(cb)?;
            if trial >= 2 {
                samples.push(start.elapsed().as_secs_f64() / reps as f64);
            }
        }
        samples.sort_by(f64::total_cmp);
        let t = samples[3];
        eprintln!(
            "dram qmv_u4 N={n} K={k} copies={copies} reps={reps} median={:.4}ms {:.0}GB/s",
            t * 1e3,
            bytes as f64 / t / 1e9
        );
    }
    // gate+up+SwiGLU fusionné du 27B (17408×5120, deux matrices).
    let (n, k) = (17408usize, 5120usize);
    let bytes = 2 * n * k * 9 / 16;
    let copies = (1usize << 30).div_ceil(bytes);
    let mut weights = Vec::new();
    for _ in 0..2 * copies {
        weights.push([
            metal.buffer_from_slice(&vec![0xa937_7ac5_u32; n * k / 8], "probe_packed")?,
            metal.buffer_from_slice(&vec![0x3a80_u16; n * k / 64], "probe_scales")?,
            metal.buffer_from_slice(&vec![0xbc80_u16; n * k / 64], "probe_biases")?,
        ]);
    }
    let lhs = metal.upload_f32_buffer(&vec![0.125; k], "probe_lhs")?;
    let out = metal.uncached_f32_buffer(n, "probe_out")?;
    let dims = [n as u32, k as u32, (k / 8) as u32, (k / 64) as u32];
    let reps = (10_000_000_000 / bytes).max(copies) / copies * copies;
    let mut samples = Vec::new();
    for trial in 0..9 {
        let cb = metal.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&metal.affine_gate_up_swiglu_fast_u4_gs64_f32);
        enc.set_buffer(0, Some(&lhs), 0);
        enc.set_buffer(7, Some(&out), 0);
        set_u32_bytes(enc, 8, &dims, "probe_dims")?;
        for rep in 0..reps {
            for (slot, w) in weights[2 * (rep % copies)..][..2].iter().enumerate() {
                for (i, buf) in w.iter().enumerate() {
                    enc.set_buffer(1 + 3 * slot as u64 + i as u64, Some(buf), 0);
                }
            }
            enc.dispatch_thread_groups(MTLSize::new(1, (n / 8) as u64, 1), MTLSize::new(64, 1, 1));
            post_dispatch_barrier_buffer(enc, &out);
        }
        enc.end_encoding();
        let start = std::time::Instant::now();
        commit_and_wait(cb)?;
        if trial >= 2 {
            samples.push(start.elapsed().as_secs_f64() / reps as f64);
        }
    }
    samples.sort_by(f64::total_cmp);
    eprintln!(
        "dram gate_up_swiglu_u4 N={n} K={k} median={:.4}ms {:.0}GB/s",
        samples[3] * 1e3,
        bytes as f64 / samples[3] / 1e9
    );
    Ok(())
}
