/// Coût T(R) des matmuls u6 gs64 du TTS Qwen3 1,7B pour R flux, sur des poids
/// synthétiques aux formes du talker (28 couches, d 2048, MLP 6144) et du code
/// predictor (5 couches, d 1024, MLP 3072) : R × qmv mono, qmv à R lignes
/// (défaut, bit-exact contre le mono dès R ≥ 3) et `QmmBatchedU6`
/// (`SARAGOSSA_RUST_QMM_U6=1`). Durée GPU d'un command buffer par passe,
/// médiane de 7. Borne le gain d'un TTS en lignes entre moteurs (carte C6).
#[test]
#[ignore = "bench GPU (≈ 1,2 Go de poids synthétiques), lancer avec --ignored --release --nocapture"]
fn tts_u6_rows_cost_bench() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let make = |out_dim: usize, in_dim: usize, seed: u32| -> Result<MetalLinearWeightBuffers> {
        let packed_cols = in_dim * 6 / 32;
        let groups = in_dim / 64;
        let packed: Vec<u32> = (0..out_dim * packed_cols)
            .map(|i| (i as u32 ^ seed).wrapping_mul(2_654_435_761))
            .collect();
        let scales: Vec<f32> = (0..out_dim * groups)
            .map(|i| bf16_round(0.002 + (i % 5) as f32 * 0.0002))
            .collect();
        let biases: Vec<f32> = (0..out_dim * groups)
            .map(|i| bf16_round(-0.01 + (i % 7) as f32 * 0.003))
            .collect();
        Ok(MetalLinearWeightBuffers::AffineQuantized {
            packed: executor.buffer_from_slice(&packed, "tts_rows_bench")?,
            packed_offset: 0,
            scales: executor.buffer_from_f32_as_bf16(&scales, "tts_rows_bench")?,
            scales_offset: 0,
            biases: executor.buffer_from_f32_as_bf16(&biases, "tts_rows_bench")?,
            biases_offset: 0,
            out_dim,
            in_dim,
            packed_cols,
            group_size: 64,
            bits: 6,
            groups,
        })
    };
    // (out, in) d'une couche : q, k, v, o, gate, up, down.
    let layer = |d: usize, kv: usize, mlp: usize| {
        [(2048, d), (kv, d), (kv, d), (d, 2048), (mlp, d), (mlp, d), (d, mlp)]
    };
    let models = [
        ("talker", 28_usize, layer(2048, 1024, 6144)),
        ("predictor", 5, layer(1024, 1024, 3072)),
    ];
    for (name, layers, shapes) in models {
        let weights = (0..layers * shapes.len())
            .map(|i| {
                let (out_dim, in_dim) = shapes[i % shapes.len()];
                make(out_dim, in_dim, i as u32)
            })
            .collect::<Result<Vec<_>>>()?;
        let bytes: usize = weights
            .iter()
            .map(|w| match w {
                MetalLinearWeightBuffers::AffineQuantized { packed, scales, biases, .. } => {
                    (packed.length() + scales.length() + biases.length()) as usize
                }
                _ => 0,
            })
            .sum();
        let lhs: Vec<f32> = (0..6 * 6144).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect();
        let lhs = executor.upload_f32_buffer(&lhs, "tts_rows_bench_lhs")?;
        let out = executor.uncached_f32_buffer(6 * 6144, "tts_rows_bench_out")?;
        let pass = |rows: usize, mono: bool| -> Result<f64> {
            let mut samples = Vec::new();
            for _ in 0..7 {
                let command_buffer = executor.queue.new_command_buffer();
                let encoder = command_buffer.new_compute_command_encoder();
                for weight in &weights {
                    let MetalLinearWeightBuffers::AffineQuantized { in_dim, .. } = weight else {
                        continue;
                    };
                    let (batch, repeat) = if mono { (1, rows) } else { (rows, 1) };
                    for _ in 0..repeat {
                        executor.encode_matmul_weight_buffers(
                            encoder, &lhs, batch, *in_dim, weight, &out, false,
                        )?;
                    }
                }
                encoder.end_encoding();
                commit_and_wait(command_buffer)?;
                samples.push(command_buffer_gpu_ns(command_buffer) as f64 / 1e6);
            }
            samples.sort_by(f64::total_cmp);
            Ok(samples[samples.len() / 2])
        };
        // NOTE: le « mono×R » ci-dessus relit chaque poids R fois de suite, donc
        // chaud en cache. Le serve multi-moteur fait plutôt R passes complètes,
        // une par moteur, sur R files concurrentes, chacune avec sa sortie
        // (comme les scratchs des moteurs). `wall(files, batch)` chronomètre en
        // temps mur, du commit des command buffers à la fin du dernier : R
        // files à 1 ligne (serve actuel) ou 1 file à R lignes, même horloge.
        // `sequential` enchaîne les R passes complètes (durée GPU).
        let queues: Vec<_> = (0..6).map(|_| executor.device.new_command_queue()).collect();
        let outs = (0..6)
            .map(|_| executor.uncached_f32_buffer(6 * 6144, "tts_rows_bench_out"))
            .collect::<Result<Vec<_>>>()?;
        let full_pass = |encoder: &metal::ComputeCommandEncoderRef,
                         batch: usize,
                         out: &metal::BufferRef|
         -> Result<()> {
            for weight in &weights {
                if let MetalLinearWeightBuffers::AffineQuantized { in_dim, .. } = weight {
                    executor.encode_matmul_weight_buffers(
                        encoder, &lhs, batch, *in_dim, weight, out, false,
                    )?;
                }
            }
            Ok(())
        };
        let wall = |files: usize, batch: usize| -> Result<f64> {
            let mut samples = Vec::new();
            for _ in 0..7 {
                let buffers = queues[..files]
                    .iter()
                    .zip(&outs)
                    .map(|(queue, out)| {
                        let command_buffer = queue.new_command_buffer().to_owned();
                        let encoder = command_buffer.new_compute_command_encoder();
                        full_pass(encoder, batch, out)?;
                        encoder.end_encoding();
                        Ok(command_buffer)
                    })
                    .collect::<Result<Vec<_>>>()?;
                let started = std::time::Instant::now();
                buffers.iter().for_each(|buffer| buffer.commit());
                buffers.iter().for_each(|buffer| buffer.wait_until_completed());
                samples.push(started.elapsed().as_secs_f64() * 1e3);
            }
            samples.sort_by(f64::total_cmp);
            Ok(samples[samples.len() / 2])
        };
        let sequential = |rows: usize| -> Result<f64> {
            let mut samples = Vec::new();
            for _ in 0..7 {
                let command_buffer = executor.queue.new_command_buffer();
                let encoder = command_buffer.new_compute_command_encoder();
                for _ in 0..rows {
                    full_pass(encoder, 1, &out)?;
                }
                encoder.end_encoding();
                commit_and_wait(command_buffer)?;
                samples.push(command_buffer_gpu_ns(command_buffer) as f64 / 1e6);
            }
            samples.sort_by(f64::total_cmp);
            Ok(samples[samples.len() / 2])
        };
        // Passe de chauffe : la première mesure paie la montée en fréquence du GPU.
        pass(6, true)?;
        let one = pass(1, true)?;
        eprintln!(
            "tts_rows {name} poids={:.0} Mo R=1 {one:.3} ms ({:.0} Go/s)",
            bytes as f64 / 1e6,
            bytes as f64 / one / 1e6
        );
        for rows in [2_usize, 4, 6] {
            let mono = pass(rows, true)?;
            // Override explicite : un SARAGOSSA_RUST_QMM_U6=1 ambiant mesurerait déjà le QMM.
            let lines = super::flags::with_qmm_u6_override(false, || pass(rows, false))?;
            let qmm = super::flags::with_qmm_u6_override(true, || pass(rows, false))?;
            eprintln!(
                "tts_rows {name} R={rows} mono×R={mono:.3} lignes={lines:.3} qmm_u6={qmm:.3} ms \
                 T(R)/T(1) lignes={:.2} qmm={:.2}",
                lines / one,
                qmm / one
            );
            let parallel = wall(rows, 1)?;
            let lines_wall = super::flags::with_qmm_u6_override(false, || wall(1, rows))?;
            let serial = sequential(rows)?;
            eprintln!(
                "tts_rows {name} R={rows} mur: files_concurrentes={parallel:.3} \
                 lignes={lines_wall:.3} ms concurrentes/lignes={:.2} ; \
                 passes_sequentielles_gpu={serial:.3} ms",
                parallel / lines_wall
            );
        }
    }
    Ok(())
}
