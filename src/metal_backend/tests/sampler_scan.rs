/// Oracle au bit du balayage top-k contre le chemin historique (kill-switch) :
/// mêmes tokens sur des lignes avec ex aequo, zéros signés, -inf, +inf, NaN,
/// -FLT_MAX, dernier bloc de 256 plus court que `top_k` et softcap. Le chemin réellement
/// exécuté est vérifié par ligne : balayage si les logits (après softcap) sont
/// réguliers, repli historique sinon, et une part significative des lignes
/// doit tirer par le balayage.
#[test]
fn sampler_topk_scan_matches_reference_bitwise() -> Result<()> {
    let executor = match MetalExecutor::new() {
        Ok(executor) => executor,
        Err(InferError::Metal(message)) if message.contains("aucun device") => return Ok(()),
        Err(error) => return Err(error),
    };
    let mut seed = 0x2545_F491_4F6C_DD1D_u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let rows = 3_usize;
    let (mut total_rows, mut scanned_rows, mut fallback_rows) = (0_usize, 0_usize, 0_usize);
    for case in 0..180_usize {
        let vocab = [300, 8192, 8193, 20_005, 248_320, 40, 257][case % 7];
        let mut logits: Vec<f32> = (0..vocab * rows)
            .map(|_| {
                let raw = next();
                match case % 5 {
                    0 => (raw % 5) as f32,
                    1 => (raw % 20_000) as f32 / 1000.0 - 10.0,
                    2 => (raw % 64) as f32 * 0.5 - 16.0,
                    3 => (raw % 50) as f32 * 0.3 - 7.0,
                    _ => ((raw % 1000) as f32 / 125.0 - 4.0) * 3.0,
                }
            })
            .collect();
        for value in logits.iter_mut() {
            if next() % 29 == 0 {
                *value = if next() % 2 == 0 { -0.0 } else { 0.0 };
            }
        }
        // Valeurs spéciales sur la ligne 1 seulement.
        let special = case % 4;
        if special != 0 {
            for value in logits[vocab..2 * vocab].iter_mut() {
                if next() % 17 == 0 {
                    *value = match special {
                        1 => f32::NEG_INFINITY,
                        2 => f32::NAN,
                        _ => -f32::MAX,
                    };
                }
            }
            if case % 8 == 3 {
                logits[vocab + vocab / 2] = f32::INFINITY;
            }
        }
        let top_k = [1, 2, 20, 32, 7][case / 7 % 5].min(vocab);
        let temperature = [0.7, 1.0, 0.05, 2.5][case % 4];
        let top_p = [0.95, 1.0, 0.5, 0.0001][case / 4 % 4];
        let softcap = (case % 7 == 0).then_some(30.0);
        let buffer = executor.upload_f32_buffer(&logits, "scan_oracle_logits")?;
        let states: Vec<u64> = (0..rows as u64).map(|row| next() ^ row).collect();
        // Toutes les lignes dans un command buffer, comme la vérification DFlash.
        let run = |scan: bool| -> Result<Vec<u32>> {
            crate::runtime_flags::with_sampler_topk_scan(scan, || {
                let index = executor.new_u32_buffer(rows, "scan_oracle_index")?;
                let cb = executor.queue.new_command_buffer();
                let encoder = cb.new_compute_command_encoder();
                executor.encode_sample_rows_with_states(
                    encoder,
                    &buffer,
                    vocab,
                    &index,
                    0,
                    (temperature, top_p, top_k),
                    &states,
                    softcap,
                )?;
                encoder.end_encoding();
                crate::metal_backend::commit_and_wait(cb)?;
                crate::metal_backend::read_u32_buffer(&index, rows)
            })
        };
        assert_eq!(run(true)?, run(false)?, "cas {case} vocab={vocab} k={top_k}");
        // Chemin réellement pris, une ligne par command buffer : le drapeau
        // `run` de scratch peut être réutilisé d'une ligne à l'autre.
        let path = |row: usize, scan: bool| -> Result<Vec<u32>> {
            crate::runtime_flags::with_sampler_topk_scan(scan, || {
                super::flags::with_private_scratch_override(false, || {
                    let _ = super::lm_head::take_sampler_scan_runs();
                    let row_logits = executor.upload_f32_buffer(
                        &logits[row * vocab..(row + 1) * vocab],
                        "scan_oracle_row",
                    )?;
                    let index = executor.new_u32_buffer(1, "scan_oracle_row_index")?;
                    let cb = executor.queue.new_command_buffer();
                    let encoder = cb.new_compute_command_encoder();
                    executor.encode_sample_rows_with_states(
                        encoder,
                        &row_logits,
                        vocab,
                        &index,
                        0,
                        (temperature, top_p, top_k),
                        &states[row..=row],
                        softcap,
                    )?;
                    encoder.end_encoding();
                    crate::metal_backend::commit_and_wait(cb)?;
                    super::lm_head::take_sampler_scan_runs()
                        .iter()
                        .map(|run| Ok(crate::metal_backend::read_u32_buffer(run, 1)?[0]))
                        .collect()
                })
            })
        };
        let flags = (0..rows)
            .map(|row| path(row, true))
            .collect::<Result<Vec<_>>>()?
            .concat();
        assert!(path(0, false)?.is_empty(), "cas {case} : le kill-switch balaie");
        let eligible = vocab % 256 == 0 || vocab % 256 >= top_k;
        if !eligible {
            assert!(flags.is_empty(), "cas {case} : balayage hors domaine");
            continue;
        }
        assert_eq!(flags.len(), rows, "cas {case} : un balayage par ligne");
        for (row, &flag) in flags.iter().enumerate() {
            let irregular = logits[row * vocab..(row + 1) * vocab].iter().any(|&raw| {
                let value = softcap.map_or(raw, |cap| cap * (raw / cap).tanh());
                !value.is_finite() || value == -f32::MAX
            });
            assert_eq!(flag, u32::from(irregular), "cas {case} ligne {row}");
            total_rows += 1;
            scanned_rows += usize::from(flag == 0);
            fallback_rows += usize::from(flag == 1);
        }
    }
    assert!(
        scanned_rows * 2 > total_rows && fallback_rows > 20,
        "balayage {scanned_rows}, repli {fallback_rows} sur {total_rows} lignes"
    );
    Ok(())
}

/// Mesure du sampler GPU LLM sur un vocabulaire Qwen3.5 (248 320) : une ligne
/// (decode AR) et 16 lignes (vérification DFlash), top-k 20 / top-p 0,95.
/// `cargo test --release --all-features llm_sampler_bench -- --ignored --nocapture`
#[test]
#[ignore = "bench GPU, à lancer sous verrou"]
fn llm_sampler_bench() -> Result<()> {
    let executor = match MetalExecutor::new() {
        Ok(executor) => executor,
        Err(InferError::Metal(message)) if message.contains("aucun device") => return Ok(()),
        Err(error) => return Err(error),
    };
    let vocab = 248_320_usize;
    let rows = 16_usize;
    let logits: Vec<f32> = (0..vocab * rows)
        .map(|i| {
            let hashed = (i as u32).wrapping_mul(2_654_435_761) >> 8;
            (hashed % 10_000) as f32 / 1250.0 - 4.0 + if i % 9973 == 0 { 12.0 } else { 0.0 }
        })
        .collect();
    let buffer = executor.upload_f32_buffer(&logits, "bench_logits")?;
    let index = executor.new_u32_buffer(rows, "bench_index")?;
    // Répartition : le kernel de blocs seul, 16 lignes.
    let partial_len = vocab.div_ceil(256) * 20;
    let partial_values = executor.private_f32_buffer(partial_len, "bench_pv")?;
    let partial_indices = executor.private_u32_buffer(partial_len, "bench_pi")?;
    let mut best = f64::MAX;
    for _ in 0..30 {
        let cb = executor.queue.new_command_buffer();
        let encoder = cb.new_compute_command_encoder();
        for row in 0..rows {
            encoder.set_compute_pipeline_state(&executor.sample_topk_blocks_f32);
            encoder.set_buffer(0, Some(&buffer), (row * vocab * 4) as u64);
            encoder.set_buffer(1, Some(&partial_values), 0);
            encoder.set_buffer(2, Some(&partial_indices), 0);
            let dims = [vocab as u32, 20_u32];
            encoder.set_bytes(3, 8, dims.as_ptr().cast());
            encoder.set_bytes(4, 4, [0.0_f32].as_ptr().cast());
            encoder.set_bytes(5, 4, [1_u32].as_ptr().cast());
            encoder.dispatch_thread_groups(
                metal::MTLSize::new(vocab.div_ceil(256) as u64, 1, 1),
                metal::MTLSize::new(256, 1, 1),
            );
            crate::metal_backend::post_dispatch_barrier(encoder);
        }
        encoder.end_encoding();
        let started = std::time::Instant::now();
        crate::metal_backend::commit_and_wait(cb)?;
        best = best.min(started.elapsed().as_secs_f64());
    }
    println!("llm_sampler blocs_seuls rows={rows} us_par_ligne={:.1}", best * 1e6 / rows as f64);
    for (scan, top_k, top_p) in [(true, 20, 0.95), (false, 20, 0.95), (true, 0, 1.0)] {
        for n in [1_usize, rows] {
            let states: Vec<u64> = (0..n as u64).collect();
            let mut best = f64::MAX;
            for _ in 0..30 {
                let cb = executor.queue.new_command_buffer();
                let encoder = cb.new_compute_command_encoder();
                crate::runtime_flags::with_sampler_topk_scan(scan, || {
                    executor.encode_sample_rows_with_states(
                        encoder,
                        &buffer,
                        vocab,
                        &index,
                        0,
                        (0.7, top_p, top_k),
                        &states,
                        None,
                    )
                })?;
                encoder.end_encoding();
                let started = std::time::Instant::now();
                crate::metal_backend::commit_and_wait(cb)?;
                best = best.min(started.elapsed().as_secs_f64());
            }
            println!(
                "llm_sampler scan={scan} top_k={top_k} top_p={top_p} rows={n} us_total={:.1} us_par_ligne={:.1}",
                best * 1e6,
                best * 1e6 / n as f64
            );
        }
    }
    Ok(())
}
