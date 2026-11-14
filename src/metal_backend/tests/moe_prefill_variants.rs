fn kb_stats(mut v: Vec<f64>) -> (f64, f64, f64) {
    v.sort_by(f64::total_cmp);
    (v[0], v[v.len() / 2], v[v.len() - 1])
}

fn kb_reps() -> usize {
    crate::runtime_flags::env_var("KB_REPS")
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(5)
        .clamp(1, 31)
}

/// MoE routé coop complet, 256/top-8 et 512/top-10, hidden 2048, inter 512,
/// u4 gs64 : 16×32 contre tuilé BM32 (override de test).
#[test]
#[ignore = "kbench"]
fn kb_moe_coop() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    if executor
        .na_gemm_coop_qb_grouped_gate_up_swiglu_u4_tiled
        .is_none()
    {
        eprintln!("KB moe: tuilé indisponible");
        return Ok(());
    }
    for (experts, top_k) in [(256_usize, 8_usize), (512, 10)] {
        let (inter, hidden) = (512_usize, 2048_usize);
        let router = (0..experts * hidden)
            .map(|i| if i / hidden == i % hidden { 1.0 } else { 0.0 })
            .collect::<Vec<f32>>();
        let weights = MetalMoeRoutedWeights {
            router: MetalLinearWeightBuffers::Dense {
                rhs: executor.upload_f32_buffer(&router, "kb_router_w")?,
                rhs_bf16: None,
                out_dim: experts,
                in_dim: hidden,
            },
            stacked: StackedMoeBuffers {
                gate: test_stacked_affine_varied_u4_group(&executor, experts, inter, hidden, 5)?,
                up: test_stacked_affine_varied_u4_group(&executor, experts, inter, hidden, 13)?,
                down: test_stacked_affine_varied_u4_group(&executor, experts, hidden, inter, 23)?,
            },
        };
        let reps = kb_reps();
        // Experts distincts, base décalée pour répartir les slots sur le routeur.
        let select = |t: usize| {
            (0..top_k)
                .map(|j| (t * 97 + j * (experts / top_k)) % experts)
                .collect::<Vec<_>>()
        };
        for rows in [64_usize, 128, 512, 4096] {
            let total = rows * top_k;
            let mut input = Vec::with_capacity(rows * hidden);
            for t in 0..rows {
                let mut row = varied_row(hidden, 300 + t);
                let chosen = select(t);
                for (e, logit) in row.iter_mut().take(experts).enumerate() {
                    *logit = if chosen.contains(&e) { 4.0 } else { -4.0 };
                }
                input.extend_from_slice(&row);
            }
            let input_buf = executor.upload_f32_buffer(&input, "kb_in")?;
            let router_buf = executor.uncached_f32_buffer(rows * experts, "kb_r")?;
            let indices_buf = executor.uncached_u32_buffer(total, "kb_i")?;
            let scores_buf = executor.uncached_f32_buffer(total, "kb_s")?;
            let down_buf = executor.uncached_f32_buffer(total * hidden, "kb_d")?;
            let out_buf = executor.uncached_f32_buffer(rows * hidden, "kb_o")?;
            let flops = 2.0 * total as f64 * (3 * inter * hidden) as f64;
            // u4 : 0,5 o/poids + scale/bias bf16 par groupe de 64 (4 o / 64).
            let wbytes = (experts * 3 * inter * hidden) as f64 * (0.5 + 4.0 / 64.0);
            let mut outs = Vec::new();
            for tiled in [false, true] {
                let mut ts = Vec::new();
                for _ in 0..reps {
                    with_flint_moe_coop_tiled_override(tiled, || -> Result<()> {
                        let cb = executor.queue.new_command_buffer();
                        let enc = cb.new_compute_command_encoder();
                        let guard = EncoderEndGuard::new(enc);
                        let mut owned = Vec::new();
                        executor.encode_moe_routed_rows_coop(
                            enc,
                            &mut owned,
                            &input_buf,
                            None,
                            &out_buf,
                            &router_buf,
                            &indices_buf,
                            &scores_buf,
                            &down_buf,
                            rows,
                            hidden,
                            &weights,
                            top_k,
                            crate::RouterScore::Softmax,
                        )?;
                        guard.end();
                        let t = std::time::Instant::now();
                        commit_and_wait(cb)?;
                        ts.push(t.elapsed().as_secs_f64());
                        Ok(())
                    })?;
                }
                outs.push(read_f32_buffer(&out_buf, rows * hidden)?);
                let (mn, md, mx) = kb_stats(ts);
                eprintln!(
                "KB moe experts={experts} rows={rows} {}: min {:.3} med {:.3} max {:.3} ms, {:.2} TFLOP/s, {:.0} Go/s poids uniques (med)",
                if tiled { "tuile BM32" } else { "coop 16x32" },
                mn * 1e3,
                md * 1e3,
                mx * 1e3,
                flops / md / 1e12,
                wbytes / md / 1e9
            );
            }
            let indices = read_u32_buffer(&indices_buf, total)?;
            let mut counts = vec![0_usize; experts];
            for e in indices {
                counts[e as usize] += 1;
            }
            let padded = |bm| counts.iter().map(|n| n.div_ceil(bm) * bm).sum::<usize>();
            eprintln!(
                "KB padding experts={experts} rows={rows}: slots={total} BM16={} BM32={}",
                padded(16),
                padded(32)
            );
            assert!(outs[0].iter().chain(&outs[1]).all(|x| x.is_finite()));
            assert!(
                outs[0]
                    .iter()
                    .zip(&outs[1])
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "BM32 != BM16"
            );
            let gap = outs[0]
                .iter()
                .zip(&outs[1])
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            eprintln!("KB moe experts={experts} rows={rows} ecart 16x32/tuile max {gap:e}");
        }
    }
    Ok(())
}

/// Couvre le grouping BM32 du chemin partagé, le résidu et sa queue partielle.
#[test]
fn moe_shared_coop_tiled_matches_16x32() -> Result<()> {
    let Some(mut executor) = test_executor()? else {
        return Ok(());
    };
    if !executor.na_library_available_for_test() {
        return Ok(());
    }
    let (experts, hidden, inter, top_k) = (512, 512, 64, 10);
    let dense = |out_dim, in_dim, seed| -> Result<_> {
        Ok(MetalLinearWeightBuffers::Dense {
            rhs: executor.upload_f32_buffer(&varied_row(out_dim * in_dim, seed), "shared_test")?,
            rhs_bf16: None,
            out_dim,
            in_dim,
        })
    };
    let weights = MetalMoeSharedWeights {
        router: dense(experts, hidden, 31)?,
        stacked: StackedMoeBuffers {
            gate: test_stacked_affine_varied_u4_group(&executor, experts, inter, hidden, 5)?,
            up: test_stacked_affine_varied_u4_group(&executor, experts, inter, hidden, 13)?,
            down: test_stacked_affine_varied_u4_group(&executor, experts, hidden, inter, 23)?,
        },
        shared_gate: dense(1, hidden, 37)?,
        shared_gate_proj: dense(inter, hidden, 41)?,
        shared_up_proj: dense(inter, hidden, 43)?,
        shared_down_proj: dense(hidden, inter, 47)?,
    };
    for rows in [5, 31, 33, 129] {
        let input = executor.upload_f32_buffer(&varied_row(rows * hidden, 17), "shared_input")?;
        let residual =
            executor.upload_f32_buffer(&varied_row(rows * hidden, 19), "shared_residual")?;
        let output = executor.uncached_f32_buffer(rows * hidden, "shared_output")?;
        for residual in [None, Some(residual.as_ref())] {
            let run = |executor: &MetalExecutor, tiled, expected_calls| -> Result<Vec<f32>> {
                let before = moe_coop_tiled_trace_for_test();
                with_flint_moe_coop_tiled_override(tiled, || {
                    executor.moe_shared_rows_coop(
                        &input, residual, &output, rows, hidden, &weights, top_k,
                    )
                })?;
                assert_eq!(moe_coop_tiled_trace_for_test() - before, expected_calls);
                read_f32_buffer(&output, rows * hidden)
            };
            let base = run(&executor, false, 0)?;
            let tiled = run(&executor, true, 1)?;
            assert!(base.iter().any(|v| *v != 0.0));
            assert!(base.iter().chain(&tiled).all(|v| v.is_finite()));
            assert!(
                base.iter()
                    .zip(&tiled)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "shared rows={rows}"
            );
            // Un pipeline absent doit revenir au même BM16, grouping compris.
            let pipeline = executor.na_gemm_coop_qb_grouped_scatter_u4_tiled.take();
            let fallback = run(&executor, true, 0)?;
            executor.na_gemm_coop_qb_grouped_scatter_u4_tiled = pipeline;
            assert!(base
                .iter()
                .zip(&fallback)
                .all(|(a, b)| a.to_bits() == b.to_bits()));
        }
    }
    Ok(())
}
