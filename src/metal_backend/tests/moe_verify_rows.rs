// Oracles de la queue MoE de vérification en lignes : chaque kernel en lignes
// doit reproduire, ligne par ligne, le kernel mono-ligne du decode résident.
// Égalité de bits sur la machine de référence (M5), ±1 ULP ailleurs
// (`assert_bits_equal`).

/// Lance `encode` dans un command buffer et attend.
fn run_verify_rows_cb(
    executor: &MetalExecutor,
    encode: impl FnOnce(&ComputeCommandEncoderRef) -> Result<()>,
) -> Result<()> {
    let command_buffer = executor.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encode(encoder)?;
    encoder.end_encoding();
    commit_and_wait(command_buffer)
}

/// Top-k 8/256 en lignes (un threadgroup par ligne) = top-k mono de chaque
/// ligne, indices et poids en bits, y compris avec des égalités forcées.
#[test]
fn topk8_rows_matches_mono_bitwise() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    const ROWS: usize = 9;
    const COUNT: usize = 256;
    const TOPK: usize = 8;
    let mut logits = Vec::with_capacity(ROWS * COUNT);
    for row in 0..ROWS {
        logits.extend(varied_row(COUNT, row * 13 + 1));
    }
    // Ligne 3 : plateau d'égalités au sommet, départagées par l'indice.
    for value in &mut logits[3 * COUNT..3 * COUNT + 40] {
        *value = 2.5;
    }
    // Ligne 7 : le même maximum à des rangs éloignés.
    logits[7 * COUNT + 5] = 9.0;
    logits[7 * COUNT + 250] = 9.0;
    let logits_buf = executor.upload_f32_buffer(&logits, "vr_topk_logits")?;
    let indices = executor.upload_u32_buffer(&[u32::MAX; ROWS * TOPK], "vr_topk_idx")?;
    let scores = executor.upload_f32_buffer(&[-7777.0; ROWS * TOPK], "vr_topk_scores")?;
    run_verify_rows_cb(&executor, |encoder| {
        executor.encode_topk_softmax_verify_rows(
            encoder,
            &logits_buf,
            &indices,
            &scores,
            ROWS,
            COUNT,
            TOPK,
        )
    })?;
    let rows_indices = read_u32_buffer(&indices, ROWS * TOPK)?;
    let rows_scores = read_f32_buffer(&scores, ROWS * TOPK)?;
    for row in 0..ROWS {
        let row_logits =
            executor.upload_f32_buffer(&logits[row * COUNT..(row + 1) * COUNT], "vr_topk_row")?;
        let mono_indices = executor.uncached_u32_buffer(TOPK, "vr_topk_mono_idx")?;
        let mono_scores = executor.uncached_f32_buffer(TOPK, "vr_topk_mono_scores")?;
        run_verify_rows_cb(&executor, |encoder| {
            executor.encode_topk_softmax(
                encoder,
                &mut Vec::new(),
                &row_logits,
                &mono_indices,
                &mono_scores,
                COUNT,
                TOPK,
            )
        })?;
        assert_eq!(
            &rows_indices[row * TOPK..(row + 1) * TOPK],
            read_u32_buffer(&mono_indices, TOPK)?.as_slice(),
            "top-k ligne {row}: indices"
        );
        assert_bits_equal(
            &rows_scores[row * TOPK..(row + 1) * TOPK],
            &read_f32_buffer(&mono_scores, TOPK)?,
            &format!("top-k ligne {row}: poids"),
        );
    }
    Ok(())
}

/// Gate+up+SwiGLU partagé en lignes (grille x = lignes) = kernel mono de
/// chaque ligne, sur la forme partagée du 35B (512×2048, u4 gs64).
#[test]
fn shared_gate_up_rows_matches_mono_bitwise() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    const ROWS: usize = 9;
    let (out_dim, in_dim) = (512_usize, 2048_usize);
    let gate = executor.resolve_linear_weight_buffers(
        &LinearWeight::AffineQuantized(test_affine_varied(out_dim, in_dim)?),
        "vr_shared_gate",
    )?;
    let up_affine = {
        let affine = test_affine_varied(out_dim, in_dim)?;
        let scales = affine
            .scales_f32()
            .as_ref()
            .iter()
            .map(|scale| bf16_round(scale * 1.5))
            .collect();
        AffineQuantizedTensor::new(
            &[out_dim, in_dim / 8],
            affine.packed_data().to_vec(),
            Tensor::from_vec(vec![out_dim, in_dim / 64], scales)?,
            Tensor::from_vec(
                affine.biases_shape().to_vec(),
                affine.biases_f32().into_owned(),
            )?,
            64,
            4,
        )?
    };
    let up = executor
        .resolve_linear_weight_buffers(&LinearWeight::AffineQuantized(up_affine), "vr_shared_up")?;
    let mut lhs = Vec::with_capacity(ROWS * in_dim);
    for row in 0..ROWS {
        lhs.extend(varied_row(in_dim, 40 + row));
    }
    let lhs_buf = executor.upload_f32_buffer(&lhs, "vr_shared_lhs")?;
    let out = executor.upload_f32_buffer(&vec![-7777.0; ROWS * out_dim], "vr_shared_out")?;
    let mut engaged = false;
    run_verify_rows_cb(&executor, |encoder| {
        engaged = executor.encode_gate_up_swiglu_fast_buffers_rows(
            encoder, &lhs_buf, &gate, &up, &out, ROWS, in_dim,
        )?;
        Ok(())
    })?;
    assert!(engaged, "gate+up partagé en lignes non engagé (u4 gs64)");
    let rows_out = read_f32_buffer(&out, ROWS * out_dim)?;
    for row in 0..ROWS {
        let row_buf =
            executor.upload_f32_buffer(&lhs[row * in_dim..(row + 1) * in_dim], "vr_shared_row")?;
        let mono_out = executor.uncached_f32_buffer(out_dim, "vr_shared_mono")?;
        run_verify_rows_cb(&executor, |encoder| {
            executor
                .encode_gate_up_swiglu_fast_buffers(
                    encoder, &row_buf, &gate, &up, &mono_out, in_dim,
                )
                .map(|_| ())
        })?;
        assert_bits_equal(
            &rows_out[row * out_dim..(row + 1) * out_dim],
            &read_f32_buffer(&mono_out, out_dim)?,
            &format!("gate+up partagé ligne {row}"),
        );
    }
    Ok(())
}

/// Combine en lignes avec partagé (un dispatch) = combine mono
/// `weighted_sum_add_shared_topk` de chaque ligne, et = ancien chemin en
/// lignes à deux dispatches (somme groupée puis `add_sigmoid_scaled_rows`).
#[test]
fn grouped_combine_with_shared_matches_mono_and_two_dispatch() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    const ROWS: usize = 5;
    const TOPK: usize = 8;
    let out_dim = 2048_usize;
    let src: Vec<f32> = (0..ROWS * TOPK)
        .flat_map(|slot| varied_row(out_dim, 200 + slot))
        .collect();
    let scores: Vec<f32> = (0..ROWS * TOPK)
        .map(|slot| 0.02 + (slot % 7) as f32 * 0.031)
        .collect();
    let residual: Vec<f32> = (0..ROWS)
        .flat_map(|row| varied_row(out_dim, 300 + row))
        .collect();
    let shared: Vec<f32> = (0..ROWS)
        .flat_map(|row| varied_row(out_dim, 400 + row))
        .collect();
    let shared_gate: Vec<f32> = (0..ROWS).map(|row| row as f32 * 0.7 - 1.3).collect();
    let src_buf = executor.upload_f32_buffer(&src, "vr_comb_src")?;
    let scores_buf = executor.upload_f32_buffer(&scores, "vr_comb_scores")?;
    let residual_buf = executor.upload_f32_buffer(&residual, "vr_comb_res")?;
    let shared_buf = executor.upload_f32_buffer(&shared, "vr_comb_shared")?;
    let gate_buf = executor.upload_f32_buffer(&shared_gate, "vr_comb_gate")?;
    let fused = executor.upload_f32_buffer(&vec![-7777.0; ROWS * out_dim], "vr_comb_fused")?;
    let two_step = executor.uncached_f32_buffer(ROWS * out_dim, "vr_comb_two")?;
    run_verify_rows_cb(&executor, |encoder| {
        executor.encode_weighted_sum_add_grouped_topk_shared(
            encoder,
            &src_buf,
            &scores_buf,
            &residual_buf,
            Some((&shared_buf, &gate_buf)),
            &fused,
            ROWS,
            TOPK,
            out_dim,
        )
    })?;
    run_verify_rows_cb(&executor, |encoder| {
        executor.encode_weighted_sum_add_grouped_topk(
            encoder,
            &mut Vec::new(),
            &src_buf,
            &scores_buf,
            &residual_buf,
            &two_step,
            ROWS,
            TOPK,
            out_dim,
        )?;
        executor.encode_add_sigmoid_scaled_rows(
            encoder, &shared_buf, &gate_buf, &two_step, ROWS, out_dim,
        )
    })?;
    let fused_out = read_f32_buffer(&fused, ROWS * out_dim)?;
    assert_bits_equal(
        &fused_out,
        &read_f32_buffer(&two_step, ROWS * out_dim)?,
        "combine fusionné contre deux dispatches",
    );
    for row in 0..ROWS {
        let slots = row * TOPK..(row + 1) * TOPK;
        let cols = row * out_dim..(row + 1) * out_dim;
        let row_src = executor.upload_f32_buffer(
            &src[slots.start * out_dim..slots.end * out_dim],
            "vr_comb_row_src",
        )?;
        let row_scores = executor.upload_f32_buffer(&scores[slots], "vr_comb_row_scores")?;
        let row_res = executor.upload_f32_buffer(&residual[cols.clone()], "vr_comb_row_res")?;
        let row_shared =
            executor.upload_f32_buffer(&shared[cols.clone()], "vr_comb_row_shared")?;
        let row_gate = executor.upload_f32_buffer(&shared_gate[row..=row], "vr_comb_row_gate")?;
        let mono = executor.uncached_f32_buffer(out_dim, "vr_comb_mono")?;
        run_verify_rows_cb(&executor, |encoder| {
            executor.encode_weighted_sum_add_shared_topk(
                encoder,
                &row_src,
                &row_scores,
                &row_res,
                &row_shared,
                &row_gate,
                &mono,
                TOPK,
                out_dim,
            )
        })?;
        assert_bits_equal(
            &fused_out[cols],
            &read_f32_buffer(&mono, out_dim)?,
            &format!("combine ligne {row} contre mono"),
        );
    }
    Ok(())
}

/// Porte gratuite de la dédup d'experts (étape 3) : coût des gathers routés
/// gate+up puis down sur 40 couches en un command buffer, forme du 35B
/// (256 experts u4 gs64, 2048→512→2048, top-8), quand les R lignes demandent
/// les MÊMES experts (recouvrement total) contre des experts DISTINCTS. Les
/// couches tournent sur des experts décalés pour que le cache ne serve pas une
/// couche depuis la précédente. L'écart borne le gain d'un kernel de dédup.
#[test]
#[ignore = "bench GPU (≈ 0,9 Go de poids synthétiques), lancer avec --ignored --release"]
fn moe_gather_overlap_cost_bench() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    const EXPERTS: usize = 256;
    const TOPK: usize = 8;
    const LAYERS: usize = 40;
    let (hidden, inter) = (2048_usize, 512_usize);
    let make = |out_dim: usize, in_dim: usize, label: &'static str| -> Result<StackedAffineBuffers> {
        let words = EXPERTS * out_dim * in_dim / 8;
        let packed: Vec<u32> = (0..words)
            .map(|i| (i as u32).wrapping_mul(2_654_435_761))
            .collect();
        let groups = in_dim / FAST_QMV_GROUP_SIZE;
        let scales: Vec<f32> = (0..EXPERTS * out_dim * groups)
            .map(|i| bf16_round(0.002 + (i % 5) as f32 * 0.0002))
            .collect();
        let biases: Vec<f32> = (0..EXPERTS * out_dim * groups)
            .map(|i| bf16_round(-0.01 + (i % 7) as f32 * 0.003))
            .collect();
        Ok(StackedAffineBuffers {
            packed: executor.buffer_from_slice(&packed, label)?,
            packed_offset: 0,
            scales: executor.buffer_from_f32_as_bf16(&scales, label)?,
            scales_offset: 0,
            biases: executor.buffer_from_f32_as_bf16(&biases, label)?,
            biases_offset: 0,
            experts: EXPERTS,
            out_dim,
            in_dim,
            packed_cols: in_dim / 8,
            group_size: FAST_QMV_GROUP_SIZE,
            bits: FAST_QMV_BITS,
            groups,
        })
    };
    let gate = make(inter, hidden, "vr_bench_gate")?;
    let up = make(inter, hidden, "vr_bench_up")?;
    let down = make(hidden, inter, "vr_bench_down")?;
    let rows_list = crate::runtime_flags::env_var("SARAGOSSA_RUST_MOE_GATHER_BENCH_ROWS")
        .map(|list| {
            list.split(',')
                .filter_map(|value| value.trim().parse::<usize>().ok())
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| vec![2, 3, 9]);
    for rows in rows_list {
        let slots = rows * TOPK;
        let lhs = executor.upload_f32_buffer(
            &(0..rows)
                .flat_map(|row| varied_row(hidden, row))
                .collect::<Vec<_>>(),
            "vr_bench_lhs",
        )?;
        let hidden_buf = executor.uncached_f32_buffer(slots * inter, "vr_bench_hidden")?;
        let down_buf = executor.uncached_f32_buffer(slots * hidden, "vr_bench_down_out")?;
        let mut medians = Vec::new();
        for same in [true, false] {
            // Experts par couche : décalage de 72 (≥ 9 × 8) par couche pour
            // casser la réutilisation entre couches.
            let index_buffers = (0..LAYERS)
                .map(|layer| {
                    let indices: Vec<u32> = (0..slots)
                        .map(|slot| {
                            let in_row = if same { slot % TOPK } else { slot };
                            ((layer * 72 + in_row * 3) % EXPERTS) as u32
                        })
                        .collect();
                    // Buffer neuf par couche : `upload_u32_buffer` recyclerait le même
                    // scratch (clé label + taille) pour les 40 couches.
                    executor.buffer_from_slice(&indices, "vr_bench_indices")
                })
                .collect::<Result<Vec<_>>>()?;
            let mut times = Vec::new();
            for rep in 0..12 {
                let started = std::time::Instant::now();
                run_verify_rows_cb(&executor, |encoder| {
                    for indices in &index_buffers {
                        let fused = executor.encode_gather_gate_up_swiglu(
                            encoder,
                            &mut Vec::new(),
                            &lhs,
                            rows,
                            &gate,
                            &up,
                            indices,
                            slots,
                            &hidden_buf,
                        )?;
                        if !fused {
                            return Err(InferError::Config(
                                "gather gate/up fusionné inéligible".to_string(),
                            ));
                        }
                        executor.encode_gather_matmul(
                            encoder,
                            &mut Vec::new(),
                            &hidden_buf,
                            slots,
                            &down,
                            indices,
                            slots,
                            &down_buf,
                        )?;
                    }
                    Ok(())
                })?;
                if rep >= 2 {
                    times.push(started.elapsed().as_secs_f64() * 1e3);
                }
            }
            times.sort_by(f64::total_cmp);
            let median = times[times.len() / 2];
            eprintln!(
                "moe_gather_overlap_bench rows={rows} same_experts={same} layers={LAYERS} median_ms={median:.3} min_ms={:.3}",
                times[0]
            );
            medians.push(median);
        }
        eprintln!(
            "moe_gather_overlap_bench rows={rows} gain_max_dedup_ms={:.3}",
            medians[1] - medians[0]
        );
    }
    Ok(())
}
