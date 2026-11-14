// Oracles sans sinks : mêmes entrées bf16 et même seuil que le NAX d256.
fn generic_attention_spec(
    seq: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> PrefillAttentionSpec {
    PrefillAttentionSpec {
        seq,
        hidden_dim: q_heads * head_dim,
        q_heads,
        kv_heads,
        head_dim,
        rope_dims: head_dim,
        rope_frequency_dim: head_dim,
        rope_theta: 10_000.0,
        attn_scalar: head_dim as f32,
        window: None,
        k_eq_v: false,
        value_norm: false,
        eps: 1.0e-6,
    }
}

#[test]
fn causal_attention_generic_shape_scope() {
    for d in [128, 256] {
        for (qh, kvh) in [(32, 8), (32, 4), (64, 8), (15, 5), (7, 1), (3, 3)] {
            assert!(attention::prefill_attn_steel_supported(
                generic_attention_spec(257, qh, kvh, d)
            ));
            assert!(!attention::prefill_attn_steel_supported(
                generic_attention_spec(256, qh, kvh, d)
            ));
        }
        for (qh, kvh) in [(0, 0), (0, 1), (1, 0), (4, 8), (7, 2)] {
            assert!(!attention::prefill_attn_steel_supported(
                generic_attention_spec(257, qh, kvh, d)
            ));
        }
    }
    assert!(!attention::prefill_attn_steel_supported(
        generic_attention_spec(257, 32, 8, 64)
    ));
    assert!(!attention::prefill_attn_steel_supported(
        PrefillAttentionSpec {
            window: Some(128),
            ..generic_attention_spec(257, 32, 8, 128)
        }
    ));
}

fn assert_attention_close(got: &[f32], want: &[f32], label: &str) -> f32 {
    assert_eq!(got.len(), want.len(), "{label}");
    assert!(
        got.iter().chain(want).all(|x| x.is_finite()),
        "{label}: non fini"
    );
    let gap = got
        .iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert!(
        gap <= NAX_D256_TOLERANCE,
        "{label}: {gap:e} > {NAX_D256_TOLERANCE:e}"
    );
    gap
}

/// Appel de production par morceaux, y compris capacité KV supérieure au préfixe
/// et miroir bf16. Aucun buffer de sinks n'est lié.
fn generic_attention_chunk(
    executor: &MetalExecutor,
    tensors: (&Tensor, &Tensor, &Tensor),
    spec: PrefillAttentionSpec,
    mirror: bool,
) -> Result<Vec<f32>> {
    let (q, k, v) = tensors;
    let (kv_rows, kv_dim) = k.as_matrix()?;
    let kv_capacity = kv_rows + 37;
    let q_buf = executor.upload_f32_buffer(q.data(), "oracle_q")?;
    let k_buf = executor.upload_f32_buffer(k.data(), "oracle_k")?;
    let v_buf = executor.upload_f32_buffer(v.data(), "oracle_v")?;
    let out = executor.uncached_f32_buffer(q.len(), "oracle_out")?;
    let k16 = executor.private_bf16_buffer(kv_capacity * kv_dim, "oracle_k16")?;
    let v16 = executor.private_bf16_buffer(kv_capacity * kv_dim, "oracle_v16")?;
    let cb = executor.queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    let guard = EncoderEndGuard::new(enc);
    if mirror {
        executor.encode_f32_to_bf16(enc, &k_buf, &k16, k.len())?;
        executor.encode_f32_to_bf16(enc, &v_buf, &v16, v.len())?;
    }
    executor.encode_causal_attention_prefill_chunk(
        enc,
        &q_buf,
        &k_buf,
        &v_buf,
        &out,
        spec,
        kv_rows,
        kv_capacity,
        mirror.then_some([k16.as_ref(), v16.as_ref()]),
    )?;
    guard.end();
    commit_and_wait(cb)?;
    read_f32_buffer(&out, q.len())
}

fn generic_attention_oracle(
    executor: &MetalExecutor,
    seq: usize,
    qh: usize,
    kvh: usize,
    d: usize,
) -> Result<()> {
    let spec = generic_attention_spec(seq, qh, kvh, d);
    assert!(
        executor.causal_attention_prefill_nax(spec).is_some(),
        "NAX d{d} absent : oracle non exécuté"
    );
    assert!(executor.causal_attention_prefill_chunk_supported(spec));
    let round = |t: Tensor| {
        Tensor::from_vec(
            t.shape().to_vec(),
            t.data().iter().copied().map(bf16_round).collect(),
        )
    };
    let (q, k, v) = causal_attention_inputs(seq, qh, kvh, d)?;
    let (q, k, v) = (round(q)?, round(k)?, round(v)?);
    let got = executor.causal_attention_prefill(&q, &k, &v, qh, kvh)?;
    let (pipeline, threads) = if d == 128 && (qh, kvh) == (32, 4) {
        (&executor.causal_attention_prefill_batch_long_d128_f32, 32)
    } else if seq <= 2048 {
        (&executor.causal_attention_prefill_mid_f32, 256)
    } else {
        (&executor.causal_attention_prefill_long_f32, 256)
    };
    let naive =
        causal_attention_prefill_with_pipeline(executor, &q, &k, &v, qh, kvh, pipeline, threads)?;
    let gap = assert_attention_close(got.data(), naive.data(), "NAX/naïf");
    let mut rows = vec![
        0,
        1,
        16,
        31,
        32,
        33,
        48,
        63,
        64,
        65,
        seq / 2,
        seq - 2,
        seq - 1,
    ];
    let tile = (seq - 1) / 64 * 64;
    rows.extend(
        [tile, tile + 16, tile + 32, tile + 48]
            .into_iter()
            .filter(|&r| r < seq),
    );
    rows.sort_unstable();
    rows.dedup();
    let expected = cpu_causal_attention_selected_rows_gqa(&q, &k, &v, qh, kvh, &rows)?;
    for (row, want) in expected {
        let start = row * qh * d;
        assert_attention_close(&got.data()[start..start + qh * d], &want, "NAX/CPU");
        assert_attention_close(&naive.data()[start..start + qh * d], &want, "naïf/CPU");
    }
    // Décalages non alignés, queue Q partielle et chaque groupe de lignes.
    for chunk in [1, 31, 32, 33, 63, 64, 65] {
        let tail = tail_rows(&q, chunk)?;
        let chunk_spec = PrefillAttentionSpec { seq: chunk, ..spec };
        let a = generic_attention_chunk(executor, (&tail, &k, &v), chunk_spec, false)?;
        let b = generic_attention_chunk(executor, (&tail, &k, &v), chunk_spec, true)?;
        assert!(
            a.iter().zip(&b).all(|(a, b)| a.to_bits() == b.to_bits()),
            "miroir différent"
        );
        assert_attention_close(
            &a,
            &got.data()[(seq - chunk) * qh * d..],
            "morceau/bloc complet",
        );
    }
    eprintln!("NAX générique d{d} {qh}/{kvh} seq={seq}: écart naïf {gap:e}, CPU et morceaux OK");
    Ok(())
}

#[test]
fn causal_attention_generic_nax_matches_naive_cpu() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    if executor.causal_attention_prefill_nax_d128.is_none() {
        eprintln!("NAX d128 absent : oracle court sauté");
        return Ok(());
    }
    for (qh, kvh) in [(32, 8), (32, 4), (64, 8), (15, 5), (7, 1), (3, 3)] {
        generic_attention_oracle(&executor, 257, qh, kvh, 128)?;
    }
    for (qh, kvh) in [(32, 8), (32, 4), (64, 8)] {
        generic_attention_oracle(&executor, 319, qh, kvh, 256)?;
    }
    Ok(())
}

/// Matrice longue explicite, sans répétitions de benchmark. Échoue si le GPU
/// ou le NAX manque : une validation demandée ne doit pas réussir par saut.
#[test]
#[ignore = "oracle GPU long à lancer sous verrou"]
fn causal_attention_generic_nax_long_oracle() -> Result<()> {
    let executor = MetalExecutor::new()?;
    for d in [128, 256] {
        for (qh, kvh) in [(32, 8), (32, 4), (64, 8)] {
            for seq in [273, 1069, 2049, 4097, 8191, 8192] {
                generic_attention_oracle(&executor, seq, qh, kvh, d)?;
            }
        }
    }
    Ok(())
}

#[test]
fn causal_attention_generic_nax_missing_falls_back() -> Result<()> {
    let Some(mut executor) = test_executor()? else {
        return Ok(());
    };
    executor.causal_attention_prefill_nax_d128 = None;
    let spec = generic_attention_spec(257, 32, 8, 128);
    assert!(!executor.causal_attention_prefill_chunk_supported(spec));
    let (q, k, v) = causal_attention_inputs(257, 32, 8, 128)?;
    let got = executor.causal_attention_prefill(&q, &k, &v, 32, 8)?;
    let naive = causal_attention_prefill_with_pipeline(
        &executor,
        &q,
        &k,
        &v,
        32,
        8,
        &executor.causal_attention_prefill_mid_f32,
        256,
    )?;
    assert_eq!(got.data(), naive.data());
    executor.causal_attention_prefill_steel_nax_d256 = None;
    // Sans NAX, les nouvelles formes d256 doivent aussi emprunter le Steel f32.
    if executor.causal_attention_prefill_steel_d256_f32.is_some() {
        for (qh, kvh) in [(32, 8), (32, 4), (64, 8)] {
            let spec = generic_attention_spec(257, qh, kvh, 256);
            assert!(executor.causal_attention_prefill_chunk_supported(spec));
            let (q, k, v) = causal_attention_inputs(257, qh, kvh, 256)?;
            let got = executor.causal_attention_prefill(&q, &k, &v, qh, kvh)?;
            let expected = cpu_causal_attention_selected_rows_gqa(
                &q,
                &k,
                &v,
                qh,
                kvh,
                &[0, 31, 32, 63, 64, 255, 256],
            )?;
            for (row, want) in expected {
                let start = row * qh * 256;
                assert_attention_close(
                    &got.data()[start..start + qh * 256],
                    &want,
                    "repli Steel/CPU",
                );
            }
        }
    }
    executor.causal_attention_prefill_steel_d256_f32 = None;
    assert!(
        !executor.causal_attention_prefill_chunk_supported(generic_attention_spec(257, 32, 8, 256))
    );
    Ok(())
}

/// Reprise de la sonde kb_attn_d128 : entrées f32, médiane commit_and_wait,
/// mêmes kernels naïfs de production et NAX incluant les conversions Q/K/V.
/// Le pipeline est retiré pour A, rétabli pour B, sans modifier les flags globaux.
#[test]
#[ignore = "microbench A/B sous verrou GPU, réservé aux mesures explicites"]
fn causal_attention_generic_nax_ab() -> Result<()> {
    let mut executor = MetalExecutor::new()?;
    assert!(
        executor
            .causal_attention_prefill_nax(generic_attention_spec(257, 32, 8, 128))
            .is_some(),
        "NAX d128 absent ou désactivé"
    );
    let reps = crate::runtime_flags::env_var("KB_REPS")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(7)
        .max(1);
    for (qh, kvh) in [(32, 8), (32, 4), (64, 8)] {
        for seq in [2048, 4096, 8192] {
            let spec = generic_attention_spec(seq, qh, kvh, 128);
            let q: Vec<f32> = (0..seq * qh * 128)
                .map(|i| ((i * 7 % 23) as f32 - 11.0) * 0.02)
                .collect();
            let k: Vec<f32> = (0..seq * kvh * 128)
                .map(|i| ((i * 5 % 19) as f32 - 9.0) * 0.02)
                .collect();
            let v: Vec<f32> = (0..k.len())
                .map(|i| ((i * 3 % 17) as f32 - 8.0) * 0.03)
                .collect();
            let qb = executor.upload_f32_buffer(&q, "ab_q")?;
            let kb = executor.upload_f32_buffer(&k, "ab_k")?;
            let vb = executor.upload_f32_buffer(&v, "ab_v")?;
            let out = executor.uncached_f32_buffer(q.len(), "ab_out")?;
            let nax = executor.causal_attention_prefill_nax_d128.take();
            let mut outputs = Vec::new();
            let mut medians = Vec::new();
            for mode in ["naïf", "NAX + Q/K/V bf16"] {
                let mut times = Vec::new();
                for _ in 0..reps {
                    let cb = executor.queue.new_command_buffer();
                    let enc = cb.new_compute_command_encoder();
                    let guard = EncoderEndGuard::new(enc);
                    executor.encode_causal_attention_prefill(enc, &qb, &kb, &vb, &out, spec)?;
                    guard.end();
                    let now = std::time::Instant::now();
                    commit_and_wait(cb)?;
                    times.push(now.elapsed().as_secs_f64());
                }
                outputs.push(read_f32_buffer(&out, q.len())?);
                times.sort_by(f64::total_cmp);
                medians.push(times[times.len() / 2]);
                eprintln!(
                    "A/B d128 {qh}/{kvh} seq={seq} {mode}: min {:.3} médiane {:.3} max {:.3} ms",
                    times[0] * 1e3,
                    times[times.len() / 2] * 1e3,
                    times[times.len() - 1] * 1e3
                );
                executor.causal_attention_prefill_nax_d128 =
                    nax.as_ref().map(|p| attention::NaxCausalAttention {
                        pipeline: p.pipeline.clone(),
                        shape: p.shape,
                    });
            }
            let gap = assert_attention_close(&outputs[0], &outputs[1], "A/B f32 naïf/NAX");
            eprintln!("A/B gain {:.2}×, écart {gap:e}", medians[0] / medians[1]);
        }
    }
    Ok(())
}

/// À lancer dans deux processus distincts : NAX=0 puis STEEL_D256=0.
#[test]
#[ignore = "validation explicite des kill-switches"]
fn causal_attention_generic_kill_switch() -> Result<()> {
    assert!(
        !crate::runtime_flags::prefill_attn_nax_enabled()
            || !crate::runtime_flags::prefill_attn_steel_d256_enabled(),
        "désactiver NAX ou STEEL_D256 pour cet oracle"
    );
    let executor = MetalExecutor::new()?;
    for (qh, kvh) in [(32, 8), (32, 4), (64, 8)] {
        let spec = generic_attention_spec(257, qh, kvh, 128);
        assert!(executor.causal_attention_prefill_nax(spec).is_none());
        assert!(!executor.causal_attention_prefill_chunk_supported(spec));
        let (q, k, v) = causal_attention_inputs(257, qh, kvh, 128)?;
        let got = executor.causal_attention_prefill(&q, &k, &v, qh, kvh)?;
        let (pipeline, threads) =
            if (qh, kvh) == (32, 4) && crate::runtime_flags::prefill_attn_batch_mid_30b_enabled() {
                (&executor.causal_attention_prefill_batch_long_d128_f32, 32)
            } else {
                (&executor.causal_attention_prefill_mid_f32, 256)
            };
        let naive = causal_attention_prefill_with_pipeline(
            &executor, &q, &k, &v, qh, kvh, pipeline, threads,
        )?;
        assert_eq!(got.data(), naive.data());
    }
    Ok(())
}
