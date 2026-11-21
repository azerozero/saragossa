/// Identity projection with candidates across block boundaries. The raw logits
/// concentrate on token 1; cap=30 spreads probability over all four candidates.
#[test]
fn softcap_sampling_matches_cpu_and_distribution() -> Result<()> {
    let executor = match MetalExecutor::new() {
        Ok(executor) => executor,
        Err(InferError::Metal(message)) if message.contains("aucun device") => return Ok(()),
        Err(error) => return Err(error),
    };
    let mut raw = vec![-100.0_f32; 300];
    for (index, value) in [(1, 45.0), (259, 38.0), (299, 31.0), (7, 25.0)] {
        raw[index] = value;
    }
    let cap = 30.0;
    let capped: Vec<f32> = raw.iter().map(|&x| cap * (x / cap).tanh()).collect();
    let linear = Linear::new(Tensor::from_vec(vec![raw.len(), 1], raw.clone())?, None)?;
    let input = Tensor::row(vec![1.0])?;
    let temperature = 2.5;
    for (top_k, top_p) in [(0, 1.0), (32, 0.9), (2, 0.5)] {
        let mut counts = vec![0_usize; raw.len()];
        let mut differs_without_cap = 0;
        for seed in 0..1024 {
            let (gpu, cpu, uncapped) = if top_k == 0 {
                (
                    executor.sample_linear_biasless_gumbel(
                        &input,
                        &linear,
                        temperature,
                        seed,
                        Some(cap),
                    )?,
                    cpu_gumbel_argmax_for_test(&capped, temperature, seed),
                    cpu_gumbel_argmax_for_test(&raw, temperature, seed),
                )
            } else {
                (
                    executor.sample_linear_biasless_topk_topp(
                        &input,
                        &linear,
                        temperature,
                        top_p,
                        top_k,
                        seed,
                        Some(cap),
                    )?,
                    crate::sample_token_top_k_top_p(
                        &capped,
                        temperature,
                        top_p,
                        top_k,
                        &mut crate::DeterministicSampler::new(seed),
                    )?,
                    crate::sample_token_top_k_top_p(
                        &raw,
                        temperature,
                        top_p,
                        top_k,
                        &mut crate::DeterministicSampler::new(seed),
                    )?,
                )
            };
            assert_eq!(gpu, cpu, "softcap seed={seed} k={top_k} p={top_p}");
            counts[gpu] += 1;
            differs_without_cap += usize::from(gpu != uncapped);
        }
        assert!(
            differs_without_cap > 100,
            "oracle must detect missing softcap"
        );
        let distribution =
            crate::sampling::token_distribution_top_k_top_p(&capped, temperature, top_p, top_k)?;
        let chi2: f64 = distribution
            .iter()
            .map(|&(index, probability)| {
                let expected = 1024.0 * f64::from(probability);
                (counts[index] as f64 - expected).powi(2) / expected.max(1.0)
            })
            .sum();
        assert!(chi2 < 24.0, "softcap distribution k={top_k}: chi2={chi2}");
    }
    // Greedy still uses raw argmax, with no softcap dispatch or RNG draw.
    assert_eq!(
        executor.argmax_linear_biasless(&input, &linear)?,
        crate::argmax(&raw)?
    );
    assert_eq!(crate::argmax(&raw)?, crate::argmax(&capped)?);
    Ok(())
}

/// All buffer-based heads must forward the cap, including batched DFlash and
/// light-batch duo. Distinct RNG states detect accidental reuse across rows.
#[test]
fn softcap_sampling_buffer_heads_match_cpu() -> Result<()> {
    let executor = match MetalExecutor::new() {
        Ok(executor) => executor,
        Err(InferError::Metal(message)) if message.contains("aucun device") => return Ok(()),
        Err(error) => return Err(error),
    };
    let raw = vec![45.0_f32, 38.0, 31.0, 25.0, -30.0, -60.0];
    let cap = 30.0;
    let capped: Vec<f32> = raw.iter().map(|&x| cap * (x / cap).tanh()).collect();
    let linear = Linear::new(Tensor::from_vec(vec![raw.len(), 1], raw)?, None)?;
    let weight = executor.resolve_linear_weight_buffers(linear.weight(), "softcap_head")?;
    let input = executor.upload_f32_buffer(&[1.0, 1.0], "softcap_input")?;
    let index = executor.new_u32_buffer(2, "softcap_index")?;
    for (top_k, top_p) in [(0, 1.0), (4, 0.9)] {
        let temperature = 2.5;
        for seed in 0..32 {
            let states = [seed, crate::sampling::rng_state_for_draw(seed, 1)];
            let expected: Vec<u32> = states
                .iter()
                .map(|&state| {
                    if top_k == 0 {
                        Ok(cpu_gumbel_argmax_for_test(&capped, temperature, state) as u32)
                    } else {
                        crate::sample_token_top_k_top_p(
                            &capped,
                            temperature,
                            top_p,
                            top_k,
                            &mut crate::DeterministicSampler::new(state),
                        )
                        .map(|token| token as u32)
                    }
                })
                .collect::<Result<_>>()?;
            for head in 0..3 {
                let cb = executor.queue.new_command_buffer();
                let encoder = cb.new_compute_command_encoder();
                let mut owned = Vec::new();
                match head {
                    0 if top_k == 0 => executor.encode_lm_head_sample_gumbel_buffers(
                        encoder,
                        &input,
                        &weight,
                        &index,
                        1,
                        temperature,
                        seed,
                        Some(cap),
                    )?,
                    0 => executor.encode_lm_head_sample_topk_topp_buffers(
                        encoder,
                        &mut owned,
                        &input,
                        &weight,
                        &index,
                        1,
                        temperature,
                        top_p,
                        top_k,
                        seed,
                        Some(cap),
                    )?,
                    1 => executor.encode_lm_head_sample_rows_buffers(
                        encoder,
                        &input,
                        &weight,
                        &index,
                        1,
                        (temperature, top_p, top_k),
                        &states,
                        Some(cap),
                    )?,
                    _ => executor.encode_lm_head_sample_duo_buffers(
                        encoder,
                        &input,
                        &weight,
                        &index,
                        1,
                        states.map(|rng_state| DuoSampleParams {
                            temperature,
                            top_p,
                            top_k,
                            rng_state,
                        }),
                        [100, 101],
                        Some(cap),
                    )?,
                }
                encoder.end_encoding();
                commit_and_wait(cb)?;
                let count = if head == 0 { 1 } else { 2 };
                assert_eq!(
                    read_u32_buffer(&index, count)?,
                    expected[..count],
                    "head={head} k={top_k} seed={seed}"
                );
            }
        }
    }
    Ok(())
}
