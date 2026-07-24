#[test]
fn noncausal_attention_prefill_matches_cpu() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let seq = 3;
    let heads = 2;
    let head_dim = 4;
    let dim = heads * head_dim;
    let q = Tensor::from_vec(
        vec![seq, dim],
        (0..seq * dim).map(|idx| idx as f32 * 0.01 - 0.1).collect(),
    )?;
    let k = Tensor::from_vec(
        vec![seq, dim],
        (0..seq * dim)
            .map(|idx| (idx as f32 * 0.02).sin())
            .collect(),
    )?;
    let v = Tensor::from_vec(
        vec![seq, dim],
        (0..seq * dim)
            .map(|idx| (idx as f32 * 0.03).cos())
            .collect(),
    )?;

    let got = executor.noncausal_attention_prefill(&q, &k, &v, heads, heads)?;
    let expected = cpu_noncausal_attention(&q, &k, &v, heads)?;

    assert_close_eps(got.data(), expected.data(), 2.0e-5);
    Ok(())
}

#[test]
fn noncausal_attention_prefill_head_dim64_matches_cpu() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let seq = 70;
    let heads = 2;
    let head_dim = 64;
    let dim = heads * head_dim;
    let q = Tensor::from_vec(
        vec![seq, dim],
        (0..seq * dim)
            .map(|idx| (idx as f32 * 0.013).sin() * 0.5)
            .collect(),
    )?;
    let k = Tensor::from_vec(
        vec![seq, dim],
        (0..seq * dim)
            .map(|idx| (idx as f32 * 0.017).cos() * 0.25)
            .collect(),
    )?;
    let v = Tensor::from_vec(
        vec![seq, dim],
        (0..seq * dim)
            .map(|idx| (idx as f32 * 0.019).sin())
            .collect(),
    )?;

    let got = executor.noncausal_attention_prefill(&q, &k, &v, heads, heads)?;
    let expected = cpu_noncausal_attention(&q, &k, &v, heads)?;

    assert_close_eps(got.data(), expected.data(), 5.0e-4);
    Ok(())
}

fn test_expert(gate_scale: f32, up_scale: f32, down_scale: f32) -> Result<GatedMlp> {
    Ok(GatedMlp::new(
        Linear::from_weight(
            LinearWeight::AffineQuantized(test_affine(64, 64, gate_scale)?),
            None,
        )?,
        Linear::from_weight(
            LinearWeight::AffineQuantized(test_affine(64, 64, up_scale)?),
            None,
        )?,
        Linear::from_weight(
            LinearWeight::AffineQuantized(test_affine(8, 64, down_scale)?),
            None,
        )?,
    ))
}

fn test_quant_linear_u8_group(out_dim: usize, in_dim: usize, group_size: usize) -> Result<Linear> {
    Linear::from_weight(
        LinearWeight::AffineQuantized(test_affine_varied_u8_group(out_dim, in_dim, group_size)?),
        None,
    )
}

fn test_expert_u8_group(hidden: usize, inter: usize, group_size: usize) -> Result<GatedMlp> {
    Ok(GatedMlp::new(
        test_quant_linear_u8_group(inter, hidden, group_size)?,
        test_quant_linear_u8_group(inter, hidden, group_size)?,
        test_quant_linear_u8_group(hidden, inter, group_size)?,
    ))
}

fn cpu_noncausal_attention(q: &Tensor, k: &Tensor, v: &Tensor, heads: usize) -> Result<Tensor> {
    let (seq, dim) = q.as_matrix()?;
    let head_dim = dim / heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0.0_f32; seq * dim];
    let mut scores = vec![0.0_f32; seq];
    for head in 0..heads {
        let head_base = head * head_dim;
        for row_q in 0..seq {
            let mut max_score = f32::NEG_INFINITY;
            for (row_k, score_value) in scores.iter_mut().enumerate().take(seq) {
                let mut dot = 0.0_f32;
                for col in 0..head_dim {
                    dot += q.data()[row_q * dim + head_base + col]
                        * k.data()[row_k * dim + head_base + col];
                }
                let score = dot * scale;
                *score_value = score;
                max_score = max_score.max(score);
            }
            let mut denom = 0.0_f32;
            for score in scores.iter_mut().take(seq) {
                *score = (*score - max_score).exp();
                denom += *score;
            }
            for col in 0..head_dim {
                let mut acc = 0.0_f32;
                for (row_v, prob) in scores.iter().take(seq).enumerate() {
                    acc += (*prob / denom) * v.data()[row_v * dim + head_base + col];
                }
                out[row_q * dim + head_base + col] = acc;
            }
        }
    }
    Tensor::from_vec(vec![seq, dim], out)
}

/// Référence CPU de l'attention prefill CAUSALE : chaque requête n'attend que les
/// positions `row_k <= row_q` (masque causal), softmax complet, produit V.
fn cpu_causal_attention(q: &Tensor, k: &Tensor, v: &Tensor, heads: usize) -> Result<Tensor> {
    cpu_causal_attention_gqa(q, k, v, heads, heads)
}

fn cpu_causal_attention_gqa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    q_heads: usize,
    kv_heads: usize,
) -> Result<Tensor> {
    let (seq, dim) = q.as_matrix()?;
    let (k_seq, kv_dim) = k.as_matrix()?;
    let (v_seq, v_dim) = v.as_matrix()?;
    if q_heads == 0 || kv_heads == 0 || q_heads % kv_heads != 0 {
        return Err(InferError::Dimension(format!(
            "cpu causal attention heads invalides q_heads={q_heads}, kv_heads={kv_heads}"
        )));
    }
    if k_seq != seq || v_seq != seq || kv_dim != v_dim {
        return Err(InferError::Dimension(format!(
            "cpu causal attention q={:?}, k={:?}, v={:?}",
            q.shape(),
            k.shape(),
            v.shape()
        )));
    }
    if dim % q_heads != 0 {
        return Err(InferError::Dimension(format!(
            "cpu causal attention q_dim={dim} incompatible avec q_heads={q_heads}"
        )));
    }
    let head_dim = dim / q_heads;
    if kv_dim != kv_heads * head_dim {
        return Err(InferError::Dimension(format!(
            "cpu causal attention kv_dim={kv_dim} incompatible avec kv_heads={kv_heads} head_dim={head_dim}"
        )));
    }
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0.0_f32; seq * dim];
    let mut scores = vec![0.0_f32; seq];
    let kv_group = q_heads / kv_heads;
    for q_head in 0..q_heads {
        let q_head_base = q_head * head_dim;
        let kv_head_base = (q_head / kv_group) * head_dim;
        for row_q in 0..seq {
            let mut max_score = f32::NEG_INFINITY;
            for (row_k, score_value) in scores.iter_mut().enumerate().take(row_q + 1) {
                let mut dot = 0.0_f32;
                for col in 0..head_dim {
                    dot += q.data()[row_q * dim + q_head_base + col]
                        * k.data()[row_k * kv_dim + kv_head_base + col];
                }
                let score = dot * scale;
                *score_value = score;
                max_score = max_score.max(score);
            }
            let mut denom = 0.0_f32;
            for score in scores.iter_mut().take(row_q + 1) {
                *score = (*score - max_score).exp();
                denom += *score;
            }
            for col in 0..head_dim {
                let mut acc = 0.0_f32;
                for (row_v, prob) in scores.iter().take(row_q + 1).enumerate() {
                    acc += (*prob / denom) * v.data()[row_v * kv_dim + kv_head_base + col];
                }
                out[row_q * dim + q_head_base + col] = acc;
            }
        }
    }
    Tensor::from_vec(vec![seq, dim], out)
}

fn cpu_causal_attention_selected_rows_gqa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    q_heads: usize,
    kv_heads: usize,
    rows: &[usize],
) -> Result<Vec<(usize, Vec<f32>)>> {
    let (seq, dim) = q.as_matrix()?;
    let (k_seq, kv_dim) = k.as_matrix()?;
    let (v_seq, v_dim) = v.as_matrix()?;
    if q_heads == 0 || kv_heads == 0 || q_heads % kv_heads != 0 {
        return Err(InferError::Dimension(format!(
            "cpu causal selected heads invalides q_heads={q_heads}, kv_heads={kv_heads}"
        )));
    }
    if k_seq != seq || v_seq != seq || kv_dim != v_dim {
        return Err(InferError::Dimension(format!(
            "cpu causal selected q={:?}, k={:?}, v={:?}",
            q.shape(),
            k.shape(),
            v.shape()
        )));
    }
    if dim % q_heads != 0 {
        return Err(InferError::Dimension(format!(
            "cpu causal selected q_dim={dim} incompatible avec q_heads={q_heads}"
        )));
    }
    let head_dim = dim / q_heads;
    if kv_dim != kv_heads * head_dim {
        return Err(InferError::Dimension(format!(
            "cpu causal selected kv_dim={kv_dim} incompatible avec kv_heads={kv_heads} head_dim={head_dim}"
        )));
    }
    let scale = 1.0 / (head_dim as f32).sqrt();
    let kv_group = q_heads / kv_heads;
    let mut selected = Vec::with_capacity(rows.len());
    for &row_q in rows {
        if row_q >= seq {
            return Err(InferError::Dimension(format!(
                "cpu causal selected row={row_q}, seq={seq}"
            )));
        }
        let mut out_row = vec![0.0_f32; dim];
        let mut scores = vec![0.0_f32; row_q + 1];
        for q_head in 0..q_heads {
            let q_head_base = q_head * head_dim;
            let kv_head_base = (q_head / kv_group) * head_dim;
            let mut max_score = f32::NEG_INFINITY;
            for (row_k, score_slot) in scores.iter_mut().enumerate() {
                let mut dot = 0.0_f32;
                for col in 0..head_dim {
                    dot += q.data()[row_q * dim + q_head_base + col]
                        * k.data()[row_k * kv_dim + kv_head_base + col];
                }
                let score = dot * scale;
                *score_slot = score;
                max_score = max_score.max(score);
            }
            let mut denom = 0.0_f32;
            for score in &mut scores {
                *score = (*score - max_score).exp();
                denom += *score;
            }
            for col in 0..head_dim {
                let mut acc = 0.0_f32;
                for (row_v, prob) in scores.iter().enumerate() {
                    acc += (*prob / denom) * v.data()[row_v * kv_dim + kv_head_base + col];
                }
                out_row[q_head_base + col] = acc;
            }
        }
        selected.push((row_q, out_row));
    }
    Ok(selected)
}

fn assert_causal_rows_close(
    got: &Tensor,
    expected_rows: &[(usize, Vec<f32>)],
    eps: f32,
) -> Result<()> {
    let (seq, dim) = got.as_matrix()?;
    for (row, expected) in expected_rows {
        if *row >= seq || expected.len() != dim {
            return Err(InferError::Dimension(format!(
                "ligne attendue row={row} len={} incompatible avec [{seq},{dim}]",
                expected.len()
            )));
        }
        let start = row * dim;
        assert_close_eps(&got.data()[start..start + dim], expected, eps);
    }
    Ok(())
}

fn causal_attention_inputs(
    seq: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    let q_dim = checked_len(q_heads, head_dim, "test causal q_dim")?;
    let kv_dim = checked_len(kv_heads, head_dim, "test causal kv_dim")?;
    let q_len = checked_len(seq, q_dim, "test causal q")?;
    let kv_len = checked_len(seq, kv_dim, "test causal kv")?;
    let q = Tensor::from_vec(
        vec![seq, q_dim],
        (0..q_len)
            .map(|idx| (idx as f32 * 0.0007).sin() * 0.5)
            .collect(),
    )?;
    let k = Tensor::from_vec(
        vec![seq, kv_dim],
        (0..kv_len)
            .map(|idx| (idx as f32 * 0.0011).cos() * 0.25)
            .collect(),
    )?;
    let v = Tensor::from_vec(
        vec![seq, kv_dim],
        (0..kv_len).map(|idx| (idx as f32 * 0.0013).sin()).collect(),
    )?;
    Ok((q, k, v))
}

#[allow(
    clippy::too_many_arguments,
    reason = "helper Metal de test: tenseurs, têtes et pipeline testé restent explicites"
)]
fn causal_attention_prefill_with_pipeline(
    executor: &MetalExecutor,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    q_heads: usize,
    kv_heads: usize,
    pipeline: &ComputePipelineState,
    threadgroup_width: u64,
) -> Result<Tensor> {
    causal_attention_prefill_with_pipeline_grid(
        executor,
        q,
        k,
        v,
        q_heads,
        kv_heads,
        pipeline,
        threadgroup_width,
        q_heads,
    )
}

#[allow(
    clippy::too_many_arguments,
    reason = "helper Metal de test: grille et paramètres du pipeline restent explicites"
)]
fn causal_attention_prefill_with_pipeline_grid(
    executor: &MetalExecutor,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    q_heads: usize,
    kv_heads: usize,
    pipeline: &ComputePipelineState,
    threadgroup_width: u64,
    grid_heads: usize,
) -> Result<Tensor> {
    causal_attention_prefill_with_pipeline_grid_rows(
        executor,
        q,
        k,
        v,
        q_heads,
        kv_heads,
        pipeline,
        threadgroup_width,
        grid_heads,
        q.as_matrix()?.0,
    )
}

#[allow(
    clippy::too_many_arguments,
    reason = "helper Metal de test: grille complète et tenseurs restent explicites"
)]
fn causal_attention_prefill_with_pipeline_grid_rows(
    executor: &MetalExecutor,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    q_heads: usize,
    kv_heads: usize,
    pipeline: &ComputePipelineState,
    threadgroup_width: u64,
    grid_heads: usize,
    grid_rows: usize,
) -> Result<Tensor> {
    let (seq, q_dim) = q.as_matrix()?;
    let (k_seq, k_dim) = k.as_matrix()?;
    let (v_seq, v_dim) = v.as_matrix()?;
    if q_heads == 0 || kv_heads == 0 || q_heads % kv_heads != 0 {
        return Err(InferError::Dimension(format!(
            "test causal attention heads invalides q_heads={q_heads}, kv_heads={kv_heads}"
        )));
    }
    if k_seq != seq || v_seq != seq || k_dim != v_dim {
        return Err(InferError::Dimension(format!(
            "test causal attention q={:?}, k={:?}, v={:?}",
            q.shape(),
            k.shape(),
            v.shape()
        )));
    }
    if q_dim % q_heads != 0 || k_dim % kv_heads != 0 {
        return Err(InferError::Dimension(format!(
            "test causal attention dims incompatibles q_dim={q_dim}, k_dim={k_dim}, q_heads={q_heads}, kv_heads={kv_heads}"
        )));
    }
    let head_dim = q_dim / q_heads;
    if k_dim / kv_heads != head_dim {
        return Err(InferError::Dimension(format!(
            "test causal attention head_dim q={}, kv={}",
            head_dim,
            k_dim / kv_heads
        )));
    }

    let q_buffer = executor.upload_f32_buffer(q.data(), "test_causal_q")?;
    let k_buffer = executor.upload_f32_buffer(k.data(), "test_causal_k")?;
    let v_buffer = executor.upload_f32_buffer(v.data(), "test_causal_v")?;
    let output_len = checked_len(seq, q_dim, "test sortie attention causale")?;
    let output_buffer = executor.uncached_f32_buffer(output_len, "test_causal_out")?;
    let dims = [
        checked_u32(seq, "test causal seq")?,
        checked_u32(q_heads, "test causal q_heads")?,
        checked_u32(kv_heads, "test causal kv_heads")?,
        checked_u32(head_dim, "test causal head_dim")?,
    ];

    let command_buffer = executor.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(&q_buffer), 0);
    encoder.set_buffer(1, Some(&k_buffer), 0);
    encoder.set_buffer(2, Some(&v_buffer), 0);
    encoder.set_buffer(3, Some(&output_buffer), 0);
    set_u32_bytes(encoder, 4, &dims, "test_causal_dims")?;
    set_f32_bytes(
        encoder,
        5,
        &[(head_dim as f32).sqrt().recip(), 0.0],
        "test_causal_scale_params",
    )?;
    encoder.dispatch_thread_groups(
        MTLSize::new(grid_heads as u64, grid_rows as u64, 1),
        MTLSize::new(threadgroup_width, 1, 1),
    );
    encoder.end_encoding();
    commit_and_wait(command_buffer)?;

    let output = read_f32_buffer(&output_buffer, output_len)?;
    Tensor::from_vec(vec![seq, q_dim], output)
}

fn causal_attention_prefill_with_steel_d256_pipeline(
    executor: &MetalExecutor,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    q_heads: usize,
    kv_heads: usize,
    pipeline: &ComputePipelineState,
) -> Result<Tensor> {
    let (seq, q_dim) = q.as_matrix()?;
    let (k_seq, k_dim) = k.as_matrix()?;
    let (v_seq, v_dim) = v.as_matrix()?;
    if q_heads == 0 || kv_heads == 0 || q_heads % kv_heads != 0 {
        return Err(InferError::Dimension(format!(
            "test steel causal attention heads invalides q_heads={q_heads}, kv_heads={kv_heads}"
        )));
    }
    if k_seq != seq || v_seq != seq || k_dim != v_dim {
        return Err(InferError::Dimension(format!(
            "test steel causal attention q={:?}, k={:?}, v={:?}",
            q.shape(),
            k.shape(),
            v.shape()
        )));
    }
    if q_dim % q_heads != 0 || k_dim % kv_heads != 0 {
        return Err(InferError::Dimension(format!(
            "test steel causal attention dims incompatibles q_dim={q_dim}, k_dim={k_dim}, q_heads={q_heads}, kv_heads={kv_heads}"
        )));
    }
    let head_dim = q_dim / q_heads;
    if head_dim != 256 || k_dim / kv_heads != head_dim {
        return Err(InferError::Dimension(format!(
            "test steel causal attention head_dim q={}, kv={}",
            head_dim,
            k_dim / kv_heads
        )));
    }

    let q_buffer = executor.upload_f32_buffer(q.data(), "test_steel_causal_q")?;
    let k_buffer = executor.upload_f32_buffer(k.data(), "test_steel_causal_k")?;
    let v_buffer = executor.upload_f32_buffer(v.data(), "test_steel_causal_v")?;
    let output_len = checked_len(seq, q_dim, "test sortie steel attention causale")?;
    let output_buffer = executor.uncached_f32_buffer(output_len, "test_steel_causal_out")?;
    let spec = PrefillAttentionSpec {
        seq,
        hidden_dim: q_dim,
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
        eps: 0.0,
    };
    let params = attention::steel_attn_params(spec, 32, 64, "test steel causal d256")?;

    let command_buffer = executor.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(&q_buffer), 0);
    encoder.set_buffer(1, Some(&k_buffer), 0);
    encoder.set_buffer(2, Some(&v_buffer), 0);
    encoder.set_buffer(3, Some(&output_buffer), 0);
    encoder.set_bytes(
        4,
        std::mem::size_of::<attention::SteelAttnParams>() as NSUInteger,
        (&params as *const attention::SteelAttnParams).cast::<std::ffi::c_void>(),
    );
    encoder.dispatch_thread_groups(
        MTLSize::new(
            checked_nsuint(seq.div_ceil(32), "test steel causal d256 NQ")?,
            checked_nsuint(q_heads, "test steel causal d256 heads")?,
            1,
        ),
        MTLSize::new(32, 4, 1),
    );
    encoder.end_encoding();
    commit_and_wait(command_buffer)?;

    let output = read_f32_buffer(&output_buffer, output_len)?;
    Tensor::from_vec(vec![seq, q_dim], output)
}

/// Verrouille le prefill causal aux frontières utiles :
/// `seq <= 256` (court), `257..=2048` (mid scores[2048]) et `>2048` (long
/// recalculé inchangé depuis le tronc). Le test CPU couvre les frontières
/// 256/2048 ; le test bit-exact ci-dessous prouve que mid conserve l'ordre
/// arithmétique du kernel court à la frontière commune.
#[test]
fn causal_attention_prefill_matches_cpu_across_256_boundary() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    for (seq, heads, head_dim) in [(256_usize, 2_usize, 128_usize), (2048, 1, 4)] {
        let (q, k, v) = causal_attention_inputs(seq, heads, heads, head_dim)?;
        let got = executor.causal_attention_prefill(&q, &k, &v, heads, heads)?;
        let expected = cpu_causal_attention(&q, &k, &v, heads)?;
        assert_close_eps(got.data(), expected.data(), 5.0e-4);
    }
    Ok(())
}

#[test]
fn causal_attention_prefill_mid_is_bit_identical_to_short_at_256() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };

    let (q, k, v) = causal_attention_inputs(256, 2, 2, 128)?;
    let short = causal_attention_prefill_with_pipeline(
        &executor,
        &q,
        &k,
        &v,
        2,
        2,
        &executor.causal_attention_prefill_f32,
        256,
    )?;
    let mid = causal_attention_prefill_with_pipeline(
        &executor,
        &q,
        &k,
        &v,
        2,
        2,
        &executor.causal_attention_prefill_mid_f32,
        256,
    )?;
    assert_bits_equal(
        short.data(),
        mid.data(),
        "causal prefill short vs mid seq256",
    );
    Ok(())
}

#[test]
fn causal_attention_prefill_batch_long_is_scoped_to_27b_30b_35b() {
    let base = PrefillAttentionSpec {
        seq: 4096,
        hidden_dim: 0,
        q_heads: 24,
        kv_heads: 4,
        head_dim: 256,
        rope_dims: 256,
        rope_frequency_dim: 256,
        rope_theta: 10_000.0,
        attn_scalar: 256.0,
        window: None,
        k_eq_v: false,
        value_norm: false,
        eps: 0.0,
    };
    assert!(attention::prefill_attn_batch_long_supported(base));
    assert!(attention::prefill_attn_batch_long_supported(
        PrefillAttentionSpec {
            q_heads: 32,
            kv_heads: 4,
            head_dim: 128,
            rope_dims: 128,
            ..base
        }
    ));
    assert!(attention::prefill_attn_batch_long_supported(
        PrefillAttentionSpec {
            q_heads: 16,
            kv_heads: 2,
            head_dim: 256,
            rope_dims: 256,
            ..base
        }
    ));
    assert!(!attention::prefill_attn_batch_long_supported(
        PrefillAttentionSpec { seq: 2048, ..base }
    ));
    assert!(attention::prefill_attn_batch_mid_30b_supported(
        PrefillAttentionSpec {
            seq: 1024,
            q_heads: 32,
            kv_heads: 4,
            head_dim: 128,
            rope_dims: 128,
            ..base
        }
    ));
    assert!(!attention::prefill_attn_batch_mid_30b_supported(
        PrefillAttentionSpec { seq: 1024, ..base }
    ));
    assert!(!attention::prefill_attn_batch_mid_30b_supported(
        PrefillAttentionSpec { seq: 256, ..base }
    ));
    assert!(attention::prefill_attn_batch_mid_35b_supported(
        PrefillAttentionSpec {
            seq: 1024,
            q_heads: 16,
            kv_heads: 2,
            head_dim: 256,
            rope_dims: 256,
            ..base
        }
    ));
    assert!(!attention::prefill_attn_batch_mid_35b_supported(
        PrefillAttentionSpec { seq: 1024, ..base }
    ));
    assert!(!attention::prefill_attn_batch_mid_35b_supported(
        PrefillAttentionSpec {
            seq: 2049,
            q_heads: 16,
            kv_heads: 2,
            head_dim: 256,
            rope_dims: 256,
            ..base
        }
    ));
    assert!(attention::prefill_attn_steel_d256_supported(
        PrefillAttentionSpec {
            q_heads: 16,
            kv_heads: 2,
            head_dim: 256,
            rope_dims: 256,
            ..base
        }
    ));
    assert!(!attention::prefill_attn_steel_d256_supported(
        PrefillAttentionSpec {
            seq: 256,
            q_heads: 16,
            kv_heads: 2,
            head_dim: 256,
            rope_dims: 256,
            ..base
        }
    ));
    assert!(!attention::prefill_attn_steel_d256_supported(
        PrefillAttentionSpec { ..base }
    ));
    assert!(!attention::prefill_attn_steel_d256_supported(
        PrefillAttentionSpec {
            q_heads: 32,
            kv_heads: 4,
            head_dim: 128,
            rope_dims: 128,
            ..base
        }
    ));
}

#[test]
fn causal_attention_prefill_batch_long_matches_cpu_selected_rows_d128_d256() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    for (seq, q_heads, kv_heads, head_dim, pipeline, eps) in [
        (
            2049_usize,
            4_usize,
            1_usize,
            128_usize,
            &executor.causal_attention_prefill_batch_long_d128_f32,
            7.0e-4_f32,
        ),
        (
            4096_usize,
            4_usize,
            1_usize,
            128_usize,
            &executor.causal_attention_prefill_batch_long_d128_f32,
            9.0e-4_f32,
        ),
        (
            1024_usize,
            4_usize,
            1_usize,
            128_usize,
            &executor.causal_attention_prefill_batch_long_d128_f32,
            7.0e-4_f32,
        ),
        (
            1024_usize,
            2_usize,
            1_usize,
            256_usize,
            &executor.causal_attention_prefill_batch_long_d256_f32,
            8.0e-4_f32,
        ),
        (
            2049_usize,
            2_usize,
            1_usize,
            256_usize,
            &executor.causal_attention_prefill_batch_long_d256_f32,
            9.0e-4_f32,
        ),
        (
            4096_usize,
            2_usize,
            1_usize,
            256_usize,
            &executor.causal_attention_prefill_batch_long_d256_f32,
            1.2e-3_f32,
        ),
        // Facteur GQA 8 = celui du 35B (16 q-heads / 2 kv-heads), têtes réduites.
        (
            4096_usize,
            8_usize,
            1_usize,
            256_usize,
            &executor.causal_attention_prefill_batch_long_d256_f32,
            1.2e-3_f32,
        ),
    ] {
        let (q, k, v) = causal_attention_inputs(seq, q_heads, kv_heads, head_dim)?;
        let got = causal_attention_prefill_with_pipeline(
            &executor, &q, &k, &v, q_heads, kv_heads, pipeline, 32,
        )?;
        // Lignes sondées selon le régime : frontières 2047/2048 quand la
        // séquence les contient (cas long), sinon des lignes VALIDES du cas
        // court/mid (seq=1024 : le levier 30B route aussi 257..=2048 sur ce
        // kernel — conflit sémantique de rebase D-PLONG-B × D-NAPRE-2 corrigé).
        let rows = if seq > 3000 {
            vec![0, 1, 2048, seq - 1]
        } else if seq > 2048 {
            vec![0, 1, 2047, 2048]
        } else {
            vec![0, 1, seq / 2, seq - 1]
        };
        let expected =
            cpu_causal_attention_selected_rows_gqa(&q, &k, &v, q_heads, kv_heads, &rows)?;
        assert_causal_rows_close(&got, &expected, eps)?;
    }
    Ok(())
}

#[test]
fn causal_attention_prefill_batch_gqa8x4_d256_matches_cpu_selected_rows() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    for seq in [1025_usize, 2049, 4097] {
        let (q, k, v) = causal_attention_inputs(seq, 8, 1, 256)?;
        let got = causal_attention_prefill_with_pipeline_grid_rows(
            &executor,
            &q,
            &k,
            &v,
            8,
            1,
            &executor.causal_attention_prefill_batch_gqa8x4_d256_f32,
            256,
            1,
            seq.div_ceil(4),
        )?;
        let rows = if seq > 3000 {
            vec![0, 1, 2048, seq - 1]
        } else if seq > 2048 {
            vec![0, 1, 2047, 2048]
        } else {
            vec![0, 1, seq / 2, seq - 1]
        };
        let expected = cpu_causal_attention_selected_rows_gqa(&q, &k, &v, 8, 1, &rows)?;
        assert_causal_rows_close(&got, &expected, 1.2e-3)?;
    }
    Ok(())
}

#[test]
fn causal_attention_prefill_steel_d256_matches_cpu_selected_rows() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let Some(pipeline) = executor.causal_attention_prefill_steel_d256_f32.as_ref() else {
        return Err(InferError::Metal(
            "pipeline steel causal d256 indisponible".to_string(),
        ));
    };
    for seq in [257_usize, 1025, 2049] {
        let (q, k, v) = causal_attention_inputs(seq, 8, 1, 256)?;
        let got =
            causal_attention_prefill_with_steel_d256_pipeline(&executor, &q, &k, &v, 8, 1, pipeline)?;
        let rows = if seq > 2048 {
            vec![0, 1, 2047, 2048]
        } else {
            vec![0, 1, seq / 2, seq - 1]
        };
        let expected = cpu_causal_attention_selected_rows_gqa(&q, &k, &v, 8, 1, &rows)?;
        assert_causal_rows_close(&got, &expected, 2.0e-3)?;
    }
    Ok(())
}

/// Oracle CPU direct du kernel chunké-GQA `chunk_delta_seq_layout` (celui que
/// dispatche `encode_chunk_delta_seq_layout`, opt-in `RETI_RUST_LINEAR_CHUNKED`) :
/// compare y ET l'état SSM final contre `naive_gdn_reference` (récurrence GDN
/// séquentielle token-par-token, zéro chunking — voir la doc de module de
/// `linear_attention`). Longueurs aux frontières du chunk C=16 (1, 15, 16, 17,
/// 3×16+7) × deux découpages GQA (repeat 2 et 4), état initial non nul inclus.
///
/// Tolérance 5e-4 ABSOLUE, justifiée : entrées d'échelle réaliste O(1) (q/k
/// post-norm ~d_k^(−1)/d_k^(−1/2), v ∈ [−1,1]) ; le chunking ré-associe des
/// sommes f32 de 128 (dots sur d_k) + 16 (corrections intra-chunk) termes et
/// introduit des ratios γ_i/γ_j bornés ≤ 0,9⁻¹⁶ ≈ 5,4 (decay ∈ [0,9, 0,999)) →
/// bruit attendu ~1e-5, marge ×50, alignée sur le test batch résident dk128.
#[test]
fn chunk_delta_seq_layout_gqa_matches_naive_oracle() -> Result<()> {
    use crate::linear_attention::tests::{
        naive_gdn_reference, synthetic_gdn_inputs, DeterministicF32, CHUNK_BOUNDARY_LENGTHS,
    };
    use crate::linear_attention::LinearAttentionConfig;

    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let head_layouts = [(2_usize, 4_usize), (1, 4)];
    for (layout_index, (num_key_heads, num_value_heads)) in head_layouts.into_iter().enumerate() {
        let config = LinearAttentionConfig {
            num_key_heads,
            num_value_heads,
            key_head_dim: 128,
            value_head_dim: 128,
            conv_kernel_dim: 4,
            rms_eps: 1.0e-6,
        };
        let key_dim = config.key_dim()?;
        let value_dim = config.value_dim()?;
        let conv_dim = 2 * key_dim + value_dim;
        let state_len = num_value_heads * config.value_head_dim * config.key_head_dim;
        for (length_index, &steps) in CHUNK_BOUNDARY_LENGTHS.iter().enumerate() {
            let seed = 0x6d4a_0000 + (layout_index * 16 + length_index) as u64;
            let inputs = synthetic_gdn_inputs(config, steps, seed);
            // conv_out interleavé [T, q̃‖k̃‖ṽ] : les zones q̃/k̃ restent du bruit
            // (le kernel ne lit que la zone ṽ), la zone ṽ reçoit les v de l'oracle.
            let mut rng = DeterministicF32::new(seed ^ 0xc0de);
            let mut conv_out = rng.fill(steps * conv_dim, -1.0, 1.0);
            for t in 0..steps {
                conv_out[t * conv_dim + 2 * key_dim..(t * conv_dim) + 2 * key_dim + value_dim]
                    .copy_from_slice(&inputs.v[t * value_dim..(t + 1) * value_dim]);
            }
            // seq=17 traverse une frontière de chunk AVEC un S₀ non nul : le terme
            // γ_i·(S₀·k_i) et le report d'état inter-chunk sont exercés ensemble.
            let initial_state = if steps == 17 {
                rng.fill(state_len, -0.1, 0.1)
            } else {
                vec![0.0_f32; state_len]
            };

            let mut oracle_state = initial_state.clone();
            let oracle_y = naive_gdn_reference(
                &inputs.q,
                &inputs.k,
                &inputs.v,
                &inputs.g,
                &inputs.beta,
                steps,
                config,
                &mut oracle_state,
            );

            let shared_f32 = |data: &[f32]| {
                executor.device.new_buffer_with_data(
                    data.as_ptr().cast::<std::ffi::c_void>(),
                    (data.len() * 4) as u64,
                    MTLResourceOptions::StorageModeShared,
                )
            };
            let conv_out_buf = shared_f32(&conv_out);
            let q_buf = shared_f32(&inputs.q);
            let k_buf = shared_f32(&inputs.k);
            let beta_buf = shared_f32(&inputs.beta);
            let decay_buf = shared_f32(&inputs.g);
            let state_buf = shared_f32(&initial_state);
            let y_buf = executor.device.new_buffer(
                (steps * value_dim * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            let cb = executor.queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            let encoded = executor.encode_chunk_delta_seq_layout(
                encoder,
                &conv_out_buf,
                &q_buf,
                &k_buf,
                &beta_buf,
                &decay_buf,
                &state_buf,
                &y_buf,
                steps,
                LinearAttentionStepSpec {
                    num_key_heads: config.num_key_heads,
                    num_value_heads: config.num_value_heads,
                    key_head_dim: config.key_head_dim,
                    value_head_dim: config.value_head_dim,
                    conv_kernel_dim: config.conv_kernel_dim,
                    rms_eps: config.rms_eps,
                },
            );
            encoder.end_encoding();
            match encoded {
                Ok(()) => {}
                // Pipeline chunké non compilé sur cette machine → test sans objet.
                Err(InferError::Config(_)) => return Ok(()),
                Err(error) => return Err(error),
            }
            commit_and_wait(cb)?;

            let gpu_y = read_f32_buffer(&y_buf, steps * value_dim)?;
            let gpu_state = read_f32_buffer(&state_buf, state_len)?;
            assert_close_eps(&gpu_y, &oracle_y, 5.0e-4);
            assert_close_eps(&gpu_state, &oracle_state, 5.0e-4);
        }
    }
    Ok(())
}

// Verrouille la CLASSE caractérisée dans `docs/reviews/35b-prefill-drift-dossier.md` :
// AU-DELÀ de 256 tokens, la forme CHUNKÉE de l'attention linéaire (kernel
// `chunk_delta_seq_layout`, empruntée par les chemins prefill résident ET batché du
// 35B) reste dans la tolérance 5e-4 du SCAN séquentiel token-par-token (la référence
// per-op). Les cas de frontière de chunk existants plafonnent à seq=55 ; ce cas
// traverse 256 (seq=300 = 18 chunks pleins + 12), soit le régime où la ré-association
// f32 inter-chunk était soupçonnée de dériver. Il prouve que l'écart observé au niveau
// modèle (dossier) est un near-tie (flip d'argmax sur clusters serrés), PAS une
// divergence numérique du kernel : le prefill résident du 35B reste byte-identique au
// scan per-op sur texte réel à 608/1329/6007 tokens.
#[test]
fn chunk_delta_seq_layout_matches_scan_beyond_256() -> Result<()> {
    use crate::linear_attention::tests::{
        naive_gdn_reference, synthetic_gdn_inputs, DeterministicF32,
    };
    use crate::linear_attention::LinearAttentionConfig;

    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let config = LinearAttentionConfig {
        num_key_heads: 2,
        num_value_heads: 4,
        key_head_dim: 128,
        value_head_dim: 128,
        conv_kernel_dim: 4,
        rms_eps: 1.0e-6,
    };
    // seq=300 > 256 : traverse la frontière plafonnée par les autres tests de chunk.
    let steps = 300_usize;
    let key_dim = config.key_dim()?;
    let value_dim = config.value_dim()?;
    let conv_dim = 2 * key_dim + value_dim;
    let state_len = config.num_value_heads * config.value_head_dim * config.key_head_dim;
    let seed = 0x0035_b300;
    let inputs = synthetic_gdn_inputs(config, steps, seed);
    // conv_out interleavé [T, q̃‖k̃‖ṽ] : seule la zone ṽ est lue par le kernel.
    let mut rng = DeterministicF32::new(seed ^ 0xc0de);
    let mut conv_out = rng.fill(steps * conv_dim, -1.0, 1.0);
    for t in 0..steps {
        conv_out[t * conv_dim + 2 * key_dim..(t * conv_dim) + 2 * key_dim + value_dim]
            .copy_from_slice(&inputs.v[t * value_dim..(t + 1) * value_dim]);
    }
    // S₀ non nul : exerce le report d'état inter-chunk sur toute la longueur.
    let initial_state = rng.fill(state_len, -0.1, 0.1);

    let mut oracle_state = initial_state.clone();
    let oracle_y = naive_gdn_reference(
        &inputs.q,
        &inputs.k,
        &inputs.v,
        &inputs.g,
        &inputs.beta,
        steps,
        config,
        &mut oracle_state,
    );

    let shared_f32 = |data: &[f32]| {
        executor.device.new_buffer_with_data(
            data.as_ptr().cast::<std::ffi::c_void>(),
            (data.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    };
    let conv_out_buf = shared_f32(&conv_out);
    let q_buf = shared_f32(&inputs.q);
    let k_buf = shared_f32(&inputs.k);
    let beta_buf = shared_f32(&inputs.beta);
    let decay_buf = shared_f32(&inputs.g);
    let state_buf = shared_f32(&initial_state);
    let y_buf = executor.device.new_buffer(
        (steps * value_dim * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let cb = executor.queue.new_command_buffer();
    let encoder = cb.new_compute_command_encoder();
    let encoded = executor.encode_chunk_delta_seq_layout(
        encoder,
        &conv_out_buf,
        &q_buf,
        &k_buf,
        &beta_buf,
        &decay_buf,
        &state_buf,
        &y_buf,
        steps,
        LinearAttentionStepSpec {
            num_key_heads: config.num_key_heads,
            num_value_heads: config.num_value_heads,
            key_head_dim: config.key_head_dim,
            value_head_dim: config.value_head_dim,
            conv_kernel_dim: config.conv_kernel_dim,
            rms_eps: config.rms_eps,
        },
    );
    encoder.end_encoding();
    match encoded {
        Ok(()) => {}
        // Pipeline chunké non compilé sur cette machine → test sans objet.
        Err(InferError::Config(_)) => return Ok(()),
        Err(error) => return Err(error),
    }
    commit_and_wait(cb)?;

    let gpu_y = read_f32_buffer(&y_buf, steps * value_dim)?;
    let gpu_state = read_f32_buffer(&state_buf, state_len)?;
    assert_close_eps(&gpu_y, &oracle_y, 5.0e-4);
    assert_close_eps(&gpu_state, &oracle_state, 5.0e-4);
    Ok(())
}

// (e) Parité GPU vs CPU du gating top-k (« norm_topk_prob ») sur entrées
// aléatoires seedées : mêmes tailles réalistes que le routage prod (jusqu'à
// 128 experts / top-8), kernel réel via encode_topk_softmax.
#[test]
fn topk_softmax_gpu_matches_cpu_routing() -> Result<()> {
    use proptest::prelude::*;
    use proptest::test_runner::{TestCaseError, TestRunner};

    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let cases = (
        prop_oneof![Just(8_usize), Just(64), Just(128)],
        prop_oneof![Just(1_usize), Just(4), Just(8)],
    )
        .prop_flat_map(|(expert_count, top_k)| {
            (
                proptest::collection::vec(-1.0e4_f32..1.0e4, expert_count),
                Just(top_k),
            )
        });
    // RNG déterministe : les cas sont seedés, la reproduction est exacte.
    let mut runner = TestRunner::deterministic();
    runner
        .run(&cases, |(logits, top_k)| {
            let (gpu_indices, gpu_scores) = gpu_topk_softmax(&executor, &logits, top_k)
                .map_err(|error| TestCaseError::fail(error.to_string()))?;
            let (cpu_indices, cpu_scores) = cpu_topk_softmax_reference(&logits, top_k);
            prop_assert_eq!(gpu_indices.len(), top_k);
            // Les égalités se départagent par indice des deux côtés, mais ±0.0
            // diverge (total_cmp ordonne -0.0 < +0.0, le kernel les traite
            // égaux) : on compare les LOGITS sélectionnés rang par rang, et les
            // indices exacts seulement quand tous les logits sont distincts.
            for (rank, (&gpu_idx, &cpu_idx)) in gpu_indices.iter().zip(&cpu_indices).enumerate() {
                prop_assert!(
                    (gpu_idx as usize) < logits.len(),
                    "indice GPU hors plage: {}",
                    gpu_idx
                );
                let gpu_value = logits[gpu_idx as usize];
                let cpu_value = logits[cpu_idx];
                prop_assert!(
                    gpu_value == cpu_value,
                    "rang {}: logit GPU {} != CPU {}",
                    rank,
                    gpu_value,
                    cpu_value
                );
            }
            let mut sorted = logits.clone();
            sorted.sort_by(f32::total_cmp);
            let distinct = sorted
                .windows(2)
                .all(|pair| pair[0].total_cmp(&pair[1]).is_ne());
            if distinct {
                let gpu_as_usize = gpu_indices
                    .iter()
                    .map(|&idx| idx as usize)
                    .collect::<Vec<_>>();
                prop_assert_eq!(&gpu_as_usize, &cpu_indices, "sélection GPU != CPU");
            }
            let mut sum = 0.0_f32;
            for (rank, (gpu, cpu)) in gpu_scores.iter().zip(&cpu_scores).enumerate() {
                prop_assert!(
                    (gpu - cpu).abs() <= 1.0e-5,
                    "rang {}: score GPU {} vs CPU {}",
                    rank,
                    gpu,
                    cpu
                );
                sum += gpu;
            }
            prop_assert!((sum - 1.0).abs() <= 1.0e-4, "somme GPU = {}", sum);
            Ok(())
        })
        .map_err(|error| InferError::Metal(format!("proptest topk GPU: {error}")))?;
    Ok(())
}

/// Encode le kernel top-k réel sur `logits` et relit indices + scores.
fn gpu_topk_softmax(
    executor: &MetalExecutor,
    logits: &[f32],
    top_k: usize,
) -> Result<(Vec<u32>, Vec<f32>)> {
    let logits_buffer = executor.upload_f32_buffer(logits, "topk_prop_logits")?;
    let indices_buffer = executor.uncached_u32_buffer(top_k, "topk_prop_indices")?;
    let scores_buffer = executor.uncached_f32_buffer(top_k, "topk_prop_scores")?;
    let command_buffer = executor.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    let mut owned = Vec::new();
    let encoded = executor.encode_topk_softmax(
        encoder,
        &mut owned,
        &logits_buffer,
        &indices_buffer,
        &scores_buffer,
        logits.len(),
        top_k,
    );
    encoder.end_encoding();
    encoded?;
    commit_and_wait(command_buffer)?;
    Ok((
        read_u32_buffer(&indices_buffer, top_k)?,
        read_f32_buffer(&scores_buffer, top_k)?,
    ))
}

/// Référence CPU du gating : tri stable décroissant (`total_cmp`) puis softmax
/// restreint aux k logits max — miroir de `mlp::top_weights` et du kernel.
fn cpu_topk_softmax_reference(logits: &[f32], top_k: usize) -> (Vec<usize>, Vec<f32>) {
    let mut indices: Vec<usize> = (0..logits.len()).collect();
    indices.sort_by(|left, right| logits[*right].total_cmp(&logits[*left]));
    indices.truncate(top_k);
    let max = indices
        .first()
        .map(|&idx| logits[idx])
        .expect("invariant: top_k >= 1");
    let exps: Vec<f32> = indices
        .iter()
        .map(|&idx| (logits[idx] - max).exp())
        .collect();
    let denom = exps.iter().sum::<f32>().max(1.0e-20);
    let scores = exps.iter().map(|&value| value / denom).collect();
    (indices, scores)
}
