/// Oracle BIT-EXACT qmm2 ↔ qmv aligned (gate E2.2) : sur les shapes denses du
/// 35B prod (qkv 9216×2048, o/out_proj 2048×4096, in_proj LA 12352×2048,
/// lm_head réduit 24832×2048 — le plein vocab est couvert par l'oracle e2e),
/// chaque ligne du qmm2 == la même ligne du qmv aligned, en BITS (pas 1e-4).
#[test]
fn qmm2_bitwise_matches_qmv_aligned_on_prod_shapes() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    for (out_dim, in_dim) in [
        (9216_usize, 2048_usize),
        (2048, 4096),
        (12352, 2048),
        (24832, 2048),
        // Shared expert MoE (E2.3) : gate/up 512×2048, down 2048×512.
        (512, 2048),
        (2048, 512),
    ] {
        let weight = test_affine_varied(out_dim, in_dim)?;
        let packed = executor.buffer_from_slice(weight.packed_data(), "qmm2_bits_packed")?;
        let scales =
            executor.buffer_from_f32_as_bf16(weight.scales().data(), "qmm2_bits_scales")?;
        let biases =
            executor.buffer_from_f32_as_bf16(weight.biases().data(), "qmm2_bits_biases")?;
        let mut lhs = varied_row(in_dim, 1);
        lhs.extend_from_slice(&varied_row(in_dim, 2));
        let lhs_buf = executor.upload_f32_buffer(&lhs, "qmm2_bits_lhs")?;
        let out_qmv = executor.uncached_f32_buffer(2 * out_dim, "qmm2_bits_out_qmv")?;
        let out_qmm2 = executor.uncached_f32_buffer(2 * out_dim, "qmm2_bits_out_qmm2")?;
        let packed_cols = in_dim / 8;
        let groups = in_dim / 64;
        let dims: [u32; 4] = [
            out_dim as u32,
            in_dim as u32,
            packed_cols as u32,
            groups as u32,
        ];

        let command_buffer = executor.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        // Référence : qmv aligned batch=2 (2 lignes indépendantes, grille x=2).
        encoder.set_compute_pipeline_state(&executor.affine_qmv_fast_aligned_u4_gs64_f32);
        encoder.set_buffer(0, Some(&lhs_buf), 0);
        encoder.set_buffer(1, Some(&packed), 0);
        encoder.set_buffer(2, Some(&scales), 0);
        encoder.set_buffer(3, Some(&biases), 0);
        encoder.set_buffer(4, Some(&out_qmv), 0);
        encoder.set_bytes(5, 16, dims.as_ptr().cast());
        encoder.dispatch_thread_groups(
            MTLSize::new(2, (out_dim as u64).div_ceil(8), 1),
            MTLSize::new(64, 1, 1),
        );
        // Candidat : qmm2 (poids lus une fois, 2 lignes accumulées séparément).
        encoder.set_compute_pipeline_state(&executor.affine_qmm2_fast_aligned_u4_gs64_f32);
        encoder.set_buffer(0, Some(&lhs_buf), 0);
        encoder.set_buffer(1, Some(&packed), 0);
        encoder.set_buffer(2, Some(&scales), 0);
        encoder.set_buffer(3, Some(&biases), 0);
        encoder.set_buffer(4, Some(&out_qmm2), 0);
        encoder.set_bytes(5, 16, dims.as_ptr().cast());
        encoder.dispatch_thread_groups(
            MTLSize::new(1, (out_dim as u64).div_ceil(8), 1),
            MTLSize::new(64, 1, 1),
        );
        encoder.end_encoding();
        commit_and_wait(command_buffer)?;

        let qmv = read_f32_buffer(&out_qmv, 2 * out_dim)?;
        let qmm2 = read_f32_buffer(&out_qmm2, 2 * out_dim)?;
        assert_bits_equal(&qmm2, &qmv, &format!("qmm2 vs qmv ({out_dim}x{in_dim})"));
    }
    Ok(())
}

/// Oracle BIT-EXACT qmm2 u8 ↔ qmv u8 aligned (gate E2.2, modèles DWQ) : sur
/// les shapes denses 8-bit du 35B prod, chaque ligne du qmm2 u8 == la même
/// ligne du qmv u8 aligned, en bits. Vérifie aussi que le ROUTAGE batch=2
/// 8-bit de `encode_matmul_weight_buffers` sélectionne bien le qmm2 u8.
#[test]
fn qmm2_u8_bitwise_matches_qmv_u8_aligned_on_prod_shapes() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    for (out_dim, in_dim) in [
        (9216_usize, 2048_usize),
        (2048, 4096),
        (12352, 2048),
        (24832, 2048),
        // Router MoE (E2.3) : 256×2048 (8-bit dans le DWQ).
        (256, 2048),
    ] {
        for group_size in [64_usize, 128] {
            let weight = test_affine_varied_u8_group(out_dim, in_dim, group_size)?;
            let resolved = executor.resolve_linear_weight_buffers(
                &LinearWeight::AffineQuantized(weight.clone()),
                "qmm2_u8_weight",
            )?;
            let mut lhs = varied_row(in_dim, 3);
            lhs.extend_from_slice(&varied_row(in_dim, 4));
            let lhs_buf = executor.upload_f32_buffer(&lhs, "qmm2_u8_lhs")?;
            let out_route = executor.uncached_f32_buffer(2 * out_dim, "qmm2_u8_out_route")?;
            let qmv = fast_qmv_u8_reference(&executor, &weight, &lhs, 2, "qmm2_u8_ref")?;

            let command_buffer = executor.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            // Candidat : routage batch=2 → qmm2 u8 (poids lus une fois).
            let routed = executor.encode_matmul_weight_buffers(
                encoder, &lhs_buf, 2, in_dim, &resolved, &out_route, false,
            )?;
            assert_eq!(routed, out_dim);
            encoder.end_encoding();
            commit_and_wait(command_buffer)?;

            let route = read_f32_buffer(&out_route, 2 * out_dim)?;
            assert_bits_equal(
                &route,
                &qmv,
                &format!("qmm2 u8 gs{group_size} vs qmv u8 ({out_dim}x{in_dim})"),
            );
        }
    }
    Ok(())
}

#[test]
fn moe_shared_batch2_matches_two_solo_rows() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let hidden = 512_usize;
    let inter = 512_usize;
    let expert_count = 16_usize;
    let top_k = 2_usize;
    let group_size = 128_usize;
    let router = test_quant_linear_u8_group(expert_count, hidden, group_size)?;
    let experts = (0..expert_count)
        .map(|_| test_expert_u8_group(hidden, inter, group_size))
        .collect::<Result<Vec<_>>>()?;
    let shared_expert = test_expert_u8_group(hidden, inter, group_size)?;
    let shared_gate = test_quant_linear_u8_group(1, hidden, group_size)?;
    let row0 = varied_row(hidden, 31);
    let row1 = varied_row(hidden, 47);
    let mut input = row0.clone();
    input.extend_from_slice(&row1);
    let batch_input = Tensor::from_vec(vec![2, hidden], input)?;

    let batch = executor.moe_gated_router_topk_shared_batch2(
        &batch_input,
        &router,
        &experts,
        top_k,
        &shared_expert,
        &shared_gate,
    )?;
    let solo0 = executor.moe_gated_router_topk_shared(
        &Tensor::from_vec(vec![1, hidden], row0)?,
        &router,
        &experts,
        top_k,
        &shared_expert,
        &shared_gate,
    )?;
    let solo1 = executor.moe_gated_router_topk_shared(
        &Tensor::from_vec(vec![1, hidden], row1)?,
        &router,
        &experts,
        top_k,
        &shared_expert,
        &shared_gate,
    )?;

    assert_eq!(batch.shape(), [2, hidden]);
    assert_close_eps(&batch.data()[0..hidden], solo0.data(), 1.0e-4);
    assert_close_eps(&batch.data()[hidden..2 * hidden], solo1.data(), 1.0e-4);
    Ok(())
}

#[test]
fn moe_shared_batch2_accepts_dense_router() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let hidden = 512_usize;
    let inter = 512_usize;
    let expert_count = 16_usize;
    let top_k = 2_usize;
    let group_size = 128_usize;
    let router = test_dense_linear(expert_count, hidden)?;
    let experts = (0..expert_count)
        .map(|_| test_expert_u8_group(hidden, inter, group_size))
        .collect::<Result<Vec<_>>>()?;
    let shared_expert = test_expert_u8_group(hidden, inter, group_size)?;
    let shared_gate = test_quant_linear_u8_group(1, hidden, group_size)?;
    let row0 = varied_row(hidden, 59);
    let row1 = varied_row(hidden, 83);
    let mut input = row0.clone();
    input.extend_from_slice(&row1);
    let batch_input = Tensor::from_vec(vec![2, hidden], input)?;

    let batch = executor.moe_gated_router_topk_shared_batch2(
        &batch_input,
        &router,
        &experts,
        top_k,
        &shared_expert,
        &shared_gate,
    )?;
    let solo0 = executor.moe_gated_router_topk_shared(
        &Tensor::from_vec(vec![1, hidden], row0)?,
        &router,
        &experts,
        top_k,
        &shared_expert,
        &shared_gate,
    )?;
    let solo1 = executor.moe_gated_router_topk_shared(
        &Tensor::from_vec(vec![1, hidden], row1)?,
        &router,
        &experts,
        top_k,
        &shared_expert,
        &shared_gate,
    )?;

    assert_eq!(batch.shape(), [2, hidden]);
    assert_close_eps(&batch.data()[0..hidden], solo0.data(), 1.0e-4);
    assert_close_eps(&batch.data()[hidden..2 * hidden], solo1.data(), 1.0e-4);
    Ok(())
}

#[test]
fn moe_shared_rows_matches_solo_rows() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let hidden = 512_usize;
    let inter = 512_usize;
    let expert_count = 16_usize;
    let top_k = 2_usize;
    let group_size = 128_usize;
    let router = test_quant_linear_u8_group(expert_count, hidden, group_size)?;
    let experts = (0..expert_count)
        .map(|_| test_expert_u8_group(hidden, inter, group_size))
        .collect::<Result<Vec<_>>>()?;
    let shared_expert = test_expert_u8_group(hidden, inter, group_size)?;
    let shared_gate = test_quant_linear_u8_group(1, hidden, group_size)?;
    let rows = [
        varied_row(hidden, 101),
        varied_row(hidden, 103),
        varied_row(hidden, 107),
        varied_row(hidden, 109),
    ];
    let mut input = Vec::with_capacity(rows.len() * hidden);
    for row in &rows {
        input.extend_from_slice(row);
    }
    let batch_input = Tensor::from_vec(vec![rows.len(), hidden], input)?;

    let batch = executor.moe_gated_router_topk_shared_rows(
        &batch_input,
        &router,
        &experts,
        top_k,
        &shared_expert,
        &shared_gate,
    )?;

    assert_eq!(batch.shape(), [rows.len(), hidden]);
    for (row_index, row) in rows.iter().enumerate() {
        let solo = executor.moe_gated_router_topk_shared(
            &Tensor::from_vec(vec![1, hidden], row.clone())?,
            &router,
            &experts,
            top_k,
            &shared_expert,
            &shared_gate,
        )?;
        let start = row_index * hidden;
        assert_close_eps(&batch.data()[start..start + hidden], solo.data(), 1.0e-4);
    }
    Ok(())
}

/// Oracle BIT-EXACT dé-fusion shared expert (gate E2.3) : le fusé
/// `gate_up_swiglu_fast` du solo == qmm2(gate) + qmm2(up) + swiglu élémentaire
/// par ligne (l'expression silu est identique dans les deux kernels), sur la
/// shape shared du 35B prod (512×2048, 4-bit gs64).
#[test]
fn shared_expert_duo_bitwise_matches_fused_gate_up_swiglu() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let (out_dim, in_dim) = (512_usize, 2048_usize);
    let gate = executor.resolve_linear_weight_buffers(
        &LinearWeight::AffineQuantized(test_affine_varied(out_dim, in_dim)?),
        "moe_duo_gate",
    )?;
    // Poids up DIFFÉRENTS du gate (salt via u8 → distribution distincte).
    let up_affine = {
        let mut affine = test_affine_varied(out_dim, in_dim)?;
        // Décale les scales pour différencier up de gate (déterministe).
        let scales = affine.scales().data().iter().map(|s| s * 1.5).collect();
        affine = AffineQuantizedTensor::new(
            &[out_dim, in_dim / 8],
            affine.packed_data().to_vec(),
            Tensor::from_vec(vec![out_dim, in_dim / 64], scales)?,
            affine.biases().clone(),
            64,
            4,
        )?;
        affine
    };
    let up = executor
        .resolve_linear_weight_buffers(&LinearWeight::AffineQuantized(up_affine), "moe_duo_up")?;
    let rows = [varied_row(in_dim, 21), varied_row(in_dim, 22)];

    // Duo : qmm2 gate + qmm2 up + swiglu élémentaire sur [2, out_dim].
    let mut lhs = rows[0].clone();
    lhs.extend_from_slice(&rows[1]);
    let lhs_buf = executor.upload_f32_buffer(&lhs, "moe_duo_lhs")?;
    let gate2 = executor.uncached_f32_buffer(2 * out_dim, "moe_duo_gate2")?;
    let up2 = executor.uncached_f32_buffer(2 * out_dim, "moe_duo_up2")?;
    let swiglu2 = executor.uncached_f32_buffer(2 * out_dim, "moe_duo_swiglu2")?;
    let command_buffer = executor.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    let mut owned = Vec::new();
    let g = executor
        .encode_matmul_weight_buffers(encoder, &lhs_buf, 2, in_dim, &gate, &gate2, false)?;
    let u =
        executor.encode_matmul_weight_buffers(encoder, &lhs_buf, 2, in_dim, &up, &up2, false)?;
    assert_eq!((g, u), (out_dim, out_dim));
    executor.encode_swiglu(encoder, &mut owned, &gate2, &up2, &swiglu2, 2 * out_dim)?;
    encoder.end_encoding();
    commit_and_wait(command_buffer)?;
    let duo = read_f32_buffer(&swiglu2, 2 * out_dim)?;

    // Référence solo : kernel fusionné par ligne.
    for (index, row) in rows.iter().enumerate() {
        let row_buf = executor.upload_f32_buffer(row, "moe_duo_row")?;
        let fused_out = executor.uncached_f32_buffer(out_dim, "moe_duo_fused")?;
        let command_buffer = executor.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let engaged = executor.encode_gate_up_swiglu_fast_buffers(
            encoder, &row_buf, &gate, &up, &fused_out, in_dim,
        )?;
        assert!(
            engaged,
            "le fusé gate_up_swiglu doit s'appliquer (4-bit gs64)"
        );
        encoder.end_encoding();
        commit_and_wait(command_buffer)?;
        let fused = read_f32_buffer(&fused_out, out_dim)?;
        assert_bits_equal(
            &duo[index * out_dim..(index + 1) * out_dim],
            &fused,
            &format!("shared expert ligne {index}"),
        );
    }
    Ok(())
}

/// Prédicats duo : éligibilité qmm2 (u4 ET u8, jamais Dense) et choix du norm
/// en miroir du solo (rms_simd ssi le solo fusionne le prologue).
#[test]
fn duo_predicates_follow_solo_routing() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let u4 = executor.resolve_linear_weight_buffers(
        &LinearWeight::AffineQuantized(test_affine_varied(2048, 4096)?),
        "duo_pred_u4",
    )?;
    let u8 = executor.resolve_linear_weight_buffers(
        &LinearWeight::AffineQuantized(test_affine_varied_u8(2048, 4096)?),
        "duo_pred_u8",
    )?;
    let dense = executor.resolve_linear_weight_buffers(
        &test_dense_linear(64, 64)?.weight().clone(),
        "duo_pred_dense",
    )?;
    assert!(executor.qmm2_eligible_weight(&u4));
    assert!(executor.qmm2_eligible_weight(&u8));
    assert!(!executor.qmm2_eligible_weight(&dense));

    assert!(executor.solo_rms_fusion_applies(&u4, 4096, true));
    assert!(executor.solo_rms_fusion_applies(&u4, 4096, false));
    assert!(executor.solo_rms_fusion_applies(&u8, 4096, true));
    assert!(executor.solo_rms_fusion_applies(&u8, 4096, false));
    assert!(!executor.solo_rms_fusion_applies(&dense, 64, false));
    Ok(())
}

/// Oracle BIT-EXACT dé-fusion rms (gate E2.2) : `rms_norm_simd` (nouveau kernel,
/// même réduction que le prologue fusionné) suivi du qmv == le kernel FUSIONNÉ
/// `affine_qmv_rms_fast` (chemin solo prod du in_proj linear-attn), en bits.
#[test]
fn rms_simd_then_qmv_bitwise_matches_fused_rms_qmv() -> Result<()> {
    let (out_dim, in_dim) = (12352_usize, 2048_usize);
    let affine = test_affine_varied(out_dim, in_dim)?;
    assert_rms_simd_then_qmv_bitwise_matches_fused(affine, out_dim, in_dim, "u4_gs64")
}

#[test]
fn rms_simd_then_qmv_bitwise_matches_fused_rms_qmv_u8_gs64() -> Result<()> {
    let (out_dim, in_dim) = (12352_usize, 2048_usize);
    let affine = test_affine_varied_u8_group(out_dim, in_dim, 64)?;
    assert_rms_simd_then_qmv_bitwise_matches_fused(affine, out_dim, in_dim, "u8_gs64")
}

#[test]
fn rms_simd_then_qmv_bitwise_matches_fused_rms_qmv_u8_gs128() -> Result<()> {
    let (out_dim, in_dim) = (12352_usize, 2048_usize);
    let affine = test_affine_varied_u8_group(out_dim, in_dim, 128)?;
    assert_rms_simd_then_qmv_bitwise_matches_fused(affine, out_dim, in_dim, "u8_gs128")
}

fn assert_rms_simd_then_qmv_bitwise_matches_fused(
    affine: AffineQuantizedTensor,
    out_dim: usize,
    in_dim: usize,
    label: &str,
) -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let eps = 1.0e-6_f32;
    let weight = executor
        .resolve_linear_weight_buffers(&LinearWeight::AffineQuantized(affine), "rms_bits_weight")?;
    let gamma: Vec<f32> = (0..in_dim)
        .map(|i| 0.9 + 0.01 * ((i % 17) as f32))
        .collect();
    let gamma_buf = executor.upload_f32_buffer(&gamma, "rms_bits_gamma")?;
    let x = varied_row(in_dim, 5);
    let x_buf = executor.upload_f32_buffer(&x, "rms_bits_x")?;
    let out_fused = executor.uncached_f32_buffer(out_dim, "rms_bits_out_fused")?;
    let normed = executor.uncached_f32_buffer(in_dim, "rms_bits_normed")?;
    let out_split = executor.uncached_f32_buffer(out_dim, "rms_bits_out_split")?;

    let command_buffer = executor.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    let fused = executor.encode_matmul_weight_buffers_rms_prologue(
        encoder, &x_buf, &gamma_buf, eps, in_dim, &weight, &out_fused,
    )?;
    assert_eq!(fused, Some(out_dim), "le chemin fusionné doit s'appliquer");
    executor.encode_rms_norm_simd_rows(encoder, &x_buf, &gamma_buf, &normed, 1, in_dim, eps)?;
    let split_out = executor
        .encode_matmul_weight_buffers(encoder, &normed, 1, in_dim, &weight, &out_split, false)?;
    assert_eq!(split_out, out_dim);
    encoder.end_encoding();
    commit_and_wait(command_buffer)?;

    let fused_values = read_f32_buffer(&out_fused, out_dim)?;
    let split_values = read_f32_buffer(&out_split, out_dim)?;
    assert_bits_equal(&split_values, &fused_values, label);
    Ok(())
}

/// Oracle BIT-EXACT dé-fusion qkv (gate E2.2) : `rms_norm_simd` rows=2 → qmm2
/// == le kernel fusionné `affine_qkv_split_rms_qmv_fast` (chemin solo prod du
/// qkv full-attn) par ligne — q/gate désinterleavés comparés via le remap
/// d'index, k/v comparés sur la zone au-delà de `q_gate_dim`.
#[test]
fn rms_simd_qmm2_bitwise_matches_fused_qkv_split() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let (q_heads, head_dim) = (16_usize, 256_usize);
    let q_gate_dim = q_heads * head_dim * 2;
    let (out_dim, in_dim) = (9216_usize, 2048_usize);
    let q_dim = q_heads * head_dim;
    let eps = 1.0e-6_f32;
    let affine = test_affine_varied(out_dim, in_dim)?;
    let weight = executor
        .resolve_linear_weight_buffers(&LinearWeight::AffineQuantized(affine), "qkv_bits_weight")?;
    let gamma: Vec<f32> = (0..in_dim)
        .map(|i| 1.1 - 0.005 * ((i % 23) as f32))
        .collect();
    let gamma_buf = executor.upload_f32_buffer(&gamma, "qkv_bits_gamma")?;
    let rows = [varied_row(in_dim, 11), varied_row(in_dim, 13)];

    // Chemin duo : les 2 lignes normées (rms simd) puis qmm2 → [2, out_dim].
    let mut lhs = rows[0].clone();
    lhs.extend_from_slice(&rows[1]);
    let lhs_buf = executor.upload_f32_buffer(&lhs, "qkv_bits_lhs")?;
    let normed2 = executor.uncached_f32_buffer(2 * in_dim, "qkv_bits_normed2")?;
    let qkv2 = executor.uncached_f32_buffer(2 * out_dim, "qkv_bits_qkv2")?;
    let command_buffer = executor.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    executor.encode_rms_norm_simd_rows(encoder, &lhs_buf, &gamma_buf, &normed2, 2, in_dim, eps)?;
    let qmm2_out = executor
        .encode_matmul_weight_buffers(encoder, &normed2, 2, in_dim, &weight, &qkv2, false)?;
    assert_eq!(qmm2_out, out_dim);
    encoder.end_encoding();
    commit_and_wait(command_buffer)?;
    let duo = read_f32_buffer(&qkv2, 2 * out_dim)?;

    // Référence solo : kernel fusionné par ligne (q/gate scatter + k/v).
    for (index, row) in rows.iter().enumerate() {
        let row_buf = executor.upload_f32_buffer(row, "qkv_bits_row")?;
        let qkv_out = executor.uncached_f32_buffer(out_dim, "qkv_bits_qkv_solo")?;
        let q_out = executor.uncached_f32_buffer(q_dim, "qkv_bits_q_solo")?;
        let gate_out = executor.uncached_f32_buffer(q_dim, "qkv_bits_gate_solo")?;
        let command_buffer = executor.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let fused = executor.encode_full_attn_qkv_split_rms_buffers(
            encoder, &row_buf, &gamma_buf, eps, in_dim, &weight, &qkv_out, &q_out, &gate_out,
            q_heads, head_dim,
        )?;
        assert_eq!(
            fused,
            Some(out_dim),
            "le chemin qkv fusionné doit s'appliquer"
        );
        encoder.end_encoding();
        commit_and_wait(command_buffer)?;

        let solo_q = read_f32_buffer(&q_out, q_dim)?;
        let solo_gate = read_f32_buffer(&gate_out, q_dim)?;
        let solo_kv = read_f32_buffer(&qkv_out, out_dim)?;
        let duo_row = &duo[index * out_dim..(index + 1) * out_dim];
        for head in 0..q_heads {
            for col in 0..head_dim {
                let interleaved = head * 2 * head_dim + col;
                let q_val = solo_q[head * head_dim + col];
                let gate_val = solo_gate[head * head_dim + col];
                assert_bits_portable(q_val, duo_row[interleaved], &|| {
                    format!("q divergent (ligne {index}, tête {head}, col {col})")
                });
                assert_bits_portable(gate_val, duo_row[interleaved + head_dim], &|| {
                    format!("gate divergent (ligne {index}, tête {head}, col {col})")
                });
            }
        }
        assert_bits_equal(
            &duo_row[q_gate_dim..],
            &solo_kv[q_gate_dim..],
            &format!("k/v ligne {index}"),
        );
    }
    Ok(())
}

#[test]
fn rms_qmv_u8_bitwise_matches_fused_qkv_split() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    std::env::set_var("RETI_RUST_FULL_QKV_SPLIT_RMS_U8", "1");
    let (q_heads, head_dim) = (16_usize, 256_usize);
    let q_gate_dim = q_heads * head_dim * 2;
    let (out_dim, in_dim) = (9216_usize, 2048_usize);
    let q_dim = q_heads * head_dim;
    let eps = 1.0e-6_f32;
    let affine = test_affine_varied_u8_group(out_dim, in_dim, 64)?;
    let weight = executor
        .resolve_linear_weight_buffers(&LinearWeight::AffineQuantized(affine), "qkv_u8_weight")?;
    let gamma: Vec<f32> = (0..in_dim)
        .map(|i| 1.1 - 0.005 * ((i % 23) as f32))
        .collect();
    let row = varied_row(in_dim, 17);
    let row_buf = executor.upload_f32_buffer(&row, "qkv_u8_row")?;
    let gamma_buf = executor.upload_f32_buffer(&gamma, "qkv_u8_gamma")?;
    let normed = executor.uncached_f32_buffer(in_dim, "qkv_u8_normed")?;
    let split = executor.uncached_f32_buffer(out_dim, "qkv_u8_split")?;
    let fused_kv = executor.uncached_f32_buffer(out_dim, "qkv_u8_fused_kv")?;
    let fused_q = executor.uncached_f32_buffer(q_dim, "qkv_u8_fused_q")?;
    let fused_gate = executor.uncached_f32_buffer(q_dim, "qkv_u8_fused_gate")?;

    let command_buffer = executor.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    executor.encode_rms_norm_simd_rows(encoder, &row_buf, &gamma_buf, &normed, 1, in_dim, eps)?;
    let split_out = executor
        .encode_matmul_weight_buffers(encoder, &normed, 1, in_dim, &weight, &split, false)?;
    assert_eq!(split_out, out_dim);
    let fused = executor.encode_full_attn_qkv_split_rms_buffers(
        encoder,
        &row_buf,
        &gamma_buf,
        eps,
        in_dim,
        &weight,
        &fused_kv,
        &fused_q,
        &fused_gate,
        q_heads,
        head_dim,
    )?;
    assert_eq!(fused, Some(out_dim));
    encoder.end_encoding();
    commit_and_wait(command_buffer)?;

    let split_values = read_f32_buffer(&split, out_dim)?;
    let q_values = read_f32_buffer(&fused_q, q_dim)?;
    let gate_values = read_f32_buffer(&fused_gate, q_dim)?;
    let kv_values = read_f32_buffer(&fused_kv, out_dim)?;
    for head in 0..q_heads {
        for col in 0..head_dim {
            let interleaved = head * 2 * head_dim + col;
            assert_eq!(
                q_values[head * head_dim + col].to_bits(),
                split_values[interleaved].to_bits(),
                "q u8 divergent (tête {head}, col {col})"
            );
            assert_eq!(
                gate_values[head * head_dim + col].to_bits(),
                split_values[interleaved + head_dim].to_bits(),
                "gate u8 divergent (tête {head}, col {col})"
            );
        }
    }
    assert_bits_equal(
        &split_values[q_gate_dim..],
        &kv_values[q_gate_dim..],
        "k/v u8",
    );
    Ok(())
}
