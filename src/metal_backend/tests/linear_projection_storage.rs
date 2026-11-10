fn shared_la_fixture(
    executor: &MetalExecutor,
    shared: bool,
) -> Result<crate::linear_attention::LinearAttention> {
    use crate::linear_attention::{LinearAttention, LinearAttentionWeights};
    let mut layer = LinearAttention::new(LinearAttentionWeights {
        in_proj_qkv: rows_exact_u4(2304, 512, 21)?,
        in_proj_z: rows_exact_u4(2048, 512, 22)?,
        in_proj_b: rows_exact_u4(16, 512, 23)?,
        in_proj_a: rows_exact_u4(16, 512, 24)?,
        out_proj: rows_exact_u4(512, 2048, 25)?,
        conv1d_weight: Tensor::from_vec(
            vec![2304, 4, 1],
            (0..2304 * 4)
                .map(|i| (i as f32 * 0.013).sin() * 0.3)
                .collect(),
        )?,
        a_log: Tensor::from_vec(vec![16], vec![0.1; 16])?,
        dt_bias: Tensor::from_vec(vec![16], vec![-0.3; 16])?,
        norm_weight: Tensor::from_vec(vec![128], vec![1.0; 128])?,
    });
    crate::runtime_flags::with_la_shared_override(shared, || {
        layer.move_packed_to_metal_shared(executor)
    })?;
    Ok(layer)
}

fn shared_la_spec() -> (LinearAttentionStepSpec, LinearAttnResidentDims) {
    (
        LinearAttentionStepSpec {
            num_key_heads: 1,
            num_value_heads: 16,
            key_head_dim: 128,
            value_head_dim: 128,
            conv_kernel_dim: 4,
            rms_eps: 1e-6,
            l2_qk_sigmoid_gate: false,
        },
        LinearAttnResidentDims {
            in_dim: 512,
            conv_dim: 2304,
            value_dim: 2048,
            key_dim: 128,
        },
    )
}

fn shared_la_state(executor: &MetalExecutor) -> Result<LinearAttentionMetalState> {
    let (spec, dims) = shared_la_spec();
    let mut state = None;
    let conv = vec![0.125; 3 * dims.conv_dim];
    let ssm = vec![0.03125; 16 * 128 * 128];
    executor.ensure_linear_attention_metal_state(
        &mut state,
        &conv,
        &ssm,
        dims.conv_dim,
        conv.len(),
        ssm.len(),
        spec,
    )?;
    state.ok_or_else(|| InferError::Metal("état LA absent".into()))
}

fn shared_la_bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// Compare les mêmes appels de production, pas deux algorithmes de projection.
/// Pas de poids externes ; l'absence de Metal fait échouer explicitement le test.
#[test]
fn shared_la_decode_and_prefill_bit_exact() -> Result<()> {
    let executor = rows_exact_executor()?;
    executor.prepare_cpu_weight_release();
    let old = shared_la_fixture(&executor, false)?;
    let mut shared = shared_la_fixture(&executor, true)?;
    assert!(
        shared
            .resident_weights()
            .in_proj_z
            .affine_quantized()
            .expect("invariant: fixture u4")
            .relocated_u4
    );
    // Le drop CPU ne détruit aucune vue et la résolution reste possible ensuite.
    shared.release_affine_cpu_data(&mut Vec::new());
    let (spec, dims) = shared_la_spec();
    let old_resolved =
        executor.resolve_linear_attn_resident_dense_weights(old.resident_weights())?;
    let new_resolved =
        executor.resolve_linear_attn_resident_dense_weights(shared.resident_weights())?;
    let input_norm = executor.buffer_from_slice(&vec![1.0f32; 512], "shared_la_norm")?;
    for rows in [1, 2, 5, 17] {
        let x = rows_exact_lhs(rows, 512, 400, 0.05);
        let input = executor.buffer_from_slice(&x, "shared_la_input")?;
        // Decode résident complet et chemins alternatifs z|b|a et split.
        // Le prefill résident utilise les originaux via encode_matmul_weight.
        for mode in 0..4 {
            let mut results = Vec::new();
            for (layer, resolved) in [(&old, &old_resolved), (&shared, &new_resolved)] {
                let state = shared_la_state(&executor)?;
                let output = executor.uncached_f32_buffer(rows * 512, "shared_la_output")?;
                let mut owned = Vec::new();
                rows_exact_cb(&executor, |enc| {
                    if mode == 3 {
                        executor.encode_linear_attn_batch_resident(
                            enc,
                            &mut owned,
                            &input,
                            &output,
                            rows,
                            layer.resident_weights(),
                            &state,
                            spec,
                            dims,
                        )
                    } else {
                        let mut resolved = resolved.clone();
                        if mode >= 1 {
                            resolved.full = None;
                        }
                        if mode >= 2 {
                            resolved.z_beta_gate = None;
                        }
                        if rows == 1 {
                            executor.encode_linear_attn_resident_dense_buffers(
                                enc,
                                &input,
                                Some((&input_norm, 1e-6)),
                                &output,
                                &resolved,
                                &state,
                                spec,
                                dims,
                            )
                        } else {
                            executor.encode_linear_attn_resident_dense_buffers_rows(
                                enc,
                                &input,
                                Some((&input_norm, 1e-6)),
                                &output,
                                rows,
                                &resolved,
                                &state,
                                None,
                                spec,
                                dims,
                            )
                        }
                    }
                })?;
                results.push((
                    shared_la_bits(&read_f32_buffer(&output, rows * 512)?),
                    shared_la_bits(&read_f32_buffer(&state.conv, state.conv_len)?),
                    shared_la_bits(&read_f32_buffer(&state.ssm, state.ssm_len)?),
                ));
            }
            assert_eq!(results[0], results[1], "mode={mode}, rows={rows}");
        }
        // Per-op GPU et oracle CPU : les originaux restent utilisables.
        let tensor = Tensor::from_vec(vec![rows, 512], x)?;
        let a = old.resident_weights();
        let b = shared.resident_weights();
        for (old, shared) in [a.in_proj_qkv, a.in_proj_z, a.in_proj_b, a.in_proj_a]
            .into_iter()
            .zip([b.in_proj_qkv, b.in_proj_z, b.in_proj_b, b.in_proj_a])
        {
            assert_eq!(
                shared_la_bits(old.forward(&tensor)?.data()),
                shared_la_bits(shared.forward(&tensor)?.data())
            );
            let old = old.affine_quantized().expect("invariant: fixture u4");
            let shared = shared.affine_quantized().expect("invariant: fixture u4");
            assert_eq!(
                shared_la_bits(executor.matmul_rhs_t_affine(&tensor, old)?.data()),
                shared_la_bits(executor.matmul_rhs_t_affine(&tensor, shared)?.data())
            );
        }
    }
    Ok(())
}

#[test]
fn shared_la_allocated_bytes_and_lifetime() -> Result<()> {
    let executor = rows_exact_executor()?;
    executor.prepare_cpu_weight_release();
    let measure = |shared| -> Result<(u64, MetalLinearAttnResidentDenseWeights)> {
        let layer = shared_la_fixture(&executor, shared)?;
        let w = layer.resident_weights();
        let resolved = executor.resolve_linear_attn_resident_dense_weights(w)?;
        let mut allocations = std::collections::HashMap::new();
        let mut add = |weight: &MetalLinearWeightBuffers| {
            if let MetalLinearWeightBuffers::AffineQuantized {
                packed,
                scales,
                biases,
                ..
            } = weight
            {
                for buffer in [packed, scales, biases] {
                    allocations.insert(buffer.as_ptr() as usize, buffer.length());
                }
            }
        };
        for source in [w.in_proj_qkv, w.in_proj_z, w.in_proj_b, w.in_proj_a] {
            add(&executor.resolve_linear_weight_buffers(source.weight(), "shared_la_source")?);
        }
        add(&resolved
            .full
            .as_ref()
            .expect("invariant: full concat compatible")
            .in_proj);
        add(resolved
            .z_beta_gate
            .as_ref()
            .expect("invariant: zba compatible"));
        for pair in [&resolved.qkv_z, &resolved.beta_gate] {
            match pair {
                MetalLinearAttnResidentPairWeights::Concat(w) => add(w),
                MetalLinearAttnResidentPairWeights::Split { first, second } => {
                    add(first);
                    add(second);
                }
            }
        }
        assert_eq!(allocations.len(), if shared { 3 } else { 18 });
        Ok((allocations.values().sum(), resolved))
    };
    let (old_bytes, _) = measure(false)?;
    let (shared_bytes, resolved) = measure(true)?;
    let source_bytes = (2304 + 2048 + 16 + 16) * 512 * 9 / 16;
    let zba_bytes = (2048 + 16 + 16) * 512 * 9 / 16;
    assert_eq!(shared_bytes, source_bytes);
    assert_eq!(old_bytes - shared_bytes, source_bytes + zba_bytes);
    // Les handles résolus survivent au propriétaire des poids.
    let input = executor.buffer_from_slice(&vec![0.05f32; 512], "shared_la_lifetime")?;
    let output = executor.uncached_f32_buffer(4384, "shared_la_lifetime_out")?;
    rows_exact_cb(&executor, |enc| {
        executor
            .encode_matmul_weight_buffers(
                enc,
                &input,
                1,
                512,
                &resolved
                    .full
                    .as_ref()
                    .expect("invariant: concat présent")
                    .in_proj,
                &output,
                false,
            )
            .map(|_| ())
    })?;
    assert!(read_f32_buffer(&output, 4384)?
        .iter()
        .all(|v| v.is_finite()));
    Ok(())
}

#[test]
fn shared_la_relocation_releases_sources_and_skips_incompatible() -> Result<()> {
    let executor = rows_exact_executor()?;
    let make = |heads| -> Result<[Linear; 4]> {
        let mut projections = [
            rows_exact_u4(2304, 512, 31)?,
            rows_exact_u4(2048, 512, 32)?,
            rows_exact_u4(heads, 512, 33)?,
            rows_exact_u4(heads, 512, 34)?,
        ];
        for linear in &mut projections {
            linear.move_packed_to_metal_shared(&executor)?;
        }
        Ok(projections)
    };
    let mut projections = make(16)?;
    let before = executor.device.current_allocated_size();
    executor.share_linear_attn_projections(projections.each_mut())?;
    let after = executor.device.current_allocated_size();
    // Le compteur device est global : les autres tests peuvent allouer en parallèle.
    eprintln!("shared_la device allocated bytes: {before} -> {after}");
    let ptr = projections[0]
        .affine_quantized()
        .expect("invariant: u4")
        .packed_metal_view()
        .expect("invariant: vue")
        .0
        .as_ptr();
    executor.share_linear_attn_projections(projections.each_mut())?;
    assert_eq!(
        ptr,
        projections[0]
            .affine_quantized()
            .expect("invariant: u4")
            .packed_metal_view()
            .expect("invariant: vue")
            .0
            .as_ptr()
    );
    // Paramètres bf16 b/a de 48 octets : pas d'alignement 256, pas de déplacement.
    let mut unaligned = make(3)?;
    executor.share_linear_attn_projections(unaligned.each_mut())?;
    assert!(unaligned
        .iter()
        .all(|w| !w.affine_quantized().expect("invariant: u4").relocated_u4));
    let mut mixed = make(16)?;
    mixed[2] = Linear::new(Tensor::from_vec(vec![16, 512], vec![0.0; 16 * 512])?, None)?;
    executor.share_linear_attn_projections(mixed.each_mut())?;
    assert!(
        !mixed[0]
            .affine_quantized()
            .expect("invariant: u4")
            .relocated_u4
    );
    // Une vue de pool/shard n'est pas recopiée, même avec des offsets alignés.
    let mut pooled = make(16)?;
    let tensor = pooled[0].affine_quantized_mut().expect("invariant: u4");
    let buffer = executor.weight_buffer_from_u32(&vec![0; tensor.packed_len() + 64])?;
    tensor.set_packed_metal_view(buffer.clone(), 64)?;
    executor.share_linear_attn_projections(pooled.each_mut())?;
    assert_eq!(
        buffer.as_ptr(),
        pooled[0]
            .affine_quantized()
            .expect("invariant: u4")
            .packed_metal_view()
            .expect("invariant: vue")
            .0
            .as_ptr()
    );
    Ok(())
}

#[test]
fn shared_la_offset_kernels_bit_exact() -> Result<()> {
    let executor = rows_exact_executor()?;
    let old = shared_la_fixture(&executor, false)?;
    let shared = shared_la_fixture(&executor, true)?;
    let input = executor.buffer_from_slice(
        &rows_exact_lhs(64, 512, 444, 0.05),
        "shared_la_kernel_input",
    )?;
    // z a un offset non nul dans les trois composantes. Appels directs pour
    // couvrir aussi les noyaux NAX, sans grossir la fixture jusqu'au 27B.
    for kernel in 0..3 {
        if (kernel == 1 && !executor.qmm_na_fused_tiled_u4_available(64))
            || (kernel == 2 && !executor.qmm_na_fused_tiled_u4_align64_available(64))
        {
            eprintln!("shared_la kernel {kernel}: NAX indisponible");
            continue;
        }
        let mut results = Vec::new();
        for layer in [&old, &shared] {
            let resolved = executor.resolve_linear_weight_buffers(
                layer.resident_weights().in_proj_z.weight(),
                "shared_la_kernel_z",
            )?;
            let MetalLinearWeightBuffers::AffineQuantized {
                packed,
                packed_offset,
                scales,
                scales_offset,
                biases,
                biases_offset,
                ..
            } = resolved
            else {
                panic!("fixture u4 attendue");
            };
            let offsets = [packed_offset, scales_offset, biases_offset];
            let output = executor.uncached_f32_buffer(64 * 2048, "shared_la_kernel_out")?;
            rows_exact_cb(&executor, |enc| match kernel {
                0 => executor.encode_affine_qmv_align64_buffers(
                    enc, &input, &packed, &scales, &biases, &output, 64, 512, 2048, 64, 8, 4,
                    offsets,
                ),
                1 => executor.encode_affine_qmm_na_fused_tiled_u4_buffers(
                    enc, &input, &packed, &scales, &biases, &output, 64, 512, 2048, offsets,
                ),
                _ => executor.encode_affine_qmm_na_fused_tiled_u4_align64_buffers(
                    enc, &input, &packed, &scales, &biases, &output, 64, 512, 2048, offsets,
                ),
            })?;
            results.push(shared_la_bits(&read_f32_buffer(&output, 64 * 2048)?));
        }
        assert_eq!(results[0], results[1], "kernel {kernel}");
    }
    Ok(())
}
