fn gdn_storage_linear(bits: usize) -> Result<Linear> {
    let cols = 512 / (32 / bits);
    Linear::new_quantized(
        AffineQuantizedTensor::new(
            &[16, cols],
            (0..16 * cols).map(|i| 0x7654_3210 ^ i as u32).collect(),
            Tensor::from_vec(vec![16, 8], vec![0.25; 128])?,
            Tensor::from_vec(vec![16, 8], vec![-0.5; 128])?,
            64,
            bits,
        )?,
        None,
    )
}

fn gdn_storage_weights<'a>(
    qkv: &'a Linear,
    z: &'a Linear,
    conv: &'a Tensor,
) -> LinearAttnResidentWeights<'a> {
    LinearAttnResidentWeights {
        in_proj_qkv: qkv,
        in_proj_z: z,
        in_proj_b: z,
        in_proj_a: z,
        out_proj: z,
        conv_weight: conv,
        a_log: &[0.0],
        dt_bias: &[0.0],
        norm_weight: &[1.0],
    }
}

#[test]
#[ignore = "device Metal requis ; exécuter aussi avec SARAGOSSA_RUST_LINEAR_FULL_CONCAT=0"]
fn gdn_storage_full_or_z_beta_gate() -> Result<()> {
    let metal = MetalExecutor::new()?;
    metal.prepare_cpu_weight_release();
    let mut qkv = gdn_storage_linear(4)?;
    let mut z = gdn_storage_linear(4)?;
    qkv.move_packed_to_metal_shared(&metal)?;
    z.move_packed_to_metal_shared(&metal)?;
    let conv = Tensor::from_vec(vec![1], vec![1.0])?;
    let weights = gdn_storage_weights(&qkv, &z, &conv);
    let resolved = metal.resolve_linear_attn_resident_dense_weights(weights)?;
    assert!(resolved.full.is_some());
    assert_eq!(
        resolved.z_beta_gate.is_some(),
        !linear_full_concat_enabled()
    );
    assert!(matches!(
        resolved.qkv_z,
        MetalLinearAttnResidentPairWeights::Split { .. }
    ));
    let cache = metal.concat_buffers.lock().expect("invariant: cache sain");
    assert_eq!(
        cache.len(),
        if linear_full_concat_enabled() { 1 } else { 2 }
    );
    drop(cache);

    // QKV u8 et z/b/a u4 : full échoue, mais le concat de repli reste valide.
    let mut qkv_u8 = gdn_storage_linear(8)?;
    qkv_u8.move_packed_to_metal_shared(&metal)?;
    let mixed = metal
        .resolve_linear_attn_resident_dense_weights(gdn_storage_weights(&qkv_u8, &z, &conv))?;
    assert!(mixed.full.is_none());
    assert!(mixed.z_beta_gate.is_some());
    Ok(())
}

fn gdn_storage_attention(bits: usize) -> Result<crate::linear_attention::LinearAttention> {
    use crate::linear_attention::{LinearAttention, LinearAttentionWeights};
    Ok(LinearAttention::new(LinearAttentionWeights {
        in_proj_qkv: gdn_storage_linear(bits)?,
        in_proj_z: Linear::new_quantized(
            if bits == 4 {
                test_affine_varied(16, 512)?
            } else {
                test_affine_varied_u8(16, 512)?
            },
            None,
        )?,
        in_proj_b: gdn_storage_linear(bits)?,
        in_proj_a: gdn_storage_linear(bits)?,
        out_proj: gdn_storage_linear(bits)?,
        conv1d_weight: Tensor::from_vec(vec![1], vec![1.0])?,
        a_log: Tensor::from_vec(vec![1], vec![0.0])?,
        dt_bias: Tensor::from_vec(vec![1], vec![0.0])?,
        norm_weight: Tensor::from_vec(vec![1], vec![1.0])?,
    }))
}

fn gdn_storage_views(weight: &AffineQuantizedTensor) -> [Buffer; 3] {
    [
        weight.packed_metal_view(),
        weight.scales_metal_view(),
        weight.biases_metal_view(),
    ]
    .map(|view| {
        let (buffer, _) = view.expect("invariant: poids partagé");
        buffer.clone()
    })
}

#[expect(
    unsafe_code,
    reason = "vérifie que seul le test retient l'ancienne allocation"
)]
fn gdn_storage_retain_count(buffer: &Buffer) -> Result<NSUInteger> {
    use metal::objc::Message;
    // SAFETY: le buffer retient un objet Metal vivant ; retainCount ne prend
    // aucun argument et retourne un NSUInteger. Aucun envoi GPU n'est en vol.
    unsafe { (*buffer.as_ptr().cast::<Object>()).send_message(Sel::register("retainCount"), ()) }
        .map_err(|error| InferError::Metal(format!("retainCount : {error:?}")))
}

#[test]
#[ignore = "device Metal requis ; exécuter aussi avec SARAGOSSA_RUST_LINEAR_FULL_CONCAT=0"]
fn gdn_storage_qkv_z_alias_bytes_and_owners() -> Result<()> {
    for (bits, is_z) in [(4, false), (4, true), (8, false), (8, true)] {
        let metal = MetalExecutor::new()?;
        metal.prepare_cpu_weight_release();
        let mut attention = gdn_storage_attention(bits)?;
        // Remplit les trois caches historiques avant le passage single-copy.
        let _ = metal.resolve_linear_weight_buffers(
            if is_z {
                attention.resident_weights().in_proj_z.weight()
            } else {
                attention.resident_weights().in_proj_qkv.weight()
            },
            "test_qkv_cache",
        )?;
        attention.move_packed_to_metal_shared(&metal)?;
        let weights = attention.resident_weights();
        let small_before = [weights.in_proj_b, weights.in_proj_a].map(|linear| {
            gdn_storage_views(
                linear
                    .weight()
                    .affine_quantized()
                    .expect("invariant: affine"),
            )
        });
        let qkv = (if is_z {
            weights.in_proj_z
        } else {
            weights.in_proj_qkv
        })
        .weight()
        .affine_quantized()
        .expect("invariant: QKV affine");
        let id = qkv.weight_id();
        let kernel = metal.select_owned_affine_matmul_kernel(1, 512, qkv, false);
        assert!(matches!(
            kernel,
            AffineMatmulKernel::FastQmvU4
                | AffineMatmulKernel::FastQmvU8
                | AffineMatmulKernel::FastQmvOneU8
        ));
        let before = gdn_storage_views(qkv);
        let bytes = before
            .iter()
            .map(|buffer| {
                super::byte_helpers::read_u16_buffer(buffer, buffer.length() as usize / 2)
            })
            .collect::<Result<Vec<_>>>()?;
        let old_cache = metal.weight_buffers.lock().expect("invariant: cache sain")
            .iter().filter(|(key, _)| {
                matches!(key.source, MetalBufferSource::Affine { weight_id, .. } if weight_id == id)
            }).map(|(_, buffer)| buffer.clone()).collect::<Vec<_>>();
        assert_eq!(old_cache.len(), 3);

        attention.materialize_metal_weight_buffers(&metal)?;
        let weights = attention.resident_weights();
        for (linear, before) in [weights.in_proj_b, weights.in_proj_a]
            .into_iter()
            .zip(small_before)
        {
            let after = gdn_storage_views(
                linear
                    .weight()
                    .affine_quantized()
                    .expect("invariant: affine"),
            );
            for (before, after) in before.iter().zip(after) {
                assert_eq!(before.as_ptr(), after.as_ptr());
            }
        }
        let qkv = (if is_z {
            weights.in_proj_z
        } else {
            weights.in_proj_qkv
        })
        .weight()
        .affine_quantized()
        .expect("invariant: QKV affine");
        assert_eq!(qkv.weight_id(), id);
        assert_eq!(qkv.shape(), &[16, 512]);
        assert_eq!(
            std::mem::discriminant(&kernel),
            std::mem::discriminant(&metal.select_owned_affine_matmul_kernel(1, 512, qkv, false),)
        );
        let after = gdn_storage_views(qkv);
        let offsets = if is_z && linear_full_concat_enabled() {
            [16 * 512 * bits / 8, 16 * 8 * 2, 16 * 8 * 2]
        } else {
            [0; 3]
        };
        assert_eq!(
            [
                affine_packed_offset(qkv)?,
                affine_scales_offset(qkv)?,
                affine_biases_offset(qkv)?
            ],
            offsets.map(|v| v as u64)
        );
        for (((before, after), bytes), offset) in before.iter().zip(&after).zip(&bytes).zip(offsets)
        {
            assert_eq!(
                super::byte_helpers::read_u16_buffer_at(after, offset / 2, bytes.len())?,
                *bytes
            );
            assert_eq!(
                before.as_ptr() != after.as_ptr(),
                linear_full_concat_enabled()
            );
        }
        let resolved = metal.resolve_linear_attn_resident_dense_weights(weights)?;
        let MetalLinearWeightBuffers::AffineQuantized {
            packed,
            scales,
            biases,
            ..
        } = &resolved
            .full
            .as_ref()
            .expect("invariant: concat retenu")
            .in_proj
        else {
            return Err(InferError::Config("concat non affine".to_string()));
        };
        if linear_full_concat_enabled() {
            for (unit, concat) in after.iter().zip([packed, scales, biases]) {
                assert_eq!(unit.as_ptr(), concat.as_ptr());
            }
            let MetalLinearAttnResidentPairWeights::Split { first, second } = &resolved.qkv_z
            else {
                return Err(InferError::Config("QKV/z n'est plus split".to_string()));
            };
            let MetalLinearWeightBuffers::AffineQuantized {
                packed,
                packed_offset,
                scales_offset,
                biases_offset,
                out_dim,
                ..
            } = (if is_z { second } else { first })
            else {
                return Err(InferError::Config("QKV/z non affine".to_string()));
            };
            assert_eq!(packed.as_ptr(), after[0].as_ptr());
            assert_eq!(
                (*packed_offset, *scales_offset, *biases_offset, *out_dim),
                (offsets[0] as u64, offsets[1] as u64, offsets[2] as u64, 16)
            );
            let cache = metal.weight_buffers.lock().expect("invariant: cache sain");
            for (key, buffer) in cache.iter() {
                if let MetalBufferSource::Affine { weight_id, part } = key.source {
                    if weight_id == id {
                        let index = match part {
                            AffineWeightPart::Packed => 0,
                            AffineWeightPart::Scales => 1,
                            AffineWeightPart::Biases => 2,
                        };
                        assert_eq!(buffer.as_ptr(), after[index].as_ptr());
                    }
                }
            }
            // Les seuls propriétaires restants des six anciennes allocations
            // sont nos témoins : les lâcher libère réellement leur stockage.
            for old in before.iter().chain(&old_cache) {
                assert_eq!(gdn_storage_retain_count(old)?, 1);
            }
        }
        drop(resolved);
        attention.materialize_metal_weight_buffers(&metal)?;
        let again = gdn_storage_views(
            (if is_z {
                attention.resident_weights().in_proj_z
            } else {
                attention.resident_weights().in_proj_qkv
            })
            .weight()
            .affine_quantized()
            .expect("invariant: QKV affine"),
        );
        for (after, again) in after.iter().zip(again) {
            assert_eq!(after.as_ptr(), again.as_ptr());
        }
    }
    Ok(())
}

#[test]
#[ignore = "device Metal requis"]
fn gdn_storage_no_alias_without_retained_full() -> Result<()> {
    let metal = MetalExecutor::new()?;
    let mut attention = gdn_storage_attention(4)?;
    // Le stockage CPU ne doit pas devenir bf16 à l'occasion de l'alias.
    metal.resolve_linear_attn_resident_dense_weights(attention.resident_weights())?;
    assert!(metal
        .retained_linear_attn_qkv_z_aliases(attention.resident_weights())?
        .is_none());
    metal
        .concat_buffers
        .lock()
        .expect("invariant: cache sain")
        .clear();
    attention.move_packed_to_metal_shared(&metal)?;
    super::matmul::with_concat_vram_fraction_override(0.0, || -> Result<()> {
        // Même un concat valide construit transitoirement n'autorise pas l'alias.
        let _ = metal.resolve_linear_attn_resident_weights(attention.resident_weights())?;
        assert!(metal
            .concat_buffers
            .lock()
            .expect("invariant: cache sain")
            .is_empty());
        assert!(metal
            .retained_linear_attn_qkv_z_aliases(attention.resident_weights())?
            .is_none());
        attention.materialize_metal_weight_buffers(&metal)?;
        Ok(())
    })?;
    let mut qkv_u8 = gdn_storage_linear(8)?;
    qkv_u8.move_packed_to_metal_shared(&metal)?;
    let mixed = LinearAttnResidentWeights {
        in_proj_qkv: &qkv_u8,
        ..attention.resident_weights()
    };
    metal.prepare_cpu_weight_release();
    assert!(metal
        .resolve_linear_attn_resident_dense_weights(mixed)?
        .full
        .is_none());
    assert!(metal.retained_linear_attn_qkv_z_aliases(mixed)?.is_none());
    Ok(())
}

#[test]
#[ignore = "device Metal requis ; validation transactionnelle des vues"]
fn gdn_storage_alias_failure_keeps_caches_and_owners() -> Result<()> {
    let metal = MetalExecutor::new()?;
    metal.prepare_cpu_weight_release();
    let mut attention = gdn_storage_attention(4)?;
    let weights = attention.resident_weights();
    for linear in [weights.in_proj_qkv, weights.in_proj_z] {
        metal.resolve_linear_weight_buffers(linear.weight(), "alias_failure_cache")?;
    }
    attention.move_packed_to_metal_shared(&metal)?;
    let weights = attention.resident_weights();
    let resolved = metal.resolve_linear_attn_resident_dense_weights(weights)?;
    drop(resolved);
    if !linear_full_concat_enabled() {
        return Ok(());
    }
    let sources = [weights.in_proj_qkv, weights.in_proj_z];
    let before = sources.map(|linear| {
        gdn_storage_views(
            linear
                .weight()
                .affine_quantized()
                .expect("invariant: affine"),
        )
    });
    let cached_before = metal
        .weight_buffers
        .lock()
        .expect("invariant: cache sain")
        .clone();
    // QKV tient encore, mais la dernière vue de z échoue : aucune publication
    // partielle des cinq premières vues ne doit atteindre le cache unitaire.
    for entry in metal
        .concat_buffers
        .lock()
        .expect("invariant: cache sain")
        .values_mut()
    {
        if let ConcatCacheEntry::Buffers(MetalLinearWeightBuffers::AffineQuantized {
            biases, ..
        }) = entry
        {
            *biases = metal.buffer_from_slice(&vec![0_u16; 16 * 8], "truncated_alias_biases")?;
        }
    }
    assert!(metal.retained_linear_attn_qkv_z_aliases(weights).is_err());
    for (source, before) in sources.into_iter().zip(before) {
        let after = gdn_storage_views(
            source
                .weight()
                .affine_quantized()
                .expect("invariant: affine"),
        );
        for (before, after) in before.iter().zip(after) {
            assert_eq!(before.as_ptr(), after.as_ptr());
        }
    }
    let cached_after = metal.weight_buffers.lock().expect("invariant: cache sain");
    assert_eq!(cached_before.len(), cached_after.len());
    for (key, before) in cached_before {
        assert_eq!(before.as_ptr(), cached_after[&key].as_ptr());
    }
    Ok(())
}
