fn gdn_storage_linear(bits: usize) -> Result<Linear> {
    let cols = 64 / (32 / bits);
    Linear::new_quantized(
        AffineQuantizedTensor::new(
            &[2, cols],
            (0..2 * cols).map(|i| 0x7654_3210 ^ i as u32).collect(),
            Tensor::from_vec(vec![2, 1], vec![0.5, 0.25])?,
            Tensor::from_vec(vec![2, 1], vec![-0.5, 0.125])?,
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
    assert_eq!(resolved.z_beta_gate.is_some(), !linear_full_concat_enabled());
    assert!(matches!(resolved.qkv_z, MetalLinearAttnResidentPairWeights::Split { .. }));
    let cache = metal.concat_buffers.lock().expect("invariant: cache sain");
    assert_eq!(cache.len(), if linear_full_concat_enabled() { 1 } else { 2 });
    drop(cache);

    // QKV u8 et z/b/a u4 : full échoue, mais le concat de repli reste valide.
    let mut qkv_u8 = gdn_storage_linear(8)?;
    qkv_u8.move_packed_to_metal_shared(&metal)?;
    let mixed = metal.resolve_linear_attn_resident_dense_weights(gdn_storage_weights(
        &qkv_u8, &z, &conv,
    ))?;
    assert!(mixed.full.is_none());
    assert!(mixed.z_beta_gate.is_some());
    Ok(())
}
