fn fast_qmv_align64_reference(
    executor: &MetalExecutor,
    weight: &AffineQuantizedTensor,
    lhs: &[f32],
    batch: usize,
    label: &'static str,
) -> Result<Vec<f32>> {
    let [out_dim, in_dim] = weight.shape() else {
        return Err(InferError::Dimension(format!(
            "{label}: poids attendu rang 2, reçu {:?}",
            weight.shape()
        )));
    };
    let [_, packed_cols] = weight.packed_shape() else {
        return Err(InferError::Dimension(format!(
            "{label}: packed_shape attendu rang 2, reçu {:?}",
            weight.packed_shape()
        )));
    };
    if lhs.len() != batch * *in_dim {
        return Err(InferError::Dimension(format!(
            "{label}: lhs len={} incompatible batch={batch} in_dim={in_dim}",
            lhs.len()
        )));
    }

    let lhs_buf = executor.upload_f32_buffer(lhs, label)?;
    let packed = executor.buffer_from_slice(weight.packed_data(), label)?;
    let scales = executor.buffer_from_f32_as_bf16(weight.scales().data(), label)?;
    let biases = executor.buffer_from_f32_as_bf16(weight.biases().data(), label)?;
    let out_buf = executor.uncached_f32_buffer(batch * *out_dim, label)?;
    let dims = [
        *out_dim as u32,
        *in_dim as u32,
        *packed_cols as u32,
        (*in_dim / weight.group_size()) as u32,
    ];
    let pipeline = match weight.bits() {
        FAST_QMV_BITS => &executor.affine_qmv_fast_u4_gs64_align64_f32,
        8 => &executor.affine_qmv_fast_u8_gs64_align64_f32,
        bits => {
            return Err(InferError::Dimension(format!(
                "{label}: qmv align64 ne supporte pas bits={bits}"
            )));
        }
    };

    let command_buffer = executor.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(&lhs_buf), 0);
    encoder.set_buffer(1, Some(&packed), 0);
    encoder.set_buffer(2, Some(&scales), 0);
    encoder.set_buffer(3, Some(&biases), 0);
    encoder.set_buffer(4, Some(&out_buf), 0);
    encoder.set_bytes(5, 16, dims.as_ptr().cast());
    encoder.dispatch_thread_groups(
        MTLSize::new(batch as u64, (*out_dim as u64).div_ceil(8), 1),
        MTLSize::new(64, 1, 1),
    );
    encoder.end_encoding();
    commit_and_wait(command_buffer)?;

    read_f32_buffer(&out_buf, batch * *out_dim)
}

fn cpu_qmv_align64_reference(
    weight: &AffineQuantizedTensor,
    lhs: &[f32],
    batch: usize,
) -> Result<Vec<f32>> {
    let [out_dim, in_dim] = weight.shape() else {
        return Err(InferError::Dimension(format!(
            "qmv align64 CPU: poids attendu rang 2, reçu {:?}",
            weight.shape()
        )));
    };
    let [_, packed_cols] = weight.packed_shape() else {
        return Err(InferError::Dimension(format!(
            "qmv align64 CPU: packed_shape attendu rang 2, reçu {:?}",
            weight.packed_shape()
        )));
    };
    if !matches!(weight.bits(), FAST_QMV_BITS | 8) {
        return Err(InferError::Dimension(format!(
            "qmv align64 CPU attend bits=4 ou bits=8, reçu {}",
            weight.bits()
        )));
    }
    if lhs.len() != batch * *in_dim {
        return Err(InferError::Dimension(format!(
            "qmv align64 CPU: lhs len={} incompatible batch={batch} in_dim={in_dim}",
            lhs.len()
        )));
    }

    let bits = weight.bits();
    let values_per_word = 32 / bits;
    let mask = (1_u32 << bits) - 1;
    let group_size = weight.group_size();
    let groups = *in_dim / group_size;
    let packed = weight.packed_data();
    let scales = weight.scales().data();
    let biases = weight.biases().data();
    let mut out = vec![0.0_f32; batch * *out_dim];
    for bb in 0..batch {
        for row in 0..*out_dim {
            let mut acc = 0.0_f32;
            for group in 0..groups {
                let affine = row * groups + group;
                let scale = bf16_round(scales[affine]);
                let bias = bf16_round(biases[affine]);
                for col in group * group_size..(group + 1) * group_size {
                    let word = packed[row * *packed_cols + col / values_per_word];
                    let q = (word >> ((col % values_per_word) * bits)) & mask;
                    acc += lhs[bb * *in_dim + col] * (q as f32 * scale + bias);
                }
            }
            out[bb * *out_dim + row] = acc;
        }
    }
    Ok(out)
}

fn routed_affine_outputs(
    executor: &MetalExecutor,
    linear: &LinearWeight,
    lhs: &[f32],
    batch: usize,
    label: &'static str,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let [out_dim, in_dim] = linear.shape() else {
        return Err(InferError::Dimension(format!(
            "{label}: poids attendu rang 2, reçu {:?}",
            linear.shape()
        )));
    };
    let buffers = executor.resolve_linear_weight_buffers(linear, label)?;
    let lhs_buf = executor.upload_f32_buffer(lhs, label)?;
    let owned_out = executor.uncached_f32_buffer(batch * *out_dim, label)?;
    let resident_out = executor.uncached_f32_buffer(batch * *out_dim, label)?;
    let mut owned_buffers = Vec::new();

    let command_buffer = executor.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    let owned_dim = executor.encode_matmul_weight(
        encoder,
        &mut owned_buffers,
        &lhs_buf,
        batch,
        *in_dim,
        linear,
        &owned_out,
    )?;
    let resident_dim = executor.encode_matmul_weight_buffers(
        encoder,
        &lhs_buf,
        batch,
        *in_dim,
        &buffers,
        &resident_out,
        false,
    )?;
    assert_eq!(owned_dim, *out_dim);
    assert_eq!(resident_dim, *out_dim);
    encoder.end_encoding();
    commit_and_wait(command_buffer)?;

    Ok((
        read_f32_buffer(&owned_out, batch * *out_dim)?,
        read_f32_buffer(&resident_out, batch * *out_dim)?,
    ))
}

fn qmm_na_tiled_u4_align64_reference(
    executor: &MetalExecutor,
    weight: &AffineQuantizedTensor,
    lhs: &[f32],
    batch: usize,
    label: &'static str,
) -> Result<Vec<f32>> {
    let [out_dim, in_dim] = weight.shape() else {
        return Err(InferError::Dimension(format!(
            "{label}: poids attendu rang 2, reçu {:?}",
            weight.shape()
        )));
    };
    if lhs.len() != batch * *in_dim {
        return Err(InferError::Dimension(format!(
            "{label}: lhs len={} incompatible batch={batch} in_dim={in_dim}",
            lhs.len()
        )));
    }

    let lhs_buf = executor.upload_f32_buffer(lhs, label)?;
    let packed = executor.buffer_from_slice(weight.packed_data(), label)?;
    let scales = executor.buffer_from_f32_as_bf16(weight.scales().data(), label)?;
    let biases = executor.buffer_from_f32_as_bf16(weight.biases().data(), label)?;
    let out_buf = executor.uncached_f32_buffer(batch * *out_dim, label)?;

    let command_buffer = executor.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    executor.encode_affine_qmm_na_fused_tiled_u4_align64_buffers(
        encoder,
        &lhs_buf,
        &packed,
        &scales,
        &biases,
        &out_buf,
        batch,
        *in_dim,
        *out_dim,
    )?;
    encoder.end_encoding();
    commit_and_wait(command_buffer)?;

    read_f32_buffer(&out_buf, batch * *out_dim)
}

#[test]
fn affine_qmv_u8_gs64_align64_matches_generic_and_routes() -> Result<()> {
    let Some(executor) = test_executor()? else {
        eprintln!("skip: aucun device Metal pour l'oracle qmv u8 align64");
        return Ok(());
    };

    // Les deux formes ont les mêmes longueurs packed/scales/biases. Les garder
    // vivantes empêche l'allocateur de recycler leurs adresses, qui identifient
    // les poids dans le cache de buffers Metal de l'exécuteur.
    let cases = [(704_usize, 2816_usize), (2816, 704)]
        .into_iter()
        .map(|(out_dim, in_dim)| {
            Ok((
                out_dim,
                in_dim,
                LinearWeight::AffineQuantized(test_affine_varied_u8(out_dim, in_dim)?),
                varied_row(in_dim, 107),
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    for (out_dim, in_dim, linear, lhs) in &cases {
        let (out_dim, in_dim) = (*out_dim, *in_dim);
        let LinearWeight::AffineQuantized(weight) = linear else {
            return Err(InferError::Dimension(
                "oracle qmv u8 align64 construit avec un poids non affine".to_string(),
            ));
        };
        assert!(matches!(
            executor.select_owned_affine_matmul_kernel(1, in_dim, weight, false),
            AffineMatmulKernel::FastQmvU8Align64
        ));
        assert!(matches!(
            executor.select_resident_affine_matmul_kernel(1, in_dim, out_dim, 64, 8, false),
            AffineMatmulKernel::FastQmvU8Align64
        ));

        let cpu = cpu_qmv_align64_reference(weight, lhs, 1)?;
        let fast = fast_qmv_align64_reference(
            &executor,
            weight,
            lhs,
            1,
            "qmv_u8_align64_fast",
        )?;
        let (owned, resident) =
            routed_affine_outputs(&executor, linear, lhs, 1, "qmv_u8_align64_routes")?;
        assert_close_eps(&fast, &cpu, 1.0e-3);
        assert_close_eps(&owned, &cpu, 1.0e-3);
        assert_close_eps(&resident, &cpu, 1.0e-3);
    }
    Ok(())
}

#[test]
fn affine_qmv_u4_gs64_align64_matches_generic_and_routes() -> Result<()> {
    let Some(executor) = test_executor()? else {
        eprintln!("skip: aucun device Metal pour l'oracle qmv u4 align64");
        return Ok(());
    };
    let cases = [2048_usize, 4096, 10_240]
        .into_iter()
        .map(|out_dim| {
            let in_dim = 2816;
            Ok((
                out_dim,
                in_dim,
                LinearWeight::AffineQuantized(test_affine_varied(out_dim, in_dim)?),
                varied_row(in_dim, 109 + out_dim),
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    for (out_dim, in_dim, linear, lhs) in &cases {
        let (out_dim, in_dim) = (*out_dim, *in_dim);
        let LinearWeight::AffineQuantized(weight) = linear else {
            return Err(InferError::Dimension(
                "oracle qmv u4 align64 construit avec un poids non affine".to_string(),
            ));
        };
        assert!(matches!(
            executor.select_owned_affine_matmul_kernel(1, in_dim, weight, false),
            AffineMatmulKernel::FastQmvU4Align64
        ));
        assert!(matches!(
            executor.select_resident_affine_matmul_kernel(
                1,
                in_dim,
                out_dim,
                FAST_QMV_GROUP_SIZE,
                FAST_QMV_BITS,
                false,
            ),
            AffineMatmulKernel::FastQmvU4Align64
        ));

        let cpu = cpu_qmv_align64_reference(weight, lhs, 1)?;
        let fast = fast_qmv_align64_reference(
            &executor,
            weight,
            lhs,
            1,
            "qmv_u4_align64_fast",
        )?;
        let (owned, resident) =
            routed_affine_outputs(&executor, linear, lhs, 1, "qmv_u4_align64_routes")?;

        // Le kernel rapide regroupe les 16 termes d'une lane avant la réduction
        // SIMD. Cette association diffère de la somme CPU ; ε=1e-3 borne l'arrondi
        // f32 tout en restant très inférieur à l'écart d'une lane u4 mal décodée.
        const EPS: f32 = 1.0e-3;
        assert_close_eps(&fast, &cpu, EPS);
        assert_close_eps(&owned, &cpu, EPS);
        assert_close_eps(&resident, &cpu, EPS);
    }

    let qwen_weight = test_affine_varied(8, 512)?;
    assert!(matches!(
        executor.select_owned_affine_matmul_kernel(1, 512, &qwen_weight, false),
        AffineMatmulKernel::FastQmvU4
    ));
    assert!(matches!(
        executor.select_resident_affine_matmul_kernel(
            1,
            512,
            8,
            FAST_QMV_GROUP_SIZE,
            FAST_QMV_BITS,
            false,
        ),
        AffineMatmulKernel::FastQmvU4
    ));
    Ok(())
}

#[test]
fn affine_qmm_na_fused_tiled_u4_align64_matches_generic_and_routes() -> Result<()> {
    const BATCH: usize = 32;
    const IN_DIM: usize = 2816;
    const EPS: f32 = 1.0e-3;
    assert!(can_use_qmm_na_fused_tiled_u4_align64_buffers(
        BATCH,
        IN_DIM,
        2048,
        FAST_QMV_GROUP_SIZE,
        FAST_QMV_BITS,
    ));
    assert!(!can_use_qmm_na_fused_tiled_u4_align64_buffers(
        1,
        IN_DIM,
        2048,
        FAST_QMV_GROUP_SIZE,
        FAST_QMV_BITS,
    ));
    assert!(!can_use_qmm_na_fused_tiled_u4_align64_buffers(
        BATCH,
        2048,
        2048,
        FAST_QMV_GROUP_SIZE,
        FAST_QMV_BITS,
    ));

    let Some(executor) = test_executor()? else {
        eprintln!("skip: aucun device Metal pour l'oracle GEMM u4 align64");
        return Ok(());
    };
    if executor.na_gemm_coop_qb_tiled_u4.is_none() {
        eprintln!("skip: Neural Accelerators indisponibles pour l'oracle GEMM u4 align64");
        return Ok(());
    }
    assert!(
        executor.na_gemm_coop_qb_tiled_u4_align64.is_some(),
        "la pipeline GEMM u4 align64 doit compiler quand le GEMM u4 nominal compile"
    );

    for out_dim in [2048_usize, 4096] {
        let linear = LinearWeight::AffineQuantized(test_affine(
            out_dim,
            IN_DIM,
            1.0 / 256.0,
        )?);
        let LinearWeight::AffineQuantized(weight) = &linear else {
            return Err(InferError::Dimension(
                "oracle GEMM u4 align64 construit avec un poids non affine".to_string(),
            ));
        };
        let lhs = (0..BATCH * IN_DIM)
            .map(|i| (((i * 37 + (i / IN_DIM) * 11) % 31) as f32 - 15.0) / 64.0)
            .collect::<Vec<_>>();
        assert!(matches!(
            executor.select_owned_affine_matmul_kernel(BATCH, IN_DIM, weight, false),
            AffineMatmulKernel::QmmNaFusedTiledU4Align64
        ));
        assert!(matches!(
            executor.select_resident_affine_matmul_kernel(
                BATCH,
                IN_DIM,
                out_dim,
                FAST_QMV_GROUP_SIZE,
                FAST_QMV_BITS,
                false,
            ),
            AffineMatmulKernel::QmmNaFusedTiledU4Align64
        ));

        let generic = generic_affine_reference(
            &executor,
            weight,
            &lhs,
            BATCH,
            "qmm_u4_align64_generic",
        )?;
        let tiled = qmm_na_tiled_u4_align64_reference(
            &executor,
            weight,
            &lhs,
            BATCH,
            "qmm_u4_align64_tiled",
        )?;
        let (owned, resident) = routed_affine_outputs(
            &executor,
            &linear,
            &lhs,
            BATCH,
            "qmm_u4_align64_routes",
        )?;

        // Activations et poids sont exactement représentables en bf16 : ε ne
        // couvre que l'association différente des réductions f32 du GEMM tuilé.
        assert_close_eps(&tiled, &generic, EPS);
        assert_close_eps(&owned, &generic, EPS);
        assert_close_eps(&resident, &generic, EPS);
    }
    Ok(())
}
