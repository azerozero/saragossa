/// Kernels du semis MTP GPU sur données synthétiques : deux normes à pas de
/// sortie remplissent les moitiés d'une concat `[rows, 2·dim]` comme
/// `concat_rows(rms_norm(a), rms_norm(b))`, puis `copy_cols` en extrait la
/// seconde moitié (découpe K|V).
#[test]
fn mtp_seed_strided_norm_and_copy_cols_match_cpu() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let (rows, dim, eps) = (5_usize, 320_usize, 1e-6_f32);
    let values = |seed: usize, len: usize| -> Vec<f32> {
        (0..len)
            .map(|i| (((i * 7 + seed * 13) % 23) as f32 - 11.0) / 7.0)
            .collect()
    };
    let left = Tensor::from_vec(vec![rows, dim], values(1, rows * dim))?;
    let right = Tensor::from_vec(vec![rows, dim], values(2, rows * dim))?;
    let left_w = Tensor::from_vec(vec![dim], values(3, dim))?;
    let right_w = Tensor::from_vec(vec![dim], values(4, dim))?;
    let left_cpu = crate::norm::rms_norm(&left, &left_w, eps)?;
    let right_cpu = crate::norm::rms_norm(&right, &right_w, eps)?;

    let concat = executor.new_f32_buffer(rows * 2 * dim, "test_concat")?;
    let tail = executor.new_f32_buffer(rows * dim, "test_tail")?;
    let command_buffer = executor.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    let guard = EncoderEndGuard::new(encoder);
    let mut inputs = Vec::new();
    // Labels distincts : `upload_f32_buffer` recycle le scratch par label.
    for (input, weight, labels) in [
        (&left, &left_w, ["test_left", "test_left_w"]),
        (&right, &right_w, ["test_right", "test_right_w"]),
    ] {
        inputs.push((
            executor.upload_f32_buffer(input.data(), labels[0])?,
            executor.upload_f32_buffer(weight.data(), labels[1])?,
        ));
    }
    for (half, (input, weight)) in inputs.iter().enumerate() {
        executor.encode_rms_norm_rows_strided(
            encoder,
            input,
            0,
            weight,
            &concat,
            (half * dim * std::mem::size_of::<f32>()) as u64,
            rows,
            dim,
            2 * dim,
            eps,
        )?;
    }
    executor.encode_copy_cols(
        encoder,
        &concat,
        (dim * std::mem::size_of::<f32>()) as u64,
        &tail,
        dim,
        2 * dim,
        rows * dim,
    )?;
    guard.end();
    commit_and_wait(command_buffer)?;

    let gpu = read_f32_buffer(&concat, rows * 2 * dim)?;
    let max_err = |gpu: &[f32], cpu: &[f32]| {
        gpu.iter()
            .zip(cpu)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max)
    };
    for row in 0..rows {
        let span = row * dim..(row + 1) * dim;
        let base = row * 2 * dim;
        assert!(max_err(&gpu[base..base + dim], &left_cpu.data()[span.clone()]) < 1e-5);
        assert!(max_err(&gpu[base + dim..base + 2 * dim], &right_cpu.data()[span]) < 1e-5);
    }
    let tail = read_f32_buffer(&tail, rows * dim)?;
    assert!(max_err(&tail, right_cpu.data()) < 1e-5);
    Ok(())
}
