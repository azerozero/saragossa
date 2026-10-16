// Microbench des matmuls u4 gs64 à petite largeur M (vérification spéculative
// de 2 à 16 drafts) : kernel choisi, temps du dispatch isolé, et coût d'un
// rembourrage à 16 lignes vers le GEMM NA. Formes du Qwen3.8-27B 4 bits.
//
//   cargo test --release --features metal small_m_qmm_microbench -- --ignored --nocapture
#[test]
#[ignore = "microbench qmm u4 gs64 à petite largeur M (GPU idle)"]
fn small_m_qmm_microbench() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let shapes = [
        ("gate/up 5120->17408", 17408_usize, 5120_usize),
        ("down 17408->5120", 5120, 17408),
        ("qkv 5120->14336", 14336, 5120),
        ("o_proj 6144->5120", 5120, 6144),
        ("lm_head 5120->248320", 248_320, 5120),
    ];
    let widths = [1_usize, 2, 3, 4, 6, 8, 12, 16];
    let time_dispatch = |encode: &dyn Fn(&ComputeCommandEncoderRef) -> Result<()>| -> Result<f64> {
        let mut ms = Vec::new();
        for _ in 0..7 {
            let cb = executor.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            let g = EncoderEndGuard::new(enc);
            encode(enc)?;
            g.end();
            let t = std::time::Instant::now();
            commit_and_wait(cb)?;
            ms.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        ms.sort_by(|a, b| a.partial_cmp(b).expect("invariant: temps fini"));
        Ok(ms[3])
    };
    for (label, out_dim, in_dim) in shapes {
        let affine = test_affine_varied(out_dim, in_dim)?;
        let weight = executor.resolve_linear_weight_buffers(
            &LinearWeight::AffineQuantized(affine.clone()),
            "small_m_weight",
        )?;
        let bytes = (out_dim * in_dim / 2) as f64;
        let lhs16: Vec<f32> = (0..16 * in_dim)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.01)
            .collect();
        let lhs16_buf = executor.upload_f32_buffer(&lhs16, "small_m_lhs16")?;
        let out16 = executor.uncached_f32_buffer(16 * out_dim, "small_m_out16")?;
        let m16 = time_dispatch(&|enc| {
            executor
                .encode_matmul_weight_buffers(enc, &lhs16_buf, 16, in_dim, &weight, &out16, false)
                .map(|_| ())
        })?;
        eprintln!("{label} ({:.0} MB u4)", bytes / 1e6);
        for m in widths {
            let kernel = executor.select_owned_affine_matmul_kernel(m, in_dim, &affine, false);
            let lhs_buf = executor.upload_f32_buffer(&lhs16[..m * in_dim], "small_m_lhs")?;
            let out = executor.uncached_f32_buffer(m * out_dim, "small_m_out")?;
            let ms = time_dispatch(&|enc| {
                executor
                    .encode_matmul_weight_buffers(enc, &lhs_buf, m, in_dim, &weight, &out, false)
                    .map(|_| ())
            })?;
            eprintln!(
                "  M={m:>2} {kernel:<22?} {ms:>7.3} ms  {:>6.0} GB/s  | rembourré à 16 : {m16:.3} ms ({:.2}x)",
                bytes / (ms / 1000.0) / 1e9,
                ms / m16,
            );
        }
    }
    Ok(())
}
