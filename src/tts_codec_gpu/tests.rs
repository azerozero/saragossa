//! Convolutions en tuiles contre le kernel gather, sur poids synthétiques.

use super::*;
use crate::metal_backend::{commit_and_wait, read_f32_buffer};

/// Valeurs déterministes dans [-1, 1] (LCG), sans zéro exact.
fn values(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0 + 1.0e-3
        })
        .collect()
}

fn weights(
    name: &str,
    shape: [usize; 3],
    bias: usize,
    seed: u64,
) -> Result<HashMap<String, Tensor>> {
    let n = shape.iter().product();
    let mut map = HashMap::new();
    map.insert(
        format!("{name}.weight"),
        Tensor::from_vec(shape.to_vec(), values(n, seed))?,
    );
    map.insert(
        format!("{name}.bias"),
        Tensor::from_vec(vec![bias], values(bias, seed + 1))?,
    );
    Ok(map)
}

fn input(gpu: &CodecGpu, time: usize, channels: usize, seed: u64) -> Result<GpuNlc> {
    let data = values(time * channels, seed);
    Ok(GpuNlc {
        buffer: gpu.upload(&data)?,
        pool_index: None,
        time,
        channels,
    })
}

/// Exécute `op` en gather puis en tuiles et renvoie les deux sorties.
fn both(
    gpu: &mut CodecGpu,
    op: impl Fn(&CodecGpu, &ComputeCommandEncoderRef, &mut BufPool) -> Result<GpuNlc>,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let mut run = |tiled: bool| -> Result<Vec<f32>> {
        gpu.set_conv_tiled(tiled);
        let mut pool = BufPool::new(gpu.device.clone(), MTLResourceOptions::StorageModeShared);
        let command_buffer = gpu.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let _barrier_scope = install_dispatch_barrier_scope();
        let out = op(gpu, encoder, &mut pool)?;
        encoder.end_encoding();
        commit_and_wait(command_buffer)?;
        read_f32_buffer(&out.buffer, out.len())
    };
    Ok((run(false)?, run(true)?))
}

fn assert_same(case: &str, (gather, tiled): (Vec<f32>, Vec<f32>)) {
    assert_eq!(gather.len(), tiled.len(), "{case}: longueurs");
    assert!(gather.iter().any(|v| *v != 0.0), "{case}: sortie nulle");
    let differ = gather
        .iter()
        .zip(&tiled)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(differ, 0, "{case}: {differ} sorties diffèrent au bit");
}

#[test]
fn tiled_convolutions_match_gather_bitwise() -> Result<()> {
    if Device::system_default().is_none() {
        eprintln!("pas de device Metal : test ignoré");
        return Ok(());
    }
    let mut gpu = CodecGpu::without_weights(Vec::new())?;
    let specs: [(&str, [usize; 3], usize); 5] = [
        ("d9", [64, 32, 7], 64),
        ("big", [64, 16, 3], 64),
        ("d3", [32, 32, 7], 32),
        ("k1", [32, 32, 1], 32),
        ("t3", [32, 64, 6], 64),
    ];
    for (seed, (name, shape, bias)) in (10_u64..).step_by(10).zip(specs) {
        let map = weights(name, shape, bias, seed)?;
        if name.starts_with('t') {
            gpu.register_transpose(&map, name)?;
        } else {
            gpu.register_conv(&map, name)?;
        }
    }

    // Causale dilatation 9 sur 40 lignes : pad 54, `src` négatif, tuiles TM = 1.
    assert!(gpu.tiled_small(&gpu.convs["d9"], 40, 1));
    let x = input(&gpu, 40, 32, 1)?;
    assert_same(
        "conv d9 TM=1",
        both(&mut gpu, |g, e, p| g.conv(e, p, &x, "d9", 9, false, None))?,
    );

    // 2600 lignes : 82 tuiles de 64 lignes, TM = 4.
    assert!(!gpu.tiled_small(&gpu.convs["big"], 2600, 1));
    let x = input(&gpu, 2600, 16, 2)?;
    assert_same(
        "conv k3 TM=4",
        both(&mut gpu, |g, e, p| g.conv(e, p, &x, "big", 1, false, None))?,
    );

    // Streaming : 12 lignes de contexte, sorties à partir de out_start.
    let x = input(&gpu, 30, 32, 3)?;
    assert_same(
        "conv d3 out_start",
        both(&mut gpu, |g, e, p| {
            g.conv_with_start(e, p, &x, "d3", 3, false, 12, None)
        })?,
    );

    // conv k1 valide + raccord résiduel fusionné (gather : conv puis add).
    let x = input(&gpu, 50, 32, 4)?;
    let residual = input(&gpu, 50, 32, 5)?;
    assert_same(
        "conv k1 résiduel",
        both(&mut gpu, |g, e, p| {
            g.conv(e, p, &x, "k1", 1, true, Some(&residual))
        })?,
    );

    // Transposée k6 s3 : trois phases, contexte de 2 lignes, TM = 1 puis TM = 4.
    let x = input(&gpu, 20, 32, 6)?;
    assert!(gpu.tiled_small(&gpu.transposes["t3"], 18, 3));
    assert_same(
        "transposée phases TM=1",
        both(&mut gpu, |g, e, p| {
            g.transpose_with_start(e, p, &x, "t3", 3, 6, 18 * 3)
        })?,
    );
    let x = input(&gpu, 900, 32, 7)?;
    assert!(!gpu.tiled_small(&gpu.transposes["t3"], 900, 3));
    assert_same(
        "transposée phases TM=4",
        both(&mut gpu, |g, e, p| g.transpose(e, p, &x, "t3", 3))?,
    );
    Ok(())
}

/// Les constantes `TILE_*` décrivent la source Metal : `CT_BN`, `CT_BK`, les
/// deux instanciations TM et les 128 threads de la disposition `tid`.
#[test]
fn tile_constants_match_metal_source() {
    assert!(CODEC_KERNELS.contains(&format!("#define CT_BN {TILE_N}u")));
    assert!(CODEC_KERNELS.contains(&format!("#define CT_BK {TILE_K}u")));
    assert!(CODEC_KERNELS.contains(&format!(
        "CODEC_CONV_TILED_KERNEL(codec_conv_tiled_f32, {}u)",
        TILE_M / 16
    )));
    assert!(CODEC_KERNELS.contains("CODEC_CONV_TILED_KERNEL(codec_conv_tiled_small_f32, 1u)"));
    assert_eq!(TILE_THREADS, 128);
    assert!(CODEC_KERNELS.contains("tid + 128u * e"));
}
