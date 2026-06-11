//! Tests de l attention lineaire CPU et Metal.

use super::*;
use crate::LinearWeight;

#[test]
fn linear_attention_cached_matches_full_prefix_for_multiple_layer_counts() {
    for layer_count in [1_usize, 2, 3, 4, 5, 6] {
        let config = test_config();
        let layers = (0..layer_count)
            .map(|_| test_linear_attention())
            .collect::<Vec<_>>();
        let mut caches = vec![LinearAttentionCache::default(); layer_count];
        let input = Tensor::from_vec(vec![3, 2], vec![1.0, 0.0, 0.25, 0.75, -0.5, 0.5])
            .expect("invariant: entrée valide");

        let mut full = input.clone();
        for layer in &layers {
            full = layer
                .forward(config, &full)
                .expect("invariant: forward complet valide");
        }

        let mut cached_rows = Vec::new();
        for token in 0..3 {
            let mut hidden = Tensor::from_vec(
                vec![1, 2],
                input
                    .row_slice(token)
                    .expect("invariant: row valide")
                    .to_vec(),
            )
            .expect("invariant: token valide");
            for (layer, cache) in layers.iter().zip(caches.iter_mut()) {
                hidden = layer
                    .forward_cached(config, &hidden, cache)
                    .expect("invariant: forward cache valide");
            }
            cached_rows.extend_from_slice(hidden.as_row().expect("invariant: row sortie"));
        }
        let cached = Tensor::from_vec(vec![3, 2], cached_rows).expect("invariant: sortie cached");

        assert_close(cached.data(), full.data(), 1.0e-5);
    }
}

#[test]
fn linear_attention_rejects_incompatible_heads() {
    let config = LinearAttentionConfig {
        num_key_heads: 2,
        num_value_heads: 3,
        ..test_config()
    };
    let err = test_linear_attention()
        .forward(
            config,
            &Tensor::row(vec![1.0, 0.0]).expect("invariant: row"),
        )
        .expect_err("invariant: heads incompatibles rejetées");
    assert!(matches!(err, InferError::Dimension(_)));
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn linear_attention_fused_metal_matches_cpu_cached_sequence() -> Result<()> {
    let executor = match crate::MetalExecutor::new() {
        Ok(executor) => executor,
        Err(InferError::Metal(message)) if message.contains("aucun device") => return Ok(()),
        Err(error) => return Err(error),
    };
    let config = test_config();
    let attn = test_linear_attention();
    let mut cpu_cache = LinearAttentionCache::default();
    let mut metal_cache = LinearAttentionCache::default();
    let input = [
        vec![1.0, 0.0],
        vec![0.25, 0.75],
        vec![-0.5, 0.5],
        vec![0.1, -0.3],
    ];

    for row in input {
        let token = Tensor::row(row).expect("invariant: token valide");
        let cpu = attn
            .forward_cached(config, &token, &mut cpu_cache)
            .expect("invariant: CPU cached valide");
        let metal = attn
            .forward_fused_metal_step(config, &token, &mut metal_cache, &executor)
            .expect("invariant: Metal cached valide");
        assert_close(cpu.data(), metal.data(), 2.0e-5);
    }
    Ok(())
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn linear_attention_resident_metal_matches_cpu_cached_sequence() -> Result<()> {
    let executor = match crate::MetalExecutor::new() {
        Ok(executor) => executor,
        Err(InferError::Metal(message)) if message.contains("aucun device") => return Ok(()),
        Err(error) => return Err(error),
    };
    let config = test_config();
    let layers = [
        test_linear_attention(),
        test_linear_attention(),
        test_linear_attention(),
    ];
    let mut cpu_caches = vec![LinearAttentionCache::default(); layers.len()];
    let mut resident_caches = vec![LinearAttentionCache::default(); layers.len()];
    let input = [
        vec![1.0, 0.0],
        vec![0.25, 0.75],
        vec![-0.5, 0.5],
        vec![0.1, -0.3],
    ];

    for row in input {
        let mut cpu = Tensor::row(row.clone()).expect("invariant: token CPU valide");
        let mut metal = Tensor::row(row).expect("invariant: token Metal valide");
        for ((layer, cpu_cache), resident_cache) in layers
            .iter()
            .zip(cpu_caches.iter_mut())
            .zip(resident_caches.iter_mut())
        {
            cpu = layer
                .forward_cached(config, &cpu, cpu_cache)
                .expect("invariant: CPU cached valide");
            metal = layer
                .forward_resident_metal_step(config, &metal, resident_cache, &executor)
                .expect("invariant: Metal résident cached valide");
        }
        assert_close(cpu.data(), metal.data(), 2.0e-5);
    }
    Ok(())
}

fn test_config() -> LinearAttentionConfig {
    LinearAttentionConfig {
        num_key_heads: 1,
        num_value_heads: 1,
        key_head_dim: 2,
        value_head_dim: 2,
        conv_kernel_dim: 2,
        rms_eps: 1.0e-6,
    }
}

fn test_linear_attention() -> LinearAttention {
    LinearAttention::new(LinearAttentionWeights {
        in_proj_qkv: linear(Tensor::from_vec(
            vec![6, 2],
            vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 1.0],
        )),
        in_proj_z: linear(identity2()),
        in_proj_b: linear(Tensor::from_vec(vec![1, 2], vec![1.0, 0.0])),
        in_proj_a: linear(Tensor::from_vec(vec![1, 2], vec![0.0, 1.0])),
        out_proj: linear(identity2()),
        conv1d_weight: Tensor::from_vec(
            vec![6, 2, 1],
            vec![
                0.25, 1.0, 0.25, 1.0, 0.25, 1.0, 0.25, 1.0, 0.25, 1.0, 0.25, 1.0,
            ],
        )
        .expect("invariant: conv valide"),
        a_log: Tensor::from_vec(vec![1], vec![0.0]).expect("invariant: A_log valide"),
        dt_bias: Tensor::from_vec(vec![1], vec![0.0]).expect("invariant: dt_bias valide"),
        norm_weight: Tensor::from_vec(vec![2], vec![1.0, 1.0]).expect("invariant: norm valide"),
    })
}

fn linear(weight: Result<Tensor>) -> Linear {
    Linear::from_weight(
        LinearWeight::Dense(weight.expect("invariant: poids valide")),
        None,
    )
    .expect("invariant: linear valide")
}

fn identity2() -> Result<Tensor> {
    Tensor::from_vec(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0])
}

fn assert_close(left: &[f32], right: &[f32], tolerance: f32) {
    assert_eq!(left.len(), right.len());
    for (idx, (l, r)) in left.iter().zip(right.iter()).enumerate() {
        assert!(
            (l - r).abs() <= tolerance,
            "idx={idx}, left={l}, right={r}, tolerance={tolerance}"
        );
    }
}
