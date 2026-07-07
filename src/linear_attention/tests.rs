//! Tests de l attention lineaire CPU et Metal, dont l'oracle GDN naïf de
//! référence (`naive_gdn_reference`) partagé avec `metal_backend/tests.rs`.

use super::*;
use crate::LinearWeight;
use proptest::prelude::*;

/// Oracle CPU NAÏF de la récurrence Gated DeltaNet (équations de la doc de module).
///
/// Boucle séquentielle token par token, f32, ZÉRO chunking, ZÉRO rayon —
/// formulation indépendante du chemin prod (`gated_delta`) : l'état décayé
/// `g_t·S_{t−1}` est matérialisé avant lecture, puis `S_t = g_t·S_{t−1} + Δ·k_tᵀ`
/// et `y_t = S_t·q_t` sont recalculés terme à terme. Sert de référence directe
/// aux chemins chunkés (kernel Metal `chunk_delta_seq_layout`) et au chemin CPU.
///
/// Layouts plats : `q`/`k` `[seq, H_k·d_k]`, `v` `[seq, H_v·d_v]`, `g`/`beta`
/// `[seq, H_v]` ; `state` `[H_v·d_v·d_k]` (identique à `cache.ssm`), mis à jour
/// in place. Renvoie `y` `[seq, H_v·d_v]`.
#[expect(
    clippy::too_many_arguments,
    reason = "oracle de référence : les cinq flux q/k/v/g/β restent explicites"
)]
pub(crate) fn naive_gdn_reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    seq: usize,
    config: LinearAttentionConfig,
    state: &mut [f32],
) -> Vec<f32> {
    let dk = config.key_head_dim;
    let dv = config.value_head_dim;
    let heads_v = config.num_value_heads;
    let repeat = heads_v / config.num_key_heads;
    let key_dim = config.num_key_heads * dk;
    let value_dim = heads_v * dv;
    assert_eq!(q.len(), seq * key_dim, "oracle: q");
    assert_eq!(k.len(), seq * key_dim, "oracle: k");
    assert_eq!(v.len(), seq * value_dim, "oracle: v");
    assert_eq!(g.len(), seq * heads_v, "oracle: g");
    assert_eq!(beta.len(), seq * heads_v, "oracle: beta");
    assert_eq!(state.len(), heads_v * dv * dk, "oracle: state");

    let mut out = vec![0.0_f32; seq * value_dim];
    for t in 0..seq {
        for value_head in 0..heads_v {
            // GQA : q/k de la tête clé κ = ⌊h/repeat⌋, v/gates propres à h.
            let key_head = value_head / repeat;
            let q_t = &q[t * key_dim + key_head * dk..t * key_dim + (key_head + 1) * dk];
            let k_t = &k[t * key_dim + key_head * dk..t * key_dim + (key_head + 1) * dk];
            let g_t = g[t * heads_v + value_head];
            let beta_t = beta[t * heads_v + value_head];
            for value_col in 0..dv {
                let row = (value_head * dv + value_col) * dk;
                let state_row = &mut state[row..row + dk];
                // g_t·S_{t−1} matérialisé, puis kv = (g_t·S_{t−1})·k_t.
                let mut kv = 0.0_f32;
                for (state_value, key_value) in state_row.iter_mut().zip(k_t.iter()) {
                    *state_value *= g_t;
                    kv += *state_value * *key_value;
                }
                // Δ = β_t·(v_t − kv), puis S_t = g_t·S_{t−1} + Δ·k_tᵀ et y_t = S_t·q_t.
                let delta = beta_t * (v[t * value_dim + value_head * dv + value_col] - kv);
                let mut y = 0.0_f32;
                for ((state_value, key_value), query_value) in
                    state_row.iter_mut().zip(k_t.iter()).zip(q_t.iter())
                {
                    *state_value += delta * *key_value;
                    y += *state_value * *query_value;
                }
                out[t * value_dim + value_head * dv + value_col] = y;
            }
        }
    }
    out
}

/// Générateur déterministe (LCG Knuth 64 bits, seed fixe) → f32 dans `[lo, hi)`.
/// Aucune source d'entropie ambiante : les entrées synthétiques des oracles sont
/// reproductibles à l'octet près d'un run à l'autre.
pub(crate) struct DeterministicF32 {
    state: u64,
}

impl DeterministicF32 {
    pub(crate) fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub(crate) fn next_in(&mut self, lo: f32, hi: f32) -> f32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // 24 bits de poids fort → une fraction exactement représentable en f32.
        let unit = ((self.state >> 40) & 0x00ff_ffff) as f32 / 16_777_216.0;
        lo + (hi - lo) * unit
    }

    pub(crate) fn fill(&mut self, len: usize, lo: f32, hi: f32) -> Vec<f32> {
        (0..len).map(|_| self.next_in(lo, hi)).collect()
    }
}

/// Entrées synthétiques GDN d'échelle réaliste (post-norm) pour un couple
/// `(config, seq)` : q ~ RMSNormUnit·d_k⁻¹, k ~ RMSNormUnit·d_k^(−1/2), v O(1),
/// decay ∈ [0,9, 0,999) (γ intra-chunk borné → ratios γ_i/γ_j ≤ 0,9⁻¹⁶ ≈ 5,4),
/// β ∈ [0,1, 0,9).
pub(crate) struct GdnInputs {
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub g: Vec<f32>,
    pub beta: Vec<f32>,
}

pub(crate) fn synthetic_gdn_inputs(
    config: LinearAttentionConfig,
    seq: usize,
    seed: u64,
) -> GdnInputs {
    let mut rng = DeterministicF32::new(seed);
    let key_dim = config.num_key_heads * config.key_head_dim;
    let value_dim = config.num_value_heads * config.value_head_dim;
    let q_amp = (config.key_head_dim as f32).powf(-1.0);
    let k_amp = (config.key_head_dim as f32).powf(-0.5);
    GdnInputs {
        q: rng.fill(seq * key_dim, -q_amp, q_amp),
        k: rng.fill(seq * key_dim, -k_amp, k_amp),
        v: rng.fill(seq * value_dim, -1.0, 1.0),
        g: rng.fill(seq * config.num_value_heads, 0.9, 0.999),
        beta: rng.fill(seq * config.num_value_heads, 0.1, 0.9),
    }
}

/// Longueurs ciblées sur les frontières du chunk Metal C=16 :
/// 1, C−1, C, C+1 et 3×C+7 (trois chunks pleins + un partiel).
pub(crate) const CHUNK_BOUNDARY_LENGTHS: [usize; 5] = [1, 15, 16, 17, 55];

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

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn linear_attention_batch_resident_dk128_matches_cpu_cached_sequence() -> Result<()> {
    let executor = match crate::MetalExecutor::new() {
        Ok(executor) => executor,
        Err(InferError::Metal(message)) if message.contains("aucun device") => return Ok(()),
        Err(error) => return Err(error),
    };
    let config = test_config_dk128();
    let attn = test_linear_attention_dk128()?;
    let input = Tensor::from_vec(vec![4, 2], vec![1.0, 0.0, 0.25, 0.75, -0.5, 0.5, 0.1, -0.3])?;

    let mut cpu_cache = LinearAttentionCache::default();
    let mut cpu_rows = Vec::new();
    for row in 0..4 {
        let token = Tensor::from_vec(vec![1, 2], input.row_slice(row)?.to_vec())?;
        let output = attn.forward_cached(config, &token, &mut cpu_cache)?;
        cpu_rows.extend_from_slice(output.as_row()?);
    }
    let cpu = Tensor::from_vec(vec![4, 2], cpu_rows)?;

    let key_dim = config.key_dim()?;
    let value_dim = config.value_dim()?;
    let conv_dim = key_dim
        .checked_mul(2)
        .and_then(|twice| twice.checked_add(value_dim))
        .ok_or_else(|| InferError::Shape("test linear-attn conv_dim déborde".to_string()))?;
    let conv_state = vec![0.0; (config.conv_kernel_dim - 1) * conv_dim];
    let ssm_state = vec![0.0; config.num_value_heads * config.value_head_dim * config.key_head_dim];
    let mut metal_state = None;
    let metal = executor.linear_attention_cached_batch_resident(
        &input,
        &attn.in_proj_qkv,
        &attn.in_proj_z,
        &attn.in_proj_b,
        &attn.in_proj_a,
        &attn.out_proj,
        &attn.conv1d_weight,
        &attn.a_log,
        &attn.dt_bias,
        &attn.norm_weight,
        &conv_state,
        &ssm_state,
        &mut metal_state,
        crate::metal_backend::LinearAttentionStepSpec {
            num_key_heads: config.num_key_heads,
            num_value_heads: config.num_value_heads,
            key_head_dim: config.key_head_dim,
            value_head_dim: config.value_head_dim,
            conv_kernel_dim: config.conv_kernel_dim,
            rms_eps: config.rms_eps,
        },
    )?;
    assert_close(cpu.data(), metal.data(), 5.0e-4);
    Ok(())
}

/// Le chemin CPU prod (`gated_delta`, rayon+GQA) reproduit EXACTEMENT l'oracle
/// naïf : même ordre séquentiel des opérations f32 par tête (rayon ne
/// parallélise que des états disjoints), donc égalité bit à bit attendue —
/// tolérance 0,0 assumée (tout écart = divergence d'algorithme, pas du bruit).
#[test]
fn gated_delta_matches_naive_oracle_across_lengths_and_gqa_heads() -> Result<()> {
    let configs = [
        // GQA repeat=2, dims quelconques (chemin générique).
        LinearAttentionConfig {
            num_key_heads: 2,
            num_value_heads: 4,
            key_head_dim: 8,
            value_head_dim: 6,
            conv_kernel_dim: 4,
            rms_eps: 1.0e-6,
        },
        // GQA repeat=4 mono-tête clé, dk128 = la forme du 35B-A3B (kernel chunké).
        LinearAttentionConfig {
            num_key_heads: 1,
            num_value_heads: 4,
            key_head_dim: 128,
            value_head_dim: 128,
            conv_kernel_dim: 4,
            rms_eps: 1.0e-6,
        },
    ];
    for (config_index, config) in configs.into_iter().enumerate() {
        let key_dim = config.key_dim()?;
        let value_dim = config.value_dim()?;
        for (length_index, &seq) in CHUNK_BOUNDARY_LENGTHS.iter().enumerate() {
            let seed = 0x5eed_0000 + (config_index * 16 + length_index) as u64;
            let inputs = synthetic_gdn_inputs(config, seq, seed);

            let mut oracle_state =
                vec![0.0_f32; config.num_value_heads * config.value_head_dim * config.key_head_dim];
            let oracle_y = naive_gdn_reference(
                &inputs.q,
                &inputs.k,
                &inputs.v,
                &inputs.g,
                &inputs.beta,
                seq,
                config,
                &mut oracle_state,
            );

            let mut cache = LinearAttentionCache::default();
            let y = gated_delta(
                &Tensor::from_vec(vec![seq, key_dim], inputs.q.clone())?,
                &Tensor::from_vec(vec![seq, key_dim], inputs.k.clone())?,
                &Tensor::from_vec(vec![seq, value_dim], inputs.v.clone())?,
                &Tensor::from_vec(vec![seq, config.num_value_heads], inputs.g.clone())?,
                &Tensor::from_vec(vec![seq, config.num_value_heads], inputs.beta.clone())?,
                config,
                &mut cache,
            )?;

            assert_eq!(
                y.data(),
                oracle_y.as_slice(),
                "y: config {config_index}, seq {seq}"
            );
            assert_eq!(
                cache.ssm, oracle_state,
                "état SSM final: config {config_index}, seq {seq}"
            );
        }
    }
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Propriété structurelle : la récurrence GDN commute avec le découpage en
    /// chunks ARBITRAIRES — traiter la séquence en un bloc ou en sous-blocs
    /// quelconques (état porté par le cache) donne une sortie bit-identique,
    /// car `gated_delta` reste séquentiel par position quel que soit le batch.
    /// C'est l'invariant qui fonde la forme chunkée C=16 du kernel Metal.
    #[test]
    fn gated_delta_invariant_par_decoupage_en_chunks(
        seq in 1_usize..40,
        seed in proptest::num::u64::ANY,
        splits in proptest::collection::vec(1_usize..8, 0..12),
    ) {
        let config = LinearAttentionConfig {
            num_key_heads: 2,
            num_value_heads: 4,
            key_head_dim: 4,
            value_head_dim: 4,
            conv_kernel_dim: 4,
            rms_eps: 1.0e-6,
        };
        let key_dim = config.key_dim().expect("invariant: key_dim");
        let value_dim = config.value_dim().expect("invariant: value_dim");
        let heads = config.num_value_heads;
        let inputs = synthetic_gdn_inputs(config, seq, seed);
        let rows = |data: &[f32], cols: usize, start: usize, len: usize| -> Tensor {
            Tensor::from_vec(vec![len, cols], data[start * cols..(start + len) * cols].to_vec())
                .expect("invariant: tensor chunk")
        };

        // Référence : toute la séquence en un seul appel.
        let mut full_cache = LinearAttentionCache::default();
        let full = gated_delta(
            &rows(&inputs.q, key_dim, 0, seq),
            &rows(&inputs.k, key_dim, 0, seq),
            &rows(&inputs.v, value_dim, 0, seq),
            &rows(&inputs.g, heads, 0, seq),
            &rows(&inputs.beta, heads, 0, seq),
            config,
            &mut full_cache,
        ).expect("invariant: gated_delta plein");

        // Même séquence découpée aux points arbitraires proposés par proptest.
        let mut chunked_cache = LinearAttentionCache::default();
        let mut chunked = Vec::with_capacity(seq * value_dim);
        let mut start = 0_usize;
        let mut split_iter = splits.into_iter();
        while start < seq {
            let len = split_iter.next().unwrap_or(seq - start).min(seq - start);
            let out = gated_delta(
                &rows(&inputs.q, key_dim, start, len),
                &rows(&inputs.k, key_dim, start, len),
                &rows(&inputs.v, value_dim, start, len),
                &rows(&inputs.g, heads, start, len),
                &rows(&inputs.beta, heads, start, len),
                config,
                &mut chunked_cache,
            ).expect("invariant: gated_delta chunk");
            chunked.extend_from_slice(out.data());
            start += len;
        }

        prop_assert_eq!(full.data(), chunked.as_slice());
        prop_assert_eq!(&full_cache.ssm, &chunked_cache.ssm);

        // Et les deux coïncident bit à bit avec l'oracle naïf (même ordre f32).
        let mut oracle_state = vec![0.0_f32; heads * config.value_head_dim * config.key_head_dim];
        let oracle_y = naive_gdn_reference(
            &inputs.q, &inputs.k, &inputs.v, &inputs.g, &inputs.beta,
            seq, config, &mut oracle_state,
        );
        prop_assert_eq!(full.data(), oracle_y.as_slice());
        prop_assert_eq!(&full_cache.ssm, &oracle_state);
    }
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

fn test_config_dk128() -> LinearAttentionConfig {
    LinearAttentionConfig {
        num_key_heads: 1,
        num_value_heads: 1,
        key_head_dim: 128,
        value_head_dim: 128,
        conv_kernel_dim: 4,
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

fn test_linear_attention_dk128() -> Result<LinearAttention> {
    let config = test_config_dk128();
    let key_dim = config.key_dim()?;
    let value_dim = config.value_dim()?;
    let conv_dim = key_dim
        .checked_mul(2)
        .and_then(|twice| twice.checked_add(value_dim))
        .ok_or_else(|| InferError::Shape("test linear-attn conv_dim déborde".to_string()))?;
    Ok(LinearAttention::new(LinearAttentionWeights {
        in_proj_qkv: linear(patterned_tensor(
            vec![conv_dim, 2],
            conv_dim * 2,
            0.045,
            0.1,
        )),
        in_proj_z: linear(patterned_tensor(
            vec![value_dim, 2],
            value_dim * 2,
            0.035,
            0.3,
        )),
        in_proj_b: linear(Tensor::from_vec(vec![1, 2], vec![0.05, -0.03])),
        in_proj_a: linear(Tensor::from_vec(vec![1, 2], vec![-0.02, 0.04])),
        out_proj: linear(patterned_tensor(
            vec![2, value_dim],
            2 * value_dim,
            0.04,
            0.7,
        )),
        conv1d_weight: patterned_tensor(
            vec![conv_dim, config.conv_kernel_dim, 1],
            conv_dim * config.conv_kernel_dim,
            0.025,
            1.1,
        )?,
        a_log: Tensor::from_vec(vec![1], vec![0.01])?,
        dt_bias: Tensor::from_vec(vec![1], vec![-0.2])?,
        norm_weight: Tensor::from_vec(vec![value_dim], vec![1.0; value_dim])?,
    }))
}

fn patterned_tensor(shape: Vec<usize>, len: usize, scale: f32, phase: f32) -> Result<Tensor> {
    let data = (0..len)
        .map(|idx| (((idx as f32) + phase) * 0.017).sin() * scale)
        .collect();
    Tensor::from_vec(shape, data)
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
