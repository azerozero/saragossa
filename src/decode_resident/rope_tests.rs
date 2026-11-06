//! Portes de précision RoPE et identité des K entre prefill et decode.
//! Ces tests exigent une phase GPU dédiée ; la compilation ne les exécute pas.

use super::tests::{
    encode_legacy_rope, lease_with, read_bf16_as_f32, read_f32, LEGACY_ROPE_DECODE_KERNEL,
};
use super::*;
use crate::metal_backend::rope::create_rope_frequencies;
use crate::metal_backend::{install_verify_tree_scope, PrefillAttentionSpec, VerifyTree};
use std::rc::Rc;

const POSITIONS: [usize; 5] = [5, 2051, 32_767, 131_071, 262_143];
const HEADS: usize = 2;
const ROWS: usize = 3;

#[derive(Clone, Copy, Debug)]
enum Path {
    Decode,
    Append,
    Rows,
    Tree,
    Prefill,
}

fn geometries() -> impl Iterator<Item = (usize, usize, usize, f32)> {
    [128, 256].into_iter().flat_map(|head_dim| {
        // RoPE complète, partielle, Gemma global (fd > rope_dims), NoPE.
        [
            (head_dim, head_dim),
            (head_dim / 2, head_dim / 2),
            (head_dim / 2, head_dim),
            (0, 0),
            (0, head_dim),
        ]
        .into_iter()
        .flat_map(move |(rope_dims, fd)| {
            [10_000.0, 1_000_000.0]
                .into_iter()
                .map(move |theta| (head_dim, rope_dims, fd, theta))
        })
    })
}

fn dims(
    head_dim: usize,
    rope_dims: usize,
    fd: usize,
    theta: f32,
    position: usize,
) -> FullAttnLayerDims {
    FullAttnLayerDims {
        hidden: HEADS * head_dim,
        q_heads: HEADS,
        kv_heads: HEADS,
        head_dim,
        attn_scalar: head_dim as f32,
        rope_dims,
        rope_frequency_dim: fd,
        position,
        window_start: 0,
        eps: 1.0e-6,
        theta,
        attn_output_gate: false,
    }
}

fn data(head_dim: usize, rows: usize) -> (Vec<f32>, Vec<f32>) {
    let input = (0..rows * HEADS * head_dim)
        .map(|i| ((i * 17 + 3) % 97) as f32 / 32.0 - 1.5)
        .collect();
    let weight = (0..head_dim)
        .map(|i| 0.5 + (i % 13) as f32 / 16.0)
        .collect();
    (input, weight)
}

/// Oracle commun : même table hôte, réduction RMS scalaire et trigonométrie
/// précise CPU. La queue reste normée et n’est jamais tournée.
fn reference(input: &[f32], weight: &[f32], dims: FullAttnLayerDims) -> Vec<f32> {
    let inv = crate::rope::inv_freq(dims.rope_frequency_dim, dims.theta);
    let mut out = input.to_vec();
    for head in out.chunks_exact_mut(dims.head_dim) {
        let sum = head.iter().map(|x| x * x).sum::<f32>();
        let inv_rms = 1.0 / (sum / dims.head_dim as f32 + dims.eps).sqrt();
        for (x, w) in head.iter_mut().zip(weight) {
            *x = *x * inv_rms * w;
        }
        let half = dims.rope_dims / 2;
        for (pair, frequency) in inv.iter().take(half).enumerate() {
            let angle = dims.position as f32 * frequency;
            let (sin, cos) = angle.sin_cos();
            let (first, second) = (head[pair], head[pair + half]);
            head[pair] = first * cos - second * sin;
            head[pair + half] = first * sin + second * cos;
        }
    }
    out
}

fn error(actual: &[f32], expected: &[f32]) -> f32 {
    assert_eq!(actual.len(), expected.len());
    actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

fn assert_precise(actual: &[f32], expected: &[f32], label: &str) {
    assert!(
        actual.iter().all(|v| v.is_finite()),
        "{label}: sortie non finie"
    );
    let max_abs = error(actual, expected);
    let relative = (actual
        .iter()
        .zip(expected)
        .map(|(a, b)| f64::from(a - b).powi(2))
        .sum::<f64>()
        / expected
            .iter()
            .map(|v| f64::from(*v).powi(2))
            .sum::<f64>()
            .max(1.0e-30))
    .sqrt();
    assert!(
        max_abs < 1.0e-5 && relative < 1.0e-5,
        "{label}: max_abs={max_abs:e}, relatif={relative:e}"
    );
}

fn prefill_spec(dims: FullAttnLayerDims, rows: usize) -> PrefillAttentionSpec {
    PrefillAttentionSpec {
        seq: rows,
        hidden_dim: dims.hidden,
        q_heads: dims.q_heads,
        kv_heads: dims.kv_heads,
        head_dim: dims.head_dim,
        rope_dims: dims.rope_dims,
        rope_frequency_dim: dims.rope_frequency_dim,
        rope_theta: dims.theta,
        attn_scalar: dims.attn_scalar,
        window: None,
        k_eq_v: false,
        value_norm: false,
        eps: dims.eps,
    }
}

fn gpu_state() -> Result<DecodeResidentState> {
    tests::try_state()?.ok_or_else(|| InferError::Metal("test RoPE : GPU requis".to_string()))
}

fn check_path(path: Path) -> Result<()> {
    let state = gpu_state()?;
    let executor = if matches!(path, Path::Prefill) {
        Some(MetalExecutor::new()?)
    } else {
        None
    };
    let legacy = kernels::compile_kernel(
        state.device(),
        LEGACY_ROPE_DECODE_KERNEL,
        "rms_norm_rope_heads_decode_f32",
    )?;
    let mut legacy_long_error = 0.0_f32;
    for (head_dim, rope_dims, fd, theta) in geometries() {
        let frequencies = create_rope_frequencies(state.device(), theta, fd)?;
        let rows = if matches!(path, Path::Rows | Path::Tree | Path::Prefill) {
            ROWS
        } else {
            1
        };
        let width = HEADS * head_dim;
        let (input, weight) = data(head_dim, rows);
        let input_gpu = lease_with(&state, &input)?;
        let weight_gpu = lease_with(&state, &weight)?;
        let output = state.scratch().lease(rows * width, GpuElement::F32)?;
        let old_output = state.scratch().lease(width, GpuElement::F32)?;
        for position in POSITIONS {
            // La position demandée apparaît aussi sur une ligne non nulle,
            // afin de tester l’addition de l’offset et les profondeurs d’arbre.
            let base = position - usize::from(rows > 1);
            let dims = dims(head_dim, rope_dims, fd, theta, base);
            assert_eq!(frequencies.length(), (fd / 2).max(1) as u64 * 4);
            let mut kv = state.full_attention(1, HEADS, HEADS, head_dim, false)?;
            let command = state.queue().new_command_buffer();
            let encoder = command.new_compute_command_encoder();
            match path {
                Path::Decode => state.encode_rms_norm_rope_decode(
                    encoder,
                    input_gpu.tensor().buffer(),
                    weight_gpu.tensor().buffer(),
                    output.tensor().buffer(),
                    HEADS,
                    head_dim,
                    rope_dims,
                    &frequencies,
                    base,
                    dims.eps,
                )?,
                Path::Append => {
                    let hit = crate::runtime_flags::with_rope_append_kv_override(true, || {
                        kv.encode_rope_append_kv(
                            encoder,
                            input_gpu.tensor().buffer(),
                            0,
                            weight_gpu.tensor().buffer(),
                            input_gpu.tensor().buffer(),
                            0,
                            &dims,
                            &frequencies,
                        )
                    })?;
                    assert!(hit, "append fusionné attendu");
                }
                Path::Rows | Path::Tree => {
                    let tree = if matches!(path, Path::Tree) {
                        Some(Rc::new(VerifyTree::new(
                            &[0, 0, 0],
                            &[0, 1, 1],
                            &[1, 3, 5],
                        )?))
                    } else {
                        None
                    };
                    let _scope = tree.map(install_verify_tree_scope);
                    state.encode_rms_norm_rope_decode_rows(
                        encoder,
                        state.rows_kernels()?,
                        input_gpu.tensor().buffer(),
                        0,
                        width,
                        weight_gpu.tensor().buffer(),
                        output.tensor().buffer(),
                        width,
                        HEADS,
                        head_dim,
                        rope_dims,
                        &frequencies,
                        base,
                        rows,
                        dims.eps,
                    )?;
                }
                Path::Prefill => executor
                    .as_ref()
                    .expect("invariant: exécuteur prefill créé")
                    .encode_rms_norm_rope_heads(
                        encoder,
                        input_gpu.tensor().buffer(),
                        weight_gpu.tensor().buffer(),
                        output.tensor().buffer(),
                        prefill_spec(dims, rows),
                        HEADS,
                        base,
                    )?,
            }
            encode_legacy_rope(
                encoder,
                &legacy,
                input_gpu.tensor(),
                weight_gpu.tensor(),
                old_output.tensor(),
                &dims,
            )?;
            encoder.end_encoding();
            commit_and_wait(command)?;
            let actual = if matches!(path, Path::Append) {
                read_f32(kv.keys())
            } else {
                read_f32(output.tensor())
            };
            for row in 0..rows {
                let offset = if matches!(path, Path::Tree) {
                    usize::from(row > 0)
                } else {
                    row
                };
                let expected = reference(
                    &input[row * width..(row + 1) * width],
                    &weight,
                    FullAttnLayerDims {
                        position: base + offset,
                        ..dims
                    },
                );
                let label = format!(
                    "{path:?} d={head_dim} rd={rope_dims} fd={fd} θ={theta} pos={} ligne={row}",
                    base + offset
                );
                assert_precise(&actual[row * width..(row + 1) * width], &expected, &label);
                // Porte explicite sur la queue, aussi pour NoPE où elle couvre toute la tête.
                for head in 0..HEADS {
                    let tail = head * head_dim + rope_dims..(head + 1) * head_dim;
                    assert_precise(
                        &actual[row * width + tail.start..row * width + tail.end],
                        &expected[tail],
                        &label,
                    );
                }
                if row == 0 && position >= 32_767 {
                    legacy_long_error =
                        legacy_long_error.max(error(&read_f32(old_output.tensor()), &expected));
                }
            }
        }
    }
    assert!(
        legacy_long_error > 1.0e-4,
        "témoin legacy insensible : {legacy_long_error:e}"
    );
    Ok(())
}

#[test]
#[ignore = "GPU Metal requis : échoue explicitement sans device"]
fn rope_decode_matches_common_cpu_at_long_positions() -> Result<()> {
    check_path(Path::Decode)
}

#[test]
#[ignore = "GPU Metal requis : échoue explicitement sans device"]
fn rope_append_matches_common_cpu_at_long_positions() -> Result<()> {
    check_path(Path::Append)
}

#[test]
#[ignore = "GPU Metal requis : échoue explicitement sans device"]
fn rope_rows_matches_common_cpu_at_long_positions() -> Result<()> {
    check_path(Path::Rows)?;
    check_path(Path::Tree)
}

#[test]
#[ignore = "GPU Metal requis : échoue explicitement sans device"]
fn rope_prefill_matches_common_cpu_at_long_positions() -> Result<()> {
    check_path(Path::Prefill)
}

/// Les mêmes entrées brutes passent seulement par RMSNorm+RoPE, sans QKV ni
/// attention. Pour pos + offset < 2^24, l’addition des entiers en f32 du
/// prefill et la conversion entière du decode sont exactes : K doit être
/// identique en f32, puis après la conversion GPU en bf16.
#[test]
#[ignore = "GPU Metal requis : échoue explicitement sans device"]
fn rope_prefill_and_decode_keys_are_bit_identical() -> Result<()> {
    let state = gpu_state()?;
    let executor = MetalExecutor::new()?;
    for (head_dim, rope_dims, fd, theta) in geometries() {
        let frequencies = create_rope_frequencies(state.device(), theta, fd)?;
        let width = HEADS * head_dim;
        let (input, weight) = data(head_dim, ROWS);
        let input_gpu = lease_with(&state, &input)?;
        let weight_gpu = lease_with(&state, &weight)?;
        let prefill = state.scratch().lease(ROWS * width, GpuElement::F32)?;
        let decode = (0..ROWS)
            .map(|_| state.scratch().lease(width, GpuElement::F32))
            .collect::<Result<Vec<_>>>()?;
        for position in POSITIONS {
            let base = position - 1;
            assert!(
                base + ROWS - 1 < 1 << 24,
                "invariant : positions exactement représentables en f32"
            );
            let dims = dims(head_dim, rope_dims, fd, theta, base);
            let mut prefill_kv =
                state.full_attention_bf16_for_test(ROWS, HEADS, HEADS, head_dim)?;
            let mut decode_kv = state.full_attention_bf16_for_test(ROWS, HEADS, HEADS, head_dim)?;
            let command = state.queue().new_command_buffer();
            let encoder = command.new_compute_command_encoder();
            executor.encode_rms_norm_rope_heads(
                encoder,
                input_gpu.tensor().buffer(),
                weight_gpu.tensor().buffer(),
                prefill.tensor().buffer(),
                prefill_spec(dims, ROWS),
                HEADS,
                base,
            )?;
            for (row, output) in decode.iter().enumerate() {
                let offset = (row * width * std::mem::size_of::<f32>()) as u64;
                state.encode_rms_norm_rope_decode_with_offset(
                    encoder,
                    input_gpu.tensor().buffer(),
                    offset,
                    weight_gpu.tensor().buffer(),
                    output.tensor().buffer(),
                    HEADS,
                    head_dim,
                    rope_dims,
                    &frequencies,
                    base + row,
                    dims.eps,
                )?;
                prefill_kv.encode_append_kv_with_offsets(
                    encoder,
                    prefill.tensor().buffer(),
                    offset,
                    input_gpu.tensor().buffer(),
                    offset,
                )?;
                decode_kv.encode_append_kv_with_offsets(
                    encoder,
                    output.tensor().buffer(),
                    0,
                    input_gpu.tensor().buffer(),
                    offset,
                )?;
            }
            encoder.end_encoding();
            commit_and_wait(command)?;
            let bits = |values: Vec<f32>| values.into_iter().map(f32::to_bits).collect::<Vec<_>>();
            let prefill_bits = bits(read_f32(prefill.tensor()));
            let decode_bits = decode
                .iter()
                .flat_map(|x| bits(read_f32(x.tensor())))
                .collect::<Vec<_>>();
            let label = format!("d={head_dim} rd={rope_dims} fd={fd} θ={theta} pos={position}");
            assert_eq!(prefill_bits, decode_bits, "{label}: K f32");
            assert_eq!(
                bits(read_bf16_as_f32(prefill_kv.keys())),
                bits(read_bf16_as_f32(decode_kv.keys())),
                "{label}: K bf16"
            );
        }
    }
    Ok(())
}
