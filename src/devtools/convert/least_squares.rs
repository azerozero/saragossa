//! Échelle et biais affines ajustés aux moindres carrés, format MLX gs64 inchangé.
//!
//! Portage de `tools/quant/opt_quant.py` : on part du min-max bf16, on
//! réajuste en alternance les codes et `(scale, bias)` par les équations
//! normales, puis chaque groupe garde la meilleure des deux solutions après
//! arrondi bf16. L'erreur quadratique d'un groupe ne peut donc pas augmenter et
//! les kernels ne changent pas. L'échelle peut être négative : seul zéro est
//! dégénéré (le loader restitue `q·scale + bias` quel que soit le signe).

use crate::quantization::{bf16_bits, bf16_value, quantize_affine_bitstream_bf16, QuantError};
use crate::{InferError, Result};

/// Itérations codes ↔ paramètres, comme `opt_quant.fit`.
const ITERS: usize = 8;

/// Codes `clip(round((w − b)/s), 0, levels)` d'un groupe, division en f64
/// comme le min-max ; une échelle nulle donne des codes nuls.
fn codes(values: &[f32], scale: f64, bias: f64, levels: f64, out: &mut [u32]) {
    for (code, &value) in out.iter_mut().zip(values) {
        *code = if scale == 0.0 {
            0
        } else {
            ((f64::from(value) - bias) / scale)
                .round()
                .clamp(0.0, levels) as u32
        };
    }
}

/// `Σ(w − (q·s + b))²`, restitution en f32 comme le loader.
fn group_error(values: &[f32], codes: &[u32], scale: f32, bias: f32) -> f64 {
    values
        .iter()
        .zip(codes)
        .map(|(&value, &code)| f64::from(value - (code as f32 * scale + bias)).powi(2))
        .sum()
}

/// Quantifie `data` (`[rows, cols]`) en affine u4/u8 gs64 : min-max bf16 de
/// [`quantize_affine_bitstream_bf16`], puis ajustement aux moindres carrés
/// des groupes qu'il améliore. Mêmes sorties et même erreur rapportée.
///
/// # Errors
///
/// Renvoie l'erreur du min-max, ou refuse une largeur autre que 4 ou 8 bits.
pub(crate) fn quantize_affine_ls_bf16(
    data: &[f32],
    cols: usize,
    group_size: usize,
    bits: usize,
    packed: &mut [u32],
    scales: &mut [u16],
    biases: &mut [u16],
) -> Result<QuantError> {
    if !matches!(bits, 4 | 8) {
        return Err(InferError::Config(format!(
            "moindres carrés : largeur {bits} bits non prise en charge (4 ou 8)"
        )));
    }
    quantize_affine_bitstream_bf16(data, cols, group_size, bits, packed, scales, biases)?;
    let levels = f64::from((1_u32 << bits) - 1);
    let lanes = 32 / bits;
    let mask = (1_u32 << bits) - 1;
    let words_per_group = group_size / lanes;
    let groups = cols / group_size;
    let n = group_size as f64;
    let mut q0 = vec![0_u32; group_size];
    let mut q = vec![0_u32; group_size];
    let mut error = QuantError::default();
    for (index, values) in data.chunks_exact(group_size).enumerate() {
        let (row, group) = (index / groups, index % groups);
        let word0 = row * (cols / lanes) + group * words_per_group;
        let words = &mut packed[word0..word0 + words_per_group];
        for (lane, code) in q0.iter_mut().enumerate() {
            *code = (words[lane / lanes] >> ((lane % lanes) * bits)) & mask;
        }
        let (s0, b0) = (bf16_value(scales[index]), bf16_value(biases[index]));
        let e0 = group_error(values, &q0, s0, b0);
        let (mut s, mut b) = (f64::from(s0), f64::from(b0));
        for _ in 0..ITERS {
            codes(values, s, b, levels, &mut q);
            let (mut sq, mut sqq, mut sw, mut sqw) = (0.0, 0.0, 0.0, 0.0);
            for (&code, &value) in q.iter().zip(values) {
                let (code, value) = (f64::from(code), f64::from(value));
                sq += code;
                sqq += code * code;
                sw += value;
                sqw += code * value;
            }
            let det = n * sqq - sq * sq;
            if det > 0.0 {
                let s1 = (n * sqw - sq * sw) / det;
                if s1 != 0.0 && s1.is_finite() {
                    s = s1;
                    b = (sw - s1 * sq) / n;
                }
            }
        }
        let (s_bits, b_bits) = (bf16_bits(s as f32), bf16_bits(b as f32));
        let (s1, b1) = (bf16_value(s_bits), bf16_value(b_bits));
        let finite = s1.is_finite() && b1.is_finite() && (levels as f32 * s1 + b1).is_finite();
        let mut e = e0;
        if finite {
            codes(values, f64::from(s1), f64::from(b1), levels, &mut q);
            let e1 = group_error(values, &q, s1, b1);
            if e1 < e0 {
                e = e1;
                scales[index] = s_bits;
                biases[index] = b_bits;
                words.fill(0);
                for (lane, &code) in q.iter().enumerate() {
                    words[lane / lanes] |= code << ((lane % lanes) * bits);
                }
            }
        }
        error.sum_sq_err += e;
        error.sum_sq += values.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>();
    }
    Ok(error)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dequant(packed: &[u32], scales: &[u16], biases: &[u16], bits: usize) -> Vec<f32> {
        let lanes = 32 / bits;
        let mask = (1_u32 << bits) - 1;
        (0..packed.len() * lanes)
            .map(|i| {
                let code = (packed[i / lanes] >> ((i % lanes) * bits)) & mask;
                code as f32 * bf16_value(scales[i / 64]) + bf16_value(biases[i / 64])
            })
            .collect()
    }

    /// Jamais pire que le min-max, groupe par groupe, en u4 comme en u8.
    #[test]
    fn never_worse_than_minmax_per_group() {
        let data: Vec<f32> = (0..4 * 256)
            .map(|i| ((i as f32 * 0.37).sin() * 3.0).powi(3) + (i % 7) as f32 * 0.01)
            .collect();
        for bits in [4, 8] {
            let words = data.len() * bits / 32;
            let groups = data.len() / 64;
            let (mut p0, mut s0, mut b0) = (vec![0; words], vec![0; groups], vec![0; groups]);
            let (mut p1, mut s1, mut b1) = (vec![0; words], vec![0; groups], vec![0; groups]);
            let mm =
                quantize_affine_bitstream_bf16(&data, 256, 64, bits, &mut p0, &mut s0, &mut b0)
                    .expect("invariant: min-max valide");
            let ls = quantize_affine_ls_bf16(&data, 256, 64, bits, &mut p1, &mut s1, &mut b1)
                .expect("invariant: moindres carrés valides");
            assert!(ls.sum_sq_err < mm.sum_sq_err, "{bits} bits");
            let (d0, d1) = (dequant(&p0, &s0, &b0, bits), dequant(&p1, &s1, &b1, bits));
            for (g, values) in data.chunks(64).enumerate() {
                let err = |d: &[f32]| -> f64 {
                    values
                        .iter()
                        .zip(&d[g * 64..])
                        .map(|(&v, &x)| f64::from(v - x).powi(2))
                        .sum()
                };
                assert!(err(&d1) <= err(&d0), "{bits} bits, groupe {g}");
            }
            let total: f64 = data
                .iter()
                .zip(&d1)
                .map(|(&v, &x)| f64::from(v - x).powi(2))
                .sum();
            assert!((total - ls.sum_sq_err).abs() <= 1e-9 * total.max(1.0));
        }
    }
}
