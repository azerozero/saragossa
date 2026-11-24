//! Échelle et biais affines ajustés aux moindres carrés, format MLX gs64 inchangé.
//!
//! Portage de `tools/quant/opt_quant.py`. Par groupe, trois candidats sont
//! comparés sur l'erreur quadratique de la restitution `q·scale + bias` lue par
//! le loader, et le meilleur est gardé (égalité : ordre ci-dessous) :
//!
//! 1. le min-max bf16 de [`quantize_affine_bitstream_bf16`] (sortie de `--minmax`) ;
//! 2. la grille de `mx.quantize` (`affine_quantize`,
//!    `mlx/backend/metal/kernels/quantized.h`), paramètres arrondis bf16 et codes
//!    recalculés contre eux, comme le `e0` de `opt_quant.fit` ;
//! 3. 8 alternances codes ↔ équations normales partant de (2), arrondies bf16.
//!
//! Exception : un groupe portant un « super-poids » (max |w| du groupe au-delà
//! de [`SUPER_WEIGHT_RATIO`] fois la médiane |w| de sa ligne) ne garde que des
//! candidats qui restituent ce bord exactement : la grille MLX (2) et des
//! moindres carrés à biais fixé sur ce bord ([`exact_edge_fit`]). L'erreur
//! quadratique non pondérée y préfère sinon un léger décalage du bord, que
//! les activations massives multiplient : sur Mistral-7B v0.3,
//! `layers.1.mlp.down_proj` pesait à lui seul plus que tout l'écart de KL
//! entre cette LS et `mx.quantize`.
//!
//! Hors de ces groupes, le résultat n'est jamais pire que l'un ou l'autre min-max. Écarts
//! connus avec `opt_quant.py` : équations normales en f64 (f32 chez MLX) et
//! candidat (1) en plus. L'alternance est une descente locale : sur une grille
//! dont des codes restent inutilisés, elle peut s'arrêter avant l'optimum
//! exact. L'échelle peut être négative (MLX en produit) : seul zéro est
//! dégénéré.

use crate::quantization::{bf16_bits, bf16_value, quantize_affine_bitstream_bf16, QuantError};
use crate::{InferError, Result};

/// Itérations codes ↔ paramètres, comme `opt_quant.fit`.
const ITERS: usize = 8;

/// Au-delà de ce rapport entre le max |w| d'un groupe et la médiane |w| de sa
/// ligne, le groupe porte un super-poids (Yu et al. 2024, « The Super Weight
/// in Large Language Models ») et son bord doit être restitué exactement.
/// 50 sépare nettement ces poids sur Mistral-7B v0.3 (lignes 2070 et 3398 de
/// `layers.1.mlp.down_proj`) d'un bruit gaussien, dont le rapport reste vers 7.
const SUPER_WEIGHT_RATIO: f32 = 50.0;

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

/// `(scale, bias)` f32 de `affine_quantize` (MLX, bits ≥ 2), à la lettre :
/// `w_max` part de 0, l'échelle prend le signe du bord de plus grande valeur
/// absolue et le biais est ce bord, sauf s'il tombe sur le code 0.
fn mlx_params(values: &[f32], levels: f32) -> (f32, f32) {
    let (w_min, w_max) = values
        .iter()
        .fold((f32::MAX, 0.0_f32), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    let scale = ((w_max - w_min) / levels).max(1.0e-7);
    let side = w_min.abs() > w_max.abs();
    let scale = if side { scale } else { -scale };
    let edge = if side { w_min } else { w_max };
    let q0 = (edge / scale).round();
    if q0 == 0.0 {
        (scale, 0.0)
    } else {
        (edge / q0, edge)
    }
}

/// Paramètres bf16 finis `(bits scale, bits biais, codes, erreur)` d'un groupe.
fn candidate(
    values: &[f32],
    scale: f32,
    bias: f32,
    levels: f64,
    q: &mut [u32],
) -> Option<(u16, u16, f64)> {
    let (s_bits, b_bits) = (bf16_bits(scale), bf16_bits(bias));
    let (s, b) = (bf16_value(s_bits), bf16_value(b_bits));
    if !(s.is_finite() && b.is_finite() && (levels as f32 * s + b).is_finite()) {
        return None;
    }
    codes(values, f64::from(s), f64::from(b), levels, q);
    Some((s_bits, b_bits, group_error(values, q, s, b)))
}

/// Médiane haute de |w| d'une ligne (élément de rang `len / 2`).
fn row_abs_median(row: &[f32], scratch: &mut Vec<f32>) -> f32 {
    scratch.clear();
    scratch.extend(row.iter().map(|v| v.abs()));
    let mid = scratch.len() / 2;
    *scratch.select_nth_unstable_by(mid, f32::total_cmp).1
}

/// Groupe à super-poids : meilleur candidat parmi la grille MLX et des
/// moindres carrés à biais fixé sur le bord de plus grande valeur absolue
/// (code 0, alternance codes ↔ échelle seule), tous deux exacts sur ce bord.
/// `None` si aucun candidat n'est fini ou ne restitue le bord à l'identique :
/// l'appelant garde alors la sélection ordinaire. Partagé par les variantes
/// de moindres carrés (pondérée comprise) pour une même règle.
fn exact_edge_fit(
    values: &[f32],
    levels: f64,
    q: &mut [u32],
    best_q: &mut [u32],
) -> Option<(u16, u16, f64)> {
    let (ms, mb) = mlx_params(values, levels as f32);
    let mut best = candidate(values, ms, mb, levels, best_q)?;
    let bias = bf16_value(best.1);
    let mut scale = f64::from(bf16_value(best.0));
    for _ in 0..ITERS {
        codes(values, scale, f64::from(bias), levels, q);
        let (mut sqq, mut sqw) = (0.0, 0.0);
        for (&code, &value) in q.iter().zip(values) {
            let code = f64::from(code);
            sqq += code * code;
            sqw += code * (f64::from(value) - f64::from(bias));
        }
        let s1 = sqw / sqq;
        if sqq > 0.0 && s1 != 0.0 && s1.is_finite() {
            scale = s1;
        }
    }
    if let Some(cand) = candidate(values, scale as f32, bias, levels, q).filter(|c| c.2 < best.2) {
        best = cand;
        best_q.copy_from_slice(q);
    }
    let peak = values.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
    let (scale, bias) = (bf16_value(best.0), bf16_value(best.1));
    let exact = values
        .iter()
        .zip(&*best_q)
        .any(|(&v, &code)| v.abs() == peak && code as f32 * scale + bias == v);
    exact.then_some(best)
}

/// Quantifie `data` (`[rows, cols]`) en affine u4/u8 gs64 : meilleur des
/// candidats décrits en tête de module, groupe par groupe. Mêmes sorties que
/// [`quantize_affine_bitstream_bf16`] ; l'erreur rapportée est celle de la
/// sortie retenue.
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
    quantize_ls_with_ratio(
        data,
        cols,
        group_size,
        bits,
        packed,
        scales,
        biases,
        SUPER_WEIGHT_RATIO,
    )
}

/// [`quantize_affine_ls_bf16`] au seuil `ratio` ; `f32::INFINITY` désactive
/// la règle du bord exact (sélection d'origine, comparée dans les tests).
#[allow(clippy::too_many_arguments)]
fn quantize_ls_with_ratio(
    data: &[f32],
    cols: usize,
    group_size: usize,
    bits: usize,
    packed: &mut [u32],
    scales: &mut [u16],
    biases: &mut [u16],
    ratio: f32,
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
    let mut best_q = vec![0_u32; group_size];
    let mut q = vec![0_u32; group_size];
    let mut error = QuantError::default();
    let (mut scratch, mut median) = (Vec::with_capacity(cols), 0.0_f32);
    for (index, values) in data.chunks_exact(group_size).enumerate() {
        let (row, group) = (index / groups, index % groups);
        if group == 0 {
            median = row_abs_median(&data[row * cols..(row + 1) * cols], &mut scratch);
        }
        let peak = values.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
        let word0 = row * (cols / lanes) + group * words_per_group;
        let words = &mut packed[word0..word0 + words_per_group];
        for (lane, code) in best_q.iter_mut().enumerate() {
            *code = (words[lane / lanes] >> ((lane % lanes) * bits)) & mask;
        }
        let mut best = (
            scales[index],
            biases[index],
            group_error(
                values,
                &best_q,
                bf16_value(scales[index]),
                bf16_value(biases[index]),
            ),
        );
        let minmax = best;
        // Comparaison fausse pour un seuil infini (∞ · 0 = NaN) : règle désactivée.
        let forced = if peak > ratio * median {
            exact_edge_fit(values, levels, &mut q, &mut best_q)
        } else {
            None
        };
        if let Some(fit) = forced {
            best = fit;
        } else {
            let mut keep = |cand: Option<(u16, u16, f64)>, q: &[u32], best_q: &mut [u32]| {
                if let Some(cand) = cand.filter(|cand| cand.2 < best.2) {
                    best = cand;
                    best_q.copy_from_slice(q);
                }
            };
            let (ms, mb) = mlx_params(values, levels as f32);
            keep(candidate(values, ms, mb, levels, &mut q), &q, &mut best_q);
            // Départ de l'alternance : la grille MLX arrondie bf16, comme opt_quant.
            let (mut s, mut b) = (
                f64::from(bf16_value(bf16_bits(ms))),
                f64::from(bf16_value(bf16_bits(mb))),
            );
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
            keep(
                candidate(values, s as f32, b as f32, levels, &mut q),
                &q,
                &mut best_q,
            );
        }
        if best != minmax {
            scales[index] = best.0;
            biases[index] = best.1;
            words.fill(0);
            for (lane, &code) in best_q.iter().enumerate() {
                words[lane / lanes] |= code << ((lane % lanes) * bits);
            }
        }
        error.sum_sq_err += best.2;
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

    fn quantize(data: &[f32], bits: usize) -> (Vec<u32>, Vec<u16>, Vec<u16>, QuantError) {
        let groups = data.len() / 64;
        let (mut p, mut s, mut b) = (
            vec![0; data.len() * bits / 32],
            vec![0; groups],
            vec![0; groups],
        );
        let error = quantize_affine_ls_bf16(data, data.len(), 64, bits, &mut p, &mut s, &mut b)
            .expect("invariant: groupe valide");
        (p, s, b, error)
    }

    /// Grilles exactes couvrant les codes extrêmes, paramètres bf16 exacts : l'optimum
    /// (erreur nulle) est unique à la symétrie `q ↔ L − q` près. Oracle
    /// indépendant : régression des valeurs sur les codes construits.
    #[test]
    fn recovers_exact_grids_like_regression_oracle() {
        for (scale, bias, bits) in [(-0.5_f32, 4.0_f32, 4), (0.375, -2.25, 4), (0.0625, -8.0, 8)] {
            let levels = (1_u32 << bits) - 1;
            // u4 : les 16 codes (7 premier avec 16) ; u8 : 0, 4, …, 248 puis 255.
            let q: Vec<f32> = (0..64_u32)
                .map(|i| match bits {
                    4 => (i * 7) % 16,
                    _ if i == 63 => levels,
                    _ => i * 4,
                } as f32)
                .collect();
            let data: Vec<f32> = q.iter().map(|&q| q * scale + bias).collect();
            let n = 64.0_f64;
            let (sq, sqq): (f64, f64) = q.iter().fold((0.0, 0.0), |(a, b), &v| {
                (a + f64::from(v), b + f64::from(v * v))
            });
            let (sw, sqw): (f64, f64) = q.iter().zip(&data).fold((0.0, 0.0), |(a, b), (&c, &w)| {
                (a + f64::from(w), b + f64::from(c * w))
            });
            let oracle = (n * sqw - sq * sw) / (n * sqq - sq * sq);
            assert_eq!(oracle as f32, scale, "oracle");
            let (p, s, b, error) = quantize(&data, bits);
            assert_eq!(error.sum_sq_err, 0.0, "{scale} {bias}");
            assert_eq!(dequant(&p, &s, &b, bits), data);
            let got = (bf16_value(s[0]), bf16_value(b[0]));
            let mirror = (-scale, bias + levels as f32 * scale);
            assert!(got == (scale, bias) || got == mirror, "{got:?}");
        }
    }

    /// Grilles creuses (codes extrêmes inutilisés) : ni le min-max bf16 ni la
    /// grille MLX ne sont exacts, les moindres carrés retrouvent l'erreur nulle.
    /// Oracle indépendant : la régression fermée sur les codes construits donne
    /// l'échelle au signe près (la solution trouvée peut être la grille miroir
    /// ou décalée, de même restitution).
    #[test]
    fn least_squares_reach_optimum_missed_by_both_minmax() {
        for (codes_used, scale, bias) in [
            (&[7_u32, 8, 10, 11, 12][..], -0.5_f32, -1.0_f32),
            (&[5, 7, 9, 10][..], -0.25, -1.0),
        ] {
            let q: Vec<f32> = (0..64)
                .map(|i| codes_used[i % codes_used.len()] as f32)
                .collect();
            let data: Vec<f32> = q.iter().map(|&q| q * scale + bias).collect();
            let n = 64.0_f64;
            let (mut sq, mut sqq, mut sw, mut sqw) = (0.0, 0.0, 0.0, 0.0);
            for (&c, &w) in q.iter().zip(&data) {
                let (c, w) = (f64::from(c), f64::from(w));
                (sq, sqq, sw, sqw) = (sq + c, sqq + c * c, sw + w, sqw + c * w);
            }
            let oracle = ((n * sqw - sq * sw) / (n * sqq - sq * sq)) as f32;
            assert_eq!(oracle, scale, "oracle");

            let (mut p0, mut s0, mut b0) = (vec![0; 8], vec![0; 1], vec![0; 1]);
            let mm = quantize_affine_bitstream_bf16(&data, 64, 64, 4, &mut p0, &mut s0, &mut b0)
                .expect("invariant: min-max valide");
            assert!(mm.sum_sq_err > 0.0, "min-max déjà exact");
            let (ms, mb) = mlx_params(&data, 15.0);
            let mlx = candidate(&data, ms, mb, 15.0, &mut [0; 64]).map_or(f64::INFINITY, |c| c.2);
            assert!(mlx > 0.0, "grille MLX déjà exacte");

            let (p, s, b, error) = quantize(&data, 4);
            assert_eq!(error.sum_sq_err, 0.0, "{codes_used:?}");
            assert_eq!(dequant(&p, &s, &b, 4), data);
            assert_eq!(bf16_value(s[0]).abs(), oracle.abs());
        }
    }

    /// Groupe constant : restitué exactement (codes nuls ou biais seul).
    #[test]
    fn constant_group_is_exact() {
        let (p, s, b, error) = quantize(&[2.0; 64], 4);
        assert_eq!(error.sum_sq_err, 0.0);
        assert_eq!(dequant(&p, &s, &b, 4), vec![2.0; 64]);
    }

    /// La grille MLX est celle de `affine_quantize` (bord de plus grande valeur
    /// absolue, échelle négative), valeurs calculées à la main.
    #[test]
    fn mlx_params_follow_metal_kernel() {
        let values: Vec<f32> = (0..64).map(|i| -3.0 + 0.5 * (i % 15) as f32).collect();
        // w_min = −3, w_max = 4 : bord 4, échelle −7/15, q0 = round(4 / −7/15) = −9.
        assert_eq!(mlx_params(&values, 15.0), (4.0 / -9.0, 4.0));
        // Tout positif : w_max part de 0 mais w_min reste le vrai minimum.
        let positive: Vec<f32> = (0..64).map(|i| 1.0 + (i % 4) as f32).collect();
        let (scale, bias) = mlx_params(&positive, 15.0);
        assert_eq!(bias, 4.0);
        assert_eq!(scale, 4.0 / (4.0_f32 / (-3.0 / 15.0)).round());
    }

    /// Jamais pire que les deux min-max, groupe par groupe, en u4 comme en u8.
    #[test]
    fn never_worse_than_either_minmax() {
        let data: Vec<f32> = (0..4 * 256)
            .map(|i| ((i as f32 * 0.37).sin() * 3.0).powi(3) + (i % 7) as f32 * 0.01)
            .collect();
        for bits in [4, 8] {
            let levels = (1_u32 << bits) - 1;
            let words = data.len() * bits / 32;
            let groups = data.len() / 64;
            let (mut p0, mut s0, mut b0) = (vec![0; words], vec![0; groups], vec![0; groups]);
            let mm =
                quantize_affine_bitstream_bf16(&data, 256, 64, bits, &mut p0, &mut s0, &mut b0)
                    .expect("invariant: min-max valide");
            let (p1, s1, b1, ls) = quantize(&data, bits);
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
                let mut q = vec![0; 64];
                let (ms, mb) = mlx_params(values, levels as f32);
                let mlx = candidate(values, ms, mb, f64::from(levels), &mut q)
                    .map_or(f64::INFINITY, |c| c.2);
                assert!(err(&d1) <= err(&d0), "{bits} bits, groupe {g}");
                assert!(err(&d1) <= mlx, "{bits} bits, groupe {g} contre MLX");
            }
            let total: f64 = data
                .iter()
                .zip(&d1)
                .map(|(&v, &x)| f64::from(v - x).powi(2))
                .sum();
            assert!((total - ls.sum_sq_err).abs() <= 1e-9 * total.max(1.0));
        }
    }

    /// Ligne pseudo-gaussienne déterministe (somme de 4 uniformes, LCG).
    fn gaussian_row(cols: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        let mut uniform = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            (state >> 40) as f32 / (1_u64 << 24) as f32 - 0.5
        };
        (0..cols)
            .map(|_| {
                let x: f32 = (0..4).map(|_| uniform()).sum();
                bf16_value(bf16_bits(x * 0.02))
            })
            .collect()
    }

    fn quantize_ratio(
        data: &[f32],
        cols: usize,
        bits: usize,
        ratio: f32,
    ) -> (Vec<u32>, Vec<u16>, Vec<u16>) {
        let groups = data.len() / 64;
        let (mut p, mut s, mut b) = (
            vec![0; data.len() * bits / 32],
            vec![0; groups],
            vec![0; groups],
        );
        quantize_ls_with_ratio(data, cols, 64, bits, &mut p, &mut s, &mut b, ratio)
            .expect("invariant: groupe valide");
        (p, s, b)
    }

    /// Super-poids (100 × la médiane, au milieu du groupe 1) : restitué à
    /// l'identique, et l'erreur du groupe reste au plus celle de la grille MLX.
    #[test]
    fn super_weight_edge_is_exact_and_not_worse_than_mlx() {
        for (bits, sign) in [(4, 1.0_f32), (4, -1.0), (8, 1.0), (8, -1.0)] {
            let mut row = gaussian_row(256, 7);
            let lane = 64 + 37;
            row[lane] = sign * bf16_value(bf16_bits(100.0 * 0.0137));
            let mut scratch = Vec::new();
            assert!(row[lane].abs() > SUPER_WEIGHT_RATIO * row_abs_median(&row, &mut scratch));
            let (p, s, b) = quantize_ratio(&row, 256, bits, SUPER_WEIGHT_RATIO);
            let d = dequant(&p, &s, &b, bits);
            assert_eq!(d[lane], row[lane], "{bits} bits, signe {sign}");
            let group = &row[64..128];
            let err: f64 = group
                .iter()
                .zip(&d[64..128])
                .map(|(&v, &x)| f64::from(v - x).powi(2))
                .sum();
            let levels = f64::from((1_u32 << bits) - 1);
            let (ms, mb) = mlx_params(group, levels as f32);
            let mlx = candidate(group, ms, mb, levels, &mut [0; 64]).map_or(f64::INFINITY, |c| c.2);
            assert!(err <= mlx, "{bits} bits : {err} > MLX {mlx}");
        }
    }

    /// Groupes normaux : sortie identique au bit à la sélection d'origine
    /// (seuil infini), y compris les autres groupes d'une ligne à super-poids.
    #[test]
    fn normal_groups_bit_identical_to_original_selection() {
        let mut data: Vec<f32> = (0..4).flat_map(|r| gaussian_row(512, 11 + r)).collect();
        for bits in [4, 8] {
            assert_eq!(
                quantize_ratio(&data, 512, bits, SUPER_WEIGHT_RATIO),
                quantize_ratio(&data, 512, bits, f32::INFINITY),
                "{bits} bits"
            );
        }
        data[2 * 512 + 3 * 64 + 5] = 1.5;
        for bits in [4, 8] {
            let ruled = quantize_ratio(&data, 512, bits, SUPER_WEIGHT_RATIO);
            let plain = quantize_ratio(&data, 512, bits, f32::INFINITY);
            let words = 64 * bits / 32;
            let flagged = 2 * 8 + 3;
            for g in (0..32).filter(|&g| g != flagged) {
                assert_eq!(ruled.1[g], plain.1[g], "{bits} bits, groupe {g}");
                assert_eq!(ruled.2[g], plain.2[g], "{bits} bits, groupe {g}");
                assert_eq!(
                    ruled.0[g * words..(g + 1) * words],
                    plain.0[g * words..(g + 1) * words]
                );
            }
        }
    }

    /// `--minmax` ne passe pas par la règle : sur un groupe à super-poids, le
    /// min-max bf16 garde son contrat de couverture et la LS en diffère.
    #[test]
    fn minmax_untouched_by_super_weight_rule() {
        let mut row = gaussian_row(64, 3);
        row[10] = 1.5;
        let (mut p, mut s, mut b) = (vec![0; 8], vec![0; 1], vec![0; 1]);
        quantize_affine_bitstream_bf16(&row, 64, 64, 4, &mut p, &mut s, &mut b)
            .expect("invariant: min-max valide");
        let (w_min, w_max) = row
            .iter()
            .fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
        // Contrat du min-max bf16 (arrondis dirigés) : la grille couvre [min, max].
        assert!(bf16_value(b[0]) <= w_min && bf16_value(b[0]) + 15.0 * bf16_value(s[0]) >= w_max);
        assert_ne!(quantize_ratio(&row, 64, 4, SUPER_WEIGHT_RATIO), (p, s, b));
    }
}
