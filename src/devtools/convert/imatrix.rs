//! Moindres carrés pondérés par une imatrix (importance diagonale `E[x²]` par
//! colonne d'entrée), toutes largeurs affines gs64 (2 à 8 bits).
//!
//! Format : celui de `tools/quant/imatrix.py`, un safetensors de vecteurs f32
//! nommés comme les modules mlx-lm (`language_model.model.layers.N.….q_proj`,
//! `language_model.lm_head`). Par groupe, la procédure est celle de
//! [`super::least_squares`] avec l'erreur `Σ_j E[x_j²]·(w_j − ŵ_j)²` : min-max
//! bf16, grille MLX, 8 alternances codes ↔ équations normales pondérées, grille
//! à zéro, et le meilleur des quatre en erreur pondérée est gardé. Avec des poids tous égaux
//! à 1, chaque opération coïncide au bit près avec la version non pondérée
//! (multiplications par 1, même ordre de sommation).

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use safetensors::{Dtype, SafeTensors};

use super::least_squares::{codes, mlx_params, zero_grid};
use super::plan::io_err;
use super::sha256_bytes;
use crate::assets::is_mtp_key;
use crate::quantization::{bf16_bits, bf16_value, quantize_affine_bitstream_bf16, QuantError};
use crate::{InferError, Result};

/// Itérations codes ↔ paramètres, comme `least_squares`.
const ITERS: usize = 8;

/// Vecteurs d'importance par module, en f64 (sommes du fit en f64).
pub(crate) struct Imatrix {
    vectors: HashMap<String, Arc<[f64]>>,
    /// SHA-256 du fichier, consigné dans l'identité de la conversion.
    pub(crate) sha256: String,
}

impl Imatrix {
    /// Lit et valide l'imatrix (vecteurs f32 finis, positifs, non tous nuls).
    ///
    /// # Errors
    ///
    /// Fichier illisible, safetensors invalide ou vecteur invalide.
    pub(crate) fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).map_err(|e| io_err(path, e))?;
        let file = SafeTensors::deserialize(&bytes)
            .map_err(|e| InferError::Config(format!("imatrix {}: {e}", path.display())))?;
        let mut vectors = HashMap::new();
        for (name, view) in file.tensors() {
            if view.dtype() != Dtype::F32 || view.shape().len() != 1 {
                return Err(InferError::Config(format!(
                    "imatrix {name}: vecteur f32 attendu, {:?} {:?}",
                    view.dtype(),
                    view.shape()
                )));
            }
            let values: Vec<f64> = view
                .data()
                .chunks_exact(4)
                .map(|b| f64::from(f32::from_le_bytes([b[0], b[1], b[2], b[3]])))
                .collect();
            if values.iter().any(|v| !v.is_finite() || *v < 0.0) || values.iter().all(|v| *v == 0.0)
            {
                return Err(InferError::Config(format!(
                    "imatrix {name}: importances finies, positives et non toutes nulles attendues"
                )));
            }
            vectors.insert(name, values.into());
        }
        // Empreinte des octets décodés : un remplacement du fichier après la
        // lecture ne peut pas désaccorder vecteurs et manifeste de reprise.
        Ok(Self {
            vectors,
            sha256: sha256_bytes(&bytes),
        })
    }

    /// Importance du tenseur HF `name` (`[rows, cols]`). `None` pour
    /// l'embedding (entrée one-hot, fit uniforme comme `imatrix.py`) et la
    /// tête MTP (non capturée) ; tout autre linéaire sans vecteur est refusé.
    ///
    /// # Errors
    ///
    /// Vecteur absent ou de longueur différente de `cols`.
    pub(crate) fn for_tensor(&self, name: &str, cols: usize) -> Result<Option<Arc<[f64]>>> {
        if name.contains("embed_tokens") || is_mtp_key(name) {
            return Ok(None);
        }
        let base = name.strip_suffix(".weight").unwrap_or(name);
        let mlx = match base {
            "lm_head" => "language_model.lm_head".to_string(),
            _ => match base.strip_prefix("model.language_model.") {
                Some(rest) => format!("language_model.model.{rest}"),
                None => base.to_string(),
            },
        };
        let vector = [mlx.as_str(), base]
            .iter()
            .find_map(|key| self.vectors.get(*key))
            .ok_or_else(|| InferError::Config(format!("imatrix absente : {name} ({mlx})")))?;
        if vector.len() != cols {
            return Err(InferError::Dimension(format!(
                "imatrix {mlx}: {} colonnes, {cols} attendues",
                vector.len()
            )));
        }
        Ok(Some(Arc::clone(vector)))
    }
}

/// Lit le code `lane` d'un groupe emballé en flux little-endian.
pub(super) fn read_code(words: &[u32], lane: usize, bits: usize) -> u32 {
    let mask = (1_u32 << bits) - 1;
    let (word, shift) = ((lane * bits) / 32, (lane * bits) % 32);
    let mut code = words[word] >> shift;
    if shift + bits > 32 {
        code |= words[word + 1] << (32 - shift);
    }
    code & mask
}

fn write_codes(words: &mut [u32], codes: &[u32], bits: usize) {
    words.fill(0);
    for (lane, &code) in codes.iter().enumerate() {
        let (word, shift) = ((lane * bits) / 32, (lane * bits) % 32);
        words[word] |= code << shift;
        if shift + bits > 32 {
            words[word + 1] |= code >> (32 - shift);
        }
    }
}

/// `Σ w·(v − (q·s + b))²`, restitution f32 comme le loader ; `w = 1` sans poids.
fn weighted_error(values: &[f32], codes: &[u32], weight: Option<&[f64]>, s: f32, b: f32) -> f64 {
    values
        .iter()
        .zip(codes)
        .enumerate()
        .map(|(j, (&value, &code))| {
            let err = f64::from(value - (code as f32 * s + b)).powi(2);
            weight.map_or(err, |w| w[j] * err)
        })
        .sum()
}

/// Paramètres bf16 finis `(scale, biais, erreur pondérée)` et codes dans `q`.
fn candidate(
    values: &[f32],
    weight: Option<&[f64]>,
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
    Some((s_bits, b_bits, weighted_error(values, q, weight, s, b)))
}

/// Affine gs64 `bits` ∈ {2, 3, 4, 5, 6, 8} aux moindres carrés pondérés par
/// `weight` (une importance par colonne de `data` `[rows, cols]`, `None` =
/// uniforme). Sorties au format de [`quantize_affine_bitstream_bf16`] ;
/// l'erreur rapportée est l'erreur **non pondérée** de la sortie retenue.
///
/// # Errors
///
/// Erreurs du min-max, ou vecteur d'importance de longueur différente de `cols`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn quantize_affine_wls_bf16(
    data: &[f32],
    cols: usize,
    group_size: usize,
    bits: usize,
    weight: Option<&[f64]>,
    packed: &mut [u32],
    scales: &mut [u16],
    biases: &mut [u16],
) -> Result<QuantError> {
    if weight.is_some_and(|w| w.len() != cols) {
        return Err(InferError::Dimension(format!(
            "imatrix : {} importances pour {cols} colonnes",
            weight.map_or(0, <[f64]>::len)
        )));
    }
    quantize_affine_bitstream_bf16(data, cols, group_size, bits, packed, scales, biases)?;
    let levels = f64::from((1_u32 << bits) - 1);
    let words_per_group = group_size * bits / 32;
    let groups = cols / group_size;
    let mut best_q = vec![0_u32; group_size];
    let mut q = vec![0_u32; group_size];
    let mut error = QuantError::default();
    for (index, values) in data.chunks_exact(group_size).enumerate() {
        let group = index % groups;
        let w = weight.map(|w| &w[group * group_size..(group + 1) * group_size]);
        // gs64 : 2·bits mots par groupe, chaque groupe commence sur un mot.
        let words = &mut packed[index * words_per_group..(index + 1) * words_per_group];
        for (lane, code) in best_q.iter_mut().enumerate() {
            *code = read_code(words, lane, bits);
        }
        let mut best = (
            scales[index],
            biases[index],
            weighted_error(
                values,
                &best_q,
                w,
                bf16_value(scales[index]),
                bf16_value(biases[index]),
            ),
        );
        let minmax = best;
        let mut keep = |cand: Option<(u16, u16, f64)>, q: &[u32], best_q: &mut [u32]| {
            if let Some(cand) = cand.filter(|cand| cand.2 < best.2) {
                best = cand;
                best_q.copy_from_slice(q);
            }
        };
        let (ms, mb) = mlx_params(values, levels as f32);
        keep(
            candidate(values, w, ms, mb, levels, &mut q),
            &q,
            &mut best_q,
        );
        let (mut s, mut b) = (
            f64::from(bf16_value(bf16_bits(ms))),
            f64::from(bf16_value(bf16_bits(mb))),
        );
        for _ in 0..ITERS {
            codes(values, s, b, levels, &mut q);
            let (mut n, mut sq, mut sqq, mut sw, mut sqw) = (0.0, 0.0, 0.0, 0.0, 0.0);
            for (j, (&code, &value)) in q.iter().zip(values).enumerate() {
                let (code, value) = (f64::from(code), f64::from(value));
                let a = w.map_or(1.0, |w| w[j]);
                n += a;
                sq += a * code;
                sqq += a * code * code;
                sw += a * value;
                sqw += a * code * value;
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
            candidate(values, w, s as f32, b as f32, levels, &mut q),
            &q,
            &mut best_q,
        );
        let starts = [(s, b), (f64::from(ms), f64::from(mb))];
        let cand = zero_grid(values, w, starts, levels, &mut q, |s, b, q| {
            candidate(values, w, s, b, levels, q)
        });
        keep(cand, &q, &mut best_q);
        if best != minmax {
            scales[index] = best.0;
            biases[index] = best.1;
            write_codes(words, &best_q, bits);
        }
        error.sum_sq_err += weighted_error(
            values,
            &best_q,
            None,
            bf16_value(best.0),
            bf16_value(best.1),
        );
        error.sum_sq += values.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>();
    }
    Ok(error)
}

#[cfg(test)]
mod tests {
    use super::super::least_squares::quantize_affine_ls_bf16;
    use super::*;

    type Out = (Vec<u32>, Vec<u16>, Vec<u16>, QuantError);

    fn data(rows: usize, cols: usize) -> Vec<f32> {
        (0..rows * cols)
            .map(|i| ((i as f32 * 0.37).sin() * 3.0).powi(3) + (i % 7) as f32 * 0.01)
            .collect()
    }

    fn run(data: &[f32], cols: usize, bits: usize, weight: Option<&[f64]>, ls: bool) -> Out {
        let groups = data.len() / 64;
        let (mut p, mut s, mut b) = (
            vec![0; data.len() * bits / 32],
            vec![0; groups],
            vec![0; groups],
        );
        let error = if ls {
            quantize_affine_ls_bf16(data, cols, 64, bits, &mut p, &mut s, &mut b)
        } else {
            quantize_affine_wls_bf16(data, cols, 64, bits, weight, &mut p, &mut s, &mut b)
        }
        .expect("invariant: entrée valide");
        (p, s, b, error)
    }

    fn dequant(out: &Out, bits: usize) -> Vec<f32> {
        let words = 2 * bits;
        (0..out.1.len() * 64)
            .map(|i| {
                let g = i / 64;
                let code = read_code(&out.0[g * words..(g + 1) * words], i % 64, bits);
                code as f32 * bf16_value(out.1[g]) + bf16_value(out.2[g])
            })
            .collect()
    }

    /// Poids uniformes (explicites ou absents) : sorties identiques au bit aux
    /// moindres carrés non pondérés, erreur comprise.
    #[test]
    fn uniform_weight_matches_unweighted_bit_for_bit() {
        let (rows, cols) = (4, 256);
        let values = data(rows, cols);
        let ones = vec![1.0; cols];
        for bits in [4, 8] {
            let reference = run(&values, cols, bits, None, true);
            assert_eq!(run(&values, cols, bits, Some(&ones), false), reference);
            assert_eq!(run(&values, cols, bits, None, false), reference);
        }
    }

    /// En erreur pondérée, jamais pire que le min-max bf16, groupe par groupe,
    /// et meilleur au total que le fit non pondéré ; importances nulles comprises.
    #[test]
    fn weighted_never_worse_than_minmax() {
        let (rows, cols) = (4, 256);
        let values = data(rows, cols);
        let weight: Vec<f64> = (0..cols)
            .map(|j| {
                if j % 11 == 0 {
                    0.0
                } else {
                    (j as f64 * 0.7).cos().powi(2) * 50.0 + 0.01
                }
            })
            .collect();
        for bits in [2, 3, 4, 5, 6, 8] {
            let groups = values.len() / 64;
            let (mut p, mut s, mut b) = (
                vec![0; values.len() * bits / 32],
                vec![0; groups],
                vec![0; groups],
            );
            quantize_affine_bitstream_bf16(&values, cols, 64, bits, &mut p, &mut s, &mut b)
                .expect("invariant: min-max valide");
            let minmax = dequant(&(p, s, b, QuantError::default()), bits);
            let fitted = dequant(&run(&values, cols, bits, Some(&weight), false), bits);
            let uniform = dequant(&run(&values, cols, bits, None, false), bits);
            let err = |d: &[f32], range: std::ops::Range<usize>| -> f64 {
                range
                    .map(|i| weight[i % cols] * f64::from(values[i] - d[i]).powi(2))
                    .sum()
            };
            for g in 0..groups {
                let range = g * 64..(g + 1) * 64;
                assert!(
                    err(&fitted, range.clone()) <= err(&minmax, range),
                    "{bits} bits, groupe {g}"
                );
            }
            let all = 0..values.len();
            assert!(
                err(&fitted, all.clone()) < err(&uniform, all),
                "{bits} bits"
            );
        }
    }

    /// L'emballage générique relit exactement ce qu'il écrit, mots chevauchants compris.
    #[test]
    fn bitstream_roundtrip() {
        for bits in [2, 3, 4, 5, 6, 8] {
            let codes: Vec<u32> = (0..64).map(|i| (i * 7 + 3) % (1 << bits)).collect();
            let mut words = vec![0; 2 * bits];
            write_codes(&mut words, &codes, bits);
            let back: Vec<u32> = (0..64).map(|i| read_code(&words, i, bits)).collect();
            assert_eq!(back, codes, "{bits} bits");
        }
    }

    #[test]
    fn rejects_wrong_weight_length() {
        let values = data(1, 128);
        let (mut p, mut s, mut b) = (vec![0; 16], vec![0; 2], vec![0; 2]);
        assert!(quantize_affine_wls_bf16(
            &values,
            128,
            64,
            4,
            Some(&[1.0; 64]),
            &mut p,
            &mut s,
            &mut b
        )
        .is_err());
    }
}
