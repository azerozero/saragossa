//! Audit d'erreur par colonne d'entrée (`--audit-report`) et signature « poids
//! seuls » des colonnes à |w| anormalement petit.
//!
//! Avec une imatrix, l'erreur d'une colonne j est `E[x_j²]·Σ_lignes δ²` : une
//! colonne qui en porte plus de [`HOT_SHARE`] est signalée (Mistral-7B v0.3,
//! `layers.1.mlp.down_proj`, colonne 7310 : 89 % de l'erreur du module). Sans
//! imatrix, la signature est la colonne dont |w| moyen est inférieur à
//! [`SMALL_RATIO`] fois la médiane des colonnes (7310 : 0,07).

use serde::Serialize;

use super::imatrix::read_code;
use crate::quantization::bf16_value;

/// Part de l'erreur d'un tenseur au-delà de laquelle une colonne est signalée.
pub(crate) const HOT_SHARE: f64 = 0.1;
/// Colonne « petite » : |w| moyen sous cette fraction de la médiane des colonnes.
pub(crate) const SMALL_RATIO: f64 = 0.1;
/// Au-delà de ce nombre de petites colonnes, c'est un sous-espace mort (q/k des
/// couches 0-1 de Mistral : ~1000 colonnes, E[x²] ≈ 1e-10), pas une
/// super-activation isolée : pas d'importance fictive.
// ponytail: seuil fixe, à remplacer par une imatrix quand un corpus existe.
const MAX_SMALL: usize = 16;
/// Lignes minimales d'un bloc pour une moyenne par colonne stable.
const MIN_ROWS: usize = 256;
/// Importance fictive d'une petite colonne sans imatrix (1 ailleurs) : le fit
/// de ses groupes privilégie une restitution proche de son poids quasi nul.
const PROXY: f64 = 1.0e4;

fn abs_sums(data: &[f32], cols: usize, out: &mut [f64]) {
    for row in data.chunks_exact(cols) {
        for (sum, &value) in out.iter_mut().zip(row) {
            *sum += f64::from(value.abs());
        }
    }
}

/// Petites colonnes `(colonne, |w| moyen / médiane)`, de la plus petite à la
/// plus grande ; médiane haute des moyennes par colonne.
pub(crate) fn small_columns(abs_sum: &[f64]) -> Vec<(usize, f64)> {
    let mut sorted = abs_sum.to_vec();
    sorted.sort_by(f64::total_cmp);
    let median = sorted[sorted.len() / 2];
    if median <= 0.0 {
        return Vec::new();
    }
    let mut small: Vec<(usize, f64)> = abs_sum
        .iter()
        .enumerate()
        .map(|(j, &sum)| (j, sum / median))
        .filter(|&(_, ratio)| ratio < SMALL_RATIO)
        .collect();
    small.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    small
}

/// Importance fictive d'un bloc `[rows, cols]` sans imatrix : [`PROXY`] sur
/// ses petites colonnes, 1 ailleurs ; `None` sans petite colonne, avec un
/// sous-espace mort ou trop peu de lignes (groupes inchangés).
pub(crate) fn proxy_importance(data: &[f32], cols: usize) -> Option<Vec<f64>> {
    if data.len() / cols < MIN_ROWS {
        return None;
    }
    let mut sums = vec![0.0; cols];
    abs_sums(data, cols, &mut sums);
    let small = small_columns(&sums);
    if small.is_empty() || small.len() > MAX_SMALL {
        return None;
    }
    let mut weight = vec![1.0; cols];
    for (j, _) in small {
        weight[j] = PROXY;
    }
    Some(weight)
}

#[derive(Debug, Serialize)]
pub(crate) struct HotColumn {
    pub(crate) col: usize,
    pub(crate) share: f64,
    pub(crate) error: f64,
    /// E[x²] de la colonne (imatrix).
    pub(crate) importance: Option<f64>,
}

#[derive(Debug, Serialize)]
pub(crate) struct SmallColumn {
    pub(crate) col: usize,
    /// |w| moyen de la colonne / médiane des colonnes.
    pub(crate) ratio: f64,
}

/// Entrée du rapport pour un tenseur qui a au moins une colonne signalée.
#[derive(Debug, Serialize)]
pub(crate) struct TensorAudit {
    pub(crate) tensor: String,
    pub(crate) bits: usize,
    /// Erreur totale `Σ_j a_j·Σ δ²` (a = imatrix, sinon 1).
    pub(crate) error: f64,
    pub(crate) hot_columns: Vec<HotColumn>,
    /// Les [`MAX_SMALL`] plus petites ; `small_count` les compte toutes.
    pub(crate) small_columns: Vec<SmallColumn>,
    pub(crate) small_count: usize,
}

/// Accumulateur par colonne d'un tenseur quantifié bloc de lignes par bloc.
pub(crate) struct ColumnAudit {
    cols: usize,
    bits: usize,
    error: Vec<f64>,
    abs_sum: Vec<f64>,
}

impl ColumnAudit {
    pub(crate) fn new(cols: usize, bits: usize) -> Self {
        Self {
            cols,
            bits,
            error: vec![0.0; cols],
            abs_sum: vec![0.0; cols],
        }
    }

    /// Ajoute un bloc `data` `[rows, cols]` et sa sortie affine gs64.
    pub(crate) fn add(
        &mut self,
        data: &[f32],
        packed: &[u32],
        scales: &[u16],
        biases: &[u16],
        importance: Option<&[f64]>,
    ) {
        const GROUP: usize = 64;
        let words = 2 * self.bits;
        abs_sums(data, self.cols, &mut self.abs_sum);
        for (index, values) in data.chunks_exact(GROUP).enumerate() {
            let col0 = (index % (self.cols / GROUP)) * GROUP;
            let group = &packed[index * words..(index + 1) * words];
            let (s, b) = (bf16_value(scales[index]), bf16_value(biases[index]));
            for (lane, &value) in values.iter().enumerate() {
                let restored = read_code(group, lane, self.bits) as f32 * s + b;
                let j = col0 + lane;
                let a = importance.map_or(1.0, |w| w[j]);
                self.error[j] += a * f64::from(value - restored).powi(2);
            }
        }
    }

    /// Entrée du rapport, `None` si aucune colonne n'est signalée.
    pub(crate) fn finish(self, tensor: &str, importance: Option<&[f64]>) -> Option<TensorAudit> {
        let total: f64 = self.error.iter().sum();
        let hot_columns: Vec<HotColumn> = self
            .error
            .iter()
            .enumerate()
            .filter(|&(_, &e)| total > 0.0 && e / total > HOT_SHARE)
            .map(|(col, &error)| HotColumn {
                col,
                share: error / total,
                error,
                importance: importance.map(|w| w[col]),
            })
            .collect();
        let small = small_columns(&self.abs_sum);
        if hot_columns.is_empty() && small.is_empty() {
            return None;
        }
        Some(TensorAudit {
            tensor: tensor.to_string(),
            bits: self.bits,
            error: total,
            hot_columns,
            small_count: small.len(),
            small_columns: small
                .into_iter()
                .take(MAX_SMALL)
                .map(|(col, ratio)| SmallColumn { col, ratio })
                .collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::imatrix::quantize_affine_wls_bf16;
    use super::super::least_squares::quantize_affine_ls_bf16;
    use super::*;

    const ROWS: usize = 512;
    const COLS: usize = 256;
    /// Colonne à poids quasi nuls (super-activation de Mistral, en réduit).
    const HOT: usize = 78;

    fn tensor() -> Vec<f32> {
        (0..ROWS * COLS)
            .map(|i| {
                let v = ((i as f32 * 0.37).sin() * 3.0).powi(3) * 0.01 + (i % 7) as f32 * 1.0e-4;
                if i % COLS == HOT {
                    v * 0.01
                } else {
                    v
                }
            })
            .collect()
    }

    fn importance() -> Vec<f64> {
        (0..COLS)
            .map(|j| if j == HOT { 2290.0 } else { 2.0e-5 })
            .collect()
    }

    type Out = (Vec<u32>, Vec<u16>, Vec<u16>);

    fn quantize(data: &[f32], weight: Option<&[f64]>, ls: bool) -> Out {
        let groups = data.len() / 64;
        let (mut p, mut s, mut b) = (vec![0; data.len() / 8], vec![0; groups], vec![0; groups]);
        if ls {
            quantize_affine_ls_bf16(data, COLS, 64, 4, &mut p, &mut s, &mut b)
        } else {
            quantize_affine_wls_bf16(data, COLS, 64, 4, weight, &mut p, &mut s, &mut b)
        }
        .expect("invariant: entrée valide");
        (p, s, b)
    }

    fn audit(data: &[f32], out: &Out, weight: Option<&[f64]>) -> (Vec<f64>, Option<TensorAudit>) {
        let mut columns = ColumnAudit::new(COLS, 4);
        // Deux blocs de lignes, comme `write_shard`.
        let half = ROWS / 2;
        for (r, rows) in [(0, half), (half, half)] {
            let (g0, g1) = (r * COLS / 64, (r + rows) * COLS / 64);
            columns.add(
                &data[r * COLS..(r + rows) * COLS],
                &out.0[g0 * 8..g1 * 8],
                &out.1[g0..g1],
                &out.2[g0..g1],
                weight,
            );
        }
        let error = columns.error.clone();
        (error, columns.finish("t", weight))
    }

    /// Avec imatrix, la colonne quasi nulle concentre l'erreur pondérée du fit
    /// non pondéré et est signalée ; la signature poids seuls la trouve aussi.
    #[test]
    fn audit_flags_hot_and_small_column() {
        let data = tensor();
        let weight = importance();
        let plain = quantize(&data, None, true);
        let (_, report) = audit(&data, &plain, Some(&weight));
        let report = report.expect("invariant: colonne signalée");
        assert_eq!(report.hot_columns.len(), 1, "{report:?}");
        assert_eq!(report.hot_columns[0].col, HOT);
        assert_eq!(report.hot_columns[0].importance, Some(2290.0));
        assert_eq!(report.small_count, 1, "{report:?}");
        assert_eq!(report.small_columns[0].col, HOT);
        let (_, unweighted) = audit(&data, &plain, None);
        let unweighted = unweighted.expect("invariant: signature poids seuls");
        assert!(unweighted.hot_columns.is_empty());
        assert_eq!(unweighted.small_columns[0].col, HOT);
        // Tenseur ordinaire : rien à signaler.
        let normal: Vec<f32> = (0..ROWS * COLS)
            .map(|i| ((i as f32 * 0.37).sin() * 3.0).powi(3))
            .collect();
        assert!(audit(&normal, &quantize(&normal, None, true), None)
            .1
            .is_none());
        assert!(proxy_importance(&normal, COLS).is_none());
    }

    /// La grille à zéro réduit l'erreur pondérée de la colonne chaude, avec
    /// l'imatrix comme avec l'importance fictive de la signature poids seuls.
    #[test]
    fn zero_grid_cuts_hot_column_error() {
        let data = tensor();
        let weight = importance();
        let (plain, _) = audit(&data, &quantize(&data, None, true), Some(&weight));
        let (fitted, _) = audit(&data, &quantize(&data, Some(&weight), false), Some(&weight));
        let proxy = proxy_importance(&data, COLS).expect("invariant: petite colonne");
        assert_eq!(proxy.iter().filter(|&&a| a != 1.0).count(), 1);
        assert_eq!(proxy[HOT], PROXY);
        let (blind, _) = audit(&data, &quantize(&data, Some(&proxy), false), Some(&weight));
        let total = |e: &[f64]| e.iter().sum::<f64>();
        assert!(
            fitted[HOT] < plain[HOT] / 10.0,
            "{} {}",
            fitted[HOT],
            plain[HOT]
        );
        assert!(
            blind[HOT] < plain[HOT] / 10.0,
            "{} {}",
            blind[HOT],
            plain[HOT]
        );
        assert!(total(&fitted) < total(&plain) && total(&blind) < total(&plain));
        // Trop peu de lignes : pas d'importance fictive.
        assert!(proxy_importance(&data[..COLS * 64], COLS).is_none());
    }
}
