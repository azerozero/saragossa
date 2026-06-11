//! Tenseur CPU dense en f32 et opérations de base.

use crate::error::{InferError, Result};
use rayon::prelude::*;

const PARALLEL_MATMUL_OUTPUT_THRESHOLD: usize = 1024;
const PARALLEL_MATMUL_INNER_THRESHOLD: usize = 128;

#[derive(Clone, Debug, PartialEq)]
/// Stocke un tenseur dense CPU avec forme explicite.
pub struct Tensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

impl Tensor {
    /// Construit un tenseur depuis sa forme et ses données.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si la forme ne correspond pas aux données.
    pub fn from_vec(shape: impl Into<Vec<usize>>, data: Vec<f32>) -> Result<Self> {
        let shape = shape.into();
        let expected = element_count(&shape)?;
        if expected != data.len() {
            return Err(InferError::Shape(format!(
                "shape={shape:?} attend {expected} éléments, reçu {}",
                data.len()
            )));
        }
        Ok(Self { shape, data })
    }

    /// Construit une ligne `[1, n]` depuis des données contiguës.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le calcul de forme déborde.
    pub fn row(data: Vec<f32>) -> Result<Self> {
        Self::from_vec(vec![1, data.len()], data)
    }

    /// Construit un tenseur rempli de zéros.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le nombre d'éléments déborde.
    pub fn zeros(shape: impl Into<Vec<usize>>) -> Result<Self> {
        let shape = shape.into();
        let len = element_count(&shape)?;
        Ok(Self {
            shape,
            data: vec![0.0; len],
        })
    }

    #[must_use]
    /// Renvoie la forme du tenseur.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    #[must_use]
    /// Renvoie les données contiguës du tenseur.
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    #[must_use]
    /// Renvoie le nombre d'éléments.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[must_use]
    /// Indique si le tenseur ne contient aucun élément.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    #[must_use]
    /// Renvoie le rang du tenseur.
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Interprète la forme comme une matrice.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le tenseur n'est pas de rang 2.
    pub fn as_matrix(&self) -> Result<(usize, usize)> {
        match self.shape.as_slice() {
            [rows, cols] => Ok((*rows, *cols)),
            _ => Err(InferError::Dimension(format!(
                "tensor attendu rang 2, reçu {:?}",
                self.shape
            ))),
        }
    }

    /// Interprète le tenseur comme une ligne unique.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si la forme n'est pas `[1, n]`.
    pub fn as_row(&self) -> Result<&[f32]> {
        match self.shape.as_slice() {
            [1, _] => Ok(&self.data),
            _ => Err(InferError::Dimension(format!(
                "ligne attendue [1, n], reçu {:?}",
                self.shape
            ))),
        }
    }

    /// Renvoie une tranche sur une ligne de matrice.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le rang ou l'indice de ligne est invalide.
    pub fn row_slice(&self, row: usize) -> Result<&[f32]> {
        let (rows, cols) = self.as_matrix()?;
        if row >= rows {
            return Err(InferError::Dimension(format!(
                "row {row} hors bornes pour {rows} lignes"
            )));
        }
        let start = row * cols;
        Ok(&self.data[start..start + cols])
    }

    /// Renvoie la dernière ligne d'une matrice.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le tenseur n'est pas une matrice non vide.
    pub fn last_row(&self) -> Result<&[f32]> {
        let (rows, _) = self.as_matrix()?;
        if rows == 0 {
            return Err(InferError::Dimension(
                "last_row sur matrice sans ligne".to_string(),
            ));
        }
        self.row_slice(rows - 1)
    }

    /// Applique une transformation élément par élément.
    pub fn map(&self, mut f: impl FnMut(f32) -> f32) -> Self {
        Self {
            shape: self.shape.clone(),
            data: self.data.iter().copied().map(&mut f).collect(),
        }
    }

    /// Additionne deux tenseurs de même forme.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si les formes diffèrent.
    pub fn add(&self, rhs: &Self) -> Result<Self> {
        if self.shape != rhs.shape {
            return Err(InferError::Dimension(format!(
                "add shape gauche={:?}, droite={:?}",
                self.shape, rhs.shape
            )));
        }
        let data = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        Ok(Self {
            shape: self.shape.clone(),
            data,
        })
    }

    /// Multiplie deux tenseurs élément par élément.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si les formes diffèrent.
    pub fn mul_elementwise(&self, rhs: &Self) -> Result<Self> {
        if self.shape != rhs.shape {
            return Err(InferError::Dimension(format!(
                "mul shape gauche={:?}, droite={:?}",
                self.shape, rhs.shape
            )));
        }
        let data = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| a * b)
            .collect();
        Ok(Self {
            shape: self.shape.clone(),
            data,
        })
    }

    /// Ajoute un biais ligne par ligne.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si les formes ne sont pas compatibles.
    pub fn add_row_bias(&self, bias: &Self) -> Result<Self> {
        let (rows, cols) = self.as_matrix()?;
        let bias_data = match bias.shape.as_slice() {
            [n] if *n == cols => bias.data.as_slice(),
            [1, n] if *n == cols => bias.data.as_slice(),
            _ => {
                return Err(InferError::Dimension(format!(
                    "bias attendu [{cols}] ou [1,{cols}], reçu {:?}",
                    bias.shape
                )))
            }
        };
        let mut out = self.data.clone();
        for row in 0..rows {
            let start = row * cols;
            for col in 0..cols {
                out[start + col] += bias_data[col];
            }
        }
        Ok(Self {
            shape: self.shape.clone(),
            data: out,
        })
    }

    /// Multiplie par la transposée logique du tenseur de droite.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si les dimensions matricielles sont incompatibles.
    pub fn matmul_rhs_t(&self, rhs_out_in: &Self) -> Result<Self> {
        let (batch, in_dim) = self.as_matrix()?;
        let (out_dim, rhs_in_dim) = rhs_out_in.as_matrix()?;
        if in_dim != rhs_in_dim {
            return Err(InferError::Dimension(format!(
                "matmul x=[{batch},{in_dim}] rhs_t_source=[{out_dim},{rhs_in_dim}]"
            )));
        }

        let mut out = vec![0.0_f32; batch * out_dim];
        if should_parallelize_matmul(out.len(), in_dim) {
            out.par_iter_mut().enumerate().for_each(|(idx, value)| {
                let b = idx / out_dim;
                let o = idx % out_dim;
                let lhs = &self.data[b * in_dim..(b + 1) * in_dim];
                let rhs = &rhs_out_in.data[o * in_dim..(o + 1) * in_dim];
                *value = dot(lhs, rhs);
            });
        } else {
            for b in 0..batch {
                for o in 0..out_dim {
                    let lhs = &self.data[b * in_dim..(b + 1) * in_dim];
                    let rhs = &rhs_out_in.data[o * in_dim..(o + 1) * in_dim];
                    out[b * out_dim + o] = dot(lhs, rhs);
                }
            }
        }
        Self::from_vec(vec![batch, out_dim], out)
    }

    /// Consomme le tenseur et renvoie ses données.
    pub fn into_data(self) -> Vec<f32> {
        self.data
    }
}

fn should_parallelize_matmul(outputs: usize, inner: usize) -> bool {
    outputs >= PARALLEL_MATMUL_OUTPUT_THRESHOLD && inner >= PARALLEL_MATMUL_INNER_THRESHOLD
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right.iter())
        .map(|(left, right)| left * right)
        .sum()
}

fn element_count(shape: &[usize]) -> Result<usize> {
    if shape.is_empty() {
        return Err(InferError::Shape("shape vide".to_string()));
    }
    shape.iter().try_fold(1_usize, |acc, dim| {
        acc.checked_mul(*dim)
            .ok_or_else(|| InferError::Shape(format!("shape trop grande: {shape:?}")))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_bad_shape() {
        let err = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0])
            .expect_err("invariant: shape incohérente rejetée");
        assert!(matches!(err, InferError::Shape(_)));
    }

    #[test]
    fn computes_matmul_rhs_t() {
        let x = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("invariant: shape valide");
        let w = Tensor::from_vec(vec![2, 3], vec![1.0, 0.0, 1.0, 0.0, 1.0, 1.0])
            .expect("invariant: shape valide");
        let y = x
            .matmul_rhs_t(&w)
            .expect("invariant: dimensions compatibles");
        assert_eq!(y.shape(), &[2, 2]);
        assert_eq!(y.data(), &[4.0, 5.0, 10.0, 11.0]);
    }

    #[test]
    fn multiplies_elementwise() {
        let a = Tensor::from_vec(vec![1, 3], vec![1.0, 2.0, 3.0]).expect("invariant: shape valide");
        let b = Tensor::from_vec(vec![1, 3], vec![4.0, 5.0, 6.0]).expect("invariant: shape valide");
        let out = a
            .mul_elementwise(&b)
            .expect("invariant: shapes compatibles");
        assert_eq!(out.data(), &[4.0, 10.0, 18.0]);
    }
}
