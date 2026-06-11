//! Rotation RoPE appliquée aux têtes d'attention.

use crate::{InferError, Result, Tensor};

pub fn apply_rope(x: &Tensor, base_theta: f32) -> Result<Tensor> {
    let (seq, dim) = x.as_matrix()?;
    if dim % 2 != 0 {
        return Err(InferError::Dimension(format!(
            "RoPE attend une dimension paire, reçu {dim}"
        )));
    }
    if base_theta <= 0.0 {
        return Err(InferError::Dimension(format!(
            "RoPE base_theta invalide: {base_theta}"
        )));
    }

    let half = dim / 2;
    let mut out = Vec::with_capacity(x.len());
    for pos in 0..seq {
        let row = x.row_slice(pos)?;
        let mut rotated = vec![0.0_f32; dim];
        for pair in 0..half {
            let even = row[2 * pair];
            let odd = row[2 * pair + 1];
            let exponent = (2 * pair) as f32 / dim as f32;
            let angle = pos as f32 / base_theta.powf(exponent);
            let cos = angle.cos();
            let sin = angle.sin();
            rotated[2 * pair] = even * cos - odd * sin;
            rotated[2 * pair + 1] = even * sin + odd * cos;
        }
        out.extend(rotated);
    }
    Tensor::from_vec(vec![seq, dim], out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_keeps_position_zero_unchanged() {
        let x = Tensor::from_vec(vec![1, 2], vec![1.0, 2.0]).expect("invariant: x valide");
        let out = apply_rope(&x, 10_000.0).expect("invariant: rope valide");
        assert_eq!(out.data(), &[1.0, 2.0]);
    }

    #[test]
    fn rope_rotates_position_one() {
        let x =
            Tensor::from_vec(vec![2, 2], vec![1.0, 0.0, 1.0, 0.0]).expect("invariant: x valide");
        let out = apply_rope(&x, 10_000.0).expect("invariant: rope valide");
        assert!((out.data()[2] - 1.0_f32.cos()).abs() < 1.0e-6);
        assert!((out.data()[3] - 1.0_f32.sin()).abs() < 1.0e-6);
    }
}
