//! Briques numériques f32 CPU du forward BERT : produit scalaire, linéaire
//! PyTorch (`x @ Wᵀ + b`), LayerNorm, gelu exact (erf) et softmax.
//!
//! Tout est **déterministe** : l'ordre des réductions est fixe (accumulateurs
//! en éventail pour le dot, somme séquentielle ailleurs) et le parallélisme
//! rayon ne découpe que par **ligne de tokens** (chaque ligne est calculée
//! entièrement par un seul thread) → même sortie quel que soit le nombre de
//! threads. La parité vs MLX/Metal est numérique (~1e-6 relatif, ordre des
//! réductions différent côté GPU), pas bit-à-bit — c'est le test de parité
//! cosinus ≥ 0,999 côté reti qui fait foi.

use rayon::prelude::*;

/// Poids d'une couche linéaire PyTorch dense : `W` row-major `[out, in]`
/// (layout HF inchangé → `y = x @ Wᵀ + b` se lit ligne à ligne), biais `[out]`.
pub(super) struct Linear {
    pub w: Vec<f32>,
    pub b: Vec<f32>,
    pub out_dim: usize,
    pub in_dim: usize,
}

/// Paire γ/β d'un LayerNorm (`weight`/`bias` HF), chacun `[hidden]`.
pub(super) struct LayerNormW {
    pub gamma: Vec<f32>,
    pub beta: Vec<f32>,
}

/// Produit scalaire f32 en 8 accumulateurs (autovectorisé NEON, ordre de
/// réduction fixe → déterministe).
#[inline]
pub(super) fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "dot: longueurs différentes");
    let mut acc = [0.0f32; 8];
    let mut ca = a.chunks_exact(8);
    let mut cb = b.chunks_exact(8);
    for (xa, xb) in (&mut ca).zip(&mut cb) {
        for k in 0..8 {
            acc[k] += xa[k] * xb[k];
        }
    }
    let mut tail = 0.0f32;
    for (xa, xb) in ca.remainder().iter().zip(cb.remainder()) {
        tail += xa * xb;
    }
    let p0 = (acc[0] + acc[4]) + (acc[2] + acc[6]);
    let p1 = (acc[1] + acc[5]) + (acc[3] + acc[7]);
    (p0 + p1) + tail
}

/// Linéaire PyTorch : `y[i,j] = dot(x[i,·], W[j,·]) + b[j]`, `x` row-major
/// `[rows, in]` → alloue et renvoie `[rows, out]`. Parallélisé par ligne `i`
/// (chaque token indépendant ; déterministe, cf. doc module).
///
/// # Panics
///
/// `debug_assert!` seulement : les shapes sont validées une fois au chargement
/// des poids (`weights.rs`), le forward s'exécute sur des dims cohérentes.
pub(super) fn linear(x: &[f32], rows: usize, lin: &Linear) -> Vec<f32> {
    debug_assert_eq!(x.len(), rows * lin.in_dim, "linear: shape x");
    let mut y = vec![0.0f32; rows * lin.out_dim];
    y.par_chunks_mut(lin.out_dim)
        .zip(x.par_chunks(lin.in_dim))
        .for_each(|(yrow, xrow)| {
            for (j, yj) in yrow.iter_mut().enumerate() {
                let wrow = &lin.w[j * lin.in_dim..(j + 1) * lin.in_dim];
                *yj = dot(xrow, wrow) + lin.b[j];
            }
        });
    y
}

/// LayerNorm in-place sur chaque ligne `[hidden]` : `y = (x-µ)/√(σ²+eps)·γ+β`.
/// µ/σ² accumulés en f64 (h=384 : coût nul, stabilité supérieure ; la parité
/// vs MLX f32 est largement sous la tolérance cosinus).
pub(super) fn layer_norm_inplace(x: &mut [f32], hidden: usize, ln: &LayerNormW, eps: f32) {
    debug_assert_eq!(x.len() % hidden, 0, "layer_norm: shape x");
    for row in x.chunks_exact_mut(hidden) {
        let mean = row.iter().map(|&v| f64::from(v)).sum::<f64>() / hidden as f64;
        let var = row
            .iter()
            .map(|&v| {
                let d = f64::from(v) - mean;
                d * d
            })
            .sum::<f64>()
            / hidden as f64;
        let inv = 1.0 / (var + f64::from(eps)).sqrt();
        for (v, (g, b)) in row.iter_mut().zip(ln.gamma.iter().zip(ln.beta.iter())) {
            let n = ((f64::from(*v) - mean) * inv) as f32;
            *v = n * g + b;
        }
    }
}

/// Fonction d'erreur de Gauss, approximation d'Abramowitz & Stegun 7.1.26
/// évaluée en f64 (erreur absolue max ~1,5e-7 — bien sous le bruit f32 du
/// forward). Rust std n'expose pas `erf` ; on évite une dépendance libm.
pub(super) fn erf(x: f64) -> f64 {
    const A1: f64 = 0.254_829_592;
    const A2: f64 = -0.284_496_736;
    const A3: f64 = 1.421_413_741;
    const A4: f64 = -1.453_152_027;
    const A5: f64 = 1.061_405_429;
    const P: f64 = 0.327_591_1;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + P * x);
    let poly = ((((A5 * t + A4) * t + A3) * t + A2) * t + A1) * t;
    sign * (1.0 - poly * (-x * x).exp())
}

/// gelu **exact** (variante erf, PAS l'approximation tanh — celle du
/// `"hidden_act": "gelu"` de BERT) appliqué in-place :
/// `gelu(x) = x · (1 + erf(x/√2)) / 2`.
pub(super) fn gelu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        let xf = f64::from(*v);
        *v = (xf * 0.5 * (1.0 + erf(xf * std::f64::consts::FRAC_1_SQRT_2))) as f32;
    }
}

/// Softmax in-place numériquement stable (max soustrait). Une ligne de scores
/// d'attention. Entrée vide : no-op.
pub(super) fn softmax_inplace(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return;
    }
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    // sum ≥ 1 (le max contribue exp(0)=1) → division toujours sûre.
    for v in x.iter_mut() {
        *v /= sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_dot(a: &[f32], b: &[f32]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| f64::from(x) * f64::from(y))
            .sum()
    }

    #[test]
    fn dot_matches_naive_on_odd_length() {
        // 387 = non multiple de 8 → exerce le remainder.
        let a: Vec<f32> = (0..387).map(|i| (i as f32 * 0.37).sin()).collect();
        let b: Vec<f32> = (0..387).map(|i| (i as f32 * 0.11).cos()).collect();
        let got = f64::from(dot(&a, &b));
        let want = naive_dot(&a, &b);
        assert!((got - want).abs() < 1e-3, "dot {got} vs naïf {want}");
    }

    #[test]
    fn linear_matches_naive() {
        let lin = Linear {
            // W [2, 3] row-major : lignes (1,2,3) et (4,5,6).
            w: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            b: vec![0.5, -0.5],
            out_dim: 2,
            in_dim: 3,
        };
        let x = vec![1.0, 0.0, -1.0, 2.0, 2.0, 2.0];
        let y = linear(&x, 2, &lin);
        // ligne 0 : (1-3)+0.5 = -1.5 ; (4-6)-0.5 = -2.5
        // ligne 1 : (2+4+6)+0.5 = 12.5 ; (8+10+12)-0.5 = 29.5
        assert_eq!(y, vec![-1.5, -2.5, 12.5, 29.5]);
    }

    #[test]
    fn layer_norm_zero_mean_unit_var() {
        let ln = LayerNormW {
            gamma: vec![1.0; 4],
            beta: vec![0.0; 4],
        };
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0];
        layer_norm_inplace(&mut x, 4, &ln, 1e-12);
        let mean: f32 = x.iter().sum::<f32>() / 4.0;
        let var: f32 = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-6, "moyenne {mean}");
        assert!((var - 1.0).abs() < 1e-4, "variance {var}");
    }

    #[test]
    fn erf_reference_values() {
        // Valeurs de référence (tables) ; tolérance = erreur max documentée A&S.
        for (x, want) in [
            (0.0, 0.0),
            (0.5, 0.520_499_877_8),
            (1.0, 0.842_700_792_9),
            (2.0, 0.995_322_265_0),
            (-1.0, -0.842_700_792_9),
        ] {
            let got = erf(x);
            assert!(
                (got - want).abs() < 2e-7,
                "erf({x}) = {got}, attendu {want}"
            );
        }
    }

    #[test]
    fn gelu_reference_values() {
        // gelu(0)=0 ; gelu(1)=0.841345 ; gelu(-1)=-0.158655 (variante erf).
        let mut x = vec![0.0f32, 1.0, -1.0];
        gelu_inplace(&mut x);
        assert!(x[0].abs() < 1e-7);
        assert!((x[1] - 0.841_345).abs() < 1e-5, "gelu(1)={}", x[1]);
        assert!((x[2] + 0.158_655).abs() < 1e-5, "gelu(-1)={}", x[2]);
    }

    #[test]
    fn softmax_sums_to_one_and_orders() {
        let mut x = vec![1.0f32, 2.0, 3.0];
        softmax_inplace(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(x[2] > x[1] && x[1] > x[0]);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        /// INVARIANT : `dot` ne panique jamais et colle au naïf f64 (tolérance
        /// proportionnelle à la magnitude — entrées bornées).
        #[test]
        fn dot_never_panics_matches_naive(
            v in prop::collection::vec((-10.0f32..10.0, -10.0f32..10.0), 0..512)
        ) {
            let a: Vec<f32> = v.iter().map(|p| p.0).collect();
            let b: Vec<f32> = v.iter().map(|p| p.1).collect();
            let got = f64::from(dot(&a, &b));
            let want: f64 = a.iter().zip(&b).map(|(&x, &y)| f64::from(x) * f64::from(y)).sum();
            prop_assert!((got - want).abs() <= 1e-2 + want.abs() * 1e-4,
                "dot {got} vs naïf {want}");
        }

        /// INVARIANT : softmax produit une distribution (somme 1, valeurs dans
        /// [0,1]) pour toute entrée finie non vide.
        #[test]
        fn softmax_is_distribution(mut x in prop::collection::vec(-50.0f32..50.0, 1..128)) {
            softmax_inplace(&mut x);
            let sum: f32 = x.iter().sum();
            prop_assert!((sum - 1.0).abs() < 1e-4, "somme {sum}");
            prop_assert!(x.iter().all(|&v| (0.0..=1.0 + 1e-6).contains(&v)));
        }

        /// INVARIANT : erf est impaire, bornée à ]-1, 1[ et croissante.
        #[test]
        fn erf_odd_bounded_monotonic(x in -6.0f64..6.0, dx in 1e-3f64..1.0) {
            prop_assert!((erf(x) + erf(-x)).abs() < 3e-7);
            prop_assert!(erf(x).abs() <= 1.0);
            prop_assert!(erf(x + dx) >= erf(x) - 3e-7);
        }

        /// INVARIANT : layer_norm ne panique pas et produit des valeurs finies
        /// pour toute entrée finie (γ=1, β=0).
        #[test]
        fn layer_norm_finite(mut x in prop::collection::vec(-100.0f32..100.0, 8..64)) {
            let h = x.len();
            let ln = LayerNormW { gamma: vec![1.0; h], beta: vec![0.0; h] };
            layer_norm_inplace(&mut x, h, &ln, 1e-12);
            prop_assert!(x.iter().all(|v| v.is_finite()));
        }
    }
}
