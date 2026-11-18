//! `exp`, sigmoïde et `tanh` déterministes, communs au CPU et au GPU.
//!
//! La libm d'Apple et les fonctions Metal n'arrondissent pas pareil ; seules
//! les opérations IEEE de base (+, −, ×, ÷, arrondi à l'entier pair) donnent le
//! même résultat au bit près des deux côtés. Les fonctions ci-dessous n'utilisent
//! qu'elles, dans le même ordre que leur jumelle MSL [`MSL`], compilée sans
//! fast-math ni contraction FMA. Les bornes ±80 évitent tout sous-normal, que le
//! GPU pourrait vider à zéro.

const LOG2E: f32 = std::f32::consts::LOG2_E;
// Cody-Waite : `n · LN2_HI` est exact pour |n| < 2⁸.
const LN2_HI: f32 = f32::from_bits(0x3f31_7200);
const LN2_LO: f32 = f32::from_bits(0x35bf_be8e);

/// `eˣ` avec `x` borné à ±80 ; erreur de quelques ulp.
pub(crate) fn exp(x: f32) -> f32 {
    // Identique au ternaire MSL pour tout x, NaN compris.
    let x = x.clamp(-80.0, 80.0);
    let n = (x * LOG2E).round_ties_even();
    let r = x - n * LN2_HI;
    let r = r - n * LN2_LO;
    // Taylor d'ordre 7 sur |r| ≤ ln2/2 : reste < 1 ulp.
    let mut p = 1.0 / 5040.0;
    p = p * r + 1.0 / 720.0;
    p = p * r + 1.0 / 120.0;
    p = p * r + 1.0 / 24.0;
    p = p * r + 1.0 / 6.0;
    p = p * r + 0.5;
    p = p * r + 1.0;
    p = p * r + 1.0;
    p * f32::from_bits(((n as i32 + 127) as u32) << 23)
}

pub(crate) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + exp(-x))
}

pub(crate) fn tanh(x: f32) -> f32 {
    let a = if x < 0.0 { -x } else { x };
    let t = 1.0 - 2.0 / (exp(a + a) + 1.0);
    if x < 0.0 {
        -t
    } else {
        t
    }
}

/// Jumelle MSL des fonctions ci-dessus, à inclure dans une source Metal.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub(crate) const MSL: &str = r"
static inline float dfx_exp(float x) {
    #pragma METAL fp contract(off)
    x = x < -80.0f ? -80.0f : (x > 80.0f ? 80.0f : x);
    const float n = rint(x * as_type<float>(0x3fb8aa3bu));
    float r = x - n * as_type<float>(0x3f317200u);
    r = r - n * as_type<float>(0x35bfbe8eu);
    float p = 1.0f / 5040.0f;
    p = p * r + 1.0f / 720.0f;
    p = p * r + 1.0f / 120.0f;
    p = p * r + 1.0f / 24.0f;
    p = p * r + 1.0f / 6.0f;
    p = p * r + 0.5f;
    p = p * r + 1.0f;
    p = p * r + 1.0f;
    return p * as_type<float>(uint(int(n) + 127) << 23);
}
static inline float dfx_sigmoid(float x) {
    #pragma METAL fp contract(off)
    return 1.0f / (1.0f + dfx_exp(-x));
}
static inline float dfx_tanh(float x) {
    #pragma METAL fp contract(off)
    const float a = x < 0.0f ? -x : x;
    const float t = 1.0f - 2.0f / (dfx_exp(a + a) + 1.0f);
    return x < 0.0f ? -t : t;
}
";

#[cfg(test)]
mod tests {
    #[test]
    fn close_to_libm() {
        for i in -2000..=2000 {
            let x = i as f32 * 0.01;
            let rel = |a: f32, b: f32| ((a - b) / b.abs().max(1e-30)).abs();
            assert!(rel(super::exp(x), x.exp()) < 1e-6, "exp({x})");
            assert!((super::sigmoid(x) - 1.0 / (1.0 + (-x).exp())).abs() < 1e-6);
            assert!((super::tanh(x) - x.tanh()).abs() < 1e-6, "tanh({x})");
        }
    }
}
