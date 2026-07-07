fn test_executor() -> Result<Option<MetalExecutor>> {
    match MetalExecutor::new() {
        Ok(executor) => Ok(Some(executor)),
        Err(InferError::Metal(message)) if message.contains("aucun device") => Ok(None),
        Err(error) => Err(error),
    }
}

fn test_dense_linear(out_dim: usize, in_dim: usize) -> Result<Linear> {
    let data = vec![0.0; out_dim * in_dim];
    Linear::new(Tensor::from_vec(vec![out_dim, in_dim], data)?, None)
}

fn test_affine(out_dim: usize, in_dim: usize, scale: f32) -> Result<AffineQuantizedTensor> {
    let bits = 4;
    let values_per_word = 32 / bits;
    let packed_cols = in_dim / values_per_word;
    let groups = in_dim / 64;
    let mut packed = Vec::with_capacity(out_dim * packed_cols);
    for row in 0..out_dim {
        for word in 0..packed_cols {
            let mut lanes = [0_u32; 8];
            for (lane, value) in lanes.iter_mut().enumerate() {
                *value = ((row + word + lane) % 15 + 1) as u32;
            }
            packed.push(pack_lanes(&lanes, bits));
        }
    }
    let scales = Tensor::from_vec(
        vec![out_dim, groups],
        vec![bf16_round(scale); out_dim * groups],
    )?;
    let biases = Tensor::from_vec(vec![out_dim, groups], vec![0.0; out_dim * groups])?;
    AffineQuantizedTensor::new(&[out_dim, packed_cols], packed, scales, biases, 64, bits)
}

fn pack_lanes(values: &[u32], bits: usize) -> u32 {
    values
        .iter()
        .enumerate()
        .fold(0_u32, |word, (idx, value)| word | (value << (idx * bits)))
}

/// Arrondit `v` à bf16 puis revient en f32 (RNE), identique à la conversion de
/// production des scales/biases. Les oracles GPU-vs-CPU utilisent des scales déjà
/// bf16-représentables : le GPU (qui lit les scales en bf16) et le CPU (qui calcule
/// en f32) partagent alors exactement la même valeur → tolérances inchangées.
fn bf16_round(v: f32) -> f32 {
    let bits = v.to_bits();
    let rounding = 0x7fff + ((bits >> 16) & 1);
    f32::from_bits(((bits + rounding) >> 16) << 16)
}

fn assert_close(left: &[f32], right: &[f32]) {
    assert_close_eps(left, right, 1.0e-5);
}

fn assert_close_eps(left: &[f32], right: &[f32], eps: f32) {
    assert_eq!(left.len(), right.len());
    for (idx, (a, b)) in left.iter().zip(right.iter()).enumerate() {
        assert!((a - b).abs() <= eps, "index={idx} left={a} right={b}");
    }
}

/// Vrai si le GPU local est la référence byte-identité des kernels (famille
/// M5, machine des campagnes) : la bit-exactitude qmm2/qmv/fused y est
/// prouvée. Les autres générations (runners CI M1/M2) arrondissent à ±ULP
/// près — on y vérifie en tolérance ULP serrée, pas à l'égalité de bits.
pub(crate) fn bitwise_reference_gpu() -> bool {
    metal::Device::system_default()
        .map(|device| device.name().contains("M5"))
        .unwrap_or(false)
}

/// Écart en ULP entre deux f32 (bits consécutifs), pour les asserts portables.
pub(crate) fn ulp_diff(a: f32, b: f32) -> u32 {
    let (x, y) = (a.to_bits() as i64, b.to_bits() as i64);
    (x - y).unsigned_abs() as u32
}

/// Assert bit-exact sur la machine de référence, tolérance relative ailleurs.
///
/// Sur M5 (machine des campagnes) les variantes qmm2/qmv/fused sont PROUVÉES
/// bit-identiques → égalité de bits stricte. Ailleurs (runners CI M1/M2), deux
/// kernels aux ordres de réduction différents divergent légitimement : jusqu'à
/// 48 ULP mesurés sur qmm2/rms/shared-expert (run 28847172905), soit ~3,6e-6
/// en erreur relative — cohérent avec l'accumulation flottante sur des
/// réductions K jusqu'à ~9216 (≈ √K·ε_f32). L'ULP n'échelonne ni avec la
/// magnitude ni avec K ; on borne donc en RELATIF (5e-5, ~14× la marge
/// observée) + un plancher absolu pour les valeurs proches de zéro. Un vrai
/// bug kernel dévie de plusieurs ordres de grandeur au-dessus.
pub(crate) fn assert_bits_portable(a: f32, b: f32, context: &dyn Fn() -> String) {
    if bitwise_reference_gpu() {
        assert_eq!(a.to_bits(), b.to_bits(), "{} (bits {a:e} vs {b:e})", context());
    } else {
        let tol = 5.0e-5 * a.abs().max(b.abs()) + 2.0e-6;
        let d = ulp_diff(a, b);
        assert!(
            (a - b).abs() <= tol,
            "{} (écart {:e} > tol {tol:e} ; ULP {d} : {a:e} vs {b:e})",
            context(),
            (a - b).abs()
        );
    }
}
