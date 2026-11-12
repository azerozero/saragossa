//! Comparaisons de sorties synthétiques, sans device ni poids de modèle.

fn check_finite(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "longueurs différentes");
    assert!(a.iter().chain(b).all(|v| v.is_finite()), "sortie non finie");
}

/// Σ(gpu − cpu)² / Σcpu² ; référence nulle : 0 si identique, +∞ sinon.
pub(crate) fn nmse(gpu: &[f32], cpu: &[f32]) -> f64 {
    check_finite(gpu, cpu);
    let mut error = 0.0;
    let mut reference = 0.0;
    for (&g, &c) in gpu.iter().zip(cpu) {
        error += (f64::from(g) - f64::from(c)).powi(2);
        reference += f64::from(c).powi(2);
    }
    if reference == 0.0 {
        if error == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        error / reference
    }
}

pub(crate) fn assert_nmse(label: &str, gpu: &[f32], cpu: &[f32], max: f64) {
    let error = nmse(gpu, cpu);
    assert!(
        error.is_finite() && error <= max,
        "{label}: NMSE={error:e} > {max:e}"
    );
}

/// Tolérance historique du decode résident : tol × (1 + |cpu[i]|).
pub(crate) fn assert_close_rel(label: &str, gpu: &[f32], cpu: &[f32], tol: f32) {
    assert_eq!(gpu.len(), cpu.len(), "{label}: longueurs");
    for (i, (g, c)) in gpu.iter().zip(cpu).enumerate() {
        assert!(
            (g - c).abs() <= tol * (1.0 + c.abs()),
            "{label}[{i}] gpu={g} cpu={c}"
        );
    }
}

/// La fermeture doit attendre la fin du travail GPU qu'elle mesure.
/// Sans opt-in : un appel ; avec opt-in : un warmup puis au moins une mesure.
pub(crate) fn time_gpu<T>(label: &str, iters: usize, mut f: impl FnMut() -> T) -> T {
    let mut result = f();
    if std::env::var("SARAGOSSA_KERNEL_PERF").as_deref() != Ok("1") {
        return result;
    }
    let mut times = Vec::with_capacity(iters.max(1));
    for _ in 0..iters.max(1) {
        let start = std::time::Instant::now();
        let next = f();
        times.push(start.elapsed().as_secs_f64() * 1e6);
        result = next;
    }
    times.sort_by(f64::total_cmp);
    let mid = times.len() / 2;
    let median = (times[mid] + times[(times.len() - 1) / 2]) / 2.0;
    eprintln!("kernel={label} median_us={median}");
    result
}

// Les oracles historiques conservent aussi leurs diagnostics et le comportement
// IEEE des comparaisons : ne pas leur appliquer check_finite.
pub(crate) fn assert_close_abs(left: &[f32], right: &[f32], tolerance: f32) {
    assert_eq!(left.len(), right.len());
    for (idx, (a, b)) in left.iter().zip(right).enumerate() {
        assert!(
            (a - b).abs() <= tolerance,
            "index={idx} left={a} right={b} tolerance={tolerance}"
        );
    }
}

/// Le comparateur sink rejetait déjà les non-finis.
pub(crate) fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    assert!(
        a.iter().chain(b).all(|value| value.is_finite()),
        "sortie non finie"
    );
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// Le produit scalaire utilise zip, les normes portent sur les tranches entières.
pub(crate) fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot = a
        .iter()
        .zip(b)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum::<f64>();
    let na = a.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    let nb = b.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    (dot / (na * nb).max(f64::MIN_POSITIVE)) as f32
}

pub(crate) fn assert_close_l2(label: &str, gpu: &[f32], cpu: &[f32], tol: f64) {
    let (rel, max) = crate::qwen4_exp::math::relative_error(gpu, cpu);
    eprintln!("{label}: rel={rel:.3e} max_abs={max:.3e}");
    assert!(
        rel <= tol,
        "{label}: erreur relative {rel:.3e} > {tol:.0e} (max {max:.3e})"
    );
}

#[test]
fn known_nmse() {
    assert_eq!(nmse(&[2.0, 2.0], &[1.0, 2.0]), 0.2);
    assert_nmse("identique", &[1.0, 2.0], &[1.0, 2.0], 0.0);
}

#[test]
fn zero_reference() {
    assert_eq!(nmse(&[0.0], &[0.0]), 0.0);
    assert_eq!(nmse(&[1.0], &[0.0]), f64::INFINITY);
    assert_eq!(nmse(&[], &[]), 0.0);
    assert_nmse("zéro", &[0.0], &[0.0], 0.0);
    assert!(
        std::panic::catch_unwind(|| assert_nmse("zéro", &[1.0], &[0.0], f64::INFINITY)).is_err()
    );
}

#[test]
#[should_panic(expected = "sortie non finie")]
fn nan_detected() {
    assert_nmse("NaN", &[f32::NAN], &[1.0], 1.0);
}

#[test]
#[should_panic(expected = "corruption: NMSE=")]
fn injected_corruption_detected() {
    let cpu = [1.0, 2.0, 3.0, 4.0];
    let mut gpu = cpu;
    gpu[2] += 1.0;
    assert_nmse("corruption", &gpu, &cpu, 1e-3);
}

#[test]
fn nmse_non_finite_comparisons_rejected() {
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for (a, b) in [([invalid], [1.0]), ([1.0], [invalid])] {
            assert!(std::panic::catch_unwind(|| nmse(&a, &b)).is_err());
        }
    }
}

#[test]
fn legacy_metrics_and_timing() {
    assert_close_rel("relatif", &[102.0, 0.01], &[100.0, 0.0], 0.02);
    assert_close_abs(&[1.25], &[1.0], 0.25);
    assert_close_l2("norme", &[2.0, 2.0], &[1.0, 2.0], 0.45);
    assert!(std::panic::catch_unwind(|| assert_close_abs(&[102.0], &[100.0], 0.02)).is_err());
    assert!(
        std::panic::catch_unwind(|| assert_close_rel("relatif", &[0.1], &[0.0], 0.02)).is_err()
    );
    assert_eq!(max_abs_diff(&[1.0, -2.0], &[2.0, 2.0]), 4.0);
    assert_eq!(cosine(&[1.0, 0.0], &[0.0, 1.0]), 0.0);
    assert_eq!(cosine(&[0.0], &[0.0]), 0.0);
    let mut calls = 0;
    let result = time_gpu("oracle_noop", 4, || {
        calls += 1;
        calls
    });
    let expected = if std::env::var("SARAGOSSA_KERNEL_PERF").as_deref() == Ok("1") {
        5
    } else {
        1
    };
    assert_eq!(calls, expected);
    assert_eq!(result, calls);
}

/// Compare le diagnostic complet, y compris les détails produits par assert_eq!.
pub(crate) fn assert_panics_with(expected: &str, f: impl FnOnce() + std::panic::UnwindSafe) {
    let payload = std::panic::catch_unwind(f).expect_err("le comparateur devait échouer");
    let message = payload
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| payload.downcast_ref::<&str>().copied());
    assert_eq!(message, Some(expected));
}

#[test]
fn nmse_rejects_different_lengths() {
    for (a, b) in [(&[1.0][..], &[][..]), (&[][..], &[1.0][..])] {
        assert_panics_with(
            &format!(
                "assertion `left == right` failed: longueurs différentes\n  left: {}\n right: {}",
                a.len(),
                b.len()
            ),
            || assert_nmse("longueurs", a, b, 1.0),
        );
    }
}

#[test]
fn legacy_relative_edges_and_diagnostics() {
    assert_close_rel("vide", &[], &[], f32::NAN);
    assert_close_rel("infini", &[1.0], &[f32::INFINITY], 0.1);
    assert_close_rel("infini", &[1.0], &[f32::NEG_INFINITY], 0.1);
    assert_close_rel("tolérance", &[f32::INFINITY], &[1.0], f32::INFINITY);
    for (g, c) in [
        (f32::NAN, 0.0),
        (0.0, f32::NAN),
        (f32::INFINITY, f32::INFINITY),
        (f32::INFINITY, 1.0),
    ] {
        assert_panics_with(&format!("relatif[1] gpu={g} cpu={c}"), || {
            assert_close_rel("relatif", &[0.0, g], &[0.0, c], 0.1);
        });
    }
    assert_panics_with("relatif[0] gpu=0 cpu=0", || {
        assert_close_rel("relatif", &[0.0], &[0.0], f32::NAN);
    });
    assert_panics_with(
        "assertion `left == right` failed: relatif: longueurs\n  left: 1\n right: 0",
        || assert_close_rel("relatif", &[f32::NAN], &[], 0.1),
    );
}

#[test]
fn legacy_absolute_edges_and_diagnostics() {
    assert_close_abs(&[], &[], f32::NAN);
    assert_close_abs(&[-0.0], &[0.0], 0.0);
    assert_close_abs(&[f32::INFINITY], &[1.0], f32::INFINITY);
    assert_close_abs(&[1.0], &[f32::NEG_INFINITY], f32::INFINITY);
    for (a, b) in [
        (f32::NAN, 0.0),
        (0.0, f32::NAN),
        (f32::INFINITY, f32::INFINITY),
        (1.0, f32::NEG_INFINITY),
        (2.0, 0.0),
    ] {
        assert_panics_with(&format!("index=1 left={a} right={b} tolerance=0.1"), || {
            assert_close_abs(&[0.0, a], &[0.0, b], 0.1);
        });
    }
    assert_panics_with("index=0 left=0 right=0 tolerance=NaN", || {
        assert_close_abs(&[0.0], &[0.0], f32::NAN);
    });
    assert_panics_with(
        "assertion `left == right` failed\n  left: 1\n right: 0",
        || assert_close_abs(&[f32::NAN], &[], 0.1),
    );
}

#[test]
fn legacy_sink_edges_and_diagnostics() {
    assert_eq!(max_abs_diff(&[], &[]), 0.0);
    assert_eq!(max_abs_diff(&[f32::MAX], &[-f32::MAX]), f32::INFINITY);
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for (a, b) in [([invalid], [0.0]), ([0.0], [invalid])] {
            assert_panics_with("sortie non finie", || {
                max_abs_diff(&a, &b);
            });
        }
    }
    assert_panics_with(
        "assertion `left == right` failed\n  left: 1\n right: 0",
        || {
            max_abs_diff(&[f32::NAN], &[]);
        },
    );
}

#[test]
fn legacy_cosine_keeps_zip_and_non_finite_results() {
    assert_eq!(cosine(&[], &[]), 0.0);
    assert_eq!(cosine(&[], &[1.0]), 0.0);
    assert_eq!(cosine(&[1.0], &[]), 0.0);
    // Les éléments sans partenaire comptent dans la norme, pas dans le produit.
    assert_eq!(cosine(&[3.0], &[3.0, 4.0]), 0.6);
    assert_eq!(cosine(&[3.0, 4.0], &[3.0]), 0.6);
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(cosine(&[invalid], &[1.0]).is_nan());
        assert!(cosine(&[1.0], &[invalid]).is_nan());
    }
    assert_eq!(cosine(&[1.0], &[1.0, f32::INFINITY]), 0.0);
    assert_eq!(cosine(&[1.0], &[1.0, f32::NAN]), f32::INFINITY);
}

#[test]
fn legacy_l2_keeps_zip_and_diagnostics() {
    assert_close_l2("vide", &[], &[], 0.0);
    assert_close_l2("zip", &[1.0, f32::NAN], &[1.0], 0.0);
    assert_close_l2("zip", &[1.0], &[1.0, f32::INFINITY], 0.0);
    assert_close_l2("zip", &[], &[f32::NAN], 0.0);
    assert_close_l2("infini", &[f32::INFINITY], &[1.0], f64::INFINITY);
    for (a, b, max) in [
        (f32::NAN, 1.0, "0.000e0"),
        (1.0, f32::NAN, "0.000e0"),
        (1.0, f32::INFINITY, "inf"),
        (f32::INFINITY, f32::INFINITY, "0.000e0"),
    ] {
        assert_panics_with(
            &format!("norme: erreur relative NaN > 1e-1 (max {max})"),
            || {
                assert_close_l2("norme", &[a], &[b], 0.1);
            },
        );
    }
    assert_panics_with(
        "norme: erreur relative 1.000e0 > 1e-1 (max 1.000e0)",
        || {
            assert_close_l2("norme", &[2.0], &[1.0], 0.1);
        },
    );
}
