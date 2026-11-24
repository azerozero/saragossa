//! Comparaisons appariées par prompt, portage de `tools/quant/paired.py`.
//!
//! Unité statistique : le prompt. IC 95 % par bootstrap percentile sur les
//! prompts, 10 000 tirages, graine 0. Le générateur reproduit `random.Random(0)`
//! de CPython (MT19937, `randrange` par rejet) et les sommes reproduisent
//! `sum()` de CPython ≥ 3.12 (compensation de Neumaier) : la sortie est
//! identique bit à bit à `paired.py` sous Python ≥ 3.12. Python ≤ 3.11 somme
//! sans compensation et peut différer au dernier ULP. La lecture JSON exige
//! la feature `float_roundtrip` de serde_json : sans elle, un flottant lu peut
//! s'écarter d'un ULP de celui de `json.loads`.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use serde::Serialize;
use serde_json::Value;

use crate::{InferError, Result};

const DRAWS: usize = 10_000;
const MAX_REGRESSION: f64 = 0.10;

fn invalid(message: impl Into<String>) -> InferError {
    InferError::Config(format!("paired: {}", message.into()))
}

/// MT19937 de CPython, initialisé comme `random.seed(0)`.
struct PyRandom {
    state: [u32; 624],
    index: usize,
}

impl PyRandom {
    fn seed_zero() -> Self {
        let mut state = [0_u32; 624];
        state[0] = 19_650_218;
        for i in 1..624 {
            state[i] = 1_812_433_253_u32
                .wrapping_mul(state[i - 1] ^ (state[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        // init_by_array(key = [0]) : `key[j] + j` vaut toujours 0.
        let mut i = 1;
        for _ in 0..624 {
            state[i] ^= (state[i - 1] ^ (state[i - 1] >> 30)).wrapping_mul(1_664_525);
            i += 1;
            if i >= 624 {
                state[0] = state[623];
                i = 1;
            }
        }
        for _ in 0..623 {
            state[i] = (state[i]
                ^ (state[i - 1] ^ (state[i - 1] >> 30)).wrapping_mul(1_566_083_941))
            .wrapping_sub(i as u32);
            i += 1;
            if i >= 624 {
                state[0] = state[623];
                i = 1;
            }
        }
        state[0] = 0x8000_0000;
        Self { state, index: 624 }
    }

    fn next_u32(&mut self) -> u32 {
        if self.index >= 624 {
            for k in 0..624 {
                let y = (self.state[k] & 0x8000_0000) | (self.state[(k + 1) % 624] & 0x7fff_ffff);
                let mag = if y & 1 == 1 { 0x9908_b0df } else { 0 };
                self.state[k] = self.state[(k + 397) % 624] ^ (y >> 1) ^ mag;
            }
            self.index = 0;
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    /// `randrange(n)` : `getrandbits(n.bit_length())` rejeté tant que ≥ n.
    fn below(&mut self, n: u32) -> u32 {
        let shift = n.leading_zeros();
        loop {
            let r = self.next_u32() >> shift;
            if r < n {
                return r;
            }
        }
    }
}

/// `sum()` de CPython ≥ 3.12 sur des flottants (Neumaier).
fn py_sum(values: impl IntoIterator<Item = f64>) -> f64 {
    let (mut sum, mut c) = (0.0_f64, 0.0_f64);
    for x in values {
        let t = sum + x;
        c += if sum.abs() >= x.abs() {
            (sum - t) + x
        } else {
            (x - t) + sum
        };
        sum = t;
    }
    if c != 0.0 && c.is_finite() {
        sum + c
    } else {
        sum
    }
}

fn mean(values: &[f64]) -> f64 {
    py_sum(values.iter().copied()) / values.len() as f64
}

fn bootstrap_ci(values: &[f64]) -> Result<[f64; 2]> {
    let n = u32::try_from(values.len())
        .ok()
        .filter(|&n| n > 0)
        .ok_or_else(|| invalid("aucun prompt ou trop de prompts"))?;
    let mut rng = PyRandom::seed_zero();
    let mut means: Vec<f64> = (0..DRAWS)
        .map(|_| py_sum((0..n).map(|_| values[rng.below(n) as usize])) / f64::from(n))
        .collect();
    // Tri stable à égalité, comme `sorted` ; les valeurs sont finies (chargement).
    means.sort_by(f64::total_cmp);
    Ok([means[DRAWS / 40], means[DRAWS * 39 / 40 - 1]])
}

struct Scores {
    /// Prompt → (métrique, catégorie), trié par identifiant.
    prompts: BTreeMap<String, (f64, String)>,
    reference: Option<String>,
}

fn load(path: &Path, metric: &str) -> Result<Scores> {
    let bytes = fs::read(path).map_err(|source| InferError::Io {
        path: path.into(),
        source,
    })?;
    let data: Value = serde_json::from_slice(&bytes).map_err(|source| InferError::Json {
        path: path.into(),
        source,
    })?;
    let where_ = || path.display().to_string();
    let prompts = data
        .get("prompts")
        .and_then(Value::as_object)
        .ok_or_else(|| invalid(format!("{}: champ prompts absent", where_())))?
        .iter()
        .map(|(id, entry)| {
            let value = entry.get(metric).and_then(Value::as_f64);
            let category = entry.get("category").and_then(Value::as_str);
            match (value, category) {
                (Some(value), Some(category)) if value.is_finite() => {
                    Ok((id.clone(), (value, category.to_string())))
                }
                _ => Err(invalid(format!(
                    "{}: prompt {id} sans {metric} fini ou sans catégorie",
                    where_()
                ))),
            }
        })
        .collect::<Result<_>>()?;
    let reference = data
        .get("ref_sha256")
        .and_then(Value::as_str)
        .map(str::to_string);
    Ok(Scores { prompts, reference })
}

struct Aligned {
    series: Vec<Vec<f64>>,
    categories: Vec<String>,
    order: Vec<String>,
}

fn aligned(paths: &[&Path], metric: &str) -> Result<Aligned> {
    let loaded = paths
        .iter()
        .map(|path| load(path, metric))
        .collect::<Result<Vec<_>>>()?;
    let first = &loaded[0];
    if loaded.iter().any(|scores| {
        !scores.prompts.keys().eq(first.prompts.keys()) || scores.reference != first.reference
    }) {
        return Err(invalid(
            "comparaison refusée : prompts ou référence différents",
        ));
    }
    if first.prompts.is_empty() {
        return Err(invalid("aucun prompt"));
    }
    Ok(Aligned {
        series: loaded
            .iter()
            .map(|scores| scores.prompts.values().map(|(value, _)| *value).collect())
            .collect(),
        categories: first.prompts.values().map(|(_, c)| c.clone()).collect(),
        order: first.prompts.keys().cloned().collect(),
    })
}

#[derive(Debug, Serialize)]
pub struct Category {
    pub candidate: f64,
    pub reference: f64,
    pub relative: f64,
}

/// Sortie de `paired diff`, champs dans l'ordre de `paired.py`.
#[derive(Debug, Serialize)]
pub struct Diff {
    pub metric: String,
    pub prompts: usize,
    pub candidate: f64,
    pub reference: f64,
    pub mean_diff: f64,
    pub ci95: [f64; 2],
    pub categories: BTreeMap<String, Category>,
    pub regressed: Vec<String>,
    pub gain: bool,
    pub worst_prompt: String,
}

/// Différences appariées candidat − référence. Gain si l'IC est entièrement
/// sous 0 et si aucune catégorie ne régresse de plus de 10 % en moyenne.
pub fn diff(candidate: &Path, reference: &Path, metric: &str) -> Result<Diff> {
    let Aligned {
        series,
        categories,
        order,
    } = aligned(&[candidate, reference], metric)?;
    let (c, r) = (&series[0], &series[1]);
    let d: Vec<f64> = c.iter().zip(r).map(|(a, b)| a - b).collect();
    let ci95 = bootstrap_ci(&d)?;
    let mut by_category = BTreeMap::new();
    for category in &categories {
        if by_category.contains_key(category) {
            continue;
        }
        let pick = |series: &[f64]| -> Vec<f64> {
            series
                .iter()
                .zip(&categories)
                .filter(|(_, c)| *c == category)
                .map(|(v, _)| *v)
                .collect()
        };
        let (mc, mr) = (mean(&pick(c)), mean(&pick(r)));
        let relative = if mr != 0.0 { (mc - mr) / mr } else { 0.0 };
        by_category.insert(
            category.clone(),
            Category {
                candidate: mc,
                reference: mr,
                relative,
            },
        );
    }
    let regressed: Vec<String> = by_category
        .iter()
        .filter(|(_, v)| v.relative > MAX_REGRESSION)
        .map(|(k, _)| k.clone())
        .collect();
    // `max(zip(d, order))` : plus grand écart, puis plus grand identifiant.
    let worst_prompt = d
        .iter()
        .zip(&order)
        .max_by(|a, b| a.0.total_cmp(b.0).then_with(|| a.1.cmp(b.1)))
        .map(|(_, id)| id.clone())
        .ok_or_else(|| invalid("aucun prompt"))?;
    Ok(Diff {
        metric: metric.into(),
        prompts: d.len(),
        candidate: mean(c),
        reference: mean(r),
        mean_diff: mean(&d),
        ci95,
        gain: ci95[1] < 0.0 && regressed.is_empty(),
        categories: by_category,
        regressed,
        worst_prompt,
    })
}

#[derive(Debug, Serialize)]
pub struct Effect {
    pub effect: f64,
    pub ci95: [f64; 2],
    pub excludes_zero: bool,
}

#[derive(Debug, Serialize)]
#[allow(non_snake_case, reason = "noms de cellules et d'effets de paired.py")]
pub struct Cells {
    pub A0F0: f64,
    pub A0F1: f64,
    pub A1F0: f64,
    pub A1F1: f64,
}

/// Sortie de `paired factorial` : effets principaux, interaction et cellules.
#[derive(Debug, Serialize)]
#[allow(non_snake_case, reason = "noms de cellules et d'effets de paired.py")]
pub struct Factorial {
    pub A: Effect,
    pub F: Effect,
    pub AxF: Effect,
    pub cells: Cells,
}

/// Plan 2×2 (facteurs A et F, niveaux 0/1), contrastes par prompt.
pub fn factorial(cells: [&Path; 4], metric: &str) -> Result<Factorial> {
    let series = aligned(&cells, metric)?.series;
    let [y00, y01, y10, y11] = [&series[0], &series[1], &series[2], &series[3]];
    let contrast = |f: fn(f64, f64, f64, f64) -> f64| -> Result<Effect> {
        let values: Vec<f64> = (0..y00.len())
            .map(|i| f(y00[i], y01[i], y10[i], y11[i]))
            .collect();
        let ci95 = bootstrap_ci(&values)?;
        Ok(Effect {
            effect: mean(&values),
            ci95,
            excludes_zero: ci95[0] > 0.0 || ci95[1] < 0.0,
        })
    };
    Ok(Factorial {
        A: contrast(|a, b, c, d| (d + c - b - a) / 2.0)?,
        F: contrast(|a, b, c, d| (d + b - c - a) / 2.0)?,
        AxF: contrast(|a, b, c, d| (d - c - b + a) / 2.0)?,
        cells: Cells {
            A0F0: mean(y00),
            A0F1: mean(y01),
            A1F0: mean(y10),
            A1F1: mean(y11),
        },
    })
}

/// JSON indenté d'un espace, comme `json.dumps(out, indent=1)`.
pub fn to_json(value: &impl Serialize) -> Result<String> {
    let mut out = Vec::new();
    let formatter = serde_json::ser::PrettyFormatter::with_indent(b" ");
    let mut serializer = serde_json::Serializer::with_formatter(&mut out, formatter);
    value
        .serialize(&mut serializer)
        .map_err(|source| InferError::Json {
            path: "stdout".into(),
            source,
        })?;
    String::from_utf8(out).map_err(|_| invalid("JSON non UTF-8"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn write(dir: &Path, name: &str, values: &[f64], sha: &str) -> PathBuf {
        let prompts: serde_json::Map<String, Value> = values
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let category = if i % 2 == 1 { "code" } else { "prose" };
                (
                    format!("p{i}"),
                    serde_json::json!({"kl_mean": v, "category": category}),
                )
            })
            .collect();
        let path = dir.join(name);
        fs::write(
            &path,
            serde_json::json!({"ref_sha256": sha, "prompts": prompts}).to_string(),
        )
        .expect("invariant: écriture fixture");
        path
    }

    #[test]
    fn rng_matches_cpython_random_seed_zero() {
        // python3 -c "import random; r=random.Random(0);
        //   print([r.getrandbits(32) for _ in range(3)], [r.randrange(7) for _ in range(8)])"
        let mut rng = PyRandom::seed_zero();
        let words: Vec<u32> = (0..3).map(|_| rng.next_u32()).collect();
        assert_eq!(words, [3_626_764_237, 1_654_615_998, 3_255_389_356]);
        let draws: Vec<u32> = (0..8).map(|_| rng.below(7)).collect();
        assert_eq!(draws, [3, 0, 2, 4, 3, 3, 6, 6]);
    }

    #[test]
    fn py_sum_compensates_like_cpython_312() {
        // sum([0.1] * 10) vaut 1.0 sous Python ≥ 3.12 (0.9999999999999999 avant).
        assert_eq!(py_sum([0.1; 10]), 1.0);
        assert_eq!(py_sum([1e100, 1.0, -1e100]), 1.0);
    }

    #[test]
    fn self_check_of_paired_py() {
        assert_eq!(bootstrap_ci(&[1.0; 5]).expect("invariant: IC"), [1.0, 1.0]);
        let [low, high] = bootstrap_ci(&[-1.0, -2.0, -1.5, -0.5]).expect("invariant: IC");
        assert!((-2.0..=high).contains(&low) && high < 0.0);
        let dir = tempfile::tempdir().expect("invariant: répertoire temporaire");
        let dir = dir.path();
        let reference = write(dir, "r.json", &[0.02, 0.03, 0.02, 0.04], "x");
        let better = write(dir, "b.json", &[0.01, 0.02, 0.015, 0.03], "x");
        let out = diff(&better, &reference, "kl_mean").expect("invariant: diff");
        assert!(out.gain && out.ci95[1] < 0.0 && out.regressed.is_empty());
        let mixed = write(dir, "m.json", &[0.01, 0.05, 0.01, 0.06], "x");
        let out = diff(&mixed, &reference, "kl_mean").expect("invariant: diff");
        assert!(!out.gain && out.regressed == ["code"]);
        let other = write(dir, "o.json", &[0.02, 0.03, 0.02, 0.04], "y");
        assert!(diff(&better, &other, "kl_mean").is_err());
        let y = [0.02, 0.03, 0.025, 0.04];
        let shifted = |name, da: f64| write(dir, name, &y.map(|v| v + da), "x");
        let cells = [
            shifted("00", 0.0),
            shifted("01", 0.0),
            shifted("10", -0.01),
            shifted("11", -0.01),
        ];
        let out = factorial(
            [&cells[0], &cells[1], &cells[2], &cells[3]].map(PathBuf::as_path),
            "kl_mean",
        )
        .expect("invariant: factoriel");
        assert!((out.A.effect + 0.01).abs() < 1e-12 && out.A.excludes_zero);
        assert!(out.F.effect.abs() < 1e-12 && out.AxF.effect.abs() < 1e-12);
    }

    /// Parité bit à bit avec `python3.13 -I tools/quant/paired.py` sur la
    /// fixture ci-dessous (valeurs relevées le 24 nov. 2026).
    #[test]
    fn parity_with_paired_py_fixture() {
        let dir = tempfile::tempdir().expect("invariant: répertoire temporaire");
        let dir = dir.path();
        let values = |offset: f64, slope: f64| -> Vec<f64> {
            (0..13)
                .map(|i| 0.03 + offset + slope * f64::from(i % 5) - 0.001 * f64::from(i % 3))
                .collect()
        };
        let reference = write(dir, "ref.json", &values(0.0, 0.004), "s");
        let candidate = write(dir, "cand.json", &values(-0.002, 0.005), "s");
        let out = diff(&candidate, &reference, "kl_mean").expect("invariant: diff");
        let expected = include_str!("paired_fixture_diff.json");
        assert_eq!(to_json(&out).expect("invariant: JSON"), expected.trim_end());
        let cells = [
            write(dir, "00.json", &values(0.0, 0.004), "s"),
            write(dir, "01.json", &values(0.001, 0.003), "s"),
            write(dir, "10.json", &values(-0.003, 0.004), "s"),
            write(dir, "11.json", &values(-0.001, 0.006), "s"),
        ];
        let out = factorial(
            [&cells[0], &cells[1], &cells[2], &cells[3]].map(PathBuf::as_path),
            "kl_mean",
        )
        .expect("invariant: factoriel");
        let expected = include_str!("paired_fixture_factorial.json");
        assert_eq!(to_json(&out).expect("invariant: JSON"), expected.trim_end());
    }

    #[test]
    fn rejects_missing_metric_and_non_finite_values() {
        let dir = tempfile::tempdir().expect("invariant: répertoire temporaire");
        let path = dir.path().join("bad.json");
        fs::write(&path, r#"{"prompts": {"p0": {"category": "code"}}}"#)
            .expect("invariant: fixture");
        assert!(load(&path, "kl_mean").is_err());
        fs::write(&path, r#"{"prompts": {}}"#).expect("invariant: fixture");
        assert!(diff(&path, &path, "kl_mean").is_err());
    }
}
