//! Référence KL plein vocabulaire, indépendante du format llama.cpp.

use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{CausalDecoder, InferError, Result};

const VERSION: u32 = 1;
const MAX_JSON: u64 = 1 << 20;
const MAX_TOKENS: usize = 65_536;
const MAX_VALUES: usize = 64 * 1024 * 1024;
const TRANSFORMS: &str = "after_softcapping_before_temperature_top_k_top_p";

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Metadata {
    version: u32,
    prompt_ids: Vec<usize>,
    forced_ids: Vec<usize>,
    vocab: usize,
    n: usize,
    config_sha256: String,
    /// Échantillon : embeddings exécutés 0/milieu/dernier et norme finale, f32 LE.
    weights_sample_sha256: String,
    transformations: String,
}

/// Mesures sur les n distributions, KL en nats et probabilités dans [0, 1].
#[derive(Debug, Default, Serialize)]
pub struct Metrics {
    pub n: usize,
    pub vocab: usize,
    pub kl_mean: f64,
    pub kl_max: f64,
    pub top1_agreement: f64,
    /// Moyenne signée de p_cur(token imposé) - p_ref(token imposé).
    pub forced_delta_p_mean: f64,
    /// Maximum absolu de l'écart du token imposé.
    pub forced_delta_p_max: f64,
    /// Moyenne de |p_cur - p_ref| pour le seul token imposé à chaque pas.
    pub forced_abs_delta_p_mean: f64,
    /// Moyenne de |p_cur - p_ref| sur les n * vocab entrées.
    pub mean_abs_delta_p: f64,
}

fn invalid(message: impl Into<String>) -> InferError {
    InferError::Config(format!("référence KL: {}", message.into()))
}

fn io<T>(path: &Path, result: std::io::Result<T>) -> Result<T> {
    result.map_err(|source| InferError::Io {
        path: path.into(),
        source,
    })
}

fn dimensions(prompt: &[usize], n: usize, vocab: usize) -> Result<usize> {
    if prompt.is_empty()
        || prompt.len() > MAX_TOKENS
        || n == 0
        || n > MAX_TOKENS
        || vocab == 0
        || prompt.iter().any(|&id| id >= vocab)
    {
        return Err(invalid("prompt, nombre de pas ou vocabulaire invalide"));
    }
    n.checked_mul(vocab)
        .filter(|&values| values <= MAX_VALUES)
        .and_then(|values| values.checked_mul(4))
        .ok_or_else(|| invalid("référence au-delà de 256 Mio"))
}

fn read_metadata(path: &Path, vocab: usize) -> Result<Metadata> {
    let json = path.join("ref.json");
    let file = io(&json, File::open(&json))?;
    let mut bytes = Vec::new();
    io(&json, file.take(MAX_JSON + 1).read_to_end(&mut bytes))?;
    if bytes.len() as u64 > MAX_JSON {
        return Err(invalid("ref.json dépasse 1 Mio"));
    }
    let meta: Metadata =
        serde_json::from_slice(&bytes).map_err(|source| InferError::Json { path: json, source })?;
    if meta.version != VERSION {
        return Err(invalid(format!("version inconnue: {}", meta.version)));
    }
    if meta.vocab != vocab {
        return Err(invalid(format!(
            "vocabulaire différent: {} != {vocab}",
            meta.vocab
        )));
    }
    let expected = dimensions(&meta.prompt_ids, meta.n, meta.vocab)?;
    if meta.forced_ids.len() != meta.n
        || meta.forced_ids.iter().any(|&id| id >= vocab)
        || meta.transformations != TRANSFORMS
        || [&meta.config_sha256, &meta.weights_sample_sha256]
            .iter()
            .any(|hash| hash.len() != 64 || !hash.bytes().all(|byte| byte.is_ascii_hexdigit()))
    {
        return Err(invalid("métadonnées incompatibles"));
    }
    let bin = path.join("ref.bin");
    if io(&bin, fs::metadata(&bin))?.len() != expected as u64 {
        return Err(invalid("ref.bin tronqué ou taille incompatible"));
    }
    Ok(meta)
}

/// Enregistre exactement n tokens greedy (EOS compris) dans PATH/ref.{bin,json}.
/// Les distributions prédisent forced_ids[i] après prompt + forced_ids[..i].
/// Aucun sampling, aucun repli du decode résident, aucun logit de prefill
/// conservé en dehors de sa dernière position.
pub fn record(
    model: &CausalDecoder,
    prompt_ids: &[usize],
    n_steps: usize,
    path: &Path,
) -> Result<()> {
    let (vocab, config_sha256, weights_sample_sha256) = model.kl_reference_identity()?;
    dimensions(prompt_ids, n_steps, vocab)?;
    let (forced_ids, rows) = record_rows(model, prompt_ids, n_steps)?;
    let meta = Metadata {
        version: VERSION,
        prompt_ids: prompt_ids.to_vec(),
        forced_ids,
        vocab,
        n: n_steps,
        config_sha256,
        weights_sample_sha256,
        transformations: TRANSFORMS.into(),
    };
    let json = path.join("ref.json");
    let bytes = serde_json::to_vec(&meta).map_err(|source| InferError::Json {
        path: json.clone(),
        source,
    })?;
    if bytes.len() as u64 > MAX_JSON {
        return Err(invalid("ref.json dépasse 1 Mio"));
    }
    // Un répertoire neuf évite de laisser un ancien JSON désigner un binaire
    // partiellement réécrit. Le JSON est publié après la fermeture du binaire.
    io(path, fs::create_dir(path))?;
    let bin = path.join("ref.bin");
    let mut writer = BufWriter::new(io(&bin, File::create(&bin))?);
    for row in rows {
        if row.len() != vocab {
            return Err(invalid("largeur des logits différente du vocabulaire"));
        }
        for value in log_softmax(&row)? {
            io(&bin, writer.write_all(&value.to_le_bytes()))?;
        }
    }
    io(&bin, writer.flush())?;
    drop(writer);
    io(&json, fs::write(&json, bytes))
}

/// Rejoue les tokens de référence et écrit une unique ligne JSON sur stdout.
/// Les empreintes sont informatives : comparer des poids différents est attendu.
pub fn compare(model: &CausalDecoder, path: &Path) -> Result<Metrics> {
    let (vocab, _, _) = model.kl_reference_identity()?;
    let meta = read_metadata(path, vocab)?;
    // Valider les valeurs avant le prefill et le decode.
    let reference = read_rows(path, &meta)?;
    let current = compare_rows(model, &meta)?;
    let mut metrics = Metrics {
        n: meta.n,
        vocab,
        ..Metrics::default()
    };
    for ((reference, current), &token) in reference.iter().zip(&current).zip(&meta.forced_ids) {
        metrics.add(reference, &log_softmax(current)?, token)?;
    }
    metrics.kl_mean /= meta.n as f64;
    metrics.top1_agreement /= meta.n as f64;
    metrics.forced_delta_p_mean /= meta.n as f64;
    metrics.forced_abs_delta_p_mean /= meta.n as f64;
    metrics.mean_abs_delta_p /= (meta.n * vocab) as f64;
    let json = serde_json::to_string(&metrics).map_err(|source| InferError::Json {
        path: path.into(),
        source,
    })?;
    io(
        Path::new("stdout"),
        writeln!(std::io::stdout().lock(), "{json}"),
    )?;
    Ok(metrics)
}

fn read_rows(path: &Path, meta: &Metadata) -> Result<Vec<Vec<f32>>> {
    let bin = path.join("ref.bin");
    let mut reader = BufReader::new(io(&bin, File::open(&bin))?);
    let mut rows = Vec::with_capacity(meta.n);
    for _ in 0..meta.n {
        let mut row = Vec::with_capacity(meta.vocab);
        for _ in 0..meta.vocab {
            let mut bytes = [0; 4];
            io(&bin, reader.read_exact(&mut bytes))?;
            row.push(f32::from_le_bytes(bytes));
        }
        let sum = row.iter().map(|&x| f64::from(x).exp()).sum::<f64>();
        if row.iter().any(|x| !x.is_finite() || *x > 0.0) || (sum - 1.0).abs() > 1e-5 {
            return Err(invalid("log-probabilités non finies ou non normalisées"));
        }
        rows.push(row);
    }
    let mut tail = [0];
    if io(&bin, reader.read(&mut tail))? != 0 {
        return Err(invalid("données superflues après ref.bin"));
    }
    Ok(rows)
}

// Même arrondi f32 pour la référence et le candidat ; les métriques sont ensuite
// calculées en f64, avec renormalisation du léger arrondi du log-softmax stocké.
fn log_softmax(logits: &[f32]) -> Result<Vec<f32>> {
    let normalized = normalize(logits)?;
    normalized
        .into_iter()
        .map(|x| {
            let x = x as f32;
            if x.is_finite() {
                Ok(x)
            } else {
                Err(invalid("log-softmax hors f32"))
            }
        })
        .collect()
}

fn normalize(values: &[f32]) -> Result<Vec<f64>> {
    if values.is_empty() || values.iter().any(|x| !x.is_finite()) {
        return Err(invalid("distribution vide ou non finie"));
    }
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let log_sum = values
        .iter()
        .map(|&x| (f64::from(x) - max).exp())
        .sum::<f64>()
        .ln();
    Ok(values
        .iter()
        .map(|&x| (f64::from(x) - max) - log_sum)
        .collect())
}

fn top1(values: &[f32]) -> usize {
    // Premier indice en cas d'égalité, comme l'argmax greedy du moteur.
    values
        .iter()
        .enumerate()
        .fold(0, |best, (i, x)| if *x > values[best] { i } else { best })
}

impl Metrics {
    fn add(&mut self, reference: &[f32], current: &[f32], token: usize) -> Result<()> {
        if reference.len() != current.len() || token >= reference.len() {
            return Err(invalid("distributions incompatibles"));
        }
        let p = normalize(reference)?;
        let q = normalize(current)?;
        let kl = p
            .iter()
            .zip(&q)
            .map(|(&p, &q)| p.exp() * (p - q))
            .sum::<f64>()
            .max(0.0);
        self.kl_mean += kl;
        self.kl_max = self.kl_max.max(kl);
        self.top1_agreement += f64::from(top1(reference) == top1(current));
        let delta = q[token].exp() - p[token].exp();
        self.forced_delta_p_mean += delta;
        self.forced_abs_delta_p_mean += delta.abs();
        self.forced_delta_p_max = self.forced_delta_p_max.max(delta.abs());
        self.mean_abs_delta_p += p
            .iter()
            .zip(q)
            .map(|(&p, q)| (q.exp() - p.exp()).abs())
            .sum::<f64>();
        Ok(())
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn record_rows(
    model: &CausalDecoder,
    prompt: &[usize],
    n: usize,
) -> Result<(Vec<usize>, Vec<Vec<f32>>)> {
    let (mut cache, state) = model.resident_reference_start(prompt, n, false)?;
    let mut logits = model.logits_from_final_state(&state)?.data().to_vec();
    let mut ids = Vec::with_capacity(n);
    let mut rows = Vec::with_capacity(n);
    for step in 0..n {
        log_softmax(&logits)?;
        let token = top1(&logits);
        ids.push(token);
        rows.push(logits);
        logits = if step + 1 < n {
            let raw = model.resident_reference_step(&mut cache, token)?;
            model
                .finalize_logits(&crate::Tensor::row(raw)?)?
                .data()
                .to_vec()
        } else {
            Vec::new()
        };
    }
    Ok((ids, rows))
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn compare_rows(model: &CausalDecoder, meta: &Metadata) -> Result<Vec<Vec<f32>>> {
    // La première distribution vient uniquement de la queue du prefill.
    // Le helper teacher forcing refait le prefill pour posséder son propre cache.
    let (_, state) = model.resident_reference_start(&meta.prompt_ids, meta.n, false)?;
    let mut rows = vec![model.logits_from_final_state(&state)?.data().to_vec()];
    let raw =
        model.resident_forced_logits(&meta.prompt_ids, &meta.forced_ids[..meta.n - 1], false)?;
    for row in raw {
        rows.push(
            model
                .finalize_logits(&crate::Tensor::row(row)?)?
                .data()
                .to_vec(),
        );
    }
    Ok(rows)
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
fn record_rows(_: &CausalDecoder, _: &[usize], _: usize) -> Result<(Vec<usize>, Vec<Vec<f32>>)> {
    Err(invalid("Metal résident requis"))
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
fn compare_rows(_: &CausalDecoder, _: &Metadata) -> Result<Vec<Vec<f32>>> {
    Err(invalid("Metal résident requis"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_identity_and_analytic_kl() {
        let p = log_softmax(&[0.75_f32.ln(), 0.25_f32.ln()]).expect("invariant: p");
        let q = log_softmax(&[0.25_f32.ln(), 0.75_f32.ln()]).expect("invariant: q");
        let mut same = Metrics::default();
        same.add(&p, &p, 0).expect("invariant: identité");
        assert_eq!(same.kl_mean, 0.0);
        assert_eq!(same.top1_agreement, 1.0);
        assert_eq!(same.mean_abs_delta_p, 0.0);
        let mut diff = Metrics::default();
        diff.add(&p, &q, 0).expect("invariant: métriques");
        assert!((diff.kl_mean - 0.5 * 3_f64.ln()).abs() < 1e-7);
        assert_eq!(diff.top1_agreement, 0.0);
        assert!((diff.forced_delta_p_mean + 0.5).abs() < 1e-7);
        assert!((diff.forced_delta_p_max - 0.5).abs() < 1e-7);
        assert!((diff.mean_abs_delta_p / 2.0 - 0.5).abs() < 1e-7);
        assert!((diff.forced_abs_delta_p_mean - 0.5).abs() < 1e-7);
        diff.add(&p, &q, 1).expect("invariant: écart opposé");
        assert!(diff.forced_delta_p_mean.abs() < 1e-7);
        assert!((diff.forced_abs_delta_p_mean / 2.0 - 0.5).abs() < 1e-7);
    }

    #[test]
    fn metrics_extreme_logits_and_invalid_values() {
        let p = log_softmax(&[10000.0, -10000.0]).expect("invariant: stable");
        assert_eq!(p, [0.0, -20000.0]);
        let mut metrics = Metrics::default();
        metrics
            .add(&p, &[-20000.0, 0.0], 1)
            .expect("invariant: stable KL");
        assert_eq!(metrics.kl_mean, 20000.0);
        assert!(log_softmax(&[f32::NAN]).is_err());
        assert!(log_softmax(&[]).is_err());
        assert!(metrics.add(&p, &[0.0], 0).is_err());
        assert!(metrics.add(&p, &p, 2).is_err());
        assert_eq!(top1(&[1.0, 1.0]), 0);
    }

    #[test]
    fn reference_rejects_truncated_incompatible_and_unbounded_files() {
        let dir = tempfile::tempdir().expect("invariant: répertoire temporaire");
        let path = dir.path();
        let mut meta = Metadata {
            version: VERSION,
            prompt_ids: vec![0],
            forced_ids: vec![1],
            vocab: 2,
            n: 1,
            config_sha256: "0".repeat(64),
            weights_sample_sha256: "1".repeat(64),
            transformations: TRANSFORMS.into(),
        };
        let save = |meta: &Metadata| {
            fs::write(
                path.join("ref.json"),
                serde_json::to_vec(meta).expect("invariant: JSON"),
            )
            .expect("invariant: écriture JSON")
        };
        save(&meta);
        let row = log_softmax(&[0.0, 1.0]).expect("invariant: distribution");
        let bytes: Vec<u8> = row.iter().flat_map(|x| x.to_le_bytes()).collect();
        fs::write(path.join("ref.bin"), &bytes).expect("invariant: binaire");
        assert!(read_metadata(path, 2).is_ok());
        assert_eq!(read_rows(path, &meta).expect("invariant: lecture"), [row]);
        assert!(read_metadata(path, 3).is_err());
        meta.version += 1;
        save(&meta);
        assert!(read_metadata(path, 2).is_err());
        meta.version = VERSION;
        meta.forced_ids[0] = 2;
        save(&meta);
        assert!(read_metadata(path, 2).is_err());
        meta.forced_ids[0] = 1;
        meta.n = usize::MAX;
        save(&meta);
        assert!(read_metadata(path, 2).is_err());
        meta.n = 1;
        save(&meta);
        fs::write(path.join("ref.bin"), &bytes[..7]).expect("invariant: troncature");
        assert!(read_metadata(path, 2).is_err());
        assert!(read_rows(path, &meta).is_err());
        fs::write(path.join("ref.bin"), [0; 8]).expect("invariant: valeurs invalides");
        assert!(read_rows(path, &meta).is_err());
        fs::write(path.join("ref.json"), vec![b' '; MAX_JSON as usize + 1])
            .expect("invariant: borne JSON");
        assert!(read_metadata(path, 2).is_err());
    }
}
