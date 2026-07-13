//! Résolution locale et cache Hugging Face pour `saragossa run`.

use std::collections::HashSet;
use std::env;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use hf_hub::api::sync::{ApiBuilder, ApiError};
use hf_hub::Cache;

use crate::{cli_error, next_value, CliResult};

const REQUIRED_FILES: &[&str] = &["config.json", "tokenizer.json"];
const OPTIONAL_FILES: &[&str] = &[
    "tokenizer_config.json",
    "chat_template.jinja",
    "model.safetensors.index.json",
];

/// Décrit un modèle déjà présent dans le cache HF.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct CachedModel {
    /// Identifiant Hugging Face reconstruit.
    pub(super) id: String,
    /// Snapshot local utilisable par `ModelAssets::load_local`.
    pub(super) snapshot: PathBuf,
    /// Taille totale des fichiers pointés par le snapshot.
    pub(super) size_bytes: u64,
}

/// Erreurs de résolution ou de téléchargement d'un modèle Hugging Face.
#[derive(Debug)]
pub(super) enum HfResolveError {
    InvalidModelRef(String),
    MissingHome,
    MissingRepoFile { repo: String, file: &'static str },
    MissingWeights(String),
    GatedModel { repo: String },
    Api { context: String, source: ApiError },
    Io { context: String, source: io::Error },
}

impl Display for HfResolveError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidModelRef(value) => write!(
                f,
                "modèle invalide: {value} — attendu un chemin local existant ou un id HF org/repo"
            ),
            Self::MissingHome => f.write_str(
                "cache Hugging Face introuvable: HF_HOME absent et HOME indisponible",
            ),
            Self::MissingRepoFile { repo, file } => {
                write!(f, "repo HF {repo}: fichier requis absent: {file}")
            }
            Self::MissingWeights(repo) => {
                write!(f, "repo HF {repo}: aucun shard racine *.safetensors trouvé")
            }
            Self::GatedModel { repo } => write!(
                f,
                "modèle gated ou privé: accepte la licence sur https://huggingface.co/{repo} puis exporte HF_TOKEN"
            ),
            Self::Api { context, source } => write!(f, "{context}: {source}"),
            Self::Io { context, source } => write!(f, "{context}: {source}"),
        }
    }
}

impl Error for HfResolveError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Api { source, .. } => Some(source),
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Résout un modèle local ou télécharge un repo HF absent.
pub(super) fn resolve_model(value: &str) -> Result<PathBuf, HfResolveError> {
    let path = PathBuf::from(value);
    if path.exists() {
        return Ok(path);
    }
    if !is_hf_model_id(value) {
        return Err(HfResolveError::InvalidModelRef(value.to_string()));
    }
    pull_hf_model(value)
}

/// Sous-commande `saragossa list`.
pub(super) fn run_list(args: impl IntoIterator<Item = String>) -> CliResult<()> {
    let mut cache_dir = None;
    let mut iter = args.into_iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                print_list_help();
                return Ok(());
            }
            "--cache-dir" => cache_dir = Some(PathBuf::from(next_value(&mut iter, "--cache-dir")?)),
            other => return Err(cli_error(format!("argument list inconnu: {other}"))),
        }
    }
    let cache_dir = match cache_dir {
        Some(path) => path,
        None => hf_cache_dir_from_env()?,
    };
    let models = list_cached_models(&cache_dir)?;
    if models.is_empty() {
        println!("aucun modèle prêt dans {}", cache_dir.display());
        return Ok(());
    }
    println!("{:<48} {:>12}", "MODEL", "SIZE");
    for model in models {
        println!("{:<48} {:>12}", model.id, human_size(model.size_bytes));
    }
    Ok(())
}

fn print_list_help() {
    println!(
        "Usage: saragossa list [--cache-dir DIR]\n\nListe les snapshots modèles du cache Hugging Face local contenant config.json."
    );
}

fn pull_hf_model(repo_id: &str) -> Result<PathBuf, HfResolveError> {
    let cache_dir = hf_cache_dir_from_env()?;
    let cache = Cache::new(cache_dir);
    let mut builder = ApiBuilder::from_cache(cache).with_progress(true);
    if let Ok(endpoint) = env::var("HF_ENDPOINT") {
        if !endpoint.trim().is_empty() {
            builder = builder.with_endpoint(endpoint);
        }
    }
    if let Some(token) = hf_token_from_env() {
        builder = builder.with_token(Some(token));
    }
    let api = builder.build().map_err(|source| HfResolveError::Api {
        context: "initialisation API Hugging Face".to_string(),
        source,
    })?;
    let repo = api.model(repo_id.to_string());
    let info = repo
        .info()
        .map_err(|source| map_hf_error(repo_id, "lecture metadata repo HF", source))?;
    let sibling_names = info
        .siblings
        .iter()
        .map(|sibling| sibling.rfilename.clone())
        .collect::<Vec<_>>();
    let files = files_to_download(repo_id, &sibling_names)?;
    let mut snapshot = None;
    for file in files {
        eprintln!("saragossa pull {repo_id}/{file}");
        let path = repo
            .get(&file)
            .map_err(|source| map_hf_error(repo_id, format!("téléchargement {file}"), source))?;
        if snapshot.is_none() {
            snapshot = path.parent().map(Path::to_path_buf);
        }
    }
    snapshot.ok_or_else(|| HfResolveError::MissingWeights(repo_id.to_string()))
}

fn map_hf_error(repo_id: &str, context: impl Into<String>, source: ApiError) -> HfResolveError {
    if is_hf_auth_error(&source) {
        HfResolveError::GatedModel {
            repo: repo_id.to_string(),
        }
    } else {
        HfResolveError::Api {
            context: context.into(),
            source,
        }
    }
}

fn is_hf_auth_error(error: &ApiError) -> bool {
    let rendered = error.to_string();
    rendered.contains("http status: 401") || rendered.contains("http status: 403")
}

fn hf_token_from_env() -> Option<String> {
    env_token("HF_TOKEN").or_else(|| env_token("HUGGING_FACE_HUB_TOKEN"))
}

fn env_token(name: &str) -> Option<String> {
    env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

/// Indique si la référence ressemble à un id Hugging Face `org/repo`.
pub(super) fn is_hf_model_id(value: &str) -> bool {
    if value.contains("://") || value.contains('\\') || value.starts_with('/') {
        return false;
    }
    let parts = value.split('/').collect::<Vec<_>>();
    if parts.len() != 2 {
        return false;
    }
    parts.iter().all(|part| {
        !part.is_empty()
            && *part != "."
            && *part != ".."
            && !part.starts_with('.')
            && !part.ends_with('.')
    })
}

fn files_to_download(repo_id: &str, siblings: &[String]) -> Result<Vec<String>, HfResolveError> {
    let names = siblings.iter().map(String::as_str).collect::<HashSet<_>>();
    let mut selected = Vec::new();
    for required in REQUIRED_FILES {
        if !names.contains(required) {
            return Err(HfResolveError::MissingRepoFile {
                repo: repo_id.to_string(),
                file: required,
            });
        }
        selected.push((*required).to_string());
    }
    for optional in OPTIONAL_FILES {
        if names.contains(optional) {
            selected.push((*optional).to_string());
        }
    }
    let mut weights = siblings
        .iter()
        .filter(|name| is_root_safetensor(name))
        .cloned()
        .collect::<Vec<_>>();
    if weights.is_empty() {
        return Err(HfResolveError::MissingWeights(repo_id.to_string()));
    }
    weights.sort();
    selected.extend(weights);
    selected.sort();
    selected.dedup();
    Ok(selected)
}

fn is_root_safetensor(name: &str) -> bool {
    !name.contains('/') && name.ends_with(".safetensors")
}

/// Renvoie le répertoire du cache HF (`HF_HOME`/hub, sinon `~/.cache/huggingface/hub`).
///
/// # Errors
///
/// Renvoie une erreur si aucun répertoire personnel n'est résolu.
pub(super) fn hf_cache_dir_from_env() -> Result<PathBuf, HfResolveError> {
    if let Ok(home) = env::var("HF_HOME") {
        if !home.trim().is_empty() {
            return Ok(PathBuf::from(home).join("hub"));
        }
    }
    let home = env::var_os("HOME").ok_or(HfResolveError::MissingHome)?;
    Ok(PathBuf::from(home)
        .join(".cache")
        .join("huggingface")
        .join("hub"))
}

/// Liste les snapshots HF locaux porteurs d'un `config.json`, avec leur taille.
///
/// # Errors
///
/// Renvoie une erreur si le cache est illisible.
pub(super) fn list_cached_models(cache_dir: &Path) -> Result<Vec<CachedModel>, HfResolveError> {
    if !cache_dir.exists() {
        return Ok(Vec::new());
    }
    let mut models = Vec::new();
    for repo_entry in read_dir(cache_dir, "lecture cache Hugging Face")? {
        let repo_entry = repo_entry.map_err(|source| HfResolveError::Io {
            context: format!("lecture entrée {}", cache_dir.display()),
            source,
        })?;
        let repo_name = repo_entry.file_name();
        let Some(repo_name) = repo_name.to_str() else {
            continue;
        };
        let Some(id) = repo_id_from_cache_folder(repo_name) else {
            continue;
        };
        let snapshots_dir = repo_entry.path().join("snapshots");
        if !snapshots_dir.is_dir() {
            continue;
        }
        for snapshot_entry in read_dir(&snapshots_dir, "lecture snapshots Hugging Face")? {
            let snapshot_entry = snapshot_entry.map_err(|source| HfResolveError::Io {
                context: format!("lecture entrée {}", snapshots_dir.display()),
                source,
            })?;
            let snapshot = snapshot_entry.path();
            if !snapshot.join("config.json").exists() {
                continue;
            }
            let size_bytes = dir_size_following_links(&snapshot)?;
            models.push(CachedModel {
                id: id.clone(),
                snapshot,
                size_bytes,
            });
        }
    }
    models.sort_by(|left, right| {
        left.id
            .cmp(&right.id)
            .then_with(|| left.snapshot.cmp(&right.snapshot))
    });
    Ok(models)
}

fn read_dir(path: &Path, context: &'static str) -> Result<fs::ReadDir, HfResolveError> {
    fs::read_dir(path).map_err(|source| HfResolveError::Io {
        context: format!("{context}: {}", path.display()),
        source,
    })
}

/// Reconstruit l'id `org/repo` depuis un dossier de cache `models--org--repo`.
pub(super) fn repo_id_from_cache_folder(name: &str) -> Option<String> {
    let encoded = name.strip_prefix("models--")?;
    if encoded.is_empty() {
        return None;
    }
    let (namespace, repo) = encoded
        .split_once("--")
        .map_or((encoded, None), |(namespace, repo)| (namespace, Some(repo)));
    if namespace.is_empty() || repo.is_some_and(str::is_empty) {
        return None;
    }
    Some(match repo {
        Some(repo) => format!("{namespace}/{repo}"),
        None => namespace.to_string(),
    })
}

fn dir_size_following_links(root: &Path) -> Result<u64, HfResolveError> {
    let mut total = 0_u64;
    let mut stack = vec![root.to_path_buf()];
    while let Some(path) = stack.pop() {
        for entry in read_dir(&path, "calcul taille snapshot HF")? {
            let entry = entry.map_err(|source| HfResolveError::Io {
                context: format!("lecture entrée {}", path.display()),
                source,
            })?;
            let entry_path = entry.path();
            let metadata = fs::metadata(&entry_path).map_err(|source| HfResolveError::Io {
                context: format!("stat {}", entry_path.display()),
                source,
            })?;
            if metadata.is_dir() {
                stack.push(entry_path);
            } else if metadata.is_file() {
                total = total.saturating_add(metadata.len());
            }
        }
    }
    Ok(total)
}

fn human_size(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0_usize;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_hf_ids_without_accepting_paths() {
        assert!(is_hf_model_id("mlx-community/Qwen3-4B"));
        assert!(!is_hf_model_id("Qwen3-4B"));
        assert!(!is_hf_model_id("/models/Qwen3-4B"));
        assert!(!is_hf_model_id("org/repo/extra"));
        assert!(!is_hf_model_id("https://huggingface.co/org/repo"));
    }

    #[test]
    fn resolve_model_prefers_existing_local_path() {
        let temp = tempfile::tempdir().expect("invariant: tempdir disponible");
        let resolved = resolve_model(
            temp.path()
                .to_str()
                .expect("invariant: chemin temporaire UTF-8"),
        )
        .expect("invariant: chemin local existant accepté");

        assert_eq!(resolved, temp.path());
    }

    #[test]
    fn reconstructs_repo_id_from_cache_folder() {
        assert_eq!(
            repo_id_from_cache_folder("models--mlx-community--Qwen3-4B").as_deref(),
            Some("mlx-community/Qwen3-4B")
        );
        assert_eq!(
            repo_id_from_cache_folder("models--gpt2").as_deref(),
            Some("gpt2")
        );
        assert_eq!(repo_id_from_cache_folder("datasets--org--repo"), None);
    }

    #[test]
    fn lists_models_from_fake_cache() {
        let temp = tempfile::tempdir().expect("invariant: tempdir disponible");
        let snapshot = temp
            .path()
            .join("models--org--repo")
            .join("snapshots")
            .join("abc123");
        fs::create_dir_all(&snapshot).expect("invariant: dossier snapshot créé");
        fs::write(snapshot.join("config.json"), b"{}").expect("invariant: config test écrite");
        fs::write(snapshot.join("tokenizer.json"), b"tok")
            .expect("invariant: tokenizer test écrit");
        fs::write(snapshot.join("model.safetensors"), vec![0_u8; 7])
            .expect("invariant: shard test écrit");

        let models = list_cached_models(temp.path()).expect("invariant: cache test lisible");

        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "org/repo");
        assert_eq!(models[0].size_bytes, 12);
    }

    #[test]
    fn selects_required_optional_and_root_weights() {
        let siblings = vec![
            "config.json".to_string(),
            "tokenizer.json".to_string(),
            "tokenizer_config.json".to_string(),
            "model.safetensors.index.json".to_string(),
            "model-00002-of-00002.safetensors".to_string(),
            "model-00001-of-00002.safetensors".to_string(),
            "onnx/model.safetensors".to_string(),
        ];

        let files = files_to_download("org/repo", &siblings)
            .expect("invariant: fichiers minimaux présents");

        assert!(files.contains(&"config.json".to_string()));
        assert!(files.contains(&"tokenizer.json".to_string()));
        assert!(files.contains(&"tokenizer_config.json".to_string()));
        assert!(files.contains(&"model.safetensors.index.json".to_string()));
        assert!(files.contains(&"model-00001-of-00002.safetensors".to_string()));
        assert!(files.contains(&"model-00002-of-00002.safetensors".to_string()));
        assert!(!files.contains(&"onnx/model.safetensors".to_string()));
    }
}
