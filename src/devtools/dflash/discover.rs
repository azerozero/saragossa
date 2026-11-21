//! Découverte du drafter DFlash sans variable d'environnement.
//!
//! Ordre : `SARAGOSSA_RUST_DFLASH_DRAFT_DIR` (vide ou `0` = désactivé), puis le
//! sous-dossier `dflash/` du modèle, puis le cache Hugging Face. Un drafter
//! découvert doit valider [`load_dflash_draft_for_target`] et déclarer le même
//! vocabulaire que le tronc, sinon il est ignoré avec un avertissement.

use super::load_dflash_draft_for_target;
use crate::ModelConfig;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};

/// Variable qui désigne, ou désactive, le drafter.
pub const DFLASH_DRAFT_DIR_ENV: &str = "SARAGOSSA_RUST_DFLASH_DRAFT_DIR";
/// Sous-dossier conventionnel du modèle qui contient le drafter.
pub const DFLASH_SIBLING_DIR: &str = "dflash";

/// Drafter retenu et phrase de log qui explique le choix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DraftResolution {
    /// Dossier du drafter, `None` pour décoder sans lui.
    pub dir: Option<PathBuf>,
    /// Explication prête à journaliser.
    pub note: String,
    /// Autres drafters compatibles, à essayer si le chargement de `dir` échoue.
    pub fallbacks: Vec<PathBuf>,
    /// Vrai si la variable désigne le drafter : DFlash est alors forcé.
    pub forced: bool,
}

/// Un drafter découvert (non forcé) n'est activé que sur une cible dense BF16 :
/// mesures du 21 nov., DFlash y bat MTP (24,7 contre 21-22 tok/s) mais fait
/// jeu égal en 4 bits et perd sur MoE.
pub fn dflash_beats_mtp(forced: bool, dense_bf16: bool) -> bool {
    forced || dense_bf16
}

/// Ligne de log quand un drafter découvert est écarté au profit de MTP.
pub const DFLASH_SKIPPED_NOTE: &str =
    "drafter DFlash disponible mais MTP plus rapide sur cette cible \
     (forcer avec SARAGOSSA_RUST_DFLASH_DRAFT_DIR)";

/// Résout le drafter du modèle `model_dir` (cible `target`).
///
/// `env` est la valeur de [`DFLASH_DRAFT_DIR_ENV`] ; `hf_cache` le dossier du
/// hub Hugging Face, ignoré s'il est `None`.
pub fn resolve_dflash_draft(
    model_dir: &Path,
    target: &ModelConfig,
    env: Option<&OsStr>,
    hf_cache: Option<&Path>,
) -> DraftResolution {
    if let Some(value) = env {
        if value.is_empty() || value == "0" {
            return DraftResolution {
                dir: None,
                note: format!("DFlash : désactivé par {DFLASH_DRAFT_DIR_ENV}"),
                forced: true,
                fallbacks: Vec::new(),
            };
        }
        let dir = PathBuf::from(value);
        return DraftResolution {
            note: format!(
                "DFlash : drafter {} ({DFLASH_DRAFT_DIR_ENV})",
                dir.display()
            ),
            dir: Some(dir),
            forced: true,
            fallbacks: Vec::new(),
        };
    }
    let mut warnings = Vec::new();
    let mut hf = hf_cache.map_or_else(Vec::new, |cache| find_in_hf_cache(cache, model_dir, target));
    let sibling = model_dir.join(DFLASH_SIBLING_DIR);
    if sibling.is_dir() {
        match check_compatible(target, &sibling) {
            Ok(()) => {
                return DraftResolution {
                    note: format!(
                        "DFlash : drafter {} (sous-dossier du modèle)",
                        sibling.display()
                    ),
                    dir: Some(sibling),
                    forced: false,
                    fallbacks: hf,
                }
            }
            Err(why) => warnings.push(format!("{} ignoré ({why})", sibling.display())),
        }
    }
    if !hf.is_empty() {
        let hit = hf.remove(0);
        return DraftResolution {
            note: format!("DFlash : drafter {} (cache Hugging Face)", hit.display()),
            dir: Some(hit),
            forced: false,
            fallbacks: hf,
        };
    }
    let mut note = format!(
        "aucun drafter DFlash : décodage MTP/AR, gain possible avec un drafter DFlash 2 dans {} \
         ou via {DFLASH_DRAFT_DIR_ENV}",
        sibling.display()
    );
    for warning in warnings {
        note.push_str(&format!(" ; avertissement : drafter {warning}"));
    }
    DraftResolution {
        dir: None,
        note,
        forced: false,
        fallbacks: Vec::new(),
    }
}

impl DraftResolution {
    /// Drafters à essayer dans l'ordre : `dir`, puis les repli.
    pub fn candidates(&self) -> impl Iterator<Item = &Path> {
        self.dir.iter().chain(&self.fallbacks).map(PathBuf::as_path)
    }
}

/// Premier candidat que `load` accepte ; chaque échec est journalisé.
pub fn first_loadable<'a, T, E: std::fmt::Display>(
    candidates: impl IntoIterator<Item = &'a Path>,
    mut load: impl FnMut(&Path) -> Result<T, E>,
) -> Option<(&'a Path, T)> {
    candidates.into_iter().find_map(|dir| match load(dir) {
        Ok(value) => Some((dir, value)),
        Err(err) => {
            eprintln!(
                "saragossa DFlash : drafter {} inutilisable ({err})",
                dir.display()
            );
            None
        }
    })
}

/// Hidden size, couches, vocabulaire : le drafter peut-il servir `target` ?
fn check_compatible(target: &ModelConfig, dir: &Path) -> Result<(), String> {
    let info = load_dflash_draft_for_target(target, dir).map_err(|err| err.to_string())?;
    if info.vocab_size != 0 && info.vocab_size != target.vocab_size {
        return Err(format!(
            "vocab_size={} incompatible avec le tronc ({})",
            info.vocab_size, target.vocab_size
        ));
    }
    // Le loader exige ces tenseurs : sans eux le chargement échouerait après coup.
    let keys =
        crate::catalog::read_safetensors_keys(&info.weight_path).map_err(|err| err.to_string())?;
    for key in super::dflash_required_keys(&info) {
        if !keys.contains(&key) {
            return Err(format!("tenseur {key} absent du checkpoint"));
        }
    }
    Ok(())
}

/// Le nom du modèle est-il `base` seul, ou `base` suivi de suffixes de
/// quantification reconnus (`-4bit`, `-bf16`, `-mlx`…) ? Écarte `-instruct`
/// et les dérivés.
fn is_same_model(name: &str, base: &str) -> bool {
    let Some(rest) = name.strip_prefix(base) else {
        return false;
    };
    rest.is_empty()
        || rest.strip_prefix('-').is_some_and(|rest| {
            rest.split('-').all(|part| {
                matches!(
                    part,
                    "mlx" | "bf16" | "fp16" | "fp8" | "nvfp4" | "mxfp4" | "mxfp8"
                ) || part
                    .strip_suffix("bit")
                    .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
            })
        })
}

/// Nom du modèle en minuscules : dernier segment d'un snapshot HF
/// (`models--org--nom/snapshots/rev`) ou nom du dossier.
fn model_name(model_dir: &Path) -> String {
    let parts: Vec<_> = model_dir.iter().filter_map(OsStr::to_str).collect();
    let cached = parts
        .iter()
        .rposition(|part| part.starts_with("models--"))
        .and_then(|idx| parts[idx].rsplit("--").next());
    cached
        .or_else(|| parts.last().copied())
        .unwrap_or_default()
        .to_lowercase()
}

/// `base_model` déclaré par le README d'un drafter, minuscules, sans organisation.
fn base_model_name(snapshot: &Path) -> Option<String> {
    let readme = std::fs::read_to_string(snapshot.join("README.md")).ok()?;
    let mut lines = readme
        .lines()
        .skip_while(|line| line.trim_end() != "base_model:");
    lines.next()?;
    let entry = lines.next()?.trim().strip_prefix("- ")?;
    Some(entry.rsplit('/').next()?.trim().to_lowercase())
}

/// Snapshots HF compatibles (ordre lexicographique) dont le README déclare une base
/// identifie le modèle (`Qwen3.8-27B` pour `Qwen3.8-27B-4bit`, pas `-Instruct`).
fn find_in_hf_cache(cache: &Path, model_dir: &Path, target: &ModelConfig) -> Vec<PathBuf> {
    let name = model_name(model_dir);
    let mut snapshots = Vec::new();
    for repo in std::fs::read_dir(cache).into_iter().flatten().flatten() {
        if !repo.file_name().to_string_lossy().starts_with("models--") {
            continue;
        }
        for snap in std::fs::read_dir(repo.path().join("snapshots"))
            .into_iter()
            .flatten()
            .flatten()
        {
            snapshots.push(snap.path());
        }
    }
    snapshots.sort();
    snapshots.retain(|snap| {
        base_model_name(snap).is_some_and(|base| !base.is_empty() && is_same_model(&name, &base))
            && check_compatible(target, snap).is_ok()
    });
    snapshots
}
