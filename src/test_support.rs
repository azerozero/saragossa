/// Les tests ordinaires sautent uniquement l'absence de device. La suite GPU
/// explicite exige son exécution et conserve toutes les autres erreurs.
#[cfg(gpu_metal)]
pub(crate) fn metal_executor() -> crate::Result<Option<crate::MetalExecutor>> {
    match crate::MetalExecutor::new() {
        Ok(executor) => Ok(Some(executor)),
        Err(crate::InferError::Metal(message))
            if message.contains("aucun device")
                && crate::runtime_flags::env_var("SARAGOSSA_REQUIRE_METAL").as_deref()
                    != Some("1") =>
        {
            Ok(None)
        }
        Err(error) => Err(error),
    }
}

use std::ffi::{OsStr, OsString};
use std::sync::{Mutex, MutexGuard};

static ENV_LOCK: Mutex<()> = Mutex::new(());

/// Verrou d'environnement unique des tests : sérialise les écritures et
/// restaure chaque variable touchée à la sortie, même sur panique.
///
/// NOTE: les tests qui seulement lisent des flags ne le prennent pas ; il
/// protège les écrivains les uns des autres et supprime les fuites.
pub(crate) struct EnvGuard {
    _lock: MutexGuard<'static, ()>,
    saved: Vec<(OsString, Option<OsString>)>,
}

/// Prend le verrou d'environnement ; un verrou empoisonné reste utilisable.
pub(crate) fn env_guard() -> EnvGuard {
    EnvGuard {
        _lock: ENV_LOCK.lock().unwrap_or_else(|poison| poison.into_inner()),
        saved: Vec::new(),
    }
}

impl EnvGuard {
    fn remember(&mut self, name: &OsStr) {
        if !self.saved.iter().any(|(saved, _)| saved == name) {
            self.saved.push((name.to_owned(), std::env::var_os(name)));
        }
    }

    pub(crate) fn set(&mut self, name: impl AsRef<OsStr>, value: impl AsRef<OsStr>) {
        self.remember(name.as_ref());
        std::env::set_var(name, value);
    }

    pub(crate) fn remove(&mut self, name: impl AsRef<OsStr>) {
        self.remember(name.as_ref());
        std::env::remove_var(name);
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        for (name, value) in self.saved.drain(..).rev() {
            match value {
                Some(value) => std::env::set_var(&name, value),
                None => std::env::remove_var(&name),
            }
        }
    }
}

use std::path::PathBuf;

/// Résout une fixture voix de référence (`reti-fr.wav`, `reti-fr.txt`).
///
/// La copie embarquée dans `assets/voices/` est cherchée d'abord : c'est la
/// seule que le miroir public reçoit, `scripts/publish-saragossa.sh` ne
/// synchronisant que `crates/saragossa`. Le repli sur `../../voices/` garde
/// l'arbre reti comme source unique historique tant que les deux coexistent.
///
/// Renvoie `None` si aucune des deux n'existe, pour que les tests concernés
/// se sautent au lieu d'échouer.
pub(crate) fn voice_fixture(name: &str) -> Option<PathBuf> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let embedded = manifest.join("assets/voices").join(name);
    if embedded.is_file() {
        return Some(embedded);
    }
    let workspace = manifest.join("../../voices").join(name);
    workspace.is_file().then_some(workspace)
}

/// Répertoire local désigné par `var` ; `None` (test sauté, ou échec en mode
/// strict) si la variable est absente, vide ou ne désigne pas un répertoire.
pub(crate) fn local_dir(var: &str) -> Option<PathBuf> {
    let dir = std::env::var_os(var)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .filter(|path| path.is_dir());
    if dir.is_none() {
        eprintln!("skip: {var} absent ou introuvable (test sur poids locaux)");
    }
    require_real_model(dir, var)
}

/// Exige une précondition de modèle réel quand le mode strict est actif.
pub(crate) fn require_real_model<T>(candidate: Option<T>, missing: &str) -> Option<T> {
    require_real_model_when(
        candidate,
        crate::runtime_flags::env_var("SARAGOSSA_REQUIRE_REAL_MODELS").as_deref() == Some("1"),
        missing,
    )
}

fn require_real_model_when<T>(candidate: Option<T>, required: bool, missing: &str) -> Option<T> {
    assert!(
        !required || candidate.is_some(),
        "SARAGOSSA_REQUIRE_REAL_MODELS=1: précondition absente: {missing}"
    );
    candidate
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_real_model_still_skips_without_strict_mode() {
        assert_eq!(require_real_model_when(None::<()>, false, "fixture"), None);
    }

    #[test]
    #[should_panic(expected = "SARAGOSSA_REQUIRE_REAL_MODELS=1: précondition absente: fixture")]
    fn missing_real_model_fails_in_strict_mode() {
        let _ = require_real_model_when(None::<()>, true, "fixture");
    }

    #[test]
    fn voice_fixture_resolves_embedded_copy() {
        // La copie embarquée est la seule que reçoit le miroir public : si elle
        // manque, les tests de parité TTS s'y sautent tous en silence.
        let wav = voice_fixture("reti-fr.wav").expect("invariant: fixture wav embarquée");
        assert!(
            wav.ends_with("assets/voices/reti-fr.wav"),
            "résolu: {wav:?}"
        );
        assert!(voice_fixture("reti-fr.txt").is_some());
        assert_eq!(voice_fixture("absente-du-disque.wav"), None);
    }

    #[test]
    fn embedded_voice_fixture_matches_workspace_copy() {
        // Deux copies d'un même fichier divergent tôt ou tard : sans cette
        // garde, reti et le miroir public testeraient des voix différentes.
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        for name in ["reti-fr.wav", "reti-fr.txt"] {
            let workspace = manifest.join("../../voices").join(name);
            if !workspace.is_file() {
                continue; // arbre standalone : une seule copie, rien à comparer
            }
            let embedded = manifest.join("assets/voices").join(name);
            assert_eq!(
                std::fs::read(&embedded).ok(),
                std::fs::read(&workspace).ok(),
                "{name} diverge entre assets/voices et voices/"
            );
        }
    }

    #[test]
    fn env_guard_restores_previous_values() {
        const SET: &str = "SARAGOSSA_TEST_GUARD_SET";
        const NEW: &str = "SARAGOSSA_TEST_GUARD_NEW";
        // Noms propres à ce test : aucun autre test ne les lit.
        std::env::set_var(SET, "a");
        {
            let mut env = env_guard();
            env.set(NEW, "x");
            env.set(SET, "b");
            env.set(SET, "c");
            env.remove(NEW);
            assert_eq!(std::env::var(SET).as_deref(), Ok("c"));
            assert!(std::env::var_os(NEW).is_none());
        }
        assert_eq!(std::env::var(SET).as_deref(), Ok("a"));
        assert!(std::env::var_os(NEW).is_none());
        std::env::remove_var(SET);
    }
}
