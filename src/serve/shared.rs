//! Poids partagés entre les moteurs de `serve --engines N`.
//!
//! Chaque moteur garde son `ServeState` (KV, scratch, caches de préfixe) mais
//! obtient ses modèles par ce registre : le premier chargement d'un chemin est
//! servi aux autres moteurs par le même `Arc`. Le registre ne retient qu'un
//! `Weak` : les poids tombent quand plus aucun moteur ne les tient (idle
//! unload, LRU, SIGHUP, panique restent des décisions par moteur).

use std::any::Any;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, Weak};

use super::error::{ServeError, ServeResult};

type Slot = Weak<dyn Any + Send + Sync>;

/// Registre clonable (un `Arc` interne) : un par process serve.
#[derive(Clone, Default)]
pub(super) struct SharedWeights {
    slots: Arc<Mutex<HashMap<(&'static str, PathBuf), Slot>>>,
}

impl SharedWeights {
    /// Renvoie les poids `kind` de `path`, chargés par `load` s'ils ne sont plus
    /// tenus par aucun moteur.
    ///
    /// ponytail: verrou global tenu pendant le chargement (un moteur qui veut
    /// un autre modèle attend) ; verrou par clé si les chargements concurrents
    /// de modèles différents comptent.
    ///
    /// # Errors
    ///
    /// Renvoie l'erreur de `load`, ou une erreur si le registre est empoisonné.
    pub(super) fn get_or_load<T: Send + Sync + 'static>(
        &self,
        kind: &'static str,
        path: &Path,
        load: impl FnOnce() -> ServeResult<T>,
    ) -> ServeResult<Arc<T>> {
        let mut slots = self
            .slots
            .lock()
            .map_err(|_| ServeError::args("registre des poids partagés empoisonné"))?;
        let key = (kind, canonical(path));
        if let Some(shared) = slots.get(&key).and_then(Weak::upgrade) {
            if let Ok(shared) = shared.downcast::<T>() {
                return Ok(shared);
            }
        }
        let loaded = Arc::new(load()?);
        let erased: Arc<dyn Any + Send + Sync> = loaded.clone();
        slots.insert(key, Arc::downgrade(&erased));
        // Les entrées mortes d'autres chemins ne coûtent qu'une clé ; on les
        // retire au passage pour borner la table aux modèles vivants.
        slots.retain(|_, slot| slot.strong_count() > 0);
        Ok(loaded)
    }

    /// Les poids `kind` de `path` sont-ils tenus par un moteur ?
    pub(super) fn is_alive(&self, kind: &'static str, path: &Path) -> bool {
        self.slots.lock().is_ok_and(|slots| {
            slots
                .get(&(kind, canonical(path)))
                .is_some_and(|slot| slot.strong_count() > 0)
        })
    }
}

/// Deux chemins vers le même dossier partagent leurs poids ; un chemin
/// illisible garde sa forme (le chargement rendra l'erreur).
fn canonical(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

/// Ce que les moteurs d'un process ont en commun : poids, sessions STT et
/// garde mémoire (une seule réservation pour tous les moteurs).
#[derive(Clone)]
pub(super) struct EngineShared {
    pub(super) weights: SharedWeights,
    pub(super) memory_guard: saragossa::MemoryGuard,
    #[cfg(feature = "stt")]
    pub(super) stt_sessions: Arc<Mutex<super::stt::SttSessions>>,
}

impl Default for EngineShared {
    fn default() -> Self {
        Self {
            weights: SharedWeights::default(),
            memory_guard: saragossa::MemoryGuard::serve(),
            #[cfg(feature = "stt")]
            stt_sessions: Arc::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_once_while_held_and_reloads_after_release() -> ServeResult<()> {
        let shared = SharedWeights::default();
        let path = Path::new("/modele");
        let mut loads = 0;
        let first = shared.get_or_load("llm", path, || {
            loads += 1;
            Ok(7_u32)
        })?;
        let second = shared.get_or_load("llm", path, || {
            loads += 1;
            Ok(8_u32)
        })?;
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(loads, 1);
        // Autre famille sur le même chemin : chargement distinct.
        let other = shared.get_or_load("tts", path, || Ok(9_u32))?;
        assert_eq!(*other, 9);
        drop((first, second));
        let reloaded = shared.get_or_load("llm", path, || {
            loads += 1;
            Ok(10_u32)
        })?;
        assert_eq!((*reloaded, loads), (10, 2));
        Ok(())
    }

    #[test]
    fn load_error_leaves_no_entry() {
        let shared = SharedWeights::default();
        let path = Path::new("/absent");
        let failed = shared.get_or_load::<u32>("stt", path, || Err(ServeError::args("absent")));
        assert!(failed.is_err());
        assert_eq!(
            shared
                .get_or_load("stt", path, || Ok(3_u32))
                .ok()
                .as_deref(),
            Some(&3)
        );
    }
}
