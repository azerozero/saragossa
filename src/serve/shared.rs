//! Poids partagés entre les moteurs de `serve --engines N`.
//!
//! Chaque moteur garde son `ServeState` (KV, scratch, caches de préfixe) mais
//! obtient ses modèles par ce registre : le premier chargement d'un chemin est
//! servi aux autres moteurs par le même `Arc`. Le registre ne retient qu'un
//! `Weak` : les poids tombent quand plus aucun moteur ne les tient (idle
//! unload, LRU, SIGHUP, panique restent des décisions par moteur).

use std::any::Any;
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex, Weak};

use super::error::ServeResult;

type Slot = Weak<dyn Any + Send + Sync>;

/// Registre clonable (un `Arc` interne) : un par process serve.
#[derive(Clone, Default)]
pub(super) struct SharedWeights {
    slots: Arc<Mutex<HashMap<(&'static str, PathBuf), Slot>>>,
}

impl SharedWeights {
    /// Renvoie les poids `kind` de `path`, chargés par `load` s'ils ne sont plus
    /// tenus par aucun moteur. `load` s'exécute sous le verrou du registre :
    /// l'admission mémoire qu'il fait avant de charger ne peut pas croiser
    /// celle d'un autre moteur.
    ///
    /// La clé est le chemin tel que donné : preset et profil de kernels se
    /// déduisent du chemin, deux alias d'un même dossier restent distincts.
    ///
    /// ponytail: verrou global tenu pendant le chargement (un moteur qui veut
    /// un autre modèle attend) ; verrou par clé si les chargements concurrents
    /// de modèles différents comptent.
    ///
    /// # Errors
    ///
    /// Renvoie l'erreur de `load`.
    pub(super) fn get_or_load<T: Send + Sync + 'static>(
        &self,
        kind: &'static str,
        path: &Path,
        load: impl FnOnce() -> ServeResult<T>,
    ) -> ServeResult<Arc<T>> {
        // Une panique pendant `load` n'a rien inséré : la table reste cohérente.
        let mut slots = self
            .slots
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let key = (kind, path.to_path_buf());
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
}

/// Ce que les moteurs d'un process ont en commun : poids, sessions STT et
/// garde mémoire (une seule réservation pour tous les moteurs).
#[derive(Clone)]
pub(super) struct EngineShared {
    pub(super) weights: SharedWeights,
    pub(super) memory_guard: saragossa::MemoryGuard,
    #[cfg(feature = "stt")]
    pub(super) stt_sessions: Arc<Mutex<super::stt::SttSessions>>,
    pub(super) stt_order: Arc<SttOrder>,
}

impl Default for EngineShared {
    fn default() -> Self {
        Self {
            weights: SharedWeights::default(),
            memory_guard: saragossa::MemoryGuard::serve(),
            #[cfg(feature = "stt")]
            stt_sessions: Arc::default(),
            stt_order: Arc::default(),
        }
    }
}

/// Ordre FIFO des morceaux STT streamés entre moteurs : un ticket est émis
/// sous le verrou de la file commune (ordre d'arrivée), le morceau attend son
/// tour avant de toucher aux sessions. À un seul moteur, aucun ticket.
#[derive(Default)]
pub(super) struct SttOrder {
    /// (prochain ticket émis, ticket servi, tickets finis en avance).
    state: Mutex<(u64, u64, BTreeSet<u64>)>,
    turn: Condvar,
}

/// Place d'un morceau STT dans [`SttOrder`] ; son drop passe la main, même
/// si le job échoue avant d'atteindre les sessions.
pub(super) struct SttTicket {
    order: Arc<SttOrder>,
    id: u64,
}

impl SttOrder {
    pub(super) fn issue(self: &Arc<Self>) -> SttTicket {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let id = state.0;
        state.0 += 1;
        SttTicket {
            order: Arc::clone(self),
            id,
        }
    }
}

impl SttTicket {
    /// Attend que tous les morceaux arrivés avant celui-ci soient finis.
    pub(super) fn wait_turn(&self) {
        let mut state = self
            .order
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        while state.1 != self.id {
            state = self
                .order
                .turn
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }
}

impl Drop for SttTicket {
    fn drop(&mut self) {
        let mut state = self
            .order
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.2.insert(self.id);
        loop {
            let serving = state.1;
            if !state.2.remove(&serving) {
                break;
            }
            state.1 += 1;
        }
        drop(state);
        self.order.turn.notify_all();
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
        let failed = shared.get_or_load::<u32>("stt", path, || {
            Err(super::super::error::ServeError::args("absent"))
        });
        assert!(failed.is_err());
        assert_eq!(
            shared
                .get_or_load("stt", path, || Ok(3_u32))
                .ok()
                .as_deref(),
            Some(&3)
        );
    }

    #[test]
    fn panic_while_loading_keeps_the_registry_usable() {
        let shared = SharedWeights::default();
        let path = Path::new("/panique");
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = shared.get_or_load::<u32>("llm", path, || panic!("chargement"));
        }));
        assert!(caught.is_err());
        let reloaded = shared.get_or_load("llm", path, || Ok(5_u32));
        assert_eq!(reloaded.ok().as_deref(), Some(&5));
    }

    #[test]
    fn stt_tickets_take_turns_in_issue_order() {
        let order = Arc::new(SttOrder::default());
        let (first, second, third) = (order.issue(), order.issue(), order.issue());
        let log = Mutex::new(Vec::new());
        std::thread::scope(|scope| {
            for ticket in [third, second] {
                let log = &log;
                scope.spawn(move || {
                    ticket.wait_turn();
                    log.lock().expect("invariant: journal sain").push(ticket.id);
                });
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
            assert!(log.lock().expect("invariant: journal sain").is_empty());
            // Fini sans avoir servi (job en erreur) : la main passe quand même.
            drop(first);
        });
        assert_eq!(*log.lock().expect("invariant: journal sain"), [1, 2]);
    }
}
