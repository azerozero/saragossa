//! Arbre de la vérification en lignes (DFlash 2, `SARAGOSSA_RUST_DFLASH_TREE`).
//!
//! La vérification encode ses lignes comme d'habitude ; seuls les kernels qui
//! supposent une chaîne (RoPE à `position + ligne`, attention causale, conv k4
//! sur les lignes précédentes, scan gated-delta séquentiel) lisent l'arbre
//! installé ici le temps de l'encodage. Hors de ce scope rien ne change. Une
//! chaîne `parent = ligne − 1` redonne exactement l'arithmétique historique.

use crate::{InferError, Result};
use std::cell::RefCell;
use std::rc::Rc;

/// Topologie aplatie de l'arbre (lignes en ordre topologique, ligne 0 = racine).
#[derive(Debug)]
pub(crate) struct VerifyTree {
    parents: Vec<u32>,
    depths: Vec<u32>,
    masks: Vec<u32>,
}

impl VerifyTree {
    /// Valide et fige la topologie : parent strictement avant chaque ligne,
    /// profondeur = profondeur du parent + 1, masque = masque du parent plus la
    /// ligne elle-même, au plus 32 lignes (masque `u32`).
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si les tableaux divergent ou violent ces invariants.
    pub(crate) fn new(parents: &[usize], depths: &[usize], masks: &[u32]) -> Result<Self> {
        let rows = parents.len();
        if rows == 0 || rows > 32 || depths.len() != rows || masks.len() != rows {
            return Err(InferError::Dimension(format!(
                "arbre de vérification: {rows} parents, {} profondeurs, {} masques (1..=32)",
                depths.len(),
                masks.len()
            )));
        }
        let valid = depths[0] == 0
            && masks[0] == 1
            && (1..rows).all(|row| {
                let parent = parents[row];
                parent < row
                    && depths[row] == depths[parent] + 1
                    && masks[row] == masks[parent] | (1u32 << row)
            });
        if !valid {
            return Err(InferError::Dimension(
                "arbre de vérification: topologie invalide".to_string(),
            ));
        }
        let to_u32 = |values: &[usize]| {
            values
                .iter()
                .map(|&value| {
                    u32::try_from(value).map_err(|_| {
                        InferError::Dimension("arbre de vérification hors u32".to_string())
                    })
                })
                .collect::<Result<Vec<_>>>()
        };
        Ok(Self {
            parents: to_u32(parents)?,
            depths: to_u32(depths)?,
            masks: masks.to_vec(),
        })
    }

    /// Lignes de l'arbre.
    pub(crate) fn rows(&self) -> usize {
        self.parents.len()
    }

    /// Parent de chaque ligne (racine : 0).
    pub(crate) fn parents(&self) -> &[u32] {
        &self.parents
    }

    /// Profondeur de chaque ligne (racine : 0).
    pub(crate) fn depths(&self) -> &[u32] {
        &self.depths
    }

    /// Masque d'ancêtres (soi compris) de chaque ligne.
    pub(crate) fn masks(&self) -> &[u32] {
        &self.masks
    }

    /// Erreur typée si l'arbre ne couvre pas exactement `rows` lignes.
    pub(crate) fn check_rows(&self, rows: usize, label: &str) -> Result<()> {
        if self.rows() != rows {
            return Err(InferError::Dimension(format!(
                "{label}: arbre de {} lignes pour {rows} lignes encodées",
                self.rows()
            )));
        }
        Ok(())
    }
}

thread_local! {
    // Par thread comme les autres scopes d'encodage : la vérification encode et
    // committe sur le thread qui l'installe.
    static VERIFY_TREE: RefCell<Option<Rc<VerifyTree>>> = const { RefCell::new(None) };
}

/// Rétablit l'arbre précédent à la sortie du scope, panique comprise.
pub(crate) struct VerifyTreeScope(Option<Rc<VerifyTree>>);

impl Drop for VerifyTreeScope {
    fn drop(&mut self) {
        let previous = self.0.take();
        VERIFY_TREE.with(|slot| *slot.borrow_mut() = previous);
    }
}

/// Installe `tree` pour les encodages en lignes de ce thread jusqu'au drop du
/// guard.
pub(crate) fn install_verify_tree_scope(tree: Rc<VerifyTree>) -> VerifyTreeScope {
    VerifyTreeScope(VERIFY_TREE.with(|slot| slot.borrow_mut().replace(tree)))
}

/// Arbre installé sur ce thread, s'il y en a un.
pub(crate) fn verify_tree() -> Option<Rc<VerifyTree>> {
    VERIFY_TREE.with(|slot| slot.borrow().clone())
}

thread_local! {
    // Vrai pendant l'encodage d'une vérification greedy (argmax) : la politique
    // d'exactitude de la requête, lue par le choix du préfixe NAX en lignes.
    static VERIFY_GREEDY: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Rétablit la politique précédente à la sortie du scope, panique comprise.
pub(crate) struct VerifyGreedyScope(bool);

impl Drop for VerifyGreedyScope {
    fn drop(&mut self) {
        VERIFY_GREEDY.with(|slot| slot.set(self.0));
    }
}

thread_local! {
    // Politique de la requête en cours (température nulle ou non), posée par les
    // boucles spéculatives : la vérification en dérive sa portée greedy quel que
    // soit le type de tête (`Skip` sous `SARAGOSSA_RUST_GPU_ARGMAX=0`).
    static REQUEST_GREEDY: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Rétablit la politique de requête précédente à la sortie du scope.
pub(crate) struct RequestGreedyScope(Option<bool>);

impl Drop for RequestGreedyScope {
    fn drop(&mut self) {
        REQUEST_GREEDY.with(|slot| slot.set(self.0));
    }
}

/// Pose la politique greedy de la requête (`true` à température nulle) pour
/// les vérifications de ce thread jusqu'au drop du guard.
pub(crate) fn install_request_greedy_scope(greedy: bool) -> RequestGreedyScope {
    RequestGreedyScope(REQUEST_GREEDY.with(|slot| slot.replace(Some(greedy))))
}

/// Politique greedy de la requête en cours, si une boucle spéculative l'a posée.
pub(crate) fn request_greedy() -> Option<bool> {
    REQUEST_GREEDY.with(std::cell::Cell::get)
}

/// Marque les encodages en lignes de ce thread comme greedy (`true`) ou
/// échantillonnés (`false`) jusqu'au drop du guard.
pub(crate) fn install_verify_greedy_scope(greedy: bool) -> VerifyGreedyScope {
    VerifyGreedyScope(VERIFY_GREEDY.with(|slot| slot.replace(greedy)))
}

/// Préfixe NAX en lignes autorisé pour l'encodage en cours : flag global, et
/// pour une vérification greedy, `SARAGOSSA_RUST_VERIFY_ROWS_NAX_GREEDY` hors
/// mode strict.
pub(crate) fn verify_rows_nax_allowed() -> bool {
    crate::runtime_flags::verify_rows_nax_enabled()
        && (!VERIFY_GREEDY.with(std::cell::Cell::get)
            || (crate::runtime_flags::verify_rows_nax_greedy_enabled() && !verify_strict_active()))
}

#[cfg(test)]
thread_local! {
    static VERIFY_STRICT_TEST_OVERRIDE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
}

/// Force le mode strict sur ce thread le temps de `run` (oracles O1).
#[cfg(test)]
pub(crate) fn with_verify_strict_override<R>(strict: bool, run: impl FnOnce() -> R) -> R {
    /// Rétablit le réglage précédent, panique comprise.
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            VERIFY_STRICT_TEST_OVERRIDE.with(|cell| cell.set(self.0));
        }
    }
    let _restore = Restore(VERIFY_STRICT_TEST_OVERRIDE.with(|cell| cell.replace(Some(strict))));
    run()
}

/// Vérification greedy stricte en cours d'encodage : portée greedy installée et
/// `SARAGOSSA_RUST_VERIFY_STRICT` actif.
pub(crate) fn verify_strict_active() -> bool {
    if !VERIFY_GREEDY.with(std::cell::Cell::get) {
        return false;
    }
    #[cfg(test)]
    if let Some(strict) = VERIFY_STRICT_TEST_OVERRIDE.with(std::cell::Cell::get) {
        return strict;
    }
    crate::runtime_flags::verify_strict_enabled()
}

#[cfg(test)]
mod greedy_scope_tests {
    use super::*;

    /// La portée greedy s'empile et se rétablit au drop.
    #[test]
    fn verify_greedy_scope_nests_and_restores() {
        let greedy = || VERIFY_GREEDY.with(std::cell::Cell::get);
        assert!(!greedy());
        {
            let _outer = install_verify_greedy_scope(true);
            assert!(greedy());
            {
                let _inner = install_verify_greedy_scope(false);
                assert!(!greedy());
            }
            assert!(greedy());
        }
        assert!(!greedy());
    }

    /// La politique de requête s'empile, se rétablit, et rend le mode strict
    /// actif quand la vérification en dérive sa portée greedy (tête `Skip`).
    #[test]
    fn request_greedy_scope_drives_strict_policy() {
        assert_eq!(request_greedy(), None);
        {
            let _request = install_request_greedy_scope(true);
            assert_eq!(request_greedy(), Some(true));
            // Comme `next_final_states_resident_verify` avec une tête non argmax.
            let _verify = install_verify_greedy_scope(request_greedy().unwrap_or(false));
            assert!(with_verify_strict_override(true, verify_strict_active));
            {
                let _sampled = install_request_greedy_scope(false);
                assert_eq!(request_greedy(), Some(false));
            }
            assert_eq!(request_greedy(), Some(true));
        }
        assert_eq!(request_greedy(), None);
    }
}
