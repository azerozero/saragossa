//! Namespace des buffers scratch du backend Metal.

use super::*;

thread_local! {
    // Namespace courant du scratch label-keyed (light-batch) : 0 = chemin
    // historique mono-flux ; un slot par flux isole les buffers mémoïsés par
    // `(label, len, element)` qui sinon s'aliaseraient entre flux concurrents.
    static SCRATCH_NAMESPACE: Cell<u64> = const { Cell::new(0) };
}

pub(super) fn current_scratch_namespace() -> u64 {
    SCRATCH_NAMESPACE.with(Cell::get)
}

/// Garde RAII restaurant le namespace scratch précédent à sa sortie de portée.
pub(crate) struct ScratchNamespaceGuard(u64);

impl Drop for ScratchNamespaceGuard {
    fn drop(&mut self) {
        SCRATCH_NAMESPACE.with(|slot| slot.set(self.0));
    }
}

/// Installe `namespace` comme namespace scratch courant du thread (light-batch :
/// un slot par flux). Le chemin mono-flux n'installe rien → namespace 0, clés de
/// scratch strictement identiques à l'historique.
pub(crate) fn install_scratch_namespace(namespace: u64) -> ScratchNamespaceGuard {
    SCRATCH_NAMESPACE.with(|slot| {
        let previous = slot.replace(namespace);
        ScratchNamespaceGuard(previous)
    })
}
