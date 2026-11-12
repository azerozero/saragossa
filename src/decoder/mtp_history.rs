//! Contrat de stockage du semis committed, limité au prefill d'une requête.
//! Le défaut f32 protège les appels CPU, DFlash et les diagnostics directs.

use std::cell::Cell;

thread_local! {
    static BF16_CAPACITY: Cell<Option<usize>> = const { Cell::new(None) };
}

pub(super) fn bf16_capacity() -> Option<usize> {
    BF16_CAPACITY.with(Cell::get)
}

/// Portée imbriquable : les callbacks et les erreurs restaurent le contrat
/// précédent. La capacité comprend le prompt et la réserve spéculative.
pub(super) fn with_storage<R>(capacity: Option<usize>, run: impl FnOnce() -> R) -> R {
    struct Restore(Option<usize>);
    impl Drop for Restore {
        fn drop(&mut self) {
            BF16_CAPACITY.with(|cell| cell.set(self.0));
        }
    }
    let _restore = Restore(BF16_CAPACITY.with(|cell| cell.replace(capacity)));
    run()
}

pub(super) fn compatible(history: &super::LayerKvCache) -> bool {
    bf16_capacity().is_some() || (!history.keys.has_bf16() && !history.values.has_bf16())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn storage_scope_restores_and_rejects_rounded_history_for_f32() {
        let mut history = super::super::LayerKvCache::default();
        history.keys.push_owned_bf16(vec![0x3f80]);
        assert!(!compatible(&history));
        with_storage(Some(4096), || {
            assert!(compatible(&history));
            with_storage(None, || assert!(!compatible(&history)));
            assert_eq!(bf16_capacity(), Some(4096));
        });
        assert_eq!(bf16_capacity(), None);
        assert!(compatible(&super::super::LayerKvCache::default()));
    }
}
