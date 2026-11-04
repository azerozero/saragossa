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

thread_local! {
    // Morceau de prefill plus court que les autres : (lignes, capacité).
    static SCRATCH_ROW_CAPACITY: Cell<Option<(usize, usize)>> = const { Cell::new(None) };
}

/// Garde RAII restaurant la capacité en lignes précédente. `!Send` et `!Sync` :
/// détruite sur un autre thread, elle restaurerait le TLS de celui-ci et
/// laisserait la garde active sur le thread d'origine.
pub(super) struct ScratchRowCapacityGuard {
    previous: Option<(usize, usize)>,
    _thread_bound: std::marker::PhantomData<std::rc::Rc<()>>,
}

impl Drop for ScratchRowCapacityGuard {
    fn drop(&mut self) {
        SCRATCH_ROW_CAPACITY.with(|slot| slot.set(self.previous));
    }
}

/// Pendant l'encodage d'un morceau de `rows` lignes, plus court que les
/// morceaux de `capacity` lignes du même prefill, laisse les scratch
/// proportionnels aux lignes reprendre les entrées des morceaux pleins. Le
/// cache scratch étant indexé par longueur exacte, le dernier morceau d'un
/// prompt impair (5079 = 2540 + 2539) allouait sinon un second jeu complet
/// (≈ 0,6 Go sur le 35B-A3B). Les kernels reçoivent toujours `rows` : ils
/// n'utilisent que le début des buffers repris.
pub(super) fn enter_scratch_row_capacity(rows: usize, capacity: usize) -> ScratchRowCapacityGuard {
    let capacity = (rows > 0 && rows < capacity).then_some((rows, capacity));
    ScratchRowCapacityGuard {
        previous: SCRATCH_ROW_CAPACITY.with(|slot| slot.replace(capacity)),
        _thread_bound: std::marker::PhantomData,
    }
}

/// Longueur servie pour une demande scratch de `len` éléments. Sous la garde,
/// une demande multiple des lignes du morceau court essaie la longueur portée
/// à la capacité, mais seulement si `cached` en connaît déjà une entrée : on
/// n'alloue jamais une longueur arrondie. Sinon la demande garde sa longueur
/// exacte, si bien que la garde ne fait jamais pire que sans elle. Les faux
/// positifs (scratch de couche déjà à la capacité, poids déquantifiés, morceau
/// d'une ligne) retombent ainsi sur leur longueur exacte.
pub(super) fn scratch_len_for_rows(len: usize, cached: impl Fn(usize) -> bool) -> usize {
    SCRATCH_ROW_CAPACITY
        .with(Cell::get)
        .filter(|&(rows, _)| len % rows == 0)
        .and_then(|(rows, capacity)| (len / rows).checked_mul(capacity))
        .filter(|&rounded| cached(rounded))
        .unwrap_or(len)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn active() -> Option<(usize, usize)> {
        SCRATCH_ROW_CAPACITY.with(Cell::get)
    }

    /// Contre-exemple des relectures : 4097 tokens → morceaux de 2049 puis 2048
    /// lignes. La scratch de couche, déjà à 2049 × 2048, ne doit pas viser la
    /// clé 2049 × 2049, absente : elle garde sa longueur et reprend son entrée.
    #[test]
    fn full_capacity_request_keeps_exact_length() {
        let _guard = enter_scratch_row_capacity(2048, 2049);
        let len = 2049 * 2048;
        assert_eq!(scratch_len_for_rows(len, |cached| cached == len), len);
        assert_eq!(scratch_len_for_rows(len, |_| false), len);
    }

    /// Clé arrondie connue : reprise. Inconnue : longueur exacte. Morceau
    /// d'une ligne ou non-multiple : longueur exacte faute d'entrée.
    #[test]
    fn rounded_length_only_when_already_cached() {
        let _guard = enter_scratch_row_capacity(2539, 2540);
        assert_eq!(scratch_len_for_rows(2539 * 8, |l| l == 2540 * 8), 2540 * 8);
        assert_eq!(scratch_len_for_rows(2539 * 8, |_| false), 2539 * 8);
        assert_eq!(scratch_len_for_rows(2539 * 8 + 1, |_| true), 2539 * 8 + 1);
        let _one_row = enter_scratch_row_capacity(1, 512);
        assert_eq!(scratch_len_for_rows(4096, |_| false), 4096);
    }

    /// Morceau plein ou vide : garde sans effet.
    #[test]
    fn full_or_empty_chunk_installs_nothing() {
        let _full = enter_scratch_row_capacity(2540, 2540);
        assert_eq!(active(), None);
        let _empty = enter_scratch_row_capacity(0, 2540);
        assert_eq!(active(), None);
    }

    /// La garde extérieure redevient active dès la sortie de l'imbriquée.
    #[test]
    fn nested_guard_restores_outer_guard() {
        let outer = enter_scratch_row_capacity(2539, 2540);
        {
            let _inner = enter_scratch_row_capacity(100, 200);
            assert_eq!(active(), Some((100, 200)));
        }
        assert_eq!(active(), Some((2539, 2540)));
        drop(outer);
        assert_eq!(active(), None);
    }

    /// Restaurée par `?` comme par une panique.
    #[test]
    fn guard_restored_after_error_and_panic() {
        fn failing() -> Result<()> {
            let _guard = enter_scratch_row_capacity(2539, 2540);
            Err(InferError::Metal("échec simulé".to_string()))?;
            Ok(())
        }
        assert!(failing().is_err());
        assert_eq!(active(), None);
        let unwound = std::panic::catch_unwind(|| {
            let _guard = enter_scratch_row_capacity(2539, 2540);
            panic!("panique simulée");
        });
        assert!(unwound.is_err());
        assert_eq!(active(), None);
    }

    /// Avec un device : le morceau court reprend l'entrée du morceau plein sans
    /// entrée neuve ; une clé arrondie absente alloue la longueur exacte.
    #[test]
    fn short_chunk_reuses_cached_entries_without_new_allocation() -> Result<()> {
        let executor = match crate::MetalExecutor::new() {
            Ok(executor) => executor,
            Err(InferError::Metal(message)) if message.contains("aucun device") => return Ok(()),
            Err(error) => return Err(error),
        };
        let (rows, capacity, width) = (2539, 2540, 8);
        let full = executor.private_f32_buffer(capacity * width, "test_rows")?;
        let _guard = enter_scratch_row_capacity(rows, capacity);
        let (entries, _) = executor.scratch_buffer_stats()?;
        let short = executor.private_f32_buffer(rows * width, "test_rows")?;
        assert_eq!(
            short.as_ptr(),
            full.as_ptr(),
            "entrée du morceau plein reprise"
        );
        assert_eq!(
            executor.scratch_buffer_stats()?.0,
            entries,
            "aucune entrée neuve"
        );
        let fresh = executor.private_f32_buffer(rows * width, "test_fresh")?;
        assert_eq!(fresh.length(), (rows * width * 4) as u64, "longueur exacte");
        assert_eq!(executor.scratch_buffer_stats()?.0, entries + 1);
        Ok(())
    }
}
