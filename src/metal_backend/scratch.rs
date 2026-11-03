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

/// Garde RAII restaurant la capacité en lignes précédente.
pub(super) struct ScratchRowCapacityGuard(Option<(usize, usize)>);

impl Drop for ScratchRowCapacityGuard {
    fn drop(&mut self) {
        SCRATCH_ROW_CAPACITY.with(|slot| slot.set(self.0));
    }
}

/// Pendant l'encodage d'un morceau de `rows` lignes, plus court que les
/// morceaux de `capacity` lignes du même prefill, dimensionne les scratch
/// proportionnels aux lignes sur `capacity`. Le cache scratch étant indexé par
/// longueur exacte, le dernier morceau d'un prompt impair (5079 = 2540 + 2539)
/// allouait sinon un second jeu complet (≈ 0,6 Go sur le 35B-A3B). Les kernels
/// reçoivent toujours `rows` : ils n'utilisent que le début des buffers.
pub(super) fn enter_scratch_row_capacity(rows: usize, capacity: usize) -> ScratchRowCapacityGuard {
    let capacity = (rows > 0 && rows < capacity).then_some((rows, capacity));
    ScratchRowCapacityGuard(SCRATCH_ROW_CAPACITY.with(|slot| slot.replace(capacity)))
}

/// Longueur à allouer pour une demande scratch de `len` éléments : un multiple
/// des lignes du morceau court est porté à la capacité. Un faux positif (buffer
/// non proportionnel aux lignes mais multiple) alloue seulement plus grand.
pub(super) fn scratch_len_for_rows(len: usize) -> usize {
    SCRATCH_ROW_CAPACITY
        .with(Cell::get)
        .filter(|&(rows, _)| len % rows == 0)
        .and_then(|(rows, capacity)| (len / rows).checked_mul(capacity))
        .unwrap_or(len)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sous la garde, une demande proportionnelle aux lignes du morceau court
    /// reprend l'entrée du morceau plein ; le reste et l'après-garde sont
    /// inchangés.
    #[test]
    fn short_chunk_scratch_reuses_full_chunk_entry() -> Result<()> {
        let executor = match crate::MetalExecutor::new() {
            Ok(executor) => executor,
            Err(InferError::Metal(message)) if message.contains("aucun device") => return Ok(()),
            Err(error) => return Err(error),
        };
        let (rows, capacity, width) = (2539, 2540, 8);
        let full = executor.private_f32_buffer(capacity * width, "test_rows")?;
        {
            let _guard = enter_scratch_row_capacity(rows, capacity);
            let short = executor.private_f32_buffer(rows * width, "test_rows")?;
            assert_eq!(
                short.as_ptr(),
                full.as_ptr(),
                "entrée du morceau plein reprise"
            );
            let other = executor.private_f32_buffer(rows * width + 1, "test_other")?;
            assert_eq!(other.length(), ((rows * width + 1) * 4) as u64);
            // Morceau plein : garde sans effet.
            let _nested = enter_scratch_row_capacity(capacity, capacity);
            assert_eq!(scratch_len_for_rows(rows * width), rows * width);
        }
        assert_eq!(
            scratch_len_for_rows(rows * width),
            rows * width,
            "garde restaurée"
        );
        Ok(())
    }
}
