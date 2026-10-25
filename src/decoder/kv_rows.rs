//! Lignes K ou V d'une couche full-attn, stockées par segments partagés.
//!
//! Un snapshot (cache de préfixe, blocs serve, rollback spéculatif) clone la
//! liste des segments, pas les données. Les ajouts suivants ne touchent que le
//! dernier segment s'il n'est pas partagé, sinon ils en ouvrent un nouveau :
//! prolonger un préfixe en cache ne recopie plus jamais le préfixe.

use std::sync::Arc;

/// Taille minimale (en f32) d'un nouveau segment : les ajouts d'une ligne en
/// decode CPU remplissent ce segment au lieu d'en ouvrir un par token.
const MIN_SEGMENT_FLOATS: usize = 1 << 18;

#[derive(Clone, Debug, Default)]
pub(super) struct KvRows {
    segments: Vec<Arc<Vec<f32>>>,
    len: usize,
}

impl KvRows {
    /// Nombre de f32 stockés.
    pub(super) fn len(&self) -> usize {
        self.len
    }

    /// Segments dans l'ordre, chacun formé de lignes entières.
    pub(super) fn segments(&self) -> impl Iterator<Item = &[f32]> {
        self.segments.iter().map(|segment| segment.as_slice())
    }

    /// Lignes de `dim` f32 dans l'ordre, sans copie.
    pub(super) fn rows(&self, dim: usize) -> impl Iterator<Item = &[f32]> {
        self.segments()
            .flat_map(move |segment| segment.chunks_exact(dim))
    }

    /// Octets réservés, capacité comprise.
    pub(super) fn capacity_bytes(&self) -> usize {
        self.segments
            .iter()
            .map(|segment| segment.capacity())
            .sum::<usize>()
            .saturating_mul(std::mem::size_of::<f32>())
    }

    pub(super) fn extend_from_slice(&mut self, data: &[f32]) {
        if data.is_empty() {
            return;
        }
        if let Some(last) = self.segments.last_mut().and_then(Arc::get_mut) {
            if last.capacity() - last.len() >= data.len() {
                last.extend_from_slice(data);
                self.len += data.len();
                return;
            }
        }
        let mut segment = Vec::with_capacity(data.len().max(MIN_SEGMENT_FLOATS));
        segment.extend_from_slice(data);
        self.segments.push(Arc::new(segment));
        self.len += data.len();
    }

    /// Ajoute un bloc déjà possédé sans le recopier (relecture du prefill).
    pub(super) fn push_owned(&mut self, data: Vec<f32>) {
        if data.is_empty() {
            return;
        }
        self.len += data.len();
        self.segments.push(Arc::new(data));
    }

    /// Ramène à `len` f32 ; seul le segment coupé en deux est recopié s'il
    /// est partagé. `len` au-delà de la longueur actuelle est ignoré.
    pub(super) fn truncate(&mut self, len: usize) {
        while self.len > len {
            let Some(last) = self.segments.last_mut() else {
                break;
            };
            let start = self.len - last.len();
            if start >= len {
                self.segments.pop();
                self.len = start;
            } else {
                Arc::make_mut(last).truncate(len - start);
                self.len = len;
            }
        }
    }

    #[cfg(test)]
    pub(super) fn to_vec(&self) -> Vec<f32> {
        self.segments().flatten().copied().collect()
    }

    #[cfg(test)]
    pub(super) fn shares_segments_with(&self, other: &Self) -> bool {
        self.segments
            .iter()
            .zip(&other.segments)
            .all(|(a, b)| Arc::ptr_eq(a, b))
    }
}

#[cfg(test)]
mod tests {
    use super::KvRows;

    #[test]
    fn snapshot_shares_prefix_and_truncate_copies_only_split_segment() {
        let mut rows = KvRows::default();
        rows.push_owned(vec![1.0, 2.0, 3.0, 4.0]);
        let snapshot = rows.clone();
        rows.extend_from_slice(&[5.0, 6.0]);
        assert!(rows.shares_segments_with(&snapshot));
        assert_eq!(snapshot.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(rows.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        // Ajout dans le dernier segment non partagé : pas de nouveau segment.
        rows.extend_from_slice(&[7.0, 8.0]);
        assert_eq!(rows.segments().count(), 2);
        rows.truncate(3);
        assert_eq!(rows.to_vec(), vec![1.0, 2.0, 3.0]);
        assert_eq!(snapshot.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(
            rows.rows(1).map(|row| row[0]).collect::<Vec<_>>(),
            vec![1.0, 2.0, 3.0]
        );
        rows.truncate(10);
        assert_eq!(rows.len(), 3);
    }
}
