//! Lignes K ou V d'une couche full-attn, stockées par segments partagés.
//!
//! Un snapshot (cache de préfixe, blocs serve, rollback spéculatif) clone la
//! liste des segments, pas les données. Les ajouts suivants ne touchent que le
//! dernier segment s'il n'est pas partagé, sinon ils en ouvrent un nouveau :
//! prolonger un préfixe en cache ne recopie plus jamais le préfixe.

use std::collections::HashSet;
use std::sync::Arc;

/// Taille minimale (en f32) d'un nouveau segment : les ajouts d'une ligne en
/// decode CPU remplissent ce segment au lieu d'en ouvrir un par token.
// NOTE: après un snapshot, le premier ajout réserve 1 Mio par K/V et par
// couche full-attn (~32 Mio sur le 27B), compté par `capacity_bytes_unseen`.
const MIN_SEGMENT_FLOATS: usize = 1 << 18;

#[derive(Clone, Debug, Default)]
pub(super) struct KvRows {
    segments: Vec<Arc<Segment>>,
    len: usize,
}

/// Bloc de lignes : possédé, ou vue figée du buffer GPU d'un prefill (relu en
/// place, sans copie ni faute de page sur une mémoire neuve).
#[derive(Clone, Debug)]
enum Segment {
    Owned(Vec<f32>),
    #[cfg(all(target_os = "macos", feature = "metal"))]
    Shared(crate::metal_backend::SharedF32Rows),
}

impl Segment {
    fn as_slice(&self) -> &[f32] {
        match self {
            Self::Owned(data) => data,
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::Shared(rows) => rows.as_slice(),
        }
    }

    fn len(&self) -> usize {
        self.as_slice().len()
    }

    fn capacity(&self) -> usize {
        match self {
            Self::Owned(data) => data.capacity(),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::Shared(rows) => rows.len(),
        }
    }

    fn truncate(&mut self, len: usize) {
        match self {
            Self::Owned(data) => data.truncate(len),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::Shared(rows) => rows.truncate(len),
        }
    }
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

    /// Lignes de `dim` f32 à partir de la ligne `first`, sans copie ; les
    /// segments entièrement avant `first` sont sautés sans être parcourus.
    pub(super) fn rows_from(&self, dim: usize, first: usize) -> impl Iterator<Item = &[f32]> {
        let mut skip = first.saturating_mul(dim);
        self.segments()
            .filter_map(move |segment| {
                let start = skip.min(segment.len());
                skip -= start;
                (start < segment.len()).then(|| &segment[start..])
            })
            .flat_map(move |segment| segment.chunks_exact(dim))
    }

    /// Octets réservés, capacité comprise, des segments absents de `seen` ;
    /// les segments comptés y sont ajoutés (un segment partagé ne compte
    /// qu'une fois sur plusieurs snapshots).
    pub(super) fn capacity_bytes_unseen(&self, seen: &mut HashSet<usize>) -> usize {
        self.segments
            .iter()
            .filter(|segment| seen.insert(Arc::as_ptr(segment) as usize))
            .map(|segment| segment.capacity())
            .sum::<usize>()
            .saturating_mul(std::mem::size_of::<f32>())
    }

    pub(super) fn extend_from_slice(&mut self, data: &[f32]) {
        if data.is_empty() {
            return;
        }
        if let Some(Segment::Owned(last)) = self.segments.last_mut().and_then(Arc::get_mut) {
            if last.capacity() - last.len() >= data.len() {
                last.extend_from_slice(data);
                self.len += data.len();
                return;
            }
        }
        let mut segment = Vec::with_capacity(data.len().max(MIN_SEGMENT_FLOATS));
        segment.extend_from_slice(data);
        self.segments.push(Arc::new(Segment::Owned(segment)));
        self.len += data.len();
    }

    /// Ajoute un bloc déjà possédé sans le recopier (relecture du prefill).
    pub(super) fn push_owned(&mut self, data: Vec<f32>) {
        if data.is_empty() {
            return;
        }
        self.len += data.len();
        self.segments.push(Arc::new(Segment::Owned(data)));
    }

    /// Ajoute le K ou V d'un prefill résident, vue GPU comprise, sans copie.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(super) fn push_prefill(&mut self, rows: crate::metal_backend::PrefillKvRows) {
        match rows {
            crate::metal_backend::PrefillKvRows::Owned(data) => self.push_owned(data),
            crate::metal_backend::PrefillKvRows::Shared(rows) => {
                if rows.len() == 0 {
                    return;
                }
                self.len += rows.len();
                self.segments.push(Arc::new(Segment::Shared(rows)));
            }
        }
    }

    /// Segments dans l'ordre, avec leur provenance pour le semis de l'arène.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(super) fn seed_segments(&self) -> Vec<crate::decode_resident::KvSeedSegment<'_>> {
        use crate::decode_resident::KvSeedSegment;
        self.segments
            .iter()
            .map(|segment| match segment.as_ref() {
                Segment::Owned(data) => KvSeedSegment::Cpu(data),
                Segment::Shared(rows) => KvSeedSegment::Shared(rows),
            })
            .collect()
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
            rows.rows_from(1, 0).map(|row| row[0]).collect::<Vec<_>>(),
            vec![1.0, 2.0, 3.0]
        );
        let mut split = KvRows::default();
        split.push_owned(vec![1.0, 2.0]);
        split.push_owned(vec![3.0, 4.0, 5.0, 6.0]);
        assert_eq!(
            split.rows_from(2, 2).map(|row| row[0]).collect::<Vec<_>>(),
            vec![5.0]
        );
        assert_eq!(split.rows_from(2, 3).count(), 0);
        rows.truncate(10);
        assert_eq!(rows.len(), 3);
    }
}
