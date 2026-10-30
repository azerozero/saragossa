//! Lignes K ou V d'une couche full-attn, stockées par segments partagés.
//!
//! Un snapshot (cache de préfixe, blocs serve, rollback spéculatif) clone la
//! liste des segments, pas les données. Les ajouts suivants ne touchent que le
//! dernier segment s'il n'est pas partagé, sinon ils en ouvrent un nouveau :
//! prolonger un préfixe en cache ne recopie plus jamais le préfixe.
//!
//! Un segment est f32, ou bf16 (bits `u16`) quand le prefill stocke son K/V
//! arrondi : les lecteurs f32 (attention CPU de repli) l'élargissent à la
//! volée et voient les valeurs arrondies, pas le f32 d'origine.

use std::borrow::Cow;
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
    /// f32 logiques, `base` compris : les segments couvrent `[base, len)`.
    len: usize,
    /// f32 absents devant le premier segment : lignes d'une couche glissante
    /// sorties de sa fenêtre, jamais relues (cf. `LayerKvCache::base_rows`).
    base: usize,
}

/// Bloc de lignes : possédé, ou vue figée du buffer GPU d'un prefill (relu en
/// place, sans copie ni faute de page sur une mémoire neuve), en f32 ou bf16.
#[derive(Clone, Debug)]
enum Segment {
    Owned(Vec<f32>),
    OwnedBf16(Vec<u16>),
    #[cfg(all(target_os = "macos", feature = "metal"))]
    Shared(crate::metal_backend::SharedF32Rows),
    #[cfg(all(target_os = "macos", feature = "metal"))]
    SharedBf16(crate::metal_backend::SharedBf16Rows),
}

/// Élargit des bits bf16 en f32 (exact).
pub(super) fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

impl Segment {
    /// Lignes en f32 : empruntées, ou élargies depuis le bf16 (copie).
    fn as_f32(&self) -> Cow<'_, [f32]> {
        let bf16 = match self {
            Self::Owned(data) => return Cow::Borrowed(data),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::Shared(rows) => return Cow::Borrowed(rows.as_slice()),
            Self::OwnedBf16(data) => data.as_slice(),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::SharedBf16(rows) => rows.as_slice(),
        };
        Cow::Owned(bf16.iter().copied().map(bf16_bits_to_f32).collect())
    }

    /// Nombre d'éléments, sans lecture des données.
    fn len(&self) -> usize {
        match self {
            Self::Owned(data) => data.len(),
            Self::OwnedBf16(data) => data.len(),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::Shared(rows) => rows.len(),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::SharedBf16(rows) => rows.len(),
        }
    }

    /// Allocation tenue (identité, octets réservés) : une vue partagée, même
    /// tronquée ou clonée, garde tout son buffer en vie.
    fn allocation(&self) -> (usize, usize) {
        match self {
            Self::Owned(data) => (
                data.as_ptr().addr(),
                data.capacity() * std::mem::size_of::<f32>(),
            ),
            Self::OwnedBf16(data) => (
                data.as_ptr().addr(),
                data.capacity() * std::mem::size_of::<u16>(),
            ),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::Shared(rows) => (rows.allocation_id(), rows.allocation_bytes()),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::SharedBf16(rows) => (rows.allocation_id(), rows.allocation_bytes()),
        }
    }

    fn truncate(&mut self, len: usize) {
        match self {
            Self::Owned(data) => data.truncate(len),
            Self::OwnedBf16(data) => data.truncate(len),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::Shared(rows) => rows.truncate(len),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::SharedBf16(rows) => rows.truncate(len),
        }
    }
}

/// Lignes de `dim` f32 de `segments` (cf. [`KvRows::f32_segments`]) à partir
/// de la ligne `first`, sans copie ; les segments entièrement avant `first`
/// sont sautés sans être parcourus.
pub(super) fn rows_from<'a>(
    segments: &'a [Cow<'_, [f32]>],
    dim: usize,
    first: usize,
) -> impl Iterator<Item = &'a [f32]> {
    let mut skip = first.saturating_mul(dim);
    segments
        .iter()
        .filter_map(move |segment| {
            let start = skip.min(segment.len());
            skip -= start;
            (start < segment.len()).then(|| &segment[start..])
        })
        .flat_map(move |segment| segment.chunks_exact(dim))
}

impl KvRows {
    /// Nombre de f32 logiques, `base` compris.
    pub(super) fn len(&self) -> usize {
        self.len
    }

    /// f32 absents devant les segments.
    pub(super) fn base(&self) -> usize {
        self.base
    }

    /// Déclare `base` f32 absents devant des lignes encore à ajouter ; refusé
    /// (`false`) sur des lignes non vides.
    pub(super) fn set_base(&mut self, base: usize) -> bool {
        if self.len != self.base {
            return false;
        }
        self.base = base;
        self.len = base;
        true
    }

    /// Segments f32 dans l'ordre, chacun formé de lignes entières ; un
    /// segment bf16 est élargi (copie, repli CPU seulement).
    pub(super) fn f32_segments(&self) -> Vec<Cow<'_, [f32]>> {
        self.segments
            .iter()
            .map(|segment| segment.as_f32())
            .collect()
    }

    /// Octets réservés, capacité comprise, des allocations absentes de `seen` ;
    /// les allocations comptées y sont ajoutées (un segment partagé entre
    /// snapshots, ou plusieurs vues d'un même buffer, ne compte qu'une fois).
    pub(super) fn capacity_bytes_unseen(&self, seen: &mut HashSet<usize>) -> usize {
        self.segments
            .iter()
            .map(|segment| segment.allocation())
            .filter(|(id, _)| seen.insert(*id))
            .map(|(_, floats)| floats)
            .sum::<usize>()
    }

    pub(super) fn extend_from_slice(&mut self, data: &[f32]) {
        if data.is_empty() {
            return;
        }
        // Après un segment bf16 ou partagé, un segment f32 neuf : les ajouts
        // ne sont jamais arrondis.
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

    /// Ajoute un bloc bf16 déjà possédé (lignes nouvelles d'un prolongement
    /// stocké en bf16).
    pub(super) fn push_owned_bf16(&mut self, data: Vec<u16>) {
        if data.is_empty() {
            return;
        }
        self.len += data.len();
        self.segments.push(Arc::new(Segment::OwnedBf16(data)));
    }

    /// Buffer GPU dont une seule vue porte toutes les lignes depuis son début
    /// (K ou V d'un prefill à froid) : un prolongement peut y écrire à la
    /// suite, cf. `PrefillResidentPrefixLayer::Full::in_place`.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(super) fn sole_shared_buffer(&self) -> Option<metal::Buffer> {
        match self.segments.as_slice() {
            [segment] => match segment.as_ref() {
                Segment::Shared(rows) if rows.byte_offset() == 0 => Some(rows.buffer().to_owned()),
                Segment::SharedBf16(rows) if rows.byte_offset() == 0 => {
                    Some(rows.buffer().to_owned())
                }
                _ => None,
            },
            _ => None,
        }
    }

    /// Premier segment, vue bf16 entière (depuis le début, qu'aucune vue
    /// vivante ne dépasse) : une arène de decode peut adopter son buffer et
    /// écrire à la suite. Simple indice : l'adoption le revalide sous claim
    /// (`DecodeResidentState::full_attention_adopting`).
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(super) fn adoptable_bf16_prefix(&self) -> Option<&crate::metal_backend::SharedBf16Rows> {
        match self.segments.first().map(AsRef::as_ref) {
            Some(Segment::SharedBf16(rows)) if rows.is_whole() => Some(rows),
            _ => None,
        }
    }

    /// Ajoute le K ou V d'un prefill résident, vue GPU comprise, sans copie.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(super) fn push_prefill(&mut self, rows: crate::metal_backend::PrefillKvRows) {
        use crate::metal_backend::PrefillKvRows;
        let segment = match rows {
            PrefillKvRows::Owned(data) => return self.push_owned(data),
            PrefillKvRows::OwnedBf16(data) => return self.push_owned_bf16(data),
            PrefillKvRows::Shared(rows) => Segment::Shared(rows),
            PrefillKvRows::SharedBf16(rows) => Segment::SharedBf16(rows),
        };
        if segment.len() == 0 {
            return;
        }
        self.len += segment.len();
        self.segments.push(Arc::new(segment));
    }

    /// Segments dans l'ordre, typés avec leur provenance (semis de l'arène,
    /// préfixe d'un prolongement résident).
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(super) fn segments(&self) -> Vec<crate::decode_resident::KvSegment<'_>> {
        use crate::decode_resident::KvSegment;
        self.segments
            .iter()
            .map(|segment| match segment.as_ref() {
                Segment::Owned(data) => KvSegment::Cpu(data),
                Segment::OwnedBf16(data) => KvSegment::CpuBf16(data),
                Segment::Shared(rows) => KvSegment::Shared(rows),
                Segment::SharedBf16(rows) => KvSegment::SharedBf16(rows),
            })
            .collect()
    }

    /// Ramène à `len` f32 ; seul le segment coupé en deux est recopié s'il
    /// est partagé. `len` au-delà de la longueur actuelle est ignoré, `len`
    /// sous `base` s'arrête à `base` (refusé plus haut, cf.
    /// `LayerKvCache::truncate`).
    pub(super) fn truncate(&mut self, len: usize) {
        while self.len > len.max(self.base) {
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
        self.f32_segments()
            .iter()
            .flat_map(|segment| segment.iter().copied())
            .collect()
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
    use super::{rows_from, KvRows};

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
        assert_eq!(rows.segments.len(), 2);
        rows.truncate(3);
        assert_eq!(rows.to_vec(), vec![1.0, 2.0, 3.0]);
        assert_eq!(snapshot.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(
            rows_from(&rows.f32_segments(), 1, 0)
                .map(|row| row[0])
                .collect::<Vec<_>>(),
            vec![1.0, 2.0, 3.0]
        );
        let mut split = KvRows::default();
        split.push_owned(vec![1.0, 2.0]);
        split.push_owned(vec![3.0, 4.0, 5.0, 6.0]);
        assert_eq!(
            rows_from(&split.f32_segments(), 2, 2)
                .map(|row| row[0])
                .collect::<Vec<_>>(),
            vec![5.0]
        );
        assert_eq!(rows_from(&split.f32_segments(), 2, 3).count(), 0);
        rows.truncate(10);
        assert_eq!(rows.len(), 3);
    }

    #[test]
    fn bf16_segments_widen_count_two_bytes_and_stay_unrounded_on_append() {
        let mut rows = KvRows::default();
        // 1.0, -2.0, 0.5 et 3.0 en bits bf16.
        rows.push_owned_bf16(vec![0x3f80, 0xc000, 0x3f00, 0x4040]);
        let snapshot = rows.clone();
        // Un ajout f32 ouvre son propre segment : jamais arrondi.
        rows.extend_from_slice(&[1.000_001, 2.0]);
        assert_eq!(rows.segments.len(), 2);
        assert_eq!(rows.len(), 6);
        assert_eq!(rows.to_vec(), vec![1.0, -2.0, 0.5, 3.0, 1.000_001, 2.0]);
        assert_eq!(
            rows_from(&rows.f32_segments(), 2, 1)
                .map(|row| row[0])
                .collect::<Vec<_>>(),
            vec![0.5, 1.000_001]
        );
        let mut seen = std::collections::HashSet::new();
        assert_eq!(snapshot.capacity_bytes_unseen(&mut seen), 4 * 2);
        rows.truncate(3);
        assert_eq!(rows.to_vec(), vec![1.0, -2.0, 0.5]);
        assert_eq!(snapshot.to_vec(), vec![1.0, -2.0, 0.5, 3.0]);
    }

    /// Seule la vue entière d'un prefill bf16 est adoptable : un clone
    /// tronqué laisserait l'arène écrire dans des lignes encore vues par un
    /// autre état (snapshot plus long).
    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn only_whole_bf16_prefill_view_is_adoptable() -> crate::Result<()> {
        let Some(device) = metal::Device::system_default() else {
            return Ok(());
        };
        let buffer = device.new_buffer(64, metal::MTLResourceOptions::StorageModeShared);
        // SAFETY: buffer neuf propre au test, jamais écrit.
        #[expect(unsafe_code, reason = "vue partagée sur un buffer de test")]
        let view = unsafe { crate::metal_backend::SharedBf16Rows::new(buffer, 0, 16)? };
        let mut rows = KvRows::default();
        rows.push_prefill(crate::metal_backend::PrefillKvRows::SharedBf16(view));
        assert_eq!(
            rows.adoptable_bf16_prefix().map(|prefix| prefix.len()),
            Some(16)
        );
        let mut truncated = rows.clone();
        truncated.truncate(8);
        assert!(truncated.adoptable_bf16_prefix().is_none());
        assert!(rows.adoptable_bf16_prefix().is_some());
        // Une queue possédée après la vue n'empêche pas l'adoption du préfixe.
        rows.push_owned_bf16(vec![0; 8]);
        assert_eq!(
            rows.adoptable_bf16_prefix().map(|prefix| prefix.len()),
            Some(16)
        );
        // Rollback (`CausalDecoderCache::truncate`) : l'état tronqué redevient
        // prolongeable en place dès qu'aucune vue plus longue ne vit.
        drop(rows);
        assert_eq!(
            truncated.adoptable_bf16_prefix().map(|prefix| prefix.len()),
            Some(8)
        );
        Ok(())
    }
}
