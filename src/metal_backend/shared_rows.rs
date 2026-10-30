//! Lignes f32 ou bf16 lues en place dans un buffer partagé que le GPU a fini
//! d'écrire.

use super::*;
use std::collections::BTreeMap;
use std::sync::{Arc, MutexGuard, PoisonError};

/// Élément d'une vue partagée : f32, ou bits bf16 (`u16`). Scellé : seuls des
/// types sans invalide ni padding peuvent être lus depuis la mémoire GPU.
pub(crate) trait SharedElement: Copy + sealed::Sealed {}

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for u16 {}
}

impl SharedElement for f32 {}
impl SharedElement for u16 {}

/// Vue CPU, sans copie, de `len` éléments d'un buffer `StorageModeShared` à
/// partir de l'élément `offset`. Le K/V d'un prefill complet y reste : sa
/// relecture vers un `Vec` neuf payait une faute de page par 16 Kio (382 ms à
/// 32k).
///
/// Toutes les vues d'une allocation partagent un registre de vivacité
/// ([`LiveEnds`]) : une seule construction par [`Self::new`], puis des vues
/// dérivées (clone, [`Self::truncate`], [`Self::extended`]). Un écrivain sous
/// [`InPlaceClaim`] n'écrit qu'au-delà de la plus longue vue vivante : une
/// plage publiée n'est jamais réécrite tant qu'une vue la couvre.
#[derive(Debug)]
pub(crate) struct SharedRows<T> {
    buffer: metal::Buffer,
    offset: usize,
    len: usize,
    live: Arc<LiveEnds>,
    element: std::marker::PhantomData<T>,
}

/// Fins (en éléments, `offset + len`) des vues vivantes d'une allocation,
/// comptées par valeur. Seul un détenteur d'[`InPlaceClaim`] peut y ajouter
/// une fin au-delà du maximum courant (cf. [`SharedRows::extended`]).
#[derive(Debug, Default)]
struct LiveEnds(Mutex<BTreeMap<usize, usize>>);

impl LiveEnds {
    fn ends(&self) -> MutexGuard<'_, BTreeMap<usize, usize>> {
        // Les sections critiques ne font que des opérations de table, qui
        // laissent le compte cohérent même si un autre thread a paniqué.
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn add(&self, end: usize) {
        *self.ends().entry(end).or_insert(0) += 1;
    }

    fn remove(&self, end: usize) {
        let mut ends = self.ends();
        let count = ends.get_mut(&end);
        // Chaque vue retire la fin qu'elle a ajoutée : une fin absente
        // trahirait un registre désynchronisé (vue créée hors dérivation).
        debug_assert!(count.is_some(), "fin {end} absente du registre");
        if let Some(count) = count {
            *count -= 1;
            if *count == 0 {
                ends.remove(&end);
            }
        }
    }

    /// Fin de la plus longue vue vivante (0 sans vue).
    fn max(&self) -> usize {
        self.ends().last_key_value().map_or(0, |(end, _)| *end)
    }
}

/// Lignes f32 (K/V ou hidden d'un prefill).
pub(crate) type SharedF32Rows = SharedRows<f32>;
/// Lignes bf16 stockées en bits `u16` (K/V d'un prefill stocké en bf16).
pub(crate) type SharedBf16Rows = SharedRows<u16>;

/// Vérifie qu'une vue de `offset + len` éléments `T` tient dans `buffer` et
/// qu'il a un pointeur CPU.
fn check_view<T>(buffer: &BufferRef, offset: usize, len: usize) -> Result<()> {
    let end = offset
        .checked_add(len)
        .and_then(|end| end.checked_mul(std::mem::size_of::<T>()))
        .ok_or_else(|| InferError::Dimension("vue partagée déborde".to_string()))?;
    if u64::try_from(end)
        .ok()
        .is_none_or(|end| end > buffer.length())
    {
        return Err(InferError::Dimension(format!(
            "vue partagée {offset}+{len} éléments hors du buffer de {} octets",
            buffer.length()
        )));
    }
    if buffer.contents().is_null() {
        return Err(InferError::Metal(
            "vue partagée sur un buffer sans pointeur CPU".to_string(),
        ));
    }
    Ok(())
}

impl<T: SharedElement> SharedRows<T> {
    /// Première vue de `buffer[offset..offset + len]` (en éléments `T`) : crée
    /// le registre de vivacité de l'allocation.
    ///
    /// # Safety
    ///
    /// - `buffer[offset..offset + len]` est initialisé et ses écritures GPU
    ///   sont terminées.
    /// - Aucune autre vue de `buffer` ne vient de `new` : les suivantes
    ///   dérivent de celle-ci (clone, troncature, [`Self::extended`]), sans
    ///   quoi deux registres s'ignoreraient. Seule exception : un buffer que
    ///   plus rien n'écrit du tout.
    /// - Ensuite, aucune écriture, GPU ou CPU, ne vise un élément couvert par
    ///   une vue vivante de l'allocation (les lectures GPU, semis de l'arène,
    ///   restent permises) : la vue expose son contenu en `&[T]`. Seul le
    ///   détenteur d'un [`InPlaceClaim`] revalidé ([`InPlaceClaim::try_after`])
    ///   écrit, et seulement au-delà de la plus longue vue vivante.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si la plage dépasse le buffer ou si le buffer n'a
    /// pas de pointeur CPU (stockage privé).
    #[expect(
        unsafe_code,
        reason = "contrat d'immuabilité du buffer reporté sur l'appelant"
    )]
    pub(crate) unsafe fn new(buffer: metal::Buffer, offset: usize, len: usize) -> Result<Self> {
        check_view::<T>(&buffer, offset, len)?;
        let live = Arc::new(LiveEnds::default());
        live.add(offset + len);
        Ok(Self {
            buffer,
            offset,
            len,
            live,
            element: std::marker::PhantomData,
        })
    }

    /// Vue `[offset, offset + len)` du même buffer, dérivée de `self` (même
    /// registre) : publie les éléments `[self.end(), offset + len)` écrits
    /// en place sous `claim`. `claim` doit avoir été pris après `self`
    /// ([`InPlaceClaim::try_after`]) : la revalidation a alors établi
    /// qu'aucune vue vivante ne couvrait ces éléments.
    ///
    /// # Safety
    ///
    /// Les éléments `[self.end(), offset + len)` ont été écrits par le
    /// détenteur de `claim` et ses écritures GPU sont terminées ; plus rien ne
    /// les écrit ensuite (contrat de [`Self::new`]).
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si `claim` ne couvre pas cette allocation après
    /// `self`, si la vue raccourcit `self` ou si elle dépasse le buffer.
    #[expect(unsafe_code, reason = "publication de lignes écrites sous claim")]
    pub(crate) unsafe fn extended(&self, claim: &InPlaceClaim, len: usize) -> Result<Self> {
        if claim.start_of(self.allocation_id()) != Some(self.end()) || len < self.len {
            return Err(InferError::Metal(format!(
                "vue prolongée à {len} hors du claim pris après {}",
                self.end()
            )));
        }
        check_view::<T>(&self.buffer, self.offset, len)?;
        self.live.add(self.offset + len);
        Ok(Self {
            buffer: self.buffer.clone(),
            offset: self.offset,
            len,
            live: Arc::clone(&self.live),
            element: std::marker::PhantomData,
        })
    }

    #[expect(
        unsafe_code,
        reason = "lecture CPU en place d'un MTLBuffer StorageModeShared figé"
    )]
    pub(crate) fn as_slice(&self) -> &[T] {
        // SAFETY: `new` a vérifié que `offset + len` éléments tiennent dans le
        // buffer partagé et que son pointeur CPU existe (aligné page, donc
        // aligné pour `T`, f32 ou u16 scellés : toute suite d'octets est une
        // valeur valide). Tant que `self` vit, sa fin compte dans le registre :
        // aucun écrivain, revalidé sous claim, ne touche cette plage (contrat
        // `# Safety` de `new`), et les vues ne donnent qu'un accès en lecture.
        unsafe {
            std::slice::from_raw_parts(
                self.buffer.contents().cast::<T>().add(self.offset),
                self.len,
            )
        }
    }

    /// Buffer source, à ne lier qu'en lecture (copie GPU vers l'arène).
    pub(crate) fn buffer(&self) -> &BufferRef {
        &self.buffer
    }

    /// Identité de l'allocation, partagée par toutes ses vues (troncatures et
    /// clones compris) : le comptage mémoire n'en retient qu'une. C'est
    /// l'adresse de l'objet `MTLBuffer`, et non `contents()` : un buffer privé
    /// n'a pas de pointeur CPU, alors que l'objet, retenu par chaque vue, reste
    /// à la même adresse tant qu'une vue vit.
    pub(crate) fn allocation_id(&self) -> usize {
        allocation_identity(&self.buffer)
    }

    /// Taille entière de l'allocation en octets, qu'une vue tronquée garde en vie.
    pub(crate) fn allocation_bytes(&self) -> usize {
        usize::try_from(self.buffer.length()).unwrap_or(usize::MAX)
    }

    pub(crate) fn byte_offset(&self) -> u64 {
        // Borné par `buffer.length()` à la construction.
        (self.offset * std::mem::size_of::<T>()) as u64
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Fin de la vue dans le buffer, en éléments.
    fn end(&self) -> usize {
        self.offset + self.len
    }

    /// Vue depuis le début du buffer qu'aucune vue vivante ne dépasse : les
    /// éléments au-delà de `len` peuvent être écrits. Sans claim, ce n'est
    /// qu'un indice (une autre vue peut naître entre-temps) ; seul
    /// [`InPlaceClaim::try_after`] le revalide avant d'écrire.
    pub(crate) fn is_whole(&self) -> bool {
        self.offset == 0 && self.live.max() == self.len
    }

    /// Raccourcit la vue ; le buffer reste alloué tant qu'une vue le tient.
    /// Les éléments qu'aucune vue ne couvre plus redeviennent inscriptibles.
    pub(crate) fn truncate(&mut self, len: usize) {
        if len < self.len {
            self.live.add(self.offset + len);
            self.live.remove(self.end());
            self.len = len;
        }
    }
}

impl<T> Clone for SharedRows<T> {
    fn clone(&self) -> Self {
        self.live.add(self.offset + self.len);
        Self {
            buffer: self.buffer.clone(),
            offset: self.offset,
            len: self.len,
            live: Arc::clone(&self.live),
            element: std::marker::PhantomData,
        }
    }
}

impl<T> Drop for SharedRows<T> {
    fn drop(&mut self) {
        self.live.remove(self.offset + self.len);
    }
}

/// Identité stable d'une allocation Metal : l'adresse de l'objet `MTLBuffer`.
/// Toute vue et tout claim retiennent l'objet, donc deux allocations vivantes
/// ne partagent jamais cette adresse, même en stockage privé (sans
/// `contents()`).
pub(crate) fn allocation_identity(buffer: &BufferRef) -> usize {
    buffer.as_ptr().addr()
}

/// Allocations K/V dont la réserve est en cours d'écriture (prolongement en
/// place d'un prefill, arène de decode qui a adopté le buffer), tous
/// executors confondus.
static IN_PLACE_ALLOCATIONS: std::sync::Mutex<Vec<usize>> = std::sync::Mutex::new(Vec::new());

/// Droit exclusif d'écrire la réserve de K et V, rendu au drop. Le détenteur
/// garantit qu'aucune commande GPU n'y écrit plus quand il le rend. Il
/// retient les buffers : leurs identités ([`allocation_identity`]) ne peuvent
/// pas être reprises par une autre allocation tant que le claim vit.
#[derive(Debug)]
pub(crate) struct InPlaceClaim {
    ids: [usize; 2],
    /// Fin, en éléments, de la vue après laquelle chaque allocation a été
    /// réservée.
    starts: [usize; 2],
    _buffers: [Buffer; 2],
}

impl InPlaceClaim {
    /// Réserve brute, sans revalidation : réservée à [`Self::try_after`].
    fn try_new(buffers: [&BufferRef; 2], starts: [usize; 2]) -> Option<Self> {
        let ids = buffers.map(allocation_identity);
        let mut claimed = IN_PLACE_ALLOCATIONS.lock().ok()?;
        if ids.iter().any(|id| claimed.contains(id)) {
            return None;
        }
        claimed.extend(ids);
        Some(Self {
            ids,
            starts,
            _buffers: buffers.map(ToOwned::to_owned),
        })
    }

    /// Droit d'écrire K et V au-delà de leurs vues `views` (K puis V). Le
    /// claim est pris **avant** de vérifier qu'aucune vue vivante ne les
    /// dépasse : sous le claim, aucune vue plus longue ne peut naître (seul
    /// [`SharedRows::extended`], qui l'exige, en crée). Vérifier d'abord
    /// laissait un autre écrivain prolonger et publier entre la vérification
    /// et le claim, puis être écrasé. `None` : la réserve est tenue ou
    /// couverte par une vue vivante, l'appelant recopie.
    pub(crate) fn try_after<T: SharedElement>(views: [&SharedRows<T>; 2]) -> Option<Self> {
        let claim = Self::try_new(views.map(SharedRows::buffer), views.map(SharedRows::end))?;
        views.iter().all(|view| view.is_whole()).then_some(claim)
    }

    /// Fin de la vue après laquelle l'allocation `id` a été réservée.
    fn start_of(&self, id: usize) -> Option<usize> {
        self.ids
            .iter()
            .position(|claimed| *claimed == id)
            .map(|index| self.starts[index])
    }
}

impl Drop for InPlaceClaim {
    fn drop(&mut self) {
        if let Ok(mut claimed) = IN_PLACE_ALLOCATIONS.lock() {
            claimed.retain(|id| !self.ids.contains(id));
        }
    }
}

/// K, V `[rows, kv_dim]` ou hidden final `[rows, hidden]` rendu par le prefill
/// résident : vue du buffer GPU qu'il a seul écrit, ou copie relue (extension
/// de K/V dont le buffer porte aussi le préfixe, dernière ligne seule). Le K/V
/// d'un prefill stocké en bf16 arrive en bits `u16` ; le hidden reste f32.
#[derive(Clone, Debug)]
pub(crate) enum PrefillKvRows {
    Owned(Vec<f32>),
    OwnedBf16(Vec<u16>),
    Shared(SharedF32Rows),
    SharedBf16(SharedBf16Rows),
}

impl PrefillKvRows {
    /// Nombre d'éléments, sans lecture des données.
    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Owned(data) => data.len(),
            Self::OwnedBf16(data) => data.len(),
            Self::Shared(rows) => rows.len(),
            Self::SharedBf16(rows) => rows.len(),
        }
    }

    /// Segment typé, vue partagée comprise (préfixe d'un prolongement).
    #[cfg(test)]
    pub(crate) fn segment(&self) -> crate::decode_resident::KvSegment<'_> {
        use crate::decode_resident::KvSegment;
        match self {
            Self::Owned(data) => KvSegment::Cpu(data),
            Self::OwnedBf16(data) => KvSegment::CpuBf16(data),
            Self::Shared(rows) => KvSegment::Shared(rows),
            Self::SharedBf16(rows) => KvSegment::SharedBf16(rows),
        }
    }

    /// Tenseur `[len / cols, cols]` ; une vue partagée est copiée.
    pub(crate) fn into_tensor(self, cols: usize) -> Result<Tensor> {
        let data = match self {
            Self::Owned(data) => data,
            Self::Shared(rows) => rows.as_slice().to_vec(),
            Self::OwnedBf16(_) | Self::SharedBf16(_) => {
                return Err(InferError::Dimension(
                    "lignes de prefill bf16 en tenseur f32".to_string(),
                ))
            }
        };
        Tensor::from_vec(vec![data.len() / cols.max(1), cols], data)
    }
}

#[cfg(test)]
impl MetalExecutor {
    /// Vue partagée d'une copie de `data`, comme le hidden d'un prefill.
    #[expect(unsafe_code, reason = "buffer neuf propre au test")]
    pub(crate) fn shared_f32_rows_for_test(&self, data: &[f32]) -> Result<SharedF32Rows> {
        let buffer = self.uncached_f32_buffer(data.len(), "test_shared_rows")?;
        write_f32_buffer(&buffer, data)?;
        // SAFETY: buffer neuf, hors scratch, que plus rien n'écrit ensuite.
        unsafe { SharedF32Rows::new(buffer, 0, data.len()) }
    }
}

#[cfg(test)]
mod identity_tests {
    use super::*;

    #[test]
    #[expect(unsafe_code, reason = "buffers neufs propres au test")]
    fn views_share_identity_and_allocations_do_not() -> Result<()> {
        let Some(device) = metal::Device::system_default() else {
            return Ok(());
        };
        let first = device.new_buffer(64, MTLResourceOptions::StorageModeShared);
        let second = device.new_buffer(64, MTLResourceOptions::StorageModeShared);
        // SAFETY: buffers neufs (mis à zéro par Metal) que rien n'écrit ensuite.
        let (whole, other) = unsafe {
            (
                SharedF32Rows::new(first.clone(), 0, 16)?,
                SharedF32Rows::new(second, 0, 16)?,
            )
        };
        // SAFETY: même buffer neuf, vue décalée en lecture seule.
        let shifted = unsafe { SharedF32Rows::new(first.clone(), 4, 8)? };
        let mut truncated = whole.clone();
        truncated.truncate(3);
        assert_eq!(whole.allocation_id(), truncated.allocation_id());
        assert_eq!(whole.allocation_id(), shifted.allocation_id());
        assert_eq!(whole.allocation_id(), allocation_identity(&first));
        assert_ne!(whole.allocation_id(), other.allocation_id());
        Ok(())
    }

    #[test]
    fn private_buffers_have_distinct_identities_and_exclusive_claims() {
        let Some(device) = metal::Device::system_default() else {
            return;
        };
        let new = || device.new_buffer(64, MTLResourceOptions::StorageModePrivate);
        let (key, value, other) = (new(), new(), new());
        // `contents()` d'un buffer privé n'est pas une identité garantie
        // (nul pour un buffer sparse) : seule l'adresse de l'objet compte.
        let ids = [&key, &value, &other].map(|buffer| allocation_identity(buffer));
        assert!(ids.iter().all(|id| *id != 0));
        assert!(ids[0] != ids[1] && ids[1] != ids[2] && ids[0] != ids[2]);
        let claim = InPlaceClaim::try_new([&key, &value], [0, 0]);
        assert!(claim.is_some());
        assert!(InPlaceClaim::try_new([&other, &value], [0, 0]).is_none());
        assert!(InPlaceClaim::try_new([&other, &new()], [0, 0]).is_some());
    }

    /// Le registre suit clones, troncatures, prolongements et drops : une
    /// vue vivante plus longue interdit l'écriture au-delà d'une vue plus
    /// courte (revalidation sous claim), sa mort l'autorise de nouveau.
    #[test]
    #[expect(unsafe_code, reason = "buffers neufs propres au test")]
    fn liveness_registry_gates_writes_beyond_live_views() -> Result<()> {
        let Some(device) = metal::Device::system_default() else {
            return Ok(());
        };
        // 16 f32 par buffer.
        let new = || device.new_buffer(64, MTLResourceOptions::StorageModeShared);
        let (key, value) = (new(), new());
        // SAFETY: buffers neufs (mis à zéro par Metal) ; seules les lignes
        // au-delà des vues sont réservées, et rien ne les écrit ici.
        let (k, v) = unsafe {
            (
                SharedF32Rows::new(key.clone(), 0, 4)?,
                SharedF32Rows::new(value.clone(), 0, 4)?,
            )
        };
        let k2 = k.clone();
        assert!(k.is_whole() && k2.is_whole(), "clone de même fin");
        let mut short = k.clone();
        short.truncate(2);
        assert!(!short.is_whole(), "vue plus longue vivante");

        let claim = InPlaceClaim::try_after([&k, &v]).expect("réserve libre");
        assert!(
            InPlaceClaim::try_after([&k, &v]).is_none(),
            "claim exclusif"
        );
        // SAFETY: rien n'écrit la réserve ; seules les fins comptent ici.
        let (long, long_v) = unsafe { (k.extended(&claim, 8)?, v.extended(&claim, 8)?) };
        // Le claim ne publie qu'après les vues pour lesquelles il a été pris.
        // SAFETY: appels refusés avant toute création de vue.
        unsafe {
            assert!(short.extended(&claim, 8).is_err(), "vue plus courte");
            assert!(long.extended(&claim, 12).is_err(), "vue déjà prolongée");
            assert!(k.extended(&claim, 17).is_err(), "hors du buffer");
        }
        drop(claim);
        assert!(long.is_whole() && !k.is_whole() && !k2.is_whole());
        // Course : `k` était entière avant le prolongement ; la revalidation
        // sous claim refuse et rend la réserve.
        assert!(InPlaceClaim::try_after([&k, &v]).is_none());
        drop((long, long_v));
        assert!(k.is_whole(), "vue plus longue morte");
        assert!(
            InPlaceClaim::try_after([&k, &v]).is_some(),
            "réserve rendue"
        );
        // Rollback : la seule vue restante, tronquée, devient entière.
        drop((k, k2));
        assert!(short.is_whole());
        Ok(())
    }
}
