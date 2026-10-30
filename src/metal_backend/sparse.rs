//! K/V en buffers « placement sparse » Metal 4 : un buffer privé virtuel,
//! dimensionné au contexte maximal, dont seules les tuiles mappées sur un heap
//! de type placement occupent de la mémoire.
//!
//! Fondation seule (A1) : rien n'est encore branché sur le prefill ni sur le
//! decode, et `SARAGOSSA_RUST_KV_SPARSE` n'est lu par personne.
//!
//! # Synchronisation
//!
//! Les mappings passent par une `MTL4CommandQueue` propre au pool. Chaque lot
//! d'opérations est suivi d'un signal monotone d'un `MTLSharedEvent`, puis
//! d'une attente CPU bornée (`waitUntilSignaledValue:timeoutMS:`) : une
//! commande soumise ensuite sur une file classique voit donc le mapping. Les
//! changements de mapping sont sérialisés par le verrou du pool.
//!
//! Un unmap exige une preuve de fin d'usage ([`UseProof`]) : un
//! `(MTLSharedEvent, valeur)` que l'appelant a signalé sur sa file classique
//! après les dernières commandes qui lisent ou écrivent les tuiles retirées.
//! Ordre : attente de la preuve, unmap, confirmation MTL4, recyclage. Tout
//! échec (preuve absente, timeout, confirmation manquante) laisse les tuiles
//! hors de la liste libre : une tuile peut-être encore lue ne sert jamais à un
//! autre buffer. Une preuve qui expire empoisonne le buffer : le drop le perd
//! au lieu de se fier à une preuve plus ancienne.
//!
//! **Toute soumission qui lie le buffer est suivie d'un
//! [`SparseKvBuffer::record_use`]** avec une preuve signalée après elle : le
//! drop recycle sur la dernière preuve connue, et recyclerait à tort des
//! tuiles qu'une commande non déclarée lit encore.
//!
//! # Contrats pour le branchement (A2)
//!
//! - **Snapshots.** Une vue (lignes `0..n`) suit le contrat de vivacité de
//!   [`super::shared_rows::SharedRows`] : on n'écrit jamais sous une vue vivante. Un
//!   prolongement n'écrit en place qu'au-delà de toutes les vues vivantes du
//!   buffer ; sinon il recopie.
//! - **Mapping avant écriture.** [`SparseKvBuffer::ensure_rows`] précède
//!   chaque soumission qui écrit, marge spéculative comprise (lignes de
//!   vérification MTP/DFlash), et pas seulement le prefill : la capacité
//!   virtuelle ne garantit pas qu'une ligne est mappée.
//! - **Écritures CPU.** Les chemins qui écrivent le K/V par `contents()`
//!   (segments CPU recopiés dans `prefill/resident.rs`, append bf16 de
//!   `decode_resident/attention.rs`) sont exclus de l'éligibilité sparse ; le
//!   seul accès CPU passe par [`super::transfer`].
//! - **Données recyclées.** Une tuile recyclée n'est pas remise à zéro : toute
//!   ligne est écrite avant d'être lue.
//! - **Résidence.** [`SparseKvBuffer::use_heaps`] s'appelle sur chaque encodeur
//!   compute qui lie le buffer. Le choix contre un residency set (qui couvrirait
//!   aussi les blits) se tranche en A2, mesure à l'appui.
//! - **Mémoire.** Recycler une tuile ne rend pas de RAM tant que son heap vit :
//!   seule la destruction d'un heap vide la rend.

use super::*;
use crate::memory_guard::{MemoryGuard, MemoryReservation};
use metal::foreign_types::ForeignType;
use metal::objc::runtime::{Object, Sel, BOOL, YES};
use metal::objc::{Message, MessageArguments};
use metal::{DeviceRef, Heap, HeapRef, MTLHeapType, MTLStorageMode, NSRange, SharedEvent};
use std::sync::Arc;

/// Taille d'un heap placement : 256 Mio, soit 4096 tuiles de 64 Kio. Assez
/// grand pour amortir la création d'un heap, assez petit pour que la
/// destruction d'un heap vide rende la RAM par paliers modérés. Pas de flag de
/// réglage avant la télémétrie de A2.
const HEAP_BYTES: u64 = 256 << 20;
/// `MTLSparsePageSize64` : tuiles de 64 Kio, mesurées par la sonde sur M5 Max.
const SPARSE_PAGE_SIZE_64: i64 = 102;
/// `MTLSparseTextureMappingModeMap` / `Unmap`.
const MAPPING_MODE_MAP: NSUInteger = 0;
const MAPPING_MODE_UNMAP: NSUInteger = 1;
/// Attente CPU bornée d'un mapping ou d'une preuve de fin d'usage. Un mapping
/// mesuré prend 1,65 ms par Gio : 5 s ne se dépassent que sur un GPU bloqué.
const WAIT_TIMEOUT_MS: u64 = 5_000;

/// `MTL4UpdateSparseBufferMappingOperation` (MTL4CommandQueue.h, macOS 26.0).
#[repr(C)]
struct MappingOperation {
    mode: NSUInteger,
    /// Plage du buffer, en tuiles.
    buffer_range: NSRange,
    /// Première tuile du heap, ignorée pour un unmap.
    heap_offset: NSUInteger,
}

/// `object` répond-il à `selector` (`respondsToSelector:` de `NSObject`) ?
#[expect(unsafe_code, reason = "respondsToSelector: via le runtime ObjC")]
fn responds(object: *mut Object, selector: Sel) -> bool {
    if object.is_null() {
        return false;
    }
    // SAFETY: `object` est un objet ObjC vivant non nul (contrat de
    // l'appelant) ; `respondsToSelector:` prend un `SEL` et rend un `BOOL`.
    let answer = unsafe {
        (*object).send_message::<_, BOOL>(Sel::register("respondsToSelector:"), (selector,))
    };
    answer.is_ok_and(|answer| answer == YES)
}

/// Envoie `selector` à `object` s'il y répond (testé à chaque appel).
///
/// # Safety
///
/// `object` est nul ou un objet ObjC vivant, et `A`, `R` correspondent
/// exactement aux arguments et au retour de `selector`.
#[expect(unsafe_code, reason = "envoi ObjC typé par l'appelant")]
unsafe fn send<A: MessageArguments, R: std::any::Any>(
    object: *mut Object,
    selector: &str,
    args: A,
) -> Result<R> {
    let sel = Sel::register(selector);
    if !responds(object, sel) {
        return Err(InferError::Metal(format!(
            "sélecteur Metal {selector} indisponible"
        )));
    }
    // SAFETY: objet vivant non nul (vérifié par `responds`) ; signature
    // garantie par l'appelant.
    unsafe { (*object).send_message(sel, args) }
        .map_err(|error| InferError::Metal(format!("{selector} : {error:?}")))
}

/// Taille de tuile en octets si `device` sait créer des buffers placement
/// sparse (`supportsPlacementSparse`, annoté macOS 26.4) et une file Metal 4 ;
/// `None` sinon. Chaque sélecteur est testé avant d'être appelé.
#[expect(unsafe_code, reason = "sélecteurs Metal 4 absents de metal-rs 0.29")]
pub(crate) fn supports(device: &DeviceRef) -> Option<u64> {
    let raw = device.as_ptr().cast::<Object>();
    let creates = [
        "newBufferWithLength:options:placementSparsePageSize:",
        "newMTL4CommandQueue",
    ]
    .into_iter()
    .all(|selector| responds(raw, Sel::register(selector)));
    // SAFETY: `raw` est le `id<MTLDevice>` vivant de `device` ; propriété
    // `BOOL` en lecture seule.
    let supported = unsafe { send::<_, BOOL>(raw, "supportsPlacementSparse", ()) };
    if !creates || !supported.is_ok_and(|supported| supported == YES) {
        return None;
    }
    // SAFETY: idem ; `MTLSparsePageSize` est un `NSInteger`, le retour un
    // `NSUInteger`.
    let tile = unsafe {
        send::<_, NSUInteger>(
            raw,
            "sparseTileSizeInBytesForSparsePageSize:",
            (SPARSE_PAGE_SIZE_64,),
        )
    }
    .ok()?;
    // Le heap doit contenir un nombre entier de tuiles.
    (tile > 0 && HEAP_BYTES % tile == 0).then_some(tile)
}

/// Attend au plus `timeout_ms` que `event` atteigne `value`.
#[expect(
    unsafe_code,
    reason = "waitUntilSignaledValue: absent de metal-rs 0.29"
)]
fn wait_signaled(event: &SharedEvent, value: u64, timeout_ms: u64) -> bool {
    // SAFETY: `id<MTLSharedEvent>` vivant ; arguments `uint64_t`, retour
    // `BOOL` (macOS 12).
    let reached = unsafe {
        send::<_, BOOL>(
            event.as_ptr().cast::<Object>(),
            "waitUntilSignaledValue:timeoutMS:",
            (value, timeout_ms),
        )
    };
    match reached {
        Ok(reached) => reached == YES,
        // Sélecteur absent : pas d'attente, seulement l'état courant.
        Err(_) => event.signaled_value() >= value,
    }
}

/// `id<MTL4CommandQueue>` possédé (+1 de `newMTL4CommandQueue`).
struct Mtl4Queue(*mut Object);

// SAFETY: une file Metal est utilisable depuis n'importe quel thread, et le
// pool sérialise en plus tous ses appels derrière son verrou.
#[expect(unsafe_code, reason = "file Metal 4 brute, thread-safe côté Metal")]
unsafe impl Send for Mtl4Queue {}

impl Mtl4Queue {
    #[expect(unsafe_code, reason = "newMTL4CommandQueue absent de metal-rs 0.29")]
    fn new(device: &DeviceRef) -> Result<Self> {
        // SAFETY: device vivant ; le retour est un objet +1 (ou nil) dont
        // `Self` prend la propriété.
        let queue = Self(unsafe {
            send::<_, *mut Object>(device.as_ptr().cast::<Object>(), "newMTL4CommandQueue", ())?
        });
        let usable = [
            "updateBufferMappings:heap:operations:count:",
            "signalEvent:value:",
        ]
        .into_iter()
        .all(|selector| responds(queue.0, Sel::register(selector)));
        if usable {
            Ok(queue)
        } else {
            Err(InferError::Metal(
                "file Metal 4 sans mapping sparse".to_string(),
            ))
        }
    }

    #[expect(unsafe_code, reason = "updateBufferMappings: absent de metal-rs 0.29")]
    fn update(
        &self,
        buffer: &BufferRef,
        heap: Option<&HeapRef>,
        ops: &[MappingOperation],
    ) -> Result<()> {
        if ops.is_empty() {
            return Ok(());
        }
        let heap = heap.map_or(std::ptr::null_mut(), |heap| heap.as_ptr().cast::<Object>());
        // SAFETY: file vivante ; buffer placement sparse vivant ; heap
        // placement vivant, ou nil pour des unmaps seuls (autorisé par
        // l'en-tête) ; `ops` est un tableau C de `ops.len()` structures au
        // layout de l'en-tête, lu pendant l'appel seulement.
        unsafe {
            send::<_, ()>(
                self.0,
                "updateBufferMappings:heap:operations:count:",
                (
                    buffer.as_ptr().cast::<Object>(),
                    heap,
                    ops.as_ptr(),
                    ops.len() as NSUInteger,
                ),
            )
        }
    }

    #[expect(unsafe_code, reason = "signalEvent:value: de MTL4CommandQueue")]
    fn signal(&self, event: &SharedEvent, value: u64) -> Result<()> {
        // SAFETY: file et événement vivants ; `(id<MTLEvent>, uint64_t)`.
        unsafe {
            send::<_, ()>(
                self.0,
                "signalEvent:value:",
                (event.as_ptr().cast::<Object>(), value),
            )
        }
    }
}

impl Drop for Mtl4Queue {
    #[expect(unsafe_code, reason = "release de la file Metal 4 possédée")]
    fn drop(&mut self) {
        // SAFETY: rend l'unique référence +1 prise à la création (nul : rien).
        drop(unsafe { send::<_, ()>(self.0, "release", ()) });
    }
}

#[expect(unsafe_code, reason = "heap placement compatible sparse")]
fn new_placement_heap(device: &DeviceRef, bytes: u64) -> Result<Heap> {
    let descriptor = metal::HeapDescriptor::new();
    descriptor.set_size(bytes);
    descriptor.set_storage_mode(MTLStorageMode::Private);
    descriptor.set_heap_type(MTLHeapType::Placement);
    let raw_descriptor = descriptor.as_ptr().cast::<Object>();
    // SAFETY: descripteur vivant ; argument `MTLSparsePageSize` (`NSInteger`).
    unsafe {
        send::<_, ()>(
            raw_descriptor,
            "setMaxCompatiblePlacementSparsePageSize:",
            (SPARSE_PAGE_SIZE_64,),
        )?;
    }
    // SAFETY: device et descripteur vivants ; retour +1 ou nil, vérifié avant
    // la prise de propriété.
    let raw = unsafe {
        send::<_, *mut Object>(
            device.as_ptr().cast::<Object>(),
            "newHeapWithDescriptor:",
            (raw_descriptor,),
        )?
    };
    if raw.is_null() {
        return Err(InferError::Metal(format!(
            "heap placement de {bytes} octets refusé"
        )));
    }
    // SAFETY: `raw` est un `id<MTLHeap>` +1 non nul dont `Heap` prend la
    // propriété.
    let heap = unsafe { Heap::from_ptr(raw.cast()) };
    // Le K/V vivant ne doit jamais être purgé sous pression mémoire.
    heap.set_purgeable_state(MTLPurgeableState::NonVolatile);
    Ok(heap)
}

#[expect(
    unsafe_code,
    reason = "newBufferWithLength:options:placementSparsePageSize:"
)]
fn new_sparse_buffer(device: &DeviceRef, bytes: u64) -> Result<Buffer> {
    // SAFETY: device vivant qui répond au sélecteur (vérifié par `supports`,
    // préalable du pool) ; retour +1 ou nil, vérifié avant la prise de
    // propriété. Le stockage privé est obligatoire pour un buffer sparse.
    let raw = unsafe {
        send::<_, *mut Object>(
            device.as_ptr().cast::<Object>(),
            "newBufferWithLength:options:placementSparsePageSize:",
            (
                bytes,
                MTLResourceOptions::StorageModePrivate,
                SPARSE_PAGE_SIZE_64,
            ),
        )?
    };
    if raw.is_null() {
        return Err(InferError::Metal(format!(
            "buffer sparse de {bytes} octets refusé"
        )));
    }
    // SAFETY: `raw` est un `id<MTLBuffer>` +1 non nul.
    Ok(unsafe { Buffer::from_ptr(raw.cast()) })
}

/// `newBufferWithLength:options:` avec test de nil : metal-rs 0.29 construit
/// sinon un wrapper non nul sur un retour nil.
///
/// # Errors
///
/// Renvoie une erreur si Metal refuse le buffer.
#[expect(
    unsafe_code,
    reason = "retour nullable de newBufferWithLength:options:"
)]
pub(super) fn checked_buffer(
    device: &DeviceRef,
    bytes: u64,
    options: MTLResourceOptions,
) -> Result<Buffer> {
    // SAFETY: device vivant ; `(NSUInteger, MTLResourceOptions)`, retour +1
    // ou nil, vérifié avant la prise de propriété.
    let raw = unsafe {
        send::<_, *mut Object>(
            device.as_ptr().cast::<Object>(),
            "newBufferWithLength:options:",
            (bytes, options),
        )?
    };
    if raw.is_null() {
        return Err(InferError::Metal(format!(
            "buffer de {bytes} octets refusé"
        )));
    }
    // SAFETY: `raw` est un `id<MTLBuffer>` +1 non nul.
    Ok(unsafe { Buffer::from_ptr(raw.cast()) })
}

/// `newSharedEvent` avec test de nil (même raison que [`checked_buffer`]).
#[expect(unsafe_code, reason = "retour nullable de newSharedEvent")]
fn new_shared_event(device: &DeviceRef) -> Result<SharedEvent> {
    // SAFETY: device vivant ; retour +1 ou nil, vérifié avant la prise de
    // propriété.
    let raw =
        unsafe { send::<_, *mut Object>(device.as_ptr().cast::<Object>(), "newSharedEvent", ())? };
    if raw.is_null() {
        return Err(InferError::Metal("événement partagé refusé".to_string()));
    }
    // SAFETY: `raw` est un `id<MTLSharedEvent>` +1 non nul.
    Ok(unsafe { SharedEvent::from_ptr(raw.cast()) })
}

/// Tuile `index` du heap rangé à `slot` dans le registre.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Tile {
    slot: usize,
    index: u32,
}

#[derive(Debug)]
struct HeapSlot<H> {
    heap: H,
    /// Tuiles libres, triées en ordre décroissant : `pop` rend la plus basse,
    /// ce qui garde des plages contiguës (un seul op de mapping par plage).
    free: Vec<u32>,
    /// Tuiles prêtes à un buffer ou perdues (jamais recyclées).
    used: u32,
    /// Tuiles perdues : le heap ne doit jamais être libéré.
    leaked: u32,
}

/// Registre des tuiles, sans appel Metal (testable sans GPU). `H` est le heap.
#[derive(Debug)]
struct Ledger<H> {
    slots: Vec<Option<HeapSlot<H>>>,
    tiles_per_heap: u32,
    leaked_tiles: u64,
}

impl<H> Ledger<H> {
    fn new(tiles_per_heap: u32) -> Self {
        Self {
            slots: Vec::new(),
            tiles_per_heap,
            leaked_tiles: 0,
        }
    }

    fn heaps(&self) -> impl Iterator<Item = &HeapSlot<H>> {
        self.slots.iter().flatten()
    }

    fn add_heap(&mut self, heap: H) {
        let slot = HeapSlot {
            heap,
            free: (0..self.tiles_per_heap).rev().collect(),
            used: 0,
            leaked: 0,
        };
        match self.slots.iter().position(Option::is_none) {
            Some(index) => self.slots[index] = Some(slot),
            None => self.slots.push(Some(slot)),
        }
    }

    /// Prend jusqu'à `count` tuiles libres, heaps les plus anciens d'abord.
    fn take(&mut self, count: usize, into: &mut Vec<Tile>) {
        for (slot, heap) in self.slots.iter_mut().enumerate() {
            let Some(heap) = heap else { continue };
            while into.len() < count {
                let Some(index) = heap.free.pop() else { break };
                heap.used += 1;
                into.push(Tile { slot, index });
            }
        }
    }

    fn recycle(&mut self, tiles: &[Tile]) {
        for tile in tiles {
            if let Some(Some(heap)) = self.slots.get_mut(tile.slot) {
                heap.free.push(tile.index);
                heap.used = heap.used.saturating_sub(1);
            }
        }
        for heap in self.slots.iter_mut().flatten() {
            heap.free.sort_unstable_by(|a, b| b.cmp(a));
        }
    }

    /// Les tuiles restent comptées comme utilisées pour toujours : leur heap
    /// n'est jamais détruit.
    fn leak(&mut self, tiles: &[Tile]) {
        self.leaked_tiles += tiles.len() as u64;
        for tile in tiles {
            if let Some(Some(heap)) = self.slots.get_mut(tile.slot) {
                heap.leaked += 1;
            }
        }
    }

    /// Retire les heaps qui portent au moins une tuile perdue, pour que le
    /// détenteur les garde à jamais.
    fn take_leaked_heaps(&mut self) -> Vec<H> {
        self.slots
            .iter_mut()
            .filter(|slot| slot.as_ref().is_some_and(|heap| heap.leaked > 0))
            .filter_map(|slot| slot.take().map(|heap| heap.heap))
            .collect()
    }

    fn free_tiles(&self) -> u64 {
        self.heaps().map(|heap| heap.free.len() as u64).sum()
    }

    fn used_tiles(&self) -> u64 {
        self.heaps().map(|heap| u64::from(heap.used)).sum()
    }

    /// Retire les heaps entièrement libres tant que la capacité libre dépasse
    /// un heap : un seul heap libre reste pour la croissance suivante.
    fn drain_empty_heaps(&mut self) -> Vec<H> {
        let mut drained = Vec::new();
        while self.free_tiles() > u64::from(self.tiles_per_heap) {
            let Some(slot) = self
                .slots
                .iter()
                .rposition(|slot| slot.as_ref().is_some_and(|heap| heap.used == 0))
            else {
                break;
            };
            if let Some(heap) = self.slots[slot].take() {
                drained.push(heap.heap);
            }
        }
        drained
    }
}

/// Compteurs du pool (octets sauf mention).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SparseKvStats {
    pub(crate) tile_bytes: u64,
    /// Capacité virtuelle des buffers vivants (ou perdus) : aucune RAM.
    pub(crate) virtual_bytes: u64,
    /// Tuiles mappées par des buffers, perdues comprises.
    pub(crate) mapped_tiles: u64,
    /// Capacité des heaps détenus : ce que la RAM paie.
    pub(crate) heap_bytes: u64,
    /// Capacité libre des heaps, réutilisable sans nouvelle allocation.
    pub(crate) free_bytes: u64,
    /// Tuiles retirées à jamais (preuve absente, timeout, empoisonnement).
    pub(crate) leaked_tiles: u64,
}

/// Heap et sa réservation auprès de la garde mémoire, rendue avec lui.
// ponytail: un heap écrit est compté deux fois (réservation et
// `phys_footprint`) ; prudent, à compenser en A2 si la garde refuse trop tôt.
struct PoolHeap {
    heap: Heap,
    _reservation: MemoryReservation,
}

struct PoolState {
    ledger: Ledger<PoolHeap>,
    queue: Mtl4Queue,
    event: SharedEvent,
    /// Dernière valeur signalée sur `event` : strictement croissante.
    event_value: u64,
    virtual_bytes: u64,
}

impl PoolState {
    /// Signale la valeur suivante après les opérations déjà soumises et
    /// l'attend, au plus `timeout_ms`.
    fn confirm(&mut self, timeout_ms: u64) -> bool {
        self.event_value += 1;
        self.queue.signal(&self.event, self.event_value).is_ok()
            && wait_signaled(&self.event, self.event_value, timeout_ms)
    }
}

impl Drop for PoolState {
    fn drop(&mut self) {
        // Une tuile perdue peut encore être lue par une commande en vol : son
        // heap n'est jamais libéré, même quand le pool disparaît.
        let leaked = self.ledger.take_leaked_heaps();
        if !leaked.is_empty() {
            eprintln!(
                "saragossa: pool K/V sparse détruit, {} heaps gardés (tuiles perdues)",
                leaked.len()
            );
        }
        for heap in leaked {
            std::mem::forget(heap);
        }
    }
}

struct PoolShared {
    device: Device,
    tile_bytes: u64,
    guard: MemoryGuard,
    /// Attente bornée des mappings et des preuves, lisible sans le verrou.
    wait_timeout_ms: std::sync::atomic::AtomicU64,
    state: Mutex<PoolState>,
}

impl PoolShared {
    fn timeout_ms(&self) -> u64 {
        self.wait_timeout_ms
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, PoolState>> {
        self.state
            .lock()
            .map_err(|_| InferError::Metal("pool K/V sparse empoisonné".to_string()))
    }
}

/// Pool de tuiles K/V, un par device : heaps placement partagés par toutes
/// les couches, liste libre de tuiles, file MTL4 des mappings.
#[derive(Clone)]
pub(crate) struct SparseKvPool {
    shared: Arc<PoolShared>,
}

impl SparseKvPool {
    /// Pool vide : aucun heap avant le premier [`SparseKvBuffer::ensure_rows`].
    /// Chaque nouveau heap est réservé auprès de `guard`.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le device ne gère pas le placement sparse.
    pub(crate) fn new(device: &DeviceRef, guard: MemoryGuard) -> Result<Self> {
        let tile_bytes = supports(device)
            .ok_or_else(|| InferError::Metal("placement sparse indisponible".to_string()))?;
        let tiles_per_heap = u32::try_from(HEAP_BYTES / tile_bytes)
            .map_err(|_| InferError::Metal("tuile sparse trop petite".to_string()))?;
        let state = PoolState {
            ledger: Ledger::new(tiles_per_heap),
            queue: Mtl4Queue::new(device)?,
            event: new_shared_event(device)?,
            event_value: 0,
            virtual_bytes: 0,
        };
        Ok(Self {
            shared: Arc::new(PoolShared {
                device: device.to_owned(),
                tile_bytes,
                guard,
                wait_timeout_ms: std::sync::atomic::AtomicU64::new(WAIT_TIMEOUT_MS),
                state: Mutex::new(state),
            }),
        })
    }

    /// Instantané des compteurs.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le verrou du pool est empoisonné.
    pub(crate) fn stats(&self) -> Result<SparseKvStats> {
        let state = self.shared.lock()?;
        let tile = self.shared.tile_bytes;
        let heaps = state.ledger.heaps().count() as u64;
        Ok(SparseKvStats {
            tile_bytes: tile,
            virtual_bytes: state.virtual_bytes,
            mapped_tiles: state.ledger.used_tiles(),
            heap_bytes: heaps * HEAP_BYTES,
            free_bytes: state.ledger.free_tiles() * tile,
            leaked_tiles: state.ledger.leaked_tiles,
        })
    }

    #[cfg(test)]
    fn set_wait_timeout_ms(&self, timeout_ms: u64) -> Result<()> {
        self.shared
            .wait_timeout_ms
            .store(timeout_ms, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// Buffer virtuel de `capacity_rows` lignes de `row_bytes` octets, arrondi
    /// à la tuile ; rien n'est mappé.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si la taille déborde ou si Metal refuse le buffer.
    pub(crate) fn new_buffer(
        &self,
        capacity_rows: usize,
        row_bytes: u64,
    ) -> Result<SparseKvBuffer> {
        let tile = self.shared.tile_bytes;
        let bytes = u64::try_from(capacity_rows)
            .ok()
            .and_then(|rows| rows.checked_mul(row_bytes))
            .and_then(|bytes| bytes.checked_next_multiple_of(tile))
            .filter(|bytes| *bytes > 0 && row_bytes > 0)
            .ok_or_else(|| InferError::Dimension("buffer sparse vide ou trop grand".to_string()))?;
        let buffer = new_sparse_buffer(&self.shared.device, bytes)?;
        self.shared.lock()?.virtual_bytes += bytes;
        Ok(SparseKvBuffer {
            pool: Arc::clone(&self.shared),
            buffer,
            row_bytes,
            capacity_rows,
            valid_rows: 0,
            tiles: Vec::new(),
            poisoned: false,
            last_proof: None,
        })
    }
}

/// Preuve de fin d'usage : `event` atteindra `value` quand les dernières
/// commandes de l'appelant qui touchent le buffer seront terminées
/// (`encode_signal_event` sur la file classique après elles).
#[derive(Clone, Debug)]
pub(crate) struct UseProof {
    pub(crate) event: SharedEvent,
    pub(crate) value: u64,
}

/// Buffer K/V privé virtuel. Trois longueurs distinctes : capacité virtuelle
/// ([`Self::capacity_rows`]), lignes mappées ([`Self::mapped_rows`]) et lignes
/// valides ([`Self::valid_rows`], tenues par l'appelant).
pub(crate) struct SparseKvBuffer {
    pool: Arc<PoolShared>,
    buffer: Buffer,
    row_bytes: u64,
    capacity_rows: usize,
    valid_rows: usize,
    /// Tuile `i` du buffer → tuile du pool ; toujours un préfixe contigu.
    tiles: Vec<Tile>,
    /// Mapping incertain après un échec : plus aucune opération, fuite au drop.
    poisoned: bool,
    last_proof: Option<UseProof>,
}

impl SparseKvBuffer {
    /// Buffer à lier (lecture ou écriture GPU des lignes mappées seulement).
    pub(crate) fn buffer(&self) -> &BufferRef {
        &self.buffer
    }

    pub(crate) fn capacity_rows(&self) -> usize {
        self.capacity_rows
    }

    pub(crate) fn valid_rows(&self) -> usize {
        self.valid_rows
    }

    pub(crate) fn mapped_bytes(&self) -> u64 {
        self.tiles.len() as u64 * self.pool.tile_bytes
    }

    /// Lignes entières couvertes par les tuiles mappées, bornées à la capacité.
    pub(crate) fn mapped_rows(&self) -> usize {
        usize::try_from(self.mapped_bytes() / self.row_bytes)
            .map_or(self.capacity_rows, |rows| rows.min(self.capacity_rows))
    }

    pub(crate) fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    /// Déclare `rows` lignes écrites et lisibles.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si ces lignes ne sont pas toutes mappées.
    pub(crate) fn set_valid_rows(&mut self, rows: usize) -> Result<()> {
        if rows > self.mapped_rows() {
            return Err(InferError::Dimension(format!(
                "{rows} lignes valides pour {} mappées",
                self.mapped_rows()
            )));
        }
        self.valid_rows = rows;
        Ok(())
    }

    /// Retient la preuve la plus récente pour le drop.
    pub(crate) fn record_use(&mut self, proof: UseProof) {
        self.last_proof = Some(proof);
    }

    fn tiles_for_rows(&self, rows: usize) -> Result<usize> {
        u64::try_from(rows)
            .ok()
            .and_then(|rows| rows.checked_mul(self.row_bytes))
            .map(|bytes| bytes.div_ceil(self.pool.tile_bytes))
            .and_then(|tiles| usize::try_from(tiles).ok())
            .ok_or_else(|| InferError::Dimension("lignes sparse trop nombreuses".to_string()))
    }

    fn ensure_usable(&self) -> Result<()> {
        if self.poisoned {
            return Err(InferError::Metal(
                "buffer K/V sparse empoisonné (mapping incertain)".to_string(),
            ));
        }
        Ok(())
    }

    /// Mappe les tuiles des lignes `0..rows` qui ne le sont pas encore, puis
    /// attend la confirmation MTL4 : une commande soumise ensuite peut les
    /// écrire. Un nouveau heap est réservé auprès de la garde mémoire.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si `rows` dépasse la capacité, si la garde ou Metal
    /// refusent un heap, ou si la confirmation n'arrive pas : le buffer est
    /// alors empoisonné et ses tuiles ne sont jamais recyclées.
    pub(crate) fn ensure_rows(&mut self, rows: usize) -> Result<()> {
        self.ensure_usable()?;
        if rows > self.capacity_rows {
            return Err(InferError::Dimension(format!(
                "{rows} lignes sparse pour une capacité de {}",
                self.capacity_rows
            )));
        }
        let needed = self.tiles_for_rows(rows)?;
        if needed <= self.tiles.len() {
            return Ok(());
        }
        let missing = needed - self.tiles.len();
        let pool = Arc::clone(&self.pool);
        let mut state = pool.lock()?;
        let mut taken = Vec::with_capacity(missing);
        state.ledger.take(missing, &mut taken);
        while taken.len() < missing {
            // NOTE: la réservation vit avec le heap. Mesuré sur M5 Max : un heap
            // placement n'entre dans `phys_footprint` qu'à la première écriture
            // de ses tuiles, et sa destruction n'en sort qu'en différé
            // (~0,5 s) ; une réservation rendue à la création laisserait
            // passer plusieurs heaps contre un footprint encore nul.
            let heap = pool
                .guard
                .reserve_allocation(HEAP_BYTES)
                .map_err(InferError::MemoryGuard)
                .and_then(|reservation| {
                    Ok(PoolHeap {
                        heap: new_placement_heap(&pool.device, HEAP_BYTES)?,
                        _reservation: reservation,
                    })
                });
            match heap {
                Ok(heap) => state.ledger.add_heap(heap),
                Err(error) => {
                    // Jamais mappées : elles retournent sans risque au pool.
                    state.ledger.recycle(&taken);
                    drop(state.ledger.drain_empty_heaps());
                    return Err(error);
                }
            }
            state.ledger.take(missing, &mut taken);
        }
        let first = self.tiles.len();
        let mut submitted = Ok(());
        for (run_start, run) in contiguous_runs(&taken) {
            let op = MappingOperation {
                mode: MAPPING_MODE_MAP,
                buffer_range: NSRange::new((first + run_start) as u64, run.len() as u64),
                heap_offset: NSUInteger::from(run[0].index),
            };
            let heap = state.ledger.slots.get(run[0].slot).and_then(Option::as_ref);
            submitted = submitted.and_then(|()| match heap {
                Some(slot) => state
                    .queue
                    .update(&self.buffer, Some(&slot.heap.heap), &[op]),
                None => Err(InferError::Metal("tuile sparse sans heap".to_string())),
            });
        }
        let timeout_ms = pool.timeout_ms();
        let confirmed = submitted.is_ok() && state.confirm(timeout_ms);
        self.tiles.extend(taken);
        if !confirmed {
            self.poison(&mut state);
            return Err(InferError::Metal(format!(
                "mapping sparse non confirmé en {timeout_ms} ms"
            )));
        }
        Ok(())
    }

    /// Rend les tuiles au-delà de la ligne `row`, en gardant la tuile
    /// partielle qui la contient. `quiesced` prouve que plus aucune commande
    /// ne touche ces tuiles : elle est attendue (bornée) avant l'unmap. Sans
    /// tuile à rendre, seule la longueur valide change.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le buffer est empoisonné, ou si la preuve ou la
    /// confirmation de l'unmap n'arrivent pas : le buffer est alors
    /// empoisonné et ses tuiles perdues.
    pub(crate) fn release_from(&mut self, row: usize, quiesced: UseProof) -> Result<()> {
        self.ensure_usable()?;
        let keep = self.tiles_for_rows(row)?;
        if keep >= self.tiles.len() {
            self.valid_rows = self.valid_rows.min(row);
            self.record_use(quiesced);
            return Ok(());
        }
        let pool = Arc::clone(&self.pool);
        let timeout_ms = pool.timeout_ms();
        // Attente hors du verrou : un decode lent ne bloque pas les mappings
        // des autres buffers.
        if !wait_signaled(&quiesced.event, quiesced.value, timeout_ms) {
            // Une commande peut encore lire : aucune preuve plus ancienne ne
            // doit servir au drop, qui perdra le buffer.
            self.last_proof = None;
            let mut state = pool.lock()?;
            self.poison(&mut state);
            return Err(InferError::Metal(format!(
                "preuve de fin d'usage {} non atteinte en {timeout_ms} ms : buffer empoisonné",
                quiesced.value
            )));
        }
        self.valid_rows = self.valid_rows.min(row);
        self.record_use(quiesced);
        let mut state = pool.lock()?;
        let op = MappingOperation {
            mode: MAPPING_MODE_UNMAP,
            buffer_range: NSRange::new(keep as u64, (self.tiles.len() - keep) as u64),
            heap_offset: 0,
        };
        let submitted = state.queue.update(&self.buffer, None, &[op]);
        if submitted.is_err() || !state.confirm(timeout_ms) {
            self.poison(&mut state);
            return Err(InferError::Metal(format!(
                "unmap sparse non confirmé en {timeout_ms} ms"
            )));
        }
        let released = self.tiles.split_off(keep);
        state.ledger.recycle(&released);
        drop(state.ledger.drain_empty_heaps());
        Ok(())
    }

    /// Déclare les heaps des tuiles mappées sur `encoder`. À appeler sur chaque
    /// encodeur compute qui lie [`Self::buffer`]. Un encodeur blit n'a pas
    /// d'équivalent : un residency set tranchera en A2.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le verrou du pool est empoisonné.
    pub(crate) fn use_heaps(&self, encoder: &ComputeCommandEncoderRef) -> Result<()> {
        let state = self.pool.lock()?;
        let mut slots: Vec<usize> = self.tiles.iter().map(|tile| tile.slot).collect();
        slots.sort_unstable();
        slots.dedup();
        for slot in slots {
            if let Some(Some(heap)) = state.ledger.slots.get(slot) {
                encoder.use_heap(&heap.heap.heap);
            }
        }
        Ok(())
    }

    /// Empoisonne le buffer : ses tuiles ne reviendront jamais au pool.
    fn poison(&mut self, state: &mut PoolState) {
        self.poisoned = true;
        state.ledger.leak(&self.tiles);
        self.tiles.clear();
    }

    /// Perd le buffer et ses tuiles : l'objet `MTLBuffer` n'est jamais rendu,
    /// donc ses mappings restent valides pour une commande encore en vol.
    fn leak(&mut self, reason: &str) {
        let tiles = self.tiles.len();
        // Un verrou empoisonné est repris quand même : sans la marque `leaked`,
        // la destruction du dernier pool libérerait des heaps encore mappés par
        // ce buffer. Marquer des tuiles perdues ne peut pas aggraver l'état.
        let mut state = self
            .pool
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !self.poisoned {
            state.ledger.leak(&self.tiles);
        }
        drop(state);
        self.tiles.clear();
        std::mem::forget(self.buffer.clone());
        eprintln!("saragossa: buffer K/V sparse perdu ({reason}, {tiles} tuiles) : jamais recyclé");
    }
}

impl Drop for SparseKvBuffer {
    fn drop(&mut self) {
        if self.poisoned {
            self.leak("empoisonné");
            return;
        }
        if !self.tiles.is_empty() {
            let Some(proof) = self.last_proof.take() else {
                self.leak("sans preuve de fin d'usage");
                return;
            };
            if let Err(error) = self.release_from(0, proof) {
                self.leak(&error.to_string());
                return;
            }
        }
        if let Ok(mut state) = self.pool.lock() {
            state.virtual_bytes = state.virtual_bytes.saturating_sub(self.buffer.length());
        }
    }
}

/// Découpe `tiles` en plages contiguës à la fois dans le buffer et dans un
/// même heap : `(début dans tiles, plage)`.
fn contiguous_runs(tiles: &[Tile]) -> impl Iterator<Item = (usize, &[Tile])> {
    let mut start = 0;
    std::iter::from_fn(move || {
        if start >= tiles.len() {
            return None;
        }
        let mut end = start + 1;
        while end < tiles.len()
            && tiles[end].slot == tiles[start].slot
            && tiles[end].index == tiles[end - 1].index + 1
        {
            end += 1;
        }
        let run = (start, &tiles[start..end]);
        start = end;
        Some(run)
    })
}

#[cfg(test)]
#[path = "sparse_tests.rs"]
mod tests;
