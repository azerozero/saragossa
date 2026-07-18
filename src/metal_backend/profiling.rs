//! Instrumentation des chemins decode et prefill Metal.

use crate::runtime_flags::env_flag;

use super::*;

// ---------------------------------------------------------------------------
// Instrumentation decode (RETI_RUST_DECODE_PROFILE) — phase 1a
//
// Split par token : `encode_us` (CPU, dérivé = total − wait − read) / `wait_us`
// (CPU bloqué sur le GPU dans `wait_until_completed`) / `read_us` (readback
// GPU→CPU) + nombre de command buffers/token. Atomics globaux (le decode est
// mono-thread) ; surcoût NUL quand le flag est absent (seul un test de bool
// caché). Guide la priorisation 1b/1c du decode résident.
// ---------------------------------------------------------------------------
pub(super) static DECODE_PROFILE_CB: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub(super) static DECODE_PROFILE_WAIT_NS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub(super) static DECODE_PROFILE_READ_NS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
static DECODE_PROFILE_DISPATCHES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
static DECODE_PROFILE_DISPATCH_SITES: OnceLock<Mutex<HashMap<DispatchProfileSite, u64>>> =
    OnceLock::new();
static DECODE_PROFILE_DISPATCH_SHAPES: OnceLock<Mutex<HashMap<DispatchProfileShape, u64>>> =
    OnceLock::new();

thread_local! {
    static PREFILL_F32_TO_BF16_SHAPES: std::cell::RefCell<HashMap<usize, u64>> =
        std::cell::RefCell::new(HashMap::new());
}

#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DispatchProfileSite {
    pub file: &'static str,
    pub line: u32,
    pub column: u32,
}

#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DispatchProfileShape {
    pub kind: &'static str,
    pub batch: usize,
    pub lhs_rows: usize,
    pub topk: usize,
    pub in_dim: usize,
    pub out_dim: usize,
    pub group_size: usize,
    pub bits: usize,
}

impl DispatchProfileShape {
    pub(crate) fn matmul(
        kind: &'static str,
        batch: usize,
        in_dim: usize,
        out_dim: usize,
        group_size: usize,
        bits: usize,
    ) -> Self {
        Self {
            kind,
            batch,
            lhs_rows: 0,
            topk: 0,
            in_dim,
            out_dim,
            group_size,
            bits,
        }
    }

    pub(crate) fn gather(
        kind: &'static str,
        lhs_rows: usize,
        topk: usize,
        in_dim: usize,
        out_dim: usize,
        group_size: usize,
        bits: usize,
    ) -> Self {
        Self {
            kind,
            batch: 0,
            lhs_rows,
            topk,
            in_dim,
            out_dim,
            group_size,
            bits,
        }
    }
}

pub(super) fn decode_profile_enabled() -> bool {
    crate::runtime_flags::decode_profile_enabled()
}

fn decode_profile_sites_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_DECODE_PROFILE_SITES", false))
}

/// `(command_buffers, wait_ns, read_ns)` cumulés — bornés par l'appelant
/// (decode loop) pour un split par token. Toujours disponible (zéros si le flag
/// est absent, car `commit_and_wait`/`read_*_buffer` ne cumulent que si activé).
#[doc(hidden)]
pub fn decode_profile_snapshot() -> (u64, u64, u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (
        DECODE_PROFILE_CB.load(Relaxed),
        DECODE_PROFILE_WAIT_NS.load(Relaxed),
        DECODE_PROFILE_READ_NS.load(Relaxed),
        DECODE_PROFILE_DISPATCHES.load(Relaxed),
    )
}

#[doc(hidden)]
pub fn decode_profile_dispatch_sites_snapshot() -> HashMap<DispatchProfileSite, u64> {
    DECODE_PROFILE_DISPATCH_SITES
        .get()
        .and_then(|sites| sites.lock().ok().map(|sites| sites.clone()))
        .unwrap_or_default()
}

#[doc(hidden)]
pub fn decode_profile_dispatch_shapes_snapshot() -> HashMap<DispatchProfileShape, u64> {
    DECODE_PROFILE_DISPATCH_SHAPES
        .get()
        .and_then(|shapes| shapes.lock().ok().map(|shapes| shapes.clone()))
        .unwrap_or_default()
}

#[track_caller]
pub(crate) fn profile_dispatch() {
    use std::sync::atomic::Ordering::Relaxed;
    if decode_profile_enabled() {
        DECODE_PROFILE_DISPATCHES.fetch_add(1, Relaxed);
        if decode_profile_sites_enabled() {
            let location = std::panic::Location::caller();
            let site = DispatchProfileSite {
                file: location.file(),
                line: location.line(),
                column: location.column(),
            };
            let sites = DECODE_PROFILE_DISPATCH_SITES.get_or_init(|| Mutex::new(HashMap::new()));
            if let Ok(mut sites) = sites.lock() {
                *sites.entry(site).or_insert(0) += 1;
            }
        }
    }
}

pub(crate) fn profile_dispatch_shape(shape: DispatchProfileShape) {
    if decode_profile_enabled() && decode_profile_sites_enabled() {
        let shapes = DECODE_PROFILE_DISPATCH_SHAPES.get_or_init(|| Mutex::new(HashMap::new()));
        if let Ok(mut shapes) = shapes.lock() {
            *shapes.entry(shape).or_insert(0) += 1;
        }
    }
}

pub(crate) fn prefill_profile_sections_enabled() -> bool {
    crate::runtime_flags::prefill_profile_sections_enabled()
}

pub(crate) fn reset_prefill_f32_to_bf16_shapes() {
    PREFILL_F32_TO_BF16_SHAPES.with(|shapes| shapes.borrow_mut().clear());
}

pub(crate) fn record_prefill_f32_to_bf16_shape(len: usize) {
    if !prefill_profile_sections_enabled() {
        return;
    }
    PREFILL_F32_TO_BF16_SHAPES.with(|shapes| {
        let mut shapes = shapes.borrow_mut();
        *shapes.entry(len).or_insert(0) += 1;
    });
}

pub(crate) fn take_prefill_f32_to_bf16_shapes() -> Vec<(usize, u64)> {
    PREFILL_F32_TO_BF16_SHAPES.with(|shapes| {
        let mut shapes = shapes.borrow_mut();
        let mut rows = shapes.drain().collect::<Vec<_>>();
        rows.sort_by_key(|(len, _)| *len);
        rows
    })
}

pub(crate) fn trace_dispatch_path(kernel: &str, batch: usize, out_dim: usize, in_dim: usize) {
    if crate::runtime_flags::trace_dispatch_path_enabled() {
        eprintln!("[dispatch_path] {kernel} M={batch} N={out_dim} K={in_dim}");
    }
}
