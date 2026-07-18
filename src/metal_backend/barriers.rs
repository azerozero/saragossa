//! Barrières, fins d'encodage et synchronisation des command buffers Metal.

use super::flags::profile_env_flag;
use super::profiling::{DECODE_PROFILE_CB, DECODE_PROFILE_WAIT_NS};
use super::*;

thread_local! {
    static DISPATCH_BARRIER_SCOPE: Cell<bool> = const { Cell::new(false) };
}

pub(crate) struct DispatchBarrierScopeGuard(bool);

impl Drop for DispatchBarrierScopeGuard {
    fn drop(&mut self) {
        DISPATCH_BARRIER_SCOPE.with(|slot| slot.set(self.0));
    }
}

pub(crate) fn install_dispatch_barrier_scope() -> DispatchBarrierScopeGuard {
    DISPATCH_BARRIER_SCOPE.with(|slot| {
        let previous = slot.replace(true);
        DispatchBarrierScopeGuard(previous)
    })
}

pub(crate) fn suspend_dispatch_barrier_scope() -> DispatchBarrierScopeGuard {
    DISPATCH_BARRIER_SCOPE.with(|slot| {
        let previous = slot.replace(false);
        DispatchBarrierScopeGuard(previous)
    })
}

#[allow(
    unsafe_code,
    reason = "metal-rs 0.29 n'expose pas memoryBarrierWithScope pour compute"
)]
pub(crate) fn memory_barrier_buffers(encoder: &ComputeCommandEncoderRef) {
    const MTL_BARRIER_SCOPE_BUFFERS: NSUInteger = 1 << 0;
    type MemoryBarrierWithScope = unsafe extern "C" fn(*mut Object, Sel, NSUInteger);
    static MEMORY_BARRIER_SCOPE_SELECTOR: OnceLock<Sel> = OnceLock::new();

    unsafe extern "C" {
        fn objc_msgSend();
    }

    // SAFETY: `encoder` est un `id<MTLComputeCommandEncoder>` valide fourni par
    // metal-rs ; `MTLBarrierScopeBuffers` vaut `1 << 0` dans les headers Metal et
    // la méthode est disponible depuis macOS 10.14. Le cast de `objc_msgSend`
    // correspond exactement à `-[MTLComputeCommandEncoder memoryBarrierWithScope:]`.
    unsafe {
        let selector = *MEMORY_BARRIER_SCOPE_SELECTOR
            .get_or_init(|| sel_registerName(c"memoryBarrierWithScope:".as_ptr()));
        let send: MemoryBarrierWithScope = std::mem::transmute(objc_msgSend as *const ());
        send(
            encoder.as_ptr().cast::<Object>(),
            selector,
            MTL_BARRIER_SCOPE_BUFFERS,
        );
    }
}

#[allow(
    unsafe_code,
    reason = "metal-rs 0.29 n'expose pas memoryBarrierWithResources:count: pour compute"
)]
pub(crate) fn memory_barrier_buffer_list(
    encoder: &ComputeCommandEncoderRef,
    buffers: &[&BufferRef],
) {
    type MemoryBarrierWithResources =
        unsafe extern "C" fn(*mut Object, Sel, *const *mut Object, NSUInteger);
    static MEMORY_BARRIER_RESOURCES_SELECTOR: OnceLock<Sel> = OnceLock::new();

    if buffers.is_empty() {
        return;
    }
    if buffers.len() > 8 {
        memory_barrier_buffers(encoder);
        return;
    }

    unsafe extern "C" {
        fn objc_msgSend();
    }

    // SAFETY: `encoder` est un `id<MTLComputeCommandEncoder>` valide et
    // chaque entree de `buffers` est un `id<MTLBuffer>` valide vivant au moins
    // jusqu'a la fin de l'encodage courant. La signature correspond a
    // `-[MTLComputeCommandEncoder memoryBarrierWithResources:count:]`.
    unsafe {
        let selector = *MEMORY_BARRIER_RESOURCES_SELECTOR
            .get_or_init(|| sel_registerName(c"memoryBarrierWithResources:count:".as_ptr()));
        let send: MemoryBarrierWithResources = std::mem::transmute(objc_msgSend as *const ());
        let mut resources = [std::ptr::null_mut::<Object>(); 8];
        for (slot, buffer) in resources.iter_mut().zip(buffers.iter()) {
            *slot = buffer.as_ptr().cast::<Object>();
        }
        send(
            encoder.as_ptr().cast::<Object>(),
            selector,
            resources.as_ptr(),
            buffers.len() as NSUInteger,
        );
    }
}

pub(crate) fn post_dispatch_barrier(encoder: &ComputeCommandEncoderRef) {
    DISPATCH_BARRIER_SCOPE.with(|slot| {
        if slot.get() {
            memory_barrier_buffers(encoder);
        }
    });
}

pub(crate) fn post_dispatch_barrier_buffer(encoder: &ComputeCommandEncoderRef, buffer: &BufferRef) {
    post_dispatch_barrier_buffers(encoder, &[buffer]);
}

pub(crate) fn post_dispatch_barrier_buffers(
    encoder: &ComputeCommandEncoderRef,
    buffers: &[&BufferRef],
) {
    DISPATCH_BARRIER_SCOPE.with(|slot| {
        if slot.get() {
            if resource_barriers_enabled() {
                memory_barrier_buffer_list(encoder, buffers);
            } else {
                memory_barrier_buffers(encoder);
            }
        }
    });
}

pub(super) fn resource_barriers_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| profile_env_flag("RETI_RUST_RESOURCE_BARRIERS", false))
}

pub(crate) struct EncoderEndGuard<'a> {
    encoder: &'a ComputeCommandEncoderRef,
    ended: bool,
}

impl<'a> EncoderEndGuard<'a> {
    pub(crate) fn new(encoder: &'a ComputeCommandEncoderRef) -> Self {
        Self {
            encoder,
            ended: false,
        }
    }

    pub(crate) fn encoder(&self) -> &'a ComputeCommandEncoderRef {
        self.encoder
    }

    pub(crate) fn end(mut self) {
        self.encoder.end_encoding();
        self.ended = true;
    }
}

impl Drop for EncoderEndGuard<'_> {
    fn drop(&mut self) {
        if !self.ended {
            self.encoder.end_encoding();
        }
    }
}

/// `commit()` + `wait_until_completed()` + `ensure_completed`, centralisant le
/// **point de synchronisation CPU↔GPU bloquant** (cf. plan : ~80-120/token).
/// Sous `RETI_RUST_DECODE_PROFILE`, compte le command buffer et chronomètre le
/// wait. Hors profil : strictement l'ancien comportement (commit+wait+ensure).
pub(crate) fn commit_and_wait(command_buffer: &metal::CommandBufferRef) -> Result<()> {
    use std::sync::atomic::Ordering::Relaxed;
    if commit_components_enabled() {
        let label = COMMIT_LABEL.with(std::cell::Cell::get);
        let started = std::time::Instant::now();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        record_commit_component(label, started.elapsed().as_secs_f64());
        return ensure_completed(command_buffer.status());
    }
    if decode_profile_enabled() {
        DECODE_PROFILE_CB.fetch_add(1, Relaxed);
        let started = std::time::Instant::now();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        DECODE_PROFILE_WAIT_NS.fetch_add(started.elapsed().as_nanos() as u64, Relaxed);
    } else {
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }
    ensure_completed(command_buffer.status())
}

// Brick #5 campagne : ventilation du temps GPU (commit_and_wait synchrone) par
// composant du prefill, via un label thread-local. Gaté RETI_RUST_TRACE_COMPONENTS.
thread_local! {
    static COMMIT_LABEL: std::cell::Cell<&'static str> = const { std::cell::Cell::new("other") };
}
#[allow(
    clippy::type_complexity,
    reason = "map de profilage label → (cumul_ms, count) derrière OnceLock<Mutex<…>>"
)]
static COMMIT_COMPONENTS: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashMap<&'static str, (f64, u64)>>,
> = std::sync::OnceLock::new();

pub(crate) fn commit_components_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| profile_env_flag("RETI_RUST_TRACE_COMPONENTS", false))
}

/// Fixe le label du prochain `commit_and_wait` (no-op si le traçage est désactivé).
pub(crate) fn set_commit_label(label: &'static str) {
    if commit_components_enabled() {
        COMMIT_LABEL.with(|cell| cell.set(label));
    }
}

fn record_commit_component(label: &'static str, secs: f64) {
    let map =
        COMMIT_COMPONENTS.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    if let Ok(mut guard) = map.lock() {
        let entry = guard.entry(label).or_insert((0.0, 0));
        entry.0 += secs;
        entry.1 += 1;
    }
}

/// Imprime la ventilation par composant accumulée (triée par temps décroissant).
pub fn dump_commit_components() {
    let Some(map) = COMMIT_COMPONENTS.get() else {
        return;
    };
    if let Ok(guard) = map.lock() {
        let mut rows: Vec<(&'static str, f64, u64)> =
            guard.iter().map(|(k, (t, n))| (*k, *t, *n)).collect();
        rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let total: f64 = rows.iter().map(|r| r.1).sum();
        for (label, secs, n) in rows {
            eprintln!(
                "[components] {label}: {:.0} ms ({n} CB, {:.0}%)",
                secs * 1.0e3,
                if total > 0.0 {
                    100.0 * secs / total
                } else {
                    0.0
                }
            );
        }
        eprintln!("[components] TOTAL commit+wait: {:.0} ms", total * 1.0e3);
    }
}

pub(crate) fn commit_nonblocking(command_buffer: &metal::CommandBufferRef) {
    use std::sync::atomic::Ordering::Relaxed;
    if decode_profile_enabled() {
        DECODE_PROFILE_CB.fetch_add(1, Relaxed);
    }
    command_buffer.commit();
}

pub(crate) fn wait_for_completion(command_buffer: &metal::CommandBufferRef) -> Result<()> {
    use std::sync::atomic::Ordering::Relaxed;
    if decode_profile_enabled() {
        let started = std::time::Instant::now();
        command_buffer.wait_until_completed();
        DECODE_PROFILE_WAIT_NS.fetch_add(started.elapsed().as_nanos() as u64, Relaxed);
    } else {
        command_buffer.wait_until_completed();
    }
    ensure_completed(command_buffer.status())
}
