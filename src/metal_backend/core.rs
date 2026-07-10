//! Helpers communs du backend Metal.

use crate::runtime_flags::{env_flag, gather_fast_mode, qmv_fast_mode};

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
static DECODE_PROFILE_CB: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static DECODE_PROFILE_WAIT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static DECODE_PROFILE_READ_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
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

thread_local! {
    static DISPATCH_BARRIER_SCOPE: Cell<bool> = const { Cell::new(false) };
    // Namespace courant du scratch label-keyed (light-batch) : 0 = chemin
    // historique mono-flux ; un slot par flux isole les buffers mémoïsés par
    // `(label, len, element)` qui sinon s'aliaseraient entre flux concurrents.
    static SCRATCH_NAMESPACE: Cell<u64> = const { Cell::new(0) };
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum KernelOptProfile {
    Default,
    Qwen36Oq8,
    ExploreFused,
}

/// Profil d'optimisation kernel piloté par `RETI_RUST_OPT_PROFILE`.
///
/// Défaut `Default`. PROMOTED infra (décision tangle 2026-07-05) : le preset
/// runtime peut poser `qwen36-oq8`, mais l'env explicite garde la priorité pour
/// les mesures et les retours arrière ciblés.
fn kernel_opt_profile() -> KernelOptProfile {
    static PROFILE: OnceLock<KernelOptProfile> = OnceLock::new();
    *PROFILE.get_or_init(|| match std::env::var("RETI_RUST_OPT_PROFILE") {
        Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
            "qwen36-oq8" | "qwen3.6-oq8" | "linear-bf16" | "bf16" => KernelOptProfile::Qwen36Oq8,
            "explore-fused" | "fused" => KernelOptProfile::ExploreFused,
            _ => KernelOptProfile::Default,
        },
        Err(_) => KernelOptProfile::Default,
    })
}

fn profile_bool_default(name: &str, default: bool) -> bool {
    match kernel_opt_profile() {
        KernelOptProfile::Default => default,
        KernelOptProfile::Qwen36Oq8 => match name {
            "RETI_RUST_LINEAR_SSM_BF16" => true,
            "RETI_RUST_LINEAR_INV_DELTA" => false,
            "RETI_RUST_LINEAR_CONV_NORM_FUSED" => false,
            "RETI_RUST_LINEAR_PAIR_BARRIER_COALESCE" => true,
            "RETI_RUST_QMV_U8_TG128" => true,
            _ => default,
        },
        KernelOptProfile::ExploreFused => match name {
            "RETI_RUST_LINEAR_SSM_BF16" => true,
            "RETI_RUST_LINEAR_INV_DELTA" => false,
            "RETI_RUST_LINEAR_CONV_NORM_FUSED" => true,
            "RETI_RUST_QMV_U8_TG128" => true,
            _ => default,
        },
    }
}

fn profile_env_flag(name: &str, default: bool) -> bool {
    env_flag(name, profile_bool_default(name, default))
}

// Ces flags u8 sont des choix de preset, pas des défauts indépendants : le
// preset Default reste OFF en prod, et l'env explicite garde la priorité pour
// les mesures/explorations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct U8KernelPresetDefaults {
    fused_gate_up: bool,
    fused_shared_gate_up: bool,
    fused_shared_gate_scalar: bool,
    fused_shared_gate_qmv: bool,
    fused_moe_down_weighted: bool,
    qmv_tg256: bool,
    qmv_dot4: bool,
    full_qkv_split_rms: bool,
    full_o_proj_gated: bool,
}

fn u8_kernel_preset_defaults_for(profile: KernelOptProfile) -> U8KernelPresetDefaults {
    match profile {
        KernelOptProfile::Default => U8KernelPresetDefaults {
            fused_gate_up: false,
            fused_shared_gate_up: false,
            fused_shared_gate_scalar: false,
            fused_shared_gate_qmv: false,
            fused_moe_down_weighted: false,
            qmv_tg256: false,
            qmv_dot4: false,
            full_qkv_split_rms: cfg!(test),
            full_o_proj_gated: cfg!(test),
        },
        KernelOptProfile::Qwen36Oq8 => U8KernelPresetDefaults {
            fused_gate_up: false,
            fused_shared_gate_up: false,
            fused_shared_gate_scalar: false,
            fused_shared_gate_qmv: false,
            fused_moe_down_weighted: true,
            qmv_tg256: false,
            qmv_dot4: false,
            full_qkv_split_rms: false,
            full_o_proj_gated: false,
        },
        KernelOptProfile::ExploreFused => U8KernelPresetDefaults {
            fused_gate_up: true,
            fused_shared_gate_up: true,
            fused_shared_gate_scalar: true,
            fused_shared_gate_qmv: true,
            fused_moe_down_weighted: true,
            qmv_tg256: false,
            qmv_dot4: true,
            full_qkv_split_rms: true,
            full_o_proj_gated: true,
        },
    }
}

fn u8_kernel_preset_defaults() -> U8KernelPresetDefaults {
    u8_kernel_preset_defaults_for(kernel_opt_profile())
}

fn profile_linear_delta_tg_rows(ssm_bf16: bool) -> u64 {
    match kernel_opt_profile() {
        KernelOptProfile::Qwen36Oq8 | KernelOptProfile::ExploreFused if ssm_bf16 => 8,
        _ => {
            if ssm_bf16 {
                8
            } else {
                4
            }
        }
    }
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

pub(crate) fn resident_concurrent_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_RESIDENT_CONCURRENT", false))
}

/// Active la linear-attn gated-delta en forme CHUNKÉE (port chunked-DeltaNet) au lieu
/// du séquentiel token-par-token. Opt-in `RETI_RUST_LINEAR_CHUNKED`. Exige head_dim=128.
pub(crate) fn linear_chunked_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_LINEAR_CHUNKED", false))
}

/// Active le MoE routé prefill via le kernel `gather_qmm` porté (cooperative_tensor
/// quantifié groupé, ~7,6× le qmv). Tri+pad par expert (16-aligné, CPU, readback des
/// indices), UN dispatch par projection lisant le poids empilé packed. Opt-in.
pub(crate) fn moe_coop_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_MOE_COOP", true))
}

/// Active F2 : fusion gate+up+SwiGLU du MoE coop résident. Byte-identique :
/// accumulation f32 inchangée puis cast bf16 direct dans l'épilogue.
pub(crate) fn moe_coop_fused_swiglu_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_MOE_COOP_FUSED_SWIGLU", true))
}

/// Active le MoE routé coop u4 gs64 au prefill (kernels
/// `gemm_nax_coop_qb_grouped_*_u4`).
///
/// Défaut ON (décision produit 2026-07-02) : kernel prouvé bit-identique au
/// chemin u8 coop à poids logiques égaux (tests `*_bit_identical_to_u8_*`) ;
/// la déviation greedy possible sur certains prompts = near-ties bf16
/// tensor-core, la MÊME classe que le dense u4/u8 NA déjà défaut ON (byte-
/// identité prompt-dépendante, sortie cohérente). Gain mesuré : prefill voix
/// 4bit @524 tokens 457 → 248 ms (×1,84). Kill-switch : `RETI_RUST_MOE_COOP_U4=0`.
pub(crate) fn moe_coop_u4_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_MOE_COOP_U4", true))
}

/// Active la bascule routed-only coop du prefill (MoE sans expert partagé, ex.
/// Qwen3-30B-A3B) vers les kernels groupés `gemm_nax_coop_qb_grouped_*`.
///
/// Défaut ON (décision produit 2026-07-04, politique near-tie). Historique :
/// la « divergence » initiale de D-30B était contaminée par le bug du plafond
/// 256 de l'attention prefill (sortie non écrite → déchet déterministe) ;
/// après ce fix, le dossier C1b texte réel (D-COOP-2) qualifie une classe
/// near-tie PROPRE : ids byte-identiques à ~1k ET ~9k tokens, un seul flip à
/// ~4k (marge 0,0084), zéro dégénérescence, zéro dérive d'accumulation ;
/// oracle multi-tuiles gather↔coop aux dimensions 30B : max_abs 1,8e-4
/// (l'ordre de réduction tensor-core groupé ≠ qmv par-ligne, byte-id
/// structurellement hors d'atteinte). Gain mesuré (tronc avec attention
/// batchée) : prefill 30B ×1,25-1,50 sur toute la courbe (@8k 345→517 tok/s,
/// @32k 138→179). Gate INDÉPENDANT de `RETI_RUST_MOE_COOP` (shared 35B).
/// Kill-switch : `RETI_RUST_MOE_ROUTED_COOP_PREFILL=0`.
pub(crate) fn moe_routed_coop_prefill_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_MOE_ROUTED_COOP_PREFILL", true))
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

#[allow(
    unsafe_code,
    reason = "lecture d'un MTLBuffer partagé après wait_until_completed"
)]
fn read_pod_buffer<T: bytemuck::Pod + Default>(buffer: &BufferRef, len: usize) -> Result<Vec<T>> {
    let read_started = decode_profile_enabled().then(std::time::Instant::now);
    let ptr = buffer.contents().cast::<T>();
    if ptr.is_null() {
        return Err(InferError::Metal(
            "MTLBuffer StorageModeShared sans pointeur CPU".to_string(),
        ));
    }
    // `out` est zéro-initialisé (aucune fenêtre non-initialisée → lint
    // `uninit_vec` évité) ; ses `len` éléments sont ensuite ÉCRASÉS par la copie.
    let mut out = vec![T::default(); len];
    // SAFETY: `buffer` est en StorageModeShared et son command buffer a terminé ;
    // on copie exactement `len` valeurs POD depuis son pointeur CPU vers `out` (même
    // longueur, déjà allouée), sans chevauchement.
    unsafe {
        std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), len);
    }
    if let Some(started) = read_started {
        DECODE_PROFILE_READ_NS.fetch_add(
            started.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
    }
    Ok(out)
}

pub(crate) fn read_f32_buffer(buffer: &BufferRef, len: usize) -> Result<Vec<f32>> {
    read_pod_buffer(buffer, len)
}

pub(crate) fn read_u32_buffer(buffer: &BufferRef, len: usize) -> Result<Vec<u32>> {
    read_pod_buffer(buffer, len)
}

#[cfg(any(test, feature = "devtools"))]
pub(crate) fn read_u16_buffer(buffer: &BufferRef, len: usize) -> Result<Vec<u16>> {
    read_pod_buffer(buffer, len)
}

#[allow(
    unsafe_code,
    reason = "écriture d'un MTLBuffer partagé avant commit du command buffer"
)]
fn write_pod_buffer<T: bytemuck::Pod>(buffer: &BufferRef, data: &[T]) -> Result<()> {
    let ptr = buffer.contents().cast::<T>();
    if ptr.is_null() {
        return Err(InferError::Metal(
            "MTLBuffer StorageModeShared sans pointeur CPU".to_string(),
        ));
    }
    // SAFETY: `buffer` est en StorageModeShared, et l'appelant fournit un
    // scratch dimensionné pour exactement `data.len()` éléments POD.
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
    }
    Ok(())
}

pub(crate) fn write_f32_buffer(buffer: &BufferRef, data: &[f32]) -> Result<()> {
    write_pod_buffer(buffer, data)
}

pub(super) fn write_u32_buffer(buffer: &BufferRef, data: &[u32]) -> Result<()> {
    write_pod_buffer(buffer, data)
}

pub(super) fn set_u32_bytes(
    encoder: &ComputeCommandEncoderRef,
    index: NSUInteger,
    data: &[u32],
    label: &'static str,
) -> Result<()> {
    let len = byte_len_nsuint::<u32>(data.len(), label)?;
    encoder.set_bytes(index, len, data.as_ptr().cast::<c_void>());
    Ok(())
}

pub(super) fn set_f32_bytes(
    encoder: &ComputeCommandEncoderRef,
    index: NSUInteger,
    data: &[f32],
    label: &'static str,
) -> Result<()> {
    let len = byte_len_nsuint::<f32>(data.len(), label)?;
    encoder.set_bytes(index, len, data.as_ptr().cast::<c_void>());
    Ok(())
}

pub(super) fn set_u64_bytes(
    encoder: &ComputeCommandEncoderRef,
    index: NSUInteger,
    data: &[u64],
    label: &'static str,
) -> Result<()> {
    let len = byte_len_nsuint::<u64>(data.len(), label)?;
    encoder.set_bytes(index, len, data.as_ptr().cast::<c_void>());
    Ok(())
}

pub(super) fn ensure_completed(status: MTLCommandBufferStatus) -> Result<()> {
    if status == MTLCommandBufferStatus::Completed {
        Ok(())
    } else {
        Err(InferError::Metal(format!(
            "command buffer terminé avec le statut {status:?}"
        )))
    }
}

pub(super) fn ensure_biasless(linear: &Linear, label: &'static str) -> Result<()> {
    if linear.bias().is_some() {
        return Err(InferError::Config(format!(
            "MoE Metal ne supporte pas les biais expert {label}"
        )));
    }
    Ok(())
}

pub(super) fn dense_vector<'a>(
    tensor: &'a Tensor,
    dim: usize,
    label: &'static str,
) -> Result<&'a [f32]> {
    match tensor.shape() {
        [n] if *n == dim => Ok(tensor.data()),
        [1, n] if *n == dim => Ok(tensor.data()),
        shape => Err(InferError::Dimension(format!(
            "{label} attendu [{dim}] ou [1,{dim}], reçu {shape:?}"
        ))),
    }
}

pub(super) fn linear_out_dim(weight: &LinearWeight) -> Result<usize> {
    match weight.shape() {
        [out_dim, _] => Ok(*out_dim),
        shape => Err(InferError::Dimension(format!(
            "poids Linear attendu rang 2, reçu {shape:?}"
        ))),
    }
}

pub(super) fn linear_in_dim(weight: &LinearWeight) -> Result<usize> {
    match weight.shape() {
        [_, in_dim] => Ok(*in_dim),
        shape => Err(InferError::Dimension(format!(
            "poids Linear attendu rang 2, reçu {shape:?}"
        ))),
    }
}

pub(super) fn expect_linear_shape(
    weight: &LinearWeight,
    expected_out: usize,
    expected_in: usize,
    label: &'static str,
) -> Result<()> {
    match weight.shape() {
        [out_dim, in_dim] if *out_dim == expected_out && *in_dim == expected_in => Ok(()),
        shape => Err(InferError::Dimension(format!(
            "{label}.weight attendu [{expected_out},{expected_in}], reçu {shape:?}"
        ))),
    }
}

pub(super) fn expect_linear_in(
    weight: &LinearWeight,
    expected_in: usize,
    label: &'static str,
) -> Result<()> {
    match weight.shape() {
        [_, in_dim] if *in_dim == expected_in => Ok(()),
        shape => Err(InferError::Dimension(format!(
            "{label}.weight entrée attendue {expected_in}, reçu {shape:?}"
        ))),
    }
}

pub(super) fn can_use_fast_affine_qmv(
    batch: usize,
    in_dim: usize,
    weight: &AffineQuantizedTensor,
) -> bool {
    let Some(out_dim) = fast_affine_qmv_out_dim(weight) else {
        return false;
    };
    fast_affine_qmv_enabled(out_dim) && can_use_fast_affine_qmv_shape(batch, in_dim, weight)
}

pub(super) fn can_use_fast_affine_qmv_shape(
    batch: usize,
    in_dim: usize,
    weight: &AffineQuantizedTensor,
) -> bool {
    batch > 0
        && weight.bits() == FAST_QMV_BITS
        && weight.group_size() == FAST_QMV_GROUP_SIZE
        && in_dim % 512 == 0
        && matches!(weight.shape(), [_, weight_in_dim] if *weight_in_dim == in_dim)
}

pub(super) fn can_use_fast_affine_qmv_u6(
    batch: usize,
    in_dim: usize,
    weight: &AffineQuantizedTensor,
) -> bool {
    let Some(out_dim) = fast_affine_qmv_out_dim(weight) else {
        return false;
    };
    can_use_fast_affine_qmv_u6_buffers(batch, in_dim, out_dim, weight.group_size(), weight.bits())
        && matches!(weight.shape(), [_, weight_in_dim] if *weight_in_dim == in_dim)
}

pub(super) fn can_use_fast_affine_qmm2(
    batch: usize,
    in_dim: usize,
    weight: &AffineQuantizedTensor,
) -> bool {
    let Some(out_dim) = fast_affine_qmv_out_dim(weight) else {
        return false;
    };
    can_use_fast_affine_qmm2_buffers(batch, in_dim, out_dim, weight.group_size(), weight.bits())
        && matches!(weight.shape(), [_, weight_in_dim] if *weight_in_dim == in_dim)
}

pub(super) fn can_use_fast_affine_qmv_u8(
    batch: usize,
    in_dim: usize,
    weight: &AffineQuantizedTensor,
) -> bool {
    let Some(out_dim) = fast_affine_qmv_out_dim(weight) else {
        return false;
    };
    can_use_fast_affine_qmv_u8_buffers(batch, in_dim, out_dim, weight.group_size(), weight.bits())
        && matches!(weight.shape(), [_, weight_in_dim] if *weight_in_dim == in_dim)
}

pub(super) fn qmm_na_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_QMM_NA", true))
}

pub(super) fn qmm_na_gs128_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_QMM_NA_GS128", true))
}

/// Active le GEMM NA fused-tiled comme tête de précédence prefill.
///
/// Défaut ON (décision produit 2026-07-02, C3) : le chemin dense/u8 NA qualifié
/// devient le routeur principal ; kill-switch `RETI_RUST_QMM_NA_FUSED_TILED=0`.
pub(super) fn qmm_na_fused_tiled_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_QMM_NA_FUSED_TILED", true))
}

/// Active le sous-chemin u4 du GEMM NA fused-tiled.
///
/// Défaut ON (décision produit 2026-07-02, C3) sous
/// `RETI_RUST_QMM_NA_FUSED_TILED`; kill-switch
/// `RETI_RUST_QMM_NA_FUSED_TILED_U4=0`.
pub(super) fn qmm_na_fused_tiled_u4_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_QMM_NA_FUSED_TILED_U4", true))
}

/// Prédicat du GEMM NA tuilé à dé-quant fusionnée : A bf16, B u8 staged en
/// threadgroup bf16 par tuile BM=BN=BK=64. Opt-in, car le gain dépend du trafic
/// poids et les résultats suivent l'ordre d'accumulation tuilé.
pub(super) fn can_use_qmm_na_fused_tiled_u8_buffers(
    batch: usize,
    in_dim: usize,
    out_dim: usize,
    group_size: usize,
    bits: usize,
) -> bool {
    qmm_na_fused_tiled_enabled()
        && batch >= 16
        && out_dim % 64 == 0
        && bits == 8
        && matches!(group_size, FAST_QMV_GROUP_SIZE | QMM_NA_GS128_GROUP_SIZE)
        && in_dim % 512 == 0
}

pub(super) fn can_use_qmm_na_fused_tiled_u8(
    batch: usize,
    in_dim: usize,
    weight: &AffineQuantizedTensor,
) -> bool {
    let Some(out_dim) = fast_affine_qmv_out_dim(weight) else {
        return false;
    };
    can_use_qmm_na_fused_tiled_u8_buffers(
        batch,
        in_dim,
        out_dim,
        weight.group_size(),
        weight.bits(),
    ) && matches!(weight.shape(), [_, weight_in_dim] if *weight_in_dim == in_dim)
}

/// Variante 4-bit gs64 du GEMM NA tuilé. Le chemin reste borné aux grandes
/// projections denses : les petites projections shared-expert divergent en
/// oracle greedy avec l'accumulation bf16 tensor-core et gardent le qmv f32.
pub(super) fn can_use_qmm_na_fused_tiled_u4_buffers(
    batch: usize,
    in_dim: usize,
    out_dim: usize,
    group_size: usize,
    bits: usize,
) -> bool {
    qmm_na_fused_tiled_enabled()
        && qmm_na_fused_tiled_u4_enabled()
        && batch >= 16
        && out_dim % 64 == 0
        && out_dim >= 2048
        && in_dim >= 2048
        && bits == 4
        && group_size == FAST_QMV_GROUP_SIZE
        && in_dim % 512 == 0
}

pub(super) fn can_use_qmm_na_fused_tiled_u4(
    batch: usize,
    in_dim: usize,
    weight: &AffineQuantizedTensor,
) -> bool {
    let Some(out_dim) = fast_affine_qmv_out_dim(weight) else {
        return false;
    };
    can_use_qmm_na_fused_tiled_u4_buffers(
        batch,
        in_dim,
        out_dim,
        weight.group_size(),
        weight.bits(),
    ) && matches!(weight.shape(), [_, weight_in_dim] if *weight_in_dim == in_dim)
}

/// Prédicat du GEMM prefill sur Neural Accelerators (matmul2d bf16) : dé-quant
/// u8→bf16 transposée du poids + activations bf16 + tensor-cores. `batch` grand
/// (prefill). Opt-in (`RETI_RUST_QMM_NA`) ; l'appelant vérifie EN PLUS que la NA est
/// dispo (`na_gemm_bf16.is_some()`, macOS≥26). bf16 ⇒ non bit-à-bit identique.
pub(super) fn can_use_qmm_na_u8(
    batch: usize,
    in_dim: usize,
    weight: &AffineQuantizedTensor,
) -> bool {
    qmm_na_enabled()
        && batch >= 16
        && weight.bits() == 8
        && weight.group_size() == FAST_QMV_GROUP_SIZE
        && in_dim % 512 == 0
        && matches!(weight.shape(), [_, weight_in_dim] if *weight_in_dim == in_dim)
}

/// Prédicat du GEMM NA fusé dense gs128 : dé-quant u8 directement dans le kernel.
/// Le kernel masque la queue M ; N reste aligné 32 pour éviter les stores hors poids.
pub(super) fn can_use_qmm_na_u8_gs128_buffers(
    batch: usize,
    in_dim: usize,
    out_dim: usize,
    group_size: usize,
    bits: usize,
) -> bool {
    qmm_na_gs128_enabled()
        && batch >= 16
        && out_dim % 32 == 0
        && bits == 8
        && group_size == QMM_NA_GS128_GROUP_SIZE
        && in_dim % 512 == 0
}

pub(super) fn can_use_qmm_na_u8_gs128(
    batch: usize,
    in_dim: usize,
    weight: &AffineQuantizedTensor,
) -> bool {
    let Some(out_dim) = fast_affine_qmv_out_dim(weight) else {
        return false;
    };
    can_use_qmm_na_u8_gs128_buffers(batch, in_dim, out_dim, weight.group_size(), weight.bits())
        && matches!(weight.shape(), [_, weight_in_dim] if *weight_in_dim == in_dim)
}

pub(super) fn can_use_fast_affine_qmv_one_u8(
    batch: usize,
    in_dim: usize,
    weight: &AffineQuantizedTensor,
) -> bool {
    let Some(out_dim) = fast_affine_qmv_out_dim(weight) else {
        return false;
    };
    can_use_fast_affine_qmv_one_u8_buffers(
        batch,
        in_dim,
        out_dim,
        weight.group_size(),
        weight.bits(),
    ) && matches!(weight.shape(), [_, weight_in_dim] if *weight_in_dim == in_dim)
}

pub(super) fn can_use_fast_affine_qmm2_u8(
    batch: usize,
    in_dim: usize,
    weight: &AffineQuantizedTensor,
) -> bool {
    let Some(out_dim) = fast_affine_qmv_out_dim(weight) else {
        return false;
    };
    can_use_fast_affine_qmm2_u8_buffers(batch, in_dim, out_dim, weight.group_size(), weight.bits())
        && matches!(weight.shape(), [_, weight_in_dim] if *weight_in_dim == in_dim)
}

pub(super) fn can_use_fast_affine_qmm2_buffers(
    batch: usize,
    in_dim: usize,
    out_dim: usize,
    group_size: usize,
    bits: usize,
) -> bool {
    fast_affine_qmv_enabled(out_dim)
        && batch == 2
        && bits == FAST_QMV_BITS
        && group_size == FAST_QMV_GROUP_SIZE
        && in_dim % 512 == 0
        && out_dim % 8 == 0
}

/// Prédicat du qmv 6-bit gs64 pour le talker TTS : même contrat de buffers que
/// les qmv rapides u4/u8, mais dépaquetage 6-bit identique au kernel générique.
pub(super) fn can_use_fast_affine_qmv_u6_buffers(
    batch: usize,
    in_dim: usize,
    out_dim: usize,
    group_size: usize,
    bits: usize,
) -> bool {
    qmv_u6_enabled()
        && fast_affine_qmv_enabled(out_dim)
        && batch > 0
        && bits == FAST_QMV_U6_BITS
        && group_size == FAST_QMV_GROUP_SIZE
        && in_dim % FAST_QMV_GROUP_SIZE == 0
}

/// Prédicat du qmv 8-bit aligné : même géométrie que le qmv 4-bit rapide,
/// mais poids oQ/DWQ en u8.
pub(super) fn can_use_fast_affine_qmv_u8_buffers(
    batch: usize,
    in_dim: usize,
    out_dim: usize,
    group_size: usize,
    bits: usize,
) -> bool {
    fast_affine_qmv_enabled(out_dim)
        && batch > 0
        && bits == 8
        && matches!(group_size, FAST_QMV_GROUP_SIZE | 128)
        && in_dim % 512 == 0
        && out_dim % 8 == 0
}

/// Prédicat du qmv scalaire 8-bit gs64 : cible `shared_expert_gate` oQ
/// (out_dim=1), actif par défaut après A/B et désactivable par env.
pub(super) fn can_use_fast_affine_qmv_one_u8_buffers(
    batch: usize,
    in_dim: usize,
    out_dim: usize,
    group_size: usize,
    bits: usize,
) -> bool {
    qmv_one_u8_enabled()
        && batch > 0
        && out_dim == 1
        && bits == 8
        && group_size == FAST_QMV_GROUP_SIZE
        && in_dim % 512 == 0
}

/// Prédicat du qmm2 8-bit (duo light-batch sur poids DWQ) : mêmes gates que le
/// qmv u8 aligned, à batch == 2.
pub(super) fn can_use_fast_affine_qmm2_u8_buffers(
    batch: usize,
    in_dim: usize,
    out_dim: usize,
    group_size: usize,
    bits: usize,
) -> bool {
    batch == 2 && can_use_fast_affine_qmv_u8_buffers(batch, in_dim, out_dim, group_size, bits)
}

pub(super) fn can_use_fast_affine_argmax_qmv(
    batch: usize,
    in_dim: usize,
    weight: &AffineQuantizedTensor,
) -> bool {
    fast_argmax_qmv_enabled()
        && batch == 1
        && weight.bits() == FAST_QMV_BITS
        && weight.group_size() == FAST_QMV_GROUP_SIZE
        && in_dim % 512 == 0
        && matches!(weight.shape(), [_, weight_in_dim] if *weight_in_dim == in_dim)
}

pub(super) fn fast_argmax_qmv_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_FAST_ARGMAX_QMV", false))
}

pub(super) fn qmv_one_u8_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_QMV_ONE_U8", true))
}

pub(super) fn qmv_u6_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_QMV_U6", true))
}

pub(super) fn qmv_u8_tg128_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| profile_env_flag("RETI_RUST_QMV_U8_TG128", false))
}

pub(super) fn qmv_u8_tg256_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        env_flag(
            "RETI_RUST_QMV_U8_TG256",
            u8_kernel_preset_defaults().qmv_tg256,
        )
    })
}

pub(super) fn qmv_u8_dot4_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        env_flag(
            "RETI_RUST_QMV_U8_DOT4",
            u8_kernel_preset_defaults().qmv_dot4,
        )
    })
}

pub(super) fn linear_z_beta_gate_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_LINEAR_Z_BETA_GATE", true))
}

pub(super) fn linear_full_concat_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_LINEAR_FULL_CONCAT", true))
}

pub(super) fn linear_pair_barrier_coalesce_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| profile_env_flag("RETI_RUST_LINEAR_PAIR_BARRIER_COALESCE", false))
}

pub(super) fn fast_affine_qmv_out_dim(weight: &AffineQuantizedTensor) -> Option<usize> {
    match weight.shape() {
        [out_dim, _] => Some(*out_dim),
        _ => None,
    }
}

pub(super) fn fast_affine_qmv_enabled(out_dim: usize) -> bool {
    match qmv_fast_mode() {
        Some("0") | Some("false") | Some("off") => false,
        Some("1") | Some("true") | Some("all") => true,
        Some("q") => out_dim == 4096,
        Some("o") => out_dim == 2048,
        Some("kv") => out_dim == 512,
        Some("qkv") => out_dim == 4096 || out_dim == 512,
        Some("wide") => (2048..=4096).contains(&out_dim),
        Some("mid") => (512..=4096).contains(&out_dim),
        Some("small") => out_dim <= 4096,
        Some("large") => out_dim >= 65_536,
        _ => true,
    }
}

pub(super) fn fused_attn_epilogue_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_FUSED_ATTN_EPILOGUE", true))
}

pub(super) fn fused_rms_prologue_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_FUSED_RMS_PROLOGUE", true))
}

pub(super) fn dense_qmv_fast_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_DENSE_QMV_FAST", true))
}

pub(super) fn can_use_dense_qmv_fast(batch: usize, in_dim: usize, out_dim: usize) -> bool {
    dense_qmv_fast_enabled() && batch > 0 && in_dim % 512 == 0 && out_dim % 8 == 0
}

pub(super) fn topk8_fast_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_TOPK8_FAST", true))
}

pub(super) fn linear_delta_dk128_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_LINEAR_DELTA_DK128", true))
}

pub(super) fn linear_delta_tg_rows() -> u64 {
    static ROWS: OnceLock<u64> = OnceLock::new();
    *ROWS.get_or_init(|| linear_delta_tg_rows_override().unwrap_or(4))
}

pub(super) fn linear_delta_tg_rows_for_state(ssm_bf16: bool) -> u64 {
    linear_delta_tg_rows_override().unwrap_or_else(|| profile_linear_delta_tg_rows(ssm_bf16))
}

fn linear_delta_tg_rows_override() -> Option<u64> {
    std::env::var("RETI_RUST_LINEAR_DELTA_TG_ROWS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| matches!(*value, 1 | 2 | 4 | 8 | 16 | 32))
}

pub(super) fn linear_inv_delta_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| profile_env_flag("RETI_RUST_LINEAR_INV_DELTA", false))
}

pub(super) fn linear_ssm_bf16_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| profile_env_flag("RETI_RUST_LINEAR_SSM_BF16", false))
}

pub(super) fn linear_rms_dv128_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_LINEAR_RMS_DV128", true))
}

pub(super) fn linear_conv_k4_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_LINEAR_CONV_K4", true))
}

pub(super) fn linear_norm_dk128_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_LINEAR_NORM_DK128", true))
}

pub(super) fn linear_conv_norm_fused_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| profile_env_flag("RETI_RUST_LINEAR_CONV_NORM_FUSED", false))
}

pub(super) fn can_use_fast_gather_qmv(lhs_rows: usize, weight: &StackedAffineBuffers) -> bool {
    fast_gather_qmv_enabled(weight)
        && lhs_rows > 0
        && ((weight.bits == FAST_QMV_BITS
            && weight.group_size == FAST_QMV_GROUP_SIZE
            && weight.in_dim % weight.group_size == 0)
            || (weight.bits == 8
                && matches!(weight.group_size, FAST_QMV_GROUP_SIZE | 128)
                && weight.in_dim % 512 == 0
                && weight.out_dim % 8 == 0))
}

pub(super) fn valid_gather_lhs_rows(lhs_rows: usize, topk: usize) -> bool {
    lhs_rows > 0 && (lhs_rows == 1 || lhs_rows == topk || topk % lhs_rows == 0)
}

pub(super) fn can_use_fast_gather_pair_qmv(
    lhs_rows: usize,
    gate: &StackedAffineBuffers,
    up: &StackedAffineBuffers,
) -> bool {
    fused_gate_up_enabled()
        && (gate.bits != 8 || fused_gate_up_u8_enabled())
        && can_use_fast_gather_qmv(lhs_rows, gate)
        && can_use_fast_gather_qmv(lhs_rows, up)
        && gate.experts == up.experts
        && gate.out_dim == up.out_dim
        && gate.in_dim == up.in_dim
        && gate.packed_cols == up.packed_cols
        && gate.group_size == up.group_size
        && gate.bits == up.bits
        && gate.groups == up.groups
}

pub(super) fn fused_gate_up_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_FUSED_GATE_UP", true))
}

pub(super) fn fused_gate_up_u8_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        env_flag(
            "RETI_RUST_FUSED_GATE_UP_U8",
            u8_kernel_preset_defaults().fused_gate_up,
        )
    })
}

/// Active la fusion gate+up+swiglu du **shared-expert** (tranche 3).
pub(super) fn fused_shared_gate_up_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_FUSED_SHARED_GATE_UP", true))
}

pub(super) fn fused_shared_gate_up_u8_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        env_flag(
            "RETI_RUST_FUSED_SHARED_GATE_UP_U8",
            u8_kernel_preset_defaults().fused_shared_gate_up,
        )
    })
}

pub(super) fn fused_shared_gate_scalar_u8_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        env_flag(
            "RETI_RUST_FUSED_SHARED_GATE_SCALAR_U8",
            u8_kernel_preset_defaults().fused_shared_gate_scalar,
        )
    })
}

pub(super) fn fused_shared_gate_qmv_u8_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        env_flag(
            "RETI_RUST_FUSED_SHARED_GATE_QMV_U8",
            u8_kernel_preset_defaults().fused_shared_gate_qmv,
        )
    })
}

pub(super) fn fused_moe_down_weighted_u8_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        env_flag(
            "RETI_RUST_FUSED_MOE_DOWN_WEIGHTED_U8",
            u8_kernel_preset_defaults().fused_moe_down_weighted,
        )
    })
}

pub(super) fn full_qkv_split_rms_u8_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        env_flag(
            "RETI_RUST_FULL_QKV_SPLIT_RMS_U8",
            u8_kernel_preset_defaults().full_qkv_split_rms,
        )
    })
}

pub(super) fn full_o_proj_gated_u8_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        env_flag(
            "RETI_RUST_FULL_O_PROJ_GATED_U8",
            u8_kernel_preset_defaults().full_o_proj_gated,
        )
    })
}

pub(super) fn can_fuse_shared_gate_up_weights(gate: &Linear, up: &Linear) -> bool {
    if !fused_shared_gate_up_enabled() {
        return false;
    }
    match (gate.weight(), up.weight()) {
        (LinearWeight::AffineQuantized(gate), LinearWeight::AffineQuantized(up))
            if gate.bits() == 8 || up.bits() == 8 =>
        {
            fused_shared_gate_up_u8_enabled()
        }
        _ => true,
    }
}

pub(super) fn can_fuse_shared_gate_up_buffers(
    gate: &MetalLinearWeightBuffers,
    up: &MetalLinearWeightBuffers,
) -> bool {
    if !fused_shared_gate_up_enabled() {
        return false;
    }
    match (gate, up) {
        (
            MetalLinearWeightBuffers::AffineQuantized {
                bits: gate_bits, ..
            },
            MetalLinearWeightBuffers::AffineQuantized { bits: up_bits, .. },
        ) if *gate_bits == 8 || *up_bits == 8 => fused_shared_gate_up_u8_enabled(),
        _ => true,
    }
}

pub(super) fn moe_shared_route_overlap_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_MOE_SHARED_ROUTE_OVERLAP", true))
}

pub(super) fn topk_parallel_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_TOPK_PARALLEL", true))
}

pub(super) fn scratch_resource_options() -> MTLResourceOptions {
    if private_scratch_enabled() {
        MTLResourceOptions::StorageModePrivate
    } else {
        MTLResourceOptions::StorageModeShared
    }
}

pub(super) fn private_scratch_enabled() -> bool {
    static PRIVATE: OnceLock<bool> = OnceLock::new();
    *PRIVATE.get_or_init(|| env_flag("RETI_RUST_PRIVATE_SCRATCH", false))
}

pub(super) fn fast_gather_qmv_enabled(weight: &StackedAffineBuffers) -> bool {
    match gather_fast_mode() {
        Some("0") | Some("false") | Some("off") => false,
        Some("gateup") => weight.out_dim <= weight.in_dim,
        Some("down") => weight.out_dim > weight.in_dim,
        _ => true,
    }
}

pub(super) fn byte_len<T>(len: usize) -> Result<u64> {
    let bytes = byte_len_usize::<T>(len)?;
    u64::try_from(bytes).map_err(|_| InferError::Metal(format!("buffer trop grand: len={len}")))
}

pub(super) fn byte_len_nsuint<T>(len: usize, label: &'static str) -> Result<NSUInteger> {
    checked_nsuint(byte_len_usize::<T>(len)?, label)
}

pub(super) fn byte_len_usize<T>(len: usize) -> Result<usize> {
    len.checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| InferError::Metal(format!("buffer trop grand: len={len}")))
}

pub(super) fn byte_offset_f32(elements: usize, label: &'static str) -> Result<u64> {
    let bytes = elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| InferError::Metal(format!("{label} déborde")))?;
    u64::try_from(bytes).map_err(|_| InferError::Metal(format!("{label} hors plage u64")))
}

pub(super) fn checked_len(left: usize, right: usize, label: &'static str) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| InferError::Metal(format!("{label} trop grande")))
}

pub(super) fn checked_u32(value: usize, label: &'static str) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| InferError::Metal(format!("{label} hors plage uint32: {value}")))
}

pub(super) fn checked_nsuint(value: usize, label: &'static str) -> Result<NSUInteger> {
    NSUInteger::try_from(value)
        .map_err(|_| InferError::Metal(format!("{label} hors plage NSUInteger: {value}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn u8_kernel_preset_defaults_preserve_default_and_test_guards() {
        let defaults = u8_kernel_preset_defaults_for(KernelOptProfile::Default);
        assert_eq!(
            defaults,
            U8KernelPresetDefaults {
                fused_gate_up: false,
                fused_shared_gate_up: false,
                fused_shared_gate_scalar: false,
                fused_shared_gate_qmv: false,
                fused_moe_down_weighted: false,
                qmv_tg256: false,
                qmv_dot4: false,
                full_qkv_split_rms: cfg!(test),
                full_o_proj_gated: cfg!(test),
            }
        );
    }

    #[test]
    fn u8_kernel_preset_defaults_follow_exploration_profiles() {
        let qwen36 = u8_kernel_preset_defaults_for(KernelOptProfile::Qwen36Oq8);
        assert!(qwen36.fused_moe_down_weighted);
        assert!(!qwen36.fused_gate_up);
        assert!(!qwen36.full_qkv_split_rms);
        assert!(!qwen36.full_o_proj_gated);

        let explore = u8_kernel_preset_defaults_for(KernelOptProfile::ExploreFused);
        assert!(explore.fused_gate_up);
        assert!(explore.fused_shared_gate_up);
        assert!(explore.fused_shared_gate_scalar);
        assert!(explore.fused_shared_gate_qmv);
        assert!(explore.fused_moe_down_weighted);
        assert!(!explore.qmv_tg256);
        assert!(explore.qmv_dot4);
        assert!(explore.full_qkv_split_rms);
        assert!(explore.full_o_proj_gated);
    }
}
