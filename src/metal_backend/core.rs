//! Helpers communs du backend Metal.

use crate::decoder::flags::{env_flag, gather_fast_mode, qmv_fast_mode};

use super::*;

// ---------------------------------------------------------------------------
// Instrumentation decode (RETI_RUST_DECODE_PROFILE) — phase 1a
//
// Split par token : `encode_us` (CPU, dérivé = total − wait − read) / `wait_us`
// (CPU bloqué sur le GPU dans `wait_until_completed`) / `read_us` (readback
// GPU→CPU) + nombre de command buffers/token. Atomics globaux (le decode est
// mono-thread) ; surcoût NUL quand le flag est absent (seul un test de bool
// caché). Guide la priorisation 1b/1c (cf. /tmp/rust_infer_plan.md).
// ---------------------------------------------------------------------------
static DECODE_PROFILE_CB: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static DECODE_PROFILE_WAIT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static DECODE_PROFILE_READ_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static DECODE_PROFILE_DISPATCHES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

pub(super) fn decode_profile_enabled() -> bool {
    crate::decoder::flags::decode_profile_enabled()
}

/// `(command_buffers, wait_ns, read_ns)` cumulés — bornés par l'appelant
/// (decode loop) pour un split par token. Toujours disponible (zéros si le flag
/// est absent, car `commit_and_wait`/`read_*_buffer` ne cumulent que si activé).
pub(crate) fn decode_profile_snapshot() -> (u64, u64, u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (
        DECODE_PROFILE_CB.load(Relaxed),
        DECODE_PROFILE_WAIT_NS.load(Relaxed),
        DECODE_PROFILE_READ_NS.load(Relaxed),
        DECODE_PROFILE_DISPATCHES.load(Relaxed),
    )
}

pub(crate) fn profile_dispatch() {
    use std::sync::atomic::Ordering::Relaxed;
    if decode_profile_enabled() {
        DECODE_PROFILE_DISPATCHES.fetch_add(1, Relaxed);
    }
}

thread_local! {
    static DISPATCH_BARRIER_SCOPE: Cell<bool> = const { Cell::new(false) };
}

pub(crate) fn resident_concurrent_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_RESIDENT_CONCURRENT", true))
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

pub(crate) fn post_dispatch_barrier(encoder: &ComputeCommandEncoderRef) {
    DISPATCH_BARRIER_SCOPE.with(|slot| {
        if slot.get() {
            memory_barrier_buffers(encoder);
        }
    });
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

pub(super) fn write_f32_buffer(buffer: &BufferRef, data: &[f32]) -> Result<()> {
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

pub(super) fn can_use_fast_gather_qmv(lhs_rows: usize, weight: &StackedAffineBuffers) -> bool {
    fast_gather_qmv_enabled(weight)
        && lhs_rows > 0
        && weight.bits == FAST_QMV_BITS
        && weight.group_size == FAST_QMV_GROUP_SIZE
        && weight.in_dim % weight.group_size == 0
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

/// Active la fusion gate+up+swiglu du **shared-expert** (tranche 3).
pub(super) fn fused_shared_gate_up_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_FUSED_SHARED_GATE_UP", true))
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
        Some("gateup") => weight.in_dim % 512 == 0,
        Some("down") => weight.in_dim % 512 != 0,
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
