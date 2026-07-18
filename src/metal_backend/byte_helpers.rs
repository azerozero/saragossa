//! Conversions de tailles et transferts de buffers Metal.

use super::profiling::DECODE_PROFILE_READ_NS;
use super::*;

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
