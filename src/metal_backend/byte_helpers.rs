//! Conversions de tailles et transferts de buffers Metal.

use super::profiling::DECODE_PROFILE_READ_NS;
use super::shared_rows::{SharedElement, SharedRows};
use super::*;

#[allow(
    unsafe_code,
    reason = "lecture d'un MTLBuffer partagé après wait_until_completed"
)]
fn read_pod_buffer<T: SharedElement + Default>(
    buffer: &BufferRef,
    offset: usize,
    len: usize,
) -> Result<Vec<T>> {
    let read_started = decode_profile_enabled().then(std::time::Instant::now);
    // SAFETY: les écritures GPU sont terminées et la plage reste figée pendant
    // la copie ; SharedRows vérifie taille et alignement des éléments Pod.
    let source = unsafe { SharedRows::<T>::borrow(buffer, offset, len)? };
    let mut out = vec![T::default(); len];
    out.copy_from_slice(source);
    #[cfg(feature = "devtools")]
    crate::runtime_flags::pairwise::record_io(false, std::mem::size_of_val(out.as_slice()));
    if let Some(started) = read_started {
        DECODE_PROFILE_READ_NS.fetch_add(
            started.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
    }
    Ok(out)
}

pub(crate) fn read_f32_buffer(buffer: &BufferRef, len: usize) -> Result<Vec<f32>> {
    read_pod_buffer(buffer, 0, len)
}

/// Lit `len` f32 à partir de l'élément `offset`.
pub(crate) fn read_f32_buffer_at(
    buffer: &BufferRef,
    offset: usize,
    len: usize,
) -> Result<Vec<f32>> {
    read_pod_buffer(buffer, offset, len)
}

pub(crate) fn read_u32_buffer(buffer: &BufferRef, len: usize) -> Result<Vec<u32>> {
    read_pod_buffer(buffer, 0, len)
}

pub(crate) fn read_u16_buffer(buffer: &BufferRef, len: usize) -> Result<Vec<u16>> {
    read_pod_buffer(buffer, 0, len)
}

#[allow(
    unsafe_code,
    reason = "écriture d'un MTLBuffer partagé avant commit du command buffer"
)]
fn write_pod_buffer<T: SharedElement>(buffer: &BufferRef, offset: usize, data: &[T]) -> Result<()> {
    // SAFETY: plage partagée initialisée, sans accès concurrent avant commit ;
    // `data` est distinct. SharedRows vérifie taille et alignement des Pod.
    unsafe {
        SharedRows::<T>::with_mut_slice(buffer, offset, data.len(), |out| {
            out.copy_from_slice(data);
        })?;
    }
    #[cfg(feature = "devtools")]
    crate::runtime_flags::pairwise::record_io(true, std::mem::size_of_val(data));
    Ok(())
}

pub(crate) fn write_f32_buffer(buffer: &BufferRef, data: &[f32]) -> Result<()> {
    write_pod_buffer(buffer, 0, data)
}

/// Écrit `data` à partir de l'élément `offset`.
pub(crate) fn write_f32_buffer_at(buffer: &BufferRef, offset: usize, data: &[f32]) -> Result<()> {
    write_pod_buffer(buffer, offset, data)
}

/// Écrit des octets bruts (lignes E4M3 du PLE Flash-Next).
pub(crate) fn write_u8_buffer(buffer: &BufferRef, data: &[u8]) -> Result<()> {
    write_pod_buffer(buffer, 0, data)
}

/// Écrit des bits bf16 à partir de l'élément `offset`.
pub(crate) fn write_u16_buffer_at(buffer: &BufferRef, offset: usize, data: &[u16]) -> Result<()> {
    write_pod_buffer(buffer, offset, data)
}

/// Relit `len` bits bf16 à partir de l'élément `offset`.
pub(crate) fn read_u16_buffer_at(
    buffer: &BufferRef,
    offset: usize,
    len: usize,
) -> Result<Vec<u16>> {
    read_pod_buffer(buffer, offset, len)
}

pub(super) fn write_u32_buffer(buffer: &BufferRef, data: &[u32]) -> Result<()> {
    write_pod_buffer(buffer, 0, data)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "nécessite un device Metal"]
    fn shared_rows_transfers_preserve_bits_and_bounds() -> Result<()> {
        let device = Device::system_default()
            .ok_or_else(|| InferError::Metal("device Metal absent".to_string()))?;
        let buffer = device.new_buffer(32, MTLResourceOptions::StorageModeShared);
        let bits = [0, 0x8000_0000, 0x7fc0_0123, 0x7f80_0000, 0x3f80_0000];
        let values = bits.map(f32::from_bits);
        write_f32_buffer_at(&buffer, 1, &values)?;
        let read = read_f32_buffer_at(&buffer, 1, values.len())?;
        assert_eq!(read.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), bits);
        assert_eq!(read_u32_buffer(&buffer, 6)?[1..], bits);
        let bf16 = [0, 0x8000, 0x7fc1, 0xffff];
        write_u16_buffer_at(&buffer, 2, &bf16)?;
        assert_eq!(read_u16_buffer_at(&buffer, 2, bf16.len())?, bf16);
        write_u8_buffer(&buffer, &[0, 127, 255])?;
        assert_eq!(read_pod_buffer::<u8>(&buffer, 0, 3)?, [0, 127, 255]);
        assert!(read_f32_buffer_at(&buffer, 8, 1).is_err());
        assert!(write_f32_buffer_at(&buffer, 8, &[1.0]).is_err());
        assert!(read_f32_buffer_at(&buffer, usize::MAX, 1).is_err());
        assert!(read_f32_buffer_at(&buffer, 8, 0)?.is_empty());
        write_f32_buffer_at(&buffer, 8, &[])?;
        Ok(())
    }
}
