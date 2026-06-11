//! Helpers CPU pour les buffers résidents partagés.

use super::*;
use crate::decoder::flags::env_flag;

/// Écrit `data` dans le buffer f32 `tensor` à partir de l'élément `offset`.
///
/// Utilisé pour l'append KV résident (réserve R3 : en decode, écriture CPU dans
/// un buffer `StorageModeShared` AVANT `commit` → visible par le GPU, pas de
/// hazard). Borné par `tensor.len()`.
///
/// # Errors
///
/// Renvoie une erreur si `tensor` n'est pas f32 ou si `offset + data.len()`
/// dépasse la longueur logique du buffer.
#[allow(
    unsafe_code,
    reason = "écriture d'un MTLBuffer StorageModeShared à un offset avant commit"
)]
pub(super) fn write_f32_at(tensor: &GpuTensor, offset: usize, data: &[f32]) -> Result<()> {
    if tensor.element() != GpuElement::F32 {
        return Err(InferError::Metal(
            "write_f32_at sur un buffer non-f32".to_string(),
        ));
    }
    let end = offset
        .checked_add(data.len())
        .ok_or_else(|| InferError::Metal("write_f32_at: offset déborde".to_string()))?;
    if end > tensor.len() {
        return Err(InferError::Metal(format!(
            "write_f32_at hors capacité: offset({offset})+len({}) > {}",
            data.len(),
            tensor.len()
        )));
    }
    if data.is_empty() {
        return Ok(());
    }
    let ptr = tensor.buffer().contents().cast::<f32>();
    if ptr.is_null() {
        return Err(InferError::Metal(
            "MTLBuffer StorageModeShared sans pointeur CPU".to_string(),
        ));
    }
    // SAFETY: buffer en StorageModeShared, écriture de `data.len()` f32 à partir
    // de l'élément `offset` ; `offset + data.len() <= tensor.len()` vérifié →
    // `ptr.add(offset)` est dans les bornes, copie sans chevauchement, avant tout
    // commit du command buffer qui lira ce buffer.
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr(), ptr.add(offset), data.len());
    }
    Ok(())
}

/// Écrit `data` dans le buffer u32 `tensor` à partir de l'élément `offset`.
///
/// # Errors
///
/// Renvoie une erreur si `tensor` n'est pas u32 ou si `offset + data.len()`
/// dépasse la longueur logique du buffer.
#[allow(
    unsafe_code,
    reason = "écriture d'un MTLBuffer StorageModeShared à un offset avant commit"
)]
pub(super) fn write_u32_at(tensor: &GpuTensor, offset: usize, data: &[u32]) -> Result<()> {
    if tensor.element() != GpuElement::U32 {
        return Err(InferError::Metal(
            "write_u32_at sur un buffer non-u32".to_string(),
        ));
    }
    let end = offset
        .checked_add(data.len())
        .ok_or_else(|| InferError::Metal("write_u32_at: offset déborde".to_string()))?;
    if end > tensor.len() {
        return Err(InferError::Metal(format!(
            "write_u32_at hors capacité: offset({offset})+len({}) > {}",
            data.len(),
            tensor.len()
        )));
    }
    if data.is_empty() {
        return Ok(());
    }
    let ptr = tensor.buffer().contents().cast::<u32>();
    if ptr.is_null() {
        return Err(InferError::Metal(
            "MTLBuffer StorageModeShared sans pointeur CPU".to_string(),
        ));
    }
    // SAFETY: buffer en StorageModeShared, écriture de `data.len()` u32 à partir
    // de l'élément `offset` ; `offset + data.len() <= tensor.len()` vérifié.
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr(), ptr.add(offset), data.len());
    }
    Ok(())
}

/// KV-cache full-attn **résident GPU** d'UNE couche (10 couches sur Qwen3.6).
///
/// Remplace, derrière le flag du decode résident, le `Vec<f32>` CPU append-only
/// (`decoder.rs` `LayerKvCache.keys/values`) et l'attention CPU
/// (`cached_attention_one`). Les buffers `keys`/`values` sont **persistants**
/// (alloués une fois via [`DecodeResidentState::persistent`], capacité bornée par
/// `prefill_len + max_new_tokens`) et restent GPU-résidents entre tokens.
///
/// **Non clonable** (réserve Codex D) : un état résident GPU est lié à une
/// session ; le `Clone` du cache englobant doit le remettre à `None` (drop des
pub(super) fn byte_offset_f32(elements: usize, label: &'static str) -> Result<u64> {
    let bytes = elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| InferError::Dimension(format!("{label} déborde")))?;
    u64::try_from(bytes).map_err(|_| InferError::Dimension(format!("{label} hors plage u64")))
}

pub(super) fn flash_sdpa_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_FLASH_SDPA", true))
}
