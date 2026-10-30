//! Transferts explicites CPU↔GPU pour les buffers sans pointeur CPU (K/V
//! sparse privé) : blit vers ou depuis un staging partagé réutilisé, par
//! plages bornées à sa taille. Le pic mémoire reste celui du staging, quel que
//! soit le volume transféré.

use super::*;
use metal::{CommandQueueRef, DeviceRef};

/// Taille du staging : 4 Mio, assez pour amortir un commit par plage sans
/// retenir de mémoire notable.
const STAGING_BYTES: u64 = 4 << 20;

/// Buffer partagé de transit. `&mut` à chaque transfert : un seul transfert à
/// la fois l'utilise.
pub(crate) struct Staging {
    buffer: Buffer,
}

impl Staging {
    /// # Errors
    ///
    /// Renvoie une erreur si Metal ne donne pas de pointeur CPU au staging.
    pub(crate) fn new(device: &DeviceRef) -> Result<Self> {
        let buffer = device.new_buffer(STAGING_BYTES, MTLResourceOptions::StorageModeShared);
        if buffer.contents().is_null() {
            return Err(InferError::Metal(
                "staging de transfert sans pointeur CPU".to_string(),
            ));
        }
        Ok(Self { buffer })
    }
}

fn check_range(buffer: &BufferRef, offset: u64, len: usize) -> Result<()> {
    let end = u64::try_from(len)
        .ok()
        .and_then(|len| offset.checked_add(len));
    if end.is_none_or(|end| end > buffer.length()) {
        return Err(InferError::Dimension(format!(
            "transfert {offset}+{len} octets hors du buffer de {} octets",
            buffer.length()
        )));
    }
    Ok(())
}

/// Blit de `len` octets puis attente : `source` → `target`.
fn blit(
    queue: &CommandQueueRef,
    (source, source_offset): (&BufferRef, u64),
    (target, target_offset): (&BufferRef, u64),
    len: u64,
) -> Result<()> {
    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_blit_command_encoder();
    encoder.copy_from_buffer(source, source_offset, target, target_offset, len);
    encoder.end_encoding();
    commit_and_wait(command_buffer)
}

/// Lit `out.len()` octets de `source` depuis `offset`, après toutes les
/// commandes déjà soumises sur `queue`.
///
/// # Errors
///
/// Renvoie une erreur si la plage déborde ou si un blit échoue.
#[expect(unsafe_code, reason = "lecture CPU du staging partagé")]
pub(crate) fn read_bytes(
    queue: &CommandQueueRef,
    staging: &mut Staging,
    source: &BufferRef,
    offset: u64,
    out: &mut [u8],
) -> Result<()> {
    check_range(source, offset, out.len())?;
    let mut done = 0usize;
    for chunk in out.chunks_mut(STAGING_BYTES as usize) {
        blit(
            queue,
            (source, offset + done as u64),
            (&staging.buffer, 0),
            chunk.len() as u64,
        )?;
        // SAFETY: le staging partagé a un pointeur CPU (vérifié à sa création)
        // et au moins `chunk.len()` octets ; le blit qui l'écrivait est
        // terminé et `&mut Staging` exclut tout autre usage concurrent.
        let staged = unsafe {
            std::slice::from_raw_parts(staging.buffer.contents().cast::<u8>(), chunk.len())
        };
        chunk.copy_from_slice(staged);
        done += chunk.len();
    }
    Ok(())
}

/// Écrit `data` dans `target` depuis `offset` ; terminé au retour.
///
/// # Errors
///
/// Renvoie une erreur si la plage déborde ou si un blit échoue.
#[expect(unsafe_code, reason = "écriture CPU du staging partagé")]
pub(crate) fn write_bytes(
    queue: &CommandQueueRef,
    staging: &mut Staging,
    target: &BufferRef,
    offset: u64,
    data: &[u8],
) -> Result<()> {
    check_range(target, offset, data.len())?;
    let mut done = 0usize;
    for chunk in data.chunks(STAGING_BYTES as usize) {
        // SAFETY: staging partagé d'au moins `chunk.len()` octets avec
        // pointeur CPU ; aucun blit ne le lit (le précédent est attendu) et
        // `&mut Staging` exclut tout autre usage.
        let staged = unsafe {
            std::slice::from_raw_parts_mut(staging.buffer.contents().cast::<u8>(), chunk.len())
        };
        staged.copy_from_slice(chunk);
        blit(
            queue,
            (&staging.buffer, 0),
            (target, offset + done as u64),
            chunk.len() as u64,
        )?;
        done += chunk.len();
    }
    Ok(())
}
