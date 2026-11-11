//! Transferts explicites CPU↔GPU pour les buffers sans pointeur CPU (K/V
//! sparse privé) : blit vers ou depuis un staging partagé réutilisé, par
//! plages bornées à sa taille. Le pic mémoire reste celui du staging, quel que
//! soit le volume transféré.

use super::shared_rows::SharedRows;
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
    /// Renvoie une erreur si Metal refuse le staging ou ne lui donne pas de
    /// pointeur CPU.
    pub(crate) fn new(device: &DeviceRef) -> Result<Self> {
        let buffer = super::sparse::checked_buffer(
            device,
            STAGING_BYTES,
            MTLResourceOptions::StorageModeShared,
        )?;
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
        let staged = unsafe { SharedRows::<u8>::borrow(&staging.buffer, 0, chunk.len())? };
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
        unsafe {
            SharedRows::<u8>::with_mut_slice(&staging.buffer, 0, chunk.len(), |staged| {
                staged.copy_from_slice(chunk);
            })?;
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "nécessite un device Metal"]
    fn shared_rows_staging_preserves_bytes_across_chunks() -> Result<()> {
        let device = Device::system_default()
            .ok_or_else(|| InferError::Metal("device Metal absent".to_string()))?;
        let queue = device.new_command_queue();
        let mut staging = Staging::new(&device)?;
        let data: Vec<u8> = (0..STAGING_BYTES as usize + 17)
            .map(|index| index as u8)
            .collect();
        let target = device.new_buffer(
            data.len() as u64 + 6,
            MTLResourceOptions::StorageModePrivate,
        );
        write_bytes(&queue, &mut staging, &target, 3, &data)?;
        let mut out = vec![0; data.len()];
        read_bytes(&queue, &mut staging, &target, 3, &mut out)?;
        assert_eq!(out, data);
        assert!(read_bytes(&queue, &mut staging, &target, 7, &mut out).is_err());
        assert!(write_bytes(&queue, &mut staging, &target, u64::MAX, &[1]).is_err());
        write_bytes(&queue, &mut staging, &target, target.length(), &[])?;
        read_bytes(&queue, &mut staging, &target, target.length(), &mut [])?;
        Ok(())
    }
}
