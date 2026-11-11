//! Oracles de copie sur de petits buffers partagés, sans calcul GPU.

use crate::{MetalExecutor, Result};

#[expect(
    unsafe_code,
    reason = "lecture des buffers partagés immuables des fixtures"
)]
pub(crate) fn buffer_bytes(buffer: &metal::Buffer) -> Vec<u8> {
    assert_eq!(buffer.storage_mode(), metal::MTLStorageMode::Shared);
    // SAFETY: buffer partagé vivant, initialisé, sans commande GPU ni écrivain ;
    // la tranche u8 est alignée et bornée par la longueur allouée par Metal.
    unsafe { std::slice::from_raw_parts(buffer.contents().cast(), buffer.length() as usize) }
        .to_vec()
}

#[test]
fn direct_file_buffers_match_copies_and_reject_short_reads() -> Result<()> {
    let metal = MetalExecutor::new()?;
    let codes = [0x7654_3210_u32, 0xfedc_ba98, 0, u32::MAX];
    let bytes: Vec<_> = codes.iter().flat_map(|v| v.to_le_bytes()).collect();
    let file = tempfile::NamedTempFile::new().expect("invariant: fichier temporaire");
    let mut payload = vec![0xaa; 13];
    payload.extend_from_slice(&bytes);
    std::fs::write(file.path(), payload).expect("invariant: fixture écrite");
    let direct = metal.weight_buffer_from_u32_file(file.path(), 13, bytes.len())?;
    let copied = metal.weight_buffer_from_u32(&codes)?;
    assert_eq!(buffer_bytes(&direct), buffer_bytes(&copied));
    let direct = metal.weight_buffer_from_bf16_file(file.path(), 13, bytes.len())?;
    let copied = metal.weight_buffer_from_bf16_bytes(&bytes)?;
    assert_eq!(buffer_bytes(&direct), buffer_bytes(&copied));
    assert!(metal
        .weight_buffer_from_u32_file(file.path(), 13, 3)
        .is_err());
    assert!(metal
        .weight_buffer_from_u32_file(file.path(), 13, 0)
        .is_err());
    assert!(metal
        .weight_buffer_from_bf16_file(file.path(), 13, 3)
        .is_err());
    assert!(metal
        .weight_buffer_from_u32_file(file.path(), 13, 20)
        .is_err());
    assert!(metal
        .weight_buffer_from_u32_file(file.path(), u64::MAX, 4)
        .is_err());
    Ok(())
}
