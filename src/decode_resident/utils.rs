//! Helpers CPU pour les buffers résidents partagés.

use super::*;
use crate::runtime_flags::{env_flag, env_flag_value};

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

fn f32_to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let lsb = (bits >> 16) & 1;
    let rounded = bits.wrapping_add(0x7fff + lsb);
    (rounded >> 16) as u16
}

/// Écrit `data` dans le buffer bf16 `tensor` à partir de l'élément `offset`.
///
/// L'entrée reste f32 (K/V ropés du chemin prefill ou decode), mais le stockage
/// résident est arrondi bf16 round-to-nearest-even comme les uploads bf16 Metal.
///
/// # Errors
///
/// Renvoie une erreur si `tensor` n'est pas bf16 ou si `offset + data.len()`
/// dépasse la longueur logique du buffer.
#[allow(
    unsafe_code,
    reason = "écriture d'un MTLBuffer StorageModeShared à un offset avant commit"
)]
pub(super) fn write_f32_as_bf16_at(tensor: &GpuTensor, offset: usize, data: &[f32]) -> Result<()> {
    if tensor.element() != GpuElement::Bf16 {
        return Err(InferError::Metal(
            "write_f32_as_bf16_at sur un buffer non-bf16".to_string(),
        ));
    }
    let end = offset
        .checked_add(data.len())
        .ok_or_else(|| InferError::Metal("write_f32_as_bf16_at: offset déborde".to_string()))?;
    if end > tensor.len() {
        return Err(InferError::Metal(format!(
            "write_f32_as_bf16_at hors capacité: offset({offset})+len({}) > {}",
            data.len(),
            tensor.len()
        )));
    }
    if data.is_empty() {
        return Ok(());
    }
    let ptr = tensor.buffer().contents().cast::<u16>();
    if ptr.is_null() {
        return Err(InferError::Metal(
            "MTLBuffer StorageModeShared sans pointeur CPU".to_string(),
        ));
    }
    // SAFETY: buffer en StorageModeShared, écriture de `data.len()` u16 à partir
    // de l'élément `offset` ; les bornes sont vérifiées plus haut et l'écriture
    // précède le command buffer qui lira le KV résident.
    unsafe {
        for (index, value) in data.iter().copied().enumerate() {
            ptr.add(offset + index).write(f32_to_bf16_bits(value));
        }
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

pub(super) fn byte_offset(
    elements: usize,
    element: GpuElement,
    label: &'static str,
) -> Result<u64> {
    let bytes = elements
        .checked_mul(element.byte_size())
        .ok_or_else(|| InferError::Dimension(format!("{label} déborde")))?;
    u64::try_from(bytes).map_err(|_| InferError::Dimension(format!("{label} hors plage u64")))
}

pub(super) fn flash_sdpa_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_FLASH_SDPA", true))
}

/// Résout le dtype du KV résident full-attn selon la politique produit C1B.
///
/// `RETI_RUST_KV_BF16` explicite (0/1) gagne toujours ; sinon défaut = **bf16 si
/// la génération échantillonne** (`sampled`, temperature > 0), **f32 en greedy**.
///
/// Motivation : bf16 divise par deux la bande passante KV (decode long-contexte
/// +12 % @32k → 89,8 tok/s) pour une
// NOTE : la comparaison « dépasse mlx 88,1 » a été retirée le 2026-07-22. La marge
// (+1,9 %) est très inférieure au bruit de mesure de la machine (7-16 % d'étendue
// sur un run unique), et la référence mlx elle-même variait de 12 % entre relevés.
// Le +12 % de bf16 reste étayé (dose-réponse monotone + réplication indépendante) ;
// c'est le classement face à mlx qui ne l'était pas.
/// perturbation **near-tie** (0,03 logit médian ≪ marge 4,5-6,0 du modèle ;
/// byte-identique à 30k) — noyée par le bruit du sampling T > 0. Le greedy reste
/// f32 pour garder les **oracles md5 e2e byte-identiques** (non re-baselinés).
pub(super) fn kv_bf16_for(sampled: bool) -> bool {
    resolve_kv_bf16(kv_bf16_env_override(), sampled)
}

/// Combine l'override explicite et l'indice `sampled` : l'override gagne toujours,
/// sinon le défaut suit la température (bf16 échantillonné, f32 greedy).
fn resolve_kv_bf16(env_override: Option<bool>, sampled: bool) -> bool {
    env_override.unwrap_or(sampled)
}

/// Override explicite de `RETI_RUST_KV_BF16` (lu une fois), cf. [`kv_bf16_for`].
fn kv_bf16_env_override() -> Option<bool> {
    static OVERRIDE: OnceLock<Option<bool>> = OnceLock::new();
    *OVERRIDE.get_or_init(|| {
        std::env::var("RETI_RUST_KV_BF16")
            .ok()
            .as_deref()
            .and_then(parse_kv_bf16_override)
    })
}

/// Parse une valeur d'env en override tri-état : `Some(true)` pour 1/true/on,
/// `Some(false)` pour 0/false/off (insensible à la casse, espaces ignorés),
/// `None` sinon (→ défaut piloté par la température).
fn parse_kv_bf16_override(value: &str) -> Option<bool> {
    env_flag_value(value)
}

/// Arrondit `value` à la précision bf16 tout en restant représenté en f32.
///
/// Diagnostic C1B (hors prod) : permet de simuler « K seul » ou « V seul » en
/// bf16 sans buffer ni kernel bf16 — on écrit la valeur bf16-arrondie dans le
/// buffer f32 existant (le kernel f32 la relit à l'identique). Numériquement
/// égal au stockage bf16 réel (le kernel upcaste `bfloat`→`float`).
pub(super) fn bf16_round_f32(value: f32) -> f32 {
    f32::from_bits(u32::from(f32_to_bf16_bits(value)) << 16)
}

/// Diagnostic C1B (hors prod) : arrondit **seulement K** au seed prefill (V reste
/// f32 exact), pour isoler l'effet de la précision des clés (→ scores) sur la
/// divergence. `RETI_RUST_KV_BF16_SIM_KONLY=1`, défaut OFF. À utiliser avec
/// `RETI_RUST_KV_BF16=0` (buffers f32). N'arrondit que le seed, pas les lignes
/// appendées en decode (négligeables devant le prompt).
pub(super) fn kv_bf16_sim_konly() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_KV_BF16_SIM_KONLY", false))
}

/// Diagnostic C1B (hors prod) : arrondit **seulement V** au seed prefill (K reste
/// f32 exact), pour isoler l'effet de la précision des valeurs (→ sortie) sur la
/// divergence. `RETI_RUST_KV_BF16_SIM_VONLY=1`, défaut OFF.
pub(super) fn kv_bf16_sim_vonly() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_KV_BF16_SIM_VONLY", false))
}

/// Active la SDPA decode **2-passes split-K** (dédup GQA + tuiles L1) au-delà de
/// `sdpa_2pass_min_len()` rows de KV. Défaut ON ; kill-switch `RETI_RUST_SDPA_2PASS=0`.
pub(super) fn sdpa_2pass_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_SDPA_2PASS", true))
}

/// Longueur de KV minimale pour basculer en 2-passes.
///
/// Défaut 2048 (décision tuning 2026-07-03, D-30B) : sous ce seuil le
/// single-pass flash est plus rapide, avec moins de dispatches et pas de
/// scratch partials. Override : `RETI_RUST_SDPA_2PASS_MIN_LEN`.
pub(super) fn sdpa_2pass_min_len() -> usize {
    static LEN: OnceLock<usize> = OnceLock::new();
    *LEN.get_or_init(|| {
        std::env::var("RETI_RUST_SDPA_2PASS_MIN_LEN")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(2048)
    })
}

/// Nombre de blocs split-K pour la SDPA 2-passes : multiple de 32 (requis par la
/// réduction passe-2 `blocks/32`), ≈ `len/128`, borné `[32, 1024]`.
pub(super) fn sdpa_2pass_blocks(len: usize) -> usize {
    let raw = (len / 128).max(1);
    let rounded = raw.div_ceil(32) * 32;
    rounded.clamp(32, 1024)
}

#[cfg(test)]
mod kv_bf16_resolution_tests {
    use super::{parse_kv_bf16_override, resolve_kv_bf16};

    #[test]
    fn parse_reconnait_les_formes_explicites() {
        for on in ["1", "true", "TRUE", "on", "On", " 1 "] {
            assert_eq!(parse_kv_bf16_override(on), Some(true), "{on:?} -> true");
        }
        for off in ["0", "false", "FALSE", "off", "Off", " 0 "] {
            assert_eq!(parse_kv_bf16_override(off), Some(false), "{off:?} -> false");
        }
        for none in ["", "2", "bf16", "yes", "no"] {
            assert_eq!(parse_kv_bf16_override(none), None, "{none:?} -> None");
        }
    }

    #[test]
    fn precedence_env_explicite_puis_temperature() {
        // Override explicite gagne toujours, quelle que soit la température.
        assert!(resolve_kv_bf16(Some(true), false), "env=1 + greedy -> bf16");
        assert!(
            !resolve_kv_bf16(Some(false), true),
            "env=0 + sampled -> f32"
        );
        // Défaut piloté par la température quand l'env est absent.
        assert!(resolve_kv_bf16(None, true), "défaut sampled -> bf16");
        assert!(!resolve_kv_bf16(None, false), "défaut greedy -> f32");
    }
}
