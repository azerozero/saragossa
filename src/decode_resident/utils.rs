//! Helpers CPU pour les buffers résidents partagés.

use super::*;
use crate::metal_backend::SharedRows;
use crate::runtime_flags::{env_flag, env_flag_value};
use rayon::prelude::*;

/// Taille des blocs convertis en parallèle vers un buffer bf16 partagé.
const PARALLEL_COPY_CHUNK: usize = 1 << 16;

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
    // SAFETY: plage partagée initialisée, exclusive avant commit et distincte
    // de `data` ; SharedRows vérifie taille et alignement des éléments Pod.
    unsafe {
        SharedRows::<f32>::with_mut_slice(tensor.buffer(), offset, data.len(), |out| {
            out.copy_from_slice(data);
        })?;
    }
    #[cfg(feature = "devtools")]
    crate::runtime_flags::pairwise::record_io(true, std::mem::size_of_val(data));
    Ok(())
}

/// Relit `count` éléments du buffer `tensor` à partir de l'élément `offset`,
/// en f32 (bf16 élargi sans arrondi), après la fin du command buffer qui l'a
/// écrit. Sert à rapatrier vers le cache CPU les lignes K/V ajoutées par un
/// kernel en lignes.
///
/// # Errors
///
/// Renvoie une erreur si `tensor` est u32 ou si `offset + count` dépasse sa
/// longueur logique.
#[allow(
    unsafe_code,
    reason = "lecture d'un MTLBuffer StorageModeShared à un offset après wait_until_completed"
)]
pub(super) fn read_f32_at(tensor: &GpuTensor, offset: usize, count: usize) -> Result<Vec<f32>> {
    let end = offset
        .checked_add(count)
        .ok_or_else(|| InferError::Metal("read_f32_at: offset déborde".to_string()))?;
    if end > tensor.len() {
        return Err(InferError::Metal(format!(
            "read_f32_at hors capacité: offset({offset})+count({count}) > {}",
            tensor.len()
        )));
    }
    if count == 0 {
        return Ok(Vec::new());
    }
    let mut out = vec![0.0_f32; count];
    match tensor.element() {
        GpuElement::F32 => {
            // SAFETY: écritures GPU terminées, plage initialisée et figée
            // pendant la copie ; SharedRows vérifie bornes et alignement Pod.
            let source = unsafe { SharedRows::<f32>::borrow(tensor.buffer(), offset, count)? };
            out.copy_from_slice(source);
        }
        GpuElement::Bf16 => {
            // SAFETY: mêmes garanties de durée de vie que le cas f32, avec
            // des bits bf16 Pod dont SharedRows vérifie taille et alignement.
            let source = unsafe { SharedRows::<u16>::borrow(tensor.buffer(), offset, count)? };
            for (value, &bits) in out.iter_mut().zip(source) {
                *value = f32::from_bits(u32::from(bits) << 16);
            }
        }
        GpuElement::U32 | GpuElement::U8 => {
            return Err(InferError::Metal(format!(
                "read_f32_at sur un buffer {:?} invalide",
                tensor.element()
            )));
        }
    }
    #[cfg(feature = "devtools")]
    crate::runtime_flags::pairwise::record_io(false, std::mem::size_of_val(out.as_slice()));
    Ok(out)
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
    // SAFETY: plage bf16 partagée initialisée, exclusive avant commit ;
    // SharedRows vérifie taille et alignement, le callback finit avant le GPU.
    unsafe {
        SharedRows::<u16>::with_mut_slice(tensor.buffer(), offset, data.len(), |out| {
            write_bf16_bits(out, data);
        })?;
    }
    #[cfg(feature = "devtools")]
    crate::runtime_flags::pairwise::record_io(true, data.len() * 2);
    Ok(())
}

// Conversion indépendante par élément : les blocs parallèles conservent le
// débit du semis KV sans allouer de copie intermédiaire.
fn write_bf16_bits(out: &mut [u16], data: &[f32]) {
    out.par_chunks_mut(PARALLEL_COPY_CHUNK)
        .zip(data.par_chunks(PARALLEL_COPY_CHUNK))
        .for_each(|(out, data)| {
            for (out, value) in out.iter_mut().zip(data) {
                *out = f32_to_bf16_bits(*value);
            }
        });
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
    // SAFETY: plage partagée initialisée, exclusive avant commit et distincte
    // de `data` ; SharedRows vérifie taille et alignement des éléments Pod.
    unsafe {
        SharedRows::<u32>::with_mut_slice(tensor.buffer(), offset, data.len(), |out| {
            out.copy_from_slice(data);
        })?;
    }
    #[cfg(feature = "devtools")]
    crate::runtime_flags::pairwise::record_io(true, std::mem::size_of_val(data));
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
    *ENABLED.get_or_init(|| env_flag("SARAGOSSA_RUST_FLASH_SDPA", true))
}

/// Résout le dtype du KV résident full-attn selon la politique produit C1B.
///
/// `SARAGOSSA_RUST_KV_BF16` explicite (0/1) gagne toujours ; sinon `default_bf16`,
/// choisi par l'appelant (`resident_kv_bf16` : génération échantillonnée ou prompt
/// long ; `false` pour les arènes dont les lignes sont relues vers le CPU).
///
/// Motivation : bf16 divise par deux la bande passante KV (decode long-contexte
/// +12 % @32k → 89,8 tok/s) pour une
// NOTE : la comparaison « dépasse mlx 88,1 » a été retirée le 2026-07-22. La marge
// (+1,9 %) est très inférieure au bruit de mesure de la machine (7-16 % d'étendue
// sur un run unique), et la référence mlx elle-même variait de 12 % entre relevés.
// Le +12 % de bf16 reste étayé (dose-réponse monotone + réplication indépendante) ;
// c'est le classement face à mlx qui ne l'était pas.
/// perturbation **near-tie** (0,03 logit médian ≪ marge 4,5-6,0 du modèle ;
/// byte-identique à 30k) — noyée par le bruit du sampling T > 0.
fn kv_bf16_for(default_bf16: bool) -> bool {
    resolve_kv_bf16(kv_bf16_env_override(), default_bf16)
}

/// Résout le format du KV résident full-attn : point unique de décision.
/// f32/bf16 selon [`kv_bf16_for`] (`SARAGOSSA_RUST_KV_BF16=0` garde f32), puis,
/// si bf16, le format de `SARAGOSSA_RUST_KV_QUANT` (bf16 par défaut). Les
/// capacités de la couche sont vérifiées ensuite ([`kv_quant_refusal`]).
pub(super) fn kv_format_for(default_bf16: bool) -> KvFormat {
    #[cfg(test)]
    if let Some(format) = KV_FORMAT_OVERRIDE.with(Cell::get) {
        return format;
    }
    resolve_kv_format(
        kv_bf16_for(default_bf16),
        crate::runtime_flags::kv_quant_format(),
    )
}

#[cfg(test)]
thread_local! {
    static KV_FORMAT_OVERRIDE: Cell<Option<KvFormat>> = const { Cell::new(None) };
}

/// Test : force le format rendu par [`kv_format_for`] pendant `run`, sur ce
/// thread. Les flags sont gelés par `OnceLock` : sans ce levier, un même
/// processus ne peut pas comparer deux formats. Les replis de capacité
/// (`kv_quant_refusal`, fenêtres glissantes) s'appliquent toujours.
#[cfg(test)]
pub(crate) fn with_kv_format_override<R>(format: KvFormat, run: impl FnOnce() -> R) -> R {
    let previous = KV_FORMAT_OVERRIDE.with(|cell| cell.replace(Some(format)));
    let result = run();
    KV_FORMAT_OVERRIDE.with(|cell| cell.set(previous));
    result
}

/// Combine le dtype résolu et le format quantifié demandé : seul un KV bf16
/// est quantifié, pour que les arènes f32 exactes le restent.
fn resolve_kv_format(bf16: bool, quant: KvFormat) -> KvFormat {
    match (bf16, quant) {
        (false, _) => KvFormat::F32,
        (true, KvFormat::F32 | KvFormat::Bf16) => KvFormat::Bf16,
        (true, quant @ (KvFormat::Q8 | KvFormat::K8V4 | KvFormat::Q4)) => quant,
    }
}

/// Raison pour laquelle une couche ne peut pas porter un KV quantifié, `None`
/// si elle le peut. Seul le 2-passes split-K d128/d256 lit le KV quantifié ; il
/// suppose une GQA réelle (comme son déclenchement dense) et les kill-switches
/// flash/2-passes demandent les kernels denses de référence.
pub(super) fn kv_quant_refusal(
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    flash: bool,
    two_pass: bool,
) -> Option<&'static str> {
    if !matches!(head_dim, 128 | 256) {
        Some("head_dim hors 128/256")
    } else if kv_heads == 0 || q_heads % kv_heads != 0 || q_heads / kv_heads < 2 {
        Some("GQA 1 ou invalide")
    } else if q_heads / kv_heads > 32 {
        // Le 2-passes lance 32 threads par tête Q du groupe : au-delà de 32
        // têtes par tête KV, le threadgroup dépasserait 1024 threads.
        Some("GQA > 32")
    } else if !flash {
        Some("SARAGOSSA_RUST_FLASH_SDPA=0")
    } else if !two_pass {
        Some("SARAGOSSA_RUST_SDPA_2PASS=0")
    } else {
        None
    }
}

/// Combine l'override explicite et le défaut : l'override gagne toujours.
fn resolve_kv_bf16(env_override: Option<bool>, default_bf16: bool) -> bool {
    env_override.unwrap_or(default_bf16)
}

/// Une arène dont le défaut est bf16 (prompt long) reste-t-elle en bf16 dense,
/// ni forcée en f32 ni quantifiée ? Condition du K/V de prefill stocké en bf16.
pub(crate) fn kv_bf16_dense_by_default() -> bool {
    kv_bf16_dense(true)
}

/// L'arène d'un KV de défaut `default_bf16` sera-t-elle en bf16 dense, seul
/// format qui adopte les buffers d'un prefill ?
pub(crate) fn kv_bf16_dense(default_bf16: bool) -> bool {
    kv_format_for(default_bf16) == KvFormat::Bf16
}

/// Octets par valeur du KV résident de défaut `default_bf16`, format résolu
/// (overrides compris) ; un format quantifié est compté comme le bf16, borne
/// haute de ses codes et échelles.
pub(crate) fn kv_element_bytes(default_bf16: bool) -> usize {
    match kv_format_for(default_bf16) {
        KvFormat::F32 => std::mem::size_of::<f32>(),
        KvFormat::Bf16 | KvFormat::Q8 | KvFormat::K8V4 | KvFormat::Q4 => std::mem::size_of::<u16>(),
    }
}

/// Override explicite de `SARAGOSSA_RUST_KV_BF16` (lu une fois), cf. [`kv_bf16_for`].
fn kv_bf16_env_override() -> Option<bool> {
    static OVERRIDE: OnceLock<Option<bool>> = OnceLock::new();
    *OVERRIDE.get_or_init(|| {
        crate::runtime_flags::env_var("SARAGOSSA_RUST_KV_BF16")
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
/// divergence. `SARAGOSSA_RUST_KV_BF16_SIM_KONLY=1`, défaut OFF. À utiliser avec
/// `SARAGOSSA_RUST_KV_BF16=0` (buffers f32). N'arrondit que le seed, pas les lignes
/// appendées en decode (négligeables devant le prompt).
pub(super) fn kv_bf16_sim_konly() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("SARAGOSSA_RUST_KV_BF16_SIM_KONLY", false))
}

/// Diagnostic C1B (hors prod) : arrondit **seulement V** au seed prefill (K reste
/// f32 exact), pour isoler l'effet de la précision des valeurs (→ sortie) sur la
/// divergence. `SARAGOSSA_RUST_KV_BF16_SIM_VONLY=1`, défaut OFF.
pub(super) fn kv_bf16_sim_vonly() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("SARAGOSSA_RUST_KV_BF16_SIM_VONLY", false))
}

/// Active la SDPA decode **2-passes split-K** (dédup GQA + tuiles L1) au-delà de
/// `sdpa_2pass_min_len()` rows de KV. Défaut ON ; kill-switch `SARAGOSSA_RUST_SDPA_2PASS=0`.
/// Chemin full-attn en lignes fusionné (RoPE, append KV et attention des `rows`
/// lignes d'une vérification en un dispatch chacun, cf. `rows.rs`). Défaut ON ;
/// `SARAGOSSA_RUST_FULL_ROWS_FUSED=0` rejoue la boucle par ligne.
pub(super) fn full_rows_fused_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("SARAGOSSA_RUST_FULL_ROWS_FUSED", true))
}

pub(super) fn sdpa_2pass_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("SARAGOSSA_RUST_SDPA_2PASS", true))
}

/// Longueur de KV minimale pour basculer en 2-passes.
///
/// Défaut 2048 (décision tuning 2026-07-03, D-30B) : sous ce seuil le
/// single-pass flash est plus rapide, avec moins de dispatches et pas de
/// scratch partials. Override : `SARAGOSSA_RUST_SDPA_2PASS_MIN_LEN`.
pub(super) fn sdpa_2pass_min_len() -> usize {
    static LEN: OnceLock<usize> = OnceLock::new();
    *LEN.get_or_init(|| {
        crate::runtime_flags::env_var("SARAGOSSA_RUST_SDPA_2PASS_MIN_LEN")
            .and_then(|value| value.parse().ok())
            .unwrap_or(2048)
    })
}

/// Clés de KV par bloc split-K (passe 1). Override : `SARAGOSSA_RUST_SDPA_2PASS_KEYS_PER_BLOCK`.
fn sdpa_2pass_keys_per_block() -> usize {
    static KEYS: OnceLock<usize> = OnceLock::new();
    *KEYS.get_or_init(|| {
        crate::runtime_flags::env_var("SARAGOSSA_RUST_SDPA_2PASS_KEYS_PER_BLOCK")
            .and_then(|value| value.parse().ok())
            .filter(|keys| *keys > 0)
            .unwrap_or(128)
    })
}

/// Nombre de blocs split-K pour la SDPA 2-passes : multiple de 32 (requis par la
/// réduction passe-2 `blocks/32`), ≈ `len/128`, borné `[32, 1024]`.
pub(super) fn sdpa_2pass_blocks(len: usize) -> usize {
    let raw = (len / sdpa_2pass_keys_per_block()).max(1);
    let rounded = raw.div_ceil(32) * 32;
    rounded.clamp(32, 1024)
}

#[cfg(test)]
mod kv_bf16_resolution_tests {
    use super::{kv_quant_refusal, parse_kv_bf16_override, resolve_kv_bf16, resolve_kv_format};
    use crate::decode_resident::KvFormat;
    use crate::runtime_flags::parse_kv_quant;

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
    fn precedence_env_explicite_puis_defaut() {
        // Override explicite gagne toujours, quel que soit le défaut.
        assert!(
            resolve_kv_bf16(Some(true), false),
            "env=1 + défaut f32 -> bf16"
        );
        assert!(
            !resolve_kv_bf16(Some(false), true),
            "env=0 + défaut bf16 -> f32"
        );
        assert!(resolve_kv_bf16(None, true), "défaut bf16");
        assert!(!resolve_kv_bf16(None, false), "défaut f32");
    }

    #[test]
    fn table_de_verite_du_format_kv() {
        // (SARAGOSSA_RUST_KV_BF16, défaut de l'appelant, SARAGOSSA_RUST_KV_QUANT) → format.
        let cases = [
            // Arènes f32 exactes (greedy court, extension, TTS, boucle DFlash CPU) :
            // jamais quantifiées.
            (None, false, KvFormat::Q8, KvFormat::F32),
            (None, false, KvFormat::Bf16, KvFormat::F32),
            // Défaut de KV_QUANT : bf16 inchangé.
            (None, true, KvFormat::Bf16, KvFormat::Bf16),
            (None, true, KvFormat::Q8, KvFormat::Q8),
            (None, true, KvFormat::K8V4, KvFormat::K8V4),
            (None, true, KvFormat::Q4, KvFormat::Q4),
            // KV_BF16=0 garde f32 en priorité ; KV_BF16=1 étend la quantification.
            (Some(false), true, KvFormat::Q4, KvFormat::F32),
            (Some(true), false, KvFormat::Q8, KvFormat::Q8),
            (Some(true), false, KvFormat::Bf16, KvFormat::Bf16),
        ];
        for (bf16_override, default_bf16, quant, expected) in cases {
            assert_eq!(
                resolve_kv_format(resolve_kv_bf16(bf16_override, default_bf16), quant),
                expected,
                "KV_BF16={bf16_override:?} défaut={default_bf16} KV_QUANT={quant:?}"
            );
        }
        for (value, expected) in [
            ("bf16", Some(KvFormat::Bf16)),
            ("8", Some(KvFormat::Q8)),
            (" Q8 ", Some(KvFormat::Q8)),
            ("k8v4", Some(KvFormat::K8V4)),
            ("K8V4", Some(KvFormat::K8V4)),
            ("4", Some(KvFormat::Q4)),
            ("q4", Some(KvFormat::Q4)),
            ("", None),
            ("f32", None),
            ("16", None),
        ] {
            assert_eq!(parse_kv_quant(value), expected, "{value:?}");
        }
        // Capacités de la couche : repli bf16 hors 2-passes d128/d256 à GQA réelle.
        assert_eq!(kv_quant_refusal(16, 2, 256, true, true), None);
        assert_eq!(kv_quant_refusal(16, 4, 128, true, true), None);
        assert!(kv_quant_refusal(16, 2, 512, true, true).is_some());
        assert!(kv_quant_refusal(16, 16, 128, true, true).is_some());
        assert!(
            kv_quant_refusal(64, 1, 128, true, true).is_some(),
            "GQA > 32"
        );
        assert_eq!(kv_quant_refusal(32, 1, 128, true, true), None);
        assert!(kv_quant_refusal(16, 0, 128, true, true).is_some());
        assert!(kv_quant_refusal(16, 2, 256, false, true).is_some());
        assert!(kv_quant_refusal(16, 2, 256, true, false).is_some());
    }
}

#[cfg(test)]
mod shared_rows_tests {
    use super::*;

    #[test]
    #[ignore = "nécessite un device Metal"]
    fn resident_accesses_preserve_values_and_logical_bounds() -> Result<()> {
        let device = Device::system_default()
            .ok_or_else(|| InferError::Metal("device Metal absent".to_string()))?;
        let tensor = |element| {
            GpuTensor::adopt(
                device.new_buffer(64, MTLResourceOptions::StorageModeShared),
                8,
                element,
            )
        };
        let f32_tensor = tensor(GpuElement::F32)?;
        let bits = [0x8000_0000, 0x7fc0_0123, 0x3f81_8000];
        write_f32_at(&f32_tensor, 2, &bits.map(f32::from_bits))?;
        assert_eq!(
            read_f32_at(&f32_tensor, 2, 3)?
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            bits
        );
        let bf16_tensor = tensor(GpuElement::Bf16)?;
        write_f32_as_bf16_at(&bf16_tensor, 1, &bits.map(f32::from_bits))?;
        assert_eq!(
            read_f32_at(&bf16_tensor, 1, 3)?
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            [0x8000_0000, 0x7fc0_0000, 0x3f82_0000]
        );
        let u32_tensor = tensor(GpuElement::U32)?;
        write_u32_at(&u32_tensor, 2, &bits)?;
        assert_eq!(
            crate::metal_backend::read_u32_buffer(u32_tensor.buffer(), 5)?[2..],
            bits
        );
        assert!(write_f32_at(&bf16_tensor, 0, &[1.0]).is_err());
        assert!(write_f32_as_bf16_at(&f32_tensor, 0, &[1.0]).is_err());
        assert!(write_u32_at(&f32_tensor, 0, &[1]).is_err());
        assert!(read_f32_at(&u32_tensor, 0, 1).is_err());
        assert!(read_f32_at(&tensor(GpuElement::U8)?, 0, 1).is_err());
        assert!(write_f32_at(&f32_tensor, 8, &[1.0]).is_err());
        assert!(write_f32_as_bf16_at(&bf16_tensor, 8, &[1.0]).is_err());
        assert!(write_u32_at(&u32_tensor, 8, &[1]).is_err());
        assert!(read_f32_at(&f32_tensor, 8, 1).is_err());
        assert!(read_f32_at(&f32_tensor, usize::MAX, 1).is_err());
        write_f32_at(&f32_tensor, 8, &[])?;
        assert!(read_f32_at(&f32_tensor, 8, 0)?.is_empty());
        Ok(())
    }

    #[test]
    fn bf16_conversion_preserves_exact_bits_across_chunks() {
        // Zéros signés, égalités RNE paires/impaires, infinis et payload NaN.
        let inputs = [
            0,
            0x8000_0000,
            0x3f80_8000,
            0x3f81_8000,
            0x7f80_0000,
            0xff80_0000,
            0x7fc0_1234,
        ];
        let expected = [0, 0x8000, 0x3f80, 0x3f82, 0x7f80, 0xff80, 0x7fc0];
        let data: Vec<f32> = (0..PARALLEL_COPY_CHUNK + 17)
            .map(|i| f32::from_bits(inputs[i % inputs.len()]))
            .collect();
        let mut out = vec![0xffff; data.len()];
        write_bf16_bits(&mut out, &data);
        for (i, bits) in out.into_iter().enumerate() {
            assert_eq!(bits, expected[i % expected.len()], "élément {i}");
        }
    }
}

/// Valeurs actives des kernels d'attention, y compris les surcharges locales.
pub(crate) fn snapshot_attention_settings() -> String {
    format!(
        "{:?}",
        (
            flash_sdpa_enabled(),
            kv_bf16_env_override(),
            kv_format_for(false),
            kv_format_for(true),
            kv_bf16_sim_konly(),
            kv_bf16_sim_vonly(),
            full_rows_fused_enabled(),
            sdpa_2pass_enabled(),
            sdpa_2pass_min_len(),
            sdpa_2pass_keys_per_block(),
            super::rows::rows_group_override(),
        )
    )
}
