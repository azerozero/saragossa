//! Flags runtime et instrumentation du décodeur.

#[cfg(all(target_os = "macos", feature = "metal"))]
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

// NOTE: posé par `serve` (multi-modèle) pour forcer le résident linear-attn
// indépendamment de l'env gelé par OnceLock — porté depuis decoder/flags.rs
// lors de l'extraction du module (rebase serve × tangle-quickwins).
#[cfg(all(target_os = "macos", feature = "metal"))]
static FORCE_RESIDENT_FULL_LINEAR: AtomicBool = AtomicBool::new(false);

#[cfg(all(target_os = "macos", feature = "metal"))]
pub(crate) fn force_resident_full_linear_decode() {
    FORCE_RESIDENT_FULL_LINEAR.store(true, Ordering::Relaxed);
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
pub(crate) fn force_resident_full_linear_decode() {}

pub fn env_flag(name: &str, default: bool) -> bool {
    std::env::var(name)
        .ok()
        .and_then(|value| env_flag_value(&value))
        .unwrap_or(default)
}

pub(crate) fn env_flag_value(value: &str) -> Option<bool> {
    let value = value.trim();
    if value == "1" || value.eq_ignore_ascii_case("true") || value.eq_ignore_ascii_case("on") {
        Some(true)
    } else if value == "0"
        || value.eq_ignore_ascii_case("false")
        || value.eq_ignore_ascii_case("off")
    {
        Some(false)
    } else {
        None
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum MtpHistoryPolicy {
    Cycle,
    Committed,
}

pub(super) fn mtp_history_policy() -> MtpHistoryPolicy {
    static POLICY: OnceLock<MtpHistoryPolicy> = OnceLock::new();
    *POLICY.get_or_init(|| {
        std::env::var("RETI_RUST_MTP_HISTORY")
            .ok()
            .map(|value| value.trim().to_ascii_lowercase())
            .and_then(|value| match value.as_str() {
                "committed" | "full" => Some(MtpHistoryPolicy::Committed),
                "cycle" | "reset" => Some(MtpHistoryPolicy::Cycle),
                _ => None,
            })
            .unwrap_or(MtpHistoryPolicy::Committed)
    })
}

/// Chemin MTP fused depth-1 SANS historique committed ni pré-draft spéculatif
/// (draft#2). **Défaut OFF** (`RETI_RUST_MTP_FRESH_CACHE=1` pour l'activer) :
/// diagnostic, pas un gain prod.
///
/// En mode fresh : cache MTP vidé par pas (attention self-only, position 0,
/// comme `generate_mtp1` de MTPLX) et draft#2 supprimé → un forward de tête MTP
/// en moins par pas, sortie byte-identique (gate oracle greedy AR ≡ MTP).
///
/// MESURE (27B, greedy, 256 tok, gpu_lock) : draft#2 alimente l'historique MTP
/// committed qui AMÉLIORE l'acceptance (code α 0,94 committed vs 0,79 fresh),
/// et son coût par pas (~5 ms) est ~exactement compensé par le gain d'acceptance
/// → spec/AR quasi identique (code 1,015× committed vs 1,019× fresh ; prose
/// 0,920× vs 0,953×). draft#2 « se paie » : garder l'historique (défaut OFF).
#[cfg(all(target_os = "macos", feature = "metal"))]
pub(super) fn mtp_fresh_cache_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_MTP_FRESH_CACHE", false))
}

#[cfg(all(target_os = "macos", feature = "metal"))]
pub(super) fn gpu_argmax_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_GPU_ARGMAX", true))
}

#[cfg(all(target_os = "macos", feature = "metal"))]
pub(super) fn gpu_sampler_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_GPU_SAMPLER", true))
}

// NOTE: Sans Metal, aucun chemin GPU n'existe : les flags retombent sur false
// pour que le decode reste sur l'argmax/sampler CPU.
#[cfg(not(all(target_os = "macos", feature = "metal")))]
pub(super) fn gpu_argmax_enabled() -> bool {
    false
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
pub(super) fn gpu_sampler_enabled() -> bool {
    false
}

const DECODE_INTERVAL_UNINITIALIZED_NS: u64 = u64::MAX;
const DECODE_INTERVAL_MAX_NS: u64 = u64::MAX / 2;
static DECODE_MIN_INTERVAL_NS: AtomicU64 = AtomicU64::new(DECODE_INTERVAL_UNINITIALIZED_NS);

/// Intervalle minimum entre deux tokens, dérivé de `RETI_RUST_MAX_TOK_S` (cap
/// tok/s du mode eco/silencieux). `None` = pas de throttle (défaut, pleine
/// vitesse). Quand actif, le decode est rate-limité pour réduire la charge GPU
/// soutenue (moins de ventilo, moins de conso) — sans coût en latence ressentie
/// car la voice loop est TTS-bound. Le pipeline résident est désactivé dans ce
/// mode (le rate-limit sérialise de toute façon).
pub(super) fn decode_min_interval() -> Option<Duration> {
    decode_min_interval_ns().map(Duration::from_nanos)
}

/// Renvoie le plafond courant de decode, si le pacing est actif.
pub fn decode_max_tokens_per_s() -> Option<f64> {
    decode_min_interval_ns().map(|nanos| 1_000_000_000.0 / nanos as f64)
}

/// Configure à chaud le plafond de decode en tokens par seconde.
///
/// `None` désactive le pacing. Les valeurs non finies ou nulles/négatives sont
/// ignorées pour éviter qu'une configuration invalide désactive un réglage
/// actif par accident.
pub fn set_decode_max_tokens_per_s(rate: Option<f64>) {
    match rate {
        Some(rate) => {
            if let Some(nanos) = decode_interval_nanos_for_rate(rate) {
                DECODE_MIN_INTERVAL_NS.store(nanos, AtomicOrdering::Relaxed);
            }
        }
        None => {
            DECODE_MIN_INTERVAL_NS.store(0, AtomicOrdering::Relaxed);
        }
    }
}

fn decode_min_interval_ns() -> Option<u64> {
    let loaded = DECODE_MIN_INTERVAL_NS.load(AtomicOrdering::Relaxed);
    let nanos = if loaded == DECODE_INTERVAL_UNINITIALIZED_NS {
        let initial = std::env::var("RETI_RUST_MAX_TOK_S")
            .ok()
            .and_then(|value| value.trim().parse::<f64>().ok())
            .and_then(decode_interval_nanos_for_rate)
            .unwrap_or(0);
        match DECODE_MIN_INTERVAL_NS.compare_exchange(
            DECODE_INTERVAL_UNINITIALIZED_NS,
            initial,
            AtomicOrdering::Relaxed,
            AtomicOrdering::Relaxed,
        ) {
            Ok(_) => initial,
            Err(current) => current,
        }
    } else {
        loaded
    };
    if nanos == 0 || nanos == DECODE_INTERVAL_UNINITIALIZED_NS {
        None
    } else {
        Some(nanos)
    }
}

fn decode_interval_nanos_for_rate(rate: f64) -> Option<u64> {
    if !rate.is_finite() || rate <= 0.0 {
        return None;
    }
    let nanos = 1_000_000_000.0 / rate;
    if !nanos.is_finite() || nanos <= 0.0 {
        return None;
    }
    Some(nanos.round().clamp(1.0, DECODE_INTERVAL_MAX_NS as f64) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_interval_nanos_for_rate_converts_tok_s_to_interval() {
        assert_eq!(decode_interval_nanos_for_rate(25.0), Some(40_000_000));
    }

    #[test]
    fn decode_interval_nanos_for_rate_rejects_invalid_rates() {
        assert_eq!(decode_interval_nanos_for_rate(0.0), None);
        assert_eq!(decode_interval_nanos_for_rate(-1.0), None);
        assert_eq!(decode_interval_nanos_for_rate(f64::NAN), None);
    }
}

/// Active le decode full-attn résident GPU (tranche 1b). **Défaut OFF** : opt-in
/// `RETI_RUST_DECODE_RESIDENT=1` tant que non prouvé plus rapide ET correct.
/// Le flag n'est lu qu'au setup post-prefill ;
/// le chemin résident s'active ensuite via la présence de `LayerKvCache::full`.
/// Ce repli 1b reste vivant pour diagnostic et sera retiré avec le split de
/// `decoder/resident.rs`, pas dans le tri des flags DG.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub(super) fn decode_resident_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_DECODE_RESIDENT", false))
}

/// Active le decode résident COMPLET (tranche 1c) : tout le forward d'un token
/// en UN command buffer. **Défaut ON**, kill-switch `RETI_RUST_DECODE_RESIDENT_FULL=0`.
/// Distinct du flag 1b (`RETI_RUST_DECODE_RESIDENT`, qui n'active que l'attention
/// full-attn) pour tester 1c indépendamment ; ils fusionneront à la fin de 1c.
///
/// Bascule justifiée par l'oracle 1c.4 sur le 27B `qwen3_5` (256 tokens greedy) :
/// sortie BYTE-IDENTIQUE vs le per-op, command buffers/token 273 → 1, decode
/// ~10 → ~25 tok/s (×2.3-2.6). Le chemin est tout-ou-rien
/// ([`crate::decoder::CausalDecoder::supports_resident_full_decode`]) :
/// `temperature > 0`,
/// lm_head biaisé ou modèle non supporté retombent automatiquement sur le per-op.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub(super) fn decode_resident_full_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_DECODE_RESIDENT_FULL", true))
}

/// Autorise les couches linear-attn dans le decode résident complet. **Défaut
/// ON** après oracle greedy 35B-oQ8 byte-identique vs per-op ; kill-switch
/// `RETI_RUST_DECODE_RESIDENT_FULL_LINEAR=0` pour replier sur le per-op.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub(super) fn decode_resident_full_linear_enabled() -> bool {
    if FORCE_RESIDENT_FULL_LINEAR.load(Ordering::Relaxed) {
        return true;
    }
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_DECODE_RESIDENT_FULL_LINEAR", true))
}

/// Autorise la projection QKV concaténée dans les couches full-attn résidentes.
/// Défaut ON ; kill-switch de diagnostic/tuning pour comparer avec les trois
/// projections séparées sans recompiler.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub(super) fn decode_resident_full_qkv_concat_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_DECODE_RESIDENT_FULL_QKV_CONCAT", true))
}

#[cfg(all(target_os = "macos", feature = "metal"))]
pub(super) fn decode_pipeline_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_DECODE_PIPELINE", true))
}

/// Diagnostic C1B (hors chemin prod) : `RETI_RUST_ORACLE_DUMP_LOGITS=k` dump sur
/// stderr les `k` plus grands logits (`id:valeur_f32`) à chaque pas de decode
/// résident, pour classer les near-ties bf16 vs une vraie dégradation. Renvoie
/// `None` si l'env est absent, non entier ou nul. Quand actif, le decode résident
/// est forcé en non-pipeliné (une complétion GPU par token pour relire l'état).
#[cfg(all(target_os = "macos", feature = "metal"))]
pub(super) fn oracle_dump_logits_topk() -> Option<usize> {
    static TOPK: OnceLock<Option<usize>> = OnceLock::new();
    *TOPK.get_or_init(|| {
        std::env::var("RETI_RUST_ORACLE_DUMP_LOGITS")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .filter(|k| *k > 0)
    })
}

/// Active le pas duo qmm2 du light-batch (E2.2) : 2 flux dans UN command
/// buffer, projections denses batchées. **Défaut ON** quand le modèle le
/// supporte (gate [`crate::decoder::CausalDecoder::supports_resident_duo`]),
/// kill-switch
/// `RETI_RUST_LIGHTBATCH_QMM2=0` → retombe sur le time-slicing E2.1.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub(super) fn lightbatch_qmm2_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_LIGHTBATCH_QMM2", true))
}

/// Active le tail MoE duo du light-batch (E2.3) : router + shared expert
/// batchés qmm2, topk/gathers routés par flux. **Défaut ON** quand les poids
/// MoE sont qmm2-éligibles, kill-switch `RETI_RUST_LIGHTBATCH_MOE2=0` →
/// retombe sur le tail MoE par flux (composition solo).
#[cfg(all(target_os = "macos", feature = "metal"))]
pub(crate) fn lightbatch_moe2_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_LIGHTBATCH_MOE2", true))
}

/// Trace le mode effectif du light-batch (duo qmm2 vs time-slicing, MoE duo)
/// sur stderr — diagnostic du repli silencieux (`RETI_RUST_TRACE_LIGHTBATCH=1`).
#[cfg(all(target_os = "macos", feature = "metal"))]
pub(crate) fn trace_lightbatch_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_TRACE_LIGHTBATCH", false))
}

/// Priorité du flux principal (light-batch E2.5) : le flux de fond (index > 0)
/// n'est décodé qu'un pas sur N (`RETI_RUST_LIGHTBATCH_BG_STRIDE`, défaut 1 =
/// même cadence). N'altère pas la byte-identité par flux (la séquence d'un
/// flux ne dépend que de son propre état, pas de QUAND ses pas s'exécutent).
#[cfg(all(target_os = "macos", feature = "metal"))]
pub(super) fn lightbatch_background_stride() -> u64 {
    static STRIDE: OnceLock<u64> = OnceLock::new();
    *STRIDE.get_or_init(|| {
        std::env::var("RETI_RUST_LIGHTBATCH_BG_STRIDE")
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
            .filter(|stride| *stride >= 1)
            .unwrap_or(1)
    })
}

/// Mesure la disjonction d'experts réelle à M=2
/// (`RETI_RUST_LIGHTBATCH_EXPERT_STATS=1`, **défaut OFF**) : readback des
/// indices top-k des 2 flux par couche MoE duo, synthèse n(2) en fin de run.
/// Diagnostic pur (readback après wait) — ne change AUCUN résultat ; coût
/// readback non nul → jamais actif en prod ni pendant un bench.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub(crate) fn lightbatch_expert_stats_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_LIGHTBATCH_EXPERT_STATS", false))
}

pub(super) fn profile_layer_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_PROFILE_LAYER", false))
}

pub(super) fn prefill_batched_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_PREFILL_BATCH", true))
}

// Défaut ON : routent le prefill des hybrides linéaire+full (qwen3_5_moe) vers le
// chemin batché GPU — couches linear via `linear_attention_cached_batch_resident`,
// couches full via `full_attention_prefill_tail_moe_shared_gated` (attention causale
// batchée GPU + MoE shared rows), MoE batché par lignes. Sans eux le prefill retombe
// tokenwise per-op (`causal_attention` CPU O(seq²)) → 16 tok/s @32k. Avec → ~322 tok/s
// (×20), byte-équivalent. No-op pour les modèles SANS couche linear (30B/27B : déjà
// batchés via les couches full ; le MoE rows n'affecte que les couches linear/le verify).
pub(super) fn prefill_linear_batched_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_PREFILL_LINEAR_BATCH", true))
}

pub(crate) fn prefill_moe_rows_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_PREFILL_MOE_ROWS", true))
}

pub(super) fn prefill_chunk_size() -> usize {
    static CHUNK: OnceLock<usize> = OnceLock::new();
    *CHUNK.get_or_init(|| {
        std::env::var("RETI_RUST_PREFILL_CHUNK")
            .ok()
            .or_else(|| std::env::var("RETI_RUST_PREFILL_STEP_SIZE").ok())
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(0)
    })
}

/// Active l'attention causale longue batchée du prefill résident (27B/30B,
/// seq > 2048 uniquement ; le 35B garde son chemin propre).
///
/// Défaut ON (décision produit 2026-07-04, politique near-tie actée) : l'ordre
/// de réduction softmax diffère du fallback long → déviation possible de la
/// classe near-tie UNIQUEMENT (dossier C1b sur texte réel : 27B ids identiques
/// à 6,3k/8,8k tokens ; 30B un near-tie à ~4k — marge 0,0084 vs 0,46, texte
/// équivalent — ids identiques à 9,2k). Gain mesuré : 27B @8k 48→268 tok/s
/// (×5,6), 30B @9k réel 13→306 (×23) ; le régime 32k devient exploitable.
/// Kill-switch : `RETI_RUST_PREFILL_ATTN_BATCH_LONG=0` (fallback long
/// byte-identique).
#[cfg(all(target_os = "macos", feature = "metal"))]
pub(crate) fn prefill_attn_batch_long_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_PREFILL_ATTN_BATCH_LONG", true))
}

/// Active l'attention causale batchée du prefill résident 30B et 35B sur la
/// fenêtre `257..=2048`.
///
/// Défaut ON (décision produit 2026-07-05, D-NAPRE-2) : le profileur prouve que
/// `causal_attention` est le poste n°1 du 30B à 1k (~85 %) ; même dossier pour
/// le 35B (régression ×2,7 @2k tant qu'il restait sur le `mid` per-query, fix
/// 2026-07-06). Les kernels réutilisés sont les variantes online-softmax déjà
/// qualifiées pour le long (d128 pour le 30B, d256 pour le 35B) ; l'ordre de
/// réduction n'est pas byte-identique au `mid`, donc le flag reste un
/// kill-switch explicite.
/// Kill-switch : `RETI_RUST_PREFILL_ATTN_BATCH_MID_30B=0`.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub(crate) fn prefill_attn_batch_mid_30b_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_PREFILL_ATTN_BATCH_MID_30B", true))
}

/// Active la variante Steel causale d256 du prefill 35B.
///
/// Défaut ON : cette voie reprend le kernel Steel tuilé (Q/K/V en blocs,
/// softmax online par tuile KV, causalité par tuile diagonale) pour remplacer le
/// repli GQA8x4 quand la spécialisation d256 compile sur le GPU courant.
/// Kill-switch dédié : `RETI_RUST_PREFILL_ATTN_STEEL_D256=0`.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub(crate) fn prefill_attn_steel_d256_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_PREFILL_ATTN_STEEL_D256", true))
}

#[cfg(all(target_os = "macos", feature = "metal"))]
pub(super) fn prefill_resident_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_PREFILL_RESIDENT", true))
}

/// Active le tail MLP dense du prefill résident (27B hybride `qwen3_5`).
///
/// Défaut ON (décision produit 2026-07-04) : BYTE-IDENTIQUE au chemin per-op
/// prouvé @312/524/1036 tokens (NA OFF, déterministe) après le fix du plafond
/// 256 de `causal_attention_prefill` — la divergence initiale venait de ce
/// kernel (sortie non écrite au-delà de seq 256), pas du tail dense. Gain
/// mesuré 27B : prefill @1k 29,7 → 192,4 tok/s (×6,5), @8k timeout 900 s →
/// 48,5 tok/s. Kill-switch : `RETI_RUST_PREFILL_DENSE_RESIDENT=0`.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub(super) fn prefill_dense_resident_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_PREFILL_DENSE_RESIDENT", true))
}

#[cfg(all(target_os = "macos", feature = "metal"))]
pub(crate) fn gpu_counters_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_GPU_COUNTERS", false))
}

#[cfg(all(target_os = "macos", feature = "metal"))]
pub(crate) fn gpu_timestamps_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_GPU_TIMESTAMPS", false))
}

#[cfg(all(target_os = "macos", feature = "metal"))]
pub(crate) fn moe_micro_split_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_MOE_MICRO_SPLIT", false))
}

#[cfg(all(target_os = "macos", feature = "metal"))]
pub(crate) fn linear_micro_split_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_LINEAR_MICRO_SPLIT", false))
}

#[cfg(all(target_os = "macos", feature = "metal"))]
pub(crate) fn trace_linear_attn_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_TRACE_LINEAR_ATTN", false))
}

#[cfg(all(target_os = "macos", feature = "metal"))]
pub(crate) fn trace_prefill_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_TRACE_PREFILL", false))
}

#[cfg(all(target_os = "macos", feature = "metal"))]
pub(crate) fn prefill_profile_sections_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_PREFILL_PROFILE_SECTIONS", false))
}

#[cfg(all(target_os = "macos", feature = "metal"))]
pub(crate) fn trace_dispatch_path_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_TRACE_DISPATCH_PATH", false))
}

#[cfg(all(target_os = "macos", feature = "metal"))]
pub(crate) fn decode_profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_DECODE_PROFILE", false))
}

#[cfg(all(target_os = "macos", feature = "metal"))]
pub(crate) fn topk_bench_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_TOPK_BENCH", false))
}

#[cfg(all(target_os = "macos", feature = "metal"))]
pub(crate) fn trace_moe_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_TRACE_MOE", false))
}

#[cfg(all(target_os = "macos", feature = "metal"))]
pub(crate) fn trace_resident_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_TRACE_RESIDENT", false))
}

pub(crate) fn attention_parallel_threshold() -> usize {
    static THRESHOLD: OnceLock<usize> = OnceLock::new();
    *THRESHOLD.get_or_init(|| {
        std::env::var("RETI_RUST_ATTENTION_PAR_THRESHOLD")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(128)
    })
}

#[cfg(all(target_os = "macos", feature = "metal"))]
pub(crate) fn qmv_fast_mode() -> Option<&'static str> {
    static MODE: OnceLock<Option<String>> = OnceLock::new();
    MODE.get_or_init(|| std::env::var("RETI_RUST_QMV_FAST").ok())
        .as_deref()
}

#[cfg(all(target_os = "macos", feature = "metal"))]
pub(crate) fn gather_fast_mode() -> Option<&'static str> {
    static MODE: OnceLock<Option<String>> = OnceLock::new();
    MODE.get_or_init(|| std::env::var("RETI_RUST_GATHER_FAST").ok())
        .as_deref()
}

pub(super) fn prefix_cache_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_PREFIX_CACHE", true))
}

pub(super) fn prefix_cache_capacity() -> usize {
    static CAPACITY: OnceLock<usize> = OnceLock::new();
    *CAPACITY.get_or_init(|| {
        std::env::var("RETI_RUST_PREFIX_CACHE_CAP")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(4)
    })
}

/// Active le prefix-cache par blocs de `saragossa serve`.
pub fn serve_prefix_cache_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_SERVE_PREFIX_CACHE", true))
}

/// Active le pool LRU de modèles de `saragossa serve`.
pub fn serve_lru_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_SERVE_LRU", true))
}

/// Active la garde OOM prédictive de `saragossa serve`.
pub fn serve_oom_guard_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| memory_guard_enabled() && env_flag("RETI_SERVE_OOM_GUARD", true))
}

/// Active la garde mémoire partagée Saragossa.
pub fn memory_guard_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_MEMORY_GUARD", true))
}

/// Renvoie le plafond mémoire statique global, en octets.
pub fn memory_static_cap_bytes() -> Option<u64> {
    static CAP: OnceLock<Option<u64>> = OnceLock::new();
    *CAP.get_or_init(|| {
        std::env::var("RETI_MEMORY_CAP_BYTES")
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
            .filter(|bytes| *bytes > 0)
    })
}

/// Renvoie la marge mémoire globale conservée hors du process.
pub fn memory_headroom_bytes() -> u64 {
    static HEADROOM: OnceLock<u64> = OnceLock::new();
    *HEADROOM.get_or_init(|| {
        std::env::var("RETI_MEMORY_HEADROOM_BYTES")
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
            .unwrap_or(2 * 1024 * 1024 * 1024)
    })
}

/// Renvoie la taille des blocs du prefix-cache serveur.
pub fn serve_prefix_block_tokens() -> usize {
    static TOKENS: OnceLock<usize> = OnceLock::new();
    *TOKENS.get_or_init(|| {
        std::env::var("RETI_SERVE_PREFIX_BLOCK_TOKENS")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .filter(|tokens| *tokens > 0)
            .unwrap_or(256)
    })
}

/// Renvoie la capacité du prefix-cache serveur en blocs.
pub fn serve_prefix_cache_blocks() -> usize {
    static BLOCKS: OnceLock<usize> = OnceLock::new();
    *BLOCKS.get_or_init(|| {
        std::env::var("RETI_SERVE_PREFIX_CACHE_BLOCKS")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(128)
    })
}

/// Renvoie le plafond de blocs du prefix-cache par session.
pub fn serve_prefix_blocks_per_session() -> usize {
    static BLOCKS: OnceLock<usize> = OnceLock::new();
    *BLOCKS.get_or_init(|| {
        std::env::var("RETI_SERVE_PREFIX_BLOCKS_PER_SESSION")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or_else(serve_prefix_cache_blocks)
    })
}

/// Renvoie le nombre cible de modèles résidents dans `serve`.
pub fn serve_model_pool_size() -> usize {
    static MODELS: OnceLock<usize> = OnceLock::new();
    *MODELS.get_or_init(|| {
        std::env::var("RETI_SERVE_MODEL_POOL")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .filter(|models| *models > 0)
            .unwrap_or(2)
    })
}

/// Renvoie le plafond mémoire statique explicite, en octets.
pub fn serve_memory_static_cap_bytes() -> Option<u64> {
    static CAP: OnceLock<Option<u64>> = OnceLock::new();
    *CAP.get_or_init(|| {
        std::env::var("RETI_SERVE_MEMORY_CAP_BYTES")
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
            .filter(|bytes| *bytes > 0)
    })
}

/// Renvoie la marge mémoire conservée hors du process.
pub fn serve_memory_headroom_bytes() -> u64 {
    static HEADROOM: OnceLock<u64> = OnceLock::new();
    *HEADROOM.get_or_init(|| {
        std::env::var("RETI_SERVE_MEMORY_HEADROOM_BYTES")
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
            .unwrap_or(2 * 1024 * 1024 * 1024)
    })
}

pub(super) fn print_layer_profile(
    total_started: Option<Instant>,
    norm: Option<Duration>,
    attention: Option<Duration>,
    tail: Option<Duration>,
) {
    let Some(total_started) = total_started else {
        return;
    };
    let Some(norm) = norm else {
        return;
    };
    let Some(attention) = attention else {
        return;
    };
    let Some(tail) = tail else {
        return;
    };
    eprintln!(
        "profile_layer norm_us={} attention_us={} tail_us={} total_us={}",
        norm.as_micros(),
        attention.as_micros(),
        tail.as_micros(),
        total_started.elapsed().as_micros()
    );
}
