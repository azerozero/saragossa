//! Flags runtime et instrumentation du décodeur.

use super::*;

pub(crate) fn env_flag(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(value) => {
            if value == "1"
                || value.eq_ignore_ascii_case("true")
                || value.eq_ignore_ascii_case("on")
            {
                true
            } else if value == "0"
                || value.eq_ignore_ascii_case("false")
                || value.eq_ignore_ascii_case("off")
            {
                false
            } else {
                default
            }
        }
        Err(_) => default,
    }
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

/// Active le decode full-attn résident GPU (tranche 1b). **Défaut OFF** : opt-in
/// `RETI_RUST_DECODE_RESIDENT=1` tant que non prouvé plus rapide ET correct
/// (cf. /tmp/rust_infer_plan.md). Le flag n'est lu qu'au setup post-prefill ;
/// le chemin résident s'active ensuite via la présence de `LayerKvCache::full`.
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
/// ([`super::CausalDecoder::supports_resident_full_decode`]) : `temperature > 0`,
/// lm_head biaisé ou modèle non supporté retombent automatiquement sur le per-op.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub(super) fn decode_resident_full_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_DECODE_RESIDENT_FULL", true))
}

#[cfg(all(target_os = "macos", feature = "metal"))]
pub(super) fn decode_pipeline_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_DECODE_PIPELINE", true))
}

pub(super) fn profile_layer_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_PROFILE_LAYER", false))
}

pub(super) fn prefill_batched_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_PREFILL_BATCH", true))
}

#[cfg(all(target_os = "macos", feature = "metal"))]
pub(super) fn prefill_resident_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_PREFILL_RESIDENT", true))
}

#[allow(
    dead_code,
    reason = "trace des normes MTP, réactivable au debug du loader"
)]
pub(super) fn mtp_norm_trace_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_MTP_NORM_TRACE", false))
}

#[cfg(all(target_os = "macos", feature = "metal"))]
pub(crate) fn gpu_counters_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_GPU_COUNTERS", false))
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

pub(crate) fn attention_parallel_threshold() -> usize {
    static THRESHOLD: OnceLock<usize> = OnceLock::new();
    *THRESHOLD.get_or_init(|| {
        std::env::var("RETI_RUST_ATTENTION_PAR_THRESHOLD")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
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
    *ENABLED.get_or_init(|| {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if linear_attn_resident_decode_enabled() {
            return false;
        }
        env_flag("RETI_RUST_PREFIX_CACHE", true)
    })
}

#[cfg(all(target_os = "macos", feature = "metal"))]
pub(super) fn linear_attn_resident_decode_enabled() -> bool {
    // Doit suivre `linear_attn_resident_step_enabled` (linear_attention.rs) :
    // défaut **ON**, kill-switch `RETI_RUST_LINEAR_ATTN_RESIDENT=0`. Quand le
    // résident est actif, le prefix-cache est désactivé (l'état récurrent Metal
    // n'est pas snapshotable dans une entrée de cache ; cf. `LinearAttentionCache::clone`
    // qui drop le buffer Metal) → multi-tour re-préfill (TTFT ↑) mais decode +43 %.
    env_flag("RETI_RUST_LINEAR_ATTN_RESIDENT", true)
}

pub(super) fn prefix_cache_capacity() -> usize {
    static CAPACITY: OnceLock<usize> = OnceLock::new();
    *CAPACITY.get_or_init(|| {
        std::env::var("RETI_RUST_PREFIX_CACHE_CAP")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(4)
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
