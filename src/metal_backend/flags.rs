//! Flags runtime du backend Metal.

use crate::runtime_flags::{env_flag, gather_fast_mode, qmv_fast_mode};

use super::*;

#[derive(Clone, Copy, PartialEq, Eq)]
enum KernelOptProfile {
    Default,
    Qwen36Oq8,
    ExploreFused,
}

/// Profil d'optimisation kernel piloté par `RETI_RUST_OPT_PROFILE`.
///
/// Défaut `Default`. PROMOTED infra (décision tangle 2026-07-05) : le preset
/// runtime peut poser `qwen36-oq8`, mais l'env explicite garde la priorité pour
/// les mesures et les retours arrière ciblés.
fn kernel_opt_profile() -> KernelOptProfile {
    static PROFILE: OnceLock<KernelOptProfile> = OnceLock::new();
    *PROFILE.get_or_init(|| match std::env::var("RETI_RUST_OPT_PROFILE") {
        Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
            "qwen36-oq8" | "qwen3.6-oq8" | "linear-bf16" | "bf16" => KernelOptProfile::Qwen36Oq8,
            "explore-fused" | "fused" => KernelOptProfile::ExploreFused,
            _ => KernelOptProfile::Default,
        },
        Err(_) => KernelOptProfile::Default,
    })
}

fn profile_bool_default(name: &str, default: bool) -> bool {
    match kernel_opt_profile() {
        KernelOptProfile::Default => default,
        KernelOptProfile::Qwen36Oq8 => match name {
            "RETI_RUST_LINEAR_SSM_BF16" => true,
            "RETI_RUST_LINEAR_INV_DELTA" => false,
            "RETI_RUST_LINEAR_CONV_NORM_FUSED" => false,
            "RETI_RUST_LINEAR_PAIR_BARRIER_COALESCE" => true,
            "RETI_RUST_QMV_U8_TG128" => true,
            _ => default,
        },
        KernelOptProfile::ExploreFused => match name {
            "RETI_RUST_LINEAR_SSM_BF16" => true,
            "RETI_RUST_LINEAR_INV_DELTA" => false,
            "RETI_RUST_LINEAR_CONV_NORM_FUSED" => true,
            "RETI_RUST_QMV_U8_TG128" => true,
            _ => default,
        },
    }
}

pub(super) fn profile_env_flag(name: &str, default: bool) -> bool {
    env_flag(name, profile_bool_default(name, default))
}

// Ces flags u8 sont des choix de preset, pas des défauts indépendants : le
// preset Default reste OFF en prod, et l'env explicite garde la priorité pour
// les mesures/explorations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct U8KernelPresetDefaults {
    fused_gate_up: bool,
    fused_shared_gate_up: bool,
    fused_shared_gate_scalar: bool,
    fused_shared_gate_qmv: bool,
    fused_moe_down_weighted: bool,
    qmv_tg256: bool,
    qmv_dot4: bool,
    full_qkv_split_rms: bool,
    full_o_proj_gated: bool,
}

fn u8_kernel_preset_defaults_for(profile: KernelOptProfile) -> U8KernelPresetDefaults {
    match profile {
        KernelOptProfile::Default => U8KernelPresetDefaults {
            fused_gate_up: false,
            fused_shared_gate_up: false,
            fused_shared_gate_scalar: false,
            fused_shared_gate_qmv: false,
            fused_moe_down_weighted: false,
            qmv_tg256: false,
            qmv_dot4: false,
            full_qkv_split_rms: cfg!(test),
            full_o_proj_gated: cfg!(test),
        },
        KernelOptProfile::Qwen36Oq8 => U8KernelPresetDefaults {
            fused_gate_up: false,
            fused_shared_gate_up: false,
            fused_shared_gate_scalar: false,
            fused_shared_gate_qmv: false,
            fused_moe_down_weighted: true,
            qmv_tg256: false,
            qmv_dot4: false,
            full_qkv_split_rms: false,
            full_o_proj_gated: false,
        },
        KernelOptProfile::ExploreFused => U8KernelPresetDefaults {
            fused_gate_up: true,
            fused_shared_gate_up: true,
            fused_shared_gate_scalar: true,
            fused_shared_gate_qmv: true,
            fused_moe_down_weighted: true,
            qmv_tg256: false,
            qmv_dot4: true,
            full_qkv_split_rms: true,
            full_o_proj_gated: true,
        },
    }
}

fn u8_kernel_preset_defaults() -> U8KernelPresetDefaults {
    u8_kernel_preset_defaults_for(kernel_opt_profile())
}

fn profile_linear_delta_tg_rows(ssm_bf16: bool) -> u64 {
    match kernel_opt_profile() {
        KernelOptProfile::Qwen36Oq8 | KernelOptProfile::ExploreFused if ssm_bf16 => 8,
        _ => {
            if ssm_bf16 {
                8
            } else {
                4
            }
        }
    }
}
pub(crate) fn resident_concurrent_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_RESIDENT_CONCURRENT", false))
}

/// Active la linear-attn gated-delta en forme CHUNKÉE (port chunked-DeltaNet) au lieu
/// du séquentiel token-par-token. Opt-in `RETI_RUST_LINEAR_CHUNKED`. Exige head_dim=128.
pub(crate) fn linear_chunked_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_LINEAR_CHUNKED", false))
}

/// Active le MoE routé prefill via le kernel `gather_qmm` porté (cooperative_tensor
/// quantifié groupé, ~7,6× le qmv). Tri+pad par expert (16-aligné, CPU, readback des
/// indices), UN dispatch par projection lisant le poids empilé packed. Opt-in.
pub(crate) fn moe_coop_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_MOE_COOP", true))
}

/// Active F2 : fusion gate+up+SwiGLU du MoE coop résident. Byte-identique :
/// accumulation f32 inchangée puis cast bf16 direct dans l'épilogue.
pub(crate) fn moe_coop_fused_swiglu_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_MOE_COOP_FUSED_SWIGLU", true))
}

/// Active le MoE routé coop u4 gs64 au prefill (kernels
/// `gemm_nax_coop_qb_grouped_*_u4`).
///
/// Défaut ON (décision produit 2026-07-02) : kernel prouvé bit-identique au
/// chemin u8 coop à poids logiques égaux (tests `*_bit_identical_to_u8_*`) ;
/// la déviation greedy possible sur certains prompts = near-ties bf16
/// tensor-core, la MÊME classe que le dense u4/u8 NA déjà défaut ON (byte-
/// identité prompt-dépendante, sortie cohérente). Gain mesuré : prefill voix
/// 4bit @524 tokens 457 → 248 ms (×1,84). Kill-switch : `RETI_RUST_MOE_COOP_U4=0`.
pub(crate) fn moe_coop_u4_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_MOE_COOP_U4", true))
}

/// Active la bascule routed-only coop du prefill (MoE sans expert partagé, ex.
/// Qwen3-30B-A3B) vers les kernels groupés `gemm_nax_coop_qb_grouped_*`.
///
/// Défaut ON (décision produit 2026-07-04, politique near-tie). Historique :
/// la « divergence » initiale de D-30B était contaminée par le bug du plafond
/// 256 de l'attention prefill (sortie non écrite → déchet déterministe) ;
/// après ce fix, le dossier C1b texte réel (D-COOP-2) qualifie une classe
/// near-tie PROPRE : ids byte-identiques à ~1k ET ~9k tokens, un seul flip à
/// ~4k (marge 0,0084), zéro dégénérescence, zéro dérive d'accumulation ;
/// oracle multi-tuiles gather↔coop aux dimensions 30B : max_abs 1,8e-4
/// (l'ordre de réduction tensor-core groupé ≠ qmv par-ligne, byte-id
/// structurellement hors d'atteinte). Gain mesuré (tronc avec attention
/// batchée) : prefill 30B ×1,25-1,50 sur toute la courbe (@8k 345→517 tok/s,
/// @32k 138→179). Gate INDÉPENDANT de `RETI_RUST_MOE_COOP` (shared 35B).
/// Kill-switch : `RETI_RUST_MOE_ROUTED_COOP_PREFILL=0`.
pub(crate) fn moe_routed_coop_prefill_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_MOE_ROUTED_COOP_PREFILL", true))
}
pub(super) fn qmm_na_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_QMM_NA", true))
}

pub(super) fn qmm_na_gs128_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_QMM_NA_GS128", true))
}

/// Active le GEMM NA fused-tiled comme tête de précédence prefill.
///
/// Défaut ON (décision produit 2026-07-02, C3) : le chemin dense/u8 NA qualifié
/// devient le routeur principal ; kill-switch `RETI_RUST_QMM_NA_FUSED_TILED=0`.
pub(super) fn qmm_na_fused_tiled_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_QMM_NA_FUSED_TILED", true))
}

/// Active le sous-chemin u4 du GEMM NA fused-tiled.
///
/// Défaut ON (décision produit 2026-07-02, C3) sous
/// `RETI_RUST_QMM_NA_FUSED_TILED`; kill-switch
/// `RETI_RUST_QMM_NA_FUSED_TILED_U4=0`.
pub(super) fn qmm_na_fused_tiled_u4_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_QMM_NA_FUSED_TILED_U4", true))
}

pub(super) fn fast_argmax_qmv_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_FAST_ARGMAX_QMV", false))
}

pub(super) fn qmv_one_u8_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_QMV_ONE_U8", true))
}

pub(super) fn qmv_u6_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_QMV_U6", true))
}

pub(super) fn qmv_u8_tg128_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| profile_env_flag("RETI_RUST_QMV_U8_TG128", false))
}

pub(super) fn qmv_u8_tg256_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        env_flag(
            "RETI_RUST_QMV_U8_TG256",
            u8_kernel_preset_defaults().qmv_tg256,
        )
    })
}

pub(super) fn qmv_u8_dot4_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        env_flag(
            "RETI_RUST_QMV_U8_DOT4",
            u8_kernel_preset_defaults().qmv_dot4,
        )
    })
}

pub(super) fn linear_z_beta_gate_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_LINEAR_Z_BETA_GATE", true))
}

pub(super) fn linear_full_concat_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_LINEAR_FULL_CONCAT", true))
}

pub(super) fn linear_pair_barrier_coalesce_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| profile_env_flag("RETI_RUST_LINEAR_PAIR_BARRIER_COALESCE", false))
}

pub(super) fn fast_affine_qmv_enabled(out_dim: usize) -> bool {
    match qmv_fast_mode() {
        Some("0") | Some("false") | Some("off") => false,
        Some("1") | Some("true") | Some("all") => true,
        Some("q") => out_dim == 4096,
        Some("o") => out_dim == 2048,
        Some("kv") => out_dim == 512,
        Some("qkv") => out_dim == 4096 || out_dim == 512,
        Some("wide") => (2048..=4096).contains(&out_dim),
        Some("mid") => (512..=4096).contains(&out_dim),
        Some("small") => out_dim <= 4096,
        Some("large") => out_dim >= 65_536,
        _ => true,
    }
}

pub(super) fn fused_attn_epilogue_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_FUSED_ATTN_EPILOGUE", true))
}

pub(super) fn fused_rms_prologue_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_FUSED_RMS_PROLOGUE", true))
}

pub(super) fn dense_qmv_fast_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_DENSE_QMV_FAST", true))
}

pub(super) fn topk8_fast_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_TOPK8_FAST", true))
}

pub(super) fn linear_delta_dk128_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_LINEAR_DELTA_DK128", true))
}

pub(super) fn linear_delta_tg_rows() -> u64 {
    static ROWS: OnceLock<u64> = OnceLock::new();
    *ROWS.get_or_init(|| linear_delta_tg_rows_override().unwrap_or(4))
}

pub(super) fn linear_delta_tg_rows_for_state(ssm_bf16: bool) -> u64 {
    linear_delta_tg_rows_override().unwrap_or_else(|| profile_linear_delta_tg_rows(ssm_bf16))
}

fn linear_delta_tg_rows_override() -> Option<u64> {
    std::env::var("RETI_RUST_LINEAR_DELTA_TG_ROWS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| matches!(*value, 1 | 2 | 4 | 8 | 16 | 32))
}

pub(super) fn linear_inv_delta_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| profile_env_flag("RETI_RUST_LINEAR_INV_DELTA", false))
}

pub(super) fn linear_ssm_bf16_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| profile_env_flag("RETI_RUST_LINEAR_SSM_BF16", false))
}

pub(super) fn linear_rms_dv128_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_LINEAR_RMS_DV128", true))
}

pub(super) fn linear_conv_k4_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_LINEAR_CONV_K4", true))
}

pub(super) fn linear_norm_dk128_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_LINEAR_NORM_DK128", true))
}

pub(super) fn linear_conv_norm_fused_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| profile_env_flag("RETI_RUST_LINEAR_CONV_NORM_FUSED", false))
}

pub(super) fn fused_gate_up_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_FUSED_GATE_UP", true))
}

pub(super) fn fused_gate_up_u8_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        env_flag(
            "RETI_RUST_FUSED_GATE_UP_U8",
            u8_kernel_preset_defaults().fused_gate_up,
        )
    })
}

/// Active la fusion gate+up+swiglu du **shared-expert** (tranche 3).
pub(super) fn fused_shared_gate_up_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_FUSED_SHARED_GATE_UP", true))
}

pub(super) fn fused_shared_gate_up_u8_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        env_flag(
            "RETI_RUST_FUSED_SHARED_GATE_UP_U8",
            u8_kernel_preset_defaults().fused_shared_gate_up,
        )
    })
}

pub(super) fn fused_shared_gate_scalar_u8_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        env_flag(
            "RETI_RUST_FUSED_SHARED_GATE_SCALAR_U8",
            u8_kernel_preset_defaults().fused_shared_gate_scalar,
        )
    })
}

pub(super) fn fused_shared_gate_qmv_u8_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        env_flag(
            "RETI_RUST_FUSED_SHARED_GATE_QMV_U8",
            u8_kernel_preset_defaults().fused_shared_gate_qmv,
        )
    })
}

pub(super) fn fused_moe_down_weighted_u8_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        env_flag(
            "RETI_RUST_FUSED_MOE_DOWN_WEIGHTED_U8",
            u8_kernel_preset_defaults().fused_moe_down_weighted,
        )
    })
}

pub(super) fn full_qkv_split_rms_u8_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        env_flag(
            "RETI_RUST_FULL_QKV_SPLIT_RMS_U8",
            u8_kernel_preset_defaults().full_qkv_split_rms,
        )
    })
}

pub(super) fn full_o_proj_gated_u8_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        env_flag(
            "RETI_RUST_FULL_O_PROJ_GATED_U8",
            u8_kernel_preset_defaults().full_o_proj_gated,
        )
    })
}

pub(super) fn moe_shared_route_overlap_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_MOE_SHARED_ROUTE_OVERLAP", true))
}

pub(super) fn topk_parallel_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("RETI_RUST_TOPK_PARALLEL", true))
}

pub(super) fn scratch_resource_options() -> MTLResourceOptions {
    if private_scratch_enabled() {
        MTLResourceOptions::StorageModePrivate
    } else {
        MTLResourceOptions::StorageModeShared
    }
}

pub(super) fn private_scratch_enabled() -> bool {
    static PRIVATE: OnceLock<bool> = OnceLock::new();
    *PRIVATE.get_or_init(|| env_flag("RETI_RUST_PRIVATE_SCRATCH", false))
}

pub(super) fn fast_gather_qmv_enabled(weight: &StackedAffineBuffers) -> bool {
    match gather_fast_mode() {
        Some("0") | Some("false") | Some("off") => false,
        Some("gateup") => weight.out_dim <= weight.in_dim,
        Some("down") => weight.out_dim > weight.in_dim,
        _ => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn u8_kernel_preset_defaults_preserve_default_and_test_guards() {
        let defaults = u8_kernel_preset_defaults_for(KernelOptProfile::Default);
        assert_eq!(
            defaults,
            U8KernelPresetDefaults {
                fused_gate_up: false,
                fused_shared_gate_up: false,
                fused_shared_gate_scalar: false,
                fused_shared_gate_qmv: false,
                fused_moe_down_weighted: false,
                qmv_tg256: false,
                qmv_dot4: false,
                full_qkv_split_rms: cfg!(test),
                full_o_proj_gated: cfg!(test),
            }
        );
    }

    #[test]
    fn u8_kernel_preset_defaults_follow_exploration_profiles() {
        let qwen36 = u8_kernel_preset_defaults_for(KernelOptProfile::Qwen36Oq8);
        assert!(qwen36.fused_moe_down_weighted);
        assert!(!qwen36.fused_gate_up);
        assert!(!qwen36.full_qkv_split_rms);
        assert!(!qwen36.full_o_proj_gated);

        let explore = u8_kernel_preset_defaults_for(KernelOptProfile::ExploreFused);
        assert!(explore.fused_gate_up);
        assert!(explore.fused_shared_gate_up);
        assert!(explore.fused_shared_gate_scalar);
        assert!(explore.fused_shared_gate_qmv);
        assert!(explore.fused_moe_down_weighted);
        assert!(!explore.qmv_tg256);
        assert!(explore.qmv_dot4);
        assert!(explore.full_qkv_split_rms);
        assert!(explore.full_o_proj_gated);
    }
}
