//! Flags runtime du backend Metal.

use crate::runtime_flags::{env_flag, gather_fast_mode, qmv_fast_mode};

use super::*;

/// Compilation Metal rapide, active par défaut (`SARAGOSSA_RUST_METAL_FAST_MATH=1`).
/// `=0` désactive fast-math pour diagnostiquer les écarts numériques. Lu une fois
/// au premier usage : régler avant de créer le runtime, puis relancer le processus.
/// Les kernels déjà précis (quantification KV et FlashNext) le restent.
///
/// Limite : les appels explicites `fast::` restent rapides, même avec `=0`.
/// Dans `src/kernels.metal`, les 22 appels sont tous des `fast::exp` :
/// - `weighted_sum_add_grouped_topk_f32` (1) ;
/// - `weighted_sum_add_shared_topk_f32` (1) ;
/// - `affine_gather_down_weighted_shared_fast_u8_gs64_f32` (1) ;
/// - `add_sigmoid_scaled_f32` (1) ;
/// - `add_sigmoid_scaled_rows_f32` (1) ;
/// - `whisper_attn_decode_vec64_f32` (3) ;
/// - `causal_attention_prefill_batch_long_d128_f32` (2) ;
/// - `attention_prefill_sink_body` (2) ;
/// - `causal_attention_prefill_batch_long_d256_f32` (2) ;
/// - `causal_attention_prefill_batch_gqa8x4_d256_f32` (8).
///
/// Les sources embarquées du decode résident et des assets peuvent aussi utiliser
/// `fast::` : ce flag ne garantit donc pas une arithmétique précise partout.
pub(super) fn metal_fast_math_enabled() -> bool {
    #[cfg(test)]
    if let Some(forced) = METAL_FAST_MATH_TEST_OVERRIDE.with(std::cell::Cell::get) {
        return forced;
    }
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("SARAGOSSA_RUST_METAL_FAST_MATH", true))
}

#[cfg(test)]
thread_local! {
    static METAL_FAST_MATH_TEST_OVERRIDE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
}

/// Force les options des compilations sur ce thread le temps de `run`.
#[cfg(test)]
pub(crate) fn with_metal_fast_math_override<R>(enabled: bool, run: impl FnOnce() -> R) -> R {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            METAL_FAST_MATH_TEST_OVERRIDE.with(|cell| cell.set(self.0));
        }
    }
    let _restore = Restore(METAL_FAST_MATH_TEST_OVERRIDE.with(|cell| cell.replace(Some(enabled))));
    run()
}

pub(crate) fn resident_concurrent_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("SARAGOSSA_RUST_RESIDENT_CONCURRENT", false))
}

/// MLP dense fusionné gate+up+SwiGLU au prefill (kernel
/// `gemm_nax_coop_qb_tiled_u4_gate_up_swiglu`, sortie bf16 vers le down_proj).
/// Actif par défaut depuis le 2026-10-21, kill-switch
/// `SARAGOSSA_RUST_PREFILL_FUSED_MLP=0`.
pub(crate) fn prefill_fused_mlp_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("SARAGOSSA_RUST_PREFILL_FUSED_MLP", true))
}

/// Flag historique retiré. Une demande `SARAGOSSA_RUST_LINEAR_CHUNKED=1`
/// est refusée dès la construction du backend : garder le NAX ou le scan par défaut.
pub(crate) fn linear_chunked_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("SARAGOSSA_RUST_LINEAR_CHUNKED", false))
}

/// Refuse le chemin retiré avant tout accès Metal, indépendamment du mode
/// résident ou per-op et de leurs replis sur erreur d'exécution.
pub(super) fn validate_linear_chunked_flag() -> Result<()> {
    if linear_chunked_enabled() {
        return Err(InferError::Config(
            "SARAGOSSA_RUST_LINEAR_CHUNKED: chemin retiré (course, bornes et ratios de decay); supprimer le flag pour conserver NAX ou le scan séquentiel".into(),
        ));
    }
    Ok(())
}

/// Active le MoE routé prefill via le kernel `gather_qmm` porté (cooperative_tensor
/// quantifié groupé, ~7,6× le qmv). Tri+pad par expert (16-aligné, CPU, readback des
/// indices), UN dispatch par projection lisant le poids empilé packed. Opt-in.
pub(crate) fn moe_coop_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("SARAGOSSA_RUST_MOE_COOP", true))
}

/// Active F2 : fusion gate+up+SwiGLU du MoE coop résident. Byte-identique :
/// accumulation f32 inchangée puis cast bf16 direct dans l'épilogue.
pub(crate) fn moe_coop_fused_swiglu_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("SARAGOSSA_RUST_MOE_COOP_FUSED_SWIGLU", true))
}

/// Active le MoE routé coop u4 gs64 au prefill (kernels
/// `gemm_nax_coop_qb_grouped_*_u4`).
///
/// Défaut ON (décision produit 2026-07-02) : kernel prouvé bit-identique au
/// chemin u8 coop à poids logiques égaux (tests `*_bit_identical_to_u8_*`) ;
/// la déviation greedy possible sur certains prompts = near-ties bf16
/// tensor-core, la MÊME classe que le dense u4/u8 NA déjà défaut ON (byte-
/// identité prompt-dépendante, sortie cohérente). Gain mesuré : prefill voix
/// 4bit @524 tokens 457 → 248 ms (×1,84). Kill-switch : `SARAGOSSA_RUST_MOE_COOP_U4=0`.
pub(crate) fn moe_coop_u4_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("SARAGOSSA_RUST_MOE_COOP_U4", true))
}

/// Active la bascule routed coop du prefill (MoE routé du tail résident : Qwen3-30B-A3B
/// sans expert partagé, et flint avec son expert partagé sans gate, cf.
/// [`flint_moe_coop_enabled`]) vers les kernels groupés `gemm_nax_coop_qb_grouped_*`.
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
/// @32k 138→179). Gate INDÉPENDANT de `SARAGOSSA_RUST_MOE_COOP` (shared 35B).
/// Kill-switch : `SARAGOSSA_RUST_MOE_ROUTED_COOP_PREFILL=0`.
pub(crate) fn moe_routed_coop_prefill_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("SARAGOSSA_RUST_MOE_ROUTED_COOP_PREFILL", true))
}

/// Étend la bascule routed coop du prefill au MoE flint (routeur sigmoïde et
/// expert partagé sans gate). Exige aussi `SARAGOSSA_RUST_MOE_ROUTED_COOP_PREFILL`.
///
/// Profilage : ce chemin publie une seule section `moe_routed_coop`, qui couvre
/// `router_topk` + `moe_gate_up` + `moe_down` + `moe_weighted_sum` du chemin qmv.
///
/// Défaut ON (mesuré le 30 oct. 2026 sur flint-int4) : à 32k, MoE routé
/// 18,5 → 5,2 s et prefill 37,2 → 24,6 s ; à 16k, logits coop contre qmv
/// max 3,3 / moyenne 0,50, argmax 15/15 et 16 tokens greedy identiques ; 16
/// tokens greedy mlx identiques. Seule la position 3 de l'oracle mlx dépasse
/// les seuils f32 (14,4 / 2,1), comme mlx bf16 contre mlx f32 (16,9 / 2,6) :
/// quasi-égalité de routage sensible à l'arrondi bf16. Les 21 écarts de
/// top-k coop contre qmv tracés sur ce prompt sont tous des quasi-égalités
/// (`flint_int4_metal_matches_mlx_oracle`).
/// Kill-switch : `SARAGOSSA_RUST_FLINT_MOE_COOP=0`.
pub(crate) fn flint_moe_coop_enabled() -> bool {
    #[cfg(test)]
    if let Some(forced) = FLINT_MOE_COOP_TEST_OVERRIDE.with(std::cell::Cell::get) {
        return forced;
    }
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("SARAGOSSA_RUST_FLINT_MOE_COOP", true))
}

/// Kernels groupés TUILÉS pour le MoE routé coop de flint
/// (`gemm_nax_coop_qb_grouped_*_u4_bm*`) : tuiles d'un expert de
/// `MOE_COOP_TILED_BM` lignes × 64 colonnes au lieu de 16 × 32, poids
/// déquantifié une fois par tuile en mémoire threadgroup. Même arithmétique
/// que les kernels 16 × 32 (oracle bit à bit dans les tests). Qwen3-30B
/// (softmax) garde les kernels 16 × 32 : pas de modèle local pour mesurer.
///
/// NOTE: Mesuré le 31 oct. 2026 sur flint-int4 (prefill seul, morceaux de
/// 4096) : MoE routé 4,49 → 3,17 s à 32k et 9,89 → 6,99 s à 64k, prefill
/// 10,6 → 9,4 s et 27,2 → 23,5 s. Hidden du prefill 16k bit à bit identiques
/// aux kernels 16 × 32 et greedy identique
/// (`flint_int4_moe_coop_tiled_matches_16x32`).
/// Kill-switch `SARAGOSSA_RUST_FLINT_MOE_COOP_TILED=0`.
pub(crate) fn flint_moe_coop_tiled_enabled() -> bool {
    #[cfg(test)]
    if let Some(forced) = FLINT_MOE_COOP_TILED_TEST_OVERRIDE.with(std::cell::Cell::get) {
        return forced;
    }
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("SARAGOSSA_RUST_FLINT_MOE_COOP_TILED", true))
}

#[cfg(test)]
thread_local! {
    static FLINT_MOE_COOP_TILED_TEST_OVERRIDE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
}

/// Force les kernels coop tuilés de flint sur ce thread le temps de `run`.
#[cfg(test)]
pub(crate) fn with_flint_moe_coop_tiled_override<R>(enabled: bool, run: impl FnOnce() -> R) -> R {
    /// Rétablit le réglage précédent, panique comprise.
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            FLINT_MOE_COOP_TILED_TEST_OVERRIDE.with(|cell| cell.set(self.0));
        }
    }
    let _restore =
        Restore(FLINT_MOE_COOP_TILED_TEST_OVERRIDE.with(|cell| cell.replace(Some(enabled))));
    run()
}

#[cfg(test)]
thread_local! {
    // Par thread : les tests parallèles comparent coop et qmv sans se gêner.
    static FLINT_MOE_COOP_TEST_OVERRIDE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
}

/// Force la bascule flint coop sur ce thread le temps de `run`, pour un
/// oracle coop contre qmv (`OnceLock` non réglable dans le processus de test).
#[cfg(test)]
pub(crate) fn with_flint_moe_coop_override<R>(enabled: bool, run: impl FnOnce() -> R) -> R {
    /// Rétablit le réglage précédent, panique comprise.
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            FLINT_MOE_COOP_TEST_OVERRIDE.with(|cell| cell.set(self.0));
        }
    }
    let _restore = Restore(FLINT_MOE_COOP_TEST_OVERRIDE.with(|cell| cell.replace(Some(enabled))));
    run()
}
#[cfg(test)]
thread_local! {
    // Par thread, comme la bascule coop : greedy NA contre qmv sur un même
    // modèle chargé, sans recharger 22 Go de poids.
    static QMM_NA_U4_MIN_DIM_TEST_OVERRIDE: std::cell::Cell<Option<usize>> =
        const { std::cell::Cell::new(None) };
}

/// Plancher du GEMM NA u4 forcé sur ce thread, prioritaire sur l'executor.
#[cfg(test)]
pub(super) fn qmm_na_u4_min_dim_test_override() -> Option<usize> {
    QMM_NA_U4_MIN_DIM_TEST_OVERRIDE.with(std::cell::Cell::get)
}

/// Force le plancher du GEMM NA u4 sur ce thread le temps de `run`.
#[cfg(test)]
pub(crate) fn with_qmm_na_u4_min_dim_override<R>(min_dim: usize, run: impl FnOnce() -> R) -> R {
    /// Rétablit le réglage précédent, panique comprise.
    struct Restore(Option<usize>);
    impl Drop for Restore {
        fn drop(&mut self) {
            QMM_NA_U4_MIN_DIM_TEST_OVERRIDE.with(|cell| cell.set(self.0));
        }
    }
    let _restore =
        Restore(QMM_NA_U4_MIN_DIM_TEST_OVERRIDE.with(|cell| cell.replace(Some(min_dim))));
    run()
}

#[cfg(test)]
thread_local! {
    // Par thread comme le plancher forcé : prouve qu'un greedy réel a bien
    // emprunté le bf16×3, sans compter les dispatches des autres tests.
    static QMM_NA_U4_X3_DISPATCHES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Compte un dispatch du GEMM NA u4 bf16×3 sur ce thread.
#[cfg(test)]
pub(super) fn count_qmm_na_u4_x3_dispatch() {
    QMM_NA_U4_X3_DISPATCHES.with(|cell| cell.set(cell.get() + 1));
}

/// Nombre de dispatches du GEMM NA u4 bf16×3 encodés sur ce thread.
#[cfg(test)]
pub(crate) fn qmm_na_u4_x3_dispatches_for_test() -> usize {
    QMM_NA_U4_X3_DISPATCHES.with(std::cell::Cell::get)
}

pub(super) fn qmm_na_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("SARAGOSSA_RUST_QMM_NA", true))
}

pub(super) fn qmm_na_gs128_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("SARAGOSSA_RUST_QMM_NA_GS128", true))
}

/// Active le GEMM NA fused-tiled comme tête de précédence prefill.
///
/// Défaut ON (décision produit 2026-07-02, C3) : le chemin dense/u8 NA qualifié
/// devient le routeur principal ; kill-switch `SARAGOSSA_RUST_QMM_NA_FUSED_TILED=0`.
pub(super) fn qmm_na_fused_tiled_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("SARAGOSSA_RUST_QMM_NA_FUSED_TILED", true))
}

/// Active le sous-chemin u4 du GEMM NA fused-tiled.
///
/// Défaut ON (décision produit 2026-07-02, C3) sous
/// `SARAGOSSA_RUST_QMM_NA_FUSED_TILED`; kill-switch
/// `SARAGOSSA_RUST_QMM_NA_FUSED_TILED_U4=0`.
pub(super) fn qmm_na_fused_tiled_u4_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("SARAGOSSA_RUST_QMM_NA_FUSED_TILED_U4", true))
}

pub(super) fn fast_argmax_qmv_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("SARAGOSSA_RUST_FAST_ARGMAX_QMV", false))
}

pub(super) fn qmv_one_u8_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("SARAGOSSA_RUST_QMV_ONE_U8", true))
}

/// Opt-in tant que l'égalité bit à bit portée de reti n'est pas validée ici.
/// `SARAGOSSA_RUST_QMM_U6=0` conserve le qmv répliqué pour le prefill.
pub(super) fn qmm_u6_enabled() -> bool {
    #[cfg(test)]
    if let Some(forced) = QMM_U6_TEST_OVERRIDE.with(std::cell::Cell::get) {
        return forced;
    }
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("SARAGOSSA_RUST_QMM_U6", false))
}

#[cfg(test)]
thread_local! {
    // Isole le flag du OnceLock et des autres tests exécutés en parallèle.
    static QMM_U6_TEST_OVERRIDE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
}

#[cfg(test)]
pub(super) fn with_qmm_u6_override<R>(enabled: bool, run: impl FnOnce() -> R) -> R {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            QMM_U6_TEST_OVERRIDE.with(|cell| cell.set(self.0));
        }
    }
    let _restore = Restore(QMM_U6_TEST_OVERRIDE.with(|cell| cell.replace(Some(enabled))));
    run()
}

pub(super) fn qmv_u6_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("SARAGOSSA_RUST_QMV_U6", true))
}

pub(super) fn linear_z_beta_gate_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("SARAGOSSA_RUST_LINEAR_Z_BETA_GATE", true))
}

/// Projection d'entrée linear-attn concaténée (`weights.full`),
/// `SARAGOSSA_RUST_LINEAR_FULL_CONCAT`, défaut ON. NOTE: un seul interrupteur
/// pour le mono-ligne ET la vérification en lignes (27B dense compris) : `=0`
/// coupe les deux, il n'y a pas de repli propre aux lignes.
pub(super) fn linear_full_concat_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("SARAGOSSA_RUST_LINEAR_FULL_CONCAT", true))
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
    #[cfg(feature = "devtools")]
    if let Some(case) = crate::runtime_flags::pairwise::current() {
        return case.p1;
    }
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("SARAGOSSA_RUST_FUSED_ATTN_EPILOGUE", true))
}

pub(super) fn fused_rms_prologue_enabled() -> bool {
    #[cfg(feature = "devtools")]
    if let Some(case) = crate::runtime_flags::pairwise::current() {
        return case.p1;
    }
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("SARAGOSSA_RUST_FUSED_RMS_PROLOGUE", true))
}

/// Fusion `down_proj` u4 + résiduel du tail dense résident (pairwise p2).
/// OFF hors harnais pairwise : aucun env, défaut de production inchangé.
pub(super) fn qmv_u4_residual_enabled() -> bool {
    #[cfg(feature = "devtools")]
    if let Some(case) = crate::runtime_flags::pairwise::current() {
        return case.p2;
    }
    false
}

pub(super) fn dense_qmv_fast_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("SARAGOSSA_RUST_DENSE_QMV_FAST", true))
}

pub(super) fn topk8_fast_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("SARAGOSSA_RUST_TOPK8_FAST", true))
}

pub(super) fn linear_delta_dk128_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("SARAGOSSA_RUST_LINEAR_DELTA_DK128", true))
}

/// Captures SSM de la vérification en lignes en mode journal : état d'entrée et
/// journal delta/k/decay au lieu d'un état complet par ligne, état accepté rejoué au
/// rollback (bit-identique). `SARAGOSSA_RUST_LINEAR_CAPTURE_JOURNAL=1` (défaut OFF).
pub(super) fn linear_capture_journal_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        crate::runtime_flags::env_flag("SARAGOSSA_RUST_LINEAR_CAPTURE_JOURNAL", false)
    })
}

/// Vérification MTP : conv/normes/gates et rms-gate batchés sur les lignes du
/// draft (kernels du prefill) au lieu d'une boucle de dispatches par ligne.
/// `SARAGOSSA_RUST_LINEAR_ROWS_BATCHED=0` rétablit la boucle (repli A/B).
/// Delta par séquence avec captures SSM dans le chemin en lignes (un dispatch au
/// lieu de `rows` delta + copies). `SARAGOSSA_RUST_LINEAR_ROWS_SEQ_DELTA=0` : boucle.
pub(super) fn linear_rows_seq_delta_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        crate::runtime_flags::env_flag("SARAGOSSA_RUST_LINEAR_ROWS_SEQ_DELTA", true)
    })
}

pub(super) fn linear_rows_batched_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("SARAGOSSA_RUST_LINEAR_ROWS_BATCHED", true))
}

pub(super) fn linear_rms_dv128_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("SARAGOSSA_RUST_LINEAR_RMS_DV128", true))
}

pub(super) fn linear_conv_k4_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("SARAGOSSA_RUST_LINEAR_CONV_K4", true))
}

pub(super) fn linear_norm_dk128_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("SARAGOSSA_RUST_LINEAR_NORM_DK128", true))
}

impl MetalExecutor {
    pub(super) fn linear_conv_norm_fused_enabled(&self) -> bool {
        #[cfg(feature = "devtools")]
        if let Some(case) = crate::runtime_flags::pairwise::current() {
            return case.p3;
        }
        self.kernel_profile.linear_conv_norm_fused
    }
}

pub(super) fn fused_gate_up_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("SARAGOSSA_RUST_FUSED_GATE_UP", true))
}

/// Active la fusion gate+up+swiglu du **shared-expert** (tranche 3).
pub(super) fn fused_shared_gate_up_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("SARAGOSSA_RUST_FUSED_SHARED_GATE_UP", true))
}

pub(super) fn moe_shared_route_overlap_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("SARAGOSSA_RUST_MOE_SHARED_ROUTE_OVERLAP", true))
}

/// Top-k large (un thread par logit) au-delà de 256 experts ; repli sur le
/// kernel parallèle avec `SARAGOSSA_RUST_TOPK_WIDE=0`. Il prime sur
/// `SARAGOSSA_RUST_TOPK_PARALLEL=0` : même résultat, bit à bit, que le kernel parallèle.
pub(super) fn topk_wide_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("SARAGOSSA_RUST_TOPK_WIDE", true))
}

pub(super) fn topk_parallel_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("SARAGOSSA_RUST_TOPK_PARALLEL", true))
}

pub(super) fn scratch_resource_options() -> MTLResourceOptions {
    if private_scratch_enabled() {
        MTLResourceOptions::StorageModePrivate
    } else {
        MTLResourceOptions::StorageModeShared
    }
}

pub(super) fn private_scratch_enabled() -> bool {
    #[cfg(test)]
    if let Some(enabled) = PRIVATE_SCRATCH_OVERRIDE.with(Cell::get) {
        return enabled;
    }
    static PRIVATE: OnceLock<bool> = OnceLock::new();
    *PRIVATE.get_or_init(|| env_flag("SARAGOSSA_RUST_PRIVATE_SCRATCH", false))
}

#[cfg(test)]
thread_local! {
    static PRIVATE_SCRATCH_OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };
}

/// Isole le mode de stockage du test sans modifier l'environnement ni son OnceLock.
#[cfg(test)]
pub(super) fn with_private_scratch_override<R>(enabled: bool, run: impl FnOnce() -> R) -> R {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            PRIVATE_SCRATCH_OVERRIDE.with(|value| value.set(self.0));
        }
    }
    let _restore = Restore(PRIVATE_SCRATCH_OVERRIDE.with(|value| value.replace(Some(enabled))));
    run()
}

pub(super) fn fast_gather_qmv_enabled(weight: &StackedAffineBuffers) -> bool {
    match gather_fast_mode() {
        Some("0") | Some("false") | Some("off") => false,
        Some("gateup") => weight.out_dim <= weight.in_dim,
        Some("down") => weight.out_dim > weight.in_dim,
        _ => true,
    }
}
