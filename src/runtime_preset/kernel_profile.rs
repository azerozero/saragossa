//! Choix de kernels résolus sans device et conservés par chaque executor.

use std::path::Path;
use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum KernelOptProfile {
    Default,
    Qwen36Oq8,
    ExploreFused,
}

impl KernelOptProfile {
    fn parse(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "qwen36-oq8" | "qwen3.6-oq8" | "linear-bf16" | "bf16" => Self::Qwen36Oq8,
            "explore-fused" | "fused" => Self::ExploreFused,
            _ => Self::Default,
        }
    }
}

/// Profil immuable, incluant les surcharges et kill-switches du démarrage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KernelProfile {
    pub(crate) qmv_u8_tg128: bool,
    pub(crate) qmv_u8_tg256: bool,
    pub(crate) qmv_u8_dot4: bool,
    pub(crate) linear_pair_barrier_coalesce: bool,
    pub(crate) linear_inv_delta: bool,
    pub(crate) linear_ssm_bf16: bool,
    pub(crate) linear_conv_norm_fused: bool,
    pub(crate) fused_gate_up_u8: bool,
    pub(crate) fused_shared_gate_up_u8: bool,
    pub(crate) fused_shared_gate_scalar_u8: bool,
    pub(crate) fused_shared_gate_qmv_u8: bool,
    pub(crate) fused_moe_down_weighted_u8: bool,
    pub(crate) full_qkv_split_rms_u8: bool,
    pub(crate) full_o_proj_gated_u8: bool,
    delta_tg_rows: Option<u64>,
}

impl KernelProfile {
    /// Résout le preset du modèle avec priorité aux surcharges du démarrage.
    #[must_use]
    pub fn for_model_dir(model_dir: &Path) -> Self {
        environment().for_model_dir(model_dir)
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) fn for_legacy_executor() -> Self {
        environment().for_preset(super::LEGACY_PRESET.get().copied())
    }

    #[cfg(test)]
    pub(crate) fn without_overrides(model_dir: &Path) -> Self {
        KernelEnvironment::capture(|_| None, false).for_model_dir(model_dir)
    }

    fn resolve(profile: KernelOptProfile, read: &impl Fn(&str) -> Option<String>) -> Self {
        let defaults = u8_kernel_preset_defaults_for(profile);
        let flag = |name, default| {
            read(name)
                .and_then(|value| crate::runtime_flags::env_flag_value(&value))
                .unwrap_or(default)
        };
        Self {
            qmv_u8_tg128: flag(
                "SARAGOSSA_RUST_QMV_U8_TG128",
                profile_bool_default(profile, "SARAGOSSA_RUST_QMV_U8_TG128", false),
            ),
            qmv_u8_tg256: flag("SARAGOSSA_RUST_QMV_U8_TG256", defaults.qmv_tg256),
            qmv_u8_dot4: flag("SARAGOSSA_RUST_QMV_U8_DOT4", defaults.qmv_dot4),
            linear_pair_barrier_coalesce: flag(
                "SARAGOSSA_RUST_LINEAR_PAIR_BARRIER_COALESCE",
                profile_bool_default(
                    profile,
                    "SARAGOSSA_RUST_LINEAR_PAIR_BARRIER_COALESCE",
                    false,
                ),
            ),
            linear_inv_delta: flag(
                "SARAGOSSA_RUST_LINEAR_INV_DELTA",
                profile_bool_default(profile, "SARAGOSSA_RUST_LINEAR_INV_DELTA", false),
            ),
            linear_ssm_bf16: flag(
                "SARAGOSSA_RUST_LINEAR_SSM_BF16",
                profile_bool_default(profile, "SARAGOSSA_RUST_LINEAR_SSM_BF16", false),
            ),
            linear_conv_norm_fused: flag(
                "SARAGOSSA_RUST_LINEAR_CONV_NORM_FUSED",
                profile_bool_default(profile, "SARAGOSSA_RUST_LINEAR_CONV_NORM_FUSED", false),
            ),
            fused_gate_up_u8: flag("SARAGOSSA_RUST_FUSED_GATE_UP_U8", defaults.fused_gate_up),
            fused_shared_gate_up_u8: flag(
                "SARAGOSSA_RUST_FUSED_SHARED_GATE_UP_U8",
                defaults.fused_shared_gate_up,
            ),
            fused_shared_gate_scalar_u8: flag(
                "SARAGOSSA_RUST_FUSED_SHARED_GATE_SCALAR_U8",
                defaults.fused_shared_gate_scalar,
            ),
            fused_shared_gate_qmv_u8: flag(
                "SARAGOSSA_RUST_FUSED_SHARED_GATE_QMV_U8",
                defaults.fused_shared_gate_qmv,
            ),
            fused_moe_down_weighted_u8: flag(
                "SARAGOSSA_RUST_FUSED_MOE_DOWN_WEIGHTED_U8",
                defaults.fused_moe_down_weighted,
            ),
            full_qkv_split_rms_u8: flag(
                "SARAGOSSA_RUST_FULL_QKV_SPLIT_RMS_U8",
                defaults.full_qkv_split_rms,
            ),
            full_o_proj_gated_u8: flag(
                "SARAGOSSA_RUST_FULL_O_PROJ_GATED_U8",
                defaults.full_o_proj_gated,
            ),
            delta_tg_rows: read("SARAGOSSA_RUST_LINEAR_DELTA_TG_ROWS")
                .and_then(|value| value.parse::<u64>().ok())
                .filter(|value| matches!(*value, 1 | 2 | 4 | 8 | 16 | 32)),
        }
    }

    pub(crate) fn linear_delta_tg_rows(self, ssm_bf16: bool) -> u64 {
        self.delta_tg_rows.unwrap_or(if ssm_bf16 { 8 } else { 4 })
    }
}

struct KernelEnvironment {
    profiles: [KernelProfile; 3],
    explicit: Option<KernelOptProfile>,
}

impl KernelEnvironment {
    fn capture(read: impl Fn(&str) -> Option<String>, profile_present: bool) -> Self {
        Self {
            profiles: [
                KernelOptProfile::Default,
                KernelOptProfile::Qwen36Oq8,
                KernelOptProfile::ExploreFused,
            ]
            .map(|profile| KernelProfile::resolve(profile, &read)),
            // Une valeur vide, inconnue ou non UTF-8 bloque aussi le preset modèle.
            explicit: profile_present.then(|| {
                KernelOptProfile::parse(&read("SARAGOSSA_RUST_OPT_PROFILE").unwrap_or_default())
            }),
        }
    }

    fn for_model_dir(&self, model_dir: &Path) -> KernelProfile {
        self.for_preset(super::runtime_preset_for_model_dir(model_dir))
    }

    fn for_preset(&self, preset: Option<super::RuntimePreset>) -> KernelProfile {
        let profile = self.explicit.unwrap_or_else(|| {
            preset.map_or(KernelOptProfile::Default, |preset| {
                KernelOptProfile::parse(preset.opt_profile)
            })
        });
        self.profiles[profile as usize]
    }
}

fn environment() -> &'static KernelEnvironment {
    // Seules les surcharges sont globales : aucun modèle ne peut changer les suivantes.
    static ENVIRONMENT: OnceLock<KernelEnvironment> = OnceLock::new();
    ENVIRONMENT.get_or_init(|| {
        KernelEnvironment::capture(
            crate::runtime_flags::env_var,
            crate::runtime_flags::env_present("SARAGOSSA_RUST_OPT_PROFILE"),
        )
    })
}

/// Fige les surcharges de kernels avant de lancer les threads du moteur.
/// Les consommateurs de la bibliothèque peuvent appeler cette fonction au démarrage ;
/// à défaut, le premier profil résolu capture l'environnement, sans le modifier.
pub fn initialize_kernel_environment() {
    let _ = environment();
}

fn profile_bool_default(profile: KernelOptProfile, name: &str, default: bool) -> bool {
    match profile {
        KernelOptProfile::Default => default,
        KernelOptProfile::Qwen36Oq8 => match name {
            "SARAGOSSA_RUST_LINEAR_SSM_BF16" => true,
            "SARAGOSSA_RUST_LINEAR_INV_DELTA" => false,
            "SARAGOSSA_RUST_LINEAR_CONV_NORM_FUSED" => false,
            "SARAGOSSA_RUST_LINEAR_PAIR_BARRIER_COALESCE" => true,
            "SARAGOSSA_RUST_QMV_U8_TG128" => true,
            _ => default,
        },
        KernelOptProfile::ExploreFused => match name {
            "SARAGOSSA_RUST_LINEAR_SSM_BF16" => true,
            "SARAGOSSA_RUST_LINEAR_INV_DELTA" => false,
            "SARAGOSSA_RUST_LINEAR_CONV_NORM_FUSED" => true,
            "SARAGOSSA_RUST_QMV_U8_TG128" => true,
            _ => default,
        },
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    const MODEL_A: &str = "models/Qwen3.6-35B-A3B-oQ8";
    const MODEL_B: &str = "models/Qwen3.6-35B-A3B-4bit";

    #[test]
    fn model_profiles_are_independent_of_load_order() {
        let ab = KernelEnvironment::capture(|_| None, false);
        let a_first = ab.for_model_dir(Path::new(MODEL_A));
        let b_second = ab.for_model_dir(Path::new(MODEL_B));
        let ba = KernelEnvironment::capture(|_| None, false);
        let b_first = ba.for_model_dir(Path::new(MODEL_B));
        let a_second = ba.for_model_dir(Path::new(MODEL_A));
        assert_eq!(a_first, a_second);
        assert_eq!(b_first, b_second);
        assert_ne!(a_first, b_first);
        assert!(a_first.linear_ssm_bf16);
        assert!(a_first.qmv_u8_tg128);
        assert!(a_first.linear_pair_barrier_coalesce);
        assert!(a_first.fused_moe_down_weighted_u8);
        assert!(!b_first.linear_ssm_bf16);
        assert!(!b_first.qmv_u8_tg128);
        assert!(!b_first.linear_pair_barrier_coalesce);
        assert!(!b_first.fused_moe_down_weighted_u8);
        for profile in [a_first, b_first] {
            assert!(!profile.linear_inv_delta);
            assert!(!profile.linear_conv_norm_fused);
            assert_eq!(profile.linear_delta_tg_rows(false), 4);
            assert_eq!(profile.linear_delta_tg_rows(true), 8);
        }
        assert_eq!(
            a_first,
            KernelProfile::without_overrides(Path::new(MODEL_A))
        );
        assert_eq!(
            b_first,
            KernelProfile::without_overrides(Path::new(MODEL_B))
        );
    }

    #[test]
    fn explicit_profile_wins_over_both_models() {
        for (value, expected) in [
            ("default", KernelOptProfile::Default),
            ("", KernelOptProfile::Default),
            ("inconnu", KernelOptProfile::Default),
            ("qwen36-oq8", KernelOptProfile::Qwen36Oq8),
            ("qwen3.6-oq8", KernelOptProfile::Qwen36Oq8),
            ("linear-bf16", KernelOptProfile::Qwen36Oq8),
            (" BF16 ", KernelOptProfile::Qwen36Oq8),
            ("explore-fused", KernelOptProfile::ExploreFused),
            ("fused", KernelOptProfile::ExploreFused),
        ] {
            let env = KernelEnvironment::capture(
                |name| (name == "SARAGOSSA_RUST_OPT_PROFILE").then(|| value.to_string()),
                true,
            );
            for model in [MODEL_A, MODEL_B] {
                assert_eq!(
                    env.for_model_dir(Path::new(model)),
                    KernelProfile::resolve(expected, &|_| None)
                );
            }
        }
        // La présence non UTF-8 empêchait déjà l'application du preset automatique.
        let env = KernelEnvironment::capture(|_| None, true);
        assert_eq!(
            env.for_model_dir(Path::new(MODEL_A)),
            KernelProfile::resolve(KernelOptProfile::Default, &|_| None)
        );
    }

    #[test]
    fn every_kernel_override_and_kill_switch_wins_over_presets() {
        macro_rules! check {
            ($($field:ident => $name:literal),+ $(,)?) => {
                $(for value in [false, true] {
                    let read = |name: &str| (name == $name).then(|| value.to_string());
                    for preset in [KernelOptProfile::Default, KernelOptProfile::Qwen36Oq8, KernelOptProfile::ExploreFused] {
                        let actual = KernelProfile::resolve(preset, &read);
                        let mut expected = KernelProfile::resolve(preset, &|_| None);
                        expected.$field = value;
                        assert_eq!(actual, expected, "{}={value} {preset:?}", $name);
                        let invalid = KernelProfile::resolve(preset, &|name| (name == $name).then(|| "invalide".to_string()));
                        assert_eq!(invalid, KernelProfile::resolve(preset, &|_| None));
                    }
                })+
            };
        }
        check! {
            qmv_u8_tg128 => "SARAGOSSA_RUST_QMV_U8_TG128",
            qmv_u8_tg256 => "SARAGOSSA_RUST_QMV_U8_TG256",
            qmv_u8_dot4 => "SARAGOSSA_RUST_QMV_U8_DOT4",
            linear_pair_barrier_coalesce => "SARAGOSSA_RUST_LINEAR_PAIR_BARRIER_COALESCE",
            linear_inv_delta => "SARAGOSSA_RUST_LINEAR_INV_DELTA",
            linear_ssm_bf16 => "SARAGOSSA_RUST_LINEAR_SSM_BF16",
            linear_conv_norm_fused => "SARAGOSSA_RUST_LINEAR_CONV_NORM_FUSED",
            fused_gate_up_u8 => "SARAGOSSA_RUST_FUSED_GATE_UP_U8",
            fused_shared_gate_up_u8 => "SARAGOSSA_RUST_FUSED_SHARED_GATE_UP_U8",
            fused_shared_gate_scalar_u8 => "SARAGOSSA_RUST_FUSED_SHARED_GATE_SCALAR_U8",
            fused_shared_gate_qmv_u8 => "SARAGOSSA_RUST_FUSED_SHARED_GATE_QMV_U8",
            fused_moe_down_weighted_u8 => "SARAGOSSA_RUST_FUSED_MOE_DOWN_WEIGHTED_U8",
            full_qkv_split_rms_u8 => "SARAGOSSA_RUST_FULL_QKV_SPLIT_RMS_U8",
            full_o_proj_gated_u8 => "SARAGOSSA_RUST_FULL_O_PROJ_GATED_U8",
        }
    }

    #[test]
    fn delta_rows_override_preserves_validation_and_state_defaults() {
        for value in [
            "1", "2", "4", "8", "16", "32", "0", "3", "64", " 8 ", "invalide",
        ] {
            let env = KernelEnvironment::capture(
                |name| (name == "SARAGOSSA_RUST_LINEAR_DELTA_TG_ROWS").then(|| value.to_string()),
                false,
            );
            let expected = value
                .parse::<u64>()
                .ok()
                .filter(|v| matches!(*v, 1 | 2 | 4 | 8 | 16 | 32));
            for model in [MODEL_A, MODEL_B] {
                let profile = env.for_model_dir(Path::new(model));
                assert_eq!(profile.linear_delta_tg_rows(false), expected.unwrap_or(4));
                assert_eq!(profile.linear_delta_tg_rows(true), expected.unwrap_or(8));
            }
        }
    }

    #[test]
    fn captured_overrides_stay_fixed_between_model_loads() {
        let value = std::cell::Cell::new("0");
        let env = KernelEnvironment::capture(
            |name| (name == "SARAGOSSA_RUST_LINEAR_SSM_BF16").then(|| value.get().to_string()),
            false,
        );
        let a = env.for_model_dir(Path::new(MODEL_A));
        value.set("1");
        let b = env.for_model_dir(Path::new(MODEL_B));
        assert!(!a.linear_ssm_bf16);
        assert!(!b.linear_ssm_bf16);
        assert_eq!(a, env.for_model_dir(Path::new(MODEL_A)));
    }

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
