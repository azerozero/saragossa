//! Presets runtime derives depuis le modele charge.

use std::path::Path;
use std::sync::OnceLock;

mod kernel_profile;
pub use kernel_profile::{initialize_kernel_environment, KernelProfile};

/// Preset runtime connu pour un checkpoint local.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RuntimePreset {
    /// Nom stable du preset.
    pub name: &'static str,
    /// Valeur recommandee de `SARAGOSSA_RUST_OPT_PROFILE`.
    pub opt_profile: &'static str,
    /// Top-p de sampling recommande pour les chemins chat.
    pub sampling_top_p: f32,
    /// Top-k de sampling recommande pour les chemins chat.
    pub sampling_top_k: usize,
}

const QWEN36_OQ8: RuntimePreset = RuntimePreset {
    name: "qwen36-oq8",
    opt_profile: "qwen36-oq8",
    sampling_top_p: 0.95,
    sampling_top_k: 20,
};

// Le premier preset reconnu reste le défaut des anciens consommateurs.
// Les profils résolus par modèle ne consultent jamais cet état.
static LEGACY_PRESET: OnceLock<RuntimePreset> = OnceLock::new();

/// Detecte le preset runtime depuis le chemin du modele.
#[must_use]
pub fn runtime_preset_for_model_dir(model_dir: &Path) -> Option<RuntimePreset> {
    let key = model_dir.to_string_lossy().to_ascii_lowercase();
    if key.contains("qwen3.6-35b-a3b-oq8") || key.contains("qwen3_6-35b-a3b-oq8") {
        return Some(QWEN36_OQ8);
    }
    None
}

/// Applique le premier preset reconnu aux futurs appels à `MetalExecutor::new`.
///
/// L'état de compatibilité est partagé entre threads, sans modifier l'environnement
/// ni les executors existants. Les surcharges explicites restent prioritaires.
/// Les profils issus de [`KernelProfile::for_model_dir`] restent indépendants de
/// cet état : les préférer pour charger plusieurs modèles dans le même processus.
#[deprecated(
    note = "utiliser KernelProfile::for_model_dir avec MetalExecutor::new_with_kernel_profile pour isoler les modèles"
)]
#[must_use]
pub fn apply_runtime_preset_for_model_dir(model_dir: &Path) -> Option<RuntimePreset> {
    if crate::runtime_flags::env_present("SARAGOSSA_RUST_OPT_PROFILE") {
        return None;
    }
    let preset = runtime_preset_for_model_dir(model_dir)?;
    let _ = LEGACY_PRESET.get_or_init(|| preset);
    Some(preset)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_qwen36_oq8_local_model_dir() {
        let preset = runtime_preset_for_model_dir(Path::new("models/Qwen3.6-35B-A3B-oQ8"))
            .expect("invariant: preset qwen36 oQ8 detecte");

        assert_eq!(preset.name, "qwen36-oq8");
        assert_eq!(preset.opt_profile, "qwen36-oq8");
        assert!((preset.sampling_top_p - 0.95).abs() < f32::EPSILON);
        assert_eq!(preset.sampling_top_k, 20);
    }

    #[test]
    fn ignores_other_qwen_models() {
        assert!(runtime_preset_for_model_dir(Path::new("models/Qwen3.6-35B-A3B-4bit")).is_none());
    }
}
