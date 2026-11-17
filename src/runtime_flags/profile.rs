//! Profil TOML optionnel : valeurs de repli pour les réglages de `serve`.
//!
//! Priorité par clé : CLI > variable d'environnement > profil > défaut. Le
//! profil ne remplace jamais une variable d'environnement : [`lookup`] n'est
//! consulté par `env_var` qu'après l'environnement et son alias hérité (une
//! variable définie mais vide l'emporte donc sur le profil). Sans profil
//! installé, rien ne change.

use std::collections::HashMap;
use std::path::Path;
use std::sync::OnceLock;

use serde::Deserialize;

/// Erreur de lecture ou de validation d'un profil.
#[derive(Debug, thiserror::Error)]
#[error("profil {context}: {message}")]
pub struct ProfileError {
    context: String,
    message: String,
}

fn err(context: &str, message: impl ToString) -> ProfileError {
    ProfileError {
        context: context.to_string(),
        message: message.to_string(),
    }
}

/// Section `[engine]` : options aussi exposées en CLI.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct EngineSection {
    /// `cpu` ou `metal` (équivalent de `--backend`).
    pub backend: Option<String>,
    /// Charge les modèles au démarrage (équivalent de `--preload`).
    pub preload: Option<bool>,
}

/// Section `[serve]`.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ServeSection {
    /// `SARAGOSSA_SERVE_MAX_CONNECTIONS`.
    pub max_connections: Option<u64>,
    /// `SARAGOSSA_SERVE_QUEUE`.
    pub queue: Option<u64>,
    /// `SARAGOSSA_SERVE_MAX_STREAMS`.
    pub max_streams: Option<u64>,
    /// `SARAGOSSA_SERVE_MODEL_POOL`.
    pub model_pool: Option<u64>,
    /// `SARAGOSSA_SERVE_MAX_INFLIGHT_BYTES`.
    pub max_inflight_bytes: Option<u64>,
    /// `SARAGOSSA_SERVE_PREFIX_CACHE_BYTES`.
    pub prefix_cache_bytes: Option<u64>,
    /// Équivalent de `--read-timeout-secs`.
    pub read_timeout_secs: Option<u64>,
    /// Équivalent de `--max-tokens-cap`.
    pub max_tokens_cap: Option<u64>,
}

/// Section `[enhance]`.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct EnhanceSection {
    /// `SARAGOSSA_ENHANCE_MAX_STREAMS`.
    pub max_streams: Option<u64>,
    /// `SARAGOSSA_ENHANCE_IDLE_TIMEOUT_SECS`.
    pub idle_timeout_secs: Option<u64>,
}

/// Section `[memory]`.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MemorySection {
    /// `SARAGOSSA_SERVE_MEMORY_CAP_BYTES`.
    pub cap_bytes: Option<u64>,
    /// `SARAGOSSA_SERVE_MEMORY_HEADROOM_BYTES`.
    pub headroom_bytes: Option<u64>,
}

/// Fichier de profil ; toute clé inconnue est refusée.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Profile {
    /// Backend et préchargement.
    #[serde(default)]
    pub engine: EngineSection,
    /// Limites et caches de `serve`.
    #[serde(default)]
    pub serve: ServeSection,
    /// Limites du débruitage DFN3.
    #[serde(default)]
    pub enhance: EnhanceSection,
    /// Garde mémoire globale.
    #[serde(default)]
    pub memory: MemorySection,
}

impl Profile {
    /// Parse et valide un profil ; les bornes reprennent celles des lecteurs.
    pub fn parse(text: &str) -> Result<Self, ProfileError> {
        let profile: Self = toml::from_str(text).map_err(|e| err("TOML", e.to_string()))?;
        let positive = [
            ("serve.max_connections", profile.serve.max_connections),
            ("serve.queue", profile.serve.queue),
            ("serve.max_streams", profile.serve.max_streams),
            ("serve.model_pool", profile.serve.model_pool),
            ("serve.max_inflight_bytes", profile.serve.max_inflight_bytes),
            ("serve.read_timeout_secs", profile.serve.read_timeout_secs),
            ("serve.max_tokens_cap", profile.serve.max_tokens_cap),
            ("enhance.max_streams", profile.enhance.max_streams),
            (
                "enhance.idle_timeout_secs",
                profile.enhance.idle_timeout_secs,
            ),
            ("memory.cap_bytes", profile.memory.cap_bytes),
        ];
        if let Some((key, _)) = positive.iter().find(|(_, v)| *v == Some(0)) {
            return Err(err(key, "doit être strictement positif"));
        }
        if let Some(backend) = &profile.engine.backend {
            if !matches!(backend.as_str(), "cpu" | "metal") {
                return Err(err("engine.backend", "attendu \"cpu\" ou \"metal\""));
            }
        }
        Ok(profile)
    }

    /// Lit et valide le fichier `path`.
    pub fn load(path: &Path) -> Result<Self, ProfileError> {
        let ctx = path.display().to_string();
        let text = std::fs::read_to_string(path).map_err(|e| err(&ctx, e))?;
        Self::parse(&text).map_err(|e| err(&ctx, format!("{}: {}", e.context, e.message)))
    }

    /// Réglages lus via `env_var`, sous leur nom canonique.
    fn env_entries(&self) -> HashMap<&'static str, String> {
        let s = &self.serve;
        let rows = [
            ("SARAGOSSA_SERVE_MAX_CONNECTIONS", s.max_connections),
            ("SARAGOSSA_SERVE_QUEUE", s.queue),
            ("SARAGOSSA_SERVE_MAX_STREAMS", s.max_streams),
            ("SARAGOSSA_SERVE_MODEL_POOL", s.model_pool),
            ("SARAGOSSA_SERVE_MAX_INFLIGHT_BYTES", s.max_inflight_bytes),
            ("SARAGOSSA_SERVE_PREFIX_CACHE_BYTES", s.prefix_cache_bytes),
            ("SARAGOSSA_ENHANCE_MAX_STREAMS", self.enhance.max_streams),
            (
                "SARAGOSSA_ENHANCE_IDLE_TIMEOUT_SECS",
                self.enhance.idle_timeout_secs,
            ),
            ("SARAGOSSA_SERVE_MEMORY_CAP_BYTES", self.memory.cap_bytes),
            (
                "SARAGOSSA_SERVE_MEMORY_HEADROOM_BYTES",
                self.memory.headroom_bytes,
            ),
        ];
        rows.into_iter()
            .filter_map(|(name, v)| v.map(|v| (name, v.to_string())))
            .collect()
    }

    /// Installe les réglages lus par `env_var` ; à appeler avant tout lecteur
    /// à `OnceLock`. Renvoie faux si un profil était déjà installé.
    pub fn install(&self) -> bool {
        INSTALLED.set(self.env_entries()).is_ok()
    }
}

static INSTALLED: OnceLock<HashMap<&'static str, String>> = OnceLock::new();

/// Valeur du profil pour `name`, si un profil est installé et la définit.
pub(super) fn lookup(name: &str) -> Option<String> {
    INSTALLED.get()?.get(name).cloned()
}

#[cfg(test)]
mod tests {
    use super::*;

    const FULL: &str = "[engine]\nbackend = \"cpu\"\npreload = true\n\
        [serve]\nmax_connections = 7\nqueue = 3\nread_timeout_secs = 9\n\
        [enhance]\nmax_streams = 4\nidle_timeout_secs = 12\n\
        [memory]\ncap_bytes = 100\nheadroom_bytes = 0\n";

    #[test]
    fn parses_short_keys() {
        let p = Profile::parse(FULL).expect("profil valide");
        assert_eq!(p.engine.backend.as_deref(), Some("cpu"));
        assert_eq!(p.engine.preload, Some(true));
        assert_eq!(p.serve.read_timeout_secs, Some(9));
        let env = p.env_entries();
        assert_eq!(env["SARAGOSSA_ENHANCE_MAX_STREAMS"], "4");
        assert_eq!(env["SARAGOSSA_SERVE_QUEUE"], "3");
        assert_eq!(env["SARAGOSSA_SERVE_MEMORY_HEADROOM_BYTES"], "0");
        assert!(!env.contains_key("SARAGOSSA_SERVE_MODEL_POOL"));
    }

    #[test]
    fn empty_profile_changes_nothing() {
        let p = Profile::parse("").expect("vide");
        assert_eq!(p, Profile::default());
        assert!(p.env_entries().is_empty());
    }

    #[test]
    fn rejects_unknown_keys_and_bad_values() {
        for bad in [
            "[enhance]\nmax_stream = 2\n",
            "[inconnu]\nx = 1\n",
            "[enhance]\nmax_streams = 0\n",
            "[enhance]\nmax_streams = \"2\"\n",
            "[engine]\nbackend = \"cuda\"\n",
            "[serve]\nread_timeout_secs = 0\n",
        ] {
            assert!(Profile::parse(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn load_reports_missing_file() {
        let e = Profile::load(Path::new("/nonexistent/saragossa.toml")).expect_err("absent");
        assert!(e.to_string().contains("/nonexistent/saragossa.toml"));
    }
}
