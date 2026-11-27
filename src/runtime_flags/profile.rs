//! Profil TOML optionnel : valeurs de repli pour les réglages de `serve`.
//!
//! Priorité par clé : CLI > variable d'environnement > profil > défaut. Le
//! profil ne remplace jamais une variable d'environnement : [`lookup`] n'est
//! consulté par `env_var` qu'après l'environnement et son alias hérité (une
//! variable définie mais vide l'emporte donc sur le profil). Sans profil
//! installé, rien ne change.
//!
//! Couches ajoutées : `extends = "preset"` (un preset embarqué, voir
//! [`PRESETS`]) puis `--set section.cle=valeur`, qui a la priorité de la CLI :
//! elle l'emporte aussi sur l'environnement ([`forced`]).

use std::collections::HashMap;
use std::path::Path;
use std::sync::OnceLock;

use serde::Deserialize;
use toml::{Table, Value};

/// Presets embarqués : mêmes clés que les profils, aucune valeur nouvelle.
///
/// NOTE: valeurs non mesurées, des points de départ dérivés des défauts
/// (`serve` : 64 connexions, file 32, pool 2, cache de préfixe 16 Gio).
pub const PRESETS: [(&str, &str); 2] = [
    (
        "low-mem",
        "[serve]\nmax_connections = 16\nqueue = 8\nmax_streams = 2\nmodel_pool = 1\n\
         max_inflight_bytes = 67108864\nprefix_cache_bytes = 2147483648\n\
         [enhance]\nmax_streams = 1\n\
         [memory]\nheadroom_bytes = 4294967296\n",
    ),
    (
        "throughput",
        "[serve]\nmax_connections = 128\nqueue = 64\nmodel_pool = 4\n\
         max_inflight_bytes = 1073741824\nprefix_cache_bytes = 34359738368\n\
         [enhance]\nmax_streams = 4\n",
    ),
];

/// Clés qu'un rechargement SIGHUP peut appliquer sans redémarrage.
pub const HOT_KEYS: [&str; 5] = [
    "serve.read_timeout_secs",
    "serve.max_tokens_cap",
    "serve.max_tok_s",
    "serve.gpu_share",
    "serve.idle_unload_secs",
];

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

/// Entrée `[[models]]` : équivalent de `--model ID=CHEMIN`.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ModelEntry {
    /// Identifiant exposé sur `/v1/models`.
    pub id: String,
    /// Répertoire local du modèle (ou référence HF au lancement seulement).
    pub path: String,
}

/// Section `[serve]`.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
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
    /// Équivalent de `--max-tok-s` : plafond de decode en tokens par seconde.
    pub max_tok_s: Option<f64>,
    /// Équivalent de `--gpu-share` : part de GPU et de CPU visée, 10 à 100.
    pub gpu_share: Option<u64>,
    /// Équivalent de `--idle-unload-secs` : décharge les modèles après ce délai
    /// sans requête (0 ou absent : jamais).
    pub idle_unload_secs: Option<u64>,
    /// Équivalent de `--auto-yield` : baisse `gpu_share` tant que l'utilisateur
    /// se sert du Mac (clavier, souris).
    pub auto_yield: Option<bool>,
    /// Part de GPU pendant le cédage automatique, 10 à 100 (défaut 25) ; en
    /// mode dynamique, part de départ et de repli si la mesure GPU échoue.
    pub auto_yield_share: Option<u64>,
    /// Part de cédage ajustée entre 10 et 50 % selon l'activité GPU des autres
    /// processus (défaut `true`) ; `false` garde la part fixe `auto_yield_share`.
    pub auto_yield_dynamic: Option<bool>,
    /// Secondes d'inactivité avant de rendre la part configurée (défaut 60).
    pub auto_yield_idle_secs: Option<u64>,
    /// Délai max de vidage de l'ancien processus après un relais SIGUSR2.
    pub drain_timeout_secs: Option<u64>,
    /// Relais SIGUSR2 : `auto` (défaut), `parallel` ou `sequential`.
    pub relay_mode: Option<String>,
    /// Équivalent de `--engines` : moteurs à poids partagés, 1 à 16.
    pub engines: Option<u64>,
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
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Profile {
    /// Preset embarqué dont ce profil surcharge les valeurs.
    pub extends: Option<String>,
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
    /// Modèles servis, rechargeables à chaud par SIGHUP.
    #[serde(default)]
    pub models: Vec<ModelEntry>,
}

/// Une clé de profil avec son origine d'environnement et son défaut lecteur.
struct Row {
    key: &'static str,
    env: Option<&'static str>,
    default: &'static str,
    value: Option<String>,
}

impl Profile {
    /// Parse et valide un profil, `extends` résolu ; sans `--set`.
    pub fn parse(text: &str) -> Result<Self, ProfileError> {
        Ok(Resolved::from_text(text, "TOML", &[])?.profile)
    }

    /// Lit et valide le fichier `path`.
    pub fn load(path: &Path) -> Result<Self, ProfileError> {
        Ok(Resolved::resolve(Some(&path.display().to_string()), &[])?.profile)
    }

    fn validate(self) -> Result<Self, ProfileError> {
        let positive = [
            ("serve.max_connections", self.serve.max_connections),
            ("serve.queue", self.serve.queue),
            ("serve.max_streams", self.serve.max_streams),
            ("serve.model_pool", self.serve.model_pool),
            ("serve.max_inflight_bytes", self.serve.max_inflight_bytes),
            ("serve.read_timeout_secs", self.serve.read_timeout_secs),
            ("serve.max_tokens_cap", self.serve.max_tokens_cap),
            ("serve.drain_timeout_secs", self.serve.drain_timeout_secs),
            ("enhance.max_streams", self.enhance.max_streams),
            ("enhance.idle_timeout_secs", self.enhance.idle_timeout_secs),
            ("memory.cap_bytes", self.memory.cap_bytes),
        ];
        if let Some((key, _)) = positive.iter().find(|(_, v)| *v == Some(0)) {
            return Err(err(key, "doit être strictement positif"));
        }
        for (i, model) in self.models.iter().enumerate() {
            if model.id.trim().is_empty() || model.path.trim().is_empty() {
                return Err(err("models", "id et path requis"));
            }
            if self.models[..i].iter().any(|m| m.id == model.id) {
                return Err(err("models", format!("id en double: {}", model.id)));
            }
        }
        if !self
            .serve
            .max_tok_s
            .is_none_or(|v| v.is_finite() && v > 0.0)
        {
            return Err(err("serve.max_tok_s", "doit être un flottant > 0"));
        }
        if !self.serve.engines.is_none_or(|v| (1..=16).contains(&v)) {
            return Err(err("serve.engines", "doit être dans 1..=16"));
        }
        if !self.serve.gpu_share.is_none_or(|v| (10..=100).contains(&v)) {
            return Err(err("serve.gpu_share", "doit être dans 10..=100"));
        }
        if !self
            .serve
            .auto_yield_share
            .is_none_or(|v| (10..=100).contains(&v))
        {
            return Err(err("serve.auto_yield_share", "doit être dans 10..=100"));
        }
        // En deçà, le seuil d'activité de 5 s du cédage l'emporterait.
        if self.serve.auto_yield_idle_secs.is_some_and(|v| v < 5) {
            return Err(err("serve.auto_yield_idle_secs", "doit être >= 5"));
        }
        if let Some(backend) = &self.engine.backend {
            if !matches!(backend.as_str(), "cpu" | "metal") {
                return Err(err("engine.backend", "attendu \"cpu\" ou \"metal\""));
            }
        }
        if let Some(mode) = &self.serve.relay_mode {
            if !matches!(mode.as_str(), "auto" | "parallel" | "sequential") {
                return Err(err(
                    "serve.relay_mode",
                    "attendu \"auto\", \"parallel\" ou \"sequential\"",
                ));
            }
        }
        Ok(self)
    }

    /// Toutes les clés, dans l'ordre d'affichage.
    fn rows(&self) -> Vec<Row> {
        let (s, e, m) = (&self.serve, &self.enhance, &self.memory);
        let num = |v: Option<u64>| v.map(|v| v.to_string());
        let r = |key, env, default, value| Row {
            key,
            env,
            default,
            value,
        };
        vec![
            r(
                "engine.backend",
                None,
                "metal si disponible, sinon cpu",
                self.engine.backend.clone(),
            ),
            r(
                "engine.preload",
                None,
                "false",
                self.engine.preload.map(|v| v.to_string()),
            ),
            r(
                "serve.max_connections",
                Some("SARAGOSSA_SERVE_MAX_CONNECTIONS"),
                "64",
                num(s.max_connections),
            ),
            r(
                "serve.queue",
                Some("SARAGOSSA_SERVE_QUEUE"),
                "32",
                num(s.queue),
            ),
            r(
                "serve.max_streams",
                Some("SARAGOSSA_SERVE_MAX_STREAMS"),
                "8",
                num(s.max_streams),
            ),
            r(
                "serve.model_pool",
                Some("SARAGOSSA_SERVE_MODEL_POOL"),
                "2",
                num(s.model_pool),
            ),
            r(
                "serve.max_inflight_bytes",
                Some("SARAGOSSA_SERVE_MAX_INFLIGHT_BYTES"),
                "268435456",
                num(s.max_inflight_bytes),
            ),
            r(
                "serve.prefix_cache_bytes",
                Some("SARAGOSSA_SERVE_PREFIX_CACHE_BYTES"),
                "17179869184",
                num(s.prefix_cache_bytes),
            ),
            r(
                "serve.read_timeout_secs",
                None,
                "30",
                num(s.read_timeout_secs),
            ),
            r("serve.max_tokens_cap", None, "4096", num(s.max_tokens_cap)),
            r(
                "serve.max_tok_s",
                Some("SARAGOSSA_RUST_MAX_TOK_S"),
                "illimité",
                s.max_tok_s.map(|v| v.to_string()),
            ),
            r("serve.gpu_share", None, "100", num(s.gpu_share)),
            r(
                "serve.idle_unload_secs",
                Some("SARAGOSSA_RUST_IDLE_UNLOAD_SECS"),
                "0",
                num(s.idle_unload_secs),
            ),
            r(
                "serve.auto_yield",
                Some("SARAGOSSA_SERVE_AUTO_YIELD"),
                "false",
                s.auto_yield.map(|v| v.to_string()),
            ),
            r(
                "serve.auto_yield_share",
                None,
                "25",
                num(s.auto_yield_share),
            ),
            r(
                "serve.auto_yield_dynamic",
                Some("SARAGOSSA_SERVE_AUTO_YIELD_DYNAMIC"),
                "true",
                s.auto_yield_dynamic.map(|v| v.to_string()),
            ),
            r(
                "serve.auto_yield_idle_secs",
                None,
                "60",
                num(s.auto_yield_idle_secs),
            ),
            r(
                "serve.drain_timeout_secs",
                None,
                "30",
                num(s.drain_timeout_secs),
            ),
            r("serve.relay_mode", None, "auto", s.relay_mode.clone()),
            r(
                "serve.engines",
                Some("SARAGOSSA_SERVE_ENGINES"),
                "1",
                num(s.engines),
            ),
            r(
                "enhance.max_streams",
                Some("SARAGOSSA_ENHANCE_MAX_STREAMS"),
                "2",
                num(e.max_streams),
            ),
            r(
                "enhance.idle_timeout_secs",
                Some("SARAGOSSA_ENHANCE_IDLE_TIMEOUT_SECS"),
                "30",
                num(e.idle_timeout_secs),
            ),
            r(
                "memory.cap_bytes",
                Some("SARAGOSSA_SERVE_MEMORY_CAP_BYTES"),
                "détection automatique",
                num(m.cap_bytes),
            ),
            r(
                "memory.headroom_bytes",
                Some("SARAGOSSA_SERVE_MEMORY_HEADROOM_BYTES"),
                "2147483648",
                num(m.headroom_bytes),
            ),
        ]
    }

    /// Réglages lus via `env_var`, sous leur nom canonique.
    fn env_entries(&self) -> HashMap<&'static str, String> {
        self.rows()
            .into_iter()
            .filter_map(|r| Some((r.env?, r.value?)))
            .collect()
    }

    /// Clés dont la valeur diffère de `other`.
    pub fn changed_keys(&self, other: &Self) -> Vec<&'static str> {
        self.rows()
            .into_iter()
            .zip(other.rows())
            .filter(|(a, b)| a.value != b.value)
            .map(|(a, _)| a.key)
            .collect()
    }

    /// Installe les réglages lus par `env_var` ; à appeler avant tout lecteur
    /// à `OnceLock`. Renvoie faux si un profil était déjà installé.
    pub fn install(&self) -> bool {
        INSTALLED.set(self.env_entries()).is_ok()
    }
}

/// Profil résolu : preset, fichier et `--set` fusionnés, avec leur provenance.
pub struct Resolved {
    /// Valeurs effectives de la couche profil (preset < fichier < `--set`).
    pub profile: Profile,
    /// Valeurs venues de `--set` seulement.
    forced: Profile,
    file: Table,
    set: Table,
    preset: Option<(&'static str, Table)>,
    label: String,
}

fn merge(base: &mut Table, over: Table) {
    for (k, v) in over {
        match (base.get_mut(&k), v) {
            (Some(Value::Table(b)), Value::Table(o)) => merge(b, o),
            (_, v) => {
                base.insert(k, v);
            }
        }
    }
}

fn has(table: &Table, key: &str) -> bool {
    key.split_once('.')
        .and_then(|(s, k)| table.get(s)?.as_table()?.get(k))
        .is_some()
}

fn from_table(table: Table, ctx: &str) -> Result<Profile, ProfileError> {
    table
        .try_into::<Profile>()
        .map_err(|e| err(ctx, e))?
        .validate()
}

impl Resolved {
    /// Résout `source` (chemin de fichier ou nom de preset ; `None` = vide)
    /// puis applique les `--set section.cle=valeur`.
    pub fn resolve(source: Option<&str>, sets: &[String]) -> Result<Self, ProfileError> {
        match source {
            None => Self::from_text("", "défaut", sets),
            Some(name) if PRESETS.iter().any(|(n, _)| *n == name) => {
                Self::from_text(&format!("extends = \"{name}\"\n"), name, sets)
            }
            Some(path) => {
                let text = std::fs::read_to_string(path).map_err(|e| err(path, e))?;
                Self::from_text(&text, path, sets)
                    .map_err(|e| err(path, format!("{}: {}", e.context, e.message)))
            }
        }
    }

    fn from_text(text: &str, label: &str, sets: &[String]) -> Result<Self, ProfileError> {
        // Premier passage sur le texte brut : les erreurs de syntaxe et de
        // schéma portent ainsi ligne et colonne.
        let typed: Profile = toml::from_str(text).map_err(|e| err("TOML", e.to_string()))?;
        let mut file: Table = text.parse().map_err(|e: toml::de::Error| err("TOML", e))?;
        file.remove("extends");
        let mut merged = Table::new();
        let mut preset = None;
        if let Some(name) = &typed.extends {
            let (n, body) = PRESETS.iter().find(|(n, _)| n == name).ok_or_else(|| {
                let known: Vec<_> = PRESETS.iter().map(|(n, _)| *n).collect();
                err(
                    "extends",
                    format!("preset inconnu {name:?}, attendu {known:?}"),
                )
            })?;
            let table: Table = body.parse().map_err(|e: toml::de::Error| err(n, e))?;
            merged = table.clone();
            preset = Some((*n, table));
        }
        merge(&mut merged, file.clone());
        let mut set = Table::new();
        for item in sets {
            let (key, value) = item
                .split_once('=')
                .ok_or_else(|| err("--set", format!("{item:?}: attendu section.cle=valeur")))?;
            let (section, name) = key
                .trim()
                .split_once('.')
                .ok_or_else(|| err("--set", format!("{key:?}: attendu section.cle")))?;
            // Valeur TOML (nombre, booléen, chaîne entre guillemets), sinon texte brut.
            let value = format!("v = {}", value.trim())
                .parse::<Table>()
                .ok()
                .and_then(|mut t| t.remove("v"))
                .unwrap_or_else(|| Value::String(value.trim().to_string()));
            let entry = set
                .entry(section.to_string())
                .or_insert_with(|| Value::Table(Table::new()));
            if let Value::Table(t) = entry {
                t.insert(name.to_string(), value);
            }
        }
        let forced = from_table(set.clone(), "--set")?;
        merge(&mut merged, set.clone());
        Ok(Self {
            profile: from_table(merged, label)?,
            forced,
            file,
            set,
            preset,
            label: label.to_string(),
        })
    }

    /// Installe le profil et la couche `--set`.
    pub fn install(&self) -> bool {
        let _ = FORCED.set(self.forced.env_entries());
        self.profile.install()
    }

    /// Configuration effective : une ligne `cle = valeur  # origine` par clé.
    ///
    /// L'environnement est lu ici sous son nom canonique ; les options CLI
    /// dédiées de `serve` (`--max-tokens-cap`...) ne sont pas connues.
    pub fn report(&self) -> String {
        let mut out = String::new();
        for row in self.profile.rows() {
            // Nom canonique puis alias hérité ; vide ou invalide = ignoré ici.
            let valid = |v: &String| {
                if row.default == "false" {
                    return super::env_flag_value(v).is_some();
                }
                v.parse::<u64>()
                    .is_ok_and(|n| n > 0 || row.key == "memory.headroom_bytes")
            };
            let from_env = row.env.and_then(|n| {
                let legacy = n.replacen("SARAGOSSA_", "RETI_", 1);
                [n.to_string(), legacy]
                    .into_iter()
                    .find_map(|name| Some((name.clone(), std::env::var(&name).ok().filter(valid)?)))
            });
            let (value, origin) = if has(&self.set, row.key) {
                (row.value.clone(), "--set".to_string())
            } else if let Some((name, v)) = from_env {
                (Some(v), format!("env {name}"))
            } else if has(&self.file, row.key) {
                (row.value.clone(), format!("profil {}", self.label))
            } else if let Some((name, _)) = self.preset.iter().find(|(_, t)| has(t, row.key)) {
                (row.value.clone(), format!("preset {name}"))
            } else {
                (None, "défaut".to_string())
            };
            let shown = value.unwrap_or_else(|| row.default.to_string());
            out.push_str(&format!("{} = {shown}  # {origin}\n", row.key));
        }
        out
    }
}

static INSTALLED: OnceLock<HashMap<&'static str, String>> = OnceLock::new();
static FORCED: OnceLock<HashMap<&'static str, String>> = OnceLock::new();

/// Valeur imposée par `--set` : prioritaire sur l'environnement.
pub(super) fn forced(name: &str) -> Option<String> {
    FORCED.get()?.get(name).cloned()
}

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
    fn presets_parse_and_file_overrides_them() {
        for (name, _) in PRESETS {
            let r = Resolved::resolve(Some(name), &[]).expect(name);
            assert!(r.profile.serve.queue.is_some(), "{name}");
        }
        let p = Profile::parse("extends = \"low-mem\"\n[serve]\nqueue = 99\n").expect("extends");
        assert_eq!(p.serve.queue, Some(99));
        assert_eq!(p.serve.model_pool, Some(1), "valeur du preset conservée");
        assert!(Profile::parse("extends = \"inconnu\"\n").is_err());
    }

    #[test]
    fn set_layer_wins_and_is_validated() {
        let sets = [
            "serve.queue=5".to_string(),
            "engine.backend=cpu".to_string(),
        ];
        let r = Resolved::from_text(
            "extends = \"low-mem\"\n[serve]\nqueue = 99\n",
            "f.toml",
            &sets,
        )
        .expect("set");
        assert_eq!(r.profile.serve.queue, Some(5));
        assert_eq!(r.profile.engine.backend.as_deref(), Some("cpu"));
        assert_eq!(r.forced.env_entries()["SARAGOSSA_SERVE_QUEUE"], "5");
        for bad in [
            "serve.queue=0",
            "serve.nope=1",
            "queue=1",
            "serve.queue",
            "serve.queue=\"x\"",
        ] {
            assert!(
                Resolved::from_text("", "f", &[bad.to_string()]).is_err(),
                "{bad}"
            );
        }
    }

    #[test]
    fn report_shows_origin_and_syntax_errors_have_line_column() {
        let sets = ["serve.max_tokens_cap=7".to_string()];
        let r = Resolved::from_text(
            "extends = \"low-mem\"\n[serve]\nread_timeout_secs = 9\n",
            "f.toml",
            &sets,
        )
        .expect("profil");
        let report = r.report();
        assert!(
            report.contains("serve.max_tokens_cap = 7  # --set"),
            "{report}"
        );
        assert!(
            report.contains("serve.read_timeout_secs = 9  # profil f.toml"),
            "{report}"
        );
        assert!(
            report.contains("serve.model_pool = 1  # preset low-mem"),
            "{report}"
        );
        assert!(
            report.contains("serve.max_streams = 2  # preset low-mem"),
            "{report}"
        );
        assert!(
            report.contains("enhance.idle_timeout_secs = 30  # défaut"),
            "{report}"
        );
        let e = Profile::parse("[serve]\nqueue = 1\nnope = 2\n")
            .expect_err("schéma")
            .to_string();
        assert!(e.contains("line 3, column 1"), "{e}");
    }

    #[test]
    fn load_reports_missing_file() {
        let e = Profile::load(Path::new("/nonexistent/saragossa.toml")).expect_err("absent");
        assert!(e.to_string().contains("/nonexistent/saragossa.toml"));
    }
}
