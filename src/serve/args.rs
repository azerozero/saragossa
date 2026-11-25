//! Parsing des options de `saragossa serve`.

use std::env;
use std::path::{Path, PathBuf};
use std::time::Duration;

use super::error::{ServeError, ServeResult};
use crate::{hf_resolve, RuntimeKind};
use saragossa::runtime_flags::{ModelEntry, Profile};

const DEFAULT_HOST: &str = "127.0.0.1";
const DEFAULT_PORT: u16 = 8081;
const DEFAULT_SOCKET: &str = "/tmp/saragossa-serve.sock";
const API_KEY_ENV: &str = "SARAGOSSA_API_KEY";
const DEFAULT_READ_TIMEOUT_SECS: u64 = 30;
const DEFAULT_MAX_TOKENS_CAP: usize = 4096;
/// Plafond de `--engines` : au-delà, la mémoire par moteur et le partage GPU
/// rendent le gain illusoire sur une seule machine.
const MAX_ENGINES: usize = 16;

/// Configuration de lancement du serveur.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct ServeArgs {
    /// Adresse TCP locale.
    pub(super) host: String,
    /// Port TCP local si activé.
    pub(super) port: Option<u16>,
    /// Socket Unix locale.
    pub(super) socket: PathBuf,
    /// Jeton bearer optionnel pour le transport TCP.
    pub(super) api_key: Option<String>,
    /// Registre des modèles exposés.
    pub(super) models: Vec<ServeModelConfig>,
    /// Répertoire du modèle STT Whisper exposé sur `/v1/audio/transcriptions`.
    pub(super) stt_model: Option<PathBuf>,
    /// Répertoire du modèle de diarisation exposé sur `/v1/audio/diarization`.
    pub(super) diar_model: Option<PathBuf>,
    /// Répertoire du modèle TTS Qwen3 exposé sur `/v1/audio/speech`.
    pub(super) tts_model: Option<PathBuf>,
    /// Répertoire du modèle d'embeddings e5-small exposé sur `/v1/embeddings`.
    pub(super) embed_model: Option<PathBuf>,
    /// Poids DeepFilterNet3 (`dfn3.safetensors`) exposés sur `/v1/audio/enhance`.
    pub(super) enhance_model: Option<PathBuf>,
    /// Backend d'inférence.
    pub(super) backend: RuntimeKind,
    /// Charge tous les modèles au démarrage.
    pub(super) preload: bool,
    /// Timeout de lecture par connexion acceptée.
    pub(super) read_timeout: Duration,
    /// Plafond dur du budget de génération par requête.
    pub(super) max_tokens_cap: usize,
    /// Plafond de decode en tokens par seconde, rechargeable à chaud.
    pub(super) max_tok_s: Option<f64>,
    /// Part de GPU et de CPU visée (1 à 100), rechargeable à chaud.
    pub(super) gpu_share: Option<u8>,
    /// Délai sans requête avant déchargement des modèles (0 : jamais), rechargeable à chaud.
    pub(super) idle_unload_secs: u64,
    /// Cédage automatique du GPU quand l'utilisateur se sert du Mac (`None` : off).
    pub(super) auto_yield: Option<super::auto_yield::AutoYield>,
    /// Cache persistant opt-in.
    pub(super) prefix_ssd: saragossa::runtime_flags::ServePrefixSsdOptions,
    /// Moteurs d'inférence parallèles partageant les poids (1 : moteur sériel).
    pub(super) engines: usize,
}

/// Modèle exposé par `saragossa serve`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ServeModelConfig {
    /// Identifiant OpenAI servi.
    pub(super) id: String,
    /// Répertoire local du modèle.
    pub(super) path: PathBuf,
    /// Identifiant HF nu à résoudre avant de lancer le serveur.
    pub(super) hf_repo: Option<String>,
}

// `parse_with` lit l'environnement : chaque appel de test prend la lecture,
// [`with_env`] l'écriture. Le fil qui tient l'écriture lit sans verrou.
#[cfg(test)]
static ENV_LOCK: std::sync::RwLock<()> = std::sync::RwLock::new(());
#[cfg(test)]
thread_local!(static ENV_OWNER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) });

/// Exécute `f` avec `name=value`, à l'abri des autres `parse_with` de test,
/// puis restaure la valeur d'origine, même sur panique.
#[cfg(test)]
pub(super) fn with_env<R>(name: &'static str, value: &str, f: impl FnOnce() -> R) -> R {
    struct Restore(&'static str, Option<std::ffi::OsString>);
    impl Drop for Restore {
        fn drop(&mut self) {
            match self.1.take() {
                Some(value) => std::env::set_var(self.0, value),
                None => std::env::remove_var(self.0),
            }
            ENV_OWNER.set(false);
        }
    }
    let _write = ENV_LOCK.write().unwrap_or_else(|e| e.into_inner());
    let _restore = Restore(name, std::env::var_os(name));
    ENV_OWNER.set(true);
    std::env::set_var(name, value);
    f()
}

impl ServeArgs {
    /// Parse les options sans profil.
    #[cfg(test)]
    pub(super) fn parse(args: impl IntoIterator<Item = String>) -> ServeResult<Self> {
        Self::parse_with(&Profile::default(), args)
    }

    /// Parse les options ; `profile` fournit les défauts, la CLI gagne.
    pub(super) fn parse_with(
        profile: &Profile,
        args: impl IntoIterator<Item = String>,
    ) -> ServeResult<Self> {
        #[cfg(test)]
        let _env = (!ENV_OWNER.get()).then(|| ENV_LOCK.read().unwrap_or_else(|e| e.into_inner()));
        let mut host = DEFAULT_HOST.to_string();
        let mut port = None;
        let mut socket = PathBuf::from(DEFAULT_SOCKET);
        let mut api_key = env::var(API_KEY_ENV).ok().filter(|value| !value.is_empty());
        let mut models = Vec::new();
        let mut stt_model = None;
        let mut tts_model = None;
        let mut diar_model = None;
        let mut embed_model = None;
        let mut enhance_model = None;
        let mut backend = match &profile.engine.backend {
            Some(name) => RuntimeKind::parse(name).map_err(|e| ServeError::args(e.to_string()))?,
            None => RuntimeKind::default_backend(),
        };
        let mut preload = profile.engine.preload.unwrap_or(false);
        let mut read_timeout = Duration::from_secs(
            profile
                .serve
                .read_timeout_secs
                .unwrap_or(DEFAULT_READ_TIMEOUT_SECS),
        );
        let mut max_tokens_cap = match profile.serve.max_tokens_cap {
            Some(cap) => usize::try_from(cap)
                .map_err(|_| ServeError::args("serve.max_tokens_cap hors plage"))?,
            None => DEFAULT_MAX_TOKENS_CAP,
        };
        // `env_var` applique la convention des autres clés : CLI > `--set` > env > profil.
        let mut max_tok_s = saragossa::runtime_flags::env_var("SARAGOSSA_RUST_MAX_TOK_S")
            .and_then(|v| saragossa::parse_max_tok_s(v.trim()).ok())
            .or(profile.serve.max_tok_s);
        // invariant: `Profile::validate` borne `serve.gpu_share` à 10..=100.
        let mut gpu_share = profile.serve.gpu_share.map(|v| v as u8);
        let mut idle_unload_secs =
            saragossa::runtime_flags::env_var("SARAGOSSA_RUST_IDLE_UNLOAD_SECS")
                .and_then(|v| v.trim().parse().ok())
                .or(profile.serve.idle_unload_secs)
                .unwrap_or(0);
        let mut auto_yield = saragossa::runtime_flags::env_flag(
            "SARAGOSSA_SERVE_AUTO_YIELD",
            profile.serve.auto_yield.unwrap_or(false),
        );
        let mut prefix_ssd = saragossa::runtime_flags::ServePrefixSsdOptions::from_env()?;
        // invariant: `Profile::validate` borne `serve.engines` à 1..=16.
        // Validé après la CLI : `--engines` l'emporte sur un environnement invalide.
        let engines_env = saragossa::runtime_flags::env_var("SARAGOSSA_SERVE_ENGINES");
        let mut engines_cli = None;
        let mut iter = args.into_iter();

        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--host" => host = next_value(&mut iter, "--host")?,
                "--port" => {
                    port = Some(
                        next_value(&mut iter, "--port")?
                            .parse::<u16>()
                            .map_err(|e| ServeError::args(format!("--port invalide: {e}")))?,
                    )
                }
                "--socket" => socket = next_value(&mut iter, "--socket")?.into(),
                "--api-key" | "--bearer-token" => {
                    api_key = Some(next_value(&mut iter, "--api-key")?)
                }
                "--model" => {
                    let model = parse_model_registration(&next_value(&mut iter, "--model")?)?;
                    upsert_model(&mut models, model)?;
                }
                "--model-27b" => {
                    upsert_model(
                        &mut models,
                        ServeModelConfig {
                            id: "reti-27b".to_string(),
                            path: next_value(&mut iter, "--model-27b")?.into(),
                            hf_repo: None,
                        },
                    )?;
                }
                "--model-35b" => {
                    upsert_model(
                        &mut models,
                        ServeModelConfig {
                            id: "reti-35b".to_string(),
                            path: next_value(&mut iter, "--model-35b")?.into(),
                            hf_repo: None,
                        },
                    )?;
                }
                "--stt-model" => {
                    require_feature("--stt-model", "stt", cfg!(feature = "stt"))?;
                    stt_model = Some(next_value(&mut iter, "--stt-model")?.into());
                }
                "--diar-model" => {
                    require_feature("--diar-model", "diar", cfg!(feature = "diar"))?;
                    diar_model = Some(next_value(&mut iter, "--diar-model")?.into());
                }
                "--tts-model" => {
                    require_feature("--tts-model", "tts", cfg!(feature = "tts"))?;
                    tts_model = Some(next_value(&mut iter, "--tts-model")?.into());
                }
                "--embed-model" => {
                    require_feature("--embed-model", "embed", cfg!(feature = "embed"))?;
                    embed_model = Some(next_value(&mut iter, "--embed-model")?.into());
                }
                "--enhance-model" => {
                    require_feature("--enhance-model", "denoise", cfg!(feature = "denoise"))?;
                    enhance_model = Some(next_value(&mut iter, "--enhance-model")?.into());
                }
                "--backend" | "--runtime" => {
                    backend = RuntimeKind::parse(&next_value(&mut iter, "--backend")?)
                        .map_err(|e| ServeError::args(e.to_string()))?;
                }
                "--prefix-cache-ssd-dir" => {
                    prefix_ssd.directory =
                        Some(next_value(&mut iter, "--prefix-cache-ssd-dir")?.into())
                }
                "--prefix-cache-ssd-gib" => {
                    prefix_ssd.gib = parse_positive_u64("--prefix-cache-ssd-gib", &mut iter)?
                }
                "--prefix-cache-ssd-priority-tokens" => {
                    prefix_ssd.priority_tokens =
                        next_value(&mut iter, "--prefix-cache-ssd-priority-tokens")?
                            .parse()
                            .map_err(|_| ServeError::args("priorité SSD invalide"))?;
                }
                "--preload" => preload = true,
                "--read-timeout-secs" => {
                    read_timeout =
                        Duration::from_secs(parse_positive_u64("--read-timeout-secs", &mut iter)?);
                }
                "--max-tokens-cap" => {
                    max_tokens_cap = parse_positive_usize("--max-tokens-cap", &mut iter)?;
                }
                "--max-tok-s" => {
                    max_tok_s = Some(
                        saragossa::parse_max_tok_s(&next_value(&mut iter, "--max-tok-s")?)
                            .map_err(ServeError::args)?,
                    );
                }
                "--gpu-share" => {
                    gpu_share = Some(
                        saragossa::parse_gpu_share(&next_value(&mut iter, "--gpu-share")?)
                            .map_err(ServeError::args)?,
                    );
                }
                "--engines" => {
                    engines_cli = Some(parse_engines(&next_value(&mut iter, "--engines")?)?);
                }
                "--idle-unload-secs" => {
                    idle_unload_secs = next_value(&mut iter, "--idle-unload-secs")?
                        .parse()
                        .map_err(|_| ServeError::args("--idle-unload-secs attend un entier"))?;
                }
                "--auto-yield" => auto_yield = true,
                "--help" | "-h" => return Err(ServeError::args(help_text())),
                other => return Err(ServeError::args(format!("argument serve inconnu: {other}"))),
            }
        }
        let engines = match (engines_cli, engines_env) {
            (Some(engines), _) => engines,
            (None, Some(value)) => parse_engines(&value)?,
            (None, None) => profile
                .serve
                .engines
                .and_then(|v| usize::try_from(v).ok())
                .unwrap_or(1),
        };
        // La CLI gagne : une entrée TOML de même id est ignorée.
        for entry in &profile.models {
            if !models.iter().any(|model| model.id == entry.id) {
                models.push(config_from_entry(entry)?);
            }
        }
        prefix_ssd.bytes()?;
        if engines > 1 && prefix_ssd.directory.is_some() {
            // Le cache SSD est un singleton du process, partagé par les moteurs.
            return Err(ServeError::args(
                "--prefix-cache-ssd-dir est incompatible avec --engines > 1 (cache commun aux moteurs)",
            ));
        }
        validate_tcp_args(&host, port, api_key.as_deref())?;
        Ok(Self {
            host,
            port,
            socket,
            api_key,
            models,
            stt_model,
            diar_model,
            tts_model,
            embed_model,
            enhance_model,
            backend,
            preload,
            read_timeout,
            max_tokens_cap,
            max_tok_s,
            gpu_share,
            idle_unload_secs,
            // invariant: `Profile::validate` borne la part à 10..=100 et le délai à >= 5.
            auto_yield: auto_yield.then(|| super::auto_yield::AutoYield {
                share: profile.serve.auto_yield_share.unwrap_or(25) as u8,
                idle: Duration::from_secs(profile.serve.auto_yield_idle_secs.unwrap_or(60)),
            }),
            prefix_ssd,
            engines,
        })
    }

    /// Renvoie l'adresse TCP `host:port` si TCP est activé.
    pub(super) fn tcp_addr(&self) -> Option<String> {
        self.port.map(|port| format!("{}:{port}", self.host))
    }

    /// Ajoute un modèle choisi après le parsing, notamment par le picker.
    pub(super) fn add_model(&mut self, model: ServeModelConfig) -> ServeResult<()> {
        upsert_model(&mut self.models, model)
    }
}

fn parse_engines(value: &str) -> ServeResult<usize> {
    value
        .trim()
        .parse::<usize>()
        .ok()
        .filter(|engines| (1..=MAX_ENGINES).contains(engines))
        .ok_or_else(|| ServeError::args(format!("--engines attend un entier de 1 à {MAX_ENGINES}")))
}

/// Renvoie l'aide concise de `serve`.
pub(super) fn help_text() -> String {
    let mut modality_options = String::new();
    let mut modality_help = String::new();
    for (enabled, flag, description) in [
        (
            cfg!(feature = "diar"),
            "--diar-model",
            "Nemotron-3-Diarization on /v1/audio/diarization",
        ),
        (
            cfg!(feature = "stt"),
            "--stt-model",
            "Whisper STT on /v1/audio/transcriptions",
        ),
        (
            cfg!(feature = "tts"),
            "--tts-model",
            "Qwen3 TTS on /v1/audio/speech",
        ),
        (
            cfg!(feature = "embed"),
            "--embed-model",
            "e5-small on /v1/embeddings",
        ),
        (
            cfg!(feature = "denoise"),
            "--enhance-model",
            "DeepFilterNet3 denoising on /v1/audio/enhance",
        ),
    ] {
        if enabled {
            let value = if flag == "--enhance-model" {
                "FILE"
            } else {
                "DIR"
            };
            modality_options.push_str(&format!("[{flag} {value}] "));
            modality_help.push_str(&format!(
                "{flag} enables {description} (opt-in, lazy-loaded).\n"
            ));
        }
    }
    format!(
        "Usage: saragossa serve [--socket PATH] [--port {DEFAULT_PORT}] \\
         [--api-key TOKEN (deprecated, use SARAGOSSA_API_KEY)] [--model ID=DIR|DIR|ORG/REPO]... \\
         [--model-27b DIR] [--model-35b DIR] \\
         {modality_options}[--backend cpu|metal] [--preload] [--max-tok-s R] [--gpu-share 10-100] [--auto-yield] [--idle-unload-secs N] [--engines N] [--profile FILE|low-mem|throughput] [--set section.key=value]...\n\
         Config: optional TOML profile (--profile/--config or SARAGOSSA_CONFIG; `extends = \"low-mem\"` inside a file); precedence CLI/--set > env > TOML > preset > default. SIGHUP reloads serve.read_timeout_secs, serve.max_tokens_cap, serve.max_tok_s, serve.gpu_share, serve.idle_unload_secs and the [[models]] list (id, path) only. See `saragossa config print`.\n\
         Upgrade (Unix): SIGUSR2 starts the same binary path (or --upgrade-binary PATH) with the same arguments on the inherited listening socket, then the old process drains for at most serve.drain_timeout_secs (30) and exits; enhance h2/WebTransport listeners follow. serve.relay_mode=auto|parallel|sequential (auto: sequential only if the memory guard refuses two resident processes).\n\
         SSD (opt-in): --prefix-cache-ssd-dir DIR --prefix-cache-ssd-gib 32; \
         --prefix-cache-ssd-priority-tokens 4096 (0 disables priority).\n\
         Models: repeat --model to expose any supported checkpoint. A bare path \\
         gets its directory name as alias; ORG/REPO is resolved through the local \\
         Hugging Face cache and downloaded when incomplete. Without --model, a TTY \\
         opens a local-model picker. \\
         Backward-compatible aliases: --model-27b registers reti-27b, \\
         --model-35b registers reti-35b.\n\
         {modality_help}Safety defaults: --read-timeout-secs {DEFAULT_READ_TIMEOUT_SECS}, \\
         --max-tokens-cap {DEFAULT_MAX_TOKENS_CAP}.\n\
         Idle unload: --idle-unload-secs N (serve.idle_unload_secs, \
         SARAGOSSA_RUST_IDLE_UNLOAD_SECS, hot-reloadable, 0 = off, default) drops all \
         models after N idle seconds; the next request reloads them. --preload models \
         are unloaded too.\n\
         Auto yield (macOS, opt-in): --auto-yield (serve.auto_yield, SARAGOSSA_SERVE_AUTO_YIELD) \
         lowers the GPU share to serve.auto_yield_share (25) while keyboard or mouse input \
         is seen, and restores the configured share after serve.auto_yield_idle_secs (60) \
         idle seconds; output tokens are unchanged, only timing.\n\
         Engines: --engines N (serve.engines, SARAGOSSA_SERVE_ENGINES, 1..={MAX_ENGINES}, \
         default 1, not hot-reloadable) runs N inference threads on one shared copy of the \
         weights, each with its own KV, scratch and block prefix cache; 1 keeps the serial \
         engine. With N > 1 the decoder's internal prefix cache (MTP/DFlash) and the TTS \
         prefill memo are disabled, --prefix-cache-ssd-dir is refused, request batching is \
         off and a request panic stops the process. Limits: one Metal command queue per model is shared (first sound grows \
         about linearly with N), streamed STT chunks are serialized across engines, a model \
         load holds the weight registry lock, a derived weight may be built twice on the \
         first concurrent call. See docs/perf/serve-multi-moteur.md.\n\
         Default transport: Unix socket {DEFAULT_SOCKET} (0600). \\
         --port enables TCP on 127.0.0.1 for grob and requires --api-key \\
         or {API_KEY_ENV}.\n\
         Endpoints: GET /v1/models, POST /v1/chat/completions, \\
         POST /v1/audio/transcriptions, POST /v1/audio/transcriptions/stream, \
         POST /v1/audio/diarization, POST /v1/audio/speech, POST /v1/audio/enhance, POST /v1/embeddings"
    )
}

fn require_feature(flag: &str, feature: &str, enabled: bool) -> ServeResult<()> {
    if !enabled {
        return Err(ServeError::args(format!(
            "{flag} indisponible: compilé sans la feature {feature}"
        )));
    }
    Ok(())
}

fn validate_tcp_args(host: &str, port: Option<u16>, api_key: Option<&str>) -> ServeResult<()> {
    if port.is_none() {
        return Ok(());
    }
    if host != DEFAULT_HOST {
        return Err(ServeError::args(format!(
            "serve TCP bind refusé: --host doit rester {DEFAULT_HOST}"
        )));
    }
    if api_key.map(str::is_empty).unwrap_or(true) {
        return Err(ServeError::args(format!(
            "--port requiert --api-key TOKEN ou {API_KEY_ENV}=TOKEN"
        )));
    }
    Ok(())
}

pub(super) fn parse_model_registration(value: &str) -> ServeResult<ServeModelConfig> {
    let value = value.trim();
    if let Some((id, path)) = value.split_once('=') {
        let id = id.trim();
        let path = path.trim();
        if id.is_empty() {
            return Err(ServeError::args("--model: id vide"));
        }
        if path.is_empty() {
            return Err(ServeError::args(format!("--model {id}: chemin vide")));
        }
        return Ok(ServeModelConfig {
            id: id.to_string(),
            path: PathBuf::from(path),
            hf_repo: None,
        });
    }
    if value.is_empty() {
        return Err(ServeError::args("--model: référence vide"));
    }
    let id = model_alias(value)?;
    let hf_repo = (!Path::new(value).exists() && hf_resolve::is_hf_model_id(value))
        .then(|| value.to_string());
    Ok(ServeModelConfig {
        id,
        path: PathBuf::from(value),
        hf_repo,
    })
}

/// Convertit une entrée `[[models]]` ; une référence HF nue est résolue au lancement.
pub(super) fn config_from_entry(entry: &ModelEntry) -> ServeResult<ServeModelConfig> {
    if entry.id.trim().is_empty() || entry.path.trim().is_empty() {
        return Err(ServeError::args("[[models]]: id et path requis"));
    }
    let hf_repo = (!Path::new(&entry.path).exists() && hf_resolve::is_hf_model_id(&entry.path))
        .then(|| entry.path.clone());
    Ok(ServeModelConfig {
        id: entry.id.trim().to_string(),
        path: PathBuf::from(entry.path.trim()),
        hf_repo,
    })
}

fn model_alias(value: &str) -> ServeResult<String> {
    Path::new(value)
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .map(ToString::to_string)
        .ok_or_else(|| {
            ServeError::args(format!(
                "--model {value}: impossible de déduire un alias depuis le dernier segment"
            ))
        })
}
fn upsert_model(models: &mut Vec<ServeModelConfig>, model: ServeModelConfig) -> ServeResult<()> {
    if let Some(index) = models.iter().position(|entry| entry.id == model.id) {
        models[index] = model;
        return Ok(());
    }
    models.push(model);
    Ok(())
}

fn next_value(iter: &mut impl Iterator<Item = String>, flag: &'static str) -> ServeResult<String> {
    iter.next()
        .ok_or_else(|| ServeError::args(format!("valeur manquante pour {flag}")))
}

fn parse_positive_u64(
    flag: &'static str,
    iter: &mut impl Iterator<Item = String>,
) -> ServeResult<u64> {
    let value = next_value(iter, flag)?;
    let parsed = value
        .parse::<u64>()
        .map_err(|e| ServeError::args(format!("{flag} invalide: {e}")))?;
    if parsed == 0 {
        return Err(ServeError::args(format!("{flag} doit être > 0")));
    }
    Ok(parsed)
}

fn parse_positive_usize(
    flag: &'static str,
    iter: &mut impl Iterator<Item = String>,
) -> ServeResult<usize> {
    let value = next_value(iter, flag)?;
    let parsed = value
        .parse::<usize>()
        .map_err(|e| ServeError::args(format!("{flag} invalide: {e}")))?;
    if parsed == 0 {
        return Err(ServeError::args(format!("{flag} doit être > 0")));
    }
    Ok(parsed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idle_unload_secs_defaults_to_off_and_parses() {
        let off = ServeArgs::parse(["--model".into(), "a=/x/a".into()]).expect("args");
        assert_eq!(off.idle_unload_secs, 0);
        let on =
            ServeArgs::parse(["--model", "a=/x/a", "--idle-unload-secs", "90"].map(String::from))
                .expect("args");
        assert_eq!(on.idle_unload_secs, 90);
    }

    #[test]
    fn auto_yield_is_off_by_default_and_set_by_cli_env_or_profile() {
        use super::super::auto_yield::AutoYield;
        let base = ["--model", "a=/x/a"].map(String::from);
        let off = ServeArgs::parse(base.clone()).expect("args");
        assert_eq!(off.auto_yield, None, "défaut inchangé");
        let default = Some(AutoYield {
            share: 25,
            idle: Duration::from_secs(60),
        });
        let mut cli = base.to_vec();
        cli.push("--auto-yield".into());
        assert_eq!(ServeArgs::parse(cli).expect("cli").auto_yield, default);
        with_env("SARAGOSSA_SERVE_AUTO_YIELD", "1", || {
            assert_eq!(
                ServeArgs::parse(base.clone()).expect("env").auto_yield,
                default
            );
        });
        let profile = saragossa::runtime_flags::Profile::parse(
            "[serve]\nauto_yield = true\nauto_yield_share = 40\nauto_yield_idle_secs = 9\n",
        )
        .expect("profil");
        let tuned = Some(AutoYield {
            share: 40,
            idle: Duration::from_secs(9),
        });
        let parsed = ServeArgs::parse_with(&profile, base.to_vec()).expect("profil");
        assert_eq!(parsed.auto_yield, tuned);
        with_env("SARAGOSSA_SERVE_AUTO_YIELD", "0", || {
            let parsed = ServeArgs::parse_with(&profile, base.to_vec()).expect("env 0");
            assert_eq!(
                parsed.auto_yield, None,
                "l'environnement prime sur le profil"
            );
        });
        for bad in ["auto_yield_share = 5", "auto_yield_idle_secs = 4"] {
            assert!(
                saragossa::runtime_flags::Profile::parse(&format!("[serve]\n{bad}\n")).is_err()
            );
        }
    }

    #[test]
    fn engines_default_to_one_and_are_bounded() {
        let parse = |value: &str| {
            ServeArgs::parse(["--model", "a=/x/a", "--engines", value].map(String::from))
                .map(|args| args.engines)
        };
        let default = ServeArgs::parse(["--model".into(), "a=/x/a".into()]).expect("args");
        assert_eq!(default.engines, 1);
        assert_eq!(parse("6").ok(), Some(6));
        for refused in ["0", "17", "x"] {
            assert!(parse(refused).is_err(), "{refused}");
        }
        // CLI > environnement, même invalide ; sans CLI, l'environnement compte.
        with_env("SARAGOSSA_SERVE_ENGINES", "17", || {
            assert_eq!(parse("2").ok(), Some(2));
            assert!(ServeArgs::parse(["--model".into(), "a=/x/a".into()]).is_err());
        });
        let ssd = ServeArgs::parse(
            [
                "--model",
                "a=/x/a",
                "--engines",
                "2",
                "--prefix-cache-ssd-dir",
                "/tmp/x",
            ]
            .map(String::from),
        );
        assert!(ssd.is_err_and(|e| e.to_string().contains("incompatible")));
        with_env("SARAGOSSA_SERVE_ENGINES", "3", || {
            let args = ServeArgs::parse(["--model".into(), "a=/x/a".into()]).expect("args");
            assert_eq!(args.engines, 3);
        });
    }

    #[test]
    fn profile_sets_defaults_and_cli_wins() {
        let profile = Profile::parse(
            "[engine]\nbackend = \"cpu\"\npreload = true\n\
             [serve]\nread_timeout_secs = 9\nmax_tokens_cap = 100\n",
        )
        .expect("profil valide");
        let from_file = ServeArgs::parse_with(&profile, Vec::new()).expect("args");
        assert_eq!(from_file.backend, RuntimeKind::Cpu);
        assert!(from_file.preload);
        assert_eq!(from_file.read_timeout, Duration::from_secs(9));
        assert_eq!(from_file.max_tokens_cap, 100);
        let cli = ServeArgs::parse_with(
            &profile,
            ["--read-timeout-secs", "5", "--max-tokens-cap", "7"].map(String::from),
        )
        .expect("args");
        assert_eq!(cli.read_timeout, Duration::from_secs(5));
        assert_eq!(cli.max_tokens_cap, 7);
    }

    #[test]
    fn modality_options_follow_compiled_features() {
        let help = help_text();
        for (enabled, flag, feature) in [
            (cfg!(feature = "diar"), "--diar-model", "diar"),
            (cfg!(feature = "stt"), "--stt-model", "stt"),
            (cfg!(feature = "tts"), "--tts-model", "tts"),
            (cfg!(feature = "embed"), "--embed-model", "embed"),
            (cfg!(feature = "denoise"), "--enhance-model", "denoise"),
        ] {
            assert_eq!(help.contains(flag), enabled, "{flag}");
            let parsed = ServeArgs::parse([flag.to_string(), "/m/modality".to_string()]);
            if enabled {
                let args = parsed.expect("invariant: modalité compilée acceptée");
                let path = match feature {
                    "diar" => args.diar_model,
                    "stt" => args.stt_model,
                    "tts" => args.tts_model,
                    "denoise" => args.enhance_model,
                    _ => args.embed_model,
                };
                assert_eq!(path, Some(PathBuf::from("/m/modality")));
            } else {
                let expected = format!("compilé sans la feature {feature}");
                let error = parsed.expect_err("invariant: modalité absente refusée");
                assert!(error.to_string().contains(&expected));
                for values in [
                    vec![flag],
                    vec!["--model", "llm=/m/llm", flag, "/m/modality"],
                    vec![flag, "/m/modality", "--model", "llm=/m/llm"],
                ] {
                    let error = ServeArgs::parse(values.into_iter().map(str::to_string))
                        .expect_err("invariant: option absente refusée avant la sélection LLM");
                    assert!(error.to_string().contains(&expected));
                }
            }
        }
    }

    #[test]
    fn parse_defaults_to_unix_socket_without_implicit_models() {
        let args = ServeArgs::parse(Vec::<String>::new()).expect("invariant: defaults valides");

        assert_eq!(args.host, "127.0.0.1");
        assert_eq!(args.port, None);
        assert_eq!(args.socket, PathBuf::from(DEFAULT_SOCKET));
        assert_eq!(args.api_key, None);
        assert!(args.models.is_empty());
        assert_eq!(args.backend, RuntimeKind::default_backend());
        assert!(!args.preload);
        assert_eq!(
            args.read_timeout,
            Duration::from_secs(DEFAULT_READ_TIMEOUT_SECS)
        );
        assert_eq!(args.max_tokens_cap, DEFAULT_MAX_TOKENS_CAP);
    }

    #[test]
    fn parse_accepts_repeatable_models_socket_and_tcp_token() {
        let args = ServeArgs::parse([
            "--port".to_string(),
            "8090".to_string(),
            "--socket".to_string(),
            "/tmp/saragossa.sock".to_string(),
            "--model".to_string(),
            "reti-35b=/m/35".to_string(),
            "--model".to_string(),
            "qwen-7b=/m/7".to_string(),
            "--api-key".to_string(),
            "secret".to_string(),
            "--preload".to_string(),
            "--read-timeout-secs".to_string(),
            "12".to_string(),
            "--max-tokens-cap".to_string(),
            "2048".to_string(),
        ])
        .expect("invariant: args valides");

        assert_eq!(args.port, Some(8090));
        assert_eq!(args.socket, PathBuf::from("/tmp/saragossa.sock"));
        assert_eq!(args.api_key, Some("secret".to_string()));
        assert_eq!(
            args.tcp_addr().expect("invariant: tcp activé"),
            "127.0.0.1:8090"
        );
        assert_eq!(
            args.models,
            vec![
                ServeModelConfig {
                    id: "reti-35b".to_string(),
                    path: PathBuf::from("/m/35"),
                    hf_repo: None,
                },
                ServeModelConfig {
                    id: "qwen-7b".to_string(),
                    path: PathBuf::from("/m/7"),
                    hf_repo: None,
                },
            ]
        );
        assert!(args.preload);
        assert_eq!(args.read_timeout, Duration::from_secs(12));
        assert_eq!(args.max_tokens_cap, 2048);
    }

    #[test]
    fn parse_accepts_legacy_model_aliases() {
        let args = ServeArgs::parse([
            "--model-27b".to_string(),
            "/m/27".to_string(),
            "--model-35b".to_string(),
            "/m/35".to_string(),
        ])
        .expect("invariant: aliases valides");

        assert_eq!(
            args.models,
            vec![
                ServeModelConfig {
                    id: "reti-27b".to_string(),
                    path: PathBuf::from("/m/27"),
                    hf_repo: None,
                },
                ServeModelConfig {
                    id: "reti-35b".to_string(),
                    path: PathBuf::from("/m/35"),
                    hf_repo: None,
                },
            ]
        );
    }

    #[test]
    fn parse_alias_overrides_repeatable_model_id() {
        let args = ServeArgs::parse([
            "--model".to_string(),
            "reti-35b=/old".to_string(),
            "--model-35b".to_string(),
            "/new".to_string(),
        ])
        .expect("invariant: override valide");

        assert_eq!(args.models.len(), 1);
        assert_eq!(args.models[0].id, "reti-35b");
        assert_eq!(args.models[0].path, PathBuf::from("/new"));
    }

    #[test]
    fn parse_accepts_bare_path_with_automatic_alias() {
        let temp = tempfile::tempdir().expect("invariant: tempdir disponible");
        let model = temp.path().join("Qwen-local");
        std::fs::create_dir(&model).expect("invariant: dossier modèle créé");

        let args = ServeArgs::parse(["--model".to_string(), model.to_string_lossy().into_owned()])
            .expect("invariant: chemin nu accepté");

        assert_eq!(args.models[0].id, "Qwen-local");
        assert_eq!(args.models[0].path, model);
        assert_eq!(args.models[0].hf_repo, None);
    }

    #[test]
    fn parse_accepts_hf_id_with_repo_alias() {
        let args = ServeArgs::parse(["--model".to_string(), "mlx-community/Qwen3-4B".to_string()])
            .expect("invariant: id HF accepté");

        assert_eq!(args.models[0].id, "Qwen3-4B");
        assert_eq!(
            args.models[0].hf_repo.as_deref(),
            Some("mlx-community/Qwen3-4B")
        );
    }

    #[test]
    fn parse_keeps_explicit_alias_path_lazy() {
        let args = ServeArgs::parse(["--model".to_string(), "custom=missing/local".to_string()])
            .expect("invariant: alias historique accepté");

        assert_eq!(args.models[0].id, "custom");
        assert_eq!(args.models[0].path, PathBuf::from("missing/local"));
        assert_eq!(args.models[0].hf_repo, None);
    }

    #[test]
    fn parse_rejects_tcp_without_token() {
        let error = validate_tcp_args(DEFAULT_HOST, Some(8090), None)
            .expect_err("invariant: token requis en TCP");

        assert!(error.to_string().contains("--port requiert --api-key"));
    }

    #[test]
    fn parse_rejects_non_loopback_tcp() {
        let error = ServeArgs::parse([
            "--port".to_string(),
            "8090".to_string(),
            "--host".to_string(),
            "0.0.0.0".to_string(),
            "--api-key".to_string(),
            "secret".to_string(),
        ])
        .expect_err("invariant: bind non loopback refusé");

        assert!(error.to_string().contains("127.0.0.1"));
    }

    #[test]
    fn parse_rejects_zero_limits() {
        let timeout = ServeArgs::parse(["--read-timeout-secs".to_string(), "0".to_string()])
            .expect_err("invariant: timeout nul refusé");
        assert!(timeout.to_string().contains("> 0"));

        let cap = ServeArgs::parse(["--max-tokens-cap".to_string(), "0".to_string()])
            .expect_err("invariant: cap nul refusé");
        assert!(cap.to_string().contains("> 0"));
    }
}

#[cfg(test)]
mod ssd_tests {
    use super::*;
    #[test]
    fn explicit_ssd_options_and_invalid_budgets() {
        let args = ServeArgs::parse(
            [
                "--prefix-cache-ssd-dir",
                "/cache",
                "--prefix-cache-ssd-gib",
                "8",
                "--prefix-cache-ssd-priority-tokens",
                "0",
            ]
            .map(str::to_owned),
        )
        .unwrap();
        assert_eq!(args.prefix_ssd.directory, Some(PathBuf::from("/cache")));
        assert_eq!(args.prefix_ssd.bytes().unwrap(), 8 << 30);
        assert_eq!(args.prefix_ssd.priority_tokens, 0);
        for value in ["0", "18446744073709551615", "wrong"] {
            assert!(
                ServeArgs::parse(["--prefix-cache-ssd-gib", value].map(str::to_owned)).is_err()
            );
        }
    }
}
