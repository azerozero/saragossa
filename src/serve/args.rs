//! Parsing des options de `saragossa serve`.

use std::env;
use std::path::{Path, PathBuf};
use std::time::Duration;

use super::error::{ServeError, ServeResult};
use crate::{hf_resolve, RuntimeKind};

const DEFAULT_HOST: &str = "127.0.0.1";
const DEFAULT_PORT: u16 = 8081;
const DEFAULT_SOCKET: &str = "/tmp/saragossa-serve.sock";
const API_KEY_ENV: &str = "SARAGOSSA_API_KEY";
const DEFAULT_READ_TIMEOUT_SECS: u64 = 30;
const DEFAULT_MAX_TOKENS_CAP: usize = 4096;

/// Configuration de lancement du serveur.
#[derive(Clone, Debug, PartialEq, Eq)]
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
    /// Répertoire du modèle TTS Qwen3 exposé sur `/v1/audio/speech`.
    pub(super) tts_model: Option<PathBuf>,
    /// Répertoire du modèle d'embeddings e5-small exposé sur `/v1/embeddings`.
    pub(super) embed_model: Option<PathBuf>,
    /// Backend d'inférence.
    pub(super) backend: RuntimeKind,
    /// Charge tous les modèles au démarrage.
    pub(super) preload: bool,
    /// Timeout de lecture par connexion acceptée.
    pub(super) read_timeout: Duration,
    /// Plafond dur du budget de génération par requête.
    pub(super) max_tokens_cap: usize,
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

impl ServeArgs {
    /// Parse les options de la sous-commande.
    pub(super) fn parse(args: impl IntoIterator<Item = String>) -> ServeResult<Self> {
        let mut host = DEFAULT_HOST.to_string();
        let mut port = None;
        let mut socket = PathBuf::from(DEFAULT_SOCKET);
        let mut api_key = env::var(API_KEY_ENV).ok().filter(|value| !value.is_empty());
        let mut models = Vec::new();
        let mut stt_model = None;
        let mut tts_model = None;
        let mut embed_model = None;
        let mut backend = RuntimeKind::default_backend();
        let mut preload = false;
        let mut read_timeout = Duration::from_secs(DEFAULT_READ_TIMEOUT_SECS);
        let mut max_tokens_cap = DEFAULT_MAX_TOKENS_CAP;
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
                    stt_model = Some(next_value(&mut iter, "--stt-model")?.into());
                }
                "--tts-model" => {
                    tts_model = Some(next_value(&mut iter, "--tts-model")?.into());
                }
                "--embed-model" => {
                    embed_model = Some(next_value(&mut iter, "--embed-model")?.into());
                }
                "--backend" | "--runtime" => {
                    backend = RuntimeKind::parse(&next_value(&mut iter, "--backend")?)
                        .map_err(|e| ServeError::args(e.to_string()))?;
                }
                "--preload" => preload = true,
                "--read-timeout-secs" => {
                    read_timeout =
                        Duration::from_secs(parse_positive_u64("--read-timeout-secs", &mut iter)?);
                }
                "--max-tokens-cap" => {
                    max_tokens_cap = parse_positive_usize("--max-tokens-cap", &mut iter)?;
                }
                "--help" | "-h" => return Err(ServeError::args(help_text())),
                other => return Err(ServeError::args(format!("argument serve inconnu: {other}"))),
            }
        }
        validate_tcp_args(&host, port, api_key.as_deref())?;
        Ok(Self {
            host,
            port,
            socket,
            api_key,
            models,
            stt_model,
            tts_model,
            embed_model,
            backend,
            preload,
            read_timeout,
            max_tokens_cap,
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

/// Renvoie l'aide concise de `serve`.
pub(super) fn help_text() -> String {
    format!(
        "Usage: saragossa serve [--socket PATH] [--port {DEFAULT_PORT}] \\
         [--api-key TOKEN] [--model ID=DIR|DIR|ORG/REPO]... \\
         [--model-27b DIR] [--model-35b DIR] \\
         [--stt-model DIR] [--tts-model DIR] [--embed-model DIR] \\
         [--backend cpu|metal] [--preload]\n\
         Models: repeat --model to expose any supported checkpoint. A bare path \\
         gets its directory name as alias; ORG/REPO is resolved through the local \\
         Hugging Face cache and downloaded when incomplete. Without --model, a TTY \\
         opens a local-model picker. \\
         Backward-compatible aliases: --model-27b registers reti-27b, \\
         --model-35b registers reti-35b.\n\
         Audio/embeddings (opt-in, lazy-loaded): --stt-model enables Whisper STT \\
         on /v1/audio/transcriptions, --tts-model enables Qwen3 TTS on \\
         /v1/audio/speech, --embed-model enables e5-small on /v1/embeddings.\n\
         Safety defaults: --read-timeout-secs {DEFAULT_READ_TIMEOUT_SECS}, \\
         --max-tokens-cap {DEFAULT_MAX_TOKENS_CAP}.\n\
         Default transport: Unix socket {DEFAULT_SOCKET} (0600). \\
         --port enables TCP on 127.0.0.1 for grob and requires --api-key \\
         or {API_KEY_ENV}.\n\
         Endpoints: GET /v1/models, POST /v1/chat/completions, \\
         POST /v1/audio/transcriptions, POST /v1/audio/speech, POST /v1/embeddings"
    )
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
