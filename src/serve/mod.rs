//! Sous-commande `saragossa serve`.
//! Les modules error, mtp et streaming sont aussi partagés avec le REPL LLM.
//!
//! Concurrence d'E/S seulement (voir `dispatch`) : un thread d'accept et un
//! thread par connexion lisent les requêtes, refusent un bearer invalide avant
//! de lire le corps et servent `/health` et `/v1/models` même pendant une
//! génération. L'inférence reste sérielle sur le thread moteur, le thread
//! principal qui possède `ServeState` ; un client parti pendant l'attente en
//! file n'est détecté qu'à l'écriture de la réponse. Une panique dans un thread de
//! connexion ne coûte que cette connexion et n'abat plus le process ; une
//! panique moteur l'abat toujours.
//!
//! Moteurs parallèles opt-in (`--engines N`, `SARAGOSSA_SERVE_ENGINES`) : N
//! threads moteur vident la même file, chacun avec son `ServeState` (KV,
//! scratch Metal cloisonné, cache de préfixe par blocs), sur une seule copie
//! des poids LLM, Whisper, TTS, diarisation et embeddings (voir `shared`) ; la
//! garde mémoire est commune. Dès N > 1 : le cache de préfixe interne du
//! décodeur (MTP, DFlash) et le mémo de prefill TTS, portés par les poids
//! partagés, sont coupés, le cache SSD (singleton du process) est refusé ; les
//! lots aussi ; une panique de requête arrête le
//! process, l'état partagé n'étant plus sûr. Limites connues : la file de
//! commandes Metal d'un modèle (prefill, Whisper, diarisation, codec TTS) est
//! partagée, le premier son croît donc à peu près linéairement avec N ; les
//! morceaux STT streamés sont sérialisés entre moteurs, dans leur ordre
//! d'arrivée ; un chargement tient le verrou du registre ; un dérivé de poids
//! peut être construit deux fois au premier appel concurrent. Mesures :
//! `docs/perf/serve-multi-moteur.md`.
//!
//! Lots opt-in (`SARAGOSSA_SERVE_BATCH=1` avec
//! `SARAGOSSA_RUST_LIGHTBATCH_MULTI=1`) : quand le moteur prend une requête
//! chat streamée greedy pour un modèle dense éligible, il y joint les requêtes
//! éligibles du même modèle déjà en file (`SARAGOSSA_SERVE_MAX_STREAMS`, 8) et
//! les décode ensemble par le pas multi-flux dense. Un lot d'une requête garde
//! le chemin solo (MTP, cache de préfixe par blocs).
//!
//! Plafonds, chacun répondant 503 immédiatement une fois atteint :
//! - `SARAGOSSA_SERVE_MAX_CONNECTIONS` (64) : connexions en cours de lecture ;
//! - `SARAGOSSA_SERVE_QUEUE` (32) : requêtes en attente du moteur, file du
//!   canal et file locale des lots confondues ;
//! - `SARAGOSSA_SERVE_MAX_INFLIGHT_BYTES` (256 Mio) : octets de corps lus ou
//!   en file.
//!
//! SIGHUP relit le profil (`--profile`/`--config`, `--set` compris) et n'applique
//! que `serve.read_timeout_secs`, `serve.max_tokens_cap`, `serve.max_tok_s`,
//! `serve.gpu_share`, `serve.idle_unload_secs` et la liste `[[models]]` ;
//! tout autre changement ou chemin de modèle illisible est refusé en bloc (voir
//! `reload`). `--auto-yield` (opt-in, macOS) baisse `gpu_share` tant que
//! l'utilisateur se sert du Mac (voir `auto_yield`).
//!
//! SIGUSR2 relaie sans coupure vers le binaire du même chemin (ou
//! `--upgrade-binary PATH`) : l'enfant hérite de la socket d'écoute, puis
//! l'ancien processus vide ses connexions et sort (voir `relay`).
//!
//! Hors débruitage duplex (`enhance`), le parseur accepte les corps à `Content-Length`.
//! `Transfer-Encoding` seul répond 501 ; avec `Content-Length`, ou un
//! `Content-Length` répété ou non numérique, le cadrage ambigu répond 400.

#[cfg(feature = "serve")]
mod anthropic;
#[cfg(feature = "serve")]
mod args;
#[cfg(any(
    feature = "stt",
    feature = "tts",
    feature = "diar",
    feature = "denoise"
))]
#[cfg(feature = "serve")]
mod audio;
#[cfg(feature = "serve")]
mod auto_yield;
#[cfg(feature = "serve")]
mod cache;
#[cfg(feature = "serve")]
mod dispatch;
#[cfg(feature = "embed")]
#[cfg(feature = "serve")]
mod embeddings;
#[cfg(all(feature = "serve", feature = "denoise"))]
mod enhance;
#[cfg(all(feature = "serve", feature = "denoise"))]
mod enhance_body;
pub(crate) mod error;
#[cfg(feature = "serve")]
mod flint_output;
#[cfg(feature = "serve")]
mod http;
#[cfg(feature = "serve")]
mod model_selection;
pub(crate) mod mtp;
#[cfg(feature = "serve")]
mod protocol;
#[cfg(all(feature = "serve", unix))]
mod relay;
#[cfg(feature = "serve")]
mod reload;
#[cfg(feature = "serve")]
mod responses;
#[cfg(feature = "serve")]
mod shared;
#[cfg(feature = "serve")]
mod state;
pub(crate) mod streaming;
#[cfg(feature = "stt")]
#[cfg(feature = "serve")]
mod stt;
#[cfg(feature = "serve")]
mod tool_output;
#[cfg(feature = "serve")]
mod warmup;

#[cfg(feature = "serve")]
use std::error::Error;

#[cfg(feature = "serve")]
use args::ServeArgs;

#[cfg(feature = "serve")]
use super::CliResult;

#[cfg(feature = "serve")]
/// Extrait `--profile`/`--config` (sinon `SARAGOSSA_CONFIG`) et les `--set` répétables.
fn take_config_args(
    args: Vec<String>,
) -> error::ServeResult<(Option<String>, Vec<String>, Vec<String>)> {
    let mut path = std::env::var("SARAGOSSA_CONFIG")
        .ok()
        .filter(|value| !value.is_empty());
    let mut sets = Vec::new();
    let mut rest = Vec::with_capacity(args.len());
    let mut iter = args.into_iter();
    while let Some(arg) = iter.next() {
        let missing = || error::ServeError::args(format!("{arg} attend une valeur"));
        match arg.as_str() {
            "--config" | "--profile" => path = Some(iter.next().ok_or_else(missing)?),
            "--set" => sets.push(iter.next().ok_or_else(missing)?),
            // NOTE: une valeur en argv est visible par `ps` ; elle passe donc par
            // l'environnement, que le relais SIGUSR2 transmet à l'enfant.
            "--api-key" | "--bearer-token" => {
                let key = iter.next().ok_or_else(missing)?;
                eprintln!(
                    "saragossa serve: --api-key est déprécié (visible dans `ps`), utiliser SARAGOSSA_API_KEY"
                );
                // NOTE: `set_var` n'est sûr que parce que ce code s'exécute au démarrage,
                // avant tout thread ; la clé passe ensuite à tous les sous-processus.
                std::env::set_var("SARAGOSSA_API_KEY", key);
            }
            _ => rest.push(arg),
        }
    }
    Ok((path, sets, rest))
}

/// Lance le serveur OpenAI-compatible local.
#[cfg(feature = "serve")]
pub(super) fn run(args: impl IntoIterator<Item = String>) -> CliResult<()> {
    let raw_args = args.into_iter().collect::<Vec<_>>();
    if raw_args
        .iter()
        .any(|arg| matches!(arg.as_str(), "--help" | "-h"))
    {
        println!("{}", args::help_text());
        return Ok(());
    }
    let (config_path, sets, mut raw_args) = take_config_args(raw_args).map_err(boxed_error)?;
    let resolved = saragossa::runtime_flags::Resolved::resolve(config_path.as_deref(), &sets)?;
    resolved.install();
    let profile = resolved.profile.clone();
    #[cfg(unix)]
    relay::configure(&mut raw_args, &profile.serve).map_err(boxed_error)?;
    let mut args = ServeArgs::parse_with(&profile, raw_args.clone()).map_err(boxed_error)?;
    // Sans profil, SIGHUP garde son effet par défaut (arrêt du processus).
    if config_path.is_some() {
        reload::watch(config_path, sets, raw_args, profile);
    }
    if let Some(root) = &args.prefix_ssd.directory {
        saragossa::decoder::configure_prefix_disk_cache(
            root,
            args.prefix_ssd.bytes()?,
            args.prefix_ssd.priority_tokens,
        )?;
    }
    model_selection::select_if_missing(&mut args).map_err(boxed_error)?;
    model_selection::resolve_models(&mut args).map_err(boxed_error)?;
    if let Some(rate) = args.max_tok_s {
        saragossa::set_decode_max_tokens_per_s(Some(rate));
    }
    if let Some(share) = args.gpu_share {
        auto_yield::set_configured(share);
    }
    if let Some(config) = args.auto_yield {
        auto_yield::spawn(config);
    }
    reload::set_launch_models(&args.models);
    reload::set_idle_unload_secs(args.idle_unload_secs);
    saragossa::force_resident_full_linear_decode();
    if args.engines > 1 {
        saragossa::runtime_flags::disable_shared_model_caches();
        eprintln!(
            "saragossa serve engines={}: cache de préfixe du décodeur et mémo de prefill TTS coupés (partagés entre moteurs)",
            args.engines
        );
    }
    let shared = shared::EngineShared::default();
    let mut state = state::ServeState::with_shared(&args, shared.clone());
    // NOTE: en relais séquentiel, le moteur précharge après la sortie du parent.
    #[cfg(unix)]
    let deferred = relay::defer_preload(args.preload);
    #[cfg(not(unix))]
    let deferred = false;
    if args.preload && !deferred {
        state.preload().map_err(boxed_error)?;
    }
    if let Some(addr) = args.tcp_addr() {
        let api_key = args
            .api_key
            .as_deref()
            .ok_or_else(|| error::ServeError::args("TCP requiert un bearer token"))
            .map_err(boxed_error)?;
        http::serve_tcp(&addr, api_key, &mut state, args.read_timeout).map_err(boxed_error)
    } else {
        http::serve_unix(&args.socket, &mut state, args.read_timeout).map_err(boxed_error)
    }
}

#[cfg(feature = "serve")]
fn boxed_error<E>(error: E) -> Box<dyn Error>
where
    E: Error + 'static,
{
    Box::new(error)
}
