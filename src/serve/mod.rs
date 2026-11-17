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
#[cfg(feature = "serve")]
mod responses;
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
/// Extrait `--config CHEMIN` (sinon `SARAGOSSA_CONFIG`) des arguments de serve.
fn take_config_arg(args: Vec<String>) -> error::ServeResult<(Option<String>, Vec<String>)> {
    let mut path = std::env::var("SARAGOSSA_CONFIG")
        .ok()
        .filter(|value| !value.is_empty());
    let mut rest = Vec::with_capacity(args.len());
    let mut iter = args.into_iter();
    while let Some(arg) = iter.next() {
        if arg == "--config" {
            path = Some(
                iter.next()
                    .ok_or_else(|| error::ServeError::args("--config attend un chemin"))?,
            );
        } else {
            rest.push(arg);
        }
    }
    Ok((path, rest))
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
    let (config_path, raw_args) = take_config_arg(raw_args).map_err(boxed_error)?;
    let profile = match config_path {
        Some(path) => saragossa::runtime_flags::Profile::load(std::path::Path::new(&path))?,
        None => saragossa::runtime_flags::Profile::default(),
    };
    profile.install();
    let mut args = ServeArgs::parse_with(&profile, raw_args).map_err(boxed_error)?;
    if let Some(root) = &args.prefix_ssd.directory {
        saragossa::decoder::configure_prefix_disk_cache(
            root,
            args.prefix_ssd.bytes()?,
            args.prefix_ssd.priority_tokens,
        )?;
    }
    model_selection::select_if_missing(&mut args).map_err(boxed_error)?;
    model_selection::resolve_models(&mut args).map_err(boxed_error)?;
    saragossa::force_resident_full_linear_decode();
    let mut state = state::ServeState::new(&args);
    if args.preload {
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
