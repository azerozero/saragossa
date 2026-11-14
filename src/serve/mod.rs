//! Sous-commande `saragossa serve`.
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
//! Le parseur HTTP accepte uniquement les corps à `Content-Length`.
//! `Transfer-Encoding: chunked` n'est pas pris en charge.

mod anthropic;
mod args;
mod audio;
mod cache;
mod dispatch;
mod embeddings;
pub(crate) mod error;
mod flint_output;
mod http;
mod model_selection;
pub(crate) mod mtp;
mod protocol;
mod responses;
mod state;
pub(crate) mod streaming;
mod tool_output;
mod warmup;

use std::error::Error;

use args::ServeArgs;

use super::CliResult;

/// Lance le serveur OpenAI-compatible local.
pub(super) fn run(args: impl IntoIterator<Item = String>) -> CliResult<()> {
    let raw_args = args.into_iter().collect::<Vec<_>>();
    if raw_args
        .iter()
        .any(|arg| matches!(arg.as_str(), "--help" | "-h"))
    {
        println!("{}", args::help_text());
        return Ok(());
    }
    let mut args = ServeArgs::parse(raw_args).map_err(boxed_error)?;
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

fn boxed_error<E>(error: E) -> Box<dyn Error>
where
    E: Error + 'static,
{
    Box::new(error)
}
