//! Sous-commande `saragossa serve`.
//!
//! Concurrence d'E/S seulement (voir `dispatch`) : un thread d'accept et un
//! thread par connexion (plafond `SARAGOSSA_SERVE_MAX_CONNECTIONS`, 64) lisent
//! les requêtes et servent `/health` et `/v1/models` même pendant une
//! génération. L'inférence reste sérielle sur le thread moteur, le thread
//! principal qui possède `ServeState` ; les requêtes en attente passent par une
//! file bornée (`SARAGOSSA_SERVE_QUEUE`, 32). Au-delà de l'un des deux
//! plafonds, le serveur répond 503 immédiatement.
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
mod state;
pub(crate) mod streaming;
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
