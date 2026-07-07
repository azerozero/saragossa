//! Sous-commande `saragossa serve`.
//!
//! Le serveur est volontairement mono-thread: il cible un usage local
//! mono-utilisateur, donc une requête longue bloque les suivantes
//! (head-of-line blocking assumé).
//!
//! Le parseur HTTP accepte uniquement les corps à `Content-Length`.
//! `Transfer-Encoding: chunked` n'est pas pris en charge.

mod args;
mod error;
mod http;
mod protocol;
mod state;

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
    let args = ServeArgs::parse(raw_args).map_err(boxed_error)?;
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
