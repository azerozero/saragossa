//! `saragossa config check|print` : inspection locale d'un profil, sans modèle ni GPU.

use saragossa::runtime_flags::{Resolved, PRESETS};

use crate::{cli_error, CliResult};

const USAGE: &str =
    "Usage: saragossa config check FILE|PRESET [--set section.cle=valeur]...\n       \
     saragossa config print [FILE|PRESET] [--set section.cle=valeur]...\n";

/// Valide (`check`) ou affiche (`print`) la configuration effective.
pub(crate) fn run(args: impl IntoIterator<Item = String>) -> CliResult<()> {
    let mut iter = args.into_iter();
    let sub = iter.next().unwrap_or_default();
    let (mut source, mut sets) = (None, Vec::new());
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--set" => sets.push(
                iter.next()
                    .ok_or_else(|| cli_error("--set attend une valeur"))?,
            ),
            _ if arg.starts_with('-') => {
                return Err(cli_error(format!("option inconnue {arg}\n{USAGE}")))
            }
            _ if source.is_none() => source = Some(arg),
            _ => return Err(cli_error(format!("un seul fichier attendu\n{USAGE}"))),
        }
    }
    match sub.as_str() {
        "check" => {
            let name = source.ok_or_else(|| cli_error(USAGE))?;
            Resolved::resolve(Some(&name), &sets)?;
            println!(
                "{name}: valide (presets: {})",
                PRESETS.map(|(n, _)| n).join(", ")
            );
        }
        "print" => print!("{}", Resolved::resolve(source.as_deref(), &sets)?.report()),
        _ => return Err(cli_error(USAGE)),
    }
    Ok(())
}
