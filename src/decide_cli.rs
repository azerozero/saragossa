//! Transport JSON local des décisions texte.
use crate::{cli_error, CliResult};
use saragossa::decision::DecisionModel;
use std::io::Write;

pub(crate) fn run(args: impl Iterator<Item = String>) -> CliResult<()> {
    let mut args = args.peekable();
    if args.peek().is_some_and(|a| a == "--help" || a == "-h") {
        println!("saragossa decide --model-dir DOSSIER [--cpu]\nRequêtes JSON {{state,questions}} sur stdin (multiligne ou JSON Lines) ; une réponse par ligne.\nTexte joint uniquement, maximum 2048 tokens, sans TTA ni thinker.");
        return Ok(());
    }
    let mut path = None;
    let mut metal = true;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model-dir" => {
                path = Some(
                    args.next()
                        .ok_or_else(|| cli_error("--model-dir exige un chemin"))?,
                )
            }
            "--cpu" => metal = false,
            _ => return Err(cli_error(format!("argument decide inconnu : {arg}"))),
        }
    }
    let path = path.ok_or_else(|| cli_error("--model-dir requis"))?;
    let model = DecisionModel::load(path, metal)?;
    let mut output = std::io::stdout().lock();
    let stream = serde_json::Deserializer::from_reader(std::io::stdin().lock())
        .into_iter::<serde_json::Value>();
    for request in stream {
        let result = model.decide(&request?).map_err(|e| e.to_string());
        let response = result.unwrap_or_else(|error| serde_json::json!({"error":error}));
        serde_json::to_writer(&mut output, &response)?;
        writeln!(output)?;
        output.flush()?;
    }
    Ok(())
}
