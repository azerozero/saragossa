//! Sous-commande `convert-qwen4exp` : FP8 officiel → MLX affine mixte q4e8.

use std::path::PathBuf;

use saragossa::devtools::convert::{convert_qwen4exp, ConvertOptions};

use crate::{cli_error, CliResult};

const USAGE: &str = "usage: saragossa convert-qwen4exp --src <fp8> --dst <sortie> --source-revision <sha> [--only <unité,…>] [--verify] [--plan]";

pub(crate) fn run(args: impl Iterator<Item = String>) -> CliResult<()> {
    let mut src = None;
    let mut dst = None;
    let mut revision = None;
    let mut only = None;
    let mut verify = false;
    let mut plan_only = false;
    let mut args = args.peekable();
    while let Some(arg) = args.next() {
        let mut value = |flag: &str| {
            args.next()
                .ok_or_else(|| cli_error(format!("{flag} attend une valeur\n{USAGE}")))
        };
        match arg.as_str() {
            "--src" => src = Some(PathBuf::from(value("--src")?)),
            "--dst" => dst = Some(PathBuf::from(value("--dst")?)),
            "--source-revision" => revision = Some(value("--source-revision")?),
            "--only" => {
                only = Some(
                    value("--only")?
                        .split(',')
                        .map(str::to_string)
                        .collect::<Vec<_>>(),
                )
            }
            "--verify" => verify = true,
            "--plan" => plan_only = true,
            "--help" | "-h" => {
                println!("{USAGE}");
                return Ok(());
            }
            other => return Err(cli_error(format!("argument inconnu: {other}\n{USAGE}"))),
        }
    }
    let options = ConvertOptions {
        src: src.ok_or_else(|| cli_error(format!("--src requis\n{USAGE}")))?,
        dst: dst.ok_or_else(|| cli_error(format!("--dst requis\n{USAGE}")))?,
        source_revision: revision
            .ok_or_else(|| cli_error(format!("--source-revision requis\n{USAGE}")))?,
        only,
        verify,
        plan_only,
    };
    let report = convert_qwen4exp(&options)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
