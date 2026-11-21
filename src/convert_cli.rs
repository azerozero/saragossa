//! Sous-commandes `convert-qwen4exp` (FP8 officiel → MLX affine mixte q4e8)
//! et `convert` (HF BF16 officiel → affine u4/u8 gs64, tête MTP conservée).

use std::path::PathBuf;

use saragossa::devtools::convert::{
    convert_hf, convert_qwen4exp, ConvertOptions, HfConvertOptions,
};

use crate::{cli_error, CliResult};

const USAGE: &str = "usage: saragossa convert-qwen4exp --src <fp8> --dst <sortie> --source-revision <sha> [--source-repo <org/nom>] [--converter-commit <sha>] [--only <unité,…>] [--verify] [--plan]";

pub(crate) fn run(args: impl Iterator<Item = String>) -> CliResult<()> {
    let mut src = None;
    let mut dst = None;
    let mut revision = None;
    let mut repo = None;
    let mut commit = None;
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
            "--source-repo" => repo = Some(value("--source-repo")?),
            "--converter-commit" => commit = Some(value("--converter-commit")?),
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
        source_repo: repo,
        converter_commit: commit,
        only,
        verify,
        plan_only,
    };
    let report = convert_qwen4exp(&options)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

const HF_USAGE: &str = "usage: saragossa convert <dossier bf16 HF> <sortie> [--bits 4|8] [--minmax] [--keep-8bit motif,…] [--keep-bf16 motif,…] [--verify]\n\
défaut : u4 gs64 aux moindres carrés ; un motif est une sous-chaîne du nom de tenseur HF";

fn patterns(value: &str) -> Vec<String> {
    value
        .split(',')
        .filter(|item| !item.is_empty())
        .map(str::to_string)
        .collect()
}

pub(crate) fn run_hf(args: impl Iterator<Item = String>) -> CliResult<()> {
    let mut positional = Vec::new();
    let mut options = HfConvertOptions {
        src: PathBuf::new(),
        dst: PathBuf::new(),
        bits: 4,
        minmax: false,
        keep_8bit: Vec::new(),
        keep_bf16: Vec::new(),
        verify: false,
    };
    let mut args = args.peekable();
    while let Some(arg) = args.next() {
        let mut value = |flag: &str| {
            args.next()
                .ok_or_else(|| cli_error(format!("{flag} attend une valeur\n{HF_USAGE}")))
        };
        match arg.as_str() {
            "--bits" => {
                options.bits = value("--bits")?
                    .parse()
                    .map_err(|_| cli_error(format!("--bits : 4 ou 8\n{HF_USAGE}")))?
            }
            "--minmax" => options.minmax = true,
            "--keep-8bit" => options.keep_8bit = patterns(&value("--keep-8bit")?),
            "--keep-bf16" => options.keep_bf16 = patterns(&value("--keep-bf16")?),
            "--verify" => options.verify = true,
            "--help" | "-h" => {
                println!("{HF_USAGE}");
                return Ok(());
            }
            other if other.starts_with('-') => {
                return Err(cli_error(format!("argument inconnu: {other}\n{HF_USAGE}")))
            }
            other => positional.push(PathBuf::from(other)),
        }
    }
    let [src, dst] = <[PathBuf; 2]>::try_from(positional)
        .map_err(|_| cli_error(format!("source et sortie requises\n{HF_USAGE}")))?;
    options.src = src;
    options.dst = dst;
    let report = convert_hf(&options)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
