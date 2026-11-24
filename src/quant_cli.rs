//! Sous-commandes `paired` et `kl-eval` de la chaîne de quantification
//! (portages de `tools/quant/paired.py` et `tools/quant/kl_eval.py`).

use std::path::PathBuf;

use saragossa::devtools::{kl_eval, paired};
use saragossa::ModelAssets;

use crate::{cli_error, load_decoder_with_runtime, CliResult, RuntimeKind};

const PAIRED_USAGE: &str = "usage: saragossa paired diff <candidat.json> <référence.json> [--metric kl_mean]\n       saragossa paired factorial <A0F0.json> <A0F1.json> <A1F0.json> <A1F1.json> [--metric kl_mean]";
const KL_USAGE: &str = "usage: saragossa kl-eval --reference <modèle bf16> --candidate <modèle quantifié> [--prompts <fichier>] [--backend cpu|metal]";

pub(crate) fn run_paired(args: impl Iterator<Item = String>) -> CliResult<()> {
    let mut metric = "kl_mean".to_string();
    let mut positional = Vec::new();
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--metric" => {
                metric = args.next().ok_or_else(|| {
                    cli_error(format!("--metric attend une valeur\n{PAIRED_USAGE}"))
                })?
            }
            "--help" | "-h" => {
                println!("{PAIRED_USAGE}");
                return Ok(());
            }
            _ => positional.push(PathBuf::from(arg)),
        }
    }
    let json = match positional.as_slice() {
        [action, candidate, reference] if action.as_os_str() == "diff" => {
            paired::to_json(&paired::diff(candidate, reference, &metric)?)?
        }
        [action, a, b, c, d] if action.as_os_str() == "factorial" => paired::to_json(
            &paired::factorial([a, b, c, d].map(PathBuf::as_path), &metric)?,
        )?,
        _ => return Err(cli_error(PAIRED_USAGE)),
    };
    println!("{json}");
    Ok(())
}

pub(crate) fn run_kl_eval(args: impl Iterator<Item = String>) -> CliResult<()> {
    let (mut reference, mut candidate, mut prompts) = (None, None, None);
    let mut backend = RuntimeKind::default_backend();
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        let mut value = |flag: &str| {
            args.next()
                .ok_or_else(|| cli_error(format!("{flag} attend une valeur\n{KL_USAGE}")))
        };
        match arg.as_str() {
            "--reference" => reference = Some(PathBuf::from(value("--reference")?)),
            "--candidate" => candidate = Some(PathBuf::from(value("--candidate")?)),
            "--prompts" => prompts = Some(PathBuf::from(value("--prompts")?)),
            "--backend" => backend = RuntimeKind::parse(&value("--backend")?)?,
            "--help" | "-h" => {
                println!("{KL_USAGE}");
                return Ok(());
            }
            other => return Err(cli_error(format!("argument inconnu: {other}\n{KL_USAGE}"))),
        }
    }
    let reference =
        reference.ok_or_else(|| cli_error(format!("--reference requis\n{KL_USAGE}")))?;
    let candidate =
        candidate.ok_or_else(|| cli_error(format!("--candidate requis\n{KL_USAGE}")))?;
    for (label, path) in [("référence", &reference), ("candidat", &candidate)] {
        if !path.is_dir() {
            return Err(cli_error(format!(
                "modèle {label} introuvable: {}",
                path.display()
            )));
        }
    }
    let prompts = kl_eval::load_prompts(prompts.as_deref())?;
    // Seuls les logits de la référence restent en mémoire (4·V·ΣT octets) ;
    // ceux du candidat sont comparés puis libérés prompt par prompt.
    println!("[1/2] forward référence {}...", reference.display());
    let mut reference_logits = Vec::with_capacity(prompts.len());
    for_each_logits(&reference, &prompts, backend, |_, logits| {
        reference_logits.push(logits);
        Ok(())
    })?;
    println!("[2/2] forward candidat {}...", candidate.display());
    let (mut divergences, mut agreements) = (Vec::new(), Vec::new());
    for_each_logits(&candidate, &prompts, backend, |index, logits| {
        let (divergence, agreement) = kl_eval::kl_and_agreement(&reference_logits[index], &logits)?;
        println!("prompt {index}: KL={divergence:.4}  accord_top1={agreement:.3}");
        divergences.push(divergence);
        agreements.push(agreement);
        Ok(())
    })?;
    let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len() as f64;
    println!(
        "\nmoyenne: KL={:.4}  accord_top1={:.3}",
        mean(&divergences),
        mean(&agreements)
    );
    println!("repères: KL < 0,05 excellent, < 0,15 bon, > 0,3 dégradé ; accord_top1 > 0,95 quasi-sans-perte");
    Ok(())
}

/// Logits `[T, V]` de chaque prompt, passés à `visit` dans l'ordre ; le
/// modèle est libéré avant le suivant : un bf16 de 27B et son quantifié ne
/// tiennent pas ensemble en mémoire.
fn for_each_logits(
    model_dir: &std::path::Path,
    prompts: &[String],
    backend: RuntimeKind,
    mut visit: impl FnMut(usize, saragossa::Tensor) -> CliResult<()>,
) -> CliResult<()> {
    let assets = ModelAssets::load_local(model_dir)?;
    let decoder = load_decoder_with_runtime(&assets, backend)?;
    for (index, prompt) in prompts.iter().enumerate() {
        // Comme `tokenizer.encode` de mlx-lm : tokens spéciaux (BOS) inclus.
        let ids: Vec<usize> = assets
            .encode_prompt_with_special(prompt)?
            .into_iter()
            .map(|id| id as usize)
            .collect();
        visit(index, decoder.teacher_forced_logits(&ids)?)?;
    }
    Ok(())
}
