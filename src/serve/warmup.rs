//! Warmup du décodeur lors du chargement d'un modèle servi.

use std::env;
use std::time::{Duration, Instant};

use saragossa::{CausalDecoder, ModelAssets};

use super::error::{ServeError, ServeResult};
use crate::RuntimeKind;

const WARMUP_PROMPT: &str = "warmup";
const DEFAULT_WARMUP_PASSES: usize = 2;
const DEFAULT_WARMUP_PROMPT_TOKENS: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct WarmupConfig {
    passes: usize,
    prompt_tokens: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct WarmupReport {
    elapsed: Duration,
    passes: usize,
    prompt_tokens: usize,
}

/// Réchauffe un décodeur Metal avec un prompt court.
///
/// # Errors
///
/// Renvoie une erreur si l'encodage ou un prefill de warmup échoue.
pub(super) fn decoder(
    decoder: &CausalDecoder,
    assets: &ModelAssets,
    backend: RuntimeKind,
    model_id: &str,
) -> ServeResult<()> {
    let Some(config) = config_from_env(backend) else {
        return Ok(());
    };
    let prompt = assets.encode_prompt_with_special(WARMUP_PROMPT)?;
    let prompt = prompt
        .into_iter()
        .map(|id| {
            usize::try_from(id).map_err(|_| {
                ServeError::args(format!(
                    "token warmup hors plage pour cette plateforme: {id}"
                ))
            })
        })
        .collect::<ServeResult<Vec<_>>>()?;

    let report = run(config, &prompt, |tokens| {
        let _ = decoder.prefill_cache_uncached(tokens)?;
        Ok(())
    })?;
    eprintln!(
        "saragossa serve warmup model={model_id} passes={} prompt_tokens={} elapsed_ms={}",
        report.passes,
        report.prompt_tokens,
        report.elapsed.as_millis()
    );
    Ok(())
}

fn config_from_env(backend: RuntimeKind) -> Option<WarmupConfig> {
    warmup_config(
        backend,
        env::var("RETI_RUST_WARMUP").ok().as_deref(),
        positive_env_usize("RETI_RUST_WARMUP_PASSES", DEFAULT_WARMUP_PASSES),
        positive_env_usize(
            "RETI_RUST_WARMUP_PROMPT_TOKENS",
            DEFAULT_WARMUP_PROMPT_TOKENS,
        ),
    )
}

fn warmup_config(
    backend: RuntimeKind,
    enabled_value: Option<&str>,
    passes: usize,
    prompt_tokens: usize,
) -> Option<WarmupConfig> {
    if backend != RuntimeKind::Metal || warmup_disabled(enabled_value) {
        return None;
    }
    Some(WarmupConfig {
        passes,
        prompt_tokens,
    })
}

fn warmup_disabled(value: Option<&str>) -> bool {
    value.is_some_and(|value| {
        value == "0" || value.eq_ignore_ascii_case("false") || value.eq_ignore_ascii_case("off")
    })
}

fn positive_env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn run(
    config: WarmupConfig,
    prompt: &[usize],
    mut prefill: impl FnMut(&[usize]) -> ServeResult<()>,
) -> ServeResult<WarmupReport> {
    if prompt.is_empty() {
        return Err(ServeError::args("prompt token vide pour le warmup serve"));
    }
    let started = Instant::now();
    let prompt = &prompt[..prompt.len().min(config.prompt_tokens)];
    for _ in 0..config.passes {
        prefill(prompt)?;
    }
    Ok(WarmupReport {
        elapsed: started.elapsed(),
        passes: config.passes,
        prompt_tokens: prompt.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warmup_is_metal_only_and_honors_kill_switch() {
        assert_eq!(
            warmup_config(RuntimeKind::Cpu, None, 2, 32),
            None,
            "le CPU ne doit jamais exécuter le warmup"
        );
        for disabled in ["0", "false", "FALSE", "off", "OFF"] {
            assert_eq!(
                warmup_config(RuntimeKind::Metal, Some(disabled), 2, 32),
                None,
                "le kill-switch {disabled} doit désactiver le warmup"
            );
        }
        assert!(warmup_config(RuntimeKind::Metal, None, 2, 32).is_some());
        assert!(warmup_config(RuntimeKind::Metal, Some("1"), 2, 32).is_some());
    }

    #[test]
    fn warmup_runs_every_pass_on_the_short_prompt() {
        let config = WarmupConfig {
            passes: 3,
            prompt_tokens: 2,
        };
        let mut seen = Vec::new();

        let report = run(config, &[10, 11, 12], |tokens| {
            seen.push(tokens.to_vec());
            Ok(())
        })
        .expect("invariant: préfill factice infaillible");

        assert_eq!(seen, vec![vec![10, 11], vec![10, 11], vec![10, 11]]);
        assert_eq!(report.passes, 3);
        assert_eq!(report.prompt_tokens, 2);
    }
}
