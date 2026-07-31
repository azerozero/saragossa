//! Sélection et résolution des modèles servis.

use std::io::{self, BufRead, IsTerminal, Write};
use std::path::Path;

use super::args::{parse_model_registration, ServeArgs, ServeModelConfig};
use super::error::{ServeError, ServeResult};
use crate::hf_resolve::{self, CachedModel};

const DEFAULT_MODELS_DIR: &str = "models";

/// Sélectionne un modèle quand aucun `--model` n'a été fourni.
pub(super) fn select_if_missing(args: &mut ServeArgs) -> ServeResult<()> {
    if !args.models.is_empty() {
        return Ok(());
    }
    let models = discover_models()?;
    if !io::stdin().is_terminal() {
        return Err(ServeError::args(non_tty_message(&models)));
    }

    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut input = stdin.lock();
    let mut output = stdout.lock();
    let model = prompt_for_model(&models, &mut input, &mut output)?;
    args.add_model(model)
}

/// Résout les références HF nues et journalise chaque enregistrement.
pub(super) fn resolve_models(args: &mut ServeArgs) -> ServeResult<()> {
    for model in &mut args.models {
        let Some(repo) = model.hf_repo.take() else {
            eprintln!(
                "saragossa serve model enregistré alias={} path={} source=local",
                model.id,
                model.path.display()
            );
            continue;
        };
        eprintln!(
            "saragossa serve résolution Hugging Face alias={} repo={repo}",
            model.id
        );
        let resolved = hf_resolve::resolve_model(&repo).map_err(|error| {
            ServeError::args(format!("résolution du modèle HF {repo}: {error}"))
        })?;
        eprintln!(
            "saragossa serve modèle HF résolu alias={} repo={repo} path={}",
            model.id,
            resolved.display()
        );
        model.path = resolved;
    }
    Ok(())
}

fn discover_models() -> ServeResult<Vec<CachedModel>> {
    let cache_dir = hf_resolve::hf_cache_dir_from_env()
        .map_err(|error| ServeError::args(format!("découverte du cache HF: {error}")))?;
    hf_resolve::discover_local_models(&cache_dir, Path::new(DEFAULT_MODELS_DIR))
        .map_err(|error| ServeError::args(format!("découverte des modèles locaux: {error}")))
}

fn prompt_for_model(
    models: &[CachedModel],
    input: &mut impl BufRead,
    output: &mut impl Write,
) -> ServeResult<ServeModelConfig> {
    write_model_list(models, output)?;
    if models.is_empty() {
        write!(output, "Quel modèle servir ? [org/repo] ")?;
    } else {
        write!(
            output,
            "Quel modèle servir ? [1-{} ou org/repo] ",
            models.len()
        )?;
    }
    output
        .flush()
        .map_err(|source| ServeError::io("flush du picker modèle", source))?;

    let mut answer = String::new();
    input
        .read_line(&mut answer)
        .map_err(|source| ServeError::io("lecture du picker modèle", source))?;
    let answer = answer.trim();
    if let Ok(index) = answer.parse::<usize>() {
        let selected = index
            .checked_sub(1)
            .and_then(|index| models.get(index))
            .ok_or_else(|| {
                ServeError::args(format!(
                    "choix modèle hors plage: {answer} (attendu 1-{})",
                    models.len()
                ))
            })?;
        return Ok(config_from_discovered(selected));
    }
    if !hf_resolve::is_hf_model_id(answer) {
        return Err(ServeError::args(format!(
            "choix modèle invalide: {answer:?} — entrez un numéro ou un id HF org/repo"
        )));
    }
    parse_model_registration(answer)
}

fn config_from_discovered(model: &CachedModel) -> ServeModelConfig {
    let id = model
        .id
        .rsplit('/')
        .next()
        .filter(|segment| !segment.is_empty())
        .unwrap_or(model.id.as_str())
        .to_string();
    ServeModelConfig {
        id,
        path: model.snapshot.clone(),
        hf_repo: None,
    }
}

fn write_model_list(models: &[CachedModel], output: &mut impl Write) -> ServeResult<()> {
    if models.is_empty() {
        writeln!(
            output,
            "Aucun modèle local trouvé dans le cache Hugging Face ou {DEFAULT_MODELS_DIR}/."
        )?;
        return Ok(());
    }
    writeln!(output, "Modèles locaux disponibles :")?;
    for (index, model) in models.iter().enumerate() {
        writeln!(
            output,
            "  {}. {} ({})",
            index + 1,
            model.id,
            model.snapshot.display()
        )?;
    }
    Ok(())
}

fn non_tty_message(models: &[CachedModel]) -> String {
    let mut message = String::from(
        "aucun --model fourni et stdin n'est pas un TTY; sélection interactive impossible\n",
    );
    if models.is_empty() {
        message.push_str("modèles locaux trouvés: aucun\n");
    } else {
        message.push_str("modèles locaux trouvés:\n");
        for model in models {
            message.push_str(&format!(
                "  - {} ({})\n",
                model.id,
                model.snapshot.display()
            ));
        }
    }
    message.push_str(
        "exemples:\n\
         saragossa serve --model models/Qwen3.6-27B-greedy36\n\
         saragossa serve --model Qwen/Qwen3-4B",
    );
    message
}

impl From<io::Error> for ServeError {
    fn from(source: io::Error) -> Self {
        Self::io("écriture du picker modèle", source)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use std::path::PathBuf;

    fn discovered(id: &str, path: &str) -> CachedModel {
        CachedModel {
            id: id.to_string(),
            snapshot: PathBuf::from(path),
            size_bytes: 42,
        }
    }

    #[test]
    fn picker_selects_numbered_local_model() {
        let models = vec![
            discovered("org/alpha", "/cache/alpha"),
            discovered("local-beta", "models/local-beta"),
        ];
        let mut input = Cursor::new(b"2\n");
        let mut output = Vec::new();

        let selected = prompt_for_model(&models, &mut input, &mut output)
            .expect("invariant: choix numérique valide");

        assert_eq!(selected.id, "local-beta");
        assert_eq!(selected.path, PathBuf::from("models/local-beta"));
        assert_eq!(selected.hf_repo, None);
        assert!(String::from_utf8(output)
            .expect("invariant: sortie UTF-8")
            .contains("Quel modèle servir ? [1-2 ou org/repo]"));
    }

    #[test]
    fn picker_accepts_hf_id() {
        let mut input = Cursor::new(b"Qwen/Qwen3-4B\n");
        let mut output = Vec::new();

        let selected =
            prompt_for_model(&[], &mut input, &mut output).expect("invariant: id HF valide");

        assert_eq!(selected.id, "Qwen3-4B");
        assert_eq!(selected.hf_repo.as_deref(), Some("Qwen/Qwen3-4B"));
    }

    #[test]
    fn non_tty_error_lists_models_and_two_examples() {
        let message = non_tty_message(&[discovered("org/alpha", "/cache/alpha")]);

        assert!(message.contains("stdin n'est pas un TTY"));
        assert!(message.contains("org/alpha (/cache/alpha)"));
        assert!(message.contains("--model models/Qwen3.6-27B-greedy36"));
        assert!(message.contains("--model Qwen/Qwen3-4B"));
    }
}
