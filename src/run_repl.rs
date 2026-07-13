//! REPL chat direct pour `saragossa run`.

use std::io::{self, BufRead, Write};
use std::path::Path;
use std::time::Instant;

use saragossa::{
    qwen_assistant_history_content, render_gemma4_chat, render_gemma_chat, render_qwen_chatml,
    ChatTemplateMessage, GenerationOptions, ModelAssets,
};

use crate::hf_resolve;
use crate::serve::error::ServeError;
use crate::serve::streaming::{CompletionStreamEvent, StreamingTextDetokenizer};
use crate::{cli_error, load_decoder_with_runtime, next_value, CliResult, RuntimeKind};

/// Lance le REPL interactif.
pub(super) fn run(args: impl IntoIterator<Item = String>) -> CliResult<()> {
    let Some(args) = RunArgs::parse(args)? else {
        print_run_help();
        return Ok(());
    };
    let model_dir = hf_resolve::resolve_model(&args.model)?;
    eprintln!("saragossa run loading {}", model_dir.display());
    let mut generator = ReplModel::load(&model_dir, &args)?;
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut reader = io::BufReader::new(stdin.lock());
    let mut writer = stdout.lock();
    run_repl_once(&mut reader, &mut writer, &mut generator)
}

fn print_run_help() {
    println!(
        "Usage: saragossa run <chemin|org/repo> [--backend cpu|metal] [--max-tokens N] [--temperature T] [--top-k N] [--top-p P] [--seed N]\n\nCtrl-D quitte le REPL. HF_TOKEN ou HUGGING_FACE_HUB_TOKEN est requis pour les modèles gated."
    );
}

#[derive(Debug)]
struct RunArgs {
    model: String,
    backend: RuntimeKind,
    max_tokens: usize,
    temperature: f32,
    top_p: Option<f32>,
    top_k: Option<usize>,
    seed: u64,
}

impl RunArgs {
    fn parse(args: impl IntoIterator<Item = String>) -> CliResult<Option<Self>> {
        let mut model: Option<String> = None;
        let mut backend = RuntimeKind::default_backend();
        let mut max_tokens = 512_usize;
        let mut temperature = 0.0_f32;
        let mut top_p: Option<f32> = None;
        let mut top_k: Option<usize> = None;
        let mut seed = 0_u64;
        let mut iter = args.into_iter();

        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--help" | "-h" => return Ok(None),
                "--backend" | "--runtime" => {
                    backend = RuntimeKind::parse(&next_value(&mut iter, "--backend")?)?;
                }
                "--max-tokens" => {
                    max_tokens = next_value(&mut iter, "--max-tokens")?.parse()?;
                }
                "--temperature" => {
                    temperature = next_value(&mut iter, "--temperature")?.parse()?;
                }
                "--top-p" => {
                    top_p = Some(next_value(&mut iter, "--top-p")?.parse()?);
                }
                "--top-k" => {
                    top_k = Some(next_value(&mut iter, "--top-k")?.parse()?);
                }
                "--seed" => {
                    seed = next_value(&mut iter, "--seed")?.parse()?;
                }
                other if other.starts_with('-') => {
                    return Err(cli_error(format!("argument run inconnu: {other}")));
                }
                other => {
                    if model.replace(other.to_string()).is_some() {
                        return Err(cli_error("saragossa run accepte un seul modèle"));
                    }
                }
            }
        }

        let Some(model) = model else {
            return Ok(None);
        };
        if max_tokens == 0 {
            return Err(cli_error("--max-tokens doit être > 0"));
        }
        if !temperature.is_finite() || temperature < 0.0 {
            return Err(cli_error("--temperature doit être un flottant positif"));
        }
        if let Some(top_p) = top_p {
            if !top_p.is_finite() || !(0.0..=1.0).contains(&top_p) || top_p <= 0.0 {
                return Err(cli_error("--top-p doit être dans ]0, 1]"));
            }
        }
        Ok(Some(Self {
            model,
            backend,
            max_tokens,
            temperature,
            top_p,
            top_k,
            seed,
        }))
    }
}

pub(super) trait ChatTurnGenerator {
    /// Génère un tour assistant en streamant les deltas dans `writer`.
    fn generate_turn(
        &mut self,
        history: &[ChatTemplateMessage],
        writer: &mut dyn Write,
    ) -> CliResult<String>;
}

/// Exécute une session REPL sur des flux injectables.
pub(super) fn run_repl_once<R, W, G>(
    reader: &mut R,
    writer: &mut W,
    generator: &mut G,
) -> CliResult<()>
where
    R: BufRead,
    W: Write,
    G: ChatTurnGenerator,
{
    let mut history = Vec::new();
    let mut line = String::new();
    loop {
        writer.write_all(b"> ")?;
        writer.flush()?;
        line.clear();
        let bytes = reader.read_line(&mut line)?;
        if bytes == 0 {
            return Ok(());
        }
        let user = line.trim_end_matches(['\r', '\n']).to_string();
        if user.trim().is_empty() {
            continue;
        }
        history.push(ChatTemplateMessage::new("user", user));
        let assistant = match generator.generate_turn(&history, writer) {
            Ok(assistant) => assistant,
            Err(error) => {
                // Une erreur récupérable sur un tour ne doit pas abattre la
                // session : on l'affiche et on retire le message user resté
                // sans réponse (sinon l'historique porterait un tour dépareillé).
                history.pop();
                writeln!(writer, "\nerreur: {error}")?;
                writer.flush()?;
                continue;
            }
        };
        writer.write_all(b"\n")?;
        writer.flush()?;
        history.push(ChatTemplateMessage::new("assistant", assistant));
    }
}

struct ReplModel {
    assets: ModelAssets,
    decoder: saragossa::CausalDecoder,
    max_tokens: usize,
    options: GenerationOptions,
}

impl ReplModel {
    fn load(model_dir: &Path, args: &RunArgs) -> CliResult<Self> {
        let _ = saragossa::apply_runtime_preset_for_model_dir(model_dir);
        let preset = saragossa::runtime_preset_for_model_dir(model_dir);
        let assets = ModelAssets::load_local(model_dir)?;
        let decoder = load_decoder_with_runtime(&assets, args.backend)?;
        let top_p = args.top_p.unwrap_or_else(|| {
            if args.temperature > f32::EPSILON {
                preset.map(|preset| preset.sampling_top_p).unwrap_or(1.0)
            } else {
                1.0
            }
        });
        let top_k = args.top_k.unwrap_or_else(|| {
            if args.temperature > f32::EPSILON {
                preset.map(|preset| preset.sampling_top_k).unwrap_or(0)
            } else {
                0
            }
        });
        let options = GenerationOptions {
            stop_token_ids: assets.stop_token_ids(),
            stop_sequences: Vec::new(),
            temperature: args.temperature,
            top_p,
            top_k,
            seed: args.seed,
            token_constraint: None,
        };
        Ok(Self {
            assets,
            decoder,
            max_tokens: args.max_tokens,
            options,
        })
    }
}

impl ChatTurnGenerator for ReplModel {
    fn generate_turn(
        &mut self,
        history: &[ChatTemplateMessage],
        writer: &mut dyn Write,
    ) -> CliResult<String> {
        let prompt_ids = encode_chat_prompt(&self.assets, history)?;
        if prompt_ids.is_empty() {
            return Err(cli_error("prompt token vide"));
        }
        let prefill_started = Instant::now();
        let prompt_state = self.decoder.prefill_prompt_state_uncached(&prompt_ids)?;
        let prefill = prefill_started.elapsed();
        let stop_texts = Vec::new();
        let mut detokenizer =
            StreamingTextDetokenizer::new(&self.assets, &stop_texts, self.max_tokens);
        let mut stream_error = None;
        let output = self
            .decoder
            .generate_greedy_timed_from_prompt_state_with_options_and_callback(
                prompt_state,
                prefill,
                self.max_tokens,
                &self.options,
                |token| {
                    if stream_error.is_some() {
                        return false;
                    }
                    let result = detokenizer.push_token(token, &mut |event| {
                        if let CompletionStreamEvent::Delta(delta) = event {
                            writer
                                .write_all(delta.as_bytes())
                                .map_err(|source| ServeError::io("écriture stdout", source))?;
                            writer
                                .flush()
                                .map_err(|source| ServeError::io("flush stdout", source))?;
                        }
                        Ok(())
                    });
                    match result {
                        Ok(()) => true,
                        Err(error) => {
                            stream_error = Some(error);
                            false
                        }
                    }
                },
            )?;
        if let Some(error) = stream_error {
            return Err(Box::new(error));
        }
        let generated = output
            .tokens
            .iter()
            .copied()
            .map(|id| {
                u32::try_from(id).map_err(|_| cli_error(format!("token généré hors plage: {id}")))
            })
            .collect::<CliResult<Vec<_>>>()?;
        let decoded = strip_empty_think(self.assets.decode_tokens(&generated, true)?);
        detokenizer.finish(&decoded, &mut |event| {
            if let CompletionStreamEvent::Delta(delta) = event {
                writer
                    .write_all(delta.as_bytes())
                    .map_err(|source| ServeError::io("écriture stdout", source))?;
            }
            Ok(())
        })?;
        writer.flush()?;
        Ok(decoded)
    }
}

fn encode_chat_prompt(
    assets: &ModelAssets,
    history: &[ChatTemplateMessage],
) -> CliResult<Vec<usize>> {
    let mut messages = history.to_vec();
    if !assets.config.is_gemma() {
        normalize_qwen_assistant_history(&mut messages);
    }
    let rendered = if assets.config.is_gemma4() {
        render_gemma4_chat(&messages, true, false)
    } else if assets.config.is_gemma() {
        render_gemma_chat(&messages, true)
    } else {
        render_qwen_chatml(&messages, true, false)
    };
    let ids = if assets.config.is_gemma() && !assets.config.is_gemma4() {
        assets.encode_prompt_with_special(&rendered)?
    } else {
        assets.encode_prompt(&rendered)?
    };
    ids.into_iter()
        .map(|id| usize::try_from(id).map_err(|_| cli_error(format!("token id hors plage: {id}"))))
        .collect()
}

fn normalize_qwen_assistant_history(messages: &mut [ChatTemplateMessage]) {
    for message in messages {
        if message.role != "assistant" {
            continue;
        }
        let content = message.content.take().unwrap_or_default();
        message.content = Some(qwen_assistant_history_content(&content, false));
    }
}

fn strip_empty_think(text: String) -> String {
    text.strip_prefix(saragossa::QWEN_EMPTY_THINK_BLOCK)
        .map_or(text.clone(), ToString::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[derive(Default)]
    struct FakeGenerator {
        prompts: Vec<Vec<ChatTemplateMessage>>,
    }

    impl ChatTurnGenerator for FakeGenerator {
        fn generate_turn(
            &mut self,
            history: &[ChatTemplateMessage],
            writer: &mut dyn Write,
        ) -> CliResult<String> {
            self.prompts.push(history.to_vec());
            let reply = format!("réponse {}", self.prompts.len());
            writer.write_all(reply.as_bytes())?;
            Ok(reply)
        }
    }

    #[test]
    fn repl_keeps_history_between_turns() {
        let input = b"bonjour\nsuite\n";
        let mut reader = Cursor::new(input);
        let mut writer = Vec::new();
        let mut generator = FakeGenerator::default();

        run_repl_once(&mut reader, &mut writer, &mut generator)
            .expect("invariant: REPL fake valide");

        assert_eq!(generator.prompts.len(), 2);
        assert_eq!(generator.prompts[0][0].role, "user");
        assert_eq!(generator.prompts[0][0].content.as_deref(), Some("bonjour"));
        assert_eq!(generator.prompts[1].len(), 3);
        assert_eq!(generator.prompts[1][1].role, "assistant");
        assert_eq!(
            generator.prompts[1][1].content.as_deref(),
            Some("réponse 1")
        );
        assert_eq!(
            String::from_utf8(writer).expect("invariant: sortie UTF-8"),
            "> réponse 1\n> réponse 2\n> "
        );
    }

    #[test]
    fn run_args_without_model_requests_help() {
        let args = RunArgs::parse(Vec::<String>::new()).expect("invariant: parse sans args");
        assert!(args.is_none());
    }
}
