//! Etat d'inférence longue durée du serveur local.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use saragossa::decoder::GenerationTimings;
use saragossa::{
    qwen_assistant_history_content, render_gemma4_chat, render_gemma_chat, render_qwen_chatml,
    CausalDecoder, ChatTemplateMessage, GenerationOptions, ModelAssets, RuntimePreset,
};

use super::args::{ServeArgs, ServeModelConfig};
use super::error::{ServeError, ServeResult};
use super::protocol::{ChatCompletionRequest, ModelInfo, ModelsResponse, Usage};
use crate::RuntimeKind;

const PREFIX_CACHE_CAP_ENV: &str = "RETI_RUST_PREFIX_CACHE_CAP";

/// Réponse d'inférence prête à sérialiser en OpenAI.
#[derive(Debug)]
pub(super) struct ServedCompletion {
    /// Identifiant OpenAI du modèle.
    pub(super) model: String,
    /// Texte assistant généré.
    pub(super) content: String,
    /// Raison de fin OpenAI.
    pub(super) finish_reason: &'static str,
    /// Usage token OpenAI.
    pub(super) usage: Usage,
    /// Nombre de tokens prompt.
    pub(super) prompt_tokens: usize,
    /// Nombre de tokens repris d'un prompt déjà vu.
    pub(super) reused_prefix_tokens: usize,
    /// Timings du moteur.
    pub(super) timings: GenerationTimings,
    /// Mur total de génération.
    pub(super) total: Duration,
}

impl ServedCompletion {
    /// Renvoie les headers de diagnostic non standards.
    pub(super) fn metric_headers(&self) -> Vec<(&'static str, String)> {
        vec![
            (
                "x-saragossa-prefill-ms",
                self.timings.prefill.as_millis().to_string(),
            ),
            (
                "x-saragossa-decode-ms",
                self.timings.decode.as_millis().to_string(),
            ),
            (
                "x-saragossa-decode-tokens",
                self.timings.decode_tokens.to_string(),
            ),
            ("x-saragossa-prompt-tokens", self.prompt_tokens.to_string()),
            (
                "x-saragossa-reused-prefix-tokens",
                self.reused_prefix_tokens.to_string(),
            ),
            ("x-saragossa-total-ms", self.total.as_millis().to_string()),
        ]
    }
}

/// Etat mutable du serveur. Les decodeurs restent vivants entre requêtes.
pub(super) struct ServeState {
    models: Vec<ModelSlot>,
    max_tokens_cap: usize,
}

impl ServeState {
    /// Construit l'état à partir des chemins CLI, sans charger les poids.
    pub(super) fn new(args: &ServeArgs) -> Self {
        Self {
            models: args
                .models
                .iter()
                .map(|model| ModelSlot::new(model, args.backend))
                .collect(),
            max_tokens_cap: args.max_tokens_cap,
        }
    }

    /// Renvoie le plafond serveur du budget de génération.
    pub(super) fn max_tokens_cap(&self) -> usize {
        self.max_tokens_cap
    }

    /// Charge tous les modèles configurés.
    pub(super) fn preload(&mut self) -> ServeResult<()> {
        for model in &mut self.models {
            model.ensure_loaded()?;
        }
        Ok(())
    }

    /// Réponse `/v1/models`.
    pub(super) fn models_response(&self) -> ModelsResponse {
        ModelsResponse::new(
            self.models
                .iter()
                .map(|model| ModelInfo::new(model.id.clone()))
                .collect(),
        )
    }

    /// Exécute une complétion chat OpenAI.
    pub(super) fn complete(
        &mut self,
        request: ChatCompletionRequest,
    ) -> ServeResult<ServedCompletion> {
        let max_tokens_cap = self.max_tokens_cap;
        let max_tokens = request.max_tokens_capped(max_tokens_cap)?;
        let model = self
            .models
            .iter_mut()
            .find(|model| model.id == request.model)
            .ok_or_else(|| ServeError::UnknownModel(request.model.clone()))?;
        model.complete(request, max_tokens)
    }
}

struct ModelSlot {
    id: String,
    path: PathBuf,
    backend: RuntimeKind,
    loaded: Option<LoadedModel>,
}

impl ModelSlot {
    fn new(model: &ServeModelConfig, backend: RuntimeKind) -> Self {
        Self {
            id: model.id.clone(),
            path: model.path.clone(),
            backend,
            loaded: None,
        }
    }

    fn ensure_loaded(&mut self) -> ServeResult<&mut LoadedModel> {
        if self.loaded.is_none() {
            eprintln!(
                "saragossa serve loading model={} path={}",
                self.id,
                self.path.display()
            );
            let _ = saragossa::apply_runtime_preset_for_model_dir(&self.path);
            let preset = saragossa::runtime_preset_for_model_dir(&self.path);
            let assets = ModelAssets::load_local(&self.path)?;
            let decoder = load_decoder_with_runtime(&assets, self.backend)?;
            self.loaded = Some(LoadedModel {
                id: self.id.clone(),
                assets,
                decoder,
                preset,
                prompt_prefixes: Vec::new(),
            });
        }
        self.loaded.as_mut().ok_or_else(|| {
            ServeError::args(format!(
                "modèle {} non chargé après initialisation",
                self.id
            ))
        })
    }

    fn complete(
        &mut self,
        request: ChatCompletionRequest,
        max_tokens: usize,
    ) -> ServeResult<ServedCompletion> {
        self.ensure_loaded()?.complete(request, max_tokens)
    }
}

struct LoadedModel {
    id: String,
    assets: ModelAssets,
    decoder: CausalDecoder,
    preset: Option<RuntimePreset>,
    prompt_prefixes: Vec<PromptPrefixEntry>,
}

#[derive(Clone)]
struct PromptPrefixEntry {
    rendered: String,
    tokens: Vec<usize>,
}

struct PromptEncoding {
    rendered: String,
    tokens: Vec<usize>,
    reused_prefix_tokens: usize,
}

impl LoadedModel {
    fn complete(
        &mut self,
        request: ChatCompletionRequest,
        max_tokens: usize,
    ) -> ServeResult<ServedCompletion> {
        let stop_texts = request.stop_texts();
        let prompt = self.prompt_encoding(&request)?;
        let prompt_tokens = prompt.tokens.len();
        let reused_prefix_tokens = prompt.reused_prefix_tokens;
        let options = self.generation_options(&request, &stop_texts)?;
        let started = Instant::now();
        let output = self.decoder.generate_greedy_timed_with_options(
            &prompt.tokens,
            max_tokens,
            &options,
        )?;
        let total = started.elapsed();
        let finish_reason = if output.tokens.len() >= max_tokens {
            "length"
        } else {
            "stop"
        };
        let generated = tokens_to_u32(&output.tokens)?;
        let content = strip_text_stops(
            strip_empty_think(self.assets.decode_tokens(&generated, true)?),
            &stop_texts,
        );
        eprintln!(
            "saragossa serve completion model={} prompt_tokens={} completion_tokens={} prefill_ms={} decode_ms={} total_ms={} reused_prefix_tokens={} prefix_cache=persistent-decoder",
            self.id,
            prompt_tokens,
            output.tokens.len(),
            output.timings.prefill.as_millis(),
            output.timings.decode.as_millis(),
            total.as_millis(),
            reused_prefix_tokens
        );
        self.remember_prompt_prefix(prompt.rendered, prompt.tokens);
        Ok(ServedCompletion {
            model: self.id.clone(),
            content,
            finish_reason,
            usage: Usage::new(prompt_tokens, output.tokens.len()),
            prompt_tokens,
            reused_prefix_tokens,
            timings: output.timings,
            total,
        })
    }

    fn prompt_encoding(&self, request: &ChatCompletionRequest) -> ServeResult<PromptEncoding> {
        let mut messages = request.template_messages();
        if !self.assets.config.is_gemma() {
            normalize_qwen_assistant_history(&mut messages);
        }
        let rendered = if self.assets.config.is_gemma4() {
            render_gemma4_chat(&messages, true, false)
        } else if self.assets.config.is_gemma() {
            render_gemma_chat(&messages, true)
        } else {
            render_qwen_chatml(&messages, true, false)
        };
        let full_tokens = self.encode_full_rendered_prompt(&rendered)?;
        prompt_encoding_with_verified_prefix(
            &self.id,
            &self.prompt_prefixes,
            rendered,
            full_tokens,
            |suffix| self.encode_rendered_suffix(suffix),
        )
    }

    fn generation_options(
        &self,
        request: &ChatCompletionRequest,
        stop_texts: &[String],
    ) -> ServeResult<GenerationOptions> {
        let temperature = request.temperature.unwrap_or(0.0);
        let top_p = request.top_p.unwrap_or_else(|| {
            if temperature > f32::EPSILON {
                self.preset
                    .map(|preset| preset.sampling_top_p)
                    .unwrap_or(1.0)
            } else {
                1.0
            }
        });
        let top_k = request.top_k.unwrap_or_else(|| {
            if temperature > f32::EPSILON {
                self.preset.map(|preset| preset.sampling_top_k).unwrap_or(0)
            } else {
                0
            }
        });
        Ok(GenerationOptions {
            stop_token_ids: self.assets.stop_token_ids(),
            stop_sequences: self.stop_sequences(stop_texts)?,
            temperature,
            top_p,
            top_k,
            seed: 0,
        })
    }

    fn stop_sequences(&self, stop_texts: &[String]) -> ServeResult<Vec<Vec<usize>>> {
        let mut sequences = Vec::new();
        for stop in stop_texts {
            let ids = self.assets.encode_prompt(stop)?;
            let sequence = tokens_to_usize(&ids)?;
            if !sequence.is_empty() {
                sequences.push(sequence);
            }
        }
        Ok(sequences)
    }

    fn encode_full_rendered_prompt(&self, text: &str) -> ServeResult<Vec<usize>> {
        let ids = if self.assets.config.is_gemma() && !self.assets.config.is_gemma4() {
            self.assets.encode_prompt_with_special(text)?
        } else {
            self.assets.encode_prompt(text)?
        };
        tokens_to_usize(&ids)
    }

    fn encode_rendered_suffix(&self, text: &str) -> ServeResult<Vec<usize>> {
        tokens_to_usize(&self.assets.encode_prompt(text)?)
    }

    fn remember_prompt_prefix(&mut self, rendered: String, tokens: Vec<usize>) {
        let capacity = prompt_prefix_capacity();
        if capacity == 0 {
            return;
        }
        if let Some(index) = self
            .prompt_prefixes
            .iter()
            .position(|entry| entry.rendered == rendered)
        {
            self.prompt_prefixes.remove(index);
        }
        self.prompt_prefixes
            .insert(0, PromptPrefixEntry { rendered, tokens });
        self.prompt_prefixes.truncate(capacity);
    }
}

fn longest_prompt_prefix_entry(
    entries: &[PromptPrefixEntry],
    rendered: &str,
) -> Option<(usize, Vec<usize>)> {
    entries
        .iter()
        .filter(|entry| rendered.starts_with(&entry.rendered))
        .max_by_key(|entry| entry.rendered.len())
        .map(|entry| (entry.rendered.len(), entry.tokens.clone()))
}

fn prompt_encoding_with_verified_prefix<F>(
    model_id: &str,
    entries: &[PromptPrefixEntry],
    rendered: String,
    full_tokens: Vec<usize>,
    mut encode_suffix: F,
) -> ServeResult<PromptEncoding>
where
    F: FnMut(&str) -> ServeResult<Vec<usize>>,
{
    if let Some((prefix_len, prefix_tokens)) = longest_prompt_prefix_entry(entries, &rendered) {
        let suffix = &rendered[prefix_len..];
        let suffix_tokens = if suffix.is_empty() {
            Vec::new()
        } else {
            encode_suffix(suffix)?
        };
        if joined_tokens_equal(&prefix_tokens, &suffix_tokens, &full_tokens) {
            return Ok(PromptEncoding {
                rendered,
                tokens: full_tokens,
                reused_prefix_tokens: prefix_tokens.len(),
            });
        }
        eprintln!(
            "saragossa serve prompt-prefix fallback model={} reason=tokenization-boundary-mismatch prefix_tokens={} suffix_tokens={} full_tokens={}",
            model_id,
            prefix_tokens.len(),
            suffix_tokens.len(),
            full_tokens.len()
        );
    }
    Ok(PromptEncoding {
        rendered,
        tokens: full_tokens,
        reused_prefix_tokens: 0,
    })
}

fn joined_tokens_equal(prefix: &[usize], suffix: &[usize], full: &[usize]) -> bool {
    if full.len() != prefix.len() + suffix.len() {
        return false;
    }
    full.starts_with(prefix) && full[prefix.len()..] == *suffix
}

fn prompt_prefix_capacity() -> usize {
    std::env::var(PREFIX_CACHE_CAP_ENV)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(4)
}

fn load_decoder_with_runtime(
    assets: &ModelAssets,
    backend: RuntimeKind,
) -> ServeResult<CausalDecoder> {
    match backend {
        RuntimeKind::Cpu => Ok(saragossa::load_causal_decoder(assets)?),
        RuntimeKind::Metal => load_decoder_metal(assets),
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn load_decoder_metal(assets: &ModelAssets) -> ServeResult<CausalDecoder> {
    let executor = saragossa::MetalExecutor::new()?;
    Ok(saragossa::load_causal_decoder(assets)?.with_metal_executor(executor))
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
fn load_decoder_metal(_assets: &ModelAssets) -> ServeResult<CausalDecoder> {
    Err(ServeError::args(
        "backend metal indisponible dans ce build — recompile avec --features metal",
    ))
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

fn strip_text_stops(text: String, stop_texts: &[String]) -> String {
    let Some(index) = stop_texts
        .iter()
        .filter(|stop| !stop.is_empty())
        .filter_map(|stop| text.find(stop))
        .min()
    else {
        return text;
    };
    text[..index].to_string()
}

fn tokens_to_usize(ids: &[u32]) -> ServeResult<Vec<usize>> {
    ids.iter()
        .copied()
        .map(|id| {
            usize::try_from(id).map_err(|_| {
                ServeError::args(format!("token id hors plage pour cette plateforme: {id}"))
            })
        })
        .collect()
}

fn tokens_to_u32(ids: &[usize]) -> ServeResult<Vec<u32>> {
    ids.iter()
        .copied()
        .map(|id| {
            u32::try_from(id)
                .map_err(|_| ServeError::args(format!("token généré hors plage: {id}")))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen_assistant_history_gets_empty_think_prefix() {
        let mut messages = vec![
            ChatTemplateMessage::new("user", "Bonjour"),
            ChatTemplateMessage::new("assistant", "Salut"),
        ];

        normalize_qwen_assistant_history(&mut messages);

        assert_eq!(
            messages[1].content.as_deref(),
            Some("<think>\n\n</think>\n\nSalut")
        );
    }

    #[test]
    fn stop_texts_strip_first_match() {
        let text = strip_text_stops(
            "abc STOP def END".to_string(),
            &["END".into(), "STOP".into()],
        );

        assert_eq!(text, "abc ");
    }

    #[test]
    fn longest_prompt_prefix_selects_longest_seen_prompt() {
        let entries = vec![
            PromptPrefixEntry {
                rendered: "abc".to_string(),
                tokens: vec![1],
            },
            PromptPrefixEntry {
                rendered: "abcdef".to_string(),
                tokens: vec![1, 2],
            },
        ];

        let hit =
            longest_prompt_prefix_entry(&entries, "abcdefghi").expect("invariant: préfixe trouvé");

        assert_eq!(hit.0, 6);
        assert_eq!(hit.1, vec![1, 2]);
    }

    #[test]
    fn prompt_prefix_mismatch_falls_back_to_full_tokenization() {
        let turn_one_rendered = "<|im_start|>user\nA<|im_end|>\n<|im_start|>assistant\n";
        let turn_two_rendered = format!("{turn_one_rendered}\nOK<|im_end|>\n");
        let entries = vec![PromptPrefixEntry {
            rendered: turn_one_rendered.to_string(),
            tokens: vec![74455, 198, 248068, 271, 248069],
        }];
        let full_tokens = vec![74455, 198, 248068, 271, 248069, 1358, 3793];

        let encoding = prompt_encoding_with_verified_prefix(
            "test-model",
            &entries,
            turn_two_rendered,
            full_tokens.clone(),
            |suffix| {
                assert_eq!(suffix, "\nOK<|im_end|>\n");
                Ok(vec![271, 198, 3793])
            },
        )
        .expect("invariant: encodage suffixe valide");

        assert_eq!(encoding.tokens, full_tokens);
        assert_eq!(encoding.reused_prefix_tokens, 0);
    }

    #[test]
    fn prompt_prefix_reuses_when_joined_tokens_equal_full_tokenization() {
        let turn_one_rendered = "<turn1>";
        let turn_two_rendered = "<turn1><turn2>".to_string();
        let entries = vec![PromptPrefixEntry {
            rendered: turn_one_rendered.to_string(),
            tokens: vec![1, 2, 3],
        }];
        let full_tokens = vec![1, 2, 3, 4, 5];

        let encoding = prompt_encoding_with_verified_prefix(
            "test-model",
            &entries,
            turn_two_rendered,
            full_tokens.clone(),
            |suffix| {
                assert_eq!(suffix, "<turn2>");
                Ok(vec![4, 5])
            },
        )
        .expect("invariant: encodage suffixe valide");

        assert_eq!(encoding.tokens, full_tokens);
        assert_eq!(encoding.reused_prefix_tokens, 3);
    }
}
