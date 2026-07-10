//! Etat d'inférence longue durée du serveur local.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use saragossa::decoder::GenerationTimings;
use saragossa::runtime_flags::{
    serve_lru_enabled, serve_model_pool_size, serve_prefix_cache_enabled,
};
use saragossa::{
    qwen_assistant_history_content, render_gemma4_chat, render_gemma_chat, render_qwen_chatml,
    CausalDecoder, CausalDecoderPromptState, ChatTemplateMessage, GenerationOptions, ModelAssets,
    RuntimePreset,
};

use super::args::{ServeArgs, ServeModelConfig};
use super::cache::{
    estimate_model_bytes, BlockAwarePrefixCache, BlockHash, MemoryProjection, ServeMemoryGuard,
};
use super::error::{ServeError, ServeResult};
use super::protocol::{ChatCompletionRequest, ModelInfo, ModelsResponse, Usage};
use crate::RuntimeKind;

/// Réponse d'inférence prête à sérialiser en OpenAI.
#[derive(Debug)]
pub(super) struct ServedCompletion {
    /// Identifiant OpenAI du modèle.
    pub(super) model: String,
    /// Texte assistant généré.
    pub(super) content: String,
    /// Raison de fin OpenAI.
    pub(super) finish_reason: &'static str,
    /// Stop sequence textuelle qui a arrêté la génération.
    pub(super) matched_stop: Option<String>,
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
    clock: u64,
    memory_guard: ServeMemoryGuard,
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
            clock: 0,
            memory_guard: ServeMemoryGuard::new(),
        }
    }

    /// Renvoie le plafond serveur du budget de génération.
    pub(super) fn max_tokens_cap(&self) -> usize {
        self.max_tokens_cap
    }

    /// Charge tous les modèles configurés.
    pub(super) fn preload(&mut self) -> ServeResult<()> {
        for index in 0..self.models.len() {
            self.ensure_loaded_index(index)?;
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
        let index = self
            .models
            .iter()
            .position(|model| model.id == request.model)
            .ok_or_else(|| ServeError::UnknownModel(request.model.clone()))?;
        self.ensure_loaded_index(index)?;
        self.clock = self.clock.saturating_add(1);
        self.models[index].last_used = self.clock;
        let model_id = self.models[index].id.clone();
        let memory_guard = self.memory_guard.clone();
        let completion = self.models[index]
            .loaded
            .as_mut()
            .ok_or_else(|| ServeError::args(format!("modèle {model_id} absent après chargement")))?
            .complete(request, max_tokens, &memory_guard)?;
        self.enforce_memory_budget(0, Some(index))?;
        Ok(completion)
    }

    fn ensure_loaded_index(&mut self, index: usize) -> ServeResult<()> {
        if self.models[index].loaded.is_some() {
            return Ok(());
        }
        let additional = self.models[index].estimated_model_bytes();
        self.enforce_memory_budget(additional, Some(index))?;
        self.enforce_model_pool_limit(index);
        self.models[index].ensure_loaded()
    }

    fn enforce_model_pool_limit(&mut self, protected: usize) {
        if !serve_lru_enabled() {
            return;
        }
        let limit = serve_model_pool_size();
        while self.loaded_count() >= limit {
            if self.evict_lru_model(Some(protected)).is_none() {
                break;
            }
        }
    }

    fn enforce_memory_budget(
        &mut self,
        additional: u64,
        protected: Option<usize>,
    ) -> ServeResult<()> {
        while let Some(projection) = self.memory_guard.projection_over_limit(additional) {
            if self.evict_one_prefix_block().is_some() {
                continue;
            }
            if self.evict_lru_model(protected).is_some() {
                continue;
            }
            return Err(ServeError::memory(memory_error_message(projection)));
        }
        Ok(())
    }

    fn loaded_count(&self) -> usize {
        self.models
            .iter()
            .filter(|model| model.loaded.is_some())
            .count()
    }

    fn evict_one_prefix_block(&mut self) -> Option<usize> {
        let index = self
            .models
            .iter()
            .enumerate()
            .filter(|(_, model)| model.loaded.is_some())
            .filter(|(_, model)| {
                model
                    .loaded
                    .as_ref()
                    .is_some_and(|loaded| loaded.prefix_cache_bytes() > 0)
            })
            .min_by_key(|(_, model)| model.last_used)
            .map(|(index, _)| index)?;
        let loaded = self.models[index].loaded.as_mut()?;
        let bytes = loaded.evict_prefix_block()?;
        eprintln!(
            "saragossa serve evicted prefix block model={} bytes={}",
            self.models[index].id, bytes
        );
        Some(bytes)
    }

    fn evict_lru_model(&mut self, protected: Option<usize>) -> Option<String> {
        if !serve_lru_enabled() {
            return None;
        }
        let index = self
            .models
            .iter()
            .enumerate()
            .filter(|(index, model)| Some(*index) != protected && model.loaded.is_some())
            .min_by_key(|(_, model)| model.last_used)
            .map(|(index, _)| index)?;
        let id = self.models[index].id.clone();
        self.models[index].loaded = None;
        eprintln!("saragossa serve evicted model={id} reason=lru-memory");
        Some(id)
    }
}

fn memory_error_message(projection: MemoryProjection) -> String {
    format!(
        "mémoire insuffisante: footprint={} projeté={} plafond={}",
        projection.current, projection.projected, projection.limit
    )
}

struct ModelSlot {
    id: String,
    path: PathBuf,
    backend: RuntimeKind,
    loaded: Option<LoadedModel>,
    last_used: u64,
    estimated_model_bytes: u64,
}

impl ModelSlot {
    fn new(model: &ServeModelConfig, backend: RuntimeKind) -> Self {
        Self {
            id: model.id.clone(),
            path: model.path.clone(),
            backend,
            loaded: None,
            last_used: 0,
            estimated_model_bytes: 0,
        }
    }

    fn estimated_model_bytes(&mut self) -> u64 {
        if self.estimated_model_bytes == 0 {
            self.estimated_model_bytes = estimate_model_bytes(&self.path);
        }
        self.estimated_model_bytes
    }

    fn ensure_loaded(&mut self) -> ServeResult<()> {
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
                prefix_cache: BlockAwarePrefixCache::from_runtime_flags(),
            });
        }
        self.loaded.as_ref().ok_or_else(|| {
            ServeError::args(format!(
                "modèle {} non chargé après initialisation",
                self.id
            ))
        })?;
        Ok(())
    }
}

struct LoadedModel {
    id: String,
    assets: ModelAssets,
    decoder: CausalDecoder,
    preset: Option<RuntimePreset>,
    prefix_cache: BlockAwarePrefixCache,
}

struct PromptEncoding {
    tokens: Vec<usize>,
}

impl LoadedModel {
    fn complete(
        &mut self,
        request: ChatCompletionRequest,
        max_tokens: usize,
        memory_guard: &ServeMemoryGuard,
    ) -> ServeResult<ServedCompletion> {
        let stop_texts = request.stop_texts();
        let prompt = self.prompt_encoding(&request)?;
        let prompt_tokens = prompt.tokens.len();
        let options = self.generation_options(&request, &stop_texts)?;
        let started = Instant::now();
        let (prompt_state, reused_prefix_tokens, prefill) =
            self.prefill_prompt_state(&prompt.tokens, memory_guard)?;
        let output = self
            .decoder
            .generate_greedy_timed_from_prompt_state_with_options(
                prompt_state,
                prefill,
                max_tokens,
                &options,
            )?;
        let total = started.elapsed();
        let generated = tokens_to_u32(&output.tokens)?;
        let decoded = strip_empty_think(self.assets.decode_tokens(&generated, true)?);
        let matched_stop = matched_text_stop(&decoded, &stop_texts);
        let finish_reason = if output.tokens.len() >= max_tokens && matched_stop.is_none() {
            "length"
        } else {
            "stop"
        };
        let content = strip_text_stops(decoded, &stop_texts);
        eprintln!(
            "saragossa serve completion model={} prompt_tokens={} completion_tokens={} prefill_ms={} decode_ms={} total_ms={} reused_prefix_tokens={} prefix_cache=block-snapshots",
            self.id,
            prompt_tokens,
            output.tokens.len(),
            output.timings.prefill.as_millis(),
            output.timings.decode.as_millis(),
            total.as_millis(),
            reused_prefix_tokens
        );
        Ok(ServedCompletion {
            model: self.id.clone(),
            content,
            finish_reason,
            matched_stop,
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
        Ok(PromptEncoding {
            tokens: full_tokens,
        })
    }

    fn prefill_prompt_state(
        &mut self,
        tokens: &[usize],
        memory_guard: &ServeMemoryGuard,
    ) -> ServeResult<(CausalDecoderPromptState, usize, Duration)> {
        let started = Instant::now();
        if !serve_prefix_cache_enabled() {
            let state = self.decoder.prefill_prompt_state_uncached(tokens)?;
            return Ok((state, 0, started.elapsed()));
        }

        let block_tokens = self.prefix_cache.block_tokens();
        let full_block_tokens = tokens.len() / block_tokens * block_tokens;
        let hit = self.prefix_cache.match_prefix(tokens);
        let reused_prefix_tokens = hit.as_ref().map_or(0, |hit| hit.tokens);

        let (mut state, mut consumed, mut hash) = if let Some(hit) = hit {
            let mut state = hit.state;
            let metal = self.decoder.copy_prompt_state_metal_snapshot(&hit.metal)?;
            self.decoder.restore_prompt_state_metal(&mut state, metal)?;
            (state, hit.tokens, hit.hash)
        } else if full_block_tokens >= block_tokens {
            let first = &tokens[..block_tokens];
            let state = self.decoder.prefill_prompt_state_uncached(first)?;
            let hash = BlockHash::root().chain(first);
            self.remember_prefix_block(hash, block_tokens, &state, memory_guard)?;
            (state, block_tokens, hash)
        } else {
            let state = self.decoder.prefill_prompt_state_uncached(tokens)?;
            return Ok((state, 0, started.elapsed()));
        };

        while consumed + block_tokens <= full_block_tokens {
            let next = consumed + block_tokens;
            let block = &tokens[consumed..next];
            self.decoder.extend_prompt_state(&mut state, block)?;
            hash = hash.chain(block);
            self.remember_prefix_block(hash, next, &state, memory_guard)?;
            consumed = next;
        }

        if consumed < tokens.len() {
            self.decoder
                .extend_prompt_state(&mut state, &tokens[consumed..])?;
        }

        Ok((state, reused_prefix_tokens, started.elapsed()))
    }

    fn remember_prefix_block(
        &mut self,
        hash: BlockHash,
        tokens: usize,
        state: &CausalDecoderPromptState,
        memory_guard: &ServeMemoryGuard,
    ) -> ServeResult<()> {
        let metal = self.decoder.snapshot_prompt_state_metal(state)?;
        let additional = usize_to_u64_saturating(
            state
                .estimated_cpu_bytes()
                .saturating_add(metal.estimated_bytes()),
        );
        while memory_guard.projection_over_limit(additional).is_some() {
            if self.prefix_cache.evict_lru_block().is_none() {
                eprintln!(
                    "saragossa serve prefix-cache skip model={} reason=oom-guard tokens={tokens}",
                    self.id
                );
                return Ok(());
            }
        }
        self.prefix_cache.insert(hash, tokens, state.clone(), metal);
        Ok(())
    }

    fn prefix_cache_bytes(&self) -> usize {
        self.prefix_cache.estimated_bytes()
    }

    fn evict_prefix_block(&mut self) -> Option<usize> {
        self.prefix_cache.evict_lru_block()
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

fn matched_text_stop(text: &str, stop_texts: &[String]) -> Option<String> {
    stop_texts
        .iter()
        .filter(|stop| !stop.is_empty())
        .filter_map(|stop| text.find(stop).map(|index| (index, stop)))
        .min_by_key(|(index, _)| *index)
        .map(|(_, stop)| stop.clone())
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

fn usize_to_u64_saturating(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
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
    fn usize_to_u64_saturating_keeps_small_values() {
        assert_eq!(usize_to_u64_saturating(42), 42);
    }

    #[test]
    fn matched_text_stop_returns_first_match() {
        let matched = matched_text_stop("abc STOP def END", &["END".into(), "STOP".into()]);

        assert_eq!(matched.as_deref(), Some("STOP"));
    }
}
