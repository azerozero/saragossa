//! Etat d'inférence longue durée du serveur local.

use std::path::PathBuf;
use std::sync::{Arc, OnceLock};
use std::thread::{self, ThreadId};
use std::time::{Duration, Instant};

use saragossa::decoder::{GenerationOutput, GenerationTimings};
use saragossa::memory_guard::{
    estimate_model_bytes, process_footprint_bytes, process_purge_registry, MemoryGuard,
    MemoryProjection, PurgeOutcome, PurgePressure, PurgeRegistry,
};
use saragossa::runtime_flags::{
    serve_lru_enabled, serve_model_pool_size, serve_prefix_cache_enabled,
};
use saragossa::{
    qwen_assistant_history_content, render_flint_chat, render_gemma4_chat, render_gemma_chat,
    render_qwen_chatml, CausalDecoder, CausalDecoderPromptState, ChatTemplateMessage,
    GenerationOptions, JsonTokenCatalog, JsonTokenConstraint, ModelAssets, RuntimePreset,
    TokenConstraint,
};

use super::args::{ServeArgs, ServeModelConfig};
use super::audio::AudioState;
use super::cache::{BlockAwarePrefixCache, BlockHash};
use super::embeddings::EmbeddingsState;
use super::error::{ServeError, ServeResult};
use super::mtp;
use super::protocol::{ChatCompletionRequest, ResponseFormatMode, Usage};
use super::streaming::{
    CompletionSink, CompletionStreamEvent, StreamTerminalError, StreamingCompletionStart,
    StreamingTextDetokenizer,
};
use super::warmup;
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
///
/// L'état appartient au thread moteur, celui qui l'a construit : les arènes et
/// pools scratch GPU sont `thread_local`, un décodeur appelé depuis un autre
/// thread perdrait ses buffers résidents.
pub(super) struct ServeState {
    engine_thread: ThreadId,
    models: Vec<ModelSlot>,
    max_tokens_cap: usize,
    clock: u64,
    memory_guard: MemoryGuard,
    audio: AudioState,
    embeddings: EmbeddingsState,
}

impl ServeState {
    /// Construit l'état à partir des chemins CLI, sans charger les poids.
    pub(super) fn new(args: &ServeArgs) -> Self {
        Self {
            engine_thread: thread::current().id(),
            models: args
                .models
                .iter()
                .map(|model| ModelSlot::new(model, args.backend))
                .collect(),
            max_tokens_cap: args.max_tokens_cap,
            clock: 0,
            memory_guard: MemoryGuard::serve(),
            audio: AudioState::new(args),
            embeddings: EmbeddingsState::new(args),
        }
    }

    /// Renvoie le plafond serveur du budget de génération.
    pub(super) fn max_tokens_cap(&self) -> usize {
        self.max_tokens_cap
    }

    /// Renvoie l'état audio mutable pour les endpoints STT/TTS.
    pub(super) fn audio_mut(&mut self) -> &mut AudioState {
        self.assert_engine_thread();
        &mut self.audio
    }

    /// Renvoie l'état embeddings mutable pour l'endpoint `/v1/embeddings`.
    pub(super) fn embeddings_mut(&mut self) -> &mut EmbeddingsState {
        self.assert_engine_thread();
        &mut self.embeddings
    }

    /// Charge tous les modèles configurés.
    pub(super) fn preload(&mut self) -> ServeResult<()> {
        for index in 0..self.models.len() {
            self.ensure_loaded_index(index)?;
        }
        Ok(())
    }

    /// Identifiants exposés par `/v1/models`, figés au démarrage : les threads
    /// de connexion servent cette route sans toucher à l'état moteur.
    pub(super) fn model_ids(&self) -> Vec<String> {
        self.models.iter().map(|model| model.id.clone()).collect()
    }

    /// Garde de l'invariant thread moteur (voir [`ServeState`]).
    fn assert_engine_thread(&self) {
        debug_assert_eq!(
            thread::current().id(),
            self.engine_thread,
            "invariant: ServeState n'est utilisé que sur le thread moteur"
        );
    }

    /// Exécute une complétion chat OpenAI.
    pub(super) fn complete(
        &mut self,
        request: ChatCompletionRequest,
        session_key: Option<&str>,
    ) -> ServeResult<ServedCompletion> {
        let max_tokens_cap = self.max_tokens_cap;
        let max_tokens = request.validate(max_tokens_cap)?;
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
            .complete(request, max_tokens, session_key, &memory_guard)?;
        self.enforce_memory_budget(0, Some(index))?;
        Ok(completion)
    }

    /// Exécute une complétion chat en envoyant les deltas via `on_event`.
    pub(super) fn complete_streaming(
        &mut self,
        request: ChatCompletionRequest,
        session_key: Option<&str>,
        mut on_event: impl FnMut(CompletionStreamEvent<'_>) -> ServeResult<()>,
    ) -> ServeResult<ServedCompletion> {
        let max_tokens_cap = self.max_tokens_cap;
        let max_tokens = request.validate(max_tokens_cap)?;
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
            .complete_streaming(
                request,
                max_tokens,
                session_key,
                &memory_guard,
                &mut on_event,
            )?;
        self.enforce_memory_budget(0, Some(index))?;
        Ok(completion)
    }

    /// Le modèle `model` peut-il servir un lot multi-flux dense ? Le charge au
    /// besoin, comme la complétion qui suit ; une erreur de chargement rend
    /// `false` et sera rapportée par le chemin solo.
    pub(super) fn batch_ready(&mut self, model: &str) -> bool {
        let Some(index) = self.models.iter().position(|slot| slot.id == model) else {
            return false;
        };
        self.ensure_loaded_index(index).is_ok()
            && self.models[index]
                .loaded
                .as_ref()
                .is_some_and(|loaded| loaded.decoder.lightbatch_multi_dense_ready())
    }

    /// Exécute un lot de complétions chat streamées greedy d'un même modèle.
    /// `sinks[i]` reçoit les événements de `requests[i]` puis est terminé ;
    /// l'issue `Err` d'un client n'a pas encore été écrite, l'appelant envoie
    /// l'erreur HTTP.
    ///
    /// Les requêtes sont découpées en sous-lots consécutifs qui tiennent dans
    /// la garde mémoire ; un sous-lot d'une requête prend le chemin solo
    /// inchangé (MTP). Rend aussi le nombre de pas multi-flux dense encodés.
    pub(super) fn complete_streaming_batch(
        &mut self,
        requests: Vec<ChatCompletionRequest>,
        sinks: &mut [Box<dyn CompletionSink + '_>],
    ) -> (Vec<ServeResult<()>>, usize) {
        let mut results: Vec<Option<ServeResult<()>>> = requests.iter().map(|_| None).collect();
        let mut multi_steps = 0;
        let Some(model) = requests.first().map(|request| request.model.clone()) else {
            return (Vec::new(), 0);
        };
        let max_tokens_cap = self.max_tokens_cap;
        let loaded = self
            .models
            .iter()
            .position(|slot| slot.id == model)
            .ok_or_else(|| ServeError::UnknownModel(model.clone()))
            .and_then(|index| self.ensure_loaded_index(index).map(|()| index));
        let index = match loaded {
            Ok(index) => index,
            Err(error) => {
                let outcomes = sinks
                    .iter_mut()
                    .map(|sink| sink.finish(Err(error.replicate())))
                    .collect();
                return (outcomes, 0);
            }
        };
        self.clock = self.clock.saturating_add(1);
        self.models[index].last_used = self.clock;
        let mut prepared = Vec::with_capacity(requests.len());
        for (slot, request) in requests.iter().enumerate() {
            let outcome = request.validate(max_tokens_cap).and_then(|max| {
                self.models[index]
                    .loaded
                    .as_ref()
                    .ok_or_else(|| ServeError::args(format!("modèle {model} absent")))?
                    .prepare(request, max)
            });
            match outcome {
                Ok(request) => prepared.push((slot, request)),
                Err(error) => results[slot] = Some(sinks[slot].finish(Err(error))),
            }
        }
        let mut requests: Vec<Option<_>> = requests.into_iter().map(Some).collect();
        let memory_guard = self.memory_guard.clone();
        let mut rest = prepared.as_slice();
        while !rest.is_empty() {
            let take = self.models[index].loaded.as_ref().map_or(1, |loaded| {
                fitting_streams(&memory_guard, rest, |request| {
                    loaded.stream_bytes_upper(request.tokens.len(), request.max_tokens)
                })
            });
            let (chunk, tail) = rest.split_at(take);
            rest = tail;
            if let [(slot, _)] = chunk {
                // NOTE: seul dans son sous-lot (mémoire) : chemin solo inchangé,
                // sans session. Une requête-session n'arrive ici que si le solo
                // la servirait en MTP, qui ignore la session (cf. `batch_chat`).
                let Some(request) = requests[*slot].take() else {
                    continue;
                };
                let sink = &mut sinks[*slot];
                let result = self.complete_streaming(request, None, |event| sink.event(event));
                results[*slot] = Some(sink.finish(result));
                continue;
            }
            let Some(loaded) = self.models[index].loaded.as_ref() else {
                continue;
            };
            let slots: Vec<usize> = chunk.iter().map(|(slot, _)| *slot).collect();
            let streams: Vec<&PreparedRequest> = chunk.iter().map(|(_, request)| request).collect();
            let mut chunk_sinks: Vec<&mut Box<dyn CompletionSink + '_>> = sinks
                .iter_mut()
                .enumerate()
                .filter(|(slot, _)| slots.contains(slot))
                .map(|(_, sink)| sink)
                .collect();
            let (outcomes, steps) = loaded.complete_lightbatch(&streams, &mut chunk_sinks);
            multi_steps += steps;
            for (slot, outcome) in slots.into_iter().zip(outcomes) {
                results[slot] = Some(outcome);
            }
        }
        if let Err(error) = self.enforce_memory_budget(0, Some(index)) {
            eprintln!("saragossa serve batch garde mémoire après lot: {error}");
        }
        let results = results
            .into_iter()
            .map(|result| {
                result.unwrap_or_else(|| Err(ServeError::Batch("requête non traitée".to_string())))
            })
            .collect();
        (results, multi_steps)
    }

    /// Le modèle `model` est-il chargé avec son decode MTP actif ? Ne charge
    /// rien : un modèle absent répond `false`.
    pub(super) fn model_routes_mtp(&self, model: &str) -> bool {
        self.models
            .iter()
            .find(|slot| slot.id == model)
            .and_then(|slot| slot.loaded.as_ref())
            .is_some_and(|loaded| loaded.mtp_active)
    }

    /// Le modèle `model` écrit-il au format flint (`<think>`, `<tool_call>`
    /// XML) ? Le charge au besoin, comme la complétion qui suit.
    pub(super) fn model_is_flint(&mut self, model: &str) -> ServeResult<bool> {
        let index = self
            .models
            .iter()
            .position(|slot| slot.id == model)
            .ok_or_else(|| ServeError::UnknownModel(model.to_string()))?;
        self.ensure_loaded_index(index)?;
        Ok(self.models[index]
            .loaded
            .as_ref()
            .is_some_and(|loaded| loaded.assets.config.is_flint()))
    }

    fn ensure_loaded_index(&mut self, index: usize) -> ServeResult<()> {
        // NOTE: tout appel décodeur LLM passe par ici, chargé ou non.
        self.assert_engine_thread();
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
            if self.purge_one_for_memory_budget(protected).is_some() {
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

    fn purge_one_for_memory_budget(&mut self, protected: Option<usize>) -> Option<String> {
        let mut registry = PurgeRegistry::<ServeState>::new();
        registry.register(10, "serve-prefix-block", |state, _| {
            state
                .evict_one_prefix_block()
                .map(|bytes| PurgeOutcome::Purged {
                    bytes: usize_to_u64_saturating(bytes),
                })
                .unwrap_or(PurgeOutcome::Empty)
        });
        // Cache de préfixe du décodeur (chemin MTP) et autres purgeables du
        // process : sans eux, une requête MTP échouait alors qu'ils étaient
        // libérables.
        registry.register(15, "process-purgeables", |_, pressure| {
            process_purge_registry()
                .lock()
                .ok()
                .and_then(|mut registry| registry.purge_one(&mut (), pressure))
                .map(|report| PurgeOutcome::Purged {
                    bytes: report.bytes,
                })
                .unwrap_or(PurgeOutcome::Empty)
        });
        registry.register(20, "serve-model", move |state, _| {
            state
                .evict_lru_model(protected)
                .map(|_| PurgeOutcome::Purged { bytes: 0 })
                .unwrap_or(PurgeOutcome::Empty)
        });
        registry
            .purge_one(self, PurgePressure::Warn)
            .map(|report| report.name)
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
            let mut decoder = load_decoder_with_runtime(&assets, self.backend)?;
            let mtp_active = mtp::active_for(&assets, self.backend);
            if mtp_active {
                let path = assets.mtp.path.as_ref().ok_or_else(|| {
                    ServeError::args(format!(
                        "modèle {} qualifié MTP sans chemin de sidecar",
                        self.id
                    ))
                })?;
                decoder = decoder.with_mtp_sidecar(path)?;
                eprintln!(
                    "saragossa serve MTP actif model={} sidecar={} max_draft={}",
                    self.id,
                    path.display(),
                    mtp::max_draft_tokens()
                );
            }
            // Le drafter n'est joignable que par le chemin MTP : inutile sans lui.
            #[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
            if mtp_active {
                decoder = decoder.with_dflash_draft_from_env(&assets.config)?;
                if let Some(draft) = decoder.dflash_draft() {
                    eprintln!(
                        "saragossa serve drafter DFlash actif model={} block_size={} quant_bits={:?}",
                        self.id,
                        draft.info.block_size,
                        draft.quant.map(|quant| quant.bits)
                    );
                }
            }
            warmup::decoder(&mut decoder, &assets, self.backend, &self.id)?;
            if mtp_active {
                warmup::mtp_decoder(&decoder, &assets, self.backend, &self.id)?;
            }
            self.loaded = Some(LoadedModel {
                id: self.id.clone(),
                assets,
                decoder,
                preset,
                mtp_active,
                json_token_catalog: OnceLock::new(),
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
    mtp_active: bool,
    /// Catalogue des bytes par token, bâti paresseusement à la 1ʳᵉ requête
    /// `json_object`/`json_lines` : un déploiement texte pur ne paie jamais le
    /// décodage du vocab entier ni son étape faillible au chargement.
    json_token_catalog: OnceLock<Arc<JsonTokenCatalog>>,
    prefix_cache: BlockAwarePrefixCache,
}

struct PromptEncoding {
    tokens: Vec<usize>,
}

/// Requête chat validée et encodée, prête pour le solo ou un lot.
struct PreparedRequest {
    tokens: Vec<usize>,
    options: GenerationOptions,
    stop_texts: Vec<String>,
    response_format: ResponseFormatMode,
    max_tokens: usize,
}

/// Etat de streaming d'un flux du lot, possédé par ce flux.
struct BatchFlux<'a> {
    detokenizer: StreamingTextDetokenizer<'a>,
    tokens: Vec<usize>,
    first_token: Option<Instant>,
    started: bool,
    /// Issue du client, posée dès que son flux s'arrête.
    outcome: Option<ServeResult<()>>,
}

/// Nombre de requêtes en tête de `rest` dont la somme des octets estimés tient
/// sous la garde mémoire ; au moins une (le solo garde sa propre garde).
fn fitting_streams<T>(
    memory_guard: &MemoryGuard,
    rest: &[(usize, T)],
    bytes: impl Fn(&T) -> u64,
) -> usize {
    let mut total = 0_u64;
    let mut take = 0;
    for (_, request) in rest {
        total = total.saturating_add(bytes(request));
        if take > 0 && memory_guard.projection_over_limit(total).is_some() {
            break;
        }
        take += 1;
    }
    take.max(1)
}

/// Etiquettes du journal de complétion.
struct CompletionPath {
    prefix_cache: &'static str,
    decode_path: &'static str,
    streaming: bool,
}

impl LoadedModel {
    /// Le solo sert-il `request` en MTP ? Même règle en non-stream et stream.
    fn routes_mtp(&self, request: &ChatCompletionRequest, prepared: &PreparedRequest) -> bool {
        self.mtp_active
            && mtp::routes_request(request, prepared.response_format)
            && mtp::supports_options(&prepared.options)
    }

    fn complete(
        &mut self,
        request: ChatCompletionRequest,
        max_tokens: usize,
        session_key: Option<&str>,
        memory_guard: &MemoryGuard,
    ) -> ServeResult<ServedCompletion> {
        let prepared = self.prepare(&request, max_tokens)?;
        let PreparedRequest {
            tokens: prompt_tokens_ids,
            options,
            ..
        } = &prepared;
        let token_constraint = options.token_constraint.clone();
        let started = Instant::now();
        let use_mtp = self.routes_mtp(&request, &prepared);
        let (output, reused_prefix_tokens, prefix_cache) = if use_mtp {
            // NOTE: v1 MTP assume son propre prefill via le prefix-cache du
            // décodeur (longest-prefix + extension). Il by-passe donc les
            // snapshots par blocs de serve pour ces seules requêtes greedy
            // non guidées ; le chemin AR reste strictement inchangé.
            let output = self.decoder.generate_greedy_mtp_batched_with_options(
                prompt_tokens_ids,
                max_tokens,
                options,
                mtp::max_draft_tokens(),
            )?;
            let reused = output.stats.reused_prefix_tokens;
            (
                mtp::into_generation_output(output, started.elapsed()),
                reused,
                "decoder-longest-prefix",
            )
        } else {
            let (prompt_state, reused_prefix_tokens, prefill) = self
                .prefill_prompt_state_reserving(
                    prompt_tokens_ids,
                    max_tokens,
                    session_key,
                    memory_guard,
                )?;
            // TODO: Câbler une annulation non-stream quand serve aura un signal partagé fiable.
            let output = self
                .decoder
                .generate_greedy_timed_from_prompt_state_with_options(
                    prompt_state,
                    prefill,
                    max_tokens,
                    options,
                )?;
            (output, reused_prefix_tokens, "block-snapshots")
        };
        ensure_guided_finished(token_constraint.as_deref())?;
        self.finish_completion(
            &prepared,
            output,
            reused_prefix_tokens,
            started.elapsed(),
            CompletionPath {
                prefix_cache,
                decode_path: if use_mtp { "mtp" } else { "ar" },
                streaming: false,
            },
        )
    }

    fn complete_streaming(
        &mut self,
        request: ChatCompletionRequest,
        max_tokens: usize,
        session_key: Option<&str>,
        memory_guard: &MemoryGuard,
        on_event: &mut impl FnMut(CompletionStreamEvent<'_>) -> ServeResult<()>,
    ) -> ServeResult<ServedCompletion> {
        let prepared = self.prepare(&request, max_tokens)?;
        let PreparedRequest {
            tokens: prompt_tokens_ids,
            options,
            stop_texts,
            ..
        } = &prepared;
        let prompt_tokens = prompt_tokens_ids.len();
        let token_constraint = options.token_constraint.clone();
        let started = Instant::now();
        let use_mtp = self.routes_mtp(&request, &prepared);
        let (prompt_state, mut reused_prefix_tokens, prefill) = if use_mtp {
            (None, 0, Duration::ZERO)
        } else {
            let (state, reused, elapsed) = self.prefill_prompt_state_reserving(
                prompt_tokens_ids,
                max_tokens,
                session_key,
                memory_guard,
            )?;
            (Some(state), reused, elapsed)
        };
        if !use_mtp {
            let start = StreamingCompletionStart {
                model: self.id.clone(),
                prompt_tokens,
                reused_prefix_tokens,
                prefill,
            };
            on_event(CompletionStreamEvent::Start(&start))?;
        }

        let mut detokenizer = StreamingTextDetokenizer::new(&self.assets, stop_texts, max_tokens);
        let mut stream_error = None;
        let mut stream_started = !use_mtp;
        let output = if use_mtp {
            // NOTE: comme le non-stream, MTP préremplit via le cache longest-prefix
            // du décodeur et by-passe les snapshots serve v1. Le callback observe
            // uniquement les tokens committés, puis le detokenizer existant garde
            // l'exacte même politique de stops et de texte visible.
            let output = self.decoder.generate_greedy_mtp_streaming_with_options(
                prompt_tokens_ids,
                max_tokens,
                options,
                mtp::max_draft_tokens(),
                |token| {
                    if stream_error.is_some() {
                        return false;
                    }
                    if !stream_started {
                        let start = StreamingCompletionStart {
                            model: self.id.clone(),
                            prompt_tokens,
                            // NOTE: le hit MTP n'est connu qu'en fin de génération ;
                            // l'en-tête reste à 0, le journal de fin le porte.
                            reused_prefix_tokens: 0,
                            // Le callback MTP commence après son prefill interne.
                            // Cette mesure inclut au plus le premier pas verify.
                            prefill: started.elapsed(),
                        };
                        if let Err(error) = on_event(CompletionStreamEvent::Start(&start)) {
                            stream_error = Some(error);
                            return false;
                        }
                        stream_started = true;
                    }
                    // Stop texte vu sans séquence d'ids terminée (découpe
                    // différente) : le decode s'arrête là aussi.
                    match detokenizer.push_token(token, on_event) {
                        Ok(()) => !detokenizer.stopped(),
                        Err(error) => {
                            stream_error = Some(error);
                            false
                        }
                    }
                },
            )?;
            reused_prefix_tokens = output.stats.reused_prefix_tokens;
            mtp::into_generation_output(output, started.elapsed())
        } else {
            self.decoder
                .generate_greedy_timed_from_prompt_state_with_options_and_callback(
                    prompt_state
                        .ok_or_else(|| ServeError::args("état de prompt AR absent avant decode"))?,
                    prefill,
                    max_tokens,
                    options,
                    |token| {
                        if stream_error.is_some() {
                            return false;
                        }
                        match detokenizer.push_token(token, on_event) {
                            Ok(()) => !detokenizer.stopped(),
                            Err(error) => {
                                stream_error = Some(error);
                                false
                            }
                        }
                    },
                )?
        };
        if !stream_started {
            let start = StreamingCompletionStart {
                model: self.id.clone(),
                prompt_tokens,
                reused_prefix_tokens,
                prefill: output.timings.prefill,
            };
            on_event(CompletionStreamEvent::Start(&start))?;
        }
        if let Some(error) = stream_error {
            return Err(error);
        }
        if let Err(error) = ensure_guided_finished(token_constraint.as_deref()) {
            if let Some(message) = error.incomplete_json_message() {
                let terminal = StreamTerminalError::incomplete_json(message);
                on_event(CompletionStreamEvent::TerminalError(&terminal))?;
            }
            return Err(error);
        }
        let completion = self.finish_completion(
            &prepared,
            output,
            reused_prefix_tokens,
            started.elapsed(),
            CompletionPath {
                prefix_cache: if use_mtp {
                    "decoder-longest-prefix"
                } else {
                    "block-snapshots"
                },
                decode_path: if use_mtp { "mtp" } else { "ar" },
                streaming: true,
            },
        )?;
        detokenizer.finish(&completion.content, on_event)?;
        Ok(completion)
    }

    /// Valide et encode une requête chat : stops, prompt, options.
    fn prepare(
        &self,
        request: &ChatCompletionRequest,
        max_tokens: usize,
    ) -> ServeResult<PreparedRequest> {
        let response_format = request.response_format_mode()?;
        let stop_texts = stop_texts_for_response_format(request, response_format)?;
        let tokens = self.prompt_encoding(request)?.tokens;
        let options = self.generation_options(request, &stop_texts, response_format)?;
        Ok(PreparedRequest {
            tokens,
            options,
            stop_texts,
            response_format,
            max_tokens,
        })
    }

    /// Texte final, raison de fin et journal d'une génération terminée.
    fn finish_completion(
        &self,
        prepared: &PreparedRequest,
        output: GenerationOutput,
        reused_prefix_tokens: usize,
        total: Duration,
        path: CompletionPath,
    ) -> ServeResult<ServedCompletion> {
        let prompt_tokens = prepared.tokens.len();
        let generated = tokens_to_u32(&output.tokens)?;
        let decoded = strip_empty_think(self.assets.decode_tokens(&generated, true)?);
        let matched_stop = matched_text_stop(&decoded, &prepared.stop_texts);
        let finish_reason = if output.tokens.len() >= prepared.max_tokens && matched_stop.is_none()
        {
            "length"
        } else {
            "stop"
        };
        let content = strip_text_stops(decoded, &prepared.stop_texts);
        eprintln!(
            "saragossa serve completion model={} prompt_tokens={} completion_tokens={} prefill_ms={} decode_ms={} total_ms={} reused_prefix_tokens={} prefix_cache={} decode_path={}{} footprint_bytes={} prefix_cache_bytes={}",
            self.id,
            prompt_tokens,
            output.tokens.len(),
            output.timings.prefill.as_millis(),
            output.timings.decode.as_millis(),
            total.as_millis(),
            reused_prefix_tokens,
            path.prefix_cache,
            path.decode_path,
            if path.streaming { " streaming=true" } else { "" },
            process_footprint_bytes().unwrap_or(0),
            self.prefix_cache_bytes()
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

    /// Décode `streams` ensemble par le light-batch (pas multi-flux dense si
    /// éligible) et termine chaque client ; rend aussi le nombre de pas multi.
    ///
    /// NOTE: le prefill passe par le cache longest-prefix du décodeur, comme
    /// MTP, pas par les snapshots par blocs de serve (les requêtes avec
    /// session restent en solo, sauf celles que le solo servirait en MTP).
    fn complete_lightbatch(
        &self,
        streams: &[&PreparedRequest],
        sinks: &mut [&mut Box<dyn CompletionSink + '_>],
    ) -> (Vec<ServeResult<()>>, usize) {
        let prompts: Vec<Vec<usize>> = streams.iter().map(|stream| stream.tokens.clone()).collect();
        let budgets: Vec<usize> = streams.iter().map(|stream| stream.max_tokens).collect();
        let stop_ids = streams[0].options.stop_token_ids.clone();
        // Tokens d'arrêt et stops texte sont vus par flux dans `drive_batch` :
        // le light-batch rend chaque token, qui finalise son flux sur-le-champ.
        let options = GenerationOptions {
            stop_token_ids: Vec::new(),
            stop_sequences: Vec::new(),
            ..streams[0].options.clone()
        };
        self.drive_batch(streams, sinks, &stop_ids, |on_token| {
            self.decoder
                .generate_greedy_lightbatch_streaming_counted(
                    &prompts, &budgets, &options, on_token,
                )
                .map(|(_, multi_steps)| multi_steps)
        })
    }

    /// Distribue les tokens de `generate` aux flux : SSE, stops, puis issue de
    /// chaque client dès que SON flux s'arrête (token d'arrêt, stop texte,
    /// budget, client parti). Une erreur de `generate` ne touche que les flux
    /// encore actifs.
    fn drive_batch(
        &self,
        streams: &[&PreparedRequest],
        sinks: &mut [&mut Box<dyn CompletionSink + '_>],
        stop_ids: &[usize],
        generate: impl FnOnce(&mut dyn FnMut(usize, usize) -> bool) -> saragossa::Result<usize>,
    ) -> (Vec<ServeResult<()>>, usize) {
        let started = Instant::now();
        let mut flux: Vec<BatchFlux<'_>> = streams
            .iter()
            .map(|stream| BatchFlux {
                detokenizer: StreamingTextDetokenizer::new(
                    &self.assets,
                    &stream.stop_texts,
                    stream.max_tokens,
                ),
                tokens: Vec::with_capacity(stream.max_tokens),
                first_token: None,
                started: false,
                outcome: None,
            })
            .collect();
        let mut on_token = |index: usize, token: usize| {
            let (Some(flux), Some(sink), Some(stream)) = (
                flux.get_mut(index),
                sinks.get_mut(index),
                streams.get(index),
            ) else {
                return false;
            };
            if flux.outcome.is_some() {
                return false;
            }
            flux.first_token.get_or_insert_with(Instant::now);
            if let Err(error) = self.start_flux(flux, sink, stream, started.elapsed()) {
                flux.outcome = Some(sink.finish(Err(error)));
                return false;
            }
            if stop_ids.contains(&token) {
                flux.outcome = Some(self.finish_flux(flux, sink, stream, started));
                return false;
            }
            flux.tokens.push(token);
            let pushed = flux
                .detokenizer
                .push_token(token, &mut |event| sink.event(event));
            if let Err(error) = pushed {
                flux.outcome = Some(sink.finish(Err(error)));
                return false;
            }
            if flux.detokenizer.stopped() || flux.tokens.len() >= stream.max_tokens {
                flux.outcome = Some(self.finish_flux(flux, sink, stream, started));
                return false;
            }
            true
        };
        let generated = generate(&mut on_token);
        let multi_steps = match &generated {
            Ok(steps) => *steps,
            Err(_) => 0,
        };
        eprintln!(
            "saragossa serve batch model={} streams={} multi_steps={} total_ms={}{}",
            self.id,
            streams.len(),
            multi_steps,
            started.elapsed().as_millis(),
            generated
                .as_ref()
                .err()
                .map_or(String::new(), |error| format!(" erreur: {error}"))
        );
        let outcomes = flux
            .iter_mut()
            .zip(sinks.iter_mut())
            .zip(streams)
            .map(|((flux, sink), stream)| {
                if let Some(outcome) = flux.outcome.take() {
                    return outcome;
                }
                match &generated {
                    // Le light-batch s'est arrêté sans token final pour ce
                    // flux : on le termine avec ce qu'il a produit.
                    Ok(_) => self.finish_flux(flux, sink, stream, started),
                    Err(error) => {
                        let message = error.to_string();
                        if flux.started {
                            let terminal = StreamTerminalError {
                                message: &message,
                                error_type: "server_error",
                            };
                            let _ = sink.event(CompletionStreamEvent::TerminalError(&terminal));
                        }
                        sink.finish(Err(ServeError::Batch(message)))
                    }
                }
            })
            .collect();
        (outcomes, multi_steps)
    }

    /// Envoie le début de réponse d'un flux s'il n'est pas encore parti.
    fn start_flux(
        &self,
        flux: &mut BatchFlux<'_>,
        sink: &mut Box<dyn CompletionSink + '_>,
        stream: &PreparedRequest,
        prefill: Duration,
    ) -> ServeResult<()> {
        if flux.started {
            return Ok(());
        }
        let start = StreamingCompletionStart {
            model: self.id.clone(),
            prompt_tokens: stream.tokens.len(),
            reused_prefix_tokens: 0,
            // Les prefills de tout le lot précèdent le premier token.
            prefill,
        };
        sink.event(CompletionStreamEvent::Start(&start))?;
        flux.started = true;
        Ok(())
    }

    /// Termine un client : texte final, `finish_reason`, usage, journal et
    /// fin de réponse SSE.
    fn finish_flux(
        &self,
        flux: &mut BatchFlux<'_>,
        sink: &mut Box<dyn CompletionSink + '_>,
        stream: &PreparedRequest,
        batch_started: Instant,
    ) -> ServeResult<()> {
        let now = Instant::now();
        let first = flux.first_token.unwrap_or(now);
        let prefill = first.saturating_duration_since(batch_started);
        if let Err(error) = self.start_flux(flux, sink, stream, prefill) {
            return sink.finish(Err(error));
        }
        let tokens = std::mem::take(&mut flux.tokens);
        let output = GenerationOutput {
            timings: GenerationTimings {
                prefill,
                decode: now.saturating_duration_since(first),
                decode_tokens: tokens.len().saturating_sub(1),
            },
            tokens,
        };
        let completion = self.finish_completion(
            stream,
            output,
            0,
            batch_started.elapsed(),
            CompletionPath {
                prefix_cache: "decoder-longest-prefix",
                decode_path: "batch",
                streaming: true,
            },
        );
        let completion = match completion {
            Ok(completion) => completion,
            Err(error) => return sink.finish(Err(error)),
        };
        if let Err(error) = flux
            .detokenizer
            .finish(&completion.content, &mut |event| sink.event(event))
        {
            return sink.finish(Err(error));
        }
        sink.finish(Ok(completion))
    }

    /// Majorant des octets propres à un flux du lot : K/V des couches
    /// full-attention en f32 (le decode passe en bf16 au-delà d'un seuil de
    /// contexte, la borne surestime alors), état récurrent des couches
    /// linear-attention et une marge fixe pour le scratch par flux.
    ///
    /// ponytail: estimation analytique, pas une mesure de l'arène ; la
    /// remplacer par la taille réelle de l'arène si la garde se montre fausse.
    fn stream_bytes_upper(&self, prompt_tokens: usize, max_tokens: usize) -> u64 {
        const F32: u64 = 4;
        const SCRATCH_MARGIN: u64 = 64 << 20;
        let config = &self.assets.config;
        let layers = config.num_hidden_layers as u64;
        let full_layers = (0..config.num_hidden_layers)
            .filter(|layer| config.is_full_attention_layer(*layer))
            .count() as u64;
        let rows = prompt_tokens.saturating_add(max_tokens) as u64;
        let kv = full_layers
            .saturating_mul(rows)
            .saturating_mul((config.num_key_value_heads * config.head_dim()) as u64)
            .saturating_mul(2 * F32);
        let linear_state = match (
            config.linear_num_value_heads,
            config.linear_key_head_dim,
            config.linear_value_head_dim,
        ) {
            (Some(heads), Some(key), Some(value)) => layers
                .saturating_sub(full_layers)
                .saturating_mul((heads * key * value) as u64)
                .saturating_mul(F32),
            _ => 0,
        };
        kv.saturating_add(linear_state)
            .saturating_add(SCRATCH_MARGIN)
    }

    fn prompt_encoding(&self, request: &ChatCompletionRequest) -> ServeResult<PromptEncoding> {
        let mut messages = request.template_messages();
        // flint garde l'historique assistant tel quel : pas de bloc think vide.
        if !self.assets.config.is_gemma() && !self.assets.config.is_flint() {
            normalize_qwen_assistant_history(&mut messages);
        }
        let rendered = if self.assets.config.is_flint() {
            let tools: Vec<String> = request
                .active_tools()?
                .iter()
                .map(|tool| tool.json.clone())
                .collect();
            render_flint_chat(&messages, &tools, true)
        } else if self.assets.config.is_gemma4() {
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

    /// [`Self::prefill_prompt_state`] avec une réserve K/V pour la queue du
    /// prompt (moins d'un bloc) et `max_tokens` : l'arène de decode adopte
    /// alors le K/V du prefill au lieu de le recopier.
    fn prefill_prompt_state_reserving(
        &mut self,
        tokens: &[usize],
        max_tokens: usize,
        session_key: Option<&str>,
        memory_guard: &MemoryGuard,
    ) -> ServeResult<(CausalDecoderPromptState, usize, Duration)> {
        let rows = self.prefix_cache.block_tokens().saturating_add(max_tokens);
        saragossa::decoder::with_prefill_kv_reserve(rows, || {
            self.prefill_prompt_state(tokens, session_key, memory_guard)
        })
    }

    fn prefill_prompt_state(
        &mut self,
        tokens: &[usize],
        session_key: Option<&str>,
        memory_guard: &MemoryGuard,
    ) -> ServeResult<(CausalDecoderPromptState, usize, Duration)> {
        let started = Instant::now();
        // Plafond flint vérifié avant toute recherche ou construction de
        // blocs : sinon les blocs complets seraient préremplis pour rien.
        self.decoder
            .ensure_prompt_within_prefill_cap(tokens.len())?;
        // NOTE: l'extension en lignes coûte ~2,3× un prefill à froid par token
        // (29k étendus en 127 s contre 32k à froid en 63 s, 27B) : sans prefill
        // résident, un hit qui couvre moins de la moitié des blocs est ignoré.
        let cheap_extend = self.decoder.prefix_extend_as_cheap_as_cold(tokens.len());
        // NOTE: serve utilise les APIs uncached/extend ; le cache interne du
        // décodeur reste hors chemin pour éviter un partage inter-session.
        if !prefix_blocks_usable(self.assets.config.is_flint(), cheap_extend) {
            let state = self.decoder.prefill_prompt_state_uncached(tokens)?;
            return Ok((state, 0, started.elapsed()));
        }

        let block_tokens = self.prefix_cache.block_tokens();
        let full_block_tokens = tokens.len() / block_tokens * block_tokens;
        let root = BlockHash::for_session(session_key);
        let hit = self
            .prefix_cache
            .match_prefix(root, tokens)
            .filter(|hit| cheap_extend || hit.tokens.saturating_mul(2) >= full_block_tokens);
        let reused_prefix_tokens = hit.as_ref().map_or(0, |hit| hit.tokens);

        // NOTE: une seule frontière mémorisée par prompt, la dernière complète.
        // Un snapshot par bloc partageait l'Arc du KV avec l'état vivant : chaque
        // extension recopiait tout le préfixe, soit O(n²) retenu (~65 Gio en 16k).
        // ponytail: un préfixe commun plus court que le prompt servi (prompt
        // système partagé) n'est réutilisé que s'il a été servi seul ; ajouter
        // une frontière déclarée (fin du message système) si ce cas compte.
        let mut state = if let Some(hit) = hit {
            let hit_kv = kv_allocations(&hit.state);
            let mut state = hit.state;
            let metal = self.decoder.copy_prompt_state_metal_snapshot(&hit.metal)?;
            self.decoder.restore_prompt_state_metal(&mut state, metal)?;
            self.decoder
                .extend_prompt_state(&mut state, &tokens[hit.tokens..full_block_tokens])?;
            // Prolongement qui a recopié tout le K/V dans un buffer neuf : les
            // blocs de la session qui tiennent l'ancien buffer retiendraient une
            // seconde copie complète de la conversation à chaque tour. Retirés
            // avant la garde mémoire, pour ne pas évincer d'autres sessions.
            let live_kv = kv_allocations(&state);
            if !hit_kv.is_empty() && hit_kv.is_disjoint(&live_kv) {
                self.prefix_cache.remove_superseded(root, &hit_kv, &live_kv);
            }
            state
        } else if full_block_tokens > 0 {
            self.decoder
                .prefill_prompt_state_uncached(&tokens[..full_block_tokens])?
        } else {
            let state = self.decoder.prefill_prompt_state_uncached(tokens)?;
            return Ok((state, 0, started.elapsed()));
        };

        if reused_prefix_tokens < full_block_tokens {
            let hash = tokens[..full_block_tokens]
                .chunks_exact(block_tokens)
                .fold(root, BlockHash::chain);
            self.remember_prefix_block(root, hash, full_block_tokens, &state, memory_guard)?;
        }
        self.decoder
            .extend_prompt_state(&mut state, &tokens[full_block_tokens..])?;

        Ok((state, reused_prefix_tokens, started.elapsed()))
    }

    fn remember_prefix_block(
        &mut self,
        session: BlockHash,
        hash: BlockHash,
        tokens: usize,
        state: &CausalDecoderPromptState,
        memory_guard: &MemoryGuard,
    ) -> ServeResult<()> {
        let metal = self.decoder.snapshot_prompt_state_metal(state)?;
        // Le snapshot partage les segments KV de l'état vivant, déjà en mémoire :
        // un second passage sur le même `seen` ne compte que le reste.
        let mut seen = std::collections::HashSet::new();
        state.estimated_cpu_bytes_unseen(&mut seen);
        let additional = usize_to_u64_saturating(
            state
                .estimated_cpu_bytes_unseen(&mut seen)
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
        self.prefix_cache
            .insert(session, hash, tokens, state.clone(), metal);
        Ok(())
    }

    fn prefix_cache_bytes(&self) -> usize {
        self.prefix_cache.estimated_bytes()
    }

    fn evict_prefix_block(&mut self) -> Option<usize> {
        self.prefix_cache.evict_lru_block()
    }

    fn json_catalog(&self) -> ServeResult<Arc<JsonTokenCatalog>> {
        if let Some(catalog) = self.json_token_catalog.get() {
            return Ok(Arc::clone(catalog));
        }
        let catalog = Arc::new(JsonTokenCatalog::from_tokenizer(&self.assets.tokenizer)?);
        // En cas de course, `set` échoue et on réutilise la valeur déjà
        // installée (le catalogue est immutable, les deux builds sont égaux).
        let _ = self.json_token_catalog.set(Arc::clone(&catalog));
        Ok(self.json_token_catalog.get().map_or(catalog, Arc::clone))
    }

    fn generation_options(
        &self,
        request: &ChatCompletionRequest,
        stop_texts: &[String],
        response_format: ResponseFormatMode,
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
        let stop_token_ids = self.assets.stop_token_ids();
        let token_constraint = match response_format {
            ResponseFormatMode::Text => None,
            ResponseFormatMode::JsonObject => Some(Arc::new(JsonTokenConstraint::new(
                self.json_catalog()?,
                &stop_token_ids,
                false,
            )) as Arc<dyn TokenConstraint>),
            ResponseFormatMode::JsonLines => Some(Arc::new(JsonTokenConstraint::new(
                self.json_catalog()?,
                &stop_token_ids,
                true,
            )) as Arc<dyn TokenConstraint>),
        };
        Ok(GenerationOptions {
            stop_token_ids,
            stop_sequences: if matches!(
                response_format,
                ResponseFormatMode::JsonObject | ResponseFormatMode::JsonLines
            ) {
                Vec::new()
            } else {
                self.stop_sequences(stop_texts)?
            },
            temperature,
            top_p,
            top_k,
            seed: 0,
            token_constraint,
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
    Ok(saragossa::load_causal_decoder_for_metal(assets, &executor)?
        .with_metal_executor(executor)?)
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
fn load_decoder_metal(_assets: &ModelAssets) -> ServeResult<CausalDecoder> {
    Err(ServeError::args(
        "backend metal indisponible dans ce build — recompile avec --features metal",
    ))
}

fn stop_texts_for_response_format(
    request: &ChatCompletionRequest,
    response_format: ResponseFormatMode,
) -> ServeResult<Vec<String>> {
    let stop_texts = request.stop_texts();
    if matches!(
        response_format,
        ResponseFormatMode::JsonObject | ResponseFormatMode::JsonLines
    ) && stop_texts.iter().any(|stop| !stop.is_empty())
    {
        let format = match response_format {
            ResponseFormatMode::JsonObject => "json_object",
            ResponseFormatMode::JsonLines => "json_lines",
            ResponseFormatMode::Text => "text",
        };
        return Err(ServeError::Http(format!(
            "stop n'est pas supporté avec response_format {format}"
        )));
    }
    Ok(stop_texts)
}

fn ensure_guided_finished(constraint: Option<&dyn TokenConstraint>) -> ServeResult<()> {
    let Some(constraint) = constraint else {
        return Ok(());
    };
    if constraint.is_finished() {
        return Ok(());
    }
    Err(ServeError::incomplete_json(
        "structured output JSON incomplet avant la fin du budget max_tokens",
    ))
}

/// Le cache de préfixe par blocs de serve est-il utilisable pour ce modèle ?
///
/// NOTE: flint n'étend un reliquat de bloc ou le suffixe d'un hit que par le
/// prefill résident prolongé (`resident_extend`), dès un token ; sans lui,
/// l'extension passerait par le forward CPU per-op (~0,6 s par token) et son
/// prompt entier reste au prefill à froid, sans blocs.
fn prefix_blocks_usable(flint: bool, resident_extend: bool) -> bool {
    serve_prefix_cache_enabled() && (!flint || resident_extend)
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
mod json_tests;

#[cfg(test)]
mod batch_drive_tests;

#[cfg(all(test, target_os = "macos", feature = "metal"))]
mod batch_tests;

/// Identités des allocations K/V (et autres segments comptés) d'un état de
/// prompt : deux états qui n'en partagent aucune ne se partagent pas le K/V.
fn kv_allocations(state: &CausalDecoderPromptState) -> std::collections::HashSet<usize> {
    let mut seen = std::collections::HashSet::new();
    state.estimated_cpu_bytes_unseen(&mut seen);
    seen
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Blocs du cache de préfixe : toujours hors flint ; pour flint, seulement
    /// si le prolongement résident existe, sans quoi le reliquat passerait par
    /// le forward CPU.
    #[test]
    fn flint_uses_prefix_blocks_only_with_resident_extend() {
        let enabled = serve_prefix_cache_enabled();
        assert_eq!(prefix_blocks_usable(false, false), enabled);
        assert_eq!(prefix_blocks_usable(true, true), enabled);
        assert!(!prefix_blocks_usable(true, false));
    }

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

    /// Le sous-lot s'arrête au premier flux qui ferait passer la garde ; un
    /// flux seul passe toujours (le solo applique sa propre garde).
    #[test]
    fn fitting_streams_cuts_batch_at_memory_limit() {
        let limits = saragossa::memory_guard::MemoryLimits {
            static_cap: Some(100),
            host_memory: None,
            headroom_bytes: 0,
            metal_cap: None,
        };
        let guard = MemoryGuard::from_limits(true, limits, Arc::new(|| Some(0)));
        let rest = [(0, 40_u64), (1, 40), (2, 40), (3, 40)];

        assert_eq!(fitting_streams(&guard, &rest, |bytes| *bytes), 2);
        assert_eq!(fitting_streams(&guard, &[(0, 500_u64)], |bytes| *bytes), 1);
        assert_eq!(
            fitting_streams(&MemoryGuard::disabled(), &rest, |bytes| *bytes),
            4
        );
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
