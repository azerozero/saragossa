//! Deltas texte sûrs pour les réponses SSE.

use std::time::Duration;

use saragossa::ModelAssets;

use super::error::{ServeError, ServeResult};

/// Métadonnées connues avant le premier token streamé.
#[derive(Debug)]
pub(crate) struct StreamingCompletionStart {
    /// Identifiant OpenAI du modèle.
    pub(super) model: String,
    /// Nombre de tokens prompt.
    pub(super) prompt_tokens: usize,
    /// Nombre de tokens repris d'un prompt déjà vu.
    pub(super) reused_prefix_tokens: usize,
    /// Durée du préfill déjà exécuté.
    pub(super) prefill: Duration,
}

impl StreamingCompletionStart {
    /// Renvoie les headers de diagnostic disponibles au démarrage du flux.
    pub(super) fn metric_headers(&self) -> Vec<(&'static str, String)> {
        vec![
            (
                "x-saragossa-prefill-ms",
                self.prefill.as_millis().to_string(),
            ),
            ("x-saragossa-decode-ms", "0".to_string()),
            ("x-saragossa-decode-tokens", "0".to_string()),
            ("x-saragossa-prompt-tokens", self.prompt_tokens.to_string()),
            (
                "x-saragossa-reused-prefix-tokens",
                self.reused_prefix_tokens.to_string(),
            ),
            ("x-saragossa-total-ms", self.prefill.as_millis().to_string()),
        ]
    }
}

/// Evénement produit par une complétion streamée.
pub(crate) enum CompletionStreamEvent<'a> {
    /// Signale que le préfill est terminé et que les headers peuvent partir.
    Start(&'a StreamingCompletionStart),
    /// Porte un delta texte prêt à écrire au client.
    Delta(&'a str),
    /// Signale une erreur terminale après démarrage du flux.
    TerminalError(&'a StreamTerminalError<'a>),
}

/// Erreur terminale sérialisable par chaque dialecte SSE.
pub(crate) struct StreamTerminalError<'a> {
    /// Message lisible par le client.
    pub(crate) message: &'a str,
    /// Type stable côté client.
    pub(crate) error_type: &'static str,
}

impl<'a> StreamTerminalError<'a> {
    /// Construit l'erreur terminale d'objet JSON tronqué.
    pub(crate) fn incomplete_json(message: &'a str) -> Self {
        Self {
            message,
            error_type: "incomplete_json",
        }
    }
}

pub(crate) struct StreamingTextDetokenizer<'a> {
    assets: &'a ModelAssets,
    stop_texts: &'a [String],
    generated: Vec<u32>,
    emitter: StreamingTextEmitter,
}

impl<'a> StreamingTextDetokenizer<'a> {
    pub(crate) fn new(
        assets: &'a ModelAssets,
        stop_texts: &'a [String],
        max_tokens: usize,
    ) -> Self {
        Self {
            assets,
            stop_texts,
            generated: Vec::with_capacity(max_tokens),
            emitter: StreamingTextEmitter::default(),
        }
    }

    pub(crate) fn push_token(
        &mut self,
        token: usize,
        on_event: &mut impl FnMut(CompletionStreamEvent<'_>) -> ServeResult<()>,
    ) -> ServeResult<()> {
        let token = u32::try_from(token)
            .map_err(|_| ServeError::args(format!("token généré hors plage: {token}")))?;
        self.generated.push(token);
        let decoded = self.assets.decode_tokens(&self.generated, true)?;
        let visible = streaming_visible_text(&decoded, self.stop_texts);
        self.emitter.emit_streaming(visible, on_event)
    }

    pub(crate) fn finish(
        &mut self,
        content: &str,
        on_event: &mut impl FnMut(CompletionStreamEvent<'_>) -> ServeResult<()>,
    ) -> ServeResult<()> {
        self.emitter.emit_final(content, on_event)
    }
}

#[derive(Default)]
struct StreamingTextEmitter {
    emitted: String,
}

impl StreamingTextEmitter {
    fn emit_streaming(
        &mut self,
        visible: &str,
        on_event: &mut impl FnMut(CompletionStreamEvent<'_>) -> ServeResult<()>,
    ) -> ServeResult<()> {
        self.emit_until(visible, false, on_event)
    }

    fn emit_final(
        &mut self,
        visible: &str,
        on_event: &mut impl FnMut(CompletionStreamEvent<'_>) -> ServeResult<()>,
    ) -> ServeResult<()> {
        self.emit_until(visible, true, on_event)
    }

    fn emit_until(
        &mut self,
        visible: &str,
        final_snapshot: bool,
        on_event: &mut impl FnMut(CompletionStreamEvent<'_>) -> ServeResult<()>,
    ) -> ServeResult<()> {
        let Some(delta) = visible.strip_prefix(&self.emitted) else {
            return Err(ServeError::args(
                "désynchronisation du detokenizer streaming",
            ));
        };
        if delta.is_empty() {
            return Ok(());
        }
        if !final_snapshot && visible.ends_with('\u{FFFD}') {
            return Ok(());
        }
        on_event(CompletionStreamEvent::Delta(delta))?;
        self.emitted = visible.to_string();
        Ok(())
    }
}

fn streaming_visible_text<'a>(decoded: &'a str, stop_texts: &[String]) -> &'a str {
    let text = strip_empty_think_streaming(decoded);
    let text = strip_after_first_stop(text, stop_texts);
    let hold = longest_stop_prefix_suffix_len(text, stop_texts);
    &text[..text.len().saturating_sub(hold)]
}

fn strip_empty_think_streaming(text: &str) -> &str {
    let prefix = saragossa::QWEN_EMPTY_THINK_BLOCK;
    if let Some(stripped) = text.strip_prefix(prefix) {
        return stripped;
    }
    if prefix.starts_with(text) {
        return "";
    }
    text
}

fn strip_after_first_stop<'a>(text: &'a str, stop_texts: &[String]) -> &'a str {
    let Some(index) = stop_texts
        .iter()
        .filter(|stop| !stop.is_empty())
        .filter_map(|stop| text.find(stop))
        .min()
    else {
        return text;
    };
    &text[..index]
}

fn longest_stop_prefix_suffix_len(text: &str, stop_texts: &[String]) -> usize {
    let mut hold = 0;
    for stop in stop_texts.iter().filter(|stop| !stop.is_empty()) {
        for (index, _) in stop.char_indices().skip(1) {
            if text.ends_with(&stop[..index]) {
                hold = hold.max(index);
            }
        }
        if text.ends_with(stop) {
            hold = hold.max(stop.len());
        }
    }
    hold
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streamed_deltas_match_non_stream_oracle_without_model() {
        let stop_texts = vec![" STOP".to_string()];
        let final_decoded = format!("{}Bonjour STOP ignoré", saragossa::QWEN_EMPTY_THINK_BLOCK);
        let final_content = strip_text_stops(strip_empty_think(final_decoded.clone()), &stop_texts);
        let snapshots = [
            "<think>".to_string(),
            format!("{}Bon", saragossa::QWEN_EMPTY_THINK_BLOCK),
            format!("{}Bonjour S", saragossa::QWEN_EMPTY_THINK_BLOCK),
            format!("{}Bonjour ST", saragossa::QWEN_EMPTY_THINK_BLOCK),
            final_decoded,
        ];

        let streamed = collect_visible_deltas(&snapshots, &final_content, &stop_texts);

        assert_eq!(streamed, final_content);
    }

    #[test]
    fn emitter_holds_incomplete_utf8_replacement_until_snapshot_is_stable() {
        let mut emitter = StreamingTextEmitter::default();
        let mut deltas = Vec::new();

        emit_streaming_snapshot(&mut emitter, "…", &mut deltas);
        deltas.clear();
        emit_streaming_snapshot(&mut emitter, "…\u{FFFD}", &mut deltas);
        emit_streaming_snapshot(&mut emitter, "…é", &mut deltas);

        assert_eq!(deltas, vec!["é".to_string()]);
    }

    #[test]
    fn emitter_releases_legitimate_replacement_character_at_finish() {
        let mut emitter = StreamingTextEmitter::default();
        let mut deltas = Vec::new();

        emit_streaming_snapshot(&mut emitter, "abc", &mut deltas);
        emit_streaming_snapshot(&mut emitter, "abc\u{FFFD}", &mut deltas);
        emit_final_snapshot(&mut emitter, "abc\u{FFFD}", &mut deltas);

        assert_eq!(deltas.concat(), "abc\u{FFFD}");
        assert_eq!(deltas, vec!["abc".to_string(), "\u{FFFD}".to_string()]);
    }

    fn collect_visible_deltas(
        snapshots: &[String],
        final_content: &str,
        stop_texts: &[String],
    ) -> String {
        let mut emitter = StreamingTextEmitter::default();
        let mut streamed = String::new();
        for snapshot in snapshots {
            let visible = streaming_visible_text(snapshot, stop_texts);
            emitter
                .emit_streaming(visible, &mut |event| {
                    if let CompletionStreamEvent::Delta(delta) = event {
                        streamed.push_str(delta);
                    }
                    Ok(())
                })
                .expect("invariant: snapshot streaming valide");
        }
        emitter
            .emit_final(final_content, &mut |event| {
                if let CompletionStreamEvent::Delta(delta) = event {
                    streamed.push_str(delta);
                }
                Ok(())
            })
            .expect("invariant: final non-stream prolonge le flux");
        streamed
    }

    fn emit_streaming_snapshot(
        emitter: &mut StreamingTextEmitter,
        snapshot: &str,
        deltas: &mut Vec<String>,
    ) {
        emitter
            .emit_streaming(snapshot, &mut |event| {
                if let CompletionStreamEvent::Delta(delta) = event {
                    deltas.push(delta.to_string());
                }
                Ok(())
            })
            .expect("invariant: snapshot streaming valide");
    }

    fn emit_final_snapshot(
        emitter: &mut StreamingTextEmitter,
        snapshot: &str,
        deltas: &mut Vec<String>,
    ) {
        emitter
            .emit_final(snapshot, &mut |event| {
                if let CompletionStreamEvent::Delta(delta) = event {
                    deltas.push(delta.to_string());
                }
                Ok(())
            })
            .expect("invariant: snapshot final valide");
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
}
