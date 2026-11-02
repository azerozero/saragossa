//! Distribution des tokens d'un lot aux clients, sans GPU : générateur simulé
//! sur le modèle minuscule des tests JSON.

use std::cell::RefCell;
use std::rc::Rc;

use saragossa::InferError;

use super::json_tests::tiny_loaded_model;
use super::*;

/// Client qui journalise ses événements dans un fil commun aux flux.
struct Recorder {
    flux: usize,
    log: Rc<RefCell<Vec<String>>>,
    started: bool,
}

impl Recorder {
    fn push(&self, what: &str) {
        self.log.borrow_mut().push(format!("{}:{what}", self.flux));
    }
}

impl CompletionSink for Recorder {
    fn event(&mut self, event: CompletionStreamEvent<'_>) -> ServeResult<()> {
        match event {
            CompletionStreamEvent::Start(_) => {
                self.started = true;
                self.push("start");
            }
            CompletionStreamEvent::Delta(_) => self.push("delta"),
            CompletionStreamEvent::TerminalError(_) => self.push("error"),
        }
        Ok(())
    }

    // Même contrat que les clients SSE : une erreur après démarrage est déjà
    // signalée dans le flux, avant démarrage elle remonte pour la réponse HTTP.
    fn finish(&mut self, result: ServeResult<ServedCompletion>) -> ServeResult<()> {
        match result {
            Ok(completion) => {
                self.push(&format!("done:{}", completion.finish_reason));
                Ok(())
            }
            Err(error) => {
                self.push("fail");
                if self.started {
                    Ok(())
                } else {
                    Err(error)
                }
            }
        }
    }
}

/// Flux court (token d'arrêt), flux long coupé par une erreur tardive du
/// décodeur, flux jamais démarré : le court est terminé dès son arrêt, avant
/// l'erreur ; seuls les flux encore actifs la reçoivent.
#[test]
fn short_stream_finishes_before_late_decoder_error() {
    let loaded = tiny_loaded_model();
    let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "tiny", "messages": [{"role": "user", "content": "{}"}],
        "temperature": 0, "stream": true, "max_tokens": 8
    }))
    .expect("invariant: requête test valide");
    let prepared = loaded
        .prepare(&request, 8)
        .expect("invariant: requête préparée");
    let stop = *prepared
        .options
        .stop_token_ids
        .first()
        .expect("invariant: token d'arrêt du modèle tiny");
    let log = Rc::new(RefCell::new(Vec::new()));
    let mut boxes: Vec<Box<dyn CompletionSink + '_>> = (0..3)
        .map(|flux| {
            Box::new(Recorder {
                flux,
                log: Rc::clone(&log),
                started: false,
            }) as Box<dyn CompletionSink + '_>
        })
        .collect();
    let mut sinks: Vec<&mut Box<dyn CompletionSink + '_>> = boxes.iter_mut().collect();
    let streams = [&prepared, &prepared, &prepared];
    let generator_log = Rc::clone(&log);

    let (outcomes, _) = loaded.drive_batch(&streams, &mut sinks, &[stop], |on_token| {
        assert!(on_token(0, 0));
        assert!(on_token(1, 0));
        assert!(!on_token(0, stop), "le token d'arrêt termine le flux court");
        generator_log.borrow_mut().push("gen:error".to_string());
        assert!(on_token(1, 0));
        Err(InferError::Config("panne simulée".to_string()))
    });

    let log = log.borrow();
    let at = |entry: &str| {
        log.iter()
            .position(|line| line == entry)
            .unwrap_or_else(|| panic!("entrée absente: {entry} dans {log:?}"))
    };
    assert!(at("0:done:stop") < at("gen:error"), "{log:?}");
    assert!(!log.contains(&"0:error".to_string()), "{log:?}");
    assert!(at("gen:error") < at("1:error") && at("1:error") < at("1:fail"));
    assert!(!log.contains(&"2:start".to_string()), "{log:?}");
    assert!(log.contains(&"2:fail".to_string()));
    assert!(outcomes[0].is_ok() && outcomes[1].is_ok());
    assert!(matches!(outcomes[2], Err(ServeError::Batch(_))));
}

/// Le budget atteint termine le flux sur son dernier token.
#[test]
fn budget_reached_finishes_stream_with_length() {
    let loaded = tiny_loaded_model();
    let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "tiny", "messages": [{"role": "user", "content": "{}"}],
        "temperature": 0, "stream": true, "max_tokens": 2
    }))
    .expect("invariant: requête test valide");
    let prepared = loaded
        .prepare(&request, 2)
        .expect("invariant: requête préparée");
    let log = Rc::new(RefCell::new(Vec::new()));
    let mut boxes: Vec<Box<dyn CompletionSink + '_>> = vec![Box::new(Recorder {
        flux: 0,
        log: Rc::clone(&log),
        started: false,
    })];
    let mut sinks: Vec<&mut Box<dyn CompletionSink + '_>> = boxes.iter_mut().collect();

    let (outcomes, _) = loaded.drive_batch(&[&prepared], &mut sinks, &[], |on_token| {
        assert!(on_token(0, 0));
        assert!(!on_token(0, 0), "budget atteint");
        Ok(0)
    });

    assert!(outcomes[0].is_ok());
    assert_eq!(
        log.borrow().last().map(String::as_str),
        Some("0:done:length")
    );
}

/// Un sous-lot réduit à une requête part en solo SANS session : une
/// requête-session (clé d'en-tête) n'y est admise que si ce solo est le MTP,
/// `stop` compris.
#[test]
fn batched_session_request_alone_would_be_served_by_mtp() {
    let mut loaded = tiny_loaded_model();
    loaded.mtp_active = true;
    for (stop, admitted) in [("", true), (r#","stop":"{}""#, true)] {
        let body = format!(
            r#"{{"model":"tiny","messages":[{{"role":"user","content":"x"}}],"stream":true{stop}}}"#
        );
        let raw = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nx-saragossa-session: s1\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        let deadline = Instant::now() + Duration::from_secs(60);
        let http =
            super::super::http::read_request(&mut std::io::Cursor::new(raw.into_bytes()), deadline)
                .expect("invariant: requête lisible")
                .expect("invariant: requête présente");
        let batched = super::super::http::batch_chat(&http, 1024, |_| loaded.mtp_active);
        assert_eq!(batched.is_some(), admitted, "stop {stop}");
        let chat: ChatCompletionRequest =
            serde_json::from_str(&body).expect("invariant: requête test valide");
        let prepared = loaded
            .prepare(&chat, 4)
            .expect("invariant: requête préparée");
        assert_eq!(loaded.routes_mtp(&chat, &prepared), admitted, "stop {stop}");
    }
}
