//! Lots multi-flux de serve sur le modèle dense hybride réel (Qwen3.8-27B) :
//! `SARAGOSSA_RUST_MTP_VERIFIER_TEST_MODEL` pointe vers ses poids, avec
//! `SARAGOSSA_RUST_LIGHTBATCH_MULTI=1`.

use std::io;
use std::path::PathBuf;

use super::super::args::ServeArgs;
use super::*;

const PROMPTS: [&str; 3] = [
    "Explique la photosynthèse en deux phrases.",
    "Donne trois fleuves européens et leur pays.",
    "Écris un haïku sur la mer.",
];
const MAX_TOKENS: usize = 48;

/// Client de test : collecte le texte streamé ; `fail_after` simule un
/// client parti après ce nombre de deltas.
struct Collect<'a> {
    text: &'a mut String,
    /// Contenu final, ou message d'erreur, posé par `finish`.
    done: &'a mut Option<Result<String, String>>,
    fail_after: Option<usize>,
    deltas: usize,
}

impl CompletionSink for Collect<'_> {
    fn event(&mut self, event: CompletionStreamEvent<'_>) -> ServeResult<()> {
        if let CompletionStreamEvent::Delta(delta) = event {
            if self.fail_after == Some(self.deltas) {
                return Err(ServeError::io(
                    "client parti",
                    io::Error::from(io::ErrorKind::BrokenPipe),
                ));
            }
            self.deltas += 1;
            self.text.push_str(delta);
        }
        Ok(())
    }

    fn finish(&mut self, result: ServeResult<ServedCompletion>) -> ServeResult<()> {
        *self.done = Some(
            result
                .map(|completion| completion.content)
                .map_err(|error| error.to_string()),
        );
        Ok(())
    }
}

fn real_state() -> Option<ServeState> {
    let Some(path) = saragossa::runtime_flags::env_var_os("SARAGOSSA_RUST_MTP_VERIFIER_TEST_MODEL")
        .map(PathBuf::from)
        .filter(|path| path.is_dir())
    else {
        eprintln!("skip: SARAGOSSA_RUST_MTP_VERIFIER_TEST_MODEL absent");
        return None;
    };
    let args = ServeArgs::parse([
        "--model".to_string(),
        format!("dense={}", path.display()),
        "--backend".to_string(),
        "metal".to_string(),
    ])
    .expect("invariant: args valides");
    Some(ServeState::new(&args))
}

fn request(prompt: &str) -> ChatCompletionRequest {
    serde_json::from_value(serde_json::json!({
        "model": "dense",
        "messages": [{"role": "user", "content": prompt}],
        "stream": true,
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
    }))
    .expect("invariant: requête test valide")
}

/// Textes du chemin solo (MTP si actif), requête par requête.
fn sequential(state: &mut ServeState) -> Vec<String> {
    PROMPTS
        .iter()
        .map(|prompt| {
            state
                .complete_streaming(request(prompt), None, |_| Ok(()))
                .expect("invariant: complétion solo")
                .content
        })
        .collect()
}

/// Lance le lot ; `fail` désigne le client qui part après deux deltas.
/// Rend l'issue finale de chaque client, son texte streamé et le nombre de
/// pas multi-flux.
fn batch(
    state: &mut ServeState,
    fail: Option<usize>,
) -> (Vec<Result<String, String>>, Vec<String>, usize) {
    let mut texts = vec![String::new(); PROMPTS.len()];
    let mut done: Vec<Option<Result<String, String>>> = vec![None; PROMPTS.len()];
    let multi_steps = {
        let mut sinks: Vec<Box<dyn CompletionSink + '_>> = texts
            .iter_mut()
            .zip(done.iter_mut())
            .enumerate()
            .map(|(index, (text, done))| {
                Box::new(Collect {
                    text,
                    done,
                    fail_after: (fail == Some(index)).then_some(2),
                    deltas: 0,
                }) as Box<dyn CompletionSink + '_>
            })
            .collect();
        let requests = PROMPTS.iter().map(|prompt| request(prompt)).collect();
        state.complete_streaming_batch(requests, &mut sinks).1
    };
    let done = done
        .into_iter()
        .map(|done| done.expect("invariant: chaque client terminé"))
        .collect();
    (done, texts, multi_steps)
}

#[test]
#[ignore = "requiert le modèle dense réel et SARAGOSSA_RUST_LIGHTBATCH_MULTI=1"]
fn batch_matches_sequential_with_multi_steps() {
    let Some(mut state) = real_state() else {
        return;
    };
    assert!(state.batch_ready("dense"), "modèle non éligible au lot");
    let solo = sequential(&mut state);

    let (results, streamed, multi_steps) = batch(&mut state, None);

    assert!(multi_steps > 0, "aucun pas multi-flux (largeur ≥ 2)");
    for (index, result) in results.into_iter().enumerate() {
        let content = result.expect("invariant: flux du lot terminé");
        assert_eq!(content, solo[index], "flux {index} ≠ solo");
        assert_eq!(streamed[index], solo[index], "flux {index} streamé ≠ solo");
    }
}

#[test]
#[ignore = "requiert le modèle dense réel et SARAGOSSA_RUST_LIGHTBATCH_MULTI=1"]
fn departed_client_does_not_stop_the_batch() {
    let Some(mut state) = real_state() else {
        return;
    };
    let solo = sequential(&mut state);

    let (results, _, multi_steps) = batch(&mut state, Some(1));

    assert!(multi_steps > 0, "aucun pas multi-flux (largeur ≥ 2)");
    let mut results = results.into_iter();
    let first = results.next().expect("invariant: flux 0");
    let departed = results.next().expect("invariant: flux 1");
    let last = results.next().expect("invariant: flux 2");
    assert!(departed.is_err(), "le client parti doit finir en erreur");
    assert_eq!(first.expect("invariant: flux 0 terminé"), solo[0]);
    assert_eq!(last.expect("invariant: flux 2 terminé"), solo[2]);
}
