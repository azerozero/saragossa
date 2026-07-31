//! Politique du decode MTP automatique dans `serve`.

use std::env;
use std::time::Duration;

use saragossa::{GenerationOutput, GenerationTimings, ModelAssets, SpeculativeOutput};

use super::protocol::{ChatCompletionRequest, ResponseFormatMode};

const MTP_DECODE_ENV: &str = "RETI_RUST_MTP_DECODE";
const MTP_MAX_DRAFT_ENV: &str = "RETI_RUST_MTP_MAX_DRAFT";

/// Indique si le modèle peut charger automatiquement sa tête MTP.
pub(super) fn active_for(assets: &ModelAssets) -> bool {
    decode_enabled(env::var(MTP_DECODE_ENV).ok().as_deref())
        && assets.mtp.path.is_some()
        && assets.config.num_experts.unwrap_or(0) == 0
}

/// Indique si une requête doit emprunter le chemin MTP.
pub(super) fn routes_request(
    request: &ChatCompletionRequest,
    response_format: ResponseFormatMode,
) -> bool {
    matches!(response_format, ResponseFormatMode::Text)
        && request
            .temperature
            .is_none_or(|temperature| temperature.abs() <= f32::EPSILON)
}

/// Renvoie la profondeur draft MTP, avec un minimum strict de un.
pub(super) fn max_draft_tokens() -> usize {
    env::var(MTP_MAX_DRAFT_ENV)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(1)
}

/// Convertit les métriques spéculatives vers le contrat de métriques serveur.
pub(super) fn into_generation_output(
    output: SpeculativeOutput,
    elapsed: Duration,
) -> GenerationOutput {
    let prefill = elapsed.saturating_sub(output.loop_duration);
    let decode_tokens = output.tokens.len();
    GenerationOutput {
        tokens: output.tokens,
        timings: GenerationTimings {
            prefill,
            decode: output.loop_duration,
            decode_tokens,
        },
    }
}

fn decode_enabled(value: Option<&str>) -> bool {
    !value.is_some_and(|value| {
        let value = value.trim();
        value == "0" || value.eq_ignore_ascii_case("false") || value.eq_ignore_ascii_case("off")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(json: &str) -> ChatCompletionRequest {
        serde_json::from_str(json).expect("invariant: requête test valide")
    }

    #[test]
    fn mtp_is_auto_on_except_for_explicit_kill_switch() {
        assert!(decode_enabled(None));
        assert!(decode_enabled(Some("1")));
        for value in ["0", " false ", "OFF"] {
            assert!(!decode_enabled(Some(value)), "{value} doit désactiver MTP");
        }
    }

    #[test]
    fn routing_accepts_only_unguided_greedy_requests() {
        let absent = request(r#"{"model":"m","messages":[]}"#);
        let zero = request(r#"{"model":"m","messages":[],"temperature":0}"#);
        let sampled = request(r#"{"model":"m","messages":[],"temperature":0.7}"#);
        let negative = request(r#"{"model":"m","messages":[],"temperature":-0.7}"#);

        assert!(routes_request(&absent, ResponseFormatMode::Text));
        assert!(routes_request(&zero, ResponseFormatMode::Text));
        assert!(!routes_request(&sampled, ResponseFormatMode::Text));
        assert!(!routes_request(&negative, ResponseFormatMode::Text));
        assert!(!routes_request(&zero, ResponseFormatMode::JsonObject));
        assert!(!routes_request(&zero, ResponseFormatMode::JsonLines));
    }
}
