//! Politique du decode MTP partagée par la CLI, `run` et `serve`.

use std::time::Duration;

use saragossa::{GenerationOutput, GenerationTimings, ModelAssets, SpeculativeOutput};

use super::protocol::{ChatCompletionRequest, ResponseFormatMode};

const MTP_DECODE_ENV: &str = "SARAGOSSA_RUST_MTP_DECODE";
const MTP_MAX_DRAFT_ENV: &str = "SARAGOSSA_RUST_MTP_MAX_DRAFT";

/// Indique si le modèle peut charger automatiquement sa tête MTP.
pub(crate) fn active_for(assets: &ModelAssets, backend: crate::RuntimeKind) -> bool {
    decode_enabled(saragossa::runtime_flags::env_var(MTP_DECODE_ENV).as_deref())
        && supports_model(&assets.config, &assets.mtp, backend)
}

fn supports_model(
    config: &saragossa::ModelConfig,
    mtp: &saragossa::MtpWeightsInfo,
    backend: crate::RuntimeKind,
) -> bool {
    backend == crate::RuntimeKind::Metal
        && mtp.path.is_some()
        && mtp.is_available()
        && !config.is_moe()
}

/// Le MTP actuel ne vérifie que le greedy non contraint.
pub(crate) fn supports_options(options: &saragossa::GenerationOptions) -> bool {
    options.temperature.abs() <= f32::EPSILON
        && options.token_constraint.is_none()
        && options.stop_sequences.is_empty()
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
pub(crate) fn max_draft_tokens() -> usize {
    saragossa::runtime_flags::env_var(MTP_MAX_DRAFT_ENV)
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
    fn model_requires_dense_config_and_detected_head() {
        let mut config: saragossa::ModelConfig = serde_json::from_value(serde_json::json!({
            "model_type": "qwen3_5_text", "hidden_size": 2, "num_hidden_layers": 1,
            "num_attention_heads": 1, "num_key_value_heads": 1,
            "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "vocab_size": 3
        }))
        .expect("invariant: configuration test valide");
        let mut mtp = saragossa::MtpWeightsInfo::default();
        assert!(!supports_model(&config, &mtp, crate::RuntimeKind::Metal));
        mtp.path = Some("mtp.safetensors".into());
        assert!(!supports_model(&config, &mtp, crate::RuntimeKind::Metal));
        mtp.has_mtp_tensors = true;
        assert!(!supports_model(&config, &mtp, crate::RuntimeKind::Metal));
        mtp.has_fc_weight = true;
        assert!(supports_model(&config, &mtp, crate::RuntimeKind::Metal));
        assert!(!supports_model(&config, &mtp, crate::RuntimeKind::Cpu));
        config.num_experts = Some(8);
        assert!(!supports_model(&config, &mtp, crate::RuntimeKind::Metal));
        config.num_experts = None;
        config.model_type = "qwen3_moe".into();
        assert!(!supports_model(&config, &mtp, crate::RuntimeKind::Metal));
        config.model_type = "qwen3_5_text".into();
        mtp.path = None;
        assert!(!supports_model(&config, &mtp, crate::RuntimeKind::Metal));
    }

    #[test]
    fn options_require_finite_greedy_temperature() {
        let mut options = saragossa::GenerationOptions {
            temperature: 0.0,
            ..Default::default()
        };
        assert!(supports_options(&options));
        for temperature in [0.7, -0.7, f32::NAN, f32::INFINITY] {
            options.temperature = temperature;
            assert!(!supports_options(&options));
        }
    }

    #[test]
    fn options_reject_custom_stop_sequences() {
        let options = saragossa::GenerationOptions {
            temperature: 0.0,
            stop_sequences: vec![vec![1, 2]],
            ..Default::default()
        };
        assert!(!supports_options(&options));
    }

    #[test]
    fn options_reject_guided_greedy() {
        struct Constraint;
        impl saragossa::guided::TokenConstraint for Constraint {
            fn mask_logits(&self, _: &mut [f32]) -> saragossa::Result<()> {
                Ok(())
            }
            fn accept_token(&self, _: usize) -> saragossa::Result<()> {
                Ok(())
            }
            fn is_finished(&self) -> bool {
                false
            }
        }
        let options = saragossa::GenerationOptions {
            temperature: 0.0,
            token_constraint: Some(std::sync::Arc::new(Constraint)),
            ..Default::default()
        };
        assert!(!supports_options(&options));
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
