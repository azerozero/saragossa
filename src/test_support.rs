/// Exige une précondition de modèle réel quand le mode strict est actif.
pub(crate) fn require_real_model<T>(candidate: Option<T>, missing: &str) -> Option<T> {
    require_real_model_when(
        candidate,
        std::env::var("RETI_REQUIRE_REAL_MODELS").as_deref() == Ok("1"),
        missing,
    )
}

fn require_real_model_when<T>(candidate: Option<T>, required: bool, missing: &str) -> Option<T> {
    assert!(
        !required || candidate.is_some(),
        "RETI_REQUIRE_REAL_MODELS=1: précondition absente: {missing}"
    );
    candidate
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_real_model_still_skips_without_strict_mode() {
        assert_eq!(require_real_model_when(None::<()>, false, "fixture"), None);
    }

    #[test]
    #[should_panic(expected = "RETI_REQUIRE_REAL_MODELS=1: précondition absente: fixture")]
    fn missing_real_model_fails_in_strict_mode() {
        let _ = require_real_model_when(None::<()>, true, "fixture");
    }
}
