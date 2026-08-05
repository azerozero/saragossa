use std::path::PathBuf;

/// Résout une fixture voix de référence (`reti-fr.wav`, `reti-fr.txt`).
///
/// La copie embarquée dans `assets/voices/` est cherchée d'abord : c'est la
/// seule que le miroir public reçoit, `scripts/publish-saragossa.sh` ne
/// synchronisant que `crates/saragossa`. Le repli sur `../../voices/` garde
/// l'arbre reti comme source unique historique tant que les deux coexistent.
///
/// Renvoie `None` si aucune des deux n'existe, pour que les tests concernés
/// se sautent au lieu d'échouer.
pub(crate) fn voice_fixture(name: &str) -> Option<PathBuf> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let embedded = manifest.join("assets/voices").join(name);
    if embedded.is_file() {
        return Some(embedded);
    }
    let workspace = manifest.join("../../voices").join(name);
    workspace.is_file().then_some(workspace)
}

/// Exige une précondition de modèle réel quand le mode strict est actif.
pub(crate) fn require_real_model<T>(candidate: Option<T>, missing: &str) -> Option<T> {
    require_real_model_when(
        candidate,
        crate::runtime_flags::env_var("SARAGOSSA_REQUIRE_REAL_MODELS").as_deref() == Some("1"),
        missing,
    )
}

fn require_real_model_when<T>(candidate: Option<T>, required: bool, missing: &str) -> Option<T> {
    assert!(
        !required || candidate.is_some(),
        "SARAGOSSA_REQUIRE_REAL_MODELS=1: précondition absente: {missing}"
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
    #[should_panic(expected = "SARAGOSSA_REQUIRE_REAL_MODELS=1: précondition absente: fixture")]
    fn missing_real_model_fails_in_strict_mode() {
        let _ = require_real_model_when(None::<()>, true, "fixture");
    }

    #[test]
    fn voice_fixture_resolves_embedded_copy() {
        // La copie embarquée est la seule que reçoit le miroir public : si elle
        // manque, les tests de parité TTS s'y sautent tous en silence.
        let wav = voice_fixture("reti-fr.wav").expect("invariant: fixture wav embarquée");
        assert!(
            wav.ends_with("assets/voices/reti-fr.wav"),
            "résolu: {wav:?}"
        );
        assert!(voice_fixture("reti-fr.txt").is_some());
        assert_eq!(voice_fixture("absente-du-disque.wav"), None);
    }

    #[test]
    fn embedded_voice_fixture_matches_workspace_copy() {
        // Deux copies d'un même fichier divergent tôt ou tard : sans cette
        // garde, reti et le miroir public testeraient des voix différentes.
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        for name in ["reti-fr.wav", "reti-fr.txt"] {
            let workspace = manifest.join("../../voices").join(name);
            if !workspace.is_file() {
                continue; // arbre standalone : une seule copie, rien à comparer
            }
            let embedded = manifest.join("assets/voices").join(name);
            assert_eq!(
                std::fs::read(&embedded).ok(),
                std::fs::read(&workspace).ok(),
                "{name} diverge entre assets/voices et voices/"
            );
        }
    }
}
