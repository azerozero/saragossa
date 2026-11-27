//! `serve --engines N` coupe les caches portés par les poids partagés.
//! Fichier à part : le drapeau est global au process de test.

#[test]
fn disabling_shared_model_caches_turns_off_decoder_and_tts_caches() {
    // Mémo TTS demandé : la coupure doit l'emporter sur l'environnement.
    std::env::set_var("SARAGOSSA_TTS_PREFILL_CACHE", "1");
    saragossa::runtime_flags::disable_shared_model_caches();
    assert!(!saragossa::runtime_flags::shared_model_caches_active());
}
