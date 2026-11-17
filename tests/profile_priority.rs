//! Priorité env > profil > défaut ; un seul test, car l'installation est globale au processus.

use saragossa::runtime_flags::{env_var, Profile};

#[test]
fn env_beats_profile_and_profile_beats_default() {
    std::env::set_var("SARAGOSSA_SERVE_MAX_STREAMS", "6");
    let profile = Profile::parse("[serve]\nqueue = 3\nmax_streams = 5\n").expect("profil");
    assert!(profile.install());
    assert_eq!(env_var("SARAGOSSA_SERVE_QUEUE").as_deref(), Some("3"));
    assert_eq!(env_var("SARAGOSSA_SERVE_MAX_STREAMS").as_deref(), Some("6"));
    assert_eq!(env_var("SARAGOSSA_SERVE_MODEL_POOL"), None);
}
