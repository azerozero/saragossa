//! Rechargement SIGHUP des seules limites sûres à chaud.
//!
//! Chaud : `serve.read_timeout_secs` (nouvelles connexions) et
//! `serve.max_tokens_cap` (prochaine requête). Le reste du profil est lu une
//! fois (caches, files, pools à `OnceLock`) : si une autre clé change, tout le
//! rechargement est refusé avec la liste des clés, rien n'est appliqué.
//! La CLI garde sa priorité : les options sont réanalysées avec le nouveau profil.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

use saragossa::runtime_flags::{Profile, Resolved, HOT_KEYS};

use super::args::ServeArgs;

// 0 = aucune valeur rechargée : on garde celle du lancement.
static READ_TIMEOUT_SECS: AtomicU64 = AtomicU64::new(0);
static MAX_TOKENS_CAP: AtomicUsize = AtomicUsize::new(0);

/// Timeout de lecture courant, `launch` tant qu'aucun rechargement n'a eu lieu.
pub(super) fn read_timeout(launch: Duration) -> Duration {
    match READ_TIMEOUT_SECS.load(Ordering::Relaxed) {
        0 => launch,
        secs => Duration::from_secs(secs),
    }
}

/// Plafond `max_tokens` courant, `launch` tant qu'aucun rechargement n'a eu lieu.
pub(super) fn max_tokens_cap(launch: usize) -> usize {
    match MAX_TOKENS_CAP.load(Ordering::Relaxed) {
        0 => launch,
        cap => cap,
    }
}

/// Relit le profil et publie les clés chaudes ; erreur = rien n'a changé.
fn reload(
    source: Option<&str>,
    sets: &[String],
    cli: &[String],
    current: &Profile,
) -> Result<(Profile, Vec<&'static str>), String> {
    if source.is_none() {
        return Err("aucun profil chargé, rien à recharger".to_string());
    }
    let next = Resolved::resolve(source, sets)
        .map_err(|e| e.to_string())?
        .profile;
    let changed = current.changed_keys(&next);
    let frozen: Vec<_> = changed
        .iter()
        .filter(|key| !HOT_KEYS.contains(key))
        .collect();
    if !frozen.is_empty() {
        return Err(format!("redémarrage requis pour {frozen:?}"));
    }
    let args = ServeArgs::parse_with(&next, cli.to_vec()).map_err(|e| e.to_string())?;
    READ_TIMEOUT_SECS.store(args.read_timeout.as_secs(), Ordering::Relaxed);
    MAX_TOKENS_CAP.store(args.max_tokens_cap, Ordering::Relaxed);
    Ok((next, changed))
}

// Un bit par numéro de signal levé et pas encore consommé.
#[cfg(unix)]
static RAISED: AtomicU64 = AtomicU64::new(0);

#[cfg(unix)]
extern "C" fn on_signal(signal: libc::c_int) {
    // NOTE: un RMW atomique sans verrou, seule opération du gestionnaire.
    RAISED.fetch_or(1 << signal, Ordering::Relaxed);
}

/// Remplace l'action de `signal` par la levée d'un drapeau, lu par [`take`].
#[cfg(unix)]
#[allow(unsafe_code)]
pub(super) fn trap(signal: libc::c_int) {
    // SAFETY: le gestionnaire ne fait qu'un `fetch_or` atomique, async-signal-safe.
    unsafe {
        libc::signal(
            signal,
            on_signal as extern "C" fn(libc::c_int) as libc::sighandler_t,
        );
    }
}

/// Consomme le drapeau de `signal` : vrai s'il a été levé depuis le dernier appel.
#[cfg(unix)]
pub(super) fn take(signal: libc::c_int) -> bool {
    RAISED.fetch_and(!(1 << signal), Ordering::Relaxed) & (1 << signal) != 0
}

/// Installe le gestionnaire SIGHUP et son thread de rechargement.
#[cfg(unix)]
pub(super) fn watch(source: Option<String>, sets: Vec<String>, cli: Vec<String>, profile: Profile) {
    trap(libc::SIGHUP);
    std::thread::spawn(move || {
        let mut current = profile;
        loop {
            std::thread::sleep(Duration::from_millis(250));
            if !take(libc::SIGHUP) {
                continue;
            }
            match reload(source.as_deref(), &sets, &cli, &current) {
                Ok((next, changed)) => {
                    eprintln!("saragossa serve: profil rechargé, clés modifiées {changed:?}");
                    current = next;
                }
                Err(message) => eprintln!("saragossa serve: rechargement refusé: {message}"),
            }
        }
    });
}

#[cfg(not(unix))]
pub(super) fn watch(_: Option<String>, _: Vec<String>, _: Vec<String>, _: Profile) {}

#[cfg(test)]
mod tests {
    use super::*;

    // Un seul test : les atomiques sont globales au processus.
    #[test]
    fn hot_keys_apply_and_frozen_keys_are_refused() {
        // Les atomiques sont globales : les remettre à 0 même en cas d'échec.
        struct Reset;
        impl Drop for Reset {
            fn drop(&mut self) {
                READ_TIMEOUT_SECS.store(0, Ordering::Relaxed);
                MAX_TOKENS_CAP.store(0, Ordering::Relaxed);
            }
        }
        let _reset = Reset;
        let dir = std::env::temp_dir().join(format!("saragossa-reload-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("dossier temporaire");
        let file = dir.join("p.toml");
        let path = file.to_str().expect("utf8").to_string();
        let write = |text: &str| std::fs::write(&file, text).expect("écriture");
        write("[serve]\nqueue = 4\nmax_tokens_cap = 100\n");
        let current = Resolved::resolve(Some(&path), &[]).expect("profil").profile;
        assert_eq!(max_tokens_cap(4096), 4096);

        write("[serve]\nqueue = 4\nmax_tokens_cap = 50\nread_timeout_secs = 7\n");
        let (next, changed) = reload(Some(&path), &[], &[], &current).expect("chaud");
        assert_eq!(changed, ["serve.read_timeout_secs", "serve.max_tokens_cap"]);
        assert_eq!(max_tokens_cap(4096), 50);
        assert_eq!(
            read_timeout(Duration::from_secs(30)),
            Duration::from_secs(7)
        );

        write("[serve]\nqueue = 9\nmax_tokens_cap = 10\n");
        let error = reload(Some(&path), &[], &[], &next).expect_err("froid");
        assert!(error.contains("serve.queue"), "{error}");
        assert_eq!(max_tokens_cap(4096), 50, "refus sans effet partiel");

        write("[serve\n");
        assert!(reload(Some(&path), &[], &[], &next).is_err());
        assert!(reload(None, &[], &[], &next).is_err());
        let _ = std::fs::remove_dir_all(dir);
    }
}
