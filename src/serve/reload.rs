//! Rechargement SIGHUP des limites sûres à chaud et de la liste des modèles.
//!
//! Chaud : `serve.read_timeout_secs` (nouvelles connexions),
//! `serve.max_tokens_cap` (prochaine requête) et `[[models]]` : la liste
//! acceptée sert `/v1/models` aussitôt, le thread moteur l'applique entre deux
//! requêtes (`ServeState::set_models`), donc une requête en cours finit sur son
//! modèle avant qu'il soit déchargé.
//! Limites : une référence HF nue est refusée à chaud (chemin local avec
//! `config.json` requis, sauf alias déjà servi) ; modifier un modèle (même
//! alias, autre chemin) vide son cache de préfixe et ses sessions, comme tout
//! déchargement. Le reste du profil est lu une
//! fois (caches, files, pools à `OnceLock`) : si une autre clé change, tout le
//! rechargement est refusé avec la liste des clés, rien n'est appliqué.
//! La CLI garde sa priorité : les options sont réanalysées avec le nouveau profil.

use std::path::Path;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Duration;

use saragossa::runtime_flags::{Profile, Resolved, HOT_KEYS};

use super::args::{config_from_entry, ServeArgs, ServeModelConfig};

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

// Liste de modèles acceptée (None avant la fin du lancement) et celle que le
// thread moteur n'a pas encore appliquée.
static MODELS: Mutex<Option<Vec<ServeModelConfig>>> = Mutex::new(None);
static PENDING: Mutex<Option<Vec<ServeModelConfig>>> = Mutex::new(None);

fn lock(
    slot: &Mutex<Option<Vec<ServeModelConfig>>>,
) -> std::sync::MutexGuard<'_, Option<Vec<ServeModelConfig>>> {
    slot.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Sérialise les tests qui lisent ou écrivent les listes globales.
#[cfg(test)]
pub(super) static TEST_LOCK: Mutex<()> = Mutex::new(());

/// Déclare la liste du lancement, HF résolus ; avant, SIGHUP est refusé.
pub(super) fn set_launch_models(models: &[ServeModelConfig]) {
    *lock(&MODELS) = Some(models.to_vec());
}

/// Identifiants servis par `/v1/models`, `launch` tant qu'aucun rechargement.
pub(super) fn model_ids(launch: &[String]) -> Vec<String> {
    match &*lock(&MODELS) {
        Some(models) => models.iter().map(|m| m.id.clone()).collect(),
        None => launch.to_vec(),
    }
}

/// Liste acceptée que le moteur doit encore appliquer, une seule fois. Elle
/// devient la liste de `/v1/models` ici, quand le moteur la prend.
pub(super) fn take_pending_models() -> Option<Vec<ServeModelConfig>> {
    let models = lock(&PENDING).take()?;
    *lock(&MODELS) = Some(models.clone());
    Some(models)
}

/// Dernière liste acceptée, appliquée ou non : base du prochain rechargement.
fn effective_models() -> Option<Vec<ServeModelConfig>> {
    lock(&PENDING).clone().or_else(|| lock(&MODELS).clone())
}

/// Un même modèle : alias et chemin identiques (sinon retrait + ajout).
pub(super) fn same_model(a: &ServeModelConfig, b: &ServeModelConfig) -> bool {
    a.id == b.id && a.path == b.path
}

/// Ids retirés et ajoutés de `old` à `new` ; un modèle modifié figure dans les deux.
pub(super) fn diff_models(
    old: &[ServeModelConfig],
    new: &[ServeModelConfig],
) -> (Vec<String>, Vec<String>) {
    let only = |from: &[ServeModelConfig], other: &[ServeModelConfig]| {
        from.iter()
            .filter(|m| !other.iter().any(|o| same_model(m, o)))
            .map(|m| m.id.clone())
            .collect()
    };
    (only(old, new), only(new, old))
}

/// Valide toute la liste ; une référence HF nue n'est jamais résolue à chaud :
/// elle n'est acceptée que si son alias est déjà servi (chemin résolu gardé),
/// les autres chemins doivent être des dossiers avec un `config.json` lisible. Toutes les erreurs
/// sont listées.
fn validate_models(
    mut wanted: Vec<ServeModelConfig>,
    current: &[ServeModelConfig],
) -> Result<Vec<ServeModelConfig>, String> {
    let mut errors = Vec::new();
    for model in &mut wanted {
        if let Some(old) = current.iter().find(|c| c.id == model.id) {
            if model.hf_repo.is_some() {
                model.path = old.path.clone();
                model.hf_repo = None;
            }
        }
        if let Err(e) = readable_dir(&model.path) {
            errors.push(format!(
                "modèle {}: {}: {e}",
                model.id,
                model.path.display()
            ));
        }
    }
    if errors.is_empty() {
        Ok(wanted)
    } else {
        Err(errors.join("; "))
    }
}

fn readable_dir(path: &Path) -> std::io::Result<()> {
    std::fs::File::open(path.join("config.json")).map(|_| ())
}

/// Nouveau profil, clés modifiées, ids de modèles retirés et ajoutés.
type Reloaded = (Profile, Vec<&'static str>, (Vec<String>, Vec<String>));

/// Relit le profil et publie les clés chaudes ; erreur = rien n'a changé.
fn reload(
    source: Option<&str>,
    sets: &[String],
    cli: &[String],
    current: &Profile,
    current_models: &[ServeModelConfig],
) -> Result<Reloaded, String> {
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
    // Base : la liste du lancement (CLI, prompt, TOML) ; sans `[[models]]` dans
    // le profil rien ne change. Sinon les modèles venus du TOML précédent sont
    // remplacés par ceux du nouveau, ceux de la CLI ou du prompt restent.
    let models = if next.models.is_empty() {
        current_models.to_vec()
    } else {
        let mut list: Vec<ServeModelConfig> = current_models
            .iter()
            .filter(|m| !current.models.iter().any(|e| e.id == m.id))
            .cloned()
            .collect();
        for entry in &next.models {
            if !list.iter().any(|m| m.id == entry.id) {
                list.push(config_from_entry(entry).map_err(|e| e.to_string())?);
            }
        }
        list
    };
    let models = validate_models(models, current_models)?;
    let diff = diff_models(current_models, &models);
    READ_TIMEOUT_SECS.store(args.read_timeout.as_secs(), Ordering::Relaxed);
    MAX_TOKENS_CAP.store(args.max_tokens_cap, Ordering::Relaxed);
    if !(diff.0.is_empty() && diff.1.is_empty()) {
        *lock(&PENDING) = Some(models);
    }
    Ok((next, changed, diff))
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
            let Some(models) = effective_models() else {
                eprintln!("saragossa serve: rechargement refusé: lancement en cours");
                continue;
            };
            match reload(source.as_deref(), &sets, &cli, &current, &models) {
                Ok((next, changed, (removed, added))) => {
                    eprintln!(
                        "saragossa serve: profil rechargé, clés modifiées {changed:?}, modèles retirés {removed:?} ajoutés {added:?}"
                    );
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
        let (next, changed, _) = reload(Some(&path), &[], &[], &current, &[]).expect("chaud");
        assert_eq!(changed, ["serve.read_timeout_secs", "serve.max_tokens_cap"]);
        assert_eq!(max_tokens_cap(4096), 50);
        assert_eq!(
            read_timeout(Duration::from_secs(30)),
            Duration::from_secs(7)
        );

        write("[serve]\nqueue = 9\nmax_tokens_cap = 10\n");
        let error = reload(Some(&path), &[], &[], &next, &[]).expect_err("froid");
        assert!(error.contains("serve.queue"), "{error}");
        assert_eq!(max_tokens_cap(4096), 50, "refus sans effet partiel");

        write("[serve\n");
        assert!(reload(Some(&path), &[], &[], &next, &[]).is_err());
        assert!(reload(None, &[], &[], &next, &[]).is_err());
        let _ = std::fs::remove_dir_all(dir);
    }

    fn model(id: &str, path: &str) -> ServeModelConfig {
        ServeModelConfig {
            id: id.to_string(),
            path: path.into(),
            hf_repo: None,
        }
    }

    #[test]
    fn diff_lists_removed_added_and_modified() {
        let old = [model("a", "/x/a"), model("b", "/x/b"), model("c", "/x/c")];
        let new = [model("a", "/x/a"), model("b", "/y/b"), model("d", "/x/d")];
        let (removed, added) = diff_models(&old, &new);
        assert_eq!(removed, ["b", "c"]);
        assert_eq!(added, ["b", "d"]);
        assert_eq!(diff_models(&old, &old), (vec![], vec![]));
    }

    // Un seul test touche aux listes globales MODELS et PENDING.
    #[test]
    fn model_list_is_atomic_and_applied_by_the_engine() {
        let _serial = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // Listes globales : les vider même en cas d'échec.
        struct Reset;
        impl Drop for Reset {
            fn drop(&mut self) {
                *lock(&MODELS) = None;
                *lock(&PENDING) = None;
            }
        }
        let _reset = Reset;
        let dir = std::env::temp_dir().join(format!("saragossa-models-{}", std::process::id()));
        let (a, b) = (dir.join("a"), dir.join("b"));
        std::fs::create_dir_all(&a).expect("dossier a");
        std::fs::create_dir_all(&b).expect("dossier b");
        for d in [&a, &b] {
            std::fs::write(d.join("config.json"), "{}").expect("config");
        }
        let file = dir.join("p.toml");
        let path = file.to_str().expect("utf8").to_string();
        let write = |text: String| std::fs::write(&file, text).expect("écriture");
        let toml = |entries: &[(&str, &Path)]| {
            entries
                .iter()
                .map(|(id, p)| {
                    format!(
                        "[[models]]\nid = \"{id}\"\npath = {:?}\n",
                        p.to_str().expect("utf8")
                    )
                })
                .collect::<String>()
        };
        let start = [
            model("a", a.to_str().expect("utf8")),
            model("b", b.to_str().expect("utf8")),
        ];
        set_launch_models(&start);
        write(toml(&[("a", &a), ("b", &b)]));
        let current = Resolved::resolve(Some(&path), &[]).expect("profil").profile;

        // Erreur atomique : un chemin illisible bloque aussi l'ajout valide.
        write(toml(&[("a", &a), ("c", &b), ("d", &dir.join("absent"))]));
        let error = reload(Some(&path), &[], &[], &current, &start).expect_err("refus");
        assert!(error.contains("modèle d"), "{error}");
        assert!(take_pending_models().is_none());
        assert_eq!(model_ids(&[]), ["a", "b"], "ancien état gardé");

        // Retrait de b : /v1/models le perd aussitôt, le moteur l'applique ensuite.
        write(toml(&[("a", &a)]));
        let (_, _, (removed, added)) =
            reload(Some(&path), &[], &[], &current, &start).expect("retrait");
        assert_eq!((removed, added), (vec!["b".to_string()], vec![]));
        assert_eq!(model_ids(&[]), ["a", "b"], "publié par le moteur seulement");
        let pending = take_pending_models().expect("liste en attente");
        assert_eq!(pending.len(), 1);
        assert_eq!(model_ids(&[]), ["a"]);
        assert!(take_pending_models().is_none());

        // Modèle choisi au prompt : ni CLI ni TOML, il survit aux rechargements.
        let picked = model("p", a.to_str().expect("utf8"));
        let launch = [start[0].clone(), picked.clone()];
        write(String::new());
        let bare = Resolved::resolve(Some(&path), &[]).expect("profil").profile;
        reload(Some(&path), &[], &[], &bare, &launch).expect("sans [[models]]");
        assert!(take_pending_models().is_none(), "liste intacte");
        write(toml(&[("b", &b)]));
        reload(Some(&path), &[], &[], &bare, &launch).expect("avec [[models]]");
        let ids: Vec<_> = take_pending_models()
            .expect("liste")
            .into_iter()
            .map(|m| m.id)
            .collect();
        assert_eq!(ids, ["a", "p", "b"]);
        let _ = std::fs::remove_dir_all(dir);
    }
}
