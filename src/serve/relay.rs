//! Relais sans coupure : SIGUSR2 remplace le binaire servi sans fermer l'écoute.
//!
//! Sur SIGUSR2, le processus relance le binaire du même chemin (ou
//! `--upgrade-binary PATH`) avec les mêmes arguments. L'enfant hérite de la
//! socket d'écoute (`SARAGOSSA_LISTEN_FDS` ; seuls les fd transmis perdent
//! FD_CLOEXEC, avec l'extrémité d'une paire de sockets `SARAGOSSA_RELAY_READY_FD`), vérifie
//! que c'est bien une socket en écoute à la même adresse, l'adopte au lieu de
//! la lier, puis écrit un octet sur la paire juste avant d'accepter. Le parent
//! cesse alors d'accepter, laisse finir ses connexions en cours (une requête
//! par connexion) et sort, au plus tard après `serve.drain_timeout_secs`
//! (30 s, comme `read_timeout_secs`) : un flux plus long est coupé. Un enfant
//! qui ne se dit pas prêt en 10 min est arrêté.
//! Si l'enfant sort avant d'être prêt, le parent continue et journalise.
//!
//! Les listeners enhance suivent dans la même variable (`7,h2=9,wt=10`).
//! HTTP/2 : le parent cesse d'accepter, envoie GOAWAY à ses connexions et
//! attend qu'elles finissent. QUIC : parent et enfant partagent la socket UDP ;
//! l'enfant ne la lit qu'après son annonce « prêt », et dès cette annonce le
//! parent ne la lit plus et ferme ses sessions
//! (CONNECTION_CLOSE, H3_NO_ERROR) pour que les clients se reconnectent chez
//! l'enfant. Une session QUIC ouverte ne migre pas, et l'enfant présente un
//! nouveau certificat auto-signé (empreinte à relire).
//!
//! `serve.relay_mode` : `parallel` garde les deux processus résidents pendant
//! le vidage ; `sequential` lance l'enfant sans précharger, son moteur attend
//! la sortie du parent (requêtes en file, plafonnée par `serve.queue`) puis
//! précharge ; `auto` (défaut) choisit `sequential` seulement si la garde
//! mémoire de `serve` refuse deux fois l'empreinte courante.

use std::io::{self, Read, Write};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::os::unix::net::UnixStream;
use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock, PoisonError};
use std::time::Duration;

use saragossa::memory_guard::{process_footprint_bytes, MemoryGuard};
use saragossa::runtime_flags::ServeSection;

use super::error::{ServeError, ServeResult};

const LISTEN_FDS_ENV: &str = "SARAGOSSA_LISTEN_FDS";
const READY_FD_ENV: &str = "SARAGOSSA_RELAY_READY_FD";
/// Pid du parent dont l'enfant séquentiel attend la sortie.
const SEQUENTIAL_ENV: &str = "SARAGOSSA_RELAY_SEQUENTIAL";
const DEFAULT_DRAIN_SECS: u64 = 30;
/// Attente max du signal « prêt » ; couvre un `--preload` de gros modèles.
const READY_TIMEOUT: Duration = Duration::from_secs(600);

static DRAINING: AtomicBool = AtomicBool::new(false);
static CONFIG: OnceLock<Config> = OnceLock::new();
/// Listeners enhance à transmettre en plus de l'écoute principale.
static SHARED: Mutex<Vec<(&'static str, RawFd)>> = Mutex::new(Vec::new());
/// Listeners enhance hérités, pas encore adoptés.
static INHERITED: Mutex<Vec<(String, RawFd)>> = Mutex::new(Vec::new());
/// Parent à attendre en relais séquentiel, 0 sinon.
static PREDECESSOR: AtomicU32 = AtomicU32::new(0);
static DEFERRED_PRELOAD: AtomicBool = AtomicBool::new(false);
/// Vrai une fois l'octet « prêt » écrit (ou hors relais, au premier `ready`).
static ANNOUNCED: AtomicBool = AtomicBool::new(false);
/// Listeners enhance transmissibles et leur type de socket.
const NAMED: [(&str, libc::c_int); 2] = [("h2", libc::SOCK_STREAM), ("wt", libc::SOCK_DGRAM)];

struct Config {
    binary: PathBuf,
    drain: Duration,
    /// `None` : auto.
    sequential: Option<bool>,
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Retire `--upgrade-binary PATH` de `args` et fige le binaire du relais, le
/// délai de vidage et le mode ; à appeler au démarrage, avant tout remplacement.
pub(super) fn configure(args: &mut Vec<String>, serve: &ServeSection) -> ServeResult<()> {
    let mut binary = None;
    if let Some(at) = args.iter().position(|arg| arg == "--upgrade-binary") {
        if at + 1 >= args.len() {
            return Err(ServeError::args("--upgrade-binary attend une valeur"));
        }
        binary = Some(PathBuf::from(args.remove(at + 1)));
        args.remove(at);
    }
    // NOTE: lu maintenant, car sous Linux le lien du binaire remplacé finit
    // en « (deleted) ».
    let binary = match binary {
        Some(path) => path,
        None => std::env::current_exe().map_err(|e| ServeError::io("binaire courant", e))?,
    };
    let drain = Duration::from_secs(serve.drain_timeout_secs.unwrap_or(DEFAULT_DRAIN_SECS));
    let sequential = match serve.relay_mode.as_deref() {
        None | Some("auto") => None,
        Some("parallel") => Some(false),
        Some("sequential") => Some(true),
        Some(other) => {
            return Err(ServeError::args(format!(
                "serve.relay_mode={other} invalide (auto, parallel, sequential)"
            )))
        }
    };
    let relayed =
        std::env::var_os(LISTEN_FDS_ENV).is_some() || std::env::var_os(READY_FD_ENV).is_some();
    if let Ok(pid) = std::env::var(SEQUENTIAL_ENV) {
        std::env::remove_var(SEQUENTIAL_ENV);
        // NOTE: hors relais, une variable égarée bloquerait le moteur.
        if relayed {
            let pid = pid
                .parse()
                .map_err(|_| ServeError::args(format!("{SEQUENTIAL_ENV}={pid} invalide")))?;
            PREDECESSOR.store(pid, Ordering::Relaxed);
        } else {
            eprintln!("saragossa serve relais: {SEQUENTIAL_ENV} ignorée hors relais");
        }
    }
    let _ = CONFIG.set(Config {
        binary,
        drain,
        sequential,
    });
    Ok(())
}

/// Vrai si le préchargement attend la sortie du parent (relais séquentiel).
pub(super) fn defer_preload(preload: bool) -> bool {
    let deferred = preload && PREDECESSOR.load(Ordering::Relaxed) != 0;
    DEFERRED_PRELOAD.store(deferred, Ordering::Relaxed);
    deferred
}

/// Sur le thread moteur, avant la première requête : en relais séquentiel,
/// attend la sortie du parent (les requêtes restent en file) ; vrai si le
/// préchargement différé est à faire maintenant.
pub(super) fn await_predecessor() -> bool {
    let parent = PREDECESSOR.swap(0, Ordering::Relaxed);
    if parent == 0 {
        return false;
    }
    eprintln!(
        "saragossa serve relais séquentiel: moteur en attente de la sortie du parent {parent}"
    );
    // NOTE: le parent sort au plus tard après son délai de vidage ; l'enfant
    // est alors rattaché à un autre parent.
    while std::os::unix::process::parent_id() == parent {
        std::thread::sleep(Duration::from_millis(50));
    }
    eprintln!("saragossa serve relais séquentiel: parent {parent} sorti, moteur actif");
    DEFERRED_PRELOAD.load(Ordering::Relaxed)
}

/// Vrai une fois l'enfant prêt : ce processus ne doit plus rien accepter.
pub(super) fn draining() -> bool {
    DRAINING.load(Ordering::Relaxed)
}

/// Inscrit un listener enhance à transmettre au prochain relais.
pub(super) fn share(name: &'static str, fd: RawFd) {
    lock(&SHARED).push((name, fd));
}

/// Socket d'écoute héritée d'un parent en relais ; `None` hors relais.
/// L'appelant vérifie qu'elle écoute sur l'adresse attendue. Les listeners
/// enhance nommés sont mis de côté pour [`inherited_named`].
pub(super) fn inherited() -> ServeResult<Option<OwnedFd>> {
    let Ok(value) = std::env::var(LISTEN_FDS_ENV) else {
        return Ok(None);
    };
    // NOTE: les sous-processus ne doivent pas croire hériter de ces fd ; std
    // sérialise ses propres accès à l'environnement.
    std::env::remove_var(LISTEN_FDS_ENV);
    let (main, named) = parse_listen(&value, env_fd(READY_FD_ENV)?)
        .ok_or_else(|| ServeError::args(format!("{LISTEN_FDS_ENV}={value} invalide")))?;
    for (name, fd, kind) in &named {
        if option(*fd, libc::SO_TYPE) != Some(*kind) {
            return Err(ServeError::args(format!(
                "socket {name} héritée (fd {fd}) de type inattendu"
            )));
        }
    }
    let named = named
        .into_iter()
        .map(|(name, fd, _)| (name.to_string(), fd))
        .collect();
    *lock(&INHERITED) = named;
    adopt(main, libc::SOCK_STREAM, "d'écoute").map(Some)
}

/// Listener enhance hérité : nom, fd et type de socket.
type Named = (&'static str, RawFd, libc::c_int);

/// Découpe `principal[,nom=fd]...` : noms connus et uniques, fd uniques,
/// distincts du fd « prêt » ; `None` sinon.
fn parse_listen(value: &str, ready: Option<RawFd>) -> Option<(RawFd, Vec<Named>)> {
    let mut parts = value.split(',');
    let main = parse_fd(parts.next()?)?;
    // NOTE: un fd en double serait possédé deux fois (double fermeture).
    let mut seen: Vec<RawFd> = ready.into_iter().collect();
    if seen.contains(&main) {
        return None;
    }
    seen.push(main);
    let mut named = Vec::new();
    for part in parts {
        let (name, fd) = part.split_once('=')?;
        let &(name, kind) = NAMED.iter().find(|(known, _)| *known == name)?;
        let fd = parse_fd(fd)?;
        if seen.contains(&fd) || named.iter().any(|(known, _, _)| *known == name) {
            return None;
        }
        seen.push(fd);
        named.push((name, fd, kind));
    }
    Some((main, named))
}

/// Vrai une fois ce processus annoncé prêt : il peut lire les sockets
/// partagées sans voler les datagrammes du parent.
pub(super) fn announced() -> bool {
    ANNOUNCED.load(Ordering::Relaxed)
}

/// Listener enhance `name` hérité, de type `kind` (`SOCK_STREAM` ou
/// `SOCK_DGRAM`) ; `None` s'il n'a pas été transmis.
pub(super) fn inherited_named(name: &str, kind: libc::c_int) -> ServeResult<Option<OwnedFd>> {
    let mut named = lock(&INHERITED);
    let Some(at) = named.iter().position(|(n, _)| n == name) else {
        return Ok(None);
    };
    let (_, fd) = named.remove(at);
    drop(named);
    adopt(fd, kind, name).map(Some)
}

/// Vérifie le type de socket `fd` héritée, la possède et rétablit FD_CLOEXEC.
#[allow(unsafe_code)]
fn adopt(fd: RawFd, kind: libc::c_int, label: &str) -> ServeResult<OwnedFd> {
    // NOTE: l'appelant vérifie l'adresse locale ; SO_ACCEPTCONN est contrôlé
    // là où le système l'expose (pas macOS, qui répond ENOPROTOOPT).
    if option(fd, libc::SO_TYPE) != Some(kind) {
        return Err(ServeError::args(format!(
            "socket {label} héritée (fd {fd}) de type inattendu"
        )));
    }
    if kind == libc::SOCK_STREAM && option(fd, libc::SO_ACCEPTCONN) == Some(0) {
        return Err(ServeError::args(format!(
            "socket {label} héritée (fd {fd}) pas en écoute"
        )));
    }
    // SAFETY: socket vérifiée ci-dessus, transmise à ce seul processus par le
    // parent et retirée des listes d'héritage ; personne d'autre ne la ferme.
    let owned = unsafe { OwnedFd::from_raw_fd(fd) };
    // SAFETY: fd possédé et valide ; rétablit FD_CLOEXEC pour les sous-processus.
    if unsafe { libc::fcntl(owned.as_raw_fd(), libc::F_SETFD, libc::FD_CLOEXEC) } != 0 {
        return Err(ServeError::io(
            "FD_CLOEXEC de la socket héritée",
            io::Error::last_os_error(),
        ));
    }
    eprintln!("saragossa serve relais: socket {label} héritée (fd {fd})");
    Ok(owned)
}

/// Prévient le parent que l'enfant accepte ; sans effet hors relais.
/// Ferme d'abord les listeners hérités que ce binaire n'a pas adoptés : une
/// écoute orpheline accepterait des connexions jamais servies. Seuls les fd
/// de `INHERITED`, validés par [`inherited`] (uniques, nom et type connus),
/// sont fermés.
#[allow(unsafe_code)]
pub(super) fn ready() -> ServeResult<()> {
    for (name, fd) in lock(&INHERITED).drain(..) {
        eprintln!("saragossa serve relais: socket {name} héritée inutilisée, fermée");
        // SAFETY: fd validé et transmis par le parent, jamais adopté ici ;
        // retiré de la liste, il n'est fermé qu'une fois.
        drop(unsafe { OwnedFd::from_raw_fd(fd) });
    }
    let Some(fd) = env_fd(READY_FD_ENV)? else {
        ANNOUNCED.store(true, Ordering::Relaxed);
        return Ok(());
    };
    // SAFETY: extrémité de la paire transmise par le parent pour ce seul
    // usage ; elle est fermée au drop, après l'octet.
    let mut stream = unsafe { UnixStream::from_raw_fd(fd) };
    std::env::remove_var(READY_FD_ENV);
    let written = stream
        .write_all(b"1")
        .map_err(|e| ServeError::io("signal de relais prêt", e));
    ANNOUNCED.store(true, Ordering::Relaxed);
    written
}

/// Option socket entière de `fd`, `None` si l'appel échoue.
#[allow(unsafe_code)]
fn option(fd: RawFd, name: libc::c_int) -> Option<libc::c_int> {
    let mut value: libc::c_int = 0;
    let mut len = std::mem::size_of::<libc::c_int>() as libc::socklen_t;
    // SAFETY: tampon et longueur valides pour un `c_int` ; un fd invalide ou
    // qui n'est pas une socket fait seulement échouer l'appel.
    let rc = unsafe {
        libc::getsockopt(
            fd,
            libc::SOL_SOCKET,
            name,
            (&mut value as *mut libc::c_int).cast(),
            &mut len,
        )
    };
    (rc == 0).then_some(value)
}

fn parse_fd(value: &str) -> Option<RawFd> {
    value.parse::<RawFd>().ok().filter(|fd| *fd > 2)
}

fn env_fd(name: &str) -> ServeResult<Option<RawFd>> {
    match std::env::var(name) {
        Ok(value) => parse_fd(&value)
            .map(Some)
            .ok_or_else(|| ServeError::args(format!("{name}={value} invalide"))),
        Err(_) => Ok(None),
    }
}

/// Connexions entrantes jusqu'au relais : attend la lisibilité de `fd` par
/// tranches de 250 ms pour voir `DRAINING` sans dépendre d'un accept bloqué.
///
/// `fd` doit être non bloquant : parent et enfant partagent la file d'accept,
/// et l'un peut prendre la connexion que l'autre a vue par `poll` ;
/// `WouldBlock` renvoie alors au `poll` au lieu de bloquer dans `accept`.
/// O_NONBLOCK est partagé par les deux processus, qui passent tous deux ici.
pub(super) fn incoming<S>(
    fd: RawFd,
    mut accept: impl FnMut() -> io::Result<S>,
) -> impl Iterator<Item = io::Result<S>> {
    std::iter::from_fn(move || loop {
        if DRAINING.load(Ordering::Relaxed) {
            return None;
        }
        let mut poll = libc::pollfd {
            fd,
            events: libc::POLLIN,
            revents: 0,
        };
        #[allow(unsafe_code)]
        // SAFETY: un seul `pollfd` valide, vivant pendant l'appel.
        let ready = unsafe { libc::poll(&mut poll, 1, 250) };
        if ready > 0 {
            match accept() {
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => continue,
                accepted => return Some(accepted),
            }
        }
        if ready < 0 {
            let error = io::Error::last_os_error();
            if error.kind() != io::ErrorKind::Interrupted {
                return Some(Err(error));
            }
        }
    })
}

/// Installe SIGUSR2 et le thread qui mène le relais de la socket `listener`.
pub(super) fn watch(listener: RawFd) {
    super::reload::trap(libc::SIGUSR2);
    std::thread::spawn(move || loop {
        std::thread::sleep(Duration::from_millis(250));
        if !super::reload::take(libc::SIGUSR2) {
            continue;
        }
        let Some(config) = CONFIG.get() else {
            eprintln!("saragossa serve relais refusé: non configuré");
            continue;
        };
        let sequential = config.sequential.unwrap_or_else(over_budget);
        if sequential {
            eprintln!(
                "saragossa serve relais séquentiel: l'enfant attendra la sortie de ce processus"
            );
        }
        match hand_over(&config.binary, listener, sequential) {
            Ok(pid) => {
                eprintln!(
                    "saragossa serve relais: enfant pid {pid} prêt, vidage (max {}s)",
                    config.drain.as_secs()
                );
                DRAINING.store(true, Ordering::Relaxed);
                std::thread::sleep(config.drain);
                eprintln!("saragossa serve relais: délai de vidage atteint, sortie");
                std::process::exit(0);
            }
            Err(error) => eprintln!("saragossa serve relais échoué, service maintenu: {error}"),
        }
    });
}

/// Vrai si la garde mémoire de `serve` refuse un second processus de même
/// empreinte ; faux si elle est désactivée ou la mesure indisponible.
fn over_budget() -> bool {
    process_footprint_bytes()
        .and_then(|footprint| MemoryGuard::serve().projection_over_limit(footprint))
        .is_some()
}

/// Lance l'enfant et attend son octet « prêt » ; renvoie son pid.
#[allow(unsafe_code)]
fn hand_over(binary: &PathBuf, listener: RawFd, sequential: bool) -> ServeResult<u32> {
    let io_error = |context: &'static str| move |e| ServeError::io(context, e);
    let (mut parent_end, child_end) = UnixStream::pair().map_err(io_error("paire de relais"))?;
    let ready = child_end.as_raw_fd();
    let mut fds = vec![listener, ready];
    let mut listen = listener.to_string();
    for (name, fd) in lock(&SHARED).iter() {
        listen.push_str(&format!(",{name}={fd}"));
        fds.push(*fd);
    }
    let mut command = Command::new(binary);
    command
        .args(std::env::args_os().skip(1))
        .env(LISTEN_FDS_ENV, listen)
        .env(READY_FD_ENV, ready.to_string());
    if sequential {
        command.env(SEQUENTIAL_ENV, std::process::id().to_string());
    }
    // SAFETY: entre fork et exec, seul `fcntl` est appelé, async-signal-safe,
    // sur un vecteur déjà alloué ; il ne retire FD_CLOEXEC qu'aux fd transmis.
    unsafe {
        command.pre_exec(move || {
            for &fd in &fds {
                if libc::fcntl(fd, libc::F_SETFD, 0) != 0 {
                    return Err(io::Error::last_os_error());
                }
            }
            Ok(())
        });
    }
    let mut child = command
        .spawn()
        .map_err(io_error("lancement du binaire de relais"))?;
    // Sans la copie du parent, la fin de l'enfant se lit comme une fin de flux.
    drop(child_end);
    parent_end
        .set_read_timeout(Some(READY_TIMEOUT))
        .map_err(io_error("délai du signal prêt"))?;
    let mut byte = [0u8; 1];
    match parent_end.read(&mut byte) {
        Ok(1) => Ok(child.id()),
        outcome => {
            // NOTE: l'enfant n'est pas prêt ; il est arrêté s'il vit encore.
            let _ = child.kill();
            let status = child.wait().map_err(io_error("attente de l'enfant"))?;
            Err(ServeError::args(format!(
                "enfant {} sorti avant d'être prêt ({status}, lecture {outcome:?})",
                child.id()
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn listen_fds_rejects_duplicates_and_unknown_names() {
        let (main, named) = parse_listen("3,h2=8,wt=9", Some(14)).expect("valide");
        assert_eq!(main, 3);
        assert_eq!(
            named,
            vec![("h2", 8, libc::SOCK_STREAM), ("wt", 9, libc::SOCK_DGRAM)]
        );
        for bad in [
            "3,h2=3",
            "3,h2=8,wt=8",
            "3,h2=8,h2=9",
            "3,ws=8",
            "3,h2=14",
            "14",
            "3,h2",
            "2",
            "",
        ] {
            assert!(parse_listen(bad, Some(14)).is_none(), "{bad}");
        }
    }
}
