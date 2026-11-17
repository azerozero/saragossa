//! Relais sans coupure : SIGUSR2 remplace le binaire servi sans fermer l'écoute.
//!
//! Sur SIGUSR2, le processus relance le binaire du même chemin (ou
//! `--upgrade-binary PATH`) avec les mêmes arguments. L'enfant hérite de la
//! socket d'écoute (`SARAGOSSA_LISTEN_FDS`, seul fd dont FD_CLOEXEC est retiré,
//! avec l'extrémité d'une paire de sockets `SARAGOSSA_RELAY_READY_FD`), vérifie
//! que c'est bien une socket en écoute à la même adresse, l'adopte au lieu de
//! la lier, puis écrit un octet sur la paire juste avant d'accepter. Le parent
//! cesse alors d'accepter, laisse finir ses connexions en cours (une requête
//! par connexion) et sort, au plus tard après `serve.drain_timeout_secs`.
//! Si l'enfant sort avant d'être prêt, le parent continue et journalise.
//!
//! Limites : les listeners enhance h2/WebTransport ne sont pas transmis ;
//! l'enfant échoue à les lier, et le parent reste donc en place. Les modèles
//! préchargés le sont deux fois pendant le relais.

use std::io::{self, Read, Write};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::os::unix::net::UnixStream;
use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;
use std::time::Duration;

use super::error::{ServeError, ServeResult};

const LISTEN_FDS_ENV: &str = "SARAGOSSA_LISTEN_FDS";
const READY_FD_ENV: &str = "SARAGOSSA_RELAY_READY_FD";
const DEFAULT_DRAIN_SECS: u64 = 30;

static DRAINING: AtomicBool = AtomicBool::new(false);
static CONFIG: OnceLock<(PathBuf, Duration)> = OnceLock::new();

/// Retire `--upgrade-binary PATH` de `args` et fige le binaire du relais et
/// le délai de vidage ; à appeler au démarrage, avant tout remplacement.
pub(super) fn configure(args: &mut Vec<String>, drain_secs: Option<u64>) -> ServeResult<()> {
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
    let drain = Duration::from_secs(drain_secs.unwrap_or(DEFAULT_DRAIN_SECS));
    let _ = CONFIG.set((binary, drain));
    Ok(())
}

/// Socket d'écoute héritée d'un parent en relais ; `None` hors relais.
/// L'appelant vérifie qu'elle écoute sur l'adresse attendue.
#[allow(unsafe_code)]
pub(super) fn inherited() -> ServeResult<Option<OwnedFd>> {
    let Some(fd) = env_fd(LISTEN_FDS_ENV)? else {
        return Ok(None);
    };
    // NOTE: macOS n'expose pas SO_ACCEPTCONN ; on vérifie une socket flux
    // ici, et l'appelant vérifie son adresse locale.
    let mut kind: libc::c_int = 0;
    let mut len = std::mem::size_of::<libc::c_int>() as libc::socklen_t;
    // SAFETY: tampon et longueur valides pour un `c_int` ; un fd invalide ou
    // qui n'est pas une socket fait seulement échouer l'appel.
    let rc = unsafe {
        libc::getsockopt(
            fd,
            libc::SOL_SOCKET,
            libc::SO_TYPE,
            (&mut kind as *mut libc::c_int).cast(),
            &mut len,
        )
    };
    if rc != 0 || kind != libc::SOCK_STREAM {
        return Err(ServeError::args(format!(
            "{LISTEN_FDS_ENV}={fd} n'est pas une socket flux"
        )));
    }
    // SAFETY: socket flux vérifiée ci-dessus, transmise à ce seul
    // processus par le parent ; personne d'autre ici ne la ferme.
    let owned = unsafe { OwnedFd::from_raw_fd(fd) };
    // SAFETY: fd possédé et valide ; rétablit FD_CLOEXEC pour les sous-processus.
    unsafe { libc::fcntl(owned.as_raw_fd(), libc::F_SETFD, libc::FD_CLOEXEC) };
    eprintln!("saragossa serve relais: socket d'écoute héritée (fd {fd})");
    Ok(Some(owned))
}

/// Prévient le parent que l'enfant accepte ; sans effet hors relais.
#[allow(unsafe_code)]
pub(super) fn ready() -> ServeResult<()> {
    let Some(fd) = env_fd(READY_FD_ENV)? else {
        return Ok(());
    };
    // SAFETY: extrémité de la paire transmise par le parent pour ce seul
    // usage ; elle est fermée au drop, après l'octet.
    let mut stream = unsafe { UnixStream::from_raw_fd(fd) };
    stream
        .write_all(b"1")
        .map_err(|e| ServeError::io("signal de relais prêt", e))
}

fn env_fd(name: &str) -> ServeResult<Option<RawFd>> {
    match std::env::var(name) {
        Ok(value) => value
            .parse::<RawFd>()
            .ok()
            .filter(|fd| *fd > 2)
            .map(Some)
            .ok_or_else(|| ServeError::args(format!("{name}={value} invalide"))),
        Err(_) => Ok(None),
    }
}

/// Connexions entrantes jusqu'au relais : attend la lisibilité de `fd` par
/// tranches de 250 ms pour voir `DRAINING` sans dépendre d'un accept bloqué.
///
/// ponytail: si l'enfant prend la connexion entre `poll` et `accept`, l'accept
/// du parent reste bloqué jusqu'à la connexion suivante (servie normalement)
/// ou jusqu'au délai de vidage.
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
            return Some(accept());
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
        let Some((binary, drain)) = CONFIG.get() else {
            eprintln!("saragossa serve relais refusé: non configuré");
            continue;
        };
        match hand_over(binary, listener) {
            Ok(pid) => {
                eprintln!(
                    "saragossa serve relais: enfant pid {pid} prêt, vidage (max {}s)",
                    drain.as_secs()
                );
                DRAINING.store(true, Ordering::Relaxed);
                std::thread::sleep(*drain);
                eprintln!("saragossa serve relais: délai de vidage atteint, sortie");
                std::process::exit(0);
            }
            Err(error) => eprintln!("saragossa serve relais échoué, service maintenu: {error}"),
        }
    });
}

/// Lance l'enfant et attend son octet « prêt » ; renvoie son pid.
#[allow(unsafe_code)]
fn hand_over(binary: &PathBuf, listener: RawFd) -> ServeResult<u32> {
    let io_error = |context: &'static str| move |e| ServeError::io(context, e);
    let (mut parent_end, child_end) = UnixStream::pair().map_err(io_error("paire de relais"))?;
    let ready = child_end.as_raw_fd();
    let mut command = Command::new(binary);
    command
        .args(std::env::args_os().skip(1))
        .env(LISTEN_FDS_ENV, listener.to_string())
        .env(READY_FD_ENV, ready.to_string());
    // SAFETY: entre fork et exec, seul `fcntl` est appelé, async-signal-safe ;
    // il ne retire FD_CLOEXEC qu'aux deux fd transmis.
    unsafe {
        command.pre_exec(move || {
            for fd in [listener, ready] {
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
