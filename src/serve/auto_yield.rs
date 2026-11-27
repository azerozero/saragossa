//! Cédage automatique du GPU (`--auto-yield`, `serve.auto_yield`, opt-in).
//!
//! Un thread lit chaque seconde le temps écoulé depuis la dernière entrée
//! clavier ou souris matérielle (`CGEventSourceSecondsSinceLastEventType` sur
//! la source HID, celle de `HIDIdleTime` : les événements synthétiques ne
//! comptent pas). Entrée de moins de [`ACTIVE`] : la part de
//! GPU tombe à `serve.auto_yield_share` ; après `serve.auto_yield_idle_secs`
//! sans entrée, la part configurée revient. Entre les deux seuils l'état ne
//! bouge pas (hystérésis). Seul le rapport cyclique change (sommeil après
//! chaque command buffer), jamais le calcul : les tokens sont identiques.
//! Le pool rayon n'est pas touché, il resterait figé à la part basse.
//!
//! En mode dynamique (défaut, `serve.auto_yield_dynamic`), la part basse suit
//! l'activité GPU des autres processus : `100 - la leur - 10`, bornée à
//! 10..=50 (voir [`dynamic_share`]), moyennée sur [`WINDOW`] secondes car le
//! signal est en pics. La mesure vient de `ioreg` : `accumulatedGPUTime` de
//! chaque client `AGXDeviceUserClient`, par pid, notre propre pid exclu.
//! Écarté à la mesure : `Device Utilization %` d'IOAccelerator moins le temps
//! `GPUStartTime..GPUEndTime` de nos command buffers. Ce temps inclut les
//! tranches où le GPU sert les autres, la soustraction devient fausse sous
//! contention et la part oscillait entre 14 et 80 % externes pour une charge
//! constante. Sans mesure (pas d'ioreg), la part de départ
//! `serve.auto_yield_share` reste en place.

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
use std::time::Duration;

/// Réglage du cédage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct AutoYield {
    /// Part de GPU pendant le cédage (10 à 100).
    pub(super) share: u8,
    /// Inactivité avant de rendre la part configurée.
    pub(super) idle: Duration,
    /// Part ajustée à l'activité GPU des autres processus.
    pub(super) dynamic: bool,
}

/// En deçà de ce délai depuis la dernière entrée, l'utilisateur est actif.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
const ACTIVE: Duration = Duration::from_secs(5);
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
const POLL: Duration = Duration::from_secs(1);
/// Bornes et marge de la part dynamique, en points de pourcentage.
const DYNAMIC_MIN: u8 = 10;
const DYNAMIC_MAX: u8 = 50;
const MARGIN: f64 = 10.0;
/// Écart minimal avant de changer la part (hystérésis), en points.
const HYSTERESIS: u8 = 5;
/// Échantillons moyennés, un par [`POLL`].
const WINDOW: usize = 5;

struct State {
    /// Part du lancement ou du dernier SIGHUP.
    configured: u8,
    /// Part basse, 100 tant que le cédage est éteint.
    low: u8,
    yielding: bool,
}

static STATE: Mutex<State> = Mutex::new(State {
    configured: 100,
    low: 100,
    yielding: false,
});

fn lock() -> std::sync::MutexGuard<'static, State> {
    STATE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Publie la part configurée (lancement, SIGHUP) ; en cédage, la part basse reste.
pub(super) fn set_configured(share: u8) {
    let mut state = lock();
    state.configured = share;
    if state.yielding {
        saragossa::set_gpu_duty_share(share.min(state.low));
    } else {
        saragossa::set_gpu_share(share);
    }
}

/// Cède dès une entrée récente, rend après `idle` sans entrée, sinon garde l'état.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn next_state(config: AutoYield, yielding: bool, since_input: Duration) -> bool {
    if since_input < ACTIVE {
        true
    } else if since_input >= config.idle {
        false
    } else {
        yielding
    }
}

/// Un tour du thread, horloge injectée, `low` part basse du moment :
/// `Some(état)` si l'état a basculé.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn step(config: AutoYield, low: u8, since_input: Duration) -> Option<bool> {
    let mut state = lock();
    let next = next_state(config, state.yielding, since_input);
    let flipped = next != state.yielding;
    if flipped || (next && low != state.low) {
        saragossa::set_gpu_duty_share(if next {
            state.configured.min(low)
        } else {
            state.configured
        });
    }
    state.low = low;
    state.yielding = next;
    flipped.then_some(next)
}

/// Loi de contrôle, `external` utilisation GPU moyenne des autres processus
/// (%) : leur reste moins la marge, borné à 10..=50. La part `current` ne
/// change que d'au moins [`HYSTERESIS`] points, sauf pour atteindre une borne.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn dynamic_share(external: f64, current: u8) -> u8 {
    let target = (100.0 - external - MARGIN)
        .round()
        .clamp(f64::from(DYNAMIC_MIN), f64::from(DYNAMIC_MAX)) as u8;
    if target.abs_diff(current) >= HYSTERESIS || matches!(target, DYNAMIC_MIN | DYNAMIC_MAX) {
        target
    } else {
        current
    }
}

/// Part dynamique : moyenne glissante de l'utilisation externe.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
struct Controller {
    window: VecDeque<f64>,
    share: u8,
    /// Part de départ, reprise dès qu'une mesure manque.
    fallback: u8,
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
impl Controller {
    fn new(share: u8) -> Self {
        Self {
            window: VecDeque::with_capacity(WINDOW),
            share,
            fallback: share,
        }
    }

    /// Ajoute un échantillon ; la part ne bouge qu'une fois la fenêtre pleine.
    /// Sans mesure (`None`), la fenêtre se vide et la part de départ revient :
    /// une part adaptée n'est jamais gardée sur une mesure périmée.
    fn push(&mut self, external: Option<f64>) -> u8 {
        let Some(external) = external else {
            self.window.clear();
            self.share = self.fallback;
            return self.share;
        };
        if self.window.len() == WINDOW {
            self.window.pop_front();
        }
        self.window.push_back(external);
        if self.window.len() == WINDOW {
            let mean = self.window.iter().sum::<f64>() / WINDOW as f64;
            self.share = dynamic_share(mean, self.share);
        }
        self.share
    }
}

/// Un tour du thread : `measure` (relevé `ioreg`) n'est appelé qu'en cédage
/// dynamique, puis l'état bascule ou non. Renvoie `Some(état)` sur bascule ;
/// la fenêtre repart alors de la part de départ.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn tick(
    config: AutoYield,
    controller: &mut Controller,
    since_input: Duration,
    measure: impl FnOnce() -> Option<f64>,
) -> Option<bool> {
    if config.dynamic && lock().yielding {
        let external = measure();
        let before = controller.share;
        let share = controller.push(external);
        if share != before {
            match external {
                Some(external) => eprintln!(
                    "saragossa serve auto-yield: part {share} % (GPU externe {external:.0} %)"
                ),
                None => eprintln!(
                    "saragossa serve auto-yield: part {share} % (mesure GPU indisponible)"
                ),
            }
        }
    }
    let flipped = step(config, controller.share, since_input);
    if let Some(yielding) = flipped {
        *controller = Controller::new(config.share);
        eprintln!(
            "saragossa serve auto-yield: {}",
            if yielding {
                "cède le GPU"
            } else {
                "part configurée rendue"
            }
        );
    }
    flipped
}

/// Lance le thread de cédage (macOS seulement).
pub(super) fn spawn(config: AutoYield) {
    #[cfg(target_os = "macos")]
    {
        eprintln!(
            "saragossa serve auto-yield: part {} % {}dès une entrée utilisateur, retour après {} s d'inactivité",
            config.share,
            if config.dynamic {
                "puis 10 à 50 % selon le GPU des autres processus "
            } else {
                ""
            },
            config.idle.as_secs()
        );
        let spawned = std::thread::Builder::new()
            .name("auto-yield".into())
            .spawn(move || {
                let mut controller = Controller::new(config.share);
                let mut sampler: Option<Sampler> = None;
                loop {
                    let flipped = tick(config, &mut controller, since_input(), || {
                        sampler.get_or_insert_with(Sampler::new).external()
                    });
                    if flipped.is_some() {
                        sampler = None;
                    }
                    std::thread::sleep(POLL);
                }
            });
        if let Err(error) = spawned {
            eprintln!("saragossa serve auto-yield: thread non lancé: {error}");
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = config;
        eprintln!("saragossa serve auto-yield: ignoré hors macOS");
    }
}

/// Temps écoulé depuis la dernière entrée clavier ou souris matérielle.
#[cfg(target_os = "macos")]
#[allow(unsafe_code)]
fn since_input() -> Duration {
    #[link(name = "CoreGraphics", kind = "framework")]
    unsafe extern "C" {
        fn CGEventSourceSecondsSinceLastEventType(state: i32, event_type: u32) -> f64;
    }
    // kCGEventSourceStateHIDSystemState = 1, kCGAnyInputEventType = !0.
    // SAFETY: fonction C sans pointeur ni état partagé côté appelant.
    let secs = unsafe { CGEventSourceSecondsSinceLastEventType(1, u32::MAX) };
    // NOTE: sans session graphique la valeur est absurde : traitée comme inactif.
    Duration::try_from_secs_f64(secs).unwrap_or(Duration::MAX)
}

/// Temps GPU cumulé (ns) par client `AGXDeviceUserClient` : (pid, id de registre).
type ClientTimes = HashMap<(u32, u64), u64>;

/// Délai maximal d'un relevé `ioreg` (une vingtaine de ms d'ordinaire).
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
const IOREG_TIMEOUT: Duration = Duration::from_millis(500);

/// Intervalle minimal entre deux relevés comparés : en deçà, la division par
/// un temps quasi nul amplifierait le bruit jusqu'à 100 %.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
const MIN_SAMPLE_INTERVAL: Duration = Duration::from_millis(500);

/// Utilisation GPU des autres processus entre deux relevés `ioreg`.
#[cfg(target_os = "macos")]
struct Sampler {
    last: Option<(ClientTimes, std::time::Instant)>,
}

#[cfg(target_os = "macos")]
impl Sampler {
    fn new() -> Self {
        Self { last: None }
    }

    /// `None` si `ioreg` échoue, sans relevé de référence ou si la référence
    /// date de moins de [`MIN_SAMPLE_INTERVAL`] (elle est alors gardée). Les
    /// deux relevés sont datés au même point, après `ioreg`.
    fn external(&mut self) -> Option<f64> {
        let Some(current) = gpu_time_snapshot() else {
            self.last = None;
            return None;
        };
        let now = std::time::Instant::now();
        let Some((previous, at)) = &self.last else {
            self.last = Some((current, now));
            return None;
        };
        let external = external_percent(
            previous,
            &current,
            std::process::id(),
            now.duration_since(*at),
        );
        if external.is_some() {
            self.last = Some((current, now));
        }
        external
    }
}

#[cfg(target_os = "macos")]
fn gpu_time_snapshot() -> Option<ClientTimes> {
    let mut command = std::process::Command::new("/usr/sbin/ioreg");
    command.args(["-r", "-c", "AGXDeviceUserClient", "-d", "1", "-w0"]);
    let stdout = output_within(command, IOREG_TIMEOUT)?;
    gpu_time_by_client(&String::from_utf8_lossy(&stdout))
}

/// Sortie standard d'une commande réussie dans le délai, sinon `None`.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn output_within(mut command: std::process::Command, timeout: Duration) -> Option<Vec<u8>> {
    use std::process::Stdio;
    let child = command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    collect_within(child, timeout)
}

/// Attend `child` au plus `timeout` ; au-delà, ou si le thread lecteur ne
/// démarre pas, l'enfant est tué puis récolté. Le thread lit la sortie pour
/// qu'un tube plein ne bloque pas l'enfant.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn collect_within(mut child: std::process::Child, timeout: Duration) -> Option<Vec<u8>> {
    use std::io::Read;
    let abort = |child: &mut std::process::Child| {
        let _ = child.kill();
        let _ = child.wait();
        None
    };
    let Some(mut stdout) = child.stdout.take() else {
        return abort(&mut child);
    };
    let reader = std::thread::Builder::new()
        .name("auto-yield-ioreg".into())
        .spawn(move || {
            let mut buffer = Vec::new();
            stdout.read_to_end(&mut buffer).map(|_| buffer)
        });
    let Ok(reader) = reader else {
        return abort(&mut child);
    };
    let deadline = std::time::Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let output = reader.join().ok()?.ok()?;
                return status.success().then_some(output);
            }
            Ok(None) if std::time::Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(5));
            }
            // NOTE: le thread lecteur se termine à la fermeture du tube.
            _ => return abort(&mut child),
        }
    }
}

/// Temps GPU cumulé (`accumulatedGPUTime`, ns) de chaque client : un bloc
/// `+-o AGXDeviceUserClient <class …, id 0x…>` porte son `IOUserClientCreator`.
/// `None` si aucun client n'est reconnu : mesure indisponible, pas GPU inactif
/// (WindowServer a toujours un client).
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn gpu_time_by_client(ioreg: &str) -> Option<ClientTimes> {
    const PID: &str = "\"IOUserClientCreator\" = \"pid ";
    const TIME: &str = "\"accumulatedGPUTime\"=";
    const ID: &str = ", id 0x";
    let digits = |text: &str, radix: u32| -> Option<u64> {
        let end = text
            .find(|c: char| !c.is_digit(radix))
            .unwrap_or(text.len());
        u64::from_str_radix(&text[..end], radix).ok()
    };
    let mut times = HashMap::new();
    for client in ioreg.split("+-o ").skip(1) {
        let header = client.lines().next().unwrap_or_default();
        let id = header
            .find(ID)
            .and_then(|at| digits(&header[at + ID.len()..], 16));
        let pid = client
            .find(PID)
            .and_then(|at| digits(&client[at + PID.len()..], 10))
            .and_then(|pid| u32::try_from(pid).ok());
        let (Some(id), Some(pid)) = (id, pid) else {
            continue;
        };
        let counters: Vec<u64> = client
            .match_indices(TIME)
            .filter_map(|(at, _)| digits(&client[at + TIME.len()..], 10))
            .collect();
        if !counters.is_empty() {
            times.insert((pid, id), counters.iter().sum());
        }
    }
    (!times.is_empty()).then_some(times)
}

/// Part du temps écoulé (`elapsed`) passée sur le GPU par les processus
/// autres que `own_pid`, bornée à 0..=100 ; `None` sous
/// [`MIN_SAMPLE_INTERVAL`]. Les deltas se prennent client par
/// client présent aux deux relevés : un client disparu ou apparu entre-temps
/// (cumul au passé inconnu) ne fausse pas ceux qui restent.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn external_percent(
    previous: &ClientTimes,
    current: &ClientTimes,
    own_pid: u32,
    elapsed: Duration,
) -> Option<f64> {
    if elapsed < MIN_SAMPLE_INTERVAL {
        return None;
    }
    let busy: u64 = current
        .iter()
        .filter(|((pid, _), _)| *pid != own_pid)
        .filter_map(|(client, now)| {
            previous
                .get(client)
                .map(|before| now.saturating_sub(*before))
        })
        .sum();
    Some((busy as f64 * 100.0 / elapsed.as_nanos() as f64).clamp(0.0, 100.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hysteresis_yields_on_input_and_restores_after_idle() {
        let _serial = super::super::reload::TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        struct Reset;
        impl Drop for Reset {
            fn drop(&mut self) {
                *lock() = State {
                    configured: 100,
                    low: 100,
                    yielding: false,
                };
                saragossa::set_gpu_duty_share(100);
            }
        }
        let _reset = Reset;
        let config = AutoYield {
            share: 25,
            idle: Duration::from_secs(60),
            dynamic: false,
        };
        let secs = Duration::from_secs;
        set_configured(80);
        assert_eq!(saragossa::gpu_share(), 80);
        // Inactif au départ : rien ne bouge, même dans la bande.
        assert_eq!(step(config, 25, secs(30)), None);
        assert_eq!(step(config, 25, secs(1)), Some(true));
        assert_eq!(saragossa::gpu_share(), 25);
        // Bande d'hystérésis : ni l'entrée qui vieillit ni 59 s ne rendent.
        for since in [2, 10, 59] {
            assert_eq!(step(config, 25, secs(since)), None, "{since} s");
            assert_eq!(saragossa::gpu_share(), 25);
        }
        // SIGHUP en cédage : nouvelle part retenue, la part basse reste.
        set_configured(90);
        assert_eq!(saragossa::gpu_share(), 25);
        set_configured(20);
        assert_eq!(saragossa::gpu_share(), 20, "jamais au-dessus du configuré");
        set_configured(90);
        assert_eq!(step(config, 25, secs(60)), Some(false));
        assert_eq!(saragossa::gpu_share(), 90);
        assert_eq!(step(config, 25, secs(4)), Some(true), "nouvelle entrée");
        assert_eq!(step(config, 25, Duration::MAX), Some(false));
        // Part dynamique : appliquée en cédage, ignorée sinon.
        assert_eq!(step(config, 25, secs(1)), Some(true));
        assert_eq!(step(config, 40, secs(1)), None);
        assert_eq!(saragossa::gpu_share(), 40);
        assert_eq!(step(config, 15, Duration::MAX), Some(false));
        assert_eq!(saragossa::gpu_share(), 90);
        assert_eq!(step(config, 45, Duration::MAX), None);
        assert_eq!(saragossa::gpu_share(), 90);
    }

    #[test]
    fn dynamic_share_leaves_others_their_gpu_minus_margin() {
        // Bornes : GPU libre -> 50, GPU saturé -> 10.
        assert_eq!(dynamic_share(0.0, 25), 50);
        assert_eq!(dynamic_share(100.0, 25), 10);
        assert_eq!(dynamic_share(85.0, 25), 10);
        // Plage linéaire : 100 - externe - 10.
        assert_eq!(dynamic_share(60.0, 25), 30);
        assert_eq!(dynamic_share(70.0, 50), 20);
        // Hystérésis : moins de 5 points d'écart, la part reste.
        assert_eq!(dynamic_share(63.0, 30), 30);
        assert_eq!(dynamic_share(56.0, 30), 30);
        assert_eq!(dynamic_share(55.0, 30), 35);
        // Une borne s'atteint même à moins de 5 points.
        assert_eq!(dynamic_share(38.0, 48), 50);
        assert_eq!(dynamic_share(81.0, 12), 10);

        // Moyenne glissante : rien avant la fenêtre pleine, un pic est dilué.
        let mut controller = Controller::new(25);
        for _ in 0..WINDOW - 1 {
            assert_eq!(controller.push(Some(0.0)), 25, "fenêtre incomplète");
        }
        assert_eq!(controller.push(Some(100.0)), 50, "moyenne 20 %, pas 100 %");
        for _ in 0..WINDOW {
            controller.push(Some(80.0));
        }
        assert_eq!(controller.share, 10);
    }

    #[test]
    fn missing_measure_falls_back_to_the_starting_share() {
        let mut controller = Controller::new(25);
        for _ in 0..WINDOW {
            controller.push(Some(0.0));
        }
        assert_eq!(controller.share, 50);
        assert_eq!(controller.push(None), 25, "repli immédiat");
        // La fenêtre repart de zéro : une seule mesure ne suffit pas à rebouger.
        assert_eq!(controller.push(Some(0.0)), 25);
    }

    #[test]
    fn external_gpu_time_excludes_the_engine_and_new_processes() {
        // Ordre des clés d'ioreg arbitraire : AppUsage avant ou après le pid.
        let ioreg = r#"+-o AGXDeviceUserClient  <class AGXDeviceUserClient, id 0x1a, !registered>
    {
      "AppUsage" = ({"API"="Metal","accumulatedGPUTime"=100},{"API"="Metal","accumulatedGPUTime"=20})
      "IOUserClientCreator" = "pid 7, game"
    }
+-o AGXDeviceUserClient  <class AGXDeviceUserClient, id 0x1b, !registered>
    {
      "IOUserClientCreator" = "pid 7, game"
      "AppUsage" = ({"API"="Metal","lastSubmittedTime"=5,"accumulatedGPUTime"=30})
    }
+-o AGXDeviceUserClient  <class AGXDeviceUserClient, id 0x2c, !registered>
    {
      "IOUserClientCreator" = "pid 9, saragossa"
      "AppUsage" = ({"API"="Metal","accumulatedGPUTime"=500})
    }
+-o AGXDeviceUserClient  <class AGXDeviceUserClient, id 0x3d, !registered>
    {
      "AppUsage" = ()
    }"#;
        let before = gpu_time_by_client(ioreg).expect("clients");
        assert_eq!(
            before,
            HashMap::from([((7, 0x1a), 120), ((7, 0x1b), 30), ((9, 0x2c), 500)])
        );
        let mut after = before.clone();
        after.insert((7, 0x1a), 120 + 600_000_000);
        after.insert((9, 0x2c), 500 + 900_000_000);
        after.insert((11, 0x4e), 5_000_000_000);
        // 0,6 s de GPU pour le jeu en 1 s ; le moteur (pid 9) et le nouveau
        // client ne comptent pas.
        let second = Duration::from_secs(1);
        let pct = external_percent(&before, &after, 9, second).expect("intervalle");
        assert!((pct - 60.0).abs() < 1e-9, "{pct}");
        after.insert((7, 0x1a), 120 + 3_000_000_000);
        assert_eq!(
            external_percent(&before, &after, 9, second),
            Some(100.0),
            "borné"
        );
        assert_eq!(
            external_percent(&after, &before, 9, second),
            Some(0.0),
            "cumul remis à zéro"
        );
        // Premier intervalle quasi nul : pas de mesure plutôt que 100 %.
        assert_eq!(
            external_percent(&before, &after, 9, Duration::from_millis(1)),
            None
        );
        assert!(external_percent(&before, &after, 9, MIN_SAMPLE_INTERVAL).is_some());
    }

    #[test]
    fn departed_client_does_not_hide_the_remaining_one() {
        // A = 10 s et B = 1 s pour le même pid, puis A part et B passe à 1,8 s :
        // 0,8 s en 1 s, et non 1,8 - 11 saturé à 0.
        let s = 1_000_000_000_u64;
        let before = HashMap::from([((7, 1), 10 * s), ((7, 2), s)]);
        let after = HashMap::from([((7, 2), s * 18 / 10)]);
        let pct = external_percent(&before, &after, 9, Duration::from_secs(1)).expect("intervalle");
        assert!((pct - 80.0).abs() < 1e-9, "{pct}");
    }

    #[test]
    fn unrecognised_ioreg_output_is_unavailable_not_idle() {
        assert_eq!(gpu_time_by_client(""), None);
        assert_eq!(gpu_time_by_client("+-o IOService  <class IOService>"), None);
        let no_counter = "+-o AGXDeviceUserClient  <class AGXDeviceUserClient, id 0x1a>\n  \"IOUserClientCreator\" = \"pid 7, game\"\n  \"AppUsage\" = ()";
        assert_eq!(gpu_time_by_client(no_counter), None);
    }

    // Utilise /bin/sleep, /bin/ps et /bin/echo : absents sous Windows.
    #[cfg(unix)]
    #[test]
    fn slow_collector_is_killed_and_reaped_at_the_deadline() {
        use std::process::{Command, Stdio};
        let child = Command::new("/bin/sleep")
            .arg("5")
            .stdout(Stdio::piped())
            .spawn()
            .expect("sleep");
        let pid = child.id().to_string();
        let started = std::time::Instant::now();
        assert_eq!(collect_within(child, Duration::from_millis(100)), None);
        let elapsed = started.elapsed();
        assert!(
            (Duration::from_millis(100)..Duration::from_millis(400)).contains(&elapsed),
            "{elapsed:?}"
        );
        // Récolté : ni vivant ni zombie, `ps` ne le trouve plus.
        let ps = Command::new("/bin/ps")
            .args(["-o", "stat=", "-p", &pid])
            .output()
            .expect("ps");
        assert!(
            !ps.status.success(),
            "{}",
            String::from_utf8_lossy(&ps.stdout)
        );
        assert!(ps.stdout.is_empty());

        let mut echo = Command::new("/bin/echo");
        echo.arg("ok");
        assert_eq!(
            output_within(echo, Duration::from_secs(5)).as_deref(),
            Some(&b"ok\n"[..])
        );
        assert_eq!(
            output_within(Command::new("/usr/bin/false"), Duration::from_secs(5)),
            None
        );
    }

    #[test]
    fn fixed_mode_never_measures() {
        let _serial = super::super::reload::TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        struct Reset;
        impl Drop for Reset {
            fn drop(&mut self) {
                *lock() = State {
                    configured: 100,
                    low: 100,
                    yielding: false,
                };
                saragossa::set_gpu_duty_share(100);
            }
        }
        let _reset = Reset;
        let fixed = AutoYield {
            share: 25,
            idle: Duration::from_secs(60),
            dynamic: false,
        };
        let calls = std::cell::Cell::new(0);
        let measure = || {
            calls.set(calls.get() + 1);
            Some(0.0)
        };
        let mut controller = Controller::new(25);
        let active = Duration::ZERO;
        assert_eq!(tick(fixed, &mut controller, active, measure), Some(true));
        for _ in 0..WINDOW + 2 {
            assert_eq!(tick(fixed, &mut controller, active, measure), None);
        }
        assert_eq!(calls.get(), 0, "ioreg lancé en mode fixe");
        assert_eq!(saragossa::gpu_share(), 25);
        // En dynamique, la même boucle mesure à chaque tour de cédage.
        let dynamic = AutoYield {
            dynamic: true,
            ..fixed
        };
        for _ in 0..WINDOW {
            tick(dynamic, &mut controller, active, measure);
        }
        assert_eq!(calls.get(), WINDOW);
        assert_eq!(saragossa::gpu_share(), 50, "GPU libre");
    }
}
