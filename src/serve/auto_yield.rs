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
//! 10..=50 (voir [`dynamic_share`]). L'utilisation vient de `ioreg`
//! (`Device Utilization %` d'IOAccelerator), moyennée sur [`WINDOW`] secondes
//! car le signal est en pics. La charge du moteur y est comprise : on retranche
//! le temps GPU de nos propres command buffers bridés, mesuré exactement par
//! `GPUStartTime`/`GPUEndTime`. Échantillonner pendant nos pauses ne suffirait
//! pas : elles durent quelques millisecondes alors que la fenêtre du pilote
//! est plus longue et y mêle encore notre travail. Sans mesure (pas d'ioreg),
//! la part de départ `serve.auto_yield_share` reste en place.

use std::collections::VecDeque;
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
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
impl Controller {
    fn new(share: u8) -> Self {
        Self {
            window: VecDeque::with_capacity(WINDOW),
            share,
        }
    }

    /// Ajoute un échantillon ; la part ne bouge qu'une fois la fenêtre pleine.
    fn push(&mut self, external: f64) -> u8 {
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
                let mut sampler = Sampler::new();
                loop {
                    let yielding = lock().yielding;
                    if config.dynamic && yielding {
                        if let Some(external) = sampler.external() {
                            let before = controller.share;
                            let share = controller.push(external);
                            if share != before {
                                eprintln!(
                                    "saragossa serve auto-yield: part {share} % (GPU externe {external:.0} %)"
                                );
                            }
                        }
                    }
                    if let Some(yielding) = step(config, controller.share, since_input()) {
                        // Nouvelle période : la fenêtre repart de la part de départ.
                        controller = Controller::new(config.share);
                        sampler = Sampler::new();
                        eprintln!(
                            "saragossa serve auto-yield: {}",
                            if yielding {
                                "cède le GPU"
                            } else {
                                "part configurée rendue"
                            }
                        );
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

/// Utilisation GPU externe : celle d'IOAccelerator moins nos command buffers.
#[cfg(target_os = "macos")]
struct Sampler {
    own_ns: u64,
    at: std::time::Instant,
}

#[cfg(target_os = "macos")]
impl Sampler {
    fn new() -> Self {
        Self {
            own_ns: saragossa::runtime_flags::throttled_gpu_ns(),
            at: std::time::Instant::now(),
        }
    }

    fn external(&mut self) -> Option<f64> {
        let output = std::process::Command::new("/usr/sbin/ioreg")
            .args(["-r", "-d", "1", "-c", "IOAccelerator", "-w0"])
            .stderr(std::process::Stdio::null())
            .output()
            .ok()?;
        let device = device_utilization(&String::from_utf8_lossy(&output.stdout))?;
        let own_ns = saragossa::runtime_flags::throttled_gpu_ns();
        let elapsed = self.at.elapsed().as_nanos().max(1) as f64;
        let own = own_ns.saturating_sub(self.own_ns) as f64 * 100.0 / elapsed;
        self.own_ns = own_ns;
        self.at = std::time::Instant::now();
        Some((device - own).max(0.0))
    }
}

/// `"Device Utilization %"=N` du premier IOAccelerator de la sortie `ioreg`.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn device_utilization(ioreg: &str) -> Option<f64> {
    let key = "\"Device Utilization %\"=";
    let rest = &ioreg[ioreg.find(key)? + key.len()..];
    let digits = rest.split(|c: char| !c.is_ascii_digit()).next()?;
    digits.parse::<u8>().ok().map(f64::from)
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
            assert_eq!(controller.push(0.0), 25, "fenêtre incomplète");
        }
        assert_eq!(controller.push(100.0), 50, "moyenne 20 %, pas 100 %");
        for _ in 0..WINDOW {
            controller.push(80.0);
        }
        assert_eq!(controller.share, 10);
    }

    #[test]
    fn device_utilization_reads_the_ioreg_statistic() {
        let ioreg = r#"| "PerformanceStatistics" = {"Tiler Utilization %"=15,"Renderer Utilization %"=22,"Device Utilization %"=25,"SplitSceneCount"=0}"#;
        assert_eq!(device_utilization(ioreg), Some(25.0));
        assert_eq!(device_utilization("\"Device Utilization %\"=x"), None);
        assert_eq!(device_utilization("rien"), None);
    }
}
