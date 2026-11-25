//! Cédage automatique du GPU (`--auto-yield`, `serve.auto_yield`, opt-in).
//!
//! Un thread lit chaque seconde le temps écoulé depuis la dernière entrée
//! clavier ou souris de la session (`CGEventSourceSecondsSinceLastEventType`,
//! même source que `HIDIdleTime`). Entrée de moins de [`ACTIVE`] : la part de
//! GPU tombe à `serve.auto_yield_share` ; après `serve.auto_yield_idle_secs`
//! sans entrée, la part configurée revient. Entre les deux seuils l'état ne
//! bouge pas (hystérésis). Seul le rapport cyclique change (sommeil après
//! chaque command buffer), jamais le calcul : les tokens sont identiques.
//! Le pool rayon n'est pas touché, il resterait figé à la part basse.

use std::sync::Mutex;
use std::time::Duration;

/// Réglage du cédage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct AutoYield {
    /// Part de GPU pendant le cédage (10 à 100).
    pub(super) share: u8,
    /// Inactivité avant de rendre la part configurée.
    pub(super) idle: Duration,
}

/// En deçà de ce délai depuis la dernière entrée, l'utilisateur est actif.
const ACTIVE: Duration = Duration::from_secs(5);
const POLL: Duration = Duration::from_secs(1);

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
fn next_state(config: AutoYield, yielding: bool, since_input: Duration) -> bool {
    if since_input < ACTIVE {
        true
    } else if since_input >= config.idle {
        false
    } else {
        yielding
    }
}

/// Un tour du thread, horloge injectée : `Some(état)` si l'état a basculé.
fn step(config: AutoYield, since_input: Duration) -> Option<bool> {
    let mut state = lock();
    state.low = config.share;
    let next = next_state(config, state.yielding, since_input);
    if next == state.yielding {
        return None;
    }
    state.yielding = next;
    saragossa::set_gpu_duty_share(if next {
        state.configured.min(config.share)
    } else {
        state.configured
    });
    Some(next)
}

/// Lance le thread de cédage (macOS seulement).
pub(super) fn spawn(config: AutoYield) {
    #[cfg(target_os = "macos")]
    {
        eprintln!(
            "saragossa serve auto-yield: part {} % dès une entrée utilisateur, retour après {} s d'inactivité",
            config.share,
            config.idle.as_secs()
        );
        let spawned = std::thread::Builder::new()
            .name("auto-yield".into())
            .spawn(move || loop {
                if let Some(yielding) = step(config, since_input()) {
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

/// Temps écoulé depuis la dernière entrée clavier ou souris de la session.
#[cfg(target_os = "macos")]
#[allow(unsafe_code)]
fn since_input() -> Duration {
    #[link(name = "CoreGraphics", kind = "framework")]
    unsafe extern "C" {
        fn CGEventSourceSecondsSinceLastEventType(state: i32, event_type: u32) -> f64;
    }
    // kCGEventSourceStateCombinedSessionState = 0, kCGAnyInputEventType = !0.
    // SAFETY: fonction C sans pointeur ni état partagé côté appelant.
    let secs = unsafe { CGEventSourceSecondsSinceLastEventType(0, u32::MAX) };
    // NOTE: sans session graphique la valeur est absurde : traitée comme inactif.
    Duration::try_from_secs_f64(secs).unwrap_or(Duration::MAX)
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
        };
        let secs = Duration::from_secs;
        set_configured(80);
        assert_eq!(saragossa::gpu_share(), 80);
        // Inactif au départ : rien ne bouge, même dans la bande.
        assert_eq!(step(config, secs(30)), None);
        assert_eq!(step(config, secs(1)), Some(true));
        assert_eq!(saragossa::gpu_share(), 25);
        // Bande d'hystérésis : ni l'entrée qui vieillit ni 59 s ne rendent.
        for since in [2, 10, 59] {
            assert_eq!(step(config, secs(since)), None, "{since} s");
            assert_eq!(saragossa::gpu_share(), 25);
        }
        // SIGHUP en cédage : nouvelle part retenue, la part basse reste.
        set_configured(90);
        assert_eq!(saragossa::gpu_share(), 25);
        set_configured(20);
        assert_eq!(saragossa::gpu_share(), 20, "jamais au-dessus du configuré");
        set_configured(90);
        assert_eq!(step(config, secs(60)), Some(false));
        assert_eq!(saragossa::gpu_share(), 90);
        assert_eq!(step(config, secs(4)), Some(true), "nouvelle entrée");
        assert_eq!(step(config, Duration::MAX), Some(false));
    }
}
