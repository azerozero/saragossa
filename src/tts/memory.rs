//! Sondes mémoire opt-in, séparées des mesures de latence.

use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

pub(crate) use crate::runtime_flags::tts_memory_profile_enabled as enabled;

/// Une capture par phase et par processus : `footprint` perturbe les timings.
/// Les champs sont des octets retenus, sauf mention explicite dans leur nom.
pub(crate) fn checkpoint(phase: &'static str, fields: &[(&str, usize)]) {
    if !enabled() {
        return;
    }
    static SEEN: OnceLock<Mutex<HashSet<&'static str>>> = OnceLock::new();
    let Ok(mut seen) = SEEN.get_or_init(Mutex::default).lock() else {
        return;
    };
    if !seen.insert(phase) {
        return;
    }
    drop(seen);
    let mut line = format!("tts_memory phase={phase} pid={}", std::process::id());
    #[cfg(gpu_metal)]
    if let Some(device) = metal::Device::system_default() {
        line.push_str(&format!(
            " metal_allocated={}",
            device.current_allocated_size()
        ));
    }
    for (name, bytes) in fields {
        line.push_str(&format!(" {name}={bytes}"));
    }
    eprintln!("{line}");
    if crate::runtime_flags::tts_memory_probe_enabled() {
        match std::process::Command::new("/usr/bin/footprint")
            .args(["-p", &std::process::id().to_string(), "-f", "bytes"])
            .output()
        {
            Ok(output) => {
                eprintln!("tts_memory probe={phase} status={}", output.status);
                eprint!("{}", String::from_utf8_lossy(&output.stdout));
                eprint!("{}", String::from_utf8_lossy(&output.stderr));
            }
            Err(error) => eprintln!("tts_memory probe={phase} error={error}"),
        }
    }
}
