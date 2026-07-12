//! Garde mémoire prédictive partagée par les chemins Saragossa.

use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use thiserror::Error;

type FootprintProbe = Arc<dyn Fn() -> Option<u64> + Send + Sync>;

/// Niveau de pression mémoire macOS observé par le watcher.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryPressureLevel {
    /// Pression normale.
    Normal,
    /// Pression élevée : purger les caches opportunistes.
    Warn,
    /// Pression critique : purger tous les caches enregistrés.
    Critical,
}

/// Niveau transmis aux callbacks de purge.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PurgePressure {
    /// Purge opportuniste.
    Warn,
    /// Purge maximale.
    Critical,
}

/// Résultat d'un callback de purge.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PurgeOutcome {
    /// Rien n'était purgeable.
    Empty,
    /// Une entrée a été purgée.
    Purged {
        /// Taille estimée libérée, si connue.
        bytes: u64,
    },
}

/// Résumé d'une purge effectuée.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PurgeReport {
    /// Nom stable du purgeable.
    pub name: String,
    /// Taille estimée libérée, si connue.
    pub bytes: u64,
}

/// Registre ordonné de callbacks de purge.
pub struct PurgeRegistry<'a, Ctx = ()> {
    next_id: u64,
    entries: Vec<PurgeableEntry<'a, Ctx>>,
}

struct PurgeableEntry<'a, Ctx> {
    id: u64,
    order: i32,
    name: String,
    callback: Box<dyn FnMut(&mut Ctx, PurgePressure) -> PurgeOutcome + Send + 'a>,
}

impl<'a, Ctx> Default for PurgeRegistry<'a, Ctx> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, Ctx> PurgeRegistry<'a, Ctx> {
    /// Construit un registre vide.
    #[must_use]
    pub fn new() -> Self {
        Self {
            next_id: 1,
            entries: Vec::new(),
        }
    }

    /// Enregistre ou remplace un purgeable nommé.
    pub fn register(
        &mut self,
        order: i32,
        name: impl Into<String>,
        callback: impl FnMut(&mut Ctx, PurgePressure) -> PurgeOutcome + Send + 'a,
    ) -> u64 {
        let name = name.into();
        if let Some(entry) = self.entries.iter_mut().find(|entry| entry.name == name) {
            entry.order = order;
            entry.callback = Box::new(callback);
            return entry.id;
        }
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        self.entries.push(PurgeableEntry {
            id,
            order,
            name,
            callback: Box::new(callback),
        });
        id
    }

    /// Supprime un purgeable par identifiant.
    pub fn unregister(&mut self, id: u64) -> bool {
        let Some(index) = self.entries.iter().position(|entry| entry.id == id) else {
            return false;
        };
        self.entries.remove(index);
        true
    }

    /// Déclenche le premier purgeable qui libère quelque chose.
    pub fn purge_one(&mut self, ctx: &mut Ctx, pressure: PurgePressure) -> Option<PurgeReport> {
        self.entries.sort_by_key(|entry| (entry.order, entry.id));
        for entry in &mut self.entries {
            let PurgeOutcome::Purged { bytes } = (entry.callback)(ctx, pressure) else {
                continue;
            };
            return Some(PurgeReport {
                name: entry.name.clone(),
                bytes,
            });
        }
        None
    }

    /// Déclenche chaque purgeable une fois, dans l'ordre.
    pub fn purge_all_once(&mut self, ctx: &mut Ctx, pressure: PurgePressure) -> Vec<PurgeReport> {
        self.entries.sort_by_key(|entry| (entry.order, entry.id));
        self.entries
            .iter_mut()
            .filter_map(|entry| match (entry.callback)(ctx, pressure) {
                PurgeOutcome::Empty => None,
                PurgeOutcome::Purged { bytes } => Some(PurgeReport {
                    name: entry.name.clone(),
                    bytes,
                }),
            })
            .collect()
    }
}

/// Registre de purgeables utilisable par un thread watcher.
pub type SharedPurgeRegistry = Arc<Mutex<PurgeRegistry<'static, ()>>>;

/// Plafonds candidats de la garde mémoire.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MemoryLimits {
    /// Plafond statique explicite ou repli host 90 %.
    pub static_cap: Option<u64>,
    /// RAM hôte totale.
    pub host_memory: Option<u64>,
    /// Marge à conserver hors du process.
    pub headroom_bytes: u64,
    /// Working-set Metal recommandé.
    pub metal_cap: Option<u64>,
}

impl MemoryLimits {
    /// Calcule le plafond effectif.
    #[must_use]
    pub fn effective_limit(&self) -> Option<u64> {
        [self.static_cap, self.dynamic_cap(), self.metal_cap]
            .into_iter()
            .flatten()
            .filter(|limit| *limit > 0)
            .min()
    }

    /// Calcule le plafond dynamique host moins marge.
    #[must_use]
    pub fn dynamic_cap(&self) -> Option<u64> {
        self.host_memory
            .map(|bytes| bytes.saturating_sub(self.headroom_bytes))
    }
}

/// Projection mémoire courante.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MemoryProjection {
    /// Empreinte process actuelle.
    pub current: u64,
    /// Réservations de chargement en cours.
    pub reserved: u64,
    /// Demande à projeter.
    pub requested: u64,
    /// Empreinte projetée.
    pub projected: u64,
    /// Plafond retenu.
    pub limit: u64,
    /// Budget disponible avant la demande.
    pub available: u64,
}

/// Erreur de refus mémoire.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
#[error(
    "mémoire insuffisante : demandé {requested} octets, plafond {limit} octets, libre {available} octets — fermez des applications ou utilisez un modèle plus petit"
)]
pub struct MemoryGuardError {
    /// Empreinte process actuelle.
    pub current: u64,
    /// Réservations de chargement en cours.
    pub reserved: u64,
    /// Demande refusée.
    pub requested: u64,
    /// Empreinte projetée.
    pub projected: u64,
    /// Plafond effectif.
    pub limit: u64,
    /// Budget disponible avant la demande.
    pub available: u64,
}

impl From<MemoryProjection> for MemoryGuardError {
    fn from(projection: MemoryProjection) -> Self {
        Self {
            current: projection.current,
            reserved: projection.reserved,
            requested: projection.requested,
            projected: projection.projected,
            limit: projection.limit,
            available: projection.available,
        }
    }
}

/// Garde mémoire prédictive.
#[derive(Clone)]
pub struct MemoryGuard {
    inner: Arc<MemoryGuardInner>,
}

struct MemoryGuardInner {
    enabled: bool,
    limits: MemoryLimits,
    reserved: AtomicU64,
    footprint: FootprintProbe,
}

impl fmt::Debug for MemoryGuard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MemoryGuard")
            .field("enabled", &self.inner.enabled)
            .field("limits", &self.inner.limits)
            .field("reserved", &self.inner.reserved.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

impl MemoryGuard {
    /// Construit la garde globale du process.
    #[must_use]
    pub fn process() -> Self {
        Self::from_runtime_flags(
            crate::runtime_flags::memory_guard_enabled(),
            crate::runtime_flags::memory_static_cap_bytes().or_else(host_memory_cap_bytes),
            crate::runtime_flags::memory_headroom_bytes(),
        )
    }

    /// Construit la garde utilisée par `saragossa serve`.
    #[must_use]
    pub fn serve() -> Self {
        Self::from_runtime_flags(
            crate::runtime_flags::serve_oom_guard_enabled(),
            crate::runtime_flags::serve_memory_static_cap_bytes().or_else(host_memory_cap_bytes),
            crate::runtime_flags::serve_memory_headroom_bytes(),
        )
    }

    /// Construit une garde depuis les plafonds réels du système.
    #[must_use]
    pub fn from_runtime_flags(enabled: bool, static_cap: Option<u64>, headroom_bytes: u64) -> Self {
        Self::from_limits(
            enabled,
            MemoryLimits {
                static_cap,
                host_memory: host_memory_bytes(),
                headroom_bytes,
                metal_cap: recommended_metal_working_set_bytes(),
            },
            Arc::new(|| process_footprint_bytes()),
        )
    }

    /// Construit une garde testable depuis des plafonds explicites.
    #[must_use]
    pub fn from_limits(enabled: bool, limits: MemoryLimits, footprint: FootprintProbe) -> Self {
        Self {
            inner: Arc::new(MemoryGuardInner {
                enabled,
                limits,
                reserved: AtomicU64::new(0),
                footprint,
            }),
        }
    }

    /// Renvoie une garde désactivée.
    #[must_use]
    pub fn disabled() -> Self {
        Self::from_limits(
            false,
            MemoryLimits {
                static_cap: None,
                host_memory: None,
                headroom_bytes: 0,
                metal_cap: None,
            },
            Arc::new(|| None),
        )
    }

    /// Calcule le plafond effectif courant.
    #[must_use]
    pub fn limit_bytes(&self) -> Option<u64> {
        self.inner.limits.effective_limit()
    }

    /// Renvoie la projection courante.
    pub fn projection(&self, requested: u64) -> Option<MemoryProjection> {
        let reserved = self.inner.reserved.load(Ordering::Relaxed);
        self.projection_with_reserved(requested, reserved)
    }

    fn projection_with_reserved(&self, requested: u64, reserved: u64) -> Option<MemoryProjection> {
        if !self.inner.enabled {
            return None;
        }
        let current = (self.inner.footprint)()?;
        let limit = self.limit_bytes()?;
        let before_request = current.saturating_add(reserved);
        let projected = before_request.saturating_add(requested);
        Some(MemoryProjection {
            current,
            reserved,
            requested,
            projected,
            limit,
            available: limit.saturating_sub(before_request),
        })
    }

    /// Renvoie `true` si la projection dépasse le plafond.
    pub fn projection_over_limit(&self, requested: u64) -> Option<MemoryProjection> {
        let projection = self.projection(requested)?;
        (projection.projected > projection.limit).then_some(projection)
    }

    /// Refuse la demande si elle dépasserait le plafond effectif.
    ///
    /// # Errors
    ///
    /// Renvoie [`MemoryGuardError`] si la demande dépasse le plafond.
    pub fn check_allocation(&self, requested: u64) -> Result<(), MemoryGuardError> {
        match self.projection_over_limit(requested) {
            Some(projection) => Err(projection.into()),
            None => Ok(()),
        }
    }

    /// Réserve temporairement une allocation projetée.
    ///
    /// # Errors
    ///
    /// Renvoie [`MemoryGuardError`] si la réservation dépasserait le plafond.
    pub fn reserve_allocation(
        &self,
        requested: u64,
    ) -> Result<MemoryReservation, MemoryGuardError> {
        if requested == 0 || !self.inner.enabled {
            return Ok(MemoryReservation::noop());
        }
        loop {
            // NOTE: la projection DOIT être validée avec la valeur exacte de
            // `reserved` que le compare_exchange va engager — sinon deux
            // réservations concurrentes (STT/TTS/LLM chargés en parallèle au
            // boot) valident chacune contre l'état d'avant l'autre et
            // sur-réservent le budget de manière transitoire.
            let current = self.inner.reserved.load(Ordering::Relaxed);
            if let Some(projection) = self.projection_with_reserved(requested, current) {
                if projection.projected > projection.limit {
                    return Err(projection.into());
                }
            }
            let next = current.saturating_add(requested);
            if self
                .inner
                .reserved
                .compare_exchange(current, next, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                return Ok(MemoryReservation {
                    guard: Some(self.clone()),
                    bytes: requested,
                });
            }
        }
    }

    fn release_reservation(&self, bytes: u64) {
        let mut current = self.inner.reserved.load(Ordering::Relaxed);
        loop {
            let next = current.saturating_sub(bytes);
            match self.inner.reserved.compare_exchange(
                current,
                next,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return,
                Err(actual) => current = actual,
            }
        }
    }
}

/// Réservation RAII d'un chargement projeté.
#[derive(Debug)]
pub struct MemoryReservation {
    guard: Option<MemoryGuard>,
    bytes: u64,
}

impl MemoryReservation {
    fn noop() -> Self {
        Self {
            guard: None,
            bytes: 0,
        }
    }
}

impl Drop for MemoryReservation {
    fn drop(&mut self) {
        if let Some(guard) = self.guard.as_ref() {
            guard.release_reservation(self.bytes);
        }
    }
}

/// Renvoie la garde mémoire globale du process.
pub fn process_memory_guard() -> &'static MemoryGuard {
    static GUARD: OnceLock<MemoryGuard> = OnceLock::new();
    GUARD.get_or_init(MemoryGuard::process)
}

/// Renvoie le registre global de purgeables du process.
pub fn process_purge_registry() -> SharedPurgeRegistry {
    static REGISTRY: OnceLock<SharedPurgeRegistry> = OnceLock::new();
    Arc::clone(REGISTRY.get_or_init(|| Arc::new(Mutex::new(PurgeRegistry::new()))))
}

/// Démarre le watcher global de pression mémoire si possible.
pub fn ensure_process_memory_pressure_watcher() {
    static WATCHER: OnceLock<Option<MemoryPressureWatcher>> = OnceLock::new();
    let _ = WATCHER.get_or_init(|| start_memory_pressure_watcher(process_purge_registry()));
}

/// Thread de surveillance de pression mémoire.
#[derive(Debug)]
pub struct MemoryPressureWatcher {
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl Drop for MemoryPressureWatcher {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Démarre un watcher macOS de pression mémoire.
#[must_use]
pub fn start_memory_pressure_watcher(
    registry: SharedPurgeRegistry,
) -> Option<MemoryPressureWatcher> {
    if !crate::runtime_flags::memory_guard_enabled() {
        return None;
    }
    match read_memory_pressure_level() {
        Ok(_) => {}
        Err(_) => {
            log_info(format_args!(
                "watcher pression mémoire désactivé: sysctl indisponible"
            ));
            return None;
        }
    }
    let stop = Arc::new(AtomicBool::new(false));
    let thread_stop = Arc::clone(&stop);
    let handle = thread::Builder::new()
        .name("saragossa-memory-pressure".to_string())
        .spawn(move || {
            while !thread_stop.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_secs(2));
                match read_memory_pressure_level() {
                    Ok(MemoryPressureLevel::Normal) => {}
                    Ok(MemoryPressureLevel::Warn) => {
                        if let Ok(mut registry) = registry.lock() {
                            let report = registry.purge_one(&mut (), PurgePressure::Warn);
                            match report {
                                Some(report) => log_info(format_args!(
                                    "pression WARN: purgeable={} bytes={}",
                                    report.name, report.bytes
                                )),
                                None => log_info(format_args!(
                                    "pression WARN: aucun purgeable disponible"
                                )),
                            }
                        }
                    }
                    Ok(MemoryPressureLevel::Critical) => {
                        if let Ok(mut registry) = registry.lock() {
                            let reports = registry.purge_all_once(&mut (), PurgePressure::Critical);
                            let bytes = reports.iter().map(|report| report.bytes).sum::<u64>();
                            log_warn(format_args!(
                                "pression CRITICAL: purgeables={} bytes={}",
                                reports.len(),
                                bytes
                            ));
                        }
                    }
                    Err(_) => {
                        log_info(format_args!(
                            "watcher pression mémoire arrêté: sysctl indisponible"
                        ));
                        break;
                    }
                }
            }
        })
        .ok()?;
    Some(MemoryPressureWatcher {
        stop,
        handle: Some(handle),
    })
}

/// Estime la taille des safetensors top-level d'un modèle.
#[must_use]
pub fn estimate_model_bytes(path: &Path) -> u64 {
    model_safetensor_bytes(path).unwrap_or(0)
}

/// Estime récursivement la taille des safetensors sous un chemin.
#[must_use]
pub fn estimate_safetensor_tree_bytes(path: &Path) -> u64 {
    if path.is_file() {
        return safetensor_file_bytes(path).unwrap_or(0);
    }
    let mut total = 0_u64;
    let Ok(entries) = fs::read_dir(path) else {
        return 0;
    };
    for entry in entries.flatten() {
        total = total.saturating_add(estimate_safetensor_tree_bytes(&entry.path()));
    }
    total
}

/// Estime la taille cumulée d'une liste de fichiers.
#[must_use]
pub fn estimate_paths_bytes(paths: &[PathBuf]) -> u64 {
    paths
        .iter()
        .filter_map(|path| fs::metadata(path).ok())
        .map(|metadata| metadata.len())
        .fold(0_u64, u64::saturating_add)
}

fn model_safetensor_bytes(path: &Path) -> Option<u64> {
    let entries = fs::read_dir(path).ok()?;
    let mut total = 0_u64;
    for entry in entries {
        let entry = entry.ok()?;
        total = total.saturating_add(safetensor_file_bytes(&entry.path()).unwrap_or(0));
    }
    Some(total)
}

fn safetensor_file_bytes(path: &Path) -> Option<u64> {
    let is_safetensor = path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension == "safetensors");
    if is_safetensor {
        fs::metadata(path).ok().map(|metadata| metadata.len())
    } else {
        None
    }
}

fn host_memory_cap_bytes() -> Option<u64> {
    host_memory_bytes().map(|bytes| bytes.saturating_mul(9) / 10)
}

fn host_memory_bytes() -> Option<u64> {
    command_stdout("sysctl", &["-n", "hw.memsize"]).and_then(|text| text.trim().parse().ok())
}

fn command_stdout(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).to_string())
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn recommended_metal_working_set_bytes() -> Option<u64> {
    metal::Device::system_default().map(|device| device.recommended_max_working_set_size())
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
fn recommended_metal_working_set_bytes() -> Option<u64> {
    None
}

#[cfg(target_os = "macos")]
#[allow(
    unsafe_code,
    reason = "FFI Mach task_info : lit phys_footprint du process pour la garde mémoire"
)]
fn process_footprint_bytes() -> Option<u64> {
    use mach_sys::kern_return::KERN_SUCCESS;
    use mach_sys::message::mach_msg_type_number_t;
    use mach_sys::task::task_info;
    use mach_sys::task_info::{task_info_t, task_vm_info, TASK_VM_INFO, TASK_VM_INFO_COUNT};
    use mach_sys::traps::mach_task_self;

    let mut info = task_vm_info::default();
    let mut count: mach_msg_type_number_t = TASK_VM_INFO_COUNT;
    // SAFETY: `info` pointe vers un buffer `task_vm_info` valide, `count` vaut
    // le nombre de mots attendu par Mach, et `mach_task_self` désigne le process
    // courant sans transférer de propriété.
    let result = unsafe {
        task_info(
            mach_task_self(),
            TASK_VM_INFO,
            &mut info as *mut task_vm_info as task_info_t,
            &mut count,
        )
    };
    (result == KERN_SUCCESS).then_some(info.phys_footprint)
}

#[cfg(not(target_os = "macos"))]
fn process_footprint_bytes() -> Option<u64> {
    None
}

#[cfg(target_os = "macos")]
#[allow(
    unsafe_code,
    reason = "FFI sysctlbyname : lit le niveau de pression mémoire macOS"
)]
fn read_memory_pressure_level() -> Result<MemoryPressureLevel, ()> {
    use std::os::raw::{c_char, c_int, c_void};

    unsafe extern "C" {
        fn sysctlbyname(
            name: *const c_char,
            oldp: *mut c_void,
            oldlenp: *mut usize,
            newp: *mut c_void,
            newlen: usize,
        ) -> c_int;
    }

    let mut level: c_int = 0;
    let mut len = std::mem::size_of::<c_int>();
    // SAFETY: le nom est une chaîne C statique NUL-terminée, `level` et `len`
    // pointent vers des buffers valides pour l'écriture du sysctl, et aucun
    // nouveau buffer n'est fourni (`newp` nul, `newlen` zéro).
    let result = unsafe {
        sysctlbyname(
            c"kern.memorystatus_vm_pressure_level".as_ptr(),
            &mut level as *mut c_int as *mut c_void,
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if result != 0 || len != std::mem::size_of::<c_int>() {
        return Err(());
    }
    Ok(match level {
        value if value >= 4 => MemoryPressureLevel::Critical,
        2 | 3 => MemoryPressureLevel::Warn,
        _ => MemoryPressureLevel::Normal,
    })
}

#[cfg(not(target_os = "macos"))]
fn read_memory_pressure_level() -> Result<MemoryPressureLevel, ()> {
    Err(())
}

fn log_info(args: fmt::Arguments<'_>) {
    eprintln!("info: saragossa memory_guard: {args}");
}

fn log_warn(args: fmt::Arguments<'_>) {
    eprintln!("warn: saragossa memory_guard: {args}");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;

    fn guard(current: Arc<AtomicU64>, limits: MemoryLimits) -> MemoryGuard {
        MemoryGuard::from_limits(
            true,
            limits,
            Arc::new(move || Some(current.load(Ordering::Relaxed))),
        )
    }

    #[test]
    fn effective_limit_takes_min_of_three_caps() {
        let limits = MemoryLimits {
            static_cap: Some(900),
            host_memory: Some(1_200),
            headroom_bytes: 200,
            metal_cap: Some(700),
        };

        assert_eq!(limits.dynamic_cap(), Some(1_000));
        assert_eq!(limits.effective_limit(), Some(700));
    }

    #[test]
    fn check_allocation_accepts_boundary_and_rejects_overflow() {
        let current = Arc::new(AtomicU64::new(100));
        let guard = guard(
            Arc::clone(&current),
            MemoryLimits {
                static_cap: Some(150),
                host_memory: None,
                headroom_bytes: 0,
                metal_cap: None,
            },
        );

        assert!(guard.check_allocation(50).is_ok());
        let err = guard
            .check_allocation(51)
            .expect_err("invariant: dépassement refusé");
        assert_eq!(err.available, 50);
        assert_eq!(err.projected, 151);
    }

    #[test]
    fn reservations_are_counted_until_drop() {
        let current = Arc::new(AtomicU64::new(100));
        let guard = guard(
            current,
            MemoryLimits {
                static_cap: Some(180),
                host_memory: None,
                headroom_bytes: 0,
                metal_cap: None,
            },
        );

        let reservation = guard
            .reserve_allocation(60)
            .expect("invariant: réservation sous plafond");
        assert!(guard.check_allocation(20).is_ok());
        assert!(guard.check_allocation(21).is_err());
        drop(reservation);
        assert!(guard.check_allocation(80).is_ok());
    }

    #[test]
    fn purge_registry_runs_callbacks_in_order() {
        let mut registry = PurgeRegistry::<Vec<&'static str>>::new();
        registry.register(20, "late", |events, _| {
            events.push("late");
            PurgeOutcome::Purged { bytes: 2 }
        });
        registry.register(10, "early", |events, _| {
            events.push("early");
            PurgeOutcome::Purged { bytes: 1 }
        });
        let mut events = Vec::new();

        let report = registry
            .purge_one(&mut events, PurgePressure::Warn)
            .expect("invariant: purge effectuée");

        assert_eq!(report.name, "early");
        assert_eq!(events, vec!["early"]);
    }

    #[test]
    fn purge_registry_registration_is_idempotent_by_name() {
        let mut registry = PurgeRegistry::<u32>::new();
        let first = registry.register(10, "cache", |count, _| {
            *count += 1;
            PurgeOutcome::Purged { bytes: 1 }
        });
        let second = registry.register(5, "cache", |count, _| {
            *count += 10;
            PurgeOutcome::Purged { bytes: 10 }
        });
        let mut count = 0;

        let report = registry
            .purge_one(&mut count, PurgePressure::Critical)
            .expect("invariant: purge effectuée");

        assert_eq!(first, second);
        assert_eq!(count, 10);
        assert_eq!(report.bytes, 10);
        assert!(registry.unregister(first));
        assert!(!registry.unregister(first));
    }

    #[test]
    fn purge_registry_critical_runs_all_once() {
        let mut registry = PurgeRegistry::<Vec<&'static str>>::new();
        registry.register(20, "b", |events, _| {
            events.push("b");
            PurgeOutcome::Purged { bytes: 2 }
        });
        registry.register(10, "a", |events, _| {
            events.push("a");
            PurgeOutcome::Purged { bytes: 1 }
        });
        let mut events = Vec::new();

        let reports = registry.purge_all_once(&mut events, PurgePressure::Critical);

        assert_eq!(events, vec!["a", "b"]);
        assert_eq!(
            reports
                .iter()
                .map(|report| report.name.as_str())
                .collect::<Vec<_>>(),
            vec!["a", "b"]
        );
    }
}
