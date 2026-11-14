//! Second niveau disque : une écriture en vol, publication atomique et LRU prioritaire.
use super::*;
use fs2::FileExt;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc, Arc, Mutex, OnceLock,
};
use std::time::{Instant, SystemTime};

static STORE: OnceLock<Store> = OnceLock::new();
thread_local! {
    static SESSION: std::cell::RefCell<String> = const { std::cell::RefCell::new(String::new()) };
}

/// Isole les instantanés persistants d'une requête par hash de session.
pub struct PrefixDiskSession {
    previous: String,
    _thread: std::marker::PhantomData<std::rc::Rc<()>>,
}
impl PrefixDiskSession {
    /// Entre dans la session ; aucune valeur réseau n'est utilisée comme chemin.
    pub fn enter(session: Option<&str>) -> Self {
        let hash = storage::digest(session.unwrap_or_default().as_bytes());
        Self {
            previous: SESSION.replace(hash),
            _thread: std::marker::PhantomData,
        }
    }
}
impl Drop for PrefixDiskSession {
    fn drop(&mut self) {
        SESSION.set(std::mem::take(&mut self.previous));
    }
}

/// Active le cache persistant avant le chargement des modèles. Le répertoire
/// dédié doit être privé ; un verrou OS exclut les autres processus.
///
/// # Errors
/// Refuse un budget nul, un répertoire partagé ou une seconde initialisation.
pub fn configure_prefix_disk_cache(root: &Path, budget: u64, priority_tokens: usize) -> Result<()> {
    let store = Store::open(root, budget, priority_tokens)?;
    STORE
        .set(store)
        .map_err(|_| invalid("cache SSD déjà initialisé"))
}

/// Attend explicitement la publication en cours, pour arrêt contrôlé ou mesure.
/// La génération normale n'appelle jamais cette barrière.
///
/// # Errors
/// Renvoie une erreur si le délai est dépassé. Les erreurs d'écriture sont
/// journalisées ; seule une relecture confirme la présence d'une entrée.
pub fn flush_prefix_disk_cache(timeout: std::time::Duration) -> Result<()> {
    let started = Instant::now();
    if let Some(store) = STORE.get() {
        while store.busy.load(Ordering::Acquire) {
            if started.elapsed() >= timeout {
                return Err(invalid("attente de publication SSD expirée"));
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }
    Ok(())
}

pub(crate) fn enabled() -> bool {
    STORE.get().is_some()
}
pub(super) fn store() -> Option<&'static Store> {
    STORE.get()
}
pub(super) fn session() -> String {
    SESSION.with(|session| session.borrow().clone())
}

pub(super) struct Payload {
    pub(super) blobs: Vec<(PathBuf, Vec<u8>)>,
    pub(super) manifest: Vec<u8>,
}
impl Payload {
    fn seal(&mut self) -> Result<()> {
        let hashes: std::collections::HashMap<_, _> = self
            .blobs
            .iter()
            .map(|(path, bytes)| (path.to_string_lossy().into_owned(), storage::digest(bytes)))
            .collect();
        fn update(
            value: &mut serde_json::Value,
            hashes: &std::collections::HashMap<String, String>,
        ) {
            match value {
                serde_json::Value::Object(fields) => {
                    if let Some(hash) = fields
                        .get("file")
                        .and_then(|file| file.as_str())
                        .and_then(|file| hashes.get(file))
                    {
                        fields.insert("sha256".into(), hash.clone().into());
                    } else {
                        for child in fields.values_mut() {
                            update(child, hashes);
                        }
                    }
                }
                serde_json::Value::Array(values) => {
                    for child in values {
                        update(child, hashes);
                    }
                }
                _ => {}
            }
        }
        let mut manifest =
            serde_json::from_slice(&self.manifest).map_err(|error| invalid(error.to_string()))?;
        update(&mut manifest, &hashes);
        self.manifest =
            serde_json::to_vec(&manifest).map_err(|error| invalid(error.to_string()))?;
        Ok(())
    }
    fn bytes(&self) -> u64 {
        self.blobs
            .iter()
            .map(|(_, bytes)| bytes.len() as u64)
            .sum::<u64>()
            + self.manifest.len() as u64
            + 64
    }
}
struct Job {
    name: String,
    tokens: usize,
    priority: bool,
    payload: Payload,
    _permit: Permit,
    _memory: Option<crate::memory_guard::MemoryReservation>,
}
pub(super) struct Permit(Arc<AtomicBool>);
impl Drop for Permit {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

struct Entry {
    path: PathBuf,
    tokens: usize,
    bytes: u64,
    priority: bool,
    used: SystemTime,
}
pub(super) struct Store {
    pub(super) root: PathBuf,
    pub(super) budget: u64,
    priority_tokens: usize,
    staging_limit: u64,
    memory_guard: crate::memory_guard::MemoryGuard,
    busy: Arc<AtomicBool>,
    tx: Option<mpsc::SyncSender<Job>>,
    worker: Option<std::thread::JoinHandle<()>>,
    disk: Arc<Mutex<Disk>>,
}
struct Disk {
    entries: Vec<Entry>,
    write_failed: bool,
    _lock: File,
}
impl Store {
    fn open(root: &Path, budget: u64, priority_tokens: usize) -> Result<Self> {
        let staging_limit = crate::runtime_flags::serve_prefix_ssd_staging_bytes()?;
        if budget == 0 {
            return Err(invalid("budget SSD nul"));
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
            let mut builder = fs::DirBuilder::new();
            builder.recursive(true).mode(0o700);
            io(root, builder.create(root))?;
            if io(root, fs::symlink_metadata(root))?
                .file_type()
                .is_symlink()
                || io(root, fs::metadata(root))?.permissions().mode() & 0o077 != 0
            {
                return Err(invalid(
                    "répertoire SSD requis privé (chmod 700), sans lien symbolique",
                ));
            }
        }
        #[cfg(not(unix))]
        io(root, fs::create_dir_all(root))?;
        let root = io(root, fs::canonicalize(root))?;
        let lock_path = root.join("cache.lock");
        if fs::symlink_metadata(&lock_path).is_ok_and(|m| !m.is_file()) {
            return Err(invalid("verrou SSD non régulier"));
        }
        let lock = io(
            &lock_path,
            OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(&lock_path),
        )?;
        io(&lock_path, lock.try_lock_exclusive())?;
        let mut disk = Disk {
            entries: Vec::new(),
            write_failed: false,
            _lock: lock,
        };
        for item in io(&root, fs::read_dir(&root))? {
            let item = io(&root, item)?;
            let name = item.file_name().to_string_lossy().into_owned();
            if name == "pending" {
                remove(&item.path())?;
                continue;
            }
            let Some(hash) = name.strip_prefix("entry-") else {
                continue;
            };
            if hash.len() != 64 || !hash.bytes().all(|b| b.is_ascii_hexdigit()) {
                continue;
            }
            let path = item.path();
            if !io(&path, fs::symlink_metadata(&path))?.is_dir() {
                remove(&path)?;
                continue;
            }
            let manifest = path.join("manifest.json");
            let bytes = match tree_bytes(&path) {
                Ok(bytes) => bytes,
                Err(_) => {
                    remove(&path)?;
                    continue;
                }
            };
            if !manifest.is_file() || bytes > budget {
                remove(&path)?;
                continue;
            }
            let header = match verify_manifest(&path) {
                Ok(header)
                    if header.key.version == FORMAT_VERSION
                        && header.key.binary_sha256 == storage::binary_sha256()?
                        && header.key.little_endian == cfg!(target_endian = "little") =>
                {
                    header
                }
                _ => {
                    remove(&path)?;
                    continue;
                }
            };
            let used = io(&manifest, fs::metadata(&manifest))?
                .modified()
                .unwrap_or(SystemTime::UNIX_EPOCH);
            disk.entries.push(Entry {
                tokens: header.key.tokens,
                priority: path.join("priority").is_file(),
                path,
                bytes,
                used,
            });
        }
        disk.evict(budget, 0, false)?;
        let disk = Arc::new(Mutex::new(disk));
        let (tx, rx) = mpsc::sync_channel::<Job>(1);
        let worker_disk = Arc::clone(&disk);
        let worker_root = root.clone();
        let worker = io(
            &root,
            std::thread::Builder::new()
                .name("prefix-ssd".into())
                .spawn(move || {
                    while let Ok(mut job) = rx.recv() {
                        let start = Instant::now();
                        let result = job.payload.seal().and_then(|()| {
                            worker_disk
                                .lock()
                                .map_err(|_| invalid("verrou SSD empoisonné"))
                                .and_then(|mut disk| disk.write(&worker_root, budget, &job))
                        });
                        match result {
                            Ok(()) => eprintln!(
                                "prefix_ssd written bytes={} write_ms={:.3}",
                                job.payload.bytes(),
                                start.elapsed().as_secs_f64() * 1000.0
                            ),
                            Err(error) => eprintln!("prefix_ssd write_refused: {error}"),
                        }
                    }
                }),
        )?;
        Ok(Self {
            root,
            budget,
            priority_tokens,
            staging_limit,
            memory_guard: crate::memory_guard::MemoryGuard::serve(),
            busy: Arc::new(AtomicBool::new(false)),
            tx: Some(tx),
            worker: Some(worker),
            disk,
        })
    }
    /// Vérifie disque et RAM avant le premier blit ou clone de données.
    pub(super) fn admit_staging(
        &self,
        disk_bytes: u64,
        ram_bytes: u64,
    ) -> Result<crate::memory_guard::MemoryReservation> {
        check_staging_limits(
            disk_bytes,
            ram_bytes,
            self.budget,
            self.staging_limit,
            io(&self.root, fs2::available_space(&self.root))?,
        )?;
        self.memory_guard
            .reserve_allocation(ram_bytes)
            .map_err(|error| invalid(error.to_string()))
    }
    pub(super) fn boundaries(&self, maximum: usize, block: usize, minimum: usize) -> Vec<usize> {
        let Ok(disk) = self.disk.try_lock() else {
            return Vec::new();
        };
        let mut lengths: Vec<_> = disk
            .entries
            .iter()
            .map(|entry| entry.tokens)
            .filter(|&tokens| tokens > minimum && tokens <= maximum && tokens % block.max(1) == 0)
            .collect();
        lengths.sort_unstable_by(|a, b| b.cmp(a));
        lengths.dedup();
        lengths
    }
    pub(super) fn contains(&self, request: &Request) -> bool {
        self.disk
            .try_lock()
            .is_ok_and(|disk| disk.entries.iter().any(|entry| entry.path == request.root))
    }
    pub(super) fn reserve(&self) -> Option<Permit> {
        self.busy
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .ok()
            .map(|_| Permit(Arc::clone(&self.busy)))
    }
    #[cfg(test)]
    pub(super) fn submit(&self, request: &Request, payload: Payload, permit: Permit) -> Result<()> {
        self.submit_reserved(request, payload, permit, None)
    }
    pub(super) fn submit_reserved(
        &self,
        request: &Request,
        payload: Payload,
        permit: Permit,
        memory: Option<crate::memory_guard::MemoryReservation>,
    ) -> Result<()> {
        let name = request
            .root
            .file_name()
            .and_then(|s| s.to_str())
            .ok_or_else(|| invalid("clé SSD invalide"))?
            .to_owned();
        self.tx
            .as_ref()
            .ok_or_else(|| invalid("worker arrêté"))?
            .try_send(Job {
                name,
                tokens: request.key.tokens,
                priority: request.key.tokens <= self.priority_tokens,
                payload,
                _permit: permit,
                _memory: memory,
            })
            .map_err(|_| invalid("file SSD occupée"))
    }
    pub(super) fn read(&self, model: &CausalDecoder, request: &Request) -> Option<State> {
        // Une écriture ne fait jamais attendre le moteur ; ce passage repart à froid.
        let mut disk = self.disk.try_lock().ok()?;
        let index = disk
            .entries
            .iter()
            .position(|entry| entry.path == request.root)?;
        let start = Instant::now();
        let result = tree_bytes(&request.root).and_then(|bytes| {
            if bytes > self.budget {
                return Err(invalid("entrée SSD agrandie au-delà du budget"));
            }
            verify_manifest(&request.root).and_then(|_| model.snapshot_read(request))
        });
        match result {
            Ok(state) => {
                disk.entries[index].used = SystemTime::now();
                let manifest = request.root.join("manifest.json");
                if let Ok(file) = OpenOptions::new().write(true).open(&manifest) {
                    let _ = file.set_modified(disk.entries[index].used);
                }
                eprintln!(
                    "prefix_ssd hit tokens={} reload_ms={:.3}",
                    request.key.tokens,
                    start.elapsed().as_secs_f64() * 1000.0
                );
                Some(state)
            }
            Err(error) => {
                if let Err(cleanup) = remove(&disk.entries[index].path) {
                    // Les octets restent comptés si le disque refuse la suppression.
                    eprintln!("prefix_ssd cleanup_refused: {cleanup}");
                } else {
                    disk.entries.remove(index);
                }
                eprintln!("prefix_ssd rejected: {error}");
                None
            }
        }
    }
}
impl Drop for Store {
    fn drop(&mut self) {
        self.tx.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}
impl Drop for Disk {
    fn drop(&mut self) {
        // Libère aussi le verrou si un fork concurrent a brièvement hérité du fd.
        let _ = FileExt::unlock(&self._lock);
    }
}
impl Disk {
    fn evict(&mut self, budget: u64, incoming: u64, priority: bool) -> Result<()> {
        self.entries
            .sort_by_key(|entry| (entry.priority, entry.used));
        while self
            .entries
            .iter()
            .map(|e| e.bytes)
            .sum::<u64>()
            .saturating_add(incoming)
            > budget
        {
            if self
                .entries
                .first()
                .is_none_or(|entry| incoming > 0 && entry.priority && !priority)
            {
                return Err(invalid("budget réservé aux préfixes prioritaires"));
            }
            remove(&self.entries[0].path)?;
            self.entries.remove(0);
        }
        Ok(())
    }
    fn write(&mut self, root: &Path, budget: u64, job: &Job) -> Result<()> {
        if self.write_failed {
            return Err(invalid("écrivain SSD arrêté après erreur de publication"));
        }
        let path = root.join(&job.name);
        if self.entries.iter().any(|entry| entry.path == path) {
            return Ok(());
        }
        let bytes = job.payload.bytes();
        if bytes > budget {
            return Err(invalid("entrée plus grande que le budget SSD"));
        }
        self.evict(budget, bytes, job.priority)?;
        let pending = root.join("pending");
        if pending.exists() {
            remove(&pending)?;
        }
        io(&pending, fs::create_dir(&pending))?;
        let result = (|| {
            io(&pending, fs::create_dir(pending.join("state-buffered")))?;
            for (name, data) in &job.payload.blobs {
                write_file(&pending.join(name), data)?;
            }
            write_file(&pending.join("manifest.json"), &job.payload.manifest)?;
            write_file(
                &pending.join("manifest.sha256"),
                storage::digest(&job.payload.manifest).as_bytes(),
            )?;
            if job.priority {
                write_file(&pending.join("priority"), &[])?;
            }
            sync_dir(&pending.join("state-buffered"))?;
            sync_dir(&pending)?;
            io(&path, fs::rename(&pending, &path))?;
            sync_dir(root)?;
            Ok(())
        })();
        if result.is_err() {
            // Un fsync du parent peut échouer après rename. Ne pas oublier ces
            // octets puis dépasser le budget lors d'une publication suivante.
            self.write_failed = true;
            if pending.exists() {
                let _ = remove(&pending);
            }
            if path.exists() {
                let _ = remove(&path);
            }
            return result;
        }
        self.entries.push(Entry {
            path,
            tokens: job.tokens,
            bytes,
            priority: job.priority,
            used: SystemTime::now(),
        });
        Ok(())
    }
}
fn check_staging_limits(disk: u64, ram: u64, budget: u64, staging: u64, free: u64) -> Result<()> {
    if disk > budget {
        return Err(invalid("entrée plus grande que le budget SSD"));
    }
    if disk > free {
        return Err(invalid("espace libre SSD insuffisant"));
    }
    if ram > staging {
        return Err(invalid("plafond RAM de staging SSD dépassé"));
    }
    Ok(())
}

fn write_file(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = io(
        path,
        OpenOptions::new().write(true).create_new(true).open(path),
    )?;
    io(path, file.write_all(bytes))?;
    io(path, file.sync_all())
}
fn sync_dir(path: &Path) -> Result<()> {
    #[cfg(not(windows))]
    {
        io(path, File::open(path).and_then(|f| f.sync_all()))
    }
    #[cfg(windows)]
    {
        let _ = path;
        Ok(())
    }
}
fn remove(path: &Path) -> Result<()> {
    if io(path, fs::symlink_metadata(path))?.is_dir() {
        io(path, fs::remove_dir_all(path))
    } else {
        io(path, fs::remove_file(path))
    }
}
fn tree_bytes(path: &Path) -> Result<u64> {
    // Profondeur fixe : entrée / génération / blobs. Aucun lien n'est suivi.
    fn walk(path: &Path, depth: usize) -> Result<u64> {
        let metadata = io(path, fs::symlink_metadata(path))?;
        if metadata.is_file() {
            return Ok(metadata.len());
        }
        if !metadata.is_dir() || depth == 0 {
            return Err(invalid("arbre SSD invalide"));
        }
        io(path, fs::read_dir(path))?.try_fold(0u64, |total, item| {
            let item = io(path, item)?;
            total
                .checked_add(walk(&item.path(), depth - 1)?)
                .ok_or_else(|| invalid("budget SSD débordé"))
        })
    }
    walk(path, 2)
}
fn verify_manifest(root: &Path) -> Result<Manifest> {
    use std::io::Read;
    let read = |name: &str, cap: u64| -> Result<Vec<u8>> {
        let path = root.join(name);
        let meta = io(&path, fs::symlink_metadata(&path))?;
        if !meta.is_file() || meta.len() > cap {
            return Err(invalid("en-tête SSD invalide"));
        }
        let mut bytes = Vec::new();
        io(&path, File::open(&path))?
            .take(cap + 1)
            .read_to_end(&mut bytes)
            .map_err(|e| invalid(e.to_string()))?;
        if bytes.len() as u64 > cap {
            return Err(invalid("en-tête SSD agrandi"));
        }
        Ok(bytes)
    };
    let bytes = read("manifest.json", MAX_MANIFEST_BYTES)?;
    if storage::digest(&bytes).as_bytes() != read("manifest.sha256", 64)? {
        return Err(invalid("checksum du manifeste SSD incorrect"));
    }
    serde_json::from_slice(&bytes).map_err(|error| invalid(error.to_string()))
}

#[cfg(test)]
mod tests;
