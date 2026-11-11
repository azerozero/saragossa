//! Publication du manifeste en dernier : un lecteur ne voit jamais un état partiel.
use crate::{InferError, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

pub(super) fn io<T>(path: &Path, result: std::io::Result<T>) -> Result<T> {
    result.map_err(|source| InferError::Io {
        path: path.to_owned(),
        source,
    })
}
pub(super) fn invalid(message: impl Into<String>) -> InferError {
    InferError::Config(format!("instantané post-prefill : {}", message.into()))
}
pub(super) fn digest(data: &[u8]) -> String {
    format!("{:x}", Sha256::digest(data))
}
pub(super) fn hash_file(path: &Path) -> Result<String> {
    let mut file = io(path, File::open(path))?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0; 1024 * 1024];
    loop {
        let count = io(path, file.read(&mut buffer))?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct Blob {
    pub(super) file: PathBuf,
    pub(super) dtype: String,
    pub(super) bytes: usize,
    sha256: String,
}
impl Blob {
    pub(super) fn validate(&self, dtype: &str, elements: usize) -> Result<()> {
        let width = match dtype {
            "f32" => 4,
            "bf16" => 2,
            _ => return Err(invalid("dtype inconnu")),
        };
        if self.dtype != dtype || elements.checked_mul(width) != Some(self.bytes) {
            return Err(invalid(
                "taille ou dtype de blob différent des dimensions attendues",
            ));
        }
        Ok(())
    }

    pub(super) fn read(&self, root: &Path) -> Result<Vec<u8>> {
        // Le manifeste reste un fichier externe : ne jamais suivre ses chemins arbitraires.
        if self.file.components().count() != 2
            || !self
                .file
                .components()
                .all(|c| matches!(c, Component::Normal(_)))
        {
            return Err(invalid("chemin de blob invalide"));
        }
        let path = root.join(&self.file);
        let canonical = io(&path, fs::canonicalize(&path))?;
        let canonical_root = io(root, fs::canonicalize(root))?;
        if !canonical.starts_with(canonical_root) {
            return Err(invalid("blob hors du répertoire"));
        }
        let mut file = io(&path, File::open(&path))?;
        if io(&path, file.metadata())?.len() != self.bytes as u64 {
            return Err(invalid("taille de blob incorrecte"));
        }
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(self.bytes)
            .map_err(|_| invalid("blob trop grand"))?;
        // La géométrie de tous les blobs est validée avant cet appel. Une
        // croissance concurrente du fichier ne peut pas agrandir cette lecture.
        bytes.resize(self.bytes, 0);
        io(&path, file.read_exact(&mut bytes))?;
        let mut extra = [0];
        if io(&path, file.read(&mut extra))? != 0 {
            return Err(invalid("blob agrandi pendant la lecture"));
        }
        if bytes.len() != self.bytes || digest(&bytes) != self.sha256 {
            return Err(invalid("checksum de blob incorrect"));
        }
        Ok(bytes)
    }
    pub(super) fn f32(&self, root: &Path) -> Result<Vec<f32>> {
        if self.dtype != "f32" {
            return Err(invalid("tenseur f32 attendu"));
        }
        f32_values(&self.read(root)?)
    }
}
pub(super) fn f32_values(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        return Err(invalid("taille f32 invalide"));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|v| f32::from_ne_bytes([v[0], v[1], v[2], v[3]]))
        .collect())
}
pub(super) fn bf16_values(bytes: &[u8]) -> Result<Vec<u16>> {
    if bytes.len() % 2 != 0 {
        return Err(invalid("taille bf16 invalide"));
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|v| u16::from_ne_bytes([v[0], v[1]]))
        .collect())
}

pub(super) struct Writer {
    root: PathBuf,
    generation: String,
    next: usize,
    published: bool,
}
impl Writer {
    pub(super) fn new(root: &Path) -> Result<Self> {
        io(root, fs::create_dir_all(root))?;
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|_| invalid("horloge invalide"))?
            .as_nanos();
        let generation = format!(
            "state-{}-{nanos}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        );
        let dir = root.join(&generation);
        io(&dir, fs::create_dir(&dir))?;
        Ok(Self {
            root: root.to_owned(),
            generation,
            next: 0,
            published: false,
        })
    }
    pub(super) fn blob(&mut self, dtype: &str, bytes: &[u8]) -> Result<Blob> {
        let relative = PathBuf::from(&self.generation).join(format!("{}.bin", self.next));
        self.next += 1;
        let path = self.root.join(&relative);
        let temp = path.with_extension("tmp");
        let mut file = io(
            &temp,
            OpenOptions::new().write(true).create_new(true).open(&temp),
        )?;
        io(&temp, file.write_all(bytes))?;
        io(&temp, file.sync_all())?;
        io(&path, fs::rename(&temp, &path))?;
        Ok(Blob {
            file: relative,
            dtype: dtype.into(),
            bytes: bytes.len(),
            sha256: digest(bytes),
        })
    }
    pub(super) fn f32(&mut self, values: &[f32]) -> Result<Blob> {
        self.blob("f32", bytemuck::cast_slice(values))
    }
    pub(super) fn publish(&mut self, manifest: &impl Serialize) -> Result<()> {
        let path = self.root.join(format!("{}.json.tmp", self.generation));
        let bytes = serde_json::to_vec_pretty(manifest).map_err(|source| InferError::Json {
            path: path.clone(),
            source,
        })?;
        let mut file = io(
            &path,
            OpenOptions::new().write(true).create_new(true).open(&path),
        )?;
        io(&path, file.write_all(&bytes))?;
        io(&path, file.sync_all())?;
        let dir = self.root.join(&self.generation);
        io(&dir, File::open(&dir).and_then(|f| f.sync_all()))?;
        io(&path, fs::rename(&path, self.root.join("manifest.json")))?;
        self.published = true;
        io(
            &self.root,
            File::open(&self.root).and_then(|f| f.sync_all()),
        )
    }
}
impl Drop for Writer {
    fn drop(&mut self) {
        if !self.published {
            let _ = fs::remove_dir_all(self.root.join(&self.generation));
            let _ = fs::remove_file(self.root.join(format!("{}.json.tmp", self.generation)));
        }
    }
}
