//! Identité du contenu relu à chaque chargement, sans cache de hachage disque.
use super::storage::{invalid, io};
use crate::Result;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{self, File, Metadata};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::Instant;

const BLOCK_BYTES: usize = 4 * 1024 * 1024;
const MAX_WORKERS: usize = 8;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
struct Stamp {
    bytes: u64,
    modified_ns: u128,
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
    // Restaurer le mtime après une écriture change encore le ctime Unix.
    #[cfg(unix)]
    changed: (i64, i64),
}
impl Stamp {
    fn read(path: &Path, metadata: &Metadata) -> Result<Self> {
        #[cfg(unix)]
        use std::os::unix::fs::MetadataExt;
        if !metadata.is_file() {
            return Err(invalid("source non régulière"));
        }
        Ok(Self {
            bytes: metadata.len(),
            modified_ns: io(path, metadata.modified())?
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|_| invalid("mtime antérieur à UNIX_EPOCH"))?
                .as_nanos(),
            #[cfg(unix)]
            device: metadata.dev(),
            #[cfg(unix)]
            inode: metadata.ino(),
            #[cfg(unix)]
            changed: (metadata.ctime(), metadata.ctime_nsec()),
        })
    }

    fn verify(&self, path: &Path, metadata: &Metadata) -> Result<()> {
        if Self::read(path, metadata)? != *self {
            return Err(invalid(
                "source modifiée depuis le chargement ou pendant le hachage",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(super) struct Source {
    path: PathBuf,
    stamp: Stamp,
    // SHA-256 de la taille, du découpage et des SHA-256 des blocs dans l'ordre.
    content_sha256: String,
}
impl Source {
    pub(super) fn path(&self) -> &Path {
        &self.path
    }

    pub(super) fn verify(&self) -> Result<()> {
        self.stamp
            .verify(&self.path, &io(&self.path, fs::metadata(&self.path))?)
    }

    pub(super) fn read(path: &Path) -> Result<Self> {
        let started = Instant::now();
        let stamp = Stamp::read(path, &io(path, fs::metadata(path))?)?;
        let workers = std::thread::available_parallelism().map_or(1, usize::from);
        let content_sha256 = hash_blocks(path, &stamp, workers)?;
        let source = Self {
            path: path.to_owned(),
            stamp,
            content_sha256,
        };
        source.verify()?;
        eprintln!(
            "state_snapshot hash_ms={:.3} bytes={}",
            started.elapsed().as_secs_f64() * 1000.0,
            source.stamp.bytes
        );
        Ok(source)
    }
}

/// Un buffer de 4 Mio par thread (au plus huit), plus 32 octets par bloc.
/// Le nombre de threads et leur ordonnancement ne changent jamais l'empreinte.
fn hash_blocks(path: &Path, stamp: &Stamp, workers: usize) -> Result<String> {
    stamp.verify(path, &io(path, fs::metadata(path))?)?;
    let blocks = usize::try_from(stamp.bytes.div_ceil(BLOCK_BYTES as u64))
        .map_err(|_| invalid("source trop grande"))?;
    let mut hashes = Vec::<[u8; 32]>::new();
    hashes
        .try_reserve_exact(blocks)
        .map_err(|_| invalid("trop de blocs source"))?;
    hashes.resize(blocks, [0; 32]);
    let per_worker = blocks.div_ceil(workers.clamp(1, MAX_WORKERS)).max(1);
    std::thread::scope(|scope| -> Result<()> {
        let mut threads = Vec::new();
        let mut result = Ok(());
        for (group, output) in hashes.chunks_mut(per_worker).enumerate() {
            let spawned = io(
                path,
                std::thread::Builder::new().spawn_scoped(scope, move || -> Result<()> {
                    // Une ouverture par thread : try_clone partagerait le curseur.
                    let mut file = io(path, File::open(path))?;
                    stamp.verify(path, &io(path, file.metadata())?)?;
                    let mut offset = (group * per_worker) as u64 * BLOCK_BYTES as u64;
                    io(path, file.seek(SeekFrom::Start(offset)))?;
                    let mut buffer = vec![0; BLOCK_BYTES];
                    for hash in output {
                        let count = (stamp.bytes - offset).min(BLOCK_BYTES as u64) as usize;
                        io(path, file.read_exact(&mut buffer[..count]))?;
                        *hash = Sha256::digest(&buffer[..count]).into();
                        offset += count as u64;
                    }
                    stamp.verify(path, &io(path, file.metadata())?)
                }),
            );
            match spawned {
                Ok(thread) => threads.push(thread),
                Err(error) => {
                    result = Err(error);
                    break;
                }
            }
        }
        // Joindre tous les threads, même si un autre a échoué : aucun panic
        // de worker ne doit être propagé par la fermeture implicite de scope.
        for thread in threads {
            let joined = thread
                .join()
                .map_err(|_| invalid("thread de hachage interrompu"))
                .and_then(|result| result);
            if result.is_ok() {
                result = joined;
            }
        }
        result
    })?;
    stamp.verify(path, &io(path, fs::metadata(path))?)?;
    let mut hash = Sha256::new();
    hash.update(b"saragossa-state-source-blocks-v1\0");
    hash.update(stamp.bytes.to_le_bytes());
    hash.update((BLOCK_BYTES as u64).to_le_bytes());
    for block in hashes {
        hash.update(block);
    }
    Ok(format!("{:x}", hash.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn block_hash_is_ordered_and_independent_of_workers() -> Result<()> {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("synthetic");
        let mut file = File::create(&path).unwrap();
        for byte in [1, 2, 3] {
            file.write_all(&vec![byte; BLOCK_BYTES]).unwrap();
        }
        file.write_all(b"partial block").unwrap();
        let stamp = Stamp::read(&path, &file.metadata().unwrap())?;
        let mut expected = Sha256::new();
        expected.update(b"saragossa-state-source-blocks-v1\0");
        expected.update(stamp.bytes.to_le_bytes());
        expected.update((BLOCK_BYTES as u64).to_le_bytes());
        for byte in [1, 2, 3] {
            expected.update(Sha256::digest(vec![byte; BLOCK_BYTES]));
        }
        expected.update(Sha256::digest(b"partial block"));
        let expected = format!("{:x}", expected.finalize());
        for workers in [1, 2, 3, 8] {
            assert_eq!(hash_blocks(&path, &stamp, workers)?, expected);
        }
        file.seek(SeekFrom::Start(BLOCK_BYTES as u64)).unwrap();
        file.write_all(&[9]).unwrap();
        assert_ne!(Source::read(&path)?.content_sha256, expected);
        Ok(())
    }

    #[test]
    fn empty_source_and_metadata_race() -> Result<()> {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("synthetic");
        let mut file = File::create(&path).unwrap();
        let empty = Source::read(&path)?;
        assert_eq!(empty.content_sha256, hash_blocks(&path, &empty.stamp, 8)?);
        file.write_all(b"changed before hashing").unwrap();
        assert!(hash_blocks(&path, &empty.stamp, 1).is_err()); // même sans bloc à lire
        assert!(empty.verify().is_err()); // contrôle final, même pour zéro octet
        let before = Source::read(&path)?;
        file.set_len(1).unwrap();
        assert!(hash_blocks(&path, &before.stamp, 1).is_err());
        Ok(())
    }
}
