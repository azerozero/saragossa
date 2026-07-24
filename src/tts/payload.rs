use crate::safetensor::bytes_to_dense_f32;
use crate::{InferError, Result, Tensor};
use safetensors::Dtype;
use serde::Deserialize;
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub(crate) struct SafetensorPayload {
    path: PathBuf,
    data_start: u64,
    pub(super) entries: HashMap<String, PayloadEntry>,
}

#[derive(Clone, Debug)]
pub(super) struct PayloadEntry {
    pub(super) dtype: Dtype,
    pub(super) shape: Vec<usize>,
    pub(super) offsets: [u64; 2],
}

#[derive(Clone, Copy, Debug)]
pub(super) struct PayloadReadSummary {
    pub(super) bytes: u64,
    pub(super) bytes_read: u64,
    pub(super) checksum: u64,
}

/// Offset basis FNV-1a 64 bits (constante officielle de l'algorithme FNV).
pub(super) const FNV1A64_OFFSET_BASIS: u64 = 0xcbf29ce484222325;

/// Prime FNV-1a 64 bits (constante officielle de l'algorithme FNV).
const FNV1A64_PRIME: u64 = 0x100000001b3;

/// Combine un octet dans un hash FNV-1a 64 bits (`(hash XOR octet) * prime`).
///
/// Sert de checksum de non-régression sur les octets bruts du payload codec
/// TTS (détecte une corruption/dérive de poids) : ce n'est PAS un usage
/// cryptographique, FNV n'offre aucune résistance aux collisions adverses.
pub(super) fn fnv1a64_update(hash: u64, byte: u8) -> u64 {
    (hash ^ u64::from(byte)).wrapping_mul(FNV1A64_PRIME)
}

impl SafetensorPayload {
    pub(crate) fn open(path: &Path) -> Result<Self> {
        let mut file = std::fs::File::open(path).map_err(|source| InferError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let mut len_bytes = [0_u8; 8];
        file.read_exact(&mut len_bytes)
            .map_err(|source| InferError::Io {
                path: path.to_path_buf(),
                source,
            })?;
        let header_len = u64::from_le_bytes(len_bytes);
        if header_len == 0 || header_len > 128 * 1024 * 1024 {
            return Err(InferError::SafetensorsHeader {
                path: path.to_path_buf(),
                message: format!("taille header invalide: {header_len}"),
            });
        }
        let header_len_usize =
            usize::try_from(header_len).map_err(|_| InferError::SafetensorsHeader {
                path: path.to_path_buf(),
                message: format!("taille header non représentable: {header_len}"),
            })?;
        let mut header = vec![0_u8; header_len_usize];
        file.read_exact(&mut header)
            .map_err(|source| InferError::Io {
                path: path.to_path_buf(),
                source,
            })?;
        let json: serde_json::Value =
            serde_json::from_slice(&header).map_err(|source| InferError::Json {
                path: path.to_path_buf(),
                source,
            })?;
        let object = json
            .as_object()
            .ok_or_else(|| InferError::SafetensorsHeader {
                path: path.to_path_buf(),
                message: "header JSON non objet".to_string(),
            })?;
        let mut entries = HashMap::new();
        for (name, value) in object {
            if name == "__metadata__" {
                continue;
            }
            let raw: RawPayloadEntry =
                serde_json::from_value(value.clone()).map_err(|source| InferError::Json {
                    path: path.to_path_buf(),
                    source,
                })?;
            entries.insert(
                name.clone(),
                PayloadEntry {
                    dtype: parse_dtype(&raw.dtype).ok_or_else(|| {
                        InferError::Config(format!(
                            "dtype safetensors TTS non supporté pour {name}: {}",
                            raw.dtype
                        ))
                    })?,
                    shape: raw.shape,
                    offsets: raw.data_offsets,
                },
            );
        }
        Ok(Self {
            path: path.to_path_buf(),
            data_start: 8 + header_len,
            entries,
        })
    }

    pub(crate) fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    pub(crate) fn names(&self) -> Vec<String> {
        self.entries.keys().cloned().collect()
    }

    pub(super) fn entry(&self, name: &str) -> Result<&PayloadEntry> {
        self.entries
            .get(name)
            .ok_or_else(|| InferError::MissingWeight(name.to_string()))
    }

    pub(super) fn read_payload_summary(&self) -> Result<PayloadReadSummary> {
        let mut bytes = 0_u64;
        let mut bytes_read = 0_u64;
        let mut checksum = FNV1A64_OFFSET_BASIS;
        let mut names = self.entries.keys().collect::<Vec<_>>();
        names.sort();
        for name in names {
            let entry = self.entry(name)?;
            let entry_bytes = self.read_entry_bytes(entry)?;
            let len = u64::try_from(entry_bytes.len()).map_err(|_| {
                InferError::Shape("entrée safetensors codec trop grande".to_string())
            })?;
            bytes = bytes.checked_add(len).ok_or_else(|| {
                InferError::Shape("payload safetensors codec trop grand".to_string())
            })?;
            bytes_read = bytes_read.checked_add(len).ok_or_else(|| {
                InferError::Shape("payload safetensors codec lu trop grand".to_string())
            })?;
            for byte in entry_bytes {
                checksum = fnv1a64_update(checksum, byte);
            }
        }
        Ok(PayloadReadSummary {
            bytes,
            bytes_read,
            checksum,
        })
    }

    pub(crate) fn read_dense_tensor(&self, name: &str) -> Result<Tensor> {
        let entry = self.entry(name)?;
        let bytes = self.read_entry_bytes(entry)?;
        Tensor::from_vec(
            entry.shape.clone(),
            bytes_to_dense_f32(&bytes, entry.dtype, name)?,
        )
    }

    pub(crate) fn read_u32_tensor(&self, name: &str) -> Result<Vec<u32>> {
        let entry = self.entry(name)?;
        if entry.dtype != Dtype::U32 {
            return Err(InferError::UnsupportedDtype {
                name: name.to_string(),
                dtype: entry.dtype,
            });
        }
        let bytes = self.read_entry_bytes(entry)?;
        bytes_to_u32(&bytes, name)
    }

    fn read_entry_bytes(&self, entry: &PayloadEntry) -> Result<Vec<u8>> {
        let len = entry.offsets[1]
            .checked_sub(entry.offsets[0])
            .ok_or_else(|| InferError::Shape("offset safetensors inversé".to_string()))?;
        let offset = self
            .data_start
            .checked_add(entry.offsets[0])
            .ok_or_else(|| InferError::Shape("offset safetensors absolu trop grand".to_string()))?;
        let len_usize = usize::try_from(len)
            .map_err(|_| InferError::Shape(format!("entrée safetensors trop grande: {len}")))?;
        let mut file = std::fs::File::open(&self.path).map_err(|source| InferError::Io {
            path: self.path.clone(),
            source,
        })?;
        file.seek(SeekFrom::Start(offset))
            .map_err(|source| InferError::Io {
                path: self.path.clone(),
                source,
            })?;
        let mut bytes = vec![0_u8; len_usize];
        file.read_exact(&mut bytes)
            .map_err(|source| InferError::Io {
                path: self.path.clone(),
                source,
            })?;
        Ok(bytes)
    }
}

#[derive(Debug, Deserialize)]
struct RawPayloadEntry {
    pub(super) dtype: String,
    pub(super) shape: Vec<usize>,
    data_offsets: [u64; 2],
}

fn parse_dtype(dtype: &str) -> Option<Dtype> {
    match dtype {
        "F32" => Some(Dtype::F32),
        "F16" => Some(Dtype::F16),
        "BF16" => Some(Dtype::BF16),
        "U32" => Some(Dtype::U32),
        "F8_E4M3" => Some(Dtype::F8_E4M3),
        "F8_E5M2" => Some(Dtype::F8_E5M2),
        _ => None,
    }
}

fn bytes_to_u32(bytes: &[u8], name: &str) -> Result<Vec<u32>> {
    let chunks = bytes.chunks_exact(4);
    if !chunks.remainder().is_empty() {
        return Err(InferError::Shape(format!(
            "tensor {name} U32 avec {} octets non multiple de 4",
            bytes.len()
        )));
    }
    chunks
        .map(|chunk| {
            let arr = <[u8; 4]>::try_from(chunk)
                .map_err(|_| InferError::Shape(format!("chunk U32 invalide pour {name}")))?;
            Ok(u32::from_le_bytes(arr))
        })
        .collect()
}
