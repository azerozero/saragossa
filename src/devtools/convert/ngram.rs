//! Table n-gram : recopie FP8 compacte et vérification des paramètres de hachage.
//!
//! Les 128 shards `[2 500 012, 160]` sont concaténés dans l'ordre numérique en
//! un seul fichier : il forme exactement la table `[320 001 536, 160]` en
//! E4M3FN, ligne `r` à l'offset `r × 160`. La valeur vaut
//! `f32(e4m3) × weight_scale` avec une échelle **per-tensor** unique.

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::plan::{hex, io_err, Source, TensorRef, IO_CHUNK};
use crate::{InferError, Result};

/// Seed du hachage n-gram : défaut de `Qwen4ExpTextConfig`, absent de
/// `config.json` ; vérifié contre les multiplicateurs du checkpoint.
pub(crate) const NGRAM_SEED: u64 = 1234;

const SPLITMIX_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
const SPLITMIX_M1: u64 = 0xBF58_476D_1CE4_E5B9;
const SPLITMIX_M2: u64 = 0x94D0_49BB_1331_11EB;
const PRIME_1: u64 = 10_007;

fn splitmix64(value: u64) -> u64 {
    let value = value.wrapping_add(SPLITMIX_GAMMA);
    let value = (value ^ (value >> 30)).wrapping_mul(SPLITMIX_M1);
    let value = (value ^ (value >> 27)).wrapping_mul(SPLITMIX_M2);
    value ^ (value >> 31)
}

/// Multiplicateurs de couche (`_build_layer_multipliers` de transformers).
pub(crate) fn layer_multipliers(
    unigram_vocab: u64,
    ngram_size: usize,
    ple_layer_index: u64,
    seed: u64,
) -> Vec<i64> {
    let max_long = i64::MAX as u64;
    let multiplier_max = max_long / unigram_vocab.max(1);
    let half_bound = (multiplier_max / 2).max(1);
    let base_seed = seed.wrapping_add(PRIME_1.wrapping_mul(ple_layer_index));
    (0..ngram_size as u64)
        .map(|index| {
            let value = base_seed.wrapping_add(SPLITMIX_GAMMA.wrapping_mul(index + 1));
            // Borné par 2·half_bound+1 ≤ i64::MAX : la conversion est exacte.
            (2 * (splitmix64(value) % half_bound) + 1) as i64
        })
        .collect()
}

fn is_prime(value: u64) -> bool {
    if value < 2 {
        return false;
    }
    if value % 2 == 0 {
        return value == 2;
    }
    let mut divisor = 3;
    while divisor * divisor <= value {
        if value % divisor == 0 {
            return false;
        }
        divisor += 2;
    }
    true
}

/// Tailles de vocabulaire par tête : `(i+1)`-ième premier après `base − 1`.
pub(crate) fn head_vocab_sizes(base: u64, heads: usize) -> Vec<i64> {
    let mut sizes = Vec::with_capacity(heads);
    let mut prime = base - 1;
    for _ in 0..heads {
        prime += 1;
        while !is_prime(prime) {
            prime += 1;
        }
        sizes.push(prime as i64);
    }
    sizes
}

/// Paramètres de hachage vérifiés, publiés dans `meta.json`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct HashParams {
    pub(crate) seed: u64,
    pub(crate) ngram_size: usize,
    pub(crate) heads: usize,
    pub(crate) layer_multipliers: Vec<i64>,
    pub(crate) head_vocab_sizes: Vec<i64>,
    pub(crate) head_offsets: Vec<i64>,
    pub(crate) total_vocab: u64,
    pub(crate) padded_rows: u64,
}

/// Paramètres structurels de la config texte utiles au hachage.
#[derive(Clone, Copy, Debug)]
pub(crate) struct NgramConfig {
    pub(crate) vocab_size: u64,
    pub(crate) ngram_size: usize,
    pub(crate) heads_per_ngram: usize,
    pub(crate) vocab_base: u64,
    pub(crate) divisible_by: u64,
}

fn i64_values(bytes: &[u8], name: &str) -> Result<Vec<i64>> {
    let chunks = bytes.chunks_exact(8);
    if !chunks.remainder().is_empty() {
        return Err(InferError::Shape(format!("{name}: I64 mal aligné")));
    }
    Ok(chunks
        .map(|chunk| {
            let mut raw = [0_u8; 8];
            raw.copy_from_slice(chunk);
            i64::from_le_bytes(raw)
        })
        .collect())
}

/// Recalcule les paramètres de hachage et les compare aux buffers du checkpoint.
///
/// # Errors
///
/// Renvoie une erreur si un buffer diverge du recalcul (seed, premiers,
/// offsets) ou si la table ne compte pas exactement `padded_rows` lignes.
pub(crate) fn verify_hash_params(
    config: NgramConfig,
    multipliers: &[u8],
    vocab_sizes: &[u8],
    offsets: &[u8],
    table_rows: u64,
) -> Result<HashParams> {
    let heads = (config.ngram_size - 1) * config.heads_per_ngram;
    let expected_mult = layer_multipliers(config.vocab_size, config.ngram_size, 0, NGRAM_SEED);
    let stored_mult = i64_values(multipliers, "layer_multipliers")?;
    if stored_mult != expected_mult {
        return Err(InferError::Config(format!(
            "multiplicateurs n-gram {stored_mult:?} ≠ recalcul seed {NGRAM_SEED} {expected_mult:?}"
        )));
    }
    let expected_sizes = head_vocab_sizes(config.vocab_base, heads);
    let stored_sizes = i64_values(vocab_sizes, "ngram_heads_vocab_sizes")?;
    if stored_sizes != expected_sizes {
        return Err(InferError::Config(format!(
            "tailles de têtes n-gram {stored_sizes:?} ≠ recalcul {expected_sizes:?}"
        )));
    }
    let mut expected_offsets = Vec::with_capacity(heads);
    let mut total = 0_i64;
    for size in &expected_sizes {
        expected_offsets.push(total);
        total += size;
    }
    let stored_offsets = i64_values(offsets, "ngram_heads_offsets")?;
    if stored_offsets != expected_offsets {
        return Err(InferError::Config(format!(
            "offsets de têtes n-gram {stored_offsets:?} ≠ recalcul {expected_offsets:?}"
        )));
    }
    let total = total as u64;
    let padded = total.div_ceil(config.divisible_by) * config.divisible_by;
    if padded != table_rows {
        return Err(InferError::Config(format!(
            "table n-gram: {table_rows} lignes, {padded} attendues ({total} arrondi à {})",
            config.divisible_by
        )));
    }
    Ok(HashParams {
        seed: NGRAM_SEED,
        ngram_size: config.ngram_size,
        heads,
        layer_multipliers: stored_mult,
        head_vocab_sizes: stored_sizes,
        head_offsets: stored_offsets,
        total_vocab: total,
        padded_rows: padded,
    })
}

/// Segment recopié : un shard source `shard_P`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct SegmentRecord {
    pub(crate) part: usize,
    pub(crate) source: String,
    /// Offset physique dans `ngram_e4m3.bin` (distinct de `ngram_heads_offsets`,
    /// qui décrit l'espace logique des lignes).
    pub(crate) offset: u64,
    pub(crate) len: u64,
    pub(crate) rows: usize,
    pub(crate) sha256: String,
}

/// Recopie le shard `tensor` à `offset` dans `file`, en refusant tout octet NaN.
///
/// # Errors
///
/// Renvoie une erreur de lecture, d'écriture, ou si le shard contient un octet
/// `0x7f`/`0xff` (NaN E4M3FN).
pub(crate) fn copy_segment(
    source: &Source,
    tensor: &TensorRef,
    part: usize,
    file: &File,
    path: &Path,
    offset: u64,
) -> Result<SegmentRecord> {
    let mut sha = Sha256::new();
    let mut start = 0;
    while start < tensor.len {
        let len = IO_CHUNK.min(tensor.len - start);
        let bytes = source.read(tensor, start, len)?;
        let nan = bytes.iter().filter(|byte| **byte & 0x7f == 0x7f).count();
        if nan != 0 {
            return Err(InferError::Config(format!(
                "{}: {nan} octets NaN E4M3 dans [{start}, {})",
                tensor.name,
                start + len
            )));
        }
        file.write_all_at(&bytes, offset + start as u64)
            .map_err(|source| io_err(path, source))?;
        sha.update(&bytes);
        start += len;
    }
    Ok(SegmentRecord {
        part,
        source: tensor.name.clone(),
        offset,
        len: tensor.len as u64,
        rows: tensor.shape[0],
        sha256: hex(sha.finalize()),
    })
}
