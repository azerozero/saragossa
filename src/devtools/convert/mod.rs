//! Convertisseur en flux `qwen4_exp` FP8 officiel → MLX affine mixte « q4e8 ».
//!
//! Experts routés u4 gs64 (déquantification FP8 par blocs 128×128), denses
//! u8 gs64, lm_head u6 gs64, petits tenseurs copiés à l'octet, table n-gram
//! recopiée en E4M3 dans un fichier mmappable, vision exclue. Un tenseur à la
//! fois, lectures bornées, en-têtes figés avant écriture, reprise par shard de
//! sortie et par segment n-gram, journal persistant les entrées du manifeste.

mod ngram;
mod plan;
mod rules;

#[cfg(test)]
mod tests;

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use self::ngram::{copy_segment, verify_hash_params, HashParams, NgramConfig, SegmentRecord};
pub use self::plan::Counts;
use self::plan::{
    build_header, file_sha256, hex, io_err, plan_shards, shard_size, write_shard, Expected,
    Inventory, PeakTracker, ShardPlan, Source, TensorRecord,
};
use self::rules::{
    LayerType, Role, DENSE_BITS, EXPERT_BITS, GROUP_SIZE, LM_HEAD_BITS, RULES_VERSION,
};
use crate::qwen_loader::read_shard_header;
use crate::{InferError, Result};

const JOURNAL: &str = ".convert/journal.json";
const LOCK: &str = ".convert/lock";
const NGRAM_FILE: &str = "ngram/ngram_e4m3.bin";
const NGRAM_META: &str = "ngram/meta.json";
const MANIFEST: &str = "convert_manifest.json";
/// Fichiers texte recopiés tels quels (le tokenizer est obligatoire).
const SIDE_FILES: &[(&str, bool)] = &[
    ("tokenizer.json", true),
    ("tokenizer_config.json", true),
    ("vocab.json", false),
    ("merges.txt", false),
    ("chat_template.jinja", false),
    ("generation_config.json", false),
    ("LICENSE", true),
];
/// Clés retirées de la config : quantification FP8 et vision (sortie texte seul).
const CONFIG_DROPPED: &[&str] = &[
    "quantization_config",
    "modules_to_convert",
    "vision_config",
    "image_token_id",
    "video_token_id",
    "vision_start_token_id",
    "vision_end_token_id",
];
const PARAMS: &str = "experts u4 gs64 (FP8 b128 déquantifié) ; denses u8 gs64 ; lm_head u6 gs64 ; \
scales/biais bf16 (biais vers −∞, scale vers +∞) ; copies à l'octet ; n-gram E4M3 compact";

/// Options du convertisseur.
#[derive(Clone, Debug)]
pub struct ConvertOptions {
    /// Checkpoint FP8 officiel (`Qwen/Qwen3.8-Flash-Next-FP8`).
    pub src: PathBuf,
    /// Dossier de sortie MLX.
    pub dst: PathBuf,
    /// Révision Hugging Face complète de la source, consignée au manifeste.
    pub source_revision: String,
    /// Unités à traiter seules (pilote) ; `None` = conversion complète et finalisation.
    pub only: Option<Vec<String>>,
    /// Vérifie le sha256 complet des unités déjà terminées au lieu de leur taille.
    pub verify: bool,
    /// Inventaire et plan seulement : aucune écriture.
    pub plan_only: bool,
}

/// Identité d'un travail : une reprise est refusée si elle change.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
struct Identity {
    source_revision: String,
    source_index_sha256: String,
    source_config_sha256: String,
    rules_version: String,
    params: String,
    converter_binary_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ShardRecord {
    file: String,
    size: u64,
    sha256: String,
    tensors: Vec<TensorRecord>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Journal {
    identity: Identity,
    shards: BTreeMap<String, ShardRecord>,
    segments: BTreeMap<usize, SegmentRecord>,
}

/// Fichier de sortie et son empreinte.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OutputFile {
    pub name: String,
    pub size: u64,
    pub sha256: String,
}

/// Bilan de la conversion.
#[derive(Clone, Debug, Serialize)]
pub struct ConvertReport {
    /// Partition de la source.
    pub counts: Counts,
    /// Octets planifiés par fichier de sortie (en-têtes compris).
    pub planned_bytes: BTreeMap<String, u64>,
    pub units_written: Vec<String>,
    pub units_resumed: Vec<String>,
    pub finalized: bool,
    pub elapsed_s: f64,
    /// Pic d'empreinte physique échantillonné (`phys_footprint`).
    pub peak_footprint_bytes: u64,
    pub output_bytes: u64,
    /// Octets par catégorie de sortie.
    pub bytes_by_category: BTreeMap<String, u64>,
    pub files: Vec<OutputFile>,
}

fn json_err(path: &Path, source: serde_json::Error) -> InferError {
    InferError::Json {
        path: path.to_path_buf(),
        source,
    }
}

fn read_json(path: &Path) -> Result<serde_json::Value> {
    let raw = std::fs::read(path).map_err(|source| io_err(path, source))?;
    serde_json::from_slice(&raw).map_err(|source| json_err(path, source))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    hex(Sha256::digest(bytes))
}

fn sync_dir(dir: &Path) -> Result<()> {
    File::open(dir)
        .and_then(|file| file.sync_all())
        .map_err(|source| io_err(dir, source))
}

/// Écriture atomique : fichier temporaire, `fsync`, renommage, `fsync` du dossier.
fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = path.with_extension("tmp-write");
    std::fs::write(&tmp, bytes).map_err(|source| io_err(&tmp, source))?;
    File::open(&tmp)
        .and_then(|file| file.sync_all())
        .map_err(|source| io_err(&tmp, source))?;
    std::fs::rename(&tmp, path).map_err(|source| io_err(path, source))?;
    sync_dir(path.parent().unwrap_or(Path::new(".")))
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value).map_err(|source| json_err(path, source))?;
    write_atomic(path, &bytes)
}

/// Verrou de destination, indépendant du verrou GPU ; libéré à la destruction.
struct DestLock(PathBuf);

impl DestLock {
    fn acquire(dst: &Path) -> Result<Self> {
        let path = dst.join(LOCK);
        std::fs::create_dir(&path).map_err(|source| {
            InferError::Config(format!(
                "destination verrouillée ou inaccessible ({}): {source}",
                path.display()
            ))
        })?;
        Ok(Self(path))
    }
}

impl Drop for DestLock {
    fn drop(&mut self) {
        // NOTE: échec ignoré à la destruction ; un verrou résiduel se retire à la main.
        let _ = std::fs::remove_dir(&self.0);
    }
}

fn text_config(config: &serde_json::Value) -> &serde_json::Value {
    config.get("text_config").unwrap_or(config)
}

fn usize_field(config: &serde_json::Value, key: &str) -> Result<usize> {
    text_config(config)
        .get(key)
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| InferError::Config(format!("config: {key} entier attendu")))
}

fn expected_from(config: &serde_json::Value) -> Result<(Expected, NgramConfig)> {
    let model_type = config.get("model_type").and_then(serde_json::Value::as_str);
    if model_type != Some("qwen4_exp") {
        return Err(InferError::Config(format!(
            "convertisseur qwen4_exp: model_type={model_type:?}"
        )));
    }
    let text = text_config(config);
    let layers = usize_field(config, "num_hidden_layers")?;
    let layer_types = text
        .get("layer_types")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| InferError::Config("config: layer_types absent".to_string()))?
        .iter()
        .map(|value| {
            value
                .as_str()
                .ok_or_else(|| InferError::Config("layer_types non textuel".to_string()))
                .and_then(LayerType::parse)
        })
        .collect::<Result<Vec<_>>>()?;
    if layer_types.len() != layers {
        return Err(InferError::Config(format!(
            "layer_types: {} entrées pour {layers} couches",
            layer_types.len()
        )));
    }
    let mtp_layers = usize_field(config, "mtp_num_hidden_layers")?;
    let mtp_type = match mtp_layers {
        0 => None,
        _ => {
            let kind = text
                .get("mtp")
                .and_then(|mtp| mtp.get("layer_types"))
                .and_then(serde_json::Value::as_array)
                .and_then(|types| types.first())
                .and_then(serde_json::Value::as_str)
                .unwrap_or("full_attention");
            Some(LayerType::parse(kind)?)
        }
    };
    let expected = Expected {
        layer_types,
        mtp_type,
        layers,
        experts: usize_field(config, "num_experts")?,
        mtp_layers,
        ngram_parts: usize_field(config, "split_ngram_parts")?,
    };
    if expected.mtp_layers > 1 {
        return Err(InferError::Config(format!(
            "mtp_num_hidden_layers={} non supporté",
            expected.mtp_layers
        )));
    }
    let ngram = NgramConfig {
        vocab_size: usize_field(config, "vocab_size")? as u64,
        ngram_size: usize_field(config, "ngram_size")?,
        heads_per_ngram: usize_field(config, "heads_per_ngram")?,
        vocab_base: usize_field(config, "ngram_vocab_size_base")? as u64,
        divisible_by: usize_field(config, "make_ngram_vocab_size_divisible_by")? as u64,
    };
    Ok((expected, ngram))
}

fn other_bytes(source: &Source, inventory: &Inventory, suffix: &str) -> Result<Vec<u8>> {
    let (_, tensor, _, _) = inventory
        .others
        .iter()
        .find(|(role, tensor, _, _)| *role == Role::CopyI64 && tensor.name.ends_with(suffix))
        .ok_or_else(|| InferError::MissingWeight(suffix.to_string()))?;
    source.read(tensor, 0, tensor.len)
}

/// Config de sortie : quantification MLX, vision retirée, texte seul.
fn output_config(
    src_config: &serde_json::Value,
    records: &[TensorRecord],
) -> Result<serde_json::Value> {
    let mut config = src_config.clone();
    let object = config
        .as_object_mut()
        .ok_or_else(|| InferError::Config("config.json non objet".to_string()))?;
    for key in CONFIG_DROPPED {
        object.remove(*key);
    }
    object.insert("language_model_only".to_string(), true.into());
    let mut quant = serde_json::Map::new();
    quant.insert("group_size".to_string(), GROUP_SIZE.into());
    quant.insert("bits".to_string(), EXPERT_BITS.into());
    quant.insert("mode".to_string(), "affine".into());
    for record in records {
        let (Some(bits), Some(base)) = (record.bits, record.name.strip_suffix(".weight")) else {
            continue;
        };
        if bits != EXPERT_BITS {
            quant.insert(
                base.to_string(),
                serde_json::json!({ "group_size": GROUP_SIZE, "bits": bits }),
            );
        }
    }
    object.insert("quantization".to_string(), quant.into());
    Ok(config)
}

fn category(name: &str, record: &TensorRecord) -> &'static str {
    if name.contains(".switch_mlp.") {
        if name.starts_with("mtp.") {
            "mtp_experts_u4"
        } else {
            "experts_u4"
        }
    } else if record.bits == Some(LM_HEAD_BITS) || name.starts_with("language_model.lm_head.") {
        "lm_head_u6"
    } else if record.bits == Some(DENSE_BITS)
        || name.ends_with(".scales")
        || name.ends_with(".biases")
    {
        if name.starts_with("mtp.") {
            "mtp_dense_u8"
        } else {
            "dense_u8"
        }
    } else {
        "copies"
    }
}

fn shard_unit(plan: &ShardPlan) -> String {
    plan.file.trim_end_matches(".safetensors").to_string()
}

fn segment_unit(part: usize) -> String {
    format!("ngram-{part:03}")
}

/// sha256 de la plage `[offset, offset + len)` d'un fichier.
fn range_sha256(file: &File, path: &Path, offset: u64, len: u64) -> Result<String> {
    let mut sha = Sha256::new();
    let mut buf = vec![0_u8; plan::IO_CHUNK];
    let mut at = 0_u64;
    while at < len {
        let take = (plan::IO_CHUNK as u64).min(len - at) as usize;
        std::os::unix::fs::FileExt::read_exact_at(file, &mut buf[..take], offset + at)
            .map_err(|e| io_err(path, e))?;
        sha.update(&buf[..take]);
        at += take as u64;
    }
    Ok(hex(sha.finalize()))
}

/// Contrôle rapide d'une unité terminée : taille, et sha256 si `verify`.
fn check_done(path: &Path, size: u64, sha: &str, verify: bool) -> Result<bool> {
    let Ok(meta) = std::fs::metadata(path) else {
        return Ok(false);
    };
    if meta.len() != size {
        return Ok(false);
    }
    if verify {
        return Ok(file_sha256(path)? == sha);
    }
    Ok(true)
}

/// Lance (ou reprend) la conversion.
///
/// # Errors
///
/// Renvoie une erreur si la source est incohérente, si l'identité du travail
/// diffère d'une reprise, si la destination est verrouillée, ou sur toute
/// erreur de lecture, de quantification ou d'écriture.
pub fn convert_qwen4exp(options: &ConvertOptions) -> Result<ConvertReport> {
    let started = Instant::now();
    let mut peak = PeakTracker::default();
    let config_path = options.src.join("config.json");
    let src_config = read_json(&config_path)?;
    let (expected, ngram_config) = expected_from(&src_config)?;
    let (source, refs) = Source::open(&options.src)?;
    let inventory = Inventory::build(&refs, &expected)?;
    let table_rows: u64 = inventory.ngram.iter().map(|t| t.shape[0] as u64).sum();
    let hash_params = verify_hash_params(
        ngram_config,
        &other_bytes(&source, &inventory, "layer_multipliers")?,
        &other_bytes(&source, &inventory, "ngram_heads_vocab_sizes")?,
        &other_bytes(&source, &inventory, "ngram_heads_offsets")?,
        table_rows,
    )?;
    let plans = plan_shards(&inventory, expected.layers)?;
    let mut planned_bytes = BTreeMap::new();
    for plan in &plans {
        planned_bytes.insert(plan.file.clone(), shard_size(plan)?);
    }
    planned_bytes.insert(
        NGRAM_FILE.to_string(),
        inventory.ngram.iter().map(|t| t.len as u64).sum(),
    );
    if options.plan_only {
        return Ok(ConvertReport {
            counts: inventory.counts.clone(),
            output_bytes: planned_bytes.values().sum(),
            planned_bytes,
            units_written: Vec::new(),
            units_resumed: Vec::new(),
            finalized: false,
            elapsed_s: started.elapsed().as_secs_f64(),
            peak_footprint_bytes: peak.peak_bytes,
            bytes_by_category: BTreeMap::new(),
            files: Vec::new(),
        });
    }

    let index_bytes = std::fs::read(source.dir.join("model.safetensors.index.json"))
        .map_err(|e| io_err(&source.dir, e))?;
    let config_bytes = std::fs::read(&config_path).map_err(|e| io_err(&config_path, e))?;
    let exe = std::env::current_exe().map_err(|e| io_err(Path::new("current_exe"), e))?;
    let identity = Identity {
        source_revision: options.source_revision.clone(),
        source_index_sha256: sha256_bytes(&index_bytes),
        source_config_sha256: sha256_bytes(&config_bytes),
        rules_version: RULES_VERSION.to_string(),
        params: PARAMS.to_string(),
        converter_binary_sha256: file_sha256(&exe)?,
    };

    let dst = &options.dst;
    for dir in [dst.clone(), dst.join(".convert"), dst.join("ngram")] {
        std::fs::create_dir_all(&dir).map_err(|e| io_err(&dir, e))?;
    }
    let _lock = DestLock::acquire(dst)?;
    let journal_path = dst.join(JOURNAL);
    let mut journal = if journal_path.exists() {
        let journal: Journal = serde_json::from_value(read_json(&journal_path)?)
            .map_err(|e| json_err(&journal_path, e))?;
        if journal.identity != identity {
            return Err(InferError::Config(format!(
                "reprise refusée: identité différente (journal {:?}, travail {:?})",
                journal.identity, identity
            )));
        }
        journal
    } else {
        Journal {
            identity: identity.clone(),
            shards: BTreeMap::new(),
            segments: BTreeMap::new(),
        }
    };
    let wanted = |unit: &str| {
        options
            .only
            .as_ref()
            .is_none_or(|only| only.iter().any(|item| item == unit))
    };
    if let Some(only) = &options.only {
        let known: Vec<String> = plans
            .iter()
            .map(shard_unit)
            .chain((0..inventory.ngram.len()).map(segment_unit))
            .collect();
        if let Some(bad) = only.iter().find(|unit| !known.contains(unit)) {
            return Err(InferError::Config(format!("unité inconnue: {bad}")));
        }
    }
    let mut written = Vec::new();
    let mut resumed = Vec::new();

    for plan in &plans {
        let unit = shard_unit(plan);
        if !wanted(&unit) {
            continue;
        }
        let path = dst.join(&plan.file);
        let size = shard_size(plan)?;
        if let Some(done) = journal.shards.get(&unit) {
            if done.size == size && check_done(&path, size, &done.sha256, options.verify)? {
                resumed.push(unit);
                continue;
            }
        }
        let tmp = dst.join(format!("{}.tmp", plan.file));
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&tmp)
            .map_err(|e| io_err(&tmp, e))?;
        file.set_len(size).map_err(|e| io_err(&tmp, e))?;
        let tensors = write_shard(&source, &inventory, plan, &file, &tmp, &mut peak)?;
        file.sync_all().map_err(|e| io_err(&tmp, e))?;
        drop(file);
        std::fs::rename(&tmp, &path).map_err(|e| io_err(&path, e))?;
        sync_dir(dst)?;
        let sha256 = file_sha256(&path)?;
        journal.shards.insert(
            unit.clone(),
            ShardRecord {
                file: plan.file.clone(),
                size,
                sha256,
                tensors,
            },
        );
        write_json(&journal_path, &journal)?;
        written.push(unit);
        peak.sample();
    }

    let segment_len = inventory.ngram[0].len as u64;
    let ngram_total: u64 = inventory.ngram.iter().map(|t| t.len as u64).sum();
    let ngram_path = dst.join(NGRAM_FILE);
    let ngram_needed = (0..inventory.ngram.len()).any(|part| wanted(&segment_unit(part)));
    if ngram_needed {
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&ngram_path)
            .map_err(|e| io_err(&ngram_path, e))?;
        // Un fichier absent, tronqué ou rallongé invalide les segments qu'il ne
        // contient plus intégralement, AVANT tout redimensionnement : `set_len`
        // comblerait les trous de zéros qu'aucun contrôle de taille ne verrait.
        let len = file.metadata().map_err(|e| io_err(&ngram_path, e))?.len();
        if len != ngram_total {
            let before = journal.segments.len();
            journal
                .segments
                .retain(|_, done| done.offset + done.len <= len.min(ngram_total));
            if journal.segments.len() != before {
                write_json(&journal_path, &journal)?;
            }
            file.set_len(ngram_total)
                .map_err(|e| io_err(&ngram_path, e))?;
        }
        for (part, tensor) in inventory.ngram.iter().enumerate() {
            let unit = segment_unit(part);
            if !wanted(&unit) {
                continue;
            }
            if tensor.len as u64 != segment_len {
                return Err(InferError::Shape(format!(
                    "{}: segment de {} octets, {segment_len} attendus",
                    tensor.name, tensor.len
                )));
            }
            let offset = part as u64 * segment_len;
            // Un segment journalisé est toujours rehaché (même sans --verify) :
            // une corruption de même taille serait sinon invisible.
            if let Some(done) = journal.segments.get(&part) {
                if done.offset == offset
                    && done.len == segment_len
                    && range_sha256(&file, &ngram_path, offset, done.len)? == done.sha256
                {
                    resumed.push(unit);
                    continue;
                }
            }
            let record = copy_segment(&source, tensor, part, &file, &ngram_path, offset)?;
            file.sync_data().map_err(|e| io_err(&ngram_path, e))?;
            journal.segments.insert(part, record);
            write_json(&journal_path, &journal)?;
            written.push(unit);
            peak.sample();
        }
    }

    let finalized = options.only.is_none();
    let files = if finalized {
        finalize(
            options,
            &source,
            &inventory,
            &plans,
            &mut journal,
            &journal_path,
            &src_config,
            &hash_params,
            &identity,
        )?
    } else {
        Vec::new()
    };
    peak.sample();

    let mut bytes_by_category = BTreeMap::new();
    for shard in journal.shards.values() {
        for record in &shard.tensors {
            *bytes_by_category
                .entry(category(&record.name, record).to_string())
                .or_insert(0) += record.data_offsets[1] - record.data_offsets[0];
        }
    }
    let ngram_done: u64 = journal.segments.values().map(|s| s.len).sum();
    bytes_by_category.insert("ngram_e4m3".to_string(), ngram_done);
    let output_bytes = journal.shards.values().map(|s| s.size).sum::<u64>() + ngram_done;
    Ok(ConvertReport {
        counts: inventory.counts.clone(),
        planned_bytes,
        units_written: written,
        units_resumed: resumed,
        finalized,
        elapsed_s: started.elapsed().as_secs_f64(),
        peak_footprint_bytes: peak.peak_bytes,
        output_bytes,
        bytes_by_category,
        files,
    })
}

#[derive(Serialize)]
struct NgramMeta<'a> {
    file: &'a str,
    dtype: &'a str,
    shape: [u64; 2],
    formula: &'a str,
    weight_scale_bf16_bits: String,
    weight_scale: f32,
    source_scale_tensor: &'a str,
    hash: &'a HashParams,
    segments: Vec<SegmentRecord>,
    sha256: String,
}

#[derive(Serialize)]
struct Manifest<'a> {
    format: &'a str,
    source: serde_json::Value,
    identity: &'a Identity,
    conventions: serde_json::Value,
    counts: &'a Counts,
    excluded_vision: &'a [String],
    consumed_expert_scales: usize,
    files: &'a [OutputFile],
    ngram: serde_json::Value,
    tensors: Vec<&'a TensorRecord>,
}

/// sha256 du fichier n-gram en une passe, et segments dont l'empreinte
/// diffère du journal.
fn ngram_file_sha(path: &Path, segments: &[SegmentRecord]) -> Result<(String, Vec<usize>)> {
    let file = File::open(path).map_err(|e| io_err(path, e))?;
    let total = file.metadata().map_err(|e| io_err(path, e))?.len();
    let expected: u64 = segments.iter().map(|s| s.len).sum();
    if total != expected {
        return Ok((String::new(), segments.iter().map(|s| s.part).collect()));
    }
    let mut whole = Sha256::new();
    let mut corrupt = Vec::new();
    let mut buf = vec![0_u8; plan::IO_CHUNK];
    for segment in segments {
        let mut sha = Sha256::new();
        let mut at = 0_u64;
        while at < segment.len {
            let take = (plan::IO_CHUNK as u64).min(segment.len - at) as usize;
            std::os::unix::fs::FileExt::read_exact_at(&file, &mut buf[..take], segment.offset + at)
                .map_err(|e| io_err(path, e))?;
            sha.update(&buf[..take]);
            whole.update(&buf[..take]);
            at += take as u64;
        }
        if hex(sha.finalize()) != segment.sha256 {
            corrupt.push(segment.part);
        }
    }
    Ok((hex(whole.finalize()), corrupt))
}

/// Validation finale puis écriture des métadonnées (config en dernier).
#[expect(
    clippy::too_many_arguments,
    reason = "finalisation: source, plan, journal et métadonnées restent explicites"
)]
fn finalize(
    options: &ConvertOptions,
    source: &Source,
    inventory: &Inventory,
    plans: &[ShardPlan],
    journal: &mut Journal,
    journal_path: &Path,
    src_config: &serde_json::Value,
    hash_params: &HashParams,
    identity: &Identity,
) -> Result<Vec<OutputFile>> {
    let dst = &options.dst;
    let mut files = Vec::new();
    let mut records: Vec<&TensorRecord> = Vec::new();
    let mut weight_map = serde_json::Map::new();
    for plan in plans {
        let unit = shard_unit(plan);
        let done = journal
            .shards
            .get(&unit)
            .ok_or_else(|| InferError::Config(format!("unité {unit} non terminée")))?;
        let path = dst.join(&plan.file);
        let header = read_shard_header(&path)?;
        let (_, offsets) = build_header(plan)?;
        if header.tensors.len() != offsets.len()
            || offsets.iter().any(|(name, (start, end))| {
                header
                    .tensors
                    .get(name)
                    .is_none_or(|entry| entry.data_offsets != [*start as usize, *end as usize])
            })
        {
            return Err(InferError::Config(format!(
                "{}: en-tête relu différent du plan",
                plan.file
            )));
        }
        let sha256 = file_sha256(&path)?;
        if sha256 != done.sha256 {
            // L'unité corrompue sort du journal : une relance la réécrit.
            let expected = done.sha256.clone();
            journal.shards.remove(&unit);
            write_json(journal_path, journal)?;
            return Err(InferError::Config(format!(
                "{}: sha256 {sha256} ≠ journal {expected} ; unité invalidée, relancer la conversion",
                plan.file
            )));
        }
        if plan.file != "mtp.safetensors" {
            for name in offsets.keys() {
                weight_map.insert(name.clone(), plan.file.clone().into());
            }
        }
        files.push(OutputFile {
            name: plan.file.clone(),
            size: done.size,
            sha256,
        });
        records.extend(done.tensors.iter());
    }
    let segments: Vec<SegmentRecord> = (0..inventory.ngram.len())
        .map(|part| {
            journal
                .segments
                .get(&part)
                .cloned()
                .ok_or_else(|| InferError::Config(format!("segment n-gram {part} non terminé")))
        })
        .collect::<Result<_>>()?;
    let ngram_path = dst.join(NGRAM_FILE);
    let (ngram_sha, corrupt) = ngram_file_sha(&ngram_path, &segments)?;
    if !corrupt.is_empty() {
        for part in &corrupt {
            journal.segments.remove(part);
        }
        write_json(journal_path, journal)?;
        return Err(InferError::Config(format!(
            "table n-gram: segments {corrupt:?} différents du journal ; invalidés, relancer la conversion"
        )));
    }
    let ngram_size = segments.iter().map(|s| s.len).sum::<u64>();
    files.push(OutputFile {
        name: NGRAM_FILE.to_string(),
        size: ngram_size,
        sha256: ngram_sha.clone(),
    });
    let scale_bytes = source.read(&inventory.ngram_scale, 0, inventory.ngram_scale.len)?;
    let scale_bits = u16::from_le_bytes([scale_bytes[0], scale_bytes[1]]);
    let width = inventory.ngram[0].shape[1] as u64;
    let meta = NgramMeta {
        file: "ngram_e4m3.bin",
        dtype: "F8_E4M3FN",
        shape: [hash_params.padded_rows, width],
        formula: "w[r, c] = f32(e4m3[r * width + c]) * weight_scale (échelle per-tensor, appliquée une seule fois)",
        weight_scale_bf16_bits: format!("0x{scale_bits:04x}"),
        weight_scale: f32::from_bits(u32::from(scale_bits) << 16),
        source_scale_tensor: &inventory.ngram_scale.name,
        hash: hash_params,
        segments,
        sha256: ngram_sha,
    };
    write_json(&dst.join(NGRAM_META), &meta)?;

    let mut index = serde_json::Map::new();
    let main_bytes: u64 = files
        .iter()
        .filter(|f| f.name.starts_with("model-"))
        .map(|f| f.size)
        .sum();
    index.insert(
        "metadata".to_string(),
        serde_json::json!({ "total_size": main_bytes }),
    );
    index.insert("weight_map".to_string(), weight_map.into());
    write_json(&dst.join("model.safetensors.index.json"), &index)?;

    for (name, required) in SIDE_FILES {
        let from = options.src.join(name);
        if !from.exists() {
            if *required {
                return Err(InferError::MissingArtifact {
                    path: from,
                    what: "fichier annexe",
                });
            }
            continue;
        }
        let bytes = std::fs::read(&from).map_err(|e| io_err(&from, e))?;
        write_atomic(&dst.join(name), &bytes)?;
        files.push(OutputFile {
            name: (*name).to_string(),
            size: bytes.len() as u64,
            sha256: sha256_bytes(&bytes),
        });
    }

    let owned: Vec<TensorRecord> = records.iter().map(|r| (*r).clone()).collect();
    let config = output_config(src_config, &owned)?;
    let tokenizer_sha = std::fs::read(options.src.join("tokenizer.json"))
        .map(|bytes| sha256_bytes(&bytes))
        .map_err(|e| io_err(&options.src, e))?;
    let manifest = Manifest {
        format: "saragossa-convert-qwen4exp/1",
        source: serde_json::json!({
            "path": options.src,
            "repo": "Qwen/Qwen3.8-Flash-Next-FP8",
            "revision": options.source_revision,
            "index_sha256": identity.source_index_sha256,
            "config_sha256": identity.source_config_sha256,
            "tokenizer_sha256": tokenizer_sha,
        }),
        identity,
        conventions: serde_json::json!({
            "trunk_prefix": rules::OUT_TRUNK_PREFIX,
            "lm_head": rules::OUT_LM_HEAD,
            "experts": "switch_mlp.{gate,up,down}_proj empilés [E, rows, packed] ; scales/biases [E, rows, groups]",
            "mtp": "sidecar mtp.safetensors, clés mtp.*, experts mtp.layers.0.mlp.switch_mlp.* u4 ; fc_embedding/fc_hidden séparés u8",
            "affine": "w = q·scale + bias, codes LSB-first en flux de bits par ligne (u4/u8 = lanes, u6 chevauche les u32)",
            "norms": "gains copiés à l'octet ; convention de lecture 1 + w (Qwen4ExpTextRMSNorm), aucun ajout",
            "ple_layer": rules::PLE_LAYER,
            "ngram": "fichier ngram/ngram_e4m3.bin + ngram/meta.json",
            "text_only": true,
        }),
        counts: &inventory.counts,
        excluded_vision: &inventory.vision,
        consumed_expert_scales: inventory.counts.expert_scales,
        files: &files,
        ngram: serde_json::to_value(&meta).map_err(|e| json_err(&dst.join(NGRAM_META), e))?,
        tensors: records,
    };
    write_json(&dst.join(MANIFEST), &manifest)?;
    // La config en dernier : sans elle, la sortie n'est pas chargeable.
    write_json(&dst.join("config.json"), &config)?;
    Ok(files)
}
