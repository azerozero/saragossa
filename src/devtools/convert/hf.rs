//! `saragossa convert` : checkpoint HF BF16 officiel → affine u4/u8 gs64, en flux.
//!
//! Contrat des conversions MLX (`mlx_lm.convert -q`) : toute matrice BF16
//! `.weight` de rang 2 dont la largeur est un multiple de 64 est quantifiée
//! (embeddings et `lm_head` compris), hors tour vision ; normes, `A_log`,
//! `dt_bias` et `conv1d` restent bf16. Le routeur MoE et `shared_expert_gate`
//! passent en 8 bits comme le `quant_predicate` de mlx-lm. Contrairement aux
//! conversions mlx-community, la tête `mtp.*` est conservée et quantifiée.
//!
//! Les noms et layouts HF bruts sont conservés (le loader applique lui-même
//! l'offset `1 + w` des normes et accepte le conv1d `[C, 1, K]`) : un shard de
//! sortie par shard source, mêmes noms de fichiers, donc la tête MTP reste dans
//! un seul shard et le loader la trouve sans sidecar.
//!
//! MoE (Qwen3.5/3.6-A3B) : les experts fusionnés `experts.gate_up_proj`
//! `[E, 2I, H]` et `experts.down_proj` `[E, H, I]` deviennent les piles
//! `switch_mlp.{gate,up,down}_proj` du loader (gate = lignes `0..I` de chaque
//! expert, up = `I..2I`), comme le `sanitize` de mlx-lm. Une tête MTP MoE est
//! recopiée en bf16 : le loader la quantifie lui-même à la charge (fc et
//! attention u8, experts u4), seul format qu'il accepte pour elle. Avec
//! `--keep-bf16`, les piles d'experts restent en bf16 : `--keep-bf16 .` donne
//! ainsi une référence BF16 que le loader charge (KL). Un tenseur est lu par
//! blocs bornés (~64 Mo), quantifié en parallèle par lignes et écrit à son
//! offset : la mémoire ne dépend pas de la taille du modèle. Sortie
//! déterministe, reprise par shard (journal), `--verify` rehache les shards.

use std::collections::BTreeMap;
use std::fs::OpenOptions;
use std::path::{Path, PathBuf};
use std::time::Instant;

use rayon::prelude::*;
use safetensors::Dtype;
use serde::{Deserialize, Serialize};

use super::imatrix::Imatrix;
use super::plan::{
    affine_outputs, dtype_name, file_sha256, io_err, shard_size, strip_weight, write_shard,
    AffineFit, Job, OutTensor, PeakTracker, ShardPlan, Source, TensorRef,
};
use super::rules::GROUP_SIZE;
use super::{
    check_done, check_shard, json_err, read_json, refuse_in_place, resolved, sha256_bytes,
    sync_dir, write_atomic, write_json, DestLock, OutputFile, ShardRecord, JOURNAL,
};
use crate::assets::is_mtp_key;
use crate::{InferError, Result};

const FORMAT: &str = "saragossa-convert-hf/1";

fn is_zero(value: &u32) -> bool {
    *value == 0
}
const MANIFEST: &str = "convert_manifest.json";
/// Fichiers annexes recopiés à l'octet ; seul `tokenizer.json` est obligatoire
/// (jamais régénéré : voir `tools/quant/tokenizer_sidecar.py`).
const SIDE_FILES: &[(&str, bool)] = &[
    ("tokenizer.json", true),
    ("tokenizer_config.json", false),
    ("vocab.json", false),
    ("merges.txt", false),
    ("special_tokens_map.json", false),
    ("added_tokens.json", false),
    ("chat_template.jinja", false),
    ("chat_template.json", false),
    ("generation_config.json", false),
    ("preprocessor_config.json", false),
    ("video_preprocessor_config.json", false),
    ("processor_config.json", false),
    ("LICENSE", false),
];

/// Options de `saragossa convert`.
#[derive(Clone, Debug)]
pub struct HfConvertOptions {
    /// Checkpoint HF BF16 (`config.json`, index, shards).
    pub src: PathBuf,
    /// Dossier de sortie (reprise si un journal compatible existe).
    pub dst: PathBuf,
    /// Largeur par défaut : 4 ou 8.
    pub bits: usize,
    /// Min-max bf16 seul au lieu des moindres carrés.
    pub minmax: bool,
    /// Sous-chaînes de noms de tenseurs gardés en 8 bits.
    pub keep_8bit: Vec<String>,
    /// Sous-chaînes de noms de tenseurs recopiés en bf16.
    pub keep_bf16: Vec<String>,
    /// Rehache les shards déjà terminés au lieu de contrôler leur taille.
    pub verify: bool,
    /// Imatrix (`tools/quant/imatrix.py`) : moindres carrés pondérés par `E[x²]`.
    pub imatrix: Option<PathBuf>,
    /// `(sous-chaîne, largeur)` : première correspondance, avant `keep_8bit`
    /// (2, 3, 4, 5, 6 ou 8 bits ; la tête MTP reste u4).
    pub bits_for: Vec<(String, usize)>,
    /// Rapport JSON des colonnes d'entrée qui concentrent l'erreur (voir
    /// `audit`), pour les shards écrits par cette exécution.
    pub audit_report: Option<PathBuf>,
}

/// Bilan de la conversion.
#[derive(Clone, Debug, Serialize)]
pub struct HfConvertReport {
    /// Tenseurs quantifiés par largeur.
    pub quantized: BTreeMap<usize, usize>,
    /// Tenseurs recopiés tels quels.
    pub copied: usize,
    pub units_written: Vec<String>,
    pub units_resumed: Vec<String>,
    pub elapsed_s: f64,
    /// Pic d'empreinte physique échantillonné (`phys_footprint`).
    pub peak_footprint_bytes: u64,
    pub output_bytes: u64,
    pub files: Vec<OutputFile>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
struct Identity {
    format: String,
    source_index_sha256: String,
    source_config_sha256: String,
    bits: usize,
    fit: AffineFit,
    keep_8bit: Vec<String>,
    keep_bf16: Vec<String>,
    // Absents des journaux antérieurs et omis quand ils sont vides : une
    // conversion sans ces options garde un manifeste identique à l'octet.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    imatrix_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    bits_for: Vec<(String, usize)>,
    // Révision des moindres carrés (1 : grille à zéro, importance fictive des
    // petites colonnes réservée aux conversions sans imatrix) : un journal
    // d'une autre révision (champ absent = 0) n'est pas repris.
    #[serde(default, skip_serializing_if = "is_zero")]
    ls_revision: u32,
}

/// Empreinte complète d'un shard source : une reprise sur d'autres poids de
/// mêmes noms et formes est refusée, même à date de modification conservée.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
struct SourceShard {
    file: String,
    size: u64,
    sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Journal {
    identity: Identity,
    sources: Vec<SourceShard>,
    shards: BTreeMap<String, ShardRecord>,
}

/// sha256 de chaque shard source, à chaque lancement : une lecture séquentielle
/// de plus (le shard est lu par blocs pendant la conversion de toute façon).
fn source_shards(source: &Source) -> Result<Vec<SourceShard>> {
    source
        .headers
        .par_iter()
        .map(|header| {
            let path = &header.path;
            let meta = std::fs::metadata(path).map_err(|e| io_err(path, e))?;
            Ok(SourceShard {
                file: path
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default(),
                size: meta.len(),
                sha256: file_sha256(path)?,
            })
        })
        .collect()
}

fn is_vision(name: &str) -> bool {
    name.starts_with("model.visual.")
        || name.starts_with("visual.")
        || name.contains("vision_tower")
}

fn is_router(name: &str) -> bool {
    name.ends_with(".mlp.gate.weight") || name.ends_with(".shared_expert_gate.weight")
}

/// Largeur de quantification d'un tenseur source, `None` pour une copie.
fn width(tensor: &TensorRef, options: &HfConvertOptions) -> Result<Option<usize>> {
    let name = tensor.name.as_str();
    // Experts fusionnés traités par `fused_experts` ; experts un par un
    // (format Qwen3-MoE) non pris en charge.
    if name.contains(".mlp.experts.") {
        return Err(InferError::Config(format!(
            "experts MoE hors format fusionné `gate_up_proj`/`down_proj` : {name}"
        )));
    }
    let matches = |list: &[String]| list.iter().any(|pattern| name.contains(pattern.as_str()));
    let quantizable = tensor.dtype == Dtype::BF16
        && name.ends_with(".weight")
        && matches!(tensor.shape[..], [rows, cols] if rows > 0 && cols % GROUP_SIZE == 0 && cols > 0)
        && !is_vision(name);
    if !quantizable || matches(&options.keep_bf16) {
        return Ok(None);
    }
    // Le loader MTP (`decoder/mtp.rs`, `load_mtp_linear`) n'accepte que u4 ou
    // bf16 : la tête reste en u4 quelle que soit `--bits`.
    if is_mtp_key(name) {
        if matches(&options.keep_8bit) {
            return Err(InferError::Config(format!(
                "tête MTP : u4 ou bf16 seulement (--keep-bf16), pas 8 bits : {name}"
            )));
        }
        return Ok(Some(4));
    }
    if let Some((_, bits)) = options
        .bits_for
        .iter()
        .find(|(pattern, _)| name.contains(pattern.as_str()))
    {
        return Ok(Some(*bits));
    }
    Ok(Some(if matches(&options.keep_8bit) || is_router(name) {
        8
    } else {
        options.bits
    }))
}

/// Experts fusionnés du tronc → segments `(base switch_mlp, lignes)`.
fn fused_experts(tensor: &TensorRef) -> Result<Option<Vec<(String, usize)>>> {
    let name = tensor.name.as_str();
    let (prefix, gate_up) = if let Some(prefix) = name.strip_suffix(".mlp.experts.gate_up_proj") {
        (prefix, true)
    } else if let Some(prefix) = name.strip_suffix(".mlp.experts.down_proj") {
        (prefix, false)
    } else {
        return Ok(None);
    };
    let valid = tensor.dtype == Dtype::BF16
        && matches!(tensor.shape[..], [experts, rows, cols]
            if experts > 0 && rows > 0 && cols > 0 && cols % GROUP_SIZE == 0
                && (!gate_up || rows % 2 == 0));
    let &[_, rows, _] = &tensor.shape[..] else {
        return Err(InferError::Dimension(name.to_string()));
    };
    if !valid {
        return Err(InferError::Config(format!(
            "experts fusionnés {name}: BF16 [E, R, C] avec C multiple de {GROUP_SIZE} attendu, \
             reçu {:?} {:?}",
            tensor.dtype, tensor.shape
        )));
    }
    let base = |proj: &str| format!("{prefix}.mlp.switch_mlp.{proj}");
    Ok(Some(if gate_up {
        vec![(base("gate_proj"), rows / 2), (base("up_proj"), rows / 2)]
    } else {
        vec![(base("down_proj"), rows)]
    }))
}

/// Shard de sortie qui reçoit toute la tête `mtp.*` : celui de `mtp.fc.weight`,
/// sinon le premier qui en porte. Le loader ne lit une tête embarquée que dans
/// un seul shard (`assets::embedded_mtp_weights`).
fn mtp_shard(refs: &[TensorRef]) -> Option<usize> {
    let mtp = || refs.iter().filter(|t| is_mtp_key(&t.name));
    mtp()
        .find(|t| t.name.ends_with("mtp.fc.weight"))
        .or_else(|| mtp().min_by_key(|t| t.shard))
        .map(|t| t.shard)
}

/// Un shard de sortie par shard source (tête MTP regroupée), tenseurs triés
/// par nom.
fn plan(
    source: &Source,
    refs: &[TensorRef],
    options: &HfConvertOptions,
    imatrix: Option<&Imatrix>,
) -> Result<Vec<ShardPlan>> {
    let fit = if options.minmax {
        AffineFit::MinMax
    } else {
        AffineFit::LeastSquares
    };
    let mut plans = source
        .headers
        .iter()
        .map(|header| {
            let file = header
                .path
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or_else(|| InferError::Config(format!("shard sans nom: {:?}", header.path)))?;
            Ok(ShardPlan {
                file: file.to_string(),
                jobs: Vec::new(),
                tensors: Vec::new(),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let mtp = mtp_shard(refs);
    let moe_mtp = refs
        .iter()
        .any(|t| is_mtp_key(&t.name) && t.name.ends_with(".mlp.experts.gate_up_proj"));
    for tensor in refs {
        let is_mtp = is_mtp_key(&tensor.name);
        let shard = match mtp {
            Some(shard) if is_mtp => shard,
            _ => tensor.shard,
        };
        let plan = &mut plans[shard];
        let fused = if is_mtp { None } else { fused_experts(tensor)? };
        if let Some(outs) = fused {
            let [experts, _, cols] = tensor.shape[..] else {
                return Err(InferError::Dimension(tensor.name.clone()));
            };
            let keep = |list: &[String]| {
                list.iter()
                    .any(|pattern| tensor.name.contains(pattern.as_str()))
            };
            // 16 : piles recopiées en bf16 (référence BF16 chargeable).
            let bits = if keep(&options.keep_bf16) {
                16
            } else if keep(&options.keep_8bit) {
                8
            } else {
                options.bits
            };
            for (base, rows) in &outs {
                if bits == 16 {
                    plan.tensors.push(OutTensor {
                        name: format!("{base}.weight"),
                        dtype: "BF16".to_string(),
                        shape: vec![experts, *rows, cols],
                        nbytes: (experts * rows * cols * 2) as u64,
                    });
                } else {
                    plan.tensors
                        .extend(affine_outputs(base, &[experts], *rows, cols, bits));
                }
            }
            plan.jobs.push(Job::Stacked {
                src: tensor.clone(),
                outs,
                bits,
                fit,
            });
            continue;
        }
        let width = if is_mtp && moe_mtp {
            None
        } else {
            width(tensor, options)?
        };
        match width {
            Some(bits) => {
                let [rows, cols] = tensor.shape[..] else {
                    return Err(InferError::Dimension(tensor.name.clone()));
                };
                let base = strip_weight(&tensor.name)?.to_string();
                plan.tensors
                    .extend(affine_outputs(&base, &[], rows, cols, bits));
                let weight = match imatrix {
                    Some(matrix) => matrix.for_tensor(&tensor.name, cols)?,
                    None => None,
                };
                plan.jobs.push(Job::Affine {
                    src: tensor.clone(),
                    out_base: base,
                    bits,
                    fit,
                    weight,
                    proxy: imatrix.is_none() && fit == AffineFit::LeastSquares,
                });
            }
            None => {
                plan.tensors.push(OutTensor {
                    name: tensor.name.clone(),
                    dtype: dtype_name(tensor.dtype),
                    shape: tensor.shape.clone(),
                    nbytes: tensor.len as u64,
                });
                plan.jobs.push(Job::Copy {
                    src: tensor.clone(),
                    out: tensor.name.clone(),
                });
            }
        }
    }
    Ok(plans)
}

/// Config source + `quantization` (et son alias) avec les exceptions par
/// tenseur : `{bits: 8}` pour les largeurs hautes, `false` pour les bf16
/// gardés (convention MLX).
fn output_config(
    src_config: &serde_json::Value,
    plans: &[ShardPlan],
    options: &HfConvertOptions,
) -> Result<serde_json::Value> {
    let mut quant = serde_json::Map::new();
    quant.insert("group_size".into(), GROUP_SIZE.into());
    quant.insert("bits".into(), options.bits.into());
    quant.insert("mode".into(), "affine".into());
    for job in plans.iter().flat_map(|plan| &plan.jobs) {
        match job {
            Job::Affine { out_base, bits, .. } if *bits != options.bits => {
                quant.insert(
                    out_base.clone(),
                    serde_json::json!({ "group_size": GROUP_SIZE, "bits": bits }),
                );
            }
            Job::Stacked { outs, bits, .. } if *bits != options.bits => {
                for (base, _) in outs {
                    let entry = if *bits == 16 {
                        false.into()
                    } else {
                        serde_json::json!({ "group_size": GROUP_SIZE, "bits": bits })
                    };
                    quant.insert(base.clone(), entry);
                }
            }
            Job::Copy { src, out } if src.shape.len() == 2 && !is_vision(out) => {
                if let Some(base) = out.strip_suffix(".weight") {
                    if options
                        .keep_bf16
                        .iter()
                        .any(|pattern| out.contains(pattern.as_str()))
                    {
                        quant.insert(base.to_string(), false.into());
                    }
                }
            }
            _ => {}
        }
    }
    let mut config = src_config.clone();
    let map = config
        .as_object_mut()
        .ok_or_else(|| InferError::Config("config.json n'est pas un objet".to_string()))?;
    map.insert("quantization".into(), quant.clone().into());
    map.insert("quantization_config".into(), quant.into());
    Ok(config)
}

/// Lance (ou reprend) la conversion.
///
/// # Errors
///
/// Renvoie une erreur si la source est incohérente ou contient des experts MoE
/// hors format fusionné,
/// si la largeur n'est ni 4 ni 8, si l'identité diffère d'une reprise, si la
/// destination est verrouillée, ou sur toute erreur de lecture, de
/// quantification ou d'écriture.
pub fn convert_hf(options: &HfConvertOptions) -> Result<HfConvertReport> {
    let started = Instant::now();
    let peak = PeakTracker::default();
    if !matches!(options.bits, 4 | 8) {
        return Err(InferError::Config(format!(
            "--bits {} : 4 ou 8 attendu",
            options.bits
        )));
    }
    if let Some((pattern, bits)) = options
        .bits_for
        .iter()
        .find(|(_, bits)| !matches!(bits, 2 | 3 | 4 | 5 | 6 | 8))
    {
        return Err(InferError::Config(format!(
            "--bits-for {pattern}={bits} : 2, 3, 4, 5, 6 ou 8 attendu"
        )));
    }
    if options.minmax && options.imatrix.is_some() {
        return Err(InferError::Config(
            "--imatrix pondère les moindres carrés : incompatible avec --minmax".to_string(),
        ));
    }
    refuse_in_place(&options.src, &options.dst)?;
    if let Some(report) = &options.audit_report {
        // Le rapport et son temporaire `*.tmp-write` remplacent leur cible :
        // dans la source ou la sortie, ils écraseraient un shard, l'index, le
        // tokenizer ou le journal. Refusé avant toute conversion. Chaque chemin
        // est testé résolu (cible d'un lien, que `File::create` suit) et comme
        // entrée de son dossier résolu (que `rename` remplace).
        let roots = [resolved(&options.src)?, resolved(&options.dst)?];
        let mut spots = Vec::new();
        for path in [report.clone(), report.with_extension("tmp-write")] {
            spots.push(resolved(&path)?);
            if let (Some(parent), Some(name)) = (path.parent(), path.file_name()) {
                let parent = if parent.as_os_str().is_empty() {
                    Path::new(".")
                } else {
                    parent
                };
                spots.push(resolved(parent)?.join(name));
            }
        }
        if spots
            .iter()
            .any(|spot| roots.iter().any(|root| spot.starts_with(root)))
        {
            return Err(InferError::Config(format!(
                "--audit-report {} : hors des dossiers source et destination attendu",
                report.display()
            )));
        }
    }
    let imatrix = options.imatrix.as_deref().map(Imatrix::load).transpose()?;
    let config_path = options.src.join("config.json");
    let src_config = read_json(&config_path)?;
    let (source, refs) = Source::open(&options.src)?;
    let plans = plan(&source, &refs, options, imatrix.as_ref())?;
    let read = |path: &Path| std::fs::read(path).map_err(|e| io_err(path, e));
    let identity = Identity {
        format: FORMAT.to_string(),
        source_index_sha256: sha256_bytes(&source.index),
        source_config_sha256: sha256_bytes(&read(&config_path)?),
        bits: options.bits,
        fit: if options.minmax {
            AffineFit::MinMax
        } else {
            AffineFit::LeastSquares
        },
        keep_8bit: options.keep_8bit.clone(),
        keep_bf16: options.keep_bf16.clone(),
        imatrix_sha256: imatrix.as_ref().map(|m| m.sha256.clone()),
        bits_for: options.bits_for.clone(),
        ls_revision: u32::from(!options.minmax),
    };

    let dst = &options.dst;
    let convert_dir = dst.join(".convert");
    std::fs::create_dir_all(&convert_dir).map_err(|e| io_err(&convert_dir, e))?;
    let _lock = DestLock::acquire(dst)?;
    let journal_path = dst.join(JOURNAL);
    let sources = source_shards(&source)?;
    let mut journal = if journal_path.exists() {
        let journal: Journal = serde_json::from_value(read_json(&journal_path)?)
            .map_err(|e| json_err(&journal_path, e))?;
        if journal.identity != identity {
            return Err(InferError::Config(format!(
                "reprise refusée: identité différente (journal {:?}, travail {:?})",
                journal.identity, identity
            )));
        }
        if journal.sources != sources {
            return Err(InferError::Config(
                "reprise refusée: shards source modifiés (taille ou sha256) depuis le journal"
                    .to_string(),
            ));
        }
        journal
    } else {
        Journal {
            identity: identity.clone(),
            sources: sources.clone(),
            shards: BTreeMap::new(),
        }
    };

    let mut written = Vec::new();
    let mut resumed = Vec::new();
    let mut audits = Vec::new();
    for plan in &plans {
        let path = dst.join(&plan.file);
        let size = shard_size(plan)?;
        if let Some(done) = journal.shards.get(&plan.file) {
            if done.size == size && check_done(&path, size, &done.sha256, options.verify)? {
                resumed.push(plan.file.clone());
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
        let audit = options.audit_report.is_some().then_some(&mut audits);
        let tensors = write_shard(&source, &BTreeMap::new(), plan, &file, &tmp, &peak, audit)?;
        file.sync_all().map_err(|e| io_err(&tmp, e))?;
        drop(file);
        std::fs::rename(&tmp, &path).map_err(|e| io_err(&path, e))?;
        sync_dir(dst)?;
        let sha256 = file_sha256(&path)?;
        journal.shards.insert(
            plan.file.clone(),
            ShardRecord {
                file: plan.file.clone(),
                size,
                sha256,
                tensors,
            },
        );
        write_json(&journal_path, &journal)?;
        written.push(plan.file.clone());
        peak.sample();
    }

    let mut files = Vec::new();
    let mut weight_map = serde_json::Map::new();
    for plan in &plans {
        let done = journal
            .shards
            .get(&plan.file)
            .ok_or_else(|| InferError::Config(format!("shard {} non terminé", plan.file)))?;
        let path = dst.join(&plan.file);
        let sha256 = match check_shard(&path, plan, &done.sha256) {
            Ok(sha256) => sha256,
            Err(error) => {
                journal.shards.remove(&plan.file);
                write_json(&journal_path, &journal)?;
                return Err(InferError::Config(format!(
                    "{}: {error} ; shard invalidé, relancer la conversion",
                    plan.file
                )));
            }
        };
        for tensor in &plan.tensors {
            weight_map.insert(tensor.name.clone(), plan.file.clone().into());
        }
        files.push(OutputFile {
            name: plan.file.clone(),
            size: done.size,
            sha256,
        });
    }
    let total: u64 = files.iter().map(|f| f.size).sum();
    write_json(
        &dst.join("model.safetensors.index.json"),
        &serde_json::json!({ "metadata": { "total_size": total }, "weight_map": weight_map }),
    )?;
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
        let bytes = read(&from)?;
        write_atomic(&dst.join(name), &bytes)?;
        files.push(OutputFile {
            name: (*name).to_string(),
            size: bytes.len() as u64,
            sha256: sha256_bytes(&bytes),
        });
    }
    let records: Vec<_> = journal.shards.values().flat_map(|s| &s.tensors).collect();
    write_json(
        &dst.join(MANIFEST),
        &serde_json::json!({
            "format": FORMAT,
            "identity": identity,
            "sources": sources,
            "files": files,
            "tensors": records,
        }),
    )?;
    if let Some(path) = &options.audit_report {
        write_json(
            path,
            &serde_json::json!({
                "hot_share": super::audit::HOT_SHARE,
                "small_ratio": super::audit::SMALL_RATIO,
                "weighted": imatrix.is_some(),
                "units_resumed": &resumed,
                "tensors": audits,
            }),
        )?;
    }
    // La config en dernier : sans elle, la sortie n'est pas chargeable.
    write_json(
        &dst.join("config.json"),
        &output_config(&src_config, &plans, options)?,
    )?;

    let mut quantized = BTreeMap::new();
    let mut copied = 0;
    for job in plans.iter().flat_map(|plan| &plan.jobs) {
        match job {
            Job::Stacked { bits: 16, .. } => copied += 1,
            Job::Affine { bits, .. } | Job::Stacked { bits, .. } => {
                *quantized.entry(*bits).or_insert(0) += 1;
            }
            _ => copied += 1,
        }
    }
    peak.sample();
    Ok(HfConvertReport {
        quantized,
        copied,
        units_written: written,
        units_resumed: resumed,
        elapsed_s: started.elapsed().as_secs_f64(),
        peak_footprint_bytes: peak.peak(),
        output_bytes: files.iter().map(|f| f.size).sum(),
        files,
    })
}
