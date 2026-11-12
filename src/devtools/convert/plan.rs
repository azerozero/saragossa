//! Inventaire de la source FP8, plan des shards de sortie et écriture en flux.
//!
//! La source est lue par `pread` borné (aucun mmap) : la mémoire résidente ne
//! dépend que des blocs en cours, jamais des 186 Go de la source.

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::path::{Path, PathBuf};

use rayon::prelude::*;
use safetensors::Dtype;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::rules::{
    classify, expected_source_names, output_name, ExpertGroup, LayerType, Placement, Proj, Role,
    EXPERT_BITS, GROUP_SIZE,
};
use crate::quantization::{quantize_affine_bitstream_bf16, QuantError};
use crate::qwen_loader::{read_shard_header, ShardHeader};
use crate::safetensor::{
    apply_fp8_block_scales_in_place, bytes_to_dense_f32, decode_f8_e4m3_into, FP8_SCALE_BLOCK,
};
use crate::safetensor::{pread_exact, pwrite_all};
use crate::{InferError, Result};

#[cfg(not(target_endian = "little"))]
compile_error!("le convertisseur écrit les U32/BF16 safetensors en little-endian natif");

/// Experts quantifiés en parallèle avant écriture séquentielle (~300 Mo en vol).
const EXPERT_CHUNK: usize = 32;
/// Éléments f32 par bloc de lignes d'une matrice dense (~64 Mo).
const DENSE_BLOCK_ELEMENTS: usize = 16 * 1024 * 1024;
/// Lignes par tâche rayon dans un bloc dense.
const ROWS_PER_TASK: usize = 64;
/// Taille des morceaux de copie brute et de hachage.
pub(crate) const IO_CHUNK: usize = 64 * 1024 * 1024;
/// Nombre de shards du tronc : 16 de trois couches + 1 de tenseurs globaux.
const LAYERS_PER_SHARD: usize = 3;

pub(crate) fn io_err(path: &Path, source: std::io::Error) -> InferError {
    InferError::Io {
        path: path.to_path_buf(),
        source,
    }
}

pub(crate) fn hex(digest: impl AsRef<[u8]>) -> String {
    digest
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn dtype_name(dtype: Dtype) -> String {
    format!("{dtype:?}")
}

fn element_count(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// Tenseur source localisé.
#[derive(Clone, Debug)]
pub(crate) struct TensorRef {
    pub(crate) name: String,
    pub(crate) shard: usize,
    pub(crate) offset: u64,
    pub(crate) len: usize,
    pub(crate) dtype: Dtype,
    pub(crate) shape: Vec<usize>,
}

/// Checkpoint source : en-têtes et descripteurs ouverts une fois.
pub(crate) struct Source {
    pub(crate) dir: PathBuf,
    pub(crate) headers: Vec<ShardHeader>,
    files: Vec<File>,
}

impl Source {
    /// Ouvre la source et contrôle la cohérence index ↔ en-têtes.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si l'index, un en-tête ou un shard manque, ou si un
    /// tenseur est absent, dupliqué ou affecté à un autre shard que l'index.
    pub(crate) fn open(dir: &Path) -> Result<(Self, Vec<TensorRef>)> {
        let index_path = dir.join("model.safetensors.index.json");
        let raw = std::fs::read(&index_path).map_err(|source| io_err(&index_path, source))?;
        let index: serde_json::Value =
            serde_json::from_slice(&raw).map_err(|source| InferError::Json {
                path: index_path.clone(),
                source,
            })?;
        let weight_map = index
            .get("weight_map")
            .and_then(|value| value.as_object())
            .ok_or_else(|| InferError::Config("index sans weight_map".to_string()))?;
        let mut shard_names: Vec<String> = weight_map
            .values()
            .filter_map(|value| value.as_str().map(str::to_string))
            .collect();
        shard_names.sort();
        shard_names.dedup();
        let mut headers = Vec::with_capacity(shard_names.len());
        let mut files = Vec::with_capacity(shard_names.len());
        for name in &shard_names {
            let path = dir.join(name);
            headers.push(read_shard_header(&path)?);
            files.push(File::open(&path).map_err(|source| io_err(&path, source))?);
        }
        let mut refs = Vec::with_capacity(weight_map.len());
        let mut seen = 0_usize;
        for (shard, header) in headers.iter().enumerate() {
            let file_len = files[shard]
                .metadata()
                .map_err(|source| io_err(&header.path, source))?
                .len();
            validate_layout(header, file_len)?;
            for (name, entry) in &header.tensors {
                let expected = weight_map.get(name).and_then(|value| value.as_str());
                if expected != Some(shard_names[shard].as_str()) {
                    return Err(InferError::Config(format!(
                        "index/en-tête incohérents pour {name}: index={expected:?}, shard={}",
                        shard_names[shard]
                    )));
                }
                let len = entry.data_offsets[1] - entry.data_offsets[0];
                refs.push(TensorRef {
                    name: name.clone(),
                    shard,
                    offset: header.data_start + entry.data_offsets[0] as u64,
                    len,
                    dtype: entry.dtype,
                    shape: entry.shape.clone(),
                });
                seen += 1;
            }
        }
        if seen != weight_map.len() {
            return Err(InferError::Config(format!(
                "index: {} entrées, en-têtes: {seen} tenseurs",
                weight_map.len()
            )));
        }
        refs.sort_by(|a, b| a.name.cmp(&b.name));
        Ok((
            Self {
                dir: dir.to_path_buf(),
                headers,
                files,
            },
            refs,
        ))
    }

    /// Lit `len` octets à partir de `start` dans les données du tenseur.
    pub(crate) fn read(&self, tensor: &TensorRef, start: usize, len: usize) -> Result<Vec<u8>> {
        if start + len > tensor.len {
            return Err(InferError::Shape(format!(
                "lecture hors tenseur {}: {start}+{len} > {}",
                tensor.name, tensor.len
            )));
        }
        let mut buf = vec![0_u8; len];
        pread_exact(
            &self.files[tensor.shard],
            &mut buf,
            tensor.offset + start as u64,
        )
        .map_err(|source| io_err(&self.headers[tensor.shard].path, source))?;
        Ok(buf)
    }
}

/// Poids et échelle d'un expert en cours d'inventaire.
type ExpertSlot = (Option<TensorRef>, Option<TensorRef>);

/// Plage `[début, fin)` de chaque tenseur dans les données d'un shard.
pub(crate) type Offsets = HashMap<String, (u64, u64)>;

/// Taille en octets d'un élément safetensors.
fn dtype_size(dtype: Dtype) -> Option<usize> {
    match dtype {
        Dtype::BOOL | Dtype::U8 | Dtype::I8 | Dtype::F8_E4M3 | Dtype::F8_E5M2 => Some(1),
        Dtype::F16 | Dtype::BF16 | Dtype::I16 | Dtype::U16 => Some(2),
        Dtype::F32 | Dtype::I32 | Dtype::U32 => Some(4),
        Dtype::F64 | Dtype::I64 | Dtype::U64 => Some(8),
        _ => None,
    }
}

/// Invariants physiques d'un shard source : longueur = produit(forme) ×
/// taille(dtype), aucun chevauchement, toutes les plages dans le fichier
/// (arithmétique vérifiée).
fn validate_layout(header: &ShardHeader, file_len: u64) -> Result<()> {
    let path = header.path.display();
    let mut ranges = Vec::with_capacity(header.tensors.len());
    for (name, entry) in &header.tensors {
        let [start, end] = entry.data_offsets;
        let size = dtype_size(entry.dtype).ok_or_else(|| InferError::UnsupportedDtype {
            name: name.clone(),
            dtype: entry.dtype,
        })?;
        let expected = entry
            .shape
            .iter()
            .try_fold(size, |acc, dim| acc.checked_mul(*dim))
            .ok_or_else(|| InferError::Shape(format!("{path}: {name}: forme trop grande")))?;
        let len = end.checked_sub(start).ok_or_else(|| {
            InferError::Shape(format!("{path}: {name}: plage inversée [{start}, {end})"))
        })?;
        if len != expected {
            return Err(InferError::Shape(format!(
                "{path}: {name}: {len} octets pour {:?} {:?} ({expected} attendus)",
                entry.dtype, entry.shape
            )));
        }
        let physical_end = header
            .data_start
            .checked_add(end as u64)
            .ok_or_else(|| InferError::Shape(format!("{path}: {name}: offset trop grand")))?;
        if physical_end > file_len {
            return Err(InferError::Shape(format!(
                "{path}: {name}: plage hors fichier ({physical_end} > {file_len})"
            )));
        }
        ranges.push((start, end, name));
    }
    ranges.sort();
    for pair in ranges.windows(2) {
        let ((_, end, a), (start, _, b)) = (pair[0], pair[1]);
        if start < end {
            return Err(InferError::Shape(format!(
                "{path}: {a} et {b} se chevauchent"
            )));
        }
    }
    Ok(())
}

/// Source d'un expert : poids FP8 et échelle par blocs.
#[derive(Clone, Debug)]
pub(crate) struct ExpertSrc {
    pub(crate) weight: TensorRef,
    pub(crate) scale: TensorRef,
}

/// Compteurs de la partition, publiés dans le manifeste.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Counts {
    pub experts: usize,
    pub expert_scales: usize,
    pub dense_u8: usize,
    pub lm_head_u6: usize,
    pub copies: usize,
    pub ngram_shards: usize,
    pub ngram_scale: usize,
    pub i64: usize,
    pub vision_excluded: usize,
}

/// Partition complète et validée de la source.
pub(crate) struct Inventory {
    pub(crate) experts: BTreeMap<(ExpertGroup, Proj), Vec<ExpertSrc>>,
    /// Tenseurs non experts conservés : rôle, source, nom et placement de sortie.
    pub(crate) others: Vec<(Role, TensorRef, String, Placement)>,
    pub(crate) ngram: Vec<TensorRef>,
    pub(crate) ngram_scale: TensorRef,
    pub(crate) vision: Vec<String>,
    pub(crate) counts: Counts,
    /// Couche PLE (indice 0-based), lue dans la config.
    pub(crate) ple_layer: usize,
}

/// Paramètres structurels attendus, lus dans la config source.
#[derive(Clone, Debug)]
pub(crate) struct Expected {
    pub(crate) layer_types: Vec<LayerType>,
    pub(crate) mtp_type: Option<LayerType>,
    /// Couche PLE, indice 0-based (`ple_layer_ids` est 1-indexé).
    pub(crate) ple_layer: usize,
    pub(crate) layers: usize,
    pub(crate) experts: usize,
    pub(crate) mtp_layers: usize,
    pub(crate) ngram_parts: usize,
}

fn require(cond: bool, message: impl FnOnce() -> String) -> Result<()> {
    if cond {
        Ok(())
    } else {
        Err(InferError::Config(message()))
    }
}

fn check_role_dtype(role: Role, tensor: &TensorRef) -> Result<()> {
    let name = &tensor.name;
    match role {
        Role::Affine { bits } => {
            require(tensor.dtype == Dtype::BF16, || {
                format!("{name}: dense attendu BF16, reçu {:?}", tensor.dtype)
            })?;
            let [rows, cols] = tensor.shape[..] else {
                return Err(InferError::Dimension(format!(
                    "{name}: dense attendu rang 2, reçu {:?}",
                    tensor.shape
                )));
            };
            require(rows > 0 && cols > 0, || {
                format!("{name}: matrice vide {:?}", tensor.shape)
            })?;
            require(cols % GROUP_SIZE == 0 && (cols * bits) % 32 == 0, || {
                format!("{name}: {cols} colonnes incompatibles avec u{bits} gs{GROUP_SIZE}")
            })
        }
        Role::Copy => require(
            matches!(tensor.dtype, Dtype::BF16 | Dtype::F32 | Dtype::F16),
            || format!("{name}: copie flottante attendue, reçu {:?}", tensor.dtype),
        ),
        Role::CopyI64 => require(tensor.dtype == Dtype::I64, || {
            format!("{name}: I64 attendu, reçu {:?}", tensor.dtype)
        }),
        Role::NgramScale => require(
            tensor.dtype == Dtype::BF16 && element_count(&tensor.shape) == 1,
            || format!("{name}: échelle n-gram BF16 scalaire attendue"),
        ),
        Role::NgramShard(_) => require(
            tensor.dtype == Dtype::F8_E4M3 && tensor.shape.len() == 2,
            || format!("{name}: shard n-gram F8_E4M3 rang 2 attendu"),
        ),
        Role::Expert { .. } => require(
            tensor.dtype == Dtype::F8_E4M3 && tensor.shape.len() == 2,
            || format!("{name}: expert F8_E4M3 rang 2 attendu"),
        ),
        Role::ExpertScale { .. } => require(
            tensor.dtype == Dtype::BF16 && tensor.shape.len() == 2,
            || format!("{name}: échelle d'expert BF16 rang 2 attendue"),
        ),
        Role::Vision => Ok(()),
    }
}

impl Inventory {
    /// Classe chaque tenseur source et vérifie la partition.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur sur tenseur inconnu, dtype inattendu, expert manquant
    /// ou en double, échelle orpheline ou mal formée, shard n-gram manquant, ou
    /// collision de noms de sortie.
    pub(crate) fn build(refs: &[TensorRef], expected: &Expected) -> Result<Self> {
        let mut experts: HashMap<(ExpertGroup, Proj), Vec<ExpertSlot>> = HashMap::new();
        let mut others = Vec::new();
        let mut ngram: BTreeMap<usize, TensorRef> = BTreeMap::new();
        let mut ngram_scale = None;
        let mut vision = Vec::new();
        let mut counts = Counts::default();
        for tensor in refs {
            let role = classify(&tensor.name, expected.ple_layer)?;
            check_role_dtype(role, tensor)?;
            match role {
                Role::Expert {
                    group,
                    expert,
                    proj,
                }
                | Role::ExpertScale {
                    group,
                    expert,
                    proj,
                } => {
                    let valid_group = match group {
                        ExpertGroup::Trunk(layer) => layer < expected.layers,
                        ExpertGroup::Mtp => expected.mtp_layers == 1,
                    };
                    require(valid_group && expert < expected.experts, || {
                        format!("{}: expert hors configuration", tensor.name)
                    })?;
                    let slots = experts
                        .entry((group, proj))
                        .or_insert_with(|| vec![(None, None); expected.experts]);
                    let slot = &mut slots[expert];
                    let target = if matches!(role, Role::Expert { .. }) {
                        counts.experts += 1;
                        &mut slot.0
                    } else {
                        counts.expert_scales += 1;
                        &mut slot.1
                    };
                    require(target.replace(tensor.clone()).is_none(), || {
                        format!("{}: doublon", tensor.name)
                    })?;
                }
                Role::NgramShard(part) => {
                    counts.ngram_shards += 1;
                    require(ngram.insert(part, tensor.clone()).is_none(), || {
                        format!("{}: doublon", tensor.name)
                    })?;
                }
                Role::NgramScale => {
                    counts.ngram_scale += 1;
                    ngram_scale = Some(tensor.clone());
                    let (out, placement) = output_name(&tensor.name)?;
                    others.push((role, tensor.clone(), out, placement));
                }
                Role::Vision => {
                    counts.vision_excluded += 1;
                    vision.push(tensor.name.clone());
                }
                Role::Affine { bits } => {
                    if bits == super::rules::LM_HEAD_BITS {
                        counts.lm_head_u6 += 1;
                    } else {
                        counts.dense_u8 += 1;
                    }
                    let (out, placement) = output_name(&tensor.name)?;
                    others.push((role, tensor.clone(), out, placement));
                }
                Role::Copy | Role::CopyI64 => {
                    if role == Role::Copy {
                        counts.copies += 1;
                    } else {
                        counts.i64 += 1;
                    }
                    let (out, placement) = output_name(&tensor.name)?;
                    others.push((role, tensor.clone(), out, placement));
                }
            }
        }
        let mut stacks = BTreeMap::new();
        let mut groups: Vec<ExpertGroup> = (0..expected.layers).map(ExpertGroup::Trunk).collect();
        if expected.mtp_layers == 1 {
            groups.push(ExpertGroup::Mtp);
        }
        for group in groups {
            for proj in Proj::ALL {
                let slots = experts.remove(&(group, proj)).ok_or_else(|| {
                    InferError::MissingWeight(format!("experts {group:?} {}", proj.name()))
                })?;
                let mut stack = Vec::with_capacity(slots.len());
                for (expert, (weight, scale)) in slots.into_iter().enumerate() {
                    let (Some(weight), Some(scale)) = (weight, scale) else {
                        return Err(InferError::MissingWeight(format!(
                            "expert {expert} {group:?} {}: poids ou échelle absent",
                            proj.name()
                        )));
                    };
                    let [rows, cols] = weight.shape[..] else {
                        return Err(InferError::Dimension(weight.name.clone()));
                    };
                    let scale_shape = [
                        rows.div_ceil(FP8_SCALE_BLOCK),
                        cols.div_ceil(FP8_SCALE_BLOCK),
                    ];
                    require(scale.shape == scale_shape, || {
                        format!(
                            "{}: échelle {:?}, attendu {scale_shape:?}",
                            scale.name, scale.shape
                        )
                    })?;
                    require(rows > 0 && cols > 0 && cols % GROUP_SIZE == 0, || {
                        format!(
                            "{}: {rows}×{cols} incompatible avec gs{GROUP_SIZE}",
                            weight.name
                        )
                    })?;
                    if let Some(first) = stack.first() {
                        let first: &ExpertSrc = first;
                        require(first.weight.shape == weight.shape, || {
                            format!("{}: forme différente du premier expert", weight.name)
                        })?;
                    }
                    stack.push(ExpertSrc { weight, scale });
                }
                stacks.insert((group, proj), stack);
            }
        }
        require(experts.is_empty(), || {
            "groupes d'experts inattendus".to_string()
        })?;
        let ngram_scale = ngram_scale
            .ok_or_else(|| InferError::MissingWeight("ngram_embedding.weight_scale".to_string()))?;
        require(
            ngram.len() == expected.ngram_parts
                && ngram.keys().copied().eq(0..expected.ngram_parts),
            || {
                format!(
                    "shards n-gram: {} présents, 0..{} attendus",
                    ngram.len(),
                    expected.ngram_parts
                )
            },
        )?;
        let ngram: Vec<TensorRef> = ngram.into_values().collect();
        require(
            ngram.iter().all(|t| t.shape[1] == ngram[0].shape[1]),
            || "shards n-gram de largeurs différentes".to_string(),
        )?;
        let mut names: HashMap<String, String> = HashMap::new();
        for (_, tensor, out, _) in &others {
            if let Some(previous) = names.insert(out.clone(), tensor.name.clone()) {
                return Err(InferError::Config(format!(
                    "collision de sortie {out}: {previous} et {}",
                    tensor.name
                )));
            }
        }
        for (group, proj) in stacks.keys() {
            let base = group.output_base(*proj);
            require(!names.contains_key(&format!("{base}.weight")), || {
                format!("collision de sortie {base}")
            })?;
        }
        let mut present: Vec<&str> = others.iter().map(|(_, t, _, _)| t.name.as_str()).collect();
        present.sort_unstable();
        let wanted =
            expected_source_names(&expected.layer_types, expected.mtp_type, expected.ple_layer);
        if let Some(missing) = wanted
            .iter()
            .find(|name| present.binary_search(&name.as_str()).is_err())
        {
            return Err(InferError::MissingWeight(missing.clone()));
        }
        if let Some(extra) = present
            .iter()
            .find(|name| wanted.binary_search_by(|w| w.as_str().cmp(name)).is_err())
        {
            return Err(InferError::Config(format!(
                "tenseur inattendu pour ce type de couche: {extra}"
            )));
        }
        others.sort_by(|a, b| a.2.cmp(&b.2));
        vision.sort();
        Ok(Self {
            experts: stacks,
            others,
            ngram,
            ngram_scale,
            vision,
            counts,
            ple_layer: expected.ple_layer,
        })
    }
}

/// Tâche d'écriture d'un shard.
#[derive(Clone, Debug)]
pub(crate) enum Job {
    Affine {
        src: TensorRef,
        out_base: String,
        bits: usize,
    },
    Experts {
        group: ExpertGroup,
        proj: Proj,
        out_base: String,
    },
    Copy {
        src: TensorRef,
        out: String,
    },
}

/// Tenseur de sortie planifié.
#[derive(Clone, Debug)]
pub(crate) struct OutTensor {
    pub(crate) name: String,
    pub(crate) dtype: String,
    pub(crate) shape: Vec<usize>,
    pub(crate) nbytes: u64,
}

/// Shard de sortie : tâches dans l'ordre d'écriture et tenseurs dans l'ordre
/// des données.
#[derive(Clone, Debug)]
pub(crate) struct ShardPlan {
    pub(crate) file: String,
    pub(crate) jobs: Vec<Job>,
    pub(crate) tensors: Vec<OutTensor>,
}

fn affine_outputs(
    base: &str,
    lead: &[usize],
    rows: usize,
    cols: usize,
    bits: usize,
) -> Vec<OutTensor> {
    let packed = cols * bits / 32;
    let groups = cols / GROUP_SIZE;
    let shape = |last: usize| {
        let mut shape = lead.to_vec();
        shape.extend([rows, last]);
        shape
    };
    let count: usize = lead.iter().product::<usize>() * rows;
    vec![
        OutTensor {
            name: format!("{base}.weight"),
            dtype: "U32".to_string(),
            shape: shape(packed),
            nbytes: (count * packed * 4) as u64,
        },
        OutTensor {
            name: format!("{base}.scales"),
            dtype: "BF16".to_string(),
            shape: shape(groups),
            nbytes: (count * groups * 2) as u64,
        },
        OutTensor {
            name: format!("{base}.biases"),
            dtype: "BF16".to_string(),
            shape: shape(groups),
            nbytes: (count * groups * 2) as u64,
        },
    ]
}

fn strip_weight(name: &str) -> Result<&str> {
    name.strip_suffix(".weight")
        .ok_or_else(|| InferError::Config(format!("matrice affine sans suffixe .weight: {name}")))
}

/// Répartit l'inventaire en shards de sortie déterministes.
///
/// # Errors
///
/// Renvoie une erreur si une matrice affine n'a pas de suffixe `.weight`.
pub(crate) fn plan_shards(inventory: &Inventory, layers: usize) -> Result<Vec<ShardPlan>> {
    let trunk_shards = layers.div_ceil(LAYERS_PER_SHARD);
    let total = trunk_shards + 1;
    let file_name = |index: usize| format!("model-{:05}-of-{total:05}.safetensors", index + 1);
    let mut plans: Vec<ShardPlan> = (0..total)
        .map(|index| ShardPlan {
            file: file_name(index),
            jobs: Vec::new(),
            tensors: Vec::new(),
        })
        .collect();
    plans.push(ShardPlan {
        file: "mtp.safetensors".to_string(),
        jobs: Vec::new(),
        tensors: Vec::new(),
    });
    let mtp_index = plans.len() - 1;
    let slot = |placement: Placement| match placement {
        Placement::TrunkLayer(layer) => layer / LAYERS_PER_SHARD,
        Placement::TrunkGlobal => trunk_shards,
        Placement::Mtp => mtp_index,
    };
    for (role, src, out, placement) in &inventory.others {
        let plan = &mut plans[slot(*placement)];
        match role {
            Role::Affine { bits } => {
                let [rows, cols] = src.shape[..] else {
                    return Err(InferError::Dimension(src.name.clone()));
                };
                let base = strip_weight(out)?.to_string();
                plan.tensors
                    .extend(affine_outputs(&base, &[], rows, cols, *bits));
                plan.jobs.push(Job::Affine {
                    src: src.clone(),
                    out_base: base,
                    bits: *bits,
                });
            }
            _ => {
                plan.tensors.push(OutTensor {
                    name: out.clone(),
                    dtype: dtype_name(src.dtype),
                    shape: src.shape.clone(),
                    nbytes: src.len as u64,
                });
                plan.jobs.push(Job::Copy {
                    src: src.clone(),
                    out: out.clone(),
                });
            }
        }
    }
    for ((group, proj), stack) in &inventory.experts {
        let index = match group {
            ExpertGroup::Trunk(layer) => layer / LAYERS_PER_SHARD,
            ExpertGroup::Mtp => mtp_index,
        };
        let [rows, cols] = stack[0].weight.shape[..] else {
            return Err(InferError::Dimension(stack[0].weight.name.clone()));
        };
        let base = group.output_base(*proj);
        let plan = &mut plans[index];
        plan.tensors.extend(affine_outputs(
            &base,
            &[stack.len()],
            rows,
            cols,
            EXPERT_BITS,
        ));
        plan.jobs.push(Job::Experts {
            group: *group,
            proj: *proj,
            out_base: base,
        });
    }
    Ok(plans)
}

/// En-tête safetensors sérialisé (aligné sur 8 octets) et offsets des tenseurs.
pub(crate) fn build_header(plan: &ShardPlan) -> Result<(Vec<u8>, Offsets)> {
    let mut map = serde_json::Map::new();
    let mut offsets = HashMap::new();
    let mut cursor = 0_u64;
    let mut metadata = serde_json::Map::new();
    metadata.insert("format".to_string(), "mlx".into());
    map.insert("__metadata__".to_string(), metadata.into());
    for tensor in &plan.tensors {
        let end = cursor + tensor.nbytes;
        map.insert(
            tensor.name.clone(),
            serde_json::json!({
                "dtype": tensor.dtype,
                "shape": tensor.shape,
                "data_offsets": [cursor, end],
            }),
        );
        if offsets.insert(tensor.name.clone(), (cursor, end)).is_some() {
            return Err(InferError::Config(format!(
                "tenseur de sortie en double: {}",
                tensor.name
            )));
        }
        cursor = end;
    }
    let mut json = serde_json::to_vec(&map).map_err(|source| InferError::Json {
        path: PathBuf::from(&plan.file),
        source,
    })?;
    while json.len() % 8 != 0 {
        json.push(b' ');
    }
    let mut header = (json.len() as u64).to_le_bytes().to_vec();
    header.extend(json);
    Ok((header, offsets))
}

/// Enregistrement d'un tenseur écrit, recopié dans le manifeste.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct TensorRecord {
    pub(crate) name: String,
    pub(crate) dtype: String,
    pub(crate) shape: Vec<usize>,
    pub(crate) data_offsets: [u64; 2],
    pub(crate) sha256: String,
    /// Sources (motif pour les experts empilés).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) bits: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) group_size: Option<usize>,
    /// `‖W−dq(q(W))‖/‖W‖` avec les paramètres bf16 sérialisés.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) rel_err: Option<f64>,
    /// Erreur relative par expert, dans l'ordre 0..E.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) expert_rel_err: Option<Vec<f32>>,
}

/// Flux d'écriture d'un tenseur : `pwrite` à l'offset courant et hachage dans
/// l'ordre des octets.
struct OutStream<'a> {
    file: &'a File,
    path: &'a Path,
    name: String,
    next: u64,
    end: u64,
    sha: Sha256,
}

impl<'a> OutStream<'a> {
    fn new(file: &'a File, path: &'a Path, name: &str, data_start: u64, range: (u64, u64)) -> Self {
        Self {
            file,
            path,
            name: name.to_string(),
            next: data_start + range.0,
            end: data_start + range.1,
            sha: Sha256::new(),
        }
    }

    fn write(&mut self, bytes: &[u8]) -> Result<()> {
        let len = bytes.len() as u64;
        if self.next + len > self.end {
            return Err(InferError::Shape(format!(
                "débordement d'écriture pour {}",
                self.name
            )));
        }
        pwrite_all(self.file, bytes, self.next).map_err(|source| io_err(self.path, source))?;
        self.sha.update(bytes);
        self.next += len;
        Ok(())
    }

    fn finish(self) -> Result<String> {
        if self.next != self.end {
            return Err(InferError::Shape(format!(
                "tenseur {} incomplet: {} octets manquants",
                self.name,
                self.end - self.next
            )));
        }
        Ok(hex(self.sha.finalize()))
    }
}

/// Résultat affine d'un bloc : codes, scales, biais (bits bf16) et erreur.
struct AffineBlock {
    packed: Vec<u32>,
    scales: Vec<u16>,
    biases: Vec<u16>,
    error: QuantError,
}

fn quantize_block(data: &[f32], cols: usize, bits: usize) -> Result<AffineBlock> {
    let rows = data.len() / cols;
    let packed_cols = cols * bits / 32;
    let groups = cols / GROUP_SIZE;
    let mut packed = vec![0_u32; rows * packed_cols];
    let mut scales = vec![0_u16; rows * groups];
    let mut biases = vec![0_u16; rows * groups];
    let error = data
        .par_chunks(ROWS_PER_TASK * cols)
        .zip(packed.par_chunks_mut(ROWS_PER_TASK * packed_cols))
        .zip(scales.par_chunks_mut(ROWS_PER_TASK * groups))
        .zip(biases.par_chunks_mut(ROWS_PER_TASK * groups))
        .map(|(((data, packed), scales), biases)| {
            quantize_affine_bitstream_bf16(data, cols, GROUP_SIZE, bits, packed, scales, biases)
        })
        .try_reduce(QuantError::default, |mut acc, part| {
            acc.add(part);
            Ok(acc)
        })?;
    Ok(AffineBlock {
        packed,
        scales,
        biases,
        error,
    })
}

/// Déquantifie un expert FP8 (poids × échelle par blocs 128×128) en f32.
fn dequantize_expert(source: &Source, expert: &ExpertSrc) -> Result<Vec<f32>> {
    let bytes = source.read(&expert.weight, 0, expert.weight.len)?;
    let mut values = vec![0.0_f32; bytes.len()];
    decode_f8_e4m3_into(&bytes, &mut values, &expert.weight.name)?;
    let scale_bytes = source.read(&expert.scale, 0, expert.scale.len)?;
    let scales = bytes_to_dense_f32(&scale_bytes, expert.scale.dtype, &expert.scale.name)?;
    // Forme exacte `[ceil(r/128), ceil(c/128)]` déjà vérifiée par l'inventaire :
    // une matrice d'un seul bloc a une échelle `[1, 1]`, équivalente au scalaire.
    apply_fp8_block_scales_in_place(
        &mut values,
        &expert.weight.shape,
        &scales,
        &expert.scale.shape,
        &expert.scale.name,
        FP8_SCALE_BLOCK,
    )?;
    Ok(values)
}

/// Suivi du pic de mémoire (empreinte physique du processus), échantillonné
/// aussi depuis les tâches rayon, blocs encore vivants.
#[derive(Default)]
pub(crate) struct PeakTracker {
    peak_bytes: std::sync::atomic::AtomicU64,
}

impl PeakTracker {
    pub(crate) fn sample(&self) {
        if let Some(bytes) = crate::memory_guard::process_footprint_bytes() {
            self.peak_bytes
                .fetch_max(bytes, std::sync::atomic::Ordering::Relaxed);
        }
    }

    pub(crate) fn peak(&self) -> u64 {
        self.peak_bytes.load(std::sync::atomic::Ordering::Relaxed)
    }
}

/// Écrit un shard complet dans `path` (déjà créé et dimensionné par l'appelant).
///
/// # Errors
///
/// Renvoie une erreur de lecture, de décodage FP8, de quantification ou
/// d'écriture.
pub(crate) fn write_shard(
    source: &Source,
    inventory: &Inventory,
    plan: &ShardPlan,
    file: &File,
    path: &Path,
    peak: &PeakTracker,
) -> Result<Vec<TensorRecord>> {
    let (header, offsets) = build_header(plan)?;
    pwrite_all(file, &header, 0).map_err(|source| io_err(path, source))?;
    let data_start = header.len() as u64;
    let shapes: HashMap<&str, &OutTensor> =
        plan.tensors.iter().map(|t| (t.name.as_str(), t)).collect();
    let mut records = Vec::new();
    let stream = |name: &str| -> Result<OutStream<'_>> {
        let range = offsets
            .get(name)
            .copied()
            .ok_or_else(|| InferError::Config(format!("tenseur non planifié: {name}")))?;
        Ok(OutStream::new(file, path, name, data_start, range))
    };
    let record = |name: &str, sha: String| -> Result<TensorRecord> {
        let tensor = shapes
            .get(name)
            .ok_or_else(|| InferError::Config(format!("tenseur non planifié: {name}")))?;
        let range = offsets[name];
        Ok(TensorRecord {
            name: name.to_string(),
            dtype: tensor.dtype.clone(),
            shape: tensor.shape.clone(),
            data_offsets: [range.0, range.1],
            sha256: sha,
            source: None,
            bits: None,
            group_size: None,
            rel_err: None,
            expert_rel_err: None,
        })
    };
    for job in &plan.jobs {
        match job {
            Job::Copy { src, out } => {
                let mut out_stream = stream(out)?;
                let mut start = 0;
                while start < src.len {
                    let len = IO_CHUNK.min(src.len - start);
                    out_stream.write(&source.read(src, start, len)?)?;
                    start += len;
                }
                let mut rec = record(out, out_stream.finish()?)?;
                rec.source = Some(src.name.clone());
                records.push(rec);
            }
            Job::Affine {
                src,
                out_base,
                bits,
            } => {
                let [rows, cols] = src.shape[..] else {
                    return Err(InferError::Dimension(src.name.clone()));
                };
                let names = [
                    format!("{out_base}.weight"),
                    format!("{out_base}.scales"),
                    format!("{out_base}.biases"),
                ];
                let mut weight = stream(&names[0])?;
                let mut scales = stream(&names[1])?;
                let mut biases = stream(&names[2])?;
                let block_rows = (DENSE_BLOCK_ELEMENTS / cols).max(1);
                let mut error = QuantError::default();
                let mut row = 0;
                while row < rows {
                    let take = block_rows.min(rows - row);
                    let bytes = source.read(src, row * cols * 2, take * cols * 2)?;
                    let data = bytes_to_dense_f32(&bytes, src.dtype, &src.name)?;
                    let block = quantize_block(&data, cols, *bits)?;
                    // Source et sortie du bloc encore vivantes : pic réel du bloc.
                    peak.sample();
                    weight.write(bytemuck::cast_slice(&block.packed))?;
                    scales.write(bytemuck::cast_slice(&block.scales))?;
                    biases.write(bytemuck::cast_slice(&block.biases))?;
                    error.add(block.error);
                    row += take;
                    peak.sample();
                }
                let mut rec = record(&names[0], weight.finish()?)?;
                rec.source = Some(src.name.clone());
                rec.bits = Some(*bits);
                rec.group_size = Some(GROUP_SIZE);
                rec.rel_err = Some(error.relative());
                records.push(rec);
                records.push(record(&names[1], scales.finish()?)?);
                records.push(record(&names[2], biases.finish()?)?);
            }
            Job::Experts {
                group,
                proj,
                out_base,
            } => {
                let stack = inventory.experts.get(&(*group, *proj)).ok_or_else(|| {
                    InferError::MissingWeight(format!("pile d'experts {out_base}"))
                })?;
                let cols = stack[0].weight.shape[1];
                let names = [
                    format!("{out_base}.weight"),
                    format!("{out_base}.scales"),
                    format!("{out_base}.biases"),
                ];
                let mut weight = stream(&names[0])?;
                let mut scales = stream(&names[1])?;
                let mut biases = stream(&names[2])?;
                let mut total = QuantError::default();
                let mut per_expert = Vec::with_capacity(stack.len());
                for chunk in stack.chunks(EXPERT_CHUNK) {
                    let blocks: Vec<AffineBlock> = chunk
                        .par_iter()
                        .map(|expert| {
                            let values = dequantize_expert(source, expert)?;
                            let block = quantize_block(&values, cols, EXPERT_BITS)?;
                            peak.sample();
                            Ok(block)
                        })
                        .collect::<Result<_>>()?;
                    for block in blocks {
                        weight.write(bytemuck::cast_slice(&block.packed))?;
                        scales.write(bytemuck::cast_slice(&block.scales))?;
                        biases.write(bytemuck::cast_slice(&block.biases))?;
                        per_expert.push(block.error.relative() as f32);
                        total.add(block.error);
                    }
                    peak.sample();
                }
                let first = &stack[0];
                let mut rec = record(&names[0], weight.finish()?)?;
                rec.source = Some(format!(
                    "{} (×{}, échelles {})",
                    first
                        .weight
                        .name
                        .replacen(".experts.0.", ".experts.{0..}.", 1),
                    stack.len(),
                    first
                        .scale
                        .name
                        .replacen(".experts.0.", ".experts.{0..}.", 1)
                ));
                rec.bits = Some(EXPERT_BITS);
                rec.group_size = Some(GROUP_SIZE);
                rec.rel_err = Some(total.relative());
                rec.expert_rel_err = Some(per_expert);
                records.push(rec);
                records.push(record(&names[1], scales.finish()?)?);
                records.push(record(&names[2], biases.finish()?)?);
            }
        }
    }
    Ok(records)
}

/// Taille totale d'un shard planifié (en-tête compris).
pub(crate) fn shard_size(plan: &ShardPlan) -> Result<u64> {
    let (header, _) = build_header(plan)?;
    Ok(header.len() as u64 + plan.tensors.iter().map(|t| t.nbytes).sum::<u64>())
}

/// sha256 d'un fichier, lu séquentiellement par morceaux bornés.
pub(crate) fn file_sha256(path: &Path) -> Result<String> {
    let file = File::open(path).map_err(|source| io_err(path, source))?;
    let len = file
        .metadata()
        .map_err(|source| io_err(path, source))?
        .len();
    let mut sha = Sha256::new();
    let mut buf = vec![0_u8; IO_CHUNK];
    let mut offset = 0_u64;
    while offset < len {
        let take = (IO_CHUNK as u64).min(len - offset) as usize;
        pread_exact(&file, &mut buf[..take], offset).map_err(|source| io_err(path, source))?;
        sha.update(&buf[..take]);
        offset += take as u64;
    }
    Ok(hex(sha.finalize()))
}
