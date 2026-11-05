//! Règles de conversion `qwen4_exp` : une règle explicite par tenseur source.
//!
//! Aucune règle par sous-chaîne : chaque suffixe de couche est listé, un nom
//! inconnu fait échouer la conversion. Les compteurs attendus (75 264 experts,
//! 525 denses, 570 copies…) sont vérifiés par l'inventaire.

use crate::{InferError, Result};

/// Version des règles, incluse dans l'identité du travail : toute
/// modification de cette table invalide une reprise.
pub(crate) const RULES_VERSION: &str = "qwen4_exp-q4e8-v2";

/// Taille de groupe affine commune à toutes les sorties quantifiées.
pub(crate) const GROUP_SIZE: usize = 64;
/// Largeur des experts routés (tronc et MTP).
pub(crate) const EXPERT_BITS: usize = 4;
/// Largeur des matrices denses.
pub(crate) const DENSE_BITS: usize = 8;
/// Largeur du lm_head.
pub(crate) const LM_HEAD_BITS: usize = 6;

const TRUNK_PREFIX: &str = "model.language_model.";
const MTP_PREFIX: &str = "mtp.";
const VISION_PREFIX: &str = "model.visual.";
/// Préfixe du tronc en sortie : celui que détecte `catalog::detect_weight_prefix`.
pub(crate) const OUT_TRUNK_PREFIX: &str = "language_model.model.";
/// Nom du lm_head en sortie, apparié au préfixe `language_model.model.`.
pub(crate) const OUT_LM_HEAD: &str = "language_model.lm_head.weight";

/// Couche portant le PLE (indice 0-based ; `ple_layer_ids=[2]` est 1-indexé).
pub(crate) const PLE_LAYER: usize = 1;

/// Projection d'un expert routé.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum Proj {
    Gate,
    Up,
    Down,
}

impl Proj {
    pub(crate) const ALL: [Self; 3] = [Self::Gate, Self::Up, Self::Down];

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Gate => "gate_proj",
            Self::Up => "up_proj",
            Self::Down => "down_proj",
        }
    }
}

/// Groupe d'experts empilés dans un même tenseur de sortie.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum ExpertGroup {
    Trunk(usize),
    Mtp,
}

impl ExpertGroup {
    /// Base du tenseur empilé en sortie, sans suffixe (`.weight`/`.scales`/`.biases`).
    pub(crate) fn output_base(self, proj: Proj) -> String {
        match self {
            Self::Trunk(layer) => format!(
                "{OUT_TRUNK_PREFIX}layers.{layer}.mlp.switch_mlp.{}",
                proj.name()
            ),
            Self::Mtp => format!("mtp.layers.0.mlp.switch_mlp.{}", proj.name()),
        }
    }
}

/// Traitement d'un tenseur source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Role {
    /// Poids FP8 d'un expert, empilé en affine u4.
    Expert {
        group: ExpertGroup,
        expert: usize,
        proj: Proj,
    },
    /// `weight_scale_inv` d'un expert, consommée par la déquantification.
    ExpertScale {
        group: ExpertGroup,
        expert: usize,
        proj: Proj,
    },
    /// Matrice dense bf16 quantifiée en affine `bits`, groupes de 64.
    Affine { bits: usize },
    /// Copie octet pour octet (normes, routeurs, gates, conv, petits tenseurs).
    Copy,
    /// Métadonnée entière conservée en I64.
    CopyI64,
    /// Morceau `shard_P` de la table n-gram, recopié dans le fichier mmappable.
    NgramShard(usize),
    /// Échelle per-tensor de la table n-gram (copiée, et reportée dans `meta.json`).
    NgramScale,
    /// Tour vision : exclue explicitement.
    Vision,
}

/// Où vit un tenseur non expert en sortie.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Placement {
    TrunkLayer(usize),
    TrunkGlobal,
    Mtp,
}

/// Types de couche auxquels s'applique une règle de couche.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Applies {
    All,
    Linear,
    Full,
}

#[derive(Clone, Copy)]
enum Kind {
    U8,
    Copy,
    I64,
}

/// Suffixes exacts de `layers.N.` hors experts et n-gram.
const LAYER_RULES: &[(&str, Kind, Applies)] = &[
    (
        "attn_hyper_connection.block_inject_weight.weight",
        Kind::Copy,
        Applies::All,
    ),
    (
        "attn_hyper_connection.hc_norm.weight",
        Kind::Copy,
        Applies::All,
    ),
    (
        "attn_hyper_connection.input_mix_weight_down.weight",
        Kind::U8,
        Applies::All,
    ),
    (
        "attn_hyper_connection.input_mix_weight_up.weight",
        Kind::U8,
        Applies::All,
    ),
    (
        "mlp_hyper_connection.block_inject_weight.weight",
        Kind::Copy,
        Applies::All,
    ),
    (
        "mlp_hyper_connection.hc_norm.weight",
        Kind::Copy,
        Applies::All,
    ),
    (
        "mlp_hyper_connection.input_mix_weight_down.weight",
        Kind::U8,
        Applies::All,
    ),
    (
        "mlp_hyper_connection.input_mix_weight_up.weight",
        Kind::U8,
        Applies::All,
    ),
    ("linear_attn.A_log", Kind::Copy, Applies::Linear),
    ("linear_attn.conv1d.weight", Kind::Copy, Applies::Linear),
    ("linear_attn.dt_bias", Kind::Copy, Applies::Linear),
    ("linear_attn.in_proj_a.weight", Kind::Copy, Applies::Linear),
    ("linear_attn.in_proj_b.weight", Kind::Copy, Applies::Linear),
    ("linear_attn.in_proj_qkv.weight", Kind::U8, Applies::Linear),
    ("linear_attn.in_proj_z.weight", Kind::U8, Applies::Linear),
    ("linear_attn.norm.weight", Kind::Copy, Applies::Linear),
    ("linear_attn.out_proj.weight", Kind::U8, Applies::Linear),
    ("mlp.gate.weight", Kind::Copy, Applies::All),
    ("mlp.shared_expert.down_proj.weight", Kind::U8, Applies::All),
    ("mlp.shared_expert.gate_proj.weight", Kind::U8, Applies::All),
    ("mlp.shared_expert.up_proj.weight", Kind::U8, Applies::All),
    ("mlp.shared_expert_gate.weight", Kind::Copy, Applies::All),
    (
        "self_attn.indexer.index_qk_proj.weight",
        Kind::U8,
        Applies::Full,
    ),
    (
        "self_attn.indexer.k_layernorm.weight",
        Kind::Copy,
        Applies::Full,
    ),
    (
        "self_attn.indexer.q_layernorm.weight",
        Kind::Copy,
        Applies::Full,
    ),
    ("self_attn.k_norm.weight", Kind::Copy, Applies::Full),
    ("self_attn.k_proj.weight", Kind::U8, Applies::Full),
    ("self_attn.o_proj.weight", Kind::U8, Applies::Full),
    ("self_attn.q_norm.weight", Kind::Copy, Applies::Full),
    ("self_attn.q_proj.weight", Kind::U8, Applies::Full),
    ("self_attn.v_proj.weight", Kind::U8, Applies::Full),
];

/// Suffixes exacts du PLE (couche [`PLE_LAYER`] du tronc uniquement).
const PLE_RULES: &[(&str, Kind)] = &[
    ("ple.conv1d.weight", Kind::Copy),
    ("ple.key_proj.weight", Kind::U8),
    ("ple.norm_conv.weight", Kind::Copy),
    ("ple.norm_key.weight", Kind::Copy),
    ("ple.norm_query.weight", Kind::Copy),
    ("ple.value_proj.weight", Kind::U8),
    ("ple.ple_embedding.layer_multipliers", Kind::I64),
    ("ple.ple_embedding.ngram_heads_offsets", Kind::I64),
    ("ple.ple_embedding.ngram_heads_vocab_sizes", Kind::I64),
];

/// Tenseurs globaux d'un mixer d'hyper-connections (tronc et MTP).
const MIXER_RULES: &[(&str, Kind)] = &[
    ("hyper_connection_mixer.hc_norm.weight", Kind::Copy),
    (
        "hyper_connection_mixer.input_mix_weight_down.weight",
        Kind::U8,
    ),
    (
        "hyper_connection_mixer.input_mix_weight_up.weight",
        Kind::U8,
    ),
];

/// Tenseurs globaux propres au MTP.
const MTP_GLOBAL_RULES: &[(&str, Kind)] = &[
    ("fc_embedding.weight", Kind::U8),
    ("fc_hidden.weight", Kind::U8),
    ("pre_fc_norm_embedding.weight", Kind::Copy),
    ("pre_fc_norm_hidden.weight", Kind::Copy),
];

const NGRAM_SHARD_PREFIX: &str = "ple.ple_embedding.ngram_embedding.shard_";
const NGRAM_SCALE: &str = "ple.ple_embedding.ngram_embedding.weight_scale";

fn kind_role(kind: Kind) -> Role {
    match kind {
        Kind::U8 => Role::Affine { bits: DENSE_BITS },
        Kind::Copy => Role::Copy,
        Kind::I64 => Role::CopyI64,
    }
}

fn lookup(rules: &[(&str, Kind)], suffix: &str) -> Option<Role> {
    rules
        .iter()
        .find(|(name, _)| *name == suffix)
        .map(|(_, kind)| kind_role(*kind))
}

fn unknown(name: &str) -> InferError {
    InferError::Config(format!("conversion qwen4_exp: tenseur sans règle: {name}"))
}

/// Indice décimal strict (pas de signe, pas de zéro de tête hors « 0 »).
fn parse_index(text: &str) -> Option<usize> {
    if text.is_empty() || (text.len() > 1 && text.starts_with('0')) {
        return None;
    }
    if !text.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    text.parse().ok()
}

/// Sépare `layers.N.reste` en `(N, reste)`.
fn split_layer(rest: &str) -> Option<(usize, &str)> {
    let rest = rest.strip_prefix("layers.")?;
    let (index, tail) = rest.split_once('.')?;
    Some((parse_index(index)?, tail))
}

/// `mlp.experts.E.{gate,up,down}_proj.{weight,weight_scale_inv}` → `(E, proj, is_scale)`.
fn parse_expert(tail: &str) -> Option<(usize, Proj, bool)> {
    let rest = tail.strip_prefix("mlp.experts.")?;
    let (index, rest) = rest.split_once('.')?;
    let expert = parse_index(index)?;
    let (proj_name, suffix) = rest.split_once('.')?;
    let proj = Proj::ALL
        .into_iter()
        .find(|proj| proj.name() == proj_name)?;
    match suffix {
        "weight" => Some((expert, proj, false)),
        "weight_scale_inv" => Some((expert, proj, true)),
        _ => None,
    }
}

fn layer_role(group: ExpertGroup, layer: usize, tail: &str, name: &str) -> Result<Role> {
    if let Some((expert, proj, is_scale)) = parse_expert(tail) {
        return Ok(if is_scale {
            Role::ExpertScale {
                group,
                expert,
                proj,
            }
        } else {
            Role::Expert {
                group,
                expert,
                proj,
            }
        });
    }
    if let Some((_, kind, _)) = LAYER_RULES.iter().find(|(suffix, _, _)| *suffix == tail) {
        return Ok(kind_role(*kind));
    }
    let is_ple_layer = group == ExpertGroup::Trunk(PLE_LAYER) && layer == PLE_LAYER;
    if is_ple_layer {
        if let Some(role) = lookup(PLE_RULES, tail) {
            return Ok(role);
        }
        if tail == NGRAM_SCALE {
            return Ok(Role::NgramScale);
        }
        if let Some(index) = tail
            .strip_prefix(NGRAM_SHARD_PREFIX)
            .and_then(|rest| rest.strip_suffix(".weight"))
            .and_then(parse_index)
        {
            return Ok(Role::NgramShard(index));
        }
    }
    Err(unknown(name))
}

/// Classe un tenseur source.
///
/// # Errors
///
/// Renvoie une erreur pour tout nom sans règle explicite.
pub(crate) fn classify(name: &str) -> Result<Role> {
    if name.starts_with(VISION_PREFIX) {
        return Ok(Role::Vision);
    }
    if name == "lm_head.weight" {
        return Ok(Role::Affine { bits: LM_HEAD_BITS });
    }
    if let Some(rest) = name.strip_prefix(TRUNK_PREFIX) {
        if rest == "embed_tokens.weight" {
            return Ok(Role::Affine { bits: DENSE_BITS });
        }
        if let Some(role) = lookup(MIXER_RULES, rest) {
            return Ok(role);
        }
        let (layer, tail) = split_layer(rest).ok_or_else(|| unknown(name))?;
        return layer_role(ExpertGroup::Trunk(layer), layer, tail, name);
    }
    if let Some(rest) = name.strip_prefix(MTP_PREFIX) {
        if let Some(role) = lookup(MIXER_RULES, rest).or_else(|| lookup(MTP_GLOBAL_RULES, rest)) {
            return Ok(role);
        }
        let (layer, tail) = split_layer(rest).ok_or_else(|| unknown(name))?;
        if layer != 0 {
            return Err(unknown(name));
        }
        return layer_role(ExpertGroup::Mtp, layer, tail, name);
    }
    Err(unknown(name))
}

/// Nom de sortie et placement d'un tenseur non expert, non n-gram.
///
/// # Errors
///
/// Renvoie une erreur si le nom ne suit aucun préfixe connu.
pub(crate) fn output_name(name: &str) -> Result<(String, Placement)> {
    if name == "lm_head.weight" {
        return Ok((OUT_LM_HEAD.to_string(), Placement::TrunkGlobal));
    }
    if let Some(rest) = name.strip_prefix(TRUNK_PREFIX) {
        let placement = match split_layer(rest) {
            Some((layer, _)) => Placement::TrunkLayer(layer),
            None => Placement::TrunkGlobal,
        };
        return Ok((format!("{OUT_TRUNK_PREFIX}{rest}"), placement));
    }
    if name.starts_with(MTP_PREFIX) {
        return Ok((name.to_string(), Placement::Mtp));
    }
    Err(unknown(name))
}

/// Type d'une couche, lu dans `layer_types`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LayerType {
    Linear,
    Full,
}

impl LayerType {
    /// # Errors
    ///
    /// Renvoie une erreur pour un type de couche inconnu.
    pub(crate) fn parse(text: &str) -> Result<Self> {
        match text {
            "linear_attention" => Ok(Self::Linear),
            "full_attention" | "indexed_attention" => Ok(Self::Full),
            other => Err(InferError::Config(format!(
                "type de couche inconnu: {other}"
            ))),
        }
    }
}

fn layer_names(prefix: &str, layer: usize, kind: LayerType, out: &mut Vec<String>) {
    for (suffix, _, applies) in LAYER_RULES {
        let keep = match applies {
            Applies::All => true,
            Applies::Linear => kind == LayerType::Linear,
            Applies::Full => kind == LayerType::Full,
        };
        if keep {
            out.push(format!("{prefix}layers.{layer}.{suffix}"));
        }
    }
}

/// Noms source attendus hors experts, shards n-gram et vision : la partition
/// doit les contenir tous, et rien d'autre.
pub(crate) fn expected_source_names(trunk: &[LayerType], mtp: Option<LayerType>) -> Vec<String> {
    let mut names = vec![
        "lm_head.weight".to_string(),
        format!("{TRUNK_PREFIX}embed_tokens.weight"),
    ];
    for (suffix, _) in MIXER_RULES {
        names.push(format!("{TRUNK_PREFIX}{suffix}"));
    }
    for (layer, kind) in trunk.iter().enumerate() {
        layer_names(TRUNK_PREFIX, layer, *kind, &mut names);
        if layer == PLE_LAYER {
            for (suffix, _) in PLE_RULES {
                names.push(format!("{TRUNK_PREFIX}layers.{layer}.{suffix}"));
            }
            names.push(format!("{TRUNK_PREFIX}layers.{layer}.{NGRAM_SCALE}"));
        }
    }
    if let Some(kind) = mtp {
        for (suffix, _) in MIXER_RULES.iter().chain(MTP_GLOBAL_RULES) {
            names.push(format!("{MTP_PREFIX}{suffix}"));
        }
        layer_names(MTP_PREFIX, 0, kind, &mut names);
    }
    names.sort();
    names
}
